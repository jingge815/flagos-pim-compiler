"""测量算子成本，并将拟合系数和元数据写入 GeneSim 文件。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .flagtree_driver import run_and_capture
from .ir_cost import analyze_ir
from .op_classify import (
    UNCOVERED_OP_TYPES,
    ShapePoint,
    build_recipes,
    flash_attention_probe,
    gemm_features,
)
from .paths import pim_options

SIDECAR_VERSION = "1.1"

# 融合注意力交叉验证使用的 WRAM 预算。
_PROBE_WRAM_BYTES = 256 * 1024


@dataclass
class Measurement:
    """一个算子在一个 shape 点上的测量结果。"""
    point: ShapePoint
    flops: float
    data_bytes: float
    dtype: str
    element_bytes: int
    kernel_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # 以下字段仅由 PIM IR 填充。
    mram_traffic_bytes: Optional[float] = None
    pim_kernels: List[Dict[str, Any]] = field(default_factory=list)
    # 按本地分片形状测量时的 (in_features, out_features)，全局口径时为 None。
    local_features: Optional[Tuple[int, int]] = None


def _measure(
    op_dict: dict,
    point: ShapePoint,
    dims: Dict[str, int],
    ir_level: str,
    local_features: Optional[Tuple[int, int]] = None,
) -> Measurement:
    """编译一个形状点的算子，并测量其成本。

    `local_features` 给出该算子在一台 DPU 上的 (in_features, out_features)，
    来自图编译器的切分结果。传了就按本地分片形状编译和测量，不传则沿用 IR 里
    的模型级全局形状。
    """
    op_type = op_dict["op_type"]
    recipes = build_recipes(dims)

    if op_type == "GEMM":
        if local_features is not None:
            in_features, out_features = local_features
        else:
            in_features, out_features = gemm_features(op_dict)
        recipe = recipes["GEMM"](point, in_features, out_features)
    else:
        recipe = recipes[op_type](point)

    pimir = ir_level == "pimir"
    captured = run_and_capture(recipe.build, emit_pimir=pimir)
    if not captured:
        raise RuntimeError(
            f"op_type={op_type} 在 {point.label} 未捕获到任何 kernel；"
            "可能 FlagGems 未接管该算子"
        )

    total_flops = 0.0
    total_mram = 0.0
    dtype = "f16"
    element_bytes = 2
    kernel_ids: List[str] = []
    notes: List[str] = []
    pim_kernels: List[Dict[str, Any]] = []
    for cap in captured:
        text = cap.pimir if pimir else cap.ttir
        cost = analyze_ir(text, cap.name, cap.grid, cap.arg_values, ir_level=ir_level)
        total_flops += cost.flops
        dtype = cost.dtype
        element_bytes = cost.element_bytes
        kernel_ids.append(f"{cost.kernel_name}@grid{cost.grid}")
        notes.extend(f"{cost.kernel_name}: {n}" for n in cost.notes)
        if pimir:
            total_mram += cost.mram_traffic_bytes
            pim_kernels.append(_pim_kernel_dict(cost))

    return Measurement(
        point=point,
        flops=total_flops,
        data_bytes=_net_data_bytes(op_dict, point, element_bytes, local_features),
        dtype=dtype,
        element_bytes=element_bytes,
        kernel_ids=kernel_ids,
        notes=notes,
        mram_traffic_bytes=total_mram if pimir else None,
        pim_kernels=pim_kernels,
        local_features=local_features,
    )


def _pim_kernel_dict(cost) -> Dict[str, Any]:
    """将 PIM 内核的搬运、WRAM 和分块信息转为 sidecar 字典。"""
    return {
        "kernel": cost.kernel_name,
        "grid": list(cost.grid),
        "mram_traffic_bytes": cost.mram_traffic_bytes,
        "wram_bytes_used": cost.wram_bytes_used,
        "wram_bytes_budget": cost.wram_bytes_budget,
        "wram_buffers": [
            {"shape": b.shape, "dtype": b.dtype, "bytes": b.bytes}
            for b in cost.wram_buffers
        ],
        "dma_ops": cost.dma_ops,
        # 记录带已确认布局属性的 DMA 数量。
        "dma_ops_with_proven_layout": cost.dma_ops_with_layout,
        "loop_trip_counts": cost.loop_trip_counts,
        # 记录硬件预算和自动选择的分块。
        "mram_bytes_budget": cost.mram_bytes_budget,
        "dma_align": cost.dma_align,
        "tile_m": cost.tile_m,
        "tile_n": cost.tile_n,
        "tile_k": cost.tile_k,
        "tile_wram_bytes": cost.tile_wram_bytes,
    }


def _resolve_dim(dim: Any, point: ShapePoint) -> int:
    """把 GeneSim 的符号维度代入具体值。"""
    if isinstance(dim, int):
        return dim
    table = {"Tq": point.tq, "Tp": point.tp, "Tp+Tq": point.lkv}
    if dim not in table:
        raise ValueError(f"未知符号维度: {dim}")
    return table[dim]


def _net_data_bytes(
    op_dict: dict,
    point: ShapePoint,
    element_bytes: int,
    local_features: Optional[Tuple[int, int]] = None,
) -> float:
    """返回输入、权重和输出的净读写字节数。

    传了 `local_features` 时，激活和权重都按本地分片宽度算：这台 DPU 只读自己
    那一份权重，产出的也只是切分后的那段激活。
    """
    if local_features is not None and op_dict["op_type"] == "GEMM":
        in_features, out_features = local_features
        # GEMM 的形状是 [Tq, in] -> [Tq, out]，Tq 由 shape point 决定。
        tq = _resolve_dim("Tq", point)
        total = tq * in_features + tq * out_features + in_features * out_features
        return float(total * element_bytes)

    total = 0
    for shape in list(op_dict["input_shapes"]) + list(op_dict["output_shapes"]):
        numel = 1
        for dim in shape:
            numel *= _resolve_dim(dim, point)
        total += numel

    if op_dict["op_type"] == "GEMM":
        in_features, out_features = gemm_features(op_dict)
        total += in_features * out_features  # 权重大小与 Tq 无关。

    return float(total * element_bytes)


def _fit_coeffs(
    op_dict: dict,
    prefill: Measurement,
    decode: Measurement,
    seq_len: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """从 prefill 和 decode 两个测量点拟合成本系数。"""
    op_type = op_dict["op_type"]

    if op_type == "GEMM":
        term_of = lambda p: p.tq
        term_key = "Tq"
    else:
        term_of = lambda p: p.tq * p.lkv
        term_key = "Tq(Tp+Tq)"

    t1, t2 = term_of(prefill.point), term_of(decode.point)
    if t1 == t2:
        raise ValueError(f"两个代表点的成本项相同 ({t1})，无法拟合 {op_type}")

    def solve(y1: float, y2: float) -> Tuple[float, float, str]:
        """拟合非负斜率和截距，必要时使用经过较大测量点的原点直线。"""
        a = (y1 - y2) / (t1 - t2)
        b = y1 - a * t1
        if a >= 0.0 and b >= 0.0:
            return a, b, "two_point_linear"
        big_t, big_y = (t1, y1) if t1 >= t2 else (t2, y2)
        return big_y / big_t, 0.0, "origin_through_large_term"

    a_f, b_f, mode_f = solve(prefill.flops, decode.flops)
    a_b, b_b, mode_b = solve(prefill.data_bytes, decode.data_bytes)

    flops_coeffs: Dict[str, float] = {term_key: a_f}
    if b_f > 1e-9:
        flops_coeffs["constant"] = b_f

    data_coeffs: Dict[str, float] = {term_key: a_b}
    if b_b > 1e-9:
        data_coeffs["constant"] = b_b

    return flops_coeffs, data_coeffs, {"flops": mode_f, "data_bytes": mode_b}


def load_local_shapes(placement_sidecar_path: Path) -> Dict[int, Tuple[int, int]]:
    """从放置结果 sidecar 读出每个算子的本地 (in_features, out_features)。

    产出者是 `placement_export.export_placement_to_genesim`。版本 2 之前的
    sidecar 没有这两个字段，此时返回空字典，调用方据此退回全局口径。

    只读不校验：op_id 是否与目标 IR 对得上由
    `validate_local_shapes_against_ir` 负责，`export_costs_to_genesim` 会调它。
    """
    sidecar = json.loads(Path(placement_sidecar_path).read_text())
    local: Dict[int, Tuple[int, int]] = {}
    for op_id, entry in sidecar.get("operators", {}).items():
        in_f = entry.get("local_in_features")
        out_f = entry.get("local_out_features")
        if in_f is None or out_f is None:
            continue
        local[int(op_id)] = (int(in_f), int(out_f))
    return local


def validate_local_shapes_against_ir(
    local_shapes: Dict[int, Tuple[int, int]], ir: Dict[str, Any]
) -> None:
    """核对本地形状表的 op_id 与目标 IR 是同一批编号。

    sidecar 里的 op_id 属于导出时那份 IR。若之后重新生成过 IR 让编号变化，或者
    传错了另一个模型的 sidecar，就会把某个算子的本地宽度套到别的算子上——不报错，
    只是成本悄悄不对。GeneSim 侧读放置结果时有同样的校验，这条成本精化路径也要有，
    否则错位会一路走到底。
    """
    operators = {op["op_id"]: op for op in ir["operators"]}

    missing = sorted(op_id for op_id in local_shapes if op_id not in operators)
    if missing:
        raise ValueError(
            f"放置结果引用了当前 IR 里不存在的 op_id：{missing[:8]}"
            f"{' ...' if len(missing) > 8 else ''}（共 {len(missing)} 个）。"
            "op_id 编号已经错位，请用当前 IR 重新导出放置结果。"
        )

    # 本地形状只对 GEMM 有意义（`_measure` 也只在 GEMM 上用它）。落到别的算子类型上
    # 说明编号错位了——即使算子数恰好相同也能被这一条抓出来。
    wrong_type = sorted(
        (op_id, operators[op_id]["op_type"])
        for op_id in local_shapes
        if operators[op_id]["op_type"] != "GEMM"
    )
    if wrong_type:
        head = ", ".join(f"op{op_id}={op_type}" for op_id, op_type in wrong_type[:4])
        raise ValueError(
            f"放置结果里有 {len(wrong_type)} 个 op_id 在当前 IR 里不是 GEMM：{head}。"
            "op_id 编号已经错位，请用当前 IR 重新导出放置结果。"
        )


def export_costs_to_genesim(
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
    seq_len: int,
    cross_validate: bool = True,
    ir_level: str = "pimir",
    local_shapes: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """更新 GeneSim 成本系数，并写入 IR 文件和 sidecar 文件。

    `local_shapes` 给出 {op_id: (in_features, out_features)}，来自图编译器的
    切分结果（见 `load_local_shapes`）。命中的算子按本地分片形状编译和测量，
    未命中的沿用 IR 里的模型级全局形状。不传则全部按全局形状，与改造前一致。
    """
    assert ir_level in ("ttir", "pimir"), f"未知 ir_level: {ir_level}"
    ir = json.loads(Path(ir_path).read_text())
    local_shapes = local_shapes or {}
    if local_shapes:
        validate_local_shapes_against_ir(local_shapes, ir)

    dims = {
        "hidden_size": ir["hidden_size"],
        "head_dim": ir["head_dim"],
        "num_heads": ir["num_heads"],
        "ffn_dim": 4 * ir["hidden_size"],
    }
    prefill_point = ShapePoint(tq=seq_len, tp=0)
    decode_point = ShapePoint(tq=1, tp=seq_len)

    # 缓存相同算子类型和形状的测量结果。
    cache: Dict[tuple, Tuple[Measurement, Measurement]] = {}
    sidecar: Dict[str, Any] = {
        "version": SIDECAR_VERSION,
        "ir_level": ir_level,
        "source_ir": str(ir_path),
        "shape_points": {
            "prefill": {"Tq": prefill_point.tq, "Tp": prefill_point.tp},
            "decode": {"Tq": decode_point.tq, "Tp": decode_point.tp},
        },
        "coverage": {"bridged": [], "template": []},
        "operators": {},
    }
    if ir_level == "pimir":
        # 保存 sidecar 对应的 PIM pass 参数。
        sidecar["pim_options"] = pim_options()

    for op in ir["operators"]:
        op_type = op["op_type"]
        op_id = op["op_id"]

        if op_type in UNCOVERED_OP_TYPES:
            sidecar["coverage"]["template"].append(op_id)
            continue

        # 本地分片形状进缓存键：同一个全局形状在不同流水段/切分宽度下可能对应
        # 不同的本地形状，漏掉它会让后面的算子错误复用前面的测量结果。
        local_features = local_shapes.get(op_id)
        key = (op_type,
               tuple(tuple(s) for s in op["input_shapes"]),
               tuple(tuple(s) for s in op["output_shapes"]),
               local_features)
        if key not in cache:
            cache[key] = (
                _measure(op, prefill_point, dims, ir_level, local_features),
                _measure(op, decode_point, dims, ir_level, local_features),
            )
        prefill, decode = cache[key]

        template_flops = dict(op.get("flops_coeffs", {}))
        template_bytes = dict(op.get("data_bytes_coeffs", {}))

        flops_coeffs, data_coeffs, fit_mode = _fit_coeffs(op, prefill, decode, seq_len)
        op["flops_coeffs"] = flops_coeffs
        op["data_bytes_coeffs"] = data_coeffs
        # 系数为空时使用零值标量。
        op["flops"] = 0.0
        op["data_bytes"] = 0.0
        op["arith_intensity"] = 0.0

        sidecar["coverage"]["bridged"].append(op_id)
        sidecar["operators"][str(op_id)] = {
            "op_type": op_type,
            "source_name": _source_name(op, dims, prefill_point),
            "dtype": prefill.dtype,
            "kernel_ids": sorted(set(prefill.kernel_ids + decode.kernel_ids)),
            "measurements": {
                "prefill": _measurement_dict(prefill),
                "decode": _measurement_dict(decode),
            },
            "fitted_coeffs": {
                "flops_coeffs": flops_coeffs,
                "data_bytes_coeffs": data_coeffs,
            },
            "fit_mode": fit_mode,
            "template_coeffs": {
                "flops_coeffs": template_flops,
                "data_bytes_coeffs": template_bytes,
            },
            # 记录分块级 MRAM 与 WRAM 搬运字节数。
            "mram_traffic_bytes": (
                None if ir_level == "ttir"
                else {
                    "prefill": prefill.mram_traffic_bytes,
                    "decode": decode.mram_traffic_bytes,
                }
            ),
            # 按本地分片形状测量时记下用的宽度；为 None 表示这个算子仍是
            # 模型级全局口径（没有放置结果，或非 GEMM）。
            "local_features": (
                None if prefill.local_features is None
                else {
                    "in_features": prefill.local_features[0],
                    "out_features": prefill.local_features[1],
                }
            ),
        }

    if cross_validate:
        sidecar["cross_validation"] = _cross_validate(
            dims, prefill_point, decode_point, ir_level
        )

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar


def _source_name(op: dict, dims: Dict[str, int], point: ShapePoint) -> str:
    op_type = op["op_type"]
    recipes = build_recipes(dims)
    if op_type == "GEMM":
        in_f, out_f = gemm_features(op)
        return recipes["GEMM"](point, in_f, out_f).source_name
    return recipes[op_type](point).source_name


def _measurement_dict(m: Measurement) -> Dict[str, Any]:
    entry = {
        "Tq": m.point.tq,
        "Tp": m.point.tp,
        "flops": m.flops,
        "data_bytes": m.data_bytes,
        "dtype": m.dtype,
        "kernel_ids": m.kernel_ids,
        "notes": m.notes,
    }
    if m.mram_traffic_bytes is not None:
        entry["mram_traffic_bytes"] = m.mram_traffic_bytes
        # 计算 MRAM 搬运相对网络数据量的放大倍数。
        entry["mram_amplification"] = (
            m.mram_traffic_bytes / m.data_bytes if m.data_bytes else None
        )
        entry["pim_kernels"] = m.pim_kernels
    return entry


def _cross_validate(
    dims: Dict[str, int],
    prefill_point: ShapePoint,
    decode_point: ShapePoint,
    ir_level: str,
) -> Dict[str, Any]:
    """测量融合注意力内核，作为分离算子成本的交叉验证基准。"""
    result: Dict[str, Any] = {
        "note": "FlagGems 实跑 attention 为融合 flash 路径；"
                "此处记录融合 kernel 的单层全 head 成本，供与代表实现求和对照",
    }
    pimir = ir_level == "pimir"
    if pimir:
        result["probe_relaxations"] = {
            "wram_bytes": _PROBE_WRAM_BYTES,
            "tile_to_budget": False,
            "why": "融合 flash kernel 有 4 个 tt.dot 且 WRAM staging 131584 B "
                   "超过真实预算 65536 B；放宽仅用于取 flops/MRAM 量级对照，"
                   "不影响真实算子的测量",
        }
    for name, point in (("prefill", prefill_point), ("decode", decode_point)):
        try:
            captured = run_and_capture(
                flash_attention_probe(dims, point), emit_pimir=pimir,
                tile_to_budget=False,
                wram_bytes=_PROBE_WRAM_BYTES if pimir else None,
            )
        except Exception as exc:                     # noqa: BLE001
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        entries = []
        total = 0.0
        total_mram = 0.0
        for cap in captured:
            text = cap.pimir if pimir else cap.ttir
            cost = analyze_ir(text, cap.name, cap.grid, cap.arg_values, ir_level=ir_level)
            total += cost.flops
            entry = {
                "kernel": cost.kernel_name,
                "grid": list(cost.grid),
                "flops": cost.flops,
                "loop_trip_counts": cost.loop_trip_counts,
                "notes": cost.notes,
            }
            if pimir:
                total_mram += cost.mram_traffic_bytes
                entry["mram_traffic_bytes"] = cost.mram_traffic_bytes
                entry["wram_bytes_used"] = cost.wram_bytes_used
            entries.append(entry)
        result[name] = {"total_flops": total, "kernels": entries}
        if pimir:
            result[name]["total_mram_traffic_bytes"] = total_mram
    return result
