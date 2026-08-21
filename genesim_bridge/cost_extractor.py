"""成本提取器主入口：把编译器测得的成本回填进 GeneSim .ir。

方案依据：spec.md 问题 4 三.(2)/(4)，按实测修正两处：

1. **回填 coeffs，不是标量。** GeneSim 的 Operator 成本现在是符号系数
   （`flops_coeffs` / `data_bytes_coeffs`，key ∈ {constant, Tq, Tp, Tp+Tq,
   Tq(Tp+Tq)}），scheduler 每个请求代入 Tq/Tp 求值（gene_sim_scheduler.py
   `_get_operator_with_runtime`）。标量 `flops` 只在 coeffs 为空时作 fallback。
   方案原文的 `op.flops = cost.flops` 会被 coeffs 直接盖掉，等于没生效。

2. **两点线性拟合。** 系数无法由单点编译得出。在 prefill (Tq=seq_len, Tp=0)
   与 decode (Tq=1, Tp=seq_len) 两个代表点各编一次，按算子的成本项形状
   解出系数。真实成本对 Tq 是带 padding 台阶的阶跃函数（实测 fc1 在 Tq=1
   时因 BLOCK_M=64 白算 64 倍），线性拟合会抹平台阶，中间区段有误差——
   故每个点的原始测量值完整落 sidecar，不丢信息。

改 .ir 走原始 JSON 而不是 ModelIR.load()/save()：`ModelIR.to_dict()` 不含
部分 GeneSim IR 携带的扩展字段，往返一趟会静默丢字段。
"""

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
    # 以下仅 ir_level="pimir" 时有值
    mram_traffic_bytes: Optional[float] = None
    pim_kernels: List[Dict[str, Any]] = field(default_factory=list)


def _measure(
    op_dict: dict,
    point: ShapePoint,
    dims: Dict[str, int],
    ir_level: str,
) -> Measurement:
    """在一个 shape 点上编译该算子并抽成本。

    ir_level="pimir" 时 driver 额外把捕获到的 TTIR 就地降成 pim mlir，成本从
    pim mlir 上抽。flops 两层必然相等（PIM pass 不改 tt.dot / arith / scf.for），
    差别是 pimir 多出 mram_traffic_bytes 与 WRAM 用量。
    """
    op_type = op_dict["op_type"]
    recipes = build_recipes(dims)

    if op_type == "GEMM":
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
        data_bytes=_net_data_bytes(op_dict, point, element_bytes),
        dtype=dtype,
        element_bytes=element_bytes,
        kernel_ids=kernel_ids,
        notes=notes,
        mram_traffic_bytes=total_mram if pimir else None,
        pim_kernels=pim_kernels,
    )


def _pim_kernel_dict(cost) -> Dict[str, Any]:
    """pim mlir 特有的 kernel 级元数据，落 sidecar。

    这些量 GeneSim 的 Operator schema 放不下，且按方案三.(3) 也不该进
    `data_bytes`；但它们是「换成 pim mlir 之后新拿到的信息」的全部内容，
    是这一步的实际产出，故完整记录。
    """
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
        # 指针分析证明了 stride 的 DMA 数。未证明不等于不连续，只是当前没证明
        # （doc 9.5），比值低说明 PIM 后端能利用的 DMA 信息有限
        "dma_ops_with_proven_layout": cost.dma_ops_with_layout,
        "loop_trip_counts": cost.loop_trip_counts,
    }


def _resolve_dim(dim: Any, point: ShapePoint) -> int:
    """把 GeneSim 的符号维度代入具体值。"""
    if isinstance(dim, int):
        return dim
    table = {"Tq": point.tq, "Tp": point.tp, "Tp+Tq": point.lkv}
    if dim not in table:
        raise ValueError(f"未知符号维度: {dim}")
    return table[dim]


def _net_data_bytes(op_dict: dict, point: ShapePoint, element_bytes: int) -> float:
    """算子对外净读写字节数：Σ输入 + 权重 + Σ输出，按 IR 真实 dtype。

    方案三.(3)：GeneSim 下游对 data_bytes 有两处消费——roofline 的 memory
    时间、以及跨 VPU 传输字节估算。两者要的都是「对外净流量」，不是
    「设备内部搬了多少次」。故这里只按张量 shape 算，不统计 tile 内部的
    重复搬运（那个属 mram_traffic_bytes，第 2 步接 pim mlir 时才有值）。

    GEMM 要额外补权重字节：GeneSim 的 GEMM `input_shapes` 只有激活
    （形如 `[["Tq", 512]]`），权重张量不在 shape 列表里——模板是把它算进
    `data_bytes_coeffs["constant"]` 的（fc1 的 constant=2097152 正是
    512*2048*2）。只按 input/output_shapes 求和会漏掉权重读取，而权重恰好
    主导 decode 阶段的访存量（Tq=1 时激活只有几 KB，权重是 MB 量级），
    漏掉会把 decode 的 data_bytes 低估到 1/250。

    attention 类算子（GEMV_SCORE / GEMV_CONTEXT）的两个操作数都在
    input_shapes 里，无需补。
    """
    total = 0
    for shape in list(op_dict["input_shapes"]) + list(op_dict["output_shapes"]):
        numel = 1
        for dim in shape:
            numel *= _resolve_dim(dim, point)
        total += numel

    if op_dict["op_type"] == "GEMM":
        in_features, out_features = gemm_features(op_dict)
        total += in_features * out_features        # 权重，与 Tq 无关

    return float(total * element_bytes)


def _fit_coeffs(
    op_dict: dict,
    prefill: Measurement,
    decode: Measurement,
    seq_len: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """两点线性拟合出 flops_coeffs / data_bytes_coeffs。

    按算子的成本项形状分两类（沿用 GeneSim 模板的项集合，不新增 key）：

    - GEMM 类：成本 ∝ Tq，模型 `cost = a*Tq + b`
        prefill (Tq=seq_len) 与 decode (Tq=1) 两点解 a、b。
        b 归到 constant（权重读取那部分与 Tq 无关）。
    - attention 类：成本 ∝ Tq*(Tp+Tq)，模型 `cost = a*Tq(Tp+Tq) + b`
        两点的 Tq(Tp+Tq) 都等于 seq_len（prefill: seq_len*seq_len 除以...
        见下），故直接用两点解。
    """
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
        """解 cost = a*term + b，返回 (a, b, fit_mode)。

        系数必须非负：GeneSim 的 scheduler 直接把求值结果当 flops /
        data_bytes 用，负值会让 roofline 退化成只收 kernel launch 开销、
        并把 arith_intensity 变成负数从而污染 PIM/GPU 分区判据。

        无约束两点解出负斜率时，说明两点不在同一个 padding 区间：小 term
        的那点（decode，Tq=1）被 tile padding 主导，实测 flops 反而更大
        （实测 GEMV_SCORE 的 decode 点 padding 开销达 159 倍）。此时改用
        「过原点 + 大 term 点」定斜率——大 term 点 padding 可忽略，斜率
        等于真实渐进标度；padding 主导那点的原始测量值仍完整落 sidecar。
        """
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


def export_costs_to_genesim(
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
    seq_len: int,
    cross_validate: bool = True,
    ir_level: str = "pimir",
) -> Dict[str, Any]:
    """主入口：读 GeneSim .ir，回填编译器测得的成本，落盘 .ir + sidecar。

    不改算子个数、类型、连接与设备归属——只换成本数值，以及把 GeneSim
    schema 放不下的元数据落 sidecar。

    ir_level="ttir"  -- 方案三.(4) 第 1 步，成本从 FlagTree 原生 TTIR 抽。
    ir_level="pimir" -- 第 2 步（默认），成本从 pim mlir 抽，额外产出
                       mram_traffic_bytes 与 WRAM 用量。回填进 .ir 的
                       flops/data_bytes 口径与第 1 步完全相同（方案三.(3)：
                       tile 级重复搬运不进 data_bytes）。
    """
    assert ir_level in ("ttir", "pimir"), f"未知 ir_level: {ir_level}"
    ir = json.loads(Path(ir_path).read_text())

    dims = {
        "hidden_size": ir["hidden_size"],
        "head_dim": ir["head_dim"],
        "num_heads": ir["num_heads"],
        "ffn_dim": 4 * ir["hidden_size"],
    }
    prefill_point = ShapePoint(tq=seq_len, tp=0)
    decode_point = ShapePoint(tq=1, tp=seq_len)

    # 同一 (op_type, shape) 只测一次：整模型里大量算子共享相同 op_type/shape。
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
        # 生效的 PIM pass 参数：换硬件配置后要能溯源到这份 sidecar 是哪组参数编的
        sidecar["pim_options"] = pim_options()

    for op in ir["operators"]:
        op_type = op["op_type"]
        op_id = op["op_id"]

        if op_type in UNCOVERED_OP_TYPES:
            sidecar["coverage"]["template"].append(op_id)
            continue

        key = (op_type,
               tuple(tuple(s) for s in op["input_shapes"]),
               tuple(tuple(s) for s in op["output_shapes"]))
        if key not in cache:
            cache[key] = (
                _measure(op, prefill_point, dims, ir_level),
                _measure(op, decode_point, dims, ir_level),
            )
        prefill, decode = cache[key]

        template_flops = dict(op.get("flops_coeffs", {}))
        template_bytes = dict(op.get("data_bytes_coeffs", {}))

        flops_coeffs, data_coeffs, fit_mode = _fit_coeffs(op, prefill, decode, seq_len)
        op["flops_coeffs"] = flops_coeffs
        op["data_bytes_coeffs"] = data_coeffs
        # 标量字段只是 coeffs 为空时的 fallback，保持 0 以免误用
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
            # tile 级 MRAM↔WRAM 真实搬运字节，只供后续精化访存时延模型参考，
            # 不进 data_bytes（方案三.(3)）。ir_level="ttir" 时为 None。
            "mram_traffic_bytes": (
                None if ir_level == "ttir"
                else {
                    "prefill": prefill.mram_traffic_bytes,
                    "decode": decode.mram_traffic_bytes,
                }
            ),
            # 局部→全局换算（方案三.(6)）当前无对象：实测 PIM pass 不重切 tile，
            # num-dpus 只记进 module 属性、dpusPerDevice 恒为全 1（FlagTree
            # doc 15.9），kernel 编的就是单 DPU 全局 shape。跨 DPU sharding
            # 落地后此处才需要按 placement 乘回去。
            "shard_participants": None,
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
        # mram / net 的放大倍数：WRAM 装不下整个算子，同一份 MRAM 数据要按 tile
        # 反复搬入，这个比值就是「tile 级重复搬运」的量化结果——第 2 步接 pim mlir
        # 拿到的核心新信息
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
    """跑一次融合 flash attention，记录其成本作交叉验证基准。

    FlagGems 实跑 attention 走的是融合 flash_fwd，而 GeneSim 图骨架要求
    96 个分离算子。代表实现（bmm/softmax）成本之和应与融合 kernel 同量级。
    """
    result: Dict[str, Any] = {
        "note": "FlagGems 实跑 attention 为融合 flash 路径；"
                "此处记录融合 kernel 的单层全 head 成本，供与代表实现求和对照",
    }
    pimir = ir_level == "pimir"
    for name, point in (("prefill", prefill_point), ("decode", decode_point)):
        try:
            captured = run_and_capture(
                flash_attention_probe(dims, point), emit_pimir=pimir
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
