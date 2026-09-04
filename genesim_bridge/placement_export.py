"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import DEVICE_DPU, SPEC_META_KEY

if TYPE_CHECKING:  # 只为类型标注；运行时不拖进算子编译那条依赖链。
    from contracts.op_contract import PIMHardwareConfig

# GeneSim IR 的 `semantic_role` → 图编译器侧 `get_attr` 节点名的后缀。
#
# 用 IR 自带的语义标签确定每个 GEMM 是哪个投影，而不是去解析权重名。IR 侧的
# `tensor_id`（形如 `layer.0.q_proj.weight`）是自由字符串，上游改命名规则不会有
# 任何编译期信号；`semantic_role` 是 model_parser 显式写下的身份，改它是一次
# 明确的破坏性变更。
#
# 这张表本身消不掉：右边的 fx 节点名是图编译器侧的事实，IR 不可能知道。消掉的是
# 「依赖 GeneSim 的权重命名约定」和「扫 dependencies 反推身份」这两件事。
_ROLE_TO_WEIGHT_PATTERN = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def _get_attr_node(gm: GraphModule, pattern: str):
    matches = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    if len(matches) != 1:
        raise ValueError(f"pattern={pattern!r} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0]


def _measure_kernel_tile_n(
    local_in: int,
    local_out: int,
    dtype: str,
    hardware: "PIMHardwareConfig | None",
    cache: Dict[tuple[int, int], tuple[int, str]],
) -> tuple[int, str]:
    """编译一个本地分片形状，返回 `(tile_n, pim mlir 路径)`。

    这是算子编译器影响 GeneSim 代价的实际入口：`pim-tile-to-budget` 按 WRAM 预算
    搜出真实分块（llama2 的 4096 宽投影上是 512），而 GeneSim 原先只能用
    `conf/sim.yaml` 里拍下的 32。

    路径一并返回，GeneSim 侧照那份 pim mlir 生成 PIM trace——这一步才让编译器的
    真实 DMA 量和循环嵌套进入周期数，只喂一个分块常量是不够的（手写模板里
    `tile_size` 会约掉）。

    同一形状只编一次，结果放进 `cache`。
    """
    key = (local_in, local_out)
    if key in cache:
        return cache[key]

    # 延迟导入：这条路径需要 CUDA 和 FlagTree，纯放置导出用不上，不该在 import
    # 期就把依赖拽进来。
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    hw = DEFAULT_HARDWARE_CONFIG if hardware is None else hardware
    # M=1 是 decode 口径：算子编译器这条链只支持 M 方向不分块，而分块的搜索只由
    # K/N 和 WRAM 预算决定，与 M 无关。
    request = OpCompileRequest(
        op="linear",
        arg_shapes=[(1, local_in), (local_out, local_in)],
        hardware=hw,
        dtype=dtype,
        num_tasklets=hw.num_tasklets,
    )
    result = compile_op(request)
    if not result.pimir:
        raise RuntimeError(
            f"算子编译没有产出 pim mlir（本地形状 {local_in}->{local_out}）。"
            "缓存里的 .so 可能是旧版本留下的，用 compile_op(force=True) 重编，"
            "或删掉 .opcompiler_cache 后重跑。"
        )
    cost = analyze_ir(
        result.pimir,
        kernel_name="linear_kernel",
        grid=(1,),
        arg_values={},
        ir_level="pimir",
    )
    tile_n = int(cost.tile_n or 0)
    if tile_n <= 0:
        raise RuntimeError(
            f"从 pim mlir 里没读出输出分块（本地形状 {local_in}->{local_out}，"
            f"tile_n={cost.tile_n!r}）。"
        )
    if not result.pimir_path:
        raise RuntimeError(
            f"算子编译没有留下 pim mlir 的落盘路径（本地形状 {local_in}->{local_out}）。"
            "GeneSim 侧要照这份 IR 生成 trace，缺了路径就只能退回手写模板。"
        )
    cache[key] = (tile_n, str(result.pimir_path))
    return cache[key]


def _assert_every_gemm_has_known_role(ir: Dict[str, Any]) -> None:
    """核对 IR 里每个 GEMM 都带一个本表认得的 `semantic_role`。

    正向校验：从 IR 出发查表，而不是从表出发查 IR。上游新增一种投影、或者改了
    role 的写法，都会在这里报错并指出具体算子，而不是等到取 fx 节点时才撞上一句
    「应恰好匹配 1 个」——那个报错指向的位置是错的。
    """
    missing: list[int] = []
    unknown: Dict[int, str] = {}
    for op in ir["operators"]:
        if op["op_type"] != "GEMM":
            continue
        role = op.get("semantic_role") or ""
        if not role:
            missing.append(op["op_id"])
        elif role not in _ROLE_TO_WEIGHT_PATTERN:
            unknown[op["op_id"]] = role

    if missing:
        raise ValueError(
            f"IR 里有 {len(missing)} 个 GEMM 没有 semantic_role（如 "
            f"op{missing[:6]}）。放置导出按语义标签确定投影身份，缺了就无法匹配；"
            "请用当前的 model_parser.py 重新生成 IR。"
        )
    if unknown:
        head = ", ".join(f"op{op_id}={role!r}" for op_id, role in list(unknown.items())[:4])
        raise ValueError(
            f"IR 里有 {len(unknown)} 个 GEMM 的 semantic_role 不在已知投影列表里："
            f"{head}。已知的是 {sorted(_ROLE_TO_WEIGHT_PATTERN)}；"
            "IR 结构变了，需要同步这张表。"
        )


def export_placement_to_genesim(
    gm: GraphModule,
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
    *,
    dtype: str = "float16",
    hardware: "PIMHardwareConfig | None" = None,
    measure_kernel_tiles: bool = False,
    dpu_to_cluster: tuple[tuple[int, int], ...] | None = None,
) -> Dict[str, Any]:
    """更新 GEMM 的设备提示并输出 IR 文件和代表 DPU 编号。

    `measure_kernel_tiles=True` 时，对每种本地分片形状真正跑一次算子编译，从
    pim mlir 里读出 `pim-tile-to-budget` 选定的输出分块，写进 sidecar 的
    `kernel_tile_n`。GeneSim 用它替掉 `conf/sim.yaml` 里拍下的 `tile_size`
    常量——这一项是「算子编译器影响 GeneSim 代价」的实际载体。

    默认关闭：它需要 CUDA、FlagTree 和 triton-opt，纯放置导出用不上。

    `dpu_to_cluster` 来自 GeneSim 给定的切分方案（`contracts/partition_plan.py`），
    原样写进 sidecar 顶层，让 GeneSim 按它把逻辑 DPU 落到指定的 ClusterPU 上，
    而不是走 `dpu_id % len(cluster_keys)` 的取模换算。往返用同一份声明，编号就不会错位。
    """
    ir = json.loads(Path(ir_path).read_text())
    num_layers = ir["num_layers"]
    if len(ir["subgraphs"]) != num_layers:
        raise ValueError(f"subgraphs 层数 {len(ir['subgraphs'])} 与 num_layers {num_layers} 不一致")

    operators_by_id = {op["op_id"]: op for op in ir["operators"]}
    # `source_ir` 只作人读线索，不能用来校验：它是绝对路径，换机器就失效，而
    # 仿真通常加载的是同一批 op_id 的另一个 IR（成本精化后的 *_pimir.ir）。
    # 真正可校验的是内容——算子总数和每个被放置算子的 op_type，见
    # GeneSim 侧 _load_compiler_placement 的比对。
    sidecar: Dict[str, Any] = {
        "version": 2,
        "source_ir": str(ir_path),
        "ir_num_operators": len(ir["operators"]),
        "operators": {},
    }
    if dpu_to_cluster is not None:
        # 顶层字段而非 per-op：它描述的是"逻辑 DPU k 落在哪个 ClusterPU"，
        # 与算子无关。GeneSim 侧按它分配，缺失则退回取模换算。
        sidecar["dpu_to_cluster"] = [list(entry) for entry in dpu_to_cluster]

    _assert_every_gemm_has_known_role(ir)

    # {(local_in, local_out): tile_n}。同一形状只编一次——llama2 的 224 个 GEMM
    # 只有 7 种本地形状，逐个编会白跑 217 次。
    tile_cache: Dict[tuple[int, int], tuple[int, str]] = {}

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]

        for op_id in gemm_op_ids:
            role = operators_by_id[op_id]["semantic_role"]
            pattern = _ROLE_TO_WEIGHT_PATTERN[role]
            weight_node = _get_attr_node(gm, f"layers.{layer}.{pattern}")
            spec = weight_node.meta[SPEC_META_KEY]
            if spec.device != DEVICE_DPU:
                continue  # host 权重不写入 DPU 设备提示。

            dpu_id = min(spec.shard_map)
            # 权重的 local_shape 是 (out_features, in_features)，与这个 GEMM 在该
            # DPU 上要算的矩阵乘宽度一一对应——IR 里每个投影都是独立的 GEMM，
            # 不需要再累加或折算。
            local_out, local_in = spec.shard_map[dpu_id].local_shape
            operators_by_id[op_id]["device_hint"] = "pim"
            # 本地分片形状写进 IR 的新字段，原 input/output_shapes 保持全局语义
            # 不动。GeneSim 的执行侧（compile_gemm 的 K/N、_execute_runtime 的
            # 循环次数）优先取这一份，切分才真正影响仿真出来的时间。
            #
            # 形状口径与全局字段一致：(Tq, in_features) → (Tq, out_features)，
            # 前导的 Tq 是符号维，原样搬过来不做解析。
            global_in = operators_by_id[op_id]["input_shapes"][0]
            global_out = operators_by_id[op_id]["output_shapes"][0]
            operators_by_id[op_id]["local_input_shapes"] = [
                [global_in[0], int(local_in)]
            ]
            operators_by_id[op_id]["local_output_shapes"] = [
                [global_out[0], int(local_out)]
            ]
            sidecar["operators"][str(op_id)] = {
                "device_hint": "pim",
                "dpu_id": dpu_id,
                # 供消费侧核对 op_id 没有错位（IR 重新生成后编号可能变化）。
                "op_type": operators_by_id[op_id]["op_type"],
                # 供成本提取按本地分片形状重新测量，而不是用模型级全局形状。
                "local_in_features": int(local_in),
                "local_out_features": int(local_out),
                # 这个 GEMM 的投影身份，以及它实际匹配到的 fx 节点——排错时不用
                # 再猜「哪个 role 落到了哪个权重上」。
                "semantic_role": role,
                "weight": str(weight_node.target),
            }

            if measure_kernel_tiles:
                tile_n, pimir_path = _measure_kernel_tile_n(
                    int(local_in), int(local_out), dtype, hardware, tile_cache
                )
                entry = sidecar["operators"][str(op_id)]
                entry["kernel_tile_n"] = tile_n
                # GeneSim 照这份 pim mlir 生成 PIM trace，而不是用手写模板。
                entry["pimir_path"] = pimir_path

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
