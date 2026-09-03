"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import DEVICE_DPU, SPEC_META_KEY

# GeneSim IR 里的权重名 → 图编译器侧 `get_attr` 节点名的后缀。
#
# 每层的 GEMM 与权重是一对一的：IR 的 `dependencies` 给出了每个 GEMM 消费的
# 权重 `tensor_id`（形如 `layer.0.q_proj.weight`），据此就能确定它的身份，
# 不必依赖 subgraph 里的出现顺序。
#
# 早先的实现是按固定顺序 zip 四个「代表权重」，因为那时的 model_parser 把
# q/k/v 合并成一个 GEMM、并且用 gate_proj 顶替整个 fc1（漏掉并列的 up_proj）。
# 那种写法有两个后果：qkv 的输出宽度要手工累加三份（GQA 下还要分别取，不能乘
# 三），fc1 的成本天然偏小。现在 IR 把七个投影逐个独立表示，按名字匹配即可，
# 两个近似一起消失。
_GEMM_WEIGHT_SUFFIXES = {
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "gate_proj.weight": "mlp.gate_proj.weight",
    "up_proj.weight": "mlp.up_proj.weight",
    "down_proj.weight": "mlp.down_proj.weight",
}


def _get_attr_node(gm: GraphModule, pattern: str):
    matches = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    if len(matches) != 1:
        raise ValueError(f"pattern={pattern!r} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0]


def _gemm_weight_names(ir: Dict[str, Any]) -> Dict[int, str]:
    """返回 {GEMM 的 op_id: 它消费的权重 tensor_id}。

    以 `dependencies` 里的 `tensor_id` 为准，而不是按 subgraph 顺序推断——顺序
    会随 model_parser 变化，权重名不会。
    """
    op_types = {op["op_id"]: op["op_type"] for op in ir["operators"]}
    names: Dict[int, str] = {}
    for edge in ir.get("dependencies", []):
        tensor_id = edge.get("tensor_id") or ""
        dst = edge.get("dst_op_id")
        if op_types.get(dst) != "GEMM" or not tensor_id.endswith(".weight"):
            continue
        if dst in names and names[dst] != tensor_id:
            raise ValueError(
                f"GEMM op{dst} 消费了多个权重：{names[dst]} 和 {tensor_id}，"
                "无法唯一确定它对应哪个投影"
            )
        names[dst] = tensor_id
    return names


def export_placement_to_genesim(
    gm: GraphModule,
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
) -> Dict[str, Any]:
    """更新 GEMM 的设备提示并输出 IR 文件和代表 DPU 编号。"""
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

    weight_names = _gemm_weight_names(ir)

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]

        for op_id in gemm_op_ids:
            tensor_id = weight_names.get(op_id)
            if tensor_id is None:
                raise ValueError(
                    f"layer {layer} 的 GEMM op{op_id} 在 dependencies 里找不到"
                    "对应的权重 tensor_id，无法确定它是哪个投影"
                )
            suffix = tensor_id.split(".", 2)[-1]
            pattern = _GEMM_WEIGHT_SUFFIXES.get(suffix)
            if pattern is None:
                raise ValueError(
                    f"GEMM op{op_id} 的权重 {tensor_id!r} 不在已知投影列表里："
                    f"{sorted(_GEMM_WEIGHT_SUFFIXES)}。IR 结构变了，需要同步这张表。"
                )

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
            sidecar["operators"][str(op_id)] = {
                "device_hint": "pim",
                "dpu_id": dpu_id,
                # 供消费侧核对 op_id 没有错位（IR 重新生成后编号可能变化）。
                "op_type": operators_by_id[op_id]["op_type"],
                # 供成本提取按本地分片形状重新测量，而不是用模型级全局形状。
                "local_in_features": int(local_in),
                "local_out_features": int(local_out),
                "weight": tensor_id,
            }

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
