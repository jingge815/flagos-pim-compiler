"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import DEVICE_DPU, SPEC_META_KEY

# 定义每层 GEMM 与代表权重的对应关系。
_GEMM_WEIGHT_PATTERNS = (
    "self_attn.q_proj.weight",   # 代表 qkv（GeneSim 把 q/k/v 三个投影合并成一个 GEMM）
    "self_attn.o_proj.weight",   # 代表 proj
    "mlp.gate_proj.weight",      # 代表 fc1（近似，见上）
    "mlp.down_proj.weight",      # 代表 fc2
)


def _get_attr_node(gm: GraphModule, pattern: str):
    matches = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    if len(matches) != 1:
        raise ValueError(f"pattern={pattern!r} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0]


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
    sidecar: Dict[str, Any] = {"version": 1, "source_ir": str(ir_path), "operators": {}}

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]
        if len(gemm_op_ids) != len(_GEMM_WEIGHT_PATTERNS):
            raise ValueError(
                f"layer {layer} 期望 {len(_GEMM_WEIGHT_PATTERNS)} 个 GEMM，"
                f"实际 {len(gemm_op_ids)} 个：{gemm_op_ids}"
            )

        for op_id, pattern in zip(gemm_op_ids, _GEMM_WEIGHT_PATTERNS):
            weight_node = _get_attr_node(gm, f"layers.{layer}.{pattern}")
            spec = weight_node.meta[SPEC_META_KEY]
            if spec.device != DEVICE_DPU:
                continue  # host 权重不写入 DPU 设备提示。

            dpu_id = min(spec.shard_map)
            operators_by_id[op_id]["device_hint"] = "pim"
            sidecar["operators"][str(op_id)] = {"device_hint": "pim", "dpu_id": dpu_id}

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
