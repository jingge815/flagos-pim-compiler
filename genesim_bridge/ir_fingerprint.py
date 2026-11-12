"""GeneSim 图骨架的结构指纹。

覆盖模型几何、执行 dtype/语义、MoE 几何、完整算子编号/类型/形状/角色、
依赖拓扑与层归属。它不含路径、时间与成本系数：精化只回填成本，指纹必须
保持不变；词表、hidden、专家几何、任一算子形状、依赖端点/张量身份或层
归属变了就必须变。

算法与 `genesim/src/ir/model_ir.py` 的 `ModelIR.structural_fingerprint` 逐字段
一致；两边由 `tests/test_ir_fingerprint.py` 的跨仓用例钉住，改动必须同步。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalized_dependencies(entries: list) -> list[list]:
    """把当前字典格式与旧式 (src, dst) 依赖规范成 [src, dst, tensor_id]。"""
    normalized = []
    for entry in entries:
        if isinstance(entry, dict):
            normalized.append([
                int(entry["src_op_id"]),
                int(entry["dst_op_id"]),
                str(entry.get("tensor_id", "")),
            ])
        else:
            src_op_id, dst_op_id = entry
            normalized.append([int(src_op_id), int(dst_op_id), ""])
    return sorted(normalized)


def fingerprint_ir(ir: dict[str, Any]) -> str:
    """按 IR 字典重算结构指纹，返回 64 位十六进制串。"""
    payload = {
        "schema": 2,
        "model_type": str(ir.get("model_type", "")),
        "num_layers": int(ir.get("num_layers", 0)),
        "num_heads": int(ir.get("num_heads", 0)),
        "num_kv_heads": int(ir.get("num_kv_heads", 0)),
        "head_dim": int(ir.get("head_dim", 0)),
        "hidden_size": int(ir.get("hidden_size", 0)),
        "vocab_size": int(ir.get("vocab_size", 0)),
        "parameter_bytes": int(ir.get("parameter_bytes", 0)),
        "execution_dtype": str(ir.get("execution_dtype", "")),
        "tie_word_embeddings": bool(ir.get("tie_word_embeddings", False)),
        "moe": {
            "num_experts": int(ir.get("moe_num_experts", 0) or 0),
            "top_k": int(ir.get("moe_top_k", 0) or 0),
            "intermediate_size": int(ir.get("moe_intermediate_size", 0) or 0),
            "execution": str(ir.get("moe_execution", "") or ""),
        },
        "operators": [
            [
                int(op["op_id"]),
                str(op["op_type"]),
                str(op.get("semantic_role", "")),
                int(op.get("expert_id", -1)),
                str(op.get("moe_cost_role", "")),
                [list(shape) for shape in op.get("input_shapes", [])],
                [list(shape) for shape in op.get("output_shapes", [])],
            ]
            for op in ir["operators"]
        ],
        # 依赖端点与张量身份、层归属都是结构的一部分；成本系数、设备提示与
        # 本地分片形状属于精化/放置字段，必须继续排除。
        "dependencies": _normalized_dependencies(
            ir.get("dependencies", [])),
        "subgraphs": [[int(op_id) for op_id in group]
                      for group in ir.get("subgraphs", [])],
    }
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fusion_signature(entries: list[dict[str, Any]]) -> str:
    """融合清单的规范化签名：层号/角色/激活/专家号排序后取哈希。

    精化缓存必须把融合清单算进键：同一份 IR 在「折激活」与「不折」两种
    输入下回填的成本不同，只看编号序列会错误复用。
    """
    normalized = sorted(
        (
            int(entry.get("layer", -1)),
            str(entry.get("role", "")),
            str(entry.get("activation", "")),
            int(entry.get("expert", -1)),
        )
        for entry in entries
    )
    text = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_ir_fingerprint(ir: dict[str, Any], source: str) -> str:
    """返回 IR 的结构指纹；IR 自报的指纹与内容不符时直接失败。"""
    actual = fingerprint_ir(ir)
    stored = str(ir.get("model_fingerprint", "") or "")
    if stored and stored != actual:
        raise ValueError(
            f"{source} 的 model_fingerprint={stored[:16]} 与内容重算的 "
            f"{actual[:16]} 不一致：IR 生成后被改过，请用当前 model_parser "
            "重新生成，不能把旧身份带到仿真。")
    return actual
