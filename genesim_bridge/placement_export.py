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

# 本地输出宽度需要把哪几个权重的份额加在一起。
#
# GeneSim 的 IR 把 q/k/v 三个投影合并成一个 GEMM（真实 llama2-7b 上是
# 4096 -> 12288 = 3 x 4096），而代表权重只是 q_proj。只拿 q_proj 的本地
# out_features 当这个 GEMM 的输出宽度，成本会被低估到三分之一。
#
# 这里逐个累加 q/k/v 的实际本地宽度，而不是把 q 乘三：分组查询注意力（GQA）
# 下 k/v 的头数少于 q，三者宽度并不相同。实测 q_heads=8 / kv_heads=4 的模型，
# 真实的本地宽度之和是 128+64+64=256，而按 q 乘三算出 384，高估一点五倍。
# llama2-7b 本身是 32/32（非 GQA）两种算法等价，但 `llama_strategy` 接受
# num_kv_heads != num_heads，所以不能依赖那个巧合。
#
# 另外三个 GEMM 与代表权重一对一，各自只累加自己。注意 fc1 的口径本来就只算了
# gate_proj、没算并列的 up_proj，这是 GeneSim 图骨架自身的近似，与本地形状
# 改造无关，改造前后一样。
_GEMM_OUT_FEATURE_SOURCES = (
    ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
    ("self_attn.o_proj.weight",),
    ("mlp.gate_proj.weight",),
    ("mlp.down_proj.weight",),
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

        for op_id, pattern, out_sources in zip(
            gemm_op_ids, _GEMM_WEIGHT_PATTERNS, _GEMM_OUT_FEATURE_SOURCES
        ):
            weight_node = _get_attr_node(gm, f"layers.{layer}.{pattern}")
            spec = weight_node.meta[SPEC_META_KEY]
            if spec.device != DEVICE_DPU:
                continue  # host 权重不写入 DPU 设备提示。

            dpu_id = min(spec.shard_map)
            # 权重的 local_shape 是 (out_features, in_features)。in_features 取
            # 代表权重的（同一个 GEMM 的几个来源权重共享输入），out_features 把
            # 各来源的份额累加，见 _GEMM_OUT_FEATURE_SOURCES。
            _, local_in = spec.shard_map[dpu_id].local_shape
            local_out = 0
            for source in out_sources:
                source_spec = _get_attr_node(
                    gm, f"layers.{layer}.{source}"
                ).meta[SPEC_META_KEY]
                local_out += source_spec.shard_map[dpu_id].local_shape[0]

            operators_by_id[op_id]["device_hint"] = "pim"
            sidecar["operators"][str(op_id)] = {
                "device_hint": "pim",
                "dpu_id": dpu_id,
                # 供消费侧核对 op_id 没有错位（IR 重新生成后编号可能变化）。
                "op_type": operators_by_id[op_id]["op_type"],
                # 供成本提取按本地分片形状重新测量，而不是用模型级全局形状。
                "local_in_features": int(local_in),
                "local_out_features": int(local_out),
            }

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
