"""图编译器放置结果 → GeneSim 映射（与 cost_extractor.py 并列，同样改写整份 .ir）。

方案依据：本轮只对接 GEMM 类算子。图编译器的 DPU 白名单
（`graph/partition.py::DPU_LOWERABLE`）把 q/k/v/o_proj、gate/up/down_proj 这类
线性层判定为可下沉 DPU（张量并行分片执行），而 GeneSim 自带的 llama2 IR 构造器
（`ir.model_ir.build_from_hf_config`）把这些 GEMM 默认标成 `device_hint="gpu"`，
只把 attention 内部的 score/softmax/context 标成 `"pim"`——两者对大 GEMM 的判断
方向相反。这里用图编译器的真实判定覆盖 GeneSim 默认值；attention 三类算子本轮
不动，继续吃 GeneSim 自己的启发式。

GEMM 在图编译器里是 8 台 DPU 张量并行分片（`PIMTensorSpec.shard_map` 一个算子
同时对应多个 dpu_id），但 GeneSim 的 `Operator.attached_pu_id` 只能绑一个 PU，
不支持"一个算子分布在多个 PU"。本轮简化处理：整个算子绑 `shard_map` 里编号最小
的那个 DPU 作代表，不用 GeneSim 的 `split_operator` 精确还原张量并行（用户已确
认这版先做简化版）。

GeneSim 没有"sidecar 在加载时合并"的机制（不同于这里说的 cost_extractor 输出的
sidecar，那份纯粹是记录，GeneSim 从不读它）——所以这里跟 `export_costs_to_genesim`
一样，直接改写整份 `.ir` 落盘，`out_ir_path` 就是 GeneSim 会加载的文件。放置 sidecar
（`{op_id: {"device_hint", "dpu_id"}}`）另外单独落盘一份，供改动四
（`gene_sim_scheduler.py` 的 gated 分支）读取——`dpu_id` 是图编译器内部的 DPU 编号
（0..num_dpus-1），不是 GeneSim 的 `attached_pu_id`，数值空间不同，翻译成
`attached_pu_id` 是 GeneSim 侧的事（见 `_load_compiler_placement`）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import DEVICE_DPU, SPEC_META_KEY

# 每层恰好 4 个 GEMM：qkv、o_proj、fc1(gate_proj)、fc2(down_proj)，顺序与
# ir.model_ir.build_from_hf_config 的 add_op 调用顺序一致（见该函数第 500-644 行）。
# fc1 用 gate_proj 做代表权重是近似：GeneSim 的 IR 模型把 MLP 建成单个 fc1+GELU，
# 与 llama2 真实的 SwiGLU（gate_proj+up_proj 两路+SiLU+逐元素乘）结构不一致，这是
# GeneSim IR 模型自带的既有简化，本轮不修——只在近似对应关系上选一个代表权重。
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
    """读 GeneSim .ir，用图编译器的 GEMM 放置结果覆盖 device_hint，落盘 .ir + sidecar。

    不改算子个数、类型、连接、shape——只改 GEMM 算子的 `device_hint`，并把
    代表 DPU 编号记进 sidecar。每层的 4 个 GEMM 数量、顺序与
    `_GEMM_WEIGHT_PATTERNS` 不一致就直接抛错，不做静默兜底。
    """
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
                continue  # host 侧权重（第 1 阶段窄白名单外），保持 GeneSim 原有 device_hint

            dpu_id = min(spec.shard_map)
            operators_by_id[op_id]["device_hint"] = "pim"
            sidecar["operators"][str(op_id)] = {"device_hint": "pim", "dpu_id": dpu_id}

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
