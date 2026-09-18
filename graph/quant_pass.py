"""插入 DynamicScaling 节点：动态量化每个矩阵乘的激活输入。

这一步贡献参考产物 200 个节点里的 **36 个**，以及 3231 个 bin 里的 **1789 个
（57%）** —— 后者是四相流水线的中间态与定标系数，数值由 `gml_bridge.phase_data`
算，本模块只负责「在哪里插节点」。

**DQ 不来自任何 aten 算子**：aten 图里没有它，是量化 pass 主动插入的。
插入位置的规则从实物反推并逐项核对（37 个节点无例外）：

    在每个矩阵乘的**激活输入**边上插一个 DQ

核对（实测参考产物）：

| 矩阵乘 | 个数 | 入边 | 其中 DQ | 说明 |
| --- | ---: | ---: | ---: | --- |
| `Gemm` | 7 | 1 | **1** | q/k/v/o/gate/up/down_proj，激活侧全要量化 |
| `MatMul` matmul2 (PV) | 32 | 2 | **1** | 吃 Softmax 输出，要量化 |
| `MatMul` matmul1 (QKᵀ) | 32 | 2 | **0** | 吃的是已量化的 Q 与 KV cache，不再量化 |

`use_dynamic_quantization` 字段的分布也吻合：Gemm 7 + MatMul 64 + Split 1。
matmul1 虽然自己不带 DQ 上游，但仍标 `use_dynamic_quantization 1` ——
它的 scale 来自上游 DQ（Q 那一路），走跨节点引用。

分组宽度（唯一按节点变化的 phase 字段，实测只有两种取值）：

| 被量化的张量 | group_size | 组数 |
| --- | ---: | ---: |
| hidden `[1,1,1,4096]` | 128 | 32 |
| MLP 中间态 `[1,1,1,11008]` | 128 | 86 |
| attention scores `[...,1024]` | **1024**（整条一组） | 1 |

与前两个 pass 一致：**只加节点、不删原算子**，图仍可执行、数值不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.fx import GraphModule, Node

from contracts.gml_quant import dq_group_size
from graph.fuse_pim import ABSORBED_META_KEY
from graph.split_heads import (
    HEAD_INDEX_META_KEY,
    HEAD_ROLE_META_KEY,
    ROLE_MATMUL_PV,
    ROLE_MATMUL_QK,
)

# DQ 节点的标记。GML 侧见到它就按 DynamicScaling 发射（4 相字段 + 相应的 bin）。
DQ_META_KEY = "pim_dynamic_scaling"

# 吃激活的矩阵乘。matmul1 排除在外：它的两个 operand 都已是定点。
_MATMUL_TARGETS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
)


@dataclass
class DynamicScalingSpec:
    """一个 DQ 节点的编译期规格。

    `group_size` 决定 phase0 的 `global_pooling_group_size_phase_0`
    与 phase3 的 `kantor_A_spg_group_size_phase_3`（实测两者在同节点上必相等）。

    `numel` / `groups` 决定各相 bin 的元素数，写盘时按它分配。
    `is_attention_scores` 记下它是不是 attention scores 那一路 ——
    那 32 个节点整条当一组。
    """

    group_size: int
    numel: int
    is_attention_scores: bool
    head_index: int | None = None

    @property
    def groups(self) -> int:
        return self.numel // self.group_size


@dataclass
class QuantReport:
    """一次量化 pass 的统计。"""

    inserted: int = 0
    attention_scores: int = 0
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        detail = (f"插入 {self.inserted} 个 DynamicScaling"
                  f"（其中 attention scores {self.attention_scores} 个）")
        if self.skipped:
            detail += f"；跳过 {len(self.skipped)} 处（{self.skipped[0]}）"
        return detail


def _numel_of(node: Node) -> int | None:
    """节点输出的元素数。取不到形状就不插 DQ —— 宁可少插也不要瞎猜分组。"""
    value = node.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _activation_input(node: Node) -> Node | None:
    """矩阵乘的**激活**输入（第一个 operand）。

    第二个 operand 是权重或 KV cache，走权重通路、不经 DQ
    —— 这也是 MatMul 的 `input_count` 比实际入边少记 1 的原因。
    """
    if not node.args:
        return None
    first = node.args[0]
    return first if isinstance(first, Node) else None


def _needs_dq(node: Node) -> bool:
    """这个矩阵乘的激活输入是否需要插 DQ。

    matmul1（QKᵀ）不需要：它吃的是已量化的 Q 与 KV cache。
    实测参考产物 32 个 matmul1 全部没有 DQ 上游，32 个 matmul2 全部有。
    """
    if node.meta.get(ABSORBED_META_KEY):
        return False
    if node.target not in _MATMUL_TARGETS:
        return False
    return node.meta.get(HEAD_ROLE_META_KEY) != ROLE_MATMUL_QK


def _insert_one(gm: GraphModule, consumer: Node, report: QuantReport) -> bool:
    """在 `consumer` 的激活输入边上插一个 DQ 节点。"""
    source = _activation_input(consumer)
    if source is None:
        report.skipped.append(f"{consumer.name}: 取不到激活输入")
        return False
    if source.meta.get(DQ_META_KEY) is not None:
        return False  # 这条边上已经有 DQ 了

    numel = _numel_of(source)
    if not numel:
        report.skipped.append(f"{consumer.name}: 取不到 {source.name} 的形状")
        return False

    # attention scores 那一路整条当一组（实测 32 个节点 group_size=1024）。
    is_scores = consumer.meta.get(HEAD_ROLE_META_KEY) == ROLE_MATMUL_PV
    group_size = dq_group_size(numel, is_attention_scores=is_scores)
    if numel % group_size:
        report.skipped.append(
            f"{consumer.name}: {numel} 不能被 group_size {group_size} 整除")
        return False

    with gm.graph.inserting_before(consumer):
        # 用 `alias` 当载体：它是恒等操作，所以图的数值完全不变
        # —— 真正的量化发生在硬件上，编译期只需要一个占位节点承载那些字段。
        # 换句话说 DQ 在 fx 图里是 no-op，在 GML 里是 4 相流水线。
        dq = gm.graph.call_function(torch.ops.aten.alias.default, (source,))
        dq.name = gm.graph._graph_namespace.create_name(
            f"dynamic_quantization_{consumer.name}", None)

    dq.meta["val"] = source.meta.get("val")
    dq.meta[DQ_META_KEY] = DynamicScalingSpec(
        group_size=group_size,
        numel=numel,
        is_attention_scores=is_scores,
        head_index=consumer.meta.get(HEAD_INDEX_META_KEY),
    )

    # 只把**这个消费者**的那一路改接到 DQ 上。其他消费者仍读原节点
    # —— 实测 hidden 被多个 Gemm 共享，但每个 Gemm 各有自己的 DQ。
    consumer.replace_input_with(source, dq)

    report.inserted += 1
    if is_scores:
        report.attention_scores += 1
    return True


def insert_dynamic_scaling(gm: GraphModule) -> QuantReport:
    """在每个矩阵乘的激活输入上插入 DynamicScaling 节点。

    必须在 `fuse_for_pim` 与 `split_attention_heads` **之后**跑：
    前者决定哪些节点还在（RMSNorm 折叠后的锚点），
    后者产出逐头的 matmul 与角色标记，而插入规则依赖那些角色
    （matmul1 不插、matmul2 插）。
    """
    report = QuantReport()
    consumers = [node for node in gm.graph.nodes if _needs_dq(node)]

    for consumer in consumers:
        _insert_one(gm, consumer, report)

    gm.graph.lint()
    gm.recompile()
    return report
