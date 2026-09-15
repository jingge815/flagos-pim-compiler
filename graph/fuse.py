"""把激活与紧随的池化折进产生它们输入的算子。

GML 把激活放进主算子的 `contraction` 块，没有独立激活节点的表达方式，所以这一步
是正确性要求而不是优化：不融合就产不出合法 GML。只做「主算子 + 激活 + 可选池化」，
不做通用融合。

与 FlagTree 的 `-pim-fuse-activation` 是同一套语义，两侧都按同样的条件拒绝融合。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from contracts.graph_meta import FUSED_TAIL_META_KEY


# 可以持有折入激活的主算子。GML 里对应 Conv / Gemm / MatMul / EltwiseAdd 这些
# 带 contraction 块的节点。
FUSION_TARGETS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    }
)

# 可以折进主算子 contraction 块的激活。目标平台用查表实现激活，所以这些在硬件上
# 是同一条指令配不同的表。
#
# `silu` 与 `rsqrt` **不在**这里：llama2 W4A8 实物显示它们是独立节点
# （`Silu` 自带 nmu_mode/fpsu_*/kantor_mode，`RMSNorm_vpu` 绑定在向量单元上并带
# vpu 专属字段），归入 gml_bridge.from_fx.OP_TYPES。放进来会被融合吃掉，
# 产出的 GML 就少了这两类节点。
ACTIVATIONS = {
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.gelu.default: "gelu",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.reciprocal.default: "reciprocal",
}

# 池化算子到 GML `op_type` 的映射。
POOLS = {
    torch.ops.aten.max_pool2d.default: "max",
    torch.ops.aten.avg_pool2d.default: "average",
    torch.ops.aten.mean.dim: "global_average",
}


@dataclass
class FusedTail:
    """折进主算子的尾部算子。

    `activation` 是激活的 GML 名；`pool` 是紧随其后的池化，没有则为 None。
    `nodes` 保留被折掉的 FX 节点，供序列化器生成 contraction 块时取参数。
    """

    activation: str
    pool: str | None
    nodes: list[Node]


def _single_consumer(node: Node) -> bool:
    """节点的输出是否只被一处读取。

    有第二个读者时不能折：折叠会把主算子的结果改写成激活后的值，
    另一个读者要的却是激活前的。
    """
    return len(node.users) == 1


def _activation_after(node: Node) -> Node | None:
    """紧跟 `node` 且唯一消费它的激活节点。"""
    if not _single_consumer(node):
        return None
    consumer = next(iter(node.users))
    if consumer.op == "call_function" and consumer.target in ACTIVATIONS:
        return consumer
    return None


def _pool_after(node: Node) -> Node | None:
    """紧跟 `node` 且唯一消费它的池化节点。"""
    if not _single_consumer(node):
        return None
    consumer = next(iter(node.users))
    if consumer.op == "call_function" and consumer.target in POOLS:
        return consumer
    return None


def fuse_graph(gm: GraphModule) -> int:
    """原地折叠 `gm` 里的激活与池化，返回折叠出的节点数。

    折叠后被吃掉的节点从图中删除，主算子的 `meta[FUSED_TAIL_META_KEY]` 记下
    折进来的是什么。调用方拿到的图里不再有独立激活节点。
    """
    fused = 0
    # 先收集：融合会删节点，边遍历边改不安全。
    candidates = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target in FUSION_TARGETS
    ]

    for main in candidates:
        # 一个节点只放一个激活，已经折过的不再折。
        if FUSED_TAIL_META_KEY in main.meta:
            continue

        activation = _activation_after(main)
        if activation is None:
            continue

        pool = _pool_after(activation)
        eaten = [activation] if pool is None else [activation, pool]
        last = eaten[-1]

        main.meta[FUSED_TAIL_META_KEY] = FusedTail(
            activation=ACTIVATIONS[activation.target],
            pool=POOLS[pool.target] if pool is not None else None,
            nodes=eaten,
        )
        # 融合后的节点产出的是整条链的结果，所以读者要改接到主算子上。
        last.replace_all_uses_with(main)
        for node in reversed(eaten):
            gm.graph.erase_node(node)
        fused += 1

    if fused:
        gm.graph.lint()
        gm.recompile()
    return fused


def format_fusions(gm: GraphModule) -> str:
    """列出图里的融合结果，供人工核对。"""
    lines = []
    for node in gm.graph.nodes:
        tail = node.meta.get(FUSED_TAIL_META_KEY)
        if tail is None:
            continue
        folded = tail.activation
        if tail.pool:
            folded += f" + {tail.pool}"
        lines.append(f"{node.name}: {folded}")
    return "\n".join(lines) if lines else "（无融合）"
