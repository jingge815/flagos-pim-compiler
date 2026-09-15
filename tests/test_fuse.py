"""验证 FX 图上的激活与池化融合。

融合是 GML 的硬要求：目标格式把激活放进主算子的 contraction 块，没有独立激活
节点的表达方式。这些测试同时钉住「该折的折了」和「不该折的没折」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse import ACTIVATIONS, format_fusions, fuse_graph


def _targets(gm: GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _linear_then(*tail) -> GraphModule:
    """构造 `linear -> tail[0] -> tail[1] -> ...` 的单链图。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    for target, extra in tail:
        node = graph.call_function(target, (node, *extra))
    graph.output(node)
    return GraphModule({}, graph)


def test_activation_folds_into_producer() -> None:
    gm = _linear_then((torch.ops.aten.relu.default, ()))

    assert fuse_graph(gm) == 1
    assert _targets(gm) == [torch.ops.aten.linear.default]

    tail = next(n.meta[FUSED_TAIL_META_KEY] for n in gm.graph.nodes
                if FUSED_TAIL_META_KEY in n.meta)
    assert tail.activation == "relu"
    assert tail.pool is None


def test_trailing_pool_folds_into_the_same_node() -> None:
    """激活与池化可以同时折进一个节点，对应参考产物的 Conv+Relu+MaxPool。"""
    gm = _linear_then(
        (torch.ops.aten.relu.default, ()),
        (torch.ops.aten.max_pool2d.default, ([3, 3],)),
    )

    assert fuse_graph(gm) == 1
    assert _targets(gm) == [torch.ops.aten.linear.default]

    tail = next(n.meta[FUSED_TAIL_META_KEY] for n in gm.graph.nodes
                if FUSED_TAIL_META_KEY in n.meta)
    assert tail.activation == "relu"
    assert tail.pool == "max"
    assert len(tail.nodes) == 2


def test_pool_without_activation_is_left_alone() -> None:
    """池化只在激活之后才折；单独的池化仍是自己的节点。"""
    gm = _linear_then((torch.ops.aten.max_pool2d.default, ([3, 3],)))

    assert fuse_graph(gm) == 0
    assert torch.ops.aten.max_pool2d.default in _targets(gm)


def test_second_reader_blocks_fusion() -> None:
    """激活前的值还有人读时不能折，否则那个读者会失去定义。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    linear = graph.call_function(torch.ops.aten.linear.default, (x, w))
    relu = graph.call_function(torch.ops.aten.relu.default, (linear,))
    # linear 的结果被 relu 之外的第二处读取。
    add = graph.call_function(torch.ops.aten.add.Tensor, (relu, linear))
    graph.output(add)
    gm = GraphModule({}, graph)

    assert fuse_graph(gm) == 0
    assert torch.ops.aten.relu.default in _targets(gm)


def test_non_fusable_producer_is_left_alone() -> None:
    """只有能持有 contraction 的主算子才折。"""
    graph = Graph()
    x = graph.placeholder("x")
    softmax = graph.call_function(torch.ops.aten._softmax.default, (x, -1, False))
    relu = graph.call_function(torch.ops.aten.relu.default, (softmax,))
    graph.output(relu)
    gm = GraphModule({}, graph)

    assert fuse_graph(gm) == 0
    assert torch.ops.aten.relu.default in _targets(gm)


def test_one_activation_per_node() -> None:
    """一个节点只放一个激活，连续两个激活只折掉第一个。"""
    gm = _linear_then(
        (torch.ops.aten.relu.default, ()),
        (torch.ops.aten.sigmoid.default, ()),
    )

    assert fuse_graph(gm) == 1
    # 第二个激活留在图里，没有节点可以容纳它。
    assert _targets(gm) == [torch.ops.aten.linear.default,
                            torch.ops.aten.sigmoid.default]


def test_repeated_fusion_is_idempotent() -> None:
    gm = _linear_then((torch.ops.aten.relu.default, ()))

    assert fuse_graph(gm) == 1
    assert fuse_graph(gm) == 0


def test_each_eltwise_add_can_hold_an_activation() -> None:
    """参考产物里有 16 个 EltwiseAdd 折了 Relu。"""
    graph = Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    add = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    relu = graph.call_function(torch.ops.aten.relu.default, (add,))
    graph.output(relu)
    gm = GraphModule({}, graph)

    assert fuse_graph(gm) == 1
    assert _targets(gm) == [torch.ops.aten.add.Tensor]


def test_format_fusions_lists_what_was_folded() -> None:
    gm = _linear_then(
        (torch.ops.aten.relu.default, ()),
        (torch.ops.aten.max_pool2d.default, ([3, 3],)),
    )
    fuse_graph(gm)

    assert "relu + max" in format_fusions(gm)

def test_real_llama_graph_leaves_no_fusable_activation_standalone() -> None:
    """真实 llama2 图上，可折的激活一个都不该剩下。

    GML 没有独立「可折激活」节点的表达方式，剩一个就产不出合法 GML。

    注意小 llama2 图里的 rsqrt 与 silu **不在** ACTIVATIONS 里：实物显示它们是
    独立节点（`RMSNorm_vpu` 与 `Silu`），由 OP_TYPES 直接映射，不参与融合。
    所以这张图可能没有任何可折激活——那也是正确状态。
    """
    from tests.test_partition import _export_random_llama

    gm = _export_random_llama()
    before = [n for n in gm.graph.nodes
              if n.op == "call_function" and n.target in ACTIVATIONS]

    fused = fuse_graph(gm)
    assert fused == len(before)

    remaining = [n for n in gm.graph.nodes
                 if n.op == "call_function" and n.target in ACTIVATIONS]
    assert remaining == []
