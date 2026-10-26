"""图层五类通用优化的结构断言与数值对拍。

每个 Pass 两条用例：一条落在适用范围内，断言图真的被改了且数值不变；
一条落在不支持范围内，断言图原样不动。合成图手写，不走 torch.export。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.fx import Graph, GraphModule

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.opt_passes import (
    eliminate_common,
    eliminate_dead,
    fold_constants,
    fuse_elementwise,
    fuse_linear_activation,
    optimize_graph,
    tile_graph,
)


def _targets(gm: GraphModule) -> list:
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _shape(gm: GraphModule, *examples) -> None:
    """给手写图的节点补上输出形状，分块要靠它比较预算。

    ShapeProp 写的是 `tensor_meta`，而分块只认已登记的 `val`，所以这里按
    推出来的形状造一个空张量挂到 `val` 上。
    """
    from torch.fx.passes.shape_prop import ShapeProp

    gm.recompile()
    ShapeProp(gm).propagate(*examples)
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if meta is not None and "val" not in node.meta:
            node.meta["val"] = torch.empty(meta.shape, dtype=meta.dtype)


def _run(gm: GraphModule, **values) -> torch.Tensor:
    """用具体张量跑一遍图，得到输出。"""
    gm.recompile()
    return gm.forward(**values)


# ---------------------------------------------------------------- 常量折叠

def _const_add_graph() -> tuple[GraphModule, torch.Tensor, torch.Tensor]:
    """两个常量相加，结果乘上输入。折叠后 add 应消失。"""
    graph = Graph()
    a = graph.get_attr("a")
    b = graph.get_attr("b")
    added = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.mul.Tensor, (added, x))
    graph.output(out)
    a_val = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b_val = torch.tensor([10.0, 20.0, 30.0, 40.0])
    gm = GraphModule({"a": a_val, "b": b_val}, graph)
    return gm, a_val, b_val


def test_fold_constant_chain() -> None:
    """常量输入的 add 被替换成常量节点，输出值等于手算。"""
    gm, a_val, b_val = _const_add_graph()
    expected = (a_val + b_val) * torch.tensor([2.0, 2.0, 2.0, 2.0])

    changed = fold_constants(gm)

    assert changed > 0, "常量链没有被折叠"
    assert torch.ops.aten.add.Tensor not in _targets(gm), _targets(gm)
    got = _run(gm, x=torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert torch.equal(got, expected), (got, expected)


def test_fold_keeps_keyword_arguments() -> None:
    """关键字参数参与求值：alpha 与 exponent 都不能在折叠时丢掉。

    add(a, b, alpha=2) 是 a + 2*b，不是 a + b；pow 的 exponent 只以关键字
    形式出现时，丢掉它会直接抛缺参错误。
    """
    graph = Graph()
    a = graph.get_attr("a")
    b = graph.get_attr("b")
    added = graph.call_function(
        torch.ops.aten.add.Tensor, (a, b), {"alpha": 2})
    graph.output(added)
    gm = GraphModule(
        {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
        graph)

    changed = fold_constants(gm)

    assert changed == 1, "带 alpha 的常量 add 没有被折叠"
    assert torch.equal(gm(), torch.tensor([7.0, 10.0])), gm()

    graph = Graph()
    base = graph.get_attr("base")
    powered = graph.call_function(
        torch.ops.aten.pow.Tensor_Scalar, (base,), {"exponent": 3})
    graph.output(powered)
    gm = GraphModule({"base": torch.tensor([2.0, 3.0])}, graph)

    changed = fold_constants(gm)

    assert changed == 1, "关键字 exponent 的常量 pow 没有被折叠"
    assert torch.equal(gm(), torch.tensor([8.0, 27.0])), gm()


def test_fold_skips_dynamic_keyword_argument() -> None:
    """关键字参数来自占位符时不折，也不能因为缺参而崩溃。"""
    graph = Graph()
    base = graph.get_attr("base")
    exponent = graph.placeholder("exponent")
    powered = graph.call_function(
        torch.ops.aten.pow.Tensor_Scalar, (base,), {"exponent": exponent})
    graph.output(powered)
    gm = GraphModule({"base": torch.tensor([2.0, 3.0])}, graph)

    changed = fold_constants(gm)

    assert changed == 0, "动态 exponent 被当成常量折掉了"
    assert torch.ops.aten.pow.Tensor_Scalar in _targets(gm)


def test_fold_skips_dynamic() -> None:
    """同型 add 的输入是占位符时，节点保留。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    added = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
    graph.output(added)
    gm = GraphModule({}, graph)

    changed = fold_constants(gm)

    assert changed == 0
    assert torch.ops.aten.add.Tensor in _targets(gm)


def test_fold_skips_oversized_result() -> None:
    """结果元素数超过上限的常量链不折，避免把大计算搬进编译期。"""
    from graph import opt_passes

    graph = Graph()
    a = graph.get_attr("a")
    b = graph.get_attr("b")
    added = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    graph.output(added)
    n = opt_passes.FOLD_MAX_ELEMENTS + 1
    gm = GraphModule(
        {"a": torch.zeros(n), "b": torch.ones(n)}, graph)

    changed = fold_constants(gm)

    assert changed == 0, "超大常量结果仍被折叠"
    assert torch.ops.aten.add.Tensor in _targets(gm)


def test_fold_skips_side_effect() -> None:
    """带副作用的算子即使输入全是常量也不折。"""
    graph = Graph()
    n = graph.get_attr("n")
    drawn = graph.call_function(torch.ops.aten.randint.low, (0, n, [4]))
    graph.output(drawn)
    gm = GraphModule({"n": torch.tensor(10)}, graph)

    changed = fold_constants(gm)

    assert changed == 0
    assert torch.ops.aten.randint.low in _targets(gm)


def test_real_llama_graph_has_nothing_to_fold() -> None:
    """真实导出图上没有可折对象，常量折叠改动数为 0 是预期。

    一层 Llama 的可折叠节点输入全是激活或模型参数，没有编译期常量链。
    这是默认通路的口径：折叠只在合成图上验证，见需求文档第五节第 3 条。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval()
    position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, 16, position_ids, dtype=torch.float32)

    assert fold_constants(gm) == 0, "真实导出图上出现了可折常量，口径需要重审"


def test_fold_dotted_parameter() -> None:
    """带点的参数名也是常量，两个参数相加应折成一个常量。

    trace 出来的参数名形如 `lin.weight`。按名字是否带点排除会把整条常量链漏掉。
    """
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2, bias=True)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
                self.lin.bias.copy_(torch.tensor([10.0, 20.0]))

        def forward(self, x):
            return self.lin.weight + self.lin.bias

    model = Model()
    expected = model.lin.weight + model.lin.bias
    gm = torch.fx.symbolic_trace(model)
    # trace 用内置运算符，执行路径的图用 aten 重载。这里换成重载，
    # 让用例打到同一条折叠判定上。
    for node in gm.graph.nodes:
        if node.op == "call_function":
            node.target = torch.ops.aten.add.Tensor

    changed = fold_constants(gm)

    assert changed == 1, "参数常量链没有被折叠"
    assert torch.equal(gm(torch.zeros(1)), expected), gm(torch.zeros(1))


def test_fold_skips_non_tensor_attribute() -> None:
    """不带点的 get_attr 不一定是张量，子模块引用不能当成常量去求值。"""
    root = torch.nn.Sequential(torch.nn.ReLU())
    graph = Graph()
    sub = graph.get_attr("0")
    graph.output(sub)
    gm = GraphModule(root, graph)

    changed = fold_constants(gm)

    assert changed == 0


def test_fold_keeps_shape_ops() -> None:
    """常量输入的 reshape、transpose、squeeze 保留，不折成常量。

    形状类不产生新数值。折掉会改变切分传播看到的节点，跨段通信边随之变化。
    """
    graph = Graph()
    a = graph.get_attr("a")
    reshaped = graph.call_function(torch.ops.aten.reshape.default, (a, [2, 2]))
    transposed = graph.call_function(torch.ops.aten.transpose.int, (a, 0, 1))
    squeezed = graph.call_function(torch.ops.aten.squeeze.dim, (a, 0))
    graph.output((reshaped, transposed, squeezed))
    gm = GraphModule({"a": torch.arange(4)}, graph)

    changed = fold_constants(gm)

    assert changed == 0, "形状类算子被折成了常量"
    kept = _targets(gm)
    assert torch.ops.aten.reshape.default in kept, kept
    assert torch.ops.aten.transpose.int in kept, kept
    assert torch.ops.aten.squeeze.dim in kept, kept


# ---------------------------------------------------------------- CSE

def _dup_mul_graph() -> GraphModule:
    """两处 mul 参数完全相同，后面各接一次加。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    m1 = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    m2 = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    s = graph.call_function(torch.ops.aten.add.Tensor, (m1, m2))
    graph.output(s)
    return GraphModule({}, graph)


def test_cse_merges_identical() -> None:
    """两处参数相同的 mul 合并成一处，输出值不变。"""
    gm = _dup_mul_graph()
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([2.0, 3.0, 4.0, 5.0])
    expected = x * y + x * y

    changed = eliminate_common(gm)

    assert changed > 0, "重复的 mul 没有合并"
    assert _targets(gm).count(torch.ops.aten.mul.Tensor) == 1, _targets(gm)
    got = _run(gm, x=x, y=y)
    assert torch.equal(got, expected)



def test_cse_keeps_shape_ops() -> None:
    """形状类算子不合并，哪怕结构相同。

    每层各有一次 unsqueeze，合到一起会让后面的层跨 stage 读第一层的结果。
    """
    graph = Graph()
    x = graph.placeholder("x")
    u1 = graph.call_function(torch.ops.aten.unsqueeze.default, (x, 1))
    u2 = graph.call_function(torch.ops.aten.unsqueeze.default, (x, 1))
    s = graph.call_function(torch.ops.aten.add.Tensor, (u1, u2))
    graph.output(s)
    gm = GraphModule({}, graph)

    changed = eliminate_common(gm)

    assert changed == 0
    assert _targets(gm).count(torch.ops.aten.unsqueeze.default) == 2



def test_cse_merges_shape_ops_without_pipeline() -> None:
    """没有流水并行时，结构相同的形状类算子也合并，输出值不变。"""
    graph = Graph()
    x = graph.placeholder("x")
    u1 = graph.call_function(torch.ops.aten.unsqueeze.default, (x, 1))
    u2 = graph.call_function(torch.ops.aten.unsqueeze.default, (x, 1))
    s = graph.call_function(torch.ops.aten.add.Tensor, (u1, u2))
    graph.output(s)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0])

    changed = eliminate_common(gm, merge_shape_ops=True)

    assert changed == 1
    assert _targets(gm).count(torch.ops.aten.unsqueeze.default) == 1
    assert torch.equal(_run(gm, x=x_val), (x_val + x_val).unsqueeze(1))


def test_real_llama_graph_cse_merges_layer_shape_ops() -> None:
    """真实两层 Llama 图上，各层重复的 rope cos/sin 形状算子被合并，输出不变。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval()
    position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, 16, position_ids, dtype=torch.float32)
    mask = torch.full((16, 16), float("-inf")).triu(1)[None, None]
    inputs = [torch.randint(0, 128, (1, 16)), mask, position_ids]
    expected = gm(*inputs)

    assert eliminate_common(gm) == 0, "默认口径不该合并形状类算子"
    changed = eliminate_common(gm, merge_shape_ops=True)
    gm.recompile()

    assert changed > 0, "两层的 unsqueeze 没有合并"
    got = gm(*inputs)
    for g, e in zip(torch.utils._pytree.tree_leaves(got),
                    torch.utils._pytree.tree_leaves(expected)):
        torch.testing.assert_close(g, e)


def test_cse_keeps_different_kwargs() -> None:
    """关键字参数不同的同型算子不合并，否则结果会静默算错。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    a1 = graph.call_function(
        torch.ops.aten.add.Tensor, (x, y), {"alpha": 1})
    a2 = graph.call_function(
        torch.ops.aten.add.Tensor, (x, y), {"alpha": 2})
    s = graph.call_function(torch.ops.aten.add.Tensor, (a1, a2))
    graph.output(s)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0])
    y_val = torch.tensor([3.0, 4.0])
    expected = (x_val + y_val) + (x_val + 2 * y_val)

    changed = eliminate_common(gm)

    assert changed == 0, "alpha 不同的两个 add 被合并了"
    got = _run(gm, x=x_val, y=y_val)
    assert torch.equal(got, expected), (got, expected)


def test_cse_keeps_literals_of_different_type() -> None:
    """`add(x, 1)` 与 `add(x, 1.0)` 不合并：两者相等但结果类型不同。"""
    graph = Graph()
    x = graph.placeholder("x")
    a = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
    b = graph.call_function(torch.ops.aten.add.Tensor, (x, 1.0))
    graph.output((a, b))
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1, 2], dtype=torch.int32)

    changed = eliminate_common(gm)

    assert changed == 0, "字面量 1 与 1.0 被当成同一个"
    gm.recompile()
    got = gm(x_val)
    assert [t.dtype for t in got] == [torch.int32, torch.float32]


def test_cse_merges_unlisted_pure_ops() -> None:
    """性质表外的有计算纯算子默认也合并：归约、矩阵乘、激活、拼接，数值不变。"""
    aten = torch.ops.aten
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    pairs = [(aten.mean.dim, (x, [-1])), (aten.linear.default, (x, w)),
             (aten.gelu.default, (x,)), (aten.sin.default, (x,)),
             (aten.cat.default, ([x, x],))]
    outs = []
    for target, args in pairs:
        outs += [graph.call_function(target, args) for _ in range(2)]
    graph.output(tuple(outs))
    gm = GraphModule({}, graph)
    x_val, w_val = torch.randn(2, 8), torch.randn(4, 8)
    gm.recompile()
    expected = gm(x_val, w_val)

    changed = eliminate_common(gm)

    assert changed == len(pairs), _targets(gm)
    for target, _ in pairs:
        assert _targets(gm).count(target) == 1, (target, _targets(gm))
    gm.recompile()
    for g, e in zip(gm(x_val, w_val), expected):
        assert torch.equal(g, e)


def test_cse_keeps_random_ops() -> None:
    """随机算子结构相同也不合并，两次采样各自独立。"""
    graph = Graph()
    x = graph.placeholder("x")
    r1 = graph.call_function(torch.ops.aten.rand_like.default, (x,))
    r2 = graph.call_function(torch.ops.aten.rand_like.default, (x,))
    graph.output((r1, r2))
    gm = GraphModule({}, graph)

    assert eliminate_common(gm, merge_shape_ops=True) == 0
    assert _targets(gm).count(torch.ops.aten.rand_like.default) == 2


def test_cse_keeps_clone_written_in_place() -> None:
    """真实导出图里一个 clone 之后被原地写，两个 clone 不能合并成一块缓冲。"""

    class CloneThenMutate(torch.nn.Module):
        def forward(self, x):
            a = x.clone()
            b = x.clone()
            b.mul_(2.0)
            return a + b

    x = torch.ones(4)
    gm = torch.export.export(CloneThenMutate(), (x,), strict=True).module()
    assert torch.ops.aten.mul_.Tensor in _targets(gm), "导出图里要有原地写"
    expected = gm(x)

    report = optimize_graph(gm, tile=False)

    assert report.merged == 0
    assert torch.equal(gm(x), expected)


def test_cse_keeps_empty_like_filled_separately() -> None:
    """两个 empty_like 各自被 fill_ 写成不同值，合并后两个输出会变成同一个值。"""
    aten = torch.ops.aten
    graph = Graph()
    x = graph.placeholder("x")
    e1 = graph.call_function(aten.empty_like.default, (x,))
    e2 = graph.call_function(aten.empty_like.default, (x,))
    f1 = graph.call_function(aten.fill_.Scalar, (e1, 1.0))
    f2 = graph.call_function(aten.fill_.Scalar, (e2, 2.0))
    graph.output((f1, f2))
    gm = GraphModule({}, graph)

    assert eliminate_common(gm, merge_shape_ops=True) == 0
    one, two = _run(gm, x=torch.zeros(4))
    assert torch.equal(one, torch.ones(4))
    assert torch.equal(two, torch.full((4,), 2.0))


def test_cse_keeps_reads_around_in_place_write() -> None:
    """同一输入被原地写的前后各读一次，键相同但值不同，不能合并。"""
    aten = torch.ops.aten
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_function(aten.clone.default, (x,))
    before = graph.call_function(aten.mul.Tensor, (y, 2.0))
    graph.call_function(aten.add_.Tensor, (y, 1.0))
    after = graph.call_function(aten.mul.Tensor, (y, 2.0))
    graph.output((before, after))
    gm = GraphModule({}, graph)

    assert eliminate_common(gm, merge_shape_ops=True) == 0
    got_before, got_after = _run(gm, x=torch.ones(4))
    assert torch.equal(got_before, torch.full((4,), 2.0))
    assert torch.equal(got_after, torch.full((4,), 4.0))


@pytest.mark.parametrize("split", [
    lambda b: b.split(2)[0],
    lambda b: b.unbind(0)[0],
], ids=["split", "unbind"])
def test_cse_keeps_clone_written_through_multi_output_view(split) -> None:
    """经多输出视图的元素原地写 clone，两个 clone 也不能合并。"""

    class WriteThroughView(torch.nn.Module):
        def forward(self, x):
            a = x.clone()
            b = x.clone()
            split(b).mul_(10.0)
            return a + 0, b + 0

    x = torch.ones(4, 2)
    gm = torch.export.export(WriteThroughView(), (x,), strict=True).module()
    expected = gm(x)

    assert eliminate_common(gm) == 0
    gm.recompile()
    for got, want in zip(gm(x), expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("produce,write", [
    (lambda x: torch.sort(x)[0], lambda v: v.mul_(10.0)),
    (lambda x: torch.max(x, 1)[0], lambda v: v.add_(5.0)),
], ids=["sort", "max_dim"])
def test_cse_keeps_multi_output_op_written_in_place(produce, write) -> None:
    """多输出算子经 getitem 取出的结果被原地写，两处同样的算子也不能合并。"""

    class WriteMultiOutput(torch.nn.Module):
        def forward(self, x):
            a = produce(x)
            b = produce(x)
            write(b)
            return a + 0, b + 0

    x = torch.randn(3, 4)
    gm = torch.export.export(WriteMultiOutput(), (x,), strict=True).module()
    expected = gm(x)

    assert eliminate_common(gm) == 0
    gm.recompile()
    for got, want in zip(gm(x), expected):
        assert torch.equal(got, want)


@pytest.mark.parametrize("wrap", [list, tuple], ids=["list", "tuple"])
def test_cse_keeps_clone_written_through_container_arg(wrap) -> None:
    """写参数是张量列表（如 `_foreach_mul_`），列表里的 clone 被写，两个 clone 也不能合并。"""

    class ForeachWrite(torch.nn.Module):
        def forward(self, x):
            a = x.clone()
            b = x.clone()
            torch._foreach_mul_(wrap([b]), 2.0)
            return a + 0, b + 0

    x = torch.ones(4)
    gm = torch.export.export(ForeachWrite(), (x,), strict=True).module()
    expected = gm(x)

    report = optimize_graph(gm, tile=False)

    assert report.merged == 0
    for got, want in zip(gm(x), expected):
        assert torch.equal(got, want)


def _two_const_chains_graph() -> GraphModule:
    """两条相同的常量 add 链分别乘上输入，再相加。"""
    graph = Graph()
    x = graph.placeholder("x")
    outs = []
    for _ in range(2):
        added = graph.call_function(torch.ops.aten.add.Tensor,
                                    (graph.get_attr("a"), graph.get_attr("b")))
        outs.append(graph.call_function(torch.ops.aten.mul.Tensor, (added, x)))
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, tuple(outs)))
    return GraphModule({"a": torch.tensor([1.0, 2.0]),
                        "b": torch.tensor([3.0, 4.0])}, graph)


def test_cse_merges_folded_constants() -> None:
    """两条相同的常量链各自折成常量后，CSE 把它们收成一个，输出不变。"""
    gm = _two_const_chains_graph()
    x = torch.tensor([2.0, 5.0])
    expected = _run(gm, x=x)

    assert fold_constants(gm) == 2
    # 两个折叠常量收成一个后，两处 mul 的参数也相同，跟着合并。
    assert eliminate_common(gm) == 2
    folded = [n for n in gm.graph.nodes
              if n.op == "get_attr" and n.target.startswith("_folded_")]
    assert len(folded) == 1
    assert torch.equal(_run(gm, x=x), expected)


def test_cse_keeps_model_params_with_equal_values() -> None:
    """模型参数值相同也不合并：参数的身份与放置不能按值合并。"""
    graph = Graph()
    x = graph.placeholder("x")
    w1 = graph.get_attr("w1")
    w2 = graph.get_attr("w2")
    graph.output(graph.call_function(torch.ops.aten.add.Tensor, (
        graph.call_function(torch.ops.aten.mul.Tensor, (x, w1)),
        graph.call_function(torch.ops.aten.mul.Tensor, (x, w2)))))
    gm = GraphModule({"w1": torch.ones(2), "w2": torch.ones(2)}, graph)

    assert eliminate_common(gm) == 0


def test_cse_shape_ops_follow_schema_alias() -> None:
    """表外的形状类（返回别名的 expand、select）默认不合并，打开开关后合并。"""
    aten = torch.ops.aten
    graph = Graph()
    x = graph.placeholder("x")
    outs = [graph.call_function(aten.expand.default, (x, [3, 2, 4]))
            for _ in range(2)]
    outs += [graph.call_function(aten.select.int, (x, 0, 1)) for _ in range(2)]
    graph.output(tuple(outs))
    gm = GraphModule({}, graph)
    x_val = torch.randn(2, 4)

    assert eliminate_common(gm) == 0
    assert eliminate_common(gm, merge_shape_ops=True) == 2
    gm.recompile()
    got = gm(x_val)
    assert torch.equal(got[1], x_val.expand(3, 2, 4))
    assert torch.equal(got[3], x_val[1])


def test_cse_keeps_distinct() -> None:
    """两处 mul 参数不同，都不动。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.placeholder("z")
    m1 = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    m2 = graph.call_function(torch.ops.aten.mul.Tensor, (x, z))
    s = graph.call_function(torch.ops.aten.add.Tensor, (m1, m2))
    graph.output(s)
    gm = GraphModule({}, graph)

    changed = eliminate_common(gm)

    assert changed == 0
    assert _targets(gm).count(torch.ops.aten.mul.Tensor) == 2


# ---------------------------------------------------------------- 融合

def test_fuse_unlisted_elementwise() -> None:
    """白名单外的逐元素组合（neg 后接 tanh）合并成一个节点，数值不变。"""
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(torch.ops.aten.tanh.default, (neg,))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([-1.0, 0.0, 0.5, 2.0])
    expected = torch.tanh(-x_val)

    changed = fuse_elementwise(gm)

    assert changed > 0, "逐元素链没有融合"
    assert len(_targets(gm)) == 1, _targets(gm)
    got = _run(gm, x=x_val)
    assert torch.allclose(got, expected)


def test_fuse_keeps_other_input_with_second_user() -> None:
    """消费者的另一个输入另有用户时仍然融合。

    那个输入作为合并节点的参数原样保留，不会被收进闭包，它的另一个
    用户拿到的值不变。只要求生产者只有一个用户。
    """
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    other = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    added = graph.call_function(torch.ops.aten.add.Tensor, (neg, other))
    out = graph.call_function(torch.ops.aten.add.Tensor, (added, other))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, -2.0, 3.0])
    y_val = torch.tensor([4.0, 5.0, -6.0])
    expected = (-x_val + x_val * y_val) + x_val * y_val

    changed = fuse_elementwise(gm)

    assert changed == 1, "另一个输入的第二个用户挡住了融合"
    # 合并算子是闭包，FX 生成代码时按模块找不回它，改用解释执行。
    from torch.fx.interpreter import Interpreter

    got = Interpreter(gm).run(x_val, y_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_blocked_by_second_user() -> None:
    """生产者另有用户时，那个用户拿到的仍是原值。

    `add` 同时用了 `neg` 和 `tanh(neg)`。`tanh` 只有一个用户所以被融合，
    闭包把两步都算进去，`add` 的另一个输入仍指向原来的 `neg`。
    """
    from torch.fx.interpreter import Interpreter

    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    t = graph.call_function(torch.ops.aten.tanh.default, (neg,))
    s = graph.call_function(torch.ops.aten.add.Tensor, (neg, t))
    graph.output(s)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([-1.0, 0.5, 2.0])
    expected = -x_val + torch.tanh(-x_val)

    fuse_elementwise(gm)

    got = Interpreter(gm).run(x_val)
    assert torch.allclose(got, expected), (got, expected)



def test_fuse_skips_side_effect() -> None:
    """生产者有副作用时不融合，哪怕它只有一个用户。"""
    graph = Graph()
    n = graph.get_attr("n")
    drawn = graph.call_function(torch.ops.aten.randint.low, (0, n, [4]))
    out = graph.call_function(torch.ops.aten.tanh.default, (drawn,))
    graph.output(out)
    gm = GraphModule({"n": torch.tensor(10)}, graph)

    changed = fuse_elementwise(gm)

    assert changed == 0
    assert torch.ops.aten.randint.low in _targets(gm)
    assert torch.ops.aten.tanh.default in _targets(gm)



def test_fuse_keeps_consumer_literal() -> None:
    """消费者的字面量参数在合并后不能丢，否则幂次会算错。"""
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(torch.ops.aten.pow.Tensor_Scalar, (neg, 3))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0, -1.0])
    expected = (-x_val) ** 3

    changed = fuse_elementwise(gm)

    assert changed > 0
    got = _run(gm, x=x_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_keeps_producer_kwargs() -> None:
    """生产者的关键字参数在合并后不能丢。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    added = graph.call_function(
        torch.ops.aten.add.Tensor, (x, y), {"alpha": 2})
    out = graph.call_function(torch.ops.aten.neg.default, (added,))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0])
    y_val = torch.tensor([3.0, 4.0])
    expected = -(x_val + 2 * y_val)

    changed = fuse_elementwise(gm)

    assert changed > 0
    got = _run(gm, x=x_val, y=y_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_keeps_consumer_kwargs() -> None:
    """消费者的关键字参数在合并后不能丢，否则幂次会算错。"""
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(
        torch.ops.aten.pow.Tensor_Scalar, (neg,), {"exponent": 3})
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0, -1.0])
    expected = (-x_val) ** 3

    changed = fuse_elementwise(gm)

    assert changed > 0
    got = _run(gm, x=x_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_binary_consumer() -> None:
    """消费者是二元逐元素算子时也融合，另一个输入原样保留。

    真实模型图里的逐元素算子几乎都是两个张量输入，只放行单输入
    消费者会让融合在真实图上一次都不触发。
    """
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    z = graph.placeholder("z")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(torch.ops.aten.add.Tensor, (neg, y))
    scaled = graph.call_function(torch.ops.aten.mul.Tensor, (out, z))
    graph.output(scaled)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, -2.0, 3.0])
    y_val = torch.tensor([4.0, 5.0, -6.0])
    z_val = torch.tensor([2.0, 0.5, -1.0])
    expected = (-x_val + y_val) * z_val

    changed = fuse_elementwise(gm)

    assert changed > 0, "二元逐元素消费者没有融合"
    got = _run(gm, x=x_val, y=y_val, z=z_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_binary_consumer_second_arg() -> None:
    """生产者结果在消费者的第二个参数上时，合并后顺序不能反。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(torch.ops.aten.sub.Tensor, (y, neg))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, -2.0, 3.0])
    y_val = torch.tensor([4.0, 5.0, -6.0])
    expected = y_val - (-x_val)

    changed = fuse_elementwise(gm)

    assert changed > 0
    got = _run(gm, x=x_val, y=y_val)
    assert torch.equal(got, expected), (got, expected)


def test_fuse_keeps_specialized_patterns() -> None:
    """RoPE、RMSNorm、门控激活这些专用融合认的节点不能提前合并。

    通用融合跑在专用融合之前。把 `mul` 融进 `add`、把 `rsqrt` 融进 `mul`、
    把 `silu` 融进下游，后面的专用融合就认不出整条链。
    """
    # RoPE 的尾部：两个 mul 汇进一个 add。
    graph = Graph()
    x = graph.placeholder("x")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")
    left = graph.call_function(torch.ops.aten.mul.Tensor, (x, cos))
    right = graph.call_function(torch.ops.aten.mul.Tensor, (x, sin))
    out = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output(out)
    gm = GraphModule({}, graph)

    assert fuse_elementwise(gm) == 0
    assert torch.ops.aten.mul.Tensor in _targets(gm)

    # 门控激活：linear 后接 silu，再乘上另一个投影。
    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    b = graph.placeholder("b")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(torch.ops.aten.silu.default, (proj,))
    gated = graph.call_function(torch.ops.aten.mul.Tensor, (act, b))
    graph.output(gated)
    gm = GraphModule({}, graph)

    assert fuse_elementwise(gm) == 0
    assert torch.ops.aten.silu.default in _targets(gm)


def test_fuse_silu_whose_producer_is_not_a_gate_projection() -> None:
    """真实图上 gate 投影的 silu 折进 linear，执行计划交给 A 路内核算。

    通用融合不碰 silu（GML 路折进 contraction 块，执行路径由
    fuse_linear_activation 折进 linear），切分传播之后每层的门控 silu 都被吸收。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from graph.partition import partition_graph
    from graph.spec_prop import propagate_specs
    from graph.strategy import llama_strategy
    from runtime.compile import causal_mask_of, export_annotated_graph

    cfg = LlamaConfig(
        vocab_size=320, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
        pad_token_id=0)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    pos = torch.arange(16).unsqueeze(0)
    gm = export_annotated_graph(model, 16, pos, dtype=torch.float32)
    mask = causal_mask_of(16, torch.float32)
    ids = torch.arange(16).unsqueeze(0)
    before = gm.forward(ids, mask, pos)

    partition_graph(gm)
    optimize_graph(gm)
    silus = [n for n in gm.graph.nodes
             if n.target is torch.ops.aten.silu.default]
    assert len(silus) == cfg.num_hidden_layers, "通用融合不能提前吃掉 silu"
    propagate_specs(gm, llama_strategy(
        num_dpus=2, num_stages=1, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=320, num_layers=1))

    assert fuse_linear_activation(gm) == cfg.num_hidden_layers, \
        "真实图上的门控 silu 没有折进 linear"
    assert not any(n.target is torch.ops.aten.silu.default
                   for n in gm.graph.nodes), "图上仍有独立 silu 节点"
    fused_linears = [n for n in gm.graph.nodes
                     if n.target is torch.ops.aten.linear.default
                     and FUSED_TAIL_META_KEY in n.meta]
    assert len(fused_linears) == cfg.num_hidden_layers
    # 折入前数值与 torch 一致；折入后 silu 由 A 路内核补上，图形求值
    # 不再含激活，数值对齐由 test_exec_plan_gen 的执行计划用例守着。
    _ = before


def test_fuse_linear_activation_absorbs_single_user_gate_chain() -> None:
    """linear -> silu -> mul：silu 折进 linear，标记记下激活种类。

    折掉后 linear 的 target 不变，执行计划按 FUSED_TAIL_META_KEY 把
    activation 下发进编译请求。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    b = graph.placeholder("b")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(torch.ops.aten.silu.default, (proj,))
    gated = graph.call_function(torch.ops.aten.mul.Tensor, (act, b))
    graph.output(gated)
    gm = GraphModule({}, graph)

    assert fuse_linear_activation(gm) == 1
    targets = _targets(gm)
    assert torch.ops.aten.silu.default not in targets, targets
    tail = proj.meta[FUSED_TAIL_META_KEY]
    assert tail.activation == "silu" and tail.pool is None
    assert gated.args[0] is proj, "mul 应直接读 linear 的结果"


def test_fuse_linear_activation_skips_multi_user_projection() -> None:
    """linear 的结果有第二个用户时 silu 不能折。

    折掉 silu 会把激活作用到 linear 的所有读者上，另一个用户拿到的值就
    变了。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(torch.ops.aten.silu.default, (proj,))
    other = graph.call_function(torch.ops.aten.tanh.default, (proj,))
    out = graph.call_function(torch.ops.aten.add.Tensor, (act, other))
    graph.output(out)
    gm = GraphModule({}, graph)

    assert fuse_linear_activation(gm) == 0
    assert torch.ops.aten.silu.default in _targets(gm)
    assert FUSED_TAIL_META_KEY not in proj.meta


def test_fuse_mul_add_that_is_not_rope() -> None:
    """与 RoPE 无关的 mul 后接 add 按性质融合。

    RoPE 的特征是两个 mul 汇入同一个 add。只有一个 mul 再加一个偏置
    不是这条链，不该因为算子名字是 mul 和 add 就被排除。
    """
    graph = Graph()
    x = graph.placeholder("x")
    cos = graph.placeholder("cos")
    bias = graph.placeholder("bias")
    scaled = graph.call_function(torch.ops.aten.mul.Tensor, (x, cos))
    out = graph.call_function(torch.ops.aten.add.Tensor, (scaled, bias))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0, 3.0, 4.0])
    cos_val = torch.tensor([0.5, 0.5, 0.5, 0.5])
    bias_val = torch.tensor([10.0, 20.0, 30.0, 40.0])
    expected = x_val * cos_val + bias_val

    changed = fuse_elementwise(gm)

    assert changed == 1, "无关的 mul 后接 add 没有融合"
    assert torch.ops.aten.add.Tensor not in _targets(gm)
    assert torch.allclose(_run(gm, x=x_val, cos=cos_val, bias=bias_val), expected)


def test_fuse_skips_reduction() -> None:
    """消费者含归约（形状变化）时不融合。"""
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    reduced = graph.call_function(
        torch.ops.aten.mean.dim, (neg, [-1], True))
    graph.output(reduced)
    gm = GraphModule({}, graph)

    changed = fuse_elementwise(gm)

    assert changed == 0
    assert torch.ops.aten.mean.dim in _targets(gm)


# ---------------------------------------------------------------- DCE

def test_dce_drops_unused() -> None:
    """结果没人用的纯算子链被删除，输出值不变。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    out = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([1.0, 2.0])
    y_val = torch.tensor([3.0, 4.0])

    changed = eliminate_dead(gm)

    assert changed > 0, "无用户的 mul 没有被删除"
    assert torch.ops.aten.mul.Tensor not in _targets(gm), _targets(gm)
    got = _run(gm, x=x_val, y=y_val)
    assert torch.equal(got, x_val + y_val)


def test_dce_drops_assert_metadata() -> None:
    """导出器插入的张量元数据断言没有输出、也没有副作用，应当删除。

    真实模型图里这类节点每层都有一份，是图上唯一确定的死代码。
    它不在纯算子名单里，但 schema 上没有任何写副作用，结果也没人用。
    """
    graph = Graph()
    x = graph.placeholder("x")
    graph.call_function(
        torch.ops.aten._assert_tensor_metadata.default, (x,))
    graph.output(x)
    gm = GraphModule({}, graph)

    changed = eliminate_dead(gm)

    assert changed > 0, "没有输出的元数据断言没有被删除"
    assert torch.ops.aten._assert_tensor_metadata.default not in _targets(gm)
    got = _run(gm, x=torch.tensor([1.0, 2.0]))
    assert torch.equal(got, torch.tensor([1.0, 2.0]))


def test_dce_keeps_output() -> None:
    """同型算子连到输出时保留。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    out = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    graph.output(out)
    gm = GraphModule({}, graph)

    changed = eliminate_dead(gm)

    assert changed == 0
    assert torch.ops.aten.mul.Tensor in _targets(gm)


def test_dce_keeps_side_effect() -> None:
    """有副作用的算子即使结果没人用也不删。"""
    graph = Graph()
    x = graph.placeholder("x")
    graph.call_function(torch.ops.aten.randint.low, (0, 10, [4]))
    graph.output(x)
    gm = GraphModule({}, graph)

    changed = eliminate_dead(gm)

    assert changed == 0
    assert torch.ops.aten.randint.low in _targets(gm)


def test_real_llama_graph_dce_removes_export_asserts() -> None:
    """真实导出图上的默认 DCE：元数据断言与无用户 arange 被删。

    与 `test_real_llama_graph_has_nothing_to_fold` 同一建图方式。每层导出的
    `_assert_tensor_metadata` 没有输出、也没有写副作用，按设计 4.5 的例外
    当死代码删；这条钉住图层 DCE 在默认通路上确实有作用对象。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval()
    position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, 16, position_ids, dtype=torch.float32)
    asserts = sum(1 for node in gm.graph.nodes
                  if node.target is torch.ops.aten._assert_tensor_metadata.default)
    assert asserts > 0, "导出图里没有元数据断言，用例失去意义"

    report = optimize_graph(gm)

    assert report.removed >= asserts, f"只删了 {report.removed} 个节点"
    assert torch.ops.aten._assert_tensor_metadata.default not in _targets(gm)


@pytest.mark.parametrize("target, args", [
    (torch.ops.aten.rand_like.default, "x"),
    (torch.ops.aten.randn_like.default, "x"),
    (torch.ops.aten.randperm.default, 4),
    (torch.ops.aten.native_dropout.default, "x"),
    (torch.ops.aten._assert_scalar.default, "x"),
    (torch.ops.aten.sym_constrain_range.default, "x"),
])
def test_dce_keeps_unlisted_side_effect(target, args) -> None:
    """名单之外的随机算子与运行期断言同样是副作用，结果没人用也保留。"""
    graph = Graph()
    x = graph.placeholder("x")
    arg = x if args == "x" else args
    graph.call_function(target, (arg,))
    graph.output(x)
    gm = GraphModule({}, graph)

    changed = eliminate_dead(gm)

    assert changed == 0, f"{target} 被当成死代码删除"
    assert target in _targets(gm)


# ---------------------------------------------------------------- 分块

def test_tile_splits_oversized() -> None:
    """输出超过 WRAM 预算的 linear 拆成多个子节点，每个都在预算内。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    w_val = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    expected = torch.nn.functional.linear(x_val, w_val)
    _shape(gm, x_val, w_val)
    # 输出 32*8*4 = 1024 字节，预算给 256，至少要拆成 4 块。
    budget = 256

    changed = tile_graph(gm, wram_bytes=budget)

    assert changed > 0, "超预算的 linear 没有拆分"
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    assert len(linears) >= 2, _targets(gm)
    for node in linears:
        out = node.meta.get("val")
        # 子块沿第一维切，行数不超过预算能容纳的行数。
        assert out is None or out.shape[0] * w_val.shape[0] * 4 <= budget, node
    got = _run(gm, x=x_val, w=w_val)
    assert torch.allclose(got, expected)



def test_tile_matches_cost_model() -> None:
    """选出的切分与评估模型手算一致：2 的幂里块数最少且每块不超预算的那个。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)
    rows, in_features, out_features = 32, 16, 8
    x_val = torch.ones(rows, in_features)
    w_val = torch.ones(out_features, in_features)
    _shape(gm, x_val, w_val)
    budget = 256
    # 工作集按输入、权重、输出三块之和算。M=2、N=4、K=8 时单块
    # 2*8+4*8+2*4 = 88 字节，共 16*2*2 = 64 块，是块数最少的合法切分。
    changed = tile_graph(gm, wram_bytes=budget)

    assert changed == 1
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    assert len(linears) == 64
    for node in linears:
        assert tuple(node.meta["val"].shape) == (2, 4), node.meta["val"].shape



def test_tile_counts_weight_and_input_in_working_set() -> None:
    """子块的工作集按输入、权重、输出三块之和计算，与算子层同一口径。

    只比输出字节时，沿输出特征维切开只缩小了输出，权重的另一维原样保留。
    这里输出只有 64 字节，但权重有 512 字节，预算 128 时只看输出不会切，
    三块相加必须切；切完每个子块的三块之和都不能超过预算。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    rows, in_features, out_features = 2, 64, 8
    torch.manual_seed(0)
    x_val = torch.randn(rows, in_features)
    w_val = torch.randn(out_features, in_features)
    _shape(gm, x_val, w_val)
    # 输出 2*8*4 = 64 字节，已在预算 128 内；权重 8*64*4 = 2048 字节，超预算。
    budget = 128

    changed = tile_graph(gm, wram_bytes=budget)

    assert changed == 1, "权重超预算但只比输出字节，没有切"
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    assert len(linears) > 1
    for linear in linears:
        total = 0
        for arg in linear.args:
            if isinstance(arg, torch.fx.Node):
                val = arg.meta.get("val")
                if val is not None:
                    total += val.numel() * 4
        out = linear.meta.get("val")
        total += out.numel() * 4
        assert total <= budget, f"子块三块合计 {total} 字节，超过预算 {budget}"
    got = _run(gm, x=x_val, w=w_val)
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val))


def test_tile_keeps_when_one_row_exceeds_budget() -> None:
    """单行输出已超过预算时不拆：拆出来的每一块仍然超预算。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)
    x_val = torch.ones(8, 4)
    w_val = torch.ones(4, 4)
    _shape(gm, x_val, w_val)
    # 单行输出 4*4 = 16 字节，预算只有 8，任何切分都压不进去。
    changed = tile_graph(gm, wram_bytes=8)

    assert changed == 0
    assert _targets(gm).count(torch.ops.aten.linear.default) == 1


def test_tile_splits_output_features_when_one_row_exceeds_budget() -> None:
    """单行超过预算时改切输出特征维：权重按列切开，结果沿最后一维拼回。

    真实模型的 batch 维是 1，输出第一维切不开，但一个 linear 的输出
    有上百万字节，不切就整块超预算。沿特征维切与整块逐元素相等。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    rows, in_features, out_features = 1, 8, 32
    torch.manual_seed(0)
    x_val = torch.randn(rows, in_features)
    w_val = torch.randn(out_features, in_features)
    _shape(gm, x_val, w_val)
    # 整行 32*4 = 128 字节，预算 64，所以每块最多 16 个输出特征。
    budget = 64

    changed = tile_graph(gm, wram_bytes=budget)

    assert changed == 1, "单行超预算时没有改切输出特征维"
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    # 只切输出特征会留下整块输入，三块之和仍超预算。M=1、N=2、K=4
    # 时单块 1*4+2*4+1*2 = 14 个元素，32 块，每块都在预算内。
    assert len(linears) == 32
    for linear in linears:
        assert tuple(linear.meta["val"].shape) == (1, 2), linear.meta["val"].shape
    got = _run(gm, x=x_val, w=w_val)
    # 分段累加的顺序与整块不同，差在浮点舍入范围内。
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val))


def test_tile_splits_non_power_of_two_extent() -> None:
    """输出特征数不是 2 的幂时也切：最后一块不满，数值仍与整块相等。

    词表投影的宽度是 32000，不是 2 的幂，跳过它就留下全图最大的一块。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(1, 4, 8)
    w_val = torch.randn(24, 8)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=64)

    assert changed == 1
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    # 归约维也要切：M=1、N=2、K=4，4 行 * 12 列 * 2 段 = 96 块。
    assert len(linears) == 96
    got = _run(gm, x=x_val, w=w_val)
    # 分段累加的顺序与整块不同，差在浮点舍入范围内。
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val))


def test_tile_picks_the_dim_with_fewer_blocks() -> None:
    """两维都能压进预算时选块数更少的那一维。

    输出 8 行 6 列、每元素 4 字节、预算 64。沿第一维每块 2 行要切 4 块，
    沿输出特征维每块 2 个特征只要切 3 块。评估模型要块数最少，所以选特征维。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(8, 4)
    w_val = torch.randn(6, 4)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=64)

    assert changed == 1
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    # 只比输出字节时沿特征维切 3 块即可，但那样输入整块保留，三块之和超预算。
    # M=1、N=2、K=4 时 8*3 = 24 块，是三块之和不超预算里块数最少的。
    assert len(linears) == 24, f"块数不是最少的 24，而是 {len(linears)}"
    got = _run(gm, x=x_val, w=w_val)
    # 分段累加的顺序与整块不同，差在浮点舍入范围内。
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val))


def test_tile_splits_matmul_when_one_row_exceeds_budget() -> None:
    """matmul 单行输出超预算时切第二个操作数的输出维，数值与整块相等。

    matmul 的输出第一维就是第一个操作数的行数。batch 为 1 时这一维切不开，
    但第二个操作数的最后一维决定输出宽度，切它再沿输出最后一维拼回，
    与整块逐元素相等。不切就让超预算的 matmul 原样留在图里。
    """
    graph = Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    node = graph.call_function(torch.ops.aten.matmul.default, (a, b))
    graph.output(node)
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    a_val = torch.randn(8, 16)
    b_val = torch.randn(16, 256)
    _shape(gm, a_val, b_val)
    # 输出 8*256*4 = 8192 字节，单行 1024 字节，预算 256 容纳不下任何一行。
    budget = 256

    changed = tile_graph(gm, wram_bytes=budget)

    assert changed == 1, "单行超预算的 matmul 没有拆分"
    matmuls = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.matmul.default]
    assert len(matmuls) > 1
    for matmul in matmuls:
        # 子块的工作集是输入、权重、输出三块之和，不是输出一个。
        total = matmul.meta["val"].numel()
        for arg in matmul.args:
            if isinstance(arg, torch.fx.Node) and arg.meta.get("val") is not None:
                total += arg.meta["val"].numel()
        assert total * 4 <= budget, matmul.meta["val"].shape
    got = _run(gm, a=a_val, b=b_val)
    assert torch.allclose(got, torch.matmul(a_val, b_val), atol=1e-4)


def test_tile_slices_bias_with_output_features() -> None:
    """沿特征维切 linear 时，bias 必须按同一段切开。

    bias 与输出特征一一对应。权重切成 4 列一段而 bias 仍是整段时，
    每个子块的输出列数与 bias 长度对不上，执行直接报形状不匹配。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    bias = graph.placeholder("bias")
    node = graph.call_function(
        torch.ops.aten.linear.default, (x, w, bias))
    graph.output(node)
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(1, 4)
    w_val = torch.randn(12, 4)
    bias_val = torch.randn(12)
    for n, val in ((x, x_val), (w, w_val), (bias, bias_val),
                   (node, torch.empty(1, 12))):
        n.meta["val"] = val
    expected = torch.nn.functional.linear(x_val, w_val, bias_val)

    # 预算 16 时最小的子块也有 28 字节，没有任何切法压得进，所以不切。
    # 放到 32 才有合法切分：M=1、N=1、K=2，单块 24 字节，共 24 块。
    assert tile_graph(gm, wram_bytes=16) == 0
    changed = tile_graph(gm, wram_bytes=32)

    assert changed == 1
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    assert len(linears) == 24
    # bias 与输出特征对齐，按同一段切开，且只加一次。
    for linear in linears:
        assert linear.args[2] is None, "bias 不应进到 K 维的子块里"
    got = _run(gm, x=x_val, w=w_val, bias=bias_val)
    assert torch.allclose(got, expected)


def test_tile_keeps_within_budget() -> None:
    """预算内的 linear 不拆。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)
    x_val = torch.ones(2, 4)
    w_val = torch.ones(2, 4)

    changed = tile_graph(gm, wram_bytes=1024)

    assert changed == 0
    assert _targets(gm).count(torch.ops.aten.linear.default) == 1
    got = _run(gm, x=x_val, w=w_val)
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val))



def _slice_start(node) -> tuple[int, int]:
    """从切片节点读出第一维与最后一维的起点。没切的维起点是 0。"""
    starts = {}
    cur = node
    # 输入提升插在切片之后，起点要往回穿过它才读得到。
    while cur.op == "call_function" and cur.target is torch.ops.aten._to_copy.default:
        cur = cur.args[0]
    while cur.op == "call_function" and cur.target is torch.ops.aten.slice.Tensor:
        dim = cur.args[1]
        starts[dim] = cur.args[2]
        cur = cur.args[0]
    rank = node.meta["val"].ndim
    return starts.get(0, 0), starts.get(rank - 1, 0)


def test_tile_float16_reduction_matches_whole() -> None:
    """float16 下 K 维切成多段再相加，结果仍与一次矩阵乘对齐。

    K=4096 在预算 4096 下切成 32 段。各段在 float16 上逐段相加，
    舍入比一次矩阵乘大，超过 allclose 默认容差。
    分段累加必须在更高精度上做，写回时再落到原 dtype。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    rows, in_features, out_features = 8, 4096, 16
    torch.manual_seed(0)
    x_val = torch.randn(rows, in_features, dtype=torch.float16)
    # 权重按初始化量级缩放：全 1 方差的权重会让输出大到把舍入差放大。
    w_val = torch.randn(out_features, in_features, dtype=torch.float16) * 0.02
    _shape(gm, x_val, w_val)

    # 预算 4096 时 K 整段放不下，只能把 K 切成多段再相加。
    changed = tile_graph(gm, wram_bytes=4096)

    assert changed == 1
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    # K 维被切成多段：段数大于 1 才会发生跨段相加。
    assert len(linears) > out_features, len(linears)
    got = _run(gm, x=x_val, w=w_val)
    # 基准按图里每个子块的切片在 float32 上重算，再按输出位置相加，
    # 与执行路径的 linear 内核同一口径。直接比 torch 的 float16 linear
    # 会把 torch 自身的归约舍入算进来。
    expected = torch.zeros(rows, out_features, dtype=torch.float32)
    covered = torch.zeros(rows, out_features, dtype=torch.bool)
    for linear in linears:
        x_node, w_node = linear.args[:2]
        x_slice, w_slice = x_node.meta["val"], w_node.meta["val"]
        r, rk = _slice_start(x_node)
        c, ck = _slice_start(w_node)
        part = torch.nn.functional.linear(
            x_val[r:r + x_slice.shape[0], rk:rk + x_slice.shape[-1]].float(),
            w_val[c:c + w_slice.shape[0], ck:ck + w_slice.shape[-1]].float())
        expected[r:r + part.shape[0], c:c + part.shape[1]] += part
        covered[r:r + part.shape[0], c:c + part.shape[1]] = True
    assert covered.all(), "子块没有覆盖全部输出"
    # 最终写回原 dtype，比较时基准也落到 float16，只留这一次舍入。
    assert torch.allclose(got, expected.half()), (got.float() - expected.half().float()).abs().max()


def test_tile_reports_unknown_dtype() -> None:
    """不认识的 dtype 报契约里的错，而不是 KeyError。"""
    import pytest

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    gm = GraphModule({}, graph)
    out.meta["val"] = torch.empty(32, 8, dtype=torch.bfloat16)

    with pytest.raises(ValueError):
        tile_graph(gm, wram_bytes=64)


def test_tile_skips_elementwise() -> None:
    """逐元素算子的工作集等于输出本身，拆了不减，不动。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    graph.call_function(torch.ops.aten.add.Tensor, (x, y))
    graph.output(next(n for n in graph.nodes if n.op == "call_function"))
    gm = GraphModule({}, graph)

    changed = tile_graph(gm, wram_bytes=4)

    assert changed == 0
    assert torch.ops.aten.add.Tensor in _targets(gm)


def _working_set(node) -> int:
    """子块实际驻留的字节：每个参数与输出按各自的 dtype 计。"""
    total = node.meta["val"].numel() * node.meta["val"].element_size()
    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and arg.meta.get("val") is not None:
            val = arg.meta["val"]
            total += val.numel() * val.element_size()
    return total


def test_tile_keeps_tail_rows() -> None:
    """行数不被块长整除时，尾部的行也要算进去，输出形状与原图一致。

    7 行、块长 2 时只生成 3 组会丢掉最后一行，结果从 (7, 64) 变成 (6, 64)。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(7, 64)
    w_val = torch.randn(64, 64)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=1024)

    assert changed == 1, "超预算的 linear 没有切"
    got = _run(gm, x=x_val, w=w_val)
    expected = torch.nn.functional.linear(x_val, w_val)
    assert tuple(got.shape) == tuple(expected.shape), (got.shape, expected.shape)
    # K 维分段累加的顺序与整块不同，差在浮点舍入范围内。
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


def test_tile_slices_multi_leading_dims() -> None:
    """输入有两个前导维时，行切片要按各维换算，不能在一个维上累加出负长度。

    (2, 8, 8) 的输入沿行切时，第 4 组的起点越过第二维的长度 8，
    切片会构造出负维度而抛异常。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(2, 8, 8)
    w_val = torch.randn(32, 8)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=512)

    assert changed == 1, "超预算的 linear 没有切"
    got = _run(gm, x=x_val, w=w_val)
    expected = torch.nn.functional.linear(x_val, w_val)
    assert tuple(got.shape) == tuple(expected.shape), (got.shape, expected.shape)
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("x_shape, budget", [((2, 3, 4), 192), ((4, 2, 4), 300)])
def test_tile_rows_cross_short_inner_dim(x_shape, budget) -> None:
    """内层前导维比块长短时，行区间要跨多个批次切开，结果与原图同形同值。

    (2, 3, 4) 选出 M=4 后曾抛「拆不开」；(4, 2, 4) 曾静默输出 (2, 1, 4)。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val = torch.randn(*x_shape)
    w_val = torch.randn(4, 4)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=budget)

    assert changed == 1, "超预算的 linear 没有切"
    got = _run(gm, x=x_val, w=w_val)
    expected = torch.nn.functional.linear(x_val, w_val)
    assert tuple(got.shape) == tuple(expected.shape), (got.shape, expected.shape)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_tile_budget_counts_widened_dtype() -> None:
    """float16 先提升到 float32 再算，预算要按提升后的 dtype 计字节。

    形状取 8×512 激活乘 1024×512 权重，块数在上限之内。按 2 字节算会选出
    更大的子块，两个输入实际是 float32，三块合计超过 65536 的预算。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    rows, in_features, out_features = 8, 512, 1024
    x_val = torch.ones(rows, in_features, dtype=torch.float16)
    w_val = torch.ones(out_features, in_features, dtype=torch.float16)
    _shape(gm, x_val, w_val)

    changed = tile_graph(gm, wram_bytes=65536)

    assert changed == 1, "超预算却没有切"
    for linear in (n for n in gm.graph.nodes
                   if n.target is torch.ops.aten.linear.default):
        total = _working_set(linear)
        assert total <= 65536, f"子块实际驻留 {total} 字节，超过预算 65536"


# ---------------------------------------------------------------- 入口

def test_optimize_graph_runs_all_five() -> None:
    """入口按顺序跑完五类，常量链折叠、重复合并、死节点清除同时发生。"""
    graph = Graph()
    a = graph.get_attr("a")
    b = graph.get_attr("b")
    added = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    x = graph.placeholder("x")
    m1 = graph.call_function(torch.ops.aten.mul.Tensor, (added, x))
    m2 = graph.call_function(torch.ops.aten.mul.Tensor, (added, x))
    graph.call_function(torch.ops.aten.neg.default, (x,))
    s = graph.call_function(torch.ops.aten.add.Tensor, (m1, m2))
    graph.output(s)
    a_val = torch.tensor([1.0, 2.0])
    b_val = torch.tensor([3.0, 4.0])
    gm = GraphModule({"a": a_val, "b": b_val}, graph)
    x_val = torch.tensor([2.0, 2.0])
    expected = (a_val + b_val) * x_val * 2

    report = optimize_graph(gm)

    assert report.folded > 0
    assert report.merged > 0
    assert report.removed > 0
    # 无用户的 neg 被删，两处相同的 mul 被 CSE 收成一处。
    assert torch.ops.aten.neg.default not in _targets(gm)
    assert _targets(gm).count(torch.ops.aten.mul.Tensor) <= 1
    got = _run(gm, x=x_val)
    assert torch.equal(got, expected)


# ---------------------------------------------------------------- 接入

def test_tiling_keeps_device_annotations() -> None:
    """分块新增的节点必须带设备标注，否则切分传播直接报错。

    分区在优化之前跑，分块后插入的切片与拼接节点没有标注。
    """
    from graph.partition import partition_graph

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule(torch.nn.Module(), graph)
    x_val = torch.ones(1, 4, 8)
    w_val = torch.ones(32, 8)
    gm.register_buffer("w", w_val)
    _shape(gm, x_val, w_val)
    partition_graph(gm)

    optimize_graph(gm, wram_bytes=64)

    missing = [n.name for n in gm.graph.nodes if "device" not in n.meta]
    assert not missing, f"分块后这些节点没有设备标注: {missing}"
    # 切出来的计算节点要带输出形状，切分传播靠它推导布局。
    unshaped = [n.name for n in gm.graph.nodes
                if n.op == "call_function" and "val" not in n.meta]
    assert not unshaped, f"分块后这些节点没有输出形状: {unshaped}"
    # 切片是设备算子，即便切的是还没标注的权重也标成设备，
    # 这样切分传播才能按它的用户反推权重的切分。
    for node in gm.graph.nodes:
        if node.target is torch.ops.aten.slice.Tensor:
            assert node.meta["device"] == "dpu"


def test_folded_and_fused_nodes_keep_device_annotations() -> None:
    """折叠与融合新建的节点必须带设备标注，否则切分传播直接报错。

    分区在优化之前跑。常量折叠产出的 get_attr 和融合产出的合并节点
    原先只写了 val，propagate_specs 在入口处因缺标注抛 ValueError，
    默认执行路径一旦真触发这两个 Pass 就会编译失败。
    """
    from graph.partition import partition_graph
    from graph.spec_prop import propagate_specs
    from graph.strategy import ShardStrategy

    # 融合：neg 后接 tanh，分区后再优化。
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    out = graph.call_function(torch.ops.aten.tanh.default, (neg,))
    graph.output(out)
    gm = GraphModule(torch.nn.Module(), graph)
    value = torch.empty(4)
    for node in (x, neg, out):
        node.meta["val"] = value
    partition_graph(gm)

    report = optimize_graph(gm)

    assert report.fused > 0
    propagate_specs(gm, ShardStrategy(name="a", num_dpus=1))

    # 折叠：两个常量相加后再乘输入。
    graph = Graph()
    root = torch.nn.Module()
    root.register_buffer("a", torch.tensor([1.0, 2.0]))
    root.register_buffer("b", torch.tensor([3.0, 4.0]))
    a = graph.get_attr("a")
    b = graph.get_attr("b")
    added = graph.call_function(torch.ops.aten.add.Tensor, (a, b))
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.mul.Tensor, (added, x))
    graph.output(out)
    gm = GraphModule(root, graph)
    for node, val in ((a, root.a), (b, root.b), (added, torch.empty(2)),
                      (x, torch.empty(2)), (out, torch.empty(2))):
        node.meta["val"] = val
    partition_graph(gm)

    report = optimize_graph(gm)

    assert report.folded > 0
    propagate_specs(gm, ShardStrategy(name="a", num_dpus=1))


def test_compile_optimizes_before_propagating_specs() -> None:
    """执行路径在切分传播之前跑优化。

    不跑完整编译（那依赖 Llama 的权重命名），只核对两步的源码顺序：
    优化必须夹在导出与传播之间，传播才能看到优化后的图。
    """
    import inspect

    import runtime.compile as compile_mod

    src = inspect.getsource(compile_mod.compile_llama2)
    opt = src.index("optimize_graph(prefill_gm")
    prop = src.index("propagate_specs(prefill_gm")
    assert opt < prop, "优化排在了切分传播之后"


def test_fuse_for_gml_runs_optimize_first(monkeypatch) -> None:
    """GML 路在六个专用融合之前跑优化，并且关掉分块。"""
    import gml_bridge.export as export_mod

    order = []
    seen = {}

    def _opt(gm, **kw):
        seen.update(kw)
        order.append("opt")

    monkeypatch.setattr(export_mod, "optimize_graph", _opt)
    for name in ("fuse_rope", "fuse_for_pim", "fuse_graph",
                 "split_attention_heads", "insert_kv_dma_and_split",
                 "insert_dynamic_scaling"):
        monkeypatch.setattr(
            export_mod, name,
            lambda *a, _n=name, **k: order.append(_n) or _stub(_n))

    from contracts.unified_ir import STAGE_PARTITIONED, mark_stage

    graph = Graph()
    x = graph.placeholder("x")
    graph.output(x)
    gm = GraphModule({}, graph)
    mark_stage(gm, STAGE_PARTITIONED)
    export_mod.fuse_for_gml(gm)
    assert order[0] == "opt", order
    assert "fuse_rope" in order
    # 夹具无论是否切块都落在预算内，只能直接断言传下去的开关。
    assert seen.get("tile") is False, seen


def test_execution_path_tiles_by_default() -> None:
    """执行路径默认分块，不需要打开任何开关。

    超预算的 linear 走 optimize_graph 的默认参数，必须被拆成多个预算内的
    子节点。GML 路是唯一的例外，见 test_fuse_for_gml_runs_optimize_first。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    node = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(node)
    gm = GraphModule({}, graph)
    x_val = torch.ones(1, 4)
    w_val = torch.ones(32, 4)
    for n, val in ((x, x_val), (w, w_val), (node, torch.empty(1, 32))):
        n.meta["val"] = val
    expected = torch.nn.functional.linear(x_val, w_val)

    report = optimize_graph(gm, wram_bytes=64)

    assert report.tiled > 0, "执行路径没有切超预算的 linear"
    linears = [n for n in gm.graph.nodes
               if n.target is torch.ops.aten.linear.default]
    assert len(linears) > 1, "linear 仍是一整块"
    assert torch.equal(_run(gm, x=x_val, w=w_val), expected)


def _stub(name):
    if name == "fuse_for_pim":
        return type("R", (), {"total": 0})()
    if name == "split_attention_heads":
        return type("R", (), {"heads": 0})()
    if name == "insert_dynamic_scaling":
        return type("R", (), {"inserted": 0})()
    return 0


def _row_weight_graph():
    """行切权重的 linear：权重 (8, 64) 沿 K 切在两台上。"""
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("down_proj_weight")
    lin = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(lin)
    mod = torch.nn.Module()
    mod.register_buffer("down_proj_weight", torch.empty(8, 64, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 4, 64, dtype=torch.float16)
    w.meta["val"] = torch.empty(8, 64, dtype=torch.float16)
    lin.meta["val"] = torch.empty(1, 4, 8, dtype=torch.float16)
    return gm


def _real_linear_graph(m: int, n: int, k: int):
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("weight")
    lin = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(lin)
    mod = torch.nn.Module()
    mod.register_buffer("weight", torch.empty(n, k, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(m, k, dtype=torch.float16)
    w.meta["val"] = torch.empty(n, k, dtype=torch.float16)
    lin.meta["val"] = torch.empty(m, n, dtype=torch.float16)
    return gm


@pytest.mark.parametrize("n, k", [
    (4096, 4096),    # q_proj
    (11008, 4096),   # gate_proj
    (4096, 11008),   # down_proj
    (32000, 4096),   # lm_head
])
@pytest.mark.parametrize("m", [1, 16])
def test_tile_leaves_real_projections_whole(m, n, k) -> None:
    """真实 7B 投影在默认 65536 下要切成数千块，超过块数上限，图层不切。

    这类算子由算子层 A 路的 tt.dot 分块压进预算。图层切成数千个子节点会让
    通信计划与执行计划的规模失控。
    """
    gm = _real_linear_graph(m, n, k)
    assert tile_graph(gm, wram_bytes=65536, tp_width=2) == 0
    subs = [x for x in gm.graph.nodes if x.target is torch.ops.aten.linear.default]
    assert len(subs) == 1


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("n, k", [(64, 64), (128, 64), (64, 128)])
def test_tile_leaves_small_model_projections_whole(m, n, k) -> None:
    """回归用小模型（hidden 64、intermediate 128、vocab 128）在默认 65536 下都在预算内，不切。

    与 7B 投影超过块数上限一起，构成图层分块在默认通路上改动数为 0 的实测依据。
    """
    gm = _real_linear_graph(m, n, k)
    assert tile_graph(gm, wram_bytes=65536, tp_width=2) == 0


def _gate_linear_graph(act=torch.ops.aten.silu.default, cast_dtype=None):
    """4x64 输入、128 输出的 linear 后接激活，模拟门控投影。

    `cast_dtype` 不为空时在 linear 与激活之间夹一层类型转换。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("weight")
    lin = graph.call_function(torch.ops.aten.linear.default, (x, w))
    head = lin
    if cast_dtype is not None:
        head = graph.call_function(
            torch.ops.aten._to_copy.default, (lin,), {"dtype": cast_dtype})
    out = graph.call_function(act, (head,))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("weight", torch.empty(128, 64, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(4, 64, dtype=torch.float16)
    w.meta["val"] = torch.empty(128, 64, dtype=torch.float16)
    lin.meta["val"] = torch.empty(4, 128, dtype=torch.float16)
    out_dtype = cast_dtype or torch.float16
    if cast_dtype is not None:
        head.meta["val"] = torch.empty(4, 128, dtype=out_dtype)
    out.meta["val"] = torch.empty(4, 128, dtype=out_dtype)
    return gm


@pytest.mark.parametrize("wram", [2048, 1024])
def test_tile_keeps_k_whole_for_tail_activation_linear(wram) -> None:
    """后接表内激活的 linear 只沿 M、N 切，K 不切，激活随后仍折进每个子块。

    K 段相加后激活要算在和上，逐段算不对，所以这类 linear 的 K 留给 A 路分块。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm = _gate_linear_graph()
    optimize_graph(gm, wram_bytes=wram)
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert all(n.args[0].meta["val"].shape[-1] == 64 for n in subs), "K 被切开了"
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.silu.default not in _targets(gm)
    assert all(FUSED_TAIL_META_KEY in n.meta for n in subs)


def test_tile_keeps_k_whole_when_cast_sits_before_tail_activation() -> None:
    """linear 与激活之间夹一层浮点转换时同样不切 K，融合认的就是这种形态。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm = _gate_linear_graph(cast_dtype=torch.float32)
    report = optimize_graph(gm, wram_bytes=2048)
    assert report.tiled == 1
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert all(n.args[0].meta["val"].shape[-1] == 64 for n in subs), "K 被切开了"
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.silu.default not in _targets(gm)
    assert all(FUSED_TAIL_META_KEY in n.meta for n in subs)


def test_tile_still_splits_k_when_cast_feeds_non_tail_op() -> None:
    """浮点转换后面接的不是可折激活时，K 照旧可以切。"""
    gm = _gate_linear_graph(act=torch.ops.aten.neg.default,
                            cast_dtype=torch.float32)
    optimize_graph(gm, wram_bytes=2048)
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert any(n.args[0].meta["val"].shape[-1] < 64 for n in subs), "K 没有切"


class _CastGate(torch.nn.Module):
    """linear 结果经 `.to(float32)` 再接 silu，导出后中间是 `aten.to.dtype`。"""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(128, 64, dtype=torch.float16))

    def forward(self, x):
        return torch.nn.functional.silu(
            torch.nn.functional.linear(x, self.w).to(torch.float32))


@pytest.mark.parametrize("wram", [65536, 2048])
def test_exported_to_dtype_before_tail_activation_still_fuses(wram) -> None:
    """`torch.export` 产出的 `to.dtype` 与 `_to_copy` 同样认，K 不切，融合生效。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm = torch.export.export(
        _CastGate(), (torch.randn(4, 64, dtype=torch.float16),)).module()
    assert torch.ops.aten.to.dtype in _targets(gm)
    optimize_graph(gm, wram_bytes=wram)
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert all(n.args[0].meta["val"].shape[-1] == 64 for n in subs), "K 被切开了"
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.silu.default not in _targets(gm)
    assert all(FUSED_TAIL_META_KEY in n.meta for n in subs)


class _CastSwiGLU(torch.nn.Module):
    """门控投影经 `.to(float32)` 接 silu，再乘上投影：silu 后面还有消费者。"""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.randn(128, 64, dtype=torch.float16))
        self.up = torch.nn.Parameter(torch.randn(128, 64, dtype=torch.float16))

    def forward(self, x):
        f = torch.nn.functional
        gate = f.silu(f.linear(x, self.gate).to(torch.float32))
        return gate * f.linear(x, self.up).to(torch.float32)


@pytest.mark.parametrize("wram", [65536, 2048])
def test_cast_swiglu_silu_is_left_for_linear_fusion(wram) -> None:
    """通用融合要给夹浮点转换的门控 silu 让路，否则它被吃进主机闭包，折不进 linear。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    torch.manual_seed(0)
    model = _CastSwiGLU().eval()
    x = torch.randn(4, 64, dtype=torch.float16)
    gm = torch.export.export(model, (x,)).module()
    report = optimize_graph(gm, wram_bytes=wram)
    assert report.fused == 0, report
    assert not [n.name for n in gm.graph.nodes if n.name.startswith("fused_")]
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    from graph.spec_prop import _consumers_through_slices
    (gate_w,) = [n for n in gm.graph.nodes
                 if n.op == "get_attr" and n.target == "gate"]
    gate_subs = [n for n in _consumers_through_slices(gate_w)
                 if n.target is torch.ops.aten.linear.default]
    assert gate_subs
    assert all(n.args[0].meta["val"].shape[-1] == 64 for n in gate_subs), "门控投影的 K 被切开了"
    # 融合前图仍可执行，分块与让路不改数值。
    torch.testing.assert_close(gm(x), model(x), rtol=2e-2, atol=2e-2)
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.silu.default not in _targets(gm)
    assert all(FUSED_TAIL_META_KEY in n.meta for n in gate_subs)


class _SharedGate(torch.nn.Module):
    """门控投影（或它的浮点转换）另有一个真实用户，silu 折不进 linear。"""

    def __init__(self, share_cast: bool):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.randn(128, 64))
        self.up = torch.nn.Parameter(torch.randn(128, 64))
        self.share_cast = share_cast

    def forward(self, x):
        f = torch.nn.functional
        g = f.linear(x, self.gate)
        if self.share_cast:
            g = g.to(torch.float64)
        return f.silu(g) * f.linear(x, self.up) + g.mean(-1, keepdim=True)


@pytest.mark.parametrize("share_cast", [False, True], ids=["linear", "cast"])
def test_shared_gate_silu_goes_to_generic_fusion(share_cast) -> None:
    """linear 折不了尾部时不让路：silu 交给通用融合，不留成独立节点。"""
    torch.manual_seed(0)
    model = _SharedGate(share_cast).eval()
    x = torch.randn(4, 64)
    gm = torch.export.export(model, (x,)).module()
    report = optimize_graph(gm)
    assert report.fused >= 1, report
    assert torch.ops.aten.silu.default not in _targets(gm), "silu 被挡在通用融合之外"
    assert fuse_linear_activation(gm) == 0
    torch.testing.assert_close(gm(x), model(x))


class _BiasGate(torch.nn.Module):
    """带 bias 的门控投影后接 silu。"""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(64, 128)

    def forward(self, x):
        return torch.nn.functional.silu(self.gate(x))


def test_bias_linear_does_not_absorb_tail_activation() -> None:
    """A 路 linear 只有 x、w 两个实参，带 bias 的不折，与 B 路发射器同一口径。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    torch.manual_seed(0)
    model = _BiasGate().eval()
    x = torch.randn(4, 64)
    gm = torch.export.export(model, (x,)).module()
    optimize_graph(gm)
    (lin,) = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert len(lin.args) == 3
    assert fuse_linear_activation(gm) == 0
    assert FUSED_TAIL_META_KEY not in lin.meta
    assert torch.ops.aten.silu.default in _targets(gm)
    torch.testing.assert_close(gm(x), model(x))


class _DoubleCastSwiGLU(torch.nn.Module):
    """门控投影连续两层浮点转换后接 silu，再乘上投影。"""

    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.randn(128, 64, dtype=torch.float16))
        self.up = torch.nn.Parameter(torch.randn(128, 64, dtype=torch.float16))

    def forward(self, x):
        f = torch.nn.functional
        g = f.linear(x, self.gate).to(torch.float32).to(torch.float64)
        return f.silu(g) * f.linear(x, self.up).to(torch.float64)


@pytest.mark.parametrize("wram", [65536, 2048])
def test_double_cast_gate_still_fuses(wram) -> None:
    """多层浮点转换三处判定口径一致：通用融合让路、K 不切、silu 折进 linear。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from graph.spec_prop import _consumers_through_slices

    torch.manual_seed(0)
    model = _DoubleCastSwiGLU().eval()
    x = torch.randn(4, 64, dtype=torch.float16)
    gm = torch.export.export(model, (x,)).module()
    report = optimize_graph(gm, wram_bytes=wram)
    assert report.fused == 0, report
    (gate_w,) = [n for n in gm.graph.nodes
                 if n.op == "get_attr" and n.target == "gate"]
    gate_subs = [n for n in _consumers_through_slices(gate_w)
                 if n.target is torch.ops.aten.linear.default]
    assert gate_subs
    assert all(n.args[0].meta["val"].shape[-1] == 64 for n in gate_subs), "门控投影的 K 被切开了"
    torch.testing.assert_close(gm(x), model(x), rtol=2e-2, atol=2e-2)
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.silu.default not in _targets(gm)
    assert all(FUSED_TAIL_META_KEY in n.meta for n in gate_subs)


def test_weight_consumers_pass_through_to_dtype_layout() -> None:
    """`is_float_cast` 认的 `to.dtype_layout` 在找权重消费者时同样穿过。"""
    from graph.spec_prop import _consumers_through_slices

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("weight")
    cast = graph.call_function(torch.ops.aten.to.dtype_layout, (w,),
                               {"dtype": torch.float32})
    lin = graph.call_function(torch.ops.aten.linear.default, (x, cast))
    graph.output(lin)
    assert _consumers_through_slices(w) == [lin]


def test_tile_leaves_tail_activation_linear_whole_when_mn_cannot_fit() -> None:
    """只沿 M、N 切也压不进预算时整块不切，融合照常生效。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm = _gate_linear_graph()
    report = optimize_graph(gm, wram_bytes=256)
    assert report.tiled == 0
    (lin,) = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert fuse_linear_activation(gm) == 1
    assert FUSED_TAIL_META_KEY in lin.meta


def test_tile_still_splits_k_without_tail_activation() -> None:
    """后面不接激活的 linear 照旧可以切 K，限制只针对可折激活的 linear。"""
    gm = _gate_linear_graph(act=torch.ops.aten.neg.default)
    optimize_graph(gm, wram_bytes=2048)
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert any(n.args[0].meta["val"].shape[-1] < 64 for n in subs), "K 没有切"


def test_tile_respects_block_cap() -> None:
    """块数不超过上限的切法照常生效，且每个子块都在预算内。"""
    from graph.opt_passes import MAX_TILE_BLOCKS

    gm = _real_linear_graph(8, 256, 256)
    assert tile_graph(gm, wram_bytes=65536, tp_width=2) == 1
    subs = [n for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default]
    assert 1 < len(subs) <= MAX_TILE_BLOCKS
    for sub in subs:
        a, b = (arg.meta["val"] for arg in sub.args[:2])
        total = (a.numel() + b.numel() + sub.meta["val"].numel()) * 4
        assert total <= 65536, (sub.name, total)


def _matmul_graph(a_shape, b_shape):
    graph = Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    graph.output(graph.call_function(torch.ops.aten.matmul.default, (a, b)))
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    a_val, b_val = torch.randn(*a_shape), torch.randn(*b_shape)
    _shape(gm, a_val, b_val)
    return gm, a_val, b_val


def _sub_matmul_bytes(gm) -> int:
    """每个子 matmul 的输入、权重、输出三块字节之和，取最大值。"""
    worst = 0
    for node in gm.graph.nodes:
        if node.target is not torch.ops.aten.matmul.default:
            continue
        total = node.meta["val"].numel() * 4
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.meta.get("val") is not None:
                total += arg.meta["val"].numel() * 4
        worst = max(worst, total)
    return worst


@pytest.mark.parametrize("a_shape, b_shape, budget", [
    ((4, 8, 16), (16, 32), 512),
    ((4, 8, 16), (16, 32), 2048),
    ((8, 8, 16), (16, 32), 1024),
])
def test_tile_batched_matmul_fits_budget(a_shape, b_shape, budget) -> None:
    """batch 前导维要计入工作集：每个子 matmul 三块合计不超预算，数值不变。"""
    gm, a_val, b_val = _matmul_graph(a_shape, b_shape)

    tile_graph(gm, wram_bytes=budget)

    got = _run(gm, a=a_val, b=b_val)
    assert torch.allclose(got, a_val @ b_val, atol=1e-4, rtol=1e-4)
    assert _sub_matmul_bytes(gm) <= budget, _sub_matmul_bytes(gm)


def test_tile_skips_matmul_with_batched_right_operand() -> None:
    """设计边界：右操作数带 batch 的 matmul 不切，数值不变。

    各批的权重不同，行窗口按左操作数切会把批次对错，见设计四·附 A 分块的适用范围。
    """
    gm, a_val, b_val = _matmul_graph((8, 16), (4, 16, 32))

    assert tile_graph(gm, wram_bytes=64) == 0
    assert torch.allclose(_run(gm, a=a_val, b=b_val), a_val @ b_val)


def test_tile_counts_bias_when_deciding_whether_to_split() -> None:
    """判断整块是否超预算要和候选成本同一口径，bias 也算进去。

    1x4 输入、8x4 权重、8 维 bias，float32：三块 16+128+32 = 176 字节，
    加上 bias 32 字节是 208，超过 176 的预算，必须切。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    b = graph.placeholder("b")
    graph.output(graph.call_function(torch.ops.aten.linear.default, (x, w, b)))
    gm = GraphModule({}, graph)
    torch.manual_seed(0)
    x_val, w_val, b_val = torch.randn(1, 4), torch.randn(8, 4), torch.randn(8)
    _shape(gm, x_val, w_val, b_val)

    assert tile_graph(gm, wram_bytes=176) == 1, "含 bias 超预算却没切"
    got = _run(gm, x=x_val, w=w_val, b=b_val)
    assert torch.allclose(got, torch.nn.functional.linear(x_val, w_val, b_val), atol=1e-5)


@pytest.mark.parametrize("target, kind", [
    (torch.ops.aten.relu.default, "relu"),
    (torch.ops.aten.sigmoid.default, "sigmoid"),
])
def test_fuse_linear_activation_absorbs_other_kinds(target, kind) -> None:
    """linear 后的单用户激活不止 silu：表里的激活都折进 linear。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    b = graph.placeholder("b")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(target, (proj,))
    gated = graph.call_function(torch.ops.aten.mul.Tensor, (act, b))
    graph.output(gated)
    gm = GraphModule({}, graph)

    assert fuse_linear_activation(gm) == 1
    assert target not in _targets(gm)
    assert proj.meta[FUSED_TAIL_META_KEY].activation == kind


def test_fuse_linear_activation_skips_kind_outside_the_table() -> None:
    """表外激活（tanh）A 路内核没有实现，不折。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(torch.ops.aten.tanh.default, (proj,))
    graph.output(act)
    gm = GraphModule({}, graph)

    assert fuse_linear_activation(gm) == 0
    assert FUSED_TAIL_META_KEY not in proj.meta




def _spliced_linear_graph(join, wrap_dtype=None):
    """两个 linear 子块经 `join` 合回再接 relu，模拟分块后的 gate 投影。"""
    graph = Graph()
    a = graph.placeholder("a")
    w0 = graph.placeholder("w0")
    w1 = graph.placeholder("w1")
    p0 = graph.call_function(torch.ops.aten.linear.default, (a, w0))
    p1 = graph.call_function(torch.ops.aten.linear.default, (a, w1))
    if join == "cat":
        joined = graph.call_function(torch.ops.aten.cat.default, ([p0, p1], -1))
    else:
        joined = graph.call_function(torch.ops.aten.add.Tensor, (p0, p1))
    if wrap_dtype is not None:
        joined = graph.call_function(torch.ops.aten._to_copy.default,
                                     (joined,), {"dtype": wrap_dtype})
    act = graph.call_function(torch.ops.aten.relu.default, (joined,))
    graph.output(act)
    return GraphModule({}, graph), (p0, p1)


@pytest.mark.parametrize("wrap_dtype", [None, torch.float16],
                         ids=["bare", "to_f16"])
def test_fuse_linear_activation_folds_into_spliced_blocks(wrap_dtype) -> None:
    """沿 N 拼回的子块（可夹浮点转换）每块都带上激活，独立 relu 被删。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm, blocks = _spliced_linear_graph("cat", wrap_dtype)
    assert fuse_linear_activation(gm) == 1
    assert torch.ops.aten.relu.default not in _targets(gm)
    for block in blocks:
        assert block.meta[FUSED_TAIL_META_KEY].activation == "relu"


@pytest.mark.parametrize("act, kind", [
    (torch.ops.aten.silu.default, "silu"),
    (torch.ops.aten.relu.default, "relu"),
    (torch.ops.aten.sigmoid.default, "sigmoid"),
])
def test_spliced_blocks_with_folded_activation_match_unfused_values(act, kind) -> None:
    """f32 子块经 `cat -> _to_copy(f16)` 拼回再接激活，折进子块后逐元素与原图一致。

    折叠后激活在 f32 子块上算、再转 f16；原图先转 f16 再算。两者只差一次 f16 舍入。
    """
    from runtime.kernels import _LINEAR_TAIL
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    torch.manual_seed(0)
    x = torch.randn(4, 64)
    w0, w1 = torch.randn(64, 64) * 0.2, torch.randn(64, 64) * 0.2
    graph = Graph()
    xa = graph.placeholder("x")
    wa0 = graph.placeholder("w0")
    wa1 = graph.placeholder("w1")
    p0 = graph.call_function(torch.ops.aten.linear.default, (xa, wa0))
    p1 = graph.call_function(torch.ops.aten.linear.default, (xa, wa1))
    joined = graph.call_function(torch.ops.aten.cat.default, ([p0, p1], -1))
    narrow = graph.call_function(torch.ops.aten._to_copy.default, (joined,),
                                 {"dtype": torch.float16})
    graph.output(graph.call_function(act, (narrow,)))
    gm = GraphModule({}, graph)
    ref = gm(x, w0, w1).float()

    assert fuse_linear_activation(gm) == 1
    # 按执行计划的口径求值：每个带尾部的子块算完 linear 后补上激活。
    def tail(i):
        y = torch.nn.functional.linear(x, (w0, w1)[i]).numpy()
        node = (p0, p1)[i]
        return torch.from_numpy(_LINEAR_TAIL[node.meta[FUSED_TAIL_META_KEY].activation](y))
    got = torch.cat([tail(0), tail(1)], -1).half().float()
    np.testing.assert_allclose(got.numpy(), ref.numpy(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("join, wrap_dtype", [
    pytest.param("add", None, id="k_split_sum"),
    pytest.param("cat", torch.int32, id="to_int"),
])
def test_fuse_linear_activation_skips_unsafe_splices(join, wrap_dtype) -> None:
    """K 段相加后激活要算在和上；整数转换前算激活会改值。两种都不折。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    gm, blocks = _spliced_linear_graph(join, wrap_dtype)
    assert fuse_linear_activation(gm) == 0
    assert torch.ops.aten.relu.default in _targets(gm)
    assert not any(FUSED_TAIL_META_KEY in block.meta for block in blocks)


def test_fuse_elementwise_leaves_linear_tail_activation_alone() -> None:
    """linear 后的 relu 留给 linear 融合，通用融合不能先把它吃进主机闭包。"""
    graph = Graph()
    a = graph.placeholder("a")
    w = graph.placeholder("w")
    b = graph.placeholder("b")
    proj = graph.call_function(torch.ops.aten.linear.default, (a, w))
    act = graph.call_function(torch.ops.aten.relu.default, (proj,))
    out = graph.call_function(torch.ops.aten.mul.Tensor, (act, b))
    graph.output(out)
    gm = GraphModule({}, graph)

    assert fuse_elementwise(gm) == 0
    assert torch.ops.aten.relu.default in _targets(gm)


def test_fuse_ignores_export_metadata_assert() -> None:
    """导出器插入的 `_assert_tensor_metadata` 不是值消费者，不挡融合。

    生产者 neg 另挂一个断言，真正的用户只有 tanh，仍要合并，数值不变；
    残留的断言由 DCE 清理。
    """
    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    graph.call_function(torch.ops.aten._assert_tensor_metadata.default,
                        (neg,), {"dtype": torch.float32})
    out = graph.call_function(torch.ops.aten.tanh.default, (neg,))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([-1.0, 0.0, 0.5, 2.0])

    changed = fuse_elementwise(gm)

    assert changed == 1, "元数据断言挡住了融合"
    from torch.fx.interpreter import Interpreter

    assert torch.allclose(Interpreter(gm).run(x_val), torch.tanh(-x_val))
    eliminate_dead(gm)
    assert torch.ops.aten.tanh.default not in _targets(gm)


@pytest.mark.parametrize("act, ref", [
    (torch.ops.aten.gelu.default, torch.nn.functional.gelu),
    (torch.ops.aten.leaky_relu.default, torch.nn.functional.leaky_relu),
], ids=["gelu", "leaky_relu"])
def test_fuse_elementwise_activation_outside_the_old_table(act, ref) -> None:
    """gelu、leaky_relu 是纯逐元素、形状不变，mul 后接它们要合并，数值不变。"""
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
    out = graph.call_function(act, (mul,))
    graph.output(out)
    gm = GraphModule({}, graph)
    x_val = torch.tensor([-1.5, -0.2, 0.3, 2.0])
    y_val = torch.tensor([0.5, 2.0, -1.0, 1.5])

    changed = fuse_elementwise(gm)

    assert changed == 1, f"mul -> {act} 没有融合"
    assert len(_targets(gm)) == 1, _targets(gm)
    from torch.fx.interpreter import Interpreter

    assert torch.allclose(Interpreter(gm).run(x_val, y_val),
                          ref(x_val * y_val))
