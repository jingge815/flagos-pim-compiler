"""FX 图 → 整算子级 PIM MLIR 的发射器。

这组测试守的是「图编译器 → 算子编译器」这段链路：发射出的 IR 必须能被
FlagTree 的 `-pim-expand-phases` 接受，且相位数与参考产物一致。

不加载真实 7B：用手工构造的小 GraphModule 覆盖形状口径与分派规则，
真实模型的端到端另见 `test_oplevel_emitter_live.py`。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.fx import Graph, GraphModule

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.fuse_rope import ROPE_META_KEY, RopeMatch
from graph.quant_pass import DQ_META_KEY, DynamicScalingSpec
from graph.split_heads import HEAD_ROLE_META_KEY, ROLE_SOFTMAX
from contracts.unified_ir import STAGE_FUSED, STAGE_PARTITIONED, mark_stage
from opcompiler_bridge.oplevel_emitter import (
    PIM_TARGET,
    emit_oplevel_mlir,
)


class _Val:
    """假的 meta["val"]，只需要 .shape。"""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _graph_with(build) -> GraphModule:
    """造一个只含 call_function 节点的最小 GraphModule。

    手搓的图直接标到 `STAGE_FUSED`：这些夹具造的就是「融合之后」那个形态
    （节点自带 DQ / RoPE / 相位标记），只是没跑真正的 pass 链。
    `emit_oplevel_mlir` 的入口断言要求这个阶段，夹具得如实声明自己是什么。
    """
    graph = Graph()
    build(graph)
    graph.output(None)
    gm = GraphModule(torch.nn.Module(), graph)
    mark_stage(gm, STAGE_PARTITIONED)
    mark_stage(gm, STAGE_FUSED)
    return gm


def _add_node(graph: Graph, name: str, shape: tuple[int, ...]):
    node = graph.call_function(torch.ops.aten.alias.default, (name,))
    node.name = name
    node.meta["val"] = _Val(shape)
    return node


def test_unfused_eltwise_activation_emits_standalone_lut() -> None:
    """带融合标记的逐元素主算子，B 路把激活发成独立的 pim.lut。

    发射端不预写 activation，算子层融合才有东西可折。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from contracts.ir_payloads import FusedTail

    def build(g):
        lhs = _add_node(g, "lhs", (1, 16))
        rhs = _add_node(g, "rhs", (1, 16))
        main = _add_node(g, "residual", (1, 16))
        main.target = torch.ops.aten.add.Tensor
        main.args = (lhs, rhs)
        act = _add_node(g, "act_relu", (1, 16))
        act.target = torch.ops.aten.relu.default
        act.args = (main,)
        main.meta[FUSED_TAIL_META_KEY] = FusedTail(
            activation="relu", pool=None, nodes=[act])

    report = emit_oplevel_mlir(_graph_with(build))
    assert "pim.eltwise" in report.text, report.text
    assert "activation =" not in report.text.split("pim.eltwise")[1].split("pim.lut")[0], report.text
    assert "pim.lut" in report.text, report.text
    assert "#pim.activation<relu>" in report.text, report.text


def test_unfused_activation_survives_full_b_path() -> None:
    """独立激活的发射产物跑完默认 B 路：LUT 折进 eltwise，降到 C 有实现。

    算子层融合把 LUT 折回主算子的 activation 属性，契约校验通过，
    `-pim-lower-to-emitc` 按 relu 补上 C 实现。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from contracts.ir_payloads import FusedTail
    from opcompiler_bridge.driver import _run_oplevel_triton_opt

    def build(g):
        lhs = _add_node(g, "lhs", (1, 16))
        rhs = _add_node(g, "rhs", (1, 16))
        main = _add_node(g, "residual", (1, 16))
        main.target = torch.ops.aten.add.Tensor
        main.args = (lhs, rhs)
        act = _add_node(g, "act_relu", (1, 16))
        act.target = torch.ops.aten.relu.default
        act.args = (main,)
        main.meta[FUSED_TAIL_META_KEY] = FusedTail(
            activation="relu", pool=None, nodes=[act])

    report = emit_oplevel_mlir(_graph_with(build))
    assert "pim.lut" in report.text, "发射端应产出独立的 pim.lut"

    pimir, emitc = _run_oplevel_triton_opt(report.text)
    # 融合发生：LUT 消失，activation 落位，契约校验已通过（没抛即过）。
    assert "pim.lut" not in pimir, pimir
    assert "activation = #pim.act_spec<kind = relu>" in pimir, pimir
    assert "pim_lut_relu" in emitc, "降级侧没有 relu 的 C 实现"


def test_module_declares_pim_target() -> None:
    """必须带 `pim.target`，否则 11008 这类宽度会被 2 的幂检查拒。

    手写 MLIR 不经过 `convert-triton-to-pim`，属性不会自动出现。
    """
    gm = _graph_with(lambda g: None)
    report = emit_oplevel_mlir(gm)
    assert f'pim.target = "{PIM_TARGET}"' in report.text


def test_dq_is_flattened_to_rank2() -> None:
    """DQ 压成 `[1, numel]`，组数才与 spec 一致。

    `(1,16,4096)` 按单轴算是 4096/128=32 组，而 spec 要 512 组；
    压平成 `[1,65536]` 后校验器算 65536/128=512，对上。
    """
    def build(g):
        node = _add_node(g, "dq0", (1, 16, 4096))
        node.meta[DQ_META_KEY] = DynamicScalingSpec(
            group_size=128, numel=65536, is_attention_scores=False)

    report = emit_oplevel_mlir(_graph_with(build))
    assert len(report.ops) == 1
    assert "tensor<1x65536xf16>" in report.text
    assert "tensor<512xf16>" in report.text
    assert "groupSize = 128" in report.text


def test_dq_attention_scores_single_group() -> None:
    """attention scores 整条当一组（gs == numel）。"""
    def build(g):
        node = _add_node(g, "dq_score", (1, 1, 16, 16))
        node.meta[DQ_META_KEY] = DynamicScalingSpec(
            group_size=256, numel=256, is_attention_scores=True)

    report = emit_oplevel_mlir(_graph_with(build))
    assert "tensor<1x256xf16>" in report.text
    assert "tensor<1xf16>" in report.text


def test_dq_skipped_when_spec_disagrees_with_shape() -> None:
    """spec.numel 与形状不符说明有一侧算错，宁可跳过也不要发错的 IR。"""
    def build(g):
        node = _add_node(g, "dq_bad", (1, 16, 4096))
        node.meta[DQ_META_KEY] = DynamicScalingSpec(
            group_size=128, numel=999, is_attention_scores=False)

    report = emit_oplevel_mlir(_graph_with(build))
    assert report.ops == []
    assert "dq_bad" in report.skipped


def test_softmax_keeps_reduction_axis() -> None:
    """Softmax 压成 `[rows, S]`，**不能**压成 `[1, rows*S]`。

    压平会让归约跨行，32 个头的分数混在一起。
    """
    def build(g):
        node = _add_node(g, "sm0", (1, 1, 16, 16))
        node.meta[HEAD_ROLE_META_KEY] = ROLE_SOFTMAX

    report = emit_oplevel_mlir(_graph_with(build))
    assert "tensor<16x16xf16>" in report.text
    assert "tensor<1x256xf16>" not in report.text
    assert "axis = 1" in report.text


def test_rope_keeps_rank4_for_broadcast() -> None:
    """RoPE **保留 rank-4**：压平会让 cos/sin 不可广播。

    src 65536 元素 vs cos/sin 2048 元素，压成 rank-2 后
    `[1,65536]` 与 `[1,2048]` 不可广播，校验器直接拒。
    广播必须沿 head 轴发生。
    """
    def build(g):
        src = _add_node(g, "q", (1, 32, 16, 128))
        cos = _add_node(g, "cos", (1, 1, 16, 128))
        sin = _add_node(g, "sin", (1, 1, 16, 128))
        node = _add_node(g, "rope0", (1, 32, 16, 128))
        node.meta[ROPE_META_KEY] = RopeMatch(source=src, cos=cos, sin=sin)

    report = emit_oplevel_mlir(_graph_with(build))
    assert "tensor<1x32x16x128xf16>" in report.text
    assert "tensor<1x1x16x128xf16>" in report.text
    # head 数从被广播掉的那一维取出。
    assert "numHeads = 32" in report.text


def test_first_rope_also_emits_dq() -> None:
    """Q 路（第一条）RoPE 尾部带量化：那个节点要发 **两个** 算子。

    对应 GML 的 `Llama2ActivationDQ` —— 参考产物里 Q 的 RoPE 后面跟 4 相 DQ。
    """
    def build(g):
        cos = _add_node(g, "cos", (1, 1, 16, 128))
        sin = _add_node(g, "sin", (1, 1, 16, 128))
        q_src = _add_node(g, "q", (1, 32, 16, 128))
        rope_q = _add_node(g, "rope_q", (1, 32, 16, 128))
        rope_q.meta[ROPE_META_KEY] = RopeMatch(source=q_src, cos=cos, sin=sin)
        k_src = _add_node(g, "k", (1, 32, 16, 128))
        rope_k = _add_node(g, "rope_k", (1, 32, 16, 128))
        rope_k.meta[ROPE_META_KEY] = RopeMatch(source=k_src, cos=cos, sin=sin)

    report = emit_oplevel_mlir(_graph_with(build))
    by_node: dict[str, set[str]] = {}
    for op in report.ops:
        by_node.setdefault(op.fx_name, set()).add(op.kind)

    # Q 路发 RoPE + DQ，K 路只发 RoPE（add 写 cache）。
    assert by_node["rope_q"] == {"rope", "dq"}
    assert by_node["rope_k"] == {"rope"}
    # 函数名靠后缀区分，否则同名的第二个会被去重吃掉。
    assert len({op.func for op in report.ops}) == len(report.ops)


def test_first_rope_emits_dq() -> None:
    """只有一条 RoPE 时按 Q 路发 DQ。"""
    def build(g):
        src = _add_node(g, "q", (1, 32, 16, 128))
        cos = _add_node(g, "cos", (1, 1, 16, 128))
        sin = _add_node(g, "sin", (1, 1, 16, 128))
        node = _add_node(g, "rope_q", (1, 32, 16, 128))
        node.meta[ROPE_META_KEY] = RopeMatch(source=src, cos=cos, sin=sin)

    report = emit_oplevel_mlir(_graph_with(build))
    assert sorted(op.kind for op in report.ops) == ["dq", "rope"]


def test_dynamic_shape_is_skipped_not_guessed() -> None:
    """取不到静态形状就跳过——MLIR 需要静态类型，不能瞎猜。"""
    def build(g):
        node = g.call_function(torch.ops.aten.alias.default, ("x",))
        node.name = "no_shape"
        node.meta[DQ_META_KEY] = DynamicScalingSpec(
            group_size=128, numel=65536, is_attention_scores=False)

    report = emit_oplevel_mlir(_graph_with(build))
    assert report.ops == []
    assert "no_shape" in report.skipped


def test_expected_phase_counts_match_reference() -> None:
    """三类算子的相位数：DQ 4、Softmax 5、RoPE 3。"""
    def build(g):
        dq = _add_node(g, "dq", (1, 4096))
        dq.meta[DQ_META_KEY] = DynamicScalingSpec(
            group_size=128, numel=4096, is_attention_scores=False)
        sm = _add_node(g, "sm", (16, 16))
        sm.meta[HEAD_ROLE_META_KEY] = ROLE_SOFTMAX
        src = _add_node(g, "q", (1, 32, 16, 128))
        cos = _add_node(g, "cos", (1, 1, 16, 128))
        sin = _add_node(g, "sin", (1, 1, 16, 128))
        rope = _add_node(g, "rope", (1, 32, 16, 128))
        rope.meta[ROPE_META_KEY] = RopeMatch(source=src, cos=cos, sin=sin)

    report = emit_oplevel_mlir(_graph_with(build))
    expected = {op.kind: op.expected_phases for op in report.ops}
    assert expected == {"dq": 4, "softmax": 5, "rope": 3}


def test_an_unfused_graph_is_rejected_at_the_entry() -> None:
    """没跑过融合的图要在入口被拦住，不能发出半成品 pimir。

    设计 §3.3.3：出口标记只是记账，前置断言才是执行点。这条链的顺序依赖
    （必须先融合）原先只写在 docstring 里，跑错顺序不会有任何编译期信号 ——
    发出去的 pimir 缺 DQ / RoPE 锚点，算子编译器照样接受，相位数静默变少。
    """
    graph = Graph()
    _add_node(graph, "x", (1, 4096))
    graph.output(None)
    gm = GraphModule(torch.nn.Module(), graph)

    with pytest.raises(ValueError, match="要求图已达到阶段"):
        emit_oplevel_mlir(gm)


def test_real_llama_b_path_fuses_every_gate_activation() -> None:
    """真实小模型走默认 B 路，算子层融合在每层门控投影上都真实触发。

    发射器按图上的主算子与尾部激活发 `pim.matmul` 加独立 `pim.lut`，
    `-pim-fuse-activation` 把激活折回去。融合前 LUT 数等于层数，融合后
    LUT 为 0、带 activation 的矩阵乘等于层数。
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from gml_bridge.export import fuse_for_gml
    from graph.partition import partition_graph
    from opcompiler_bridge.phase_source import _run_passes, opcompiler_available
    from runtime.compile import export_annotated_graph

    if not opcompiler_available():
        pytest.skip("triton-opt 没有相位展开 pass")
    layers = 2
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    gm = export_annotated_graph(model, 1, torch.tensor([[0]]),
                                dtype=torch.float32)
    partition_graph(gm)
    fuse_for_gml(gm)

    report = emit_oplevel_mlir(gm)
    assert report.text.count("pim.lut") == layers, \
        "发射器没有把门控激活发成独立 pim.lut，B 路融合没有对象"
    assert "activation = #pim.act_spec" not in report.text, \
        "发射端不该预先写好 activation，否则算子层融合只是空跑"
    expanded = _run_passes(report.text)
    matmuls = [line for line in expanded.splitlines() if "pim.matmul" in line]
    fused = expanded.count("activation = #pim.act_spec<kind = silu>")
    assert fused == layers, (fused, matmuls)


def _tail_graph(main_target, main_args, main_shape, *, main_kwargs=None,
                arg_shapes=(), pool=None):
    """主算子加 relu 尾部的最小图，实参按给定形状造成占位节点。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from contracts.ir_payloads import FusedTail

    def build(g):
        nodes = []
        for i, shape in enumerate(arg_shapes):
            nodes.append(_add_node(g, f"in{i}", shape))
        args = tuple(nodes[a] if isinstance(a, int) else a for a in main_args)
        main = _add_node(g, "main", main_shape)
        main.target = main_target
        main.args = args
        main.kwargs = main_kwargs or {}
        act = _add_node(g, "act_relu", main_shape)
        act.target = torch.ops.aten.relu.default
        act.args = (main,)
        main.meta[FUSED_TAIL_META_KEY] = FusedTail(
            activation="relu", pool=pool, nodes=[act])

    return _graph_with(build)


def test_fused_addmm_is_skipped_not_misread() -> None:
    """addmm 的实参是 (bias, x, w)，pim.matmul 表达不了 bias，要跳过而不是把 bias 当 x。"""
    gm = _tail_graph(torch.ops.aten.addmm.default, (0, 1, 2), (2, 8),
                     arg_shapes=[(8,), (2, 4), (4, 8)])
    report = emit_oplevel_mlir(gm)
    assert "pim.matmul" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_linear_with_bias_is_skipped() -> None:
    """带 bias 的 linear 发成 pim.matmul 会丢掉 bias，要跳过。"""
    gm = _tail_graph(torch.ops.aten.linear.default, (0, 1, 2), (2, 8),
                     arg_shapes=[(2, 4), (8, 4), (8,)])
    report = emit_oplevel_mlir(gm)
    assert "pim.matmul" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_linear_with_keyword_bias_is_skipped() -> None:
    """bias 按关键字传入时同样表达不了，要跳过而不是丢掉 bias。"""
    gm = _tail_graph(torch.ops.aten.linear.default, (0, 1), (2, 8),
                     arg_shapes=[(2, 4), (8, 4), (8,)])
    main = next(n for n in gm.graph.nodes if n.name == "main")
    main.kwargs = {"bias": next(n for n in gm.graph.nodes if n.name == "in2")}
    report = emit_oplevel_mlir(gm)
    assert "pim.matmul" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_linear_with_explicit_none_bias_still_emits() -> None:
    """`linear(x, w, None)` 没有 bias，照常发 pim.matmul 加独立 pim.lut。"""
    gm = _tail_graph(torch.ops.aten.linear.default, (0, 1, None), (2, 8),
                     arg_shapes=[(2, 4), (8, 4)])
    report = emit_oplevel_mlir(gm)
    assert "pim.matmul" in report.text and "pim.lut" in report.text, report.text
    assert "main" not in report.skipped


def test_fused_linear_with_pool_is_skipped() -> None:
    """尾部带池化时 pim.matmul 加 pim.lut 表达不了池化，要跳过。"""
    gm = _tail_graph(torch.ops.aten.linear.default, (0, 1), (2, 8),
                     arg_shapes=[(2, 4), (8, 4)], pool="global_average")
    report = emit_oplevel_mlir(gm)
    assert "pim.matmul" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_mm_keeps_operand_shapes() -> None:
    """mm 的实参是 (x, w)，K 取 x 的末维，发出的两个操作数形状与图一致。"""
    gm = _tail_graph(torch.ops.aten.mm.default, (0, 1), (2, 8),
                     arg_shapes=[(2, 4), (4, 8)])
    report = emit_oplevel_mlir(gm)
    assert "tensor<2x4xi8>, tensor<4x8xi8>" in report.text, report.text


@pytest.mark.parametrize("main_args, arg_shapes, kwargs", [
    pytest.param((0, 2.0), [(2, 8)], {}, id="scalar"),
    pytest.param((0, 1), [(2, 8), (1, 8)], {}, id="broadcast"),
    pytest.param((0, 1), [(2, 8), (2, 8)], {"alpha": 2}, id="alpha"),
])
def test_fused_eltwise_that_two_same_shape_slots_cannot_express_is_skipped(
        main_args, arg_shapes, kwargs) -> None:
    """标量、广播或 alpha 不为 1 的 add 发成两个同形操作数会改掉计算，要跳过。"""
    gm = _tail_graph(torch.ops.aten.add.Tensor, main_args, (2, 8),
                     main_kwargs=kwargs, arg_shapes=arg_shapes)
    report = emit_oplevel_mlir(gm)
    assert "pim.eltwise" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_eltwise_with_pool_is_skipped() -> None:
    """逐元素主算子尾部带池化时同样跳过，并记进 skipped。"""
    gm = _tail_graph(torch.ops.aten.add.Tensor, (0, 1), (2, 8),
                     arg_shapes=[(2, 8), (2, 8)], pool="max")
    report = emit_oplevel_mlir(gm)
    assert "pim.eltwise" not in report.text, report.text
    assert "main" in report.skipped


def test_fused_eltwise_with_two_same_shape_operands_still_emits() -> None:
    """两个同形节点操作数、alpha 缺省的 add 照常发 pim.eltwise 加独立 pim.lut。"""
    gm = _tail_graph(torch.ops.aten.add.Tensor, (0, 1), (2, 8),
                     arg_shapes=[(2, 8), (2, 8)])
    report = emit_oplevel_mlir(gm)
    assert "pim.eltwise" in report.text and "pim.lut" in report.text, report.text
