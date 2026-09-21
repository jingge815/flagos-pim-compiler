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
from opcompiler_bridge.oplevel_emitter import (
    PIM_TARGET,
    emit_oplevel_mlir,
)


class _Val:
    """假的 meta["val"]，只需要 .shape。"""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def _graph_with(build) -> GraphModule:
    """造一个只含 call_function 节点的最小 GraphModule。"""
    graph = Graph()
    build(graph)
    graph.output(None)
    return GraphModule(torch.nn.Module(), graph)


def _add_node(graph: Graph, name: str, shape: tuple[int, ...]):
    node = graph.call_function(torch.ops.aten.alias.default, (name,))
    node.name = name
    node.meta["val"] = _Val(shape)
    return node


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
