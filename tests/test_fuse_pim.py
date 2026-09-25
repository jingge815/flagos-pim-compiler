"""硬件算子级折叠的判据：RMSNorm 六合一、Gemm+SiLU、attention 定标吸收。

这些断言盯的是「折完之后图长什么样」，而不是折叠内部怎么实现的——
所以换实现不会误报，但少折/多折一定失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse_pim import (
    ABSORBED_META_KEY,
    RMS_NORM_CHAIN,
    RMS_NORM_META_KEY,
    fuse_for_pim,
)


@pytest.fixture(scope="module")
def graph():
    """一层 llama2 的小图（真实结构、小维度），折叠前。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=8, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    return export_annotated_graph(model, 8, position_ids, dtype=torch.float32)


@pytest.fixture(scope="module")
def fused(graph):
    """折叠后的图与报告。图是模块级共享的，所以先深拷一份再折。"""
    import copy

    clone = copy.deepcopy(graph)
    report = fuse_for_pim(clone)
    return clone, report


def test_rms_norm_chains_are_folded(fused) -> None:
    """每条 RMSNorm 链折成一个节点。

    一层 llama2 有 3 个 RMSNorm：input_layernorm、post_attention_layernorm、
    以及模型末尾的 norm。折成 3 个带标记的节点。
    """
    clone, report = fused
    assert report.rms_norms == 3

    tagged = [n for n in clone.graph.nodes if RMS_NORM_META_KEY in n.meta]
    assert len(tagged) == 3


def test_rms_norm_chain_ops_are_gone(fused) -> None:
    """链中间的算子必须真被删掉，只留链首那一个锚点。

    这条抓的是「标记打了但节点没删」——那样 `rsqrt` 会残留，
    而 `OP_TYPES` 把 `rsqrt` 也映射成 `RMSNorm_vpu`，
    于是 GML 里冒出 6 个 RMSNorm_vpu 而实际只该有 3 个（踩过这个坑）。
    """
    clone, _ = fused
    anchor = RMS_NORM_CHAIN[0]

    for target in RMS_NORM_CHAIN[1:]:
        remaining = [
            n for n in clone.graph.nodes
            if n.op == "call_function" and n.target is target
            # add/mul 在别处也合法（残差、MLP 的逐元素乘），只查 RMSNorm 用的那些。
            and RMS_NORM_META_KEY not in n.meta
            and any(RMS_NORM_META_KEY in u.meta for u in n.users)
        ]
        assert not remaining, f"{target} 还剩 {len(remaining)} 个未删"

    anchors = [n for n in clone.graph.nodes
               if n.op == "call_function" and n.target is anchor]
    assert len(anchors) == 3


def test_rms_norm_records_epsilon_and_weight(fused) -> None:
    """折叠要记下 eps 与那个一维缩放张量。

    eps 从图里读，不从 config.json 猜——图是唯一真源。
    """
    clone, _ = fused

    for node in clone.graph.nodes:
        fusion = node.meta.get(RMS_NORM_META_KEY)
        if fusion is None:
            continue
        assert fusion.epsilon == pytest.approx(1e-6)
        assert fusion.weight_node is not None
        assert "layernorm.weight" in str(fusion.weight_node.target) or \
            "norm.weight" in str(fusion.weight_node.target)


def test_silu_is_folded_into_its_gemm(fused) -> None:
    """`linear -> silu` 折成一个带 FusedTail 的 Gemm。

    实物里 SiLU 在 Gemm 195 的 `contraction[fused_Silu_act]` 内，
    不是顶层节点——所以这里必须折，而 `graph/fuse.py` 有意不折它。
    """
    clone, report = fused
    assert report.activations == 1

    hosts = [n for n in clone.graph.nodes if FUSED_TAIL_META_KEY in n.meta]
    assert len(hosts) == 1
    assert hosts[0].meta[FUSED_TAIL_META_KEY].activation == "silu"

    # silu 仍留在图里（保持可执行），但被标记为「已吸收」，
    # GML 侧不会把它发射成独立节点。
    silus = [n for n in clone.graph.nodes
             if n.op == "call_function"
             and n.target is torch.ops.aten.silu.default]
    assert len(silus) == 1
    assert silus[0].meta.get(ABSORBED_META_KEY) is True


def test_fusion_reduces_the_emitted_node_count(graph, fused) -> None:
    """折叠必须减少**发射到 GML 的**节点数——这是它存在的理由。

    注意判据不是「fx 图的节点数变少」：折叠只打标记不删节点（那样图才仍可执行，
    见 test_fused_graph_still_runs）。真正的收敛发生在 GML 侧。
    """
    from gml_bridge.from_fx import convert

    clone, _ = fused
    before = len(convert(graph)[0])
    after = len(convert(clone)[0])

    assert after < before
    # 3 条 RMSNorm 链各折掉若干项 + 1 个 silu，量级要对得上。
    assert before - after >= 3


def test_fused_graph_still_runs(graph, fused) -> None:
    """折叠后的图必须仍能执行且数值不变。

    这是最强的一条：折叠是图变换，**不能改变语义**。若 RMSNorm 的链被折错
    （比如把 `mul` 的两个操作数搞反），这里的数值会立刻不一致。
    """
    clone, _ = fused

    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (1, 8))
    causal_mask = torch.zeros(1, 1, 8, 8)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        expected = graph(input_ids, causal_mask, position_ids)
        actual = clone(input_ids, causal_mask, position_ids)

    expected_tensor = expected[0] if isinstance(expected, tuple) else expected
    actual_tensor = actual[0] if isinstance(actual, tuple) else actual
    torch.testing.assert_close(actual_tensor, expected_tensor)


def test_report_totals_add_up(fused) -> None:
    _, report = fused
    assert report.total == (
        report.rms_norms + report.activations + report.attention_scales)
    assert "RMSNorm" in str(report)


def test_fusion_is_idempotent(fused) -> None:
    """折过的图再折一次不应有变化——meta 标记要挡住重复折叠。"""
    clone, _ = fused
    before = len(list(clone.graph.nodes))

    second = fuse_for_pim(clone)
    assert second.rms_norms == 0
    assert second.activations == 0
    assert len(list(clone.graph.nodes)) == before


def _aten_graph(module, *example_inputs):
    """导出成 **aten 级** 图。

    不能用 `torch.fx.symbolic_trace` —— 它产出的是 python 级的
    `operator.truediv` / `torch.bmm`，而生产路径（`export_annotated_graph`）
    产出的是 aten 重载。折叠 pass 匹配的是后者，用前者测等于测了个空。
    """
    exported = torch.export.export(module.eval(), example_inputs)
    return exported.module()


def test_attention_scale_absorption_matches_one_over_sqrt_d() -> None:
    """手写 attention 的 `1/√d` 要被吸收进上游 matmul 的 meta。

    真实 llama2 把这个缩放藏在 `scaled_dot_product_attention` 内部，图里看不到，
    所以这里用一个手写的小图验证吸收逻辑本身。逐头展开 pass 拆开 SDPA 之后，
    生产图里就会出现这个形态。
    """
    import math

    from graph.fuse_pim import ATTENTION_SCALE_META_KEY, _absorb_attention_scale

    head_dim = 64

    class Attention(torch.nn.Module):
        def forward(self, q, k):
            return torch.bmm(q, k) / math.sqrt(head_dim)

    graph = _aten_graph(
        Attention(), torch.randn(2, 4, head_dim), torch.randn(2, head_dim, 4))
    assert _absorb_attention_scale(graph) == 1

    scales = [
        node.meta[ATTENTION_SCALE_META_KEY]
        for node in graph.graph.nodes
        if ATTENTION_SCALE_META_KEY in node.meta
    ]
    assert len(scales) == 1
    assert scales[0] == pytest.approx(1.0 / math.sqrt(head_dim), rel=1e-3)

    # 除法节点仍在图里（保持可执行），但已标记为被吸收。
    divisions = [
        n for n in graph.graph.nodes
        if n.op == "call_function" and n.target is torch.ops.aten.div.Tensor
    ]
    assert len(divisions) == 1
    assert divisions[0].meta.get(ABSORBED_META_KEY) is True


def test_unrelated_scalar_division_is_not_absorbed() -> None:
    """不像 `1/√d` 的缩放不能被吸收——否则会静默吃掉别的算子。

    3.0 不是任何整数 head_dim 的 √d，所以必须放过。
    """
    from graph.fuse_pim import _absorb_attention_scale

    class Scaled(torch.nn.Module):
        def forward(self, q, k):
            return torch.bmm(q, k) / 3.0

    graph = _aten_graph(
        Scaled(), torch.randn(2, 4, 8), torch.randn(2, 8, 4))
    assert _absorb_attention_scale(graph) == 0
