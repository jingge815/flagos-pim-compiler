"""逐头展开的判据。

这一步贡献参考产物 200 个节点里的 128 个，所以「数量对不对」是核心断言。
但最强的一条仍是语义等价：展开是图变换，不能改变数值
（`fuse_pim` 那次返工的教训，见 docs/gml-parser-output-plan-20260917.md §11.5.1）。
"""

from __future__ import annotations

import collections
import copy
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.fuse_pim import ABSORBED_META_KEY, ATTENTION_SCALE_META_KEY
from graph.split_heads import (
    HEAD_INDEX_META_KEY,
    HEAD_ROLE_META_KEY,
    ROLE_MASK,
    ROLE_MATMUL_PV,
    ROLE_MATMUL_QK,
    ROLE_SOFTMAX,
    split_attention_heads,
)

HEADS = 4


@pytest.fixture(scope="module")
def graph():
    """一层 llama2 的小图，4 个头。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=HEADS,
            num_key_value_heads=HEADS, max_position_embeddings=8,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    return export_annotated_graph(model, 8, position_ids, dtype=torch.float32)


@pytest.fixture(scope="module")
def expanded(graph):
    clone = copy.deepcopy(graph)
    report = split_attention_heads(clone)
    return clone, report


def _roles(gm) -> collections.Counter:
    return collections.Counter(
        node.meta[HEAD_ROLE_META_KEY]
        for node in gm.graph.nodes
        if HEAD_ROLE_META_KEY in node.meta
    )


def test_every_head_gets_the_full_chain(expanded) -> None:
    """每个头都要产出完整的 5 节点链：QKᵀ、Mask、Softmax、PV。

    实测参考产物 32 头共 64 个 MatMul、32 个 Mask、32 个 Softmax
    —— 比例是每头 2:1:1，这里按 4 头核对同样的比例。
    """
    clone, report = expanded
    assert report.attentions == 1
    assert report.heads == HEADS
    assert not report.dropped

    roles = _roles(clone)
    assert roles[ROLE_MATMUL_QK] == HEADS
    assert roles[ROLE_MATMUL_PV] == HEADS
    assert roles[ROLE_MASK] == HEADS
    assert roles[ROLE_SOFTMAX] == HEADS


def test_head_indices_are_zero_based_and_complete(expanded) -> None:
    """头下标必须是 0..heads-1 且不重不漏——它进 `split_channel_number`。"""
    clone, _ = expanded

    for role in (ROLE_MATMUL_QK, ROLE_MATMUL_PV, ROLE_SOFTMAX, ROLE_MASK):
        indices = sorted(
            node.meta[HEAD_INDEX_META_KEY]
            for node in clone.graph.nodes
            if node.meta.get(HEAD_ROLE_META_KEY) == role
        )
        assert indices == list(range(HEADS)), f"{role}: {indices}"


def test_attention_scale_lands_on_matmul1(expanded) -> None:
    """`1/√head_dim` 折进 matmul1 的定标 meta，不单独成节点。

    实测参考产物 32 个 matmul1 的 `Scaling_buffer_file` 全是 1/√128 = 0.088388。
    漏掉它等于丢掉 attention scale —— 数值全错而结构校验查不出来。
    """
    import math

    clone, _ = expanded
    head_dim = 32 // HEADS

    scales = [
        node.meta[ATTENTION_SCALE_META_KEY]
        for node in clone.graph.nodes
        if node.meta.get(HEAD_ROLE_META_KEY) == ROLE_MATMUL_QK
    ]
    assert len(scales) == HEADS
    for scale in scales:
        assert scale == pytest.approx(1.0 / math.sqrt(head_dim), rel=1e-3)

    # matmul2 不带定标（实测是 1.0，即不写这个 meta）。
    pv_scaled = [
        node for node in clone.graph.nodes
        if node.meta.get(HEAD_ROLE_META_KEY) == ROLE_MATMUL_PV
        and ATTENTION_SCALE_META_KEY in node.meta
    ]
    assert not pv_scaled


def test_original_sdpa_is_absorbed_not_deleted(expanded) -> None:
    """SDPA 留在图里但标记为已吸收。

    留着才能保持图可执行；标记让 GML 侧跨过它，不会既发射 SDPA
    又发射逐头链（那样节点数会翻倍）。
    """
    clone, _ = expanded

    sdpas = [
        node for node in clone.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.scaled_dot_product_attention.default
    ]
    assert len(sdpas) == 1
    assert sdpas[0].meta.get(ABSORBED_META_KEY) is True


def test_expansion_preserves_numerics(graph, expanded) -> None:
    """展开后的图必须跑出与原图相同的数值。

    这是最强的一条。逐头链把一个 SDPA 换成 `slice → matmul → mul → add →
    softmax → matmul → cat`，任何一处接错（转置方向、mask 加错头、
    concat 轴错）都会在这里暴露，而结构校验完全看不见。
    """
    clone, _ = expanded

    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (1, 8))
    causal_mask = torch.zeros(1, 1, 8, 8)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        expected = graph(input_ids, causal_mask, position_ids)
        actual = clone(input_ids, causal_mask, position_ids)

    expected_tensor = expected[0] if isinstance(expected, tuple) else expected
    actual_tensor = actual[0] if isinstance(actual, tuple) else actual
    torch.testing.assert_close(
        actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)


def test_expansion_is_idempotent(expanded) -> None:
    """展开过的图再展开一次不应有变化——吸收标记要挡住重复展开。"""
    clone, _ = expanded
    before = len(list(clone.graph.nodes))

    second = split_attention_heads(clone)
    assert second.attentions == 0
    assert len(list(clone.graph.nodes)) == before


# ---------------------------------------------------------------------------
# GML 侧：角色如何变成字段
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gml_text(graph):
    """走完整流水线（融合 + 逐头展开）后的 GML 文本。"""
    from graph.fuse_pim import fuse_for_pim
    from gml_bridge.from_fx import convert
    from gml_bridge.writer import write_gml

    clone = copy.deepcopy(graph)
    fuse_for_pim(clone)
    split_attention_heads(clone)
    nodes, edges, _, _ = convert(clone)
    return write_gml(nodes, edges, version="26.2.1")


def test_op_type_counts_follow_the_reference_ratio(gml_text: str) -> None:
    """`op_type` 分布要符合每头 2 MatMul : 1 Mask : 1 Softmax。

    实测参考产物（32 头）：MatMul 64、Mask 32、Softmax 32。
    """
    counts = collections.Counter(re.findall(r'op_type "([^"]+)"', gml_text))

    assert counts["Mask"] == HEADS
    assert counts["Softmax"] == HEADS
    # MatMul 含逐头的 2×heads，加上图里原有的（本例没有别的 matmul）。
    assert counts["MatMul"] == 2 * HEADS


def test_matmul_weight_format_splits_evenly(gml_text: str) -> None:
    """一半转置一半不转：QKᵀ 要 K 的转置，PV 不要。

    这是**数学决定的**，图编译器完全知道哪个是 QKᵀ，
    所以不需要算子编译器参与（见计划 §6.1）。
    """
    counts = collections.Counter(
        re.findall(r'weight_format "([^"]+)"', gml_text))
    assert counts["weights_transpose"] == HEADS
    assert counts["weight"] == HEADS


def test_split_channel_number_only_on_matmul1(gml_text: str) -> None:
    """`split_channel_number` 只出现在 matmul1 上，实测如此。"""
    assert gml_text.count("split_channel_number") == HEADS

    numbers = sorted(
        int(value) for value in
        re.findall(r"split_channel_number (\d+)", gml_text))
    assert numbers == list(range(HEADS))


def test_matmul_input_as_weight_is_set(gml_text: str) -> None:
    """每个 MatMul 都带 `MatMul_input_as_weight 1`。

    它表示第二个 operand 走权重通路 —— 也是 `input_count` 少记 1 的原因。
    """
    assert gml_text.count("MatMul_input_as_weight 1") == 2 * HEADS


def test_group_attention_num_records_total_heads(gml_text: str) -> None:
    """逐头展开后每个 MatMul 仍记录**总头数**，不是自己的下标。"""
    values = set(re.findall(r'group_attention_data_num (\d+)', gml_text))
    assert values == {"32"}, values


def test_input_count_identity_still_holds(graph) -> None:
    """展开之后 `Σ input_count + MatMul 数 == 边数` 必须仍成立。

    这条恒等式来自实物统计（267 + 64 == 331），是独立于生成器的判据。
    """
    from graph.fuse_pim import fuse_for_pim
    from gml_bridge.from_fx import convert

    clone = copy.deepcopy(graph)
    fuse_for_pim(clone)
    split_attention_heads(clone)
    nodes, edges, _, _ = convert(clone)

    total = sum(int(n.fields.get("input_count", 0)) for n in nodes)
    matmuls = sum(1 for n in nodes if n.fields.get("op_type") == "MatMul")
    assert total + matmuls == len(edges)
