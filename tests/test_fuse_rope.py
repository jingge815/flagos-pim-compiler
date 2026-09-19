"""RoPE 折叠的判据：六算子链折成一个节点，数值不变。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.fuse_pim import ABSORBED_META_KEY, fuse_for_pim
from graph.fuse_rope import ROPE_META_KEY, fuse_rope
from graph.split_heads import split_attention_heads
from graph.kv_dma_pass import KV_DMA_META_KEY, SPLIT_META_KEY, insert_kv_dma_and_split


@pytest.fixture(scope="module")
def graph():
    from transformers import LlamaConfig, LlamaForCausalLM
    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=8, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval()
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    return export_annotated_graph(model, 8, pos, dtype=torch.float32)


@pytest.fixture(scope="module")
def fused(graph):
    clone = copy.deepcopy(graph)
    report = fuse_rope(clone)
    return clone, report


def test_two_rope_chains_are_folded(fused) -> None:
    """一层 llama 有两条 RoPE：Q 一条、K 一条。"""
    clone, report = fused
    assert report.fused == 2
    tagged = [n for n in clone.graph.nodes if ROPE_META_KEY in n.meta]
    assert len(tagged) == 2


def test_rotate_half_ops_are_absorbed(fused) -> None:
    """链中间的 slice / neg / cat / mul 必须被吸收，不能单独发射。"""
    clone, _ = fused
    for node in clone.graph.nodes:
        match = node.meta.get(ROPE_META_KEY)
        if match is None:
            continue
        for absorbed in match.absorbed:
            assert absorbed.meta.get(ABSORBED_META_KEY) is True


def test_rope_preserves_numerics(graph, fused) -> None:
    """折叠后数值必须不变。"""
    clone, _ = fused
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, 8))
    mask = torch.zeros(1, 1, 8, 8)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        expected = graph(ids, mask, pos)
        actual = clone(ids, mask, pos)
    a = expected[0] if isinstance(expected, tuple) else expected
    b = actual[0] if isinstance(actual, tuple) else actual
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_kv_dma_and_split_counts(graph) -> None:
    """实测参考产物：2 个 KV_Cache_DMA、3 个 Split。"""
    clone = copy.deepcopy(graph)
    fuse_rope(clone)
    fuse_for_pim(clone)
    split_attention_heads(clone)
    report = insert_kv_dma_and_split(clone)
    assert report.kv_dma == 2
    assert report.splits == 3
    assert not report.skipped


def test_gml_op_types_match_reference_ratio(graph) -> None:
    """走完整流水线后，RoPE / KV / Split 的 op_type 要与实物个数相等。"""
    import collections, re
    from gml_bridge.from_fx import convert
    from gml_bridge.writer import write_gml
    from graph.quant_pass import insert_dynamic_scaling

    clone = copy.deepcopy(graph)
    fuse_rope(clone)
    fuse_for_pim(clone)
    split_attention_heads(clone)
    insert_kv_dma_and_split(clone)
    insert_dynamic_scaling(clone)
    nodes, edges, _, _ = convert(clone)
    text = write_gml(nodes, edges, version="26.2.1")
    counts = collections.Counter(re.findall(r'op_type "([^"]+)"', text))
    assert counts["Llama2Activation"] == 1
    assert counts["Llama2ActivationDQ"] == 1
    assert counts["KV_Cache_DMA"] == 2
    assert counts["Split"] == 3
    # 折叠后这三类必须收敛到实物个数。
    assert counts["EltwiseMul"] == 1
    assert counts["EltwiseAdd"] == 2
    assert counts["Concat"] == 1
