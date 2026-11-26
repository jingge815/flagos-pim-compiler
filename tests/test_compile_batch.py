"""batch 维从编译入口进图的判据：B=1 与现状一致，B=4 导出形状正确。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.compile import causal_mask_of, export_annotated_graph

SEQ_LEN = 8


def _model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).eval()


def _placeholder_shapes(gm) -> dict[str, tuple[int, ...]]:
    """按占位符名字收集形状。"""
    shapes = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            shapes[str(node.target)] = tuple(node.meta["val"].shape)
    return shapes


def _node_names(gm) -> list[str]:
    return [node.name for node in gm.graph.nodes]


@pytest.fixture(scope="module")
def model():
    return _model()


def test_b1_export_identical(model) -> None:
    """batch_size=1（显式）与缺省调用的图逐项一致，占位符形状与现状相同。"""
    pos = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    default_gm = export_annotated_graph(model, SEQ_LEN, pos, dtype=torch.float32)
    explicit_gm = export_annotated_graph(
        model, SEQ_LEN, pos, dtype=torch.float32, batch_size=1)
    assert _node_names(default_gm) == _node_names(explicit_gm)
    assert _placeholder_shapes(default_gm) == _placeholder_shapes(explicit_gm)
    shapes = _placeholder_shapes(explicit_gm)
    input_key = next(k for k in shapes if "input_ids" in k)
    pos_key = next(k for k in shapes if "position_ids" in k)
    mask_key = next(k for k in shapes if "mask" in k)
    assert shapes[input_key] == (1, SEQ_LEN)
    assert shapes[pos_key] == (1, SEQ_LEN)
    assert shapes[mask_key] == (1, 1, SEQ_LEN, SEQ_LEN)


def test_b4_export_shapes(model) -> None:
    """batch_size=4：prefill 导出 [4,S]，decode 导出 [4,1]，掩码共享广播。"""
    pos = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).repeat(4, 1)
    gm = export_annotated_graph(
        model, SEQ_LEN, pos, dtype=torch.float32, batch_size=4)
    shapes = _placeholder_shapes(gm)
    input_key = next(k for k in shapes if "input_ids" in k)
    pos_key = next(k for k in shapes if "position_ids" in k)
    mask_key = next(k for k in shapes if "mask" in k)
    assert shapes[input_key] == (4, SEQ_LEN)
    assert shapes[pos_key] == (4, SEQ_LEN)
    assert shapes[mask_key] == (1, 1, SEQ_LEN, SEQ_LEN)

    decode_gm = export_annotated_graph(
        model, 1, torch.zeros(4, 1, dtype=torch.long), dtype=torch.float32,
        batch_size=4)
    decode_shapes = _placeholder_shapes(decode_gm)
    decode_input = next(k for k in decode_shapes if "input_ids" in k)
    assert decode_shapes[decode_input] == (4, 1)


def test_position_ids_batch_mismatch_raises(model) -> None:
    """position_ids 行数与 batch_size 不符是契约错误，直接抛。"""
    pos = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    with pytest.raises(ValueError, match="batch"):
        export_annotated_graph(
            model, SEQ_LEN, pos, dtype=torch.float32, batch_size=2)


def test_b4_batched_forward_matches_single(model) -> None:
    """[B,S] 输入配共享掩码的 batched 前向与逐样本前向逐位一致。"""
    torch.manual_seed(7)
    prompt = torch.randint(0, 128, (4, SEQ_LEN), dtype=torch.long)
    pos = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).repeat(4, 1)
    mask = causal_mask_of(SEQ_LEN, dtype=torch.float32)
    batched = model(input_ids=prompt, attention_mask=mask,
                    position_ids=pos, use_cache=False).logits
    for b in range(4):
        single = model(
            input_ids=prompt[b:b + 1],
            attention_mask=mask,
            position_ids=pos[b:b + 1], use_cache=False).logits
        assert torch.equal(batched[b], single[0])
