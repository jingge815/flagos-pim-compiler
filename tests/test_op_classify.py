"""op_classify 的代表形状要和真实图里的节点粒度对齐，不能用早期的单头近似。"""
from __future__ import annotations

from genesim_bridge.op_classify import ShapePoint, _concat_ir, _lut_ir, _softmax_ir

_DIMS = {"hidden_size": 4096, "head_dim": 128, "num_heads": 32, "ffn_dim": 11008}


def test_lut_ir_uses_ffn_width_not_hidden_size() -> None:
    """SILU 作用在 FFN 中间宽度上，不是 hidden_size。

    llama2-7b 的 hidden_size=4096、intermediate_size=11008，用 hidden_size
    会让搬运量少算约 2.69 倍。
    """
    text = _lut_ir(_DIMS, ShapePoint(tq=128, tp=0))
    assert "tensor<128x11008x" in text, text


def test_concat_ir_joins_all_heads_not_two() -> None:
    """CONCAT 把 num_heads 个头拼回 hidden_size，不是固定拼 2 段。"""
    text = _concat_ir(_DIMS, ShapePoint(tq=128, tp=0))
    # 函数签名里 32 个输入参数（%a0..%a31），每个是 [128, 128]（Tq x head_dim）。
    assert text.count("%a0:") == 1 and text.count("%a31:") == 1, text
    assert "%a32:" not in text, text


def test_softmax_ir_is_per_head_not_folded() -> None:
    """真实图里 SOFTMAX 是每个头单独一个节点（形状 [Tq, Tp+Tq]），
    不是把 num_heads 折进行数的一个大节点。"""
    text = _softmax_ir(_DIMS, ShapePoint(tq=128, tp=64))
    # lkv = tp + tq = 192，行数应该是 tq 本身，不是 tq*num_heads。
    assert "tensor<128x192x" in text, text
    assert "tensor<4096x" not in text, text
