"""切分通过形状进 GML，不通过字段。

守两件事：默认 tp=1 时本地形状等于全局形状（GML 产出不变的前提）；
多卡时各投影按列切/行切算对，GQA 下 k/v 按 kv_width 而不是 hidden_size 切。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gml_bridge.sharding import (
    describe,
    local_shapes,
    single_device_strategy,
)
from graph.strategy import llama_strategy

# llama2-7B
HIDDEN = 4096
HEADS = 32
KV_HEADS = 32
INTERMEDIATE = 11008
VOCAB = 32000
LAYERS = 32


def _strategy(num_dpus: int, num_stages: int = 1):
    return llama_strategy(
        num_dpus=num_dpus, num_stages=num_stages,
        num_heads=HEADS, num_kv_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE, vocab_size=VOCAB, num_layers=LAYERS,
    )


def test_default_strategy_is_single_device() -> None:
    """默认单卡：tp=1。这是 GML 当前导出走的路径。"""
    strategy = single_device_strategy(
        num_heads=HEADS, num_kv_heads=KV_HEADS,
        intermediate_size=INTERMEDIATE, vocab_size=VOCAB, num_layers=LAYERS)
    assert strategy.tp_width == 1


def test_tp1_local_equals_global() -> None:
    """tp=1 时本地形状恒等于全局形状。

    这条成立，接入算子编译器前后 GML 才可能逐字节相同。
    """
    shapes = local_shapes(
        _strategy(1), hidden_size=HIDDEN, num_heads=HEADS,
        num_kv_heads=KV_HEADS, intermediate_size=INTERMEDIATE)
    assert shapes.is_trivial
    assert shapes.q_out == shapes.k_out == shapes.v_out == HIDDEN
    assert shapes.o_in == HIDDEN
    assert shapes.gate_out == shapes.up_out == INTERMEDIATE
    assert shapes.down_in == INTERMEDIATE


def test_tp4_splits_column_and_row_projections() -> None:
    """tp=4：q/k/v/gate/up 列切（分输出维），o/down 行切（分输入维）。"""
    shapes = local_shapes(
        _strategy(4), hidden_size=HIDDEN, num_heads=HEADS,
        num_kv_heads=KV_HEADS, intermediate_size=INTERMEDIATE)
    assert not shapes.is_trivial
    assert shapes.q_out == HIDDEN // 4 == 1024
    assert shapes.gate_out == shapes.up_out == INTERMEDIATE // 4 == 2752
    assert shapes.o_in == HIDDEN // 4 == 1024
    assert shapes.down_in == INTERMEDIATE // 4 == 2752
    # 行切不改输出宽度。
    assert shapes.hidden_size == HIDDEN


def test_gqa_splits_kv_by_kv_width() -> None:
    """GQA 下 k/v 按 `num_kv_heads * head_dim` 切，不是按 hidden_size。

    用 hidden_size 会把 k/v 切错——这是 GQA 模型上最容易踩的一处。
    """
    shapes = local_shapes(
        _strategy(4), hidden_size=HIDDEN, num_heads=32,
        num_kv_heads=8, intermediate_size=INTERMEDIATE)
    head_dim = HIDDEN // 32           # 128
    kv_width = 8 * head_dim           # 1024
    assert shapes.k_out == shapes.v_out == kv_width // 4 == 256
    # q 仍按 hidden 切，与 k/v 不同。
    assert shapes.q_out == HIDDEN // 4 == 1024


def test_indivisible_width_is_rejected() -> None:
    """切不开就报错，不静默取整——取整会让各卡形状不一致。

    11009 是质数，tp=4 切不开（11008 能，所以这里刻意 +1）。
    """
    strategy = _strategy(4)
    with pytest.raises(ValueError, match="不能被 tp_width"):
        local_shapes(strategy, hidden_size=HIDDEN, num_heads=HEADS,
                     num_kv_heads=KV_HEADS, intermediate_size=11009)


def test_describe_mentions_single_device_for_tp1() -> None:
    shapes = local_shapes(
        _strategy(1), hidden_size=HIDDEN, num_heads=HEADS,
        num_kv_heads=KV_HEADS, intermediate_size=INTERMEDIATE)
    assert "单卡" in describe(shapes)


def test_describe_lists_widths_for_multi_device() -> None:
    shapes = local_shapes(
        _strategy(4), hidden_size=HIDDEN, num_heads=HEADS,
        num_kv_heads=KV_HEADS, intermediate_size=INTERMEDIATE)
    text = describe(shapes)
    assert "tp=4" in text and "1024" in text and "2752" in text


def test_gml_has_no_placement_fields() -> None:
    """GML 里不该有设备/切分字段——切分只通过形状体现。

    参考产物 616 个键名里这类词零命中，所以我方也不该产出。
    """
    from contracts import gml_coverage

    forbidden = ("device", "dpu", "shard", "tp_", "pp_", "stage",
                 "rank", "cluster", "placement")
    emitted = {name.lower() for name in gml_coverage.EMITTED}
    hits = [name for name in emitted
            if any(word in name for word in forbidden)]
    assert hits == [], f"GML 不该产出放置类字段: {hits}"
