"""验证 DPU、层和张量并行宽度的切分关系。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.strategy import (
    LLAMA_WEIGHT_RULES,
    ShardStrategy,
    format_strategy,
    llama_strategies,
    llama_strategy,
)

# Llama-2-7B 结构参数。
LLAMA2_7B = dict(
    num_heads=32, num_kv_heads=32, intermediate_size=11008, vocab_size=32000, num_layers=32
)


def test_tensor_parallel_covers_every_dpu_in_one_stage() -> None:
    """验证单流水段包含全部 DPU 和层。"""
    strategy = ShardStrategy(name="tp8", num_dpus=8, num_stages=1)

    assert strategy.tp_width == 8
    assert strategy.kind == "tensor"
    assert strategy.dpus_of_stage(0) == (0, 1, 2, 3, 4, 5, 6, 7)
    for layer in range(32):
        assert strategy.dpus_of_layer(layer, 32) == tuple(range(8))
        assert strategy.stage_of_layer(layer, 32) == 0
    assert strategy.layers_of_dpu(3, 32) == list(range(32))


def test_pure_pipeline_gives_each_dpu_one_stage_and_a_layer_slice() -> None:
    """tp_width=1：8 stage × 1 台，32 层每台 4 层，段内不切。"""
    strategy = ShardStrategy(name="pp8", num_dpus=8, num_stages=8)

    assert strategy.tp_width == 1
    assert strategy.kind == "pipeline"
    assert strategy.dpus_of_layer(0, 32) == (0,)
    assert strategy.dpus_of_layer(3, 32) == (0,)
    assert strategy.dpus_of_layer(4, 32) == (1,)
    assert strategy.dpus_of_layer(31, 32) == (7,)
    assert strategy.layers_of_dpu(0, 32) == [0, 1, 2, 3]
    assert strategy.layers_of_dpu(7, 32) == [28, 29, 30, 31]


def test_hybrid_splits_layers_across_stages_and_tensors_within() -> None:
    """2 stage × tp4：前 16 层在 dpu0-3、后 16 层在 dpu4-7。"""
    strategy = ShardStrategy(name="tp4_pp2", num_dpus=8, num_stages=2)

    assert (strategy.tp_width, strategy.kind) == (4, "hybrid")
    assert strategy.dpus_of_layer(0, 32) == (0, 1, 2, 3)
    assert strategy.dpus_of_layer(15, 32) == (0, 1, 2, 3)
    assert strategy.dpus_of_layer(16, 32) == (4, 5, 6, 7)
    assert strategy.layers_of_dpu(2, 32) == list(range(16))
    assert strategy.layers_of_dpu(6, 32) == list(range(16, 32))


@pytest.mark.parametrize("num_stages", [1, 2, 4, 8])
def test_partition_is_exact_every_dpu_and_layer_belongs_to_one_stage(num_stages: int) -> None:
    """划分自洽性：stage 的 DPU 集合两两不交且并集为全体；层同理。"""
    strategy = ShardStrategy(name="s", num_dpus=8, num_stages=num_stages)

    seen_dpus: list[int] = []
    seen_layers: list[int] = []
    for stage in range(num_stages):
        seen_dpus.extend(strategy.dpus_of_stage(stage))
        seen_layers.extend(strategy.layers_of_stage(stage, 32))
    assert sorted(seen_dpus) == list(range(8))  # 无重复、无遗漏
    assert sorted(seen_layers) == list(range(32))
    # DPU 与流水段的双向映射。
    for dpu_id in range(8):
        stage = strategy.stage_of_dpu(dpu_id)
        assert dpu_id in strategy.dpus_of_stage(stage)
        assert strategy.layers_of_dpu(dpu_id, 32) == strategy.layers_of_stage(stage, 32)


def test_physical_dpu_ids_are_grouped_in_given_order() -> None:
    """自定义物理编号：按 dpu_ids 的顺序切段，不重排。"""
    strategy = ShardStrategy(name="s", num_dpus=4, num_stages=2, dpu_ids=(5, 2, 7, 3))

    assert strategy.dpus_of_stage(0) == (5, 2)
    assert strategy.dpus_of_stage(1) == (7, 3)
    assert strategy.stage_of_dpu(2) == 0
    assert strategy.stage_of_dpu(7) == 1


@pytest.mark.parametrize("dpu_ids", [(), (2,), (2, 2), (2, -1)])
def test_invalid_physical_dpu_ids_are_rejected(dpu_ids: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="dpu_ids"):
        ShardStrategy(name="s", num_dpus=2, dpu_ids=dpu_ids)


def test_num_stages_must_divide_num_dpus() -> None:
    """每个 stage 必须持同样多的 DPU，否则各段算力不等、切分契约无从表达。"""
    with pytest.raises(ValueError, match="不能整除 num_dpus"):
        ShardStrategy(name="s", num_dpus=8, num_stages=3)


def test_num_stages_must_divide_num_layers() -> None:
    """层数不整除时各 stage 层数不等，权重/KV 分布随之不均——抛错不静默取整。"""
    strategy = ShardStrategy(name="s", num_dpus=8, num_stages=8)
    with pytest.raises(ValueError, match="不能被 num_stages"):
        strategy.stage_of_layer(0, 30)


@pytest.mark.parametrize("bad", [(-1, 32), (32, 32), (0, 0)])
def test_out_of_range_layer_is_rejected(bad: tuple[int, int]) -> None:
    strategy = ShardStrategy(name="s", num_dpus=8, num_stages=2)
    with pytest.raises(ValueError):
        strategy.stage_of_layer(*bad)


def test_out_of_range_stage_is_rejected() -> None:
    strategy = ShardStrategy(name="s", num_dpus=8, num_stages=2)
    with pytest.raises(ValueError, match="stage=2 越界"):
        strategy.dpus_of_stage(2)


# Llama 策略构造和枚举。


def test_llama_strategy_carries_the_megatron_pairing() -> None:
    strategy = llama_strategy(8, num_stages=2, **LLAMA2_7B)

    assert strategy.weight_rules == LLAMA_WEIGHT_RULES
    assert strategy.match("model.layers.0.self_attn.q_proj.weight") == "col"
    assert strategy.match("model.layers.0.self_attn.o_proj.weight") == "row"
    assert strategy.match("model.layers.0.input_layernorm.weight") is None


def test_llama_strategy_validates_contract_against_tp_width_not_num_dpus() -> None:
    """验证 Llama 维度按每个流水段的张量并行宽度校验。"""
    kwargs = dict(num_heads=4, num_kv_heads=4, intermediate_size=176, vocab_size=32000, num_layers=8)

    with pytest.raises(ValueError, match="num_heads=4 不能被 tp_width=8 整除"):
        llama_strategy(8, num_stages=1, **kwargs)

    strategy = llama_strategy(8, num_stages=4, **kwargs)
    assert strategy.tp_width == 2


def test_llama_strategy_rejects_non_power_of_two_tp_width() -> None:
    with pytest.raises(ValueError, match="tp_width=3 不是 2 的整数次幂"):
        llama_strategy(6, num_stages=2, **{**LLAMA2_7B, "num_layers": 32})


def test_llama_strategies_enumerates_four_points_for_llama2_7b() -> None:
    """真实 7B / 8 DPU 的搜索空间：tp8、tp4×pp2、tp2×pp4、tp1×pp8。"""
    strategies = llama_strategies(8, **LLAMA2_7B)

    assert [(s.num_stages, s.tp_width) for s in strategies] == [(1, 8), (2, 4), (4, 2), (8, 1)]
    assert [s.kind for s in strategies] == ["tensor", "hybrid", "hybrid", "pipeline"]
    assert [s.name for s in strategies] == ["tp8_pp1", "tp4_pp2", "tp2_pp4", "tp1_pp8"]


def test_llama_strategies_skips_contract_violating_stage_counts() -> None:
    """验证策略枚举排除无法均分层数的流水段数量。"""
    wide = llama_strategies(4, num_heads=4, num_kv_heads=4, intermediate_size=128,
                            vocab_size=128, num_layers=4)
    assert [s.num_stages for s in wide] == [1, 2, 4]

    narrow = llama_strategies(4, num_heads=4, num_kv_heads=4, intermediate_size=128,
                              vocab_size=128, num_layers=2)
    assert [s.num_stages for s in narrow] == [1, 2]  # 2 层不能均分到 4 个流水段。


def test_llama_strategies_raises_when_nothing_is_legal() -> None:
    with pytest.raises(ValueError, match="没有任何契约内的切分策略"):
        llama_strategies(8, num_heads=1, num_kv_heads=1, intermediate_size=1,
                         vocab_size=1, num_layers=1)


def test_format_strategy_is_readable() -> None:
    text = format_strategy(llama_strategy(8, num_stages=4, **LLAMA2_7B), 32)

    assert "4 stage × tp2" in text
    assert "stage0: dpus=[0, 1] layers=[0..7]（8 层）" in text
    assert "stage3: dpus=[6, 7] layers=[24..31]（8 层）" in text
