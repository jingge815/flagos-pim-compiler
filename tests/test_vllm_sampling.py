"""主机侧采样与 vLLM 语义对齐的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from backend.sampling import RequestSampler, SamplingConfig


def _sampler(**kwargs) -> RequestSampler:
    return RequestSampler(SamplingConfig(**kwargs))


def test_greedy_matches_argmax() -> None:
    """温度 0 与 argmax 逐元素一致。"""
    logits = np.array([0.1, 2.5, -1.0, 1.2], dtype=np.float16)
    for _ in range(10):
        result = _sampler(temperature=0.0).sample(logits, [], [])
        assert result.token_id == int(np.argmax(logits))


def test_seed_reproducible_across_steps() -> None:
    """同一个 seed 的多步采样逐 token 相同。"""
    logits = np.array([1.0, 1.1, 0.9, 1.0], dtype=np.float32)

    def run(seed: int) -> list[int]:
        sampler = _sampler(temperature=1.0, seed=seed)
        return [sampler.sample(logits, [], []).token_id for _ in range(16)]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_top_k_one_is_greedy() -> None:
    """top_k=1 时只会抽到最大值。"""
    logits = np.array([0.0, 5.0, 1.0], dtype=np.float32)
    sampler = _sampler(temperature=1.0, top_k=1, seed=1)
    for _ in range(10):
        assert sampler.sample(logits, [], []).token_id == 1


def test_top_p_filters_tail() -> None:
    """top_p 截断后，概率极低的 token 不会被抽到。"""
    logits = np.array([0.0, 10.0, 0.0], dtype=np.float32)
    sampler = _sampler(temperature=1.0, top_p=0.1, seed=1)
    for _ in range(20):
        assert sampler.sample(logits, [], []).token_id == 1


def test_repetition_penalty_positive_and_negative() -> None:
    """正 logit 除以惩罚、负 logit 乘以惩罚。"""
    sampler = _sampler(temperature=0.0, repetition_penalty=2.0)
    logits = torch.tensor([2.0, -2.0, 1.0])
    sampler._apply_penalties(logits, [0, 1], [])
    assert logits[0].item() == pytest.approx(1.0)
    assert logits[1].item() == pytest.approx(-4.0)
    assert logits[2].item() == pytest.approx(1.0)


def test_frequency_and_presence_penalty() -> None:
    """频率按出现次数扣分，存在性按是否出现扣分，且只看已生成 token。"""
    sampler = _sampler(
        temperature=0.0, frequency_penalty=0.5, presence_penalty=0.25)
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    sampler._apply_penalties(logits, [0], [0, 0, 2])
    assert logits[0].item() == pytest.approx(1.0 - 0.5 * 2 - 0.25)
    assert logits[1].item() == pytest.approx(1.0)
    assert logits[2].item() == pytest.approx(1.0 - 0.5 - 0.25)
    # prompt 里的 token 只受 repetition 影响，不受 frequency/presence 影响。
    assert logits[3].item() == pytest.approx(1.0)


def test_logprobs_use_raw_logits() -> None:
    """logprobs 取自惩罚/温度之前的原始 logits，且首项是采样 token。"""
    logits = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    sampler = _sampler(temperature=0.0, logprobs=2, presence_penalty=1.0)
    result = sampler.sample(logits, [], [], logprobs_width=2)
    raw = torch.log_softmax(torch.from_numpy(logits), dim=-1)
    assert result.token_id == 1
    assert result.logprob_token_ids[0] == 1
    assert result.logprobs[0] == pytest.approx(float(raw[1]), abs=1e-5)
    assert set(result.logprob_token_ids[1:].tolist()) == {1, 2}
    assert result.rank == 0


def test_logprobs_width_follows_batch_max() -> None:
    """请求只写 logprobs=1 时，仍按批内最大宽度返回，与 vLLM 一致。"""
    logits = np.arange(6, dtype=np.float32)
    sampler = _sampler(temperature=0.0, logprobs=1)
    result = sampler.sample(logits, [], [], logprobs_width=3)
    assert len(result.logprob_token_ids) == 4


def test_config_from_vllm_roundtrip() -> None:
    """SamplingParams 转配置后字段一致。"""
    params = SamplingParams(
        temperature=0.7, top_p=0.9, top_k=10, seed=5, logprobs=2)
    config = SamplingConfig.from_vllm(params)
    assert config.temperature == pytest.approx(0.7)
    assert config.top_p == pytest.approx(0.9)
    assert config.top_k == 10
    assert config.seed == 5
    assert config.logprobs == 2


@pytest.mark.parametrize("params", [
    SamplingParams(min_p=0.5),
    SamplingParams(logit_bias={1: 1.0}),
    SamplingParams(allowed_token_ids=[1]),
    SamplingParams(bad_words=["x"]),
    SamplingParams(prompt_logprobs=1),
    SamplingParams(flat_logprobs=True),
    SamplingParams(logprobs=-1),
])
def test_unsupported_params_rejected(params: SamplingParams) -> None:
    """未实现的采样能力必须显式报错，不能静默降级。"""
    with pytest.raises(ValueError, match="PIM 后端"):
        SamplingConfig.from_vllm(params)
