"""主机侧采样：对齐 vLLM 0.14 的 Sampler 语义。

vLLM 的采样器在 GPU 上批量执行；PIM 的 logits 在主机上产生，所以这里用
PyTorch CPU 复刻同一套公式：惩罚 -> 贪心/温度 -> top-k/top-p -> 随机采样。
logprobs 按 vLLM 默认的 `raw_logprobs` 模式，从惩罚和温度之前的 logits 计算。

只实现 PIM 支持的参数；其余参数在 `SamplingConfig.from_vllm` 里显式拒绝，
不做静默降级。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

# vLLM 用这个阈值区分贪心与随机采样。
_SAMPLING_EPS = 1e-5


@dataclass(frozen=True)
class SamplingConfig:
    """一次请求用到的采样参数（与 vLLM 的 SamplingParams 解耦）。"""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: int | None = None

    @classmethod
    def from_vllm(cls, params) -> "SamplingConfig":
        """把 vLLM 的 SamplingParams 转成采样配置，并拒绝未实现的参数。"""
        _reject_unsupported(params)
        return cls(
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            presence_penalty=float(params.presence_penalty),
            frequency_penalty=float(params.frequency_penalty),
            repetition_penalty=float(params.repetition_penalty),
            seed=None if params.seed is None else int(params.seed),
            logprobs=None if params.logprobs is None else int(params.logprobs),
        )


@dataclass(frozen=True)
class SampleResult:
    """一次采样的结果；未请求 logprobs 时后三项为 None。"""

    token_id: int
    logprob_token_ids: np.ndarray | None = None
    logprobs: np.ndarray | None = None
    rank: int | None = None


class RequestSampler:
    """单个请求的采样器；随机数状态跨步保留，显式 seed 时结果可复现。"""

    def __init__(self, config: SamplingConfig) -> None:
        self.config = config
        seed = config.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(
        self,
        logits: np.ndarray,
        prompt_ids: list[int],
        output_ids: list[int],
        logprobs_width: int | None = None,
    ) -> SampleResult:
        """从一行 logits 采样一个 token。

        `logprobs_width` 是本步批内最大的 top-k 宽度；vLLM 对同批请求统一用
        这个宽度，所以这里也统一返回 `width + 1` 项（采样 token + top-k）。
        """
        raw = torch.from_numpy(np.asarray(logits, dtype=np.float32)).clone()
        # logprobs 用的是惩罚与温度之前的原始 logits（vLLM 默认 raw_logprobs）。
        raw_logprobs = None
        if self.config.logprobs is not None:
            raw_logprobs = torch.log_softmax(raw, dim=-1)

        scores = raw.clone()
        self._apply_penalties(scores, prompt_ids, output_ids)

        # 温度低于阈值时按贪心处理，与 vLLM 的 all_greedy 分支一致。
        if self.config.temperature < _SAMPLING_EPS:
            token = int(torch.argmax(scores))
        else:
            scores = scores / self.config.temperature
            scores = self._apply_top_k_top_p(scores)
            probs = torch.softmax(scores, dim=-1)
            q = torch.empty_like(probs).exponential_(generator=self._generator)
            token = int(torch.argmax(probs / q))

        if raw_logprobs is None or logprobs_width is None:
            return SampleResult(token)
        return self._gather_logprobs(raw_logprobs, token, logprobs_width)

    def _apply_penalties(
        self, logits: torch.Tensor, prompt_ids: list[int], output_ids: list[int]
    ) -> None:
        """原地施加 repetition / frequency / presence 惩罚。"""
        config = self.config
        if (
            config.repetition_penalty == 1.0
            and config.frequency_penalty == 0.0
            and config.presence_penalty == 0.0
        ):
            return
        vocab = logits.numel()
        prompt_mask = torch.zeros(vocab, dtype=torch.bool)
        output_mask = torch.zeros(vocab, dtype=torch.bool)
        output_counts = torch.zeros(vocab, dtype=torch.float32)
        if prompt_ids:
            prompt_mask[torch.tensor(prompt_ids, dtype=torch.long)] = True
        if output_ids:
            output_mask[torch.tensor(output_ids, dtype=torch.long)] = True
            output_counts.scatter_add_(
                0, torch.tensor(output_ids, dtype=torch.long),
                torch.ones(len(output_ids), dtype=torch.float32),
            )
        if config.repetition_penalty != 1.0:
            seen = prompt_mask | output_mask
            scale = torch.where(
                logits > 0,
                torch.tensor(1.0 / config.repetition_penalty),
                torch.tensor(config.repetition_penalty),
            )
            logits[seen] *= scale[seen]
        if config.frequency_penalty != 0.0:
            logits -= config.frequency_penalty * output_counts
        if config.presence_penalty != 0.0:
            logits -= config.presence_penalty * output_mask.to(torch.float32)

    def _apply_top_k_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """按 vLLM 的 apply_top_k_top_p 实现裁剪，返回新张量。"""
        top_k = self.config.top_k
        top_p = self.config.top_p
        if top_p >= 1.0 and top_k <= 0:
            return logits
        sorted_logits, indices = torch.sort(logits, descending=False)
        if 0 < top_k < sorted_logits.numel():
            kth = sorted_logits[-top_k]
            sorted_logits[sorted_logits < kth] = -float("inf")
        if top_p < 1.0:
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative <= 1.0 - top_p
            mask[-1] = False
            sorted_logits[mask] = -float("inf")
        return sorted_logits.scatter(0, indices, sorted_logits)

    @staticmethod
    def _gather_logprobs(
        raw_logprobs: torch.Tensor, token: int, width: int
    ) -> SampleResult:
        """取采样 token 与 top-k 的 logprob，形状与 vLLM 的 LogprobsLists 一致。"""
        width = min(width, raw_logprobs.numel())
        topk_values, topk_indices = torch.topk(raw_logprobs, width)
        token_logprob = raw_logprobs[token]
        rank = int((raw_logprobs > token_logprob).sum())
        token_ids = np.empty(width + 1, dtype=np.int32)
        values = np.empty(width + 1, dtype=np.float32)
        token_ids[0] = token
        values[0] = float(token_logprob)
        token_ids[1:] = topk_indices.numpy()
        values[1:] = topk_values.numpy()
        return SampleResult(token, token_ids, values, rank)


def _reject_unsupported(params) -> None:
    """拒绝 PIM 没有实现的采样能力，避免静默产生不符合语义的输出。"""
    unsupported = {
        "min_p": params.min_p > 0,
        "prompt_logprobs": params.prompt_logprobs is not None,
        "flat_logprobs": params.flat_logprobs,
        "logit_bias": params.logit_bias is not None,
        "allowed_token_ids": params.allowed_token_ids is not None,
        "bad_words": bool(params.bad_words),
        "logits_processors": params.logits_processors is not None,
        "structured_outputs": params.structured_outputs is not None,
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(
            f"PIM 后端不支持采样参数 {', '.join(enabled)}；"
            "请去掉这些参数，或改用 vLLM 原生 GPU 后端"
        )
    if params.logprobs is not None and int(params.logprobs) < 0:
        raise ValueError(
            "PIM 后端只支持 logprobs >= 0，不支持 logprobs=-1（返回全词表）")
