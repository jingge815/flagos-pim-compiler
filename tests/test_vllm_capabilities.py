"""PIM 后端启动期与请求期能力矩阵的单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from backend.vllm_worker import (
    validate_engine_config,
    validate_new_request,
    validate_scheduler_output,
)

PREFILL_SEQ_LEN = 4


def _configs(**overrides) -> dict:
    """构造一组 PIM 支持的引擎配置，测试按需覆盖单项。"""
    values = {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1),
        "lora_config": None,
        "speculative_config": None,
        "model_config": SimpleNamespace(
            dtype=torch.float16,
            quantization=None,
            hf_config=SimpleNamespace(model_type="llama"),
        ),
        "scheduler_config": SimpleNamespace(
            enable_chunked_prefill=True,
            # 一个 prefill 分块不超过一张 prefill 图的长度。
            max_num_batched_tokens=PREFILL_SEQ_LEN,
        ),
        "prefill_seq_len": PREFILL_SEQ_LEN,
    }
    values.update(overrides)
    return values


def test_supported_config_passes() -> None:
    """能力矩阵全绿时不抛异常。"""
    validate_engine_config(**_configs())


@pytest.mark.parametrize(
    "overrides, param",
    [
        ({"parallel_config": SimpleNamespace(
            tensor_parallel_size=2, pipeline_parallel_size=1)},
         "tensor_parallel_size"),
        ({"parallel_config": SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=2)},
         "pipeline_parallel_size"),
        ({"lora_config": SimpleNamespace(max_loras=1)}, "lora_config"),
        ({"speculative_config": SimpleNamespace(num_speculative_tokens=4)},
         "speculative_config"),
        ({"model_config": SimpleNamespace(
            dtype=torch.float16, quantization="fp8",
            hf_config=SimpleNamespace(model_type="llama"))}, "quantization"),
        ({"model_config": SimpleNamespace(
            dtype=torch.float32, quantization=None,
            hf_config=SimpleNamespace(model_type="llama"))}, "dtype"),
        ({"model_config": SimpleNamespace(
            dtype=torch.float16, quantization=None,
            hf_config=SimpleNamespace(model_type="qwen2"))}, "model_type"),
        ({"scheduler_config": SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_batched_tokens=PREFILL_SEQ_LEN + 1)},
         "max_num_batched_tokens"),
    ],
)
def test_unsupported_engine_config_rejected(overrides: dict, param: str) -> None:
    """每项不支持的能力都抛 ValueError，且消息点名参数。"""
    with pytest.raises(ValueError, match=param):
        validate_engine_config(**_configs(**overrides))


def test_tensor_parallel_message_mentions_dpu() -> None:
    """TP>1 的错误提示要说明 PIM 内部已按 DPU 并行。"""
    configs = _configs(parallel_config=SimpleNamespace(
        tensor_parallel_size=2, pipeline_parallel_size=1))
    with pytest.raises(ValueError, match="DPU"):
        validate_engine_config(**configs)


def test_disabled_chunked_prefill_skips_budget_check() -> None:
    """关掉分块 prefill 后，预算不足也不再报错（由 chunked 检查以外的地方兜）。"""
    configs = _configs(scheduler_config=SimpleNamespace(
        enable_chunked_prefill=False, max_num_batched_tokens=1))
    validate_engine_config(**configs)


def _req(**overrides) -> SimpleNamespace:
    """构造一个合法的调度新请求。"""
    values = {
        "req_id": "req-0",
        "prompt_token_ids": [1, 2, 3],
        "sampling_params": object(),
        "lora_request": None,
        "mm_features": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_supported_request_passes() -> None:
    """纯文本、无 LoRA 的请求通过校验。"""
    validate_new_request(_req())


@pytest.mark.parametrize(
    "overrides, param",
    [
        ({"lora_request": object()}, "lora_request"),
        ({"prompt_token_ids": None}, "prompt_token_ids"),
        ({"mm_features": [object()]}, "mm_features"),
    ],
)
def test_unsupported_request_rejected(overrides: dict, param: str) -> None:
    """不支持的请求输入抛 ValueError，且消息点名参数。"""
    with pytest.raises(ValueError, match=param):
        validate_new_request(_req(**overrides))


def _scheduled(**overrides) -> SimpleNamespace:
    """构造一份 SchedulerOutput 里与本校验相关的字段。"""
    values = {
        "has_structured_output_requests": False,
        "pending_structured_output_tokens": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_plain_scheduler_output_passes() -> None:
    """没有结构化输出请求时通过。"""
    validate_scheduler_output(_scheduled())


@pytest.mark.parametrize(
    "overrides",
    [
        {"has_structured_output_requests": True},
        {"pending_structured_output_tokens": True},
    ],
)
def test_structured_output_rejected(overrides: dict) -> None:
    """结构化输出请求在第一步执行前失败。"""
    with pytest.raises(ValueError, match="structured_output"):
        validate_scheduler_output(_scheduled(**overrides))
