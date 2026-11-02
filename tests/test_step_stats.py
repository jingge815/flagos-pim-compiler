"""验证每步运行时指标的聚合与打印，以及 worker 上的开关接入。"""

from __future__ import annotations

import atexit
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.metrics import StepStats


def _slot(blocks: list[int]) -> SimpleNamespace:
    """伪造只带块号记录的请求槽。"""
    return SimpleNamespace(blocks=SimpleNamespace(slots=blocks))


def test_record_aggregates_waves_and_tokens() -> None:
    """wave 数按宽度向上取整，最宽 wave 取历史最大值。"""
    stats = StepStats(pool_total=9)
    stats.record({"a": 1, "b": 1}, width=2, slots=[])
    stats.record({"a": 1, "b": 1, "c": 1}, width=2, slots=[])

    assert (stats.steps, stats.requests, stats.tokens) == (2, 5, 5)
    assert (stats.waves, stats.widest_wave) == (3, 2)
    assert stats.last_requests == 3


def test_record_counts_shared_blocks_once() -> None:
    """两个请求共用块号时，槽池占用按去重后的块数算。"""
    stats = StepStats(pool_total=4)
    stats.record({"a": 1, "b": 1}, width=2,
                 slots=[_slot([0, 1]), _slot([1, 2])])

    assert stats.pool_used == 3


def test_format_lists_step_fields() -> None:
    """日志行含步数、wave、槽池与注意力路径。"""
    stats = StepStats(pool_total=9)
    stats.record({"a": 1}, width=1, slots=[_slot([0])])
    line = stats.format()

    assert "步 1" in line and "wave 1（最宽 1）" in line
    assert "槽池 1/9" in line and "paged_attn" in line


def test_vllm_reports_stats_and_profile(tmp_path, monkeypatch, caplog) -> None:
    """开启指标后每步打印一行；开启 profile 能拿到调度记录与 DPU 命令记录。"""
    from backend.vllm_executor import PIMExecutor
    from vllm import LLM, SamplingParams

    seq_len = 16
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("PIM_LOG_STEPS", "1")
    monkeypatch.setenv("PIM_PROFILE", "1")
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(seq_len))
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny-llama"
    LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=seq_len,
    )).eval().half().save_pretrained(model_dir)
    caplog.set_level(logging.INFO)

    llm = LLM(
        model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
        max_model_len=seq_len, max_num_seqs=2, max_num_batched_tokens=seq_len,
        gpu_memory_utilization=0.2, distributed_executor_backend=PIMExecutor)
    try:
        llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}],
                     SamplingParams(max_tokens=2, temperature=0.0))
        worker = llm.llm_engine.model_executor.driver_worker
        atexit.unregister(worker._dump_profile)

        stats = worker.stats
        assert stats.steps >= 2 and stats.requests == stats.steps
        assert 0 < stats.pool_used <= stats.pool_total
        assert "槽池" in caplog.text and "paged_attn" in caplog.text
        assert worker._profiler.steps, "请求级调度没有采到步"
        assert worker._backend_profiler.records, "DPU 命令没有采到"
    finally:
        llm.llm_engine.engine_core.shutdown()
