"""验证 prefill 分桶：短桶的输出与单张最长图逐 token 一致。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import (
    causal_mask_of,
    compile_llama2,
    default_prefill_buckets,
    load_weights,
)
from runtime.executor import RequestTable, prefill_bucket, run_one_step, step_request
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
MAX_SEQ = 16
NUM_SLOTS = 2
MAX_NEW_TOKENS = 3
BUCKETS = (2, 4, 8)

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128, num_layers=NUM_LAYERS,
)


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**MODEL_KWARGS)).eval().to(torch.float16)


@pytest.fixture(scope="module")
def hw_and_hardware():
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    return hw, hardware


def _compile(small_model, hw_and_hardware, buckets=()) -> object:
    hw, hardware = hw_and_hardware
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    return compile_llama2(
        small_model, strategy, prefill_seq_len=MAX_SEQ,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        num_request_slots=NUM_SLOTS, kv_slot_tokens=MAX_SEQ,
        prefill_seq_lens=buckets,
    )


def _backend(compiled, hardware) -> NumpyBackend:
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=compiled.num_parallel_slots,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    return backend


def _run(compiled, backend, prompt: list[int]) -> list[int]:
    """跑一次 prefill + 若干步解码，返回生成的 token。"""
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt)
    return [step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(MAX_NEW_TOKENS)]


@pytest.fixture(scope="module")
def baseline(small_model, hw_and_hardware):
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw_and_hardware)
    return compiled, _backend(compiled, hardware)


@pytest.fixture(scope="module")
def bucketed(small_model, hw_and_hardware):
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw_and_hardware, BUCKETS)
    return compiled, _backend(compiled, hardware)


def test_bucket_table_is_sorted_and_ends_at_max(bucketed) -> None:
    """桶表升序、末项是最长桶，且默认桶按 128/512/2048 选取。"""
    compiled, _ = bucketed
    assert compiled.prefill_seq_lens == (*BUCKETS, MAX_SEQ)
    assert default_prefill_buckets(4096) == (128, 512, 2048, 4096)
    assert default_prefill_buckets(64) == (64,)


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 4, 5, 8, 9, MAX_SEQ])
def test_bucket_choice(prompt_len: int, bucketed) -> None:
    """按最短可容纳原则选桶。"""
    compiled, _ = bucketed
    expected = next(length for length in (*BUCKETS, MAX_SEQ) if prompt_len <= length)
    assert prefill_bucket(compiled, prompt_len) == expected


def test_prompt_longer_than_longest_bucket_raises(bucketed) -> None:
    """超过最长桶直接报错，不做隐式截断。"""
    compiled, _ = bucketed
    with pytest.raises(ValueError, match="超过最长 prefill 桶"):
        prefill_bucket(compiled, MAX_SEQ + 1)


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 4, 5, 8, 9, MAX_SEQ - MAX_NEW_TOKENS])
def test_bucket_tokens_match_baseline(prompt_len, baseline, bucketed) -> None:
    """任意长度 prompt 的分桶输出与「单张最长图」逐 token 一致。"""
    base_compiled, base_backend = baseline
    buck_compiled, buck_backend = bucketed
    prompt = [1 + i for i in range(prompt_len)]
    assert _run(buck_compiled, buck_backend, prompt) == _run(
        base_compiled, base_backend, prompt)


def test_buckets_share_weight_and_kv_layout(baseline, bucketed) -> None:
    """各桶共用权重区、KV 区与激活窗口步长，只多出几张图。"""
    base_compiled, _ = baseline
    buck_compiled, _ = bucketed
    assert buck_compiled.act_stride == base_compiled.act_stride
    for dpu_id, plan in buck_compiled.mem_plans.items():
        base = base_compiled.mem_plans[dpu_id]
        assert plan.weight == base.weight
        assert plan.kv_base == base.kv_base
        assert plan.act_base == base.act_base


def test_prefill_uses_fused_kernel_and_matches_host_path(monkeypatch, bucketed) -> None:
    """标准因果掩码的 prefill 走融合内核，生成 token 与主机两遍路径一致。"""
    import runtime.kernels as km

    compiled, backend = bucketed
    prompt = [1, 2, 3, 4, 5]
    km.reset_route_counts()
    fused = _run(compiled, backend, prompt)
    routes = km.route_counts()
    assert routes.get(("paged_attn", "hit"), 0) > 0, "prefill 没有走融合内核"
    assert routes.get(("paged_attn", "fallback"), 0) == 0, routes

    monkeypatch.setattr(km, "_compiled_paged_attn", lambda *a, **kw: None)
    host = _run(compiled, backend, prompt)
    assert fused == host, "融合内核与主机路径的生成 token 不一致"


def test_windows_and_buckets_combined(bucketed) -> None:
    """同一批里不同桶的请求分到不同窗口，输出与单独执行一致。"""
    compiled, backend = bucketed
    prompts = [[1, 2], [3, 4, 5, 6, 7, 8]]
    expected = []
    for prompt in prompts:
        table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
        slot = table.add_request("r", prompt)
        expected.append([
            step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(MAX_NEW_TOKENS)])

    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    slots = [table.add_request(f"r{index}", prompt)
             for index, prompt in enumerate(prompts)]
    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of)
    assert [list(slot.generated) for slot in slots] == expected
