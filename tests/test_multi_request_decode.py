"""验证多请求并发解码：两路生成的 token 与单独运行逐 token 一致。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, run_decode_loop, run_one_step
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
MAX_SEQ = 16
NUM_SLOTS = 2
MAX_NEW_TOKENS = 3

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128, num_layers=NUM_LAYERS,
)

PROMPT_A = [1, 2, 3, 4]
PROMPT_B = [5, 6, 7, 8]


def _greedy(logits):
    return int(np.argmax(logits))


@pytest.fixture(scope="module")
def small_model():
    # 不要改全局梯度开关：会让后续 torch.export 导出不同的图。
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


def _backend_with_weights(compiled, hardware) -> NumpyBackend:
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    return backend


def _compile(model, hw, hardware, num_slots):
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    return compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware, num_slots=num_slots,
    )


def test_two_requests_match_individual_runs(small_model, hw_and_hardware) -> None:
    """同时提交 2 个不同 prompt，两路生成结果与单独运行逐 token 一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, NUM_SLOTS)
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS)
    table.add_request("a", PROMPT_A)
    table.add_request("b", PROMPT_B)

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    generated = {s.request_id: list(s.generated) for s in table.active_slots()}
    assert len(generated["a"]) == MAX_NEW_TOKENS
    assert len(generated["b"]) == MAX_NEW_TOKENS

    # 每次 run_one_step 每个请求只产生一个 token（与 vLLM 一步一 token 对齐）。
    assert all(len(s.generated) == MAX_NEW_TOKENS for s in table.active_slots())

    # 单独运行同一 prompt 作为参考（单 slot 路径）。
    for request_id, prompt in (("a", PROMPT_A), ("b", PROMPT_B)):
        compiled.state.valid_len = 0
        ref = run_decode_loop(
            compiled.prefill.plan, compiled.decode.plan, backend,
            prompt_ids=torch.tensor(prompt), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, state=compiled.state, sample_fn=_greedy,
            prefill_output_cmd_id=compiled.prefill.output_cmd_id,
            decode_output_cmd_id=compiled.decode.output_cmd_id,
            causal_mask_of=causal_mask_of,
        )
        assert generated[request_id] == ref, (
            f"请求 {request_id} 多路生成 {generated[request_id]} 与单独运行 {ref} 不一致")


def test_freed_slot_can_be_reused(small_model, hw_and_hardware) -> None:
    """slot 归还后可被新请求复用，且复用后的结果仍与单独运行一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, NUM_SLOTS)
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS)
    slot = table.add_request("a", PROMPT_A)
    used_slot = slot.slot_id

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    first = list(slot.generated)

    table.free_slot("a")
    reused = table.add_request("c", PROMPT_B)
    assert reused.slot_id == used_slot, "释放的 slot 应被重新分配"
    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    second = list(reused.generated)

    compiled.state.valid_len = 0
    ref = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=torch.tensor(PROMPT_B), max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, state=compiled.state, sample_fn=_greedy,
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )
    assert second == ref, f"复用 slot 后生成 {second} 与单独运行 {ref} 不一致"


def test_request_table_rejects_overflow() -> None:
    table = RequestTable(1)
    table.add_request("a", PROMPT_A)
    with pytest.raises(RuntimeError, match="slot"):
        table.add_request("b", PROMPT_B)
