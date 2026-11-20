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
from runtime.executor import (
    KVBlockTable, RequestSlot, RequestTable, available_windows,
    kv_blocks_needed, plan_wave, run_decode_loop, run_one_step,
)
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
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
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


def _compile(model, hw, hardware, num_request_slots, kv_slot_tokens=MAX_SEQ,
             dynamic=False, num_parallel_slots=None, act_pool_slots=None):
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    return compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        num_request_slots=num_request_slots, kv_slot_tokens=kv_slot_tokens,
        dynamic=dynamic, num_parallel_slots=num_parallel_slots,
        act_pool_slots=act_pool_slots,
    )


def _mixed_table(compiled, backend, prompts: dict) -> RequestTable:
    """建一张 4 请求表：前两个先跑一步进入 decode，后两个保持未 prefill。"""
    table = RequestTable(4, compiled.kv_pool_slots)
    for request_id in ("a", "b"):
        table.add_request(request_id, prompts[request_id], eos_id=-1)
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    for request_id in ("c", "d"):
        table.add_request(request_id, prompts[request_id], eos_id=-1)
    return table


def _run_steps(compiled, backend, table, **kwargs) -> None:
    """反复执行整步，直到每个请求都生成满 MAX_NEW_TOKENS 个 token。"""
    while any(len(s.generated) < MAX_NEW_TOKENS for s in table.active_slots()):
        run_one_step(compiled, backend, table, causal_mask_of,
                     sample_fn=_greedy, **kwargs)


def _assert_match_individual(compiled, backend, table, prompts: dict) -> None:
    """逐请求对照单独运行（贪心）的结果，前 MAX_NEW_TOKENS 个 token 须一致。"""
    for slot in table.active_slots():
        ref = run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(prompts[slot.request_id]),
            max_new_tokens=MAX_NEW_TOKENS, eos_id=-1, sample_fn=_greedy,
            causal_mask_of=causal_mask_of,
        )
        assert list(slot.generated)[:MAX_NEW_TOKENS] == ref, (
            f"请求 {slot.request_id} 混批生成 {list(slot.generated)} "
            f"与单独运行 {ref} 不一致")


def test_two_requests_match_individual_runs(small_model, hw_and_hardware) -> None:
    """同时提交 2 个不同 prompt，两路生成结果与单独运行逐 token 一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, NUM_SLOTS)
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
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
        ref = run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(prompt), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
        )
        assert generated[request_id] == ref, (
            f"请求 {request_id} 多路生成 {generated[request_id]} 与单独运行 {ref} 不一致")


def test_two_requests_wave_match_individual_runs_dynamic(
        small_model, hw_and_hardware, monkeypatch) -> None:
    """动态编译 × 多请求 wave：两路并发绑定各自 T，仍与单独运行逐 token 一致。"""
    monkeypatch.setenv("PIM_DYN_SHAPE", "1")
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, NUM_SLOTS, dynamic=True)
    backend = _backend_with_weights(compiled, hardware)
    # 两个 prompt 长度不同，同一 wave 的两条泳道并发绑定不同 T。
    prompts = {"a": PROMPT_A, "b": PROMPT_B[:-1]}
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    for request_id, prompt in prompts.items():
        table.add_request(request_id, prompt)

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    generated = {s.request_id: list(s.generated) for s in table.active_slots()}
    for request_id, prompt in prompts.items():
        ref = run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(prompt), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
        )
        assert generated[request_id] == ref, (
            f"请求 {request_id} 动态 wave 生成 {generated[request_id]} "
            f"与单独运行 {ref} 不一致")


def test_freed_slot_can_be_reused(small_model, hw_and_hardware) -> None:
    """slot 归还后可被新请求复用，且复用后的结果仍与单独运行一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, NUM_SLOTS)
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    slot = table.add_request("a", PROMPT_A)
    used_slot = slot.request_slot

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    first = list(slot.generated)

    table.free_slot("a")
    reused = table.add_request("c", PROMPT_B)
    assert reused.request_slot == used_slot, "释放的请求槽应被重新分配"
    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    second = list(reused.generated)

    ref = run_decode_loop(
        compiled, backend,
        prompt_ids=torch.tensor(PROMPT_B), max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
    )
    assert second == ref, f"复用请求槽后生成 {second} 与单独运行 {ref} 不一致"


def test_multi_block_decode_matches_single_block(small_model, hw_and_hardware) -> None:
    """slot_tokens=2 的多块页表解码与单块（slot_tokens=max_seq）逐 token 一致。"""
    hw, hardware = hw_and_hardware
    generated = []
    for slot_tokens in (MAX_SEQ, 2):
        compiled = _compile(small_model, hw, hardware, 1, kv_slot_tokens=slot_tokens)
        backend = _backend_with_weights(compiled, hardware)
        generated.append(run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(PROMPT_A), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
        ))
    assert generated[0] == generated[1]


def test_request_table_rejects_overflow() -> None:
    table = RequestTable(1, 1)
    table.add_request("a", PROMPT_A)
    with pytest.raises(RuntimeError, match="请求槽"):
        table.add_request("b", PROMPT_B)


def test_plan_wave_fcfs() -> None:
    """plan_wave 纯函数：保序连续切分、组宽不超上限、末组可短、非法宽度报错。"""
    items = ["a", "b", "c", "d", "e"]
    assert plan_wave(items, width=2) == [["a", "b"], ["c", "d"], ["e"]]
    assert plan_wave(items, width=5) == [items]
    assert plan_wave(items, width=99) == [items]
    assert plan_wave([], width=2) == []
    with pytest.raises(ValueError, match="宽度"):
        plan_wave(items, width=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_mixed_prefill_decode_wave_matches_individual(
        small_model, hw_and_hardware, monkeypatch, dynamic) -> None:
    """一个 wave 混排 prefill 与 decode 请求，各路结果仍与单独运行一致。"""
    if dynamic:
        monkeypatch.setenv("PIM_DYN_SHAPE", "1")
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, 4, dynamic=dynamic,
                        num_parallel_slots=4)
    backend = _backend_with_weights(compiled, hardware)
    # 两个新请求长度不同（3 与 4），prefill 桶按各自长度选。
    prompts = {"a": PROMPT_A, "b": PROMPT_B, "c": [1, 2, 3], "d": [9, 10, 11, 12]}
    table = _mixed_table(compiled, backend, prompts)
    _run_steps(compiled, backend, table, parallel_slots=4)
    _assert_match_individual(compiled, backend, table, prompts)


def test_run_one_step_external_waves(small_model, hw_and_hardware) -> None:
    """外部给定 waves（乱序、不等宽）原样执行，结果与 FCFS 一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, 4, num_parallel_slots=4)
    backend = _backend_with_weights(compiled, hardware)
    prompts = {"a": PROMPT_A, "b": PROMPT_B, "c": [1, 2, 3], "d": [9, 10, 11, 12]}
    table = _mixed_table(compiled, backend, prompts)
    slots = table.active_slots()
    waves = [slots[3:], slots[:3]]
    _run_steps(compiled, backend, table, waves=waves)
    _assert_match_individual(compiled, backend, table, prompts)

    # 空 waves 合法：整步空转，不产生 token。
    before = {s.request_id: list(s.generated) for s in table.active_slots()}
    run_one_step(compiled, backend, table, causal_mask_of,
                 sample_fn=_greedy, waves=[])
    assert {s.request_id: list(s.generated) for s in table.active_slots()} == before

    # wave 比并行窗口数宽直接报错，且报错发生在执行之前。
    with pytest.raises(ValueError, match="并行窗口"):
        run_one_step(compiled, backend, table, causal_mask_of,
                     sample_fn=_greedy, waves=[slots + slots])
    assert {s.request_id: list(s.generated) for s in table.active_slots()} == before


def test_available_windows_backpressure(small_model, hw_and_hardware) -> None:
    """接入前问容量：available_windows 归零就不再加请求，不会触发 RuntimeError。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, 2, kv_slot_tokens=2)
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(2, compiled.kv_pool_slots)
    assert available_windows(compiled, table) == 2

    admitted = []
    for request_id in ("a", "b", "c"):
        if available_windows(compiled, table) > 0:
            admitted.append(
                table.add_request(request_id, PROMPT_A, eos_id=-1).request_id)
    assert admitted == ["a", "b"]
    assert available_windows(compiled, table) == 0

    free_initial = table.kv_slots_free()
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    free_after_prefill = table.kv_slots_free()
    assert isinstance(free_after_prefill, int)
    assert free_after_prefill < free_initial
    for _ in range(2):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    assert table.kv_slots_free() < free_after_prefill  # decode 跨块边界继续长块

    # 块号外部托管的表没有池，剩余数返回 None。
    assert RequestTable(2).kv_slots_free() is None


@pytest.mark.parametrize("dynamic", [False, True])
def test_batch8_full_pool_no_conflict(
        small_model, hw_and_hardware, monkeypatch, dynamic) -> None:
    """8 请求全部驻留占满槽池，wave 并发互不串扰，结果与单独运行一致。"""
    if dynamic:
        monkeypatch.setenv("PIM_DYN_SHAPE", "1")
    hw, hardware = hw_and_hardware
    compiled = _compile(
        small_model, hw, hardware, 8, dynamic=dynamic,
        num_parallel_slots=None if dynamic else 8,
        act_pool_slots=8 if dynamic else None)
    backend = _backend_with_weights(compiled, hardware)
    prompts = [PROMPT_A, PROMPT_B] * 4
    table = RequestTable(8, compiled.kv_pool_slots)
    for index, prompt in enumerate(prompts):
        table.add_request(f"r{index}", prompt, eos_id=-1)
    assert available_windows(compiled, table) == 0

    _run_steps(compiled, backend, table)
    _assert_match_individual(
        compiled, backend, table,
        {f"r{index}": prompt for index, prompt in enumerate(prompts)})


def _pure_slot(request_id, *, valid_len=0, prompt=(), held_blocks=0,
               prefill_done=False) -> RequestSlot:
    """直接拼一个 RequestSlot，供 kv_blocks_needed 纯函数测试用。"""
    slot = RequestSlot(
        request_id=request_id, request_slot=0, blocks=KVBlockTable())
    slot.blocks.adopt(list(range(held_blocks)))
    slot.prompt_ids = list(prompt)
    slot.valid_len = valid_len
    slot.is_prefill_done = prefill_done
    return slot


def test_kv_blocks_needed_pure() -> None:
    """kv_blocks_needed：decode 跨界才算一块，未 prefill 按整段长块扣已持有。"""
    decode_cross = _pure_slot("x", valid_len=4, prompt=[1] * 4, prefill_done=True)
    decode_cross_next = _pure_slot("x", valid_len=8, prompt=[1] * 8, prefill_done=True)
    decode_within = _pure_slot("x", valid_len=6, prompt=[1] * 6, prefill_done=True)
    prefill_fresh = _pure_slot("x", prompt=[1] * 5)
    prefill_partial = _pure_slot("x", prompt=[1] * 8, held_blocks=1)
    prefill_covered = _pure_slot("x", prompt=[1] * 4, held_blocks=1)

    assert kv_blocks_needed([decode_cross], 4) == 1
    assert kv_blocks_needed([decode_cross_next], 4) == 1
    assert kv_blocks_needed([decode_within], 4) == 0
    assert kv_blocks_needed([prefill_fresh], 4) == 2
    assert kv_blocks_needed([prefill_partial], 4) == 1
    assert kv_blocks_needed([prefill_covered], 4) == 0
    # 混排 wave 求和：跨界 decode + 不跨界 decode + 新 prefill（3 token 一块）。
    assert kv_blocks_needed(
        [decode_cross, decode_within, _pure_slot("x", prompt=[1] * 3)], 4) == 2


def test_kv_pool_exhaustion_narrows_wave(small_model, hw_and_hardware) -> None:
    """池将耗尽时收窄 wave：装不下的请求晚一步，最终 token 仍与单独运行一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, 3, kv_slot_tokens=4,
                        num_parallel_slots=3)
    backend = _backend_with_weights(compiled, hardware)
    prompts = {"a": PROMPT_A, "b": PROMPT_B, "c": [9, 10, 11, 12]}
    # 5 块池：3 路 prefill 各占 1 块后剩 2 块，恰够 2 路跨界 decode。
    table = RequestTable(3, 5)
    for request_id, prompt in prompts.items():
        table.add_request(request_id, prompt, eos_id=-1)

    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    assert table.kv_slots_free() == 2

    # 三路同一步都跨块边界但只剩 2 块：收窄为 2 路，不抛 RuntimeError。
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    lengths = {s.request_id: len(s.generated) for s in table.active_slots()}
    assert lengths == {"a": 2, "b": 2, "c": 1}
    assert table.kv_slots_free() == 0

    # c 仍装不下（跨界要 1 块、空闲 0），继续被移出；a/b 不需新块照常跑满。
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    finished = {
        s.request_id: list(s.generated) for s in table.active_slots()}
    assert len(finished["a"]) == 3
    assert len(finished["b"]) == 3 and len(finished["c"]) == 1

    # 释放 a/b 的块后 c 追上，三路 token 都与单独运行一致。
    for request_id in ("a", "b"):
        table.free_slot(request_id)
    for _ in range(2):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    finished["c"] = list(table.get("c").generated)
    assert len(finished["c"]) == 3
    for request_id, prompt in prompts.items():
        ref = run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(prompt), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
        )
        assert finished[request_id] == ref, (
            f"请求 {request_id} 收窄后生成 {finished[request_id]} "
            f"与单独运行 {ref} 不一致")


def test_external_blocks_skip_narrowing(small_model, hw_and_hardware) -> None:
    """块号外部托管（无池）时不收窄：跨界步也全员执行，结果与单独运行一致。"""
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, 2, kv_slot_tokens=4,
                        num_parallel_slots=2)
    backend = _backend_with_weights(compiled, hardware)
    prompts = {"a": PROMPT_A, "b": PROMPT_B}
    table = RequestTable(2)
    table.add_request("a", prompts["a"], eos_id=-1, block_ids=[0, 1])
    table.add_request("b", prompts["b"], eos_id=-1, block_ids=[2, 3])

    generated = {}
    for step in range(1, MAX_NEW_TOKENS + 1):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
        # 每步两路都各前进一步，没有被移出的请求。
        lengths = {s.request_id: len(s.generated) for s in table.active_slots()}
        assert lengths == {"a": step, "b": step}
        generated = {s.request_id: list(s.generated)
                     for s in table.active_slots()}

    for request_id, prompt in prompts.items():
        ref = run_decode_loop(
            compiled, backend,
            prompt_ids=torch.tensor(prompt), max_new_tokens=MAX_NEW_TOKENS,
            eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of,
        )
        assert generated[request_id] == ref, (
            f"请求 {request_id} 外部块号生成 {generated[request_id]} "
            f"与单独运行 {ref} 不一致")
