"""验证多请求按激活窗口并发执行，token 与单请求逐 token 一致。"""

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
from memory.kv_layout import write_block_table
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, run_decode_loop, run_one_step
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
MAX_SEQ = 16
NUM_SLOTS = 4
MAX_NEW_TOKENS = 3

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128, num_layers=NUM_LAYERS,
)

PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]


def _greedy(logits):
    return int(np.argmax(logits))


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


@pytest.fixture(scope="module", params=[MAX_SEQ, 2])
def parallel_setup(request, small_model, hw_and_hardware):
    """每种 KV 槽大小编译一次，全部用例共用。"""
    hw, hardware = hw_and_hardware
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    compiled = compile_llama2(
        small_model, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        num_request_slots=NUM_SLOTS, kv_slot_tokens=request.param,
        num_parallel_slots=NUM_SLOTS,
    )
    return request.param, compiled, hardware


def _backend_with_weights(compiled, hardware) -> NumpyBackend:
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=compiled.num_parallel_slots,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    return backend


def _reference(compiled, backend, prompt, max_new_tokens=MAX_NEW_TOKENS):
    return run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor(prompt),
        max_new_tokens=max_new_tokens, eos_id=-1,
        sample_fn=_greedy, causal_mask_of=causal_mask_of,
    )


def test_four_requests_match_single_runs(parallel_setup) -> None:
    """4 个请求同一 wave 并发，生成结果分别等于单独运行。"""
    _, compiled, hardware = parallel_setup
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS):
        table.add_request(f"r{index}", prompt, eos_id=-1)

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    for index, prompt in enumerate(PROMPTS):
        got = list(table.get(f"r{index}").generated)
        assert got == _reference(compiled, backend, prompt), (
            f"r{index} 并发生成 {got} 与单独运行不一致")


def test_mixed_prefill_and_decode_in_one_wave(parallel_setup) -> None:
    """一个请求在 decode、另一个刚接入要 prefill，同一步并发互不影响。"""
    _, compiled, hardware = parallel_setup
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    table.add_request("a", PROMPTS[0], eos_id=-1)
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    table.add_request("b", PROMPTS[1], eos_id=-1)

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    assert list(table.get("a").generated) == _reference(
        compiled, backend, PROMPTS[0], MAX_NEW_TOKENS + 1)
    assert list(table.get("b").generated) == _reference(
        compiled, backend, PROMPTS[1])


def test_repeated_runs_are_deterministic(parallel_setup) -> None:
    """同样的并发执行重复 5 次，token 序列完全一致。"""
    _, compiled, hardware = parallel_setup
    backend = _backend_with_weights(compiled, hardware)
    runs = []
    for _ in range(5):
        table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
        for index in range(2):
            table.add_request(f"r{index}", PROMPTS[index], eos_id=-1)
        for _ in range(MAX_NEW_TOKENS):
            run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
        runs.append(tuple(tuple(table.get(f"r{i}").generated) for i in range(2)))
    assert all(run == runs[0] for run in runs)


def test_unused_windows_receive_no_writes(parallel_setup) -> None:
    """只有 1 个请求在飞时只开窗口 0，其余激活窗口与块表窗口的字节一动不动。"""
    _, compiled, hardware = parallel_setup
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    table.add_request("r0", PROMPTS[0], eos_id=-1)

    stride = compiled.act_stride
    act_bases = {dpu: plan.act_base for dpu, plan in compiled.mem_plans.items()}
    bt_specs = {spec.dpu_id: spec for spec in compiled.kv_specs.values()}

    def read(dpu: int, offset: int, size: int) -> np.ndarray:
        return backend.read_local(dpu, offset, (size,), np.uint8)

    def spare_windows() -> dict[tuple[str, int, int], np.ndarray]:
        """窗口 1 起的激活区与块表字节快照，键是 (区, DPU, 窗口)。"""
        snap = {}
        for dpu, base in act_bases.items():
            for window in range(1, compiled.num_parallel_slots):
                snap[("act", dpu, window)] = read(dpu, base + window * stride, stride)
        for dpu, spec in bt_specs.items():
            for window in range(1, compiled.num_parallel_slots):
                snap[("bt", dpu, window)] = read(
                    dpu, spec.block_table_base + window * spec.block_table_bytes,
                    spec.block_table_bytes)
        return snap

    before = spare_windows()
    used_before = {dpu: read(dpu, base, stride) for dpu, base in act_bases.items()}
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)
    after = spare_windows()

    for key, old in before.items():
        assert np.array_equal(old, after[key]), f"{key} 被写了"
    # 窗口 0 确实在用，否则上面的断言没有意义。
    assert any(not np.array_equal(used_before[dpu], read(dpu, base, stride))
               for dpu, base in act_bases.items())


def test_block_table_windows_are_independent(parallel_setup) -> None:
    """每份块表窗口独立可写，互不覆盖。"""
    _, compiled, hardware = parallel_setup
    backend = _backend_with_weights(compiled, hardware)
    for spec in compiled.kv_specs.values():
        write_block_table(spec, backend, [0], slot_index=0)
        write_block_table(spec, backend, [1], slot_index=1)
        window0 = backend.read_local(
            spec.dpu_id, spec.block_table_base, (spec.blocks_per_request,), np.int32)
        window1 = backend.read_local(
            spec.dpu_id, spec.block_table_base + spec.block_table_bytes,
            (spec.blocks_per_request,), np.int32)
        assert window0[0] == 0
        assert window1[0] == 1
