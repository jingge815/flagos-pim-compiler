"""验证两个请求窗口在同一 DPU 上的命令区间真实重叠。"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, run_one_step
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
MAX_SEQ = 16
PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8]]
KERNEL_SLEEP = 0.005  # 拉宽内核占用，让重叠可见。


def _compile():
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
    )).eval().to(torch.float16)
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=4, intermediate_size=128,
        vocab_size=128, num_layers=NUM_LAYERS)[0]
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=2, num_parallel_slots=2,
    )
    return compiled, hardware


def _instrument(backend):
    """包装内核与 host op，记录 (窗口, DPU, 命令号, 起止) 和命令提交顺序。"""
    launches: list[tuple[int, int, int, float, float]] = []
    hosts: list[tuple[int, int, float, float]] = []
    order: dict[tuple[int, int], int] = {}
    lock = threading.Lock()
    counter = itertools.count()

    for name, kernel in list(backend._kernels._kernels.items()):
        def wrapped(hal, dpu_id, cmd, _kernel=kernel):
            lane = int(cmd.payload.get("block_slot", 0))
            t0 = time.perf_counter()
            time.sleep(KERNEL_SLEEP)
            try:
                return _kernel(hal, dpu_id, cmd)
            finally:
                with lock:
                    launches.append((lane, dpu_id, cmd.id, t0, time.perf_counter()))
        backend._kernels._kernels[name] = wrapped

    original_host = backend._run_host_or_dma

    def host_run(cmd, deps, ctx):
        lane = 0 if ctx is None else int(ctx.lane_key)
        t0 = time.perf_counter()
        try:
            return original_host(cmd, deps, ctx)
        finally:
            with lock:
                hosts.append((lane, cmd.id, t0, time.perf_counter()))

    backend._run_host_or_dma = host_run

    original_submit = backend.submit

    def submit(cmd, ctx=None):
        lane = 0 if ctx is None else int(ctx.lane_key)
        with lock:
            order[(lane, cmd.id)] = next(counter)
        return original_submit(cmd, ctx)

    backend.submit = submit
    return launches, hosts, order


def test_two_windows_overlap_on_the_same_dpu() -> None:
    compiled, hardware = _compile()
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=2,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    launches, hosts, order = _instrument(backend)

    table = RequestTable(2, compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS):
        table.add_request(f"r{index}", prompt, eos_id=-1)
    run_one_step(compiled, backend, table, causal_mask_of)

    lane0 = [item for item in launches if item[0] == 0]
    lane1 = [item for item in launches if item[0] == 1]
    assert lane0 and lane1, "两个窗口都应执行到 launch 命令"

    same_dpu_overlap = any(
        d0 == d1 and s0 < e1 and s1 < e0
        for _, d0, _, s0, e0 in lane0
        for _, d1, _, s1, e1 in lane1
    )
    assert same_dpu_overlap, "同一 DPU 上两个窗口的 launch 区间没有重叠"

    # 越级：后提交的窗口 1 命令在先提交的窗口 0 命令结束前就开始。
    overtake = any(
        d0 == d1
        and order[(0, c0)] < order[(1, c1)]
        and s1 < e0
        for _, d0, c0, s0, e0 in lane0
        for _, d1, c1, s1, e1 in lane1
    )
    assert overtake, "窗口 1 没有在任何更早提交的窗口 0 命令结束前开始"

    # 窗口 1 的内核在窗口 0 的 host 命令还在飞时开始。
    host0 = [(s, e) for lane, _, s, e in hosts if lane == 0]
    assert host0, "这个模型应当有 host 侧重分布命令"
    kernel_during_host = any(
        s1 < e0 and s0 < e1
        for s0, e0 in host0
        for _, _, _, s1, e1 in lane1
    )
    assert kernel_during_host, "窗口 1 的内核没有和窗口 0 的 host 命令重叠"
