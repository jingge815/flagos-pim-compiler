"""并行执行基准：同一微型 Llama 在多个并发窗口数下的墙钟、忙碌率与主机锁占用。

用法（先 source 环境）：
    python scripts/bench_parallel.py
    python scripts/bench_parallel.py --parallel-slots 1 2 --json /tmp/bench.json

各窗口数的生成 token 必须一致，否则脚本以非零码退出。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig  # noqa: E402
from contracts.op_contract import PIMHardwareConfig  # noqa: E402
from graph.strategy import llama_strategies  # noqa: E402
from memory.mem_planner import HwBudget  # noqa: E402
from runtime.compile import causal_mask_of, compile_llama2, load_weights  # noqa: E402
from runtime.executor import RequestTable, run_one_step  # noqa: E402
from runtime.kernels import register_all  # noqa: E402


class _TimedLock:
    """包装主机锁，累计等待时间与持有时间。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.wait = 0.0
        self.hold = 0.0
        self._t_hold = 0.0

    def __enter__(self) -> "_TimedLock":
        start = time.perf_counter()
        self._lock.acquire()
        self.wait += time.perf_counter() - start
        self._t_hold = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.hold += time.perf_counter() - self._t_hold
        self._lock.release()

    def reset(self) -> None:
        self.wait = 0.0
        self.hold = 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIM 并行执行基准")
    parser.add_argument("--num-dpus", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prefill", type=int, default=8)
    parser.add_argument("--max-seq", type=int, default=64)
    parser.add_argument("--slot-tokens", type=int, default=16)
    parser.add_argument("--parallel-slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeat", type=int, default=3, help="每个窗口数重复次数，取墙钟中位数")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def _build(args: argparse.Namespace):
    """构造微型模型、编译蓝图并加载权重，返回 (编译产物, backend)。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=args.hidden, intermediate_size=args.intermediate,
        num_hidden_layers=args.layers, num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads, vocab_size=args.vocab,
    )).eval().to(torch.float16)
    strategy = llama_strategies(
        args.num_dpus, num_heads=args.heads, num_kv_heads=args.kv_heads,
        intermediate_size=args.intermediate, vocab_size=args.vocab,
        num_layers=args.layers)[0]
    hw = HwBudget(mram_bytes=1 << 30, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=args.num_dpus, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=args.prefill, max_seq=args.max_seq,
        hw=hw, hardware=hardware, num_request_slots=args.requests,
        kv_slot_tokens=args.slot_tokens, num_parallel_slots=args.requests)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=args.num_dpus, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _instrument(backend: NumpyBackend) -> dict:
    """包装内核记录忙碌区间，包装主机路径统计命令数与耗时。"""
    stats: dict = {"intervals": [], "ops": {}, "kernel_cpu": 0.0}
    lock = threading.Lock()
    for name, kernel in list(backend._kernels._kernels.items()):
        def wrapped(hal, dpu_id, cmd, _kernel=kernel):
            t0 = time.perf_counter()
            c0 = time.thread_time()
            try:
                return _kernel(hal, dpu_id, cmd)
            finally:
                cpu = time.thread_time() - c0
                with lock:
                    stats["intervals"].append((dpu_id, t0, time.perf_counter()))
                    stats["kernel_cpu"] += cpu
        backend._kernels._kernels[name] = wrapped

    original_host = backend._run_host_or_dma

    def timed_host(cmd, deps, ctx):
        t0 = time.perf_counter()
        try:
            return original_host(cmd, deps, ctx)
        finally:
            with lock:
                stat = stats["ops"].setdefault(cmd.op, {"count": 0, "time": 0.0})
                stat["count"] += 1
                stat["time"] += time.perf_counter() - t0

    backend._run_host_or_dma = timed_host
    return stats


def _run(compiled, backend, args: argparse.Namespace, parallel_slots: int,
         host_lock: _TimedLock, stats: dict) -> dict:
    """跑一次「预填充全部请求 + 固定步数解码」，返回墙钟、忙碌率与 token。"""
    stats["intervals"].clear()
    stats["ops"].clear()
    host_lock.reset()
    table = RequestTable(args.requests, compiled.kv_pool_slots)
    for i in range(args.requests):
        prompt = list(range(i * args.prefill, (i + 1) * args.prefill))
        table.add_request(f"r{i}", prompt, eos_id=-1)
    sample = lambda logits: int(np.argmax(logits))  # noqa: E731
    run_one_step(compiled, backend, table, causal_mask_of, sample_fn=sample)
    stats["intervals"].clear()
    stats["kernel_cpu"] = 0.0

    t0 = time.perf_counter()
    c0 = time.process_time()
    for _ in range(args.steps):
        run_one_step(compiled, backend, table, causal_mask_of,
                     sample_fn=sample, parallel_slots=parallel_slots)
    wall = time.perf_counter() - t0
    proc_cpu = time.process_time() - c0

    busy = {d: 0.0 for d in range(args.num_dpus)}
    kernel_busy = 0.0
    for dpu_id, start, end in stats["intervals"]:
        clipped = max(0.0, end - max(start, t0))
        busy[dpu_id] += clipped
        kernel_busy += clipped
    util = sum(busy.values()) / (args.num_dpus * wall) * 100
    tokens = [list(table.get(f"r{i}").generated) for i in range(args.requests)]
    return {
        "parallel_slots": parallel_slots,
        "wall_ms": wall * 1000,
        "step_ms": wall / args.steps * 1000,
        "util": util,
        "host_hold_ms": host_lock.hold * 1000,
        "host_wait_ms": host_lock.wait * 1000,
        "host_share": host_lock.hold / wall * 100,
        "kernels": len(stats["intervals"]),
        "kernel_busy_ms": kernel_busy * 1000,
        "kernel_cpu_ms": stats["kernel_cpu"] * 1000,
        "kernel_cpu_ratio": stats["kernel_cpu"] / kernel_busy if kernel_busy else 0.0,
        "proc_cpu_ms": proc_cpu * 1000,
        "cores": proc_cpu / wall,
        "ops": {op: dict(v) for op, v in stats["ops"].items()},
        "tokens": tokens,
    }


def main() -> int:
    args = _parse_args()
    compiled, backend = _build(args)
    host_lock = _TimedLock()
    backend._host_lock = host_lock
    stats = _instrument(backend)
    print(f"槽池 {compiled.kv_pool_slots}，并行窗口 {compiled.num_parallel_slots}，"
          f"槽步长 {compiled.act_stride} 字节，每组重复 {args.repeat} 次取中位数")

    results = []
    baseline = None
    for p in args.parallel_slots:
        runs = [_run(compiled, backend, args, p, host_lock, stats)
                for _ in range(max(1, args.repeat))]
        for run in runs:
            if baseline is None:
                baseline = run["tokens"]
            elif run["tokens"] != baseline:
                print(f"token 不一致：P={p} 与 P={args.parallel_slots[0]}")
                return 1
        runs.sort(key=lambda r: r["wall_ms"])
        results.append(runs[len(runs) // 2])

    print("\n| P | 墙钟 ms | ms/步 | DPU 忙碌率 | 进程核数 | 内核 CPU/墙钟 | 内核数 | 主机锁持有 ms | 锁占比 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['parallel_slots']} | {r['wall_ms']:.0f} | {r['step_ms']:.0f} | "
              f"{r['util']:.1f}% | {r['cores']:.2f} | {r['kernel_cpu_ratio'] * 100:.0f}% | "
              f"{r['kernels']} | {r['host_hold_ms']:.0f} | {r['host_share']:.1f}% |")
    print(f"\ntoken（{len(baseline)} 个请求）：{baseline}")
    if args.json is not None:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"结果写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
