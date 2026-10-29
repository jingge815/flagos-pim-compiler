#!/usr/bin/env python3
"""DPU 执行时间线采集与利用率分析。

用法：
    # 在测试或脚本里嵌入
    from scripts.profile_execution import profile_backend, print_report

    profile = profile_backend(backend)       # 开始采集
    run_one_step(compiled, backend, table, causal_mask_of)
    profile.stop()                            # 结束采集
    print_report(profile)                     # 文本报告
    profile.to_html("timeline.html")          # 甘特图

    # 也可以直接当脚本跑一个小型 Llama 演示
    python scripts/profile_execution.py

    # 换成真实 Llama-2-7B：把 __main__ 里的 _demo() 改成 complete_7b()

    # 采集 vLLM 的调度：PIM_PARALLEL_SLOTS 取 1 和 2 各跑一次并对比
    python scripts/profile_execution.py vllm
"""

from __future__ import annotations

import html
import itertools
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ────────────────────────────────────────────────────────────
# 数据结构
# ────────────────────────────────────────────────────────────


@dataclass
class CommandRecord:
    """一条命令的执行记录。"""

    cmd_id: int
    op: str                # "launch" / "dma_in" / "dma_out" / "host"
    dpu_id: int | None     # host 命令为 None
    lane: int              # 请求窗口编号（ExecutionContext.lane_key）
    start: float           # perf_counter 秒
    end: float             # perf_counter 秒
    waits: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def kind(self) -> str:
        if self.op == "launch":
            return "kernel"
        if self.op in ("dma_in", "dma_out"):
            return "dma"
        return "host"


@dataclass
class ProfileResult:
    """采集结果：全部命令记录 + 全局起止时间。"""

    records: list[CommandRecord]
    t0: float               # 采集起点
    t1: float               # 采集终点
    num_dpus: int

    @property
    def wall(self) -> float:
        return self.t1 - self.t0

    def per_dpu_busy(self) -> dict[int, float]:
        """每台 DPU 的总忙碌秒数（kernel + DMA，不含 host）。"""
        busy: dict[int, float] = {}
        for r in self.records:
            if r.dpu_id is not None and r.kind != "host":
                busy[r.dpu_id] = busy.get(r.dpu_id, 0.0) + r.duration
        return busy

    def per_dpu_utilization(self) -> dict[int, float]:
        """每台 DPU 的利用率 = busy / wall。"""
        busy = self.per_dpu_busy()
        return {dpu: busy.get(dpu, 0.0) / self.wall for dpu in range(self.num_dpus)}

    def overlap_matrix(self) -> dict[tuple[int, int], int]:
        """同一时刻在飞的 (dpu_i, dpu_j) 对数（i < j），衡量跨 DPU 并行度。"""
        matrix: dict[tuple[int, int], int] = {}
        dpu_cmds = [r for r in self.records if r.dpu_id is not None and r.kind != "host"]
        for a in dpu_cmds:
            for b in dpu_cmds:
                if a.dpu_id >= b.dpu_id:
                    continue
                if a.start < b.end and b.start < a.end:
                    key = (a.dpu_id, b.dpu_id)
                    matrix[key] = matrix.get(key, 0) + 1
        return matrix

    def kernel_parallelism(self) -> list[tuple[float, int]]:
        """按时间切片，每秒同时执行的内核数。返回 [(time, count), ...]。"""
        events: list[tuple[float, int]] = []
        for r in self.records:
            if r.kind == "kernel":
                events.append((r.start, +1))
                events.append((r.end, -1))
        events.sort()
        result = []
        running = 0
        for t, delta in events:
            running += delta
            result.append((t - self.t0, running))
        return result

    def max_parallel_kernels(self) -> int:
        """同一时刻最多的并发内核数。"""
        return max((c for _, c in self.kernel_parallelism()), default=0)

    def per_lane_summary(self) -> dict[int, dict]:
        """每个请求窗口（lane）的命令数和总 kernel 时间。"""
        lanes: dict[int, dict] = {}
        for r in self.records:
            if r.lane not in lanes:
                lanes[r.lane] = {"cmds": 0, "kernel_time": 0.0, "host_time": 0.0}
            lanes[r.lane]["cmds"] += 1
            if r.kind == "kernel":
                lanes[r.lane]["kernel_time"] += r.duration
            elif r.kind == "host":
                lanes[r.lane]["host_time"] += r.duration
        return lanes

    def cross_lane_overlap(self) -> float:
        """不同 lane 的 kernel 区间互相重叠的总秒数。"""
        by_lane: dict[int, list[CommandRecord]] = {}
        for r in self.records:
            if r.kind == "kernel":
                by_lane.setdefault(r.lane, []).append(r)
        total = 0.0
        lane_ids = sorted(by_lane)
        for i, la in enumerate(lane_ids):
            for lb in lane_ids[i + 1:]:
                for a in by_lane[la]:
                    for b in by_lane[lb]:
                        overlap = min(a.end, b.end) - max(a.start, b.start)
                        if overlap > 0:
                            total += overlap
        return total


# ────────────────────────────────────────────────────────────
# 采集器
# ────────────────────────────────────────────────────────────


class _Profiler:
    """包装 NumpyBackend 的内核和 host 路径，记录每条命令的起止时间。"""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.records: list[CommandRecord] = []
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._t1 = self._t0

        # 包装所有已注册内核
        for name, kernel in list(backend._kernels._kernels.items()):
            self._wrap_kernel(name, kernel)

        # 包装 host/DMA 路径
        self._original_host = backend._run_host_or_dma
        backend._run_host_or_dma = self._host_run

    def _wrap_kernel(self, name, kernel):
        def wrapped(hal, dpu_id, cmd, _kernel=kernel):
            lane = int(cmd.payload.get("block_slot", 0))
            t0 = time.perf_counter()
            try:
                return _kernel(hal, dpu_id, cmd)
            finally:
                t1 = time.perf_counter()
                with self._lock:
                    self.records.append(CommandRecord(
                        cmd_id=cmd.id, op="launch", dpu_id=dpu_id,
                        lane=lane, start=t0, end=t1,
                        waits=list(cmd.waits),
                    ))
        self.backend._kernels._kernels[name] = wrapped

    def _host_run(self, cmd, deps, ctx):
        lane = 0 if ctx is None else int(ctx.lane_key)
        t0 = time.perf_counter()
        try:
            return self._original_host(cmd, deps, ctx)
        finally:
            t1 = time.perf_counter()
            kind = "host" if cmd.dpu_id is None else "dma"
            with self._lock:
                self.records.append(CommandRecord(
                    cmd_id=cmd.id, op=kind, dpu_id=cmd.dpu_id,
                    lane=lane, start=t0, end=t1,
                    waits=list(cmd.waits),
                ))

    def stop(self) -> ProfileResult:
        """停止采集，返回结果。"""
        self._t1 = time.perf_counter()
        # 还原 host 路径
        self.backend._run_host_or_dma = self._original_host
        num_dpus = self.backend.config.num_dpus
        return ProfileResult(
            records=list(self.records),
            t0=self._t0, t1=self._t1,
            num_dpus=num_dpus,
        )


def profile_backend(backend) -> _Profiler:
    """开始采集。调用 `.stop()` 结束。"""
    return _Profiler(backend)


# ────────────────────────────────────────────────────────────
# 报告输出
# ────────────────────────────────────────────────────────────


def print_report(result: ProfileResult) -> None:
    """打印文本报告到 stdout。"""
    wall_ms = result.wall * 1000
    print(f"\n{'=' * 65}")
    print(f"执行时间线报告")
    print(f"{'=' * 65}")
    print(f"总命令数: {len(result.records)}")
    print(f"Wall clock: {wall_ms:.1f} ms")
    print(f"最大并发内核: {result.max_parallel_kernels()}")

    # 按类型统计
    by_kind: dict[str, int] = {}
    for r in result.records:
        by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
    print(f"命令构成: {by_kind}")

    # DPU 利用率
    util = result.per_dpu_utilization()
    busy = result.per_dpu_busy()
    print(f"\n--- DPU 利用率 ---")
    total_busy = 0.0
    for dpu in sorted(util):
        b_ms = busy.get(dpu, 0.0) * 1000
        u = util[dpu] * 100
        total_busy += b_ms
        bar = "█" * int(u / 2) + "░" * (50 - int(u / 2))
        print(f"  DPU {dpu}: {bar} {u:5.1f}%  ({b_ms:.1f} ms)")
    avg_util = (total_busy / (wall_ms * result.num_dpus)) * 100 if result.num_dpus else 0
    print(f"  {'平均':>5}: {avg_util:5.1f}%")

    # 跨 lane 分析
    lanes = result.per_lane_summary()
    if len(lanes) > 1:
        print(f"\n--- 请求窗口（Lane）---")
        for lane_id in sorted(lanes):
            info = lanes[lane_id]
            print(f"  Lane {lane_id}: {info['cmds']} 命令, "
                  f"kernel {info['kernel_time'] * 1000:.1f} ms, "
                  f"host {info['host_time'] * 1000:.1f} ms")
        overlap_ms = result.cross_lane_overlap() * 1000
        print(f"  跨 lane kernel 重叠: {overlap_ms:.1f} ms")

    # 内核并行度分布
    print(f"\n--- 内核并行度时间分布 ---")
    par = result.kernel_parallelism()
    if par:
        # 统计每种并行度持续的时间
        durations: dict[int, float] = {}
        prev_t, prev_c = par[0]
        for t, c in par[1:]:
            durations[prev_c] = durations.get(prev_c, 0.0) + (t - prev_t)
            prev_t, prev_c = t, c
        for count in sorted(durations):
            d_ms = durations[count] * 1000
            pct = d_ms / wall_ms * 100
            bar = "█" * max(1, int(pct / 2))
            print(f"  {count} 个并发: {bar} {d_ms:.1f} ms ({pct:.1f}%)")

    print(f"{'=' * 65}\n")


# ────────────────────────────────────────────────────────────
# vLLM 调度采集
# ────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """一次 `PIMWorker.execute_model` 的调度记录。"""

    step: int
    req_ids: list[str]                          # 本步被调度的请求
    tokens: list[int]                           # 各请求本步调度的 token 数
    start: float
    end: float
    tasks: list[tuple[str, int, float, float]]  # (请求, 窗口号, 起, 止)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def busy(self) -> float:
        """各请求执行时间之和；真并发时大于本步墙钟。"""
        return sum(end - start for _, _, start, end in self.tasks)

    @property
    def efficiency(self) -> float:
        """并行效率 = 忙碌和 / 墙钟：1 是完全串行，k 是 k 路真重叠。"""
        return self.busy / self.duration if self.duration > 0 else 0.0

    @property
    def concurrency(self) -> int:
        """同一时刻最多有几个请求在跑。"""
        edges: list[tuple[float, int]] = []
        for _, _, start, end in self.tasks:
            edges.append((start, 1))
            edges.append((end, -1))
        edges.sort()
        running = peak = 0
        for _, delta in edges:
            running += delta
            peak = max(peak, running)
        return peak

    @property
    def is_prefill(self) -> bool:
        """本步是否含预填充：调度 token 数大于 1 即预填充。"""
        return any(count > 1 for count in self.tokens)

    def slot_conflicts(self) -> list[tuple[str, str, int]]:
        """时间重叠却共用同一个激活窗口的请求对。"""
        conflicts = []
        for index, (req_a, slot_a, start_a, end_a) in enumerate(self.tasks):
            for req_b, slot_b, start_b, end_b in self.tasks[index + 1:]:
                if slot_a == slot_b and start_a < end_b and start_b < end_a:
                    conflicts.append((req_a, req_b, slot_a))
        return conflicts


@dataclass
class VllmProfileResult:
    """vLLM 调度采集结果：逐步的请求构成与请求级执行区间。"""

    steps: list[StepRecord]

    def mean_efficiency(self, prefill: bool) -> float:
        """按阶段统计的平均并行效率。"""
        values = [s.efficiency for s in self.steps if s.is_prefill == prefill]
        return sum(values) / len(values) if values else 0.0

    def max_concurrency(self) -> int:
        return max((s.concurrency for s in self.steps), default=0)

    def slot_conflicts(self) -> list[tuple[str, str, int]]:
        return [c for s in self.steps for c in s.slot_conflicts()]


class _VllmProfiler:
    """包装 PIMWorker 的 execute_model 和 _step，记录每个请求自己的执行区间。

    两者的方法都换在实例上：`execute_model` 划步骤边界，`_step` 记请求起止。
    拿不到 worker 实例（多进程模式）时用不了，因为那时 worker 在子进程里。
    """

    def __init__(self, worker) -> None:
        self.worker = worker
        self.steps: list[StepRecord] = []
        self._tasks: list[tuple[str, int, float, float]] = []
        self._lock = threading.Lock()
        self._orig_execute_model = worker.execute_model
        self._orig_step = worker._step
        worker.execute_model = self._execute_model
        worker._step = self._step

    def _execute_model(self, scheduler_output):
        with self._lock:
            mark = len(self._tasks)
        start = time.perf_counter()
        try:
            return self._orig_execute_model(scheduler_output)
        finally:
            end = time.perf_counter()
            # execute_model 返回前已 join 完全部 wave，这里取到的区间是完整的。
            with self._lock:
                tasks = self._tasks[mark:]
            self.steps.append(StepRecord(
                step=len(self.steps),
                req_ids=list(scheduler_output.num_scheduled_tokens),
                tokens=list(scheduler_output.num_scheduled_tokens.values()),
                start=start, end=end, tasks=tasks))

    def _step(self, req_id, slot_index=0):
        start = time.perf_counter()
        try:
            return self._orig_step(req_id, slot_index)
        finally:
            with self._lock:
                self._tasks.append(
                    (req_id, int(slot_index), start, time.perf_counter()))

    def stop(self) -> VllmProfileResult:
        """还原 worker 的方法并返回采集结果。"""
        self.worker.execute_model = self._orig_execute_model
        self.worker._step = self._orig_step
        return VllmProfileResult(steps=list(self.steps))


def profile_vllm(worker) -> _VllmProfiler:
    """开始采集 vLLM 调度；`worker` 是 PIMWorker 实例。"""
    return _VllmProfiler(worker)


def print_vllm_report(result: VllmProfileResult, *, title: str = "vLLM 调度报告") -> None:
    """打印每步的请求构成、并行效率、并发度与窗口使用。"""
    prefill = sum(1 for s in result.steps if s.is_prefill)
    decode = len(result.steps) - prefill
    conflicts = result.slot_conflicts()
    print(f"\n{'=' * 65}")
    print(title)
    print(f"{'=' * 65}")
    print(f"步数: {len(result.steps)}（预填充 {prefill}，解码 {decode}）")
    print(f"平均并行效率: 预填充 {result.mean_efficiency(True):.2f}，"
          f"解码 {result.mean_efficiency(False):.2f}"
          f"  （1 = 完全串行，2 = 两请求真重叠）")
    print(f"最大并发请求: {result.max_concurrency()}")
    print(f"激活窗口冲突: {'无' if not conflicts else conflicts}")
    print(f"\n  步  阶段     请求          效率   并发  窗口  墙钟ms  忙碌ms")
    for s in result.steps:
        stage = "prefill" if s.is_prefill else "decode"
        slots = ",".join(str(task[1]) for task in s.tasks)
        print(f"  {s.step:>2}  {stage:<8} {'+'.join(s.req_ids):<12}  "
              f"{s.efficiency:>5.2f}  {s.concurrency:>4}  {slots:<4}  "
              f"{s.duration * 1000:>6.1f}  {s.busy * 1000:>6.1f}")
    print(f"{'=' * 65}\n")


# ────────────────────────────────────────────────────────────
# HTML 甘特图
# ────────────────────────────────────────────────────────────


def _to_html(result: ProfileResult, path: str) -> None:
    """生成 HTML 甘特图。"""
    wall_ms = result.wall * 1000
    px_per_ms = 2  # 缩放
    width = max(800, int(wall_ms * px_per_ms) + 200)

    # 行：每台 DPU 一行，host 一行
    rows: list[str] = []
    row_height = 28
    colors = {"kernel": "#4a90d9", "dma": "#e6a23c", "host": "#67c23a"}
    lane_colors = ["#4a90d9", "#e6574a", "#67c23a", "#e6a23c", "#9b59b6"]

    dpu_rows = {}
    for dpu in range(result.num_dpus):
        dpu_rows[dpu] = len(rows)
        rows.append(f"DPU {dpu}")
    host_row = len(rows)
    rows.append("Host")

    total_height = len(rows) * row_height + 80

    # 生成 SVG 矩形
    rects = []
    for r in result.records:
        x = (r.start - result.t0) * 1000 * px_per_ms + 100
        w = max(1, r.duration * 1000 * px_per_ms)
        if r.dpu_id is not None and r.kind != "host":
            y = dpu_rows[r.dpu_id] * row_height + 40
        else:
            y = host_row * row_height + 40
        color = colors.get(r.kind, "#999")
        # 不同 lane 用不同色调
        if r.kind == "kernel":
            color = lane_colors[r.lane % len(lane_colors)]
        tooltip = (f"id={r.cmd_id} op={r.op} dpu={r.dpu_id} "
                   f"lane={r.lane} {r.duration * 1000:.2f}ms")
        rects.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{row_height - 4}" '
            f'fill="{color}" opacity="0.85" rx="2">'
            f'<title>{html.escape(tooltip)}</title></rect>')

    # 行标签
    labels = []
    for i, name in enumerate(rows):
        y = i * row_height + 40 + row_height // 2 + 4
        labels.append(f'<text x="5" y="{y}" font-size="12" '
                      f'font-family="monospace">{html.escape(name)}</text>')

    # 时间轴刻度
    ticks = []
    step = max(1, int(wall_ms / 20))
    for ms in range(0, int(wall_ms) + 1, step):
        x = ms * px_per_ms + 100
        ticks.append(f'<line x1="{x}" y1="30" x2="{x}" y2="{total_height - 30}" '
                     f'stroke="#ddd" stroke-width="0.5"/>')
        ticks.append(f'<text x="{x}" y="25" font-size="10" '
                     f'font-family="monospace" text-anchor="middle">{ms}ms</text>')

    svg = f"""<svg width="{width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#fafafa"/>
  {''.join(ticks)}
  {''.join(labels)}
  {''.join(rects)}
</svg>"""

    # 统计面板
    util = result.per_dpu_utilization()
    stats = [
        f"Wall: {wall_ms:.1f} ms",
        f"Commands: {len(result.records)}",
        f"Max parallel kernels: {result.max_parallel_kernels()}",
    ]
    for dpu in sorted(util):
        stats.append(f"DPU {dpu}: {util[dpu] * 100:.1f}%")

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DPU Timeline</title>
<style>
  body {{ font-family: monospace; margin: 20px; background: #f5f5f5; }}
  .stats {{ background: #fff; padding: 12px; border-radius: 6px;
            margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stats span {{ margin-right: 24px; }}
  .chart {{ background: #fff; padding: 16px; border-radius: 6px;
            overflow-x: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .legend {{ margin: 8px 0; }}
  .legend span {{ display: inline-block; width: 14px; height: 14px;
                  vertical-align: middle; margin-right: 4px; border-radius: 2px; }}
</style></head><body>
<h2>DPU 执行时间线</h2>
<div class="stats">{'  '.join(f'<span>{s}</span>' for s in stats)}</div>
<div class="legend">
  <span style="background:#4a90d9"></span>Lane 0 kernel
  <span style="background:#e6574a"></span>Lane 1 kernel
  <span style="background:#67c23a"></span>Lane 2 kernel
  <span style="background:#e6a23c"></span>DMA
  <span style="background:#67c23a"></span>Host
</div>
<div class="chart">{svg}</div>
</body></html>"""

    Path(path).write_text(page, encoding="utf-8")


def _to_chrome_trace(result: ProfileResult, path: str) -> None:
    """导出 Chrome Trace Event 格式（chrome://tracing 或 Perfetto 可打开）。"""
    import json

    events = []
    for r in result.records:
        cat = r.kind
        name = f"{r.op}" + (f":dpu{r.dpu_id}" if r.dpu_id is not None else ":host")
        pid = r.dpu_id if r.dpu_id is not None else result.num_dpus  # host 用独立 pid
        tid = r.lane
        events.append({
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": (r.start - result.t0) * 1e6,   # 微秒
            "dur": r.duration * 1e6,
            "pid": pid,
            "tid": tid,
            "args": {"cmd_id": r.cmd_id, "lane": r.lane,
                     "waits": r.waits[:5]},        # 只存前 5 个避免太长
        })
    # process/thread 元数据
    for dpu in range(result.num_dpus):
        events.append({"name": "process_name", "ph": "M", "pid": dpu, "tid": 0,
                        "args": {"name": f"DPU {dpu}"}})
    events.append({"name": "process_name", "ph": "M", "pid": result.num_dpus,
                    "tid": 0, "args": {"name": "Host"}})

    Path(path).write_text(json.dumps({"traceEvents": events}, indent=None),
                          encoding="utf-8")


# 把 to_html / to_chrome_trace 挂到 ProfileResult 上
ProfileResult.to_html = lambda self, path: _to_html(self, path)
ProfileResult.to_chrome_trace = lambda self, path: _to_chrome_trace(self, path)


# ────────────────────────────────────────────────────────────
# 独立运行：小型 Llama 演示
# ────────────────────────────────────────────────────────────


def _demo():
    """编译小型 Llama 并采集执行时间线。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategies
    from memory.mem_planner import HwBudget
    from runtime.compile import causal_mask_of, compile_llama2, load_weights
    from runtime.executor import RequestTable, run_one_step
    from runtime.kernels import register_all

    NUM_DPUS, NUM_LAYERS = 4, 4
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=256,
    )).eval().to(torch.float16)

    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=4, intermediate_size=256,
        vocab_size=256, num_layers=NUM_LAYERS)[0]

    # 串行模式
    print("编译串行模式（P=1）...")
    compiled_serial = compile_llama2(
        model, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=hardware, num_request_slots=2,
        num_parallel_slots=1,
    )
    backend_s = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=1,
    ))
    register_all(backend_s)
    load_weights(compiled_serial, backend_s)

    table_s = RequestTable(2, compiled_serial.kv_pool_slots)
    for i, p in enumerate([[1, 2, 3, 4], [5, 6, 7, 8]]):
        table_s.add_request(f"r{i}", p, eos_id=-1)

    prof = profile_backend(backend_s)
    run_one_step(compiled_serial, backend_s, table_s, causal_mask_of)
    result_s = prof.stop()
    print_report(result_s)
    result_s.to_html("timeline_serial.html")
    print("甘特图 → timeline_serial.html")

    # 并行模式
    print("\n编译并行模式（P=2）...")
    compiled_par = compile_llama2(
        model, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=hardware, num_request_slots=2,
        num_parallel_slots=2,
    )
    backend_p = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=2,
    ))
    register_all(backend_p)
    load_weights(compiled_par, backend_p)

    table_p = RequestTable(2, compiled_par.kv_pool_slots)
    for i, p in enumerate([[1, 2, 3, 4], [5, 6, 7, 8]]):
        table_p.add_request(f"r{i}", p, eos_id=-1)

    prof = profile_backend(backend_p)
    run_one_step(compiled_par, backend_p, table_p, causal_mask_of)
    result_p = prof.stop()
    print_report(result_p)
    result_p.to_html("timeline_parallel.html")
    print("甘特图 → timeline_parallel.html")

    # 对比
    print(f"\n{'=' * 40}")
    print(f"串行 vs 并行对比:")
    print(f"  Wall: {result_s.wall * 1000:.1f} ms → {result_p.wall * 1000:.1f} ms "
          f"({result_p.wall / result_s.wall * 100:.0f}%)")
    s_util = sum(result_s.per_dpu_utilization().values()) / NUM_DPUS * 100
    p_util = sum(result_p.per_dpu_utilization().values()) / NUM_DPUS * 100
    print(f"  平均利用率: {s_util:.1f}% → {p_util:.1f}%")
    print(f"  最大并发内核: {result_s.max_parallel_kernels()} → "
          f"{result_p.max_parallel_kernels()}")


# ────────────────────────────────────────────────────────────
# 独立运行：真实 Llama-2-7B
# ────────────────────────────────────────────────────────────


NUM_DPUS_7B = 8
PREFILL_LEN_7B = 4
MAX_SEQ_7B = 64
DECODE_STEPS_7B = 2
PARALLEL_SLOTS_7B = (1, 2)
PROMPTS_7B = [[i * PREFILL_LEN_7B + j + 1 for j in range(PREFILL_LEN_7B)]
              for i in range(2)]


def _profile_7b_case(model, strategy, hw, hardware, parallel_slots, html_path):
    """编译一次 7B 蓝图，采集预填充加若干解码步的时间线，返回 ProfileResult。"""
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, compile_llama2, load_weights
    from runtime.executor import RequestTable, run_one_step
    from runtime.kernels import register_all, reset_route_counts, route_counts

    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_LEN_7B, max_seq=MAX_SEQ_7B,
        hw=hw, hardware=hardware, num_request_slots=len(PROMPTS_7B),
        num_parallel_slots=parallel_slots,
    )
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS_7B, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=compiled.num_parallel_slots,
    ))
    register_all(backend)
    load_weights(compiled, backend)

    table = RequestTable(len(PROMPTS_7B), compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS_7B):
        table.add_request(f"r{index}", prompt, eos_id=-1)

    reset_route_counts()
    prof = profile_backend(backend)
    # 第一次调用是预填充，其后每次解码一个 token
    for _ in range(1 + DECODE_STEPS_7B):
        run_one_step(compiled, backend, table, causal_mask_of,
                     parallel_slots=parallel_slots)
    result = prof.stop()

    print_report(result)
    result.to_html(html_path)
    print(f"甘特图 → {html_path}")
    print(f"内核路由: {route_counts()}")
    return result


def complete_7b():
    """编译真实 Llama-2-7B 并采集执行时间线，P=1 与 P=2 各跑一次。

    模型目录取 paths.local.json 的 llama2_7b_model_dir。
    """
    import torch
    from transformers import LlamaForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from contracts.op_contract import PIMHardwareConfig
    from genesim_bridge.paths import llama2_7b_model_dir
    from graph.strategy import llama_strategies
    from memory.mem_planner import HwBudget

    model_dir = llama2_7b_model_dir()
    print(f"模型目录: {model_dir}")
    # 与 tests 里的 7B 用例保持一致：导出时不带梯度，避免多出 host 节点
    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, local_files_only=True).eval()
    cfg = model.config
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} kv_heads={cfg.num_key_value_heads} "
          f"vocab={cfg.vocab_size}")

    hw = HwBudget(mram_bytes=4 << 30, align=64, sys_reserve_bytes=64 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS_7B, num_tasklets=4,
        mram_bytes_per_dpu=hw.mram_bytes, wram_bytes_per_dpu=65536,
        dma_align=64,
    )
    strategy = llama_strategies(
        NUM_DPUS_7B, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )[0]

    results = {}
    for parallel_slots in PARALLEL_SLOTS_7B:
        print(f"\n编译并运行 P={parallel_slots} ...")
        results[parallel_slots] = _profile_7b_case(
            model, strategy, hw, hardware, parallel_slots,
            f"timeline_7b_p{parallel_slots}.html",
        )

    if len(results) < 2:
        return
    (p1, r1), (p2, r2) = sorted(results.items())
    util1 = sum(r1.per_dpu_utilization().values()) / NUM_DPUS_7B
    util2 = sum(r2.per_dpu_utilization().values()) / NUM_DPUS_7B
    print(f"\n{'=' * 40}")
    print(f"7B P={p1} vs P={p2}:")
    print(f"  Wall: {r1.wall * 1000:.0f} ms → {r2.wall * 1000:.0f} ms "
          f"({r2.wall / r1.wall * 100:.0f}%)")
    print(f"  平均利用率: {util1 * 100:.1f}% → {util2 * 100:.1f}%")
    print(f"  跨 lane kernel 重叠: {r2.cross_lane_overlap() * 1000:.0f} ms")
    print("  注意: numpy 后端受 GIL 限制，毫秒数只反映执行结构，不代表硬件性能")


# ────────────────────────────────────────────────────────────
# 独立运行：vLLM 调度演示
# ────────────────────────────────────────────────────────────


VLLM_PARALLEL_SLOTS = (1, 2)
VLLM_MAX_SEQ = 16
VLLM_PREFILL_LEN = 4
VLLM_MAX_NEW_TOKENS = 3
VLLM_PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8]]


def _tiny_model_dir(root: Path) -> Path:
    """存一个随机初始化的微型 Llama，作为 vLLM 的模型目录。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=VLLM_MAX_SEQ,
    )).eval().half()
    path = root / "tiny-llama"
    model.save_pretrained(path)
    return path


def _vllm_case(model_dir: Path, parallel_slots: int) -> VllmProfileResult:
    """用 vLLM 真引擎跑一次生成，返回 PIMWorker 的调度记录。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # 引擎与 worker 必须同进程，否则拿不到 worker 实例，也驱动不了 NumpyBackend。
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["PIM_PARALLEL_SLOTS"] = str(parallel_slots)
    os.environ["PIM_PREFILL_SEQ_LEN"] = str(VLLM_PREFILL_LEN)
    os.environ.setdefault("PIM_MRAM_BYTES", str(1 << 28))

    from vllm import LLM, SamplingParams

    from backend.vllm_executor import PIMExecutor

    llm = LLM(
        model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
        max_model_len=VLLM_MAX_SEQ, max_num_seqs=len(VLLM_PROMPTS),
        # 一步内放下所有并发请求的整段 prompt，static 图不支持分块预填充。
        max_num_batched_tokens=len(VLLM_PROMPTS) * VLLM_PREFILL_LEN,
        enable_prefix_caching=False, gpu_memory_utilization=0.2,
        distributed_executor_backend=PIMExecutor,
    )
    # 单进程模式下引擎直接握着 PIMWorker：model_executor → wrapper → worker。
    worker = llm.llm_engine.model_executor.driver_worker.worker
    prof = profile_vllm(worker)
    try:
        llm.generate(
            [{"prompt_token_ids": prompt} for prompt in VLLM_PROMPTS],
            SamplingParams(max_tokens=VLLM_MAX_NEW_TOKENS, temperature=0.0),
        )
    finally:
        result = prof.stop()
        llm.llm_engine.engine_core.shutdown()
    return result


def demo_vllm() -> None:
    """比较 `PIM_PARALLEL_SLOTS` 对 vLLM 调度的影响。"""
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _tiny_model_dir(Path(tmp))
        results: dict[int, VllmProfileResult] = {}
        for parallel_slots in VLLM_PARALLEL_SLOTS:
            print(f"\n编译并运行 vLLM，PIM_PARALLEL_SLOTS={parallel_slots} ...")
            results[parallel_slots] = _vllm_case(model_dir, parallel_slots)
            print_vllm_report(
                results[parallel_slots],
                title=f"vLLM 调度报告（P={parallel_slots}）")

    if len(results) < 2:
        return
    (p1, r1), (p2, r2) = sorted(results.items())
    print(f"\n{'=' * 40}")
    print(f"vLLM P={p1} vs P={p2}:")
    print(f"  最大并发请求: {r1.max_concurrency()} → {r2.max_concurrency()}")
    print(f"  解码平均并行效率: {r1.mean_efficiency(False):.2f} → "
          f"{r2.mean_efficiency(False):.2f}")
    print(f"  激活窗口冲突: {len(r1.slot_conflicts())} → {len(r2.slot_conflicts())}")
    print("  效率接近 2 说明这一步的请求真的同时在跑，接近 1 说明是串行")


if __name__ == "__main__":
    if "vllm" in sys.argv[1:]:
        demo_vllm()
    else:
        complete_7b()
