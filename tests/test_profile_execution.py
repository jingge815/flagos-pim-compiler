"""验证 vLLM 调度采集的指标计算与步归属，不需要 vLLM 环境。"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.profile_execution import (  # noqa: E402
    StepRecord, VllmProfileResult, profile_vllm)


def _sched(tokens: dict[str, int]) -> SimpleNamespace:
    """伪造 SchedulerOutput，采集器只读其中的 num_scheduled_tokens。"""
    return SimpleNamespace(num_scheduled_tokens=tokens)


def _step(step, tasks, *, tokens=(1, 1)) -> StepRecord:
    """按给定区间拼一条步记录，任务形如 (请求, 窗口, 起, 止)。"""
    return StepRecord(step=step, req_ids=[t[0] for t in tasks], tokens=list(tokens),
                      start=0.0, end=1.0, tasks=tasks)


class _FakeWorker:
    """替身 worker：每步按脚本调用 `_step`，可选在某个请求上抛异常。"""

    def __init__(self, plan: list[list[tuple[str, int]]], boom: str | None = None) -> None:
        self.plan = plan
        self.boom = boom
        self.calls = 0

    def execute_model(self, scheduler_output):
        tasks = self.plan[self.calls]
        self.calls += 1
        for req_id, slot in tasks:
            self._step(req_id, slot)
        return None

    def _step(self, req_id, slot_index=0):
        time.sleep(0.02)
        if req_id == self.boom:
            raise RuntimeError("wave 中途失败")
        return slot_index


def test_step_record_metrics() -> None:
    """错开的请求效率为 1，重叠的请求效率接近 2，并发度取同时刻峰値。"""
    serial = _step(0, [("a", 0, 0.0, 0.5), ("b", 1, 0.5, 1.0)])
    assert serial.concurrency == 1
    assert serial.efficiency == 1.0
    assert not serial.is_prefill

    parallel = StepRecord(step=0, req_ids=["a", "b"], tokens=[1, 4],
                          start=0.0, end=0.5,
                          tasks=[("a", 0, 0.0, 0.5), ("b", 1, 0.0, 0.5)])
    assert parallel.concurrency == 2
    assert parallel.efficiency == 2.0
    assert parallel.is_prefill


def test_slot_conflicts() -> None:
    """区间重叠且共用窗口才算冲突。"""
    assert _step(0, [("a", 0, 0.0, 0.5), ("b", 0, 0.2, 0.7)]).slot_conflicts() == [
        ("a", "b", 0)]
    assert _step(0, [("a", 0, 0.0, 0.2), ("b", 0, 0.2, 0.7)]).slot_conflicts() == []
    assert _step(0, [("a", 0, 0.0, 0.5), ("b", 1, 0.2, 0.7)]).slot_conflicts() == []


def test_mean_efficiency_skips_empty_steps() -> None:
    """没跑请求的步不参与平均，否则会把结果拉成 0。"""
    result = VllmProfileResult(steps=[
        _step(0, [("a", 0, 0.0, 0.5), ("b", 1, 0.0, 0.5)], tokens=(1, 1)),
        _step(1, [], tokens=(1,)),
    ])
    assert result.mean_efficiency(False) == 1.0
    assert result.mean_efficiency(True) == 0.0


def test_profiler_records_each_step() -> None:
    """采集器按 wave 记下请求、窗口与区间，stop 后还原 worker 的方法。"""
    worker = _FakeWorker([[("a", 0), ("b", 1)], [("a", 0)]])
    orig_execute_model, orig_step = worker.execute_model, worker._step
    prof = profile_vllm(worker)

    worker.execute_model(_sched({"a": 4, "b": 4}))
    worker.execute_model(_sched({"a": 1}))
    result = prof.stop()

    assert [s.req_ids for s in result.steps] == [["a", "b"], ["a"]]
    assert [s.tokens for s in result.steps] == [[4, 4], [1]]
    assert [t[1] for t in result.steps[0].tasks] == [0, 1]
    assert result.steps[0].is_prefill and not result.steps[1].is_prefill
    assert result.slot_conflicts() == []
    assert worker.execute_model == orig_execute_model
    assert worker._step == orig_step


def test_parallel_wave_shows_overlap() -> None:
    """两个请求真并发时效率明显大于 1（用线程模拟 wave）。"""

    class _WaveWorker(_FakeWorker):
        def execute_model(self, scheduler_output):
            threads = [
                threading.Thread(target=self._step, args=(req_id, slot))
                for req_id, slot in self.plan[self.calls]
            ]
            self.calls += 1
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return None

    worker = _WaveWorker([[("a", 0), ("b", 1)]])
    prof = profile_vllm(worker)
    worker.execute_model(_sched({"a": 1, "b": 1}))
    result = prof.stop()

    assert result.steps[0].concurrency == 2
    assert result.steps[0].efficiency > 1.4


def test_interrupted_wave_not_attributed_to_next_step() -> None:
    """某步中途失败时，还在跑的请求不能算进下一步。"""
    release = threading.Event()
    late: list[threading.Thread] = []

    class _LateWorker:
        """第一步起一个迟到的请求后立刻失败；第二步放它完成再返回。"""

        def __init__(self) -> None:
            self.steps = 0
            self.entered = threading.Event()

        def execute_model(self, scheduler_output):
            self.steps += 1
            if self.steps == 1:
                thread = threading.Thread(target=self._step, args=("late", 1))
                late.append(thread)
                thread.start()
                # 等它读好步号再失败，保证这条记录带的是第一步的标记。
                self.entered.wait(5.0)
                raise RuntimeError("wave 中途失败")
            release.set()
            late[0].join()
            return None

        def _step(self, req_id, slot_index=0):
            self.entered.set()
            if req_id == "late":
                release.wait(5.0)
            return 0

    worker = _LateWorker()
    prof = profile_vllm(worker)
    try:
        worker.execute_model(_sched({"a": 1, "b": 1}))
    except RuntimeError:
        pass
    worker.execute_model(_sched({"c": 1}))
    result = prof.stop()

    assert len(result.steps) == 2
    assert [s.req_ids for s in result.steps] == [["a", "b"], ["c"]]
    assert all(s.tasks == [] for s in result.steps)
    assert result.mean_efficiency(False) == 0.0
