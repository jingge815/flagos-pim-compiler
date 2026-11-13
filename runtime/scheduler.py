"""事件驱动调度器：按依赖就绪度提交命令，gate 在调度侧异步解析。"""

from __future__ import annotations

import threading
from dataclasses import replace

from contracts.exec_plan import Command, ExecutionPlan
from runtime.shape_bind import bind_command_shapes, dynamic_shape_enabled


class EventScheduler:
    """把静态计划交给运行期调度：依赖完成才提交，跳过命令不进执行队列。

    D1 仍以计划的 `waits` 作为依赖 DAG（含 host 闭包的数据依赖），区别是提交
    时机由事件回调驱动，不再一次性全量 submit；gate 命令在依赖就绪后直接判定
    跳过/提交，主线程不再阻塞在 `hal.submit` 里等 dispatch。
    """

    def __init__(self, plan: ExecutionPlan, hal, ctx=None) -> None:
        self.plan = plan
        self.hal = hal
        self.ctx = ctx
        self.events = hal.events_table(ctx)
        self.commands = {cmd.id: cmd for cmd in plan.commands}
        self.waits = {cmd.id: set(cmd.waits) for cmd in plan.commands}
        unknown = {dep for waits in self.waits.values() for dep in waits
                   if dep not in self.commands}
        if unknown:
            raise ValueError(f"计划依赖了不存在的命令：{sorted(unknown)}")
        self.done: set[int] = set()
        self.started: set[int] = set()
        self.error: BaseException | None = None
        self._cv = threading.Condition()
        self._thread: threading.Thread | None = None

    def run(self, drain: bool = True) -> dict[int, object]:
        """启动调度线程；`drain=True` 时等全部命令完成再返回。"""
        self._thread = threading.Thread(
            target=self._loop, name="event-scheduler", daemon=True)
        self._thread.start()
        if drain:
            self.wait()
        return self.events

    def wait(self) -> dict[int, object]:
        """等待全部命令完成；调度线程里的异常在这里抛出。"""
        with self._cv:
            self._cv.wait_for(
                lambda: self.error is not None
                or len(self.done) == len(self.commands))
            if self.error is not None:
                raise self.error
        return self.events

    def _loop(self) -> None:
        while True:
            with self._cv:
                if self.error is not None or len(self.done) == len(self.commands):
                    self._cv.notify_all()
                    return
                ready = [
                    cid for cid in self.commands
                    if cid not in self.started and self.waits[cid] <= self.done]
                if not ready:
                    self._cv.wait(timeout=0.05)
                    continue
                for cid in ready:
                    self.started.add(cid)
            for cid in sorted(ready):
                try:
                    self._submit(self.commands[cid])
                except BaseException as exc:  # 调度线程不能静默退出
                    with self._cv:
                        self.error = exc
                        self._cv.notify_all()
                    return

    def _shape_env(self) -> dict[str, int]:
        """本次执行的符号维取值：T = 真实 token 数。"""
        num_tokens = (None if self.ctx is None else self.ctx.num_tokens)
        if num_tokens is None:
            num_tokens = self.hal.bound_num_tokens
        return {"T": int(num_tokens)} if num_tokens else {}

    def _submit(self, cmd: Command) -> None:
        if self.plan is not None and dynamic_shape_enabled():
            env = self._shape_env()
            if env:
                cmd = bind_command_shapes(cmd, env)
        if "gate" in cmd.payload and self.hal.moe_dispatch is not False:
            gate_id = int(cmd.payload["gate"])
            lane_key = 0 if self.ctx is None else int(self.ctx.lane_key)
            expert = int(cmd.payload["expert"])
            stats = self.hal.gate_stats.setdefault(
                (lane_key, gate_id), {"submitted": 0, "skipped": 0})
            if expert in self.hal.gate_skips.get((lane_key, gate_id), ()):
                stats["skipped"] += 1
                event = self.hal.make_skipped_event(
                    cmd, lane_key, self.events, self.ctx)
            else:
                stats["submitted"] += 1
                event = self.hal.submit(self._ungated(cmd), self.ctx)
        else:
            event = self.hal.submit(cmd, self.ctx)
        self.events[cmd.id] = event
        event.future.add_done_callback(
            lambda _future, cid=cmd.id: self._on_done(cid))

    @staticmethod
    def _ungated(cmd: Command) -> Command:
        """跳过 HAL 的同步 gate 分支；依赖已由调度器满足，waits 保留做记录。"""
        payload = {key: value for key, value in cmd.payload.items()
                   if key != "gate"}
        return replace(cmd, payload=payload)

    def _on_done(self, cid: int) -> None:
        with self._cv:
            self.done.add(cid)
            self._cv.notify_all()
