"""执行计划调度器：计划序提交 + 逐命令形状绑定（动态模式）。"""

from __future__ import annotations

import threading

from contracts.exec_plan import ExecutionPlan
from runtime.shape_bind import bind_command_shapes, dynamic_shape_enabled

# waits 引用校验按计划身份只做一次；计划常驻 CompiledModel，id 不会复用。
_VALIDATED: set[int] = set()


def _validate_plan(plan: ExecutionPlan) -> None:
    ids = {cmd.id for cmd in plan.commands}
    unknown = {dep for cmd in plan.commands for dep in cmd.waits
               if dep not in ids}
    if unknown:
        raise ValueError(f"计划依赖了不存在的命令：{sorted(unknown)}")
    _VALIDATED.add(id(plan))


class EventScheduler:
    """按计划序把命令交给 HAL，完成等待直接落在命令 future 上。

    D1 的就绪驱动提交经剖析证伪（保序约束下相对全量提交无收益），调度器
    保留的价值是逐命令形状绑定（PIM_DYN_SHAPE）与统一的动态入口。
    """

    def __init__(self, plan: ExecutionPlan, hal, ctx=None) -> None:
        self.plan = plan
        self.hal = hal
        self.ctx = ctx
        self.events = hal.events_table(ctx)
        self.error: BaseException | None = None
        self._events: list = []  # 本次 run 提交的 event，wait() 逐个直等
        self._thread: threading.Thread | None = None
        _validate_plan(plan)

    def run(self, drain: bool = True) -> dict[int, object]:
        """启动提交线程；`drain=True` 时等全部命令完成再返回。"""
        self._thread = threading.Thread(
            target=self._loop, name="event-scheduler", daemon=True)
        self._thread.start()
        if drain:
            self.wait()
        return self.events

    def wait(self) -> dict[int, object]:
        """等待本次 run 的全部命令完成；提交或执行异常在这里抛出。

        旧的「逐命令回调 + 条件变量 notify」让每条命令完成都唤醒主线程一次
        （万级命令 = 万级线程切换，4L decode 每步多耗 ~1s），改为 join 提交
        线程后逐 future 等待，与静态路径同构。
        """
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise self.error
        for event in self._events:
            event.future.result()
        return self.events

    def _loop(self) -> None:
        try:
            env = self._shape_env() if dynamic_shape_enabled() else None
            for cmd in self.plan.commands:
                if env:
                    cmd = bind_command_shapes(cmd, env)
                event = self.hal.submit(cmd, self.ctx)
                self.events[cmd.id] = event
                self._events.append(event)
        except BaseException as exc:
            self.error = exc

    def _shape_env(self) -> dict[str, int]:
        """本次执行的符号维取值：T = 真实 token 数。"""
        num_tokens = (None if self.ctx is None else self.ctx.num_tokens)
        if num_tokens is None:
            num_tokens = self.hal.bound_num_tokens
        return {"T": int(num_tokens)} if num_tokens else {}
