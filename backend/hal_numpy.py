import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Literal

import numpy as np

from backend.dpu_sdk import DpuSet, dpu_alloc, dpu_copy_from, dpu_copy_to
from contracts.exec_plan import Command


@dataclass
class NumpyBackendConfig:
    num_dpus: int
    mram_bytes_per_dpu: int
    wram_bytes_per_dpu: int = 65536
    allow_device_to_device: bool = False
    num_parallel_slots: int = 1       # 同一步并发的请求上下文数，只影响线程池尺寸。


@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]
    future: Future[object]


@dataclass
class ExecutionContext:
    """一次请求执行的独立输入、事件表和执行泳道。

    为空时所有字段都退回 `NumpyBackend` 的全局语义，旧的单请求路径不变。
    """

    lane_key: int = 0
    values: dict[str, object] = field(default_factory=dict)
    pos: int | None = None
    num_tokens: int | None = None
    events: dict[int, Event] = field(default_factory=dict)


@dataclass(frozen=True)
class _AccessRecord:
    """一次 tasklet 对 WRAM/MRAM 某个地址区间的访问，用于同一 epoch 内的重叠检测。"""

    tasklet_id: int
    loc: Literal["wram", "mram"]
    offset: int
    length: int
    is_write: bool

    def overlaps(self, other: "_AccessRecord") -> bool:
        return (
            self.loc == other.loc
            and self.offset < other.offset + other.length
            and other.offset < self.offset + self.length
        )


class TaskletHazardError(RuntimeError):
    """表示同一同步区间内的 tasklet 存在冲突访问。"""


class HazardTracker:
    """记录一个同步区间内的 tasklet 内存访问并检测写冲突。"""

    def __init__(self) -> None:
        self._epoch: list[_AccessRecord] = []

    def record(self, tasklet_id: int, loc: Literal["wram", "mram"],
               offset: int, length: int, is_write: bool) -> None:
        rec = _AccessRecord(tasklet_id, loc, offset, length, is_write)
        for prev in self._epoch:
            if prev.tasklet_id != rec.tasklet_id and prev.overlaps(rec) and (prev.is_write or rec.is_write):
                raise TaskletHazardError(
                    f"tasklet {prev.tasklet_id} 与 tasklet {rec.tasklet_id} 在 "
                    f"{rec.loc}[{rec.offset},{rec.offset + rec.length}) 无 barrier "
                    f"重叠访问（至少一方写）——缺少同步或切分区间算错"
                )
        self._epoch.append(rec)

    def check_and_advance(self) -> None:
        """遇到 barrier：验证已经在每次 record 时做完，这里只清空进入下一 epoch。"""
        self._epoch = []


class KernelStubRegistry:
    def __init__(self) -> None:
        self._kernels: dict[str, Callable[["NumpyBackend", int, Command], None]] = {}

    def register(self, name: str, fn: Callable[["NumpyBackend", int, Command], None]) -> None:
        self._kernels[name] = fn

    def lookup(self, name: str) -> Callable[["NumpyBackend", int, Command], None]:
        return self._kernels[name]


class MRAMAllocator:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def check(self, offset: int, nbytes: int) -> None:
        if offset < 0 or nbytes < 0 or offset + nbytes > self.capacity:
            raise ValueError(
                f"invalid MRAM range offset={offset} nbytes={nbytes} capacity={self.capacity}"
            )


class NumpyBackend:
    """提供异步命令执行、每 (上下文, DPU) 有序流和共享 MRAM 的 HAL。

    每台 DPU 的每个执行上下文有一条命令链：同链命令严格按提交序执行。执行
    计划依赖这一点，把同 DPU 的显式等待折叠成「只等最后一条前驱」。不同上下文
    （不同请求）的链互不排队，才能在一步里同时执行。host 命令没有这种链，
    必须逐条声明依赖。

    忙碌率是墙钟口径：某条链有命令在跑的时间占墙钟的比例，不是 CPU 占用。本后端
    受进程内 GIL 限制，链再多进程也只用到一两个核，解读忙碌率必须同时看进程核数。
    """

    def __init__(self, config: NumpyBackendConfig) -> None:
        if config.allow_device_to_device:
            raise ValueError("device-to-device copies are not supported in Phase 1")
        self.config = config
        self._kernels = KernelStubRegistry()
        lanes = max(1, config.num_parallel_slots)
        self._pool = ThreadPoolExecutor(
            max_workers=min(64, config.num_dpus * lanes + lanes + 1))
        # 键是 (lane_key, dpu_id)；lane_key=0 是旧的串行路径。
        self._stream_locks: dict[tuple[int, int], Lock] = {}
        self._stream_tail: dict[tuple[int, int], Future[object]] = {}
        self._stream_meta_lock = Lock()
        self._events_by_id: dict[int, Event] = {}
        self._ctx_local = threading.local()
        self._dpu_set: DpuSet = dpu_alloc(
            config.num_dpus, mram_bytes=config.mram_bytes_per_dpu,
            wram_bytes=config.wram_bytes_per_dpu,
        )
        self._alloc = MRAMAllocator(config.mram_bytes_per_dpu)
        # 每个并发执行的 launch 使用独立的冲突检测器。
        self._tracker_local = threading.local()
        # host 命令（comm 的 collect/writeback 等）共用机器的 `xfer_buffers`：
        # `dpu_prepare_xfer` 登记的缓冲区指针是按 DPU 存放的全局状态，两个并发的
        # host op 对同一台 DPU 做 push_xfer 时会互相拿走对方的缓冲区，把错误的
        # 数据写进各自不相交的目标区域——区间审计看不见，数值上却是真腐蚀。
        # host op 在伪硬件上本就当 DMA 引擎语义处理，串行化不改行为。
        self._host_lock = Lock()
        self._bound_values: dict[str, object] = {}
        self._bound_pos: int | None = None
        self._bound_num_tokens: int | None = None

    def reset_events(self) -> None:
        """清空本次执行的命令事件表，保留各 DPU 的执行顺序。"""
        self._events_by_id = {}

    @property
    def dpu_set(self) -> DpuSet:
        """底层 SDK 集合：通信库（comm/lowering.py）经此与 HAL 共享同一伪硬件。"""
        return self._dpu_set

    def register_plan(self, plan: object) -> None:
        self._plan = plan

    def register_kernel(self, name: str, fn: Callable[["NumpyBackend", int, Command], None]) -> None:
        self._kernels.register(name, fn)

    def copy_to_dpu(self, dpu_id: int, offset: int, data: np.ndarray) -> None:
        blob = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self._alloc.check(offset, blob.nbytes)
        dpu_copy_to(self._dpu_set.dpu(dpu_id), offset, blob, blob.nbytes)

    def copy_from_dpu(
        self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._alloc.check(offset, size)
        data = np.empty(shape, dtype=dtype)
        dpu_copy_from(self._dpu_set.dpu(dpu_id), offset, data, size)
        return data

    def write_local(self, dpu_id: int, offset: int, data: np.ndarray) -> None:
        self.copy_to_dpu(dpu_id, offset, data)

    def raw_mram_ptr(self, dpu_id: int) -> int:
        """返回指定 DPU 的 MRAM 起始地址，供已编译内核直接访问。"""
        _, dpu = self._dpu_set.dpu(dpu_id)._member()
        return dpu.mram.ctypes.data

    def wram_ptr(self, dpu_id: int) -> int:
        """返回指定 DPU 的 WRAM 起始地址，供已编译内核直接访问。"""
        _, dpu = self._dpu_set.dpu(dpu_id)._member()
        return dpu.wram.ctypes.data

    def record_access(self, tasklet_id: int, loc: Literal["wram", "mram"],
                       offset: int, length: int, is_write: bool) -> None:
        """多 tasklet kernel 每次访问 WRAM/MRAM 前调用，交给当前 launch 命令的
        `HazardTracker`（只在一次 launch 的执行期间有效，见 `submit`）。"""
        self._tracker_local.tracker.record(tasklet_id, loc, offset, length, is_write)

    def barrier(self) -> None:
        """多 tasklet kernel 遇到 barrier 时调用，清空当前 epoch 的访问记录。"""
        self._tracker_local.tracker.check_and_advance()

    def read_local(
        self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        return self.copy_from_dpu(dpu_id, offset, shape, dtype)

    def push_xfer(self, transfers: list[tuple[int, int, np.ndarray]]) -> None:
        for dpu_id, offset, data in transfers:
            self.copy_to_dpu(dpu_id, offset, data)

    def submit(self, cmd: Command, ctx: ExecutionContext | None = None) -> Event:
        """提交一条命令；`ctx` 为空时沿用全局事件表和 0 号泳道。"""
        events = self._events_by_id if ctx is None else ctx.events
        if cmd.id in events:
            raise ValueError(f"duplicate command id {cmd.id}")
        deps = self._dependency_futures(cmd.waits, events)
        lane_key = 0 if ctx is None else int(ctx.lane_key)
        if cmd.op == "launch":
            kernel = self._kernels.lookup(str(cmd.payload["kernel"]))
            tracker = HazardTracker()

            def run_with_tracker(dpu_id: int = int(cmd.dpu_id), cmd: Command = cmd,
                                  tracker: HazardTracker = tracker) -> object:
                self._tracker_local.tracker = tracker
                try:
                    return kernel(self, dpu_id, cmd)
                finally:
                    del self._tracker_local.tracker

            event = Event(
                "dpu",
                self._submit_on_dpu(int(cmd.dpu_id), lane_key, deps, ctx, run_with_tracker),
            )
        elif cmd.op == "dma_in":
            data = np.ascontiguousarray(np.asarray(cmd.payload["data"])).copy()
            event = Event(
                "dma",
                self._submit_on_dpu(
                    int(cmd.dpu_id),
                    lane_key,
                    deps,
                    ctx,
                    self.copy_to_dpu,
                    int(cmd.dpu_id),
                    int(cmd.payload["offset"]),
                    data,
                ),
            )
        elif cmd.op == "dma_out":
            event = Event(
                "dma",
                self._submit_on_dpu(
                    int(cmd.dpu_id),
                    lane_key,
                    deps,
                    ctx,
                    self.copy_from_dpu,
                    int(cmd.dpu_id),
                    int(cmd.payload["offset"]),
                    tuple(cmd.payload["shape"]),
                    np.dtype(cmd.payload["dtype"]),
                ),
            )
        elif cmd.dpu_id is None:
            event = Event("host", self._pool.submit(self._run_host_or_dma, cmd, deps, ctx))
        else:
            raise ValueError(f"unsupported command op {cmd.op!r}")
        events[cmd.id] = event
        return event

    def _dependency_futures(self, waits: list[int],
                            events: dict[int, Event]) -> list[Future[object]]:
        deps: list[Future[object]] = []
        for wait in waits:
            event = events.get(wait)
            if event is None:
                raise ValueError(f"unknown dependency command id {wait}")
            deps.append(event.future)
        return deps

    def _submit_on_dpu(
        self,
        dpu_id: int,
        lane_key: int,
        deps: list[Future[object]],
        ctx: ExecutionContext | None,
        fn: Callable[..., object],
        *args: object,
    ) -> Future[object]:
        key = (lane_key, dpu_id)
        with self._stream_meta_lock:
            lock = self._stream_locks.get(key)
            if lock is None:
                lock = self._stream_locks[key] = Lock()
        with lock:
            prev = self._stream_tail.get(key)

            def run() -> object:
                if prev is not None:
                    prev.result()
                for dep in deps:
                    dep.result()
                with self._install_context(ctx):
                    return fn(*args)

            future = self._pool.submit(run)
            self._stream_tail[key] = future
            return future

    @contextmanager
    def _install_context(self, ctx: ExecutionContext | None):
        """在 worker 线程里安装本次命令的上下文，退出时清理。"""
        if ctx is None:
            yield
            return
        self._ctx_local.ctx = ctx
        try:
            yield
        finally:
            del self._ctx_local.ctx

    def _current_context(self) -> ExecutionContext | None:
        return getattr(self._ctx_local, "ctx", None)

    def _run_host_or_dma(self, cmd: Command, deps: list[Future[object]],
                         ctx: ExecutionContext | None) -> object:
        for dep in deps:
            dep.result()
        operation = cmd.payload.get("fn")
        if callable(operation):
            with self._install_context(ctx):
                with self._host_lock:
                    return operation(self, cmd)
        return None

    def wait(self, event: Event, timeout: float | None = None) -> object:
        return event.future.result(timeout=timeout)

    def query(self, event: Event) -> bool:
        return event.future.done()

    def bind_inputs(self, values: dict[str, object], *, pos: int | None = None,
                    num_tokens: int | None = None) -> None:
        """绑定本次计划执行的占位符值、KV 写入位置和真实 token 数。"""
        self._bound_values = values
        self._bound_pos = pos
        self._bound_num_tokens = num_tokens

    def bound_value(self, name: str) -> object:
        """取本次绑定的某个 placeholder 值（`exec_plan_gen._resolve_value` 用）。"""
        ctx = self._current_context()
        if ctx is not None:
            return ctx.values[name]
        return self._bound_values[name]

    def bound_flag(self, name: str) -> bool:
        """本次绑定里附加的布尔开关；没设置就是 False。

        用来下发「图上的掩码就是标准因果掩码」这类编译期看不出来的事实。
        """
        ctx = self._current_context()
        values = ctx.values if ctx is not None else self._bound_values
        return bool((values or {}).get(name, False))

    @property
    def bound_pos(self) -> int | None:
        """取本次 `bind_inputs` 绑定的 `pos`（KV/SDPA handler 读写位置用）。"""
        ctx = self._current_context()
        return ctx.pos if ctx is not None else self._bound_pos

    @property
    def bound_num_tokens(self) -> int | None:
        """取本次 `bind_inputs` 绑定的真实 token 数（限制 KV 写入行数）。"""
        ctx = self._current_context()
        return ctx.num_tokens if ctx is not None else self._bound_num_tokens

    def result_of(self, cmd_id: int) -> object:
        """返回指定命令的执行结果。"""
        ctx = self._current_context()
        events = self._events_by_id if ctx is None else ctx.events
        return events[cmd_id].future.result()
