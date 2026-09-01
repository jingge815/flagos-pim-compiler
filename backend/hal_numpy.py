import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass
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


@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]
    future: Future[object]


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
    """提供异步命令执行、每 DPU 有序流和共享 MRAM 的 HAL。"""

    def __init__(self, config: NumpyBackendConfig) -> None:
        if config.allow_device_to_device:
            raise ValueError("device-to-device copies are not supported in Phase 1")
        self.config = config
        self._kernels = KernelStubRegistry()
        self._pool = ThreadPoolExecutor(max_workers=config.num_dpus + 1)
        self._stream_locks = {dpu_id: Lock() for dpu_id in range(config.num_dpus)}
        self._stream_tail: dict[int, Future[object]] = {}
        self._events_by_id: dict[int, Event] = {}
        self._dpu_set: DpuSet = dpu_alloc(
            config.num_dpus, mram_bytes=config.mram_bytes_per_dpu,
            wram_bytes=config.wram_bytes_per_dpu,
        )
        self._alloc = MRAMAllocator(config.mram_bytes_per_dpu)
        # 每个并发执行的 launch 使用独立的冲突检测器。
        self._tracker_local = threading.local()

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

    def submit(self, cmd: Command) -> Event:
        if cmd.id in self._events_by_id:
            raise ValueError(f"duplicate command id {cmd.id}")
        deps = self._dependency_futures(cmd.waits)
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
                self._submit_on_dpu(int(cmd.dpu_id), deps, run_with_tracker),
            )
        elif cmd.op == "dma_in":
            data = np.ascontiguousarray(np.asarray(cmd.payload["data"])).copy()
            event = Event(
                "dma",
                self._submit_on_dpu(
                    int(cmd.dpu_id),
                    deps,
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
                    deps,
                    self.copy_from_dpu,
                    int(cmd.dpu_id),
                    int(cmd.payload["offset"]),
                    tuple(cmd.payload["shape"]),
                    np.dtype(cmd.payload["dtype"]),
                ),
            )
        elif cmd.dpu_id is None:
            event = Event("host", self._pool.submit(self._run_host_or_dma, cmd, deps))
        else:
            raise ValueError(f"unsupported command op {cmd.op!r}")
        self._events_by_id[cmd.id] = event
        return event

    def _dependency_futures(self, waits: list[int]) -> list[Future[object]]:
        deps: list[Future[object]] = []
        for wait in waits:
            event = self._events_by_id.get(wait)
            if event is None:
                raise ValueError(f"unknown dependency command id {wait}")
            deps.append(event.future)
        return deps

    def _submit_on_dpu(
        self,
        dpu_id: int,
        deps: list[Future[object]],
        fn: Callable[..., object],
        *args: object,
    ) -> Future[object]:
        with self._stream_locks[dpu_id]:
            prev = self._stream_tail.get(dpu_id)

            def run() -> object:
                if prev is not None:
                    prev.result()
                for dep in deps:
                    dep.result()
                return fn(*args)

            future = self._pool.submit(run)
            self._stream_tail[dpu_id] = future
            return future

    def _run_host_or_dma(self, cmd: Command, deps: list[Future[object]]) -> object:
        for dep in deps:
            dep.result()
        operation = cmd.payload.get("fn")
        if callable(operation):
            return operation(self, cmd)
        return None

    def wait(self, event: Event, timeout: float | None = None) -> object:
        return event.future.result(timeout=timeout)

    def query(self, event: Event) -> bool:
        return event.future.done()

    def bind_inputs(self, values: dict[str, object], *, pos: int | None = None) -> None:
        """绑定本次计划执行的占位符值和可选 KV 写入位置。"""
        self._bound_values = values
        self._bound_pos = pos

    def bound_value(self, name: str) -> object:
        """取本次绑定的某个 placeholder 值（`exec_plan_gen._resolve_value` 用）。"""
        return self._bound_values[name]

    @property
    def bound_pos(self) -> int | None:
        """取本次 `bind_inputs` 绑定的 `pos`（KV/SDPA handler 读写位置用）。"""
        return self._bound_pos

    def result_of(self, cmd_id: int) -> object:
        """返回指定命令的执行结果。"""
        return self._events_by_id[cmd_id].future.result()
