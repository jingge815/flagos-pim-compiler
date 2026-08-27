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
    """两个 tasklet 在同一 epoch 内访问重叠地址且至少一方写——漏加 barrier 或
    行区间切分算错，真实硬件上这就是一次数据竞争。"""


class HazardTracker:
    """单次 launch 命令内、一个 epoch（两次 barrier 之间）的访问记录。

    按方案确认的模型：多个 tasklet 按固定顺序（0..num_tasklets-1）依次跑完
    自己的工作分片，不是真并发（不用 pthread，保持数值确定可复现）。每次
    WRAM/MRAM 读写在 `record()` 里就地和本 epoch 已有记录比对——顺序模拟下
    后到的访问和已记录的比较即可发现冲突，不需要等 epoch 结束再批量扫描。
    命中重叠即抛异常，模拟真实硬件上"少加 barrier 导致数据竞争"的后果。
    """

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
    """问题 6 的异步 HAL：submit/wait/query + 每 DPU 一条有序流。

    存储与 DMA 底座架在 backend/dpu_sdk（厂商 SDK 的 numpy 镜像）之上：本类只提供
    Command/Event 异步语义，MRAM 字节数组由 dpu_sdk 的机器持有，通信库与编排器共享
    同一份伪硬件状态。
    """

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
        # 每次 launch 命令私有的 HazardTracker——多个 DPU 的 launch 在
        # ThreadPoolExecutor 的不同 worker 线程上并发跑，不能挂在 self 的
        # 单一属性上，否则会互相踩；见 submit() 的 "launch" 分支。
        self._tracker_local = threading.local()

    def reset_events(self) -> None:
        """清空命令 id -> Event 的查找表（问题 6 `execute_plan` 每次调用前必调）。

        命令 id 只在单次 `build_execution_plan` 产出的一份 `ExecutionPlan`
        内唯一，两图各自从 0 编号；decode 循环对同一份 decode_plan 重复调
        `execute_plan` 多次（每步一次），id 会重复出现。`_events_by_id` 是
        这唯一会跨调用累积、导致 `submit` 误判"重复 id"的状态——不清空
        `_stream_tail`（每 DPU 的异步 future 链要跨步真实串行，代表"这台
        DPU 上一步的最后一条命令必须先跑完"，这是 KV cache 跨步累积写入
        MRAM 的正确性所需要的，问题 6 三.(1) 的"编排器持有横跨两张图的状态"
        正是靠它落地）。
        """
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
        """DPU `dpu_id` 的 MRAM 起始地址（`ctypes` 裸指针，`int`）。

        供 `opcompiler_bridge` 编译出的 C kernel 直接原地读写用——`read_local`/
        `write_local` 永远拷贝，这里给的是内存的地址，跟 `dpu_sdk.py` 里
        `_Dpu.mram`（每 DPU 一块独立连续 `np.zeros(mram_bytes, dtype=uint8)`）
        是同一份存储，调用方对着这个地址加 offset 写等价于直接改 MRAM。

        调用方必须保证：(1) 只在本 DPU 的 launch 线程内使用，不跨线程持有；
        (2) offset+length 落在 `mram_bytes_per_dpu` 之内——本方法不做越界检查，
        因为它连长度都不知道，检查在编译出的 kernel 自己算下标之前完成。
        """
        _, dpu = self._dpu_set.dpu(dpu_id)._member()
        return dpu.mram.ctypes.data

    def wram_ptr(self, dpu_id: int) -> int:
        """DPU `dpu_id` 的 WRAM 起始地址（`ctypes` 裸指针），镜像 `raw_mram_ptr`。

        供 opcompiler_bridge 编译出的多 tasklet C kernel 用——每个 tasklet
        把自己的行区间 snapshot 进 WRAM 再算，不是像单 tasklet 版本一样
        直接算 MRAM。
        """
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
        """问题 6 编排器的输入绑定入口（方案三.(3) `execute_plan` 的 `hal.bind_inputs`）。

        `values`：本次调用的图 placeholder 节点名 -> 具体值（如
        `{"input_ids": tensor, "causal_mask": tensor}`），供
        `runtime/exec_plan_gen.py` 的 `_resolve_value` 在运行时读取。
        `pos`：写 KV 的位置（= 编排器 `DecodeState.valid_len`），KV/SDPA
        handler 用；不是图输入，图本身不含"位置"这个 placeholder。每次
        `execute_plan` 前必须重新绑定，本方法不做跨调用累积。
        """
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
        """取某条已提交命令的返回值（问题 6 host handler 读上游数值用，见
        `runtime/exec_plan_gen.py` 模块 docstring 的"编译期只捕获命令 id、
        运行时按 id 查具体值"设计）。"""
        return self._events_by_id[cmd_id].future.result()
