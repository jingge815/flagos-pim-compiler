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
    allow_device_to_device: bool = False


@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]
    future: Future[object]


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
        self._dpu_set: DpuSet = dpu_alloc(config.num_dpus, mram_bytes=config.mram_bytes_per_dpu)
        self._alloc = MRAMAllocator(config.mram_bytes_per_dpu)

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
            event = Event(
                "dpu",
                self._submit_on_dpu(int(cmd.dpu_id), deps, kernel, self, int(cmd.dpu_id), cmd),
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
