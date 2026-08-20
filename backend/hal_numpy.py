from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

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


class DPUDevice:
    def __init__(self, dpu_id: int, capacity: int) -> None:
        self.dpu_id = dpu_id
        self.mram = np.zeros(capacity, dtype=np.uint8)
        self.alloc = MRAMAllocator(capacity)


class NumpyBackend:
    def __init__(self, config: NumpyBackendConfig) -> None:
        if config.allow_device_to_device:
            raise ValueError("device-to-device copies are not supported in Phase 1")
        self.config = config
        self._kernels = KernelStubRegistry()
        self._pool = ThreadPoolExecutor(max_workers=config.num_dpus + 1)
        self._stream_locks = {dpu_id: Lock() for dpu_id in range(config.num_dpus)}
        self._stream_tail: dict[int, Future[object]] = {}
        self._events_by_id: dict[int, Event] = {}
        self._devices = {
            dpu_id: DPUDevice(dpu_id, config.mram_bytes_per_dpu)
            for dpu_id in range(config.num_dpus)
        }

    def register_plan(self, plan: object) -> None:
        self._plan = plan

    def register_kernel(self, name: str, fn: Callable[["NumpyBackend", int, Command], None]) -> None:
        self._kernels.register(name, fn)

    def copy_to_dpu(self, dpu_id: int, offset: int, data: np.ndarray) -> None:
        blob = np.ascontiguousarray(data).view(np.uint8)
        dev = self._devices[dpu_id]
        dev.alloc.check(offset, blob.nbytes)
        dev.mram[offset : offset + blob.nbytes] = blob.reshape(-1)

    def copy_from_dpu(
        self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        dev = self._devices[dpu_id]
        dev.alloc.check(offset, size)
        blob = dev.mram[offset : offset + size].copy()
        return blob.view(dtype).reshape(shape).copy()

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
            return operation(cmd)
        return None

    def wait(self, event: Event, timeout: float | None = None) -> object:
        return event.future.result(timeout=timeout)

    def query(self, event: Event) -> bool:
        return event.future.done()
