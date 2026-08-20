# NumpyBackend Design

**Goal:** Build the runtime validation backend that simulates multiple isolated DPU MRAM spaces on CPU for problems 3, 6, 7, and 8.

## Scope

`NumpyBackend` is the runtime-only fake HAL. It does not do graph partitioning, placement propagation, KV layout planning, or memory planning. Its job is to execute and validate the plans produced by compile-time passes while preserving the target hardware semantics:

- one independent MRAM address space per DPU;
- Host↔DPU DMA only by default;
- asynchronous submit / wait / query semantics;
- direct local reads and writes for KV and memory-layout checks;
- kernel-stub execution for runtime logic.

`hal_vendor.py` will expose the same public surface later, but it stays a stub in phase 1.

## Public Interface

`backend/hal_numpy.py` will expose:

```python
@dataclass
class NumpyBackendConfig:
    num_dpus: int
    mram_bytes_per_dpu: int
    allow_device_to_device: bool = False


@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]
    future: object


class NumpyBackend:
    def register_kernel(self, name: str, fn: Callable[["NumpyBackend", int, object], None]) -> None: ...
    def register_plan(self, plan: object) -> None: ...
    def copy_to_dpu(self, dpu_id: int, offset: int, data: np.ndarray) -> None: ...
    def copy_from_dpu(self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray: ...
    def push_xfer(self, transfers: list[tuple[int, int, np.ndarray]]) -> None: ...
    def write_local(self, dpu_id: int, offset: int, data: np.ndarray) -> None: ...
    def read_local(self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray: ...
    def submit(self, cmd: object) -> Event: ...
    def wait(self, event: Event, timeout: float | None = None) -> None: ...
    def query(self, event: Event) -> bool: ...
```

`contracts/exec_plan.py` and `contracts/pim_tensor_spec.py` will provide the minimal shared plan objects that `NumpyBackend` consumes.

## Internal Model

The implementation stays in one file but is split into private classes:

- `DeviceManager`: owns all DPU instances and the global DPU id registry.
- `DPUDevice`: owns one local MRAM buffer and one ordered stream context.
- `MRAMAllocator`: validates ranges, lifetimes, and per-DPU isolation.
- `StreamContext`: serializes commands per DPU and returns events.
- `EventSync`: wraps future completion and timeout handling.
- `DMAEngine`: implements Host↔DPU copies and the optional simulated fan-out path.
- `KernelStubRegistry`: maps kernel names to Python callables.

Each DPU has one byte-addressed MRAM buffer. Reads and writes are copied, not shared, so writing DPU0 cannot affect DPU1.

## Execution Semantics

`submit()` dispatches by command type:

- `launch` runs a registered kernel stub on the target DPU stream.
- `dma_in` and `dma_out` simulate Host↔DPU copies.
- host-side ops execute on CPU and complete immediately.

Kernel stubs receive `(backend, dpu_id, cmd)` and are responsible for reading
and writing local MRAM through `read_local()` / `write_local()`.

`wait()` blocks on the returned event. `query()` only checks completion state. In phase 1, synchronous behavior is allowed, but the API shape stays asynchronous.

`allow_device_to_device=False` means no DPU-to-DPU direct copy exists. If enabled for experiments, the implementation may route through host staging, but it must stay explicit.

## Memory Rules

`write_local()` and `read_local()` are the debug primitives for problems 7 and 8:

- offsets are byte offsets inside one DPU only;
- bounds are checked against that DPU's MRAM size;
- no cross-DPU aliasing is allowed;
- reads return copies, not views, so inspection cannot mutate storage accidentally.

This is enough to validate KV append/read/mask logic and to catch bad `mram_offset` placement early.

## GPT-2 Smoke Use

The backend does not load GPT-2 itself. It must, however, be able to carry the random GPT-2-shaped tensors used by the runtime tests and execute the corresponding command plan without any model-specific code.

## Tests

`tests/test_hal_numpy.py` will cover:

- DPU isolation: writes to one DPU do not change another DPU;
- copy semantics: `copy_to_dpu` / `copy_from_dpu` round-trip tensor bytes correctly;
- async semantics: `submit` returns incomplete events until the worker finishes;
- host/DPU and local read/write support for KV and memory-plan offsets;
- optional `push_xfer` fan-out behavior;
- plan execution against a small kernel stub.

## Reference

This design covers `docs/spec.md:605-825` (problem 3), `docs/spec.md:1248-1659` (problem 6), `docs/spec.md:1660-1918` (problem 7), and `docs/spec.md:1919-2156` (problem 8).
