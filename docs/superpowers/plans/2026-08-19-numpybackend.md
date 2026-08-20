# NumpyBackend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CPU-only fake HAL that simulates isolated DPU MRAM spaces, DMA, async execution, and kernel stubs for problems 3, 6, 7, and 8.

**Architecture:** Keep all backend behavior in `backend/hal_numpy.py` as one file with private helper classes only. Shared runtime contracts live in `contracts/`, and `hal_vendor.py` stays API-compatible but unimplemented. Tests exercise DPU isolation, copy semantics, async submit/query, local MRAM access, and a GPT-2-shaped smoke path.

**Tech Stack:** Python 3.10, NumPy, PyTorch/Transformers for tests only, pytest.

## Global Constraints

- Phase 1 only: do not add graph partitioning, placement propagation, memory planning, KV policy, or a runtime executor.
- Keep `backend/hal_numpy.py` as a single module; helper classes stay private and internal.
- `copy_to_dpu` and `copy_from_dpu` are synchronous communication helpers; async semantics live only in `submit` / `wait` / `query`.
- Host↔DPU is the default transfer model; DPU-to-DPU direct copy remains opt-in and explicit.
- `read_local` / `write_local` must use byte offsets, bounds checks, and copy semantics.
- GPT-2 is only a smoke fixture for realistic tensor shapes and payloads; backend code must stay model-agnostic.
- Preserve existing import paths and unrelated metadata.

---

## File Structure

- Modify: `contracts/pim_tensor_spec.py` - minimal `Placement`, `TensorShardDetail`, and `PIMTensorSpec`.
- Modify: `contracts/exec_plan.py` - minimal `Access`, `Command`, and `ExecutionPlan`.
- Modify: `backend/hal_numpy.py` - full fake HAL plus private runtime helpers.
- Modify: `backend/hal_vendor.py` - API-compatible stub.
- Modify: `backend/__init__.py` - convenient exports.
- Create: `tests/test_hal_numpy.py` - contract, isolation, DMA, async, and GPT-2 smoke tests.
- Create: `docs/hal_numpy.md` - short public module note.

### Task 1: Lock the Shared Runtime Contracts

**Files:**
- Create: `tests/test_hal_numpy.py`
- Modify: `contracts/pim_tensor_spec.py`
- Modify: `contracts/exec_plan.py`

**Interfaces:**
- Produces: `Placement`, `TensorShardDetail`, `PIMTensorSpec`, `Access`, `Command`, `ExecutionPlan`.

- [ ] **Step 1: Add failing contract tests**

```python
from contracts.exec_plan import Access, Command, ExecutionPlan
from contracts.pim_tensor_spec import Placement, PIMTensorSpec, TensorShardDetail


def test_runtime_contracts_have_the_expected_fields() -> None:
    shard = TensorShardDetail(
        dpu_id=0,
        shard_dim=1,
        start_idx=0,
        end_idx=4,
        local_shape=(2, 2),
    )
    spec = PIMTensorSpec(
        device="dpu",
        placement=Placement(kind="Shard", dim=1),
        residency="pinned",
        pinned_dpu_id=0,
        shard_map={0: shard},
        reduce_type=None,
    )
    plan = ExecutionPlan(
        commands=[
            Command(
                id=0,
                op="launch",
                dpu_id=0,
                payload={"kernel": "noop"},
                reads=[],
                writes=[Access(("dpu", 0), 0, 16)],
                waits=[],
            )
        ]
    )
    assert spec.shard_map[0].local_shape == (2, 2)
    assert plan.commands[0].writes[0].offset == 0
```

- [ ] **Step 2: Run the new test file and confirm it fails before implementation**

Run:

```bash
python -m pytest tests/test_hal_numpy.py -q
```

Expected: import/attribute failure because the runtime contracts do not exist yet.

- [ ] **Step 3: Implement the minimal shared dataclasses**

```python
@dataclass(frozen=True)
class Placement:
    kind: Literal["Shard", "Replicate", "Partial"]
    dim: int | None = None
    reduce_type: str | None = None
```

```python
@dataclass(frozen=True)
class TensorShardDetail:
    dpu_id: int
    shard_dim: int
    start_idx: int
    end_idx: int
    local_shape: tuple[int, ...]
    mram_offset: int = 0
```

```python
@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None
```

```python
@dataclass(frozen=True)
class Access:
    loc: tuple[str, int | None]
    offset: int
    length: int
```

```python
@dataclass
class Command:
    id: int
    op: Literal["launch", "dma_in", "dma_out", "host_reduce", "host_concat", "host_permute", "host_slice", "host_op"]
    dpu_id: int | None
    payload: dict[str, object]
    reads: list[Access] = field(default_factory=list)
    writes: list[Access] = field(default_factory=list)
    waits: list[int] = field(default_factory=list)
```

```python
@dataclass
class ExecutionPlan:
    commands: list[Command]
```

- [ ] **Step 4: Re-run the focused test**

Run:

```bash
python -m pytest tests/test_hal_numpy.py -q
```

Expected: the contract test passes.

### Task 2: Implement `NumpyBackend` in One File

**Files:**
- Modify: `backend/hal_numpy.py`
- Modify: `backend/__init__.py`
- Modify: `tests/test_hal_numpy.py`

**Interfaces:**
- Consumes: the contracts from Task 1.
- Produces: `NumpyBackendConfig`, `Event`, `NumpyBackend`, and the private helper classes inside `hal_numpy.py`.

- [ ] **Step 1: Add failing backend tests**

```python
def test_dpu_isolation_and_copy_round_trip() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=64))
    payload = np.arange(8, dtype=np.int32)
    backend.copy_to_dpu(0, 0, payload)
    assert np.array_equal(backend.copy_from_dpu(0, 0, payload.shape, payload.dtype), payload)
    assert np.array_equal(
        backend.copy_from_dpu(1, 0, payload.shape, payload.dtype),
        np.zeros_like(payload),
    )
```

```python
def test_submit_wait_query_and_kernel_stub() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=64))
    gate = threading.Event()

    def fill(hal, dpu_id, cmd) -> None:
        gate.wait()
        hal.write_local(dpu_id, cmd.payload["offset"], np.asarray(cmd.payload["value"], dtype=np.int32))

    backend.register_kernel("fill", fill)
    event = backend.submit(
        Command(
            id=1,
            op="launch",
            dpu_id=0,
            payload={"kernel": "fill", "offset": 0, "value": [1, 2, 3, 4]},
            reads=[],
            writes=[Access(("dpu", 0), 0, 16)],
            waits=[],
        )
    )
    assert not backend.query(event)
    gate.set()
    backend.wait(event)
    assert backend.query(event)
```

```python
def test_read_local_returns_a_copy_and_push_xfer_fans_out() -> None:
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=64))
    payload = np.arange(4, dtype=np.float32)
    backend.push_xfer([(0, 0, payload), (1, 16, payload)])
    host_copy = backend.read_local(0, 0, payload.shape, payload.dtype)
    host_copy[0] = -1
    assert backend.read_local(0, 0, payload.shape, payload.dtype)[0] != -1
    assert np.array_equal(backend.read_local(1, 16, payload.shape, payload.dtype), payload)
```

```python
def test_gpt2_smoke_payload_stays_model_agnostic() -> None:
    torch.manual_seed(0)
    model = GPT2LMHeadModel(
        GPT2Config(n_layer=4, n_head=8, n_embd=512, n_positions=128, n_ctx=128)
    ).eval()
    input_ids = torch.arange(128, dtype=torch.long).unsqueeze(0)
    logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.detach().cpu().numpy()
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=logits.nbytes * 2))
    backend.copy_to_dpu(0, 0, logits)
    assert np.array_equal(backend.copy_from_dpu(0, 0, logits.shape, logits.dtype), logits)
```

- [ ] **Step 2: Run the new backend tests and confirm they fail**

Run:

```bash
python -m pytest tests/test_hal_numpy.py -q
```

Expected: missing `NumpyBackend` / methods / async event behavior.

- [ ] **Step 3: Implement the minimal fake HAL**

```python
@dataclass
class NumpyBackendConfig:
    num_dpus: int
    mram_bytes_per_dpu: int
    allow_device_to_device: bool = False


@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]
    future: Future[object]
```

```python
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
            raise ValueError(f"invalid MRAM range offset={offset} nbytes={nbytes} capacity={self.capacity}")


class DPUDevice:
    def __init__(self, dpu_id: int, capacity: int) -> None:
        self.dpu_id = dpu_id
        self.mram = np.zeros(capacity, dtype=np.uint8)
        self.alloc = MRAMAllocator(capacity)
```

```python
class NumpyBackend:
    def __init__(self, config: NumpyBackendConfig) -> None:
        self.config = config
        self._kernels = KernelStubRegistry()
        self._pool = ThreadPoolExecutor(max_workers=config.num_dpus + 1)
        self._devices = {dpu_id: DPUDevice(dpu_id, config.mram_bytes_per_dpu) for dpu_id in range(config.num_dpus)}

    def register_kernel(self, name: str, fn: Callable[["NumpyBackend", int, Command], None]) -> None:
        self._kernels.register(name, fn)

    def register_plan(self, plan: object) -> None:
        self._plan = plan

    def copy_to_dpu(self, dpu_id: int, offset: int, data: np.ndarray) -> None:
        blob = np.ascontiguousarray(data).view(np.uint8)
        dev = self._devices[dpu_id]
        dev.alloc.check(offset, blob.nbytes)
        dev.mram[offset:offset + blob.nbytes] = blob.reshape(-1)

    def copy_from_dpu(self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        dev = self._devices[dpu_id]
        dev.alloc.check(offset, size)
        blob = dev.mram[offset:offset + size].copy()
        return blob.view(dtype).reshape(shape).copy()
```

```python
    def write_local(self, dpu_id: int, offset: int, data: np.ndarray) -> None:
        self.copy_to_dpu(dpu_id, offset, data)

    def read_local(self, dpu_id: int, offset: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        return self.copy_from_dpu(dpu_id, offset, shape, dtype)

    def push_xfer(self, transfers: list[tuple[int, int, np.ndarray]]) -> None:
        for dpu_id, offset, data in transfers:
            self.copy_to_dpu(dpu_id, offset, data)

    def submit(self, cmd: Command) -> Event:
        if cmd.op == "launch":
            kernel = self._kernels.lookup(str(cmd.payload["kernel"]))
            return Event("dpu", self._pool.submit(kernel, self, int(cmd.dpu_id), cmd))
        return Event("host" if cmd.dpu_id is None else "dma", self._pool.submit(self._run_host_or_dma, cmd))

    def wait(self, event: Event, timeout: float | None = None) -> None:
        event.future.result(timeout=timeout)

    def query(self, event: Event) -> bool:
        return event.future.done()
```

Implementation rules:

- each DPU owns a private `np.uint8` MRAM buffer;
- local reads and writes always copy bytes, never hand out views;
- `write_local` and `read_local` validate bounds against one DPU only;
- `submit` runs `launch`, `dma_in`, `dma_out`, and host ops through the stream context;
- `wait` and `query` work against `Event` only, not raw DPU ids;
- `allow_device_to_device=False` keeps DPU-to-DPU copy disabled by default.

- [ ] **Step 4: Re-run the focused backend tests**

Run:

```bash
python -m pytest tests/test_hal_numpy.py -q
```

Expected: DPU isolation, copy round-trip, async submit/query, and GPT-2 smoke all pass.

### Task 3: Stub the Vendor Backend and Document the Module

**Files:**
- Modify: `backend/hal_vendor.py`
- Modify: `backend/__init__.py`
- Create: `docs/hal_numpy.md`

**Interfaces:**
- Consumes: the Task 2 backend API.
- Produces: an API-compatible stub and short user-facing documentation.

- [ ] **Step 1: Add the vendor stub**

```python
class VendorBackend:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Vendor backend is not wired yet")
```

Keep the module-level public names aligned with `hal_numpy.py` so runtime code can switch backends without changing call sites.

- [ ] **Step 2: Write the module note**

`docs/hal_numpy.md` should stay short and cover only:

- public classes and methods;
- isolated DPU MRAM semantics;
- synchronous copy helpers vs async submit/query;
- problem 3 / 6 / 7 / 8 coverage;
- the GPT-2-shaped smoke use case.

- [ ] **Step 3: Run verification**

Run:

```bash
python -m pytest tests/test_hal_numpy.py -q
python -m pytest tests/ -x -q
git diff --check
git diff --stat
```

Expected: all tests pass, no whitespace errors, and the diff stays limited to the runtime-contract and backend surface.

- [ ] **Step 4: Leave changes uncommitted**

Report the final test output and net line count. Do not create a Git commit unless explicitly requested.
