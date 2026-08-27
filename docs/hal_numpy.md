# NumpyBackend

`NumpyBackend` is the runtime validation backend for the fixed-shape Phase 1
path. `NumpyBackendConfig(num_dpus, mram_bytes_per_dpu,
wram_bytes_per_dpu=65536, allow_device_to_device=False)` configures it. The
public API is:

- `copy_to_dpu(dpu_id, offset, data)` and `copy_from_dpu(dpu_id, offset, shape, dtype)`
- `write_local(dpu_id, offset, data)` and `read_local(dpu_id, offset, shape, dtype)`
- `push_xfer(transfers)` for a list of `(dpu_id, offset, data)` transfers
- `register_kernel(name, fn)`, `register_plan(plan)`, `submit(cmd)`, `wait(event, timeout=None)`, and `query(event)`
- `Event(kind, future)` for asynchronous command completion
- `wram_ptr(dpu_id)` for a raw pointer into that DPU's WRAM buffer (mirrors `raw_mram_ptr`), used by multi-tasklet compiled kernels
- `record_access(tasklet_id, loc, offset, length, is_write)` and `barrier()`, used by a `launch` kernel to report tasklet-level WRAM/MRAM accesses for hazard detection

Each DPU owns an isolated NumPy MRAM buffer **and** an isolated WRAM buffer
(`_Dpu.wram`, a real byte array, not just a budget number). An operation
addressing one DPU cannot read or modify another DPU's buffer; offsets and
byte ranges are checked against that DPU's configured capacity. Since problem
3, the MRAM/WRAM storage and the DMA primitives are delegated to
`backend/dpu_sdk.py` (the NumPy mirror of the vendor SDK); this module only
adds the async `Command`/`Event` semantics on top, so the comm library and the
backend share one fake-hardware state. `dpu_sdk.py` also mirrors the vendor
SDK's rank grouping (`dpu_alloc_ranks`, `dpu_get_nr_ranks`, `DpuSet.by_rank`) —
a read-only label on top of the flat `dpu_id` space, not a second addressing
layer; the graph compiler's tensor-parallel split still targets flat `dpu_id`s
only.

## Tasklet-level concurrency (`Command.num_tasklets`)

A `launch` command's `num_tasklets` field (default 4) says how many tasklets
that DPU's kernel splits its work across. The model is **deterministic
in-order simulation, not real threads**: tasklets run their row range one
after another in a fixed order, so results stay bit-reproducible and
comparable against a single-threaded NumPy/torch reference. This is
deliberate — real `pthread`-based concurrency would make outputs
non-deterministic and defeat the elementwise comparison this repo relies on
for correctness (see `runtime/kernels.py`'s docstring on why BLAS threads are
clamped to 1 for the same reason).

What this buys instead of real concurrency is **hazard detection**: a kernel
reports every WRAM/MRAM access via `hal.record_access(tasklet_id, loc, offset,
length, is_write)`, and `HazardTracker` (in `backend/hal_numpy.py`) raises
`TaskletHazardError` the moment two different tasklets touch overlapping
addresses with at least one write, before a `hal.barrier()` call clears the
epoch. This catches exactly the class of bug that would be a real data race on
actual hardware — a missing barrier, or a row-range split that overlaps —
without needing real hardware or real threads to reproduce it. Each `launch`
command gets its own `HazardTracker` (kept in a `threading.local`, since
different DPUs' launches run concurrently on the backend's own thread pool),
so hazard state never leaks across commands or across DPUs.

`runtime/kernels.py::tasklet_linear_kernel` is the reference implementation:
it splits `linear`'s M dimension into `ceil(M/num_tasklets)`-row blocks (the
UPMEM convention of one tasklet per contiguous row range), records every
read/write through `hal.record_access`, and calls `hal.barrier()` once after
all blocks are done. `compiled_linear_kernel` runs the equivalent split inside
the FlagTree-compiled `.so` (see `pim-lower-to-emitc` in FlagTree's
`LowerPIMToEmitC.cpp`), unrolled at compile time rather than simulated at
runtime — the C code has no runtime hazard checking, since a compiled
artifact is assumed to already be correctly synchronized (matching real DPU
binary semantics); the hazard-catching value lives entirely on the NumPy side.

The copy helpers and `push_xfer` are synchronous. `submit` returns an `Event`
for a kernel launch, DMA command, or host operation; `wait` blocks for it and
`query` checks completion. This models the runtime contract without hardware.

The backend is the validation base for the runtime portions of problem 3
(redistribute lowered to DMA), problem 6 (host orchestration), problem 7
(local KV cache access), and problem 8 (MRAM layout and bounds). Problem 1/2
compile-time graph analysis and problem 4 cost analysis are outside this
module.

The smoke use case is a LLaMA2-shaped activation payload: copy a
`[batch, seq, hidden]` NumPy tensor matching the 4096-wide LLaMA2 hidden state
into a DPU MRAM buffer, then copy it back and compare elementwise. The same
runtime contract is intended to be used when switching to `VendorBackend`.

`VendorBackend` is exported from `backend` as a construction-time stub. Its
construction raises `NotImplementedError` until the vendor SDK is wired.
