# NumpyBackend

`NumpyBackend` is the runtime validation backend for the fixed-shape Phase 1
path. `NumpyBackendConfig(num_dpus, mram_bytes_per_dpu,
allow_device_to_device=False)` configures it. The public API is:

- `copy_to_dpu(dpu_id, offset, data)` and `copy_from_dpu(dpu_id, offset, shape, dtype)`
- `write_local(dpu_id, offset, data)` and `read_local(dpu_id, offset, shape, dtype)`
- `push_xfer(transfers)` for a list of `(dpu_id, offset, data)` transfers
- `register_kernel(name, fn)`, `register_plan(plan)`, `submit(cmd)`, `wait(event, timeout=None)`, and `query(event)`
- `Event(kind, future)` for asynchronous command completion

Each DPU owns an isolated NumPy MRAM buffer. An operation addressing one DPU
cannot read or modify another DPU's buffer; offsets and byte ranges are checked
against that DPU's configured capacity.

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
