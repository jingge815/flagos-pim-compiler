# Comm Library (Problem 3)

`comm/` lowers problem 2's `RedistributeEdge` annotations into host-mediated DMA
on the vendor SDK mirror. Spec: `docs/spec.md` problem 3 (line 605) and
Appendix B (line 2429).

## Interfaces

```python
# comm/plan.py — compile time
build_comm_plan(edges: list[RedistributeEdge]) -> list[CommPlanEntry]
dma_sequence(entry: CommPlanEntry) -> list[DmaOp]
plan_cost(entry, model: HostStarCostModel | None = None) -> CommCost
format_comm_plan(entries, *, max_segments=None) -> str

# comm/lowering.py — run time
DmaEngine(dpu_set: backend.dpu_sdk.DpuSet)
all_reduce(entry, engine) -> np.ndarray      # Partial  -> Replicate
all_gather(entry, engine) -> np.ndarray      # Shard    -> Replicate
all_to_all(entry, engine) -> np.ndarray      # Shard(i) -> Shard(j)
scatter(entry, engine, host_buf) -> None     # Replicate -> Shard
broadcast(entry, engine, host_buf) -> None   # one host buffer -> every dst DPU
```

`CommPlanEntry` is one redistribute edge expanded into per-segment rows
(`CommSegment`, fields per the spec table) plus the edge-level `wait_for`.
`dst_ready_after` stays empty until problem 8 fills it. `local_slice` edges keep
an empty placeholder entry (zero DMA).

## Design

- Segment ranges are computed in the **flattened row-major element coordinate**.
  A `Shard(d)` shard unfolds into `prod(shape[:d])` contiguous runs, each
  contiguous in both the global buffer and the local buffer. This fixes the
  spec's `nbytes` formula, which only holds for 1-D/innermost-dim sharding, and
  makes multi-dim cases (e.g. logits `Shard(2)` with S>1) byte-exact.
- `src_addr`/`dst_addr` bake `mram_offset + local_offset * itemsize` into each
  segment, so the table is self-contained for the orchestrator; problem 8 only
  changes `mram_offset` values, not the rules.
- A DPU-local `Shard(i) -> Shard(j)` edge is emitted as `all_to_all`; plan
  construction verifies that both endpoint locations exactly match their
  `PIMTensorSpec.shard_map`, so stale node metadata cannot target a wrong DPU.
- Collect segments (`dst_dpu=None`) vs writeback segments (`src_dpu=None`); no
  writeback rows when `dst_loc` is host. An all-gather from a Replicate source
  collects a single copy. `all_to_all` rows are pairwise run intersections and
  carry both endpoints.
- `dma_sequence` batches same-address/same-length per-DPU transfers into one
  `push_xfer`, and a uniform full-tensor writeback into one `broadcast_to`.
  `plan_cost` counts exactly those SDK calls (fixed setup + host-bandwidth
  linear term, `topology="host_star"`).
- The comm library never waits: `wait_for`/`dst_ready_after` become
  `Command.waits` in problem 6. Collect reads each segment into its own host
  buffer (partials overlap), then the primitive reduces/concats by
  `global_range` placement — order-independent of DPU ids (Appendix B.4).

## SDK mirror

`backend/dpu_sdk.py` mirrors pre-g-driver-api (`api/include/dpu.h`) on NumPy
fake hardware: `dpu_alloc/free/get_nr_dpus`, `dpu_copy_to/from`,
`dpu_prepare_xfer` + `dpu_push_xfer` (two-phase batch), `dpu_broadcast_to`,
`dpu_load/launch/sync/status/log_read`. `dpu_set_t` maps to `DpuSet`;
`DPU_FOREACH` maps to iterating single-DPU subsets; errors raise `DpuError`.
Deviations: programs are Python callables `fn(dpu_id, mram)` (binaries are
registered but not executable), the rank layer is not mirrored, and all calls
are synchronous. `NumpyBackend` (problem 6 HAL) stores its MRAM in the same
machine, so the comm library and the future orchestrator share one fake
hardware state. There is no DPU→DPU primitive anywhere (host-star topology).
All `dpu_prepare_xfer` buffers must be `numpy.ndarray` and C-contiguous;
DPU-to-host output buffers must additionally be writable. The mirror rejects
views or Python containers that NumPy would otherwise silently copy to a
temporary buffer. `dpu_push_xfer` also accepts only the two `dpu_xfer_t`
directions from the vendor API and rejects unknown values.

Verification: `tests/test_dpu_sdk.py`, `tests/test_comm_plan.py`,
`tests/test_comm_lowering.py`, `tests/test_comm_llama2_7b.py` (real Llama-2-7B:
Megatron-pair numerics against single-card torch, byte-exact gather/scatter).
