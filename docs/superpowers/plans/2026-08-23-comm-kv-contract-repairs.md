# Communication and KV Contract Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the discovered problem 2→3 and problem 2→7 contract gaps without adding problem 6 or 8 functionality.

**Architecture:** Preserve compile-time `RedistributeEdge` as the sole source for communication planning and `KVRegionSpec` as the sole source for KV layout. Tighten their input validation at the owning boundary: layout transitions in `graph/spec_prop.py`, host DMA buffers in `backend/dpu_sdk.py`, and KV model/position inputs in `memory/kv_layout.py`.

**Tech Stack:** Python 3.10, NumPy, PyTorch FX, pytest, NumPyBackend.

## Global Constraints

- Implement only the confirmed regressions; do not add ExecutionPlan, memory-planner, asynchronous DMA, or dynamic-sequence features.
- Follow the vendor signatures in `/media/disk/fengjingge/src/pre-g-driver-api/api/include/dpu.h`.
- Write a failing pytest regression before every production behavior change.
- Keep static communication addresses and KV offsets unchanged for valid existing inputs.

---

### Task 1: Permit and Lower DPU Shard-to-Shard Redistributions

**Files:**
- Modify: `tests/test_spec_prop.py`
- Modify: `graph/spec_prop.py`
- Test: `tests/test_spec_prop.py`

**Interfaces:**
- Consumes: `_edge_type(node, actual, req) -> str`
- Produces: DPU `Shard(i) -> Shard(j)` transitions return `"all_to_all"`.

- [x] **Step 1: Write the failing test**

```python
def test_shard_dimension_change_between_dpus_is_all_to_all():
    assert _edge_type(node, shard_dim_zero, _Req("dpu", Placement("Shard", 1))) == "all_to_all"
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_spec_prop.py::test_shard_dimension_change_between_dpus_is_all_to_all -q`
Expected: FAIL because `_edge_type` rejects distinct Shard dimensions.

- [x] **Step 3: Write minimal implementation**

```python
if a.kind == r.kind == "Shard" and a.dim != r.dim:
    return "all_to_all"
```

Place it before the generic unsupported-layout guard, which otherwise rejects this valid transition.

- [x] **Step 4: Run affected tests**

Run: `python -m pytest tests/test_spec_prop.py tests/test_comm_plan.py tests/test_comm_lowering.py -q`
Expected: PASS.

### Task 2: Reject Non-Contiguous SDK Transfer Buffers

**Files:**
- Modify: `tests/test_dpu_sdk.py`
- Modify: `backend/dpu_sdk.py`
- Test: `tests/test_dpu_sdk.py`

**Interfaces:**
- Consumes: `dpu_copy_from(dpu_set, offset, dst, length)` and `dpu_prepare_xfer(dpu_set, buffer)`.
- Produces: host output buffers must be writable and C-contiguous; invalid buffers raise `DpuError` rather than writing a temporary.

- [x] **Step 1: Write the failing test**

```python
def test_copy_from_and_prepare_xfer_reject_non_contiguous_host_buffers():
    destination = np.zeros((2, 2), dtype=np.int32)[:, 0]
    with pytest.raises(DpuError, match="C-contiguous"):
        dpu_copy_from(dpu_set.dpu(0), 0, destination, 8)
```

Also assert `dpu_prepare_xfer` rejects the same view.

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dpu_sdk.py::test_copy_from_and_prepare_xfer_reject_non_contiguous_host_buffers -q`
Expected: FAIL because the current mirror silently materializes a temporary.

- [x] **Step 3: Write minimal implementation**

```python
def _writable_c_buffer(buffer: np.ndarray) -> np.ndarray:
    array = np.asarray(buffer)
    if not array.flags.c_contiguous or not array.flags.writeable:
        raise DpuError("host buffer must be writable and C-contiguous")
    return array.view(np.uint8).reshape(-1)
```

Use it in DPU-to-host transfer and `dpu_prepare_xfer`; preserve copying semantics for host-to-DPU input buffers.

- [x] **Step 4: Run affected tests**

Run: `python -m pytest tests/test_dpu_sdk.py tests/test_comm_lowering.py -q`
Expected: PASS.

### Task 3: Validate KV Model and Runtime Inputs

**Files:**
- Modify: `tests/test_kv_layout.py`
- Modify: `memory/kv_layout.py`
- Test: `tests/test_kv_layout.py`

**Interfaces:**
- Consumes: `kv_specs_from_placement(...)`, `PIMStaticKVCache.update(...)`, `prefill_mask(...)`, `decode_mask(...)`.
- Produces: complete, non-overlapping KV-head assignment; exact configured NumPy dtype; checked sequence bounds.

- [x] **Step 1: Write failing tests**

```python
with pytest.raises(ValueError, match="覆盖"):
    kv_specs_from_placement(incomplete_k_proj, num_kv_heads=4, ...)
with pytest.raises(ValueError, match="dtype"):
    cache.update(0, 0, uint16_kv, uint16_kv)
with pytest.raises(ValueError, match="prompt_len"):
    prefill_mask(5, 4)
with pytest.raises(ValueError, match="valid_len"):
    decode_mask(4, 4)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kv_layout.py -k 'rejects_wrong_dtype or validates_kv_head_coverage or masks_reject_invalid_lengths' -q`
Expected: FAIL because the existing code accepts those invalid values.

- [x] **Step 3: Write minimal implementation**

```python
expected_dtype = _NP_DTYPE[spec.dtype_bytes]
if vec.dtype != expected_dtype:
    raise ValueError(...)
```

Derive assigned head IDs after building specs and reject values other than `set(range(num_kv_heads))`; reject `prompt_len` outside `[0, max_seq]` and `valid_len` outside `[0, max_seq)`.

- [x] **Step 4: Run affected tests**

Run: `python -m pytest tests/test_kv_layout.py tests/test_kv_layout_llama2_7b.py -q`
Expected: PASS.

### Task 4: Verify Integrated Contract Behavior

**Files:**
- Modify: `docs/comm.md`
- Modify: `docs/kv_layout-20260822.md`
- Test: `tests/`

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: concise documentation of the stricter SDK and KV validation guarantees.

- [x] **Step 1: Update documentation**

Document the all-to-all propagation path, strict writable C-contiguous receive buffers, exact KV dtype, complete head coverage, and position bounds.

- [x] **Step 2: Run scoped regression suite**

Run: `python -m pytest tests/test_spec_prop.py tests/test_dpu_sdk.py tests/test_comm_plan.py tests/test_comm_lowering.py tests/test_comm_llama2_7b.py tests/test_kv_layout.py tests/test_kv_layout_llama2_7b.py -q`
Expected: PASS.

- [x] **Step 3: Run full suite**

Run: `python -m pytest tests/ -x -q`
Expected: the pre-existing `tests/test_genesim_bridge.py::test_refined_ir_preserves_structure[llama2_7b_flagtree.ir]` fails because the external IR fixture lacks `max_seq`; no preceding test fails.
