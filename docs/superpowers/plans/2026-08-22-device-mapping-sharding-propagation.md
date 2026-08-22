# Device Mapping and Sharding Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make problem 2 produce validated per-DPU tensor layouts and complete logical redistribution metadata for a partitioned Llama FX graph.

**Architecture:** Keep propagation as a metadata-only pass over the original FX graph. ShardConfig selects a fixed physical DPU set; PIMTensorSpec records each tensor's actual layout; and each RedistributeEdge materializes the destination input layout needed by later DMA and memory passes. Host nodes remain explicit Replicate@host glue nodes.

**Tech Stack:** Python 3.10, PyTorch 2.9 FX/torch.export, NumPy, pytest, NumpyBackend.

## Global Constraints

- Source PyTorch before every test command: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh.
- Work from the existing uncommitted problem-2 baseline; do not discard unrelated user changes.
- Do not create FX redistribute nodes, execute operators, generate DMA commands, or assign MRAM offsets.
- Keep the phase-1 fixed manual mapping; do not add automatic placement optimization, variable shapes, fusion, or async dispatch.
- Keep contracts/ as the only definition site for cross-module metadata types and graph_meta.py as the only definition site for metadata keys.
- Tests precede each behavior change. The final test command is python -m pytest tests/ -x -q.
- Do not create a git commit unless the user explicitly asks for one.

---

## File Structure

- Modify contracts/pim_tensor_spec.py: validate Placement, TensorShardDetail, and PIMTensorSpec; extend RedistributeEdge with endpoint specs.
- Modify graph/spec_prop.py: normalize fixed physical DPU IDs, build maps using them, materialize both edge endpoints, inspect args and kwargs, clear stale annotations, and report shard details.
- Modify tests/test_spec_prop.py: add focused contract, mapping, kwargs, cleanup, endpoint, and report tests.
- Modify tests/test_spec_prop_llama2_7b.py: assert real-model edges carry endpoint specs.
- Modify docs/spec_prop.md: document the final public interfaces and phase-1 boundary.
- Modify docs/子图拆分映射-20260822.md: retain the existing Chinese code-reading note and update it with the final mapping, endpoint-spec, and validation behavior.

## Task 1: Shared Contract and Physical DPU Mapping

**Files:** contracts/pim_tensor_spec.py; graph/spec_prop.py; tests/test_spec_prop.py

**Interfaces:**
- Consumes ShardConfig(num_dpus, weight_rules, dpu_ids=None).
- Produces normalized config.dpu_ids, validated PIMTensorSpec/TensorShardDetail values keyed by physical IDs.
- Preserves llama_shard_config and propagate_specs call compatibility.

- [ ] Step 1: Write failing tests.

Add tests proving that ShardConfig(num_dpus=2, dpu_ids=(2, 5), ...) creates shard_map keys (2, 5), that empty/short/duplicate/negative dpu_ids raise ValueError matching dpu_ids, and that Placement("Shard"), Placement("Shard", -1), Placement("Partial"), and Placement("Replicate", 0) raise from placement.validate().

- [ ] Step 2: Run the new tests and verify the expected failure.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_spec_prop.py -k 'physical_dpu or invalid_physical or invalid_dtensor' -q
Expected: failure because ShardConfig has no dpu_ids parameter and Placement.validate is absent.

- [ ] Step 3: Implement the minimal contract and mapping behavior.

In contracts/pim_tensor_spec.py, add Placement.validate(): Shard requires a non-negative dim and no reduce_type; Replicate requires no dim/reduce_type; Partial requires reduce_type sum or mean and no dim. Add TensorShardDetail.validate() for non-negative ID/ranges/offset and non-negative local dimensions. Add PIMTensorSpec.validate() to enforce placement consistency, key/detail ID agreement, empty host maps, non-empty DPU maps, and Shard/Replicate/Partial shard_dim conventions.

In graph/spec_prop.py, add optional dpu_ids to ShardConfig and normalize it in __post_init__: default to tuple(range(num_dpus)); require num_dpus > 0, exactly num_dpus IDs, unique non-negative IDs; store the tuple with object.__setattr__. Change _dpu_spec and _shard_map to accept dpu_ids and use each physical ID as the TensorShardDetail.dpu_id. Validate shard dimensions before indexing shape. Pass dpu_ids through llama_shard_config and call spec.validate() after constructing host/DPU specs.

- [ ] Step 4: Run the focused and existing problem-2 tests.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_spec_prop.py -q
Expected: all current Appendix A, tiny-Llama, and new mapping/validation tests pass.

## Task 2: Redistribution Endpoint Specs, kwargs, and Idempotence

**Files:** contracts/pim_tensor_spec.py; graph/spec_prop.py; tests/test_spec_prop.py; tests/test_spec_prop_llama2_7b.py

**Interfaces:**
- Consumes validated PIMTensorSpec, ShardConfig.dpu_ids, problem-1 device metadata, and the DPU whitelist.
- Produces RedistributeEdge.src_spec (producer output) and dst_spec (materialized layout required at the consumer input).
- Preserves propagate_specs(gm, config) -> list[RedistributeEdge] and consumer node.meta["redistribute"].

- [ ] Step 1: Write failing tests.

Add a hand-built graph where aten.add.Tensor receives left positionally and right through kwargs["other"]. After partition_graph and propagation with dpu_ids=(2, 5), assert both source nodes produce redistribution edges, each edge.src_spec equals the source node spec, each edge.dst_spec is Replicate@dpu, and dst_spec.shard_map keys are (2, 5). Add a rerun test that seeds stale node.meta["redistribute"], runs propagation twice, and asserts equal edge lists, contiguous stable edge IDs, and empty stale metadata. Extend the report assertion to require DPU2:[...] and DPU5:[...] ranges. Extend the real-Llama edge test to assert edge.src_spec equals the source spec and edge.dst_spec.placement equals edge.to_placement.

- [ ] Step 2: Run the new tests and verify the expected failure.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_spec_prop.py -k 'keyword or stale or endpoint' -q
Expected: failure because endpoint fields are absent, only node.args are inspected, stale metadata is retained, and reports omit shard ranges.

- [ ] Step 3: Implement endpoint materialization and operand normalization.

Add src_spec: PIMTensorSpec and dst_spec: PIMTensorSpec to RedistributeEdge. Add _required_spec(src, req, config): host requirements return _host_spec(); DPU requirements call _dpu_spec(req.placement, tuple(src.meta["val"].shape), config.dpu_ids). Change _diff_edge to accept config, construct dst_spec from the source shape, derive dst_loc from dst_spec, and never synthesize locations with range(num_dpus).

At the start of propagate_specs, remove stale SPEC_META_KEY and reset REDISTRIBUTE_META_KEY on every node. Require the device key on every graph node. Keep output nodes without tensor specs but retain an empty redistribution list where appropriate.

Replace positional-only operand collection with a helper that checks node.args[position] first and node.kwargs[keyword] second. Use it for linear (input/weight/bias), add/mul (self/other), and unary (self). Rules must return explicit source-node/requirement pairs so kwargs edges are passed to _append_edge. Reject missing required operands, tensor-valued unsupported bias/alpha, nested tensor containers, and unsupported layouts with node name and layouts in the error. Preserve legal column/row Linear, elementwise, host glue, and local_slice behavior.

Extend format_spec_report with a compact formatter that prints each DPU detail as DPU<ID>:[start,end) shape=<local_shape> for DPU specs. Keep placement and edge summary lines unchanged.

- [ ] Step 4: Run focused graph tests and real Llama structural tests.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_spec_prop.py -q && python -m pytest tests/test_spec_prop_llama2_7b.py -q
Expected: both files pass; the real model still has 64 all-reduces, pinned weights, an output all-gather, and complete endpoint specs.

## Task 3: Documentation and Full Verification

**Files:** docs/spec_prop.md; docs/子图拆分映射-20260822.md; tests/test_partition.py; tests/test_spec_prop.py; tests/test_spec_prop_llama2_7b.py; tests/test_hal_numpy.py

**Interfaces:** final ShardConfig, PIMTensorSpec, RedistributeEdge, and format_spec_report interfaces from Tasks 1 and 2.

- [ ] Step 1: Update the short module document.

Keep docs/spec_prop.md within roughly 60 lines and document these exact interfaces: ShardConfig(num_dpus, weight_rules, dpu_ids=None), llama_shard_config(..., dpu_ids=None), propagate_specs(gm, config) -> list[RedistributeEdge], and format_spec_report(...). State that dpu_ids are fixed physical IDs, src_spec is the producer output, dst_spec is the materialized required input, args and kwargs are inspected, host nodes are Replicate@host, local_slice is not cross-DPU DMA, Appendix A is the NumpyBackend numeric proof, and Llama-2-7B is export/partition/propagate structural validation only. Retain docs/子图拆分映射-20260822.md and add a concise update section that explains the final physical mapping, endpoint-spec, argument handling, and validation changes.

Also apply the two approved review cleanups: change the `_diff_edge` docstring parameter wording from `num_dpus` to `config`, and rename the physical-DPU shard-range report test to include `endpoint` so the documented `-k 'keyword or stale or endpoint'` red command selects every behavior it claims to cover.

- [ ] Step 2: Run focused regression tests.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_partition.py tests/test_spec_prop.py tests/test_spec_prop_llama2_7b.py tests/test_hal_numpy.py -q
Expected: all focused graph, backend, and real-model structural tests pass.

- [ ] Step 3: Run repository verification and inspect the patch.

Run: source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/ -x -q; git diff --check; git diff --stat; git status --short
Expected: full suite passes, diff check is clean, and status contains only intended problem-2 contract/graph/test/documentation changes. Report net added/removed lines; do not commit without an explicit request.

## Plan Self-Review

Spec coverage: fixed physical mapping and contract validation are Task 1; propagation, host glue, edge types, endpoint specs, args/kwargs, cleanup, and reports are Task 2; NumpyBackend Appendix A, tiny Llama, real Llama-2-7B structural validation, documentation, and full verification are Task 3. DMA lowering, MRAM offsets, KV runtime, automatic optimization, and complete 7B inference remain explicitly deferred.

Placeholder scan: no TODO, TBD, or unspecified test action appears in this plan.

Type consistency: ShardConfig.dpu_ids is optional only at construction and normalized to tuple; RedistributeEdge endpoint specs are PIMTensorSpec; propagate_specs keeps its existing return type; format_spec_report remains a string diagnostic interface.
