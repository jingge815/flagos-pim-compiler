# Problem 2: Device Mapping and Sharding Propagation Design

## Scope

This design implements the phase-1 device-mapping and sharding-propagation pass
between problem 1 graph partitioning and problems 3, 6, 7, and 8. It works on
the original exported FX graph and produces metadata only. It does not insert
FX nodes, execute operators, generate DMA commands, allocate MRAM, or solve an
automatic placement optimization problem.

The validation boundary is:

- A minimal Appendix A two-Linear graph is numerically checked on
  `NumpyBackend` using the generated shard maps.
- The real local Llama-2-7B model is checked through
  `export -> partition_graph -> propagate_specs`; this is a structural
  compilation-time validation, not NumpyBackend prefill or decode execution.
- Full Llama-2-7B inference is deferred until problems 3, 6, 7, and 8 provide
  redistribute DMA lowering, execution planning, KV handling, and memory
  offsets.

## Architecture

`partition_graph(gm)` remains the required entry point and marks every FX node
with `node.meta["device"]` and DPU nodes with `node.meta["part_id"]`.
`propagate_specs(gm, config)` runs over that same complete FX graph in
topological order:

1. Remove stale problem-2 annotations so repeated runs are deterministic.
2. Assign source layouts to placeholders and tensor `get_attr` nodes.
3. For each node, derive the input layout requirements and output layout.
4. Compare each producer's actual layout with its consumer requirement and
   attach a `RedistributeEdge` when they differ.

The pass writes the produced tensor layout to `node.meta["spec"]` and the
incoming logical redistribution edges to
`node.meta["redistribute"]`. It returns the graph-wide edge list in stable
topological order. Host operators are part of the graph: they require
`Replicate@host` inputs and produce `Replicate@host` outputs.

Physical mapping is per tensor, not per partition. A tensor-parallel DPU
operator runs on all configured DPUs; `part_id` identifies a partitioned
operator group rather than exclusive ownership of a DPU. `ShardConfig` carries
the fixed participating physical DPU IDs, defaulting to consecutive IDs only
when that is explicitly selected by the caller. All `shard_map` keys and edge
locations use those physical IDs.

## Shared Contract

`PIMTensorSpec` remains the only tensor-layout contract consumed by downstream
problems. It contains:

- `device`: `"host"` or `"dpu"`.
- `placement`: the DTensor-equivalent `Shard(dim)`, `Replicate`, or
  `Partial(reduce_type)` algebra.
- `residency`: `"transient"` or `"pinned"`.
- `pinned_dpu_id`: used by future KV layouts; distributed weights leave it
  `None`.
- `shard_map`: one `TensorShardDetail` per physical DPU, including global range
  and local shape. Problem 8 fills `mram_offset` later.

`RedistributeEdge` describes one logical mismatch only: stable edge ID, source
and destination node names, source and required placement, redistribution
type, locations, logical byte count, and optional reduction type. It does not
contain DMA segments, offsets, or command dependencies; those are owned by
problems 3, 8, and 6 respectively.

Shard maps enforce the phase-1 contract: a single uniformly sharded dimension,
one contiguous range per DPU, and all configured DPUs participating.
`Replicate` and `Partial` record the full local shape on every participating
DPU. Invalid dimensions, uneven splits, duplicate physical DPU IDs, and empty
device sets fail immediately.

## Initial Layouts and Rules

The initial configuration is manual and fixed. For Llama models, Q/K/V, gate,
up, and LM-head weights are column parallel (`Shard(0)` in HuggingFace
`[out, in]` storage); O and down weights are row parallel (`Shard(1)`).
RMSNorm weights are replicated on DPU. Embeddings and host-only constants stay
on host. All weights are pinned and cannot appear in a redistribution edge.

The rule table covers exactly the phase-1 DPU whitelist from `graph.partition`:

- Column-parallel `aten.linear`: requires replicated input and produces a
  shard along the last output dimension.
- Row-parallel `aten.linear`: requires an input shard along its last dimension
  and produces `Partial(sum)`.
- `aten.add` and `aten.mul`: preserve matching shard layouts, locally slice a
  replicated operand to a shard when legal, reduce `Partial + Replicate` to
  replicated inputs, and preserve matching partials.
- Unary DPU elementwise operators preserve non-partial placement.

Supported conversions are `Partial -> Replicate` (`all_reduce`),
`Shard -> Replicate` (`all_gather`), and `Replicate -> Shard` (`scatter` across
locations or `local_slice` on an existing DPU). An identical placement crossing
host and DPU becomes the corresponding host-star transfer. Unsupported
conversions, such as incompatible shard dimensions or partial nonlinear
operations, fail with the affected node and layouts rather than silently
changing behavior.

Tensor inputs in both positional arguments and keyword arguments participate
in propagation. Non-tensor arguments are not data-flow edges.

## Error Handling and Testing

The propagation pass requires problem-1 annotations. It fails if a DPU node
has no tensor value metadata, a DPU whitelist operator has no rule, a linear
weight lacks a configured initial sharding mode, or a pinned tensor would need
redistribution.

Tests cover:

- Appendix A placements, per-DPU shard ranges, and expected redistribution
  types.
- Appendix A numerical reconstruction on `NumpyBackend`: concatenate column
  shards and sum row-parallel partials, then compare with PyTorch.
- Mapping and contract failures, including invalid DPU IDs and unequal splits.
- Positional and keyword tensor inputs, repeated propagation cleanup, stable
  edge IDs, and zero-communication `local_slice` behavior.
- A small random Llama graph for end-to-end problem-1 to problem-2 integration.
- The local Llama-2-7B graph for all-layer Megatron pairing, pinned weights,
  expected all-reduces, output all-gather, complete tensor annotations, and a
  printable report.

## Deferred Work

Automatic mapping/cost optimization, broader operator coverage, variable
shapes, fusion, asynchronous dispatch, and actual DTensor runtime use are out
of scope. KV tensor construction and pinned head mapping are introduced with
problem 7. Problem 3 lowers `RedistributeEdge` to host-mediated DMA segments;
problem 8 fills MRAM offsets; problem 6 consumes those plans to execute model
inference on `NumpyBackend`.
