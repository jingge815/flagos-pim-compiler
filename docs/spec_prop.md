# Sharding Propagation (Problem 2)

## Interfaces

```python
ShardConfig(num_dpus, weight_rules, dpu_ids=None)
llama_shard_config(..., dpu_ids=None) -> ShardConfig
propagate_specs(gm, config) -> list[RedistributeEdge]
format_spec_report(gm, edges, *, max_nodes=None) -> str
```

`dpu_ids` are fixed physical DPU IDs, normalized to a tuple; shard-map keys and
`TensorShardDetail.dpu_id` use those IDs directly. `weight_rules` match
`get_attr` name substrings and select `"col"` or `"row"` sharding.

`propagate_specs` requires `partition_graph` annotations. It writes
`node.meta["spec"]` for tensor nodes and `node.meta["redistribute"]` for each
consumer, then returns the ordered `RedistributeEdge` list. It is metadata-only:
it neither executes operators nor transfers data.

## Design

- The rule table derives each node's required input layouts and actual output
  layout. It inspects tensor operands in both `args` and `kwargs`.
- `src_spec` is the producer's actual output. `dst_spec` materializes the
  consumer's required input, so lower layers have concrete specs at both ends.
- Placeholders and host nodes are `Replicate@host`; host/DPU boundaries become
  redistribution edges. Pinned weights are never redistributed.
- In HF `[out, in]` weights, `"col"` is `Shard(0)` and `"row"` is `Shard(1)`.
  The column-to-row pair produces `Partial(sum)` as in Appendix A.
- `Partial -> Replicate` is `all_reduce`; `Shard -> Replicate` is `all_gather`;
  cross-location `Replicate -> Shard` is `scatter`. `local_slice` is a local
  view selection, not cross-DPU DMA. Unsupported layout changes raise.
- `Replicate@dpu -> Replicate@host` remains `all_gather`, but Problem 3 must
  lower it as a mandatory degenerate transfer: DMA exactly one full source copy
  from `min(edge.src_spec.shard_map)` to the host `dst_spec`, never concatenate
  DPU replicas. The edge retains the complete producer `src_spec` and host
  `dst_spec` so this condition is explicit to the lowerer.
- `shard_map` contains equal, continuous physical-DPU ranges. Replicate and
  Partial retain the full flattened range on every participating DPU.

## Validation And Scope

Implements `docs/spec.md:275-603` (problem 2). Appendix A (`:2358`) is the
NumpyBackend numeric proof of the placement and shard ranges. Tiny Llama is a
structural test; Llama-2-7B validates export, partition, and propagation
structure only, not complete 7B inference. KV layout and MRAM offsets belong to
problems 7 and 8; cost-driven repartitioning is deferred.
