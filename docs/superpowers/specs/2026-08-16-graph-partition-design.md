# Graph Partition Design

**Goal:** Implement problem 1 of the PIM inference compiler: annotate a
`torch.export` FX graph with DPU/host capability and group only directly
connected DPU nodes into deterministic DPU partitions.

## Scope

This is a compile-time pass in `graph/partition.py`. It preserves the original
`GraphModule`; it neither creates submodules nor changes edges. It does not
perform capacity checks, DPU-id assignment, tensor placement propagation,
communication planning, QKV rewriting, or any runtime work.

The initial target is the offline random GPT-2 shape used by
`3-install-model-inference.sh`: four layers, eight heads, embedding width 512,
and maximum sequence length 128.

## Public Interface

`graph/partition.py` will expose:

```python
@dataclass
class Partition:
    part_id: int
    nodes: list[torch.fx.Node]


def partition_graph(gm: torch.fx.GraphModule) -> list[Partition]:
    ...
```

`DPU_LOWERABLE` is a module-level immutable set containing exactly these ATen
targets for phase 1:

```python
torch.ops.aten.addmm.default
torch.ops.aten.linear.default
torch.ops.aten.add.Tensor
torch.ops.aten.mul.Tensor
torch.ops.aten.tanh.default
```

All other nodes, including placeholders, `get_attr`, `output`, shape/view
operations, attention, LayerNorm, softmax, and ReLU, are host nodes.

## Algorithm

1. Walk all FX nodes and overwrite `node.meta["device"]` with `"dpu"` when the
   node is a `call_function` whose target belongs to `DPU_LOWERABLE`; otherwise
   set it to `"host"`.
2. For every DPU node, inspect only `node.all_input_nodes`. Union a node with an
   input node when both endpoints are DPU nodes. No transitive reachability
   through host nodes is considered.
3. Collect the union-find components. Sort every component by original FX graph
   order; sort components by their first node's order; assign consecutive
   `part_id` values beginning at zero.
4. Write `node.meta["part_id"]` only for DPU nodes. Remove a stale `part_id`
   from each host node so a repeated pass produces the same result while leaving
   unrelated metadata, notably `node.meta["val"]`, untouched.
5. Return the sorted `Partition` directory. Its nodes are the same FX nodes in
   topological order.

This makes a host node a hard partition boundary. For example,
`DPU -> host -> DPU` produces two partitions, while a directly connected DPU
fork and join remains one partition.

## Partitioning Choice

The implementation intentionally does not use
`torch.fx.passes.infra.partitioner.CapabilityBasedPartitioner` as the final
grouping algorithm. Although it provides cycle-safe partition construction, it
also applies horizontal fusion: DPU consumers of the same host producer can be
merged into one partition. That violates the required rule that encountering a
host node terminates a DPU group. FX graphs from `torch.export` are DAGs, so
connected components of the induced DPU-only graph give the required semantics
without a separate cycle-handling mechanism.

## GPT-2 Export Fixture

Tests will create a CPU-only `GPT2LMHeadModel` with the random installer
configuration and wrap it in a logits-only module. The wrapper accepts a fixed
4D causal mask and calls the model with `use_cache=False`.

The fixed mask is required because the installed Transformers 4.57.6 default
causal-mask path uses `vmap` behavior that fails `torch.export(...,
strict=True)`. Passing a precomputed fixed-shape 4D causal mask bypasses that
path and produces a strict ATen graph. This is an export-fixture constraint,
not a graph-partitioning fallback.

## Tests and Acceptance

`tests/test_partition.py` will include hand-built FX graph tests for:

- whitelist versus host annotation;
- a host node splitting adjacent DPU nodes into separate partitions;
- directly connected DPU forks and joins forming one partition;
- stable topological partition numbering; and
- repeated execution removing stale host `part_id` while preserving unrelated
  metadata.

An integration test will strictly export the fixed-shape random GPT-2 fixture
and assert that:

- export succeeds without a graph break;
- every FX node has a `device` annotation;
- DPU nodes have consecutive partition ids and host nodes have none;
- the returned directory covers every DPU node exactly once; and
- both host and DPU nodes occur in the exported graph.

The available CPU-only pytest verification command is:

```bash
/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree/python/bin/python \
  -m pytest tests/test_partition.py -q
```

The configured PyTorch 2.9.1 environment does not currently include pytest;
the same test functions are also exercised directly there during verification.

## Documentation

The implementation will add `docs/partition.md`, limited to the public API,
the narrow whitelist and direct-edge grouping decision, and the relevant
problem-1 specification reference.
