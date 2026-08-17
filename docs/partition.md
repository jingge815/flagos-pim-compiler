# Graph Partition

## Interface

```python
partition_graph(gm: GraphModule) -> list[Partition]
```

The pass preserves `gm` and annotates every FX node with
`node.meta["device"]` (`"dpu"` or `"host"`). DPU nodes also receive
`node.meta["part_id"]`; host nodes do not. Returned `Partition` entries and
their node lists follow FX topological order.

The `device` and `part_id` keys, along with the `dpu` and `host` values, are
defined in `contracts/graph_meta.py` for use by later graph and runtime passes.

## Design

Phase 1 lowers only `aten.addmm.default`, `aten.linear.default`,
`aten.add.Tensor`, `aten.mul.Tensor`, and `aten.tanh.default`. The pass unions
only direct FX data edges whose two endpoints are DPU nodes, making host nodes
hard boundaries. This deliberately avoids `CapabilityBasedPartitioner`: its
horizontal fusion can merge DPU consumers across a host producer.

## Specification

Implements problem 1 in `docs/spec.md:159-273`: capability annotation and DPU
connected grouping only. Capacity, placement, communication, and QKV rewriting
belong to later passes.
