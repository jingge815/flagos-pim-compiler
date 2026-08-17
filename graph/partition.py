"""Problem-1 DPU capability annotation and direct-edge graph partitioning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, PART_ID_META_KEY


DPU_LOWERABLE = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.tanh.default,
    }
)


@dataclass
class Partition:
    """A DPU-only connected component in FX topological order."""

    part_id: int
    nodes: list[Node]


def _is_dpu_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in DPU_LOWERABLE


def partition_graph(gm: GraphModule) -> list[Partition]:
    """Annotate ``gm`` in place and return its DPU direct-edge components."""

    nodes = list(gm.graph.nodes)
    node_order = {node: index for index, node in enumerate(nodes)}
    parent: dict[Node, Node] = {}

    for node in nodes:
        node.meta[DEVICE_META_KEY] = DEVICE_DPU if _is_dpu_node(node) else DEVICE_HOST
        node.meta.pop(PART_ID_META_KEY, None)
        if node.meta[DEVICE_META_KEY] == DEVICE_DPU:
            parent[node] = node

    def find(node: Node) -> Node:
        while parent[node] is not node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: Node, right: Node) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root is not right_root:
            parent[right_root] = left_root

    for node in parent:
        # Host nodes are not in parent, so they can never bridge two DPU groups.
        for input_node in node.all_input_nodes:
            if input_node in parent:
                union(node, input_node)

    components: dict[Node, list[Node]] = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)

    sorted_components = sorted(
        (sorted(component, key=node_order.__getitem__) for component in components.values()),
        key=lambda component: node_order[component[0]],
    )

    partitions: list[Partition] = []
    for part_id, component in enumerate(sorted_components):
        for node in component:
            node.meta[PART_ID_META_KEY] = part_id
        partitions.append(Partition(part_id=part_id, nodes=component))
    return partitions
