"""验证 FX 图的 DPU 标记和连通子图划分。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule, Node
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, PART_ID_META_KEY
from graph.partition import DPU_LOWERABLE, partition_graph


def _module_with_host_break() -> tuple[GraphModule, dict[str, Node]]:
    graph = Graph()
    input_node = graph.placeholder("input")
    add = graph.call_function(torch.ops.aten.add.Tensor, (input_node, 1))
    relu = graph.call_function(torch.ops.aten.relu.default, (add,))
    mul = graph.call_function(torch.ops.aten.mul.Tensor, (relu, 2))
    graph.output(mul)
    return GraphModule({}, graph), {
        "input": input_node,
        "add": add,
        "relu": relu,
        "mul": mul,
    }


def test_partition_marks_whitelist_and_host_breaks_components() -> None:
    gm, nodes = _module_with_host_break()

    partitions = partition_graph(gm)

    assert DPU_LOWERABLE == frozenset(
        {
            torch.ops.aten.addmm.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.tanh.default,
        }
    )
    assert nodes["input"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert nodes["add"].meta[DEVICE_META_KEY] == DEVICE_DPU
    assert nodes["relu"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert nodes["mul"].meta[DEVICE_META_KEY] == DEVICE_DPU
    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [nodes["add"]]),
        (1, [nodes["mul"]]),
    ]
    assert nodes["add"].meta[PART_ID_META_KEY] == 0
    assert nodes["mul"].meta[PART_ID_META_KEY] == 1
    assert PART_ID_META_KEY not in nodes["relu"].meta


def test_partition_keeps_direct_dpu_fork_and_join_together() -> None:
    graph = Graph()
    input_node = graph.placeholder("input")
    source = graph.call_function(torch.ops.aten.add.Tensor, (input_node, 1))
    left = graph.call_function(torch.ops.aten.mul.Tensor, (source, 2))
    right = graph.call_function(torch.ops.aten.tanh.default, (source,))
    joined = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output(joined)
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert len(partitions) == 1
    assert partitions[0].part_id == 0
    assert partitions[0].nodes == [source, left, right, joined]
    assert {node.meta[PART_ID_META_KEY] for node in partitions[0].nodes} == {0}


def test_partition_does_not_merge_dpu_consumers_across_a_host_fan_out() -> None:
    graph = Graph()
    input_node = graph.placeholder("input")
    relu = graph.call_function(torch.ops.aten.relu.default, (input_node,))
    left = graph.call_function(torch.ops.aten.add.Tensor, (relu, 1))
    right = graph.call_function(torch.ops.aten.mul.Tensor, (relu, 2))
    graph.output((left, right))
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [left]),
        (1, [right]),
    ]


def test_partition_numbers_disconnected_components_by_fx_order() -> None:
    graph = Graph()
    first_input = graph.placeholder("first_input")
    second_input = graph.placeholder("second_input")
    first = graph.call_function(torch.ops.aten.add.Tensor, (first_input, 1))
    second = graph.call_function(torch.ops.aten.mul.Tensor, (second_input, 2))
    graph.output((first, second))
    gm = GraphModule({}, graph)

    partitions = partition_graph(gm)

    assert [(partition.part_id, partition.nodes) for partition in partitions] == [
        (0, [first]),
        (1, [second]),
    ]


def test_partition_replaces_stale_metadata_without_touching_other_metadata() -> None:
    gm, nodes = _module_with_host_break()
    nodes["relu"].meta[PART_ID_META_KEY] = 99
    nodes["relu"].meta["sentinel"] = "preserve-me"
    nodes["add"].meta["sentinel"] = "preserve-me-too"

    first = partition_graph(gm)
    second = partition_graph(gm)

    assert [partition.nodes for partition in second] == [partition.nodes for partition in first]
    assert nodes["relu"].meta[DEVICE_META_KEY] == DEVICE_HOST
    assert PART_ID_META_KEY not in nodes["relu"].meta
    assert nodes["relu"].meta["sentinel"] == "preserve-me"
    assert nodes["add"].meta["sentinel"] == "preserve-me-too"


class _FixedMaskLlama(torch.nn.Module):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def _export_random_llama() -> GraphModule:
    sequence_length = 16
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=sequence_length,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.float32)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    return torch.export.export(
        _FixedMaskLlama(model),
        (input_ids, causal_mask),
        strict=True,
    ).module()


def test_partition_covers_the_strictly_exported_random_llama_graph() -> None:
    gm = _export_random_llama()

    partitions = partition_graph(gm)
    nodes = list(gm.graph.nodes)
    dpu_nodes = [node for node in nodes if node.meta[DEVICE_META_KEY] == DEVICE_DPU]
    host_nodes = [node for node in nodes if node.meta[DEVICE_META_KEY] == DEVICE_HOST]
    listed_nodes = [node for partition in partitions for node in partition.nodes]

    assert dpu_nodes
    assert host_nodes
    assert set(listed_nodes) == set(dpu_nodes)
    assert len(listed_nodes) == len(set(listed_nodes))
    assert {partition.part_id for partition in partitions} == set(range(len(partitions)))
    assert {node.meta[PART_ID_META_KEY] for node in dpu_nodes} == set(range(len(partitions)))
    assert all(PART_ID_META_KEY not in node.meta for node in host_nodes)
