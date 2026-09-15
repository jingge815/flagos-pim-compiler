"""端到端验证：真实 llama2 图 → 融合 → GML，且通过全部结构规则。

这是第三轮的主判据。参考产物是 ResNet50，我们没有 llama2 的 GML 实物可比对，
所以判据是「产出的 GML 满足从参考产物验证出的那五条规则」，而不是逐字节对比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.fuse import ACTIVATIONS, fuse_graph
from gml_bridge.from_fx import OP_TYPES, convert
from gml_bridge.writer import write_gml
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    parse_blocks,
)


@pytest.fixture(scope="module")
def gml_text() -> str:
    from tests.test_partition import _export_random_llama

    gm = _export_random_llama()
    fuse_graph(gm)
    nodes, edges, _ = convert(gm)
    return write_gml(nodes, edges, version="26.10.1")


@pytest.fixture(scope="module")
def blocks(gml_text: str) -> tuple[list[str], list[str]]:
    return parse_blocks(gml_text, "node"), parse_blocks(gml_text, "edge")


def test_graph_is_non_trivial(blocks) -> None:
    nodes, edges = blocks
    assert len(nodes) > 20
    assert len(edges) > 20


def test_output_satisfies_every_structure_rule(blocks) -> None:
    """五条规则同时成立，才说明这份 GML 结构上是合法的。"""
    nodes, edges = blocks
    assert check_rule1_fusion(nodes) == []
    assert check_rule2_buffer_naming(nodes, edges) == []
    assert check_rule3_shape_on_edges(nodes, edges) == []
    assert check_rule4_edge_direction(nodes, edges) == []
    assert check_rule5_absent_fields(nodes) == []


def test_llama_operators_are_mapped(gml_text: str) -> None:
    """llama2 的骨干算子都要落到 GML 的算子类型上。"""
    for op_type in ("Gemm", "EltwiseAdd", "EltwiseMul", "Transpose", "Reshape"):
        assert f'op_type "{op_type}"' in gml_text


def test_attention_is_a_real_node(gml_text: str) -> None:
    """attention 在 GML 里是真实节点。

    本仓的 NumPy 路径把它放在主机侧执行，但那是执行策略而非图结构；若在这里
    跨过它，它的 q/k/v 三个上游会被下游节点误当成自己的输入。
    """
    assert 'op_type "MatMul"' in gml_text
    assert "input_count 3" in gml_text


def test_no_standalone_lut_node(gml_text: str) -> None:
    """`Lut` 只能作为 contraction 内的融合项，不能是独立节点。

    llama2 的 rsqrt 与 silu 映射到 `RMSNorm_vpu` / `Silu` 这两个独立主算子，
    不走 contraction，所以这张图可能没有 contraction 块——但即便如此，
    顶层也绝不该出现 `Lut`。
    """
    from scripts.gml_structure_check import strip_contraction

    for block in parse_blocks(gml_text, "node"):
        assert 'op_type "Lut"' not in strip_contraction(block)


def test_llama2_specific_operators_are_emitted(gml_text: str) -> None:
    """RMSNorm 与 SiLU 作为独立节点产出。"""
    assert 'op_type "RMSNorm_vpu"' in gml_text
    assert 'op_type "Silu"' in gml_text


def test_residual_input_buffer_is_a_repeated_key(gml_text: str) -> None:
    """多输入节点的 residual_input_buffer 是重复键，每个输入一条。

    最初用 dict 存字段时这里被覆盖成了单值，参考产物 node 6 的写法纠正了它。
    """
    lines = gml_text.splitlines()
    starts = [i for i, line in enumerate(lines) if "input_count 3" in line]
    assert starts, "图里应当有三输入节点"
    # 往上找该节点的 residual_input_buffer，应当有三条。
    node_start = max(i for i, line in enumerate(lines[:starts[0]])
                     if line.strip() == "node [")
    window = lines[node_start:starts[0]]
    assert sum(1 for line in window if "residual_input_buffer" in line) == 3


def test_arrays_are_never_written_as_lists(gml_text: str) -> None:
    """数组字段展开成重复键，GML 里不该出现 `[` 除了块开头。"""
    for line in gml_text.splitlines():
        stripped = line.strip()
        if stripped.endswith("["):
            continue
        assert "[" not in stripped, f"数组未展开: {line}"


def test_no_fusable_activation_survives_fusion() -> None:
    """转 GML 之前，融合必须已经吃掉所有**可折**激活。"""
    from tests.test_partition import _export_random_llama

    gm = _export_random_llama()
    fuse_graph(gm)
    remaining = [n for n in gm.graph.nodes
                 if n.op == "call_function" and n.target in ACTIVATIONS]
    assert remaining == []


def test_every_mapped_op_has_a_gml_type() -> None:
    """映射表里不该有空值——那会写出没有 op_type 的节点。"""
    assert all(OP_TYPES.values())
    assert len(set(OP_TYPES.values())) < len(OP_TYPES), (
        "多个 aten 算子映射到同一个 GML 类型是正常的（如 linear/addmm 都是 Gemm）")
