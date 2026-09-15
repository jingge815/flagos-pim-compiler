"""验证 GML 序列化器的文本格式。

格式对了但细节错一处，底层编译器就读不进去，而且不会在我们这侧报错。所以这些
测试盯的是那几个容易写错的地方：数组展开成重复键、形状只在边上、缩进层级、
以及产出的 GML 能通过结构校验器。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
from gml_bridge.writer import Edge, Node, write_gml
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    parse_blocks,
)


def _minimal_graph() -> tuple[list[Node], list[Edge]]:
    """输入缓冲 -> conv（折入 relu）-> 输出缓冲。

    节点 id 逆拓扑编号，缓冲区按消费者编号，都与参考产物的约定一致。
    """
    nodes = [
        Node(1, {
            "label": "in", "name": "in", "is_buffer": 1, "from_tvm": 1,
            "output_buffer": names.data_buffer(3),
            "residual_output_buffer": 3, "output0_node_id": 3,
        }),
        Node(2, {
            "label": "out", "name": "out", "is_buffer": 1,
            "input_buffer": names.data_buffer(2),
            "residual_input_buffer": 3, "input0_node_id": 3, "input_count": 1,
        }),
        Node(3, {
            "label": "conv3", "name": "conv3", "op_type": "Conv",
            "input_buffer": names.data_buffer(3), "input_buffer_dtype": "int8",
            "input_sf": names.scale(3),
            "weight_buffer": names.weight_buffer(3),
            "output_buffer": names.data_buffer(2), "output_buffer_dtype": "int8",
            "kernel_shape": [3, 3], "strides": [1, 1],
            "pads": [1, 1, 1, 1], "dilations": [1, 1], "group": 1,
            "nmu_mode": "fixed_point", "fpsu_mode": "fixed_point",
            "fpsu_spc": 1, "fpsu_spg": 0,
            "Scaling_buffer_file": names.fpsu_scale(3),
            "kantor_mode": "scalar",
            "activation_lut_file": names.activation_lut(3),
            "activation_mode": 0,
            "residual_input_buffer": 3, "input0_node_id": 1, "input_count": 1,
            "residual_output_buffer": 2, "output0_node_id": 2,
        }, contraction=[
            ("fused_relu_3", {
                "name": "relu_3", "op_type": "Lut",
                "activation_op_type": "Relu",
            }),
        ]),
    ]
    edges = [Edge(1, 3, "1x3x224x224"), Edge(3, 2, "1x64x112x112")]
    return nodes, edges


def _write() -> str:
    nodes, edges = _minimal_graph()
    return write_gml(nodes, edges, version="26.10.1")


def test_graph_wrapper_and_version() -> None:
    text = _write()
    assert text.startswith("graph [\n")
    assert "  directed 1\n" in text
    assert '  relay2gml_version "26.10.1"\n' in text
    assert text.endswith("]\n")


def test_arrays_expand_into_repeated_keys() -> None:
    """PDF 明确要求数组拆成单值，不是 `[3, 3]`。"""
    text = _write()
    assert "    kernel_shape 3\n    kernel_shape 3\n" in text
    assert "    pads 1\n    pads 1\n    pads 1\n    pads 1\n" in text
    assert "[3, 3]" not in text


def test_strings_are_quoted_and_ints_are_not() -> None:
    text = _write()
    assert '    op_type "Conv"\n' in text
    assert "    group 1\n" in text
    assert "    fpsu_spc 1\n" in text


def test_id_and_node_id_are_both_written() -> None:
    """前者给 networkx，后者给 L2，参考产物两个都有。"""
    text = _write()
    assert "    id 3\n    node_id 3\n" in text


def test_contraction_block_nests_the_fused_operator() -> None:
    text = _write()
    assert "    contraction [\n" in text
    assert "      fused_relu_3 [\n" in text
    assert '        activation_op_type "Relu"\n' in text


def test_edges_carry_the_shape() -> None:
    """形状只在边上，节点内不带。"""
    text = _write()
    assert '    dims "1x3x224x224"\n' in text
    assert '    label "1x64x112x112"\n' in text


def test_output_passes_every_structure_rule() -> None:
    """最强的一条：产出的 GML 要通过与参考产物同一套校验。"""
    text = _write()
    nodes = parse_blocks(text, "node")
    edges = parse_blocks(text, "edge")

    assert len(nodes) == 3
    assert len(edges) == 2
    assert check_rule1_fusion(nodes) == []
    assert check_rule2_buffer_naming(nodes, edges) == []
    assert check_rule3_shape_on_edges(nodes, edges) == []
    assert check_rule4_edge_direction(nodes, edges) == []
    assert check_rule5_absent_fields(nodes) == []


def test_booleans_are_rejected() -> None:
    """GML 没有布尔类型，误传 True 会写出 `1` 之外的东西，所以直接拒绝。"""
    import pytest

    with pytest.raises(TypeError, match="没有布尔类型"):
        write_gml([Node(1, {"flag": True})], [], version="26.10.1")
