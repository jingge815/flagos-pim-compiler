"""验证 GML 序列化器的文本格式。

格式对了但细节错一处，底层编译器就读不进去，而且不会在我们这侧报错。所以这些
测试盯的是那几个容易写错的地方：数组展开成重复键、形状只在边上、缩进层级、
以及产出的 GML 能通过结构校验器。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# 一层嵌套块（vpu_params）与连接键名
# ---------------------------------------------------------------------------


def test_nested_block_is_one_level_deep() -> None:
    """`vpu_params` 的字段直接写在块内，比 contraction 少一层。

    两种块的层级不同，混用会让对方的解析器读不到字段：
        vpu_params [ Vpu_Axis -1 ]                       <- 一层
        contraction [ fused_Silu_act [ name "..." ] ]    <- 两层
    """
    node = Node(25, {"Use_Scaling": 0}, nested={
        "vpu_params": {"Vpu_Axis": -1, "Weights_buffer_file": "weight_buffer_25.bin"}})
    text = write_gml([node], [], version="26.2.1")

    assert "    vpu_params [\n" in text
    assert '      Vpu_Axis -1\n' in text
    assert '      Weights_buffer_file "weight_buffer_25.bin"\n' in text
    # 块内不该再嵌一层。
    assert "        " not in text


def test_nested_block_indentation_matches_the_reference() -> None:
    """缩进要与实物逐字符一致：块名 4 空格、字段 6 空格。"""
    node = Node(25, {}, nested={"vpu_params": {"Vpu_Axis": -1}})
    lines = write_gml([node], [], version="26.2.1").split("\n")

    block = next(i for i, line in enumerate(lines) if "vpu_params" in line)
    assert lines[block] == "    vpu_params ["
    assert lines[block + 1] == "      Vpu_Axis -1"
    assert lines[block + 2] == "    ]"


def test_both_nested_kinds_can_coexist() -> None:
    """一个节点同时带两种块时，各自的层级都要对。"""
    node = Node(
        195, {"label": "gate"},
        contraction=[("fused_Silu_act", {"op_type": "Lut"})],
        nested={"vpu_params": {"Vpu_Axis": -1}})
    text = write_gml([node], [], version="26.2.1")

    assert "    vpu_params [\n      Vpu_Axis -1\n    ]\n" in text
    assert '    contraction [\n      fused_Silu_act [\n        op_type "Lut"\n' in text


def test_residual_buffer_key_grows_an_underscore_past_port_nine() -> None:
    """端口号 >= 10 时 `residual_*_buffer` 后面多一个下划线。

    这是对方生成器的键名 bug，实测 662 处连接无例外（输出侧 88、输入侧 22 处
    带下划线）。为兼容它的解析器，我方照样复现。
    """
    from contracts.gml_names import port_node_id_key, residual_buffer_key

    assert residual_buffer_key("output", 0) == "residual_output_buffer"
    assert residual_buffer_key("output", 9) == "residual_output_buffer"
    assert residual_buffer_key("output", 10) == "residual_output_buffer_"
    assert residual_buffer_key("input", 31) == "residual_input_buffer_"

    # 端口号那一份键名不带这个 bug。
    assert port_node_id_key("output", 9) == "output9_node_id"
    assert port_node_id_key("output", 10) == "output10_node_id"


def test_residual_buffer_key_rejects_bad_direction() -> None:
    from contracts.gml_names import residual_buffer_key

    with pytest.raises(ValueError, match="只能是 input 或 output"):
        residual_buffer_key("sideways", 0)


def test_repeated_keys_render_once_per_port() -> None:
    """多端口节点上同名键重复出现，每个端口一次——不是写成数组。

    实测 Split 节点有 32 个 `residual_output_buffer`，靠 list 展开机制产出。
    """
    node = Node(21, {"residual_output_buffer": [20, 41, 46]})
    text = write_gml([node], [], version="26.2.1")

    assert text.count("residual_output_buffer ") == 3
    assert "residual_output_buffer 20\n" in text
    assert "residual_output_buffer 41\n" in text
