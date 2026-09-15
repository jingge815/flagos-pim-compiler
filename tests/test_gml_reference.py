"""用参考产物校验 GML 结构规则与字段清单。

参考产物是 ResNet50（`gml_reference_dir`），只确定格式，不是内容基准。这两个
校验器是第三轮序列化器的验收工具，先在参考产物上确认工具本身是对的：规则若在
参考产物上就不成立，说明规则读错了。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import gml_reference_dir
from scripts.gml_field_inventory import classify, field_families
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    parse_blocks,
)

_REFERENCE_DIR = gml_reference_dir(required=False)
_REFERENCE_GML = (
    _REFERENCE_DIR / "relay2gml_graph.gml" if _REFERENCE_DIR else None
)

pytestmark = pytest.mark.skipif(
    _REFERENCE_GML is None or not _REFERENCE_GML.is_file(),
    reason="缺少 GML 参考产物，配置 paths.json 的 gml_reference_dir 后可跑",
)


@pytest.fixture(scope="module")
def graph() -> tuple[str, list[str], list[str]]:
    text = _REFERENCE_GML.read_text()
    return text, parse_blocks(text, "node"), parse_blocks(text, "edge")


def test_reference_graph_shape(graph) -> None:
    _, nodes, edges = graph
    assert len(nodes) == 74
    assert len(edges) == 89


def test_activations_are_always_fused(graph) -> None:
    """规则 1：激活与池化只能折进 contraction，不得作为独立节点。"""
    _, nodes, _ = graph
    assert check_rule1_fusion(nodes) == []


def test_buffers_are_named_after_consumers(graph) -> None:
    """规则 2：生产者的 output_buffer 就是它某个消费者的 input_buffer。"""
    _, nodes, edges = graph
    assert check_rule2_buffer_naming(nodes, edges) == []


def test_shapes_live_only_on_edges(graph) -> None:
    """规则 3：节点内不带形状，形状只在 edge.dims。"""
    _, nodes, edges = graph
    assert check_rule3_shape_on_edges(nodes, edges) == []


def test_graph_has_one_entry_and_one_exit(graph) -> None:
    """规则 4：边是数据流方向，图恰有一个入口与一个出口。"""
    _, nodes, edges = graph
    assert check_rule4_edge_direction(nodes, edges) == []


def test_no_placement_or_execution_fields(graph) -> None:
    """规则 5：不表达硬件放置，也不产出由 L2Analyzer 推导的字段。"""
    _, nodes, _ = graph
    assert check_rule5_absent_fields(nodes) == []


def test_every_field_family_is_classified(graph) -> None:
    """字段清单必须完整：未归类的族说明清单漏了，第二层验证就会有盲区。"""
    text, _, _ = graph
    families = field_families(text)
    _, unclassified = classify(families)
    assert unclassified == [], f"未归类字段族: {unclassified}"
    assert len(families) == 118


def test_multi_input_nodes_carry_per_slot_scaling(graph) -> None:
    """多输入算子的每个输入槽各带一套定标，不是共用一套。

    这一条决定了 `pim.eltwise` 必须能持有两个 datapath。
    """
    text, _, _ = graph
    families = field_families(text)
    for slot in (0, 1):
        assert families[f"fpsu_{slot}_spc"] == 16
        assert families[f"fpsu_mode_{slot}"] == 16
        assert families[f"Scaling_buffer_file_{slot}"] == 16
