"""用 llama2 W4A8 的 GML 实物校验结构规则与算子覆盖。

这份实物比 ResNet50 那份更贴近目标：它是 decode block，带动态量化、KV cache 和
attention，所以规则要在两份上同时成立才算可靠——只在 ResNet50 上成立的规则，
很可能只是那个模型的特性。

实测它推翻了三条从 ResNet50 归纳出的判断，见 docs/gml-lowering-20260914.md 第 18 节。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import gml_llama2_reference_dir
from gml_bridge.from_fx import OP_TYPES
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    field,
    parse_blocks,
)

_REFERENCE_DIR = gml_llama2_reference_dir(required=False)
_REFERENCE_GML = (
    _REFERENCE_DIR / "relay2gml_graph.gml" if _REFERENCE_DIR else None
)

pytestmark = pytest.mark.skipif(
    _REFERENCE_GML is None or not _REFERENCE_GML.is_file(),
    reason="缺少 llama2 GML 参考产物，配置 paths.json 的 "
           "gml_llama2_reference_dir 后可跑",
)


@pytest.fixture(scope="module")
def graph() -> tuple[str, list[str], list[str]]:
    text = _REFERENCE_GML.read_text()
    return text, parse_blocks(text, "node"), parse_blocks(text, "edge")


def test_reference_graph_shape(graph) -> None:
    _, nodes, edges = graph
    assert len(nodes) == 200
    assert len(edges) == 331


def test_every_structure_rule_holds(graph) -> None:
    """五条规则要在这份实物上同时成立。

    修正前有两条不成立：规则 2 不认按生产者命名的缓冲区，规则 4 要求单入口单出口。
    """
    _, nodes, edges = graph
    assert check_rule1_fusion(nodes) == []
    assert check_rule2_buffer_naming(nodes, edges) == []
    assert check_rule3_shape_on_edges(nodes, edges) == []
    assert check_rule4_edge_direction(nodes, edges) == []
    assert check_rule5_absent_fields(nodes) == []


def test_decode_block_has_several_entries_and_exits(graph) -> None:
    """decode block 天然多入口多出口，不是完整网络。

    入口 7 个（hidden state、KV cache、mask 等），出口 3 个（output 加两个
    KV cache 写回）。规则 4 原先按 ResNet50 写成「恰好一个」，过窄了。
    """
    _, nodes, edges = graph
    by_id = {field(b, "id") for b in nodes}
    sources = {field(e, "source") for e in edges}
    targets = {field(e, "target") for e in edges}

    entries = by_id - targets
    exits = by_id - sources
    assert len(entries) == 7
    assert len(exits) == 3

    # 能要求的是每个入口/出口都是缓冲区节点，算子悬空才是真的错。
    by_block = {field(b, "id"): b for b in nodes}
    for node_id in entries | exits:
        assert field(by_block[node_id], "is_buffer") == "1"


def test_two_buffer_naming_conventions_coexist(graph) -> None:
    """常规节点按消费者命名，DynamicScaling 按生产者自己命名。"""
    _, nodes, _ = graph
    by_producer = 0
    by_consumer = 0
    for block in nodes:
        produced = field(block, "output_buffer")
        node_id = field(block, "id")
        if not produced:
            continue
        if produced == f"output_buffer_{node_id}.bin":
            by_producer += 1
        elif produced.startswith("input_buffer"):
            by_consumer += 1

    assert by_producer >= 30, "DynamicScaling 那一族按生产者命名"
    assert by_consumer >= 100, "常规节点仍按消费者命名"


def test_dynamic_quantization_is_used(graph) -> None:
    """llama2 用的是**动态**量化，与 ResNet50 的全静态相反。

    这直接影响方案：`use_dynamic_quantization` 全为 1，且 `DynamicScaling`
    是独立节点类型，带 phase 0..4 的分阶段字段。
    """
    text, nodes, _ = graph
    assert set(re.findall(r"use_dynamic_quantization (\d+)", text)) == {"1"}

    scaling_nodes = [b for b in nodes if field(b, "op_type") == "DynamicScaling"]
    assert len(scaling_nodes) == 36

    # phase 字段确实存在，且不止一个 phase。
    for phase in range(4):
        assert f"input_buffer_phase_{phase}" in text


def test_w4a8_quantization(graph) -> None:
    """权重 int4、激活 int8——名字里的 W4A8。"""
    text, _, _ = graph
    weight_dtypes = set(re.findall(r'weight_buffer_dtype "([^"]+)"', text))
    assert "int4" in weight_dtypes

    input_dtypes = set(re.findall(r'input_buffer_dtype "([^"]+)"', text))
    assert "int8" in input_dtypes


def test_per_group_weight_quantization(graph) -> None:
    """权重按 group=128 分组量化。

    这印证了 `#pim.quant_spec` 的 per_group 档不是预造抽象——ResNet50 没用它，
    llama2 用了。
    """
    text, _, _ = graph
    assert set(re.findall(r"DEBUG_weight_buffer_spg (\d+)", text)) == {"1"}
    group_sizes = set(
        re.findall(r"DEBUG_weight_buffer_spg_group_size (\d+)", text))
    assert group_sizes == {"128"}


def test_llama2_specific_operators_exist(graph) -> None:
    """llama2 有几个 ResNet50 没有的算子类型。"""
    text, _, _ = graph
    for op_type in ("RMSNorm_vpu", "Silu", "Mask", "KV_Cache_DMA",
                    "Llama2Activation", "DynamicScaling", "Split"):
        assert f'op_type "{op_type}"' in text


def test_operator_coverage_gap_is_known(graph) -> None:
    """记录当前映射表与 llama2 实际算子集的差距。

    第 4 轮要补的就是这些——把它写成断言，实现补齐后测试会提醒更新。
    """
    text, _, _ = graph
    actual = set(re.findall(r'op_type "([^"]+)"', text))
    mapped = set(OP_TYPES.values())

    covered = actual & mapped
    assert covered >= {
        "Gemm", "MatMul", "Softmax", "EltwiseAdd", "EltwiseMul", "Transpose",
        "Reshape", "Concat",
        # 按实物校正后补上的：这四个在 GML 里是独立节点，不是折入项。
        "RMSNorm_vpu", "Silu", "Mask", "Split",
    }

    # 剩下五个各有原因：前四个需要量化路径（scale 在运行时算），
    # `Lut` 只作为 contraction 内的融合项出现，不作为独立映射目标。
    missing = actual - mapped
    assert missing == {
        "DynamicScaling", "KV_Cache_DMA", "Llama2Activation",
        "Llama2ActivationDQ", "Lut",
    }, f"映射缺口变了，更新这个断言并同步文档: {sorted(missing)}"

def test_multiple_kantor_blocks_are_used(graph) -> None:
    """B 块确实在用，印证 `kantorBlocks` 该是列表而不是单个 mode。

    PDF 写的是 `kantor_{block}_*`（占位符），但 ResNet50 只用 A 块，无法验证。
    llama2 里 RoPE 节点同时用了 A 与 B。
    """
    text, _, _ = graph
    blocks = set(re.findall(r"[Kk]antor_([A-Z])[_\s]", text))
    assert {"A", "B"} <= blocks, f"预期至少 A/B 两块，实际 {sorted(blocks)}"


def test_rope_is_a_fused_multi_stage_node(graph) -> None:
    """RoPE 在硬件上是一个融合的多级操作，不是若干基本算子的序列。

    它带 6 个 FPSU 块（按 Add_Cos / Add_Sin / Sin / Cos 分工）与 2 个 kantor 块，
    所以 `pim.rope` 应当是单算子——拆开就要重新融合。
    """
    text, nodes, _ = graph
    rope = [b for b in nodes if field(b, "op_type") == "Llama2Activation"]
    assert len(rope) == 1

    body = rope[0]
    fpsu_blocks = set(re.findall(r"fpsu_mode_(\d+)_Llama2Activation", body))
    assert len(fpsu_blocks) == 6, f"预期 6 个 FPSU 块，实际 {sorted(fpsu_blocks)}"

    # cos 与 sin 两路各有自己的定标。
    assert "Llama2Activation_Add_Cos" in body
    assert "Llama2Activation_Add_Sin" in body



def test_reference_edge_dims_are_decode_slots(graph) -> None:
    """参考 decode block 的边形状：token 轴恒为 1，1024 只属于 KV 长度轴。

    我方导出必须对上这份分布。把导出 seq_len 换成 slots.seq 会把 hidden
    写成 `1x1024x4096`，对参考差 120 条。
    """
    import collections
    import re

    _, _, edges = graph
    dims = []
    for block in edges:
        m = re.search(r'dims "([^"]*)"', block)
        assert m, block[:80]
        dims.append(m.group(1))
    got = collections.Counter(dims)
    expected = {
        "1x1x1x1024": 160,
        "1x1x1x128": 68,
        "1x1x128x1024": 32,
        "1x1x1024x128": 32,
        "1x1x1x4096": 19,
        "1x32x1024x128": 6,
        "1x32x1x128": 4,
        "1x1x1x11008": 4,
        "1x1x32x128": 3,
        "3x1x32x1": 2,
        "1x32x128x1024": 1,
    }
    assert got == expected, f"参考边 dims 变了: {dict(got)}"
    assert all("16" not in d.split("x") for d in dims)
    assert "unknown" not in dims
