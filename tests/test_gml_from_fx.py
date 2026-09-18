"""端到端验证：真实 llama2 图 → 融合 → GML，且通过全部结构规则。

这是第三轮的主判据。参考产物是 ResNet50，我们没有 llama2 的 GML 实物可比对，
所以判据是「产出的 GML 满足从参考产物验证出的那五条规则」，而不是逐字节对比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
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
    """attention 在 GML 里是真实节点，且带三个输入槽。

    本仓的 NumPy 路径把它放在主机侧执行，但那是执行策略而非图结构；若在这里
    跨过它，它的 q/k/v 三个上游会被下游节点误当成自己的输入。

    判据看**槽数**（`input2_node_id` 存在）而不是 `input_count`：
    MatMul 的第二个 operand 走权重通路，`input_count` 故意比槽数少 1
    （见 test_matmul_under_reports_input_count_by_one）。
    """
    assert 'op_type "MatMul"' in gml_text

    matmul = next(
        block for block in parse_blocks(gml_text, "node")
        if 'op_type "MatMul"' in block)
    assert "input2_node_id" in matmul, "attention 应当有三个输入槽"
    assert "input_count 2" in matmul, "三槽的 MatMul 记 2"


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

    用三输入的 MatMul 作样本，按**槽数**定位（不按 `input_count`，理由同上）。
    """
    matmul = next(
        block for block in parse_blocks(gml_text, "node")
        if 'op_type "MatMul"' in block and "input2_node_id" in block)

    assert sum(
        1 for line in matmul.splitlines()
        if line.strip().startswith("residual_input_buffer ")) == 3


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


# ---------------------------------------------------------------------------
# 三份连接信息的同步，以及 MatMul 的 input_count 例外
# ---------------------------------------------------------------------------


def _llama_like_graph():
    """一个带多输入算子与 attention 的小图，用来验连接字段。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=8, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    return export_annotated_graph(model, 8, position_ids, dtype=torch.float32)


def test_residual_output_buffer_is_emitted() -> None:
    """`residual_output_buffer` 必须与 `outputN_node_id` 成对出现。

    原实现只写了 `outputN_node_id`，漏了 residual 那一份 —— 三份连接信息
    （edge / outputN / residual）少一份，对方按 residual 推依赖时会缺边。
    """
    nodes, edges, _ = convert(_llama_like_graph())

    with_ports = [n for n in nodes if any(
        key.startswith("output") and key.endswith("_node_id")
        for key in n.fields)]
    assert with_ports, "应当有带输出端口的节点"

    for node in with_ports:
        residual = [
            value
            for key in ("residual_output_buffer", "residual_output_buffer_")
            for value in (node.fields.get(key) or [])
        ]
        ports = [
            node.fields[f"output{index}_node_id"]
            for index in range(32)
            if f"output{index}_node_id" in node.fields
        ]
        assert sorted(residual) == sorted(ports), \
            f"节点 {node.node_id}: residual {residual} vs 端口 {ports}"


def test_residual_input_buffer_matches_ports() -> None:
    """输入侧同理，三份信息要同步。"""
    nodes, _, _ = convert(_llama_like_graph())

    for node in nodes:
        residual = [
            value
            for key in ("residual_input_buffer", "residual_input_buffer_")
            for value in (node.fields.get(key) or [])
        ]
        ports = [
            node.fields[f"input{index}_node_id"]
            for index in range(32)
            if f"input{index}_node_id" in node.fields
        ]
        assert sorted(residual) == sorted(ports), \
            f"节点 {node.node_id}: residual {residual} vs 端口 {ports}"


def test_matmul_under_reports_input_count_by_one() -> None:
    """MatMul 的 `input_count` 比实际槽数少 1，并带 `MatMul_input_as_weight`。

    第二个 operand 走**权重通路**、不占输入槽。实测参考产物 64 个 MatMul 全如此，
    且恒等式 `Σ input_count + MatMul 数 == 边数` 依赖它（267 + 64 == 331）。
    """
    nodes, edges, _ = convert(_llama_like_graph())

    matmuls = [n for n in nodes if n.fields.get("op_type") == "MatMul"]
    assert matmuls, "小图里应当有 MatMul"

    for node in matmuls:
        slots = sum(
            1 for index in range(32) if f"input{index}_node_id" in node.fields)
        assert node.fields["input_count"] == max(1, slots - 1)
        assert node.fields["MatMul_input_as_weight"] == 1


def test_input_count_identity_holds() -> None:
    """`Σ input_count + MatMul 数 == 边数`——参考产物成立，我方也要成立。

    这条恒等式是独立于生成器推导出来的（从实物统计），所以它能抓到
    生成器与结构校验器「共享同一个错误假设」的那类 bug。
    """
    nodes, edges, _ = convert(_llama_like_graph())

    total = sum(int(n.fields.get("input_count", 0)) for n in nodes)
    matmuls = sum(1 for n in nodes if n.fields.get("op_type") == "MatMul")
    assert total + matmuls == len(edges)


def test_port_keys_use_the_naming_helpers() -> None:
    """端口键名一律走 contracts.gml_names，不在这里拼字符串。

    钉住这条是因为端口 >= 10 时 residual 的键名多一个下划线，
    自己拼字符串迟早会漏掉那个规则。
    """
    nodes, _, _ = convert(_llama_like_graph())

    for node in nodes:
        for key in node.fields:
            if key.startswith("residual_output_buffer"):
                assert key in (
                    names.residual_buffer_key("output", 0),
                    names.residual_buffer_key("output", 10))
            if key.startswith("residual_input_buffer"):
                assert key in (
                    names.residual_buffer_key("input", 0),
                    names.residual_buffer_key("input", 10))
