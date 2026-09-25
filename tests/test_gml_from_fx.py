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
from contracts import gml_hw_table as hw_table
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
    nodes, edges, _, _ = convert(gm)
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
        # 参考把某些轴序写成带引号的字面量（`axes "[0, 2, 1, 3]"`），不是数组字段。
        if '"' in stripped and "[" in stripped:
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


def test_dtype_cast_is_not_silently_skipped() -> None:
    """`to.dtype` 必须有 GML 落点，不能被静默跨过。

    评审九轮问题 4：它不在 `OP_TYPES` 里，`_tensor_inputs` 把它当「不可映射」
    直接跨过——一次真实的数据运动在 GML 里消失，两侧都不报错。
    """
    import torch

    gm = _llama_like_graph()
    casts = [n for n in gm.graph.nodes
             if n.target is torch.ops.aten.to.dtype]
    assert casts, "合成 Llama 图里应该有 to.dtype"
    for node in casts:
        assert node.target in OP_TYPES, (
            f"{node.target} 不在 OP_TYPES 里，GML 导出会静默跨过它")


def test_every_dtype_cast_overload_is_handled() -> None:
    """`to` 的每个重载都要有处置，不能只覆盖 `to.dtype`。

    评审九轮问题 4 的修复只认了 `to.dtype`，而 `torch.export` 也会发
    `to.dtype_layout`——同一件事的另一个重载，漏掉它会让跨过守卫把合法图判错。
    """
    import torch

    from gml_bridge.from_fx import _WALK_THROUGH

    for overload in (torch.ops.aten.to.dtype, torch.ops.aten.to.dtype_layout):
        assert overload in OP_TYPES or overload in _WALK_THROUGH, (
            f"{overload} 既没有 GML 落点，也不在跨过名单里")


def test_embedding_lookup_is_emitted_as_gather() -> None:
    """图入口是 `input_ids`（真带词嵌入）时要发 `Gather` 节点。

    参考 decode 产物以隐藏态为图入口、词嵌入在块外，所以那条导出不发；但那只该是
    decode 块的性质，不能变成「embedding 一律跨过」——全模型导出少了这个词表节点，
    查表这一步就在 GML 里整块消失了。
    """
    nodes, _, weight_params, _ = convert(_llama_like_graph())

    gathers = [n for n in nodes if n.fields.get("op_type") == "Gather"]
    assert len(gathers) == 1, "含词嵌入的图应当恰好发一个 Gather 节点"
    gather = gathers[0]
    # 两个操作数都要指到真实缓冲：表是编译期常量、走权重通路，索引是图入口那一路。
    assert gather.fields["table"] == names.weight_buffer(gather.node_id)
    assert gather.fields["indices"] == gather.fields["input_buffer"]
    assert weight_params[gather.node_id].endswith("embed_tokens.weight")


def test_decode_block_export_omits_gather() -> None:
    """同一条图按 decode 块口径导出时不发 `Gather`——参考产物里没有这个节点。"""
    nodes, _, _, _ = convert(_llama_like_graph(), decode_block_only=True)

    assert not [n for n in nodes if n.fields.get("op_type") == "Gather"]


def _cast_graph(*dtypes):
    """最小图：输入 → 一串 `to.dtype` → 输出，用来验 `Convert` 的发射条件。"""
    import torch
    from torch.fx import Graph, GraphModule

    graph = Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.zeros(1, 4, dtype=dtypes[0])
    current = source
    for dtype in dtypes[1:]:
        cast = graph.call_function(torch.ops.aten.to.dtype, (current, dtype))
        cast.meta["val"] = torch.zeros(1, 4, dtype=dtype)
        current = cast
    graph.output(current)
    return GraphModule(torch.nn.Module(), graph)


def test_only_real_dtype_changes_become_convert_nodes() -> None:
    """位宽真变了才发 `Convert`，恒等转换（同 dtype）不发。

    图：f32 → `to(f16)`（真换）→ `to(f16)`（恒等）。恒等那一步是 `export` 自己
    塞进来的，跨过；真换那一步是一次数据运动（两侧位宽不同），必须有节点承载
    `input_buffer_dtype` / `output_buffer_dtype`。
    """
    import torch

    nodes, _, _, _ = convert(
        _cast_graph(torch.float32, torch.float16, torch.float16))

    converts = [n for n in nodes if n.fields.get("op_type") == "Convert"]
    assert len(converts) == 1, "恒等转换不该成节点"
    cast = converts[0]
    assert cast.fields["input_buffer_dtype"] == "float32"
    assert cast.fields["output_buffer_dtype"] == "float16"
    # 扩展位只覆盖 fp16 与 int8：f32 那侧没有编码可写，就不写，宁缺勿猜。
    assert "input_data_extensions" not in cast.fields
    assert cast.fields["output_data_extension"] == hw_table.data_extension(
        "float16")


def test_residual_output_buffer_is_emitted() -> None:
    """`residual_output_buffer` 必须与 `outputN_node_id` 成对出现。

    原实现只写了 `outputN_node_id`，漏了 residual 那一份 —— 三份连接信息
    （edge / outputN / residual）少一份，对方按 residual 推依赖时会缺边。
    """
    nodes, edges, _, _ = convert(_llama_like_graph())

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
    nodes, _, _, _ = convert(_llama_like_graph())

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
    nodes, edges, _, _ = convert(_llama_like_graph())

    matmuls = [n for n in nodes if n.fields.get("op_type") == "MatMul"]
    assert matmuls, "小图里应当有 MatMul"

    for node in matmuls:
        slots = sum(
            1 for index in range(32) if f"input{index}_node_id" in node.fields)
        assert node.fields["input_count"] == max(1, slots - 1)
        assert node.fields["MatMul_input_as_weight"] == 1
        assert node.fields.get("weight_buffer") == names.weight_buffer(node.node_id)
        assert node.fields.get("weight_sf") == names.weight_scale(node.node_id)
        assert node.fields.get("weight_zp") == names.weight_zero_point(node.node_id)
        assert node.fields.get("weight_buffer_dtype") == "int8"


def test_input_count_identity_holds() -> None:
    """`Σ input_count + MatMul 数 == 边数`——参考产物成立，我方也要成立。

    这条恒等式是独立于生成器推导出来的（从实物统计），所以它能抓到
    生成器与结构校验器「共享同一个错误假设」的那类 bug。
    """
    nodes, edges, _, _ = convert(_llama_like_graph())

    total = sum(int(n.fields.get("input_count", 0)) for n in nodes)
    matmuls = sum(1 for n in nodes if n.fields.get("op_type") == "MatMul")
    assert total + matmuls == len(edges)


def test_idx_names_the_consumer_input_port() -> None:
    """`idx` 是本节点输出挂在消费者的第几个输入端口。

    参考产物 197 个节点各带且仅带一个 `idx`，推导规则是
    `consumer.input<idx>_node_id == self.node_id`（197/197 成立）。
    它不是「第几个输入」——每个节点只有一个，且取值 0..31 是 head 序号，
    `input{i}_node_id` 不编码任何 head 信息。
    """
    nodes, _, _, _ = convert(_llama_like_graph())

    inputs_of: dict[int, dict[int, int]] = {}
    for node in nodes:
        ports = {}
        for index in range(64):
            value = node.fields.get(f"input{index}_node_id")
            if isinstance(value, int):
                ports[index] = value
        inputs_of[node.node_id] = ports

    with_output = [
        n for n in nodes
        if any(key.startswith("output") and key.endswith("_node_id")
               for key in n.fields)
    ]
    assert with_output, "应当有带输出端口的节点"

    for node in with_output:
        assert "idx" in node.fields, f"节点 {node.node_id} 缺 idx"
        idx = node.fields["idx"]
        consumers = [
            nid for nid, ports in inputs_of.items()
            if idx in ports and ports[idx] == node.node_id
        ]
        assert consumers, (
            f"节点 {node.node_id} 的 idx={idx} 在任何消费者的"
            f" input{idx}_node_id 里都对不上")
    """端口键名一律走 contracts.gml_names，不在这里拼字符串。

    钉住这条是因为端口 >= 10 时 residual 的键名多一个下划线，
    自己拼字符串迟早会漏掉那个规则。
    """
    nodes, _, _, _ = convert(_llama_like_graph())

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


def test_every_computing_op_declares_node_dtypes() -> None:
    """做计算的算子必须声明节点级 dtype。

    少一个 dtype 就是底层编译器少一项位宽配置，而那不会在我们这侧报错 ——
    要到对方解析器才炸（评审 4 §2.5）。
    """
    nodes, _, _, _ = convert(_llama_like_graph())

    computing = {"Gemm", "MatMul", "Softmax", "Mask", "EltwiseAdd",
                 "EltwiseMul", "RMSNorm_vpu", "DynamicScaling"}
    checked = 0
    for node in nodes:
        op = node.fields.get("op_type")
        if op not in computing:
            continue
        checked += 1
        assert "output_buffer_dtype" in node.fields, f"{op} 缺 output dtype"
        assert "output_data_extension" in node.fields, f"{op} 缺 output extension"
    assert checked, "小图里应当有做计算的算子"


def test_layout_ops_propagate_dtype_and_carry_no_scale() -> None:
    """布局算子的 dtype 沿边传播，且不带 sf/zp。

    **不能按 op_type 写死**：参考里同一个 `Transpose` 既有 fp16 的
    （concat 之后那条）也有 int8 的（KV 与 QK 那条），`Reshape` 同样两种。
    写死会让一半节点位宽错一倍。sf/zp 则是压根不该有 —— 换轴改形状不改
    动态范围，下游沿用上游的 scale。
    """
    nodes, _, _, _ = convert(_llama_like_graph())
    by_id = {n.node_id: n for n in nodes}

    layout = [n for n in nodes
              if n.fields.get("op_type") in ("Transpose", "Reshape", "Split")]
    assert layout, "小图里应当有布局算子"
    for node in layout:
        op = node.fields["op_type"]
        assert "input_sf" not in node.fields, f"{op} 不该带 input_sf"
        assert "input_zp" not in node.fields, f"{op} 不该带 input_zp"
        # 输出 dtype 要么来自上游（传播），要么是该类的固定值（Split 落定点）。
        assert "output_buffer_dtype" in node.fields
        if op == "Split":
            assert node.fields["output_buffer_dtype"] == "int8"
            continue
        producers = [int(p) for p in
                     (node.fields.get(names.residual_buffer_key("input", 0)) or [])]
        upstream = next((by_id[p] for p in producers if p in by_id), None)
        if upstream is not None and "output_buffer_dtype" in upstream.fields:
            assert (node.fields["output_buffer_dtype"]
                    == upstream.fields["output_buffer_dtype"]), (
                f"{op} 的 dtype 没有沿边传播")


# --- 边形状的槽位口径（评审 20260923 的 P0-1 / P0-2）-------------------------
#
# 这两条钉的是「边的 dims 必须是编译期槽位，不是某一次导出的 seq_len」。
# 之前的判据写成「末维是 1 或 16 就换」，两处错：序列轴不一定在末维
# （`1x32x16x128` 的 16 在下标 2），16 又是字面量（换 128 导出一条都不改写）。
# 实测那时 331 条边里有 121 条带 16、1 条是 `unknown`，而参考产物一条都没有。


def _dims_of(gml_text: str) -> list[str]:
    import re

    return re.findall(r'^\s*dims "([^"]*)"', gml_text, re.M)


def test_small_graph_keeps_its_export_shapes(gml_text: str) -> None:
    """小图不改写：`rewrite_dims` 的判据是图里看得到 hidden=4096。

    合成小图（hidden=64）没有 decode 槽位这回事，按 7B 的 seq=1024 改写它
    只会写出一个与自己的权重形状矛盾的声明。这里钉住「不改写」，
    真实 7B 图上「必须改写」由 `test_gml_export.py` 钉。
    """
    dims = _dims_of(gml_text)
    assert dims, "这张图应当有带形状的边"
    assert "unknown" not in dims, "形状是可知的，不允许写 unknown"
    assert any("16" in d.split("x") for d in dims), (
        "小图应当沿用导出形状（seq_len=16），不该被改写成 1024")


def test_seq_axis_rewrite_finds_the_axis_by_value_not_by_position() -> None:
    """token 轴按**值**定位，换成 1，再左补到四维——这是 decode 参考口径。

    `1x32x128x128` 在 seq_len=128 时末维也是 128，但末维是 head_dim。
    只按值匹配会有两个候选，按位置猜则会挑错一维。把 token 轴换成
    `slots.seq=1024` 会对参考差 120 条边：hidden 参考是 `1x1x1x4096`，
    不是 `1x1024x4096`。
    """
    from contracts.compile_slots import DEFAULT_SLOTS as s
    from gml_bridge.from_fx import _rewrite_seq_axis

    # 序列轴在不同下标上都要命中，换成 1 后补到四维。
    assert _rewrite_seq_axis("1x32x16x128", 16, s) == "1x32x1x128"
    assert _rewrite_seq_axis("1x16x4096", 16, s) == "1x1x1x4096"
    assert _rewrite_seq_axis("1x16", 16, s) == "1x1x1x1"
    # 换个 seq_len 导出也要命中（原来的字面量 16 在这里就失效了）。
    assert _rewrite_seq_axis("1x32x128x128", 128, s) == "1x32x1x128"
    assert _rewrite_seq_axis("1x128x4096", 128, s) == "1x1x1x4096"
    # 已经是槽位口径的 KV 全量（token 轴不是 16）不动，只补维。
    assert _rewrite_seq_axis(f"1x32x{s.seq}x128", 16, s) == f"1x32x{s.seq}x128"
    # hidden / MLP 中间态补到四维。
    assert _rewrite_seq_axis("1x4096", 16, s) == "1x1x1x4096"
    assert _rewrite_seq_axis("1x11008", 16, s) == "1x1x1x11008"


def test_shapeless_edge_and_non_numeric_dims_both_raise() -> None:
    """算不出形状、或形状不是纯数字，都必须抛，不许补一个默认值。

    原来 `_element_count("unknown")` 返回 1，于是那条边的缓冲按 1 个元素落盘；
    声明也是 `unknown`，校验器比的正是这两者，两边一起错就永远绿。
    """
    from contracts.compile_slots import DEFAULT_SLOTS as s
    from gml_bridge.export import _element_count
    from gml_bridge.from_fx import _slot_dims

    with pytest.raises(ValueError, match="unknown"):
        _slot_dims("Gemm", None, s, None, rewrite=True, export_seq=16)
    with pytest.raises(ValueError, match="不是纯数字形状"):
        _element_count("unknown")
    assert _element_count(f"1x32x{s.seq}x128") == 32 * s.seq * 128


def test_split_out_dims_follow_the_consumer_role() -> None:
    """Split 的三条出边按消费者角色分：Q / Kᵀ / V，不是同一份 KV 全量。"""
    from contracts.compile_slots import DEFAULT_SLOTS as s
    from gml_bridge.from_fx import _split_out_dims
    from graph.split_heads import ROLE_MATMUL_PV, ROLE_MATMUL_QK

    class N:
        def __init__(self, role=None):
            self.meta = {"pim_head_role": role} if role else {}

    producer = N()
    qk = N(ROLE_MATMUL_QK)
    pv = N(ROLE_MATMUL_PV)

    def tensor_inputs(node, _emittable):
        # 左操作数：producer 是 inputs[0]；右操作数：producer 是 inputs[1]。
        if node is qk or node is pv:
            return [producer if getattr(node, "_left", False) else N(),
                    producer if not getattr(node, "_left", False) else N()]
        return []

    import gml_bridge.from_fx as fx
    orig = fx._tensor_inputs
    fx._tensor_inputs = tensor_inputs
    try:
        qk._left = True
        assert _split_out_dims(producer, qk, set(), s) == f"1x1x1x{s.head_dim}"
        qk._left = False
        assert _split_out_dims(producer, qk, set(), s) == f"1x1x{s.head_dim}x{s.seq}"
        pv._left = True
        assert _split_out_dims(producer, pv, set(), s) == f"1x1x1x{s.head_dim}"
        pv._left = False
        assert _split_out_dims(producer, pv, set(), s) == f"1x1x{s.seq}x{s.head_dim}"
    finally:
        fx._tensor_inputs = orig


def test_dq_layout_rejects_unknown_last_dim() -> None:
    """未知末维必须抛，不许静默落到 hidden。"""
    from contracts.compile_slots import DEFAULT_SLOTS as s
    import pytest

    assert s.dq_layout(4096, is_attention_scores=False, group_size=128) == (4096, 128)
    assert s.dq_layout(11008, is_attention_scores=False, group_size=128) == (11008, 128)
    with pytest.raises(ValueError, match="last_dim 是 0"):
        s.dq_layout(0, is_attention_scores=False, group_size=128)
    # 未知宽度用张量自己的末维，不许猜成 hidden（32000 曾经静默变成 4096）。
    assert s.dq_layout(32000, is_attention_scores=False, group_size=128) == (32000, 128)
    assert s.dq_layout(512, is_attention_scores=False, group_size=128) == (512, 128)


def test_rtl_version_disagreement_is_rejected() -> None:
    """接了算子编译器时，`rtl_version` 必须与 IR 的模块属性一致。

    评审九轮问题 8：GML 侧一直写本仓常量，IR 上的 `pim.rtl-version` 没人读。
    两处版本号各写一份、谁都不核对，改一边另一边不会报错。
    """
    import pytest

    from gml_bridge.from_fx import _resolve_rtl_version

    assert _resolve_rtl_version(None) == hw_table.RTL_VERSION
    assert _resolve_rtl_version("1.4") == "1.4"
    with pytest.raises(ValueError, match="rtl_version"):
        _resolve_rtl_version("9.9")
