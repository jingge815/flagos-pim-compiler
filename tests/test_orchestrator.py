"""编排器：层展开、Layer ID 发号、L2 分配、net.ini。

层展开那条闭合公式是这组测试的核心：用 `LAYERS_PER_OP` 套参考产物的节点数
必须算出 422 层，与文档步骤 A 的分项完全一致。这条守住了，展开规则就是对的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gml_bridge.writer import Edge, Node
from orchestrator import l2_alloc, net_ini
from orchestrator.layer_expand import (
    FOLDED_OPS,
    LAYERS_PER_OP,
    expand_layers,
)
from orchestrator.layer_id import EXTRA_LAYER_ID_BASE, assign_ids
from orchestrator.plan import orchestrate

# 参考产物（纯 decode block）的非逐头节点数。逐头的 32 组另算。
REFERENCE_NON_PER_HEAD = {
    "RMSNorm_vpu": 2,
    "DynamicScaling": 4,        # 36 个里 32 个是逐头的
    "Gemm": 7,
    "Llama2Activation": 1,
    "Llama2ActivationDQ": 1,
    "EltwiseAdd": 2,
    "EltwiseMul": 1,
}
# 每头 12 层：bmm1 + mask + sm×5 + DQ×4 + bmm2
PER_HEAD_LAYERS = 12
NUM_HEADS = 32


def _node(node_id: int, label: str, op_type: str, **fields) -> Node:
    return Node(node_id, {"label": label, "op_type": op_type, **fields})


def test_layer_expansion_closes_at_422() -> None:
    """用 LAYERS_PER_OP 套参考产物的节点数，必须得 422 层。

    这是文档步骤 A 的闭合公式。对不上说明每类算子拆几层记错了，
    而那会让 prepare_out 少层或多层。
    """
    non_per_head = sum(
        count * LAYERS_PER_OP[op] for op, count in REFERENCE_NON_PER_HEAD.items())
    per_head = NUM_HEADS * PER_HEAD_LAYERS
    assert non_per_head == 38
    assert per_head == 384
    assert non_per_head + per_head == 422


def test_dq_expands_to_four_layers() -> None:
    nodes = [_node(10, "dq0", "DynamicScaling")]
    report = expand_layers(nodes)
    assert report.total == 4
    assert [layer.phase for layer in report.layers] == [0, 1, 2, 3]


def test_softmax_expands_to_five_layers() -> None:
    nodes = [_node(10, "sm0", "Softmax")]
    report = expand_layers(nodes)
    assert report.total == 5


def test_rope_dq_expands_to_seven_layers() -> None:
    """`Llama2ActivationDQ` 是 RoPE 3 连 + DQ 4 相 = 7 层。

    参考产物里那个节点同时有 4 个相位号和 `Llama2Activation_*` 子块。
    """
    nodes = [_node(10, "rope_k", "Llama2ActivationDQ")]
    report = expand_layers(nodes)
    assert report.total == 7
    assert LAYERS_PER_OP["Llama2ActivationDQ"] == 7


def test_single_layer_ops_carry_no_phase() -> None:
    """单层算子不带相位号——文件名靠这个决定要不要 `_phase_N` 后缀。"""
    nodes = [_node(10, "gemm0", "Gemm")]
    report = expand_layers(nodes)
    assert report.total == 1
    assert report.layers[0].phase is None


def test_layout_ops_do_not_occupy_layers() -> None:
    """Transpose / Reshape / Concat / Split / KV_Cache_DMA 不单独占层。"""
    nodes = [_node(i, f"n{i}", op) for i, op in enumerate(sorted(FOLDED_OPS))]
    report = expand_layers(nodes)
    assert report.total == 0
    assert len(report.folded) == len(FOLDED_OPS)


def test_buffer_nodes_are_not_layers() -> None:
    """边界缓冲节点不是算子。"""
    nodes = [Node(99, {"label": "in_x", "is_buffer": 1})]
    assert expand_layers(nodes).total == 0


def test_unknown_op_type_is_reported_not_dropped() -> None:
    """不认识的 op_type 要报出来——静默丢层会让 prepare_out 少一层。"""
    nodes = [_node(10, "weird", "SomethingNew")]
    report = expand_layers(nodes)
    assert report.total == 0
    assert report.unknown == ["weird:SomethingNew"]


def test_per_head_detected_from_label() -> None:
    """逐头靠 label 里的 `headN` 认，不是靠字段。

    实测 GML 的 node.fields 里没有头下标字段：只有 QK 的
    `split_channel_number` 和 Split 的 `num_heads`，逐头的 Mask / Softmax /
    DQ 都不带。
    """
    nodes = [
        _node(10, "mha_softmax_head0", "Softmax"),
        _node(11, "mha_softmax_head31", "Softmax"),
        _node(12, "dynamic_quantization_linear", "DynamicScaling"),
    ]
    report = expand_layers(nodes)
    assert len(report.per_head) == 10          # 两个 softmax × 5 相
    assert len(report.non_per_head) == 4       # 一个 DQ × 4 相
    assert {layer.head_index for layer in report.per_head} == {0, 31}


def test_concat_heads_is_not_per_head() -> None:
    """`mha_concat_heads` 带 head 但没跟数字——它是把 32 头拼回去，只有一个。"""
    nodes = [_node(10, "mha_concat_heads", "EltwiseAdd")]
    report = expand_layers(nodes)
    assert report.layers[0].head_index is None


# --- Layer ID 发号 ---------------------------------------------------------


def test_last_phase_reuses_node_id() -> None:
    """末相沿用 node_id，前几相从 201 起另发。

    不是美学选择：下游的 `Datain file` 引用上游 `Dataout`，而那个名字按
    node_id 生成。末相另发新号会让整张图的 buffer 引用错位。
    """
    report = assign_ids(expand_layers([_node(42, "dq0", "DynamicScaling")]).layers)
    ids = [identity.layer_id for identity in report.identities]
    assert ids[-1] == 42
    assert ids[:-1] == [EXTRA_LAYER_ID_BASE, EXTRA_LAYER_ID_BASE + 1,
                        EXTRA_LAYER_ID_BASE + 2]


def test_task_chain_links_only_within_operator() -> None:
    """Task ID = 相位号，Prev/Next 只连本链内部。

    Softmax 不是线性链：p2 扇出到 p3 和 p5，p5 等 p2 和 p4。
    跨算子依赖走 Datain/Dataout 文件名，不写 Prev/Next。
    """
    report = assign_ids(expand_layers([_node(42, "sm0", "Softmax")]).layers)
    tasks = [(i.task_id, i.prev_tasks, i.next_tasks) for i in report.identities]
    assert tasks[0] == (0, (), (1,))
    assert tasks[1] == (1, (0,), (2, 4))
    assert tasks[2] == (2, (1,), (3,))
    assert tasks[3] == (3, (2,), (4,))
    assert tasks[4] == (4, (1, 3), ())


def test_dq_task_chain_fans_out_from_phase1() -> None:
    """DQ 的 p1 扇出到 p2 和 p3；p3 读 p1，不读 p2。"""
    report = assign_ids(expand_layers([_node(24, "dq0", "DynamicScaling")]).layers)
    tasks = [(i.task_id, i.prev_tasks, i.next_tasks) for i in report.identities]
    assert tasks[0] == (0, (), (1, 2))
    assert tasks[1] == (1, (0,), ())
    assert tasks[2] == (2, (0,), (3,))
    assert tasks[3] == (3, (2,), ())


def test_single_layer_op_has_empty_task_links() -> None:
    report = assign_ids(expand_layers([_node(7, "gemm0", "Gemm")]).layers)
    identity = report.identities[0]
    assert identity.task_id == 0
    assert identity.prev_tasks == () and identity.next_tasks == ()


def test_filename_carries_phase_suffix_only_when_multi() -> None:
    dq = assign_ids(expand_layers([_node(9, "dq0", "DynamicScaling")]).layers)
    assert dq.identities[0].filename == "dq0_phase_0_params_201.txt"
    assert dq.identities[-1].filename == "dq0_phase_3_params_9.txt"

    gemm = assign_ids(expand_layers([_node(9, "g0", "Gemm")]).layers)
    assert gemm.identities[0].filename == "g0_params_9.txt"


def test_layer_ids_are_unique() -> None:
    """发号不能撞——撞了就有两层写同一个文件。"""
    nodes = [_node(i, f"dq{i}", "DynamicScaling") for i in range(2, 20)]
    report = assign_ids(expand_layers(nodes).layers)
    ids = [identity.layer_id for identity in report.identities]
    assert len(ids) == len(set(ids))


# --- L2 分配 ---------------------------------------------------------------


def test_l2_output_bytes_follows_closed_formula() -> None:
    """`L2 output size = (align16(Width) + 16) * elem_bytes`（文档步骤 C）。"""
    assert l2_alloc.l2_output_bytes(4096, 2) == (4096 + 16) * 2
    # 非 16 倍数要先向上对齐。
    assert l2_alloc.l2_output_bytes(100, 2) == (112 + 16) * 2


def test_l2_reuses_slot_when_lifetimes_disjoint() -> None:
    """生命周期不重叠就复用同一地址。"""
    buffers = [
        l2_alloc.L2Buffer("a", 256, produced_at=0, last_read_at=1),
        l2_alloc.L2Buffer("b", 256, produced_at=3, last_read_at=4),
    ]
    plan = l2_alloc.allocate(buffers)
    assert plan.slots == 1
    assert plan.offsets["a"] == plan.offsets["b"]


def test_l2_keeps_overlapping_buffers_apart() -> None:
    """生命周期重叠必须分开——共用地址会互相覆盖。"""
    buffers = [
        l2_alloc.L2Buffer("a", 256, produced_at=0, last_read_at=5),
        l2_alloc.L2Buffer("b", 256, produced_at=2, last_read_at=7),
    ]
    plan = l2_alloc.allocate(buffers)
    assert plan.slots == 2
    assert plan.offsets["a"] != plan.offsets["b"]


def test_l2_lifetime_uses_strict_inequality() -> None:
    """边界相接（一个读完的那步另一个就写）也要分开。

    取等意味着某层同一趟里既读旧缓冲又写新缓冲，写会覆盖未读完的输入。
    与 `memory/mem_planner.greedy_reuse` 的判据保持一致。
    """
    buffers = [
        l2_alloc.L2Buffer("a", 256, produced_at=0, last_read_at=2),
        l2_alloc.L2Buffer("b", 256, produced_at=2, last_read_at=4),
    ]
    plan = l2_alloc.allocate(buffers)
    assert plan.slots == 2


def test_l2_offsets_are_aligned() -> None:
    buffers = [l2_alloc.L2Buffer(f"b{i}", 100, produced_at=i * 3,
                                last_read_at=i * 3 + 1) for i in range(4)]
    plan = l2_alloc.allocate(buffers)
    assert all(offset % l2_alloc.L2_ALIGN == 0
               for offset in plan.offsets.values())


def test_consumers_come_from_edges_not_topo_order() -> None:
    """消费者关系按边算。

    早先用「层列表里紧跟的下一个算子」近似，但层列表本身就是拓扑序，
    近似退化成常量 1，所有缓冲寿命都是 1、复用率虚高到 99%+。
    """
    edges = [Edge(source=10, target=20, dims="1x4096"),
             Edge(source=10, target=30, dims="1x4096")]
    consumers = l2_alloc.consumers_by_node(edges)
    assert consumers == {10: {20, 30}}


def test_output_width_read_from_edge_dims() -> None:
    """单层算子的输出宽度从边的 dims 末维取。

    它们拿不到 `pim.phase-bytes`（算子编译器只给多相算子产），漏掉会让
    复用率虚高——这批恰恰是寿命长的那些。
    """
    edges = [Edge(source=10, target=20, dims="1x16x4096")]
    assert l2_alloc.output_width_by_node(edges) == {10: 4096}


def test_qman_window_matches_reference() -> None:
    """QMAN 段：offset 0x1FFF0000、size 65536，参考产物全层相同。"""
    assert l2_alloc.QMAN_OFFSET == 0x1FFF0000
    assert l2_alloc.QMAN_SIZE == 65536


# --- net.ini ---------------------------------------------------------------


def test_net_ini_lists_layers_in_execution_order() -> None:
    """`[layers]` 按执行序，不重排。

    重排会破坏 `force_consecutive`（RoPE 三连必须连续，中间结果不落 DDR）。
    """
    nodes = [_node(10, "dq0", "DynamicScaling"), _node(11, "g0", "Gemm")]
    identities = assign_ids(expand_layers(nodes).layers).identities
    text = net_ini.render_layers_section(identities)
    lines = text.splitlines()
    assert lines[0] == "[layers]"
    assert lines[1:] == [
        f"layer = {identity.filename.removesuffix('.txt')}"
        for identity in identities]


def test_single_phase_op_attrs_really_drive_txt() -> None:
    """单相算子（矩阵乘）的 txt 硬件域来自 IR，不是查表巧合。

    这条是**反证**：`gemm_gate` 的查表值恰好也是 `Flp (10,17,3)`、`Fpsu 2`，
    所以「导出后 txt 是 10/17/3」并不能说明 IR 接上了 —— 假依赖（完全无视
    `op_attrs`）同样满足。这里把 IR 的值改成一组不可能来自查表的数，
    txt 必须跟着变；不变就是 `_attach_op_attrs` 没接上。
    """
    from dataclasses import replace as dc_replace

    from opcompiler_bridge.oplevel_emitter import EmittedOp
    from opcompiler_bridge.phase_plan import PhasePlan
    from opcompiler_bridge.phase_source import PhaseSource
    from orchestrator.layer_expand import Layer
    from orchestrator.plan import _attach_op_attrs

    layer = Layer(gml_node_id=11, label="linear_4", op_type="Gemm")
    plan = PhasePlan(func="linear_4__matmul", op_attrs={
        # 查表是 (10, 17, 3) / fpsu 2；这里故意全不一样。
        "flp-min-exp": 1, "flp-max-exp": 2, "flp-mantisa": 0,
        "fpsu-mode": 7, "activation-mode": 9,
    })
    source = PhaseSource(
        by_node={"linear_4": {"fused_matmul": plan}},
        ops=[EmittedOp(func="linear_4__matmul", kind="fused_matmul",
                       fx_name="linear_4", expected_phases=0)])

    filled = _attach_op_attrs(layer, source)
    assert filled.flp == (1, 2, 0), "IR 的 FLP 没填进 Layer"
    assert filled.fpsu_mode == 7
    assert filled.activation_mode == 9

    # 再确认这些值真的写进了 txt 字段，而不是停在 Layer 上。
    from orchestrator.layer_fields import build_layer_fields
    from orchestrator.layer_id import LayerIdentity

    ident = LayerIdentity(layer=filled, layer_id=11, task_id=0)
    node = Node(11, {"label": "linear_4", "op_type": "Gemm",
                     "input_buffer": "x.bin", "output_buffer": "y.bin",
                     "input0_node_id": 13},
                contraction=[("fused", {"activation_op_type": "Silu"})])
    fields = build_layer_fields(ident, node, widths={11: 11008, 13: 4096},
                                l2_offsets={})
    assert fields["Flp min exp"] == 1, "txt 的 Flp 仍来自静态表"
    assert fields["Flp max exp"] == 2
    assert fields["Fpsu mode"] == 7


def test_net_ini_general_section_has_strides() -> None:
    text = net_ini.render_general_section()
    assert "is_seq_test = 0" in text
    assert "input_line_stride = 8" in text
    assert "dumps_txt_path" in text


def test_net_ini_line_endings_match_reference() -> None:
    """参考产物的 net.ini 是 CRLF，**只有两条 dumps 路径**是 LF。

    对方解析器按行切分时若把 `\\r` 当值的一部分，行尾写错会让每个值末尾多
    一个字符。这条钉住那个混合写法。
    """
    text = net_ini.render_general_section()
    assert "[general]\r\n" in text
    assert "is_seq_test = 0\r\n" in text
    # 这两行参考是 LF，不带 \r。
    assert "dumps_bin_path = llama2_w4a8_decode_block_0/parser_output\n" in text
    assert "dumps_bin_path = llama2_w4a8_decode_block_0/parser_output\r\n" not in text


def test_net_ini_last_layer_has_no_trailing_newline() -> None:
    """参考 net.ini 末字节是层名最后一个字符，没有 LF。"""
    from orchestrator.layer_id import LayerIdentity
    from orchestrator.layer_expand import Layer
    identities = [
        LayerIdentity(layer=Layer(gml_node_id=1, label="a", op_type="Gemm"),
                      layer_id=1, task_id=0),
        LayerIdentity(layer=Layer(gml_node_id=2, label="b", op_type="Gemm"),
                      layer_id=2, task_id=0),
    ]
    text = net_ini.render_layers_section(identities, stems=["first", "last"])
    assert not text.endswith("\n")
    assert text.endswith("layer = last")


def test_layer_txt_uses_crlf() -> None:
    """422 个层文件参考全是 CRLF。"""
    from collections import OrderedDict

    from orchestrator.layer_render import render_layer_txt

    text = render_layer_txt(OrderedDict([("Number of frames", 1),
                                         ("Layer ID", 7)]))
    assert text == "Number of frames: 1\r\nLayer ID: 7\r\n"


def test_net_ini_constants_match_manual() -> None:
    """全层恒定的硬件口，来自手册 Table 7-11。"""
    from orchestrator.layer_hw_table import CONSTANTS
    assert CONSTANTS["Bytes in cycle internal memory read"] == 64
    assert CONSTANTS["Number of frames"] == 1
    assert CONSTANTS["L2 qman buffer size"] == 65536


# --- 端到端 ----------------------------------------------------------------


class _Artifact:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges


def test_orchestrate_runs_all_four_steps() -> None:
    nodes = [_node(10, "dq0", "DynamicScaling"), _node(11, "g0", "Gemm")]
    edges = [Edge(source=10, target=11, dims="1x4096")]
    plan = orchestrate(_Artifact(nodes, edges))
    assert plan.total_layers == 5           # 4 相 + 1 层
    assert plan.identity.total == 5
    assert "[layers]" in plan.net_ini_text
    # 没有 phase_source 时多相层没有字节数，但单层算子能从边算出来。
    assert plan.l2.buffers >= 1
