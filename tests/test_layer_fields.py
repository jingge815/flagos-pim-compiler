"""23 类层：应有键在、不应有键不在、闭合域数值。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gml_bridge.writer import Node
from orchestrator.layer_expand import Layer
from orchestrator.layer_fields import build_layer_fields, classify
from orchestrator.layer_id import LayerIdentity
from orchestrator.layer_render import render_layer_txt


def _identity(node_id, label, op_type, *, phase=None, head=None, layer_id=None,
              task_id=0, prev=(), next_=()):
    layer = Layer(gml_node_id=node_id, label=label, op_type=op_type,
                  phase=phase, head_index=head)
    return LayerIdentity(layer=layer, layer_id=layer_id or node_id,
                         task_id=task_id, prev_tasks=prev, next_tasks=next_)


def _node(node_id, label, op_type, **fields):
    return Node(node_id, {"label": label, "op_type": op_type, **fields})


def _fields(identity, node, widths=None):
    return build_layer_fields(
        identity, node, widths=widths or {identity.layer.gml_node_id: 4096},
        l2_offsets={})


def test_dq_p1_has_pooling_and_phase():
    ident = _identity(24, "dynamic_quantization_linear", "DynamicScaling",
                      phase=0, layer_id=213, task_id=0, next_=(1, 2))
    node = _node(24, "dynamic_quantization_linear", "DynamicScaling",
                 original_shape="[1, 1, 4096]",
                 global_pooling_group_size_phase_0=128,
                 input_buffer="input_buffer_24.bin",
                 residual_input_buffer=[25],
                 residual_output_buffer=[23, 31, 36])
    fields = _fields(ident, node, {24: 4096})
    assert classify(ident, node, {24: 4096}) == "dq_p1"
    assert fields["layer type"] == "pooling"
    assert fields["Pooling Type"] == 4
    assert fields["Pooling Filter Width"] == 128
    assert fields["Output Width"] == 32
    assert fields["dynamic quantization phase"] == 1
    assert fields["Task ID"] == 0
    assert fields["Next task 0"] == 1
    assert fields["Next task 1"] == 2
    text = render_layer_txt(fields)
    assert "Number of frames:" in text
    assert "Layer ID:" in text


def test_softmax_p2_fans_out():
    ident = _identity(18, "mha_softmax_head0", "Softmax",
                      phase=1, head=0, layer_id=319, task_id=1,
                      prev=(0,), next_=(2, 4))
    node = _node(18, "mha_softmax_head0", "Softmax",
                 input_buffer="input_buffer_18.bin")
    fields = _fields(ident, node, {18: 1024})
    assert classify(ident, node, {18: 1024}) == "sm_p2"
    assert fields["layer type"] == "activation"
    assert fields["Activation Type"] == 13
    assert fields["softmax phase"] == 2
    assert fields["Next task 0"] == 2
    assert fields["Next task 1"] == 4
    assert fields["Split Head Index"] == 0


def test_ir_flp_overrides_hw_table():
    """改一处 IR attr，txt 的 Flp 必须变（评审 4 §3.1 反证）。"""
    from dataclasses import replace
    ident = _identity(18, "mha_softmax_head0", "Softmax",
                      phase=1, head=0, layer_id=319)
    ident = replace(ident, layer=replace(ident.layer, flp=(1, 2, 3)))
    node = _node(18, "mha_softmax_head0", "Softmax",
                 input_buffer="input_buffer_18.bin")
    fields = _fields(ident, node, {18: 1024})
    assert fields["Flp min exp"] == 1
    assert fields["Flp max exp"] == 2
    assert fields["Flp mantisa"] == 3


def test_residual_scale_prefers_dual_consumer():
    """残差 output scale 取双输入消费者，不取 RMSNorm。"""
    ident = _identity(14, "add_1", "EltwiseAdd")
    rms = _node(13, "rms", "RMSNorm_vpu")
    add2 = _node(6, "add_2", "EltwiseAdd")
    node = _node(14, "add_1", "EltwiseAdd",
                 input_buffer_0="a.bin", input_buffer_1="b.bin",
                 residual_output_buffer=[13, 6], input_count=2)
    fields = build_layer_fields(
        ident, node, widths={14: 4096}, l2_offsets={},
        nodes_by_id={14: node, 13: rms, 6: add2})
    assert fields["output scale factor buffer"] == "input_0_sf_6.bin"


def test_bmm_weight_orig_uses_head_index():
    ident = _identity(20, "mha_batch_matmul1_head5", "MatMul", head=5)
    node = _node(20, "mha_batch_matmul1_head5", "MatMul",
                 weight_format="weights_transpose",
                 input_buffer="q.bin", output_buffer="s.bin")
    fields = _fields(ident, node, {20: 1024})
    assert fields["DDR Weight Orig Buffer Name"] == "buffer23_map5"


def test_gemm_gate_has_silu_lut():
    ident = _identity(195, "linear_4", "Gemm")
    node = Node(195, {"label": "linear_4", "op_type": "Gemm",
                      "input_buffer": "x.bin", "output_buffer": "y.bin",
                      "input0_node_id": 13},
                contraction=[("fused", {"activation_op_type": "Silu"})])
    fields = _fields(ident, node, {195: 11008, 13: 4096})
    assert classify(ident, node, {195: 11008, 13: 4096}) == "gemm_gate"
    assert fields["Activation Type"] == 13
    assert "Activation LUT file" in fields
    assert fields["Output Width"] == 11008


def test_bmm1_weight_format_3():
    ident = _identity(20, "mha_batch_matmul1_head0", "MatMul", head=0)
    node = _node(20, "mha_batch_matmul1_head0", "MatMul",
                 weight_format="weights_transpose",
                 input_buffer="q.bin", output_buffer="s.bin")
    fields = _fields(ident, node, {20: 1024})
    assert classify(ident, node, {20: 1024}) == "bmm1"
    assert fields["Weight Format"] == 3
    assert fields["Cache idx"] == 0
    assert fields["layer type"] == "matmul"


def test_semantic_stem_follows_gml_label():
    """文件名主干直接用 GML 的语义 label，不在编排器再拼一套。

    参考产物的 label 就是文件名主干（`..._qidx<relay>_params_<node_id>`），
    两侧各拼一次必然发散。
    """
    from orchestrator.layer_fields import semantic_stem
    ident = _identity(23, "self_attn_q_proj_MatMul_qidx2_params_23", "Gemm")
    node = _node(23, "self_attn_q_proj_MatMul_qidx2_params_23", "Gemm",
                 pim_weight_param="model.layers.0.self_attn.q_proj.weight")
    assert semantic_stem(ident, node, "gemm_qko") == (
        "self_attn_q_proj_MatMul_qidx2_params_23")


def test_rmsnorm_omits_kantor_mode():
    ident = _identity(197, "mul_9", "RMSNorm_vpu")
    node = _node(197, "mul_9", "RMSNorm_vpu",
                 input_buffer="a.bin", output_buffer="b.bin")
    fields = _fields(ident, node, {197: 4096})
    assert fields["layer type"] == "vpu"
    assert fields["sublayer type"] == "rmsnorm"
    assert fields["Vpu Axis"] == -1
    assert fields["Use FPSU"] == 0
    assert "Kantor mode" not in fields


def test_classify_all_23_kinds_exist():
    from orchestrator.layer_hw_table import TABLE
    assert len(TABLE) == 24  # 23 + rope_add_q
    for kind in ("dq_p1", "dq_p4", "sm_p1", "sm_p5", "gemm_qko", "gemm_v",
                 "gemm_gate", "bmm1", "bmm2", "mask", "residual", "mlp_mul",
                 "rope_mul_cos", "rope_add_k", "rmsnorm"):
        assert kind in TABLE


def test_mask_with_causal_edge_is_dual_input():
    """Mask 真的接上 causal mask 边界节点（`input_count=2`）时按双输入写。

    复核 20260921 §2.6：这条不能只按 `kind == "mask"` 判，要读 GML 节点
    自己的 `input_count`——否则 seq_len==1 这类没有 causal mask 边的退化
    场景会被硬当成双输入，写出 GML 里不存在对应边的 `input_buffer_1`。
    """
    ident = _identity(103, "mha_mask_head15", "Mask")
    node = _node(103, "mha_mask_head15", "Mask",
                 input_buffer_0="input_buffer_0_103.bin",
                 input_buffer_1="input_buffer_1_178.bin",
                 input0_node_id=104, input1_node_id=202,
                 residual_input_buffer=[104, 202],
                 residual_output_buffer=[102],
                 input_count=2)
    fields = _fields(ident, node, {103: 1024})
    assert fields["number of inputs"] == 2
    assert fields["Datain file 0"] == "input_buffer_0_103.bin"
    assert fields["Datain file 1"] == "input_buffer_1_178.bin"
    assert "L2 input buffer offset 1" in fields


def test_mask_without_causal_edge_is_single_input():
    """Mask 没有接上 causal mask 边界节点（`input_count=1`）时按单输入写。

    这是 §2.6 要防的退化场景：GML 侧只有一条真实边时，`Datain file`/
    `L2 input buffer offset` 就不该分裂成两槎，否则槎 1 引用的名字在 GML
    里没有对应节点声明它，写盘阶段不会落这个 bin，是新的悬空引用。
    """
    ident = _identity(103, "mha_mask_head15", "Mask")
    node = _node(103, "mha_mask_head15", "Mask",
                 input_buffer="input_buffer_103.bin",
                 input0_node_id=104,
                 residual_input_buffer=[104],
                 residual_output_buffer=[102],
                 input_count=1)
    fields = _fields(ident, node, {103: 1024})
    assert fields["number of inputs"] == 1
    assert "Datain file 1" not in fields
    assert "L2 input buffer offset 1" not in fields


def test_is_dual_input_reads_mask_edge_from_node():
    """`l2_alloc._is_dual_input` 对 Mask 的判据要能读 GML 节点的
    `input_count`，不能无条件按 op_type 判（复核 20260921 §2.6，
    与上面两条 `layer_fields` 测试对称）。
    """
    from orchestrator.l2_alloc import _is_dual_input

    dual_node = _node(103, "mha_mask_head15", "Mask", input_count=2)
    single_node = _node(103, "mha_mask_head15", "Mask", input_count=1)
    assert _is_dual_input("Mask", None, dual_node) is True
    assert _is_dual_input("Mask", None, single_node) is False
    # 不传 node 时保持旧的向后兼容行为（纯按 op_type）。
    assert _is_dual_input("Mask", None) is True


def test_rope_mul_scale_unit_splits_q_and_k():
    """Q cos 用段 5、K cos 用段 6、sin 用段 4（评审 3 §2.4）。"""
    q_cos = _identity(22, "rope_q", "Llama2ActivationDQ", phase=0)
    q_node = _node(22, "rope_q", "Llama2ActivationDQ",
                   input_buffer_0="input_buffer_0_22.bin",
                   input_buffer_1="input_buffer_1_22.bin",
                   input_buffer_2="input_buffer_2_22.bin")
    q_fields = _fields(q_cos, q_node, {22: 4096})
    assert "Scaling_buffer_file_5_Llama2Activation_Cos_22.bin" in q_fields[
        "Scaling buffer file 0"]

    k_cos = _identity(30, "rope_k", "Llama2Activation", phase=0)
    k_node = _node(30, "rope_k", "Llama2Activation",
                   input_buffer_0="input_buffer_0_30.bin",
                   input_buffer_1="input_buffer_1_30.bin",
                   input_buffer_2="input_buffer_2_30.bin")
    k_fields = _fields(k_cos, k_node, {30: 4096})
    assert "Scaling_buffer_file_6_Llama2Activation_Cos_30.bin" in k_fields[
        "Scaling buffer file 0"]

    q_sin = _identity(22, "rope_q", "Llama2ActivationDQ", phase=1)
    s_fields = _fields(q_sin, q_node, {22: 4096})
    assert "Scaling_buffer_file_4_Llama2Activation_Sin_22.bin" in s_fields[
        "Scaling buffer file 0"]


def test_cache_dma_buffer_reads_declared_slot0():
    """Original cache file 读 GML 已声明的 input_buffer_0，不自己拼。"""
    from orchestrator.layer_fields import _cache_dma_buffer

    dma = _node(181, "kv_k", "KV_Cache_DMA",
                input_buffer_0="input_buffer_0_181.bin",
                pim_kv_is_key=1)
    got = _cache_dma_buffer({181: dma}, is_key=True)
    assert got == "input_buffer_0_181.bin"


def test_ddr_input_orig_uses_producer_not_slot():
    """DDR Input Orig 名是生产者 buffer<id>，不是槽号占位 buffer0。"""
    ident = _identity(20, "mha_batch_matmul1_head0", "MatMul", head=0)
    node = _node(20, "mha_batch_matmul1_head0", "MatMul",
                 weight_format="weights_transpose",
                 input_buffer="q.bin", output_buffer="s.bin",
                 residual_input_buffer=[21], input0_node_id=21)
    fields = _fields(ident, node, {20: 1024})
    assert fields["DDR Input Orig Buffer Name 0"] == "buffer21"


def test_ddr_output_orig_bmm2_uses_concat_downstream():
    """32 个 bmm2 共享 Concat 下游的 buffer 名。"""
    ident = _identity(16, "mha_batch_matmul2_head0", "MatMul", head=0)
    concat = _node(15, "concat", "Concat", residual_output_buffer=[4])
    node = _node(16, "mha_batch_matmul2_head0", "MatMul",
                 residual_output_buffer=[15], output_buffer="input_buffer_14.bin")
    fields = build_layer_fields(
        ident, node, widths={16: 128}, l2_offsets={},
        nodes_by_id={16: node, 15: concat, 4: _node(4, "down", "DynamicScaling")})
    assert fields["DDR Output Orig Buffer Name"] == "buffer4"


def test_dual_slot1_size_add_is_full_width():
    """rope_add 的槎 1 与槎 0 同宽；Q 路 mul_cos 槎 1 是数据（8192）。"""
    from orchestrator.l2_alloc import _dual_slot1_size
    assert _dual_slot1_size("Llama2Activation", 2, 8192) == 8192
    assert _dual_slot1_size("Llama2ActivationDQ", 0, 8192) == 256
    assert _dual_slot1_size("Llama2ActivationDQ", 0, 256, bcast=0) == 8192
    assert _dual_slot1_size("Llama2Activation", 0, 8192, bcast=1) == 256


def test_naming_normalise_keeps_slot_and_phase():
    """对拍归一：槽位/段号/头号保留，节点号通配（评审 4 §3.3）。"""
    from scripts.diff_prepare_out import _normalise_naming_value
    assert (_normalise_naming_value("input_buffer_0_18.bin")
            == _normalise_naming_value("input_buffer_0_25.bin"))
    assert (_normalise_naming_value("Scaling_buffer_file_5_Llama2Activation_Cos_22.bin")
            != _normalise_naming_value(
                "Scaling_buffer_file_6_Llama2Activation_Cos_184.bin"))
    assert (_normalise_naming_value("input_buffer_18.bin")
            != _normalise_naming_value("input_buffer_0_18.bin"))
    assert (_normalise_naming_value("buffer19_map0")
            != _normalise_naming_value("buffer19_map5"))


def test_residual_datain_reads_declared_buffer():
    """残差 Datain 读 GML 已声明的 input_buffer，不合成 input_buffer_0_N。"""
    ident = _identity(14, "add_1", "EltwiseAdd")
    node = _node(14, "add_1", "EltwiseAdd",
                 input_buffer="input_buffer_14.bin",
                 input_count=1, residual_input_buffer=[15],
                 residual_output_buffer=[13])
    fields = _fields(ident, node, {14: 4096})
    assert fields["Datain file 0"] == "input_buffer_14.bin"
    assert "Datain file 1" not in fields


def test_q_dq_p1_datain_uses_phase0():
    """Q 路 DQ p1 的 Datain 是 input_buffer_phase_0，不是裸 input_buffer。"""
    ident = _identity(184, "rope_q", "Llama2ActivationDQ", phase=3)
    node = _node(184, "rope_q", "Llama2ActivationDQ",
                 input_buffer_0="input_buffer_0_184.bin",
                 input_buffer_phase_0="input_buffer_phase_0_184.bin")
    fields = _fields(ident, node, {184: 4096})
    assert classify(ident, node, {184: 4096}) == "dq_p1"
    assert fields["Datain file"] == "input_buffer_phase_0_184.bin"


def test_compile_slots_kv_and_bmm():
    from contracts.compile_slots import DEFAULT_SLOTS
    assert DEFAULT_SLOTS.bmm_weight_elems == 131072
    assert DEFAULT_SLOTS.kv_cache_elems == 4194304
    assert DEFAULT_SLOTS.kv_index_elems == 96
    assert DEFAULT_SLOTS.kv_new_elems == 4096
