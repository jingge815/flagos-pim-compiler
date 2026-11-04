"""CEVA 链路导出：厂商子图解析、参数导入、三类子图与 runner 契约。

全部用小张量和临时目录，不启动厂商工具（设计文档 7.1）。
"""

from __future__ import annotations

import subprocess
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_old_flag_rejected(tmp_path) -> None:
    """`--orchestrate` 在加载模型前非零退出，且不生成 prepare_out。"""
    out_dir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "export_gml.py"),
         "--orchestrate", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        env={"LLAMA2_7B_MODEL_DIR": str(tmp_path / "no-model"),
             **{k: v for k, v in __import__("os").environ.items()
                if k != "LLAMA2_7B_MODEL_DIR"}})
    assert proc.returncode != 0
    assert "--orchestrate" in proc.stderr + proc.stdout
    # 模型目录不存在：若先加载模型，报错里会出现模型加载的痕迹。
    assert "from_pretrained" not in proc.stderr
    assert not (out_dir / "prepare_out").exists()


def test_old_prepare_out_branch_removed() -> None:
    """旧支线模块已删除；mem_layout 只保留通用的三项排布规则。"""
    import importlib.util

    assert importlib.util.find_spec("orchestrator") is None
    assert not (REPO_ROOT / "scripts" / "diff_prepare_out.py").exists()
    from contracts import mem_layout

    public = {n for n in vars(mem_layout) if not n.startswith("_")
              and callable(getattr(mem_layout, n))
              and getattr(getattr(mem_layout, n), "__module__", "") == mem_layout.__name__}
    assert public == {"align_up", "row_major_strides", "check_elem_strides"}


# ---- 厂商子图解析与语义键 ----

_SMALL_GML = '''graph [
  directed 1
  relay2gml_version "19.2.0"
  node [
    id 1
    node_id 1
    label "nprm_0_i0"
    name "nprm_0_i0"
    is_buffer 1
    output_buffer "input_buffer_3.bin"
  ]
  node [
    id 2
    node_id 2
    label "output"
    name "output"
    is_buffer 1
    input_buffer "input_buffer_2.bin"
  ]
  node [
    id 3
    node_id 3
    label "dynamic_quantization_params_3"
    name "dynamic_quantization_params_3"
    op_type "DynamicScaling"
    LUT_phase_2 "LUT_phase_2_3.bin"
  ]
  node [
    id 4
    node_id 4
    label "self_attn_q_proj_MatMul_qidx2_params_4"
    name "self_attn_q_proj_MatMul_qidx2_params_4"
    op_type "Gemm"
    weight_buffer "weight_buffer_4.bin"
    weight_sf "weight_sf_4.bin"
    input_buffer "input_buffer_4.bin"
    contraction [
      fused_Silu_act [
        op_type "Lut"
      ]
    ]
    vpu_params [
      Weights_buffer_file "weight_buffer_4.bin"
    ]
  ]
  node [
    id 5
    node_id 5
    label "dynamic_quantization_params_5"
    name "dynamic_quantization_params_5"
    op_type "DynamicScaling"
    LUT_phase_2 "LUT_phase_2_5.bin"
  ]
  edge [
    source 1
    target 3
    dims "1x1x1x8"
  ]
]
'''


def _write_vendor_graph(root: Path, text: str = _SMALL_GML) -> Path:
    rf = root / "parser_output" / "tvmgen_default_nprm_main_0" / "runtime_files"
    rf.mkdir(parents=True)
    (rf / "relay2gml_graph.gml").write_text(text)
    (rf / "LUT_phase_2_3.bin").write_bytes(b"\x01\x02")
    (rf / "LUT_phase_2_5.bin").write_bytes(b"\x03\x04")
    (rf / "weight_buffer_4.bin").write_bytes(bytes(range(8)))
    (rf / "weight_sf_4.bin").write_bytes(b"\x00\x3c")
    (rf / "input_buffer_4.bin").write_bytes(b"\x00" * 4)
    (rf / "IO_info.txt").write_text(
        "{'inputs': {'1': {'sf': array(1., dtype=float32), 'dtype': 'float16', "
        "'node_name': 'nprm_0_i0', 'input_idx': 0, 'shape': [1, 1, 8], 'size': 8}}, "
        "'outputs': {'2': {'sf': array(1., dtype=float32), 'dtype': 'float16', "
        "'node_name': 'tvmgen_default_nprm_main_0_output_0', 'previous_name': 'output', "
        "'output_idx': 0, 'shape': [1, 1, 8], 'size': 8}}}")
    return rf


def test_operator_key() -> None:
    """去掉 qidx/params 后缀；同名按 id 升序加序号。"""
    from gml_bridge.ceva import operator_key_of

    assert operator_key_of("self_attn_q_proj_MatMul_qidx2_params_23") == "self_attn_q_proj_MatMul"
    assert operator_key_of("dynamic_quantization_params_12") == "dynamic_quantization"
    assert operator_key_of("RMSNorm_params_25") == "RMSNorm"
    assert operator_key_of("mha_softmax_head3_qidx74_params_49") == "mha_softmax_head3"
    assert operator_key_of("nprm_0_i0") == "nprm_0_i0"


def test_parse_vendor_graph_keys(tmp_path) -> None:
    """解析小 GML：节点按语义键索引，同名带序号；嵌套块里的文件也收录。"""
    from gml_bridge.ceva import parse_vendor_graph

    rf = _write_vendor_graph(tmp_path)
    graph = parse_vendor_graph(rf)
    assert graph.version == "19.2.0"
    assert set(graph.nodes) >= {"dynamic_quantization#0", "dynamic_quantization#1",
                                "self_attn_q_proj_MatMul#0"}
    assert graph.nodes["dynamic_quantization#1"]["node_id"] == 5
    gemm = graph.nodes["self_attn_q_proj_MatMul#0"]
    assert gemm["fields"]["weight_buffer"] == "weight_buffer_4.bin"
    assert gemm["nested"]["vpu_params"]["Weights_buffer_file"] == "weight_buffer_4.bin"


def test_load_ceva_spec_contract_only(tmp_path) -> None:
    """契约只取厂商子图的位置、版本与 I/O 约定，不登记厂商 bin（全部 bin 由我方生成）。"""
    from contracts.compile_slots import DEFAULT_SLOTS
    from gml_bridge.ceva import RUNTIME_DIR, graph_dir, load_ceva_spec

    _write_vendor_graph(tmp_path / "decode" / "model_layers_0_decode_v2")
    spec = load_ceva_spec(tmp_path, "decode", 0, DEFAULT_SLOTS)
    assert spec.runtime_dir == graph_dir(tmp_path, "decode", 0) / RUNTIME_DIR
    assert spec.version == "19.2.0"
    assert not hasattr(spec, "files")
    io = spec.io_for("nprm_0_i0")
    assert (io.direction, io.index, io.shape, io.dtype) == ("input", 0, (1, 1, 8), "float16")
    assert spec.io_for("tvmgen_default_nprm_main_0_output_0").label == "output"


def test_load_ceva_spec_missing_graph_fails(tmp_path) -> None:
    import pytest

    from contracts.compile_slots import DEFAULT_SLOTS
    from gml_bridge.ceva import load_ceva_spec

    with pytest.raises(FileNotFoundError):
        load_ceva_spec(tmp_path, "decode", 3, DEFAULT_SLOTS)


# ---- 三类包装器：几何与数值 ----

def _small_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=176,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64)).eval()


def _vendor_rope(model, positions):
    """HF cos/sin 换成厂商编码：sin 后半维取反，形状 [1,T,1,hd]。"""
    import torch

    x = torch.zeros(1, len(positions), model.config.hidden_size)
    cos, sin = model.model.rotary_emb(x, torch.tensor([positions]))
    half = cos.shape[-1] // 2
    sin = sin.clone()
    sin[..., half:] *= -1
    return cos.unsqueeze(2), sin.unsqueeze(2)


def test_prefill_and_decode_match_hf_layer() -> None:
    """prefill 包装器等于 HF 层；decode 包装器用 prefill 的 K/V 续算最后一个 token 也等于 HF。"""
    import torch

    from gml_bridge.ceva import DecodeLayer, PrefillLayer, causal_mask

    model = _small_llama()
    layer = model.model.layers[1]
    S, T, nh, hd = 16, 6, 4, 16
    torch.manual_seed(1)
    hidden = torch.randn(1, T, 64)
    cos_all, sin_all = _vendor_rope(model, list(range(S)))

    pos_ids = torch.arange(T).unsqueeze(0)
    emb = model.model.rotary_emb(hidden, pos_ids)
    ref = layer(hidden, attention_mask=causal_mask(T, T)[:, :, :T, :T],
                position_ids=pos_ids, position_embeddings=emb)
    ref = ref[0] if isinstance(ref, tuple) else ref

    prefill = PrefillLayer(layer, cos_all[:, :T], sin_all[:, :T])
    h, key, value = prefill(hidden, causal_mask(T, T))
    torch.testing.assert_close(h, ref, rtol=1e-4, atol=1e-5)

    # decode：前 T-1 个 token 的 K/V 放进长度 S 的 cache，在位置 T-1 续算。
    _, k_prev, v_prev = PrefillLayer(layer, cos_all[:, :T - 1], sin_all[:, :T - 1])(
        hidden[:, :T - 1], causal_mask(T - 1, T - 1))
    key_cache = torch.zeros(1, nh, S, hd)
    value_cache = torch.zeros(1, nh, S, hd)
    key_cache[:, :, :T - 1] = k_prev
    value_cache[:, :, :T - 1] = v_prev
    position = torch.full((3, 1, nh, 1), T - 1, dtype=torch.int16)
    mask = torch.full((1, 1, 1, S), torch.finfo(torch.float32).min)
    mask[..., :T] = 0
    decode = DecodeLayer(layer)
    h_dec, key_out, value_out = decode(
        hidden[:, T - 1:], cos_all[:, T - 1:T], sin_all[:, T - 1:T],
        key_cache, position, mask, value_cache)
    torch.testing.assert_close(h_dec, ref[:, T - 1:], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(key_out[:, :, :T], key, rtol=1e-5, atol=1e-6)


def test_cache_update() -> None:
    """decode 只改写目标行，其余字节保留；位置 0、1、S-1 都成立。"""
    import torch

    from gml_bridge.ceva import DecodeLayer

    model = _small_llama()
    S, nh, hd = 16, 4, 16
    decode = DecodeLayer(model.model.layers[0])
    cos, sin = _vendor_rope(model, [0])
    for row in (0, 1, S - 1):
        torch.manual_seed(row)
        key = torch.randn(1, nh, S, hd)
        value = torch.randn(1, nh, S, hd)
        position = torch.full((3, 1, nh, 1), row, dtype=torch.int16)
        mask = torch.zeros(1, 1, 1, S)
        _, key_out, value_out = decode(torch.randn(1, 1, 64), cos, sin,
                                       key.clone(), position, mask, value.clone())
        keep = [i for i in range(S) if i != row]
        assert torch.equal(key_out[:, :, keep], key[:, :, keep])
        assert torch.equal(value_out[:, :, keep], value[:, :, keep])
        assert not torch.equal(key_out[:, :, row], key[:, :, row])


def _fused(module, args):
    import torch

    from gml_bridge.export import fuse_for_gml
    from graph.partition import partition_graph

    gm = torch.export.export(module, args, strict=True).module()
    partition_graph(gm)
    report = fuse_for_gml(gm)
    return gm, report


def _decode_args(S=16, nh=4, hd=16, H=64):
    import torch

    return (torch.randn(1, 1, H), torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd),
            torch.randn(1, nh, S, hd), torch.zeros(3, 1, nh, 1, dtype=torch.int16),
            torch.zeros(1, 1, 1, S), torch.randn(1, nh, S, hd))


def test_stage_geometry_decode() -> None:
    """decode：两个 KV_Cache_DMA、三个 Split，K 走整体转置进 Split，图仍可执行且数值不变。"""
    import collections

    import torch

    from gml_bridge.ceva import DecodeLayer
    from gml_bridge.export import serialize_gml

    model = _small_llama()
    decode = DecodeLayer(model.model.layers[0])
    args = _decode_args()
    expected = decode(*args)
    gm, report = _fused(decode, args)
    for got, want in zip(gm(*args), expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    artifact = serialize_gml(gm, report)
    ops = collections.Counter(n.fields.get("op_type") for n in artifact.nodes)
    assert ops["KV_Cache_DMA"] == 2
    assert ops["Split"] == 3
    assert ops["Transpose"] == 4
    assert ops["Llama2ActivationDQ"] == 1 and ops["Llama2Activation"] == 1
    by_id = {n.node_id: n for n in artifact.nodes}
    preds = collections.defaultdict(list)
    for e in artifact.edges:
        preds[e.target].append(by_id[e.source].fields.get("op_type"))
    for node in artifact.nodes:
        if node.fields.get("op_type") == "Split":
            assert preds[node.node_id] in (["Llama2ActivationDQ"], ["Transpose"],
                                           ["KV_Cache_DMA"])
    # 位置下标那条链被 DMA 吸收，不单独成节点。
    assert ops["Convert"] == 0 and ops["Reshape"] == 2


def test_rope_encoding() -> None:
    """厂商 sin 编码：融合吸收符号转换，RoPE 直接绑定原始 sin 输入。"""
    from contracts.graph_meta import ROPE_META_KEY
    from gml_bridge.ceva import DecodeLayer

    model = _small_llama()
    gm, _ = _fused(DecodeLayer(model.model.layers[0]), _decode_args())
    ropes = [n for n in gm.graph.nodes if ROPE_META_KEY in n.meta]
    assert len(ropes) == 2
    for rope in ropes:
        match = rope.meta[ROPE_META_KEY]
        assert match.sin.op == "placeholder" and match.sin.name == "sin"
        assert match.cos.op == "placeholder" and match.cos.name == "cos"
        assert match.sin_encoding == "half_negated"


def test_rope_num_heads_from_head_axis() -> None:
    """RoPE 输出为 [1,T,nh,hd] 时 num_heads 取倒数第二维。"""
    from gml_bridge.ceva import DecodeLayer
    from gml_bridge.export import serialize_gml

    model = _small_llama()
    gm, report = _fused(DecodeLayer(model.model.layers[0]), _decode_args())
    artifact = serialize_gml(gm, report)
    ropes = [n for n in artifact.nodes
             if n.fields.get("op_type") in ("Llama2Activation", "Llama2ActivationDQ")]
    assert [n.fields["num_heads"] for n in ropes] == ["4", "4"]


# ---- 编号、命名与逐节点核对 ----

def _node(node_id, label, op=None, inputs=(), outputs=(), **extra):
    from gml_bridge.writer import Node

    fields = {"label": label, "name": label}
    if op:
        fields["op_type"] = op
    else:
        fields["is_buffer"] = 1
    for slot, src in enumerate(inputs):
        fields[f"input{slot}_node_id"] = src
    if inputs:
        fields["residual_input_buffer"] = list(inputs)
    for slot, dst in enumerate(outputs):
        fields[f"output{slot}_node_id"] = dst
    if outputs:
        fields["residual_output_buffer"] = list(outputs)
    fields.update(extra)
    return Node(node_id, fields)


def _tiny_graph():
    """in(10) -> a(11) -> b(12) -> out(13)；in 另有旁路进 b 的槽 0。"""
    from gml_bridge.writer import Edge

    nodes = [
        _node(13, "out", inputs=(12,), input_buffer="input_buffer_13.bin"),
        _node(12, "b_Add_qidx7_params_12", "EltwiseAdd", inputs=(10, 11), outputs=(13,),
              output_buffer="input_buffer_13.bin"),
        _node(11, "a_MatMul_qidx3_params_11", "Gemm", inputs=(10,), outputs=(12,),
              weight_buffer="weight_buffer_11.bin"),
        _node(10, "in", outputs=(11, 12), output_buffer="input_buffer_11.bin"),
    ]
    edges = [Edge(10, 11, "1x1x1x8"), Edge(10, 12, "1x1x1x8"),
             Edge(11, 12, "1x1x1x8"), Edge(12, 13, "1x1x1x8")]
    return nodes, edges


def test_vendor_order_numbering() -> None:
    """入口按契约槽号先编，其余从出口按输入槽深度优先；文件名尾号随之改。"""
    from gml_bridge.ceva import renumber_vendor_order

    nodes, edges = _tiny_graph()
    nodes, edges = renumber_vendor_order(nodes, edges, sources=[10], sinks=[13])
    by_label = {n.fields["label"]: n for n in nodes}
    assert [by_label[k].node_id for k in ("in", "out", "b_Add_qidx7_params_12",
                                          "a_MatMul_qidx3_params_11")] == [1, 2, 3, 4]
    assert by_label["a_MatMul_qidx3_params_11"].fields["weight_buffer"] == "weight_buffer_4.bin"
    # 输出端口按新编号升序。
    assert by_label["in"].fields["residual_output_buffer"] == [3, 4]
    assert by_label["in"].fields["output0_node_id"] == 3
    assert {(e.source, e.target) for e in edges} == {(1, 4), (1, 3), (4, 3), (3, 2)}


def test_vendor_order_rejects_unreachable() -> None:
    import pytest

    from gml_bridge.ceva import renumber_vendor_order
    from gml_bridge.writer import Edge

    nodes, edges = _tiny_graph()
    nodes.append(_node(20, "stray", "Gemm"))
    with pytest.raises(ValueError, match="走不到"):
        renumber_vendor_order(nodes, edges, sources=[10], sinks=[13])


def test_align_to_vendor_checks_structure() -> None:
    """并行深度优先对齐：op_type 与边形状一致时给出语义键；不一致时报错。"""
    import pytest

    from gml_bridge.ceva import VendorGraph, align_to_vendor, renumber_vendor_order

    nodes, edges = _tiny_graph()
    nodes, edges = renumber_vendor_order(nodes, edges, sources=[10], sinks=[13])
    vendor = VendorGraph(version="19.2.0")
    for key, nid, label, op, ins in (
            ("nprm_0_i0#0", 1, "nprm_0_i0", None, []),
            ("output#0", 2, "output", None, [3]),
            ("b_Add#0", 3, "b_Add_qidx9_params_3", "EltwiseAdd", [1, 4]),
            ("a_MatMul#0", 4, "a_MatMul_qidx5_params_4", "Gemm", [1])):
        fields = {"label": label, **({"op_type": op} if op else {})}
        fields.update({f"input{i}_node_id": s for i, s in enumerate(ins)})
        vendor.nodes[key] = {"node_id": nid, "label": label, "fields": fields, "nested": {}}
        vendor.by_id[nid] = key
    vendor.edges = [(1, 4, "1x1x1x8"), (1, 3, "1x1x1x8"), (4, 3, "1x1x1x8"), (3, 2, "1x1x1x8")]
    keys = align_to_vendor(nodes, edges, vendor)
    assert keys == {1: "nprm_0_i0#0", 2: "output#0", 3: "b_Add#0", 4: "a_MatMul#0"}

    vendor.edges[2] = (4, 3, "1x1x1x16")
    with pytest.raises(ValueError, match="形状"):
        align_to_vendor(nodes, edges, vendor)


def test_q_rope_edge_carries_absorbed_transpose_shape() -> None:
    """Q 路 RoPE 吸收其后的换轴，出边形状是换轴后的 [1,nh,1,hd]。"""
    from gml_bridge.ceva import DecodeLayer
    from gml_bridge.export import serialize_gml

    model = _small_llama()
    gm, report = _fused(DecodeLayer(model.model.layers[0]), _decode_args())
    artifact = serialize_gml(gm, report)
    q = next(n for n in artifact.nodes if n.fields.get("op_type") == "Llama2ActivationDQ")
    assert q.fields.get("transpose") == 1
    assert {e.dims for e in artifact.edges if e.source == q.node_id} == {"1x4x1x16"}


# ---- 导入字节与写盘 ----

def _aligned_tiny(tmp_path):
    """厂商小图与我方小图结构一致，返回 (nodes, edges, keys, vendor, spec)。"""
    from contracts.compile_slots import DEFAULT_SLOTS
    from gml_bridge.ceva import load_ceva_spec, parse_vendor_graph, renumber_vendor_order

    root = tmp_path / "ref"
    rf = root / "llama_lm_head" / "parser_output" / "tvmgen_default_nprm_main_0" / "runtime_files"
    rf.mkdir(parents=True)
    (rf / "relay2gml_graph.gml").write_text('''graph [
  directed 1
  relay2gml_version "26.8.0"
  node [
    id 1
    node_id 1
    label "nprm_0_i0"
    name "nprm_0_i0"
    is_buffer 1
    output_buffer "input_buffer_3.bin"
  ]
  node [
    id 2
    node_id 2
    label "output"
    name "output"
    is_buffer 1
    input_buffer "input_buffer_2.bin"
  ]
  node [
    id 3
    node_id 3
    label "lm_head_MatMul_qidx2_params_3"
    name "lm_head_MatMul_qidx2_params_3"
    op_type "Gemm"
    input_buffer "input_buffer_3.bin"
    weight_buffer "weight_buffer_3.bin"
    weight_sf "weight_sf_3.bin"
    input0_node_id 1
  ]
  edge [
    source 1
    target 3
    dims "1x1x1x8"
  ]
  edge [
    source 3
    target 2
    dims "1x1x1x4"
  ]
]
''')
    (rf / "weight_buffer_3.bin").write_bytes(bytes(range(32)))
    (rf / "weight_sf_3.bin").write_bytes(b"\x11\x22")
    (rf / "input_buffer_3.bin").write_bytes(b"\x07" * 16)
    (rf / "input_buffer_2.bin").write_bytes(b"\x07" * 8)
    (rf / "IO_info.txt").write_text(
        "{'inputs': {'1': {'sf': array(1., dtype=float32), 'dtype': 'float16', "
        "'node_name': 'nprm_0_i0', 'input_idx': 0, 'shape': [1, 1, 8], 'size': 8}}, "
        "'outputs': {'2': {'sf': array(1., dtype=float32), 'dtype': 'float16', "
        "'node_name': 'tvmgen_default_nprm_main_0_output_0', 'previous_name': 'output', "
        "'output_idx': 0, 'shape': [1, 1, 4], 'size': 4}}}")
    spec = load_ceva_spec(root, "lm_head", None, DEFAULT_SLOTS)
    vendor = parse_vendor_graph(rf)
    nodes = [
        _node(7, "output", inputs=(9,), input_buffer="input_buffer_7.bin"),
        _node(9, "head_proj", "Gemm", inputs=(8,), outputs=(7,),
              input_buffer="input_buffer_9.bin", weight_buffer="weight_buffer_9.bin",
              weight_sf="weight_sf_9.bin", output_buffer="input_buffer_7.bin"),
        _node(8, "nprm_0_i0", outputs=(9,), output_buffer="input_buffer_9.bin"),
    ]
    from gml_bridge.writer import Edge
    edges = [Edge(8, 9, "1x1x1x8"), Edge(9, 7, "1x1x1x4")]
    nodes, edges = renumber_vendor_order(nodes, edges, sources=[8], sinks=[7])
    keys = {1: "nprm_0_i0#0", 2: "output#0", 3: "lm_head_MatMul#0"}
    return nodes, edges, keys, vendor, spec


def test_io_info_matches_contract(tmp_path) -> None:
    """IO_info 按契约写名字、槽号、形状、dtype 与 sf，节点号用我方编号。"""
    from gml_bridge.ceva import _parse_io_info, write_ceva_io_info

    nodes, edges, keys, vendor, spec = _aligned_tiny(tmp_path)
    path = write_ceva_io_info(nodes, spec, tmp_path / "out")
    info = _parse_io_info(path.read_text())
    assert info["inputs"]["1"]["node_name"] == "nprm_0_i0"
    assert info["outputs"]["2"]["previous_name"] == "output"
    assert info["outputs"]["2"]["shape"] == [1, 1, 4]
    assert "array(1., dtype=float32)" in path.read_text()


def test_vendor_order_keeps_split_residual_lists() -> None:
    """端口 >= 10 的出口分在 `residual_output_buffer_`，重排后前 10 个与其余仍分开。"""
    from gml_bridge.ceva import renumber_vendor_order
    from gml_bridge.writer import Edge

    heads = list(range(30, 42))
    nodes = [_node(1, "in", outputs=tuple(heads), output_buffer="input_buffer_30.bin"),
             _node(50, "out", inputs=tuple(heads), input_buffer="input_buffer_50.bin")]
    nodes[0].fields["residual_output_buffer"] = heads[:10]
    nodes[0].fields["residual_output_buffer_"] = heads[10:]
    for h in heads:
        nodes.append(_node(h, f"h{h}", "Gemm", inputs=(1,), outputs=(50,)))
    edges = [Edge(1, h, "1x8") for h in heads] + [Edge(h, 50, "1x8") for h in heads]
    nodes, _ = renumber_vendor_order(nodes, edges, sources=[1], sinks=[50])
    entry = next(n for n in nodes if n.fields["label"] == "in")
    assert len(entry.fields["residual_output_buffer"]) == 10
    assert len(entry.fields["residual_output_buffer_"]) == 2
    assert entry.fields["residual_output_buffer"] + entry.fields["residual_output_buffer_"] \
        == sorted(entry.fields[f"output{i}_node_id"] for i in range(12))


def test_export_ceva_graph_small(tmp_path) -> None:
    """小模型走完整导出：GML、IO_info、bin 落盘，引用集等于落盘集，字段问题如实返回。"""
    import pytest
    import torch

    from gml_bridge.ceva import export_ceva_graph

    import dataclasses

    from contracts.compile_slots import CompileSlots

    model = _small_llama()
    nodes, edges, keys, vendor, spec = _aligned_tiny(tmp_path)
    spec = dataclasses.replace(
        spec, slots=CompileSlots.from_config(model.config, max_seq=16))
    with pytest.raises(ValueError, match="节点数|op_type|入边|契约"):
        # 小模型与厂商小图结构不同，对齐必须失败而不是硬写。
        export_ceva_graph(model, spec, tmp_path / "bundle", vendor=vendor,
                          use_opcompiler=False)


def test_vendor_conventions_on_boundaries() -> None:
    """边界与残差的厂商口径：入口带 original_name 与旁路定标，出口带输入定标，组注意力数为字符串。"""
    from gml_bridge.ceva import apply_vendor_conventions
    from gml_bridge.writer import Edge

    nodes = [
        _node(1, "nprm_0_i0", outputs=(4, 3), output_buffer="input_buffer_3.bin",
              pim_is_hidden_entry=1),
        _node(2, "output", inputs=(4,), input_buffer="input_buffer_2.bin"),
        _node(3, "RMSNorm_params_3", "RMSNorm_vpu", inputs=(1,), outputs=(4,),
              input_sf="input_sf_3.bin", input_sf_dtype="float32"),
        _node(4, "add_1_Add_qidx9_params_4", "EltwiseAdd", inputs=(1, 3), outputs=(2,),
              input_sf_dtype="float16", input_0_sf="input_0_sf_4.bin",
              input_0_zp="input_0_zp_4.bin"),
        _node(5, "mha_batch_matmul1_head0", "MatMul", group_attention_data_num=32,
              group_attention_weight_num=32),
    ]
    edges = [Edge(1, 4, "x"), Edge(1, 3, "x"), Edge(3, 4, "x"), Edge(4, 2, "x")]
    apply_vendor_conventions(nodes, edges)
    by = {n.node_id: n.fields for n in nodes}
    assert by[1]["original_name"] == "nprm_0_i0"
    assert by[1]["output_sf"] == "input_0_sf_4.bin" and by[1]["output_zp"] == "input_0_zp_4.bin"
    assert by[2]["input_sf"] == "input_sf_2.bin" and by[2]["input_sf_dtype"] == "float16"
    assert by[2]["input_zp"] == "input_zp_2.bin"
    assert by[3]["input_zp"] == "input_zp_3.bin"
    assert "input_sf_dtype" not in by[4]
    assert by[5]["group_attention_data_num"] == "32"

    # 输出头：hidden 入口只喂 RMSNorm，定标取 RMSNorm 的单槽名字。
    head = [_node(1, "nprm_0_i0", outputs=(3,), pim_is_hidden_entry=1),
            _node(2, "output", inputs=(4,)),
            _node(3, "RMSNorm_params_3", "RMSNorm_vpu", inputs=(1,), outputs=(4,),
                  input_sf="input_sf_3.bin"),
            _node(4, "lm_head_MatMul_qidx2_params_4", "Gemm", inputs=(3,), outputs=(2,),
                  weight_sf="weight_sf_4.bin", nmu_mode="fixed_point")]
    apply_vendor_conventions(head, [Edge(1, 3, "x"), Edge(3, 4, "x"), Edge(4, 2, "x")])
    assert head[0].fields["output_sf"] == "input_sf_3.bin"
    assert head[0].fields["output_zp"] == "input_zp_3.bin"
    assert head[3].fields["weight_sf_dtype"] == "float16"


def test_runner_contract_labels() -> None:
    """对齐后节点名取厂商契约名：runner 绑定的 RoPE 内部缓冲名由 Q 路 RoPE 的 label 派生。"""
    from gml_bridge.ceva import VendorGraph, publish_vendor_labels

    nodes = [_node(1, "nprm_0_i0"), _node(2, "self_attn_Reshape_qidx23_params_185",
                                          "Llama2ActivationDQ")]
    vendor = VendorGraph(version="19.2.0")
    vendor.nodes = {"nprm_0_i0#0": {"node_id": 1, "label": "nprm_0_i0", "fields": {}, "nested": {}},
                    "self_attn_Reshape#0": {"node_id": 2, "label": "self_attn_Reshape_qidx4_params_22",
                                            "fields": {}, "nested": {}}}
    publish_vendor_labels(nodes, {1: "nprm_0_i0#0", 2: "self_attn_Reshape#0"}, vendor)
    assert nodes[1].fields["label"] == nodes[1].fields["name"] == "self_attn_Reshape_qidx4_params_22"


def test_cli_ceva_flags() -> None:
    """CEVA 开关：--ceva-reference、--stage、--layer-index 可解析，默认 all。"""
    from scripts.export_gml import build_parser

    args = build_parser().parse_args(["--ceva-reference", "/r", "--stage", "decode",
                                      "--layer-index", "3"])
    assert (str(args.ceva_reference), args.stage, args.layer_index) == ("/r", "decode", 3)
    assert build_parser().parse_args([]).stage == "all"


def test_cli_ceva_requires_opcompiler(monkeypatch, capsys) -> None:
    """CEVA 模式必须跑算子编译器：只给 --ceva-reference 时直接报参数错误。"""
    import sys

    import pytest

    from scripts import export_gml

    monkeypatch.setattr(sys, "argv", ["export_gml.py", "--ceva-reference", "/r"])
    with pytest.raises(SystemExit) as exc:
        export_gml.main()
    assert exc.value.code == 2 and "--use-opcompiler" in capsys.readouterr().err


def test_bundle_targets() -> None:
    """--stage/--layer-index 展开成待导出的 (阶段, 层) 列表。"""
    from gml_bridge.ceva import bundle_targets

    assert len(bundle_targets("all", None, 32)) == 65
    assert bundle_targets("decode", 0, 32) == [("decode", 0)]
    assert bundle_targets("lm_head", None, 32) == [("lm_head", None)]
    assert bundle_targets("prefill", None, 2) == [("prefill", 0), ("prefill", 1)]


def test_split_weight_output_named_after_last_consumer() -> None:
    """走权重通路的 Split 输出按编号最大的消费者命名（厂商口径），不与首个消费者自己的权重重名。"""
    from gml_bridge.ceva import renumber_vendor_order
    from gml_bridge.writer import Edge

    nodes = [_node(1, "in", outputs=(5,), output_buffer="input_buffer_5.bin"),
             _node(5, "Split_params_5", "Split", inputs=(1,), outputs=(6, 7),
                   output_buffer="weight_buffer_6.bin"),
             _node(6, "mha_batch_matmul1_head0", "MatMul", inputs=(1, 5), outputs=(9,),
                   weight_buffer="weight_buffer_6.bin"),
             _node(7, "mha_batch_matmul1_head1", "MatMul", inputs=(1, 5), outputs=(9,),
                   weight_buffer="weight_buffer_7.bin"),
             _node(9, "out", inputs=(6, 7), input_buffer="input_buffer_9.bin")]
    edges = [Edge(1, 5, "x"), Edge(1, 6, "x"), Edge(1, 7, "x"), Edge(5, 6, "x"),
             Edge(5, 7, "x"), Edge(6, 9, "x"), Edge(7, 9, "x")]
    nodes, _ = renumber_vendor_order(nodes, edges, sources=[1], sinks=[9])
    split = next(n for n in nodes if n.fields.get("op_type") == "Split")
    last = max(split.fields[f"output{i}_node_id"] for i in range(2))
    assert split.fields["output_buffer"] == f"weight_buffer_{last}.bin"


def test_stage_geometry_head() -> None:
    """输出头：final norm + 投影，不插 DQ；投影是 int8 静态量化、fixed2float。"""
    import collections

    import torch

    from gml_bridge.ceva import HeadLayer
    from gml_bridge.export import serialize_gml

    from gml_bridge.ceva import mark_static_quant
    from gml_bridge.export import fuse_for_gml
    from graph.partition import partition_graph

    model = _small_llama()
    head = HeadLayer(model)
    x = torch.randn(1, 1, 64)
    gm = torch.export.export(head, (x,), strict=True).module()
    partition_graph(gm)
    mark_static_quant(gm)
    report = fuse_for_gml(gm)
    torch.testing.assert_close(gm(x), model.lm_head(model.model.norm(x)))
    artifact = serialize_gml(gm, report)
    ops = collections.Counter(n.fields.get("op_type") for n in artifact.nodes)
    assert ops == {"RMSNorm_vpu": 1, "Gemm": 1, None: 2}
    gemm = next(n.fields for n in artifact.nodes if n.fields.get("op_type") == "Gemm")
    norm = next(n.fields for n in artifact.nodes if n.fields.get("op_type") == "RMSNorm_vpu")
    assert "use_dynamic_quantization" not in gemm
    assert (gemm["weight_buffer_dtype"], gemm["nmu_mode"], gemm["fpsu_mode"],
            gemm["input_buffer_dtype"], gemm["pooling_dtype"]) == (
        "int8", "fixed_point", "fixed2float", "int8", "floating_point")
    assert (norm["output_buffer_dtype"], norm["output_data_extension"]) == ("int8", 1)


def _prefill(T=8):
    import torch

    from gml_bridge.ceva import PrefillLayer

    model = _small_llama()
    cos, sin = _vendor_rope(model, list(range(T)))
    return model, PrefillLayer(model.model.layers[0], cos, sin), (
        torch.randn(1, T, 64), torch.zeros(1, 1, T, T))


def test_stage_geometry_prefill() -> None:
    """prefill：无 DMA；cos/sin 是节点内常量；K 由一个整体换轴喂 Split，另一个换轴出 cache。"""
    import collections

    import torch

    from gml_bridge.export import serialize_gml

    _, module, args = _prefill()
    expected = module(*args)
    gm, report = _fused(module, args)
    for got, want in zip(gm(*args), expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    artifact = serialize_gml(gm, report)
    ops = collections.Counter(n.fields.get("op_type") for n in artifact.nodes)
    assert ops["KV_Cache_DMA"] == 0 and ops["Split"] == 3 and ops["Transpose"] == 4
    assert ops[None] == 5  # hidden、mask 两个入口，hidden、K、V 三个出口
    by_id = {n.node_id: n for n in artifact.nodes}
    ropes = [n for n in artifact.nodes
             if n.fields.get("op_type") in ("Llama2Activation", "Llama2ActivationDQ")]
    for rope in ropes:
        assert rope.fields["input_count"] == 1
        assert {"input_buffer_1", "input_buffer_2"} <= set(rope.fields)
    k_rope = next(n for n in ropes if n.fields["op_type"] == "Llama2Activation")
    outs = [by_id[e.target] for e in artifact.edges if e.source == k_rope.node_id]
    assert sorted(o.fields.get("op_type") for o in outs) == ["Transpose", "Transpose"]
    perms = sorted(o.fields["axes"] for o in outs)
    assert perms == ["[0, 2, 1, 3]", "[0, 2, 3, 1]"]


def test_prefill_graph_outputs_get_exit_buffers() -> None:
    """有下游的算子若同时是图输出（prefill 的 K/V 换轴），也要带出口缓冲。"""
    from gml_bridge.export import serialize_gml

    _, module, args = _prefill()
    gm, report = _fused(module, args)
    artifact = serialize_gml(gm, report)
    by_id = {n.node_id: n for n in artifact.nodes}
    targets = {e.target for e in artifact.edges}
    sinks = [n for n in artifact.nodes if n.fields.get("is_buffer") and n.node_id in targets]
    assert len(sinks) == 3
    feeders = sorted(by_id[e.source].fields["op_type"] for e in artifact.edges
                     if by_id[e.target] in sinks)
    assert feeders == ["EltwiseAdd", "Transpose", "Transpose"]
    assert sorted(s.fields.get("pim_kv_is_key", -1) for s in sinks) == [-1, 0, 1]


def test_prefill_edges_keep_token_axis() -> None:
    """prefill 几何按查询长度 T：QK/Mask/Softmax 为 1x1xTxS，PV 为 1x1xTxhd，DQ 按行分组。"""
    import dataclasses

    from contracts.compile_slots import DEFAULT_SLOTS
    from gml_bridge.export import serialize_gml

    T = 8
    _, module, args = _prefill(T)
    gm, report = _fused(module, args)
    slots = dataclasses.replace(DEFAULT_SLOTS, hidden=64, intermediate=176, head_dim=16,
                                seq=T, heads=4, tokens=T)
    artifact = serialize_gml(gm, report, slots=slots)
    by_id = {n.node_id: n for n in artifact.nodes}
    dims = collections_by_op(artifact, by_id)
    assert dims["Softmax"] == {f"1x1x{T}x{T}"}
    assert dims["Mask"] == {f"1x1x{T}x{T}"}
    dq = [n.fields for n in artifact.nodes if n.fields.get("op_type") == "DynamicScaling"]
    shapes = {(f["original_shape"], f["output_shape_by_group"]) for f in dq}
    assert (f"[1, 1, {T}, {T}]", f"[1, 1, {T}, 1, {T}]") in shapes


def collections_by_op(artifact, by_id):
    out: dict = {}
    for e in artifact.edges:
        op = by_id[e.source].fields.get("op_type")
        out.setdefault(op, set()).add(e.dims)
    return out


def test_vendor_conventions_kv_exit_writer() -> None:
    """写 KV 出口的算子：output_buffer 按出口编号，并带出口的定标名（prefill 的换轴、decode 的 DMA 同口径）。"""
    from gml_bridge.ceva import apply_vendor_conventions
    from gml_bridge.writer import Edge

    nodes = [
        _node(1, "nprm_0_i0", outputs=(3,)),
        _node(2, "key_cache_out", inputs=(3,), input_buffer="input_buffer_2.bin",
              input_sf="input_sf_2.bin", pim_kv_is_key=1),
        _node(3, "self_attn_Transpose_2", "Transpose", inputs=(1,), outputs=(2,),
              output_buffer="input_buffer_9.bin"),
    ]
    edges = [Edge(1, 3, "x"), Edge(3, 2, "x")]
    apply_vendor_conventions(nodes, edges)
    t = nodes[2].fields
    assert (t["output_buffer"], t["output_sf"], t["output_zp"]) == (
        "input_buffer_2.bin", "input_sf_2.bin", "input_zp_2.bin")


def test_vendor_conventions_prefill_specifics() -> None:
    """prefill 口径：吃 RoPE-DQ 的 Split 输入记 fp16、无 input_sf；EltwiseMul 不带 transpose。"""
    from gml_bridge.ceva import apply_vendor_conventions
    from gml_bridge.writer import Edge

    nodes = [_node(1, "q", "Llama2ActivationDQ", outputs=(2,)),
             _node(2, "Split_params_2", "Split", inputs=(1,), input_buffer_dtype="int8",
                   input_sf="output_buffer_phase_1_1.bin"),
             _node(3, "mlp_mul", "EltwiseMul", transpose=1)]
    apply_vendor_conventions(nodes, [Edge(1, 2, "x")], stage="prefill")
    assert nodes[1].fields["input_buffer_dtype"] == "float16"
    assert "input_sf" not in nodes[1].fields
    assert nodes[1].fields["input_buffer"] == "input_buffer_2.bin"
    assert "transpose" not in nodes[2].fields
    nodes[1].fields.update(input_buffer_dtype="int8", input_sf="x.bin")
    nodes[2].fields["transpose"] = 1
    apply_vendor_conventions(nodes, [Edge(1, 2, "x")], stage="decode")
    assert nodes[1].fields["input_sf"] == "x.bin" and nodes[2].fields["transpose"] == 1


def test_head_fpsu_mode_comes_from_opcompiler() -> None:
    """输出头的 FPSU 模式由算子编译器决定：emitter 发 int8 静态矩阵乘，FlagTree 保留 fixed2float，GML 按读回值写。"""
    import pytest
    import torch

    from opcompiler_bridge.phase_source import opcompiler_available, phase_source_from_graph

    if not opcompiler_available():
        pytest.skip("需要 FlagTree triton-opt")
    from gml_bridge.ceva import HeadLayer, mark_static_quant
    from gml_bridge.export import fuse_for_gml, serialize_gml
    from graph.partition import partition_graph

    model = _small_llama()
    x = torch.randn(1, 1, 64)
    gm = torch.export.export(HeadLayer(model), (x,), strict=True).module()
    partition_graph(gm)
    mark_static_quant(gm)
    report = fuse_for_gml(gm)
    source = phase_source_from_graph(gm)
    head = next(n.name for n in gm.graph.nodes if n.target is torch.ops.aten.linear.default)
    plan = source.plan(head, "static_matmul")
    assert plan.fpsu_mode_name == "fixed2float"
    artifact = serialize_gml(gm, report, phase_source=source)
    gemm = next(n.fields for n in artifact.nodes if n.fields.get("op_type") == "Gemm")
    assert gemm["fpsu_mode"] == "fixed2float"


def test_head_fpsu_missing_plan_fails() -> None:
    """接了算子编译器却没有输出头计划时直接报错，不能静默退回 fixed2float。"""
    import pytest
    import torch

    from gml_bridge.ceva import HeadLayer, mark_static_quant
    from gml_bridge.export import fuse_for_gml, serialize_gml
    from graph.partition import partition_graph
    from opcompiler_bridge.phase_source import PhaseSource

    model = _small_llama()
    gm = torch.export.export(HeadLayer(model), (torch.randn(1, 1, 64),), strict=True).module()
    partition_graph(gm)
    mark_static_quant(gm)
    report = fuse_for_gml(gm)
    with pytest.raises(ValueError, match="static_matmul"):
        serialize_gml(gm, report, phase_source=PhaseSource())


def test_full_pipeline_ceva_verdict(tmp_path) -> None:
    """全流程 CEVA 后端只认 PASS 且无 FAIL/SKIP 的判定报告。"""
    import pytest
    from scripts.run_full_pipeline import StepFailed, ceva_verdict

    verdict = tmp_path / "verdict.txt"
    with pytest.raises(StepFailed):
        ceva_verdict(verdict)
    ok = "[PASS] token 与 baseline 一致     本次 [13]\n  结论: PASS（全部必需检查通过）\n"
    verdict.write_text(ok)
    assert len(ceva_verdict(verdict)) == 1
    verdict.write_text("[SKIP] token 与 baseline 一致\n" + ok)
    with pytest.raises(StepFailed):
        ceva_verdict(verdict)
    verdict.write_text("  结论: FAIL\n")
    with pytest.raises(StepFailed):
        ceva_verdict(verdict)


def test_full_pipeline_ceva_needs_run_name(monkeypatch) -> None:
    """`--backend ceva` 不给实验名直接退出，不启动任何步骤。"""
    from scripts import run_full_pipeline

    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--backend", "ceva"])
    monkeypatch.setattr(run_full_pipeline, "run_ceva",
                        lambda *a: (_ for _ in ()).throw(AssertionError("不应运行")))
    assert run_full_pipeline.main() == 2


def test_bundle_complete_counts_reference_graphs(tmp_path) -> None:
    """complete 按厂商参考目录里实际有的子图算，截层导出的调试包不能标成完整。"""
    from gml_bridge.ceva import GML_NAME, bundle_complete, graph_dir, RUNTIME_DIR

    for stage, layer in [("prefill", 0), ("prefill", 1), ("decode", 0), ("decode", 1),
                         ("lm_head", None)]:
        d = graph_dir(tmp_path, stage, layer) / RUNTIME_DIR
        d.mkdir(parents=True)
        (d / GML_NAME).write_text("graph [ ]")
    one_layer = [{"stage": s, "layer_id": i, "generation_digest": "d1"}
                 for s, i in [("prefill", 0), ("decode", 0), ("lm_head", None)]]
    assert not bundle_complete(tmp_path, one_layer, "d1")
    both = one_layer + [{"stage": "prefill", "layer_id": 1, "generation_digest": "d1"},
                        {"stage": "decode", "layer_id": 1, "generation_digest": "d1"}]
    assert bundle_complete(tmp_path, both, "d1")
    # 摘要不一致的图不能算进完整包。
    assert not bundle_complete(tmp_path, both[:3] + [dict(both[3], generation_digest="d2"),
                                                     both[4]], "d1")


def test_bundle_marked_incomplete_before_first_graph(tmp_path, monkeypatch) -> None:
    """旧包 complete=true 时，若本次在第一张图前就失败，磁盘清单也必须已改为 false。"""
    import json

    import pytest

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    out = tmp_path / "bundle"
    out.mkdir()
    (out / "bundle.json").write_text(json.dumps({"complete": True, "graphs": []}))

    def boom(*a, **k):
        raise FileNotFoundError("参考图不存在")

    monkeypatch.setattr(ceva, "load_ceva_spec", boom)
    with pytest.raises(FileNotFoundError):
        ceva.export_ceva_bundle(_small_llama(), tmp_path / "ref", out, [("decode", 0)],
                                calibration=LlamaCalibration(kv=[(0.1, 0.2)]))
    assert json.loads((out / "bundle.json").read_text())["complete"] is False


def test_ceva_manifest_requires_generated_files(tmp_path) -> None:
    """CEVA 包的每个文件都必须是我方生成、摘要与落盘一致；出现导入的厂商文件即判失败。"""
    import hashlib
    import json

    from scripts.verify_gml_artifact import Report, check_ceva_manifest

    graph = tmp_path / "bundle" / "g"
    graph.mkdir(parents=True)
    (graph / "weight_buffer_3.bin").write_bytes(b"\x01")
    good = {"path": "weight_buffer_3.bin", "size_bytes": 1, "role": "generated",
            "sha256": hashlib.sha256(b"\x01").hexdigest(), "source": None}
    path = tmp_path / "bundle" / "bundle.json"
    for entry, ok in ((good, True), (dict(good, role="imported"), False),
                      (dict(good, sha256="0" * 64), False)):
        path.write_text(json.dumps({"graphs": [{"directory": "g", "files": [entry]}]}))
        report = Report()
        check_ceva_manifest(path, report)
        assert report.failed != ok, entry

def test_shared_passes_have_no_target_specific_code() -> None:
    """共享的图编译、算子编译与 GML 序列化不认 CEVA：目标相关的口径只能在 gml_bridge/ceva.py。"""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    files = (list((root / "graph").glob("*.py")) + list((root / "opcompiler_bridge").glob("*.py"))
             + [root / "gml_bridge" / "from_fx.py", root / "gml_bridge" / "export.py",
                root / "contracts" / "unified_ir.py", root / "contracts" / "ir_payloads.py"])
    hits = [f"{p.relative_to(root)}:{i}" for p in files
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"ceva", line, re.I) and "producer=" not in line]
    assert hits == [], hits


def test_prefill_kv_outputs_found_by_structure() -> None:
    """K/V 作为图输出、图里没有 cache 更新时，不插 DMA，出口由结构识别，不需要目标标记。"""
    import torch

    from contracts.graph_meta import KV_DMA_META_KEY, ROPE_META_KEY
    from gml_bridge.ceva import PrefillLayer
    from gml_bridge.export import fuse_for_gml, serialize_gml
    from graph.partition import partition_graph

    model = _small_llama()
    T, hd = 8, 16
    layer = PrefillLayer(model.model.layers[0], torch.randn(1, T, 1, hd), torch.randn(1, T, 1, hd))
    gm = torch.export.export(layer, (torch.zeros(1, T, 64), torch.zeros(1, 1, T, T)),
                             strict=True).module()
    partition_graph(gm)
    report = fuse_for_gml(gm)
    assert not any(KV_DMA_META_KEY in n.meta for n in gm.graph.nodes)
    assert sum(ROPE_META_KEY in n.meta for n in gm.graph.nodes) == 2
    artifact = serialize_gml(gm, report)
    labels = {n.fields.get("label") for n in artifact.nodes}
    assert {"key_cache_out", "value_cache_out"} <= labels
    rope = [n.fields for n in artifact.nodes if n.fields.get("op_type", "").startswith("Llama2Activation")]
    assert rope and all(int(f["num_heads"]) == 4 for f in rope if "num_heads" in f)


def test_static_quant_marks_output_projection_by_structure() -> None:
    """静态量化按结构选中直接产出图输出的投影，与名字无关。"""
    import torch

    from contracts.unified_ir import STATIC_QUANT_META_KEY
    from gml_bridge.ceva import mark_static_quant

    class Tail(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(64)
            self.proj = torch.nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.proj(self.norm(x))

    gm = torch.export.export(Tail(), (torch.zeros(1, 1, 64),), strict=True).module()
    mark_static_quant(gm)
    marked = [n for n in gm.graph.nodes if n.meta.get(STATIC_QUANT_META_KEY)]
    assert [n.target for n in marked] == [torch.ops.aten.linear.default]


def test_weight_multiplier_comes_from_weight_data() -> None:
    """次正规倍数按权重数值算：scale 落进 fp16 次正规区的投影才带，倍数与量化器一致。"""
    import torch

    from contracts.compile_slots import CompileSlots
    from gml_bridge.ceva import DecodeLayer
    from gml_bridge.export import serialize_gml
    from quant.weights import subnormal_multiplier

    model = _small_llama()
    layer = model.model.layers[0]
    with torch.no_grad():
        layer.self_attn.v_proj.weight.mul_(1e-5)
    gm, report = _fused(DecodeLayer(layer), _decode_args())
    artifact = serialize_gml(gm, report, slots=CompileSlots.from_config(model.config, max_seq=16))
    by_param = {n.fields["pim_weight_param"]: n.fields for n in artifact.nodes
                if n.fields.get("pim_weight_param")}
    v = next(f for p, f in by_param.items() if "v_proj" in p)
    q = next(f for p, f in by_param.items() if "q_proj" in p)
    assert v.get("weight_sf_multiplier") == subnormal_multiplier(
        layer.self_attn.v_proj.weight.detach().numpy()) > 1
    assert "weight_sf_multiplier" not in q


def _hd128_llama():
    """head_dim=128 的小模型：DQ 组宽 128 与真实 7B 一致，通用写盘器能走通。"""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()


def _written(module, args, model, *, head=False, scales=None):
    import tempfile

    import torch

    from contracts.compile_slots import CompileSlots
    from gml_bridge.ceva import mark_static_quant
    from gml_bridge.export import fuse_for_gml, serialize_gml, write_runtime_files
    from graph.partition import partition_graph

    gm = torch.export.export(module, args, strict=True).module()
    partition_graph(gm)
    if head:
        mark_static_quant(gm)
    report = fuse_for_gml(gm)
    artifact = serialize_gml(gm, report, slots=CompileSlots.from_config(model.config, max_seq=16))
    out = Path(tempfile.mkdtemp())
    if scales is not None:
        scales = scales(artifact)
    return artifact, write_runtime_files(artifact, out, gm=gm, scales=scales), out


def test_prefill_rope_kantor_scale_follows_key_cache_exit() -> None:
    """prefill 的 K 路 RoPE 直接写 key_cache_out 出口：定点化 scale 取 K cache 的倒数。"""
    import numpy as np
    import torch

    from gml_bridge import calib_data
    from gml_bridge.ceva import PrefillLayer

    model = _hd128_llama()
    module = PrefillLayer(model.model.layers[0], torch.randn(1, 8, 1, 128), torch.randn(1, 8, 1, 128))
    artifact, files, out = _written(module, (torch.randn(1, 8, 256), torch.zeros(1, 1, 8, 8)), model)
    rope = next(n for n in artifact.nodes if n.fields.get("op_type") == "Llama2Activation")
    name = rope.fields["Kantor_A_Llama2Activation_add_scale_buffer_file"]
    value = float(np.fromfile(out / name, np.float16)[0])
    assert value == pytest.approx(1 / calib_data.kv_cache_scale(is_key=True), rel=1e-3)


def test_head_static_int8_input_and_fpsu() -> None:
    """输出头：int8 入口的 sf 取 norm 输出标定 scale，FPSU 是定点 mul/shift，权重 int8 per-tensor。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import HeadLayer
    from gml_bridge.runtime_files import fpsu_mul_shift

    from quant.calibration import ActivationScales

    def scales(artifact):
        name = next(n.fields["pim_static_quant_input"] for n in artifact.nodes
                    if n.fields.get("pim_static_quant_input"))
        return ActivationScales(static_input={name: 0.03})

    model = _hd128_llama()
    artifact, files, out = _written(HeadLayer(model), (torch.randn(1, 1, 256),), model,
                                    head=True, scales=scales)
    gemm = next(n for n in artifact.nodes if n.fields.get("op_type") == "Gemm")
    in_sf = float(np.fromfile(out / gemm.fields["input_sf"], np.float16)[0])
    w_sf = np.fromfile(out / gemm.fields["weight_sf"], np.float16)
    assert w_sf.shape == (1,) and in_sf == pytest.approx(0.03, rel=1e-3)
    norm = next(n for n in artifact.nodes if n.fields.get("op_type") == "RMSNorm_vpu")
    assert float(np.fromfile(out / norm.nested["vpu_params"]["output_scale_factor_buffer"],
                             np.float32)[0]) == pytest.approx(0.03)
    mul = np.fromfile(out / gemm.fields["Scaling_buffer_file"], np.int16)[0]
    shift = np.fromfile(out / gemm.fields["Scaling_PS_buffer_file"], np.uint8)[0]
    assert abs(mul / 2.0 ** shift / (in_sf * float(w_sf[0])) - 1) < 1e-3
    weight = np.fromfile(out / gemm.fields["weight_buffer"], np.int8)
    assert weight.size == 128 * 256 and np.abs(weight.astype(int)).max() >= 127


def test_attention_matmul_weight_sf_is_the_cache_scale() -> None:
    """QKᵀ 的权重是 K cache、PV 的是 V cache：weight_sf 取对应 cache 的标定 scale。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import DecodeLayer
    from quant.calibration import ActivationScales

    model = _hd128_llama()
    S, nh, hd, H = 16, 2, 128, 256
    args = (torch.randn(1, 1, H), torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd),
            torch.randn(1, nh, S, hd), torch.zeros(3, 1, nh, 1, dtype=torch.int16),
            torch.zeros(1, 1, 1, S), torch.randn(1, nh, S, hd))
    artifact, files, out = _written(DecodeLayer(model.model.layers[0]), args, model,
                                    scales=lambda a: ActivationScales(key=0.05, value=0.003))
    matmuls = [n for n in artifact.nodes if n.fields.get("op_type") == "MatMul"]
    got = {round(float(np.fromfile(out / n.fields["weight_sf"], np.float16)[0]), 4)
           for n in matmuls}
    assert got == {0.05, 0.003}


def test_prefill_rope_tables_hold_the_graph_constants() -> None:
    """prefill 的 cos/sin 是编译期常量：槽 1 落 sin、槽 2 落 cos，内容取图里的常量（fp16）。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import PrefillLayer

    model = _hd128_llama()
    cos, sin = torch.randn(1, 8, 1, 128), torch.randn(1, 8, 1, 128)
    artifact, files, out = _written(PrefillLayer(model.model.layers[0], cos, sin),
                                    (torch.randn(1, 8, 256), torch.zeros(1, 1, 8, 8)), model)
    ropes = [n for n in artifact.nodes
             if str(n.fields.get("op_type", "")).startswith("Llama2Activation")]
    assert len(ropes) == 2
    for rope in ropes:
        got_sin = np.fromfile(out / rope.fields["input_buffer_1"], np.float16)
        got_cos = np.fromfile(out / rope.fields["input_buffer_2"], np.float16)
        assert got_sin.tobytes() == sin.numpy().astype(np.float16).tobytes()
        assert got_cos.tobytes() == cos.numpy().astype(np.float16).tobytes()


def _vendor_gemm_graph(root: Path, *, out_f=8, in_f=256, multiplier=2, head=False):
    """一张只含一个投影的厂商子图：int 码值按「输入维 × 输出维」，scale 按「输出 × 输入/128」。"""
    import numpy as np

    label = "lm_head_MatMul_qidx2_params_3" if head else "self_attn_q_proj_MatMul_qidx2_params_3"
    extra = "" if head or multiplier == 1 else f"\n    weight_sf_multiplier {multiplier}"
    dtype = "int8" if head else "int4"
    text = f'''graph [
  directed 1
  node [
    id 3
    node_id 3
    label "{label}"
    name "{label}"
    op_type "Gemm"{extra}
    weight_buffer "weight_buffer_3.bin"
    weight_buffer_dtype "{dtype}"
    weight_sf "weight_sf_3.bin"
  ]
]
'''
    rf = root / "parser_output" / "tvmgen_default_nprm_main_0" / "runtime_files"
    rf.mkdir(parents=True)
    (rf / "relay2gml_graph.gml").write_text(text)
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, (out_f, in_f)).astype(np.int8)
    scales = (rng.random(1 if head else (out_f, in_f // 128)) * 0.01 + 0.001).astype(np.float16)
    np.ascontiguousarray(codes.T).tofile(rf / "weight_buffer_3.bin")
    (scales * (1 if head else multiplier)).astype(np.float16).tofile(rf / "weight_sf_3.bin")
    if head:
        expected = codes.astype(np.float32) * np.float32(scales[0])
    else:
        expected = (codes.reshape(out_f, -1, 128).astype(np.float32)
                    * (scales.astype(np.float32) * multiplier / multiplier)[..., None]
                    ).reshape(out_f, in_f)
    return rf, expected


def test_vendor_projection_dequantizes_to_hf_layout(tmp_path) -> None:
    """厂商投影还原成 HF 的「输出 × 输入」浮点：码值转置、scale 除以次正规倍数。"""
    import numpy as np

    from gml_bridge.ceva import vendor_float_weights

    rf, expected = _vendor_gemm_graph(tmp_path, multiplier=2)
    got = vendor_float_weights(rf, {"self_attn_q_proj_MatMul#0": (8, 256)})
    np.testing.assert_array_equal(got["self_attn_q_proj_MatMul#0"], expected)


def test_vendor_head_dequantizes_per_tensor(tmp_path) -> None:
    import numpy as np

    from gml_bridge.ceva import vendor_float_weights

    rf, expected = _vendor_gemm_graph(tmp_path, head=True)
    got = vendor_float_weights(rf, {"lm_head_MatMul#0": (8, 256)})
    np.testing.assert_array_equal(got["lm_head_MatMul#0"], expected)


def test_vendor_weight_shape_mismatch_fails(tmp_path) -> None:
    from gml_bridge.ceva import vendor_float_weights

    rf, _ = _vendor_gemm_graph(tmp_path)
    with pytest.raises(ValueError, match="尺寸"):
        vendor_float_weights(rf, {"self_attn_q_proj_MatMul#0": (16, 256)})


def test_load_vendor_checkpoint_replaces_projections(tmp_path) -> None:
    """厂商 65 图的投影权重装进 HF 模型：7 个投影取反量化值，norm γ 已折进投影故置 1。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import load_vendor_checkpoint

    model = _hd128_llama()
    H, F = model.config.hidden_size, model.config.intermediate_size
    roles = {"q": (H, H), "k": (H, H), "v": (H, H), "o": (H, H),
             "gate": (F, H), "up": (F, H), "down": (H, F)}
    rng = np.random.default_rng(0)
    expected = {}
    for layer in range(model.config.num_hidden_layers):
        rf = tmp_path / "decode" / f"model_layers_{layer}_decode_v2" / "parser_output" \
            / "tvmgen_default_nprm_main_0" / "runtime_files"
        rf.mkdir(parents=True)
        blocks = []
        for i, (role, (o, n)) in enumerate(roles.items(), start=1):
            group = "self_attn" if role in "qkvo" else "mlp"
            label = f"{group}_{role}_proj_MatMul_qidx{i}_params_{i}"
            blocks.append(f'''  node [
    id {i}
    node_id {i}
    label "{label}"
    op_type "Gemm"
    weight_buffer "weight_buffer_{i}.bin"
    weight_buffer_dtype "int4"
    weight_sf "weight_sf_{i}.bin"
  ]''')
            codes = rng.integers(-8, 8, (o, n)).astype(np.int8)
            scales = (rng.random((o, n // 128)) * 0.01 + 0.001).astype(np.float16)
            np.ascontiguousarray(codes.T).tofile(rf / f"weight_buffer_{i}.bin")
            scales.tofile(rf / f"weight_sf_{i}.bin")
            expected[(layer, group, role)] = (
                codes.reshape(o, -1, 128) * scales.astype(np.float32)[..., None]).reshape(o, n)
        (rf / "relay2gml_graph.gml").write_text("graph [\n" + "\n".join(blocks) + "\n]\n")
    load_vendor_checkpoint(model, tmp_path, include_head=False)
    for (layer, group, role), want in expected.items():
        got = getattr(getattr(model.model.layers[layer], group), f"{role}_proj").weight
        np.testing.assert_array_equal(got.detach().numpy(), want)
    for layer in model.model.layers:
        assert torch.all(layer.input_layernorm.weight == 1)
        assert torch.all(layer.post_attention_layernorm.weight == 1)


def test_generated_files_manifest_and_io_sf(tmp_path) -> None:
    """CEVA 写盘走我方写盘器：清单全部 generated、不读厂商 bin；IO_info 的 int8 sf 取标定。"""
    import dataclasses

    import torch

    from contracts.ceva import CevaIoBinding
    from gml_bridge.ceva import generated_io, write_generated_files
    from quant.calibration import ActivationScales

    model = _hd128_llama()
    S, nh, hd, H = 16, 2, 128, 256
    args = (torch.randn(1, 1, H), torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd),
            torch.randn(1, nh, S, hd), torch.zeros(3, 1, nh, 1, dtype=torch.int16),
            torch.zeros(1, 1, 1, S), torch.randn(1, nh, S, hd))
    from gml_bridge.ceva import DecodeLayer
    artifact, _, _ = _written(DecodeLayer(model.model.layers[0]), args, model,
                              scales=lambda a: ActivationScales(key=0.05, value=0.003))
    gm, report = _fused(DecodeLayer(model.model.layers[0]), args)
    scales = ActivationScales(key=0.05, value=0.003)
    out = tmp_path / "g"
    out.mkdir()
    manifest = write_generated_files(artifact, gm, out, scales)
    assert manifest and {f["role"] for f in manifest} == {"generated"}
    assert {f["path"] for f in manifest} == artifact.buffer_names
    io = [CevaIoBinding("input", 3, "nprm_0_i3", None, (1, nh, S, hd), "int8", 0.9),
          CevaIoBinding("input", 0, "nprm_0_i0", None, (1, 1, H), "float16", 1.0),
          CevaIoBinding("output", 2, "out_v", "value_cache_out", (1, nh, S, hd), "int8", 0.9)]
    from gml_bridge.writer import Node
    nodes = [Node(1, {"label": "nprm_0_i3", "is_buffer": 1, "pim_kv_is_key": 1}),
             Node(2, {"label": "nprm_0_i0", "is_buffer": 1}),
             Node(3, {"label": "value_cache_out", "is_buffer": 1, "pim_kv_is_key": 0})]
    fixed = generated_io(io, nodes, scales)
    assert [b.sf for b in fixed] == [0.05, 1.0, 0.003]


def test_bundle_passes_layer_scales_and_head_inputs(tmp_path, monkeypatch) -> None:
    """整包导出：每张图拿到本层 KV 标定；输出头拿到 final norm 输入作为标定样本。"""
    import json

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    calls = []
    monkeypatch.setattr(ceva, "load_ceva_spec",
                        lambda root, stage, layer, slots: (stage, layer))
    monkeypatch.setattr(ceva, "bundle_complete", lambda root, graphs, digest: False)

    def fake_export(model, spec, out_dir, *, use_opcompiler, scales, calibration_inputs):
        calls.append((spec, scales, calibration_inputs))
        return {"stage": spec[0], "layer_id": spec[1], "files": []}

    monkeypatch.setattr(ceva, "export_ceva_graph", fake_export)
    import torch

    h0, h1 = torch.zeros(2), torch.ones(2)
    cal = LlamaCalibration(kv=[(0.1, 0.2), (0.3, 0.4)], head_inputs=[h0, h1])
    ceva.export_ceva_bundle(_small_llama(), tmp_path / "ref", tmp_path / "b",
                            [("decode", 1), ("lm_head", None)], calibration=cal)
    (dec, dec_scales, _), (head, head_scales, head_inputs) = calls
    assert dec_scales.kv(is_key=True) == 0.3 and dec_scales.kv(is_key=False) == 0.4
    assert head_inputs[0] is h0 and head_inputs[1] is h1 and head_scales.key is None
    manifest = json.loads((tmp_path / "b" / "bundle.json").read_text())
    assert manifest["calibration"]["kv"] == [[0.1, 0.2], [0.3, 0.4]]


def test_cli_ceva_loads_vendor_checkpoint(monkeypatch, tmp_path) -> None:
    """CEVA 模式先把厂商反量化权重装进模型，再整包导出（bin 全部重新生成）。"""
    import sys

    from gml_bridge import ceva
    from scripts import export_gml

    order = []
    monkeypatch.setattr(export_gml, "_load_model", lambda layers: _small_llama())
    monkeypatch.setattr(ceva, "load_vendor_checkpoint",
                        lambda model, root, include_head=True: order.append(("ckpt", include_head)))
    monkeypatch.setattr(ceva, "export_ceva_bundle",
                        lambda *a, **k: order.append(("bundle",)) or tmp_path / "bundle.json")
    monkeypatch.setattr(sys, "argv", ["export_gml.py", "--ceva-reference", str(tmp_path),
                                      "--use-opcompiler", "--out-dir", str(tmp_path / "o")])
    assert export_gml.main() == 0
    assert order == [("ckpt", True), ("bundle",)]


def test_calibrate_model_uses_runner_embedding(monkeypatch, tmp_path) -> None:
    """标定跑在 runner 同源的已旋转 embedding 上：先换进模型，再按标定语料前向。"""
    import torch

    from gml_bridge import ceva
    import quant.calibration as calibration

    model = _small_llama()
    table = torch.randn(model.config.vocab_size, model.config.hidden_size)
    path = tmp_path / "embed.pth"
    torch.save({"weight": table}, path)
    monkeypatch.setattr(ceva, "ceva_embedding", lambda: path)

    class Tok:
        def __call__(self, text, return_tensors):
            return type("T", (), {"input_ids": torch.tensor([[1, 2, 3]])})()

    monkeypatch.setattr(ceva, "_calibration_tokenizer", lambda: Tok())
    seen = {}
    monkeypatch.setattr(calibration, "calibrate_llama",
                        lambda m, samples: seen.setdefault("w", m.model.embed_tokens.weight.clone()))
    ceva.calibrate_model(model)
    assert torch.equal(seen["w"], table)


def test_full_pipeline_ceva_passes_token_check(monkeypatch, tmp_path) -> None:
    """全流程强制走 FlagTree，透传对拍档位；我方 bin 默认 warn。"""
    from scripts import run_full_pipeline

    calls = []
    monkeypatch.setattr(run_full_pipeline, "_run", lambda cmd, **k: calls.append(cmd))
    monkeypatch.setattr(run_full_pipeline, "ceva_workspace", lambda: tmp_path)
    monkeypatch.setattr(run_full_pipeline, "ceva_verdict", lambda p: [])
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--backend", "ceva",
                                      "--run-name", "x", "--token-check", "strict"])
    args = run_full_pipeline.parse_args()
    run_full_pipeline.run_ceva(args, tmp_path)
    assert "--use-opcompiler" in calls[0]
    run_all = next(c for c in calls if "run_all.sh" in c)
    assert run_all[run_all.index("--token-check") + 1] == "strict"
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--backend", "ceva", "--run-name", "x"])
    assert run_full_pipeline.parse_args().token_check == "warn"


def test_full_pipeline_ceva_warn_does_not_claim_token_match(monkeypatch, tmp_path, capsys) -> None:
    """告警档通过只说明仿真成功，最终摘要不能宣称 token 一致。"""
    from scripts import run_full_pipeline

    verdict = tmp_path / "verdict.txt"
    verdict.write_text("[WARN] token 与 baseline 不一致  本次 [13, 42]\n结论: PASS\n")
    lines = run_full_pipeline.ceva_verdict(verdict)
    assert len(lines) == 1 and "[WARN]" in lines[0]
    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--backend", "ceva",
                                      "--run-name", "x", "--log-dir", str(tmp_path / "logs")])
    monkeypatch.setattr(run_full_pipeline, "run_ceva", lambda args, log_dir: verdict)
    assert run_full_pipeline.main() == 0
    output = capsys.readouterr().out
    assert "CEVA 全流程通过" in output
    assert "token 与厂商基线一致" not in output
    assert "token 对拍：告警" in output
    verdict.write_text("[PASS] token 与 baseline 一致  本次 [13]\n结论: PASS\n")
    assert run_full_pipeline.main() == 0
    assert "token 对拍：通过" in capsys.readouterr().out


def test_full_pipeline_ceva_requires_decode(monkeypatch, tmp_path) -> None:
    """完整验收至少跑一步 decode，不能只跑 prefill 就报通过。"""
    from scripts import run_full_pipeline

    monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--backend", "ceva",
                                      "--run-name", "x", "--decode-steps", "0",
                                      "--log-dir", str(tmp_path)])
    monkeypatch.setattr(run_full_pipeline, "run_ceva",
                        lambda *a: (_ for _ in ()).throw(AssertionError("不应运行")))
    assert run_full_pipeline.main() == 2


def test_prefill_stage_uses_real_rotary_tables() -> None:
    """prefill 子图的 cos/sin 常量取模型 rotary 的位置 0..T-1（sin 为厂商编码），不能是占位零。"""
    import dataclasses

    import torch

    from contracts.compile_slots import CompileSlots
    from contracts.ceva import CevaGraphSpec
    from gml_bridge.ceva import _stage_module

    model = _small_llama()
    slots = dataclasses.replace(CompileSlots.from_config(model.config, max_seq=16), tokens=8)
    spec = CevaGraphSpec(stage="prefill", layer_id=0, query_tokens=8, slots=slots,
                         version="x", io=(), runtime_dir=Path("."))
    module, _ = _stage_module(model, spec)
    cos, sin = _vendor_rope(model, list(range(8)))
    assert torch.equal(module.cos, cos) and torch.equal(module.sin, sin)


def test_prefill_attention_matmul_weight_sf_traces_the_producer() -> None:
    """prefill 没有 DMA：QKᵀ 的 K、PV 的 V 来自本图的 RoPE 与 v_proj，weight_sf 仍取对应 cache 的 scale。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import PrefillLayer
    from quant.calibration import ActivationScales

    model = _hd128_llama()
    module = PrefillLayer(model.model.layers[0], torch.randn(1, 8, 1, 128), torch.randn(1, 8, 1, 128))
    artifact, files, out = _written(module, (torch.randn(1, 8, 256), torch.zeros(1, 1, 8, 8)), model,
                                    scales=lambda a: ActivationScales(key=0.05, value=0.003))
    matmuls = [n for n in artifact.nodes if n.fields.get("op_type") == "MatMul"]
    got = {round(float(np.fromfile(out / n.fields["weight_sf"], np.float16)[0]), 4)
           for n in matmuls}
    assert got == {0.05, 0.003}


def _fake_bundle_export(monkeypatch):
    """替换单图导出：把收到的 KV scale 写进小文件，保留真实的清单合并与完整性判断。"""
    from gml_bridge import ceva

    monkeypatch.setattr(ceva, "load_ceva_spec",
                        lambda root, stage, layer, slots: (stage, layer))
    monkeypatch.setattr(ceva, "bundle_complete",
                        lambda root, graphs, digest: {(g["stage"], g["layer_id"]) for g in graphs}
                        >= {("prefill", 0), ("decode", 0)})

    def fake_export(model, spec, out_dir, *, use_opcompiler, scales, calibration_inputs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kv.txt").write_text(f"{scales.key},{scales.value}")
        return {"stage": spec[0], "layer_id": spec[1], "files": []}

    monkeypatch.setattr(ceva, "export_ceva_graph", fake_export)


def test_partial_reexport_refuses_mixed_calibration(tmp_path, monkeypatch) -> None:
    """换了标定值后只重导 decode，不能留下用旧 scale 写 cache 的 prefill 还标成完整。"""
    import json

    import pytest

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    _fake_bundle_export(monkeypatch)
    model, out = _small_llama(), tmp_path / "b"
    both = [("prefill", 0), ("decode", 0)]
    ceva.export_ceva_bundle(model, tmp_path, out, both,
                            calibration=LlamaCalibration(kv=[(0.1, 0.2)]))
    assert json.loads((out / "bundle.json").read_text())["complete"] is True
    with pytest.raises(ValueError, match="生成口径"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)],
                                calibration=LlamaCalibration(kv=[(0.3, 0.4)]))
    assert json.loads((out / "bundle.json").read_text())["complete"] is False
    # 口径相同可以局部续导；口径不同但覆盖全部已有图（整包重导）也可以。
    ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)],
                            calibration=LlamaCalibration(kv=[(0.1, 0.2)]))
    ceva.export_ceva_bundle(model, tmp_path, out, both,
                            calibration=LlamaCalibration(kv=[(0.3, 0.4)]))
    manifest = json.loads((out / "bundle.json").read_text())
    assert manifest["complete"] is True and manifest["calibration"]["kv"] == [[0.3, 0.4]]
    assert (out / ceva.bundle_subdir("prefill", 0) / "kv.txt").read_text() == "0.3,0.4"


def test_partial_reexport_refuses_changed_checkpoint(tmp_path, monkeypatch) -> None:
    """标定值相同但 checkpoint 变了，同样不能局部续导。"""
    import pytest
    import torch

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    _fake_bundle_export(monkeypatch)
    model, out = _small_llama(), tmp_path / "b"
    cal = LlamaCalibration(kv=[(0.1, 0.2)])
    ceva.export_ceva_bundle(model, tmp_path, out, [("prefill", 0), ("decode", 0)],
                            calibration=cal)
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(1.0)
    with pytest.raises(ValueError, match="生成口径"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)], calibration=cal)


def test_interrupted_full_reexport_never_marks_mixed_bundle_complete(tmp_path, monkeypatch) -> None:
    """整包重导中途失败后，只重试第一张图不能把新旧口径混合的包标成完整。"""
    import json

    import pytest

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    _fake_bundle_export(monkeypatch)
    model, out = _small_llama(), tmp_path / "b"
    both = [("prefill", 0), ("decode", 0)]
    old = LlamaCalibration(kv=[(0.1, 0.2)])
    new = LlamaCalibration(kv=[(0.3, 0.4)])
    ceva.export_ceva_bundle(model, tmp_path, out, both, calibration=old)

    # 换口径整包重导：第一张图成功后第二张失败。
    calls = {"n": 0}

    def flaky_export(model, spec, out_dir, *, use_opcompiler, scales, calibration_inputs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("第二张图失败")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kv.txt").write_text(f"{scales.key},{scales.value}")
        return {"stage": spec[0], "layer_id": spec[1], "files": []}

    monkeypatch.setattr(ceva, "export_ceva_graph", flaky_export)
    with pytest.raises(RuntimeError, match="第二张图失败"):
        ceva.export_ceva_bundle(model, tmp_path, out, both, calibration=new)
    assert json.loads((out / "bundle.json").read_text())["complete"] is False

    # 只重试第一张图：未重导的第二张仍是旧口径，必须拒绝。
    with pytest.raises(ValueError, match="生成口径"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("prefill", 0)], calibration=new)


def _manifest_case(tmp_path, *, ours_bins, vendor_bins, gml_bins=None, register=None):
    """造一张单节点的参考图与我方包，返回清单路径。`register` 给要登记的文件名（默认全部）。"""
    import hashlib
    import json

    from gml_bridge.ceva import GML_NAME, RUNTIME_DIR, bundle_subdir, graph_dir

    def gml(bins):
        fields = " ".join(f'f{i} "{name}"' for i, name in enumerate(bins))
        return f'graph [ relay2gml_version "1" node [ id 0 label "a_params_0" {fields} ] ]'

    ref = graph_dir(tmp_path / "ref", "decode", 0) / RUNTIME_DIR
    ref.mkdir(parents=True)
    (ref / GML_NAME).write_text(gml(list(vendor_bins)))
    for name, size in vendor_bins.items():
        (ref / name).write_bytes(b"\0" * size)
    out = tmp_path / "bundle"
    d = out / bundle_subdir("decode", 0)
    d.mkdir(parents=True)
    (d / GML_NAME).write_text(gml(gml_bins if gml_bins is not None else list(ours_bins)))
    (d / "IO_info.txt").write_text("io")
    for name, size in ours_bins.items():
        (d / name).write_bytes(b"\1" * size)
    names = register if register is not None else [GML_NAME, "IO_info.txt", *ours_bins]
    files = [{"path": n, "size_bytes": (d / n).stat().st_size, "role": "generated",
              "sha256": hashlib.sha256((d / n).read_bytes()).hexdigest()} for n in names]
    manifest = {"reference_root": str(tmp_path / "ref"), "complete": False,
                "graphs": [{"stage": "decode", "layer_id": 0,
                            "directory": str(bundle_subdir("decode", 0)), "files": files}]}
    (out / "bundle.json").write_text(json.dumps(manifest))
    return out / "bundle.json"


def test_ceva_manifest_checks_references_and_reference_sizes(tmp_path) -> None:
    """清单自洽还不够：要核对 GML 引用集、落盘集与登记集，并按参考图同字段文件核尺寸。"""
    import json

    from gml_bridge.ceva import GML_NAME
    from scripts.verify_gml_artifact import Report, check_ceva_manifest

    def failed(path):
        report = Report()
        check_ceva_manifest(path, report)
        return bool(report.failed)

    cases = {
        "ok": dict(ours_bins={"w_0.bin": 4}, vendor_bins={"v_0.bin": 4}),
        "引用的 bin 不存在": dict(ours_bins={}, vendor_bins={"v_0.bin": 4},
                             gml_bins=["missing.bin"], register=[GML_NAME, "IO_info.txt"]),
        "漏登记": dict(ours_bins={"w_0.bin": 4}, vendor_bins={"v_0.bin": 4},
                    register=[GML_NAME, "IO_info.txt"]),
        "GML 没登记": dict(ours_bins={"w_0.bin": 4}, vendor_bins={"v_0.bin": 4},
                         register=["IO_info.txt", "w_0.bin"]),
        "尺寸与参考不符": dict(ours_bins={"w_0.bin": 2}, vendor_bins={"v_0.bin": 4}),
    }
    for i, (name, case) in enumerate(cases.items()):
        assert failed(_manifest_case(tmp_path / str(i), **case)) == (name != "ok"), name
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"graphs": []}))
    assert failed(empty)
    # 标成完整的包必须覆盖参考目录的全部子图。
    path = _manifest_case(tmp_path / "c", ours_bins={"w_0.bin": 4}, vendor_bins={"v_0.bin": 4})
    manifest = json.loads(path.read_text())
    (tmp_path / "c" / "ref" / "llama_lm_head").mkdir()
    (path).write_text(json.dumps(dict(manifest, complete=True)))
    assert failed(path)


def test_ceva_manifest_rejects_mixed_generation_digest(tmp_path) -> None:
    """标成完整的包，每张图记录的生成摘要必须与包级摘要一致。"""
    import json

    from scripts.verify_gml_artifact import Report, check_ceva_manifest

    path = _manifest_case(tmp_path, ours_bins={"w_0.bin": 4}, vendor_bins={"v_0.bin": 4})
    manifest = json.loads(path.read_text())
    manifest.update(complete=True, generation_digest="new")
    manifest["graphs"][0]["generation_digest"] = "old"
    path.write_text(json.dumps(manifest))
    report = Report()
    check_ceva_manifest(path, report)
    assert report.failed

    manifest["graphs"][0]["generation_digest"] = "new"
    path.write_text(json.dumps(manifest))
    report = Report()
    check_ceva_manifest(path, report)
    assert not report.failed, report.failed

    # 没有包级摘要时口径未知，同样不能算完整。
    del manifest["generation_digest"]
    path.write_text(json.dumps(manifest))
    report = Report()
    check_ceva_manifest(path, report)
    assert report.failed


def test_generated_runtime_buffers_are_zero_but_static_files_are_not(tmp_path) -> None:
    """动态缓冲（中间激活、相位缓冲、KV 权重通路）按尺寸写零；权重、LUT、定标与 prefill 常量表不动。"""
    import numpy as np
    import torch

    from gml_bridge.ceva import (DecodeLayer, PrefillLayer, runtime_buffer_names,
                                 write_generated_files)
    from quant.calibration import ActivationScales

    model = _hd128_llama()
    S, nh, hd, H = 16, 2, 128, 256
    cos, sin = torch.randn(1, 8, 1, hd), torch.randn(1, 8, 1, hd)
    cases = {
        "decode": (DecodeLayer(model.model.layers[0]),
                   (torch.randn(1, 1, H), torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd),
                    torch.randn(1, nh, S, hd), torch.zeros(3, 1, nh, 1, dtype=torch.int16),
                    torch.zeros(1, 1, 1, S), torch.randn(1, nh, S, hd))),
        "prefill": (PrefillLayer(model.model.layers[0], cos, sin),
                    (torch.randn(1, 8, H), torch.zeros(1, 1, 8, 8))),
    }
    scales = ActivationScales(key=0.05, value=0.003)
    for stage, (module, args) in cases.items():
        artifact, _, _ = _written(module, args, model, scales=lambda a: scales)
        gm, _ = _fused(module, args)
        out = tmp_path / stage
        out.mkdir()
        manifest = write_generated_files(artifact, gm, out, scales)
        sizes = {f["path"]: f["size_bytes"] for f in manifest}
        runtime = runtime_buffer_names(artifact.nodes)
        static = set(sizes) - runtime
        assert any(n.startswith("input_buffer_phase_") for n in runtime), stage
        assert any(n.startswith("weight_buffer_") for n in runtime), stage
        for name in runtime:
            data = (out / name).read_bytes()
            assert len(data) == sizes[name] and not any(data), (stage, name)
        for prefix in ("weight_buffer_", "weight_sf_", "LUT_phase_", "activation_lut_file_"):
            assert any((out / n).read_bytes().strip(b"\0") for n in static
                       if n.startswith(prefix)), (stage, prefix)

        # DQ 的输出 scale 与相位 Kantor scale 是逐组动态值，必须写零。
        dq_nodes = [n for n in artifact.nodes if n.fields.get("op_type") in
                    ("DynamicScaling", "Llama2ActivationDQ")]
        assert dq_nodes, stage
        for node in dq_nodes:
            for key in ("output_sf", "kantor_A_scale_buffer_file_phase_3"):
                name = node.fields[key]
                assert name in runtime, (stage, key)
                assert not any((out / name).read_bytes()), (stage, key)
            # 固定量化常量 1/256 仍要按标定口径写出。
            assert np.fromfile(out / node.fields["Scaling_buffer_phase_1"],
                               np.float16)[0] > 0, stage
        # Softmax 的两处运行时落点写零；phase1 的固定 0.5 常量保留。
        softmax_nodes = [n for n in artifact.nodes if n.fields.get("op_type") == "Softmax"]
        assert softmax_nodes, stage
        for node in softmax_nodes:
            for key in ("Bias_buffer_phase_1", "Scaling_buffer_phase_4"):
                name = node.fields[key]
                assert name in runtime, (stage, key)
                assert not any((out / name).read_bytes()), (stage, key)
            assert float(np.fromfile(out / node.fields["Scaling_buffer_phase_1"],
                                     np.float16)[0]) == 0.5, stage
        # KV cache 的标定 scale 是静态参数，不能误清零。
        dmas = [n for n in artifact.nodes if n.fields.get("op_type") == "KV_Cache_DMA"]
        for node in dmas:
            for key in ("input_sf", "output_sf", "updates_sf"):
                name = node.fields[key]
                assert name not in runtime and any((out / name).read_bytes()), (stage, key)
    ropes = [n for n in artifact.nodes if n.fields.get("pim_rope_const")]
    assert ropes
    for rope in ropes:
        got_cos = np.fromfile(tmp_path / "prefill" / rope.fields["input_buffer_2"], np.float16)
        assert got_cos.tobytes() == cos.numpy().astype(np.float16).tobytes()


def _rename_v_projection(layer) -> None:
    """把 `v_proj` 注册名改成 `val_lin`，前向仍用同一个模块。"""
    attn = layer.self_attn
    module = attn.v_proj
    del attn.v_proj
    attn.val_lin = module
    object.__setattr__(attn, "v_proj", module)


@pytest.mark.parametrize("stage", ["prefill", "decode"])
def test_value_projection_found_by_structure_not_name(stage) -> None:
    """V 投影改名后仍输出 int8 并带 kantor 字段，GML 除展示名外与改名前逐字节相同。"""
    import torch

    from gml_bridge.ceva import DecodeLayer, PrefillLayer
    from gml_bridge.export import serialize_gml

    def build(rename: bool):
        model = _small_llama()
        layer = model.model.layers[0]
        if rename:
            _rename_v_projection(layer)
        if stage == "prefill":
            T, hd = 8, 16
            module = PrefillLayer(layer, torch.randn(1, T, 1, hd), torch.randn(1, T, 1, hd))
            args = (torch.zeros(1, T, 64), torch.zeros(1, 1, T, T))
        else:
            module, args = DecodeLayer(layer), _decode_args()
        gm, report = _fused(module, args)
        return serialize_gml(gm, report)

    plain, renamed = build(False), build(True)
    params = {str(n.fields.get("pim_weight_param")) for n in renamed.nodes}
    assert any("val_lin" in p for p in params) and not any("v_proj" in p for p in params)
    v = [n.fields for n in renamed.nodes
         if n.fields.get("op_type") == "Gemm" and "val_lin" in str(n.fields.get("pim_weight_param"))]
    assert len(v) == 1
    assert v[0]["output_buffer_dtype"] == "int8"
    assert "kantor_A_scale_buffer_file" in v[0]
    # label/name 是按参数名起的展示名，不进数值口径；其余字段应逐字节相同。
    def body(text):
        return [l for l in text.splitlines() if l.split()[:1] not in (["label"], ["name"])]
    assert body(renamed.text) == body(plain.text)


def test_compiler_revision_tracks_uncommitted_source(tmp_path) -> None:
    """编译版本随生成代码内容变化：同一 HEAD 下改了 gml_bridge 的源码，摘要也要变。"""
    from gml_bridge import ceva

    root = tmp_path / "repo"
    (root / "gml_bridge").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "gml_bridge" / "a.py").write_text("x = 1\n")
    (root / "docs" / "note.md").write_text("n")
    first = ceva.compiler_revision(root, head="abc1234")
    (root / "docs" / "note.md").write_text("changed")
    assert ceva.compiler_revision(root, head="abc1234") == first
    (root / "gml_bridge" / "a.py").write_text("x = 2\n")
    second = ceva.compiler_revision(root, head="abc1234")
    assert second != first and second.startswith("abc1234+")


def test_partial_reexport_refuses_changed_compiler_source(tmp_path, monkeypatch) -> None:
    """HEAD 不变但生成代码改了，局部重导要拒绝。"""
    import pytest

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    _fake_bundle_export(monkeypatch)
    model, out = _small_llama(), tmp_path / "b"
    cal = LlamaCalibration(kv=[(0.1, 0.2)])
    monkeypatch.setattr(ceva, "compiler_revision", lambda *a, **k: "bf5b8a5+0000")
    ceva.export_ceva_bundle(model, tmp_path, out, [("prefill", 0), ("decode", 0)],
                            calibration=cal)
    monkeypatch.setattr(ceva, "compiler_revision", lambda *a, **k: "bf5b8a5+1111")
    with pytest.raises(ValueError, match="生成口径"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)], calibration=cal)


def test_value_writer_may_be_the_gemm_itself() -> None:
    """V 出口就是投影本身（没有布局算子）时也认作 V 投影。"""
    from gml_bridge.from_fx import _feeds_value_cache

    class Node:
        users: list = []

    gemm = Node()
    assert _feeds_value_cache(gemm, [(gemm, False)])
    assert not _feeds_value_cache(gemm, [(gemm, True)])


def _flagtree_tree(root):
    """造一棵必需的四个源码目录都非空的 FlagTree 源码树。"""
    files = {
        "include/Ops.h": "h1",
        "lib/Ops.cpp": "v1",
        "python/src/x.py": "v1",
        "python/triton/y.py": "v1",
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_opcompiler_revision_tracks_flagtree_source(tmp_path, monkeypatch) -> None:
    """算子编译器版本随 FlagTree 源码变化；读不到源码树时为 unknown。"""
    import genesim_bridge.paths as paths
    from gml_bridge import ceva

    _flagtree_tree(tmp_path)
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "t.mlir").write_text("noise")
    monkeypatch.setattr(paths, "flagtree_source", lambda required=True: tmp_path)
    first = ceva.opcompiler_revision()
    (tmp_path / "test" / "t.mlir").write_text("other noise")
    assert ceva.opcompiler_revision() == first
    (tmp_path / "lib" / "Ops.cpp").write_text("v2")
    assert ceva.opcompiler_revision() != first
    monkeypatch.setattr(paths, "flagtree_source", lambda required=True: None)
    assert ceva.opcompiler_revision() == ceva.OPCOMPILER_UNKNOWN


def test_opcompiler_revision_is_unknown_for_incomplete_source(tmp_path, monkeypatch) -> None:
    """空树、缺必需子目录或只有无关文件时都不能给出有效版本号。"""
    import genesim_bridge.paths as paths
    from gml_bridge import ceva

    monkeypatch.setattr(paths, "flagtree_source", lambda required=True: tmp_path)
    assert ceva.opcompiler_revision() == ceva.OPCOMPILER_UNKNOWN

    _flagtree_tree(tmp_path)
    (tmp_path / "python" / "triton" / "y.py").unlink()
    assert ceva.opcompiler_revision() == ceva.OPCOMPILER_UNKNOWN

    _flagtree_tree(tmp_path)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            path.unlink()
    (tmp_path / "lib" / "noise.mlir").write_text("noise")
    assert ceva.opcompiler_revision() == ceva.OPCOMPILER_UNKNOWN


def test_partial_reexport_refuses_changed_opcompiler(tmp_path, monkeypatch) -> None:
    """HEAD 与本仓源码都不变，只换 FlagTree：局部重导要拒绝；版本未知同样拒绝。"""
    import pytest

    from gml_bridge import ceva
    from quant.calibration import LlamaCalibration

    _fake_bundle_export(monkeypatch)
    model, out = _small_llama(), tmp_path / "b"
    cal = LlamaCalibration(kv=[(0.1, 0.2)])
    monkeypatch.setattr(ceva, "opcompiler_revision", lambda: "ft-aaa")
    ceva.export_ceva_bundle(model, tmp_path, out, [("prefill", 0), ("decode", 0)],
                            calibration=cal)
    monkeypatch.setattr(ceva, "opcompiler_revision", lambda: "ft-bbb")
    with pytest.raises(ValueError, match="生成口径"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)], calibration=cal)
    monkeypatch.setattr(ceva, "opcompiler_revision", lambda: ceva.OPCOMPILER_UNKNOWN)
    with pytest.raises(ValueError, match="算子编译器版本"):
        ceva.export_ceva_bundle(model, tmp_path, out, [("decode", 0)], calibration=cal)
