"""验证 GML 导出入口。

这是第三轮的对外接口：模型进去，GML 文本与缓冲区清单出来。缓冲区清单尤其要紧——
第四轮按它写 `.bin`，两侧靠它保持一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
from gml_bridge.export import (
    GML_VERSION,
    export_llama2,
    format_summary,
    write_artifact,
    write_runtime_files,
)
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    parse_blocks,
)

_SEQ_LEN = 16


@pytest.fixture(scope="module")
def graph_and_artifact():
    """导出图与产物一起返回：写盘要用 gm 取 f32 权重。"""
    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = _model()
    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(
        model, _SEQ_LEN, position_ids, dtype=torch.float32)
    from gml_bridge.export import export_graph

    return gm, export_graph(gm)


def _model():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=64,
            intermediate_size=176,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=_SEQ_LEN,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()


@pytest.fixture(scope="module")
def artifact():
    torch.manual_seed(0)
    return export_llama2(_model(), seq_len=_SEQ_LEN, dtype=torch.float32)


def test_two_layer_model_yields_a_non_trivial_graph(artifact) -> None:
    assert len(artifact.nodes) > 40
    assert len(artifact.edges) > 40
    # 融合数不作断言：llama2 的 rsqrt 与 silu 是独立节点（`RMSNorm_vpu` / `Silu`），
    # 不参与融合，所以这张图可能一处都不折——那是正确状态，见文档 18.6。
    assert artifact.fusions >= 0


def test_output_satisfies_every_structure_rule(artifact) -> None:
    nodes = parse_blocks(artifact.text, "node")
    edges = parse_blocks(artifact.text, "edge")
    assert check_rule1_fusion(nodes) == []
    assert check_rule2_buffer_naming(nodes, edges) == []
    assert check_rule3_shape_on_edges(nodes, edges) == []
    assert check_rule4_edge_direction(nodes, edges) == []
    assert check_rule5_absent_fields(nodes) == []


def test_version_is_declared(artifact) -> None:
    assert f'relay2gml_version "{GML_VERSION}"' in artifact.text


def test_buffer_names_follow_the_naming_rules(artifact) -> None:
    """清单里的每个名字都要能被命名规则生成——这是跨语言一致性的凭据。"""
    assert artifact.buffer_names

    allowed = set()
    for node_id in range(1, len(artifact.nodes) + 10):
        allowed |= {
            names.data_buffer(node_id),
            names.scale(node_id),
            names.weight_buffer(node_id),
            names.weight_scale(node_id),
            names.activation_lut(node_id),
            names.fpsu_scale(node_id),
            names.fpsu_post_shift(node_id),
            # RMSNorm 走向量单元：eps 常量 + 输出 requant scale
            # （后者只在 vpu_params 子块里出现，顶层没有）。
            names.rms_norm_epsilon(node_id),
            names.output_scale(node_id),
            names.output_zero_point(node_id),
            names.weight_zero_point(node_id),
            names.zero_point(node_id),
            names.fpsu_bias(node_id),
            names.phase_output_buffer_self(node_id),
            names.kantor_scale(node_id),
            names.kantor_bias(node_id),
            names.kantor_shift(node_id),
            names.kantor_scale(node_id, "B"),
            names.kantor_bias(node_id, "B"),
            names.kantor_shift(node_id, "B"),
        }
        # FPSU 三族与量化零点在逐元素算子上按槽出现。
        for slot in range(4):
            allowed |= {
                names.fpsu_scale(node_id, slot),
                names.fpsu_post_shift(node_id, slot),
                names.fpsu_bias(node_id, slot),
                names.zero_point(node_id, slot),
            }
        # DQ 的 4 相族。
        for phase in range(5):
            allowed |= {
                names.phase_input_buffer(node_id, phase),
                names.phase_output_buffer(node_id, phase),
                names.phase_fpsu_scale(node_id, phase),
                names.phase_fpsu_post_shift(node_id, phase),
                names.phase_fpsu_bias(node_id, phase),
                names.phase_lut(node_id, phase),
                names.phase_kantor_scale(node_id, phase),
                names.phase_kantor_bias(node_id, phase),
                names.phase_kantor_shift(node_id, phase),
            }
        # RoPE 子块：6 个单元 × 定标/零点/Kantor + 中间态。
        from contracts.gml_hw_table import ROPE_UNITS, ROPE_SCALE_BLOCKS, ROPE_KANTOR_BLOCKS
        for unit, block in ROPE_UNITS:
            allowed |= {
                names.rope_scale(node_id, unit, block),
                names.rope_post_shift(node_id, unit, block),
                names.rope_bias(node_id, unit, block),
            }
        for block in ROPE_SCALE_BLOCKS:
            allowed |= {
                names.rope_quant_scale(node_id, block),
                names.rope_quant_zero_point(node_id, block),
            }
        for block in ROPE_KANTOR_BLOCKS:
            for side in ("A", "B"):
                allowed |= {
                    names.rope_kantor_bias(node_id, block, side),
                    names.rope_kantor_shift(node_id, block, side),
                    names.rope_kantor_scale(node_id, block, side),
                }
        allowed |= {
            names.rope_kantor_bias(node_id, "Llama2Activation_add", "A"),
            names.rope_kantor_scale(node_id, "Llama2Activation_add", "A"),
            names.rope_kantor_shift(node_id, "Llama2Activation_add", "A"),
            names.rope_intermediate(node_id, "cos"),
            names.rope_intermediate(node_id, "sin"),
            names.kv_updates_scale(node_id),
            names.kv_updates_zero_point(node_id),
        }
        for slot in range(4):
            allowed |= {names.data_buffer(node_id, slot), names.scale(node_id, slot)}

    unexpected = artifact.buffer_names - allowed
    assert unexpected == set(), f"这些名字不符合命名规则: {sorted(unexpected)[:5]}"


def test_every_referenced_buffer_ends_with_the_suffix(artifact) -> None:
    assert all(name.endswith(names.SUFFIX) for name in artifact.buffer_names)


def test_write_artifact_uses_the_expected_filename(artifact, tmp_path) -> None:
    """文件名必须与参考产物一致——对方的流程按这个名字找图。"""
    path = write_artifact(artifact, tmp_path / "out")
    assert path.name == "relay2gml_graph.gml"
    assert path.read_text() == artifact.text


def test_summary_reports_the_shape_of_the_graph(artifact) -> None:
    summary = format_summary(artifact)
    assert "节点" in summary
    assert "Gemm" in summary


def test_llama2_specific_nodes_are_emitted(artifact) -> None:
    """RMSNorm 与 SiLU 要作为独立节点产出，与 llama2 实物一致。

    它们曾被误放进融合表，结果被折进主算子、产物里一个都没有。实物显示
    `RMSNorm_vpu` 绑定在向量单元上、`Silu` 自带 nmu_mode 与 kantor_mode，
    两者都是主算子而非激活。
    """
    assert 'op_type "RMSNorm_vpu"' in artifact.text
    assert 'op_type "Silu"' in artifact.text

def test_gml_and_runtime_files_agree(graph_and_artifact, tmp_path) -> None:
    """端到端闭环：GML 引用的每个名字都要在磁盘上真实存在，反之亦然。

    这是架构 C 的那道防线（文档第 10、28 节）。名字由 FlagTree（C++）写进 GML，
    文件由这里（Python）写——两侧不一致就是悬空引用，而这不会在我们这侧报错，
    要到对方的解析器才炸。所以必须在产出的同一处拦住。

    `write_runtime_files` 内部就调用 `verify_against_graph`，所以它不抛就说明
    两个集合相等；这里再显式比一次，避免将来把校验从实现里移走时测试失去意义。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "out"
    write_artifact(artifact, out_dir)
    files = write_runtime_files(artifact, out_dir, gm=gm)

    assert files.names_written == artifact.buffer_names
    assert files.total_bytes > 0

    # 每个名字对应的文件真的落盘了。
    for name in artifact.buffer_names:
        assert (out_dir / name).is_file(), f"{name} 没有落盘"


def test_data_buffer_sizes_follow_the_edge_shapes(graph_and_artifact, tmp_path) -> None:
    """数据缓冲区的尺寸由边上的形状决定——形状只在边上（规则 3）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "out"
    write_artifact(artifact, out_dir)
    write_runtime_files(artifact, out_dir, gm=gm)

    # 找一条边，核对它目标节点的输入缓冲区大小。
    by_target = {edge.target: edge.dims for edge in artifact.edges}
    checked = 0
    for node in artifact.nodes:
        dims = by_target.get(node.node_id)
        buffer_name = node.fields.get("input_buffer")
        if not dims or not isinstance(buffer_name, str) or dims == "unknown":
            continue
        # 上游是 phase 型 DQ 时，这一路引用的是**生产者**的文件
        # （`output_buffer_<DQ>.bin`），不由本节点的入边形状决定尺寸 ——
        # 那份是 DQ 量化后的完整输出。跳过（另有 test_quant_pass 逐字节核对）。
        if buffer_name.startswith(("output_buffer", "weight_buffer")):
            continue

        expected = 1
        for part in dims.split("x"):
            expected *= int(part)
        # 字节宽度跟着**声明的** dtype：DQ 吃 fp16（上游还没量化），
        # 其余吃 int8。一律按 int8 算会在 fp16 那几个节点上差一倍。
        if node.fields.get("input_buffer_dtype") == "float16":
            expected *= 2
        assert (out_dir / buffer_name).stat().st_size == expected
        checked += 1
    assert checked > 0, "应当有可核对尺寸的数据缓冲区"

def test_weights_are_quantized_and_written(graph_and_artifact, tmp_path) -> None:
    """f32 权重要量化成 int4 + per-group scale 一起落盘。

    这是第四轮的收口：GML 里的 `weight_buffer` 引用必须对应磁盘上真实的量化权重，
    尺寸符合「一字节一个 int4」与「每 128 个共享一个 fp16 scale」的布局。
    """
    import numpy as np

    from contracts import gml_names as names
    from contracts.gml_quant import INT4_MAX, INT4_MIN, WEIGHT_GROUP_SIZE

    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "out"
    write_artifact(artifact, out_dir)
    write_runtime_files(artifact, out_dir, gm=gm)

    assert artifact.weight_params, "图里应当有带权重的算子"

    # 权重分两类，判据完全不同（实测参考产物 73 个权重里 int4 只有 7 个）：
    #   int4 per-group  值域 [-8,7]，每 128 个共享一个 **fp16** scale
    #   int8 per-tensor 值域 [-128,127]，整张张量一个 **fp32** scale
    # RMSNorm 的一维缩放张量走后者，所以不能统一按 int4 断言。
    by_id = {node.node_id: node for node in artifact.nodes}

    for node_id in artifact.weight_params:
        weight_path = out_dir / names.weight_buffer(node_id)
        scale_path = out_dir / names.weight_scale(node_id)
        assert weight_path.is_file()
        assert scale_path.is_file()

        weights = np.fromfile(weight_path, dtype=np.int8)
        fields = by_id[node_id].fields

        if fields.get("weight_sf_dtype") == "float32":
            scales = np.fromfile(scale_path, dtype=np.float32)
            assert scales.size == 1, "per-tensor 只有一个 scale"
            assert weights.min() >= -128 and weights.max() <= 127
        else:
            scales = np.fromfile(scale_path, dtype=np.float16)
            # 一字节一个 int4，值域严格。
            assert weights.min() >= INT4_MIN
            assert weights.max() <= INT4_MAX
            # 每组一个 scale。
            assert weights.size == scales.size * WEIGHT_GROUP_SIZE

        # scale 不能有 nan——全零组会踩到除零。
        assert not np.isnan(scales).any()
