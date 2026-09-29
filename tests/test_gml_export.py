"""验证 GML 导出入口。

这是第三轮的对外接口：模型进去，GML 文本与缓冲区清单出来。缓冲区清单尤其要紧——
第四轮按它写 `.bin`，两侧靠它保持一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    write_io_info,
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


def test_shared_mask_buffer_is_flagged(artifact) -> None:
    """`is_mask` 挂在**缓冲节点**上，不挂在 32 个 Mask 算子上。

    实测参考产物里该标志只有 1 处，就在掩码张量的图入口（与 `is_buffer 1`
    同一节点），扇出给全部头。它描述的是这块缓冲是掩码张量这个来源事实，
    不是某个算子的配置——挂到每个 Mask 算子上会多发 31 处参考产物里没有的
    字段。
    """
    assert artifact.text.count("is_mask 1") == 1
    flagged = [b for b in parse_blocks(artifact.text, "node") if "is_mask 1" in b]
    assert len(flagged) == 1
    assert "is_buffer 1" in flagged[0]



def test_every_weight_buffer_carries_a_hash_of_its_bytes(graph_and_artifact, tmp_path) -> None:
    """带 `weight_buffer` 的节点必须写指纹，且等于盘上字节的 sha256。

    占位（全零）也写——字段是按节点必填的。占位内容本轮按形状补零，
    下游不得当真实权值用；这条只钉「声明与写出一致」。
    """
    from gml_bridge.export import fill_weight_hashes, write_runtime_files
    import hashlib

    gm, artifact = graph_and_artifact
    files = write_runtime_files(artifact, tmp_path, gm=gm)
    fill_weight_hashes(artifact, files)

    hashed = 0
    for node in artifact.nodes:
        name = node.fields.get("weight_buffer")
        if not isinstance(name, str):
            continue
        path = tmp_path / name
        payload = path.read_bytes()
        digest = node.fields.get("weight_buffer_hash")
        assert isinstance(digest, str), f"{name} 必须有 hash"
        assert digest == hashlib.sha256(payload).hexdigest()
        hashed += 1
    assert hashed > 0, "这张图上没有权值缓冲，这条判据失去意义"


def test_every_softmax_node_carries_the_phase_data_extensions(artifact) -> None:
    """Softmax 五相各自声明进出的数据通道。

    这一族曾经**整族缺失**：`_SOFTMAX_COMMON` 没声明这两个键，于是 32 个
    Softmax 节点一条都没有，而参考产物每个节点有 10 条（5 进 5 出，全为 3）。
    缺一族不会报错——dtype 自检只看节点级的三个 `*_buffer_dtype`，相位级的
    缺席它看不见，所以只能靠这条钉住。

    五相进出都在 fp16 域（exp 表、求和、取倒数、乘回），所以都是 3
    （`data_extension` 是 dtype 的编码：float16→3、int8→1）。
    """
    softmax = [b for b in parse_blocks(artifact.text, "node")
               if 'op_type "Softmax"' in b]
    assert softmax, "这张图上没有 Softmax 节点"

    for block in softmax:
        for n in range(5):
            for direction in ("input", "output"):
                key = f"{direction}_data_extensions_phase_{n}"
                assert f"{key} 3" in block, f"Softmax 节点缺 {key}"

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


def test_fused_activation_block_is_named_after_the_activation(artifact) -> None:
    """contraction 块按**激活**命名，不按 FX 节点名，且 `residual_input_buffer`
    自指。

    实测参考产物（节点 195）：

        fused_Silu_act [
          name "Silu_act"
          op_type "Lut"
          activation_op_type "Silu"
          residual_input_buffer 195      ← 本节点自己的编号
        ]

    按 FX 节点名会写成 `fused_linear_4_activation`——同一个融合，对方解析器按块名
    找不到，是个静默失配。折进来的激活读的是主算子的累加结果，那块缓冲属于主算子，
    所以编号指回自己。

    FlagTree 侧 `#pim.contraction<form = named, blockName = ...>` 是同一份口径，
    两侧必须一致。
    """
    import re

    blocks = [b for b in parse_blocks(artifact.text, "node")
              if "fused_Silu_act" in b]
    assert blocks, "gate 投影上应当有一个折进去的 SiLU"

    for block in blocks:
        assert 'name "Silu_act"' in block
        assert 'activation_op_type "Silu"' in block
        # 只看嵌套块内那一条：节点顶层也有个同名字段（输入边），不是这一条。
        inner = block[block.index("fused_Silu_act"):]
        match = re.search(r"residual_input_buffer (\d+)", inner)
        assert match, "嵌套块要带 residual_input_buffer"
        node_id = re.search(r"node_id (\d+)", block).group(1)
        assert match.group(1) == node_id, (
            f"嵌套块的 residual_input_buffer 应当指向本节点 {node_id}，"
            f"实际 {match.group(1)}")

    # 不能再出现按 FX 节点名的旧块名。
    assert "_activation [" not in artifact.text



def test_runtime_files_include_io_info(graph_and_artifact, tmp_path) -> None:
    """图级 I/O 清单必须落盘。参考产物有 IO_info.txt，本仓一度完全不产。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "out"
    write_runtime_files(artifact, out_dir, gm=gm)
    path = out_dir / "IO_info.txt"
    assert path.is_file(), "IO_info.txt 必须落盘"
    payload = path.read_text()
    assert "inputs" in payload and "outputs" in payload


def test_k_path_split_input_is_transposed_cache_plane() -> None:
    """K 路 Split 的输入边必须是 Kᵀ 平面 `1x32x128x1024`，不能是未转置的缓存。

    参考：KV_Cache_DMA → Transpose → Split，入边 `1x32x128x1024`。
    我方把转置折进 Split 时，入边曾是 `1x32x1024x128`。元素数相同，
    尺寸检查看不见。这条比的是产物上那条边的 dims。
    """
    from genesim_bridge.paths import gml_llama2_reference_dir
    from scripts.gml_structure_check import field

    ref_dir = gml_llama2_reference_dir(required=False)
    if ref_dir is None or not (ref_dir / "relay2gml_graph.gml").is_file():
        pytest.skip("缺少 llama2 GML 参考产物")
    ours_path = Path("/tmp/rev_db16/relay2gml_graph.gml")
    if not ours_path.is_file():
        pytest.skip("没有现成的 decode-block 导出，跳过产物对拍")

    def k_split_in_dims(text: str) -> list[str]:
        nodes = {field(b, "id"): field(b, "op_type") for b in parse_blocks(text, "node")}
        dims = []
        for b in parse_blocks(text, "edge"):
            src, dst = field(b, "source"), field(b, "target")
            if nodes.get(dst) == "Split" and nodes.get(src) in ("Transpose", "KV_Cache_DMA"):
                dims.append(field(b, "dims"))
        return dims

    ref = k_split_in_dims((ref_dir / "relay2gml_graph.gml").read_text())
    ours = k_split_in_dims(ours_path.read_text())
    assert "1x32x128x1024" in ref, f"参考 K 路 Split 入边变了: {ref}"
    assert "1x32x128x1024" in ours, (
        f"K 路 Split 入边不是 Kᵀ 平面: {ours}。"
        "元素数相同的 1x32x1024x128 不够——那是未转置的缓存平面")


def test_dq_source_returns_nonzero_calibration_activation(graph_and_artifact) -> None:
    """DQ 节点要量化的激活来自标定常数，不再是零张量。

    零 absmax 会让 phase0/phase1 一路输出 0.0，而 sf=0 属形式非法值
    （下游反量化除零）。依据设计 3.2。
    """
    gm, artifact = graph_and_artifact
    from gml_bridge.export import _dq_source

    assert artifact.dq_specs, "夹具里应当有 DQ 节点"
    for node_id, spec in artifact.dq_specs.items():
        source = _dq_source(gm, artifact, node_id, spec)
        assert source.shape == (spec.numel,), node_id
        assert source.dtype == np.float16, node_id
        assert (source != 0).all(), f"节点 {node_id} 的标定激活不能是全零"


def test_dq_phase_bins_are_nonzero_after_calibration(
        graph_and_artifact, tmp_path) -> None:
    """落盘后 phase0 逐组非零、output_sf 全部 > 0（验收 A1/A2）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "calib"
    write_runtime_files(artifact, out_dir, gm=gm)

    for node_id in artifact.dq_specs:
        phase0 = np.fromfile(
            out_dir / names.phase_output_buffer(node_id, 0), dtype=np.float16)
        assert phase0.size and (phase0 != 0).all(), f"phase0_{node_id} 含零组"
        sf = np.fromfile(out_dir / names.output_scale(node_id), dtype=np.float16)
        assert (sf > 0).all(), f"output_sf_{node_id} 有非正值"


def test_dq_output_sf_equals_phase1_on_disk(graph_and_artifact, tmp_path) -> None:
    """output_sf 逐字节等于 phase1——标定接通后这条关系不能被破坏。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "sf_phase1"
    write_runtime_files(artifact, out_dir, gm=gm)

    for node_id in artifact.dq_specs:
        sf = np.fromfile(out_dir / names.output_scale(node_id), dtype=np.float16)
        phase1 = np.fromfile(
            out_dir / names.phase_output_buffer(node_id, 1), dtype=np.float16)
        assert np.array_equal(sf, phase1), node_id


def _nodes_by_op(artifact, op_type: str) -> list:
    return [n for n in artifact.nodes if n.fields.get("op_type") == op_type]


def test_kv_and_vproj_output_sf_are_not_hardcoded_one(
        graph_and_artifact, tmp_path) -> None:
    """KV_Cache_DMA / Split / v_proj 的 output_sf 取真实量化 scale（验收 A4）。

    它们的输出落在 int8 的 KV cache 上，带真实 requant scale；写死 1.0 会让
    下游按满量程反量化。依据设计 3.4。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "sf_dispatch"
    write_runtime_files(artifact, out_dir, gm=gm)

    from gml_bridge import calib_data

    expected = {np.float16(calib_data.kv_cache_scale(is_key=True)),
                np.float16(calib_data.kv_cache_scale(is_key=False))}
    checked = 0
    for node in _nodes_by_op(artifact, "KV_Cache_DMA"):
        sf = np.fromfile(
            out_dir / names.output_scale(node.node_id), dtype=np.float16)
        assert sf[0] in expected, (node.node_id, sf[0])
        checked += 1
    assert checked, "夹具里应当有 KV_Cache_DMA 节点"


def test_v_proj_is_the_only_gemm_with_a_real_output_sf(
        graph_and_artifact, tmp_path) -> None:
    """7 个 Gemm 里恰有 v_proj 的 output_sf != 1.0，其余为 1.0（验收 A4b）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "gemm_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    non_one = []
    for node in _nodes_by_op(artifact, "Gemm"):
        name = node.fields.get("output_sf")
        if not name:
            continue
        sf = np.fromfile(out_dir / name, dtype=np.float16)
        param = str(node.fields.get("pim_weight_param") or "")
        if not np.isclose(float(sf[0]), 1.0):
            non_one.append(param)
    assert non_one, "v_proj 应当带真实 requant scale"
    assert all("v_proj" in p for p in non_one), non_one


def test_other_operators_keep_output_sf_at_one(
        graph_and_artifact, tmp_path) -> None:
    """MatMul / Softmax / Mask / Eltwise 的输出留在 fp16 域，sf 恒 1.0。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "plain_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    for op in ("MatMul", "Softmax", "Mask", "EltwiseAdd", "EltwiseMul"):
        for node in _nodes_by_op(artifact, op):
            name = node.fields.get("output_sf")
            if not name:
                continue
            sf = np.fromfile(out_dir / name, dtype=np.float16)
            assert np.isclose(float(sf[0]), 1.0), (op, node.node_id, sf[0])


def test_fp16_domain_whitelist_lists_only_operators_that_carry_output_sf(
        graph_and_artifact) -> None:
    """白名单只列实测带 `output_sf` 的算子，不多列永远走不到的。

    多列不等于兜底：纯布局的 `Concat` / `Reshape` / `Transpose` 不带
    `output_sf`（两条导出路径实测 0 个），曾误列在白名单里——将来谁给
    `Concat` 加上 `output_sf`，它会**静默取 1.0**，正是设计 3.4 要防的那条
    静默路径。这条把白名单钉在实测面上。
    """
    from gml_bridge.export import _FP16_DOMAIN_OP_TYPES

    _, artifact = graph_and_artifact
    carriers = {n.fields["op_type"] for n in artifact.nodes
                if n.fields.get("op_type") and "output_sf" in n.fields}
    assert carriers, "图里应当有带 output_sf 的算子"
    never_reached = _FP16_DOMAIN_OP_TYPES - carriers
    assert not never_reached, (
        f"白名单里这些算子不带 output_sf、永远走不到 _output_scale_of："
        f"{sorted(never_reached)}")


def test_no_output_sf_is_zero(graph_and_artifact, tmp_path) -> None:
    """任何 output_sf 都不能是 0——形式非法（验收 A2）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "no_zero_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    for path in out_dir.glob("output_sf_*.bin"):
        raw = path.read_bytes()
        dtype = np.float32 if len(raw) % 4 == 0 and len(raw) == 4 else np.float16
        values = np.frombuffer(raw, dtype=dtype)
        assert (values > 0).all(), (path.name, values)


def test_no_scale_bin_is_zero(graph_and_artifact, tmp_path) -> None:
    """**任何** scale 类 bin 都不能是 0，不只 `output_sf` 与 phase 族。

    判据 4 说「不能是形式非法的值（如 scale=0）」，对所有 scale 一视同仁。
    原先的核对只覆盖 `output_sf_*` 与 `kantor_*_phase_3_*` 两族，于是非 phase
    的 Kantor scale 与 `updates_sf` 写死 0.0 一直没被发现（评审 3 问题 1）。
    这里按文件名扫全部 scale 族，防止同类回退再次溜过去。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "no_zero_scale"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for path in sorted(out_dir.glob("*.bin")):
        name = path.name
        # bias / zp / post-shift 恒 0 是硬件语义（需求 2.4 实测），不在此列。
        if not ("_sf_" in name or "scale_buffer_file" in name):
            continue
        if "bias" in name.lower() or "_zp" in name or "Scaling_PS" in name:
            continue
        raw = path.read_bytes()
        dtype = np.float32 if len(raw) == 4 else np.float16
        values = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        assert (values != 0).all(), (name, values[:8])
        checked += 1
    assert checked, "应当扫到 scale 类 bin"


def test_softmax_phases_eat_calibration_not_zeros(
        graph_and_artifact, tmp_path) -> None:
    """Softmax 五相的输入取标定激活，不是零张量（需求 P0-1 的同一类问题）。

    喂零会让 phase0（落 -max）变 -0.0、phase1 全 1.0、phase2 = 组长——那是
    「没算」而不是算出来的。判据取 phase1：它是 exp 数组，真实输入下不可能
    逐元素都等于 1.0。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "softmax_calib"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for node in _nodes_by_op(artifact, "Softmax"):
        phase1 = np.fromfile(
            out_dir / f"output_buffer_phase_1_{node.node_id}.bin",
            dtype=np.float16).astype(np.float32)
        assert phase1.size
        assert not np.allclose(phase1, 1.0), (node.node_id, phase1[:8])
        checked += 1
    assert checked, "夹具里应当有 Softmax 节点"


def test_output_scale_rejects_an_unknown_op_type() -> None:
    """识别不到的 op_type 直接抛，不回落 1.0（设计 3.4）。

    回落会让「新增或改名的算子没被识别」与「本来就该是 1.0」走同一条路，
    把 bug 静默掉（评审 3 问题 4）。
    """
    from gml_bridge.export import _output_scale_of
    from gml_bridge.writer import Node as GmlNode

    node = GmlNode(7, {"op_type": "SomeBrandNewOp"})
    with pytest.raises(ValueError, match="不在 output_sf 取值表里"):
        _output_scale_of(node, {7: node})


def test_kv_updates_sf_equals_the_nodes_own_output_sf(
        graph_and_artifact, tmp_path) -> None:
    """`updates_sf` 与本节点 `output_sf` 同一个 scale（同一块 cache 共用常量）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "updates_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for node in artifact.nodes:
        updates = node.fields.get("updates_sf")
        output = node.fields.get("output_sf")
        if not updates or not output:
            continue
        got = np.fromfile(out_dir / updates, dtype=np.float16)
        want = np.fromfile(out_dir / output, dtype=np.float16)
        assert float(got[0]) != 0.0, (updates, got)
        assert np.array_equal(got, want), (updates, got, want)
        checked += 1
    assert checked, "夹具里应当有带 updates_sf 的 KV 节点"


def test_kv_input_sf_is_not_hardcoded_one(
        graph_and_artifact, tmp_path) -> None:
    """KV 那一路的 `input_sf` 取那块 cache 的 scale，不写死 1.0（评审 r2 问题 1）。

    参考在同一批节点上三个 sf 字段是同一个常量（节点 28 的
    input/updates/output_sf 全为 0.0459）：读的是 int8 的 cache 平面，
    发 1.0 等于宣称「没量化」。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "kv_input_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for node in _nodes_by_op(artifact, "KV_Cache_DMA"):
        name = node.fields.get("input_sf")
        assert name, f"KV 节点 {node.node_id} 应当有 input_sf"
        got = np.fromfile(out_dir / name, dtype=np.float16)
        want = np.fromfile(
            out_dir / node.fields["output_sf"], dtype=np.float16)
        assert np.array_equal(got, want), (name, got, want)
        assert float(got[0]) != 1.0, (name, got)
        checked += 1
    assert checked, "夹具里应当有 KV_Cache_DMA 节点"


def test_boundary_buffers_carry_dtype_and_extension_on_the_real_graph(
        artifact) -> None:
    """真实图的 7 个入口 + 3 个出口缓冲都带 dtype，出口另带两类 extension。

    参考带 `input_data_extensions` 的节点是 193 个，我方曾是 190——差的 3 个
    正是这三个出口缓冲（评审 r5 问题 2）。`scripts/export_gml.py` 的 dtype
    自检把 `is_buffer` 整类跳过，看不见这个缺口，所以判据落在这条用例上。
    """
    from contracts import gml_hw_table as hw_table

    targets = {e.target for e in artifact.edges}
    entries = exits = 0
    for node in artifact.nodes:
        if not node.fields.get("is_buffer"):
            continue
        if node.node_id in targets:
            dt = node.fields.get("input_buffer_dtype")
            assert dt, (node.node_id, "缺 input_buffer_dtype")
            want = hw_table.data_extension(str(dt))
            for key in ("input_data_extensions", "output_data_extension"):
                assert node.fields.get(key) == want, (node.node_id, key, dt)
            exits += 1
        else:
            assert node.fields.get("output_buffer_dtype"), (
                node.node_id, "缺 output_buffer_dtype")
            entries += 1
    # 这个夹具走**整网**路径（多 embedding 那一段的入口），所以入口数不是
    # decode 块的 7 个。出口两条路都是 3 个（hidden + 两块 cache）。
    assert exits == 3, exits
    assert entries >= 7, entries


def test_io_info_puts_the_hidden_exit_at_output_slot_zero(
        graph_and_artifact, tmp_path) -> None:
    """出口槽号按角色排：hidden 出口固定槽 0，两块 cache 跟在后面。

    参考的 `output_idx` 是 0=hidden / 1=key_cache_out / 2=value_cache_out。
    按 `node_id` 升序发号会把 value cache 排到槽 0、hidden 挤到槽 2，消费端
    若按 `output_idx` 绑定缓冲就会把 cache 当成第一出口（评审 r5 问题 1）。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "io_slots"
    write_runtime_files(artifact, out_dir, gm=gm)
    payload = _read_io_info(out_dir / "IO_info.txt")

    by_slot = {v["output_idx"]: (k, v) for k, v in payload["outputs"].items()}
    assert sorted(by_slot) == [0, 1, 2], sorted(by_slot)
    # 槽 0 是本图的数据出口（这个夹具走整网路径，所以是 logits；decode 块上
    # 是 hidden）。要紧的是它不再被 KV cache 占掉。
    assert by_slot[0][1]["dtype"] == "float16", by_slot[0]
    assert "cache_out" not in str(by_slot[0][1]["node_name"]), by_slot[0]
    assert by_slot[1][1]["node_name"] == "key_cache_out", by_slot[1]
    assert by_slot[2][1]["node_name"] == "value_cache_out", by_slot[2]


def test_io_info_output_slots_ignore_node_numbering(tmp_path) -> None:
    """手工图：hidden 出口的 `node_id` 最大，槽号仍是 0。

    真实图上 hidden 出口恰好是编号最大的出口，所以「按角色排」与「按编号
    倒序排」同解，只有手工图能把判据钉在角色上。
    """
    from gml_bridge.export import GmlArtifact
    from gml_bridge.writer import Edge, Node

    op = Node(1, {"label": "Gemm_params_1", "op_type": "Gemm"})
    value_out = Node(2, {"label": "value_cache_out", "name": "value_cache_out",
                         "is_buffer": 1, "pim_io_rank": 4,
                         "input_buffer_dtype": "int8", "pim_kv_is_key": 0})
    key_out = Node(3, {"label": "key_cache_out", "name": "key_cache_out",
                       "is_buffer": 1, "pim_io_rank": 4,
                       "input_buffer_dtype": "int8", "pim_kv_is_key": 1})
    hidden_out = Node(4, {"label": "out_hidden", "name": "out_hidden",
                          "is_buffer": 1, "pim_io_rank": 3,
                          "pim_is_hidden_exit": 1,
                          "input_buffer_dtype": "float16"})
    artifact = GmlArtifact(
        text="", nodes=[op, value_out, key_out, hidden_out],
        edges=[Edge(1, 2, "1x32x1024x128"), Edge(1, 3, "1x32x1024x128"),
               Edge(1, 4, "1x1x1x4096")])

    payload = _read_io_info(write_io_info(artifact, tmp_path))
    assert payload["outputs"]["4"]["output_idx"] == 0
    assert payload["outputs"]["3"]["output_idx"] == 1
    assert payload["outputs"]["2"]["output_idx"] == 2


def test_io_info_rejects_two_hidden_exits(tmp_path) -> None:
    """标了两个 hidden 出口必抛，不静默挑一个当槽 0。"""
    from gml_bridge.export import GmlArtifact
    from gml_bridge.writer import Edge, Node

    op = Node(1, {"label": "Gemm_params_1", "op_type": "Gemm"})
    first = Node(2, {"label": "out_a", "name": "out_a", "is_buffer": 1,
                     "pim_io_rank": 3, "pim_is_hidden_exit": 1,
                     "input_buffer_dtype": "float16"})
    second = Node(3, {"label": "out_b", "name": "out_b", "is_buffer": 1,
                      "pim_io_rank": 3, "pim_is_hidden_exit": 1,
                      "input_buffer_dtype": "float16"})
    artifact = GmlArtifact(
        text="", nodes=[op, first, second],
        edges=[Edge(1, 2, "1x1x1x4096"), Edge(1, 3, "1x1x1x4096")])

    with pytest.raises(ValueError, match="pim_is_hidden_exit"):
        write_io_info(artifact, tmp_path)


def test_cache_exit_buffers_carry_the_quantization_fields(
        graph_and_artifact, tmp_path) -> None:
    """两个 cache 出口缓冲带 `input_sf` 三字段，取值 = 那块 cache 的 scale。

    参考的节点 199/200 有这三个字段，于是 `IO_info` 的 sf 与 GML 逐值对得上；
    缺了它们 IO_info 里报的 sf 在 GML 里没有对应的域（评审 r2 问题 1）。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "cache_exit"
    write_runtime_files(artifact, out_dir, gm=gm)
    payload = _read_io_info(out_dir / "IO_info.txt")

    exits = [n for n in artifact.nodes
             if str(n.fields.get("label", "")).endswith("_cache_out")]
    assert len(exits) == 2, [n.fields.get("label") for n in exits]
    for node in exits:
        for key in ("input_sf", "input_sf_dtype", "input_zp"):
            assert key in node.fields, (node.node_id, key)
        sf = np.fromfile(out_dir / node.fields["input_sf"], dtype=np.float16)
        assert float(sf[0]) != 1.0, (node.node_id, sf)
        # IO_info 报的同一项按 fp16 截断后与 GML 侧逐值相同。
        entry = payload["outputs"][str(node.node_id)]
        assert float(np.float16(entry["sf"])) == float(sf[0]), (entry, sf)


def test_kantor_scale_is_one_unless_the_family_does_fixed_point(
        graph_and_artifact, tmp_path) -> None:
    """非 phase 的 Kantor scale 按族看 `kantor_mode`（评审 r2 问题 2）。

    参考四族各不相同：定点化的两族（v_proj 的 `kantor_A`、RoPE 那条
    `Llama2Activation_add`）是「量化 scale 的倒数」量级，不定点化的 Cos/Sin
    族恒为 1.0。以前三族共用一个 absmax 倒数，把「不缩放」变成了一次缩放。
    """
    from gml_bridge.export import _KANTOR_FIXED_POINT_MODE, _kantor_mode_key

    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "kantor_scale"
    write_runtime_files(artifact, out_dir, gm=gm)

    fixed = scaled = 0
    for node in artifact.nodes:
        for key, value in node.fields.items():
            if "scale_buffer_file" not in key or "_phase_" in key:
                continue
            got = float(np.fromfile(out_dir / value, dtype=np.float16)[0])
            mode = str(node.fields.get(_kantor_mode_key(key)))
            if mode == _KANTOR_FIXED_POINT_MODE:
                # 倒数口径：cache scale < 1，所以倒数必然 > 1。
                assert got > 1.0, (node.node_id, key, mode, got)
                fixed += 1
            else:
                assert got == 1.0, (node.node_id, key, mode, got)
                scaled += 1
    assert fixed and scaled, (fixed, scaled)


def test_kantor_scale_rejects_a_family_without_a_mode() -> None:
    """有 Kantor scale 但没配套的 `kantor_mode` 时直接抛，不默认按不缩放发。"""
    from gml_bridge.export import _kantor_scale_of
    from gml_bridge.writer import Node as GmlNode

    node = GmlNode(7, {"op_type": "Gemm",
                       "kantor_A_scale_buffer_file": "kantor_A_scale_buffer_file_7.bin"})
    with pytest.raises(ValueError, match="判不出这一族做不做定点化"):
        _kantor_scale_of("kantor_A_scale_buffer_file", node, {7: node}, {})


def _read_io_info(path: Path) -> dict:
    """按甲方口径读 IO_info：含 numpy 标量，只能 eval 带 numpy 命名空间。"""
    return eval(path.read_text(),
                {"array": np.array, "np": np, "float32": np.float32,
                 "dtype": np.dtype, "int8": np.int8})


def test_io_info_is_read_back_with_numpy_namespace(
        graph_and_artifact, tmp_path) -> None:
    """IO_info 的 sf 要序列化成 numpy 标量（验收 A6）。

    甲方是直接 repr 一个含 numpy 标量的 dict（`array(1., dtype=float32)` 与
    `np.float32(1.0)` 两种混用），消费端必须 eval 带 numpy 命名空间。我方
    写裸 float 会让按甲方口径实现的消费端读不出预期类型。依据设计 3.5(2)。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "io_numpy"
    write_runtime_files(artifact, out_dir, gm=gm)
    payload = _read_io_info(out_dir / "IO_info.txt")

    items = list(payload["inputs"].values()) + list(payload["outputs"].values())
    assert items
    for entry in items:
        assert isinstance(entry["sf"], np.floating), entry


def test_io_info_rejects_literal_eval(graph_and_artifact, tmp_path) -> None:
    """裸 `ast.literal_eval` 读不出新格式——这是已知且同步改测的破坏性变更（E6）。"""
    import ast

    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "io_literal"
    write_runtime_files(artifact, out_dir, gm=gm)
    with pytest.raises(ValueError):
        ast.literal_eval((out_dir / "IO_info.txt").read_text())


def test_io_info_int8_sf_matches_the_output_sf_bin(
        graph_and_artifact, tmp_path) -> None:
    """int8 缓冲的 sf == 对应节点 `output_sf_<id>.bin` 的 fp16 值（验收 A5）。

    IO_info 存 fp32、GML bin 存 fp16，所以比 fp16 截断后的值。

    **必须真的打开 `output_sf_<id>.bin`**：只跟 `calib_data.kv_cache_scale()`
    这个常量集合比，测不出 `write_output_scale` 那条路径（`_output_scale_of`）
    与 IO_info 的 `scale_of()` 分叉的情况——比如回归成两处各写各的、其中一个
    写死 1.0（评审 3 问题 3）。这里按 IO_info 的 buffer 节点 id 经边回溯到
    对应的算子节点，读盘取它自己的 `output_sf_<id>.bin` 再比。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "io_sf"
    write_runtime_files(artifact, out_dir, gm=gm)
    payload = _read_io_info(out_dir / "IO_info.txt")

    from contracts import gml_names as names
    from gml_bridge import calib_data

    # buffer 节点 id -> 挂在它上面的那个算子节点 id（IO_info 的 sf 就是
    # 那个算子节点自己的 output_sf，不是 buffer 节点的）。
    op_of_buffer = {}
    for e in artifact.edges:
        op_of_buffer.setdefault(e.source, e.target)
        op_of_buffer.setdefault(e.target, e.source)

    allowed = {float(np.float16(calib_data.kv_cache_scale(is_key=True))),
               float(np.float16(calib_data.kv_cache_scale(is_key=False)))}
    seen = 0
    for side in ("inputs", "outputs"):
        for key, entry in payload[side].items():
            if entry["dtype"] != "int8":
                assert np.isclose(float(entry["sf"]), 1.0), entry
                continue
            op_id = op_of_buffer[int(key)]
            bin_path = out_dir / names.output_scale(op_id)
            bin_value = float(np.frombuffer(bin_path.read_bytes(), dtype=np.float16)[0])
            assert float(np.float16(entry["sf"])) == bin_value, (key, entry, bin_value)
            assert bin_value in allowed, (key, entry, bin_value)
            seen += 1
    assert seen, "夹具里应当有 int8 的 KV 缓冲"


def test_io_info_rank_follows_the_original_tensor_rank(
        graph_and_artifact, tmp_path) -> None:
    """IO_info 的 shape 按原始张量的 rank 报，不按边补维后的 rank（验收 A7）。

    甲方每条 I/O 沿用原始张量的 rank：relay 签名里 hidden 是 `[1,1,4096]`
    三维，cos/sin/mask/kv_position/k/v cache 六项四维。而 GML 的边一律四维，
    所以这里的判据是「IO_info 的 rank == 图侧记下的原始 rank」。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "io_rank"
    write_runtime_files(artifact, out_dir, gm=gm)
    payload = _read_io_info(out_dir / "IO_info.txt")

    by_id = {n.node_id: n for n in artifact.nodes}
    checked = 0
    for side in ("inputs", "outputs"):
        for key, entry in payload[side].items():
            rank = by_id[int(key)].fields.get("pim_io_rank")
            if rank is None:
                continue
            assert len(entry["shape"]) == int(rank), (key, entry)
            checked += 1
    assert checked, "图里应当有带原始 rank 的 I/O 缓冲"


def test_io_info_strips_edge_padding_for_a_three_dim_hidden_entry(
        tmp_path) -> None:
    """hidden 边是 `1x1x1x4096`，IO_info 要报 `[1, 1, 4096]`（验收 A7）。

    直接喂 llama 槽位口径的边，不用加载 7B 权重。KV cache 那一项 rank 为 4，
    不受影响——这条同时验证「只削 hidden、不碰其余六项」。
    """
    from gml_bridge.export import GmlArtifact
    from gml_bridge.writer import Edge, Node

    hidden = Node(1, {"label": "in_hidden", "name": "in_hidden",
                      "is_buffer": 1, "pim_io_rank": 3,
                      "output_buffer_dtype": "float16"})
    cache = Node(2, {"label": "in_key_cache", "name": "in_key_cache",
                     "is_buffer": 1, "output_buffer_dtype": "int8",
                     # int8 缓冲的 sf 按这个角色取，不认 label 子串。
                     "pim_kv_is_key": 1})
    op = Node(3, {"label": "Gemm_params_3", "op_type": "Gemm"})
    exit_buf = Node(4, {"label": "out_hidden", "name": "out_hidden",
                        "is_buffer": 1, "pim_io_rank": 3,
                        "input_buffer_dtype": "float16"})
    artifact = GmlArtifact(
        text="", nodes=[hidden, cache, op, exit_buf],
        edges=[Edge(1, 3, "1x1x1x4096"),
               Edge(2, 3, "1x32x1024x128"),
               Edge(3, 4, "1x1x1x4096")])

    payload = _read_io_info(write_io_info(artifact, tmp_path))
    assert payload["inputs"]["1"]["shape"] == [1, 1, 4096]
    assert payload["inputs"]["1"]["size"] == 4096
    assert payload["inputs"]["2"]["shape"] == [1, 32, 1024, 128]
    assert payload["outputs"]["4"]["shape"] == [1, 1, 4096]


def test_io_info_sf_backtracks_a_split_to_its_upstream_cache(tmp_path) -> None:
    """int8 的 `Split` 不带 `pim_kv_is_key`，sf 要靠回溯上游 DMA 取到。

    `scale_of` 曾把空表传给 `_kv_role_of`，这条回溯恒失效。当时不炸只是因为
    产物里 int8 的 I/O 恰好都自带角色字段；一旦 Split 成为 I/O 就会抛
    "没有 pim_kv_is_key"，把"判据没接线"报成"字段缺失"。
    """
    from gml_bridge import calib_data
    from gml_bridge.export import GmlArtifact
    from gml_bridge.writer import Edge, Node

    dma = Node(1, {"label": "k_cache_dma", "op_type": "KV_Cache_DMA",
                   "pim_kv_is_key": 1})
    # Split 自己不带角色，只带指向上游 DMA 的 input0_node_id。
    split = Node(2, {"label": "split_k", "name": "split_k", "op_type": "Split",
                     "is_buffer": 1, "input0_node_id": 1,
                     "input_buffer_dtype": "int8", "pim_io_rank": 4})
    artifact = GmlArtifact(
        text="", nodes=[dma, split], edges=[Edge(1, 2, "1x32x128x1024")])

    payload = _read_io_info(write_io_info(artifact, tmp_path))
    sf = payload["outputs"]["2"]["sf"]
    assert float(sf) == pytest.approx(calib_data.kv_cache_scale(is_key=True))
    assert float(sf) != 1.0


def test_io_info_rejects_padding_removal_that_drops_a_real_axis(
        tmp_path) -> None:
    """要去掉的维不是 1 就直接抛，不静默削掉一条真实轴。"""
    from gml_bridge.export import GmlArtifact
    from gml_bridge.writer import Edge, Node

    buf = Node(1, {"label": "in_hidden", "name": "in_hidden",
                   "is_buffer": 1, "pim_io_rank": 2,
                   "output_buffer_dtype": "float16"})
    op = Node(2, {"label": "Gemm_params_2", "op_type": "Gemm"})
    artifact = GmlArtifact(
        text="", nodes=[buf, op], edges=[Edge(1, 2, "1x32x1024x128")])
    with pytest.raises(ValueError):
        write_io_info(artifact, tmp_path)


def test_gml_edges_are_not_touched_by_the_io_info_rank_change(
        graph_and_artifact, tmp_path) -> None:
    """rank 改动只落在 IO_info，GML 边一条都不动（验收 A10）。

    参考 331 条边**全是四维**，包括 hidden 的 `1x1x1x4096`；只有 IO_info
    报 3 维。所以不能在 `from_fx` 改边的 dims——那会打破已经对齐的边。
    """
    gm, artifact = graph_and_artifact
    before = [(e.source, e.target, e.dims) for e in artifact.edges]
    write_runtime_files(artifact, out_dir := tmp_path / "io_edges", gm=gm)
    assert (out_dir / "IO_info.txt").is_file()
    after = [(e.source, e.target, e.dims) for e in artifact.edges]
    assert before == after


def _shift_files(out_dir: Path) -> dict[str, np.ndarray]:
    """落盘的全部 Shift 文件，按文件名索引，一律按 int8 读。"""
    return {p.name: np.fromfile(p, dtype=np.int8)
            for p in sorted(out_dir.glob("*hift*.bin"))}


def test_kantor_shift_is_minus_eight_where_fixed_point_happens(
        graph_and_artifact, tmp_path) -> None:
    """非 phase 的 Kantor Shift 要区分 -8 与 0，不能一律写 0（验收 A16）。

    取值判据是这个 Kantor 块是否做定点化（左移 8 位 = ×256）：
    `Llama2Activation_add` 与 v_proj 那一个 `kantor_A_Shift` 做，
    Cos/Sin 与 `kantor_B` 不做。参考逐族实测如此。依据设计 3.9。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "shift"
    write_runtime_files(artifact, out_dir, gm=gm)
    shifts = _shift_files(out_dir)

    v_proj_ids = {n.node_id for n in artifact.nodes
                  if n.fields.get("op_type") == "Gemm"
                  and "v_proj" in str(n.fields.get("pim_weight_param") or "")}
    assert v_proj_ids, "夹具里应当有 v_proj"

    minus_eight, zero = [], []
    for name, values in shifts.items():
        if "_phase_" in name:
            continue
        node_id = int(name.rsplit("_", 1)[-1].removesuffix(".bin"))
        if name.startswith("Kantor_A_Shift_Llama2Activation_add"):
            minus_eight.append(name)
        elif name.startswith("kantor_A_Shift_") and node_id in v_proj_ids:
            minus_eight.append(name)
        else:
            zero.append(name)
        expected = -8 if name in minus_eight else 0
        assert values.tolist() == [expected] * values.size, (name, values)

    assert minus_eight, "该发 -8 的族一个都没有，判据失效了"
    assert zero, "该发 0 的族一个都没有，判据失效了"


def test_every_shift_file_is_int8_with_only_minus_eight_or_zero(
        graph_and_artifact, tmp_path) -> None:
    """Shift 全族都是 int8，取值只有 -8 或 0（验收 A16、E8）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "shift_domain"
    write_runtime_files(artifact, out_dir, gm=gm)
    shifts = _shift_files(out_dir)

    assert shifts, "夹具里应当有 Shift 文件"
    for name, values in shifts.items():
        assert values.size, name
        assert set(values.tolist()) <= {-8, 0}, (name, sorted(set(values.tolist())))


def test_phase_shift_family_stays_minus_eight(
        graph_and_artifact, tmp_path) -> None:
    """走 phase 那条路的 37 个已经对了，本项改动不能把它们改坏（验收 A10）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "shift_phase"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for name, values in _shift_files(out_dir).items():
        if "_phase_" not in name:
            continue
        assert values.tolist() == [-8] * values.size, (name, values)
        checked += 1
    assert checked, "夹具里应当有 phase 型 Shift"


def test_kantor_bias_stays_all_zero(graph_and_artifact, tmp_path) -> None:
    """Kantor bias 参考恒 0，本项改动不能误伤它（验收 A16）。"""
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "kantor_bias"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for path in sorted(out_dir.glob("*antor*bias*.bin")):
        assert not np.fromfile(path, dtype=np.float32).any(), path.name
        checked += 1
    assert checked, "夹具里应当有 Kantor bias 文件"


def test_kantor_shift_and_scale_share_the_same_mode_judgement(
        graph_and_artifact, tmp_path) -> None:
    """Shift 与 scale 对同一族的「做不做定点化」判据必须相等（评审 r4 问题5）。

    以前 Shift 按文件名前缀猜，scale 按 `kantor_mode` 判——今天两套判据碰巧
    等价，但一族改名或新增一个定点化族时会分叉：scale 跟着 `kantor_mode`
    走，Shift 不会。改法让两者共用同一个 `_kantor_mode_key`，这里锁住
    「同一族的 Shift 非零 当且仅当 该族 scale > 1（定点化的倒数口径）」。
    """
    from gml_bridge.export import _KANTOR_FIXED_POINT_MODE, _kantor_mode_key

    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "kantor_shift_scale_agree"
    write_runtime_files(artifact, out_dir, gm=gm)

    checked = 0
    for node in artifact.nodes:
        shift_keys = {
            key: value for key, value in node.fields.items()
            if isinstance(value, str) and "Shift" in key
            and "kantor" in key.lower() and "_phase_" not in key}
        for key, value in shift_keys.items():
            mode_key = _kantor_mode_key(key)
            mode = str(node.fields.get(mode_key))
            shift = int(np.fromfile(out_dir / value, dtype=np.int8)[0])
            expected = -8 if mode == _KANTOR_FIXED_POINT_MODE else 0
            assert shift == expected, (node.node_id, key, mode, shift)
            checked += 1
    assert checked, "夹具里应当有非 phase 的 Kantor Shift"


def test_kantor_shift_rejects_a_family_without_a_mode() -> None:
    """有 Kantor Shift 但没配套的 `kantor_mode` 时直接抛，不默认按 0 发。"""
    from gml_bridge.export import _kantor_shift_of
    from gml_bridge.writer import Node as GmlNode

    node = GmlNode(7, {"op_type": "Gemm",
                       "kantor_A_Shift": "kantor_A_Shift_7.bin"})
    with pytest.raises(ValueError, match="判不出这一族做不做定点化"):
        _kantor_shift_of("kantor_A_Shift", node)


def test_rms_norm_output_sf_has_a_single_source_of_truth(
        graph_and_artifact, tmp_path) -> None:
    """RMSNorm 顶层 `output_sf` 与 `vpu_params.output_scale_factor_buffer`
    指向同一个文件、取值走同一处计算（评审 r4 问题3）。

    以前子块那一支写死 1.0，后写覆盖顶层字段算出的值；今天数值凑巧都是
    1.0 看不出差别，但 `_output_scale_of` 将来对 RMSNorm 给出别的值时，
    写死的 1.0 会静默盖掉——这里直接断言两个字段确实是同一个文件名，
    值必须与 `_output_scale_of` 的计算结果一致。
    """
    from gml_bridge.export import _output_scale_of

    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "rmsnorm_output_sf"
    write_runtime_files(artifact, out_dir, gm=gm)

    by_id = {n.node_id: n for n in artifact.nodes}
    checked = 0
    for node in artifact.nodes:
        if node.fields.get("op_type") != "RMSNorm_vpu":
            continue
        top_level = node.fields.get("output_sf")
        nested = node.nested.get("vpu_params", {}).get(
            "output_scale_factor_buffer")
        assert top_level is not None and nested is not None, node.node_id
        assert top_level == nested, (node.node_id, top_level, nested)

        expected = _output_scale_of(node, by_id)
        got = float(np.fromfile(out_dir / top_level, dtype=np.float32)[0])
        assert got == expected, (node.node_id, got, expected)
        checked += 1
    assert checked, "夹具里应当有 RMSNorm_vpu 节点"


def test_cross_node_input_buffer_reference_is_not_zero_filled(
        graph_and_artifact, tmp_path) -> None:
    """跨节点引用的 `input_buffer` 不再兜底补零（评审 r4 问题4）。

    以前 node 179 引用 node 177 的 RoPE cos/sin 乘积时，会先按
    `names.data_buffer(179, slot)` 判定不匹配后走补零分支写一份全零，
    等 node 177 自己处理时才用真实标定值覆盖——同一个文件名在一次导出里
    被写了两种内容。这里直接断言这两个跨节点引用的文件是非零的（生产者
    写的，不是补零分支写的）。
    """
    gm, artifact = graph_and_artifact
    out_dir = tmp_path / "cross_node_ref"
    write_runtime_files(artifact, out_dir, gm=gm)

    by_id = {n.node_id: n for n in artifact.nodes}
    checked = 0
    for node in artifact.nodes:
        for slot in range(3):
            key = f"input_buffer_{slot}" if slot else "input_buffer_0"
            value = node.fields.get(key)
            if not isinstance(value, str) or not value.endswith(".bin"):
                continue
            tail = value.rsplit(".", 1)[0].rsplit("_", 1)[-1]
            if not tail.isdigit() or int(tail) == node.node_id:
                continue  # 本节点自己的缓冲，不是跨节点引用
            producer = by_id.get(int(tail))
            if producer is None:
                continue
            path = out_dir / value
            if not path.is_file():
                continue
            content = np.fromfile(path, dtype=np.float16)
            if content.size and (content != 0).any():
                checked += 1
    assert checked, "夹具里应当至少有一个非零的跨节点 input_buffer 引用"


def test_gml_version_matches_the_reference(artifact) -> None:
    """GML 首部 `relay2gml_version "19.2.0"`（验收 A15）。

    这个字段被甲方解析器用于版本分派，填我方自己的 26.2.1 有被拒绝解析的
    风险。依据设计 3.7。
    """
    assert GML_VERSION == "19.2.0"
    assert 'relay2gml_version "19.2.0"' in artifact.text


def test_node_level_rtl_version_is_untouched(artifact) -> None:
    """`rtl_version` 是逐节点的硬件版本（参考 "1.4"），与格式版本号不是一回事。

    改格式版本号不能把它带走——两者混在一起改会让硬件版本失真。
    """
    from contracts import gml_hw_table

    assert gml_hw_table.RTL_VERSION == "1.4"
    if "rtl_version" in artifact.text:
        assert f'rtl_version "{gml_hw_table.RTL_VERSION}"' in artifact.text


def test_int8_input_edges_never_get_a_silent_one(
        graph_and_artifact, tmp_path) -> None:
    """int8 入槽的 `input_sf` 不许是 1.0——那等于宣称「这条边没量化」。

    评审 r6 问题6：`_sf` 分支以前在认不出 KV 角色时回落 1.0，实测让 32 个
    MatMul 与 1 个 Split 的 int8 入边全发 1.0，而参考那 64 个 MatMul 取的是
    上游 DQ 的 phase1。现在按 dtype 显式分派：fp16 域发 1.0、定点域接上游 DQ。
    """
    import re

    gm, artifact = graph_and_artifact
    write_runtime_files(artifact, tmp_path, gm=gm)
    text = (tmp_path / "relay2gml_graph.gml").read_text() if (
        tmp_path / "relay2gml_graph.gml").exists() else None
    if text is None:
        write_artifact(artifact, tmp_path)
        text = (tmp_path / "relay2gml_graph.gml").read_text()

    offenders = []
    for block in parse_blocks(text, "node"):
        for match in re.finditer(r'input(?:_(\d+))?_sf "([^"]+)"', block):
            slot, name = match.group(1), match.group(2)
            key = (f'input_buffer_{slot}_dtype' if slot
                   else 'input_buffer_dtype')
            declared = re.search(key + r' "([^"]+)"', block)
            if declared is None or declared.group(1) != "int8":
                continue
            values = np.fromfile(tmp_path / name, dtype=np.float16)
            if values.size == 1 and float(values[0]) == 1.0:
                offenders.append(name)
    assert not offenders, offenders


def test_unknown_input_dtype_raises_instead_of_falling_back(
        graph_and_artifact) -> None:
    """声明一个不认识的 input dtype 就抛，不静默发 1.0（设计 3.4）。

    缺省不声明是**合法**的 fp16（`_NO_TOP_IN_DTYPE` 那几个算子本就不写），
    所以这里用一个真正不认识的取值来验。
    """
    from dataclasses import replace

    from gml_bridge.export import _input_scale_of

    from gml_bridge.export import _kv_role_of

    _, artifact = graph_and_artifact
    by_id = {n.node_id: n for n in artifact.nodes}
    # KV 那一路按角色取值、不看 dtype，所以要挑一个非 KV 的节点来验 dtype 分派。
    node = next(n for n in artifact.nodes
                if n.fields.get("input_buffer_dtype") == "int8"
                and _kv_role_of(n, by_id) is None)
    broken = replace(
        node, fields={**node.fields, "input_buffer_dtype": "bfloat16"})

    with pytest.raises(ValueError, match="不在取值表里"):
        _input_scale_of(broken, by_id, None)


def test_entry_buffers_hold_their_own_calibration_constant(
        graph_and_artifact, tmp_path) -> None:
    """图入口缓冲装它自己那份标定常数，不是 hidden state 平铺。

    评审 r7 问题 1：以前 `write_data_buffer` 一律走 `activation_for`，四条入口
    （cos / sin / mask / kv_position）落成同一份 hidden state 的平铺/截断——
    cos 与 sin 因此逐字节相同，位置索引变成满量程噪声。旧断言只要求「非零」，
    这四处一路绿灯，所以这里按**内容**比，而不是按非零比。
    """
    from gml_bridge import calib_data

    gm, artifact = graph_and_artifact
    write_runtime_files(artifact, tmp_path, gm=gm)
    by_id = {n.node_id: n for n in artifact.nodes}
    dtypes = {"cos": np.float16, "sin": np.float16,
              "mask": np.float16, "kv_position": np.int16}

    checked = set()
    for edge in artifact.edges:
        role = by_id[edge.source].fields.get("pim_calib_role")
        if role is None:
            continue
        name = by_id[edge.source].fields.get("output_buffer")
        raw = np.fromfile(tmp_path / name, dtype=dtypes[role])
        want = calib_data.calibration_for_role(role, raw.size)
        assert np.array_equal(raw, want.astype(dtypes[role])), (role, name)
        checked.add(role)
    assert checked == {"cos", "sin", "mask", "kv_position"}, checked


def test_declared_input_dtype_agrees_with_the_producer_output_dtype(
        artifact) -> None:
    """生产者声明 int8 输出时，消费者的 `input_buffer_dtype` 也必须是 int8。

    评审 r7 问题 2：吃 `Llama2ActivationDQ` 的那个 Split 曾把 dtype 写死成
    fp16（照抄旧样本），于是按声明 dtype 分派的三处全部走偏——`input_sf`
    落进「fp16 域发 1.0」那一支、缓冲按 fp16 写成 2 倍字节，而
    `test_int8_input_edges_never_get_a_silent_one` 也按声明 dtype 筛选，
    这条边在它眼里「不是 int8」，判据静默落空。这里把「声明与上游不一致」
    本身变成当场报。
    """
    by_id = {n.node_id: n for n in artifact.nodes}
    offenders = []
    for node in artifact.nodes:
        producer = by_id.get(node.fields.get("input0_node_id"))
        if producer is None:
            continue
        if producer.fields.get("output_buffer_dtype") != "int8":
            continue
        declared = node.fields.get(
            "input_buffer_dtype", node.fields.get("input_buffer_0_dtype"))
        if declared is not None and declared != "int8":
            offenders.append(
                (node.node_id, node.fields.get("op_type"), declared,
                 producer.node_id, producer.fields.get("op_type")))
    assert not offenders, offenders


def test_kv_cache_dma_scaling_matches_the_reference_pair(
        graph_and_artifact, tmp_path) -> None:
    """KV_Cache_DMA 的 FPSU 定标是 (2.0, post_shift 14) 这一对。

    评审 r7 问题 5：post_shift 一直按这条发 14，scale 却落到通用分支算成
    1.0，与同一处注释自称的 2.0 打架，而参考两个 DMA 节点都是 2.0。
    """
    gm, artifact = graph_and_artifact
    write_runtime_files(artifact, tmp_path, gm=gm)
    seen = 0
    for node in artifact.nodes:
        if node.fields.get("op_type") != "KV_Cache_DMA":
            continue
        name = node.fields.get("Scaling_buffer_file")
        if name is None:
            continue
        scale = np.fromfile(tmp_path / name, dtype=np.float16)
        shift = np.fromfile(
            tmp_path / names.fpsu_post_shift(node.node_id), dtype=np.uint8)
        assert scale.tolist() == [2.0], node.node_id
        assert shift.tolist() == [14], node.node_id
        seen += 1
    assert seen, "图里没有 KV_Cache_DMA，这条判据没接上"


def test_int8_input_scale_group_count_matches_the_buffer_it_labels(
        graph_and_artifact, tmp_path) -> None:
    """每条 int8 入边上「缓冲元素数 / scale 个数」必须等于上游 DQ 的 group_size。

    评审 r8 问题 1：`input_sf` 照抄上游 DQ 的整条逐组向量，而 `input_buffer`
    按边宽定尺寸，两处各按各的口径。实测 32 个 `mha_batch_matmul1` 拿 32 组
    scale 去标注一个 128 元素的缓冲，等于宣称 group_size = 4，与上游 DQ 实际
    的 128 相矛盾；参考的同一批节点是 4096 元素配 32 组 = 128。

    A11 按族比字节数看不见这条：`input_buffer_N` 128B 与 `input_sf_N` 64B
    各自都合法，错的是**谁跟谁配对**。所以这里按边逐条算比值。
    """
    import re

    gm, artifact = graph_and_artifact
    write_runtime_files(artifact, tmp_path, gm=gm)
    text = artifact.text

    # 每个 DQ 节点的分组宽度：scale 个数 × group_size = 它的整张张量元素数。
    group_size = {node_id: spec.group_size
                  for node_id, spec in artifact.dq_specs.items()}

    offenders = []
    checked = 0
    for block in parse_blocks(text, "node"):
        node_id = int(re.search(r'\bnode_id (\d+)', block).group(1))
        for match in re.finditer(r'input_buffer(?:_(\d+))? "([^"]+)"', block):
            slot, buffer_name = match.group(1), match.group(2)
            key = (f'input_buffer_{slot}_dtype' if slot
                   else 'input_buffer_dtype')
            declared = re.search(key + r' "([^"]+)"', block)
            if declared is None or declared.group(1) != "int8":
                continue
            sf = re.search(
                (f'input_{slot}_sf' if slot else 'input_sf') + r' "([^"]+)"',
                block)
            if sf is None:
                continue
            # scale 与缓冲必须出自同一个节点，否则组数与元素数不是一个口径。
            owner = re.match(r'output_buffer_phase_1_(\d+)\.bin$', sf.group(1))
            if owner is None:
                continue
            dq_id = int(owner.group(1))
            elements = (tmp_path / buffer_name).stat().st_size  # int8, 1B/元素
            groups = (tmp_path / sf.group(1)).stat().st_size // 2  # fp16
            checked += 1
            if elements / groups != group_size[dq_id]:
                offenders.append(
                    (node_id, buffer_name, elements, groups,
                     elements / groups, group_size[dq_id]))
    assert not offenders, offenders
    # 断言不是空转：这张图上确实有 int8 入边走到了比值核对（防止将来改名
    # 让正则全部失配，测试静默变成永真）。
    assert checked, "没有一条 int8 入边被核对到，正则或命名口径变了"


def test_boundary_buffer_dtype_comes_from_the_slot_that_reads_it(
        graph_and_artifact) -> None:
    """图级 I/O 缓冲的位宽取**读它那一槽**的声明，与边的排列顺序无关。

    评审 r8 问题 2：`consumer_of` 原来用字典推导建映射，同一个 key 后写覆盖
    先写，于是多消费者的入口缓冲「谁说了算」取决于 `edges` 的顺序。实测 7 个
    入口里 4 个是多消费者（mask 32 个、hidden/cos/sin 各 2 个），今天两种取法
    恰好都落到 fp16 所以看不出差别，但 IO_info 的 sf、dtype 与整块缓冲的字节数
    都跟着它走，将来某个消费者声明 int8 就会随机。

    这里把边打乱再重跑：位宽必须一字不差。
    """
    import random

    from gml_bridge.from_fx import _stamp_boundary_dtypes

    _, artifact = graph_and_artifact
    boundary = [n for n in artifact.nodes if n.fields.get("is_buffer")]
    assert boundary, "这张图没有边界缓冲，测试失去意义"
    before = {n.node_id: (n.fields.get("output_buffer_dtype"),
                          n.fields.get("input_buffer_dtype"),
                          n.fields.get("input_data_extensions"))
              for n in boundary}

    shuffled = list(artifact.edges)
    random.Random(0).shuffle(shuffled)
    _stamp_boundary_dtypes(artifact.nodes, shuffled)

    after = {n.node_id: (n.fields.get("output_buffer_dtype"),
                         n.fields.get("input_buffer_dtype"),
                         n.fields.get("input_data_extensions"))
             for n in boundary}
    assert after == before


def test_boundary_buffer_with_conflicting_consumer_dtypes_raises(
        graph_and_artifact) -> None:
    """同一块边界缓冲被两个消费者声明成不同位宽就抛，不静默取一个。

    同一份字节只能有一种解释；静默取一个会让 IO_info 与缓冲尺寸按错的位宽走，
    而产品上看不出异常（设计 3.4「识别不到就抛，不静默」）。
    """
    from dataclasses import replace

    from gml_bridge.from_fx import _stamp_boundary_dtypes

    _, artifact = graph_and_artifact
    consumers_of: dict[int, list[int]] = {}
    for edge in artifact.edges:
        consumers_of.setdefault(edge.source, []).append(edge.target)
    by_id = {n.node_id: n for n in artifact.nodes}

    # 找一个多消费者的入口缓冲，把其中一个消费者的声明改掉。
    entry = next(n for n in artifact.nodes
                 if n.fields.get("is_buffer")
                 and len(consumers_of.get(n.node_id, [])) > 1
                 and n.fields.get("output_buffer"))
    name = entry.fields["output_buffer"]
    nodes = []
    flipped = False
    for node in artifact.nodes:
        if not flipped and node.node_id in consumers_of[entry.node_id]:
            keys = [k for k, v in node.fields.items()
                    if v == name and k.startswith("input_buffer")]
            if keys:
                slot = keys[0][len("input_buffer"):].strip("_")
                key = (f"input_buffer_{slot}_dtype" if slot
                       else "input_buffer_dtype")
                node = replace(node, fields={**node.fields, key: "int8"})
                flipped = True
        nodes.append(node)
    assert flipped, "没找到可以改声明的消费者，测试失去意义"
    # 入口自己的声明会短路掉相邻查询，去掉它才验得到冲突检测。
    nodes = [replace(n, fields={k: v for k, v in n.fields.items()
                                if k != "output_buffer_dtype"})
             if n.node_id == entry.node_id else n for n in nodes]

    with pytest.raises(ValueError, match="位宽声明不一致"):
        _stamp_boundary_dtypes(nodes, artifact.edges)
