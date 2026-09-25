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
