"""插入 DynamicScaling 的判据。

这一步贡献参考产物 3231 个 bin 里的 1789 个（57%），所以「插在哪、分几组」
是核心断言。插入规则从实物反推，37 个节点无例外：

    在每个矩阵乘的**激活输入**边上插一个 DQ；matmul1（QKᵀ）除外
"""

from __future__ import annotations

import collections
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.fuse_pim import fuse_for_pim
from graph.quant_pass import DQ_META_KEY, insert_dynamic_scaling
from graph.split_heads import (
    HEAD_ROLE_META_KEY,
    ROLE_MATMUL_PV,
    ROLE_MATMUL_QK,
    split_attention_heads,
)

HEADS = 4
HIDDEN = 32
SEQ = 8


@pytest.fixture(scope="module")
def base_graph():
    """一层 llama2 的小图，未做任何变换。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128, hidden_size=HIDDEN, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=HEADS,
            num_key_value_heads=HEADS, max_position_embeddings=SEQ,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)
    return export_annotated_graph(model, SEQ, position_ids, dtype=torch.float32)


@pytest.fixture(scope="module")
def quantized(base_graph):
    """走完整流水线：融合 → 逐头展开 → 插 DQ。"""
    clone = copy.deepcopy(base_graph)
    fuse_for_pim(clone)
    split_attention_heads(clone)
    report = insert_dynamic_scaling(clone)
    return clone, report


def _specs(gm) -> list:
    return [n.meta[DQ_META_KEY] for n in gm.graph.nodes if DQ_META_KEY in n.meta]


def test_every_matmul_gets_a_dq_on_its_activation_input(quantized) -> None:
    """每个吃激活的矩阵乘都要有一个 DQ 上游。

    实测参考产物：q/k/v 共用 1 条、o/down 各 1、gate/up 共用 1，
    再加逐头 matmul2；matmul1 **零个**。
    """
    clone, report = quantized
    assert report.inserted > 0
    assert not report.skipped, report.skipped

    for node in clone.graph.nodes:
        role = node.meta.get(HEAD_ROLE_META_KEY)
        if role == ROLE_MATMUL_PV:
            source = node.args[0]
            assert DQ_META_KEY in source.meta, f"{node.name} 的激活输入没有 DQ"
        elif role == ROLE_MATMUL_QK:
            source = node.args[0]
            assert DQ_META_KEY not in source.meta, \
                f"{node.name}(matmul1) 不该有 DQ 上游"


def test_attention_scores_are_one_group(quantized) -> None:
    """attention scores 整条当一组，其余按 128 切。

    实测参考产物 `global_pooling_group_size_phase_0` 只有两种取值：
    1024（32 个 attention scores 节点）与 128（5 个 hidden/MLP 节点）。
    """
    clone, report = quantized
    assert report.attention_scores == HEADS

    for spec in _specs(clone):
        if spec.is_attention_scores:
            assert spec.groups == 1, "scores 应当整条一组"
            assert spec.group_size == spec.numel
        else:
            assert spec.group_size == 128


def test_group_count_is_numel_over_group_size(quantized) -> None:
    """组数 = numel / group_size —— 它决定各相 bin 的元素数。"""
    clone, _ = quantized
    for spec in _specs(clone):
        assert spec.numel % spec.group_size == 0
        assert spec.groups == spec.numel // spec.group_size


def test_shared_activation_shares_one_dq(quantized) -> None:
    """同一源张量的多个矩阵乘共用一条 DQ。

    参考产物：q/k/v 共用节点 24，gate/up 共用一条。
    逐头分数 DQ 源各不相同，仍是一对一。
    """
    clone, _ = quantized

    gemm_dqs = []
    for node in clone.graph.nodes:
        if DQ_META_KEY not in node.meta:
            continue
        if node.meta[DQ_META_KEY].is_attention_scores:
            continue
        gemm_dqs.append(node)
    # 一层：attn 前、o_proj 前、mlp 前、down 前、lm_head 前。
    assert 1 <= len(gemm_dqs) <= 5
    for dq in gemm_dqs:
        users = [u for u in dq.users
                 if u.target is not torch.ops.aten._assert_tensor_metadata.default]
        assert users, dq.name


def test_dq_preserves_numerics(base_graph, quantized) -> None:
    """插 DQ 后图的数值必须不变。

    DQ 在 fx 图里是恒等操作（`alias`），真正的量化发生在硬件上 ——
    编译期只需要一个占位节点承载那 4 相字段。若这里数值变了，
    说明插错了位置（比如接到了权重那一路）。
    """
    clone, _ = quantized

    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (1, SEQ))
    causal_mask = torch.zeros(1, 1, SEQ, SEQ)
    position_ids = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        expected = base_graph(input_ids, causal_mask, position_ids)
        actual = clone(input_ids, causal_mask, position_ids)

    expected_tensor = expected[0] if isinstance(expected, tuple) else expected
    actual_tensor = actual[0] if isinstance(actual, tuple) else actual
    torch.testing.assert_close(
        actual_tensor, expected_tensor, rtol=1e-4, atol=1e-5)


def test_insertion_is_idempotent(quantized) -> None:
    """再插一次不应有变化——已有 DQ 的边要被跳过。"""
    clone, _ = quantized
    before = len(list(clone.graph.nodes))

    second = insert_dynamic_scaling(clone)
    assert second.inserted == 0
    assert len(list(clone.graph.nodes)) == before


# ---------------------------------------------------------------------------
# GML 侧：DQ 如何变成 4 相字段与 bin
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact_and_gm(base_graph):
    """走完整导出流水线后的产物，连同变换后的图。

    写盘要 `gm` 才能取出真实权重去量化；传 None 会让 `weight_buffer`
    悬空（交叉校验直接抛）。
    """
    from gml_bridge.export import export_graph

    clone = copy.deepcopy(base_graph)
    return export_graph(clone), clone


@pytest.fixture(scope="module")
def artifact(artifact_and_gm):
    return artifact_and_gm[0]


def test_dq_emits_four_phases(artifact) -> None:
    """每个 DQ 节点要发射 4 相的字段族，不是 5 相。

    实测 DynamicScaling 是 4 相、Softmax 是 5 相。多算一相会为不存在的相
    分配文件与字段。
    """
    import re

    dq_nodes = [
        node for node in artifact.nodes
        if node.fields.get("op_type") == "DynamicScaling"
    ]
    assert dq_nodes

    for node in dq_nodes:
        phases = {
            int(match.group(1))
            for key in node.fields
            if (match := re.search(r"_phase_(\d+)$", key))
        }
        assert phases == {0, 1, 2, 3}, f"节点 {node.node_id}: {sorted(phases)}"


def test_dq_declares_dynamic_quantization(artifact) -> None:
    """DQ 节点带 `use_dynamic_quantization 1`。"""
    for node in artifact.nodes:
        if node.fields.get("op_type") == "DynamicScaling":
            assert node.fields["use_dynamic_quantization"] == 1


def test_dq_group_size_fields_agree(artifact) -> None:
    """两个 group_size 字段在同一节点上必须相等（实测 37/37）。"""
    for node in artifact.nodes:
        if node.fields.get("op_type") != "DynamicScaling":
            continue
        pooling = node.fields["global_pooling_group_size_phase_0"]
        kantor = node.fields["kantor_A_spg_group_size_phase_3"]
        assert pooling == kantor


def test_phase_bins_are_written_with_reference_byte_sizes(
    artifact_and_gm, tmp_path
) -> None:
    """4 相的 bin 要按「组数 × dtype 宽度」落盘。

    字节数是校验器逐文件比对的项，也是内容不可比时最强的判据
    （参考产物是合成数据，见计划 §5）。
    """
    import numpy as np

    from gml_bridge.export import write_artifact, write_runtime_files

    artifact, gm = artifact_and_gm
    write_artifact(artifact, tmp_path)
    files = write_runtime_files(artifact, tmp_path, gm=gm)

    checked = 0
    for node in artifact.nodes:
        if node.fields.get("op_type") != "DynamicScaling":
            continue
        node_id = node.node_id
        groups = artifact.dq_specs[node_id].groups
        numel = artifact.dq_specs[node_id].numel
        checked += 1

        # p0/p1/p2 是逐组的 fp16；p3 是逐元素的 int8。
        for phase in (0, 1, 2):
            path = tmp_path / f"output_buffer_phase_{phase}_{node_id}.bin"
            assert path.stat().st_size == groups * 2, path.name
        assert (tmp_path / f"output_buffer_phase_3_{node_id}.bin"
                ).stat().st_size == numel

        # Kantor 三族：scale fp16、shift int8、bias fp32，都是逐组。
        assert (tmp_path / f"kantor_A_scale_buffer_file_phase_3_{node_id}.bin"
                ).stat().st_size == groups * 2
        assert (tmp_path / f"kantor_A_Shift_buffer_file_phase_3_{node_id}.bin"
                ).stat().st_size == groups
        assert (tmp_path / f"kantor_A_bias_buffer_file_phase_3_{node_id}.bin"
                ).stat().st_size == groups * 4

        # 两张 LUT 各 288 字节。
        for phase in (1, 2):
            assert (tmp_path / f"LUT_phase_{phase}_{node_id}.bin"
                    ).stat().st_size == 288

        # phase0 的 bias 是 fp32 常量 2^-63。
        bias = np.fromfile(
            tmp_path / f"Bias_buffer_phase_0_{node_id}.bin", dtype=np.float32)
        assert bias[0] == pytest.approx(2.0 ** -63)

    assert checked > 0
    assert files.names_written


def test_attention_scale_reaches_the_scaling_buffer(
    artifact_and_gm, tmp_path
) -> None:
    """matmul1 的 `Scaling_buffer_file` 要装 1/√head_dim。

    实测参考产物 32 个 matmul1 全是 1/√128 = 0.088388。
    漏掉它等于丢掉 attention scale —— 数值全错而结构校验查不出来。
    """
    import math

    import numpy as np

    from gml_bridge.export import write_artifact, write_runtime_files

    artifact, gm = artifact_and_gm
    write_artifact(artifact, tmp_path)
    write_runtime_files(artifact, tmp_path, gm=gm)

    head_dim = HIDDEN // HEADS
    expected = np.float16(1.0 / math.sqrt(head_dim))

    scales = collections.Counter()
    for path in tmp_path.glob("Scaling_buffer_file*.bin"):
        if path.stat().st_size == 2:
            scales[float(np.fromfile(path, dtype=np.float16)[0])] += 1

    assert float(expected) in scales, f"没有 1/√{head_dim}: {dict(scales)}"
    assert scales[float(expected)] == HEADS, "每个 matmul1 一份"


def test_internal_fields_never_reach_the_gml_text(artifact) -> None:
    """`pim_` 前缀的内部字段只在编译期传值，不能进 GML 文本。

    对方的解析器不认识它们。这条钉住那个过滤 —— 它很容易在加新字段时被绕过。
    """
    assert "pim_" not in artifact.text
