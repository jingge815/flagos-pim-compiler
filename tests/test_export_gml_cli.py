"""验证 GML 导出的 CLI 入口。

这是唯一的对外交付路径——`gml_bridge` 各模块只有它和测试在调用。所以它的自检
（结构五规则 + 交叉校验）必须真的会失败，否则等于没有守卫。

不加载真实 7B 权重：那要 13 GiB 内存、导一次很慢。用随机小模型验证 CLI 的装配
逻辑，真实权重的导出由 `scripts/export_gml.py --layers 1` 手工跑（已实测通过）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

import genesim_bridge.paths as paths
from scripts.export_gml import (
    CheckLog,
    _check_dtype_coverage,
    _check_parser_families,
    _check_structure,
)

_SEQ_LEN = 16


def _unconfigure_reference(monkeypatch, tmp_path: Path) -> None:
    """把参考产物配成"没配"：既没有环境变量，配置文件也不存在。"""
    monkeypatch.delenv("GML_REFERENCE_DIR", raising=False)
    monkeypatch.delenv("GML_LLAMA2_REFERENCE_DIR", raising=False)
    monkeypatch.setattr(paths, "_CONFIG_FILE", tmp_path / "no-such-paths.json")


def _small_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000, hidden_size=64, intermediate_size=176,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=_SEQ_LEN, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """走 CLI 用的同一条装配路径，产出图与运行时文件。"""
    from gml_bridge.export import export_graph, write_artifact, write_runtime_files
    from runtime.compile import export_annotated_graph

    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        _small_model(), _SEQ_LEN, position_ids, dtype=torch.float32)

    out_dir = tmp_path_factory.mktemp("gml")
    artifact = export_graph(graph)
    gml_path = write_artifact(artifact, out_dir)
    files = write_runtime_files(artifact, out_dir, gm=graph)
    return artifact, gml_path, files, out_dir


def test_gml_is_written_under_the_expected_name(exported) -> None:
    """文件名必须是 `relay2gml_graph.gml`——对方按这个名字找图。"""
    _, gml_path, _, _ = exported
    assert gml_path.name == "relay2gml_graph.gml"
    assert gml_path.stat().st_size > 0


def test_structure_check_passes_on_our_output(exported) -> None:
    """CLI 的结构自检在正常产物上不该报问题。"""
    artifact, _, _, _ = exported
    assert _check_structure(artifact.text) == []


def test_structure_check_actually_catches_a_violation() -> None:
    """自检必须真的会失败，否则它只是装饰。

    构造一个违规图：把激活作为独立节点发出去（规则 1 禁止），自检要抓到。
    """
    from gml_bridge.writer import Edge, Node, write_gml

    bad = write_gml(
        [
            Node(1, {"label": "in", "name": "in", "is_buffer": 1,
                     "output_buffer": "input_buffer_3.bin",
                     "output0_node_id": 3}),
            # 独立的 Lut 节点——GML 没有这种表达方式。
            Node(3, {"label": "act", "name": "act", "op_type": "Lut",
                     "input_buffer": "input_buffer_3.bin",
                     "input_count": 1, "residual_input_buffer": [1],
                     "input0_node_id": 1}),
        ],
        [Edge(1, 3, "1x16x64")],
        version="26.10.1",
    )

    problems = _check_structure(bad)
    assert problems, "独立 Lut 节点应当被规则 1 拦下"
    assert any("规则 1" in problem for problem in problems)


def test_weights_and_buffers_land_on_disk(exported) -> None:
    """交叉校验通过意味着引用集与落盘集相等；这里再确认文件真的存在。"""
    artifact, _, files, out_dir = exported

    assert files.names_written == artifact.buffer_names
    assert artifact.weight_params, "图里应当有带权重的算子"

    for name in artifact.buffer_names:
        assert (out_dir / name).is_file(), f"{name} 没有落盘"


def test_skipping_weights_is_caught_by_cross_validation(tmp_path) -> None:
    """不传 `gm` 就没有权重可写，交叉校验必须报悬空引用。

    宁可在这里报错，也不要产出对方解析时才炸的悬空引用。
    """
    from gml_bridge.export import export_graph, write_runtime_files
    from runtime.compile import export_annotated_graph

    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        _small_model(), _SEQ_LEN, position_ids, dtype=torch.float32)
    artifact = export_graph(graph)

    with pytest.raises(ValueError, match="GML 引用了但没写盘"):
        write_runtime_files(artifact, tmp_path / "out", gm=None)


def test_parser_families_reads_the_configured_reference(tmp_path, monkeypatch) -> None:
    """参考目录必须来自配置，不能是写死的开发机路径。

    写死路径在别的机器上恒为"目录不存在"，这项检查会静默跳过——等于没有守卫。
    """
    ref = tmp_path / "parser_output"
    ref.mkdir()
    (ref / "activation_lut_file_195.bin").write_bytes(b"x")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "Bias_buffer_file_0.bin").write_bytes(b"x")
    monkeypatch.setenv("GML_LLAMA2_REFERENCE_DIR", str(ref))

    log = CheckLog()
    _check_parser_families(out_dir, log)

    check = log.checks[0]
    assert check.name == "参考独有文件族都已产出"
    assert "无参考目录" not in check.detail, "配了参考目录却走了跳过分支"
    assert not check.passed, "参考独有的 activation_lut 族没产出，应当判失败"


def test_parser_families_skips_without_reference(tmp_path, monkeypatch) -> None:
    """没配参考产物时跳过，不算失败——参考产物是独立交付物，可能不随包提供。"""
    _unconfigure_reference(monkeypatch, tmp_path)

    log = CheckLog()
    _check_parser_families(tmp_path / "out", log)

    assert log.checks[0].passed
    assert "无参考目录，跳过" == log.checks[0].detail


def test_dtype_coverage_skips_without_reference(tmp_path, monkeypatch) -> None:
    """同上：没配参考产物时 dtype 覆盖检查跳过，且不读图内容。"""
    _unconfigure_reference(monkeypatch, tmp_path)

    log = CheckLog()
    _check_dtype_coverage("", log)

    assert log.checks[0].passed
    assert "无参考产物，跳过" == log.checks[0].detail
