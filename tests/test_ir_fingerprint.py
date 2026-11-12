"""结构指纹：跨仓算法一致、能挡住跨模型复用，并进入精化缓存键。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.ir_fingerprint import (
    assert_ir_fingerprint, fingerprint_ir, fusion_signature)

_TINY_MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 32,
    "moe_intermediate_size": 16,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "head_dim": 8,
    "vocab_size": 100,
    "torch_dtype": "float16",
}


def _genesim_ir(tmp_path: Path, **overrides):
    """用 GeneSim 的构造器生成一份真实骨架，跳过未配置 genesim 的环境。"""
    from genesim_bridge.paths import genesim_root

    root = genesim_root(required=False)
    if root is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    src = str(root / "src")
    # 追加而不是插到最前：两仓都有 `backend` 包，遮住本仓的会让 vLLM
    # 子进程导入不到 `backend.vllm_executor`。
    if src not in sys.path:
        sys.path.append(src)
    from ir.model_ir import build_from_hf_config

    config = {**_TINY_MOE_CONFIG, **overrides}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return build_from_hf_config(str(config_path), execution_dtype="float16")


def test_compiler_and_genesim_compute_the_same_fingerprint(tmp_path) -> None:
    """两仓各有一份算法，必须逐字段一致，否则合法产物会被误判。"""
    ir = _genesim_ir(tmp_path)
    self_dict = ir.to_dict()
    assert fingerprint_ir(self_dict) == ir.structural_fingerprint()
    assert assert_ir_fingerprint(self_dict, "骨架") == ir.model_fingerprint


def test_fingerprint_changes_with_vocab_and_hidden(tmp_path) -> None:
    base = _genesim_ir(tmp_path).to_dict()
    vocab = {**base, "vocab_size": base["vocab_size"] + 1}
    hidden = {**base, "hidden_size": base["hidden_size"] + 32}
    assert fingerprint_ir(vocab) != fingerprint_ir(base)
    assert fingerprint_ir(hidden) != fingerprint_ir(base)


def test_edited_content_is_rejected(tmp_path) -> None:
    """同算子序列、不同词表：只改内容不改指纹必须失败。"""
    data = _genesim_ir(tmp_path).to_dict()
    data["vocab_size"] += 1
    with pytest.raises(ValueError, match="model_fingerprint"):
        assert_ir_fingerprint(data, "骨架")


def test_fingerprint_tracks_dependency_topology_and_layer_ownership(tmp_path) -> None:
    """换边、换张量身份、跨层移动算子都必须改变结构指纹。

    评审 round04 问题3：指纹原先只覆盖算子清单，`0→1` 改成 `1→0` 后
    SHA256 完全相同，同算子列表但数据流不同的图会被当成同一份产物。
    """
    import copy

    data = _genesim_ir(tmp_path).to_dict()
    base = fingerprint_ir(data)

    internal = next(
        index for index, dep in enumerate(data["dependencies"])
        if dep["src_op_id"] != -1 and dep["dst_op_id"] != -1)
    flipped = copy.deepcopy(data)
    edge = flipped["dependencies"][internal]
    edge["src_op_id"], edge["dst_op_id"] = (
        edge["dst_op_id"], edge["src_op_id"])
    assert fingerprint_ir(flipped) != base

    retagged = copy.deepcopy(data)
    retagged["dependencies"][internal]["tensor_id"] += ".rewired"
    assert fingerprint_ir(retagged) != base

    moved = copy.deepcopy(data)
    source_group = moved["subgraphs"][0]
    moved["subgraphs"][0] = source_group[:-1]
    moved["subgraphs"].append([source_group[-1]])
    assert fingerprint_ir(moved) != base


def test_fingerprint_ignores_refinement_and_placement_fields(tmp_path) -> None:
    """成本系数、设备提示与本地分片属于精化/放置字段，必须继续排除。"""
    data = _genesim_ir(tmp_path).to_dict()
    base = fingerprint_ir(data)
    for op in data["operators"]:
        op["flops_coeffs"] = {"Tq": 1.0}
        op["data_bytes_coeffs"] = {"Tq": 2.0}
        op["device_hint"] = "pim"
        op["local_input_shapes"] = [["Tq", 8]]
        op["local_output_shapes"] = [["Tq", 16]]
    assert fingerprint_ir(data) == base


def test_fusion_signature_ignores_order_and_sees_the_list() -> None:
    first = [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu", "expert": 2},
    ]
    second = list(reversed(first))
    assert fusion_signature(first) == fusion_signature(second)
    assert fusion_signature(first) != fusion_signature(first[:1])
    assert fusion_signature([]) != fusion_signature(first)


class _StopRefine(Exception):
    pass


def _fake_a2_files(tmp_path, *, refined_vocab=None):
    """搭一套 A2 的输入/缓存文件，返回 (genesim 根, 骨架路径, placement)。"""
    ir = _genesim_ir(tmp_path)
    skeleton = ir.to_dict()
    models = tmp_path / "models"
    models.mkdir()
    ir_path = models / "qwen3_moe.ir"
    ir_path.write_text(json.dumps(skeleton))

    refined = dict(skeleton)
    if refined_vocab is not None:
        refined = {**refined, "vocab_size": refined_vocab}
        refined["model_fingerprint"] = fingerprint_ir(refined)
    (models / "qwen3_moe_pimir.ir").write_text(json.dumps(refined))
    return tmp_path, ir_path


def test_a2_cache_key_covers_fingerprint_and_fusion(tmp_path, monkeypatch) -> None:
    import scripts.run_full_pipeline as pipeline

    genesim, ir_path = _fake_a2_files(tmp_path)
    skeleton = json.loads(ir_path.read_text())
    fingerprint = assert_ir_fingerprint(skeleton, str(ir_path))
    # 匹配的缓存：同指纹、同融合清单、同成本版本，直接跳过精化。
    (genesim / "models" / "qwen3_moe_oplevel_extensions.json").write_text(
        json.dumps({
            "version": pipeline.SIDECAR_VERSION,
            "ir_level": "oplevel",
            "ir_fingerprint": fingerprint,
            "fusion_signature": fusion_signature([]),
        }))
    monkeypatch.setattr(pipeline, "_refine_ir_level", lambda: "oplevel")
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *a, **k: (_ for _ in ()).throw(_StopRefine()))
    result = pipeline.step_a2_refine_pimir_ir(
        genesim, ir_path, genesim / "logs", "qwen3_moe")
    assert result.name == "qwen3_moe_pimir.ir"

    # 融合清单一变，缓存键就不匹配，必须重新精化。
    placement = genesim / "models" / "placement.json"
    placement.write_text(json.dumps({"fused_activations": [
        {"layer": 0, "role": "gate_proj", "expert": 0, "activation": "silu"},
    ]}))
    with pytest.raises(_StopRefine):
        pipeline.step_a2_refine_pimir_ir(
            genesim, ir_path, genesim / "logs", "qwen3_moe",
            placement=placement)


def test_a2_rejects_a_refined_ir_from_another_model(tmp_path, monkeypatch) -> None:
    """精化 IR 的词表变了（同编号、不同结构），不能当作同一份缓存。"""
    import scripts.run_full_pipeline as pipeline

    genesim, ir_path = _fake_a2_files(
        tmp_path, refined_vocab=_TINY_MOE_CONFIG["vocab_size"] + 1)
    (genesim / "models" / "qwen3_moe_oplevel_extensions.json").write_text(
        json.dumps({
            "version": pipeline.SIDECAR_VERSION,
            "ir_level": "oplevel",
            "ir_fingerprint": assert_ir_fingerprint(
                json.loads(ir_path.read_text()), str(ir_path)),
            "fusion_signature": fusion_signature([]),
        }))
    monkeypatch.setattr(pipeline, "_refine_ir_level", lambda: "oplevel")
    monkeypatch.setattr(
        pipeline, "_run",
        lambda *a, **k: (_ for _ in ()).throw(_StopRefine()))
    # 结构不一致 → 缓存判为无效 → 重新精化。
    with pytest.raises(_StopRefine):
        pipeline.step_a2_refine_pimir_ir(
            genesim, ir_path, genesim / "logs", "qwen3_moe")
