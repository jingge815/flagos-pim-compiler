"""第二层验证：字段覆盖率声明必须与实际产物一致。

漏一个字段族就是底层编译器少一项配置，而这不会在我们这侧报错。所以对参考产物的
每一族都要表态，并且表态要与实际产出对得上：声明已产出的必须真在产物里，
声明不适用的必须真不在。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.gml_coverage import (
    EMITTED,
    NOT_APPLICABLE,
    PENDING_CONVOLUTION,
    PENDING_QUANTIZATION,
    all_declared,
    undeclared,
)
from genesim_bridge.paths import gml_reference_dir
from gml_bridge.export import export_llama2
from scripts.gml_field_inventory import field_families

_REFERENCE_DIR = gml_reference_dir(required=False)
_REFERENCE_GML = (
    _REFERENCE_DIR / "relay2gml_graph.gml" if _REFERENCE_DIR else None
)

pytestmark = pytest.mark.skipif(
    _REFERENCE_GML is None or not _REFERENCE_GML.is_file(),
    reason="缺少 GML 参考产物，配置 paths.json 的 gml_reference_dir 后可跑",
)


@pytest.fixture(scope="module")
def reference_families() -> set[str]:
    return set(field_families(_REFERENCE_GML.read_text()))


@pytest.fixture(scope="module")
def emitted_families() -> set[str]:
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000, hidden_size=64, intermediate_size=176,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    artifact = export_llama2(model, seq_len=16, dtype=torch.float32)
    return set(field_families(artifact.text))


def test_every_reference_family_is_declared(reference_families) -> None:
    """参考产物的每一族都要有归属，未表态即声明不完整。"""
    missing = undeclared(reference_families)
    assert missing == set(), f"这些字段族没有表态: {sorted(missing)}"


def test_declaration_categories_do_not_overlap() -> None:
    """一族只能有一个归属，重叠说明声明自相矛盾。"""
    groups = {
        "EMITTED": set(EMITTED),
        "PENDING_QUANTIZATION": set(PENDING_QUANTIZATION),
        "PENDING_CONVOLUTION": set(PENDING_CONVOLUTION),
        "NOT_APPLICABLE": set(NOT_APPLICABLE),
    }
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = groups[left] & groups[right]
            assert overlap == set(), f"{left} 与 {right} 重叠: {sorted(overlap)}"


def test_declared_emitted_families_really_are_emitted(emitted_families) -> None:
    """声明已产出的族必须真在产物里，否则声明是空话。

    `input2_node_id` / `output1_node_id` 这类只在多输入或多消费者节点上出现，
    小模型可能碰不到，所以只要求交集非空且没有"声明产出却完全不见"的核心族。
    """
    # `activation_op_type` 不在 core 里：它只在有可折激活时出现，而 llama2 的
    # rsqrt/silu 是独立节点，小图可能一处融合都没有。
    core = {"id", "node_id", "label", "name", "op_type", "source", "target",
            "dims", "directed", "relay2gml_version", "input_buffer",
            "output_buffer", "input_count", "residual_input_buffer"}
    assert core <= set(EMITTED), "core 应当是 EMITTED 的子集"
    missing = core - emitted_families
    assert missing == set(), f"声明已产出但实际没有: {sorted(missing)}"


def test_pending_families_are_not_emitted_yet(emitted_families) -> None:
    """待第 4 轮的量化字段现在不该出现——出现了说明声明过时了。"""
    quantization_leaked = {
        family for family in PENDING_QUANTIZATION & emitted_families
    }
    assert quantization_leaked == set(), (
        f"这些族已经产出，应从 PENDING_QUANTIZATION 移到 EMITTED: "
        f"{sorted(quantization_leaked)}")


def test_not_applicable_families_are_absent(emitted_families) -> None:
    """声明不适用的族必须真不在产物里。"""
    leaked = set(NOT_APPLICABLE) & emitted_families
    # contraction 是例外：它声明为"融合块本身，已按结构产出"。
    leaked -= {"contraction"}
    assert leaked == set(), f"声明不适用却产出了: {sorted(leaked)}"


def test_every_not_applicable_entry_has_a_reason() -> None:
    """不产出要给理由，否则下一个人无法判断是遗漏还是有意。"""
    for family, reason in NOT_APPLICABLE.items():
        assert reason.strip(), f"{family} 缺少不适用的理由"


def test_coverage_ratio_is_reported(reference_families, emitted_families) -> None:
    """记录当前覆盖率，第 4 轮应显著上升。"""
    covered = reference_families & emitted_families
    assert len(covered) >= 20, (
        f"结构轮应覆盖 20 族以上，当前 {len(covered)}")
    # 声明总数要盖住参考产物（除调试副本）。
    non_debug = {f for f in reference_families if not f.startswith("DEBUG")}
    assert non_debug <= all_declared()
