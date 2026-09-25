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
from genesim_bridge.paths import gml_llama2_reference_dir
from gml_bridge.export import export_llama2
from scripts.gml_field_inventory import field_families, field_families_by_op

_REFERENCE_DIR = gml_llama2_reference_dir(required=False)
_REFERENCE_GML = (
    _REFERENCE_DIR / "relay2gml_graph.gml" if _REFERENCE_DIR else None
)

pytestmark = pytest.mark.skipif(
    _REFERENCE_GML is None or not _REFERENCE_GML.is_file(),
    reason="缺少 GML 参考产物，配置 paths.json 的 gml_llama2_reference_dir 后可跑",
)


@pytest.fixture(scope="module")
def reference_families() -> set[str]:
    return set(field_families(_REFERENCE_GML.read_text()))


@pytest.fixture(scope="module")
def emitted_text() -> str:
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000, hidden_size=64, intermediate_size=176,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=16, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    return export_llama2(model, seq_len=16, dtype=torch.float32).text


@pytest.fixture(scope="module")
def emitted_families(emitted_text) -> set[str]:
    return set(field_families(emitted_text))


def test_debug_exemption_does_not_hide_semantic_flags(reference_families) -> None:
    """`DEBUG` 前缀豁免只能放过数据转储，不能放过语义标记。

    `DEBUG_sub_normal_weights_sf` 与 `weight_sf_multiplier` 同节点成对出现，
    `DEBUG_weight_buffer_spc` 那一组记的是权值定标粒度——它们是语义，不是
    某份缓冲的浮点副本。按前缀一刀切会把它们连同 40 多个 `*_float` 一起放过。
    """
    hidden = {
        f for f in undeclared(reference_families)
        if not (f.endswith("_float") or f.endswith("_hash"))
    }
    assert hidden == set(), f"这些 DEBUG 字段是语义标记，不能靠前缀豁免: {sorted(hidden)}"
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
            "output_buffer", "input_count", "residual_input_buffer",
            "input_sf_dtype", "output_sf_dtype"}
    assert core <= set(EMITTED), "core 应当是 EMITTED 的子集"
    # 权值指纹在写盘后才回填，小图的 `export_llama2` 文本里没有它；
    # 但它必须留在 EMITTED，否则声明会再次把它漏掉。
    assert "weight_buffer_hash" in EMITTED
    missing = core - emitted_families
    assert missing == set(), f"声明已产出但实际没有: {sorted(missing)}"


def test_pending_families_are_not_emitted_yet(emitted_families) -> None:
    """待产出的字段现在不该出现——出现了说明声明过时了。"""
    leaked = {
        family for family in (PENDING_QUANTIZATION | PENDING_CONVOLUTION) & emitted_families
    }
    assert leaked == set(), (
        f"这些族已经产出，应从 PENDING_* 移到 EMITTED: {sorted(leaked)}")


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

# 参考产物里确认是调试副本、不要求我方产出的字段族。
# 逐个列名而不是按 `DEBUG` 前缀一刀切：前缀豁免会把与真源字段成对的
# 语义标记（如 `DEBUG_sub_normal_weights_sf`）一起放过。
_DEBUG_COPIES = {
    "cos_mul_output_hash", "sin_mul_output_hash", "lut_debug",
}


def test_reference_only_debug_copies_are_unaccounted(reference_families) -> None:
    """参考有、我方既没产出也没给理由的字段族，必须都在调试副本白名单里。

    「逐节点字段差缺 0 / 多 0」原先没有可复跑的判据：覆盖测试把
    `NOT_APPLICABLE` 和整个 `DEBUG` 前缀都排除了，于是把一个字段错标成
    不适用、或靠前缀豁免放过，断言都恒真。这条直接拿参考产物的字段全集
    对三张声明表，差集只能是白名单里的调试副本。
    """
    accounted = all_declared() | _DEBUG_COPIES
    gap = {f for f in reference_families if f not in accounted
           and not f.startswith("DEBUG")}
    assert gap == set(), f"参考有、我方没有表态的字段族: {sorted(gap)}"


def test_every_reference_family_is_emitted_on_the_same_operator(
        emitted_text) -> None:
    """判据下沉到算子：参考在某算子上有的字段族，我方在同一个算子上也要有。

    上面的族级判据只看整份产物有没有这一族——一族只要在任意一个节点上出现
    就算通过。于是「Concat 的 32 槽定标整块没写」和「32 个 Mask 各多写一套
    槽位定标」两种错都能绿灯。这条按 `op_type` 比，槽位号抹平（参考是 32 头、
    本地用例是 4 头，槽数天然不同）。
    """
    reference = field_families_by_op(_REFERENCE_GML.read_text())
    emitted = field_families_by_op(emitted_text)

    def named(op: str, families) -> list[str]:
        return [f"{op}.{f}" for f in sorted(families)]

    # 「我方在任何算子上都发不出这一族」不归这条判据管——那是上面族级判据的
    # 事。本地用例是未量化的合成模型，没有权值缓冲，`weight_buffer_hash`
    # 在这里天然不存在；把它算进来只会把夹具差异当成缺陷。
    reachable = set().union(*emitted.values()) if emitted else set()

    missing: list[str] = []
    for op in sorted(set(reference) & set(emitted)):
        want = {f for f in reference[op]
                if not f.startswith("DEBUG") and f not in NOT_APPLICABLE
                and f in reachable}
        missing += named(op, want - emitted[op])

    extra: list[str] = []
    for op in sorted(set(reference) & set(emitted)):
        extra += named(op, emitted[op] - reference[op])

    absent = sorted(set(reference) - set(emitted))
    assert not absent, f"参考有、我方完全没有的算子: {absent}"
    assert not missing, f"参考有、我方在同一算子上没有的字段族: {missing}"
    assert not extra, f"参考没有、我方多写的字段族: {extra}"
