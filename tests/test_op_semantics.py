"""算子语义的单一真源与四份派生视图。

P0-3 要的是「同一事实两处维护」消失：新增一个算子只改登记表一处，
四份既有清单（OP_TYPES / _ROLE_OP_TYPES / _OPLEVEL_OPS / MNEMONICS）
全部由它派生，且派生结果与改动前的字面量逐项相同。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.op_semantics import (
    OP_SEMANTICS,
    OpSemantics,
    aten_to_gml,
    mnemonics,
    oplevel_ops,
    role_to_gml,
)

# 改动前 _OPLEVEL_OPS 的字面量（driver.py:83）。
OLD_OPLEVEL_OPS = frozenset({
    "softmax", "dynamic_quant", "gather", "rope", "matmul", "normalize",
    "mask", "transpose", "reshape", "concat", "convert", "lut", "eltwise",
    "kv_cache", "split_heads",
})

# 改动前 MNEMONICS 的字面量（op_classify.py:249），**含顺序**。
OLD_MNEMONICS = (
    "pim.normalize", "pim.matmul", "pim.softmax", "pim.mask", "pim.rope",
    "pim.lut", "pim.eltwise", "pim.dynamic_quant", "pim.kv_cache", "pim.gather",
    "pim.transpose", "pim.reshape", "pim.split_heads", "pim.concat",
)

# 改动前 OP_TYPES 的字面量（from_fx.py:44），28 项映射到 19 个 GML 类型。
OLD_OP_TYPES = {
    "linear.default": "Gemm",
    "addmm.default": "Gemm",
    "mm.default": "MatMul",
    "bmm.default": "MatMul",
    "matmul.default": "MatMul",
    "add.Tensor": "EltwiseAdd",
    "sub.Tensor": "EltwiseSub",
    "mul.Tensor": "EltwiseMul",
    "div.Tensor": "EltwiseDiv",
    "_softmax.default": "Softmax",
    "rsqrt.default": "RMSNorm_vpu",
    "silu.default": "Silu",
    "masked_fill.Scalar": "Mask",
    "where.self": "Mask",
    "split.Tensor": "Split",
    "split_with_sizes.default": "Split",
    "transpose.int": "Transpose",
    "permute.default": "Transpose",
    "view.default": "Reshape",
    "reshape.default": "Reshape",
    "cat.default": "Concat",
    "max_pool2d.default": "MaxPool",
    "avg_pool2d.default": "AveragePool",
    "convolution.default": "Conv",
    "to.dtype": "Convert",
    "to.dtype_layout": "Convert",
    "embedding.default": "Gather",
    "scaled_dot_product_attention.default": "MatMul",
}

# 改动前 _ROLE_OP_TYPES 的字面量（from_fx.py:224）。
OLD_ROLE_OP_TYPES = {
    "matmul1": "MatMul",
    "matmul2": "MatMul",
    "mask": "Mask",
    "softmax": "Softmax",
}


def _aten_name(target) -> str:
    """`torch.ops.aten.to.dtype` → "to.dtype"；`aten.mm.default` → "mm.default"。"""
    text = str(target)
    assert text.startswith("aten."), text
    return text.removeprefix("aten.")


def test_oplevel_ops_match_the_literal_they_replace() -> None:
    assert oplevel_ops() == OLD_OPLEVEL_OPS


def test_mnemonics_match_the_literal_they_replace() -> None:
    """含顺序 —— 它只进错误信息，但顺序稳定便于人读，且改动前就是这个顺序。"""
    assert mnemonics() == OLD_MNEMONICS


def test_aten_mapping_matches_the_literal_it_replaces() -> None:
    """28 项 aten 目标 → 19 个 GML 类型，逐项相同。"""
    derived = {_aten_name(k): v for k, v in aten_to_gml().items()}
    assert derived == OLD_OP_TYPES
    assert len(set(derived.values())) == 19


def test_role_mapping_matches_the_literal_it_replaces() -> None:
    assert role_to_gml() == OLD_ROLE_OP_TYPES


def test_role_lookup_still_takes_priority_over_aten() -> None:
    """逐头节点必须按角色判 op_type，不能退回按 aten 判。

    `from_fx` 先查角色再查 aten。两张表若合并成一张，这个优先级就丢了 ——
    mask 角色的节点其 aten 目标可能是 where.self，判定路径会变。
    """
    roles = set(role_to_gml())
    aten_keys = {_aten_name(k) for k in aten_to_gml()}
    assert roles == {"matmul1", "matmul2", "mask", "softmax"}
    assert not (roles & aten_keys), "角色别名与 aten 目标名撞车会让优先级失去意义"


def test_adding_an_operator_touches_only_the_registry() -> None:
    """在登记表加一个算子，四个派生视图自动包含它，不必改别处。

    用 `relu.default` 当例子：它是真实存在的 aten 算子，但不在 GML 映射里
    （激活走融合，不落独立 GML 节点）。
    """
    import contracts.op_semantics as mod

    extra = OpSemantics("fake_op", "FakeGml", has_kernel=True, is_mnemonic=True,
                        aten_targets=("relu.default",))
    original = mod.OP_SEMANTICS
    try:
        mod.OP_SEMANTICS = original + (extra,)
        assert "fake_op" in mod.oplevel_ops()
        assert "pim.fake_op" in mod.mnemonics()
        assert mod.aten_to_gml()[torch.ops.aten.relu.default] == "FakeGml"
    finally:
        mod.OP_SEMANTICS = original


def test_registry_rejects_duplicate_names() -> None:
    """重名会让派生视图取决于遍历顺序。"""
    import contracts.op_semantics as mod

    dup = (OpSemantics("matmul", "MatMul", True, True),
           OpSemantics("matmul", "Gemm", False, False))
    with pytest.raises(ValueError, match="重名"):
        mod._validate_registry(dup)


def test_registry_rejects_one_aten_target_mapping_to_two_ops() -> None:
    """同一 aten 目标映射到两个算子，派生结果就不确定了。"""
    import contracts.op_semantics as mod

    clash = (OpSemantics("a", "A", False, False, aten_targets=("mm.default",)),
             OpSemantics("b", "B", False, False, aten_targets=("mm.default",)))
    with pytest.raises(ValueError, match="同时映射到"):
        mod._validate_registry(clash)


def test_unknown_aten_name_raises_immediately() -> None:
    """拼错算子名必须立刻暴露，不能静默少一项映射。"""
    import contracts.op_semantics as mod

    with pytest.raises(ValueError, match="torch.ops.aten 没有"):
        mod._resolve_aten("no_such_operator.default")
    with pytest.raises(ValueError, match="没有重载"):
        mod._resolve_aten("mm.no_such_overload")


def test_aten_reachable_gml_types_are_all_registered() -> None:
    """能由 aten 目标或角色别名查到的 GML 类型，必须都在登记表里。

    否则那类节点在图上会被判成「无法映射到 GML」。
    """
    gml_types = {s.gml_op_type for s in OP_SEMANTICS if s.gml_op_type}
    mapped = set(aten_to_gml().values()) | set(role_to_gml().values())
    assert mapped <= gml_types


def test_pass_created_ops_have_no_aten_route() -> None:
    """没有 aten 路径的 GML 类型，恰好是那些由 pass 造出来的。

    它们不出现在 OP_TYPES 里不是遗漏：RoPE 是一条链折成的、
    DQ / KV_Cache_DMA 是插出来的、折进 contraction 的激活发成 Lut。
    """
    gml_types = {s.gml_op_type for s in OP_SEMANTICS if s.gml_op_type}
    mapped = set(aten_to_gml().values()) | set(role_to_gml().values())
    assert gml_types - mapped == {
        "Llama2Activation", "DynamicScaling", "KV_Cache_DMA", "Lut"}


def test_fusion_targets_must_be_known_to_the_registry() -> None:
    """融合表引用的 aten 目标必须在算子登记表里。

    两表不同步时融合会静默不发生 —— 不报错，只是少折一个算子。
    """
    from contracts.fusion_contract import (
        FUSION_TARGETS,
        GATE_TARGETS,
        _validate_fusion_targets,
    )

    _validate_fusion_targets()          # 现状自洽
    known = set(aten_to_gml())
    assert FUSION_TARGETS <= known
    assert GATE_TARGETS <= known
    assert GATE_TARGETS < FUSION_TARGETS, "门控表是主算子表的真子集"

    import contracts.fusion_contract as mod

    stray = frozenset({torch.ops.aten.relu.default})   # 激活，不是主算子
    original = mod.FUSION_TARGETS
    try:
        mod.FUSION_TARGETS = stray
        with pytest.raises(ValueError, match="未登记的算子"):
            mod._validate_fusion_targets()
    finally:
        mod.FUSION_TARGETS = original
