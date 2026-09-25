"""融合条件表单点维护：两个 pass 引用的必须是契约里的同一份对象。

方案 4.10：主算子与可折激活原先在 `fuse.py`、`fuse_pim.py` 各存一份，改一处
另外几处不会跟着变。这里钉住「两边拿到的就是 `contracts/fusion_contract.py`
里的那个对象」（is 同一性，不是内容相等），并逐元素比对合并前的原表。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts import fusion_contract
from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph import fuse, fuse_pim


# 合并前的原表，逐元素抄自 `graph/fuse.py`（FUSION_TARGETS / ACTIVATIONS）
# 与 `graph/fuse_pim.py`（ACTIVATION_HOSTS / FUSABLE_ACTIVATIONS）。
FUSE_TARGETS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    }
)
FUSE_ACTIVATIONS = {
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.gelu.default: "gelu",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.reciprocal.default: "reciprocal",
}
PIM_TARGETS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
    }
)
PIM_ACTIVATIONS = {
    # 小写（评审 20260923 的 P2-1）：下游 `_ACTIVATION_NAMES` 是小写键的表，
    # 写大写会绕过归一化——`"Silu"` 恰好就是 GML 要的拼写所以没暴露，
    # 换一个不巧合的就会写出对方解析器找不到的块名。
    torch.ops.aten.silu.default: "silu",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.gelu.default: "gelu",
}


@pytest.fixture(scope="module")
def llama_graph():
    """一层 llama2 的小图（真实结构、小维度），未融合。"""
    from tests.test_partition import _export_random_llama

    return _export_random_llama()


@pytest.fixture(scope="module")
def fused_graph(llama_graph):
    """跑过 `fuse_for_pim` 的副本（图是共享的，先深拷再折）。"""
    gm = copy.deepcopy(llama_graph)
    return gm, fuse_pim.fuse_for_pim(gm)


def test_both_passes_reference_the_contract_objects() -> None:
    """is 同一性：拿到的是契约里的那个对象，不是各自复制的等价副本。"""
    assert fuse.FUSION_TARGETS is fusion_contract.FUSION_TARGETS
    assert fuse.ACTIVATIONS is fusion_contract.ACTIVATIONS
    assert fuse_pim.GATE_TARGETS is fusion_contract.GATE_TARGETS
    assert fuse_pim.GATE_ACTIVATIONS is fusion_contract.GATE_ACTIVATIONS


def test_the_old_per_pass_constants_are_gone() -> None:
    """两份旧表必须删干净，留着就是第二份真源。"""
    assert not hasattr(fuse_pim, "ACTIVATION_HOSTS")
    assert not hasattr(fuse_pim, "FUSABLE_ACTIVATIONS")


def test_main_op_table_is_unchanged() -> None:
    """主算子表取两份原表里覆盖面更广的那份（matmul 类 + eltwise 类）。"""
    assert fusion_contract.FUSION_TARGETS == FUSE_TARGETS
    assert fusion_contract.FUSION_TARGETS >= PIM_TARGETS


def test_generic_activation_table_is_unchanged() -> None:
    assert fusion_contract.ACTIVATIONS == FUSE_ACTIVATIONS


def test_gate_tables_are_unchanged() -> None:
    assert fusion_contract.GATE_TARGETS == PIM_TARGETS
    assert fusion_contract.GATE_ACTIVATIONS == PIM_ACTIVATIONS


def test_gate_targets_are_the_matmul_subset_of_the_main_ops() -> None:
    """门控主算子就是主算子表里的 matmul 类，少的正是 eltwise 那两个。"""
    assert fusion_contract.GATE_TARGETS < fusion_contract.FUSION_TARGETS
    assert fusion_contract.FUSION_TARGETS - fusion_contract.GATE_TARGETS == {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    }


def test_silu_folds_only_into_gate_projections() -> None:
    """silu 只列在门控表里：通用表没有它，而门控主算子只有 matmul 类。"""
    assert torch.ops.aten.silu.default in fusion_contract.GATE_ACTIVATIONS
    assert torch.ops.aten.silu.default not in fusion_contract.ACTIVATIONS

    for eltwise in (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor):
        assert eltwise not in fusion_contract.GATE_TARGETS


def test_silu_really_folds_into_the_gate_projection(fused_graph) -> None:
    """真实 llama2 图上唯一那个 silu 折进 gate 投影（`mlp.gate_proj.weight`）。"""
    gm, report = fused_graph
    assert report.activations == 1

    host = next(n for n in gm.graph.nodes if FUSED_TAIL_META_KEY in n.meta)
    assert host.target in fusion_contract.GATE_TARGETS
    assert host.meta[FUSED_TAIL_META_KEY].activation == "silu"
    assert any(
        "gate_proj" in str(arg.target)
        for arg in host.all_input_nodes
        if arg.op == "get_attr"
    )


def test_the_generic_pass_leaves_silu_alone(llama_graph) -> None:
    """通用表没有 silu，所以 `fuse_graph` 不吃它 —— 走通用路径的图仍发独立节点。"""
    gm = copy.deepcopy(llama_graph)
    fuse.fuse_graph(gm)

    silus = [n for n in gm.graph.nodes
             if n.op == "call_function" and n.target is torch.ops.aten.silu.default]
    assert len(silus) == 1
    assert FUSED_TAIL_META_KEY not in silus[0].meta


def test_rms_norm_stays_independent(fused_graph) -> None:
    """rsqrt 两张表都没有：RMSNorm 折成自己的独立节点，不折进 matmul。"""
    for table in (fusion_contract.ACTIVATIONS, fusion_contract.GATE_ACTIVATIONS):
        assert torch.ops.aten.rsqrt.default not in table

    gm, report = fused_graph
    assert report.rms_norms == 3

    rsqrt = [n for n in gm.graph.nodes
             if n.op == "call_function"
             and n.target is torch.ops.aten.rsqrt.default]
    assert rsqrt
    assert all(n.meta.get(fuse_pim.ABSORBED_META_KEY) is True for n in rsqrt)


def test_every_activation_name_is_normalised_downstream() -> None:
    """两张表的每个值都要能被 `_ACTIVATION_NAMES` 命中（评审 20260923 的 P2-1）。

    `from_fx.py` 那行是 `_ACTIVATION_NAMES.get(name, name)`：命不中就**原样透出**。
    门控表原来写 `Silu` / `Relu` / `Gelu`，三个都命不中，靠 `Silu` 恰好等于 GML
    要的拼写才对——归一化那一层实际被绕过了。这条断言让「靠巧合」变成「有判据」：
    多一个拼写不巧合的激活（比如 `LeakyRelu`）会在这里失败，而不是在对方的
    解析器里静默找不到块名。
    """
    from gml_bridge.from_fx import _ACTIVATION_NAMES

    values = (set(fusion_contract.ACTIVATIONS.values())
              | set(fusion_contract.GATE_ACTIVATIONS.values()))
    assert values, "两张表不该是空的"
    assert all(v == v.lower() for v in values), (
        f"激活名必须小写: {sorted(v for v in values if v != v.lower())}")
    missing = sorted(v for v in values if v not in _ACTIVATION_NAMES)
    assert not missing, (
        f"这些激活名下游归一化表里没有，会原样透出: {missing}")


# --- 与 FlagTree 侧 `isFusionTarget` 对齐（评审 20260923 的 P2-2）------------
#
# 方案 §4.10 要求两边描述同一张融合表。C++ 读不了 Python，所以原来只有
# `FuseActivation.cpp` 上方一句「改一边必须改另一边」的注释 —— 注释提醒人类，
# 不是判据。这里读那份 C++ 源码的文本，把它的 op 集合与本仓的表对起来。

# 本仓 aten 目标 → FlagTree 的 op 类名。`conv` 在 C++ 那边也是融合目标，但
# llama2 图上没有卷积，所以本仓的表里没有它 —— 这不是不一致，是覆盖面之差，
# 所以下面只要求「本仓的每一项都在 C++ 那边」，不要求反向相等。
_ATEN_TO_PIM_OP = {
    "addmm.default": "MatmulOp",
    "linear.default": "MatmulOp",
    "mm.default": "MatmulOp",
    "add.Tensor": "EltwiseOp",
    "mul.Tensor": "EltwiseOp",
}


def _cpp_fusion_targets() -> set[str]:
    """从 `FuseActivation.cpp` 的 `isFusionTarget` 里抽出 op 类名集合。"""
    import re

    from genesim_bridge.paths import flagtree_source

    source = (flagtree_source()
              / "lib/Dialect/TritonPIM/Transforms/FuseActivation.cpp")
    text = source.read_text()
    match = re.search(r"isFusionTarget\(Operation \*op\)\s*\{\s*"
                      r"return\s+isa<([^>]*)>\(op\);", text)
    assert match, (
        f"没能在 {source} 里认出 `isFusionTarget` 的 isa<...> 列表。"
        f"那个函数改了形状，这条测试要跟着改 —— 而不是删掉")
    return {name.strip() for name in match.group(1).split(",")}


def test_main_op_table_agrees_with_flagtrees_isfusiontarget() -> None:
    """本仓每个融合主算子，在 FlagTree 的 `isFusionTarget` 里都要是融合目标。

    不一致的后果是单向的静默：图编译器把激活折进了某个主算子，而
    `-pim-fuse-activation` 不认那个 op，于是 pass 跑过「无事可做」，
    `activation` 属性还在、GML 也照发 —— 但两边对「什么能持有折入的激活」
    的判断已经分叉，下一个改动会踩在这个缝上。
    """
    cpp = _cpp_fusion_targets()
    assert cpp, "C++ 那边的融合目标集合不该是空的"

    missing = sorted(
        f"{name} -> {_ATEN_TO_PIM_OP[name]}"
        for name in (str(t).split("aten.")[-1]
                     for t in fusion_contract.FUSION_TARGETS)
        if _ATEN_TO_PIM_OP[name] not in cpp
    )
    assert not missing, (
        f"本仓这些主算子在 FlagTree 的 isFusionTarget 里不是融合目标: {missing}；"
        f"C++ 侧认的是 {sorted(cpp)}")


def test_the_aten_to_pim_op_map_covers_the_whole_table() -> None:
    """上面那张映射表必须覆盖 `FUSION_TARGETS` 的每一项。

    漏一项，上面那条测试会跳过它而不是失败 —— 那正是「判据本身不可能失败」。
    """
    names = {str(t).split("aten.")[-1] for t in fusion_contract.FUSION_TARGETS}
    assert names <= set(_ATEN_TO_PIM_OP), (
        f"这些主算子没有对应的 FlagTree op 名，上面那条测试会静默跳过它们: "
        f"{sorted(names - set(_ATEN_TO_PIM_OP))}")


def test_flagtree_fusion_targets_match_the_contract() -> None:
    """FlagTree 的 `isFusionTarget` 必须覆盖契约里的主算子，并保留 conv。

    本仓那张表没有 conv，是因为目标模型没有卷积，不是禁止 conv 折激活。
    设计要求算子编译器侧保持 `Matmul/Conv/Eltwise` 三个，少了 conv 会让
    `pim.conv` 后的激活融不进去。C++ 读不到 Python，所以扫源码核对。
    """
    import re

    cpp = Path("/media/disk/fengjingge/src/flagOS/flagOS-installers/FlagTree"
               "/lib/Dialect/TritonPIM/Transforms/FuseActivation.cpp")
    text = cpp.read_text(encoding="utf-8")
    match = re.search(r"isFusionTarget[^{]*\{[^}]*isa<([^>]*)>", text, re.S)
    assert match, "没找到 isFusionTarget 的 isa 列表"
    ops = {name.strip() for name in match.group(1).split(",")}
    assert ops == {"MatmulOp", "ConvOp", "EltwiseOp"}, (
        f"与设计不一致：{sorted(ops)}")
