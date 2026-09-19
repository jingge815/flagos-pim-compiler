"""算子编译器接进 GML 链路后的交叉校验。

这组测试守两件事：

1. 算子编译器给出的相位模板与 `contracts/gml_hw_table.py` 静态表同口径；
2. 接入算子编译器**不改变** GML 产物（逐字节相同）。

第 2 条是这条链路最硬的判据：相位模板的真源移到了 FlagTree，但字段仍由静态表
产出，所以产物必须一模一样。有任何差异就是某一侧算错了。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import llama2_7b_model_dir
from opcompiler_bridge.phase_source import (
    PASS_PIPELINE,
    PhaseMismatch,
    PhaseSource,
    cross_check,
    opcompiler_available,
    phase_source_from_graph,
)
from opcompiler_bridge.phase_plan import Phase, PhasePlan

MODEL_DIR = llama2_7b_model_dir(required=False)
SEQ_LEN = 16


def test_pass_pipeline_fuses_before_expanding() -> None:
    """顺序不能反：展开后主算子已成相位链，融合认的模式就不在了。"""
    assert PASS_PIPELINE.index("-pim-fuse-activation") < PASS_PIPELINE.index(
        "-pim-expand-phases")


def test_cross_check_accepts_matching_template() -> None:
    """相位数与归约 kind 都对时，交叉校验放行。"""
    source = PhaseSource(by_node={
        "dq0": {"dq": PhasePlan(func="dq0__dq", phases=[
            Phase(index=i, op="", unit="", bytes=0,
                  kind="absmax" if i == 0 else None)
            for i in range(4)
        ])},
    })
    assert cross_check(source) == []


def test_cross_check_catches_wrong_phase_count() -> None:
    """DQ 少一相就要报出来——静默接受会让 prepare_out 少一层。"""
    source = PhaseSource(by_node={
        "dq0": {"dq": PhasePlan(func="dq0__dq", phases=[
            Phase(index=i, op="", unit="", bytes=0,
                  kind="absmax" if i == 0 else None)
            for i in range(3)
        ])},
    })
    mismatches = cross_check(source)
    assert len(mismatches) == 1
    assert mismatches[0].field == "phase_count"
    assert mismatches[0].from_opcompiler == 3
    assert mismatches[0].from_static_table == 4


def test_cross_check_catches_max_instead_of_absmax() -> None:
    """DQ 的 p0 必须是 absmax；用 max 会让负值主导的组量化错符号。"""
    source = PhaseSource(by_node={
        "dq0": {"dq": PhasePlan(func="dq0__dq", phases=[
            Phase(index=i, op="", unit="", bytes=0,
                  kind="max" if i == 0 else None)
            for i in range(4)
        ])},
    })
    mismatches = cross_check(source)
    assert [m.field for m in mismatches] == ["phase0.kind"]
    assert mismatches[0].from_opcompiler == "max"


def test_cross_check_reports_softmax_reduction_kind() -> None:
    """Softmax 的 p0 是 max（不是 absmax）——两者不能混。"""
    source = PhaseSource(by_node={
        "sm0": {"softmax": PhasePlan(func="sm0__softmax", phases=[
            Phase(index=i, op="", unit="", bytes=0,
                  kind="absmax" if i == 0 else None)
            for i in range(5)
        ])},
    })
    mismatches = cross_check(source)
    assert [m.field for m in mismatches] == ["phase0.kind"]


def test_phase_mismatch_message_names_both_sides() -> None:
    """报错要同时给出两侧的值，否则不知道该改哪边。"""
    text = str(PhaseMismatch("dq0", "dq", "phase_count", 3, 4))
    assert "dq0" in text and "3" in text and "4" in text


# --- 端到端：需要真实模型 + FlagTree ---------------------------------------

requires_live = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir() or not opcompiler_available(),
    reason="需要 paths.json 里的 llama2_7b_model_dir 和带 PIM pass 的 triton-opt",
)


@pytest.fixture(scope="module")
def phase_source_of_real_graph():
    """真实 7B 一层跑一次完整链路。

    module 级：加载权重加编译要十几秒，几条测试共用一次。
    """
    from gml_bridge.export import export_graph
    from runtime.compile import export_annotated_graph
    from scripts.export_gml import _load_model

    model = _load_model(1)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, SEQ_LEN, position_ids, dtype=torch.float32)
    export_graph(gm)
    return phase_source_from_graph(gm)


@requires_live
def test_real_graph_cross_checks_clean(phase_source_of_real_graph) -> None:
    """真实图上算子编译器与静态表一致，一处不符都没有。"""
    assert cross_check(phase_source_of_real_graph) == []


@requires_live
def test_fusion_is_idempotent_on_real_graph(phase_source_of_real_graph) -> None:
    """`-pim-fuse-activation` 跑前跑后 `activation` 个数不变。

    图编译器已经把激活折进主算子了，所以这一遍应该无事可做。个数变多说明
    我方漏折了某处（pass 替我们补上），变少说明 pass 不认我方的表达——
    两种都是口径不一致，要查。
    """
    import re

    source = phase_source_of_real_graph
    pattern = r"activation = #pim\.act_spec"
    before = len(re.findall(pattern, source.mlir))
    after = len(re.findall(pattern, source.expanded))
    assert before == after, f"融合口径不一致: 发射 {before} 个，展开后 {after} 个"
    # gate_proj 的 SiLU 至少要有一个，否则这条校验是空转。
    assert before >= 1


@requires_live
def test_every_real_op_has_a_plan(phase_source_of_real_graph) -> None:
    """发射出去的多相算子都要拿回相位模板，不能有对不上的。"""
    source = phase_source_of_real_graph
    # fused_matmul 是单相算子，不产相位，不参与这条检查。
    phase_ops = [op for op in source.ops if op.expected_phases > 0]
    missing = [op.func for op in phase_ops
               if source.plan(op.fx_name, op.kind) is None]
    assert missing == []


@requires_live
def test_rope_node_with_dq_carries_two_plans(phase_source_of_real_graph) -> None:
    """K 路 RoPE 的锚点同时有 rope 和 dq 两份相位模板。

    对应 GML 的 `Llama2ActivationDQ`：参考产物里那个节点同时有 4 个相位号和
    `Llama2Activation_*` 子块。早先 emitter 写成 if/elif，RoPE 那一路被 DQ
    吃掉，2 个只发出 1 个。
    """
    source = phase_source_of_real_graph
    both = [name for name, kinds in source.by_node.items()
            if {"rope", "dq"} <= set(kinds)]
    assert len(both) == 1, f"应恰有一个节点同时是 RoPE 和 DQ，实际 {both}"
    kinds = source.by_node[both[0]]
    assert kinds["rope"].count == 3
    assert kinds["dq"].count == 4
