"""端到端：真实 Llama2-7B → 图编译 → 算子编译器展开相位。

这是「模型 → 图编译器 → 算子编译器」整条链路的验收。与
`test_oplevel_emitter.py` 的手工图不同，这里走真实权重和真实的六个融合
pass，所以能抓到「形状口径对了但真实图里有第三种情况」这类问题——
K 路 RoPE 同时带 DQ 就是这么发现的。

需要真实模型 + 带 PIM pass 的 triton-opt，缺任一则跳过。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import flagtree_prefix, llama2_7b_model_dir
from opcompiler_bridge.oplevel_emitter import emit_oplevel_mlir
from opcompiler_bridge.phase_plan import parse_phase_plans

MODEL_DIR = llama2_7b_model_dir(required=False)
SEQ_LEN = 16
# llama2-7B 的 32 头。逐头展开后 softmax / score-DQ 各 32 个。
NUM_HEADS = 32


def _triton_opt() -> Path | None:
    path = flagtree_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"
    return path if path.is_file() else None


def _has_expand_phases() -> bool:
    binary = _triton_opt()
    if binary is None:
        return False
    proc = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
    return "--pim-expand-phases" in proc.stdout


pytestmark = [
    pytest.mark.skipif(
        MODEL_DIR is None or not MODEL_DIR.is_dir(),
        reason="需要在 paths.json 配置 llama2_7b_model_dir",
    ),
    pytest.mark.skipif(
        not _has_expand_phases(),
        reason="当前 triton-opt 没有 -pim-expand-phases，需重跑 0-install-flagtree.sh",
    ),
]


@pytest.fixture(scope="module")
def expanded():
    """跑一次完整链路，返回 (发射报告, 相位计划)。

    module 级 fixture：加载 7B 权重要十几秒，几条测试共用一次。
    """
    from gml_bridge.export import export_graph
    from runtime.compile import export_annotated_graph
    from scripts.export_gml import _load_model

    model = _load_model(1)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, SEQ_LEN, position_ids, dtype=torch.float32)
    export_graph(gm)

    report = emit_oplevel_mlir(gm)

    binary = _triton_opt()
    assert binary is not None
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(report.text)
        path = handle.name
    try:
        proc = subprocess.run(
            [str(binary), path, "-pim-expand-phases"],
            capture_output=True, text=True,
        )
    finally:
        Path(path).unlink()

    assert proc.returncode == 0, f"triton-opt 失败:\n{proc.stderr[:3000]}"
    return report, parse_phase_plans(proc.stdout)


def test_nothing_is_skipped(expanded) -> None:
    """真实图里每个待展开算子都发得出去。

    跳过就意味着某类形状没处理到，而那会静默少一批相位。
    """
    report, _ = expanded
    assert report.skipped == []


def test_operator_counts_match_real_graph(expanded) -> None:
    """算子个数：DQ 38、Softmax 32、RoPE 2。

    Softmax 与逐头 score-DQ 各 32 个（每头一个）；RoPE 两条（Q 路与 K 路）。
    线性 DQ：q/k/v 共用、gate/up 共用，再加 o/down/lm_head 与 Q 路，共 6 条非逐头。
    """
    report, _ = expanded
    assert len(report.by_kind("softmax")) == NUM_HEADS
    assert len(report.by_kind("rope")) == 2
    # 38 = 32 个逐头 score DQ + 6 条非逐头
    assert len(report.by_kind("dq")) == 38


def test_every_op_expands_to_expected_phase_count(expanded) -> None:
    """DQ 4 相、Softmax 5 相、RoPE 3 相，逐个核对，不允许有例外。"""
    report, plans = expanded
    mismatches = [
        (op.func, op.kind, plans[op.func].count if op.func in plans else None,
         op.expected_phases)
        for op in report.ops
        if op.func not in plans or plans[op.func].count != op.expected_phases
    ]
    assert mismatches == [], f"相位数不符: {mismatches[:5]}"


def test_reduction_phases_land_on_vector_unit(expanded) -> None:
    """归约走 VPU、非线性走 CSTL —— 引擎单功能的硬约束。"""
    report, plans = expanded
    for op in report.by_kind("softmax"):
        assert plans[op.func].units() == ["vpu", "cstl", "vpu", "cstl", "cstl"]
    for op in report.by_kind("dq"):
        assert plans[op.func].units()[0] == "vpu"


def test_dq_group_widths_follow_tensors(expanded) -> None:
    """分组宽度按张量变化，不是常量。

    实测三种：hidden 512 组、MLP 1376 组、attention scores 1 组。
    每组一个 fp16 统计量，所以 phase0 的字节数 = 组数 × 2。
    """
    report, plans = expanded
    group_bytes = {plans[op.func].phases[0].bytes for op in report.by_kind("dq")}
    assert group_bytes == {512 * 2, 1376 * 2, 1 * 2}


def test_rope_phases_are_force_consecutive(expanded) -> None:
    """RoPE 三连必须连续执行，中间结果不落 DDR。"""
    report, plans = expanded
    for op in report.by_kind("rope"):
        plan = plans[op.func]
        assert all(phase.force_consecutive for phase in plan.phases)
        assert [p.rotate_half for p in plan.phases] == [False, True, False]


def test_softmax_uses_absmax_free_reduction(expanded) -> None:
    """Softmax 的两个归约是 max 与 add；DQ 的才是 absmax。

    混了会让 softmax 的数值稳定化用错归约。
    """
    report, plans = expanded
    for op in report.by_kind("softmax"):
        assert plans[op.func].kinds() == [
            "max", "exp", "add", "reciprocal", "mul"]
    for op in report.by_kind("dq"):
        assert plans[op.func].phases[0].kind == "absmax"
