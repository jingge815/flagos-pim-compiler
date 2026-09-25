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
def graphs():
    """跑一次完整链路，返回 (整图报告, 相位计划, decode block 报告, decode GML)。

    module 级 fixture：加载 7B 权重要十几秒，几条测试共用一次。四份产物都由同一
    次加载导出：整图那份带 lm_head，decode block 那份把它裁掉后再导一遍，
    GML 留着给边形状那条测试（它要真实 7B 图，合成小图不改写槽位）。
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

    # decode block 要裁掉 lm_head：方案 1.2.4 的覆盖表里它只随 `--layers 1` 发，
    # 裁不裁差一个 DQ 锚点（37/36 对 38/37），7.3 的 36 是裁掉之后的口径。
    model.lm_head = torch.nn.Identity()
    block_gm = export_annotated_graph(model, SEQ_LEN, position_ids,
                                      dtype=torch.float32)
    block_artifact = export_graph(block_gm)
    return (report, parse_phase_plans(proc.stdout), emit_oplevel_mlir(block_gm),
            block_artifact)


@pytest.fixture(scope="module")
def expanded(graphs):
    """整图口径的 (发射报告, 相位计划)，给下面几条形状与相位测试用。"""
    report, plans, _, _ = graphs
    return report, plans


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


def test_reduction_phases_land_on_the_right_unit(expanded) -> None:
    """Softmax 求 max 走 VPU；DQ 相 0 是分组归约，走池化单元。"""
    report, plans = expanded
    for op in report.by_kind("softmax"):
        assert plans[op.func].units() == [
            "vpu", "activation", "vpu", "activation", "combiner"]
    for op in report.by_kind("dq"):
        assert plans[op.func].units()[0] == "pooling"


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


def _func_costs(text: str) -> dict[str, float]:
    """把展开后的 IR 按 `tt.func` 切开，逐条量出 flops。

    合起来量只有一个总数，看不出是哪个算子没计到成本——而漏计一个 softmax 与
    漏计一个 DQ 在总数上没区别。
    """
    from genesim_bridge.ir_cost import analyze_ir

    blocks: list[str] = []
    for line in text.splitlines():
        if line.startswith("  tt.func "):
            blocks.append(line)
        elif blocks:
            blocks[-1] += "\n" + line

    costs: dict[str, float] = {}
    for block in blocks:
        name = block.split("@")[1].split("(")[0]
        costs[name] = analyze_ir(block, name, (1,), {}, ir_level="pimir").flops
    return costs


def test_decode_block_mnemonic_counts_and_costs(graphs) -> None:
    """方案 7.3 的断言：decode 单层图上 Softmax 32、DQ 36、RoPE 2，且都有成本。

    数的是 `ir_cost` 看到的那份文本。`pim.dynamic_quant` 在文本里有 37 条，因为
    K 路 RoPE 的锚点同时要发 RoPE 与 DQ（`add_1__rope` 配 `add_1__dq`），最后那条
    是挂在 RoPE 上的尾段量化——方案 1.2.4 正文写作「37（36 DQ + 1 RoPE-DQ）」，
    7.3 表里的 36 是不含它的独立 DQ 锚点数。

    每一项 flops 都必须非零：成本为 0 的算子不会报错，只会让仿真少算一段。
    """
    from genesim_bridge.flagtree_driver import lower_oplevel_to_pimir
    from genesim_bridge.ir_cost import count_mnemonics

    _, _, block, _ = graphs
    counts = count_mnemonics(block.text)
    assert int(counts["pim.softmax"]) == NUM_HEADS
    assert int(counts["pim.rope"]) == 2

    # 37 条量化里只有 K 路 RoPE 的那条是尾段 DQ，其余 36 条才是独立 DQ 锚点。
    rope_names = {op.fx_name for op in block.by_kind("rope")}
    forward_dq = [op for op in block.by_kind("dq") if op.fx_name not in rope_names]
    assert int(counts["pim.dynamic_quant"]) == 37
    assert len(forward_dq) == 36

    costs = _func_costs(lower_oplevel_to_pimir(block.text))
    assert len(costs) == len(block.ops), "展开后少了一批算子"
    zero = sorted(name for name, flops in costs.items() if flops <= 0)
    assert zero == [], f"这些算子没算到 flops: {zero[:5]}"


def test_apath_dot_cost_is_not_zeroed() -> None:
    """方案 7.3 的断言：A 路 linear 的 `tt.dot` 成本不得变成 0。

    B 路那一轮改的是整算子级 IR 的计价，A 路走的还是 `tt.dot` + 显式 DMA 那条链。
    两边共用 `analyze_ir`，所以这里钉一条：linear 编译产物里的 `tt.dot` 仍按
    `2MNK` 记账，M=1 时也不例外。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    m, k, n = 1, 64, 128
    result = compile_op(OpCompileRequest(
        "linear", [(m, k), (n, k)], DEFAULT_HARDWARE_CONFIG, "float32"))
    assert "tt.dot" in result.pimir
    cost = analyze_ir(result.pimir, "linear", (1,), {}, ir_level="pimir")
    assert cost.flops == 2 * m * k * n, cost.flops
    assert cost.mram_traffic_bytes > 0


def test_edge_dims_are_compile_slots_not_the_export_seq_len(graphs) -> None:
    """真实 7B 图上，边的形状一条都不许是导出图的 seq_len，也不许是 `unknown`。

    评审 20260923 的 P0-1 / P0-2。原来的改写判据是「末维是 1 或 16 就换」，
    漏了序列轴不在末维的那一整类（`1x32x16x128` 的 16 在下标 2），实测 331 条边
    里 121 条仍是 16-token 口径、1 条是 `unknown`，而参考产物一条都没有。

    为什么这条必须在真实 7B 图上钉：改写只在图里看得到 hidden=4096 时启用
    （合成小图没有 decode 槽位这回事），所以合成图上这条永远成立、钉不住东西。

    为什么必须在这里而不是只看产物：声明与落盘是同一个来源（写盘侧读的就是
    这条边的 dims），两边一起错就自洽，尺寸自检结构上发现不了。
    """
    import re

    from contracts.compile_slots import DEFAULT_SLOTS

    _, _, _, artifact = graphs
    dims = re.findall(r'^\s*dims "([^"]*)"', artifact.text, re.M)
    assert dims, "这张图应当有带形状的边"
    assert "unknown" not in dims, "形状是可知的，不允许写 unknown"

    assert SEQ_LEN != DEFAULT_SLOTS.seq, "这条测试要求导出口径与槽位口径不同"
    leaked = sorted({d for d in dims if str(SEQ_LEN) in d.split("x")})
    assert not leaked, f"这些边仍是导出图的 {SEQ_LEN}-token 口径: {leaked}"
