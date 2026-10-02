"""P1-2：算子编译器的布局决策回传，并且真正参与下游产出。

既有反向通道（`PhaseSource`）只覆盖相位数据。这里把它推广到布局维度，
判据有三条：每个回传字段都有生产方与消费方、不带回传时产物与今天一致、
**改变回传值会改变下游产物**（变异测试——这条才证明通道真的接通了，
否则就是一个「有定义、无消费者」的字段）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.ir_payloads import (
    LayoutFeedback,
    layout_feedback_of_module,
    module_int_attrs,
)

SCAN_DIRS = ("gml_bridge", "memory", "genesim_bridge", "opcompiler_bridge",
             "contracts")


def _code_lines(text: str) -> list[str]:
    """去掉注释后的源码行（行号不动），免得注释里的字段名被算成调用点。"""
    lines = text.splitlines()
    out = []
    for line in lines:
        out.append(line.split("#", 1)[0] if not line.lstrip().startswith("#") else "")
    return out


def test_every_feedback_field_has_a_producer_and_a_consumer() -> None:
    """P1-2 的核心判据：不允许只写不读或只读不写的字段。

    这是防「有定义、无消费者」类腐化的执行点。`combine_mode` 曾经正是这个
    形态：枚举完整、五处调用都传空值占位、零消费者；它已在本轮补上消费方
    （`EltwiseOp::verify` 校验 skip_connection 必须是 add），不再作反例。
    仍然成立的同类例子是 `VpuParamsAttr` —— 只在 ODS 与手写测试里出现，
    全仓 `.cpp` 零引用。

    「消费方」认属性读取（`fb.wram_bytes_used`）。回传值当前只在 A 路
    （过 `-pim-tile-to-budget`）非空，B 路全 None 走静态分支 —— 所以判据是
    「代码里有读取点」，不是「每次跑都读到非空值」。
    """
    root = Path(__file__).parent.parent
    src = "\n".join("\n".join(_code_lines(p.read_text(encoding="utf-8")))
                    for d in SCAN_DIRS for p in (root / d).rglob("*.py"))
    for f in fields(LayoutFeedback):
        writes = len(re.findall(rf"\b{f.name}\s*=", src))
        reads = len(re.findall(rf"\.{f.name}\b(?!\s*=)", src))
        assert writes >= 1, f"{f.name} 没有生产方"
        assert reads >= 1, f"{f.name} 没有消费方（只写不读 = 死字段）"


def test_the_cost_model_consumes_the_feedback() -> None:
    """成本模型是回传的消费方：tile 与 WRAM 用量从载体来，不自己解析。

    原先 `genesim_bridge/ir_cost.py` 有一份自己的 `_module_int_attr` 正则，
    与回传通道是两条独立的提取路径 —— 通道那条零消费者（目标四不达标），
    而这条绕开了统一 IR（目标三不达标）。现在收成一条。
    """
    from genesim_bridge import ir_cost

    text = ir_cost.__file__ and Path(ir_cost.__file__).read_text(encoding="utf-8")
    assert "_module_int_attr" not in text, "私有正则还在，提取路径没收口"
    assert "layout_feedback_of_module(" in text, "成本模型没有消费回传载体"


def test_feedback_present_changes_the_product() -> None:
    """变异测试：改变回传值必须改变下游产物。

    这条才证明回传真的「生效」，而不只是「存在」。注入一份带 tile 与
    WRAM 用量的模块头，成本模型的产物就得跟着变；不变就说明消费点没接上，
    只是一个写了没人读的字段。
    """
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @kernel(%arg0: tensor<1x16xf16>) {\n'
            '    %0 = pim.eltwise %arg0, %arg0 {kind = #pim.eltwise<add>} '
            ': tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>\n'
            '    tt.return\n  }\n')
    bare = 'module attributes {pim.target = "pim:v1"} {\n' + body + '}\n'
    rich = ('module attributes {pim.target = "pim:v1", '
            '"pim.tile-m" = 1 : i64, "pim.tile-n" = 512 : i64, '
            '"pim.wram-bytes" = 65536 : i32, '
            '"pim.wram-bytes-used" = 33856 : i32} {\n' + body + '}\n')

    without = analyze_ir(bare, "k", (1,), {}, ir_level="oplevel")
    with_fb = analyze_ir(rich, "k", (1,), {}, ir_level="oplevel")

    # 没有回传：走静态分支，字段为 None（不造假值）。
    assert without.tile_m is None and without.wram_bytes_used is None
    # 有回传：产物带上算子编译器的实测决策。
    assert with_fb.tile_m == 1 and with_fb.tile_n == 512
    assert with_fb.wram_bytes_used == 33856
    assert with_fb.layout_feedback.tile_shape == (1, 512)
    assert with_fb.tile_m != without.tile_m, "注入回传后产物没变，消费点没接上"


def test_a_tile_over_the_wram_budget_is_reported() -> None:
    """回传说超了 WRAM 预算，成本模型要把这条记进 notes。

    这是回传值**改变结论**（而不只是被搬运）的那一处：同一份 IR，
    只改 `pim.wram-bytes-used` 一个数，产物里多出一条超限说明。
    """
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @kernel(%arg0: tensor<1x16xf16>) {\n'
            '    %0 = pim.eltwise %arg0, %arg0 {kind = #pim.eltwise<add>} '
            ': tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>\n'
            '    tt.return\n  }\n')

    def head(used: int) -> str:
        return ('module attributes {pim.target = "pim:v1", '
                '"pim.wram-bytes" = 65536 : i32, '
                f'"pim.wram-bytes-used" = {used} : i32}}' + " {\n" + body + "}\n")

    fits = analyze_ir(head(1024), "k", (1,), {}, ir_level="oplevel")
    over = analyze_ir(head(99999), "k", (1,), {}, ir_level="oplevel")
    assert not any("WRAM 超预算" in n for n in fits.notes)
    assert any("WRAM 超预算" in n for n in over.notes), over.notes
    assert over.layout_feedback.over_wram_budget


def test_the_carrier_is_the_contract_level_parser() -> None:
    """回传载体是契约层的解析函数，不是 `PhaseSource` 上的一份副本。

    原先 `PhaseSource` 带 `layout_back` 字段与 `layout()` 访问器，但那条路
    生产侧永远取不到值：这些属性只出现在 A 路（过 `-pim-tile-to-budget` 的
    `linear_kernel`，实测 26/26 份带、B 路 175 份全不带），而 `PhaseSource`
    只在 B 路产生 —— 结构完整、端到端恒为全 None，且生产代码零消费。
    按 CLAUDE.md「删优于加」删掉，载体收敛为这一处解析函数。
    """
    from opcompiler_bridge import phase_source

    text = Path(phase_source.__file__).read_text(encoding="utf-8")
    assert "layout_back" not in text, "已删的死载体又回来了"
    assert "LayoutFeedback" not in text, "回传载体不该再有第二处副本"


def test_the_parser_reads_the_module_dictionary_only() -> None:
    """属性只在 `module attributes {...}` 那一处算数。

    正文里同名的字符串不算 —— 照 `rtl_version_of` 的既有口径。
    """
    text = ('module attributes {"pim.dma-align" = 64 : i32, '
            '"pim.num-dpus" = 8 : i32} {\n'
            '  // "pim.wram-bytes-used" = 999 出现在注释里\n'
            '}\n')
    attrs = module_int_attrs(text)
    assert attrs["pim.dma-align"] == 64
    assert attrs["pim.num-dpus"] == 8
    assert "pim.wram-bytes-used" not in attrs


def test_a_bare_module_head_yields_an_empty_feedback() -> None:
    """B 路模块头只有 `pim.target`：合法的「算子编译器没意见」，不是错误。"""
    assert module_int_attrs('module attributes {pim.target = "pim:v1"} {\n}\n') == {}
    assert layout_feedback_of_module(
        'module attributes {pim.target = "pim:v1"} {\n}\n') == LayoutFeedback()


def test_the_tile_shape_needs_both_halves() -> None:
    """`tile-m` / `tile-n` 缺一个就是没给，不拿单边凑一个形状。"""
    head = 'module attributes {{{}}} {{\n}}\n'
    both = layout_feedback_of_module(
        head.format('"pim.tile-m" = 128 : i64, "pim.tile-n" = 64 : i64'))
    half = layout_feedback_of_module(head.format('"pim.tile-m" = 128 : i64'))
    assert both.tile_shape == (128, 64)
    assert half.tile_shape is None


def test_the_feedback_reaches_the_simulation_input() -> None:
    """回传的 tile 要真的走到仿真输入构造（需求目标四的第三个落点）。

    链路：`analyze_ir` 读回传 → `cost.tile_n` → `_measure_kernel_tile_n`
    写 sidecar 的 `kernel_tile_n` → GeneSim 的 `Operator.kernel_tile_size`
    替掉 `conf/sim.yaml` 里拍下的 `tile_size` 常量。

    这里只钉住本仓那一段（`cost.tile_n` 来自回传而非另一条正则），
    GeneSim 侧的消费由它自己的 `tests/sim/` 守。
    """
    from genesim_bridge.ir_cost import analyze_ir

    body = ('  tt.func @linear_kernel(%arg0: tensor<1x16xf16>) {\n'
            '    %0 = pim.eltwise %arg0, %arg0 {kind = #pim.eltwise<add>} '
            ': tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>\n'
            '    tt.return\n  }\n')
    text = ('module attributes {pim.target = "pim:v1", '
            '"pim.tile-m" = 1 : i64, "pim.tile-n" = 512 : i64} {\n'
            + body + '}\n')
    cost = analyze_ir(text, "linear_kernel", (1,), {}, ir_level="oplevel")
    # sidecar 的 `kernel_tile_n` 取的就是这个值（`_measure_kernel_tile_n`）。
    assert cost.tile_n == 512
    assert cost.layout_feedback.tile_shape == (1, 512)


def test_a_footprint_over_the_declared_mram_budget_is_reported() -> None:
    """算子编译器自己声明的预算装不下它自己算出的单台占用 —— 必须说出来。

    `pim.mram-bytes` 是下发的每台预算的回显，`pim.placed-mram-bytes` 是那个
    pass 定完分块后算出的单台占用。两个数在同一份文本里自相矛盾，说明预算与
    分块不是同一套配置下算的。改动前这个字段只被搬进 `cost.mram_bytes_budget`
    就没人再看了 —— 与本轮反复强调的「回传必须改变下游产物」不符。
    """
    from genesim_bridge.ir_cost import analyze_ir

    def head(budget: int, placed: int) -> str:
        return ('module attributes {pim.target = "pim:v1", '
                f'"pim.mram-bytes" = {budget} : i64, '
                f'"pim.placed-mram-bytes" = {placed} : i64}}' + " {\n}\n")

    fits = analyze_ir(head(1 << 32, 4096), "k", (1,), {}, ir_level="pimir")
    over = analyze_ir(head(4096, 1 << 20), "k", (1,), {}, ir_level="pimir")
    assert not any("MRAM" in n for n in fits.notes), fits.notes
    assert any("MRAM" in n and "超" in n for n in over.notes), over.notes


def test_the_mram_budget_field_is_actually_read_from_the_feedback() -> None:
    """`mram_bytes` 必须来自回传载体，而不是另起一条正则。"""
    from genesim_bridge.ir_cost import analyze_ir

    text = ('module attributes {pim.target = "pim:v1", '
            '"pim.mram-bytes" = 8589934592 : i64} {\n}\n')
    cost = analyze_ir(text, "k", (1,), {}, ir_level="pimir")
    assert cost.mram_bytes_budget == 8589934592
    assert cost.layout_feedback.mram_bytes == 8589934592


def test_the_gml_feedback_comes_from_the_a_path(monkeypatch) -> None:
    """GML 的回传宽度必须来自真正过 `-pim-tile-to-budget` 的那路。

    `pim.placed-elem-bytes` 只有 A 路产出，而 `phase_source.expanded` 是 B 路
    文本，里面永远没有这个属性。从它读，`_placed_widths_of` 恒为 None，
    `_stamp_dtypes` 恒走无回传分支。

    这里不跑真编译：把 A 路探测换成一份带 `placed_elem_bytes` 的回传，
    GML 入口必须把它收进去。收不到就说明生产方还是空的。
    """
    import torch
    from torch.fx import GraphModule

    from contracts.ir_payloads import PlacementBack
    from contracts.op_contract import PIMHardwareConfig
    from gml_bridge.export import FusionReport, serialize_gml

    class M(torch.nn.Module):
        def forward(self, x, w):
            return torch.nn.functional.linear(x, w)

    x = torch.zeros(1, 4)
    w = torch.zeros(4, 4)
    gm = torch.export.export(M(), (x, w), strict=True).module()
    seen = {}

    def fake_probe(nodes, *, hardware, widths=None):
        seen["called"] = True
        if widths is not None:
            widths["Gemm"] = PlacementBack(placed_elem_bytes=1)
        return 0

    monkeypatch.setattr("runtime.compile.peak_kernel_mram_bytes", fake_probe)
    hw = PIMHardwareConfig(num_dpus=1, num_tasklets=4, mram_bytes_per_dpu=1 << 20,
                           wram_bytes_per_dpu=65536, dma_align=64)
    serialize_gml(gm, FusionReport(0, 0, 0), hardware=hw)
    assert seen.get("called"), "GML 入口没有走 A 路探测，回传生产方是空的"

    # 真实入口是 export_graph：scripts/export_gml.py 走它，不传 hardware。
    # 它不探测，回传就到不了产物。
    from contracts.unified_ir import STAGE_PARTITIONED, mark_stage
    from gml_bridge.export import export_graph
    mark_stage(gm, STAGE_PARTITIONED)
    seen.clear()
    export_graph(gm)
    assert seen.get("called"), "export_graph 没有走 A 路探测，真实导出收集不到回传"
