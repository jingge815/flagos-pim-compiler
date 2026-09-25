"""算子编译器给出的相位模板要与 GML 侧静态表一致。

这组测试是「相位由谁决定」的交叉校验。`contracts/gml_hw_table.py` 的
`DQ_PHASES` / `SOFTMAX_PHASES` / `ROPE_UNITS` 目前仍是 GML 的字段来源，而
FlagTree 的 `-pim-expand-phases` 是相位结构的新真源。两边对不上就说明其中
一处退化了——这正是这里要拦住的。

不跑 triton-opt：用一份固化的 pass 输出（下面的 `_EXPANDED_*`，逐字取自
`triton-opt test/Dialect/TritonPIM/expand_phases.mlir -pim-expand-phases`），
这样没装 FlagTree 也能跑，且 pass 改了输出格式这里会失败——正是想要的。
真实端到端调用另见 `test_phase_plan_live.py`（需要 FlagTree）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.gml_quant import PHASE_COUNTS
from contracts import gml_hw_table as hw_table
from opcompiler_bridge.phase_plan import parse_phase_plans

_NORMALIZE = """
module {
  tt.func @rms(%x: tensor<1x128x4096xf16>, %g: tensor<4096xf16>, %e: tensor<1xf32>) {
    %y = pim.normalize %x, %g eps %e {axis = 1 : i64, rmsNorm,
        vpuParams = #pim.vpu_params<axis = -1, useScaling = false>}
       : tensor<1x128x4096xf16>, tensor<4096xf16> eps tensor<1xf32> -> tensor<1x128x4096xf16>
    tt.return
  }
}
"""


def test_vpu_params_default_to_the_measured_values_when_omitted() -> None:
    """空体 `#pim.vpu_params<>` 也要读出默认值。

    MLIR 打印时省略等于默认的参数，展开后的 IR 里就是空体。默认值
    （axis=-1、不缩放）恰好是参考产物的实测值，读成「没有」会让 GML
    退回图编译器写死的常量。
    """
    text = _NORMALIZE.replace(
        "vpuParams = #pim.vpu_params<axis = -1, useScaling = false>",
        "vpuParams = #pim.vpu_params<>")
    plan = parse_phase_plans(text)["rms"]
    assert plan.op_attrs.get("vpu-axis") == -1
    assert plan.op_attrs.get("use-scaling") == 0
    """`pim.normalize` 的 `vpuParams` 要能读回，不能只写不读。

    它是单相算子，没有 `pim.phase`，硬件域走 `op_attrs`。读不回就意味着
    GML 的 `Vpu_Axis` / `Use_Scaling` 只能继续由图编译器硬编码。
    """
    plans = parse_phase_plans(_NORMALIZE)
    plan = plans["rms"]
    assert plan.op_attrs.get("vpu-axis") == -1
    assert plan.op_attrs.get("use-scaling") == 0
_EXPANDED_DQ = """
module {
  tt.func @dq_four_phases(%arg0: tensor<1x4096xf16>, %arg1: tensor<32xf16>, %arg2: tensor<1x4096x!tt.ptr<i8>>) {
    %0 = pim.global_pool %arg0 {groupSize = 128 : i64, kind = #pim.pool_kind<absmax>, phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu>], unit = #pim.unit<vpu>} : tensor<1x4096xf16> -> tensor<1x32xf16>
    %1 = pim.lut %0 {activationMode = 1 : i64, flpMantisa = 3 : i64, flpMaxExp = 17 : i64, flpMinExp = 10 : i64, fpsu = #pim.fpsu_spec<mode = floating_point>, kind = #pim.activation<identity>, phases = [#pim.phase_spec<index = 1, bytes = 64, unit = activation, reads = [0]>], specialOperators = 0 : i64, unit = #pim.unit<activation>} : tensor<1x32xf16> -> tensor<1x32xf16>
    %2 = pim.lut %0 {activationMode = 0 : i64, flpMantisa = 0 : i64, flpMaxExp = 15 : i64, flpMinExp = 15 : i64, fpsu = #pim.fpsu_spec<mode = floating_point>, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 2, bytes = 64, unit = activation, reads = [0]>], purpose = #pim.transpose_purpose<purpose = layout_reorder, cardValue = 1>, specialOperators = 4 : i64, unit = #pim.unit<activation>} : tensor<1x32xf16> -> tensor<1x32xf16>
    %3 = pim.quantize %arg0, %2 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, kantorBlocks = [#pim.kantor_block<id = "A", mode = fp2int_converter>]>, fpsu = #pim.fpsu_spec<mode = floating_point>, kantor = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>, phases = [#pim.phase_spec<index = 3, bytes = 4096, unit = kantor>], spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>, unit = #pim.unit<kantor>} : tensor<1x4096xf16>, tensor<1x32xf16> -> tensor<1x4096xi8>
    tt.store %arg2, %3 : tensor<1x4096x!tt.ptr<i8>>
    tt.return
  }
}
"""

_EXPANDED_SOFTMAX = """
module {
  tt.func @softmax_five_phases(%arg0: tensor<1x1024xf16>, %arg1: tensor<1x1024x!tt.ptr<f16>>) {
    %0 = pim.reduce_axis %arg0 {axis = 1 : i64, kind = #pim.eltwise<max>, phases = [#pim.phase_spec<index = 0, bytes = 2, unit = vpu>], unit = #pim.unit<vpu>} : tensor<1x1024xf16> -> tensor<1x1xf16>
    %1 = pim.eltwise %arg0, %0 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, kind = #pim.eltwise<sub>, unit = #pim.unit<fpsu>} : tensor<1x1024xf16>, tensor<1x1xf16> -> tensor<1x1024xf16>
    %2 = pim.lut %1 {activationMode = 0 : i64, flpMantisa = 3 : i64, flpMaxExp = 16 : i64, flpMinExp = 9 : i64, fpsu = #pim.fpsu_spec<mode = floating_point>, kind = #pim.activation<exp>, phases = [#pim.phase_spec<index = 1, bytes = 2048, unit = activation>], purpose = #pim.transpose_purpose<purpose = layout_reorder, cardValue = 1>, specialOperators = 0 : i64, unit = #pim.unit<activation>} : tensor<1x1024xf16> -> tensor<1x1024xf16>
    %3 = pim.reduce_axis %2 {axis = 1 : i64, fpsu = #pim.fpsu_spec<mode = floating_point>, kind = #pim.eltwise<add>, phases = [#pim.phase_spec<index = 2, bytes = 4, unit = vpu>], unit = #pim.unit<vpu>} : tensor<1x1024xf16> -> tensor<1x1xf32>
    %4 = pim.lut %3 {activationMode = 0 : i64, flpMantisa = 0 : i64, flpMaxExp = 15 : i64, flpMinExp = 15 : i64, fpsu = #pim.fpsu_spec<mode = floating_point_32>, kind = #pim.activation<reciprocal>, phases = [#pim.phase_spec<index = 3, bytes = 2, unit = activation>], purpose = #pim.transpose_purpose<purpose = layout_reorder, cardValue = 2>, specialOperators = 4 : i64, unit = #pim.unit<activation>} : tensor<1x1xf32> -> tensor<1x1xf16>
    %5 = pim.eltwise %2, %4 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, kind = #pim.eltwise<mul>, phases = [#pim.phase_spec<index = 4, bytes = 2048, unit = combiner, reads = [1, 3]>], unit = #pim.unit<combiner>} : tensor<1x1024xf16>, tensor<1x1xf16> -> tensor<1x1024xf16>
    tt.store %arg1, %5 : tensor<1x1024x!tt.ptr<f16>>
    tt.return
  }
}
"""

_EXPANDED_ROPE = """
module {
  tt.func @rope_three_phases(%arg0: tensor<1x4096xf16>, %arg1: tensor<1x4096xf16>, %arg2: tensor<1x4096xf16>, %arg3: tensor<1x4096x!tt.ptr<f16>>) {
    %0 = pim.eltwise %arg0, %arg1 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, kantorBlocks = [#pim.kantor_block<id = "A", mode = elementwise_mul_fp16>]>, fpsu = #pim.fpsu_spec<mode = floating_point>, kantor = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>, kind = #pim.eltwise<mul>, phases = [#pim.phase_spec<index = 0, bytes = 8192, unit = kantor, forceConsecutive = true>], unit = #pim.unit<kantor>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    %1 = pim.eltwise %arg0, %arg2 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, kantorBlocks = [#pim.kantor_block<id = "A", mode = elementwise_mul_fp16>]>, fpsu = #pim.fpsu_spec<mode = floating_point>, kantor = #pim.kantor_spec<mode = elementwise_mul_fp16, cardValue = 5>, kind = #pim.eltwise<mul>, phases = [#pim.phase_spec<index = 1, bytes = 8192, unit = kantor, forceConsecutive = true>], rotateHalf, unit = #pim.unit<kantor>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    %2 = pim.eltwise %0, %1 {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, fpsu = #pim.fpsu_spec<mode = floating_point>, kantor = #pim.kantor_spec<mode = off, cardValue = 0>, kind = #pim.eltwise<add>, phases = [#pim.phase_spec<index = 2, bytes = 8192, unit = kantor, forceConsecutive = true>], unit = #pim.unit<kantor>} : tensor<1x4096xf16>, tensor<1x4096xf16> -> tensor<1x4096xf16>
    tt.store %arg3, %2 : tensor<1x4096x!tt.ptr<f16>>
    tt.return
  }
}
"""


def test_dq_phase_count_matches_static_table() -> None:
    """DQ 的相位数由算子编译器给出，要与 GML 侧的 4 相一致。"""
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert plan.count == PHASE_COUNTS["DynamicScaling"] == 4
    assert [p.index for p in plan.phases] == [0, 1, 2, 3]


def test_softmax_phase_count_matches_static_table() -> None:
    plan = parse_phase_plans(_EXPANDED_SOFTMAX)["softmax_five_phases"]
    assert plan.count == PHASE_COUNTS["Softmax"] == 5
    assert [p.index for p in plan.phases] == [0, 1, 2, 3, 4]


def test_rope_phase_count_matches_rope_units() -> None:
    """RoPE 三相 vs 静态表的四个子块名——两者不是一对一，这里钉住对应关系。

    实测（参考文档 §9.5）物理上是 3 次遍历：mul_cos / mul_sin / add。
    但静态表 `ROPE_UNITS` 去重后有 **4** 个子块名，因为 add 那一相有**两个
    输入**，每个输入各带一套 scale/zp：

        Llama2Activation_Cos      mul_cos 相的乘数配置
        Llama2Activation_Sin      mul_sin 相的乘数配置
        Llama2Activation_Add_Cos  add 相的 input0（来自 cos 路）
        Llama2Activation_Add_Sin  add 相的 input1（来自 sin 路）

    所以「4 个子块名 = 2 个 mul 相 + 1 个 add 相的 2 个输入槽」。若哪天 pass
    产出 4 个相位，就是把 add 的两个输入槽误当成两次遍历了。
    """
    plan = parse_phase_plans(_EXPANDED_ROPE)["rope_three_phases"]
    assert plan.count == 3
    assert plan.kinds() == ["mul", "mul", "add"]

    distinct_blocks = {block for _, block in hw_table.ROPE_UNITS}
    mul_blocks = {b for b in distinct_blocks if not b.startswith(
        "Llama2Activation_Add_")}
    add_slots = {b for b in distinct_blocks if b.startswith(
        "Llama2Activation_Add_")}
    # 两个 mul 相各一个子块；add 相一个，但占两个输入槽。
    assert len(mul_blocks) == 2
    assert len(add_slots) == 2
    assert len(mul_blocks) + 1 == plan.count


def test_dq_reduction_is_a_grouped_absmax() -> None:
    """p0 必须是 absmax，而且必须是**分组**归约。用 max 会让负值主导的组量化
    错一个符号；用 tile 级的 `reduce_axis` 加两次 reshape 冒充分组归约，则是在
    用不表达分组语义的算子（没有组大小、没有逐通道/逐组标志）发一组它说不出的
    字段。池化单元的分组归约才是这一步的实际硬件行为。
    """
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert plan.phases[0].kind == "absmax"
    assert plan.phases[0].op == "pim.global_pool"


def test_dq_kinds_match_static_table_semantics() -> None:
    """四相语义：absmax -> 恒等 -> 倒数 -> 定点化。

    相 1 是恒等表，不是 relu——它曾经被写成 `relu`，描述的是错的那个函数；
    ÷256 落在 FPSU 定标里，所以这一相只是原样过一遍表。

    静态表里 p2 带 `activation_special_operators=4`（选倒数表）、
    p3 带 `kantor_mode=fp2int_converter`，与这里的 kind 一一对应。
    """
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert plan.kinds() == ["absmax", "identity", "reciprocal", None]
    assert plan.phases[3].op == "pim.quantize"
    assert hw_table.DQ_PHASES[3]["kantor_mode"] == "fp2int_converter"
    assert hw_table.DQ_PHASES[2]["activation_special_operators"] == 4


def test_softmax_kinds_match_static_table_semantics() -> None:
    """五相语义：max -> exp -> 求和 -> 倒数 -> 归一化。"""
    plan = parse_phase_plans(_EXPANDED_SOFTMAX)["softmax_five_phases"]
    assert plan.kinds() == ["max", "exp", "add", "reciprocal", "mul"]
    # 静态表的 p3 同样是倒数那一相。
    assert hw_table.SOFTMAX_PHASES[3]["activation_special_operators"] == 4


def test_reduction_phases_run_on_vector_unit() -> None:
    """归约走 VPU，非线性走 CSTL——这是引擎单功能的硬约束。"""
    dq = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert dq.phases[0].unit == "vpu"
    # 查表相落在激活块上。以前六个块共用 `cstl` 一个名字，成本抽取分不出
    # 遍历类型；现在各自具名。
    assert dq.phases[1].unit == dq.phases[2].unit == "activation"


def test_dq_p1_parses_flp_and_activation_mode() -> None:
    """txt 的 Flp / Activation mode 从 IR attr 来，不是只靠静态表。"""
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    p1 = plan.phases[1]
    assert p1.activation_mode == 1
    assert p1.flp_min == 10 and p1.flp_max == 17 and p1.flp_mantisa == 3
    assert plan.phases[3].kantor_mode == 3

    # 寻址卡值（`transpose_purpose` 的 cardValue）必须真的解析出来：
    # 夹具若还停在旧的裸属性形态，这里会是 None 而测试照样绿。
    assert plan.phases[2].transpose_type == 1

    sm = parse_phase_plans(_EXPANDED_SOFTMAX)["softmax_five_phases"]
    # 两次归约在向量单元，两次查表在激活块，末尾的归一化乘在合成块。
    assert sm.units() == ["vpu", "activation", "vpu", "activation", "combiner"]
    assert sm.phases[1].transpose_type == 1
    assert sm.phases[3].transpose_type == 2


def test_stabilization_sub_is_not_a_phase() -> None:
    """Softmax 的 `x - max` 是折进 exp 相的 FPSU 仿射，不是第 6 个相位。

    若它被误标成相位，相位数会变 6，prepare_out 会多出一层。
    """
    plan = parse_phase_plans(_EXPANDED_SOFTMAX)["softmax_five_phases"]
    assert plan.count == 5
    assert "sub" not in [p.kind for p in plan.phases]


def test_layout_reshape_is_not_a_phase() -> None:
    """DQ 里的两个 `pim.reshape` 是布局，不占引擎遍历。"""
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert plan.count == 4
    assert "pim.reshape" not in [p.op for p in plan.phases]


def test_rope_phases_are_force_consecutive() -> None:
    """三连必须连续执行，中间结果不落 DDR。"""
    plan = parse_phase_plans(_EXPANDED_ROPE)["rope_three_phases"]
    assert all(p.force_consecutive for p in plan.phases)


def test_rope_sin_path_is_tagged_rotate_half() -> None:
    """只有 sin 那一路做 rotate_half；标错会让 Q/K 旋转错位。"""
    plan = parse_phase_plans(_EXPANDED_ROPE)["rope_three_phases"]
    assert [p.rotate_half for p in plan.phases] == [False, True, False]


def test_phase_bytes_follow_logical_shape() -> None:
    """逻辑缓冲字节数 = numel × elem_bytes。

    DQ：4096 个 fp16 输入分 32 组 -> 组统计量 32×2=64 字节，输出 4096 个 int8。
    Softmax：1024 个 fp16 = 2048 字节；两个标量归约相各占一个元素，但**宽度不同**
    —— 相 0 的极值是 fp16 值（2 字节），相 2 的求和是**真 fp32**（4 字节）。
    这不是记账口径的差异：把求和也截成 fp16，再由它取倒数，倒数就会带上求和
    自身的舍入，约千分之一，比这套算术本身的噪声大一个量级。
    """
    dq = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert [p.bytes for p in dq.phases] == [64, 64, 64, 4096]

    sm = parse_phase_plans(_EXPANDED_SOFTMAX)["softmax_five_phases"]
    assert [p.bytes for p in sm.phases] == [2, 2048, 4, 2, 2048]

    rope = parse_phase_plans(_EXPANDED_ROPE)["rope_three_phases"]
    assert [p.bytes for p in rope.phases] == [8192, 8192, 8192]


def test_dq_group_count_matches_quant_contract() -> None:
    """组数要与量化契约一致：4096 / 128 = 32 组，每组一个 fp16 统计量。"""
    from contracts.gml_quant import dq_group_size

    group_size = dq_group_size(4096, is_attention_scores=False)
    assert group_size == 128
    groups = 4096 // group_size
    plan = parse_phase_plans(_EXPANDED_DQ)["dq_four_phases"]
    assert plan.phases[0].bytes == groups * 2


def test_parse_ignores_unexpanded_operators() -> None:
    """单相算子不带 `pim.phase`，不该被当成相位。"""
    text = """
module {
  tt.func @single_phase(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    %m = pim.matmul %a, %b {datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xi8>
    tt.return
  }
}
"""
    plan = parse_phase_plans(text)["single_phase"]
    assert plan.count == 0


def test_single_phase_operator_still_carries_hardware_fields() -> None:
    """单相算子的硬件域进 `op_attrs`，**不进** `phases`。

    矩阵乘不展开，但 pass 仍在它身上盖 FPSU / FLP / 激活模式（FlagTree
    `stampMatmul`）。混进 `phases` 会让相位数把单相算子算成一相，
    `cross_check` 立刻判不符；而丢掉它们又会让 txt 的 `Flp *` 退回查表。
    """
    text = """
module {
  tt.func @gate_matmul(%a: tensor<4x8xi8>, %b: tensor<8x4xi8>) {
    %m = pim.matmul %a, %b {activation = #pim.act_spec<kind = silu>, activationMode = 0 : i64, datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>, flpMantisa = 3 : i64, flpMaxExp = 17 : i64, flpMinExp = 10 : i64, fpsu = #pim.fpsu_spec<mode = floating_point_32>} : tensor<4x8xi8>, tensor<8x4xi8> -> tensor<4x4xf16>
    tt.return
  }
}
"""
    plan = parse_phase_plans(text)["gate_matmul"]
    assert plan.count == 0, "单相算子不能被算成相位"
    assert plan.flp == (10, 17, 3)
    # `floating_point_32` 是 32 位累加通路的模式，对应卡值 2。
    assert plan.op_attrs["fpsu-mode"] == 2
    assert plan.op_attrs["activation-mode"] == 0
