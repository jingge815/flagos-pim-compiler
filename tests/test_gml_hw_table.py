"""硬件常量表与参考产物的逐格对拍。

这个测试是常量表的存在理由：它证明「这些字段由 `(op_type, phase)` 唯一决定」
这个判断成立 —— 若某个字段其实按节点变化，`test_no_hardware_field_varies_by_node`
会立刻失败，说明它不该进常量表。
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.gml_hw_table import (
    PHASE_AXIS_FIELD_STEMS,
    PHASE_TABLE,
    TOP_AXIS_FIELD_STEMS,
    TOP_LEVEL,
    HardwareAxes,
    data_extension,
    derive_hardware_axes,
    gemm_kantor_mode,
    matmul_weight_format,
    phase_fields,
    rope_fields,
    top_level_fields,
)
from contracts.gml_quant import QuantLayout
from genesim_bridge.paths import gml_llama2_reference_dir

# 参考产物这一路的量化规格：激活与权重都是逐组、组宽 128、沿最后一维。
# 三族 spc/spg 由它派生，与 `contracts.gml_quant.ACTIVATION_LAYOUT` 同值。
_SPEC = QuantLayout("per_group", group_size=128, axis=-1)

# 归常量表的字段族。与 gml_hw_table 的覆盖范围一致。
_HARDWARE_FIELD = re.compile(
    r"^(nmu_|fpsu_|kantor_|Kantor_|global_pooling_|pooling_dtype|flp_"
    r"|activation_mode|activation_special|transpose$|weight_format"
    r"|rtl_version|MatMul_input_as_weight|group_attention_"
    # 每相的数据通道声明也是硬件字段：dtype 决定通道宽度
    # （`*_data_extensions` 就是 dtype 的编码）。漏掉它们会让
    # 「表里有、实物没有」的断言误报 —— 实物明明有，只是这个
    # 白名单没放进来。
    r"|input_buffer_dtype|output_buffer_dtype"
    r"|input_data_extensions|output_data_extension)")

# 这些字段实测在同一 op_type 内有两个值，各自由一条规则解出（见 gml_hw_table）。
_NODE_DEPENDENT = {
    "kantor_mode",                        # Gemm：按输出 dtype
    "weight_format",                      # MatMul：按矩阵乘角色
    "global_pooling_group_size_phase_0",  # DQ：按量化契约
    "kantor_A_spg_group_size_phase_3",    # DQ：同上
    # dtype 是**沿边传播**的，不是每个节点独立的常量：同一个 Transpose
    # 在 DQ 下游吃 int8、在别处吃 fp16，所以同 op_type 内必然多值。
    # `*_data_extensions` 是 dtype 的编码（float16→3、int8→1），同理。
    # 这几族由「上游是谁」解出，不该进常量表 —— 正是本轮 §11.5.6 的结论。
    "input_buffer_dtype",
    "output_buffer_dtype",
    "input_data_extensions",
    "output_data_extension",
}


def _nodes() -> list[dict[str, list[str]]]:
    """解析参考产物的节点块。按括号深度切，因为 node 块可嵌套。"""
    text = (gml_llama2_reference_dir() / "relay2gml_graph.gml").read_text()

    blocks, current, depth = [], None, 0
    for line in text.split("\n"):
        if line.strip().startswith("node [") and current is None:
            current, depth = [line], 1
            continue
        if current is not None:
            current.append(line)
            depth += line.count("[") - line.count("]")
            if depth == 0:
                blocks.append(current)
                current = None

    parsed = []
    for block in blocks:
        fields: dict[str, list[str]] = collections.defaultdict(list)
        # 只取节点自身那一层（4 空格缩进），跳过 contraction / vpu_params 内部。
        for line in block[1:-1]:
            match = re.match(r"^\s{4}([A-Za-z_][A-Za-z_0-9]*)\s+(.+?)\s*$", line)
            if match:
                fields[match.group(1)].append(match.group(2).strip('"'))
        parsed.append(fields)
    return parsed


@pytest.fixture(scope="module")
def nodes() -> list[dict[str, list[str]]]:
    return _nodes()


def _hardware_values(nodes) -> dict[str, dict[str, set[str]]]:
    """按 op_type 收集每个硬件字段出现过的值。文件名字段不算。"""
    table: dict[str, dict[str, set[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for fields in nodes:
        op_type = fields.get("op_type", ["BUFFER"])[0]
        for key, values in fields.items():
            if not _HARDWARE_FIELD.match(key):
                continue
            if any(value.endswith(".bin") for value in values):
                continue
            table[op_type][key].update(values)
    return table


def test_no_hardware_field_varies_by_node(nodes) -> None:
    """核心论证：硬件字段在同一 op_type 内必须是单值。

    这是「常量表可行」的判据。允许的例外只有 4 个字段族，每个都由一条
    图编译器可算的规则解出——不是分块或排布的结果。
    """
    table = _hardware_values(nodes)

    varying = {
        (op_type, key): sorted(values)
        for op_type, fields in table.items()
        for key, values in fields.items()
        if len(values) > 1
    }
    unexpected = {
        (op, key): values for (op, key), values in varying.items()
        if key not in _NODE_DEPENDENT
    }
    assert not unexpected, f"这些字段按节点变化，不该进常量表: {unexpected}"


def test_hardware_field_coverage_is_large(nodes) -> None:
    """常量表覆盖的字段项数量级要对得上（实测 364 项、353 项单值）。"""
    table = _hardware_values(nodes)
    total = sum(len(fields) for fields in table.values())
    single = sum(
        1 for fields in table.values()
        for values in fields.values() if len(values) == 1)
    assert total > 300
    assert single / total > 0.95


@pytest.mark.parametrize("op_type", sorted(TOP_LEVEL))
def test_top_level_matches_the_reference(op_type: str, nodes) -> None:
    """常量表里每个顶层字段的值都要与参考产物一致。

    只比**表里声明的**字段：实物还有量化参数、文件名、拓扑等字段，
    那些不归常量表。
    """
    table = _hardware_values(nodes)
    if op_type not in table:
        pytest.skip(f"参考产物里没有 {op_type}")

    reference = table[op_type]
    for key, expected in TOP_LEVEL[op_type].items():
        assert key in reference, f"{op_type} 的 {key} 在参考产物里不存在"
        assert reference[key] == {str(expected)}, \
            f"{op_type}.{key}: 表里 {expected}，实物 {sorted(reference[key])}"


@pytest.mark.parametrize("op_type", sorted(PHASE_TABLE))
def test_phase_fields_match_the_reference(op_type: str, nodes) -> None:
    """phase 字段逐相逐字段对拍。"""
    table = _hardware_values(nodes)
    if op_type not in table:
        pytest.skip(f"参考产物里没有 {op_type}")

    reference = table[op_type]
    for phase in range(len(PHASE_TABLE[op_type])):
        for key, expected in phase_fields(op_type, phase).items():
            assert key in reference, f"{op_type} 的 {key} 在参考产物里不存在"
            assert reference[key] == {str(expected)}, \
                f"{op_type}.{key}: 表里 {expected}，实物 {sorted(reference[key])}"


def test_phase_counts_match_the_reference(nodes) -> None:
    """相数按算子分：DQ 4 相、Softmax 5 相。

    判据取每个算子实际出现的最大相号——若表里相数多算一相，
    生成器会为不存在的相分配文件与字段。
    """
    table = _hardware_values(nodes)
    for op_type, phases in PHASE_TABLE.items():
        if op_type not in table:
            continue
        seen = {
            int(match.group(1))
            for key in table[op_type]
            if (match := re.search(r"_phase_(\d+)$", key))
        }
        assert max(seen) == len(phases) - 1, \
            f"{op_type}: 实物最大相号 {max(seen)}，表里 {len(phases)} 相"


def test_softmax_has_no_global_pooling(nodes) -> None:
    """Softmax 求 max 不走 Pooling 块——它没有 global_pooling_* 字段。

    DQ 的 p0 有这一族（求组 absmax），Softmax 没有。混淆会写出实物不存在的字段。
    """
    table = _hardware_values(nodes)
    assert not [key for key in table["Softmax"] if key.startswith("global_pooling")]
    assert [key for key in table["DynamicScaling"]
            if key.startswith("global_pooling")]

    for phase in range(len(PHASE_TABLE["Softmax"])):
        emitted = phase_fields("Softmax", phase)
        assert not [key for key in emitted if key.startswith("global_pooling")]


def test_softmax_declares_spg_axis_but_dq_does_not(nodes) -> None:
    """Softmax 显式写 spg_axis/-group_size = -1，DQ 完全不写这两个字段。"""
    table = _hardware_values(nodes)
    assert table["Softmax"]["fpsu_spg_axis_phase_0"] == {"-1"}
    assert "fpsu_spg_axis_phase_0" not in table["DynamicScaling"]

    assert "fpsu_spg_axis_phase_0" in phase_fields("Softmax", 0)
    assert "fpsu_spg_axis_phase_0" not in phase_fields("DynamicScaling", 0)


def test_gemm_kantor_mode_follows_output_dtype(nodes) -> None:
    """7 个 Gemm 里只有输出 int8 的那个是 fp2int_converter。"""
    for fields in nodes:
        if fields.get("op_type", [""])[0] != "Gemm":
            continue
        output_dtype = fields["output_buffer_dtype"][0]
        assert fields["kantor_mode"][0] == gemm_kantor_mode(output_dtype=output_dtype)

    assert gemm_kantor_mode(output_dtype="int8") == "fp2int_converter"
    assert gemm_kantor_mode(output_dtype="float16") == "off"


def test_matmul_weight_format_splits_evenly(nodes) -> None:
    """64 个 MatMul 里 32 个转置（QK^T）、32 个不转（PV）。

    判据用 label 里的 matmul1/matmul2 —— 这正是图编译器构图时知道的角色信息。
    """
    counts: collections.Counter[str] = collections.Counter()
    for fields in nodes:
        if fields.get("op_type", [""])[0] != "MatMul":
            continue
        label = fields["label"][0]
        transposed = "matmul1" in label
        expected = matmul_weight_format(transposed=transposed)
        assert fields["weight_format"][0] == expected, label
        counts[expected] += 1

    assert counts == {"weights_transpose": 32, "weight": 32}


def test_dq_group_size_pair_always_agrees(nodes) -> None:
    """两个 group_size 字段在同一节点上必须相等（实测 37/37）。"""
    for fields in nodes:
        pooling = fields.get("global_pooling_group_size_phase_0")
        kantor = fields.get("kantor_A_spg_group_size_phase_3")
        if pooling and kantor:
            assert pooling[0] == kantor[0]

    emitted = phase_fields("DynamicScaling", 0, group_size=1024)
    assert emitted["global_pooling_group_size_phase_0"] == 1024
    emitted = phase_fields("DynamicScaling", 3, group_size=1024)
    assert emitted["kantor_A_spg_group_size_phase_3"] == 1024


def test_rope_fields_match_the_reference(nodes) -> None:
    """RoPE 的 6 个单元 × 子块字段逐个对拍，含那个缺下划线的键名 bug。

    只对拍**硬件字段**（`_HARDWARE_FIELD` 能匹配的）：`rope_fields()` 现在
    也含节点级 dtype / data_extensions，那些沿边传播、不进常量表。
    """
    table = _hardware_values(nodes)
    reference = table["Llama2Activation"]

    for key, expected in rope_fields().items():
        if not _HARDWARE_FIELD.match(key):
            continue
        assert key in reference, f"RoPE 字段 {key} 在参考产物里不存在"
        assert reference[key] == {str(expected)}, key

    # 键名 bug：scale_axis 后面直接接子块名，没有下划线。实测 12 处。
    assert "fpsu_1_scale_axisLlama2Activation_Add_Cos" in rope_fields()


def test_data_extension_follows_dtype() -> None:
    """通道宽度编码由 dtype 推，不是独立配置。"""
    assert data_extension("int8") == 1
    assert data_extension("float16") == 3
    with pytest.raises(ValueError, match="未知 dtype"):
        data_extension("float32")


# 已知例外。当前参考（v2）里**一处都没有**——v1 有一处：
#
# `Split_params_21` 声明 `input_buffer_dtype "float16"` 但 `input_data_extensions 1`
# （1 对应 int8），而 `input_buffer_21.bin` 是 8192 字节 = 4096 个 fp16、入边 dims
# 也是 `1x32x1x128` = 4096 个元素，两条独立证据都指向 fp16。所以那是对方生成器
# 在当时那一版里的一处笔误，v2 已修掉，这也是切换参考后这条断言从 1 变 0 的原因。
#
# 留着这个集合而不是删掉：v1 的参考路径仍在（`llama2_w4a8_decode_block_0`），
# 谁把它指回去，这个例外就要重新生效。
_DATA_EXTENSION_EXCEPTIONS: frozenset[str] = frozenset()


def test_data_extension_matches_the_reference(nodes) -> None:
    """实物里 dtype 与 data_extension 必须一一对应（一处已知例外）。"""
    mismatched = []
    for fields in nodes:
        label = fields.get("label", ["?"])[0]
        for dtype_key, extension_key in (
            ("input_buffer_dtype", "input_data_extensions"),
            ("output_buffer_dtype", "output_data_extension"),
        ):
            if dtype_key in fields and extension_key in fields:
                dtype = fields[dtype_key][0]
                if dtype not in ("int8", "float16"):
                    continue
                if int(fields[extension_key][0]) != data_extension(dtype):
                    mismatched.append((label, dtype_key, dtype))

    unexpected = [
        item for item in mismatched if item[0] not in _DATA_EXTENSION_EXCEPTIONS]
    assert not unexpected, f"dtype 与 data_extension 不一致: {unexpected}"
    # 例外要恰好用完，不能多也不能少：多了说明规则没跟上实物，少了说明某个
    # 本该报出来的不一致被顺手改成了例外。当前参考下应为 0。
    assert len(mismatched) == len(_DATA_EXTENSION_EXCEPTIONS)


def _derived_value(stems, key: str, spec: QuantLayout) -> int:
    """派生表里 `key` 该取的值（键名去掉 `_phase_<k>` 后缀再查）。"""
    block, name = stems[key.split("_phase_")[0]]
    return int(getattr(derive_hardware_axes(spec, block), name))


def test_derived_axes_equal_the_constants() -> None:
    """派生出的 spc/spg 与常量表**逐格相同**。

    这是迁移的等价性判据：三族从「查表」改成「从量化规格派生」，值必须一格
    不差，否则 GML 就不再逐字节相同。表里三族的每个键都要有一个派生值与之
    对应且相等；派生规则改了而表没跟着改（或反之），这里立刻失败。
    """
    cells = 0
    for op_type in TOP_LEVEL:
        for key, expected in top_level_fields(op_type).items():
            if key not in TOP_AXIS_FIELD_STEMS:
                continue
            assert _derived_value(TOP_AXIS_FIELD_STEMS, key, _SPEC) == expected, \
                f"{op_type}.{key}: 派生 {_derived_value(TOP_AXIS_FIELD_STEMS, key, _SPEC)}，表 {expected}"
            cells += 1

    for op_type, phases in PHASE_TABLE.items():
        for phase in range(len(phases)):
            # 带上组宽：它也是派生出来的一格（规格的一部分），不是常量。
            for key, expected in phase_fields(
                    op_type, phase, group_size=_SPEC.group_size).items():
                if key.split("_phase_")[0] not in PHASE_AXIS_FIELD_STEMS:
                    continue
                derived = _derived_value(PHASE_AXIS_FIELD_STEMS, key, _SPEC)
                assert derived == expected, \
                    f"{op_type} p{phase} {key}: 派生 {derived}，表 {expected}"
                cells += 1

    # 逐槽 fpsu_0/1_* 不进派生表（与 DQ 分组无关），覆盖从 86 降到 78。
    assert cells == 78, f"派生覆盖 {cells} 格，实测 78 格"


def test_axes_follow_the_spec_not_the_table() -> None:
    """派生跟着规格走，不是查表：规格一变，三族跟着变。

    表里只有**一种**规格的值（全图只有一种量化配置），所以「查表能跑通」现在
    成立；这条测试钉住的是升级的理由——换了组宽或换成逐通道，查表会给错值且
    不报错，派生则会跟着变。
    """
    per_group = QuantLayout("per_group", group_size=1024, axis=-1)
    assert derive_hardware_axes(per_group, "pooling") == \
        HardwareAxes(True, 2, True, 3, 1024)
    assert derive_hardware_axes(per_group, "kantor_a") == \
        HardwareAxes(True, 2, True, 3, 1024)
    # FPSU 不做分组归约：规格是逐组，它也只在逐通道那一位上置位。
    assert derive_hardware_axes(per_group, "fpsu") == \
        HardwareAxes(True, 1, False, -1, -1)

    # 逐通道：spg 不置位，组宽与 spg 轴写 -1。
    per_channel = QuantLayout("per_channel", axis=-1)
    assert derive_hardware_axes(per_channel, "pooling") == \
        HardwareAxes(True, 2, False, -1, -1)

    # 逐张量：连 spc 都不置位（单元不按通道取值）。
    assert derive_hardware_axes(QuantLayout("per_tensor"), "fpsu").spc is False

    with pytest.raises(ValueError, match="未知硬件块"):
        derive_hardware_axes(per_channel, "nmu")

    # 组宽也来自规格：1024 落到池化与 Kantor A 两处，而不是表里的缺省。
    fields = phase_fields("DynamicScaling", 0, spec=per_group)
    assert fields["global_pooling_group_size_phase_0"] == 1024
    fields = phase_fields("DynamicScaling", 3, spec=per_group)
    assert fields["kantor_A_spg_group_size_phase_3"] == 1024


@pytest.mark.parametrize("op_type", sorted(TOP_LEVEL))
def test_top_level_axes_with_a_spec_match_the_reference(op_type, nodes) -> None:
    """走派生路径发出来的顶层 spc/spg 也要与参考产物逐格相同。"""
    table = _hardware_values(nodes)
    if op_type not in table:
        pytest.skip(f"参考产物里没有 {op_type}")

    emitted = top_level_fields(op_type, spec=_SPEC)
    derived_keys = [key for key in emitted if key in TOP_AXIS_FIELD_STEMS]
    for key in derived_keys:
        assert key in table[op_type], f"{op_type} 的 {key} 在参考产物里不存在"
        assert table[op_type][key] == {str(emitted[key])}, \
            f"{op_type}.{key}: 派生 {emitted[key]}，实物 {sorted(table[op_type][key])}"


@pytest.mark.parametrize("op_type", sorted(PHASE_TABLE))
def test_phase_axes_with_a_spec_match_the_reference(op_type, nodes) -> None:
    """走派生路径发出来的相位 spc/spg 也要与参考产物逐格相同。

    只比轴与标志（派生覆盖的那些键）：组宽按节点变化（128 与 1024 两种），
    有自己的解，见 `_NODE_DEPENDENT`。
    """
    table = _hardware_values(nodes)
    if op_type not in table:
        pytest.skip(f"参考产物里没有 {op_type}")

    for phase in range(len(PHASE_TABLE[op_type])):
        emitted = phase_fields(op_type, phase, spec=_SPEC)
        for key, value in emitted.items():
            if key.split("_phase_")[0] not in PHASE_AXIS_FIELD_STEMS:
                continue
            if key.endswith("group_size_phase_%d" % phase):
                continue
            assert key in table[op_type], f"{op_type} 的 {key} 在参考产物里不存在"
            assert table[op_type][key] == {str(value)}, \
                f"{op_type}.{key}: 派生 {value}，实物 {sorted(table[op_type][key])}"


def test_rope_and_eltwise_kantor_axes_are_not_derived() -> None:
    """RoPE 子块与逐元素乘的 Kantor 轴**不**进派生表。

    它们是各块操作数自己的定标轴（A/B 各一套、轴 1），与 DQ 的分组决策无关；
    实测 Kantor A 在 DQ 的 p3 是轴 2、在逐元素乘是轴 1，混起来会静默发错值。
    """
    eltwise = top_level_fields("EltwiseMul", spec=_SPEC)
    assert eltwise["kantor_A_scale_axis"] == 1
    assert eltwise["kantor_A_spg"] == 0
    assert eltwise["kantor_B_scale_axis"] == 1

    rope = rope_fields()
    assert rope["fpsu_1_spc_Llama2Activation_Add_Cos"] == 1
    assert rope["fpsu_1_spg_axis_Llama2Activation_Add_Cos"] == -1
    # 派生表里没有它们，所以 `_apply_axes` 一动也不动。
    derived = [key for key in eltwise if key in TOP_AXIS_FIELD_STEMS]
    assert not derived, f"逐槽/逐操作数轴不该进派生表: {derived}"
    assert not [key for key in rope if key.split("_phase_")[0]
                in PHASE_AXIS_FIELD_STEMS]


def test_top_level_rejects_unknown_op() -> None:
    with pytest.raises(ValueError, match="常量表里没有"):
        top_level_fields("NoSuchOp")


def test_phase_fields_reject_out_of_range() -> None:
    with pytest.raises(ValueError, match="不是 phase 型算子"):
        phase_fields("Gemm", 0)
    with pytest.raises(ValueError, match="只有 4 相"):
        phase_fields("DynamicScaling", 4)
