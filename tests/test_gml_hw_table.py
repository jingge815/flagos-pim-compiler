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
    PHASE_TABLE,
    TOP_LEVEL,
    data_extension,
    gemm_kantor_mode,
    matmul_weight_format,
    phase_fields,
    rope_fields,
    top_level_fields,
)
from genesim_bridge.paths import gml_llama2_reference_dir

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


# 实物里唯一一处 dtype 与 data_extension 不一致的节点。
#
# `Split_params_21` 声明 `input_buffer_dtype "float16"` 但 `input_data_extensions 1`
# （1 对应 int8）。判定 dtype 才是真的，两条独立证据：
#   - `input_buffer_21.bin` 是 8192 字节 = 4096 个 fp16（按 int8 读则是 8192 个元素）
#   - 入边 dims 是 `1x32x1x128` = 4096 个元素
# 两者都指向 fp16。所以这是对方生成器的一处笔误，不是另一条规则。
_DATA_EXTENSION_EXCEPTIONS = {"Split_params_21"}


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
    # 例外只有那一处；若实物换版本后变多，这条会提醒重新审视规则。
    assert len(mismatched) == 1


def test_top_level_rejects_unknown_op() -> None:
    with pytest.raises(ValueError, match="常量表里没有"):
        top_level_fields("NoSuchOp")


def test_phase_fields_reject_out_of_range() -> None:
    with pytest.raises(ValueError, match="不是 phase 型算子"):
        phase_fields("Gemm", 0)
    with pytest.raises(ValueError, match="只有 4 相"):
        phase_fields("DynamicScaling", 4)
