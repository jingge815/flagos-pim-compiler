"""GML 硬件字段常量表：`(op_type, phase) -> 字段字典`。

这一族字段描述「用哪个硬件单元、走哪条数据通路」，对应 Ceva-NeuPro-M 的
NMU / FPSU / KANTOR / Pooling / Activation 五个单元。

**为什么归图编译器而不是算子编译器**：把参考产物的 200 个节点逐字段统计
`op_type -> distinct 值集合`，364 个字段项里 **353 项是单值**，
只有 11 项在同一 op_type 内出现两个值，且这 11 项全部由图编译器已知的信息决定
（见下面 `resolve` 的四条规则）。也就是说它们是**算子模板配置**，
不是分块/排布的结果 —— 所以查表即可，不必等算子编译器。

表是声明式的，没有逻辑。改这里之前先跑 tests/test_gml_hw_table.py，
它拿参考产物逐格对拍。

表里 `fpsu_*` / `global_pooling_*` / `kantor_A_*` 三族的 spc/spg 是**缺省值**：
它们其实是同一个量化决策投到三个硬件块上的结果，有量化规格时由
`derive_hardware_axes(...)` 派生（见本文件开头那张三行表），表值只在
「手上没有规格」时兜底。
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.gml_hw_constants import (
    DQ_PHASES,
    SOFTMAX_PHASES,
    TOP_LEVEL,
    PHASE_TABLE,
    ROPE_UNITS,
    ROPE_KANTOR_MODES,
    ROPE_KANTOR_BLOCKS,
    ROPE_SCALE_BLOCKS,
)
from contracts.gml_quant import QuantLayout

# phase 型节点声明的硬件 RTL 版本。实测 37 个带 phase 的节点全是 "1.4"，
# 其余节点不带这个字段 —— 它同时也是「是不是 phase 型节点」的判据。
RTL_VERSION = "1.4"


# ---------------------------------------------------------------------------
# 量化规格 -> 硬件块的 spc/spg
# ---------------------------------------------------------------------------
#
# spc（逐通道定标）与 spg（逐组定标）不是三组独立配置，是**同一个量化决策的
# 三处投影**：定点单元拿它定标、池化单元拿它分组求 absmax、逐元素乘单元拿它
# 分组反量化。所以它们从量化规格派生，不查表 —— 下面常量表里的那几项是
# 「手上没有规格」时的缺省值，有规格时被这里的派生覆盖。
#
# 硬件块的轴编号是**块自己的口径**，与量化规格里的张量轴（Llama2 上是 -1）
# 不是一回事，两套编号禁止混用。出处是 GML 实物的这几族：
#
#   | 硬件块                          | spc 轴 | spg 轴                | 出处 |
#   | 定点单元（FPSU）                | 1      | 顶层不写 groupSize    | TOP_LEVEL |
#   | 池化单元（Pooling）             | 2      | 3                     | DQ_PHASES[0] |
#   | 逐元素乘单元（Kantor A，DQ p3） | 2      | 3                     | DQ_PHASES[3] |
_BLOCK_AXES: dict[str, tuple[int, int]] = {
    "fpsu": (1, -1),
    "pooling": (2, 3),
    "kantor_a": (2, 3),
}


@dataclass(frozen=True)
class HardwareAxes:
    """量化规格投影到一个硬件块上的 spc/spg 五个字段。"""

    spc: bool
    spc_axis: int
    spg: bool
    spg_axis: int
    spg_group_size: int


def derive_hardware_axes(layout: QuantLayout, block: str) -> HardwareAxes:
    """量化规格 -> 某个硬件块该写的 spc/spg。

    三条规则：

    1. `spc` 看规格是否 per_tensor —— 整张量一个 scale 时单元不按通道取值。
    2. `spg` 看规格是否 per_group，**且该块真的做分组**：定点单元只吃逐通道
       那一位，分组归约是池化与 Kantor 的活，所以它永远不置位（实测 FPSU 的
       `fpsu_spg` 全图 0）。
    3. 轴取该块自己的编号；不分组时写 -1，与实物一致。

    与 `lib/Dialect/TritonPIM/IR/Dialect.cpp` 的 `deriveHardwareAxes` 同一条
    规则：算子编译器在构造相位 IR 时派生一遍，图编译器在这里派生一遍，两边
    都以量化规格为输入。
    """
    if block not in _BLOCK_AXES:
        raise ValueError(f"未知硬件块 {block!r}，只有 {sorted(_BLOCK_AXES)}")
    spc_axis, spg_axis = _BLOCK_AXES[block]
    spc = layout.granularity != "per_tensor"
    spg = layout.granularity == "per_group" and block != "fpsu"
    return HardwareAxes(
        spc=spc,
        spc_axis=spc_axis,
        spg=spg,
        spg_axis=spg_axis if spg else -1,
        spg_group_size=layout.group_size if spg else -1,
    )


# 由量化规格派生的键：GML 键名（去掉 `_phase_<k>` 后缀）-> (硬件块, 派生项)。
#
# 只列上面那张三行表覆盖的键，出处逐行对得上：FPSU 族取 TOP_LEVEL 与各相、
# 池化族取 DQ_PHASES[0]、Kantor A 族取 DQ_PHASES[3]。
#
# **不列** RoPE 子块（`fpsu_1_spc_Llama2Activation_Cos`）与顶层 EltwiseMul 的
# 逐槽 Kantor —— 那是各块**操作数自己的定标轴**：A/B 各一套，与 DQ 的分组决策
# 无关，数值也不同（实测 Kantor A 在 DQ 的 p3 是轴 2，在逐元素乘是轴 1）。
_FPSU_STEMS: dict[str, tuple[str, str]] = {
    "fpsu_spc": ("fpsu", "spc"),
    "fpsu_spc_axis": ("fpsu", "spc_axis"),
    "fpsu_spg": ("fpsu", "spg"),
    "fpsu_spg_axis": ("fpsu", "spg_axis"),
    "fpsu_spg_group_size": ("fpsu", "spg_group_size"),
}

# 顶层：三块里只有 FPSU 在顶层出面。逐槽 fpsu_0/1_* 是各块操作数自己的
# 定标轴，与 DQ 的分组决策无关，不进派生表。
TOP_AXIS_FIELD_STEMS: dict[str, tuple[str, str]] = dict(_FPSU_STEMS)

# 相内：三族都在（池化在 p0、Kantor A 在 p3、FPSU 每相都有）。
PHASE_AXIS_FIELD_STEMS: dict[str, tuple[str, str]] = {
    **_FPSU_STEMS,
    "global_pooling_spc": ("pooling", "spc"),
    "global_pooling_spc_axis": ("pooling", "spc_axis"),
    "global_pooling_spg": ("pooling", "spg"),
    "global_pooling_spg_axis": ("pooling", "spg_axis"),
    "global_pooling_group_size": ("pooling", "spg_group_size"),
    "kantor_A_spc": ("kantor_a", "spc"),
    "kantor_A_spg": ("kantor_a", "spg"),
    "kantor_A_scale_axis": ("kantor_a", "spc_axis"),
    "kantor_A_spg_axis": ("kantor_a", "spg_axis"),
    "kantor_A_spg_group_size": ("kantor_a", "spg_group_size"),
}


def _apply_axes(fields: dict[str, object], spec: QuantLayout,
                stems: dict[str, tuple[str, str]]) -> None:
    """把 spec 派生的值盖回字段字典。

    只改**已经声明**的键的值，不增不减键：实物哪一相写哪几个字段是固定的
    （DQ 就不写 `fpsu_spg_axis`），派生负责的是值，不是字段集合。
    """
    derived: dict[str, HardwareAxes] = {}
    for key in fields:
        entry = stems.get(key.split("_phase_")[0])
        if entry is None:
            continue
        block, name = entry
        axes = derived.setdefault(block, derive_hardware_axes(spec, block))
        fields[key] = int(getattr(axes, name))


# ---------------------------------------------------------------------------
# 顶层（非 phase）算子
# ---------------------------------------------------------------------------
#
# 空字典表示该算子不带任何硬件字段（实测 RMSNorm_vpu 就是这样——它绑在向量
# 单元 VPU 上，配置走 vpu_params 嵌套块，不走这五个单元）。
#
# `fpsu_spc` / `fpsu_spc_axis` / `fpsu_spg` 与 `fpsu_<槽>_spc` / `fpsu_<槽>_spg`
# 是**缺省值**：有量化规格时由 `derive_hardware_axes("fpsu", ...)` 覆盖。

def rope_fields(*, dq: bool = False) -> dict[str, object]:
    """RoPE 子块的全部硬件字段，按单元号展开。

    `dq=True` 给 `Llama2ActivationDQ` 用：它比基础变体多一套 4 相字段
    （由 `phase_fields("Llama2ActivationDQ", ...)` 提供，实测 62 个独有
    字段里 44 个被它精确覆盖），且末段 kantor 模式取 "off"。
    """
    fields: dict[str, object] = {}
    for unit, block in ROPE_UNITS:
        fields[f"fpsu_mode_{unit}_{block}"] = "floating_point"
        fields[f"fpsu_{unit}_spc_{block}"] = 1
        fields[f"fpsu_{unit}_spg_{block}"] = 0
        fields[f"fpsu_{unit}_spg_axis_{block}"] = -1
        fields[f"fpsu_{unit}_spg_group_size_{block}"] = -1
        # 键名缺下划线，照实物复现。
        fields[f"fpsu_{unit}_scale_axis{block}"] = 1
        fields[f"pooling_dtype_{unit}_{block}"] = "floating_point"
    for block, mode in ROPE_KANTOR_MODES.items():
        fields[f"kantor_mode_{block}"] = mode

    # Kantor A/B 两组：cos/sin 各要一整套 spc/spg 配置。
    for block in ROPE_KANTOR_BLOCKS:
        for side in ("A", "B"):
            fields[f"Kantor_{side}_spc_{block}"] = 1
            fields[f"Kantor_{side}_spg_{block}"] = 0
            fields[f"Kantor_{side}_spg_axis_{block}"] = -1
            fields[f"Kantor_{side}_spg_group_size_{block}"] = -1
    # 末段加法只有 A 侧，而且只挂在 K 路那个节点上：参考 node 30
    # （Llama2Activation）有这两项、node 22（DQ）没有。与
    # `Kantor_A_Llama2Activation_add_{scale,bias}_buffer_file` 同一处判据。
    if not dq:
        fields["Kantor_A_spc_Llama2Activation_add"] = 1
        fields["Kantor_A_spg_Llama2Activation_add"] = 0

    # 各子块的定标 dtype。
    for block in ROPE_SCALE_BLOCKS:
        fields[f"{block}_sf_dtype"] = "float16"

    # 末段加法的 kantor 模式随变体而变（见 ROPE_KANTOR_MODES 的说明）。
    fields["kantor_mode_Llama2Activation_add"] = (
        "off" if dq else "fp2int_converter")

    # 节点级数据通道。RoPE 有**三个**输入槽（被旋转的张量 + cos + sin），
    # 都吃 fp16；输出落 int8。
    for slot in range(3):
        fields[f"input_buffer_{slot}_dtype"] = "float16"
    fields["input_data_extensions"] = 3
    fields["output_buffer_dtype"] = "int8"
    fields["output_data_extension"] = 1
    fields["output_sf_dtype"] = "float16"
    # 两个中间态：x*cos 与 rotate_half(x)*sin 各自的乘积。
    fields["cos_mul_output_dtype"] = "float16"
    fields["sin_mul_output_dtype"] = "float16"
    # 头数进字段（cos/sin 要广播到每个头）。基础变体还带 transpose 0 ——
    # DQ 变体是 1，所以这一项只在非 DQ 时给（DQ 侧由调用方写）。
    if not dq:
        fields["transpose"] = 0
    return fields


# ---------------------------------------------------------------------------
# 由 dtype 决定的数据通道宽度
# ---------------------------------------------------------------------------
#
# `data_extension` 不是独立配置，而是 dtype 的编码：int8 -> 1、float16 -> 3。
# 实测全图无例外，所以按 dtype 推即可，不必进常量表。



DATA_EXTENSION: dict[str, int] = {
    "int8": 1,
    "float16": 3,
}


def data_extension(dtype: str) -> int:
    """一个缓冲区 dtype 对应的通道宽度编码。"""
    if dtype not in DATA_EXTENSION:
        raise ValueError(f"未知 dtype {dtype!r}，实测只出现 int8 与 float16")
    return DATA_EXTENSION[dtype]


# ---------------------------------------------------------------------------
# 四条按节点决定的规则（那 11 个「多值」字段项的来源）
# ---------------------------------------------------------------------------


def gemm_kantor_mode(*, output_dtype: str) -> str:
    """Gemm 的 kantor_mode 由**输出是否量化**决定。

    实测 7 个 Gemm 里 6 个输出 float16 -> `off`，只有 v_proj（节点 36）
    输出 int8 -> `fp2int_converter`。因为 value 要以 int8 存进 KV cache，
    定点化在这个节点上完成。图编译器知道每个节点的输出 dtype，所以能算。
    """
    return "fp2int_converter" if output_dtype == "int8" else "off"


def matmul_weight_format(*, transposed: bool) -> str:
    """MatMul 的 weight_format 由矩阵乘在 attention 里的角色决定。

    QK^T 吃 K 的转置 -> `weights_transpose`（实测 32 个 matmul1）；
    PV 直接吃 V -> `weight`（实测 32 个 matmul2）。
    **这是数学决定的，不是排布优化决定的** —— 所以归图编译器，不归算子编译器。
    """
    return "weights_transpose" if transposed else "weight"


def dq_group_size_fields(group_size: int) -> dict[str, object]:
    """DQ 的两个 group_size 字段，实测在全部 37 个节点上取值一致。

    这是唯一按节点变化的 phase 字段族，由量化契约（group_size）决定：
    hidden 与 MLP 中间态取 128，attention scores 整条一组取 1024。
    """
    return {
        "global_pooling_group_size_phase_0": group_size,
        "kantor_A_spg_group_size_phase_3": group_size,
    }


def phase_fields(op_type: str, phase: int, *, group_size: int | None = None,
                 spec: QuantLayout | None = None) -> dict:
    """一个算子某一相的硬件字段，键名已加 `_phase_<k>` 后缀。

    `group_size` 只对 DQ 有意义（p0 与 p3 各要一个 group_size 字段）；
    给了 `spec` 就以规格里的组宽为准（规格是量化决策的真源，组宽是它的一部分）。

    `spec` 是这一相的量化规格：给了它，三族 spc/spg（FPSU / 池化 / Kantor A）
    由 `derive_hardware_axes(...)` 派生，表里的值是没规格时的缺省。
    """
    phases = PHASE_TABLE.get(op_type)
    if phases is None:
        raise ValueError(f"{op_type} 不是 phase 型算子")
    if not 0 <= phase < len(phases):
        raise ValueError(f"{op_type} 只有 {len(phases)} 相，没有第 {phase} 相")

    fields = {f"{key}_phase_{phase}": value
              for key, value in phases[phase].items()}

    size = spec.group_size if spec is not None else group_size
    if op_type in ("DynamicScaling", "Llama2ActivationDQ") and size is not None:
        for key, value in dq_group_size_fields(size).items():
            if key.endswith(f"_phase_{phase}"):
                fields[key] = value
    if spec is not None:
        _apply_axes(fields, spec, PHASE_AXIS_FIELD_STEMS)
    return fields


def top_level_fields(
    op_type: str, *, output_dtype: str | None = None, transposed: bool = False,
    spec: QuantLayout | None = None,
) -> dict[str, object]:
    """一个算子的顶层硬件字段，已解出那些按节点变化的项。

    `spec` 是这类算子的量化规格：给了它，FPSU 族的 spc/spg 由
    `derive_hardware_axes("fpsu", ...)` 派生（顶层不写 groupSize）。
    """
    if op_type not in TOP_LEVEL:
        raise ValueError(f"常量表里没有 {op_type!r}")
    fields = dict(TOP_LEVEL[op_type])

    if op_type == "Gemm":
        fields["kantor_mode"] = gemm_kantor_mode(output_dtype=output_dtype or "float16")
    elif op_type == "MatMul":
        fields["weight_format"] = matmul_weight_format(transposed=transposed)
    elif op_type in ("Llama2Activation", "Llama2ActivationDQ"):
        fields.update(rope_fields())
    if spec is not None:
        _apply_axes(fields, spec, TOP_AXIS_FIELD_STEMS)
    return fields
