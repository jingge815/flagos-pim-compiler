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
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 顶层（非 phase）算子
# ---------------------------------------------------------------------------
#
# 空字典表示该算子不带任何硬件字段（实测 RMSNorm_vpu 就是这样——它绑在向量
# 单元 VPU 上，配置走 vpu_params 嵌套块，不走这五个单元）。

TOP_LEVEL: dict[str, dict[str, object]] = {
    "Gemm": {
        "nmu_mode": "floating_point",
        "fpsu_mode": "floating_point_32",
        "fpsu_spc": 1,
        "fpsu_spc_axis": 1,
        "fpsu_spg": 0,
        "pooling_dtype": "floating_point",
        # kantor_mode 见 resolve()：输出 int8 的那个节点是 fp2int_converter。
    },
    "MatMul": {
        "nmu_mode": "floating_point",
        "fpsu_mode": "floating_point_32",
        "fpsu_spc": 1,
        "fpsu_spc_axis": 1,
        "fpsu_spg": 0,
        "pooling_dtype": "floating_point",
        "kantor_mode": "off",
        # 第二个 operand 走权重通路，不占 input 槽——所以 input_count 少记 1。
        "MatMul_input_as_weight": 1,
        # 逐头展开后每个节点仍记录总头数（= num_attention_heads）。
        "group_attention_data_num": 32,
        "group_attention_weight_num": 32,
        # weight_format 见 resolve()：QK^T 转置、PV 不转。
    },
    "KV_Cache_DMA": {
        # 全图唯一走定点 FPSU 的算子。配 Scaling_buffer_file=2.0、Scaling_PS=14。
        "fpsu_mode": "fixed_point",
        "fpsu_spc": 1,
        "fpsu_spc_axis": 1,
        "fpsu_spg": 0,
        "pooling_dtype": "fixed_point",
        "kantor_mode": "off",
    },
    "EltwiseAdd": {
        # 双输入算子逐槽配置，槽号后缀而非 phase。
        "fpsu_mode_0": "floating_point",
        "fpsu_0_spc": 1,
        "fpsu_0_spg": 0,
        "pooling_dtype_0": "floating_point",
        "fpsu_mode_1": "floating_point",
        "fpsu_1_spc": 1,
        "fpsu_1_spg": 0,
        "pooling_dtype_1": "floating_point",
        "kantor_mode": "off",
    },
    "EltwiseMul": {
        "fpsu_mode_0": "floating_point",
        "fpsu_0_spc": 1,
        "fpsu_0_spg": 0,
        "pooling_dtype_0": "floating_point",
        "fpsu_mode_1": "floating_point",
        "fpsu_1_spc": 1,
        "fpsu_1_spg": 0,
        "pooling_dtype_1": "floating_point",
        # 逐元素相乘由 KANTOR 做，两块系数各一套。
        "kantor_mode": "elementwise_mul_fp16",
        "kantor_A_spc": 1,
        "kantor_A_spg": 0,
        "kantor_A_scale_axis": 1,
        "kantor_B_spc": 1,
        "kantor_B_spg": 0,
        "kantor_B_scale_axis": 1,
        "transpose": 1,
    },
    "Mask": {
        "kantor_mode": "off",
        "transpose": 1,
    },
    # 布局类算子只声明 kantor_mode=off，其余单元不参与。
    "Split": {"kantor_mode": "off"},
    "Concat": {"kantor_mode": "off"},
    "Transpose": {"kantor_mode": "off"},
    "Reshape": {"kantor_mode": "off"},
    # RMSNorm 走 VPU，配置在 vpu_params 块里，这里没有五单元字段。
    "RMSNorm_vpu": {},
    # phase 型算子的顶层部分。
    "DynamicScaling": {
        "rtl_version": "1.4",
        "transpose": 1,
    },
    "Llama2ActivationDQ": {
        "rtl_version": "1.4",
        "transpose": 1,
    },
    "Llama2Activation": {
        "transpose": 0,
    },
    "Softmax": {},
}


# ---------------------------------------------------------------------------
# phase 流水线
# ---------------------------------------------------------------------------
#
# phase 在 GML 里**不是独立节点**，而是同一个节点内的 `*_phase_<k>` 字段族。
#
# DQ 四相：p0 求组统计量 -> p1 乘 1/256 -> p2 取倒数 -> p3 Kantor 定点化
# Softmax 五相：p0 求 max -> p1 exp -> p2 求和 -> p3 取倒数 -> p4 归一化

_DQ_COMMON = {
    "fpsu_mode": "floating_point",
    "fpsu_spc": 1,
    "fpsu_spc_axis": 1,
    "fpsu_spg": 0,
    "pooling_dtype": "floating_point",
    "kantor_mode": "off",
    # 每相的数据通道声明。前三相都在 fp16 域里算（求 absmax、除 256、取倒数），
    # 只有 p3 的**输出**落到 int8 —— 那一相才是量化真正落定的地方。
    # `*_data_extensions` 不是独立配置，是 dtype 的编码（float16→3、int8→1），
    # 所以这里跟着 dtype 一起给，不单列。
    "input_buffer_dtype": "float16",
    "input_data_extensions": 3,
    "output_buffer_dtype": "float16",
    "output_data_extensions": 3,
}

DQ_PHASES: list[dict[str, object]] = [
    {
        **_DQ_COMMON,
        # p0 用 Pooling 块求每组的对称动态范围（abs max），所以只有这一相
        # 带 global_pooling_*。group_size 见 resolve()：随张量变化。
        "global_pooling_spc": 1,
        "global_pooling_spc_axis": 2,
        "global_pooling_spg": 1,
        "global_pooling_spg_axis": 3,
    },
    {
        **_DQ_COMMON,
        # p1 走 LUT 通路但用恒等表：activation_mode=1 表示直通，
        # 真正的 1/256 由 Scaling_buffer_phase_1 完成。
        "flp_min_exp": 10,
        "flp_max_exp": 17,
        "flp_mantisa": 3,
        "activation_mode": 1,
        "activation_special_operators": 0,
    },
    {
        **_DQ_COMMON,
        # p2 取倒数：special_operators=4 选倒数表。
        "flp_min_exp": 15,
        "flp_max_exp": 15,
        "flp_mantisa": 0,
        "activation_mode": 0,
        "activation_special_operators": 4,
    },
    {
        **_DQ_COMMON,
        # p3 由 KANTOR 做浮点转定点，这是量化真正落定的一相 ——
        # 也是唯一输出 int8 的一相（data_extensions 随之从 3 变 1）。
        "kantor_mode": "fp2int_converter",
        "kantor_A_spc": 1,
        "kantor_A_spg": 1,
        "kantor_A_scale_axis": 2,
        "kantor_A_spg_axis": 3,
        "output_buffer_dtype": "int8",
        "output_data_extensions": 1,
    },
]

_SOFTMAX_COMMON = {
    "fpsu_mode": "floating_point",
    "fpsu_spc": 1,
    "fpsu_spc_axis": 1,
    "fpsu_spg": 0,
    # Softmax 显式写 -1 表示不分组（DQ 侧则完全不写这两个字段）。
    "fpsu_spg_axis": -1,
    "fpsu_spg_group_size": -1,
    "pooling_dtype": "floating_point",
    "kantor_mode": "off",
    "nmu_output_type": "floating_point",
}

SOFTMAX_PHASES: list[dict[str, object]] = [
    # p0 求 max。注意 Softmax **没有** global_pooling_*——求 max 走另一条通路。
    {**_SOFTMAX_COMMON},
    {
        **_SOFTMAX_COMMON,
        # p1 过 exp 表。注意 flp 三元组与 DQ 的 p1 不同（9/16/3 vs 10/17/3）。
        "flp_min_exp": 9,
        "flp_max_exp": 16,
        "flp_mantisa": 3,
        "activation_mode": 0,
        "activation_special_operators": 0,
    },
    # p2 求和，用 fp32 累加（落盘就是真 fp32，与 p0 的编码不同）。
    {**_SOFTMAX_COMMON},
    {
        **_SOFTMAX_COMMON,
        # p3 取倒数，且 FPSU 升到 32 位——因为被除的和是 fp32。
        "fpsu_mode": "floating_point_32",
        "flp_min_exp": 15,
        "flp_max_exp": 15,
        "flp_mantisa": 0,
        "activation_mode": 0,
        "activation_special_operators": 4,
    },
    # p4 施加 1/sum。
    {**_SOFTMAX_COMMON},
]

PHASE_TABLE: dict[str, list[dict[str, object]]] = {
    "DynamicScaling": DQ_PHASES,
    "Llama2ActivationDQ": DQ_PHASES,
    "Softmax": SOFTMAX_PHASES,
}


# ---------------------------------------------------------------------------
# RoPE 子块
# ---------------------------------------------------------------------------
#
# Llama2Activation / Llama2ActivationDQ 内部按固定命名的子块配置，单元号 1..6
# 与子块一一绑定。子块名进字段名后缀，例如
# `fpsu_mode_1_Llama2Activation_Add_Cos`。
#
# 注意 `fpsu_<n>_scale_axis` 后面**没有下划线**就接子块名（实测 12 处），
# 这是对方生成器的键名 bug，为兼容需照样复现。

ROPE_UNITS: list[tuple[int, str]] = [
    (1, "Llama2Activation_Add_Cos"),
    (2, "Llama2Activation_Add_Sin"),
    (3, "Llama2Activation_Sin"),
    (4, "Llama2Activation_Sin"),
    (5, "Llama2Activation_Cos"),
    (6, "Llama2Activation_Cos"),
]

# 各子块的 kantor 模式：cos/sin 相乘用逐元素乘。
#
# `Llama2Activation_add` 的取值**随变体不同**，不是常量：
#   Llama2Activation   -> "fp2int_converter"（末尾要落定点）
#   Llama2ActivationDQ -> "off"（定点化交给它自己的 p3 那一相）
# 原先写死成前者，DQ 变体上就错了 —— 见 rope_fields(dq=...)。
ROPE_KANTOR_MODES: dict[str, str] = {
    "Llama2Activation_Cos": "elementwise_mul_fp16",
    "Llama2Activation_Sin": "elementwise_mul_fp16",
}

# 带 Kantor A/B 两组配置的子块（实测只有 cos/sin 两个，各一整套）。
ROPE_KANTOR_BLOCKS = ("Llama2Activation_Sin", "Llama2Activation_Cos")

# 声明了 fp16 定标的子块。`_Broadcast` 是 cos/sin 广播到各头的那一路。
ROPE_SCALE_BLOCKS = (
    "Llama2Activation_Add_Cos",
    "Llama2Activation_Add_Sin",
    "Llama2Activation_Sin",
    "Llama2Activation_Sin_Broadcast",
    "Llama2Activation_Cos",
    "Llama2Activation_Cos_Broadcast",
)


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
    # 末段加法只有 A 侧。
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

# phase 型节点声明的硬件 RTL 版本。实测 37 个带 phase 的节点全是 "1.4"，
# 其余节点不带这个字段 —— 所以它同时也是「这是不是 phase 型节点」的判据
# （见 contracts/gml_names.phase_output_buffer_self 那条自命名例外）。
RTL_VERSION = "1.4"


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


def phase_fields(op_type: str, phase: int, *, group_size: int | None = None) -> dict:
    """一个算子某一相的硬件字段，键名已加 `_phase_<k>` 后缀。

    `group_size` 只对 DQ 有意义（p0 与 p3 各要一个 group_size 字段）。
    """
    phases = PHASE_TABLE.get(op_type)
    if phases is None:
        raise ValueError(f"{op_type} 不是 phase 型算子")
    if not 0 <= phase < len(phases):
        raise ValueError(f"{op_type} 只有 {len(phases)} 相，没有第 {phase} 相")

    fields = {f"{key}_phase_{phase}": value
              for key, value in phases[phase].items()}

    if op_type in ("DynamicScaling", "Llama2ActivationDQ") and group_size is not None:
        for key, value in dq_group_size_fields(group_size).items():
            if key.endswith(f"_phase_{phase}"):
                fields[key] = value
    return fields


def top_level_fields(
    op_type: str, *, output_dtype: str | None = None, transposed: bool = False
) -> dict[str, object]:
    """一个算子的顶层硬件字段，已解出那些按节点变化的项。"""
    if op_type not in TOP_LEVEL:
        raise ValueError(f"常量表里没有 {op_type!r}")
    fields = dict(TOP_LEVEL[op_type])

    if op_type == "Gemm":
        fields["kantor_mode"] = gemm_kantor_mode(output_dtype=output_dtype or "float16")
    elif op_type == "MatMul":
        fields["weight_format"] = matmul_weight_format(transposed=transposed)
    elif op_type in ("Llama2Activation", "Llama2ActivationDQ"):
        fields.update(rope_fields())
    return fields
