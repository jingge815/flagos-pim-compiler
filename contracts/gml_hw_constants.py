"""GML 硬件字段的**常量数据表**：op_type / phase -> 字段字典。

与 `gml_hw_table.py` 分开：那个文件放派生算法，这里放声明式数据。表本身没有
逻辑，改之前先跑 `tests/test_gml_hw_table.py`（拿参考产物逐格对拍）。

`fpsu_*` / `global_pooling_*` / `kantor_A_*` 三族的 spc/spg 是**缺省值**：
有量化规格时由 `gml_hw_table.derive_hardware_axes(...)` 派生覆盖。
"""

from __future__ import annotations

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
        # 这里的 spc/spg 是**逐槽乘法块自己的定标轴**（A/B 各一套、轴 1），
        # 不是 DQ 量化决策的投影 —— 所以不进派生表，照常量写。
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
#
# 下面各相的 `fpsu_*` / `global_pooling_*` / `kantor_A_*` 也是**缺省值**：
# 有量化规格时由 `derive_hardware_axes(...)` 按规格覆盖（p0 的池化、
# p3 的 Kantor A、每相的 FPSU 三族都在覆盖范围内）。

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
    # 每相的数据通道声明，同 DQ 那族。Softmax 五相进出都在 fp16 域
    # （exp 表、求和、取倒数、乘回），所以五相都是 3，实测参考 32/32 节点
    # 十个字段全为 3。**不写这一族不会报错**：字段族整体消失，而 dtype 自检
    # 只看节点级的三个 `*_buffer_dtype`，相位级的缺席它看不见。
    "input_data_extensions": 3,
    "output_data_extensions": 3,
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


