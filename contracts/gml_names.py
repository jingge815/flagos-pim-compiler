"""GML 缓冲区文件名规则——全仓唯一真源。

这套规则是从参考产物的 89 条边逐条验证出来的，见 docs/gml-lowering-20260914.md
规则 2。要紧之处在于它跨两种语言：FlagTree（C++）把文件名写进 GML 文本，图编译器
（Python）写文件本身，两边必须字节一致，否则底层编译器读到的是悬空引用。

所以规则集中在这里，C++ 侧照抄，并用 tests/test_gml_names.py 交叉校验：
序列化器产出的每个名字都要在磁盘上真实存在，反之亦然。

命名的反直觉之处：**缓冲区按消费者编号，不按生产者**。一个节点的 output_buffer
写的是它下游的编号，因为缓冲区代表的是「边」而不是「某个节点的输出」。
"""

from __future__ import annotations

# 缓冲区文件的扩展名。
SUFFIX = ".bin"


def data_buffer(consumer_id: int, slot: int | None = None) -> str:
    """一条边上流动的数据缓冲区名。

    `consumer_id` 是**读取**这块数据的节点 id。多输入算子的每个输入槽各有一块，
    用 `slot` 区分；单输入算子不带槽位号。

        data_buffer(8)      -> "input_buffer_8.bin"
        data_buffer(6, 1)   -> "input_buffer_1_6.bin"
    """
    if slot is None:
        return f"input_buffer_{consumer_id}{SUFFIX}"
    return f"input_buffer_{slot}_{consumer_id}{SUFFIX}"


def scale(consumer_id: int, slot: int | None = None) -> str:
    """输入数据的量化 scale。粒度是标量（per-tensor）。"""
    if slot is None:
        return f"input_sf_{consumer_id}{SUFFIX}"
    return f"input_{slot}_sf_{consumer_id}{SUFFIX}"


def zero_point(consumer_id: int, slot: int | None = None) -> str:
    """输入数据的量化 zero-point。粒度是标量。"""
    if slot is None:
        return f"input_zp_{consumer_id}{SUFFIX}"
    return f"input_{slot}_zp_{consumer_id}{SUFFIX}"


# 权重、偏置及其量化参数按**本节点**编号——它们属于节点自己，不属于某条边。

def weight_buffer(node_id: int) -> str:
    return f"weight_buffer_{node_id}{SUFFIX}"


def weight_scale(node_id: int) -> str:
    return f"weight_sf_{node_id}{SUFFIX}"


def weight_zero_point(node_id: int) -> str:
    return f"weight_zp_{node_id}{SUFFIX}"


def bias_buffer(node_id: int) -> str:
    return f"bias_buffer_{node_id}{SUFFIX}"


def bias_scale(node_id: int) -> str:
    return f"bias_sf_{node_id}{SUFFIX}"


def bias_zero_point(node_id: int) -> str:
    return f"bias_zp_{node_id}{SUFFIX}"


def output_scale(node_id: int) -> str:
    return f"output_sf_{node_id}{SUFFIX}"


def output_zero_point(node_id: int) -> str:
    return f"output_zp_{node_id}{SUFFIX}"


# 累加后的定标系数。粒度是 per-channel（数组，长度等于输出通道数），
# 与输入/权重的标量 scale 不同。多输入算子每个槽各一套。

def fpsu_scale(node_id: int, slot: int | None = None) -> str:
    """FPSU 的 per-channel scale。"""
    if slot is None:
        return f"Scaling_buffer_file_{node_id}{SUFFIX}"
    return f"Scaling_buffer_file_{slot}_{node_id}{SUFFIX}"


def fpsu_post_shift(node_id: int, slot: int | None = None) -> str:
    """FPSU 的 per-channel 后移位量。"""
    if slot is None:
        return f"Scaling_PS_buffer_file_{node_id}{SUFFIX}"
    return f"Scaling_PS_buffer_file_{slot}_{node_id}{SUFFIX}"


def fpsu_bias(node_id: int, slot: int | None = None) -> str:
    """FPSU 的 per-channel bias。"""
    if slot is None:
        return f"Bias_buffer_file_{node_id}{SUFFIX}"
    return f"Bias_buffer_file_{slot}_{node_id}{SUFFIX}"


# kantor 重定标块的系数，按物理块编号（A、B……）。粒度同样是 per-channel。

def kantor_scale(node_id: int, block: str = "A") -> str:
    return f"kantor_{block}_scale_buffer_file_{node_id}{SUFFIX}"


def kantor_bias(node_id: int, block: str = "A") -> str:
    return f"kantor_{block}_bias_buffer_file_{node_id}{SUFFIX}"


def kantor_shift(node_id: int, block: str = "A") -> str:
    return f"kantor_{block}_Shift_{node_id}{SUFFIX}"


def activation_lut(node_id: int) -> str:
    """激活查找表。必须按节点生成——表里折进了该节点自己的 scale。"""
    return f"activation_lut_file_{node_id}{SUFFIX}"

# 激活前的中间态。只有多输入算子（EltwiseAdd）才显式声明它——实测 16 处，
# 宽度是 int16，比周围的 int8 宽，因为它是累加器截断后、查表之前的值。
# 名字里的 `Relu` 是 GML 的字面写法，与实际激活函数无关。

def activation_input(node_id: int) -> str:
    return f"Relu_input_{node_id}{SUFFIX}"


def activation_input_scale(node_id: int) -> str:
    return f"Relu_input_sf_{node_id}{SUFFIX}"


def activation_input_zero_point(node_id: int) -> str:
    return f"Relu_input_zp_{node_id}{SUFFIX}"


# ---------------------------------------------------------------------------
# 连接信息的键名
# ---------------------------------------------------------------------------
#
# 每条连接在节点里记三次：`residual_*_buffer`、`<方向><端口>_node_id`、以及边。
# 三者必须同步。
#
# **端口号 >= 10 时 `residual_*_buffer` 后面多一个下划线**——这是对方生成器的
# 键名 bug（大概是端口号拼接时的字符串处理问题），实测 662 处无例外：
# 输出侧 88 处、输入侧 22 处带下划线，全部对应端口号 >= 10。
#
# 为兼容对方的解析器，照样复现。


def residual_buffer_key(direction: str, port: int) -> str:
    """`residual_input_buffer` / `residual_output_buffer` 的键名。

    `direction` 取 "input" 或 "output"。端口号 >= 10 时补那个多余的下划线。

        residual_buffer_key("output", 0)   -> "residual_output_buffer"
        residual_buffer_key("output", 10)  -> "residual_output_buffer_"
    """
    if direction not in ("input", "output"):
        raise ValueError(f"direction 只能是 input 或 output，给了 {direction!r}")
    suffix = "_" if port >= 10 else ""
    return f"residual_{direction}_buffer{suffix}"


def port_node_id_key(direction: str, port: int) -> str:
    """`input0_node_id` / `output10_node_id` 这类键名。"""
    if direction not in ("input", "output"):
        raise ValueError(f"direction 只能是 input 或 output，给了 {direction!r}")
    return f"{direction}{port}_node_id"


# ---------------------------------------------------------------------------
# phase 流水线（占全部 bin 的 57%）
# ---------------------------------------------------------------------------
#
# phase 型算子（DynamicScaling 4 相、Softmax 5 相）的每一相各有一套缓冲与定标
# 系数。这些文件**按本节点自命名**，不按消费者编号——因为它们是节点内部的
# 中间态，不流经任何边。实测 37 个带 rtl_version 的节点全部如此。
#
# 注意与非 phase 版的两处命名差异，都实测确认过：
#   - phase 版的 kantor Shift 多一段 `buffer_file`：
#       非 phase  kantor_A_Shift_36.bin
#       phase 版  kantor_A_Shift_buffer_file_phase_3_12.bin
#   - phase 版的定标三族用 `_phase_<k>` 而不是 `_file`：
#       非 phase  Scaling_buffer_file_195.bin
#       phase 版  Scaling_buffer_phase_0_12.bin


def phase_input_buffer(node_id: int, phase: int) -> str:
    return f"input_buffer_phase_{phase}_{node_id}{SUFFIX}"


def phase_output_buffer(node_id: int, phase: int) -> str:
    """某一相的输出。下游若走动态量化，会直接引用这个名字（不是自己的 sf）。"""
    return f"output_buffer_phase_{phase}_{node_id}{SUFFIX}"


def phase_output_buffer_self(node_id: int) -> str:
    """phase 型节点自命名的 `output_buffer`。

    这是命名契约的**唯一例外**：一般节点的 `output_buffer` 按消费者编号
    （缓冲区代表边），但 phase 型节点按**自己**编号。

    实测 37 个自命名节点与 37 个带 `rtl_version` 的节点完全重合
    （DynamicScaling 36 + Llama2ActivationDQ 1），其余 152 个算子节点
    都按消费者编号 —— 所以判据就是「有没有 rtl_version」。
    """
    return f"output_buffer_{node_id}{SUFFIX}"


def phase_fpsu_scale(node_id: int, phase: int) -> str:
    return f"Scaling_buffer_phase_{phase}_{node_id}{SUFFIX}"


def phase_fpsu_post_shift(node_id: int, phase: int) -> str:
    return f"Scaling_PS_buffer_phase_{phase}_{node_id}{SUFFIX}"


def phase_fpsu_bias(node_id: int, phase: int) -> str:
    return f"Bias_buffer_phase_{phase}_{node_id}{SUFFIX}"


def phase_lut(node_id: int, phase: int) -> str:
    return f"LUT_phase_{phase}_{node_id}{SUFFIX}"


def phase_kantor_scale(node_id: int, phase: int, block: str = "A") -> str:
    return f"kantor_{block}_scale_buffer_file_phase_{phase}_{node_id}{SUFFIX}"


def phase_kantor_bias(node_id: int, phase: int, block: str = "A") -> str:
    return f"kantor_{block}_bias_buffer_file_phase_{phase}_{node_id}{SUFFIX}"


def phase_kantor_shift(node_id: int, phase: int, block: str = "A") -> str:
    """比非 phase 版多一段 `buffer_file`——实测如此，不是笔误。"""
    return f"kantor_{block}_Shift_buffer_file_phase_{phase}_{node_id}{SUFFIX}"


# ---------------------------------------------------------------------------
# RoPE 子块
# ---------------------------------------------------------------------------
#
# RoPE 的两个节点（Llama2Activation / Llama2ActivationDQ）按固定命名的子块组织，
# 单元号 1..6 与子块一一绑定。
#
# **这里的 kantor 首字母大写**（`Kantor_A_...`），而 phase 版是小写
# （`kantor_A_...`）—— 实测 23 个大写文件全部属于 RoPE 子块，119 个小写文件
# 全部属于 phase 或非 phase 的普通节点。大小写写错就是悬空引用。

# 子块名。Sin/Cos 另有 `_Broadcast` 变体承载广播后的量化参数。
ROPE_BLOCKS = (
    "Llama2Activation_Add_Cos",
    "Llama2Activation_Add_Sin",
    "Llama2Activation_Sin",
    "Llama2Activation_Cos",
)


def rope_scale(node_id: int, unit: int, block: str) -> str:
    """RoPE 子块的 FPSU scale，带单元号。"""
    return f"Scaling_buffer_file_{unit}_{block}_{node_id}{SUFFIX}"


def rope_post_shift(node_id: int, unit: int, block: str) -> str:
    return f"Scaling_PS_buffer_file_{unit}_{block}_{node_id}{SUFFIX}"


def rope_bias(node_id: int, unit: int, block: str) -> str:
    return f"Bias_buffer_file_{unit}_{block}_{node_id}{SUFFIX}"


def rope_quant_scale(node_id: int, block: str) -> str:
    """子块输出的量化 scale。`block` 可带 `_Broadcast` 后缀。"""
    return f"{block}_sf_{node_id}{SUFFIX}"


def rope_quant_zero_point(node_id: int, block: str) -> str:
    return f"{block}_zp_{node_id}{SUFFIX}"


def rope_kantor_scale(node_id: int, block: str, unit: str = "A") -> str:
    """注意首字母大写的 `Kantor_`，与 phase 版的小写不同。"""
    return f"Kantor_{unit}_{block}_scale_buffer_file_{node_id}{SUFFIX}"


def rope_kantor_bias(node_id: int, block: str, unit: str = "A") -> str:
    return f"Kantor_{unit}_{block}_bias_buffer_file_{node_id}{SUFFIX}"


def rope_kantor_shift(node_id: int, block: str, unit: str = "A") -> str:
    return f"Kantor_{unit}_Shift_{block}_{node_id}{SUFFIX}"


# ---------------------------------------------------------------------------
# 算子专属常量
# ---------------------------------------------------------------------------


def rms_norm_epsilon(node_id: int) -> str:
    """RMSNorm 的 eps 常量，取自 config.json 的 rms_norm_eps。"""
    return f"RMSNorm_Add_Const_{node_id}{SUFFIX}"


def kv_cache_update_scale(node_id: int) -> str:
    """KV_Cache_DMA 写入值的量化 scale。"""
    return f"updates_sf_{node_id}{SUFFIX}"


def kv_cache_update_zero_point(node_id: int) -> str:
    return f"updates_zp_{node_id}{SUFFIX}"


def rope_intermediate(node_id: int, which: str) -> str:
    """RoPE 的两个中间态：`x*cos` 与 `rotate_half(x)*sin` 各自的乘积。

    `which` 取 "cos" 或 "sin"。命名不带 `Llama2Activation_` 前缀
    —— 实测就是 `cos_mul_output_<id>.bin`。
    """
    return f"{which}_mul_output_{node_id}{SUFFIX}"


def kv_updates_scale(node_id: int) -> str:
    """KV_Cache_DMA 写入 cache 的那一路（updates）的量化 scale。"""
    return f"updates_sf_{node_id}{SUFFIX}"


def kv_updates_zero_point(node_id: int) -> str:
    return f"updates_zp_{node_id}{SUFFIX}"
