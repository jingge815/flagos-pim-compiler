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
