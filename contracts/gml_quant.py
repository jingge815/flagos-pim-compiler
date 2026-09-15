"""GML 量化格式契约——全部来自 llama2 W4A8 实物的实测。

第 4 轮的量化模块按这里写 `.bin`。每一条都标注了实测依据，不是从 PDF 推的：
PDF 只给字段名，位宽与布局要看实物。

对应文档：第 19 节（定标折叠）、第 22 节（动态量化 phase）、第 25 节（LUT）。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 位宽与存储布局
# ---------------------------------------------------------------------------

# int4 权重一字节存一个值，**不打包**。实测 weight_buffer 的字节数等于权重元素数，
# 值域严格落在 [-8, 7]（16 个唯一值全覆盖），高 4 位是符号扩展。
INT4_BYTES_PER_VALUE = 1
INT4_MIN = -8
INT4_MAX = 7

# 权重按组量化，每组共享一个 fp16 scale。实测 16777216 字节权重 / 131072 个
# scale = 128，与 GML 的 DEBUG_weight_buffer_spg_group_size 吻合。
WEIGHT_GROUP_SIZE = 128

# 各类缓冲区的元素类型。实测取值，不是可选项。
DTYPES = {
    "activation": ("int8", "float16"),   # 激活：定点段 int8，浮点段 fp16
    "weight": ("int4", "int8"),          # W4A8 里权重是 int4；部分算子仍用 int8
    "bias": ("int32",),                  # 偏置恒为 int32
    "scale": ("float16", "float32"),     # scale 主要 fp16，少数 fp32
    "output": ("int8", "float16", "int16"),
}


@dataclass(frozen=True)
class QuantLayout:
    """一个张量的量化参数布局。

    `granularity` 取 per_tensor / per_channel / per_group，与 `#pim.quant_spec`
    的三档一一对应。`group_size` 只在 per_group 时有意义。
    """

    granularity: str
    group_size: int = 0
    axis: int = 0

    def scale_count(self, shape: tuple[int, ...]) -> int:
        """这个布局需要多少个 scale。"""
        if self.granularity == "per_tensor":
            return 1
        extent = shape[self.axis]
        if self.granularity == "per_group":
            if extent % self.group_size:
                raise ValueError(
                    f"轴长 {extent} 不能被 group_size {self.group_size} 整除")
            return extent // self.group_size
        return extent


# 实测的两种布局：激活按整张量一个 scale，权重按 128 分组。
ACTIVATION_LAYOUT = QuantLayout("per_tensor")
WEIGHT_LAYOUT = QuantLayout("per_group", group_size=WEIGHT_GROUP_SIZE, axis=0)


# ---------------------------------------------------------------------------
# 定标：Scaling 不是量化因子
# ---------------------------------------------------------------------------
#
# 第 19 节解开的一条：`Scaling_buffer_file` 承载的是**算子自身的数学缩放**，
# 不是 input_sf·weight_sf/output_sf。实测 llama2 全图 Scaling 都是标量，取值：
#
#   fp16(1/√128) × 32   ← attention scores，32 个 head
#   1.0          × 37   ← 不缩放
#   0.5 / 0.25   × 各1  ← 2 的幂修正
#   2.0          × 2    ← KV_Cache_DMA
#
# 量化定标全部由 per-group 的 weight_sf 与 per-tensor 的 input_sf/output_sf 承担。

# attention 的 1/√head_dim。按 fp16 存，所以要先转再写。
def attention_scale(head_dim: int) -> float:
    """attention scores 的缩放因子，即 1/√head_dim。"""
    return head_dim ** -0.5


# ---------------------------------------------------------------------------
# 动态量化的 phase 流水线
# ---------------------------------------------------------------------------
#
# 第 22 节：llama2 的 use_dynamic_quantization 全为 1，scale 在硬件上算，
# 不是编译期常量。phase1 沿 1024 宽度做归约求极值，phase2/3/4 变换与施加。

# phase 数。GML 字段是 *_phase_0 到 *_phase_4。
DQ_PHASE_COUNT = 5

# phase1 的归约宽度，实测恒为 1024。
DQ_REDUCTION_WIDTH = 1024


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------
#
# 第 25 节：固定 288 字节 = 144 个 fp16，分三段。llama2 用满约 100 项，
# ResNet50 的 Relu 只用 26 项，全零表也合法（llama2 有 37 个）。
#
# **采样规则未确认**：按等距采样激活函数拟合不成立（SiLU 最大误差 0.27）。
# 所以这里只固定尺寸，不提供生成函数——生成规则要等对方确认。

LUT_BYTES = 288
LUT_ENTRY_COUNT = 144
LUT_ENTRY_DTYPE = "float16"


def lut_placeholder() -> bytes:
    """结构层的占位 LUT：全零。

    注意实物里**没有**全零表。139 个 LUT 的非零项数分布是
    `{1: 37, 98: 32, 99: 1, 102: 69}`——那 37 个是「仅下标 0 为 1.0」的恒等表，
    不是空表。所以这个占位值只能用来把结构跑通，不能当成合法产物交付。
    """
    return bytes(LUT_BYTES)


def lut_identity() -> bytes:
    """恒等 LUT：仅第 0 项为 1.0，其余为 0。

    实物里有 37 个这样的表，是真实的合法值——用在不需要变换的位置。
    """
    import numpy as np

    table = np.zeros(LUT_ENTRY_COUNT, dtype=np.float16)
    table[0] = 1.0
    return table.tobytes()
