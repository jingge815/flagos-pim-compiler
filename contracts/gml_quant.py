"""GML 量化格式契约——全部来自 llama2 W4A8 实物的实测。

第 4 轮的量化模块按这里写 `.bin`。每一条都标注了实测依据，不是从 PDF 推的：
PDF 只给字段名，位宽与布局要看实物。

对应文档：第 19 节（定标折叠）、第 22 节（动态量化 phase）、第 25 节（LUT）。
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 图头
# ---------------------------------------------------------------------------

# GML 格式版本号，对方的解析器按它选解析分支。照填参考产物
# llama2_w4a8_decode_block_0 的值。它标识**格式版本**，不表示我方用了 relay
# —— 我方不走 TVM，直接从 HF 模型 + torch.fx 构图。
#
# 仍待对方确认我方该填什么（见 docs/gml-parser-output-plan-20260917.md §7 问题 2）。
GML_VERSION = "26.2.1"


# ---------------------------------------------------------------------------
# 位宽与存储布局
# ---------------------------------------------------------------------------

# int4 权重一字节存一个值，**不打包**。实测 weight_buffer 的字节数等于权重元素数，
# 值域严格落在 [-8, 7]（16 个唯一值全覆盖），高 4 位是符号扩展。
INT4_BYTES_PER_VALUE = 1
INT4_MIN = -8
INT4_MAX = 7

# int8 激活的值域。
INT8_MIN = -128
INT8_MAX = 127

# 定 scale 用的满量程分母，与上面的 clamp 边界**不是同一个数**。
# 分母取 2^(bits-1) 而不是 2^(bits-1)-1：absmax 映射到 8（int4）/ 128（int8）。
#
# 实测依据（不是推导）：
#   - 分母若取 7，则 |q| = 8 在数学上不可能出现（round(absmax/(absmax/7)) = 7）。
#     实测 5 个权重张量共 15000 组，max|q| 分布 {7: 5538, 8: 7801, ...}
#     —— 7801 组（52%）含 q = -8，直接排除分母 7。
#   - 激活侧：实测 output_sf == absmax/128 在 32/32 组成立（而非 /127），
#     且 int8 用满 [-128, 127]。
#     硬件路径是 p0 = 2*absmax、再 ×256，合起来等价于除以 absmax/128（见 DQ 四相）。
INT4_SCALE_DIVISOR = 8
INT8_SCALE_DIVISOR = 128

# 权重按组量化，每组共享一个 fp16 scale。实测 16777216 字节权重 / 131072 个
# scale = 128，与 GML 的 DEBUG_weight_buffer_spg_group_size 吻合。
WEIGHT_GROUP_SIZE = 128

# 各类缓冲区的元素类型。实测取值，不是可选项。
DTYPES = {
    "activation": ("int8", "float16"),   # 激活：定点段 int8，浮点段 fp16
    "weight": ("int4", "int8"),          # W4A8 里权重是 int4；部分算子仍用 int8
    # 偏置：定点通路是 int32，浮点通路是 fp32。两者都是 4 字节，所以落盘宽度相同，
    # 但**写值时必须按对应类型打包** —— DQ 的 Bias_buffer_phase_0 = 2⁻⁶³
    # 只有按 fp32 解才成立（按 int32 解是个无意义的小整数）。
    "bias": ("int32", "float32"),
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
        """这个布局需要多少个 scale。

        per_group 是 **per-channel + per-group 同时开启**（实测 spc=1 且 spg=1），
        分组沿最后一维切，所以 scale 总数是 `numel / group_size`
        —— 每行切若干组、所有行累加，不是「某一根轴的长度 / group_size」。

        实测核对：
            gate_proj [11008, 4096] -> 4096/128=32 组/行 × 11008 = 352256
            down_proj [4096, 11008] -> 11008/128=86 组/行 × 4096 = 352256
        两者 scale 元素数相同（都等于 numel/128）但**排布顺序不同**，
        因为分组轴上的长度不同。搞错会让反量化整体错位。
        """
        if self.granularity == "per_tensor":
            return 1
        if self.granularity == "per_group":
            if shape[-1] % self.group_size:
                raise ValueError(
                    f"最后一维 {shape[-1]} 不能被 group_size {self.group_size} 整除")
            numel = 1
            for extent in shape:
                numel *= extent
            return numel // self.group_size
        return shape[self.axis]


# 实测的布局：**激活与权重都是 per-channel + per-group、group_size=128、沿最后一维**。
#
# 激活侧原先记为 per_tensor 是错的：实测 output_sf_12 有 32 个 scale（4096/128）、
# output_sf_193 有 86 个（11008/128），不是标量。按 per_tensor 分配会让 *_sf
# 文件只有 2 字节而实际需要 64 / 172 字节。
ACTIVATION_LAYOUT = QuantLayout("per_group", group_size=WEIGHT_GROUP_SIZE, axis=-1)
WEIGHT_LAYOUT = QuantLayout("per_group", group_size=WEIGHT_GROUP_SIZE, axis=-1)


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
# llama2 的 use_dynamic_quantization 全为 1，scale 在硬件上算，不是编译期常量。
#
# phase 在 GML 里**不是独立节点**，而是同一个节点内的字段族（*_phase_0..4）。
# 它只在层参数文本里才展开成独立层（200 节点 -> 422 层）。

# 相数按算子分，不是统一的。实测 DynamicScaling 36/36 个节点是 4 相
# （p0 求组统计量 -> p1 ×1/256 -> p2 取倒数 -> p3 Kantor fp2int），
# Softmax 32/32 个节点是 5 相
# （p0 求 max -> p1 exp -> p2 求和 -> p3 取倒数 -> p4 归一化）。
# 原先统一记 5 相会给 DQ 多分配一整套 phase 文件与字段。
PHASE_COUNTS = {
    "DynamicScaling": 4,
    "Llama2ActivationDQ": 4,
    "Softmax": 5,
}

# 动态量化的分组宽度。**按被量化的张量变化**，不是常量——
# 它是唯一按节点变化的 phase 字段（global_pooling_group_size_phase_0）。
#
#   hidden [1,1,1,4096]      -> 128（32 组）
#   MLP 中间态 [1,1,1,11008] -> 128（86 组）
#   attention scores [...1024] -> 1024（整条一组）
#
# 规则：默认取量化契约的 group_size；attention scores 整条当一组。
DQ_GROUP_SIZE_DEFAULT = WEIGHT_GROUP_SIZE
DQ_GROUP_SIZE_ATTENTION_SCORES = 1024


def dq_group_size(numel: int, *, is_attention_scores: bool) -> int:
    """一个 DQ 节点的分组宽度。attention scores 整条当一组，其余按 128 切。"""
    if is_attention_scores:
        return numel
    return DQ_GROUP_SIZE_DEFAULT


# ---------------------------------------------------------------------------
# LUT
# ---------------------------------------------------------------------------
#
# 固定 288 字节 = 144 个 fp16。结构已由硬件规范 §4.3.3 确定：
# **32 段分段线性（PWL），每段一个 slope 与一个 intercept**，求值 y = A[i]·x + B[i]。
#
#   [0:32]    slope A[i]        [32:64]  intercept B[i]
#   [64:104]  未初始化残留，非参数（恒等表该区全 0 仍能正常工作）-> 写 0
#   [104:144] 填充 -> 写 0
#
# 生成函数在 contracts/gml_lut.py：4 张表里 3 张可自行合成
# （恒等表字节级复现参考产物、倒数表精度反超参考产物），只有 exp 表需拷贝一次。
# 原先「采样规则未确认所以不提供生成函数」的判断已被硬件规范推翻。

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
    """恒等 LUT：仅第 0 项（slope A[0]）为 1.0，其余为 0。

    实物里有 37 个这样的表，是真实的合法值——用在走 LUT 通路但不做变换的位置
    （DQ 的 phase1）。实测与参考产物 `LUT_phase_1_12.bin` 字节完全相同。

    实现委托给 `contracts.gml_lut.synth_identity()`，那里是 LUT 的单一真源
    （按 32 段 PWL 的 slope/intercept 布局打包）。这里保留这个名字是因为
    既有调用方按它引用；两者字节相同，有测试钉住。
    """
    from contracts.gml_lut import synth_identity

    return synth_identity()
