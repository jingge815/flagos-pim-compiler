"""合成 GML 的分段线性查找表（LUT）。

硬件依据：Ceva-NeuPro-M 规范 §4.3.3 —— Activation 单元用 **32 段分段线性**
逼近非线性函数，每段由一个 slope 与一个 intercept 定义，求值 `y = A[i]*x + B[i]`。

288 字节 = 144 个 fp16，实测布局与规范精确吻合：

    [0:32]     slope     A[i]     （第 32 项恒 0，实际用 31 段）
    [32:64]    intercept B[i]     （第 32 项恒 0）
    [64:104]   未初始化残留，**非参数** -> 写 0
    [104:144]  填充 -> 写 0

`[64:104]` 是残留而非参数的判据：恒等表该区 40 项全为 0，而它是一张能正常工作的
表（DQ phase1 走它）；若该区承载必需参数，全零的表不可能工作。

全图 139 个 LUT 只有 **4 种内容**：倒数 69、恒等 37、exp 32、SiLU 1。
其中 3 种可自行合成，只有 exp 需要拷贝一次（段索引规则未能反推，见
docs/gml-parser-output-plan-20260917.md §7 问题 1）。
"""

from __future__ import annotations

import math
import struct

from contracts.gml_quant import LUT_ENTRY_COUNT

# 段数与可用段数。第 31 段（下标 31）在实物里恒为 0，实际只用 0..30。
LUT_SEGMENTS = 32
LUT_USABLE_SEGMENTS = 31


def pack_lut(slopes: list[float], intercepts: list[float]) -> bytes:
    """按硬件布局打包一张表：slope 段 + intercept 段 + 80 个 0。

    `[64:144]` 那 80 项全写 0：前 40 项是未初始化残留（非参数），后 40 项是填充。
    """
    if len(slopes) != LUT_SEGMENTS or len(intercepts) != LUT_SEGMENTS:
        raise ValueError(f"slope 与 intercept 都要 {LUT_SEGMENTS} 项")
    values = list(slopes) + list(intercepts) + [0.0] * 80
    assert len(values) == LUT_ENTRY_COUNT
    return struct.pack("<" + "e" * LUT_ENTRY_COUNT, *values)


def synth_identity() -> bytes:
    """恒等表：只有 `A[0] = 1.0`，其余全 0，配 `activation_mode = 1`。

    DQ 的 phase1 走 LUT 通路但不做非线性变换——真正的 `/256` 由
    `Scaling_buffer_phase_1` 完成。

    实测：按此合成的 288 字节与参考产物 `LUT_phase_1_12.bin` **字节完全相同**。
    """
    slopes = [0.0] * LUT_SEGMENTS
    slopes[0] = 1.0
    return pack_lut(slopes, [0.0] * LUT_SEGMENTS)


def synth_reciprocal() -> bytes:
    """倒数表 `1/x`：每段取 `1/x` 的**切线**，切点为均匀中点。

    过点 p 的 `1/x` 切线是 `y = -x/p^2 + 2/p`，故 `A = -1/p^2`、`B = 2/p`，
    两者满足代数签名 **`A = -B^2/4`** —— 实测参考产物 31 段全部满足
    （平均偏差 0.000167，在 fp16 分辨率内即精确），这是判定「切线族而非弦线」的依据。

    切点取 `p_i = 1 + (i+0.5)/32`，覆盖 fp16 归一化尾数域 `[1, 2)`。

    精度实测（密集扫过尾数域全部 992 个 fp16 值，不是只在切点上比 ——
    在切点上比是自证，切线在切点处误差恒为 0）：

        本函数（表项已按 fp16 落盘）  平均 0.045%
        参考产物                      平均 0.387%

    对硬件实测输出（`output_buffer_phase_2 == 1/p0`）逐节点对拍同样领先：
    节点 12 / 193 / 196 分别是 0.046% / 0.049% / 0.039%。

    注：系数本身用 f64 算时平均误差是 0.004%，落成 fp16 后退化到 0.045%
    —— **fp16 的表项精度是瓶颈，不是切点选取**。即便如此仍比参考产物准 8 倍，
    所以这张表自行合成即可，不必拷贝。
    """
    slopes = [0.0] * LUT_SEGMENTS
    intercepts = [0.0] * LUT_SEGMENTS
    for i in range(LUT_USABLE_SEGMENTS):
        p = 1.0 + (i + 0.5) / LUT_SEGMENTS
        slopes[i] = -1.0 / (p * p)
        intercepts[i] = 2.0 / p
    return pack_lut(slopes, intercepts)


def synth_decaying(fn, lo: float, hi: float, *, segments: int = 30) -> bytes:
    """合成 exp / SiLU 一类在 -inf 侧衰减到 0 的函数：每段取弦线。

    段 0 留 0（承担「饱和到 0」），有效段放在 1..segments，均匀覆盖 `[lo, hi)`。
    判据：实测 exp 与 SiLU 表的 `A[0] = B[0] = 0`，而倒数表的第 0 段是实值
    （`1/x` 在 `[1,2)` 内不衰减）——所以只有衰减型函数才留空第 0 段。
    """
    slopes = [0.0] * LUT_SEGMENTS
    intercepts = [0.0] * LUT_SEGMENTS
    width = (hi - lo) / segments
    for k in range(segments):
        x0, x1 = lo + k * width, lo + (k + 1) * width
        y0, y1 = fn(x0), fn(x1)
        slope = (y1 - y0) / (x1 - x0)
        slopes[1 + k] = slope
        intercepts[1 + k] = y0 - slope * x0
    return pack_lut(slopes, intercepts)


def synth_silu() -> bytes:
    """SiLU 表 `x*sigmoid(x)`，折进 Gemm 的 contraction。

    定域 `[-4, 4)` 是按真值拟合选出的，**不是实测确认的** —— 参考产物只有
    1 张 SiLU 表，无法从单一样本反推对方的定域。本函数在该定域内的
    平均绝对误差实测为 0.0011（30 段弦线，采样 300 点）。

    若对精度有疑虑，拷参考产物的 `activation_lut_file_195.bin` 最稳妥；
    但那张表与本函数的系数有可见差异，说明对方用了不同的定域或段点准则。
    """
    return synth_decaying(lambda x: x / (1.0 + math.exp(-x)), -4.0, 4.0)


def decode_reciprocal(value: float, table: bytes) -> float:
    """解码侧参考实现：用一张倒数表求 `1/value`，用来对拍合成结果。

    段索引取 fp16 **尾数高 5 位**，指数部分由硬件单独处理（先把值归一化到
    `[1,2)` 查表，再按指数还原）。这个取址方式是实测反推的：按它配合
    `synth_reciprocal()` 重算 DQ 节点的倒数，与硬件输出平均差 0.04%
    （用参考产物那张表则是 0.4%）。
    """
    entries = struct.unpack("<" + "e" * LUT_ENTRY_COUNT, table)
    slopes, intercepts = entries[0:32], entries[32:64]

    bits = struct.unpack("<H", struct.pack("<e", value))[0]
    exponent = (bits >> 10) & 0x1F
    mantissa_bits = bits & 0x3FF
    segment = mantissa_bits >> 5
    mantissa = 1.0 + mantissa_bits / 1024.0
    return (slopes[segment] * mantissa + intercepts[segment]) * 2.0 ** (15 - exponent)
