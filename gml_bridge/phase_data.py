"""phase 流水线的数值计算：DynamicScaling 四相与 Softmax 五相。

这两条流水线贡献了参考产物 3231 个 bin 里的 1789 个（57%）。公式全部由实测反推
并逐元素验证过，判据记在每个函数的 docstring 里。

**两个归约相的 4 字节编码不同**，这是最容易静默写错的地方：

    Softmax phase0（-max）  fp16 位模式放**高 2 字节**，低 2 字节为 0
    Softmax phase2（Σexp）  真正的 fp32

判据：若 phase0 是真 fp32，`fp32(-2.98047)` 应为 `00c03ec0`，而实测是 `0000f6c1`，
正是 `pack('<HH', 0, fp16_bits(-max))` 的字节。两者在 32/32 个节点上都成立。

**与实物的一处已知差异（有意为之）**：硬件的 Softmax phase1 走 31 段 PWL exp 表，
与精确 `exp` 差约 0.7%（逐元素峰值相对误差），求和差约 2.1%。本模块用**精确
`exp`**，所以 phase1/2/3/4 的数值与参考产物逐字节不同 —— 这是逼近误差的差异，
不是公式错：

- 只依赖 `max(x)` 的 phase0 与参考产物**逐字节相同**（实测 32/32 节点）。
- 把参考产物自己的 phase1 喂进来，phase4 = phase1 × phase3 逐元素吻合
  1024/1024，`Σ = 0.99975` —— 公式本身已验证。
- 我方产物内部的结构恒等式（`Bias_phase_1 == phase0`、
  `Scaling_phase_4 == phase3`）按构造成立，与实物的 32/32 一致。

若将来要求逐字节复现硬件，把 `contracts.gml_lut` 的 exp 表接进 phase1 即可
（那张表的段索引规则尚未反推出，见计划 §7 问题 1）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from contracts.gml_quant import INT8_MAX, INT8_MIN

# DQ 的定标常量。实测在全部 36 个 DynamicScaling 节点上逐字节一致。
#
# 这三个不是「随便的缩放」，它们合起来构成量化除数：
#   p0 = 2 * absmax     （左移 1 位）
#   p1 = p0 / 256       （DQ_PHASE1_SCALE）
#   q  = x * (1/p0) * 256
# 即等效满量程分母是 2*256/2 = 128，与 output_sf == absmax/128 吻合。
DQ_PHASE1_SCALE = 1.0 / 256.0
DQ_PHASE3_SCALE = 256.0

# DQ phase0 的 FPSU bias，实测恒为 2^-63 = 1.0842021724855044e-19。
#
# 这是写进 `Bias_buffer_phase_0_<id>.bin` 的**fp32 常量**，不能写 0。
# 它加在 FPSU 的 32 位累加器上（硬件规范：FPSU 加 32 位 bias、乘 16 位 scale、
# round 后右移），起下限保护作用。
#
# **注意它不参与 fp16 域的 phase0 数值**：2^-63 在 fp16 里下溢成 0
# （fp16 最小次正规数是 5.96e-08），所以把它加进 phase0 再截 fp16 是个空操作。
# 全零组的除零保护另做，见 `dynamic_scaling`。
DQ_PHASE0_BIAS = 2.0 ** -63

# DQ phase3 的 Kantor 右移量，实测 int8 恒为 -8（即左移 8 位 = ×256）。
DQ_PHASE3_SHIFT = -8

# Softmax phase1 的 FPSU scale，实测恒为 0.5。
SOFTMAX_PHASE1_SCALE = 0.5


def _fp16(values: np.ndarray) -> np.ndarray:
    """按 fp16 落盘的精度截断。中间量必须逐相截断，否则与硬件对不上。"""
    return values.astype(np.float16)


def pack_fp16_in_high_half(value: float) -> bytes:
    """把一个 fp16 放进 4 字节的**高 2 字节**，低 2 字节填 0。

    这是 Softmax phase0 的落盘编码。实测 32/32 个节点：低 2 字节全为 0，
    高 2 字节等于 `fp16(-max)` 的位模式。

    副作用：这 4 字节按 fp32 解会得到一个「看似合理」的值
    （实测 -30.75，而真实的 -max 是 -2.98），所以按 fp32 读会静默读出错误的数。
    """
    bits = struct.unpack("<H", struct.pack("<e", np.float16(value)))[0]
    return struct.pack("<HH", 0, bits)


def unpack_fp16_from_high_half(raw: bytes) -> float:
    """`pack_fp16_in_high_half` 的逆操作，用于回读校验。"""
    if len(raw) != 4:
        raise ValueError(f"要 4 字节，给了 {len(raw)}")
    return float(struct.unpack("<e", raw[2:])[0])


@dataclass
class DynamicScalingPhases:
    """DynamicScaling 一个节点的四相中间态。

    `group_size` 是分组宽度：hidden 与 MLP 中间态取 128，
    attention scores 整条一组（group_size = numel）。
    """

    source: np.ndarray          # 输入（fp16），phase0 的 input_buffer
    phase0: np.ndarray          # 2 * absmax，逐组（fp16）
    phase1: np.ndarray          # phase0 / 256，逐组 —— 就是 output_sf
    phase2: np.ndarray          # 1 / phase0，逐组 —— 就是 kantor_A_scale
    phase3: np.ndarray          # 量化结果（int8）
    group_size: int

    @property
    def output_scale(self) -> np.ndarray:
        """输出量化 scale。**逐字节等于 phase1**，实测 32/32 组。

        下游节点走动态量化时，它的 `input_sf` 直接引用本节点的
        `output_buffer_phase_1_<id>.bin`，而不是自己写一份。
        """
        return self.phase1

    @property
    def kantor_scale(self) -> np.ndarray:
        """phase3 的 Kantor scale。**逐字节等于 phase2**，实测 32/32 组。"""
        return self.phase2


def dynamic_scaling(
    source: np.ndarray, *, group_size: int | None = 128
) -> DynamicScalingPhases:
    """算一个 DynamicScaling 节点的四相。

    公式（每一步都在实物上逐元素验证过，节点 12 / 4096 元素 / 32 组）：

        absmax[g] = max(|x[i]|)  for i in group g
        p0[g]     = 2 * absmax[g]                          # 32/32 组吻合
        p1[g]     = p0[g] / 256                            # 32/32 组吻合
        p2[g]     = 1 / p0[g]                              # 32/32 组吻合
        q[i]      = clamp(round(x[i] * p2[g] * 256), -128, 127)   # 4096/4096 吻合

    `p0` 用 `2*absmax` 而不是 `absmax`，配合 `p1 = p0/256`，等效满量程分母是
    **128** —— 所以 `output_sf == absmax/128`（实测成立）。

    全零组：`p0 = 0` 会让 `p2` 溢出成 inf，进而让 phase3 出 nan。这里把这种组的
    `p2` 直接置 0 —— 全零输入乘任何有限系数都是 0，所以量化结果不受影响，
    而 inf/nan 会污染落盘字节。

    （`DQ_PHASE0_BIAS` 不能用来兜这个底：2^-63 在 fp16 域下溢成 0。它是写进
    `Bias_buffer_phase_0` 的 fp32 常量，作用在硬件的 32 位累加器上。）
    """
    flat = np.ascontiguousarray(source, dtype=np.float32).ravel()
    width = flat.size if group_size is None else group_size
    if width <= 0 or flat.size % width:
        raise ValueError(f"{flat.size} 个元素不能被 group_size {width} 整除")

    grouped = flat.reshape(-1, width)

    # phase0：Pooling 块求每组对称动态范围，再左移 1 位。
    absmax = np.abs(grouped).max(axis=1)
    phase0 = _fp16(2.0 * absmax)

    # phase1：乘 1/256。这一相同时是输出的量化 scale。
    phase1 = _fp16(phase0.astype(np.float32) * DQ_PHASE1_SCALE)

    # phase2：取倒数，走倒数 LUT。全零组置 0 而不是 inf（见 docstring）。
    nonzero = phase0.astype(np.float32)
    phase2 = _fp16(np.divide(
        1.0, nonzero, out=np.zeros_like(nonzero), where=nonzero != 0))

    # phase3：Kantor 定点化，×256 由 DQ_PHASE3_SCALE / Shift=-8 完成。
    scaled = grouped * phase2.astype(np.float32)[:, None] * DQ_PHASE3_SCALE
    phase3 = np.clip(np.rint(scaled), INT8_MIN, INT8_MAX).astype(np.int8)

    return DynamicScalingPhases(
        source=_fp16(flat), phase0=phase0, phase1=phase1, phase2=phase2,
        phase3=phase3.ravel(), group_size=width)


@dataclass
class SoftmaxPhases:
    """Softmax 一个节点的五相中间态。

    `phase0` 与 `phase2` 都是标量归约结果，但**落盘编码不同**——
    用 `phase0_bytes` / `phase2_bytes` 取字节，不要自己打包。
    """

    source: np.ndarray          # 输入（fp16）
    phase0: float               # -max(x)
    phase1: np.ndarray          # exp(x - max)（fp16）
    phase2: float               # Σ phase1（fp32 归约）
    phase3: float               # 1 / phase2
    phase4: np.ndarray          # phase1 * phase3，即 softmax 输出（fp16）

    @property
    def phase0_bytes(self) -> bytes:
        """fp16 位模式放高 2 字节。实测 32/32 个节点如此。"""
        return pack_fp16_in_high_half(self.phase0)

    @property
    def phase2_bytes(self) -> bytes:
        """真正的 fp32。与 phase0 的编码**不同**。"""
        return struct.pack("<f", np.float32(self.phase2))

    @property
    def phase1_bias(self) -> bytes:
        """`Bias_buffer_phase_1` 的内容：逐字节等于 phase0 的输出。

        实测 32/32 个节点吻合。这说明它是**运行时归约落点**而不是常量——
        参考产物里解出 -30.75 只是那份合成输入的 max，不能硬编码。
        """
        return self.phase0_bytes

    @property
    def phase4_scale(self) -> bytes:
        """`Scaling_buffer_phase_4` 的内容：逐字节等于 phase3 的输出。

        实测 32/32 个节点吻合。同样是运行时落点，不是常量。
        """
        return struct.pack("<e", np.float16(self.phase3))


def softmax(source: np.ndarray) -> SoftmaxPhases:
    """算一个 Softmax 节点的五相。

    公式（节点 18 / 1024 元素，逐元素验证）：

        m    = max(x)          # phase0 落盘 -m
        e[i] = exp(x[i] - m)   # phase1
        S    = Σ e[i]          # phase2，fp32 归约（实测 84.875 vs 求和 84.8706）
        r    = 1 / S           # phase3（实测 1 个 fp16 ULP 内）
        y[i] = e[i] * r        # phase4（实测 1024/1024 逐元素吻合，Σ = 0.99975）

    减 max 是标准的数值稳定化。phase1 在硬件上是「FPSU 做 0.5x + bias 仿射，
    再过 exp 表」，其中 bias 就是 phase0 的输出 —— 所以那个 0.5
    （`SOFTMAX_PHASE1_SCALE`）与 exp 表的定域是配套的。
    """
    flat = np.ascontiguousarray(source, dtype=np.float32).ravel()
    if not flat.size:
        raise ValueError("Softmax 的输入不能为空")

    # 用 fp16 截断后的 max 参与计算：硬件的 phase0 就是按 fp16 落盘的，
    # 后续各相读的是那个截断值，不是原始 f32 的 max。
    maximum = float(np.float16(flat.max()))
    phase1 = _fp16(np.exp(flat - maximum))

    total = float(np.float32(phase1.astype(np.float32).sum()))
    reciprocal = float(np.float16(1.0 / total))
    phase4 = _fp16(phase1.astype(np.float32) * reciprocal)

    return SoftmaxPhases(
        source=_fp16(flat), phase0=-maximum, phase1=phase1,
        phase2=total, phase3=reciprocal, phase4=phase4)
