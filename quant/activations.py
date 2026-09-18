"""把 f32 激活量化成 GML 要的 int8 + per-group scale。

**动态量化的四相公式已完整反推并逐元素验证**（见
docs/gml-parser-output-plan-20260917.md §1.2），所以 scale 的定法不是猜的：

    p0[g]  = 2 * absmax[g]            # 硬件 Pooling 块算对称 DR 后左移 1 位
    sf[g]  = p0[g] / 256              # == output_sf，实测 32/32 组逐字节吻合
           = absmax[g] / 128          # 等价形式：absmax 映射到 int8 满量程 128
    q[i]   = clamp(round(x[i] / sf[g]), -128, 127)

两条关键修正（原实现两处都错）：

- **分母是 128 不是 127**：实测 `output_sf == absmax/128` 在 32/32 组成立，
  且 int8 用满 `[-128, 127]` 全域。
- **粒度是 per-group（group_size=128）不是 per-tensor**：实测
  `output_sf_12` 有 32 个 scale（4096/128）、`output_sf_193` 有 86 个（11008/128）。
  按 per-tensor 只写 2 字节，而实际需要 64 / 172 字节。

attention scores 那 32 个节点是整条当一组（group_size = numel），
所以 per-tensor 是 per-group 的一个特例，用 `group_size=None` 表示。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 值域与「定 scale 用的满量程分母」不是同一个数：clamp 到 [-128, 127]，
# 但分母取 128（absmax 映射到 128，正端那一个值会 clamp 成 127）。
from contracts.gml_quant import (
    INT8_MAX,
    INT8_MIN,
    INT8_SCALE_DIVISOR,
    WEIGHT_GROUP_SIZE,
)


@dataclass
class QuantizedActivation:
    """量化后的激活与它的 per-group scale。

    `scales` 是 fp16 数组，每 `group_size` 个元素一个。整条一组时长度为 1
    —— 那正是 attention scores 的情形，落盘就是 2 字节。
    """

    values: np.ndarray
    scales: np.ndarray
    group_size: int

    @property
    def scale(self) -> np.float16:
        """单组时的标量 scale。多组时取它是语义错误，直接抛。"""
        if self.scales.size != 1:
            raise ValueError(
                f"这是 {self.scales.size} 组的 per-group 量化，没有单一 scale")
        return self.scales[0]

    def dequantize(self) -> np.ndarray:
        grouped = self.values.reshape(-1, self.group_size).astype(np.float32)
        return (grouped * self.scales.astype(np.float32)[:, None]).ravel()

    def saturation_ratio(self) -> float:
        """取到 ±128 的比例。

        实物的中位值是 0.0034。这个指标能反映 scale 是否定得合理：过高说明
        scale 偏小、大量值被裁剪；恰好为 0 反而可疑——说明 scale 偏大，
        精度没用满。
        """
        if not self.values.size:
            return 0.0
        extremes = (self.values == INT8_MIN) | (self.values == INT8_MAX)
        return float(extremes.mean())


def quantize_activation(
    activation: np.ndarray, *, group_size: int | None = WEIGHT_GROUP_SIZE
) -> QuantizedActivation:
    """按 per-group 对称量化一个激活张量。

    `group_size=None` 表示整条当一组（attention scores 的情形）。
    元素数必须能被 `group_size` 整除——不能整除就抛，静默补零会让 scale 与
    数据错位，而这种错在结构校验里看不出来。

    全零组的 scale 取 1 而不是 0——除以 0 会产出 nan，而全零激活量化后本就
    该是全零。
    """
    flat = np.ascontiguousarray(activation, dtype=np.float32).ravel()
    width = flat.size if group_size is None else group_size
    if width <= 0 or flat.size % width:
        raise ValueError(
            f"激活有 {flat.size} 个元素，不能被 group_size {width} 整除")

    grouped = flat.reshape(-1, width)
    peak = np.abs(grouped).max(axis=1)
    scales = np.where(
        peak > 0, peak / INT8_SCALE_DIVISOR, 1.0).astype(np.float16)

    # fp16 的 scale 可能舍入成 0（peak 极小时），那样反量化会全零。
    scales = np.where(
        scales.astype(np.float32) == 0.0,
        np.float16(np.finfo(np.float16).tiny), scales).astype(np.float16)

    quantized = np.rint(grouped / scales.astype(np.float32)[:, None])
    values = np.clip(quantized, INT8_MIN, INT8_MAX).astype(np.int8)
    return QuantizedActivation(values.ravel(), scales, width)


def quantization_error(
    activation: np.ndarray, quantized: QuantizedActivation
) -> float:
    """量化前后的最大绝对误差，相对于激活的幅度。"""
    flat = np.ascontiguousarray(activation, dtype=np.float32).ravel()
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    if peak == 0:
        return 0.0
    return float(np.abs(flat - quantized.dequantize()).max() / peak)
