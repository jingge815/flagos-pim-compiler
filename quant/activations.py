"""把 f32 激活量化成 GML 要的 int8 + per-tensor scale。

与权重不同，激活的数值公式**无法与实物做字节级对照**——实物没交付
`DEBUG_*_float` 浮点副本（见文档 26.6）。所以这里的实现依据是实物激活缓冲区的
**可观测特征**（见文档 34 节）：

    114 个激活缓冲区，全局值域 [-128, 127]，饱和率中位 0.34%

用满 int8 值域、饱和率很低但非零，说明是 per-tensor 对称量化，scale 按
`max(|x|)/127` 定，不做裁剪保护。这是标准做法，也与 `input_sf` 是标量的事实吻合。

**这是特征匹配，不是公式验证。** 真正确认要等与硬件对拍。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# int8 对称量化的值域。用 127 而不是 128 定 scale：值域 [-128, 127] 不对称，
# 按 128 缩放会让正端的 max(|x|) 映射到 128，超出上界后被裁剪。
INT8_MIN = -128
INT8_MAX = 127


@dataclass
class QuantizedActivation:
    """量化后的激活与它的 per-tensor scale。

    `scale` 是标量——实物的 `input_sf` 就是 2 字节 fp16 一个值，不是数组。
    """

    values: np.ndarray
    scale: np.float16

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float32) * np.float32(self.scale)

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


def quantize_activation(activation: np.ndarray) -> QuantizedActivation:
    """按 per-tensor 对称量化一个激活张量。

    全零张量的 scale 取 1 而不是 0——除以 0 会产出 nan，而全零激活量化后本就
    该是全零。
    """
    flat = np.ascontiguousarray(activation, dtype=np.float32).ravel()
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    scale = np.float16(peak / INT8_MAX) if peak > 0 else np.float16(1.0)

    # fp16 的 scale 可能舍入成 0（peak 极小时），那样反量化会全零。
    if float(scale) == 0.0:
        scale = np.float16(np.finfo(np.float16).tiny)

    quantized = np.rint(flat / np.float32(scale))
    values = np.clip(quantized, INT8_MIN, INT8_MAX).astype(np.int8)
    return QuantizedActivation(values, scale)


def quantization_error(
    activation: np.ndarray, quantized: QuantizedActivation
) -> float:
    """量化前后的最大绝对误差，相对于激活的幅度。"""
    flat = np.ascontiguousarray(activation, dtype=np.float32).ravel()
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    if peak == 0:
        return 0.0
    return float(np.abs(flat - quantized.dequantize()).max() / peak)
