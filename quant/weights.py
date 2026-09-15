"""把 f32 权重量化成 GML 要的 int4 + per-group scale。

公式与布局都已用实物字节级验证过（见 docs/gml-lowering-20260914.md 26.5）：
16777216 个权重的反量化再量化 100% 一致，所以这里的量化方向也是确定的。

只做权重。激活量化的数值公式无从对照——实物没交付 `DEBUG_*_float` 浮点副本
（见 26.6），所以那部分等与硬件对拍时再定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.gml_quant import INT4_MAX, INT4_MIN, WEIGHT_GROUP_SIZE


@dataclass
class QuantizedWeight:
    """量化后的权重与它的 per-group scale。

    `values` 是 int8 数组，但每个元素的值域限定在 int4 的 `[-8, 7]`——实物就是
    这样存的，一字节一个值，不打包。`scales` 是 fp16，每 `group_size` 个权重一个。
    """

    values: np.ndarray
    scales: np.ndarray
    group_size: int

    def dequantize(self) -> np.ndarray:
        """还原成 f32，用来量化误差。"""
        grouped = self.values.reshape(-1, self.group_size).astype(np.float32)
        return (grouped * self.scales.astype(np.float32)[:, None]).ravel()


def quantize_weight(
    weight: np.ndarray, *, group_size: int = WEIGHT_GROUP_SIZE
) -> QuantizedWeight:
    """按组量化一个权重张量。

    每组独立取 scale：`scale = max(|w|) / 7`，这样组内最大值恰好映射到 int4 的
    正端。用 7 而不是 8 是因为 `[-8, 7]` 不对称，按 8 缩放会让正端溢出。

    元素数必须能被 `group_size` 整除——不能整除就抛，静默补零会让 scale 与权重
    错位，而这种错在结构校验里看不出来。
    """
    flat = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    if flat.size % group_size:
        raise ValueError(
            f"权重有 {flat.size} 个元素，不能被 group_size {group_size} 整除")

    grouped = flat.reshape(-1, group_size)
    # 全零组的 scale 取 1 而不是 0：除以 0 会产出 nan，而全零权重量化后本就该是全零。
    peak = np.abs(grouped).max(axis=1)
    scales = np.where(peak > 0, peak / INT4_MAX, 1.0).astype(np.float16)

    quantized = np.rint(grouped / scales.astype(np.float32)[:, None])
    values = np.clip(quantized, INT4_MIN, INT4_MAX).astype(np.int8)

    return QuantizedWeight(values.ravel(), scales, group_size)


def quantization_error(weight: np.ndarray, quantized: QuantizedWeight) -> float:
    """量化前后的最大绝对误差，相对于权重的幅度。

    返回相对误差便于跨张量比较——绝对误差随权重量级变化，看不出好坏。
    """
    original = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    peak = np.abs(original).max()
    if peak == 0:
        return 0.0
    return float(np.abs(original - quantized.dequantize()).max() / peak)
