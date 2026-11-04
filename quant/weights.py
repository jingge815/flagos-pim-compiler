"""把 f32 权重量化成 GML 要的 int4 + per-group scale。

公式与布局都已用实物字节级验证过（见 docs/gml-lowering-20260914.md 26.5）：
16777216 个权重的反量化再量化 100% 一致，所以这里的量化方向也是确定的。

只做权重。激活量化的数值公式无从对照——实物没交付 `DEBUG_*_float` 浮点副本
（见 26.6），所以那部分等与硬件对拍时再定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.gml_quant import (
    INT4_MAX,
    INT4_MIN,
    INT4_SCALE_DIVISOR,
    INT8_MAX,
    INT8_MIN,
    INT8_SCALE_DIVISOR,
    WEIGHT_GROUP_SIZE,
)


@dataclass
class QuantizedWeight:
    """量化后的权重与它的 per-group scale。

    `values` 是 int8 数组，但每个元素的值域限定在 int4 的 `[-8, 7]`——实物就是
    这样存的，一字节一个值，不打包。`scales` 是 fp16，每 `group_size` 个权重一个。
    """

    values: np.ndarray
    scales: np.ndarray
    group_size: int
    # 次正规保护倍数：落盘 scale 已乘它，FPSU 再除回（GML 的 `weight_sf_multiplier`）。
    multiplier: int = 1

    def dequantize(self) -> np.ndarray:
        """还原成 f32，用来量化误差。"""
        grouped = self.values.reshape(-1, self.group_size).astype(np.float32)
        scales = self.scales.astype(np.float32) / self.multiplier
        return (grouped * scales[:, None]).ravel()


def quantize_weight(
    weight: np.ndarray, *, group_size: int = WEIGHT_GROUP_SIZE,
    guard_subnormal: bool = False,
) -> QuantizedWeight:
    """按组量化一个权重张量。

    每组独立取 scale：`scale = max(正峰 / 7, 负峰 / 8)`，再 clamp 到 `[-8, 7]`。
    两端各按自己的满量程定 scale，取较紧的一侧，所以两端都不饱和。

    这是厂商量化器的口径：厂商第 0 层 4 个投影用它重新量化，码值 99.998%、
    scale 99.99% 一致，absmax/8 只有 92%。

    `guard_subnormal` 打开时（动态量化的 Gemm），最小 scale 落进 fp16 次正规区就整张
    乘 `2^ceil(log2(tiny / 最小 scale))` 再存，倍数记在 `multiplier`（厂商 `save_scale` 规则）。

    元素数必须能被 `group_size` 整除——不能整除就抛，静默补零会让 scale 与权重
    错位，而这种错在结构校验里看不出来。
    """
    flat = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    if flat.size % group_size:
        raise ValueError(
            f"权重有 {flat.size} 个元素，不能被 group_size {group_size} 整除")

    grouped = flat.reshape(-1, group_size)
    exact = _group_scales(grouped)
    multiplier = _multiplier_for(exact) if guard_subnormal else 1
    scales = (exact * multiplier).astype(np.float16)

    # 码值按落盘 scale 还原后的有效值算，反量化与硬件看到的是同一个数。
    effective = scales.astype(np.float32) / multiplier
    quantized = np.rint(grouped / effective[:, None])
    values = np.clip(quantized, INT4_MIN, INT4_MAX).astype(np.int8)

    return QuantizedWeight(values.ravel(), scales, group_size, multiplier)


def _group_scales(grouped: np.ndarray) -> np.ndarray:
    """逐组精确 scale（f32）。全零组取 1：除以 0 会产出 nan，而全零权重量化后本就该是全零。"""
    peak = np.maximum(grouped.max(axis=1) / INT4_MAX,
                      -grouped.min(axis=1) / INT4_SCALE_DIVISOR)
    return np.where(peak > 0, peak, 1.0)


def _multiplier_for(scales: np.ndarray) -> int:
    tiny = float(np.finfo(np.float16).tiny)
    if scales.min() >= tiny:
        return 1
    return int(2 ** np.ceil(np.log2(tiny / scales.min()) - 1e-9))


def subnormal_multiplier(weight: np.ndarray, *, group_size: int = WEIGHT_GROUP_SIZE) -> int:
    """这张权重的次正规保护倍数，与 `quantize_weight(guard_subnormal=True)` 同一口径。"""
    flat = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    return _multiplier_for(_group_scales(flat.reshape(-1, group_size)))


def quantize_weight_int8(weight: np.ndarray) -> QuantizedWeight:
    """int8 per-tensor 对称量化，scale = absmax / 128（输出头的静态量化口径）。

    厂商输出头权重用这条规则重新量化，码值 100% 一致。
    """
    flat = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    peak = float(np.abs(flat).max()) if flat.size else 0.0
    scale = np.float16(peak / INT8_SCALE_DIVISOR if peak > 0 else 1.0)
    values = np.clip(np.rint(flat / np.float32(scale)), INT8_MIN, INT8_MAX).astype(np.int8)
    return QuantizedWeight(values, np.full(1, scale, dtype=np.float16), flat.size)


def to_storage_order(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """「输出维 × 输入维」的码值转成落盘的「输入维 × 输出维」（厂商 GC 按此读）。"""
    out_features, in_features = shape
    return np.ascontiguousarray(values.reshape(out_features, in_features).T).ravel()


def from_storage_order(stored: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`to_storage_order` 的逆：落盘字节还原成「输出维 × 输入维」，供核对反量化。"""
    out_features, in_features = shape
    return np.ascontiguousarray(stored.reshape(in_features, out_features).T).ravel()


def quantization_error(weight: np.ndarray, quantized: QuantizedWeight) -> float:
    """量化前后的最大绝对误差，相对于权重的幅度。

    返回相对误差便于跨张量比较——绝对误差随权重量级变化，看不出好坏。
    """
    original = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    peak = np.abs(original).max()
    if peak == 0:
        return 0.0
    return float(np.abs(original - quantized.dequantize()).max() / peak)
