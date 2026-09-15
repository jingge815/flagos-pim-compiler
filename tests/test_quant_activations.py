"""验证激活量化。

与权重不同，这里**没有字节级对照**——实物没交付量化前的浮点副本。所以依据是实物
激活缓冲区的可观测特征：值域用满 int8、饱和率中位 0.0034。这些测试确认我方产出
落在同一特征区间，并且拦住几类会让数值悄悄错掉的实现失误。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.activations import (
    INT8_MAX,
    INT8_MIN,
    quantization_error,
    quantize_activation,
)


def _activation(shape=(128, 4096), scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def test_values_stay_in_the_int8_range() -> None:
    quantized = quantize_activation(_activation())
    assert quantized.values.dtype == np.int8
    assert quantized.values.min() >= INT8_MIN
    assert quantized.values.max() <= INT8_MAX


def test_scale_is_a_scalar() -> None:
    """实物的 `input_sf` 是 2 字节 fp16 一个值，不是数组。"""
    quantized = quantize_activation(_activation())
    assert quantized.scale.dtype == np.float16
    assert np.ndim(quantized.scale) == 0
    assert quantized.scale.tobytes().__len__() == 2


def test_int8_range_is_actually_used() -> None:
    """要用满值域——正端应当触到 127。

    若用 128 而不是 127 定 scale，正端会停在 126 附近并有裁剪；若 scale 偏大，
    值域只用到几十，精度白扔。实物的全局值域是 [-128, 127]。
    """
    quantized = quantize_activation(_activation())
    assert quantized.values.max() == INT8_MAX


def test_saturation_stays_low_for_normal_data() -> None:
    """正常分布的饱和率应当很低。

    实物的中位数是 0.0034。饱和率高说明 scale 偏小、大量值被裁剪。
    """
    quantized = quantize_activation(_activation())
    assert quantized.saturation_ratio() < 0.01


def test_error_matches_eight_bit_precision() -> None:
    """int8 有 256 级，相对误差应在千分之几。

    这条能拦住量级性的错误：scale 差一个 2 的幂，误差会跳一个数量级。
    """
    for shape in [(128, 4096), (1, 4096), (32, 128)]:
        activation = _activation(shape)
        error = quantization_error(activation, quantize_activation(activation))
        assert 0.0 < error < 0.02, f"{shape} 的相对误差 {error}"


def test_all_zero_activation_does_not_produce_nan() -> None:
    """全零张量的 scale 取 1 而非 0，否则除法产出 nan。"""
    activation = np.zeros((16, 64), dtype=np.float32)
    quantized = quantize_activation(activation)

    assert not np.isnan(float(quantized.scale))
    assert (quantized.values == 0).all()
    assert quantization_error(activation, quantized) == 0.0


def test_tiny_values_do_not_collapse_the_scale() -> None:
    """幅度极小时 fp16 的 scale 可能舍入成 0，那样反量化会全零。

    实现里兜了这一手，用 fp16 的最小正规数代替。
    """
    activation = np.full((8, 8), 1e-8, dtype=np.float32)
    quantized = quantize_activation(activation)

    assert float(quantized.scale) > 0.0
    assert not np.isnan(quantized.dequantize()).any()


def test_outlier_dominates_the_scale() -> None:
    """一个离群值会压低其余元素的分辨率——这是 per-tensor 量化的固有代价。

    钉住这个行为是为了说明它是已知的而非疏漏：若将来改成带裁剪的 scale 策略，
    这条会失败，提醒同步文档。
    """
    activation = _activation()
    activation.ravel()[0] = float(np.abs(activation).max()) * 8

    quantized = quantize_activation(activation)
    # 离群值占满正端，其余元素被压到很小的范围。
    assert quantized.values.max() == INT8_MAX
    assert abs(int(quantized.values.min())) < 64


def test_dequantize_recovers_the_magnitude() -> None:
    activation = _activation((64, 256))
    recovered = quantize_activation(activation).dequantize()

    assert recovered.shape == activation.ravel().shape
    peak_ratio = np.abs(recovered).max() / np.abs(activation).max()
    assert abs(peak_ratio - 1) < 0.02


def test_quantize_is_deterministic() -> None:
    activation = _activation((32, 128))
    first = quantize_activation(activation)
    second = quantize_activation(activation)

    assert first.values.tobytes() == second.values.tobytes()
    assert first.scale == second.scale
