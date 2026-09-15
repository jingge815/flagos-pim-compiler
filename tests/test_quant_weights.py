"""验证 W4 权重量化。

量化方向的正确性已由实物字节级往返确立（文档 26.5），所以这里盯的是我们这侧的
实现：值域、分组、误差量级、以及不能整除时必须抛而不是静默补零。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.gml_quant import INT4_MAX, INT4_MIN, WEIGHT_GROUP_SIZE
from quant.weights import quantization_error, quantize_weight


def _weight(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def test_values_stay_in_the_int4_range() -> None:
    """一字节一个 int4，值域严格 [-8, 7]。"""
    quantized = quantize_weight(_weight((4096, 4096)))
    assert quantized.values.dtype == np.int8
    assert quantized.values.min() >= INT4_MIN
    assert quantized.values.max() <= INT4_MAX


def test_one_scale_per_group() -> None:
    weight = _weight((4096, 4096))
    quantized = quantize_weight(weight)

    assert quantized.values.size == weight.size
    assert quantized.scales.size == weight.size // WEIGHT_GROUP_SIZE
    assert quantized.scales.dtype == np.float16


def test_error_matches_four_bit_precision() -> None:
    """int4 只有 16 级，相对误差应在几个百分点量级。

    这个断言的作用是拦住量级性的错误——比如 scale 用 8 而不是 7 会让正端饱和，
    误差跳一个数量级。
    """
    for shape in [(4096, 4096), (128, 256), (11008, 4096)]:
        weight = _weight(shape)
        error = quantization_error(weight, quantize_weight(weight))
        assert 0.0 < error < 0.15, f"{shape} 的相对误差 {error}"


def test_indivisible_size_raises() -> None:
    """不能整除就抛：静默补零会让 scale 与权重错位，结构校验查不出来。"""
    with pytest.raises(ValueError, match="不能被 group_size"):
        quantize_weight(_weight((100,)))


def test_all_zero_group_does_not_produce_nan() -> None:
    """全零组的 scale 取 1 而不是 0，否则除法产出 nan。"""
    weight = np.zeros(WEIGHT_GROUP_SIZE * 4, dtype=np.float32)
    quantized = quantize_weight(weight)

    assert not np.isnan(quantized.scales).any()
    assert (quantized.values == 0).all()
    assert quantization_error(weight, quantized) == 0.0


def test_dequantize_recovers_the_magnitude() -> None:
    """反量化后的幅度要与原权重相当，不能整体偏移或缩放。"""
    weight = _weight((512, 512))
    quantized = quantize_weight(weight)
    recovered = quantized.dequantize()

    assert recovered.shape == weight.ravel().shape
    # 峰值不该差出 10% 以上——差多了说明 scale 的算法有偏。
    assert abs(np.abs(recovered).max() / np.abs(weight).max() - 1) < 0.1


def test_quantize_is_deterministic() -> None:
    """同一输入两次量化结果必须一致，否则缓存与对拍都失去意义。"""
    weight = _weight((256, 256))
    first = quantize_weight(weight)
    second = quantize_weight(weight)

    assert first.values.tobytes() == second.values.tobytes()
    assert first.scales.tobytes() == second.scales.tobytes()


def test_requantization_is_stable() -> None:
    """反量化再量化应回到原值——这是实物往返 100% 一致的那条性质。"""
    weight = _weight((1024, 128))
    quantized = quantize_weight(weight)

    again = quantize_weight(quantized.dequantize())
    assert again.values.tobytes() == quantized.values.tobytes()
