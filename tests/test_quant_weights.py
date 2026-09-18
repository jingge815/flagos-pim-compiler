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
    """int4 只有 16 级，峰值相对误差应在十几个百分点以内。

    分母 8 下的上界是 **0.125**（正端 `+absmax -> 8` clamp 成 7，吃满一个
    量化步长），不是半步的 0.0625。这条只能拦量级性错误，
    抓不到分组轴错误——那个靠下面的饱和度探针。
    """
    for shape in [(4096, 4096), (128, 256), (11008, 4096)]:
        weight = _weight(shape)
        error = quantization_error(weight, quantize_weight(weight))
        assert 0.0 < error < 0.13, f"{shape} 的相对误差 {error}"


def test_every_group_reaches_the_int4_boundary() -> None:
    """每组的 max|q| 必须触到边界 {7, 8}——这是定位分组轴错误的探针。

    对称量化下组内 absmax 一定映射到满量程，所以每组的 max|q| 必然是 8
    （峰在负侧）或 7（峰在正侧被 clamp）。**沿错误的轴分组则不满足**：
    scale 与权重错位后，多数组的 max|q| 会落在 2~6。

    这一条比峰值相对误差强得多，且不需要参考产物也不需要真实模型。
    实测参考产物 `weight_buffer_195` 的 352256 组全部满足。
    """
    for shape in [(256, 128), (512, 4096), (1024, 11008)]:
        quantized = quantize_weight(_weight(shape))
        grouped = quantized.values.reshape(-1, WEIGHT_GROUP_SIZE)
        peaks = np.abs(grouped).max(axis=1)
        touching = np.isin(peaks, (INT4_MAX, -INT4_MIN)).mean()
        assert touching > 0.95, f"{shape} 只有 {touching:.1%} 的组触到边界"


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


def test_requantization_is_stable_under_a_fixed_scale() -> None:
    """给定同一个 scale，反量化再量化必须回到原值。

    这是量化方向正确的判据：`q -> q*sf -> q` 是恒等。

    注意这里**固定 sf**，而不是让第二轮重新推导 sf。分母改成 8 之后，
    「反量化再完整量化一遍」**不再是恒等**，原因见下一个测试。
    """
    weight = _weight((1024, 128))
    quantized = quantize_weight(weight)

    scales = quantized.scales.astype(np.float32)[:, None]
    grouped = quantized.dequantize().reshape(-1, WEIGHT_GROUP_SIZE)
    again = np.clip(np.rint(grouped / scales), INT4_MIN, INT4_MAX).astype(np.int8)

    assert again.ravel().tobytes() == quantized.values.tobytes()


def test_full_round_trip_shifts_only_positive_peaked_groups() -> None:
    """完整往返（含重新推导 scale）在正峰组上不恒等，这是分母 8 的必然结果。

    机制：分母 8 让组内 absmax 映射到 8，但 int4 上界是 7。
      - 峰在**负**侧：`-absmax -> -8`，落在值域内，clamp 不触发，
        反量化后组 absmax 仍是 `8*sf`，第二轮推出同一个 sf，往返恒等。
      - 峰在**正**侧：`+absmax -> 8` 被 clamp 成 **7**，反量化后组 absmax
        变成 `7*sf`，第二轮推出 `sf2 = 7*sf/8`，量化值整体放大 8/7 —— 不恒等。

    钉住这条是为了说明它是**已知的、由不对称值域决定的**，不是量化实现的 bug。
    参考产物同样如此：`weight_buffer_195` 的 352256 组里含 `-8` 的占 58.5%、
    含 `+8` 的占 0，且 absmax 落负侧的组占 87.2%。
    """
    weight = _weight((256, 128))
    quantized = quantize_weight(weight)
    again = quantize_weight(quantized.dequantize())

    grouped = weight.reshape(-1, WEIGHT_GROUP_SIZE)
    peak_is_negative = grouped.min(axis=1) == -np.abs(grouped).max(axis=1)

    first = quantized.values.reshape(-1, WEIGHT_GROUP_SIZE)
    second = again.values.reshape(-1, WEIGHT_GROUP_SIZE)
    stable = np.array([
        np.array_equal(first[i], second[i]) for i in range(len(first))])

    # 负峰组必须全部稳定；不稳定的组必须全是正峰组。
    assert stable[peak_is_negative].all()
    assert not stable[~peak_is_negative].all(), "正峰组应当出现漂移"
