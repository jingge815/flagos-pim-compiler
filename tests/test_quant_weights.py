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

    两端都不饱和，误差上界是半个量化步长（约 0.0625 + fp16 舍入）。这条只能拦
    量级性错误，抓不到分组轴错误——那个靠下面的饱和度探针。
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

    第二轮重新推导 sf 的完整往返见 `test_quantize_is_idempotent_on_its_own_output`。
    """
    weight = _weight((1024, 128))
    quantized = quantize_weight(weight)

    scales = quantized.scales.astype(np.float32)[:, None]
    grouped = quantized.dequantize().reshape(-1, WEIGHT_GROUP_SIZE)
    again = np.clip(np.rint(grouped / scales), INT4_MIN, INT4_MAX).astype(np.int8)

    assert again.ravel().tobytes() == quantized.values.tobytes()


def test_scale_takes_the_tighter_side_of_the_asymmetric_range() -> None:
    """scale = max(正峰/7, 负峰/8)：两端都不饱和，正峰组映射到 7、负峰组映射到 -8。

    厂商 GML 第 0 层 4 个投影用这条规则重新量化，码值 99.998% 一致、scale 99.99% 一致；
    absmax/8 只有 92%。
    """
    weight = np.zeros((2, WEIGHT_GROUP_SIZE), dtype=np.float32)
    weight[0, :2] = (0.7, -0.4)      # 正峰组：0.7/7 = 0.1 > 0.4/8
    weight[1, :2] = (0.2, -0.8)      # 负峰组：0.8/8 = 0.1 > 0.2/7
    quantized = quantize_weight(weight)

    np.testing.assert_allclose(quantized.scales.astype(np.float32), [0.1, 0.1], rtol=1e-3)
    grouped = quantized.values.reshape(2, -1)
    assert grouped[0].max() == INT4_MAX and grouped[1].min() == INT4_MIN


def test_quantize_is_idempotent_on_its_own_output() -> None:
    """反量化后再完整量化一遍回到原码值：两端都按各自满量程定 scale，不再漂移。"""
    weight = _weight((256, 128))
    quantized = quantize_weight(weight)
    again = quantize_weight(quantized.dequantize())

    assert again.values.tobytes() == quantized.values.tobytes()


def test_int8_per_tensor_uses_absmax_over_128() -> None:
    """输出头权重：int8 per-tensor，scale = absmax/128（厂商 head 码值 100% 一致）。"""
    from quant.weights import quantize_weight_int8

    weight = np.array([0.5, -1.28, 0.64], dtype=np.float32)
    quantized = quantize_weight_int8(weight)

    assert quantized.scales.shape == (1,)
    assert float(quantized.scales[0]) == pytest.approx(0.01, rel=1e-3)
    assert quantized.values.tolist() == [50, -128, 64]


def test_subnormal_scales_get_a_power_of_two_multiplier() -> None:
    """最小 scale 低于 fp16 最小正规数时整张乘 2^k，落盘 scale 全部回到正规区。

    厂商规则（relay2gml `save_scale`）：k = ceil(log2(tiny / min_scale))。
    厂商第 0 层 q_proj 最小 scale 约 4.1e-5，得到倍数 2，与厂商 GML 的字段一致。
    """
    tiny = float(np.finfo(np.float16).tiny)
    weight = np.zeros((2, WEIGHT_GROUP_SIZE), dtype=np.float32)
    weight[0, 0] = -8 * tiny * 0.3       # 该组 scale = 0.3 * tiny，需要乘 4
    weight[1, 0] = -8 * 0.01
    quantized = quantize_weight(weight, guard_subnormal=True)

    assert quantized.multiplier == 4
    assert (quantized.scales.astype(np.float32) >= tiny).all()
    np.testing.assert_allclose(quantized.dequantize(), weight.ravel(), rtol=1e-3, atol=1e-9)


def test_normal_scales_keep_multiplier_one() -> None:
    quantized = quantize_weight(_weight((64, 128)), guard_subnormal=True)
    assert quantized.multiplier == 1


def test_subnormal_multiplier_matches_quantizer() -> None:
    """序列化侧只拿倍数、不拿码值，必须与写盘那次量化得出同一个数。"""
    from quant.weights import subnormal_multiplier

    tiny = float(np.finfo(np.float16).tiny)
    for scale in (0.3 * tiny, 0.9 * tiny, 1.0):
        weight = np.zeros((1, WEIGHT_GROUP_SIZE), dtype=np.float32)
        weight[0, 0] = -8 * scale
        assert subnormal_multiplier(weight) == quantize_weight(
            weight, guard_subnormal=True).multiplier
