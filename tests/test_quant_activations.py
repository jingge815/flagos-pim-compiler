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

from gml_bridge import phase_data
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


def test_scale_count_matches_the_reference_byte_sizes() -> None:
    """scale 是 per-group 数组，元素数 = numel/128。

    实测参考产物：`output_sf_12` 是 64 字节（hidden 4096 -> 32 组）、
    `output_sf_193` 是 172 字节（MLP 11008 -> 86 组）。
    原先按 per-tensor 只写 2 字节，是错的。
    """
    hidden = quantize_activation(_activation((1, 4096)))
    assert hidden.scales.dtype == np.float16
    assert hidden.scales.size == 32
    assert hidden.scales.nbytes == 64

    mlp = quantize_activation(_activation((1, 11008)))
    assert mlp.scales.size == 86
    assert mlp.scales.nbytes == 172


def test_attention_scores_are_one_group() -> None:
    """attention scores 整条当一组，落盘就是 2 字节标量。

    实测那 32 个节点的 global_pooling_group_size_phase_0 是 1024（整条一组），
    与 hidden/MLP 的 128 不同。
    """
    scores = quantize_activation(_activation((1, 1024)), group_size=None)
    assert scores.scales.size == 1
    assert scores.scales.nbytes == 2
    assert scores.scale.dtype == np.float16


def test_scale_property_rejects_multi_group() -> None:
    """多组时取单一 scale 是语义错误，要直接抛而不是静默取第一个。"""
    quantized = quantize_activation(_activation((1, 4096)))
    with pytest.raises(ValueError, match="没有单一 scale"):
        _ = quantized.scale


def test_scale_equals_absmax_over_128() -> None:
    """逐组核对 scale 的定法：absmax/128，不是 absmax/127。

    实测参考产物 `output_sf == absmax/128` 在 32/32 组成立。
    """
    activation = _activation((1, 4096))
    quantized = quantize_activation(activation)

    grouped = activation.ravel().reshape(-1, 128)
    expected = (np.abs(grouped).max(axis=1) / 128).astype(np.float16)
    assert quantized.scales.tobytes() == expected.tobytes()


def test_indivisible_size_raises() -> None:
    """不能整除就抛：静默补零会让 scale 与数据错位，结构校验查不出来。"""
    with pytest.raises(ValueError, match="不能被 group_size"):
        quantize_activation(np.zeros(100, dtype=np.float32))


def test_int8_range_is_actually_used() -> None:
    """要用满值域——峰值应当触到边界。

    分母是 128：absmax 映射到 128，峰在负侧时得 -128，峰在正侧时 clamp 成 127。
    所以判据是 `max|q|` 触到 {127, 128}，而不是「正端恰好等于 127」。
    若 scale 偏大，值域只用到几十，精度白扔。实物的全局值域是 [-128, 127]。
    """
    quantized = quantize_activation(_activation())
    assert abs(int(quantized.values.min())) in (INT8_MAX, -INT8_MIN) or \
        int(quantized.values.max()) in (INT8_MAX, -INT8_MIN)


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
    quantized = quantize_activation(activation, group_size=None)

    assert not np.isnan(float(quantized.scale))
    assert (quantized.values == 0).all()
    assert quantization_error(activation, quantized) == 0.0


def test_matches_the_phase_data_truth_source() -> None:
    """本模块必须与 `phase_data.dynamic_scaling` 逐字节一致。

    这正是它成为**第二份公式**时出的事：原先它走 `scale = absmax/128` 再取商，
    真源走 `fp16(1/(2·absmax))` 再乘 256，两次 fp16 截断的位置不同，同一输入
    实测 23/4096 处差 1。两份并存时改了真源、这里会冒充真值，对拍就白做。
    """
    activation = _activation((1, 4096))
    quantized = quantize_activation(activation)
    phases = phase_data.dynamic_scaling(
        activation.ravel().astype(np.float32), group_size=128)

    assert quantized.values.tobytes() == phases.phase3.tobytes()
    assert quantized.scales.tobytes() == phases.phase1.tobytes()


def test_tiny_values_quantize_away_without_producing_garbage() -> None:
    """幅度低到 fp16 表示不了时，这组就量化成零——但不能出 nan / inf。

    曾经这里断言 `scale > 0`，靠一个「scale 舍入成 0 就换成 fp16 最小数」的
    兜底撑着。那个兜底**没有**达成它声称的目的：换完之后 `rint(1e-8/6.1e-5)`
    仍是 0，反量化出来照样是全零，它只是让这条断言成立。
    真源 `phase_data` 对退化组的选择是把倒数置 0（见其 docstring：该组本就是
    零，而 inf/nan 会污染落盘字节），这里改断言其真实成立的几条。
    """
    activation = np.full((8, 8), 1e-8, dtype=np.float32)
    quantized = quantize_activation(activation, group_size=None)
    dequantized = quantized.dequantize()

    assert not np.isnan(dequantized).any()
    assert not np.isinf(dequantized).any()
    assert (quantized.values == 0).all()
    # 误差不超过输入自身的幅度——量化把这段信号丢了，但没有放大它。
    assert np.abs(dequantized).max() <= np.abs(activation).max()


def test_outlier_is_contained_within_its_own_group() -> None:
    """离群值只压低**同组**元素的分辨率，不波及其他组。

    这正是 per-group 量化相对 per-tensor 的意义所在：分组前一个离群值会把
    整张张量的分辨率拖低，分组后影响半径被限制在 128 个元素内。

    钉住这条是为了说明「其他组仍能用满值域」是预期行为，不是量化没生效。
    """
    activation = _activation((1, 4096))
    activation.ravel()[0] = float(np.abs(activation).max()) * 8

    quantized = quantize_activation(activation)
    grouped = quantized.values.reshape(-1, 128)

    # 第 0 组：离群值占满正端，同组其余元素被压到很小的范围。
    assert int(grouped[0].max()) == INT8_MAX
    assert abs(int(grouped[0][1:].min())) < 64

    # 其余组不受影响，各自仍触到满量程边界。
    # 注意先升到 int16 再取绝对值：int8 的 abs(-128) 会溢出成 -128。
    others = np.abs(grouped[1:].astype(np.int16)).max(axis=1)
    assert np.isin(others, (INT8_MAX, -INT8_MIN)).mean() > 0.95


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
    assert first.scales.tobytes() == second.scales.tobytes()


def test_silu_numpy_mirror_uses_the_closed_form() -> None:
    """SiLU 的 numpy 镜像走闭式，与编译内核同一条公式。

    31 段弦线表在 32 层里累积后，整网 logits 与 torch 对不上，所以镜像
    与编译内核都改成闭式。288 B 表仍由 synth_silu 合成并写入 GML 产物。
    """
    import numpy as np

    from runtime.kernels_pim import _apply_activation

    xs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    got = _apply_activation(xs, "silu")
    closed = xs / (1.0 + np.exp(-xs))
    assert np.allclose(got.astype(np.float32), closed, atol=1e-6)
