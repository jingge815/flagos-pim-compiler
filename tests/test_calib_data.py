"""标定常数模块：形式与取数口径。

标定数值嵌成源码常数，导出流程不读甲方目录下任何 bin（需求 A3b）。
本测试只断言形式（元素数、dtype、非零、absmax），不断言逐元素数值——
甲方那份样本本身是随机数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gml_bridge import calib_data


def test_small_tensors_have_reference_shapes_and_dtypes() -> None:
    """五个小张量的元素数与 dtype 要与甲方 IO_info 声明的 size 一致。"""
    assert calib_data.HIDDEN_STATES.shape == (4096,)
    assert calib_data.HIDDEN_STATES.dtype == np.float32
    assert calib_data.ATTENTION_MASK.shape == (1024,)
    assert calib_data.ATTENTION_MASK.dtype == np.float32
    assert calib_data.COS_EMBEDDING.shape == (128,)
    assert calib_data.COS_EMBEDDING.dtype == np.float32
    assert calib_data.SIN_EMBEDDING.shape == (128,)
    assert calib_data.SIN_EMBEDDING.dtype == np.float32
    assert calib_data.CACHE_POSITION.shape == (96,)
    assert calib_data.CACHE_POSITION.dtype == np.int64


def test_hidden_states_matches_reference_statistics() -> None:
    """hidden_states 的统计量要与甲方那份吻合，证明常数是从它提取的。"""
    a = calib_data.HIDDEN_STATES
    assert np.isclose(np.abs(a).max(), 26.611622, rtol=1e-6)
    assert np.isclose(a.mean(), -0.1864, atol=1e-3)
    assert (a != 0).all()


def test_kv_cache_absmax_are_positive_scalars() -> None:
    """KV cache 只嵌 absmax 标量：4M 个元素全量嵌入是 112MB，不可接受。"""
    assert np.isclose(calib_data.KEY_CACHE_ABSMAX, 35.222347, rtol=1e-6)
    assert np.isclose(calib_data.VALUE_CACHE_ABSMAX, 38.633411, rtol=1e-6)
    assert calib_data.KEY_CACHE_ABSMAX > 0
    assert calib_data.VALUE_CACHE_ABSMAX > 0


def test_activation_for_hidden_size_returns_hidden_states() -> None:
    """numel == 4096 直接返回内置 hidden_states 转 fp16。"""
    got = calib_data.activation_for(4096)
    assert got.shape == (4096,)
    assert got.dtype == np.float16
    assert np.array_equal(got, calib_data.HIDDEN_STATES.astype(np.float16))


def test_activation_for_other_lengths_is_nonzero_and_bounded() -> None:
    """其余长度由 hidden_states 平铺/截断得到，保证非零、量级合理。"""
    for numel in (1, 128, 1024, 11008, 4097):
        got = calib_data.activation_for(numel)
        assert got.shape == (numel,), numel
        assert got.dtype == np.float16, numel
        assert (got != 0).all(), numel
        assert np.abs(got).max() <= 32.0, numel


def test_activation_for_is_deterministic() -> None:
    """同一 numel 两次取数必须逐元素相同，否则产物不可复现。"""
    assert np.array_equal(
        calib_data.activation_for(11008), calib_data.activation_for(11008))


def test_activation_for_rejects_non_positive() -> None:
    """numel <= 0 直接抛，不返回空数组兜底（E1）。"""
    with pytest.raises(ValueError):
        calib_data.activation_for(0)
    with pytest.raises(ValueError):
        calib_data.activation_for(-1)


def test_absmax_per_group_is_nonzero_for_every_group() -> None:
    """逐组 absmax 全非零 —— 这是 phase0 不为 0 的前提（需求 A1/A2）。"""
    for numel, group in ((4096, 128), (11008, 128), (1024, 1024)):
        a = calib_data.activation_for(numel).astype(np.float32)
        groups = np.abs(a.reshape(-1, group)).max(axis=1)
        assert (groups > 0).all(), (numel, group)


def test_kv_cache_scale_is_absmax_over_int8_max() -> None:
    """KV cache 的量化 scale 按 absmax/127 算，取值集中在这一个函数里。

    这不复现甲方的 0.0458939 / 0.00261151——那两个是甲方真实模型标定的定值，
    实测无法从标定数据反推。按 absmax 算自洽、非 1.0、不除零、不饱和。
    """
    assert np.isclose(
        calib_data.kv_cache_scale(is_key=True),
        calib_data.KEY_CACHE_ABSMAX / 127)
    assert np.isclose(
        calib_data.kv_cache_scale(is_key=False),
        calib_data.VALUE_CACHE_ABSMAX / 127)


def test_kv_cache_scale_is_neither_zero_nor_one() -> None:
    """sf=0 会让下游反量化除零，sf=1.0 是本轮要消除的写死值。"""
    for is_key in (True, False):
        scale = calib_data.kv_cache_scale(is_key=is_key)
        assert scale > 0
        assert not np.isclose(scale, 1.0)


def test_kv_cache_scale_survives_fp16_roundtrip() -> None:
    """scale 要写进 fp16 的 bin，转一圈后不能变成 0 或 1.0。"""
    for is_key in (True, False):
        raw = np.float16(calib_data.kv_cache_scale(is_key=is_key))
        assert raw > 0
        assert not np.isclose(float(raw), 1.0)


def test_calibration_for_role_returns_that_role_own_constant() -> None:
    """四个图入口各取自己那份常数，长度相等时逐元素相同（评审 r7 问题 1）。

    以前一律走 `activation_for`，于是 cos 与 sin 落成同一份、kv_position 这种
    位置索引变成噪声。这条断言把「每个域的含义」钉在取数口径上。
    """
    expected = {
        "cos": calib_data.COS_EMBEDDING,
        "sin": calib_data.SIN_EMBEDDING,
        "mask": calib_data.ATTENTION_MASK,
        "kv_position": calib_data.CACHE_POSITION,
    }
    for role, values in expected.items():
        got = calib_data.calibration_for_role(role, values.size)
        assert np.array_equal(got, values), role


def test_cos_and_sin_roles_are_not_the_same_content() -> None:
    """cos 与 sin 是两个不同的域，内容必须不同。"""
    cos = calib_data.calibration_for_role("cos", 128)
    sin = calib_data.calibration_for_role("sin", 128)
    assert not np.array_equal(cos, sin)


def test_kv_position_role_keeps_the_index_values() -> None:
    """位置索引按值落盘：甲方那份是 96 个 0，不做满量程量化。"""
    got = calib_data.calibration_for_role("kv_position", 96)
    assert got.dtype == np.int64
    assert not got.any()


def test_calibration_for_role_tiles_other_lengths() -> None:
    """长度不等时平铺/截断，供不重写槽位的小图夹具用。"""
    got = calib_data.calibration_for_role("cos", 300)
    assert got.shape == (300,)
    assert np.array_equal(got[:128], calib_data.COS_EMBEDDING)


def test_calibration_for_role_rejects_non_positive() -> None:
    """numel <= 0 直接抛，与 `activation_for` 同口径。"""
    with pytest.raises(ValueError):
        calib_data.calibration_for_role("cos", 0)


def test_calibration_for_role_rejects_unknown_role() -> None:
    """未登记的角色直接 KeyError，不静默回落到 hidden state 平铺。"""
    with pytest.raises(KeyError):
        calib_data.calibration_for_role("no_such_role", 128)
