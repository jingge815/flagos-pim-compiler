"""phase 流水线的数值判据。

分两类：
- **对拍实物**：把参考产物自己的输入喂进来，比对它落盘的各相输出。
  DQ 四相能逐字节对上；Softmax 只有 phase0 能（原因见 phase_data 的模块 docstring）。
- **内部自洽**：不需要参考产物，纯数学恒等式。这一层将来对我方自己的产物同样适用。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gml_bridge.phase_data import (
    DQ_PHASE0_BIAS,
    DQ_PHASE1_SCALE,
    DQ_PHASE3_SCALE,
    DQ_PHASE3_SHIFT,
    dynamic_scaling,
    pack_fp16_in_high_half,
    softmax,
    unpack_fp16_from_high_half,
)
from genesim_bridge.paths import gml_llama2_reference_dir

# 参考产物里的 DQ 节点：(node_id, group_size)。
# 前三个是 128 分组，最后一个是 attention scores 整条一组。
DQ_NODES = [(12, 128), (193, 128), (196, 128), (17, None)]
SOFTMAX_NODES = [18, 39, 44, 49, 54]


def _read(name: str, dtype) -> np.ndarray:
    return np.fromfile(gml_llama2_reference_dir() / name, dtype=dtype)


def _raw(name: str) -> bytes:
    return (gml_llama2_reference_dir() / name).read_bytes()


# ---------------------------------------------------------------------------
# DynamicScaling：与实物逐字节对拍
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_id,group_size", DQ_NODES)
def test_dq_all_four_phases_match_the_reference(node_id: int, group_size) -> None:
    """喂实物自己的 phase0 输入，四相输出必须逐字节相同。

    这是最强的一条：四相里每一步的常量（×2、/256、取倒数、×256）若有任何一个
    写错，对应相立刻对不上。DQ 不含 LUT 逼近误差，所以能做到逐字节。
    """
    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    result = dynamic_scaling(source, group_size=group_size)

    assert result.phase0.tobytes() == _raw(f"output_buffer_phase_0_{node_id}.bin")
    assert result.phase1.tobytes() == _raw(f"output_buffer_phase_1_{node_id}.bin")
    assert result.phase2.tobytes() == _raw(f"output_buffer_phase_2_{node_id}.bin")

    # phase3 允许 ±1 —— 落在 .5 边界上的元素受 fp16 舍入方向影响。
    expected = _read(f"output_buffer_phase_3_{node_id}.bin", np.int8)
    delta = np.abs(result.phase3.astype(np.int16) - expected.astype(np.int16))
    # 舍入方向：差恒为 1，且每一处不一致都落在 .5 平局上。判据是「能被解释」
    # 而不是「一致率够高」——后者是个随参考版本漂移的经验数。
    assert delta.max() <= 1
    differing = delta > 0
    if differing.any():
        src32 = source.astype(np.float32)
        grouped = src32.reshape(-1, group_size)
        amax = np.abs(grouped).max(axis=1, keepdims=True) * np.float32(2.0)
        inv = np.where(amax == 0, np.float32(0.0),
                       np.float32(1.0) / amax).astype(np.float16).astype(
                           np.float32)
        inv = np.repeat(inv, group_size, axis=1).ravel()
        pre = src32.ravel() * inv * np.float32(256.0)
        distance = np.abs(pre - np.rint(pre))
        off_tie = differing & (distance < 0.4)
        assert not off_tie.any(), (
            f"{int(off_tie.sum())} 处不一致不在 .5 平局上，"
            f"最小距离 {float(distance[off_tie].min()):.4f}")


@pytest.mark.parametrize("node_id,group_size", DQ_NODES)
def test_dq_output_scale_is_phase1(node_id: int, group_size) -> None:
    """`output_sf` 逐字节等于 phase1 的输出。

    这条决定了下游怎么拿 scale：走动态量化的节点直接引用上游的
    `output_buffer_phase_1_*.bin`，而不是自己写一份 sf。
    """
    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    result = dynamic_scaling(source, group_size=group_size)

    assert result.output_scale.tobytes() == _raw(f"output_sf_{node_id}.bin")


@pytest.mark.parametrize("node_id,group_size", DQ_NODES)
def test_dq_kantor_scale_is_phase2(node_id: int, group_size) -> None:
    """phase3 的 `kantor_A_scale` 逐字节等于 phase2 的输出。"""
    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    result = dynamic_scaling(source, group_size=group_size)

    assert result.kantor_scale.tobytes() == _raw(
        f"kantor_A_scale_buffer_file_phase_3_{node_id}.bin")


@pytest.mark.parametrize("node_id,group_size", DQ_NODES)
def test_dq_group_count_matches_the_scale_file_size(node_id: int, group_size) -> None:
    """组数 = scale 文件字节数 / 2。这条抓的是分组宽度取错。"""
    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    result = dynamic_scaling(source, group_size=group_size)

    expected_groups = len(_raw(f"output_sf_{node_id}.bin")) // 2
    assert result.phase1.size == expected_groups


def test_dq_shift_is_minus_eight() -> None:
    """Kantor 右移量恒为 -8，即左移 8 位 = ×256，与 DQ_PHASE3_SCALE 一致。"""
    assert DQ_PHASE3_SHIFT == -8
    assert DQ_PHASE3_SCALE == 2.0 ** -DQ_PHASE3_SHIFT

    shift = _read("kantor_A_Shift_buffer_file_phase_3_12.bin", np.int8)
    assert set(shift.tolist()) == {DQ_PHASE3_SHIFT}


def test_dq_phase0_bias_is_two_to_the_minus_63() -> None:
    """`Bias_buffer_phase_0` 是 fp32 常量 2^-63，不能写 0。

    实测 36 个 DQ 节点的这个文件**字节完全相同**，所以它是常量而非运行时值。
    按 int32 解会得到一个无意义的小整数——必须按 fp32 打包。
    """
    assert DQ_PHASE0_BIAS == 2.0 ** -63

    raw = _raw("Bias_buffer_phase_0_12.bin")
    assert len(raw) == 4
    assert struct.unpack("<f", raw)[0] == pytest.approx(DQ_PHASE0_BIAS)


def test_dq_phase0_bias_underflows_in_fp16() -> None:
    """这个 bias 作用在硬件的 32 位累加器上，**不在** fp16 域。

    钉住这条是因为它反直觉：2^-63 按 fp16 存就是 0，所以不能拿它给
    fp16 域的 phase0 兜除零的底。误当成 fp16 域的保护会让全零组产出 inf。
    """
    assert np.float16(DQ_PHASE0_BIAS) == np.float16(0.0)
    assert DQ_PHASE0_BIAS < float(np.nextafter(np.float16(0), np.float16(1)))


def test_dq_phase_scaling_constants_match_the_reference() -> None:
    """p1 与 p3 的定标常量：1/256 与 256。"""
    assert struct.unpack("<e", _raw("Scaling_buffer_phase_1_12.bin"))[0] == \
        DQ_PHASE1_SCALE
    assert struct.unpack("<e", _raw("Scaling_buffer_phase_3_12.bin"))[0] == \
        DQ_PHASE3_SCALE


# ---------------------------------------------------------------------------
# DynamicScaling：内部自洽（不需要参考产物）
# ---------------------------------------------------------------------------


def test_dq_formula_is_self_consistent() -> None:
    """四相之间的数学恒等式，对任意输入都要成立。"""
    rng = np.random.default_rng(0)
    source = (rng.standard_normal(4096) * 3).astype(np.float16)
    result = dynamic_scaling(source, group_size=128)

    grouped = source.astype(np.float32).reshape(-1, 128)
    absmax = np.abs(grouped).max(axis=1)

    assert np.allclose(result.phase0.astype(np.float32), 2 * absmax, rtol=2e-3)
    assert np.allclose(
        result.phase1.astype(np.float32),
        result.phase0.astype(np.float32) / 256, rtol=2e-3)
    assert np.allclose(
        result.phase2.astype(np.float32),
        1 / result.phase0.astype(np.float32), rtol=2e-3)


def test_dq_effective_divisor_is_128() -> None:
    """等效满量程分母是 128：`output_sf == absmax/128`。

    ×2 与 /256 合起来就是这个数。写成 127 会让 int8 用不满值域。
    """
    rng = np.random.default_rng(1)
    source = (rng.standard_normal(1024) * 2).astype(np.float16)
    result = dynamic_scaling(source, group_size=128)

    absmax = np.abs(source.astype(np.float32).reshape(-1, 128)).max(axis=1)
    assert np.allclose(
        result.output_scale.astype(np.float32), absmax / 128, rtol=2e-3)


def test_dq_uses_the_full_int8_range() -> None:
    """每组的 max|q| 必须触到边界，否则说明 scale 定小了。"""
    rng = np.random.default_rng(2)
    source = (rng.standard_normal(4096)).astype(np.float16)
    result = dynamic_scaling(source, group_size=128)

    # 先升 int16 再取绝对值：int8 的 abs(-128) 会溢出。
    peaks = np.abs(result.phase3.reshape(-1, 128).astype(np.int16)).max(axis=1)
    assert np.isin(peaks, (127, 128)).mean() > 0.95


def test_dq_all_zero_group_does_not_produce_inf() -> None:
    """全零组的倒数会溢出成 inf，靠 2^-63 的 bias 兜底。"""
    result = dynamic_scaling(np.zeros(256, dtype=np.float16), group_size=128)

    assert np.isfinite(result.phase2.astype(np.float32)).all()
    assert (result.phase3 == 0).all()


def test_dq_rejects_indivisible_group_size() -> None:
    with pytest.raises(ValueError, match="不能被 group_size"):
        dynamic_scaling(np.zeros(100, dtype=np.float16), group_size=128)


def test_dq_whole_tensor_as_one_group() -> None:
    """`group_size=None` 表示整条一组，对应 attention scores。"""
    result = dynamic_scaling(np.arange(1024, dtype=np.float16), group_size=None)

    assert result.group_size == 1024
    assert result.phase0.size == 1
    assert result.phase1.nbytes == 2


# ---------------------------------------------------------------------------
# Softmax：两个归约相的编码
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_id", SOFTMAX_NODES)
def test_softmax_phase0_bytes_match_the_reference(node_id: int) -> None:
    """phase0 的 4 字节必须与实物**逐字节相同**。

    它只依赖 `max(x)`，不过 LUT，所以是 Softmax 里唯一能做字节级对拍的一相。
    这条同时钉住那个编码：fp16 位模式放高 2 字节。
    """
    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    result = softmax(source)

    assert result.phase0_bytes == _raw(f"output_buffer_phase_0_{node_id}.bin")


@pytest.mark.parametrize("node_id", SOFTMAX_NODES)
def test_reference_phase0_low_half_is_zero(node_id: int) -> None:
    """实物 phase0 的低 2 字节恒为 0——这是「fp16 放高半」的直接证据。"""
    raw = _raw(f"output_buffer_phase_0_{node_id}.bin")
    assert len(raw) == 4
    assert raw[:2] == b"\x00\x00"

    source = _read(f"input_buffer_phase_0_{node_id}.bin", np.float16)
    assert unpack_fp16_from_high_half(raw) == pytest.approx(-float(source.max()))


def test_the_two_reduction_phases_use_different_encodings() -> None:
    """phase0 是「fp16 放高半」，phase2 是真 fp32。混用会静默写错。

    判据：参考的 phase0 字节按高半解出一个 fp16，再按同一种方式装回去必须
    逐字节复现；而按真 fp32 打包同样的数得到的字节与它不同。
    """
    reference = _raw("output_buffer_phase_0_18.bin")
    # 数值从参考里取，不写死：参考换一版，两边的数都跟着换，判据不变。
    value = unpack_fp16_from_high_half(reference)
    high_half = pack_fp16_in_high_half(value)
    true_fp32 = struct.pack("<f", np.float32(value))

    assert high_half == reference, "高半编码没能复现参考字节"
    assert high_half != true_fp32
    # 低半恒为 0 是这个编码的直接后果，不是巧合。
    assert high_half[:2] == b"\x00\x00"

    # 按 fp32 误读 phase0 会得到一个「看似合理」的数，这才是危险之处：它有限、
    # 不为零、量级也像那么回事，只是完全不是那个 max。
    misread = struct.unpack("<f", high_half)[0]
    assert np.isfinite(misread) and misread != value


@pytest.mark.parametrize("node_id", SOFTMAX_NODES)
def test_reference_phase2_is_true_fp32(node_id: int) -> None:
    """实物 phase2 按 fp32 解等于 phase1 的和。"""
    total = struct.unpack("<f", _raw(f"output_buffer_phase_2_{node_id}.bin"))[0]
    phase1 = _read(f"output_buffer_phase_1_{node_id}.bin", np.float16)

    assert total == pytest.approx(float(phase1.astype(np.float32).sum()), rel=2e-3)


@pytest.mark.parametrize("node_id", SOFTMAX_NODES)
def test_reference_phase4_equals_phase1_times_phase3(node_id: int) -> None:
    """公式 `phase4 = phase1 * phase3` 在实物上逐元素成立。

    这条用实物自己的 phase1 做输入，所以绕开了 exp 表的逼近误差——
    验证的是**公式**，不是我方的 exp 实现。
    """
    phase1 = _read(f"output_buffer_phase_1_{node_id}.bin", np.float16)
    phase3 = _read(f"output_buffer_phase_3_{node_id}.bin", np.float16)[0]
    phase4 = _read(f"output_buffer_phase_4_{node_id}.bin", np.float16)

    recomputed = (phase1.astype(np.float32) * np.float32(phase3)).astype(np.float16)
    assert recomputed.tobytes() == phase4.tobytes()
    assert float(phase4.astype(np.float32).sum()) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("node_id", SOFTMAX_NODES)
def test_reference_runtime_landings_hold(node_id: int) -> None:
    """两处「运行时落点」在实物上逐字节成立，所以不能硬编码常量。

    `Bias_buffer_phase_1` 是 phase0 的输出、`Scaling_buffer_phase_4` 是 phase3 的
    输出。参考产物里前者解出 -30.75，那只是那份合成输入的 max，不是常量。
    """
    assert _raw(f"Bias_buffer_phase_1_{node_id}.bin") == \
        _raw(f"output_buffer_phase_0_{node_id}.bin")
    assert _raw(f"Scaling_buffer_phase_4_{node_id}.bin") == \
        _raw(f"output_buffer_phase_3_{node_id}.bin")


# ---------------------------------------------------------------------------
# Softmax：内部自洽
# ---------------------------------------------------------------------------


def test_softmax_is_normalised() -> None:
    """softmax 的输出必须和为 1——最基本的自洽。"""
    rng = np.random.default_rng(3)
    for size in (128, 1024):
        result = softmax((rng.standard_normal(size) * 3).astype(np.float16))
        total = float(result.phase4.astype(np.float32).sum())
        assert total == pytest.approx(1.0, abs=3e-3), f"size={size} 和为 {total}"


def test_softmax_runtime_landings_are_wired_to_the_phases() -> None:
    """我方产物内部：两处落点按构造等于对应相的输出字节。"""
    result = softmax(np.linspace(-4, 4, 256).astype(np.float16))

    assert result.phase1_bias == result.phase0_bytes
    assert result.phase4_scale == struct.pack("<e", np.float16(result.phase3))


def test_softmax_subtracts_the_max_for_stability() -> None:
    """减 max 之后 phase1 的最大值恰为 1，不会溢出。"""
    result = softmax(np.array([100.0, 101.0, 102.0], dtype=np.float16))

    assert float(result.phase1.max()) == pytest.approx(1.0, abs=1e-3)
    assert np.isfinite(result.phase1.astype(np.float32)).all()
    assert result.phase0 == pytest.approx(-102.0)


def test_softmax_phase3_is_the_reciprocal_of_phase2() -> None:
    result = softmax(np.linspace(-2, 2, 512).astype(np.float16))
    assert result.phase3 == pytest.approx(1.0 / result.phase2, rel=2e-3)


def test_softmax_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        softmax(np.array([], dtype=np.float16))


def test_our_softmax_is_closer_to_the_truth_than_the_hardware_table() -> None:
    """我方用精确 exp，比走 PWL exp 表的硬件更接近真值 softmax。

    钉住这条是为了说明「Softmax 各相与实物不逐字节相同」是**已知且更准**，
    不是实现错：实物 phase1 走 31 段 PWL 表，与精确 exp 差 0.7%（峰值相对误差），
    我方是 0.02%。

    判据用「与真值 softmax 的距离」，**不用「和是否为 1」** —— 后者由 fp16
    舍入主导（两者都是 2e-4 量级，互有胜负），区分不出逼近质量的差别。
    """
    source = _read("input_buffer_phase_0_18.bin", np.float16)
    exact = np.exp(
        source.astype(np.float64) - float(np.float16(source.max())))
    truth = exact / exact.sum()

    ours = softmax(source).phase4.astype(np.float64)
    theirs = _read("output_buffer_phase_4_18.bin", np.float16).astype(np.float64)

    our_error = np.abs(ours - truth).max()
    their_error = np.abs(theirs - truth).max()
    assert our_error < their_error / 10, \
        f"我方 {our_error:.2e} 未显著优于实物 {their_error:.2e}"
