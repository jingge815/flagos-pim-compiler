"""LUT 合成的判据。

这些断言的价值在于**不需要参考产物也能跑**（除了那两条字节级/精度对拍）：
切线签名与段布局是代数性质，写错任何一项都会破坏它。
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.gml_lut import (
    LUT_SEGMENTS,
    LUT_USABLE_SEGMENTS,
    decode_reciprocal,
    pack_lut,
    synth_decaying,
    synth_identity,
    synth_reciprocal,
    synth_silu,
)
from contracts.gml_quant import LUT_BYTES, LUT_ENTRY_COUNT
from genesim_bridge.paths import gml_llama2_reference_dir


def _entries(table: bytes) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """拆出 slope 段与 intercept 段。"""
    values = struct.unpack("<" + "e" * LUT_ENTRY_COUNT, table)
    return values[0:32], values[32:64]


def _tail(table: bytes) -> tuple[float, ...]:
    return struct.unpack("<" + "e" * LUT_ENTRY_COUNT, table)[64:144]


ALL_TABLES = [synth_identity(), synth_reciprocal(), synth_silu()]


@pytest.mark.parametrize("table", ALL_TABLES)
def test_every_table_is_288_bytes(table: bytes) -> None:
    assert len(table) == LUT_BYTES == LUT_ENTRY_COUNT * 2


@pytest.mark.parametrize("table", ALL_TABLES)
def test_tail_region_is_zero(table: bytes) -> None:
    """`[64:144]` 必须全 0。

    前 40 项实测是未初始化残留而非参数（恒等表该区全 0 仍能正常工作），
    后 40 项是填充。写入残留字节等于把对方的内存垃圾当参数交付。
    """
    assert all(value == 0.0 for value in _tail(table))


@pytest.mark.parametrize("table", ALL_TABLES)
def test_last_segment_is_unused(table: bytes) -> None:
    """第 31 段恒为 0——实物 32 段里只用 31 段。"""
    slopes, intercepts = _entries(table)
    assert len(slopes) == len(intercepts) == LUT_SEGMENTS
    assert LUT_USABLE_SEGMENTS == LUT_SEGMENTS - 1
    assert slopes[LUT_USABLE_SEGMENTS] == 0.0
    assert intercepts[LUT_USABLE_SEGMENTS] == 0.0


def test_pack_rejects_wrong_segment_count() -> None:
    """段数不对就抛——静默补齐会让查表整体错位。"""
    with pytest.raises(ValueError, match="都要 32 项"):
        pack_lut([0.0] * 31, [0.0] * 32)


def test_identity_matches_the_reference_byte_for_byte() -> None:
    """恒等表必须与实物逐字节相同。

    这是唯一能做字节级对拍的一张表，因为它不含任何拟合选择：
    只有 `A[0] = 1.0`。对上了说明布局、字节序、fp16 编码三者全对。
    """
    reference = (gml_llama2_reference_dir() / "LUT_phase_1_12.bin").read_bytes()
    assert synth_identity() == reference


def test_identity_is_only_one_nonzero_entry() -> None:
    slopes, intercepts = _entries(synth_identity())
    assert slopes[0] == 1.0
    assert all(value == 0.0 for value in slopes[1:])
    assert all(value == 0.0 for value in intercepts)


def test_reciprocal_segments_are_tangents() -> None:
    """倒数表每段必须满足切线签名 `A = -B^2/4`。

    这是「切线族而非弦线」的代数判据，也是这张表最强的自检：
    A 与 B 是独立写入的两段，若任一段算错，这个关系立刻不成立。
    """
    slopes, intercepts = _entries(synth_reciprocal())
    for i in range(LUT_USABLE_SEGMENTS):
        expected = -intercepts[i] * intercepts[i] / 4
        assert abs(slopes[i] - expected) < 1e-3, f"段 {i} 不满足切线签名"


def test_reciprocal_tangent_points_cover_the_mantissa_domain() -> None:
    """反解切点应落在 fp16 归一化尾数域 `[1, 2)` 内并单调递增。"""
    _, intercepts = _entries(synth_reciprocal())
    points = [2.0 / intercepts[i] for i in range(LUT_USABLE_SEGMENTS)]
    assert 1.0 <= points[0] < 1.05
    assert 1.95 < points[-1] < 2.0
    assert all(a < b for a, b in zip(points, points[1:]))


def test_reciprocal_beats_the_reference_table() -> None:
    """密集扫过尾数域，合成表的精度必须优于参考产物。

    只在切点上比是自证（切线在切点处误差恒为 0），所以这里扫全部 fp16 值。
    实测：合成表平均 0.045%、参考产物 0.387%。
    """
    ours = synth_reciprocal()
    theirs = (gml_llama2_reference_dir() / "LUT_phase_2_12.bin").read_bytes()

    our_errors, their_errors = [], []
    for mantissa_bits in range(1024):
        if (mantissa_bits >> 5) >= LUT_USABLE_SEGMENTS:
            continue
        value = struct.unpack("<e", struct.pack("<H", (15 << 10) | mantissa_bits))[0]
        truth = 1.0 / value
        our_errors.append(abs(decode_reciprocal(value, ours) - truth) / truth)
        their_errors.append(abs(decode_reciprocal(value, theirs) - truth) / truth)

    our_mean = sum(our_errors) / len(our_errors)
    their_mean = sum(their_errors) / len(their_errors)
    assert our_mean < 0.001, f"合成表平均误差 {our_mean:.5f} 偏大"
    assert our_mean < their_mean / 4, f"合成 {our_mean:.5f} 未显著优于参考 {their_mean:.5f}"


def test_reciprocal_matches_the_hardware_reduction_output() -> None:
    """用合成表重算 DQ 的 phase2，要与硬件实测输出吻合。

    `output_buffer_phase_2 == 1 / output_buffer_phase_0`，两者都是实物落盘的
    真实硬件结果，所以这条是拿我方的表去对拍**硬件行为**，不是对拍对方的表。
    """
    table = synth_reciprocal()
    reference = gml_llama2_reference_dir()

    for node_id in (12, 193, 196):
        source = _read_fp16(reference / f"output_buffer_phase_0_{node_id}.bin")
        expected = _read_fp16(reference / f"output_buffer_phase_2_{node_id}.bin")
        errors = [
            abs(decode_reciprocal(a, table) - b) / b
            for a, b in zip(source, expected) if a > 0 and b > 0
        ]
        assert errors, f"节点 {node_id} 没有可比对的组"
        mean = sum(errors) / len(errors)
        assert mean < 0.002, f"节点 {node_id} 平均偏差 {mean:.5f}"


def _read_fp16(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack("<" + "e" * (len(raw) // 2), raw))


def test_decaying_tables_leave_segment_zero_empty() -> None:
    """exp / SiLU 一类衰减函数的第 0 段留空，承担「饱和到 0」。

    实测 exp 与 SiLU 表的 `A[0] = B[0] = 0`，而倒数表的第 0 段是实值——
    两类函数在这一点上行为不同，混用会让定域整体偏移一段。
    """
    slopes, intercepts = _entries(synth_silu())
    assert slopes[0] == 0.0 and intercepts[0] == 0.0

    reciprocal_slopes, _ = _entries(synth_reciprocal())
    assert reciprocal_slopes[0] != 0.0


def test_silu_approximates_the_true_function() -> None:
    """SiLU 表在定域内的平均绝对误差应当很小。"""
    slopes, intercepts = _entries(synth_silu())
    lo, hi, segments = -4.0, 4.0, 30
    width = (hi - lo) / segments

    errors = []
    for step in range(300):
        x = lo + (hi - lo) * step / 299
        index = 1 + min(segments - 1, int((x - lo) / width))
        approximated = slopes[index] * x + intercepts[index]
        errors.append(abs(approximated - x / (1.0 + math.exp(-x))))

    assert sum(errors) / len(errors) < 0.02


def test_decaying_is_exact_at_segment_boundaries() -> None:
    """弦线在段端点上必须精确——这是弦线（而非切线）的定义性质。

    判据用**相对**误差：`exp` 在 `[-4,4)` 上跨了 e^8 ≈ 3000 倍量程，
    端点绝对误差在大值端自然被放大。实测残余误差全部来自 fp16 表项舍入
    （同样的弦线用 f64 系数算，端点相对误差是 3e-16，即精确）。
    """
    lo, hi, segments = -4.0, 4.0, 30
    table = synth_decaying(math.exp, lo, hi, segments=segments)
    slopes, intercepts = _entries(table)
    width = (hi - lo) / segments

    for k in range(segments):
        x = lo + k * width
        truth = math.exp(x)
        approximated = slopes[1 + k] * x + intercepts[1 + k]
        assert abs(approximated - truth) / truth < 1e-2, f"段 {k} 端点偏差过大"


def test_all_four_reference_tables_are_covered() -> None:
    """实物 139 个 LUT 只有 4 种内容，我方要能合成其中 3 种。

    第 4 种（exp）的段索引规则未能反推，当前靠拷贝规避——这条测试钉住
    「3 种可合成」这个事实，等 exp 补齐后再改成 4。
    """
    reference = gml_llama2_reference_dir()
    distinct = {path.read_bytes() for path in reference.glob("LUT_phase_*.bin")}
    distinct |= {
        path.read_bytes() for path in reference.glob("activation_lut_file_*.bin")}
    assert len(distinct) == 4

    # 恒等表在其中；倒数与 SiLU 我方自合成（数值优于/不同于参考，故不比字节）。
    assert synth_identity() in distinct
