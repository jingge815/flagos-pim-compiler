"""新增 bin 族写出的判据：字节数、字节内容、与实物的差异范围。

最强的一条是 `test_every_written_file_matches_the_reference_size` —— 参考产物是
合成数据，内容不可比，但**字节数必须逐文件相等**。字节数对不上意味着元素数或
dtype 判断错了，是最常见也最致命的一类错。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gml_bridge.phase_data import dynamic_scaling, softmax
from gml_bridge.runtime_files import (
    WrittenFiles,
    write_activation_scale,
    write_dq_phases,
    write_fused_silu_lut,
    write_rms_norm_epsilon,
    write_scaling,
    write_softmax_phases,
    write_zero_point,
)
from genesim_bridge.paths import gml_llama2_reference_dir

# 参考产物里的 DQ / Softmax 节点，用来做逐文件对拍。
DQ_NODES = [(12, 128), (193, 128), (196, 128)]
SOFTMAX_NODES = [18, 39]

# 这些文件与实物**有意不同**，原因各自明确，不参与字节级比对。
#
# - LUT：我方自行合成，精度优于实物（倒数表误差 0.045% vs 0.387%）
# - Softmax 各相：硬件走 31 段 PWL exp 表，与精确 exp 差 0.7%
# - DQ phase3：fp16 舍入使少量元素差 ±1（实测 99.4% 逐字节相同）
_EXPECTED_TO_DIFFER = ("LUT_phase_2", "LUT_phase_3", "activation_lut_file")
_SOFTMAX_LUT_PHASES = ("phase_1_", "phase_2_", "phase_3_", "phase_4_")


@pytest.fixture
def written(tmp_path: Path) -> WrittenFiles:
    """把所有新增族各写一份，节点号取自参考产物以便对拍。"""
    reference = gml_llama2_reference_dir()
    files = WrittenFiles(tmp_path)

    for node_id, group_size in DQ_NODES:
        source = np.fromfile(
            reference / f"input_buffer_phase_0_{node_id}.bin", dtype=np.float16)
        write_dq_phases(files, node_id, dynamic_scaling(source, group_size=group_size))

    for node_id in SOFTMAX_NODES:
        source = np.fromfile(
            reference / f"input_buffer_phase_0_{node_id}.bin", dtype=np.float16)
        write_softmax_phases(files, node_id, softmax(source))

    write_scaling(files, 195, 1.0)
    write_rms_norm_epsilon(files, 25, 1e-5)
    write_zero_point(files, "output_zp_12.bin")
    write_fused_silu_lut(files, 195)
    return files


def test_every_written_file_exists_in_the_reference(written: WrittenFiles) -> None:
    """写出的每个名字都要是实物里真实存在的名字。

    这条抓命名错误——大小写、下划线、`buffer_file` 那一段少写，
    都会产出对方读不到的悬空引用。
    """
    reference = gml_llama2_reference_dir()
    missing = [
        name for name in sorted(written.names_written)
        if not (reference / name).exists()
    ]
    assert not missing, f"这些名字在实物里不存在: {missing}"


def test_every_written_file_matches_the_reference_size(written: WrittenFiles) -> None:
    """**字节数必须逐文件相等**——这是内容不可比时最强的判据。

    参考产物是合成数据（RMSNorm 权重全 127、cos/sin 超出 [-1,1]），
    所以数值不能比；但字节数编码了「元素数 × dtype 宽度」，必须对上。
    """
    reference = gml_llama2_reference_dir()
    wrong = []
    for name in sorted(written.names_written):
        ours = (written.directory / name).stat().st_size
        theirs = (reference / name).stat().st_size
        if ours != theirs:
            wrong.append((name, ours, theirs))
    assert not wrong, f"字节数不符: {wrong}"


def test_most_files_are_byte_identical(written: WrittenFiles) -> None:
    """除已知的 LUT 与 Softmax 逼近差异外，其余应逐字节相同。

    这条是「常量类 bin 逐值相等」那一层的落地：定标常量、bias、shift、zp
    这些不含逼近的文件必须一字不差。
    """
    reference = gml_llama2_reference_dir()
    unexpected = []
    for name in sorted(written.names_written):
        if any(tag in name for tag in _EXPECTED_TO_DIFFER):
            continue
        # Softmax 的各相受 exp 表影响；DQ phase3 有 ±1 舍入，各自单独测。
        if any(f"{tag}{node}" in name
               for tag in _SOFTMAX_LUT_PHASES for node in SOFTMAX_NODES):
            continue
        if name.startswith("output_buffer_phase_3_"):
            continue
        if (written.directory / name).read_bytes() != (reference / name).read_bytes():
            unexpected.append(name)
    assert not unexpected, f"这些文件本应逐字节相同: {unexpected}"


@pytest.mark.parametrize("node_id,group_size", DQ_NODES)
def test_dq_phase3_differs_by_at_most_one(node_id: int, group_size: int) -> None:
    """DQ 的量化结果允许 ±1 —— 落在 .5 边界的元素受 fp16 舍入方向影响。

    上界钉在 ±1：若出现 ±2 以上，说明 scale 而非舍入出了问题。
    """
    reference = gml_llama2_reference_dir()
    source = np.fromfile(
        reference / f"input_buffer_phase_0_{node_id}.bin", dtype=np.float16)
    phases = dynamic_scaling(source, group_size=group_size)

    theirs = np.fromfile(
        reference / f"output_buffer_phase_3_{node_id}.bin", dtype=np.int8)
    delta = np.abs(phases.phase3.astype(np.int16) - theirs.astype(np.int16))

    assert delta.max() <= 1
    assert (delta == 0).mean() > 0.99


def test_fpsu_triple_widths(tmp_path: Path) -> None:
    """FPSU 三族的宽度各不相同：fp16 / u8 / fp32。

    对应硬件规范里 FPSU 的三个操作数。**bias 是 fp32 不是 int32** ——
    按 int32 写会让 DQ 的 2^-63 变成一个无意义的小整数。
    """
    files = WrittenFiles(tmp_path)
    write_scaling(files, 195, 0.088388, post_shift=0, bias=0.0)

    assert (tmp_path / "Scaling_buffer_file_195.bin").stat().st_size == 2
    assert (tmp_path / "Scaling_PS_buffer_file_195.bin").stat().st_size == 1
    assert (tmp_path / "Bias_buffer_file_195.bin").stat().st_size == 4


def test_kv_cache_dma_uses_post_shift_fourteen(tmp_path: Path) -> None:
    """KV_Cache_DMA 是全图唯一 post_shift 非零的算子（=14），配 scale=2.0。"""
    files = WrittenFiles(tmp_path)
    write_scaling(files, 28, 2.0, post_shift=14)

    reference = gml_llama2_reference_dir()
    assert (tmp_path / "Scaling_buffer_file_28.bin").read_bytes() == \
        (reference / "Scaling_buffer_file_28.bin").read_bytes()
    assert (tmp_path / "Scaling_PS_buffer_file_28.bin").read_bytes() == \
        (reference / "Scaling_PS_buffer_file_28.bin").read_bytes()


def test_attention_scale_is_one_over_sqrt_head_dim(tmp_path: Path) -> None:
    """matmul1 的定标系数是 1/√head_dim —— attention scale 折在这里。

    漏掉它等于丢掉 attention 的缩放，数值全错，而结构校验查不出来。
    """
    files = WrittenFiles(tmp_path)
    write_scaling(files, 20, 1.0 / np.sqrt(128))

    reference = gml_llama2_reference_dir()
    ours = np.fromfile(tmp_path / "Scaling_buffer_file_20.bin", dtype=np.float16)
    theirs = np.fromfile(
        reference / "Scaling_buffer_file_20.bin", dtype=np.float16)
    assert ours.tobytes() == theirs.tobytes()
    assert float(ours[0]) == pytest.approx(0.088388, abs=1e-5)


def test_zero_point_is_four_byte_int32_zero(tmp_path: Path) -> None:
    """全部 381 个 zp 文件都是 4 字节 int32 的 0，但文件必须存在。"""
    files = WrittenFiles(tmp_path)
    write_zero_point(files, "output_zp_12.bin")

    raw = (tmp_path / "output_zp_12.bin").read_bytes()
    assert len(raw) == 4
    assert struct.unpack("<i", raw)[0] == 0


def test_rms_norm_epsilon_is_fp32(tmp_path: Path) -> None:
    """eps 是 fp32，取值就是 config.json 的 rms_norm_eps。"""
    files = WrittenFiles(tmp_path)
    write_rms_norm_epsilon(files, 25, 1e-5)

    reference = gml_llama2_reference_dir()
    assert (tmp_path / "RMSNorm_Add_Const_25.bin").read_bytes() == \
        (reference / "RMSNorm_Add_Const_25.bin").read_bytes()


def test_activation_scale_accepts_per_group_arrays(tmp_path: Path) -> None:
    """`*_sf` 可以是 per-group 数组：hidden 32 个、MLP 86 个。

    原实现只写标量（2 字节），而实物是 64 / 172 字节。
    """
    files = WrittenFiles(tmp_path)
    write_activation_scale(files, 12, np.ones(32, dtype=np.float16))
    write_activation_scale(files, 193, np.ones(86, dtype=np.float16))

    assert (tmp_path / "input_sf_12.bin").stat().st_size == 64
    assert (tmp_path / "input_sf_193.bin").stat().st_size == 172


def test_activation_scale_supports_fp32_for_rms_norm(tmp_path: Path) -> None:
    """RMSNorm 系列的 sf 是 fp32（实测 2 处），其余是 fp16。"""
    files = WrittenFiles(tmp_path)
    write_activation_scale(files, 25, 1.0 / 127, dtype=np.float32)

    assert (tmp_path / "input_sf_25.bin").stat().st_size == 4


def test_dq_writes_the_full_family_set(written: WrittenFiles) -> None:
    """一个 DQ 节点该写的族要齐：4 相缓冲 + 定标三族 + 2 张 LUT + Kantor 三族。"""
    names = {name for name in written.names_written if name.endswith("_12.bin")}

    for phase in range(4):
        assert f"output_buffer_phase_{phase}_12.bin" in names
        assert f"Scaling_buffer_phase_{phase}_12.bin" in names
        assert f"Bias_buffer_phase_{phase}_12.bin" in names
        assert f"Scaling_PS_buffer_phase_{phase}_12.bin" in names

    assert "LUT_phase_1_12.bin" in names
    assert "LUT_phase_2_12.bin" in names
    assert "kantor_A_scale_buffer_file_phase_3_12.bin" in names
    assert "kantor_A_Shift_buffer_file_phase_3_12.bin" in names
    assert "kantor_A_bias_buffer_file_phase_3_12.bin" in names


def test_softmax_reduction_phases_have_distinct_encodings(
    written: WrittenFiles,
) -> None:
    """落盘后再回读，两个归约相的编码差异必须还在。

    phase0 按 fp32 误读会得到 -30.75 这种「看似合理」的值，所以这条钉住
    低 2 字节为 0 这个特征。
    """
    phase0 = (written.directory / "output_buffer_phase_0_18.bin").read_bytes()
    phase2 = (written.directory / "output_buffer_phase_2_18.bin").read_bytes()

    assert len(phase0) == len(phase2) == 4
    assert phase0[:2] == b"\x00\x00"
    # phase0 与实物逐字节相同（它只依赖 max(x)，不过 LUT）。
    reference = gml_llama2_reference_dir()
    assert phase0 == (reference / "output_buffer_phase_0_18.bin").read_bytes()


def test_softmax_runtime_landings_are_written(written: WrittenFiles) -> None:
    """两处运行时落点落盘后仍等于对应相的输出。"""
    directory = written.directory
    assert (directory / "Bias_buffer_phase_1_18.bin").read_bytes() == \
        (directory / "output_buffer_phase_0_18.bin").read_bytes()
    assert (directory / "Scaling_buffer_phase_4_18.bin").read_bytes() == \
        (directory / "output_buffer_phase_3_18.bin").read_bytes()


def test_dq_phase_constants_are_written(written: WrittenFiles) -> None:
    """p1 = 1/256、p3 = 256、p0 bias = 2^-63、shift = -8。"""
    directory = written.directory

    scale1 = struct.unpack(
        "<e", (directory / "Scaling_buffer_phase_1_12.bin").read_bytes())[0]
    scale3 = struct.unpack(
        "<e", (directory / "Scaling_buffer_phase_3_12.bin").read_bytes())[0]
    bias0 = struct.unpack(
        "<f", (directory / "Bias_buffer_phase_0_12.bin").read_bytes())[0]
    shift = np.fromfile(
        directory / "kantor_A_Shift_buffer_file_phase_3_12.bin", dtype=np.int8)

    assert scale1 == 1 / 256
    assert scale3 == 256
    assert bias0 == pytest.approx(2.0 ** -63)
    assert set(shift.tolist()) == {-8}


def test_total_bytes_is_tracked(written: WrittenFiles) -> None:
    """`total_bytes` 要等于实际落盘字节数之和。"""
    actual = sum(
        (written.directory / name).stat().st_size
        for name in written.names_written)
    assert written.total_bytes == actual
