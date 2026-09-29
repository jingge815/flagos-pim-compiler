"""验证 `.bin` 写盘与交叉校验。

交叉校验是架构 C 的防线：GML 里的文件名由 FlagTree（C++）写，文件本身由这里
（Python）写，两侧不一致就是悬空引用——而这不会在我们这侧报错，要到对方的
解析器才炸。所以「引用的名字集合 == 落盘的名字集合」必须是硬断言。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
from contracts.gml_quant import LUT_BYTES, WEIGHT_GROUP_SIZE
from gml_bridge.runtime_files import (
    WrittenFiles,
    verify_against_graph,
    write_activation_scale,
    write_activation_zero_point,
    write_identity_lut,
    write_scaling,
    write_weight,
)
from quant.weights import quantize_weight


@pytest.fixture
def files(tmp_path) -> WrittenFiles:
    return WrittenFiles(tmp_path)


def _quantized(shape=(256, 128)):
    rng = np.random.default_rng(0)
    return quantize_weight((rng.standard_normal(shape) * 0.02).astype(np.float32))


def test_weight_and_scale_sizes_match_the_layout(files) -> None:
    """权重字节数 == 元素数（一字节一个 int4），scale 数 == 元素数/128。"""
    quantized = _quantized((256, 128))
    write_weight(files, 8, quantized)

    weight_path = files.directory / names.weight_buffer(8)
    scale_path = files.directory / names.weight_scale(8)

    assert weight_path.stat().st_size == 256 * 128
    assert scale_path.stat().st_size == (256 * 128 // WEIGHT_GROUP_SIZE) * 2


def test_weight_is_numbered_by_its_own_node(files) -> None:
    """权重属于节点自己，不属于某条边。"""
    write_weight(files, 8, _quantized())
    assert names.weight_buffer(8) in files.names_written
    assert "input_buffer_8.bin" not in files.names_written


def test_activation_scale_is_numbered_by_consumer(files) -> None:
    """缓冲区代表边，编号取读它的那个节点。"""
    write_activation_scale(files, 8, 0.0556)
    write_activation_scale(files, 6, 0.0556, slot=1)

    assert names.scale(8) in files.names_written
    assert names.scale(6, 1) in files.names_written
    # 单输入不带槽位号，多输入带。
    assert "input_sf_8.bin" in files.names_written
    assert "input_1_sf_6.bin" in files.names_written


def test_scalar_buffers_are_two_bytes(files) -> None:
    """sf 是 fp16 标量，实物里就是 2 字节。"""
    write_activation_scale(files, 8, 0.0556)
    assert (files.directory / names.scale(8)).stat().st_size == 2


def test_zero_point_is_int32(files) -> None:
    """zp 是 int32，实物里 4 字节且恒为 0（对称量化）。"""
    write_activation_zero_point(files, 8)
    path = files.directory / names.zero_point(8)

    assert path.stat().st_size == 4
    assert np.fromfile(path, dtype=np.int32)[0] == 0


def test_identity_lut_is_written_at_full_size(files) -> None:
    write_identity_lut(files, 8)
    path = files.directory / names.activation_lut(8)

    assert path.stat().st_size == LUT_BYTES
    table = np.fromfile(path, dtype=np.float16)
    assert np.nonzero(table)[0].tolist() == [0]
    assert float(table[0]) == 1.0


def test_scaling_stores_the_operator_scale(files) -> None:
    """Scaling 存算子自身的数学缩放，attention 就是 1/√d（见文档 19 节）。"""
    write_scaling(files, 8, 128 ** -0.5)

    scaling = np.fromfile(files.directory / names.fpsu_scale(8), dtype=np.float16)
    assert scaling[0] == np.float16(128 ** -0.5)
    # 实物里它是标量，不是 per-channel 数组。
    assert len(scaling) == 1

    shift = np.fromfile(
        files.directory / names.fpsu_post_shift(8), dtype=np.int8)
    assert shift[0] == 0


def test_cross_validation_passes_when_sets_match(files) -> None:
    write_weight(files, 8, _quantized())
    write_activation_scale(files, 8, 0.0556)

    verify_against_graph(files, files.names_written)


def test_cross_validation_catches_a_dangling_reference(files) -> None:
    """GML 引用了但没写盘——这是最危险的一种，对方解析时才炸。"""
    write_weight(files, 8, _quantized())

    with pytest.raises(ValueError, match="GML 引用了但没写盘"):
        verify_against_graph(files, files.names_written | {"input_sf_9.bin"})


def test_cross_validation_catches_an_orphan_file(files) -> None:
    """写了盘但 GML 没引用——垃圾文件，说明两侧命名不一致。"""
    write_weight(files, 8, _quantized())
    write_activation_scale(files, 99, 0.1)

    with pytest.raises(ValueError, match="写了盘但 GML 没引用"):
        verify_against_graph(files, {names.weight_buffer(8),
                                     names.weight_scale(8)})


def test_total_bytes_is_tracked(files) -> None:
    write_weight(files, 8, _quantized((256, 128)))
    write_identity_lut(files, 8)

    # 权重 32768 + scale 512 + LUT 288
    assert files.total_bytes == 256 * 128 + (256 * 128 // 128) * 2 + LUT_BYTES


# ---- 零填充分类（设计 3.3）----------------------------------------------
# 判定原则：恒 0 是硬件语义的（zp、bias、post_shift）保留；表示「没算」的
# （激活、scale、RoPE 系数）改为吃标定数据。参考产物逐族实测为判据。


def test_zero_point_stays_all_zero(files) -> None:
    """对称量化的 zp 恒 0，不能被标定改动误伤（E4）。"""
    write_activation_zero_point(files, 8)
    raw = np.fromfile(files.directory / names.zero_point(8), dtype=np.int32)
    assert raw.tolist() == [0]


def test_phase_output_buffer_holds_the_quantized_result(files) -> None:
    """DQ 自命名的 output_buffer 逐字节等于 phase3（参考 3/3 个节点成立）。"""
    from gml_bridge.phase_data import dynamic_scaling
    from gml_bridge.runtime_files import write_phase_output_buffer

    phases = dynamic_scaling(
        np.arange(1, 257, dtype=np.float16), group_size=128)
    write_phase_output_buffer(files, 12, 256, phases.phase3)
    raw = np.fromfile(
        files.directory / names.phase_output_buffer_self(12), dtype=np.int8)
    assert np.array_equal(raw, phases.phase3)
    assert (raw != 0).any(), "量化结果不能是全零"


def test_phase_output_buffer_rejects_a_length_mismatch(files) -> None:
    """内容长度与这条边报的元素数不符时直接抛，不静默写一份长度不对的 bin。

    评审 r2 问题 4：以前 `content` 可缺省、缺省补零，「忘了传内容」会静默产零。
    现在 `content` 必填，`element_count` 退化成校验。
    """
    import pytest

    from gml_bridge.runtime_files import write_phase_output_buffer

    with pytest.raises(ValueError, match="口径不一致"):
        write_phase_output_buffer(files, 12, 256, np.ones(128, dtype=np.int8))


def test_dq_phase_bias_and_post_shift_stay_zero(files) -> None:
    """Kantor bias 恒 0、FPSU post_shift 恒 0（验收 A16）。"""
    from gml_bridge.phase_data import dynamic_scaling
    from gml_bridge.runtime_files import write_dq_phases

    phases = dynamic_scaling(
        np.arange(1, 257, dtype=np.float16), group_size=128)
    write_dq_phases(files, 12, phases)
    bias = np.fromfile(
        files.directory / names.phase_kantor_bias(12, 3), dtype=np.float32)
    assert bias.size and not bias.any()
    for phase in range(4):
        shift = np.fromfile(
            files.directory / names.phase_fpsu_post_shift(12, phase),
            dtype=np.uint8)
        assert shift.tolist() == [0]


def test_data_buffer_holds_calibration_activation(files) -> None:
    """边上的数据缓冲装标定激活，不再是全零（设计 3.3）。"""
    from gml_bridge.runtime_files import write_data_buffer

    write_data_buffer(files, 9, 128, dtype=np.int8)
    raw = np.fromfile(files.directory / names.data_buffer(9), dtype=np.int8)
    assert raw.size == 128
    assert (raw != 0).any(), "激活缓冲全零与「算出来就是 0」无法区分"


def test_data_buffer_is_deterministic(files, tmp_path) -> None:
    """同一条边两次写出必须逐字节相同，否则产物不可复现。"""
    from gml_bridge.runtime_files import write_data_buffer

    other = WrittenFiles(tmp_path / "again")
    other.directory.mkdir()
    write_data_buffer(files, 9, 128, dtype=np.int8)
    write_data_buffer(other, 9, 128, dtype=np.int8)
    assert ((files.directory / names.data_buffer(9)).read_bytes()
            == (other.directory / names.data_buffer(9)).read_bytes())


def test_rope_scale_families_are_one_not_zero(files) -> None:
    """RoPE 走 write_rope_buffer 的定标族参考实测是 1.0；写 0 属形式非法。

    `Kantor_*` 那几族不在这里——它们在 `export.py` 的 kantor 分支处理。
    """
    from gml_bridge.runtime_files import write_rope_buffer

    for key in ("Llama2Activation_Cos_sf",
                "Llama2Activation_Sin_Broadcast_sf",
                "Scaling_buffer_file_1_Llama2Activation_Add_Cos"):
        name = f"{key}_30.bin"
        write_rope_buffer(files, key, name, 1)
        raw = np.fromfile(files.directory / name, dtype=np.float16)
        assert (raw == 1.0).all(), key


def test_rope_cos_sin_products_are_nonzero(files) -> None:
    """cos_mul_output / sin_mul_output 参考逐元素非零，取自标定 cos/sin。"""
    from gml_bridge.runtime_files import write_rope_buffer

    for key in ("cos_mul_output", "sin_mul_output"):
        name = f"{key}_30.bin"
        write_rope_buffer(files, key, name, 128)
        raw = np.fromfile(files.directory / name, dtype=np.float16)
        assert raw.size == 128, key
        assert (raw != 0).all(), key


def test_rope_zp_and_scaling_ps_stay_zero(files) -> None:
    """RoPE 的 zp（int32）与 Scaling_PS（uint8）参考恒 0。"""
    from gml_bridge.runtime_files import write_rope_buffer

    write_rope_buffer(files, "Llama2Activation_Cos_zp", "zp_30.bin", 1)
    assert np.fromfile(files.directory / "zp_30.bin", dtype=np.int32).tolist() == [0]

    write_rope_buffer(
        files, "Scaling_PS_buffer_file_1_Llama2Activation_Add_Cos", "ps_30.bin", 1)
    assert np.fromfile(files.directory / "ps_30.bin", dtype=np.uint8).tolist() == [0]


def test_rope_scalar_families_ignore_element_count(files) -> None:
    """RoPE 的定标 / 零点 / Shift / bias 四类都是单元素标量。

    参考逐族实测：`Scaling_buffer_file_*` 2B、`Scaling_PS_buffer_file_*` 1B、
    `*_bias_buffer_file` 4B、`*_zp` 4B —— 与 head_dim 无关。传一个大的
    `element_count` 进去也不该改变宽度（评审 r6 问题1：以前按它算，
    写成 64B / 32B，是 A11 里 30 个不符族的来源）。
    """
    from gml_bridge.runtime_files import write_rope_buffer

    expected = {
        "Scaling_buffer_file_1_Llama2Activation_Add_Cos": 2,
        "Llama2Activation_Cos_sf": 2,
        "Scaling_PS_buffer_file_1_Llama2Activation_Add_Cos": 1,
        "Kantor_A_Llama2Activation_Cos_bias_buffer_file": 4,
        "Llama2Activation_Cos_zp": 4,
    }
    for key, size in expected.items():
        name = f"{key}_30.bin"
        write_rope_buffer(files, key, name, 4096)
        assert (files.directory / name).stat().st_size == size, key


def test_rope_products_still_follow_element_count(files) -> None:
    """只有 cos/sin 乘积按整张中间态算：参考 8192B = 4096 个 fp16。"""
    from gml_bridge.runtime_files import write_rope_buffer

    write_rope_buffer(files, "cos_mul_output", "cos_mul_output_30.bin", 4096)
    assert (files.directory / "cos_mul_output_30.bin").stat().st_size == 8192


def test_fpsu_triple_is_per_group_only_for_rope_dq(files) -> None:
    """FPSU 三族的宽度按**节点类型**分，不按组数。

    参考实测：`Llama2ActivationDQ`（节点 22）逐组发 32 份，而同为 32 组的
    普通 `DynamicScaling`（节点 12/24/196）只发标量。按组数判会把后三个
    也写成向量（评审 r6 问题1）。
    """
    import numpy as np
    from gml_bridge.phase_data import dynamic_scaling
    from gml_bridge.runtime_files import write_dq_phases

    phases = dynamic_scaling(
        np.arange(4096, dtype=np.float32) + 1.0, group_size=128)
    assert phases.phase1.size == 32

    write_dq_phases(files, 12, phases)
    assert (files.directory / "Scaling_buffer_phase_0_12.bin").stat().st_size == 2
    assert (files.directory / "Bias_buffer_phase_0_12.bin").stat().st_size == 4
    assert (files.directory
            / "Scaling_PS_buffer_phase_0_12.bin").stat().st_size == 1

    write_dq_phases(files, 22, phases, per_group_fpsu=True)
    assert (files.directory / "Scaling_buffer_phase_0_22.bin").stat().st_size == 64
    assert (files.directory / "Bias_buffer_phase_0_22.bin").stat().st_size == 128
    assert (files.directory
            / "Scaling_PS_buffer_phase_0_22.bin").stat().st_size == 32
