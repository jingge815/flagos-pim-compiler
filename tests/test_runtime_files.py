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
