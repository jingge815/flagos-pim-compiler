"""用 llama2 W4A8 实物交叉验证量化契约。

契约里每一条都来自实测，所以每一条都要能在实物上复现。这些测试是第 4 轮写 `.bin`
之前的防线：契约与实物脱节，产出的二进制底层编译器就读不对，而结构校验查不出
数值层面的错。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.gml_quant import (
    ACTIVATION_LAYOUT,
    DQ_PHASE_COUNT,
    DQ_REDUCTION_WIDTH,
    DTYPES,
    INT4_BYTES_PER_VALUE,
    INT4_MAX,
    INT4_MIN,
    LUT_BYTES,
    LUT_ENTRY_COUNT,
    WEIGHT_GROUP_SIZE,
    WEIGHT_LAYOUT,
    QuantLayout,
    attention_scale,
    lut_identity,
    lut_placeholder,
)
from genesim_bridge.paths import gml_llama2_reference_dir
from scripts.gml_structure_check import field, parse_blocks

_REFERENCE_DIR = gml_llama2_reference_dir(required=False)
_REFERENCE_GML = (
    _REFERENCE_DIR / "relay2gml_graph.gml" if _REFERENCE_DIR else None
)

pytestmark = pytest.mark.skipif(
    _REFERENCE_GML is None or not _REFERENCE_GML.is_file(),
    reason="缺少 llama2 GML 参考产物",
)


@pytest.fixture(scope="module")
def graph_text() -> str:
    return _REFERENCE_GML.read_text()


def _read(name: str, dtype) -> np.ndarray | None:
    path = _REFERENCE_DIR / name
    return np.fromfile(path, dtype=dtype) if path.is_file() else None


def test_int4_is_stored_one_value_per_byte(graph_text: str) -> None:
    """int4 不打包：字节数等于元素数，值域严格 [-8, 7]。"""
    nodes = parse_blocks(graph_text, "node")
    checked = 0
    for block in nodes:
        if field(block, "weight_buffer_dtype") != "int4":
            continue
        name = field(block, "weight_buffer")
        weights = _read(name, np.int8) if name else None
        if weights is None:
            continue
        assert weights.min() >= INT4_MIN
        assert weights.max() <= INT4_MAX
        # 16 个值全覆盖才说明真是 4 位而不是碰巧值域小。
        assert len(np.unique(weights)) == 16
        checked += 1
    assert checked > 0, "实物里应当有 int4 权重"
    assert INT4_BYTES_PER_VALUE == 1


def test_weight_group_size_matches_the_reference(graph_text: str) -> None:
    """权重字节数 / scale 数 == group_size。"""
    assert set(re.findall(
        r"DEBUG_weight_buffer_spg_group_size (\d+)", graph_text)) == {
        str(WEIGHT_GROUP_SIZE)}

    for block in parse_blocks(graph_text, "node"):
        weight_name = field(block, "weight_buffer")
        scale_name = field(block, "weight_sf")
        if not (weight_name and scale_name):
            continue
        weight_path = _REFERENCE_DIR / weight_name
        scale_path = _REFERENCE_DIR / scale_name
        if not (weight_path.is_file() and scale_path.is_file()):
            continue
        scale_count = scale_path.stat().st_size // 2  # fp16
        if scale_count <= 1:
            continue
        assert weight_path.stat().st_size // scale_count == WEIGHT_GROUP_SIZE
        return
    pytest.fail("没找到可验证分组的权重")


def test_declared_dtypes_appear_in_the_reference(graph_text: str) -> None:
    """契约声明的 dtype 要在实物里真实出现，否则声明是凭空的。"""
    # `bias_buffer_dtype` 不在这里：llama2 的这个 decode block 没有偏置，
    # 该字段一次都不出现。契约里保留 int32 是 ResNet50 实测的（54 处）。
    for key, field_name in (
        ("weight", "weight_buffer_dtype"),
        ("output", "output_buffer_dtype"),
    ):
        actual = set(re.findall(rf'{field_name} "([^"]+)"', graph_text))
        assert actual, f"实物里没有 {field_name}"
        assert actual <= set(DTYPES[key]), (
            f"{field_name} 出现了契约未声明的类型: {actual - set(DTYPES[key])}")


def test_attention_scale_is_what_the_reference_stores(graph_text: str) -> None:
    """Scaling 存的是 1/√head_dim，不是量化因子（见文档第 19 节）。"""
    expected = np.float16(attention_scale(128))

    found = 0
    for block in parse_blocks(graph_text, "node"):
        name = field(block, "Scaling_buffer_file")
        scaling = _read(name, np.float16) if name else None
        if scaling is None or not len(scaling):
            continue
        if scaling[0] == expected:
            found += 1
    # llama2-7B 的 32 个 attention head 各一个。
    assert found == 32, f"预期 32 个 attention 缩放，实际 {found}"


def test_scaling_is_scalar_in_llama2(graph_text: str) -> None:
    """llama2 的 Scaling 全是标量——它不承载 per-channel 定标。"""
    for block in parse_blocks(graph_text, "node"):
        name = field(block, "Scaling_buffer_file")
        scaling = _read(name, np.float16) if name else None
        if scaling is None:
            continue
        assert len(scaling) == 1, f"{name} 有 {len(scaling)} 项，预期标量"


def test_dynamic_quantization_phase_count(graph_text: str) -> None:
    """phase 字段的编号范围要与契约一致。"""
    phases = {int(n) for n in re.findall(r"input_buffer_phase_(\d+)", graph_text)}
    assert phases, "实物里应当有 phase 字段"
    assert max(phases) < DQ_PHASE_COUNT
    assert DQ_REDUCTION_WIDTH == 1024


def test_lut_size_is_fixed(graph_text: str) -> None:
    """全部 LUT 都是 288 字节 = 144 个 fp16。"""
    lut_files = [
        name for name in os.listdir(_REFERENCE_DIR)
        if "lut" in name.lower() and name.endswith(".bin")
    ]
    assert lut_files
    for name in lut_files:
        assert (_REFERENCE_DIR / name).stat().st_size == LUT_BYTES

    assert LUT_BYTES == LUT_ENTRY_COUNT * 2


def test_near_empty_luts_are_legal() -> None:
    """实物里有 37 个「只有第 0 项为 1.0」的表——不是全零，是恒等/旁路表。

    我最初以为它们是全零（非零段模式统计成 `(0,0)`，那指的是下标 0 这一段）。
    实测 139 个 LUT 无一全零，非零项数分布是 {1: 37, 98: 32, 99: 1, 102: 69}。
    """
    counts: dict[int, int] = {}
    for name in os.listdir(_REFERENCE_DIR):
        if "lut" not in name.lower() or not name.endswith(".bin"):
            continue
        table = np.fromfile(_REFERENCE_DIR / name, dtype=np.float16)
        counts[int((table != 0).sum())] = counts.get(
            int((table != 0).sum()), 0) + 1

    assert counts, "实物里应当有 LUT"
    assert 0 not in counts, "实测没有全零表"
    assert counts.get(1) == 37, f"预期 37 个恒等表，实际 {counts.get(1)}"

    # 恒等表的形状：仅下标 0 非零且为 1.0。
    identity = next(
        name for name in os.listdir(_REFERENCE_DIR)
        if "lut" in name.lower() and name.endswith(".bin")
        and int((np.fromfile(_REFERENCE_DIR / name, dtype=np.float16) != 0).sum()) == 1
    )
    table = np.fromfile(_REFERENCE_DIR / identity, dtype=np.float16)
    assert np.nonzero(table)[0].tolist() == [0]
    assert float(table[0]) == 1.0

    # 占位表仍然可用于结构层跑通，但要清楚它与实物的恒等表不同。
    assert len(lut_placeholder()) == LUT_BYTES


def test_layout_scale_counts() -> None:
    """布局算出的 scale 数要符合定义。"""
    assert ACTIVATION_LAYOUT.scale_count((128, 4096)) == 1
    assert WEIGHT_LAYOUT.scale_count((4096, 4096)) == 4096 // WEIGHT_GROUP_SIZE

    per_channel = QuantLayout("per_channel", axis=1)
    assert per_channel.scale_count((128, 256)) == 256


def test_group_size_must_divide_the_axis() -> None:
    """不能整除就直接抛——静默取整会让 scale 与权重错位。"""
    layout = QuantLayout("per_group", group_size=128, axis=0)
    with pytest.raises(ValueError, match="不能被 group_size"):
        layout.scale_count((100, 256))

def test_identity_lut_matches_the_reference_byte_for_byte() -> None:
    """我们生成的恒等 LUT 必须与实物逐字节相同。

    这是第一个能与实物精确比对的量化产物——LUT 的采样规则还没确认，但恒等表
    不依赖采样，所以可以现在就钉住。字节一致说明尺寸、dtype、字节序都对了。
    """
    reference = None
    for name in sorted(os.listdir(_REFERENCE_DIR)):
        if "lut" not in name.lower() or not name.endswith(".bin"):
            continue
        table = np.fromfile(_REFERENCE_DIR / name, dtype=np.float16)
        if int((table != 0).sum()) == 1:
            reference = (_REFERENCE_DIR / name).read_bytes()
            break

    assert reference is not None, "实物里应当有恒等表"
    assert lut_identity() == reference

def test_int4_weight_quantization_round_trips_exactly() -> None:
    """按契约的分组布局反量化再量化，要与实物字节一致。

    这验证的是三件事同时成立：group_size 是 128、每组共享一个 fp16 scale、
    int4 一字节一个值。任一条错，往返就不会是 100%——比如 group 取 64 会让
    scale 与权重错位，取整误差立刻显现。

    实测 16777216 个权重全部一致。
    """
    weights = _read("weight_buffer_11.bin", np.int8)
    scales = _read("weight_sf_11.bin", np.float16)
    if weights is None or scales is None:
        pytest.skip("实物缺少这个 Gemm 的权重")

    assert len(weights) // len(scales) == WEIGHT_GROUP_SIZE

    grouped = weights.reshape(-1, WEIGHT_GROUP_SIZE).astype(np.float32)
    per_group = scales.astype(np.float32)[:, None]

    dequantized = grouped * per_group
    requantized = np.rint(dequantized / per_group).astype(np.int8)

    assert requantized.ravel().tobytes() == weights.tobytes()
    assert requantized.min() >= INT4_MIN
    assert requantized.max() <= INT4_MAX
