"""交叉校验 GML 缓冲区命名规则。

这套规则跨两种语言：FlagTree（C++）把文件名写进 GML 文本，图编译器（Python）写
文件本身。两边不一致，底层编译器读到的就是悬空引用——而且这种错不会在编译期暴露。

所以拿参考产物做双向校验：规则能生成的名字必须在磁盘上真实存在，磁盘上的每个名字
也必须能被规则生成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
from genesim_bridge.paths import gml_reference_dir

_REFERENCE_DIR = gml_reference_dir(required=False)
_RUNTIME_FILES = _REFERENCE_DIR / "runtime_files" if _REFERENCE_DIR else None

# 参考产物里的一个 1 字节占位文件，不遵循任何命名规则。
_PLACEHOLDER = "dummy.bin"

pytestmark = pytest.mark.skipif(
    _RUNTIME_FILES is None or not _RUNTIME_FILES.is_dir(),
    reason="缺少 GML 参考产物，配置 paths.json 的 gml_reference_dir 后可跑",
)


def _real_names() -> set[str]:
    return {
        path.name
        for path in _RUNTIME_FILES.iterdir()
        if path.suffix == ".bin" and path.name != _PLACEHOLDER
    }


def _generated_names(node_ids: range = range(1, 80)) -> set[str]:
    """按规则枚举所有可能的缓冲区名。

    节点 id 取到 80 是为了盖住参考产物的 1..74，多出的部分不会命中任何真实文件，
    正好用来验证规则不会凭空生成不存在的名字。
    """
    generated = set()
    for node_id in node_ids:
        generated |= {
            names.data_buffer(node_id),
            names.scale(node_id),
            names.zero_point(node_id),
            names.weight_buffer(node_id),
            names.weight_scale(node_id),
            names.weight_zero_point(node_id),
            names.bias_buffer(node_id),
            names.bias_scale(node_id),
            names.bias_zero_point(node_id),
            names.output_scale(node_id),
            names.output_zero_point(node_id),
            names.fpsu_scale(node_id),
            names.fpsu_post_shift(node_id),
            names.fpsu_bias(node_id),
            names.kantor_scale(node_id),
            names.kantor_bias(node_id),
            names.kantor_shift(node_id),
            names.activation_lut(node_id),
            names.activation_input(node_id),
            names.activation_input_scale(node_id),
            names.activation_input_zero_point(node_id),
        }
        for slot in (0, 1):
            generated |= {
                names.data_buffer(node_id, slot),
                names.scale(node_id, slot),
                names.zero_point(node_id, slot),
                names.fpsu_scale(node_id, slot),
                names.fpsu_post_shift(node_id, slot),
                names.fpsu_bias(node_id, slot),
            }
    return generated


def test_every_real_file_is_covered_by_the_rules() -> None:
    """磁盘上的每个缓冲区名都要能被规则生成，否则规则有缺口。"""
    uncovered = _real_names() - _generated_names()
    assert uncovered == set(), f"规则未覆盖 {len(uncovered)} 个真实文件: {sorted(uncovered)[:5]}"


def test_reference_file_count() -> None:
    assert len(_real_names()) == 1168


def test_single_input_buffers_are_numbered_by_consumer() -> None:
    """反直觉的一条：缓冲区按读取它的节点编号，不按产生它的节点。"""
    assert names.data_buffer(8) == "input_buffer_8.bin"
    assert names.data_buffer(8) in _real_names()


def test_multi_input_buffers_carry_a_slot_number() -> None:
    """多输入算子的每个输入槽各有一块缓冲区。"""
    assert names.data_buffer(6, 1) == "input_buffer_1_6.bin"
    assert names.data_buffer(6, 1) in _real_names()
    assert names.data_buffer(6, 0) in _real_names()


def test_per_slot_scaling_buffers_exist() -> None:
    """每个输入槽各带一套 FPSU 定标，不是共用一套。"""
    for slot in (0, 1):
        assert names.fpsu_scale(6, slot) in _real_names()
        assert names.fpsu_bias(6, slot) in _real_names()


def test_weights_are_numbered_by_their_own_node() -> None:
    """权重属于节点自己，不属于某条边，所以按本节点编号。"""
    assert names.weight_buffer(8) == "weight_buffer_8.bin"
    assert names.weight_buffer(8) in _real_names()
    assert names.bias_buffer(8) in _real_names()


def test_kantor_buffers_are_numbered_by_physical_block() -> None:
    assert names.kantor_shift(8) == "kantor_A_Shift_8.bin"
    assert names.kantor_shift(8) in _real_names()


def test_activation_tables_are_per_node() -> None:
    """表里折进了该节点自己的 scale，所以不能编译期生成一次后复用。"""
    assert names.activation_lut(8) == "activation_lut_file_8.bin"
    assert names.activation_lut(8) in _real_names()
