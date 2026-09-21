"""对拍器命名归一：节点号通配，槽位/相位/段号/头号保留。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.diff_prepare_out import _normalise_naming_value


def test_node_id_is_wildcarded_regardless_of_magnitude():
    assert (_normalise_naming_value("input_buffer_18.bin")
            == _normalise_naming_value("input_buffer_8.bin")
            == "input_buffer_#.bin")
    assert (_normalise_naming_value("buffer8")
            == _normalise_naming_value("buffer193")
            == "buffer#")


def test_slot_and_phase_are_kept():
    assert (_normalise_naming_value("input_buffer_0_18.bin")
            == _normalise_naming_value("input_buffer_0_25.bin")
            == "input_buffer_0_#.bin")
    assert (_normalise_naming_value("input_buffer_18.bin")
            != _normalise_naming_value("input_buffer_0_18.bin"))
    assert (_normalise_naming_value("input_buffer_phase_0_22.bin")
            == "input_buffer_phase_0_#.bin")


def test_rope_unit_and_map_head_are_kept():
    a = _normalise_naming_value(
        "Scaling_buffer_file_5_Llama2Activation_Cos_22.bin")
    b = _normalise_naming_value(
        "Scaling_buffer_file_6_Llama2Activation_Cos_184.bin")
    assert a != b
    assert "5" in a and "6" in b
    assert (_normalise_naming_value("buffer19_map0")
            != _normalise_naming_value("buffer19_map5"))
    assert _normalise_naming_value("buffer19_map5").endswith("map5")


def test_qidx_and_params_are_wildcarded():
    assert (_normalise_naming_value("self_attn_Reshape_qidx4_params_22")
            == _normalise_naming_value("self_attn_Reshape_qidx36_params_184"))
