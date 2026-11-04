"""PWLA 查找表生成器：自产的 288 字节表要与厂商参考文件逐字节一致。"""

import json
from pathlib import Path

import pytest
import torch

from contracts.pwla import LUT_BYTES, PWLA_PARAMS, pwla_table

_ROOT = Path(__file__).resolve().parents[1]
_REF_SUBDIR = "decode/model_layers_0_decode_v2/parser_output/tvmgen_default_nprm_main_0/runtime_files"

# 参考文件 -> 表种类。softmax 的倒数与 DQ 的倒数是同一组参数。
_CASES = [
    ("LUT_phase_1_12.bin", "identity"),
    ("LUT_phase_2_12.bin", "reciprocal"),
    ("LUT_phase_1_18.bin", "softmax_exp"),
    ("LUT_phase_3_18.bin", "reciprocal"),
    ("activation_lut_file_195.bin", "silu"),
]


def _ref_dir() -> Path:
    cfg = json.loads((_ROOT / "paths.json").read_text())
    return Path(cfg["ceva_reference_dir"]) / _REF_SUBDIR


@pytest.mark.parametrize("kind", sorted(PWLA_PARAMS))
def test_table_is_288_bytes(kind):
    assert len(pwla_table(kind)) == LUT_BYTES == 288


@pytest.mark.parametrize("ref_name,kind", _CASES)
def test_table_matches_vendor(ref_name, kind):
    ref = _ref_dir() / ref_name
    if not ref.is_file():
        pytest.skip(f"缺少参考文件 {ref}")
    want = ref.read_bytes()
    got = pwla_table(kind)
    diff = sum(a != b for a, b in zip(got, want))
    assert diff == 0, f"{kind} 与 {ref_name} 有 {diff} 字节不同"


def test_default_dtype_unchanged():
    pwla_table.cache_clear()
    before = torch.get_default_dtype()
    for kind in PWLA_PARAMS:
        pwla_table(kind)
    assert torch.get_default_dtype() == before


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        pwla_table("tanh")
