"""P0-6：排布规则搬家后的数值不变与「无重复实现」。

规则从 `orchestrator/layer_fields.py` 搬进 `contracts/mem_layout.py`，逻辑一字
不改。下表是**搬家前**用编排器现算出来的值，冻在这里当基线 —— 搬家后两侧都
要能对上它，改公式就会立刻失败。

（对照测试只比 `contracts/mem_layout` 与 `orchestrator`，搬家后后者 import 前者，
自比恒等、失去意义，所以基线必须落在本文件里。）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import mem_layout as ml

WIDTHS = (1, 16, 86, 128, 1024, 4096, 11008)
KINDS = ("bmm1", "bmm2", "dq_p2", "dq_p4", "sm_p2", "gemm", "mask", "softmax")

# 搬家前 align16
FROZEN_ALIGN16 = {1: 16, 16: 16, 86: 96, 128: 128, 1024: 1024, 4096: 4096,
                  11008: 11008}

# 搬家前 stride_z，每项按 (final, scalar_align16) 的 (F,F) (F,T) (T,F) (T,T) 排
FROZEN_STRIDE_Z = {
    1: (1, 16, 31, 31),
    16: (16, 16, 31, 31),
    86: (86, 86, 111, 111),
    128: (128, 128, 143, 143),
    1024: (1024, 1024, 1039, 1039),
    4096: (4096, 4096, 4111, 4111),
    11008: (11008, 11008, 11023, 11023),
}

# 搬家前 L2 输入段（dt=0，即 int8 平面）
FROZEN_L2_IN = {
    "bmm1": (17, 32, 102, 144, 1040, 4112, 11024),
    "bmm2": (17, 32, 102, 144, 1040, 4112, 11024),
    "dq_p2": (1, 16, 86, 128, 1024, 4096, 11008),
    "dq_p4": (64, 64, 224, 288, 2080, 8224, 22048),
    "sm_p2": (64, 64, 224, 288, 2080, 8224, 22048),
    "gemm": (1, 16, 86, 128, 1024, 4096, 11008),
    "mask": (1, 16, 86, 128, 1024, 4096, 11008),
    "softmax": (1, 16, 86, 128, 1024, 4096, 11008),
}

# 搬家前 L2 输出段（dt=0）
FROZEN_L2_OUT = {
    "bmm1": (32, 32, 112, 144, 1040, 4112, 11024),
    "bmm2": (8224, 8224, 8224, 8224, 8224, 8224, 8224),
    "dq_p2": (32, 64, 208, 288, 2080, 8224, 22048),
    "dq_p4": (32, 32, 112, 144, 1040, 4112, 11024),
    "sm_p2": (32, 32, 112, 144, 1040, 4112, 11024),
    "gemm": (32, 32, 112, 144, 1040, 4112, 11024),
    "mask": (32, 32, 112, 144, 1040, 4112, 11024),
    "softmax": (32, 32, 112, 144, 1040, 4112, 11024),
}

# 搬家前走 `_elem_bytes` 那一路的值（dt=1 是 fp16、dt=3 是 fp32）
FROZEN_ELEM_BYTES = {
    ("gemm", 128, 1): (256, 288),
    ("gemm", 128, 3): (512, 576),
    ("gemm", 4096, 1): (8192, 8224),
    ("gemm", 4096, 3): (16384, 16448),
    ("mask", 4096, 1): (8192, 8224),
    ("mask", 4096, 3): (16384, 16448),
}


@pytest.mark.parametrize("width", WIDTHS)
def test_align16_matches_the_frozen_baseline(width: int) -> None:
    assert ml.align16(width) == FROZEN_ALIGN16[width]


@pytest.mark.parametrize("width", WIDTHS)
def test_stride_z_matches_the_frozen_baseline(width: int) -> None:
    got = tuple(ml.stride_z(width, final=final, scalar_align16=scalar)
                for final in (False, True) for scalar in (False, True))
    assert got == FROZEN_STRIDE_Z[width]


@pytest.mark.parametrize("kind", KINDS)
def test_l2_in_size_matches_the_frozen_baseline(kind: str) -> None:
    got = tuple(ml.l2_in_size(kind, width, 0) for width in WIDTHS)
    assert got == FROZEN_L2_IN[kind]


@pytest.mark.parametrize("kind", KINDS)
def test_l2_out_size_matches_the_frozen_baseline(kind: str) -> None:
    got = tuple(ml.l2_out_size(kind, width, 0) for width in WIDTHS)
    assert got == FROZEN_L2_OUT[kind]


def test_element_byte_path_matches_the_frozen_baseline() -> None:
    """fp16 / fp32 平面那条路（`Width × 元素宽度`）也要钉住。"""
    for (kind, width, dt), (want_in, want_out) in FROZEN_ELEM_BYTES.items():
        assert ml.l2_in_size(kind, width, dt) == want_in, (kind, width, dt)
        assert ml.l2_out_size(kind, width, dt) == want_out, (kind, width, dt)


def test_dual_in_size_is_width_times_two() -> None:
    """双输入层每一路都是 fp16 平面：`Width × 2`。"""
    for width in WIDTHS:
        assert ml.l2_dual_in_size("residual", width) == width * 2
        assert ml.l2_dual_in_size("mask", width) == width * 2


def test_the_dq_p2_exception_is_preserved() -> None:
    """`dq_p2` 单组的例外是实测来的，用例要能钉住它。

    Gn=32→96、86→208（`align16((Gn+16)×2)`），Gn=1 例外取 32 —— 按公式得 48。
    """
    assert ml.l2_out_size("dq_p2", 32, 0) == 96
    assert ml.l2_out_size("dq_p2", 86, 0) == 208
    assert ml.l2_out_size("dq_p2", 1, 0) == 32


def test_the_final_phase_output_does_not_double() -> None:
    """`dq_p4` 已是 int8 平面，不乘 2：4096→4112、11008→11024。"""
    assert ml.l2_out_size("dq_p4", 4096, 0) == 4112
    assert ml.l2_out_size("dq_p4", 11008, 0) == 11024


def test_l2_output_bytes_follows_the_closed_formula() -> None:
    assert ml.l2_output_bytes(4096, 2) == (4096 + 16) * 2
    assert ml.l2_output_bytes(100, 2) == (112 + 16) * 2


def test_the_orchestrator_has_no_duplicate_implementation() -> None:
    """P0-6 的硬判据：规则在 IR 侧只有一份，编排器侧无重复实现。

    编排器退化为渲染层 —— 从 IR 读出步幅与对齐，按目标格式写成层参数文本。
    """
    root = Path(__file__).parent.parent / "orchestrator"
    banned = (r"\s*def align_up\b", r"\s*def align16\b", r"\s*def stride_z\b",
              r"\s*def _l2_in_size\b", r"\s*def _l2_out_size\b",
              r"\s*def _l2_dual_in_size\b", r"\s*def l2_output_bytes\b")
    hits = [f"{py.name}:{i}"
            for py in root.glob("*.py")
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1)
            if any(re.match(p, line) for p in banned)]
    assert hits == [], f"编排器里还有排布规则的本地实现：{hits}"


def test_the_orchestrator_actually_consumes_the_ir_rules() -> None:
    """删完本地实现后要真的从 IR 侧引进来，否则是「删了没接」。"""
    root = Path(__file__).parent.parent / "orchestrator"
    text = (root / "layer_fields.py").read_text(encoding="utf-8")
    assert "from contracts.mem_layout import" in text


# 搬家前 net.ini [general] 的四个全网恒定步幅（`orchestrator/layer_hw_table.py`
# 原值，域含义见 `docs/prepare_out-域确认表-20260918.md` Q10）。
NET_INI_STRIDE_BASELINE = {
    "input_line_stride": 8,
    "input_map_stride": 4,
    "output_line_stride": 12,
    "output_map_stride": 5,
}


def test_net_ini_strides_live_in_the_ir() -> None:
    """四个全网恒定步幅归统一 IR，取值与搬家前逐项相同。

    设计 §4.6.5 把它们列为 P0-6 的搬家对象：它们是排布规则（步幅），
    不是 `net.ini` 的文件格式，所以真源不该留在编排器支线。
    """
    assert ml.NET_INI_STRIDES == NET_INI_STRIDE_BASELINE


def test_the_orchestrator_consumes_the_net_ini_strides() -> None:
    """编排器改为引用，不保留第二份取值（P0-6「不保留双份」）。"""
    root = Path(__file__).parent.parent / "orchestrator"
    table = (root / "layer_hw_table.py").read_text(encoding="utf-8")
    assert "NET_INI_STRIDES" in table, "编排器没有引用 IR 侧的步幅"
    for name, value in NET_INI_STRIDE_BASELINE.items():
        assert f'"{name}": {value}' not in table, \
            f"{name} 在编排器里还有一份字面量取值"
