"""放置导出脚本的 MRAM 预算入口。

默认与 GeneSim 的 CEVA 64 核 4e 画像一致：每核 8,000,000,000 B、无系统预留；
其他场景可用命令行显式覆盖。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "export_pp_placement.py"
_spec = importlib.util.spec_from_file_location("export_pp_placement", _SCRIPT)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def _parse(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["export_pp_placement.py", *args])
    return _module.parse_args()


def test_defaults_use_the_ceva_budget(monkeypatch):
    args = _parse(monkeypatch, "--num-stages", "4")
    assert args.mram_bytes == 8_000_000_000
    assert args.sys_reserve_bytes == 0


def test_mram_bytes_override(monkeypatch):
    args = _parse(
        monkeypatch, "--num-stages", "4", "--mram-bytes", "8000000000")
    assert args.mram_bytes == 8000000000


def test_sys_reserve_override(monkeypatch):
    args = _parse(
        monkeypatch, "--num-stages", "4", "--sys-reserve-bytes", "0")
    assert args.sys_reserve_bytes == 0


def test_budget_uses_the_requested_bytes():
    hw = _module.build_hw_budget(8000000000, 64 * 2**20)
    assert hw.mram_bytes == 8000000000
    assert hw.sys_reserve_bytes == 64 * 2**20
    assert hw.align == 64


def test_rejects_non_positive_mram_bytes():
    with pytest.raises(SystemExit):
        _module.build_hw_budget(0, 0)


def test_rejects_reserve_larger_than_capacity():
    with pytest.raises(SystemExit):
        _module.build_hw_budget(1 << 20, 1 << 20)
