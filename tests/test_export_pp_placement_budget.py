"""放置导出脚本的硬件参数入口。

num-dpus / mram-bytes / sys-reserve-bytes 缺省从 CEVA 画像推导（64 核、
每核 8,000,000,000 B、无系统预留）；其他场景可用命令行显式覆盖。
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


def test_defaults_come_from_the_ceva_profile(monkeypatch):
    """未显式给出时，核数与字节预算都从 CEVA 64 核画像推导。"""
    args = _parse(monkeypatch, "--num-stages", "4")
    assert args.num_dpus is None
    assert args.mram_bytes is None
    assert args.sys_reserve_bytes is None
    profile = _module.apply_profile_defaults(args)
    assert profile.total_clusters == 64
    assert args.num_dpus == 64
    assert args.mram_bytes == 8_000_000_000
    assert args.sys_reserve_bytes == 0


def test_explicit_num_dpus_beats_the_profile(monkeypatch):
    args = _parse(monkeypatch, "--num-stages", "4", "--num-dpus", "32")
    _module.apply_profile_defaults(args)
    assert args.num_dpus == 32
    # 未覆盖的项仍从画像取。
    assert args.mram_bytes == 8_000_000_000


def test_mram_bytes_override(monkeypatch):
    args = _parse(
        monkeypatch, "--num-stages", "4", "--mram-bytes", "7000000000")
    _module.apply_profile_defaults(args)
    assert args.mram_bytes == 7000000000
    assert args.num_dpus == 64


def test_sys_reserve_override(monkeypatch):
    args = _parse(
        monkeypatch, "--num-stages", "4", "--sys-reserve-bytes", str(64 * 2**20))
    _module.apply_profile_defaults(args)
    assert args.sys_reserve_bytes == 64 * 2**20


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
