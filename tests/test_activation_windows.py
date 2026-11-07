"""验证激活窗口、块表窗口和容量判据的几何。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table
from memory.mem_planner import DPUPlan, HwBudget, fit_activation_windows

HEAD_DIM = 4
DTYPE_BYTES = 4
ALIGN = 64
MAX_SEQ = 6
SLOT_TOKENS = 2


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=2**16))


def _spec(kv_base: int = 1024, pool_slots: int = 6,
          block_table_slots: int = 1) -> KVRegionSpec:
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=MAX_SEQ, head_dim=HEAD_DIM, dtype_bytes=DTYPE_BYTES,
        kv_base=kv_base, slot_tokens=SLOT_TOKENS,
    )
    return build_kv_layout(spec, ALIGN, pool_slots=pool_slots,
                           block_table_slots=block_table_slots)


def _plan(act_base: int, extent: int) -> DPUPlan:
    return DPUPlan(
        weight={}, kv_base=0, act_base=act_base, act_prefill={}, act_decode={},
        total=act_base + extent, pending_readers_prefill={},
        pending_readers_decode={}, act_extent=extent,
    )


def test_default_layout_keeps_single_block_table() -> None:
    """默认一份块表：槽池紧贴块表之后，与改动前逐字节一致。"""
    spec = _spec()
    assert spec.block_table_slots == 1
    assert spec.block_table_base == 1024
    assert spec.block_table_bytes == 64
    assert spec.slots_base == 1024 + 64
    assert spec.kv_allocated_bytes == 64 + 6 * 128


def test_multiple_block_tables_shift_slots_base_and_allocation() -> None:
    """k 份块表把槽池整体后移 k*块表字节，总占用按份数计。"""
    spec = _spec(block_table_slots=3)
    bt = spec.block_table_bytes
    assert spec.block_table_slots == 3
    assert spec.slots_base == 1024 + 3 * bt
    assert spec.kv_allocated_bytes == 3 * bt + 6 * spec.slot_bytes
    assert spec.kv_off[(0, 0, "k")] == spec.slots_base


def test_write_block_table_only_touches_its_own_window() -> None:
    """写第 k 份块表不影响其它窗口的表项。"""
    hal = _backend()
    spec = _spec(block_table_slots=3)
    for k, table in enumerate(([0], [2, 1], [4])):
        write_block_table(spec, hal, table, slot_index=k)

    for k, table in enumerate(([0], [2, 1], [4])):
        offset = spec.block_table_base + k * spec.block_table_bytes
        entries = hal.read_local(0, offset, (spec.blocks_per_request,), np.int32)
        expected = table + [-1] * (spec.blocks_per_request - len(table))
        assert entries.tolist() == expected


def test_write_block_table_rejects_out_of_range_window() -> None:
    hal = _backend()
    spec = _spec(block_table_slots=2)
    with pytest.raises(ValueError, match="块表窗口"):
        write_block_table(spec, hal, [0], slot_index=2)


def test_fit_activation_windows_uses_one_uniform_stride() -> None:
    """多台 DPU 统一取最大占用对齐后的步长，每台 total = act_base + k*stride。"""
    plans = {0: _plan(act_base=1000, extent=100),
             1: _plan(act_base=2000, extent=200)}
    hw = HwBudget(mram_bytes=10**6, align=ALIGN, sys_reserve_bytes=0)

    stride = fit_activation_windows(plans, act_slots=3, hw=hw)

    assert stride == 256  # align_up(max(100, 200), 64)
    for plan in plans.values():
        assert plan.act_stride == stride
        assert plan.total == plan.act_base + 3 * stride


def test_fit_activation_windows_reports_overflow_per_dpu() -> None:
    """窗口数放大后超出预算要抛错，并指出是哪台 DPU。"""
    plans = {0: _plan(act_base=500, extent=100)}
    hw = HwBudget(mram_bytes=800, align=ALIGN, sys_reserve_bytes=0)
    with pytest.raises(ValueError, match="dpu0.*内存超限"):
        fit_activation_windows(plans, act_slots=3, hw=hw, kernel_mram_bytes=10)


def test_single_window_is_byte_identical_to_extent() -> None:
    """act_slots=1 时 total 回到 act_base+对齐后的 extent。"""
    plans = {0: _plan(act_base=1000, extent=128)}
    hw = HwBudget(mram_bytes=10**6, align=ALIGN, sys_reserve_bytes=0)
    fit_activation_windows(plans, act_slots=1, hw=hw)
    assert plans[0].total == 1000 + 128
