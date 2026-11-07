"""验证共享块表下的两请求切换：物理槽不相交，重写块表即换目标。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table
from runtime.executor import KVBlockTable, KVSlotPool, execute_plan
from tests.kv_helpers import read_rows, write_row


HEAD_DIM = 4
MAX_SEQ = 8
DTYPE_BYTES = 4  # float32
SLOT_TOKENS = 2


def _make_backend(num_dpus: int = 1) -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=num_dpus, mram_bytes_per_dpu=4 * 2**20))


def _spec(dpu_id: int = 0, pool_slots: int = 4) -> KVRegionSpec:
    spec = KVRegionSpec(
        dpu_id=dpu_id,
        layers=[0],
        kv_heads=[0],
        q_heads_by_kv={0: [0]},
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype_bytes=DTYPE_BYTES,
        kv_base=0,
        slot_tokens=SLOT_TOKENS,
    )
    return build_kv_layout(spec, align=64, pool_slots=pool_slots)


def test_two_requests_use_disjoint_physical_slots() -> None:
    """两个请求的物理槽互不相交；重写块表后读到各自的槽。"""
    hal = _make_backend()
    spec = _spec()
    pool = KVSlotPool(spec.pool_slots)
    first = KVBlockTable(pool)
    first.ensure(2)
    second = KVBlockTable(pool)
    second.ensure(2)
    assert set(first.slots).isdisjoint(second.slots)

    write_row(hal, spec, first.slots[0], 0, 0, "k", 0, np.full(HEAD_DIM, 7.0))
    write_row(hal, spec, second.slots[0], 0, 0, "k", 0, np.full(HEAD_DIM, 3.0))

    write_block_table(spec, hal, first.slots)
    assert np.array_equal(read_rows(hal, spec, 0, 0, "k", 1)[0],
                          np.full(HEAD_DIM, 7.0))

    write_block_table(spec, hal, second.slots)
    assert np.array_equal(read_rows(hal, spec, 0, 0, "k", 1)[0],
                          np.full(HEAD_DIM, 3.0))


def test_bind_inputs_passes_pos_and_num_tokens() -> None:
    """bind_inputs 默认不绑定位置；显式传入后按原值暴露。"""
    hal = _make_backend()
    hal.bind_inputs({}, pos=None)
    assert hal.bound_pos is None
    assert hal.bound_num_tokens is None

    hal.bind_inputs({}, pos=5, num_tokens=2)
    assert hal.bound_pos == 5
    assert hal.bound_num_tokens == 2


def test_execute_plan_passes_pos_and_num_tokens_to_hal() -> None:
    """execute_plan 把 pos 和 token 数透传给 hal。"""
    hal = _make_backend()
    plan = ExecutionPlan(commands=[])
    execute_plan(plan, hal, values={}, pos=0, num_tokens=1)
    assert hal.bound_pos == 0
    assert hal.bound_num_tokens == 1
