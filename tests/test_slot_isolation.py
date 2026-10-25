"""验证请求槽隔离：两个请求的页表指向不相交的物理槽。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import KVRegionSpec, build_kv_layout, write_page_table
from runtime.executor import KVBlockTable, KVSlotPool, execute_plan
from tests.kv_helpers import read_rows, write_row


HEAD_DIM = 4
MAX_SEQ = 8
DTYPE_BYTES = 4  # float32
SLOT_TOKENS = 2


def _make_backend(num_dpus: int = 1) -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=num_dpus, mram_bytes_per_dpu=4 * 2**20))


def _spec(dpu_id: int = 0, num_request_slots: int = 2) -> KVRegionSpec:
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
    return build_kv_layout(spec, align=64, num_request_slots=num_request_slots)


def test_request_slots_use_disjoint_physical_slots() -> None:
    """请求槽 0/1 的页表与写入互不干扰。"""
    hal = _make_backend()
    spec = _spec()
    pool = KVSlotPool(spec.pool_slots)
    blocks = []
    for request_slot in range(2):
        table = KVBlockTable(pool)
        table.ensure(2)
        write_page_table(spec, hal, request_slot, table.slots)
        blocks.append(table.slots)
    assert set(blocks[0]).isdisjoint(blocks[1])

    write_row(hal, spec, blocks[0][0], 0, 0, "k", 0, np.full(HEAD_DIM, 7.0))
    write_row(hal, spec, blocks[1][0], 0, 0, "k", 0, np.full(HEAD_DIM, 3.0))

    first = read_rows(hal, spec, 0, 0, 0, "k", 1)
    second = read_rows(hal, spec, 1, 0, 0, "k", 1)
    assert np.array_equal(first[0], np.full(HEAD_DIM, 7.0))
    assert np.array_equal(second[0], np.full(HEAD_DIM, 3.0))


def test_bind_inputs_request_slot_stored() -> None:
    """bind_inputs 应正确存储 request_slot 和 num_tokens，默认 0/None。"""
    hal = _make_backend()
    hal.bind_inputs({}, pos=None)
    assert hal.bound_request_slot == 0
    assert hal.bound_num_tokens is None

    hal.bind_inputs({}, pos=5, request_slot=3, num_tokens=2)
    assert hal.bound_request_slot == 3
    assert hal.bound_num_tokens == 2
    assert hal.bound_pos == 5


def test_execute_plan_passes_request_slot_to_hal() -> None:
    """execute_plan(request_slot=N) 应把编号和 token 数透传给 hal。"""
    hal = _make_backend()
    plan = ExecutionPlan(commands=[])
    execute_plan(plan, hal, values={}, pos=0, request_slot=2, num_tokens=1)
    assert hal.bound_request_slot == 2
    assert hal.bound_num_tokens == 1
