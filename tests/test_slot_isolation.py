"""验证 slot_id 隔离：不同 slot 的 KV 写入不互相影响。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Command, ExecutionPlan
from memory.kv_layout import KVRegionSpec, build_kv_layout, kv_specs_multi_slot
from runtime.executor import execute_plan
from runtime.kernels import sdpa_kv_info


HEAD_DIM = 4
MAX_SEQ = 8
DTYPE_BYTES = 4  # float32


def _make_backend(num_dpus: int = 1) -> NumpyBackend:
    # 每 DPU 4MB，足够放 2 个 slot 的 KV
    return NumpyBackend(NumpyBackendConfig(num_dpus=num_dpus, mram_bytes_per_dpu=4 * 2**20))


def _base_spec(dpu_id: int = 0) -> KVRegionSpec:
    return KVRegionSpec(
        dpu_id=dpu_id,
        layers=[0],
        kv_heads=[0],
        q_heads_by_kv={0: [0]},
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype_bytes=DTYPE_BYTES,
        kv_base=0,
    )


def test_slot_isolation_write_does_not_affect_other_slot() -> None:
    """slot_id=0 的 KV 写入不影响 slot_id=1 的 MRAM 区域。"""
    hal = _make_backend(num_dpus=1)
    base = {0: _base_spec()}
    multi = kv_specs_multi_slot(base, num_slots=2, align=64)

    slot0, slot1 = multi[0][0], multi[0][1]

    # 向 slot0 的 KV 区写已知数据
    row = MAX_SEQ * HEAD_DIM * DTYPE_BYTES
    data_slot0 = np.ones(HEAD_DIM, dtype=np.float32) * 7.0
    k_off0 = slot0.kv_off[(0, 0, "k")]
    hal.write_local(0, k_off0, np.zeros((MAX_SEQ, HEAD_DIM), dtype=np.float32))
    hal.write_local(0, k_off0, data_slot0.reshape(1, HEAD_DIM))  # 只写第一行

    # 向 slot1 的 KV 区写不同数据
    k_off1 = slot1.kv_off[(0, 0, "k")]
    hal.write_local(0, k_off1, np.ones((MAX_SEQ, HEAD_DIM), dtype=np.float32) * 3.0)

    # 读回 slot0 第一行，不应被 slot1 的写入污染
    read_back = hal.read_local(0, k_off0, (1, HEAD_DIM), np.dtype(np.float32))
    assert np.allclose(read_back[0], data_slot0), (
        f"slot0 KV 数据被 slot1 写入污染：{read_back[0]}"
    )

    # slot0 和 slot1 的起始地址不重叠
    assert k_off0 != k_off1
    assert abs(k_off1 - k_off0) >= slot0.kv_allocated_bytes


def test_bind_inputs_slot_id_stored() -> None:
    """bind_inputs 应正确存储 slot_id，默认为 0。"""
    hal = _make_backend()
    hal.bind_inputs({}, pos=None)
    assert hal.bound_slot_id == 0

    hal.bind_inputs({}, pos=5, slot_id=3)
    assert hal.bound_slot_id == 3
    assert hal.bound_pos == 5


def test_execute_plan_passes_slot_id_to_hal() -> None:
    """execute_plan(slot_id=N) 应把 slot_id 透传给 hal。"""
    hal = _make_backend()
    # 空计划，只验证 slot_id 被绑定
    plan = ExecutionPlan(commands=[])
    execute_plan(plan, hal, values={}, pos=0, slot_id=2)
    assert hal.bound_slot_id == 2
