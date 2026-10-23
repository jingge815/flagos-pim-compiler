"""验证多 slot KV 布局的地址不重叠和容量计算。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.kv_layout import KVRegionSpec, build_kv_layout, kv_specs_multi_slot
from memory.mem_planner import HwBudget, plan_dpu
from tests.test_mem_planner import _two_appendix_a_graphs


def _base_specs(num_dpus: int = 2) -> dict[int, KVRegionSpec]:
    return {
        d: KVRegionSpec(
            dpu_id=d,
            layers=[0],
            kv_heads=[d],
            q_heads_by_kv={d: [d]},
            max_seq=8,
            head_dim=4,
            dtype_bytes=4,
            kv_base=0,
        )
        for d in range(num_dpus)
    }


def test_multi_slot_offsets_do_not_overlap() -> None:
    """slot 0 和 slot 1 的 kv_off 不重叠。"""
    base = _base_specs()
    multi = kv_specs_multi_slot(base, num_slots=2, align=64)
    for dpu_id, slots in multi.items():
        slot0_offs = set(slots[0].kv_off.values())
        slot1_offs = set(slots[1].kv_off.values())
        assert not (slot0_offs & slot1_offs), (
            f"dpu{dpu_id}: slot 0 和 slot 1 的 kv_off 存在重叠"
        )
        # slot 1 的每个 offset 必须大于 slot 0 的所有 offset
        assert min(slot1_offs) > max(slot0_offs)


def test_multi_slot_total_bytes_equals_num_slots_times_single() -> None:
    """总分配字节 = num_slots × 单 slot 字节。"""
    base = _base_specs()
    # 先取单 slot 字节数
    single_specs = {d: build_kv_layout(KVRegionSpec(
        dpu_id=s.dpu_id, layers=list(s.layers), kv_heads=list(s.kv_heads),
        q_heads_by_kv=dict(s.q_heads_by_kv), max_seq=s.max_seq,
        head_dim=s.head_dim, dtype_bytes=s.dtype_bytes, kv_base=0,
    ), align=64) for d, s in base.items()}
    single_bytes = {d: spec.kv_allocated_bytes for d, spec in single_specs.items()}

    for num_slots in (1, 2, 4):
        multi = kv_specs_multi_slot(base, num_slots=num_slots, align=64)
        for dpu_id, slots in multi.items():
            assert len(slots) == num_slots
            # 从 slot 0 基址到 slot N-1 末端的跨度 = num_slots × 单 slot 字节
            span = (slots[-1].kv_base - slots[0].kv_base) + slots[-1].kv_allocated_bytes
            assert span == num_slots * single_bytes[dpu_id], (
                f"dpu{dpu_id} num_slots={num_slots}: "
                f"span={span} != {num_slots} * {single_bytes[dpu_id]}"
            )


def test_multi_slot_kv_specs_invalid_num_slots() -> None:
    base = _base_specs()
    with pytest.raises(ValueError, match="num_slots"):
        kv_specs_multi_slot(base, num_slots=0, align=64)


def test_plan_dpu_num_slots_multiplies_kv_bytes() -> None:
    """plan_dpu 的 KV 区总字节随 num_slots 线性增长。"""
    (gm1, _), (gm2, _) = _two_appendix_a_graphs()
    base_kv = {
        d: KVRegionSpec(
            dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
            max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0,
        )
        for d in (0, 1)
    }
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    nodes1, nodes2 = list(gm1.graph.nodes), list(gm2.graph.nodes)

    import copy
    plan1 = plan_dpu(0, nodes1, nodes2, copy.deepcopy(base_kv), hw, num_slots=1)
    plan2 = plan_dpu(0, nodes1, nodes2, copy.deepcopy(base_kv), hw, num_slots=2)

    kv_bytes_1 = plan1.act_base - plan1.kv_base
    kv_bytes_2 = plan2.act_base - plan2.kv_base
    assert kv_bytes_2 == 2 * kv_bytes_1, (
        f"num_slots=2 的 KV 区 {kv_bytes_2} 应为 num_slots=1 的 {kv_bytes_1} 的 2 倍"
    )
