"""验证 bind_plan 只平移激活区地址和块表窗口。"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.exec_plan import Access, Command, ExecutionPlan
from runtime.binding import bind_plan
from runtime.kernels import _ctx_of

ACT_BASE = {0: 300, 1: 800}
BT_STRIDE = 64


def _plan() -> ExecutionPlan:
    launch = Command(
        id=0, op="launch", dpu_id=0,
        payload={
            "kernel": "linear", "mram_offset": 64, "align_bytes": 64,
            "elem_strides": (1,),
            "sdpa": {"layer": 0, "dpus": {
                0: {"bt_base": 5000, "slot_bytes": 128},
                1: {"bt_base": 6000, "slot_bytes": 128},
            }},
        },
        reads=[Access(("dpu", 0), 300, 16),   # 激活区
               Access(("dpu", 0), 100, 16)],  # 权重区
        writes=[Access(("dpu", 0), 400, 16)],
        num_tasklets=4,
    )
    host = Command(
        id=1, op="host_slice", dpu_id=None, payload={"edge_id": 7},
        reads=[Access(("dpu", 1), 900, 8)], writes=[], waits=[0],
    )
    return ExecutionPlan([launch, host])


def test_window_zero_returns_the_original_plan() -> None:
    plan = _plan()
    assert bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=0,
                     block_slot=0, bt_stride=BT_STRIDE) is plan


def test_activation_accesses_shift_but_weight_and_kv_do_not() -> None:
    plan = _plan()
    bound = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=1000,
                      block_slot=1, bt_stride=BT_STRIDE)

    launch, host = bound.commands
    assert launch.reads[0].offset == 1300          # 激活区 +shift
    assert launch.reads[1].offset == 100           # 权重区不动
    assert launch.writes[0].offset == 1400
    assert host.reads[0].offset == 1900            # dpu1 激活区
    assert launch.id == 0 and host.id == 1
    assert host.waits == [0]
    assert launch.num_tasklets == 4


def test_sdpa_block_table_base_shifts_per_window() -> None:
    bound = bind_plan(_plan(), act_base_by_dpu=ACT_BASE, act_shift=1000,
                      block_slot=2, bt_stride=BT_STRIDE)
    dpus = bound.commands[0].payload["sdpa"]["dpus"]
    assert dpus[0]["bt_base"] == 5000 + 2 * BT_STRIDE
    assert dpus[1]["bt_base"] == 6000 + 2 * BT_STRIDE
    assert dpus[0]["slot_bytes"] == 128             # KV 几何不动


def test_mram_offset_is_never_shifted_so_kernels_share_one_cache() -> None:
    plan = _plan()
    window1 = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=1000,
                        block_slot=1, bt_stride=BT_STRIDE)
    window2 = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=2000,
                        block_slot=2, bt_stride=BT_STRIDE)
    for bound in (window1, window2):
        assert bound.commands[0].payload["mram_offset"] == 64
    # 编译缓存键只看硬件/切分/排布，窗口不同也必须相等。
    assert _ctx_of(window1.commands[0]) == _ctx_of(window2.commands[0])
    assert _ctx_of(window1.commands[0]) == _ctx_of(plan.commands[0])


def test_original_plan_and_payloads_are_not_mutated_or_aliased() -> None:
    plan = _plan()
    snapshot = copy.deepcopy(plan)
    bound = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=1000,
                      block_slot=1, bt_stride=BT_STRIDE)

    assert plan == snapshot, "原计划必须逐字段不变"
    assert bound.commands[0].payload is not plan.commands[0].payload
    assert (bound.commands[0].payload["sdpa"]
            is not plan.commands[0].payload["sdpa"])
    bound.commands[0].payload["sdpa"]["dpus"][0]["bt_base"] = -1
    assert plan == snapshot, "改窗口 payload 不能波及原计划"


def test_other_window_payloads_do_not_alias_each_other() -> None:
    plan = _plan()
    window1 = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=1000,
                        block_slot=1, bt_stride=BT_STRIDE)
    window2 = bind_plan(plan, act_base_by_dpu=ACT_BASE, act_shift=2000,
                        block_slot=2, bt_stride=BT_STRIDE)
    assert window1.commands[0].payload["sdpa"] is not window2.commands[0].payload["sdpa"]
    window1.commands[0].payload["sdpa"]["dpus"][0]["bt_base"] = 0
    assert window2.commands[0].payload["sdpa"]["dpus"][0]["bt_base"] == 5000 + 2 * BT_STRIDE
