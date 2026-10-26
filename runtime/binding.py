"""按请求窗口平移执行计划：只动激活区地址和块表窗口。"""

from __future__ import annotations

import copy
from dataclasses import replace

from contracts.exec_plan import Access, Command, ExecutionPlan


def _shift_access(access: Access, act_base_by_dpu: dict[int, int],
                  act_shift: int) -> Access:
    """DPU 侧落在激活区的访问平移；权重与 KV 区不动。"""
    if access.loc[0] != "dpu":
        return access
    base = act_base_by_dpu.get(access.loc[1])
    if base is None or access.offset < base:
        return access
    return replace(access, offset=access.offset + act_shift)


def _bind_payload(payload: dict, act_base_by_dpu: dict[int, int],
                  act_shift: int, block_slot: int, bt_stride: int) -> dict:
    """复制 payload，平移 SDPA 块表基址，并注入窗口信息供 DMA 与归因使用。"""
    bound = dict(payload)
    sdpa = bound.get("sdpa")
    if sdpa is not None:
        sdpa = copy.deepcopy(sdpa)
        for geo in sdpa["dpus"].values():
            geo["bt_base"] += block_slot * bt_stride
        bound["sdpa"] = sdpa
    # `mram_offset` 是编译缓存键，绝不平移；窗口信息只作为额外字段下发。
    bound["act_shift"] = act_shift
    bound["act_base_by_dpu"] = dict(act_base_by_dpu)
    bound["block_slot"] = block_slot
    return bound


def bind_plan(
    plan: ExecutionPlan,
    *,
    act_base_by_dpu: dict[int, int],
    act_shift: int,
    block_slot: int,
    bt_stride: int,
) -> ExecutionPlan:
    """把计划平移到请求窗口；窗口 0（无平移）原样返回，零拷贝。"""
    if act_shift == 0 and block_slot == 0:
        return plan
    commands = [
        replace(
            cmd,
            payload=_bind_payload(
                cmd.payload, act_base_by_dpu, act_shift, block_slot, bt_stride),
            reads=[_shift_access(a, act_base_by_dpu, act_shift) for a in cmd.reads],
            writes=[_shift_access(a, act_base_by_dpu, act_shift) for a in cmd.writes],
        )
        for cmd in plan.commands
    ]
    return ExecutionPlan(commands=commands)
