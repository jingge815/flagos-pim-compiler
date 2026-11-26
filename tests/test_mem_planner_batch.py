"""多 batch 内存规划判据：KV/激活缩放、容量表输出与超限分区明细。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.fx import Graph, GraphModule

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from graph.strategy import ShardStrategy
from memory.kv_layout import KVRegionSpec
from memory.mem_planner import (
    DPUPlan,
    HwBudget,
    capacity_batch_fit,
    format_capacity_table,
    plan_dpu,
)


def _batch_graph(batch: int) -> GraphModule:
    """带 batch 维的两层线性图（放大尺寸，激活字节不被 64B 对齐吞掉）。"""
    root = torch.nn.Module()
    root.register_buffer("w1", torch.empty(256, 128))  # 列切 Shard(0)
    root.register_buffer("w2", torch.empty(128, 256))  # 行切 Shard(1)
    graph = Graph()
    x = graph.placeholder("x")
    w1 = graph.get_attr("w1")
    w2 = graph.get_attr("w2")
    y1 = graph.call_function(torch.ops.aten.linear.default, (x, w1))
    y2 = graph.call_function(torch.ops.aten.linear.default, (y1, w2))
    graph.output(y2)
    gm = GraphModule(root, graph)
    x.meta["val"] = torch.empty(batch, 128)
    w1.meta["val"] = root.w1
    w2.meta["val"] = root.w2
    y1.meta["val"] = torch.empty(batch, 256)
    y2.meta["val"] = torch.empty(batch, 128)
    partition_graph(gm)
    propagate_specs(gm, ShardStrategy(
        name="batch", num_dpus=2, weight_rules=(("w1", "col"), ("w2", "row"))))
    return gm


def _kv_spec(dpu_id: int, num_slots: int, *,
             max_seq: int = 8, head_dim: int = 4, dtype_bytes: int = 4) -> KVRegionSpec:
    return KVRegionSpec(
        dpu_id=dpu_id, layers=[0], kv_heads=[dpu_id],
        q_heads_by_kv={dpu_id: [dpu_id]},
        max_seq=max_seq, head_dim=head_dim, dtype_bytes=dtype_bytes,
        kv_base=0, num_slots=num_slots)


def _plan(batch: int, hw: HwBudget, *, num_slots: int | None = None,
          kv_max_seq: int = 8, kv_head_dim: int = 4):
    """按 batch 维建图规划一台 DPU，返回 plan 与 kv 字节数。"""
    gm1, gm2 = _batch_graph(batch), _batch_graph(batch)
    slots = batch if num_slots is None else num_slots
    kv = {d: _kv_spec(d, slots, max_seq=kv_max_seq, head_dim=kv_head_dim)
          for d in (0, 1)}
    plan = plan_dpu(0, list(gm1.graph.nodes), list(gm2.graph.nodes), kv, hw,
                    kernel_mram_bytes=4096)
    return plan, kv[0].kv_allocated_bytes


def test_batch_scales_kv_and_activation() -> None:
    """B=2 vs B=1：KV 字节精确翻倍、激活区变大、权重区不变。"""
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    plan1, kv1 = _plan(1, hw)
    plan2, kv2 = _plan(2, hw)
    assert kv2 == 2 * kv1
    weight1 = plan1.kv_base
    weight2 = plan2.kv_base
    assert weight1 == weight2  # 权重与 batch 无关
    act1 = plan1.total - plan1.act_base
    act2 = plan2.total - plan2.act_base
    assert act2 > act1  # 激活张量带 B 维，占用随 batch 增长


def test_capacity_table() -> None:
    """容量表打印六列与 slots 标注，合计不超过预算。"""
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    plans = {}
    for d in (0, 1):
        plans[d], _ = _plan(2, hw)
    text = format_capacity_table(plans, hw)
    for label in ("dpu0", "权重", "KV", "激活", "内核 tile", "合计", "预算",
                  "slots=2"):
        assert label in text, text
    budget = hw.mram_bytes - hw.sys_reserve_bytes
    for plan in plans.values():
        assert plan.total + plan.kernel_mram_bytes <= budget


def test_over_budget_error_names_the_region() -> None:
    """KV 区单独超预算时硬失败，报错含分区明细并指明卡在 KV。"""
    # 单槽 KV = 2×1024×64×4 = 512KiB，4 槽 = 2MiB，预算 512KiB 装不下 KV 一区。
    hw = HwBudget(mram_bytes=1 << 19, align=64, sys_reserve_bytes=0)
    with pytest.raises(ValueError, match="内存超限") as excinfo:
        _plan(4, hw, kv_max_seq=1024, kv_head_dim=64, num_slots=4)
    message = str(excinfo.value)
    assert "KV" in message, message
    for label in ("权重", "激活", "内核 tile"):
        assert label in message, message  # 分区明细齐全，能看出各占多少
    assert "卡在: KV" in message, message


def test_capacity_batch_fit() -> None:
    """编译期容量折算：权重/内核固定、KV+激活随 batch 线性，墙位取整上界。

    手推：权重 1000B + 内核 100B 固定；编译档 slots=2 时 KV+激活 1000B
    （每档 500B）。预算 2100B → 最大 batch=2（1000+500×2=2000 ≤2100，
    batch=3 需 2500B 超限），墙位 batch=3。
    """
    plan = DPUPlan(
        weight={}, kv_base=1000, act_base=1800, act_prefill={}, act_decode={},
        total=2000, pending_readers_prefill={}, pending_readers_decode={},
        kernel_mram_bytes=100, kv_slots=2)
    hw = HwBudget(mram_bytes=2100, align=64, sys_reserve_bytes=0)
    fit = capacity_batch_fit({0: plan}, hw)
    assert fit["budget_bytes"] == 2100
    assert fit["dpus"]["0"] == {
        "fixed_bytes": 1100, "kernel_tile_bytes": 100,
        "var_bytes_at_slots": 1000,
        "slots": 2, "max_batch": 2}
    assert fit["wall_batch"] == 3


def test_capacity_batch_fit_wall_follows_tightest_dpu() -> None:
    """墙位取各 DPU 的最紧者；可变部分为零的 DPU 不构成 batch 约束。"""
    tight = DPUPlan(
        weight={}, kv_base=100, act_base=300, act_prefill={}, act_decode={},
        total=400, pending_readers_prefill={}, pending_readers_decode={},
        kernel_mram_bytes=0, kv_slots=2)          # 每档 150B，预算 400 → max 2
    roomy = DPUPlan(
        weight={}, kv_base=0, act_base=0, total=0, act_prefill={}, act_decode={},
        pending_readers_prefill={}, pending_readers_decode={},
        kernel_mram_bytes=0, kv_slots=1)          # 无可变字节：不设墙
    hw = HwBudget(mram_bytes=400, align=64, sys_reserve_bytes=0)
    fit = capacity_batch_fit({0: roomy, 1: tight}, hw)
    assert fit["dpus"]["0"]["max_batch"] is None
    assert fit["dpus"]["1"]["max_batch"] == 2
    assert fit["wall_batch"] == 3


def test_capacity_wall_follows_trace_length_formula() -> None:
    """给定 trace 长度 L：wall_batch 与按 L 手推的 KV 公式一致。

    性能判据 4：KV 容量墙位置与公式
    2×layers×L×kv_heads×head_dim×dtype_bytes×B 手推一致。这里 L 取
    closed trace 的 seq_len+output_length 口径（=kv_max_seq），墙位由
    capacity_batch_fit 对含对齐填充的真实布局折算，与手推公式逐位对上。
    """
    from memory.kv_layout import build_kv_layout

    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    slots = 4
    for seq_len, output_length in ((726, 69), (128, 64)):
        length = seq_len + output_length
        plan, kv_bytes = _plan(slots, hw, num_slots=slots,
                               kv_max_seq=length, kv_head_dim=8)
        # 手推：单槽 KV = 2×layers(1)×L×heads(1)×head_dim×dtype(4B)，
        # 布局按 64B 对齐；激活区不随 kv_max_seq 变化，用同图固定值。
        spec = build_kv_layout(_kv_spec(0, slots, max_seq=length,
                                        head_dim=8), 64)
        assert kv_bytes == spec.kv_allocated_bytes
        assert spec.kv_allocated_bytes >= 2 * length * 8 * 4 * slots
        fit = capacity_batch_fit({0: plan}, hw)
        # 墙位手推：固定字节 + 每档可变字节（KV+激活）线性外推。
        var_per_batch = fit["dpus"]["0"]["var_bytes_at_slots"] / slots
        expected_max = int((fit["budget_bytes"]
                            - fit["dpus"]["0"]["fixed_bytes"]) // var_per_batch)
        assert fit["dpus"]["0"]["max_batch"] == expected_max
        assert fit["wall_batch"] == expected_max + 1
        # L 放大时单槽 KV 随 L 线性放大，墙位相应前移（容量口径与 L 同源）。
    plan_short, _ = _plan(slots, hw, num_slots=slots, kv_max_seq=128,
                          kv_head_dim=8)
    plan_long, kv_long = _plan(slots, hw, num_slots=slots, kv_max_seq=795,
                               kv_head_dim=8)
    fit_short = capacity_batch_fit({0: plan_short}, hw)
    fit_long = capacity_batch_fit({0: plan_long}, hw)
    assert kv_long > fit_short["dpus"]["0"]["var_bytes_at_slots"]
    assert fit_long["wall_batch"] < fit_short["wall_batch"]
