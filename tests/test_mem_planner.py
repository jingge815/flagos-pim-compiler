"""验证权重、KV 和激活区的内存规划。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import REDISTRIBUTE_META_KEY
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from memory.kv_layout import KVRegionSpec
from memory.mem_planner import (
    HwBudget,
    TransientTensor,
    _build_step_index,
    _pack_weights,
    bytes_of,
    format_mem_plan,
    greedy_reuse,
    pending_readers_of_reused_addresses,
    plan_dpu,
    transient_tensors,
    weights_of,
)
from tests.test_spec_prop import _appendix_a_config, _appendix_a_graph


def _built_appendix_a():
    """一次性 partition + propagate 的附录 A 图，返回 (gm, nodes-by-key)。"""
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)
    propagate_specs(gm, _appendix_a_config())
    return gm, nodes


def _two_appendix_a_graphs():
    """两次独立构图（模拟 prefill/decode 两张导出图，权重形状相同）。"""
    return _built_appendix_a(), _built_appendix_a()


def _kv_specs() -> dict[int, KVRegionSpec]:
    return {
        d: KVRegionSpec(
            dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
            max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0,
        )
        for d in (0, 1)
    }


# 字节统计和权重区布局。


def test_bytes_of() -> None:
    assert bytes_of((3, 4), 4) == 48


def test_weights_of_picks_dpu_owned_get_attr() -> None:
    gm, nodes = _built_appendix_a()
    owned = weights_of(list(gm.graph.nodes), dpu_id=0)
    assert set(owned) == {nodes["w1"].target, nodes["w2"].target}
    assert "x" not in owned  # placeholder，非 get_attr


def test_pack_weights_offsets_and_alignment() -> None:
    """w1 本地 (3,4) fp32=48B，w2 本地 (4,3) fp32=48B，按名字排序对齐到 64。"""
    (gm1, nodes1), (gm2, nodes2) = _two_appendix_a_graphs()
    offsets, total = _pack_weights(0, list(gm1.graph.nodes), list(gm2.graph.nodes), align=64)
    w1_name, w2_name = nodes1["w1"].target, nodes1["w2"].target
    assert sorted(offsets) == sorted([w1_name, w2_name])
    first, second = sorted(offsets, key=offsets.get)
    assert offsets[first] == 0
    assert offsets[second] == 64  # 48 对齐到 64
    assert total == 128
    # 两张图的同名权重共用 MRAM 偏移。
    for name in (w1_name, w2_name):
        off = offsets[name]
        for nodes in (nodes1, nodes2):
            key = "w1" if nodes["w1"].target == name else "w2"
            assert nodes[key].meta["spec"].shard_map[0].mram_offset == off


def test_pack_weights_rejects_cross_graph_shape_mismatch() -> None:
    (gm1, nodes1), (gm2, nodes2) = _two_appendix_a_graphs()
    spec = nodes2["w1"].meta["spec"]
    bad_detail = replace(spec.shard_map[0], local_shape=(99, 4))
    spec.shard_map[0] = bad_detail
    with pytest.raises(ValueError, match="切分不一致"):
        _pack_weights(0, list(gm1.graph.nodes), list(gm2.graph.nodes), align=64)


# 临时张量和重分布读者。


def test_transient_tensors_local_and_redistribute_readers() -> None:
    gm, nodes = _built_appendix_a()
    all_nodes = list(gm.graph.nodes)
    node_step, edge_step = _build_step_index(all_nodes)
    tensors = {t.name: t for t in transient_tensors(all_nodes, dpu_id=0)}

    y1, y2, norm = nodes["y1"], nodes["y2"], nodes["norm"]
    # scatter 输入作为 DPU 的落地缓冲。
    (scatter_edge,) = y1.meta[REDISTRIBUTE_META_KEY]
    assert set(tensors) == {y1.name, y2.name, f"redist_dst:e{scatter_edge.edge_id}"}

    # y1 的本地读取按细粒度步骤编号记录。
    assert tensors[y1.name].readers == [y2.name]
    assert tensors[y1.name].last_read_at == node_step[y2]

    # y2 的归约读取使用重分布命令步骤编号。
    (edge,) = norm.meta[REDISTRIBUTE_META_KEY]
    assert tensors[y2.name].readers == [f"redist:e{edge.edge_id}"]
    assert tensors[y2.name].last_read_at == edge_step[edge.edge_id]


def test_redistribute_landing_tensor_offset_backfilled_after_plan_dpu() -> None:
    """验证重分布落地缓冲获得激活区偏移。"""
    (gm1, nodes1), (gm2, nodes2) = _two_appendix_a_graphs()
    kv_specs = _kv_specs()
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    plan = plan_dpu(0, list(gm1.graph.nodes), list(gm2.graph.nodes), kv_specs, hw)

    (scatter_edge,) = nodes1["y1"].meta[REDISTRIBUTE_META_KEY]
    detail = scatter_edge.dst_spec.shard_map[0]
    assert detail.mram_offset != 0
    assert plan.act_base <= detail.mram_offset < plan.total
    # 偏移与贪心复用结果一致。
    assert plan.act_prefill[f"redist_dst:e{scatter_edge.edge_id}"] == detail.mram_offset


def test_transient_tensors_no_readers_defaults_last_read_to_produced() -> None:
    """验证无读者临时张量的读写区间仅包含生成步骤。"""
    tensor = TransientTensor(name="t", size=8, produced_at=5, last_read_at=5)
    assert tensor.readers == []
    assert tensor.last_read_at == tensor.produced_at


# 贪心复用临时缓冲。


def test_greedy_reuse_shares_offset_for_disjoint_lifetimes() -> None:
    a = TransientTensor("a", size=100, produced_at=0, last_read_at=2)
    b = TransientTensor("b", size=100, produced_at=3, last_read_at=5)  # 与 a 不重叠
    c = TransientTensor("c", size=100, produced_at=1, last_read_at=6)  # 与 a、b 都重叠

    offsets, end = greedy_reuse([a, b, c], base=0, align=1)

    assert offsets["a"] == offsets["b"]  # 复用同一 offset
    assert offsets["c"] != offsets["a"]  # 时间重叠，另开新 offset
    assert end == 200  # 两个 slot，各 100B


def test_greedy_reuse_processes_largest_first_so_slots_always_fit_later_tensors() -> None:
    """验证较小且生命周期不重叠的张量复用较大张量的地址。"""
    big = TransientTensor("big", size=100, produced_at=0, last_read_at=1)
    small_disjoint = TransientTensor("small", size=8, produced_at=2, last_read_at=3)  # 与 big 不冲突

    offsets, end = greedy_reuse([big, small_disjoint], base=0, align=1)

    assert offsets["small"] == offsets["big"] == 0  # 复用同一 slot（slot 大小取开者的 size）
    assert end == 100


def test_greedy_reuse_aligns_new_slots() -> None:
    a = TransientTensor("a", size=10, produced_at=0, last_read_at=1)
    b = TransientTensor("b", size=10, produced_at=0, last_read_at=1)  # 与 a 重叠，须新开
    offsets, end = greedy_reuse([a, b], base=0, align=64)
    assert offsets["a"] == 0
    assert offsets["b"] == 64  # 新 slot 对齐到 64
    assert end == 128


def test_greedy_reuse_never_aliases_same_step_read_and_write() -> None:
    """产出步与前一个张量的最后读取步相同时，不得复用同一地址。

    这正是「某节点在同一 step 里既读旧张量、又写自己的输出」的情形。已编译内核
    逐块读输入、逐块写输出，共用地址会覆盖尚未读取的输入行，算出错误结果
    （实测相对误差可达 5.9 到 110）。两个生命周期判据都必须取严格不等号：
    大张量先进槽，所以危险场景可能落在任一分支上。
    """
    # 旧张量在 step 5 被最后一次读取，新张量正好在 step 5 产出。
    consumed = TransientTensor("consumed", size=512, produced_at=4, last_read_at=5)
    produced = TransientTensor("produced", size=2048, produced_at=5, last_read_at=9)

    offsets, _ = greedy_reuse([consumed, produced], base=0, align=64)
    assert offsets["consumed"] != offsets["produced"]

    # 反向的大小关系（输出更小）走的是另一个判据分支，同样不得别名。
    big_in = TransientTensor("big_in", size=2048, produced_at=4, last_read_at=5)
    small_out = TransientTensor("small_out", size=512, produced_at=5, last_read_at=9)

    offsets, _ = greedy_reuse([big_in, small_out], base=0, align=64)
    assert offsets["big_in"] != offsets["small_out"]

    # 真正隔开一步的生命周期仍然可以复用，严格化没有把复用能力废掉。
    early = TransientTensor("early", size=512, produced_at=0, last_read_at=2)
    late = TransientTensor("late", size=512, produced_at=3, last_read_at=5)
    offsets, _ = greedy_reuse([early, late], base=0, align=64)
    assert offsets["early"] == offsets["late"]


# 复用地址的待完成读者。


def test_pending_readers_only_for_shared_addresses() -> None:
    a = TransientTensor("a", size=8, produced_at=0, last_read_at=2, readers=["r1", "r2"])
    b = TransientTensor("b", size=8, produced_at=3, last_read_at=5, readers=["r3"])
    c = TransientTensor("c", size=8, produced_at=6, last_read_at=7, readers=["r4"])
    offsets = {"a": 0, "b": 0, "c": 64}  # a/b 共用 offset 0，c 独占 64

    pending = pending_readers_of_reused_addresses([a, b, c], offsets, dpu_id=0)

    assert pending == {(("dpu", 0), 0): ["r1", "r2"]}  # 只有先占用者 a 的读者需等待
    assert (("dpu", 0), 64) not in pending  # c 未被复用，不出现


# plan_dpu 和内存计划格式化。


def _plan_both_dpus(hw: HwBudget):
    (gm1, nodes1), (gm2, nodes2) = _two_appendix_a_graphs()
    kv_specs = _kv_specs()
    plans = {
        d: plan_dpu(d, list(gm1.graph.nodes), list(gm2.graph.nodes), kv_specs, hw)
        for d in (0, 1)
    }
    return plans, kv_specs, (nodes1, nodes2)


def test_plan_dpu_regions_do_not_overlap() -> None:
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    plans, kv_specs, _ = _plan_both_dpus(hw)

    for d, plan in plans.items():
        weight_end = max(plan.weight.values()) if plan.weight else 0
        # 对齐后的权重区终点不超过 KV 区起点。
        assert weight_end < plan.kv_base
        assert plan.kv_base + kv_specs[d].kv_allocated_bytes == plan.act_base
        assert plan.act_base <= plan.total
        # 两张图的激活区偏移不超过总容量。
        for offsets in (plan.act_prefill, plan.act_decode):
            for name, off in offsets.items():
                assert plan.act_base <= off < plan.total


def test_plan_dpu_weight_offsets_shared_across_both_graphs() -> None:
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    plans, _, (nodes1, nodes2) = _plan_both_dpus(hw)
    plan = plans[0]
    for key in ("w1", "w2"):
        name = nodes1[key].target
        off = plan.weight[name]
        assert nodes1[key].meta["spec"].shard_map[0].mram_offset == off
        assert nodes2[key].meta["spec"].shard_map[0].mram_offset == off


def test_plan_dpu_backfills_activation_node_own_spec_offset() -> None:
    """验证计划将 DPU 节点输出偏移写回张量规格。"""
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    plans, _, (nodes1, nodes2) = _plan_both_dpus(hw)
    plan = plans[0]
    for key in ("y1", "y2"):
        off = plan.act_prefill[nodes1[key].name]
        assert nodes1[key].meta["spec"].shard_map[0].mram_offset == off
        assert off != 0
        # decode 图使用独立的激活区偏移表。
        off2 = plan.act_decode[nodes2[key].name]
        assert nodes2[key].meta["spec"].shard_map[0].mram_offset == off2


def test_plan_dpu_raises_when_over_budget() -> None:
    hw = HwBudget(mram_bytes=16, align=64, sys_reserve_bytes=0)  # 明显装不下
    with pytest.raises(ValueError, match="内存超限"):
        _plan_both_dpus(hw)


def test_format_mem_plan_printable() -> None:
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=1024)
    plans, _, _ = _plan_both_dpus(hw)
    text = format_mem_plan(plans, hw)
    assert "dpu0" in text and "dpu1" in text and "margin=" in text
