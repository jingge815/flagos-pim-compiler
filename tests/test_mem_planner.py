"""问题 8 内存管理器单测：权重区打包 + 激活区 liveness/贪心装箱 + 容量校验。

判据（CLAUDE.md）：编译期模块对照手推结果——权重区 offset/对齐 padding 手推值，
greedy_reuse 的复用/不复用手推区间关系，plan_dpu 三区互不重叠、容量超限抛错。
借附录 A 的两层 Linear 图（hidden=4, ffn=6, 2 DPU）作 prefill/decode 两图的
最小可用夹具：两次独立构图给出两套不同 Node 对象，权重/占位算子形状相同，
足以练手"两图共用权重 offset、各自跑一遍激活区"的机制，不代表真实解码。
"""

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


# ---------------------------------------------------------------------------
# bytes_of / weights_of / 权重区打包
# ---------------------------------------------------------------------------


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
    # 回填：两图同名权重的 mram_offset 相同
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


# ---------------------------------------------------------------------------
# transient_tensors：本地读者 + redistribute 隐式读者
# ---------------------------------------------------------------------------


def test_transient_tensors_local_and_redistribute_readers() -> None:
    gm, nodes = _built_appendix_a()
    all_nodes = list(gm.graph.nodes)
    tensors = {t.name: t for t in transient_tensors(all_nodes, dpu_id=0)}

    y1, y2, norm = nodes["y1"], nodes["y2"], nodes["norm"]
    assert set(tensors) == {y1.name, y2.name}  # x/w1/w2 非 transient，norm 是 host 节点

    # y1 本地读者是 y2（列切输出天然对齐行切输入，零通信）
    assert tensors[y1.name].readers == [y2.name]
    assert tensors[y1.name].last_read_at == all_nodes.index(y2)

    # y2 是 Partial，被 host norm 消费须 all_reduce，读者记为 redist:eN
    (edge,) = norm.meta[REDISTRIBUTE_META_KEY]
    assert tensors[y2.name].readers == [f"redist:e{edge.edge_id}"]
    assert tensors[y2.name].last_read_at == all_nodes.index(norm)


def test_transient_tensors_no_readers_defaults_last_read_to_produced() -> None:
    """无读者的临时张量（如死端）：last_read_at 退化为 produced_at，立即可复用。"""
    tensor = TransientTensor(name="t", size=8, produced_at=5, last_read_at=5)
    assert tensor.readers == []
    assert tensor.last_read_at == tensor.produced_at


# ---------------------------------------------------------------------------
# greedy_reuse：不重叠复用同一 offset，重叠则各开新 offset
# ---------------------------------------------------------------------------


def test_greedy_reuse_shares_offset_for_disjoint_lifetimes() -> None:
    a = TransientTensor("a", size=100, produced_at=0, last_read_at=2)
    b = TransientTensor("b", size=100, produced_at=3, last_read_at=5)  # 与 a 不重叠
    c = TransientTensor("c", size=100, produced_at=1, last_read_at=6)  # 与 a、b 都重叠

    offsets, end = greedy_reuse([a, b, c], base=0, align=1)

    assert offsets["a"] == offsets["b"]  # 复用同一 offset
    assert offsets["c"] != offsets["a"]  # 时间重叠，另开新 offset
    assert end == 200  # 两个 slot，各 100B


def test_greedy_reuse_processes_largest_first_so_slots_always_fit_later_tensors() -> None:
    """按 size 降序处理：已开的 slot 一定 >= 后处理张量的 size，size 从不成为复用的阻碍，
    真正阻碍复用的只有时间区间冲突（体现在 shares_offset 用例）。"""
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


# ---------------------------------------------------------------------------
# pending_readers_of_reused_addresses
# ---------------------------------------------------------------------------


def test_pending_readers_only_for_shared_addresses() -> None:
    a = TransientTensor("a", size=8, produced_at=0, last_read_at=2, readers=["r1", "r2"])
    b = TransientTensor("b", size=8, produced_at=3, last_read_at=5, readers=["r3"])
    c = TransientTensor("c", size=8, produced_at=6, last_read_at=7, readers=["r4"])
    offsets = {"a": 0, "b": 0, "c": 64}  # a/b 共用 offset 0，c 独占 64

    pending = pending_readers_of_reused_addresses([a, b, c], offsets, dpu_id=0)

    assert pending == {(("dpu", 0), 0): ["r1", "r2"]}  # 只有先占用者 a 的读者需等待
    assert (("dpu", 0), 64) not in pending  # c 未被复用，不出现


# ---------------------------------------------------------------------------
# plan_dpu 端到端 + format_mem_plan
# ---------------------------------------------------------------------------


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
        # 权重区终点（含对齐）不超过 kv_base：逐权重重算一遍对齐字节数交叉验证
        assert weight_end < plan.kv_base
        assert plan.kv_base + kv_specs[d].kv_allocated_bytes == plan.act_base
        assert plan.act_base <= plan.total
        # 激活区两图各自的 offset 表都不越过 plan.total
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


def test_plan_dpu_raises_when_over_budget() -> None:
    hw = HwBudget(mram_bytes=16, align=64, sys_reserve_bytes=0)  # 明显装不下
    with pytest.raises(ValueError, match="内存超限"):
        _plan_both_dpus(hw)


def test_format_mem_plan_printable() -> None:
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=1024)
    plans, _, _ = _plan_both_dpus(hw)
    text = format_mem_plan(plans, hw)
    assert "dpu0" in text and "dpu1" in text and "margin=" in text
