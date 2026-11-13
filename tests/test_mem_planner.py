"""验证权重、KV 和激活区的内存规划。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.fx import Graph

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import REDISTRIBUTE_META_KEY
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from memory.kv_layout import KVRegionSpec, kv_bytes
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
    redistribute_landing_tensors,
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


def test_the_kernel_tile_footprint_counts_against_the_same_budget() -> None:
    """内核 tile 占用与三区相加才是单台峰值 —— 判据必须把两者加在一起。

    三区（权重 + KV + 激活）是常驻，内核的 x/w/out tile 是算子执行期的临时占用，
    两者并存于同一块 MRAM。改动前这里只校验三区，而算子编译器回传的 tile 占用
    只被 GeneSim 那侧核对 —— 于是「三区刚好装满、内核再要一份」这种真实超限
    两边都看不见。

    判据设计成「不带内核占用时通过、带上就超」，所以它只在相加真的发生时才红。
    """
    probe_hw = HwBudget(mram_bytes=1 << 30, align=64, sys_reserve_bytes=0)
    (gm1, _), (gm2, _) = _two_appendix_a_graphs()
    kv_specs = _kv_specs()
    nodes1, nodes2 = list(gm1.graph.nodes), list(gm2.graph.nodes)
    three_regions = plan_dpu(0, nodes1, nodes2, kv_specs, probe_hw).total

    # 预算恰好等于三区：不带内核占用通过。
    tight = HwBudget(mram_bytes=three_regions, align=64, sys_reserve_bytes=0)
    plan_dpu(0, nodes1, nodes2, kv_specs, tight)

    # 同一个预算，带上内核 tile：必须被拒，且报错要说清是哪两部分相加。
    with pytest.raises(ValueError, match="内存超限") as excinfo:
        plan_dpu(0, nodes1, nodes2, kv_specs, tight, kernel_mram_bytes=4096)
    assert "内核 tile" in str(excinfo.value), str(excinfo.value)

    # 预算放宽到容得下两部分：通过（判据不是一味从严）。
    roomy = HwBudget(mram_bytes=three_regions + 4096, align=64,
                     sys_reserve_bytes=0)
    plan_dpu(0, nodes1, nodes2, kv_specs, roomy, kernel_mram_bytes=4096)


def test_plan_reports_peak_single_allocation() -> None:
    """单块最大分配进入计划，供多 TensorPU 容量核验使用。"""
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    (gm1, _), (gm2, _) = _two_appendix_a_graphs()
    kv_specs = _kv_specs()
    nodes1, nodes2 = list(gm1.graph.nodes), list(gm2.graph.nodes)
    plan = plan_dpu(0, nodes1, nodes2, kv_specs, hw)
    # KV 单张量 = max_seq × head_dim × dtype = 8 × 4 × 4 = 128 B。
    assert plan.peak_allocation_bytes >= 128
    assert plan.peak_allocation_bytes < plan.total + 1
    with_kernel = plan_dpu(
        0, nodes1, nodes2, kv_specs, hw, kernel_mram_bytes=4096)
    assert with_kernel.peak_allocation_bytes == 4096


def test_format_mem_plan_printable() -> None:
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=1024)
    plans, _, _ = _plan_both_dpus(hw)
    text = format_mem_plan(plans, hw)
    assert "dpu0" in text and "dpu1" in text and "margin=" in text


def test_shared_redistribute_edge_merges_landing_lifetime() -> None:
    """同一条重分布边挂在多个消费者上时，落地缓冲只算一份。

    生命周期要并到最后一个读者、读者集合取全部；按消费者各记一份会
    让缓冲被提前复用，后面的消费者读到覆盖后的数据（tp2_pp2 分块实测）。
    """
    from contracts.pim_tensor_spec import (
        PIMTensorSpec, Placement, RedistributeEdge, TensorShardDetail)

    graph = Graph()
    src = graph.placeholder("src")
    c1 = graph.call_function(torch.ops.aten.add.Tensor, (src, src))
    c2 = graph.call_function(torch.ops.aten.add.Tensor, (c1, src))
    c3 = graph.call_function(torch.ops.aten.add.Tensor, (c1, src))
    nodes = list(graph.nodes)

    dst_spec = PIMTensorSpec(
        "dpu", Placement("Replicate", None), "transient", None,
        {0: TensorShardDetail(0, -1, 0, 8, (2, 4))}, None, dtype="float16")
    shared = RedistributeEdge(
        edge_id=7, src="src", dst="c1",
        from_placement=Placement("Shard", 0), to_placement=Placement("Replicate", None),
        src_spec=dst_spec, dst_spec=dst_spec, type="all_gather",
        src_loc={"device": "dpu", "dpus": [0]},
        dst_loc={"device": "dpu", "dpus": [0]},
        nbytes=16, dtype="float16")
    for n in (c1, c2, c3):
        n.meta[REDISTRIBUTE_META_KEY] = [shared]
    node_step, edge_step = _build_step_index(nodes)

    tensors = redistribute_landing_tensors(nodes, dpu_id=0,
                                           node_step=node_step,
                                           edge_step=edge_step)
    assert len(tensors) == 1, [t.name for t in tensors]
    (landing,) = tensors
    assert landing.name == "redist_dst:e7"
    # 生命周期并到最后一个消费者，读者是全部三个。
    assert landing.last_read_at == node_step[c3]
    assert set(landing.readers) == {c1.name, c2.name, c3.name}


# ---- 评审 round03：逐张量常驻台账与逐 TensorPU 聚合 ----

def test_plan_reports_per_tensor_resident_ledger() -> None:
    """计划必须给出逐笔常驻分配，而不是只给一个最大单块。"""
    from memory.mem_planner import ResidentAllocation

    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    (gm1, _), (gm2, _) = _two_appendix_a_graphs()
    kv_specs = _kv_specs()
    nodes1, nodes2 = list(gm1.graph.nodes), list(gm2.graph.nodes)
    plan = plan_dpu(0, nodes1, nodes2, kv_specs, hw,
                    kernel_mram_bytes=4096)
    weights = [item for item in plan.resident_allocations
               if item.tensor_key in plan.weight]
    assert weights
    assert all(isinstance(item, ResidentAllocation) for item in weights)
    assert all(item.model_bytes > 0 for item in weights)
    # 权重必须记录所属执行算子，不能把权重名当成放置单元。
    assert all(allocation.owner != allocation.tensor_key for allocation in weights)
    kv_blocks = [
        item for item in plan.resident_allocations
        if item.tensor_key.startswith("kv:")]
    # 单层单头：K/V 两份各 128 B，请求峰值合计等于完整 KV 口径。
    assert len(kv_blocks) == 2
    assert {item.tensor_key for item in kv_blocks} == {"kv:0:0:k", "kv:0:0:v"}
    assert sum(item.request_bytes_by_request[0] for item in kv_blocks) == (
        kv_bytes(kv_specs[0]))
    assert kv_bytes(kv_specs[0]) == 256
    kernel = next(
        item for item in plan.resident_allocations
        if item.tensor_key == "kernel")
    assert kernel.model_bytes == 4096


def test_tensor_pu_usage_dedupes_and_tracks_concurrency() -> None:
    """同 PU 同身份只算一次；并发请求增加时占用不得下降。"""
    from memory.mem_planner import ResidentAllocation, tensor_pu_usage

    duplicated = [
        ResidentAllocation("w", "w", 100),
        ResidentAllocation("w-alias", "w", 100),
    ]
    usage = tensor_pu_usage(duplicated, tensors_per_cluster=1)
    assert usage[0]["model_bytes"] == 100

    base = [ResidentAllocation("kv", "kv", 0, (10,))]
    more = base + [ResidentAllocation("kv2", "kv2", 0, (10, 20))]
    one_request = tensor_pu_usage(base, 1, concurrency=2)[0]["request_bytes"]
    two_requests = tensor_pu_usage(more, 1, concurrency=2)[0]["request_bytes"]
    assert two_requests >= one_request
    assert one_request == 10
    assert two_requests == 40


def test_tensor_pu_usage_rejects_invalid_cluster_size() -> None:
    from memory.mem_planner import tensor_pu_usage

    with pytest.raises(ValueError, match="tensors_per_cluster"):
        tensor_pu_usage([], tensors_per_cluster=0)


# ---- 评审 round04：执行算子归属、完整 KV 与两侧台账对账 ----

def _execution_order_graph():
    """weight 名字典序与算子执行顺序相反的两层图。"""
    from graph.strategy import ShardStrategy
    from graph.spec_prop import propagate_specs
    from torch.fx import Graph, GraphModule

    root = torch.nn.Module()
    root.register_buffer("zz_first", torch.empty(6, 4))
    root.register_buffer("aa_second", torch.empty(4, 6))
    graph = Graph()
    x = graph.placeholder("x")
    first_weight = graph.get_attr("zz_first")
    second_weight = graph.get_attr("aa_second")
    y1 = graph.call_function(torch.ops.aten.linear.default, (x, first_weight))
    y2 = graph.call_function(torch.ops.aten.linear.default, (y1, second_weight))
    graph.output(y2)
    gm = GraphModule(root, graph)
    x.meta["val"] = torch.empty(1, 4)
    first_weight.meta["val"] = root.zz_first
    second_weight.meta["val"] = root.aa_second
    y1.meta["val"] = torch.empty(1, 6)
    y2.meta["val"] = torch.empty(1, 4)
    partition_graph(gm)
    propagate_specs(gm, ShardStrategy(
        name="execution_order", num_dpus=1,
        weight_rules=(("zz_first", "col"), ("aa_second", "row"))))
    return gm, {"y1": y1, "y2": y2, "zz_first": first_weight,
                "aa_second": second_weight}


def test_weight_ledger_follows_execution_operator_not_name_order() -> None:
    """权重按消费它的执行算子分组，顺序是执行顺序而不是名称字典序。"""
    gm, nodes = _execution_order_graph()
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
    plan = plan_dpu(0, list(gm.graph.nodes), list(gm.graph.nodes),
                    {0: spec}, hw)
    weight_owners = [
        item.owner for item in plan.resident_allocations
        if item.tensor_key in ("zz_first", "aa_second")]
    # 执行顺序是 zz_first 所在算子 -> aa_second 所在算子；名称字典序相反。
    assert weight_owners == [nodes["y1"].name, nodes["y2"].name]
    assert weight_owners != ["aa_second", "zz_first"]


def test_kv_ledger_registers_every_layer_head_and_side() -> None:
    """多层多头 KV 必须按每层、每 head、K/V 两份逐块登记。"""
    gm, _ = _execution_order_graph()
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0, 1], kv_heads=[0, 1],
        q_heads_by_kv={0: [0], 1: [1]},
        max_seq=16, head_dim=8, dtype_bytes=2, kv_base=0)
    plan = plan_dpu(0, list(gm.graph.nodes), list(gm.graph.nodes),
                    {0: spec}, hw)
    blocks = [item for item in plan.resident_allocations
              if item.tensor_key.startswith("kv:")]
    assert len(blocks) == 2 * 2 * 2
    assert sum(item.request_bytes_by_request[0] for item in blocks) == (
        kv_bytes(spec))
    assert kv_bytes(spec) == 2 * 2 * 2 * 16 * 8 * 2


def test_tensor_pu_usage_groups_all_tensors_of_one_unit() -> None:
    """同一执行算子的多块权重不能被拆到不同 TensorPU。"""
    from memory.mem_planner import ResidentAllocation, tensor_pu_usage

    allocations = [
        ResidentAllocation("opA", "a0", 100),
        ResidentAllocation("opB", "b0", 40),
        ResidentAllocation("opA", "a1", 60),
    ]
    usage = tensor_pu_usage(allocations, tensors_per_cluster=2)
    assert usage[0]["model_bytes"] == 160
    assert usage[1]["model_bytes"] == 40


def test_tensor_pu_ledger_matches_genesim_capacity_usage() -> None:
    """同一批常驻张量在编译侧与 GeneSim 侧得到逐 PU 相同字节数。"""
    from unittest.mock import MagicMock

    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from autotune.placement import PlacementProblem
    from ir.model_ir import ModelIR, Operator
    from vpu.vpu import PIMVPU

    from memory.mem_planner import ResidentAllocation, tensor_pu_usage

    operators = [
        Operator(op_id=op_id, op_type="GEMM",
                 input_shapes=[(1, 16)], output_shapes=[(1, 16)])
        for op_id in (0, 1)
    ]
    ir = ModelIR(
        model_id="ledger-test", num_layers=1, num_heads=1, head_dim=1,
        hidden_size=1, operators=operators, dependencies=[],
        subgraphs=[[0, 1]])
    resources = [
        PIMVPU(vpu_id=index, node_id=0, backend=MagicMock(), pu_id=index,
               pim_device_id=0, cluster_id=0, tensor_id=index,
               device_local_pu_id=index)
        for index in (0, 1)
    ]
    problem = PlacementProblem(
        ir, resources, pim_supported_ops={"GEMM"},
        tensor_pu_capacity_bytes=1 << 20, cluster_capacity_bytes=1 << 21,
        resident_tensors_by_op={
            0: [("weight-0", "model", 100)],
            1: [("weight-1", "model", 200)],
        },
    )
    state = problem.state_from_vpu_ids({0: 0, 1: 1})
    genesim = problem.capacity_usage(state)["tensor_pus"]
    compiler = tensor_pu_usage(
        [ResidentAllocation("op0", "weight-0", 100),
         ResidentAllocation("op1", "weight-1", 200)],
        tensors_per_cluster=2)
    assert genesim == {0: 100, 1: 200}
    assert {slot: stats["model_bytes"] for slot, stats in compiler.items()} == (
        genesim)


# ---- 评审 round05：KV 按执行单元归属、逐请求峰值与两侧算法同源 ----

def test_kv_ledger_groups_each_heads_two_sides_under_one_unit() -> None:
    """同一执行单元（层 + KV head）的 K/V 两块必须同一 owner、同一 PU。"""
    gm, _ = _execution_order_graph()
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0, 1],
        q_heads_by_kv={0: [0], 1: [1]},
        max_seq=16, head_dim=8, dtype_bytes=2, kv_base=0)
    plan = plan_dpu(0, list(gm.graph.nodes), list(gm.graph.nodes),
                    {0: spec}, hw)
    kv = [item for item in plan.resident_allocations
          if item.tensor_key.startswith("kv:")]
    owners_by_head = {}
    for item in kv:
        head_key = item.tensor_key.rsplit(":", 1)[0]
        owners_by_head.setdefault(head_key, set()).add(item.owner)
    # 每 head 的 K/V 共用一个 owner；两个 head 是不同执行单元。
    assert all(len(owners) == 1 for owners in owners_by_head.values())
    assert len({next(iter(owners)) for owners in owners_by_head.values()}) == 2

    from memory.mem_planner import tensor_pu_usage

    usage = tensor_pu_usage(plan.resident_allocations, 2)
    # 两个 head 轮转到不同 PU；每 PU 的 K/V 两块合计 2 × 16 × 8 × 2 B。
    assert usage[0]["request_bytes"] == 512
    assert usage[1]["request_bytes"] == 512


def test_kv_request_peaks_follow_the_trace_and_concurrency() -> None:
    """KV 台账按轨迹逐请求登记峰值，并按并发上限取最大的若干个。"""
    gm, _ = _execution_order_graph()
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0],
        q_heads_by_kv={0: [0]},
        max_seq=32, head_dim=8, dtype_bytes=2, kv_base=0)
    plan = plan_dpu(0, list(gm.graph.nodes), list(gm.graph.nodes),
                    {0: spec}, hw, request_kv_tokens=(10, 20, 5))
    kv = [item for item in plan.resident_allocations
          if item.tensor_key.startswith("kv:")]
    # 每 token 16 B：逐请求峰值 = tokens × head_dim × dtype_bytes。
    assert {item.request_bytes_by_request for item in kv} == {
        (160, 320, 80)}

    from memory.mem_planner import tensor_pu_usage

    usage_one = tensor_pu_usage(
        plan.resident_allocations, 1, concurrency=1)
    usage_two = tensor_pu_usage(
        plan.resident_allocations, 1, concurrency=2)
    # 单 PU 上 K/V 两块，各取最大 1 或 2 个请求峰值后相加。
    assert usage_one[0]["request_bytes"] == 2 * 320
    assert usage_two[0]["request_bytes"] == 2 * (320 + 160)


def test_tensor_pu_usage_sums_request_peaks_by_index() -> None:
    """逐请求峰值按请求下标相加后再取 top-N，与 GeneSim 最终审计一致。"""
    from memory.mem_planner import ResidentAllocation, tensor_pu_usage

    allocations = [
        ResidentAllocation("a", "tensor-a", 0, (10, 0)),
        ResidentAllocation("b", "tensor-b", 0, (0, 10)),
    ]
    usage = tensor_pu_usage(allocations, 1, concurrency=1)
    # 请求 0 合计 10、请求 1 合计 10，top-1 是 10 而不是逐张量 top 再相加。
    assert usage[0]["request_bytes"] == 10
    usage_both = tensor_pu_usage(allocations, 1, concurrency=2)
    assert usage_both[0]["request_bytes"] == 20


def test_shared_kv_tensor_is_counted_once_per_tensor_pu() -> None:
    """GQA 多个消费者引用同一 KV 张量时，同一 PU 上只计一次。"""
    from memory.mem_planner import ResidentAllocation, tensor_pu_usage

    allocations = [
        ResidentAllocation("q-head-0", "layer.0.k_cache.0", 0, (100,)),
        ResidentAllocation("q-head-1", "layer.0.k_cache.0", 0, (100,)),
    ]
    usage = tensor_pu_usage(allocations, 1, concurrency=1)
    assert usage[0]["request_bytes"] == 100


@pytest.mark.parametrize("concurrency", [1, 2])
def test_kv_ledger_matches_genesim_capacity_audit(concurrency) -> None:
    """同一批 KV 张量在编译侧与 GeneSim 最终容量审计得到相同字节数。"""
    from types import SimpleNamespace

    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from ir.model_ir import Dependency, HOST_OP_ID, ModelIR, Operator
    from ir.trace_ir import Request
    from pim.capacity import tensor_pu_capacity_metrics

    from memory.mem_planner import tensor_pu_usage

    ir = ModelIR(
        model_id="kv-peak", num_layers=1, num_heads=1, head_dim=1,
        hidden_size=1,
        operators=[Operator(0, "GEMV_SCORE", [], [])],
        dependencies=[
            Dependency(
                HOST_OP_ID, 0,
                resident_bytes_coeffs={"Tp+Tq": 16.0},
                first_use_scope="request",
                tensor_id="layer.0.k_cache.0"),
            Dependency(
                HOST_OP_ID, 0,
                resident_bytes_coeffs={"Tp+Tq": 16.0},
                first_use_scope="request",
                tensor_id="layer.0.v_cache.0"),
        ],
        subgraphs=[[0]],
    )
    ir.operators[0].vpu_id = 5
    ir.operators[0].node_id = 0
    ir.operators[0].attached_pu_id = 0
    requests = [
        Request(request_id="r0", input_tokens=[], seq_len=10,
                output_length=3, arrival_time_ns=0, is_prefill=False),
        Request(request_id="r1", input_tokens=[], seq_len=5,
                output_length=2, arrival_time_ns=0, is_prefill=False),
    ]
    vpu = SimpleNamespace(
        vpu_type="PIM", vpu_id=5, node_id=0, pim_device_id=0,
        cluster_id=0, tensor_id=0)
    metrics = tensor_pu_capacity_metrics(
        ir, requests, {5: vpu}, limit_bytes=1 << 20,
        tensors_per_cluster=1, max_concurrent_requests=concurrency)
    genesim_bytes = metrics["tensor_pus"]["5"]["request_bytes"]

    gm, _ = _execution_order_graph()
    hw = HwBudget(mram_bytes=1 << 20, align=64, sys_reserve_bytes=0)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=13, head_dim=8, dtype_bytes=2, kv_base=0)
    plan = plan_dpu(0, list(gm.graph.nodes), list(gm.graph.nodes),
                    {0: spec}, hw, request_kv_tokens=(13, 7))
    usage = tensor_pu_usage(
        plan.resident_allocations, 1, concurrency=concurrency)[0]
    assert genesim_bytes == (416 if concurrency == 1 else 640)
    assert usage["request_bytes"] == genesim_bytes


def test_kv_owner_uses_the_real_attention_execution_unit() -> None:
    """有 attention 节点时，KV 的 owner 取自真实执行节点与 KV head。"""
    import torch
    from torch.fx import Graph

    from memory.mem_planner import _attention_unit_map

    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr(
        "model.model.layers.0.self_attn.q_proj.weight")
    projected = graph.call_function(
        torch.ops.aten.linear.default, (x, weight))
    sdpa = graph.call_function(
        torch.ops.aten.scaled_dot_product_attention.default,
        (projected, projected, projected))
    units = _attention_unit_map(list(graph.nodes))
    assert units == {0: sdpa.name}
