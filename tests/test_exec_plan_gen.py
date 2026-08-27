"""问题 6 编译期半单测：`build_execution_plan` 的依赖算法 + 端到端数值验证。

判据（CLAUDE.md 运行时模块判据）：结构判据对照方案表格（RAW/WAR 依赖、四类
redistribute 命令序列）；数值判据是编排器执行完整 `ExecutionPlan` 与单卡
PyTorch 逐元素对齐——借附录 A 两层 Linear + LayerNorm 的最小图，在
NumpyBackend 上真正跑一遍 `submit`/`wait`，而不是像问题 2/3 测试那样手动
摆数据验证单个算子。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from comm.plan import build_comm_plan
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from memory.kv_layout import KVRegionSpec
from memory.mem_planner import HwBudget, plan_dpu
import runtime.exec_plan_gen as exec_plan_gen_module
from runtime.exec_plan_gen import build_execution_plan, overlap
from runtime.kernels import register_all
from contracts.exec_plan import Access
from tests.test_spec_prop import _appendix_a_config, _appendix_a_graph


def _built_appendix_a():
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)
    edges = propagate_specs(gm, _appendix_a_config())
    return gm, nodes, edges


def _plan_and_compile(gm, edges, num_tasklets=4):
    """跑一遍问题 8（拿两台 dpu 的 pending_readers 合并）+ 问题 3 + 问题 6。"""
    kv_specs = {
        d: KVRegionSpec(dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
                         max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
        for d in (0, 1)
    }
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, nodes, nodes, kv_specs, hw) for d in (0, 1)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)
    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}
    compiled = build_execution_plan(nodes, gm, entries_by_id, pending, num_tasklets=num_tasklets)
    return compiled, plans


# ---------------------------------------------------------------------------
# 结构判据
# ---------------------------------------------------------------------------


def test_overlap_detects_intersecting_and_disjoint_ranges() -> None:
    a = Access(("dpu", 0), 0, 16)
    b = Access(("dpu", 0), 8, 16)
    c = Access(("dpu", 0), 16, 16)
    d = Access(("dpu", 1), 0, 16)
    assert overlap(a, b)  # [0,16) 与 [8,24) 相交
    assert not overlap(a, c)  # [0,16) 与 [16,32) 恰好相邻，不相交
    assert not overlap(a, d)  # 不同 loc，恒不相交


def test_pending_readers_join_reader_cmds_so_war_deps_are_not_dropped() -> None:
    """WAR 依赖必须真的查得到命令编号——问题 8 的 `pending_readers` 给的是
    **读者节点名**，`reader_cmds` 必须按同一套名字建键，两张表才能 join 上。

    这条测试固化一个真实 bug：早先 `register_reader` 错按"被读张量名"建键，
    与 `pending_readers` 的"读者名"方向相反，导致**全部 WAR 依赖被静默丢弃**
    （实测查询命中率 11%），并发执行下就会出现"新命令覆盖旧值、旧值的读者
    还没读完"的数据损坏——症状是 decode 循环偶发 NaN、argmax 崩成 0，且同
    一份输入多次运行结果不一致，靠端到端数值断言只能偶尔抓到。这里改为直接
    断言两张表的 join 命中率，让回归立刻暴露而不依赖运气。
    """
    gm, nodes, edges = _built_appendix_a()
    kv_specs = {
        d: KVRegionSpec(dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
                         max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
        for d in (0, 1)
    }
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    graph_nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, graph_nodes, graph_nodes, kv_specs, hw) for d in (0, 1)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)
    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}

    assert pending, "附录 A 图上应当存在被复用的地址，否则这条测试没有覆盖到东西"

    # 直接探测真正的 join：把 `_war_waits` 换成同逻辑但带统计的版本，数清
    # "查到命令编号"与"查不到"的次数。查不到又不属于前向引用（读者排在这次
    # 写之后、还没发射，本就不需要等）的，就是被丢弃的真实 WAR 依赖。
    graph_order = {n.name: i for i, n in enumerate(graph_nodes)}
    stats = {"resolved": 0, "dropped": 0}

    def counting_war_waits(loc, offset, pending_readers, reader_cmds):
        out: list[int] = []
        for reader in pending_readers.get((loc, offset), []):
            cids = reader_cmds.get((reader, loc[1]), [])
            if cids:
                stats["resolved"] += 1
                out.extend(cids)
            elif reader in graph_order or reader.startswith("redist:"):
                stats["dropped"] += 1
        return out

    original = exec_plan_gen_module._war_waits
    exec_plan_gen_module._war_waits = counting_war_waits
    try:
        build_execution_plan(graph_nodes, gm, entries_by_id, pending)
    finally:
        exec_plan_gen_module._war_waits = original

    assert stats["resolved"] + stats["dropped"] > 0, "没有触发任何 WAR 查询，测试无效"
    # 修复前这里 resolved 恒为 0（两张表方向相反）；修复后必须有真实命中。
    assert stats["resolved"] > 0, (
        f"全部 {stats['dropped']} 次 WAR 查询都没命中——pending_readers 与 "
        f"reader_cmds 的键对不上，WAR 依赖被静默丢弃"
    )


def test_scatter_edge_emits_host_slice_then_dma_out_only() -> None:
    """scatter（x -> linear_default 的输入）：无收集段，一条 host_slice + 回写。"""
    gm, nodes, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)
    cmds = compiled.plan.commands
    host_slice = [c for c in cmds if c.op == "host_slice"]
    assert len(host_slice) == 1
    assert host_slice[0].waits == []  # scatter 源在 host，无 dpu 收集依赖
    # 本实现把该边的收集/回写并入同一条粗粒度命令（设计决策 2），没有独立的
    # dma_in/dma_out——消费该 scatter 结果的两台 launch 直接等 host_slice。
    assert all(c.op != "dma_in" and c.op != "dma_out" for c in cmds)


def test_all_reduce_edge_waits_for_both_producer_launches() -> None:
    """all_reduce（linear_default_1 的 Partial 输出 -> norm）：等两台 launch 完成。"""
    gm, nodes, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)
    cmds = compiled.plan.commands
    (host_reduce,) = [c for c in cmds if c.op == "host_reduce"]
    launches = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default_1"]
    assert len(launches) == 2
    assert set(host_reduce.waits) >= {c.id for c in launches}


def test_dpu_node_with_redistributed_input_waits_for_its_landing_buffer() -> None:
    """linear_default 的输入 x 经 scatter 落地：launch 必须等 host_slice 完成。"""
    gm, nodes, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)
    cmds = compiled.plan.commands
    (host_slice,) = [c for c in cmds if c.op == "host_slice"]
    launches = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default"]
    assert len(launches) == 2
    for launch in launches:
        assert host_slice.id in launch.waits
        # reads 里必须含落地缓冲区间（长度与 x 的本地分片一致：1x4 fp32=16B）
        assert any(a.length == 16 for a in launch.reads)


def test_raw_dependency_between_consecutive_local_launches() -> None:
    """同 DPU 连续 launch（linear_default -> linear_default_1）天然 RAW 依赖。"""
    gm, nodes, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)
    cmds = compiled.plan.commands
    l1 = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default" and c.dpu_id == 0][0]
    l2 = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default_1" and c.dpu_id == 0][0]
    assert l1.id in l2.waits


# ---------------------------------------------------------------------------
# 端到端数值：真正在 NumpyBackend 上 submit/wait 跑完整个 ExecutionPlan
# ---------------------------------------------------------------------------


def test_execute_plan_matches_torch_reference() -> None:
    """附录 A 完整图（scatter 输入 -> 两层 Linear -> all_reduce -> LayerNorm）
    在 NumpyBackend 上跑一遍 ExecutionPlan，权重真实写入问题 8 规划的 MRAM
    地址、用问题 6 阶段 B 的真实 `runtime.kernels` 计算，与单卡 torch 参考
    逐元素对齐——不是手写替身 kernel，是真正要跑的那套代码。
    """
    gm, nodes_by_key, _ = _built_appendix_a()
    # layer_norm 是 host 算子，第 1 阶段 host_op 直接调 node.target；把它换成能
    # 接 numpy 输入的包装（真机场景下这层转换由 SDPA 之外的通用 host handler
    # 负责，这里手工模拟，不影响依赖算法/kernel 本身的验证）。
    for n in gm.graph.nodes:
        if "layer_norm" in str(n.target):
            n.target = lambda x, *a, orig=n.target, **k: orig(torch.from_numpy(np.asarray(x)), *a, **k).numpy()

    edges = propagate_specs(gm, _appendix_a_config())
    compiled, plans = _plan_and_compile(gm, edges)

    torch.manual_seed(0)
    gm.w1.copy_(torch.randn(6, 4))
    gm.w2.copy_(torch.randn(4, 6))
    w1n, w2n = gm.w1.detach().numpy(), gm.w2.detach().numpy()

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
    register_all(backend)

    # 权重真实搬进各 DPU 的 MRAM（对应方案"运行时层：照蓝图把权重一次性搬进
    # 去"），地址取问题 8 plan_dpu 回填的 offset，不是随便挑的。
    for dpu_id in (0, 1):
        w1_off = plans[dpu_id].weight[nodes_by_key["w1"].target]
        w2_off = plans[dpu_id].weight[nodes_by_key["w2"].target]
        backend.write_local(dpu_id, w1_off, w1n[dpu_id * 3:(dpu_id + 1) * 3].astype(np.float32))
        backend.write_local(dpu_id, w2_off, w2n[:, dpu_id * 3:(dpu_id + 1) * 3].astype(np.float32))

    x_val = torch.randn(1, 4)
    backend.bind_inputs({"x": x_val.numpy()})
    events = {}
    for cmd in compiled.plan.commands:
        for w in cmd.waits:
            backend.wait(events[w])
        events[cmd.id] = backend.submit(cmd)
    result = backend.wait(events[compiled.output_cmd_id])

    ref = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(torch.nn.functional.linear(x_val, gm.w1), gm.w2),
        [4], None, None, 1e-5,
    ).detach().numpy()
    assert np.allclose(result, ref, atol=1e-4)


@pytest.mark.parametrize("num_tasklets", [1, 4])
def test_build_execution_plan_stamps_num_tasklets_on_every_launch(num_tasklets) -> None:
    """`build_execution_plan(..., num_tasklets=N)` 要落到每一条 launch 命令上——
    覆盖默认值 4（主线路径）和显式 1（退化情形回归，泛化后单 tasklet 行为不变）。
    非 launch 命令（host_op 等）不受影响，仍是 Command 的默认值。
    """
    gm, _, edges = _built_appendix_a()
    for n in gm.graph.nodes:
        if "layer_norm" in str(n.target):
            n.target = lambda x, *a, orig=n.target, **k: orig(torch.from_numpy(np.asarray(x)), *a, **k).numpy()
    compiled, _ = _plan_and_compile(gm, edges, num_tasklets=num_tasklets)
    launch_cmds = [c for c in compiled.plan.commands if c.op == "launch"]
    assert launch_cmds, "附录 A 图应该至少有一条 launch 命令（两层 Linear）"
    assert all(c.num_tasklets == num_tasklets for c in launch_cmds)
