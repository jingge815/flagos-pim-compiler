"""验证执行计划的依赖关系和端到端数值。"""

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
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import ShardStrategy
from tests.test_spec_prop import _appendix_a_config, _appendix_a_graph


def _built_appendix_a():
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)
    edges = propagate_specs(gm, _appendix_a_config())
    return gm, nodes, edges


def _hardware(num_tasklets=4):
    return PIMHardwareConfig(
        num_dpus=2,
        num_tasklets=num_tasklets,
        mram_bytes_per_dpu=1 << 16,
        wram_bytes_per_dpu=65536,
        dma_align=64,
    )


def _plan_and_compile(gm, edges, num_tasklets=4):
    """生成内存、通信和执行计划。"""
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
    compiled = build_execution_plan(
        nodes, gm, entries_by_id, pending, hardware=_hardware(num_tasklets),
        num_tasklets=num_tasklets,
    )
    return compiled, plans


# 结构验证。


def test_overlap_detects_intersecting_and_disjoint_ranges() -> None:
    a = Access(("dpu", 0), 0, 16)
    b = Access(("dpu", 0), 8, 16)
    c = Access(("dpu", 0), 16, 16)
    d = Access(("dpu", 1), 0, 16)
    assert overlap(a, b)  # [0,16) 与 [8,24) 相交
    assert not overlap(a, c)  # [0,16) 与 [16,32) 恰好相邻，不相交
    assert not overlap(a, d)  # 不同 loc，恒不相交


def test_pending_readers_join_reader_cmds_so_war_deps_are_not_dropped() -> None:
    """验证 pending_readers 能正确转换为 WAR 等待命令。"""
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

    # 统计 WAR 依赖的命中情况。
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
        build_execution_plan(graph_nodes, gm, entries_by_id, pending, hardware=_hardware())
    finally:
        exec_plan_gen_module._war_waits = original

    assert stats["resolved"] + stats["dropped"] > 0, "没有触发任何 WAR 查询，测试无效"
    # 必须存在已解析的 WAR 依赖。
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
    # 收集和回写由同一条主机命令完成。
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
        # reads 包含落地缓冲区间。
        assert any(a.length == 16 for a in launch.reads)


def test_raw_dependency_between_consecutive_local_launches() -> None:
    """同 DPU 连续 launch（linear_default -> linear_default_1）天然 RAW 依赖。"""
    gm, nodes, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)
    cmds = compiled.plan.commands
    l1 = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default" and c.dpu_id == 0][0]
    l2 = [c for c in cmds if c.op == "launch" and c.payload["node"] == "linear_default_1" and c.dpu_id == 0][0]
    assert l1.id in l2.waits


# 在 NumpyBackend 上执行完整计划并校验数值。


def test_execute_plan_matches_torch_reference() -> None:
    """执行完整计划并将结果与单卡 Torch 参考比较。"""
    gm, nodes_by_key, _ = _built_appendix_a()
    # 为 host layer_norm 提供 NumPy 包装。
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

    # 将权重写入各 DPU 的 MRAM 规划地址。
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
    覆盖默认值 4 和显式值 1。非 launch 命令保持 Command 的默认值。
    """
    gm, _, edges = _built_appendix_a()
    for n in gm.graph.nodes:
        if "layer_norm" in str(n.target):
            n.target = lambda x, *a, orig=n.target, **k: orig(torch.from_numpy(np.asarray(x)), *a, **k).numpy()
    compiled, _ = _plan_and_compile(gm, edges, num_tasklets=num_tasklets)
    launch_cmds = [c for c in compiled.plan.commands if c.op == "launch"]
    assert launch_cmds, "附录 A 图应该至少有一条 launch 命令（两层 Linear）"
    assert all(c.num_tasklets == num_tasklets for c in launch_cmds)


def test_build_execution_plan_requires_explicit_hardware() -> None:
    gm, _, edges = _built_appendix_a()
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

    with pytest.raises(TypeError, match="hardware"):
        build_execution_plan(graph_nodes, gm, entries_by_id, pending)


def _appendix_a_planned() -> tuple[list, object, dict, dict]:
    """生成执行计划所需的图、通信条目和读者信息。"""
    gm, _, edges = _built_appendix_a()
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
    return graph_nodes, gm, {e.edge_id: e for e in build_comm_plan(edges)}, pending


def test_build_execution_plan_accepts_shard_count_below_num_dpus() -> None:
    """验证张量分片数少于硬件 DPU 总数时仍可生成执行计划。"""
    graph_nodes, gm, entries_by_id, pending = _appendix_a_planned()

    compiled = build_execution_plan(
        graph_nodes, gm, entries_by_id, pending,
        hardware=PIMHardwareConfig(
            num_dpus=4, num_tasklets=4, mram_bytes_per_dpu=1 << 16,
            wram_bytes_per_dpu=65536, dma_align=64,
        ),
    )
    assert [c for c in compiled.plan.commands if c.op == "launch"]


def test_build_execution_plan_rejects_out_of_range_dpu_id() -> None:
    """验证执行计划拒绝超出硬件地址空间的 DPU 编号。"""
    gm, _ = _appendix_a_graph()
    partition_graph(gm)
    edges = propagate_specs(
        gm,
        ShardStrategy(name="oob", num_dpus=2, dpu_ids=(2, 5),
                      weight_rules=(("w1", "col"), ("w2", "row"))),
    )
    kv_specs = {
        d: KVRegionSpec(dpu_id=d, layers=[0], kv_heads=[i], q_heads_by_kv={i: [i]},
                         max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
        for i, d in enumerate((2, 5))
    }
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    graph_nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, graph_nodes, graph_nodes, kv_specs, hw) for d in (2, 5)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)
    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}

    with pytest.raises(ValueError, match="越界 dpu_id"):
        build_execution_plan(
            graph_nodes, gm, entries_by_id, pending,
            hardware=PIMHardwareConfig(
                num_dpus=4, num_tasklets=4, mram_bytes_per_dpu=1 << 16,
                wram_bytes_per_dpu=65536, dma_align=64,
            ),
        )


def test_build_execution_plan_stamps_hardware_payload_on_every_launch() -> None:
    gm, _, edges = _built_appendix_a()
    hardware = _hardware(num_tasklets=4)

    compiled, _ = _plan_and_compile(gm, edges, num_tasklets=hardware.num_tasklets)

    launch_cmds = [c for c in compiled.plan.commands if c.op == "launch"]
    assert launch_cmds, "附录 A 图应该至少有一条 launch 命令（两层 Linear）"
    assert all(c.payload["hardware"] == hardware.to_payload() for c in launch_cmds)


def test_build_execution_plan_rejects_tasklet_mismatch() -> None:
    gm, _, edges = _built_appendix_a()
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

    with pytest.raises(ValueError, match="num_tasklets"):
        build_execution_plan(
            graph_nodes, gm, entries_by_id, pending, hardware=_hardware(num_tasklets=4),
            num_tasklets=2,
        )
