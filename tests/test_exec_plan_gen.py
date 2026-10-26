"""验证执行计划的依赖关系和端到端数值。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.fx import Graph, GraphModule

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


def _plan_and_compile(gm, edges, num_tasklets=4, mram_bytes=1 << 16):
    """生成内存、通信和执行计划。"""
    kv_specs = {
        d: KVRegionSpec(dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
                         max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
        for d in (0, 1)
    }
    hw = HwBudget(mram_bytes=mram_bytes, align=64, sys_reserve_bytes=0)
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


# ---- 「同形 ⇒ 同切分」这条外推（评审 round2 问题 4 第 2 条）----
#
# 发射侧 `driver._attach_layout` 把结果类型的编码一并贴到**与结果同形**的
# 操作数上（不贴的话 `-pim-expand-phases` 重建结果类型会把编码整个丢掉）。
# 代价是它默认「同形状 ⇒ 同切分」，而切分决策只取自输出的 shard_map。
# 这条外推在计划生成期钉住，不成立就抛错，不静默写一个与事实不符的
# dpusPerDevice。

def test_the_returned_width_is_not_a_copy_of_the_graph_dtype() -> None:
    """与图上 dtype 相同的回传宽度不得被收集。

    探测按图上的 dtype 去编译，回传的 `placed_elem_bytes` 必然等于
    `dtype_bytes(spec.dtype)`，用它替换同一个值，产物一个字节都不变。
    所以真实计划上不该收集到任何宽度；只有与图上不同的才算信息。
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from contracts.dtypes import dtype_bytes
    from contracts.graph_meta import SPEC_META_KEY
    from contracts.op_contract import PIMHardwareConfig
    from graph.partition import partition_graph
    from graph.spec_prop import propagate_specs
    from graph.strategy import llama_strategy
    from runtime.compile import peak_kernel_mram_bytes
    from tests.test_partition import _FixedMaskLlama

    seq = 16
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=320, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=8, num_kv_heads=8,
        max_position_embeddings=seq, bos_token_id=1, eos_token_id=2,
        pad_token_id=0)).eval()
    ids = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((1, 1, seq, seq))
    mask.masked_fill_(blocked, torch.finfo(torch.float32).min)
    gm = torch.export.export(_FixedMaskLlama(model), (ids, mask),
                             strict=True).module()
    partition_graph(gm)
    propagate_specs(gm, llama_strategy(
        2, num_stages=1, num_heads=8, num_kv_heads=8,
        intermediate_size=176, vocab_size=320, num_layers=1))
    hw = PIMHardwareConfig(num_dpus=2, num_tasklets=4, mram_bytes_per_dpu=1 << 32,
                           wram_bytes_per_dpu=65536, dma_align=8)
    widths = {}
    peak_kernel_mram_bytes(list(gm.graph.nodes), hardware=hw, widths=widths)
    from gml_bridge.from_fx import _op_type_of

    for name, back in widths.items():
        matched = [n for n in gm.graph.nodes
                   if getattr(n.meta.get(SPEC_META_KEY), "dtype", "")
                   and _op_type_of(n) == name]
        assert matched and any(
            dtype_bytes(n.meta[SPEC_META_KEY].dtype) != back.placed_elem_bytes
            for n in matched), (
            f"{name} 的回传宽度 {back.placed_elem_bytes} 与图上 dtype 相同，"
            f"不该被收集")


def test_the_returned_element_width_sizes_the_access() -> None:
    """回传的元素宽度要改变 numpy 执行计划里的访问字节数。

    同一张图，回传说 Gemm 是 1 字节时，它的写访问必须比不回传时短。
    不变就说明回传没进执行计划。
    """
    from contracts.ir_payloads import PlacementBack

    gm, _, edges = _built_appendix_a()
    plain, _ = _plan_and_compile(gm, edges)
    gm2, _, edges2 = _built_appendix_a()
    nodes = list(gm2.graph.nodes)
    from memory.kv_layout import KVRegionSpec
    from memory.mem_planner import HwBudget, plan_dpu
    kv = {d: KVRegionSpec(dpu_id=d, layers=[0], kv_heads=[d], q_heads_by_kv={d: [d]},
                          max_seq=8, head_dim=4, dtype_bytes=4, kv_base=0)
          for d in (0, 1)}
    hw = HwBudget(mram_bytes=1 << 16, align=64, sys_reserve_bytes=0)
    plans = {d: plan_dpu(d, nodes, nodes, kv, hw) for d in (0, 1)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)
    from comm.plan import build_comm_plan
    from runtime.exec_plan_gen import build_execution_plan
    narrowed = build_execution_plan(
        nodes, gm2, {e.edge_id: e for e in build_comm_plan(edges2)}, pending,
        hardware=_hardware(), num_tasklets=4,
        placed_elem_bytes={"Gemm": PlacementBack(placed_elem_bytes=1)})

    def write_len(compiled):
        return next(c.writes[0].length for c in compiled.plan.commands
                    if c.op == "launch" and c.writes)

    assert write_len(narrowed) < write_len(plain), (
        write_len(narrowed), write_len(plain))


def _sharded_spec():
    from contracts.pim_tensor_spec import (
        PIMTensorSpec,
        Placement,
        TensorShardDetail,
    )

    details = {
        0: TensorShardDetail(dpu_id=0, shard_dim=1, start_idx=0, end_idx=4,
                             local_shape=(1, 4)),
        1: TensorShardDetail(dpu_id=1, shard_dim=1, start_idx=4, end_idx=8,
                             local_shape=(1, 4)),
    }
    spec = PIMTensorSpec(device="dpu", placement=Placement("Shard", dim=1),
                         residency="transient", pinned_dpu_id=None,
                         shard_map=details, reduce_type=None, dtype="float16")
    return spec, details[0]


class _FakeNode:
    """只需要 `target` 与 `name` 两样。"""

    def __init__(self, target: str, name: str = "n0") -> None:
        self.target = target
        self.name = name


def test_a_same_shape_operand_with_a_different_decision_is_rejected() -> None:
    """B 路上同形但切分方式不同的实参必须抛错，而不是被贴上输出的决策。"""
    from runtime.exec_plan_gen import _assert_same_shape_args_share_the_decision

    spec, out_detail = _sharded_spec()
    args = [("replicated", (1, 4), None)]      # 同形，但没有切分决策
    with pytest.raises(ValueError, match="与输出同形"):
        _assert_same_shape_args_share_the_decision(
            _FakeNode("aten.cat.default"), 0, args, spec, out_detail)


@pytest.mark.parametrize("target", [
    "aten.mul.Tensor", "aten.add.Tensor", "aten.sub.Tensor", "aten.div.Tensor",
    "aten.silu.default",
])
def test_the_elementwise_and_silu_paths_are_guarded_too(target) -> None:
    """逐元素与 silu 也走 B 路贴编码，守卫不能把它们当成 A 路放过。

    它们的 GML 名（EltwiseMul / Silu）挂在 `has_kernel=False` 的登记行上，
    但运行时走的是 `eltwise` / `lut` 内核入口 —— `_attach_layout` 照样把编码
    贴到同形实参上。真实 tp2 计划里带切分的 `mul` 与 `silu` 各有若干条，
    守卫在那里必须真的生效。
    """
    from runtime.exec_plan_gen import _assert_same_shape_args_share_the_decision

    spec, out_detail = _sharded_spec()
    args = [("replicated", (1, 4), None)]
    with pytest.raises(ValueError, match="与输出同形"):
        _assert_same_shape_args_share_the_decision(
            _FakeNode(target), 0, args, spec, out_detail)


def test_the_a_path_is_out_of_scope_for_that_assertion() -> None:
    """A 路（`linear`）的同形不同切分放过，不抛错。

    豁免的理由是 A 路那些张量编码**当前没有按轴取值的消费者**：读编码的只有
    FlagTree 的 `verifyAgreesWith`（只比切分宽度）与 `TritonSplitOpPattern`，
    资源算术走的是模块属性。注意不是「实参不会被贴上编码」—— FlagTree 的
    TypeConverter 会把模块级 placement 贴到该 kernel 所有没有编码的张量上，
    实测连索引张量都带着。

    这里不写「实测 N 处同形不同切分」那类数字：原先注释里的 60 处在同一路径上
    复算（tiny llama 的 tp2 / tp4 / tp2-GQA）得到的是 **0 处**，同形实参对分别
    86 / 172 / 86 个而切分全部相同。所以这条用例守的是「A 路不抛」这个行为本身，
    而不是某个不可复核的计数。
    """
    from runtime.exec_plan_gen import _assert_same_shape_args_share_the_decision

    spec, out_detail = _sharded_spec()
    args = [("replicated", (1, 4), None)]
    # 不抛 —— 这就是判据。
    _assert_same_shape_args_share_the_decision(
        _FakeNode("aten.linear.default"), 0, args, spec, out_detail)


def test_a_matching_operand_decision_passes() -> None:
    """同形且同切分：正常放过。"""
    from runtime.exec_plan_gen import _assert_same_shape_args_share_the_decision

    spec, out_detail = _sharded_spec()
    # 决策形态是 `DpuShard.to_payload()`：[dim, num_dpus, kind, reduce]。
    args = [("also_sharded", (1, 4), [1, 2, "shard", None])]
    _assert_same_shape_args_share_the_decision(
        _FakeNode("aten.cat.default"), 0, args, spec, out_detail)


# ---- 排布字段（Memory Layout 第 3 层）随命令下发 ----
#
# 统一 IR 里 `TensorShardDetail.elem_strides` 是排布的唯一载体；下发链上它必须
# 与切分决策取同一个分片、走同一条命令，否则算出的 `order` 会描述另一块张量。

def test_the_layout_field_travels_with_the_shard_decision() -> None:
    """带切分决策的命令必须同时带排布字段，且取自同一个分片。"""
    from contracts.graph_meta import SPEC_META_KEY

    gm, _, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)

    nodes_by_name = {n.name: n for n in gm.graph.nodes}
    seen = 0
    for cmd in compiled.plan.commands:
        if cmd.payload.get("shard") is None:
            continue
        spec = nodes_by_name[cmd.payload["node"]].meta[SPEC_META_KEY]
        detail = spec.shard_map[cmd.dpu_id]
        assert cmd.payload["elem_strides"] == list(detail.elem_strides), (
            f"{cmd.payload['node']} 的排布字段与切分决策不同源："
            f"{cmd.payload.get('elem_strides')} vs {detail.elem_strides}")
        seen += 1
    assert seen, "这张图上没有带切分的命令，用例失去意义"


def test_every_dpu_tensor_command_carries_a_layout_field() -> None:
    """所有 DPU 张量命令都带排布字段，不只是带切分的那些。

    排布是 Memory Layout 维自己的内容：不依赖跨 DPU 决策，单 DPU 上也该有
    （切分决策才需要「单 DPU 一个字不下发」那条纪律）。
    """
    from contracts.graph_meta import SPEC_META_KEY

    gm, _, edges = _built_appendix_a()
    compiled, _ = _plan_and_compile(gm, edges)

    nodes_by_name = {n.name: n for n in gm.graph.nodes}
    seen = 0
    for cmd in compiled.plan.commands:
        node_name = cmd.payload.get("node")
        if node_name is None or "arg_shapes" not in cmd.payload:
            continue
        spec = nodes_by_name[node_name].meta.get(SPEC_META_KEY)
        if spec is None or cmd.dpu_id not in (spec.shard_map or {}):
            continue
        detail = spec.shard_map[cmd.dpu_id]
        assert cmd.payload["elem_strides"] == list(detail.elem_strides), (
            f"{node_name} 的命令没带排布字段")
        seen += 1
    assert seen, "这张图上没有 DPU 张量命令，用例失去意义"


def test_tiled_col_sharded_linear_matches_torch() -> None:
    """按列切的权重被分块沿行切开后，结果要与整块 linear 逐元素相等。

    权重按列切成连续的两半，每台持有自己那一半行。分块再沿行把权重切成
    几段，每段的切片下标是全局的。下发时每台都拿这个全局下标去切自己的
    本地权重，第二段在第一台上越界被夹成空，第一段在第二台上读到的是别人
    的行。结果的后半列因此与前半列重复，对不上整块 linear。
    """
    from torch.fx import Graph, GraphModule

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from graph.opt_passes import optimize_graph

    torch.manual_seed(0)
    # 12 行、两台各持 6 行，切出三段 [0:4)、[4:8)、[8:12)：第一段整段在
    # 第一台，中段跨分界，第三段整段在第二台，与 gate_proj 的真实形态一致。
    weight = torch.randn(12, 4)
    activation = torch.randn(1, 2, 4)
    reference = torch.nn.functional.linear(activation, weight)

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", weight.to(torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 2, 4, dtype=torch.float16)
    w.meta["val"] = torch.empty(12, 4, dtype=torch.float16)
    out.meta["val"] = torch.empty(1, 2, 12, dtype=torch.float16)
    partition_graph(gm)
    # 预算 64 时工作集按输入、权重、输出三块之和算，浮点按提升到
    # float32 后的宽度计。最小的合法切分是每块 2 行，12 行切成 6 段。
    # 再小一档没有任何切法压得进，分块不触发。
    optimize_graph(gm, wram_bytes=64, tp_width=2)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))
    edges = propagate_specs(gm, strategy)
    compiled, plans = _plan_and_compile(gm, edges, num_tasklets=1)

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
    register_all(backend)
    stored = weight.detach().numpy().astype(np.float16)
    for dpu_id in (0, 1):
        offset = plans[dpu_id].weight["gate_proj_weight"]
        backend.write_local(dpu_id, offset, stored[dpu_id * 6:(dpu_id + 1) * 6])
    # 每段 2 行。第一台持有前 6 行，本地下标是 [0:2)、[2:4)、[4:6)，
    # 各出现两次（两行激活各切一次）；第二台同样三段，从 0 起算。
    expected = []
    for start in (0, 2, 4):
        expected += [((start, start + 2), (2, 4))] * 2
    # 只比较切权重的切片。激活上也有切片，它的下标不换算成本地。
    weight_slices = {n.name for n in gm.graph.nodes
                     if n.target is torch.ops.aten.slice.Tensor
                     and n.args[0].op == "get_attr"}
    seen = {0: [], 1: []}
    for cmd in compiled.plan.commands:
        if cmd.payload.get("kernel") != "aten.slice.Tensor":
            continue
        if cmd.payload["node"] not in weight_slices:
            continue
        start, end = cmd.payload["arg_kinds"][2:4]
        seen[cmd.dpu_id].append(((start, end), tuple(cmd.payload["out_shape"])))
    for dpu_id in (0, 1):
        assert sorted(seen[dpu_id]) == sorted(expected), seen[dpu_id]

    backend.bind_inputs({"x": activation.numpy().astype(np.float16)})
    events = {}
    for cmd in compiled.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    result = np.array(backend.wait(events[compiled.output_cmd_id]))

    ref = reference.detach().numpy()
    assert np.allclose(result.astype(np.float32), ref, atol=1e-2)


def test_cat_joins_only_the_segments_of_its_own_row() -> None:
    """沿两根轴同时切时，每一行的拼接只收这一行的各段。

    行维也被切开后，同一段权重在每一行里各算一次，两次的持有者相同。
    拼接若按「这台持不持有」收输入，会把另一行的结果也拼进来，输出的
    列数比这一行的段数多，对不上整块 linear。
    """
    from torch.fx import Graph, GraphModule

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from graph.opt_passes import optimize_graph

    torch.manual_seed(0)
    # 12 行切成 6 段、激活的行维切成 2 行：同一段权重在每一行里各算一次。
    weight = torch.randn(12, 4)
    activation = torch.randn(1, 2, 4)
    reference = torch.nn.functional.linear(activation, weight)

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", weight.to(torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 2, 4, dtype=torch.float16)
    w.meta["val"] = torch.empty(12, 4, dtype=torch.float16)
    out.meta["val"] = torch.empty(1, 2, 12, dtype=torch.float16)
    partition_graph(gm)
    optimize_graph(gm, wram_bytes=32, tp_width=2)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))
    edges = propagate_specs(gm, strategy)
    compiled, plans = _plan_and_compile(gm, edges, num_tasklets=1)

    cats = {n.name: n for n in gm.graph.nodes
            if n.target is torch.ops.aten.cat.default}
    # 沿特征维拼的那一层，每台只收自己持有的那几段，不收另一行的。
    from contracts.graph_meta import SPEC_META_KEY
    for cmd in compiled.plan.commands:
        node = cats.get(cmd.payload.get("node"))
        if node is None or int(node.args[1]) != node.meta["val"].ndim - 1:
            continue
        held = sum(1 for seg in node.args[0]
                   if cmd.dpu_id in seg.meta[SPEC_META_KEY].shard_map)
        got = cmd.payload["arg_kinds"].count("tensor")
        assert got == held, (
            f"{cmd.payload['node']} 在 DPU{cmd.dpu_id} 上收了 {got} 段，"
            f"这台只持有 {held} 段")

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
    register_all(backend)
    stored = weight.detach().numpy().astype(np.float16)
    for dpu_id in (0, 1):
        offset = plans[dpu_id].weight["gate_proj_weight"]
        half = stored.shape[0] // 2
        backend.write_local(dpu_id, offset, stored[dpu_id * half:(dpu_id + 1) * half])
    backend.bind_inputs({"x": activation.numpy().astype(np.float16)})
    events = {}
    for cmd in compiled.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    result = np.array(backend.wait(events[compiled.output_cmd_id]))
    assert np.allclose(result.astype(np.float32), reference.detach().numpy(), atol=1e-2)


def test_weight_slice_on_a_non_shard_axis_keeps_global_indices() -> None:
    """切片轴不是权重的切分轴时，下标不能按其它轴的分片起点平移。

    按行切的权重第 1 维分成两半，每台都持有完整的 8 行。再沿第 0 维切出
    [0:4)，两台都持有这 4 行，下发的下标应保持全局的 0 和 4；按第 1 维的
    分片起点平移会把第二台切到不存在的行上。
    """
    from torch.fx import Graph, GraphModule

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("down_proj_weight")
    part = graph.call_function(torch.ops.aten.slice.Tensor, (w, 0, 0, 4))
    out = graph.call_function(torch.ops.aten.linear.default, (x, part))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("down_proj_weight", torch.empty(8, 4))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 2, 8)
    w.meta["val"] = torch.empty(8, 4)
    part.meta["val"] = torch.empty(4, 4)
    out.meta["val"] = torch.empty(1, 2, 4)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("down_proj_weight", "row"),))
    edges = propagate_specs(gm, strategy)
    compiled, _ = _plan_and_compile(gm, edges, num_tasklets=1)

    seen = {}
    for cmd in compiled.plan.commands:
        if cmd.payload.get("kernel") != "aten.slice.Tensor":
            continue
        seen[cmd.dpu_id] = (
            tuple(cmd.payload["arg_kinds"][2:4]),
            tuple(cmd.payload["out_shape"]),
        )
    assert seen == {0: ((0, 4), (4, 2)), 1: ((0, 4), (4, 2))}, seen


def test_later_writer_waits_for_earlier_writer_of_same_range() -> None:
    """后写同一区间的命令必须等前一个写者。

    decode 图里 host 广播（cmd 39）和 DPU 上的 `to_2`（cmd 33）写同一块激活
    缓冲。原先只记读后写和写后读，不记写后写：`to_2` 慢一步就把广播结果盖掉，
    张量并行下某台 DPU 的 K 行随机出错。
    """
    builder = exec_plan_gen_module._PlanBuilder(None, {})
    first = builder.append("launch", 3, {}, [], [Access(("dpu", 3), 100, 4)], [])
    other = builder.append("launch", 2, {}, [], [Access(("dpu", 2), 100, 4)], [])
    later = builder.append("host_slice", None, {}, [], [
        Access(("dpu", 3), 96, 64), Access(("dpu", 2), 200, 8)], [])
    assert first.id in later.waits
    assert other.id not in later.waits


def test_cat_of_partial_sums_reads_the_reduced_buffer() -> None:
    """行切 linear 的部分和喂给 cat 时，cat 要读归约后的副本，不能读各台的部分和。

    两段行切 linear 沿特征维拼接，是行切权重分块后沿 N 拼回的形态。
    """
    from graph.opt_passes import tile_graph
    from runtime.kernels import register_all as _register

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("down_proj_weight")
    lin = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(lin)
    mod = torch.nn.Module()
    torch.manual_seed(0)
    weight = torch.randn(8, 8)
    mod.register_buffer("down_proj_weight", weight.clone())
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 8)
    w.meta["val"] = weight.clone()
    lin.meta["val"] = torch.empty(1, 8)
    partition_graph(gm)
    strategy = ShardStrategy(name="tp2", num_dpus=2,
                             weight_rules=(("down_proj_weight", "row"),))
    # 只切 N：两段输出 (1, 4) 沿最后一维拼回。
    assert tile_graph(gm, wram_bytes=144, tp_width=2) == 1
    assert any(n.target is torch.ops.aten.cat.default for n in gm.graph.nodes)
    partition_graph(gm)
    edges = propagate_specs(gm, strategy)
    compiled, plans = _plan_and_compile(gm, edges)

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
    _register(backend)
    wn = weight.numpy()
    for dpu_id in (0, 1):
        off = plans[dpu_id].weight["down_proj_weight"]
        backend.write_local(dpu_id, off,
                            np.ascontiguousarray(wn[:, dpu_id * 4:(dpu_id + 1) * 4]))
    # 跑两遍：解码循环每步都重放同一份计划，权重必须在第一遍后保持原样。
    # 权重切片若被标成常驻，会写到权重自己的偏移上，第二遍读到的是被盖过的行。
    for step in range(2):
        x_val = torch.randn(1, 8)
        backend.bind_inputs({"x": x_val.numpy()})
        backend.reset_events()
        events = {}
        for cmd in compiled.plan.commands:
            for wait in cmd.waits:
                backend.wait(events[wait])
            events[cmd.id] = backend.submit(cmd)
        result = backend.wait(events[compiled.output_cmd_id])
        ref = torch.nn.functional.linear(x_val, weight).numpy()
        # 编译出的 concat 内核按 float16 搬数据，容差取 float16 的精度。
        assert np.allclose(result, ref, atol=1e-2, rtol=1e-3), (step, result, ref)


def _run_single_linear(weight_name: str, mode: str, x_shape, w_shape, budget: int):
    """一条 linear 按策略分块、传播、生成计划并在 NumpyBackend 上跑，返回 (结果, 参考)。"""
    from contracts.graph_meta import SPEC_META_KEY
    from graph.opt_passes import tile_graph
    from runtime.kernels import register_all as _register

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr(weight_name)
    lin = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(lin)
    mod = torch.nn.Module()
    torch.manual_seed(0)
    weight = torch.randn(*w_shape).to(torch.float16)
    mod.register_buffer(weight_name, weight.clone())
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(*x_shape, dtype=torch.float16)
    w.meta["val"] = weight.clone()
    lin.meta["val"] = torch.empty(*x_shape[:-1], w_shape[0], dtype=torch.float16)
    partition_graph(gm)
    strategy = ShardStrategy(name="tp2", num_dpus=2,
                             weight_rules=((weight_name, mode),))
    tiled = tile_graph(gm, wram_bytes=budget, tp_width=2)
    partition_graph(gm)
    edges = propagate_specs(gm, strategy)
    # 分块后子块很多，中间结果放不进 64KB，计划与后端都放宽到 1MB。
    compiled, plans = _plan_and_compile(gm, edges, mram_bytes=1 << 20)

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 20))
    _register(backend)
    wn = weight.numpy()
    spec = gm.graph.find_nodes(op="get_attr")[0].meta[SPEC_META_KEY]
    for dpu_id, det in spec.shard_map.items():
        index = [slice(None)] * wn.ndim
        index[det.shard_dim] = slice(det.start_idx, det.end_idx)
        backend.write_local(dpu_id, plans[dpu_id].weight[weight_name],
                            np.ascontiguousarray(wn[tuple(index)]))
    x_val = torch.randn(*x_shape).to(torch.float16)
    backend.bind_inputs({"x": x_val.numpy()})
    backend.reset_events()
    events = {}
    for cmd in compiled.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    result = backend.wait(events[compiled.output_cmd_id])
    ref = torch.nn.functional.linear(x_val.float(), weight.float()).numpy()
    return tiled, gm, result, ref


def test_col_weight_k_tiles_execute() -> None:
    """列切权重沿 K 切成多段时，各段同区间相加不能再插多余的收集。

    8x64 激活、64x64 float16 权重、预算 512：M、N、K 都要切，K 段输出持有者
    只是部分台，相加后仍是同一区间，执行计划与数值都要正确。
    """
    tiled, gm, result, ref = _run_single_linear(
        "gate_proj_weight", "col", (8, 64), (64, 64), 512)
    assert tiled == 1
    k_cuts = [n for n in gm.graph.nodes
              if n.target is torch.ops.aten.slice.Tensor and n.args[1] in (1, -1)]
    assert k_cuts, "预算 512 应沿 K 切"
    assert np.allclose(result, ref, atol=5e-2, rtol=1e-2), np.abs(result - ref).max()


@pytest.mark.parametrize("x_shape, w_shape, budget", [
    ((8, 64), (64, 64), 512),
    ((8, 48), (16, 48), 384),
])
def test_row_weight_k_tiles_execute(x_shape, w_shape, budget) -> None:
    """行切权重沿 K 切成多段：各段部分和只落在持有这段 K 的台上，相加后一次归约。

    K 段与分片边界对齐时一段只在一台，跨边界时拆成两段各在一台；两种都要
    构建得出通信计划、数值与整块一致。
    """
    tiled, gm, result, ref = _run_single_linear(
        "down_proj_weight", "row", x_shape, w_shape, budget)
    assert tiled == 1
    def on_weight(node):
        # 两维都切时 K 那一刀挂在 N 那一刀上，顺着切片链找到权重。
        while node.target is torch.ops.aten.slice.Tensor:
            node = node.args[0]
        return node.op == "get_attr"

    k_cuts = [n for n in gm.graph.nodes
              if n.target is torch.ops.aten.slice.Tensor and n.args[1] in (1, -1)
              and on_weight(n)]
    assert k_cuts, "行切权重应沿 K 切"
    assert np.allclose(result, ref, atol=5e-2, rtol=1e-2), np.abs(result - ref).max()


def test_fused_elementwise_host_op_binds_numpy_input() -> None:
    """融合出的主机节点吃 numpy 绑定的输入，必须算出正确数值。

    占位符经 `resolve_node` 原样返回 numpy 数组，而闭包里的 aten 算子只认
    torch 张量，直接执行会抛「Expected Tensor」。主机节点读取的其余张量
    都已过 `_as_torch`，占位符是漏掉的那一处。
    """
    from graph.opt_passes import optimize_graph

    graph = Graph()
    x = graph.placeholder("x")
    neg = graph.call_function(torch.ops.aten.neg.default, (x,))
    tanh = graph.call_function(torch.ops.aten.tanh.default, (neg,))
    graph.output(tanh)
    gm = GraphModule({}, graph)
    x_val = torch.randn(2, 4)
    for n, val in ((x, x_val), (neg, torch.empty(2, 4)),
                   (tanh, torch.empty(2, 4))):
        n.meta["val"] = val
    partition_graph(gm)
    assert optimize_graph(gm).fused == 1, "neg -> tanh 应合成一个节点"
    strategy = ShardStrategy(name="tp2", num_dpus=2, dpu_ids=(0, 1),
                             weight_rules=())
    edges = propagate_specs(gm, strategy)
    compiled, _ = _plan_and_compile(gm, edges, num_tasklets=1)

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 16))
    register_all(backend)
    backend.bind_inputs({"x": x_val.numpy()})
    events = {}
    for cmd in compiled.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    # 主机节点的结果是 torch 张量；np.array 会让 numpy 2 报 copy 关键字弃用告警。
    result = np.asarray(backend.wait(events[compiled.output_cmd_id]))
    ref = np.tanh(-x_val.numpy())
    assert np.allclose(result, ref, atol=1e-6), np.abs(result - ref).max()


@pytest.mark.parametrize("compiled", [False, True])
def test_linear_silu_executes_end_to_end(compiled) -> None:
    """gate 投影的 silu 折进 linear 后，执行计划两种内核都要对齐 torch。

    镜像与编译内核按 payload 的 activation 补上 silu，数值与
    torch 的 `silu(linear(x, gate_w)) * linear(x, up_w)` 一致。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from graph.opt_passes import fuse_linear_activation, optimize_graph

    torch.manual_seed(0)
    gate_w = torch.randn(128, 64)
    up_w = torch.randn(128, 64)
    activation = torch.randn(2, 64)
    reference = (torch.nn.functional.silu(
        torch.nn.functional.linear(activation, gate_w))
        * torch.nn.functional.linear(activation, up_w))

    graph = Graph()
    x = graph.placeholder("x")
    gw = graph.get_attr("gate_w")
    uw = graph.get_attr("up_w")
    proj = graph.call_function(torch.ops.aten.linear.default, (x, gw))
    act = graph.call_function(torch.ops.aten.silu.default, (proj,))
    up = graph.call_function(torch.ops.aten.linear.default, (x, uw))
    out = graph.call_function(torch.ops.aten.mul.Tensor, (act, up))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_w", gate_w.to(torch.float16))
    mod.register_buffer("up_w", up_w.to(torch.float16))
    gm = GraphModule(mod, graph)
    for n, val in ((x, torch.empty(2, 64, dtype=torch.float16)),
                   (gw, torch.empty(128, 64, dtype=torch.float16)),
                   (uw, torch.empty(128, 64, dtype=torch.float16)),
                   (proj, torch.empty(2, 128, dtype=torch.float16)),
                   (act, torch.empty(2, 128, dtype=torch.float16)),
                   (up, torch.empty(2, 128, dtype=torch.float16)),
                   (out, torch.empty(2, 128, dtype=torch.float16))):
        n.meta["val"] = val
    partition_graph(gm)
    optimize_graph(gm, wram_bytes=65536, tp_width=2)
    strategy = ShardStrategy(name="tp2", num_dpus=2, dpu_ids=(0, 1),
                             weight_rules=(("gate_w", "col"), ("up_w", "col")))
    edges = propagate_specs(gm, strategy)
    assert fuse_linear_activation(gm) == 1, "gate 投影的 silu 没有折进 linear"
    assert FUSED_TAIL_META_KEY in proj.meta
    compiled_plan, plans = _plan_and_compile(gm, edges, num_tasklets=1)

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 20))
    register_all(backend, use_compiled_linear=compiled)
    for name, tensor in (("gate_w", gate_w), ("up_w", up_w)):
        stored = tensor.detach().numpy().astype(np.float16)
        for dpu_id in (0, 1):
            backend.write_local(dpu_id, plans[dpu_id].weight[name],
                                stored[dpu_id * 64:(dpu_id + 1) * 64])
    backend.bind_inputs({"x": activation.numpy().astype(np.float16)})
    events = {}
    for cmd in compiled_plan.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    result = np.array(backend.wait(events[compiled_plan.output_cmd_id]))
    err = np.abs(result.astype(np.float32) - reference.numpy()).max()
    assert np.allclose(result.astype(np.float32), reference.numpy(),
                       atol=5e-2, rtol=1e-2), err


def _tail_activation_graph(act, second_linear):
    """linear → act（→ linear）小图，两台按列切，激活输出要收集。"""
    torch.manual_seed(0)
    weights = {"w1": torch.randn(128, 64)}
    if second_linear:
        weights["w2"] = torch.randn(64, 128)
    graph = Graph()
    x = graph.placeholder("x")
    w1 = graph.get_attr("w1")
    proj = graph.call_function(torch.ops.aten.linear.default, (x, w1))
    tail = graph.call_function(act, (proj,))
    vals = [(x, (2, 64)), (w1, (128, 64)), (proj, (2, 128)), (tail, (2, 128))]
    out = tail
    if second_linear:
        w2 = graph.get_attr("w2")
        out = graph.call_function(torch.ops.aten.linear.default, (tail, w2))
        vals += [(w2, (64, 128)), (out, (2, 64))]
    graph.output(out)
    mod = torch.nn.Module()
    for name, w in weights.items():
        mod.register_buffer(name, w.to(torch.float16))
    gm = GraphModule(mod, graph)
    for node, shape in vals:
        node.meta["val"] = torch.empty(shape, dtype=torch.float16)
    return gm, weights, proj


@pytest.mark.parametrize("second_linear", [False, True],
                         ids=["to_output", "to_linear"])
@pytest.mark.parametrize("act", [torch.ops.aten.relu.default,
                                 torch.ops.aten.sigmoid.default],
                         ids=["relu", "sigmoid"])
def test_fused_tail_activation_with_gathered_output(act, second_linear) -> None:
    """激活输出带收集边时折进 linear，执行计划照常生成，数值与 torch 一致。

    折叠删掉激活节点，以它为源的重分布边必须改指 linear，否则执行计划
    按边的源名查不到节点。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY
    from graph.opt_passes import fuse_linear_activation, optimize_graph

    gm, weights, proj = _tail_activation_graph(act, second_linear)
    partition_graph(gm)
    optimize_graph(gm, wram_bytes=65536, tp_width=2)
    rules = tuple((name, "col") for name in weights)
    strategy = ShardStrategy(name="tp2", num_dpus=2, dpu_ids=(0, 1),
                             weight_rules=rules)
    edges = propagate_specs(gm, strategy)
    assert any(e.src == f"{act.__name__.split('.')[0]}_default" for e in edges), \
        "用例前提：激活输出要有重分布边"
    assert fuse_linear_activation(gm, edges) == 1
    assert FUSED_TAIL_META_KEY in proj.meta
    compiled_plan, plans = _plan_and_compile(gm, edges, num_tasklets=1)

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 20))
    register_all(backend)
    for name, w in weights.items():
        stored = w.numpy().astype(np.float16)
        half = stored.shape[0] // 2
        for dpu_id in (0, 1):
            backend.write_local(dpu_id, plans[dpu_id].weight[name],
                                stored[dpu_id * half:(dpu_id + 1) * half])
    x = torch.randn(2, 64)
    backend.bind_inputs({"x": x.numpy().astype(np.float16)})
    events = {}
    for cmd in compiled_plan.plan.commands:
        for wait in cmd.waits:
            backend.wait(events[wait])
        events[cmd.id] = backend.submit(cmd)
    result = np.array(backend.wait(events[compiled_plan.output_cmd_id]))
    xh = x.half().float()
    ref = act(torch.nn.functional.linear(xh, weights["w1"].half().float()))
    if second_linear:
        ref = torch.nn.functional.linear(ref.half().float(),
                                         weights["w2"].half().float())
    np.testing.assert_allclose(result.astype(np.float32), ref.numpy(),
                               atol=5e-2, rtol=1e-2)
