"""真实 Llama-2-7B 的问题 1 → 问题 2 → 问题 3 端到端验证（编译期结构 + 运行时数值）。

结构判据（继承 test_spec_prop_llama2_7b 的 Megatron 配对，方案二.(4)/附录 B）：
每层 o_proj、down_proj 各一条 all_reduce，展开为 8 收集段 + 8 回写段，DMA 序列
= [push_from ×1, broadcast_to ×1]；logits 经 Shard(2) all_gather 回 host，S=16
时 8 台 × 16 行 = 128 收集段、无广播段；scatter 边全部 src 在 host。
数值判据：真实权重 + 真实计划条目，gate/up→down 的 Megatron 链在 numpy 伪硬件
上经 SDK DMA 与通信原语执行，与单卡 torch 参考对齐；logits all_gather 与一条
scatter 边做纯搬运的逐字节对齐（fp16 无算术，必须精确相等）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.dpu_sdk import dpu_alloc
from comm.lowering import DmaEngine, all_gather, all_reduce, scatter
from comm.plan import build_comm_plan, dma_sequence, format_comm_plan, plan_cost
from contracts.graph_meta import REDISTRIBUTE_META_KEY
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
SEQ_LEN = 16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


@pytest.fixture(scope="module")
def llama2_comm_plan():
    """真实 7B：加载 → export → partition → propagate → build_comm_plan，模块内一次。"""
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config
    input_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(_FixedMaskLlama(model), (input_ids, causal_mask), strict=True).module()
    partition_graph(gm)
    shard_config = llama_shard_config(
        NUM_DPUS,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
    )
    edges = propagate_specs(gm, shard_config)
    entries = build_comm_plan(edges)
    return model, cfg, gm, edges, {e.edge_id: e for e in entries}


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _engine() -> DmaEngine:
    return DmaEngine(dpu_alloc(NUM_DPUS, mram_bytes=2**22))


def _get_attr_node(gm, pattern: str):
    (node,) = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    return node


def _all_reduce_entry(gm, entries, pattern: str):
    """某权重（如 layers.0.mlp.down_proj.weight）行切 linear 后的 all_reduce 条目。"""
    (linear,) = _get_attr_node(gm, pattern).users
    (add,) = [u for u in linear.users if u.target == torch.ops.aten.add.Tensor]
    (edge,) = [e for e in add.meta[REDISTRIBUTE_META_KEY] if e.src == linear.name]
    return entries[edge.edge_id]


def test_every_edge_expands_to_a_plan_entry(llama2_comm_plan) -> None:
    _, _, _, edges, entries = llama2_comm_plan
    assert sorted(entries) == [e.edge_id for e in edges]
    for edge in edges:
        assert entries[edge.edge_id].type == edge.type
        assert entries[edge.edge_id].segments  # llama2 契约内无 local_slice 边


def test_all_reduce_entries_match_megatron_structure(llama2_comm_plan) -> None:
    _, cfg, _, edges, entries = llama2_comm_plan
    reduce_entries = [entries[e.edge_id] for e in edges if e.type == "all_reduce"]
    assert len(reduce_entries) == 2 * cfg.num_hidden_layers
    for entry in reduce_entries:
        assert entry.wait_for == tuple(range(NUM_DPUS))  # 边级读前等待 = 全体源 DPU
        collect, writeback = entry.collect_segments, entry.writeback_segments
        assert len(collect) == NUM_DPUS and len(writeback) == NUM_DPUS
        numel = entry.numel
        for seg in collect:  # Partial：每台全量部分和，去向 host
            assert seg.src_dpu is not None and seg.dst_dpu is None
            assert seg.global_range == (0, numel) and seg.src_local_range == (0, numel)
            assert seg.nbytes == numel * 2 and seg.reduce == "sum"
        for seg in writeback:  # 残差 add 要求 Replicate@dpu：回写全体
            assert seg.src_dpu is None and seg.dst_dpu is not None
            assert seg.global_range == (0, numel) and seg.dst_local_offset == 0
        # DMA 序列：收集合批为一次 push_xfer，回写合并为一次 broadcast_to
        assert [(op.kind, len(op.segments)) for op in dma_sequence(entry)] == [
            ("push_from", NUM_DPUS),
            ("broadcast_to", NUM_DPUS),
        ]
        cost = plan_cost(entry)
        assert cost.transfers == 2 and cost.nbytes == 2 * NUM_DPUS * numel * 2


def test_logits_all_gather_unfolds_per_row_segments(llama2_comm_plan) -> None:
    _, cfg, gm, edges, entries = llama2_comm_plan
    output = next(n for n in gm.graph.nodes if n.op == "output")
    (edge,) = output.meta[REDISTRIBUTE_META_KEY]
    entry = entries[edge.edge_id]

    collect = entry.collect_segments
    assert entry.type == "all_gather" and entry.dst_loc == {"device": "host"}
    assert entry.writeback_segments == []  # dst host：无广播段，结果只落 host
    assert len(collect) == NUM_DPUS * SEQ_LEN  # 每台每行一段（交错布局）
    vocab_per_dpu = cfg.vocab_size // NUM_DPUS
    for dpu_id in range(NUM_DPUS):
        segs = [s for s in collect if s.src_dpu == dpu_id]
        assert len(segs) == SEQ_LEN
        for row, seg in enumerate(segs):
            start = row * cfg.vocab_size + dpu_id * vocab_per_dpu
            assert seg.global_range == (start, start + vocab_per_dpu)
            assert seg.src_local_range == (row * vocab_per_dpu, (row + 1) * vocab_per_dpu)
            assert seg.dst_local_offset == start and seg.nbytes == vocab_per_dpu * 2


def test_scatter_entries_are_host_sourced(llama2_comm_plan) -> None:
    _, _, _, edges, entries = llama2_comm_plan
    scatter_entries = [entries[e.edge_id] for e in edges if e.type == "scatter"]
    assert scatter_entries
    for entry in scatter_entries:
        assert entry.wait_for == ()  # 源在 host，无 DPU 生产者
        assert all(seg.src_dpu is None for seg in entry.segments)
        assert {seg.dst_dpu for seg in entry.segments} == set(range(NUM_DPUS))


def test_plan_report_and_total_cost(llama2_comm_plan) -> None:
    _, _, _, _, entries = llama2_comm_plan
    entries = list(entries.values())
    report = format_comm_plan(entries, max_segments=2)
    assert "all_reduce" in report and "all_gather" in report and "scatter" in report
    total = sum(plan_cost(e).nbytes for e in entries)
    print(f"\n通信计划表: {len(entries)} 条, 总搬运 {total / 2**20:.1f} MiB")
    print("\n".join(report.splitlines()[:6]))
    assert total > 0


def test_megatron_mlp_numeric_matches_torch(llama2_comm_plan) -> None:
    """第 0 层 MLP：gate/up 列切 → silu·mul → down 行切 → all_reduce，对单卡 torch。

    数值在 fp32 下模拟 DPU 本地计算，DMA 搬运按图 dtype 走 fp16（验证搬运与归约
    正确性，不验证算子精度），容差按 fp16 精度给。
    """
    model, _, gm, _, entries = llama2_comm_plan
    entry = _all_reduce_entry(gm, entries, "layers.0.mlp.down_proj.weight")
    mlp = model.model.layers[0].mlp
    w_gate = mlp.gate_proj.weight.detach().float().numpy()  # [11008, 4096]
    w_up = mlp.up_proj.weight.detach().float().numpy()
    w_down = mlp.down_proj.weight.detach().float().numpy()  # [4096, 11008]

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, SEQ_LEN, 4096), dtype=np.float32)
    ref = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.from_numpy(x) @ torch.from_numpy(w_gate).T)
        * (torch.from_numpy(x) @ torch.from_numpy(w_up).T),
        torch.from_numpy(w_down),
    ).numpy()

    engine = _engine()
    width = 11008 // NUM_DPUS
    for dpu_id in range(NUM_DPUS):
        cols = slice(dpu_id * width, (dpu_id + 1) * width)
        hidden = _silu(x @ w_gate[cols].T) * (x @ w_up[cols].T)  # 本地 hidden [1, S, 1376]
        partial = hidden @ w_down[:, cols].T  # 行切 → 全形部分和 [1, S, 4096]
        engine.copy_to_dpu(dpu_id, 0, partial.astype(entry.dtype).reshape(-1))

    acc = all_reduce(entry, engine)

    assert acc.shape == ref.shape
    assert np.allclose(acc.astype(np.float32), ref, rtol=2e-2, atol=1e-2)


def test_logits_all_gather_numeric_byte_exact(llama2_comm_plan) -> None:
    _, cfg, gm, _, entries = llama2_comm_plan
    output = next(n for n in gm.graph.nodes if n.op == "output")
    (edge,) = output.meta[REDISTRIBUTE_META_KEY]
    entry = entries[edge.edge_id]

    rng = np.random.default_rng(1)
    global_logits = rng.standard_normal((1, SEQ_LEN, cfg.vocab_size)).astype(entry.dtype)
    engine = _engine()
    vocab_per_dpu = cfg.vocab_size // NUM_DPUS
    for dpu_id in range(NUM_DPUS):
        shard = global_logits[:, :, dpu_id * vocab_per_dpu : (dpu_id + 1) * vocab_per_dpu]
        engine.copy_to_dpu(dpu_id, 0, np.ascontiguousarray(shard).reshape(-1))

    merged = all_gather(entry, engine)

    assert np.array_equal(merged, global_logits)  # 纯搬运，逐字节相等


def test_scatter_numeric_byte_exact(llama2_comm_plan) -> None:
    _, _, _, edges, entries = llama2_comm_plan
    entry = next(entries[e.edge_id] for e in edges if e.type == "scatter")

    rng = np.random.default_rng(2)
    host_tensor = rng.standard_normal(entry.shape).astype(entry.dtype)
    engine = _engine()

    scatter(entry, engine, host_tensor)

    flat = host_tensor.reshape(-1)
    for seg in entry.segments:
        got = engine.copy_from_dpu(
            seg.dst_dpu, seg.dst_addr, seg.nbytes // entry.dtype.itemsize, entry.dtype
        )
        assert np.array_equal(got, flat[seg.global_range[0] : seg.global_range[1]])
