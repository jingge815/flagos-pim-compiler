"""验证 Llama-2-7B 的内存分区、容量和回填地址。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.graph_meta import SPEC_META_KEY
from genesim_bridge.paths import llama2_7b_model_dir
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import PIMStaticKVCache, kv_specs_from_placement
from memory.mem_planner import HwBudget, format_mem_plan, plan_dpu
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
PREFILL_SEQ_LEN = 16
DECODE_SEQ_LEN = 1
MAX_SEQ = 256
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 llama2_7b_model_dir",
)


def _export_and_annotate(model: LlamaForCausalLM, shard_config, seq_len: int):
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()
    partition_graph(gm)
    propagate_specs(gm, shard_config)
    return gm


@pytest.fixture(scope="module")
def llama2_two_graphs():
    """导出并标记预填充和解码两张图。"""
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config
    shard_config = llama_shard_config(
        NUM_DPUS,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
    )
    prefill_gm = _export_and_annotate(model, shard_config, PREFILL_SEQ_LEN)
    decode_gm = _export_and_annotate(model, shard_config, DECODE_SEQ_LEN)
    return model, cfg, prefill_gm, decode_gm


@pytest.fixture(scope="module")
def kv_specs(llama2_two_graphs):
    _, cfg, prefill_gm, _ = llama2_two_graphs
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    (k_proj,) = [
        n for n in prefill_gm.graph.nodes
        if n.op == "get_attr" and "layers.0.self_attn.k_proj.weight" in n.target
    ]
    return kv_specs_from_placement(
        k_proj.meta[SPEC_META_KEY],
        layers=list(range(cfg.num_hidden_layers)),
        num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        max_seq=MAX_SEQ,
        dtype_bytes=KV_DTYPE_BYTES,
        kv_base=0,
    )


@pytest.fixture(scope="module")
def hw_budget():
    # 容量覆盖每个 DPU 的权重分片。
    return HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)


@pytest.fixture(scope="module")
def plans(llama2_two_graphs, kv_specs, hw_budget):
    _, _, prefill_gm, decode_gm = llama2_two_graphs
    prefill_nodes = list(prefill_gm.graph.nodes)
    decode_nodes = list(decode_gm.graph.nodes)
    return {
        d: plan_dpu(d, prefill_nodes, decode_nodes, kv_specs, hw_budget)
        for d in range(NUM_DPUS)
    }


def _weight_node(gm, pattern: str):
    (node,) = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    return node


def test_weight_region_matches_hand_computed_bytes(llama2_two_graphs, plans) -> None:
    """验证权重区容纳该 DPU 的全部本地分片。"""
    _, _, prefill_gm, _ = llama2_two_graphs
    by_target = {n.target: n for n in prefill_gm.graph.nodes if n.op == "get_attr"}
    for dpu_id, plan in plans.items():
        expected = 0
        for name in plan.weight:
            node = by_target[name]
            detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
            itemsize = node.meta["val"].element_size()
            size = 1
            for dim in detail.local_shape:
                size *= dim
            expected += size * itemsize
        # 对齐后的权重区覆盖未对齐字节总数。
        assert plan.kv_base >= expected


def test_weight_offsets_shared_between_prefill_and_decode(llama2_two_graphs, plans) -> None:
    _, _, prefill_gm, decode_gm = llama2_two_graphs
    plan = plans[0]
    pattern = "layers.0.self_attn.q_proj.weight"
    prefill_node = _weight_node(prefill_gm, pattern)
    decode_node = _weight_node(decode_gm, pattern)
    off = plan.weight[prefill_node.target]
    assert prefill_node.meta[SPEC_META_KEY].shard_map[0].mram_offset == off
    assert decode_node.meta[SPEC_META_KEY].shard_map[0].mram_offset == off


def test_regions_are_disjoint_and_ordered(plans, kv_specs) -> None:
    for dpu_id, plan in plans.items():
        assert max(plan.weight.values()) < plan.kv_base if plan.weight else True
        assert plan.kv_base + kv_specs[dpu_id].kv_allocated_bytes == plan.act_base
        assert plan.act_base <= plan.total
        for offsets in (plan.act_prefill, plan.act_decode):
            for off in offsets.values():
                assert plan.act_base <= off < plan.total or not offsets


def test_capacity_check_passes_with_realistic_budget(plans, hw_budget) -> None:
    for plan in plans.values():
        assert plan.total <= hw_budget.mram_bytes - hw_budget.sys_reserve_bytes


def test_capacity_check_rejects_too_small_budget(llama2_two_graphs, kv_specs) -> None:
    _, _, prefill_gm, decode_gm = llama2_two_graphs
    tiny = HwBudget(mram_bytes=1 << 20, align=1024, sys_reserve_bytes=0)  # 1MiB，装不下 7B 权重分片
    with pytest.raises(ValueError, match="内存超限"):
        plan_dpu(0, list(prefill_gm.graph.nodes), list(decode_gm.graph.nodes), kv_specs, tiny)


def test_weight_offset_is_a_real_usable_address_on_numpy_backend(llama2_two_graphs, plans, hw_budget) -> None:
    """验证回填的权重偏移可用于读写本地分片。"""
    model, cfg, prefill_gm, _ = llama2_two_graphs
    dpu_id = 0
    plan = plans[dpu_id]
    node = _weight_node(prefill_gm, "layers.0.self_attn.q_proj.weight")
    detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
    off = plan.weight[node.target]
    assert off == detail.mram_offset

    ref = model.model.layers[0].self_attn.q_proj.weight.detach()
    local_shape = detail.local_shape
    local_ref = ref[detail.start_idx : detail.end_idx].contiguous().numpy()
    assert tuple(local_ref.shape) == local_shape

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw_budget.mram_bytes))
    backend.write_local(dpu_id, off, local_ref)
    readback = backend.read_local(dpu_id, off, local_shape, np.float16)
    assert np.array_equal(readback, local_ref)


def test_kv_region_lands_at_planned_kv_base_and_stays_usable(kv_specs, plans, hw_budget) -> None:
    """验证规划的 KV 区起点可用于缓存读写。"""
    dpu_id = 0
    spec = kv_specs[dpu_id]
    assert spec.kv_base == plans[dpu_id].kv_base
    assert all(off >= spec.kv_base for off in spec.kv_off.values())
    assert max(spec.kv_off.values()) + MAX_SEQ * spec.head_dim * spec.dtype_bytes == plans[dpu_id].act_base

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw_budget.mram_bytes))
    cache = PIMStaticKVCache(backend, kv_specs, wram_budget_bytes=2**20)
    rng = np.random.default_rng(0)
    head = spec.kv_heads[0]
    k = rng.standard_normal(spec.head_dim).astype(np.float16)
    v = rng.standard_normal(spec.head_dim).astype(np.float16)
    cache.update(0, 0, {d: {h: k for h in s.kv_heads} for d, s in kv_specs.items()},
                 {d: {h: v for h in s.kv_heads} for d, s in kv_specs.items()})
    K_tile, V_tile = cache.read_tile(0, dpu_id, head, 0, 1)
    assert np.array_equal(K_tile[0], k) and np.array_equal(V_tile[0], v)


def test_format_mem_plan_printable(plans, hw_budget) -> None:
    text = format_mem_plan(plans, hw_budget)
    assert "dpu0" in text and "margin=" in text
    print("\n" + "\n".join(text.splitlines()[:4]))


def test_redistribute_landing_offsets_reach_comm_plan_dst_addr(llama2_two_graphs, plans) -> None:
    """验证通信回写地址包含规划的落地缓冲偏移。"""
    from comm.plan import build_comm_plan

    _, _, prefill_gm, _ = llama2_two_graphs
    edges = [e for n in prefill_gm.graph.nodes for e in n.meta.get("redistribute", [])]
    assert edges
    entries = build_comm_plan(edges)
    by_edge_id = {e.edge_id: e for e in edges}

    checked_nonzero = 0
    for entry in entries:
        edge = by_edge_id[entry.edge_id]
        itemsize = np.dtype(edge.dtype).itemsize
        for seg in entry.writeback_segments:
            detail = edge.dst_spec.shard_map[seg.dst_dpu]
            expected = detail.mram_offset + seg.dst_local_offset * itemsize
            assert seg.dst_addr == expected
            if detail.mram_offset != 0:
                checked_nonzero += 1
    assert checked_nonzero > 0
