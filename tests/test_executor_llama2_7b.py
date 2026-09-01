"""验证真实 Llama-2-7B 的整图 prefill 输出。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig
from genesim_bridge.paths import llama2_7b_model_dir
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.compile import sdpa_layer_map, write_weight_shards
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, execute_plan, make_sdpa_handler
from runtime.kernels import register_all
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
SEQ_LEN = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 llama2_7b_model_dir",
)


@pytest.fixture(scope="module")
def llama2_compiled_plan():
    """编译真实 7B 的 prefill 图并生成执行计划。"""
    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config

    input_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()

    partition_graph(gm)
    shard_config = llama_shard_config(
        NUM_DPUS,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
    )
    edges = propagate_specs(gm, shard_config)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    (k_proj,) = [
        n
        for n in gm.graph.nodes
        if n.op == "get_attr" and "layers.0.self_attn.k_proj.weight" in n.target
    ]
    kv_specs = kv_specs_from_placement(
        k_proj.meta[SPEC_META_KEY],
        layers=list(range(cfg.num_hidden_layers)),
        num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        max_seq=MAX_SEQ,
        dtype_bytes=KV_DTYPE_BYTES,
        kv_base=0,
    )
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS,
        num_tasklets=4,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        dma_align=64,
    )
    nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, nodes, nodes, kv_specs, hw) for d in range(NUM_DPUS)}

    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)

    state = DecodeState(valid_len=0)
    sdpa_layer = sdpa_layer_map(gm)

    def host_handler_of(node):
        if "scaled_dot_product_attention" in str(node.target):
            return make_sdpa_handler(
                sdpa_layer[node.name], kv_specs, state, np.dtype(np.float16)
            )
        return None

    compiled = build_execution_plan(
        nodes,
        gm,
        entries_by_id,
        pending,
        hardware=hardware,
        host_handler_of=host_handler_of,
    )
    return model, gm, input_ids, causal_mask, compiled, plans, state


@pytest.fixture(scope="module")
def llama2_executed(llama2_compiled_plan):
    """将权重写入 DPU 后执行一次完整 prefill 计划。"""
    model, gm, input_ids, causal_mask, compiled, plans, state = llama2_compiled_plan
    hw_mram_bytes = 4 * 2**30

    backend = NumpyBackend(
        NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw_mram_bytes)
    )
    register_all(backend)
    write_weight_shards(gm, plans, backend)

    events = execute_plan(
        compiled.plan,
        backend,
        values={"input_ids": input_ids, "causal_mask": causal_mask},
        pos=state.valid_len,
    )
    result = np.asarray(backend.wait(events[compiled.output_cmd_id]))
    ref = model(input_ids=input_ids, attention_mask=causal_mask, use_cache=False).logits
    return result, ref


def test_logits_match_torch_reference(llama2_executed) -> None:
    """验证整图 logits 与单卡 PyTorch 输出一致。"""
    result, ref = llama2_executed
    diff = np.abs(result.astype(np.float32) - ref.numpy().astype(np.float32))
    assert diff.max() < 0.2
    assert diff.mean() < 0.02


def test_greedy_argmax_matches_at_every_position(llama2_executed) -> None:
    """验证每个位置的贪心 token 与单卡 PyTorch 一致。"""
    result, ref = llama2_executed
    ours = np.asarray(result).argmax(-1)
    theirs = ref.numpy().argmax(-1)
    assert np.array_equal(ours, theirs)
