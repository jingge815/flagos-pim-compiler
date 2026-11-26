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
from runtime.executor import DecodeState, execute_plan
from runtime.kernels import register_all, sdpa_kv_info
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

# 同文件用例固定在同一 worker：共享 module 级 7B 夹具，也避免多 worker 各持一份模型
pytestmark = [pytestmark, pytest.mark.xdist_group("llama2-executor")]


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
    hw = HwBudget(mram_bytes=4 * 2**30, align=64, sys_reserve_bytes=64 * 2**20)
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

    def sdpa_info_of(node):
        if "scaled_dot_product_attention" not in str(node.target):
            return None
        return sdpa_kv_info(node, kv_specs, sdpa_layer, np.dtype(np.float16))

    compiled = build_execution_plan(
        nodes, gm, entries_by_id, pending, hardware=hardware,
        sdpa_info_of=sdpa_info_of,
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


# --- B=2：7B 规模的编排器多 batch 路径（L2 层级，与既有 7B 用例同组） ---------


@pytest.fixture(scope="module")
def llama2_b2():
    """编译真实 7B 的 B=2 图（[2,S] prefill / [2,1] decode）并装载权重。"""
    from graph.strategy import llama_strategy
    from runtime.compile import causal_mask_of as _causal_mask_of
    from runtime.compile import compile_llama2, load_weights
    from runtime.executor import run_decode_loop as _run_decode_loop
    from runtime.kernels import register_all as _register_all

    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config
    strategy = llama_strategy(
        NUM_DPUS, num_stages=1, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers)
    hw = HwBudget(mram_bytes=4 * 2**30, align=64, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, kv_dtype_bytes=KV_DTYPE_BYTES,
        batch_size=2)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    _register_all(backend)
    load_weights(compiled, backend)
    return model, compiled, backend, _run_decode_loop, _causal_mask_of


def test_b2_prefill_and_decode_match_torch(llama2_b2) -> None:
    """B=2 的 7B：prefill 与每步 decode 的 logits 与 batched PyTorch 容差对齐。

    覆盖 7B 规模下 B>1 的 exec_plan/comm plan 与 sdpa 逐槽读写路径（round05
    问题 9 的判据：与单卡 PyTorch batched forward 对 logits，
    max<0.2/mean<0.02）。判据取 logits 容差而非贪心 token 相等：7B fp16 下
    实测本用例 prefill diff max≈0.094、decode diff max≈0.031（均在容差内），
    而 decode 首步槽 1 的参考 top-2 间距仅 0.0039——容差内的数值漂移会翻转
    argmax，贪心相等不是数值正确性的等价锚点（小型 fixture 的 B=2/4 贪心
    一致用例继续覆盖该较强口径）。
    """
    model, compiled, backend, _, mask_of = llama2_b2
    prompts = [[(i * 7 + b * 13) % (32000 - 1) + 1 for i in range(SEQ_LEN)]
               for b in range(2)]
    # 公开契约：消费编译结果的 state（B=2 两槽），不手工另建。
    state = compiled.state
    events = execute_plan(
        compiled.prefill.plan, backend,
        values={"input_ids": torch.tensor(prompts, dtype=torch.long),
                "causal_mask": mask_of(SEQ_LEN),
                "position_ids": (torch.arange(SEQ_LEN, dtype=torch.long)
                                 .unsqueeze(0).repeat(2, 1))},
        pos=tuple(state.valid_lens),
    )
    logits = np.asarray(backend.wait(events[compiled.prefill.output_cmd_id]))
    state.valid_lens = (SEQ_LEN, SEQ_LEN)
    ref = model(
        input_ids=torch.tensor(prompts, dtype=torch.long),
        attention_mask=mask_of(SEQ_LEN), use_cache=False).logits
    diff = np.abs(logits.astype(np.float32) - ref.numpy().astype(np.float32))
    assert diff.max() < 0.2
    assert diff.mean() < 0.02

    # 每步 decode 的 [2, vocab] logits 与全序列重算参考同容差对拍。
    seqs = [list(p) for p in prompts]
    last = [int(np.argmax(logits[b, -1])) for b in range(2)]
    for seq, token in zip(seqs, last):
        seq.append(token)
    for _ in range(2):
        events = execute_plan(
            compiled.decode.plan, backend,
            values={
                "input_ids": torch.tensor([[t] for t in last],
                                          dtype=torch.long),
                "causal_mask": mask_of(1),
                "position_ids": torch.tensor(
                    state.valid_lens).reshape(-1, 1)},
            pos=tuple(state.valid_lens),
        )
        step = np.asarray(
            backend.wait(events[compiled.decode.output_cmd_id]))
        state.valid_lens = tuple(v + 1 for v in state.valid_lens)
        cur = len(seqs[0])
        with torch.no_grad():
            step_ref = model(
                input_ids=torch.tensor(seqs, dtype=torch.long),
                attention_mask=mask_of(cur),
                position_ids=(torch.arange(cur, dtype=torch.long)
                              .unsqueeze(0).repeat(2, 1)),
                use_cache=False).logits[:, -1]
        ours = step[:, -1, :] if step.ndim == 3 else step
        diff = np.abs(ours.astype(np.float32)
                      - step_ref.numpy().astype(np.float32))
        assert diff.max() < 0.2, diff.max()
        assert diff.mean() < 0.02, diff.mean()
        last = [int(np.argmax(ours[b])) for b in range(2)]
        for seq, token in zip(seqs, last):
            seq.append(token)
