"""验证 KV 池与并行窗口的自动定容：按 MRAM 剩余空间收敛，超限时报出所需与可用。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.kv_layout import KVRegionSpec, build_kv_layout
from memory.mem_planner import DPUPlan, HwBudget, fit_pool_slots
from runtime.compile import compile_llama2

NUM_DPUS = 4
MAX_SEQ = 16
KV_SLOT_TOKENS = 4
ALIGN = 256
RESERVE = 1 << 20


def test_fit_pool_slots_counts_remaining_mram() -> None:
    """槽数 = （预算 − 权重 − 块表 − 激活窗口）÷ 单槽字节数。"""
    hw = HwBudget(mram_bytes=1 << 16, align=ALIGN, sys_reserve_bytes=1 << 12)
    spec = KVRegionSpec(
        dpu_id=0, layers=[0], kv_heads=[0], q_heads_by_kv={0: [0]},
        max_seq=8, head_dim=8, dtype_bytes=2, kv_base=4096)
    build_kv_layout(spec, ALIGN, pool_slots=1, block_table_slots=1)
    plan = DPUPlan(
        weight={}, kv_base=4096, act_base=0, act_prefill={}, act_decode={},
        total=0, pending_readers_prefill={}, pending_readers_decode={},
        act_extent=2048)

    room = (1 << 16) - (1 << 12) - 4096 - spec.block_table_bytes - 2048
    assert fit_pool_slots({0: plan}, {0: spec}, hw) == room // spec.slot_bytes
    # 每多一个激活窗口就少一个槽。
    assert fit_pool_slots({0: plan}, {0: spec}, hw, act_slots=2) == (
        room - 2048) // spec.slot_bytes


def _compile(mram_bytes: int, **kwargs):
    """按给定 MRAM 预算编译微型 Llama。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=MAX_SEQ,
    )).eval().half()
    hw = HwBudget(mram_bytes=mram_bytes, align=ALIGN, sys_reserve_bytes=RESERVE)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=4, intermediate_size=128,
        vocab_size=128, num_layers=4)[0]
    return compile_llama2(
        model, strategy, prefill_seq_len=MAX_SEQ, max_seq=MAX_SEQ, hw=hw,
        hardware=hardware, num_request_slots=4, kv_slot_tokens=KV_SLOT_TOKENS,
        **kwargs)


def test_auto_capacity_fits_where_fixed_config_fails() -> None:
    """同一份预算下，显式满配装不下，自动定容能把槽数收到装得下。"""
    roomy = _compile(1 << 28)
    spec = next(iter(roomy.kv_specs.values()))
    blocks = spec.blocks_per_request
    # 上限是满并发容量再加一块，留给 vLLM 的 null 块。
    assert roomy.kv_pool_slots == 4 * blocks + 1
    assert roomy.num_parallel_slots == 2

    # 预算只留「一个请求的槽 + 两个窗口」的空间，再扣掉内核 tile 就装不下满配。
    tight = max(plan.total for plan in roomy.mem_plans.values())
    tight -= (roomy.kv_pool_slots - blocks) * spec.slot_bytes
    with pytest.raises(ValueError, match="内存超限"):
        _compile(tight + RESERVE, kv_pool_slots=roomy.kv_pool_slots)

    auto = _compile(tight + RESERVE)
    assert blocks < auto.kv_pool_slots < roomy.kv_pool_slots
    assert auto.num_parallel_slots <= roomy.num_parallel_slots


def test_explicit_pool_over_budget_names_need_and_available() -> None:
    """显式给的槽数装不下时报错，并写出峰值与可用预算。"""
    with pytest.raises(ValueError, match="内存超限") as excinfo:
        _compile(1 << 28, kv_pool_slots=1 << 20)
    message = str(excinfo.value)
    assert "peak=" in message and "可用预算=" in message


def test_explicit_pool_that_fits_is_kept() -> None:
    """显式给的槽数装得下时原样保留，不被自动值覆盖。"""
    compiled = _compile(1 << 28, kv_pool_slots=9)
    assert compiled.kv_pool_slots == 9


def test_vllm_starts_without_pim_env(tmp_path, monkeypatch) -> None:
    """不设任何 PIM_* 环境变量也能启动，vLLM 分到的块数等于自动定容的槽数。"""
    from backend.vllm_executor import PIMExecutor
    from vllm import LLM, SamplingParams

    for name in ("PIM_KV_POOL_SLOTS", "PIM_PARALLEL_SLOTS", "PIM_PREFILL_SEQ_LEN",
                 "PIM_PREFILL_SEQ_LENS", "PIM_NUM_DPUS", "PIM_MRAM_BYTES",
                 "PIM_SYS_RESERVE_BYTES"):
        monkeypatch.delenv(name, raising=False)
    # 与 serve_pim 一致：引擎与 worker 同进程，才能读回编译期的定容结果。
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    seq_len = 32
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny-llama"
    LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=seq_len,
    )).eval().half().save_pretrained(model_dir)

    llm = LLM(
        model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
        max_model_len=seq_len, max_num_seqs=2, max_num_batched_tokens=seq_len,
        gpu_memory_utilization=0.2, distributed_executor_backend=PIMExecutor)
    out = llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}],
                       SamplingParams(max_tokens=2, temperature=0.0))[0]

    worker = llm.llm_engine.model_executor.driver_worker
    # 两条满长请求的块数再加一块给 null 块；MRAM 宽裕时就是这个上限。
    assert worker.kv_pool_slots == worker.blocks_per_slot() * 2 + 1
    assert worker.compiled.num_parallel_slots == 2
    assert len(out.outputs[0].token_ids) == 2
