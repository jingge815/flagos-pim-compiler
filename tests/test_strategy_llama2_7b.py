"""验证 Llama-2-7B 代表切分策略的推理和 KV 缓存。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from genesim_bridge.paths import llama2_7b_model_dir
from graph.strategy import llama_strategies
from memory.kv_layout import PIMStaticKVCache
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
PREFILL_SEQ_LEN = 16
DECODE_STEPS = 8
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 llama2_7b_model_dir",
)


def _strategies():
    """返回 Llama-2-7B 的切分策略集合。"""
    return llama_strategies(
        NUM_DPUS, num_heads=32, num_kv_heads=32, intermediate_size=11008,
        vocab_size=32000, num_layers=32,
    )


def _representative_strategies():
    """返回用于真实模型验证的代表策略。"""
    return [strategy for strategy in _strategies() if strategy.name == "tp4_pp2"]


def test_real_7b_suite_keeps_tp4_pp2_as_representative_strategy() -> None:
    assert [strategy.name for strategy in _representative_strategies()] == ["tp4_pp2"]


@pytest.fixture(scope="module")
def llama2_model():
    torch.set_grad_enabled(False)
    return LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()


@pytest.fixture(scope="module")
def hf_reference(llama2_model):
    """生成单卡 HF 的贪心解码结果和最终 KV cache。"""
    prompt = torch.arange(PREFILL_SEQ_LEN, dtype=torch.long)
    tokens: list[int] = []
    with torch.no_grad():
        out = llama2_model(input_ids=prompt.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        tokens.append(nxt)
        for _ in range(DECODE_STEPS - 1):
            out = llama2_model(
                input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
    return tokens, past


@pytest.fixture(scope="module")
def run_cache():
    """缓存按策略名称索引的推理结果。"""
    return {}


def _run_strategy_cached(cache, model, strategy):
    if strategy.name not in cache:
        cache[strategy.name] = _run_strategy(model, strategy)
    return cache[strategy.name]


def _run_strategy(model, strategy):
    """编译策略、加载权重并返回解码结果和运行时对象。"""
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, kv_dtype_bytes=KV_DTYPE_BYTES,
    )
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    compiled.state.valid_len = 0

    generated = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=torch.arange(PREFILL_SEQ_LEN, dtype=torch.long),
        max_new_tokens=DECODE_STEPS, eos_id=-1, state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )
    return generated, backend, compiled


@pytest.mark.parametrize("strategy", _representative_strategies(), ids=lambda s: s.name)
def test_real_llama2_7b_inference_matches_hf_under_every_strategy(
    strategy, llama2_model, hf_reference, run_cache
) -> None:
    """判据 1：真实 7B 在这种切分下贪心解码序列与 HF 单卡逐 token 一致。"""
    ref_tokens, _ = hf_reference
    generated, _, _ = _run_strategy_cached(run_cache, llama2_model, strategy)

    assert generated == ref_tokens, (
        f"策略 {strategy.name}（{strategy.num_stages} stage × tp{strategy.tp_width}）"
        f"解码序列偏离 HF: {generated} vs {ref_tokens}"
    )


@pytest.mark.parametrize("strategy", _representative_strategies(), ids=lambda s: s.name)
def test_real_llama2_7b_kv_region_matches_hf_cache_under_every_strategy(
    strategy, llama2_model, hf_reference, run_cache
) -> None:
    """验证各策略的 KV 区数据与 HF 缓存一致。"""
    _, ref_past = hf_reference
    _, backend, compiled = _run_strategy_cached(run_cache, llama2_model, strategy)

    kv_specs = compiled.kv_specs
    cache = PIMStaticKVCache(backend, kv_specs, wram_budget_bytes=2**20)
    total_len = compiled.state.valid_len
    assert total_len == PREFILL_SEQ_LEN + DECODE_STEPS - 1

    for layer in (0, 15, 31):  # 首/中/末层抽查，覆盖不同 stage
        ref_k = ref_past.layers[layer].keys
        ref_v = ref_past.layers[layer].values
        owners = [(d, s) for d, s in kv_specs.items() if layer in s.layers]
        assert owners, f"策略 {strategy.name}: layer{layer} 没有任何 DPU 持有 KV"
        for dpu_id, spec in owners:
            for head in spec.kv_heads:
                K_tile, V_tile = cache.read_tile(layer, dpu_id, head, 0, total_len)
                ref_k_h = ref_k[0, head].numpy().astype(np.float32)
                ref_v_h = ref_v[0, head].numpy().astype(np.float32)
                assert np.allclose(
                    K_tile.astype(np.float32), ref_k_h, rtol=2e-2, atol=3e-2
                ), (
                    f"{strategy.name} layer{layer} dpu{dpu_id} head{head} K 区不匹配: "
                    f"max diff {np.abs(K_tile.astype(np.float32) - ref_k_h).max()}"
                )
                assert np.allclose(
                    V_tile.astype(np.float32), ref_v_h, rtol=2e-2, atol=3e-2
                ), (
                    f"{strategy.name} layer{layer} dpu{dpu_id} head{head} V 区不匹配: "
                    f"max diff {np.abs(V_tile.astype(np.float32) - ref_v_h).max()}"
                )


@pytest.mark.parametrize("strategy", _representative_strategies(), ids=lambda s: s.name)
def test_real_llama2_7b_memory_footprint_fits_and_scales(strategy, llama2_model) -> None:
    """验证各策略的内存布局不超过单 DPU 容量。"""
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    compiled = compile_llama2(
        llama2_model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, kv_dtype_bytes=KV_DTYPE_BYTES,
    )

    layers_per_stage = 32 // strategy.num_stages
    for dpu_id, plan in compiled.mem_plans.items():
        assert plan.total <= hw.mram_bytes, f"{strategy.name} dpu{dpu_id} 超预算"
        assert compiled.kv_specs[dpu_id].layers == strategy.layers_of_dpu(dpu_id, 32)
        assert len(compiled.kv_specs[dpu_id].layers) == layers_per_stage
