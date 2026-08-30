"""真实 Llama-2-7B 逐策略完整推理（验证 3：每种切分下整体模型走通正常推理流程）。

`tests/test_strategy_sweep.py` 用小 config 覆盖策略遍历（秒级、进常规集），本文件
用**真实 7B 权重**逐策略验证同一条链路——用户明确要求的判据不能用玩具模型代替。
每种策略一个独立用例（分钟级，按需单跑）：

    pytest tests/test_strategy_llama2_7b.py -q                      # 全部策略
    pytest tests/test_strategy_llama2_7b.py -q -k tp1_pp8           # 只跑纯流水

判据与既有 `tests/test_decode_loop_llama2_7b.py`（张量并行的验证 3）一致：

1. 贪心解码的 token 序列与 HF 原生 `use_cache=True` 自回归推进**逐 token 一致**；
2. KV 区里的真实字节与 HF `DynamicCache` 的 post-RoPE K/V 逐元素对齐——写对了图的
   输出和写对了 KV 区里的数据是两件独立的事。

流水策略下第 2 条尤其关键：它是「每台 DPU 只为自己那几层存 KV、handler 按层找对
了归属 DPU」这件事的直接证据。`make_sdpa_handler` 的 `dpu_of_head` 若不按层过滤，
后写的 stage 会覆盖前面的映射，读到别的 stage 的 KV 区——logits 仍然「算得出来」，
只有这条判据能抓住。
"""

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
from graph.strategy import llama_strategies
from memory.kv_layout import PIMStaticKVCache
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
PREFILL_SEQ_LEN = 16
DECODE_STEPS = 8
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


def _strategies():
    """7B / 8 DPU 的搜索空间：tp8、tp4×pp2、tp2×pp4、tp1×pp8。"""
    return llama_strategies(
        NUM_DPUS, num_heads=32, num_kv_heads=32, intermediate_size=11008,
        vocab_size=32000, num_layers=32,
    )


@pytest.fixture(scope="module")
def llama2_model():
    torch.set_grad_enabled(False)
    return LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()


@pytest.fixture(scope="module")
def hf_reference(llama2_model):
    """HF 单卡自回归贪心解码 + 最终 KV cache——全部策略共用的参考。"""
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
    """按策略名缓存整跑结果：一种策略只跑一次 7B 推理，两条判据共用。

    没有这层缓存，4 种策略 × 2 条判据 = 8 次完整 7B 推理（每次约 10 分钟）；
    有了它是 4 次。判据之间共享同一次运行的产物不影响独立性——两条判据检查的是
    同一次推理的不同侧面（图输出 vs KV 区字节）。
    """
    return {}


def _run_strategy_cached(cache, model, strategy):
    if strategy.name not in cache:
        cache[strategy.name] = _run_strategy(model, strategy)
    return cache[strategy.name]


def _run_strategy(model, strategy):
    """编译 + 载权重 + 跑完整 prefill/decode，返回 (生成序列, backend, compiled)。"""
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


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
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


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_real_llama2_7b_kv_region_matches_hf_cache_under_every_strategy(
    strategy, llama2_model, hf_reference, run_cache
) -> None:
    """判据 2：KV 区里的真实字节与 HF DynamicCache 的 post-RoPE K/V 对齐。

    容差用相对+绝对混合（fp16 的 ULP ≈ |x| × 2^-10，深层 K/V 幅值可到几十，固定
    绝对容差会误报），与 `test_decode_loop_llama2_7b.py` 同一套。
    """
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


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_real_llama2_7b_memory_footprint_fits_and_scales(strategy, llama2_model) -> None:
    """结构判据：真实 7B 尺度下每台 DPU 都装得进预算，且 KV 区随 stage 数缩小。

    这条是纯编译期的（不跑推理），单独成例是因为「装不装得下」在真机上是切分策略
    的首要约束：13GB 权重 8 台分摊后每台约 1.6GB，KV 区在流水下只留本 stage 的层。
    """
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
