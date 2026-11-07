"""真实 Qwen3-8B 全链路验证：注册表加载 → 切分编译 → 贪心生成对拍 HF。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from genesim_bridge.paths import qwen3_8b_model_dir
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = qwen3_8b_model_dir(required=False)
NUM_DPUS = 8
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 6

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 qwen3_8b_model_dir",
)


@pytest.fixture(scope="module")
def compiled_and_backend():
    """走 vLLM 同款注册表路径：按 model_type 选加载与切分。"""
    loaded = ARCHITECTURES["qwen3"].load(str(MODEL_DIR))
    strategy = ARCHITECTURES["qwen3"].strategy(NUM_DPUS, loaded.config)
    hw = HwBudget(mram_bytes=8 * 2**30, align=64, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=8, max_seq=32,
        hw=hw, hardware=hardware, num_request_slots=1, num_parallel_slots=1)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend)
    load_weights(compiled, backend)
    return loaded, compiled, backend


def test_qwen3_8b_generate_matches_hf(compiled_and_backend) -> None:
    """贪心生成与 HF `generate` 逐 token 一致。"""
    loaded, compiled, backend = compiled_and_backend
    # 8B 恰好 head_dim == hidden/heads；解耦特性（两者不等）由小模型用例锁定。
    assert loaded.config.head_dim == 128

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]

    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)
    ref = loaded.generate(
        prompt_ids.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    assert got == ref[0][len(prompt_ids):].tolist()
    text = tokenizer.decode(prompt_ids.tolist() + list(got), skip_special_tokens=True)
    assert text.startswith(PROMPT)
