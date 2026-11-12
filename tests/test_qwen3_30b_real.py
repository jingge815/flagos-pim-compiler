"""真实 Qwen3-30B-A3B（MoE）全链路验证：注册表加载 → 切分编译 → 贪心生成对拍 HF。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.architectures import ARCHITECTURES, restore_hf_moe_forward
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from genesim_bridge.paths import qwen3_30b_model_dir
from memory.mem_planner import default_site_capacity
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = qwen3_30b_model_dir(required=False)
NUM_DPUS = 8
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 6

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 qwen3_30b_model_dir",
)


@pytest.fixture(scope="module")
def compiled_and_backend():
    """8 DPU 下首个可行策略是 tp4_pp2（4 个 KV 头挡掉 tp8）。"""
    loaded = ARCHITECTURES["qwen3_moe"].load(str(MODEL_DIR))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, loaded.config)
    assert strategy.name == "tp4_pp2"
    # 需求 §2.1 P0-3：8 DPU × 8 GiB 是硬约束，测试与主链共用同一份契约；
    # 容量三件套（预算 + 硬件 + 对齐）只从同一工厂派生。
    hw, hardware = default_site_capacity(
        num_dpus=NUM_DPUS, num_tasklets=4)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=8, max_seq=32,
        hw=hw, hardware=hardware, num_request_slots=1, num_parallel_slots=1)
    # MoE 默认不做图级分块：48 层 × 128 专家的图应停在数万节点量级，
    # 而不是被专家 linear 的分块子节点放大两个数量级。
    node_count = len(list(compiled.decode_gm.graph.nodes))
    assert node_count < 200_000, (
        f"decode 图有 {node_count} 个节点，MoE 的图级分块默认没有关闭")
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend)
    load_weights(compiled, backend)
    # 编译之后把 MoE 前向恢复成 HF 原始稀疏实现：参考必须独立于静态改写，
    # 否则两边共享同一套路由/溢出/归一化假设，发现不了改写自身的错误。
    restored = restore_hf_moe_forward(loaded)
    assert restored == loaded.config.num_hidden_layers
    for layer in loaded.model.layers:
        assert not getattr(layer.mlp.forward, "_static_moe", False), (
            "参考路径仍在用静态改写前向")
    return loaded, compiled, backend


def test_qwen3_30b_generate_matches_hf(compiled_and_backend) -> None:
    """真实 30B-A3B（48 层 × 128 专家）贪心生成与 HF `generate` 逐 token 一致。"""
    loaded, compiled, backend = compiled_and_backend
    assert loaded.config.num_experts == 128
    assert loaded.config.num_experts_per_tok == 8

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
