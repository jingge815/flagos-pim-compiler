"""验证小 MoE 在动态形状绑定下与静态基线逐 token 一致。

4L 真实 MoE 暴露过“融合节点丢表达式 / T 在中间维”两类问题；这里用
随机小模型把注意力、路由分派和形状绑定串起来回归。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, step_request
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 2
MAX_SEQ = 16
PREFILL_SEQ_LEN = 8
MAX_NEW_TOKENS = 3

MODEL_CONFIG = Qwen3MoeConfig(
    max_position_embeddings=MAX_SEQ, hidden_size=64,
    moe_intermediate_size=32, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    vocab_size=128, num_experts=4, num_experts_per_tok=2,
    norm_topk_prob=True)


def _compile(model, dynamic: bool):
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = ARCHITECTURES["qwen3_moe"].strategy(NUM_DPUS, model.config)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=1,
        kv_slot_tokens=MAX_SEQ, dynamic=dynamic)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _tokens(compiled, backend, prompt: list[int]) -> list[int]:
    table = RequestTable(compiled.num_request_slots, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt, eos_id=-1)
    return [
        step_request(compiled, backend, slot, causal_mask_of)
        for _ in range(MAX_NEW_TOKENS)
    ]


@pytest.fixture(scope="module")
def moe_model():
    torch.manual_seed(3)
    return Qwen3MoeForCausalLM(MODEL_CONFIG).eval().to(torch.float16)


@pytest.fixture(scope="module")
def baselines(moe_model, tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe_dyn")
    moe_model.save_pretrained(path)
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    static = _compile(loaded, dynamic=False)
    dynamic = _compile(loaded, dynamic=True)
    return static, dynamic


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 5, 7])
def test_dynamic_moe_tokens_match_static(prompt_len, baselines, monkeypatch) -> None:
    """动态绑定下 prefill/decode 的 token 与静态桶基线一致。"""
    monkeypatch.setenv("PIM_DYN_SHAPE", "1")
    (static_c, static_b), (dyn_c, dyn_b) = baselines
    prompt = [1 + i for i in range(prompt_len)]
    assert _tokens(dyn_c, dyn_b, prompt) == _tokens(static_c, static_b, prompt)
