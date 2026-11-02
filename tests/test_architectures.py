"""验证架构白名单：受支持的架构能编译运行，其余在启动期拒绝。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from backend.vllm_worker import validate_engine_config
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, run_decode_loop, run_one_step
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
MAX_SEQ = 16
MAX_NEW_TOKENS = 3
PROMPTS = [[1, 2, 3, 4], [5, 6, 7, 8]]
# GQA：4 个 Q 头共用 2 个 KV 头。
MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
)


def _configs(hf_config: SimpleNamespace) -> dict:
    """构造一份全绿的能力矩阵配置，只替换 hf_config。"""
    return {
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1),
        "lora_config": None,
        "speculative_config": None,
        "model_config": SimpleNamespace(
            dtype=torch.float16, quantization=None, hf_config=hf_config),
        "scheduler_config": SimpleNamespace(
            enable_chunked_prefill=True, max_num_batched_tokens=PREFILL_SEQ_LEN),
        "prefill_seq_len": PREFILL_SEQ_LEN,
    }


def test_unknown_architecture_lists_supported() -> None:
    """白名单之外报错，并列出受支持的架构。"""
    with pytest.raises(ValueError, match="model_type") as excinfo:
        validate_engine_config(**_configs(SimpleNamespace(model_type="qwen2")))

    assert "llama" in str(excinfo.value)


def test_rope_scaling_rejected() -> None:
    """位置插值/缩放会算出不同的位置编码，启动期直接拒绝。"""
    hf_config = SimpleNamespace(model_type="llama", rope_scaling={"type": "linear"})
    with pytest.raises(ValueError, match="rope_scaling"):
        validate_engine_config(**_configs(hf_config))


def test_sliding_window_rejected() -> None:
    """滑窗注意力不是全注意力，启动期直接拒绝。"""
    hf_config = SimpleNamespace(model_type="llama", sliding_window=4096)
    with pytest.raises(ValueError, match="sliding_window"):
        validate_engine_config(**_configs(hf_config))


@pytest.fixture(scope="module")
def gqa_compiled_and_backend():
    """按 2 个 KV 头编译，4 台 DPU 分成 2 段、段内张量并行宽度 2。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        max_position_embeddings=MAX_SEQ, **MODEL_KWARGS)).eval().to(torch.float16)
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=2, intermediate_size=128,
        vocab_size=128, num_layers=NUM_LAYERS)[0]
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=2, kv_slot_tokens=4,
        num_parallel_slots=2)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _greedy(logits):
    return int(np.argmax(logits))


def test_gqa_kv_layout_keeps_two_heads(gqa_compiled_and_backend) -> None:
    """KV 区按 2 个 KV 头切分，每台 DPU 各拿一个。"""
    compiled, _ = gqa_compiled_and_backend
    heads = {head for spec in compiled.kv_specs.values() for head in spec.kv_heads}

    assert heads == {0, 1}


def test_gqa_two_requests_match_single_runs(gqa_compiled_and_backend) -> None:
    """GQA 模型两请求并发，生成 token 与各自单独运行一致。"""
    compiled, backend = gqa_compiled_and_backend
    alone = [
        run_decode_loop(compiled, backend, prompt_ids=torch.tensor(prompt),
                        max_new_tokens=MAX_NEW_TOKENS, eos_id=-1,
                        sample_fn=_greedy, causal_mask_of=causal_mask_of)
        for prompt in PROMPTS
    ]

    table = RequestTable(2, compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS):
        table.add_request(f"r{index}", prompt, eos_id=-1)
    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    got = [list(table.get(f"r{index}").generated) for index in range(2)]
    assert got == alone
