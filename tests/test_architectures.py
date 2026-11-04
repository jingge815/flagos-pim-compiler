"""验证架构白名单：受支持的架构能编译运行，其余在启动期拒绝。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import (
    LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,
    Qwen3Config, Qwen3ForCausalLM)

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.architectures import ARCHITECTURES
from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from backend.vllm_worker import validate_engine_config
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import (
    RequestTable, prefill_chunk, run_decode_loop, run_one_step)
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
        validate_engine_config(**_configs(SimpleNamespace(model_type="gpt2")))

    assert "llama" in str(excinfo.value) and "qwen2" in str(excinfo.value)
    assert "qwen3" in str(excinfo.value)


def test_qwen2_config_passes() -> None:
    """Qwen2 配置默认带滑窗长度，`use_sliding_window=False` 时必须放行。"""
    hf_config = SimpleNamespace(
        model_type="qwen2", rope_scaling=None,
        sliding_window=4096, use_sliding_window=False)
    validate_engine_config(**_configs(hf_config))


def test_qwen2_sliding_window_still_rejected() -> None:
    """真的打开滑窗注意力时仍然拒绝。"""
    hf_config = SimpleNamespace(
        model_type="qwen2", rope_scaling=None,
        sliding_window=4096, use_sliding_window=True)
    with pytest.raises(ValueError, match="sliding_window"):
        validate_engine_config(**_configs(hf_config))


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
def gqa_model():
    """4 个 Q 头共用 2 个 KV 头的 Llama。"""
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        max_position_embeddings=MAX_SEQ, **MODEL_KWARGS)).eval().to(torch.float16)


def _compile_and_load(model, strategy, num_request_slots=2, num_parallel_slots=2):
    """编译 + 建后端 + 装权重，供各架构的数值测试共用。"""
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN, max_seq=MAX_SEQ,
        hw=hw, hardware=hardware, num_request_slots=num_request_slots,
        kv_slot_tokens=4, num_parallel_slots=num_parallel_slots)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


@pytest.fixture(scope="module")
def gqa_compiled_and_backend(gqa_model):
    """按 2 个 KV 头编译，4 台 DPU 分成 2 段、段内张量并行宽度 2。"""
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=2, intermediate_size=128,
        vocab_size=128, num_layers=NUM_LAYERS)[0]
    return _compile_and_load(gqa_model, strategy)


def _greedy(logits):
    return int(np.argmax(logits))


def test_gqa_kv_layout_keeps_two_heads(gqa_compiled_and_backend) -> None:
    """KV 区按 2 个 KV 头切分，每台 DPU 各拿一个。"""
    compiled, _ = gqa_compiled_and_backend
    heads = {head for spec in compiled.kv_specs.values() for head in spec.kv_heads}

    assert heads == {0, 1}


def test_gqa_prefill_matches_hf(gqa_model, gqa_compiled_and_backend) -> None:
    """GQA 的 PIM prefill logits 与 HF 对齐。

    一个 KV 头服务多个 Q 头，缓存行只能写一次；重复写会让后一组的 K/V
    覆盖前一组的缓存位置，logits 直接跑偏到另一个 token。
    """
    compiled, backend = gqa_compiled_and_backend
    prompt = PROMPTS[0]
    table = RequestTable(1, compiled.kv_pool_slots)
    slot = table.add_request("r0", prompt, eos_id=-1)
    got = np.asarray(prefill_chunk(
        compiled, backend, slot, prompt, 0, causal_mask_of))

    with torch.no_grad():
        ref = gqa_model(
            input_ids=torch.tensor([prompt]),
            attention_mask=causal_mask_of(len(prompt)),
            position_ids=torch.arange(len(prompt)).unsqueeze(0)).logits.numpy()
    assert np.abs(got - ref).max() < 1e-3


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


@pytest.fixture(scope="module")
def qwen2_model(tmp_path_factory):
    """带随机 q/k/v 偏置的小 Qwen2，存盘后按注册表路径加载。"""
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(Qwen2Config(
        max_position_embeddings=MAX_SEQ, **MODEL_KWARGS)).eval().to(torch.float16)
    # 默认初始化的偏置全为 0，显式随机化才能覆盖到偏置路径。
    with torch.no_grad():
        for layer in model.model.layers:
            for name in ("q_proj", "k_proj", "v_proj"):
                getattr(layer.self_attn, name).bias.normal_(0, 0.02)
    path = tmp_path_factory.mktemp("qwen2")
    model.save_pretrained(path)
    return model, path


def test_qwen2_generate_matches_hf(qwen2_model) -> None:
    """Qwen2 走注册表的加载与切分，贪心生成的 token 与 HF `generate` 逐字一致。"""
    model, path = qwen2_model
    loaded = ARCHITECTURES["qwen2"].load(str(path))
    strategy = ARCHITECTURES["qwen2"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(
        loaded, strategy, num_request_slots=1, num_parallel_slots=1)

    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()


@pytest.fixture(scope="module")
def qwen3_model(tmp_path_factory):
    """小 Qwen3：head_dim 显式且不等于 hidden/heads（Qwen3 的解耦特性），
    q/k 的 RMSNorm 权重显式随机化（默认全 1，随机化才能测到装载路径）。"""
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(Qwen3Config(
        max_position_embeddings=MAX_SEQ, hidden_size=64, intermediate_size=128,
        num_hidden_layers=NUM_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128,
    )).eval().to(torch.float16)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.q_norm.weight.normal_(1.0, 0.1)
            layer.self_attn.k_norm.weight.normal_(1.0, 0.1)
    path = tmp_path_factory.mktemp("qwen3")
    model.save_pretrained(path)
    return model, path


def test_qwen3_generate_matches_hf(qwen3_model) -> None:
    """Qwen3（qk-norm + 无偏置 qkv + 显式 head_dim）贪心生成与 HF 逐 token 一致。"""
    model, path = qwen3_model
    assert model.config.head_dim != model.config.hidden_size // model.config.num_attention_heads

    loaded = ARCHITECTURES["qwen3"].load(str(path))
    strategy = ARCHITECTURES["qwen3"].strategy(NUM_DPUS, loaded.config)
    compiled, backend = _compile_and_load(
        loaded, strategy, num_request_slots=1, num_parallel_slots=1)

    prompt = torch.tensor(PROMPTS[0])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=MAX_NEW_TOKENS,
        eos_id=-1, sample_fn=_greedy, causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=MAX_NEW_TOKENS,
                         do_sample=False)
    assert got == ref[0][len(PROMPTS[0]):].tolist()
