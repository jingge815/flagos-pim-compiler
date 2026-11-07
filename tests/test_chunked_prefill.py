"""验证分块 prefill：分块结果与整段逐 token 一致，中间段不采样。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from vllm import SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, prefill_bucket, prefill_chunk, step_request
from runtime.kernels import register_all

NUM_DPUS = 4
MAX_SEQ = 16
BUCKETS = (2, 4, 8)
NEW_TOKENS = 3
LONG_SEQ = 3072
LONG_BUCKETS = (64, 512, 2048)
LONG_PROMPT_LEN = 3000


def _build(max_seq: int, buckets: tuple[int, ...], kv_slot_tokens: int,
           kv_pool_slots: int):
    """编译一个微型 Llama 并装好后端，供同模块的用例共用。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=max_seq,
    )).eval().half()
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=4, intermediate_size=128,
        vocab_size=128, num_layers=4)[0]
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=max_seq, max_seq=max_seq, hw=hw,
        hardware=hardware, num_request_slots=2, kv_slot_tokens=kv_slot_tokens,
        kv_pool_slots=kv_pool_slots, prefill_seq_lens=buckets)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


@pytest.fixture(scope="module")
def compiled_and_backend():
    """短句用的编译产物：桶 (2, 4, 8, 16)，KV 每块 4 个 token。"""
    return _build(MAX_SEQ, BUCKETS, kv_slot_tokens=4, kv_pool_slots=8)


@pytest.fixture(scope="module")
def long_compiled_and_backend():
    """长句用的编译产物：桶 (64, 512, 2048, 3072)，KV 每块 128 个 token。"""
    return _build(LONG_SEQ, LONG_BUCKETS, kv_slot_tokens=128, kv_pool_slots=32)


def _whole(compiled, backend, prompt: list[int]) -> list[int]:
    """一次跑完整段 prompt，再解码，返回生成的 token。"""
    table = RequestTable(2, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt)
    return [step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(NEW_TOKENS)]


def _chunked(compiled, backend, prompt: list[int], chunk: int) -> list[int]:
    """按 chunk 切段跑完 prefill，再用解码续上，返回生成的 token。"""
    table = RequestTable(2, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt)
    token = None
    for pos in range(0, len(prompt), chunk):
        part = prompt[pos:pos + chunk]
        logits = prefill_chunk(
            compiled, backend, slot, part, pos, causal_mask_of)
        token = int(np.argmax(logits[0, len(part) - 1]))
    slot.is_prefill_done = True
    slot.generated.append(token)
    return [token] + [
        step_request(compiled, backend, slot, causal_mask_of)
        for _ in range(NEW_TOKENS - 1)]


@pytest.mark.parametrize("prompt_len, chunk", [(8, 2), (8, 4), (6, 4), (5, 1), (9, 8)])
def test_chunked_prefill_matches_whole(prompt_len: int, chunk: int,
                                       compiled_and_backend) -> None:
    """任意切法的分块 prefill 与整段 prefill 生成逐 token 一致。"""
    compiled, backend = compiled_and_backend
    prompt = [1 + index for index in range(prompt_len)]
    assert _chunked(compiled, backend, prompt, chunk) == _whole(
        compiled, backend, prompt)


def test_chunked_matches_whole_on_host_path(monkeypatch, compiled_and_backend) -> None:
    """强制走主机两遍路径时，分块 prefill 同样与整段一致。"""
    import runtime.kernels as km

    compiled, backend = compiled_and_backend
    monkeypatch.setattr(km, "_compiled_paged_attn", lambda *args, **kwargs: None)
    prompt = [1, 2, 3, 4, 5, 6]
    assert _chunked(compiled, backend, prompt, 4) == _whole(
        compiled, backend, prompt)


@pytest.mark.slow
@pytest.mark.parametrize("chunk", [64, 512])
def test_long_prompt_chunked_matches_whole(chunk: int,
                                           long_compiled_and_backend) -> None:
    """3000 token 的 prompt 按 64 / 512 切块，与整段 prefill 逐 token 一致。"""
    compiled, backend = long_compiled_and_backend
    prompt = [1 + index % 100 for index in range(LONG_PROMPT_LEN)]
    assert _chunked(compiled, backend, prompt, chunk) == _whole(
        compiled, backend, prompt)


def test_interleaved_chunks_do_not_disturb_each_other(compiled_and_backend) -> None:
    """两个请求交错分块 prefill，各自输出与单独运行一致。"""
    compiled, backend = compiled_and_backend
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    alone = [_chunked(compiled, backend, prompt, 4) for prompt in prompts]

    table = RequestTable(2, compiled.kv_pool_slots)
    slots = [table.add_request(f"r{index}", prompt)
             for index, prompt in enumerate(prompts)]
    first: list[int] = [0, 0]
    for pos in (0, 4):
        for index, slot in enumerate(slots):
            part = slot.prompt_ids[pos:pos + 4]
            logits = prefill_chunk(compiled, backend, slot, part, pos,
                                   causal_mask_of, slot_index=index)
            first[index] = int(np.argmax(logits[0, len(part) - 1]))
    for index, slot in enumerate(slots):
        slot.is_prefill_done = True
        slot.generated.append(first[index])

    mixed = [[tokens] for tokens in first]
    for _ in range(NEW_TOKENS - 1):
        for index, slot in enumerate(slots):
            mixed[index].append(step_request(
                compiled, backend, slot, causal_mask_of, slot_index=index))
    assert mixed == alone


def test_interim_chunk_leaves_prefill_open(compiled_and_backend) -> None:
    """中间段只写 KV：不置 prefill 完成，KV 有效长度停在本段末尾。"""
    compiled, backend = compiled_and_backend
    table = RequestTable(2, compiled.kv_pool_slots)
    slot = table.add_request("r", [1, 2, 3, 4, 5, 6])
    logits = prefill_chunk(compiled, backend, slot, [1, 2, 3, 4], 0, causal_mask_of)
    assert slot.valid_len == 4 and not slot.is_prefill_done
    assert logits.shape == (1, prefill_bucket(compiled, 4), 128)


def test_chunks_write_kv_at_absolute_positions(compiled_and_backend) -> None:
    """两段都落在同一块 KV 里时，第二段按绝对位置命中第一段的历史。"""
    compiled, backend = compiled_and_backend
    prompt = [1, 2, 3, 4, 5, 6]
    table = RequestTable(2, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt)
    prefill_chunk(compiled, backend, slot, prompt[:4], 0, causal_mask_of)
    prefill_chunk(compiled, backend, slot, prompt[4:], 4, causal_mask_of)
    # 6 个 token 用两块；第一段已写满第一块，第二段从第二块开头接上。
    assert len(slot.blocks.slots) == 2
    assert slot.valid_len == 6


def test_vllm_engine_chunks_long_prompt(tmp_path, monkeypatch) -> None:
    """预算小于 prompt 长度时 vLLM 自动分块，生成结果与不分块一致。"""
    from backend.vllm_executor import PIMExecutor
    from vllm import LLM

    # 其它测试模块会在导入时设 PIM_PREFILL_SEQ_LEN，这里显式钉住图长。
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(MAX_SEQ))

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=MAX_SEQ)
    model_dir = tmp_path / "tiny-llama"
    LlamaForCausalLM(config).eval().half().save_pretrained(model_dir)

    def _run(budget: int) -> list[list[int]]:
        llm = LLM(
            model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
            max_model_len=MAX_SEQ, max_num_seqs=2,
            max_num_batched_tokens=budget, enable_prefix_caching=False,
            gpu_memory_utilization=0.2, distributed_executor_backend=PIMExecutor)
        outputs = llm.generate(
            [{"prompt_token_ids": [1, 2, 3, 4, 5, 6]}],
            SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0))
        return [list(output.outputs[0].token_ids) for output in outputs]

    assert _run(4) == _run(MAX_SEQ)
