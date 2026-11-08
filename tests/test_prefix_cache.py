"""验证前缀缓存：命中段只读，新 token 只写私有块，输出与不共享时一致。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from vllm import SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from backend.vllm_worker import PIMWorker
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, prefill_chunk
from runtime.kernels import register_all

NUM_DPUS = 4
MAX_SEQ = 16
BUCKETS = (2, 4, 8)
KV_SLOT_TOKENS = 4
PREFIX_LEN = 8
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def _worker(num_slots: int = 2) -> PIMWorker:
    """跳过 vLLM 初始化，只装配块表记账需要的字段。"""
    worker = object.__new__(PIMWorker)
    worker.table = RequestTable(num_slots)
    worker.eos_id = 2
    worker._samplers = {}
    worker._prompt_ids = {}
    worker._output_ids = {}
    return worker


def _new_req(num_computed_tokens: int) -> SimpleNamespace:
    """伪造一条命中前缀缓存的新请求。"""
    return SimpleNamespace(
        req_id="r1", prompt_token_ids=list(PROMPT),
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        block_ids=([0, 1, 2],), lora_request=None, mm_features=[],
        num_computed_tokens=num_computed_tokens,
    )


def test_new_request_starts_after_cached_prefix() -> None:
    """命中 8 个 token 时，本请求从第 8 个位置开始写，不重算已缓存的段。"""
    worker = _worker()
    worker._add_request(_new_req(num_computed_tokens=PREFIX_LEN))
    assert worker.table.get("r1").valid_len == PREFIX_LEN


@pytest.fixture(scope="module")
def compiled_and_backend():
    """微型 Llama：KV 每块 4 个 token，8 个物理块。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=MAX_SEQ,
    )).eval().half()
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = llama_strategies(
        NUM_DPUS, num_heads=4, num_kv_heads=4, intermediate_size=128,
        vocab_size=128, num_layers=4)[0]
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=MAX_SEQ, max_seq=MAX_SEQ, hw=hw,
        hardware=hardware, num_request_slots=3, kv_slot_tokens=KV_SLOT_TOKENS,
        kv_pool_slots=8, prefill_seq_lens=BUCKETS)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _slot_snapshot(compiled, backend, slots: list[int]) -> dict[int, np.ndarray]:
    """按物理槽号读回每台 DPU 上的 KV 字节，用于比较内容是否被改写。"""
    return {
        spec.dpu_id: np.concatenate([
            backend.read_local(
                spec.dpu_id, spec.slots_base + slot * spec.slot_bytes,
                (spec.slot_bytes,), np.uint8)
            for slot in slots])
        for spec in compiled.kv_specs.values()
    }


def _prefill(compiled, backend, prompt: list[int], pos: int = 0,
             block_ids: list[int] | None = None, slot_index: int = 0):
    """建一个请求槽，从 pos 起把剩余 token 写进 KV，返回首个采样 token。"""
    table = (RequestTable(3) if block_ids is not None
             else RequestTable(3, compiled.kv_pool_slots))
    slot = table.add_request("r", prompt, block_ids=block_ids)
    slot.valid_len = pos
    part = prompt[pos:]
    logits = prefill_chunk(
        compiled, backend, slot, part, pos, causal_mask_of, slot_index=slot_index)
    return table, slot, int(np.argmax(logits[0, len(part) - 1]))


def test_shared_prefix_blocks_are_not_rewritten(compiled_and_backend) -> None:
    """命中段所在的两个块内容不变，只有私有尾块被写。"""
    compiled, backend = compiled_and_backend
    _table, _slot, reference = _prefill(compiled, backend, PROMPT)
    # 前两块的 KV 与上面这次全量 prefill 写入的完全一致（同一段前缀）。
    before = _slot_snapshot(compiled, backend, [0, 1])

    _table, _slot, cached = _prefill(
        compiled, backend, PROMPT, pos=PREFIX_LEN, block_ids=[0, 1, 2])
    assert cached == reference, "命中前缀后的生成 token 与全量 prefill 不一致"
    after = _slot_snapshot(compiled, backend, [0, 1])
    for dpu_id, chunk in before.items():
        assert np.array_equal(chunk, after[dpu_id]), f"dpu{dpu_id} 改写了共享前缀块"


def test_shared_prefix_with_two_windows(compiled_and_backend) -> None:
    """三个请求共享前缀、两个并发窗口时，各自输出与单独全量 prefill 一致。"""
    compiled, backend = compiled_and_backend
    tails = {"a": PROMPT, "b": PROMPT, "c": PROMPT}
    alone = {name: _prefill(compiled, backend, prompt)[2]
             for name, prompt in tails.items()}

    # 三条请求共用块 0/1（前 8 个 token），尾块各分一块。
    table = RequestTable(3)
    slots = [
        table.add_request(name, prompt, block_ids=[0, 1, 2 + index])
        for index, (name, prompt) in enumerate(tails.items())
    ]
    for index, slot in enumerate(slots):
        slot.valid_len = PREFIX_LEN
    got = {}
    for index, (name, slot) in enumerate(zip(tails, slots)):
        logits = prefill_chunk(
            compiled, backend, slot, tails[name][PREFIX_LEN:], PREFIX_LEN,
            causal_mask_of, slot_index=index % 2)
        got[name] = int(np.argmax(logits[0, len(tails[name]) - PREFIX_LEN - 1]))
    assert got == alone


def test_vllm_engine_reuses_cached_prefix(tmp_path, monkeypatch) -> None:
    """vLLM 开启前缀缓存：第二次相同 prompt 命中缓存，生成 token 与关闭时一致。"""
    from backend.vllm_executor import PIMExecutor
    from vllm import LLM

    # 前缀缓存只缓存整块，且 vLLM 总要重算最后一个 token，所以 prompt
    # 至少要有一个整块再加一个 token（块大小 16）。
    e2e_seq = 48
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(e2e_seq))
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=e2e_seq)
    model_dir = tmp_path / "tiny-llama"
    LlamaForCausalLM(config).eval().half().save_pretrained(model_dir)
    prompt = {"prompt_token_ids": [index % 100 + 1 for index in range(20)]}
    params = SamplingParams(max_tokens=3, temperature=0.0)

    def _run(prefix_caching: bool) -> tuple[list[list[int]], int]:
        llm = LLM(
            model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
            block_size=16, max_model_len=e2e_seq, max_num_seqs=2,
            max_num_batched_tokens=e2e_seq,
            enable_prefix_caching=prefix_caching,
            gpu_memory_utilization=0.2, distributed_executor_backend=PIMExecutor)
        first = llm.generate([prompt], params)[0]
        second = llm.generate([prompt], params)[0]
        return ([list(first.outputs[0].token_ids),
                 list(second.outputs[0].token_ids)], second.num_cached_tokens)

    cached_tokens_out, hits = _run(True)
    plain_tokens, _ = _run(False)
    assert hits > 0, "vLLM 没有命中前缀缓存，用例没覆盖到共享块"
    assert cached_tokens_out == plain_tokens
