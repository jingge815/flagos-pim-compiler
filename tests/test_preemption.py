"""验证抢占恢复：整段重放与命中缓存的续算都与不抢占基线一致。"""

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
from runtime.executor import RequestTable, prefill_chunk, step_request
from runtime.kernels import register_all

NUM_DPUS = 4
MAX_SEQ = 16
BUCKETS = (2, 4, 8)
KV_SLOT_TOKENS = 4
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
NEW_TOKENS = 3


def _worker(num_slots: int = 2) -> PIMWorker:
    """跳过 vLLM 初始化，只装配块表记账需要的字段。"""
    worker = object.__new__(PIMWorker)
    worker.table = RequestTable(num_slots)
    worker.eos_id = 2
    worker._samplers = {}
    worker._prompt_ids = {}
    worker._output_ids = {}
    return worker


def _new_req() -> SimpleNamespace:
    """伪造一条新请求。"""
    return SimpleNamespace(
        req_id="r1", prompt_token_ids=list(PROMPT),
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        block_ids=([0, 1, 2],), lora_request=None, mm_features=[],
        num_computed_tokens=0,
    )


def _cached(computed: int, blocks, tokens, *, resumed: bool):
    """伪造一条 cached 请求；抢占恢复时 vLLM 会带全量 token 与新块号。"""
    return SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(
        req_ids=["r1"], num_computed_tokens=[computed],
        new_block_ids=[tuple(blocks)], resumed_req_ids={"r1"} if resumed else set(),
        all_token_ids={"r1": list(tokens)}))


def test_preempted_mid_prefill_restarts_from_zero() -> None:
    """prefill 没跑完就被抢占：块表按新块号整段替换，从头重算。"""
    worker = _worker()
    worker._add_request(_new_req())
    slot = worker.table.get("r1")
    slot.valid_len = 4

    worker._sync_cached(_cached(0, [[5, 6, 7]], PROMPT, resumed=True))
    assert slot.blocks.slots == [5, 6, 7]
    assert slot.valid_len == 0 and not slot.is_prefill_done
    assert slot.prompt_ids == PROMPT


def test_resumed_with_cached_prefix_keeps_computed_length() -> None:
    """恢复时命中 8 个 token 的前缀缓存：从第 8 个位置接着算，不重跑前缀。"""
    worker = _worker()
    worker._add_request(_new_req())
    slot = worker.table.get("r1")
    slot.is_prefill_done = True
    slot.valid_len = len(PROMPT)

    worker._sync_cached(_cached(8, [[5, 6, 7]], PROMPT, resumed=True))
    assert slot.blocks.slots == [5, 6, 7]
    assert slot.valid_len == 8 and not slot.is_prefill_done


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
        hardware=hardware, num_request_slots=2, kv_slot_tokens=KV_SLOT_TOKENS,
        kv_pool_slots=8, prefill_seq_lens=BUCKETS)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _prefill_range(compiled, backend, slot, start: int, end: int) -> int:
    """把 `[start, end)` 写进 KV（与 worker 的分块 prefill 一致），返回末行采样 token。"""
    token = 0
    for pos in range(start, end, KV_SLOT_TOKENS):
        part = slot.prompt_ids[pos:min(pos + KV_SLOT_TOKENS, end)]
        logits = prefill_chunk(compiled, backend, slot, part, pos, causal_mask_of)
        token = int(np.argmax(logits[0, len(part) - 1]))
    slot.generated.append(token)
    return token


def _decode(compiled, backend, slot, count: int) -> list[int]:
    """解码若干个 token，返回新采出的序列。"""
    return [step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(count)]


def _baseline(compiled, backend) -> list[int]:
    """不抢占的参照：整段 prefill 后解码。"""
    table = RequestTable(2, compiled.kv_pool_slots)
    slot = table.add_request("r", PROMPT)
    return _decode(compiled, backend, slot, NEW_TOKENS)


def test_preempted_decode_replays_and_matches_baseline(compiled_and_backend) -> None:
    """解码中途被抢占：按全量 token 重放 prefill，生成 token 与基线一致。"""
    compiled, backend = compiled_and_backend
    baseline = _baseline(compiled, backend)

    table = RequestTable(2)
    slot = table.add_request("r", PROMPT, block_ids=[4, 5, 6, 7])
    first = _prefill_range(compiled, backend, slot, 0, len(PROMPT))
    slot.is_prefill_done = True

    # 抢占：vLLM 收回块，恢复时给新块号并重发全量 token（prompt + 已生成的那个）。
    # 重放的末行是已生成的 token 本身，它采出的就是下一个 token。
    slot.reset(PROMPT + [first], 0)
    slot.blocks.adopt([4, 5, 6, 7], replace=True)
    replayed = _prefill_range(compiled, backend, slot, 0, len(PROMPT) + 1)
    slot.is_prefill_done = True
    rest = _decode(compiled, backend, slot, NEW_TOKENS - 2)
    assert [first, replayed, *rest] == baseline


def test_preempted_mid_prefill_replays_and_matches_baseline(
        compiled_and_backend) -> None:
    """prefill 只算了一半就被抢占：整段重放后生成 token 与基线一致。"""
    compiled, backend = compiled_and_backend
    baseline = _baseline(compiled, backend)

    table = RequestTable(2)
    slot = table.add_request("r", PROMPT, block_ids=[4, 5, 6, 7])
    _prefill_range(compiled, backend, slot, 0, 6)

    # 抢占时 KV 全被收回，vLLM 重发整段 prompt，只能从头重算。
    slot.reset(PROMPT, 0)
    slot.blocks.adopt([4, 5, 6, 7], replace=True)
    first = _prefill_range(compiled, backend, slot, 0, len(PROMPT))
    slot.is_prefill_done = True
    assert [first, *_decode(compiled, backend, slot, NEW_TOKENS - 1)] == baseline


def test_resume_from_shared_prefix_matches_baseline(compiled_and_backend) -> None:
    """恢复时命中 8 token 的前缀缓存：只算剩下半段，生成 token 与基线一致。"""
    compiled, backend = compiled_and_backend
    baseline = _baseline(compiled, backend)
    # 前半段的 KV 由上一次全量 prefill 写进块 0/1，模拟共享块。
    _baseline(compiled, backend)

    table = RequestTable(2)
    slot = table.add_request("r", PROMPT, block_ids=[0, 1, 2, 3])
    slot.reset(PROMPT, 8)
    slot.blocks.adopt([0, 1, 2, 3], replace=True)
    token = _prefill_range(compiled, backend, slot, 8, len(PROMPT))
    slot.is_prefill_done = True
    assert [token, *_decode(compiled, backend, slot, NEW_TOKENS - 1)] == baseline


def test_vllm_engine_preempts_and_matches_baseline(tmp_path, monkeypatch) -> None:
    """槽池只够一个半请求时 vLLM 必然抢占，两请求输出与不抢占时逐 token 一致。"""
    from backend.vllm_executor import PIMExecutor
    from vllm import LLM

    e2e_seq = 32
    monkeypatch.setenv("PIM_PREFILL_SEQ_LEN", str(e2e_seq))
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=e2e_seq)
    model_dir = tmp_path / "tiny-llama"
    LlamaForCausalLM(config).eval().half().save_pretrained(model_dir)
    prompts = [{"prompt_token_ids": [index % 100 + 1 for index in range(12)]},
               {"prompt_token_ids": [index % 100 + 1 for index in range(12)]}]
    params = SamplingParams(max_tokens=6, temperature=0.0)

    def _run(pool: int, *, with_metrics: bool) -> tuple[list[list[int]], int]:
        # 块大小 16、max_model_len 32：每条请求最终占 2 块，池 3 块装不下
        # 两条，必然抢占；池 8 块时不抢。
        monkeypatch.setenv("PIM_KV_POOL_SLOTS", str(pool))
        llm = LLM(
            model=str(model_dir), skip_tokenizer_init=True, dtype="float16",
            max_model_len=e2e_seq, max_num_seqs=2,
            max_num_batched_tokens=e2e_seq, enable_prefix_caching=False,
            gpu_memory_utilization=0.2, distributed_executor_backend=PIMExecutor,
            disable_log_stats=not with_metrics)
        outputs = llm.generate(prompts, params)
        preempted = sum(
            metric.value for metric in llm.get_metrics()
            if metric.name == "vllm:num_preemptions") if with_metrics else 0
        return [list(output.outputs[0].token_ids) for output in outputs], preempted

    tight, preempted = _run(pool=3, with_metrics=True)
    roomy, _ = _run(pool=8, with_metrics=False)
    assert preempted > 0, "槽池紧张时没有发生抢占，用例没覆盖到恢复路径"
    assert tight == roomy
