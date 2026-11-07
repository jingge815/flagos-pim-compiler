"""验证 KV 块号以 vLLM 为唯一真源：新请求采用、cached 追加、resumed 替换。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
from runtime.executor import KVBlockTable, RequestTable, step_request
from runtime.kernels import register_all


def _worker(num_slots: int = 2) -> PIMWorker:
    """跳过 vLLM 初始化，只装配块表记账需要的字段。"""
    worker = object.__new__(PIMWorker)
    worker.table = RequestTable(num_slots)
    worker.eos_id = 2
    worker._samplers = {}
    worker._prompt_ids = {}
    worker._output_ids = {}
    return worker


def _new_req(req_id: str = "r1", blocks=(3,), prompt=(1, 2, 3, 4),
             computed: int = 0) -> SimpleNamespace:
    """伪造 NewRequestData；`block_ids` 与 vLLM 一样按 KV 分组放在元组里。"""
    return SimpleNamespace(
        req_id=req_id, prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
        block_ids=(list(blocks),), lora_request=None, mm_features=[],
        num_computed_tokens=computed,
    )


def _cached(req_ids, *, computed, new_blocks, resumed=()) -> SimpleNamespace:
    """伪造 CachedRequestData；`new_blocks` 每项按 KV 分组包一层元组，与 vLLM 一致。"""
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(req_ids), num_computed_tokens=list(computed),
            new_block_ids=[tuple(groups) for groups in new_blocks],
            resumed_req_ids=set(resumed),
            all_token_ids={req_id: [1, 2, 3, 4, 5] for req_id in req_ids},
        ))


def test_new_request_adopts_vllm_blocks() -> None:
    """新请求直接采用 vLLM 给的块号，不做任何重映射。"""
    worker = _worker()
    worker._add_request(_new_req(blocks=(3,)))
    assert worker.table.get("r1").blocks.slots == [3]


def test_cached_request_appends_new_blocks() -> None:
    """解码阶段新分配的块追加到块表尾部。"""
    worker = _worker()
    worker._add_request(_new_req(blocks=(3,)))
    worker._sync_cached(_cached(["r1"], computed=[4], new_blocks=[([7, 8],)]))
    assert worker.table.get("r1").blocks.slots == [3, 7, 8]


def test_resumed_request_replaces_blocks() -> None:
    """抢占恢复时 vLLM 重发整段块号，块表整段替换并重新 prefill。"""
    worker = _worker()
    worker._add_request(_new_req(blocks=(3,)))
    slot = worker.table.get("r1")
    slot.is_prefill_done = True
    slot.valid_len = 4
    worker._sync_cached(_cached(
        ["r1"], computed=[0], new_blocks=[([5, 6],)], resumed={"r1"}))
    assert slot.blocks.slots == [5, 6]
    assert not slot.is_prefill_done and slot.valid_len == 0


def test_finished_request_does_not_return_blocks() -> None:
    """请求结束后块号由 vLLM 收回，PIM 侧只清本地记录。"""
    worker = _worker()
    worker._add_request(_new_req(blocks=(3,)))
    worker.table.free_slot("r1")
    with pytest.raises(KeyError):
        worker.table.get("r1")


def test_external_block_table_rejects_short_list() -> None:
    """块号不够覆盖本次长度时立刻报错，不静默读到未分配块。"""
    blocks = KVBlockTable()
    blocks.adopt([3])
    blocks.ensure(1)
    with pytest.raises(ValueError, match="块表只有 1 项"):
        blocks.ensure(2)


def test_block_source_must_be_exactly_one() -> None:
    """自管池与外部块号二选一，两个都不给或都给都报错。"""
    with pytest.raises(ValueError, match="块号来源必须二选一"):
        RequestTable(1).add_request("r", [1])
    with pytest.raises(ValueError, match="块号来源必须二选一"):
        RequestTable(1, 4).add_request("r", [1], block_ids=[0])


MAX_SEQ = 16
PROMPT = [1, 2, 3, 4]
NEW_TOKENS = 3


@pytest.fixture(scope="module")
def compiled_and_backend():
    """微型 Llama 的编译产物：4 个物理块，够放乱序的外部块号。"""
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
        max_position_embeddings=MAX_SEQ,
    )).eval().half()
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=4, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    strategy = llama_strategies(
        4, num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128,
        num_layers=4)[0]
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=MAX_SEQ, max_seq=MAX_SEQ, hw=hw,
        hardware=hardware, num_request_slots=2, kv_slot_tokens=MAX_SEQ,
        kv_pool_slots=4)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=4, mram_bytes_per_dpu=hw.mram_bytes,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def _generate(compiled, backend, block_ids: list[int] | None) -> list[int]:
    """按给定块号来源（None = 自管池）跑一次 prefill + 解码。"""
    table = (RequestTable(2) if block_ids is not None
             else RequestTable(2, compiled.kv_pool_slots))
    slot = table.add_request("r", PROMPT, block_ids=block_ids)
    return [step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(NEW_TOKENS)]


def test_external_block_numbers_change_nothing(compiled_and_backend) -> None:
    """块号只是间接寻址：换成乱序的外部块号，生成 token 逐字不变。"""
    compiled, backend = compiled_and_backend
    pooled = _generate(compiled, backend, None)
    external = _generate(compiled, backend, [3])
    assert external == pooled
