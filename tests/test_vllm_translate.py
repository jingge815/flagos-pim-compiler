"""验证 execute_model 的翻译层：请求注册、块号刷新、结束清理、wave 分配与报错。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.vllm_worker import PIMWorker
from runtime.executor import RequestTable


def _worker(*, num_slots: int = 4, parallel_slots: int = 2) -> PIMWorker:
    """跳过 vLLM 初始化，只装配翻译层用到的字段。"""
    worker = object.__new__(PIMWorker)
    worker.table = RequestTable(num_slots)
    worker.eos_id = 2
    worker._samplers = {}
    worker._prompt_ids = {}
    worker._output_ids = {}
    worker.num_parallel_slots = parallel_slots
    worker._executor = None
    worker.stats = None
    worker.stats_interval = 0
    return worker


def _new_req(req_id: str, blocks=(3,), prompt=(1, 2, 3, 4),
             computed: int = 0) -> SimpleNamespace:
    """伪造 NewRequestData；`block_ids` 与 vLLM 一样按 KV 分组放在元组里。"""
    return SimpleNamespace(
        req_id=req_id, prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=2, temperature=0.0),
        block_ids=(list(blocks),), lora_request=None, mm_features=[],
        num_computed_tokens=computed,
    )


def _cached(req_ids=(), *, computed=(), new_blocks=(), resumed=(),
            all_token_ids=None) -> SimpleNamespace:
    """伪造 CachedRequestData；`all_token_ids` 是请求编号到整段 token 的字典。"""
    return SimpleNamespace(
        req_ids=list(req_ids), num_computed_tokens=list(computed),
        new_block_ids=[tuple(groups) for groups in new_blocks],
        resumed_req_ids=set(resumed),
        all_token_ids=dict(all_token_ids or {}),
    )


def _output(*, new=(), cached=None, finished=(), num_tokens,
            structured: bool = False) -> SimpleNamespace:
    """伪造 SchedulerOutput；只填 execute_model 会读的字段。"""
    return SimpleNamespace(
        finished_req_ids=set(finished),
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached or _cached(),
        num_scheduled_tokens=dict(num_tokens),
        has_structured_output_requests=structured,
        pending_structured_output_tokens=False,
    )


def _fake_step(tokens: dict[str, int | None], calls: list[tuple[str, int]]):
    """把 `_step` 换成按请求编号出 token 的桩，并记下每次用的窗口号。"""

    def step(self, req_id, slot_index=0, logprobs_width=None, scheduled=None):
        calls.append((req_id, slot_index))
        token = tokens.get(req_id)
        return token, None

    return step


def test_new_request_registers_and_returns_token(monkeypatch) -> None:
    """新请求按 vLLM 给的块号注册，本步返回它采到的 token。"""
    worker = _worker()
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({"r1": 5}, []))

    out = worker.execute_model(_output(
        new=[_new_req("r1", blocks=(3,), computed=2)], num_tokens={"r1": 4}))

    assert out.req_ids == ["r1"]
    assert out.req_id_to_index == {"r1": 0}
    assert out.sampled_token_ids == [[5]]
    assert out.logprobs is None
    slot = worker.table.get("r1")
    assert slot.blocks.slots == [3]
    assert slot.valid_len == 2


def test_chunked_prefill_returns_empty_token_list(monkeypatch) -> None:
    """prefill 没跑完的请求返回空 token 列表，与 vLLM 的约定一致。"""
    worker = _worker()
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({}, []))

    out = worker.execute_model(_output(
        new=[_new_req("r1")], num_tokens={"r1": 2}))

    assert out.sampled_token_ids == [[]]


def test_wave_assigns_window_and_keeps_order(monkeypatch) -> None:
    """四个请求按窗口数 2 分两 wave，窗口号 0/1 交替，结果顺序不变。"""
    worker = _worker(num_slots=4, parallel_slots=2)
    assigned: dict[str, int] = {}
    tokens = {f"r{i}": 10 + i for i in range(4)}

    def step(self, req_id, slot_index=0, logprobs_width=None, scheduled=None):
        assigned[req_id] = slot_index
        return tokens[req_id], None

    monkeypatch.setattr(PIMWorker, "_step", step)
    new = [_new_req(f"r{i}", blocks=(3 + i,)) for i in range(4)]

    out = worker.execute_model(_output(
        new=new, num_tokens={f"r{i}": 4 for i in range(4)}))

    assert assigned == {"r0": 0, "r1": 1, "r2": 0, "r3": 1}
    assert out.sampled_token_ids == [[10], [11], [12], [13]]


def test_finished_request_is_forgotten(monkeypatch) -> None:
    """结束的请求清掉采样状态并归还请求槽。"""
    worker = _worker()
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({"r1": 5}, []))
    worker.execute_model(_output(new=[_new_req("r1")], num_tokens={"r1": 4}))

    out = worker.execute_model(_output(finished={"r1"}, num_tokens={}))

    assert out.req_ids == []
    assert "r1" not in worker._samplers
    assert worker.table.num_free == 4


def test_preempted_request_resets_from_computed_length(monkeypatch) -> None:
    """被抢占后按 vLLM 重发的整段 token 与已算长度重放，块表整段替换。"""
    worker = _worker()
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({"r1": 5}, []))
    worker.execute_model(_output(new=[_new_req("r1", blocks=(3,))],
                                 num_tokens={"r1": 4}))
    slot = worker.table.get("r1")
    slot.is_prefill_done = True
    slot.valid_len = 4

    worker.execute_model(_output(
        cached=_cached(["r1"], computed=[2], new_blocks=[([9],)],
                       resumed={"r1"}, all_token_ids={"r1": [1, 2]}),
        num_tokens={"r1": 2}))

    assert slot.blocks.slots == [9]
    assert slot.prompt_ids == [1, 2]
    assert slot.valid_len == 2
    assert not slot.is_prefill_done


def test_decode_step_with_more_than_one_token_raises(monkeypatch) -> None:
    """已经进入解码的请求只允许调度 1 个 token，否则立刻报错并点名请求。"""
    worker = _worker()
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({"r1": 5}, []))
    worker.execute_model(_output(new=[_new_req("r1")], num_tokens={"r1": 4}))
    worker.table.get("r1").is_prefill_done = True

    with pytest.raises(RuntimeError, match="r1.*2 个 token"):
        worker.execute_model(_output(num_tokens={"r1": 2}))


def test_structured_output_is_rejected_before_execution(monkeypatch) -> None:
    """带语法约束的请求在跑任何一步之前就失败。"""
    worker = _worker()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(PIMWorker, "_step", _fake_step({"r1": 5}, calls))

    with pytest.raises(ValueError, match="structured_output"):
        worker.execute_model(_output(
            new=[_new_req("r1")], num_tokens={"r1": 4}, structured=True))

    assert calls == []
