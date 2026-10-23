"""执行命令计划、维护解码状态，并通过 KV 缓存计算 SDPA。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import PIMStaticKVCache, decode_mask, prefill_mask
from runtime.kernels import softmax


@dataclass
class DecodeState:
    """保存跨 prefill 和 decode 调用的有效 KV 长度。"""

    valid_len: int = 0

    def commit_one_position(self) -> None:
        self.valid_len += 1


@dataclass
class RequestSlot:
    """单个并发请求的状态。"""

    request_id: str
    slot_id: int
    valid_len: int = 0
    prompt_ids: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    is_prefill_done: bool = False
    eos_id: int = 2

    def reset(self, prompt_ids: list[int]) -> None:
        """丢弃本槽的 KV 状态，按新 prompt 重新 prefill（抢占重算用）。"""
        self.prompt_ids = list(prompt_ids)
        self.valid_len = 0
        self.generated.clear()
        self.is_prefill_done = False


class RequestTable:
    """管理多请求并发状态，每请求独立 valid_len 和 slot_id。"""

    def __init__(self, num_slots: int) -> None:
        self._num_slots = num_slots
        self._slots: dict[str, RequestSlot] = {}   # request_id -> RequestSlot
        self._free: list[int] = list(range(num_slots))

    def add_request(self, request_id: str, prompt_ids: list[int],
                    eos_id: int = 2) -> RequestSlot:
        """申请一个空闲 slot 并注册新请求；无空闲 slot 时抛 RuntimeError。"""
        if not self._free:
            raise RuntimeError(
                f"所有 {self._num_slots} 个 slot 已占满，无法接入新请求")
        if request_id in self._slots:
            raise ValueError(f"request_id={request_id!r} 已存在")
        slot_id = self._free.pop(0)
        slot = RequestSlot(
            request_id=request_id,
            slot_id=slot_id,
            prompt_ids=list(prompt_ids),
            eos_id=eos_id,
        )
        self._slots[request_id] = slot
        return slot

    def free_slot(self, request_id: str) -> None:
        """释放请求占用的 slot。"""
        slot = self._slots.pop(request_id, None)
        if slot is not None:
            self._free.append(slot.slot_id)
            self._free.sort()

    def get(self, request_id: str) -> RequestSlot:
        """按请求编号取 slot；不存在时抛 KeyError，让上游问题立刻暴露。"""
        return self._slots[request_id]

    def active_slots(self) -> list[RequestSlot]:
        """返回当前所有活跃请求的 slot 列表。"""
        return list(self._slots.values())

    @property
    def num_free(self) -> int:
        return len(self._free)


def step_request(
    compiled,
    hal,
    slot: RequestSlot,
    causal_mask_of,
    *,
    sample_fn=None,
    pad_to: int | None = None,
    pad_id: int = 0,
) -> int:
    """对单个请求执行一步（未 prefill 则 prefill，否则 decode），返回新 token。

    每次调用只产生一个 token，与 vLLM 一次 `execute_model` 一步的语义一致。
    `pad_to` 非空时把 prompt 右侧补齐到该长度，适配静态导出的 prefill 图；
    补齐位置的 KV 后续会被 decode 覆盖，且被因果掩码屏蔽。
    """
    if sample_fn is None:
        sample_fn = lambda logits: int(np.argmax(logits))

    if not slot.is_prefill_done:
        prompt_len = len(slot.prompt_ids)
        seq_len = pad_to if pad_to is not None else prompt_len
        ids = list(slot.prompt_ids) + [pad_id] * (seq_len - prompt_len)
        events = execute_plan(
            compiled.prefill.plan, hal,
            values={
                "input_ids": _as_batch_tensor(ids),
                "causal_mask": causal_mask_of(seq_len),
                "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            },
            pos=0,
            slot_id=slot.slot_id,
        )
        logits = hal.wait(events[compiled.prefill.output_cmd_id])
        slot.valid_len = prompt_len
        slot.is_prefill_done = True
        token = sample_fn(np.asarray(logits)[0, prompt_len - 1])
    else:
        events = execute_plan(
            compiled.decode.plan, hal,
            values={
                "input_ids": _as_batch_tensor([slot.generated[-1]]),
                "causal_mask": causal_mask_of(1),
                "position_ids": torch.tensor([[slot.valid_len]], dtype=torch.long),
            },
            pos=slot.valid_len,
            slot_id=slot.slot_id,
        )
        logits = hal.wait(events[compiled.decode.output_cmd_id])
        slot.valid_len += 1
        token = sample_fn(np.asarray(logits)[0, -1])

    slot.generated.append(token)
    return token


def run_one_step(
    compiled,
    hal,
    table: RequestTable,
    causal_mask_of,
    *,
    sample_fn=None,
) -> list[tuple[str, list[int]]]:
    """对表中每个活跃请求各执行一步，返回刚生成完毕的 (request_id, tokens)。

    串行逐 slot 执行，每 slot 使用独立的 valid_len 和 slot_id。
    """
    results: list[tuple[str, list[int]]] = []
    for slot in table.active_slots():
        if slot.generated and slot.generated[-1] == slot.eos_id:
            results.append((slot.request_id, list(slot.generated)))
            continue
        step_request(compiled, hal, slot, causal_mask_of, sample_fn=sample_fn)
    return results


def execute_plan(plan: ExecutionPlan, hal, *, values: dict[str, object] | None = None,
                  pos: int | None = None, slot_id: int = 0) -> dict[int, object]:
    """执行计划并返回按命令编号索引的事件。"""
    hal.reset_events()
    hal.bind_inputs(values or {}, pos=pos, slot_id=slot_id)
    events: dict[int, object] = {}
    for cmd in plan.commands:
        for w in cmd.waits:
            hal.wait(events[w])
        events[cmd.id] = hal.submit(cmd)
    # 等待当前计划的全部命令。
    for event in events.values():
        hal.wait(event)
    return events


def run_decode_loop(
    prefill_plan: ExecutionPlan,
    decode_plan: ExecutionPlan,
    hal,
    *,
    prompt_ids,
    max_new_tokens: int,
    eos_id: int,
    state: DecodeState,
    sample_fn,
    prefill_output_cmd_id: int,
    decode_output_cmd_id: int,
    causal_mask_of,
) -> list[int]:
    """执行预填充和逐 token 解码，返回新生成的 token 编号。"""
    prompt = list(prompt_ids) if not hasattr(prompt_ids, "tolist") else prompt_ids.reshape(-1).tolist()
    prompt_len = len(prompt)
    input_ids = _as_batch_tensor(prompt)

    events = execute_plan(
        prefill_plan, hal,
        values={
            "input_ids": input_ids, "causal_mask": causal_mask_of(prompt_len),
            "position_ids": torch.arange(prompt_len, dtype=torch.long).unsqueeze(0),
        },
        pos=state.valid_len,
    )
    logits = hal.wait(events[prefill_output_cmd_id])
    state.valid_len = prompt_len
    next_token = sample_fn(np.asarray(logits)[0, prompt_len - 1])

    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        token_input = _as_batch_tensor([generated[-1]])
        events = execute_plan(
            decode_plan, hal,
            values={
                "input_ids": token_input, "causal_mask": causal_mask_of(1),
                "position_ids": torch.tensor([[state.valid_len]], dtype=torch.long),
            },
            pos=state.valid_len,
        )
        logits = hal.wait(events[decode_output_cmd_id])
        state.commit_one_position()
        next_token = sample_fn(np.asarray(logits)[0, -1])
        generated.append(next_token)
    return generated


def _as_batch_tensor(ids: list[int]):
    """token id 列表 -> `[1, len(ids)]` 的 int64 张量（图 placeholder 期望的形状）。"""
    return torch.tensor([ids], dtype=torch.long)
