"""执行命令计划、维护请求槽和 KV 物理槽，并驱动 prefill/decode。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import write_page_table


class KVSlotPool:
    """管理请求之间共享的物理 KV 槽池。"""

    def __init__(self, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots 必须 >= 1，got {num_slots}")
        self._num_slots = num_slots
        self._free = list(range(num_slots))

    def alloc(self) -> int:
        """分配一个物理槽；池耗尽时抛 RuntimeError。"""
        if not self._free:
            raise RuntimeError(f"KV 物理槽已耗尽（共 {self._num_slots} 个）")
        return self._free.pop(0)

    def free(self, slot: int) -> None:
        """归还物理槽。"""
        self._free.append(slot)
        self._free.sort()


class KVBlockTable:
    """保存一个请求的逻辑块到物理槽的映射。"""

    def __init__(self, pool: KVSlotPool) -> None:
        self.pool = pool
        self.slots: list[int] = []

    def ensure(self, n_blocks: int) -> None:
        """保证至少 n_blocks 个物理槽，按需追加。"""
        while len(self.slots) < n_blocks:
            self.slots.append(self.pool.alloc())

    def release(self) -> None:
        """归还全部物理槽。"""
        for slot in self.slots:
            self.pool.free(slot)
        self.slots.clear()


@dataclass
class RequestSlot:
    """单个并发请求的状态。"""

    request_id: str
    request_slot: int
    blocks: KVBlockTable
    valid_len: int = 0
    prompt_ids: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    is_prefill_done: bool = False
    eos_id: int = 2

    def reset(self, prompt_ids: list[int]) -> None:
        """归还全部物理槽，按新 prompt 重新 prefill（抢占重算用）。"""
        self.blocks.release()
        self.prompt_ids = list(prompt_ids)
        self.valid_len = 0
        self.generated.clear()
        self.is_prefill_done = False


class RequestTable:
    """管理多请求并发状态，每请求独立的请求槽、页表和块表。"""

    def __init__(self, num_request_slots: int, kv_pool_slots: int) -> None:
        self._num_request_slots = num_request_slots
        self._pool = KVSlotPool(kv_pool_slots)
        self._slots: dict[str, RequestSlot] = {}   # request_id -> RequestSlot
        self._free: list[int] = list(range(num_request_slots))

    def add_request(self, request_id: str, prompt_ids: list[int],
                    eos_id: int = 2) -> RequestSlot:
        """申请一个空闲请求槽并注册新请求；无空闲槽时抛 RuntimeError。"""
        if not self._free:
            raise RuntimeError(
                f"所有 {self._num_request_slots} 个请求槽已占满，无法接入新请求")
        if request_id in self._slots:
            raise ValueError(f"request_id={request_id!r} 已存在")
        request_slot = self._free.pop(0)
        slot = RequestSlot(
            request_id=request_id,
            request_slot=request_slot,
            blocks=KVBlockTable(self._pool),
            prompt_ids=list(prompt_ids),
            eos_id=eos_id,
        )
        self._slots[request_id] = slot
        return slot

    def free_slot(self, request_id: str) -> None:
        """释放请求占用的请求槽和全部物理槽。"""
        slot = self._slots.pop(request_id, None)
        if slot is not None:
            slot.blocks.release()
            self._free.append(slot.request_slot)
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
    补齐位置不写 KV（`num_tokens` 只算真实 prompt），也不参与采样。
    """
    if sample_fn is None:
        sample_fn = lambda logits: int(np.argmax(logits))

    if not slot.is_prefill_done:
        prompt_len = len(slot.prompt_ids)
        seq_len = pad_to if pad_to is not None else prompt_len
        ids = list(slot.prompt_ids) + [pad_id] * (seq_len - prompt_len)
        slot.blocks.ensure(_blocks_for(prompt_len, compiled.kv_slot_tokens))
        _write_page_tables(compiled, hal, slot)
        events = execute_plan(
            compiled.prefill.plan, hal,
            values={
                "input_ids": _as_batch_tensor(ids),
                "causal_mask": causal_mask_of(seq_len),
                "position_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            },
            pos=0,
            request_slot=slot.request_slot,
            num_tokens=prompt_len,
        )
        logits = hal.wait(events[compiled.prefill.output_cmd_id])
        slot.valid_len = prompt_len
        slot.is_prefill_done = True
        token = sample_fn(np.asarray(logits)[0, prompt_len - 1])
    else:
        total_len = slot.valid_len + 1
        slot.blocks.ensure(_blocks_for(total_len, compiled.kv_slot_tokens))
        _write_page_tables(compiled, hal, slot)
        events = execute_plan(
            compiled.decode.plan, hal,
            values={
                "input_ids": _as_batch_tensor([slot.generated[-1]]),
                "causal_mask": causal_mask_of(1),
                "position_ids": torch.tensor([[slot.valid_len]], dtype=torch.long),
            },
            pos=slot.valid_len,
            request_slot=slot.request_slot,
            num_tokens=1,
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

    串行逐 slot 执行，每 slot 使用独立的 valid_len、请求槽和页表。
    """
    results: list[tuple[str, list[int]]] = []
    for slot in table.active_slots():
        if slot.generated and slot.generated[-1] == slot.eos_id:
            results.append((slot.request_id, list(slot.generated)))
            continue
        step_request(compiled, hal, slot, causal_mask_of, sample_fn=sample_fn)
    return results


def execute_plan(plan: ExecutionPlan, hal, *, values: dict[str, object] | None = None,
                 pos: int | None = None, request_slot: int = 0,
                 num_tokens: int | None = None) -> dict[int, object]:
    """执行计划并返回按命令编号索引的事件。"""
    hal.reset_events()
    hal.bind_inputs(values or {}, pos=pos, request_slot=request_slot,
                    num_tokens=num_tokens)
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
    compiled,
    hal,
    *,
    prompt_ids,
    max_new_tokens: int,
    eos_id: int,
    sample_fn,
    causal_mask_of,
) -> list[int]:
    """执行预填充和逐 token 解码，返回新生成的 token 编号。"""
    prompt = list(prompt_ids) if not hasattr(prompt_ids, "tolist") else prompt_ids.reshape(-1).tolist()
    table = RequestTable(1, compiled.kv_pool_slots)
    slot = table.add_request("decode", prompt, eos_id=eos_id)

    generated = [step_request(compiled, hal, slot, causal_mask_of, sample_fn=sample_fn)]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        generated.append(
            step_request(compiled, hal, slot, causal_mask_of, sample_fn=sample_fn))
    table.free_slot(slot.request_id)
    return generated


def _blocks_for(n_tokens: int, slot_tokens: int) -> int:
    """覆盖 n_tokens 个 token 需要的块数。"""
    return -(-n_tokens // slot_tokens)


def _write_page_tables(compiled, hal, slot: RequestSlot) -> None:
    """把本请求的块表写进每台 DPU 对应请求槽的页表区。"""
    for spec in compiled.kv_specs.values():
        write_page_table(spec, hal, slot.request_slot, slot.blocks.slots)


def _as_batch_tensor(ids: list[int]):
    """token id 列表 -> `[1, len(ids)]` 的 int64 张量（图 placeholder 期望的形状）。"""
    return torch.tensor([ids], dtype=torch.long)
