"""执行命令计划、维护请求槽和 KV 物理槽，并驱动 prefill/decode。"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from backend.hal_numpy import ExecutionContext
from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import write_block_table


class KVSlotPool:
    """管理请求之间共享的物理 KV 槽池；多请求并发申请，读写都要加锁。"""

    def __init__(self, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots 必须 >= 1，got {num_slots}")
        self._num_slots = num_slots
        self._free = list(range(num_slots))
        self._lock = threading.Lock()

    def alloc(self) -> int:
        """分配一个物理槽；池耗尽时抛 RuntimeError。"""
        with self._lock:
            if not self._free:
                raise RuntimeError(f"KV 物理槽已耗尽（共 {self._num_slots} 个）")
            return self._free.pop(0)

    def free(self, slot: int) -> None:
        """归还物理槽。"""
        with self._lock:
            self._free.append(slot)
            self._free.sort()


class KVBlockTable:
    """保存一个请求的逻辑块到物理块号的映射；`pool` 为空表示块号由外部给定。"""

    def __init__(self, pool: KVSlotPool | None = None) -> None:
        self.pool = pool
        self.slots: list[int] = []

    def ensure(self, n_blocks: int) -> None:
        """保证至少 n_blocks 个块：自管时按需申请，外部给定时只校验够不够。"""
        if self.pool is None:
            if len(self.slots) < n_blocks:
                raise ValueError(
                    f"块表只有 {len(self.slots)} 项，本次需要 {n_blocks} 项")
            return
        while len(self.slots) < n_blocks:
            self.slots.append(self.pool.alloc())

    def adopt(self, block_ids: list[int], *, replace: bool = False) -> None:
        """采用外部块号：`replace` 为真时整段替换，否则追加。"""
        ids = [int(block) for block in block_ids]
        self.slots = ids if replace else [*self.slots, *ids]

    def release(self) -> None:
        """归还物理槽；外部给定的块号只清本地记录。"""
        if self.pool is not None:
            for slot in self.slots:
                self.pool.free(slot)
        self.slots.clear()


@dataclass
class RequestSlot:
    """单个并发请求的状态。"""

    request_id: str
    request_slot: int          # host 侧并发记账，不参与设备寻址。
    blocks: KVBlockTable
    valid_len: int = 0
    prompt_ids: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    is_prefill_done: bool = False
    eos_id: int = 2

    def reset(self, prompt_ids: list[int], computed: int = 0) -> None:
        """清空块表并按新 prompt 重新 prefill（抢占重算用）。

        `computed` 是已经算过的长度：命中前缀缓存时从它之后接着算，不重跑这段。
        """
        self.blocks.release()
        self.prompt_ids = list(prompt_ids)
        self.valid_len = computed
        self.generated.clear()
        self.is_prefill_done = False


class RequestTable:
    """管理多请求并发状态；`kv_pool_slots` 为空表示块号由外部（vLLM）提供。"""

    def __init__(self, num_request_slots: int, kv_pool_slots: int | None = None) -> None:
        self._num_request_slots = num_request_slots
        self._pool = None if kv_pool_slots is None else KVSlotPool(kv_pool_slots)
        self._slots: dict[str, RequestSlot] = {}   # request_id -> RequestSlot
        self._free: list[int] = list(range(num_request_slots))
        self._step_executor: ThreadPoolExecutor | None = None

    def step_executor(self, max_workers: int) -> ThreadPoolExecutor:
        """返回执行各请求一步的线程池，首次调用时按并发窗口数创建。"""
        if self._step_executor is None:
            self._step_executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._step_executor

    def add_request(self, request_id: str, prompt_ids: list[int], eos_id: int = 2,
                    block_ids: list[int] | None = None) -> RequestSlot:
        """注册新请求；块号要么由 `block_ids` 给定，要么由本表的池分配。"""
        if (block_ids is None) == (self._pool is None):
            raise ValueError(
                "块号来源必须二选一：给 block_ids，或给 kv_pool_slots 由本类分配")
        if not self._free:
            raise RuntimeError(
                f"所有 {self._num_request_slots} 个请求槽已占满，无法接入新请求")
        if request_id in self._slots:
            raise ValueError(f"request_id={request_id!r} 已存在")
        request_slot = self._free.pop(0)
        blocks = KVBlockTable(self._pool)
        if block_ids is not None:
            blocks.adopt(block_ids, replace=True)
        slot = RequestSlot(
            request_id=request_id,
            request_slot=request_slot,
            blocks=blocks,
            prompt_ids=list(prompt_ids),
            eos_id=eos_id,
        )
        self._slots[request_id] = slot
        return slot

    def free_slot(self, request_id: str) -> None:
        """释放请求槽；外部托管的块号只清本地记录，不归还给谁。"""
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
    pad_id: int = 0,
    slot_index: int = 0,
) -> int:
    """对单个请求执行一步（未 prefill 则整段 prefill，否则 decode），返回新 token。

    每次调用只产生一个 token。分块 prefill 走 `prefill_chunk` 自己拼段，
    这里只处理「一次跑完整段 prompt」，桶按 prompt 长度选，补齐位置不写 KV。

    `slot_index` 指定本次用的激活窗口和块表窗口（0..并行窗口数-1）。同一步
    并发的请求各用各的窗口，激活、块表都不相交；物理块号由块表决定（外部块号
    或自管槽池）。
    """
    if sample_fn is None:
        sample_fn = lambda logits: int(np.argmax(logits))

    if not slot.is_prefill_done:
        logits = prefill_chunk(
            compiled, hal, slot, slot.prompt_ids, 0, causal_mask_of,
            pad_id=pad_id, slot_index=slot_index)
        slot.is_prefill_done = True
        token = sample_fn(logits[0, len(slot.prompt_ids) - 1])
    else:
        total_len = slot.valid_len + 1
        slot.blocks.ensure(_blocks_for(total_len, compiled.kv_slot_tokens))
        _write_block_table(compiled, hal, slot, slot_index)
        events = execute_plan(
            _window_plan(compiled, "decode", slot_index), hal,
            ctx=ExecutionContext(lane_key=slot_index),
            values={
                "input_ids": _as_batch_tensor([slot.generated[-1]]),
                "causal_mask": causal_mask_of(1),
                "position_ids": torch.tensor([[slot.valid_len]], dtype=torch.long),
            },
            pos=slot.valid_len,
            num_tokens=1,
            causal_mask_standard=True,
        )
        logits = hal.wait(events[compiled.decode.output_cmd_id])
        slot.valid_len += 1
        token = sample_fn(np.asarray(logits)[0, -1])

    slot.generated.append(token)
    return token


def prefill_chunk(
    compiled,
    hal,
    slot: RequestSlot,
    token_ids: list[int],
    pos: int,
    causal_mask_of,
    *,
    pad_id: int = 0,
    slot_index: int = 0,
) -> np.ndarray:
    """把 `[pos, pos+len(token_ids))` 这段 token 写进 KV，返回 `[1, 桶长, vocab]` logits。

    桶按**本段长度**选，右侧补 pad（不参与计算、不写 KV）；`pos` 之前的历史
    已在 KV 里，注意力按绝对位置读取。`causal_mask_of` 生成标准因果掩码，
    据此走融合内核。分块 prefill 的中间段不采样，由调用方决定。
    """
    length = len(token_ids)
    seq_len = prefill_bucket(compiled, length)
    ids = list(token_ids) + [pad_id] * (seq_len - length)
    slot.blocks.ensure(_blocks_for(pos + length, compiled.kv_slot_tokens))
    _write_block_table(compiled, hal, slot, slot_index)
    events = execute_plan(
        _prefill_plan(compiled, seq_len, slot_index), hal,
        ctx=ExecutionContext(lane_key=slot_index),
        values={
            "input_ids": _as_batch_tensor(ids),
            "causal_mask": causal_mask_of(seq_len),
            "position_ids": torch.arange(
                pos, pos + seq_len, dtype=torch.long).unsqueeze(0),
        },
        pos=pos,
        num_tokens=length,
        causal_mask_standard=True,
    )
    logits = np.asarray(hal.wait(events[_prefill_output_cmd(compiled, seq_len)]))
    slot.valid_len = pos + length
    return logits


def prefill_bucket(compiled, n_tokens: int) -> int:
    """选最小的能放下 n_tokens 行的 prefill 桶；没有桶表时按原长。"""
    lengths = getattr(compiled, "prefill_seq_lens", ()) or ()
    if not lengths:
        return n_tokens
    for length in lengths:
        if n_tokens <= length:
            return length
    raise ValueError(
        f"token 数 {n_tokens} 超过最长 prefill 桶 {lengths[-1]}；"
        f"建议调大 PIM_PREFILL_SEQ_LEN 或调小 --max-num-batched-tokens")


def _prefill_plan(compiled, seq_len: int, slot_index: int):
    """取第 `seq_len` 桶第 `slot_index` 号窗口的计划；没有桶表时退回原计划。"""
    plans = getattr(compiled, "prefill_plans", None) or {}
    windows = plans.get(seq_len)
    if windows:
        return windows[slot_index]
    return compiled.prefill.plan


def _prefill_output_cmd(compiled, seq_len: int) -> int:
    """取第 `seq_len` 桶输出节点对应的命令编号。"""
    ids = getattr(compiled, "prefill_cmd_ids", None) or {}
    return ids.get(seq_len, compiled.prefill.output_cmd_id)


def _window_plan(compiled, phase: str, slot_index: int):
    """取某阶段第 `slot_index` 号窗口的计划；手搓蓝图没有窗口表时退回原计划。"""
    plans = getattr(compiled, f"{phase}_plans", None)
    if plans:
        return plans[slot_index]
    return getattr(compiled, phase).plan


def run_one_step(
    compiled,
    hal,
    table: RequestTable,
    causal_mask_of,
    *,
    sample_fn=None,
    parallel_slots: int | None = None,
) -> list[tuple[str, list[int]]]:
    """对表中每个活跃请求各执行一步，返回刚生成完毕的 (request_id, tokens)。

    活跃请求按 `parallel_slots` 分 wave，wave 内并发执行，每个请求用 wave 内
    自己的窗口（激活区与块表都不相交）。wave 之间串行，结果语义与逐个执行一致。
    """
    slots = table.active_slots()
    results: list[tuple[str, list[int]]] = []
    pending: list[RequestSlot] = []
    for slot in slots:
        if slot.generated and slot.generated[-1] == slot.eos_id:
            results.append((slot.request_id, list(slot.generated)))
        else:
            pending.append(slot)

    max_windows = getattr(compiled, "num_parallel_slots", 1)
    width = parallel_slots if parallel_slots is not None else max_windows
    width = max(1, min(width, max_windows, len(pending))) if pending else 1
    if width == 1:
        for slot in pending:
            step_request(compiled, hal, slot, causal_mask_of, sample_fn=sample_fn)
    else:
        executor = table.step_executor(max_windows)
        for start in range(0, len(pending), width):
            wave = pending[start:start + width]
            futures = [
                executor.submit(
                    step_request, compiled, hal, slot, causal_mask_of,
                    sample_fn=sample_fn, slot_index=index)
                for index, slot in enumerate(wave)
            ]
            for future in futures:
                future.result()
    return results


def execute_plan(plan: ExecutionPlan, hal, *, ctx=None,
                 values: dict[str, object] | None = None,
                 pos: int | None = None,
                 num_tokens: int | None = None,
                 causal_mask_standard: bool = False) -> dict[int, object]:
    """执行计划并返回按命令编号索引的事件。

    命令按计划顺序全部提交即可：`hal.submit` 内部会让每个 worker 先等
    自己的依赖完成，主线程不必逐条前置等待，否则提交无法提前、DPU 排不满。
    `ctx` 非空时输入值、事件表和泳道都走该上下文，多个请求才能同时在飞；
    为空时沿用全局事件表，保持旧的单请求语义。
    `causal_mask_standard` 声明本次传入的掩码就是标准因果掩码，注意力内核
    据此走融合路径；自定义掩码（滑动窗口、bias）不要置位。
    """
    if causal_mask_standard:
        values = {**(values or {}), "causal_mask_standard": True}
    if ctx is None:
        hal.reset_events()
        hal.bind_inputs(values or {}, pos=pos, num_tokens=num_tokens)
        events: dict[int, object] = {}
        for cmd in plan.commands:
            events[cmd.id] = hal.submit(cmd)
    else:
        ctx.events.clear()
        ctx.values = values or {}
        ctx.pos = pos
        ctx.num_tokens = num_tokens
        for cmd in plan.commands:
            hal.submit(cmd, ctx)
        events = dict(ctx.events)
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


def _write_block_table(compiled, hal, slot: RequestSlot, slot_index: int = 0) -> None:
    """把本请求的块表写进每台 DPU 的第 `slot_index` 份块表缓冲。"""
    for spec in compiled.kv_specs.values():
        write_block_table(spec, hal, slot.blocks.slots, slot_index)


def _as_batch_tensor(ids: list[int]):
    """token id 列表 -> `[1, len(ids)]` 的 int64 张量（图 placeholder 期望的形状）。"""
    return torch.tensor([ids], dtype=torch.long)
