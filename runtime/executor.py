"""执行命令计划、维护逐槽解码状态，并通过 KV 缓存计算 SDPA。"""

from __future__ import annotations

import numpy as np
import torch

from contracts.exec_plan import ExecutionPlan
from runtime.kernels import softmax


class DecodeState:
    """保存跨 prefill 和 decode 调用的逐槽有效 KV 长度（closed batch 锁步）。"""

    def __init__(self, valid_lens: tuple[int, ...] | int = (0,), *,
                 valid_len: int | None = None):
        # valid_len 是旧标量入参的兼容口：单槽语义，等价 valid_lens=(v,)。
        if valid_len is not None:
            valid_lens = (valid_len,)
        if isinstance(valid_lens, int):
            valid_lens = (valid_lens,)
        self.valid_lens = tuple(valid_lens)

    @property
    def valid_len(self) -> int:
        """首槽有效长度；B=1 调用方与旧标量语义零改动。"""
        return self.valid_lens[0]

    @valid_len.setter
    def valid_len(self, value: int) -> None:
        """标量赋值广播到全部槽（closed batch 各槽锁步推进）。"""
        self.valid_lens = (value,) * len(self.valid_lens)

    def commit_one_position(self) -> None:
        self.valid_lens = tuple(v + 1 for v in self.valid_lens)


def execute_plan(plan: ExecutionPlan, hal, *, values: dict[str, object] | None = None,
                  pos: int | tuple[int, ...] | None = None,
                  dpu_window: int = 0) -> dict[int, object]:
    """执行计划并返回按命令编号索引的事件。

    `dpu_window` 是 DP 副本路由的物理块起点（副本 g = g×每副本 DPU 数）：
    计划命令的基数 DPU 编号整体平移到该副本的物理块上执行（R21-02b）。
    """
    hal.reset_events()
    hal.bind_inputs(values or {}, pos=pos, dpu_window=dpu_window)
    events: dict[int, object] = {}
    try:
        for cmd in plan.commands:
            for w in cmd.waits:
                hal.wait(events[w])
            events[cmd.id] = hal.submit(cmd)
        # 等待当前计划的全部命令。
        for event in events.values():
            hal.wait(event)
    finally:
        # 计划外的直接读写（权重装载等）必须落在窗口 0，执行完即复位。
        hal.bind_inputs({}, pos=None, dpu_window=0)
    return events


def replica_of_group(group_index: int, replicas: int) -> int:
    """DP 副本按组路由：组 i 走副本 i mod G（与 genesim 副本路由同口径）。"""
    if replicas <= 0:
        raise ValueError(f"replicas={replicas} 必须为正")
    return group_index % replicas


def replica_dpu_window(replica: int, dpus_per_replica: int) -> int:
    """副本 g 的物理 DPU 窗口起点：g × 每副本 DPU 数（物理块不重叠）。"""
    if replica < 0 or dpus_per_replica <= 0:
        raise ValueError(
            f"replica={replica} / dpus_per_replica={dpus_per_replica} 不合法")
    return replica * dpus_per_replica


def _as_prompt_rows(prompt_ids) -> list[list[int]]:
    """prompt 入参规整为逐槽行序列：list[int]/一维数组视作 B=1 单行。

    二维输入按行解释为一槽一条（走等长校验）；三维及以上违反契约直接抛，
    不做静默压平——压平会把 [B,S] 误读成 B=1 的单条超长序列。
    """
    if hasattr(prompt_ids, "tolist"):
        ndim = int(getattr(prompt_ids, "ndim", 1))
        if ndim == 2:
            prompt_ids = prompt_ids.tolist()
        elif ndim == 1:
            prompt_ids = prompt_ids.reshape(-1).tolist()
        else:
            raise ValueError(
                f"prompt 只接受一维或二维输入，got ndim={ndim}")
    if isinstance(prompt_ids[0], int):
        return [list(prompt_ids)]
    rows = [list(row) for row in prompt_ids]
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"closed batch 要求各槽 prompt 等长，got {sorted(lengths)}")
    return rows


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
    pad_token_id: int = 0,
    per_slot_prefill=None,
    dpu_window: int = 0,
) -> list[int] | list[list[int]]:
    """执行预填充和逐 token 解码；B=1 返回 list[int]，多槽返回逐槽 list。

    closed batch 整组进出：各槽 valid_len 锁步推进，已结束槽喂 pad token
    占位、其后续输出丢弃，全部槽结束即停。

    `per_slot_prefill`（B 份 [1,S] 逐槽计划，R21-02a）给定 prefill 改为
    B 个 M=S 步（每步一个槽的前向、KV 落本槽段，`prefill_plan` 不消费），
    数值与合批一致（各槽行独立）；decode 步保持合并 `[B,1]`。
    `dpu_window` 是 DP 副本路由的物理块起点（R21-02b），计划命令整体
    平移到该副本的物理 DPU 上执行。
    """
    prompts = _as_prompt_rows(prompt_ids)
    batch = len(prompts)
    prompt_len = len(prompts[0])
    if len(state.valid_lens) != batch:
        raise ValueError(
            f"DecodeState 槽数 {len(state.valid_lens)} 与 batch={batch} 不符")

    if per_slot_prefill is not None:
        # 逐槽 prefill：步序 0..B-1，第 b 步喂槽 b 的 [1,S]、计划 payload
        # 已把 KV 读写平移到槽 b 段；逐槽取首 token。
        if len(per_slot_prefill) != batch:
            raise ValueError(
                f"逐槽计划份数 {len(per_slot_prefill)} 与 batch={batch} 不符")
        first_logits = []
        for b in range(batch):
            slot_plan = per_slot_prefill[b]
            events = execute_plan(
                slot_plan.plan, hal,
                values={
                    "input_ids": _as_batch_tensor([prompts[b]]),
                    "causal_mask": causal_mask_of(prompt_len),
                    "position_ids": torch.arange(
                        prompt_len, dtype=torch.long).unsqueeze(0),
                },
                pos=(state.valid_lens[b],),
                dpu_window=dpu_window,
            )
            logits = np.asarray(
                hal.wait(events[slot_plan.output_cmd_id]))
            first_logits.append(logits[0, prompt_len - 1])
        state.valid_lens = (prompt_len,) * batch
        last_tokens = [int(sample_fn(logit)) for logit in first_logits]
    else:
        events = execute_plan(
            prefill_plan, hal,
            values={
                "input_ids": _as_batch_tensor(prompts),
                "causal_mask": causal_mask_of(prompt_len),
                "position_ids": (torch.arange(prompt_len, dtype=torch.long)
                                 .unsqueeze(0).repeat(batch, 1)),
            },
            pos=tuple(state.valid_lens),
            dpu_window=dpu_window,
        )
        logits = np.asarray(hal.wait(events[prefill_output_cmd_id]))
        state.valid_lens = (prompt_len,) * batch
        last_tokens = [int(sample_fn(logits[b, prompt_len - 1]))
                       for b in range(batch)]

    generated = [[token] for token in last_tokens]
    finished = [token == eos_id for token in last_tokens]
    for _ in range(max_new_tokens - 1):
        if all(finished):
            break
        # 已结束槽喂 pad 占位：只推进自己的 KV 槽位，不影响其它槽。
        fed = [pad_token_id if fin else tokens[-1]
               for fin, tokens in zip(finished, generated)]
        events = execute_plan(
            decode_plan, hal,
            values={
                "input_ids": _as_batch_tensor([[t] for t in fed]),
                "causal_mask": causal_mask_of(1),
                "position_ids": torch.tensor(state.valid_lens).reshape(-1, 1),
            },
            pos=tuple(state.valid_lens),
            dpu_window=dpu_window,
        )
        logits = np.asarray(hal.wait(events[decode_output_cmd_id]))
        state.commit_one_position()
        for b in range(batch):
            if finished[b]:
                continue
            token = int(sample_fn(logits[b, -1]))
            generated[b].append(token)
            if token == eos_id:
                finished[b] = True
    return generated[0] if batch == 1 else generated


def _as_batch_tensor(ids):
    """token id 序列（单行或多行等长）-> `[B, len]` 的 int64 张量。"""
    rows = [ids] if isinstance(ids[0], int) else ids
    return torch.tensor(rows, dtype=torch.long)
