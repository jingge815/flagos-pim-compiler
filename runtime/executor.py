"""执行命令计划、维护解码状态，并通过 KV 缓存计算 SDPA。"""

from __future__ import annotations

from dataclasses import dataclass

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


def execute_plan(plan: ExecutionPlan, hal, *, values: dict[str, object] | None = None,
                  pos: int | None = None) -> dict[int, object]:
    """执行计划并返回按命令编号索引的事件。"""
    hal.reset_events()
    hal.bind_inputs(values or {}, pos=pos)
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
