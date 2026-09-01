"""执行命令计划、维护解码状态，并通过 KV 缓存计算 SDPA。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import PIMStaticKVCache, decode_mask, prefill_mask


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


def _host_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def make_sdpa_handler(layer: int, kv_specs, state: DecodeState, np_dtype: np.dtype):
    """构造读写指定层 KV 缓存的 SDPA 主机处理函数。"""
    # 建立当前层 KV head 到 DPU 的映射。
    dpu_of_head: dict[int, int] = {}
    for dpu_id, spec in kv_specs.items():
        if layer not in spec.layers:
            continue
        for head in spec.kv_heads:
            dpu_of_head[head] = dpu_id

    def handler(hal, cmd, args, kwargs):
        q, k, v, _mask = args
        q = np.asarray(q, dtype=np_dtype)  # [1, num_heads, Tq, head_dim]
        k = np.asarray(k, dtype=np_dtype)
        v = np.asarray(v, dtype=np_dtype)
        scale = kwargs.get("scale") or 1.0 / np.sqrt(q.shape[-1])
        num_heads, tq, head_dim = q.shape[1], q.shape[2], q.shape[3]
        cache = PIMStaticKVCache(hal, kv_specs, wram_budget_bytes=2**20)

        pos = hal.bound_pos if hal.bound_pos is not None else 0
        # 将当前序列的 K/V 逐位置写入本层缓存。
        owners = {dpu_id: spec for dpu_id, spec in kv_specs.items() if layer in spec.layers}
        for t in range(tq):
            k_by_dpu = {dpu_id: {h: k[0, h, t] for h in spec.kv_heads} for dpu_id, spec in owners.items()}
            v_by_dpu = {dpu_id: {h: v[0, h, t] for h in spec.kv_heads} for dpu_id, spec in owners.items()}
            cache.update(layer, pos + t, k_by_dpu, v_by_dpu)

        valid_len = pos + tq  # 当前步骤后的有效长度。
        max_seq = next(iter(kv_specs.values())).max_seq
        out = np.zeros((1, num_heads, tq, head_dim), dtype=np.float32)
        for head in range(num_heads):
            dpu_id = dpu_of_head[head]
            K_hist, V_hist = cache.read_tile(layer, dpu_id, head, 0, max_seq)
            for t in range(tq):
                mask = prefill_mask(t + 1, max_seq) if tq > 1 else decode_mask(valid_len - 1, max_seq)
                mask_row = mask[t] if tq > 1 else mask
                scores = K_hist.astype(np.float32) @ q[0, head, t].astype(np.float32) * scale
                weights = _host_softmax(scores + mask_row)
                out[0, head, t] = weights @ V_hist.astype(np.float32)
        return out.astype(np_dtype)

    return handler


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
