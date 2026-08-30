"""问题 6（运行时半）：命令 DAG 解释器 + 解码循环（方案问题 6 三.(3)）。

`execute_plan` 完全照方案伪代码——按拓扑序逐条提交命令，`waits` 精确等待
前驱事件（第 1 阶段串行调度下 `wait` 对已完成事件直接返回，等价于同步执行，
不改变 `NumpyBackend` 的调度语义）。`run_decode_loop` 实现方案 1550-1561 行
的自回归链条：prefill 执行一次、取真实末位 logits 采样出首个 token，随后
逐步喂回 decode 图、`valid_len` 递增，直到遇到 eos 或步数用尽。

**KV 感知的 SDPA handler**（`make_sdpa_handler`）是本轮编排器实现"KV cache
真正被使用"的落地点（技术方案要求，也是与用户确认过的边界：不改问题 1 白名单
分解 SDPA，KV 读写通过执行器专用 handler 接入，见 `runtime/exec_plan_gen.py`
`host_handler_of` 钩子与项目计划"SDPA 与 KV 的衔接设计"一节）：

- SDPA 节点的图内 Q/K/V 参数（`torch.export` 后天然 `Replicate@dpu`——白名单
  RoPE 逐元素算子不区分 head 边界，问题 2 传播出的是"每台 DPU 都有一份完整
  Q/K/V"）本身是这一步的真实计算结果，直接取用，不重算。
- handler 按层号 + `kv_specs` 找到每个 KV head 归属的 DPU，把这一步的
  post-RoPE K/V 切出对应 head、写入 `PIMStaticKVCache`（真实 MRAM 地址）；
  再用 `read_tile`/mask 读回 `[0, valid_len+1)` 的历史 K/V 做 numpy 版
  attention（QK^T → mask → softmax → weights@V），结果按 SDPA 原始输出形状
  拼回，供下游 `o_proj` 等节点消费。
- prefill（`valid_len` 初始为 0）时这样算出的结果应与图内原生 SDPA 完全
  一致（图内 SDPA 本来就在算整段因果注意力，只是没有把 K/V 落到 KV 区），
  测试据此可交叉验证 handler 本身的正确性；decode 时 KV 区历史是唯一数据
  来源，图内 SDPA 的单 token 结果不代表真实解码语义（`torch.export` 静态
  形状无法表达"读可变长度历史"），必须用 handler 重算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from contracts.exec_plan import ExecutionPlan
from memory.kv_layout import PIMStaticKVCache, decode_mask, prefill_mask


@dataclass
class DecodeState:
    """跨 prefill/decode 两次 `execute_plan` 调用维持的状态（方案三.(1)）。

    `valid_len` 是位置的唯一真值源；KV 区基址编译期定死、两图共用，这里只
    记录用于断言/调试，不参与地址计算（地址已经烤进 `KVRegionSpec.kv_off`）。
    """

    valid_len: int = 0

    def commit_one_position(self) -> None:
        self.valid_len += 1


def execute_plan(plan: ExecutionPlan, hal, *, values: dict[str, object] | None = None,
                  pos: int | None = None) -> dict[int, object]:
    """运行时解释器：绑定输入，按拓扑序逐条 wait 前驱、submit（方案 1538-1545 行）。

    入: plan —— `build_execution_plan` 产物；hal —— `NumpyBackend`；
    values —— 本次调用的图 placeholder 名 -> 具体值；pos —— 供 KV/SDPA
    handler 读取的写入位置（= `DecodeState.valid_len`）。
    出: `{cmd.id: Event}`，供调用方用 `hal.wait(events[cmd_id])` 取任意
    命令的结果（如图输出对应的 `CompiledPlan.output_cmd_id`）。
    """
    hal.reset_events()  # 命令 id 只在本次 plan 内唯一，跨 execute_plan 调用会重复（见 reset_events docstring）
    hal.bind_inputs(values or {}, pos=pos)
    events: dict[int, object] = {}
    for cmd in plan.commands:
        for w in cmd.waits:
            hal.wait(events[w])
        events[cmd.id] = hal.submit(cmd)
    # 等全部命令真正跑完才返回，不止等 output 的祖先——不是每条命令都在
    # output 的依赖链上（如与推理输出无关的死端 guard/断言），若不等它们，
    # 它们可能仍在线程池排队/执行时，调用方就发起了下一次 execute_plan，
    # 后者的 hal.bind_inputs() 会覆盖 hal.bound_pos/bound_value——这个死端
    # 命令若正巧此刻才真正跑（比如它依赖别的还没轮到的命令），读到的就是
    # 下一步的 pos/输入，不是自己这一步的（真实复现过：decode 循环偶发在
    # 随机步数后 argmax 崩成 0，两次相同输入的运行结果不一致，是线程池排队
    # 延迟造成的真实竞态，不是确定性 bug）。KV/SDPA handler 读 `hal.bound_pos`
    # 属于这类"不在 output 依赖链上也会被覆盖影响"的风险点，必须等全部命令
    # 完成才能安全地在下一步覆盖绑定。
    for event in events.values():
        hal.wait(event)
    return events


def _host_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def make_sdpa_handler(layer: int, kv_specs, state: DecodeState, np_dtype: np.dtype):
    """构造一个 SDPA 节点的 KV 感知 handler（供 `host_handler_of(node)` 返回）。

    入: layer —— 该 SDPA 节点所属层号（编译期从 q_proj 权重名解出，静态
    捕获）；kv_specs —— 问题 7/8 产出的 `dict[dpu_id, KVRegionSpec]`（已
    `build_kv_layout`/`plan_dpu` 落地真实 `kv_base`）；state —— 跨两次
    `execute_plan` 共享的 `DecodeState`（读 `valid_len`，运行时最新值，
    不在构造时固化）；np_dtype —— K/V numpy dtype（与 `KVRegionSpec
    .dtype_bytes` 一致）。
    出: `handler(hal, cmd, args, kwargs) -> np.ndarray`，`args` =
    `(q, k, v, mask)`（`exec_plan_gen` 已解析成具体 numpy/torch 值）。
    """
    # 只看持有**本层**的 DPU：流水切分下同一个 KV head 在不同层归属不同 DPU
    # （每个 stage 为自己那几层各存一份 KV），不按层过滤就会被后写的 stage 覆盖，
    # 于是读到别的 stage 的 KV 区——数值全错且不报错。张量并行下每台 DPU 都持有
    # 全部层，过滤条件恒真，与推广之前等价。
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
        # 本步新算的 K/V 写入 KV 区：一次写 tq 个位置（prefill 一次写整段
        # 提示词，decode 每步写 1 个新 token），逐位置调用 update（接口按
        # 单步定义，这里循环覆盖 prefill 的多位置写入）。
        # 只喂持有本层的 DPU（与 `PIMStaticKVCache.update` 的过滤一致）：流水下
        # 别的 stage 的 DPU 没有本层的 KV 区。
        owners = {dpu_id: spec for dpu_id, spec in kv_specs.items() if layer in spec.layers}
        for t in range(tq):
            k_by_dpu = {dpu_id: {h: k[0, h, t] for h in spec.kv_heads} for dpu_id, spec in owners.items()}
            v_by_dpu = {dpu_id: {h: v[0, h, t] for h in spec.kv_heads} for dpu_id, spec in owners.items()}
            cache.update(layer, pos + t, k_by_dpu, v_by_dpu)

        valid_len = pos + tq  # 写完这一步后，历史可读到的有效长度
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
    """自回归解码循环（方案 1550-1561 行）：prefill 一次 + 逐步喂回 decode 图。

    入: prefill_plan/decode_plan —— 两图各自的 `ExecutionPlan`（`build_
    execution_plan` 产物，KV 区/权重区共用同一份 `plan_dpu` 蓝图）；
    prompt_ids —— 提示词 token id（1D 或 [1, P] 张量/序列）；state ——
    prefill 执行前 `valid_len` 须为 0；sample_fn(logits_1d) -> int ——
    采样策略（贪心传 `lambda x: int(np.argmax(x))`）；
    prefill_output_cmd_id/decode_output_cmd_id —— 两图 `CompiledPlan
    .output_cmd_id`；causal_mask_of(seq_len) -> mask —— 按长度构造因果 mask
    （decode 图固定长度 1，mask 形状 `[1,1,1,1]`，图内值不参与 KV 感知
    handler 的历史读取，只是占位满足图输入契约）。两张图都必须显式接
    `position_ids` 输入（`_PositionalLlama` 而非 `_FixedMaskLlama` 导出）——
    `torch.export` 若不给 `position_ids` 会把 `arange(0, seq_len)` 烤成
    编译期常量，decode 图（`seq_len=1`）每步都会按位置 0 算 RoPE，只有
    prefill 首步正确，第 2 步起 K/V 全错（真实复现过，见项目计划"decode
    图的 RoPE 位置写死为 0"一节）。prefill 传 `arange(prompt_len)`，decode
    每步传 `[[state.valid_len]]`（新 token 的真实位置）。
    出: 采样出的 token id 列表（不含提示词），遇 `eos_id` 提前停。
    """
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
