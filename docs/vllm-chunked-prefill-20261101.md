# 分块 prefill（2026-11-01）

## 改了什么

此前 worker 要求一步调度完整个 prompt：`scheduled != len(prompt)` 直接抛
「不支持分块 prefill」。vLLM 0.14 默认 `enable_chunked_prefill=True`、
`max_num_batched_tokens=2048`，于是任何长于预算的 prompt 都跑不起来，长短请求
混批时短请求也会被拆开。

现在一段 prefill 只写本段的 KV，中间段不采样，最后一段才出第一个 token。

| 文件 | 改动 |
| --- | --- |
| `runtime/executor.py` | 新增 `prefill_chunk()`（任意 `pos` 起跑一段，返回每行 logits）；`step_request` 的 prefill 分支改为调用它；`prefill_bucket` 按**本段长度**选桶 |
| `backend/vllm_worker.py` | `_step` 拆成 `_prefill_step` / `_decode_step`；中间段返回空 token 列表 |
| `runtime/kernels.py` | `_host_paged_attention` 的按位置重算掩码改用绝对位置 `pos+t` |
| `scripts/serve_pim.py` | 预检规则倒转：分块预算不得超过最长 prefill 图 |
| `tests/test_chunked_prefill.py` | 新增 12 例（含 2 例 slow） |

## 一段 prefill 的执行

```text
vLLM 本步给 num_scheduled_tokens=n
        │
        ├─ 请求还没 prefill 完 ──► 取 prompt[pos : pos+n] 跑一张桶图
        │       pos = slot.valid_len（已写进 KV 的 token 数）
        │       num_tokens = 本段真实 token 数（补齐位不算、不写 KV）
        │       position_ids = arange(pos, pos+本段长度)
        │       ├─ pos+n < len(prompt) ──► 返回空 token 列表，等下一段
        │       └─ 最后一段 ──► 采样最后一个真实行，退出 prefill
        └─ 已 prefill 完 ──► 解码一个 token（仍要求 n == 1）
```

中间段返回空列表是 vLLM 的约定：`_update_request_with_output` 的注释写明
「还在 prefill 的请求，模型执行器应返回空 token 列表」。中途采样会污染罚项
与输出记账。

## 位置与掩码

- 每段仍按**本段长度**选桶（`prefill_bucket`），补齐位置靠 `num_tokens` 挡住，
  既不参与注意力也不写 KV。
- `position_ids` 从 `pos` 起算，KV 行号由 `position // kv_slot_tokens` 决定，
  所以第二段自然接着第一段写。
- 融合内核 `pim.paged_attn` 按绝对位置判可见性，`pos>0` 本来就正确；
  主机两遍路径原先用的是「第 t 行看到 t 个位置」的相对位置掩码，`pos>0` 时会把
  前半段历史遮掉，现改为 `prefill_mask(hist_len, hist_len)[pos+t]`（`pos=0` 与
  原来逐位等价）。

## 预算的约束

| 关系 | 含义 |
| --- | --- |
| `max_num_batched_tokens ≤ 最长 prefill 图` | 一段分块必须能塞进一张图，否则没有图可跑 |
| `prompt 长度` | 不再受 prefill 图长度限制，只受 `max_model_len` 限制 |

约束方向与改动前相反：以前「预算必须装下整段 prompt」，现在「预算不得超过图长」。
启动预检（`scripts/serve_pim.py`）与 worker 启动校验（`validate_engine_config`）
都按新方向报错，并给出两条改法（调小 `--max-num-batched-tokens` 或调大
`--pim-prefill-seq-len`）。

## 已实施的测试

`tests/test_chunked_prefill.py`（10 例快测 + 2 例 slow）：

| 测试 | 验证点 |
| --- | --- |
| `test_chunked_prefill_matches_whole`（5 组） | 8/4、6/4、5/1、9/8 等切法与整段 prefill 逐 token 一致 |
| `test_chunked_matches_whole_on_host_path` | 强制走主机两遍路径时分块同样一致（绝对位置掩码） |
| `test_interleaved_chunks_do_not_disturb_each_other` | 两请求交错分块，各自输出与单独运行一致 |
| `test_interim_chunk_leaves_prefill_open` | 中间段不置 prefill 完成，KV 有效长度停在本段末尾 |
| `test_chunks_write_kv_at_absolute_positions` | 第二段接着第一段的块继续写 |
| `test_vllm_engine_chunks_long_prompt` | vLLM 预算 4 < prompt 6 时自动分块，token 与不分块一致 |
| `test_long_prompt_chunked_matches_whole`（slow，64/512） | 3000 token prompt 的 64 / 512 切法与整段一致 |

配套改动的用例：`tests/test_vllm_capabilities.py`、`tests/test_serve_pim_preflight.py`
的预算用例按新规则（预算 > 图长才报错）重写，`tests/test_vllm_executor.py` 的
`max_num_batched_tokens` 改为等于图长。

## 当前不足

- 验收里的「4096 token prompt + 默认预算 2048 跑通」只在微型模型上等比验证
  （prompt 6 / 预算 4），没有用真实 7B 跑 4096；3000 token 的逐 token 对拍用的
  也是微型模型（3072 长度的图已在 numpy 后端跑满 8 分钟，7B 量级成本另算）。
- prefill 中途被抢占仍是整段重放，不利用已经写好的 KV（D3）。
- 同一请求的相邻分块串行执行，不重叠 host 重分布与下一块的 DPU 计算。
