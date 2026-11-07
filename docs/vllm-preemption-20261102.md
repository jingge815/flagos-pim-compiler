# 抢占与恢复（2026-11-02）

## 改了什么

此前 worker 只在「已经 prefill 完且 `num_computed_tokens == 0`」时重置请求，
其余抢占情况（prefill 中途被抢占、恢复时命中前缀缓存）会被静默跳过：块号换了，
但 host 侧的 `valid_len` / `is_prefill_done` 还是旧值，之后再写 KV 就会落在错误的
位置。现在按 vLLM 给的 `resumed_req_ids` 与 `num_computed_tokens` 明确区分
「整段重放」和「从缓存长度续算」。

| 文件 | 改动 |
| --- | --- |
| `runtime/executor.py` | `RequestSlot.reset(prompt_ids, computed=0)`：重置时把有效长度直接设成已计算的长度 |
| `backend/vllm_worker.py` | `_sync_cached` 改为 `resumed or computed < slot.valid_len` 时重置，并沿用 vLLM 重发的全量 token |
| `tests/test_preemption.py` | 新增 6 例（含强制抢占的 vLLM 端到端） |

## 两种恢复

```text
vLLM 本步回报 cached 请求
        │
        ├─ 在 resumed_req_ids 里（刚被抢占，块号整段换了）
        │       reset(all_token_ids, num_computed_tokens)
        │       ├─ num_computed_tokens == 0 ──► 从头重算整段（prompt + 已生成的 token）
        │       └─ num_computed_tokens > 0  ──► 命中了前缀缓存，从那之后接着算
        └─ 正常运行 ──► 只追加本步新分配的块
```

重置时 `all_token_ids` 是 vLLM 重发的「prompt + 已经生成的 token」，所以重放
prefill 会把已生成的部分一起算进去，末行采出的就是下一个 token；采样状态
（`_output_ids`）不需要回退，它本来就记着这些 token。

## 已实施的测试

`tests/test_preemption.py`（微型 Llama，6 例）：

| 测试 | 验证点 |
| --- | --- |
| `test_preempted_mid_prefill_restarts_from_zero` | prefill 没跑完就被抢占：块表整段替换、`valid_len` 归零 |
| `test_resumed_with_cached_prefix_keeps_computed_length` | 恢复时命中 8 token：`valid_len == 8`，不重跑前缀 |
| `test_preempted_mid_prefill_replays_and_matches_baseline` | prefill 中途被抢占 → 整段重放，生成 token 与不抢占基线一致 |
| `test_preempted_decode_replays_and_matches_baseline` | 解码中途被抢占 → 重放 prompt + 已生成 token，后续 token 一致 |
| `test_resume_from_shared_prefix_matches_baseline` | 恢复时命中共享前缀：只算剩下半段，token 一致 |
| `test_vllm_engine_preempts_and_matches_baseline` | 真实 vLLM：池 3 块跑两条各占 2 块的请求，`vllm:num_preemptions > 0`，输出与池 8 块（不抢占）逐 token 相同 |

端到端用例里池大小的取法是试出来的：块 16、`max_model_len` 32，每条请求最终占
2 块；池 4 块时 vLLM 并没有抢占（可用块比预想多 1），池 3 块才真正触发。

## 当前不足

- 只测了「重算型」抢占（vLLM 默认的 RECOMPUTE）；SWAP 型需要 CPU 侧 KV 池，
  PIM 后端没有配置，未覆盖。
- 恢复时的数值路径与正常路径共用 `prefill_chunk`，没有单独的混合精度/舍入测试；
  末行 logits 与解码路径的微小差异靠对照基线 token 覆盖。
- 抢占只发生在池紧张时，端到端用例只有一条；并发的多次抢占（反复抢占/恢复）
  没有专门用例。
