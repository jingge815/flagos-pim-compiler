# KV 块号以 vLLM 为唯一真源（2026-11-01）

## 改了什么

此前 vLLM 已经分配好块号（`NewRequestData.block_ids`、`CachedRequestData.new_block_ids`），
但 worker 只取 `req_id` 与 prompt，PIM 又在 `RequestTable` 里自建 `KVSlotPool` 另分一套
物理槽，两套账靠「块数相等」的启动断言勉强对齐。块号一旦不一致（抢占、前缀共享、
调换分配顺序）就会读写到别人的 KV。

现在块表直接采用 vLLM 给的块号，PIM 不再分配物理槽。

| 文件 | 改动 |
| --- | --- |
| `runtime/executor.py` | `KVBlockTable` 的 `pool` 可为空（外部块号）+ `adopt()`；`RequestTable` 的 `kv_pool_slots` 可选，`add_request(..., block_ids=)` |
| `backend/vllm_worker.py` | 新请求传 `block_ids[0]`；`_sync_preempted` → `_sync_cached`，追加 `new_block_ids` / 替换 `resumed_req_ids` |
| `tests/test_vllm_block_table.py` | 新增 7 例 |

## 块号的生命周期

```text
新请求          vLLM 给 block_ids        ──► add_request(block_ids=…) 整段采用
解码新开块      vLLM 给 new_block_ids    ──► adopt(…, replace=False) 追加到尾部
抢占后恢复      vLLM 重发整段块号        ──► reset() 清空 → adopt(…, replace=True)
请求结束        vLLM 收回块号            ──► free_slot() 只清本地记录
```

设备侧仍只有一份块表缓冲（每窗口一份），`step_request` 每步用
`slot.blocks.slots` 整段覆盖，尾部补 `-1`。

## 谁负责什么

| 职责 | vLLM | PIM |
| --- | --- | --- |
| 决定一个请求用哪几个物理块 | ✅ | — |
| 块号回收与复用 | ✅ | — |
| 把块号写进 DPU 块表缓冲 | — | ✅ |
| 块号越界 / 块数不足的检查 | — | ✅ |

`write_block_table` 已经校验「块号 ∈ [0, pool_slots)」「块数 ≤ 每请求块表项数」，
所以 vLLM 给出未分配或越界的块号会在写表时立即抛错，不会静默读到脏数据。

## 两种来源二选一

`RequestTable` 的构造参数决定块号来源，混用直接报错：

| 用法 | 谁给块号 | 使用者 |
| --- | --- | --- |
| `RequestTable(n)` + `block_ids=` | vLLM | `PIMWorker` |
| `RequestTable(n, kv_pool_slots)` 不给 `block_ids` | 自管 `KVSlotPool` | 离线 `run_decode_loop`、测试 |

`KVSlotPool` / `KVBlockTable.ensure()` 的自管分支保留给不经过 vLLM 的路径，
不再参与 vLLM 在线服务。

## 已实施的测试

`tests/test_vllm_block_table.py`（7 例，10.91 s）：

| 测试 | 验证点 |
| --- | --- |
| `test_new_request_adopts_vllm_blocks` | 新请求块表 == vLLM 的 `block_ids[0]` |
| `test_cached_request_appends_new_blocks` | `new_block_ids` 追加到尾部 |
| `test_resumed_request_replaces_blocks` | 抢占恢复整段替换，并重新 prefill |
| `test_finished_request_does_not_return_blocks` | 结束后只清本地记录，不触碰池 |
| `test_external_block_table_rejects_short_list` | 块数不够立即报错 |
| `test_block_source_must_be_exactly_one` | 两种来源混用/都缺都报错 |
| `test_external_block_numbers_change_nothing` | 换成外部块号（非 0 起始）生成 token 逐字不变 |

端到端：`tests/test_vllm_executor.py` 两个并发请求仍通过（vLLM 真实分配块号）。

## 当前不足

- 抢占只在 `num_computed_tokens == 0` 时整体重放；部分重算留给 D3。
- 前缀缓存本文写作时仍被拒绝；后续 D2 已支持，见
  `docs/vllm-prefix-cache-20261101.md`。
- 分块 prefill 本文写作时还不支持：当时一个请求的块表必须一次给全本步需要的块；
  后续 D1 已支持，见 `docs/vllm-chunked-prefill-20261101.md`。
