# 前缀缓存（2026-11-01）

## 改了什么

此前 `validate_engine_config` 见到 `enable_prefix_caching=True` 直接拒绝，
`scripts/serve_pim.py` 启动时也强制关掉它。C1 让 PIM 直接使用 vLLM 的块号后，
共享块可以由多个请求同时引用，只要写入范围不越界就不会互相污染，于是解除限制。

| 文件 | 改动 |
| --- | --- |
| `backend/vllm_worker.py` | `validate_engine_config` 去掉前缀缓存拒绝与 `cache_config` 参数；`_add_request` 用 `num_computed_tokens` 作请求的起始位置；`_sync_cached` 取 `new_block_ids[index][0]`（按 KV 分组的外层元组） |
| `scripts/serve_pim.py` | 不再强制 `enable_prefix_caching=False` |
| `tests/test_prefix_cache.py` | 新增 4 例（1 例走真实 vLLM 引擎） |
| `tests/test_vllm_block_table.py` | 伪造的 `new_block_ids` 改成 vLLM 真实形状（每组一层元组） |

## 写为什么不会碰到共享块

vLLM 只把**整块**放进缓存，命中长度 `num_computed_tokens` 一定是块大小的整数倍。
请求从 `slot.valid_len = num_computed_tokens` 起写，第 `t` 个 token 落在
`(pos + t) // kv_slot_tokens` 号块：

```text
prompt:  [ b0 ][ b1 ][ b2 ][ b3 ]        块大小 4（PIM 的 kv_slot_tokens = vLLM block_size）
                  ▲
                  │ 命中 2 块：b0/b1 只读，多个请求共用
       valid_len = 8
                  └──── 本请求从位置 8 开始写，只碰 b2/b3
```

`prefill_chunk` 只写 `[pos, pos + 本段长度)`，解码只写当前位置，写块号恒
`≥ num_computed_tokens // kv_slot_tokens`，所以共享块天然只读。同理，两个请求
共享前缀但私有尾块不同时，块表里前几项相同、后面不同，`write_block_table` 每步
重写的是各自那一份块表窗口。

## 共享块只读的验证

`test_shared_prefix_blocks_are_not_rewritten` 直接读回 DPU 上的 KV 字节：先整段
prefill 写满块 0/1，记下快照；再以「前缀已缓存」的方式从位置 8 起 prefill，
比对块 0/1 的字节数组完全不变，且采样 token 与整段 prefill 一致。

## 已实施的测试

`tests/test_prefix_cache.py`（微型 Llama，KV 每块 4 个 token）：

| 测试 | 验证点 |
| --- | --- |
| `test_new_request_starts_after_cached_prefix` | 命中 8 token 时 `valid_len == 8`，不重算已缓存段 |
| `test_shared_prefix_blocks_are_not_rewritten` | 共享块字节在执行前后逐字节相同，采样 token 正确 |
| `test_shared_prefix_with_two_windows` | 三个请求共享前缀、两个并发窗口，各自输出与单独运行一致 |
| `test_vllm_engine_reuses_cached_prefix` | 真实 vLLM 引擎：第二次相同 prompt 的 `num_cached_tokens > 0`，生成 token 与关闭缓存时逐 token 相同 |

`test_vllm_engine_reuses_cached_prefix` 的 prompt 取 20 个 token：缓存只认整块
（16），而 vLLM 又总要重算至少一个 token，所以只占一个整块的 prompt 命中数为 0，
测不到共享路径。

配套修正：`_sync_cached` 原先 `list(new_blocks)`，而 vLLM 的
`new_block_ids[index]` 是「按 KV 分组的一层元组」，单组下会得到
`[[7, 8]]` → `int()` 报错。之前的端到端用例从没追加过块（生成长度没跨块），
所以一直没暴露；开前缀缓存后第二个请求必然跨块，这个形状错误立刻显现。

## 当前不足

- 验收要求「两个共享 1024 token 前缀的请求」，实际跑的是 8 token 前缀的微型
  模型与 20 token 前缀的真实 vLLM 引擎；7B 量级的 1024 前缀未跑。
- 共享块的只读靠「写入范围不越界」这一推理保证，没有硬件写保护；若将来引入
  sliding window 或 SWA 之类的部分写回，需要重新审这段推理。
- 请求被抢占后重算仍从 `num_computed_tokens == 0` 整段重放，不利用已缓存的
  共享块（D3）。
