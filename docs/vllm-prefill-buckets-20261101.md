# prefill 图分桶（2026-11-01）

## 改了什么

此前 prefill 只有一张固定长度的图，长度等于 `PIM_PREFILL_SEQ_LEN`（默认
`max_model_len`）。哪怕 prompt 只有 4 个 token，也要按最长长度付计算与激活开销。

现在按长度编译多张 prefill 图（桶），运行期选**最小的能放下 prompt 的桶**，
右侧补齐到桶长；仍是一次跑完整个 prompt，不做分块 prefill。

| 文件 | 改动 |
| --- | --- |
| `runtime/compile.py` | `compile_llama2` 新增 `prefill_seq_lens`；每个桶一张图、一份计划；新增 `default_prefill_buckets` |
| `runtime/executor.py` | 新增 `prefill_bucket`、`_prefill_plan`、`_prefill_output_cmd`；`step_request` 自动选桶 |
| `backend/vllm_worker.py` | 计算桶表并传给编译器；不再传 `pad_to` |
| `tests/test_prefill_buckets.py` | 新增 21 例 |

## 桶的选取

```text
prompt_len=5 ──► 桶 8 ──► 右侧补 3 个 pad（不写 KV、不参与采样）
prompt_len=9 ──► 桶 16
prompt_len=17 ─► 报错：超过最长桶
```

默认桶表 `default_prefill_buckets(max_len)`：`128 / 512 / 2048` 中不超过
`max_len` 的项，加上 `max_len` 本身。小模型（`max_len=16`）只有 `(16,)`，
行为与改动前完全一致。

显式指定桶表有两种方式：

| 环境变量 | 含义 |
| --- | --- |
| `PIM_PREFILL_SEQ_LEN` | **最长桶**，默认 `max_model_len` |
| `PIM_PREFILL_SEQ_LENS` | 逗号分隔的桶表，如 `128,512,2048,4096`；最长项不得超过 `PIM_PREFILL_SEQ_LEN` |

## 内存布局

```text
每台 DPU 的 MRAM
┌──────────────┬───────────────┬───────────────────────────────┐
│   权重区     │     KV 区     │        激活窗口 × P           │
└──────────────┴───────────────┴───────────────────────────────┘
      ↑ 所有桶共用      ↑ 所有桶共用      ↑ 按最长桶规划步长 act_stride
```

- 权重区与 KV 区的偏移只由参数名、层数和 KV 几何决定，与序列长度无关。
  编译器对每个桶都跑一遍 `plan_dpu()`，然后**校验**每桶的 `weight` 与 `kv_base`
  和最长桶逐字节相同；不一致直接报错（说明窗口平移的前提被破坏了）。
- 激活区按最长桶的最坏占用对齐成统一窗口步长 `act_stride`；短桶用到的
  偏移更小，落在同一个窗口的前一段，不额外占内存。
- `kernel_mram`（算子编译器回传的内核 tile 占用）取所有桶 + decode 的**峰值**，
  不是求和。
- 因此总内存与单张最长图时一致，编译期预算判据不变。

## 执行计划

| CompiledModel 字段 | 含义 |
| --- | --- |
| `prefill_seq_lens` | 可用的桶（升序） |
| `prefill_plans` | 桶长 → 各执行窗口的计划列表 |
| `prefill_cmd_ids` | 桶长 → 输出节点对应的命令编号 |
| `prefill` / `prefill_gm` / `prefill_edges` | 最长桶的那份（兼容旧字段） |

`step_request()` 未显式给 `pad_to` 时调用 `prefill_bucket()` 选桶，再用该桶的
计划与输出命令编号；`decode` 不受影响。

## 已实施的测试

`tests/test_prefill_buckets.py`（21 例，53.04s）：

| 测试 | 验证点 |
| --- | --- |
| `test_bucket_table_is_sorted_and_ends_at_max` | 桶表升序、末项最长；`default_prefill_buckets` 对 4096 / 64 的取值 |
| `test_bucket_choice` | 按最短可容纳原则选桶（1/2/3/4/5/8/9/16） |
| `test_prompt_longer_than_longest_bucket_raises` | 超长 prompt 报错，不隐式截断 |
| `test_bucket_tokens_match_baseline` | prompt 1/2/3/4/5/8/9/13 的输出 token 与单张最长图基线**逐 token 相同** |
| `test_buckets_share_weight_and_kv_layout` | 各桶 `weight` / `kv_base` / `act_base` 与基线一致，`act_stride` 相同 |
| `test_windows_and_buckets_combined` | 两个不同桶的请求分到不同窗口并发执行，输出与单独执行一致 |

## 实测

小模型微基准（4 DPU、4 层、hidden 512、8 头、`max_seq=1024`、`kv_slot_tokens=16`、
单请求、100 个 token 的 prompt、`NumpyBackend`、取 3 次最小值）：

| 场景 | prefill 墙钟 | 相对上一行 |
| --- | --- | --- |
| 单张 1024 长度图（改动前） | 7675.0 ms | — |
| 选桶 128（分桶后） | 3102.7 ms | −59.6% |
| 桶 128 + 融合内核 | 2846.6 ms | −8.3% |

分桶把长度 100 的 prompt 从 1024 行降到 128 行，墙钟降到 40.4%；激活区占用同样
按 128 行计。数值与两张图的 token 完全一致（见 `test_bucket_tokens_match_baseline`）。

## 当前不足

- 每个桶都要重新 export + 分区 + 规划，编译时间与缓存条目随桶数线性增长；
  7B 模型多桶编译明显变慢，桶表不宜过密。
- 桶表只有环境变量入口，还没接到 `serve_pim.py` 的命令行。
- 短桶只省计算与激活占用，**省不了 KV**：KV 仍按 `max_model_len` 预留。
- 分块 prefill 本文写作时还不支持（后续 D1 已补上，见 `docs/vllm-chunked-prefill-20261101.md`）：
  当时 prompt 超过最长桶直接报错。
