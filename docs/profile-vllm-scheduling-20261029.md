# vLLM 调度采集（2026-10-29）

## 改了什么

`scripts/profile_execution.py` 原来只能通过 `run_one_step` 采集 DPU 命令时间线，
测的是运行时内部的命令并发，碰不到 vLLM 那一层。本次新增一套**请求级**采集，
直接包住 `PIMWorker` 的 `execute_model` 和 `_step`，回答"vLLM 把几个请求排在同一步、
它们有没有真的同时在跑、各用了哪个激活窗口"。

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `StepRecord` | 数据类 | 一次 `execute_model` 的请求构成与请求级执行区间 |
| `VllmProfileResult` | 数据类 | 全部步骤；平均并行效率、最大并发、窗口冲突 |
| `_VllmProfiler` | 采集器 | 在 worker 实例上换掉 `execute_model` / `_step` |
| `profile_vllm(worker)` | 函数 | 开始采集，返回采集器 |
| `print_vllm_report(result)` | 函数 | 文本报告 |
| `demo_vllm()` | 入口 | `PIM_PARALLEL_SLOTS` 取 1 和 2 各跑一次并对比 |

## 怎么算并行

`StepRecord` 记下每个请求自己 `_step` 的起止（由工作线程写入），于是：

- **并行效率** = 各请求执行时间之和 / 本步墙钟。1 表示完全串行，2 表示两个请求真重叠。
- **并发度** = 请求区间扫描线上同时刻的最大数量。
- **窗口冲突** = 区间重叠却用同一个 `slot_index` 的请求对，正常应为空。

这套指标不依赖 vLLM 内部怎么切 wave：不猜宽度，只看实际执行区间，
所以即使 `execute_model` 的实现改了，结论仍然成立。

## 流程

```text
llm.generate
  └─ vLLM 调度器 → PIMWorker.execute_model ── 包一层，划步骤边界，记请求列表与 token 数
       └─ PIMWorker._step(请求, 窗口号) ────── 包一层，记该请求自己的起止
            └─ step_request → execute_plan → hal.submit
```

## 已实施的验证

`python scripts/profile_execution.py vllm`（微型 Llama，两个 prompt）：

| PIM_PARALLEL_SLOTS | 每步请求数 | 并行效率（预填充 / 解码） | 最大并发 | 窗口 |
| --- | --- | --- | --- | --- |
| 1 | 2（同一步） | 1.00 / 1.00 | 1 | 0,0 |
| 2 | 2（同一步） | 1.98 / 1.87 | 2 | 0,1 |

两个请求在 P=1 下被同一个 wave 串行执行，在 P=2 下真正重叠，且窗口不相交，
无窗口冲突。说明 vLLM 这条路的 wave 循环确实生效。

## 发现的问题：全局槽池默认少一块

vLLM 的 `BlockPool` 在初始化时会永久占掉一个 `null_block`
（`vllm/v1/core/block_pool.py:174`）。而 `PIMWorker` 默认把池子开成
`每请求块数 × 请求槽数`，正好没有余量，于是最后一路并发永远进不来：

```text
并发上限 = floor((B×N − 1) / B) = N − 1
```

实测 `max_num_seqs=2`、`max_seq=16`、块大小 16 时池子只有 2 块，可用 1 块，
两个请求被完全排成前后两段。演示脚本里用 `PIM_KV_POOL_SLOTS = 请求数 + 1`
绕开；生产侧建议把 `backend/vllm_worker.py` 的默认池改成多一块：

```python
self.kv_pool_slots = int(raw) if raw else self.blocks_per_slot() * self.num_request_slots + 1
```

## 不足

- 采集器要在实例上换方法，只能用于单进程模式（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）。
- 只记请求级区间，不记 DPU 命令；要看内核重叠仍需另外用 `profile_backend(worker.hal)`。
- 端到端延迟里还含调度器与输出处理的开销，本报告不拆分它们。
