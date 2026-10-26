# 等待边按泳道折叠（2026-10-26）

Phase 0a 把 `execute_plan` 的前置等待去掉、把同区间写者收成一个，等待边从
61 262 降到 29 455。剩下的边里 5 236 条是**同 DPU 的 launch→launch**，
而 `hal_numpy` 本来就保证同一 DPU 的命令按提交序串行——这些显式等待全是重复的。

## 规则

在 `_PlanBuilder.append` 加一个统一的规范化点：

| 前驱所在 | 保留方式 |
| --- | --- |
| DPU i | 只保留命令号最大的那一条 |
| host | 每条都保留 |

```
waits 里的每条前驱 -> 按泳道分组 -> DPU 泳道取 max，host 各自成组
```

host 命令之间没有命令链：它们都丢进同一个线程池，只靠 `waits` 排序，
所以不能折叠。DPU 命令由 `_stream_tail` 串起来，等最后一条就覆盖了同 DPU
的全部前驱。

这个规则同时作用于写后读（WAR）、读后写（RAW）、写后写（WAW）和显式值依赖，
所以比只在 `_war_waits` 里做「最后一个读者」更彻底，`_war_waits` 保持纯查询。

## 为什么安全

1. **同 DPU FIFO**：`_submit_on_dpu` 在每台 DPU 的锁内记住 `_stream_tail`，
   worker 先 `prev.result()` 再执行。提交顺序就是 `execute_plan` 的计划顺序，
   也就是命令号顺序。
2. **被折叠的边都是更早的命令**：`waits` 只引用已生成命令，泳道里的 max
   一定在依赖者之前提交并完成。
3. **host 不折叠**：host 命令无序，保留全部前驱。
4. 折叠**不删除依赖关系本身**，只是把「等一串同 DPU 前驱」换成「等这串的最后一条」。

契约写在 `NumpyBackend` 的类注释里：同一 DPU 命令严格按提交序执行。
`hal_vendor` 目前是空壳，接入时必须满足同样语义。

## 结果

同一进程内编译两版计划并各跑 8 步 decode（4 DPU / 4 层 / hidden 1024）：

| 计划 | 等待边 | 单条最多 | wall | 利用率 |
| --- | --- | --- | --- | --- |
| 不折叠（Phase 0a 后） | 29 455 | 277 | 725.5 ms | 49.7% |
| 泳道折叠 | 3 889 | 11 | 548.9 ms | 58.0% |

生成的 token 完全一致。剪枝后边构成：RAW 1 944、WAR 1 380、WAW 493、
显式值依赖 72。

`−24.3%` 是同一进程内的对比，不含机器负载漂移。相对最初的 1025 ms，
两步合计约 `−46%`。

## 剩余空间

剩下的 3 889 条基本都是**跨泳道**的：DPU 内核等 host 重分布写回（RAW）、
host 重分布等各 DPU 上最后一条读者（WAR）。它们对应真实的数据搬运/归约
屏障，靠折叠已经压不下去；再想快只能减少搬运本身（例如合并重分布）或做
跨请求并行（Phase 1）。

## 测试

- 新增 `test_same_lane_waits_collapse_to_the_last_command`：同 DPU 折叠到最大
  命令号，host 前驱逐条保留。
- 原有「同 DPU 连续 launch 有 RAW 依赖」测试仍通过——折叠后保留的正是
  那条边的最后一条。
- `test_exec_plan_gen` + `test_multi_request_decode` + `test_slot_isolation` +
  `test_executor_llama2_7b` + `test_concurrency_llama2_7b`：41 passed。
- 全量（排除本机挂起的 3 个 7B 文件）：1413 passed, 2 skipped, 1 deselected。

## 改动文件

| 文件 | 改动 |
| --- | --- |
| `runtime/exec_plan_gen.py` | `append` 增加泳道折叠；`_PlanBuilder` 记录 `lane_of` |
| `backend/hal_numpy.py` | 类注释写明「同 DPU 按提交序串行」契约 |
| `tests/test_exec_plan_gen.py` | 新增泳道折叠用例 |
