# 执行器 Phase 0a：去掉前置等待 + 依赖边剪枝（2026-10-26）

单步 decode 的 DPU 利用率只有三分之一，瓶颈不在计算，而在主机编排：
主线程逐条命令前置等待，提交和执行无法重叠；同时命令的等待表把历史上
所有写过同一缓冲的命令都列了一遍，依赖边严重膨胀。本次两处小改，不动物理布局。

> 2026-10-26 追加：等待边又按执行泳道折叠了一次（29 455 → 3 889），
> 见 `docs/exec-plan-lane-collapse-20261026.md`。

## 测量（4 DPU / 4 层 / hidden 1024 / 8 步 decode 中位数）

| 组合 | 单步 wall | DPU 利用率 | 每 DPU 忙碌 |
| --- | --- | --- | --- |
| 改动前（前置等待 + 旧依赖表） | 1025 ms | 36.4% | 403 / 382 / 373 / 332 ms |
| 只换执行器（旧依赖表） | 768 ms | 49.6% | 388 / 385 / 381 / 368 ms |
| 执行器 + 依赖剪枝 | 689 ms | 54.2% | 409 / 358 / 370 / 356 ms |

三组生成的 token 完全相同。单步耗时 `-32.8%`。

## 改动一：`runtime/executor.py::execute_plan` 不再逐条前置等待

```python
# 改前
for cmd in plan.commands:
    for w in cmd.waits:
        hal.wait(events[w])      # 等前驱跑完才提交下一条
    events[cmd.id] = hal.submit(cmd)

# 改后
for cmd in plan.commands:
    events[cmd.id] = hal.submit(cmd)
```

安全性依据：`hal.submit` 内部已经把等待下沉到 worker——DPU 路径
（`_submit_on_dpu` 的 `run`）先等同一 DPU 的上一条命令、再等全部 `deps`；
host 路径（`_run_host_or_dma`）同样先等 `deps`。主线程再等一遍纯属重复，
代价是提交无法提前，DPU 空转。计划末尾的 join 保留：它是「本计划全部结束」
的屏障，下一步写共享块表前必须成立。

## 改动二：`runtime/exec_plan_gen.py::_deps_of` 只留最后一个写者

decode 图里每层复用同一块激活缓冲，同一 `(loc, offset, length)` 会被反复写。
`_deps_of` 原先列出全部相交写者，现在同一区间只保留命令号最大的那个：

```
旧：61 262 条等待边，单条最多 560
新：29 455 条等待边，单条最多 277
```

安全性：写者对同一区间的写是程序序串起来的——后写者生成时已经等过前写者
（`append` 的写后写逻辑）。所以等最后一个写者，沿等待链自然覆盖前面的。
区间只部分重叠时不省（各写者之间不保证有链），仍全部保留。

## 当前等待边构成（剪枝后 29 455 条）

| 类型 | 条数 |
| --- | --- |
| 写后读 WAR | 24 198 |
| 读后写 RAW | 9 852 |
| 写后写 WAW | 2 857 |
| 其他（值依赖） | 72 |

一条边可能同时属于多类（例如既读写同一块），所以各项之和大于总边数。

WAR 占了绝大部分，来自 `_war_waits`：host 重分布写回目标缓冲前，要等
**所有**读过该缓冲旧值的命令。这是下一步的主要空间。

## 测试

- `tests/test_exec_plan_gen.py` 新增两条：
  `test_repeated_writer_of_same_range_only_leaves_the_last_one_as_dep`
  固定「同区间多次写只留最后一个」；`test_overlapping_but_not_identical_writes_are_all_kept`
  固定「部分重叠不省」。
- `tests/test_exec_plan_gen.py` + `test_multi_request_decode.py` + `test_slot_isolation.py`：36 passed。
- `tests/test_executor_llama2_7b.py` + `test_concurrency_llama2_7b.py`：4 passed。
- 全量（排除本机挂起的 3 个 7B 文件）：1412 passed, 2 skipped, 1 deselected（12 分钟）。

## 不足与下一步

- WAR 边可再剪：同一 DPU 上的命令由 `_stream_tail` 按提交序串行，因此写者
  只需等该 DPU 上**最后一个**读者。落地前要把「同一 DPU 命令串行」从实现细节
  升级成 HAL 硬约束（`hal_vendor` 必须同样满足），否则只在 numpy 后端成立。
- 跨请求并行仍被三样全局状态挡住：`bind_inputs`、事件表、每 DPU 单块表（Phase 1）。
- 54% 利用率下，把单步压到「DPU 忙碌合计 / DPU 数 ≈ 373 ms」是串行架构的参考下限。
