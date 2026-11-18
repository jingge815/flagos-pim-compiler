# M3 混批调度策略层（2026-11-17）

## 做了什么

wave 组批从 `run_one_step` 的内联切片提取成命名策略函数，外部批处理器
（vLLM）可以把自己的组批结果直接喂进来；同时补齐容量查询三件套，接入方
「先问后进」，池耗尽通过查询归零实现背压，突变调用仍按仓规 fail-fast。

```
调用方（vLLM / 基准 / 测试）
  │ 问容量：available_windows(compiled, table)、table.kv_slots_free()
  │ 归零 → 不接入（背压），不抛异常
  ▼
run_one_step(waves=None) ──默认──▶ plan_wave(slots, width=width)  # FCFS 纯函数
run_one_step(waves=[[...], ...]) ─外部给定──▶ 原样执行（先全量校验宽度）
  ▼
wave 内并发（slot_index = wave 内位置），wave 间串行，语义与逐个执行一致
```

## 增删文件与函数

| 文件 | 改动 |
| --- | --- |
| `runtime/executor.py`（+44/-10） | 新增 `plan_wave`（FCFS 纯函数，泛型 `list[T] -> list[list[T]]`，width<1 报错）；`run_one_step` 新参 `waves`（None 走默认策略，外部给定则原样执行，任一 wave 超并行窗口数在执行前抛 ValueError；全单例 wave 走串行快路径不建线程池）；新增 `available_windows(compiled, table)`；`KVSlotPool.num_free` 属性；`RequestTable.kv_slots_free()`（外部托管块号返回 None） |
| `backend/vllm_worker.py`（+11/-12） | `execute_model` 的内联 range 切片删除，改用 `plan_wave`；`sampled`/`results` 改为按序追加，行为不变（vllm 测试守恒） |
| `tests/test_multi_request_decode.py`（+144/-3） | `_compile` 扩 `num_parallel_slots`/`act_pool_slots`；`_mixed_table`/`_run_steps`/`_assert_match_individual` 辅助；新增 5 个测试（见下） |

## 关键决策

1. **策略是纯函数不是调度器类**：wave 组合无状态，宽度上限属编译产物
   （`num_parallel_slots`，动态路径编译期已折算 `min(默认4, act_pool, 请求槽数)`）、
   请求槽属 `RequestTable`，各归各的 owner。泛型签名是因为 vllm_worker 切的是
   `req_ids`（字符串），同一策略函数两个调用方，不另造包装。
2. **背压靠查询，突变仍抛**：`available_windows()==0` / `kv_slots_free()==0`
   即「装不下」，调用方不接入；`add_request` 溢出照旧 RuntimeError
   （`test_request_table_rejects_overflow` 锁定，fail-fast 不兜底）。
3. **外部 waves 先全量校验再执行**：超宽 wave 在任何命令提交前抛错，
   不留半执行的副作用状态。

## 测试（12 passed，含参数化）

| 测试 | 覆盖 |
| --- | --- |
| `test_plan_wave_fcfs` | 纯函数：保序、截断、末组短、宽度边界报错 |
| `test_mixed_prefill_decode_wave_matches_individual`（静态+动态） | **M3 验收**：2 变长新请求 + 2 解码中请求同 wave == 逐个执行 |
| `test_run_one_step_external_waves` | 外部组批乱序/不等宽 == FCFS；空 waves 空转；超宽先报错后无副作用 |
| `test_available_windows_backpressure` | 查询归零即停（不触 RuntimeError）；`kv_slots_free` 随 prefill/跨块递减；外部托管返回 None |
| `test_batch8_full_pool_no_conflict`（静态宽 8 / 动态池 8 宽 4） | **M3 验收**：8 请求全驻留满池，并发互不串扰 == 逐个执行 |

回归：`tests/test_vllm_executor.py` 2 passed（worker 迁移行为等价）；
`-k "runtime_materialize or shape_binding or multi_request or vllm"` 102 passed。

## 当前不足 / 已知边界

- `--batch` 基准口径未新增：`scripts/bench_parallel.py` 的 `--requests` +
  `--parallel-slots` 已覆盖「N 常驻 × 宽度扫描」，加别名是投机。
- 混批的 KV 感知准入（按 wave 的 KV 需求算可容纳性）未做：当前查询只数槽，
  不预估请求将申请的块数；vLLM 路径块号本来就外部托管，自管路径够用。
- GeneSim 演练默认 FCFS 策略属 M4a，本层只提供策略函数。

净增删：+199 / -25（三个文件）。已提交落库。
