# 运行期窗口选择与分配器设计（2026-11-02）

## 目的

G2 原计划「编译 P=1..Pmax 的多套 plan 变体，运行期按在飞请求数选一套」。
动手前先核对现有代码，结论是**变体不用编**，真正缺的只是可观测性。
另外把设备端分配器的接口先设计出来（G2.3 明确只写文档，不实现）。

## G2.1 不实现：窗口本身就是变体

`runtime/binding.py::bind_plan` 只做两件事：平移激活区地址 `act_shift = k * act_stride`、
平移 SDPA 块表基址 `k * block_table_bytes`；`act_shift == 0 and block_slot == 0` 时
**直接返回原对象**（`tests/test_exec_binding.py::test_window_zero_returns_the_original_plan`）。

所以窗口 0 的计划就是 P=1 的那份计划，`compile_llama2` 已经按 `k in range(P)`
建好了 P 份平移结果（`decode_plans` / `prefill_plans[bucket]`）。再按 P 编一套
变体，只是把同一批命令照抄一遍，编译时间与产物体积翻倍而语义不变。

窗口数的选择也早已存在：`runtime/executor.py::run_one_step` 先算
`width = min(parallel_slots, num_parallel_slots, len(pending))`，再按 `width` 切 wave，
wave 内第 i 个请求用窗口 i。所以「按本步在飞请求数选最小的够用 P」在结构上就成立，
缺的只是让它出现在日志里。

## G2.2 已实现：把选中的窗口写进指标

`backend/metrics.py::StepStats` 增加四个字段，日志行同时给出本步与累计：

| 字段 | 含义 |
|---|---|
| `max_width` | 编译期窗口上限，即最多能同时跑几个请求 |
| `window_bytes` | 单个窗口在每台 DPU 上占的字节（激活区步长 + 一份块表） |
| `last_width` | 本步选中的窗口数 |
| `width_steps` | 每档窗口数各用了多少步，用来算「P=1 的步占比」 |

`_record_stats` 每步调用一次，`PIM_LOG_STEPS=1` 时打印一行：

```
PIM 步 5：本步请求 1、窗口 1，累计 wave 5（最宽 1），窗口分布 1×5，
窗口占用 3072/6144 B/DPU，token 5，槽池 3/64，paged_attn 命中 8 / 退回 0
```

`window_bytes` 由 `PIMWorker` 在编译后填进去：`act_stride + block_table_bytes`。

## G2.3 设备端分配器接口（只设计，不实现）

现状：MRAM 是一次性预分配的字节数组，三区（权重 / 激活 / KV 池 + 块表）在编译期
定容，运行期没有任何 alloc/free。真硬件上要按请求数伸缩窗口，必须先有分配器。

草案接口（HAL 侧，`hal_numpy` 与 `hal_vendor` 同签名）：

| 方法 | 语义 | 备注 |
|---|---|---|
| `alloc(bytes, align) -> offset` | 从 MRAM 申请一块对齐的区间 | 空间不足抛错，不返回 `None` |
| `free(offset)` | 归还该区间 | 必须原样归还 `alloc` 给出的起点 |
| `watermark() -> int` | 历史最高占用字节 | 定容与容量回归用 |
| `fragmentation() -> float` | 空闲总量 / 最大连续空闲块 | 大于 1 说明有洞 |

不实现的三条理由：

1. 真 HAL 的语义还没定（G4）：跨进程句柄、DMA 引擎能否直接读运行时分配的地址，
   都会改变接口形状；现在写出来的一定要重写。
2. 伪后端上 MRAM 是预分配的 `np.zeros`，分配器只能省地址，省不了真实内存，
   测不出任何东西。
3. 现阶段的窗口是**编译期定容**的，`free` 之后也没有人能复用它——
   请求结束即整步结束，窗口复用是「同一 wave 内换请求」，不是「释放再申请」。

净结论：G2.3 留作 G4 的输入，等真 HAL 语义确定后再落地。

## 验收

| 验收项 | 判据 | 结果 |
|---|---|---|
| 混合负载 token 与固定 P=2 逐 token 一致 | `test_mixed_prefill_and_decode_in_one_wave`、`test_four_requests_match_single_runs` | 通过（既有用例） |
| 单请求在飞时选 P=1 的步占比 ≥ 90% | `test_record_tracks_the_chosen_width_per_step`、`test_vllm_reports_stats_and_profile` | 实测 100%（`width_steps == {1: steps}`） |
| 未使用窗口不产生写操作 | `test_unused_windows_receive_no_writes` | 激活窗口与块表窗口的字节快照逐字节相同 |
| 编译时间与产物体积增加 ≤ Pmax 倍 | 不新增编译变体 | ×1 |
| 全量快测 | `pytest tests/ -q -x --ignore=…`（三个 7B 文件除外） | 见 `TODO.md` 证据表 |

「单步墙钟不高于 P=2 的 105%」这条不再单独测：`width=1` 与 `width=2` 对单请求走的是
**同一个窗口 0 的计划对象**，命令数与路径完全一致，不存在 P 带来的额外开销。
而且 G1.5 已把并行吞吐的论证移交给 G4，这里不再用墙钟做判据。

## 已实施的测试

| 文件 | 用例 | 判据 |
|---|---|---|
| `tests/test_step_stats.py` | `test_record_aggregates_waves_and_tokens` | 窗口档位按步累计 |
| `tests/test_step_stats.py` | `test_record_tracks_the_chosen_width_per_step` | 3/2/1 窗依次出现，空步不计数 |
| `tests/test_step_stats.py` | `test_format_lists_window_usage` | 日志含本步窗口、分布与每 DPU 占用 |
| `tests/test_step_stats.py` | `test_vllm_reports_stats_and_profile` | 单请求跑 vLLM，`width_steps` 全是 1 |
| `tests/test_parallel_requests.py` | `test_unused_windows_receive_no_writes` | 窗口 1 起的激活区与块表字节不变，窗口 0 确实变了 |

## 当前不足

- 分配器只有接口表，没有实现，也没有对应的 mock 单测（实现跟着 G4 走）。
- `window_bytes` 只算「激活步长 + 块表」，不含权重与 KV 池；它衡量的是窗口的
  边际占用，不是整块 MRAM。
- 日志里的窗口分布是累计值，看不到「哪一步用了哪一窗」的逐条轨迹；
  需要逐条时用 `PIM_PROFILE=1` 的命令记录。
