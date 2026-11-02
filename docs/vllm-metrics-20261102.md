# 运行时指标与调度报告（2026-11-02）

## 改了什么

以前只能回答「有没有跑错」，答不了「这一步 vLLM 派了几个请求、PIM 并行了几路、
注意力走了融合还是主机路径、槽池用了多少」。现在 worker 每步累加一组指标，
按需每 N 步打一行日志；需要更细的请求级/DPU 级数据时，用同一个开关在退出时
打印调度报告。

| 文件 | 改动 |
| --- | --- |
| `backend/metrics.py` | 新增 `StepStats`：每步累加 wave、请求、token、槽池占用与 attention 路由，并拼成一行日志 |
| `backend/vllm_worker.py` | `PIM_LOG_STEPS` 控制打印间隔，`PIM_PROFILE=1` 时挂请求级与后端级采集器，退出时打印 |
| `scripts/serve_pim.py` | 新增 `--pim-log-steps`、`--pim-profile` 两个开关 |
| `scripts/profile_execution.py` | 修 `_VllmProfiler._step` 的签名：worker 多传了 `scheduled`，分块 prefill 之后一直会抛 `TypeError` |
| `tests/test_step_stats.py` | 新增 4 例（含 vLLM 端到端） |

## 指标字段

| 字段 | 含义 |
| --- | --- |
| `steps` | 已执行的 `execute_model` 步数 |
| `last_requests` / `requests` | 本步 / 累计调度的请求数 |
| `tokens` | 累计调度的 token 数（prefill 按分块长度算） |
| `waves` / `widest_wave` | 累计 wave 数（按 `ceil(请求数 / 宽度)`）与历史最宽 wave |
| `pool_used` / `pool_total` | 当前被请求占用的物理块数（共享块只算一次）/ 槽池总数 |
| `attn_hit` / `attn_fallback` | `paged_attn` 融合内核与主机两遍路径的累计调用次数 |

一行日志的样子：

```text
PIM 步 800：本步请求 2，累计 wave 800（最宽 2），token 1602，槽池 6/9，paged_attn 命中 800 / 退回 0
```

`paged_attn` 的计数来自 `runtime.kernels.route_counts()`，是按头、按层累加的调用
次数，不是步数；退回计数长期不涨说明融合路径一直生效。

## 两个开关

| 开关 | 环境变量 | 作用 |
| --- | --- | --- |
| `--pim-log-steps N` | `PIM_LOG_STEPS` | 每 N 步打一行；默认 0，关闭时整步不做任何累加 |
| `--pim-profile` | `PIM_PROFILE=1` | 进程退出时打印请求级调度报告，以及各 DPU 的忙碌时间与利用率 |

`PIM_PROFILE` 会同时挂两个采集器：

```text
PIMWorker.__init__
        ├─ profile_vllm(self)      ── 包裹 execute_model/_step，记每个请求的起止
        └─ init_device
              └─ profile_backend(hal) ── 包裹内核与 host 命令，记每台 DPU 的忙碌区间
退出时 atexit
        ├─ print_vllm_report(...)  ── 逐步的请求构成、并行效率、窗口冲突
        └─ print_report(...)       ── 每台 DPU 的忙碌时间/利用率、主机与 DMA 占比
```

两个采集器都直接复用 `scripts/profile_execution.py` 的现有实现，没有新写统计口径。

## 开销

`StepStats.record` 在 4 请求 × 4 块时单次约 3.2 µs（本机实测，20 000 次取均值）。
相对一次 decode 步（合成用例 2.7 s，真实 7B 更长）或 100 ms 的一步，占比都在
0.01% 以下，远小于 2% 的验收线。关闭时只有一次环境变量读取。

## 已实施的测试

`tests/test_step_stats.py`（4 例）：

| 测试 | 验证点 |
| --- | --- |
| `test_record_aggregates_waves_and_tokens` | 两步的请求/token 累加，wave 数向上取整，最宽 wave 取最大值 |
| `test_record_counts_shared_blocks_once` | 两个请求共用块号时槽池占用去重 |
| `test_format_lists_step_fields` | 日志行含步数、wave、槽池与注意力路径 |
| `test_vllm_reports_stats_and_profile` | 真实 vLLM：`PIM_LOG_STEPS=1` 打出含「槽池」的日志，`PIM_PROFILE=1` 时请求级与 DPU 级采集器都采到数据 |

## 当前不足

- 只有日志，没有 HTTP 指标端点；接 Prometheus 需要再包一层。
- 槽池占用按块号去重统计，前缀缓存共享的块不会重复计数，但也不区分共享与私有。
- `--pim-profile` 的两个采集器都会给每条命令加计时，开启后自身有开销，
  不适合长期挂在生产上。
- 日志行里的注意力计数是累计值，不是区间增量；要算区间命中率需要两次采样相减。
