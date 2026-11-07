# 执行器 Phase 1：跨请求并行（2026-10-26）

Phase 0a 与泳道折叠之后，单请求单步已经从 1025 ms / 36.4% 降到
548.9 ms / 58.0%，剩下的 3 889 条等待边都是跨泳道真实屏障（DMA、归约），
再怎么剪也换不来更多重叠。这一阶段换思路：**让多个互不相关的请求同时
在飞**，用一个请求的 DPU 计算去填另一个请求等待主机重分布的空泡。

> 单请求内部的泳道折叠（`docs/exec-plan-lane-collapse-20261026.md`）仍是前提：
> 一个请求内部命令严格按提交序执行，跨请求才允许交错。

## 五个机制

| 机制 | 位置 | 作用 |
| --- | --- | --- |
| 每请求激活窗口 | `memory/mem_planner.py` | 激活区复制 `P` 份，不同请求不互相覆盖 |
| 每请求块表窗口 | `memory/kv_layout.py` | KV 区放 `P` 份块表，各写各的 |
| 编译期计划变体 | `runtime/binding.py` | 为窗口 `k` 生成平移后的计划，0 号即原计划 |
| 执行上下文 | `backend/hal_numpy.py` | 输入、位置、令牌数、事件表按请求隔离 |
| 按 (上下文, DPU) 分泳道 | `backend/hal_numpy.py` | 同请求仍 FIFO，不同请求不互相排队 |

驱动：`runtime/executor.py::run_one_step` 把活跃请求按 `P` 分成 wave，
wave 内用线程池并发跑 `step_request(slot_index=i)`；vLLM 的
`backend/vllm_worker.py::execute_model` 同样按 wave 并发。

## 窗口几何：统一步长 S

```
每 DPU 地址空间：
  权重 [0, kv_base)
  KV   [kv_base, kv_base + P*bt_bytes + pool_slots*slot_bytes)
  激活 [act_base, act_base + P*S)

  S = align_up(max_d(act_extent_d), hw.align)
  total = act_base + P*S
```

窗口 `k` 的激活基址是 `act_base + k*S`，块表基址是
`block_table_base + k*bt_bytes`。同一请求的 prefill 与 decode 共用自己那份
窗口（两者不会同时跑）。

**为什么必须统一步长**：`comm/plan.py` 的批量 `push_from / push_to /
broadcast_to` 用 `first.src_addr` 一个基址扇出到所有 DPU。若各 DPU 步长不同，
同一批量的偏移就错位。统一成全局最大值后，任何 DPU 的窗口 `k` 都在自己
`act_base + k*S` 处，批量扇出的单一偏移对所有 DPU 同时成立。

## 计划变体：`bind_plan` 只平移激活与块表

`runtime/binding.py::bind_plan(plan, act_base_by_dpu, act_shift, block_slot, bt_stride)`：

- `act_shift == 0 and block_slot == 0` 时**直接返回原计划对象**（零拷贝、行为不变）。
- 否则对每条命令：激活区地址（`loc[0]=="dpu" and offset >= act_base[dpu]`）加
  `act_shift`；`sdpa.dpus[*]["bt_base"]` 加 `block_slot*bt_stride`（深拷贝后改）。
- **不动**权重、KV 区，也**不动** `payload["mram_offset"]`。

最后一条最关键：`mram_offset` 是 `runtime/kernels._ctx_of` 的编译缓存键，
从不作数据地址。让它保持窗口 0 的值，所有窗口才能共享同一份已编译内核，
不因窗口数触发重编译。

## 执行上下文隔离

`NumpyBackend` 原本有四处实例级全局状态，两请求并发会互相污染。现在改成
`ExecutionContext(lane_key, values, pos, num_tokens, events)`：

- `submit(cmd, ctx)` 从 `ctx.events` 解等待、写回 `ctx.events`。命令 id 每计划
  从 0 开始也没关系，不同上下文各有自己的事件表。
- worker 线程用 thread-local 安装 `ctx`，`bound_value / bound_pos /
  bound_num_tokens / result_of` 优先读本上下文，读不到才回退旧的全局路径。
- `_stream_locks / _stream_tail` 的键从 `dpu_id` 改成 `(lane_key, dpu_id)`：
  同一请求内部仍严格 FIFO（泳道折叠前提不变），不同请求在同一 DPU 上互不排队。
- 线程池按 `min(64, num_dpus*P + P + 1)` 扩容；host 命令仍由 `_host_lock` 串行
  （`xfer_buffers` 是机器级全局），所以跨请求加速来自 DPU 填 host 空泡。

不传 `ctx` 的旧调用全部保留原语义，`submit(cmd)`、`bind_inputs`、
`reset_events` 等行为不变。

## wave 语义与内存倍数

`run_one_step` 把当前活跃请求（排除已 EOS 的）按 `P` 切 wave，wave 内第 `i`
个请求用窗口 `i`。窗口数是**在飞的请求数**，不是 `request_slot` 编号。
vLLM 里 `P = min(num_request_slots, PIM_PARALLEL_SLOTS)`，默认 2。

代价是激活区 ×P、块表 ×P。7B 若超预算，`compile_llama2` 会在
`fit_activation_windows` 里抛出带 per-DPU 详情的错误，指引调小 P 或调大
`P` 的预算。`num_request_slots=1` 或 `PIM_PARALLEL_SLOTS=1` 时逐字节回到
Phase 0a 行为。

## 测量（4 DPU / 4 层 / hidden 1024 / 4 请求 × 8 步 decode）

同一份 `num_parallel_slots=4` 蓝图，只改 `run_one_step` 的 `parallel_slots`：

| 组合 | 8 步 wall | 每步 wall | DPU 利用率 |
| --- | --- | --- | --- |
| 串行（`parallel_slots=1`） | 27 758 ms | 3 470 ms | 61.2% |
| 并发（`parallel_slots=4`） | 19 913 ms | 2 489 ms | 76.2% |

总工作量相同，吞吐 ×1.39，利用率 +15 个百分点。串行两栏的每 DPU 忙碌合计
（约 16.8 s）与并发两栏（约 15.2 s）接近，说明提速来自**把空泡填上了**，
不是把计算变快。

## 测试

| 测试文件 | 覆盖点 |
| --- | --- |
| `tests/test_activation_windows.py` | 默认单块表逐字节不变；多块表平移；`write_block_table` 只写本窗口、越界报错；`fit_activation_windows` 统一步长与超限报错 |
| `tests/test_exec_binding.py` | 0 号窗口返回原对象；只平移激活与块表；`mram_offset` 不变且编译缓存键跨窗口相等；原计划不被改、窗口间不别名 |
| `tests/test_hal_numpy.py` | 不同上下文可复用命令 id；`bound_pos/result_of` 各读各的；同 DPU 不同泳道可并发、默认泳道仍串行 |
| `tests/test_parallel_requests.py` | 4 请求并发 token 逐字等于单跑（`kv_slot_tokens∈{max_seq,2}`）；混合 prefill/decode；5 次确定性；块表窗口独立 |
| `tests/test_parallel_overlap.py` | 同 DPU 上两窗口 launch 区间真重叠；后提交窗口的越级；窗口 1 内核与窗口 0 host 命令重叠 |
| `tests/test_concurrency_llama2_7b.py`（slow，真实 7B/8 DPU，本地 344s 通过） | 7B 两请求并发 token 与单跑一致 + 同 DPU 跨窗口重叠 |

`test_parallel_overlap` 的关键判据不是「有重叠」，而是**越级**：后提交的
窗口 1 命令要在更早提交的窗口 0 命令结束前就在同一 DPU 上开始。这排除了
「只是两条流各自跑、恰好时间相邻」的假象，证明共享线程池配合每泳道独立
FIFO 确实产生了交错。

## 不足与下一步

- **建模选择**：同 DPU 允许两条请求的命令并发执行。数值安全（窗口、块表、
  KV 都不相交），但相对「单物理流」是乐观抽象。若硬件组要求更忠实，后续可
  做前瞻合并调度。
- **host 仍串行**：`_host_lock` 保留，跨请求加速上限由 host 重分布占比决定。
- **内存倍数**：激活与块表都 ×P；大模型需按预算调小 `PIM_PARALLEL_SLOTS`。
- 不在本阶段：跨请求调度策略（vLLM 负责）、前缀缓存、乱序合并调度器、
  真硬件 `hal_vendor`。
