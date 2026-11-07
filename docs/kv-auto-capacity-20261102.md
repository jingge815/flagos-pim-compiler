# KV 池与并行窗口自动定容（2026-11-02）

## 改了什么

以前 `PIM_KV_POOL_SLOTS` 与 `PIM_PARALLEL_SLOTS` 必须手工给：给大了要到编译期
才报 per-DPU 内存超限，给小了白白浪费 MRAM。现在只要给模型、`max_model_len`
和 `max_num_seqs`，编译期按剩余 MRAM 反推槽池大小，并在放不下满并发时收窄
并行窗口；两个环境变量都变成可选覆盖。

| 文件 | 改动 |
| --- | --- |
| `memory/mem_planner.py` | 新增 `fit_pool_slots`：减去权重、块表、激活窗口与内核 tile 后，按单槽字节数求最多能放几个槽 |
| `runtime/compile.py` | `kv_pool_slots`/`num_parallel_slots` 为空时自动定容，先用最小槽池探一次几何 |
| `backend/vllm_worker.py` | 两个环境变量未设时把 `None` 交给编译期，`load_model` 后回读编译结果 |
| `scripts/serve_pim.py` | 不设槽池时跳过 KV 字节估算；显式槽池小于「一个满长请求 + 一块」时报错 |
| `tests/test_auto_capacity.py` | 新增 5 例（含不设 `PIM_*` 环境变量的 vLLM 端到端） |

## 容量公式

三区布局自下而上是权重 → KV 池 → 激活窗口，所以槽池只把激活区往后推，
不改变权重区终点 `kv_base`，也不改变激活窗口步长 `act_stride`：

```text
budget = mram_bytes - sys_reserve_bytes
stride = align_up(max(act_extent), align)
room   = budget - kernel_mram - kv_base - bt_slots × block_table_bytes
                                      - P × stride
cap    = floor(room / slot_bytes)          # 每台 DPU 各算一次，取最小
pool   = min(num_request_slots × blocks_per_request + 1, cap)
```

`kernel_mram` 是算子编译器的分块 tile 占用，与三区共用同一块 MRAM，必须减掉。
多出的一块留给 vLLM 的 null 块：`BlockPool` 构造时从空闲队列头永久拿走一块，
可用块数是 `num_blocks - 1`。

`kv_base` 与 `act_extent` 与槽池大小无关，所以「先用最小槽池规划一份蓝图，
再拿它的占用反推容量」是精确的，不需要迭代求解。编译期只在需要自动定容时
多规划一次。

## 窗口与槽池如何收敛

```text
P = min(2, num_request_slots)            # 块表窗口跟随 P
           │
           ├─ 最大容量仍装不下一个满长请求 ──► P 减 1，重算
           └─ 装得下 ──► pool = min(N×B + 1, cap)
                          └─ pool ≤ B ──► 报错，写出所需槽数与当前能放的槽数
```

显式给了 `kv_pool_slots` 时不自动改，装不下由 `plan_dpu` 报「内存超限」，
消息里同时给出峰值与可用预算。

## 回报给 vLLM

`determine_available_memory` 报 `kv_pool_slots × 单页字节数`，于是 vLLM 算出的
`num_gpu_blocks` 正好等于槽数；`initialize_from_config` 再校验一次，不一致直接
启动失败而不是跑出错误结果。自动定容的结果在 `load_model` 结束后写回 worker，
`num_parallel_slots` 也一并更新给 `run_one_step`。

## 测量（4 DPU / 4 层 / hidden 1024 / 4 请求 × 8 步 decode）

同一份 `num_parallel_slots=4` 的蓝图，只改每步的 wave 宽度：

| 窗口数 | 8 步 wall | 每步 | DPU 利用率 | 生成 token |
| --- | --- | --- | --- | --- |
| 1 | 27 557 ms | 3 445 ms | 60.6% | 三组一致 |
| 2（默认） | 21 403 ms | 2 675 ms | 71.2% | 三组一致 |
| 4 | 21 360 ms | 2 670 ms | 73.8% | 三组一致 |

P=2 相对串行吞吐 ×1.29；再开到 4 只多 0.2%，因为 host 侧仍由 `_host_lock`
串行，加速已经吃满。默认取 2 不再额外占激活内存，也不再开多余线程。

不设任何 `PIM_*` 环境变量时（微型 Llama，`max_model_len=32`、块 16、
`max_num_seqs=2`）：`blocks_per_request=2`、`pool=5=2×2+1`、`P=2`，
vLLM 的 `num_gpu_blocks` 与它相等，请求正常生成。

## 已实施的测试

`tests/test_auto_capacity.py`（5 例）：

| 测试 | 验证点 |
| --- | --- |
| `test_fit_pool_slots_counts_remaining_mram` | 容量 =（预算 − 权重 − 块表 − 激活窗口）÷ 单槽字节；每多一个窗口少一个槽 |
| `test_auto_capacity_fits_where_fixed_config_fails` | 同一预算下显式满配报错，自动定容把槽数收进 `(B, N×B+1)` |
| `test_explicit_pool_over_budget_names_need_and_available` | 超预算时消息含 `peak=` 与 `可用预算=` |
| `test_explicit_pool_that_fits_is_kept` | 显式槽数装得下就原样保留 |
| `test_vllm_starts_without_pim_env` | 不设任何 `PIM_*`：启动成功，`num_gpu_blocks` 等于定容槽数，`P=2` |

## 当前不足

- 默认窗口数上限 2 是测量折中值，不是求解结果；更大机器上开更多窗口是否有收益
  没有数据。
- 容量只按编译期静态占用估算，`sys_reserve_bytes` 之外的运行时开销（线程栈、
  vLLM 侧数据结构）不在模型里。
- 显式槽池仍可能过小：预检只用 KV 区粗估 MRAM，权重不计入，真正的判据仍在
  编译期的 `plan_dpu`。
