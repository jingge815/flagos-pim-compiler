# 全局块表缓冲 + 全局物理槽池（2026-10-25）

## 目标

上一版（`docs/kv-slots-20261024.md`）把 KV 改成了「固定大小槽 + 每请求一份 MRAM 页表」，
但页表按 `num_request_slots` 静态分区，物理槽池按「并发数 × 每请求块数」预留。
结果是内存占用的最坏情况和旧方案一样，只是 KV 的摆法变了；请求与页表分区绑定，
无法让池容量独立于并发数。

本次把设备侧改成 vLLM 式结构：

- **每台 DPU 只保留一份块表缓冲**，每步执行前由 host 写入当前请求的
  「逻辑块 → 物理槽」列表，`request_slot` 退出设备寻址路径；
- **物理槽池全局一份**，大小由编译参数 `kv_pool_slots` 决定，默认为
  `num_request_slots × ceil(max_seq / slot_tokens)`；
- 执行仍串行，不做块共享、不做并发执行。

vLLM 的块表本来就是每请求的，全局的是「分配器 + 容量」；这一步让 PIM 侧与
这个模型对齐。

## 内存布局

每台 DPU 的 KV 区 = 一块共享块表缓冲 + 一个全局槽池：

```
kv_base
 ├─ block_table_bytes = align_up(blocks_per_request * 4, align)
 └─ slots_base = kv_base + block_table_bytes
     pool_slots × slot_bytes

blocks_per_request = ceil(max_seq / slot_tokens)
kv_allocated_bytes = block_table_bytes + pool_slots * slot_bytes
```

槽内几何不变：

```
plane_stride = align_up(slot_tokens * head_dim * dtype_bytes, align)
slot_bytes   = 2 * len(layers) * len(kv_heads) * plane_stride
```

`build_kv_layout(spec, align, pool_slots=None)`：

- `pool_slots is None` 时默认取 `blocks_per_request`（刚好装下一个满长请求）；
- 要求 `pool_slots >= blocks_per_request`，否则抛 `ValueError`。

## 地址公式

DPU `d`、逻辑块 `b = position // slot_tokens`、块内行 `t = position % slot_tokens`：

```
物理槽   s = block_table[b]                      # int32，-1 表示未分配
平面地址 = kv_off[(layer, head, k|v)] + s * slot_bytes
行地址   = 平面地址 + t * head_dim * dtype_bytes
```

内核查表时每个 owner 只读一次块表：

```
block_table = hal.read_local(owner, bt_base, (max_blocks,), np.int32)
```

历史 gather（`_gather_kv_history`）仍按逻辑块逐块查表，只在 `start < hist_len`
时才读。

## 全局块表缓冲

`write_block_table(spec, hal, block_table)`：

- 始终写满 `blocks_per_request` 个 int32，不足补 `-1`。**必须整段写**，
  防止短请求继承长请求残留在表尾的块号；
- 写到 `spec.block_table_base`（= `kv_base`），没有 `request_slot` 偏移；
- 校验 `len(block_table) <= blocks_per_request`，以及每个块号
  `0 <= b < pool_slots`。

## 执行流程与不变量

```
step_request
  ├─ prefill: blocks.ensure(ceil(prompt_len / slot_tokens))
  ├─ decode : blocks.ensure(ceil((valid_len+1) / slot_tokens))
  ├─ _write_block_table(compiled, hal, slot)   # 覆盖每台 DPU 的同一块缓冲
  └─ execute_plan(..., pos=..., num_tokens=...)
```

不变量：

1. **共享缓冲依赖串行执行**：`execute_plan` 等待本计划全部命令结束，
   `step_request` 下一步写表时不能有上一步的内核还在读表。现有实现已满足。
2. 任何 launch 之前都必须先写块表，不再有「按 `request_slot` 找对自己那段」的兜底。
3. 表尾补 `-1` 只用于暴露越界读，正常路径不会读到尾部。

## 槽池容量

| 参数 | 位置 | 含义 |
| --- | --- | --- |
| `kv_pool_slots` | 编译参数 / `CompiledModel` | 全局物理槽池大小 |
| `PIM_KV_POOL_SLOTS` | 环境变量 | vLLM 侧覆盖池大小 |
| `num_request_slots` | host 侧 | 并发上限记账，不参与设备寻址 |

`kv_pool_slots > num_request_slots × blocks_per_request` 只表示池里余量更多。
在 `max_num_seqs` 仍限制并发、且没有块共享的前提下，它**不会**提高并发数；
调大池子是为后续按真实容量配置和细粒度调度留接口。

## 与 vLLM 块表的对应关系

| vLLM | 本方案 |
| --- | --- |
| `block_table[request_id]` 每请求一块表 | 共享缓冲，每步由 host 覆盖 |
| `block_manager` 分配物理 block | `KVSlotPool` + `KVBlockTable` |
| `num_gpu_blocks` | `kv_pool_slots`（vLLM 账面块数） |
| `block_size` | `kv_slot_tokens` |

受 MRAM 无虚拟地址层限制，设备侧只能持有一份表；因此执行必须串行，
并发上限仍由 `max_num_seqs` 决定。

## 改动文件

| 文件 | 改动 |
| --- | --- |
| `memory/kv_layout.py` | `page_table_*` → `block_table_*`；`build_kv_layout(..., pool_slots=None)`；`write_page_table` → `write_block_table` |
| `memory/mem_planner.py` | `plan_dpu(..., kv_pool_slots=None)` |
| `runtime/compile.py` | `compile_llama2(..., kv_pool_slots=None)`；默认值与一致性检查改按显式池大小 |
| `runtime/kernels.py` | SDPA payload `dpus` 改为 `{bt_base, slot_bytes, max_blocks}`；每 owner 读一次块表 |
| `runtime/executor.py` | `execute_plan` 删除 `request_slot`；`_write_page_tables` → `_write_block_table` |
| `backend/hal_numpy.py` | 删除 `request_slot` 绑定与 `bound_request_slot` |
| `backend/vllm_worker.py` | `PIM_KV_POOL_SLOTS`；`determine_available_memory` 按池大小记账 |

## 已实施的测试

| 测试文件 | 验证点 |
| --- | --- |
| `tests/test_kv_slots.py` | 块表几何；默认池 = 满长单请求块数；乱序映射；长→短请求重写后表尾为 -1；块号越界/表太长报错；池只够一个请求时第二个满长请求耗尽；重写块表切换目标 |
| `tests/test_kv_layout.py` | `block_table_bytes` 精确值；`write_block_table` 补齐与校验；注意力数值与 torch 对拍 |
| `tests/test_slot_isolation.py` | 两请求物理槽不相交、块表重写切换；`bind_inputs` / `execute_plan` 的 `pos`、`num_tokens` 透传 |
| `tests/test_kernels.py` | 手工 SDPA payload 新格式 + 块表写入 |
| `tests/test_mem_planner*.py`、`tests/test_kv_layout_llama2_7b.py` | 规划字段改名、块表缓冲 + 池正好铺满 KV 区 |
| `tests/test_executor_llama2_7b.py`、`tests/test_concurrency_llama2_7b.py`、`tests/test_strategy_*.py` | 写 identity 块表后整图执行、KV 与 HF cache 对拍 |
| `tests/test_multi_request_decode.py`、`tests/test_vllm_executor.py` | 两路生成与单独运行一致、vLLM 引擎跑通 |

验证判据：

- 块表乱序映射（0→2、1→0）下 prefill/decode 结果与顺序映射一致；
- 长→短请求重写后表尾全为 -1，不继承上一请求的物理槽；
- `kv_pool_slots` 显式调大后几何/容量按值变化，`pool_slots < blocks_per_request` 报错；
- `PIM_KV_POOL_SLOTS` 未设时，vLLM 默认路径行为与改动前一致。

## 当前不足

- 多请求仍串行执行（`NumpyBackend` 的 `bind_inputs`/events 是实例全局）。
- 没有物理槽共享、前缀缓存和写时复制。
- vLLM 调度器给出的 `block_ids` 尚未消费，块表由 PIM 自己的分配器维护。
- 改 `kv_pool_slots` 或 `kv_slot_tokens` 必须重新编译。
