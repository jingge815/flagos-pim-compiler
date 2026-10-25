# vLLM 执行后端接入（2026-10-23）

> **后续变更（2026-10-24）**：本文描述的「静态槽分区 / 多槽 KV」已被
> 「固定大小槽 + 间接寻址」取代，`num_slots`、`multi_slot_kv_specs`、
> `slot_id`、`kv_specs_multi_slot()` 等名称均已过时。vLLM 侧参数改为
> `num_request_slots`（= `max_num_seqs`）与 `kv_slot_tokens`
> （= `cache_config.block_size`）。
>
> **后续变更（2026-10-25）**：页表改为**每台 DPU 一份全局块表缓冲 + 全局槽池**，
> `request_slot` 退出设备寻址；池大小由 `kv_pool_slots` / `PIM_KV_POOL_SLOTS`
> 决定。详见 `docs/kv-global-block-table-20261025.md`。

## 目标

把 PIM 静态编译后端接到 vLLM V1 引擎上：vLLM 负责 HTTP、调度和连续批处理，PIM 负责执行。
最终可以用 `scripts/serve_pim.py` 起一个 OpenAI 兼容服务。

## 核心约束

MRAM 地址由硬件绑定，没有虚拟地址层，做不了 PagedAttention。替代方案是
**固定大小槽 + 间接寻址**（早期版本是「一个请求独占一整段连续 MRAM」，
已被取代）：

| 项目 | vLLM 常规做法 | PIM 做法 |
| --- | --- | --- |
| 并发单元 | 分页 block，运行期自由拼接 | 固定 `slot_tokens` 的物理槽，运行期分配 |
| KV 位置 | 每请求 block 表动态映射 | 每台 DPU 一份共享块表缓冲，每步由 host 覆盖 |
| 容量 | 由显存决定 | 全局槽池 `kv_pool_slots` |
| 并发上限 | 由显存决定 | `max_num_seqs`（host 侧记账） |
| 改并发 | 改配置即可 | 必须重新编译 |

vLLM 看到的 block 是账面数字：`num_gpu_blocks = kv_pool_slots`，默认
`max_num_seqs × ceil(max_seq / block_size)`。槽几何、地址公式与块表布局见
`docs/kv-global-block-table-20261025.md`。在串行执行且没有块共享的前提下，
把池调大不会提高并发数。

## 改动清单

### 内存层

| 文件 | 改动 |
| --- | --- |
| `memory/kv_layout.py` | `build_kv_layout()` 规划块表缓冲与全局槽池；`write_block_table()` 写逻辑块映射 |
| `memory/mem_planner.py` | `plan_dpu()` 接受 `kv_pool_slots`，KV 区按块表 + 槽池整体计入 |

### 执行层

| 文件 | 改动 |
| --- | --- |
| `backend/hal_numpy.py` | `bind_inputs()` 透传 `pos`、`num_tokens` |
| `runtime/kernels.py` | `sdpa_kv_info()` 输出槽几何与块表地址；`sdpa_kernel` 读块表后按物理槽读写 KV |
| `runtime/executor.py` | `execute_plan()` 透传 `pos` / `num_tokens`；新增 `KVSlotPool`、`KVBlockTable`、`RequestSlot`、`RequestTable`、`step_request()` |
| `runtime/compile.py` | `compile_llama2()` 新增 `num_request_slots`、`kv_slot_tokens`、`kv_pool_slots`；`CompiledModel` 新增对应字段 |

### vLLM 适配层（新增）

| 文件 | 作用 |
| --- | --- |
| `backend/vllm_worker.py` | `PIMWorker`：把 vLLM 每步的 `SchedulerOutput` 翻译成逐请求执行 |
| `backend/vllm_executor.py` | `PIMExecutor`：单进程执行器，只把 worker 类换成 `PIMWorker` |
| `scripts/serve_pim.py` | 用 `PIMExecutor` 启动 OpenAI 兼容服务 |

## 执行流程

```
HTTP 请求
   │
   ▼
vLLM 调度器 ──SchedulerOutput──▶ PIMWorker.execute_model
                                    │
                  ┌─────────────────┼──────────────────┐
                  ▼                 ▼                  ▼
            新请求入表        被抢占请求重置        逐请求执行一步
            RequestTable       slot.reset()        step_request()
                                                       │
                                                       ▼
                                      prefill 图 / decode 图（读共享块表）
                                                       │
                                                       ▼
                                                   NumpyBackend
```

`step_request()` 每次只执行一张图并产出一个 token：没做过 prefill 的请求走 prefill 图，
否则走 decode 图。这和 vLLM「一次 `execute_model` 一步」的语义对齐。

## 关键结构体

### `RequestSlot`

单个请求的状态：`request_slot`（host 侧并发记账，不参与设备寻址）、
`blocks`（逻辑块 → 物理槽）、`valid_len`、`prompt_ids`、`generated`、
`is_prefill_done`。

### `RequestTable`

管理 `num_request_slots` 个请求槽和共享物理槽池的借还。`add_request()` 在槽满时
直接抛 `RuntimeError`；`reset()` 供被抢占的请求按全量 token 重放 prefill。
设备侧只有一份块表缓冲，每步执行前由 `step_request()` 覆盖。

### `CompiledModel` 新增字段

| 字段 | 含义 |
| --- | --- |
| `num_request_slots` | host 侧并发上限（请求槽数） |
| `kv_slot_tokens` | 一个物理槽容纳的 token 数（= `block_size`） |
| `kv_pool_slots` | 全局物理槽池大小（= vLLM 的 `num_gpu_blocks`） |
| `prefill_seq_len` | prefill 图固定的序列长度 |

## 关键实现细节

1. **槽几何必须在 `plan_dpu()` 里构建。** `plan_dpu()` 会把 KV 区挪到权重区末尾并回填
   `kv_base`，页表区和槽池地址都从这里推导；提前构建会指向权重区。
2. **块尺寸跟随 vLLM。** `kv_slot_tokens` 等于 `cache_config.block_size`
   （CUDA 平台默认 16），每请求块数为 `ceil(max_seq / block_size)`。
   池大小默认 `num_request_slots × 每请求块数`，可通过 `PIM_KV_POOL_SLOTS`
   覆盖；vLLM 分到的 block 数必须等于池大小，否则启动失败。
3. **不支持分块 prefill。** 静态图一次跑完整个 prompt；若调度器把 prompt 拆到多步，
   worker 直接抛错。服务端应把 `--max-num-batched-tokens` 设到至少
   `并发数 × 最长 prompt`。短 prompt 右侧补齐到 `PIM_PREFILL_SEQ_LEN`，补齐位被因果掩码屏蔽。
4. **采样固定贪心。** 运行时用 argmax，温度等采样参数不生效。
5. **引擎必须与 worker 同进程。** `serve_pim.py` 和测试都设置
   `VLLM_ENABLE_V1_MULTIPROCESSING=0`，否则 worker 类无法解析。
6. **不要关闭全局梯度模式。** `torch.set_grad_enabled(False)` 会让 `torch.export`
   多导出 `aten.expand` 和 `wrap_with_autocast` 两个主机算子，与编译器平时
   校验的图不一致；权重搬运本身已经 `detach()`，无需关闭梯度。

## 启动方式

```bash
python scripts/serve_pim.py --model /path/to/Llama-2-7b-hf \
    --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096
```

vLLM 0.14 的 `--model` 默认值是 `Qwen/Qwen3-0.6B`，必须显式指定本地模型目录。

## 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PIM_PREFILL_SEQ_LEN` | `max_model_len` | prefill 图固定长度，prompt 补齐到它 |
| `PIM_NUM_DPUS` | 4 | PIM 内部张量并行的 DPU 数 |
| `PIM_MRAM_BYTES` | 4 GiB | 单台 DPU 的 MRAM 总量 |
| `PIM_SYS_RESERVE_BYTES` | 64 MiB | 系统预留，不参与三区规划 |
| `PIM_KV_POOL_SLOTS` | `max_num_seqs × 每请求块数` | 全局物理槽池大小，必须 ≥ 每请求块数 |

## 已实施的测试

| 测试文件 | 数量 | 验证点 |
| --- | --- | --- |
| `tests/test_kv_slots.py` | 9 | 槽几何、块表乱序映射与重写、池耗尽与复用、未分配项报错 |
| `tests/test_slot_isolation.py` | 3 | 两请求物理槽不相交、块表切换、`pos`/`num_tokens` 透传 |
| `tests/test_multi_request_decode.py` | 4 | 两路并发生成与单独运行一致、`slot_tokens=2` 与整段一致、槽满报错 |
| `tests/test_vllm_executor.py` | 1 | vLLM 引擎经 `PIMExecutor` 跑通两个并发请求 |

另外用微型模型做过一次 HTTP 冒烟：启动 `serve_pim.py` 后并发发两个
`/v1/completions` 请求，各自返回 `completion_tokens=3`、`finish_reason=length`。

受影响的既有测试（`test_kernels`、`test_hal_numpy`、`test_mem_planner`、
`test_kv_layout`、`test_exec_plan_gen`、`test_runtime_compiled_coverage`、
`test_executor_llama2_7b`、`test_op_semantics` 等）共 250 项全部通过。

全量按默认顺序运行时，`test_partition`、`test_placement_export`、`test_gml_export`
曾随执行顺序偶发失败。根因不在后端代码：有的 7B 用例在 module 级 fixture 里
`torch.set_grad_enabled(False)` 后不恢复，之后的 `torch.export` 会导出另一张图
（主机侧多出 `aten.expand` 和 `wrap_with_autocast`）。现在 `tests/conftest.py`
在每个用例结束后把梯度开关恢复成会话开始时的状态，新增的多请求测试也去掉了
这行全局设置，三个文件按任意顺序运行都通过。

## 当前不足

- 只支持贪心采样，`temperature`、`top_p` 等参数无效。
- prompt 超过 `PIM_PREFILL_SEQ_LEN` 或 `max_model_len` 时无法运行。
- 不支持分块 prefill，长 prompt 需要一次性排进 batch。
- 不支持前缀缓存、LoRA、投机解码、多卡张量并行（PIM 内部已有 TP）。
- 多槽仍是串行逐槽执行，没有利用 `NumpyBackend` 的线程池并发。
- 改并发数必须重新编译。
