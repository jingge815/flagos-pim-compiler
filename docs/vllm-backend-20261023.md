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
>
> **后续变更（2026-11-01）**：TP/PP、LoRA、投机解码、量化、dtype、模型架构、
> 结构化输出等能力在启动期与请求期显式校验，见
> `docs/vllm-capabilities-20261101.md`。

## 目标

把 PIM 静态编译后端接到 vLLM V1 引擎上：vLLM 负责 HTTP、调度和连续批处理，PIM 负责执行。
最终可以用 `scripts/serve_pim.py` 起一个 OpenAI 兼容服务。

## 核心约束

> **更正（2026-10-26）**：早期版本这里写「MRAM 没有虚拟地址层，做不了
> PagedAttention」，这是错的。PagedAttention 本质是**软件层的块表间接寻址**
> ——逻辑块 → 物理块，由算子在设备内核里查表拼接，不需要 MMU 或虚拟地址。
> 我们已实现分页 KV：固定大小物理槽 + 块表 + 内核按表 gather
> （`pim.paged_attn`）。与 vLLM 的真实差异只剩「几何是编译期常量、块表窗口
> 数固定、无前缀缓存/COW」，不再是「做不了」。

MRAM 地址由硬件绑定，没有虚拟地址层。替代方案是**固定大小槽 + 块表间接
寻址**（早期版本是「一个请求独占一整段连续 MRAM」，已被取代）：

| 项目 | vLLM 常规做法 | PIM 做法 |
| --- | --- | --- |
| 并发单元 | 分页 block，运行期自由拼接 | 固定 `slot_tokens` 的物理槽，运行期分配 |
| KV 位置 | 每请求 block 表动态映射 | 每台 DPU 一份共享块表缓冲，每步由 host 覆盖 |
| 间接寻址 | 块表 → 物理块，内核查表 | 相同：`pim.paged_attn` 内核按块表 gather |
| 容量 | 由显存决定 | 全局槽池 `kv_pool_slots` |
| 并发上限 | 由显存决定 | `max_num_seqs`（host 侧记账） |
| 改几何/并发 | 改配置即可 | 必须重新编译 |

vLLM 看到的 block 是账面数字：`num_gpu_blocks = kv_pool_slots`，默认
`max_num_seqs × ceil(max_seq / block_size)`。槽几何、地址公式与块表布局见
`docs/kv-global-block-table-20261025.md`。多请求已可并发执行（Phase 1），
块表窗口数 `P` 决定同时在飞的请求数，见
`docs/executor-parallel-phase1-20261026.md`。

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
3. **支持分块 prefill（2026-11-01 起）。** 每段按自身长度选 prefill 桶，从任意 `pos`
   续写 KV，中间段不采样，最后一段才出第一个 token；约束变成
   `--max-num-batched-tokens ≤ PIM_PREFILL_SEQ_LEN`（一段要装进一张图），prompt 长度
   只受 `max_model_len` 限制。详见 `docs/vllm-chunked-prefill-20261101.md`。
4. **采样在主机侧完成。** 2026-11-01 起支持 temperature/top-k/top-p、三种惩罚、
   seed 与 logprobs，未实现的参数显式报错；详见 `docs/vllm-sampling-20261101.md`。
5. **引擎必须与 worker 同进程。** `serve_pim.py` 和测试都设置
   `VLLM_ENABLE_V1_MULTIPROCESSING=0`，否则 worker 类无法解析。
6. **不要关闭全局梯度模式。** `torch.set_grad_enabled(False)` 会让 `torch.export`
   多导出 `aten.expand` 和 `wrap_with_autocast` 两个主机算子，与编译器平时
   校验的图不一致；权重搬运本身已经 `detach()`，无需关闭梯度。

## 启动方式

```bash
python scripts/serve_pim.py --model /path/to/Llama-2-7b-hf \
    --max-model-len 512 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --pim-num-dpus 4 --pim-mram-bytes 4294967296
```

vLLM 0.14 的 `--model` 默认值是 `Qwen/Qwen3-0.6B`，必须显式指定本地模型目录。

2026-11-01 起 PIM 自己的开关有命令行形式，解析后转成同名环境变量：

| 选项 | 环境变量 | 含义 |
| --- | --- | --- |
| `--pim-num-dpus` | `PIM_NUM_DPUS` | PIM 内部张量并行的 DPU 数 |
| `--pim-mram-bytes` | `PIM_MRAM_BYTES` | 单台 DPU 的 MRAM 字节数 |
| `--pim-kv-pool-slots` | `PIM_KV_POOL_SLOTS` | 全局物理槽池大小 |
| `--pim-parallel-slots` | `PIM_PARALLEL_SLOTS` | 同时在飞的请求窗口数 |
| `--pim-prefill-seq-len` | `PIM_PREFILL_SEQ_LEN` | prefill 图固定长度 |

vLLM 的 `--help` 只列配置组名，这组选项用 `--help=PIMConfig` 查看。

`main()` 在 `run_server` 之前跑一次 `preflight()`：槽池容量、prefill 图长度、
`max_num_batched_tokens` 预算、MRAM 预算有问题时**一次列全**并以非零码退出，
不会等权重加载完才在编译期报错。详见 `docs/vllm-serving-preflight-20261101.md`。

## 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `PIM_PREFILL_SEQ_LEN` | `max_model_len` | **最长** prefill 桶；prompt 补齐到选中的桶 |
| `PIM_PREFILL_SEQ_LENS` | `128/512/2048` + 最长桶 | 逗号分隔的 prefill 桶表，见 `docs/vllm-prefill-buckets-20261101.md` |
| `PIM_NUM_DPUS` | 4 | PIM 内部张量并行的 DPU 数 |
| `PIM_MRAM_BYTES` | 4 GiB | 单台 DPU 的 MRAM 总量 |
| `PIM_SYS_RESERVE_BYTES` | 64 MiB | 系统预留，不参与三区规划 |
| `PIM_KV_POOL_SLOTS` | `max_num_seqs × 每请求块数 + 1` | 全局物理槽池大小，必须 ≥ 每请求块数；多出的一块给 vLLM 的 null 块 |

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

## 当前不足（2026-11-02 更新）

原先这一节列的「不支持前缀缓存、分块 prefill」已经落地，按现状重写；各项的
设计与验收证据见对应的 `docs/vllm-*-20261*.md`。

- **版本**：只支持 vLLM 0.14.x，启动时校验；升级到 0.31 的差距见
  [`vllm-031-gap-20261102.md`](./vllm-031-gap-20261102.md)。
- **架构**：只有 Llama（`model_type == "llama"`）。GQA 可用；`rope_scaling`、
  `sliding_window` 明确拒绝，其余架构换权重文件即可扩表。
- **采样**：贪心、温度、top-k/top-p、三种惩罚、seed 与 logprobs 已支持；`min_p`、
  `logit_bias`、结构化输出等仍未实现（显式报错）。
- **并行**：vLLM 层 `tensor_parallel_size` / `pipeline_parallel_size` 必须为 1，
  张量并行由 PIM 内部按 DPU 完成；不支持 LoRA、投机解码、量化（只收 fp16）。
- **执行**：只有单进程（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）与 `NumpyBackend`，
  多进程与真硬件等 `hal_vendor` 落地（F2，阻塞中）。
- **前缀缓存、分块 prefill、抢占**已支持：命中段只读、分块按桶跑、
  抢占后整段或从命中长度重放。
- **并发**：活跃请求按 `P` 分 wave，wave 内每请求各用一份激活窗口和块表窗口。
  同 DPU 允许两条请求交错是建模选择，主机侧仍由 `_host_lock` 串行。
- **内存**：KV 槽池与并行窗口默认自动定容；激活区与块表随 `P` 倍增。
- 改几何/并发数必须重新编译。
- **测试**：`execute_model` 翻译层已有单测（`tests/test_vllm_translate.py`），
  端到端仍是 `tests/test_vllm_executor.py`。
