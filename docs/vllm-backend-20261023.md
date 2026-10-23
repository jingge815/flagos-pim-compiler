# vLLM 执行后端接入（2026-10-23）

## 目标

把 PIM 静态编译后端接到 vLLM V1 引擎上：vLLM 负责 HTTP、调度和连续批处理，PIM 负责执行。
最终可以用 `scripts/serve_pim.py` 起一个 OpenAI 兼容服务。

## 核心约束

MRAM 地址由硬件绑定，没有虚拟地址层，做不了 PagedAttention。替代方案是**静态槽分区**：

| 项目 | vLLM 常规做法 | PIM 做法 |
| --- | --- | --- |
| 并发单元 | 分页 block，运行期自由拼接 | 编译期预留 `num_slots` 个槽 |
| KV 位置 | 由 block 表动态映射 | 每槽独占一段连续 MRAM，编译期定死 |
| 并发上限 | 由显存决定 | 等于 `num_slots`（= `max_num_seqs`） |
| 改并发 | 改配置即可 | 必须重新编译 |

vLLM 看到的 block 只是账面数字：每槽按调度器块尺寸记 `ceil(max_seq / block_size)` 个 block，
总 block 数正好等于 `num_slots × 每槽块数`，真正限制并发的是 `max_num_seqs = num_slots`。

## 改动清单

### 内存层

| 文件 | 改动 |
| --- | --- |
| `memory/kv_layout.py` | 新增 `kv_specs_multi_slot()`：把单槽 KV 规格复制 `num_slots` 份，步长取单槽已分配字节数 |
| `memory/mem_planner.py` | `plan_dpu()` 新增 `num_slots` 参数，KV 区按槽数放大 |

### 执行层

| 文件 | 改动 |
| --- | --- |
| `backend/hal_numpy.py` | `bind_inputs()` 新增 `slot_id`，暴露 `bound_slot_id` 属性 |
| `runtime/kernels.py` | `sdpa_kv_info()` 可输出 `k_offs_by_slot` / `v_offs_by_slot`；`sdpa_kernel` 按当前槽取 KV 偏移 |
| `runtime/executor.py` | `execute_plan()` 透传 `slot_id`；新增 `RequestSlot`、`RequestTable`、`step_request()`；`run_one_step()` 改为每请求只走一步 |
| `runtime/compile.py` | `compile_llama2()` 新增 `num_slots`；`CompiledModel` 新增 `num_slots`、`multi_slot_kv_specs`、`prefill_seq_len` |

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
                                      prefill 图 / decode 图（含 slot_id）
                                                       │
                                                       ▼
                                                   NumpyBackend
```

`step_request()` 每次只执行一张图并产出一个 token：没做过 prefill 的请求走 prefill 图，
否则走 decode 图。这和 vLLM「一次 `execute_model` 一步」的语义对齐。

## 关键结构体

### `RequestSlot`

单个请求的状态：`slot_id`、`valid_len`、`prompt_ids`、`generated`、`is_prefill_done`。

### `RequestTable`

管理 `num_slots` 个槽的借还。`add_request()` 在槽满时直接抛 `RuntimeError`；
`reset()` 供被抢占的请求按全量 token 重放 prefill。

### `CompiledModel` 新增字段

| 字段 | 含义 |
| --- | --- |
| `num_slots` | 编译期预留的请求槽数 |
| `multi_slot_kv_specs` | `dpu_id → 槽号 → KV 规格`；单槽时为 `None` |
| `prefill_seq_len` | prefill 图固定的序列长度 |

## 关键实现细节

1. **多槽 KV 规格必须在 `plan_dpu()` 之后构建。** `plan_dpu()` 会把 KV 区挪到权重区末尾
   并回填 `kv_base`；提前构建会让每槽偏移指向权重区，破坏权重。
2. **块尺寸跟随 vLLM。** worker 上报的 `FullAttentionSpec.block_size` 等于
   `cache_config.block_size`（CUDA 平台默认 16），每槽块数为 `ceil(max_seq / block_size)`。
   这样调度器算出的总 block 数正好够 `num_slots` 个满长请求。
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

## 已实施的测试

| 测试文件 | 数量 | 验证点 |
| --- | --- | --- |
| `tests/test_multi_slot_kv.py` | 4 | 槽间偏移不重叠、总字节数按槽数放大、非法槽数报错 |
| `tests/test_slot_isolation.py` | 3 | 写一个槽不影响另一个槽、`slot_id` 透传 |
| `tests/test_multi_request_decode.py` | 3 | 两路并发生成与单独运行逐 token 一致、槽可复用、槽满报错 |
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
