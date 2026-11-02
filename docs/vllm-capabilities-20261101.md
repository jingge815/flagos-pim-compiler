# vLLM 能力矩阵校验（2026-11-01）

## 改了什么

此前 `PIMWorker` 只在两个地方硬拦：前缀缓存（`load_model`）和分块 prefill
（`execute_model` 里逐请求判断）。TP/PP、LoRA、投机解码、量化、非 Llama 架构、
非 fp16、结构化输出都会一路跑下去，最后算错也不报错。

本次把这些检查集中成三个纯函数，在**启动期**和**请求期**两个入口调用。

| 文件 | 改动 |
| --- | --- |
| `backend/vllm_worker.py` | 新增 `validate_engine_config`、`validate_scheduler_output`、`validate_new_request` 与 `_unsupported`；`__init__` / `execute_model` / `_add_request` 各调用一次；`load_model` 里的前缀缓存检查上移 |
| `tests/test_vllm_capabilities.py` | 新增 19 例单测，逐项覆盖拒绝分支 |

## 校验点

### 引擎级（`PIMWorker.__init__`）

| 参数 | 允许值 | 拒绝理由 |
| --- | --- | --- |
| `tensor_parallel_size` | 1 | PIM 内部已按 DPU 做张量并行 |
| `pipeline_parallel_size` | 1 | 只编译单卡图 |
| `lora_config` | `None` | 权重在编译期写死 |
| `speculative_config` | `None` | 不支持投机解码 |
| `enable_prefix_caching` | `False` | 无前缀缓存 / COW |
| `quantization` | `None` | 只支持 fp16 权重 |
| `dtype` | `float16` | 设备内核按 fp16 写死 |
| `hf_config.model_type` | `llama` | 只导出 Llama 蓝图 |
| `max_num_batched_tokens` | ≥ `prefill_seq_len × max_num_seqs`（仅在开启分块 prefill 时检查） | 静态图一步跑完整个 prompt |

槽几何的两项检查维持原处：
`__init__` 检查 `PIM_KV_POOL_SLOTS ≥ blocks_per_slot`，
`initialize_from_config` 检查 vLLM 分到的块数 `== kv_pool_slots`。

### 请求级（`execute_model` / `_add_request`）

| 输入 | 拒绝理由 |
| --- | --- |
| `has_structured_output_requests` / `pending_structured_output_tokens` | 采样时不施加语法约束 |
| `lora_request` | 同上，无 LoRA 权重 |
| `prompt_token_ids is None` | 不支持 `prompt_embeds` |
| `mm_features` 非空 | 不支持多模态输入 |

`logprobs` 自 A1.3 起已实现，不再拒绝；`n > 1` 由 vLLM 自己展开成多个请求，无需处理。

## 错误格式

所有拒绝都走 `_unsupported(param, current, supported, advice)`，一句话说清三件事：

```text
tensor_parallel_size=2 不受支持；PIM 支持 1；建议 去掉 --tensor-parallel-size；PIM 内部已按 DPU 做张量并行
```

## 为何放在这三个入口

```text
LLM(...) 构造
   └─ PIMWorker.__init__        ← 引擎级能力矩阵：配置错就起不来
         │
SchedulerOutput 每步
   └─ execute_model
         ├─ validate_scheduler_output   ← 整批的结构化输出标记
         ├─ _add_request → validate_new_request  ← 新请求的输入形态
         └─ 逐请求执行
```

- 引擎级错误在 `__init__` 抛出，早于 `load_model` 的 7B 权重加载与图编译，省掉几分钟等待。
- 请求级错误在第一步执行前抛出，不会先生成违规 token 再补救。
- 三个函数都不依赖 `self`，只吃配置对象/请求对象，因此单测用 `SimpleNamespace`
  就能覆盖全部分支，不需要起真引擎。

## 已实施的测试

| 测试 | 验证点 |
| --- | --- |
| `test_supported_config_passes` / `test_supported_request_passes` / `test_plain_scheduler_output_passes` | 支持项组合全绿 |
| `test_unsupported_engine_config_rejected`（9 组参数） | TP/PP/LoRA/投机/前缀缓存/量化/dtype/架构/prefill 预算 逐项抛 `ValueError` 且消息含参数名 |
| `test_tensor_parallel_message_mentions_dpu` | TP>1 的提示说明 PIM 内部已有 TP |
| `test_disabled_chunked_prefill_skips_budget_check` | 关掉分块 prefill 后不做预算检查 |
| `test_unsupported_request_rejected`（3 组参数） | LoRA / prompt_embeds / 多模态 |
| `test_structured_output_rejected`（2 组标记） | 结构化输出在第一步前失败 |

实测：`tests/test_vllm_capabilities.py` 19 passed；
`tests/test_vllm_executor.py` 2 passed（同次运行共 21 passed，22.83s）。

## 当前不足

- `validate_engine_config` 只覆盖「已知会算错」的项；`kv_transfer_config`、
  `enable_sleep_mode` 等尚未纳入，遇到再说。
- 分块 prefill 的预算检查是「一步放下全部并发请求」的粗判；真正的限制是
  `prefill_graph` 固定长度，A3 的启动预检会把这条约束暴露成 CLI 建议。
- 错误消息是中文单行，未做 i18n。
