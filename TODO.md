# vLLM × PIM 接入 TODO

> 范围：`flagos-pim-compiler` 的 vLLM V1 后端（`backend/vllm_worker.py`、`backend/vllm_executor.py`、`scripts/serve_pim.py`）及其依赖的运行时/KV/编译模块。
> 基线环境：vLLM **0.14.0**（`flagOS-installed/pytorch`），Python 3.10，Llama-2 架构，`NumpyBackend` 模拟。
> 本文只拆分任务与验收标准，不在本文记录实现细节；实现细节按仓库约定写进 `docs/<module>-<日期>.md`。

---

## 0. 状态与验收规则

### 0.1 状态图例

| 标记 | 含义 |
|---|---|
| ⬜ | 未开始 |
| 🟡 | 进行中 |
| 🔍 | 待评估（方案未定或依赖外部结论） |
| ✅ | 已完成（验收全部通过且证据已记录） |
| ⛔ | 阻塞（写明阻塞原因与解除条件） |

### 0.2 状态流转

1. 开始开发时把任务从 ⬜ 改为 🟡，并在任务下写「当前进展」一行。
2. 只有验收清单**全部勾选**、证据已记录，才能改为 ✅。
3. 已 ✅ 的任务被后续改动破坏时，退回 🟡 并附回归命令与失败现象。
4. 阻塞任务必须写：阻塞原因、解除条件、临时绕过方式。

### 0.3 每个任务的完成定义（DoD）

- [ ] 代码 + 单测在同一次改动内完成，不写「测试稍后补」。
- [ ] 验收命令在本文「验收」小节全部通过，输出粘贴到任务下的「验收证据」。
- [ ] 行为对拍：小模型 token 与基线逐字一致，或明确记录允许的数值误差。
- [ ] 新增/修改的对外接口写进 `docs/<module>-<日期>.md`；影响启动参数的同步更新 `README.md` 与 `scripts/serve_pim.py` 的 `--help`。
- [ ] 不新增防御性兜底；不支持的配置必须显式报错，消息包含「哪个参数、当前值、期望值、怎么办」。
- [ ] 报告净增删行数（如 `+120 / -35`）。

### 0.4 验收证据怎么记

每个任务末尾维护一个表格，完成后填写：

```text
| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 单测 | python -m pytest tests/test_vllm_sampling.py -x -q | 8 passed | 2026-11-xx |
| 端到端 token 对拍 | ... | 与基线逐字一致 | 2026-11-xx |
```

---

## 1. 现状基线与总进度

### 1.1 基线（改动前必须先记录）

- [x] 记录 `vllm.__version__`、`torch.__version__`、`PIM_*` 环境变量。
      vLLM 0.14.0 / torch 2.9.1+cu128（2026-11-01）。
- [x] 跑通 `python -m pytest tests/test_vllm_executor.py -x -q`，记录结果。
      1 passed in 23.55s（2026-11-01，A1 改动前）。
- [x] 跑 `python scripts/profile_execution.py vllm`，记录 `P=1` / `P=2` 的并行效率、最大并发、窗口冲突。
      P=1：prefill 1.00 / decode 1.00，最大并发 1，冲突 0；
      P=2：prefill 2.00 / decode 1.98，最大并发 2，冲突 0（2026-11-01）。
      采集器 `_VllmProfiler._step` 的包装签名未跟上 A1 的 `logprobs_width` 参数，
      已同步修正（`scripts/profile_execution.py` 与两个测试替身）。
- [x] 用固定小模型 + 固定 prompt 记录一组贪心输出 token（作为后续所有对拍的基线）。
      微型 Llama（4 层 / hidden 64 / 4 头 / vocab 128 / `max_seq=16`）、4 DPU、
      seed 0、prompt `[1,2,3,4]`、贪心、生成 3 个 token → `[46,46,46]`（2026-11-01）。
- [x] 记录一次编译的 `kv_pool_slots`、`num_parallel_slots`、`prefill_seq_len`、单次 prefill/decode 墙钟。
      同一模型 2 个请求槽：`kv_pool_slots=2`、`num_parallel_slots=2`、
      `prefill_seq_lens=(16,)`、`act_stride=23552`、`block_table_bytes=256`；
      编译 3.5 s，一次 prefill + 3 次 decode 共 2.75 s（2026-11-01）。

### 1.2 总进度

| 阶段 | 任务 | 状态 |
|---|---|---|
| A 护栏与正确性 | A1 采样参数贯通 + logprobs | ✅ |
| A | A2 启动期能力矩阵校验 | ✅ |
| A | A3 serve_pim 预检与 CLI | ✅ |
| B 性能与长度 | B1 prefill 图分桶 | ✅ |
| B | B2 prefill 走融合内核 | ✅ |
| C KV 统一 | C1 以 vLLM 块表为唯一真源 | ✅ |
| D 长序列与共享 | D1 分块 prefill | ✅ |
| D | D2 前缀缓存 | ✅ |
| D | D3 抢占与部分重算 | ✅ |
| E 运维与扩展 | E1 KV 池/窗口自动定容 | ✅ |
| E | E2 可观测性 | ✅ |
| E | E3 模型覆盖 | ✅ |
| F 长期 | F1 vLLM 版本兼容 | ✅ |
| F | F2 真硬件与多进程 | ⛔ |
| F | F3 测试与文档加固 | ✅ |

> 建议顺序：A → B1 → B2 → C1 → D1 → D2 → D3 → E → F。A1/A2 先做，因为它们是「错了但不报错」的问题；C1 放在 D1 前，避免分块 prefill 的 KV 逻辑写两遍。

---

## A. 护栏与正确性

### A1 采样参数贯通 + logprobs

**目标**：vLLM 的 `SamplingParams` 在 PIM 主机侧真实生效；不支持的采样参数显式报错，不再静默 argmax。

**当前问题**：`backend/vllm_worker.py:38-39, 227` 固定 `np.argmax`，temperature/top_p/top_k/penalty/seed/logprobs 全部无效。

**拆分**

- [x] A1.1 `RequestSlot` 保存 `sampling_params` 与每请求随机数发生器；新请求从 `NewRequestData.sampling_params` 取，cached 请求沿用。
- [x] A1.2 实现主机采样器：greedy、temperature、top_k、top_p、repetition/frequency/presence penalty、seed。
- [x] A1.3 回填 `ModelRunnerOutput.logprobs`（至少 selected token 的 logprob；请求 top-k 时回填 top-k）。
- [x] A1.4 未实现的参数（如 `min_p`、`guided_decoding`、`logit_bias`）在请求/启动时抛明确错误。
- [x] A1.5 新增 `tests/test_vllm_sampling.py`。

**验收**

- [x] `temperature=0` 的输出与当前 argmax 基线**逐 token 一致**。
- [x] `temperature=1.0` 且 `seed=0/1` 两次输出不同；同 seed 重跑逐 token 相同。
- [x] top_k/top_p 的单测：构造已知 logits，被截断 token 的采样概率为 0。
- [x] penalty 单测：按 vLLM 公式手算结果一致。
- [x] `SamplingParams(logprobs=1)` 时 `ModelRunnerOutput.logprobs is not None`，selected token logprob 与 `log_softmax(logits)[token]` 误差 < 1e-3。
- [x] 不支持的参数有单测覆盖，错误消息包含参数名。
- [x] 既有 `test_vllm_executor.py` 仍通过。

**依赖**：无。

**风险**：vLLM 采样器包含大量细节（penalty 顺序、top_p 实现），需以 vLLM 源码为准逐项对齐；不要自创语义。

**实现**：`backend/sampling.py` + `backend/vllm_worker.py`；文档 `docs/vllm-sampling-20261101.md`。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 采样单测 | `python -m pytest tests/test_vllm_sampling.py -x -q` | 16 passed (3.49s) | 2026-11-01 |
| vLLM 端到端 | `python -m pytest tests/test_vllm_executor.py -x -q` | 2 passed (20.22s) | 2026-11-01 |

---

### A2 启动期能力矩阵校验

**目标**：vLLM 配置或请求使用了 PIM 不支持的能力时，在启动/首步就失败，且错误可操作。

**当前问题**：只拒绝了 `enable_prefix_caching`（`backend/vllm_worker.py:92-94`）和分块 prefill（`:189-193`）；TP/PP、LoRA、投机解码、结构化输出、logprobs 等会静默产生错误结果。

**拆分**

- [x] A2.1 在 `PIMWorker.__init__` / `load_model` 校验引擎级配置：`tensor_parallel_size`、`pipeline_parallel_size`、`lora_config`、`speculative_config`、`enable_prefix_caching`、模型架构白名单、`block_size` 与槽几何。
- [x] A2.2 在 `execute_model` 校验请求级配置：结构化输出（`has_structured_output_requests` / `pending_structured_output_tokens`）、`sampling_params.logprobs`（A1.3 完成前）、`n > 1` 的说明性警告。
- [x] A2.3 统一错误消息格式：`参数=当前值，PIM 支持=...，建议=...`。
- [x] A2.4 新增 `tests/test_vllm_capabilities.py`，逐项覆盖拒绝分支。

**验收**

- [x] 参数矩阵每一项都有一条单测，断言抛 `ValueError` / `RuntimeError` 且消息含参数名。
- [x] 所有支持项组合启动不回归。
- [x] 结构化输出请求在第一步前失败，而不是生成违规 token 后只告警。
- [x] `--tensor-parallel-size 2` 启动失败，消息说明 PIM 内部已有 TP。

**依赖**：无（A1.3 完成后可放开 logprobs 拒绝）。

**实现**：`backend/vllm_worker.py`（`validate_engine_config`、`validate_new_request`、
`validate_scheduler_output` 三个纯函数 + 三处调用点）；文档 `docs/vllm-capabilities-20261101.md`。

**说明**：
- 槽几何的两项检查留在原处：`__init__` 的 `kv_pool_slots >= blocks_per_slot`（`PIM_KV_POOL_SLOTS` 装不装得下满长请求），以及 `initialize_from_config` 的 `num_blocks == kv_pool_slots`（vLLM 分到的块数）。
- `logprobs` 已由 A1.3 实现，不再拒绝；`n > 1` 由 vLLM 自己展开成多个请求，PIM 无需特殊处理，故不加警告。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 能力矩阵单测 | `python -m pytest tests/test_vllm_capabilities.py -x -q` | 19 passed（含 2 个 executor 用例共 21 passed, 22.83s） | 2026-11-01 |
| 支持项不回归 | `python -m pytest tests/test_vllm_executor.py -x -q` | 2 passed | 2026-11-01 |

---

### A3 serve_pim 预检与 CLI

**目标**：服务入口把 PIM 关键配置暴露成 CLI，并在启动前完成一致性预检。

**当前问题**：`scripts/serve_pim.py` 只设置 `distributed_executor_backend` 和关闭前缀缓存；PIM 配置全靠环境变量，配置冲突到运行期才暴露。

**拆分**

- [x] A3.1 暴露 `--pim-num-dpus`、`--pim-mram-bytes`、`--pim-kv-pool-slots`、`--pim-parallel-slots`、`--pim-prefill-seq-len`，写入对应环境变量。
- [x] A3.2 启动预检：`max_num_batched_tokens` 与 `max_num_seqs × 最长 prompt`、chunked prefill 开关、`block_size`、槽池容量、MRAM 预算；失败时退出码非 0 并给出建议命令。
- [x] A3.3 README「启动方式」同步更新（启动方式一节在 `docs/vllm-backend-20261023.md`）。

**验收**

- [x] `python scripts/serve_pim.py --help` 能看到全部 `--pim-*` 选项。
- [x] 故意给一个装不下一个满长请求的 `--pim-kv-pool-slots`，启动在 `run_server` 前失败，消息给出所需槽数。
- [x] 默认配置 + 长 prompt 会提示「要么加 `--no-enable-chunked-prefill`，要么提高 `--max-num-batched-tokens`」。
- [x] HTTP 冒烟仍通过（两个并发 `/v1/completions`）。

**依赖**：A2。

**实现**：`scripts/serve_pim.py`（`PIM_FLAGS`、`add_pim_args`、`export_env`、
`preflight`、`_kv_bytes_per_dpu`）；文档 `docs/vllm-serving-preflight-20261101.md`。

**说明**：
- vLLM 0.14 的 `--help` 只列配置组名，`--pim-*` 挂在 `PIMConfig` 组下，用
  `--help=PIMConfig` 查看。
- 预检用的是 `EngineArgs.from_cli_args(args).create_engine_config()` 解析后的
  配置，所以 `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` 的
  默认值已由 vLLM 填好，不需要自己复刻默认值逻辑。
- MRAM 检查只算 KV 区（不含权重），偏松；权重上限仍由编译期 `plan_dpu()` 判定。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 预检与参数翻译单测 | `python -m pytest tests/test_serve_pim_preflight.py -x -q` | 9 passed (3.95s) | 2026-11-01 |
| `--help=PIMConfig` | `python scripts/serve_pim.py --help=PIMConfig` | 五个 `--pim-*` 选项与说明齐全 | 2026-11-01 |
| 槽池过小启动失败 | 微型模型 + `--pim-kv-pool-slots 0` | `run_server` 前退出，退出码 1，提示「建议加 --pim-kv-pool-slots 3」 | 2026-11-01 |
| HTTP 冒烟 | 微型模型起服务，两个并发 `/v1/completions` | 均 `completion_tokens=3`、`finish_reason=length` | 2026-11-01 |

---

## B. 性能与长度

### B1 prefill 图分桶

**目标**：不再用一张 `max_model_len` 长度的 prefill 图跑所有 prompt；按长度分桶，短 prompt 付短图代价。

**当前问题**：`PIM_PREFILL_SEQ_LEN` 默认等于 `max_model_len`（`backend/vllm_worker.py:56-57`），短 prompt 也按最长图计算与占内存。

**拆分**

- [x] B1.1 `compile_llama2` 支持多张 prefill 图（如 128 / 512 / 2048 / max）及各自的计划/内存规划。
- [x] B1.2 worker 按 `prompt_len` 选最小的、能放下的桶；仍整段 prefill，不做分块。
- [x] B1.3 `CompiledModel` 暴露桶列表；`PIM_PREFILL_SEQ_LEN` 退化为显式覆盖或废弃，文档写清楚。
- [x] B1.4 内存规划：多桶共享权重/KV 区，激活区按最大桶规划。
- [x] B1.5 新增 `tests/test_prefill_buckets.py`。

**验收**

- [x] prompt 长度 1/127/128/129/511/512/513/2047/2048/2049 的输出 token 与「单张 max 图」基线逐字一致。
- [x] 短 prompt（≤128）的 prefill 墙钟相对基线下降 ≥50%（微基准，记录数值）。
- [x] 编译期内存不超预算；各桶的 `mram_offset`/缓存键不冲突。
- [x] 既有全量快测通过。

**依赖**：无。

**风险**：多图会放大编译时间与缓存目录；需要确认 `kernel_mram`/激活区规划在多图下仍复用同一套三区布局。

**实现**：`runtime/compile.py`（`compile_llama2(prefill_seq_lens=)`、`default_prefill_buckets`）、
`runtime/executor.py`（`prefill_bucket`、`_prefill_plan`、`_prefill_output_cmd`）、
`backend/vllm_worker.py`（桶表计算与环境变量）；文档 `docs/vllm-prefill-buckets-20261101.md`。

**说明**：
- 桶长列表 = 显式桶 + `prefill_seq_len`，升序去重，最长项必须是 `prefill_seq_len`；
  每桶单独 export 一张图、单独 `plan_dpu`，但**校验**各桶 `weight` / `kv_base` 与最长桶逐字节相同。
- 激活窗口仍按最长桶规划 `act_stride`，短桶落在同一窗口前一段，总内存与单图一致；
  `kernel_mram` 取各桶 + decode 的峰值。内核缓存键含 `q_shape`，各桶不冲突。
- 验收里的 1/127/…/2049 边界用等比缩小复现：测试模型 `max_seq=16`，桶 `(2,4,8,16)`，
  覆盖 1、桶边界、跨桶（5/9）、最长桶内（13）；更大尺度的同一组边界由微基准佐证。
- `PIM_PREFILL_SEQ_LEN` 仍是最长桶；新增 `PIM_PREFILL_SEQ_LENS` 显式给桶表，
  默认 `default_prefill_buckets(max_len)`（128/512/2048 中不超过 max_len 的项 + max_len）。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 分桶单测 | `python -m pytest tests/test_prefill_buckets.py -x -q` | 21 passed (53.04s) | 2026-11-01 |
| token 对拍 | 同上 `test_bucket_tokens_match_baseline` | 1/2/3/4/5/8/9/13 与单张 16 长度图逐 token 相同 | 2026-11-01 |
| 墙钟 | `/tmp/bench_buckets.py`（100 token prompt，取 3 次最小值） | 7675.0 ms → 3102.7 ms（−59.6%） | 2026-11-01 |
| 内存布局 | 同上 `test_buckets_share_weight_and_kv_layout` | `weight`/`kv_base`/`act_base` 与基线一致，`act_stride` 相同 | 2026-11-01 |
| 全量快测 | `python -m pytest tests/ -x -q --ignore=…（三个 7B 文件）` | 1524 passed, 2 skipped (844.40s)，含 C1 新增用例 | 2026-11-01 |

---

### B2 prefill 走融合内核

**目标**：标准因果掩码的 prefill 命中 `pim.paged_attn`，不再全部回退主机两遍路径。

**当前问题**：`runtime/kernels.py:1407-1417` 以「mask 长度 == 历史长度」判定图掩码有效；prefill 在 `pos=0` 时恰好满足，于是始终走 `_host_paged_attention`。

**拆分**

- [x] B2.1 在编译期/绑定期标记「该 sdpa 节点的 mask 就是标准因果掩码」，随 payload 传给内核选路。
- [x] B2.2 `_paged_attention` 在标记为真时允许融合路径；非标准掩码继续回退。
- [x] B2.3 增加路由计数断言（`record_route("paged_attn", "hit")`）。
- [x] B2.4 记录 prefill 墙钟与数值误差。

**验收**

- [x] 标准 causal prefill 的 `paged_attn` 路由计数 hit > 0，fallback 只在自定义掩码/非 f16 时出现。
- [x] 固定 prompt 的生成 token 与主机路径一致；若不一致，logits 最大绝对误差 ≤ 2.4e-4 且需要文档说明。
- [x] prefill 墙钟相对 B1 后基线下降，记录数值。
- [x] `test_kernels.py` 中自定义掩码回退用例仍通过。

**依赖**：B1（先有桶，再优化每张图的注意力路径）。

**风险**：融合内核是单遍在线 softmax，与主机两遍不是逐位一致；验收以 token 一致 + logits 容差为准。

**实现**：`runtime/executor.py`（`execute_plan(..., causal_mask_standard=)`，`step_request` 两处置位）、
`backend/hal_numpy.py`（`bound_flag`）、`runtime/kernels.py`（`sdpa_kernel` 读开关、
`_paged_attention(standard_mask=)`）；文档 `docs/vllm-fused-prefill-20261101.md`。

**说明**：
- 原方案是「编译期/绑定期打标」，实现改为**调用方声明**：图上的掩码只是 placeholder，
  按节点名判断不可靠，自定义掩码会静默走融合路径。只有 `step_request` 这条既定路径置位，
  手工传掩码的调用方默认不置位，行为与改动前一致。
- `uses_graph_mask = 掩码存在 && 长度匹配 && 未声明标准`，因此声明后 prefill 全走融合；
  非 f16 KV 仍回退。
- 数值：融合内核与 numpy 重放逐位相同（`test_opcompiler_ops` 口径），与主机两遍路径
  最大绝对误差 < 3e-4，生成 token 逐 token 相同。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 内核路由与数值 | `python -m pytest tests/test_kernels.py -x -q` | 21 passed (2.28s) | 2026-11-01 |
| 端到端 token 对拍 | `python -m pytest tests/test_prefill_buckets.py -x -q` | 21 passed，含融合 vs 强制主机逐 token 相同 | 2026-11-01 |
| 路由计数 | `/tmp/bench_buckets.py` | `paged_attn hit=384`、无 fallback；强制主机时 `fallback=384` | 2026-11-01 |
| 墙钟 | 同上 | 3102.7 ms → 2846.6 ms（−8.3%），相对强制主机 6585.0 ms 快 56.8% | 2026-11-01 |

---

## C. KV 统一

### C1 以 vLLM 块表为唯一真源

**目标**：删除 PIM 自己的 `KVSlotPool` 双记账，直接使用 vLLM 的 `block_ids` / `new_block_ids` 写块表。

**当前问题**：vLLM 已分配块号（`NewRequestData.block_ids`、`CachedRequestData.new_block_ids`），但 `backend/vllm_worker.py:176-179` 只取 `req_id`/`prompt_token_ids`；PIM 在 `runtime/executor.py:48-57,166` 另分配一套物理槽，两套靠 `num_gpu_blocks == kv_pool_slots` 断言保持一致。

**拆分**

- [x] C1.1 `RequestSlot` 改为持有「逻辑块 → vLLM 块号」列表，不再持有 `KVBlockTable` 的分配权。
- [x] C1.2 新请求：写入 `req.block_ids[0]`；cached 请求：`new_block_ids` 追加；`resumed_req_ids` 替换。
- [x] C1.3 `finished_req_ids` 只释放 host 记账，不再归还物理槽（槽由 vLLM 管）。
- [x] C1.4 删除或降级 `KVSlotPool`：保留仅用于离线 `run_decode_loop` 等不经过 vLLM 的路径。
- [x] C1.5 每步断言：被调度请求的块表内容 == vLLM 给出的块号；越界/未分配立即抛错。
- [x] C1.6 新增 `tests/test_vllm_block_table.py`。

**验收**

- [x] 单请求、两并发请求的输出 token 与 C1 前逐字一致。
- [x] 断言 `slot.block_ids == req.block_ids[0]`，且 vLLM 路径不再调用 `KVSlotPool.alloc`（可用计数器/日志验证）。
- [x] 小槽池强制抢占：被抢占请求重算后的 token 与不抢占基线一致（D3 补上，见 `tests/test_preemption.py`）。
- [x] `kv_pool_slots` 默认仍为 `每请求块数 × max_num_seqs + 1`，null 块行为不变。
- [x] 快测通过，`test_kv_slots.py` / `test_slot_isolation.py` 不回归。

**依赖**：无（但建议在 D1 前完成）。

**风险**：vLLM 块号的所有权语义必须完全接管；`block_table_slots` 仍由 PIM 编译期几何决定，块号范围必须与 `pool_slots` 一致。

**实现**：`runtime/executor.py`（`KVBlockTable(pool=None)` + `adopt`、`RequestTable(kv_pool_slots=None)`、
`add_request(block_ids=)`）、`backend/vllm_worker.py`（`_add_request` 传块号、`_sync_cached`）；
文档 `docs/vllm-block-table-20261101.md`。

**说明**：
- 两种块号来源二选一：给 `block_ids` 或给 `kv_pool_slots`，混用/都缺都直接抛 `ValueError`。
  vLLM 路径 `RequestTable(n)` 不带池，`KVSlotPool.alloc` 根本不可能被调用。
- 「每步断言」由两处现成校验承担：`write_block_table` 检查块号 ∈ `[0, pool_slots)` 且
  块数 ≤ 每请求块表项数；外部块数不够时 `KVBlockTable.ensure` 立即抛错。
- 抢占重算的端到端用例（小槽池强制抢占）留到 D3：当前抢占只在
  `num_computed_tokens == 0` 时整体重放，部分重算尚未支持。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 块表单测 | `python -m pytest tests/test_vllm_block_table.py -x -q` | 7 passed (10.91s) | 2026-11-01 |
| vLLM 端到端 | `python -m pytest tests/test_vllm_executor.py -x -q` | 2 passed（两并发请求输出正常） | 2026-11-01 |
| token 对拍 | 同上 `test_external_block_numbers_change_nothing` | 外部块号 `[3]` 与自管池 `[0]` 的生成 token 逐字相同 | 2026-11-01 |
| 旧路径不回归 | `pytest tests/test_kv_slots.py tests/test_slot_isolation.py tests/test_multi_request_decode.py` | 全部通过（合计 54 passed 批次内） | 2026-11-01 |
| 全量快测 | `python -m pytest tests/ -x -q --ignore=…（三个 7B 文件）` | 1524 passed, 2 skipped (844.40s) | 2026-11-01 |

---

## D. 长序列与共享

### D1 分块 prefill

**目标**：支持 vLLM 原生分块 prefill 和「prompt 长于单张图」的场景；每块从任意 `pos` 执行，只在最后一块采样。

**当前问题**：`backend/vllm_worker.py:189-193` 对 `scheduled != prompt_len` 直接抛错；vLLM 0.14 默认 `enable_chunked_prefill=True`、`max_num_batched_tokens=2048`，默认配置下长 prompt 必然失败。

**拆分**

- [x] D1.1 worker 根据 `num_scheduled_tokens` / `num_computed_tokens` 计算本次 chunk 的 `pos`、`num_tokens`，选能容纳 chunk 的 prefill 桶。
- [x] D1.2 `position_ids` 改为 `arange(pos, pos+chunk)`；`num_tokens` 只算真实 token，补齐位不写 KV。
- [x] D1.3 prefill 未完成时不采样、不推进 `is_prefill_done`；最后一块才采样。
- [x] D1.4 支持 `num_computed_tokens > 0` 的续算（与 C1 的块表追加配合）。
- [x] D1.5 移除 `PIM_PREFILL_SEQ_LEN` 对 prompt 长度的硬限制，更新文档。
- [x] D1.6 新增 `tests/test_chunked_prefill.py`。

**验收**

- [x] prompt 长度 3000、chunk 大小 64/512 的分块结果与整段 prefill 逐 token 一致（微型模型 + 3072 长度的图；7B 量级未跑，见验收证据）。
- [x] 分块 prefill 不再抛「不支持分块 prefill」：vLLM 预算 4 < prompt 6 时自动拆块并跑通（4096 prompt + 默认 2048 预算未用 7B 跑）。
- [x] 多请求交错（A/B 各自分块）时，A 的输出与单独运行一致（证明补齐位未污染 KV）。
- [x] prefill 中途被抢占后重算，输出与不抢占一致（D3 补上，见 `tests/test_preemption.py`）。
- [x] 既有单请求/多请求测试通过。

**依赖**：B1（桶）、C1（块表）。

**风险**：`sdpa_kernel` 的图掩码与位置掩码在 `pos>0` 时的选路必须验证；末块补齐行的 logits 不能参与采样。

**实现**：`runtime/executor.py`（新增 `prefill_chunk`，`step_request` 的 prefill 分支改为调用它，`prefill_bucket` 按本段长度选桶，删掉已无调用方的 `pad_to`）、
`backend/vllm_worker.py`（`_step` 拆成 `_prefill_step` / `_decode_step`，中间段返回空 token 列表，
`validate_engine_config` 的分块规则倒转为「预算不得超过最长 prefill 图」，去掉 `num_request_slots` 参数）、
`runtime/kernels.py`（`_host_paged_attention` 按位置重算的掩码改用绝对位置 `pos+t`）、
`scripts/serve_pim.py`（预检规则同步倒转）；文档 `docs/vllm-chunked-prefill-20261101.md`。

**说明**：
- 中间段返回空 token 列表是 vLLM 的约定（`_update_request_with_output` 的注释：还在 prefill
  的请求应返回空 token 列表）；中途采样会污染罚项与输出记账。
- 主机两遍路径原先用相对位置掩码（第 t 行看到 t 个位置），`pos>0` 时会把前半段历史遮掉，
  分块结果与整段不一致；改为 `prefill_mask(hist_len, hist_len)[pos+t]` 后 `pos=0` 逐位等价、
  `pos>0` 正确。融合内核本来就按绝对位置判可见性，无需改动。
- 分块预算与图长的约束方向与改动前相反：以前「预算要装下整段 prompt」，现在
  「预算不得超过图长」，prompt 长度只受 `max_model_len` 限制。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 分块对拍（快测） | `python -m pytest tests/test_chunked_prefill.py -x -q -m "not slow"` | 10 passed (76.81s)，5 组切法 + 主机两遍路径 + 交错请求 | 2026-11-01 |
| vLLM 端到端分块 | 同上 `test_vllm_engine_chunks_long_prompt` | 预算 4 + prompt 6 与预算 16 的生成 token 相同 | 2026-11-01 |
| vLLM executor | `python -m pytest tests/test_vllm_executor.py -x -q` | 2 passed；调度器把 4 token 拆成 3+1 也跑通 | 2026-11-01 |
| 预算规则 | `python -m pytest tests/test_vllm_capabilities.py tests/test_serve_pim_preflight.py -x -q` | 29 passed，按新规则（预算 > 图长才报错） | 2026-11-01 |
| 分块 prefill 相关回归 | `python -m pytest tests/test_chunked_prefill.py tests/test_vllm_block_table.py tests/test_prefill_buckets.py -q -m "not slow"` | 38 passed (133.29s) | 2026-11-01 |
| 3000 token 分块对拍 | `python -m pytest tests/test_chunked_prefill.py -q -m slow` | 2 passed (612.10s)：64 / 512 两种切法与整段 prefill 逐 token 一致（微型模型 + 3072 长度图） | 2026-11-01 |

---

### D2 前缀缓存

**目标**：开启 vLLM 前缀缓存时不报错，多个请求共享前缀块，输出正确。

**当前问题**：`backend/vllm_worker.py:92-94` 直接拒绝；C1 完成后 vLLM 的共享块号可以直接用于 PIM 块表。

**拆分**

- [x] D2.1 C1 完成后移除拒绝逻辑，按 vLLM 的 `block_ids` 读共享块。
- [x] D2.2 确认写 KV 只发生在私有尾块；共享整块只读。
- [x] D2.3 处理 `new_block_ids is None`、`resumed_req_ids` 替换。
- [x] D2.4 新增 `tests/test_prefix_cache.py`。

**验收**

- [x] 两个共享 1024 token 前缀的请求：vLLM 前缀缓存命中数 > 0，输出与关闭缓存时逐 token 一致（前缀长度按微型模型等比缩小到 20 token，见验收证据）。
- [x] 共享块内容在后续请求执行前后哈希不变（或只读断言）。
- [x] 三个请求共享前缀 + 两个并发窗口，输出正确、无块冲突。
- [x] 关闭前缀缓存时行为与 D2 前完全一致。

**依赖**：C1、D1。

**风险**：vLLM 只共享整块，最后一块私有；若 PIM 写 KV 的范围计算错误，会污染共享块。必须用内容哈希或只读页验证。

**实现**：`backend/vllm_worker.py`（`validate_engine_config` 去掉前缀缓存拒绝与 `cache_config` 参数、
`_add_request` 用 `num_computed_tokens` 作起始位置、`_sync_cached` 取 `new_block_ids[index][0]`）、
`scripts/serve_pim.py`（不再强制关闭前缀缓存）、`tests/test_vllm_block_table.py`（伪造的
`new_block_ids` 改成 vLLM 真实形状）；文档 `docs/vllm-prefix-cache-20261101.md`。

**说明**：
- 共享块只读靠「写入范围不越界」保证，不需要单独保护：vLLM 只缓存整块，
  命中长度是块大小的整数倍，请求从 `num_computed_tokens` 起写，写块号恒 ≥ 命中块数。
- `_sync_cached` 原先 `list(new_block_ids[index])` 会把「按 KV 分组的一层元组」拆成
  列表的列表，单组下 `adopt` 收到 `[[7, 8]]` 直接 `int()` 报错。之前没有用例追加过块
  （生成长度没跨块）所以没暴露，开前缀缓存后第二个请求必然跨块才显现。
- vLLM 命中前缀时仍会重算最后一个 token，所以 prompt 只占一个整块时命中数为 0；
  端到端用例的 prompt 取 20 token（整块 16 + 4）。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 前缀缓存单测 | `python -m pytest tests/test_prefix_cache.py -x -q` | 4 passed（快测部分），含共享块字节比对的只读断言 | 2026-11-02 |
| vLLM 端到端命中 | `python -m pytest tests/test_prefix_cache.py::test_vllm_engine_reuses_cached_prefix -x -q` | 1 passed (36.12s)：`num_cached_tokens > 0`，生成 token 与关闭缓存时逐 token 相同 | 2026-11-02 |
| 块表形状回归 | `python -m pytest tests/test_vllm_block_table.py tests/test_prefix_cache.py -q` | 10 passed（含改形状后的 `new_block_ids` 用例） | 2026-11-02 |
| 全量快测 | `python -m pytest tests/ -x -q --ignore=…（三个 7B 文件）` | 见下 | 2026-11-02 |

---

### D3 抢占与部分重算

**目标**：vLLM 抢占后，PIM 能按 `num_computed_tokens` / `resumed_req_ids` 正确续算或重放。

**当前问题**：`_sync_preempted` 只在 `num_computed_tokens == 0` 时整体重置（`backend/vllm_worker.py:232-239`）；部分重算路径未处理。

**拆分**

- [x] D3.1 根据 C1 的块表与 `num_computed_tokens` 决定「续算」还是「重放」。
- [x] D3.2 处理 `resumed_req_ids` 的块表替换。
- [x] D3.3 强制抢占测试：小槽池 + 多请求。

**验收**

- [x] 强制抢占场景下，所有请求输出与不抢占基线逐 token 一致。
- [x] `num_computed_tokens > 0` 的 cached 请求不再被静默忽略。
- [x] 抢占发生在 prefill 中途、decode 中途各一条用例。

**依赖**：C1、D1。

**实现**：`runtime/executor.py`（`RequestSlot.reset(prompt_ids, computed=0)`）、
`backend/vllm_worker.py`（`_sync_cached` 改为 `resumed or computed < slot.valid_len`
时重置，并用 `computed` 作起始位置）；文档 `docs/vllm-preemption-20261102.md`。

**说明**：
- 旧逻辑要求「已 prefill 完」才重置，prefill 中途被抢占的请求会被静默漏掉：
  块号换了但 `valid_len` 还是旧值，之后的写入位置全错。
- 恢复时 vLLM 重发的 `all_token_ids` 是 prompt + 已生成 token，重放 prefill
  会把已生成的部分一起算进去，末行采出的就是下一个 token；采样状态不用回退。
- 端到端用例的池大小是试出来的：块 16、`max_model_len` 32 时每条请求最终占
  2 块，池 4 块实测**没有**抢占，池 3 块才触发。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 抢占单测与数值对拍 | `python -m pytest tests/test_preemption.py -x -q` | 6 passed (68.84s)：prefill 中途 / decode 中途 / 共享前缀续算三种恢复各一组 | 2026-11-02 |
| 强制抢占端到端 | 同上 `test_vllm_engine_preempts_and_matches_baseline` | 池 3 块时 `vllm:num_preemptions > 0`，两请求输出与池 8 块（不抢占）逐 token 相同 | 2026-11-02 |
| 回归 | 见 D2 行的全量快测 | 1538 passed, 2 skipped (933.29s) | 2026-11-02 |

---

## E. 运维与扩展

### E1 KV 池/窗口自动定容

**目标**：只给模型、`max_model_len`、`max_num_seqs`，自动算出能放下的 `kv_pool_slots` 与并行窗口 `P`，并回报给 vLLM。

**当前问题**：`PIM_KV_POOL_SLOTS`、`PIM_PARALLEL_SLOTS` 全靠环境变量；配置冲突到编译期才报错，默认 `P=2` 会限制吞吐。

**拆分**

- [x] E1.1 在内存规划后求解「剩余 MRAM 能放多少槽」，反推 `kv_pool_slots`。
  `memory/mem_planner.fit_pool_slots` 扣掉权重区终点、块表窗口、激活窗口与内核
  tile 后按单槽字节数求容量，各 DPU 取最小。
- [x] E1.2 在槽数/激活预算内选择默认 `P`，并受 `max_num_seqs` 限制。
  默认 `min(2, num_request_slots)`，装不下一个满长请求时逐个减到 1，
  `block_table_slots` 跟随。
- [x] E1.3 显式配置超预算时给出「需要 / 可用」数值与建议。
  编译期 `内存超限` 消息带 `peak=` 与 `可用预算=`；预检点名 `PIM_KV_POOL_SLOTS`
  并给出建议槽数。
- [x] E1.4 `determine_available_memory` / `initialize_from_config` 与自动值保持一致。
  槽池字节由编译产物算出，`num_gpu_blocks == kv_pool_slots` 不符即启动失败。

**验收**

- [x] 不设任何 `PIM_*` 环境变量，启动成功，`num_gpu_blocks == kv_pool_slots`。
- [x] 记录默认 `P` 下的吞吐，相对固定 `P=2` 有提升或持平（数值记录）。
- [x] 故意超预算的配置启动失败，消息含所需值与可用值。
- [x] 满长请求仍能装下，高并发不被槽数意外卡住。

**依赖**：C1。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 不设 `PIM_*` 启动 | `pytest tests/test_auto_capacity.py::test_vllm_starts_without_pim_env -q` | 通过；`max_model_len=32`、块 16、`max_num_seqs=2` → 每请求 2 块，`pool=5=2×2+1`，`P=2`，`num_gpu_blocks` 相等 | 2026-11-02 |
| 默认 P 的吞吐 | `/tmp/bench_parallel_e1.py`（4 DPU / 4 层 / hidden 1024 / 4 请求 × 8 步） | P=1 27 557 ms / 60.6%；P=2 21 403 ms / 71.2%；P=4 21 360 ms / 73.8%，token 三组一致 —— 默认 2 已吃满，开 4 只多 0.2% | 2026-11-02 |
| 超预算报错 | `pytest tests/test_auto_capacity.py -q` | `test_explicit_pool_over_budget_names_need_and_available` 断言 `peak=`/`可用预算=`；`test_auto_capacity_fits_where_fixed_config_fails` 同一预算下显式满配报错、自动定容成功 | 2026-11-02 |
| 满长请求不被卡住 | `pytest tests/test_auto_capacity.py -q` | 自动槽数恒 `> blocks_per_request`（多出的一块给 null 块），5 例通过 | 2026-11-02 |

---

### E2 可观测性

**目标**：能回答「这一步 vLLM 派了几个请求、PIM 并行了几路、注意力走了哪条路径、槽池用了多少」。

**拆分**

- [x] E2.1 统计每步 wave 宽度、槽池占用、`paged_attn` hit/fallback、各 DPU 忙碌时间。
  `backend/metrics.StepStats` 每步累加；各 DPU 忙碌时间由 `--pim-profile` 挂上的
  `profile_backend` 采集。
- [x] E2.2 通过日志（每 N 步）或 HTTP/指标端点暴露。
  `PIM_LOG_STEPS=N`（`--pim-log-steps`）每 N 步打一行，默认关闭。
- [x] E2.3 复用 `scripts/profile_execution.py` 的请求级采集，提供 `--profile` 开关。
  `PIM_PROFILE=1`（`--pim-profile`）在 worker 里挂 `profile_vllm` 与 `profile_backend`，
  退出时打印两份报告；顺带修掉 `_VllmProfiler._step` 与分块 prefill 后新签名的冲突。

**验收**

- [x] 日志/端点能看到上述指标，字段含义写进文档。
- [x] 指标聚合有单测。
- [x] 开启指标后吞吐下降 < 2%。

**依赖**：A1（采样）、C1（槽池统计）。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 指标可见 | `pytest tests/test_step_stats.py -q` | 4 例通过；端到端用例断言日志含「槽池」与 `paged_attn`，`PIM_PROFILE=1` 时请求级与 DPU 级采集器都有数据 | 2026-11-02 |
| 聚合单测 | 同上 | `record` 的 wave 取整、共用块去重、日志字段三例 | 2026-11-02 |
| 开销 | `/tmp` 内联微基准（4 请求 × 4 块，20 000 次） | 单步 3.2 µs；相对 2.7 s 的合成 decode 步约 0.0001%，远低于 2% | 2026-11-02 |

---

### E3 模型覆盖

**目标**：不再硬编码 Llama；不支持的架构在启动期明确拒绝。

**当前问题**：`backend/vllm_worker.py:96-106` 固定 `LlamaForCausalLM` + `llama_strategies`；`get_kv_cache_spec` 只报 `FullAttentionSpec`。

**拆分**

- [x] E3.1 用 `model_config.hf_config` 驱动策略选择；至少抽出 Llama 专用的权重切分规则入口。
  新增 `backend/architectures.py`：`Architecture`（加载回调 + 切分回调）与
  `ARCHITECTURES` 白名单，`load_model` 按 `hf_config.model_type` 取用。
- [x] E3.2 支持 GQA + 标准 RoPE 的同类架构；sliding-window 等先显式拒绝。
  GQA 的 KV 几何与内核本来按 KV 头数走，补了端到端验证；`rope_scaling` 与
  `sliding_window` 在 `validate_engine_config` 里点名拒绝。
- [x] E3.3 维护支持架构白名单与测试。
  `tests/test_architectures.py` 5 例覆盖白名单、两类拒绝与 GQA 运行。

**验收**

- [x] 不支持的架构启动失败，消息列出支持列表。
- [x] Llama 基线输出逐字不变。
- [x] 若实现第二个架构（如 Qwen2 tiny），两请求冒烟通过；否则文档明确标注不支持并有拒绝用例。
  未实现第二类架构：文档标注只支持 Llama，并有拒绝用例。

**依赖**：无（可与 E1 并行）。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 不支持架构被拒 | `pytest tests/test_architectures.py -q` | `model_type="qwen2"` 报错且消息含 `llama`；`rope_scaling`、`sliding_window` 各点名参数 | 2026-11-02 |
| Llama 基线 | `pytest tests/test_vllm_executor.py tests/test_parallel_requests.py -q` | 全部通过，接入点重排不改数值路径 | 2026-11-02 |
| GQA | `pytest tests/test_architectures.py -q` | 4 Q 头 / 2 KV 头 / 4 DPU（2 段 × TP2）编译通过，两请求 token 与单独运行一致 | 2026-11-02 |

---

## F. 长期

### F1 vLLM 版本兼容

**目标**：锁定受支持的 vLLM 版本；评估升级到 0.31 的改动面。

**现状**：实际环境是 0.14.0；工作区 `.venv` 是 0.31.0，`WorkerBase` 接口已变（`load_model(*, load_dummy_weights=False)`、`compile_or_warm_up_model -> CompilationTimes`、`sample_tokens` 拆分等）。

**拆分**

- [x] F1.1 在依赖/文档中锁定 `vllm==0.14.x`。
      `backend/vllm_compat.py::SUPPORTED_VLLM` 是唯一来源；`README.md` 环境节新增
      「vLLM 版本」小节写明 `pip install "vllm==0.14.*"`。
- [x] F1.2 启动时校验 `vllm.__version__`，不匹配给出明确错误。
      `PIMExecutor._init_executor` 第一步调 `check_vllm_version()`，早于任何 worker 构造。
- [x] F1.3 输出一份 0.31 适配差距清单（接口、`ModelRunnerOutput`、KV layout、async scheduling）。
      `docs/vllm-031-gap-20261102.md`：12 条差异 + 5 步升级顺序。

**进展**：0.14.0 环境基线对照工作区 `.venv` 的 0.31.0 逐项读源码核实，差异只记录
已确认的条目，未确认的（如 `execute_model` 同步返回是否被允许）标为需实测。

**验收**

- [x] 版本不匹配时启动失败，消息说明支持范围。
- [x] CI/本地回归脚本按锁定版本运行。
- [x] 差距清单文档完成，每条差异对照 0.31 源码核实。

**依赖**：无。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 单测 | `pytest tests/test_vllm_compat.py tests/test_vllm_capabilities.py -q` | 20 passed（2 + 18），0.14 放行、0.31 报错且消息含两个版本 | 2026-11-02 |
| 启动期拦截 | 阅读 `backend/vllm_executor.py::_init_executor` | 校验在 `super()._init_executor()` 之前，worker 尚未构造 | 2026-11-02 |

---

### F2 真硬件与多进程 ⛔

**目标**：`hal_vendor` 落地后，把「同 DPU 命令串行」升级为 HAL 硬约束；评估 `PIMExecutor` 的多进程形态。

**阻塞原因**：`backend/hal_vendor.py` 仍是桩；当前并发结论建立在 `NumpyBackend` 线程语义上。

**解除条件**：真硬件可执行至少一层 Llama，且 HAL 提供确定的完成/依赖语义。

**拆分（解除后）**

- [ ] F2.1 `hal_vendor` 实现与 `hal_numpy` 同接口，跑通小模型单请求。
- [ ] F2.2 校验泳道/依赖语义在真硬件上成立。
- [ ] F2.3 评估 `MultiProcExecutor` / Ray 的接入方式。

**验收**

- [ ] 真硬件单请求 token 与 `NumpyBackend` 一致。
- [ ] 两请求并发不串数据。
- [ ] 文档记录硬件约束与进程模型。

---

### F3 测试与文档加固

**目标**：vLLM 适配层不再只有一条端到端用例。

**拆分**

- [x] F3.1 `execute_model` 翻译层单测：new/cached/finished/preempted、wave 分配、异常路径。
      `tests/test_vllm_translate.py` 7 例，桩掉 `_step` 只测翻译层，不加载模型。
- [x] F3.2 每个 A–E 任务新增的测试文件纳入 `tests/README.md` 依赖矩阵。
      与 `ls tests/test_*.py` 逐项比对后补上 `test_vllm_compat.py`（2）、
      `test_vllm_translate.py`（7），表头更新为 102 个文件 / 1589 条用例。
- [x] F3.3 更新 `docs/vllm-backend-20261023.md` 的「当前不足」，或新开 `docs/vllm-integration-<日期>.md` 汇总。
      「当前不足」按现状重写；新增 `docs/vllm-integration-20261102.md` 做能力→代码→文档→测试索引。
- [x] F3.4 `scripts/verify.sh` 增加 vLLM 冒烟入口（可选标记，避免默认拖慢快测）。
      新增 `--vllm`：跑版本锁定 + 翻译层 + 能力矩阵（约 6 秒），默认不执行。

**验收**

- [x] `pytest tests/ -x -q` 通过（按现有约定排除已知会挂起的 7B 用例）。
- [x] `tests/README.md` 与测试文件一一对应。
- [x] 文档中的限制清单与实际行为一致，没有已实现但仍写「不支持」的条目。

**依赖**：随各任务推进。

**验收证据**

| 验收项 | 命令 | 结果 | 日期 |
|---|---|---|---|
| 翻译层单测 | `pytest tests/test_vllm_translate.py tests/test_vllm_compat.py -q` | 9 passed，3.40s | 2026-11-02 |
| 冒烟入口 | `scripts/verify.sh --vllm` | 27 passed，verify 通过 | 2026-11-02 |
| 文件与表格对应 | `comm -23 <(ls tests/test_*.py…) <(README 行…)` | 无缺失 | 2026-11-02 |

---

## 2. 明确不做的事

- 不重写 vLLM 的调度器；请求生命周期、连续批处理、抢占策略继续由 vLLM 负责。
- 不把 vLLM 的 CUDA attention backend / CUDA Graph 接到 PIM；PIM 的执行面是静态蓝图 + `NumpyBackend`/`hal_vendor`。
- 不为「以后可能用到」预造采样器插件、模型注册表或分布式抽象；出现第二个真实实现再抽象。
- 不在不支持的能力上做静默降级；一律显式报错。

---

## 3. 每周检查清单

- [ ] 总进度表的状态与实际代码一致。
- [ ] 每个 🟡 任务有「当前进展」一行，说明卡在哪。
- [ ] 新完成的 ✅ 任务有验收证据表。
- [ ] 回归：`pytest tests/test_vllm_executor.py tests/test_parallel_requests.py tests/test_kv_slots.py -x -q`。
- [ ] 基线 token 对拍仍然逐字一致。
