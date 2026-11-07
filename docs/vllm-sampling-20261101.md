# vLLM 采样参数接入（2026-11-01）

## 改了什么

此前 `PIMWorker` 固定用 `np.argmax` 采样，vLLM 的 `SamplingParams` 全部被忽略。
本次把采样搬到主机侧，按 vLLM 0.14 的公式复刻，并回填 `ModelRunnerOutput.logprobs`。

| 文件 | 改动 |
| --- | --- |
| `backend/sampling.py` | 新增：`SamplingConfig`、`SampleResult`、`RequestSampler` |
| `backend/vllm_worker.py` | 新请求注册采样状态；`_step` 用请求自己的采样器；返回 logprobs |
| `tests/test_vllm_sampling.py` | 新增 16 例单测 |
| `tests/test_vllm_executor.py` | 新增同 seed 复现与 logprobs 端到端用例 |

## 采样流程

```text
logits [vocab]
  ├─ 原始 log_softmax（仅在请求 logprobs 时算，作为返回值）
  ├─ 施加 repetition / frequency / presence 惩罚
  ├─ temperature < 1e-5 ? argmax : logits / temperature
  │    ├─ top-k：排序后把小于第 k 大的 logits 置 -inf
  │    ├─ top-p：softmax 后累加，掩掉累计概率 <= 1-p 的尾部
  │    └─ 随机采样：q ~ Exp(1)，取 argmax(probs / q)（与 vLLM 同算法）
  └─ 采样 token；按需返回 top-k logprobs
```

与 vLLM 对齐的细节：

- **惩罚**：repetition 对 prompt + 已生成 token 生效，正 logit 除以惩罚、负 logit
  乘以惩罚；frequency / presence 只对已生成 token 生效。
- **温度阈值**：`temperature < 1e-5` 走贪心，与 vLLM 的 `_SAMPLING_EPS` 一致。
- **top-k / top-p**：沿用 vLLM 的「排序 + `< kth` 掩码」与「`cumsum <= 1-p`」实现，
  包含「至少保留一个 token」的边界处理。
- **logprobs**：默认 `raw_logprobs` 模式，取惩罚和温度之前的原始 logits；
  首项是采样 token，其后是 top-k，rank 为「比采样 token 严格更大的个数」。

## 关键结构体

| 名称 | 作用 |
| --- | --- |
| `SamplingConfig` | 与 vLLM 解耦的采样参数；`from_vllm()` 负责校验和转换 |
| `SampleResult` | 一个 token + 可选的 logprob token ids / 值 / 排名 |
| `RequestSampler` | 单请求采样器，持有 CPU `torch.Generator`；显式 seed 时跨步可复现 |

## Worker 侧的状态管理

- `_samplers` / `_prompt_ids` / `_output_ids` 按请求号保存。
- 原始 prompt 与已生成 token 分开记：被抢占重放 prefill 后，`RequestSlot.prompt_ids`
  会变成「prompt + 已生成」，但惩罚仍按 vLLM 的口径使用原始 prompt 与 output。
- 同一步所有请求的 logprobs 宽度取批内最大值，与 vLLM 的
  `SamplingMetadata.max_num_logprobs` 一致；未请求 logprobs 的请求行留空，
  调度器会忽略。
- 请求结束时随 `finished_req_ids` 一起清理。

## 已实施的测试

| 测试 | 验证点 |
| --- | --- |
| `test_greedy_matches_argmax` | 温度 0 与 argmax 一致 |
| `test_seed_reproducible_across_steps` | 同 seed 多步相同，不同 seed 不同 |
| `test_top_k_one_is_greedy` / `test_top_p_filters_tail` | 截断生效 |
| `test_repetition_penalty_positive_and_negative` | 正负 logit 的惩罚方向 |
| `test_frequency_and_presence_penalty` | 频次/存在性惩罚且只看出 output |
| `test_logprobs_use_raw_logits` / `test_logprobs_width_follows_batch_max` | logprobs 取原始 logits、宽度随批 |
| `test_unsupported_params_rejected` | min_p / logit_bias / allowed_token_ids / bad_words / prompt_logprobs / flat_logprobs / logprobs=-1 显式报错 |
| `test_seeded_sampling_and_logprobs`（vLLM 端到端） | 同 seed 两次生成逐 token 相同，logprobs 非空且长度匹配 |

实测：单测 16 passed；`tests/test_vllm_executor.py` 2 passed（20.22s）。

## 当前不足

- CPU 的 `torch.Generator` 与 vLLM GPU 采样器的随机数序列不逐位一致；只保证
  「同 seed 在 PIM 内可复现」，不保证与 vLLM 原生后端同 token。
- 未实现 `min_p`、`logit_bias`、`allowed_token_ids`、`bad_words`、
  `logits_processors`、结构化输出、`prompt_logprobs`、`logprobs=-1`；均显式报错。
- 默认 `temperature=1.0`（与 vLLM 一致）会随机采样；要复现旧的贪心行为需显式
  `temperature=0`。
- 采样在主机逐请求执行，没有像 vLLM 那样批量向量化；vocab 很大时是主要开销。
