# 架构白名单与 GQA 支持（2026-11-02）

> 同日后续：Qwen2 已接入，GQA 也发现并修掉一个缓存覆盖缺陷，
> 以 `docs/qwen2-support-20261102.md` 为准。本文保留白名单抽表的记录，
> 其中「GQA 不需要加新代码」的结论已被推翻。

## 改了什么

以前 worker 直接写死 `LlamaForCausalLM` 与 `llama_strategies`，`validate_engine_config`
里也只比一个字符串 `model_type == "llama"`。现在把「装权重 + 按 DPU 切分」抽成
一张架构表，白名单之外、位置缩放与滑窗注意力都在启动期拒绝；同时补上 GQA 模型
（Q 头多于 KV 头）的端到端验证。

| 文件 | 改动 |
| --- | --- |
| `backend/architectures.py` | 新增 `Architecture`（加载回调 + 切分回调）与 `ARCHITECTURES` 白名单 |
| `backend/vllm_worker.py` | `load_model` 按 `hf_config.model_type` 取接入点；能力矩阵拒绝未知架构、`rope_scaling`、`sliding_window` |
| `tests/test_architectures.py` | 新增 5 例：三种拒绝 + GQA 的 KV 布局与两请求一致性 |

## 白名单与拒绝

| 检查项 | 支持 | 拒绝时的提示 |
| --- | --- | --- |
| `model_type` | `llama`、`qwen2`（含 GQA） | 列出白名单里的架构名，建议换权重目录 |
| `rope_scaling` | 未设置（标准 RoPE） | 说明只实现标准位置编码 |
| `sliding_window` | 未设置（全注意力） | 说明只实现全注意力 |

拒绝都走 `_unsupported`，消息里有参数名、当前值、期望值和建议，测试按参数名匹配。

## GQA 怎么落的

GQA 不影响图结构，只影响 KV 区的几何：一个 KV 头服务连续几个 Q 头，切分时必须
按 KV 头数而不是 Q 头数判断能否整除。

```text
4 Q 头 / 2 KV 头 / 4 台 DPU
        │
        ├─ tp_width=4 ──► 2 头不能被 4 整除，跳过
        └─ 2 段 × TP2 ──► 每台 DPU 1 个 KV 头，q_heads_by_kv = {0:[0,1], 1:[2,3]}
```

`memory/kv_layout` 生成的 `KVRegionSpec` 本来就有 `q_heads_by_kv` 与 `kv_heads`。
但内核没有按这张映射展开——见 `docs/qwen2-support-20261102.md`：当时按 KV 头
生成注意力条目，Q 头 2、3 根本没算，缓存还被重复覆盖。现已改为按 Q 头出条目。

## 已实施的测试

`tests/test_architectures.py`（5 例）：

| 测试 | 验证点 |
| --- | --- |
| `test_unknown_architecture_lists_supported` | `model_type="gpt2"` 报错，消息里能看到白名单 |
| `test_rope_scaling_rejected` | 带位置缩放时点名 `rope_scaling` |
| `test_sliding_window_rejected` | 带滑窗时点名 `sliding_window` |
| `test_gqa_kv_layout_keeps_two_heads` | 2 个 KV 头分别落到两台 DPU，KV 区没有按 Q 头数重复 |
| `test_gqa_two_requests_match_single_runs` | GQA 微型模型两请求并发，生成 token 与各自单独运行逐 token 一致 |

Llama 基线不变由既有用例保证：端到端 vLLM 用例、`tests/test_vllm_*` 与
`tests/test_parallel_requests.py` 全部原样通过，加载与切分只是换了调用点。

## 当前不足

- Llama 与 Qwen2 两类架构；Mistral 等滑窗模型仍拒绝。
- GQA 只在微型模型（4 Q 头 / 2 KV 头）上验证，真实 7B/70B 的 GQA 权重没有跑过。
- 拒绝项是静态配置检查，不覆盖模型文件本身与配置不一致的情况（例如配置写的是
  全注意力、权重里其实带滑窗）。
