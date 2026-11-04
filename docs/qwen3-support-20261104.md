# 接入 Qwen3（2026-11-04）

## 目的

把模型覆盖从 Llama / Qwen2 扩到 Qwen3。Qwen3 与 Qwen2 用同一套算子
（RMSNorm + 标准 RoPE + 全注意力 + SwiGLU），结构差别是：

| 差别 | Qwen2 | Qwen3 | 对链路的影响 |
| --- | --- | --- | --- |
| q/k 各一个 head 维 RMSNorm（qk-norm） | 无 | 有 | 走既有 RMSNorm 算子路径，权重按 replicate，无新算子 |
| q/k/v 偏置 | 有 | 无（`attention_bias=false`） | 更简单，偏置链路不用 |
| head_dim | `hidden / heads` 推导 | 配置显式声明，可不等于 `hidden / heads` | **唯一的真实改动点**（见下） |
| tie_word_embeddings 默认 | false | false | 与 Qwen2 相同，tied 小模型仍不支持 |

真实 Qwen3-8B 配置干净：无滑窗、无 rope_scaling、GQA（32 Q 头 / 8 KV 头）。

## 改动

| 文件 | 内容 |
| --- | --- |
| `runtime/compile.py` | `head_dim` 取 `config.head_dim`（Qwen3 显式声明且可与 `hidden/heads` 解耦），未声明的架构按老口径 `hidden // heads` 推导——与 `vllm_worker.py`、`serve_pim.py` 既有写法对齐 |
| `backend/architectures.py` | 注册 `"qwen3"`：`_load_qwen3`（fp16 加载）+ 复用 `_dense_strategy` |
| `backend/vllm_worker.py` | 白名单注释更新（`validate_engine_config` 本身按注册表驱动，无逻辑改动） |
| `genesim_bridge/paths.py` + `paths.json` | 新配置键 `qwen3_8b_model_dir`（环境变量 `QWEN3_8B_MODEL_DIR`） |
| `tests/test_architectures.py` | Qwen3 小模型用例：head_dim=32 ≠ hidden/heads=16（锁定解耦特性），q/k norm 权重随机化 |
| `tests/test_qwen3_8b_real.py` | 真实 8B 用例：注册表加载 → 8 DPU 编译 → 贪心生成对拍 HF `generate` |

模型来源：ModelScope（`Qwen/Qwen3-8B`、`Qwen/Qwen3-1.7B`）下载到
`/mnt/sda/butj/fjg/model/`；本机 huggingface.co 直连不稳定，hf-mirror 时通时断，
modelscope.cn 最稳（Qwen 官方分发）。

## head_dim 为什么必须改

`compile_llama2` 原先写死 `head_dim = hidden_size // num_attention_heads`。
Qwen3Config 默认 `head_dim=128` 且**不随 hidden/heads 变**：小探针里
hidden=64、heads=4 的模型 q_proj 是 `(512, 64)`（4 头 × 128）。按老口径算出
head_dim=16 后，`kv_specs_from_placement` 把 k_proj 分片切出了 16 个"KV 头"
（期望 2 个），KV 布局直接报错。真实 8B 恰好 `4096 / 32 = 128` 与显式值相同，
所以这个问题只有 head_dim 解耦的小模型（0.6B/1.7B/4B，head_dim=128、hidden 更小）
才会踩中——但推导口径本身就该以配置声明为准。

## 测试

| 用例 | 判据 |
| --- | --- |
| `test_architectures.py::test_qwen3_generate_matches_hf` | 随机小 Qwen3（GQA + qk-norm + head_dim=32≠16）贪心生成与 HF `generate` 逐 token 一致 |
| `test_architectures.py::test_unknown_architecture_lists_supported` | 白名单报错信息含 `qwen3` |
| `test_qwen3_8b_real.py::test_qwen3_8b_generate_matches_hf` | 真实 8B（36 层 / hidden 4096 / 32+8 头 / head_dim 128）8 DPU 编译，贪心 6 token 与 HF 一致 |
| 回归 | `test_kernels/partition/spec_prop/mem_planner/kv_layout/activation_windows/exec_binding` 共 107 条通过（head_dim 改动影响所有架构的编译路径） |

## 当前不足

- **Tied 词嵌入（1.7B / 0.6B / 4B 小模型）仍不支持**：tied 导出里 lm_head 直接
  消费 `embed_tokens.weight`，`spec_prop` 要求被 linear 消费的权重有切分规则，
  而 embed_tokens 同时被 embedding 查表消费，需要 vocab 并行语义（Qwen2 时代
  就有这个缺口，此前文档"测试模型恰好走共享路径"的说法不对——Qwen2Config
  默认 `tie_word_embeddings=false`，tied 路径从未被测过）。
- `gml_bridge/sharding.py` 的 `local_shapes` 仍按 `hidden/heads` 推 head_dim，
  当前无调用方（死代码），未动。
- Qwen3 的滑动窗口变体（部分小模型默认开滑窗）按既有规则在启动期拒绝。
