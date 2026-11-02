# 接入 Qwen2 与 GQA 缓存修复（2026-11-02）

## 目的

G5 要把模型覆盖从「只有 Llama」扩到第二类真实架构。选型落在 Qwen2：
它与 Llama 用同一套算子（RMSNorm + 标准 RoPE + 全注意力 + SwiGLU），
唯一的结构差别是 q/k/v 投影带偏置；滑窗模型（Mistral 一类）会改注意力
可见性语义，单独留到后面。

接入过程中发现一个**已在 Llama 上存在的 GQA 缺陷**：KV 头数少于 Q 头数时
（Llama-3、Qwen2-7B 都是这种），PIM 的 logits 与 HF 相差约 0.55，生成结果
直接变成另一个 token；非 GQA 模型则一致到 5e-4。这次一并修掉。

## 一、GQA 缺陷与修复

### 现象

同一份权重、1 台 DPU、4 个 Q 头共用 2 个 KV 头：

| 模型 | max|Δ| vs HF |
|---|---|
| Llama（4 KV 头，非 GQA） | 5e-4 |
| Llama（2 KV 头，GQA） | 0.55 |

逐节点对拍（`torch.fx.Interpreter` 记录每个节点的输出）定位到第一处发散
就是 `scaled_dot_product_attention`。

### 原因

`runtime/kernels.py::sdpa_kv_info` 原先按 **KV 头** 生成条目：一个 KV 头只有
一条 `{"head": kv_head, ...}`。但内核里 `head` 同时被当作三样东西：

1. `out[0, head]` 的下标——这是 **Q 头**编号；
2. `k[0, head]` / `v[0, head]` 的行来源——图上 `repeat_kv` 之后的张量按
   **Q 头**组织；
3. `k_off` / `v_off` 的缓存位置——这才是 **KV 头**。

GQA 下三者不相等，于是：Q 头 2、3 的输出没被计算（保持 0），而缓存写入
却按 Q 头编号去写 KV 头区域，互相覆盖。

### 修复

`sdpa_kv_info` 改为按 **Q 头** 出条目，条目里带上它所属的 `kv_head`：

```python
for head in spec.kv_heads:
    for q_head in spec.q_heads_by_kv[head]:
        heads.append({"head": q_head, "kv_head": head,
                      "k_off": ..., "v_off": ...})
```

`sdpa_kernel` 的缓存写入按 `(kv_head, dpu)` 去重：组内多个 Q 头的 K/V 是
同一份复制，只写一次。计算循环照旧逐 Q 头调用 `_paged_attention`。

修复后 GQA 与非 GQA 都回到 4e-4 量级，argmax 与 HF 一致。

## 二、Qwen2 接入

### 偏置链路

Qwen2 的 `q_proj` / `k_proj` / `v_proj` 带偏置，这条链路此前完全没通：

| 位置 | 改动 |
|---|---|
| `graph/strategy.py` | `LLAMA_WEIGHT_RULES` 增加三条偏置规则，统一按列切 `col` |
| `graph/spec_prop.py` | `_optional_tensor_operand`：缺省或 `None` 返回 `None`；linear 规则接受 `Shard(0)` 偏置并给偏置发一条 DPU 请求 |
| `runtime/kernels.py` | `linear_kernel` 按位置读取实参，第三个存在就相加 |
| `runtime/kernels.py` | `compiled_linear_kernel` 只在恰好两个实参时走编译模板，带偏置退回 numpy |

偏置只能跟列切走：输出每一列独立，本 DPU 加本片偏置；行切分的输出是
部分和，加偏置会重复累加，所以规则里对行切直接报错。

### 注册表

`backend/architectures.py` 增加 `"qwen2": Architecture(_load_qwen2, _dense_strategy)`。
切分策略与 Llama 共用（头数、层数、FFN 宽度都从 `hf_config` 读，字段名相同）。
原 `_llama_strategy` 改名为 `_dense_strategy`，因为它已经服务两种架构。

### 滑窗校验

`validate_engine_config` 原来见到 `sliding_window` 非空就拒绝。真实 Qwen2
权重目录的 `config.json` 里写着 `"sliding_window": 32768` 和
`"use_sliding_window": false`——这只是配置残留，注意力仍是全注意力。
现在按 `use_sliding_window` 判断：真正打开滑窗仍然启动期报错，没打开就放行。
没有这个字段的配置（Llama）按「声明了就用」处理。

## 三、测试

| 文件 | 用例 | 判据 |
|---|---|---|
| `tests/test_kernels.py` | `test_sdpa_gqa_writes_each_kv_head_once` | 4 个 Q 头 / 2 个 KV 头：条目为 `[0,0,1,1]`，每个 Q 头的结果与所属 KV 头对拍 |
| `tests/test_architectures.py` | `test_gqa_prefill_matches_hf` | GQA 的 prefill logits 与 HF 差 < 1e-3 |
| `tests/test_architectures.py` | `test_qwen2_generate_matches_hf` | 随机偏置的小 Qwen2 存盘 → 走注册表加载/切分 → 贪心生成与 HF `generate` 逐 token 一致 |
| `tests/test_architectures.py` | `test_qwen2_config_passes` | `sliding_window=4096, use_sliding_window=False` 放行 |
| `tests/test_architectures.py` | `test_qwen2_sliding_window_still_rejected` | `use_sliding_window=True` 仍然报错 |
| `tests/test_architectures.py` | `test_unknown_architecture_lists_supported` | 未知架构报错并列出 `llama`、`qwen2` |
| `tests/test_vllm_capabilities.py` | `test_unsupported_engine_config_rejected` | 拒绝项改用 `gpt2` |

Qwen2 的用例必须显式随机化 q/k/v 偏置：transformers 初始化把线性层偏置
置零，不随机化就测不到偏置路径。

## 四、当前不足

- 只有 Qwen2 一类新架构；Mistral 等滑窗模型仍拒绝。
- Qwen2 的 `tie_word_embeddings`（小模型默认共享词嵌入）没有单独验证，
  测试模型恰好走共享路径但未断言。
- 偏置在 MLP 层不存在，所以 `gate_proj` / `up_proj` / `down_proj` 的规则没动。
- GQA 修复只覆盖 `sdpa_kernel` 这一条路径；per-op matmul 注意力
  （`gml_bridge` 导出用）不经过这里，未验证 GQA 下的表现。
