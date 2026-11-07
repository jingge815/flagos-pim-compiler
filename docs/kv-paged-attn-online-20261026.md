# 分页注意力的单遍在线 softmax（2026-10-26）

## 目标

10-25 的融合内核把一次注意力拆成三遍扫 KV：先算满整行分数
`tokens × max_blocks × slot_tokens` 落 f16，再做五相 softmax，最后逐块算 PV。
长 prompt 下分数矩阵是内核里最大的临时缓冲，而且 K/V 各读了一遍。

本次把 softmax 改成 vLLM 式的逐块在线形式：

- 每算完一个逻辑块，立刻把本块结果合并进 running max / running sum / 输出累加器；
- 分数缓冲只留**一个块**：`tokens × slot_tokens`；
- K/V 各只扫一遍，不再有独立的第二遍 PV 扫描。

算子的签名、块表寻址、地址范围检查、融合路径的选择条件都不变，只有
`LowerPIMToEmitC.cpp` 里 `pim.paged_attn` 的展开体被替换。

## 算法

```
for t in tokens:  m[t] = -1e30;  l[t] = 0;  acc[t][:] = 0

for 逻辑块 b in [0, num_blocks):
  slot   = bt[b]（越界夹到 0）
  kbase  = k_off + slot * slot_elems
  vbase  = v_off + slot * slot_elems
  rows   = min((b+1)*slot_tokens, hist) - b*slot_tokens

  # 2a. 本块 QK
  for 行 r in [0, rows):
    for t in tokens:
      logits[t][r] = 掩码( roundF16(Σ_d q[t,d]*K[r,d]) * scale )   # f16

  # 2b. 逐 token 在线合并
  for t in tokens:
    m_new = max(m[t], max_r logits[t][r])
    alpha = expf(m[t] - m_new)
    for 行 r in [0, rows):
      p[t][r] = roundF16(expf(logits[t][r] - m_new))               # f16
    l[t] = l[t]*alpha + Σ_r p[t][r]                                # f32
    for 维 d in [0, head_dim):
      acc[t][d] = acc[t][d]*alpha + Σ_r p[t][r]*V[r][d]            # f32
    m[t] = m_new

for t, d:  out[t][d] = roundF16(acc[t][d] / l[t])
```

`O_i = Σ_j exp(s_ij - m_i) V_j / Σ_j exp(s_ij - m_i)`。running max 只是为了防止
`expf` 溢出，最终除以 running sum 后与全局 softmax 完全等价。

alpha 的两个边界：

- 本块没有刷新最大值时 `m_new == m[t]`，`expf(0) == 1`，重标定是精确的恒等；
- 第一块 `m[t] = -1e30`，`expf(-1e30 - m_new) == 0`，累加器从零起步。

整块都被掩码时（`blockStart > pos + t`），块内最大值保持 `-1e30`，`m_new`
不变、`alpha == 1`、本块贡献为 0，不影响 running 统计。

## 生成的 C

一次调用的临时缓冲：

| 缓冲 | 形状 | 类型 | 用途 |
| --- | --- | --- | --- |
| `logits` | `tokens × slot_tokens` | f16 | 本块分数；原地复用为指数 |
| `acc` | `tokens × head_dim` | f32 | 未归一化输出累加器 |
| `runningMax` | `tokens` | f32 | 逐 token 的历史最大值 |
| `runningSum` | `tokens` | f32 | 逐 token 的未归一化分母 |

`logits` 从 `tokens × max_blocks × slot_tokens` 缩到 `tokens × slot_tokens`：
`max_blocks` 那一维被块循环消掉了。

## 数值口径

| 相 | 设备内核 |
| --- | --- |
| QK 点积 | f32 顺序累加，`roundF16` 后乘 scale |
| 掩码 | 位置可见保留，否则 `-1e30`，落 f16 时成为 `-inf` |
| 块最大值 | f16 分数读回 f32，逐行比较 |
| 平移指数 | `pim_lut_exp`（`expf`），结果落 f16 |
| 分母 | f32，按 `l*alpha + 块和` 累加 |
| 输出累加 | f32，按 `acc*alpha + 块贡献` 累加 |
| 归一化 | f32 除法 `acc / l`，最后落 f16 |

这与相位链（`gml_bridge/phase_data.py::softmax`：f16 最大值、`expf` 落 f16、
f32 求和、f16 倒数、f16 乘）**不再逐元素相等**：

- 相位链先求整行和、取 f16 倒数再逐元素乘；在线形式在末尾做一次 f32 除法；
- 在线形式在最大值增长时用 `alpha` 重标定历史累加，相位链没有这一步；
- 相位链的 PV 每块先落 f16 再进 f32；在线形式全程 f32 累加。

实测（`head_dim=8`、`slot_tokens=4`、乱序块表，覆盖 prefill 与 decode）：

| 对比 | 最大绝对误差 |
| --- | --- |
| 融合内核 vs 主机相位链两遍 | `2.4e-4` |
| 融合内核 vs float64 定义参考 | `6.0e-4` |
| 主机两遍 vs float64 定义参考 | `6.0e-4` |

所以运行时两条路径的判据从 `array_equal` 改成
`np.allclose(rtol=1e-3, atol=1e-3)`；对 float64 定义参考仍用 `atol=0.05`。
算子级测试改为复现同一个在线算法（每一步都在同一位置取整），与内核
`assert_array_equal` 逐元素相等。

## 不变的部分

- 算子形态、C 入口、`k_off` / `v_off` 相对槽池起点、`pos` / `num_blocks` /
  `num_tokens` 的语义；
- 块表缓冲每台 DPU 一份、执行必须串行、表尾 `-1` 夹到槽 0；
- i32 寻址范围 `2**30` 个元素的编译期检查；
- 融合路径的选择条件（图掩码生效、f32 KV、工具链不在位都走主机兜底）。

主机兜底路径（`runtime/kernels.py::_host_paged_attention`）继续是相位链两遍，
它是 `pim.softmax` 数值契约的镜像，不再声称与融合内核逐元素相等。

## 改动文件

| 文件 | 改动 |
| --- | --- |
| `FlagTree/lib/.../LowerPIMToEmitC.cpp` | 替换 `paged_attn` 的五段展开为单遍在线循环（`+131 / -124`） |
| `runtime/kernels.py` | 只改 `_paged_attention` / `_host_paged_attention` 的文档字符串 |
| `tests/test_opcompiler_ops.py` | `test_paged_attn_matches_the_online_reference`：在线参考，`assert_array_equal` |
| `tests/test_kernels.py` | `test_sdpa_fused_paged_attn_tracks_the_host_path`：融合 vs 主机改 `allclose` |
| `tests/README.md` | `test_kernels` 一行说明改成单遍在线 |
| 两遍版设计记录 | 已删除，本文档即当前口径 |

## 测试

| 文件 | 用例 | 判据 |
| --- | --- | --- |
| `tests/test_opcompiler_ops.py` | `test_paged_attn_matches_the_online_reference` | 真编 `.so`，与逐步复现的在线参考 `assert_array_equal` |
| `tests/test_kernels.py` | `test_sdpa_fused_paged_attn_tracks_the_host_path` | 乱序块表 `[2, 0]`，融合路由命中，融合 vs 主机 `allclose 1e-3`，两者各与 float64 参考 `allclose 0.05` |
| `tests/test_kernels.py` | `test_paged_attention_rejects_history_longer_than_the_block_table` | 历史超过块表容量时抛 `RuntimeError` |
| `tests/test_three_path_generation.py` | `paged_attn` 请求 | 编出 `.so`，与 numpy 镜像相对误差 `< 0.05` |

## 不足

- 融合内核与主机兜底不再逐元素相等，两路切换时输出会有 f16 末位差。这是
  单遍形式的固有代价：重标定与末尾除法改变了取整位置。
- 每块仍把分数落一次 f16，是为了与相位链的分数存储保持同一取整；若全程
  f32 可再少一次取整，但会离相位链更远。
- PV 的块内归约仍是逐 token、逐维、逐行串行，没有按 token 或 head_dim 并行。
- 分块流式 softmax 已经在了，但每个头仍是一次核调用，头间不复用归约。
