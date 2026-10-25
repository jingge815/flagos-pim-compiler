# 分页注意力融合算子 pim.paged_attn（2026-10-25）

## 目标

原来的 `runtime/kernels.py::sdpa_kernel` 在主机侧做两遍：读块表、逐逻辑块查物理槽、
逐块算 QKᵀ 与 PV，中间把分数矩阵放在 numpy 数组里。块表已经搬上设备（见
`docs/kv-global-block-table-20261025.md`），但读表、寻址、循环仍全在主机。

本次新增一个真正的融合算子 `pim.paged_attn`：把槽池指针和块表指针交给内核，
**逻辑块到物理槽的解析、QK / softmax / PV 三段都在设备上完成**，主机只给
`k_off` / `v_off` / `pos` / `num_blocks` / `num_tokens` 五个标量。

主机两遍的实现完整保留，作为兜底路径：f32 的 KV（只有测试会造）、图上传入了
对不上的掩码、工具链不在位，都走它。

## 三层结构

| 层 | 文件 | 内容 |
| --- | --- | --- |
| 方言算子 | `FlagTree/include/.../PIMOps.td` | `TTPIM_PagedAttnOp` 定义与 assemblyFormat |
| 方言校验 | `FlagTree/lib/.../IR/Ops.cpp` | `PagedAttnOp::verify()` |
| 设备 lowering | `FlagTree/lib/.../LowerPIMToEmitC.cpp` | EmitC 五段展开（见下） |
| 编译契约 | `contracts/op_contract.py` | `OpCompileRequest` 新增 `slot_tokens` / `slot_elems` / `scale` |
| 语义登记 | `contracts/op_semantics.py` | `paged_attn`：有内核、不是助记符、无 GML 名 |
| MLIR 生成 | `opcompiler_bridge/oplevel_kernel.py` | `paged_attn_kernel()` |
| 分派与校验 | `opcompiler_bridge/driver.py` | `_make_oplevel_mlir` 的 `paged_attn` 分支 |
| 运行时 | `runtime/kernels.py` | `_paged_attention` / `_host_paged_attention` / `_compiled_paged_attn` |
| 成本模型 | `genesim_bridge/ir_cost.py` | `pim.paged_attn` 的 flops 计费 |

## 算子签名

MLIR 形态（每个头、每个 token 步调一次）：

```
%y = pim.paged_attn %q, %c, %bt, %ko, %vo, %pos, %nb, %nt
    {slotTokens = 16 : i64, slotElems = 512 : i64, scale = 0.35355339059327373 : f64}
    : tensor<4x128xf16>, !pim.memdesc<4096x i16, #pim.mram>, tensor<8xi32>,
      i32, i32, i32, i32, i32 -> tensor<4x128xf16>
```

生成的 C 入口：

```c
void kernel(const half *q, const half *cache, const int32_t *bt,
            int32_t k_off, int32_t v_off, int32_t pos,
            int32_t num_blocks, int32_t num_tokens, half *out);
```

- `q` / `out`：一个头的 `(tokens, head_dim)`，每头单独调一次；内核缓存才能在
  头、层、步之间共用。
- `cache`：**槽池起点**，长度 `pool_slots * slot_elems`；物理槽总数由
  `pool_slots = cache_count / slot_elems` 推出来，不用另传。
- `bt`：块表缓冲，每项一个物理槽号。
- `k_off` / `v_off`：K / V 平面相对 `cache` 的元素偏移（层的差别全在这两个值）。
- `pos` / `num_tokens`：本步起点与本步 token 数，历史长度 = `pos + num_tokens`。

## 设备侧算术

`LowerPIMToEmitC.cpp` 把算子展开成五段循环：

```
1. 输出累加器 acc[tokens × head_dim] 清零（f32）
2. QK：for 逻辑块 b in [0, num_blocks)
         slot = bt[b]，越界（含表尾 -1）夹到 0
         base = k_off + slot * slot_elems
         for 历史行 j in [b*slot_tokens, min((b+1)*slot_tokens, hist))
           for token t: dot = Σ_d q[t,d] * cache[base + (j-b*slot_tokens)*head_dim + d]
                        scores[t, j] = mask(roundF16(dot) * scale)
3. softmax：逐行五相，长度为 hist
4. PV：for 逻辑块 b
         for token t, head 维 d: dot = Σ_j weights[t,j] * v[...]
         acc += roundF16(dot)          // 每块先落 f16 再进 f32
5. out[i] = roundF16(acc[i])
```

数值链与主机两遍逐相对齐，这是硬要求（运行时两条路径必须逐元素相等）：

| 相 | 主机两遍 | 设备内核 |
| --- | --- | --- |
| QK 矩阵乘 | `matmul` f32 累加后落 f16 | 同样，`roundF16(dot)` |
| × scale | f32 乘法，存进 f16 分数矩阵 | 同样，`storeFlat` 时落 f16 |
| 掩码 | `add_mask` 写 `-1e30` | `j <= pos + t` 可见，否则 `-1e30` |
| softmax 最大值 | f16 载入后转 f32 比较 | 同 |
| 指数 | `pim_lut_exp`（expf），落 f16 | 同 |
| 求和 | f32 | 同 |
| 倒数 | `pim_lut_reciprocal`，落 f16 | 同 |
| 归一化 | f16 乘 | 同 |
| PV | 每块 `matmul` 后相加 | 每块 `roundF16` 后进 f32 累加器 |

掩码为 `j <= pos + t`：prefill 时 `pos = 0`、`tq > 1`，就是因果掩码；decode 时
`tq = 1`、`pos = 历史长度`，当前 token 可见。运行时不会出现 `tq > 1 且 pos > 0`
的调用，这个式子对两种真实调用都正确。

## 地址与寻址范围

```
pool_slots = cache_count / slot_elems
slot       = bt[block]
base       = k_off + slot * slot_elems          # 元素，相对槽池起点
row_addr   = base + (j - block * slot_tokens) * head_dim
```

内核里所有地址都是 **i32 元素偏移**，最坏接近 `2 * cache_elems`。所以驱动在
编译期挡住过大的池：

```python
if cache_elems > 2**30:
    raise ValueError(f"槽池 {cache_elems} 个元素超出 paged_attn 的 i32 寻址范围")
```

静默回绕会把读写落到池子之外，比编译失败危险得多。

## 融合路径的选择

`_paged_attention` 每头调用一次，决策如下：

```
图上传入的掩码长度 == 历史长度 ? ──是──► 主机两遍（掩码语义不可丢）
           │否
           ▼
     KV 是 f16 ? ──否──► 主机两遍（内核相位链按 f16 落盘）
           │是
           ▼
  工具链能编出 .so ? ──否──► 主机两遍
           │是
           ▼
      设备内核（record_route("paged_attn", "hit")）
```

融合内核的可见性只按位置算，图上换成滑动窗口或自定义 bias 时会静默算错，所以
掩码一旦真的生效就走主机路径。

内核缓存键：

```python
(ctx, "paged_attn", q_shape, cache_elems, max_blocks, slot_tokens, slot_elems, scale)
```

**不包含**层、头、步、`pos`、`num_blocks`、`num_tokens`——这些都按值传。同一
DPU 上所有层、所有头、所有 decode 步共用一个 `.so`。

## 不变量

1. **块表缓冲每台 DPU 只有一份**，由 `step_request` 在每次 `execute_plan` 前覆盖。
   融合内核在设备上读这份缓冲，所以执行必须串行：上一步的内核没跑完就不能写表。
   现有 `execute_plan` 会等齐全部事件，保持。
2. 表尾的 `-1` 只用于暴露越界读。内核把越界槽号夹到 0，不会算出池外地址；
   主机路径遇到 `-1` 直接抛错。
3. 内核不拼 K/V 历史：分数矩阵是唯一的 `tokens × max_blocks × slot_tokens` f16
   临时块，PV 逐块累加。峰值内存与历史长度线性相关，但只是分数矩阵本身。

## 本次不做

- 多请求并发执行（块表缓冲是共享的）。
- 物理槽共享 / 前缀缓存 / COW。
- 把 softmax 分数矩阵也做成分块流式（现在一次性开满）。
- 算子校验器里的 `cache.count % slotElems == 0` 静态检查（驱动侧已查）。

## 测试

| 文件 | 用例 | 判据 |
| --- | --- | --- |
| `tests/test_three_path_generation.py` | `paged_attn` 加入 `_REQUESTS` | 真编出 `.so`，与 numpy 镜像逐元素相等 |
| `tests/test_opcompiler_ops.py` | `test_paged_attn_matches_the_phase_chain` | 顺序 f32 累加 → f16 的参考实现，`assert_array_equal` |
| `tests/test_kernels.py` | `test_sdpa_fused_paged_attn_matches_the_host_path` | 乱序块表 `[2, 0]`，融合与主机两遍 `array_equal`，numpy 因果参考 `allclose 0.05`；断言路由命中 |
| `tests/test_kernels.py` | `test_paged_attention_rejects_history_longer_than_the_block_table` | 历史超过块表容量时抛 `RuntimeError` |
| `tests/test_kernels.py` | 两条手工 SDPA payload | 补 `slots_base` / `cache_elems` 字段 |
| `tests/test_three_path_coverage.py` | 归属表 | `paged_attn` 无 genesim 名、无独立 GML 名 |
| `tests/test_genesim_bridge.py` | `test_paged_attn_costs_qk_softmax_and_pv` | flops = `4·tokens·hist·head_dim + 5·tokens·hist`，无「未识别」提示 |
| `tests/test_runtime_compiled_coverage.py` | 编译集合清单 | 同步加入 `paged_attn` |

另有一次性的离线对拍（未入库）：用 `runtime.kernels` 的 `matmul/softmax/add_mask`
与 `memory.kv_layout` 掩码做参考，覆盖 prefill 整块、prefill 补零行、块内截断、
decode 首 token、块边界、跨块、长历史、满长度八种真实调用形态，融合内核与主机
两遍逐元素相等。

## 改动量

- FlagTree：`+378 / -0`（3 个文件，纯新增）。
- 编译器源码：`+234 / -43`（`runtime/kernels.py` 把原两遍代码搬进
  `_host_paged_attention` 并新增融合入口，净增主要在这里）。
- 编译器测试：`+209 / -16`。
- 本文档：新增 192 行。

## 已知不足

- i32 寻址把槽池上限定在 `2**30` 个元素（f16 即 2 GiB）；更大的池需要改成
  i64 偏移或按槽分段传参。
- 一个头一次核调用，头间不复用归约。头数多时调用次数线性增长，但每次只传
  9 个参数，开销主要在 `ctypes` 本身。
- softmax 的分数矩阵按 `tokens × max_blocks × slot_tokens` 一次开满，没有按
  查询块切分；长 prompt 下这是内核里最大的临时缓冲。
