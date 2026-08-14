# genesim_bridge（问题 4 第 1 步：TTIR 接入）

把 FlagGems + FlagTree 编出的 TTIR 测得的算子成本，回填进 GeneSim 的
ModelIR。对应 `spec.md:829` 问题 4。GeneSim 源码零改动。

## 对外接口

```python
from genesim_bridge import prepare_triton_env, export_costs_to_genesim

prepare_triton_env()                 # 必须在 import triton / flag_gems 之前
sidecar = export_costs_to_genesim(
    ir_path=Path("models/gpt2_builtin.ir"),          # GeneSim 图骨架（输入）
    out_ir_path=Path("models/gpt2_builtin_flagtree.ir"),
    sidecar_path=Path("models/gpt2_builtin_flagtree_extensions.json"),
    seq_len=128,                     # prefill 代表点的 Tq
    cross_validate=True,
)
```

入口脚本在 GeneSim 侧：`genesim/scripts/refine_ir_with_flagtree.py`。

| 模块 | 职责 |
| --- | --- |
| `env.py` | 补 `CPATH` / `TRITON_PTXAS_PATH` |
| `flagtree_driver.py` | 包 `LibEntry.run`，捕获 grid + TTIR + 实参 |
| `ttir_cost.py` | 解析 TTIR 得算子级 flops 与 dtype |
| `op_classify.py` | op_type ↔ FlagGems 代表实现 + shape 构造 |
| `cost_extractor.py` | 两点拟合 coeffs、回填、落 sidecar |

## 关键设计决策

**1. 回填 `flops_coeffs` / `data_bytes_coeffs`，不是标量。**
GeneSim 的成本已改为符号系数，scheduler 每请求代入 Tq/Tp 求值
（`gene_sim_scheduler.py:186`），标量 `flops` 只在 coeffs 为空时作 fallback。
方案原文的 `op.flops = cost.flops` 会被 coeffs 直接盖掉，等于没生效。

**2. 两点线性拟合，且强制系数非负。**
系数无法由单点编译得出，故在 prefill(Tq=seq_len,Tp=0) 与
decode(Tq=1,Tp=seq_len) 各编一次。但真实成本对 Tq 是带 padding 台阶的
阶跃函数：实测 `GEMV_SCORE` 的 decode 点 term 更小(129 vs 16384) 而实测
flops 反而更大(2.62e6 vs 2.10e6)——`bmm` 把 1 行输入 padding 到 32 行 tile，
padding 开销达 159 倍。无约束两点解会给出负斜率，在 Tq=512 的真实 prompt
上算出**负 flops**，让 roofline 退化成只收 launch 开销、并把
`arith_intensity` 变负从而污染 PIM/GPU 分区判据。
故负斜率时降级为「过原点 + 大 term 点」定斜率——大 term 点 padding 可忽略，
斜率等于真实渐进标度（`GEMV_SCORE` 因此还原出 128 = 2·head_dim，模板是 127）。
两点原始测量值完整落 sidecar，不丢信息。

**3. 走原始 JSON 改写，不用 `ModelIR.load()/save()`。**
`ModelIR.to_dict()` 不含 gpt2 IR 携带的 `max_seq` / `vocab_size`，往返一趟
会静默丢字段。

**4. 包 `LibEntry.run` 而非 `JITFunction.run`。**
FlagGems 命中自身 kernel_cache 后直接 `kernel[grid](...)` 发射，不再经过
`JITFunction.run`；在那一层挂钩子会漏掉所有热路径 launch（实测捕获 0 个）。

**5. GEMM 的 `data_bytes` 要补权重字节。**
GeneSim 的 GEMM `input_shapes` 只有激活（`[["Tq",512]]`），权重不在 shape
列表里——模板把它算进 `data_bytes_coeffs["constant"]`。只按 input/output
求和会漏掉权重读取，而权重主导 decode 访存量，漏掉会低估到 1/250。

## 算子覆盖（a+c 路线，与方案原文的差异）

方案二.算子边界假设原文设定「GEMV 走 FlagGems 分离实现、1:1 对齐」。
**实测不成立**：gpt2 推理的 994 个 IR dump 里，带 `tt.dot` 的 kernel 只有
`linear_kernel` / `addmm_kernel` / `flash_fwd_kernel`，没有独立的
score/softmax/context kernel——FlagGems 的 attention 走融合 flash 路线，
一个 kernel 内做完 score→softmax→context 且所有 head 一起做。而 GeneSim
图骨架把 attention 拆成 96 个算子（每层每 head 三个，全在 PIM 侧）。

| GeneSim op_type | 个数 | 处理 |
| --- | --- | --- |
| `GEMM` | 16 | FlagGems `linear`，边界天然 1:1，抽真实成本 |
| `GEMV_SCORE` | 32 | `bmm` 按单 head shape 编，作代表实现 |
| `SOFTMAX` | 32 | `softmax` 按单 head shape 编 |
| `GEMV_CONTEXT` | 32 | `bmm` 按单 head shape 编 |
| `GELU` | 4 | 不覆盖，沿用模板成本 |

代表实现编出的 kernel 不是推理实跑的那个，但算子边界与 GeneSim 对齐。
融合 `flash_fwd` 的成本另记 sidecar 的 `cross_validation` 作对照：
prefill 下代表实现之和 34,342,912 vs 融合 34,489,360（**1.00×**，理论
`8·2·(2·Tq·L·D)` = 33,554,432），量级吻合。decode 为 0.46×，因
`flash_fwd_splitkv_kernel` 的循环边界依赖运行时 split 参数、折不出次数，
sidecar 已按 note 标注该 kernel flops 被低估——不静默当 0。

## 已知局限

- **线性拟合抹平 padding 台阶。** 中间区段（Tq 在 1~BLOCK_M 之间）有误差。
  两点原始值在 sidecar，需要精确值时可改为分段。
- **PIM 侧 96 个算子的成本回填了但不影响时延。** GeneSim 的 PIM pipeline
  走 `.pim_trace` 指令流（`PIMVPU.execute` 对这三类 op_type 走
  `_execute_runtime`），按 shape 生成指令、不读 `flops`/`data_bytes`。回填值
  当前只影响 `arith_intensity` 参与的分区判据。要让 PIM 时延也反映编译产物，
  需接第 2 步的 pim mlir 并改 `pim_compiler` 的指令生成——不在本步范围。
- **仅覆盖 fp16 单精度路径**，未测 bf16 / int8。
- `mram_traffic_bytes` / `shard_participants` 在 sidecar 里恒为 `None`，
  待第 2 步接 pim mlir（依赖问题 5）与问题 2 的 shard_map。
