# genesim_bridge（问题 4：TTIR / pim mlir 接入）

把 FlagGems + FlagTree 编出的 IR 测得的算子成本，回填进 GeneSim 的
ModelIR。对应 `spec.md:829` 问题 4。GeneSim 与 FlagTree 源码均零改动。

方案三.(4) 的两步都已接入，由 `ir_level` 切换：第 1 步 `"ttir"`（FlagTree
原生 TTIR），第 2 步 `"pimir"`（`convert-triton-to-pim` + `pim-explicit-dma`
产物，默认）。

## 对外接口

```python
from genesim_bridge import (
    prepare_triton_env, assert_pim_passes_available, export_costs_to_genesim)

prepare_triton_env(pim=True)         # 必须在 import triton / flag_gems 之前
assert_pim_passes_available()        # pimir 路专用，缺 PIM pass 直接抛
sidecar = export_costs_to_genesim(
    ir_path=Path("models/gpt2_builtin.ir"),          # GeneSim 图骨架（输入）
    out_ir_path=Path("models/gpt2_builtin_pimir.ir"),
    sidecar_path=Path("models/gpt2_builtin_pimir_extensions.json"),
    seq_len=128,                     # prefill 代表点的 Tq
    cross_validate=True,
    ir_level="pimir",                # 或 "ttir"（第 1 步，回归对照）
)
```

入口脚本在 GeneSim 侧：`genesim/scripts/refine_ir_with_flagtree.py --ir-level`。

| 模块 | 职责 |
| --- | --- |
| `paths.py` | 两份 flagTree 安装路径 + PIM pass 硬件参数（唯一允许绝对路径处） |
| `env.py` | 补 `CPATH` / `TRITON_PTXAS_PATH`；`pim=True` 时切到 PIM triton 安装 |
| `flagtree_driver.py` | 包 `LibEntry.run` 捕获 grid + TTIR + 实参；就地降 pim mlir |
| `ir_cost.py` | 解析 TTIR / pim mlir 得 flops、dtype、`mram_traffic_bytes` |
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

**6. pim mlir 靠 `sys.path` 前插切安装，不装 wheel、不改 FlagTree。**
pytorch env 里那份 triton 的 `libtriton.so` **没有** PIM 支持（0 个
`convert-triton-to-pim` 符号，也没有 `backends/pim_sidecar.py`）；带 PIM 的
`flagTree-pim` 安装是个裸 venv、没有 torch / flag_gems。两个 wheel 只差 5 个
文件。把 PIM 安装的 site-packages 前插 `sys.path` 后三者共存正常（实测 torch
2.9.1 + CUDA + flag_gems 全通）。`TRITON_CACHE_DIR` 必须隔离——两份 triton 的
kernel 二进制不可互换。

**7. pim mlir 从捕获到的 TTIR 文本就地降，不读 `pim_sidecar` dump 的 `.pimir`。**
三个理由：不依赖 `FLAGTREE_EMIT_PIM` / `TRITON_DUMP_DIR`；不依赖编译缓存
miss（热路径 launch 命中缓存时 `make_ttir()` 根本不跑，sidecar 也就不 dump）；
`pim_sidecar.emit_pim_ir` 吞掉所有异常只发 warning，pass 失败会静默变成
「没有 pim mlir」。走 `ir.parse_mlir_module` 重新 parse 而不是对主编译路径的
module 动手——pass 会原地改 module，会污染 GPU 编译。实测重 parse 的产物与
sidecar dump 的结构等价（`wram-bytes-used` 逐位相同）。

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

## 第 2 步（pim mlir）的实际收益边界

**`flops` 在两层 IR 上逐位相等**，这不是巧合而是 pass 语义：
`convert-triton-to-pim` 只给张量类型加 `#pim.tasklet_tiled` 布局，
`pim-explicit-dma` 只把 `tt.load`/`tt.store` 换成
`wram_alloc` + `dma_load/store` + `barrier` + `wram_load/store`；
`tt.dot`、`arith.*`、`scf.for` 原样保留。实测 gpt2 的 112 个算子 × 2 个代表点
全部相等，GeneSim 仿真出的总时延两条路也完全一致（2.024890 s）。

pim mlir 真正新增的是 `mram_traffic_bytes`——MRAM↔WRAM 的显式搬运字节数。
按方案三.(3) 它**只落 sidecar、不进 `data_bytes`**（塞进去会把跨 VPU 传输量
估算污染成虚高值）。实测放大倍数：

| op_type | prefill amp | decode amp |
| --- | --- | --- |
| `GEMM` | 4.69× | 1.56× |
| `GEMV_SCORE` | 2.50× | 8.48× |
| `SOFTMAX` | 1.00× | 1.98× |
| `GEMV_CONTEXT` | 2.25× | 4.12× |

sidecar 另记每 kernel 的 `wram_bytes_used` / `wram_buffers` / `dma_ops` /
`dma_ops_with_proven_layout`，以及生效的 `pim_options`。

## 已知局限

- **线性拟合抹平 padding 台阶。** 中间区段（Tq 在 1~BLOCK_M 之间）有误差。
  两点原始值在 sidecar，需要精确值时可改为分段。
- **PIM 侧 96 个算子的成本回填了但不影响时延。** GeneSim 的 PIM pipeline
  走 `.pim_trace` 指令流（`PIMVPU.execute` 对这三类 op_type 走
  `_execute_runtime`），按 `pim.compile.tile_size`/`chunk_size` 生成指令、
  不读 `flops`/`data_bytes`。回填值当前只影响 `arith_intensity` 参与的分区
  判据。要让 PIM 时延也反映编译产物，需把 pim mlir 的 tile 形状喂进
  `pim.compile.*`——超出方案问题 4 的范围，未做。
- **`num-dpus` / `wram-bytes` 当前是纯记录参数。** 实测扫
  `num_tasklets ∈ {16,64,256}` × `wram_bytes ∈ {64KB,256KB}`，
  `wram-bytes-used` 恒为 10368、tile 形状不变——tile 由 FlagGems 在 GPU 上
  autotune 定，PIM pass 不重切（FlagTree doc 15.9：`dpusPerDevice` 默认全 1）。
  故方案三.(6) 的局部→全局换算当前无对象，`shard_participants` 仍为 `None`。
- **flash attention 的 WRAM 超预算。** 实测 `flash_fwd_kernel` 用 66048 B、
  `flash_fwd_splitkv_kernel` 用 82432 B，都超过 65536 B 预算。FlagTree 此处
  只发 warning、不重切 tile（doc 15.3），IR 仍可能不可执行。已按 note 落进
  sidecar，不静默放过。
- **仅覆盖 fp16 单精度路径**，未测 bf16 / int8。
