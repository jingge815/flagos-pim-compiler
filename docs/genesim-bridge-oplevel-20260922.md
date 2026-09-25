# B 路成本桥接：整算子级 IR 接进 GeneSim（2026-09-22）

## 一、一句话概括

GeneSim 的成本精化原先只有 A 路（跑 FlagGems → 抓 TTIR / pim mlir）。本轮把
**B 路**接上：不跑 FlagGems、不碰 GPU，按算子类型发一段整算子级 PIM IR，交给
`triton-opt` 展开成相位链，再由 `ir_cost` 计价。表 1.2.4 的 14 个 mnemonic 一个
不少，缺一个名字就抛 —— 成本 0 不会报错，只会让仿真少算一段。

```
GeneSim .ir ──> 按 op_type 选配方 ──┬─ A 路：FlagGems 算子 → TTIR → pim mlir
                                    └─ B 路：整算子级 IR → 融合/展开/校验 → 相位链
                                                              ↓
                                                      ir_cost 逐行计价
                                                              ↓
                                                回填 .ir 系数 + sidecar
```

## 二、改了哪些文件

| 文件 | 改动 |
| --- | --- |
| `genesim_bridge/op_classify.py` | 新增 14 个 mnemonic 的代表 IR 生成器与 `oplevel_ir()`；配方表从 4 条扩到 8 条，每条带 `mnemonic` |
| `genesim_bridge/flagtree_driver.py` | 新增 `lower_oplevel_to_pimir()`，**改走 `triton-opt` 可执行文件** |
| `genesim_bridge/cost_extractor.py` | 新增 `_measure_oplevel()`；`ir_level` 增加 `oplevel` 一档；B 路桥接四类原来只留模板的算子 |
| `genesim_bridge/ir_cost.py` | 认得全部 14 个 mnemonic（含展开后的相位算子） |
| `scripts/refine_ir_with_flagtree.py` | 新增 `--ir-level oplevel`（本仓脚本，不动 genesim 仓） |
| `tests/test_genesim_bridge.py` | 14 个 mnemonic 逐个计价、未知名字抛错、B 路 sidecar 三个测试 |
| `tests/test_oplevel_emitter_live.py` | 方案 7.3 的两条断言 |

## 三、14 个 mnemonic 怎么发出来

`op_classify.oplevel_ir(dims, mnemonic, point)` 是入口，`_OPLEVEL_IR` 是表。
每个 mnemonic 一小段 `tt.func`，包在 `module attributes {pim.target = "pim:v1"}` 里。
缺名字直接 `KeyError`，不返回空串。

分成两拨计价，因为硬件上确实是两种东西：

| 类别 | mnemonic | 计价 |
| --- | --- | --- |
| 做算术（8） | normalize / matmul / softmax / mask / rope / lut / eltwise / dynamic_quant | `flops > 0` |
| 只搬字节（6） | kv_cache / gather / transpose / reshape / split_heads / concat | `mram_traffic_bytes > 0` |

`kv_cache` 与 `gather` 在方案里归在"计算类"十个里，但硬件上只是寻址加搬运，
给它们记 flops 会凭空多出一笔运算，所以和四个视图类一起只记搬运。

两拨清单合起来必须**正好**等于 `MNEMONICS`：漏一个就是没测，多一个就是名字对不上，
测试里两条都断言了。

### 实测（hidden 512 / head_dim 64 / 8 头 / ffn 2048）

| mnemonic | decode（Tq=1,Tp=128）flops | prefill（Tq=128,Tp=0）flops |
| --- | --- | --- |
| normalize | 20480 | 2621440 |
| matmul | 33024 | 4194304 |
| softmax | 20672 | 2625536 |
| mask | 1032 | 131072 |
| rope | 1536 | 196608 |
| lut | 512 | 65536 |
| eltwise | 512 | 65536 |
| dynamic_quant | 1032 | 132096 |

六个搬运类两个点都不为零（gather / transpose / reshape 在 prefill 点各 131072 字节，
kv_cache 两个点都是 64 字节）。所有 14 × 2 个点的 `notes` 都是空，也就是没有
"循环次数没折叠、按 1 次计" 这类低估。

## 四、踩过的坑：进程内 libtriton 与 triton-opt 不是同一份构建

`lower_oplevel_to_pimir` 一开始用的是进程内的 `triton._C.libtriton`，结果
`stationarity`、`pim.gather`、transpose 的 `purpose`、`contraction` 全都报
"unknown attribute / custom op is unknown"。原因是 pytorch 环境里那份
`libtriton.so` 落后于 FlagTree 源码，而算子编译器
（`opcompiler_bridge.driver._run_oplevel_triton_opt`）走的是
`$FLAGTREE_PREFIX/build/flagtree-cmake/bin/triton-opt` 这个可执行文件。

现在 B 路也走可执行文件，pass 列表与算子编译器逐字相同。这不只是"能跑通"：
用进程内那份展开，genesim 量到的**不是算子编译器真正产出的 IR**，成本看着正常，
对的是另一份产物。

## 五、方案 7.3 的两条断言落在哪

放在 `tests/test_oplevel_emitter_live.py`，因为那两条都要真实模型：

1. `test_decode_block_mnemonic_counts_and_costs` —— decode 单层图上
   `pim.softmax` 32、`pim.rope` 2、独立 DQ 锚点 36，且展开后每个算子 flops > 0。

   DQ 数是这本轮最容易数错的一处：文本里 `pim.quantize` 其实是 **37** 条，
   因为 K 路 RoPE 的锚点同时要发 RoPE 与 DQ（`add_1__rope` 配 `add_1__dq`），
   最后那条是挂在 RoPE 上的尾段量化。方案 1.2.4 正文写作「37（36 DQ + 1 RoPE-DQ）」，
   7.3 表里的 36 是不含它的独立锚点数。测试两条都钉住了：总数 37、独立 36。

   另一个口径差：decode block 要裁掉 lm_head（覆盖表里它只随 `--layers 1` 发）。
   不裁会多一个 DQ 锚点，数出来是 38 / 37。fixture 里两个口径都导了，
   整图那份给形状与相位测试，裁掉那份给 7.3。

2. `test_apath_dot_cost_is_not_zeroed` —— A 路 linear 编译产物里的 `tt.dot`
   仍按 `2MNK` 记账（M=1 也不为例外），mram 搬运非零。B 路那轮改的是整算子级
   IR 的计价，两边共用 `analyze_ir`，所以这条要钉住。

## 六、脚本入口

`scripts/refine_ir_with_flagtree.py --ir-level oplevel`。genesim 仓的
同名脚本不动，那边照旧只有 `ttir` / `pimir` 两档。B 路不 `import triton`，
所以跳过 `prepare_triton_env` 与 `assert_pim_passes_available`。

在真实 `llama2_7b.ir`（3491 个算子，seq_len=128）上：

```
[oplevel] 桥接 3489 个算子，2 个保留模板成本
[oplevel] mnemonic: ['pim.eltwise', 'pim.lut', 'pim.matmul', 'pim.normalize', 'pim.softmax']
```

保留模板的那 2 个是 MODEL_INPUT / MODEL_OUTPUT。sidecar 里
`source_name` 的分布是 matmul 2272 / softmax 1024 / eltwise 96 / normalize 65 / lut 32，
`cross_validation` 字段不写（B 路不跑 FlagGems，没有融合注意力基准可比）。
GEMM 在 prefill 点的 flops = `4294967296` = `2 × 128 × 4096 × 4096`，
与 A 路同量级。3489 个算子 × 2 个形状点里没有一个成本为 0。

## 七、测试

- `tests/test_genesim_bridge.py`：14 个 mnemonic × 2 个形状点逐个计价；
  未知 mnemonic 抛 `KeyError`；真实 IR 跑一遍 B 路，sidecar 里每个桥接算子的
  `source_name` 都是 mnemonic 且成本非零。
- `tests/test_oplevel_emitter_live.py`：上面第五节两条。

## 八、不足（照实说）

1. **7.3 里"decode 注意力 mram 应小于 prefill"这条对不上。** 当前 `ir_cost`
   按驻留侧计费：GEMV_SCORE 的 mram 是 prefill（lkv=128）16384 字节、
   decode（lkv=129）16512 字节，decode 反而略大。这一条的物理解释（decode 只读
   新来的一个位置）没有在当前记账口径里体现出来。本轮没改，因为改它要动
   `stationarity` 的计费定义，超出"接 B 路"的范围。
2. **A 路的 `_KERNELS` 镜像只被间接覆盖。** `tests/test_partition.py`
   断言了每个设备侧算子都有切分规则和 numpy 内核，但没有测试逐个把 30 个镜像
   跑一遍。新加的逐元素镜像（pow/rsqrt/neg/silu/relu/sigmoid/exp/sqrt/reciprocal、
   matmul/bmm/_softmax/masked_fill/where/embedding/permute/transpose/view/reshape/
   slice/cat/split/to）只有真进设备分区时才会被执行到。
3. **B 路覆盖的是 8 个 GeneSim op_type**（GEMM / GEMV_SCORE / SOFTMAX /
   GEMV_CONTEXT / RMSNORM / SILU / VECTOR_ADD / VECTOR_MUL），不是 14 个
   mnemonic 各有对应 op_type。14 个 mnemonic 的代表 IR 与成本是逐条测过的，
   但真实 IR 里只会走到其中 5 个。
4. **softmax 的编排仍在主机。** `runtime/executor.py` 删掉了 `_host_softmax`，
   改调 `runtime.kernels.softmax`（编译内核优先、镜像兜底），但整个
   `scaled_dot_product_attention` 节点仍在 `partition.HOST_ONLY` 里 ——
   要拆头之后 softmax 与两次矩阵乘才各自成为设备节点，那一步不在本轮。
   三个端到端的 `llama2_7b` token 测试被 `-k "not llama2_7b"` 排除，
   所以这条改动没有在整模型上跑过。

---

## 补充（2026-09-23）：B 路成本当前的作用边界

评审 20260923 的 P0-4。上面那句「14 个 mnemonic 一个不少」说的是**本仓 sidecar
里的数字非零**，不是「仿真周期因此改变」。这两件事之间隔着三处阻断，
每一处单独都足以让 B 路成本到不了 GeneSim 的周期模型：

| # | 位置（genesim 仓） | 阻断 |
| ---: | --- | --- |
| 1 | `src/pim/pimir_trace.py::parse_pimir` | 三条硬抛：缺 `pim.tile-m/n/k` 模块属性抛、`tt.dot` 不恰好 1 个抛、`scf.for` 不是 1~2 层抛。B 路产物是相位链，没有 `tt.dot`、没有 `tile-*`，第一条就挂 |
| 2 | 全仓 | **没有任何地方读 `*_extensions.json`**。所有 `json.load` 指向 placement sidecar 或 `.ir` |
| 3 | `src/vpu/vpu.py::PIMVPU.execute` | PIM 周期来自**指令 trace**，不是 `op.flops`；后者只进 `useful_flops` 这类报告指标 |

第 3 条有个值得单独记下的推论：**A 路的 flops 同样不驱动仿真周期**。所以
「让 B 路成本影响仿真」不是一步之遥，而是整条消费机制对两条路都不存在。

因此 `--ir-level oplevel` 现在的用途是：**在不跑 FlagGems、不碰 GPU 的前提下，
拿到按算子类型分的成本数字，用于对照与回归**。它不改变 GeneSim 报出的周期数。

### 方案 §8.3 第 30 项的判据已改写

原文：「genesim `--ir-level oplevel` 写入 `docs/llama-2.md`，判据：文档与命令可跑，
备注：不改 genesim 仓」。三处自相矛盾：

- 写 `<genesim>/docs/llama-2.md` **就是**改 genesim 仓，与同一行的备注冲突；
- genesim 自己的 `scripts/refine_ir_with_flagtree.py` 是 `choices=["ttir", "pimir"]`，
  从那边跑 `oplevel` 必然报错；
- genesim 仓内 `oplevel` 全文零命中。

改写为：**本仓 `docs/` 记录用法与作用边界**（就是本节），genesim 仓不动。
`oplevel` 那一档只存在于本仓的 `scripts/refine_ir_with_flagtree.py`。

### 方案 §7.4 那条判据同样不属于 `run_full_pipeline.py`

原文要求它输出「B 路 sidecar 含 softmax / rope / dq / normalize / mask 且均非零」。
实测该脚本对这些名字以及 `oplevel` / `refine_ir_with_flagtree` **零引用**：
四步是 `model_parser` → `export_fixed_pu_mapping` → `export_pp_placement` → simulate，
断言查的是 `shards` / `local_in/out_features` / `semantic_role` / `kernel_tile_n` /
`pimir_path`，全是 A 路 GEMM placement 字段。它退出码 0 是真的，但那不构成
B 路的任何证据。

两条判据分开：

```bash
# 全链路闭环（图编译 → 算子编译 → 仿真），判据只到退出码 0
python scripts/run_full_pipeline.py --num-stages 4

# B 路成本非零，单独验
python scripts/refine_ir_with_flagtree.py \
    --ir <genesim>/models/llama2_7b.ir \
    --out-ir /tmp/oplevel.ir --sidecar /tmp/oplevel_ext.json \
    --seq-len 128 --ir-level oplevel
```

### 两份 `refine_ir_with_flagtree.py` 会漂移

本仓 108 行、genesim 158 行，`diff` 186 行；本仓那份不 import genesim 那份。
跨仓不合并是有意的（那边照旧用 `--ir-level pimir`），但两边各自演化这件事
要记着——genesim 侧加了新档位，本仓不会自动知道。
