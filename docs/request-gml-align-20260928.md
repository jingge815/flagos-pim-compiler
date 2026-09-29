# 需求文档：GML 与 bin 产物对齐甲方参考格式

> 文档编号：request-gml-align-20260928
> 创建日期：2026-09-28
> 关联项目：flagos-pim-compiler（存算一体大模型推理编译器）

## 一、需求背景与目标

### 1.1 业务背景

我方 `scripts/export_gml.py` 已能产出 `relay2gml_graph.gml` + 3235 个运行时 `.bin` +
`IO_info.txt`，但产物要交给甲方（芯方舟）的 L2Analyzer / NPM 设备消费，格式必须与甲方
TVM parser 的产出对齐。本轮任务是把两边逐项比对，定位差异的代码根源，给出修正方案。

对标基准：

| 项 | 路径 |
| --- | --- |
| 甲方样本 | `/media/disk/fengjingge/src/xinfangzhou-resource/model_layers_0_decode_v2/parser_output/tvmgen_default_nprm_main_0/runtime_files` |
| 格式规范 | `xinfangzhou-resource/VBU-GML Structure-281025-031239.pdf`（20 页） |
| 硬件手册 | `xinfangzhou-resource/Ceva-NeuPro-M_High_Level_ArchSpec_V1.6.6.GA.pdf` |
| 我方产出 | `python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out` |

### 1.2 预期目标

**形式一致，数值不必一致。** 具体判据（用户确认口径）：

1. GML 里引用的每个 bin 都能在磁盘上找到
2. 每个 bin 的域形式一致——字节数、字宽、字节序、分段结构相同，每个域的含义对得上
3. 生成本身的逻辑与参考完全一致（不能靠写死常量绕过计算步骤）
4. 数值可以不同（权重不同、标定数据不同），但不能是形式非法的值（如 scale=0）
5. `DEBUG_*` 调试字段允许不同

### 1.3 关联范围

| 仓库 | 本轮角色 |
| --- | --- |
| `flagos-pim-compiler` | **主战场**，GML 与 bin 的全部生成逻辑都在这里 |
| `FlagTree` | 实测不在 GML 生成链路上（`gml_bridge/` 不 import 它；`export_gml.py` 只有一个可选的相位模板校验开关）。本轮预计不改 |
| `genesim` | 同上，仅 `genesim_bridge.paths` 提供参考目录路径。本轮预计不改 |

## 二、核心功能需求

### 2.1 实测差异总览

两边跑通后的机械对比结果（全部为本轮实测，非推测）：

| 维度 | 甲方 | 我方 | 结论 |
| --- | --- | --- | --- |
| 文件总数 | 3229 | 3237 | 接近 |
| 文件名模式类 | 71 类 | **同 71 类** | 无缺类、无多类 |
| 非 bin 文件 | `relay2gml_graph.gml`、`IO_info.txt` | 同 | 一致 |
| GML 行数 / 字节 | 16091 / 531072 | 15875 / 506248 | 接近 |
| node / edge | 200 / 331 | 204 / 335 | 接近 |
| `relay2gml_version` | `"19.2.0"` | `"26.2.1"` | 形式差异，**改为 `"19.2.0"`** |
| GML 引用 bin → 磁盘 | 3463 引用 / 3227 落盘，**379 悬空 + 143 孤儿** | 3235 / 3235，**0 悬空 0 孤儿** | **我方已优于甲方** |
| 字段名总数 | 620 | 567 | 差值几乎全在 `DEBUG_*` |
| 同名文件 | 625 个 | 其中 **377 逐字节相同**、248 不同 | 见 2.3 |

**注**：上表是**不带** `--decode-block-only` 的导出。带上该开关后 node/edge 与
`op_type` 分布即与甲方精确一致，详见 P0-5。

甲方的 379 个悬空引用全是 `DEBUG_*`（`DEBUG_input_buffer_float_*` 153、
`DEBUG_input_buffer_N_float_*` 112、`DEBUG_weight_buffer_float_*` 73、
`DEBUG_div_value_*` 32 等），非 DEBUG 的只有 1 个 `lut_debug_195.bin`。
我方一个 DEBUG 字段都不发，反而自洽——`contracts/gml_coverage.py` 里已显式标注
"本方案不产调试副本"。**按判据 5，这一族归为允许差异，不列入修正范围。**

### 2.2 差异的三层归因

逐项追下来，差异不是四个平铺的维度，而是三层，且主因只有一条：

```
第一层  缺生成步骤（主因）
  激活标定缺失 → 喂进去的激活是全零张量
      ↓
  DQ 四相全零 → 32 个 2B sf = 0.0、4 个 64B、1 个 172B 全零
  Llama2ActivationDQ 的 64B sf 全零
  33 个 input_buffer 全零
      ↓
  62 个 output_sf、33 个 input_buffer 的内容差异

  外加 Kantor scale（= phase2）也随之全零 —— 它不是独立写死项，见 2.4

第二层  写死常量（独立于第一层，即便标定接通也不对）
  KV_Cache_DMA / Split / v_proj 的 output_sf 写死 1.0   (export.py:507)
  IO_info 的 sf 无条件 1.0                              (export.py:284)
  Kantor Shift 3 个文件                                 (export.py:458-459 非 phase 分支一律写 0)

第三层  纯形式
  子图边界                 我方含 lm_head 直出 logits → 开 --decode-block-only 即对齐（P0-5）
  hidden 边多一维          [1,1,1,4096] → 应为 [1,1,4096]
  IO_info 的 sf 序列化     裸 1.0 → 应为 numpy repr
  节点编号分配顺序          我方 197…206 再倒序 → 应照甲方从 1 顺序递增
  relay2gml_version        "26.2.1" → "19.2.0"
  我方多出的 Gather 节点    保留，声明为扩展
```

### 2.3 功能清单

#### P0-1 接通激活标定（主因，影响面最大）

**现状**：`gml_bridge/runtime_files.py` 有十余处 `np.zeros(...)` 直接落盘
（138、164、208、257、304、324、390、402、458、460、462、464 行），激活缓冲全是零。

**实证**：DQ 节点 12 的四相对比——

```
ph0 甲方: 64B n=32 first8=[0.6348, 0.6221, 0.6250, ...]  nonzero=32
ph0 我方: 64B n=32 全零                                   nonzero=0
ph3 甲方: 4096B nonzero=4078
ph3 我方: 4096B nonzero=0
input_buffer_12 甲方: 4096 elems absmax=0.3242 nonzero=4096
input_buffer_12 我方: 4096 elems absmax=0.0    nonzero=0
```

**关键结论：公式是对的，缺的是输入。** 用甲方自己的 `input_buffer_12.bin` 当输入，
套我方 `gml_bridge/phase_data.py:122-145` 的四相公式重算：

| 相 | 公式 | 与甲方逐元素相同 |
| --- | --- | --- |
| phase0 | `2 · absmax`（每 128 个一组） | **True**（32/32 组） |
| phase1 | `phase0 / 256` | **True**（32/32 组） |

absmax=0 → p0=0 → p1=0，`phase_data.py` 的全零组保护（把 p2 从 inf 置 0）正常触发，
于是 sf 一路输出 0.0。`sf=0.0` 属形式非法（下游反量化除零），不在判据 4 的豁免范围。

**标定输入来源（已确认）**：甲方 `parser_output/` 上层的 7 个文件就是标定输入，
按 **fp32** 解释后与 `IO_info.txt` 声明的 size 逐一吻合：

| 文件 | 字节 | fp32 元素数 | IO_info size |
| --- | ---: | ---: | ---: |
| `hidden_states.bin` | 16384 | 4096 | 4096 ✓ |
| `key_cache.bin` | 16777216 | 4194304 | 4194304 ✓ |
| `value_cache.bin` | 16777216 | 4194304 | 4194304 ✓ |
| `attention_mask.bin` | 4096 | 1024 | 1024 ✓ |
| `cos_position_embedding.bin` | 512 | 128 | 128 ✓ |
| `sin_position_embedding.bin` | 512 | 128 | 128 ✓ |
| `cache_position.bin` | 768 | 96（int64） | 96 ✓ |

注意这 7 个文件都在 `parser_output/` 下，**不在 `runtime_files/` 里**（逐一查过，
`runtime_files/` 无同名文件）。它们是标定的**输入**，不是我方要产出的交付物。

**要求**：导出流程按真实 absmax 算四相，不再喂全零。

**标定数据的落地形式（用户指定：嵌成源码常数，不在程序里读 bin）**

按元素规模分两档，理由是全量嵌入的体积不可接受：

| 张量 | 元素数 | 若全量嵌成源码 |
| --- | ---: | ---: |
| hidden_states | 4096 | 0.05 MB |
| **key_cache** | 4194304 | **56 MB** |
| **value_cache** | 4194304 | **56 MB** |
| attention_mask | 1024 | 0.01 MB |
| cos / sin / cache_position | 128 / 128 / 96 | ~0 |
| **合计** | | **112 MB** |

（对比：`docs/spec.md` 全文 287KB，`gml_bridge/` 整个模块 4067 行。）

因此：

- **小张量全量嵌入**：hidden_states 4096、attention_mask 1024、cos 128、sin 128、
  cache_position 96 —— 合计 5472 个值，约 0.07MB。以 fp32 常数数组写入源码，
  **每个常数上方注释写明来源文件、提取日期、dtype**。
- **KV cache 只嵌 absmax 标量**：标定链路实际只用到 absmax，不需要 4M 个元素。
  嵌两个常数即可，同样带来源注释：

  ```
  # 来源：xinfangzhou-resource/model_layers_0_decode_v2/parser_output/key_cache.bin
  #       fp32，4194304 元素，2026-09-28 提取
  KEY_CACHE_ABSMAX = 35.222347
  # 来源：同目录 value_cache.bin，fp32，4194304 元素，2026-09-28 提取
  VALUE_CACHE_ABSMAX = 38.633411
  ```

两档都满足"不在程序里读 bin"，且仓库不会多出 112MB 源码。

补充事实：`hidden_states.bin` 分布为 mean=-0.19、std=6.88、range=[-26.2, 26.6]，
是**均匀随机数不是真实推理激活**。所以我方不需要真实语料，只要标定链路跑通即可对齐形式。

`simulated_dynamic_quantize` 的分组口径实测为 **32 个 group_size=1024 + 5 个
group_size=128**，与我方"hidden/MLP 取 128、attention scores 整条一组（=S=1024）"
的现有实现一致，**不需要改**。

#### P0-2 KV cache 的 sf 不再写死 1.0

**现状**：`gml_bridge/export.py:507` `write_output_scale(files, node.node_id, 1.0, ...)`
无条件写 1.0，导致 2 个 `KV_Cache_DMA` + 2 个 `Split` 节点的 `output_sf` 为 1.0，
甲方是 0.0458984 / 0.00261116。

**同一根因还带一个 Gemm 特例**（原 Q12，已核实并并入本项）：甲方 7 个 Gemm 里只有
`self_attn_v_proj_MatMul_qidx37_params_36` 的 `output_sf` = 1.0192394e-05，其余 6 个
（q/k/o_proj、mlp 的 down/gate/up）都是 1.0。原因是 v_proj 的输出要写进 int8 的
value cache，这一路带真实 requant scale；其余 Gemm 输出留在 fp16 域故为 1.0。
我方 8 个 Gemm 全是 1.0，缺这个特例。修正时 v_proj 与 KV_Cache_DMA / Split 同源取值。

**甲方这两个常量的源头已追到**：不是子图内算的，是 TVM 子图**外层**的静态量化 scale，
在 `parser_output/qdq_mod_pre_build.txt` 里写得很明确：

```
%4 = qnn.quantize(%key_cache,   0.0458939f, 0, out_dtype="int8", axis=0)
%7 = qnn.quantize(%value_cache, 0.00261151f, 0, out_dtype="int8", axis=0)
...
%13 = qnn.dequantize(%10, 0.0458939f,  0, out_dtype="float32", axis=0)
%14 = qnn.dequantize(%11, 0.00261151f, 0, out_dtype="float32", axis=0)
```

进出同一块 cache 共用同一个 sf，relay 侧闭合。

**取值口径（用户确认：由我方按 absmax 算）**，附一条必须知晓的实测结论：

甲方这两个常量**无法从标定数据反推**。实测——

| | absmax | sf | sf·128 | int8 饱和比例 |
| --- | ---: | ---: | ---: | ---: |
| key_cache | 35.222 | 0.0458939 | 5.874 | **40.50%** |
| value_cache | 38.633 | 0.00261151 | 0.334 | **96.22%** |

数据 absmax 是 sf 量程的 6 倍 / 116 倍；`absmax/127`、`absmax/128`、只取前 N 槽位、
P99.9~P99.999 分位全部试过，无一命中。甲方那两个 scale 是**真实模型标定的定值**，
与其随机测试 bin 无关。

因此按 absmax 算会得到 0.2773 / 0.3042 一类的值，**不复现甲方常量**，但自洽、非 1.0、
反量化不除零、不饱和，符合判据 4。此为已接受的取舍，若甲方要求数值一致见 Q6。

#### P0-3 IO_info.txt 对齐

`IO_info.txt` 是甲方 parser 与 GML 并列落的**图级 I/O 清单**，单行 Python dict
字面量，记录子图入口/出口缓冲的 `dtype`、`shape`、`size`、`sf`、`input_idx`/
`output_idx`、`node_name`，mask 输入额外带 `mask: True`。规范 PDF 未涉及它，
我方 `gml_bridge/export.py:229 write_io_info()` 已按参考口径实现，
`contracts/compile_slots.py` 的 S=1024 就是从它读出来的。

**sf 的含义已确认**：等于该 I/O 缓冲对应 GML 节点的 `output_sf`，只是 IO_info 存
fp32、GML bin 存 fp16——

```
output_sf_28.bin fp16 = 0.0458984375  ←→ IO_info inputs['4'].sf = 0.04589387
output_sf_33.bin fp16 = 0.0026111603  ←→ IO_info inputs['7'].sf = 0.00261151
```

取值规律：

| dtype | sf |
| --- | --- |
| float16（hidden、cos/sin、mask） | 恒 1.0 |
| int16（kv_position 索引） | 1.0 |
| **int8（key/value cache）** | 该 cache 的量化 scale |

四项待修：

| 项 | 甲方 | 我方 | 处理 |
| --- | --- | --- | --- |
| int8 缓冲的 sf | 真实 scale | 1.0（`export.py:284` 无条件 `"sf": 1.0`） | 从 P0-2 的 scale 取 |
| sf 序列化 | `array(1., dtype=float32)` / `np.float32(1.0)`（**同一文件里两种混用**） | 裸 `1.0` | 改为 numpy repr |
| hidden 入口 rank | `[1, 1, 4096]` | `[1, 1, 1, 4096]` | 见 P0-4 |
| hidden 出口 | `[1, 1, 4096]` | `[1, 1, 1, 32000]` | 见 P0-5 |

序列化形式说明：甲方是直接 `repr()` 一个含 numpy 标量的 dict，消费端不可能用
`ast.literal_eval`，必须 `eval` 带 numpy 命名空间。我方写裸 float 会让按甲方口径
实现的消费端读不出预期类型。

#### P0-4 `IO_info` 里 hidden 入口的 rank 改回 3 维

**规则已确认**：甲方不是统一 3 维或统一 4 维，而是**每条边沿用原始张量的 rank**。
relay 子图签名（`qdq_mod_pre_build.txt`）：

```
nprm_0_i0:  Tensor[(1, 1, 4096), float16]      ← hidden，3 维
nprm_0_i9:  Tensor[(1, 1, 1, 128), float16]    ← cos，4 维
nprm_0_i10: Tensor[(1, 1, 1, 128), float16]    ← sin，4 维
nprm_0_i3:  Tensor[(1, 32, 1024, 128), int8]   ← kcache，4 维
nprm_0_i4:  Tensor[(3, 1, 32, 1), int16]       ← kv_position，4 维
nprm_0_i15: Tensor[(1, 1, 1, 1024), float16]   ← mask，4 维
```

hidden state 本来就是 `[batch, seq, hidden]` 三维。我方把它补成 `[1,1,1,4096]`
是多塞了一维；**其余六项两边 rank 完全一致，不需要改**。

**范围限定（评审 r6 问题3 实测修正）**：这条只改 `IO_info.txt` 报的 rank，
**GML 的边一条都不动**。原措辞"hidden **边** rank"与 2.2 第三层"hidden 边多一维"
的判断有误——实测参考的 331 条边**全是四维**，hidden 那条边就是
`dims "1x1x1x4096"`，只有 `IO_info` 报 `[1, 1, 4096]`。照"改边"的字面去做会把
已对齐的 331 条边改坏，直接撞 A10。A7 的判据同此口径：只看 `IO_info` 的 rank。

#### P0-5 子图边界：把 lm_head 移出这张图

**差异**：甲方 `nprm` 子图是一个标准 transformer block——进来 4096 维 hidden，
出去还是 4096 维 hidden，接着喂下一层：

```
nprm_0_i0 : [1, 1, 4096]  ← 本层 hidden 输入
输出       : [1, 1, 4096]  ← 本层 hidden 输出，previous_name 'output'
```

我方把 `lm_head`（4096 → 32000 词表投影）包进了同一张图，直出 logits
`[1, 1, 1, 32000]`。

**为什么要改**：

1. `lm_head` 是整个 32 层跑完之后才做一次的事，不属于 decode block 的边界。
   多层堆叠时第 1 层的输出要喂给第 2 层，logits 喂不进去
2. 判据 3 要求生成逻辑与参考一致——边界不同等于两张图在描述不同的计算
3. `lm_head` 权重 4096×32000，留在每层图里会在多层导出时重复 32 次

**要求**：图切到 hidden state 为止，出口 `[1, 1, 4096]`。

**重要发现：这个能力已经存在，只是默认关闭。** `scripts/export_gml.py:620` 有
`--decode-block-only` 开关，链路是 `export_gml.py` → `export.py:149` →
`from_fx.py:907 convert(decode_block_only=)` → `from_fx.py:2094 _trim_decode_block()`，
裁掉末尾 RMSNorm + lm_head + 它们的 DQ，并在 `convert` 里就不发 `Gather`
（`from_fx.py:933`，注释写明"必须在这里裁，留到收尾会让全图编号平移"）。

实测带上这个开关重跑（`/tmp/gml_dbo`），一批差异当即消失：

| 项 | 甲方 | 不带开关 | **带 `--decode-block-only`** |
| --- | --- | --- | --- |
| node / edge | 200 / 331 | 204 / 335 | **200 / 331 精确一致** |
| `op_type` 分布 | — | 多 Gather/Gemm/RMSNorm/DQ 各 1 | **逐类完全一致（diff 为空）** |
| `Gather` 节点 | 无 | 1 个 | **0 个** |
| 文件数 | 3229 | 3237 | 3186 |
| 模式类 | 71 | 71 | **71，无缺无多** |
| 悬空 / 孤儿 | 379 / 143 | 0 / 0 | **0 / 0** |

**所以 P0-5 的改动不是重写图切分，而是把这个开关设为默认行为**（或在导出参考
产物时固定传入）。代价从"改 `from_fx.py` 图切分"降为"改默认值 + 更新受影响测试"。

**连带影响**：原先判断"P0-5 与 P1-2 必须同批做、是本轮最大改动"**不再成立**。
带开关后 id 顺序仍是 `196 198 199…205 195 194…`（逆拓扑），P1-2 仍需独立做。

**另一处连带**：`Gather` 节点在这条路径上本来就不发，所以 7.1 里"保留 Gather 并
声明为扩展"的决策**只对不带开关的导出成立**。参考产物对齐路径上没有 Gather，
与甲方一致——Q5 因此降级为非阻塞项。

#### P1-1 尺寸量级不符的 bin

248 个同名内容不同的文件里，多数是 P0-1 的下游（`output_sf` 62 个、
`input_buffer` 33 个）。但有一批是**尺寸量级差**，属独立问题：

| 文件 | 甲方 | 我方 | 推测原因 |
| --- | ---: | ---: | --- |
| `input_buffer_0_28.bin` | 4194304 | 2048 | KV cache 整块 vs 单槽位 |
| `input_buffer_0_33.bin` | 4194304 | 2048 | 同上 |
| `input_buffer_0_9.bin` | 8192 | 22016 | 4096×fp16 vs 11008×fp16，**拿错张量**（hidden vs MLP 中间态） |
| `Bias_buffer_phase_0_22.bin` | 128 | 4 | 32 通道 vs 1 个标量 |
| `input_buffer_0_19.bin` | 2048 | 256 | 待查 |
| `input_buffer_101.bin` | 8192 | 2048 | 待查（注意两边 id 101 不是同一算子，需按算子配对后重查） |

**注意**：由于两边节点编号规则不同（见 P1-2），按文件名后缀配对可能在比不同算子。
~~本项的准确清单需在 P1-2 之后、或改用"按算子类型+角色配对"的方式重新统计。~~

**已完成（评审 r6 问题1）**：P1-2 完成后按"算子类型+角色"归族（去掉节点号归并族名）
重统计，194 个共有族里不符族 **33 → 1**。上表那六行是按文件名配对时的旧观察，
其中 `input_buffer_0_9` / `input_buffer_101` 等"拿错张量/待查"经重统计后不成立
（两边 id 不是同一算子）。真实根因两条、都已修：RoPE 子块的标量族按 head_dim 算宽
（应为单元素）、FPSU 三族按组数而非节点类型分宽。第 3 条根因在评审 r8 问题1 找到
并修掉（缓冲与定标"穿不穿过布局算子"两处口径不一致），现在 **0 族不符**、族集合两向
完全相同。

#### P1-2 节点编号分配顺序

**实测排除了"偏移"假设**——同一个 node_id 两边挂的算子类型都不同：

| node_id | 甲方 | 我方 |
| --- | --- | --- |
| 101 | MatMul `mha_batch_matmul1_head13` | DynamicScaling `dynamic_quantization_params_101` |
| 102 | MatMul `mha_batch_matmul2_head14` | Softmax `mha_softmax_head15` |
| 103 | DynamicScaling | Mask `mha_mask_head15` |

甲方 id 从 1 顺序排到 200；我方先发 197…206（图级 I/O），再从 196 倒序回落，
范围 3~206。这不影响单个 bin 的形式，但让**按文件名配对的对比全部失真**，
也让人工核对困难。

**要求（用户已定）**：照甲方口径，节点编号**从 1 开始顺序递增**。

这会让全部 `*_<node_id>.bin` 文件名随之变化，是一次大范围改动。**必须与 P0-5
（子图边界）一起做**：lm_head 留在图里会多占节点、把编号顶偏，两件事分两轮做要返工。

#### P1-3 DEBUG 字段：维持不产，文档显式声明

按判据 5，`DEBUG_*`（约 50 类）允许不同。我方一个不发，且甲方自身存在 379 个悬空
引用——我方不产反而使 GML 自洽。`contracts/gml_coverage.py:317-333` 已有显式标注。
**本项不改代码**，只在交付说明里声明。

同族的 4 个非 DEBUG 甲方独有字段一并处理：

| 字段 | 甲方含义（依 PDF p2） | 处理 |
| --- | --- | --- |
| `from_tvm`（7 处） | "Flag for TVM (and not cdnn)" | 我方非 TVM 路径，不发；文档声明 |
| `original_name`（10 处） | "the name as it appears in ONNX" | 我方无 ONNX 源，不发；文档声明 |
| `lut_debug`（1 处） | 调试 | 不发 |
| `cos_mul_output_hash` / `sin_mul_output_hash`（各 2 处） | 调试校验和 | 不发 |

#### P1-4 relay2gml_version 改为 "19.2.0"

我方 `"26.2.1"`，甲方 `"19.2.0"`。**要求（用户已定）：改成 `"19.2.0"`**，
与甲方样本一致。这个字段大概率被甲方解析器用于版本分派，填我方自己的版本号有被
拒绝解析的风险。

#### P1-5 Kantor Shift：3 个文件的取值

**本节已按全量实测修正两次判断**，最终结论（依据 49 个 Shift 文件逐字节实测）：

早期两处误判，一并记录以免后人重蹈：

| 曾经的判断 | 实测结论 |
| --- | --- |
| "Kantor 三族系数写死 0" | 只有 Shift 是独立问题。`scale` 逐元素等于 phase2（属 P0-1 下游）、`bias` 恒 0（我方已正确） |
| "Shift 两种宽度并存：32B fp16 `[-40704.0,…]` / 1B int8 `[-8]`，语义不明" | **dtype 读错**。全 49 个文件都是 **int8**；`f8f8f8f8…` 按 fp16 解释才得出 -40704。32B/86B 是**逐组**（32 = 4096/128、86 = 11008/128） |

**域规律（已完全确定，无需问甲方）**：int8；标量 1B 或逐组 N 字节；取值只有
`-8`（左移 8 位 = ×256）或 `0`。我方 `gml_bridge/phase_data.py:60` 早已定义
`DQ_PHASE3_SHIFT = -8`。

**逐族对账**：我方 **46/49 已对**，只差 3 个：

| 文件族 | 个数 | 甲方 | 我方 | 状态 |
| --- | ---: | --- | --- | --- |
| `kantor_A_Shift_buffer_file_phase_3_*` | 37 | -8 | -8 | 已对 |
| `Kantor_A/B_Shift_Llama2Activation_Cos/Sin_*` | 8 | 0 | 0 | 已对 |
| `kantor_B_Shift_*` | 1 | 0 | 0 | 已对 |
| `Kantor_A_Shift_Llama2Activation_add_*` | 1 | **-8** | 0 | **待修** |
| `kantor_A_Shift_*` | 2 | **-8 / 0** | 0 / 0 | **待修 1 个** |

**根因**：`gml_bridge/export.py:458-459` 对所有非 `_phase_` 的 Kantor Shift 一律写
`np.zeros`；走 `_phase_` 的 37 个由 `runtime_files.py:321` 正确写 -8。甲方那个该发
-8 的是 `kantor_A_Shift_36.bin`（node 36 = v_proj），判据与 P0-2 的 v_proj 识别同源。

### 2.4 已确认一致、不需修改的部分

为避免后续返工，记录本轮实测确认**已经对齐**的项：

| 项 | 实测结论 |
| --- | --- |
| 文件名模式类 | 71 类完全一致，无缺无多 |
| GML 引用完整性 | 我方 0 悬空 0 孤儿，优于甲方 |
| 四相公式 | 用甲方输入重算，phase0/phase1 逐元素相同（32/32 组） |
| DQ 分组口径 | 32×1024 + 5×128，与我方一致 |
| Kantor bias | 恒 0，我方已正确 |
| Kantor scale | 等于 phase2，标定接通后自然对齐，非独立项（见 P1-5） |
| Kantor Shift 46/49 | `phase_3` 族 37 个、Cos/Sin 8 个、`kantor_B` 1 个全部已对；仅 3 个待修 |
| 结构校验器规则 | 拿甲方样本跑 `gml_structure_check.py`，5/5 全绿（原 Q9 已核实，规则对 Llama decode 适用） |
| bin 域形式 | 逐类抽样比字节：`output_zp`(4B 零)、`input_sf`(2B fp16 `003c`=1.0；int8 边的定标**不自命名**，改为引用上游 DQ 的 `output_buffer_phase_1_<DQ>.bin`，与参考同——见下面 A11 那行)、`input_zp`、`Scaling_buffer_file`(2B)、`Bias_buffer_file`(4B)、`weight_zp`、`Bias_buffer_phase`、`kantor_A_Shift_buffer_file_phase`(4B `00000020`)、`LUT_phase`(`003c` 后接零)、`activation_lut_file`(`f8` 重复) —— 字宽、字节序、填充形态全部一致 |
| 非标定类 sf（**只就字宽与数值**） | MatMul×64、Softmax×32、Mask×32、EltwiseAdd×2、EltwiseMul×1 的 `output_sf` 两边都是 2B/1.0。**这一句不覆盖「这个域指向哪个文件」**：参考是「跟着边上的缓冲名走（消费者编号）」，如 Mask 节点 `output_sf "input_sf_18.bin"`、Split `"weight_sf_20.bin"`；我方是「按本节点自命名」`output_sf_173.bin`。连带文件数不等（`output_sf_*` 参考 146 我方 182、`input_sf_*` 参考 110 我方 72）。两边都自洽（A8 悬空 0 孤儿 0 成立），且改动前就如此，**按评审 r5 问题5 维持现状、记为已知差异**（见实施文档 §六 待确认 3） |
| `IO_info` 入口槽号（**只就 outputs 已对齐**） | 出口槽号已按角色排到与参考一致（0=hidden / 1=key / 2=value）。**inputs 侧维持节点号升序**：参考的入口槽号是 relay 子图签名顺序（`nprm_0_i3` 排槽 3、`nprm_0_i15` 排槽 5，既不按 id 也不按角色），推不出规则，照搬等于把一张硬编码顺序表写进代码。实测 7 项里 4 项槽号不同（我方 mask3/kv_position4/value5/key6，参考 key3/kv_position4/mask5/value6），两边域名相同、只是顺序不同。**按评审 r5 问题1 维持现状、记为已知差异**（评审 r6 问题2 补入册） |
| ~~A11 剩余 2 族：`input_buffer` / `input_sf` 尺寸~~ **已修，A11 共有族 194 全部相符** | 原记载把这两族说成"只是文件名口径不同、数值语义两边一致、两边都自洽"，**与实测不符**（评审 r8 问题1）：错的是**配对关系**。参考那 32 个 `mha_batch_matmul1` 的 `input_buffer` 是上游 DQ 的整块 `output_buffer_22.bin`（4096B = 4096 个 int8）配 32 组 scale，隐含 group_size = 128，与上游 DQ 的实际分组一致；我方把缓冲按边宽写成 128B 却照抄整条 32 组 scale，等于宣称 group_size = **4**。全图逐边算"缓冲元素数 / scale 个数"：参考同口径全为 128，我方只有这 32 个节点是 4，是唯一异类。根因是 `from_fx` 的缓冲/定标"穿不穿过布局算子"与 `export` 的取值口径不一致，已按参考对齐（缓冲与定标都引用那个 DQ）。修后 A11 **194 个共有族 0 族不符**、族集合两向完全相同，这两族差异消失 |
| 逐字节相同的文件 | 377 个，含 `output_zp`×137、`input_zp`×66、`input_sf`×60 |
| 六项 I/O 的 rank | cos/sin/mask/kv_position/key_cache/value_cache 两边完全一致 |
| op_type 集合 | 除我方多一个 `Gather` 外完全一致 |
| 图拓扑：Q 路径与 K 缓存读路径的 `Transpose` 位置（**已知差异，等价改写**） | 参考是 `Gemm(q_proj) → DQ → Split`、`KV_Cache_DMA → Transpose → Split`；我方是 `Gemm(q_proj) → Transpose → DQ → Split`、`KV_Cache_DMA → Split`。Q 路径的 reshape 我方显式发一个 `Transpose`，K 缓存读路径的转置折进了 DMA 的出边宽（`1x32x128x1024`，与参考 `Transpose` 的出边同宽）。多一个与少一个恰好相抵，所以 node/edge = 200/331 与 `op_type` 逐类计数**仍然相等**——**A10 的"`op_type` 分布一致"只证明各类算子的个数一致，不证明边接得一样**，这一条不在它的核对面上。已核实数据布局等价：`Split→MatMul` 两边都是 `1x1x1x128`。连带可见差异：4 个 `Transpose` 的 `input_buffer_dtype` 跟着换了边（参考 3 int8 + 1 fp16、我方 2 + 2），每一侧对自己都自洽（都按各自生产者的输出位宽声明）。旧样本 `llama2_w4a8_decode_block_0` 在这一段与新基线相同，**不是换样本造成的**（评审 r7 问题 3） |

### 2.5 边界定义

**包含**：`tvmgen_default_nprm_main_0/runtime_files/` 内部的 `relay2gml_graph.gml`、
全部 `.bin`、`IO_info.txt` 的形式对齐；差异清单、根因定位、修正方案。

**不包含**：
- 本轮不写业务代码，只产需求文档
- 数值精度对齐（权重与标定数据不同，判据 4 已豁免）
- `DEBUG_*` 字段族（判据 5 豁免）
- **上层 7 个 fp32 bin 的产出**。实测确认它们在 `parser_output/` 下、
  **不在 `runtime_files/` 里**（逐一查过无同名文件），角色是标定输入而非交付物。
  本轮按"取其数值嵌成源码常数"处理（见 P0-1），不产出这些文件
- **`tvmgen_default_sp_main_0/` 子图的产出**（`output_0.bin`、`sp_0_i0.bin`、
  `tvmgen_default_sp_main_0.json`，共 3 个文件 6.4KB）。它与 `nprm` 平级、
  同样不在 `runtime_files/` 内，是甲方切给主机跑的另一张图（只处理
  `cache_position`）。要不要产见 Q2
- TVM relay 中间文本（`*_relay_mod.txt`、`qdq_mod_pre_build.txt`）与 parser 日志的复刻

## 三、非功能需求

1. **可复现且无外部路径依赖**：导出命令与现在一致（`--layers 1 --seq-len 16`）。
   标定数据以源码常数形式内置（见 P0-1），**运行时不读甲方目录下的任何 bin**，
   所以换机器、甲方目录不在时导出仍能跑通。
2. **不引入防御性兜底**：标定输入缺失、shape 不符、sf 算出 0 或 inf 时直接抛，
   不用默认值掩盖（遵循 `CLAUDE.md`"不写防御性兜底"）。
3. **代码量控制**：优先改现有函数（`runtime_files.py` 的 `np.zeros` 路径、
   `export.py` 的写死常量路径），不新增并列模块。标定常数模块另计约 0.07MB。
4. **可读**：标定这一步的中间产物（各节点 absmax、四相）要能 print 成可读文本。

## 四、技术约束与依赖

### 4.1 代码改动影响范围

| 文件 | 行数 | 预计改动 |
| --- | ---: | --- |
| `gml_bridge/runtime_files.py` | 464 | **主要改动**：十余处 `np.zeros` 改为接受标定数据 |
| 新增一个标定常数模块 | — | 小张量 5472 个 fp32 值 + 2 个 KV absmax 标量，带来源注释（见 P0-1）。约 0.07MB |
| `gml_bridge/export.py` | 840 | 写死常量：271/285 `sf:1.0`、507 `output_sf 1.0`（含 v_proj 特例）、458-459 Kantor **Shift**；`write_io_info` 的 sf 序列化与 rank |
| `gml_bridge/from_fx.py` | 2264 | **改动最大**：节点编号从 1 递增（P1-2）；hidden 边 rank（P0-4）；`relay2gml_version` 改 `"19.2.0"`（P1-4）。P0-5 不改这里，只改开关默认值 |
| `gml_bridge/phase_data.py` | 245 | **公式不改**（已验证正确），可能只调全零组保护的触发条件 |
| `scripts/export_gml.py` | — | 拆出 `build_parser`；加 `--no-decode-block-only`。**不加标定输入目录参数**——P0-1 要求标定值嵌成源码常数、不在程序里读 bin，A3b 还要"甲方目录改名后导出仍成功"（评审 r6 问题3） |
| `contracts/gml_coverage.py` | — | 不产字段的声明表需同步 P1-3 的 4 个字段 |

`gml_bridge/` 实测不 import `opcompiler_bridge` / `genesim_bridge`，改动闭合在本模块。

### 4.2 现有可复用的验证工具

| 脚本 | 行数 | 本轮用途 |
| --- | ---: | --- |
| `scripts/gml_structure_check.py` | 255 | 五条结构规则自检。**已核实规则适用**：拿甲方 Llama decode 样本跑，5/5 全绿（虽然规则当初以 ResNet50 立） |
| `scripts/gml_field_inventory.py` | 246 | 字段族覆盖率 |
| `scripts/verify_gml_artifact.py` | 977 | 回读自洽性（尺寸、int4 值域、per-group scale 数量、反量化误差） |
| `scripts/diff_prepare_out.py` | 456 | 按 (层类, phase, head) 配对而非文件名——**P1-1/P1-2 的配对思路可借用** |

### 4.3 潜在技术风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| 按 absmax 算的 KV sf 不复现甲方常量 | 已实测确认（2.3 P0-2） | 文档声明为已接受取舍；若甲方要求数值一致，需改为外部配置传入 |
| **P1-2 会改全部 bin 文件名** | 节点编号从 1 递增，所有 `*_<node_id>.bin` 随之变 | 做完后 2.4 的"已对齐项"要整表复测（A10）。P0-5 已降级为开关默认值改动，不再与 P1-2 强耦合 |
| P1-1 的清单可能失真 | 两边 node_id 规则不同，按文件名配对在比不同算子 | 先做 P1-2，或改用按算子类型+角色配对 |
| 改 `relay2gml_version` 为 `"19.2.0"` 可能与我方实际字段集不符 | 我方发的字段与 19.2.0 版解析器的预期未必一致（如多出 `Gather`） | 与 Q5 一起问甲方 |
| 标定常数与甲方样本绑定 | 嵌入的是这一份样本的数值；甲方换样本则常数过期 | 注释写明来源文件与提取日期，便于追溯重取 |

## 五、验收标准

每条都是可执行的判据，不用人工目测：

| # | 对应 | 判据 |
| --- | --- | --- |
| A1 | P0-1 | 所有 DQ 节点的 `output_buffer_phase_0_*.bin` 非零组数 == 组总数（现在是 0）。**范围就是 DQ 节点**（实测 37 个）：Softmax 的 phase0 是 fp16 位模式放 32 位字的高半字、低半字恒 0，按 fp16 逐元素读必然见零，两边同构，不算缺陷（评审 r6 问题5）。Softmax 五相另有判据（phase1 是 exp 数组，不逐元素等于 1.0） |
| A2 | P0-1 | 所有 `output_sf_*.bin` 解出的 fp16 值 > 0（现在有 37 个为 0.0） |
| A3 | P0-1 | 用**甲方自己的 DQ 输入**（如 `input_buffer_12.bin`）重算四相公式，phase0/phase1 与甲方文件**逐元素相同**（公式已验证成立）。**注意**：这条不等价于「用我方内置的 `HIDDEN_STATES` 常数时 phase0/phase1 与甲方逐元素相同」——3.1/Q14 的取舍下后者恒不成立（见 P0-1 与 Q14：`activation_for` 对所有 DQ 节点给同一份平铺/截断的 `HIDDEN_STATES`，而甲方是各节点自己的中间激活，两者 absmax 不同，phase0=2·absmax 自然不等） |
| A3b | P0-1 | 导出流程不读甲方目录下任何 bin：把 `xinfangzhou-resource` 整个目录改名后导出仍成功 |
| A4 | P0-2 | `KV_Cache_DMA`、`Split`、`v_proj` 的 `output_sf` != 1.0，且 int8 饱和比例 < 1% |
| A4b | P0-2 | 7 个 Gemm 中恰有 1 个（v_proj）的 `output_sf` != 1.0，其余 6 个 == 1.0 |
| A5 | P0-3 | `IO_info.txt` 中每个 int8 项的 `sf` == 对应 `output_sf_<id>.bin` 的 fp16 值 |
| A6 | P0-3 | `IO_info.txt` 能被 `eval` + numpy 命名空间读出，且 `sf` 的类型是 numpy 标量 |
| A7 | P0-4 | `IO_info.txt` 里 hidden 入口 rank == 3，其余六项 rank == 4 |
| A8 | 全局 | GML 引用的 bin 悬空数 == 0、孤儿数 == 0（**现已满足，不能回退**） |
| A9 | 全局 | 文件名模式类仍为 71 类（**现已满足，不能回退**） |
| A10 | 2.4 | 2.4 表中列出的"已对齐项"逐项复测仍通过（防止修 P0 时打破已对的部分） |
| A11 | P1-1 | 按算子类型+角色配对后，同角色 bin 的字节数相同。**已重统计并整改**（评审 r6 问题1）：194 个共有族里不符族 33 → 1；评审 r8 问题1 修掉最后那族后 **0 族不符、族集合两向完全相同**。**注意这条判据的盲区**：它只比「每个族有哪些字节数」，不比「谁跟谁配对」——`input_buffer_N` 128B 与 `input_sf_N` 64B 各自都合法，错的是拿 32 组 scale 去标注 128 个元素。配对关系另由 A18 守 |
| A18 | P0-2 | **每条 int8 入边的「缓冲元素数 / scale 个数」等于上游 DQ 的 group_size**（评审 r8 问题1 新增）。实测参考同口径全为 128；整改前我方那 32 个 `mha_batch_matmul1` 是 4，整改后全部为 128、与参考逐项相同。判据落在 `tests/test_gml_export.py::test_int8_input_scale_group_count_matches_the_buffer_it_labels`，并在 `export.py` 的 `_input_scale_of` 里加了同口径的 raise |
| A13 | P0-5 | `IO_info.txt` 出口 shape == `[1, 1, 4096]`，图里不含 `lm_head` 对应节点 |
| A14 | P1-2 | 节点 id 集合 == `1..N` 连续无空洞，且首个节点 id == 1 |
| A15 | P1-4 | GML 首部 `relay2gml_version "19.2.0"` |
| A16 | P1-5 | 拆成三句可逐条核对（按评审 r5 问题3 改写，口径同 A3 的先例）：(a) `phase_3` 族的 Kantor scale 逐元素等于同节点 phase2；(b) 非 phase 的 Kantor scale 每个 bin 非 0 且形式合法——**不与参考比数值**，理由同 P0-2 的 KV scale 取舍，且这四族没有 phase2 可比；(c) bias 全 0、Shift 全族为 int8 且取值符合本节对账表（`phase_3`/add/v_proj 为 -8，Cos/Sin/`kantor_B` 为 0）。**原措辞「Kantor scale 逐元素等于同节点 phase2」只对 (a) 那 37 个成立**，非 phase 的四族无 phase2 可比，按字面无法核对 |
| A17 | 全局 | 结构校验器 `scripts/gml_structure_check.py` 对我方产物 5/5 通过（已确认规则适用） |
| A12 | 全局 | `python -m pytest tests/ -x -q -k "not llama2_7b"` 全绿 |

## 六、调研补充信息

### 6.1 规范 PDF 的有效信息（VBU-GML Structure，20 页）

| 页 | 内容 |
| --- | --- |
| 1-2 | 图总体结构；输入/输出节点的字段表（`id`/`node_id`/`label`/`name`/`idx`/`is_buffer`/`original_name`/`from_tvm`/`output{i}_node_id`/`output_sf`/`output_zp`/`output_buffer`/`output_buffer_dtype`） |
| 3-7 | 通用算子字段。明确标注**无关字段**：`in/out_virtual`（L2A 自行推导）、`residual_<info>_buffer`（irrelevant）、`prev/next_task`（只需 `execution_id`）、`subnetwork`（NGC 当前不用） |
| 4 | `input_data_extensions`：1 有符号 / 2 无符号 / 3 浮点 |
| 5-6 | `nmu_mode`（floating_point / fixed_point / fixed2float）、`fpsu_mode`、`fpsu_spc`/`spg`/`spg_axis`/`spg_group_size`、`use_dynamic_quantization` |
| 6 | `kantor_mode` 六种取值：off / elementwise_mul_fp16 / float_elt_wise_and_scale / fp2int_converter / elementwise_mul_fixed_point / scalar |
| 7 | `contraction`：存放全部融合节点信息（激活、conv 后的 pool） |
| 8-16 | 各算子专属字段：Clip / Concat / EltwiseMul / LayerNorm / LeakyRelu / **Llama2Activation** / **Matmul** / Pad / Pool / Relu / Resize / **Silu** / **Softmax** / **Dynamic Quantization** / **Split** / **Transpose** / Upsample |
| 17 | **数组展开规则**：`pads [1, 2]` 要写成两行 `pads 1` / `pads 2` |
| 17-19 | 完整 Conv 节点示例（含 `contraction` 嵌套块与 `activation_lut_file`） |
| 20 | 已验证参考：ResNet18 / ResNet50 的 TVM parser 产物 |

**PDF 里没有 `Gather` 算子**，20 页的验证清单也只有 ResNet 系列。

### 6.2 甲方 TVM 流水线的可见环节

从 `parser_output/` 的文本产物可还原甲方的处理顺序：

```
原始模型
  → qdq_mod_pre_build.txt        插入 qnn.quantize / simulated_quantize
  → quantized_relay_full_mod.txt 量化后完整图
  → device_partitioned_relay_mod.txt  切分出 nprm（NPM 设备）与 sp（主机）两个子图
  → block_0_relay_mod.txt
  → runtime_files/               GML + bin
  （全程日志 Parser_ceva_logger_10_08_2026_13_04_35.txt，5.1MB）
```

`nprm` = NPM 设备子图（本轮对标目标），`sp` = 另一个子图（只处理 `cache_position`，
产物是 `output_0.bin` + `sp_0_i0.bin` + TVM json，各几百字节）。

### 6.3 参考资料

所有结论均来自本地实测与上述本地文件，未引用外部网络资料。可核查的原始位置：

- 甲方样本：`xinfangzhou-resource/model_layers_0_decode_v2/parser_output/`
- 规范：`xinfangzhou-resource/VBU-GML Structure-281025-031239.pdf`
- 我方产物：`/tmp/gml_out`（本轮由 `scripts/export_gml.py` 生成，退出码 0）
- 我方既有分析文档：`docs/GML字段与bin映射策略.md`（2028 行）、
  `docs/prepare_out-域确认表-20260918.md`（2473 行）、`docs/pim-compiler-v0.0.5.md`

## 七、待确认事项

### 7.1 已决策事项（原 Q1/Q2/Q3，本轮已定）

| 原 # | 事项 | 决策 |
| --- | --- | --- |
| Q1 | 子图边界：lm_head 归属 | **改**：图切到 hidden state 为止，出口 `[1,1,4096]`。落为 P0-5。**实测该能力已存在**（`--decode-block-only`），只需改默认值 |
| Q2 | 上层 7 个 fp32 bin 是否产出 | **不产出**。实测确认它们不在 `runtime_files/` 里，是标定输入而非交付物；数值以源码常数内置（P0-1） |
| Q2' | `tvmgen_default_sp_main_0/` 子图是否产出 | **暂不产出**，留 Q13 问甲方是否必需 |
| Q3 | 节点编号顺序 | **照甲方从 1 顺序递增**。落为 P1-2，与 P0-5 同批做 |
| — | `relay2gml_version` | **改为 `"19.2.0"`**。落为 P1-4 |
| — | 标定数据落地形式 | 嵌成源码常数、不在程序里读 bin；小张量全量嵌入、KV cache 只嵌 absmax 标量（P0-1） |
| — | `Gather` 节点 | 参考对齐路径（带 `--decode-block-only`）**本来就不发** Gather，与甲方一致；仅不带开关的整网导出会发，保留并声明为扩展。Q5 降为非阻塞 |

### 7.2 需要向甲方确认

| # | 问题 |
| --- | --- |
| Q4 | 本轮已按用户决定把 `relay2gml_version` 改成 `"19.2.0"`。需确认：甲方解析器是否按此字段做版本分派，我方实际发出的字段集（如多出 `Gather`）与 19.2.0 版的预期是否兼容 |
| Q5 | NPM 硬件是否支持 `Gather` 算子？规范 PDF 无此算子、甲方 GML 也没有。我方 embedding 查表节点本轮按用户决定**保留并声明为扩展**，但需甲方确认能否消费 |
| Q6 | KV cache 的量化 scale 是否必须与甲方数值一致？本轮按 absmax 自算（已确认取舍），若甲方要求数值一致则需改为外部配置传入 |
| Q7 | `IO_info.txt` 的 `sf` 序列化，甲方同一文件里 `array(1., dtype=float32)` 与 `np.float32(1.0)` 两种 repr 混用——消费端是否两种都接受 |
| Q8 | 甲方 GML 自身有 379 个悬空 bin 引用（全为 `DEBUG_*`）与 143 个孤儿文件，是否为预期状态 |
| Q13 | `tvmgen_default_sp_main_0/` 子图（3 个文件 6.4KB，只处理 `cache_position`）是否为交付必需项 |

### 7.3 待技术核实

原 Q9、Q12 已在本轮核实完毕（见下），剩余 1 条：

| # | 问题 |
| --- | --- |
| Q10 | ~~P1-1 的尺寸差异清单需在节点配对方式修正后重新统计~~ **已完成，Q10 关闭**。P1-2 完成后按"算子类型+角色"归族重统计（去掉节点号归并族名），194 个共有族里不符族 **33 → 1**。根因两条、都已修：① RoPE 子块的定标/零点/Shift/bias 四族按 head_dim 算宽，实测参考是单元素标量（30 个族）；② FPSU 三族按组数分宽，实测判据是节点类型——`Llama2ActivationDQ` 逐组、普通 `DynamicScaling` 发标量（12 个族）。第 3 条根因在评审 r8 问题1 找到并修掉：缓冲与定标"穿不穿过布局算子"两处口径不一致，改后 **0 族不符** |

本轮核实结果：

| 原 # | 问题 | 结论 |
| --- | --- | --- |
| Q9 | 结构校验器规则是否适用 Llama decode | **适用**。拿甲方样本跑 `gml_structure_check.py`，200 节点 / 331 边，5/5 规则全绿，EXIT=0。规则可继续当判据用 |
| Q12 | Gemm 的 `output_sf` = 1.019e-05 特例是哪个算子 | **v_proj**。甲方 7 个 Gemm 里只有 `self_attn_v_proj_MatMul_qidx37_params_36` 非 1.0，因为它的输出要写进 int8 value cache，带真实 requant scale；其余 6 个（q/k/o_proj、mlp down/gate/up）输出留在 fp16 域故为 1.0。**已并入 P0-2**，不再单列 |
| 原 Q11 | "Kantor 三族系数写死 0" | **两次判断均有误，已纠正**。`scale` 等于 phase2（P0-1 下游）、`bias` 恒 0（已正确）；`Shift` 全为 int8 取值 -8/0（早先按 fp16 读出的 -40704 是误读），我方 46/49 已对，剩 3 个是 `export.py:458-459` 分支漏判。**Q11 关闭，非外部问题** |
