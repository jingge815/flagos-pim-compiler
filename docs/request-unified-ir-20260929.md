# 需求文档：图编译阶段统一 IR 与 PIMMLIR 四维贯通

> 文档编号：request-unified-ir-20260929
> 创建日期：2026-09-29
> 关联项目：flagos-pim-compiler（图编译器，主）、FlagTree（算子编译器 / PIMMLIR）、GeneSim（仿真器）

## 一、需求背景与目标

### 1.1 业务背景

编译器端到端通路已经打通：能从 PIMMLIR 编译生成 GML 与 bin、能驱动 numpy 后端执行、能通过 GeneSim 完成功能仿真。

### 1.2 预期目标

**目标一：图编译阶段建立统一 IR**，覆盖算子语义 / 数据类型 / Placement / Memory Layout 四个维度的表达、校验与查询。

**目标二：PIMMLIR 侧同样覆盖这四个维度，并具备与统一 IR 对接的传递接口。** 不是单向下发，而是双向贯通——统一 IR 的四维信息能完整传进 PIMMLIR，PIMMLIR 处理优化后的结果能传回来。

**目标三：四维信息的唯一来源是统一 IR。** 本轮之后，所有涉及四维的分析与提取都必须从统一 IR 上做，不允许各模块自行维护副本或绕过它去问 PyTorch。

**目标四：PIMMLIR 回传的信息要对后续 pass 生效。** 传回来的不能是只写不读的死数据，必须真正参与下游的 GML 生成、numpy 执行与仿真输入构造。

**本轮不新增任何对外能力。** 三条通路已经走通，改动前后行为应当完全一致——唯一的区别是四维信息的取数来源从「散落各处」收敛为「统一 IR 一处」。所以判据不是「跑通」，而是「可依赖」：

| 判据 | 含义 | 今天的反例（实测） |
| --- | --- | --- |
| 四维表达齐 | 每维有明确载体 | `PIMTensorSpec` 查不到 dtype |
| 单一真源 | 每维只有一处定义，其余只能引用 | `MNEMONICS` 与 `_OPLEVEL_OPS` 重合 14 项却各自硬编码 |
| 可校验 | 非法组合直接抛错 | dtype 与 Memory Layout 无校验 |
| 双向贯通 | 统一 IR ↔ PIMMLIR 可传可回 | `dpusPerDevice` 预留未填 |
| 对下游生效 | 回传信息真正被消费 | 已有多个「有定义、无消费者」的字段 |

### 1.3 关联范围

| 仓库 | 路径 | 本轮角色 |
| --- | --- | --- |
| flagos-pim-compiler | `/media/disk/fengjingge/src/flagOS/flagos-pim-compiler` | **主改动**：统一 IR 载体、四维字段、真源收口、校验、与 PIMMLIR 的双向接口 |
| FlagTree | `/media/disk/fengjingge/src/flagOS/flagOS-installers/FlagTree` | **四维覆盖 + 传递接口**：接收四维信息、处理优化、回传 |
| GeneSim | `/media/disk/fengjingge/src/genesim` | **验收消费**：四维经统一 IR 流转后仿真仍正确 |

## 二、核心功能需求

### 2.1 功能清单

#### P0-1 统一 IR 的契约收口

**现状（实测）**：全仓 14 个元数据键，契约文件只登记 5 个，且已出现绕过常量的裸字符串。

| 已登记（`contracts/graph_meta.py`） | 未登记（散在写入它的 pass 里） |
| --- | --- |
| `device`（:3）`part_id`（:4）`spec`（:5）`redistribute`（:6）`fused_tail`（:8） | `pim_dynamic_scaling`（`graph/quant_pass.py:52`）、`pim_rms_norm`（`graph/fuse_pim.py:51`）、`pim_attention_scale`（`graph/fuse_pim.py:52`）、`pim_absorbed`（`graph/fuse_pim.py:62`）、`pim_kv_cache_dma`（`graph/kv_dma_pass.py:40`）、`pim_split`（`graph/kv_dma_pass.py:41`）、`pim_rope`（`graph/fuse_rope.py:39`）、`pim_head_role`（`graph/split_heads.py:45`）、`pim_head_index`（`graph/split_heads.py:46`） |

**契约腐化的既有实例**：`graph/kv_dma_pass.py:122` 与 `:151` 以字面量 `"pim_head_role"` 读取该键，而常量定义在 `graph/split_heads.py:45`。跨 pass 传递语义却不走常量，改名时不会有任何报错。

**需求**：四维信息的挂载点收归一处定义，新增 pass 不得私建键，跨 pass 读取必须走常量。

**验收判据**：契约声明的键集合**恰好等于**全仓在用集合，有测试守着（多一个少一个都失败）；全仓 `.meta[...]` / `.meta.get(...)` 的非测试调用点不再出现四维相关的裸字符串。

---

#### P0-2 数据类型维度补载体并收敛真源

**现状（实测）**：`PIMTensorSpec`（`contracts/pim_tensor_spec.py:45-72`）字段为 device / placement / residency / pinned_dpu_id / shard_map / reduce_type，**无 dtype**。唯一带类型的是 `RedistributeEdge.dtype`（:92），裸字符串。类型定义散在六处：

| # | 位置 | 内容 |
| --- | --- | --- |
| 1 | `contracts/gml_quant.py:58` `DTYPES` | 按缓冲区类的允许类型：activation `(int8, float16)`、weight `(int4, int8)`、bias `(int32, float32)`、scale `(float16, float32)`、output `(int8, float16, int16)` |
| 2 | `contracts/gml_quant.py:32-55` | 位宽与定标常量：`INT4_BYTES_PER_VALUE=1`、`INT4_MIN/MAX=-8/7`、`INT8_MIN/MAX=-128/127`、`INT4_SCALE_DIVISOR=8`、`INT8_SCALE_DIVISOR=128`、`WEIGHT_GROUP_SIZE=128` |
| 3 | `contracts/gml_hw_table.py:237` | `DATA_EXTENSION = {"int8": 1, "float16": 3}`，未知类型抛错 |
| 4 | `gml_bridge/from_fx.py:561` | `_BUFFER_DTYPES = {int8, int16, float16, float32}` |
| 5 | `orchestrator/layer_fields.py:20-21` | `DT_INT8/DT_FP16/DT_FP32 = 0/1/3`、`EXT_SIGNED/EXT_FLOAT = 1/3` |
| 6 | `node.meta["val"].dtype` | PyTorch 侧类型，在 IR 之外；全仓 `.meta["val"]` 32 次 + `.meta.get("val")` 33 次 |

注意第 5 处与第 3 处的编码口径不同（`DT_FP16=1` 对 `DATA_EXTENSION["int8"]=1`），两套编号各有含义，合并时必须区分「缓冲元素类型」与「通道宽度编码」两个概念，不能简单归一。

**已有的传播机制**：`gml_bridge/from_fx.py:657` 的 `_stamp_dtypes()` 已实现沿边递归解析 dtype，其文档注释明确记录了「布局算子的 dtype 是沿边传播的，不是按 op_type 固定」以及「`nodes` 并非拓扑序，按顺序传播会静默退回默认 fp16」这两条踩过的坑。**这份逻辑是本轮要保留并上移到统一 IR 的资产，不是要丢掉的东西。**

**需求**：元素类型与量化参数（粒度、分组、轴）在统一 IR 上可查；六处收敛为单一真源，其余改为引用。

**边界**：**不扩充计算与落盘类型集合**，保持 int4 / int8 / int16 / int32 / fp16 / fp32，只做表达规范、校验与全链路一致。索引类型 int64 单列登记（见 §2.3）。

**验收判据**：不访问 `node.meta["val"]` 即可查出任一节点的 dtype 与量化布局；六处中其余五处改为引用同一真源；`_stamp_dtypes` 的沿边传播语义与非拓扑序容错行为不退化。

---

#### P0-3 算子语义单一真源

**现状（实测）**：四份清单各自硬编码，其中两份高度重合。

| # | 位置 | 内容 | 规模 |
| --- | --- | --- | --- |
| 1 | `gml_bridge/from_fx.py:44` `OP_TYPES` | aten 目标 → GML 算子类型 | 28 项映射到 19 个 GML 类型 |
| 2 | `genesim_bridge/op_classify.py:249` `MNEMONICS` | 助记符元组 | 14 个 |
| 3 | `opcompiler_bridge/driver.py:83` `_OPLEVEL_OPS` | 算子编译器内核入口 | 15 个 |
| 4 | `contracts/fusion_contract.py:24/35/46/62` | 融合目标与可折激活 | 5 / 7 / 3 / 3 |

**关键实测**：第 2 与第 3 份去掉 `pim.` 前缀后比对，**交集 14 个、差集只有 `convert` 一项**——`MNEMONICS` 是 `_OPLEVEL_OPS` 的严格真子集，却写成两处独立字面量。这是「同一事实两处维护」的典型，改一处不会让另一处报错。

模块之间确有 import 关系（`from_fx.py` 引入 5 个 pass 的 META_KEY；`op_classify.py` 引入 `oplevel_kernel` 的 kernel 函数），但**清单本身彼此无推导关系**——问题在数据，不在模块依赖。

另有融合语义由各 pass 私有键承载：算子被融合后 `node.target` 不变而语义已变。

**需求**：一个算子在图阶段的完整语义（数学类型、融合结果、硬件单元归属）有唯一定义处；派生清单由真源推导而非重复书写。

**验收判据**：新增算子只需改一处；`MNEMONICS` 与 `_OPLEVEL_OPS` 之间存在显式推导或断言关系（而非两处字面量）；四份清单之间有可校验的引用关系。

---

#### P0-4 Memory Layout 维度补排布层

**现状（实测）**：切分与地址都有，排布层缺失。

| 已有 | 位置 | 内容 |
| --- | --- | --- |
| 切分明细 | `contracts/pim_tensor_spec.py:26-42` `TensorShardDetail` | dpu_id / shard_dim / start_idx / end_idx / local_shape / mram_offset |
| 三区规划 | `memory/mem_planner.py` `DPUPlan` | weight / kv_base / act_base / act_prefill / act_decode / total / pending_readers |
| KV 分块 | `memory/kv_layout.py` `KVRegionSpec` | layers / kv_heads / q_heads_by_kv / max_seq / head_dim / dtype_bytes / kv_base / kv_off |
| **对齐** | `contracts/op_contract.py:15` `dma_align` | **图侧已有**，带 2 的幂校验（:30），经 `driver.py:276` 下发为 `dma-align=`，`ir_cost.py:544` 从 `pim.dma-align` 读回 |
| 对齐工具 | `memory/kv_layout.py:27` `align_up` | KV 块按 DMA 边界对齐（:81） |

**缺的是排布层**：`memory/mem_planner.py:50-52` 的 `bytes_of()` 直接 `prod(local_shape) * itemsize`，等价于假定行主序紧密排列，这个假设不写在任何字段里。真实的步幅规则存在，但只活在编排器支线（详见 P0-6）。

**另一处重复**：`align_up` 有两份独立实现——`memory/kv_layout.py:27`（带参数校验）与 `orchestrator/l2_alloc.py:42`（无校验），两者互不引用。

**需求**：Memory 维度在统一 IR 上同时表达「如何切分」「落在哪」「怎么摆」。层级范围见 §6.2 四层说明，本轮做到第 3 层（排布语义），不做第 4 层（bank 交错）。

**验收判据**：任一张量的内存描述可从统一 IR 读出「切分 + 偏移 + 排布 + 对齐」四项，不依赖隐含假设；`align_up` 只有一份实现。

---

#### P0-5 统一 IR 是四维信息的唯一来源

**需求**：本轮之后，所有涉及四维的分析与提取都从统一 IR 做：

- GML 生成（`gml_bridge/`）不再自行从 FX 图推导四维，改为读统一 IR。
- 仿真输入构造（`genesim_bridge/`）同上。
- 算子编译契约下发（`opcompiler_bridge/`）同上。
- 内存规划（`memory/`）的产出回填统一 IR，而不是自成一套。

**关于下游的消费粒度（Q4 决策）**：三条通路**各自只消费它需要的那几维**，不要求每维都被完整消费。GeneSim 继续把 dtype 压成字节宽度整数（f16 与 bf16 在它那里不可区分），**不补 dtype 载体**——按分叉粒度它本就不需要区分。本轮约束的是「取数来源必须是统一 IR」，不是「每维都要传到底」。

**验收判据**：这三个 bridge 不再出现绕过统一 IR 直接读 `node.meta["val"]` 或自建四维副本的代码路径。

---

#### P0-6 编排器改为消费统一 IR

**现状（实测）**：排布规则的真源只在编排器支线。`orchestrator/` 共 3014 行，`layer_fields.py` 占 1660 行。其中 14 处 stride 字段**实为两类不同的东西**：

| 类别 | 字段 | 处数 | 性质 |
| --- | --- | --- | --- |
| **内存排布步幅** | `Input Stride X`（:488）、`Output Stride X/Z`（:494-498）、`Eltwise broadcast Input Stride X`（:1174）、`Data scale stride X/Z`（:1276-1277）、`DDR data scale stride X/Z`（:1293-1294）、`DDR Output stride X/Z`（:1370-1371）、`DDR Weight stride X/Z`（:1402-1403） | **10** | 真正的字节排布，属 Memory Layout |
| 卷积滑窗步长 | `Filter Horizontal/Vertical Stride`（:502-503）、`Pooling Horizontal/Vertical Stride`（:528-529） | 4 | 几何参数，值恒为 0/1，属算子语义 |

排布规则的核心是两个函数（`orchestrator/layer_fields.py:27-37`）：

```python
def align16(width: int) -> int:
    return l2_alloc.align_up(width, 16)

def stride_z(width: int, *, final: bool, scalar_align16: bool = False) -> int:
    """终相 align16(W)+15；中间相 =W；Width=1 的中间相有时 16。"""
```

外加 `orchestrator/net_ini.py:51-61` 四个全网恒定步幅、`orchestrator/l2_alloc.py:42` 的第二份 `align_up`。规则里有实测得来的特例（如 `dq_p2` 的 `Output Stride Z = out_w + 15` 不按 16 对齐，见 `:491-495` 注释），搬家时必须整体保留。

**需求（Q3 决策）**：把上表「内存排布步幅」这 10 处的规则收进统一 IR 的 Memory 维度，**编排器改为从统一 IR 消费，不保留双份**。编排器退化为渲染层——从 IR 读出步幅与对齐，按目标格式写成层参数文本，自己不再计算。

**边界**：4 处卷积几何 stride 不属本维度，保持在编排器内。

**验收判据**：
- `stride_z` / `align16` 的计算逻辑在统一 IR 侧只有一份，编排器侧无重复实现。
- 编排器产出的层参数文本与改动前**逐字节一致**（与 §5.2 同理，证明是重构而非改行为）。

---

#### P1-1 PIMMLIR 的四维覆盖与传递接口

**现状（实测）**：PIMMLIR 是真正的 MLIR 方言，四维表达能力**强于 Python 侧**，缺的是「有人给它喂真值」。精确计数（本轮实测，修正了此前的估算）：

| 项 | 数量 | 出处 |
| --- | --- | --- |
| 算子总数 | **37** = 20 个分块级（`TTPIM_Op`）+ 17 个算子级（`TTPIM_OperatorOp`） | `PIMOps.td` |
| 结构化属性 | **19** | `PIMAttrDefs.td` |
| 枚举 | **22** | `PIMAttrDefs.td` |
| 自定义类型 | **4** | `PIMTypes.td` |
| 算子 verifier | **34** 处 `hasVerifier = 1` | `PIMOps.td` |
| 属性 verifier | **14** 处 `genVerifyDecl` | `PIMAttrDefs.td` |

17 个算子级 op：`quantize` `dynamic_quant` `dequantize` `convert` `matmul` `conv` `lut` `fpsu_scale` `kantor` `eltwise` `pool` `gather` `global_pool` `reduce_axis` `rope` `normalize` `softmax`。

四维对应的属性：

| 维度 | PIMMLIR 载体 |
| --- | --- |
| 算子语义 | `#pim.contraction` `#pim.act_spec` `#pim.pool_spec` `#pim.window` `#pim.phase_spec` `#pim.datapath` `#pim.kantor_spec` `#pim.kantor_block` `#pim.fpsu_spec` `#pim.vpu_params` `#pim.broadcast_spec` `#pim.transpose_purpose` |
| 数据类型 | `#pim.quant_spec`（granularity / axis / groupSize / dataExt / role / fpDtype / range）、`#pim.weight_binding`（format / role / elemBits / groupSize / sfMultiplier / contentHash） |
| Placement | 内存空间 4 种（`#pim.wram` `#pim.mram` `#pim.l1` `#pim.l2`，经 `TTPIM_MemSpace` 归并）+ 功能单元枚举 |
| Memory Layout | `#pim.tasklet_tiled`（sizePerTasklet / taskletsPerDpu / **dpusPerDevice** / order）+ 分配算子 `alignment` + 模块属性 `pim.dma-align` |

**`dpusPerDevice` 的单边预留，四条证据**：

1. ODS 注释明写：该字段存在是为了让「graph-level compiler, which is what decides cross-DPU sharding, has somewhere to record its decision」。
2. builder 实现是 `(void)numDpus;`，即接了参数但丢弃。
3. `opcompiler_bridge/*.py` 里搜 `tasklet_tiled` / `dpusPerDevice`：**命中 0**。
4. **printer 主动省略全 1 的该字段**（`lib/Dialect/TritonPIM/IR/Dialect.cpp:114-116`，注释 "Elide the all-ones default"）。实测 175 份 pimir 缓存中 26 份含 `tasklet_tiled`，序列化文本里 `dpusPerDevice` 出现 **0 次**——按 printer 逻辑，这正证明它们全是默认的全 1。

**图编译器当前发出的 pimir**（实测 `.opcompiler_cache/00de4ae3032d74ec.pimir.mlir` 全文），四维几乎全空：

```mlir
module attributes {pim.target = "pim:v1"} {
  tt.func @kernel(%arg0: tensor<1x6x4096xf16>) {
    %0 = pim.reshape %arg0 : tensor<1x6x4096xf16> -> tensor<1x6x32x128xf16>
    tt.return
  }
}
```

对比 FlagTree 自己 TTIR 降级出来的那条路径（`0d4b0767145c3f8c.pimir.mlir`），模块属性与布局编码都齐备：

```mlir
module attributes {"pim.dma-align" = 64 : i32, "pim.mram-bytes" = 4294967296 : i64,
                   "pim.num-dpus" = 8 : i32, "pim.num-tasklets" = 4 : i32, ...} {
  ... tensor<1x512xf32, #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
                                            taskletsPerDpu = [1, 4], order = [1, 0]}>>
```

即：**同一个方言，两条产生路径的四维富度差距悬殊**。图编译器手写那条是主路（供 GML 与执行），却是信息最少的一条。

**需求**：

1. 统一 IR 的四维信息完整写进图编译器下发的 pimir，含 `dpusPerDevice` 的跨 DPU 切分决策。
2. PIMMLIR 侧四维覆盖的缺口补齐。
3. 提供明确的传递接口，两侧口径一致、字段可对应。

**边界**：`dpusPerDevice` 这一项**方言定义无需改动**（字段与 parser / printer / verifier 均已就绪），只补写入侧。其余维度按对齐结果决定是否需扩 ODS。

**验收判据**：多 DPU 切分时下发的 pimir 里 `dpusPerDevice` 出现且不等于全 1（利用 printer 的省略规则：字段一旦出现即非默认值）；四维字段在两侧有逐项对应关系与交叉校验测试。

---

#### P1-2 PIMMLIR 回传信息对后续 pass 生效

**现状（实测）**：已有一条可用的反向通道 `opcompiler_bridge/phase_source.py`，把算子编译器算出的相位数据回灌给 GML。它的结构是 `PhaseSource`（:68）+ `phase_source_from_graph()`（:167）+ `cross_check()`（:241），消费点在 `gml_bridge/export.py:88/105/149/155`，并有专门的测试 `tests/test_gml_depends_on_opcompiler.py` 断言「带与不带 phase_source 产出不同」。

**这条通道的模式正是本轮要推广的范式**——它已经证明了「回传能改变下游产物」是可测的。但它只覆盖相位数据，不覆盖四维。

**需求**：把这条反向通道推广到四维——PIMMLIR 处理优化后的四维结果回传统一 IR，并真正参与下游的 GML 生成、numpy 执行与仿真输入构造。

**反面判据（必须避免）**：已有多个「有定义、无消费者」的既有实例，说明这类腐化很容易发生：

| 实例 | 状态 |
| --- | --- |
| placement sidecar 的 `semantic_role`、`weight` | 我方写出，GeneSim 侧 **0 处消费**（注释说明是排错用的调试信息，属有意为之） |
| placement sidecar 的 `shard_axis` | 我方在 `len(shards) > 1` 时写出，GeneSim 侧 **0 处消费** |
| `#pim.combine_mode` | 枚举 3 值，`ExpandPhases.cpp` 五处（:408/440/470/480/491）均传空值占位；GML 无对应字段（详见 P2-1） |

区别在于：调试信息无消费者是可接受的（应显式声明），而**语义字段无消费者就是缺陷**。本轮新增的回传字段不允许出现只写不读或只读不写。

**验收判据**：每个新增回传字段都能指出**生产方与消费方各一处**；有测试证明改变回传值会改变下游产物（照 `test_gml_depends_on_opcompiler.py` 的做法）；无消费者的调试字段必须显式登记理由。

---

#### P1-3 四维可校验

**现状（实测）**：校验分布极不均衡。

| 维度 | 校验现状 |
| --- | --- |
| Placement | **最完整**：`Placement.validate()`（`contracts/pim_tensor_spec.py:11-22`）、`TensorShardDetail.validate()`（:34-42）、`PIMTensorSpec.validate()`（:54-72）、策略契约（`graph/strategy.py:34-51`、`contracts/partition_plan.py:76-114`）；PIMMLIR 侧 4 种内存空间均带 verifier |
| 算子语义 | FlagTree 侧充分（34 处算子 verifier）；**Python 侧无** |
| 数据类型 | 仅 `data_extension()` 对未知 dtype 抛错（`gml_hw_table.py:244`）、`HardwareBudget` 对 `dma_align` 做 2 的幂校验；**张量级 dtype 无校验** |
| Memory Layout | `align_up` 对 `align <= 0` 抛错（`kv_layout.py:29`）、`KVRegionSpec.validate()` 存在；**排布层无校验**（因为尚无载体） |

**需求**：四维都有校验，非法组合直接抛错，不用默认值掩盖（沿用 `CLAUDE.md`「不写防御性兜底」）。

**验收判据**：每维至少一组「非法输入必须抛错」的反例测试。

---

### 2.2 核心流程

四维信息的流转是**一条经过 PIMMLIR 的环路**，不是三条平行分叉各自从 FX 图取数：

```
torch.export 导出 FX 图
        ↓
  ┌─────────────────────────────────────────────────────────┐
  │ 统一 IR（图编译阶段，本轮建设核心）                          │
  │   算子语义 │ 数据类型 │ Placement │ Memory Layout           │
  │                                                         │
  │ 由图拆分、切分传播、融合 pass、内存规划共同填充                │
  └─────────────────────────────────────────────────────────┘
        │  ① 四维下发（含 dpusPerDevice 的跨 DPU 切分决策）
        ↓
  ┌─────────────────────────────────────────────────────────┐
  │ FlagTree / PIMMLIR（算子编译阶段）                          │
  │   四维覆盖 + 处理优化                                       │
  │   -pim-fuse-activation → -pim-expand-phases              │
  │                        → -pim-verify-gml-contract         │
  └─────────────────────────────────────────────────────────┘
        │  ② 优化结果回传统一 IR（推广现有 phase_source 通道）
        ↓
  ┌─────────────────────────────────────────────────────────┐
  │ 统一 IR（已带 PIMMLIR 回传信息）                             │
  └─────────────────────────────────────────────────────────┘
        │  ③ 下游从统一 IR 取数，不再自行从 FX 图推导
        ├──────────────┬──────────────┐
        ↓              ↓              ↓
    GML + bin      numpy 后端      GeneSim 输入
   (gml_bridge)    (runtime/)     (genesim_bridge)
                                   ↓
                              编排器（orchestrator）
                              从 IR 读步幅，渲染层参数
```

各节点的现状与本轮动作：

| 节点 | 现状 | 本轮动作 |
| --- | --- | --- |
| 统一 IR 填充 | 四维散在 14 个元数据键、四份清单、六处类型定义 | 收口为统一契约 |
| ① 四维下发 | 只发算子与张量类型，四维几乎全空（见 P1-1 对比） | 补全，含 `dpusPerDevice` |
| PIMMLIR 处理优化 | pass 链已就绪，37 个算子 / 19 个属性 / 34 处 verifier | 四维缺口补齐 |
| ② 回传 | `phase_source` 通道已验证可行，只覆盖相位数据 | 推广到四维 |
| ③ 下游取数 | 三个 bridge 各自从 FX 图推导 | 改为读统一 IR |
| 编排器 | 自行计算步幅（10 处布局 stride） | 改为消费 IR（P0-6） |

**顺序约束**：图编译器侧的融合 pass 有固定顺序且有硬依赖（`gml_bridge/export.py:117-144`：`fuse_rope` → `fuse_for_pim` → `fuse_graph` → `split_attention_heads` → `insert_kv_dma_and_split` / `insert_dynamic_scaling`，注释明确记录了「RoPE 必须在逐头展开之前折」「split_attention_heads 必须在前三个之后」）。统一 IR 的引入不得破坏这些约束。

### 2.3 边界定义

**包含范围**

- 图编译阶段统一 IR 的载体与四维字段（**具体如何设计留给设计文档**）。
- 四维各自收归单一真源，其余位置改为引用。
- 四维的校验逻辑。
- 统一 IR ↔ PIMMLIR 的双向传递接口；PIMMLIR 侧四维覆盖补齐。
- 下游三条通路改为从统一 IR 取数。
- 编排器的排布规则收进统一 IR，编排器改为消费 IR（P0-6）。
- `rope.subBlocks` / `transpose.purpose` 补生产方（随 P1-1）。
- 三个仓库各自的单元测试适配。

**不包含范围**

- **不扩充计算与落盘类型集合**：保持 int4 / int8 / int16 / int32 / fp16 / fp32，不引入 bf16、fp8、亚字节打包。
  - **索引类型 int64 单列**，与计算类型分开登记（`contracts/dtypes.py::INDEX_DTYPES`）。导出的 llama 图里有 3 个 int64 索引张量（input_ids / arange / unsqueeze），其中一个是 DPU 节点、内存规划要对它算字节数；不登记就不是等价重构。它不参与数值计算，所以不属本条禁止的「计算与落盘类型」——这与 P0-2 区分 `DATA_EXTENSION` 与 `DT_*` 两套编号是同一口径。
- **不新增硬件层级**：Placement 保持 host / DPU / 功能单元 / 内存空间（wram / mram / l1 / l2），不引入 bank / 通道 / 核心。
- **不引入反向 / 训练语义**：三仓均为纯推理编译器。
- **不改变 GML 产物**：字段集与上一轮对齐甲方参考格式的结果一致，且要求逐字节不变。
- **不改卷积几何 stride**：`Filter` / `Pooling` 的 4 处滑窗步长不属 Memory Layout，留在编排器。
- **不在需求阶段定义统一 IR 的设计方案**：载体形态已定为方案 A（见 §6.1），字段结构与接口签名属设计文档范畴。
- **不做 32 层全量导出**（沿用既有取舍）。
- **不额外编写用于证明「统一 IR 够用」的样本 pass**：判据就是三条通路正常打通且原有测试全通过（Q1 决策）。

## 三、非功能需求

### 3.1 性能要求

- 统一 IR 不得让编译期耗时出现可感知增长。基线：测试全量（排除 `llama2_7b`）177.50 秒。
- 校验只在编译期执行，不进入运行时路径。

### 3.2 兼容性要求

- **运行环境**：`source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh`（torch 2.9.1 / transformers 4.57.6 / python 3.10.20）。每个新 shell 必须先 source，否则 `import torch` 失败。
- **FlagTree 重新编译**：`bash /media/disk/fengjingge/src/flagOS/flagOS-installers/0-install-flagtree.sh`。该脚本自 `371691b` 起在构建并验证通过后**自动把新编译的 PIM Triton 同步进 PyTorch 环境**（`--skip-pytorch-sync` 可关闭），因此重建 FlagTree 后**不需要**再跑 `2-install-pytorch.sh`。首次安装顺序仍为 0 → 1 → 2。
- **两条 pimir 路径必须同源**：A 路走进程内 `libtriton`（PyTorch 环境侧），B 路走 `triton-opt` 可执行文件（`flagTree/build/flagtree-cmake/bin/`，`opcompiler_bridge/driver.py:107` 硬编码此路径）。改动 PIMMLIR 后两者都要重建，否则 A 路认新属性、B 路不认。
- 三条下游通路的既有测试必须全部保持通过。

### 3.4 可维护性与可观测性

- 代码量受 `CLAUDE.md` 约束：先搜索再写、最小实现、不预造抽象、删优于加；每次改动报告净增删行数。

## 四、技术约束与依赖

### 4.1 现有技术栈约束

| 约束 | 说明 |
| --- | --- |
| 图阶段宿主结构 | `torch.fx.GraphModule`，四维挂在 `node.meta`；本轮沿用（方案 A） |
| `contracts/` 是地基 | 14 个文件共 2299 行，全仓唯一真源；改任何字段先 grep 调用方，改完同步全链路 |
| PIMMLIR 是真正的 MLIR 方言 | 定义在 FlagTree `include/triton/Dialect/TritonPIM/IR/`（`.td` + C++ verifier），不是文本约定；37 算子 / 19 属性 / 22 枚举 / 4 类型 / 34 处算子 verifier / 14 处属性 verifier |
| pimir 有 A / B 两条产生路径 | A 路进程内 `libtriton`（TTIR 降级，四维富）；B 路 `triton-opt` 可执行文件（图编译器手写整算子级文本，四维空，但是主路）。两者必须同源 |
| 融合 pass 顺序有硬依赖 | 见 §2.2 顺序约束 |
| 统一 IR 载体形态 | **已定为方案 A**（扩展现有 Python 结构，见 §6.1）；字段结构与接口签名留设计文档 |

### 4.2 代码改动影响范围

**flagos-pim-compiler（主改动）**

| 文件 | 现状 | 本轮动作 |
| --- | --- | --- |
| `contracts/graph_meta.py` | 11 行、5 个键 | 四维挂载点收口至此 |
| `contracts/pim_tensor_spec.py` | 93 行；`PIMTensorSpec` 无 dtype | 补 dtype 与排布字段 |
| `contracts/gml_quant.py:32-58` | 位宽常量 + `DTYPES` + `QuantLayout` | 收敛为类型真源或改引用 |
| `contracts/gml_hw_table.py:237` | `DATA_EXTENSION` | 改引用（注意与 `layer_fields.py` 的编号口径不同） |
| `contracts/fusion_contract.py` | 融合表 4 组 | 纳入算子语义真源 |
| `contracts/op_contract.py:15` | `dma_align`（已有，带 2 的幂校验） | 纳入 Memory 维度的对齐表达 |
| `gml_bridge/from_fx.py:44,561,657` | `OP_TYPES`、`_BUFFER_DTYPES`、`_stamp_dtypes` | 前两者改引用真源；`_stamp_dtypes` 的沿边传播逻辑上移到统一 IR（**保留其非拓扑序容错**） |
| `gml_bridge/export.py:88-155` | 融合 pass 编排 + `phase_source` 消费点 | 适配统一 IR，保持顺序约束 |
| `genesim_bridge/op_classify.py:249` | `MNEMONICS`（14） | 改为由 `_OPLEVEL_OPS` 推导或加断言 |
| `genesim_bridge/placement_export.py:398-420` | sidecar 导出（含 3 个无消费者字段） | 改为读统一 IR；无消费者字段显式登记 |
| `opcompiler_bridge/driver.py:83,107,276` | `_OPLEVEL_OPS`、`triton-opt` 路径、`dma-align` 下发 | 改引用真源 |
| `opcompiler_bridge/oplevel_emitter.py`、`oplevel_kernel.py` | 手写 MLIR 文本，四维几乎全空 | **补四维写入，含 `dpusPerDevice`** |
| `opcompiler_bridge/phase_source.py:68,167,241` | 反向通道，只覆盖相位 | **推广到四维** |
| `memory/mem_planner.py:50-52` | `bytes_of()` 隐含行主序 | 排布显式化；规划结果回填统一 IR |
| `memory/kv_layout.py:27` | `align_up`（第一份实现） | 与 `l2_alloc.py:42` 合并为一份 |
| `orchestrator/layer_fields.py`（1660 行，`align16`/`stride_z` :27-37，10 处布局 stride）、`net_ini.py:51-61`、`l2_alloc.py:42` | 排布规则的现有真源 | **P0-6：规则搬进统一 IR，编排器改为消费**，退化为渲染层；4 处卷积几何 stride 不动 |
| `graph/kv_dma_pass.py:122,151` | 裸字符串 `"pim_head_role"` | 改为走常量 |
| `graph/` 下 5 个带私有键的 pass | 各自声明键 | 改用统一 IR 接口 |
| `tests/`（71 文件 / 1024 用例） | — | 补契约完整性、校验反例、跨仓口径对齐、回传生效性测试 |

**FlagTree（四维覆盖 + 传递接口）**

| 文件 | 本轮动作 |
| --- | --- |
| `include/triton/Dialect/TritonPIM/IR/PIMAttrDefs.td` | `#pim.tasklet_tiled` 的 `dpusPerDevice` **无需改**（parser :87-103 / printer :109-120 / verifier :125-132 均已就绪）；其余按对齐结果定 |
| `include/triton/Dialect/TritonPIM/IR/PIMOps.td` | `rope.subBlocks`（:1122）、`transpose.purpose`（:1407）补生产方后核对 verifier |
| `lib/Dialect/TritonPIM/IR/Ops.cpp:1032` | ROPE `order[]` 与 Python `ROPE_UNITS` 已逐字一致，保持同步 |
| `lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp:408/440/470/480/491` | `CombineModeAttr{}` 空值占位处**本轮不动**（Q2 挂账） |
| `test/Dialect/TritonPIM/` | 补 `dpusPerDevice` 非全 1 的正例与四维传递用例 |

**GeneSim（验收消费）**

| 文件 | 本轮动作 |
| --- | --- |
| `src/ir/model_ir.py` | 四维流转后的输入适配；**不补 dtype 载体**（Q4） |
| `src/scheduler/gene_sim_scheduler.py:447,691-701` | sidecar 消费逻辑核对（现读 `dpu_id`、`local_in_features`、`local_out_features`、`op_type`、`device_hint`、`shards`） |
| `tests/sim/` | 补四维消费验收用例 |

### 4.3 潜在技术风险与应对建议

| # | 风险 | 应对建议 |
| --- | --- | --- |
| 1 | 契约收口时漏掉裸字符串调用点，改名后静默失效 | 已实测发现 `kv_dma_pass.py:122/151` 一处；收口后加「非测试代码不得出现四维裸字符串键」的源码级断言 |
| 2 | 四份清单合并时发现语义确实不同（粒度差异是刻意的） | 先出差异对照表。已知 `MNEMONICS` ⊂ `_OPLEVEL_OPS`（差 `convert`），这一对可直接推导；`OP_TYPES` 是 aten→GML 映射、`fusion_contract` 是融合规则，两者视角不同，建立引用而非合并 |
| 3 | `_stamp_dtypes` 的沿边传播逻辑上移时丢失容错 | 该函数注释记录了两条踩过的坑（布局算子 dtype 不能按 op_type 写死、`nodes` 非拓扑序需按需递归）。迁移后必须保留这两条行为，并保留原注释 |
| 4 | Python 对齐 PIMMLIR 口径时，某些字段在图阶段无数据可填（如 `weight_binding.contentHash`） | 允许留空并显式声明，不造假值 |
| 5 | `dpusPerDevice` 写入后 FlagTree verifier 拒绝 | verifier 要求三个数组与 `order` 等秩（`Dialect.cpp:130-132`）。先跑通最小样例（tp=2）再铺开 |
| 6 | 步幅规则搬进 IR 后编排器改为消费（Q3 已定，不保留双份），`layer_fields.py` 1660 行里 10 处布局 stride 改造面不小 | **分两步落地**：先在 IR 侧实现 `align16`/`stride_z` 并与编排器现有结果做只读全等对照（全等才继续），对照通过后再删编排器侧计算、改为读 IR。层参数文本逐字节不变是最终判据。注意保留实测特例（如 `dq_p2` 的 `Output Stride Z = out_w + 15` 不按 16 对齐） |
| 7 | 回传通道推广后出现新的「只写不读」死字段 | P1-2 要求每个回传字段指明生产方与消费方各一处；无消费者的调试字段显式登记 |
| 8 | 两条 pimir 路径不同源导致假结果 | 改 PIMMLIR 后 A / B 两路都重建；仓内已有探针 `genesim_bridge/flagtree_driver.py::_check_inprocess_matches_triton_opt()`，本轮实测通过 |
| 9 | 同步清单在 `0-install-flagtree.sh` 与 `2-install-pytorch.sh` 各存一份（脚本注释已警示「改同步清单时两处都要改」） | 改时两处一起改；或后续抽公共脚本 |

## 五、验收标准

### 5.1 回归判据（硬门槛）

**flagos-pim-compiler**

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -q -k "not llama2_7b"
```

- 基线（2026-09-29 实测）：**981 passed, 1 skipped, 42 deselected in 177.50s**
- 验收：新增用例后仍全绿，**不允许新增失败或跳过**

**GeneSim**

```bash
cd /media/disk/fengjingge/src/genesim
./run.sh --test          # 全部
```

**FlagTree**

- lit 测试通过。两个陷阱：① 必须对着**源码树**跑（`$TRITON_BUILD_DIR/test/Dialect/TritonPIM` 是空目录，对着它跑会得到假的「全绿」）；② `/dev/shm` 占满时用私有 tmpfs（`unshare -Umr`）或按 `RUN` 行逐条执行 `triton-opt` + `FileCheck`。

### 5.2 产物不变判据

**全部产物必须与改动前一致。** 这是「重构而非改行为」的硬证据——本轮只改四维的取数来源，不改行为。

| 产物 | 判据 | 验证方式 |
| --- | --- | --- |
| GML 字段集 | 不变 | `contracts/gml_coverage.py` 的 EMITTED **444** 族、PENDING_QUANTIZATION **10** 族、PENDING_CONVOLUTION **6** 族、NOT_APPLICABLE **26** 族约束不被破坏 |
| GML 文本与 bin | **逐字节不变** | 改动前后各导出一份，`.gml` 与全部 `.bin` 逐字节比对 |
| GML 结构规则 | 不变 | `scripts/export_gml.py` 的 5 条结构规则 + dtype 覆盖检查 + 引用集恰好等于落盘集 |
| 编排器层参数文本 | **逐字节不变** | P0-6 搬家后 `prepare_out` 产物逐字节比对 |
| GeneSim 输入 | 语义不变 | `.ir` JSON + placement sidecar 既有字段语义不变 |

### 5.3 逐项验收条件

| 编号 | 验收条件 | 验证方式 |
| --- | --- | --- |
| P0-1 | 契约键集合**恰好等于**全仓在用集合；非测试代码无四维裸字符串键 | 集合相等断言（今天 5 vs 14）+ 源码级检查 |
| P0-2 | 不访问 `node.meta["val"]` 可查出 dtype 与量化布局；六处收敛为一处；沿边传播语义不退化 | 查询接口 + 引用关系 + 传播行为测试 |
| P0-3 | 新增算子只需改一处；`MNEMONICS` 与 `_OPLEVEL_OPS` 有显式推导或断言关系 | 引用关系测试 |
| P0-4 | 任一张量可读出「切分 + 偏移 + 排布 + 对齐」四项；`align_up` 只有一份实现 | 结构完整性 + 源码级检查 |
| P0-5 | 三个 bridge 无绕过统一 IR 的路径 | 源码级检查（参照 `tests/test_runtime_compiled_coverage.py` 从源码文本抠调用点的做法） |
| P0-6 | `stride_z`/`align16` 在 IR 侧只有一份，编排器无重复实现；层参数文本逐字节不变 | 源码级检查 + 产物比对 |
| P1-1 | 多 DPU 时 pimir 里 `dpusPerDevice` **出现且非全 1**（利用 printer 省略默认值的特性：字段一旦出现即非默认）；两侧四维字段逐项对应 | 解析 pimir 断言 + 交叉校验（参照 `tests/test_flagtree_ods_hygiene.py`） |
| P1-2 | 每个回传字段可指明生产方与消费方各一处；改回传值会改变下游产物 | 生产消费对 + 变异测试（参照 `tests/test_gml_depends_on_opcompiler.py`） |
| P1-3 | 四维各有「非法输入必须抛错」反例 | 反例测试 |

### 5.4 「可依赖」的最终判据（Q1 决策）

统一 IR 建成的标志不是「新增了什么能力」，而是**同样的三条通路，改为从统一 IR 上分析和转换之后，行为完全不变**。

落为三条可验证条件，**不额外编写样本 pass 来证明**：

| # | 条件 | 证明了什么 |
| --- | --- | --- |
| 1 | 三条通路正常打通，全部产物不变（§5.2） | 是重构，没有偷偷改行为 |
| 2 | 三仓原有测试全部通过（§5.1） | 没有破坏既有能力 |
| 3 | 四维唯一来源、无旁路（P0-5、P0-6） | 后续分析与优化只能参照统一 IR |

三者关系：**第 1、2 条守住「没改坏」，第 3 条守住「以后只能走这条路」。** 与改动前唯一的区别就是取数来源——四维信息从「散落在 14 个元数据键、四份算子清单、六处类型定义、编排器支线」收敛为「统一 IR 一处」。

## 六、调研补充信息

以下结论全部来自本地实测（代码文件、构建产物、测试运行），未引用外部网络资料。可核查位置均已给出 `文件:行号`。

### 6.1 统一 IR 载体形态：已定为方案 A

**决策：方案 A（扩展现有 Python 结构）。** 三个选项的对比如下，供设计阶段参考实现路径。

| 选项 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **A（已选）** | 保留 FX 图 + `node.meta`；在 `contracts/` 补齐契约，`PIMTensorSpec` 补 dtype 与排布字段，四维各自收归真源 | 改动面最小，不碰现有 pass 的工作对象；与 `CLAUDE.md`「最小实现、不预造抽象」一致；可逐 pass 试点迁移；FX 图既有生态（`meta["val"]`、torch.export 契约）继续可用 | 四维仍挂在弱类型字典上，靠约定而非结构约束；「统一」程度取决于契约执行力度 |

### 6.2 Memory Layout 的四个层次（Q3 的背景解释）

「内存布局」在本项目里可分四层，由粗到细：

| 层 | 回答的问题 | 状态 | 载体 |
| --- | --- | --- | --- |
| **第 1 层：切分** | 张量切成几份、每份归哪台 DPU、切在哪一维 | **已有** | `TensorShardDetail`（shard_dim / start_idx / end_idx / local_shape） |
| **第 2 层：地址** | 每份分片在本 DPU 的 MRAM 里从第几字节开始 | **已有** | `mram_offset` + `DPUPlan` 三区 + `KVRegionSpec.kv_off` |
| **第 3 层：排布** | 这段字节内部怎么摆——行优先还是列优先、有无对齐填充、相邻元素间隔多少字节（步幅） | **部分有**：对齐已有（`op_contract.dma_align`、`kv_layout.align_up`）；**步幅缺** | 步幅规则只活在编排器支线（`layer_fields.py:27-37` 的 `align16`/`stride_z`、`net_ini.py:51-61` 四个全网步幅），既不进 IR 也不进 GML。`mem_planner.py:50-52` 按「元素数 × 元素宽度」算字节，等价于假定行主序紧密排列，该假设无字段承载 |
| **第 4 层：介质交错** | 这段地址落在哪个 bank / 通道、多 bank 间怎么交错 | **不做** | 三仓 Python 侧 `bank` 零命中；GeneSim 有地址映射器但用算子编号乘 64 当假地址；甲方参考 GML 亦无 bank 字段 |

**「做到哪一层」的影响**：

- 做到第 2 层 = 保持现状，「布局」名不副实（只有地址，没有排布）。
- **做到第 3 层 = 本轮决策**。把编排器已有的真实步幅规则收进统一 IR，让「行主序连续」从隐含假设变成显式字段。好处是后续做访存分析、判断两张量能否复用同段地址、评估 DMA 效率时有据可查。代价可控——规则已存在且经过验证，只是搬家。
- 做到第 4 层 = 超出本轮（已在 §2.3 排除），需新增硬件层级，与「不扩充」口径冲突。

所以 **P0-4 / P0-6 的口径是：做到第 3 层，不做第 4 层**；且按 Q3 决策，第 3 层搬进 IR 后**编排器改为消费 IR，不保留双份**。

**注意 stride 一词在编排器里混用了两种含义**，只有前者属本维度：

| 含义 | 字段 | 处数 | 归属 |
| --- | --- | --- | --- |
| 内存排布步幅 | `Input/Output/DDR ... Stride X/Z`、`Data scale stride X/Z` 等 | 10 | **Memory 第 3 层，搬进 IR** |
| 卷积滑窗步长 | `Filter Horizontal/Vertical Stride`、`Pooling Horizontal/Vertical Stride` | 4 | 算子语义（几何参数），值恒为 0/1，留在编排器 |

### 6.3 四维在三仓的覆盖度（实测）

| 维度 | FlagTree（PIMMLIR） | flagos-pim-compiler | GeneSim |
| --- | --- | --- | --- |
| 算子语义 | **强**：37 算子（20 分块级 + 17 算子级）、34 处算子 verifier、14 处属性 verifier、独立 `-pim-verify-gml-contract` pass | **中**：28 项 aten 映射到 19 个 GML 类型；四份清单各自硬编码（其中两份重合 14/15） | **中**：22 个编译模板 + 2 条 pim mlir 前端 |
| 数据类型 | **强**：`#pim.quant_spec` / `#pim.weight_binding` 字段完备 | **中**：GML 侧 dtype 字段族齐全、`_stamp_dtypes` 沿边传播成熟；但真源分散六处、IR 无载体 | **弱**：dtype 压成字节宽度整数 |
| Placement | **中**：内存空间 4 种（wram/mram/l1/l2）+ 功能单元枚举，均带 verifier；`dpusPerDevice` 预留未填 | **中**：切分结构完整且三层校验齐全（注意本仓 `Placement` 指**张量切分**，非硬件拓扑） | **中**：自建三层资源模型 + 完整拓扑 |
| Memory Layout | **中**：`#pim.tasklet_tiled` + `alignment` + DMA `contiguous_dim`/`elem_stride` | **弱**：切分与地址齐、对齐有（`dma_align`）、**步幅缺**，行主序是隐含假设 | **弱**：地址映射器存在但用假地址 |

### 6.4 术语消歧：本项目里「布局」有三个含义

这条必须记入文档，否则后续沟通持续踩坑。技术方案（`docs/spec.md` 指向的 2482 行文档）里：

| 出处 | 「布局」所指 | 载体 |
| --- | --- | --- |
| 问题 2 §（9） | **张量的切分分布**（Shard / Replicate / Partial） | `PIMTensorSpec` |
| 问题 8 §二 | **MRAM 物理偏移规划**（权重区 / KV 区 / 激活区三区） | `DPU_k.plan` |
| §3.3 | 图编译器下发算子编译器的「MRAM 中的布局」 | 算子编译契约 |

本需求里的 Placement / Memory Layout 按用户定义为「算子放到哪执行」与「内存如何切分及布局」，与上表既有交叠也有新增，实现时需明确对应关系。

### 6.5 GML 里以其他名字存在的四维对应物（实测）

甲方参考 GML（`model_layers_0_decode_v2/parser_output/tvmgen_default_nprm_main_0/runtime_files/relay2gml_graph.gml`）字段统计：`dtype` **1619**、`data_extension` **1002**、`transpose` **115**、`nmu_mode` **71**、`weight_format` **64**、`residual_input_buffer` **332**；而 `bank` / `placement` / `shard` / `dpu` / `device` / `core` / `stride` / `layout` / `align` / `interleave` **全部为 0**。

我方产物（`/tmp/gml_r5/relay2gml_graph.gml`，15635 行）：`data_extension` **996**、`transpose` **115**、`nmu_mode` **71**、`weight_format` **64**、`split_channel_number` **32**；`stride` / `layout` / `align` / `bank` 均为 **0**。

即 **Placement 与 Memory Layout 的语义对应物在 GML 里是存在的**（功能单元归属靠 `nmu_mode` 等字段族、权重排布靠 `weight_format`、通道切分靠 `split_channel_number`、残差连接靠 `residual_input_buffer`），只是不叫这个名字，也没在 IR 层被统一建模。这正是本轮要解决的。

`gml_bridge/writer.py:9` 明确记录了不产出硬件放置字段的理由：执行顺序由对方的 L2Analyzer 自行推导。甲方 GML 里的 `use_input_buffer_1 = "L2A_ignore"`（2 处）也印证了这一分工。

### 6.6 环境陷阱现状（本轮实测复核）

| # | 陷阱 | 状态 |
| --- | --- | --- |
| 1 | FlagTree lit 假绿：对着空的构建目录跑得到「全绿」 | **仍然有效**，必须对着源码树跑 |
| 2 | `/dev/shm` 占满时 lit 报 `ENOSPC` | **仍然有效**，用私有 tmpfs（`unshare -Umr`）或按 `RUN` 行逐条执行 |
| 3 | 两份 `libtriton.so` 不同源 | **已修正，本轮实测确认**。详见下方 |

## 七、已决策事项与待确认事项

### 7.1 本轮已决策

| 原 # | 事项 | 决策 | 落在 |
| --- | --- | --- | --- |
| Q1 | 「可依赖」是否需要额外的样本证明（如新写一个分析 pass） | **不需要**。判据就是「基于统一 IR 把已有三条通路正常打通，保持原有测试验证全部通过」。唯一不同点是四维信息之前散落各处，本轮之后必须从统一 IR 上分析和转换 | §5.4、§2.3 |
| Q2 | `combine_mode` 的处理方向 | **不考虑，作为遗留问题挂账** | P2-1、§2.3 |
| Q3 | 步幅规则搬进 IR 后编排器是否改为消费 IR | **编排器改为消费统一 IR**，不保留双份 | P0-6、§6.2 |
| Q4 | GeneSim 侧是否补齐 dtype 载体 | **继续只消费它需要的部分**，不补 dtype 载体 | P0-5、§2.3 |
| Q6 | 统一 IR 载体形态 | **方案 A（扩展现有 Python 结构）** | §6.1、§4.1 |

### 7.2 待确认事项

| # | 问题 |
| --- | --- |
| Q5 | `docs/request-gml-align-20260928.md` §7.2 仍有 6 条需向甲方确认的事项未闭环（原 Q4~Q8、Q13）。本轮 GML 要求逐字节不变，理论上不触碰这些点，但需一并核对确认。**状态：待确认** |

### 7.3 本轮实测对既有文档的修正

本轮逐文件核实时，发现既有文档与本文档早期版本中有几处结论不准，一并记录以免后续沿用：

| # | 原结论 | 实测修正 |
| --- | --- | --- |
| 1 | `docs/pimmlir-primitives-20260926.md` §8.3：`pim.rope` 的 `subBlocks` 与 Python 静态表「两处语义已经不一致」 | **不成立**。`ROPE_UNITS`（`gml_hw_constants.py:244`）与 C++ `order[]`（`Ops.cpp:1032`）6 个名字逐字相同，C++ 侧注释也明写要求两处同步 |
| 2 | 本文档早期版本：PIMMLIR 有 8 个结构化属性、12 个属性 verifier、38 个算子 verifier | **修正为** 19 个结构化属性、22 个枚举、4 个自定义类型、14 处 `genVerifyDecl`、34 处 `hasVerifier = 1` |
| 3 | 本文档早期版本：四份算子清单「互不 import」 | **表述不准**。模块之间确有 import（`from_fx.py` 引入 5 个 pass 的 META_KEY、`op_classify.py` 引入 `oplevel_kernel` 的 kernel 函数），但**清单数据本身**无推导关系。更强的证据是 `MNEMONICS` ⊂ `_OPLEVEL_OPS`（重合 14/15，仅差 `convert`）却各自硬编码 |
| 4 | 本文档早期版本：图编译器侧无对齐表达 | **不成立**。`contracts/op_contract.py:15` 已有 `dma_align`（带 2 的幂校验），经 `driver.py:276` 下发、`ir_cost.py:544` 读回；`memory/kv_layout.py:27` 有 `align_up`。缺的只是**步幅**，不是整个第 3 层 |
| 5 | 本文档早期版本：26 份含 `tasklet_tiled` 的缓存里 `dpusPerDevice` 命中 0，据此说「图编译器没填」 | **结论对，但证据需补一步**。`dpusPerDevice` 在文本里不出现是因为 printer 主动省略全 1 默认值（`Dialect.cpp:114-116`）。这反而给出了更干净的验收方法：字段一旦出现即非默认值 |
| 6 | 本文档早期版本：编排器 12 处 stride | **修正为 14 处**，且分两类——10 处内存排布步幅（属 Memory Layout）+ 4 处卷积滑窗步长（属算子语义，值恒 0/1） |

### 7.4 实施复盘（2026-09-30）对本文档的修正

实施完成后按 `docs/review-unified-ir-20260930.md` 逐条实测复核，本文档有四处需要修正或补明。**目标与验收判据本身不变**——变的是几处事实描述与边界口径。

| # | 本文档原文 | 实测修正 | 落在 |
| --- | --- | --- | --- |
| 1 | §2.3 不包含范围：「不扩充数据类型集合：保持 int4 / int8 / int16 / int32 / fp16 / fp32」 | **补 `int64`，单列为索引类型**。实测导出的 llama 图里有 3 个 int64 索引张量（input_ids / arange / unsqueeze），其中一个还是 DPU 节点，内存规划要对它算字节数；不补就不是等价重构。原口径的本意是禁止 bf16 / fp8 / 亚字节打包这类**计算与落盘**类型，索引类型不在其列。修订后的表述为「不扩充**计算与落盘**类型集合」，`int64` 作为索引类型单独登记（`contracts/dtypes.py::INDEX_DTYPES`），与计算类型分开——理由同 §2.1 P0-2 对 `DATA_EXTENSION` 与 `DT_*` 两套编号口径的区分 | §2.3 |
| 2 | P1-1「`dpusPerDevice` 的跨 DPU 切分决策由统一 IR 下发」 | **补一条准确性要求**：切分决策取自节点**输出**的 `shard_map`，编码必须描述同一块张量。实施中出现「决策取自输出的 `shard_dim`、编码却贴在第一个实参的类型文本上」的坐标系错配（输入输出秩不同的算子会标错轴），详见评审问题 6。判据补：「编码所指张量必须与决策所依据的分片一致」 | P1-1 |
| 3 | §4.2 flagos-pim-compiler 表的 FlagTree 与 GeneSim 两仓行动项 | **本轮两仓均未改动**，行动项未落实。FlagTree 的 `dpusPerDevice` 正例与四维传递 lit 用例、GeneSim 的 `src/ir/model_ir.py` 适配与 `tests/sim/` 验收用例都还是待办。GeneSim 侧的零改动是 P1-2 未接通消费点的**结果**而非独立结论，不能以回归通过代替验收 | §4.2、§5.1 |
| 4 | P1-2 的「三个消费点」只在设计文档中给出（§4.8.4），本文档未写具体落点 | **补明本轮口径与未闭环状态**。实施复核实测：设计选定的三个消费点都不成立——`gml_bridge/from_fx.py` 没有 `bytes_of` 调用（GML 缓冲尺寸由边 dims×dtype 定，且 GML 路径没有 spec）、`memory/` 有 WRAM 概念（`memory/kv_layout.py:198,238-239` 的 `wram_budget_bytes` 与超限检查，此前「零命中」的说法不成立）但不在规划路径上、`genesim_bridge/cost_extractor.py` 本身就是 `tile` / `wram` 的**生产方**（从 pimir 抽）而非消费方。此外 `PhaseSource` 只在 GML 路径产生，`_layout_feedback_of` 读的模块属性只在 A 路出现，B 路从不带——回传通道两端皆空。**但目标四与 P1-2 判据不放宽**：本轮之后必须补一个真实消费点并补上设计 §4.8.5 要求的变异测试，否则即为本文档 §2.1 P1-2 反面判据明令禁止的「只写不读」 | §2.1 P1-2、§5.3 |

**关于第 4 条的取舍说明**：本文档 §1.2 目标四把「回传信息对后续 pass 生效」与另外三项目标并列为硬要求，§5.3 的 P1-2 验收条件也写明「每个回传字段可指明生产方与消费方各一处；改回传值会改变下游产物」。实施结果只完成了通道与生产方，消费侧为空。经复核，设计挑的三个消费点确实不适用，但这是**选点选错**，不构成「消费侧无需落地」的依据——否则本轮产出的正是一个结构完整、端到端空转的死通道，与 §2.1 P1-2 用 `combine_mode` 立下的反面案例同类。因此本文档保留目标四与 P1-2 判据原样，把消费点的重新选点与变异测试列为未闭环项。

### 7.5 第五轮（2026-10-01）：四维逐项补齐

**本节替换了本节此前的一版**。上一版把目标二按「收窄口径」处理——承认 `replicate` /
`partial`、`mram_offset` 等不下发，理由是「PIMMLIR 侧没有消费者」。经确认该做法不接受：
**要求是把消费者补上，不是把缺口改成不算缺口**。本轮逐项补齐，下面是补齐后的实际口径。

判定一项该不该下发，仍用 §2.1 P1-2 的纪律：**下发的字段必须在 PIMMLIR 侧有真实消费方**。
每补一项都同时给出「下发点 + 消费点（改变哪个真实决策）+ 可失败判据」。

| 维度 | 本轮补齐 | 消费点（改变什么决策） | 仍未闭环及理由 |
| --- | --- | --- | --- |
| 算子语义 | `combineMode` 五处占位改填真实值 | `EltwiseOp::verify` 校验 `skip_connection` 必须是 add —— 该属性此前 5 个生产方、0 个读取方 | `purpose = absorbed` **不改**：它并非「只产不销」——图层面由 `from_fx.py:233,925` 的 `ABSORBED_META_KEY` 消费（决定发不发节点），而 IR 侧三种 purpose 降成同一段物理置换，**在 PIMMLIR 层没有合法消费点**。试过一版按 absorbed 少计搬运的实现，实测把真实搬运漏掉，已撤回。`VpuParamsAttr`、`DmaDir`、`StationarityAttr` 仍无生产方，属 ODS 预留 |
| 数据类型 | 累加宽度进入 footprint 与 WRAM 判据；`pim.placed-elem-bytes` 回传 | 前者：`out` 缓冲按 f32 累加器计宽（f16 操作数下此前少算一半，实测 tile-wram-bytes 37120 → 41216）；后者：成本模型的元素宽度由「按类型名猜」改为「按回传」（实测 f16 猜 2、回传 1 时按 1） | 「存储 + 累加 + 量化」仍未合成单一载体。三者的信息都到得了 PIMMLIR，只是分别到；合成属于重构既有表达，等真有消费方需要同时看三者再做 |
| Placement | `replicate` / `partial` 两档 + `partial` 的 `reduce` 真下发；模块级 `#pim.placement` 两条路都发 | `partial` 的单台 footprint 计入跨 DPU 归约暂存（实测 7168 → 8192，另回传 1024）；`replicate` 激活既有 verifier（三档下发后才在真实链路上可触发） | `dpuIds` 仍只用于范围校验、`stage` 仍零消费方。两者服务的是跨 kernel 的多 stage 调度，而现有三个 pass 都是单 kernel 粒度，加不出真实决策；等编排层出现再做 |
| Memory Layout | 切分、地址、对齐下发，排布有保留 | 切分：`dpusPerDevice` 进张量编码，`verifyLayoutsMatchPlacement` 与容量判据读它。排布：`elem_strides` 经 `#pim.placement` 的 `order` 下发，A 路 `-pim-explicit-dma` 按维序重算 `elem_stride`；**当前唯一生产方是行主序，取值恒为默认**，判据由合成输入的用例覆盖，登记为已知限制。地址：`mram_offset` 下发为 DMA 的 `mram_offset`，只盖到结果张量那条，作为信息属性保留；**不参与地址计算**——运行时已经按每个 access 的真实地址传指针，内核再加一遍会写到 2 倍偏移处。对齐：`align_bytes` 下发为 `#pim.placement` 的 `alignBytes`，`-pim-tile-to-budget` 在它比 `pim.dma-align` 更严时按它选分块（实测 tile 从 `m=8,n=64` 变为 `m=16,n=32`） | 排布层生产方恒为行主序，是已知限制，见上 |

**两条下发路径的不对称一并修掉**：A 路此前只有模块级 placement、B 路只有张量级编码，
现在两条路都发模块级 placement，下游才能读到切分意图。但**发了不等于校验了**——
跨载体漂移校验（`verifyLayoutsMatchPlacement`）原先只挂在 `-convert-triton-to-pim`
的出口，B 路的 pass 链（`-pim-fuse-activation -pim-expand-phases
-pim-verify-gml-contract`）根本不经过它，所以 B 路发不发模块属性，那条校验都不触发。
已把同一条校验补进 `-pim-verify-gml-contract`，两条路现在都在各自的出口守着。

**未闭环项的共同性质**：都不是「载体缺失」，而是**缺少真实的生产方或消费者**——
继续补需要先有上游的决策（多 stage 调度、分片对齐要求），不是继续加字段能解决的。
按 CLAUDE.md「不写没有消费者的代码」，这些留白是刻意的，逐条记在上面，不是遗漏。

### 7.6 决策变更记录

| 原 # | 事项 | 原决策 | 本次变更 | 依据 |
| --- | --- | --- | --- | --- |
| Q2 | `combine_mode` 的处理方向 | 「不考虑，作为遗留问题挂账」（§7.1） | **本轮补齐**：5 处占位填真实值、新增 verifier 消费方 | 经确认要求补全四维。已实测其安全性：① `combineMode` 是 `OptionalAttr`，不填不打印；② `gml_bridge` 完全不读 pimir；③ 相位解析器用别名白名单，新属性被安全忽略。所以填它不改 GML 产物 —— 逐字节比对实测零差异 |
