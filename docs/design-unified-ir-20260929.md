# 技术设计文档：图编译阶段统一 IR 与 PIMMLIR 四维贯通

> 文档编号：design-unified-ir-20260929
> 创建日期：2026-09-29
> 对应需求：`docs/request-unified-ir-20260929.md`（request-unified-ir-20260929）
> 关联仓库：flagos-pim-compiler（主）、FlagTree（PIMMLIR）、GeneSim（消费侧）

## 一、需求回顾与设计目标

### 1.1 需求覆盖对照表

本设计逐条覆盖需求文档 §2.1 的全部功能点。**P2-1 的 `combine_mode` 已按 Q2 决策挂账，不在本设计范围**。

| 需求编号 | 需求要点 | 本文档对应章节 | 状态 |
| --- | --- | --- | --- |
| P0-1 | 统一 IR 的契约收口（14 个键收归一处，禁裸字符串） | §4.1 | 全覆盖 |
| P0-2 | 数据类型补载体并收敛真源（六处→一处） | §4.2 | 全覆盖 |
| P0-3 | 算子语义单一真源（四份清单建立推导关系） | §4.3 | 全覆盖 |
| P0-4 | Memory Layout 补排布层（步幅 + 对齐） | §4.4 | 全覆盖 |
| P0-5 | 统一 IR 是四维唯一来源（三个 bridge 去旁路） | §4.5 | 全覆盖 |
| P0-6 | 编排器改为消费统一 IR（10 处布局 stride） | §4.6 | 全覆盖 |
| P1-1 | PIMMLIR 四维覆盖与传递接口 | §4.7 | 全覆盖；**FlagTree 方言零改动**（四维按内存层级逐条核实，§4.7.1~3.7.5） |
| P1-2 | PIMMLIR 回传对后续 pass 生效 | §4.8 | 全覆盖 |
| P1-3 | 四维可校验（每维反例测试） | §4.9 | 全覆盖 |
| P2-1（部分） | `rope.subBlocks` / `transpose.purpose` 补生产方 | §4.7.4 | 随 P1-1 |
| P2-1（部分） | `combine_mode` | — | **按 Q2 挂账，不做** |

### 1.2 设计目标与非目标

**设计目标**（对应需求 §1.2 四条）：

1. 在 `contracts/` 内建立四维的统一契约，四维各有明确载体、单一真源、可查询、可校验。
2. 统一 IR 的四维信息能写进下发给 PIMMLIR 的 pimir 文本，PIMMLIR 的优化结果能回传。
3. 三条下游通路（GML/bin、numpy 后端、GeneSim）改为从统一 IR 取数，不再自行从 FX 图推导。
4. 回传信息真正参与下游产出，不做只写不读的死数据。

**非目标**（对应需求 §2.3 不包含范围）：不扩类型集合、不加硬件层级、不做 bank 交错、不引入反向语义、不改 GML 产物字节、不补 GeneSim 的 dtype 载体、不动 4 处卷积几何 stride、不处理 `combine_mode`、不额外写样本 pass。

**本轮性质是重构。** 全部产物（GML 文本与 bin、编排器层参数文本）必须逐字节不变——这是唯一能证明「只改取数来源、没改行为」的硬判据。

### 1.3 关键约束

| 约束 | 来源 | 对设计的影响 |
| --- | --- | --- |
| 载体形态 = 方案 A | 需求 Q6 决策 | 保持 `FX 图 + node.meta`，不新建图级数据结构 |
| 产物逐字节不变 | 需求 §5.2 | 所有重构必须行为等价；新字段默认值必须让旧路径走出完全相同的结果 |
| `contracts/` 不能 import 上层 | 实测：`contracts/*.py` 无任何 `graph`/`gml_bridge`/… 的 import | 契约文件无法直接引用住在 `graph/` 的载荷 dataclass，需专门设计（§4.1.2） |
| 编译期耗时不可感知增长 | 需求 §3.1，基线 177.50 秒 | 校验默认开启但必须 O(1)/O(n) 级；不引入重复遍历 |
| 契约不满足直接抛错 | 需求 §3.3、`CLAUDE.md` | 不写 `try/except` 兜底、不用默认值掩盖上游错误 |
| 最小实现、不预造抽象 | `CLAUDE.md` | 不引入注册表/插件机制；只有出现 2 个以上真实实现才抽象 |

## 二、现状基线（本轮实测核实）

### 2.1 代码基线

#### 2.1.1 `contracts/` 是干净的地基

实测确认：`contracts/` 下 14 个文件共 2299 行，**不 import 任何上层模块**（`graph`/`gml_bridge`/`genesim_bridge`/`opcompiler_bridge`/`memory`/`runtime`/`orchestrator`/`comm` 全部零命中）。依赖方向是单向的 `上层 → contracts`。

`contracts/graph_meta.py` 已被 **12 个非测试模块** import，覆盖全部层次：

```
opcompiler_bridge/oplevel_emitter.py   memory/mem_planner.py    memory/kv_layout.py
gml_bridge/export.py                   gml_bridge/from_fx.py    genesim_bridge/placement_export.py
runtime/exec_plan_gen.py               runtime/compile.py       graph/spec_prop.py
graph/fuse_pim.py                      graph/fuse.py            graph/partition.py
```

**结论：把统一契约放在 `contracts/` 不需要新增任何依赖边**，这是方案 A 成本低的根本原因。

#### 2.1.2 四维当前的载体分布

| 维度 | 当前载体 | 问题 |
| --- | --- | --- |
| 算子语义 | `node.target`（aten）+ 9 个私有 meta 键 + 四份独立清单 | 无单一真源；融合后 `target` 不变而语义已变 |
| 数据类型 | **无 IR 载体**；靠 `node.meta["val"].dtype` 问 PyTorch（非测试代码 **41 处**）+ 六处定义表 | `PIMTensorSpec` 无 dtype 字段 |
| Placement | `node.meta["device"]` + `PIMTensorSpec`（device/placement/residency/pinned_dpu_id/shard_map/reduce_type） | **最完整**，三层 `validate()` 齐备 |
| Memory Layout | `TensorShardDetail.mram_offset` + `DPUPlan` 三区 + `KVRegionSpec` + `op_contract.dma_align` | 切分、地址、对齐都有，**步幅缺**；行主序是 `bytes_of()` 的隐含假设 |

`.meta["val"]` 的 41 处非测试调用点分布（改造工作量的直接依据）：

| 模块 | 处数 | 取的是什么 |
| --- | --- | --- |
| `graph/split_heads.py` | 11 | 新建节点时写 `val`（示例张量），以及读 shape |
| `graph/spec_prop.py` | 10 | 读 `ndim` / `shape` 做切分推导 |
| `gml_bridge/from_fx.py` | 6 | 读 dtype / shape |
| `runtime/exec_plan_gen.py` | 4 | 读 `element_size()` 与 `dtype` |
| `memory/mem_planner.py` | 2 | 读 `element_size()` |
| `graph/quant_pass.py` / `graph/kv_dma_pass.py` / `gml_bridge/export.py` | 各 2 | 读 shape / numel |
| `opcompiler_bridge/oplevel_emitter.py` / `graph/fuse_rope.py` | 各 1 | 读 shape |

**关键区分**：其中「读 dtype / element_size」的是**本轮要改为从 IR 取的**；而「`graph/split_heads.py` 写入 `val` 示例张量」与「`spec_prop.py` 读 ndim 做推导」属于 FX 图自身的生态，**必须保留**——`val` 是 torch.export 的产物，统一 IR 的 dtype 字段正是从它派生而来，不是要取代它。

#### 2.1.3 载荷 dataclass 全部住在 `graph/`，且已有循环依赖痕迹

实测 8 个载荷类型的位置：

| 载荷类型 | 位置 | 对应 meta 键 |
| --- | --- | --- |
| `FusedTail` | `graph/fuse.py:33` | `fused_tail` |
| `KvDmaSpec` / `SplitSpec` | `graph/kv_dma_pass.py:45` / `:57` | `pim_kv_cache_dma` / `pim_split` |
| `DynamicScalingSpec` | `graph/quant_pass.py:65` | `pim_dynamic_scaling` |
| `RopeMatch` | `graph/fuse_rope.py` | `pim_rope` |
| `RmsNormFusion` | `graph/fuse_pim.py` | `pim_rms_norm` |

**既有的循环依赖痕迹**：`graph/fuse_pim.py:230` 把 import 写在函数体内部：

```python
    from graph.fuse import FusedTail
```

这是 `graph/` 内部互相引用载荷类型时的规避写法。`contracts/graph_meta.py:7` 也只能用注释描述载荷类型（`# 折进本节点的激活与池化，值是 FusedTail`），因为它不能 import `graph/`。

**这是设计必须解决的核心矛盾**：契约要登记「每个键的载荷类型」，但载荷类型住在契约无法 import 的层。解法见 §4.1.2。

#### 2.1.4 内存层次的分管边界（全文最关键的架构事实）

本项目的内存层次由**两个互不相交的组件**分管。这条边界决定了 P0-4 / P0-6 / P1-1 三个功能点的范围，必须先立清。

| 层级 | 负责组件 | 排布信息的流向 | 是否经 PIMMLIR |
| --- | --- | --- | --- |
| WRAM / MRAM | 图编译器 + 算子编译器 | 统一 IR → pimir → 算子编译器 | **经过** |
| L1 / L2 | 编排器 | GML → 编排器 → 层参数 | **不经过** |
| DDR | 编排器 | 同上 | **不经过** |

四条实测证据：

**证据一：编排器是链路上串行的第三段**（`orchestrator/__init__.py` 模块文档原文）：

```
编排器：GML 之后的那一段。链路里的位置是**串行的第三段**，不回填 GML：

    图编译器 + 算子编译器 → GML（200 节点）→ 编排器 → 层参数 + net.ini
...
**不管切分放置**——那是图编译器的事（`graph/strategy.py`）。
```

**证据二：L2 地址分配在架构上就归编排器**（`orchestrator/l2_alloc.py` 模块文档原文）：

```
这是编排器存在的核心理由。... 单个算子的 pass 看不到全局，
图编译器也从没做过地址分配，所以只能在这里做。
```

**证据三：编排器完全看不到 `PIMTensorSpec`**。实测 `orchestrator/` 下搜 `PIMTensorSpec` / `TensorShardDetail` / `shard_map`：**零命中**。它的几何「从 GML 边取」（`layer_fields.py:3`）。

**证据四：内存层级词频印证分工**：

| 组件 | L2 | DDR | WRAM | MRAM |
| --- | --- | --- | --- | --- |
| `orchestrator/` | 87 | 69 | 1 | 1 |
| `contracts/op_contract.py` + `memory/` | 0 | 0 | 3 | 11 |
| PIMMLIR `.td` | 14 | 0 | 27 | 15 |

**对设计的三条硬约束**：

1. P0-4 给 `TensorShardDetail` 新增的 `elem_strides` 是 **MRAM 级**的排布（§4.4.5），随 `#pim.placement` 的 `order` 下发，改变 `-pim-explicit-dma` 证明出的 `elem_stride`。L1 / L2 / DDR 级的步幅仍归编排器，不经 PIMMLIR。
2. P0-6 搬进统一 IR 的 10 处 stride 全在 L2/DDR 级，与 PIMMLIR 的表达能力无关。
3. P1-1 判断 PIMMLIR 表达能力时，只能用 WRAM/MRAM 级的事实作证据；用 L2/DDR 级的事实会得出错误结论（本设计第二版即犯此错，见 §4.7.2）。

#### 2.1.5 `TensorShardDetail` 是 frozen，回填走 `replace`

`contracts/pim_tensor_spec.py:25` 标注 `@dataclass(frozen=True)`。`memory/mem_planner.py` 回填 `mram_offset` 时用 `dataclasses.replace`，共 3 处：`:89`、`:274`、`:281`。

**设计含义**：新增的排布字段必须沿用同一不可变回填模式，不能改成可变 dataclass——否则会破坏 frozen 带来的共享安全性。

#### 2.1.6 两份 `align_up` 实现

| 位置 | 是否带校验 | 调用方 |
| --- | --- | --- |
| `memory/kv_layout.py:27` | **有**（`align <= 0` 抛错） | `kv_layout.py:81`、`mem_planner.py:91/215/219` |
| `orchestrator/l2_alloc.py:42` | 无 | `l2_alloc.py:107/180/191`、`layer_fields.py:28`（`align16`） |

两者算法相同（`(n + align - 1) // align * align`），互不引用。`orchestrator/layer_fields.py` 已经 import `contracts`（`:12-14`），所以把统一实现放进 `contracts/` 对两侧都无新增依赖。

### 2.2 改造动作与依赖顺序

#### 2.2.1 方案选型

需求 Q6 已定为**方案 A（扩展现有 Python 结构）**，故不再比较 A/B/C。但 A 内部仍有一个真实的分岔需要决策——**契约文件的组织方式**，见 §4.1.2 的三个子方案对比。

#### 2.2.2 分层与数据流

统一 IR 不是一个新的数据结构，而是**一组住在 `contracts/` 的契约 + 挂在 `node.meta` 上的四维载荷**。分层保持不变：

```
                    ┌──────────────────────────────────────┐
                    │  contracts/  （统一 IR 的契约层）        │
                    │                                      │
                    │  unified_ir.py   四维键登记表 + 查询 API │
                    │  ir_payloads.py  载荷 dataclass（新）    │
                    │  pim_tensor_spec.py  + dtype + 排布     │
                    │  op_semantics.py 算子语义真源（新）       │
                    │  mem_layout.py   步幅/对齐真源（新）      │
                    └──────────────────────────────────────┘
                         ▲ 只被 import，不 import 上层
     ┌───────────────────┼───────────────────┬──────────────┐
     │                   │                   │              │
  graph/ 各 pass    gml_bridge/      genesim_bridge/   orchestrator/
  （写四维）         （读四维）          （读四维）        （读排布）
     │                   │                   │              │
     └──── node.meta ────┴───────────────────┴──────────────┘
                （四维载荷的挂载点，键名由契约登记）
                         │
                    opcompiler_bridge/
                    （四维 → pimir 文本 ⇄ 回传）
                         │
                    FlagTree / PIMMLIR
```

#### 2.2.3 六个改造动作的依赖顺序

各动作之间有硬依赖，必须按序落地：

```
  ① P0-1 契约收口 ────────┐
     （键登记表建立）        │
                          ├──→ ④ P0-5 三个 bridge 去旁路
  ② P0-2 dtype 载体 ──────┤      （依赖 ①②③ 的载体就位）
     （PIMTensorSpec 补字段）│
                          │
  ③ P0-3 算子语义真源 ─────┘
                          
  ⑤ P0-4 排布字段 ──→ ⑥ P0-6 编排器改消费
     （TensorShardDetail 补）   （依赖 ⑤ 的字段就位）

  ⑦ P1-1 pimir 四维写入 ──→ ⑧ P1-2 回传推广
     （依赖 ①②③⑤ 全部就位）      （依赖 ⑦）

  ⑨ P1-3 校验 ── 贯穿 ①~⑧，每个载体就位即补其反例测试
```

**关键路径**是 ①→②→③→④，其中 ① 是全部后续动作的前置。⑤→⑥ 与 ①→④ 可并行。

## 三、总体设计

本章回答三个问题：中间表示**如何设计**才能覆盖算子语义 / 数据类型 / Placement / Memory Layout 四维；如何与**当前 pass** 交互；图层统一表示与 **PIMMLIR** 两级如何对应。

####  载体总览

```
node.meta（15 个键，契约见 contracts/unified_ir.py）
  │
  ├── device ──────────────────┐
  ├── part_id ────────────────  ├─→ 【Placement】算子在哪执行
  │                            │
  ├── spec: PIMTensorSpec ─────┤
  │     ├── device/placement/  ┘
  │     │   residency/pinned_dpu_id
  │     │
  │     ├── dtype: str ────────┐
  │     ├── quant: QuantLayout ┴─→ 【数据类型】张量是什么类型、怎么量化
  │     │
  │     └── shard_map: {dpu_id: TensorShardDetail}
  │           ├── shard_dim/start_idx/end_idx/local_shape ─┐
  │           ├── mram_offset ───────────────────────────  ├─→ 【Memory Layout】
  │           ├── elem_strides ──────────────────────────  │    切分+地址+排布+对齐
  │           └── align_bytes ───────────────────────────┘
  │
  ├── redistribute: list[RedistributeEdge] ─→ 【Placement】跨 DPU 通信
  │
  ├── fused_tail: FusedTail ───────┐
  ├── pim_rope: RopeMatch ─────────┤
  ├── pim_rms_norm: RmsNormFusion ─┤
  ├── pim_attention_scale ─────────┼─→ 【算子语义】融合后的真实语义
  ├── pim_absorbed: bool ──────────┤
  ├── pim_kv_cache_dma: KvDmaSpec ─┤
  ├── pim_split: SplitSpec ────────┤
  ├── pim_head_role: str ──────────┤
  ├── pim_head_index: int ─────────┘
  │
  ├── pim_dynamic_scaling: DynamicScalingSpec ─→ 【算子语义 + 数据类型】跨两维
  │
  └── val: FakeTensor ─→ 【基础设施】torch.export 产物，dtype 的派生来源
```

**一个关键的结构决策**：数据类型与 Memory Layout **都挂在 `spec` 内部**，不另开顶层键。理由是它们与 Placement 同源——`shard_map` 既是切分（Placement）也是本地形状与地址（Memory Layout），拆成两个键会让同一个 `TensorShardDetail` 被两处引用，必然分裂。

#### 3.2.2 四条跨维依赖（实测）

**依赖一：数据类型 → Memory Layout（字节数与地址）**

```
spec.dtype ──dtype_bytes()──→ itemsize ──bytes_of()──→ 字节数 ──align_up()──→ mram_offset
```

实测链条 `memory/mem_planner.py:88-91`：

```python
            spec.shard_map[dpu_id] = replace(spec.shard_map[dpu_id], mram_offset=off)
        itemsize = nodes[0].meta["val"].element_size()      # ← 本轮改为查 spec.dtype
        off += align_up(bytes_of(first.local_shape, itemsize), align)
```

**归属**：dtype 是真源，字节数与偏移是派生。所以**不新增「字节数」字段**（原则一）。

**依赖二：Placement → Memory Layout（本地形状）**

实测 `graph/spec_prop.py:157-175` 的 `_shard_map`：切分维与 DPU 数共同决定 `local_shape`：

```python
        width = length // len(dpu_ids)
        return {dpu_id: TensorShardDetail(
                    shard_dim=dim, start_idx=i * width, end_idx=(i + 1) * width,
                    local_shape=shape[:dim] + (width,) + shape[dim + 1:])
                for i, dpu_id in enumerate(dpu_ids)}
```

**归属**：`placement` 是真源，`local_shape` 是派生。校验守住一致性——`PIMTensorSpec.validate():68-72` 已断言 `detail.shard_dim == placement.dim`。

**依赖三：算子语义 → Placement（切分合法性）**

实测 `graph/spec_prop.py` 的规则表：不同算子对输入布局有硬性要求。`:229` linear 要求 `x` 必须 `REPLICATE`、weight 按策略切；`:267,269,302,360` 逐元素与视图族一律要求 `REPLICATE`。不满足就发 redistribute 边。

**归属**：算子语义是约束的**来源**，Placement 是被约束方。这条依赖不存字段，而是体现为 `propagate_specs` 的推导规则。

**依赖四：算子语义 → 数据类型（但比想象的弱）**

这一条我原以为是「量化改变 dtype」，实测发现**不是**。`graph/quant_pass.py:163-172` 的注释写得很清楚：

```python
        # 用 `alias` 当载体：它是恒等操作，所以图的数值完全不变
        # —— 真正的量化发生在硬件上，编译期只需要一个占位节点承载那些字段。
        # 换句话说 DQ 在 fx 图里是 no-op，在 GML 里是 4 相流水线。
        dq = gm.graph.call_function(torch.ops.aten.alias.default, (source,))
    dq.meta["val"] = source.meta.get("val")        # 继承源节点，fp16 仍是 fp16
```

**所以 DQ 节点的 `val` 继承源节点是正确的**，不是 bug。

**这给出了「数据类型」维度的精确口径**，必须写进契约注释否则必然误解：

| 层面 | 表达什么 | 载体 | llama2 W4A8 里 DQ 节点的取值 |
| --- | --- | --- | --- |
| FX 图数值类型 | 图上这个张量的元素类型 | `spec.dtype` | `float16`（alias 是恒等） |
| 硬件量化语义 | 硬件上这一步怎么定点化 | `spec.quant` + `pim_dynamic_scaling` | per_group/128/末轴 |
| GML 缓冲类型 | 落盘缓冲的元素类型 | GML `output_buffer_dtype`，由 `_stamp_dtypes` 按 GML 语义补 | `int8` |

#### 3.2.3 真源与派生的完整划分

| 事实 | 真源 | 派生方式 | 为何不存第二份 |
| --- | --- | --- | --- |
| 元素类型 | `node.meta["val"].dtype` | `_dtype_of()` 派生到 `spec.dtype` 一次 | 跨维校验守住两者一致（§4.9.2 跨维一） |
| 元素字节宽度 | `contracts/dtypes.py::_DTYPE_BYTES` | `dtype_bytes(spec.dtype)` | 存了就会与 dtype 分裂 |
| 本地形状 | `placement` + `shape` + `dpu_ids` | `_shard_map()` 算出 | `validate()` 断言与 placement 一致 |
| 分片字节数 | 上面三者 | `bytes_of(local_shape, itemsize, elem_strides)` | 同上 |
| MRAM 偏移 | 内存规划算法 | `replace(detail, mram_offset=off)` 回填 | 它是规划结果，只能存 |
| 算子 GML 类型 | `contracts/op_semantics.py::OP_SEMANTICS` | `aten_to_gml()` / `role_to_gml()` | 四份清单曾各自硬编码 |
| 融合后语义 | 各融合 pass 的判定 | 写 `fused_tail` 等键 | 它是 pass 决策，只能存 |
| 排布步幅（L2/DDR） | `contracts/mem_layout.py::stride_z` | 编排器调用 | 曾在编排器本地实现 |

**判读规则**：凡「算法结果」必须存（内存偏移、融合决策）；凡「函数值」一律派生（字节数、本地形状、GML 类型）。

### 3.3 pass 交互协议

#### 3.3.1 现有 pass 的统一形态（实测）

八个 pass 的签名高度一致——**原地改图 + 返回报告**：

| pass | 签名 | 返回 |
| --- | --- | --- |
| `partition_graph` | `(gm) -> list[Partition]` | DPU 直连子图 |
| `propagate_specs` | `(gm, strategy) -> list[RedistributeEdge]` | 重分布边 |
| `fuse_graph` | `(gm) -> int` | 折叠数 |
| `fuse_for_pim` | `(gm) -> FusionReport` | 融合报告 |
| `fuse_rope` | `(gm) -> RopeReport` | RoPE 报告 |
| `split_attention_heads` | `(gm) -> int` | 头数 |
| `insert_kv_dma_and_split` | `(gm) -> KvDmaReport` | DMA 报告 |
| `insert_dynamic_scaling` | `(gm) -> QuantReport` | 量化报告 |

**这个形态本身是好的，本设计不改它。** 交互协议要补的只有两件事：**前置条件可执行**、**出口状态可查**。

#### 3.3.2 问题：顺序依赖靠注释维系

实测前置条件的执行力**1 有 3 无**：

| 位置 | 状态 |
| --- | --- |
| `graph/spec_prop.py:551-553` | **有真断言**：`if DEVICE_META_KEY not in node.meta: raise ValueError(...请先运行 graph.partition.partition_graph)`；`:553` 的 `pop(SPEC_META_KEY, None)` 还给了幂等保护 |
| `opcompiler_bridge/oplevel_emitter.py:339` | 只有注释：「要求 `gm` 已经跑过 export_graph 的那串 pass」 |
| `gml_bridge/from_fx.py:957` | **注释与实现有落差**：承诺「图里若还有独立激活节点…这里会直接抛」，但实际只有一句泛化的 `raise ValueError("图里没有可映射到 GML 的算子")` |
| `gml_bridge/export.py:128-136` | 只有注释：5 条顺序依赖的理由写得很清楚，但无一条可执行 |

`export.py:128-136` 的注释值得摘录，它记录了顺序错了会怎么坏：

```
4. split_attention_heads —— 把批量 attention 拆成逐头链。**必须在前三个
   之后**：它产出的逐头节点带角色标记，若先跑，前面的 pass 会把那些
   add / mul 当成普通逐元素算子去折。
5. insert_kv_dma_and_split / insert_dynamic_scaling —— 最后跑：
   插 DQ 的规则依赖逐头角色（matmul1 不插、matmul2 插），KV 写回的位置
   判据依赖拆头留下的头下标标记。
```

这些理由很清楚，但**机器读不到**。

#### 3.3.3 解决方案：阶段标记 + 前置断言

**阶段标记挂在 `GraphModule` 上，不是 `node.meta`**——它描述整张图的状态，不是单个节点的属性，也避免给 15 键登记表塞特殊条目。

```python
# contracts/unified_ir.py
STAGE_EXPORTED    = "exported"      # torch.export 出来，未标注
STAGE_PARTITIONED = "partitioned"   # partition_graph 跑过：device/part_id 齐
STAGE_SPECS       = "specs"         # propagate_specs 跑过：DPU 节点有 spec
STAGE_FUSED       = "fused"         # fuse_for_gml 跑过：融合语义键齐
STAGE_PLANNED     = "planned"       # mem_planner 跑过：mram_offset 已回填

GRAPH_STAGE_KEY = "pim_graph_stage"

# 阶段的偏序。注意 SPECS 与 FUSED 是**两条分叉**，不是先后关系
_STAGE_PREDECESSOR = {
    STAGE_EXPORTED:    None,
    STAGE_PARTITIONED: STAGE_EXPORTED,
    STAGE_SPECS:       STAGE_PARTITIONED,
    STAGE_FUSED:       STAGE_PARTITIONED,     # ← 与 SPECS 并列，互不要求
    STAGE_PLANNED:     STAGE_SPECS,
}

_STAGE_PRODUCER = {
    STAGE_PARTITIONED: "graph.partition.partition_graph",
    STAGE_SPECS:       "graph.spec_prop.propagate_specs",
    STAGE_FUSED:       "gml_bridge.export.fuse_for_gml",
    STAGE_PLANNED:     "memory.mem_planner.plan_memory",
}


def stages_of(gm) -> frozenset:
    """图已达成的全部阶段。

    返回集合而非单值 —— 一张图可以同时是 SPECS 与 FUSED（若两条 pass 都跑过），
    但两条产出路径各自只跑一条（实测），所以通常只含其中之一。
    """
    return frozenset(getattr(gm, "meta", {}).get(GRAPH_STAGE_KEY, ()) or ()) \
           | {STAGE_EXPORTED}


def mark_stage(gm, stage: str) -> None:
    """pass 在出口处标记。重复标记同一阶段允许（幂等）。"""
    if stage not in _STAGE_PREDECESSOR:
        raise ValueError(f"未知阶段 {stage!r}，允许 {sorted(_STAGE_PREDECESSOR)}")
    pred = _STAGE_PREDECESSOR[stage]
    if pred is not None and pred not in stages_of(gm):
        raise ValueError(
            f"不能标记 {stage}：前驱阶段 {pred} 未达成。"
            f"当前 {sorted(stages_of(gm))}")
    gm.meta = getattr(gm, "meta", {})
    gm.meta[GRAPH_STAGE_KEY] = tuple(stages_of(gm) | {stage})


def require_stage(gm, stage: str, *, who: str) -> None:
    """前置断言。把注释里的顺序依赖变成可执行契约的唯一入口。

    `who` 是调用方名字，出现在错误信息里 —— 照 spec_prop.py:552 的既有风格。
    """
    if stage not in stages_of(gm):
        raise ValueError(
            f"{who} 要求图已达到阶段 {stage}，当前 {sorted(stages_of(gm))}。"
            f"请先运行 {_STAGE_PRODUCER.get(stage, '对应 pass')}")
```

**为什么用集合而非线性序号**：这是原则三的直接落地。`STAGE_SPECS` 与 `STAGE_FUSED` 是两条分叉（执行路径走前者、GML 路径走后者），若用递增序号就会得出「FUSED 隐含 SPECS 已完成」的错误结论，而实测 GML 路径根本没有 spec。

#### 3.3.4 接入点（每处 1 行）

**出口标记**：

```python
# graph/partition.py::partition_graph 末尾
    mark_stage(gm, STAGE_PARTITIONED)
    return partitions

# graph/spec_prop.py::propagate_specs 末尾（连同出口校验）
    for node in nodes:
        validate_node_dimensions(node, stage=STAGE_SPECS)
    mark_stage(gm, STAGE_SPECS)
    return edges

# gml_bridge/export.py::fuse_for_gml 末尾
    mark_stage(gm, STAGE_FUSED)
    return report
```

**前置断言**（补上 §4.3.2 那 3 处只有注释的）：

```python
# opcompiler_bridge/oplevel_emitter.py::emit_oplevel_mlir 入口
    require_stage(gm, STAGE_FUSED, who="emit_oplevel_mlir")

# gml_bridge/from_fx.py::convert 入口
    require_stage(gm, STAGE_FUSED, who="gml_bridge.from_fx.convert")
    _assert_no_standalone_activation(gm)      # 补上注释承诺却未实现的检查

# memory/mem_planner.py::plan_memory 入口
    require_stage(gm, STAGE_SPECS, who="plan_memory")
```

补 `from_fx.py:957` 注释承诺却缺失的检查：

```python
def _assert_no_standalone_activation(gm) -> None:
    """注释（:957）承诺过这件事，但从来没实现。

    GML 没有独立激活节点的表达方式，漏折的激活会被静默丢掉 ——
    产物少一个算子而不报错，这比抛错难查得多。
    """
    stray = [n.name for n in gm.graph.nodes
             if n.op == "call_function" and n.target in ACTIVATIONS
             and not n.meta.get(ABSORBED_META_KEY)]
    if stray:
        raise ValueError(
            f"图里还有 {len(stray)} 个未折叠的独立激活节点：{stray[:5]}。"
            f"GML 无法表达，请先运行 graph.fuse.fuse_graph")
```

#### 3.3.5 `.meta.get()` 的两类用法，只改一类

实测 35 处 `.meta.get()`。**不一律改成 `[]`**，要分类：

| 类别 | 例子 | 处置 |
| --- | --- | --- |
| **合法缺失**：该维度在本阶段可以不存在 | `n.meta.get(ABSORBED_META_KEY)`（未被吸收的节点本就没这个键）；`node.meta.get("val")`（非张量节点没有） | **保留 `.get()`** |
| **缺陷**：前置 pass 没跑却静默继续 | `spec_prop.py:188` 的 `u.meta.get(DEVICE_META_KEY) == DEVICE_DPU`——partition 没跑时得到「无 DPU 消费者」而非报错 | 收紧为 `[]` |

`spec_prop.py:188` 这一处**实际已被 `:551` 的循环断言覆盖**（整图每个节点都查过 device），所以收紧它不是「加检查」，而是**删掉一个已经不需要的容忍**：

```python
# graph/spec_prop.py:188
- dpu_users = [u for u in node.users if u.meta.get(DEVICE_META_KEY) == DEVICE_DPU]
+ # :551 的入口断言已保证全图每个节点都有 device，这里不必再容忍缺失
+ dpu_users = [u for u in node.users if u.meta[DEVICE_META_KEY] == DEVICE_DPU]
```

#### 3.3.6 新增 pass 如何接入（三步）

这是「统一 IR 可依赖」的落脚点——写新 pass 的人只需面对这三步：

```python
from contracts.unified_ir import (
    STAGE_SPECS, require_stage, mark_stage, spec_of, meta_keys_of, DIM_MEM_LAYOUT)

def my_analysis_pass(gm) -> MyReport:
    # ① 声明前置：缺了直接抛，不静默出错
    require_stage(gm, STAGE_SPECS, who="my_analysis_pass")

    # ② 通过契约 API 读四维，不碰 node.meta 的键名字符串
    for node in gm.graph.nodes:
        spec = spec_of(node)                  # 缺 spec 直接抛
        nbytes = bytes_of(spec.shard_map[0].local_shape,
                          dtype_bytes(spec.dtype),        # 数据类型维
                          spec.shard_map[0].elem_strides)  # Memory Layout 维
        ...

    # ③ 若改了图，在出口标记新阶段（只读 pass 不需要）
    return report
```

**对比今天**：写同样的 pass 要先读 5 个 pass 文件搞清 9 个私有键的含义、再自己判断 partition 跑过没有、再从 `node.meta["val"]` 问 PyTorch 拿 dtype。三步协议把这些都收进了契约。

#### 3.3.7 回传的时序约束（P1-2 的前提）

回传**不在 pass 链之后**，而是插在融合与序列化之间。实测 `scripts/export_gml.py:648-665`：

```
fuse_for_gml(graph)                              ① 融合 → STAGE_FUSED
     ↓
_run_opcompiler(graph, log) → phase_source       ② 采集回传（内部跑 triton-opt）
     ↓
serialize_gml(graph, fusion, phase_source=...)   ③ 消费回传
```

该文件注释说明了为何必须拆开：

```
融合与序列化拆开跑，中间插算子编译器：GML 的 `*_phase_N` 字段套数由它
决定，而它又要吃融合后的图。**不能调两次 export_graph** —— 那不幂等
（实测第二次会变成 206 节点而非 200）。
```

**推论：回传不能写回 `node.meta`。** 因为②发生在图已达 `STAGE_FUSED` 终态之后，二次修改会破坏「融合结果即最终语义」的不变式，且 `export_graph` 本身不幂等。所以回传走旁路对象（`PhaseSource`，§4.8.3），不碰图。

### 3.4 两级表示的对应关系

#### 3.4.1 分层原则：按内存层级划边界

图层统一 IR 与 PIMMLIR 不是「同一件事的两种写法」，而是**各管一段内存层次**。这条边界（§2.1.4 四条实测证据）决定了什么该下发、什么不该。

| 内存层级 | 负责组件 | 排布信息流向 | 经 PIMMLIR |
| --- | --- | --- | --- |
| WRAM / MRAM | 图编译器 + 算子编译器 | 统一 IR → pimir | **经过** |
| L1 / L2 | 编排器 | GML → 编排器 → 层参数 | **不经过** |
| DDR | 编排器 | 同上 | **不经过** |

**实现纪律**：`TensorShardDetail.elem_strides` 是 MRAM 级排布，经 `contracts/mlir_layout.py` 写进 `#pim.placement` 的 `order`。下发的是**维序**而不是步幅数值——步幅数值仍由 PIMMLIR 侧的指针分析证明，所以这不改 ODS 的表达能力，只是给既有的 `order` 一个真实来源。

#### 3.4.2 四维的两级对应表

| 维度 | 图层统一 IR | PIMMLIR | 传递方式 |
| --- | --- | --- | --- |
| 算子语义 | `node.target` + 9 个融合语义键 + `contracts/op_semantics.py` | 37 算子 + `#pim.contraction` / `#pim.act_spec` / `#pim.phase_spec` / `#pim.datapath` | 发射 op 与属性（`oplevel_emitter.py` 各 `_emit_*`） |
| 数据类型 | `spec.dtype`（FX 数值类型）+ `spec.quant`（量化布局） | 张量元素类型 + `#pim.quant_spec` + `#pim.weight_binding.elemBits` | `_tensor()` 的 dtype 串 + 量化属性 |
| Placement | `spec.device` / `spec.placement` / `spec.shard_map` 的 DPU 集 | 4 内存空间 + 功能单元枚举 + `pim.dpu_id` + **`#pim.tasklet_tiled.dpusPerDevice`** | 张量编码的 `dpusPerDevice` |
| Memory Layout | `mram_offset` / `align_bytes`（MRAM 级） | `!pim.memdesc<shape, elemType, memorySpace>` + `alignment` + 模块属性 `pim.dma-align` | 模块属性 + memdesc |
| Memory Layout（排布） | `elem_strides`（**MRAM 级**） | `#pim.placement` 的 `order` → 张量编码的 `order` | 决定 `-pim-explicit-dma` 的 `elem_stride` |

#### 3.4.3 PIMMLIR 侧无需改动的证据

四维逐项核实（§4.7.1~3.7.4 详列），结论是 **FlagTree 方言零改动**：

- **算子语义、数据类型、Placement**：载体完备（19 结构化属性、22 枚举、34 处 verifier）。
- **Memory Layout**：`!pim.memdesc` 的 ODS 明写「carries no layout encoding, because a scratchpad buffer on a PIM device is just a contiguous span of bytes」——**在它负责的 WRAM/MRAM 层级这是正确的**：`bytes_of` 是紧密乘积、该层级搜 `align16`/`+15` 零命中、非连续分片被 `comm/plan.py:62-79` 的 `_runs()` 展开为多段连续而非跨步视图。

三组 `triton-opt` 实验确认传递通道就绪：`dpusPerDevice = [2,1]` 可 parse、原样打印、穿过 `-pim-fuse-activation -pim-expand-phases` 存活；秩不符被 verifier 拒绝。

**`dpusPerDevice` 与 `pim.dpu_id` 的 ODS 注释都明写为图编译器预留**——接口是设计时就留好的，本轮补的是写入侧。

#### 3.4.4 走通示例：q_proj 的 Gemm（tp=2）

以 llama2-7b 第 0 层的 `q_proj` 为例，列出四维在每个阶段的取值。假设 `num_dpus=2`、按列切（`mode="col"` → `Shard(0)`）、W4A8。

**阶段 0：`torch.export` 之后**

```
node: linear (aten.linear.default)
  meta["val"]: FakeTensor(shape=(1, 1, 4096), dtype=torch.float16)
  weight 节点 meta["val"]: FakeTensor(shape=(4096, 4096), dtype=torch.float16)
  四维：全空
```

**阶段 1：`partition_graph` 之后**（`STAGE_PARTITIONED`）

```
  meta["device"]  = "dpu"
  meta["part_id"] = 0
  Placement 维：device 已定；其余三维仍空
```

**阶段 2：`propagate_specs` 之后**（`STAGE_SPECS`）

weight 节点（`_weight_spec:184`，本轮补 dtype/quant）：

```python
PIMTensorSpec(
    device="dpu",
    placement=Placement("Shard", dim=0),          # col 切 → 输出特征维
    residency="pinned",
    pinned_dpu_id=None,
    shard_map={
        0: TensorShardDetail(dpu_id=0, shard_dim=0, start_idx=0,    end_idx=2048,
                             local_shape=(2048, 4096), mram_offset=0,
                             elem_strides=(), align_bytes=0),
        1: TensorShardDetail(dpu_id=1, shard_dim=0, start_idx=2048, end_idx=4096,
                             local_shape=(2048, 4096), mram_offset=0,
                             elem_strides=(), align_bytes=0),
    },
    reduce_type=None,
    dtype="int4",                                  # 本轮新增（W4A8 权重）
    quant=QuantLayout("per_group", group_size=128, axis=-1),   # 本轮新增
)
```

四维此刻的取值：

| 维度 | 取值 | 来源 |
| --- | --- | --- |
| Placement | `Shard(0)`，DPU {0,1} | `strategy.match(q_proj) == "col"` |
| 数据类型 | `dtype="int4"`、`quant=per_group/128/末轴` | `_dtype_of()` 派生 + `WEIGHT_LAYOUT` |
| Memory Layout | `local_shape=(2048,4096)`、`elem_strides=()`（确认紧密） | `_shard_map()` 派生 |
| 算子语义 | `linear` → 待融合 | `node.target` |

**阶段 3：`mem_planner` 之后**（`STAGE_PLANNED`）

```
字节数 = bytes_of((2048, 4096), dtype_bytes("int4"), ()) 
       = 2048 * 4096 * 1 = 8,388,608      ← int4 不打包，一字节一值（实测）
mram_offset 回填 = align_up(8388608, 64) 之后的权重区偏移
  → shard_map[0] = replace(..., mram_offset=0)
  → shard_map[1] = replace(..., mram_offset=0)   # 各 DPU 独立地址空间
```

注意这里体现了**依赖一**（dtype → 字节数 → 地址）：`dtype="int4"` 决定 `itemsize=1`，进而决定偏移。若 dtype 错记成 `int8` 结果相同（都是 1 字节），但记成 `float16` 就会多算一倍。

**阶段 4：下发 pimir**（P1-1）

```mlir
module attributes {pim.target = "pim:v1", "pim.rtl-version" = "...",
                   "pim.num-dpus" = 2 : i32, "pim.num-tasklets" = 16 : i32,
                   "pim.dma-align" = 64 : i32} {
  tt.func @q_proj_gemm(
      %x: tensor<1x1x4096xf16>,
      %w: tensor<2048x4096xi8,
                 #pim.tasklet_tiled<{sizePerTasklet = [1, 1],
                                     taskletsPerDpu = [16, 1],
                                     dpusPerDevice = [2, 1],
                                     order = [1, 0]}>>) {
    ...
  }
}
```

三处对应关系：

| pimir 片段 | 来自统一 IR 的 |
| --- | --- |
| `dpusPerDevice = [2, 1]` | `placement.dim == 0` 且 `len(shard_map) == 2` → 第 0 维分到 2 台 DPU |
| `tensor<2048x4096xi8>` | `shard_map[k].local_shape` + `spec.dtype`（int4 按 i8 载体发，`elemBits=4` 由 `#pim.weight_binding` 表达） |
| `"pim.num-dpus" = 2` | `PIMHardwareConfig.num_dpus` |
| **`elem_strides` 经 `order` 下发** | 非行主序时写进 `#pim.placement` 的 `order`，A 路按它重算 `elem_stride`；行主序与不写逐字节相同 |

**`dpusPerDevice` 出现即证明写入成功**——printer 会省略全 1 默认值（`Dialect.cpp:114-116` 实测），所以文本里有它就意味着非默认。

**阶段 5：GML 产出**

```
node [
    id 12
    op_type "Gemm"
    weight_buffer "weight_buffer_12.bin"
    weight_sf "weight_sf_12.bin"
    weight_format "weight"
    input_buffer "input_buffer_12.bin"
    input_sf "input_sf_12.bin"
    input_data_extensions 1          ← int8 激活，有符号
    output_buffer_dtype "float16"    ← 由 _stamp_dtypes 按 GML 语义补（不上移）
    output_data_extension 3          ← 浮点
    nmu_mode "floating_point"
]
```

这里体现了 §4.2.2 **依赖四**的三层口径：`spec.dtype="int4"`（FX 图/权重存储）、`spec.quant`（硬件量化布局）、GML 的 `output_buffer_dtype="float16"`（缓冲落盘类型，由 `_stamp_dtypes` 算）——**三者不同且都对**。

#### 3.4.5 回传闭环

```
统一 IR ──①下发──→ pimir ──②算子编译器 pass 链──→ 展开后 IR
                                                      │
统一 IR ←──④消费───  PhaseSource ←──③解析──────────────┘
   │                （旁路对象，不写 node.meta）
   └─→ GML / numpy / GeneSim
```

②的真实产出是**融合决策**，实测样本过 pass 链后 op 从 6 降到 5、相位号 1 消失（§4.8.1）。③解析出 `{函数名: PhasePlan}` 与 `LayoutFeedback`。④遵循「回传优先、静态表兜底」——这是 `phase_value()` 已确立的口径（`phase_source.py:98` docstring）。

### 3.5 总体设计如何统摄九个功能点

| 功能点 | 由哪条原则/机制统摄 |
| --- | --- |
| P0-1 契约收口 | 四维载体总览（§4.2.1）定义了 15 键归属；查询 API 是执行点 |
| P0-2 dtype 载体 | 依赖一（dtype → 字节数）确立 dtype 为真源；依赖四确立三层口径 |
| P0-3 算子语义真源 | 原则一（真源唯一）；四份清单由 `OP_SEMANTICS` 派生 |
| P0-4 Memory 排布 | 原则四（跨维依赖显式化）把 `bytes_of` 的隐含假设变成 `elem_strides` |
| P0-5 唯一来源 | 原则三（阶段决定此刻该有什么）划清「唯一来源 ≠ 全读」 |
| P0-6 编排器消费 | §4.4.1 内存层级边界说明它消费的是**规则**而非数据 |
| P1-1 pimir 四维 | §4.4.2 对应表：`elem_strides` 经 `#pim.placement` 的 `order` 下发 |
| P1-2 回传生效 | §3.3.7 时序约束决定回传走旁路对象 |
| P1-3 四维可校验 | 原则二（空值语义）+ 真源/派生划分（§4.2.3）决定校验什么 |

### 3.6 仍未解决的问题

| # | 问题 | 现状 |
| --- | --- | --- |
| 1 | 两条产出路径的四维填充分叉（§2.1 原则三），本设计用阶段集合表达它，但**没有统一它** | 有意为之：统一会改变 GML 产物，违反逐字节不变。若将来要统一，需先解决「GML 路径引入 spec 是否改变字段」 |
| 2 | `part_id` 生产代码零读取 | 契约里显式标注，不删；待确认它是否该有生产用途 |
| 3 | `elem_strides` 恒为行主序 | `spec_prop` 已对每个分片填 `row_major_strides`，A 路按维序重算 `elem_stride`。取值恒为默认是因为当前没有非行主序的生产方，属已知限制，不是「无消费者」 |
| 4 | `_l2_in_size` / `_l2_out_size` 的实测特例（`bmm` 加 16、`dq_p4` 乘 2、`dq_p2` 的 Gn=1 例外） | 搬迁时整体保留，不简化公式。这些数字来自 422 层实测，无闭合推导 |
## 四、各功能点详细设计

### 4.1 P0-1：统一 IR 的契约收口

#### 4.1.1 问题一：15 个键只登记了 5 个，且第 15 个是缺陷

需求 §1.1 记载 14 个键。本轮逐文件核实后是 **15 个**，多出的那个是一处缺陷。

`graph/split_heads.py:176-177` 连续两行：

```python
                node.meta[ROLE_SPLIT] = True          # :176 ← 把「角色取值」当「键名」用
                node.meta[HEAD_ROLE_META_KEY] = ROLE_SPLIT   # :177 ← 正确写法
```

`ROLE_SPLIT` 定义在 `:54`，值是 `"split"`，它本是 `HEAD_ROLE_META_KEY` 的**取值之一**（同组还有 `ROLE_MATMUL_QK`/`ROLE_MASK`/`ROLE_SOFTMAX`/`ROLE_DQ`/`ROLE_MATMUL_PV`/`ROLE_CONCAT`，`:49-55`）。`:176` 误把它当键名，图上因此多出一个名为 `"split"` 的键，实测**全仓无任何读者**（含测试）。

**解决方案：删除 `:176` 这一行。** `:177` 已经正确表达了同一语义，删除后图上少一个无意义的键。这比裸字符串那处更隐蔽——它不是拼错字符串，而是混淆了「键」与「值」两个概念，靠人眼 review 很难发现。

#### 4.1.2 问题二：已发生的契约腐化

`graph/kv_dma_pass.py:122` 与 `:151` 用字面量读跨 pass 传递的键：

```python
        if n.meta.get("pim_head_role") != ROLE_SPLIT:      # :122
        if node.meta.get("pim_head_role") != ROLE_SPLIT:   # :151
```

而该键的常量定义在 `graph/split_heads.py:45`，且该文件**已经被 import**（`kv_dma_pass.py:37` `from graph.split_heads import HEAD_INDEX_META_KEY, ROLE_SPLIT`）——即常量就在手边却没用。改名时这两处不会有任何报错。

**解决方案**：两行各改一处引用。

```python
# graph/kv_dma_pass.py:37
- from graph.split_heads import HEAD_INDEX_META_KEY, ROLE_SPLIT
+ from graph.split_heads import HEAD_INDEX_META_KEY, HEAD_ROLE_META_KEY, ROLE_SPLIT

# :122 与 :151
- if n.meta.get("pim_head_role") != ROLE_SPLIT:
+ if n.meta.get(HEAD_ROLE_META_KEY) != ROLE_SPLIT:
```

#### 4.1.3 问题三：`graph/strategy.py` 不在改造范围（对需求的修正）

需求 §1.1 说「9 个键散在 5 个 pass 文件」。实测 `graph/strategy.py` 的 `.meta` 命中 **0 处**——它是纯 `ShardStrategy` 切分数学，不接触 FX 图。那 5 个文件是 `quant_pass.py` / `fuse_pim.py` / `kv_dma_pass.py` / `fuse_rope.py` / `split_heads.py`。本设计据此把 `strategy.py` 排除。

另一处需求未提及的事实：`part_id`（`contracts/graph_meta.py:4`）写入于 `partition.py:149`，但**生产代码零读取**，仅 `tests/test_partition.py` 读它（5 处）。它不是死字段（图拆分的分组结果需要它做断言），但契约登记时必须显式标注，否则后续会误判为死字段而删除。

#### 4.1.4 核心设计难点：契约不能 import 载荷类型

**矛盾**：契约要登记「每个键挂什么类型的载荷」，但 6 个载荷 dataclass 全住在 `graph/`，而 `contracts/` 不能 import `graph/`（否则循环）。

实测证据：`contracts/graph_meta.py:7` 只能用注释描述（`# ...值是 FusedTail`）；`graph/fuse_pim.py:230` 为规避循环把 `from graph.fuse import FusedTail` 写在函数体内。

载荷的完整字段（下移时要照抄）：

| 载荷 | 位置 | 字段 | 持 `Node` 引用 |
| --- | --- | --- | --- |
| `FusedTail` | `fuse.py:33` | `activation: str` / `pool: str \| None` / `nodes: list[Node]` | 1 |
| `RopeMatch` | `fuse_rope.py:43` | `source: Node` / `cos: Node` / `sin: Node` / `absorbed: list[Node]` | 4 |
| `RmsNormFusion` | `fuse_pim.py:66` | `epsilon: float` / `weight_node: Node \| None` / `eaten: list[Node]` | 5 |
| `KvDmaSpec` | `kv_dma_pass.py:45` | `is_key: bool` / `numel: int` | 0 |
| `SplitSpec` | `kv_dma_pass.py:57` | `heads: int` / `numel: int` | 0 |
| `DynamicScalingSpec` | `quant_pass.py:65` | `group_size: int` / `numel: int` / `is_attention_scores: bool` / `head_index: int \| None` | 0 |

共 **10 个字段持有 FX `Node` 引用**。这曾是 A1 方案（下移到 `contracts/`）的唯一顾虑——地基层是否要因此依赖 `torch.fx`。

**本轮实测排除了这个顾虑。** 用 `TYPE_CHECKING` 配合 `from __future__ import annotations`，类型引用只在检查期生效，运行时零导入：

```python
# /tmp/a1_probe.py（实测脚本）
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.fx import Node          # 仅类型检查期

@dataclass
class FusedTail:
    activation: str
    pool: str | None
    nodes: list[Node] = field(default_factory=list)

t = FusedTail("Silu", None, [])
import sys
print("torch 是否被导入:", "torch" in sys.modules)
```

实测输出：

```
构造成功: FusedTail(activation='Silu', pool=None, nodes=[])
torch 是否被导入: False
```

所以 **A1 的依赖成本确实为零**——既不引入运行时依赖，也不产生模块循环。`contracts/fusion_contract.py:19` 已有 `from __future__ import annotations`，是同一写法的先例。

三个子方案对比：

| 子方案 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **A1（已决策）** | 载荷 dataclass 下移到 `contracts/ir_payloads.py`，`graph/` 各 pass 改为 import | 契约完整（键 + 类型同处，可静态校验）；顺带消掉 `fuse_pim.py:230` 的函数内 import；**依赖成本实测为零** | 要动 5 个 pass 的 import |
| A2 | 契约只登记键名，载荷类型留 `graph/` | 改动最小 | 契约不完整——键与类型分离，仍需读两处；无法静态校验载荷类型 |
| A3 | `contracts/` 定义 Protocol，`graph/` 的 dataclass 隐式满足 | 无需移动代码 | 违反「不预造抽象」；Protocol 对 dataclass 字段校验能力有限 |

#### 4.1.5 解决方案：登记表的具体结构

新增 `contracts/unified_ir.py`：

```python
"""图编译阶段统一 IR 的键登记表。

这是四维信息挂载点的**唯一真源**。新增 pass 不得私建键 ——
tests/test_unified_ir_contract.py 的集合相等断言是这条规约的执行点。

contracts/graph_meta.py 保留为兼容 re-export（12 个既有 import 点不动）。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # 仅检查期，运行时零导入（实测见 §4.1.4）
    from torch.fx import Node

# ── 四维 + 基础设施 ────────────────────────────────────────────────
DIM_OP_SEMANTICS = "op_semantics"
DIM_DTYPE        = "dtype"
DIM_PLACEMENT    = "placement"
DIM_MEM_LAYOUT   = "mem_layout"
DIM_INFRA        = "infra"              # 不属四维的基础设施键

DIMENSIONS = (DIM_OP_SEMANTICS, DIM_DTYPE, DIM_PLACEMENT, DIM_MEM_LAYOUT)


@dataclass(frozen=True)
class MetaKeySpec:
    """一个 node.meta 键的完整契约。"""
    key: str                            # 键名
    dimensions: tuple[str, ...]         # 所属维度，可跨维
    payload: str                        # 载荷类型名（字符串，避免循环 import）
    producer: str                       # 写入方 "模块:函数"
    consumers: tuple[str, ...]          # 读取方；空 = 仅测试读取（须显式声明）
    note: str = ""
```

`payload` 用**类型名字符串**而非类型对象：登记表要能被静态 diff、且不必在导入时解析类型。真正的类型定义在 `contracts/ir_payloads.py`，两者由测试断言绑定（§4.1.7）。

15 个键的登记（维度归属依据写在 `note`）：

```python
META_KEYS = (
    MetaKeySpec("device", (DIM_PLACEMENT,), "str",
                producer="graph.partition:partition_graph",
                consumers=("graph.spec_prop", "runtime.exec_plan_gen"),
                note="取值 DEVICE_DPU / DEVICE_HOST"),
    MetaKeySpec("part_id", (DIM_PLACEMENT,), "int",
                producer="graph.partition:partition_graph",
                consumers=(),
                note="生产代码零读取，仅 tests/test_partition.py 断言。"
                     "显式声明以免被误判为死字段而删除"),
    MetaKeySpec("spec", (DIM_PLACEMENT, DIM_MEM_LAYOUT, DIM_DTYPE), "PIMTensorSpec",
                producer="graph.spec_prop:propagate_specs",
                consumers=("memory.mem_planner", "memory.kv_layout",
                           "runtime.exec_plan_gen", "runtime.compile",
                           "comm.plan", "genesim_bridge.placement_export"),
                note="跨三维：切分属 Placement、mram_offset/elem_strides 属 "
                     "Memory Layout、dtype/quant 属数据类型（本轮新增）"),
    MetaKeySpec("redistribute", (DIM_PLACEMENT,), "list[RedistributeEdge]",
                producer="graph.spec_prop:propagate_specs",
                consumers=("comm.plan", "memory.mem_planner", "runtime.exec_plan_gen")),
    MetaKeySpec("fused_tail", (DIM_OP_SEMANTICS,), "FusedTail",
                producer="graph.fuse:fuse_graph | graph.fuse_pim:fuse_for_pim",
                consumers=("gml_bridge.from_fx",),
                note="融合后 node.target 不变而语义已变，靠此键表达"),
    MetaKeySpec("pim_rope", (DIM_OP_SEMANTICS,), "RopeMatch",
                producer="graph.fuse_rope:fuse_rope",
                consumers=("gml_bridge.from_fx", "opcompiler_bridge.oplevel_emitter")),
    MetaKeySpec("pim_dynamic_scaling", (DIM_OP_SEMANTICS, DIM_DTYPE),
                "DynamicScalingSpec",
                producer="graph.quant_pass:insert_dynamic_scaling",
                consumers=("gml_bridge.from_fx", "opcompiler_bridge.oplevel_emitter"),
                note="跨两维：量化既是算子也是类型变换"),
    # ... 其余 8 个键同构声明；"split" 键本轮删除（§4.1.1）
)
```

#### 4.1.6 查询 API：只提供实际需要的三个

按「不预造抽象」，不做通用框架：

```python
_BY_KEY = {spec.key: spec for spec in META_KEYS}


def meta_keys_of(dimension: str) -> tuple[str, ...]:
    """某一维度登记的全部键名。给分析 pass 用。"""
    if dimension not in DIMENSIONS and dimension != DIM_INFRA:
        raise ValueError(f"未知维度 {dimension!r}，允许 {DIMENSIONS}")
    return tuple(s.key for s in META_KEYS if dimension in s.dimensions)


def dimensions_of(key: str) -> tuple[str, ...]:
    """键属于哪些维度。未登记的键直接抛错 —— 这是契约收口的执行点。"""
    spec = _BY_KEY.get(key)
    if spec is None:
        raise ValueError(
            f"未登记的 meta 键 {key!r}。新增键必须先登记进 "
            f"contracts/unified_ir.py::META_KEYS，当前已登记 {sorted(_BY_KEY)}")
    return spec.dimensions


def spec_of(node) -> PIMTensorSpec:
    """节点的张量规格。缺失直接抛错，不返回 None。

    这样调用方不必到处写 `if spec is None` —— 缺 spec 说明 propagate_specs
    没跑或该节点是 host 节点，两种情况都该由调用方在更早处理。
    """
    spec = node.meta.get(SPEC_META_KEY)
    if spec is None:
        raise ValueError(
            f"节点 {node.name} 没有 spec。请先运行 "
            f"graph.spec_prop.propagate_specs，或先判断 device 是否为 host")
    return spec
```

三个函数都遵循「契约不满足直接抛错」（需求 §3.3）。这样新增 pass 若私建键，第一次调 `dimensions_of` 就失败。

#### 4.1.7 验收判据落地

需求 P0-1 要求「契约声明的键集合恰好等于全仓在用集合」。沿用本仓既有的源码正则扫描 + 集合相等断言范式（`tests/test_runtime_compiled_coverage.py:155-171`）：

```python
def test_registered_keys_exactly_match_keys_in_use() -> None:
    """契约登记的键集合 == 全仓实际在用的键集合。

    多一个 = 契约里有没人用的键（该删或该标注）。
    少一个 = 有 pass 私建了键，契约漏登记 —— 这是 P0-1 要防的主要腐化。
    """
    import re
    from pathlib import Path
    from contracts.unified_ir import META_KEYS

    root = Path(__file__).parent.parent
    # 三种写法都要扫：常量、字面量、.get()
    pat = re.compile(r'\.meta(?:\[|\.get\()\s*(?:"([^"]+)"|([A-Z_]+))')
    in_use, consts = set(), {}
    for d in ("graph", "gml_bridge", "genesim_bridge", "opcompiler_bridge",
              "memory", "runtime", "comm", "contracts"):
        for py in (root / d).rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            # 先收集该文件里 XXX_META_KEY = "yyy" 的定义
            consts.update(re.findall(r'^([A-Z_]*META_KEY[A-Z_]*)\s*=\s*"([^"]+)"',
                                     text, re.M))
            for lit, const in pat.findall(text):
                in_use.add(lit or consts.get(const, const))

    registered = {s.key for s in META_KEYS}
    assert in_use == registered, (
        f"契约与实际不符。\n仅在用未登记：{sorted(in_use - registered)}"
        f"\n仅登记未在用：{sorted(registered - in_use)}")


def test_no_four_dimension_key_is_read_by_bare_string() -> None:
    """四维键不得用裸字符串读取（val 等基础设施键豁免）。

    反例就在 graph/kv_dma_pass.py:122,151 —— 本轮修掉的那两处。
    """


def test_every_payload_type_actually_exists() -> None:
    """登记表声明的 payload 类型名必须在 contracts/ir_payloads.py 里有定义。

    这把「键」与「类型」两半绑起来 —— A1 方案的价值正在于此，
    A2 做不到这条。
    """
    import contracts.ir_payloads as payloads
    for spec in META_KEYS:
        base = spec.payload.removeprefix("list[").removesuffix("]")
        if base in ("str", "int", "bool", "float"):
            continue
        assert hasattr(payloads, base) or base == "PIMTensorSpec", \
            f"{spec.key} 声明的载荷类型 {base} 不存在"
```

#### 4.1.8 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/unified_ir.py` | **新增**：`MetaKeySpec` + 维度常量 + 15 键登记 + 3 个查询函数 | ~180 行 |
| `contracts/ir_payloads.py` | **新增**：6 个载荷 dataclass 下移（用 `TYPE_CHECKING` 引用 `Node`） | ~90 行 |
| `contracts/graph_meta.py` | 改为从 `unified_ir` re-export，12 个既有 import 点不动 | 11 → ~15 行 |
| `graph/fuse.py` `fuse_pim.py` `fuse_rope.py` `kv_dma_pass.py` `quant_pass.py` | 载荷定义移走改 import；`fuse_pim.py:230` 函数内 import 提回顶部 | 5 文件，各减 10~20 行 |
| `graph/kv_dma_pass.py:122,151` | 裸字符串改常量 | 2 行 |
| `graph/split_heads.py:176` | **删除**误用 `ROLE_SPLIT` 当键的那行 | -1 行 |
| `tests/test_unified_ir_contract.py` | **新增**：键集合相等 + 裸字符串禁令 + 载荷类型存在性 | ~140 行 |

### 4.2 P0-2：数据类型维度补载体并收敛真源

#### 4.2.1 问题一：`_stamp_dtypes` 不能「上移」（Q1 已决策）

需求 P0-2 原文要求「`_stamp_dtypes` 的沿边传播逻辑上移到统一 IR」。实测三条证据表明这条不可行：

| # | 证据 | 位置 |
| --- | --- | --- |
| 1 | 它作用在 **GML 节点**，不是 FX 节点 | 签名 `_stamp_dtypes(nodes: list[Node])`，`Node` 来自 `from_fx.py:40` `from gml_bridge.writer import Edge, Node`；FX 节点在同文件叫 `FxNode`（`:17`） |
| 2 | 调用点在 FX→GML 转换**之后** | 唯一调用点 `from_fx.py:2173`，此时 `gml_nodes` 已构造完毕 |
| 3 | 依赖的判据全是 GML 侧概念 | `op_type` / `output_buffer_dtype` / `kantor_mode` 在 `graph/` 与 `contracts/pim_tensor_spec.py` 命中**全为 0** |

它的算法判据也全部基于 GML 语义：

```python
# from_fx.py:538-549
_OUT_INT8_OPS = frozenset({"DynamicScaling", "Llama2Activation",
                           "Llama2ActivationDQ", "KV_Cache_DMA", "Split"})
_OUT_FP16_OPS = frozenset({"MatMul", "Softmax", "Mask", "EltwiseAdd",
                           "EltwiseMul", "RMSNorm_vpu"})
_IN_INT8_OPS  = frozenset({"Gemm", "MatMul"})
_NO_SCALE_OPS = frozenset({"Split", "Transpose", "Reshape", "Concat"})
```

这些名字（`Llama2ActivationDQ`、`RMSNorm_vpu`）是 GML 的 `op_type`，FX 图上不存在。

**决策（Q1）：不上移。** 改为在其上游补一个图阶段的 dtype 载体，`_stamp_dtypes` **零改动**。两者分工：

```
FX 图（node.meta["val"].dtype 是 torch.export 给的真值）
   │
   ├─→ ① 图阶段载体（本轮新增）：PIMTensorSpec.dtype
   │      派生一次，之后图阶段一律查这里
   │      消费：memory/（字节宽度）、runtime/（dtype 串）、opcompiler_bridge/
   │
   └─→ FX→GML 转换
          └─→ ② _stamp_dtypes（保留不动）
                 按 GML 语义补齐节点级 dtype 字段
                 依赖：GML op_type / kantor_mode / 缓冲命名
```

不动它还有一层价值：`_stamp_dtypes` 的 docstring 记录了两条踩过的坑——「布局算子的 dtype 是沿边传播的，不是按 op_type 固定」（按 op_type 写死会让一半节点位宽错一倍且不报错）、「`nodes` 并非拓扑序，按顺序传播会静默退回默认 fp16」。这些是资产，搬动它就要重新验证这两条。

#### 4.2.2 问题二：`PIMTensorSpec` 无 dtype 字段

实测 `contracts/pim_tensor_spec.py:45-72` 的字段是 device / placement / residency / pinned_dpu_id / shard_map / reduce_type——**无 dtype**。唯一带类型的是 `RedistributeEdge.dtype`（`:92`），且是裸字符串。

**解决方案**：末尾加两个带默认值的字段。

```python
# contracts/pim_tensor_spec.py
@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None
    # 本轮新增，**必须在末尾**：graph/spec_prop.py 的三个构造点
    # （_host_spec:128 / _dpu_spec:135 / _weight_spec:184）都用位置参数，
    # 插在中间会静默错位。
    dtype: str = ""                        # 元素类型名，见 contracts/dtypes.py
    quant: QuantLayout | None = None       # 量化布局；None = 未量化
```

`dtype` 用 `str` 而非 `torch.dtype`，沿用仓内既有口径（`RedistributeEdge.dtype:92` 已是 str、`exec_plan_gen.py:323` 也是 `str(...).removeprefix("torch.")`），且避免把 torch 运行时对象带进契约层。

`quant` 复用**已有的** `QuantLayout`（`contracts/gml_quant.py:71`），不新造。它的三档 `per_tensor`/`per_channel`/`per_group` 与 PIMMLIR 的 `#pim.quant_spec` 一一对应（该文件注释已说明），正好是 P1-1 两侧口径统一所需。

**为什么 `quant` 是真正的新能力**：实测 `ACTIVATION_LAYOUT` / `WEIGHT_LAYOUT` 在 GML 侧被用了 7 处（`from_fx.py:1410,1439,1886,1967,2096,2114,2122`），但**全部是硬编码常量传入**：

```python
                spec=ACTIVATION_LAYOUT))       # 7 处都是这样
```

即「这个张量是什么量化布局」今天无法按张量查询，只能按调用点写死。`spec.quant` 让它变成可查属性。

#### 4.2.3 问题三：六处类型定义（Q2 已决策）

需求说「六处收敛为一处」。实测六处**性质不同**，不能简单归一：

| # | 位置 | 内容 | 本轮动作 | 理由 |
| --- | --- | --- | --- | --- |
| 1 | `gml_quant.py:58` `DTYPES` | 按缓冲区**类**的允许集合：activation `(int8,float16)`、weight `(int4,int8)`、bias `(int32,float32)`、scale `(float16,float32)`、output `(int8,float16,int16)` | **保留** | 回答「weight 缓冲允许哪些类型」，不是「某张量是什么类型」 |
| 2 | `gml_quant.py:32-55` | `INT4_BYTES_PER_VALUE=1`、`INT4_MIN/MAX=-8/7`、`INT8_MIN/MAX`、`INT4_SCALE_DIVISOR=8`、`INT8_SCALE_DIVISOR=128`、`WEIGHT_GROUP_SIZE=128` | **保留** | 量化算法参数。注释记录了实测依据（分母取 8 而非 7 是因为 15000 组里 52% 含 q=-8），不属类型枚举 |
| 3 | `gml_hw_table.py:237` `DATA_EXTENSION` | `{int8: 1, float16: 3}` | **保留 + 加交叉校验** | 这是**通道宽度编码**（1 有符号/3 浮点），不是类型集合 |
| 4 | `from_fx.py:561` `_BUFFER_DTYPES` | `{int8, int16, float16, float32}` | **改为引用真源** | 这是唯一真正的「允许的元素类型集合」 |
| 5 | `layer_fields.py:20-21` | `DT_INT8/DT_FP16/DT_FP32=0/1/3`、`EXT_SIGNED/EXT_FLOAT=1/3` | **保留 + 加交叉校验** | 编排器侧独立编码表，**编号口径与第 3 处不同**（`DT_FP16=1` vs `DATA_EXTENSION["int8"]=1`） |
| 6 | `node.meta["val"].dtype` | PyTorch 真值 | **保留** | torch.export 产物，是 dtype 字段的派生来源，不是要取代的对象 |

**决策（Q2）：新增元素类型名真源，1 处改引用 + 2 处加交叉校验 + 3 处不动。**

```python
# contracts/dtypes.py（新增）
"""元素类型名的唯一真源。

本轮不扩充集合（需求 §2.3）：不引入 bf16、fp8、亚字节打包。

int4 记 1 字节是**实测事实**而非疏漏：weight_buffer 的字节数等于权重元素数，
值域严格落在 [-8,7]，高 4 位是符号扩展（见 contracts/gml_quant.py:30-32）。
PyTorch 侧这些张量的 dtype 实际是 int8，所以 element_size() 也返回 1 ——
两者一致，这是 §4.5.3 行为等价的前提。
"""

ELEMENT_DTYPES = frozenset({"int4", "int8", "int16", "int32", "float16", "float32"})

_DTYPE_BYTES = {"int4": 1, "int8": 1, "int16": 2,
                "int32": 4, "float16": 2, "float32": 4}


def validate_dtype(name: str) -> None:
    """不在集合内直接抛错（不写防御性兜底，见 CLAUDE.md）。"""
    if name not in ELEMENT_DTYPES:
        raise ValueError(f"未知元素类型 {name!r}，允许 {sorted(ELEMENT_DTYPES)}")


def dtype_bytes(name: str) -> int:
    """单个元素占几字节。取代散落各处的 .meta["val"].element_size()。"""
    validate_dtype(name)
    return _DTYPE_BYTES[name]
```

第 4 处改引用：

```python
# gml_bridge/from_fx.py:561
- _BUFFER_DTYPES = frozenset({"int8", "int16", "float16", "float32"})
+ # GML 缓冲能落盘的元素类型，是真源的子集（int4 不单独落 —— 它按 int8 存）。
+ _BUFFER_DTYPES = ELEMENT_DTYPES - {"int4"}
```

第 3、5 处加交叉校验（不改值，只断言键在真源内）：

```python
# tests/test_dtype_carrier.py
def test_encoding_tables_only_key_on_known_dtypes() -> None:
    """两张编码表的键必须是真源子集。

    它们的**值**是两套不同的编号（DT_FP16=1 vs DATA_EXTENSION["int8"]=1），
    刻意不归一；但**键**必须来自同一个类型集合，否则会出现
    「某处认识 bf16 而另一处不认识」的分裂。
    """
    from contracts.dtypes import ELEMENT_DTYPES
    from contracts.gml_hw_table import DATA_EXTENSION
    assert set(DATA_EXTENSION) <= ELEMENT_DTYPES
    assert {"int8", "float16", "float32"} <= ELEMENT_DTYPES   # layer_fields 的 DT_*
```

#### 4.2.4 dtype 的派生：唯一入口

`graph/spec_prop.py:543` 已经在做派生：

```python
        dtype=str(val.dtype).removeprefix("torch."),
```

但它写在 `RedistributeEdge` 上。抽成函数，供 spec 构造与边构造共用：

```python
# graph/spec_prop.py（新增）
def _dtype_of(node: Node) -> str:
    """节点输出的元素类型名。

    这是 dtype 从 PyTorch 进入统一 IR 的**唯一入口**（§4.5.4 的源码扫描
    测试把白名单锁死在本文件）。之后全链路查 spec.dtype，
    §4.9.2 的跨维校验守住两者不漂移。
    """
    val = node.meta.get("val")
    if val is None:
        return ""                      # 非张量节点
    return str(val.dtype).removeprefix("torch.")
```

三个 spec 构造点加关键字参数（现有调用点不传也能跑）：

```python
def _dpu_spec(placement, shape, dpu_ids, residency="transient", *,
              dtype: str = "", quant: QuantLayout | None = None) -> PIMTensorSpec:
    spec = PIMTensorSpec(
        DEVICE_DPU, placement, residency, None,
        _shard_map(shape, placement, dpu_ids), placement.reduce_type,
        dtype=dtype, quant=quant,
    )
    spec.validate()
    return spec
```

`_weight_spec`（`:184`）已经有 `node`，直接派生；它还能顺便填 `quant`——权重的量化布局是已知的 `WEIGHT_LAYOUT`：

```python
def _weight_spec(node, strategy, num_layers) -> PIMTensorSpec:
    shape = tuple(node.meta["val"].shape)
    dt = _dtype_of(node)
    # 权重量化布局：定点权重按 WEIGHT_LAYOUT（per_group/128/末轴），
    # 浮点权重不带量化（§4.9.2 的校验会拒绝浮点+quant 的组合）。
    qz = WEIGHT_LAYOUT if dt in ("int4", "int8") else None
    ...
        return _dpu_spec(Placement("Shard", 0 if mode == "col" else 1),
                         shape, dpus, residency="pinned", dtype=dt, quant=qz)
```

#### 4.2.5 验收判据落地

```python
def test_dtype_is_queryable_without_touching_pytorch() -> None:
    """P0-2 的核心判据：不访问 meta["val"] 即可查出 dtype 与量化布局。"""
    gm = _two_layer_llama_graph()
    partition_graph(gm)
    propagate_specs(gm, _strategy())

    for node in gm.graph.nodes:
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or spec.device != "dpu":
            continue
        assert spec.dtype, f"{node.name} 的 spec 没有 dtype"
        validate_dtype(spec.dtype)
        # 与 PyTorch 真值一致 —— 这是「派生而非第二真源」的证明
        assert spec.dtype == str(node.meta["val"].dtype).removeprefix("torch.")


def test_weight_specs_carry_their_quant_layout() -> None:
    """定点权重必须带 WEIGHT_LAYOUT，浮点权重必须不带。"""


def test_dtype_bytes_matches_pytorch_element_size() -> None:
    """dtype_bytes() 对本项目全部类型与 element_size() 等值。

    这是 §4.5.3 把 5 处 element_size() 改成 dtype_bytes() 的行为等价依据。
    int4 走 int8 路径（PyTorch 无 int4 类型），两者都是 1 字节。
    """
    import torch
    for name, torch_dt in (("int8", torch.int8), ("int16", torch.int16),
                           ("int32", torch.int32), ("float16", torch.float16),
                           ("float32", torch.float32)):
        assert dtype_bytes(name) == torch.empty(0, dtype=torch_dt).element_size()
    assert dtype_bytes("int4") == 1          # 不打包，一字节一值（实测）
```

#### 4.2.6 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/dtypes.py` | **新增**：`ELEMENT_DTYPES` + `validate_dtype` + `dtype_bytes` | ~40 行 |
| `contracts/pim_tensor_spec.py` | `PIMTensorSpec` 末尾加 `dtype` / `quant`；`validate()` 加类型与量化相容性校验（§4.9.2） | +2 字段，+12 行 |
| `graph/spec_prop.py` | 新增 `_dtype_of()`；三个构造点加关键字参数；`:543` 改为复用；`_weight_spec` 填 `quant` | +15 行 |
| `gml_bridge/from_fx.py:561` | `_BUFFER_DTYPES` 改为从真源派生 | 1 行 |
| `gml_bridge/from_fx.py:657` `_stamp_dtypes` | **零改动**（Q1 决策） | 0 |
| `tests/test_dtype_carrier.py` | **新增**：可查询、量化布局、位宽等价、编码表交叉校验、反例 | ~130 行 |

### 4.3 P0-3：算子语义单一真源

#### 4.3.1 问题：四份清单性质不同，只有一对是真重复

| # | 清单 | 规模 | 性质 | 消费方式（实测） |
| --- | --- | --- | --- | --- |
| 1 | `from_fx.py:44` `OP_TYPES` | 28 项 → 19 类 | aten 目标 → GML `op_type` 映射 | 查表，`:138` `OP_TYPES.get(node.target)` |
| 2 | `from_fx.py:224` `_ROLE_OP_TYPES` | 4 项 | **角色 → GML op_type，优先于 1** | 查表，`:136-137` 先查它 |
| 3 | `op_classify.py:249` `MNEMONICS` | 14 个 | 助记符元组 | **仅错误信息**：`:283` `list(MNEMONICS)` |
| 4 | `driver.py:83` `_OPLEVEL_OPS` | 15 个 | 内核入口 | 判成员，`:678` `if request.op in _OPLEVEL_OPS` |
| 5 | `fusion_contract.py:24/35/46/62` | 5/7/3/3 | 融合规则 | 判成员 |

两条实测修正了我此前的判断：

**修正一：`op_type` 判定有两条路径，不是一张表。** `from_fx.py:136-138`：

```python
    if role in _ROLE_OP_TYPES:
        return _ROLE_OP_TYPES[role]          # 角色优先
    return OP_TYPES.get(node.target)         # 再按 aten 目标
```

逐头展开后的节点带角色标记（`matmul1`/`matmul2`/`mask`/`softmax`），走第一条；其余走第二条。**派生设计必须保留这个优先级**，否则逐头节点的 `op_type` 会错。

**修正二：`MNEMONICS` 的顺序不敏感。** 它唯一的使用点是 `:283` 拼错误信息，不参与任何逻辑判定：

```python
    if mnemonic not in _OPLEVEL_IR:
        raise KeyError(f"没有 {mnemonic} 的整算子级代表实现；"
                       f"表 1.2.4 的 14 个是 {list(MNEMONICS)}")
```

所以此前风险表里「`MNEMONICS` 派生丢失顺序会改 GeneSim 产物」这条**不成立**——顺序只影响一句错误信息的可读性。真正的成员判定用的是 `_OPLEVEL_IR` 字典。这降低了 P0-3 的风险等级。

**真重复只有第 3 与第 4 份**：去掉 `pim.` 前缀后交集 14 个、差集仅 `convert`，`MNEMONICS` 是 `_OPLEVEL_OPS` 的严格真子集，却各自硬编码。

#### 4.3.2 解决方案：一份登记表 + 四个派生函数

新增 `contracts/op_semantics.py`：

```python
@dataclass(frozen=True)
class OpSemantics:
    """一个 PIM 算子在图阶段的完整语义。

    这是算子语义维度的唯一真源：四份既有清单全部由它派生或对它加断言。
    """
    name: str                          # 算子名，不带 pim. 前缀，如 "matmul"
    gml_op_type: str | None            # 对应的 GML op_type；None = 不落 GML
    has_kernel: bool                   # 算子编译器有内核入口（→ _OPLEVEL_OPS）
    is_mnemonic: bool                  # 进 GeneSim 助记符统计（→ MNEMONICS）
    aten_targets: tuple = ()           # 映射到它的 aten 目标（→ OP_TYPES）
    role_aliases: tuple = ()           # 角色别名，优先于 aten（→ _ROLE_OP_TYPES）


OP_SEMANTICS = (
    OpSemantics("matmul", "MatMul", has_kernel=True, is_mnemonic=True,
                aten_targets=("bmm", "matmul", "mm", "scaled_dot_product_attention"),
                role_aliases=("matmul1", "matmul2")),
    OpSemantics("softmax", "Softmax", has_kernel=True, is_mnemonic=True,
                aten_targets=("_softmax",), role_aliases=("softmax",)),
    OpSemantics("mask", "Mask", has_kernel=True, is_mnemonic=True,
                aten_targets=("masked_fill.Scalar", "where.self"),
                role_aliases=("mask",)),
    OpSemantics("convert", "Convert", has_kernel=True, is_mnemonic=False,
                aten_targets=("to.dtype", "to.dtype_layout")),   # 差集里的那一个
    # ... 其余 11 个算子同构声明
)
```

四个派生函数**保持既有数据形态**（frozenset 仍是 frozenset、tuple 仍是 tuple），调用方零改动：

```python
def oplevel_ops() -> frozenset:
    """算子编译器内核入口集合。取代 driver.py:83 的字面量。"""
    return frozenset(s.name for s in OP_SEMANTICS if s.has_kernel)

def mnemonics() -> tuple:
    """GeneSim 助记符。取代 op_classify.py:249 的字面量。

    顺序按登记表声明序——它只进错误信息（op_classify.py:283），不参与判定，
    但保持稳定顺序便于人读。
    """
    return tuple(f"pim.{s.name}" for s in OP_SEMANTICS if s.is_mnemonic)

def aten_to_gml() -> dict:
    """aten 目标 → GML op_type。取代 from_fx.py:44 的 OP_TYPES。"""
    out = {}
    for s in OP_SEMANTICS:
        if s.gml_op_type is None:
            continue
        for t in s.aten_targets:
            out[_resolve_aten(t)] = s.gml_op_type
    return out

def role_to_gml() -> dict:
    """角色 → GML op_type。取代 from_fx.py:224 的 _ROLE_OP_TYPES。

    **必须与 aten_to_gml 分开**：from_fx.py:136-138 先查角色再查 aten，
    合并成一张表会丢掉这个优先级，逐头节点的 op_type 会退回按 aten 判定。
    """
    return {r: s.gml_op_type for s in OP_SEMANTICS
            for r in s.role_aliases if s.gml_op_type}
```

调用方改动各 1 行：

```python
# gml_bridge/from_fx.py
OP_TYPES = aten_to_gml()          # 替代 28 行字面量
_ROLE_OP_TYPES = role_to_gml()    # 替代 6 行字面量

# opcompiler_bridge/driver.py
_OPLEVEL_OPS = oplevel_ops()      # 替代 5 行字面量

# genesim_bridge/op_classify.py
MNEMONICS = mnemonics()           # 替代 5 行字面量
```

#### 4.3.3 融合规则不派生，加断言

`fusion_contract.py` 的四组是**融合规则**而非算子集合，性质不同，不派生。但要加一致性断言（实现见 §4.9.2(3)）：融合目标必须是已登记算子，否则融合表与算子表会不同步而静默失效。

#### 4.3.4 `aten_targets` 用字符串而非 torch 对象

登记表里 `aten_targets=("bmm", "matmul", ...)` 是字符串，`_resolve_aten()` 在派生时转成 `torch.ops.aten.*` 对象。两个理由：

1. `contracts/` 应尽量少持有 torch 运行时对象——虽然 `fusion_contract.py:21` 已 `import torch`，但字符串声明让登记表可读、可 diff、可静态检查。
2. `OP_TYPES` 的键今天就是 `torch.ops.aten.xxx.default` 对象，`_resolve_aten()` 负责这层转换，调用方拿到的字典与今天完全同构。

```python
def _resolve_aten(name: str):
    """把 "bmm" / "to.dtype" 解析成 torch.ops.aten 的重载对象。

    名字里带点的是显式重载（to.dtype）；不带点的补 .default。
    解析失败直接抛——拼错算子名必须立刻暴露。
    """
    import torch
    parts = name.split(".")
    op = getattr(torch.ops.aten, parts[0], None)
    if op is None:
        raise ValueError(f"torch.ops.aten 没有 {parts[0]!r}")
    overload = parts[1] if len(parts) > 1 else "default"
    if not hasattr(op, overload):
        raise ValueError(f"aten.{parts[0]} 没有重载 {overload!r}")
    return getattr(op, overload)
```

#### 4.3.5 验收判据落地

需求 P0-3 要求「新增算子只需改一处」「四份清单有可校验引用关系」。测试沿用本仓既有的集合相等断言范式：

```python
def test_derived_lists_match_the_literals_they_replace() -> None:
    """派生结果必须与改动前的字面量逐项相同。

    这是「重构而非改行为」的直接证据：四份清单的内容一个都不能变，
    只是来源从字面量变成派生。
    """
    assert oplevel_ops() == {
        "softmax", "dynamic_quant", "gather", "rope", "matmul", "normalize",
        "mask", "transpose", "reshape", "concat", "convert", "lut", "eltwise",
        "kv_cache", "split_heads",
    }
    assert set(mnemonics()) == {
        "pim.normalize", "pim.matmul", "pim.softmax", "pim.mask", "pim.rope",
        "pim.lut", "pim.eltwise", "pim.dynamic_quant", "pim.kv_cache",
        "pim.gather", "pim.transpose", "pim.reshape", "pim.split_heads",
        "pim.concat",
    }

def test_role_lookup_still_takes_priority_over_aten() -> None:
    """逐头节点必须按角色判 op_type，不能退回按 aten 判。

    反例：matmul1 的 aten 目标是 bmm，两条路径都给 "MatMul" 所以看不出问题；
    但 mask 角色的节点其 aten 目标可能是 where.self —— 若丢了角色优先级，
    某些节点的 op_type 会变。这里直接断言两张表分离。
    """
    assert set(role_to_gml()) == {"matmul1", "matmul2", "mask", "softmax"}
    assert not (set(role_to_gml()) & {str(k) for k in aten_to_gml()})

def test_adding_an_operator_touches_only_the_registry() -> None:
    """在登记表加一个算子，四个派生视图自动包含它。"""
```

#### 4.3.6 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/op_semantics.py` | **新增**：`OpSemantics` + 15 个算子登记 + 4 个派生函数 + `_resolve_aten` + `_validate_registry` | ~150 行 |
| `gml_bridge/from_fx.py:44,224` | `OP_TYPES` / `_ROLE_OP_TYPES` 改为派生 | -34 行，+2 行 |
| `opcompiler_bridge/driver.py:83` | `_OPLEVEL_OPS` 改为派生 | -5 行，+1 行 |
| `genesim_bridge/op_classify.py:249` | `MNEMONICS` 改为派生 | -5 行，+1 行 |
| `contracts/fusion_contract.py` | 加融合目标断言 | +12 行 |
| `tests/test_op_semantics.py` | **新增**：派生一致性 + 角色优先级 + 新增算子只改一处 | ~120 行 |

### 4.4 P0-4：Memory Layout 补排布层

#### 4.4.1 问题：四层里第 3 层缺载体

| 层 | 状态 | 载体 |
| --- | --- | --- |
| 1 切分 | 已有 | `TensorShardDetail.shard_dim / start_idx / end_idx / local_shape` |
| 2 地址 | 已有 | `mram_offset` + `DPUPlan` 三区 + `KVRegionSpec.kv_off` |
| 3 排布 | **对齐已有、步幅缺** | 对齐：`op_contract.dma_align:15`（带 2 的幂校验 `:30`）、`kv_layout.align_up:27`。步幅：无字段 |
| 4 介质交错 | 不做 | — |

缺口的具体表现是 `memory/mem_planner.py:50-52`：

```python
def bytes_of(local_shape: tuple[int, ...], itemsize: int) -> int:
    """返回本地分片的字节数。"""
    return prod(local_shape) * itemsize
```

**「行主序紧密排列」是这个函数的隐含假设，不写在任何字段里。** 三个调用点（`:91` 权重区、`:133` / `:180` 激活区与 KV 区）都继承这个假设。

#### 4.4.2 解决方案：`TensorShardDetail` 补两个字段

```python
# contracts/pim_tensor_spec.py
@dataclass(frozen=True)
class TensorShardDetail:
    dpu_id: int
    shard_dim: int
    start_idx: int
    end_idx: int
    local_shape: tuple[int, ...]
    mram_offset: int = 0
    # 本轮新增：排布层（第 3 层）
    elem_strides: tuple[int, ...] = ()   # 逐维步幅（元素数）；() = 行主序紧密
    align_bytes: int = 0                 # 该分片起始地址的额外对齐；0 = 无额外要求
```

**默认值的选择是产物不变的关键**：`elem_strides = ()` 显式表示「行主序紧密」，正是 `bytes_of()` 今天的隐含假设。所以全部现有路径行为完全不变，而假设从此有了字段承载。空值语义的完整口径见 §4.9.4。

`frozen=True` 保持不变，回填沿用既有的 `dataclasses.replace` 模式（`mem_planner.py:89` / `:274` / `:281` 三处先例）：

```python
# 内存规划回填排布时的写法，与现有 mram_offset 回填同构
spec.shard_map[dpu_id] = replace(
    spec.shard_map[dpu_id], mram_offset=off, align_bytes=align)
```

#### 4.4.3 `bytes_of` 的行为等价改造

三个调用点都传 `detail`，所以签名从「拆开传形状」改为「可选传步幅」：

```python
def bytes_of(local_shape: tuple[int, ...], itemsize: int,
             elem_strides: tuple[int, ...] = ()) -> int:
    """本地分片占的字节数。

    `elem_strides` 为空时按行主序紧密算 —— 与本轮改动前**逐字节等价**，
    这是三个调用点（:91 权重区、:133 激活区、:180 KV 区）的现状。
    非空时按最外维步幅算，覆盖带填充的排布：此时字节数不等于
    `prod(shape) * itemsize`，因为每行末尾有填充。
    """
    if not elem_strides:
        return prod(local_shape) * itemsize       # 原路径，一字未改
    if len(elem_strides) != len(local_shape):
        raise ValueError(
            f"elem_strides 秩 {len(elem_strides)} 与 local_shape 秩 "
            f"{len(local_shape)} 不符")
    return local_shape[0] * elem_strides[0] * itemsize
```

调用点改动是**加一个参数**，不改变现有行为：

```python
# memory/mem_planner.py:91
- off += align_up(bytes_of(first.local_shape, itemsize), align)
+ off += align_up(bytes_of(first.local_shape, itemsize, first.elem_strides), align)

# :133 与 :180 同构
- size=bytes_of(detail.local_shape, itemsize),
+ size=bytes_of(detail.local_shape, itemsize, detail.elem_strides),
```

**为什么最外维步幅就够**：MRAM 分片是一段连续区间（§4.7.3 已实测），带填充时填充在每行末尾，所以总字节数 = 最外维长度 × 最外维步幅 × 元素宽度。不需要遍历全部维度。若将来出现更复杂的排布，`validate()` 的「相邻元素重叠」检查（§4.9.2）会先抓住不自洽的步幅组合。

#### 4.4.4 `align_up` 去重

两份实现算法相同、校验不同：

| 位置 | 校验 | 调用方 |
| --- | --- | --- |
| `memory/kv_layout.py:27` | **有**（`align <= 0` 抛错） | `kv_layout.py:81`、`mem_planner.py:91/215/219` |
| `orchestrator/l2_alloc.py:42` | 无 | `l2_alloc.py:107/180/191`、`layer_fields.py:28` |

合并到 `contracts/mem_layout.py`（§4.6.2 已给实现），**保留带校验的那份**——取严不取宽。两侧都已依赖 `contracts`，无新增依赖方向。

#### 4.4.5 对激活区复用算法的影响（必须确认无副作用）

`memory/mem_planner.py:188` 的 `greedy_reuse` 按 `t.size` 降序放槽并判生命周期重叠。它的注释记录了一个关键坑：

```
两个生命周期判据都用严格不等号，故意不取等：取等意味着某个节点在同一个
step 里既读旧张量又写新张量，两者拿到同一个基址。已编译内核按裸指针逐块
读写，写输出的前几行会覆盖尚未读取的输入行，算出错误结果。
```

`elem_strides` 非空会让 `t.size` 变大（含填充），于是**槽位复用决策可能改变**。但这不影响正确性，且本轮不会触发：

| 事实 | 依据 |
| --- | --- |
| 本轮全部张量的 `elem_strides` 都是 `()` | MRAM 级排布连续（§4.7.3 三条实测） |
| 因此 `t.size` 与今天完全相同 | `bytes_of` 走原路径 |
| 槽位复用决策不变 | 输入相同、算法未改 |

**即 P0-4 在 MRAM 级是「把隐含假设显式化」，不是「引入新排布」。** 字段留给将来真正出现非连续 MRAM 排布时用，今天全部取默认值——这也是产物逐字节不变的保证。

#### 4.4.6 验收判据落地

```python
def test_default_strides_are_byte_identical_to_the_old_formula() -> None:
    """空步幅必须与改动前的 prod(shape)*itemsize 完全相同。

    这是 P0-4 「只加字段不改行为」的直接证据。
    """
    for shape in [(3, 4), (1, 4096), (32, 128), (11008, 4096), (1,)]:
        for itemsize in (1, 2, 4):
            assert bytes_of(shape, itemsize) == bytes_of(shape, itemsize, ())
            assert bytes_of(shape, itemsize) == prod(shape) * itemsize


def test_padded_strides_account_for_the_padding() -> None:
    """带填充时字节数大于紧密排列 —— 这是新增能力的证明。"""
    # 4 行、每行 200 个元素、行间跨 256（每行末尾 56 个填充）
    assert bytes_of((4, 200), 2, (256, 1)) == 4 * 256 * 2


def test_align_up_has_exactly_one_implementation() -> None:
    """源码扫描：全仓只能有一处 def align_up。"""
    import re
    from pathlib import Path
    root = Path(__file__).parent.parent
    hits = [f"{p.relative_to(root)}:{i}"
            for p in root.rglob("*.py") if "test" not in p.name
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.match(r"\s*def align_up\b", ln)]
    assert len(hits) == 1, f"align_up 应只有一处实现，实际 {hits}"
```

#### 4.4.7 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/pim_tensor_spec.py` | `TensorShardDetail` 补 `elem_strides` / `align_bytes`；`validate()` 补排布校验（§4.9.2） | +2 字段，+20 行校验 |
| `contracts/mem_layout.py` | `align_up` 唯一实现（与 §4.6.2 同一文件） | 见 §4.6 |
| `memory/mem_planner.py:50` | `bytes_of` 加可选参数（行为等价） | +8 行 |
| `memory/mem_planner.py:91,133,180` | 三个调用点传 `detail.elem_strides` | 3 行 |
| `memory/kv_layout.py:27` | 删除 `align_up`，改 import | -5 行，+1 行 |
| `tests/test_mem_layout.py` | **新增**：默认值等价、带填充、唯一实现、反例 | ~100 行 |

### 4.5 P0-5：统一 IR 是四维唯一来源

#### 4.5.1 问题：取数点必须分类处置，不能一律改掉

实测非测试代码里从 PyTorch 取 dtype / 字节宽度的**精确命中是 6 处**（此前估的 9 处偏高）：

| 文件:行号 | 取什么 | 用途 |
| --- | --- | --- |
| `runtime/exec_plan_gen.py:67` | `element_size()` | 算 MRAM 访问区间字节数 |
| `runtime/exec_plan_gen.py:323` | `str(dtype).removeprefix("torch.")` | 内核实参 dtype 串 |
| `runtime/exec_plan_gen.py:329` | 同上 | 内核输出 dtype 串 |
| `memory/mem_planner.py:90` | `element_size()` | 权重区字节数 |
| `memory/mem_planner.py:160` | `element_size()` | 激活区字节数 |
| `graph/spec_prop.py:539` | `numel() * element_size()` 与 `str(dtype)` | 构造 `RedistributeEdge` 的 nbytes 与 dtype |

其余 35 处 `.meta["val"]` / `.meta.get("val")` 取的是 **shape / ndim**（21 处，要全局形状而非本地分片形状）或**写入 val 示例张量**（11 处，`graph/split_heads.py` 新建节点时的 FX 契约），这两类**必须保留**。

**关键发现：dtype 的派生逻辑已经存在。** `graph/spec_prop.py:543` 已在做：

```python
        dtype=str(val.dtype).removeprefix("torch."),
```

只是它写在 `RedistributeEdge` 上，而 `PIMTensorSpec` 没有这个字段。所以 P0-2 的落地不是「发明派生逻辑」，而是**把已有的派生挪到 spec 构造处，让全链路共用一份**。

#### 4.5.2 解决方案（一）：dtype 在 spec 构造处派生一次

三个 spec 构造点（`_host_spec:128` / `_dpu_spec:135` / `_weight_spec:184`）都用**位置参数**调用：

```python
    spec = PIMTensorSpec(DEVICE_HOST, REPLICATE, "transient", None, {}, None)
```

所以新字段**必须加在末尾且带默认值**，否则这些调用全部要改。这与 §4.2.2 的设计一致：

```python
# contracts/pim_tensor_spec.py
@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]
    placement: Placement
    residency: Literal["transient", "pinned"]
    pinned_dpu_id: int | None
    shard_map: dict[int, TensorShardDetail]
    reduce_type: str | None
    # 本轮新增，必须在末尾：上面六个字段在 graph/spec_prop.py 的三个构造点
    # 都是位置参数传入的，插在中间会静默错位。
    dtype: str = ""
    quant: QuantLayout | None = None
```

`_dpu_spec` / `_host_spec` 增加一个 `dtype` 关键字参数，从调用方已有的 `val` 派生：

```python
def _dpu_spec(
    placement: Placement,
    shape: tuple[int, ...],
    dpu_ids: tuple[int, ...],
    residency: Literal["transient", "pinned"] = "transient",
    *,
    dtype: str = "",              # 关键字参数：现有 6 个调用点不传也能跑
) -> PIMTensorSpec:
    spec = PIMTensorSpec(
        DEVICE_DPU, placement, residency, None,
        _shard_map(shape, placement, dpu_ids), placement.reduce_type,
        dtype=dtype,
    )
    spec.validate()
    return spec
```

派生的唯一入口（`graph/spec_prop.py` 内新增一个小函数）：

```python
def _dtype_of(node: Node) -> str:
    """节点输出的元素类型名。

    这是 dtype 从 PyTorch 进入统一 IR 的**唯一入口**。之后全链路查
    `spec.dtype`，不再问 `meta["val"]` —— §4.9.2 的跨维校验会守住两者一致。
    """
    val = node.meta.get("val")
    if val is None:
        return ""                 # 非张量节点（如 get_attr 的标量）
    return str(val.dtype).removeprefix("torch.")
```

#### 4.5.3 解决方案（二）：5 处消费点改为查 spec

```python
# runtime/exec_plan_gen.py:67
- itemsize = node.meta["val"].element_size()
+ itemsize = dtype_bytes(node.meta[SPEC_META_KEY].dtype)

# runtime/exec_plan_gen.py:323
- arg_dtypes.append(str(arg.meta["val"].dtype).removeprefix("torch."))
+ arg_dtypes.append(arg.meta[SPEC_META_KEY].dtype)

# runtime/exec_plan_gen.py:329
- "dtype": str(node.meta["val"].dtype).removeprefix("torch."),
+ "dtype": spec.dtype,            # spec 在 :263 已取出

# memory/mem_planner.py:90
- itemsize = nodes[0].meta["val"].element_size()
+ itemsize = dtype_bytes(nodes[0].meta[SPEC_META_KEY].dtype)

# memory/mem_planner.py:160
- itemsize = node.meta["val"].element_size()
+ itemsize = dtype_bytes(node.meta[SPEC_META_KEY].dtype)
```

`graph/spec_prop.py:539` 那一处**保留** `val.numel() * val.element_size()`——它在构造 `RedistributeEdge`，此刻正处于派生阶段本身，查 spec 会形成循环。但它的 `dtype=` 参数改为复用 `_dtype_of()`，消除重复的 `removeprefix` 写法。

**行为等价性论证**（这是产物不变的关键）：`dtype_bytes()` 的取值表来自实测（int4 一字节一值不打包），与 `torch.Tensor.element_size()` 对本项目使用的 6 种类型逐一相同。唯一差异是 int4——但 PyTorch 没有 int4 类型，图上 int4 权重的 `val.dtype` 实际是 `int8`，所以 `element_size()` 返回 1，与 `_DTYPE_BYTES["int4"] = 1` 一致。

#### 4.5.4 解决方案（三）：无旁路的可执行判据

需求 P0-5 的判据是「三个 bridge 无绕过统一 IR 的路径」。由于 shape / val 写入两类合法保留，断言不能是「零 `.meta["val"]`」，而要**精确针对 dtype 与字节宽度这两种取数**：

```python
def test_dtype_enters_the_ir_through_exactly_one_door() -> None:
    """除派生入口外，生产代码不得从 val 取 dtype 或 element_size。

    白名单只有 graph/spec_prop.py —— dtype 在 _dtype_of() 里派生一次，
    之后全链路查 spec.dtype。多一处白名单 = 又开了一条旁路，
    两个真源会漂移（§4.9.2 的跨维校验一正是为此）。
    """
    import re
    from pathlib import Path

    root = Path(__file__).parent.parent
    dirs = ("memory", "runtime", "gml_bridge", "genesim_bridge",
            "opcompiler_bridge", "comm", "graph", "orchestrator")
    pattern = re.compile(r'\.element_size\(\)|\.meta\[.val.\]\.dtype')
    allowed = {"graph/spec_prop.py"}

    offenders = []
    for d in dirs:
        for py in (root / d).rglob("*.py"):
            rel = f"{d}/{py.name}"
            if rel in allowed:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{rel}:{i}")
    assert offenders == [], f"这些位置绕过统一 IR 直接问 PyTorch 取 dtype：{offenders}"
```

这个测试沿用 `tests/test_runtime_compiled_coverage.py:155-171` 的源码正则扫描范式（本仓既有做法，非新发明）。

#### 4.5.5 GeneSim 侧的消费粒度（Q4 决策落地）

Q4 已定 GeneSim 继续只消费所需维度。实测 sidecar 的 5 个 entry 字段消费情况：

| 字段 | GeneSim 侧消费 | 处置 |
| --- | --- | --- |
| `device_hint` / `op_type` / `shards` | 1 / 12 / 6 处 | 改为从统一 IR 取数，**字段内容不变** |
| `shards[*].dpu_id` / `local_in_features` / `local_out_features` | 5 / 2 / 2 处 | 同上 |
| `semantic_role` / `weight` / `shard_axis` | **0 处** | 保留。`placement_export.py:414-417` 注释已说明是排错用；本轮**显式登记为调试字段** |

调试字段的登记方式（满足 P1-2「无消费者字段必须显式登记理由」）：

```python
# genesim_bridge/placement_export.py
# 这三个字段 GeneSim 侧没有读者，是有意为之的排错信息，不是遗漏。
# 登记在此以免后续被当死字段删掉；P1-2 的「回传字段必须有消费方」这条
# 判据不适用于它们——它们是导出侧的调试副本，不是回传通道的一部分。
DEBUG_ONLY_SIDECAR_FIELDS = frozenset({"semantic_role", "weight", "shard_axis"})
```

#### 4.5.6 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/pim_tensor_spec.py` | `PIMTensorSpec` 末尾加 `dtype` / `quant`；`validate()` 加类型校验 | +2 字段，+12 行校验 |
| `contracts/dtypes.py` | **新增**：`ELEMENT_DTYPES` / `validate_dtype` / `dtype_bytes` | ~40 行 |
| `graph/spec_prop.py` | 新增 `_dtype_of()`；三个构造点加 `dtype=` 传参；`:543` 改为复用 | +10 行 |
| `runtime/exec_plan_gen.py:67,323,329` | 改为查 `spec.dtype` | 3 行 |
| `memory/mem_planner.py:90,160` | 同上 | 2 行 |
| `genesim_bridge/placement_export.py` | 加调试字段登记 | +5 行 |
| `tests/test_no_bypass.py` | **新增**：源码扫描断言 | ~40 行 |

### 4.6 P0-6：编排器改为消费统一 IR

#### 4.6.1 问题：排布规则的真源只在编排器，且范围比需求描述的更大

`orchestrator/` 共 3014 行，`layer_fields.py` 占 1660 行。实测该文件里 `align16` / `stride_z` 服务**两组不同的字段**——需求文档只提了第一组：

| 组 | 字段 | 处数 | 计算函数 | 是否需求已列 |
| --- | --- | --- | --- | --- |
| **A. 排布步幅** | `Input Stride X`(:488)、`Output Stride X/Z`(:494-498)、`Eltwise broadcast Input Stride X`(:1174)、`Data scale stride X/Z`(:1276-1277)、`DDR data scale stride X/Z`(:1293-1294)、`DDR Output stride X/Z`(:1370-1371)、`DDR Weight stride X/Z`(:1402-1403) | 10 | `stride_z`(:31) → `align16`(:27) | 是 |
| **B. L2 段尺寸** | `L2 input buffer size 0/1`(:1471,:1485)、`L2 output buffer size`(:1495) | 3 | `_l2_in_size`(:575)、`_l2_dual_in_size`(:596)、`_l2_out_size`(:605)，内部直接调 `align16` 6 处（:589,592,617,621,623） | **否，需求漏了** |
| C. 卷积滑窗 | `Filter`/`Pooling Horizontal/Vertical Stride`(:502-503,:528-529) | 4 | 常量 0/1 | 明确排除 |

两组都依赖 `align16`，共用同一个对齐粒度（16）。**若只搬 A 组，`align16` 仍要留在编排器给 B 组用，去重目标落空**。所以本设计把 A、B 两组一并搬迁，C 组不动。

B 组的规则含大量实测特例（注释里记着 422 层的实测值）：

```python
# :585-592 _l2_in_size
      if kind in ("bmm1", "bmm2"):
          return in_w + 16                       # hd→144、S→1040
      if kind == "dq_p4":
          return (align16(in_w) + 16) * 2        # 4096→8224、1024→2080

# :617-623 _l2_out_size
      if kind == "bmm2":
          return (align16(H) + 16) * 2
      if kind == "dq_p2":
          return 32 if out_w <= 1 else align16((out_w + 16) * 2)   # Gn=1 例外
      return align16(out_w) + 16                 # 4096→4112、11008→11024，不乘 2
```

这些数字是逐层实测对出来的，搬家时**不能"顺手简化"公式**。

#### 4.6.2 解决方案：规则搬进 contracts/，编排器只做渲染

新增 `contracts/mem_layout.py`，承接三类内容：

```python
"""L2/DDR 级的排布与尺寸规则。

这一层归编排器消费（见 §2.1.4 的内存层次分管边界），**不下发给 PIMMLIR**。
规则本身来自参考产物的 422 层实测，注释里的数字是核对依据，改公式必须重新对拍。
"""

L2_ALIGN = 16          # 对齐粒度。文档 B7：L2 output size 按 align16(Width)+16 算。
L2_OUTPUT_PAD = 16


def align_up(n: int, align: int) -> int:
    """向上对齐。合并 memory/kv_layout.py:27 与 orchestrator/l2_alloc.py:42 两份实现。

    保留 kv_layout 那份的参数校验 —— l2_alloc 那份没有校验，
    合并时取严不取宽。
    """
    if align <= 0:
        raise ValueError(f"align 必须为正，got {align}")
    return (n + align - 1) // align * align


def align16(width: int) -> int:
    return align_up(width, L2_ALIGN)


def stride_z(width: int, *, final: bool, scalar_align16: bool = False) -> int:
    """终相 align16(W)+15；中间相 =W；Width=1 的中间相有时 16。

    照搬 orchestrator/layer_fields.py:31-37，逻辑一字不改。
    `scalar_align16` 只在 dq_p2 且 out_w==1 时为真（见调用点 :473-474）。
    """
    if final:
        return align16(width) + 15
    if scalar_align16 and width == 1:
        return 16
    return width
```

L2 段尺寸三个函数同样整体搬入（保留全部实测特例与注释），签名不变。

**编排器侧的改动是「删除 + import」**：

```python
# orchestrator/layer_fields.py:27-37 —— 删除本地实现
- def align16(width: int) -> int:
-     return l2_alloc.align_up(width, 16)
-
- def stride_z(width: int, *, final: bool, scalar_align16: bool = False) -> int:
-     ...
+ from contracts.mem_layout import align16, stride_z

# orchestrator/l2_alloc.py:42 —— 删除第二份 align_up
- def align_up(value: int, align: int) -> int:
-     return (value + align - 1) // align * align
+ from contracts.mem_layout import align_up

# memory/kv_layout.py:27 —— 删除第一份 align_up
- def align_up(n: int, align: int) -> int:
-     ...
+ from contracts.mem_layout import align_up
```

`orchestrator/layer_fields.py:12-14` 已经 import `contracts`，`memory/` 也已依赖 `contracts`，所以**无新增依赖方向**。

#### 4.6.3 两步落地法：先对照，后删除

需求 Q3 已定「不保留双份」。但直接改 1660 行的文件风险过高，设计为两步，中间用**全等对照**把风险挡住。

**第一步：IR 侧实现 + 只读对照（不删编排器代码）**

```python
# tests/test_stride_parity.py（第一步的临时判据，第二步后转为防回归）
def test_ir_side_rules_match_orchestrator_exactly() -> None:
    """IR 侧新实现与编排器现有实现，对全部输入组合逐一全等。

    这是搬家安全的前提：不全等就不能进第二步。
    覆盖 422 层实测涉及的全部 kind × width 组合。
    """
    from contracts import mem_layout as new
    from orchestrator import layer_fields as old

    widths = [1, 16, 32, 86, 128, 144, 1024, 2048, 4096, 11008]
    for w in widths:
        for final in (False, True):
            for scalar in (False, True):
                assert new.stride_z(w, final=final, scalar_align16=scalar) == \
                       old.stride_z(w, final=final, scalar_align16=scalar), \
                       f"stride_z 不一致：w={w} final={final} scalar={scalar}"
        assert new.align16(w) == old.align16(w), f"align16 不一致：w={w}"

    for kind in ("bmm1", "bmm2", "dq_p2", "dq_p4", "gemm", "mask", "softmax"):
        for w in widths:
            for dt in (0, 1, 3):
                assert new._l2_in_size(kind, w, dt) == old._l2_in_size(kind, w, dt)
                assert new._l2_out_size(kind, w, dt) == old._l2_out_size(kind, w, dt)
```

**第二步：删除编排器本地实现，改为 import**

第二步完成后，上面的对照测试失去对象（`old` 已经 import 自 `new`，自比恒等）。此时把它改为**对照落盘产物**：

```python
def test_layer_fields_output_is_byte_identical_to_baseline() -> None:
    """层参数文本与改动前逐字节一致。

    基线文件由第二步开始前生成并存入 test-results/，
    这是 P0-6 的最终判据（需求 §5.2）。
    """
```

**为什么两步是必要的**：第一步验证「新实现算得对」，第二步验证「接线接得对」。合成一步做，一旦产物不一致就无法区分是公式抄错还是调用点接错——而 1660 行文件里有 13 处调用点。

#### 4.6.4 编排器不持有 spec，所以「消费 IR」是消费规则而非消费数据

这是一处容易误解的地方，必须写清。实测 `orchestrator/` 下搜 `PIMTensorSpec` / `TensorShardDetail` / `shard_map`：**零命中**（§2.1.4 证据三）。它的几何「从 GML 边取」（`layer_fields.py:3` 模块注释）。

所以 P0-6 的「编排器改为消费统一 IR」**不是**让编排器去读 `spec.shard_map[*].elem_strides`——那需要把 spec 传进编排器，而编排器在链路上位于 GML 之后、拿不到 FX 图。

准确含义是：**编排器消费统一 IR 的「规则」，而非「数据」**。

| 消费什么 | 从哪来 | 本轮动作 |
| --- | --- | --- |
| 排布**规则**（`stride_z` / `align16` / L2 尺寸公式） | `contracts/mem_layout.py` | 搬迁 + import |
| 几何**数据**（width / height / kind） | GML 边与节点字段 | 不变 |

这也解释了为何 `TensorShardDetail.elem_strides`（P0-4 新增）与编排器无直接关系——那个字段是 MRAM 级的排布，下发给 PIMMLIR（见 §4.7.7），编排器管的是 L2/DDR 级。两者共用 `contracts/mem_layout.py` 的对齐工具，但不共享数据通路。

#### 4.6.5 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/mem_layout.py` | **新增**：`align_up`（唯一实现）+ `align16` + `stride_z` + L2 尺寸三函数（含全部实测特例与注释） | ~130 行 |
| `orchestrator/layer_fields.py:27-37` | 删除本地 `align16`/`stride_z`，改 import | -11 行，+1 行 |
| `orchestrator/layer_fields.py:575-623` | 删除 L2 尺寸三函数，改 import | -50 行，+1 行 |
| `orchestrator/l2_alloc.py:42` | 删除第二份 `align_up`，改 import | -3 行，+1 行 |
| `memory/kv_layout.py:27` | 删除第一份 `align_up`，改 import | -5 行，+1 行 |
| `orchestrator/net_ini.py:51-61` | 四个全网恒定步幅改为引用 `contracts/mem_layout.py` | +4 行 |
| `tests/test_stride_parity.py` | **新增**：第一步全等对照，第二步后转产物比对 | ~120 行 |

净效果是**删多于加**（编排器减约 69 行，contracts 增约 130 行），符合 `CLAUDE.md`「删优于加」。

### 4.7 P1-1：PIMMLIR 四维覆盖与传递接口

> **本节经两轮实测重写。** 第一版只验证 `dpusPerDevice` 一个字段便断言「FlagTree 零改动」，属以点代面。第二版据编排器的 `align16(W)+15` 断言「Memory Layout 有缺口、需改 ODS」，**该论证前提错误**（用 L2/DDR 层级的事实去论证 WRAM/MRAM 层级的能力）。本版按**内存层级**分层核查后给出结论。

#### 4.7.1 核心认识：Memory Layout 必须分层讨论

「PIMMLIR 能不能表达内存布局」这个问题若不分层就无法回答，因为**本项目的内存层次由两个互不相交的组件分管**。

实测证据一，编排器的链路位置（`orchestrator/__init__.py` 模块文档原文）：

```
编排器：GML 之后的那一段。链路里的位置是**串行的第三段**，不回填 GML：

    图编译器 + 算子编译器 → GML（200 节点）→ 编排器 → 层参数 + net.ini
...
**不管切分放置**——那是图编译器的事（`graph/strategy.py`）。切分决定「算什么、
多大」（进 GML 的形状），编排决定「片上怎么放、按什么序走」。
```

实测证据二，L2 地址分配在架构上归编排器（`orchestrator/l2_alloc.py` 模块文档原文）：

```
这是编排器存在的核心理由。实测参考产物的 L2 物理段折叠率 96.4%，而
`L2 input buffer offset` 全图出现 0 次（只分配输出段）——这不是「每层各要
一块」，是跨整张图做生命周期分析后大量复用同一地址。单个算子的 pass 看不到
全局，图编译器也从没做过地址分配，所以只能在这里做。
```

实测证据三，编排器**完全看不到** `PIMTensorSpec`：

```bash
$ grep -rn "PIMTensorSpec|TensorShardDetail|shard_map" orchestrator/*.py
# 零命中
```

它的几何「从 GML 边取」（`layer_fields.py:3`），即走的是「GML → 编排器」这条串行路径，不走「统一 IR → PIMMLIR」那条。

实测证据四，内存层级词频分布印证分工：

| 组件 | L2 | DDR | WRAM | MRAM |
| --- | --- | --- | --- | --- |
| `orchestrator/` | 87 | 69 | 1 | 1 |
| `contracts/op_contract.py` + `memory/` | 0 | 0 | 3 | 11 |
| PIMMLIR `.td` | 14 | 0 | 27 | 15 |

**结论**：内存层次的分管边界是

| 层级 | 负责组件 | 排布信息的流向 |
| --- | --- | --- |
| **WRAM / MRAM** | 图编译器 + 算子编译器 | 统一 IR → pimir → 算子编译器 |
| **L1 / L2** | 编排器 | GML → 编排器 → 层参数（**不经 PIMMLIR**） |
| **DDR** | 编排器 | 同上 |

#### 4.7.2 撤回第二版的 ODS 扩展方案

第二版用编排器的 `Output Stride Z = align16(4096)+15 = 4111` 作为「PIMMLIR 表达不了带填充排布」的证据，并据此提出扩展 `#pim.tasklet_tiled` 加 `elemStrides` 参数（方案 M1）。

**该方案撤回**，两条理由：

1. **那 14 处 stride 全在 L2/DDR 层级。** 逐行核实字段名：`DDR Output stride X/Z`（`layer_fields.py:1370-1371`）、`DDR Weight stride X/Z`（`:1402-1403`）、`DDR data scale stride X/Z`（`:1293-1294`）、`Data scale stride X/Z`（`:1276-1277`）、`Output Stride X/Z`（`:494-498`，L2 输出段）、`Input Stride X`（`:488`）、`Eltwise broadcast Input Stride X`（`:1174`）。按 §4.7.1 的分管边界，这些**不流经 PIMMLIR**。
2. **MRAM/WRAM 级的连续假设实测成立**（§4.7.3）。

强行加这个参数会引入一个无生产方的字段——正是需求 P1-2 与 P2-1 要防的那类死字段，且违反需求 §2.3「不改 PIMMLIR 方言定义」的边界。

#### 4.7.3 MRAM/WRAM 级的连续假设实测成立

`PIMTypes.td:21-25` 的 ODS 注释声明：

```
This is a deliberately trimmed-down analogue of `!ttg.memdesc`. It carries
no layout encoding, because a scratchpad buffer on a PIM device is just a
contiguous span of bytes -- there is no swizzling to describe.
```

**在它自己的作用域（WRAM/MRAM）内，这个声明是正确的**，三条实测：

| # | 事实 | 证据 |
| --- | --- | --- |
| 1 | MRAM 张量字节数就是紧密乘积 | `memory/mem_planner.py:50-52` `bytes_of()` = `prod(local_shape) * itemsize` |
| 2 | MRAM/WRAM 级不存在行内填充 | 在 `memory/`、`contracts/op_contract.py`、`comm/` 搜 `align16` / `+ 15` / `+ 16`：**零命中**（对比编排器侧大量存在） |
| 3 | 非连续分片被展开成多段连续，而非跨步视图 | `comm/plan.py:62-79` `_runs()`，注释原文「分片到全局摊平坐标的**连续区间**」；`local_slice`（`:248-268`）是整段拷贝，不是跨步视图 |

另一条印证：下发算子编译器的契约 `PIMHardwareConfig`（`contracts/op_contract.py:11-16`）只有 `mram_bytes_per_dpu` / `wram_bytes_per_dpu` / `dma_align`，**没有任何 L2 字段**——即这条接口在设计上就不传 L2 信息。

#### 4.7.4 四维覆盖结论（按层级限定）

| 维度 | PIMMLIR 是否覆盖 | 载体与实测依据 |
| --- | --- | --- |
| 算子语义 | **覆盖** | 37 算子（20 分块级 + 17 算子级）、19 结构化属性、34 处 `hasVerifier`、14 处 `genVerifyDecl`；`#pim.contraction` / `#pim.act_spec` / `#pim.phase_spec` / `#pim.datapath` |
| 数据类型 | **覆盖** | `#pim.quant_spec`（granularity / axis / groupSize / dataExt / role / fpDtype / range）；`#pim.weight_binding.elemBits`（`PIMAttrDefs.td:659`：模型权重 4、当权重用的激活 8），int4 有明确表达 |
| Placement | **覆盖** | 4 个内存空间（`#pim.wram` / `#pim.mram` / `#pim.l1` / `#pim.l2`，各带 verifier）+ 功能单元枚举（`nmu`=0 / `vpu`=1 / `cstl`=2 / `dma`=3 前四序号冻结 + cstl 细分块）+ `pim.dpu_id` 算子 + `#pim.tasklet_tiled.dpusPerDevice` |
| Memory Layout | **在其负责的层级（WRAM/MRAM）覆盖** | `!pim.memdesc<shape, memorySpace>` + `#pim.tasklet_tiled`（sizePerTasklet / taskletsPerDpu / dpusPerDevice / order）+ 分配算子 `alignment` + 模块属性 `pim.dma-align` + DMA 的 `contiguous_dim`/`elem_stride`。该层级排布连续（§4.7.3），故无需多维步幅 |

**所以 P1-1 的缺口不在表达能力，而在「图编译器没把四维写进去」这个动作。** 这与第一版的结论一致，但本版是按层级逐条验证得出，而非从单个字段外推。

一条旁证：`pim.dpu_id` 的 ODS 注释（`PIMOps.td:57-59`）与 `dpusPerDevice` 的注释同调，都明写为「a future graph-level compiler that maps shards onto DPUs」预留——**Placement 与 Memory Layout 的图编译器接口是设计时就留好的，等的就是本轮要做的写入侧**。

#### 4.7.5 传递接口已验证可用（零 ODS 改动）

三组 `triton-opt` 实验确认 `dpusPerDevice` 全链路就绪：

**实验一：非全 1 值 parse 并原样打印回来**

```mlir
#pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [1, 4],
                    dpusPerDevice = [2, 1], order = [1, 0]}>
// triton-opt 输出与输入完全一致
```

**实验二：verifier 正确拒绝秩不匹配**（这是 P1-3 校验能力的来源）

```
输入 dpusPerDevice = [2]（rank-2 张量）
error: sizePerTasklet, taskletsPerDpu, dpusPerDevice and order must all have
       the same rank; got 2, 2, 1, 2
```

**实验三：穿过完整 pass 链后存活**

```bash
triton-opt probe.mlir -pim-fuse-activation -pim-expand-phases
# 输出仍含 dpusPerDevice = [2, 1]
```

结论：**P1-1 的 FlagTree 侧改动为零**，parser（`Dialect.cpp:69-107`）、printer（`:109-120`）、verifier（`:122-132`）全部就绪。

#### 4.7.6 printer 省略规则给出干净的验收方法

`Dialect.cpp:114-116`：

```cpp
  // Elide the all-ones default, keeping single-DPU kernels readable.
  if (!llvm::all_of(getDpusPerDevice(), [](unsigned v) { return v == 1; }))
    printer << ", dpusPerDevice = [" << ArrayRef(getDpusPerDevice()) << "]";
```

实验确认全 1 时该字段完全不出现在输出中。**验收判据因此无歧义**：文本里出现 `dpusPerDevice` ⟺ 非默认值 ⟺ 图编译器确实写入了跨 DPU 决策。不需解析数值，只需检查字段是否出现。

#### 4.7.7 图编译器侧的写入点

实测发射 pimir 是**纯字符串拼接**（`oplevel_emitter.py`，404 行），两个精确插入点：

| 插入点 | 位置 | 现状 | 动作 |
| --- | --- | --- | --- |
| 张量类型 | `:138` `_tensor()` | `f"tensor<{'x'.join(...)}x{dtype}>"` | 加 `layout=""` 关键字参数；非空时追加 `, #pim.tasklet_tiled<{...}>` |
| 模块属性 | `:399-402` | 仅 `pim.target` + `pim.rtl-version` | 补 `pim.num-dpus` / `pim.num-tasklets` / `pim.dma-align` 等，对齐 A 路富度 |

对比两条路径的模块头（实测）：

```mlir
// B 路（图编译器手写，主路）——信息最少
module attributes {pim.target = "pim:v1", "pim.rtl-version" = "..."} {

// A 路（FlagTree 从 TTIR 降级）——模块属性齐备
module attributes {"pim.dma-align" = 64 : i32, "pim.mram-bytes" = 4294967296 : i64,
                   "pim.num-dpus" = 8 : i32, "pim.num-tasklets" = 4 : i32, ...} {
```

`_tensor()` 的 `layout` 默认空字符串——单 DPU 路径输出与今天**逐字节相同**，这是产物不变的保证。

布局编码的字段来源：

| MLIR 字段 | 来源（统一 IR） |
| --- | --- |
| `dpusPerDevice` | `spec.placement` + `spec.shard_map` 的 DPU 数与切分维 |
| `sizePerTasklet` / `taskletsPerDpu` | `PIMHardwareConfig.num_tasklets` + 本地分片形状 |
| `order` | `elem_strides` 推出的维序（步幅最小的维在最前）；空步幅按行主序 |

`elem_strides` 经 `#pim.placement` 的 `order` 下发，A 路的 `-pim-explicit-dma` 按它重算 `elem_stride`（`tests/test_pimir_layout.py` 有「只改排布、DMA 步幅必须变」的用例）。下发的是维序不是步幅数值，不新增 ODS 字段。

#### 4.7.8 顺带补齐两个死字段（需求 P2-1 前两条）

| 字段 | 现状 | 动作 |
| --- | --- | --- |
| `pim.rope` 的 `subBlocks`（`PIMOps.td:1122`） | 有 verifier 无生产方；C++ `order[]`（`Ops.cpp:1032`）与 Python `ROPE_UNITS`（`gml_hw_constants.py:244`）**6 个名字逐字相同**，C++ 注释明写要求两处同步 | `_emit_rope()`（`oplevel_emitter.py:224`）补写，值取 `ROPE_UNITS`，**必须保序**（C++ 注释：「顺序就是语义：读回侧按位置展开，乱序会让整块字段错位而不报错」） |
| `pim.transpose` 的 `purpose`（`PIMOps.td:1407`） | 定义在案，无写入点 | 发射 transpose 时补写 |

#### 4.7.9 改动清单

| 仓库 | 文件 | 动作 |
| --- | --- | --- |
| **FlagTree** | 方言定义（`.td` / `Dialect.cpp`） | **零改动**（§4.7.4、§4.7.5 实测） |
| **FlagTree** | `test/Dialect/TritonPIM/` | 补 lit 用例：`dpusPerDevice` 非全 1 的往返（仅补测试，不改方言） |
| pim-compiler | `opcompiler_bridge/oplevel_emitter.py:138` | `_tensor()` 加 `layout` 参数 |
| pim-compiler | `opcompiler_bridge/oplevel_emitter.py:399-402` | 模块属性补硬件配置 |
| pim-compiler | `opcompiler_bridge/oplevel_emitter.py:224` | `_emit_rope()` 补 `subBlocks`（保序） |
| pim-compiler | `contracts/mlir_layout.py` | **新增**：四维 → `#pim.tasklet_tiled` 文本生成（约 80 行） |
| pim-compiler | `tests/test_pimir_layout.py` | **新增**：多 DPU 时字段出现、单 DPU 时省略、往返一致、穿 pass 链存活、`subBlocks` 保序 |

**重建要求**：虽然方言无改动，但阶段五验证前仍建议重建 FlagTree 以确保 A/B 两路同源（需求 §3.2）。仓内探针 `_check_inprocess_matches_triton_opt()` 本轮实测通过。

### 4.8 P1-2：PIMMLIR 回传对后续 pass 生效

#### 4.8.1 实测：算子编译器回传的到底是什么

本轮拿真实样本跑通了一次完整回传，结果修正了「回传的是 tile 尺寸与 WRAM 占用」这个直觉判断。

取一个多相样本 `.opcompiler_cache/3222b9dd5bb95eaa.pimir.mlir`，过完整 pass 链：

```bash
triton-opt 3222b9dd5bb95eaa.pimir.mlir -pim-fuse-activation -pim-expand-phases
```

| 项 | 输入（图编译器发的） | 输出（算子编译器处理后） |
| --- | --- | --- |
| op 序列 | `reduce_axis` `eltwise` **`lut`** `reduce_axis` `lut` `eltwise`（6 个） | `reduce_axis` `eltwise` `reduce_axis` `lut` `eltwise`（**5 个**） |
| 相位号 | `index = 0,1,2,3,4` | `index = 0,`**缺 1**`,2,3,4` |
| 模块属性 | `{pim.target = "pim:v1"}` | `{pim.target = "pim:v1"}`（**未变**） |

**两条关键结论**：

1. **算子编译器的产出是「融合决策」**：`-pim-fuse-activation` 把第一个 `pim.lut` 折进了前驱 op，于是 op 数 6→5、相位号 1 消失。这正是 GML 的 `*_phase_N` 字段套数由算子编译器决定的机制。
2. **B 路（图编译器手写的主路）模块头不长新属性**。`pim.tile-k/m/n`、`pim.wram-bytes-used` 这些只出现在 A 路（FlagTree 从 TTIR 降级）。所以 P1-2 的回传载体**不是模块属性，而是 op 属性与 op 结构本身**。

这解释了为什么既有的 `phase_source` 是按「函数名 → 相位序列」组织的（`phase_plan.py:197` `parse_phase_plans`）——它解析的正是展开后 IR 的 op 属性。

#### 4.8.2 既有反向通道的完整机制（要复用的范式）

| 环节 | 位置 | 做什么 |
| --- | --- | --- |
| 采集入口 | `phase_source.py:167` `phase_source_from_graph(gm)` | 发射 pimir → 跑 `triton-opt` → 解析 |
| 解析 | `phase_plan.py:197` `parse_phase_plans(text)` | 从展开后 IR 读出 `{函数名: PhasePlan}` |
| 载体 | `phase_source.py:68` `PhaseSource` | `by_node` / `ops` / `mlir` / `expanded` |
| 属性访问器 | `:86` `rtl_version`、`:90` `plan`、`:94` `phase_count`、`:98` `phase_value` | 从 `expanded` 文本按需提取 |
| 消费 | `gml_bridge/export.py:88,105,149,155` | `phase_source=None` 时退回静态表 |
| 判据 | `tests/test_gml_depends_on_opcompiler.py:141-148` | 断言「带与不带产出不同」 |

`phase_value` 的 docstring 说清了这个范式的价值：

```
GML 的值字段（flp_min_exp / kantor_mode / activation_mode 这类）原先整族查
常量表，算子编译器改了值 GML 也一个字节不变、两边都不报错。取值时优先用
这里给出的，取不到才退回常量表——表的定位是**缺省值**。
```

**这正是 P1-2 要的「回传生效」语义**：回传值优先、静态表兜底。本设计不发明新机制，扩展这一个。

#### 4.8.3 解决方案：扩展 `PhaseSource`，不碰 node.meta

回传**不能写回 `node.meta`**。理由是时序：回传发生在融合之后、序列化之前（`scripts/export_gml.py:648-665`），此刻图已是 `STAGE_FUSED` 终态，二次修改会破坏「融合结果即最终语义」的不变式，且 `export_graph` 本身不幂等（注释实测第二次 206 节点而非 200）。

所以沿用 `PhaseSource` 的旁路对象形态：

```python
# contracts/ir_payloads.py（新增载荷类型）
@dataclass(frozen=True)
class LayoutFeedback:
    """算子编译器对一个算子的布局与资源决策。

    与 PhasePlan 并列：PhasePlan 回答「这个算子分几相、每相多少字节」，
    本结构回答「它实际怎么摆、占多少 WRAM」。

    三个字段都可能是 None —— 算子编译器没算出来（或这条路径不适用）时
    不造假值，消费方据此退回静态规则。这与 phase_value() 的
    「取不到才退回常量表」是同一口径。
    """
    tile_shape: tuple[int, ...] | None = None   # 实际 tile 形状（来自 pim.tile-*）
    wram_bytes_used: int | None = None          # 实际 WRAM 占用
    mram_bytes: int | None = None               # 单台 MRAM 预算回显（pim.mram-bytes），不是实测占用
```

回传不挂在 `PhaseSource` 上。这些属性只在 A 路（过 `-pim-tile-to-budget`）
出现，而 `PhaseSource` 只在 B 路产生，挂上去端到端恒为空。载体收敛为契约层的
一个解析函数 `contracts/ir_payloads.py::layout_feedback_of_module`，谁拿到 A 路
文本谁调用。

#### 4.8.4 三个消费点，每个都遵循「回传优先、静态兜底」

需求 P1-2 要求每个回传字段能指出生产方与消费方各一处。

| 字段 | 生产方 | 消费方 | 消费逻辑 |
| --- | --- | --- | --- |
| `mram_bytes` | `layout_feedback_of_module`（A 路模块头的 `pim.mram-bytes`，是预算回显，不是占用） | `genesim_bridge/ir_cost.py` | 与回传的单台占用对照，超预算记一条 note |
| `wram_bytes_used` | 同上（`pim.wram-bytes-used`） | `genesim_bridge/ir_cost.py` | 超 WRAM 预算记一条 note |
| `tile_shape` | 同上（`pim.tile-m` / `pim.tile-n`） | `genesim_bridge/ir_cost.py` | 取 `tile_n` 进成本，再进仿真 sidecar 的 `kernel_tile_n` |

消费侧写法（以 GML 缓冲字节数为例）：

```python
# gml_bridge/from_fx.py
fb = phase_source.layout(node.name) if phase_source else None
if fb is not None and fb.mram_bytes is not None:
    nbytes = fb.mram_bytes                      # 算子编译器的实测值优先
else:
    nbytes = bytes_of(detail.local_shape,        # 退回图编译器的静态计算
                      dtype_bytes(spec.dtype), detail.elem_strides)
```

**产物不变的保证**：B 路模块头当前不带 `pim.mram-bytes`（§4.8.1 实测），所以 `fb.mram_bytes` 恒为 `None`，全部走 `else` 分支——与本轮改动前逐字节相同。回传通道建好了但当前不改变产物，这正是「先立管道、后灌数据」的安全顺序。

#### 4.8.5 验收判据落地

```python
def test_every_feedback_field_has_a_producer_and_a_consumer() -> None:
    """P1-2 的核心判据：不允许只写不读或只读不写的字段。

    这是防 combine_mode 类腐化的执行点 —— 那个字段枚举定义完整、
    五处调用却都传空值占位，零消费者（需求 P2-1）。
    """
    import re
    from dataclasses import fields
    from pathlib import Path
    from contracts.ir_payloads import LayoutFeedback

    root = Path(__file__).parent.parent
    src = "\n".join(p.read_text(encoding="utf-8")
                    for d in ("gml_bridge", "memory", "genesim_bridge", "opcompiler_bridge")
                    for p in (root / d).rglob("*.py"))
    for f in fields(LayoutFeedback):
        writes = len(re.findall(rf"{f.name}\s*=", src))      # 生产：赋值
        reads = len(re.findall(rf"\.{f.name}\b(?!\s*=)", src))  # 消费：属性读取
        assert writes >= 1, f"{f.name} 没有生产方"
        assert reads >= 1, f"{f.name} 没有消费方（只写不读 = 死字段）"


def test_feedback_absent_reproduces_the_static_path_byte_for_byte() -> None:
    """没有回传时，产物必须与不带回传完全一致。

    照 tests/test_gml_depends_on_opcompiler.py:141-148 的既有范式。
    这条保证「建通道」这个动作本身不改变产物（需求 §5.2）。
    """


def test_feedback_present_changes_the_product() -> None:
    """人为注入一个 mram_bytes 值，产物必须随之变化。

    反面判据：若注入后产物不变，说明消费点没接上 ——
    通道白建了，就是 combine_mode 的重演。
    """
```

第三个测试是**变异测试**，也是本节最关键的一条：它证明回传真的「生效」，而不只是「存在」。

#### 4.8.6 改动清单

| 文件 | 动作 | 规模 |
| --- | --- | --- |
| `contracts/ir_payloads.py` | **新增** `LayoutFeedback` 与 `layout_feedback_of_module`（回传载体，不挂 `PhaseSource`） | ~40 行 |
| `genesim_bridge/ir_cost.py` | 成本模型消费 `tile_shape` / `wram_bytes_used` / `mram_bytes` | +10 行 |
| `tests/test_layout_feedback.py` | **新增**：生产消费对 + 空值等价 + 变异测试 | ~110 行 |

### 4.9 P1-3：四维可校验

#### 4.9.1 现状：校验分布极不均衡

| 维度 | 现有校验 | 缺口 |
| --- | --- | --- |
| Placement | **三层齐备**：`Placement.validate()`（`pim_tensor_spec.py:11-22`）、`TensorShardDetail.validate()`（`:34-42`）、`PIMTensorSpec.validate()`（`:54-72`）；另有策略契约 `graph/strategy.py:34-51`、`contracts/partition_plan.py:76-114` | 无 |
| 算子语义 | FlagTree 侧 34 处 `hasVerifier`；**Python 侧零校验** | 融合目标可引用不存在的算子而不报错 |
| 数据类型 | 仅 `data_extension()` 对未知 dtype 抛错（`gml_hw_table.py:244`）、`HardwareBudget` 对 `dma_align` 做 2 的幂校验 | **张量级 dtype 无校验**（因为本轮之前无载体） |
| Memory Layout | `align_up` 对 `align <= 0` 抛错（`kv_layout.py:29`）、`KVRegionSpec.validate()` | **排布层无校验**（同样因为无载体） |

#### 4.9.2 解决方案：扩展现有 `validate()`，不新建校验框架

按 `CLAUDE.md`「不预造抽象」，**不引入 Validator 基类或校验注册表**——现有的 `validate()` 方法模式已经够用，照它扩展即可。错误信息沿用 `contracts/` 的既有风格（中文 + `got {value!r}`）。

**（1）数据类型维度**

新增 `contracts/dtypes.py`：

```python
# 本轮不扩充集合（需求 §2.3）。int4 一字节存一个值、不打包——
# 这是实测事实（contracts/gml_quant.py:30-32），位宽表据此给 int4 也记 1 字节。
ELEMENT_DTYPES = frozenset({"int4", "int8", "int16", "int32", "float16", "float32"})

_DTYPE_BYTES = {
    "int4": 1,      # 不打包：weight_buffer 字节数 == 权重元素数（实测）
    "int8": 1, "int16": 2, "int32": 4, "float16": 2, "float32": 4,
}

def validate_dtype(name: str) -> None:
    if name not in ELEMENT_DTYPES:
        raise ValueError(f"未知元素类型 {name!r}，允许 {sorted(ELEMENT_DTYPES)}")

def dtype_bytes(name: str) -> int:
    """单个元素占几字节。取代散落各处的 `.meta["val"].element_size()`。"""
    validate_dtype(name)
    return _DTYPE_BYTES[name]
```

`PIMTensorSpec.validate()` 扩展（新增部分，现有逻辑不动）：

```python
    def validate(self) -> None:
        # ...现有 Placement / shard_map 校验保持原样...

        # 本轮新增：数据类型维度
        if self.dtype:                      # 空串 = 尚未填充（见 §4.9.4 的阶段语义）
            validate_dtype(self.dtype)
        if self.quant is not None:
            if not self.dtype:
                raise ValueError("给了 quant 布局却没有 dtype，两者必须同时填")
            # 量化布局只对定点类型有意义：fp16 权重不带 scale/zp（实测）
            if self.dtype.startswith("float"):
                raise ValueError(
                    f"浮点类型 {self.dtype} 不应带量化布局 {self.quant.granularity!r}")
            if self.quant.granularity == "per_group" and self.quant.group_size <= 0:
                raise ValueError(
                    f"per_group 量化的 group_size 必须为正，got {self.quant.group_size}")
```

**（2）Memory Layout 维度**

`TensorShardDetail.validate()` 扩展：

```python
    def validate(self) -> None:
        # ...现有 dpu_id / shard range / mram_offset / local_shape 校验保持原样...

        # 本轮新增：排布层
        if self.elem_strides:               # () = 行主序紧密（今天的隐含假设）
            if len(self.elem_strides) != len(self.local_shape):
                raise ValueError(
                    f"elem_strides 秩 {len(self.elem_strides)} 与 local_shape 秩 "
                    f"{len(self.local_shape)} 不符")
            if any(s <= 0 for s in self.elem_strides):
                raise ValueError(f"elem_strides 必须全为正，got {self.elem_strides}")
            # 步幅必须能容纳该维的实际长度，否则相邻行会重叠
            for axis in range(len(self.local_shape) - 1):
                span = self.local_shape[axis + 1] * (self.elem_strides[axis + 1]
                                                     if axis + 1 < len(self.elem_strides) else 1)
                if self.elem_strides[axis] < span:
                    raise ValueError(
                        f"第 {axis} 维步幅 {self.elem_strides[axis]} 小于内层跨度 {span}，"
                        f"相邻元素会重叠")
        if self.align_bytes:
            if self.align_bytes & (self.align_bytes - 1):
                raise ValueError(f"align_bytes 必须是 2 的幂，got {self.align_bytes}")
            if self.mram_offset % self.align_bytes:
                raise ValueError(
                    f"mram_offset {self.mram_offset} 不满足 {self.align_bytes} 字节对齐")
```

**「相邻行重叠」这条是本设计新增的实质校验**：它能抓住「步幅算小了」这类错误。今天没有载体，所以这类错误只会表现为数值静默出错。

**（3）算子语义维度**

`contracts/op_semantics.py` 内，模块加载时即校验登记表自身自洽：

```python
def _validate_registry() -> None:
    """登记表自洽性。模块导入时跑一次——契约错了就不该能 import 成功。"""
    names = [s.name for s in OP_SEMANTICS]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"算子登记表有重名：{sorted(dup)}")
    # aten 目标不能映射到两个算子，否则 OP_TYPES 派生结果取决于遍历顺序
    seen: dict[str, str] = {}
    for s in OP_SEMANTICS:
        for t in s.aten_targets:
            if t in seen:
                raise ValueError(f"aten 目标 {t} 同时映射到 {seen[t]} 与 {s.name}")
            seen[t] = s.name

_validate_registry()
```

`contracts/fusion_contract.py` 加「融合目标必须是已登记算子」的断言：

```python
def _validate_fusion_targets() -> None:
    known = {s.name for s in OP_SEMANTICS} | _ATEN_ONLY_TARGETS
    for group, label in ((FUSION_TARGETS, "FUSION_TARGETS"),
                         (GATE_TARGETS, "GATE_TARGETS")):
        unknown = {_name_of(t) for t in group} - known
        if unknown:
            raise ValueError(f"{label} 引用了未登记的算子：{sorted(unknown)}")
```

**（4）跨维一致性**

单维校验抓不到的是**维度之间的矛盾**。一个校验函数专门处理：

```python
# contracts/unified_ir.py
def validate_node_dimensions(node, *, stage: str) -> None:
    """一个节点上四维的交叉一致性。在 pass 出口按图调用。

    只校验该阶段应当已填的维度（见 DIMENSION_READY_AT），
    未到阶段的维度缺失是合法的，不报错。
    """
    spec = node.meta.get(SPEC_META_KEY)
    if spec is None:
        return                              # 未到 STAGE_SPECS，或 host 节点

    spec.validate()                          # 单维校验先跑

    # 跨维一：dtype 与 val 的真值必须一致（dtype 由 val 派生，不该漂移）
    val = node.meta.get("val")
    if val is not None and spec.dtype:
        actual = str(val.dtype).removeprefix("torch.")
        if actual != spec.dtype:
            raise ValueError(
                f"{node.name} 的 spec.dtype={spec.dtype!r} 与 "
                f"meta['val'].dtype={actual!r} 不符")

    # 跨维二：Placement 的切分维必须落在 local_shape 的秩内
    for dpu_id, detail in spec.shard_map.items():
        if detail.shard_dim >= 0 and detail.shard_dim >= len(detail.local_shape):
            raise ValueError(
                f"{node.name} DPU{dpu_id} 的 shard_dim={detail.shard_dim} "
                f"超出 local_shape 秩 {len(detail.local_shape)}")
```

**跨维一是本设计最重要的一条校验**：它守住「`spec.dtype` 是 `val.dtype` 的派生物、不是第二个真源」这个不变式（§4.2 的核心设计）。若二者漂移，说明有代码绕过派生直接改了 spec——正是 P0-5 要防的旁路。

#### 4.9.3 调用点设计

校验在**哪里跑**决定它有没有用。三个层次：

| 层次 | 调用点 | 覆盖什么 | 成本 |
| --- | --- | --- | --- |
| 构造时 | `PIMTensorSpec.validate()` / `TensorShardDetail.validate()` | 单个结构的字段自洽 | O(rank)，rank ≤ 4 |
| 契约加载时 | `op_semantics.py` / `fusion_contract.py` 的模块级 `_validate_*()` | 登记表自洽 | 一次性，import 时 |
| pass 出口 | `propagate_specs` 末尾、`mem_planner` 回填后 | 跨维一致性、全图覆盖 | O(节点数)，单次遍历 |

pass 出口的写法（`graph/spec_prop.py::propagate_specs` 末尾）：

```python
    # 出口校验：此刻全图应当满足 STAGE_SPECS 的不变式
    for node in nodes:
        validate_node_dimensions(node, stage=STAGE_SPECS)
    mark_stage(gm, STAGE_SPECS)
    return edges
```

**性能**：全部校验都是 O(rank) 的字段检查或单次节点遍历，无嵌套遍历、无重复计算。需求 §3.1 要求不引入可感知耗时——按 llama2 两层图约 200 节点估算，出口校验是 200 次 O(4) 检查，相对 177.50 秒的基线不可测量。

#### 4.9.4 「空值」的阶段语义（这是校验设计的关键分歧点）

新增字段都带默认值（`dtype=""`、`quant=None`、`elem_strides=()`、`align_bytes=0`），于是必须回答：**空值是「尚未填充」还是「已确认无此属性」？**

本设计的口径：

| 字段 | 空值含义 | 理由 |
| --- | --- | --- |
| `dtype=""` | **尚未填充** | 任何张量都有类型，空串只可能是没填 |
| `quant=None` | **已确认未量化** | 浮点张量本就不带量化布局，是合法终态 |
| `elem_strides=()` | **已确认行主序紧密** | 这正是今天 `bytes_of()` 的隐含假设，把它显式化为「确认连续」而非「未知」 |
| `align_bytes=0` | **已确认无额外对齐要求** | 与 `dma_align` 的全局要求叠加，0 表示不额外加严 |

**注意 `elem_strides=()` 这一条与 PIMMLIR 的口径相反**：PIMMLIR 的 DMA 属性明写「*absence carries meaning*: it says the transfer has not been proven expressible as a strided DMA, **not that it is contiguous by default**」（`PIMOps.td:142-145`）。

两者不同是**有意的**，因为回答的问题不同：

- 统一 IR 侧：图编译器**知道**张量怎么摆（它做的内存规划），所以空值是「确认紧密」。
- PIMMLIR 侧：算子编译器**需要证明**访问模式可被 DMA 表达，未证明就不能假设。

这条差异必须写进 `contracts/pim_tensor_spec.py` 的字段注释，否则跨仓阅读时必然误解。§4.7.7 的 `contracts/mlir_layout.py` 在生成 pimir 时也**不得**把 `elem_strides=()` 翻译成 DMA 的 `elem_stride` 缺省——那会把「确认连续」误传成「未证明」。

#### 4.9.5 反例测试清单

每维至少一组「非法输入必须抛错」，且每条都指明**这个错误今天会怎样静默发生**：

| 维度 | 反例 | 期望 | 今天的静默失败方式 |
| --- | --- | --- | --- |
| 数据类型 | `dtype="bf16"` | 抛错，列出 6 个允许值 | 需求 §2.3 不支持 bf16，但今天无载体也无校验，写进去不会报错 |
| 数据类型 | `dtype="float16"` + `quant=per_group` | 抛错「浮点不应带量化布局」 | fp16 权重被当定点处理，scale 文件多出无意义字节 |
| 数据类型 | `spec.dtype` 与 `val.dtype` 不一致 | 抛错 | **两个真源漂移**，下游按哪个算取决于代码路径 |
| 算子语义 | 融合目标引用未登记算子 | 抛错（import 时） | 融合表与算子表不同步，融合静默不发生 |
| 算子语义 | 同一 aten 目标映射两个算子 | 抛错（import 时） | `OP_TYPES` 派生结果取决于遍历顺序，不确定 |
| Placement | `Shard` 但 `dim=None` | 抛错（已有逻辑，补测试） | — |
| Placement | `shard_dim` 超出 `local_shape` 秩 | 抛错 | 切分维索引越界，本地形状算错 |
| Memory Layout | `elem_strides` 秩 ≠ `local_shape` 秩 | 抛错 | 步幅按错的维度解释 |
| Memory Layout | 步幅小于内层跨度（如 `[100, 1]` 配 `local_shape=[4, 200]`） | 抛错「相邻元素会重叠」 | **相邻行覆盖**，数值静默错误 |
| Memory Layout | `align_bytes=24`（非 2 的幂） | 抛错 | 对齐计算出错 |
| Memory Layout | `mram_offset` 不满足 `align_bytes` | 抛错 | DMA 未对齐访问 |

## 五、数据结构与接口变更汇总

### 5.1 新增文件

| 文件 | 职责 | 预计行数 |
| --- | --- | --- |
| `contracts/unified_ir.py` | 四维键的唯一登记表 + 3 个查询函数 | ~150 |
| `contracts/ir_payloads.py` | 载荷 dataclass（从 `graph/` 下移 6 个 + 新增 `LayoutFeedback`） | ~100 |
| `contracts/dtypes.py` | 元素类型名唯一真源 + 位宽查询 + 校验 | ~40 |
| `contracts/op_semantics.py` | 算子语义唯一真源 + 三种派生视图 | ~120 |
| `contracts/mem_layout.py` | `align_up` 唯一实现 + 步幅规则（承接编排器） | ~90 |
| `contracts/mlir_layout.py` | 四维 → `#pim.tasklet_tiled` 文本生成 | ~80 |

合计新增约 580 行，全部在 `contracts/` 内——符合「地基层集中」的既有分层。

### 5.2 修改的数据结构

| 结构 | 文件 | 变更 | 兼容性 |
| --- | --- | --- | --- |
| `PIMTensorSpec` | `contracts/pim_tensor_spec.py:45` | 补 `dtype: str = ""`、`quant: QuantLayout \| None = None` | **带默认值，现有构造点零改动** |
| `TensorShardDetail` | 同上 `:25` | 补 `elem_strides: tuple = ()`、`align_bytes: int = 0` | 同上；frozen 不变，回填仍用 `replace` |
| `LayoutFeedback` | `contracts/ir_payloads.py` | 新增，字段全可 None | 取不到即「没意见」，消费方退回静态规则 |
| `FusedTail` 等 6 个载荷 | `graph/*` → `contracts/ir_payloads.py` | 位置迁移，字段不变 | `graph/` 各 pass 改 import |

**全部变更都是「加带默认值的字段」或「移动位置」，没有删除或改签名**——这是产物逐字节不变的结构性保证。

**FlagTree 方言零改动**：四维按内存层级逐条核实后确认 PIMMLIR 在其负责的层级（WRAM/MRAM）上四维都已覆盖，缺的只是图编译器的写入动作（§4.7）。

### 5.3 接口变更

| 接口 | 变更 | 兼容性 |
| --- | --- | --- |
| `bytes_of(local_shape, itemsize)` | 加可选参数 `elem_strides=()` | 空值走原路径，逐字节等价 |
| `_tensor(shape, dtype)` | 加关键字参数 `layout=""` | 空值输出与今天完全相同 |
| `align_up(n, align)` | 位置迁移到 `contracts/mem_layout.py` | 两处调用方改 import；算法不变 |
| `MNEMONICS` / `_OPLEVEL_OPS` / `OP_TYPES` | 从字面量改为派生 | 派生结果必须与今天逐项相同（含 `MNEMONICS` 顺序） |

## 六、实施计划

### 6.1 阶段划分与依赖

按 §4.2.3 的依赖顺序，分五个阶段。每阶段结束都要跑全量回归（`pytest tests/ -q -k "not llama2_7b"`），基线 981 passed / 1 skipped。

| 阶段 | 内容 | 依赖 | 交付判据 |
| --- | --- | --- | --- |
| **一** | P0-1 契约收口：新增 `unified_ir.py` + `ir_payloads.py`；载荷下移；修 2 处裸字符串；删 `split_heads.py:176` | 无 | 键集合相等断言通过；回归全绿 |
| **二** | P0-2 dtype 载体 + `contracts/dtypes.py`；P0-3 `op_semantics.py` 四视图派生 | 阶段一 | 甲类 6 处取数改完（§4.5.1 实测精确命中 6 处，此前估的 9 处偏高）；四份清单派生结果与原字面量逐项相同；角色优先级保留；回归全绿 |
| **三** | P0-4 排布字段 + `mem_layout.py`；`align_up` 去重 | 阶段一 | 默认值等价性测试通过；回归全绿 |
| **四** | P0-6 编排器改消费（两步法：先对照、后删除） | 阶段三 | 步幅全等对照通过；**层参数文本逐字节不变** |
| **五** | P1-1 pimir 四维写入 + 两个死字段；P1-2 回传扩展 | 阶段一~三 | 多 DPU 时 `dpusPerDevice` 出现；**GML 逐字节不变**；变异测试通过 |

P1-3 的校验与反例测试**不单列阶段**——每个载体就位时同批补上（阶段一补 Placement 补测、阶段二补 dtype/算子语义、阶段三补 Memory Layout）。

### 6.2 每阶段的回归门槛

| 阶段 | 必须通过 |
| --- | --- |
| 一、二、三 | `pytest tests/ -q -k "not llama2_7b"` 全绿 |
| 四 | 上述 + `prepare_out` 层参数文本逐字节比对 |
| 五 | 上述 + GML 文本与全部 bin 逐字节比对 + GeneSim `./run.sh --test` |

阶段五需先重建 FlagTree（`bash 0-install-flagtree.sh`，自 `371691b` 起会自动同步进 PyTorch 环境，无需再跑 `2-install-pytorch.sh`）。

### 6.3 建议的提交粒度

每阶段至少拆成「新增契约文件」与「改调用方」两个提交，便于出问题时二分定位。`CLAUDE.md` 要求每次改动报告净增删行数。

## 七、验证方案

### 7.1 单元测试清单

沿用仓内既有的两种断言范式：集合相等断言（`tests/test_runtime_compiled_coverage.py:166`）与源码级正则扫描（`:163`）。

| 测试文件 | 用例 | 操作步骤 | 预期结果 |
| --- | --- | --- | --- |
| `test_unified_ir_contract.py` | 键集合恰好相等 | 正则扫全仓 `.meta[CONST]`/`.meta["lit"]`/`.meta.get(...)`，与 `unified_ir.py` 登记表求对称差 | 对称差为空集；非空时报出多/少哪个键 |
| 同上 | 裸字符串禁令 | 扫非测试代码的 `.meta["..."]` 字面量，排除 `val` 等基础设施键 | 命中集合为空 |
| 同上 | 载荷类型可查 | 对 15 个键各调 `dimension_of(key)` | 全部返回四维之一，无抛错 |
| 同上 | 未登记键抛错 | `dimension_of("not_a_key")` | 抛 `ValueError`，信息含允许的键名 |
| `test_dtype_carrier.py` | 不查 PyTorch 得 dtype | 构造 spec，读 `spec.dtype` | 与 `node.meta["val"].dtype` 派生值一致 |
| 同上 | 非法 dtype 抛错 | `validate_dtype("bf16")` | 抛错，信息列出 6 个允许值 |
| 同上 | `_BUFFER_DTYPES` 引用真源 | 断言它是 `ELEMENT_DTYPES` 子集 | 是子集 |
| 同上 | 编码表键是真源子集 | `DATA_EXTENSION` 与 `layer_fields` 的 `DT_*` 键集合 ⊆ 真源 | 是子集 |
| `test_op_semantics.py` | 三视图派生一致 | 派生出的 `_OPLEVEL_OPS`/`MNEMONICS`/`OP_TYPES` 与改动前的字面量逐项比对 | 完全相同 |
| 同上 | 角色优先级保留 | 断言 `role_to_gml()` 键集合 == {matmul1,matmul2,mask,softmax}，且与 `aten_to_gml()` 键集合不相交 | 两表分离；丢掉优先级会让逐头节点 `op_type` 改变 |
| 同上 | 新增算子只改一处 | 在登记表加一个算子，检查三视图是否自动包含 | 三视图都变化，无需改其他文件 |
| 同上 | 融合目标校验 | 构造引用未登记算子的融合目标 | 抛错 |
| `test_mem_layout.py` | 默认值等价 | `bytes_of(shape, itemsize)` vs `bytes_of(shape, itemsize, ())` | 两者相等，且与改动前数值相同 |
| 同上 | 步幅秩校验 | `elem_strides` 长度 ≠ `local_shape` 长度 | 抛错 |
| 同上 | 对齐 2 的幂校验 | `align_bytes=24` | 抛错 |
| 同上 | `align_up` 唯一实现 | 源码扫描 `def align_up` 定义处 | 恰好 1 处 |
| `test_stride_parity.py` | 步幅全等对照 | 对全部层×全部相位，IR 算出的 10 个 stride 字段 vs 编排器现算值 | 逐字段全等 |
| 同上 | `dq_p2` 特例保留 | Gn=32 / 86 / 1 三种输入 | 分别得 47 / 101 / 16 |
| `test_pimir_layout.py` | 多 DPU 时字段出现 | tp=2 切分，发射 pimir，检查文本 | 含 `dpusPerDevice` |
| 同上 | 单 DPU 时字段不出现 | tp=1，同上 | 不含（printer 省略全 1） |
| 同上 | 往返一致 | 发射的 pimir 过 `triton-opt` 再解析 | `dpusPerDevice` 值不变 |
| 同上 | 穿过 pass 链存活 | 过 `-pim-fuse-activation -pim-expand-phases` | 字段仍在 |
| 同上 | `subBlocks` 顺序 | 发射 rope，检查 6 个子块名 | 与 `ROPE_UNITS` 声明序逐字相同 |
| `test_layout_feedback.py` | 回传改变产物 | 同一份 IR，带 vs 不带回传各分析一次 | 产物不同（照 `test_gml_depends_on_opcompiler.py` 范式） |
| 同上 | 每字段有生产消费方 | 源码扫描 `LayoutFeedback` 各字段的读写点 | 每字段 ≥1 写 + ≥1 读 |
| `test_no_bypass.py` | 甲类取数清零 | 扫 `.element_size()`/`.dtype` 调用点，白名单仅 `graph/spec_prop.py` | 白名单外零命中 |

### 7.2 集成验证

| 项 | 命令 | 预期 |
| --- | --- | --- |
| 全量回归 | `python -m pytest tests/ -q -k "not llama2_7b"` | 981 passed / 1 skipped（不新增失败或跳过） |
| GML 逐字节 | 改动前后各跑 `scripts/export_gml.py`，`diff` 文本 + `cmp` 全部 bin | 完全一致 |
| 层参数逐字节 | 改动前后各生成 `prepare_out`，逐文件 `cmp` | 完全一致 |
| GeneSim | `cd /media/disk/fengjingge/src/genesim && ./run.sh --test` | 全部通过 |
| FlagTree lit | 对**源码树**跑（不是空的 build 目录） | 通过；`/dev/shm` 满时用 `unshare -Umr` |
| 编译期耗时 | 对比全量回归耗时 | 与基线 177.50 秒无可感知差异 |

### 7.3 不回归的重点监控项

| 风险点 | 监控方式 |
| --- | --- |
| `_stamp_dtypes` 的非拓扑序容错 | 该函数本轮不动（§4.2.1），但要跑 `tests/test_gml_from_fx.py` 确认未被间接影响 |
| `MNEMONICS` 顺序 | 专项断言（§6.1），顺序变化会改 GeneSim 产物 |
| `dq_p2` 步幅特例 | 专项断言三种 Gn 取值 |
| 融合 pass 顺序 | `gml_bridge/export.py:117-144` 的顺序约束不变，跑 `tests/test_fuse*.py` |
| frozen dataclass 回填 | `mem_planner.py:89/274/281` 三处 `replace` 仍生效 |

## 八、风险与应对

| # | 风险 | 影响 | 应对 |
| --- | --- | --- | --- |
| 1 | 载荷下移（A1）后 `contracts/` 依赖 `torch.fx.Node` 类型 | 地基层类型依赖扩大 | `contracts/fusion_contract.py:21` 已 `import torch`，非新增方向；`FusedTail.nodes` 仅作类型标注，不产生模块循环 |
| 2 | 派生 `_ROLE_OP_TYPES` 时丢掉「角色优先于 aten」的优先级 | 逐头节点 `op_type` 退回按 aten 判定，GML 产物变化 | `role_to_gml()` 与 `aten_to_gml()` **必须保持两张表分离**（`from_fx.py:136-138` 先查角色再查 aten）；专项断言两表键集合不相交（§4.3.5）。注：`MNEMONICS` 的顺序经实测**不敏感**，唯一使用点 `op_classify.py:283` 只把它拼进错误信息，成员判定用的是 `_OPLEVEL_IR` 字典 |
| 3 | 编排器 10 处 stride 改造改错 | 层参数文本变化 | 两步落地法（§4.6.3）：先全等对照再删除；实测特例（`dq_p2`、四个全网步幅、L2 公式）逐条列入测试 |
| 4 | `dpusPerDevice` 写入后 verifier 拒绝 | pimir 编译失败 | 已实验确认秩一致时接受、秩不符时报错（§4.7.5）；先跑 tp=2 最小样例 |
| 4b | **把 L2/DDR 级的步幅数值下发给 PIMMLIR** | 引入无消费方的字段（本设计第二版即犯此错） | 下发的只是维序（`order`），步幅数值仍由 PIMMLIR 的指针分析证明；`elem_strides` 不写进任何数值字段 |
| 5 | 甲乙丙三类取数点误判，把必须保留的改掉 | 切分推导崩溃（乙类）或 FX 契约破坏（丙类） | §4.5.1 已逐处分类；乙类的理由是 `local_shape` 是本地形状而非全局形状，测试白名单固化这个边界 |
| 6 | 新增字段的默认值选错，导致产物变化 | 违反 §5.2 硬门槛 | 全部新字段默认值都对应「今天的隐含行为」（`elem_strides=()` = 行主序、`layout=""` = 无编码、回传字段全 None = 走原路径） |
| 7 | 两条 pimir 路径不同源 | 假结果（A 路认新属性、B 路不认） | 阶段五前重建 FlagTree；仓内探针 `_check_inprocess_matches_triton_opt()` 本轮实测通过 |
| 8 | 回传通道推广后出现新死字段 | 重演 `combine_mode` 类问题 | P1-2 验收要求每字段指明生产+消费方各一处；`semantic_role`/`weight`/`shard_axis` 三个既有调试字段显式登记白名单 |
| 9 | 同步清单在两个安装脚本各存一份 | 改一处漏另一处 | 脚本注释已警示；改动时两处同改 |

## 九、对需求文档的修正记录

本轮读码核实发现需求文档有四处需要修正，均已在对应章节说明：

| # | 需求文档原文 | 实测修正 | 章节 |
| --- | --- | --- | --- |
| 1 | 全仓在用 **14 个** meta 键 | **15 个**。多出的是 `graph/split_heads.py:176` 把角色取值 `ROLE_SPLIT`（`"split"`）误当键名写入，全仓零读者。**本轮删除该行** | §4.1.1 |
| 2 | 9 个键散在 **5 个 pass 文件**（含 `strategy.py`） | `graph/strategy.py` 实测 `.meta` **零命中**，是纯切分数学，不在改造范围 | §4.1.1 |
| 3 | `_stamp_dtypes` 的沿边传播逻辑**上移**到统一 IR | **不宜上移**。它作用在 GML 节点（`gml_bridge/writer.py:21`）而非 FX 节点，调用点在 FX→GML 转换之后（`from_fx.py:2173`），依赖的 `op_type`/`output_buffer_dtype`/`kantor_mode` 在 FX 侧命中全为 0。改为「在其上游补图阶段 dtype 载体，该函数保持不动」——这样也更安全地保证 GML 逐字节不变 | §4.2.1 |
| 4 | 数据类型「六处收敛为一处」 | 六处**性质不同**：只有 `_BUFFER_DTYPES` 是真正的类型集合；`DTYPES` 是缓冲区约束、`DATA_EXTENSION` 与 `DT_*` 是两套不同的编码表（`DT_FP16=1` vs `DATA_EXTENSION["int8"]=1`）、位宽常量是量化算法参数、`val` 是派生来源。准确做法是「新增类型名真源，1 处改引用、2 处加交叉校验、3 处性质不同不动」 | §4.2.3 |

| 5 | （需求未提及）内存层次的分管边界 | 本项目内存层次由两个互不相交的组件分管：WRAM/MRAM 归图编译器+算子编译器（经 PIMMLIR），L1/L2/DDR 归编排器（**不经 PIMMLIR**）。四条实测证据见 §2.1.4。这条边界是判断 PIMMLIR 表达能力的前提——用错层级会得出相反结论 | §2.1.4 |

另有一处需求未提及的发现：`part_id`（`contracts/graph_meta.py:4`）在**生产代码中零读取**，仅测试断言使用。契约登记时必须显式标注，否则后续会误判为死字段而删除。

### 9.1 本设计自身的两次修正记录

P1-1（PIMMLIR 四维覆盖）的结论经过两次推翻，记录在此以免后续重蹈：

| 版本 | 结论 | 错在哪 |
| --- | --- | --- |
| 第一版 | 「FlagTree 零改动」 | **以点代面**：只验证了 `dpusPerDevice` 一个字段，就外推到四维覆盖 |
| 第二版 | 「Memory Layout 有缺口，需扩展 `#pim.tasklet_tiled` 加 `elemStrides`」 | **层级混淆**：用编排器的 `align16(W)+15`（L2/DDR 级）去论证 PIMMLIR（WRAM/MRAM 级）的表达能力。那 14 处 stride 全是 `DDR *`/`Data scale *`/L2 输出段字段，按 §2.1.4 的边界不经 PIMMLIR |
| **第三版（本版）** | 「FlagTree 方言零改动，缺的是图编译器的写入动作」 | 按内存层级逐条核实：MRAM/WRAM 级连续假设实测成立（`bytes_of` 是紧密乘积、该层级搜 `align16`/`+15` 零命中、非连续分片被 `_runs()` 展开为多段连续而非跨步视图） |

教训：**判断一个 IR 的表达能力，必须先确认待表达的信息是否真的流经它。** 第二版的证据本身都是真的（`memdesc` 确实不带 layout、三次 parse 尝试确实失败），但论证对象错了。

## 十、工作量估算

| 阶段 | 新增代码 | 修改代码 | 新增测试 |
| --- | --- | --- | --- |
| 一（P0-1） | ~250 行 | 7 文件（载荷下移 + 2 处裸字符串 + 删 1 行） | ~120 行 |
| 二（P0-2、P0-3） | ~160 行 | 6 文件 | ~180 行 |
| 三（P0-4） | ~90 行 | 4 文件 | ~100 行 |
| 四（P0-6） | — | 3 文件（`layer_fields.py` 为主） | ~150 行 |
| 五（P1-1、P1-2） | ~180 行 | 5 文件 + FlagTree lit 用例 | ~200 行 |

合计新增约 680 行生产代码、约 750 行测试。净增删行数按 `CLAUDE.md` 要求在每次提交时报告。

**注意**：阶段四虽无新增代码，但它改的是 1660 行的 `layer_fields.py`，是全部阶段中单文件风险最高的一处。

## 十一、已决策事项与待确认事项

### 11.1 本轮已决策

| # | 事项 | 决策 | 落在 |
| --- | --- | --- | --- |
| Q1 | `_stamp_dtypes` 是否上移到统一 IR | **不上移**。它是 GML 序列化阶段的补齐器，作用在 GML 节点（`gml_bridge/writer.py:21`）而非 FX 节点，依赖的 `op_type`/`output_buffer_dtype`/`kantor_mode` 在 FX 侧命中全为 0。改为在其上游补图阶段 dtype 载体，该函数保持零改动——这同时是 GML 逐字节不变的最强保证 | §4.2.1 |
| Q2 | 六处 dtype 定义如何收敛 | **按分类处置**：新增 `contracts/dtypes.py` 作为元素类型名唯一真源；`_BUFFER_DTYPES` 改为引用它；`DATA_EXTENSION` 与 `layer_fields` 的 `DT_*` 两处加「键 ⊆ 真源」交叉校验（两者编号口径不同，不能归一）；`DTYPES`（缓冲区类约束）、位宽定标常量、`node.meta["val"].dtype`（派生来源）三处性质不同不动 | §4.2.3 |
| Q3 | 契约文件组织形态 | **A1**：载荷 dataclass 下移到 `contracts/ir_payloads.py`，`graph/` 各 pass 改为 import。契约因此完整（键 + 类型同处，可静态校验），顺带修掉 `graph/fuse_pim.py:230` 的函数内 import | §4.1.2 |

### 11.2 待确认事项

| # | 问题 | 状态 |
| --- | --- | --- |
| Q4 | `docs/request-gml-align-20260928.md` §7.2 的 6 条甲方待确认事项（原 Q4~Q8、Q13）。本轮要求 GML 逐字节不变，理论上不触碰，但需核对 | 待确认（需求 Q5 继承） |
