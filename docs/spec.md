# 面向存算一体架构的大模型推理编译器技术方案

## 一、研究背景

大模型推理对内存带宽提出极高需求，传统冯・诺依曼架构存在显著的存储墙瓶颈；存算一体架构是突破存储墙约束的重要途径。然而当前主流开源编译器架构，例如：FlagOS、llama.cpp、Triton等主要支持GPU。GPU 的一个根本前提是**统一地址空间**，并且其数据排布和通信由较为成熟的硬件和运行时负责。而我们的目标硬件是存算一体架构，其核心特征如下：

- 计算核心是 AIPU（ARM 核，后续可能更换，待更新），在其基础上引入近存计算特性；
- 由多个 DPU 组成，每个 DPU **地址空间独立**，最高内存 8GB；
- 编程模型接近 UPMEM：主机是唯一的编排中心，DPU 只做"被喂数据 → 计算 → 被取回结果"。

```
                         ┌──────────────────────────────────┐
                         │              主机 (Host)         │
                         │  编排器 / 通信库 / 采样 / KV 管理  │
                         │        （唯一的编排中心）          │
                         └──────────────────────────────────┘
                          │      │        │      │      │
              显式 DMA ───┤      │        │      │      │─── 显式 DMA
              （唯一通路） │      │        │      │      │
                          ▼      ▼        ▼      ▼      ▼
                       ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
                       │DPU 0││DPU 1││DPU 2││DPU 3││ ... │
                       │AIPU ││AIPU ││AIPU ││AIPU ││     │
                       │近存 ││近存 ││近存 ││近存  │ │    │
                       │≤8GB ││≤8GB ││≤8GB ││≤8GB ││     │
                       └─────┘└─────┘└─────┘└─────┘└─────┘
                          ╳      ╳        ╳      ╳
                       DPU 之间无直连（不能互相通信）
```

与 GPU 的核心差异如下：


| 维度       | GPU                         | 存算一体                | 由此驱动的工作                          |
| ---------- | --------------------------- | ----------------------- | --------------------------------------- |
| 地址空间   | 统一，共享显存              | 每个 DPU 独立           | 数据放置、搬运必须显式管理              |
| 设备间通信 | 高速直连（NVLink/PCIe P2P） | **无直连，必须过 host** | 不能用集合通信库，自写 host 中转通信    |
| 内存管理   | CUDA allocator 自动         | 无 allocator，手动管理  | 本地内存生命周期自己管（尤其 KV cache） |
| 并行范式   | 对称 SPMD + 集合通信        | 主机星型编排（          | 编排器自写，不能套 vLLM 运行时          |
| 单设备容量 | 几十 GB                     | 单 DPU ≤ 8GB           | 图拆分是硬约束，权重必须切分            |
| 计算单元   | warp / SM / 统一显存        | ARM 核 + 近存           | kernel 层不能套 GPU 的 TTGIR 抽象       |

这套"独立地址空间 + 主机星型编排 + 显式数据搬运"的模型，和 GPU 的"统一地址空间 + 对称并行 + 自动集合通信"是两种范式。正是这些差异，驱动了后面 AI 编译器框架所有的改造工作。

## 二、研发目标

大模型可在存算一体硬件平台正常推理运行（硬件到位前先在 NumpyBackend 上验证功能正确性）。具体如下：

1. **大模型图拆分**：把模型切成能放进单个 DPU的子图；
2. **设备映射**：决定每个子图、每份权重放在哪个 DPU；
3. **编排与通信**：主机按顺序调度各 DPU，显式搬运数据；
4. 编译器：生成适用于存算一体架构的程序；
5. **结果合并与迭代**：收集各 DPU 的部分结果，合并后推进到下一步；
6. **模拟器接入**：在 GeneSim 性能模拟器上走通 `HuggingFace → PyTorch → FlagGems → FlagTree → GeneSim` 端到端流程，用编译器的真实 IR 产物替换 GeneSim 原有的硬编码算子成本，提升性能评估精度。

**分两阶段实施**：

* 第 1 阶段（本期）把序列长度 `max_seq` 定为编译期常量，prompt、激活、KV cache 都按 `max_seq` 满预留、满计算再用 mask 盖掉多余部分，目标是打通全链路；GeneSim 接入先接 TTIR 走通链路（访存侧近似），再随算子编译器产出的 pim mlir 补齐访存侧精度；
* 第 2 阶段在阶段1基础上支持序列变长，引入代价模型和自动切分。

## 三、整体方案概述

### 3.1 总体流程

本方案的目标，是将一个以 HuggingFace 格式给出的大模型，经过一条编译软件栈的分析与转换，最终生成能够在存算一体硬件平台上执行的推理程序（硬件到位前先在 NumpyBackend 上验证功能正确性）。整条链路的入口是 HuggingFace 模型，出口是自回归解码得到的 token 序列。

编译软件栈由三个组件构成：**图层编译器**、**算子编译器**与**编排器（运行时）**；此外还有一个横跨编译期与运行时的**通信库**为跨 DPU 数据搬运提供支撑。四者对应第四章的具体问题如下：图层编译器对应图拆分（问题 1）、设备映射与切分传播（问题 2）、内存静态布局（问题 8）；算子编译器对应算子实现（问题 5）；编排器对应主机编排（问题 6）与 KV cache 管理（问题 7）；通信库对应 redistribute 下降为显式 DMA（问题 3）。这四者构成模型真实执行、产出 token 的**执行栈**，其功能正确性在 NumpyBackend 上验证。

除执行栈外，本方案还需接入 GeneSim 性能模拟器（模拟器接入，问题 4），用真实成本替换其原有的硬编码算子成本，从而走通 `HuggingFace → PyTorch → FlagGems → FlagTree → GeneSim` 的端到端性能评估流程。 GeneSim 是**性能评估旁支**，不参与真实推理执行、不产出 token，与上述执行栈是两条并行链路。

贯穿整个软件栈的一条主线是**编译期决策、运行时执行的严格分离**。图层编译器与算子编译器工作在编译期，它们不执行任何真实计算，只对计算图做分析、切分、映射与内存规划，并将全部决策固化为一组静态蓝图；编排器工作在运行时，它不做任何切分或映射决策，只依据这组静态蓝图驱动多个 DPU 完成推理。

### 3.2 三大组件及其职责

**图层编译器（编译期，问题 1 / 2 / 8）。** 它是整条链路的第一级，输入是 `torch.export` 从 HuggingFace 模型导出的完整计算图，输出是一组供下游直接消费的静态蓝图。其职责分为三步：一是**图拆分**，为每个算子标注它在 DPU 上计算还是留在主机上计算，并把相邻且都落 DPU 的算子聚成子图；二是**设备映射与切分传播**，为每个子图确定具体的 `dpu_id`、为每份权重确定切分维度与方式，并沿计算图逐算子推导中间张量的布局，标出需要跨 DPU 通信的 redistribute 边；三是**内存静态布局**，为每个 DPU 的权重区、KV 区、激活区计算内存偏移，产出内存蓝图 `DPU_k.plan` 并做容量校验。图层编译器本身不做计算，只生成元数据与布局。它的策略是借用现成的编译期分析能力（PyTorch 的图基础设施、DTensor 的切分传播、ExecuTorch 的静态内存规划），自写的仅是面向存算一体的标注规则与规划 pass。

**算子编译器（编译期，问题 5）。** 它承接图层编译器下发的单个算子，将其编译成能在**单个 DPU** 上执行的 kernel 二进制。这里合并了两部分工作：一是**算子的数学逻辑**，直接复用 FlagGems 面向 GPU 的成熟实现，因其数学与硬件无关；二是**面向存算一体的访存与同步下降**，由自研的 `ttir→upmem` 桥完成，把全局显存访存改写为 MRAM 与 WRAM 之间的显式搬运、按 WRAM 容量重调 tile、补出 DPU 内的同步原语，再复用 Cinnamon 的 `upmem` dialect 及其 lowering 下降到厂商 SDK。算子编译器的关键特性是**只认本 DPU 的地址空间**：每个 kernel 只从给定的 MRAM 地址读输入、向给定的 MRAM 地址写输出，既不感知全局切分方式，也不生成任何跨 DPU 通信逻辑。

**编排器（运行时，问题 6 / 7）。** 它是自回归推理的实际执行体，定位是**执行者而非决策者**。它依据图层编译器给出的静态蓝图，在主机上驱动多个 DPU 协同完成推理，负责的事务包括：按拓扑序调度各 DPU 的 kernel；对同一 DPU 上的连续算子直接顺序下发，使中间结果留在本地 MRAM、不返回主机；遇到 redistribute 边时经主机中转搬运数据；维护跨 DPU 的同步；管理各 DPU 本地内存的生命周期，尤其是跨 step 存活的 KV cache；用主机侧算子为未落 DPU 的节点兜底（即 host 胶水）；并在 step 之间推进序列状态、执行采样。它自写执行器主体，同时借用 ExecuTorch 的分区与委派模式作为"DPU 子图加主机胶水"的骨架。

**通信库（横跨编译期与运行时，问题 3）。** 由于 DPU 之间没有直连、且不适用 GPU 的集合通信库，图层编译器标出的每一条 redistribute 边，都必须下降为一段经主机中转的显式 DMA 序列。通信库正是这些 DMA 模板的集合，向上以对齐主流集合通信语义的原语（如 all-reduce、all-gather）供编排器调用，向下架在厂商 SDK 的 DMA 原语之上。需要强调其定位：**通信要不要发生、属于哪种类型、涉及哪些 DPU，全部在编译期定死，通信库自身不做任何决策，只负责执行。**

GeneSim 性能模拟器（本方案负责接入适配，问题 4）。GeneSim模拟器能按模型配置展开算子图、生成动态请求 trace，并做调度、roofline 时延估算与 PIM trace 生成。本方案将AI编译器对其适配和接入。

### 3.3 组件间的关系与接口契约

三个组件之间存在两类关系，分别对应两条不同性质的接口。

**图层编译器与算子编译器：编译期两级之间的双向接口。** 二者都工作在编译期，但分工不同：图层编译器做"图级"决策，算子编译器做"算子级"实现。图层编译器向算子编译器**下发**每个落 DPU 算子的编译契约——算子的数学类型、本 DPU 上的本地分片 shape 与 dtype、该算子输入输出在 MRAM 中的布局、以及 WRAM 的 tile 约束；算子编译器向图层编译器**回传**编译结果的描述符——生成的 DPU 二进制、tasklet 数、WRAM 占用、以及它期望的 MRAM 输入输出布局。这条接口以**单个算子**为粒度，而非子图：图层编译器只把单 DPU 的本地视图传下去，算子编译器无须理解全局切分与子图概念，二者由此解耦——切分策略调整时，只要本地 shape 不变，kernel 无须重新编译。

**图层编译器与编排器：编译期决策到运行时执行的单向蓝图下发。** 图层编译器把全部决策固化为一组静态蓝图，一次性交给编排器；编排器不做任何切分决策，只照蓝图执行。这组蓝图包括：标注完备的静态图（每个节点带有 device、dpu_id、part_id、placement、redistribute 等元信息）、描述每个张量分布的 `PIMTensorSpec`、逐条 redistribute 边的通信计划表、以及每个 DPU 的内存蓝图 `DPU_k.plan`；再加上算子编译器产出的 kernel 二进制，共同构成编排器运行时"吃"的全部输入。这组蓝图在编排器启动时被一次性编译为一份显式的执行命令 DAG——`ExecutionPlan`（结构定义见问题 6 三.（2）），DAG 中每条命令携带精确到地址级的等待列表；编排器运行时只解释这份 DAG 并按命令顺序发起 launch 与 DMA，不再需要在标注图、通信计划表与内存布局表三者之间做任何隐式的依赖推断，这一编译动作本身不引入新的切分或映射决策，只是把已有决策展开为可直接执行的形式。

**通信库横跨两侧。** 通信计划在图层编译器生成，执行在运行时由编排器通过通信库完成：编排器遍历静态图，遇到 redistribute 边时读取对应的通信计划表项，调用通信库的相应原语，原语内部再展开为对厂商 SDK 的 DMA 调用序列。

**GeneSim 接入与算子编译器：共用 FlagTree 前端的单向成本抽取。** 问题 4 与算子编译器（问题 5）**共用同一套 FlagTree 编译前端**，但用途不同，问题 5 把算子编成可在 DPU 上真执行的 kernel 二进制（面向执行），问题 4 只从编译产物中抽取成本元数据喂给 GeneSim（面向仿真估算）。这条接口是单向的——问题 4 读取算子编译器产出的 IR（第 1 步用 FlagTree 原生 TTIR，第 2 步用问题 5 的 pim mlir），从中统计 `flops` / `data_bytes`，回填 GeneSim `Operator` 的成本字段；不向执行栈回传任何东西，也不改动 GeneSim 的方法。这与"编译期决策、运行时执行"的主线正交：它既不改变任何切分 / 映射决策，也不参与运行时执行，是挂在算子编译器旁的一条独立评估支路。

图层编译器、算子编译器和编排器的逻辑关系如下：

```
图层编译器（编译期）  决定：算子归哪个 DPU、哪些边跨 DPU 通信 ← 通信"要不要"在此定死
   │
   ├─ 交给算子编译器：每个算子的 {数学类型, 本地 shape, dtype, WRAM tile}
   │                   → 编成单个 kernel 二进制（只认本 DPU MRAM 地址、零通信）
   │                   ← 回传 {二进制, tasklet 数, WRAM 占用, 期望 MRAM 布局}
   │
   └─ 交给编排器：标注完备的静态图（含 dpu_id 与哪几条边是 redistribute）
        ├─ 同一 DPU 的连续算子：顺序 launch，上个输出地址 = 下个输入地址，本地流转不返回主机
        └─ 遇 redistribute 边：copy_from → 主机归约 → copy_to（跨 DPU 两跳）
```

### 3.4 整体架构

综合上述职责与接口，整体架构按“**编译期一次决策，运行期照蓝图执行**”的主链组织；GeneSim 作为性能评估旁支挂在算子编译器旁，不参与真实推理。图中只保留职责边界和主要输入输出；详细数据契约列在图下。

```
                         HuggingFace 模型
                              │  torch.export
                              ▼  完整计算图（aten 算子，shape / dtype 附于 node.meta）
        ┌───────────────────────────────────────────────────┐
        │              图层编译器（编译期）                   │
        │   问题 1 图拆分       → device / part_id 标注       │
        │   问题 2 设备映射+传播 → dpu_id / placement / 通信边 │
        │   问题 7 KV 布局      →  build_kv_layout供问题8     │
        │   问题 8 内存规划      → 三区 offset，产出 DPU_k.plan │
        └───────────────────────────────────────────────────┘
              │  下发：单算子编译契约                ▲  回传：算子描述符
              │  {数学类型, 本地分片 shape, dtype,    │  {DPU 二进制, tasklet 数,
              │   MRAM 输入输出布局, WRAM tile 约束}   │   WRAM 占用, 期望 MRAM 布局}
              ▼                                      │
        ┌───────────────────────────────────────────────────┐         ┌──────────────────────────────────────┐
        │              算子编译器（编译期）                    │  共用    │   GeneSim 接入（性能评估旁支，问题4） │
        │   问题 5：ttir→upmem 桥 + Cinnamon upmem dialect     │ FlagTree │   从 IR 抽 flops / data_bytes      │
        │   单个算子 → 单 DPU kernel 二进制（只认本地、零通信） │──前端──▶ │   回填 GeneSim Operator 成本字段     │
        └───────────────────────────────────────────────────┘  IR     │   第1步 TTIR（不依赖问题5）→第2步pimmlir│
              │                                                         └─────────────────────────────────────┘
              ▼   编译期总产物一并下传运行时：
              ▼   标注静态图 + PIMTensorSpec + 通信计划表 + DPU_k.plan + kernel 二进制
        ┌───────────────────────────────────────────────────┐
        │         执行计划生成（编译期最后一步，问题 6）       │
        │   汇总以上产物 → 展开为显式命令 DAG ExecutionPlan   │
        │   （Command序列+逐条 waits） │
        └───────────────────────────────────────────────────┘
              │
              ▼
        ┌───────────────────────────────────────────────────┐
        │              编排器 / 运行时（执行者）               │
        │   问题 6 解释 ExecutionPlan：按 Command 顺序发起     │
        │   launch / DMA，按 waits 精确等待，不做切分决策       │
        │   问题 7 KV cache：按 head 切、本地驻留、跨 step 存活 │
        │   问题 3 通信库：Collective 命令 → 主机中转 DMA 序列  │
        └───────────────────────────────────────────────────┘
              │  厂商 SDK（类 UPMEM）：copy_to / copy_from / launch / sync
              ▼
              存算一体模拟器 / 硬件（多个独立地址空间的 DPU）
```

## 四、具体问题及方案

### 问题 1：图拆分

#### 一. 问题描述

单个 DPU 内存不超过 8GB，整个模型放不进去，所以第一步必须把计算图切成能放进单个 DPU 的子图。图拆分分析整张模型计算图，给每个算子贴一个标签——"这个算子在 DPU 上算"还是"留在 host 上算"，再把相邻且都落 DPU 的算子聚成一个个子图。本节不决定某个子图放到哪一个具体的 DPU，这些都是问题 2 的事。这一步是整条编译链路的入口：它产出的标注全图，是问题 2 做切分传播、问题 5 编 kernel、问题 6 编排器串图的共同起点。

#### 二. 图拆分功能与结构

本节只做两件事——**能力标注**（每个算子贴 `device` = dpu 或 host）和**连通分组**（相邻且都落 DPU 的算子编到同一个 `part_id`）。产物仍是同一张完整的图，只是每个节点多了两个标记。分组原理：是在图上找"连通块"。原理是：

> **相邻 + 都落 DPU → 归入同一组；一旦遇到 host 节点或断连 → 这一组到此为止，另起一组。**

下面举例说明：

```
[LayerNorm] → host   (不在白名单)
[Q 投影]   → dpu ┐
[K 投影]   → dpu ┼ 三个都落 DPU 且相邻   → 聚成第 0 组
[V 投影]   → dpu ┘
[softmax]  → host   (不在白名单，把链断开)
[O 投影]   → dpu     被 softmax 隔开     → 另起第 1 组
```

**`part_id` 是给这些组编的号**，写在每个节点上（`node.meta["part_id"]`），表示"这个节点属于第几个 DPU 子图"。同一个 `part_id` 的节点，保证是图上连通、且都落 DPU 。同一份分组信息有两个视角，下游各取所需：

- **`part_id`**（站在单个节点看）：我属于第几组。上例中 Q/K/V 投影的 `part_id` 都是 0，softmax投影是 1。
- **分区列表**（站在整体看）：每一组都有哪些节点的目录，形如 `[{part_id:0, 节点:[Q,K,V投影]}, {part_id:1, 节点:[O投影]}, ...]`。问题6编排器执行时用 **`part_id`** 比对一条边两端的组号，判断这条边是否跨组、要不要通信。

**输入与输出契约**：

```
输入：
  ① 整张计算图 gm —— torch.export(model, example_inputs) 产出的 GraphModule
     （节点是 aten 算子，边是张量依赖，shape/dtype 在 node.meta["val"] 里）
  ② DPU_LOWERABLE 白名单（自写）—— 哪些 aten 算子能落近存
              │
        问题 1（能力标注 + 连通分组）
              │
输出：还是同一张完整的图，只是每个节点多了两个标记 ——
  · node.meta["device"]  = "dpu" 或 "host"      （能不能落 DPU）
  · node.meta["part_id"] = 整数（仅 dpu 节点有） （属于第几个子图）
  外加一份分区列表：[{part_id, 节点清单}, ...]   （组的目录）

  · dpu 节点 → 按 part_id 聚成子图（交问题 2 定每个张量的分片如何散布到各 DPU、怎么切；交问题 5 编 kernel）

```

#### 三. 实现思路

本节的核心思路是复用 PyTorch FX 的 `CapabilityBasedPartitioner` 机制，只自写一个"算子支不支持近存"的判断规则。 "贴标签 → 找连通块 → 编号"这套流程正是 `CapabilityBasedPartitioner` 自动做的：本文只提供白名单，它遍历全图聚合连通块，还顺带做**环检测**（避免两个组互相依赖、排不出先后），这是它唯一复杂的部分，直接复用、不用自己写。自写的只有白名单查询规则，要点如下：

- 只有能落近存的算子（矩阵乘、向量乘、逐元素）才标记为可卸载，后续大部分算子都会走 DPU，第 1 阶段可先只支持简单算子；
- **能力判断只看算子类型，不看容量**（8GB 容量校验是问题 8 的事）；
- **分组时尽量减少跨 DPU 的边**：因为无直连，每条跨 DPU 边 = 两次 DMA + 一次 host 处理，切不好 host 会成为瓶颈。

**白名单宽窄的取舍**（`DPU_LOWERABLE` 里放哪些算子）：


| 方案               | 白名单                      | 优点                                                                                     | 代价                                                                           |
| ------------------ | --------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **窄（推荐先做）** | 只放 GEMM/GEMV/逐元素       | kernel 少、好验证；LayerNorm/softmax 留 host 当胶水（GQA 模型的 RoPE、RMSNorm 同属此类） | 每层进出 DPU 次数多，host 胶水偏厚                                             |
| **宽（后续优化）** | 再纳入 LayerNorm/softmax 等 | 连续算子聚成大子图、少进出 DPU                                                           | 这些算子要在近存实现（问题 5 kernel 变多），且规约类算子跨 head 时可能引入搬运 |

先窄后宽：先用窄白名单跑通全链路（落地路线第 1 步），再按 host 是否成为瓶颈，逐个把边界算子挪进 DPU。宽窄不影响分区器骨架，只改 `DPU_LOWERABLE` 这张白名单。第 1 阶段固定停在"窄"，宽化是 `[阶段2]`。

**代码骨架**（能力判断是自写的核心，只查类型、不查容量）：

```python
from torch.export import export
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupportBase

class DPUSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        if node.op != "call_function":
            return False
        # 只判断算子类型是否能落近存（GEMM/GEMV/逐元素）；不在这里判 8GB 容量
        return node.target in DPU_LOWERABLE

ep = export(model, example_inputs)
gm = ep.module()
# propose_partitions 只分组、不折叠，保留全图供问题 2 在完整图上传播
parts = CapabilityBasedPartitioner(gm, DPUSupport()).propose_partitions()
for pid, part in enumerate(parts):          # 把分组结果写回每个节点
    for node in part.nodes:
        node.meta["device"]  = "dpu"
        node.meta["part_id"] = pid
# 未被分到任何 part 的节点即 host 节点（问题 6 编排器的“胶水”）
```

还有一个问题，为适配存算一体 DPU 推理的张量并行约束，GPT2 融合 QKV 权重的原生连续布局，会导致多头 QKV 被拆分至不同 DPU，引发 KV Cache 跨设备传输、带宽爆炸的核心问题。最优工程方案是通过编译 Pass 做图层面优化，拆解融合 QKV 算子为三个独立 Q/K/V 线性算子，分别按注意力头做列并行切分，让单 DPU 持有完整头部的 QKV 权重，实现 KV Cache 本地驻留、无跨设备通信。同时支持性能优化分支，可通过编译阶段一次性权重重排，将权重布局改为单头 QKV 连续排布后再分片，兼顾分布式合规性与算子融合性能，仅需改动编译与权重加载逻辑，运行时无开销、改动成本最低。

四. 工作量与推进建议

图拆分是纯编译期 pass，自写量很小，连通分组、环检测在 `CapabilityBasedPartitioner` 里，本文补充一张白名单：

- 可借鉴Pytorch中的`partitioner.py` 整文件 400 行左右，唯一复杂处是"合并子图时的环检测"，**原样复用、不改**。自写的只有 `DPUSupport`（一张白名单查询，几十行）。
- 原版分区器采用贪心策略、只避免环、不看通信成本**，满足不了"减少跨 DPU 边"。要减少跨 DPU 边，需在它**外面**再套一层成本驱动的后处理 pass。这一步准备在 `[阶段2]` 完成，由问题 8 描述的顶层驱动 pass 统筹——读问题 1/2/8 共写的同一张 meta 图，把跨 DPU 边的 host 成本反馈给本节的分区器重切。**第 1 阶段不做此后处理，只用朴素贪心 + 手工指定的切分。**

**推进建议**：先窄白名单跑通全链路，再按 host 是否成为瓶颈逐个把边界算子挪进 DPU（宽化，`[阶段2]`）。

#### 五. 最终产物与依赖

图拆分是图层（编译期）的第一个 pass，产出一个 Python 文件：


| 文件           | 含义                                                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `partition.py` | 自写`DPUSupport`（白名单查询规则）+ 调 `CapabilityBasedPartitioner` 分组 + 把 `device`/`part_id` 打回 `node.meta`；`[阶段2]` 再加成本驱动后处理 |

它交付的产物是**同一张完整的图 + 每个节点的 `device` / `part_id` 标记 + 一份分区列表**，是问题 2（切分传播）、问题 5（编 kernel）、问题 6（编排器串图）共用的同一数据源。

依赖：

- **PyTorch（必需，唯一大依赖）**：`torch.export` 产出 `GraphModule` 与拓扑序；`CapabilityBasedPartitioner` 做连通分组 + 环检测；`OperatorSupportBase` 供继承。全部 import 即用，**不重编 PyTorch**。

### 问题 2：设备映射与切分传播

#### 一. 问题描述

上节已判定哪些算子落 DPU、把相邻且都落 DPU 的算子聚成若干子图。本节在此基础上，解决以下三个问题：

1. **设备映射**：每个子图对应的数据放到哪一个具体的 DPU（`dpu_id`）；
2. **权重切分**：每份权重沿哪一维、按什么方式切分到各 DPU；
3. **通信定位**：哪些边需要跨 DPU 通信。

本节只做切分规划，不生成任何指令。输出是一张带完整布局标注的图，即每个张量的分布方式、每条边是否需要重分布及其类型。

#### 二. 设备映射功能与结构

本节是图层的第二个 pass，位于问题 1 与问题 3 之间，包含4个功能：

- **初始切分**：为每份权重指定一个起始布局，作为后续自动推导的输入。
- **切分表达**：提供一套描述"张量在各 DPU 上如何分布"的标注语言，能够表达"整份复制""沿某维切成分片""部分和尚未合并"等分布形态。
- **切分传播**：从权重的初始布局出发，自动推导全图每一个中间张量的布局，并定位需要跨 DPU 重分布的边。
- **标注收口**：将上述结果统一收敛为挂在每个张量上的分布描述，作为下游唯一的数据源。

整体数据流如下：

```
问题 1 的标注全图（每节点带 device / part_id）
        │
   ① 初始切分：人工给定权重起始布局
   ② 切分表达：确定分布标注语言
   ③ 切分传播：逐算子推导 → 得到每张量布局 + 每条 redistribute 边
   ④ 标注收口：收敛为每个张量的分布描述
        │
        ▼
带完整布局标注的图
   ├─→ 问题 3（生成 DMA 序列）
   ├─→ 问题 6（编排器调度）
   ├─→ 问题 7（KV 布局）
   └─→ 问题 8（内存规划）
```

#### 三. 实现思路

本节的思路是：**算子的数学属性决定输入布局如何传递到输出布局。** 给定初始切分方式后，通过一套传播规则逐算子推导中间张量的布局；当某算子要求的输入布局与上游实际产出的布局不一致时，即在该边上标注一个 `redistribute`（数据重分布）。

（1）本节的输入与输出

- **输入**：问题 1 产出的标注全图——每个节点带 `device`（dpu / host）与 `part_id`（子图编号）标记。本节在完整图上工作：dpu 节点参与 placement 传播，host 节点作为一种 location 一并参与（详见下文（10）），其输出恒为 `Replicate@host`。
- **输出**：每个张量的完整 `PIMTensorSpec` 布局标注 + 每条边的 `redistribute` 标注（含转换类型与 location 端点）。

**（2）初始切分方法**

设备映射从一组人工指定的初始切分方式开始。第 1 阶段这些切分方式写成固定配置表，不做自动求优。各类张量的初始切分约定如下：


| 要决策的对象     | 切分方式                        | 由谁确定                                         |
| ---------------- | ------------------------------- | ------------------------------------------------ |
| 各类 Linear 权重 | 列切 / 行切交替                 | 查算子规则表，手工定                             |
| 注意力权重       | 按 head 切                      | 手工定                                           |
| Embedding 权重   | Replicate 或按 vocab 切         | 手工定                                           |
| KV cache         | 按 head 切 + pinned + 钉 dpu_id | 手工钉死（第一优先级）                           |
| 模型输入 X       | Replicate                       | 手工定（几乎固定）                               |
| 所有中间激活     | 无需指定                        | 传播器自动推导                                   |
| 算子落哪个 DPU   | 无需单独指定                    | 跟随输入分片，第 1 阶段由`shard_config` 手工钉死 |

表中只有权重和 KV cache 需要人工指定，中间激活的布局由传播器自动推导。KV cache 的切分（按 head 切、pinned、钉死 `dpu_id`）具有最高优先级，原因见问题 7。

需要明确设备映射在第 1 阶段的性质：每个分片映射到哪一个物理 DPU，**由 `shard_config` 手工指定并继承问题 1 的 `part_id`，不做自动求解**。传播器只负责推导中间张量的 placement 类型与 `redistribute` 位置，不决定物理放置。自动映射（成本模型求解与反馈重切）属 `[阶段2]`。本节名为"设备映射"，第 1 阶段提供的是映射的容器与手工填值，第 2 阶段才引入求解器。

此外，第 1 阶段采用张量并行，设备映射的单位是 **per-tensor 的 `shard_map`（分片映射到 DPU）**，而非把整个子图整体搬迁到某一个 DPU。一个列并行 Linear 是同时在所有 DPU 上各算一个分片，不存在"这个子图放在某个 DPU"的说法。`part_id` 在张量并行下表示"这批算子参与同一次并行切分"，而非"这批算子独占某个 DPU"。

**（3）切分表达：借用 DTensor 标注代数**

切分表达采用 PyTorch DTensor 的布局表达。DTensor 最有价值的是其**标注代数**与**切分传播规则**，二者均为纯编译期机制，可直接借用；但**不使用其运行时**——DTensor 运行时假设统一地址空间与集合通信，与存算一体的独立地址空间范式冲突。

DTensor 提供三种布局标注：

- `Shard(dim)`：张量沿某一维切开，每个 DPU 存储一个分片；
- `Replicate`：每个 DPU 存储完整副本；
- `Partial`：每个 DPU 存储的是一个"部分和"，尚未合并。

**（4）切分传播算法**

切分传播的输入仅是权重的初始布局；传播规则据此逐算子推出每一个中间张量的布局，以及需要重分布的位置。当某算子要求的输入布局与上游实际产出的布局不一致时，即在该边插入一次 `redistribute`。以两层 Linear 为例：

```
输入 X: Replicate
  │  Linear1（权重列切 Shard(1)）
  ▼
Y1: Shard(1)              ← 传播推出，无需通信
  │  Linear2（权重行切 Shard(0)）
  ▼
Y2: Partial()            ← 每个 DPU 持有部分和
  │  下一层要求 Replicate
  ▼
★ 要求 Replicate ≠ 上游 Partial → 插入 redistribute
```

上例中三个布局转换步骤（列切产出 `Shard`、行切产出 `Partial`、`Partial` 与下游要求的 `Replicate` 不一致而插入 `redistribute`）的数值推演过程，见**附录 A**。附录以最小形状逐元素追踪，说明 `Shard`、`Replicate`、`Partial` 三种布局的本质区别，以及 `redistribute` 为何在特定边上产生。

需要明确的一点：切分传播全程只计算元数据，不真正执行算子、也不搬运任何数据。所谓"重分布"并非算子执行的结果，而是沿传播规则逐步推导布局，直到出现"上游产出布局 ≠ 本算子要求布局"的位置，即在该边打上 `redistribute` 标注。整个过程只涉及两类计算：查算子的布局规则表、比较本算子要求的输入布局与上游实际产出布局之间的差异。产出是一张图，每条边标注了是否需要重分布及其类型。

**（5）传播的代码骨架**

传播的代码骨架如下（借 DTensor 传播器充当"分析器"，在其发起通信之前截断）：

```python
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._sharding_prop import ShardingPropagator

# DeviceMesh 的每个坐标 = 一个独立地址空间的 DPU
# 权重：Linear 按列 / 行切 -> Shard(1)/Shard(0)；embedding 通常 Replicate 或按 vocab Shard
prop = ShardingPropagator()
for node in dpu_graph.nodes:            # 图已是拓扑序，上游先算
    out_spec = prop.propagate_op_sharding(
        OpSchema(node.target, in_specs, {}))   # 按规则推出输出 placement
    node.meta["placement"] = out_spec          # 只记录，不执行
    # 本算子要求的输入布局 与 上游实际产出的布局 不一致处 = redistribute 点
    node.meta["redistribute"] = diff(in_specs, upstream_specs)
```

关键在于：截至 `node.meta["redistribute"]`，全部为编译期元数据。DTensor 默认会在此处立即发起集合通信，本方案必须在此截断，只保留标注。

**（6）在 DTensor 之上的扩展：三个缺口与四元组标注**

DTensor 的 `Shard/Replicate/Partial` 回答的是"一个张量此刻在各 DPU 上如何摆放"。作为**空间分布**的表达，它是完备的、可直接借用。但它面向 GPU 设计，默认了三件在存算一体上不成立的前提，因此仅借用这一层不够，需要在其之上再包一层存算一体专属的标注。

**三个缺口**，均源于"独立地址空间 + 无直连 + host 是唯一瓶颈"这一范式差异：


| 缺口                          | DTensor 的假设               | 存算一体的现实                                      | 不补的后果                                                  |
| ----------------------------- | ---------------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| **① 只描述空间，不描述时间** | 每个算子都可随时自由重分布   | KV cache 跨 step 存活、每步追加，一旦搬走即每步两跳 | 传播器给 KV 每步插一次 redistribute，数百步后 host 带宽耗尽 |
| **② 假定搬运成本均一**       | 各种重分布成本相近，不计代价 | 每次重分布都是`DPU→host→DPU` 两跳，host 是瓶颈    | 分区器无法区分不同切法产生的跨 DPU 搬运代价差异             |
| **③ 假定所有算子都上设备**   | 张量必在某设备上，只问如何切 | LayerNorm/softmax 等留在 host 充当胶水              | 无法表达"某算子根本不落 DPU"，capability 维度缺失           |

**解决方案：给每个张量挂一个"四元组"标注，DTensor 的 `placement` 只是其中一个字段；并给每条重分布边计算一笔 host 成本。** 四元组的最简形式如下（完整形式见下文（9）的 `PIMTensorSpec`）：

```python
@dataclass
class PIMTensorSpec:
    device:    "host" | "dpu"           # 来自问题 1 的能力标注（补缺口③）
    placement: Placement                # 借 DTensor：Shard/Replicate/Partial
    residency: "transient" | "pinned"   # 补缺口①：pinned = 跨 step 驻留、禁止搬运
    dpu_id:    Optional[int]            # pinned 张量钉死在哪个 DPU
```

**（7）执行顺序**

```
问题 1 的产出：全图 + 每个节点的 device/part_id 标记
   │
 ① 拿到完整图（缺口③已由问题 1 的能力标注解决）
   │   host 节点作为一种 location 参与传播，其输出恒为 Replicate@host
   │   本节在完整图上工作：dpu 节点传播 placement，host 节点承担合并与胶水
   ▼
 完整图（dpu 节点 + host 节点）
   │
 ② 先钉 KV，再传播（补缺口①）—— 对应关键结论"先定 KV 局部性，再让其他层迁就"
   │   先手动把 KV 标为 Shard(head) + pinned + 钉死 dpu_id
   │   再运行 DTensor 传播，遇到 pinned 张量绕开、不允许插 redistribute
   │   → 注意力的 Q 投影被推导为按 head 切，KV 永远本地驻留、零搬运
   ▼
 ③ 给每条 redistribute 边计算 host 成本（补缺口②）
   │   按问题 3 的四种类型 × 字节数 × 两跳，套 host-star 成本模型
   ▼
 产出：每张量的四元组标注 + 每条边的 DMA 成本
       下传给运行时：编排器据此调度、按 pinned 管理 KV、按边发起 DMA
```

**（8）算子布局规则表**

切分传播的核心是一张"输入布局 + 算子 → 输出布局"的推导规则表。下表给出主要算子类别在特定输入布局下的输出布局与重分布触发条件（下表由模型辅助生成，待审查）：


| 算子大类                                                               | 具体算子                            | 输入布局                                                   | 输出布局                | 核心规则说明                                                                                                                                                                                                                                                 | 重分布触发条件                                                                                                                                                                                                |
| ---------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **逐元素算子**（Element-wise）                                         | add、mul、sub、div等                | 两输入均为`Shard(d)`，且切分维度 d 一致                    | `Shard(d)`              | 逐元素算子仅做同位置元素运算，不跨维度交互，切分方式完全继承输入                                                                                                                                                                                             | 两输入切分维度不一致、或一个 Shard 一个 Replicate 时，需先对齐布局再计算；第 1 阶段默认将 Replicate 端本地切片为对应 Shard（从本 DPU 已有的完整副本中直接切出所需分片，只在本 DPU 内存内发生，零跨 DPU 通信） |
|                                                                        |                                     | 一输入`Replicate`，一输入`Shard(d)`                        | `Shard(d)`              |                                                                                                                                                                                                                                                              |                                                                                                                                                                                                               |
|                                                                        |                                     | 两输入均为`Replicate`                                      | `Replicate`             |                                                                                                                                                                                                                                                              |                                                                                                                                                                                                               |
|                                                                        |                                     | 两输入均为`Partial`（同规约类型）                          | `Partial`               |                                                                                                                                                                                                                                                              |                                                                                                                                                                                                               |
|                                                                        |                                     | 一输入`Replicate`，一输入`Partial`                         | `Replicate`（重分布后） | 两输入布局不一致，不能直接逐元素相加：`Partial` 是部分和，`Replicate` 是完整值。需先对 `Partial` 端插入 `Partial → Replicate` 的重分布，还原为完整张量后再相加                                                                                              | `Replicate` 与 `Partial` 相遇即触发 `Partial` 端的 all-reduce，还原后再算。这是每个 Transformer 层残差连接（`x + attn_out`、`x + mlp_out`）的固定路径                                                         |
| **矩阵乘法**（GEMM/GEMV/Linear）记`C = A @ B`A:[M,K], B:[K,N], C:[M,N] | torch.matmul、nn.Linear             | A:`Replicate`B:`Shard(1)`（权重按列切）                    | `Shard(1)`              | 对应 Megatron 列并行：每台 DPU 独立算出输出的一列分片，拼接可得完整结果，计算阶段零通信                                                                                                                                                                      | 权重切分维与输入切分维不匹配 contraction 规则时，需重分布输入或权重；第 1 阶段仅启用列切 + 行切的标准配对                                                                                                     |
|                                                                        |                                     | A:`Shard(0)`（激活按行切）B:`Replicate`                    | `Shard(0)`              | 激活行切与完整权重相乘，输出保留行维度切分                                                                                                                                                                                                                   |                                                                                                                                                                                                               |
|                                                                        |                                     | A:`Shard(1)`（激活沿 K 维切）B:`Shard(0)`（权重沿 K 维切） | `Partial(sum)`          | contraction 维（K）被切开，每个 DPU 仅算出部分和，最终需逐元素累加还原完整结果；对应 Megatron 行并行的 AllReduce 节点                                                                                                                                        |                                                                                                                                                                                                               |
|                                                                        |                                     | A:`Replicate`B:`Shard(0)`（权重沿 K 维切）                 | `Partial(sum)`          | 激活为完整副本、权重沿 contraction 维（K）切。每个 DPU 从本地完整激活中本地切片取对应 K 段（`A[:, k_i:k_{i+1}]`，零通信），与本地权重段相乘得部分和，输出 `Partial(sum)`。这是行并行的另一种合法入口，计算阶段零通信                                         | 无（激活本地切片，计算阶段零通信；部分和的合并在下游`Partial → Replicate` 边上完成）                                                                                                                         |
| **规约算子**（Reduce）                                                 | sum、mean规约维度为`reduce_dim`     | 输入:`Shard(d)`且`d ≠ reduce_dim`                         | `Shard(d)`              | 规约维度未被切分，每个 DPU 可独立完成本地规约，结果仍保留原切分                                                                                                                                                                                              | 切分维度恰好是规约维度时，每个 DPU 仅持有部分规约数据，输出为 Partial，需全量归约                                                                                                                             |
|                                                                        |                                     | 输入:`Shard(reduce_dim)`                                   | `Partial(sum/mean)`     |                                                                                                                                                                                                                                                              |                                                                                                                                                                                                               |
|                                                                        |                                     | 输入:`Replicate`                                           | `Replicate`             |                                                                                                                                                                                                                                                              |                                                                                                                                                                                                               |
| **Softmax**（单列，非普通规约）                                        | softmax规约维度为`reduce_dim`       | 输入:`Shard(d)`且`d ≠ reduce_dim`                         | `Shard(d)`              | 规约维未被切分，每个 DPU 在本地完整完成 softmax，结果保留原切分。典型场景：按 head 切分后，head 内的 softmax 本地执行，第 1 阶段恒走此格                                                                                                                     | 无（本地执行）                                                                                                                                                                                                |
|                                                                        |                                     | 输入:`Shard(reduce_dim)`                                   | 第 1 阶段不出现         | softmax 含沿规约维的 max 与 sum-of-exp 两次规约及指数重缩放，规约维被切时不能用单个`Partial(sum)` 表达，需 flash-attention 式合并（`Partial(max)` + `Partial(sum-exp)` + 重缩放）。第 1 阶段靠按 head 切保证规约维不被切、不触发此情形，通用支持留 `[阶段2]` | 规约维被切时不可按普通规约处理；第 1 阶段由切分契约排除此情形                                                                                                                                                 |
| **维度变换算子**                                                       | transpose、permute、reshape         | 输入:`Shard(dim_old)`                                      | `Shard(dim_new)`        | 切分标记随维度索引同步迁移。reshape 在切分维不被拆分或合并、或切分边界恰好落在新形状的维边界上时本地执行；第 1 阶段的切分契约（见下文（11））保证此条件成立                                                                                                  | 契约外的一般情形（切分维被卷入形变且边界不对齐）触发重分布为 Replicate，留`[阶段2]`                                                                                                                           |
| **拼接 / 拆分算子**                                                    | cat、chunk、split操作维度为`op_dim` | 输入:`Shard(d)`且`d ≠ op_dim`                             | `Shard(d)`              | 操作维度未被切分，每个 DPU 可独立完成本地拼接 / 拆分；操作维与切分维重合但切点落在分片边界上时同样本地执行（第 1 阶段契约保证融合权重的 split 点对齐 head 边界）                                                                                             | 操作维与切分维重合且切点不在分片边界时，该操作等价于 all-gather / split，属于 redistribute 范畴，不通过算子本身实现                                                                                           |
| **Embedding 层**                                                       | nn.Embedding权重形状`[V, H]`        | 权重:`Replicate`输入 ID:`Replicate`                        | `Replicate`             | 第 1 阶段推荐方案，实现最简单                                                                                                                                                                                                                                | 权重按 vocab 切分时需路由逻辑，第 1 阶段不支持                                                                                                                                                                |
|                                                                        |                                     | 权重:`Shard(1)`（按隐藏维切）输入 ID:`Replicate`           | `Shard(1)`              | 与 Linear 列切规则一致                                                                                                                                                                                                                                       |                                                                                                                                                                                                               |

**（9）布局标注的收口：完整的张量描述结构**

推导全部完成后，每个张量的布局与每条边的重分布标注统一收敛为挂载在每个张量（FX 图的边 / 节点输出）上的 `PIMTensorSpec` 结构。它是切分传播的唯一数据源，仅在图层内部使用，精确到每个 DPU 上的每一个分片；前文（6）的四元组是它的最简形式，完整定义如下：

```python
@dataclass
class TensorShardDetail:
    dpu_id: int               # 分片所属 DPU 编号
    shard_dim: int            # 沿哪一维切分（仅 Shard 有效）
    start_idx: int            # 该分片在全局张量中的起始索引（编译期常量）
    end_idx: int              # 该分片在全局张量中的结束索引
    local_shape: tuple[int]   # 该 DPU 上的本地张量形状
    mram_offset: int = 0      # 在本 DPU MRAM 中的地址偏移（内存规划后填充）

@dataclass
class PIMTensorSpec:
    device: Literal["host", "dpu"]              # 来自问题 1 的能力标注
    placement: Placement                        # DTensor 原生：Shard/Replicate/Partial
    residency: Literal["transient", "pinned"]   # 是否跨 step 驻留（KV cache 用）
    pinned_dpu_id: Optional[int]                # pinned 张量绑定的固定 DPU 编号
    shard_map: dict[int, TensorShardDetail]     # 核心：每个 DPU 对应的分片明细
    reduce_type: Optional[str]                  # Partial 对应的规约类型；第 1 阶段仅 sum/mean，softmax 型 partial（flash 式合并）留 [阶段2] 扩充
```

`redistribute` 的标注方式为：打在两个算子节点之间的边上，记录 `from_placement → to_placement` 的转换类型与 location 端点；一条边对应一个逻辑重分布动作（`DPU → host → DPU` 的两跳中转是其实现细节，留待问题 3 展开），是问题 3 生成 DMA 序列的输入依据。边标注结构如下：

```python
@dataclass
class RedistributeEdge:
    from_placement: Placement     # 上游产出布局，如 Partial(sum)
    to_placement:   Placement     # 下游要求布局，如 Replicate
    type: str                     # all_reduce / all_gather / all_to_all / scatter
    src_loc: dict                 # 数据来源端点：{"device": "dpu", "dpus": [...]} 或 {"device": "host"}
    dst_loc: dict                 # 数据去向端点：同上
    # 字节数由本节的分片 shape 算出；地址（mram_offset）由问题 8 的内存规划填充
```

`src_loc` 与 `dst_loc` 不额外采集，由边两端节点已有的 `PIMTensorSpec` 推出：`src_loc` 取上游节点的 `(device, shard_map.keys())`，`dst_loc` 取下游节点的同一对信息。其中 `device` 来自问题 1 的能力标注，参与的 DPU 集合来自 `shard_map` 的键集合。此推导在标注收口（第 ④ 步）完成，是纯编译期查表，无运行时输入。

需要明确 `RedistributeEdge` 的描述粒度：`src_loc` 与 `dst_loc` 只回答"哪些 DPU 参与这次重分布"，不回答"每个 DPU 具体搬运全局张量的哪一段、按什么顺序拼接、写入目标 DPU 的哪个偏移"。后一层信息由问题 3 在生成通信计划表时读取该边两端节点各自的 `PIMTensorSpec.shard_map`——其中每个分片的 `start_idx`、`end_idx`、`local_shape` 已在本节精确算出——按重分布类型展开得到，不在 `RedistributeEdge` 中重复存储，具体展开规则见问题 3。

这里`PIMTensorSpec` 在编译期完整计算并写入节点 `meta`，图优化、内存规划、调度表生成均基于此结构，运行时不再修改。它向下游分两路交付：

```
【图层（编译期）】  全量切分规则 + 完整 PIMTensorSpec + redistribute 边标注
   ├─ 提取本地 shape / 地址 → 【FlagGems / FlagTree】→ 单 DPU Kernel 二进制
   │                          （无全局切分感知，仅本地计算）
   └─ 提取调度表 + 通信表 + 内存计划 → 【编排器（运行时）】
                              （照表执行，无切分决策逻辑）
```

**（10）host 作为一种 location：合并与胶水的统一机制**

传播不将 host 节点排除在外，而是把 host 视为一种 location 一并纳入。`device=host` 的节点其输出恒为 `Replicate@host`，即 host 上持有完整张量。它承担两类作用：

- **胶水**：运行 LayerNorm、softmax、GELU、位置嵌入相加等留在 host、不落 DPU 的算子（GQA 模型的 RoPE、RMSNorm 同属此类）；
- **合并**：作为各 DPU 分片或部分和的物化与汇总点。`Partial → Replicate`（累加）与 `Shard → Replicate`（拼接）的动作本身即发生在 host（见问题 3 的下降形态：各 DPU 收集到 host、host 上累加或拼接、再回写各 DPU）。

传播器遇到 host 算子时，要求其输入为 `Replicate@host`，从而在上游边上自动触发 gather 或 reduce；host 算完后喂回 DPU 子图时，触发 scatter。因此"合并"与"胶水"由同一条"指向 host 的 redistribute 边"机制承担，图不再在 DPU 子图与 host 节点交界处断开，边界搬运也不再遗漏。

以注意力块为例：输出投影 `W_O` 的 contraction 维是被切的 head 维，`attn_out @ W_O` 产出 `Partial`；其下游的残差 add 与 LayerNorm 要求完整张量，于是在该边落地一条 `Partial → Replicate` 的 all-reduce，参与者为全体 DPU、去向为 host。这条边正是设备之间数据合并、进而驱动后续层推进的节点。

**（11）第 1 阶段切分契约**

`PIMTensorSpec` 在第 1 阶段不扩充语义，其正确性建立在以下一组不变量之上；`shard_config` 必须遵守，传播规则表也据此简化：

1. **单维切**：每个张量至多沿一个逻辑维切分（一维 DeviceMesh）。二维、三维张量并行留 `[阶段2]`。
2. **单段连续**：每个 DPU 持有恰好一段连续分片，`shard_map` 中每个 DPU 对应单个 `TensorShardDetail`；不支持 block-cyclic 或 strided 分布。
3. **全体参与**：每条 redistribute 的参与者为 mesh 内全体 DPU，不存在子集规约组。
4. **规约类型**：`Partial.reduce_type ∈ {sum, mean}`；softmax 型 partial 靠"按 head 切、规约维不被切"回避，不在第 1 阶段出现。
5. **切分对齐**：DPU 数取 2 的整数次幂，整除 head 数（GQA 按 KV head 数），且切点对齐 head 边界。由此保证 reshape、cat、split 均落在本地执行的合法情形。

上述不变量把复杂的切分场景排除在配置层之外，使第 1 阶段的规则表只需处理本地执行与标准 redistribute 两类情形。契约外的一般情形（二维切分、非连续分片、子集规约、通用 reshape、softmax 规约维被切）统一留 `[阶段2]`。第 1 阶段的固定模型为 **GPT-2**（HuggingFace `openai-community/gpt2`，标准 MHA decoder-only、绝对位置编码），其全部权重切分均落在本契约范围内；16 头可被 2/4/8 整除，切点对齐 head 边界即可覆盖 4/8 DPU 张量并行。规则表本身按标准 GQA / MHA decoder-only 通用编写（GQA 是一般情形、MHA 为其特例），因此后续宽化到 GQA 模型（如 Llama、Qwen 类）无需改动切分契约。

#### 四. 工作量与推进建议

本节是编译期 pass。连通分组与环检测等重活已在问题 1 完成；本节的主要自写量在于**自维护一张算子布局传播规则表，规则表这部分是净自写。按组件拆分代码量如下（均为 Python）：


| 组件                                                                      | 代码量（估）                  |
| ------------------------------------------------------------------------- | ----------------------------- |
| 初始切分配置加载（读固定`shard_config`，产出每份权重的 `PIMTensorSpec`）  | 约 150 行                     |
| 算子布局传播规则表（自写 GEMM/GEMV/softmax/逐元素/reshape 等 6~8 类规则） | 约 400~600 行，本节最大的一块 |
| 传播驱动（拓扑序遍历、查规则、比对上下游、打`redistribute` 标注）         | 约 200 行                     |
| KV 钉死与 pinned 绕行逻辑（先钉 KV、传播时遇 pinned 不插 redistribute）   | 约 150 行                     |
| `PIMTensorSpec` / `TensorShardDetail` 数据结构与 `shard_map` 展开         | 约 150 行                     |
| 成本模型 + 反馈重切（`[阶段2]`，第 1 阶段不做）                           | 约 300 行（暂不计入）         |

本节合计约 **1050~1250 行 Python**（不含 `[阶段2]` 的成本模型），其中自写算子规则表是主要工作量。

本节最独特、也最容易写错顺序的部分是"先钉 KV、再传播其余"。传播驱动的骨架如下（与（5）的 `ShardingPropagator` 代码骨架互补——（5）展示"借 DTensor 分析器"，此处展示（5）末尾推荐的"自维护规则表"落地形态）：

```python
def propagate_shardings(dpu_graph, shard_config, kv_nodes):
    # ① 先钉 KV：Shard(head) + pinned + 钉死 dpu_id，传播时不许被重分布
    for node in kv_nodes:
        node.meta["spec"] = shard_config.pin_kv(node)    # Shard(head) + pinned
    # ② 拓扑序传播其余节点（上游先算）
    for node in dpu_graph.nodes:
        if node.meta.get("spec"):                        # 已钉死（KV）→ 跳过
            continue
        in_specs = [n.meta["spec"] for n in node.args]
        out_spec = RULE_TABLE[op_class(node)](in_specs)  # 查自写规则表推出输出布局
        node.meta["spec"] = out_spec
        # ③ 上下游布局不一致处打 redistribute（上游为 pinned 则绕行、不插）
        node.meta["redistribute"] = diff_or_none(in_specs, upstream_specs)
```

第 1 阶段的简化如下：

- **成本模型与反馈重切**（上述第 ③ 步）归入 `[阶段2]`。第 1 阶段切分方式全部手工指定——按 head 切、每个 DPU 分配哪些 head 和层，写成固定配置表；只保留第 ① ② 步，即传播器仍用于推导中间张量布局与 `redistribute` 位置，但不计算成本、不自动求最优。
- **"每个分片用什么算子、在哪个设备"的自动推导**归入 `[阶段2]`。第 1 阶段直接使用手工填好的固定 `PIMTensorSpec.shard_map`（人工给定每个分片的 `local_shape` 与 `dpu_id`）；算子的设备归属已由问题 1 打标（`device` / `part_id`）确定，无需再推。

推进建议：优先在 NumpyBackend 上验证传播规则与 `redistribute` 标注的正确性，再接入下游；算子规则表先覆盖 GEMM/GEMV/softmax/逐元素等少数几类，随白名单宽化逐步补充。

#### 五. 最终产物与依赖

本节是图层（编译期）的第二个 pass，紧接问题 1。其交付产物是：**每个张量的完整 `PIMTensorSpec` 标注 + 每条边的 `redistribute` 标注**，是问题 3（生成 DMA 序列）、问题 6（编排器调度）、问题 7（KV 布局）、问题 8（内存规划）共用的同一数据源。产出文件及含义：


| 文件                 | 含义                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| `device_map.py`      | 设备映射主 pass：读初始切分配置 → 钉 KV → 调传播驱动 → 把`PIMTensorSpec` / `redistribute` 写回 `node.meta` |
| `sharding_rules.py`  | 自写算子布局传播规则表（`RULE_TABLE`：输入布局 + 算子 → 输出布局），本节核心自写件                           |
| `pim_tensor_spec.py` | `PIMTensorSpec` / `TensorShardDetail` 数据结构定义                                                            |
| `shard_config.py`    | 第 1 阶段手工固定切分配置（每份权重切法、KV 的 head → dpu_id 映射）                                          |
| `cost_model.py`      | host-star 搬运成本模型 + 反馈重切（`[阶段2]`，第 1 阶段为占位空实现）                                         |

依赖：

- **PyTorch DTensor（必需，仅编译期）**：借用 `Shard/Replicate/Partial` 标注代数与 `ShardingPropagator` 切分传播分析；**不使用**其运行时（集合通信假设统一地址空间，与本架构冲突）。
- **问题 1 的产物（必需）**：标注了 `device` / `part_id` 的完整图，是本节传播的起点。
- **问题 7 的 KV 切分约束（必需）**：KV 按 head 切、pinned、钉死 `dpu_id`，是本节传播的第一约束（先钉 KV 再传播）。

本节依赖问题 1 的标注图，输出供问题 3/6/7/8 共用的标注图，与问题 6 的产出文件在数据上对接——问题 6 编排器运行时读取的调度表、通信表、内存计划，正来自本节写入 `node.meta` 的 `PIMTensorSpec` 与 `redistribute` 标注。

### 问题 3：redistribute 下沉

#### 一. 问题描述

问题 2 在图上标注了每条 `redistribute` 边及其转换类型，但并未生成任何实际的搬运指令。这一步在传统 GPU 上是隐式的，一个 `redistribute` 会被自动翻译为一条 NCCL 集合通信。但 NCCL 不支持存算一体架构，它假设设备间可直接传递，而存算一体的 DPU 之间无直连，任何跨 DPU 的数据交换都必须经主机中转两跳（`DPU → host → DPU`）。因此本节解决的问题是为每种 `redistribute` 类型提供一套等价的、全部经主机的 DMA 实现，并将这些实现打包为本方案的**通信库**。

不同 `redistribute` 类型与 GPU NCCL 指令、存算一体下降形态的对应关系如下：


| redistribute 类型      | GPU 上对应 | 存算一体下降形态（全部经 host）               |
| ---------------------- | ---------- | --------------------------------------------- |
| `Shard → Replicate`   | all-gather | 各 DPU → host 收片 → 拼接 → 广播回各 DPU   |
| `Partial → Replicate` | all-reduce | 各 DPU → host 收部分和 → 累加 → 回写各 DPU |
| `Shard(i) → Shard(j)` | all-to-all | host 收全部 → 按新维度重排 → 重新分发       |
| `Replicate → Shard`   | 本地切分   | host 按分片下发到各 DPU                       |

上表四种类型对通信计划表的描述粒度要求并不相同。`Partial → Replicate` 的每个 DPU 持有形状相同的完整部分和，逐元素相加不依赖顺序，按参与的 DPU 列出地址与字节数即可完整描述这次搬运。`Shard → Replicate` 与 `Shard(i) → Shard(j)` 不具备这个性质：`Shard → Replicate` 的拼接顺序由各分片在全局张量中的位置决定，与 DPU 编号本身无关；`Shard(i) → Shard(j)` 中一个源 DPU 的本地数据可能需要拆分后发往多个目标 DPU。仅记录参与的 DPU 列表、一个笼统的地址与字节数，不足以描述这两种情况下的搬运。通信计划表按数据段（segment）而非按 DPU 描述搬运，具体字段定义与各类型的生成规则见下文二.（4）。

其中 `Partial → Replicate` 是最典型的一种，其下降后为一串带地址的搬运指令：

```
DMA  dpu0.local → host.buf0     ┐
DMA  dpu1.local → host.buf1     │ 收集各 DPU 部分和
DMA  dpu2.local → host.buf2     │
DMA  dpu3.local → host.buf3     ┘
HOST acc = buf0 + buf1 + buf2 + buf3    合并（host 上累加）
DMA  host.acc → dpu0.local      ┐
DMA  host.acc → dpu1.local      │ 回写各 DPU
...                             ┘
```

一个完整的下降与执行示例（MLP 行切后接 LayerNorm）见**附录 B**。

#### 二. 通信库功能与结构

**（1）技术选型：自写通信库，不基于 FlagCX 加后端**

一种可选方案是在 FlagCX 通信库上新增存算一体后端。但 FlagCX 的核心假设与本架构冲突，强行适配的代价高于自写。具体对比如下：


| 判断点   | FlagCX 的假设                                                                    | 存算一体架构                           |
| -------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| 拓扑     | 设备间能直接 send/recv（ring/tree 算法）                                         | DPU 无直连，必须过 host                |
| 设备模型 | 有 stream（异步执行队列）、IPC（Inter-Process Communication）GPU之间共享显存句柄 | ARM 近存核，无这些                     |
| 接口成本 | CCL adaptor 34 函数 + device adaptor 46 函数                                     | 30+ 个只能写成空桩，需要额外产生无用功 |

若强行新增后端，等于把主机星型（host-star）模型硬塞进点对点（P2P）接口，接口上看似实现，内部却全是绕行。自写反而更简单：host-star 的 all-reduce 本质就是主机上一个"收集 → 累加 → 回写"的循环，约 1000~1500 行即可覆盖全部原语。本文自写通信库的 API 命名对齐 FlagCX，内部为纯 host-DMA 实现。将来若接入 FlagOS 生态或出现 DPU 直连，替换成本较低。

**（2）通信库的形态与设计要点**

通信库本质是把上述 DMA 模板打包为一组原语，全部架在厂商 SDK 的 `copy_to` / `copy_from` / `push_xfer` 之上。设计遵循三个要点：

- **只做点对点 + host 归约**，不实现 ring / tree（这些是集合通信的经典算法，依赖设备直连，在无直连的 DPU 上既无法实现也无意义）；
- **尽量使用批量传输**（`push_xfer`），把"对 N 个 DPU 的 N 次 DMA"合并为一次，以缓解主机带宽瓶颈；
- 后期可引入**计算 / 传输重叠**（某 DPU 计算时，主机同时搬运另一 DPU 的数据），归入 `[阶段2]`。

**（3）通信库要实现的函数**

通信库的函数分为两组。第一组是对外的通信原语，一一对应上表的 `redistribute` 类型；第二组是底层 DMA 封装，架在厂商 SDK 之上，是第一组函数的底层实现。

**第一组：通信原语**（相当于 FlagCX 的 CCL 层，仅保留 host-star 下可实现的）


| 函数               | 对应 redistribute 类型             | GPU/NCCL 等价      | 内部实现（全部经 host）                                                          |
| ------------------ | ---------------------------------- | ------------------ | -------------------------------------------------------------------------------- |
| `all_reduce(plan)` | `Partial → Replicate`             | all-reduce         | 各 DPU 部分和`copy_from` 收到 host → host 累加 → `copy_to` 回写各 DPU          |
| `all_gather(plan)` | `Shard → Replicate`               | all-gather         | 各 DPU 分片`copy_from` 收到 host → host `concat` 拼接 → `copy_to` 广播回各 DPU |
| `all_to_all(plan)` | `Shard(i) → Shard(j)`             | all-to-all         | 全部`copy_from` 收到 host → 按新维度重排 → `copy_to` 重新分发                  |
| `scatter(plan)`    | `Replicate → Shard`               | scatter / 本地切分 | host 按切分方案切片 →`copy_to` 逐片下发到各 DPU                                 |
| `broadcast(plan)`  | （`scatter` 退化 / gather 回写段） | broadcast          | host 一份数据 →`copy_to` 复制到列表里每个 DPU                                   |

上表五个函数统一接收本节二.（4）定义的通信计划表条目 `plan`（内含 `segments` 列表），不再单独接收裸的缓冲区字典或分片列表；`plan.segments` 携带每段的地址、字节数、`wait_for`、`dst_ready_after`，函数内部据此展开 DMA 序列，具体实现见下文三.（1）。

**第二组：底层 DMA 封装**

通信库的 DMA 封装是架在厂商 SDK 之上。厂商库预计类似 UPMEM——UPMEM 已定义好 host-star 编程模型，其原语即本方案运行时与通信库的 API 蓝本。下表列齐厂商原语并标出各自归属的层次：通信库只使用其中的 DMA 三件套（`copy_to` / `copy_from` / `push_xfer`）与主机端归约；其余 `alloc` / `load` / `launch` / `sync` 不属于通信，而是编排器（问题 6）的设备管理与调度基础。


| 本方案封装 / 用途 | 签名或作用                                                                       | 厂商 SDK 原语（类 UPMEM）       | 归属层     |
| ----------------- | -------------------------------------------------------------------------------- | ------------------------------- | ---------- |
| `copy_from_dpu`   | `(dpu_id, mram_addr, nbytes)` 从一 DPU MRAM 读回 host（点对点一跳）              | `dpu_copy_from`                 | **通信库** |
| `copy_to_dpu`     | `(dpu_id, mram_addr, host_buf)` 从 host 写到 DPU MRAM（点对点一跳）              | `dpu_copy_to`                   | **通信库** |
| `push_xfer`       | `(dpu_list, mram_addr, host_bufs)` 把"对 N 个 DPU 的 N 次 DMA"合并成一次批量传输 | `dpu_push_xfer`                 | **通信库** |
| host 归约         | `host_zeros_like` / `host_concat` / `+=`（累加、拼接本身）                       | 纯 host 内存操作（numpy 起步）  | **通信库** |
| 设备管理          | 分配 / 枚举 DPU                                                                  | `dpu_alloc` / `dpu_get_nr_dpus` | 编排器     |
| kernel 下发       | 载入 kernel 二进制                                                               | `dpu_load`                      | 编排器     |
| dispatch          | 触发 DPU 执行                                                                    | `dpu_launch`                    | 编排器     |
| barrier           | 等待完成                                                                         | `dpu_sync`                      | 编排器     |

**（4）通信计划表：按数据段描述搬运**

通信计划表是问题 2 的 `redistribute` 标注与本节通信原语之间的中间产物，由图层编译器在编译期写入，由编排器在运行时逐行读取并照表调用。它必须完整描述每一段数据从哪里来、到哪里去，而不能只按参与的 DPU 列出一个笼统的地址与字节数，否则 `all_gather` 与 `all_to_all` 均无法正确实现：`Shard → Replicate` 的拼接顺序由分片在全局张量中的位置决定，与 DPU 编号无关；`Shard(i) → Shard(j)` 中一个源 DPU 的本地数据可能拆分后发往多个目标 DPU，一个源 DPU 对应多段目标是常态，无法用一个源 DPU 对应一个地址的方式描述。

通信计划表以数据段（segment）为行，一条 `redistribute` 边对应一行或多行，字段定义如下：


| 字段               | 含义                                                               | 数据来源                                                      |
| ------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------- |
| `edge_id`          | 所属的`redistribute` 边编号                                        | 问题 2 的`RedistributeEdge`                                   |
| `type`             | 重分布类型：`all_reduce` / `all_gather` / `all_to_all` / `scatter` | 问题 2 的`RedistributeEdge.type`                              |
| `src_dpu`          | 该段数据的来源 DPU；来源为 host 时记为`None`                       | 源张量`PIMTensorSpec.shard_map` 的键                          |
| `src_local_range`  | 该段数据在源 DPU 本地缓冲区中的`[start, end)`                      | 源`shard_map[src_dpu].local_shape` 换算                       |
| `global_range`     | 该段数据在全局逻辑张量中的`[start_idx, end_idx)`                   | 源`shard_map[src_dpu].start_idx` / `end_idx`（问题 2 已算出） |
| `dst_dpu`          | 该段数据的去向 DPU；去向为 host 时记为`None`                       | 目标张量`PIMTensorSpec.shard_map` 的键                        |
| `dst_local_offset` | 写入目标 DPU 本地缓冲区的偏移                                      | `global_range` 相对目标 `shard_map[dst_dpu].start_idx` 的差值 |
| `nbytes`           | 该段字节数                                                         | `(global_range.end - global_range.start) × dtype_size`       |
| `reduce`           | 归约方式，仅`all_reduce` 使用                                      | 问题 2 的`PIMTensorSpec.reduce_type`                          |

以下两个字段不是逐段字段，字段粒度不同，分别说明：


| 字段              | 粒度                          | 含义                                                                                   | 数据来源                                                                               |
| ----------------- | ----------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `wait_for`        | 整条边一份，全体 segment 共用 | 读前等待：`copy_from` 之前必须等待其执行完成的生产者 DPU 列表                          | 该边全部来源节点在图上的生产者                                                         |
| `dst_ready_after` | 逐段各一份                    | 写前等待：`copy_to` 之前必须等待其执行完成的、该段目标地址上尚未消费旧值的读者节点列表 | 该段目标地址在问题 8`pending_readers` 中记录的读者（问题 8 三."原地写回的安全性"一节） |

`wait_for` 挂在整条边上、由该边全部生产者 DPU 共同决定，因为一条 `redistribute` 边的收集阶段（`copy_from`）要等全部源 DPU 都产出完毕才能开始归约或拼接；`dst_ready_after` 挂在每个 segment 上，因为不同 segment 写入不同的目标地址，各自待等待的读者互不相同，必须逐段判定，不能合并成边级别的一份。

`wait_for` 与 `dst_ready_after` 对应两个不同方向的同步：`wait_for` 保证读之前生产者已经写完（read-after-write），`dst_ready_after` 保证写之前目标地址上的旧值已经被读完（write-after-read）。二者不能相互替代——`wait_for` 只检查源地址一侧的生产依赖，完全不涉及目标地址；目标地址是否可以安全覆盖，由问题 8 的 `pending_readers` 给出，通信计划表在此直接引用该结果。这两个字段本身只是编译期的中间数据；问题 6 的 `exec_plan.py`（问题 6 二.（2）、三.（2））在生成 `ExecutionPlan` 时把它们分别转换成对应 `dma_in` / `dma_out` 命令的 `Command.waits`，运行时不再单独解释 `wait_for` 与 `dst_ready_after` 这两个字段名，只解释 `waits`。

各类型按以下规则生成数据段，全部依据问题 2 已经算出的 `shard_map`（`start_idx` / `end_idx` / `local_shape`）做编译期的整数运算，不引入新的分析步骤。**生成规则统一遵循一条原则：段的去向由 `RedistributeEdge.dst_loc` 决定，不由重分布类型单独决定**——`dst_loc` 已在问题 2 由下游节点的真实 `PIMTensorSpec` 推出（问题 2 二.（9）），本节只读取、不重新判断：

- **`all_reduce`（`Partial → Replicate`）**：与 `all_gather` 同样分收集段、回写段两组：
  - **收集段**：各 DPU 持有形状相同的完整部分和，不涉及全局位置或顺序，`global_range` 对所有参与 DPU 相同，每个 DPU 生成一行，`dst_dpu` 记为 `None`（去向为 host 上的归约缓冲区），`reduce` 填规约方式。
  - **回写段**：按 `dst_loc` 分两种情况生成，不再固定为"回写全体 DPU"：`dst_loc = {"device": "host"}`（下游为 host 胶水节点，如第 1 阶段留 host 的 LayerNorm）时不生成任何回写段，归约结果只落在 host 端的归约缓冲区，由下游 host 节点直接读取；`dst_loc = {"device": "dpu", "dpus": [...]}`（下游为要求 `Replicate` 的 DPU 节点）时，仅对 `dst_loc.dpus` 列出的目标 DPU 各生成一行回写段（`src_dpu` 记为 `None`），不隐含"参与本次重分布的源 DPU 集合 = 目标 DPU 集合"，两个集合按各自的 `shard_map` 独立取。
- **`all_gather`（`Shard → Replicate`）**：分两组、方向相反的数据段，二者不可合并——收集段与广播段所搬运的数据范围不同，`all_reduce` 恰好因为写回地址等于收集地址而可以合用一行，`all_gather` 不具备这个巧合。广播段同样按 `dst_loc` 生成：`dst_loc` 为 host 时不生成广播段，结果留在 host 合并缓冲区；`dst_loc` 为 DPU 集合时才对该集合逐一生成：
  - **收集段**：源张量 `shard_map` 中每个 DPU 生成一行，`dst_dpu` 记为 `None`（去向为 host 上的合并缓冲区），`global_range` 直接取该 DPU 的 `start_idx` / `end_idx`，`dst_local_offset` 即该分片在合并缓冲区中的位置。通信库执行拼接时按 `global_range.start` 排序后再拼接，顺序依据全局位置而非 `src_dpu` 的编号顺序，避免分片顺序与 DPU 编号顺序不一致时拼接错位。
  - **广播段**：合并完成后，对参与重分布的每个目标 DPU 各生成一行，`src_dpu` 记为 `None`（来源为 host 上的合并缓冲区），`dst_dpu` 为该目标 DPU，`global_range` 为整个张量的范围，`dst_local_offset=0`，`nbytes` 为整个张量的字节数。
- **`all_to_all`（`Shard(i) → Shard(j)`）**：取源张量（旧切分）与目标张量（新切分）两份 `shard_map` 的全部 `[start_idx, end_idx)` 区间，两两求交集，每个非空交集生成一行；交集范围即为该行的 `global_range`，交集所属的源 DPU、目标 DPU 及其本地相对位置由此反查得出。一个源 DPU 的区间与多个目标 DPU 的区间相交时，自然生成多行，对应该 DPU 的数据被拆分发往多个目标。
- **`scatter`（`Replicate → Shard`）**：来源为 host 上的完整张量，`src_dpu` 记为 `None`；按目标张量 `shard_map` 的 `start_idx` / `end_idx` 对该完整张量切片，每个目标 DPU 生成一行。

#### 三. 实现思路

**（1）核心实现：收集 → host 归约 → 回写的三段循环**

所有通信原语的核心都是一个"收集 → 主机归约 → 回写"的三段循环，`all_reduce` 最为典型。**通信库内部不做任何等待**：`wait_for` 与 `dst_ready_after` 已在编译期由问题 6 的 `exec_plan.py` 展开进对应 `dma_in` / `dma_out` 命令的 `Command.waits`（问题 6 二.（2）、三.（2）），运行时的 `execute_plan` 在调用到通信库这层之前已经对 `cmd.waits` 逐条 `hal.wait`（等待前驱事件）；通信库拿到的每一段都已经满足全部等待条件，只需执行对应的 DMA/归约动作：

```python
def all_reduce(plan):                  # plan: 本节二.（4）定义的通信计划表条目，Partial → Replicate
                                        # 调用时 plan.segments 涉及的全部等待已由 ExecutionPlan 满足，本函数不再等待
    # 1. 收集：每个 DPU 的部分和 DMA 回 host（点对点，N 次一跳）
    collect_segs = [s for s in plan.segments if s.dst_dpu is None]   # 收集段：DPU → host
    acc = host_zeros_like(collect_segs[0])
    for seg in collect_segs:
        acc += copy_from_dpu(seg.src_dpu, seg.src_local_range, seg.nbytes)   # 厂商 DMA
    # 2. 归约：在 host CPU 上累加（已在上面的循环内完成）
    # 3. 回写：仅对 dst_loc 列出的目标 DPU 执行；dst_loc 为 host 时 writeback_segs 为空，
    #    函数到此为止，结果留在 host 的 acc 缓冲区，交由下游 host 节点直接读取，不发生任何 copy_to_dpu
    writeback_segs = [s for s in plan.segments if s.src_dpu is None]         # 回写段：host → DPU
    for seg in writeback_segs:
        copy_to_dpu(seg.dst_dpu, seg.dst_local_offset, acc)                  # 厂商 DMA
    return acc                                                              # dst_loc 为 host 时，调用方读取此返回值
```

`all_gather` 的收集阶段把中间的 `+=` 换成按 `global_range` 排序后 `concat`（见二.（4）的排序说明），得到合并缓冲区；`all_gather` 的收集段（各 DPU 分片汇总到 host）与广播段（合并缓冲区回写到每个目标 DPU）本就是二.（4）中划分好的两组不同 segment，广播段是否生成同样取决于 `dst_loc`，broadcast 阶段要对广播段逐一 `copy_to`：

```python
def all_gather(plan):
    collect_segs = [s for s in plan.segments if s.dst_dpu is None]     # 收集段：DPU → host
    broadcast_segs = [s for s in plan.segments if s.src_dpu is None]   # 广播段：host → DPU
    #   dst_loc 为 host 时，二.（4）的生成规则不产出广播段，broadcast_segs 为空，
    #   下面的循环自然不执行，合并结果留在 merged 中，交由下游 host 节点直接读取
    parts = sorted(collect_segs, key=lambda s: s.global_range.start)   # 按全局位置排序，见二.（4）
    merged = host_concat([copy_from_dpu(s.src_dpu, s.src_local_range, s.nbytes) for s in parts])
    for seg in broadcast_segs:
        copy_to_dpu(seg.dst_dpu, seg.dst_local_offset, merged)
    return merged
```

整个实现没有 ring、没有 tree，仅是主机上的两组 for 循环，这正是"只做点对点 + host 归约"的落地形态。`all_reduce` 与 `all_gather` 均不判断"是否要广播回 DPU"——是否生成广播/回写段这一决策已经在二.（4）由 `dst_loc` 在编译期做出，通信库只读取 `plan.segments` 里实际存在的段并执行，segments 为空的一侧自然是空循环，不需要额外的 `if` 分支去猜测下游是谁；是否需要等待也已经在编译期由 `exec_plan.py` 决定，通信库不重复做这层判断。

**关于 `fence_copy` 的落点**：第 1 阶段 `copy_to_dpu` / `copy_from_dpu` 均为同步阻塞调用（问题 6 三.），DMA 完成即函数返回，因此不需要额外确认；`[阶段2]` 若引入异步 DMA，写后确认落地的调用属于 `hal.submit(cmd)` 处理 `dma_out` 命令时的职责（在触发下一条依赖该地址的命令之前完成），不属于 `all_reduce` / `all_gather` 内部逻辑，理由与 `wait_for` / `dst_ready_after` 归入 `Command.waits` 相同：同步是 `ExecutionPlan` 与其解释器的职责，通信库只提供无状态的 DMA 原语。

**（2）调用关系：图层决定，编排器执行，通信库不做决策**

通信库自身不做任何决策。是否需要通信、属于哪种类型、涉及哪些 DPU、地址与字节数为何，全部在编译期的图中确定；编排器在运行时只读取通信计划表的对应行、调用相应原语。四层的职责链如下：

```
图层（编译期，问题 2/3）
  切分传播发现某条边 placement 不一致（如 Partial ≠ 要求的 Replicate）
  → 打 redistribute 标签，类型 = Partial → Replicate
  → 按二.（4）的规则，读取边两端的 shard_map 展开为按数据段描述的「通信计划表」
        │
        ▼
图层（编译期，问题 6 exec_plan.py）
  汇总通信计划表 + 标注静态图 + 内存布局表
  → 生成 ExecutionPlan：dma_in / host_reduce / dma_out 等 Command，等待关系落进 Command.waits
        │
        ▼
编排器（运行时，问题 6 execute_plan）  按 ExecutionPlan 的拓扑序发送命令；每条命令先按 waits 精确等待，再 dispatch
        │
        ▼
通信库（本节）           收到已满足等待条件的段，内部展开为 copy_from_dpu / host 累加 / copy_to_dpu 的 DMA 序列，自身不再等待
        │
        ▼
厂商 SDK                dpu_copy_from / dpu_copy_to / push_xfer
```

上述职责链的一个完整实例——以 MLP 行切 `Shard(0)` 后接 LayerNorm 为例，展示从编译期写通信计划表、到运行时编排器照表调用、再到通信库内部 DMA 序列的完整过程——见**附录 B**。

#### 四. 工作量与推进建议

通信库是本方案三个核心自研点之一，代码量约 1000~1500 行 Python，构成如下：

- **第一组通信原语**：几十行的循环模板，`all_reduce` / `all_gather` 优先，`all_to_all` / `scatter` 后补；
- **第二组 DMA 封装**：主要工作量所在，对厂商 SDK DMA 三件套的封装与 `push_xfer` 批量优化。

推进建议：

- 起步阶段优先实现 `all_reduce` 与 `all_gather` 两个原语即可覆盖张量并行中最常见的边；
- 先在 NumpyBackend 上验证每种 `redistribute` 的 DMA 序列数值正确（对应落地路线第 2 步，零硬件），再接入厂商 SDK；
- 计算 / 传输重叠归入 `[阶段2]`，第 1 阶段不做。

#### 五. 最终产物与依赖

本节的产物即通信库，是一组把 `redistribute` 下降为主机中转 DMA 序列的原语，落地为编排器产物中的 `comm_lib.py`（见问题 6 产物表）。

依赖：

- **厂商 SDK 的 DMA 原语（必需）**：`copy_to` / `copy_from` / `push_xfer`，经 HAL 隔离；SDK 到位前先在 NumpyBackend 上以 numpy 内存操作模拟。
- **问题 2 的 redistribute 边标注（必需）**：通信计划表按数据段描述所需的 `global_range`、`src_local_range`、`dst_local_offset` 均由问题 2 的 `PIMTensorSpec.shard_map`（`start_idx` / `end_idx` / `local_shape`）换算得出，类型与参与 DPU 集合来自 `RedistributeEdge`；地址（`mram_offset`）来自问题 8 的内存规划，不由问题 2 给出。三者共同构成本节生成 DMA 序列的输入依据。
- **问题 8 的 `pending_readers`（必需）**：通信计划表的 `dst_ready_after` 字段取自问题 8 对目标地址算出的复用前必须等待的读者，本节不重复做这项分析；等待动作本身由问题 6 的 `exec_plan.py` 转换进 `Command.waits`，本节的通信原语不直接消费 `dst_ready_after`。
- **不依赖**：FlagCX、NCCL，其P2P/集合通信模型假设设备直连，与主机星型架构冲突，是本节自写通信库的根本原因。

### 问题 4：模拟器接入（GeneSim）

#### 一. 问题描述

本方案先走通模拟器流程，在模拟器上验证整条链路。选定的模拟器是 **GeneSim**，一个面向存算一体（PIM）架构的**性能模拟器**：它做的是调度、roofline 时延估算与 PIM trace 生成。

当前GeneSim已经能走通"大模型 → 模拟器"的推理流程。然而，这条现成流程的算子成本粒度太粗。 `build_from_hf_config()` 只按 `num_hidden_layers / num_attention_heads / head_dim / hidden_size` 和一个固定的 `default_seq_len=2048` 硬编码估算每个算子的 `flops` 与 `data_bytes`，既不区分 dtype、也不反映真实的算子实现与访存行为，它是配置驱动的近似结构，不是从真实编译产物抽取出的成本。这层近似正是 GeneSim 性能评估的误差源。

因此本节要解决的问题是：**把编译器（FlagTree）接入 GeneSim**——让每个算子经 FlagTree 编译成中间表示后，从 IR 中分析出真实的算子级成本信息（FP 运算密度、compute/memory 占比、`flops`、`data_bytes` 等），回填 GeneSim 的相应字段，从而在**不改动 GeneSim 模型图结构、不改动动态 trace 两条原有路线**的前提下，用编译器产物替换掉那套硬编码成本，提升性能评估精度。走通的全链路即：`HuggingFace → PyTorch → FlagGems → FlagTree → GeneSim`。

#### 二. FlagOS 编译器接入 GeneSim 的功能与结构

本节实现的功能是：**将 FlagOS 编译器（FlagGems + FlagTree）接入 GeneSim，走通"大模型 → 编译器 → 模拟器"的流程。** 关键设计是把喂给 GeneSim 的信息拆成**三条独立的路**，其中只有一条改道走编译器，另外两条完全复用 GeneSim 原有链路：


| 信息类别                         | 走哪条路                                                                                                                                                       | 是否改动     | 依据                                     |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------- |
| **动态请求（Trace IR）**         | GeneSim 原路：`trace_generator.py` + tokenizer → 请求级 `input_tokens / output_length / arrival_time_ns / is_prefill / seq_len`                               | **不动**     | 保持 GeneSim 双输入模型的动态侧不变      |
| **模型图级结构（ModelIR 骨架）** | GeneSim 原路：`config.json → build_from_hf_config()` 按模板展开粗粒度算子图（`operators / dependencies / op_type / device_hint`）                             | **不动**     | 图结构与设备归属沿用 GeneSim，作结构底座 |
| **算子级成本信息**               | **★ 接入 FlagTree**：每个算子经 FlagTree 编译出中间表示，从 IR 分析出 `flops / data_bytes / arith_intensity` 等成本量，**回填** GeneSim `Operator` 的对应字段 | **本节新增** | 用真实编译产物替换硬编码成本             |

全链路分段如下：

```
   HuggingFace 模型（config.json / 权重）
        │
        ├──────────────────────────────────────────────┐
        │ 图结构路（GeneSim 原有，不动）                  │ 动态请求路（GeneSim 原有，不动）
        ▼                                                ▼
   config.json                                      dataset / synthetic
        │ build_from_hf_config()                         │ trace_generator.py + tokenizer
        ▼                                                ▼
   ModelIR 骨架                                      Trace IR
   （operators / dependencies                       （input_tokens / output_length /
     / op_type / device_hint）                        arrival_time_ns / is_prefill / seq_len）
        │                                                │
        │  ★ 成本路（本节接入 FlagTree）                  │
        │  每个算子： FlagGems 实现 → FlagTree 编译        │
        │            → TTIR（或 pim mlir）→ IR 分析       │
        │            → flops / data_bytes / arith_intensity
        │  回填 ModelIR 里对应 Operator 的成本字段         │
        ▼                                                ▼
        └───────────────────► GeneSim scheduler / roofline / pim_compiler ◄───────────┘
                                        │
                                        ▼
                        调度结果 + 时延估算 + PIM trace
```

**编译链形状**（本节与问题 5 统一使用同一套术语）：

```
TTIR （目标无关，FlagTree 原有；不带 GPU layout / 地址空间等硬件细节）
  ├── TTGIR             （NVIDIA 线，FlagTree 原有）
  └── pim mlir(参照 upmem) （存算一体线，与 TTGIR 平级、位于 TTIR 下层；即问题 5 的
                            `ttir→upmem` 桥产出的那层，含存算一体的显式访存细节）
```

> 术语约定：编译链中这层 IR 全文统一称 **pim mlir(参照 upmem)**（首次出现用全称，后文简称 **pim mlir**）；它由问题 5 的 `ttir→upmem` 桥产出，实现上复用 Cinnamon 的 `upmem` dialect。它与"厂商 SDK（类 UPMEM）"是两回事——后者是硬件编程模型，不在编译链的 IR 层。

**接口契约（从 IR 抽什么、回填到哪）**：本节只写 GeneSim `Operator` 已有的字段，不新增 schema。可回填的字段与来源如下：


| GeneSim`Operator` 字段                                      | 现状（`build_from_hf_config` 硬编码） | 本节改由编译器 IR 提供                                                                                                                                                                                 |
| ----------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `flops`                                                     | 模板公式按结构超参估算                | 从 IR 统计 FP 运算指令计数（第 2 步需按三.(6) 的规则做局部→全局换算）                                                                                                                                 |
| `data_bytes`                                                | 元素计数 × 2（dtype 未标）           | 沿用 GeneSim 原有语义——**算子对外的净读写字节数**（Σ输入 + Σ输出，按张量的全局 shape），只把 dtype 从"固定×2"换成 IR 里的真实 dtype；**不**统计 tile 内部的 MRAM↔WRAM 重复搬运量（理由见三.(3)） |
| `arith_intensity`                                           | `flops / data_bytes` 自动计算         | 随上两项一并精化                                                                                                                                                                                       |
| `op_type` / `device_hint` / `input_shapes` / `dependencies` | 模板给定                              | **不改**，沿用 GeneSim 图骨架                                                                                                                                                                          |

`layer_idx / head_idx / source_name / kernel_ids` 等 GeneSim `Operator` schema 里没有的字段，放入 versioned sidecar（`flagos_genesim_extensions.json`），以 `op_id` 关联，不传给 `ModelIR.load()`，从而基础 `.ir` 仍可被当前 GeneSim loader 直接读取（对齐参考文档的分离策略）。第 2 步起，sidecar 额外新增两个字段：`mram_traffic_bytes`（tile 级 MRAM↔WRAM 真实搬运字节，仅供后续精化访存时延模型参考，不进 `data_bytes`）与 `shard_participants`（该算子在问题 2 中被分摊到的 `dpu_id` 列表及每份的 `placement`，供三.(6) 的局部→全局换算使用）。

**算子边界假设：GeneSim `op_type` 与 FlagGems/FlagTree kernel 的对应关系（白名单，第 1 阶段显式声明）。** `export_costs_to_genesim` 对每个白名单算子调 `flaggems_lookup(op)` 取实现、再调 `flagtree_compile()` 编译，这里隐含一个前提：**GeneSim 模板展开的算子边界，与 FlagGems 的实现边界、FlagTree 编出的 kernel 边界三者对齐**——即不存在"一个 GeneSim Operator 对应的计算，被 FlagTree 融合进了别的算子的 kernel 里"的情况。第 1 阶段该前提对下表逐条成立，超出下表范围的不适用：


| GeneSim`op_type`（白名单 A 类）                                     | FlagGems 实现                                                                   | FlagTree/pim mlir kernel 粒度                                                                                                | 假设成立的依据                                                                |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `GEMM`                                                              | FlagGems 的`mm` / `addmm` 等独立线性层实现                                      | 1 个 Operator ↔ 1 个 FlagTree kernel（若按 WRAM tile 拆分为多个 kernel，允许 1 : N，`analyze_cost` 对多个 kernel 求和即可） | FlagGems 现有 GEMM 实现是独立算子，未与其他算子融合                           |
| `GEMV_SCORE` / `GEMV_CONTEXT`                                       | FlagGems 中 attention score / context 的独立实现（非 flash-attention 融合路径） | 1 : 1                                                                                                                        | 前提是 FlagGems 侧走的是分离实现，不是融合 kernel；若换成融合实现，本假设失效 |
| 逐元素（add/mul/GELU 等）                                           | FlagGems 逐元素算子                                                             | 1 : 1                                                                                                                        | 逐元素算子本身就是独立 kernel，天然满足                                       |
| `SOFTMAX` / LayerNorm / GELU（B/C 类，GQA 模型另含 RoPE / RMSNorm） | 第 1 阶段留 host，不进白名单                                                    | 不适用                                                                                                                       | 不在本节抽成本范围内，沿用 GeneSim 模板成本，与问题 1/5 的窄白名单口径一致    |

这一假设是"先窄后宽"路线（对齐问题 1/5）能够回避的问题，而不是已经解决的问题：窄白名单专挑本来就是分离实现的算子，天然避开融合。若后续宽化白名单、或 FlagGems/FlagTree 引入融合 kernel（如把 score+softmax+context 编进一个 kernel，`[阶段2]`），一对多/多对一关系会真实出现，需要先定义成本分摊规则（例如按理论 flops 占比拆分融合 kernel 的总成本回填到各 Operator）才能继续用本节流程；第 1 阶段不做这件事，只在 sidecar 的 `kernel_ids` 里如实记录每个 Operator 关联的 kernel 数量，供后续审计是否已出现融合。

#### 三. 实现思路

整体思路是动态 trace 与模型图骨架都走 GeneSim 原路，本节只新增"算子成本提取器"这一条改道。具体分四点。

**（1）不动的两条路。** 动态 `Trace IR` 继续由 GeneSim 的 `trace_generator.py` 生成；模型图骨架继续由 `build_from_hf_config()` 从 `config.json` 展开。本节不触碰 GeneSim 的 schema、scheduler、图结构与 trace 生成逻辑——这是"不改模拟器"的落地含义。

**（2）算子成本提取器（本节核心自写件）。** 遍历 GeneSim 图骨架里的每个算子，对白名单内的算子驱动 FlagTree 编译、从 IR 抽成本、回填字段，其余沿用 GeneSim 模板成本。提取器不改变算子的个数、类型、连接与设备归属，只把每个 `Operator` 的成本数值从"模板硬编码"换成"IR 分析所得"，再把 GeneSim 无处安放的元数据落到 sidecar。产物是一份**精化成本后的 `.ir`**（当前 GeneSim 可直接消费）+ 一份 sidecar。主流程如下：

```python
# ir_level ∈ {"ttir", "pim mlir"}；whitelist_A = A 类算子集合（GEMM/GEMV/逐元素）
# shard_map：问题 2 产出的 op_id -> dpu_id 列表，ir_level="ttir" 时不需要（第1步已是全局 shape）
def export_costs_to_genesim(model_ir, whitelist_A, ir_level, shard_map=None):
    sidecar = {}
    for op in model_ir.operators:              # 复用 build_from_hf_config 展开的算子列表，不增删
        if op.op_type not in whitelist_A:      # 非 A 类：沿用 GeneSim 模板成本，跳过
            continue
        impl    = flaggems_lookup(op)          # 取 FlagGems 数学实现
        # 第1步：喂全局 shape；第2步：喂问题5的单 DPU 本地分片 shape（局部值，下面换算回全局）
        kernels = flagtree_compile(impl, level=ir_level)  # 编到 TTIR 或 pim mlir
        cost    = analyze_cost(kernels, ir_level)         # 从 IR 抽局部 flops / data_bytes
        if ir_level == "pim mlir":             # 第2步才需要局部→全局换算，见三.(6)
            cost = scale_local_to_global(cost, op, shard_map)
        op.flops           = cost.flops        # 回填 GeneSim Operator 已有字段（schema 不变）
        op.data_bytes      = cost.data_bytes
        op.arith_intensity = cost.flops / cost.data_bytes if cost.data_bytes else 0
        sidecar[op.op_id]  = {                 # schema 放不下的元数据 → sidecar
            "kernel_ids": cost.kernel_ids, "dtype": cost.dtype,
            "source_name": impl.name, "ir_level": ir_level,
            "mram_traffic_bytes": cost.mram_traffic_bytes,      # 见三.(3)，仅 pim mlir 有值
            "shard_participants": shard_map[op.op_id] if shard_map else None,  # 见三.(6)
        }
    model_ir.save("flagos_genesim_model.ir")             # 当前 GeneSim loader 可直接读
    dump_json(sidecar, "flagos_genesim_extensions.json") # 不传给 ModelIR.load()
```

**（3）IR 成本分析器：`data_bytes` 必须对齐 GeneSim 的消费语义"。** GeneSim 下游对 `data_bytes` 有两处消费（见 `genesim_huggingface_to_model_ir_analysis.md` 第 8 节）——① backend `execute()` 用它算 roofline 的 memory-bound 时间；② scheduler 的 `_execute_sequential/_execute_interleaved` 用它估算跨 VPU 的传输字节数。这两处消费的都是**算子对外的净数据流量**（输入读一次、输出写一次，按全局 shape 计），而不是"设备内部为了完成这次计算实际搬了多少次内存"。

问题 5 的 pim mlir 里，`is_mram_access` 统计的是 **MRAM↔WRAM 的显式搬运指令**——因为 WRAM 容量远小于算子的输入/输出，同一份 MRAM 数据往往要按 tile 分批搬入 WRAM 多次，这个累计搬运量系统性大于"净读写字节数"，且倍数取决于 tile 大小、与算子语义无关。如果直接把它塞进 `data_bytes`，会把"跨 VPU 传输量估算"污染成不可用的虚高值，即便"memory-bound 时间估算"看起来变准了。

因此`data_bytes` 只统计净读写字节数，不统计 tile 级重复搬运；tile 级搬运量单独落到 sidecar 的 `mram_traffic_bytes`，留给后续如果要精化 roofline 的 memory time 模型时再决定怎么用，但不经过 `Operator.data_bytes` 这条通道。两步的差别缩小为：`flops` 在 TTIR 与 pim mlir 上都能准；`data_bytes` 的 dtype 与 shape 来源在两步都从 IR 读（比模板的"固定×2"准），第 2 步能额外拿到 `mram_traffic_bytes` 这个新指标，但不改变 `data_bytes` 本身的计算方式。

```python
def analyze_cost(kernels, ir_level):
    flops, mram_traffic = 0, 0
    for k in kernels:
        module = k.asm[ir_level]               # 第1步 asm["ttir"]（FlagTree 原生）；
                                               # 第2步 asm["pim mlir"]（依赖问题5注册该 stage）
        for mop in walk_mlir_ops(module):
            if is_fp_arith(mop):               # compute 侧：TTIR / pim mlir 都能准
                flops += fp_flops_of(mop)      # 按 tensor shape × 每元素浮点运算数
            if ir_level == "pim mlir" and is_mram_access(mop):
                mram_traffic += transfer_bytes_of(mop)  # tile 级真实搬运，只进 sidecar，不进 data_bytes

    # data_bytes：净读写字节数，两步都走同一套算法——从 IR 声明的输入/输出 shape + 真实 dtype 算，
    # 不管 ir_level 是 ttir 还是 pim mlir，都不统计 tile 内部的重复搬运
    dtype = infer_dtype(kernels)
    data_bytes = bytes_from_io_shapes(kernels, dtype)   # Σ输入 shape + Σ输出 shape，按 dtype 字节数

    return Cost(flops=flops, data_bytes=data_bytes,
                mram_traffic_bytes=mram_traffic if ir_level == "pim mlir" else None,
                kernel_ids=[k.id for k in kernels], dtype=dtype)
```

> `asm["pim mlir"]` 依赖问题 5 把 `pim mlir(参照 upmem)` 注册为 FlagTree 的一个 stage / asm 键；`asm["ttir"]` 是 FlagTree 原生产物，第 1 步无需问题 5。`mram_traffic_bytes` 只在 `ir_level == "pim mlir"` 时有值，第 1 步（TTIR）该字段为 `None`。

**（4）先接入ttir后接入pim mlir：

```python
# 第1步（本期，不依赖问题5）：TTIR 接入，先把 HF→…→GeneSim 全链路走通（访存侧近似）
export_costs_to_genesim(model_ir, whitelist_A, ir_level="ttir")

# 第2步（本期，依赖问题5产出的 pim mlir）：换 pim mlir，补齐访存侧精度
export_costs_to_genesim(model_ir, whitelist_A, ir_level="pim mlir")
```

第 1 步不依赖问题 5、可最先起步、先把链路走通（访存侧近似）；第 2 步随问题 5 的 pim mlir 产出即接入、补齐访存侧精度。这与"先走通流程、再细化优化"的推进原则一致。

**（5）算子覆盖：A 类先行。** 与问题 5 的窄白名单口径一致——第 1 阶段只对 A 类算子（GEMM / GEMV / 逐元素）跑 FlagTree 抽成本；B/C 类（softmax / LayerNorm / GELU 等，第 1 阶段留 host；GQA 模型的 RoPE / RMSNorm 同属此类）暂沿用 GeneSim 模板成本，随问题 5 白名单宽化再补全。这样保证两节覆盖范围口径统一，不会出现"某算子在问题 5 没编 kernel、却在问题 4 要求抽 IR 成本"的矛盾。

**（6）局部 DPU 分片成本 → 全局 Operator 成本的映射（简单规则，第 1 阶段）。** 这里要先分清两步的编译输入不是同一个粒度：

- **第 1 步（TTIR）**：`flagtree_compile()` 编译时喂的是 `op.input_shapes` / `op.output_shapes`——即 GeneSim `build_from_hf_config()` 展开出的**全局模型级 shape**（如 QKV 投影的 `[hidden_size] → [3·hidden_size]`），并不经过问题 2 的切分、也不绑定具体 `dpu_id`。此时编出的 kernel 本身就是"整个算子"的近似，`analyze_cost` 算出的 `flops` / `data_bytes` 天然就是全局尺度，**不存在局部/全局不一致的问题**。
- **第 2 步（pim mlir）**：喂给 `flagtree_compile()` 的是问题 5 实际使用的**单 DPU 本地分片 shape**（如列切后的 `[hidden_size, N/dpu_count]`），这是问题 5 里"算子只认本 DPU 地址空间"的直接后果。此时 `analyze_cost` 算出的 `flops` / `data_bytes` 是**一个 DPU 的局部值**，而 GeneSim `Operator` 语义上表示的是整个模型的算子，两者不在同一尺度上，必须做一次局部→全局的换算，否则会把全局算子成本系统性低估到"1/参与 DPU 数"的量级。

第 1 阶段给出一个**简单映射规则**，只处理"均匀分片、无跨层不对齐"的常规情况（对应问题 2 表格中"手工定、按 head/列/行均匀切"的第 1 阶段配置），不追求处理非均匀切分：

1. **取参与该算子的 DPU 集合**：从问题 2 产出的 `PIMTensorSpec.shard_map` 里，读出该算子输出张量分摊到的 `dpu_id` 列表，记 `N = len(shard_participants)`（即 sidecar 里新增的 `shard_participants` 字段，见二.接口契约）。第 1 阶段切分是手工均匀指定的，`N` 就是配置表里给该算子分片的 DPU 数。
2. **`flops` 按 `N` 线性放大**：GEMM/GEMV/逐元素算子的计算量随分片维度线性缩放，均匀分片下每个 DPU 的局部 `flops` 相等，故 `global_flops = local_flops_per_dpu × N`。
3. **`data_bytes` 按 `placement` 分类处理，不能直接乘 `N`**：同一算子的输入/输出可能是 `Shard` 也可能是 `Replicate`（如 Megatron 列并行里激活是 `Replicate`、权重是 `Shard(1)`），乘 `N` 对 `Replicate` 部分会重复计数：
   - `Shard(dim)` 的张量：局部 shape 已经是全局 shape 的 `1/N`，`N` 份局部字节数相加正好等于全局字节数——即对这部分乘 `N`；
   - `Replicate` 的张量：每个 DPU 持有的就是完整副本，局部字节数已等于全局字节数——**不乘 `N`**，只取一份；
   - `Partial` 的张量（部分和）：shape 与全量一致，同样**不乘 `N`**，只取一份。
4. **跨 DPU 通信不计入本算子的 `data_bytes`**：`redistribute` 边产生的搬运字节属于问题 3 通信库的 DMA 成本，不是该 `Operator` 自身的访存代价，两者边界不能混。

代码骨架（接在三.(2) `export_costs_to_genesim` 的循环里，只在 `ir_level == "pim mlir"` 时触发）：

```python
def scale_local_to_global(cost, op, shard_map):
    """把单 DPU 局部成本换算为 GeneSim Operator 期望的全局成本（简单规则，均匀分片）。
       入:  cost —— analyze_cost() 算出的局部 Cost（单个 DPU 的 kernel）
            op   —— 对应的 GeneSim Operator（读其 input/output 的 placement）
            shard_map —— 问题 2 的 PIMTensorSpec.shard_map，取该算子的 dpu_id 列表
       出:  换算后的全局 Cost；flops 乘 N，data_bytes 按 placement 分类累加。"""
    participants = shard_map[op.op_id]          # 该算子分摊到的 dpu_id 列表（来自问题 2）
    n = len(participants)
    global_flops = cost.flops * n               # 均匀分片：局部 flops 相等，求和 = 乘 N

    global_bytes = 0
    for tensor_spec in op.io_placements():       # 逐个输入/输出张量查 placement（来自问题 2）
        local_bytes = cost.bytes_of(tensor_spec)
        if tensor_spec.placement.is_shard():
            global_bytes += local_bytes * n      # Shard：N 份局部相加 = 全局
        else:                                    # Replicate / Partial：只取一份，不乘 N
            global_bytes += local_bytes

    return Cost(flops=global_flops, data_bytes=global_bytes,
                mram_traffic_bytes=cost.mram_traffic_bytes,   # tile 级搬运量不做局部/全局换算，仍按 sidecar 记录
                kernel_ids=cost.kernel_ids, dtype=cost.dtype)
```

**局限，留给 `[阶段2]`**：该规则假设分片均匀、且同一算子的每份局部 kernel 编译产物完全一致（第 1 阶段"手工均匀切分"下成立）。一旦引入自动切分、非均匀负载均衡，或算子的某个分片因边界效应（如 vocab 切分的最后一份不满）导致局部 shape 不同，就不能再用"局部值 × N"的捷径，需要改为对 `shard_map` 里每个 `dpu_id` 各自编译、各自统计再求和。第 1 阶段不做这件事，只覆盖白名单 A 类算子在均匀切分下的场景。

#### 四. 工作量与推进建议

按组件拆分代码量如下（均为 Python）：


| 组件                      | 内容                                                                                                              | 代码量（估）  |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------- |
| 算子成本提取器骨架        | 遍历`ModelIR.operators`，串起 FlagGems 查实现 → FlagTree 编译 → 回填字段                                        | 约 200~300 行 |
| FlagTree 编译驱动         | 对单算子触发 FlagTree 编译、取回`asm['ttir']`（第 2 步取 pim mlir）                                               | 约 150~250 行 |
| IR 成本分析器             | 从 IR 统计 FP 运算数；按净读写字节数算`data_bytes`；第 2 步另统计 `mram_traffic_bytes`（不进 `data_bytes`）       | 约 250~400 行 |
| 局部→全局成本换算        | `scale_local_to_global`：读问题 2 的 `shard_map` / `placement`，按三.(6) 规则把第 2 步的单 DPU 局部成本换算回全局 | 约 100~150 行 |
| 算子归类表 + sidecar 导出 | aten/FlagGems 算子 ↔ GeneSim`op_type` 映射；`extensions.json` 落盘                                               | 约 150~250 行 |
| 对接与回归脚本            | 精化后`.ir` 能被 GeneSim loader 读入、跑通 scheduler                                                              | 约 100~200 行 |

合计约 **950~1550 行 Python**。

推进建议：

- **先走通、后细化**：第 1 步用 TTIR 把 `HF→PyTorch→FlagGems→FlagTree→GeneSim` 全链路先跑通（访存侧可先近似），验证精化后的 `.ir` 能被当前 GeneSim 直接消费、scheduler 正常出结果；第 2 步在第 1 阶段内、随问题 5 的 pim mlir 产出即接入，补齐访存侧精度。两步都属第 1 阶段，是接入顺序而非阶段划分。
- **A 类先行**：只对 GEMM/GEMV/逐元素抽成本，其余沿用模板，与问题 5 白名单同步宽化。
- 已给出简单规则、第 1 阶段按此执行的三点（详见二.算子边界假设、三.(3)、三.(6)）：① `data_bytes` 统一取净读写字节数，tile 级 MRAM 搬运量另记 `mram_traffic_bytes`，不混入 `data_bytes`；② GeneSim `op_type` 与 FlagGems/FlagTree kernel 的对应关系按白名单表逐条声明为 1:1（或 tile 拆分下的 1:N），融合 kernel 场景推迟到 `[阶段2]`；③ 第 2 步的局部 DPU 分片成本按 `flops × N`、`data_bytes` 按 `placement`（Shard 乘 N、Replicate/Partial 不乘）换算回全局，仅覆盖均匀切分场景。
- 遗留难点（`[阶段2]`）：① 局部→全局换算规则扩展到非均匀切分（每个 `dpu_id` 各自编译统计求和）；② 融合 kernel 出现后的一对多/多对一成本分摊规则；③ 是否需要把 `mram_traffic_bytes` 接入更精细的 roofline memory time 模型。

#### 五. 最终产物与依赖

本节交付的是 FlagOS 编译器与 GeneSim 之间的一层**成本桥接**，产物为一组 Python 脚本加两份数据文件：


| 文件                             | 含义                                                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `flagos_genesim_export.py`       | 成本提取器主入口：遍历算子、驱动 FlagTree 编译、分析 IR、回填字段、落盘                                                               |
| `flagtree_ir_cost.py`            | IR 成本分析器：从 TTIR（第 2 步 pim mlir）统计`flops / data_bytes / 占比`                                                             |
| `op_classify.py`                 | 算子归类表：aten/FlagGems 算子 ↔ GeneSim`op_type` / `device_hint`                                                                    |
| `flagos_genesim_model.ir`        | 精化成本后的`ModelIR`：结构与 GeneSim 原路一致，仅成本字段被编译器产物替换                                                            |
| `flagos_genesim_extensions.json` | versioned sidecar：以`op_id` 关联 `layer_idx / head_idx / source_name / kernel_ids / dtype / mram_traffic_bytes / shard_participants` |

依赖描述：

- **GeneSim（必需）**：提供图骨架生成（`build_from_hf_config`）、动态 trace 生成（`trace_generator`）、以及消费方（scheduler / roofline / `pim_compiler`）。本节**不改其 schema、图结构与 trace 两条原路**，只回填 `Operator` 成本字段——这是"不改模拟器"的边界。
- **FlagTree（必需）**：提供算子的编译前端与 IR 产物。第 1 步用其原生 `TTIR`；第 2 步用问题 5 产出的 `pim mlir(参照 upmem)`。**与问题 5 共用同一 FlagTree 编译前端**，但用途不同：问题 5 把算子编成**可在 DPU 上真执行的 kernel 二进制**（面向执行），本节只从编译产物中**提取成本元数据喂给模拟器**（面向仿真估算）。
- **FlagGems（必需）**：提供算子的数学实现，作为 FlagTree 编译的输入。
- **问题 5 的 pim mlir（第 2 步必需，但同属第 1 阶段）**：本节第 2 步的访存侧成本依赖问题 5 产出 pim mlir；**第 1 步（TTIR）不依赖问题 5，可独立最先起步。** 两步均在第 1 阶段内完成。
- **问题 2 的 `PIMTensorSpec.shard_map` / `placement`（第 2 步必需）**：第 2 步做局部→全局成本换算（三.(6)）时，需要读取该算子分摊到的 `dpu_id` 列表及每个张量的 `placement`，这是本节新增的依赖，第 1 步（TTIR，全局 shape 编译）不需要。
- **明确不依赖 / 不改动**：GeneSim 的 `ModelIR` schema、`build_from_hf_config` 的图展开逻辑、`trace_generator` 的动态请求生成——三者保持原样，本节仅在其产出的 `.ir` 上做成本回填。

### 问题 5：算子实现

#### 一. 问题描述

问题 1/2/8 在编译期已确定每个算子的设备归属、切分方式与内存布局，但这些仅是元数据；真正执行计算的 kernel 仍需在存算一体上实现。本节的职责是将 FlagGems 已有的算子落到存算一体硬件上执行（硬件到位前先在 NumpyBackend 上验证功能正确性）。

FlagGems 的现有算子均面向 GPU，无法直接搬运到存算一体上运行。**但关键问题不在数学逻辑，而在把算子的执行放到 DPU 的地址空间与执行模型上**——这包含三层：数据搬运（MRAM↔WRAM 显式 memcpy）、DPU 内部执行映射与物理布局（tasklet 怎么切、WRAM 里怎么摆）、以及目标设备代码生成（下降到 upmem dialect / LLVM / 厂商 SDK）。一个算子可拆分为两部分：**数学逻辑**与**数据搬运**。数学逻辑与硬件无关，可直接借用；数据搬运在 GPU 上由运行时代为完成。存算一体既无运行时代管数据搬运的机制，架构又与 GPU 不同，图层向下传递给 kernel 的信息也随之不同，增加了本地 shape、MRAM 地址、WRAM 容量约束。因此每个落 DPU 的算子必须**显式补齐三项搬运信息**：


| 要素             | 是什么                                                                                                 | GPU                        | 存算一体                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------ | -------------------------- | --------------------------------------------------------- |
| **① 输入映射**  | 这个算子拿到的不是全局张量，而是**本 DPU 的分片**：本地 shape（如列切后 `[K, N/2]`）+ 在 MRAM 的读地址 | allocator 给指针，形状无关 | 问题 2 的切分定 shape，问题 8 的`DPU_k.plan` 定 MRAM 地址 |
| **② 落哪个 PU** | 这个算子在**哪个 DPU** 上算；DPU 内部再按 tasklet 拆分、按 WRAM 容量定分块                             | 运行时自动分派到各 rank    | 问题 2 的`dpu_id` 定落哪个 DPU；尺寸根据WRAM 容量定       |
| **③ 输出去向**  | 结果先写回**本 DPU 的 MRAM**；之后要不要送到别的 DPU，看图上有没有 redistribute 边                     | NCCL 紧邻 kernel 自动做    | 本地 store 由 kernel 生成；跨 DPU 通信剥离给编排器        |

**算子本身只认"本 DPU 的地址空间"**：上述三要素中的"切多大、落哪个 DPU、是否跨 DPU 传送"全部由图层（问题 1/2/8）在编译期算好算子只需照单执行。

需区分两层粒度：**跨 DPU 层**（这片数据在哪个 DPU、MRAM 哪个地址）由问题 2/8 在编译期算好，问题 5 只读取、不重做；**DPU 内部层**（这片本地 shape 再怎么切给 tasklet、WRAM 里怎么分块摆放、tasklet 间同步点插在哪）问题 2/8 不管，由问题 5 的桥自己设计并生成。后者正是下文"PIM 执行映射与物理布局"的准确指代范围。

#### 二. 算子实现功能与结构

**（1）输出去向的两种形态决定 kernel 是否生成通信指令**

上述第 ③ 项"输出去向"须区分两种"存到哪"，这决定 kernel 内是否生成通信指令：


| 种类            | 是什么                                                                 | 谁生成                                    |
| --------------- | ---------------------------------------------------------------------- | ----------------------------------------- |
| **本地 store**  | 把结果写回**本 DPU 自己的 MRAM 地址**（等价 GPU kernel 的 `tl.store`） | **FlagTree 在 kernel 里生成**             |
| **跨 DPU 通信** | 结果要送到**别的 DPU**（all-reduce/all-gather）                        | **不进 kernel**，编排器在 kernel 跑完后做 |

kernel 的视野中**只有本 DPU 的地址空间**，既看不到其他 DPU，也看不到 host——这正是其零通信的原因。跨 DPU 通信本质是 `DPU → host → DPU` 两跳，属于"host 上的循环"而非"设备上的指令"，即便强行写入 kernel 也无法执行（DPU 无法访问其他 DPU）。

```
FlagTree 编出的 kernel：  读本地 MRAM → 算 → 写回本地 MRAM     ← 到此为止，纯本地
                                              │  (kernel 结束，控制权回 host 编排器)
编排器看图上的 redistribute 边： copy_from_dpu → host 归约 → copy_to_dpu   ← 通信在这里，kernel 看不见
```

**（2）下沉粒度：一个算子一个 kernel，而非子图**

正因算子只认本地视图，向 FlagTree 下沉的仍是一个算子一个 kernel，而非子图。图层只把单个 DPU 的本地视图传下去算子数学类型（matmul/add 等）、输入输出的本地 shape 与 dtype、WRAM tile 约束、MRAM 输入输出基址偏移（或符号名，由内存规划器给出）；**绝不传递**全局切分方式、其他 DPU 的存在、跨 DPU 通信逻辑，kernel 无需知晓自己处理的是分片还是完整副本。如此 FlagTree 无需理解 `part_id` / `dpu_id` / 子图，复杂度不增加。这样做的收益是分层解耦：切分策略调整时，只要本地 shape 不变，Kernel 层无需重新编译。

**编译下沉粒度与装载粒度要分开。** 上面说的"一个算子一个 kernel"是**编译下沉**的粒度，保持算子级以维持解耦。但它不应等同于**装载**的粒度：在类 UPMEM 硬件上，跑一个 kernel 要先 `dpu_load` 把二进制装进 DPU 的 IRAM，再 `dpu_launch`；IRAM 容量很小，若每个算子都是独立二进制、每次都重装 IRAM，则"层数 × 每层算子数 × DPU 数"次装载的开销可能盖过计算本身。GPU 上所有 kernel 一次性驻留、切换几乎零成本，没有这一环节，因此这是近存架构特有的问题，不能照搬 GPU 直觉。缓解办法是把**同一个 DPU 上连续、且中间结果本地驻留的一串算子打包进同一个 DPU 二进制**（一个程序内按算子顺序分派或多 tasklet），`dpu_load` 一次、连续 `dpu_launch` 跑完整串，减少 IRAM 重装。打包只在"同 DPU + 中间结果本地驻留"的连续段内进行；被 redistribute 边隔开（需跨 DPU 通信）的算子分属不同二进制，不能打进同一个。装载成本应纳入第一阶段成本模型，由问题 4 的 GeneSim 量出重装 IRAM 的实际占比，据此决定每个二进制打包多少算子——第一阶段可先按算子级二进制跑通，再按度量结果收紧打包边界。

#### 三. 实现思路

**（1）哪些变、哪些不变**

数学逻辑对 A 类（GEMM/逐元素等主力算子）**不变**，改变的是三件事：（a）**DPU 内部执行映射与 WRAM 物理布局**（tasklet 怎么切、tile 怎么摆）；（b）**访存与同步下降**（MRAM↔WRAM memcpy、tasklet barrier 等 DPU 内同步指令）；（c）**目标设备代码生成**（下降到 upmem dialect → LLVM IR → 厂商 SDK）。这三者均由 `ttir→upmem` 桥统一**自动生成**，不逐算子手写。只有 B/C 类（依赖 GPU warp/atomic 原语的规约类算子，如 softmax、LayerNorm，以及原子操作类算子）的**数值实现本身**需要手工重构，因为近存无对应原语。这些改动按落点区分后，主要集中在两处——**参数**（shape / 地址，替换实参即可）与**桥**（执行映射 + 物理布局 + 访存 + 同步 + 设备代码生成，一次性统一下降），二者均不触碰数学 IR；真正需手改 kernel 主体的仅剩少数 GPU 专属原语算子。

复用的仅限 TTIR 中**硬件无关**的部分：数学 IR 本身、shape/dtype 信息、算子操作序列结构。TTIR 中与硬件强相关的决策——**layout、tile/BLOCK 划分**——不复用：GPU 那套 layout（基于 shared memory/warp）对 PIM 不适用，需按 WRAM 容量重新设计；即便复用 Triton autotune 的**搜索机制**，其搜出的**具体 layout 结果**仍是全新的，而非从 GPU 版本移植。下表逐维度标出是否需改、改什么：


| 维度                  | GPU                                             | 存算一体                                | 要不要改算子                | 改的具体内容                                                                                             |
| --------------------- | ----------------------------------------------- | --------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------- |
| 数学逻辑（乘加/归一） | Triton kernel 里的 matmul/softmax 计算          | 完全相同                                | **不改**                    | 数学 IR 原样保留，这是 FlagGems 的价值所在                                                               |
| 访存方式              | `tl.load/store` 全局显存，SRAM 缓存**硬件隐式** | 必须**MRAM↔WRAM 显式 memcpy** 搬进搬出 | **改，但不手改**            | 桥`ttir→upmem` 把 `tl.load`→`upmem.memcpy`，一次性下降、所有算子共用                                   |
| 内存地址              | kernel 传指针实参（`a_ptr`）                    | 传 MRAM offset，**等价 GPU 指针**       | **不改**                    | 当 launch 参数传（视 SDK；若符号链接期烧死则进二进制，见问题 3）                                         |
| 本地 shape            | 传 M/N/K                                        | 传本地分片 shape（如`[K, N/2]`）        | **不改**                    | 换 shape 实参、按需重编，kernel 本就 shape-generic                                                       |
| tile / BLOCK          | BLOCK 受 shared memory 上限约束                 | BLOCK 受更小的 WRAM 约束                | **不改（调参）**            | 桥按 WRAM 容量重选 BLOCK，复用 Triton autotune 的搜索机制（结果按 WRAM 容量重新搜出，非移植 GPU layout） |
| DPU 内同步            | warp 天然同步 /`__syncthreads`                  | 多 tasklet 要**barrier / mutex**        | **改，多数自动**            | 转 pim mlir 时**新增同步原语**（UPMEM barrier），主力算子由桥/FlagTree 自动生成                          |
| 跨线程规约            | warp shuffle / cross-warp reduce                | 近存**无此原语**                        | **要重构（少数）**          | 手写 PIM 版：warp 规约 →**tasklet barrier + WRAM 手写规约**，典型如 softmax / LayerNorm                 |
| 原子操作              | atomic add 等                                   | 近存**无 atomic**                       | **要重构或退 host（少数）** | 单独写 PIM 版，或退回 host 当胶水                                                                        |
| 跨 DPU 通信           | NCCL 紧邻 kernel                                | kernel 看不到别的 DPU                   | **不改（反而更省）**        | 通信剥离给编排器，kernel 零跨 DPU 逻辑                                                                   |

此处需澄清一个易混点：**访存下降的抽象本身是硬件无关的。** `tl.load` / `tl.store` 并不绑定特定硬件。GPU 后端将其下降为"全局 load + 隐式 SRAM"，`ttir→upmem` 桥则将同一个 `tl.load` 下降为 `upmem.memcpy`，先把 MRAM 搬进 WRAM 再读。数学部分的 IR 原样保留，仅访存下降的目标发生变化。

```
GPU 上的 Triton：   tl.load(全局显存指针) → 算 → tl.store(全局显存指针)   （SRAM 缓存硬件隐式）
存算一体上：        MRAM→WRAM 显式搬 → 算 → WRAM→MRAM 搬回              （两级内存，桥补出这段）
```

**（2）研发路线：复用 Cinnamon 的 `upmem` dialect**

本节自研的 `ttir→upmem` 桥，其下降的目标层即编译链中与 TTGIR 平级、位于 TTIR 下层的 **pim mlir(参照 upmem)**；该层在实现上直接复用 Cinnamon 的 `upmem` dialect（原语如 `upmem.memcpy` / `upmem.pwram_alloc` 等）。因此下文"桥名 `ttir→upmem`"与"目标层 pim mlir(参照 upmem)"指的是同一条下降路径的两端——桥是转换 pass，pim mlir 是其产物层。这层产物也正是问题 4 第 2 步提取算子访存成本的来源。

当前 FlagTree 实现版本无 PIM 支持，分布式原语较少，仅 TLE dialect 包含部分同步 barrier 信息。分层参照如下：


| 层次                                           | 参照什么                                | 结论                   |
| ---------------------------------------------- | --------------------------------------- | ---------------------- |
| 图层分布式（哪片数据在哪个 DPU）               | DTensor 的 Shard/Replicate/Partial 代数 | 借标注语言（问题 2/3） |
| Kernel 层分布式（数据进出 DPU bank、存内规约） | FlagTree 的 TLE dialect 工程结构        | 照结构自写             |

Cinnamon 是学术界已完成的 UPMEM MLIR 编译器，覆盖 kernel 层。本方案直接复用其 `upmem` dialect（在 UPMEM 原语基础上进一步扩展）。二者的对应与复用程度如下：


| 本方案需要的         | Cinnamon`upmem` 已有                                | 复用度                |
| -------------------- | --------------------------------------------------- | --------------------- |
| 近存 scratchpad 分配 | `upmem.pwram_alloc`（WRAM 分配）                    | 直接用                |
| 数据进出 memory bank | `upmem.memcpy`（`mram_to_wram` / `wram_to_mram`）   | 直接用                |
| host↔DPU 传输       | `upmem.scatter` / `upmem.gather`                    | 改 affine map         |
| kernel 启动          | `upmem.launch`                                      | 塌缩 rank 层          |
| 下降到 SDK           | `UPMEMToLLVM.cpp`（611 行，已 emit UPMEM SDK 调用） | ~95% 可用             |
| 运行时               | `upmem_rt.c`（调真实 UPMEM SDK）                    | ~80%，换厂商 SDK 符号 |

产出一个算子并非逐个手写：数学部分的 IR 原样保留，通过桥一次性下降到近存、所有算子共用。三档归纳如下：

- **复用**：数学 IR（A/B/C 类共享）；TTIR 硬件无关结构信息（shape/dtype/算子序列）；Cinnamon `upmem` dialect 与 `UPMEMToLLVM`（~95%）；UPMEM 运行时（~80%，换厂商 SDK 符号）。
- **重新设计（桥自动生成，不逐算子手写）**：DPU 内 tasklet 执行映射、WRAM 物理布局/tiling、访存与同步下降、目标设备代码生成——这四项是 `ttir→upmem` 桥的核心产出，对所有落 DPU 的算子统一生效。
- **手写（仅 B/C 类少数算子）**：规约类（softmax/LayerNorm）与原子操作类的数值实现本身，因近存无 warp shuffle / atomic 原语可直接映射。

需要改动的是两处，外加搭建一座桥：

```
   FlagGems 算子（TTIR，数学逻辑）  ← 借用，不动
          │  TTIR
          ▼
   ┌─────────────────────────────────────────────┐
   │ ttir → upmem 桥  （要搭，约 500-1000 行）      │  两套 MLIR 栈无现成交集
   │  · tl.load/tl.store → upmem.memcpy            │  按①本地 shape 和 MRAM 布局搬
   │  · tile 尺寸按②的 WRAM 容量重调                │  换参数，不是重写数学
   └─────────────────────────────────────────────┘
          │
          ▼
   Cinnamon upmem dialect  ← 复用，改两处：
   ① 塌缩 rank 层（架构与 UPMEM 不同处按需改）
   ② 运行时换成模拟映射或者厂商 SDK 符号
          │  UPMEMToLLVM（复用）
          ▼
   LLVM IR + 厂商 SDK 调用
          +
   回传描述符 {DPU 二进制, tasklet 数, WRAM 占用, 期望 MRAM 输入/输出布局}
```

回传的"期望 MRAM 布局"是第 ③ 项输出去向的落地凭据——编排器据此执行 `copy_to_dpu` / `copy_from_dpu`，搬错地址即导致结果错误。

**（3）三个决策要点**

- kernel 由 FlagGems / FlagTree 生成，增加 `ttir→pim mlir` 转换 pass，将其下降到 **pim mlir(参照 upmem)**（实现上复用 Cinnamon 的 `upmem` dialect 及其 lowering 逻辑）。
- **不在 TTGIR 中加入 PIM 原语**：TTGIR 是 GPU 特化的（warp / layout），PIM 原语应基于 pim mlir(参照 upmem)、与 TTGIR 并列。

**（4）衔接编排器的接口约定**

要使"连续算子留在同一 DPU、中间结果不返回 host"这种融合得以运行，kernel 侧只需一个轻量约定——**从指定的 MRAM 地址读取输入、向指定的 MRAM 地址写出输出**。地址由编排器按内存计划在 launch 时传入，如此连续 kernel 依靠"上一个输出地址 = 下一个输入地址"将中间结果留在本 DPU 的 MRAM（本地往返成本低，规避昂贵的 host 两跳）。

（5）算子编译器Pass编写

除了算子修改外，还需要考虑算子优化Pass编写，例如：循环分块、dpu通信原语生成、软流水等等。

#### 四. 工作量与推进建议

按工作量可把算子归为三类（与前述"是否需改"一一对应）：


| 类别           | 例子                                      | 实现方式                                                                                                                    | 工作量                             |
| -------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **A 主力算子** | GEMM / GEMV / 逐元素                      | 数学逻辑直接借用，只经桥做访存下降 + tile 重调                                                                              | 每个几乎零成本，占绝大多数         |
| **B 规约类**   | softmax / LayerNorm / reduce              | 若规约维落在同一 DPU 内（如按 head 切后 head 内规约）即为本地算子；若规约跨分片，则被图层标为 redistribute 边、交编排器两跳 | 中等，取决于规约维与切分维是否对齐 |
| **C GPU 专属** | warp shuffle / cross-warp reduce / atomic | 近存无对应原语，无法映射：单独编写 PIM 版，或退回 host 充当胶水                                                             | 少数，逐个处理                     |

搭桥本身约 500~1000 行（两套 MLIR 栈之间无现成交集），是本节的主要工作量；Cinnamon 的 `UPMEMToLLVM` 约 95% 可用、运行时约 80% 可用（替换厂商 SDK 符号）。

**第 1 阶段简化**：窄白名单只放入 A 类，B、C 类算子（softmax / LayerNorm / GELU 等，GQA 模型另含 RoPE / RMSNorm）全部留在 host 充当胶水，先跑通全链路。将 B、C 挪进 DPU 归入 `[阶段2]`（对应问题 1 的"先窄后宽"），因此第 1 阶段几乎无需手改 kernel。

#### 五. 最终产物与依赖

本节的产物是 `ttir→upmem` 桥（本方案三个核心自研点之一）与由其产出的单 DPU kernel 二进制。每个 kernel 只认本 DPU 的 MRAM 地址、零通信，并回传描述符 `{DPU 二进制, tasklet 数, WRAM 占用, 期望 MRAM 输入/输出布局}` 供编排器使用。此外，本节产出的 **pim mlir(参照 upmem)** 中间层也是问题 4（GeneSim 接入）第 2 步抽取算子访存成本的来源——问题 4 从这层 IR 统计真实读写字节，回填 GeneSim 的成本字段（面向仿真，不影响本节面向执行的 kernel 产物）。

依赖：

- **FlagGems（借数学逻辑）**：现有算子的数学 IR 原样借用，是本节"不重写数学"的基础。
- **FlagTree（借 kernel 生成）**：TTIR 的产出与 TLE dialect 的工程结构。
- **Cinnamon `upmem` dialect（复用，MIT 许可）**：`pwram_alloc` / `memcpy` / `scatter` / `gather` / `launch` 等原语、`UPMEMToLLVM` 下降与运行时，改动限于塌缩 rank 层与替换厂商 SDK 符号。
- **问题 2 的切分（必需）**：确定每个算子的本地分片 shape。
- **问题 8 的内存计划（必需）**：确定 MRAM 读写地址。
- **明确不依赖**：GPU 特化的 TTGIR（warp / layout 抽象与近存不匹配，这也是坚持从硬件无关的 TTIR 出发的原因）。

### 问题 6：主机编排

#### 一. 问题描述

传统主机编排器仅考虑了GPU架构特性，未考虑到近存架构独立访问空间的特点。为了实现面向近存架构的大模型推理，需要实现面向存算一体架构的主机编排器。

#### 二. 编排器功能和结构

主机编排器是自回归 decode 的实际执行体，是**执行者不是决策者**——"算子落哪个 DPU、哪些边跨 DPU 通信、内存怎么排"这些决策在编译期决策。编排器作为纯执行者，不需要知道切分推导过程，只拿到「执行说明书」：
算子调度表：拓扑序的算子列表，每个算子标注 dpu_id、对应 kernel 二进制索引、输入输出的 MRAM 地址，这是由图编译器给出；
通信计划表：每条 redistribute 边对应的类型（如 Partial→Replicate）、按数据段（segment）描述的源/目的地址与字节数、host 端归约方式、读前等待 `wait_for` 与逐段写前等待 `dst_ready_after`（字段定义与生成规则见问题 3 二.（4）），这是由图编译器给出；
内存布局表：每个 DPU 的权重区、KV 区、激活区的基地址与偏移，对应问题 8 的 DPU_k.plan，这是由图编译器给出。
编排器完全照表执行：同 DPU 的连续算子直接 launch（中间结果留本地 MRAM，零 host 交互），遇到 redistribute 边就执行 host 中转的 DMA 序列，不做任何切分决策。

这三份表在编排器启动时被一次性编译为一份显式的**执行命令 DAG**——`ExecutionPlan`（结构定义与生成方式见下文三.（2）），DAG 中的每条命令携带精确到地址级的等待列表（`waits`）。编排器运行时只解释这份 DAG、按命令顺序发起 launch 与 DMA，不再需要在标注图、通信计划表、内存布局表三者之间做任何隐式的依赖推断；这一编译动作只是把已有决策展开为可直接执行的形式，不引入新的切分或映射决策，仍然符合"编排器不做决策"这一定位。

编排器负责下面事务：

- **多 DPU 并行调度**：`ExecutionPlan` 中彼此没有 `waits` 依赖的命令要**同时** launch（异步）——ExecuTorch 给不了，是自写核心；
- **精确等待**：通信/取结果之前必须确认对应前驱命令已算完——是并行调度的正确性保证，按 `Command.waits` 逐条等待，见下面"运行时解释器"；
- **各 DPU 本地内存生命周期**：独立地址空间没有 CUDA caching allocator，按编译期 `plan` 静态执行（权重区/KV 区/激活区），不做运行时动态分配；
- **胶水补齐**：把 host 节点和 DPU 子图按图串成完整模型；
- **step 间状态推进**：尤其是 KV cache（问题 7）。

编排器分三层


| 层                                                       | 自写还是借                   | 说明                                                                             |
| -------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------- |
| 上层：解码循环 + 采样 + 状态推进                         | **自写，很简单**             | 几十行的 for 循环                                                                |
| 中层：`ExecutionPlan` 生成（编译期最后一步）             | **自写生成器，借编译期规划** | 遍历标注图一次，把依赖关系展开为命令 DAG（详见三.（2）），产物是数据不是执行行为 |
| 下层：`ExecutionPlan` 解释执行（并行调度 + 胶水 + 内存） | **自写执行器**               | 只解释 DAG，按`waits` 精确等待，不再遍历标注图、不再判断边的类型                 |

编排器的输入/输出如下所示：

```
输入：
  ① 标注完备的静态图（问题 1/2/3 产物）——每 node.meta 带 device/dpu_id/part_id/placement/redistribute
  ② 内存计划 DPU_k.plan（问题 8）——权重区/KV 区(持久，两图共用同一 offset)/激活区(两图各一张 offset 表、共享 act_base)的 MRAM 布局；pending_readers 按图各一份(pending_readers_prefill / pending_readers_decode)
  ③ kernel 二进制 + 描述符（FlagTree 产物）——{二进制, tasklet 数, WRAM 占用, 期望 MRAM 布局}
  ④ 通信库 + 厂商 SDK（问题 3）——编排器调用的底层原语
        │
     问题 6 中层：exec_plan.py 生成 ExecutionPlan（编译期，仅一次）
        │
     问题 6 下层：execute_plan 解释执行（运行时，每次 prefill / 每个 decode step）
        │
输出：自回归解码出的 token 序列
```

编排器需要执行DPU 子图下发以及结果拼合的功能：

```
   ┌──    host   ──┐        ┌──    host   ──┐
   └───────┬───────┘        └───────┬───────┘
           │ copy_to_dpu            │ copy_to_dpu
           ▼                        ▼
      [DPU 子图: QKV 投影]     [DPU 子图: MLP]
           │ copy_from_dpu          │ copy_from_dpu
           ▼                        ▼
        (回到 host ，进入下一段)
```

**下表统计的是**：第一阶段采用"按注意力头切分"策略时，解码每一步中、每个 Transformer 层里哪些位置会产生"DPU→主机→DPU"的往返，以及原因。


| 位置                                  | placement 变化              | 是否回主机                                                                           |
| ------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------ |
| QKV 投影（列切）                      | →`Shard`                   | 否，DPU 本地                                                                         |
| attention 主体（QK^T、×V，DPU 本地） | 规约维未切，KV 本地驻留不搬 | 否，DPU 本地                                                                         |
| softmax（第 1 阶段留 host）           | 分数矩阵需回 host 做归一化  | 是，分数矩阵一次 DPU→主机收集 + 一次主机→DPU 回写（非 all-reduce），按同步搬运计入 |
| 输出投影（行切）                      | →`Partial`                 | **是，all-reduce ×1，目标为 host（下游 LayerNorm 留 host）**                        |
| LayerNorm（第 1 阶段留 host）         | 输入需`Replicate@host`      | 是，在 host 上执行，不回写 DPU                                                       |
| MLP up（列切）→ down（行切）         | →`Partial`                 | **是，all-reduce ×1，目标为 host**                                                  |

**第一阶段注意力的落点与 KV 的关系需要讲清，避免与问题 7 的"KV 永不搬"冲突。** 第一阶段的划分是：QKV 投影、QK^T 打分、softmax 之后的 ×V 都在 DPU 上按 head 本地完成，KV cache 全程留在本 DPU 的 MRAM、不参与任何搬运；只有 softmax 归一化这一步留在 host。因此每层注意力有一次"分数矩阵 DPU→主机"的收集与一次"归一化结果主机→DPU"的回写——搬运的是分数矩阵，不是 KV，问题 7 的"KV 永不搬"约束的是 KV 本身，与此并不冲突。这两次搬运不属于 all-reduce，但仍是实打实的跨设备数据往返，按同步搬运计入通信量与主机缓冲区，由下文 `emit_host_op` 生成为显式的 `dma_in` / `dma_out` 命令，而非隐式发生。

每层有两处 all-reduce（注意力输出投影后一次、MLP 下投影后一次），但第 1 阶段这两次 all-reduce 的目标端点均为 host——因为按问题 1/4 的窄白名单，softmax、LayerNorm 第 1 阶段都留 host 执行，二者的输入是各自的下游消费者，对应的 `RedistributeEdge.dst_loc` 为 `{"device": "host"}`，问题 3 二.（4）据此不生成任何回写 DPU 的广播段，归约结果只落在 host、直接供 host 上的 LayerNorm 读取，随后再经 `Replicate → Shard` 的 `scatter` 送回 DPU 供下一层 QKV 投影使用。整步大约是"两倍层数"次 all-reduce 加"两倍层数"次 scatter。根源是行切分产出的是部分和（`Partial`），而 host 上的归一化要求完整值（`Replicate@host`），这条边必须经主机归约（问题 3）。这个次数是第一阶段解码延迟的主要来源，也正是第三阶段做算子融合要削减的目标；若后续宽化白名单把 LayerNorm 移入 DPU，对应边的 `dst_loc` 会随之从 host 变为 DPU 集合，问题 3 的生成规则不需要改动，只是取值不同。

#### 三. 实现思路

为了实现编排器，首先想到的是参照 vLLM / FlagScale 的运行时，然而它们建立在统一地址空间 + NCCL 集合通信上，范式不匹配，改比工作量反而更大。

本文借鉴Pytorch中的ExecuTorch，是官方端侧 / 边缘推理全栈框架，其目标是把大模型部署到手机、嵌入式等资源受限设备本地推理。ExecuTorch 有两部分，能借的是其中的编译期的静态内存规划和分区模式。但不能用其运行时跑多 DPU。多 DPU 并行调度那段必须自写，这也正是编排器的价值所在。


| ExecuTorch 的部分                                                           | 能不能借     | 原因                                                                                                            |
| --------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| **编译期：静态内存规划**（`MemoryPlanningPass` 算 offset、静态 arena 复用） | ✅**借思路** | 纯编译期 pass，"编译期规划、运行时静态执行"正好适配独立地址空间（无 allocator）                                 |
| **编译期：partition / delegate 模式**（子图交后端、其余 host 兜底）         | ✅**借模式** | 正是我们"DPU 子图 + host 胶水"的心智模型                                                                        |
| **运行时：executor 执行引擎**（单流、串行、遇 delegate 阻塞调用）           | ❌**借不了** | 单流串行，**无多设备并行 dispatch、无精确到命令粒度的等待、无跨设备通信原语**——恰恰是我们最需要的三样它都没有 |

**（1）张量并行下的值类型**

编排器遍历时 DPU 节点走"下发+执行+取回"，host 节点直接用 PyTorch 算。这一遍历本身不是编排器在运行时执行的东西，而是编译期生成 `ExecutionPlan` 时走的一遍图；生成规则见下文（2），运行时只解释生成结果。

**先说清 `env` 里存的是什么**：`env` 是一张"张量名 → 值"的表，值有两种形态——要么是**主机上的真张量**，要么是**一个 `DistributedRef`**：一个逻辑张量在张量并行下同时散布在多个 DPU 上，`DistributedRef` 直接复用问题 2 已有的 `PIMTensorSpec.shard_map`（`dict[dpu_id -> TensorShardDetail]`）。一次列切 Linear 的输出，`env` 里对应的一格是"参与该次并行切分的全部 DPU 各自的分片"。这与问题 2"设备映射的单位是 per-tensor 的 `shard_map`。同一个 DPU 上连续的算子之间只传引用、不搬数据，这就是"中间结果留本地、零主机交互"的落地方式；只有当一个 host 节点要读到这个引用时，才真正把数据搬回主机并按 placement 合并。

```python
# env 里每个张量有两种形态：主机真张量，或一个 DistributedRef（携带 shard_map，覆盖张量并行下的多 DPU 分布）
Value          = 主机张量 | DistributedRef
DistributedRef = { placement, shard_map }   # shard_map: dict[dpu_id -> TensorShardDetail]，与问题 2 同一结构

def plan_graph(graph, env_spec):         # 编译期：遍历图，生成命令而非执行命令
    commands = []
    for node in graph.nodes:             # torch.export 给的拓扑序，按算子逐个走
        if node.meta["device"] == "dpu":
            spec = node.meta["spec"]                    # 问题 2 产出的 PIMTensorSpec
            for dpu, detail in spec.shard_map.items():   # 张量并行：为参与的每个 DPU 各生成一条命令
                waits = resolve_waits(node, dpu, env_spec)   # 见下文（2），按访问区间求交得到精确等待列表
                cmd = Command(op="launch", dpu_id=dpu,
                               payload={"kernel": node.kernel, "addr": detail.mram_offset}, waits=waits)
                commands.append(cmd)
                # 把本命令的写区间登记进依赖表，供后续节点查询（完整逻辑见下文（2）的 writers）
            env_spec[node.name] = DistributedRef(spec.placement, spec.shard_map)
        else:
            # host 兜底节点：输入来自 DPU 时插入收集/回写搬运命令（emit_host_op，完整版见下文（2））
            commands += emit_host_op(node, env_spec)
    return commands
```

这里采用**ExecuTorch 的 partitioner + delegate (委派后端)机制**——被 delegate 的子图交给后端（DPU），没被 delegate 的节点由 runtime 用自带算子在 host 上兜底，进而让编排器把 host 节点和 DPU 子图按图串起来。

每个算子在编译下沉时对应一个独立的核函数（问题 5 已定"一个算子一个 kernel"），编译期一个一个地生成命令，和上面伪代码逐算子遍历是一致的，实现最简单。需要说明的是，这里的"一个算子一个 kernel"是编译下沉粒度，不等于装载粒度——同一 DPU 上连续、中间结果本地驻留的一串算子会打包进同一个 DPU 二进制、`dpu_load` 一次跑完整串（问题 5 三.（2）），因此 launch 命令的 `payload["kernel"]` 指向的是"该 DPU 子图打包二进制索引 + 段内算子偏移"，而非独立的单算子二进制。真正需要等的只有存在数据依赖的位置：一个 DPU 连续算好几个算子时，中间结果一直留在它自己的本地内存里，只把一个"地址引用"传给下一个算子；只有当某个算子的输入来自**别的 DPU**，或某次写入复用了一个仍有未执行读者的地址（问题 8 三."原地写回的安全性"一节），才需要一条精确的等待边。

真实图存在分支、汇合、多输入算子、一个生产者对多个消费者、同一 DPU 上的写后读与写后写等情形，按"相邻两条重分布边之间"划分无法精确表达这些依赖，划分出的等待边界也无法保证既不遗漏依赖、也不引入不必要的全局等待。改为在编译期把全部依赖精确展开为一份**执行命令 DAG**——`ExecutionPlan`，其中每条命令携带的等待列表精确到产生依赖的那一条边（或那一个地址），不是"等整组"。定义与生成方式见下文（2）；运行时只需按该 DAG 的拓扑序发出命令、按每条命令的等待列表等待，不需要再"发现"依赖边界。

> 上面的 `plan_graph` 是"遍历一次图、生成一份线性命令列表"的最简形态，突出胶水节点如何插入通信命令；下文（2）在此基础上给出完整的 `ExecutionPlan` 结构与依赖生成算法，（3）给出运行时的解释器，取代原来的解码循环骨架。

**为什么分成两张图**：第一阶段形状固定，而预填充（prefill）一次处理整段提示词的 `max_seq` 个位置、解码（decode）每步只处理一个新 token，两者形状本就不同；一张固定形状的静态图装不下两种形状，只能各导出、各编译一张。等第二阶段支持可变长度后才谈合并。两张图各自在编译期生成一份独立的 `ExecutionPlan`（见下文（2）），运行时按需分别解释执行。

**两张图共用同一份 KV，状态由编排器跨图持有**：KV 区在编译期只规划一次（问题 8），基址全局唯一、不属于任何一张图，预填充图和解码图共用它、不随图切换重新规划。编排器持有一个横跨两张图的状态对象，把有效长度 `valid_len` 从预填充连续交接给解码：

```python
state = DecodeState(valid_len=0, kv基址=plan.kv区基址)  # valid_len 是位置唯一真值源；kv基址编译期定死，两张图共用
execute_plan(prefill_plan, hal, tokens=提示词)  # 提示词作为图输入喂入；写入提示词那段 KV（KV[0..P-1]）
state.valid_len = 提示词长度 P            # ★ 衔接点：有效长度交给解码
logits = hal.result_of_at_position(prefill_plan, pos=P-1)  # ★ 取真实末位 P-1 的 logits，非填充末位
token = sample(logits)                     # 从 prefill logits 采出首个生成 token
for step in 解码步:
    pos = state.valid_len                  # 本步在 KV[pos] 写、读 KV[0..pos]
    execute_plan(decode_plan, hal, pos=pos, token=token)  # ★ 上一步采样的 token 喂回，写入 KV[pos]
    state.commit_one_position()             # valid_len += 1，代替旧的 advance(n)
    logits = hal.result_of(decode_plan.commands[-1].id)   # 解码图只算一个位置，末命令即是它
    token = sample(logits)
    if token == eos: break
```

**（2）执行命令 DAG：`ExecutionPlan` 的结构与生成**

`ExecutionPlan` 是一份线性的命令列表，每条命令携带自己的等待列表，取代"stage + wait_all"这种按粗粒度分组等待的方式：

```python
@dataclass
class Access:
    loc: tuple                    # ("dpu", dpu_id) 或 ("host", None)
    offset: int                    # 该访问在本地缓冲区中的起始字节偏移
    length: int                    # 访问长度（字节），区间为 [offset, offset+length)

@dataclass
class Command:
    id: int
    op: Literal["launch", "dma_in", "dma_out",
                "host_reduce", "host_concat", "host_permute", "host_slice", "host_op"]
    dpu_id: Optional[int]        # host 命令为 None
    payload: dict                 # kernel/地址/字节数等，按 op 类型解释
    reads:  list[Access]           # 本命令读取的全部地址区间
    writes: list[Access]           # 本命令写入的全部地址区间
    waits: list[int]               # 依赖的前驱 Command.id 列表，精确到产生依赖的那一条边

@dataclass
class ExecutionPlan:
    commands: list[Command]       # 已按拓扑序排列
```

每条命令显式声明自己读、写了哪些地址区间。依赖不再按单个起始地址精确匹配，而是按区间求交生成：一条命令的每个读区间，与此前所有写区间求交，凡相交的写命令都进入它的 `waits`（读后写）；写区间与此前的读、写区间求交则给出写后读、写后写依赖。这样地址范围部分重叠、多输入、别名、以及不同大小的张量落在重叠地址上的情形都能被正确覆盖，而不会因为起始地址对不上而漏掉依赖。

命令天然"产出"一个以自身 `id` 标识的事件，其余命令用 `waits` 引用该 `id`；一条命令的 `waits` 只列出它真正依赖的前驱，不引用与它无关的命令。

生成算法在编译期遍历标注图一次，维护两张表。一张是 `writers: dict[loc, list[(Access, Command.id)]]`（`loc` 为 `("dpu", dpu_id)` 或 `("host", None)`），按地址区间记录每个位置的历史写入命令；查依赖时用区间求交（见下方 `overlap`），而非按单个起始地址精确匹配。另一张是 `reader_cmds: dict[(reader_node, dpu), list[Command.id]]`，记录每个读者节点在张量并行下实际展开成了哪几条命令——问题 8 的 `pending_readers` 给出的是读者**节点**，而 `waits` 需要的是**命令编号**，这张表负责把节点翻译成编号（问题 8 只提供节点，不提供命令编号，命令编号是问题 6 生成 `ExecutionPlan` 时才产生的，翻译只能落在这里）。生成时对每个地址查阅问题 8 产出的 `pending_readers`（问题 8 三."原地写回的安全性"一节），取出该地址在被下一次写覆盖前必须等待的读者节点，再经 `reader_cmds` 换成命令编号：

```python
def overlap(a, b):                              # 两个访问区间是否相交
    return a.loc == b.loc and a.offset < b.offset + b.length and b.offset < a.offset + a.length

def deps_of(reads, writers):                    # 读后写：与任一读区间相交的历史写命令
    return [cid for a in reads for (wa, cid) in writers.get(a.loc, []) if overlap(a, wa)]

def build_execution_plan(graph, comm_plan_table, mem_plan):
    plan = ExecutionPlan(commands=[])
    writers = {}                                # loc -> list[(Access, Command.id)]，按区间记录历史写入
    reader_cmds = {}                            # (reader_node, dpu) -> [Command.id]，节点读操作展开出的命令编号
    # pending_readers 按当前构建的图取对应表：prefill 图取 mem_plan.pending_readers_prefill，
    #   decode 图取 mem_plan.pending_readers_decode（问题 8 按图各产出一份，两图激活 offset 表不同）
    pending_readers = mem_plan.pending_readers_for(graph)   # 见下方说明；返回的是读者节点
    for node in graph.nodes:                     # 拓扑序，上游先算
        if node.meta["device"] == "dpu":
            spec = node.meta["spec"]
            for dpu, detail in spec.shard_map.items():        # 张量并行：为参与的每个 DPU 各生成一条命令
                reads  = access_of(node.args, dpu)            # 各输入分片在本地的地址区间
                writes = [Access(("dpu", dpu), detail.mram_offset, detail.nbytes)]
                waits  = deps_of(reads, writers)              # RAW：与读区间相交的历史写
                for rn in pending_readers.get((("dpu", dpu), detail.mram_offset), []):   # WAR：等旧值读者
                    waits += reader_cmds.get((rn, dpu), [])   # 节点 -> 命令编号
                cmd = Command(op="launch", dpu_id=dpu,
                               payload={"kernel": node.kernel, "addr": detail.mram_offset},
                               reads=reads, writes=writes, waits=waits)
                plan.commands.append(cmd)
                writers.setdefault(("dpu", dpu), []).append((writes[0], cmd.id))
                for arg_node in producer_nodes(node.args):    # 登记本命令是这些输入的读者
                    reader_cmds.setdefault((arg_node, dpu), []).append(cmd.id)
        elif node.meta.get("redistribute") is not None:
            emit_redistribute(plan, node.meta["redistribute"], comm_plan_table,
                               pending_readers, writers, reader_cmds)
        else:
            emit_host_op(plan, node, comm_plan_table, pending_readers, writers, reader_cmds)
    return plan
```

其中 `mem_plan.pending_readers_for(graph)` 按当前构建的图返回对应的那份 `pending_readers`——问题 8 对 prefill、decode 两图各产出一份（激活 offset 表两图不同），prefill 图取 `pending_readers_prefill`、decode 图取 `pending_readers_decode`；两图各调一次 `build_execution_plan`，互不串用。

对每条 `redistribute` 边，按 `edge.type` 分派到固定的命令序列模板，不再依据"段的源/目标 DPU 是否为空"临时判断处于哪个阶段——后一种做法会漏掉 `Shard(i) → Shard(j)` 这类源、目标 DPU 均非空的段，也会给本不需要归约的 `scatter` 硬塞一条求和命令。四种类型与其命令序列一一对应：


| 通信类型                       | 生成的命令序列                                                   |
| ------------------------------ | ---------------------------------------------------------------- |
| 部分和到完整值（`all_reduce`） | 多个`dma_in` → 一条 `host_reduce`（求和）→ 零或多个 `dma_out`  |
| 分片到完整值（`all_gather`）   | 多个`dma_in` → 一条 `host_concat`（拼接）→ 零或多个 `dma_out`  |
| 分片维度转换（`all_to_all`）   | 多个`dma_in` → 一条 `host_permute`（重排）→ 多个 `dma_out`     |
| 完整值到分片（`scatter`）      | 一条`host_slice`（切片）→ 多个 `dma_out`（无 `dma_in`、无归约） |

其中 `dma_in` 等待对应源地址此前的写命令（读后写），中间的 host 命令等待全部 `dma_in`，`dma_out` 等待该 host 命令以及目标地址的 `pending_readers`（即问题 3 中原 `dst_ready_after` 字段现在的来源）。是否生成 `dma_out`（即结果是否回写 DPU）仍由 `dst_loc` 决定：`dst_loc` 为 host 时不生成任何 `dma_out`，与问题 3 二.（4）的生成规则完全一致：

```python
HOST_OP = {"all_reduce": "host_reduce", "all_gather": "host_concat", "all_to_all": "host_permute"}

def war_waits(loc, offset, pending_readers, reader_cmds):        # WAR：目标地址旧值读者（节点 -> 命令编号）
    return [cid for rn in pending_readers.get((loc, offset), []) for cid in reader_cmds.get((rn, loc[1]), [])]

def emit_redistribute(plan, edge, comm_plan_table, pending_readers, writers, reader_cmds):
    segs = comm_plan_table[edge.id].segments

    if edge.type == "scatter":                                  # 完整值到分片：host 切片，无收集、无归约
        host_cmd = _append(plan, "host_slice", None, {"edge": edge}, reads=[], writes=[], waits=[])
    else:                                                       # 其余三类：先收集，再按类型做归约/拼接/重排
        collect = []
        for s in [x for x in segs if x.src_dpu is not None]:     # 收集段：读某源 DPU 本地区间
            r = [Access(("dpu", s.src_dpu), s.src_local_range.start, s.nbytes)]
            collect.append(_append(plan, "dma_in", s.src_dpu, s, reads=r, writes=[],
                                   waits=deps_of(r, writers)).id)
        host_cmd = _append(plan, HOST_OP[edge.type], None, {"segments": segs}, reads=[], writes=[], waits=collect)
    writers.setdefault(("host", None), []).append((Access(("host", None), edge.host_buf_id, edge.nbytes), host_cmd.id))

    for s in [x for x in segs if x.dst_dpu is not None]:         # 回写段：dst_loc 为 host 时该列表为空，不生成
        loc, off = ("dpu", s.dst_dpu), s.dst_local_offset
        w = [host_cmd.id] + war_waits(loc, off, pending_readers, reader_cmds)
        cmd = _append(plan, "dma_out", s.dst_dpu, s,
                      reads=[], writes=[Access(loc, off, s.nbytes)], waits=w)
        writers.setdefault(loc, []).append((cmd.writes[0], cmd.id))
```

（`_append` 为构造 `Command`、追加到 `plan.commands` 并返回该命令的小工具，省略其定义。）`all_to_all` 的收集段与回写段都带非空的源、目标 DPU，按上面"源非空即收集、目标非空即回写"处理即可自然覆盖，`scatter` 则只切片加回写、不产生任何归约命令。

`emit_host_op` 处理留在 host 的节点（如第一阶段的 softmax、LayerNorm）。这类节点的跨设备数据搬运不能隐式发生，必须同样展开为显式命令：输入若来自 DPU，先为每个来源分片生成 `dma_in` 收集到 host；再生成一条 `host_op` 做本节点的纯 host 计算（等待全部 `dma_in`）；输出若下游在 DPU，生成 `dma_out` 回写（下游仍在 host 则不回写，结果留在 host 供后续 host 节点读取）。搬运的字节数与 host 缓冲区由此计入计划，与 `emit_redistribute` 用的是同一套 `writers` / `pending_readers` / `reader_cmds` 依赖机制：

```python
def emit_host_op(plan, node, comm_plan_table, pending_readers, writers, reader_cmds):
    collect = []
    for s in dpu_inputs_of(node):                    # 输入来自 DPU：逐分片收集到 host
        r = [Access(("dpu", s.src_dpu), s.src_local_range.start, s.nbytes)]
        collect.append(_append(plan, "dma_in", s.src_dpu, s, reads=r, writes=[],
                               waits=deps_of(r, writers)).id)
    host_cmd = _append(plan, "host_op", None, {"node": node}, reads=[], writes=[], waits=collect)
    for s in dpu_outputs_of(node):                   # 下游在 DPU 才回写；下游在 host 则此列表为空
        loc, off = ("dpu", s.dst_dpu), s.dst_local_offset
        w = [host_cmd.id] + war_waits(loc, off, pending_readers, reader_cmds)
        cmd = _append(plan, "dma_out", s.dst_dpu, s, reads=[], writes=[Access(loc, off, s.nbytes)], waits=w)
        writers.setdefault(loc, []).append((cmd.writes[0], cmd.id))
```

`emit_redistribute` 与本地 `launch` 命令查的是同一张 `writers` 区间表、同一份 `pending_readers`（当前图对应的那份，prefill 图取 `pending_readers_prefill`、decode 图取 `pending_readers_decode`）——跨 DPU 通信触发的地址复用与同一 DPU 上两个本地 kernel 之间的地址复用，在这份依赖表里没有区别，都表示为"某条命令的 `waits` 引用了另一条命令的 `id`"。这正是把问题 8 的 liveness 结果从"只服务 redistribute 目标地址"扩展为"服务全部复用地址"之后的直接效果：`build_execution_plan` 不需要为"这次复用是不是通信引起的"写任何分支判断。

**KV 读写必须作为显式地址区间纳入 `writers` 依赖表。** KV cache 的写（K/V 投影 kernel 把新 K/V 写入 `KV[layer, dpu, pos]`）与读（attention 的 QK^T kernel 读 `KV[layer, dpu, 0:valid_len+1]`）发生在 kernel 内部，不体现为计算图上的张量边——若只按 `node.args` 建依赖，这条"写后才能读"的关系会缺失。同步串行发射时"恰好正确"，但一旦 `[阶段2]` 引入异步 launch 或 DMA 重叠，attention 可能在 KV 写完成前启动，读到旧值或半写入数据。修法是：**把 KV 子块地址区间（`build_kv_layout` 算出的 `(layer, head, k/v) -> MRAM offset` 与长度，`loc = ("dpu", dpu_id)`）连同权重、激活的访问区间一起登记进命令的 `reads` / `writes`**——K/V 投影命令把对应 KV 子块区间放进 `writes`，attention 的 QK^T 命令把这些区间放进 `reads`，依赖照常由 `deps_of` 区间求交得出。这样静态图仍然简单（KV 不必建模成图上的张量），而 `ExecutionPlan` 能正确表达"attention 读必须等本层 KV 写完成"的依赖，异步化下也不会漏。KV 区是 pinned、不参与 `greedy_reuse` 复用，因此不需要 `pending_readers`（无写后读），只需登记这一条读后写依赖。

预填充图与解码图分别调用 `build_execution_plan` 各生成一份 `ExecutionPlan`；KV 区基址在问题 8 编译期只规划一次、两张图共用同一基址；`state.valid_len` 由编排器在两次 `execute_plan` 调用之间维护，不进入 `ExecutionPlan` 本身。

**（3）运行时解释器**

编排器运行时不再遍历标注图、不再判断"这是不是跨 DPU 的边"，只解释 `ExecutionPlan`：

```python
def execute_plan(plan, hal, tokens=None, token=None, pos=None):  # 输入 token 喂 embedding；pos 供 KV 写入/mask
    events = {}                          # Command.id -> Event，前驱命令产出的完成事件
    hal.bind_inputs(plan, tokens=tokens, token=token, pos=pos)   # 把本次输入绑定到图的输入命令
    for cmd in plan.commands:            # 编译期已排好的拓扑序，顺序发送即可
        for w in cmd.waits:
            hal.wait(events[w])           # 精确等待某一条前驱命令的事件，不是等待整组
        events[cmd.id] = hal.submit(cmd)  # 提交命令，返回其完成事件
```

解码循环相应改为：

```python
execute_plan(prefill_plan, hal, tokens=提示词)  # 提示词喂入；初始化各 DPU 本地 KV cache，写 KV[0..P-1]
state.valid_len = 提示词长度 P
logits = hal.result_of_at_position(prefill_plan, pos=P-1)  # ★ 取真实末位 P-1 的 logits，非填充末位
next_token = sample(logits)               # 从 prefill logits 采出首个生成 token
for step in 解码步:
    pos = state.valid_len
    execute_plan(decode_plan, hal, pos=pos, token=next_token)  # ★ 上一步 token 喂回，写入 KV[pos]
    state.commit_one_position()              # valid_len += 1（本地、零跨 DPU）
    logits = hal.result_of(decode_plan.commands[-1].id)   # 解码图只算一个位置，末命令即是它（host 侧）
    next_token = sample(logits)              # host 侧采样
    if next_token == eos: break
```

预填充图形状固定为 `max_seq`、按填充长度执行，其最后一条命令对应的是填充末位而非真实提示词末位，因此必须用 `result_of_at_position(prefill_plan, pos=P-1)` 取真实末位 `P-1` 的 logits；解码图每步只处理一个新位置，末命令即是该位置，直接取 `commands[-1]` 即可。`execute_plan` 通过 `tokens` / `token` 入参把输入 token 喂给图的 embedding，并在解码时写入 `KV[pos]`——采样得到的 token 由此进入下一步，自回归链条不再中断。

`hal.submit(cmd)`、`hal.wait(ev)`、`hal.query(ev)` 是运行时唯一需要对接厂商 SDK 的三个入口。命令有四类完成事件——DPU kernel 完成、DPU 到主机搬运完成、主机归约/拼接/重排/算子完成、主机到 DPU 搬运完成——不能一律用 `dpu_sync` 表达（主机命令的 `dpu_id` 甚至为空）。因此统一定义一个事件对象 `Event`，内部按命令类型区分 DPU、DMA、主机三种，`submit` 按 `cmd.op` 生成对应事件，`wait` / `query` 只认事件、不再直接面对 `dpu_id`：

```python
@dataclass
class Event:
    kind: Literal["dpu", "dma", "host"]   # 由命令类型决定
    handle: object                         # kind 决定其语义：DPU 句柄 / 传输句柄 / 已完成结果

def submit(cmd) -> Event:                  # 按 op 生成对应事件
    if cmd.op == "launch":                 return Event("dpu", launch_async(cmd.dpu_id, cmd.payload))
    if cmd.op in ("dma_in", "dma_out"):    return Event("dma", start_copy(cmd))
    else:                                  return Event("host", run_host_now(cmd))   # 阶段1直接算完

def wait(ev, timeout=T):                   # 按 kind 分派，不再只对 dpu_id 做 dpu_sync
    ok = dpu_sync(ev.handle, timeout) if ev.kind == "dpu" else \
         fence_copy(ev.handle, timeout) if ev.kind == "dma" else True   # host 已完成
    if not ok: raise HALTimeout(describe(ev))   # 超时定位：DPU 报 dpu_id+kernel，DMA 报源/目的
```


| HAL 底层原语（由`submit` / `wait` / `query` 内部调用）                 | 底层 DPU（类 UPMEM）实现                                              |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `launch_async(dpu, kernel)`——非阻塞下发，不等完成                    | `dpu_launch(dpu, DPU_ASYNCHRONOUS)`                                   |
| `launch_sync(dpu, kernel)`——阻塞下发，launch 即等完成（第 1 阶段用） | `dpu_launch(dpu, DPU_SYNCHRONOUS)`（自带 barrier，`wait` 退化 no-op） |
| `dpu_sync(handle)`——等某个 DPU kernel 事件完成                       | `dpu_sync(dpu)`                                                       |
| `fence_copy(handle)`——确保 DMA 传输已落地再继续                      | DMA 完成回调 / 传输队列`dpu_sync`；同步 `dpu_copy_to/from` 则天然阻塞 |
| `dpu_status(handle)`——非阻塞查询是否完成（`query` 用）               | `dpu_status(dpu)` / 轮询 launch 句柄                                  |

> 右列标注"类 UPMEM"，因为厂商 SDK 的确切签名待确认（对齐问题 6 厂商 SDK 待办）；先按 UPMEM 语义写，落地时只换 HAL 实现。第 1 阶段可让全部事件同步完成、`wait` 退化为直接返回，但 `submit` / `wait` / `query` 的接口形状从一开始就统一，异步化时只换 `Event` 内部实现，不动执行器主循环。若厂商 SDK 只提供整设备粒度的等待（无法等待某一条具体命令），`wait` 在 HAL 内部退化为对该命令所在 DPU 的整体 `dpu_sync`，语义仍然正确，只是第 2 阶段的并发精度会受限。

**读前等待与写前等待均已归入 `waits`，不再依赖调用方按顺序编排。** `build_execution_plan` 生成 `dma_in` 命令时查 `writers` 区间表（相当于原 `wait_for`：读之前生产者必须已经写完），生成 `dma_out` 与本地 `launch` 命令时查 `pending_readers`（相当于原 `dst_ready_after`：写之前目标地址的旧值必须已被读完）——二者是同一套 `waits` 机制在不同命令类型上的应用。第 1 阶段串行调度下，`hal.wait` 对已经执行完的前驱事件直接返回，退化为 no-op；`[阶段2]` 引入异步 launch 或计算/传输重叠后，`waits` 里的每一条依赖仍然是显式的。

**第一阶段做最小超时 + 定位（恢复/重试 = `[阶段2]`）**：`wait` 是阻塞等待，若某个 DPU 的 kernel 有 bug 陷入死循环或永不返回，等待会静默卡死、且不知道是哪个 DPU、哪个 kernel。第一阶段不要求做恢复/重试，但要求带超时，超时即由上面 `wait` 中的 `describe(ev)` 抛出携带定位信息的异常（DPU 事件报 `dpu_id` 与 kernel 名，DMA 事件报源/目的地址），避免整个编排器静默 hang、无从定位。

#### 四. 工作量与推进建议

编排器是"编排者"不是"实现者"，重活都委托给通信库、厂商 SDK、PyTorch。按组件拆分代码量如下（均为 Python，不含通信库那约 1000 行）：


| 组件                                                                                            | 代码量（估）                        |
| ----------------------------------------------------------------------------------------------- | ----------------------------------- |
| `ExecutionPlan` 生成器（`build_execution_plan` + `emit_redistribute` + `emit_host_op`，编译期） | 约 250~350 行，本节新增的核心自写件 |
| 运行时解释器（`execute_plan` + 解码循环）                                                       | 约 200 行                           |
| HAL（`submit` / `wait` / `query` + 底层原语，两套实现，含事件对象）                             | 约 300 行                           |
| 假后端 NumpyBackend（模拟多 DPU 独立地址空间与 DMA）                                            | 约 500~800 行，第一阶段最大的一块   |
| 张量抽象（`env` / `DistributedRef`）                                                            | 约 300 行                           |
| 逐节点对拍器                                                                                    | 约 300 行                           |
| 预填充 / 解码 KV 衔接 + 状态对象（`DecodeState`；KV 布局/读写逻辑见问题 7 `kv_cache.py`）       | 约 200 行                           |
| 超时与错误定位                                                                                  | 约 100 行                           |

编排器合计约 **2150~2450 行 Python**；其中假后端 NumpyBackend 仍是第一阶段最主要的一块自写工作量，应作为独立交付物，不是"换一个类"那么轻；`ExecutionPlan` 生成器是本节相对旧版新增的部分，替代了原来分散在解码循环骨架里的 stage 划分逻辑，工作量集中、职责单一。注意：上表"KV 衔接"一行（约 200 行）仅为 `DecodeState.valid_len` 的跨图交接胶水，**不含**问题 7 的 `kv_cache.py`。

难点：① `ExecutionPlan` 生成算法的依赖表正确性（`writers` 区间求交与 `pending_readers` 查表不能有遗漏，否则退化为隐式 bug）；② 静态内存规划；③ 数值对齐调试（数据流一处标错就结果错，靠和单卡 PyTorch 逐元素对拍定位）。第 1 阶段运行时解释器可先按同步 `launch_sync` 语义跑，`hal.wait` 全部退化为 no-op，正确性优先，异步化是后续优化，`ExecutionPlan` 的结构本身不需要为异步化重新设计。

#### 五. 最终产物与依赖

编排器本体就是一组 Python 文件，是"编排者"不是"实现者"：它自己不实现算子数学、不碰硬件细节，只把静态图串起来、按表调度、必要时经主机中转搬数据。产出文件及含义：


| 文件               | 含义                                                                                                                                                                                         |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exec_plan.py`     | 编译期：`build_execution_plan` + `emit_redistribute` + `emit_host_op`，汇总标注图/通信计划表/内存布局表，生成 `ExecutionPlan`（定义与算法见二.（2）、三.（2））                              |
| `orchestrator.py`  | 运行时：解释`ExecutionPlan`（`execute_plan`）+ 解码循环 + 采样 + 状态推进                                                                                                                    |
| `comm_lib.py`      | 通信库：`exec_plan.py` 生成的 `dma_in` / `host_reduce` / `dma_out` 命令的具体执行（问题 3），不再自行判断 `wait_for` / `dst_ready_after`，等待列表已由 `exec_plan.py` 展开进 `Command.waits` |
| `tensor.py`        | 张量抽象：`env` 值类型、`DistributedRef`（携带 `shard_map`）、按 placement 合并                                                                                                              |
| `hal.py`           | HAL：`submit / wait / query` 三个入口 + `launch_async / launch_sync / fence_copy` 等底层原语，两套实现，含事件对象与超时定位                                                                 |
| `hal_numpy.py`     | 假后端：用 N 块独立 numpy buffer 模拟 N 个 DPU，第一阶段验证用                                                                                                                               |
| `hal_vendor.py`    | 真硬件后端：厂商 SDK（类 UPMEM）绑定，SDK 到位后接入                                                                                                                                         |
| `kv_cache.py`      | 静态 KV cache：编译期算 KV 区大小/offset、运行时 append/read/mask/推指针（定义见问题 7）                                                                                                     |
| `debug_compare.py` | 逐节点对拍器：分片按 placement 合并后与单卡 PyTorch 逐元素比                                                                                                                                 |

依赖描述：

- **PyTorch（必需，唯一大依赖）**：三个用途——host 兜底节点直接调 PyTorch 算子计算、遍历用的静态图与拓扑序来自 `torch.export` / FX、采样也用它。只用它的图和算子，**不用**它的分布式运行时。
- **NumPy（必需）**：给假后端 `hal_numpy.py` 用，模拟多 DPU 独立地址空间。
- **厂商 SDK 的 Python 绑定（后期必需）**：类 UPMEM 的设备原语，经 HAL（`hal_vendor.py`）隔开；SDK 到位前用假后端顶替，换硬件只改这一个文件。
- **明确不依赖**：vLLM、FlagScale 的运行时、NCCL——它们建立在统一地址空间 + 集合通信之上，范式不匹配，是自写编排器的原因。

编排器交付的是上面这组 `.py` 文件；运行时需要编译期产出的 `ExecutionPlan`（由 `exec_plan.py` 汇总**标注静态图（问题 1/2/3）、内存蓝图 `DPU_k.plan`（问题 8）**生成）与 **kernel 二进制（FlagTree 产物）**，不再直接消费标注静态图、通信计划表、内存布局表三份原始数据。

推进顺序对齐落地路线，且**先在 NumpyBackend 上跑通再上真硬件**——把厂商 SDK 抽象成一个 HAL 接口，用 N 块独立 numpy buffer 模拟 N 个 DPU 的独立地址空间，整个编排器/通信库/胶水/KV 先在这个假后端上验证数值正确，等 SDK 到位只换 HAL 一个类。这样编排器正确性与厂商 SDK 解耦，能独立验证。

NumpyBackend 是**运行时层的统一验证底座**，覆盖通信库（问题 3）、编排器（问题 6）、KV cache（问题 7）、内存布局的执行与校验（问题 8）；而问题 1/2/4 是纯编译期的图分析与契约，产出静态蓝图，不经过 backend。所以"先在 NumpyBackend 上跑通"指的是把运行时这几块的数值正确性先在假后端验证，与编译期图分析是两回事。

**逐节点对拍器（第一阶段主要调试基础设施）**：数值对齐是三大难点之一，难在中间张量以分片形态散在 N 个 DPU 的独立 buffer 里，形状和单卡的完整张量对不上，不能直接比。做法是用问题 2 的 `placement` 元信息，把分片自动合并回完整张量，再和单卡 PyTorch 逐元素比——第一处不匹配即定位到具体是哪个 node 出错，而不是只看最终 logits 错了却不知错在哪：

```python
def assert_node_matches_ref(node, env, ref_env):
    val  = env[node.name]                                   # DistributedRef，携带 shard_map（问题 6 三.（1））
    full = gather_by_placement(val, node.meta["placement"]) # Shard→拼接 / Partial→求和 / Replicate→取一份
    torch.testing.assert_close(full, ref_env[node.name])    # 与单卡逐元素比
```

它复用 `placement` 自动合并、不用给每个算子手写比对逻辑，必须先于真硬件在 NumpyBackend 上就位。

### 问题 7：KV cache管理

#### 一. 问题描述

原有 KV cache 面向 GPU 设计，隐含三条前提：一是**统一地址空间**，注意力每步读取全部历史 KV，由硬件与运行时自动定位数据所在；二是**显存分配器动态增长**，cache 缓冲区随解码步数变长，由框架托管（如 HuggingFace 的 `past_key_values` 动态拼接、vLLM 的 PagedAttention 分块）；三是**框架自动管理生命周期**。然而，存算一体架构把这三条前提全部打破：每个 DPU 地址空间独立、无内存分配器、DPU 之间无直连。因此 GPU 那套动态托管的 KV cache 无法直接移植，必须自写一套面向存算一体的 KV cache 逻辑。

如何管理KV cache 是值得讨论的问题，因为它是推理过程中**唯一跨解码步存活、只增不减、每步必读**的大块状态，处理不当，整个解码循环就无法运转。面向存算一体，KV cache 需解决四个子问题：


| 子问题          | 是什么                                     | 归属                            |
| --------------- | ------------------------------------------ | ------------------------------- |
| **① 怎么切**   | KV 按什么维度分到各 DPU                    | 本节（同时是问题 2 的第一约束） |
| **② 怎么存**   | 无分配器，空间从何而来、多大、何种布局     | 本节定大小与布局，问题 8 定位置 |
| **③ 怎么用**   | 每步注意力如何读到所需的 KV                | 本节                            |
| **④ 怎么更新** | 每步新 token 的 K/V 如何追加、状态如何推进 | 本节，与问题 6 解码循环呼应     |

#### 二. KV cache 管理器功能与结构

KV cache 管理器与编排器一样是**执行者而非决策者**：切分维度、内存位置由编译期定死，管理器只负责在既定布局上完成追加、读取与状态推进。其功能沿编译期与运行时两层划分——编译期算好 KV 区的大小与布局，运行时按编排器传入的 `valid_len` 逐步追加并读取。两层职责如下：


| 层     | 职责                                                                        | 产物                      | 持有者                          |
| ------ | --------------------------------------------------------------------------- | ------------------------- | ------------------------------- |
| 编译期 | 按`max_seq` 计算 KV 区**大小**，并算出每个 `(layer, head)` 的 **MRAM 偏移** | 供问题 8 的内存规划器使用 | 规划器（planner）               |
| 运行时 | 按编排器传入的**`valid_len`** 追加新 K/V、读取本地历史 KV、生成 mask        | 不新分配内存，只写数据    | 编排器的`DecodeState`（问题 6） |

**在真实硬件上，追加与读取发生在 kernel 内部**，K/V 投影 kernel 把新算出的 K/V 写入本 DPU MRAM 的序列位置，注意力 kernel 从本地 MRAM 读取历史 KV。因此 KV 管理器负责四件事：编译期计算大小与偏移、运行时按传入的 `valid_len` 定位写入位置、把序列位置作为 launch 参数传入 kernel、生成 mask（位置真值源为编排器的 `DecodeState.valid_len`，本类不自持指针）。

管理器的输入与输出如下：

```
输入：
  ① 本 DPU 的 KV 规格 KVRegionSpec（本节定义）——持有哪些层、哪些 KV head、q_heads_by_kv 映射、max_seq、head_dim
  ② KV 区基址 plan.kv_base（问题 8）——由内存规划器分配
  ③ 有效长度 valid_len（编排器的 DecodeState 跨预填充/解码两图持有，唯一真值源）
        │
     问题 7（编译期算布局 + 运行时管理）
        │
输出：
  · 编译期：kv_allocated_bytes（含对齐 padding 的真实占用，喂问题 8）+ 每个 (layer, head) 的 MRAM 偏移表
  · 运行时：在本地 MRAM 上完成的 update / read_tile（按 tile 分块）+ 每步的 causal / 长度 mask
```

#### 三. 实现思路

**① 怎么切——按 (KV) head 切，是硬约束不是优化项**

```
好的设计（KV 跟着 head 切，本地驻留）：
  DPU0 只算 head 0-1 → 只存 head 0-1 的 KV → 每步本地追加、本地读
  → 跨 DPU 零搬运 ✓

坏的设计（KV 按 seq 切，与 head 不对齐）：
  注意力要看全部历史 token → 每步都要跨 DPU 收集 KV
  → 每步 "DPU→host→DPU" 两跳 → 几百步后 host 带宽爆炸 ✗
```

因为无直连，KV 一旦要跨 DPU 搬，每步都是两跳。所以注意力必须按 head 切、KV 本地驻留永不搬。这反过来约束了整个设备映射策略，**先定 KV 局部性，再让其他层的切分去迁就**（这条同时是问题 2 设备映射的第一约束）。

**② 怎么存——编译期按 `max_seq` 静态预留**

DPU 上没有内存分配器，因此把 KV 区当作**编译期定长空间**一次算死，不做运行时动态增长。当本 DPU 持有 `|L_k|` 层、`|H_k|` 个 KV head 时：

```
kv_bytes = 2 × |L_k| × max_seq × |H_k| × head_dim × dtype_bytes
         （2 = K 和 V 两份）
```

这个大小是本节的产物；把它连同权重区、激活区一起排进那块 ≤8GB 的地址空间、算出偏移，是问题 8 通用内存规划器的职责。**GQA 模型（如 Llama 3）按 KV head 数计算**，而非 Q head 数（一个 KV head 对应多个 Q head）；第 1 阶段固定模型 GPT-2是 MHA，本公式对它自然成立，GQA 一般情形的处理能力仍保留、待后续宽化到 GQA 模型时启用。

**③ 怎么用——本地按 tile 分块读取 + mask，KV 不跨 DPU**

每步注意力从**本 DPU 的 MRAM** 读所需 KV（因为①保证了本 DPU 所需 head 的 KV 就在本地）。**读取按 key token 分块（tile）搬进 WRAM，不满块搬入**：WRAM 容量远小于整块 `[max_seq, head_dim]`（如 `max_seq=256/head_dim=64/fp32` 单 head 已 64KB，超单 DPU WRAM），必须逐 tile 读、逐 tile 处理，tile 尺寸受 WRAM 容量约束（问题 5 的桥按容量定）。再按当前有效长度 `valid_len` 生成因果 / 长度 mask，盖掉预留区里尚未填充的部分。注意：softmax 第 1 阶段在 host（见下文注意力使用部分），故 `QK^T` 与 `weights @ V` 两个 GEMV 在 DPU 上各自独立分块，中间隔一次 host softmax，无需 flash 式在线累计。

**④ 怎么更新——运行时按 `valid_len` 追加，位置真值源唯一**

每步把新 token 的 K/V 写到预留区里 `valid_len` 指向的位置，随后 `valid_len += 1`。这不涉及分配（空间在②已定长预留），只是向后写入并推进。**位置的唯一真值源是编排器的 `DecodeState.valid_len`（问题 6）**，`PIMStaticKVCache` 不自持指针、也没有 `advance`——写入位置由编排器把 `valid_len` 作为 `pos` 显式传入 `update()`。**该 `valid_len` 由编排器持有、跨预填充与解码两图连续推进**——预填充图跑完时 `valid_len = 提示词长度`，解码图每步在其后追加（对齐问题 6 的 `DecodeState`）。这样消除了旧设计里 `DecodeState`、`PIMStaticKVCache.seq_len`、`update(pos=)` 三处真值源不同步导致的"mask 按位置 N 生成、KV 却写到位置 N+1"静默错误。

值得注意的是 "全静态"指的是 KV 区的大小、位置、布局在编译期按 `max_seq` 定长算死、全程不变；但每步"写入新 token 的 K/V、`valid_len` 加一、按当前长度生成 mask"这些运行时动作**必须保留**，它们是自回归生成的核心，不能因为追求"静态"就省去。mask 只是数据、不改变张量形状，因此不破坏静态性。

**代码骨架**分编译期布局与运行时管理两块。编译期部分计算 KV 区大小与每个 `(layer, head)` 的 MRAM 偏移：

```python
# ---------- 编译期：算 KV 区大小与每个 (layer,head) 的 MRAM offset ----------
@dataclass
class KVRegionSpec:
    dpu_id: int              # 本规格属于哪个 DPU
    layers: list[int]        # 本 DPU 持有哪些层的 KV
    kv_heads: list[int]      # 本 DPU 持有哪些 KV head（GQA 按 KV head，不是 Q head）
    q_heads_by_kv: dict      # {kv_head: [该 KV head 服务的 q_head 列表]}，GQA 映射
                             #   持有某 KV head 的 DPU 必须同时持有它服务的全部 Q head 分片
                             #   MHA 是特例：q_heads_by_kv[h] == [h]
    max_seq: int             # 编译期定长，KV 区按它预留
    head_dim: int            # 单个 head 的维度
    dtype_bytes: int         # 每个元素字节数
    kv_base: int             # KV 区基址，来自问题 8 的 plan.kv_base
    kv_off: dict = None      # 输出：(layer, head, 'k'/'v') -> MRAM offset
    kv_allocated_bytes: int = 0  # 输出：build_kv_layout 算出的、含对齐 padding 的真实占用（问题 8 用它排 arena，不用 kv_bytes）

def kv_bytes(spec: KVRegionSpec) -> int:
    """KV 区未对齐下界估算，仅供容量粗估/文档说明用，【不是分配依据】。
       真实分配量（含每子块对齐 padding）由 build_kv_layout 的 kv_allocated_bytes 给出，
       问题 8 排 arena 必须用后者，不能用本函数的返回值自行推进 offset。
       入:  spec —— 本 DPU 的 KV 规格
       出:  int —— KV 区未对齐字节数下界。"""
    return (2 * len(spec.layers) * spec.max_seq
              * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes)  # 2 = K 和 V

def build_kv_layout(spec: KVRegionSpec, align: int) -> KVRegionSpec:
    """把 KV 区切成一个个 [max_seq, head_dim] 定长子块，编译期算死每块 offset。
       入:  spec  —— 本 DPU KV 规格（含 kv_base）
            align —— DMA 对齐要求（来自硬件）
       出:  同一个 spec，kv_off 被填好：(layer,head,k|v) -> 绝对 MRAM offset；
            kv_allocated_bytes 填好：含全部对齐 padding 的真实占用（= off - kv_base）。
            问题 8 必须用 kv_allocated_bytes 推进激活区起点，不得用 kv_bytes 公式重算，
            否则对齐 padding 累积会让 KV 实际写入范围侵入激活区。"""
    spec.kv_off = {}
    off = spec.kv_base
    block = spec.max_seq * spec.head_dim * spec.dtype_bytes   # 一个 (layer,head,k|v) 子块
    for L in spec.layers:                 # 稳定顺序 => 结果可复现
        for h in spec.kv_heads:
            for which in ('k', 'v'):
                spec.kv_off[(L, h, which)] = off
                off = off + align_up(block, align)
    spec.kv_allocated_bytes = off - spec.kv_base   # 准确占用，问题 8 用它，不重算公式
    return spec
```

运行时部分按编排器传入的 `pos`（= `valid_len`），逐步在本地 MRAM 上追加与读取 K/V，全程零跨 DPU：

```python
# ---------- 运行时：update + read_tile + mask（本地、零跨 DPU；位置由编排器传入）----------
class PIMStaticKVCache:
    """面向存算一体的静态 KV cache：每个 DPU 只持本地 head 的 KV，定长预留、跨 step
       驻留、本地读写、永不跨 DPU 搬。接口形态借 HF StaticCache。
       【无内部序列指针】：位置的唯一真值源是编排器的 DecodeState.valid_len（问题 6），
       每次读写由调用方显式传 pos 进来，本类不自持指针、不做推进（去掉了旧的 seq_len/advance，
       避免"三处真值源"不同步导致的 mask 位置与写入位置错位）。"""
    def __init__(self, backend, specs: dict):
        # backend: HAL（NumpyBackend 或厂商 SDK）；specs: dpu_id -> KVRegionSpec
        self.backend = backend
        self.specs = specs

    def update(self, layer: int, pos: int, k_by_dpu: dict, v_by_dpu: dict) -> None:
        """把本步各 DPU 本地算出的新 K/V 写到位置 pos 处（真机在 kernel 内做）。
           入:  layer —— 第几层
                pos   —— 写到 KV 区的第几格（= 调用方 DecodeState.valid_len，唯一真值源）
                k_by_dpu / v_by_dpu —— {dpu_id: {head: 张量}}，本步新算的 K/V
           出:  无（原地写入各 DPU 本地 MRAM）。"""
        for dpu_id, spec in self.specs.items():
            assert 0 <= pos < spec.max_seq, f"pos={pos} 越界 [0,{spec.max_seq})"  # 写前校验
            row = spec.head_dim * spec.dtype_bytes
            for h in spec.kv_heads:
                koff = spec.kv_off[(layer, h, 'k')] + pos * row
                voff = spec.kv_off[(layer, h, 'v')] + pos * row
                self.backend.write_local(dpu_id, koff, k_by_dpu[dpu_id][h])  # 本地写
                self.backend.write_local(dpu_id, voff, v_by_dpu[dpu_id][h])

    def read_tile(self, layer: int, dpu_id: int, head: int, tile_start: int, tile_end: int):
        """按 key token 分块读本 DPU 某层某 head 的 K/V tile（不满块搬入 WRAM）。
           WRAM 容量远小于整块 [max_seq, head_dim]（如 max_seq=256/head_dim=64/fp32 单 head 已 64KB），
           不能满读；由 QK^T / weights@V 两个 GEMV kernel 逐 tile 读入处理，tile 尺寸受 WRAM 容量约束
           （问题 5 的 ttir→upmem 桥按 WRAM 容量定）。无效尾部（pos >= valid_len）由 host softmax 前的 mask 盖掉。
           入:  layer/dpu_id/head；[tile_start, tile_end) —— 本次读取的 key token 区间
           出:  (K_tile, V_tile)，各为 [tile_end-tile_start, head_dim]。"""
        spec = self.specs[dpu_id]
        assert (tile_end - tile_start) * spec.head_dim * spec.dtype_bytes <= WRAM_BUDGET  # 单 tile 不超 WRAM
        row = spec.head_dim * spec.dtype_bytes
        n = (tile_end - tile_start) * row
        koff = spec.kv_off[(layer, head, 'k')] + tile_start * row
        voff = spec.kv_off[(layer, head, 'v')] + tile_start * row
        K_tile = self.backend.read_local(dpu_id, koff, n)
        V_tile = self.backend.read_local(dpu_id, voff, n)
        return K_tile, V_tile
```

运行时用到两种 mask，因第 1 阶段采用"满预留 + 满算"，二者形状不同、需分开处理，负责在 softmax 之前（第 1 阶段 softmax 在 host，故 mask 加在 DPU 产出的 scores 上、随 scores 一起送 host）抹掉两类不该参与计算的位置：**① 预留但尚未填充的尾部（`pos >= valid_len`），② 因果上不该看到的未来 token**。

```python
def prefill_mask(prompt_len: int, max_seq: int):
    """prefill 用：一次处理整段 prompt，scores 是 [max_seq, max_seq] 方阵。
       入:  prompt_len —— 真实 prompt 长度；max_seq —— 预留长度
       出:  [max_seq, max_seq]，可见处为 0、其余为 -inf。"""
    i = torch.arange(max_seq)[:, None]                 # query 位置
    j = torch.arange(max_seq)[None, :]                 # key 位置
    visible = (j <= i) & (j < prompt_len)              # 不看未来(因果) 且 是真实 token(非预留)
    return torch.where(visible, 0.0, float('-inf'))

def decode_mask(valid_len: int, max_seq: int):
    """decode 用：只有一个新 token 在 valid_len 处，scores 是 [max_seq] 向量。
       入:  valid_len —— 已填充的历史长度（= DecodeState.valid_len，唯一真值源）；max_seq —— 预留长度
       出:  [max_seq]，[0, valid_len] 为 0、其余预留位为 -inf（因果自动满足）。"""
    j = torch.arange(max_seq)
    return torch.where(j <= valid_len, 0.0, float('-inf'))
```

最后是注意力如何使用上述 KV cache。此处需区分两个层次：

**QK^T 与 `weights @ V` 本地、KV 本地驻留不跨 DPU；但 softmax 第 1 阶段在 host。** 按问题 1/4/5 的窄白名单，softmax 不落 DPU（B 类规约算子），因此 SDPA 核心并非"全程本地"：DPU 先算 `scores = QK^T/√d + mask`（tiled GEMV，按 key token 分块读 K），把完整 `scores` 送回 host 做 softmax，host 再把 `weights` 送回 DPU 算 `weights @ V`（tiled GEMV，按 key token 分块读 V）。这引入**每层每 Q head 一次 `scores → host → weights` 的 host 胶水往返**——区别于输出投影处的跨 DPU all-reduce。因 host 拿到的是完整 `scores`，softmax 不需要 flash 式在线累计（running max/sum），两个 GEMV 各自独立分块即可；flash 式合并仍留 `[阶段2]`（对齐问题 2 的 463/491 行）。

**输出投影的 all-reduce** 不在下面的函数内：整个注意力块还含 `W_O`，其 contraction 维正是被切的 head 维，`attn_out @ W_O` 产出 `Partial`，需一次 `Partial → Replicate`（all-reduce，经 host 两跳）才能喂给残差 add 与下一层 LayerNorm，由下游 redistribute 边承担。KV 本地化消除的是"每步按全部历史 token 收集 KV"的 per-step all-gather，而非输出投影处每层一次的 all-reduce。

```python
def attention_on_dpu(kv: PIMStaticKVCache, dpu_id, layer, q_local, mask, valid_len):
    """算单 DPU 上的注意力，KV 本地驻留、按 tile 分块读，softmax 在 host。
       输出投影 W_O 及其 Partial->Replicate 的 all-reduce 不在本函数内，由下游 redistribute 边承担。
       入:  kv/dpu_id/layer；q_local —— 本地 Q 分片（按 Q head 索引）；
            mask —— prefill_mask / decode_mask 的产物；valid_len —— 有效历史长度
       出:  {q_head: 注意力输出}（按 Q head 索引，不是 KV head）。"""
    spec = kv.specs[dpu_id]
    out = {}
    for kv_h in spec.kv_heads:                          # GQA：一个 KV head 服务多个 Q head
        for q_h in spec.q_heads_by_kv[kv_h]:            # ★ 遍历该 KV head 服务的每个 Q head
            # ① QK^T：DPU 上 tiled GEMV，逐 tile 读 K，拼出完整 scores
            scores = tiled_qk(kv, dpu_id, layer, kv_h, q_local[q_h], valid_len)  # [max_seq]
            scores = scores / sqrt(spec.head_dim) + mask
            # ② softmax：送回 host 做（B 类算子留 host），得完整 weights
            weights = host_softmax(scores)              # host 胶水，非 DPU 调用
            # ③ weights @ V：DPU 上 tiled GEMV，逐 tile 读 V，加权累加
            out[q_h] = tiled_av(kv, dpu_id, layer, kv_h, weights, valid_len)
    return out
```

#### 四. 工作量与推进建议

KV cache 管理器逻辑简单，其所需要的内存排布、跨 DPU 搬运、kernel 执行分别委托给问题 8、问题 3、问题 5，本节只负责"算布局 + 推指针 + 生成 mask"。**代码量约 200~350 行 Python**，构成如下：

- **编译期布局**：`build_kv_layout`（切定长子块、算每块 MRAM 偏移，并给出含对齐 padding 的 `kv_allocated_bytes`，问题 8 用它排 arena）+ `kv_bytes`（仅未对齐下界估算，非分配依据），约几十行；
- **运行时管理**：`PIMStaticKVCache` 的 `update` / `read_tile`，约百余行，第 1 阶段 NumpyBackend 上顺带做真实读写以验数值；无内部序列指针，位置由编排器 `DecodeState.valid_len` 传入；
- **mask 生成**：`prefill_mask` / `decode_mask` 两个函数，形状不同、逻辑简单，约几十行。

推进建议：

- KV 读取按 key token 分块（`read_tile`），**单 tile 的 WRAM 占用受 WRAM 容量约束**（如 `max_seq=256/head_dim=64/fp32` 单 head 整块已 64KB，远超 WRAM，不能满读），tile 尺寸由问题 5 的 `ttir→upmem` 桥按 WRAM 容量定；第 1 阶段 `max_seq` 取小值（如 128 或 256）配合小模型控制 KV 区总量，但读取路径不依赖"整块恰好装得下 WRAM"这一前提；
- 先在 NumpyBackend（N 块独立 numpy 缓冲区模拟 N 个 DPU）上验证 `update` / `read_tile` / mask 的数值对齐，再上真实硬件；接入 FlagTree 后 `update` / `read_tile` 下沉进 kernel，Python 侧退回只管"偏移 + tile 划分 + mask"；
- **GQA 模型（如 Llama 3）的 KV 区按 KV head 数计算**，而非 Q head 数（一个 KV head 对应多个 Q head）；且 `KVRegionSpec` 必须带 `q_heads_by_kv` 映射，注意力按 Q head 遍历、每个 KV head 服务其对应的多个 Q head——只按 KV head 遍历会丢失 Q head 输出、使 attention 输出 head 数塌成 KV head 数，是最易算错的一处；第 1 阶段固定模型 GPT-2为 MHA，，该映射退化为恒等、不会触发上述塌缩，但代码仍按 GQA 通用写法实现，宽化到 GQA 模型时零改动；
- **序列位置单一真值源**：位置只由编排器 `DecodeState.valid_len` 持有并传入，`PIMStaticKVCache` 不自持指针、无 `advance`；避免"三处真值源"不同步导致 mask 位置与写入位置错位；
- softmax 第 1 阶段在 host（B 类算子留 host），是宽白名单里优先级最高的收回项（规约天然本地、在热路径上）；
- prefill（计算密集）与 decode（访存密集）的最优切分可能不同，或需一次中途重分布，作为后续优化项，第 1 阶段两图共用同一套固定切分。

#### 五. 最终产物与依赖

KV cache 独立为一个 Python 文件，兼编译期布局与运行时管理：


| 文件          | 含义                                                                                                                                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kv_cache.py` | `KVRegionSpec`（含 `q_heads_by_kv`）+ `build_kv_layout`（编译期算大小与偏移，产出 `kv_allocated_bytes` 喂问题 8）+ `kv_bytes`（仅下界估算）+ `PIMStaticKVCache`（运行时 `update` / `read_tile`，无内部指针）+ `prefill_mask` / `decode_mask` |

它交付的产物分两类：编译期的 `kv_allocated_bytes`（含对齐 padding 的真实占用）与 `(layer, head)` 偏移表，供问题 8 排进 arena；运行时的 `update` / `read_tile`（按 tile 分块）/ mask 逻辑，供问题 6 解码循环调用（softmax 在 host）。

依赖：

- **PyTorch（必需）**：借用 HuggingFace `StaticCache` 的接口形态与 mask 处理；`torch.export` 的定长形状要求与本节的静态预留天然契合。
- **NumpyBackend / HAL（必需）**：`read_local` / `write_local` 在各 DPU 独立缓冲区上做 `update` / `read_tile`，厂商 SDK 到位前先在假后端验证数值。
- **问题 8 的 `plan.kv_base`（必需）**：KV 区基址由内存规划器给定，区内偏移表由本节 `build_kv_layout` 算出；KV 区大小以本节 `kv_allocated_bytes` 为准。
- **问题 6 的 `DecodeState.valid_len`（必需）**：有效长度由编排器跨预填充 / 解码两图持有，作为唯一位置真值源传入本节。
- **明确不依赖**：vLLM PagedAttention（其统一显存 + 动态分块的范式与本架构冲突）。

### 问题 8：内存管理

#### 一. 问题描述

DPU 本地一开机什么都没有——权重、输入、kernel 二进制全得由主机显式加载进去；而且 DPU 上没有 allocator，运行时没人负责分配、回收本地内存。所以每个 DPU 那块 ≤8GB 内存怎么摆，必须在**编译期**用一个规划器（planner）一次算死，产出一份静态内存规划蓝图 DPU_k.plan`。

规划器和切分（问题 2）**双向耦合**，不能分开解：

- **切分 → 内存（落实）**：切分定 placement——每块权重的本地分片 shape、每个 DPU 分到哪些 head / 层；内存规划把这些 shape 换算成字节、排进内存、算出 offset，是切分的"落地执行"。
- **内存 → 切分（反馈）**：规划最后做容量校验 `total ≤ 单 DPU 内存 − 系统预留`，装不下就把结果反馈给切分 pass **重切**。

#### 二. 内存管理器功能描述

内存管理器（规划器）负责**给归属某个 DPU 的全部张量，在那块 ≤8GB 内存里各算出一个 MRAM offset，拼成静态蓝图 `DPU_k.plan`。分配单位是 DPU。** 一个 DPU 上会顺序跑该模型的多个子图（QKV 投影、MLP、各层……），规划器对**归属该 DPU 的全部节点统一规划一份蓝图**。"全部节点"含两个层次：同一张导出图内的多个子图取并集统一规划；prefill 与 decode 两张导出图（问题 6）之间，按下文三."prefill / decode 两图的联合规划"规则联合规划——权重与 KV 两图共用同一套 offset，激活区两图取峰值互斥复用。每个 DPU 的内存分三个区：

```
┌──────────────┬──────────────────┬────────────────────┐
│    权重区     │    KV 区(预留)    │    激活区(复用)      │
│ 常驻、全程不变 │ 按 max_seq 定长预留│ 临时 buffer、算完可覆盖│
└──────────────┴──────────────────┴────────────────────┘
```

三个区的分配对象、生命周期、大小来源如下（按"是否常驻、是否复用"归类）：


| 分配对象                                                                     | 归哪个区 | 生命周期                                 | 大小怎么来                      |
| ---------------------------------------------------------------------------- | -------- | ---------------------------------------- | ------------------------------- |
| 各类 Linear 权重、注意力 QKV 权重                                            | 权重区   | 开机搬一次、全程常驻                     | 切分后的本地分片 shape × dtype |
| KV cache                                                                     | KV 区    | pinned，跨 step 存活、只增不减、每步必读 | 公式按`max_seq` 定长预留        |
| 中间激活（Y1/Y2…、attention score）、每 step 喂进来的输入 token / embedding | 激活区   | 算完即可覆盖                             | liveness 区间内的峰值           |

**蓝图同时依赖模型、切分、硬件三组输入，喂给同一个规划器：**


| 来源                            | 参数                                                        | 决定蓝图的哪部分         |
| ------------------------------- | ----------------------------------------------------------- | ------------------------ |
| 模型（从`node.meta["val"]` 读） | 层数、hidden、head 数、head_dim、d_ff、dtype、max_seq       | 各区**字节数**           |
| 切分（问题 2 的 placement）     | 每个权重的本地分片 shape、每个 DPU 分到哪些 head / 层       | 哪块权重落哪个 DPU、多大 |
| 硬件（厂商 SDK）                | 内存总量（≤8GB）、DMA 对齐、地址编译期烧死还是 launch 传参 | 对齐、容量上界、地址给法 |

模型决定"要放多少"，切分决定"哪块放哪个 DPU"，硬件决定"怎么摆、摆得下吗、地址怎么给"。

**`sys_reserve`（系统预留）扣除项**：规划器可用预算为 `mram_budget − sys_reserve`，其中 `sys_reserve` 须覆盖四项非三区占用——① kernel 二进制（`dpu_load` 载入的代码占 MRAM）；② DPU runtime、tasklet 栈及 runtime 元数据；③ WRAM staging 相关的对齐余量；④ 单个 kernel 运行期在 MRAM 上使用的中间缓冲（workspace / scratch）。kernel 二进制由 `dpu_load` 管理、不进三区 offset 分配，以 `sys_reserve` 一并扣除。第 4 项中，第 1 阶段窄白名单（GEMV、逐元素）的算子多数无需 MRAM workspace，统一按保守常量并入 `sys_reserve` 上界；`[阶段2]` 若出现需要大块 MRAM workspace 的算子，改为按算子登记 workspace 字节、单独计入该 DPU 预算。第 1 阶段 `sys_reserve` 取保守常量（按 SDK 文档估上界），SDK 到位后精确化。

**管理器分编译期、运行时两层**：


| 层     | 干什么                                                        | 产物               |
| ------ | ------------------------------------------------------------- | ------------------ |
| 编译期 | 算三个区的大小与每个张量的 offset，做容量校验                 | `DPU_k.plan`       |
| 运行时 | 照蓝图把权重一次性搬进去、把 offset 当 launch 参数传给 kernel | 不新分配，只搬数据 |

#### 三. 实现思路

三个区各有各的布局办法，难点只在激活区。

**权重区（静态打包，无算法）**：权重是常量、全程不变，是该 DPU 上**所有子图权重的并集**（一次推理要顺序跑完这些子图，权重必须同时常驻，不能跑完一个子图换一批——换权重就是每步重搬，带宽立刻爆）。按稳定顺序把每块本地分片对齐后依次排布，一次算出各自 offset，对应 ExecuTorch 的常量段。

**KV 区（布局预留）**：按 `max_seq` 预留一块定长空间，是该 DPU 所持所有层 / head 的并集。**职责边界：本节只给 KV 区基址 `kv_base`、把它当三区之一排进内存；区内每个 `(layer, head)` 的 offset 与 KV 区真实占用（含对齐 padding）都由问题 7 的 `build_kv_layout` 基于此 `kv_base` 一并给出——本节用它返回的 `kv_allocated_bytes` 推进激活区起点（GQA 按 KV head 数算），不重复算内部 offset，也不用 `kv_bytes` 公式重算大小（否则对齐 padding 累积会侵入激活区）。** KV 区跨 step 常驻、不属于任何一张图，预填充图和解码图共用同一基址、不随图切换重新规划（对齐问题 6 的跨图状态推进）。运行时的"追加"只是按当前 `valid_len` 往这块预留区后写，不涉及分配。

**激活区（复用，唯一有算法的部分）**：中间结果算完即可覆盖，靠 **liveness 分析 + 贪心装箱**让生命周期不重叠的张量共用地址——算出每个临时张量的 `[首次产出, 最后被用]` 区间，按大小从大到小放到最低的不冲突 offset。复用**恰恰跨子图发生**：前一个子图的激活过了 death 点，后一个子图的激活就覆盖同一块地址，所以要把所有子图的临时张量放到同一条时间轴（全局拓扑序）统一算。这块借 ExecuTorch 的 `greedy` 算法。第 1 阶段激活区也按 `max_seq` 满预留即可，按实际长度的紧凑复用是 `[阶段2]`。

**prefill / decode 两图的联合规划**：第 1 阶段用两张形状不同的静态图（问题 6），二者共享同一份权重与同一块 KV。持久区（权重区 + KV 区）在编译期只规划一次，两张图共用完全相同的 offset——否则 prefill 把 KV 写到地址 X、decode 按地址 Y 读，会产生静默数据错误。激活区是两图的互斥 overlay：prefill 跑完再进 decode，两者激活不同时存活，可复用同一段地址范围，激活区大小取 `max(prefill 激活峰值, decode 激活峰值)`。规划器为两图各保留一份激活 offset 表（同一逻辑张量两图 shape 不同、offset 也不同），共享同一激活基址。本节产物由此确定：持久区两图共用一套 offset，激活区按图各存一份 offset 表、共享基址、大小取两图峰值。

**任何地址复用的安全性都必须由 liveness 分析给出，不能由通信库、算子 kernel 或编排器自行假定；这条要求不区分复用发生在哪个方向。** 一块 buffer 的地址被下一次写覆盖之前，必须确认它此前的全部读者都已执行完毕，否则会覆盖尚未消费的数据，产生不报错、不崩溃、只是结果错误的静默 bug。这类复用有两种发生方式，二者对正确性的要求完全相同：

- **跨 host/DPU 边界的复用**：通信原语（如 `all_reduce`）向目标地址 `copy_to` 写回结果，覆盖该地址此前的旧值。问题 2 的算子布局规则表（问题 2 二.（8））已说明 `Partial` 张量可能同时被多条边消费——例如两个 `Partial` 输入的逐元素相加——此时同一个 `Partial` 张量除了参与 `Partial → Replicate` 的重分布外，还可能有另一个读者尚未执行。
- **同一 DPU 上的本地复用**：激活区 `greedy_reuse` 判定两个生命周期不重叠的本地张量共用同一 offset 后，后一个 kernel 写入该 offset 之前，必须确认前一个 kernel 的全部读者（可能是本地的下一个算子，也可能是发往通信库的 `copy_from`）都已执行完毕。这类复用完全发生在单个 DPU 内部、不经过任何 redistribute 边，此前仅由"编排器按拓扑序串行发射"这一调度细节偶然保证，本身没有被纳入显式的依赖描述。

两种情况在数据结构上没有区别：都是"某个 `(loc, offset)` 上，下一次写之前要等哪些读者"，`loc` 取 `("dpu", dpu_id)` 或 `("host", None)`。因此本节不再区分"redistribute 目标地址"与"本地复用地址"，而是对**每一个**由 `greedy_reuse` 判定复用的地址，统一产出该地址在复用前必须等待的读者列表，记为该地址上的 `pending_readers`。

具体做法：把 `redistribute` 节点的 `copy_from`（从源地址读出到主机）与写回（写入目标地址）都作为该地址上的一次普通读、写事件，纳入本节激活区依赖的同一套 liveness 分析（即 `transient_tensors` 统计的读写关系，包含 `node.users` 之外由通信边隐式产生的读写）。据此可以推出：

- 若某个张量的全部读者中，最后一个读者已经执行完毕才发生下一次写，则该张量的死亡点就是这最后一个读者，下一次写复用同一 offset 是安全的——但这是 liveness 分析推导出的结论，不是通信库或编排器预先假定的行为；
- 若该张量在某次写发生前还有其他读者尚未执行，liveness 分析会得出更晚的死亡点，规划器必须为这次写分配一个当前空闲的新 offset，而不允许复用。

因此，"某次写能否复用原 offset"是 `greedy_reuse` 计算的一个结果，而非本节或问题 3、问题 6 预先规定的默认行为；下文规划器骨架的 `transient_tensors` 与 `greedy_reuse` 均按此扩展，产出覆盖全部复用地址（不止 redistribute 目标）的 `pending_readers` 表，供问题 6 生成 `ExecutionPlan`（结构定义见问题 6 三.（2））时统一读取、转换为对应命令的 `waits` 字段。

补充两点执行语义，避免异步下歧义：

- **事件以 DMA 落地为准**：`writers` 依赖表与 `pending_readers` 跟踪的"写完 / 读完"，指 DMA 传输在硬件上真正落地完成，而非主机侧发起调用返回。第 1 阶段同步 DMA 下二者一致；`[阶段2]` 引入异步 DMA 后，以传输落地为准（对齐问题 6 `hal.wait` 与 `fence_copy` 的语义，见问题 6 三.）。
- **读者须映射到命令 `id`**：`pending_readers` 记录的读者，在问题 6 生成 `ExecutionPlan` 时须映射到具体命令的 `id`。`copy_from` 这类读者在 FX 图上没有 `node.users` 边，因此本节为每个读写事件赋一个稳定标识——本地读写取来源节点，redistribute 读写取 `edge_id` 加 segment 序号——问题 6 据此把每个读者翻译为对应 `dma_in` / `launch` 命令的 `waits`，保证本节的 liveness 结论与问题 6 的命令依赖逐条对齐。

需要注意的点：

- **KV"初始化"= `valid_len` 归零，不载入数据**。KV 区无需初始化数据，仅置 `DecodeState.valid_len = 0`；预留区脏值由 mask 盖掉。
- **Embedding 权重第 1 阶段不落 DPU**。方案里 embedding 第 1 阶段是 `Replicate`，最简做法是 embedding lookup 直接在 host 胶水里做，DPU 只收 embedding 之后的激活；要落 DPU 就按一份 Replicate 常量算进权重区。
- **注意力权重与 KV 的切分必须一致**。KV 按 head 切是硬约束（问题 7），注意力权重也按 head 切，本 DPU 持有的 head 数 `|H_k|` **同时**决定注意力权重区大小和 KV 区大小——这两个数必须来自同一份 placement，不能各算各的。
- **host 侧中转缓冲不进 `DPU_k.plan`**。redistribute 的归约缓冲、合并缓冲、pinned staging 等都在主机内存上，主机有常规分配器（第 1 阶段用 numpy / CPU 内存），按需分配即可，无需静态 MRAM 规划，因此不出现在本节的三区 offset 表中。本节只规划各 DPU 的 MRAM；host 缓冲的容量与生命周期由编排器（问题 6）在主机侧管理。
- **view 类算子在 DPU 边界前物化，禁止未经证明安全的原地算子**。transpose、permute、reshape 产出视图（view），与源张量共享同一块存储，存储跨度不等于 `shape × dtype`。第 1 阶段规则：进入 DPU 子图边界前，所有 view 物化为连续张量（`bytes_of(local_shape)` 由此成立），并禁止未经别名分析证明安全的原地算子；确需保留、不物化的纯视图不单独占用激活区地址，而是与源张量共享同一存储对象（借 ExecuTorch 的 view / `mem_obj_id` 语义），使该存储在任一别名存活期间都不被 `greedy_reuse` 复用。此处只处理"同一存储被多名共享"（别名），与前文"原地写回安全性"处理的"不同张量复用同一地址"（WAR）是两件事。完整物理描述（`storage_span` / `strides` / `storage_offset` / `alias_group` / `mutability` / `memory_space`）与"不物化、按真实跨度规划"留 `[阶段2]`。

**规划器骨架（编译期，每个 DPU 各调一次）。** 输入 / 输出契约：

```
输入：
  prefill_nodes, decode_nodes   # 该 DPU 在两张导出图上各自的【全部】节点(每张图内多子图取并集)
  dpu_id
  placement_map  # 问题 2 产出的 PIMTensorSpec：取 shard_map[dpu_id].local_shape
                 #   拿本地分片 shape、拿本 DPU 分到哪些 head / 层
  model_meta     # 层数、hidden、head 数、head_dim、d_ff、dtype、max_seq(多从 node.meta["val"] 读)
  hw             # mram_budget(≤8GB)、align(DMA 对齐)、sys_reserve
输出：
  DPU_k.plan = { weight:{name->off}, kv_base:off,
                 act_base:off, act_prefill:{name->off}, act_decode:{name->off}, total }
                 # 权重与 kv_base 两图共用同一套 offset；激活区两图各一张表、共享 act_base；
                 #   total = 持久区 + max(两图激活峰值)
                 # KV 区只给 kv_base；区内 (layer,head) offset 与真实占用（kv_allocated_bytes）由问题 7 build_kv_layout 展开
  并回填 PIMTensorSpec.shard_map[dpu_id].mram_offset，供问题 3/6/7 共用同一数据源
  并为两图各产出一份 pending_readers（两图各有独立 ExecutionPlan），
                 #   每份覆盖该图全部复用地址、不止 redistribute 目标地址
                 #   该地址在 greedy_reuse 的 liveness 分析中得到的、复用前必须等待的读者节点列表
                 #   供问题 6 生成 ExecutionPlan 时统一转换为命令的 waits 字段
```

```python
def plan_dpu(prefill_nodes, decode_nodes, dpu_id, placement_map, model_meta, hw):
    plan = DPUPlan(weight={}, kv_base=0, act_base=0,
                   act_prefill={}, act_decode={}, total=0,
                   pending_readers_prefill={}, pending_readers_decode={})   # 先建产物对象
    off = 0
    # ① 权重区（持久，两图共用同一套 offset）：两图权重相同，取并集去重后顺序打包 + 对齐
    for w in sorted(weights_of(prefill_nodes) | weights_of(decode_nodes), key=lambda w: w.name):  # 稳定顺序，可复现
        local_shape = placement_map[w.name].shard_map[dpu_id].local_shape
        plan.weight[w.name] = off
        placement_map[w.name].shard_map[dpu_id].mram_offset = off   # 回填，供下游共用
        off += align_up(bytes_of(local_shape, w.dtype), hw.align)
    # ② KV 区（持久，两图共用同一 kv_base）：本 DPU 所持层 / (KV)head 的并集，按 max_seq 定长预留
    #    本节只定基址；区内 offset 与真实占用都由问题 7 build_kv_layout 基于此 kv_base 展开
    #    ★ 用 build_kv_layout 返回的 kv_allocated_bytes（含对齐 padding）推进 off，
    #      不用 kv_bytes 公式重算——否则对齐 padding 累积会让 KV 实际写入范围侵入激活区
    plan.kv_base = off
    kv_spec = build_kv_layout(kv_spec_of(dpu_id, kv_base=off), hw.align)
    off += kv_spec.kv_allocated_bytes
    # ③ 激活区：prefill / decode 互斥 overlay，共享同一基址，大小取两图峰值
    #    每张图内把【所有子图】的临时张量放到同一条时间轴(全局拓扑序)统一算
    #    liveness 跨子图：前一子图激活过 death 点，后一子图激活即可覆盖同一地址
    #    tensors 中每个条目携带 redistribute 边隐式产生的读写事件（见 transient_tensors）
    plan.act_base = off
    tp = transient_tensors(prefill_nodes)
    td = transient_tensors(decode_nodes)
    plan.act_prefill, end_p = greedy_reuse(tp, base=off)   # 两图共享 act_base=off，各自装箱
    plan.act_decode,  end_d = greedy_reuse(td, base=off)
    plan.total = max(end_p, end_d)                         # 互斥 overlay，取峰值
    # 对 greedy_reuse 判定复用的每一个地址（不区分是否为 redistribute 目标），记录复用前必须
    # 等待的读者列表；两图各有独立 ExecutionPlan，故分别产出。问题 6 生成 ExecutionPlan 时对每条
    # 写命令查表取 waits，问题 3 不再单独计算 dst_ready_after
    plan.pending_readers_prefill = pending_readers_of_reused_addresses(tp, plan.act_prefill)
    plan.pending_readers_decode  = pending_readers_of_reused_addresses(td, plan.act_decode)
    assert plan.total <= hw.mram_budget - hw.sys_reserve            # 容量校验，超了 → 反馈重切
    return plan
```

上述 helper 均为伪代码占位，落地时按下表实现（表中形参名 `dpu_nodes` 泛指"传入的那组 DPU 节点"；`plan_dpu` 骨架按图分别以 `prefill_nodes` / `decode_nodes` 调用同一 helper）：


| helper                                              | 职责                                                                                                                                                                                                                                   | 实现提示                                                                                                                                                                                         |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `weights_of(dpu_nodes)`                             | 从该 DPU 节点中挑出权重常量                                                                                                                                                                                                            | FX 中为`get_attr` / placeholder 常量                                                                                                                                                             |
| `transient_tensors(dpu_nodes)`                      | 挑出临时激活（算完即废的中间结果），并把 redistribute 节点的`copy_from`/写回、以及本地 kernel 间的读写计入该地址的读写事件；view 类输出二选一：物化的视图作为独立连续张量正常计入，未物化的纯视图不单独占地址、与源张量共享存储对象 id | `call_function` 的输出，非常量；redistribute 边的读写取自问题 2 的 `RedistributeEdge`，本地读写取自 `node.users`；view / 别名借 ExecuTorch 的 `mem_obj_id`，第 1 阶段默认物化，纯视图共享存储 id |
| `kv_spec_of(dpu_id, kv_base)`                       | 取该 DPU 的`KVRegionSpec`（带 `kv_base`）                                                                                                                                                                                              | 喂给问题 7 的`build_kv_layout`，用其返回的 `kv_allocated_bytes` 推进 off                                                                                                                         |
| `greedy_reuse(tensors, base)`                       | 激活区 liveness + 贪心装箱                                                                                                                                                                                                             | 借 ExecuTorch`greedy`，`dpu_id → mem_id`                                                                                                                                                        |
| `pending_readers_of_reused_addresses(tensors, act)` | 对`act` 中每一个被判定复用的地址（`(loc, offset)`），从 liveness 分析的读写关系中筛出复用前必须等待的读者节点，不限定 `loc` 是否为 redistribute 目标                                                                                   | 复用`greedy_reuse` 内部算出的 `[首次产出, 最后被用]` 区间，取区间右端点对应的读者集合；`loc` 为 `("dpu", dpu_id)` 或 `("host", None)`                                                            |
| `bytes_of(shape, dtype)`                            | 按本地分片 shape + dtype 算字节                                                                                                                                                                                                        | —                                                                                                                                                                                               |
| `align_up(n, align)`                                | 向上对齐到 DMA 对齐边界                                                                                                                                                                                                                | —                                                                                                                                                                                               |

**运行时（一次性载入，之后常驻）。** "一次性载入"仅指搬入权重，三个区启动时状态如下：


| 区     | 启动时状态                                               |
| ------ | -------------------------------------------------------- |
| 权重区 | `copy_to_dpu` 搬入数据（唯一真正载入的区），全程常驻不动 |
| KV 区  | 仅留空地址，内容为空；运行时逐步 append K/V              |
| 激活区 | 仅留空地址；每步计算现算，按 offset 复用覆盖             |

原因：权重是常量、其值在编译期已定，可一次搬入且全程不变（且绝不能每步重搬，否则带宽爆）；KV 启动时尚无值（随 decode 逐步 append），激活亦无值（每步现算）——二者启动时仅"预留地址范围"，无数据可搬。

```python
for k in DPU:
    copy_to_dpu(k, plan[k].weight, 该 DPU 的权重分片)   # 只在启动时做一次
    dpu_load(k, 该 DPU 子图编译出的 kernel 二进制)
```

权重开机搬一次即常驻，**绝不随 step 重复搬**（否则每步搬最大的权重，带宽立刻爆）。

**【待确认项，问题 3 与问题 8 共用】MRAM 地址是"launch 时传参"还是"编译期烧死"**：前者编排器可令"上个 kernel 输出地址 = 下个输入地址"接链、更灵活，后者只能编译期写死；取决于厂商 SDK，须待 SDK 文档确定。兜底方案（不阻塞第 1 阶段）：以 HAL 隔离——规划器永远只产出逻辑 offset 表，由 HAL 翻译成 launch 参数或编译期符号；NumpyBackend 上两种语义均可模拟，故 SDK 到位前即可验证规划器数值正确，换硬件仅改 HAL 一处。

**为什么切法把问题 1、2、8 绑成一个约束**：设一层权重 `W = [K, N]`，一个 DPU 装不下要切两半——列切 `Shard(1)`（`[K, N/2]×2`）各算一半、拼起来即可，只需 all-gather；行切 `Shard(0)`（`[K/2, N]×2`）各算部分和，必须相加，需要 all-reduce。而跨 DPU 的收集 / 归约都得走 `DPU→host→DPU` 两跳（无直连）。所以切法同时决定了**装多少**（问题 8 容量）和**通信类型 / 通信量**（问题 3），不能先随便切再看够不够——三者得一起解。三区拼起来 `total = 权重区 + KV 区 + 激活区`，最后的容量校验 `total ≤ 单 DPU 内存 − 系统预留` 装不下就反馈重切，正是这个联合约束落到代码里的闭环点。

`[阶段2]` 的自动反馈重切由一个顶层驱动 pass 统筹：读问题 1/2/8 共写的同一张 meta 图，执行"规划 → 校验 → 按溢出 DPU 及溢出量调整（迁 head/层 → 改切法 → 回退算子至 host）→ 重规划"的定点迭代，带 `max_iter` 收敛保护，并将跨 DPU 边的 host 成本反馈给问题 1 的分区器。第 1 阶段不做此迭代，维持"手工配置 + `assert` 校验"的人工闭环。

#### 四. 工作量与推进建议

规划器是一个纯编译期 pass，重活只在激活区的 liveness + 贪心装箱，其余两个区都是顺序打包，工作量不大。

- **能借的 PyTorch 现成件**：张量大小从 `node.meta["val"]`（FakeTensor 带 shape / dtype）算；拓扑序和最后使用点用 FX 的 `graph.nodes` 顺序 + `node.users`——但 `node.users` 只覆盖图上的显式数据流边，覆盖不到 redistribute 边隐式产生的 `copy_from` 读取，因此 `transient_tensors` 必须额外把问题 2 的 `RedistributeEdge` 计入同一份读写清单，否则原地写回的判定会遗漏这类读者；激活区复用直接抬 ExecuTorch `memory_planning` 的 `greedy` 算法，并把 **`dpu_id` 映射成它的 `mem_id`**——`mem_id`（一块独立 arena）的抽象天然对上"每个 DPU 一个独立地址空间"。只借编译期这半，不借其 runtime（它假设单地址空间，我们是 N 个物理隔离的 DPU）。
- **推进顺序**：先在 NumpyBackend（N 块独立 numpy buffer 模拟 N 个 DPU）上验证规划器数值正确，落在落地路线第 1 步、零硬件。
- **第 1 阶段简化**：`max_seq` 取小值（如 128 或 256）配合小模型，保证一次装得下；激活区也按 `max_seq` 满预留即可，按实际生命周期的紧凑复用是 `[阶段2]`；装不下自动反馈重切的闭环也是 `[阶段2]`。

代码量归在"图编译器（问题 1 / 2 / 8）"一档，规划器本体主要是薄封装加一个借来的 `greedy`，**约 150~300 行 Python**（三区 offset 计算与容量校验薄封装 + 激活区 `greedy_reuse` 复用；权重区、KV 区均为顺序打包）。

#### 五. 最终产物与依赖

规划器是图层（编译期）的一个 pass，产出一个 Python 文件：


| 文件             | 含义                                                                                                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mem_planner.py` | `plan_dpu`（每个 DPU 各调一次，算三区 offset + 容量校验）+ 权重区静态打包 + 激活区 `greedy_reuse`；KV 区只定 `kv_base`，区内 offset 调问题 7 的 `build_kv_layout` |

它交付的产物是每个 DPU 一份 `DPU_k.plan`（持久区权重与 KV 两图共用同一套 offset，激活区 prefill / decode 两图各一张 offset 表、共享 `act_base`），并回填 `PIMTensorSpec.shard_map[dpu_id].mram_offset`、按图各一份的 `pending_readers`（`pending_readers_prefill` / `pending_readers_decode`），让问题 3（通信地址）、问题 6（生成 `ExecutionPlan` 的 `waits`）、问题 7（KV offset）共用同一数据源。

依赖：

- **PyTorch（必需）**：`node.meta["val"]` 算张量字节数，FX `graph.nodes` / `node.users` 给拓扑序与 liveness 区间。
- **ExecuTorch 的 `greedy` 算法（借思路）**：激活区复用，`dpu_id` 映射 `mem_id`；只借编译期，不借其 runtime。
- **问题 2 的 placement（必需）**：本地分片 shape、每个 DPU 分到哪些 head / 层。
- **问题 7 的 `build_kv_layout`（必需）**：KV 区真实占用（`kv_allocated_bytes`，含对齐 padding）与区内 offset；`kv_bytes` 仅为未对齐下界估算，非分配依据。
- **HAL（必需）**：把逻辑 offset 翻译成 launch 参数或编译期符号；厂商 SDK 到位前先在 NumpyBackend 上验数值。
- **明确不依赖**：任何运行时 allocator（独立地址空间无 allocator，全部编译期定死）。

---

## 五、工作量评估

第 1 阶段自写量合计约 **6200~8800 行 Python + MLIR**，具体如下表所示。当前只考虑 AI 编译器的研发部分与 GeneSim 接入，未考虑测试。

**汇总表：**


| 组件                  | 对应问题       | 主要内容                                                                                                                                                                                                                             | 自写 / 借                                                   | 代码量（估）    |
| --------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | --------------- |
| 图编译器              | 问题 1 / 2 / 8 | 图拆分、切分传播、redistribute 标记、内存三区规划                                                                                                                                                                                    | 参考 PyTorch/DTensor/ExecuTorch 编译期，自写标注与规划 pass | 约 1250~1600 行 |
| 算子编译器            | 问题 5         | `ttir→upmem` 桥：访存下降、tiling、DPU 内同步                                                                                                                                                                                       | 参考 Cinnamon`upmem` dialect + FlagTree                     | 约 500~1000 行  |
| 算子库                | 问题 5         | A 类主力算子（GEMM/GEMV/逐元素）数学逻辑借 FlagGems、访存下降由桥统一做，但每个算子仍需**I/O 适配**与 aten→kernel 登记接入；B/C 类规约算子（softmax/LayerNorm，输入输出逻辑随近存重构而变）第 1 阶段留 host，PIM 版重写归 `[阶段2]` | 参考 FlagGems 数学逻辑，自写 I/O 适配 + 接入                | 约 150~350 行   |
| 通信库                | 问题 3         | redistribute 下降为主机中转 DMA 序列，架在厂商 SDK 之上                                                                                                                                                                              | **自写**                                                    | 约 1000~1500 行 |
| 运行时（编排器 + KV） | 问题 6 / 7     | `ExecutionPlan` 生成器 + 解释执行 + 解码循环 + HAL + 假后端 + 张量抽象 + 对拍器 + KV cache                                                                                                                                           | **自写**（借 ExecuTorch 编译期骨架）                        | 约 2350~2800 行 |
| GeneSim 成本桥接      | 问题 4         | 成本提取器 + FlagTree 编译驱动 + IR 成本分析器（TTIR→pim mlir）+ 局部→全局成本换算 + 算子归类表 + sidecar 导出 + 回归脚本                                                                                                          | **自写**（借 FlagTree 编译前端、GeneSim 既有图骨架/trace）  | 约 950~1550 行  |

## 六、实施阶段

### 6.1 三阶段总览

1. **第 1 阶段（本期）——能跑对**：固定 shape 全链路打通，和单卡 PyTorch 数值对齐。`max_seq` 为常量、满算加 mask、切分手工指定、模型选定 **GPT-2**（HuggingFace `openai-community/gpt2`，研发初期可配置，例如：n_layer=4, n_head=8, n_embd=512, max_seq=128，编排器可串行。GeneSim 接入同属本期，分两步：第 1 步接 FlagTree 原生 TTIR 先走通 `HF→…→GeneSim` 全链路（访存侧近似）、不依赖问题 5，可最先起步；第 2 步随问题 5 产出的 pim mlir 接入、补齐访存侧精度。
2. **第 2 阶段——跑得全**：在第1阶段的基础上，添加序列变长（符号维加上界）、代价模型、自动切分、激活区紧凑复用、图拆分宽化、异步 dispatch。
3. **第 3 阶段——跑得快**：性能优化，包括：算子融合、把单算子逐次下发合并为批量下发以摊薄主机-设备通信开销，对齐第七节第 5 条与第八节的通信开销风险。

### 6.2 并行分组

**代码分成"编译期图分析"和"运行时执行"两条链，前者产出静态蓝图、不经过后端，后者在 NumpyBackend，也就是N 块独立 numpy buffer 模拟 N 个 DPU上验证；两条链只在数据（`node.meta` 标注）上耦合，可大幅并行。** 算子编译器（问题 5）与 GeneSim 成本桥接（问题 4）则是两条可独立并行推进的支路——前者仅被真硬件/厂商 SDK 阻塞，后者的 TTIR 接入步不依赖任何组件、可最先起步（pim mlir 步依赖问题 5）。

依赖关系：

```
                    ┌─────────────── 编译期链（纯 CPU，不经 backend）───────────────┐
问题1 图拆分打标 ──▶ 问题2 切分传播/redistribute 标注 ──▶ 问题8 内存三区规划(offset)
（白名单+分组）        （先钉 KV，再传播；同时需要 P7 的 KV 切分约束）      （吃 P2 placement + P7 build_kv_layout）
                                    │                                      │
                                    │ redistribute 边标注                   │ mram_offset 回填
                                    ▼                                      ▼
                    ┌─────────────── 运行时链（NumpyBackend 上验证）──────────────────┐
   ★ hal_numpy 假后端 ─(独立、可最先起，第1阶段最大一块)─┐
   问题3 通信库(redistribute→DMA 序列) ──┐                │
   问题7 KV runtime(append/read/mask) ──┼──▶ 问题6 编排器(exec_plan.py 生成 ExecutionPlan+execute_plan 解释+解码循环) ──▶ 对拍器逐节点对齐
                                        │       （吃 P1/2/3/8 全部蓝图 + hal_numpy）
   ─────────────── MLIR 线（可延后、与上面两链并行）───────────────
   问题5 ttir→upmem 桥 + hal_vendor  ──(仅被真硬件/厂商 SDK 阻塞，第1阶段窄白名单下不上主链)
              │ 产出 pim mlir
              ▼
                   ─────────────── GeneSim 线  ───────────────
   问题4 模拟器接入：第1步 TTIR 接入(不依赖任何组件，可最先起) ──▶ 第2步 pim mlir 接入(依赖问题5)
                  从 IR 抽 flops/data_bytes → 回填 GeneSim Operator 成本字段（GeneSim 由另一团队开发）
```

据此，第 1 阶段可分成**三组并行推进**，组内有序、组间靠数据契约解耦：


| 并行组                | 内容                                                   | 可否并行起步                                                                                               | 阻塞关系                                                      |
| --------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **A 组 编译期图分析** | 问题 1 → 问题 2 → 问题 8                             | 组内串行（P2 需 P1 的标注图，P8 需 P2 的 placement + P7 的 build_kv_layout）                               | 不依赖任何 backend，纯 CPU，最先动、风险最低                  |
| **B 组 运行时底座**   | `hal_numpy` 假后端 + 问题 3 通信库 + 问题 7 KV runtime | `hal_numpy` **可与 A 组同时最先起步**（不依赖图分析）；通信库/KV 需 A 组产出的标注做输入数据，但骨架可先写 | 三者汇入问题 6 编排器                                         |
| **C 组 算子编译器**   | 问题 5`ttir→upmem` 桥 + `hal_vendor`                  | 与 A、B 组完全并行；仅被真硬件/厂商 SDK 到位阻塞                                                           | 第 1 阶段窄白名单下 A 类走桥、B/C 类留 host，不阻塞全链路打通 |
| **D 组 GeneSim 接入** | 问题 4 成本桥接（成本提取器 + IR 分析器 + sidecar）    | 第 1 步 TTIR 接入**可与 A 组同时最先起步**（不依赖任何组件）；第 2 步 pim mlir 接入依赖 C 组问题 5 产出    | 独立评估旁支，不上执行主链、不阻塞功能全链路打通              |

**收口点**：执行栈有唯一总装节点——问题 6 编排器，它需要 A 组的三份静态蓝图 + B 组的通信库/KV/假后端，跑通后接对拍器逐节点对齐；C 组在 SDK 到位后把 `hal_numpy` 换成 `hal_vendor`、单 DPU 子图真上硬件。D 组（GeneSim 接入）是并行的性能评估旁支，自有收口——精化成本后的 `.ir` 能被 GeneSim loader 读入、scheduler 正常出结果，与执行栈的功能验证互不阻塞。

### 6.3 实施步骤与每步验证标准

步骤按先零硬件把数据流验对，再逐步接真件排序。**第 0~3 步完全不依赖存算一体硬件**，在纯 CPU / NumpyBackend 上即可验证，是风险最低的起点；标 `∥` 的步骤可与相邻步骤并行。GeneSim 接入（问题 4）是并行旁支，其两步（TTIR / pim mlir）分别挂在下面第 1 步、第 5 步之后，见步骤图中的 `[D]` 标注。

```
第 0 步 ∥：搭 NumpyBackend 假后端（B 组，可与第 1 步同时最先起）
  用 N 块独立 numpy buffer 模拟 N 个 DPU 的独立地址空间 + DMA 三件套
  ✅ 验证：单元测试——host↔DPU 的 copy_to/copy_from/push_xfer 读写数值正确；
           跨 buffer 无隐式共享（写 DPU_i 不影响 DPU_j），坐实"独立地址空间"
        │
        ▼
第 1 步：编译期图分析链跑通（A 组，纯 CPU，零硬件）
  问题1 图拆分打标 → 问题2 placement/redistribute 标注 → 问题8 三区 offset 规划
  ✅ 验证 P1：GPT-2 torch.export 不 graph break；每个 aten 节点带 device/part_id；
            核对分组符合"相邻+都落DPU→同组，遇host断开"
  ✅ 验证 P2：中间张量 placement 与附录 A 手推一致；redistribute 边落在预期位置
             （如两层 Linear 之间的 Partial→Replicate）；KV 节点带 pinned、不被插 redistribute
  ✅ 验证 P8：三区 offset 无重叠、总量 ≤ 8GB 容量校验通过；mram_offset 正确回填 PIMTensorSpec
        │
        ├──▶ [D] GeneSim 接入·第 1 步 ∥（D 组，TTIR，不依赖问题 5，纯 CPU，可与第 1 步并行起步）
        │      问题4 用 FlagTree 原生 TTIR 对 A 类算子抽 flops/data_bytes，回填 GeneSim Operator 成本字段
        │      ✅ 验证：精化成本后的 .ir 能被 GeneSim loader 读入、scheduler 正常出结果；
        │               成本较 build_from_hf_config 硬编码更贴近真实（访存侧此步为近似，需显式标注）
        ▼
第 2 步：redistribute → DMA 下降 pass（B 组，仍用 NumpyBackend 模拟多地址空间）
  问题3 通信库：把每条 redistribute 边展开为经 host 中转的 DMA 序列
  问题7 编译期算 KV 区大小/offset
  ✅ 验证：逐类型数值对拍——Partial→Replicate（收部分和累加回写）、Shard→Replicate
           （收片拼接广播）等，每种 redistribute 的输出与单卡 PyTorch 对应结果逐元素相等；
           bring-up 只需先验 all_reduce / all_gather 两个原语
        │
        ▼
第 3 步：编排器总装 + 跑通 prefill 一个 step（B 组收口，NumpyBackend，零硬件）
  问题6 exec_plan.py 汇总 P1/2/3/8 蓝图生成 ExecutionPlan；execute_plan 解释该 DAG；host 胶水节点直接调 PyTorch 兜底
  ✅ 验证：逐节点对拍器（assert_node_matches_ref）——按 placement 把分片合并回完整张量，
           与单卡 PyTorch 逐元素比，第一处不匹配即定位到具体 node；prefill 后 logits 对齐
        │
        ▼
第 4 步：打通 KV cache，跑通完整 decode 循环（B 组，NumpyBackend，零硬件）
  问题7 KV runtime：运行时 update/read_tile/mask + 编排器按 valid_len 推进（含首 token 采样）
  ✅ 验证：多步 decode 与单卡 PyTorch（HF StaticCache）逐 step logits 对齐；
           KV 数据 update/read 全程本地、零跨 DPU（softmax 的 scores 经 host 往返，不搬 KV）；
           固定 max_seq 分块读 + mask 结果正确
        │
        ▼
第 5 步 ∥：算子编译器上硬件（C 组，前面可全程并行，此步起需 SDK）
  问题5 ttir→upmem 桥让单 DPU 子图真能编译执行；hal_numpy → hal_vendor 换后端
  ✅ 验证：单 DPU kernel 二进制在硬件上执行结果与 NumpyBackend 一致（换后端不换数值）；
           A 类算子（GEMM/GEMV/逐元素）逐个对拍通过后再宽化白名单
        │
        └──▶ [D] GeneSim 接入·第 2 步（D 组，pim mlir，依赖问题 5 产出）
               问题4 把成本来源从 TTIR 换成 pim mlir，从 MRAM↔WRAM 显式搬运统计真实读写字节，补齐访存侧精度
               ✅ 验证：pim mlir 抽出的 data_bytes 较第 1 步 TTIR 近似值更准；.ir 仍能被 GeneSim 直接消费
```

> 第 0 到 5 步全程固定 shape；序列变长是走通之后的独立阶段（第 2 阶段）。
> **验证总原则**：本方案有两条彼此独立的验证轴。**功能正确性轴**——NumpyBackend 是运行时层的统一验证底座（覆盖问题 3/6/7/8 的执行与校验），编译期图分析（问题 1/2/7/8 的元数据产出）则靠与附录 A 手推、容量校验、torch.export 成功导出来校验；两条链都以"和单卡 PyTorch 逐元素对齐"为最终正确性判据，换真硬件时要求"换后端不换数值"。**性能评估轴**——GeneSim 接入（问题 4）的判据是精化后的 `.ir` 能被 GeneSim loader 读入、scheduler 正常出结果，且成本较硬编码更贴近真实；它只评估性能、不产出数值结果，与功能正确性轴互不替代。

### 6.4 里程碑节点


| 周期                     | 核心工作内容                                               | 细分落地任务                                                                                                                                                                                                                                                                                                                                                                                                                                   | 阶段交付产物                                                                                                                                                                              | 验收标准                                                                                                                                                                                                                      | 对应文档模块                   |
| ------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 第 1-2 周（初期打通）    | GeneSim 仿真链路打通基础环境搭建<br>+ 编译期图分析底座开发 | 1. 搭建全套研发环境，在 GPU 环境下尝试 HuggingFace→PyTorch→FlagGems→FlagTree 链路；<br>2. 完成 GeneSim 接入第一步：开发 PyTorch 模型的成本提取器，统计算子 flops/data_bytes，替换 GeneSim 硬编码成本;<br>3. 实现问题 1 图拆分核心逻辑：DPU 连通分组、节点 device/part_id 标注；<br>4. 完成假后端开发，模拟多 DPU 独立地址空间与基础 DMA 读写；<br />5. 基础单元测试搭建，验证环境与底座可用性。                                             | 1. 可运行的 GeneSim 仿真链路；<br />2. 图拆分核心代码；<br />3. 基础仿真底座；<br />4. 环境适配文档 + 基础单元测试用例。                                                                  | 1. 在CPU或GPU上完整走通 HF→FlagTree 端到端链路，仿真可正常输出结果；<br />2. 模型导出无 graph break，算子分组符合预期规则；<br />3. 多 DPU 独立地址空间模拟生效，无隐式数据共享；<br />4. GeneSim 可正常输出大致的仿真数据。 | 问题 1、问题 4、运行时底座     |
| 第 3 周（迭代完善）      | 编译期图层核心能力补齐 + 通信库基础能力开发 + 内存规划     | 1. 完成问题 2 切分传播核心逻辑：权重初始切分配置、DTensor 布局适配、PIMTensorSpec 四元组标注、redistribute 边识别；<br />2. 实现 KV 张量 pinned 常驻标注，规避 KV 跨 step 重分发；<br />3. 开发通信库基础原语，实现 redistribute 边识别与编译期通信计划表生成；<br />4. 完善图分析链路，打通问题 1→问题 2 的数据流转。  <br /> 5. 进行问题 8 内存规划开发：权重 / KV / 激活三区静态布局、贪心复用、容量校验、pending_readers 依赖生成；<br /> | 1. 切分传播完整代码、算子布局规则表；<br />2. 带完整 placement/redistribute 标注的计算图；<br />3. KV 局部性约束逻辑代码。<br /> 4. 内存规划完整代码、各 DPU 内存规划表 DPU_k.plan <br /> | 1. 中间张量布局推导与附录 A 手推结果逻辑相符；<br />2. KV 张量无冗余 redistribute 边，常驻属性生效；<br />3. 可精准识别 all-reduce/all-gather 等重分布场景；<br />4. 全图标注完整、无缺失、无逻辑冲突。                       | 问题 2、问题 3、问题 7、问题 8 |
| 第 4 周（pim适配模拟器） | 算子编译器下沉 + GeneSim （核心里程碑）                    | 1. 完成问题 5 算子编译器核心开发，自研 pim mlir 桥接、近存访存下降、DPU 同步原语补齐；<br />2. 完成 GeneSim 接入第二步：实现自研的 pim mlir接入；<br />3. 完成 A 类主力算子（GEMM/GEMV/ 逐元素）I/O 适配与算子接入；<br />4. 对齐编译期算子契约，实现图层与算子编译器双向接口打通。                                                                                                                                                            | 1. 顶层 Pytorch module 模型到 pim mlir 编译桥完整代码；<br />2. 各级 mlir 成本提取模块，GeneSim 适配相关代码；<br />3. 适配存算一体的 A 类算子 kernel 编译能力。                          | 1. 单 DPU 算子可正常编译；<br> 2. GeneSim 可正常解析 pim mlir 产物。                                                                                                                                                          | 问题 4、问题 5                 |
| 第 5 周（全链路串联）    | KV 缓存运行时 + 编排器核心逻辑开发                         | 1. 开发问题 6 编排器核心：ExecutionPlan 生成、命令 DAG 调度、DMA 时序控制、host 胶水算子适配；<br />2. 实现问题 7 KV 缓存运行时逻辑：KV 分区布局、逐 step 追加、掩码适配、本地驻留管理；<br />3. 打通编译期蓝图→运行时执行的全数据链路。<br />4. 完成问题 8 内存规划开发：权重 / KV / 激活三区静态布局、贪心复用、容量校验、pending_readers 依赖生成；<br />                                                                                  | 1. 内存规划完整代码、各 DPU 内存规划表 DPU_k.plan；<br />2. KV 缓存全量运行时逻辑；<br />3. 编排器调度核心、执行计划生成模块；<br />4. 全链路静态蓝图产物（标注图、通信表、内存表）。     | 1. 三区内存无重叠、无溢出，8GB 容量校验通过；<br />2. KV 缓存跨 step 稳定驻留，解码掩码生效；<br />3. 编排器可自动调度多 DPU 执行；                                                                                           | 问题 6、问题 7、问题 8         |
| 第 6 周（阶段收口）      | 全链路闭环调试 + 数值对拍 + 阶段验收收尾                   | 1. 打通 prefill + 完整 decode 自回归解码全流程；<br />2. 搭建逐节点对拍器，完成与原生 PyTorch 单卡数值对齐；<br />3. 修复全链路时序、依赖、数据搬运 bug，统一链路口径；<br />4. 整理所有代码产物、文档、测试报告，完成第 1 阶段全量收口；<br />5. 固化 NumpyBackend 功能验证体系。                                                                                                                                                             | 1. 可完整运行的存算一体大模型推理全链路；<br />2. 第 1 阶段全套代码产物、技术文档、接口契约文档；<br />3. 稳定的 GeneSim 仿真评估链路。                                                   | 1. GPT-2 模型可正常自回归解码，模拟代价估计正常；<br />2. 编译、调度、通信、内存、KV 缓存全模块稳定运行；<br />3. 满足第 1 阶段所有研发目标。                                                                                 | 全模块汇总验收                 |

### 6.5 分工

第 1 阶段共 7 人，分 5 组。不设专职测试岗，集成主战场在编排器，故对拍器与集成收口由编排器组的第 2 人专职承担；单元测试分散到各组自测，KV/内存组在 W1–3 编译期任务较轻时兼补公共单测。关键路径为「图级编译器 → 运行时（编排器/通信/KV）→ 集成对拍」。


| 工作内容                                                                                                                                                    | 对应任务     | 负责人 | 人数 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------ | ---- |
| **图级编译器**（编译期，纯 CPU）<br />· 1.图拆分打标 + 切分传播<br />· 2.通信库（编译期通信计划表生成 + 运行时 DMA 下降与对拍）                           | 问题 1、2、3 | 待指派 | 2    |
| **主机编排器 + 集成**（运行时，NumpyBackend）<br />· 1.编排器执行体（ExecutionPlan/解码循环）<br />· 2.`hal_numpy` 假后端 + 逐节点对拍器 + 全链路集成收口 | 问题 6       | 待指派 | 2    |
| **KV cache + 内存管理**（编译期布局 + 运行时）                                                                                                              | 问题 7、8    | 待指派 | 1    |
| **算子编译器**                                                                                                                                              | 问题 5       | 待指派 | 1    |
| **编译器 + 模拟器适配**                                                                                                                                     | 问题 4 | 待指派 | 1    |

### 6.6 代码仓库排布

实施需要的仓库：


| 仓库                    | 内容                                                   | 说明                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **flagos-pim-compiler** | 问题 1/2/3/6/7/8 + NumpyBackend + 对拍器 + 问题 4 桥接 | 这 6 个任务围绕同一组共享数据结构（`PIMTensorSpec`、标注静态图、通信计划表、`DPU_k.plan`、`ExecutionPlan`）；放一仓避免契约跨仓漂移，正对风险点 #2「收口点单一」 |
| **FlagTree**            | 问题 5 算子编译器（`ttir→pim mlir` 桥）               | 算子编译仓库                                                                                                                                                     |
| **flagOS-installers**   | 环境 / 构建工具                                        | flagOS编译和安装                                                                                                                                                 |

**flagos-pim-compiler主仓目录结构**：

```
flagos-pim-compiler/
├── contracts/          # ★共享数据契约，全组 import 的唯一真源
│   ├── pim_tensor_spec.py      # PIMTensorSpec 四元组
│   ├── exec_plan.py            # ExecutionPlan / 命令 DAG schema
│   └── graph_meta.py           # node.meta 字段约定（device/part_id/placement/redistribute）
├── graph/              # 图级）: 问题1,2  图拆分打标 + 切分传播
├── comm/               # 图级: 问题3    编译期通信计划表 + 运行时 DMA 下降
├── memory/             # KV/内存:   问题7,8  build_kv_layout + 三区 offset 规划
├── runtime/            # 编排器）: 问题6  ExecutionPlan 生成 + 解码循环
├── backend/            # 编排器: hal_numpy 假后端 / hal_vendor 真硬件
├── genesim_bridge/     # 模拟器:   问题4    成本桥接（sidecar，产 .ir）
├── tests/              # : 逐节点对拍器 + 各组单测
├── examples/           # 端到端示例（GPT-2 全链路跑通）
├── scripts/            # 开发与构建脚本
└── docs/               # 设计文档与接口契约
```

`contracts/` 是全仓地基，甲乙丙丁与 KV 组都依赖它——W1 第一件事是把这几个 schema 的字段名钉死，之后各组在自己目录内并行推进，靠 `node.meta` 解耦，对应 6.2「两条链只在数据上耦合、可大幅并行」。

flagos-pim-compiler：https://github.com/jingge815/flagos-pim-compiler

FlagTree: https://github.com/jingge815/FlagTree

FlagGems: https://github.com/jingge815/FlagGems

Flir: https://github.com/jingge815/flir

flagOS-installers: https://github.com/jingge815/flagOS-installers

## 七、进一步考虑问题

1. 算子跨 DPU访问。
2. 动态形状模型处理。
3. 当前为粗粒度方案，各模块细粒度设计待细化。
4. 多 session / 并发请求需要进一步考虑。
5. 后续需要进一步考虑代价模型。
6. prefill 与 decode 最优切分问题。
7. 第1阶段以走通为目标，其热路径主要在 host，且 GeneSim评测当前只是估算。
8. 通信成本进 GeneSim 的通路。本架构性能瓶颈在 DPU 无直连、每条 `redistribute` 过 host 两跳的通信，但问题 4 当前只回填计算算子成本，通信成本未进入 GeneSim（问题 4 三.(3) 还特意把 `redistribute` 字节排除在算子 `data_bytes` 之外）。通信库在 NumpyBackend 上可验功能正确，但模拟器上量不出通信瓶颈。可行方向：问题 3 通信计划表已逐段带 `nbytes` / `wait_for`，按 `edge_id` 聚合成 host 中转 DMA 的字节与跳数、导为 GeneSim 通信节点。落地前需先核对 GeneSim scheduler 源码，明确两点：与其原生跨 VPU 传输估算是否重复计数、通信节点按 DPU↔host 链路（而非 host DRAM）带宽计时的配置落点。

## 八、风险点

1. 算子覆盖面广，需要实现对全部算子的支持，上百个，涉及算子修改、FlagTree 支持以及上层图编译优化的数据布局转换。缓解：先走通整体流程，第 1 阶段用窄白名单只放 A 类（GEMM/GEMV/逐元素），B/C 类留 host 当胶水，几乎无需手改 kernel；再按 host 是否成为瓶颈逐个宽化（`[阶段2]`）。**第 1 阶段选定 GPT-2 （HuggingFace `openai-community/gpt2`，标准 MHA decoder-only、绝对位置编码，权重切分落在问题 2 第 1 阶段切分契约范围内）；选它而非 GQA 模型是为把第 1 阶段难度压到最低——MHA 免去 `q_heads_by_kv` 映射、绝对位置编码免去 RoPE 留 host、且 HF 开放下载无 gating；16 头可被 2/4/8 整除，能验 4/8 DPU 张量并行。避开 MoE（专家路由是真动态控制流，导出会 graph break）。功能验证只要求编译器输出与单卡 PyTorch 逐元素对齐，与权重是否预训练无关，故日常迭代可用缩小层数的随机权重配置加速，验收时再加载完整 24 层权重跑一遍，二者架构一致、代码不变。**
2. **全链路修改、收口点单一**，本方案涉及编译期图分析、算子编译、运行时编排全链路的协同修改，任一环节数据流标错都会导致最终结果错。缓解：以逐节点对拍器（问题 6）为核心调试基础设施，按 placement 合并分片后与单卡 PyTorch 逐元素比，第一处不匹配即定位到具体 node；全链路先在 NumpyBackend 零硬件验对，再上真硬件。
3. **GeneSim 接入的成本换算精度**，问题 4 从 IR 抽成本回填 GeneSim，难点有三：① IR 到 GeneSim 字段的成本换算规则，尤其访存字节的准确统计；② 一个 GeneSim `Operator` 与 FlagTree kernel 的一对多 / 多对一关联；③ 第 1 步 TTIR 目标无关、访存侧拿不准，须显式标注为近似、不与第 2 步 pim mlir 混淆。缓解：先用 TTIR 走通链路（访存侧近似），再随问题 5 的 pim mlir 补齐访存精度；A 类算子先行、与问题 5 白名单口径一致，避免"某算子没编 kernel 却要抽成本"的矛盾。此项为性能评估旁支，不影响功能正确性。

## 九、关键结论

1. **一切源于地址空间差异**：GPU 统一、存算一体独立且无直连，这一条驱动了图拆分、通信、KV、编排的全部改造。
2. 利用DTensor 的切分传播分析当编译期的求解器。
3. 自写通信库，不要硬套 FlagCX 的 P2P 模型。
4. KV 局部性是设备映射的第一约束，先定KV 按 head 切、本地驻留永不搬，再让其他层迁就。
5. 编译期决策、运行时分离，所有切分、映射、通信、内存决策在编译期一次定死为静态蓝图，编排器只照表执行、不做任何决策。
6. 从零硬件的 CPU 原型起步、以对拍器守正确性，最难的数据流逻辑先在 NumpyBackend（N 块独立 numpy buffer 模拟 N 个 DPU）上验证对，靠逐节点对拍器与单卡 PyTorch 逐元素对齐，最后上真实硬件，全程换后端不换数值。
7. **功能与性能两条验证轴分离**：功能正确性由 NumpyBackend 保证，性能评估经 GeneSim（另一团队开发，本方案负责接入适配）。GeneSim 接入用编译器真实 IR 产物替换其原有硬编码成本，是一条不参与执行、不产出 token 的评估旁支，与执行栈并行推进、互不阻塞。

---

## 附录 A：Shard / Replicate / Partial 布局转换的数值推演

本附录为问题 2 第三部分中两层 Linear 的布局转换过程提供逐元素的数值推演，用以说明 `Shard`、`Replicate`、`Partial` 三种布局的本质区别，以及 `redistribute` 为何在特定边上产生。理解的关键在于看清矩阵乘法中"哪一维被切分"，而非纠结张量的具体形状。

**最小示例的形状约定**（隐藏维 `hidden = 4`、中间维 `ffn = 6`、2 台 DPU）：

```
X:  [1, 4]                  → Replicate（每台都持有完整的 X）
W1: [4, 6]   Linear1 权重
Y1 = X @ W1  → [1, 6]
W2: [6, 4]   Linear2 权重
Y2 = Y1 @ W2 → [1, 4]
```

`Y1` 形状为 `[1, 6]`、`Y2` 形状为 `[1, 4]`。需要注意：这两个张量并非不可分的整体，其内部的多个元素会被分配到不同 DPU 上，这正是布局分析的核心。

### A.1 Linear1：权重按列切 Shard(1)，Y1 得 Shard(1)，无需通信

将 `W1[4,6]` 按列（第 1 维）切分给 2 台 DPU，`X` 在每台上均为完整副本：

```
DPU0:  X[1,4] @ W1[:,0:3][4,3]  = Y1[:,0:3]   → Y1 的前 3 个元素
DPU1:  X[1,4] @ W1[:,3:6][4,3]  = Y1[:,3:6]   → Y1 的后 3 个元素
```

`Y1` 共 6 个元素，两台各持有不同的列，需**拼接**才完整，即 `Shard(1)`。计算前 3 列只需完整的 `X` 与 `W1` 的前 3 列，不涉及另一台的数据，因此该步**零通信**。

### A.2 Linear2：权重按行切 Shard(0)，Y2 得 Partial

`Y2 = Y1[1,6] @ W2[6,4]`，其中间维 6 是"相乘后累加消去"的 **contraction 维**：

```
Y2[0,j] = Σ(k=0..5) Y1[0,k] * W2[k,j]     ← 对 k = 0..5 全部求和
```

每个输出元素都需用到 `Y1` 的全部 6 个元素。而 `Y1` 已被切成两半，因此将 `W2[6,4]` 也按行（第 0 维）切开，令每台只计算属于自己的那部分贡献：

```
DPU0:  Y1[0:3][1,3] @ W2[0:3,:][3,4] = Σ(k=0..2) Y1[0,k]*W2[k,j]  → [1,4]
DPU1:  Y1[3:6][1,3] @ W2[3:6,:][3,4] = Σ(k=3..5) Y1[0,k]*W2[k,j]  → [1,4]
```

核心在此：真正的 `Y2[0,j]` 是 6 项之和，而 DPU0 只计算了 `k=0,1,2` 这 3 项之和，DPU1 只计算了 `k=3,4,5` 这 3 项之和。两台的结果都是完整的 `[1,4]` 形状，但各自只是"部分和"：

```
真正的 Y2 = DPU0 的结果 + DPU1 的结果
```

这即是 `Partial`——**形状完整、数值为部分和，尚差一次逐元素相加**。

### A.3 Shard 与 Partial 的区别

`Shard` 与 `Partial` 是布局分析中最易混淆的两种布局，二者的区别如下：


|                | Shard(1)（Y1）                        | Partial（Y2）               |
| -------------- | ------------------------------------- | --------------------------- |
| 每台存储的形状 | **一个分片**（前 3 个 / 后 3 个元素） | **完整** `[1,4]`            |
| 还原方式       | **拼接**（concat，前半 ⊕ 后半）      | **相加**（sum，逐元素加） |
| 本质           | 每台持有一段，拼接才完整              | 每台算了部分和，相加才正确  |

**为什么 `Shard(1)` 输入 `Shard(0)` 权重必然产出 `Partial`**：因为 `Y1` 的列维（6）恰好是矩阵乘法要累加消去的 contraction 维，这一维被切开，等于"求和"被拆分到两台 DPU 上各做一半，各得部分和，合起来才是完整结果。由此得到规则：**矩阵乘法的 contraction 维被 Shard，则输出为 Partial。**

### A.4 下一层要求 Replicate，插入 redistribute

LayerNorm 等算子需对完整的 `Y2` 做归一化，要求每台都持有完整且正确的 `Y2`（`Replicate`），而上游产出的是 `Partial`（各持部分和），二者不一致。因此在该边插入一个 `Partial → Replicate` 的 `redistribute`——其本质是将两台的部分和相加后再发回每台。该操作在 GPU 上对应 all-reduce；本方案将其落地为"各 DPU → host 累加 → 写回各 DPU"，具体的 DMA 序列见问题 3。

需要再次强调：上述全过程没有真正执行任何算子、也没有搬运任何字节的数据。所谓"重分布"并非算子执行的结果，而是沿布局规则逐步推导 `placement`，直到 `Partial ≠ 要求的 Replicate` 这一不一致点，即在该边打上 `redistribute` 标注。整个推导只涉及两类计算：查算子的 `placement` 规则表、比较本算子要求的输入布局与上游实际产出布局之间的差异。

---

## 附录 B：redistribute 从编译期到运行时的完整下降实例

本附录为问题 3 提供一个完整的下降与执行实例，贯穿图层（编译期）、编排器（运行时）、通信库三层，说明一条 `redistribute` 边如何从图上的标注最终展开为一串主机中转的 DMA 指令。

**场景**：MLP 的权重按行切 `Shard(0)`，其后紧接 LayerNorm，对应 `Partial → Replicate` 类型。设权重 `W2 = [K, N]` 按行切为两半，DPU0 与 DPU1 各算出一个部分和（`Partial`）；LayerNorm 按第 1 阶段窄白名单留在 host，需对完整的 `Y` 做归一化（要求 `Replicate@host`），因此这条边的 `dst_loc = {"device": "host"}`。

### B.1 编译期（图层）：写入通信计划表

切分传播推导到这条边时，发现上游产出的 `Partial` 与下游要求的 `Replicate@host` 不一致，于是打上 `Partial → Replicate` 标注，`dst_loc` 取自下游 LayerNorm 节点的 `PIMTensorSpec.device = "host"`。按问题 3 二.（4）的规则，`all_reduce` 的收集段与之前一致，两个 DPU 各生成一段；因 `dst_loc` 为 host，回写段不生成，向通信计划表写入：

```
edge_id: 42
type:     all_reduce        (Partial → Replicate@host)
reduce:   sum
dst_loc:  {"device": "host"}
wait_for: [DPU0, DPU1]       ← 读前等待，整条边一份：copy_from 之前必须等这些 DPU 的生产 kernel 执行完
segments:
  segment 0: src_dpu=DPU0, src_local_range=[0, N), global_range=[0, N), dst_dpu=None, dst_local_offset=0, nbytes=N×4
  segment 1: src_dpu=DPU1, src_local_range=[0, N), global_range=[0, N), dst_dpu=None, dst_local_offset=0, nbytes=N×4
```

两段的 `dst_dpu` 均为 `None`，即不存在任何回写 DPU 的段——`dst_loc` 为 host 时，问题 3 二.（4）的生成规则不产出广播段，故本例没有 `dst_ready_after` 需要计算。若后续宽化白名单把 LayerNorm 移入 DPU，`dst_loc` 会变为 `{"device": "dpu", "dpus": [...]}`，生成规则随之为该集合各产出一段回写段，`dst_ready_after` 才会出现，取值仍由问题 8 三."原地写回的安全性"一节给出。

紧接着，问题 6 的 `exec_plan.py`（问题 6 三.（2）的 `emit_redistribute`）把这条边展开为 `ExecutionPlan` 里的具体命令：两条 `dma_in`（`waits` 分别指向 DPU0、DPU1 上生产 `Y` 的 `launch` 命令，对应本例的 `wait_for`）、一条 `host_reduce`（`waits` 为两条 `dma_in` 的 `id`）；因本例没有回写段，不生成任何 `dma_out` 命令。

### B.2 运行时（编排器）：解释 `ExecutionPlan`

编排器不再遍历 `graph.edges` 也不再读 `comm_plan`，只按 `ExecutionPlan` 的命令顺序执行，全部等待已经固化在每条命令的 `waits` 里：

```python
events = {}                                 # Command.id -> Event
for cmd in exec_plan.commands:              # 已是编译期排好的拓扑序
    for w in cmd.waits:
        hal.wait(events[w])                 # 逐条精确等待前驱事件，取代原来的 barrier(plan.wait_for)
    events[cmd.id] = hal.submit(cmd)        # 提交命令，dispatch 内部按 op 展开如下
    # dma_in:      copy_from_dpu 收集分片到 host 的归约缓冲区
    # host_reduce: acc = sum(收集到的各分片)      —— 本例的归约动作，对应 all_reduce 内部逻辑
    # dma_out:     copy_to_dpu 回写各 DPU          —— 本例不出现，dst_loc 为 host
    # host_op:     host_env[node] = acc            —— dst_loc 为 host 时，归约结果直接交给下游 host 节点
```

### B.3 通信库内部：展开为 DMA 序列

`dma_in` / `host_reduce` 两类命令内部真正发生的是：`copy_from` 两次收集各 DPU 的部分和到主机，主机执行 `acc = buf0 + buf1`；因本例 `dst_loc` 为 host，序列到此为止，不发生任何 `copy_to` 回写 DPU——`ExecutionPlan` 中没有对应的 `dma_out` 命令。此后 LayerNorm 直接在 host 上对 `acc` 做归一化，其输出经 `Replicate → Shard` 的 `scatter` 送回各 DPU，对应 `ExecutionPlan` 中后续的 `dma_out` 命令，供下一层 QKV 投影使用。全程 kernel 从未感知这次通信——正如问题 5 所述，kernel 的视野中只有本 DPU 的地址空间。

### B.4 补充示例：`Shard → Replicate` 的分片顺序与 DPU 编号无关

`all_reduce` 中每个 DPU 持有形状相同的完整部分和，不涉及顺序问题；`all_gather` 则不同，用一个例子说明为何拼接顺序不能按 `dpu_id` 排序、必须按 `global_range` 排序。

设某张量长度为 6，沿 0 维切成两片，分区器把后半段分给了 DPU0、前半段分给了 DPU1（`dpu_id` 与全局位置的先后顺序不一致，这种分配在张量并行中合法且常见）：

```
edge_id: 43
type:    all_gather        (Shard → Replicate)
segments:
  # 收集段：DPU → host，dst_dpu 记为 None
  segment 0: src_dpu=DPU0, src_local_range=[0, 3), global_range=[3, 6), dst_dpu=None, dst_local_offset=3, nbytes=12, dst_ready_after=[]
  segment 1: src_dpu=DPU1, src_local_range=[0, 3), global_range=[0, 3), dst_dpu=None, dst_local_offset=0, nbytes=12, dst_ready_after=[]
  # 广播段：host → DPU，src_dpu 记为 None，global_range 为整个张量
  segment 2: src_dpu=None, global_range=[0, 6), dst_dpu=DPU0, dst_local_offset=0, nbytes=24, dst_ready_after=[]
  segment 3: src_dpu=None, global_range=[0, 6), dst_dpu=DPU1, dst_local_offset=0, nbytes=24, dst_ready_after=[]
```

收集阶段只取 `dst_dpu is None` 的收集段（segment 0、1），按 `global_range.start` 排序后再拼接，即先取 segment 1（`global_range=[0,3)`）、再取 segment 0（`global_range=[3,6)`），得到正确的全局顺序 `concat(DPU1 的分片, DPU0 的分片)`。若按 `segments` 列表原有顺序或按 `src_dpu` 编号顺序拼接，会先取 DPU0、再取 DPU1，结果错位，且此错误不引发任何运行时异常，是最难排查的一类错误——这正是通信计划表必须携带 `global_range`、通信库必须按其排序而非按 DPU 编号处理的原因。合并完成后，广播阶段再取 `src_dpu is None` 的广播段（segment 2、3），把合并结果写回每个目标 DPU。

这一实例印证了问题 3 的核心结论：通信的类型、参与方、地址在编译期已完全确定，运行时只是照表展开为 DMA 序列；同步（`barrier`）是通信的前置条件，由数据依赖天然导出，而非一条独立的指令流。
