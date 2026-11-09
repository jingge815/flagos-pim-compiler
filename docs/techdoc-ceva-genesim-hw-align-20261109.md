# 技术文档：GeneSim 对齐 CEVA 多核硬件与性能模型

> 文档编号：techdoc-ceva-genesim-hw-align-20261109
> 创建日期：2026-11-09
> 关联文档：request-ceva-genesim-hw-align-20261106、design-ceva-genesim-hw-align-20261106、implement-ceva-genesim-hw-align-20261106、review-ceva-genesim-hw-align-20261107-round01 至 review-ceva-genesim-hw-align-20261109-round34
> 关联项目：genesim（主）、flagos-pim-compiler、FlagTree、CEVA 工作区（只读参考）
> 代码基线：genesim `fjg-dev@34481fe`（含未提交改动）；flagos-pim-compiler `main@f918586`（含未提交改动）；FlagTree `develop@0d7273bac`（工作区干净）
> 核心目标：为 GeneSim 增加 CEVA-NPM 后端，让仿真按 CEVA 硬件参数配置运行，并保持大模型到 GeneSim 的完整链路可用。

## 一、概述

### 1.1 背景与问题

　　本需求的核心目标是给 GeneSim 增加 CEVA-NPM 后端，让仿真按 CEVA 硬件参数配置运行。现有基础是 GeneSim 支持模型解析、图切分、PIM 轨迹编译和流水线仿真，图编译器与 FlagTree 也能产出放置 sidecar 和 PIM MLIR。但默认画像面向 GPU 加通用 PIM，缺少 CEVA 场景、整核映射、字节容量和 CEVA 专用计时路径，无法直接评估甲方硬件。本需求要求在不改变通用与 UPMEM 行为的前提下，把 CEVA 场景、后端、参数消费和结果分级接入同一条链路，并解决纯 PIM 资源落位、整核映射、字节预算、工作量缩放、产物溯源和校准边界等问题。

### 1.2 方案摘要

　　围绕上述核心目标，总体采用“CEVA 场景与硬件画像 → CEVA-NPM 后端分派 → 硬件参数进入容量、计时和通信 → 真实产物门禁 → 结果分级”的技术路线。GeneSim 新增五份 CEVA 场景和 `ceva` 参数段，`CEVANPMPIMCore` 承接内置轨迹求值，容量、相位计时和通信按同一套硬件参数计算；图编译器按同一字节预算导出放置 sidecar 与算子产物，FlagTree 提供 PIM 分块与量化 IR。整核模型是首期可检验的近似，不声称已捕获核内跨作业重叠。最终 1/64/128 核场景可创建并通过两级校验，真实产物可加载计时，结果按模型预测或范围内已验证标注。

### 1.3 完成情况

| 需求编号 | 需求内容（≤20字） | 完成情况 | 实现位置 | 验证用例 |
|----------|-------------------|----------|----------|----------|
| R1 | 独立纯 PIM 场景 | 已完成 | `conf/ceva_*.yaml`、`config_loader.py:_validate_ceva`、`main.py:main` | `TestCevaProfile`、`test_host_boundary_ops_land_on_cpu_in_pure_pim` |
| R2 | 1/64/128 核与 4e/1e 分层 | 已完成 | `main.py:_ceva_core_inventory`、`export_fixed_pu_mapping.py:build_mapping` | `TestCevaResourceInventory`、`TestOneToOneMapping` |
| R3 | 容量按字节统一 | 已完成 | `capacity.py:_ceva_capacity_limits`、`export_pp_placement.py:build_hw_budget` | `TestCevaCapacityBudget`、`test_sidecar_exports_the_same_byte_budget` |
| R4 | 整核计时、数据类型与工作量 | 部分完成 | `pimir_trace.py:compile_phase_chain`、`pim_core.py:CEVANPMPIMCore` | `TestPhaseWorkload`、`TestStaticPhaseTraceRuntime` |
| R5 | 编译链实际接入 | 部分完成 | `gene_sim_scheduler.py:_validate_ceva_compiler_artifact`、`run_manifest.py:build_ceva_run_manifest` | `TestCevaArtifactGate`、`TestCevaRunManifest` |
| R6 | 通信、共享存储与多核扩展 | 部分完成 | `comm_lib.py:_host_relay`、`config_loader.py:collective_edge_errors` | `TestCollectiveEdges`、`TestCevaLinkContract` |
| R7 | 同边界校准与结果分级 | 部分完成 | `gene_sim_scheduler.py:_ceva_result_maturity` | `TestCevaPlacementMapping` 成熟度用例 |

　　需求共 7 项，已完成 3 项，部分完成 4 项，未完成 0 项；本次运行基因仿真测试 1297 条通过、1 条跳过；当前遗留问题 29 条，详见第八节。

## 二、总体流程

### 2.1 总体流程图

　　本需求的核心目标是给 GeneSim 增加 CEVA-NPM 后端，让仿真按 CEVA 硬件参数配置运行。现有 GeneSim 只能按通用 PIM 画像评估，缺少 CEVA 场景、整核映射和硬件参数消费路径。因此总体流程以“新增 CEVA 场景与画像 → CEVA-NPM 后端分派 → 硬件参数进入容量、计时和通信”为主线。

　　首先，输入准备阶段读取模型 IR、请求轨迹、放置 sidecar 和 CEVA 硬件画像，并确定每核字节预算。然后，场景校验阶段按结构模式或正式模式检查纯 PIM 开关、整核层级、成本表与来源台账。其次，资源创建与后端分派阶段按 1/64/128 核建立 ClusterPU，把 `CEVA-NPM` 分派到内置轨迹核心，并用显式一对一映射把逻辑 DPU 落到物理核。接下来，轨迹编译与产物门禁阶段核对 PIM MLIR 的哈希、目标、语义和形状，再把相位工作量编成可缩放的循环。最后，流水线仿真按 CEVA 硬件参数计时，跨核通信单独计费，结果按证据强度分级并写出运行清单。如图所示，新增场景、后端和参数消费三个节点是本次改造的核心，通用与 UPMEM 路径保持原行为。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff,color:#fff

  A["现状与输入<br/>通用 PIM 仿真 编译产物"]:::stage --> B["新增 CEVA 场景<br/>ceva 参数段 1/64/128 核"]:::stage
  B --> C["后端分派<br/>CEVA-NPM 到内置核心"]:::stage
  C --> D["硬件参数消费<br/>字节容量 相位计时 通信"]:::stage
  D --> E["产物门禁<br/>sidecar PIM MLIR 哈希形状"]:::stage
  E --> F["结果交付<br/>成熟度 运行清单 资源清点"]:::stage
  W["甲方硬件参数<br/>假设 实测 拟合"]:::input -.-> B
  X["通用与 UPMEM 路径<br/>保持原行为"]:::input -.-> C
```

### 2.2 模块关系

　　本次改动涉及 GeneSim、图编译器与 FlagTree 三个模块：GeneSim 是主体，图编译器提供同一份字节预算和放置产物，FlagTree 提供 PIM 分块与量化 IR 且未改源码；CEVA 工作区只读参考。

| 模块/文件 | 职责（≤20字） | 改动类型 | 依赖的模块 |
|-----------|---------------|----------|------------|
| genesim（53 个文件） | CEVA 场景、后端、参数消费与结果分级 | 修改并新增 | 放置 sidecar、PIM MLIR、请求轨迹 |
| flagos-pim-compiler（6 个文件） | 预算导出、放置 sidecar、全流程接线 | 修改并新增 | FlagTree 编译产物、模型 IR |
| FlagTree | PIM 分块与量化 IR | 未改动 | 无 |
| CEVA 工作区 | 厂商目标与测量参考 | 只读 | 无 |

## 三、核心原理

### 3.1 关键机制

　　为了达成核心目标，需要保证硬件参数可配置、后端可分派、参数真正进入容量与计时、产物可追溯。现有 GeneSim 只有通用 PIM 画像和固定成本，参数没有来源台账。总体采用硬件画像与校验、CEVA-NPM 后端与参数消费、产物门禁与成熟度三种机制。

　　**CEVA 硬件画像与两级校验**旨在把硬件参数变成可校验、可追溯的配置。采用独立场景文件加来源台账的路线：五份场景覆盖 1/64/128 核与 4e/1e 画像，`ceva` 段记录 `execution_scope`、`engines_per_core`、`target_name`、`local_memory_bytes`、`sys_reserve_bytes` 和 `parameter_sources`；结构模式检查层级、类型、单位和字段，正式模式再要求时钟、周期表、来源台账和链路。这样未知参数不会流入通用默认值，1e 假设也不会写成实测。实现位于 `src/config_loader.py:1889`（`_validate_ceva`）。

　　**CEVA-NPM 后端与参数消费**旨在让硬件参数改变仿真结果。采用后端分派加参数消费的路线：`CEVA-NPM` 分派到 `CEVANPMPIMCore`，缺失轨迹直接报错，周期表逐项来自配置；容量以 `ceva.local_memory_bytes` 为真源；相位轨迹把工作量写成混合进制循环，运行时按实际形状重建张量并解析计时；区间倍率、中转开销和一对一映射也来自配置。这样避免换了后端仍按通用默认值计算。实现位于 `src/backend/pim_backend.py:115`、`src/pim/pim_core.py:429`、`src/pim/capacity.py:124`、`src/pim/pimir_trace.py:1513`、`src/vpu/vpu.py:504` 与 `src/scheduler/gene_sim_scheduler.py:3153`。

　　**真实产物门禁与成熟度分级**旨在证明仿真消费的是本次编译产物。采用单次读取、内容身份、结束复核的方法：配置、模型 IR、放置 sidecar、请求轨迹与 PIM MLIR 在首次加载时登记 SHA256，运行中改写文件会显式失败；加载时核对目标、原语、形状、分块和放置字节；结束前按快照复核并写出 `run_manifest.json`。成熟度由报告、参数来源和实际后端派生，证据不足只输出 `model_prediction`。实现位于 `src/scheduler/gene_sim_scheduler.py:2405`、`src/run_manifest.py:476` 与 `src/scheduler/gene_sim_scheduler.py:701`。

### 3.2 设计取舍

| 取舍点 | 采用方案 | 未采用方案 | 理由（≤30字） |
|--------|----------|------------|---------------|
| 核内资源层级 | 每核一个 TensorPU | 每 engine 一个 TensorPU | 无厂商 engine 调度证据，避免虚假并发 |
| 计时粒度 | 整核等效成本 | 指令级 CEVA 仿真 | 首期可校准、可复核，边界明确 |
| 旧配置兼容 | CEVA 严格拒绝，通用保留旧行为 | 全路径统一严格 | 不改变通用与 UPMEM 结果 |
| 场景维护 | 五份独立完整 YAML | 基础配置加成本配置继承 | 加载器没有继承语义 |
| 容量真源 | 整数 `ceva.local_memory_bytes` | 只写 GiB 浮点字段 | 避免 8 GB 与 8 GiB 混淆 |

### 3.3 适用前提与边界

- 前提：CEVA 场景必须声明 `system.pim_type=CEVA-NPM` 与 `ceva.execution_scope=npm_core`；否则校验失败。
- 前提：每核容量以 `ceva.local_memory_bytes` 为真源，8,000,000,000 B 是待确认假设；否则校验失败。
- 前提：正式评估必须有放置 sidecar、模型 IR、请求轨迹和每个计算节点的 PIM MLIR；缺项在生成轨迹前拒绝。
- 边界：4e 只记物理 engine 数，不建模核内重叠；1e 成本为假设值。
- 边界：跨核通信为共享链路抽象，未按甲方拓扑建模；秒级数字依赖假设时钟。

## 四、关键问题与解决方案

　　本需求的难点都围绕“让 CEVA-NPM 后端按硬件参数工作”这一目标，按阻碍目标达成的环节分为资源落位、整核映射、字节预算、工作量缩放、形状绑定、产物溯源、通信数值和结果分级八类。以下问题均来自需求、设计、实施或评审记录，方案以最终代码为准。

### 4.1 纯 PIM 的主机节点落位（来源：request §R1、implement §3.3 I6）

　　为了达成核心目标中的“纯 PIM 场景可运行”，需要为模型出入口、TP 归约和合成中转找到执行资源。其原因是旧能力筛选只认注意力算子，CPU 关闭后不参与候选，这些节点没有资源可落。实现把 CPU 加入候选资源，只承接 `MODEL_INPUT`、`MODEL_OUTPUT`、`ALL_REDUCE` 与中转节点，并补齐 CPU 与 HOST 的双向链路；主机中转节点使用零计算剖面，通信成本只在跨核边上计一次。实现位于 `gene_sim_scheduler.py:1071` 与 `cpu_backend.py:144`，由 `test_host_boundary_ops_land_on_cpu_in_pure_pim` 验证。

### 4.2 整核映射与 engine 分层（来源：design §3.3、request §R2）

　　为了达成核心目标中的“逻辑 DPU 正确落到 NPM 核”，需要确定核内资源如何映射。若把 4 个 engine 直接映射成 4 个 TensorPU，会引入编译产物并不支持的核内并发和容量均分；其原因是 GeneSim 的放置只看到整算子，没有厂商 GC 的 engine 切分与同步信息。实现采用整核等效：一个 ClusterPU 加唯一 TensorPU 表示一个 NPM 核，engine 数只写进 `ceva.engines_per_core` 画像；`build_mapping(one_to_one=True)` 要求每个逻辑 DPU 独占一个物理核，并拒绝取模换算与重复映射。实现位于 `export_fixed_pu_mapping.py:223` 与 `gene_sim_scheduler.py:3153`，由 `TestOneToOneMapping` 和 `TestCevaResourceInventory` 验证。

### 4.3 每核容量与跨仓预算一致（来源：design §4.1、request §R3、review round14/18）

　　为了达成核心目标中的“每核容量按 CEVA 字节预算生效”，需要让仿真与图编译器使用同一份预算。甲方标称每核 8 GB，但 GiB 字段与字节真源可能给出两套数字；其原因是旧配置只写 `cluster_memory_capacity_gb`，图编译器固定 4 GiB。实现以整数 `ceva.local_memory_bytes` 为真源，校验 GiB 字段的换算值逐字节相等；图编译器新增 `--mram-bytes` 与 `--sys-reserve-bytes`，放置 sidecar 升到 version 4 并写出物理量、预留和可用量，仿真侧加载时逐项核对。实现位于 `capacity.py:124` 与 `placement_export.py:537`，由 `TestCevaCapacityBudget` 和 `test_sidecar_exports_the_same_byte_budget` 验证。

### 4.4 相位工作量缩放与不截断（来源：design §5.2、request §R4、implement I3、review round16）

　　为了达成核心目标中的“计时随 M/N/K、元素数和归约长度变化”，需要修正工作量。旧相位链对任意矩阵都只发一条 MAC，超过 14 位立即数还会截断，导致大小形状代价相同；其原因是 `PhaseStep` 只保留汇总字节，编译端没有按真实工作量计量。实现把每相工作量写成混合进制固定循环：每个数位一层循环，超限时每高一位再套一层 16383 次循环，乘积精确等于工作量；MAC 按每个输出分别取 `ceil(K/mac_width)`，非整除尾量不合并；CEVA 路径使用 24 位长循环并拒绝越界，并由内置 `CEVANPMPIMCore` 求值：缺轨迹直接报错，周期表逐项来自配置，不依赖外部仿真库。实现位于 `pimir_trace.py:1513` 与 `:1566`，由 `TestPhaseWorkload` 和 `test_fused_activation_costs_one_lookup` 验证。

### 4.5 运行时形状绑定与解析求值（来源：design §5.2、implement I7/I11/I15、review round04/30）

　　为了达成核心目标中的“解码时延随实际查询长度和 KV 长度变化”，需要让相位轨迹绑定运行时形状。相位轨迹按符号维 128 编译后，运行时再按查询行数重复整条轨迹会放大 128 倍；整条比例缩放又会混淆不同操作数并缩小固定成本；其原因是编译期只保留汇总字节，没有逐张量形状。实现让相位模板登记每个张量的轴绑定和维度来源，运行时按实际预填充或解码形状重建张量，再用 `phase_cycle_breakdown()` 按与指令求值相同的分段和混合进制公式直接计算周期，避免逐形状物化指令。实现位于 `pimir_trace.py:1315`、`:1668` 与 `vpu.py:504`，由 `TestStaticPhaseTraceRuntime` 验证。

### 4.6 产物门禁与输入快照（来源：request §R5、review round01/18/23/27/31、implement I19）

　　为了达成核心目标中的“仿真消费真实编译产物”，需要证明产物来源和内容。只声明产物路径不足以证明仿真消费的就是本次编译结果；空报告、缺哈希、形状不符或运行中改写文件都能让错误计时流进结论。实现建立三层门禁：加载时按字节读取并登记不可变快照，核对 sidecar 哈希、`pim:v1` 目标、计算原语、逐轴形状、分块与放置字节；运行结束前复核快照，输入被改写时本次结果直接失败；CEVA 结果目录归档 `run_manifest.json`，记录配置、模型 IR、放置 sidecar、请求轨迹（以及使用时的成本 sidecar）和全部 PIM MLIR 的 SHA256 以及工具链身份。实现位于 `gene_sim_scheduler.py:2405` 与 `run_manifest.py:476`，由 `TestCevaArtifactGate`、`TestPimirHashVerification` 和 `TestCevaRunManifest` 验证。

### 4.7 通信数值契约与主机中转（来源：request §R6、review round12/13/32/33）

　　为了达成核心目标中的“通信按 CEVA 参数计费”，需要同时覆盖 HOST 中转并拒绝非法数值。通信成本既不能漏掉主机中转，也不能把任意精度整数或下溢系数当成免费传输；其原因是旧契约只检查正整数，`float(nbytes)` 会溢出，`nbytes/seq_len` 会下溢成 0，生产端也能写出消费端无法计费的产物。实现把“字节量、序列长度、每 token 系数”收敛为同一数值契约：`nbytes` 与非空 `seq_len` 必须可表示为有限正数，派生系数必须有限且严格大于 0；配置校验、sidecar 导出、第一次消费和补边共用同一入口。HOST 中转固定开销改为配置项，缺省仍为 100 ns。实现位于 `config_loader.py:212`、`placement_export.py:96` 与 `comm_lib.py:313`，由 `TestCollectiveEdges` 验证。

### 4.8 结果成熟度分级（来源：request §R7、design §7.3、review round01/10/20）

　　为了达成核心目标中的“结果可信分级”，需要区分模型预测和范围内已验证。没有独立验证时把结果称为已校准，会误导性能结论；其原因是旧实现只看验证报告文件是否存在。实现定义 `schema_version=11` 的报告契约，核对包与内核、目标、engine 数、配置与使用核数、请求边界、时钟、数据类型、形状范围、成本模型、通信模型和逐点误差；程序复算中位数与 p90，并核对 `ceva.parameter_sources` 的来源等级。任一证据缺失、不一致或来源为假设或未确认，结果保持 `model_prediction`。实现位于 `gene_sim_scheduler.py:701`，由成熟度正反用例验证。

## 五、改动说明

### 5.1 改动总览

　　本次共改动 59 个文件（新增 10 个、修改 49 个、删除 0 个），代码增加 21,499 行、删除 713 行。其中 genesim 改动 53 个文件（新增 9 个），增加 21,032 行、删除 677 行；图编译器改动 6 个文件（新增 1 个），增加 467 行、删除 36 行；FlagTree 工作区干净，未改动。核心改动集中在 genesim 的 CEVA 场景、后端分派、参数消费与运行清单，以及图编译器的预算导出与放置 sidecar。

### 5.2 文件改动清单

| 文件路径 | 改动类型 | 改动内容（≤30字） | 对应需求 |
|----------|----------|-------------------|----------|
| `genesim/conf/ceva_single_core_4e.yaml` | 新增 | 1 核 4e 校准场景 | R1、R2 |
| `genesim/conf/ceva_64core_4e.yaml` | 新增 | 64 核 4e 预测场景 | R1、R2 |
| `genesim/conf/ceva_64core_1e.yaml` | 新增 | 64 核 1e 产品场景 | R1、R2 |
| `genesim/conf/ceva_128core_4e.yaml` | 新增 | 128 核 4e 预测场景 | R1、R2 |
| `genesim/conf/ceva_128core_1e.yaml` | 新增 | 128 核 1e 产品场景 | R1、R2 |
| `genesim/run.sh` | 修改 | 默认场景切到 CEVA 64 核 | —（入口调整） |
| `genesim/scripts/export_fixed_pu_mapping.py` | 修改 | CEVA 一对一映射与资源契约 | R2 |
| `genesim/scripts/export_partition_plan.py` | 修改 | 切分参数类型与符号校验 | R2 |
| `genesim/scripts/pim_compiler.py` | 修改 | 透传长循环开关 | R4 |
| `genesim/src/autotune/dataset_observer.py` | 修改 | 溢出数值按缺失处理 | R4 |
| `genesim/src/autotune/placement.py` | 修改 | 权重数值溢出报错 | R4 |
| `genesim/src/backend/comm_lib.py` | 修改 | 链路字段契约与中转开销 | R6 |
| `genesim/src/backend/cpu_backend.py` | 修改 | 主机参数契约与中转零成本 | R1、R6 |
| `genesim/src/backend/pim_backend.py` | 修改 | CEVA 核心分派与循环总次数 | R4 |
| `genesim/src/config_loader.py` | 修改 | CEVA 段、两级校验、来源与通信契约 | R1～R7 |
| `genesim/src/ir/model_ir.py` | 修改 | 按字节解析模型 IR | R5 |
| `genesim/src/ir/trace_ir.py` | 修改 | 请求形状绑定与轨迹 v3 | R4、R7 |
| `genesim/src/main.py` | 修改 | 结构干跑、资源清点、清单落盘 | R1、R2、R5、R7 |
| `genesim/src/numeric_utils.py` | 新增 | 有限浮点统一转换 | R3、R4、R6 |
| `genesim/src/performance_objective.py` | 修改 | 数值溢出处理 | R4 |
| `genesim/src/pim/capacity.py` | 修改 | 字节真源容量与预留 | R3 |
| `genesim/src/pim/pim_compiler.py` | 修改 | 长循环与数值校验 | R4 |
| `genesim/src/pim/pim_core.py` | 修改 | 内置核心与 CEVA 严格核心 | R4 |
| `genesim/src/pim/pimir_trace.py` | 修改 | 相位工作量与运行时模板 | R4 |
| `genesim/src/pim/pim_isa.py` | 修改 | 区间倍率外部注入 | R4 |
| `genesim/src/pim/pim_trace_loader.py` | 修改 | 严格周期表与缓存身份 | R4 |
| `genesim/src/pim/pu_pool.py` | 修改 | CEVA 池资源配置校验 | R2 |
| `genesim/src/predictor/benchmark.py` | 修改 | 有限浮点转换 | R4 |
| `genesim/src/predictor/dataset.py` | 修改 | 有限浮点转换 | R4 |
| `genesim/src/run_manifest.py` | 新增 | 运行清单与工具链身份 | R5 |
| `genesim/src/scheduler/base_scheduler.py` | 修改 | 单次读取登记输入 | R5 |
| `genesim/src/scheduler/gene_sim_scheduler.py` | 修改 | CEVA 门禁、成熟度、清单 | R1～R7 |
| `genesim/src/vpu/factory.py` | 修改 | CEVA 资源契约与 CPU 模型 | R1、R2、R6 |
| `genesim/src/vpu/vpu.py` | 修改 | 运行时相位模板求值 | R4 |
| `genesim/tests/sim/test_collective_edges.py` | 修改 | 集合通信契约与中转用例 | R6 |
| `genesim/tests/sim/test_comm_lib.py` | 修改 | 链路数值契约用例 | R6 |
| `genesim/tests/sim/test_compiler_placement.py` | 修改 | 门禁、映射、成熟度用例 | R1～R7 |
| `genesim/tests/sim/test_config_loader.py` | 修改 | 场景、来源、数值契约用例 | R1～R7 |
| `genesim/tests/sim/test_cost_sidecar.py` | 修改 | 成本 sidecar 契约用例 | R5 |
| `genesim/tests/sim/test_cpu_backend.py` | 修改 | 中转零成本用例 | R1 |
| `genesim/tests/sim/test_export_fixed_pu_mapping.py` | 新增 | 一对一映射用例 | R2 |
| `genesim/tests/sim/test_main.py` | 修改 | 资源清点、摘要、清单用例 | R1、R2、R5、R7 |
| `genesim/tests/sim/test_pim_backend.py` | 修改 | CEVA 核心与纳秒断言 | R4 |
| `genesim/tests/sim/test_pim_capacity.py` | 修改 | 非有限容量反例 | R3 |
| `genesim/tests/sim/test_pim_compiler.py` | 修改 | 长循环编译用例 | R4 |
| `genesim/tests/sim/test_pimir_trace.py` | 修改 | 工作量、模板、形状用例 | R4 |
| `genesim/tests/sim/test_pim_trace_loader.py` | 修改 | 严格周期表与缓存身份 | R4 |
| `genesim/tests/sim/test_pu_pool.py` | 修改 | CEVA 池契约用例 | R2 |
| `genesim/tests/sim/test_run_manifest.py` | 新增 | 运行清单用例 | R5 |
| `genesim/tests/sim/test_scheduler.py` | 修改 | CEVA 调度契约用例 | R2、R4 |
| `genesim/tests/sim/test_trace_ir.py` | 修改 | 请求形状绑定用例 | R4、R7 |
| `genesim/tests/sim/test_vpu.py` | 修改 | 运行时相位求值用例 | R4 |
| `genesim/tests/sim/test_zone_cycles.py` | 修改 | 区间倍率用例 | R4 |
| `flagos-pim-compiler/genesim_bridge/placement_export.py` | 修改 | sidecar v4、预算、占位值、通信契约 | R3、R4、R5、R6 |
| `flagos-pim-compiler/scripts/export_pp_placement.py` | 修改 | 新增 MRAM 与预留参数 | R3 |
| `flagos-pim-compiler/scripts/run_full_pipeline.py` | 修改 | 全流程接入 CEVA 场景与预算 | R1、R3、R5 |
| `flagos-pim-compiler/tests/test_export_pp_placement_budget.py` | 新增 | 预算参数单测 | R3 |
| `flagos-pim-compiler/tests/test_ir_cost_eval.py` | 修改 | 融合激活按循环次数计工作量 | R4 |
| `flagos-pim-compiler/tests/test_placement_export.py` | 修改 | sidecar v4 与通信契约用例 | R3、R6 |

### 5.3 核心函数清单

| 函数/类 | 位置 | 改动类型 | 作用（≤30字） | 调用方 |
|---------|------|----------|---------------|--------|
| `GeneSimConfig._validate_ceva()` | `genesim/src/config_loader.py:1889` | 新增 | CEVA 结构与正式两级校验 | `validate()` |
| `collective_edge_errors()` | `genesim/src/config_loader.py:212` | 新增 | 集合通信数值契约 | 配置校验、调度器 |
| `compile_phase_chain()` | `genesim/src/pim/pimir_trace.py:1566` | 修改 | 按工作量发循环指令 | 轨迹编译 |
| `phase_cycle_breakdown()` | `genesim/src/pim/pimir_trace.py:1668` | 新增 | 按相位参数解析求值 | `PIMVPU` |
| `phase_runtime_template()` | `genesim/src/pim/pimir_trace.py:1315` | 新增 | 生成运行时形状模板 | 调度器 |
| `PIMVPU._static_phase_metrics()` | `genesim/src/vpu/vpu.py:504` | 新增 | 按实际形状求值与缓存 | `execute_profile()` |
| `_validate_ceva_compiler_artifact()` | `genesim/src/scheduler/gene_sim_scheduler.py:2405` | 新增 | 产物语义、目标与形状门禁 | `_register_compiler_artifacts()` |
| `_validate_ceva_dpu_mapping()` | `genesim/src/scheduler/gene_sim_scheduler.py:3153` | 新增 | 一对一核映射校验 | `_load_compiler_placement()` |
| `_ceva_result_maturity()` | `genesim/src/scheduler/gene_sim_scheduler.py:701` | 修改 | 派生结果成熟度 | `run()` |
| `build_ceva_run_manifest()` | `genesim/src/run_manifest.py:476` | 新增 | 生成运行清单与身份 | `main()`、调度器 |

### 5.4 数据结构与接口变更

　　为了让 CEVA 硬件参数、执行形状和产物身份能够跨模块传递，需要在配置、放置 sidecar、请求轨迹和相位轨迹中记录原实现没有的信息。已有 `cluster_memory_capacity_gb`、单一 `seq_len` 和汇总字节无法区分 GiB 与 GB、填充长度与执行长度、静态形状与运行形状，因此新增或扩展以下结构。

　　**CEVA 配置段**（`genesim/conf/ceva_64core_4e.yaml:178`、`src/config_loader.py:1889`）

```yaml
ceva:
  execution_scope: "npm_core"
  engines_per_core: 4
  target_name: "m32k_vpu500_dtcm64_8m_1024k_4e"
  local_memory_bytes: 8000000000
  sys_reserve_bytes: 0
  parameter_sources:
    ceva.local_memory_bytes: {origin: assumption, evidence: ..., unit: byte, scope: ...}
```

　　默认值仅对 CEVA 场景生效，通用配置不读取 `ceva` 段。

　　**放置 sidecar version 4**（`flagos-pim-compiler/genesim_bridge/placement_export.py:537`）

```json
{
  "version": 4,
  "hardware": {
    "mram_bytes_per_dpu": 8000000000,
    "sys_reserve_bytes": 0,
    "usable_mram_bytes": 8000000000
  },
  "dpu_to_cluster": [[0, 0], [0, 1]],
  "operators": {"6867": {"symbolic_dim_placeholder": {"Tq": 128}}},
  "collectives": [{"type": "all_gather", "nbytes": 4096, "seq_len": null, "dpu_ids": [0, 1]}]
}
```

　　旧消费方忽略新增字段；CEVA 严格模式要求 `hardware` 与画像逐项一致，并要求符号维占位值存在。

　　**请求形状绑定**（`genesim/src/ir/trace_ir.py:20`）

```python
@dataclass(frozen=True)
class RequestShapeBinding:
    valid_prompt_length: int
    query_length: int
    initial_kv_length: int
```

　　`Request` 新增 `padded_query_length` 与 `initial_kv_length`；旧轨迹仅在 `seq_len` 等于 token 数时可唯一推断，否则正式运行拒绝。

　　**相位运行时模板**（`genesim/src/pim/pimir_trace.py:1315`）

```python
{
  "dimensions": ["Tq", "Tp"],
  "dimension_sources": {"Tq": ["in", 0, 0], "Tp": ["in", 1, 0]},
  "steps": [{"op": "pim.matmul", "unit": "nmu", "operands": [...], "results": [...]}]
}
```

　　模板只记录轴绑定，不记录秒级结果；运行时按实际维度重建张量，缓存键覆盖轨迹代号、形状、周期表、时钟和严格模式。新增字段都是可选或带版本，旧 sidecar 在通用路径仍按原行为读取。

## 六、验证

### 6.1 验证环境

　　代码基线：genesim `fjg-dev@34481fe`、flagos-pim-compiler `main@f918586`，均含未提交改动；FlagTree `develop@0d7273bac`，工作区干净。运行环境：Linux，Python 3.10，torch 2.9.1，transformers 4.57.6；基因仿真测试使用仓库 `.venv` 与 PyTorch 环境各跑一遍。

### 6.2 验证命令与结果

　　为了验证“CEVA-NPM 后端按硬件参数工作”这一核心目标，本文采用分层验证方法。首先通过基因仿真全量与官方入口确认既有功能未受影响；然后围绕 CEVA 新增特性运行配置、容量、映射、工作量与产物门禁用例；其次用五份场景的实际资源工厂核对画像；最后用 64 核真实产物冒烟和全流程脚本确认端到端行为。这样既能定位局部问题，又能避免仅凭单元测试推断系统可运行。

| 验证项 | 命令 | 结果 | 结论 | 说明 |
|--------|------|------|------|------|
| 基因仿真全量 | `python -m pytest tests/sim -q` | 通过 1297 条，失败 0 条，跳过 1 条 | 通过 | 覆盖全部新增与既有仿真用例 |
| 基因仿真官方入口 | `./run.sh --test sim` | 46/46 个文件通过，合计 1292 条，跳过 1 条，失败 0 条 | 通过 | 与 pytest 统计基数不同 |
| 预测器与 UPMEM | `./run.sh --test predictor`、`./run.sh --test upmem_checker` | 预测器 7/7 个文件 87 条；UPMEM 49 条 | 通过 | 通用路径行为未变 |
| 图编译器定向测试 | `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q` | 通过 36 条，跳过 4 条，失败 0 条 | 通过 | 预算入口与融合工作量 |
| 放置 sidecar 测试 | `python -m pytest tests/test_placement_export.py -q` | 通过 57 条，失败 0 条 | 通过 | sidecar v4 与通信契约 |
| 图编译器全量回归 | `python -m pytest tests/ -q -k "not llama2_7b"` | 通过 1685 条，跳过 8 条，排除 49 条，失败 0 条 | 通过 | 排除 7B 长用例，耗时 17 分 50 秒 |
| 五份场景资源核对 | `PYTHONPATH=src uv run python /tmp/ceva_review_round31_inventory.py` | 5/5 场景结构与正式校验通过；创建核 1/64/64/128/128，GPU 0，engine 4/256/64/512/128 | 通过 | 评审临时探针；等价单测见 `TestCevaResourceInventory` |
| 真实产物冒烟 | `uv run python -u src/main.py --config /tmp/techdoc-ceva-smoke.yaml` | 7138 条轨迹全部重编、0 复用；创建/使用/空闲核 64/8/56；`total_time_s=672.363454`、`tokens/s=1.182396`、`result_maturity=model_prediction` | 通过 | 64 核、1 请求、8 台 DPU；配置由场景文件覆盖结果目录与放置 sidecar 生成 |
| 全流程脚本 | `python -u scripts/run_full_pipeline.py --num-stages 4` | 仿真成功；图编译 6852 算子、224 GEMM；算子编译 6626 节点；7138 条轨迹（7074 条 `pimir`、64 条 `absorbed`、0 条模板）；`total_time_s=7117.252156`、`tokens/s=1.135832`、`processed_tokens=8084`、`completed_requests=10` | 通过 | CEVA 分支按单请求并发串行执行，与实施文档早期通用配置的 1812.037 s 不同 |
| FlagTree 工具链 | `triton-opt --split-input-file -pim-tile-to-budget test/Dialect/TritonPIM/tile_to_budget_matmul.mlir` | `tile = 4x8x64`、`elemBits=4`、`groupDequantAccum=true` | 通过 | 工作区干净，未改源码 |

　　复现命令：

```bash
# 基因仿真全量与官方入口
cd /data/fjg/flagOS/genesim
source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q
PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim
./run.sh --test predictor
./run.sh --test upmem_checker

# 图编译器
cd /data/fjg/flagOS/flagos-pim-compiler
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q
python -m pytest tests/test_placement_export.py -q
python -m pytest tests/ -q -k "not llama2_7b"

# 五份场景与真实产物冒烟
cd /data/fjg/flagOS/genesim
PYTHONPATH=src uv run python /tmp/ceva_review_round31_inventory.py
uv run python -u src/main.py --config /tmp/techdoc-ceva-smoke.yaml

# 全流程
cd /data/fjg/flagOS/flagos-pim-compiler
python -u scripts/run_full_pipeline.py --num-stages 4

# FlagTree 工具链
cd /data/fjg/flagOS/flagOS-installers/FlagTree
/data/fjg/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/bin/triton-opt \
  --split-input-file -pim-tile-to-budget test/Dialect/TritonPIM/tile_to_budget_matmul.mlir
```

### 6.3 验收标准对照

| 验收标准（来自需求文档） | 验证方法 | 结论 | 证据 |
|--------------------------|----------|------|------|
| GPU 资源数与执行算子数为零 | 五份场景资源工厂清点 | 满足 | 6.2“五份场景资源核对”，GPU 均为 0 |
| 分别报告配置/创建/使用/engine/槽位 | 64 核冒烟与资源清点 | 满足 | 6.2“真实产物冒烟”，创建/使用/空闲 64/8/56 |
| 逐核容量到限额通过、超出 1 B 失败 | `TestCevaCapacityBudget` | 满足 | 边界用例 |
| 工作量随 M/N/K 与元素数增长、不截断 | `TestPhaseWorkload` | 满足 | 大形状与超限用例 |
| 正字节跨核传输有有限成本 | `TestCollectiveEdges`、64 核冒烟 | 满足 | 极值与下溢反例；跨核累计传输 0.776152 s |
| 产物哈希、目标、形状一致 | `TestCevaArtifactGate`、真实 sidecar | 满足 | 6,850 条记录、21 份 MLIR |
| 完整模型与请求边界可闭环 | 全流程脚本 | 满足（结构联调） | 10 请求、8084 token、7138 条轨迹 |
| 只在独立验证达标时称为已校准 | 成熟度正反用例 | 部分满足 | 当前运行均为 `model_prediction` |

### 6.4 评审情况

　　本需求共进行 34 轮评审，前 33 轮形成 95 条报告项（其中严重 20 条、高 48 条、中 26 条、低 1 条），第 34 轮未发现新增或复发问题。全部报告项均在当轮或后续轮次修复，没有判定不成立或长期不修复的条目。

| 评审轮次 | 核心问题 | 修复状态 | 最终证据 |
|----------|----------|----------|----------|
| 第 1 轮 | 空报告可升级成熟度、缺哈希、转置 MAC、1e 来源、负成本 | 已修复 | 新增报告契约与反例用例 |
| 第 2～4 轮 | 相位运行时重复缩放、严格模式可关、精度与请求边界未绑定 | 已修复 | 按实际形状重建轨迹，全量通过 |
| 第 5～7 轮 | 预测器可绕过、产物语义与形状门禁不足、p90 未复算 | 已修复 | 产物门禁与最近秩复算用例 |
| 第 8～10 轮 | 解码边界少一步、核数未绑定、搬运原语绕过、假设来源升级 | 已修复 | 请求边界与来源核对用例 |
| 第 11～13 轮 | 单行内核少计、通信语义比较失效、主机成本与 TP 归约漏计 | 已修复 | 单行契约、链路语义、主机指标用例 |
| 第 14～16 轮 | 大 M 产物被拒、预算未核对、分块元数据脱节、循环截断 | 已修复 | 分块重复、预算门禁、长循环用例 |
| 第 17～19 轮 | CPU↔HOST 链路缺失、清单缺身份、TP 校准仍用全局形状 | 已修复 | 链路、清单、本地分片用例 |
| 第 20～22 轮 | 单行产物冻结形状、成本覆盖虚报、归约尾量合并 | 已修复 | 行视图展开、覆盖台账、逐输出取整 |
| 第 23～25 轮 | 输入快照换绑、主机参数失守、链路数值未校验 | 已修复 | 快照复核、数值契约用例 |
| 第 26～28 轮 | 请求边界合并、资源类型失守、GPU 链路与配置快照 | 已修复 | 三字段绑定、域收敛、配置身份用例 |
| 第 29～31 轮 | 非相位计算漏计、缓存身份脱离内容、执行视图旁路 | 已修复 | 相位账本、内容缓存、首次消费视图 |
| 第 32～33 轮 | 集合通信大整数溢出与系数下溢 | 已修复并闭环 | 生产端与消费端统一契约 |
| 第 34 轮 | 无新增问题 | 无问题 | 全量回归与探针复跑通过 |

## 七、与设计的差异

　　最终实现与设计文档的主要差异如下，均如实记录，不写成缺陷。

| 设计条目 | 设计方案（≤20字） | 最终实现（≤20字） | 差异原因（≤30字） | 是否经确认 |
|----------|-------------------|-------------------|-------------------|------------|
| design §3.1 | 画像生成完整 YAML | 手工维护五份 YAML | 加载器没有继承语义 | 否（最小实现取舍） |
| design §4.2 | sidecar 物理布局描述 | 未实现 | 现有产物为 FP16，W4A8 映射未确认 | 否 |
| design §5.2 | 按执行节点选成本表 | 同一 opcode 一份表 | 缺分阶段测量数据 | 否 |
| design §5.2 | 静态循环不重复缩放 | 运行时按模板重建 | 避免整条比例混淆操作数 | 是（评审 round04） |
| design §5.2 | 超限不截断 | CEVA 长循环，通用保留 12 位兼容截断 | 不改变通用与 UPMEM 行为 | 否（记录为 K22） |
| design §7.1 | 产物形状逐轴核对 | 按函数签名核对，单行内核单独登记 | 真实产物有查表与参数张量 | 否（记录为 K18） |
| design §7.3 | 程序派生成熟度 | JSON 报告契约 `schema_version=11` | 契约字段由评审逐步补齐 | 否（待产出方确认） |
| design §3.1 | 通用配置继续保留 | 默认入口改为 CEVA 64 核 | 交付入口调整，帮助文本未同步 | 否（实施文档未登记） |

## 八、当前遗留问题

　　当前遗留问题按对核心目标的影响排列，来源为实施文档已知问题、评审增补和本次验证未执行项。

| 编号 | 问题描述（≤30字） | 影响范围（≤20字） | 来源 | 状态 |
|------|-------------------|-------------------|------|------|
| L1 | W4A8 物理布局描述未实现 | 容量、通信、搬运字节 | K1、Q5 | 未修复 |
| L2 | 分阶段有效成本表未实现 | 混合精度计时 | K2 | 未修复 |
| L3 | CEVA 锚点校准报告未生成 | 单核校准结论 | K3、Q6 | 未修复 |
| L4 | engine 级调度未建模 | 核内重叠 | K4 | 已规避 |
| L5 | 多核互连仍为共享链路假设 | 跨核争用精度 | K5、Q4 | 未修复 |
| L6 | 未知时钟仍输出秒级结果 | 秒级指标可信度 | K6 | 未修复 |
| L7 | 1e 成本表为假设值 | 1e 预测精度 | K7、Q3 | 未修复 |
| L8 | 来源台账未覆盖全部参数 | 参数可追溯性 | K8、K10、Q10 | 未修复 |
| L9 | 64/128 核计划合法性未自动校验 | 多核计划正确性 | K9、Q1 | 未修复 |
| L10 | 相位中间张量不随 Tq 缩放 | 解码小 Tq 高估 | K15 | 未修复 |
| L11 | 旧 sidecar 非 CEVA 行为保留 | 旧产物数字失真 | K12 | 已规避 |
| L12 | `dtype_layout` 按 op_type 单值 | 混合类型成熟度 | K13 | 未修复 |
| L13 | SPLIT 只核元素总数 | 排布错误漏检 | K16 | 未修复 |
| L14 | GEMM A 路与基础设施未逐项核对 | 错误产物漏检 | K17、K19 | 未修复 |
| L15 | 单行内核跨仓契约未定 | 算子编译形态变化 | K18、Q19 | 未修复 |
| L16 | 主机证据独立性依赖产出方 | 校准证据强度 | K14、K20 | 未修复 |
| L17 | B 路产物预算未参与门禁 | 相位产物预算 | K21 | 未修复 |
| L18 | 通用路径仍截断超 4095 次循环 | 通用大形状计时 | K22、Q17 | 未修复 |
| L19 | 工具链身份由评估端反推 | 产物与生产端绑定 | K23、K24、Q18 | 未修复 |
| L20 | 通用后端长循环接口语义不一致 | 通用回退路径 | K25 | 未修复 |
| L21 | `run.sh` 默认配置与帮助文本不一致 | 通用入口易误用 | 代码可见 | 未修复 |
| L22 | 每核 8 GB 字节口径未确认 | 容量预算 | Q2 | 待确认 |
| L23 | 请求轨迹三字段契约待确认 | 带填充的请求 | Q20 | 待确认 |
| L24 | 验证报告与成本 sidecar 契约待对齐 | 成熟度升级 | Q8、Q9、Q13 | 待确认 |
| L25 | 本次未执行 7B 长用例与 CEVA 仿真器全流程 | 完整性能验收 | 6.2 验证项 | 未执行 |
| L26 | `_execution_sidecar` 执行边界无静态约束 | 新增消费者遗漏 | round31 增补 | 未修复 |
| L27 | 工具链预检与首次读取间有改写窗口 | 预检结论 | round31 增补 | 已规避 |
| L28 | 轨迹缓存每次命中读取整份文件哈希 | 大规模轨迹库开销 | round32 增补 | 已规避 |
| L29 | 运行期系数乘积仍可能溢出 | 极端形状通信 | round33 增补 | 已规避 |
