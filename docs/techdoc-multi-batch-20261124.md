# 技术文档：多 batch 全链路支持与性能评估
> 文档编号：techdoc-multi-batch-20261124  
> 创建日期：2026-11-24（round16 收口轮终态首版）；更新日期：2026-11-25（round26 静态收口终态重生成——round25 三条严重问题已根因级修复并经 round26 复核零复发，即实施文档第八节 R5/R11 登记条目的执行）  
> 关联文档：request-multi-batch-20261121、design-multi-batch-20261121、implement-multi-batch-20261121、review-multi-batch-20261121-round01～round04、review-multi-batch-20261123-round05～round10、review-multi-batch-20261124-round11～round16、review-multi-batch-20261125-round17～round26  
> 关联项目：flagos-pim-compiler（图编译器，主仓）、genesim（仿真器）、FlagTree（算子编译器，零改动）  
> 代码基线：pim-compiler `main@1ec124f`（含未提交改动：20 个跟踪文件 19 修改 + 1 删除，+2697/−257，另 6 个未跟踪测试文件 3132 行）、genesim `fjg-dev@e7c2020`（含未提交改动：23 个跟踪文件 +2502/−474，另 2 个未跟踪文件 2718 行）、FlagTree `develop@a53187033`（零改动）；行数统计为本文档撰写时以 `git diff --numstat` 实际核验（本次实际验证）  
> 核心目标：把 batch 维从图编译入口贯穿到 genesim 仿真报告——以 closed batch 打通 B>1 全链路，让 PIM 的权重复用收益真实出现在仿真数字里，并交付可归因、可扫参、可推荐最优配置的评估体系  

## 一、概述

　　本章用一页回答三件事：为什么做多 batch（背景与缺口）、用什么路线做（方案摘要）、做成了没有（需求与核心目标完成度）。详细机制与证据分别见第三、四、六节；「做成了没有」按 2026-11-25 用户终裁的静态收口口径表述——机制、测试与受控实跑为验收依据，盘上重产物数字为旧口径留存且已就地标注，重扫前最优推荐视为未定。

### 1.1 背景与问题
　　本需求的核心目标是让整条推理链路支持多 batch 并给出性能结论。现有基础是大模型到图编译、算子编译再到 genesim 仿真的全流程已经走通（标准形态 tp16_pp4/64 核，`scripts/run_full_pipeline.py`），但它只考虑单 batch：图导出、KV cache 物理布局、内存规划、执行编排、算子成本口径、仿真调度六层全部把 batch 写死为 1。其关键缺口是跨请求执行按时间线性叠加，PIM 架构最核心的收益来源——一次权重装载摊薄给 M 维多行激活——在模型里结构性不存在。本需求要求在不做连续 batching、不做 cascade 级联、CEVA 厂商路径保持 B=1 的边界内（request §2.3），完成 closed batch（等长请求、整组进出）的编译、执行、仿真、评估、优化与扫参全链路。

### 1.2 方案摘要
　　围绕上述目标，总体技术路线是「closed batch 组在编译、执行、计价三条链路里都表现为合并请求」：编译侧 `compile_llama2(batch_size=B)` 导出 `[B,S]`/`[B,1]` 图，KV 与内存按槽位规划，编译结果的公开运行时状态按 B 槽初始化；成本侧按相位出真实算子产物——decode 档逐 B 编 `M=B`、prefill 档逐 B 编 `M=B×S`（合批，与图编译一次执行 `[B,S]` 的真实语义一致），集合通信字节按单行口径写出并以 sidecar 版本 6 固化契约；仿真侧把 B 个等长请求合成一个组请求，prefill 合批一步执行（`Tq=B×S`、注意力按 B 个块对角序列计价）、decode 以合并行数执行（`Tq=B`）。计价的步级绑定收敛为单一真源（`closed_batch_binding`：组行数、每槽 KV 长度），算子、依赖边、集合通信与首次使用四类消费面共用同一份定义；所有 B 的首 token 时点统一在 prefill 完成时刻，TTFT/TPOT 同一定义；流水与权重复用关闭档的逐槽执行策略作为统一编译输入贯穿生产导出链（`--per-slot-prefill`），导出交付 B 份逐槽主机计划、sidecar 声明策略、genesim 与自身执行策略互检。评估归因按「等待区间 ∩ 目标资源占用者」四项守恒分账；四项优化（权重复用、流水重叠、DP 副本、KV head 钉扎）各自独立开关；报告生成器带口径门禁（`result_schema=3`），扫参产物逐档冻结落位。最终在机制与测试层面全部闭合：round25 三条严重问题（依赖传输少算 B 倍、TTFT/TPOT 定义错位、逐槽优化未贯穿生产链）经根因级修复并沉淀 13 个常驻黄金值/生产入口用例，round26 复核零复发；b4g1 单档端到端受控对照（r23-smoke，round23 实跑 EXIT=0）因 R25-01/02 修复已降级标注为旧口径，盘上 15+1 份报告全部「旧口径/推荐未定」，18 档重扫与全档最优推荐按用户终裁登记不执行、重扫前推荐视为未定（见第八节 L7）。

### 1.3 需求完成情况
　　需求 R1–R9 的实现位置与验证情况见表 1。

<p align="center">表 1　需求完成情况</p>  

| 需求 | 需求内容（≤20字） | 完成情况 | 实现位置 | 验证用例 |
|------|-------------------|----------|----------|----------|
| R1 | 多 batch 编译入口与图导出 | 已完成 | `runtime/compile.py:98,247,363`、`scripts/run_full_pipeline.py:1920` | `test_compile_batch.py` |
| R2 | KV 多槽位布局与容量公式 | 已完成 | `memory/kv_layout.py:40,62,69` | `test_kv_layout_batch.py` |
| R3 | 内存 batch 缩放与容量表 | 已完成 | `memory/mem_planner.py:670,699,721` | `test_mem_planner_batch.py` |
| R4 | 编排器批量执行与数值对齐 | 已完成 | `runtime/executor.py:12,102`、`runtime/kernels.py:1238,1335` | `test_executor_batch.py`（含 `compiled.state` 公开契约、逐槽/副本对拍与生产链消费用例）、7B B=2 对拍 |
| R5 | 算子编译相位产物与成本口径 | 已完成 | `genesim_bridge/placement_export.py:234,640,905,934` | `test_cost_batch.py`（含 M=8 前置实验、导出档无关性、prefill 跨层锚点与逐槽计划交付用例） |
| R6 | genesim 组执行与计费 | 已完成 | genesim `gene_sim_scheduler.py:8940,9322,9372`、`capacity.py:765` | `test_closed_batch.py`（官方入口 54 例） |
| R7 | 多 batch 评估模型 | 已完成 | genesim 四项守恒目标资源口径、链路份额与步级绑定单一真源（见 3.1 机制三） | 守恒用例 + 依赖传输/首 token 黄金值用例 |
| R8 | 四项优化独立可开关 | 已完成 | `--weight-reuse`、`--pipeline-overlap`、`--replicas`、`--kv-head-pin`（`run_full_pipeline.py:2157-2171`）；主机侧 `runtime/compile.py:282,409-416`（逐槽/副本） | 两态黄金值用例（关 120/开 90）、`TestWeightReuseSwitch` 3 例、生产入口 5 例、对拍 21 例 |
| R9 | 扫参与最优配置输出 | 部分完成 | `scripts/run_full_pipeline.py:1749,1830,1920` | `test_batch_sweep_report.py`（69 例）；盘上全部报告为旧口径（含 b4g1，R17 登记）；全档最优推荐按终裁登记未定（L7） |

　　需求共 9 项：已完成 8 项，部分完成 1 项（R9——机制层证据完整，盘上全部产物数字与最优推荐为旧口径、按用户终裁登记不重扫，不作为完成度缺口但重扫前推荐结论不得引用）。测试方面，复用已有验证的 round25 终跑 L0 2343 条通过 0 失败、genesim 全量 1643 条通过 0 失败（round26 复核工作区同态适用）；本次实际验证的双锚点（`test_batch_sweep_report` 69 例、genesim 官方入口 `closed_batch` 54 例）与盘上产物核对全部通过（见 6.2）；当前遗留问题 9 条（1 条外部依赖、1 条需求边界待确认、3 条终裁登记边界、4 条边界内取舍或已规避），详见第八节。

### 1.4 核心目标完成度
　　原始需求提出五个核心目标（request §九），逐条完成度、证据与缺口见表 2。该矩阵对应 round26 静态收口终态（review round26 §三逐目标判定，round25 三条严重问题修复后）：代码与机制层全部闭合，常驻黄金值与生产入口用例钉住口径；盘上重产物（含 r23-smoke/b4g1 单档）为旧口径并已就地标注，按用户 2026-11-25 终裁属登记边界、不构成完成度缺口，但重扫前推荐结论不得引用。

<p align="center">表 2　核心目标完成度矩阵</p>  

| 核心目标 | 状态 | 证据（出处） | 缺口/外部阻塞 |
|----------|------|-------------|---------------|
| 1. 多 batch 全链路打通（图编译→算子编译→genesim，含内存、KV、编排器） | 已完成 | prefill 三层同一合批口径：`placement_export.py:905` 逐 B 导出 `prefill_b{B}`（M=B×S，「prefill」键为 prefill_b1 同值副本）、genesim 合批一步（Tq=B×S、lkv 每槽 S、kv_slots=B、缺档硬抛）、`run_full_pipeline.py` 导出侧逐档校验；步级绑定单一真源贯穿算子/依赖边/集合通信/首次使用四类消费面（R25-01 修复，`closed_batch_binding`）；执行策略作为统一编译输入贯穿生产导出链（R25-03 修复，`--per-slot-prefill`）；跨层锚点用例 B=2/4×S=17（图执行折叠 M 与产物 tile_m/权重字节逐项一致）；b4g1 端到端受控对照 EXIT=0（round23 评审实跑 r23-smoke：forward_steps=69、数值与 r19-smoke 逐位一致，复用；该产物经 R25-01/02 修复已降级为旧口径，机制层证据不受影响）；主机批量执行 B=2/4 与 PyTorch 对拍 25 例（round23 实跑，复用）；7B B=2 锚点 1 passed（round23 复跑，复用）；真实 M>1 编译表（round21/25 两轮实编，M=1/8/17/68 权重份数 1.000/1.010/3.039/9.107，M=8 tile_m=8）；FlagTree 零改动 | 无（18 档产物刷新属终裁登记边界，见第 5 行） |
| 2. 多 batch 评估模型（收益与代价并计） | 已完成 | 归因按「等待区间 ∩ 目标资源占用者」四项守恒（组内=同组占用、组间=其它组占用，`gene_sim_scheduler.py:12705-12775`），r23-smoke 实跑四项守恒 busy 53.895 + 组内 194.046 + 组间 0 + sync 0 = wall 247.941 逐位成立（复用 round23）；round21「9 秒链路等待被本组独立 PU 忙时覆盖 3 秒」反例沉淀常驻黄金值（inter_link=9/sync=0，复用 round22）；链路份额 `link_wait_s/intra_link/inter_link` 与全局对账常驻测试（`:12896`）；传输计费按组行数口径（合批含 B、KV 项按槽数放大）与首 token 时点统一两条由 round25 动态反例转绿并常驻黄金值钉住（`TestPrefillTransferGroupBinding` 4 例、`TestFirstTokenMetrics` 2 例，round26 实跑 8 passed）；抢占换出/换入/切换三分项（`:12999-13008`，两方向独立带宽）；收益侧权重摊薄 `ceil(M/tile_m)` 贯穿计费与执行（`vpu.py:918-924`）、组间重叠测度、流水两态 | 抢占系数无实测锚点（解析项，标注未校准）；甲方 SLO 未定（见第八节 L1）；共享带宽仲裁为声明边界 |
| 3. 多 batch 性能优化（权重复用/流水/并行，独立可开关） | 已完成 | 四开关独立在位（argparse `run_full_pipeline.py:2157-2171`）：权重复用（`--weight-reuse` 固定 B 对照，genesim/扫参/主机三层与逐槽机制共用一处判定）、流水（`pipeline_overlap` 两态，黄金值 120/90）、DP 副本（`--replicas`，静态纯权重与动态搬运分列 `placement_export.py:1142`）、KV head 钉扎（`--kv-head-pin`）；R25-03 修复后执行策略贯穿生产导出链——main 按开关传 `per_slot_prefill`（`run_full_pipeline.py:2312-2321`）、CLI 透传 `--per-slot-prefill`（`:1111`）、导出脚本校验 B 份逐槽计划真实生成（`export_pp_placement.py:274-283`，缺失即 SystemExit）、sidecar 顶层策略三态声明（`placement_export.py:744-748`，B=1 不写保逐位）、genesim 与自身执行策略互检（`gene_sim_scheduler.py:9252-9266`，逐槽消费合批声明直接抛）；主机侧最小可执行能力（`compile_llama2(per_slot_prefill=True)` B 份 [1,S] 槽位平移计划、`mem_plans_of_replica` 每副本计划与权重装载、`execute_plan(dpu_window=)` 副本窗口，B=2/4 数值/执行序/KV 落位/副本隔离对拍 21 例）；生产入口 5 例（round26 实跑通过）；缺省关闭 B=1 逐位不变由锁定用例钉住；推荐行携带完整配置身份（`run_full_pipeline.py:610-639`） | 成对 7B 曲线（R13）与主机侧 7B 规模锚点（R14）按终裁登记不执行（见第八节 L8/L9）；两态全档曲线数字为旧口径（L7）；流水收益按产物原义只表述吞吐改善/步间重叠生效 |
| 4. KV cache、内存管理与排布优化 | 已完成 | KV 逐槽分区计费与容量按 B 倍核算（r23-smoke KV zone 6.056e10 = 4×单槽，复用）；`mem_planner.py:670-721` 容量表逐 DPU 拆 fixed_bytes=权重+内核 tile；第三图（逐槽 [1,S]）激活独立规划与权重跨图去重（`plan_dpu` 的 `per_slot_nodes`，`mem_planner.py:313`）；容量零溢出核验读 `pim_capacity_metrics` 的 fits/overflow_*、缺字段硬失败（`run_full_pipeline.py:193-214`）；kv_head_pin 主机侧 KV 读写位置与导出钉扎同源（`kernels.py:1238` 逐 head 持有 DPU 烘进 payload）；DP 逐副本容量视图与物理块互检 | 无（全档容量墙表数字属登记边界 L7） |
| 5. genesim 不同 batch 评估、给出瓶颈或最优配置 | 部分完成（机制层完成；全档最优推荐按终裁定格未定） | 机制层完整：单命令扫参（`--batch/--groups`）产出逐档表、容量墙表、时间构成归因（含链路列）、瓶颈归因（tile_m 上限/容量墙/链路争用三类）与推荐段（携带可执行身份）；报告生成器口径门禁 `_caliber_findings`（`run_full_pipeline.py:261`）按 `result_schema`（现值 3，`gene_sim_scheduler.py:12983`）对旧口径产物自动降级标注；口径门禁对当期逐槽策略档豁免（`test_caliber_gate_exempts_deliberate_per_slot_tiers`，合批档异常仍降级）；末档机制饱和可判（tile_m 封顶证明与观察窗持平分口径）；门禁与报告全文件 69 例（本次实际验证） | 盘上 15+1 份报告（含 r23-smoke/b4g1 单档——R25-01/02 修复使其数字失效并已就地降级）与最优推荐全部为旧口径——按用户 2026-11-25 终裁属登记边界（R9/R10/R12/R17 在册、不执行），重扫前「最优配置推荐视为未定」定格为终态表述、推荐结论不得引用；以机制层与常驻黄金值为验收依据 |

## 二、总体流程

　　本章从端到端视角讲清链路怎么跑：先给总体流程图，再说明各模块的职责边界与依赖。

### 2.1 总体流程图
　　本需求的核心目标是把 B 个等长请求变成一条可编译、可执行、可计价、可归因的执行单元。现有单 batch 链路只把请求维度当作串行外层循环，缺少 batch 维进入形状、产物与调度三层的通路。因此总体流程以「closed batch 组」为主线：首先，编译入口按档位导出 `[B,S]` prefill 图与 `[B,1]` decode 图，KV 按槽位规划、内存按 B 缩放并输出容量表，这三步保证「装得下且地址不冲突」是编译期事实。然后，算子编译按相位出真实产物——decode 档逐 B 编 `M=B`、prefill 档逐 B 编 `M=B×S`，placement sidecar 以 `pimir_by_phase` 索引逐相位逐档产物，集合通信边按单行口径写出并由版本 6 声明契约，成本 IR 沿用两点拟合在 `Tq=B` 处求值。其次，genesim 把 B 个 trace 请求静态预分组为一个组请求：prefill 合批一步执行（`Tq=B×S`、注意力按 B 个块对角序列计价）、decode 步合并执行（`Tq=B`），算子与依赖边计费共用步级绑定单一真源，权重搬运按 `ceil(B/tile_m)` 折算，KV 按槽计费与核算容量，首 token 时点在 prefill 完成统一记录，运行输入按内容寻址冻结并落位到逐档结果目录。最后，扫参入口对多档重复执行并聚合出曲线、瓶颈归因与最优配置推荐，报告以产物指纹自证引用哪一代盘上产物，并由口径门禁对旧语义产物自动降级标注。如图 1 所示，该图的主链正是「batch 维从入口贯穿到报告」这一核心改造本身。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff,color:#fff

  A["编译入口<br/>[B,S]/[B,1] 图导出"]:::stage --> B["槽位规划<br/>KV B 槽 + 容量表"]:::stage
  B --> C["逐相位算子产物<br/>pimir_by_phase 索引"]:::stage
  C --> D["genesim 组执行<br/>prefill 合批 + decode Tq=B"]:::stage
  D --> E["归因与扫参<br/>目标资源守恒 + 口径门禁"]:::stage
  E --> F["交付报告<br/>指纹自证 + 冻结落位"]:::stage
  W["closed batch 前提<br/>等长请求、整组进出"]:::input -.-> A
  W -.-> D
```

<p align="center">图 1　多 batch 全链路总体流程</p>  

### 2.2 模块关系
　　本次改动横跨两个仓库的六个模块层与 FlagTree（零改动），各模块的职责与依赖见表 3。

<p align="center">表 3　模块关系</p>  

| 模块/文件 | 职责（≤20字） | 改动类型 | 依赖的模块 |
|-----------|---------------|----------|------------|
| `runtime/`（compile/executor/kernels） | batch 图导出、批量与逐槽 decode、副本路由 | 修改 | `contracts/`、`memory/` |
| `memory/`（kv_layout/mem_planner） | KV 槽位布局、三区规划、第三图与容量表 | 修改 | `contracts/` |
| `backend/hal_numpy.py` | pos 按槽元组、DPU 副本窗口 | 修改 | — |
| `genesim_bridge/placement_export.py` | 逐相位产物、单行口径、sidecar 载荷 | 修改 | FlagTree 编译缓存 |
| `scripts/`（run_full_pipeline 等） | 扫参入口、报告聚合与口径门禁、冻结落位 | 修改 | 全部上游 |
| genesim `src/`（scheduler/capacity 等） | 组执行、合并计费、目标资源归因 | 修改 | placement sidecar |
| FlagTree | 算子编译（M 分块已就绪） | 零改动 | — |

## 三、核心原理

　　流程图回答「怎么跑」，本章回答「为什么这样跑能解决问题」：先讲三个关键机制的技术路线，再列设计取舍与适用边界。

### 3.1 关键机制
　　为了达成「batch 维贯穿全链路且数字可信」，需要同时保证三件事：形状真源唯一（batch 只从导出图进入）、计价真实（成本来自真实 M>1 编译产物且与编译档解耦）、归因可对账（收益与代价从事件流直接累计且按目标资源守恒）。现有实现分别把 batch 写死在六层、代表编译钉死 M=1、跨请求执行线性叠加，三者皆缺。总体采用「组即合并请求 + 逐相位真实产物 + 可对账口径」的技术方法，通过下面三个机制解决。

　　**机制一：closed batch「组即合并请求」**旨在让 B 个等长请求成为 genesim 的一个原生执行单元。采用在准入层合组而非在派发层跨请求合并的技术路线：`_form_closed_batch_groups`（`gene_sim_scheduler.py:8940`）把 B 个等待请求静态预分组成一个组请求（占一个 `request_index`、携带 `slots=B`，`src/ir/trace_ir.py:59`），步链长度为 prefill 步数加 `output_length − 1`；`prefill_step_count`（`:9322`）缺省返回 1 步合批 prefill，`pipeline_overlap` 开启或 `weight_reuse` 关闭时返回逐槽 `slots` 步（`pipeline_overlap or not weight_reuse` 在 `:9251` 单处判定，三种策略共用一个入口）；步级形状由单一真源 `closed_batch_binding`（`:317-326`，返回组行数 `slots×query`、每槽 past、每槽 KV 长度）定义，`step_shape`（`:9372-9401`）把合批 prefill 步绑定为 `Tq=B×S`、lkv 每槽 S、`kv_slots=B`（phase_key=`prefill_b{B}`，注意力按 B 个块对角序列分别计价），其余步绑定为合并 decode（`Tq=B`、lkv=`tp+1`），依赖边计费、集合通信与首次使用消费同一份绑定（R25-01 修复，见 4.13）。首 token 时点在指标层同一定义：`ttft_step = prefill_step_count(request) − 1`（`:10935`），所有 B 都在「prefill 完成、首 token 可采样」时刻记录 TTFT、TPOT 以该时点起算（R25-02 修复，见 4.14）。prefill 合批的依据来自真实算子编译：设计期推演曾以等式 `ceil(B×S/tile_m)=B×ceil(S/tile_m)`（S≥tile_m）为由选择逐槽执行，round17 以真实编译证伪该等式（S=17、tile_m=8 时合批装载 9.107 份、逐槽 12.155 份，S 非 tile_m 整数时合批严格更少），且注意力的正确做法是块对角绑定而非放弃合并（用户 2026-11-25 裁决）。这样派发层、DAG 实例化、资源互斥排队全部复用——组就是一个请求，天然继承现有争用与流水重叠机制，避免在万行调度器里重写事件循环。

　　**机制二：逐相位真实产物与解耦口径计价**旨在让权重复用收益来自真实编译产物而非外推，且计价不随编译档漂移。产物侧采用相位产物加拟合求值的路线：`_measure_kernel_tile_n` 的缓存键加入行数 M（`placement_export.py:234`），`export_placement_to_genesim`（`:640`）按相位出产物——decode 档逐 B 编 `M=B`，prefill 档逐 B 编 `M=B×S` 合批产物（`:905`，「prefill」键为 prefill_b1 同值副本服务逐槽策略与旧产物）；sidecar 算子条目扩为 `pimir_by_phase`（`:934`），顶层旧字段保留为 decode_b1 同值副本，旧消费方零改动。genesim 按相位选取产物后，GEMM 执行走 `ceil(M/tile_m)` 折算重放（`src/vpu/vpu.py:918-924`），权重搬运份数随 B 摊薄；成本系数无需逐档重测——两点线性拟合的截距即权重字节（与 M 无关），在 `Tq=B` 处求值即覆盖合并行数。KV 侧由 `Operator.kv_slots`（`src/ir/model_ir.py:140`）在合并绑定（prefill 与 decode 皆记组槽数）时落槽数，`resolve_phase_template`（`src/pim/pimir_trace.py:1417`）只对「跨度含 KV 长度轴且不含 Tq 轴」的操作数把字节乘槽数，span 形状与 MAC 口径不动。口径侧有三道保障：其一，集合通信边在 `[B_export, S]` 导出图上测量，`_collective_entry`（`placement_export.py:138`）对形状首维等于 B_export 的边把 nbytes 除回（掩码广播形状首维为 1 不除），sidecar 以版本 6 声明该单行口径（`:711`），genesim 装载入口对带 collectives 的旧版本产物直接拒绝（`gene_sim_scheduler.py:2048`）；其二，运行输入按内容寻址冻结进结果目录（`src/run_manifest.py:1297` 的 `freeze_manifest_inputs`），并在复制到逐档目录后由 `relocate_manifest_to_tier`（`scripts/run_full_pipeline.py:1339`）把 manifest 与 36 份 PIM MLIR 重定位为本档内容寻址副本，结果目录脱离共享路径独立可复现；其三，报告生成器以 `_caliber_findings`（`run_full_pipeline.py:261`）校验 `result_schema`（现值 3）、prefill 步数语义与归因字段集，旧语义产物自动带「旧口径警示」且推荐段标注未定。执行策略也进入产物契约：sidecar 顶层 `per_slot_prefill` 三态声明（B>1 恒写 true/false、B=1 不写保逐位，`placement_export.py:744-748`），genesim 消费时与自身执行策略互检——逐槽策略消费合批声明直接抛、合批消费逐槽声明是保守方向放行（`gene_sim_scheduler.py:9252-9266`）。前置实验证实机制可用：同一 linear 的 M=8 产物 tile_m=8、权重搬运 1.010 份（Q1，round25 复编复证）。

　　**机制三：可对账的评估口径**旨在让「收益与代价各花在哪」可从产物直接核验。时间构成采用目标资源交段的四项守恒：对每份等待区间找到其目标资源的占用者，组内争用=等待∩同组同资源忙时、组间争用=等待∩其它组同资源忙时、busy=忙时并集减被争用类别认领的段、sync=余项，四项在时间轴上按组内→组间→busy→sync 的优先级互斥分账（`gene_sim_scheduler.py:12705-12775`），负数直接抛。该口径由 round21 反例驱动定案：旧判据用「暴露等待」与「链路等待−本组全资源忙时覆盖」两个代理量，本组在无关 PU 上同时有计算即把真实组间链路等待挤进 sync——9 秒链路等待中 3 秒被本组独立 PU 覆盖的反例已沉淀为常驻黄金值用例（TestLinkWaitCoveredByOwnBusy，inter_link=9/sync=0/守恒=wall）。传输派发（`dispatch_transfers`，`:10978`）按组补记 (ready, start, link) 等待区间与 PU 同口径进区间台账，链路份额单列 `link_wait_s`（原始等待合计，与 `performance_metrics.total_transfer_queue_wait_time_s` 按组求和对账）、`intra_link_contention_s`（链路等待与同组同链路忙时交段）、`inter_link_contention_s`（其它组占用部分），字段落 `:12896`。字节构成按 zone（权重/激活/KV）归因：GEMM 产物按静态占比拆分，其余轨迹整算子归 sidecar 声明区间，`Σzone = 搬运总字节` 逐组断言；依赖边的字节按步级绑定求值（`evaluate_step_binding`，`src/ir/model_ir.py:27-54`，行轴取组行数、与 Tq 轴配对的 KV 长度保持每槽、不带 Tq 轴的 KV 项按槽数放大）。抢占是纯解析项且覆盖完整往返：`_preemption_link_bytes_per_s`（`:8900`）按方向取生效拓扑 PIM↔HOST 各方向链路最小带宽，换出分子是一次换出的整组 KV，分项输出 `swap_out_s/swap_in_s/switch_cost_s` 三项（`:12999-13008`，切换常数缺省 0），报告单列并标注「analytical，未校准」。扫参侧以产物指纹（`_artifact_fingerprint`，`run_full_pipeline.py:223`）、口径维度集（`sweep_scope_dimensions`，`:1749`，枚举全部影响放置与容量口径的输入，含 weight_reuse 维）与索引口径标注（`_write_report_index`，`:1830`，从各报告头部口径警示行读取世代）建立报告与盘上产物的一致性约束，被覆盖的行显式标注过期；`batch_metrics` 落 `result_schema` 口径锚点（现值 3，`gene_sim_scheduler.py:12983`，R25-01/02 修复升版），报告门禁按该值自动降级旧产物；容量核验读逐档 `pim_capacity_metrics` 的 `fits/overflow_*`，缺档缺字段硬失败、非零溢出报警（`:193-214`）。

### 3.2 设计取舍
　　设计与评审过程中的主要取舍（均有文档与用户裁决记录）见表 4。

<p align="center">表 4　设计取舍</p>  

| 取舍点 | 采用方案 | 未采用方案 | 理由（≤30字） |
|--------|----------|------------|---------------|
| batch 语义落点（design §三） | 组即合并请求 | 派发时跨请求合并 | 后者侵入依赖跟踪，复杂度留给连续 batching |
| KV 布局键结构 | 槽位外层主序 | 每槽独立 spec | 抢占按槽迁移 KV，槽连续最友好 |
| 成本链 batch 表达 | 相位产物 + 拟合求值 | 新增 Tb 符号维 | 避免改 genesim IR 契约三处 |
| prefill 是否合并 | 合批一步 M=B×S | 逐槽 Tq=S（中间口径已作废） | 真实编译证伪等式，与真实执行一致（用户裁决） |
| 流水两态（I29） | 可切换执行策略 | 删除其一 | 删开关砍 R8 流水交付；两态口径自洽 |
| 组间争用判据（round22） | 等待∩其它组同资源忙时 | 暴露等待等代理量 | 代理量与占用者无关，反例实证（用户裁决） |
| 权重复用开关（round22） | `--weight-reuse` 固定 B 对照 | `--batch` 档位本身 | 后者耦合组大小/形状/装载，收益无法隔离 |
| 主机侧执行能力（round22） | 最小可执行切片 + API 对拍 | 完整 stage 级重叠调度 | 用户裁决最小切片；时间维重叠仍在 genesim |
| weight_reuse 关档 decode（I34） | 合并步 + decode_b1 产物 | 拆 B 个 M=1 子步 | 重放同价，事件结构零改动 |
| kv_head_pin 主机侧（round22） | 不另做执行计划 | 第二套钉扎开关 | KV 读写位置本就钉在 head 持有 DPU |
| 旧口径标注（round22） | 生成器口径门禁 | 手工补警示行 | 索引重建会丢手工标注（round16 已证） |
| 冻结落位（round22） | copytree 后重定位复验 | 只冻在 genesim 工作目录 | 结果目录需自包含、可独立搬移 |
| B=1 G>1 容量（round22） | 从实际 trace 推导口径 | 回落 MAX_SEQ=64 | 容量表与 trace 同源是硬契约 |
| 组准入时机 | FCFS 静态预分组 | 运行时凑组成组 | 到达序确定下等价，生命周期不变量零改动（用户确认） |
| 小档容量 | 折算 + 显式标注 | 逐档真算 | 省一次 7B 图导出，判据只对编译档逐位成立 |
| 变长 padding | 维持等长硬前提 | 长度抖动 + padded 对齐 | 变长属连续 batching 范围（需求 §2.3） |
| DP 物理规模口径 | 校验侧按 replicas 放大期望 | hardware.num_dpus 记总规模 | sidecar 载荷零改动，与每副本口径自洽 |
| 通信字节口径落点 | 导出侧除回 B_export | 消费侧挂系数前除回 | 生产入口唯一，B=1 导出零变化（round14） |
| sidecar 版本门禁范围 | 带 collectives 产物须 v6 | 装载入口无条件门禁 | 保住 legacy 兼容面，危害面零遗漏（I28） |
| 步级行数定义（round25） | `closed_batch_binding` 单一真源 | 各消费面自行乘槽数 | 两套行数曾致通信量少算 B 倍 |
| 首 token 时点（round25） | prefill 完成即记 TTFT | 首个合并 decode 完成时刻 | 与真实编排器逐槽采样对齐，TPOT 不为零 |
| 成对扫参 sidecar（round25） | 两态共用逐槽导出 | 两态各导一次 7B 图 | 合批消费逐槽产物为保守方向，省分钟级导出 |
| 口径门禁豁免（round25） | 当期逐槽策略档免降级 | forward_steps 无差别判旧 | 区分旧口径逐槽与当期策略档，合批档仍门禁 |
| 冻结失败处理（round18） | 逐条复制、失败即抛 | 两阶段回滚 | 冻结失败即整轮失败，半改写清单无消费面 |

### 3.3 适用前提与边界
　　本方案的适用前提与边界如下。

- 前提：closed batch 等长硬前提——组内请求等长、等输出长度、请求数为 B 的整倍数；不满足时：组准入直接抛 `ValueError`。  
- 前提：容量装不下保持硬失败（`PIM_CAPACITY_EXCEEDED` 标记，`capacity.py:765`）；不满足时：无兜底、无降级告警。  
- 前提：B=1 走完全相同路径——全部新参数缺省值等价旧行为；不满足时：B=1 锁定用例即失败。  
- 边界：连续 batching（动态合批、变长 padding、真抢占执行侧）、cascade 多 CEVA 级联、CEVA 厂商路径 batch 化均不做（request §2.3）。  
- 边界：主机侧逐槽/副本为最小可执行切片——逐槽步顺序执行、副本为物理块路由；时间维重叠仍在 genesim 事件循环内。  
- 边界：带 collectives 的 v5 旧口径 sidecar 被 genesim 装载门禁拒绝；跨口径数字不可比。  
- 边界：盘上 18 档 sweep 产物、15+1 份报告与最优推荐为 R25-01/02 口径修复前的旧口径（round16/round18 产物及 r23-smoke/b4g1 单档），已全部就地标注「旧口径警示」；新口径数字以实施文档第八节 R9/R10/R12/R17 登记条目为准，重跑登记为不执行、重扫前推荐视为未定。  

## 四、关键问题与解决方案

　　评审与实施暴露的问题按「是否阻碍核心目标达成」可分为四类：计价正确性（数字对错）、契约完整性（公开入口与产物契约）、交付完整性（产物与证据在盘）、口径可信性（报告与产物语义一致）。下面按此顺序逐条展开；与核心目标无关的局部修复不单列。

### 4.1 prefill 的三层 batch 语义不一致（来源：round17 R17-01、round18 问题 1，严重）
　　为了让「一次执行 `[B,S]` prefill」的全链语义一致，图执行、算子产物与 genesim 计价必须采用同一种 batch 语义。缺陷形态是三层各自为政：图编译与编排器一次执行整张 `[B,S]` prefill 图（算子驱动把前导维全部折进行数，真实编译行数 M=B×S），placement 侧却只导出 M=S 的单档 prefill 产物，genesim 又把组内前 slots 步逐槽绑定（每步 Tq=S）——prefill 的权重复用漏计，全链性能数字失真。设计期曾以等式 `ceil(B×S/tile_m)=B×ceil(S/tile_m)`（S≥tile_m）为由选择逐槽，round17 以真实算子编译证伪该等式（S=17、tile_m=8 时合批装载 9.107 份、逐槽 12.155 份，S 非 tile_m 整数时合批严格更少），且注意力的正确口径是按 B 个块对角序列计价而非放弃合并（用户 2026-11-25 裁决，request §七.6 与 design §八.4 保留被证伪结论原文并标注作废）。修复按「与真实执行一致」统一三层：`placement_export` 逐 B 导出 `prefill_b{B}`（`M=B×S`，「prefill」键为 prefill_b1 同值副本，`placement_export.py:905`）、genesim prefill 合批一步（`Tq=B×S`、lkv 每槽 S、`kv_slots=B`、注意力块对角计价，步链 `gene_sim_scheduler.py:9322` 起、绑定 `:9372` 起）、GEMM 权重字节改取算子绑定行数；`pipeline_overlap` 开启或 `weight_reuse` 关闭时切换为逐槽 micro-batch 两态策略（4.6）。验证由跨层锚点用例（`test_cost_batch.py::test_prefill_merged_product_matches_graph_execution`，图执行折叠 M 与产物 tile_m/权重字节逐项一致，先红后绿）、b4g1 端到端受控对照（round19 与 round23 两代实跑 EXIT=0、prefill 恰合批 1 步 forward_steps=69、四项守恒 0.0，数值逐位一致）锚定；受影响的 18 档产物按终裁登记不执行重跑（第八节 L7）。

### 4.2 合并 decode 的 KV 计费不随 B 放大（来源：round05 问题 3，严重）
　　为了实现「KV 按槽位分区计价」，需要 IR 系数层与轨迹层给出同一个数。缺陷形态是两层各自实现且无一致性约束：IR 系数层 `Tq(Tp+Tq)` 在 lkv=tp+1 下正确乘 B，轨迹层静态 span 只表达一份 KV——实测对照实验 Tq=1 与 Tq=4 的 KV 输入字节同为 129×128（应为 4 份），产物侧 KV zone 仅 1.74×（应为约 8×）。修复以 `Operator.kv_slots` 为单一真源：合并绑定时记槽数，`resolve_phase_template` 对「跨度含 KV 长度轴且不含 Tq 轴」的操作数把字节乘槽数（`pimir_trace.py:1417`），缓存身份三处同步加 kv_slots 防串档；并新增真实 `PIMVPU` + 真实相位链 pimir 的跨层锚点用例 `test_merged_decode_kv_bytes_scale_with_slots`（先红后绿）。该用例所在的真实路径正是此前 mock 用例（round05 问题 8）掩盖缺陷三层未暴露的盲区。产物复核 KV zone 组均严格 B 倍：b4g1 冒烟 KV zone 6.056e10 = 4×单槽（round19 与 round23 两代一致），旧口径全档 14.10→451.22 GB 三个 G 档严格 B 倍（复用 round16）。

### 4.3 归因口径的三次演进：从同义反复到目标资源交段（来源：round04 问题 2、round05 问题 4、round17 R17-02、round21 R21-01，严重）
　　为了满足「时间构成守恒」验收判据，归因口径必须能看见组内争用，且等待的每一份都要归到正确的资源类别。该口径经三次演进闭合。第一层（round04/05）：原并集口径下单组（G=1）在 64 核上忙时并集恒等于 wall，暴露等待与 sync 构造性为零、守恒式恒成立，而真实的组内 PU 争用（等待占 wall 57%–60%）报告不可见——重构为四项守恒（等待区间带目标资源、忙时按组与资源分账、组内争用按目标资源维度累计）。第二层（round17/18）：链路排队等待的记账点只在算子派发路径定义，传输派发没有对称的逐组区间落账，链路等待被守恒式余项静默吸收——修复把传输派发按组记等待区间进同一台账（`dispatch_transfers`），链路份额单列三项并与 `performance_metrics` 全局对账。第三层（round21/22）：组间判据仍用「暴露等待」与「链路等待−本组全资源忙时覆盖」两个代理量，都与「谁占用着等待的目标资源」无关——round21 以最小事件反例实证：两组共享链路、9 秒等待中 3 秒被本组另一独立 PU 的忙时覆盖，真实组间链路等待被挤进 sync（inter_link=6/sync=4，守恒仍成立——「守恒通过」不能证明「分类正确」）。round22 按用户裁决改为目标资源交段口径：组间=等待∩其它组同资源忙时、组内=等待∩同组同资源忙时、busy 让位、sync 为余项，时间轴按组内→组间→busy→sync 优先级互斥分账（`gene_sim_scheduler.py:12705-12775`）。反例沉淀为常驻黄金值（TestLinkWaitCoveredByOwnBusy：inter_link=9/sync=0/守恒=wall），round23 再以两条全新对抗路径探针与真实 7B 负载复现（r23-smoke：busy 53.895 + 组内 194.046 + 组间 0 + sync 0 = wall 247.941）。守恒断言保留为口径回归门——后续性能优化（4.5）的首版错误聚合正是被该断言当场拦下。

### 4.4 编译期容量墙与仿真序列长度脱节（来源：round04 问题 3、round05 问题 5、round22 问题 5，严重）
　　为了回答「多 batch 首先被什么卡住」，编译期容量表的序列长度必须与实际执行的 trace 同源。缺陷形态是导出脚本模块常量写死 `MAX_SEQ=64`，而 trace 实为 726+69=795 token，墙位 4370 与真实墙位相差约 90 倍，扫参档位永远碰不到墙。round05 修复建立同源契约：`synthesize_closed_trace` 前置到导出之前，`closed_trace_max_seq` 传入 `--max-seq` 建 KV 布局与 `capacity_batch_fit`（`mem_planner.py:699`），sidecar `capacity.max_seq` 记录口径，扫参读入断言一致、缺键或不一致直接抛。round22 又钉住残留分支：B=1 且扫多组时推导以「是否合成 closed trace」为触发条件而直接返回 None，容量口径退回 MAX_SEQ=64——修复为 B=1、G>1 时从仿真实际消费的原始 trace 首请求推导容量口径（`run_full_pipeline.py:1717-1735`），经典 B=1、G=1 单档路径按用户裁决保留缺省。修复后墙位与手推公式一致（b4g1 冒烟 B≥976，复用 round19；旧口径全档 975，复用 round16），并与运行时容量占用（6%→9%、溢出 0）构成编译-运行时互检。容量核验本身经 round18/22 两轮加固：读逐档 `pim_capacity_metrics` 的 `fits/overflow_*`（`run_full_pipeline.py:193-214`）——此前的 `.get` 默认值链把「数据缺失」静默折叠成「零溢出通过」（round17 R17-08）。

### 4.5 新口径事件循环仿真开销高一个数量级（来源：Q10、I23，实施记录）
　　为了让四项守恒台账可用，逐算子区间记账被引入，但同规模仿真耗时约为旧口径十倍（单组 b4g1 四请求 42 分钟以上未完），全档扫参不可行。定位经 cProfile 剖析与 SIGINT 栈采样收窄为三层：`latency_cache_key` 对 PIMTrace 纯求值核恒返回 None 导致逐任务全量重放（窗口内约 47% 墙钟）；`step_context`/`_get_operator_with_runtime` 逐任务 `replace` 拷贝 122 万次；末段组内争用聚合逐等待区间物化交段，规模 O(等待数×队列深度)。修复对应三层：PIMTrace 核按算子身份入键缓存（带仿真状态的原生核维持不缓存）、形状与运行时请求按（请求,步）记忆化、聚合改为「等待按资源并集 + 互斥有序忙时双指针求交 + 全局并集收口」。受控对照 42 分钟以上未完 → 2 分 52 秒含落盘；重放次数锚点用例 `TestEstimateReplayAnchor` 防止再次引入超线性记账。该修复使全档 18 档扫参重新可用，是最终交付形态的前提。

### 4.6 R8 优化的完成口径两度裁决与全链落地（来源：round04 问题 1、round05 问题 1/2、round21 R21-02/R21-04，严重；用户裁决 2026-11-23 与 2026-11-25）
　　为了满足「每项优化独立可开关」，R8 的交付口径经两次用户裁决收敛。第一次（2026-11-23）：设计曾自行把 R8 降级为「三个档位都是扫参维度、DP 只做解析估算」，与需求「消减气泡」「空间映射」直接冲突，被评审判为需求未完成；裁决以需求为准回写设计后，三项优化在 genesim 侧真实落地——流水是事件循环内的 stage 间 micro-batch 重叠调度（`pipeline_overlap` 缺省关闭），DP 是 `ShardStrategy.replicas` 副本维度 + sidecar 物理放置 + 逐算子副本路由，另有 KV head 钉扎第四开关。第二次（2026-11-25，round21）：评审钉住「优化只在仿真侧可执行、完成口径被混用」——主机编排器恒一次 `[B,S]` prefill、runtime 无 overlap/replicas 执行逻辑，且 `--batch` 同时改变组大小/执行形状/装载份数，权重复用收益无法在固定 B 下隔离归因。round22 按裁决补齐两端：其一，主机侧最小可执行切片——`compile_llama2(per_slot_prefill=True)`（`runtime/compile.py:247,282`）额外导出 [1,S] 逐槽图并构建 B 份槽位平移计划（`payload` 的 k_off/v_off 平移 slot×槽步长，每步 M=S、KV 落本槽段），`run_decode_loop`（`runtime/executor.py:102,116`）B 步逐槽执行且数值与合批一致；`CompiledModel.replicas`/`mem_plans_of_replica` 每副本物理块计划、`load_weights` 按副本各装一份权重、`execute_plan(dpu_window=)` 副本窗口（submit 派发/copy/ptr/dpu_set 视图四处同窗口、finally 复位）。其二，`--weight-reuse` 固定 B 对照开关（`run_full_pipeline.py:2157`；genesim `scheduler.weight_reuse` 缺省开=合批现状逐位不变，关=逐槽 prefill 步 + decode_b1 产物，重放 ceil(B/1)=B 次与逐子步拆分装载同价）。kv_head_pin 主机侧不另做计划：sdpa 命令的 KV 读写位置本就钉在 head 持有 DPU（`kernels.py:1238,1355-1392`），与导出钉扎同源。验证为 B=2/4 公开 API 数值与执行顺序对拍（含 KV 落位与副本隔离断言 21 例）、`TestWeightReuseSwitch` 3 例（固定 B 装载份数 4 倍差）、缺省关闭不变用例，以及 7B B=2 锚点在 round22 改动后的复跑（round23 实跑 1 passed）。round25 第三度钉住该口径的最后一环：逐槽 API 虽在，生产导出链从未传 `per_slot_prefill`，优化策略与同次图编译产物脱节（R25-03，修复见 4.15）——修复后执行策略作为统一编译输入贯穿生产链并由 5 例生产入口测试钉住（round26 实跑通过）。

### 4.7 DP 副本链两处真源失配与静态/动态混写（来源：round09 问题 1、I25、round17 R17-04，高）
　　为了让 `--replicas 2` 端到端跑通且代价口径可解释，sidecar 的物理规模必须有单一口径、权重复制量必须区分静态与动态。缺陷形态有三：其一，副本物理扩展（`dpu_to_cluster`/`kv_layout` 按 128 落）与编译契约摘要的每副本基数（hardware.num_dpus=64）是两个未对齐真源，genesim 严格身份核验按单一基数期望即拒；其二，`_validate_dp_replica_capacity` 逐副本窗口的 KV 峰值向量按局部组下标落，执行段抢占分子按全局下标索引即越界；其三，报告把动态权重搬运量（全生命周期逐相位累计，b4g2dp2 两组合计 1.003e13 B）写成静态「每副本全量权重复制」（sidecar 静态量 1.853e10 B），两口径相差约 270 倍，优化权衡解释错误。定案口径为「物理 DPU 总数 = hardware.num_dpus × strategy.replicas」（与 `strategy.total_dpus` 互检），genesim 核验按该口径放大期望（`src/run_manifest.py`），向量并入前展开到全局组下标；DP 载荷拆纯字段 `weight_bytes_per_replica`（纯权重）与 `kernel_tile_bytes_per_replica`（`placement_export.py:1142`），报告 DP 段静态副本量与动态权重搬运分列呈现。修复后 `--replicas 2` 端到端跑通并出实测产物；补「dp2 sidecar 过严格身份核验」一致性单测作为回归门。

### 4.8 集合通信字节携带编译档 batch 维（来源：round09 问题 2、round12–13 严重 2，K6/I27）
　　为了让扫参报告可核验、仿真结果不随无关输入漂移，报告与产物之间以及产物与导出档之间都必须有一致性约束。缺陷的发现路径跨越三轮：round09 先暴露「报告引用已消亡产物世代」——B 曲线报告的 B=4 行与盘上产物同点矛盾，修复先建立指纹与独立产物根等审计机制；round12 以受控盘上对照钉住同一 (B,G) 档 wall 差 16.5% 且单算子服务量逐位相同、两份 genesim.log 逐字节相同（排除排程差异）；round13 以受控重跑完成对账闭合——两代 sidecar 逐字段 diff 仅四组，B_export 相关通信份额恰 2.0 倍、通信服务差额 91.845s 与总时间差逐位相等，根因定位于导出侧 `_export_seq_len` 只除序列维、batch 维留在 nbytes 里，消费侧按 `nbytes/seq_len × Tq` 计价使同一 B 运行的通信量正比于 B_export。round14 修复为导出侧单行口径（`_graph_export_shape`/`_collective_entry`，`placement_export.py:75,138`），B=1 导出零变化；round15 补 sidecar 版本 5→6 三处同步与装载门禁（`gene_sim_scheduler.py:2048`）。验证为 k6r2 受控对（仅改导出档列表 1,2,4,8 对 4）：修复后两侧 `total_time_s` 同为 508.3315908233462、batch_metrics 全文 JSON 逐位一致；导出档无关性单测常驻测试集；旧口径收口轮实测 b1g1 total 5096.97→2330.65（通信虚高消除，旧口径复用 round16）。

### 4.9 公开编排契约 `compiled.state` 断链（来源：round12–14 严重 1）
　　为了让「编译结果原样驱动编排器」成为公开契约，`compile_llama2` 返回的全部字段必须随 batch 语义一次改齐。缺陷形态是图导出、KV 槽数、内存计划、双相位命令计划均已按 B 伸缩，唯独运行时状态恒为 `state = DecodeState(valid_len=0)`（单槽），而 `run_decode_loop` 要求状态槽数等于 prompt 槽数——`compiled.state` 原样直进公开入口时 B>1 在 prefill 前即抛错。该断链被既有全绿测试掩盖三轮：批量测试全部手工另建 `DecodeState` 绕过编译器契约。round12 首次钉住并以同款配置复现，round14 修复：`state = DecodeState(valid_lens=(0,) * batch_size)`（`runtime/compile.py:363`），新增 B=2/4 公开契约用例（编译结果不另建状态直接进 `run_decode_loop` 与 PyTorch 对拍），既有批量用例全部改为消费公开字段，从机制上消灭「测试自建状态」这一掩盖路径；7B B=2 锚点在收口轮与 round22 改动后两次以新契约复核（1 passed，复用 round16/round23）。

### 4.10 产物证据链可复现、抢占解析项不完整与冻结落位（来源：round17 R17-03/R17-05、round21 R21-06，高）
　　为了让盘上结论可复核、抢占代价不残缺、结果目录可独立搬移，产物生命周期需要「逐运行冻结 + 逐档落位」约束，解析建模需要覆盖完整往返。缺陷形态有三：其一，导出脚本缺省输出共享可变路径，manifest 只记身份不保存内容，后续运行覆盖同一路径后历史档不可复现——final2 的 18 档产物即因共享 sidecar 被后续冒烟再生而失配；其二，抢占估算只建模单向换出，换入、恢复与切换成本从未进入公式；其三，round18 的 `freeze_manifest_inputs` 把输入冻结在 genesim 工作目录，但结果目录经 `copytree` 落位后 manifest 仍指共享 `genesim/results/inputs`、36 份 PIM MLIR 只记原路径——搬移结果目录或清理共享目录后「独立复现」不成立。修复分三层：输入冻结（`src/run_manifest.py:1297`，manifest 改指冻结副本并留 origin_path，`main.py:438-440` 落盘前调用）；抢占三分项（`swap_out_s/swap_in_s/switch_cost_s` 按方向独立带宽，`gene_sim_scheduler.py:12999-13008`，报告分项呈现并保持 `mode=analytical/calibrated=false` 标注，某方向缺链路时分项保持 None 不并入总值）；冻结落位（`relocate_manifest_to_tier` 挂 copytree 后：inputs 改指本档副本、pim_mlir 逐份冻成本档内容寻址副本、全部复验哈希，`run_full_pipeline.py:1339`）。验证为冻结三失败样例与双向带宽分项用例常驻测试集、r19-smoke 产物静态落位（round22）、r23-smoke 端到端冻结落位生效（inputs 3 条 + MLIR 36 份全指本档，round23 实跑）与「全档→单档→重建报告逐字节一致」静态回归。

### 4.11 调度器文件事故与 pyc 对照重建（来源：I20，实施记录）
　　为了在事故后恢复多轮未提交修复，重建必须以事故前的编译产物为真源。事故是修复中途误执行 `git checkout -- src/scheduler/gene_sim_scheduler.py`，该文件 rounds 01–05 的未提交改动（约 +1100/−260）被还原到 HEAD，且部分修复会话转录不可得。恢复采用三路并逐指令验证：重放可得的 46 个 Edit；本轮读过的最终态区域按上下文逐字恢复；其余缺口以事故前最后一次 3.12 解释器编译的 `__pycache__` 产物为真源，按反汇编逐指令翻译恢复，重建文件编译后与 pyc 做函数级对照，319 个代码对象零缺失、零多余、零指令差异。随后定向与全量套件复跑全绿。该事故的教训已固化为约定：不对未提交工作区执行还原类 git 命令。

### 4.12 旧口径产物错标「新口径」与报告口径门禁（来源：round21 R21-03、round22 问题 3，严重）
　　为了让「盘上报告的口径标签」可信，口径判定必须来自产物内容的机械校验而非生成时刻的自述。缺陷形态由产物结构直接证明：final2 的 b4g1 `forward_steps=72`（旧路径 4 次逐槽 prefill + 68 decode），而当前代码语义的 round19 产物为 69（1 次合批 + 68 decode）；`report-full-grid.md` 的 prefill 服务时间全 B 恒 8359.23s（逐槽口径的构造性结果）、表头无链路争用列、最优推荐 `B=4,G=4（1.74 tokens/s）` 无当前聚合器的开关/核数/指纹身份行——但索引把该报告标为「新口径」，14 份报告零旧口径警示。round22 修复把口径判定内建到生成器：`_caliber_findings`（`run_full_pipeline.py:261`）校验 `result_schema`、forward_steps 合批语义、链路归因字段集与推荐身份，不符即头部「旧口径警示」+ 推荐段「推荐未定」；14 份旧口径报告与索引静态补标注并重建，验证为门禁降级单测两例（先红后绿）与 r23-smoke 产物过门禁无警示（round23 实跑）。round25 再证「过门禁」不等于「数字对」——R25-01/02 揭示该产物本身的通信量与 TTFT 仍错（门禁只能校验已知口径维度，不能发现未入模的口径维度），处置为 `result_schema` 升 3（`gene_sim_scheduler.py:12983`）、b4g1 报告与索引行就地降级重标（头部补 R17 警示行，本次盘上核对在位）、门禁对当期逐槽策略档加豁免判据（`test_caliber_gate_exempts_deliberate_per_slot_tiers`，合批档异常仍降级）。盘上索引现 16 行全部「旧口径/推荐未定」，无「新口径」残留。

### 4.13 合批 prefill 的依赖传输量少算 B 倍（来源：round25 R25-01，严重）
　　为了让「合批 prefill 的通信工作量」与算子工作量同口径，依赖边计费必须消费与算子绑定相同的行数。缺陷形态是两套行数并存：算子绑定层在合批时把 `tq` 乘槽数（真实行数 B×S），而步级形状的 prefill 分支返回单槽 `S`——依赖传输、部署首次使用与 sidecar 集合通信边恰好消费后者，合批组的通信量比真实少 B 倍；普通依赖边与 all-gather/scatter 边共享同一入口，传输服务时间、链路占用与排队全部被污染，且四项守恒只能证明「已建模区间被完整分账」、拦截不了「入模工作量本身错」。round25 以真实 `_execute_pipeline` 动态复现：4 条请求×3 token、链路每字节 1 秒，B=1 总传输 12 字节/12 秒，B=4 合批组仅 3 字节/3 秒（预期同为 12 字节）。修复把「组行数、每槽 KV 长度」收敛为模块级单一真源 `closed_batch_binding`（`gene_sim_scheduler.py:317-326`），`step_shape`（`:9372-9401`）合批 prefill 返回组行数、逐槽步返回 S；依赖边计费（`:10471` 起）、部署首次使用（`:9944-9958`）、sidecar 集合通信边（`:7563-7575`）与张量形状解析 `tensor_bytes`（`:9657-9686`）全部消费同一绑定，系数求值收敛到 `evaluate_step_binding`（`src/ir/model_ir.py:27-54`，旧解析聚合委托它、slots=1 逐位等价）。修复没有全局放大的误伤：与 Tq 轴配对的项（块对角注意力）KV 轴保持每槽、不带 Tq 轴的 KV 项按槽数放大，两类口径各归其位。黄金值常驻 `TestPrefillTransferGroupBinding` 4 例（显式 Tq 边 B=1≡B=4 同 12 字节/12 秒、块对角 KV 72 字节、KV 项 48 字节、collective 合批/逐槽/B=1 同 48 字节；round26 实跑 8 例含此类全过）；`result_schema` 同步升 3，旧产物由门禁自动降级。

### 4.14 多槽 TTFT/TPOT 与单槽不同定义（来源：round25 R25-02，严重）
　　为了让 TTFT/TPOT 跨 B 可比，所有 B 必须在同一物理事件上记录首 token 时点。缺陷形态是定义分裂：多槽且 `output_length>1` 时仿真把 `ttft_step` 记在首个合并 decode 步完成时刻（prefill 之后又跑了一步才记 TTFT），B=1 则取第 0 步——而真实编排器在 prefill 返回后立即逐槽采样首 token（`runtime/executor.py:174-180`），仿真时点凭空多出一个 decode 步。round25 以固定 10 秒前向、`output_length=2` 的最小反例实证：B=1 得 TTFT=10s/TPOT=10s，B=4 得 TTFT=20s/TPOT=0s（零 TPOT 直接证明错位）；盘上 r23-smoke 的 `request_metrics.json` 同路径产出错位 TTFT（forward_steps=69、ttft=206.97s，本次直读核对），证明问题进入正式产物而非停留于夹具。修复统一时点：`ttft_step = prefill_step_count(request) − 1`（`gene_sim_scheduler.py:10935`），合批组在唯一 prefill 步完成时记录、逐槽组在最后一个逐槽 prefill 步完成时记录——后者与执行器「全部 B 个 prefill 步完成后统一采样」（`executor.py:137-161`）边界一致；TPOT 以该时点起算。设计文档 §4.6 其四的旧约定（「按首个合并 decode 步计 TTFT」）同步改写并在 §八.6 追加修正记录（design:156/282，本次直读核对）。黄金值常驻 `TestFirstTokenMetrics` 2 例（B=4 output=2 由 20/0 修正为 10/10、逐槽档 TTFT=40s，覆盖合批与逐槽两种 prefill 与 output=1/2/3），随 round26 实跑通过。

### 4.15 逐槽优化只接通 genesim，生产导出链未同步（来源：round25 R25-03，严重）
　　为了让「优化开关」模拟的执行策略与同次编译产物语义一致，逐槽策略必须贯穿生产图编译与主机编排，而不是只在仿真侧换执行方式。缺陷形态是断链三处：genesim 把 `pipeline_overlap or not weight_reuse` 折算为逐槽 prefill（`gene_sim_scheduler.py:9251`），主仓 `compile_llama2(per_slot_prefill=True)` 与执行器逐槽分支也都存在（单测直调通过），但生产链从未传该开关——`run_full_pipeline` 只把开关写进 genesim 配置，`export_pp_placement` 与 `export_placement_for_strategy` 的导出调用一律缺省 `False`，全仓检索确认 `per_slot_prefill=True` 只出现在测试直调；`PlacementExportResult` 也不交付主机计划。「有独立 API 且单测通过」掩盖了「优化全链路未接通」。修复把执行策略作为统一编译输入：main 在 `--pipeline-overlap`/`--weight-reuse` 时向导出传 `per_slot_prefill`（`run_full_pipeline.py:2312-2321`），CLI 透传 `--per-slot-prefill`（`:1111`），导出脚本校验 B 份逐槽主机计划真实生成、缺失即 `SystemExit`（`export_pp_placement.py:274-283`），`export_placement_for_strategy` 透传 `compile_llama2`（`placement_export.py:1039,1070-1077`，生产调用点唯一无旁路），主机计划经 `PlacementExportResult.compiled` 交付（`:983-986`）；sidecar 顶层 `per_slot_prefill` 三态声明（B>1 恒写、B=1 不写保逐位，`:744-748`），genesim 消费时与自身执行策略互检——逐槽策略消费合批声明直接抛、合批消费逐槽声明是保守方向放行（`gene_sim_scheduler.py:9252-9266`）。成对扫参两态共用一份带逐槽编译的 sidecar（容量只多算不少算）。生产入口测试 5 例常驻（导出交付 `test_cost_batch.py:544`、执行器消费 `test_executor_batch.py:464`、CLI 与 main 接线 `test_batch_sweep_report.py:1539/1565`、门禁豁免 `:1601`），round26 实跑全过。

## 五、改动说明

　　机制落到代码是哪些文件与函数、哪些结构变了，本章给出与工作区 `git diff` 一致的完整清单。

### 5.1 改动总览
　　本次改动全部以未提交工作区承载：pim-compiler 20 个跟踪文件（19 修改 + 1 删除）+2697/−257，加 6 个未跟踪测试文件（3132 行）；genesim 23 个跟踪文件 +2502/−474，加 2 个未跟踪文件（`tests/sim/test_closed_batch.py` 2666 行与 `scripts/check_test_main_order.py` 52 行）；FlagTree 零改动（符合 R5 预期）。核心改动集中在 `scripts/run_full_pipeline.py`（扫参、报告与口径门禁、执行策略接线）与 genesim `src/scheduler/gene_sim_scheduler.py`（组执行、步级绑定单一真源、计费与归因）。上述数字为本文档撰写时以 `git diff --numstat` 实际核验（本次实际验证）。

### 5.2 文件改动清单
　　pim-compiler 的全部改动文件见表 5。

<p align="center">表 5　pim-compiler 文件改动清单</p>  

| 文件路径 | 改动类型 | 改动内容（≤30字） | 对应需求 |
|----------|----------|-------------------|----------|
| `runtime/compile.py` | 修改 | `batch_size`、`[B,S]`/`[B,1]` 导出、state 按 B 槽、逐槽计划与副本 mem_plans | R1/R4/R8 |
| `memory/kv_layout.py` | 修改 | `num_slots` 槽位主序布局与按槽寻址 | R2 |
| `memory/mem_planner.py` | 修改 | 容量表、逐 DPU 墙位折算、tile 分解与第三图规划 | R3/R8 |
| `runtime/executor.py` | 修改 | `valid_lens` 批量 decode、逐槽执行与副本路由 | R4/R8 |
| `backend/hal_numpy.py` | 修改 | pos 按槽元组、DPU 副本窗口视图 | R4/R8 |
| `runtime/kernels.py` | 修改 | `sdpa` 逐槽偏移、读写与掩码 | R2/R4 |
| `genesim_bridge/placement_export.py` | 修改 | 逐相位产物（含 prefill_b{B}）、单行口径、容量/KV/副本载荷、钉扎 | R5/R8 |
| `genesim_bridge/refine_profile.py` | 修改 | 消费上限常量 `SIDECAR_SCHEMA_VERSION=6` | R5 |
| `graph/strategy.py` | 修改 | `ShardStrategy.replicas` 副本维度 | R8 |
| `contracts/partition_plan.py` | 修改 | `PartitionPlan.replicas` 物理映射 | R8 |
| `scripts/export_pp_placement.py` | 修改 | `--batch/--replicas/--kv-head-pin/--max-seq/--per-slot-prefill` | R1/R5/R8 |
| `scripts/run_full_pipeline.py` | 修改 | 扫参、报告聚合、口径门禁、复用对照、冻结落位、容量核验 | R1/R7/R8/R9 |
| `tests/assert_node_matches_ref.py` | 删除 | init 起空文件、无调用方 | R4 清理 |
| `CLAUDE.md` | 修改 | 对拍口径改为 pytest fixture 表述 | 文档 |
| `tests/test_executor_llama2_7b.py` | 修改 | 7B fixture B=2 用例改消费 `compiled.state` | R4 |
| `tests/test_kernels.py` | 修改 | sdpa 掩码与逐槽回归 | R2/R4 |
| `tests/test_kv_layout.py` | 修改 | 多槽布局断言更新 | R2 |
| `tests/test_placement_export.py` | 修改 | 桩修复、副本与单行口径用例 | R5/R8 |
| `tests/test_strategy.py` | 修改 | 副本维度 4 例 | R8 |
| `tests/test_tune_steps.py` | 修改 | fixture 版本同步 6 | R5 |
| `tests/test_compile_batch.py` | 新增 | B=1 逐位一致与 B 维形状用例 | R1 |
| `tests/test_kv_layout_batch.py` | 新增 | 槽数容量与寻址用例 | R2 |
| `tests/test_mem_planner_batch.py` | 新增 | 容量折算与墙位公式用例 | R3 |
| `tests/test_executor_batch.py` | 新增 | 批量执行、对拍、逐槽/副本与公开契约用例 | R4/R8 |
| `tests/test_cost_batch.py` | 新增 | 前置实验、sidecar 载荷与 prefill 跨层锚点用例 | R5/R8 |
| `tests/test_batch_sweep_report.py` | 新增 | 扫参、报告、门禁、落位与生产链接线用例 69 例 | R7/R8/R9 |

　　genesim 的全部改动文件见表 6。

<p align="center">表 6　genesim 文件改动清单</p>  

| 文件路径 | 改动类型 | 改动内容（≤30字） | 对应需求 |
|----------|----------|-------------------|----------|
| `src/ir/trace_ir.py` | 修改 | `Request.slots` 字段与校验 | R6 |
| `src/ir/model_ir.py` | 修改 | `Operator.kv_slots`（:140）、`evaluate_step_binding`（:27） | R6/R7 |
| `src/config_loader.py` | 修改 | 组大小、三布尔开关校验（含无效标志） | R6/R8 |
| `src/scheduler/gene_sim_scheduler.py` | 修改 | 组准入、合批 prefill、步级绑定单一真源、目标资源归因、路由 | R6/R7/R8 |
| `src/pim/capacity.py` | 修改 | 组容量 ×slots、稳定标记、KV 落账 | R6 |
| `src/pim/pimir_trace.py` | 修改 | zone 字节拆分与 KV 槽数乘数 | R6/R7 |
| `src/vpu/vpu.py` | 修改 | 相位变体查找、`zone_bytes`、缓存 | R6/R7 |
| `src/run_manifest.py` | 修改 | 相位产物收集、版本上限 6、DP 核验、输入冻结 | R5/R8 |
| `src/main.py` | 修改 | `batch_metrics.json` 落盘与清单冻结（:438-459） | R7 |
| `scripts/export_fixed_pu_mapping.py` | 修改 | `--replicas` 副本映射 | R8 |
| `tests/sim/test_closed_batch.py` | 新增 | 组执行、归因、门禁、传输绑定与首 token 黄金值 54 例 | R6/R7/R8 |
| `tests/sim/test_model_ir.py` | 修改 | 步级绑定求值用例（round25 新增） | R6/R7 |
| `scripts/check_test_main_order.py` | 新增 | main 块序扫描固化检查 | 测试纪律 |
| `tests/sim/test_vpu.py` | 修改 | 分量归一契约与缓存回归用例 | R6/R7 |
| `tests/sim/test_pim_capacity.py` | 修改 | 容量指标可序列化与 KV 向量用例 | R6/R7 |
| `tests/sim/test_run_manifest.py` | 修改 | 相位产物、DP 核验与冻结三态用例 | R5/R8 |
| `tests/sim/test_config_loader.py` | 修改 | 组大小与三开关校验用例 | R6/R8 |
| `tests/sim/test_export_fixed_pu_mapping.py` | 修改 | 副本映射 3 例与 main 块修正 | R8 |
| `tests/sim/test_collective_edges.py` | 修改 | main 块移至文件尾 | 测试纪律 |
| `tests/sim/test_cost_sidecar.py` | 修改 | main 块移至文件尾 | 测试纪律 |
| `tests/sim/test_gpu_backend.py` | 修改 | main 块移至文件尾 | 测试纪律 |
| `tests/sim/test_compiler_placement.py` | 修改 | fixture 版本同步 6 | R5 |
| `tests/sim/test_main.py` | 修改 | fixture 版本同步 6 | R5 |
| `tests/sim/test_refine_profile.py` | 修改 | fixture 版本同步 6 | R5 |
| `tests/sim/test_tune_placement_audit.py` | 修改 | fixture 版本同步 6 | R5 |

### 5.3 核心函数清单
　　理解本次改动所需的核心函数见表 7。

<p align="center">表 7　核心函数清单</p>  

| 函数/类 | 位置 | 改动类型 | 作用（≤30字） | 调用方 |
|---------|------|----------|---------------|--------|
| `compile_llama2()` | `runtime/compile.py:247` | 修改 | 编译入口透传 batch、槽数、state、逐槽与副本 | 导出脚本、测试 |
| `kv_bytes()` / `build_kv_layout()` | `memory/kv_layout.py:62,69` | 修改 | 槽位主序布局与容量乘槽数 | 规划与导出链 |
| `capacity_batch_fit()` | `memory/mem_planner.py:699` | 新增 | 逐 DPU 容量折算与墙位 | sidecar 容量载荷 |
| `run_decode_loop()` | `runtime/executor.py:102` | 修改 | 批量 prefill+decode、逐槽步与逐槽采样 | 执行器入口 |
| `execute_plan()` / `replica_dpu_window()` | `runtime/executor.py:38,70` | 修改 | 计划执行与副本物理块窗口路由 | `run_decode_loop` |
| `sdpa_kernel()` | `runtime/kernels.py:1335` | 修改 | 逐槽 KV 读写与掩码 | HAL 执行 |
| `export_placement_to_genesim()` | `genesim_bridge/placement_export.py:640` | 修改 | 逐相位逐档产物与 sidecar 载荷 | `export_placement_for_strategy()` |
| `_collective_entry()` | `genesim_bridge/placement_export.py:138` | 修改 | 通信边按 B_export 除回单行口径 | 导出写出循环 |
| `_form_closed_batch_groups()` | genesim `gene_sim_scheduler.py:8940` | 新增 | 等长请求静态预分组成组 | 主循环准入 |
| `closed_batch_binding()` | genesim `gene_sim_scheduler.py:317` | 新增 | 步级绑定单一真源（组行数、每槽 KV） | 算子绑定与传输计费 |
| `prefill_step_count()` / `step_shape()` | genesim `gene_sim_scheduler.py:9322,9372` | 修改 | 合批/逐槽步链与步级形状绑定 | 任务绑定、依赖计费 |
| `evaluate_step_binding()` | genesim `src/ir/model_ir.py:27` | 新增 | 紧凑系数按步级绑定求值 | 依赖边/集合通信/首次使用 |
| `resolve_phase_template()` | genesim `src/pim/pimir_trace.py:1417` | 修改 | KV 操作数字节按槽数放大 | 轨迹静态度量 |
| `freeze_manifest_inputs()` / `verify_frozen_manifest_inputs()` | genesim `src/run_manifest.py:1297,1339` | 新增 | 输入内容寻址冻结与复验 | `main.py`、验证脚本 |
| `_caliber_findings()` | `scripts/run_full_pipeline.py:261` | 新增 | 报告口径门禁（旧语义降级标注） | `aggregate_batch_report()` |
| `relocate_manifest_to_tier()` | `scripts/run_full_pipeline.py:1339` | 新增 | 结果目录冻结落位与哈希复验 | `step_e_simulate()` |
| `run_batch_sweep()` | `scripts/run_full_pipeline.py:1920` | 修改 | 扫参循环、成对对照与报告聚合入口 | CLI |
| `sweep_scope_dimensions()` | `scripts/run_full_pipeline.py:1749` | 新增 | 扫参口径维度集单一真源 | 报告命名与覆盖行 |

### 5.4 数据结构与接口变更
　　为了表达「batch 槽数」这一贯穿两仓的关键信息，已有结构的 offset 键只有 `(layer, head, which)` 三元组、容量公式无槽数项，无法表达多槽 KV 的物理布局与寻址。因此 `KVRegionSpec` 增加槽数字段并让布局、容量、寻址三处同步感知。

**`KVRegionSpec`**（`memory/kv_layout.py:40`）
```python
num_slots: int = 1        # batch 槽数；每槽一份完整 KV，槽位主序连续。

def kv_bytes(spec):
    return (2 * len(spec.layers) * spec.max_seq
            * len(spec.kv_heads) * spec.head_dim
            * spec.dtype_bytes * spec.num_slots)
```
缺省 1 等价改动前行为；`kv_off` 键扩为 `(slot, layer, head, which)`，`kv_access`/`update`/`read_tile` 加 `slot` 缺省 0 参数，既有调用方无需改动。

　　为了让 genesim 按相位选取真实产物且口径可检测，sidecar 算子条目原只有单一 `pimir_path`，无法区分 prefill 与逐 B decode 档，也无法声明通信字节口径。因此条目扩为相位索引结构并以版本字段固化契约，顶层旧字段保留为兼容副本。

**sidecar 算子条目 `pimir_by_phase` 与版本契约**（`genesim_bridge/placement_export.py:934`，版本写入 `:711`）
```python
entry["pimir_by_phase"] = {
    "prefill":   {"path": ..., "sha256": ..., "kernel_tile_n": ..., "tile_m": ...},
    "prefill_b1": {...}, "prefill_b4": {...},   # M=B×S 合批产物，prefill 键为 prefill_b1 同值副本
    "decode_b1": {...}, "decode_b4": {...},
}
# 顶层旧字段 pimir_path/kernel_tile_n = decode_b1 同值副本
sidecar["version"] = 6   # v6 起 collectives 为单行口径（K6 修复）
```
缺省单档（`--batch 1`）时结构与旧版消费方完全兼容；sidecar 另增顶层 `capacity`（含 `max_seq`/`slots`/墙位）、`kv_layout`（逐 DPU 物理布局）、`dp_replicas`（副本声明，纯权重与内核 tile 分列）三个载荷。genesim 装载入口对带 `collectives` 的低版本产物直接抛，无 collectives 的旧产物维持 legacy 兼容路径。

　　为了让组语义进入仿真器运行时、让编译结果的运行时状态成为公开契约，`Request` 原无槽数概念、调度配置原把组大小当禁令参数、`compile_llama2` 原返回单槽状态。因此请求对象与编译结果各增加槽数语义，配置项语义从「拒绝大于 1」变为「接受并校验前提」。

**genesim 组请求与配置、编译结果状态**（`src/ir/trace_ir.py:59`、`src/config_loader.py:1398`、`runtime/compile.py:363`）
```python
slots: int = 1            # Request 组槽数，缺省 1 等价单请求

execution_microbatch_size = self.get("scheduler.execution_microbatch_size")
# 正整数校验；>1 时组准入再校验 trace 等长、等输出、整倍数，违反直接抛

state = DecodeState(valid_lens=(0,) * batch_size)   # 编译结果公开状态按 B 槽
```
`slots` 与 `state` 的 B=1 形态（`(0,)`）均与旧单槽语义等价，B=1 路径不经组准入分支、legacy 行为不变。新增配置键 `scheduler.pipeline_overlap`（缺省 False）、`scheduler.weight_reuse`（缺省 True，等于合批现状）、`scheduler.dp_replicas`（缺省 1）同理，缺省不写即配置逐字节不变。

　　为了让流水与 DP 的两态策略在主机侧可执行，`CompiledModel` 原只有合批计划与单一 mem_plans 集合，无法表达逐槽步与副本物理块。因此编译结果扩展逐槽计划与副本视图两个公开接口，执行层增加窗口路由。

**主机侧逐槽与副本接口**（`runtime/compile.py:45,52`、`runtime/executor.py:38,70`、`backend/hal_numpy.py:94,139`）
```python
per_slot_prefill: tuple[CompiledPlan, ...] | None = None   # B 份槽位平移 [1,S] 计划
def mem_plans_of_replica(self, replica: int) -> dict[int, DPUPlan]: ...

def execute_plan(plan, hal, *, values=None, pos=None, dpu_window: int = 0): ...
def replica_dpu_window(replica: int, dpus_per_replica: int) -> int: ...
```
`per_slot_prefill` 缺省 None、`dpu_window` 缺省 0、`replicas` 缺省 1，均等价改动前行为；窗口在 `execute_plan` 的 finally 中复位 0，防止计划外直写被窗口劫持。

　　为了让逐槽执行策略与同次编译产物语义一致（R25-03），执行策略原只存在于 genesim 配置、导出链无从感知，导致优化档的仿真策略与图编译/主机产物脱节。因此执行策略升为统一编译输入并进入产物契约。

**执行策略编译输入与 sidecar 策略声明**（`run_full_pipeline.py:2312-2321`、`placement_export.py:744-748,983-986,1039`）
```python
per_slot_prefill=bool(args.pipeline_overlap or args.weight_reuse)  # main 折算
sidecar["per_slot_prefill"] = bool(per_slot_prefill)   # 仅 max(batch_sizes) > 1 时写
compiled: "object | None" = None    # PlacementExportResult 携带主机编译结果
```
缺省（两开关均关）时 `per_slot_prefill=False`，经典路径产物逐字节不变；B=1 不写该键，genesim 侧对缺键旧产物维持兼容。导出脚本对 `--per-slot-prefill` 开启而 B 份逐槽计划未生成的情况直接 `SystemExit`（`export_pp_placement.py:274-283`），消费方无需改动、错误在导出端暴露。

## 六、验证

　　改动正确性最终要靠可复现的证据说话，本章给出验证环境、命令与结果、验收标准对照和二十四轮评审的闭环情况。

### 6.1 验证环境
代码基线：pim-compiler `main@1ec124f` + 未提交工作区改动、genesim `fjg-dev@e7c2020` + 未提交工作区改动（两仓即表 5/表 6 所列状态）。本文档撰写时以 mtime 核验：round25 修复终态（15:29–15:44，decisions.tsv 逐条登记）之后、经 round26 评审（21:45–22:05）至当前，两仓除 `docs/decisions.tsv` 评审登记行外再无任何源码/测试修改（`find -newer` 全量扫描，本次实际验证），工作区与 round26 复核终态同态；round25 终跑（L0 2343、genesim 1643，日志 `final-runs/r25_l0_final.log` 与 genesim `test-results/genesim_r25_full.log`，15:35–15:37 落盘、EXIT=0，本次直读核对）在该代码态上执行，round26 实跑的黄金值与生产入口用例同态适用。本次实际验证为双锚点（`test_batch_sweep_report` 69 例 4.29s、genesim 官方入口 `closed_batch` Ran 54 tests OK）与盘上产物核对；未重跑任何全量回归、7B 全量或扫参仿真。运行环境：torch 2.9.1 / transformers 4.57.6 / python 3.10.20，经 `paths.json` 指向的 `env-pytorch.sh` 激活；genesim 官方入口 `./run.sh`。

### 6.2 验证命令与结果
　　为了验证「多 batch 全链路正确且数字可信」，本文采用分层验证方法。首先通过快速回归与 7B 对拍确认既有功能未受影响且 B>1 数值与 PyTorch 对齐；然后围绕组执行、合批计价、传输绑定与生产链贯通核对官方入口、全量套件、定向套件和跨层锚点；其次通过单档端到端受控对照确认编译期产物结构正确、仿真出数、守恒闭合且冻结落位生效；最后以盘上交付物核对锚定证据链（盘上全部产物为旧口径历史证据，新口径重跑按用户终裁登记不执行）。各验证项的命令与证据状态见表 8；证据分三档标注，本文档撰写过程中执行的验证为双锚点测试与盘上核对（行 5、11、12），未重跑任何全量回归、7B 全量或扫参仿真。

<p align="center">表 8　验证命令与结果</p>  

| 验证项 | 命令 | 结果 | 结论 | 证据状态、来源、基线与环境 |
|--------|------|------|------|----------------------------|
| pim-compiler L0 快速回归 | `python -m pytest tests/ -q -k "not llama2_7b"` | 2343 passed / 9 skipped / 0 failed（193.65s） | 通过 | 复用已有验证；round25 修复轮终跑（全部代码修改完成后单次后台执行），日志固化 `test-results/final-runs/r25_l0_final.log`（15:37 落盘、EXIT=0，本次直读核对）；其后 pim 侧零源码变更（mtime 全量扫描），适用性成立 |
| pim-compiler 7B 全量 | `python -m pytest tests/ -q -k "llama2_7b" -n 8` | 47 passed / 0 failed（44.8 分钟） | 通过 | 复用已有验证；11-22 Q8 基线（implement §4.3）；其后触碰 7B 文件的改动均以 B=2 锚点单独复核（行 3） |
| 7B B=2 执行器对拍 | `python -m pytest tests/test_executor_llama2_7b.py::test_b2_prefill_and_decode_match_torch -q` | 1 passed（约 11 分钟；prefill 与每步 decode logits 过 max<0.2/mean<0.02 容差） | 通过 | 复用已有验证；round23 评审实跑——round22 改动 compile/executor/hal_numpy/mem_planner 后该锚点首次复跑，缺省路径数值与单卡 PyTorch 仍对拍一致；round25 改动不改缺省执行路径（per_slot 缺省关） |
| genesim 全量 | `python -m pytest tests/ -q --ignore=tests/predictor --ignore=tests/perf` | 1643 passed / 1 skipped / 0 failed（98.38s） | 通过 | 复用已有验证；round25 修复轮终跑，日志 genesim `test-results/genesim_r25_full.log`（15:35 落盘、EXIT=0，本次直读核对）；其后 genesim 零源码变更，适用性成立 |
| genesim 官方入口 | `./run.sh --test closed_batch` | Ran 54 tests in 3.605s, OK（含合批块对角、覆盖反例黄金值、9s 黄金值、全局对账、抢占分项、weight_reuse×3、流水两态 120/90、传输绑定黄金值×4、首 token 黄金值×2、策略互检） | 通过 | 本次实际验证；genesim 工作区（`fjg-dev@e7c2020`+未提交改动，round26 终态）；与 round25/26 评审锚点同款（round26 亦实跑 Ran 54 OK） |
| R25 黄金值与生产入口用例 | `pytest tests/sim/test_closed_batch.py::TestPrefillTransferGroupBinding TestFirstTokenMetrics TestPerSlotStrategySidecarGate`；`pytest test_cost_batch.py:544 test_executor_batch.py:464 test_batch_sweep_report.py:1539/1565/1601` | 黄金值 8 passed（3.85s）；生产入口 5 passed（24.37s） | 通过 | 复用已有验证；round26 评审实跑（基因与当前工作区同态）；修复前同款探针 FAILED（failures=6, errors=1），round25 修复后转绿（implement §7.3） |
| 跨层锚点套件 | `python -m pytest tests/test_cost_batch.py -q` | 10 例全过（含 prefill 合批跨层锚点 B=2/4×S=17、sidecar 相位条目、逐槽计划交付） | 通过 | 复用已有验证；round25 受影响子集 172 例与 r25 L0 内通过；单文件 10 例计数本次静态核对 |
| 主机批量执行对拍与受影响子集 | `python -m pytest tests/test_executor_batch.py tests/test_compile_batch.py -q`（round25 扩为 7 文件受影响子集） | round23 实跑 25 passed；round25 修复后受影响子集 172 passed（含新增 5 例生产入口） | 通过 | 复用已有验证；round23/round25 实跑（implement §7.3 round25 行） |
| KV/内存/成本/导出定向 | `python -m pytest tests/test_mem_planner_batch.py tests/test_kv_layout_batch.py tests/test_cost_batch.py tests/test_placement_export.py -q` | round23 实跑 87 passed；round25 子集复跑含前两文件全过 | 通过 | 复用已有验证；round23/round25 实跑（同上基线） |
| b4g1 单档端到端受控对照 | `python scripts/run_full_pipeline.py --num-stages 4 --batch 4 --groups 1 --max-requests 4 --log-dir test-results/full-pipeline-r23-smoke` | EXIT=0（约 5 分钟）：total_time_s=247.941 与 r19-smoke 逐位一致；prefill 恰合批 1 步（forward_steps=69）、executed_batch=4、四项守恒 53.895+194.046+0+0=247.941、KV zone 6.056e10=4×单槽、冻结落位生效（inputs 3 条 + pim_mlir 36 份全指本档） | 通过（机制层） | 复用已有验证；round23 评审实跑（round22 终态）。该产物 `result_schema=2`：其通信量与 TTFT/TPOT 数字被 R25-01/02 修复判为旧口径，已就地降级标注（见行 11）；步数结构、守恒、KV 倍数与冻结落位等机制层结论不受影响，新口径端到端数字待 R17 重扫 |
| 盘上交付物与实现核对 | ——（非测试命令） | r23-smoke 产物在位（forward_steps=69、冻结 inputs/pim_mlir 指本档，request_metrics 的 ttft=206.97s 为错位时点产物，本次直读）；索引 16 行全部标「旧口径/推荐未定」（b4g1 行指向 R17，无「新口径」残留）；report-b4g1 头部带 R17 警示行；final-runs 日志含 r25 系列两份终跑；两仓 diff 统计与本文全部行号核验无误 | 通过 | 本次实际验证（产物与实现核对，非测试运行）；本仓与 genesim 工作区，2026-11-25 |
| 报告与门禁单测（当前态） | `python -m pytest tests/test_batch_sweep_report.py -q` | 69 passed（4.29s；含 AST 不可达扫描、口径门禁降级、复用对照、冻结落位、生产链接线×3、门禁豁免） | 通过 | 本次实际验证；round26 亦实跑 69 passed（4.20s，闭合 15:34 编辑窗口） |
| genesim config_loader 官方入口（当前态） | `./run.sh --test config_loader` | Ran 117 tests, OK（含 pipeline_overlap/weight_reuse 非布尔拒绝子用例） | 通过 | 复用已有验证；round24 修复轮实跑（校验标志补齐后先红后绿）；round25 未触碰该校验块 |
| 快速端到端（CEVA/L1/冒烟） | `python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out`；`python scripts/run_full_pipeline.py --num-stages 4`（及 `--batch 4 --max-requests 8`）；genesim `./run_all.sh` | 均通过；经典 L1 total 1079.69s（−29.8% 为已确认必修项，B=1 时合批 M=B×S=S 与逐槽 M=S 数值一致）；genesim 冒烟 SUCCESS（10/10 请求，1040.41s） | 通过 | 复用已有验证；implement §4.3 |
| 全档 18 档扫参与专项曲线（旧口径） | 六条 `--batch 1,2,4,8,16,32 --groups <G> --sim-batches … --max-requests 32` 命令（产物根 `test-results/full-pipeline-final2/`）与六条专项（`full-pipeline-r16-{b,g,ov,sat,dp,khp}`） | 六条命令全链 EXIT=0、18/18 档；四点核验（KV zone 严格 B 倍、组内争用单调、容量零溢出、tile_m 封顶）通过；跨全档最优 B=4,G=4（1.74 tokens/s） | 通过（旧口径历史证据） | 复用已有验证；round16 收口轮实跑，日志 `final-runs/r16_cmd_*.out`；该批产物为 prefill 合批、目标资源归因与传输计价/首 token 口径修复前的旧口径，绝对值失真已声明（机制结论不受影响），数字以实施文档第八节 R9 登记条目为准；15+1 份报告已就地标注旧口径警示（round22 静态修复 + round25 b4g1 重标，本次盘上核对）——本行结论指命令当时实际执行且四点核验通过，不构成新口径数字的验收 |
| 18 档新口径重扫、全档报告重建、专项重跑、wr 成对曲线、B=1 G>1 档重出 | `python scripts/run_full_pipeline.py --num-stages 4 --batch 1,2,4,8,16,32 --groups <G> … --log-dir test-results/full-pipeline-final3` 等（实施文档第八节 R9/R10/R12–R16） | 未执行 | 未执行 | 按用户 2026-11-25 终裁登记不执行（decisions.tsv 09:43:39「收口轮改为静态修复，不执行重跑」）；round25 新增 R17（传输计价与首 token 口径重扫）同属登记边界；实施文档所载命令为示例，不作为已执行证据；重跑前「最优配置推荐视为未定」为终态表述 |

复现命令（按验证层级排列；前段为历史已实测命令的原文，末段为登记不执行的示例命令——示例即使可复制运行，也不得据此标记通过；旧口径 18 档六条命令的原文见实施文档第八节 R9 登记条目）：
```bash
# 环境（每个新 shell 先激活）
source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh

# L0 快速回归（约 3 分钟）
python -m pytest tests/ -q -k "not llama2_7b"

# 7B 全量与 B=2 锚点（小时级，仅阶段末执行）
python -m pytest tests/ -q -k "llama2_7b" -n 8
python -m pytest tests/test_executor_llama2_7b.py::test_b2_prefill_and_decode_match_torch -q

# 双锚点与定向套件（分钟级）
python -m pytest tests/test_cost_batch.py -q
python -m pytest tests/test_executor_batch.py tests/test_compile_batch.py -q
python -m pytest tests/test_batch_sweep_report.py -q

# R25 黄金值与生产入口用例（round26 实跑；genesim 仓内执行）
python -m pytest tests/sim/test_closed_batch.py::TestPrefillTransferGroupBinding \
  tests/sim/test_closed_batch.py::TestFirstTokenMetrics \
  tests/sim/test_closed_batch.py::TestPerSlotStrategySidecarGate -q

# CEVA 厂商路径 B=1 快速验证与单档端到端
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out
python scripts/run_full_pipeline.py --num-stages 4
python scripts/run_full_pipeline.py --num-stages 4 --batch 4 --max-requests 8

# 单档端到端受控对照（约 5 分钟；round23 已实测，产物存档 full-pipeline-r23-smoke；
# 该产物的通信量/TTFT 数字已被 round25 修复判为旧口径并降级标注，机制层结论有效）
python scripts/run_full_pipeline.py --num-stages 4 --batch 4 --groups 1 \
  --max-requests 4 --log-dir test-results/full-pipeline-r23-smoke

# genesim 全量与官方入口
cd /data/fjg/flagOS/genesim && source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -q --ignore=tests/predictor --ignore=tests/perf
./run.sh --test closed_batch
./run.sh --test config_loader
./run_all.sh

# ——以下为登记不执行的示例（按 2026-11-25 终裁静态收口，未实测）——
# 18 档新口径全量重扫（六条命令分批等价完成单条 --sweep；小时级）
python scripts/run_full_pipeline.py --num-stages 4 --batch 1,2,4,8,16,32 --groups 1 \
  --sim-batches 1 --max-requests 32 --log-dir test-results/full-pipeline-final3
# （G∈{2,4} 与 --sim-batches 2,4,8,16,32 组合、含 --per-slot-prefill 的专项档
#   同实施文档第八节 R9/R17 清单）
```

### 6.3 验收标准对照
　　逐条对照需求文档验收标准（request §五）的结果见表 9。

<p align="center">表 9　验收标准对照</p>  

| 验收标准（来自需求文档） | 验证方法 | 结论 | 证据 |
|--------------------------|----------|------|------|
| L0 快速回归全绿且 B=1 产物不变 | 表 8 行 1；B=1 锁定用例 | 满足 | 2343/0 failed；`test_b1_export_identical` 等 |
| 快速端到端（export_gml、双档 run_full_pipeline） | 表 8 行 14 | 满足 | 均通过 |
| 逐项验收 R1–R9 | 表 1 | 满足（R1–R8；R9 机制层满足、全档推荐为登记边界 L7） | 各功能点均有代码与用例 |
| B=1 全链路逐位回归不变 | L1 total −29.8% 为已确认必修项（prefill M=S；B=1 时合批 M=B×S=S 数值一致）；K6 修复对 B=1 导出零变化；decode 与组机制逐位不变 | 满足（按确认口径） | 用户 2026-11-21 确认（Q4）；closed_batch 54 例 B=1 门禁 |
| B>1 与 PyTorch 逐元素对齐 | 小型 fixture B=2/4 logits 容差对拍（含 `compiled.state` 公开契约、逐槽/副本）+ 7B B=2 | 满足 | 表 8 行 3、8；容差 max<0.2/mean<0.02 |
| decode 权重字节/token 趋于 1/min(B,tile_m) | 旧口径 B 曲线界列 + 合批装载黄金值断言 | 满足（机制层；全档曲线待 L7 登记边界） | 旧口径 b1–b8 逐档恰压线（1.000/0.500/0.250/0.125）；`test_b4_weight_per_token`（合批 6 份装载断言） |
| KV 容量墙位置与公式手推一致 | 容量表墙位与手推 | 满足 | b4g1 冒烟 B≥976（复用 round19/23）；旧口径全档 975（复用 round16） |
| 吞吐曲线形态符合分档分析 | 旧口径全档 18 档核验（机制结论不受口径影响） | 满足（旧口径形态证据） | G=1 随 B 增速递减至持平、G 维收益集中 B=1–8、组数不足时 G 档退化已标注 |
| 不回归判据（容量硬失败、无兜底、真实产物计价） | 代码与用例 | 满足 | `PIM_CAPACITY_EXCEEDED` 硬失败；容量核验缺字段硬失败；无 try/except 兜底；结论均出自真实产物与 sidecar（v6 门禁、result_schema=3 口径锚点） |

### 6.4 评审情况
　　本需求共进行 26 轮评审，形成 82 条问题条目（按实施文档 7.1 最终台账，含跨轮拆分与承接复核；round04 的 10 条并入 round05 清单复核，round07 的 5 条因 API 过载未修复、由 round08 以同集重验承接，round12/13 的 4 条严重问题钉住后修复被终止、由 round14 承接修复，round17 的 8 条由 round18 承接复核修复，round19 的 3 条因修复进程被用户终裁终止、由 round20 承接，round21 的 6 条由 round22 承接修复，round23 的 4 条因修复进程死于网关瞬断、由 round24 承接修复；round25 为循环退出后的外部补充评审，其 3 条严重问题由驱动脚本按循环同机制补跑修复、round26 复核确认）。其中严重 17 条、高 14 条、中 30 条、低 21 条；1 条经复核判为误判（round04 问题 6「zone 归因 KV 恒 0%」，round05 纠正），其余全部修复或按二选一取舍关闭。各轮核心问题及最终状态见表 10。

<p align="center">表 10　评审轮次与修复状态</p>  

| 轮次 | 核心问题 | 修复状态 | 最终证据 |
|------|----------|----------|----------|
| round01 | 容量墙判类失配、容量表未接线、抢占带宽缺省、对拍口径弱化等 9 条 | 全部已修复 | `PIM_CAPACITY_EXCEEDED` 标记、容量五处消费点、对拍参数化用例 |
| round02 | 全墙终态聚合崩溃、相位轨迹软回退、档位解析双入口 | 全部已修复 | 空 runs 契约用例、缺档硬抛、`parse_batch_list` 单点 |
| round03 | 组级 zone/DMA 归因缺失、容量墙丢 G 维、抢占分子口径 | 全部已修复 | `test_group_dma_zone_attribution`、(B,G) 墙记录、迁移口径用例 |
| round04 | R8 未实现、代价侧恒零、容量墙 64 折算等 10 条 | 并入 round05 修复；1 条误判纠正 | 见 round05/06 行 |
| round05 | KV 计费不随 B 放大、四项守恒、max_seq 同源、机制饱和等 12 条 | 全部已修复 | 跨层锚点用例先红后绿；产物 KV zone 严格 B 倍 |
| round06 | DP 切片未动工、新口径产物缺失、流水未接扫参 | 已修复（跨轮切片） | `--replicas` 全链、`--pipeline-overlap` 两态、Q10 修复后产物落盘 |
| round07 | 新口径产物缺失、DP 用例官方入口漏跑、DP 资源指标失真等 5 条 | 修复未执行，由 round08 承接 | 见 round08 行 |
| round08 | 同集重验并扩大（main 块序第二文件）、Q10 剖析收窄 | 全部已修复 | 官方入口 Ran 30/27、`check_test_main_order.py` 固化、42min+→2m52s |
| round09 | DP 契约失配、报告引用旧世代、文档预写完成态、去重瑕疵 | 全部已修复（含 I25） | `--replicas 2` 端到端跑通、指纹机制、文档回写 |
| round10 | 全档未跑、终跑数字未回填、口径维度集漏钉扎 | 全部已修复 | 18 档单一世代 + `report-full-grid.md`、final-runs 日志固化 |
| round11 | 抢占段重复行、证据指针与时长失真、khp 表头措辞 | 全部已修复 | 报告去重、实测时长回填、口径二分表述 |
| round12 | 两条严重钉住：`compiled.state` 单槽断链、K6 仿真不变性 | 由 round14 修复 | 公开入口复现（B=2 prefill 前抛错）；受控对照 wall 差 16.5% |
| round13 | K6 根因定位与定量闭环（通信字节对账 91.845s 全额解释） | 修复被终止，由 round14 承接 | k6r 受控对逐位复现、通信份额 2.0 倍分解 |
| round14 | state 按 B 槽、collectives 单行口径、文档回写与收口清单 | 全部已修复 | 公开契约用例先红后绿；k6r2 受控对逐位一致；导出档无关性单测 |
| round15 | sidecar 版本门禁（5→6）、旧口径警示、三件套记录门槛；0 严重 0 高触发收口轮 | 全部已修复 | 版本三处同步 + 装载门禁三用例；冒烟与 k6r2 wall 逐位一致证零数值影响 |
| round16 | 索引口径标注生成器化、验证记录检查点纪律；收口轮 R1–R8 执行完毕（旧口径重扫） | 全部已修复 | 索引口径列先红后绿；R1–R8 全部完成（产物后被 round18/22 口径修复判为旧口径） |
| round17 | 推翻 round16「全部完成」：prefill 三层语义不一致、链路等待入 sync、证据链不可复现、DP 静态/动态混写、抢占半程等 8 条（严重 2/高 3/中 3） | 修复未执行，由 round18 承接 | 真实算子编译证伪合批等式（S=17/tile_m=8 合批 9.107 vs 逐槽 12.155 份）；用户 2026-11-25 裁决合批方向 |
| round18 | round17 八条复核成立 + 新增文档回写 1 条，共 9 条根因级修复 | 全部已修复 | 反例沉淀为常驻测试：跨层锚点/合批计价/块对角/9s 黄金值/冻结三态/容量四态/推荐身份/末档饱和/抢占分项 |
| round19 | 0 严重 0 高 2 中 1 低；修复进程 09:43:39 按终裁终止（exit 143），三条由 round20 承接 | 已修复（由 round20） | b4g1 单档端到端受控对照 EXIT=0（评审代跑）；genesim 全量 1629 复跑覆盖终态 |
| round20 | 静态收口轮：3 中 1 低（口径判据清零、三件套补登记、收口计划终态表述、b4g1 销账），0 严重 0 高；R9–R11 按终裁登记不执行 | 全部已修复 | grep 机械判据（`final-runs/r20_doc_static_closure.log`）；双锚点复核 |
| round21 | 推翻 round20「五目标全部闭合」：归因代理判据（9s 中 3s 落 sync）、优化仅仿真侧可执行、旧口径错标「新口径」等 6 条（严重 3/高 1/中 2） | 修复未执行，由 round22 承接 | 最小事件反例实跑；final2 forward_steps=72 vs r19 69 产物直证；M>1 真实编译表 |
| round22 | R21 六条复核成立 + 新立文档回写 1 条，共 7 条根因级修复（目标资源归因口径、主机最小可执行、报告口径门禁、`--weight-reuse` 开关、B=1 多组容量同源、冻结落位、回写条款） | 全部已修复 | 覆盖反例黄金值×2、门禁降级单测×2、L0 2337/genesim 1634 终跑、14+1 份报告静态标注（`r22_static_fix.log`） |
| round23 | 复核 round22 修复全部根因级并实跑 r23-smoke 端到端与 7B B=2 锚点复跑；新立 2 中 2 低（设计文档回写缺口、死代码、校验标志、编号笔误） | 修复进程死于网关瞬断，由 round24 承接 | r23-smoke 数值与 r19 逐位一致、25/65/87/46 例全过、收口条件成立 |
| round24 | R23 四条复核成立 + R21 六条复发抽验零复发；0 严重 0 高，静态收口轮 | 全部已修复（静态收口） | `test_batch_sweep_report` 66 例、`config_loader` 117 例、设计文档 §4.7/§4.8/§八.6 回写（`r24_static_fix.log`） |
| round25 | 循环退出后的外部补充评审，推翻 round24「五目标已完成」：合批 prefill 依赖传输少算 B 倍、多槽 TTFT/TPOT 定义错位、逐槽优化未贯穿生产导出链 3 条严重（动态反例实跑复现） | 全部已修复（驱动脚本按循环同机制补跑，15:30–15:44 登记） | 单一真源/首 token 时点/统一编译输入根因级落地；黄金值与生产入口 13 例先红（failures=6, errors=1）后绿；L0 2343/genesim 1643 终跑；b4g1 产物降级重标、result_schema 升 3、R17 登记 |
| round26 | R25 三条修复根因级复核 + R21/R22/R23 修复复发抽验零复发；0 严重 0 高 0 中 0 低，静态收口条件重新满足 | 无待修复项 | 黄金值 8 例与生产入口 5 例实跑全过；`test_batch_sweep_report` 69 例、closed_batch 官方入口 Ran 54 OK；盘上索引 16 行全旧口径、无「新口径」残留；五目标机制层成立、产物层维持 R17 登记边界 |

　　当前未修复 0 条；唯一未关闭项 Q7 为甲方 SLO 外部依赖（见第八节）。K6、prefill 口径、目标资源归因、传输计费与首 token 时点均已根因修复并闭环：单行口径、合批口径、目标资源交段口径、步级绑定单一真源与统一首 token 时点落地，反例全部沉淀为常驻黄金值（round26 复核零复发）；受影响的 18 档重产物（含 b4g1 单档）按用户 2026-11-25 终裁登记不执行（R9/R10/R12–R17 在册，重跑前最优推荐视为未定为终态）。跨轮重复问题已按最终结果合并核对。

## 七、与设计的差异

　　本节记录最终实现与设计文档的出入及原因。设计文档已随 round18/round20/round24/round25 修复轮按用户裁决与代码终态回写（prefill 合批口径、链路份额与目标资源归因、抢占三分项、`--weight-reuse` 开关与主机侧最小可执行边界、步级绑定单一真源、首 token 时点与执行策略统一编译输入），逐条对照设计功能点的结果见表 11；其中六项为用户确认的取舍，其余为评审驱动的口径修正。

<p align="center">表 11　与设计的差异</p>  

| 设计条目 | 设计方案（≤20字） | 最终实现（≤20字） | 差异原因（≤30字） | 是否经确认 |
|----------|-------------------|-------------------|-------------------|------------|
| §6.2 任务 worktree | 任务级 worktree | 两仓主工作区直改 | 编译缓存绝对路径会失效 | 否（T1 级可逆偏差） |
| §4.4 数值对齐 fixture | 7B fixture 模式 | 小型 fixture + 7B B=2 用例（L2） | 避免 L0 拖慢；7B 规模由 L2 关闭 | 是（round05 问题 9 补齐） |
| §4.6 组准入机制 | 运行时准入层凑组 | FCFS 静态预分组 | 到达序确定下等价，侵入最小 | 是（用户 2026-11-21 确认） |
| §4.5 B=1 成本逐位不变 | 逐位不变 | prefill 必修项使 B=1 total −29.8% | 需求自身列必修项，与逐位判据冲突 | 是（用户 2026-11-21 确认） |
| §4.7 等待归因口径 | 并集 + 暴露等待 | 目标资源交段四项守恒 | 代理量与占用者无关，反例实证 | 否（用户裁决目标资源口径，round22 回写） |
| §4.8 R8 优化范围 | 曾降级为扫参维度 | 四项优化 + 主机最小可执行落地 | 两次用户裁决以需求为准回写设计 | 是（用户裁决 2026-11-23/11-25） |
| §4.8 权重复用开关 | 曾表述为「--batch 档位本身」 | `--weight-reuse` 固定 B 对照开关 | --batch 耦合组大小/形状/装载 | 是（用户裁决 2026-11-25，round24 回写） |
| §4.8 主机侧执行能力 | 设计只定义 genesim 侧 | 增主机逐槽/副本最小可执行接口 | 评审钉住完成口径混用 | 是（用户裁决 2026-11-25，round24 回写） |
| §4.1 每档 B 一次导出 | 逐档真算 | 按最大档编译一次 + 折算标注 | 省分钟级图导出；判据只对编译档成立 | 是（round05 问题 11 取舍） |
| R6 padding 判据 | 输出 padding 比例 | 维持等长、删除恒零字段 | 变长属连续 batching 边界 | 是（round05 问题 10 取舍） |
| §4.9 报告落盘 | 未定义世代一致性 | 指纹 + 口径门禁 + 过期自检 | 报告曾引用已消亡产物世代且错标口径 | 否（round09/round21 评审驱动） |
| 通信字节口径（设计未定义） | 未定义编译档解耦 | 导出侧单行口径 + v6 版本契约 | K6 破缺：计价随 B_export 漂移 | 否（round12–14 评审驱动） |
| sidecar 版本契约 | 版本字段恒 5 | 版本 6 + 装载门禁（collectives 面） | 口径语义需机器可检测 | 否（round15 评审驱动） |
| 产物可复现（设计未定义） | 共享路径 + manifest 记身份 | 冻结 + 逐档落位（内容寻址副本） | 共享路径被覆盖致历史档失配 | 否（round17/21 评审驱动） |
| §3.1 步级行数（round25） | 步级形状返回单槽 S | `closed_batch_binding` 单一真源 | 两套行数致合批通信量少算 B 倍 | 否（round25 评审驱动，§八.6 已回写） |
| §4.6 首 token 时点（round25） | 按首个合并 decode 步计 TTFT | prefill 完成即记、全 B 同一定义 | 与真实编排器逐槽采样对齐 | 否（round25 评审驱动，§4.6/§八.6 已回写） |
| §4.8 执行策略输入（round25） | 策略只进 genesim 配置 | 统一编译输入贯穿导出链 | 优化档须与同次编译产物同语义 | 否（round25 评审驱动，§4.8 已回写） |

## 八、当前遗留问题

　　本节汇总当前尚未关闭的问题、其影响与来源，逐条状态见表 12。

<p align="center">表 12　当前遗留问题</p>  

| 编号 | 问题描述（≤30字） | 影响范围（≤20字） | 来源 | 状态 |
|------|-------------------|-------------------|------|------|
| L1 | 甲方 TTFT/TPOT SLO 目标未提供 | 最优配置推荐准则只能按吞吐-时延 Pareto | request §八.5（Q7） | 待确认（外部依赖） |
| L2 | 抢占解析项系数无实测锚点 | 抢占代价为解析估算、标注未校准 | 设计 §九.3（Q3 关闭说明） | 已规避（带宽按方向从拓扑推导；实测校准待外部锚点） |
| L3 | 含 b1 的三条全档命令 90.6–93.9 分钟，超 90 分钟纪律最多约 4 分钟 | 全档重跑的时间预算 | implement I26、round11 问题 2 | 已规避（`--sim-batches 1` 已是最细粒度，计划内偏差） |
| L4 | 非编译档容量占用为线性折算 | 小档容量列精度 | 设计 §九.6（round05 问题 11） | 已规避（显式标注；逐档真算列为后续可选） |
| L5 | DP 严格同规模对照与 khp 同代对照不可构造 | 两开关收益为输入级受控/逐副本口径呈现 | implement §4.4 | 已规避（口径如实标注；机制与相对结论不受影响） |
| L6 | 连续 batching、cascade 级联、真抢占执行侧未做 | 变长/动态形态与多芯粒扩展 | request §2.3 边界 | 待确认（需求边界明确不做，建议后续单独立项） |
| L7 | 盘上 18 档 sweep、15+1 份报告（含 b4g1 单档）与最优推荐为旧口径 | 新口径全档数字与推荐缺位，重扫前推荐视为未定 | 用户终裁 decisions.tsv 09:43:39、implement §八 R9/R10/R12/R15/R16/R17 | 登记边界（按终裁静态收口不执行；产物已全部就地标注旧口径警示） |
| L8 | `--weight-reuse` 成对 7B 对照曲线未产出 | 权重复用收益只有机制层与单测证据 | implement §八 R13 | 登记边界（开关与报告管线已就绪；按终裁与常规轮纪律不执行） |
| L9 | 主机侧逐槽/副本的 7B 规模锚点未执行 | 7B 规模主机侧对拍证据缺位（小 fixture 已覆盖） | implement §八 R14 | 登记边界（B=2/4 小 fixture 21 例对拍在位；7B 扩展用例留待后续轮） |

　　除表 12 所列外，已核对：实施文档已知问题 K1–K6 均已关闭（K6 于 round14 根因修复），评审 82 条问题除误判与取舍外全部修复并有测试或产物证据（round24 复发抽验 R21-01..06 零复发、round26 复核 R25-01..03 根因级且 R21/R22/R23 修复零复发），6.2 全部验证项通过（除按终裁登记不执行的 L7–L9 边界项），1.3 全部需求已完成或部分完成于登记边界内（1.4 完成度以机制层与常驻黄金值为验收依据）；旧口径 18 档产物与 11 份报告归档保留为历史证据（`test-results/archive-old-caliber-20261124/`，机制结论可溯、绝对值失真已声明），旧口径 final2 与 r23-smoke 产物目录原样留存（实施文档第八节 R9/R17 登记条目指向）。本文档（techdoc）即实施文档第八节 R5/R11 登记条目的执行——由循环通过后自动重生成，其引用的全档数字与最优推荐为旧口径产物，以 R9/R10/R12/R17 登记条目为准。
