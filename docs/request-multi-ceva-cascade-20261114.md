# 需求文档：多 CEVA 级联仿真与 Llama2 7B 全流程性能评估

> 文档编号：request-multi-ceva-cascade-20261114
> 创建日期：2026-11-14
> 关联项目：flagos-pim-compiler（图编译器）、FlagTree（算子编译器）、/data/fjg/ceva（CEVA 工作区）、genesim（参考）

## 一、需求背景与目标

　　本仓已初步完成基于 PIM 架构的存算一体编译器，并通过三个仓库的 `support ceva Simulator` / `support ceva backend` 系列提交走通了「单 CEVA 硬件仿真器」的 Llama2 7B 全流程：`scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10` 一条命令完成图编译导出 65 张子图、算子编译、GML 与全部 bin 生成、厂商 GC 编译、DDR 布局与 CEVA 仿真，最终按 stage 级 cycles 对基线判定并核对 tokens（`scripts/run_full_pipeline.py:730-749`）。但该流程只支持单个 CEVA 仿真器——一个 TCS 仿真进程内的 1 个 NPM 核（controller 跑 Llama2Runner.elf，另加 4 个 VPU engine，`/data/fjg/ceva/scripts/07_run.sh:27-33`、`src/automation/run_tools.py:92-139`），无法评估多个 PIM 核并行的收益，也无法为多核目标架构的编译决策提供性能依据。

　　本需求把这条链路从单核扩展到多核级联，要做三件事。其一，建立多个 CEVA 仿真器级联的仿真底座与配套的编译、内存布局改造，使 Llama2 7B 能拆分到多核上跑通并正常输出 tokens。其二，在 pim-compiler 建立面向多 CEVA 级联的编排器，负责任务拆分、映射、内存分配、通信编排与并行调度，把图编译、算子编译、GML 生成与 CEVA 仿真整套串联起来；仓库里面向 genesim 的编排器（`runtime/executor.py` 的 ExecutionPlan 命令表与解码循环、`comm/plan.py` 的五类通信计划表（all_reduce、all_gather、all_to_all、scatter、local_slice）、`graph/strategy.py` 的 TP/PP 混合策略）提供了可直接借鉴的结构。其三，建立面向多 CEVA 级联的评估模型：每个 CEVA 仿真器只输出自身核内的分项 cycles，新模型要在这些实测数据之上叠加核间依赖、通信与并行代价，给出准确、客观、可外推的整体性能结论。

　　经与用户确认，本轮的关键决策如下。级联采用多进程形态：每个 CEVA 仿真器是独立进程、每进程仿真 1 个 NPM 核（即现有已验证的单核路径），跨核数据经 host 侧编排器真实搬运——这样每核计时沿用可信口径，跨核通信代价可实测标定，同时规避厂商自述的「单进程多核并行仿真计时不准」问题。切分策略先 PP2（每核 16 层）走通全链路，再 TP2 体现单流 decode 的并行收益；PP2 通信最少、子图几乎不用改、数值与单 CEVA 逐位等价最好对拍，TP2 才能在单个 decode 步内并行、拿到单请求的加速（本期 PP2 单请求不分块，只能串行推进）。编排器落点在 pim-compiler，`/data/fjg/ceva` 工作区只做仿真侧适配。编译调优链路本期不接入（CEVA 实跑一轮约 10 小时，其中仿真约 7 小时，无法逐调优候选评估），流程接受手工指定策略，调优衔接留待后续。实测规模先做 2 个 CEVA，评估模型按核数与策略参数化，为后续 4/8/16 核扩展留好抽象。

## 二、核心功能需求

### 2.1 功能清单

　　功能点按依赖顺序组织：先建多进程级联底座与多核内存布局（R1、R2），再改图编译器产出多核切分导出（R3），在此之上实现编排器并以 PP2 走通 token 对拍（R4），随后扩展 TP2 并行编排拿并行收益（R5），最后把全流程串成一条命令入口（R6），并建立消费实测数据的组合评估模型（R7）。

**R1（P0）多进程 CEVA 仿真器级联底座**

　　多核级联的前提是编排器能同时驾驭多个仿真器进程，而当前仿真入口只面向单核。现状是 `run_tools.py` 打开一个 controller 会话加载 Llama2Runner.elf、四个 VPU 会话加载 vpu_runtime.elf，整个模型在一个核内一次跑完（`/data/fjg/ceva/src/automation/run_tools.py:92-139`）；厂商 `llama2_npm.cpp:9-160` 的三个运行函数（`run_prompt`、`run_decode`、`run_lm_head`）都只接收单个 `NeuProMDeviceIF*`，没有任何多核分支。因此需要在 `/data/fjg/ceva` 提供多进程启动与驱动能力：按核数拉起 N 个独立 TCS 仿真器进程（每进程 1 个 NPM 核），并暴露编排器所需的控制原语——分步执行（按子图或按阶段推进，而非一次跑完）、读写指定核的 DDR 地址区间、跨核搬运数据块。厂商原始包里的多核基础设施（`system_app/NPMMultiCore_API` 的按核加载与 AllGather、`dsp_app` 的 mailbox 消息协议与 `core_id<<32` 每核独立 DDR 地址空间）可作为消息语义与地址机制的参考，但不在其单进程框架内落地。本文以「2 个仿真器进程并行存活，编排器能分别驱动执行并完成一次跨核数据搬运，且各进程能分别产出自身的 cycles 计数」作为本功能点的验收判据。

**R2（P0）多核 DDR 布局与内存分配**

　　级联后每个核只装自己那份产物，DDR 布局必须按核独立生成。现状是 `src/ddr_manager/llama2_ddr_manager.py:7` 把起始地址 `0x100000000` 硬编码为单核布局，单次运行把 65 张子图约 8.2GiB 的 SubgraphRegistry 连同 embedding、KV cache 等预分配区（合计约 8.9GiB）全部装在一块连续 DDR 上。需要把布局器多核化：输入按核切分后的子图集合与 bin 清单（来自 R3），为每个核生成自己的 `*_map.ini`（子图、权重、输入、KV、输出），地址按核隔离：厂商单进程多核口径为 `core_id<<32` 每核对齐基址，多进程路线下每进程地址空间独立，布局器按核生成互不重叠的地址区间；并做容量核对——PP2 与 TP2 下每核各装约一半权重（约 4.1~4.5GiB 级），须与每核实际 DDR 窗口核对。本文以「2 核布局地址空间无重叠、每核产物齐全可加载、容量不超每核窗口上限」作为验收判据；每核窗口的精确上限列为待确认事项，在设计阶段按 tcs 配置核实。

**R3（P0）图编译器多核切分导出（PP2/TP2）**

　　当前 CEVA 导出完全没有切分概念：`export_ceva_subgraph` 只做 torch.export 加设备标注加融合，产出 65 张完整单层子图（`gml_bridge/ceva.py:729-740`，全文件无 strategy/num_dpus/tp_width）。需要把 `graph/strategy.py` 的 ShardStrategy（PP 段数、TP 宽度）接入 CEVA 导出：PP2 下每核导出自己 16 层的子图集（子图内容不变，归属按层分段，lm_head 归末核）并在层边界标注跨核交接；TP2 下按既有切分规则导出分片子图——q/k/gate/up/lm_head 列切、o/down 行切、注意力头对半、KV 头跟随切分，并在行切投影后标注归约点。全部 bin 仍由我方生成，沿用现有清单与摘要校验。本文以「按策略导出的 bundle 通过 `verify_gml_artifact.py` 清单校验，且每核子图分别过厂商 GC 编译成功」作为验收判据。

**R4（P0）多 CEVA 编排器与 PP2 全流程 token 对拍**

　　编排器是本需求的核心新增物。现状是两条链路都没有它：genesim 链路的 `runtime/executor.py` 提供 ExecutionPlan 命令表执行与解码循环（计划由 `runtime/compile.py` 生成，执行侧面向 HAL/NumpyBackend），CEVA 路径没有调用方；CEVA 链路的层间编排由厂商 runner 在单核内逐层串行完成。需要在 pim-compiler 新建面向 CEVA 级联的编排模块，借鉴 genesim 编排器的结构分为两层：编译期从策略生成多核蓝图（任务拆分与映射、每核内存计划、跨核通信计划表——记录每次搬运的字节数、源目的核与触发时机，可打印成可读文本）；运行时按蓝图驱动 N 个仿真器进程执行 prefill、decode 循环与采样，完成 KV 推进、层边界数据搬运与 lm_head 派发。首个落点是 PP2：核 0 跑前 16 层、核 1 跑后 16 层，每步在层边界把 hidden 交给下一核，lm_head 在末核执行。本文以「2 CEVA PP2 走完 prompt 加 10 步 decode，输出 tokens 与我方单 CEVA 基线（已归档 ours_e2e 运行的 11 个 token：`[13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]`）逐个严格一致」作为验收判据——PP2 是层间拆分，数值应与单核逐位等价，这是拆分正确性的最强信号。

**R5（P1）TP2 并行编排与并行收益**

　　PP2 走通后，把编排扩展到 TP2 才能体现并行：两核各自持有半份权重，每个 decode 步内两核并行执行同层的分片算子，行切投影（o_proj、down_proj）的部分和在 host 侧归约。这要求编排器的通信计划表支持归约类操作（对应 genesim 编排器的 all_reduce 语义，但落地为跨仿真器进程的 host 搬运），并处理注意力头切分下的 KV 读写与 lm_head 列切下的部分 logits 拼接。本文以「2 CEVA TP2 走完全流程并输出 tokens，与我方单 CEVA 基线前若干 token 一致（建议不少于 4 个，其余允许告警漂移），且编排器记录的核间时间线显示两核同层分片并行执行（执行区间重叠），而不是串行依次执行」作为验收判据。

**R6（P0）多 CEVA 全流程命令入口**

　　单 CEVA 已有一条命令入口（`scripts/run_ceva_pipeline.sh:1-18` 与 `scripts/run_full_pipeline.py:730-749` 的 `run_ceva`），多核路径也要同样的一键形态，避免手工分步拼装。需要扩展 `scripts/run_ceva_pipeline.sh` 与 `run_full_pipeline.py`（或新增并行入口）：增加核数与策略参数（如 `--num-ceva 2 --strategy pp2|tp2`），按策略串起 R3 导出、清单校验、按核 GC、R2 布局、R1 级联仿真、token 判定与性能数据归集；不带多核参数时保持现有单核行为不变。本文以「一条命令完成 2 CEVA PP2 全流程并产出判定报告，单核调用路径回归不变」作为验收判据。

**R7（P0）多 CEVA 组合评估模型**

　　每个仿真器只知道自己核内的 cycles，整体性能结论需要一个新的组合评估模型。现状是仓库里没有多 DPU 整体评估代码（`comm/plan.py:347-375` 的 `HostStarCostModel`/`plan_cost` 只有单条通信条目粒度且仅测试消费，多 DPU 的端到端指标由 genesim 给出），CEVA 侧输出又只有 cycles 口径（`performance_counters.txt` 的 stage 级计数与每核 profiler 文件）。需要在 pim-compiler 新建组合式评估模块：输入为各核仿真器的实测分项 cycles、编排器通信计划表的字节数与模式、级联链路参数；输出为整体指标——端到端总时间（按周期与频率换算）、tokens 吞吐、相对单 CEVA 的加速比、通信与依赖代价占比。模型按核数与策略参数化，硬件抽象遵循芯方舟厂商口径：PIM 颗粒即设备本身，每核单运算核时三层仿真抽象为 N,1,1（2 核实测即 2,1,1，可外推 8,1,1 或 16,1,1）。通信代价参数用实测级联运行（编排器记录的传输字节数与耗时）标定。本文以「评估模型对 2 CEVA PP2 与 TP2 端到端性能的预测与实测偏差不超过 20%，且相对结论方向正确——TP2 体现并行加速、PP2 的通信与流水串行代价被如实反映」作为验收判据。

### 2.2 核心流程

　　多核全流程从策略指定开始。首先由入口（R6）接收核数与策略（本期实测固定 2 核，PP2 或 TP2），图编译器按策略完成多核切分导出（R3）：PP2 产出每核 16 层、lm_head 归末核的子图 bundle，TP2 产出按列切/行切规则分片的子图 bundle，全部 bin 由我方生成并过清单校验。随后每个核的子图分别经厂商 GC 编译成 SubgraphRegistry（GC 可按核独立执行，产物互不干扰），DDR 布局器为每个核生成独立的 map.ini 与容量核对（R2）。在此基础上进入仿真阶段：编排器（R4/R5）按编译期蓝图拉起 2 个仿真器进程（R1），先各自加载本核 Registry 与输入，再按蓝图执行——PP2 下两核以层边界为交接点依次推进每步前向，TP2 下两核并行执行同层分片、行切结果经 host 归约——直到 prompt 完成取首 token，随后循环执行 decode 步并采样，最终输出 tokens 并做对拍判定。最后，评估模型（R7）归集两核的 stage 级 cycles、编排器记录的跨核传输明细与换算参数，输出整体性能指标与相对单 CEVA 的对比结论，连同判定报告一起归档。

### 2.3 边界定义

　　本需求包含：2 个 CEVA 仿真器进程级联的完整链路（多进程底座、多核 DDR 布局、多核切分导出、PP2 与 TP2 两种策略的编排、一键入口、token 对拍）与组合评估模型（含通信参数实测标定与按核数参数化的外推能力）。不包含：编译调优链路接入（用户明确本期不做，CEVA 实跑一轮约 10 小时、其中仿真约 7 小时，无法逐候选评估，流程以手工指定策略运行）、4 个及以上 CEVA 的实测（评估模型保留参数化外推，实测另行立项）、张量 batch 与多请求并发（属另一需求）、对厂商 GC 与仿真器内核算法的修改（只做接入适配）、以及 `/home/jgfeng/ceva` 原始参考目录的任何改动（绝对只读）。genesim 本期不改代码，其代价模型与 sidecar 契约仅作为评估模型的设计参考。

## 三、技术约束与依赖

1. **代码改动影响范围**

　　pim-compiler 是主战场：`gml_bridge/ceva.py` 的导出链需要接入 ShardStrategy 产出多核 bundle（R3）；新增编排模块（建议放 runtime/ 旁的新单文件模块，超过约 400 行再拆）承载蓝图生成与运行时驱动（R4/R5）；新增评估模块（R7）；`scripts/run_full_pipeline.py:730-749` 的 `run_ceva` 扩展多核参数（R6），`scripts/run_ceva_pipeline.sh` 继续透传。`/data/fjg/ceva` 工作区承担仿真侧适配：`src/automation/run_tools.py` 或新增多进程启动脚本支持按核拉起与分步驱动（R1），`src/ddr_manager/llama2_ddr_manager.py` 多核化（R2），`src/llama_runner` 需要改造或新增轻量核侧程序以接受编排器的分步指令（现状单体一次跑完，`llama2_npm.cpp` 全程序单设备），`scripts/08_result.sh` 与 `verify_run.py` 的判定扩展到多核 tokens 归集。FlagTree 预计不改：分片只是改变了喂给算子编译的本地形状，编译口径本身不变，若 TP2 分片形状触发缓存键问题则做小修。genesim 不改。`/home/jgfeng/ceva` 只读，工作区 run_all 第 7 步的原始目录快照对比机制继续作为不越界的校验。

2. **潜在技术风险与应对建议**

　　最大的技术风险是跨进程分步驱动的可行性：编排器要能在子图或阶段粒度上推进每个仿真器进程并读写其 DDR，`run_tools.py` 现有的 ceva_budi 会话用法只覆盖「加载即一次跑完」，halt/read/write 级控制能力需要先行探针验证；建议在设计阶段先做一个单层图双进程的最小探针，确认驱动机制后再铺开。其次是 GC 对分片子图的兼容性：PP2 不改子图内容风险低，TP2 改变 GEMM 本地宽度后厂商 GC 的行为需要单独验证，建议先拿一张 TP2 切分的层图过 GC 探路。资源约束是硬边界：`/data` 仅剩约 632G，单次 run 约 63G，全流程工作区预估需约 130G 可用磁盘（完整 65 图 GC 不复用缓存时历史记录约需 576G），双核与 TP2 会让图数量和中间产物叠加，需要复用 GC 缓存并按核清理中间产物；每轮仿真约 7 小时，验收排期要按天预留。TP2 的数值漂移（归约顺序改变 fp16 结果）会波及 token 对拍，已通过「前若干 token 一致、其余告警」的口径放宽。每核 DDR 窗口上限未核实（GC 硬件配置 `NPMEngine.xml` 写的 DDR 为 3983MB，但单核仿真实际装载约 8.9GiB 产物未报错，两者口径不一致；多核隔离后每核可用上限要在设计阶段按 tcs 配置确认），若 TP2/PP2 每核约 4.1~4.5GiB 超限则需要回收 KV 或分段加载策略。

## 四、非功能需求

　　可复现性：多核全流程必须一条命令跑完（R6），实验产物（bundle、按核 map.ini、蓝图、通信计划表、两核 cycles、评估输入输出、判定报告）按 run 名归档，任何性能数字都能追溯到具体的运行目录。可审计性：蓝图、通信计划表与评估模型的中间产物要能 print 出可读文本（与仓内关键中间产物约定一致），编排器的每次跨核搬运留有字节数与耗时的日志记录，供评估模型标定与问题定位。兼容性：现有单 CEVA 路径、pytest 测试套件与 `/home/jgfeng/ceva` 只读约束零回归；多核入口的默认行为等价现有单核行为。可信度：性能结论必须区分「实测（哪次运行、哪个文件）」与「模型外推（哪些参数是标定值）」，评估模型输出要附参数来源标注，沿用 genesim 配置里 parameter_sources 的成熟度思路。

## 五、验收标准

1. **回归判据**

　　改动前先跑基线：`python -m pytest tests/ -x -q -k "not llama2_7b"` 最近一次基线（2026-11-13，见 `docs/pim-compiler-v0.0.12.md` 7.2）为 2145 通过、1 失败、9 跳过、49 项被排除（失败项是 `tests/test_opcompiler_linear.py::test_cache_key_lock_serializes_processes` 的已知 fork 抖动，单跑 1.49 秒通过，全绿口径为 2146 通过）；多核改造合入后同一命令必须保持全绿，该抖动不计为回归。`-k "llama2_7b"` 的 48 个慢用例约 2.5 小时，按里程碑抽跑。单 CEVA 全流程不要求每轮重跑（仿真约 7 小时、含 GC 约 10 小时），以测试套件与入口的干跑检查代替日常回归，里程碑节点用 `--num-ceva 1`（或等价默认路径）确认单核行为不变。

2. **快速端到端验收**

　　新入口提供干跑模式（不启动仿真）：走完多核导出、清单校验、多核 DDR 布局与蓝图生成，GC 只做命令与清单核对或复用已有缓存，检查每核产物齐全、布局无重叠、通信计划表完整。复用缓存时干跑在数十分钟内完成，用于日常迭代；冷启动 GC 按约 3 小时、完整仿真按约 7 小时量级排期。

3. **逐项验收条件**

　　R1 以双进程并行驱动与跨核搬运演示验收；R2 以两核 map.ini 地址无重叠与容量核对验收；R3 以策略 bundle 过清单校验且每核子图过 GC 验收；R4 以 2 CEVA PP2 输出 tokens 与我方单 CEVA 基线（已归档 ours_e2e 的 11 个 token）逐个严格一致验收；R5 以 TP2 tokens 前若干一致（建议不少于 4 个）且两核同层分片时间线重叠验收；R6 以一条命令走通 2 CEVA PP2 全流程并出判定报告验收；R7 以模型输入输出落盘、参数来源可追溯并满足本节第 4 条偏差判据验收。

4. **性能判据**

　　评估模型对 2 CEVA PP2 与 TP2 端到端性能（总时间或总 cycles 口径，设计阶段钉死其一）的预测与实测偏差不超过 20%；相对结论方向正确，即 TP2 相对单 CEVA 体现并行加速、PP2 的通信代价与流水串行特性被如实反映（预测与实测同向）。通信参数以实测级联的传输记录标定，标定与验证不能用同一组运行。单 CEVA 基线数字取自已归档的 ours_e2e 运行，不做重测。

5. **不回归判据**

　　单 CEVA 路径的 tokens 判定口径、bundle 清单校验、原始目录快照对比（`/home/jgfeng/ceva` 零改动）全部保持；以下情形视为无效验收：tokens 一致但两核实际串行复用了同一份数据（未真正跨核搬运）、性能数字无法溯源到运行目录、评估模型用验证数据本身标定后再报告偏差。

## 六、调研补充信息

　　厂商多核基础设施（来源：`/home/jgfeng/ceva` 原始包 `full_release_package/vbu_npm_system/`，只读核对）：host 侧 `NPMMultiCore_API` 提供 `npm_initialize(number_of_cores, ...)`、按核占位符加载 registry、广播推理、AllGather 与按核读写内存；核侧 `dsp_app` 以 mailbox 消息循环（kInference/kAllGather/kSpKernelOp 协议）响应 host，地址空间按 `core_id<<32` 每核 4GB 对齐隔离。该框架适用于单进程多核路线；本项目选择多进程路线后，仅借鉴其消息语义与地址机制，不直接依赖。genesim 代价模型（来源：`genesim/src/scheduler/gene_sim_scheduler.py` 与 `genesim/conf/ceva_64core_4e.yaml` 实读）：链路表通信计价（基础延迟加字节除带宽）、离散事件并行流水、跨设备传输占用双端端口的竞争模型、放置 sidecar 契约与每 PU 分项指标输出，这些机制适用于评估模型的通信与并行代价建模参考；其周期成本参数多为 assumption 占位值，与 CEVA 实测口径未对齐，不能直接搬用。资源实测（2026-11-14 核对）：`/data` 剩余约 632G，`/data/fjg/ceva/runs/` 累计 298G，单次完整 run 约 63G（其中 CycleEstimation.csv 约 1.4G），10 步 decode 仿真约 7 小时；单 CEVA 全流程（含导出与 GC）约 10 小时。

## 七、本轮实测对既有结论的修正

1. 原始需求描述称「当前已经初步走通了单 ceva 级联到 pim 架构存算一体模拟器」，实测确认成立但需要澄清边界：走通的是「单 CEVA 仿真器」路径（一个 TCS 进程、1 个 NPM 核），且层间编排由厂商 llama_runner 在核内完成，pim-compiler 的编排器在该路径上完全未参与（`gml_bridge/ceva.py` 无切分概念、`runtime/executor.py` 无 CEVA 调用方）。依据：`run_full_pipeline.py:730-749`、`/data/fjg/ceva/src/llama_runner/src/llama2_npm.cpp` 全文核对。
2. 既有设计文档 `design-ceva-llama2-7b-20261028.md` 把多核（vbu_npm_system 最多 8 核）列为「本期不用」；本轮需求正式将其纳入范围，且选择了与厂商单进程框架不同的多进程路线。这是范围演进而非文档错误，记录于此供后续设计对照。

## 八、待确认事项

1. 每核 DDR 窗口的精确上限：厂商按 `core_id<<32`（4GB 对齐）分核隔离，GC 硬件配置 `/data/fjg/ceva/env/gc_w3/L2A/hw_cfg/m32k_vpu500_8192k_1024k_4e/NPMEngine.xml` 写的 DDR 为 3983MB，但单核仿真实际装载约 8.9GiB 产物未报错，两者口径不一致；多核隔离后每核可用上限需在设计阶段按 tcs 配置核实，直接决定 PP2/TP2 每核约 4.1~4.5GiB 产物能否一次装载。
2. 跨进程分步驱动的具体机制：ceva_budi 会话是否支持 halt/read/write/run 级控制、控制粒度多细，需最小探针（单层图双进程）验证后才能定编排器运行时的实现方式。
3. TP2 token 对拍「前若干一致」的具体数量：建议不少于 4 个，需在 TP2 首次跑通后按实际漂移情况定稿。
4. 评估模型 20% 偏差的口径：按端到端总时间还是总 cycles、按 PP2 与 TP2 分别验证还是合并验证，设计阶段钉死并在评估输出中注明；若用总时间，频率参数来源（TDF 或厂商口径）也需一并钉死。
5. lm_head 与 embedding 在 TP2 下的切分细节（lm_head 列切加部分 logits 拼接、embedding 复制还是切分）与 KV 头切分下的布局细节，属设计决策，不影响本需求边界。
6. 编译调优链路的衔接方式（评估模型作为调优评分后端、真实级联做终点验证）留待下一期立项；本期流程以手工指定策略运行。

## 九、附原始需求描述

背景：
1.当前已经初步完成基于pim架构的存算一体编译器；
2.当前已经初步走通了单ceva级联到pim架构存算一体模拟器，但是当前只支持单个ceva硬件仿真器，不支持多ceva硬件仿真器的级联。

核心目标：
1.建立多个ceva级联的硬件模拟器仿真器，走通llama2 7b->编译调优空间选择->图编译器->算子编译器->gml+对应gml的所有bin->多级联硬件仿真器评估模拟->输出性能数据
2.建立面向多ceva级联的编排器，能够实现上述的调度编排任务，确保最终可正常输出tokens，能够编排通信，任务拆分，映射，并行；
3.建立面向多个ceva级联的评估模型，每个硬件模拟器可输出特定的性能数据，但是不同硬件模拟器之间存在依赖，通信和并行等关系，需要能够在每个硬件仿真器输出数据的基础上，同时考虑多个ceva硬件仿真器的通信或者并行代价，准确，客观，具有通用性。

思路：
1.当前已经走通单ceva全流程，详见pim-compiler，genesim，和flagtree的提交commit: 包含：support ceva Simulator 这样字样的commit。运行脚本：图编译->算子编译->gml->cevl模拟器命令：
scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10；
2.我们可以修改ceva相关的脚本目录：/data/fjg/ceva
3.原始ceva模拟器相关参考资料，注意绝对不能修改的（只有1份）：/home/jgfeng/ceva，这里是核心参考
4.遇到ceva仿真运行问题，就是主要参考ceva中的相关资料或者程序；
5.需要建立ceva仿真器的编排器，来级联多个ceva，需要参考之前单节点实现的ceva的代码实现多节点的大模型任务拆分，映射，内存分配等。其实在pim-compiler有面向genesim模拟器的编排器，不是ceva，思路可以借鉴一下；
6.总之就是希望从先前的单ceva，扩展到多ceva的级联，发挥并行作用；
7.主要需要修改的代码仓库：
/data/fjg/flagOS/flagos-pim-compiler  图编译器                                                                                   
/data/fjg/flagOS/genesim  模拟器                                                                                              
/data/fjg/flagOS/flagOS-installers/FlagTree  算子编译器 
8.代价模型也是一个需要考虑的关键问题，之前在genesim的代价模型主要是面向genesim的，当前每个ceva硬件仿真器都会出一个性能数据，但是是单ceva内部运行的性能，现在要设计的评估模型是要考虑多级连下的整体性能；
9.因为ceva运行占据空间较大，我们可以先尝试2个ceva的级联。
10.值得注意的点：芯方舟厂商特别强调的是：在一台服务器上，PIM颗粒就是设备本身，虽然每一dimm 条上可能会有8个PIM 或16个PIM。对于host来说，这个就是8个或16个PIM核设备。如果每个PIM上只放单个运算核，代价模型的三层仿真抽象设置就是8，1，1或16，1，1。

主要审核点：
1.多ceva硬件仿真器下，走通llama2 7b->编译调优空间选择->图编译器->算子编译器->gml+对应gml的所有bin->多级联硬件仿真器评估模拟->输出性能数据 全流程；
2.多级连ceva下可正常输出tokens；
3.评估模型准确、客观、能够在单ceva仿真器评估数据基础上，考虑多级列下的依赖和通信代价，能够评估出大模型在多个ceva并行下的优化效果；
4.建立多ceva级联的编排器，能够调度和映射，任务拆分，映射，并行多个ceva硬件仿真器，同时能够把图编译，算子，gml以及ceva那一套都串联起来。
