# 需求文档：PIM 编译调优全流程闭环

> 文档编号：request-tune-loop-20261109
> 创建日期：2026-11-09
> 关联项目：flagos-pim-compiler（图编译与编排）、genesim（模拟器）、FlagTree（算子编译器）
> 状态：评审修订稿。范围与优先级可作为设计输入；准确性阈值与实测证据见第八节，未齐备前不得宣称完成硬件准确性验收。

## 一、需求背景与目标

　　PIM 推理性能由图切分、算子实现、放置和硬件配置共同决定，单项优化无法指导整套软硬件设计。因此需要把“候选配置→图编译→算子编译→GeneSim 仿真→性能数据→下一候选”做成可配置、可迭代、可追溯的闭环，并保证仿真消费本轮编译产物。

　　当前代码已有单次链路组成能力：`scripts/run_full_pipeline.py:815-862` 串联模型 IR、固定映射、图/算子编译、sidecar 和仿真，`:602-703` 校验产物；`graph/strategy.py:149-182` 能枚举切分；GeneSim 支持三种算法和八类放置移动（`genesim/src/autotune/api.py:98-177`、`genesim/src/autotune/placement.py:1311-1495`）；FlagTree 支持 WRAM 预算分块和形状提示（`FlagTree/include/triton/Dialect/TritonPIM/Transforms/Passes.td:112-179`）。

　　这些能力尚未形成外层闭环：单次脚本没有候选评分与选优，生产导出未调用策略枚举，原生搜索不驱动图和算子重编，硬件参数仍分散在 `contracts/op_contract.py:54-62`、`genesim_bridge/paths.py:30-47`、`scripts/export_pp_placement.py:49-55` 和 GeneSim YAML 中。CEVA 多项时钟、周期和链路参数仍为假设（`genesim/conf/ceva_64core_4e.yaml:182-335`），一次成功仿真不能证明准确性。

　　本文要求组合图策略、算子参数、硬件画像三层外层空间与 GeneSim 原生放置搜索。编译侧固定本地形状和 PIM MLIR；通用 PIM 可搜索放置，CEVA 保留一对一映射。全链路共用画像并按内容指纹隔离产物，最终以独立校准集和留出验证集区分“模型预测”和“范围内已验证”。

　　目标 1 对应 R6，目标 2 对应 R2/R8，目标 3 对应 R3/R4，目标 4/6 对应 R5，目标 5 对应 R7，目标 7 对应 R1/R9。P0 是闭环与可信性机制的必需集合，P1 随后交付；R7 实测达标是目标 5 的关闭条件，不被其他用例通过替代。

## 二、核心功能需求

### 2.1 功能清单

　　功能点按“配置与空间→图和算子接入→仿真一致性→迭代输出→校准与硬件建议→用户示例”的顺序组织。每项均给出代码现状、目标行为和可复现判据，避免仅以“命令退出成功”代替需求验收。

**R1（P0）统一调优配置与命令行入口**

　　当前 GeneSim 搜索参数、图导出参数和算子硬件参数分别来自 YAML、命令行、`PIMHardwareConfig` 与 `paths.json`，没有一处能完整描述一轮调优。本文要求用一份版本化 YAML 描述模型与轨迹、三层搜索空间、原生放置策略、外层与内层预算、算法及种子、硬件画像、缓存档位和输出目录。命令行至少支持 `--config`、`--max-evaluations`、`--max-time-s`、`--algorithm`、`--profile`、`--cache-mode`、`--dry-run`、`--resume` 和 `--output-dir`；高频覆盖项必须合并回同一份生效配置并落盘，不能形成第二份隐式状态。

　　未知字段、非法取值、未支持的优化开关必须在任何编译前报错。轮数和时间分别约束外层候选与内层放置搜索，任一预算耗尽后不再启动新的昂贵评估；正在执行的编译或仿真允许完成并记录，因此墙钟预算是“停止发起新任务”的上限语义，不承诺强杀当前任务。本文采用同配置、同种子、同源码和同输入下候选顺序及最优候选一致进行验收，并以负预算、未知字段和不存在画像三个反例验证前置失败。

**R2（P0）分层调优空间与 GeneSim 策略覆盖**

　　外层空间分为三层。图策略层枚举满足层数、头数和 TP 宽度约束的 `num_dpus/num_stages`，复用 `llama_strategies()` 的合法性规则，并允许用户收窄范围。算子层搜索当前编译路径真实支持的参数，至少包括 `num_tasklets`；`full_m/full_n/full_k` 只是从实际算子形状得到的编译提示，不得作为搜索变量，显式分块也只有在外部接口真实支持后才能开放。硬件层可扫描设备数、每设备簇数、每簇 TensorPU 数、容量、时钟、周期成本、链路带宽与延迟以及 WRAM/MRAM/DMA 对齐，所有变动都要进入同一画像并触发必要的重编。

　　除三层外，通用 PIM 必须保留 GeneSim 原生放置空间，不得用外层候选协议替换后丢失已有能力。用户可配置三种搜索算法及各自参数、八类移动权重、片段和层块大小、放置预算、`total_time_s` 或 `system_efficiency_utility` 目标，以及 pipeline/predictor 筛选后端；`system_efficiency_utility` 只允许 pipeline/pipeline，predictor 筛选使用 `total_time_s`。最终排序和结论必须使用 pipeline 重放，CEVA 不启用该内层搜索。小空间另支持确定性枚举用于 dry-run 和回归，大空间默认不展开三层笛卡尔积。

　　验收分为覆盖与合法性两部分：三种算法各跑一个最小真实搜索，八类移动均能通过权重配置启用并被日志记录；每个开放参数至少有一个测试证明它改变了预期编译产物或生效仿真配置；非法 TP、超物理资源、未知移动和非法算法参数在编译前被拒绝。相同种子下候选顺序一致，预算计数包含基线且遵循 GeneSim `Evaluation.evaluator_invoked` 约定。

**R3（P0）图编译接入候选并原子化导出产物**

　　图导出现状是 `scripts/export_pp_placement.py:126-202` 将模型加载、策略构造、预算构造、编译和导出集中在 `main()`，调优器无法安全注入候选。本文要求抽出可复用函数，把候选转换为同一套 `ShardStrategy`、`HwBudget`、`PIMHardwareConfig` 和可选 `PartitionPlan`，再调用 `runtime/compile.py:199-315` 的 `compile_llama2()`；不得另写一条只服务调优的图编译路径。

　　每个候选应输出 `placed.ir`、`placement.json`、实际使用的 `partition_plan.json`、生效候选配置和产物清单。内容指纹至少覆盖模型内容、图策略、算子参数、编译相关画像和源码/工具链身份；创建时间只写入元数据，不参与内容键。产物先在临时目录生成，完整校验后原子发布到 `compile_key` 目录，失败候选不得作为可复用完整缓存出现。sidecar 需能回读策略、DPU 映射、物理/预留/可用容量及画像哈希，不能只保留当前的三项容量字段（`genesim_bridge/placement_export.py:590-614`）。

　　本文采用小模型合法/非法策略、两次相同候选和一次注入失败进行验收：合法候选的 `dpu_to_cluster`、策略名和画像一致，相同候选得到相同内容键，不同候选产物并存，失败目录没有完成标记且不会被后续轮次命中。单候选失败只淘汰自己并保留诊断，不终止整个搜索。

**R4（P0）算子编译参数、形状与缓存真实贯通**

　　A 路通过 `a_path_passes()` 将硬件五元组传入 `convert-triton-to-pim`（`opcompiler_bridge/pipeline.py:34-55`）；B 路 `_module_text()` 已使用 `request.hardware`，而 `_run_oplevel_triton_opt()` 的全局 `pim_options()` 仅在缺预算属性时兜底（`opcompiler_bridge/driver.py:418-470`、`:796-828`）。真正需要补齐的是导出脚本硬编码、成本提取的全局参数、A 路 `full-m/full-n/full-k` 透传以及请求级优化参数的统一缓存键，不能误判为 B 路完全绕过请求。

　　候选中的算子参数和画像必须进入 A/B 两路及成本提取。`full_m/full_n/full_k` 由请求的真实未分块形状推导并进入 `_cache_key()` 和 pass 选项；WRAM 等硬件预算只能在硬件层变更，不得在算子层另设一份互相覆盖的值。现有融合、相位展开和契约校验继续执行，目标格式要求的融合不提供会导致产物失真的伪开关；用户可配置的是已支持的搜索策略、移动、预算和合法优化范围。成本精化在调优正式路径同时使用 `--placement` 与 `--fusion-from`，按本地分片形状测量并保留融合清单；旧单次链路的全局形状行为保持不变。

　　验收至少覆盖一个 A 路 linear 和一个多 DPU B 路矩阵算子：tasklet 或 WRAM 变化时，实际 PIM MLIR 中对应属性、编码或分块按语义变化，缓存键必然变化；`full_m/n/k` 与请求形状不符时前置报错；A/B 与成本提取读到相同画像；已声明 PIM MLIR 读取失败时直接失败，不回退手写模板。不得要求所有单 DPU B 路算子在 tasklet 改变时凭空产生不存在的编码。

**R5（P0）GeneSim 配置生成与编译—仿真一致性**

　　现有启动前校验能发现过期 PIM MLIR 和 IR 编号错位，但不能证明硬件参数同源。本文要求新增唯一 `HardwareProfile`，统一生成 `HwBudget`、`PIMHardwareConfig`、固定 PU 映射输入和 GeneSim 生效配置。容量在跨仓接口中统一用整数字节，时间用纳秒，GeneSim 的 GiB 字段只在边界按 `2**30` 换算；CEVA 当前 8,000,000,000 B 是待确认假设，不能与 8 GiB 混写。配置需显式包含模型 IR、请求轨迹、放置与成本 sidecar、`output.results_dir`、`output.pim_traces_dir`、严格 PIM 模式和结果成熟度输入。

　　仿真前依次核对候选画像哈希、图 sidecar 画像与容量、`PartitionPlan` 和 sidecar 的 DPU 映射、PIM MLIR 哈希与工具链身份、仿真 IR 的算子编号及类型。正式调优要求 `scheduler.use_compiler_local_shapes=true`、`scheduler.require_compiler_pimir=true`，轨迹来源中 `template=0`、`unknown=0`，只有已被编译图吸收的算子可记为 `absorbed`。任一校验失败不得启动仿真，也不能沿用旧结果目录。

　　CEVA 保持 `autotune.enabled=false` 和编译器一对一映射，允许只使用部分物理核，但 `num_dpus` 不得超过 `num_pim_devices × clusters_per_device`，并遵守 CEVA 单设备、64/128 核和每簇一个 TensorPU 的现有约束（`genesim/scripts/export_fixed_pu_mapping.py:249-295`）。通用 PIM 可开启原生放置搜索，但不得改变编译器确定的切分、本地形状和 PIM MLIR；需保存最终放置 IR/清单及输入 sidecar 哈希，并用 pipeline 对最终放置重放。这样才能同时满足“覆盖原生调优”和“仿真的是本轮编译及调优结果”。

　　验收包含画像、容量、`num_dpus`、PIM MLIR 哈希和算子编号五类篡改反例，以及两种画像的正例：CEVA 的最终映射与 sidecar 完全一致；通用 PIM 的最终放置来自本轮搜索，最终重放分数与搜索最优值一致，且本地形状和 PIM MLIR 哈希未变。

**R6（P0）迭代闭环、缓存、续跑与结果输出**

　　当前 `run_full_pipeline.py` 只执行单次链路，GeneSim 缓存也只覆盖一次放置搜索。本文要求外层编排依次产生候选、编译图和算子、精化成本、生成仿真配置、执行一致性校验和仿真、解析结果，再把标量评分交回搜索器。支持最大候选评估数、最大墙钟时间、算法原生早停、失败隔离、dry-run 和断点续跑；原生放置内层预算单独记录，不能用外层一轮掩盖内部多次仿真。

　　快速档按 `compile_key` 复用完整图/算子产物，对新 `sim_key` 仍执行仿真；同一 `sim_key` 的续跑可复用成功结果。严格档强制重编并重跑最优候选。`compile_key` 覆盖模型、策略、算子参数、编译画像和源码/工具链内容，`sim_key` 再覆盖时钟、周期、链路、轨迹、放置策略和基配置；源码身份不能只用可能含未提交改动的 `HEAD` 或文件时间代替。缓存只改变实际耗时，不得改变模型输出的 `total_time_s`。

　　每轮以 JSONL 追加候选编号、三层与放置配置、产物路径和哈希、缓存状态、仿真总时间、计算/通信/访存效率、容量指标、结果成熟度、实际耗时和失败原因；最终输出所有成功候选排序、`best.json`、人类可读摘要和严格复现命令。验收使用两个真实候选跑完闭环，另用故障注入验证失败隔离和续跑；对同一记录候选序列进行快速/严格复跑时，最优候选一致，`total_time_s` 在 `1e-9` 相对容差内一致，实际耗时允许不同。

**R7（P0）代价模型校准、留出验证与结果分级**

　　目标 5 的“准确、通用”不能由结构校验或单次 CEVA 运行推出。现有 `_ceva_result_maturity()` 已能依据独立验证报告、参数来源、后端、形状和请求边界返回 `model_prediction` 或 `validated_in_scope`（`genesim/src/scheduler/gene_sim_scheduler.py:701-788`），应复用该机制，而不是另造一份只用于调优的“准确性通过”标志。

　　本文要求提供可执行的校准与验证输入契约：观测数据来自 CEVA 厂商仿真器或真硬件，校准集和留出验证集分离，覆盖代表性 prefill/decode、计算、DMA、主机编排和通信测点；每个样本保留目标配置、形状、数据类型、预测值、观测值及误差。拟合结果写回画像及参数来源台账，调优输出必须携带适用范围和 `result_maturity`。缺少实测、来源仍为假设、超出验证范围或最终使用 predictor 时，结果只能标为模型预测，不得进入“已验证性能”结论。

　　准确性按组报告带符号误差、绝对相对误差的 median、p90（最近秩）和 max，统计公式须能由逐点样本复算。合格阈值需由甲方或技术负责人确认；阈值未确认或独立数据未提供前，只验收报告与分级机制，整体目标 5 明确保持未关闭。通用 PIM 画像在没有对应实测数据时只用于趋势和排序，不与 CEVA 实测误差混比；“通用”指同一画像、产物和报告契约适用于已支持的 PIM/CEVA 场景，不凭空承诺未接入模型或硬件的绝对精度。

**R8（P1）硬件画像敏感性与厂商建议**

　　在闭环可复现之后，需要把逐候选数据转化为硬件设计依据。本文要求固定最优算子参数和策略选择规则，逐字段扫描核数、容量、时钟、周期成本、链路带宽/延迟及 WRAM。只改仿真字段时复用编译产物；改变核数、容量或编译预算时必须按同一规则重编，不能拿旧 PIM MLIR 假装评估新硬件。非法扫描点要记录原因，不得静默删除。

　　输出原始扫描数据、敏感性排序、拐点和至少一组二维 Pareto 前沿。连续参数给出单位变化及弹性，离散或枚举参数只给分组差分；拐点阈值和计算公式写入配置与报告。报告必须列出固定项、重编项、扫描范围、参数来源、结果成熟度和“相关性不等于因果”的边界。本文采用原始 JSONL 独立复算排序、拐点和 Pareto 前沿，并由脚本重新生成图表进行验收。

**R9（P1）示例与用户文档**

　　用户需要能直接修改和运行的示例，而不是只有内部 API。仓库应提供快速闭环、策略搜索和硬件扫描三份 YAML 及使用文档。快速闭环使用真实图编译、算子编译和 pipeline 仿真，限制请求数与两个候选以控制成本，不能用 `--skip-simulation` 或假目标代替端到端验收；策略搜索演示 TP/PP、tasklet、三种算法与原生放置配置；硬件扫描演示核数、容量和链路参数及重编规则。小模型假后端仅用于单元测试，不称为真实闭环示例。

　　每份示例均给出命令、输入准备、预计耗时的实测口径、输出位置、关键字段解释和严格复现方法，并说明如何调整轮数、时间、优化范围和画像。本文采用干净输出目录逐条执行示例，检查两份真实候选结果、排序和产物哈希进行验收；耗时只能在交付环境实测后填写，不能沿用未经验证的估计。

### 2.2 核心流程

　　一轮调优先由配置生成完整候选，候选包含图策略、算子参数、硬件画像和原生放置策略。图编译随后确定切分、本地形状、内存计划和初始 DPU 映射，算子编译再按同一画像生成 PIM MLIR 与分块信息；成本精化使用这些本地产物，避免另按全局形状构造一套不对应的成本。

　　在此基础上，调优器生成候选专属的 GeneSim 配置、结果目录和轨迹目录，并执行画像、映射、产物和编号校验。CEVA 直接按编译器放置执行 pipeline；通用 PIM 若开启原生搜索，则先在固定编译产物上搜索放置，再对最优放置做 pipeline 重放并保存最终放置清单。仿真输出的指标和成熟度写回本轮记录，搜索器据此产生下一候选。

　　达到预算或算法早停条件后，调优器停止发起新评估并汇总成功候选。最后对最优候选执行严格重编、重仿真和产物复核；硬件扫描沿用同一执行入口。任一候选失败只记录并淘汰自己，已完成轮次不回滚，未完成目录不进入缓存。

### 2.3 边界定义

　　本文包含 Llama2-7B 主线及小模型测试、通用 PIM 与 CEVA 64/128 核画像、三层外层空间、GeneSim 原生放置策略、单机单进程编排、内容缓存、硬件敏感性和结果分级。已有模型入口支持的其他模型可复用契约，但本轮不承诺新增模型族。

　　本文不包含多服务器编排、真实张量批处理、功耗/面积/成本模型、厂商 GC 时间线重构、新 MLIR pass 或新算子实现、预测器训练框架、通用插件框架和图形界面。CEVA engine 级调度、W4A8 物理布局和分阶段混合精度成本等既有遗留问题不得因闭环完成而宣称解决；其影响继续通过结果成熟度和适用范围披露。

## 三、技术约束与依赖

1. **代码改动影响范围**

　　图编译器新增画像、候选、编排、循环、报告和扫描，复用 `graph/strategy.py`、`runtime/compile.py:199-315`、`genesim_bridge/placement_export.py:537-749` 与 `opcompiler_bridge/driver.py:950-1035`；`export_pp_placement.py` 解除硬编码，成本提取接收同一画像和本地形状，旧单次命令不变。GeneSim 复用 `src/autotune/`、输出目录、运行清单和成熟度，仅增加画像与最终放置审计接口。FlagTree 已有 pass 支持所需参数，只改调用方并回归，不新增 pass。

2. **潜在技术风险与应对建议**

　　双层搜索可能使评估爆炸，需分别记账并限制预算，predictor 只作通用筛选。缓存混用以内容键、完成标记和原子发布隔离；每个开放参数用产物断言证明生效。CEVA 禁止原生放置，通用 PIM 固定本地形状/PIM MLIR 并保存最终放置。成本精化使用本地形状，结论携带成熟度；未校准只报告趋势。当前回归远高于原始“16 秒”描述，阶段门禁可分类拆分，但不得删除慢用例。

## 四、非功能需求

　　同配置、种子、源码和输入必须复现候选顺序与最优解；每个候选的输入、产物、最终放置、哈希、指标、成熟度和失败原因均需可回查。既有 `run_full_pipeline.py --num-stages 4`、`export_gml.py`、GeneSim `./run.sh --test` 和默认配置不得回归，新增可选字段缺省等价旧行为。

　　缓存和预算用于降低成本，但不承诺未实测的耗时；示例记录交付环境实测耗时。实现遵循 dataclass、类型标注和单文件优先，不引入插件框架。结果必须区分模拟时延与实际耗时、结构通过与数值通过、模型预测与范围内已验证。

## 五、验收标准

1. **回归判据**

　　平时执行 `python -m pytest tests/ -q -k "not llama2_7b"`，最终另跑 `python -m pytest tests/ -q -k "llama2_7b"`。`test-results/pytest-latest.md:1-7` 的当前记录为 1685 passed、8 skipped、49 deselected、约 1080 秒，原始“232 个、16 秒”不再是基线。GeneSim 执行 `./run.sh --test`，FlagTree 执行现有 PIM pass 与算子编译回归；新增失败必须处理，跳过项不算新增通过。

2. **快速端到端验收**

　　用两个合法候选和 `trace.num_requests=1` 跑真实编译、成本精化、pipeline 与排序，分别产出 sidecar、PIM MLIR、配置、轨迹、摘要和最终放置清单。续跑不得重复已完成 `sim_key`，随后严格复跑最优候选。`python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out` 只作既有导出回归，不能替代真实闭环。

3. **逐项验收条件**

　　R1 验收配置复现与前置报错；R2 验收三种算法、八类移动、参数产物断言和预算；R3 验收映射回读、原子发布和失败隔离；R4 验收 A/B 参数、形状、缓存和严格失败；R5 验收五类篡改反例、CEVA 固定映射、通用最终重放及零模板/未知轨迹；R6 验收两候选闭环、续跑和严格指标一致；R7 验收样本复算、留出集和成熟度；R8 验收扫描复算与图表重生成；R9 验收三份示例实际运行。

4. **性能判据**

　　排序使用 pipeline 的 `total_time_s`，并保留吞吐、效率和容量；predictor 只用于通用 PIM 筛选。严格复跑相对容差为 `1e-9`，它验证确定性而非硬件精度。硬件精度按 R7 的分组 median/p90/max 与独立范围判断；无实测或仍为 `model_prediction` 时不得宣称准确。

5. **不回归判据**

　　最终执行 `python scripts/export_gml.py --seq-len 128 --out-dir /tmp/gml_full`（32 层，约需 13 GB 内存）、`python scripts/run_full_pipeline.py --num-stages 4`、GeneSim 模型 IR/轨迹/成本提取与 `./run.sh`，以及 `python scripts/run_full_pipeline.py --backend ceva --run-name <固定名> --decode-steps 10`。CEVA 通过不等于 GeneSim 代价模型准确。既有 GML/CEVA 导出、NumpyBackend 对拍、通用与 UPMEM 行为不得回归；跳过哈希、复用旧结果或跳过仿真的验收无效。

## 六、调研补充信息

　　本轮未做外部网络调研，结论来自三个仓库的代码、`docs/cost-model-config-20261105.md`、`docs/techdoc-ceva-genesim-hw-align-20261109.md` 和既有测试记录。代价模型采用指令轨迹周期、链路服务时间与事件流水；其本地 DMA 与计算不重叠、区间倍率和原生后端拆分等边界以 `docs/cost-model-config-20261105.md:271-284` 为准，不能因接入调优而扩大解释范围。

　　用户提供的 `/data/fjg/flagOS/xinfangzhou-resource/Genesim详细设计.docx` 在当前文件系统中不存在，按该目录及 `/data/fjg/flagOS` 下同名模式检索也未找到，因此本轮未引用其内容。后续若补充该文档，应复核原生调优策略清单，但不得以此推翻已核实的代码行为。

## 七、本轮实测对既有结论的修正

1. B 路并非完全读取独立 `pim_options()`：模块文本已用 `request.hardware`，全局选项只补缺预算；真正缺口是导出与成本提取多源（`opcompiler_bridge/driver.py:418-470`、`:796-828`）。
2. `full_m/full_n/full_k` 是真实形状提示而非搜索变量，应由形状派生（`FlagTree/include/triton/Dialect/TritonPIM/Transforms/Passes.td:120-130`）。原生放置有八类移动和三种算法，故通用 PIM 保留内层搜索，CEVA 按现有约束禁搜；CEVA 也允许部分核，上限为设备数乘每设备簇数（`genesim/scripts/export_fixed_pu_mapping.py:263-295`）。
3. 缓存只能改变实际耗时，不能改变模型输出的 `total_time_s`；快速与严格档应比较相同候选的模型指标，而不是允许指标漂移。
4. 调优成本精化不能继续沿用全局形状，应同时传 `--placement` 与 `--fusion-from`；旧单次链路保留原行为（`genesim/scripts/refine_ir_with_flagtree.py:149-159`、`:184-210`）。
5. CEVA 实测计数并非现有输入。当前多项参数仍为假设，独立实测与阈值是目标 5 的未关闭前提（`genesim/conf/ceva_64core_4e.yaml:182-335`、`genesim/src/scheduler/gene_sim_scheduler.py:701-738`）。
6. GeneSim 已支持 `output.results_dir` 和 `output.pim_traces_dir`，多候选应直接据此隔离产物（`genesim/src/main.py:974-975`、`genesim/src/scheduler/gene_sim_scheduler.py:4700-4701`）。

## 八、待确认事项

1. CEVA 分组绝对相对误差 median/p90/max 的合格阈值，以及目标 5 要求覆盖的最小测点集；未确认前只能交付报告与分级机制，不能关闭准确性目标。
2. CEVA 每核“8 GB”究竟指 8,000,000,000 B 还是 8 GiB；当前代码采用前者且明确标为假设。
3. 独立校准集和留出验证集由谁提供、使用厂商仿真器还是真硬件、覆盖哪些形状和请求边界；这决定 R7 何时能实际达标。
4. 默认外层预算、内层放置预算和默认算法应在交付环境重新测量编译/仿真耗时后确定；本文不虚构默认值。
5. 原始“快速回归 16 秒”与当前约 18 分钟基线不一致，是否新增 pytest 标记拆分日常快速集需要另行确认；拆分不得删除最终全量门禁。
6. 通用 PIM 是否允许用 predictor 做大规模初筛已有接口基础，但具体模型是否已训练、误差是否可接受尚无证据；未验证前示例默认 pipeline。

## 九、附原始需求描述

下面是我的原始需求：背景：
1.当前已完成面向存算一体的AI编译器整体框架；
2.初步实现genesim仓库内容的编译调优流程，但是还没有真正走通从编译调优空间搜索->图编译->算子编译->genesim仿真->输出性能数据->迭代调优全流程。

目标：
1.走通从编译调优空间搜索->图编译->算子编译->genesim仿真->输出结果->迭代调优全流程；
2.编译调优空间支持当前所有或者大部分genesim内容的调优空间策略，侧重指导硬件厂商对硬件的配置和设计；
3.图编译，算子编译器支持当前编译调优的所有优化策略，能够正常转换；
4.从编译调优->图编译->算子编译->genesim仿真，以及genesim仿真硬件配置相关参数以及优化是一致的，不能genesim仿真不是编译器优化的结果；
5.genesim仿真评估能够真实反映整体编译调优流程，且是准确的，通用的；
6.从编译调优->图编译->算子编译->genesim仿真应该是调优的面向pim架构的ceva仿真参数一致；
7.整体编译调优框架方便用户调整调优空间或者策略，用户可以调整总体的轮数，或者调优时间，能够自行配置某些优化以及范围，同时能够给到用户示例。

思路：
1.当前genesim仓库已有编译调优基础框架，详见相关代码，具体说明文档详见：/data/fjg/flagOS/xinfangzhou-resource/Genesim详细设计.docx；
2.主要涉及修改的仓库包括：
/data/fjg/flagOS/flagos-pim-compiler  图编译器                                                                                   
/data/fjg/flagOS/genesim  模拟器                                                                                              
/data/fjg/flagOS/flagOS-installers/FlagTree  算子编译器
3.原始从调优->图编译->算子编译->genesim流程命令：python scripts/run_full_pipeline.py --num-stages 4
4.后面验证给出的命令通常运行时间过长，临时验证可用python -m pytest tests/ -q -k "not llama2_7b"，只有到最后阶段才进行python -m pytest tests/ -q -k "llama2_7b"。当然前面提到的必要的编译调优验证这个相关的酌情考虑；
5.需要代价模型准确，相关文档详见：/data/fjg/flagOS/flagos-pim-compiler/docs/cost-model-config-20261105.md

验证：
1.根据上述目标，都需要覆盖相关的测试用例；
2.原始的测试用例都需要通过，例如：
# 快速回归，约 16 秒，232 个用例
python -m pytest tests/ -q -k "not llama2_7b" （平时迭代用这个）

# 7B 全量，另开一轮
python -m pytest tests/ -q -k "llama2_7b"  （只有最后验证才用这个）

# 快速验证：取 1 层，约 1 分钟
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out

# 完整模型：32 层，需 13 GB 内存
python scripts/export_gml.py --seq-len 128 --out-dir /tmp/gml_full

完整一轮 genesim->图编译/算子编译->genesim
python scripts/run_full_pipeline.py --num-stages 4

genesim：
 ./run.sh --test

# 1. 激活环境
source  /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh

# 3. 生成模型 IR
python scripts/model_parser.py \
  --model_name /data/fjg/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --output models/llama2_7b.ir

# 4. 生成 Trace
./run.sh --trace --synthetic --seed 0 --num_requests 10 --output traces/llama2_7b.trace

# 5. FlagGems + FlagTree 成本提取（TTIR 对照，再 PIM MLIR 最终产物）
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_flagtree.ir \
  --sidecar models/llama2_7b_flagtree_extensions.json \
  --seq-len 128 --ir-level ttir

python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_pimir.ir \
  --sidecar models/llama2_7b_pimir_extensions.json \
  --seq-len 128 --ir-level pimir

# 6. 运行默认模拟
./run.sh。，利用skills编写需求文档，/home/jgfeng/.claude/skills/requirement-doc，对 docs/request-tune-loop-20261109.md,校对评审修复。利用skills :  ~/.claude/skills/tech-design-doc, 评审修复技术设计文档docs/design-tune-loop-20261109.md
