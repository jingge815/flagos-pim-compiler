# 技术文档：PIM 编译调优全流程闭环
> 文档编号：techdoc-tune-loop-20261112
> 创建日期：2026-11-12；本次修订：2026-11-13
> 关联文档：request-tune-loop-20261109、design-tune-loop-20261109、implement-tune-loop-20261109、review-tune-loop-20261110-round01 至 review-tune-loop-20261112-round23（共 23 份）、tune-bpath-tasklet-dpu-20261113、tune-optimization-switches-20261113、tune-search4d-20261113、tune-user-guide-20261113
> 关联项目：flagos-pim-compiler（图编译与编排）、genesim（模拟器）、FlagTree（算子编译器）
> 代码基线：flagos-pim-compiler main@df9b05c（含未提交改动）；genesim fjg-dev@3844941（含未提交改动）；FlagTree develop@0d7273bac（含未提交改动：`LowerPIMToEmitC.cpp` 按 tasklet 切分）
> 核心目标：把"候选配置 → 图编译 → 算子编译 → GeneSim 仿真 → 性能数据 → 下一候选"做成一条仿真严格消费本轮编译产物、可配置、可迭代、可追溯的闭环。

## 一、概述

### 1.1 背景与问题

　　本需求的核心目标是把"候选配置 → 图编译 → 算子编译 → GeneSim 仿真 → 性能数据 → 下一候选"做成一条仿真严格消费本轮编译产物、可配置、可迭代、可追溯的闭环。现有基础是 `run_full_pipeline.py` 的单次链路、`graph/strategy.py` 的切分枚举、GeneSim 的三种搜索算法与八类放置移动、FlagTree 的预算分块与形状提示。但这些能力没有外层候选评分与选优，硬件参数分散在图导出、成本提取和仿真配置中，产物没有内容键与原子发布，tasklet 等参数没有真实进入 B 路产物与仿真成本。本需求在不改变旧单次命令和既有契约的前提下，完成统一配置、分层搜索空间、图与算子编译贯通、跨仓一致性校验、缓存续跑、校准分级、硬件扫描和示例文档。

### 1.2 方案摘要

　　围绕核心目标，总体采用"单一硬件画像 + 外层候选状态 + 内容键隔离 + 跨仓产物契约"的技术路线。G 仓新增 `tune` 包，负责配置解析、四层候选生成、候选执行、结果记录、校准与扫描；图编译复用 `compile_llama2()` 与 `export_placement_for_strategy()`，算子编译 A/B 两路消费同一画像、优化开关与 tasklet 参数；S 仓 GeneSim 在启动前核对画像契约、候选编译身份、输入轨迹与最终放置，并按 tasklet 并行度计算矩阵算子成本；F 仓 EmitC 降级按 tasklet 切分输出行。通过 `compile_key`、`sim_key`、画像契约 schema 3 与放置 sidecar schema 5 连接各阶段，最终实现真实四维 16 候选闭环、续跑、严格复跑与硬件扫描。

### 1.3 完成情况

| 需求编号 | 需求内容（≤20字） | 完成情况 | 实现位置 | 验证用例 |
|----------|-------------------|----------|----------|----------|
| R1 | 统一配置与命令行入口 | 已完成 | `tune/config.py:344`（`load_config`）、`scripts/tune_pipeline.py:82`、`scripts/tune_oneclick.sh` | `tests/test_tune_config.py`；dry-run 列出合法候选 |
| R2 | 分层空间与原生策略覆盖 | 已完成 | `tune/candidate.py:109`（`CandidateSpace`）、`tune/loop.py:188`（`run_tune`）、`genesim/src/autotune/placement.py` | `tests/test_tune_candidate.py`；四维 16 候选真实闭环；三算法与七类移动（历史） |
| R3 | 图编译接入与原子导出 | 已完成 | `genesim_bridge/placement_export.py:867`（`export_placement_for_strategy`）、`tune/steps.py:458`（`build_candidate`） | `tests/test_placement_export.py`（64 条）；真实编译目录与 `artifacts.json` |
| R4 | 算子编译参数与缓存贯通 | 已完成 | `opcompiler_bridge/driver.py:243`（`_cache_key`）、`opcompiler_bridge/pipeline.py:58`、FlagTree `LowerPIMToEmitC.cpp:1160`、`genesim/src/pim/pim_trace_loader.py:383` | `tests/test_tune_compile_bridge.py`（25 条）、`test_opcompiler_ops.py` B 路矩阵乘对拍 |
| R5 | 配置生成与编译仿真一致 | 已完成 | `tune/steps.py:579`（`verify_compiled`）、`genesim/src/run_manifest.py:989`（`verify_compiled_artifact`） | S `tests/sim/test_run_manifest.py` 等；round18～22 正反例探针 |
| R6 | 迭代闭环、缓存与续跑 | 已完成 | `tune/loop.py:188`、`tune/steps.py:882`（`evaluate_candidate`）、`tune/report.py:239`（`write_best`） | `tests/test_tune_loop.py` 等；16 轮闭环、严格复跑同分（1e-9） |
| R7 | 校准、留出验证与分级 | 部分完成 | `tune/calibration.py`、`genesim/src/ceva_validation.py:1402`（`report_evidence_error`） | `tests/test_tune_calibration.py`（60 条，implement §4.2）；缺独立实测与用户阈值 |
| R8 | 硬件敏感性与厂商建议 | 已完成 | `tune/sweep.py`、`scripts/tune_sweep.py` | `tests/test_tune_sweep.py`（43 条）；两点实测与历史 16 点扫描 |
| R9 | 示例与用户文档 | 已完成 | `examples/tune/*.yaml`、`scripts/tune_oneclick.sh`、`docs/tune-usage.md`、`docs/tune-user-guide-20261113.md` | 三份示例 dry-run；一键脚本真实 16 轮 |

　　需求共 9 项，已完成 8 项，部分完成 1 项（R7，外部证据阻塞）。调优单测与 GeneSim 仿真（sim）套件全部由用户手动执行通过；真实四维 16 候选闭环、严格复跑与身份核对通过。当前遗留问题 12 条，详见第八节。

## 二、总体流程

### 2.1 总体流程图

　　本需求的核心目标是让仿真严格消费本轮编译产物，并把性能数据交回搜索器形成迭代。现有单次链路只能完成一次图编译、算子编译和仿真，缺少候选评分、内容键与跨阶段归属证明，tasklet 等搜索参数也没有真实进入 B 路产物与仿真成本。因此总体流程以"候选身份与产物契约"为主线：首先，`load_config()` 合并唯一生效配置，输入快照把模型、IR 与轨迹固化为内容身份；然后，外层搜索产生四层 `TuneState`（图策略、tasklet、优化开关、画像）并计算 `compile_key`；其次，图编译复用生产入口生成放置 sidecar；再次，算子编译 A/B 两路按同一画像与开关生成 PIM MLIR 与精化 IR；随后，`verify_compiled()` 与 GeneSim 统一产物入口核对画像、策略、映射、PIM MLIR 与输入轨迹；最后，仿真按 tasklet 并行度计成本，结果写回轮次日志，搜索器产生下一候选，或进入续跑、严格复跑与硬件扫描。如图所示，"契约与一致性校验"和"结果记录与迭代"是本次新增的核心节点，去掉任一节点后链路仍能跑，但无法证明仿真消费的是本轮编译和调优结果。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff,color:#fff

  A["统一配置与输入快照<br/>effective_config.yaml · inputs/"]:::stage --> B["四维候选搜索<br/>TuneState · compile_key · rounds.jsonl"]:::stage
  B --> C["图编译（复用生产入口）<br/>placed.ir · partition_plan.json · placement.json"]:::stage
  C --> D["算子编译 A/B 路<br/>pimir/*.mlir（带 DPU/tasklet/开关）· refined.ir"]:::stage
  D --> E["契约与一致性校验<br/>hardware_contract.json · verify_compiled"]:::stage
  E --> F["GeneSim 仿真<br/>tasklet 并行成本 · pim_traces · summary.json"]:::stage
  F --> G["结果记录与迭代<br/>best.json · search_coverage · 续跑/严格复跑/扫描"]:::stage
  G -.->|下一候选| B
  W["输入约束<br/>Llama2-7B · CEVA 64/128 核 · 通用 PIM"]:::input -.-> A
```

### 2.2 模块关系

| 模块/文件 | 职责（≤20字） | 改动类型 | 依赖的模块 |
|-----------|---------------|----------|------------|
| `contracts/hardware_profile.py`、`contracts/profiles/*.json` | 硬件画像单一真源与转换 | 新增 | 无 |
| `contracts/op_contract.py`（`PimOptimizations`） | 优化开关与编译目标契约 | 修改 | 无 |
| `tune/{config,candidate,inputs,steps,loop,report}.py` | 配置、候选、执行、日志、报告 | 新增 | 画像、图/算子编译链 |
| `tune/{calibration,sweep}.py`、`scripts/{calibrate_profile,tune_sweep}.py` | 校准分级与硬件扫描 | 新增 | 候选执行与报告 |
| `scripts/tune_pipeline.py`、`scripts/tune_oneclick.sh` | 统一入口与一键脚本 | 新增 | `tune` 包 |
| `genesim_bridge/{placement_export,refine_profile}.py` | 候选导出与成本精化契约 | 修改/新增 | `runtime/compile.py` |
| `opcompiler_bridge/`、`runtime/`、`memory/mem_planner.py` | 算子编译参数、形状与容量台账 | 修改 | `contracts/` |
| `genesim/src/{hardware_profile,ceva_validation,run_manifest}.py` | 画像契约、校准证据、产物契约 | 新增/修改 | G 仓 sidecar 与画像契约 |
| `genesim/src/{main.py,scheduler/…,pim/pim_trace_loader.py}` | 启动校验、最终放置、成本缩放 | 修改 | 上述 S 模块 |
| FlagTree `lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp` | EmitC 降级按 tasklet 切 M | 修改 | 模块属性 `pim.num-tasklets` |
| `examples/tune/*.yaml`、`docs/tune-usage.md`、`docs/tune-user-guide-20261113.md` | 示例与用户文档 | 新增 | `tune` 包 |

### 2.3 轮数、时间、算法与优化范围

　　调优的轮数、时间和算法都在同一份 YAML 中配置，命令行只做高频覆盖并合并回生效配置。外层预算约束"候选评估"，内层预算约束"放置搜索"，两者分别记账；墙钟预算只限制"发起新评估"，正在执行的评估允许完成，回放与离线等待不计入。

| 维度 | 配置项/命令行 | 语义 | 当前示例取值 |
|------|---------------|------|----------------|
| 外层轮数 | `search.max_evaluations`、`--max-evaluations` | 候选评估次数上限，含基线 | quick 为 16（四维全枚举） |
| 外层时间 | `search.max_time_s`、`--max-time-s` | 墙钟预算，只停止发起新评估 | 示例为 0（不限制） |
| 外层提案 | `search.max_proposals` | 算法提案上限 | 示例为 null |
| 内层轮数/时间 | `placement.max_evaluations`、`max_time_s` | 放置搜索独立预算 | strategy 内层 2 次评估 |
| 外层算法 | `search.algorithm`、`--algorithm` | `enumerate` 与三种随机算法 | quick 枚举；strategy 默认退火 |
| 内层算法 | `placement.algorithm` | GeneSim 三算法与八类移动权重 | 仅通用 PIM；CEVA 强制关闭 |
| 优化范围 | `space.{strategy,op,optimization,hardware}`、`placement.moves.weights` | DPU/PP、tasklet、优化开关、硬件字段、移动权重 | quick 四维各 2 值共 16 候选 |
| 优化开关 | `optimization`（顶层） | 四类 pass 开关的基线值 | 缺省全开，旧配置逐字一致 |
| 缓存档位 | `cache.mode`、`--cache-mode` | `fast` 复用完整产物；`strict` 强制重编 | 严格复跑用 strict |

　　实测口径：16 候选一键闭环由用户在干净目录手动执行完成（见 6.2）；单候选约需一次完整图编译、算子编译与仿真。换机器或换数据需重新测量，文档不固化未实测的默认值。

### 2.4 一键入口与三份示例

| 示例 | 用途 | 覆盖环节 |
|------|------|----------|
| `examples/tune/quick.yaml` | CEVA 64 核、四维 16 候选、1 个真实请求 | 配置→图编译→算子编译→仿真→排序 |
| `examples/tune/strategy.yaml` | 通用 PIM、TP/PP/tasklet 搜索、内层放置搜索 | 外层算法、内层搜索与最终重放 |
| `examples/tune/hardware_sweep.yaml` | 扫描每设备核数、容量、时钟与跨簇路由 | 复用 best、按字段重编或只重仿真 |

```bash
# 0. 一键脚本自动激活环境、先 dry-run 再正式调优
scripts/tune_oneclick.sh                       # 默认 quick，16 个候选
scripts/tune_oneclick.sh --dry-run             # 只检查候选组合
scripts/tune_oneclick.sh --config examples/tune/strategy.yaml --output-dir /data/fjg/tmp_strategy

# 1. 手工入口等价命令
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python scripts/tune_pipeline.py --config examples/tune/quick.yaml --dry-run
python scripts/tune_pipeline.py --config examples/tune/quick.yaml

# 2. 续跑、严格复跑与轮次重放（目录换成自己的输出目录）
python scripts/tune_pipeline.py --config examples/tune/quick.yaml --output-dir test-results/tune-quick-4d --resume
python scripts/tune_pipeline.py --evaluate-candidate test-results/tune-quick-4d/best.json --cache-mode strict
python scripts/tune_pipeline.py --replay-rounds --output-dir test-results/tune-quick-4d

# 3. 硬件扫描（先有 quick 的 best）
python scripts/tune_sweep.py --config examples/tune/hardware_sweep.yaml --best test-results/tune-quick-4d/best.json
```

　　完整字段说明、约束与常见报错见 `docs/tune-user-guide-20261113.md`；命令细节与契约边界见 `docs/tune-usage.md`。

## 三、核心原理

### 3.1 关键机制

　　为了达成闭环，需要同时保证三件事：编译、精化与仿真消费同一套硬件参数和优化配置；每个候选有可复算的内容身份和可复用产物；任一阶段都不能消费不属于本候选或本轮输入的产物。现有实现分别把参数写在图导出、成本提取、仿真 YAML 和命令行中，tasklet 不进 B 路产物，优化 pass 列表固定，缓存只按少量字段比较。总体采用单一画像与请求级契约、四层候选搜索、跨仓产物契约三种手段。

　　**单一硬件画像与请求级编译契约**旨在消除参数多源。冻结 dataclass `HardwareProfile` 是唯一参数源：先从画像派生 `PIMHardwareConfig`、`HwBudget`、GeneSim 覆盖值和单 TensorPU 容量，再把编译相关字段摘要与完整画像摘要分开——`compile_profile_sha256()` 进入编译键与放置 sidecar，`profile_sha256()` 进入仿真键与画像契约。tasklet、编译目标与优化开关随 `PIMHardwareConfig` 成为请求级契约，A/B 两路、算子缓存键与 sidecar 都从它派生，不另立第二份状态。这样时钟扫描可复用编译产物，而容量、DPU、WRAM 或开关改变必然触发重编。在本项目中，该机制位于 `contracts/hardware_profile.py:83`（`HardwareProfile`）、`:166`（`to_pim_hardware_config`）、`:195`（`compile_profile_sha256`）与 `contracts/op_contract.py:18`（`PimOptimizations`）。

　　**四层候选搜索与预算续跑**旨在把单次执行变成可迭代、可复现的搜索。`TuneState` 由图策略、tasklet、优化开关、画像四层构成；`CandidateSpace` 实现初始状态、稳定键、单字段邻居与交叉，开关层通过 `space.optimization` 把某个开关升为搜索维度。`run_tune()` 用统一 `Objective` 调用候选执行，按调用序号写响应日志；续跑时逐条回放已提交响应，只对尾部执行新评估，墙钟预算只累计新评估自身耗时。四层任一取值变化都会改变 `compile_key`，不同候选不会复用彼此的编译产物。在本项目中，该机制位于 `tune/candidate.py:57`（`TuneState`）、`:109`（`CandidateSpace`）、`tune/loop.py:188`（`run_tune`）、`:320`（`_search_coverage`）。

　　**编译—仿真产物契约**旨在证明"仿真的就是本轮编译和调优结果"。采用三层绑定：第一层是 `compile_key`/`sim_key`，覆盖模型、原始 IR、轨迹窗口、策略、tasklet、优化开关、画像、源码与工具内容；第二层是放置 sidecar schema 5，逐条声明 PIM MLIR 的哈希、A/B 路由和预算属性；第三层是画像契约 schema 3，绑定 compile key、图策略、计划映射、精化 IR/放置/成本 sidecar 与轨迹摘要。GeneSim 的统一入口 `verify_compiled_artifact()` 一次核对工具链、引用哈希、路由、预算属性与物理范围；最终放置从干净展开图保存，重放时先应用映射再重建 relay 与 stage 视图，最后核对图摘要。在本项目中，该机制位于 `tune/steps.py:191`（`compile_key`）、`:579`（`verify_compiled`）、`genesim/src/run_manifest.py:989`（`verify_compiled_artifact`）、`genesim/src/scheduler/gene_sim_scheduler.py:7865`（保存）与 `:7951`（重放）。

### 3.2 设计取舍

| 取舍点 | 采用方案 | 未采用方案 | 理由（≤30字） |
|--------|----------|------------|---------------|
| 编译缓存目录 | `compile_key/attempt-<id>/` 独立 attempt | 每键一个可变扁平目录 | 严格档要保留旧产物与引用 |
| 候选身份契约 | 画像契约 schema 3 绑定完整候选身份 | 只绑定硬件五元组与容量 | 同硬件不同 TP/PP 会混用产物 |
| 校准报告 | GeneSim `ceva_validation` 唯一契约 | 调优侧另建报告与成熟度 | 两套契约会"一边过一边拒" |
| 校准拟合 | 同类别联合拟合、按字段一次写回 | 按阶段分别写回 | 避免 2× 与 3× 连乘成 6× |
| 单 TensorPU 限额 | 物理容量减系统预留后均分 | 直接按物理容量均分 | 与 GeneSim 容量审计逐字节一致 |
| tasklet 成本缩放 | 只缩计算周期，访存不缩 | 访存一起缩放 | DMA 是共享资源，缩放跌破带宽下界 |
| 优化开关位置 | 顶层 `optimization` 加第四搜索层 | 开关作为独立硬件字段 | 开关属编译请求，随候选进缓存键 |
| 旧产物兼容 | 顶层显式 `version<5` 才走旧分支 | 字段缺失即回退旧逻辑 | 删新字段不能退回宽松校验 |

### 3.3 适用前提与边界

- 前提：模型为 Llama2-7B 或已有模型入口支持的单模型；不满足时：本轮不承诺新增模型族。
- 前提：单机单进程编排；不满足时：不支持多服务器与分布式调度。
- 前提：CEVA 保持编译器一对一映射且禁用内层搜索；不满足时：配置在编译前报错。
- 前提：优化开关层一次只扫描一个字段；不满足时：需要多次运行并隔离输出目录。
- 前提：准确性结论必须有独立校准集与用户阈值；不满足时：结果只能标为 `model_prediction`。

## 四、关键问题与解决方案

　　实施与评审暴露的问题集中在五类：调优参数没有真实进入产物与成本、产物身份不可信、放置缺少编译语义、缓存与续跑语义分裂、校准与扫描结论超出证据。以下按"阻塞闭环的程度"排序，只收录根因不显然、解法有取舍或确实踩过坑的问题。

### 4.1 tasklet 参数不进 B 路代码与仿真成本（来源：tune-bpath-tasklet-dpu-20261113 §二、§六）

　　为了达成"搜索 tasklet 真实改变结果"，tasklet 必须同时进入 B 路 PIMIR、生成的代码和仿真成本。原有实现里 B 路由 `_module()` 拼模块头，只写 `pim.target` 与容量三项，改 tasklet 时 B 路文本完全不变；仿真成本也不随 tasklet 变化，搜索该维度形同虚设。本文采用三段贯通：G 仓 `_bpath_pimir()` 构造复制档请求并复用 `_module_text()`，多 DPU 时写出 `pim.num-dpus`、`pim.num-tasklets` 与 `#pim.placement<kind = replicate>`（单 DPU 不写，旧文本不变）；FlagTree EmitC 降级按 `rowsPerTasklet`（无分块）或 `tilesPerTasklet`（有分块）静态切 M，缺属性按 1 处理；GeneSim 把 tasklet 写进编译签名（版本升到 13），矩阵算子计算周期按有效并行度 `min(tasklet, 行数或块数)` 缩放。实现位于 `genesim_bridge/placement_export.py:295`、FlagTree `LowerPIMToEmitC.cpp:1160`、`genesim/src/scheduler/gene_sim_scheduler.py:4821`、`genesim/src/pim/pim_trace_loader.py:383`。实施中踩过两个坑：第一版只改效率指标的 `estimate_cycle_breakdown()`，而仿真执行走 `estimate_total_cycles()`，分数逐位不变；第二版把访存周期一起缩放，被 GeneSim 按峰值带宽下界拒绝。现在两条入口同源且访存保持原值。由 `tests/test_tune_compile_bridge.py`（25 条）、`test_opcompiler_ops.py` B 路矩阵乘 tasklet 1/2/4 与 numpy 镜像逐元素一致、S 仓 101 条 trace 用例与真实 16 轮（tasklet 4→8 平均时延 −0.840143 秒）验证。

### 4.2 编译优化开关缺少单一来源（来源：tune-optimization-switches-20261113 §一、§六）

　　为了让用户能按需求文档配置"合法优化范围"，开关必须同时作用于 A 路、B 路与成本精化，并进入产物身份。原有实现的 pass 列表固定在代码里，不同开关组合的产物无法区分，也不能作为搜索维度。本文新增 `PimOptimizations`（`constant_fold`、`cse`、`dce`、`loop_tiling`，缺省全开）作为 `PIMHardwareConfig` 字段：`a_path_passes()`/`b_path_passes()` 按开关剔除 pass，缓存键、`compile_key` 与 sidecar 的 `compiler.hardware.optimizations` 记录生效开关，`verify_compiled()` 在仿真前核对 sidecar 开关与候选一致。融合、相位展开与契约校验是目标格式硬要求，不设开关。原理是"开关属于编译请求的一部分，身份随请求走"，换开关必然重编、不会复用旧产物。实现位于 `contracts/op_contract.py:18`、`opcompiler_bridge/pipeline.py:58`、`tune/config.py:118`（`OptimizationSpace`）、`tune/steps.py:615`。由 `test_op_contract`、`test_tune_config`、`test_tune_compile_bridge`、`test_tune_steps` 分组共 433 条用例与一次真实 A 路编译对照（默认写 `pim.tile-*`，全关后不写分块属性）验证。

### 4.3 编译—仿真产物归属漂移（来源：review round19 问题1、round20 问题1、round21 问题1、round22 问题1）

　　为了让仿真只消费属于当前候选和当前输入的产物，需要把画像、策略、映射、实际产物文件和请求轨迹绑定为一个不可分割的身份。原有画像契约 schema 2 只绑定硬件五元组与容量，同硬件不同 TP/PP 的候选可以交换 refined IR、放置 sidecar 与 PIM MLIR 并通过校验；后续评审又发现契约不绑定模型、原始 IR 与轨迹，异模型或 24→25 token 的轨迹可以复用旧键，同一输入对象在准备后被改写文件时仍能发布新结果。本文把契约升级到 schema 3，并新增 `verify_input_snapshot()` 在编译发布前后、仿真前、配置生成、完成清单发布、快速缓存读取与两种重放中按磁盘当前字节复核模型完整指纹、原始 IR 与轨迹窗口；`verify_candidate_identity()` 按当前配置、输入与候选重算 compile key，核对清单输入角色、五个产物路径与内容、精化 IR 派生身份。原理是"内存对象不是信任锚点，磁盘当前字节与同源派生键才是"。实现位于 `tune/inputs.py:202`、`tune/steps.py:308`、`:579`、`genesim/src/run_manifest.py:854`（`_verify_compile_binding`）。round22 的探针显示，校验通过后、消费前改写轨迹、模型或 IR 文件时，正式校验、配置生成、缓存读取、完成发布与真实 pipeline 全部拒绝，合法基线与同内容搬迁通过。

### 4.4 放置缺少编译语义约束（来源：review round01 问题2、round10/11 问题3）

　　为了让内层搜索只改变资源而不破坏编译器确定的切分，需要把 TP 子节点、融合单元、本地形状、内核和通信纳入合法性判据。原有 `materialize()` 只深拷贝并修改资源三元组，TP/融合/通信/容量都没有进入放置合法性，搜索评分可能对应一个编译器不支持的图。本文采用"编译语义冻结 + 干净图重建"：TP 子节点与编译器钉住的算子目标冻结，融合宿主与吸收节点成原子单元，本地形状与内核原样保留，每次物化都从未附加通信边的展开图重建 relay、集合通信与 stage 视图，再检查容量。原理是把"编译器支持该算子"提升为"候选放置保持编译语义"。实现位于 `genesim/src/autotune/placement.py`（`_validate_compilation_constraints`、`materialize_complete`）与 `genesim/src/scheduler/gene_sim_scheduler.py:7661`（`_finalize_placed_graph`）。七类移动逐类真实 pipeline 重放成功；`layer_block_relocate` 在固定 TP4 图上因每层含钉住的 TP 分片而无合法目标，按 100 次采样、0 提案显式记录，未误计成功。

### 4.5 外置产物的工具链身份缺失（来源：review round01 问题1、round12～16）

　　为了让 GeneSim 相信仿真消费的是本轮编译工具，需要把编译器与四类实际工具的内容身份写进产物。原有实现从产物路径反推源码根，外置输出目录不在源码树内时直接失败；后来虽增加声明字段，伪造摘要、空前缀和删除工具链块仍可绕过。本文采用"编译期显式声明 + 消费端按本机权威路径核对内容"：导出 sidecar 写 `compiler.toolchain`（图编译源码摘要、编译指纹、`triton-opt`/`libtriton`/`mlir_translate`/`gcc` 内容哈希），GeneSim 按环境变量、图仓库 `paths.json`、声明前缀的顺序解析首个有效工具并核对内容，本机存在工具时空证据失败关闭。原理是把"路径存在"换成"内容相等"，路径与时间戳不参与身份。实现位于 `opcompiler_bridge/driver.py:137`（`_tool_identity`）、`:175`（`compiler_identity`）与 `genesim/src/run_manifest.py:530`（`_verify_installed_tools`）、`:717`（`_declared_toolchain`）。真实 quick 产物声明与当前 `compiler_identity()` 一致；伪造 `gcc` 与异 tasklet 的真实 CEVA 启动反例均在创建调度器前退出码 1。

### 4.6 残缺缓存、伪完成与续跑对账（来源：review round01 问题3/4、round06 问题4/5、round08 问题2/4）

　　为了让缓存只复用完整且未篡改的产物、续跑只回放同一运行身份下的已提交响应，需要定义完成协议与提交点。原有快速档只凭 `results/summary.json` 命中，编译清单可以缺少 `placed.ir` 甚至把角色别名到另一个文件；续跑只按候选键回放，异配置或 fast↔strict 切换会复用旧结果，崩溃在轮次写入后、响应写入前的孤儿轮次会计入排序。本文采用"响应是唯一提交点 + 必需角色完成协议"：编译目录必须包含 `partition_plan.json`、`placed.ir`、`placement.json`、`refined.ir`、`cost.json` 五个规范角色与 sidecar 引用的全部 PIM MLIR；仿真完成清单必须包含 summary、autotune 指标、配置、manifest、画像契约、最终放置与全部 trace，trace 算子集合与最终挂 PU 目标精确相等；恢复前逐调用核对响应与轮次的状态、分数、键和失败原因，孤儿轮次移入 `orphan_rounds.jsonl` 并重跑，坏目录改名为 `invalid-attempt-*` 隔离。实现位于 `tune/report.py:354`（`verify_compile_products`）、`tune/steps.py:1405`（`_publish_sim_complete`）、`tune/loop.py:359`（`_reconcile_rounds`）。真实运行中 trace 数与挂 PU 目标精确相等（历史两候选 7714/7138），续跑为逻辑 2 次、实际 0 次、回放 2 次，最优不变。

### 4.7 校准与结果分级失真（来源：review round01 问题10、round03 问题1/2、round07/08）

　　为了让"已验证性能"只来自独立证据，需要把报告结构、逐点证据、统计、阈值、适用范围和产物绑定收敛到唯一契约。原有调优侧另建报告与成熟度，出现调优侧 `passed`、GeneSim 判"缺证据"的分裂；拟合按类别汇总，单点会改动未测原语，多个身份还会把同一字段连乘成 6×。本文采用 GeneSim `ceva_validation` 唯一构造与校验入口：报告保存阈值与训练点，升级时重算拟合、冻结预测、统计与阈值；训练与留出按完整测点身份互斥，禁止改名两用；同一画像字段先汇总冻结值，同值合并、异值在写回前拒绝；缺独立实测、用户阈值、适用产物或后端证据时保持 `model_prediction`。原理是把"报告标签"换成"可复算的证据链"。实现位于 `tune/calibration.py`、`genesim/src/ceva_validation.py:563`（`measurement_identity`）、`:1402`（`report_evidence_error`）。真实 quick 报告可通过 GeneSim 运行前核对，但因参数来源仍为假设，成熟度保持 `model_prediction`。

### 4.8 硬件扫描口径与 best 绑定（来源：review round01 问题13、round10 问题2、round11 问题3）

　　为了让扫描结论只对应同一负载与同一基线，需要使用 best 自己的配置与输入重算历史键，并精确到实际变化的参数叶子。原有扫描只恢复候选状态，不核对成功状态、输入内容、基线画像与生效配置，跨负载 best 可以进入扫描；来源降级按父键整组覆盖，未扫描的原语会失去原有证据。本文采用 `verify_best_reference()` 在发布输入与写生效配置前核对成功状态、有限分数、模型/IR/轨迹/请求数、基线画像与生效配置语义，并用 best 自己的配置重算 `compile_key`/`sim_key`；来源键从 `genesim_overrides()` 的叶子差异派生，父键只表示整组字段本身；只改时钟等仿真字段时复用编译产物，改容量或核数时按同一规则重编。实现位于 `tune/sweep.py`（`verify_best_reference`、`_mark_scanned`）。两点扫描复用同一 compile key、产生两个 sim key；历史 16 点扫描含 4 个编译键、16 个仿真键、12 个重编点，Pareto 独立复算一致。

## 五、改动说明

### 5.1 改动总览

　　本次共修改 42 个受版本控制文件（G 17 个、S 24 个、F 1 个），受控文件代码增加 7537 行、删除 555 行（G +1563/−135、S +5954/−415、F +20/−5，统计范围为三仓工作区 `git diff HEAD`）；另新增 38 个非文档文件（G 33 个、S 5 个），其中 `tune/` 包 10 个文件共约 6900 行。核心改动集中在 G 的 `tune/`、`contracts/`、`genesim_bridge/`、`opcompiler_bridge/`，S 的 `src/run_manifest.py`、`src/scheduler/gene_sim_scheduler.py`、`src/pim/`，以及 F 的 EmitC 降级。

### 5.2 文件改动清单

| 文件路径 | 改动类型 | 改动内容（≤30字） | 对应需求 |
|----------|----------|-------------------|----------|
| `contracts/op_contract.py` | 修改 | `PimOptimizations`、编译目标、tasklet 可空 | R4 |
| `contracts/hardware_profile.py` | 新增 | 画像、派生、摘要、校验 | R2/R5 |
| `contracts/profiles/{generic,ceva_64core_4e,ceva_128core_4e}.json` | 新增 | 三份基线画像 | R2/R5 |
| `genesim_bridge/placement_export.py` | 修改 | 候选导出、schema 5、B 路 DPU/tasklet | R3/R4 |
| `genesim_bridge/refine_profile.py` | 新增 | 精化前置策略/映射/产物契约 | R4/R5 |
| `genesim_bridge/cost_extractor.py` | 修改 | 成本链消费请求硬件与本地形状 | R4 |
| `genesim_bridge/flagtree_driver.py` | 修改 | 硬件参数与开关进算子编译 | R4 |
| `genesim_bridge/paths.py` | 修改 | 新增 GeneSim 根路径解析 | R1 |
| `opcompiler_bridge/driver.py` | 修改 | tasklet 同源、内容指纹、B 路模块头 | R4 |
| `opcompiler_bridge/pipeline.py` | 修改 | A/B 路 pass 与开关、full_m/n/k | R4 |
| `runtime/compile.py` | 修改 | 请求 tasklet、KV 峰值与并发上限 | R3/R4 |
| `runtime/kernels.py` | 修改 | 内核缓存键改稳定 JSON 串 | R4 |
| `memory/mem_planner.py` | 修改 | 逐 TensorPU 常驻分配台账 | R5 |
| `scripts/export_pp_placement.py` | 修改 | 复用候选导出，保持旧默认 | R3 |
| `scripts/run_full_pipeline.py` | 修改 | 轨迹白名单与输出路径 | R5 |
| `tune/{__init__,_genesim,config,candidate,inputs,steps,loop,report,calibration,sweep}.py` | 新增 | 调优包：配置到报告全链路 | R1～R8 |
| `scripts/tune_pipeline.py` | 新增 | 调优统一命令行 | R1/R6 |
| `scripts/tune_oneclick.sh` | 新增 | 一键激活、检查、执行 | R9 |
| `scripts/tune_sweep.py` | 新增 | 硬件扫描入口 | R8 |
| `scripts/calibrate_profile.py` | 新增 | 校准入口与写回 | R7 |
| `examples/tune/{quick,strategy,hardware_sweep}.yaml`、`trace_quick.json` | 新增 | 四维示例与真实请求 | R9 |
| `tests/test_hardware_profile.py` 等 10 个 `tests/test_tune_*.py` | 新增 | 调优与画像单测 | R1～R8 |
| `tests/test_{mem_planner,op_contract,opcompiler_linear,opcompiler_ops,placement_export}.py` | 修改 | 容量台账、契约、B 路矩阵乘回归 | R4/R5 |
| `genesim/src/hardware_profile.py` | 新增 | 生效画像与产物摘要校验 | R5 |
| `genesim/src/ceva_validation.py` | 新增 | 校准证据唯一契约 | R7 |
| `genesim/src/run_manifest.py` | 修改 | 工具链、契约、重放与清单 | R5 |
| `genesim/src/{main,config_loader}.py` | 修改 | 启动前产物与画像校验 | R5 |
| `genesim/src/pim/{capacity,pim_core,pim_trace_loader,pimir_trace}.py` | 修改 | 容量预留、原生台账、tasklet 成本 | R5/R4 |
| `genesim/src/backend/pim_backend.py` | 修改 | 原生后端身份汇总 | R5 |
| `genesim/src/autotune/{placement,objective}.py` | 修改 | 编译约束与完整物化 | R2/R5 |
| `genesim/src/scheduler/gene_sim_scheduler.py` | 修改 | 通信物化、重放、签名 v13 | R2/R4/R5 |
| `genesim/scripts/refine_ir_with_flagtree.py` | 修改 | `--profile/--force-compile` | R4 |
| `genesim/tests/sim/test_{hardware_profile,refine_profile,tune_placement_audit}.py` | 新增 | 画像、精化、放置审计测试 | R4/R5 |
| `genesim/tests/sim/` 其余 12 个测试文件 | 修改 | 契约矩阵、原生台账、成本缩放 | R2/R4/R5 |
| FlagTree `lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp` | 修改 | EmitC 降级按 tasklet 静态切 M | R4 |
| `docs/tune-usage.md`、`docs/tune-user-guide-20261113.md` | 新增 | 使用说明与用户手册 | R9 |

### 5.3 核心函数清单

| 函数/类 | 位置 | 改动类型 | 作用（≤30字） | 调用方 |
|---------|------|----------|---------------|--------|
| `load_config()` | `tune/config.py:344` | 新增 | 解析并校验唯一生效配置 | `tune_pipeline.main()` |
| `CandidateSpace` | `tune/candidate.py:109` | 新增 | 四层候选枚举与邻居生成 | `run_tune()`、dry-run |
| `run_tune()` | `tune/loop.py:188` | 新增 | 外层搜索、预算、续跑与汇总 | `tune_pipeline.main()` |
| `build_candidate()` | `tune/steps.py:458` | 新增 | 编译缓存、原子发布与精化 | `evaluate_candidate()` |
| `verify_compiled()` | `tune/steps.py:579` | 新增 | 仿真前核对候选与产物 | `evaluate_candidate()` |
| `build_sim()` | `tune/steps.py:712` | 新增 | 生成候选专属仿真配置与契约 | `evaluate_candidate()` |
| `evaluate_candidate()` | `tune/steps.py:882` | 新增 | 单候选完整执行与记录 | `run_tune()`、重放 |
| `verify_input_snapshot()` | `tune/inputs.py:202` | 新增 | 按磁盘字节复核输入快照 | 编译、仿真、缓存、重放 |
| `export_placement_for_strategy()` | `genesim_bridge/placement_export.py:867` | 新增 | 复用生产图编译与导出 | `build_candidate()`、旧 CLI |
| `_bpath_pimir()` | `genesim_bridge/placement_export.py:295` | 修改 | B 路构造复制档请求写 DPU/tasklet | 放置导出 |
| `verify_compiled_artifact()` | `genesim/src/run_manifest.py:989` | 新增 | 统一产物契约核验 | 启动、清单、重放 |
| `_tasklet_parallelism()` | `genesim/src/pim/pim_trace_loader.py:383` | 新增 | 矩阵算子有效并行度 | 两条周期估算入口 |

### 5.4 数据结构与接口变更

　　为了表达跨仓的候选身份与产物归属，需要统一记录硬件参数、优化开关、外层候选和轮次结果。原有的 `PIMHardwareConfig` 只覆盖硬件五元组，`TuneState` 只有三层，放置 sidecar 的 B 路条目不带 DPU/tasklet，同硬件异开关或异策略的结果无法区分。因此新增 `PimOptimizations` 与画像契约 schema 3，扩展 `TuneState` 与候选编号。

**`PimOptimizations`**（`contracts/op_contract.py:18`）
```python
@dataclass(frozen=True)
class PimOptimizations:
    constant_fold: bool = True
    cse: bool = True
    dce: bool = True
    loop_tiling: bool = True
```
作为 `PIMHardwareConfig.optimizations` 字段随编译请求传递；缺省全开等价于改动前行为，旧调用方无需改动，旧 payload 缺该块时按全开读回。

**候选状态 `TuneState`**（`tune/candidate.py:57`）
```python
@dataclass(frozen=True)
class TuneState:
    strategy: StrategyChoice          # num_dpus, num_stages（TP 宽度派生）
    op: OpKnobs                       # num_tasklets
    profile: HardwareProfile
    optimization: PimOptimizations | None = None   # None 沿用顶层配置
```
四层取值共同构成候选；候选编号带开关位，例如 `tp4_pp2_dpu8_tl8_opt1011_ceva_64core_4e_p8f99f4a`（`opt1011` 依次为四类开关的布尔串）。旧记录不写开关时按顶层配置解释，行为不变。

**B 路 PIM MLIR 模块属性**（`genesim_bridge/placement_export.py:295`）
```mlir
module attributes {"pim.dma-align" = 64 : i32, "pim.mram-bytes" = 8000000000 : i64,
  "pim.num-dpus" = 8 : i32, "pim.num-tasklets" = 4 : i32,
  pim.placement = #pim.placement<kind = replicate, numDpus = 8>,
  pim.target = "pim:v1", "pim.wram-bytes" = 65536 : i64} {
```
多 DPU 时 B 路与 A 路一样携带 DPU/tasklet；单 DPU 不写这些属性，文本与旧产物一致。契约允许 B 路缺省 DPU/tasklet，但写出后必须与 `compiler.hardware` 完全一致。

**画像契约 schema 3 与最终放置**（`tune/steps.py:712`（`build_sim` 契约构造）、`genesim/src/scheduler/gene_sim_scheduler.py:7865`）
```json
{
  "schema_version": 3,
  "profile_sha256": "...",
  "compile": {
    "compile_key": "...",
    "hardware": {"num_dpus": 8, "num_tasklets": 4, "compiler_target": "pim:v1",
                  "optimizations": {"constant_fold": true, "cse": false, "dce": true, "loop_tiling": true}},
    "strategy": {"name": "tp4_pp2", "num_dpus": 8, "num_stages": 2,
                 "tp_width": 4, "dpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]},
    "dpu_to_cluster": [[0, 0]],
    "artifacts": {"refined_ir_sha256": "...", "placement_sidecar_sha256": "...",
                  "trace_sha256": "...", "trace_num_requests": 1}
  }
}
```
最终放置保存 `inputs_sha256`、`execution_graph_sha256`、`final_graph_sha256` 与执行算子到资源三元组的映射，目标来自干净展开图。新增字段对旧命令不可见；显式旧 schema 产物按旧分支兼容。

## 六、验证

### 6.1 验证环境

　　代码基线与文档头部一致（三仓均含未提交改动）；运行环境为 Linux，Python 3.10.20、torch 2.9.1、transformers 4.57.6，GeneSim 使用 `uv` 虚拟环境，FlagTree 使用重新编译的 `triton-opt`。本次为文档改写，未执行任何命令；下表全部结果由用户手动执行并记录于过程文档，属复用已有验证。

### 6.2 验证命令与结果

　　验证方法分四层：调优与画像单测覆盖配置、候选、输入、执行、报告、校准与扫描；GeneSim 仿真（sim）套件覆盖画像契约、放置审计、产物清单、成本缩放与重放；dry-run 验证前置校验与候选组合；真实闭环用四维 16 候选验证端到端，并用严格复跑与身份核对确认确定性。单测定位局部缺陷，端到端运行证明各阶段真实消费同一批产物。

| 验证项 | 命令 | 记录结果 | 结论 | 证据状态与来源 |
|--------|------|----------|------|----------------|
| 调优单测 | `python -m pytest tests/test_hardware_profile.py tests/test_tune_config.py tests/test_tune_candidate.py tests/test_tune_inputs.py tests/test_tune_steps.py tests/test_tune_loop.py tests/test_tune_report.py tests/test_tune_calibration.py tests/test_tune_sweep.py tests/test_tune_compile_bridge.py -q` | round22 口径 410 条全部通过；追加优化开关与 B 路 tasklet 用例后分组执行全部通过（约 433 条） | 通过 | 复用已有验证（用户手动执行）；implement §4.1、tune-optimization-switches §七、tune-bpath §五 |
| G 主链全量回归 | `python -m pytest tests/ -q -k "not llama2_7b"` | 最近一次全量为 2100 passed、8 skipped、49 deselected（round21）；round22 后仅改调优包与 docs，按纪律未重跑 | 通过 | 复用已有验证；implement §4.3 |
| GeneSim sim 套件 | `cd /data/fjg/flagOS/genesim && ./run.sh --test sim` | 49/49 个测试文件，1564 条用例、1 条跳过、实际通过 1563 条；追加 tasklet 成本用例后 `test_pim_trace_loader`、`test_pimir_trace` 等 101 条通过 | 通过 | 复用已有验证；implement §4.3、tune-bpath §6.4 |
| 三份 dry-run | `python scripts/tune_pipeline.py --config examples/tune/quick.yaml --dry-run` 等 | quick 列出 16 个合法候选（四维），strategy 与 hardware_sweep 正常；非法组合给出拒绝原因 | 通过 | 复用已有验证；tune-search4d §四、user-guide §十四 |
| 真实四维闭环 | `scripts/tune_oneclick.sh examples/tune/quick.yaml` | 16/16 候选成功，`stop_reason=search_space_exhausted`；16 次新 pipeline、0 次缓存命中、16 次最终映射重放，`real_search=true`；最优 `tp4_pp2_dpu8_tl8_opt1011` 为 7.5950386510624055 秒 | 通过 | 复用已有验证（用户手动执行）；tune-search4d §5.1 |
| 单维影响对照 | 同上运行的逐对样本 | tasklet 4→8 平均 −0.840143 秒；DPU 4→8 平均 −9.486760 秒；PP 2→4 平均 +9.486816 秒；`cse` 开关在本负载不影响分数 | 通过 | 复用已有验证；tune-search4d §5.3 |
| 严格复跑 | `python scripts/tune_pipeline.py --evaluate-candidate test-results/tune-quick-4d/best.json --cache-mode strict` | 强制重编并新开仿真尝试目录，分数 7.5950386510624055，相对容差 1e-9 内一致 | 通过 | 复用已有验证；tune-search4d §5.1 |
| 身份与契约核对 | 独立脚本逐候选断言 | 16 个组合、`compile_key`、`sim_key` 两两不同；sidecar 的 DPU/tasklet/开关/策略与候选逐项一致；A/B 两路 PIMIR 均带候选的 `pim.num-dpus/-tasklets`；trace 的 tasklet 与 GEMM 有效并行度等于候选值；best 分数等于最小值 | 通过 | 复用已有验证；tune-search4d §5.2 |
| B 路 tasklet 真实编译 | 真实 `eltwise` 与矩阵乘编译 | 8 DPU/tasklet=4 产出复制档属性；tasklet 4 与 8 的文本和 SHA 不同；B 路矩阵乘 tasklet 1/2/4 的 M 循环块数为 1/2/4，与 numpy 镜像逐元素一致 | 通过 | 复用已有验证；tune-bpath §五、§6.4 |
| FlagTree 编译链接 | 重建 `triton-opt` | `LowerPIMToEmitC.cpp` 编译并通过 `triton-opt` 链接 | 通过 | 复用已有验证；tune-bpath §6.4 |
| 历史真实运行 | 三算法、16 点扫描、七类移动、两候选 quick 续跑/轮次重放 | SA/GA/Tabu 各 2 次新评估且最优一致；扫描 16/16 成功；七类移动真实重放成功；续跑 0 执行、2 回放 | 通过 | 复用已有验证；implement §4.3（round10/11/22，产物存于 `/tmp` 各轮目录） |
| 未执行项 | `python -m pytest tests/ -q -k "llama2_7b"`；CEVA 128 核整轮 | 7B 子集约 2.5 小时、128 核整轮约 10 小时，本轮未执行 | 未执行 | 不作为已通过结论；implement §六 Q5/Q6 |

　　复现命令（输出目录按需替换；每候选含一次完整编译与仿真，16 轮为真实成本）：

```bash
# 一键：激活环境 → dry-run → 正式 16 轮
scripts/tune_oneclick.sh

# 分步等价
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python scripts/tune_pipeline.py --config examples/tune/quick.yaml --dry-run
python scripts/tune_pipeline.py --config examples/tune/quick.yaml
python scripts/tune_pipeline.py --evaluate-candidate test-results/tune-quick-4d/best.json --cache-mode strict
python scripts/tune_pipeline.py --replay-rounds --output-dir test-results/tune-quick-4d
cd /data/fjg/flagOS/genesim && ./run.sh --test sim
```

### 6.3 验收标准对照

| 验收标准（来自需求文档） | 验证方法 | 结论 | 证据 |
|--------------------------|----------|------|------|
| R1 配置复现与前置报错 | 未知字段/负预算/缺画像/双轨迹源用例与 dry-run | 满足 | `tests/test_tune_config.py`；6.2 dry-run 行 |
| R2 三种算法、八类移动、参数产物断言 | 候选测试、放置审计矩阵、真实搜索 | 满足 | `test_tune_candidate.py`；历史三算法与七类重放；16 轮真实闭环 |
| R3 映射回读、原子发布和失败隔离 | 编译产物角色、哈希、坏 attempt 隔离 | 满足 | `test_placement_export.py`（64 条）；编译目录 |
| R4 A/B 参数、形状、缓存与严格失败 | 桥接测试、真实 PIM MLIR 属性、numpy 对拍 | 满足 | `test_tune_compile_bridge.py`（25 条）；B 路矩阵乘对拍 |
| R5 五类篡改反例、CEVA 固定映射、通用最终重放 | 契约反例、启动校验、重放 | 满足 | round18～22 探针；16 候选身份核对 |
| R6 两候选闭环、续跑和严格指标一致 | 16 候选闭环、严格复跑、历史续跑/轮次重放 | 满足 | 6.2 对应行 |
| R7 样本复算、留出集和成熟度 | 校准测试与报告核对 | 部分满足 | 机制通过；缺独立实测与用户阈值 |
| R8 扫描复算与图表重生成 | 扫描测试与真实扫描 | 满足 | `test_tune_sweep.py`（43 条）；两点实测与历史 16 点 |
| R9 三份示例实际运行 | dry-run、一键脚本与用户手册 | 满足 | 16 候选真实运行；`docs/tune-user-guide-20261113.md` |

### 6.4 评审情况

　　本需求共进行 23 轮评审，前 22 轮形成 114 项报告项（严重 23、高 52、中 39），全部确认成立并按根因修复；第 23 轮复查为 0 项。评审结束后（2026-11-13）追加实施了三项增量（B 路 tasklet 贯通、优化开关、四维搜索与一键入口），随做随验并记录于三份 11-13 文档，未再走独立评审轮。各轮核心问题及最终状态如下。

| 评审轮次 | 核心问题 | 修复状态 | 最终证据 |
|----------|----------|----------|----------|
| 第 1～3 轮 | 外置产物身份、放置缺编译约束、续跑语义、残缺缓存、校准跨仓断裂 | 已修复 | implement §7.1；对应单测与真实 strategy |
| 第 4～6 轮 | 校准外层证据、容量台账、输入身份、完成清单协议、响应轮次对账 | 已修复 | implement §7.1；`test_tune_steps.py`、`test_tune_loop.py` |
| 第 7～9 轮 | 训练留出改名两用、共享周期连乘、编译目标被环境篡改、非 PIM 残留 trace | 已修复 | implement §7.1；桥接与清单用例 |
| 第 10～12 轮 | 原生回退错绑、扫描 best 身份、算法仅回放、reset 生命周期、工具链对账 | 已修复 | implement §7.1；原生与扫描探针 |
| 第 13～15 轮 | 空前缀绕过工具核验、精化未核对策略预算、空工具证据、通用共享映射误拒 | 已修复 | implement §7.1；伪造 E2E 反例退出码 1 |
| 第 16～18 轮 | 删除契约块、预算属性缺失、共享引用绕过、通用路径绕过核验 | 已修复 | implement §7.1；`/tmp/round16～18` 探针 |
| 第 19～22 轮 | 画像未绑定编译产物、漏绑图策略、输入身份未绑定、快照与磁盘分离 | 已修复 | implement §7.1；`/tmp/round19～22` 探针 |
| 第 23 轮 | 无仓库内可修复问题 | 无需修复 | review-tune-loop-20261112-round23 |
| 11-13 增量 | B 路 tasklet 两处缩放坑、开关单一来源、四维身份 | 已随做随验 | tune-bpath §六、tune-optimization-switches §七、tune-search4d §5 |

　　当前未修复 0 条；R7 的外部证据缺失按已确认外部约束披露，不列为仓库缺陷。

## 七、与设计的差异

| 设计条目 | 设计方案（≤20字） | 最终实现（≤20字） | 差异原因（≤30字） | 是否经确认 |
|----------|-------------------|-------------------|-------------------|------------|
| design §4.3 | 三层候选空间（策略/算子/硬件） | 四层：增加编译优化开关层 | 2026-11-13 追加需求"开关特定优化" | 是（三份 11-13 文档） |
| design §5.1 | FlagTree 无修改，仅回归 | EmitC 降级按 tasklet 静态切 M | tasklet 需真实进入 B 路生成代码 | 是（tune-bpath §6.1） |
| design §4.5 | B 路只含目标与容量属性 | 多 DPU 时写 DPU/tasklet 与复制档 | 审计需证明 B 路按候选参数编译 | 是（tune-bpath §三） |
| design §4.2 | 运行形状旧开关无实际读取不开放 | 四类优化开关开放并可搜索 | 用户需求要求可配置优化范围 | 是（tune-optimization-switches） |
| design §4.8 | 编译缓存扁平目录 | `compile_key/attempt-<id>/` | 严格档保留旧产物与引用 | 是（评审驱动） |
| design §4.6 | 画像契约扁平期望值 | schema 3 绑定完整候选身份 | 同硬件异 TP/PP 会混用产物 | 是（评审驱动） |
| design §4.7 | 运行前后核对输入 | `verify_input_snapshot()` 全生命周期复核 | 内存对象与磁盘内容会分离 | 是（评审驱动） |
| design §4.9 | 调优适配器生成报告 | GeneSim `ceva_validation` 唯一契约 | 两套契约会分裂 | 是（评审驱动） |
| design §4.9 | 按阶段分别拟合写回 | 同类别联合拟合、按字段一次写回 | 避免 2× 与 3× 连乘 | 是（评审驱动） |
| design §4.2 | 单 PU 限额按物理容量均分 | 先扣系统预留再均分 | 与 GeneSim 容量审计一致 | 是（评审驱动） |
| design §4.10 | ΔT/T0 以基准画像为 T0 | 基准不在扫描点时取首个成功点并标注 | 扫描点可能不含基准 | 否（待确认） |
| design §4.1 | 画像内基配置相对画像文件 | 找不到再相对 GeneSim 根 | 画像与基配置跨仓存放 | 否（待确认） |
| design §4.4 | 新导出函数固定产物名 | 增加可选 `file_stem` 保持旧名 | 旧 CLI 不回归 | 否（待确认） |

## 八、当前遗留问题

| 编号 | 问题描述（≤30字） | 影响范围（≤20字） | 来源 | 状态 |
|------|-------------------|-------------------|------|------|
| L1 | 缺 R7 独立实测数据与用户 median/p90/max 阈值 | 目标 5 无法关闭，成熟度只能 `model_prediction` | implement §五 K3/Q2；request §八 | 未修复（外部证据） |
| L2 | llama2_7b 子集（约 2.5 小时）与完整 GML 32 层导出未重跑 | 大规模回归覆盖不足 | implement §4.3、Q6 | 未执行 |
| L3 | CEVA 128 核整轮（约 10 小时）未执行 | 128 核画像只有配置与单测覆盖 | implement §六 Q5 | 未执行 |
| L4 | `layer_block_relocate` 在 7B 图上无合法目标，单次提案成本高 | 该类移动的真实重放不可用 | implement §五 K23/Q10 | 未关闭（上游采样器限制） |
| L5 | 原生 DRAMsim3/AttAcc 真实端到端计时未覆盖 | 原生与回退计时只能靠单测与探针 | implement §4.3；round23 | 未验证 |
| L6 | 离线无法区分真正的旧 schema 与主动降级版本号 | 严格契约的离线可证明性边界 | implement §五 K32/K36 | 已规避（披露边界） |
| L7 | CEVA 每核"8 GB"按 8,000,000,000 B 假设 | 容量结论带假设 | request §八；Q1 | 待确认 |
| L8 | predictor 训练与检查点无交付证据，示例默认 pipeline | predictor 筛选只能作可选路径 | design §九；implement §五 | 未验证 |
| L9 | `tune/config.py` 与 `tune/steps.py` 体量较大 | 可维护性 | implement §五 K7 | 未关闭（非阻塞） |
| L10 | 优化开关层一次只扫描一个字段，多字段需多次运行 | 开关组合搜索不便捷 | tune-search4d §六 | 已知限制 |
| L11 | 关闭 `loop_tiling` 后本地形状推导读不到输出分块 | 该开关不宜作搜索维度 | tune-optimization-switches §八、user-guide §5.3 | 已知限制 |
| L12 | 访存周期不随 tasklet 缩放；B 路 tasklet 切分仅覆盖 `pim.matmul` | tasklet 收益只体现在矩阵计算周期 | tune-bpath §6.5 | 保守口径，待实测收紧 |

　　以上遗留问题均已在过程文档中记录。tasklet 成本模型变化后，quick/strategy 的历史分数基准（如 10.304023382370014 秒）不能与当前 7.5950386510624055 秒直接对比。目标 5 的关闭条件是独立观测、用户阈值与适用范围证据齐备；严格复跑的 1e-9 只证明确定性，不代表硬件精度。
