# 技术文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：techdoc-qwen3-moe-20261112
> 创建日期：2026-11-12
> 关联文档：request-qwen3-moe-20261105、design-qwen3-moe-20261105、implement-qwen3-moe-20261105、review-qwen3-moe-20261109-round01 至 review-qwen3-moe-20261112-round20（共 20 份）
> 关联项目：flagos-pim-compiler、genesim、flagOS-installers/FlagTree
> 代码基线：flagos-pim-compiler `pengcheng@b2f07ca`、genesim `pengcheng@434e147`、FlagTree `feat/ceva-with-paged-attn@6e34f0299`；改动已提交（compiler `5b5d539`、genesim `6d24d30`、FlagTree `3690b47c2`）
> 状态入口：qwen3-moe-final-20261112（文档-代码一致性核对与当前状态汇总）
> 核心目标：让 `qwen3_moe`（Qwen3-30B-A3B 及其 4 层切片）走通「注册表加载 → 图编译切分 → FlagTree 算子编译 → GeneSim 仿真」链路，4 层真实权重与 HF 逐 token 一致，并交付动态分派与专家并行。

## 一、概述

### 1.1 背景与问题
　　本需求的核心目标是让 Qwen3 MoE（`model_type="qwen3_moe"`，代表型号 Qwen3-30B-A3B：48 层、128 专家、top-8、hidden 2048、专家宽 768）进入 PIM 推理链路。现有基础是链路已支持 llama、qwen2、qwen3 dense 三类架构，具备黑名单式主机下沉、主机通用回退、TP/PP 切分与三区内存规划。关键缺口有四处：`qwen3_moe` 在启动期被白名单拒绝；HF `Qwen3MoeSparseMoeBlock.forward` 用 `expert_hit.nonzero()` 与按 token 子集 gather 做数据依赖分派，`torch.export` 直接报 data-dependent expression；专家权重的放置、容量与路由规则缺失；MoE 的分派、专家并行与代价模型缺失。需求边界是保留既有 llama/qwen2/qwen3 dense 行为不变，不包含 vLLM serve 实测、235B、权重量化、滑窗 MoE 与 tied 词嵌入。

### 1.2 方案摘要
　　围绕核心目标，总体采用「静态稠密模拟 + 运行期动态分派」的两阶段路线。`backend/architectures.py` 在加载后把每个 MoE 块的 forward 换成静态形状版本，用 top-k 选中掩码完成全专家 masked 计算与门控加权；图编译复用既有 TP/PP 规则并补 router 与专家规则；FlagTree 把每个设备算子编译成 pim mlir；GeneSim 加载骨架 IR 与放置 sidecar，生成 trace 与 summary。两仓通过 sidecar 的 `moe_config`、结构指纹、容量三件套与 B 路编译单位建立同一份契约，最终实现 4 层真实权重与 HF 逐 token 一致，并交付动态分派与 PP×EP 专家并行。

### 1.3 核心目标完成度

| 编号 | 核心目标 | 状态 | 证据 | 缺口或外部阻塞 |
|------|----------|------|------|------------------|
| P0-1 | shape-static MoE 导出与架构注册 | 已完成 | `backend/architectures.py:22`（`make_static_moe_forward`）、`:210`（`qwen3_moe` 注册）；4 层目标命令 36 passed、1 skipped（日志 `/tmp/r20-fix-4l-tests.log`）；小模型 `test_qwen3_moe_generate_matches_hf` | 8B 无权重，SKIP 未覆盖 |
| P0-2 | 图规则支持 MoE 算子链 | 已完成 | `graph/partition.py:55-66`、`graph/strategy.py:13-28`；4 层 A 阶段 3681 算子、1557 GEMM；放置 sidecar 3163 条（1557 GEMM + 1606 算子级） | 无 |
| P0-3 | 内存规划容纳专家权重 | 已完成 | 4 层当前代码集群常驻 `3,140,834,304 / 8,589,934,592`（36.56%），`fits=True`；48 层当前代码（2026-11-13 重跑）dpu0..3 所需/可用 `8,258,576,576 / 8,522,825,728 B`、dpu4..7 `7,791,833,024 / 8,522,825,728 B`；GeneSim 集群常驻 `8,412,147,712 / 8,589,934,592`（97.93%），`fits=True` | 无；48 层数值对拍另列 |
| P0-4 | MoE 执行路径（内核） | 已完成 | dense_sim 全链 `63.637 s / 12.493 t/s`；`runtime/exec_plan_gen.py:169`、`backend/hal_numpy.py:233`；4 层逐 token 对拍通过 | 8B 未覆盖 |
| P0-5 | 真实 30B 端到端 | 部分完成 | 4 层真实权重当前代码 36 passed、1 skipped；2026-11-13 当前代码 48 层主链 exit 0（A/BCD/A2/E，成本见 G5）；48 层数值历史：round01 23 passed、1 skipped（25,426.83 s），round06 26 passed、1 skipped（17,254.56 s） | 48 层数值未用当前代码重跑；round02 的 `full30b-r2` 被 systemd-oomd 杀死（共享 CEVA 内存压力，环境阻塞）；8B SKIP |
| P1-1 | 主机动态分派原语 | 已完成 | 提交前跳过与层域门控；token 与稠密模拟完全一致；4 层执行期 prefill 1.148×、decode 1.202×（历史实测，用户已接受，本轮未重测） | 剩余执行后端优化另立需求 |
| P1-2 | 专家并行放置维度 | 已完成 | PP×EP 落地（设计 §九.4，经用户确认）；`graph/strategy.py:40`（`ep_width`）、`:84`（`ep_dpu_of_expert`）；16 DPU 用例与映射均衡断言 | TP×EP 复合放置未实现；仅小模型验证 |
| G1 | GeneSim MoE 图骨架与代价 | 已完成 | 4 层 IR 3681 算子、1557 GEMM、1536 专家 GEMM、`parameter_bytes=6,229,628,928`；dynamic `60.785 s / 13.079 t/s`、dense `63.637 s / 12.493 t/s`；48 层当前代码 44117 算子、18673 GEMM、18432 专家 GEMM，`192.49 s / 4.13 t/s` | 无 |
| G2 | 严格产物契约（五入口） | 已完成 | 23 份 pimir 全为 `dma-align=64`、`mram-bytes=8 GiB`；3163 条 sidecar 路径/哈希有效；round18–20 探针全部 REJECTED；4 层真实产物 ACCEPTED | 第 20 轮修复为修复者自测，尚未经独立评审复核 |
| G3 | 实施证据一致性检查 | 已完成 | `scripts/check_implement_evidence.py` 本次生成文档时重跑 exit 0（「实施文档数值与产物一致」），48 层结果回填后再次 exit 0 | 历史值（alt 与旧 48l 条目）不在机器核验范围；48 层当前运行见 G5 |
| G4 | FlagTree paged_attn 修复 | 已完成 | 源码对象 sha256 `068d2624…`、`518cd1e2…`；安装内 `triton-opt` 本次重跑 exit 0，输出含 1 处 `pim_block_slot` 与 5 处 `pim_f32_to_f16` | 无 |
| G5 | 48 层全模型成本验收 | 已完成 | 2026-11-13 当前代码 48 层主链 exit 0：`192.49 s / 4.13 t/s`，trace `{'pimir': 106230, 'absorbed': 24576}`（130806 条），三个严格验证器 ACCEPTED；旧 round08/round10 的 `566.863 s / 1.402 t/s` 仅作历史 | 无（旧成本证据已由当前运行替代） |

　　核心需求共 7 项：已完成 6 项（P0-1、P0-2、P0-3、P0-4、P1-1、P1-2），部分完成 1 项（P0-5，因 48 层数值对拍未用当前代码重跑）。评审扩展目标 G1–G5 全部完成。2026-11-13 已用当前工作区重跑 48 层主链（A/BCD/A2/E），容量、成本与严格契约均以当前代码为准；当前状态仍是第 20 轮修复已实施并由修复者自测通过、尚未经第 21 轮独立评审复核，本文不把当前状态写成「全部评审通过」。

## 二、总体流程

### 2.1 总体流程图
　　本需求的核心目标是让 MoE 的数据依赖分派进入静态蓝图。现有编译链只能导出静态图，HF 的 `nonzero`/gather 分派会在 `torch.export` 处失败；因此总体流程以「加载时静态化 MoE 前向」为主线。首先由注册表选择 `qwen3_moe` 并加载 fp16 权重；然后在加载后把每个专家块替换为 top-k 选中掩码版本；其次走既有图编译与 TP/PP 切分，产出骨架 IR、精化 IR 与放置 sidecar；再由 FlagTree 对每个设备算子编译出 pim mlir；最后由 GeneSim 加载骨架 IR 与 sidecar，生成 trace 与 summary，并用 HF 稀疏前向、容量契约与证据检查器验收。如图所示，静态化节点是本次新增的核心环节，去掉它整条链路在导出阶段即不成立。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff

  A["输入：HF 配置与权重<br/>4 层切片 / 48 层模型"]:::input --> B["本次改造：静态 MoE 前向<br/>selected_mask + 注册表"]:::stage
  B --> C["图编译 A/A2<br/>骨架 IR → 精化 IR"]:::stage
  C --> D["切分与放置 B/C/D<br/>TP4×PP2 + sidecar"]:::stage
  D --> E["算子编译 FlagTree<br/>23 份 pim mlir"]:::stage
  E --> F["GeneSim 仿真<br/>trace → summary"]:::stage
  F --> G["验证<br/>HF 逐 token / 容量 / checker"]:::stage
  W["约束：8 DPU × 8 GiB<br/>128 专家 top-8 / 专家宽 768"]:::input -.-> D
```

### 2.2 模块关系

| 模块/文件 | 职责（≤20字） | 改动类型 | 依赖的模块 |
|-----------|---------------|----------|------------|
| `flagos-pim-compiler/backend/architectures.py` | 注册与静态 MoE 前向 | 修改 | 无 |
| `flagos-pim-compiler/graph/` | 图拆分、切分与规格传播 | 修改 | `contracts/` |
| `flagos-pim-compiler/runtime/` | 编译入口、执行计划与解码循环 | 修改 | `graph/`、`memory/`、`contracts/` |
| `flagos-pim-compiler/memory/` | 三区规划与容量工厂 | 修改 | `contracts/op_contract.py` |
| `flagos-pim-compiler/genesim_bridge/` | IR 成本、结构指纹与放置导出 | 修改/新增 | GeneSim IR 契约 |
| `flagos-pim-compiler/scripts/run_full_pipeline.py` | A/A2/BCD/E 主链与五入口验证 | 修改 | 全部编译器模块、GeneSim |
| `genesim/src/ir/model_ir.py` | MoE 图骨架、身份与指纹 | 修改 | 无（两仓不互相 import） |
| `genesim/src/scheduler/` | 仿真调度、覆盖校验与成本角色 | 修改 | `model_ir.py`、`pimir_trace.py` |
| `genesim/src/vpu/vpu.py` | B 路 trace 重复与字节折算 | 修改 | `model_ir.py` |
| `genesim/src/pim/` | 容量指标、pimir 事实与语义表 | 修改/新增 | 无 |
| `FlagTree/lib/Dialect/TritonPIM/Transforms/` | paged_attn 相位与块表 lowering | 修改 | 无 |

### 2.3 三段证据链（图编译 → 算子编译 → GeneSim）
　　下表按「图编译 → 算子编译（FlagTree）→ GeneSim 仿真」列出入口命令、关键产物、断言或日志结果，以及该产物是否被下一段真实消费。表中路径均为当前工作区可核对的产物；命令原文与结果引自实施文档 §8，并在本次生成文档时核对日志与产物存在。

| 段 | 入口命令 | 关键产物路径 | 断言或日志结果 | 是否被下一段真实消费 |
|----|----------|--------------|----------------|----------------------|
| 图编译（A/A2/B/C/D） | `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1` | `<run>/artifacts/qwen3_moe.ir`、`qwen3_moe_pimir.ir`、`qwen3_moe_tp4_pp2_placement.json`、`qwen3_moe_tp4_pp2_placed.ir` | A 阶段 3681 算子、1557 GEMM、1536 专家 GEMM、`parameter_bytes=6,229,628,928`；结构指纹 `4519c683…`；sidecar 3163 条（1557 GEMM + 1606 算子级）、豁免 518、融合/吸收 512/512 | 是：A2 以 `qwen3_moe_tp4_pp2_placed.ir` 为输入产出 `qwen3_moe_pimir.ir`；E 阶段临时配置的 `model.ir_path` 指向该精化 IR，`compiler_placement_file` 指向本次 sidecar，均已回读复核 |
| 算子编译（FlagTree） | 由 B/C/D 内部调用 FlagTree `triton-opt`（`genesim_bridge/placement_export.py:430` 的 `_bpath_pimir` 与 `:1029` 的 `export_gemm`） | `.opcompiler_cache/*.pimir.mlir`，共 23 份（8 份 A 路 GEMM + 15 份 B 路），sidecar 逐条记录 `pimir_path`/`pimir_sha256` | 23/23 份 `pim.dma-align=64`、`pim.mram-bytes=8 GiB`；15 份带 `pim.traceTokens`；本次复核 23 个唯一路径全部存在；3163 条 sidecar 哈希全部有效 | 是：GeneSim 从 sidecar 路径编译 10906 条 trace（10906 compiled、0 reused）；trace 来源 `{'pimir': 8858, 'absorbed': 2048}`，无 template/unknown |
| GeneSim 仿真（E） | 主链 E 调用 `./run.sh --config <run>/sim_qwen3_moe_pimir_pimir.yaml` | `<run>/pim_traces_pimir/*.pim_trace`（10906 条）、`<run>/results_pimir/summary.json`、`pim_capacity_metrics.json` | dynamic：`total_time_s=60.785`、`tokens/s=13.079`、1 请求/795 token；集群常驻 `3,140,834,304 / 8,589,934,592`（36.56%），`fits=True`。dense：`63.637 s / 12.493 t/s` | 是：summary 被 `scripts/check_implement_evidence.py` 读取并逐值核验；本次重跑 exit 0 |
| 产物验收 | `python scripts/check_implement_evidence.py --doc docs/implement-qwen3-moe-20261105.md --summary dynamic=<run>/results_pimir/summary.json --summary dense=<dense_run>/results_pimir/summary.json` | 证据块中的 current 标签与两份 summary | 退出码 0，输出「实施文档数值与产物一致」 | 验证链终点；证明文档数值可追溯到本次产物 |

　　48 层全模型走同一条三段链，2026-11-13 已用当前工作区重跑（A→0→BCD→A2→E）：结构为 44117 算子、18673 GEMM、18432 专家 GEMM、`parameter_bytes=61,064,245,248`、23 份 pim mlir（8 A + 15 B，全部 64 字节对齐、8 GiB）；容量 dpu0..3 所需/可用 `8,258,576,576 / 8,522,825,728 B`、dpu4..7 `7,791,833,024 / 8,522,825,728 B`；E 为 `192.49 s / 4.13 t/s`、130806 条 trace（pimir 106230 + absorbed 24576），三个严格验证器 ACCEPTED。round08/round10 的旧主链（`566.863 s / 1.402 t/s`）是历史产物，其 19,250 条 B 路条目缺 `trace_tokens`/`local_*_shapes`，在当前契约下仍会被严格入口拒绝，只作追溯，不再作为当前成本证据（K16/Q12 已关闭）。

| 环节 | 4 层切片（当前代码） | 48 层全模型 |
|------|----------------------|-------------|
| 图编译 A/A2 | 已跑：3681 算子、1557 GEMM、1536 专家 GEMM | 已跑（当前代码）：44117 算子、18673 GEMM、18432 专家 GEMM |
| 算子编译 FlagTree | 已跑：23 份 pim mlir（8 A 路 + 15 B 路） | 已跑（当前代码）：23 份 pim mlir（8 A + 15 B），全 64 字节对齐 |
| 放置 sidecar | 已跑：3163 条（1557 GEMM + 1606 算子级） | 已跑（当前代码）：37923 条（18673 GEMM + 19250 算子级），19250 条 B 路全部带单位 |
| GeneSim 仿真 | 已跑：10906 条 trace，`60.785 s / 13.079 t/s` | 已跑（当前代码）：130806 条 trace，`192.49 s / 4.13 t/s` |
| 数值对拍 HF | 已跑：36 passed、1 skipped（8B SKIP） | round06 历史 26 passed、1 skipped（17,254.56 s）；当前代码未重跑 |
| 8 GiB 容量 | 已跑：集群 36.56%，`fits=True` | 已跑（当前代码）：dpu0..3 `8,258,576,576 / 8,522,825,728 B`、dpu4..7 `7,791,833,024 / 8,522,825,728 B` |
| 8B 真实权重 | SKIP，未覆盖 | SKIP，未覆盖 |

## 三、核心原理

### 3.1 关键机制
　　为了达成核心目标，需要同时保证三件事：静态图能表达 MoE 语义；运行期只执行命中专家且数值与稠密模拟等价；跨仓产物能被下一段真实消费而不是模板回退。现有实现只有稠密 FFN 骨架与静态蓝图，缺少 MoE 的命中语义、专家放置与代价契约。总体采用「静态稠密模拟 + 运行期门控 + 统一产物契约」的方法，分别通过下列三个机制解决。

　　**静态化 MoE 前向与选中掩码**旨在让数据依赖分派变成静态图。采用 top-k 选中索引 scatter 出布尔 `selected_mask`，全专家执行后用 `where(selected_mask, out, 0)` 掩蔽未选中专家，再乘路由权重累加。关键点是命中只由选中索引决定，路由权重只参与加权；权重下溢为零的选中专家仍执行，未选中专家的 inf/nan 被掩掉。实现位于 `backend/architectures.py:22`（`make_static_moe_forward`）与 `:40`，dispatch 读取同一掩码（`runtime/exec_plan_gen.py:169`）；4 层真实权重与 HF 稀疏前向逐 token 对拍通过。

　　**动态分派与专家并行**旨在 decode 时只执行命中专家。采用「层域专家归属 + 提交前跳过」：`_expert_provenance` 按 `(层, 专家)` 传播，跨层激活按中性、层内多专家汇合在累加出口终止；`hal_numpy.submit` 先等本层 dispatch 结果，未命中命令不再入队；EP 用 `ep_width` 把整专家轮转落到段内 DPU。实现位于 `runtime/exec_plan_gen.py:122`、`backend/hal_numpy.py:233`、`graph/strategy.py:40`；逐层 gate 断言与 16 DPU 用例验证。

　　**统一产物契约与成本模型**旨在让图编译、算子编译与仿真消费同一份证据。采用 sidecar 严格验证器：A/B 路分类只由源 IR 的 `op_type` 决定；B 路 pimir 是「1 token × 本地宽」单位并带 `pim.traceTokens`；A 路 GEMM 必须是分片实测分块 GEMM；两仓各持一张 18 类闭合语义指纹表；容量三件套随 sidecar 回传。实现位于 `scripts/run_full_pipeline.py:742`（`validate_sidecar_contract`）、`:1181`（`_BPATH_PIMIR_SEMANTICS`）、`genesim_bridge/placement_export.py:338`（`_pimir_unit_shapes`）、`genesim/src/pim/pimir_trace.py:732`（`BPATH_PIMIR_SEMANTICS`）、`genesim/src/scheduler/gene_sim_scheduler.py:1761`（`_validate_pimir_source_coverage`）。4 层真实产物 ACCEPTED，round18–20 的伪造/错值探针全部 REJECTED。

### 3.2 设计取舍

| 取舍点 | 采用方案 | 未采用方案 | 理由（≤30字） |
|--------|----------|------------|---------------|
| 单 DPU 容量 | 8 DPU × 8 GiB、系统预留 64 MiB、对齐 64 字节 | 把预算抬到 16 GiB | 用户确认 8 GiB 为硬约束（2026-11-10） |
| 专家并行组合 | PP×EP（tp=1，如 pp4×ep4） | TP×EP 复合放置 | 与现有 spec/通信/KV 契约冲突，列为后续课题 |
| 图级分块 | MoE 默认关闭，保留算子层 `tt.dot` 分块 | 对专家 linear 逐节点拆分 | 4 层实测膨胀到 139 万节点，无执行收益 |
| 选择与加权累加落点 | 主机共享工作（`ne`/`where` 归 host） | 设备侧加性 mask 模拟 select | PIM 原语集没有逐元素 select |
| 专家通信计费 | 依赖边同时看源、目的专家角色 | 只看目标角色 | 专家输出到共享累加须按源命中并集折算 |
| B 路编译单位 | 1 token × 该 DPU 本地宽 | 源 IR 全局形状 + 128 token 占位 | 运行期按实际 Tq 与本地宽重复，必须同单位 |
| 结构指纹 | schema 2，含依赖端点/张量身份/层归属 | 只哈希算子清单 | 只换一条依赖方向时旧指纹相同 |

### 3.3 适用前提与边界
- 前提：模型为 `qwen3_moe` 且按 fp16 加载；不满足时：不支持的类型显式报错，不退回 dense FFN。
- 前提：E 阶段消费本次 BCD 返回的 placement；不满足时：仿真启动前按策略/DPU/段数/Cluster/本地形状拒绝。
- 前提：严格产物满足 8 GiB/64 MiB/64 B 三件套；不满足时：严格入口拒绝，只有明确标识的非严格旧产物才兼容。
- 前提：48 层成本结论需当前代码重跑；不满足时：只引用结构与容量，不宣称成本正确。
- 前提：8B 权重存在；不满足时：用例 SKIP，记为未覆盖，不作为通过。

## 四、关键问题与解决方案

　　本需求的问题按「阻碍核心目标达成」分为四类：数值语义类（静态前向、top-k 命中、增量成本）、执行与容量类（动态分派、专家并行、8 GiB 预算）、产物契约类（严格验证、指纹、A/B 路 pimir 语义）、工具链类（FlagTree 相位与块表）。以下只收录根因不显然、解法有取舍、且在实施或评审中确实踩过坑的问题。

### 4.1 静态分派必须区分「选中」与「权重非零」（来源：round01 问题3；round06/07 问题1）
　　为了让数据依赖分派进入静态图，必须把「命中专家循环 + token 子集 gather」改成形状静态的全专家计算。HF 原版用 `expert_hit.nonzero()` 与 `hidden_states[None, top_x]`，形状随数据变化，`torch.export` 直接失败；早期静态改写又用 `routing != 0` 兼任命中条件，导致权重下溢为零的选中专家被跳过，而 HF 会执行它。本文改为由 top-k 选中索引 scatter 出布尔 `selected_mask`，全专家执行后用 `where` 掩蔽未选中专家，路由权重只参与加权。实现位于 `backend/architectures.py:22`（`make_static_moe_forward`）与 `:40`，由 4 层真实权重与 HF 稀疏前向逐 token 对拍验证（`/tmp/r20-fix-4l-tests.log`，36 passed、1 skipped）。状态：已实测通过。

### 4.2 图级分块使 4 层图膨胀到 139 万节点（来源：round01 问题5）
　　为了让 30B 规模的图可编译，需要控制图节点数。通用图级分块会把每个专家 linear 按 WRAM 拆成大量子节点，round01 探针从 7,079 节点膨胀到 1,390,344 节点，峰值内存 24.29 GiB（当前非分块
基线为 7,048 节点，实施文档 §8.4），而算子层的 `tt.dot` 分块已经能把本地形状压进 WRAM。本文新增 `graph_tiling` 编译参数与 `PIM_DISABLE_TILE` 入口，并按 MoE 身份默认关闭图级分块。实现位于 `runtime/compile.py:227`（`graph_tiling_enabled`）与 `graph/opt_passes.py`，由 `test_moe_graph_tiling_policy`、`test_optimize_graph_graph_tiling_switch` 验证。状态：已实测通过。

### 4.3 动态分派只在首层生效（来源：round05 问题1）
　　为了让动态分派在每一层真实生效，需要把专家归属限制在本层。旧实现的 `_expert_provenance` 无层边界，上一层累加的多专家状态传到下一层后，整层被判为多专家，2 层探针的 97 条门控命令全部属于第 0 层。本文把归属改为 `(层, 专家)`/`(层, None)`，层号取 `nn_module_stack` 或权重名，跨层输入按中性、层内汇合在累加出口终止；`hal_numpy.submit` 在提交前等 dispatch 结果，未命中命令不再入队。实现位于 `runtime/exec_plan_gen.py:122`（`_expert_provenance`）、`:104`（`_node_layer_of`）、`:163`（`_expert_tag_of`）与 `backend/hal_numpy.py:233`，由 `test_every_layer_gates_its_own_experts`、`test_pp_stages_each_gate_their_own_layers` 与 4 层执行期基准验证（prefill 1.148×、decode 1.202×，用户已接受）。状态：已实测通过（加速比为历史实测，本轮未重测）。

### 4.4 8 GiB 容量三处分裂（来源：round01 问题7；round02 问题1；round03 问题2；round06/07 问题2/3；round11 问题1）
　　为了让 30B 专家权重在 8 DPU 上可规划，必须有单一容量契约。旧实现主链写 4 GiB、真实测试写 16 GiB、GeneSim 写 8 GiB，三处各自硬编，48 层导出在原理上必然超预算。本文把单 DPU 总容量、64 MiB 系统预留与 64 字节对齐收成不可拆分的三件套：`contracts/op_contract.py:58` 定义 `DPU_MRAM_BYTES=8 GiB`，`memory/mem_planner.py:32`（`default_hw_budget`）/`:44`（`default_site_capacity`）派生 `HwBudget` 与 `PIMHardwareConfig`，sidecar 回传 `hw_budget`，GeneSim 在 `_validate_hw_budget`（`genesim/src/scheduler/gene_sim_scheduler.py:1121`）与每份 pimir 的 MLIR 属性上核对。4 层当前代码集群常驻 `3,140,834,304 / 8,589,934,592`（36.56%），`fits=True`；48 层当前代码（2026-11-13 重跑）容量 dpu0..3 `8,258,576,576 / 8,522,825,728`、dpu4..7 `7,791,833,024 / 8,522,825,728`，与 round08 一致且三入口 ACCEPTED。状态：4 层与 48 层当前代码均已实测通过（K16/Q12 关闭）。

### 4.5 严格产物契约的五个入口各自为政（来源：round04 问题2；round05 问题3/4；round08 问题4；round12 问题1；round13/14 问题3/4；round15 问题1/2）
　　为了让五段入口消费同一份产物，必须有共享验证器。旧实现把严格性拆成配置开关与 sidecar 自报两套状态，导出、skip-simulation、freshness、仿真 IR 与 trace 各自决定检查哪些字段，导致缺哈希、缺 `moe_config`、伪造 host 豁免、专家号互换等都能从某条路径放行。本文把 MoE 元数据、结构指纹、容量三件套、逐 op_id 专家身份、吸收/豁免证据与 A/B 路 pimir 语义收进 `validate_sidecar_contract`（`scripts/run_full_pipeline.py:742`）与 `audit_sidecar_coverage`（`:1493`），严格性取「配置 ∪ 自报」，五个入口共同调用。4 层真实产物 ACCEPTED，round18–20 的假分片、错单位、A 路指向 B 路产物、host 自报等探针全部 REJECTED。状态：已实测通过；第 20 轮修复为修复者自测，尚未独立复核。

### 4.6 B 路成本单位错配（来源：round16 问题1；round17 问题1/2/3；round18 问题1/2）
　　为了让 B 路算子的代价与真实执行几何一致，必须定义编译单位。旧实现按源 IR 全局形状与 128 token 占位生成 pimir：4 层 `VECTOR_MUL` 的 trace 是 `[Tq,192]`，字节却按 128×768 计，VECTOR_MUL 被放大到 65.244 s；48 层为 782.930 s。本文把 B 路 pimir 固定为「1 token × 该 DPU 的本地宽」，sidecar 回传 `trace_tokens` 与保留符号维的 `local_*_shapes`，pimir 带 `pim.traceTokens`，运行期按 `ceil(实际 Tq / tokens)` 重复。实现位于 `genesim_bridge/placement_export.py:338`（`_pimir_unit_shapes`）、`:579`（`_attach_bpath_pimir`）、`genesim/src/vpu/vpu.py:570`（`_unit_repetitions`）、`genesim/src/scheduler/gene_sim_scheduler.py:978`（`_validate_pimir_unit`）。4 层修复后 `VECTOR_MUL` 65.244→0.151 s，dynamic 总时长 88.483→60.785 s；48 层当前代码 `VECTOR_MUL` 从历史 782.930 s 降到 1.81 s，总时长 566.863→192.49 s（历史值来自错误单位，仅作对照）。状态：已实测通过。

### 4.7 A 路 GEMM 与 host 证据未绑定（来源：round19 问题1/2/3；round20 问题1）
　　为了让 A 路 GEMM 与 host 落点不可伪造，需要把它们绑定到源 IR 与实产。旧实现只核对 GEMM 的 `pimir_path` 与哈希，1557 条 GEMM 全部指向一份逐元素产物时，五入口与直接 GeneSim 仍接受，仿真总时长从 60.785 s 降到 1.696 s；`lm_head` 自报 `device_hint=cpu` 也能跳过 A 路检查。本文要求 A 路 pimir 必须是该分片的实测分块 GEMM（`tt.dot`/`pim.matmul`、tile/循环/分片宽/`act_spec` 一致），host 判定需同时具备源 IR `device_hint=cpu`、`pimir_exempt_ops` 登记与 `host_gemms` 角色/权重证据；B 路另用 18 类闭合语义指纹表核对 pimir 内容。实现位于 `scripts/run_full_pipeline.py:1415`（`_apath_unit_problems`）、`:1274`（`_bpath_unit_contract_problems`）、`:1181`（`_BPATH_PIMIR_SEMANTICS`）与 `genesim/src/pim/pimir_trace.py:732`。4 层/48 层/llama2 真实产物 0 问题（48 层 15 份、llama2 14 份 B 路 pimir 正例），round19/20 探针在修复前四入口接受、修复后全部 REJECTED。状态：已实测通过（第 20 轮自测，未独立复核）。

### 4.8 FlagTree 分页注意力相位取整与块表槽号（来源：round13/14 问题1/2）
　　为了让分页注意力的设备数值与主机参考在同一相位边界上，需要显式落 f16。旧 lowering 在 PV 循环里用 f32 累加 `dot` 后直接 `acc*alpha+dot`，与契约要求的「每块 PV 乘积先落 f16」相差一次 RNE，单块探针最大差 0.00048828125；块表只判上界，`-1` 会生成负偏移，上界非法项被静默换成槽 0。本文在 `FlagTree/lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp:3069` 对块乘积调用 `roundF16`（定义在 `:1420`），在 `:275` 增加 `pim_block_slot`，主机 `runtime/kernels.py:1405`（`_physical_slot`）与写 KV/主机兜底共用 `0 <= slot < pool_slots`。安装内 `triton-opt` 本次重跑 exit 0，输出含 1 处 `pim_block_slot` 与 5 处 `pim_f32_to_f16`；源码对象与安装对象 sha256 为 `068d2624…`、`518cd1e2…`。状态：已实测通过。

### 4.9 核心改动证据分类
　　下表把每个核心改动映射到「原理/不变式 → 代码位置 → 测试或产物证据」，并给出证据状态。状态定义：已实测通过＝有本次或记录在案的运行结果/产物；仅代码推理＝只做了代码路径核对；未验证/待确认＝缺当前代码产物或独立复核。

| 核心改动 | 原理/不变式 | 代码位置 | 测试或产物证据 | 状态 |
|----------|-------------|----------|----------------|------|
| 静态 MoE 前向 | 命中只由 top-k 选中掩码决定，路由权重只参与加权 | `backend/architectures.py:22,40` | 4 层 HF 逐 token；零权重选中专家两态用例 | 已实测通过 |
| 门控链留主机 | PIM 原语集没有逐元素 select | `graph/partition.py:55-66` | 4 层豁免 518=4+512+2；`test_moe_tail_is_host_shared_work` | 已实测通过 |
| 图级分块关闭 | 4 层 139 万节点回到 7048 节点 | `runtime/compile.py:227` | `test_moe_graph_tiling_policy` 等 3 条 | 已实测通过 |
| 层域门控与提交前跳过 | 每层只执行命中专家，未命中命令不入队 | `runtime/exec_plan_gen.py:122,169`；`backend/hal_numpy.py:233` | 逐层 gate 断言；4 层基准 1.148×/1.202× | 已实测通过（加速比为历史） |
| 专家并行 | 整专家轮转落段内 DPU，每 DPU 差 ≤1 | `graph/strategy.py:40,84` | 16 DPU 用例与映射均衡断言 | 已实测通过（小模型） |
| 容量契约 | 8 GiB/64 MiB/64 B 不可拆分，单一工厂派生 | `contracts/op_contract.py:58`；`memory/mem_planner.py:32,44`；`genesim/src/scheduler/gene_sim_scheduler.py:1121` | 4 层 `fits=True`、集群 36.56%；48 层容量历史 | 已实测通过（4 层）；48 层历史 |
| B 路编译单位 | 1 token × 本地宽，运行期 `ceil(Tq/tokens)` | `genesim_bridge/placement_export.py:338,579`；`genesim/src/vpu/vpu.py:570` | 23 份 pimir 中 15 份带 `pim.traceTokens`；VECTOR_MUL 65.244→0.151 s | 已实测通过 |
| A 路 GEMM 绑定 | pimir 必须是该分片的实测分块 GEMM | `scripts/run_full_pipeline.py:1415`；`genesim/src/pim/pimir_trace.py:830` | 4 层 1557 GEMM；全部指向 B 路产物等探针 REJECTED | 已实测通过 |
| host 证据链 | 源 IR 落点 + 豁免登记 + `host_gemms` 角色/权重 | `scripts/run_full_pipeline.py:460,488` | `lm_head`/router 自报 host 探针 REJECTED；真实产物 0 host 条目 | 已实测通过 |
| B 路语义指纹 | 18 类闭合表，未登记类型拒绝 | `scripts/run_full_pipeline.py:1181`；`genesim/src/pim/pimir_trace.py:732`；`genesim/src/scheduler/gene_sim_scheduler.py:2039` | 38 条用例；4L/48L/llama2 真实产物 0 问题 | 已实测通过（第 20 轮自测，未独立复核） |
| 五入口严格验证 | 严格性 = 配置 ∪ sidecar 自报 | `scripts/run_full_pipeline.py:742,1493` | 正例 ACCEPTED；缺项/错值/伪造探针 REJECTED | 已实测通过 |
| 结构指纹 schema 2 | 依赖端点、张量身份、层归属入指纹 | `genesim/src/ir/model_ir.py:443`；`genesim_bridge/ir_fingerprint.py:35` | 换边/换张量身份/跨层移动必须变指纹 | 已实测通过 |
| 证据一致性 checker | 运行标签 → summary → 指标 → 正文 | `scripts/check_implement_evidence.py` | 本次重跑 exit 0 | 已实测通过 |
| FlagTree PV/槽号 | 每块 PV 先落 f16；`0 <= slot < pool_slots` | `FlagTree/lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp:275,3069`；`runtime/kernels.py:1405` | lit exit 0；源码/安装对象 sha256 一致 | 已实测通过 |
| A 路 `pim.matmul` 分支 | 另一种合法分块原语 | `scripts/run_full_pipeline.py:1415` | 只有代码路径核对，无真实产物样例 | 仅代码推理/待确认 |

## 五、改动说明

### 5.1 改动总览
　　截至第 20 轮修复完成，代码/测试改动 80 个文件（修改 72：flagos-pim-compiler 50、genesim 20、FlagTree 2；新增 8），文档改动 33 个（修改 11、新增 22）。新增代码/测试文件为 `genesim_bridge/ir_fingerprint.py`、`scripts/bench_moe_dispatch.py`、`scripts/check_implement_evidence.py`、`tests/test_implement_evidence.py`、`tests/test_ir_fingerprint.py`、`genesim/conf/sim_qwen3_moe_pimir.yaml`、`genesim/tests/sim/test_moe_execution_cost.py`、`FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir`。按三个提交统计：compiler `5b5d539` 87 个文件 `+15,847 / -881`（代码/测试 55、文档 32）；genesim `6d24d30` 23 个文件 `+7,358 / -519`（代码/测试 22、文档 1）；FlagTree `3690b47c2` 3 个文件 `+59 / -12`（代码/测试 3）。核心改动集中在三仓的 MoE 静态前向、执行计划/调度、放置导出与严格验证器。

### 5.2 文件改动清单
　　完整清单见实施文档 §3.2（78 个修改文件 + 8 个新增文件，按文件逐条列出改动内容与对应任务 ID）。下表列出理解本次改动必需的核心文件，按「业务代码在前、测试代码在后」排序。

| 文件路径 | 改动类型 | 改动内容（≤30字） | 对应需求 |
|----------|----------|-------------------|----------|
| `backend/architectures.py` | 修改 | 静态 MoE 前向、注册、策略校验 | P0-1/P0-2/P1-2 |
| `graph/partition.py` | 修改 | 门控算子留主机、`ne`/`where` 归 host | P0-2 |
| `graph/strategy.py` | 修改 | router 列切规则、`ep_width` | P0-2/P1-2 |
| `graph/spec_prop.py` | 修改 | 专家权重与 EP 放置规则 | P0-2/P1-2 |
| `graph/opt_passes.py` | 修改 | 图级分块开关 | P0-1 |
| `runtime/compile.py` | 修改 | `graph_tiling`、`moe_execution` 配置 | P0-1/P1-1 |
| `runtime/exec_plan_gen.py` | 修改 | 层域专家归属、dispatch、`skip_zero` | P1-1 |
| `runtime/executor.py` | 修改 | 按编译配置启用动态分派 | P1-1 |
| `runtime/kernels.py` | 修改 | 主机分页注意力 f16 边界、槽号预检 | P0-4 |
| `backend/hal_numpy.py` | 修改 | 提交前跳过、跳过写零、计数 | P1-1 |
| `contracts/op_contract.py` | 修改 | 8 GiB 容量契约与硬件工厂 | P0-3 |
| `contracts/dtypes.py` | 修改 | bool 掩码类型与注释 | P0-1 |
| `memory/mem_planner.py` | 修改 | `default_hw_budget`/`default_site_capacity` | P0-3 |
| `genesim_bridge/placement_export.py` | 修改 | 专家定位、B 路单位、容量回传 | P0-3/P1-1/G2 |
| `genesim_bridge/ir_fingerprint.py` | 新增 | 编译器侧结构指纹 | G2 |
| `genesim_bridge/op_classify.py` | 修改 | 逐算子成本配方与依赖项 | G1 |
| `genesim_bridge/cost_extractor.py` | 修改 | 逐算子测量、融合清单、成本版本 | G1 |
| `scripts/run_full_pipeline.py` | 修改 | 注册表主链、五入口验证、运行隔离 | G2 |
| `scripts/export_pp_placement.py` | 修改 | 注册表加载、策略校验、容量报告 | P0-3/P1-2 |
| `scripts/bench_moe_dispatch.py` | 新增 | 同一计划/后端的执行期 A/B | P1-1 |
| `scripts/check_implement_evidence.py` | 新增 | 证据块与 summary 机器核对 | G3 |
| `backend/vllm_worker.py` | 修改 | 容量工厂与白名单跟随 | P0-3 |
| `scripts/serve_pim.py` | 修改 | 启动预检复用真实 KV 布局 | P0-3 |
| `tests/test_architectures.py` | 修改 | MoE 用例、注册表、容量守卫 | P0-1/P0-2 |
| `tests/test_moe_dispatch.py` | 修改 | 逐层 gate、提交前跳过、零值 | P1-1 |
| `tests/test_moe_ep.py` | 修改 | EP 合法路径逐层门控 | P1-2 |
| `tests/test_genesim_simulation.py` | 修改 | 五入口契约正反向矩阵 | G2 |
| `tests/test_placement_export.py` | 修改 | 专家身份、B 路单位、A 路语义 | G2 |
| `tests/test_qwen3_30b_real.py` | 修改 | 4 层真实权重目标用例 | P0-5 |
| `tests/test_implement_evidence.py` | 新增 | checker 反例与 CI 用例 | G3 |
| `tests/test_ir_fingerprint.py` | 新增 | 跨仓指纹一致性 | G2 |
| `genesim/src/ir/model_ir.py` | 修改 | MoE 骨架、身份、指纹、批量修改 | G1 |
| `genesim/src/main.py` | 修改 | summary MoE 字段与容量 scope | G1/G2 |
| `genesim/src/pim/capacity.py` | 修改 | ClusterPU/TensorPU 容量口径 | G2 |
| `genesim/src/pim/pimir_trace.py` | 修改 | 单位事实、A 路 GEMM 事实、语义表 | G2 |
| `genesim/src/scheduler/gene_sim_scheduler.py` | 修改 | 覆盖校验、成本角色、动态计费 | G1/G2 |
| `genesim/src/vpu/vpu.py` | 修改 | 绑定 Tq、单位重复、字节折算 | G1 |
| `genesim/scripts/model_parser.py` | 修改 | `--moe-execution`、参数量交叉校验 | G1 |
| `genesim/conf/sim_qwen3_moe_pimir.yaml` | 新增 | 独立入口模板与说明 | G1 |
| `genesim/tests/sim/test_moe_execution_cost.py` | 新增 | any-hit 成本与依赖方向用例 | G1 |
| `genesim/tests/sim/test_compiler_placement.py` | 修改 | 契约/单位/语义正反向用例 | G2 |
| `genesim/tests/sim/test_model_ir.py` | 修改 | MoE 几何/指纹/批量修改用例 | G1 |
| `genesim/tests/sim/test_vpu.py` | 修改 | 单位重复/绑定 Tq 用例 | G1 |
| `genesim/tests/sim/test_pimir_trace.py` | 修改 | trace 单位与语义解析用例 | G2 |
| `FlagTree/lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp` | 修改 | PV 取整、块表槽号兜底 | P0-4 |
| `FlagTree/lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp` | 修改 | PV 相位说明 | P0-4 |
| `FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir` | 新增 | 槽号与 5 处 f16 取整 lit | P0-4 |

### 5.3 核心函数清单

| 函数/类 | 位置 | 改动类型 | 作用（≤30字） | 调用方 |
|---------|------|----------|---------------|--------|
| `make_static_moe_forward()` | `backend/architectures.py:22` | 新增 | 生成静态 MoE 前向闭包 | `_load_qwen3_moe()` |
| `_moe_strategy()` | `backend/architectures.py:173` | 新增 | 按专家宽与专家数选择策略 | `ARCHITECTURES["qwen3_moe"]` |
| `_expert_provenance()` | `runtime/exec_plan_gen.py:122` | 修改 | 按层域传播专家归属 | `build_execution_plan()` |
| `_emit_moe_dispatch()` | `runtime/exec_plan_gen.py:169` | 新增 | 发射本层 dispatch 主机命令 | `build_execution_plan()` |
| `submit()` | `backend/hal_numpy.py:233` | 修改 | 提交前等 dispatch、未命中不入队 | 执行器 |
| `graph_tiling_enabled()` | `runtime/compile.py:227` | 新增 | 按 MoE 身份决定图级分块 | `compile_llama2()` |
| `compile_llama2()` | `runtime/compile.py:243` | 修改 | 统一编译入口与执行语义 | 主链/测试 |
| `default_site_capacity()` | `memory/mem_planner.py:44` | 新增 | 同时派生 HwBudget 与硬件 | 真实测试/主链/基准 |
| `_bpath_pimir()` | `genesim_bridge/placement_export.py:430` | 修改 | 按算子类型生成 B 路 pimir | `_attach_bpath_pimir()` |
| `_attach_bpath_pimir()` | `genesim_bridge/placement_export.py:579` | 修改 | 回传 `trace_tokens`/本地形状 | 放置导出 |
| `validate_sidecar_contract()` | `scripts/run_full_pipeline.py:742` | 修改 | 五入口共享严格验证器 | 导出/skip/freshness/sim_ir/trace |
| `audit_sidecar_coverage()` | `scripts/run_full_pipeline.py:1493` | 修改 | 覆盖、豁免、吸收、单位审计 | `validate_sidecar_contract()` |
| `_bpath_pimir_semantic_problems()` | `scripts/run_full_pipeline.py:1227` | 新增 | 核对 B 路 pimir 语义指纹 | `audit_sidecar_coverage()` |
| `_apath_unit_problems()` | `scripts/run_full_pipeline.py:1415` | 新增 | 核对 A 路分块 GEMM 语义/形状 | `audit_sidecar_coverage()` |
| `build_from_hf_config()` | `genesim/src/ir/model_ir.py:1379` | 修改 | 从 HF 配置构建 MoE 图骨架 | `model_parser.py` |
| `structural_fingerprint()` | `genesim/src/ir/model_ir.py:443` | 新增 | 计算 schema 2 结构指纹 | 主链/GeneSim |
| `batch_edit()` | `genesim/src/ir/model_ir.py:664` | 新增 | 批量增删算子并一次重建索引 | TP 展开/relay 插入 |
| `_validate_pimir_source_coverage()` | `genesim/src/scheduler/gene_sim_scheduler.py:1761` | 修改 | 源 IR 驱动的覆盖与语义校验 | 调度器 |
| `_assert_pimir_operator_semantics()` | `genesim/src/scheduler/gene_sim_scheduler.py:2039` | 新增 | 按 18 类表核对 B 路语义 | `_validate_pimir_unit()` |
| `_unit_repetitions()` | `genesim/src/vpu/vpu.py:570` | 修改 | 按绑定 Tq 与单位计算重复次数 | `PIMVPU.execute()` |
| `_host_paged_attention()` | `runtime/kernels.py:1481` | 修改 | 主机路径显式 f16 相位边界 | `sdpa_kernel()` |

### 5.4 数据结构与接口变更
　　为了让 MoE 身份、执行语义、成本角色与产物证据跨段传递，需要把原先散落在 IR 自描述、sidecar 与调度器里的信息收成显式字段。已有结构缺少专家几何、成本角色、编译单位与容量三件套，导致下游只能靠字符串自洽或形状猜测。因此新增或修改下列结构与接口。

　　**`ModelIR` 的 MoE 元数据与指纹**（`genesim/src/ir/model_ir.py:420-423`、`:443`）
```python
moe_num_experts: int = 0
moe_top_k: int = 0
moe_intermediate_size: int = 0
moe_execution: str = ""          # dense_sim / dynamic_dispatch
def structural_fingerprint(self) -> str:  # schema 2：含依赖端点/张量身份/层归属
```
　　默认值为 0/空串，非 MoE 模型行为不变；MoE 模型由 HF config 派生后逐项核对。

　　**`Operator.moe_cost_role` 与 `runtime_query_tokens`**（`genesim/src/ir/model_ir.py:116`、`:142`）
```python
moe_cost_role: str = ""          # expert_work / expert_comm / shared_reduce
runtime_query_tokens: int = 0    # 调度器绑定实际 Tq，不落 IR
```
　　成本角色区分可门控专家工作与共享累加；`runtime_query_tokens` 只服务运行期重复次数，序列化时剔除。

　　**`ShardStrategy.ep_width`**（`graph/strategy.py:40`）
```python
ep_width: int = 1                # 1 = 关闭；PP×EP 时 tp=1、num_dpus=stages*ep_width
def ep_dpu_of_expert(self, layer, expert, num_layers) -> int:  # 段内轮转，每 DPU 差 ≤1
```
　　默认 1 与旧行为逐字节一致；既有调用无需改动。

　　**sidecar 的容量与编译单位字段**（`genesim_bridge/placement_export.py:579`、`scripts/run_full_pipeline.py:1042`）
```json
"hw_budget": {"mram_bytes_per_dpu": 8589934592, "sys_reserve_bytes": 67108864, "align": 64},
"moe_config": {"num_experts": 128, "top_k": 8, "intermediate_size": 768, "execution": "dynamic_dispatch"},
"operators": {"<op_id>": {"trace_tokens": 1, "local_in_shapes": ["Tq", 192], ...}}
```
　　严格产物缺字段、类型错误或任一值偏离即失败；非严格旧式 sidecar 才允许无容量字段。

　　**执行计划的门控字段**（`backend/hal_numpy.py:146`、`:233`）
```python
self.gate_skips: dict[tuple[int, int], set[int]] = {}   # (lane, dispatch) -> 跳过专家
if "gate" in cmd.payload and self.moe_dispatch is not False:
    if int(cmd.payload["expert"]) in self.gate_skips.get(...):
        event = self._skipped_event(...)                # 提交前返回，不入执行队列
```
　　默认空字典等价于改动前行为；只有带 `gate` 的命令受影响。

　　**B 路语义指纹表**（`genesim/src/pim/pimir_trace.py:732`、`scripts/run_full_pipeline.py:1181`）
```python
BPATH_PIMIR_SEMANTICS: Dict[str, PimirSemantics] = {
    "RMSNORM": {"required": ("pim.normalize",), "attrs": ("rmsNorm",)},
    "SILU":    {"required": ("pim.lut",), "attrs": ("activation",)},
    "ROPE":    {"required": ("pim.eltwise",) * 3, "attrs": ("rotateHalf", "sourceSubBlocks")},
    # ... 共 18 类；未登记类型直接拒绝
}
```
　　两仓各持一份，用跨仓测试逐值钉住；表内容按真实编译实产校准，`phase_count` 与集合成员不再是跳过语义核对的出口。

## 六、验证

### 6.1 验证环境
　　代码基线：flagos-pim-compiler `pengcheng@b2f07ca`、genesim `pengcheng@434e147`、FlagTree `feat/ceva-with-paged-attn@6e34f0299`；改动已提交（compiler `5b5d539`、genesim `6d24d30`、FlagTree `3690b47c2`）。运行环境：Linux，Python 3.10.20，torch 2.9.1+cpu，transformers 4.57.6；真实权重目录 `/data2/pengcheng-work/Qwen3-30B-A3B-4L`。

### 6.2 验证命令与结果
　　验证采用分层方法：既有回归确认原有功能未受影响；4 层真实权重目标命令确认 MoE 数值与 HF 一致；全链运行确认图编译、FlagTree 与 GeneSim 逐段消费；产物静态检查确认 sidecar 路径/哈希、pimir 属性与容量；证据 checker 确认文档数值可追溯到 summary。下表中标注「引用记录」的行是实施文档已记录、本次生成文档时核对日志/产物存在的命令；标注「本次重跑」的行是本次生成文档时实际执行的命令。

| 验证项 | 命令 | 结果 | 结论 | 说明 |
|--------|------|------|------|------|
| 4 层真实权重目标命令 | `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider` | 36 passed、1 skipped、2 warnings in 242.71 s | 通过 | 引用记录；日志 `/tmp/r20-fix-4l-tests.log`；8B SKIP 未覆盖 |
| 4 层全链 dynamic | `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1` | `PIPELINE_EXIT=0`；`total_time_s=60.785`、`tokens/s=13.079`；trace 10906（pimir 8858 / absorbed 2048） | 通过 | 引用记录；运行目录 `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/`；summary/trace 本次核对 |
| 4 层全链 dense | 同命令追加 `--moe-execution dense_sim --log-dir /tmp/r20-fix-dense` | `PIPELINE_EXIT=0`；`63.637 s / 12.493 t/s` | 通过 | 引用记录；summary 本次核对 |
| GeneSim 套件 | `cd genesim && ./run.sh --test sim` | 45/45 个测试文件通过、exit 0 | 通过 | 引用记录；日志 `/tmp/r20-fix-genesim-sim2.log` |
| 编译器全量回归 | `python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider --basetemp=/data/fjg/pengcheng/flagOS/.pytest-tmp-r21` | 2207 passed、11 skipped、52 deselected、0 failed in 1373.42 s | 通过 | 引用记录；日志 `/tmp/r20-fix-regression.log`；此前一次因根盘 `/tmp` 被历史产物占满在 51% 处 ENOSPC（`/tmp/r20-fix-regression-enospc.log`），改 basetemp 到 `/data` 后干净通过，是环境事实不是代码失败 |
| 证据一致性 checker | `python scripts/check_implement_evidence.py --doc docs/implement-qwen3-moe-20261105.md --summary dynamic=<run>/results_pimir/summary.json --summary dense=<dense_run>/results_pimir/summary.json` | exit 0，输出「实施文档数值与产物一致」 | 通过 | 本次重跑；dynamic/dense 两份 summary 逐值核验 |
| FlagTree lit | `/data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/bin/triton-opt -split-input-file -pim-expand-phases -pim-lower-to-emitc test/Dialect/TritonPIM/paged_attn_emitc.mlir` | exit 0；输出含 1 处 `pim_block_slot` 调用、5 处 `pim_f32_to_f16` 调用 | 通过 | 本次重跑；源码对象 sha256 `068d2624…`、`518cd1e2…` |
| 48 层数值（历史） | 同 4 层目标命令，`QWEN3_30B_MODEL_DIR=/data2/pengcheng-models/Qwen3-30B-A3B` | round01 23 passed、1 skipped（25,426.83 s）；round06 26 passed、1 skipped（17,254.56 s） | 引用历史，未执行 | `/tmp/test_target_full_30b_systemd.log`、`/tmp/qwen3-review-round06/target48l.log`；round02 的 `full30b-r2` 被 systemd-oomd 杀死（共享 CEVA 内存压力，环境阻塞） |
| 48 层主链（当前代码，2026-11-13） | `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-models/Qwen3-30B-A3B --num-stages 2 --max-requests 1 --log-dir /data/fjg/pengcheng/qwen3-48l-verify-20261112` | exit 0；A 44117/18673/18432；BCD 18673+19250，23 份 pim mlir；E `192.49 s / 4.13 t/s`、130806 条 trace；容量 dpu0..3 `8,258,576,576/8,522,825,728`、dpu4..7 `7,791,833,024/8,522,825,728`；三入口 ACCEPTED | 通过（当前代码） | 运行目录 `/data/fjg/pengcheng/qwen3-48l-verify-20261112/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-221320-497107/`；全链约 7h17m |
| 48 层主链（历史 round08/10） | 同上 | round08 exit 0；E `566.863 s / 1.402 t/s`；成本在新契约下失效 | 引用历史 | 仅作追溯；旧 B 路条目缺单位，严格入口仍拒绝 |
| P1-1 执行期基准（历史） | `python scripts/bench_moe_dispatch.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --new-tokens 2 --repeat 3 --warmup 1 --num-dpus 8` | prefill 1.148×、decode 1.202×，token 完全一致 | 引用历史，用户已接受 | 本轮未重测；Q7 关闭 |

　　复现命令（按执行顺序；48 层主链已于 2026-11-13 重跑，48 层数值与 llama2_7b 慢速子集未重跑，命令与参数照实施文档原文）：

```bash
source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /data/fjg/pengcheng/flagOS/flagos-pim-compiler

# 4 层真实权重目标命令
QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L \
  python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py \
  tests/test_qwen3_30b_real.py -q -p no:cacheprovider

# 4 层全链（dynamic_dispatch）
python scripts/run_full_pipeline.py \
  --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
  --num-stages 2 --max-requests 1

# 4 层全链（dense_sim 对照）
python scripts/run_full_pipeline.py \
  --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
  --num-stages 2 --max-requests 1 \
  --moe-execution dense_sim --log-dir /tmp/r20-fix-dense

# GeneSim 套件
cd /data/fjg/pengcheng/flagOS/genesim && ./run.sh --test sim

# 编译器全量回归（basetemp 放在 /data，避免根盘写满）
cd /data/fjg/pengcheng/flagOS/flagos-pim-compiler
python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider \
  --basetemp=/data/fjg/pengcheng/flagOS/.pytest-tmp-r21

# 实施证据一致性
python scripts/check_implement_evidence.py \
  --doc docs/implement-qwen3-moe-20261105.md \
  --summary dynamic=test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/results_pimir/summary.json \
  --summary dense=/tmp/r20-fix-dense/qwen3_moe_tp4_pp2_dense_sim_20261112-182507-3141801/results_pimir/summary.json

# FlagTree lit
cd /data/fjg/pengcheng/flagOS/flagOS-installers/FlagTree
/data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/bin/triton-opt \
  -split-input-file -pim-expand-phases -pim-lower-to-emitc \
  test/Dialect/TritonPIM/paged_attn_emitc.mlir
```

### 6.3 验收标准对照

| 验收标准（来自需求文档） | 验证方法 | 结论 | 证据 |
|--------------------------|----------|------|------|
| `python -m pytest tests/ -q -k "llama2_7b"` 基线 48 passed | 慢速回归 | 未验证 | 本轮未执行；实施文档 Q4 记录历史 48 passed（0:53:44） |
| P0-1/2/3/4 以小随机 MoE 全链路 + `test_qwen3_moe` 系列通过 | 4 层真实权重目标命令 + 小模型用例 | 满足 | 6.2 第 1 行：36 passed、1 skipped |
| P0-5 真实 30B 短序列贪心与 HF 一致 | 4 层切片与 48 层 | 部分满足 | 4 层当前代码通过；48 层主链当前代码通过（2026-11-13）；48 层数值仍为 round01/round06 历史，当前代码未重跑数值对拍 |
| P1-1 token 一致 + 加速比记录 | 执行期 A/B | 满足（历史） | prefill 1.148×、decode 1.202×，用户已接受（Q7） |
| P1-2 16 DPU 专家并行 + 映射均衡 | 16 DPU 用例 | 满足（小模型） | `test_ep_16_dpu_generate_matches_hf` 等；PP×EP |
| 不回归：Llama-2-7B 与 Qwen3-8B 既有端到端结果不变 | 慢速回归 + 8B 用例 | 未验证 | Llama2 慢回归未跑（K12）；8B 无权重 SKIP（K4） |

### 6.4 评审情况
　　本需求共进行 20 轮评审，形成 94 条报告项（含跨轮重复项）：高优先级（严重+高）57 条、中优先级 29 条、低优先级 8 条。round01–round19 的问题均已修复并由后续轮次复测；第 20 轮的 2 项（高 1、低 1）已实施并由修复者自测通过，但按用户设定的 20 轮上限停止，没有再进入第 21 轮独立评审，因此当前状态不能写成「全部评审通过」。各轮核心问题及最终状态如下。

| 评审轮次 | 核心问题 | 修复状态 | 最终证据 |
|----------|----------|----------|----------|
| 第 1 轮 | GeneSim 误建稠密 MoE、主链写死 Llama2、溢出 NaN、模板回退、默认分块、共享 HF 参考、8 GiB、分派不完整 | 已修复 | 后续轮次 4 层真实权重、48 层骨架、严格契约与容量证据 |
| 第 2 轮 | 容量 4/8/16 GiB 分裂、仿真语义脱离动态分派、产物契约可绕过、VECTOR_MUL 写成 add、非 GEMM FLOPs 虚高、无执行加速 | 已修复 | 8 GiB 契约、`moe_execution` 贯穿、`pim.eltwise<mul>`、逐算子宽度、1.148×/1.202× |
| 第 3 轮 | 图骨架缺 lm_head/q_norm/k_norm、容量未统一、any-hit 计费、产物身份不全 | 已修复 | 4L 3681/1557、`parameter_bytes=6,229,628,928`、结构指纹 |
| 第 4 轮 | 非 GEMM 成本挂 `Tq(Tp+Tq)`、GPU 落点绕过覆盖、指纹缺拓扑、旧式依赖漏折算 | 已修复 | 成本依赖项进配方、源 IR 覆盖、schema 2 指纹、统一折算出口 |
| 第 5 轮 | 动态分派只首层生效、加权累加错标主机、吸收/豁免/容量契约不闭合 | 已修复 | 层域 provenance、host 共享工作、消费者方向与三件套 |
| 第 6 轮 | 零权重选中专家被跳过、对齐 64/8 分裂、专家身份集合比较、48 层未闭环、单专家误判 | 已修复（round07 根治） | `selected_mask`、`default_site_capacity`、逐 op_id 身份、1 专家合法 |
| 第 7 轮 | 同一根因复发，另有非专家 GEMM 任意豁免、TP 展开 O(N²) | 已修复 | 落点证据、`batch_edit()`、48 层一次索引重建 |
| 第 8 轮 | E 消费旧 `tp4_pp2` sidecar、48 层产物未重导、策略校验绕过、共享验证器子集、预检 256 对齐 | 已修复 | `_assert_placement_matches_run`、round08 48L 产物、`strategy_check`、`validate_sidecar_contract`、`build_kv_layout` |
| 第 9 轮 | 吸收漏查主算子多消费者、豁免未核放置矛盾 | 已修复 | 消费者方向 + 阶段化闭合；豁免/放置/host 证据交叉核对 |
| 第 10 轮 | 专家输出边按全量计费、dynamic FLOPs 誊写错误 | 已修复 | 依赖边两端角色；证据 checker 与更正后的 `5,415,241,254,987.654` |
| 第 11 轮 | 直接 GeneSim 可抬到 16 GiB、运行产物共享删除、证据检查无归属 | 已修复 | `DPU_MRAM_BYTES` 常量、运行目录隔离、证据块 |
| 第 12 轮 | 显式严格配置下容量可绕过、正文证据错配 | 已修复 | `strict_pimir` 并集；正文标签绑定 |
| 第 13 轮 | PV 未按相位取整、负槽号寻址、严格入口缺产物、trace 无哈希、正文归属 | 已修复（round14 根治） | `roundF16`、`pim_block_slot`、严格入口矩阵、trace lineage、`facts` |
| 第 14 轮 | 同一根因复测成立，另有实施文档未登记 round13 | 已修复 | 源码与安装对象逐字节相同；§7.1 逐条登记 |
| 第 15 轮 | 严格入口漏检 pimir 产物、trace 可无配置签发、E 依赖未入库 trace、§8.4 历史证据越界、注释错位 | 已修复 | 产物身份固定规则、配置必需、合成 trace、证据分栏 |
| 第 16 轮 | B 路 pimir 按全局形状/128 token 计费、A 段未核 top-k、bool 注释、回归计数 | 已修复 | 1 token × 本地宽、HF config 派生身份、注释修正、计数回填 |
| 第 17 轮 | 运行期重复次数按形状猜、四节点注意力被单位误拒、单位未进共享验证器、独立配置指向旧 dense 产物 | 已修复 | 绑定 Tq、符号维单位、`audit_sidecar_coverage` 严格分支、共享产物刷新 |
| 第 18 轮 | A/B 分类取自 sidecar `shards`、单位未与 pimir 绑定、缓存锁 5 秒超时、checker 四份声明、容量跨口径、测试手写容量 | 已修复 | 源 IR 分类、pimir 实产唯一真源、spawn + 60 秒、两份 current summary、集群口径、`default_site_capacity` |
| 第 19 轮 | A 路 GEMM 语义未绑定、host 自报跳过检查、条目字段未对源 IR、主机分页注意力未落 f16、容量指标跨口径 | 已修复 | `_apath_unit_problems`、host 三件套、`op_type`/`semantic_role` 绑定、显式 f16 边界、判定利用率同源 |
| 第 20 轮 | B 路 pimir 未绑定算子语义、容量文档 scope 措辞 | 已修复并自测，未独立复核 | 18 类闭合语义指纹表（两仓逐值钉住）；4L 真实产物、48L 15 份、llama2 14 份 pimir 正例；ROPE/PAGED_ATTN 互换探针两仓 REJECTED；summary 新增三个 scope 字段 |

　　当前未修复项为 48 层数值对拍（L2）与工具链/环境项（K4、K10、K11/Q10、K12/Q11），48 层成本证据已由 2026-11-13 当前代码重跑恢复（K16/Q12 关闭），见第八节。跨轮重复问题按最终结果合并核对，不再按轮次重复计数。

## 七、与设计的差异

| 设计条目 | 设计方案（≤20字） | 最终实现（≤20字） | 差异原因（≤30字） | 是否经确认 |
|----------|-------------------|-------------------|-------------------|------------|
| design §4.9 | TP×EP 复合放置 | PP×EP（tp=1） | 与 spec/通信/KV 契约冲突，改为整专家落 DPU | 是（用户确认 2026-11-05） |
| design §4.7 | 30B 测试预算 16 GiB | 统一为 8 GiB 契约 | 用户确认 8 DPU × 8 GiB 硬约束 | 是（用户确认 2026-11-10） |
| design §4.5 | 视需要新增 `_rule_weighted_partial` | 未新增 | 既有 mul 规则已覆盖 Partial×Replicate | 是（设计预留，实施 I3） |
| design §3.1/§十.3 | 路由加权与累加走 DPU | 选择/加权/累加归 host | 真实计划中已折成主机闭包，PIM 无 select | 是（评审 round05；设计已同步） |
| design §4.8 | gate 标志位运行期跳过 | 提交前等 dispatch 再决定入队 | 执行期跳过没有加速，改为提交面收敛 | 是（评审 round02；用户接受 1.148×/1.202×） |
| design §十.4 | B 路单位与产物契约 | 增加 A 路分块 GEMM、host 三件套、18 类语义表 | 评审 round16–20 逐步闭合伪造路径 | 是（评审要求；设计已同步） |
| design §十.5 | 容量三件套与利用率 | 增加单 TensorPU/集群/诊断比三个 scope 字段 | 避免 `fits=True` 与 145% 利用率自相矛盾 | 是（评审 round19/20；设计已同步） |
| design §4.7/§七 | 48 层全模型当前代码验收 | 2026-11-13 用当前代码重跑 A/BCD/A2/E（`192.49 s / 4.13 t/s`，三入口 ACCEPTED） | 用户 2026-11-12 指示重跑；48 层数值仍为历史 | 是（用户指示；K16/Q12 关闭） |
| design §4.9/§九.4 | 16 DPU 组合示例 tp2×pp2×ep4 | pp4×ep4（PP×EP） | 与 §4.9 路线修正一致 | 是（用户确认） |

## 八、当前遗留问题

| 编号 | 问题描述（≤30字） | 影响范围（≤20字） | 来源 | 状态 |
|------|-------------------|-------------------|------|------|
| L1 | 48 层旧 B 路条目缺编译单位，成本证据失效 | 48 层代价验收 | K16/Q12 | 已关闭（2026-11-13 当前代码重跑；三入口 ACCEPTED） |
| L2 | 48 层数值为 round01/round06 历史，非当前代码 | P0-5 全模型验收 | 实施 §8.4/K15 | 未验证；主链已重跑，数值对拍仍为历史 |
| L3 | 8B 无权重，用例 SKIP | P0-5 与回归 | K4 | 外部阻塞；未覆盖 |
| L4 | P1-1 加速比未在本轮重测 | 性能验收 | Q7/实施 §9 | 已规避；1.148×/1.202× 为历史实测、用户接受 |
| L5 | TP×EP 复合放置未实现 | 更大规模专家并行 | design §九.4 | 未实现；列为后续课题 |
| L6 | A 路 `pim.matmul` 无真实产物样例 | A 路语义覆盖 | round20 问题1 | 仅代码推理；待真实样例确认 |
| L7 | Llama2 旧 sidecar 缺 `hw_budget`/指纹 | Llama2 慢速仿真 | K12/Q11 | 未修复；需当前主链重导 |
| L8 | 单专家真实 BCD 受 FlagTree N=1 分块限制 | 单专家全链 | K11/Q10 | 未修复；工具链能力 |
| L9 | 第 20 轮修复未经独立评审复核 | 最终验收结论 | 20 轮上限 | 待确认；修复已实施并自测 |
| L10 | 一次全量回归因根盘 `/tmp` 满在 51% 处 ENOSPC | 回归执行环境 | I96/日志 | 已规避；basetemp 改到 `/data` 后通过 |
| L11 | 48 层主链容量为 round08 历史产物 | 8 GiB 全模型容量 | K7/K16 | 已关闭（2026-11-13 当前代码实测容量一致，`fits=True`） |

　　当前无「已修复但未登记」的评审项；round01–20 的确认与修复记录见实施文档 §7.1。第 20 轮的两项问题已按实施文档 §7.1 登记，但其结论属于修复者自测，不是独立评审结论；在下一轮独立评审前，本文不将其表述为「已通过评审」。
