# 技术文档：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理

> 文档编号：techdoc-ceva-llama2-7b-20261104  
> 创建日期：2026-11-04  
> 关联文档：`request-ceva-llama2-7b-20261028`、`design-ceva-llama2-7b-20261028`、`implement-ceva-llama2-7b-20261029`、`review-ceva-llama2-7b-round1-20261030` 至 `review-ceva-llama2-7b-round11-20261103`  
> 关联项目：flagos-pim-compiler、FlagTree、GeneSim、CEVA 工作区  
> 代码基线：图编译器 `main@bf5b8a5`（含未提交改动）；FlagTree `develop@262adb3b7`（含未提交改动）；CEVA `main@717407d`（含未提交改动）；GeneSim `fjg-dev@cdf6c99`（工作区干净）

## 一、概述

### 1.1 背景与问题

现有图编译器已能在 GeneSim 中运行并生成 GML，CEVA 工作区也已用厂商产物跑通 Llama2 7B。需求的核心不是替换厂商工具链，而是让我方图编译器和算子编译器生成的 GML 与全部配套 `bin`，经厂商 GC、运行器和仿真器完成同样的全流程推理。需要解决的是自研产物的完整覆盖、接口兼容和来源可信，同时保持生成策略通用。

### 1.2 方案摘要

总体采用“自研编译产物加厂商执行工具链”的技术路线。图编译器负责完整模型结构、量化标定和 GML/`bin` 写盘，FlagTree 负责相位与硬件数值字段，厂商 GC、运行器和 CEVA 仿真器原样复用。通过统一的接口契约与来源校验连接两端，最终完成 65 张自研子图编译和 10 步全模型解码。

### 1.3 完成情况

| 需求编号 | 需求内容 | 完成情况 | 实现位置 | 验证用例或证据 |
| --- | --- | --- | --- | --- |
| R1（P0-1） | 固定浮点基线，全部 `bin` 自研生成 | 已完成 | `gml_bridge/ceva.py:175`、`:259` | 清单 327/327；199353 个文件均为 `generated` |
| R2（P0-2） | 导出并编译完整 65 图 | 已完成 | `gml_bridge/ceva.py:1034`、`/data/fjg/ceva/scripts/compile_gml.py:159` | 65/65 图；编译清单无问题 |
| R3（P0-3） | 符合 GML 与运行器接口 | 已完成 | `contracts/ceva.py:15`、`gml_bridge/ceva.py:758` | 引用、尺寸、结构和字段检查通过 |
| R4（P0-4） | 接通厂商 GC 与 CEVA 执行链 | 已完成 | `scripts/run_full_pipeline.py:696`、`/data/fjg/ceva/run_all.sh:202` | Registry 全部来自本运行目录；原始目录检查通过 |
| R5（P0-5） | 完整运行并判定词元 | 已完成 | `/data/fjg/ceva/scripts/verify_run.py:105` | 10 步 `decode`；21 项通过、1 项告警 |

需求共 5 项，全部完成。本次计数的单元与方言测试共 593 条，全部通过；另有 327 项产物检查和 22 项硬件结果检查。当前记录 3 项边界或已规避事项，详见第八节。

## 二、总体流程

### 2.1 总体流程图

整体流程以 GML 和配套 `bin` 为自研编译与厂商执行的边界。首先，图编译器把完整模型展开为 65 张计算图，并完成量化、标定和文件写盘。然后，算子编译器为图内算子生成相位与硬件数值字段。其次，来源与接口校验确认产物完整且符合运行器契约。最后，厂商 GC、DDR 工具和 CEVA 仿真器原样接收这些产物，完成注册表生成、内存布局和全模型推理。

如图所示，自研部分负责生成正确、完整、可追溯的编译产物，厂商部分负责层内编排和硬件执行。浮点检查点与厂商图契约只作为输入约束，不替代我方生成的 GML 和 `bin`。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff,color:#fff

  A["图编译<br/>65 张计算图"]:::stage --> B["算子编译<br/>相位与硬件字段"]:::stage
  B --> C["产物写盘<br/>GML + 全部 bin"]:::stage
  C --> D["来源与接口校验<br/>65 图一致"]:::stage
  D --> E["厂商 GC<br/>Registry"]:::stage
  E --> F["DDR 布局<br/>厂商运行器"]:::stage
  F --> G["CEVA 硬件仿真<br/>结果判定"]:::stage
  W["浮点检查点<br/>厂商权重反量化"]:::input -.-> C
  R["厂商图契约<br/>结构与接口"]:::input -.-> D
```

### 2.2 模块关系

| 模块 | 职责 | 改动类型 | 依赖 |
| --- | --- | --- | --- |
| flagos-pim-compiler | 图生成、标定、量化、写盘与清单 | 新增并修改 | FlagTree、厂商浮点权重与图契约 |
| FlagTree | 相位、数值模式和硬件字段 | 修改 | 图编译器生成的共享 IR |
| CEVA 工作区 | GC 接线、DDR、仿真与判定 | 新增并修改 | 自研包、厂商 GC 和运行器 |
| GeneSim | 保留原有多设备评估链 | 未修改 | 与 CEVA 链无运行时依赖 |
| 厂商工具 | GC、输入工具、运行器、硬件仿真 | 原有复用 | 只接收本运行目录中的自研产物 |

## 三、核心原理

### 3.1 关键机制

为了让自研 GML 和 `bin` 被厂商 GC 正确编译并在 CEVA 上执行，需要同时保证结构可接、数值可行、来源可查。然而，旧链路存在模型覆盖不全、数值模式不一致和厂商 `bin` 混入等不足。总体采用契约约束的分层生成方法：以浮点检查点统一权重基线，以图结构驱动数值策略，以双层摘要保证产物来源。

**浮点检查点与交付产物分离**旨在保证进入 GC 的数据文件均由我方生成，同时继承厂商权重已有的旋转与归一化折叠。采用“厂商量化权重反量化、浮点模型重建、我方重新量化写盘”的技术路线。`vendor_float_weights()` 先恢复浮点权重，`load_vendor_checkpoint()` 再装入模型，`write_generated_files()` 最后生成全部 `bin`。厂商文件由此只参与浮点检查点构造和结构核对，不再进入交付包。

**按图结构选择数值路径**旨在让量化与标定规则摆脱层号和参数命名，适应不同包装方式。采用以数据流和形状为依据的通用生成方法：沿下游缓存写入者识别 V 投影，用 `CompileSlots.tokens` 区分单行与多行几何，用 `pim_static_quant` 表达静态量化。这样即使投影名称改变，数值路径仍由真实拓扑决定，不会发生静默回退。

**双层级来源摘要**旨在保证同一包内生成口径一致，并确保仿真注册表对应当前输入。采用“导出侧逐图摘要加 GC 侧编译摘要”的两级追踪方法。导出侧覆盖两端编译器源码、浮点检查点和标定结果，GC 侧覆盖全部输入、注册表和编译环境。任何一级摘要不一致都阻止复用，从而形成从源码到硬件输入的可追溯链。

### 3.2 设计取舍

| 取舍点 | 采用方案 | 未采用方案 | 理由 |
| --- | --- | --- | --- |
| 浮点权重来源 | 厂商量化权重反量化 | 复现旋转与归一化折叠 | 保留同一坐标系，避免复制厂商 `bin` |
| 动态缓冲 | 按声明尺寸写零 | 写入标定样本 | 样本不代表硬件运行时数据 |
| 输出头 | 整型静态量化加 `fixed2float` | 通用动态量化 | 与厂商输出头数值接口一致 |
| 注册表复用 | 三类摘要一致才复用 | 每次重编或只看 GML | 兼顾编译耗时与来源正确性 |
| 词元判定 | 自研包告警、厂商包严格 | 全部逐词元一致 | 允许自研标定造成合法分叉 |

### 3.3 适用前提与边界

- 前提：模型为当前 Llama2 7B 契约；其他模型不在本次验收范围内。
- 前提：CEVA 导出必须带 `--use-opcompiler`；缺失时入口直接拒绝。
- 前提：完整包必须覆盖参考目录全部 65 图；不完整时不能进入 DDR。
- 前提：告警档只接受全部文件为 `generated` 的完整包；否则判定失败。
- 边界：词元合法不等于与厂商一致；数值差异以告警形式保留。

## 四、关键问题与解决方案

围绕上述链路，实施主要涉及产物来源、完整图几何、数值模式、生成通用性和执行可信性五类问题。以下问题均来自需求、设计或评审记录，解决方案以最终代码为准。

### 4.1 厂商数据误入交付包（来源：request §P0-1，design §4.1）

为了保证进入 CEVA 的文件都来自自研编译，需要切断厂商 `bin` 的直接复制路径。其原因是早期写盘按语义绑定厂商文件，图虽重新生成，数据却没有重新生成。本文采用“反量化后再生成”的方法：厂商权重只还原为浮点检查点，我方量化、标定、相位和写盘器重新生成全部 `bin`，从原理上消除厂商字节混入。实现位于 `gml_bridge/ceva.py:145`、`:175`、`:259`，清单校验 327/327 通过。

### 4.2 单行几何无法覆盖预填充（来源：design §4.2～4.4）

为了覆盖完整的 65 图推理，需要解决旧导出只有单行 `decode`、缺少多行归约和逐层缓存的问题。本文采用三类包装器构造真实计算图，用 `index_copy` 表达缓存原地更新，并按查询行组织注意力归约，使 `prefill`、`decode` 和输出头共享同一套结构约束。实现位于 `gml_bridge/ceva.py:729`、`graph/kv_dma_pass.py:86`、`gml_bridge/phase_data.py:239`，由 `test_stage_geometry`、`test_softmax_rows` 和 `test_cache_update` 验证。

### 4.3 输出头数值模式被覆盖（来源：design §4.8～4.10，R2-6）

为了让输出头按整型静态量化执行，需要避免相位展开把显式定点模式改回浮点。其原因是旧方言只接受四位模型权重，`stampMatmul()` 又会覆盖已有配置。本文扩展四/八位权重契约，保留显式 `fixed2float`，并强制校验定点累加模式，使算子编译结果与 GML 数值字段一致。实现位于 `FlagTree/lib/Dialect/TritonPIM/IR/Ops.cpp:518`、`Transforms/ExpandPhases.cpp:558`，方言测试 265 条通过。

### 4.4 参数名称改变数值口径（来源：R6-1、R7-3）

为了让生成策略不依赖特定命名，需要解决 V 投影改名后量化路径失效的问题。本文沿布局算子追踪到 V 缓存写入者，同时覆盖缓存 DMA 和直接图输出两种拓扑；写入者就是投影本身时也直接识别。这样数值策略由数据流决定，而非由 `v_proj` 字符串决定。实现位于 `gml_bridge/from_fx.py:2281`、`:2297`，由投影改名和直接输出两类测试验证。

### 4.5 标定样本冒充运行时数据（来源：R5-3、R8-4）

为了区分编译期常量与硬件运行时数据，需要避免把标定样本写入动态缓冲。本文按字段语义识别 DQ 输出缩放、Kantor 缩放和 Softmax 运行时落点，并统一写零；权重、LUT、静态标定和旋转表仍保留真实生成值。这样既提供 GC 所需尺寸，又不伪造运行时输入。实现位于 `gml_bridge/ceva.py:233`、`:267`，由动态缓冲写零及静态文件非零测试验证。

### 4.6 局部续导混用生成口径（来源：R5-1、R6-2、R7-2、R8-1）

为了保证同一包内所有图使用一致的量化与相位口径，需要防止局部续导混用不同源码、标定或 FlagTree 版本。本文把图编译器工作树、FlagTree 源码、浮点检查点和标定结果纳入逐图摘要，保留图中任一项不一致就拒绝续导。该约束从来源上保证 `prefill` 写入和 `decode` 读取使用相同缓存缩放。实现位于 `gml_bridge/ceva.py:975`、`:993`、`:1020`、`:1034`，由中断续导测试验证。

### 4.7 旧注册表被错误复用（来源：R1-1、R3-1、R3-3）

为了保证仿真使用的注册表确实对应当前 GML 和 `bin`，需要阻止内容或 GC 配置变化后的旧产物复用。本文同时比较全部输入文件、注册表和 GC 环境摘要，并只在编译与 DDR 成功后写入完成标记。这样复用不会绕过来源检查，失败残留也不能进入后续步骤。实现位于 `/data/fjg/ceva/scripts/compile_gml.py:71`、`:80`、`:86`、`:119`，由 `test_does_not_reuse_when_bin_changes` 和 DDR 摘要检查验证。

### 4.8 共享目录外部写入导致误判（来源：implement §10～11）

为了保留原始目录保护并排除其他用户的写入，需要区分外部变化和本流程可写路径变化。本文采用写权限归因：只有当前用户无权限造成的变化才允许告警，可写路径变化仍失败；`--from 7` 复用原快照，避免续跑覆盖证据。实现位于 `/data/fjg/ceva/scripts/diff_snapshot.py:28`、`/data/fjg/ceva/run_all.sh:265`，4 条单元测试和 28 秒真实续跑均通过。

## 五、改动说明

### 5.1 改动总览

已跟踪差异中，图编译器涉及 59 个文件，FlagTree 涉及 10 个文件，CEVA 涉及 21 个文件。三者合计增加 3099 行、删除 5287 行，统计含运行日志与归档，不含未跟踪新文件。另有图编译器新增源码/测试 8 个、CEVA 新增测试 2 个。GeneSim 无改动。

### 5.2 文件改动清单

下表按功能分组列出当前工作树文件。过程文档、技术汇报和运行归档单独列在最后；本技术文档为本次新增，不计入上述统计。

| 文件路径 | 改动类型 | 改动内容 | 对应需求 |
| --- | --- | --- | --- |
| `contracts/ceva.py`、`contracts/pwla.py` | 新增 | CEVA 契约与分段线性表 | R1、R3 |
| `gml_bridge/ceva.py` | 新增 | 三类图导出、浮点还原、写盘与摘要 | R1～R3 |
| `quant/calibration.py` | 新增 | 整模型标定与静态入口缩放 | R1、R3 |
| `scripts/run_ceva_pipeline.sh` | 新增 | 自动加载环境的一键入口 | R4 |
| `tests/test_ceva_export.py`、`test_pwla.py`、`test_quant_calibration.py` | 新增 | 导出、查表与标定测试 | R1～R3 |
| `contracts/compile_slots.py`、`gml_quant.py`、`ir_payloads.py`、`mem_layout.py`、`unified_ir.py` | 修改 | 查询几何、分组、通用语义标记 | R2、R3 |
| `genesim_bridge/paths.py`、`paths.json` | 修改 | CEVA、FlagTree 与嵌入路径 | R1、R4 |
| `gml_bridge/export.py`、`from_fx.py`、`phase_data.py`、`runtime_files.py` | 修改 | 序列化、量化写盘、多行相位 | R1～R3 |
| `graph/fuse_rope.py`、`kv_dma_pass.py`、`quant_pass.py` | 修改 | 旋转编码、缓存与结构化量化 | R2、R3 |
| `opcompiler_bridge/oplevel_emitter.py`、`oplevel_kernel.py`、`phase_plan.py`、`phase_source.py` | 修改 | 输出头相位与跨相依赖 | R3 |
| `quant/weights.py` | 修改 | 整型量化、布局与次正规保护 | R1、R3 |
| `scripts/export_gml.py`、`run_full_pipeline.py`、`verify_gml_artifact.py` | 修改 | CEVA 入口、全流程与清单验收 | R1～R5 |
| `orchestrator/__init__.py`、`l2_alloc.py`、`layer_expand.py`、`layer_fields.py`、`layer_hw_table.py`、`layer_id.py`、`layer_render.py`、`net_ini.py`、`plan.py` | 删除 | 移除自研 `prepare_out` 支线 | R4 |
| `scripts/diff_prepare_out.py` | 删除 | 移除旧支线比较工具 | R4 |
| `tests/conftest.py`、`test_dtype_carrier.py`、`test_gml_from_fx.py`、`test_gml_quant.py`、`test_ir_cost_eval.py` | 修改 | 图、量化与流程回归 | R1～R4 |
| `tests/test_mem_layout.py`、`test_no_bypass.py`、`test_oplevel_emitter_live.py`、`test_paths.py` | 修改 | 删除旧职责并覆盖新接口 | R3、R4 |
| `tests/test_phase_data.py`、`test_phase_plan.py`、`test_phase_source.py` | 修改 | 多行相位与读取关系测试 | R3 |
| `tests/test_quant_pass.py`、`test_quant_weights.py`、`test_runtime_files.py`、`test_runtime_files_phases.py` | 修改 | 量化、边界与写盘测试 | R1、R3 |
| `tests/test_unified_ir_contract.py`、`test_verify_layers.py` | 修改 | 通用契约与逐层校验 | R3 |
| `tests/test_diff_prepare_out.py`、`test_layer_fields.py`、`test_orchestrator.py`、`test_stride_parity.py` | 删除 | 移除旧支线专用测试 | R4 |
| `CLAUDE.md`、`README.md`、三份 `docs/prepare_out-*.md` | 修改 | 更新职责与废止说明 | R4 |
| FlagTree `PIMAttrDefs.td`、`Dialect.cpp`、`Ops.cpp`、`ExpandPhases.cpp`、`LowerPIMToEmitC.cpp` | 修改 | 八位权重与 `fixed2float` 契约 | R3 |
| FlagTree `expand_phases.mlir`、`lower_to_emitc_ops_negative.mlir`、`operator_ops_negative.mlir` | 修改 | 新模式正负例 | R3 |
| FlagTree `README.md`、`claude.md` | 修改 | 同步方言索引 | R3 |
| CEVA `run_all.sh`、`scripts/compile_gml.py`、`scripts/verify_run.py` | 修改 | GC、DDR、仿真与严格判定 | R2、R4、R5 |
| CEVA `scripts/diff_snapshot.py` | 修改 | 区分外部目录变化 | R4 |
| CEVA `tests/test_compile_gml.py`、`test_verify_run.py` | 修改 | 来源、摘要和判定边界测试 | R2、R5 |
| CEVA `tests/test_diff_snapshot.py`、`test_gitignore.py` | 新增 | 目录检查与归档一致性测试 | R4 |
| CEVA `.gitignore`、`CLAUDE.md`、`README.md` | 修改 | 归档规则与使用说明 | R4 |
| `docs/request-*`、`design-*`、`implement-*`、11 份 `docs/review-*`、`docs/ceva-technical-ppt-20261102.md` | 新增 | 需求、设计、实施、评审与汇报 | 全部 |
| CEVA `runs/ours_e2e/**`、`ours_e2e2/ARCHIVE.md`、`ours_full/ARCHIVE.md`、`own_bins_*/**` 等 | 新增或更新 | 硬件日志、判定与归档证据 | R2、R4、R5 |

### 5.3 核心函数清单

| 函数/类 | 位置 | 改动类型 | 作用 | 调用方 |
| --- | --- | --- | --- | --- |
| `run_ceva()` | `scripts/run_full_pipeline.py:696` | 修改 | 串联导出、校验与硬件流程 | `main()` |
| `_main_ceva()` | `scripts/export_gml.py:501` | 新增 | 加载浮点检查点并导出包 | `main()` |
| `load_vendor_checkpoint()` | `gml_bridge/ceva.py:175` | 新增 | 反量化厂商权重装入模型 | `_main_ceva()` |
| `export_ceva_bundle()` | `gml_bridge/ceva.py:1034` | 新增 | 逐图导出并阻止混合口径 | `_main_ceva()` |
| `export_ceva_graph()` | `gml_bridge/ceva.py:758` | 新增 | 序列化、核对并写出一张图 | `export_ceva_bundle()` |
| `write_generated_files()` | `gml_bridge/ceva.py:259` | 新增 | 生成全部 `bin` 与来源条目 | `export_ceva_graph()` |
| `quantize_weight()` | `quant/weights.py:48` | 修改 | 逐组生成四位码值与缩放 | `write_runtime_files()` |
| `phase_source_from_graph()` | `opcompiler_bridge/phase_source.py:171` | 修改 | 运行 FlagTree 并返回相位 | `export_ceva_graph()` |
| `_compile_one()` | `/data/fjg/ceva/scripts/compile_gml.py:159` | 修改 | 隔离编译一张图并记录摘要 | `compile_gml.main()` |
| `verify_run.main()` | `/data/fjg/ceva/scripts/verify_run.py:105` | 修改 | 判定硬件输出与注册表来源 | `08_result.sh` |

### 5.4 数据结构与接口变更

为了统一表达三类子图及运行器按名称绑定地址的接口，需要描述阶段、层号、查询长度、输入输出槽位、形状、类型和缩放因子。已有共享导出结构没有承载这些 CEVA 边界信息，无法在图生成、字段核对和写盘之间保持一致。因此新增图与输入输出契约，并把厂商目录限定为只读契约来源。

**图与输入输出契约**（`contracts/ceva.py:15`）

```python
@dataclass(frozen=True)
class CevaIoBinding:
    direction: Literal["input", "output"]
    index: int
    name: str
    label: str | None
    shape: tuple[int, ...]
    dtype: str
    sf: float

@dataclass(frozen=True)
class CevaGraphSpec:
    stage: Literal["prefill", "decode", "lm_head"]
    layer_id: int | None
    query_tokens: int
    slots: CompileSlots
    version: str
    io: tuple[CevaIoBinding, ...]
    runtime_dir: Path
```

该结构只在 CEVA 导出内部传递，不要求共享导出调用方增加参数。

为了保证同层 `prefill` 与 `decode` 使用一致的缓存缩放，需要把整模型标定结果按层传递到各子图，并保留输出头的标定输入。已有量化接口主要处理单个张量，缺少跨层标定结果载体。因此新增整模型标定结果和单图缩放视图，使一次标定贯穿全部子图。

**标定结果**（`quant/calibration.py:47`）

```python
@dataclass(frozen=True)
class ActivationScales:
    key: float | None = None
    value: float | None = None
    static_input: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class LlamaCalibration:
    kv: list[tuple[float, float]]
    head_inputs: list = field(default_factory=list)
```

新增类型不改变旧量化函数；CEVA 导出调用方需使用同一次标定结果。

为了保留相位之间的读取依赖和输出头的显式数值模式，需要在算子编译结果中记录“当前相读取哪些前序相”以及 FPSU 名称。已有相位结构只描述执行单元和字节等属性，无法完整交叉校验这些信息。因此补充 `reads` 和 `fpsu_mode_name` 两个字段。

**算子相位**（`opcompiler_bridge/phase_plan.py:143`）

```python
@dataclass
class Phase:
    # 既有相位字段省略
    reads: tuple[int, ...] = ()

@dataclass
class PhasePlan:
    # 既有计划字段省略
    fpsu_mode_name: str | None = None
```

默认值保持旧构造方式；启用算子编译器后，输出头缺少显式模式会直接失败。

为了在共享目录被其他用户写入时仍能判断本流程是否造成污染，需要区分外部变化和可写路径变化，并支持只续跑目录检查。原入口只有统一失败语义，续跑还会覆盖原快照。因此新增检查模式和第 7 步续跑接口，默认仍保持严格检查。

**原始目录检查接口**（`/data/fjg/ceva/run_all.sh:13`）

```bash
--orig-check strict
--orig-check external
--from 7 --to 7
```

默认 `strict` 等价于原行为；`external` 只豁免当前用户无权限造成的变化。

## 六、验证

### 6.1 验证环境

代码基线见文档头部。运行环境为 Linux、Python 3.10.20、PyTorch 2.9.1、Transformers 4.57.6。硬件验证使用单个 CEVA NPM 核、4 个 engine、固定 prompt 和贪心解码。FlagTree 使用已同步的安装与构建目录。

### 6.2 验证命令与结果

为了同时证明“既有功能未回归”和“自研产物能够完成 CEVA 全流程”，本文采用分层验证方法。首先用图编译器与 FlagTree 回归检查既有图变换、量化和相位功能；然后围绕全部 `bin` 自研生成、65 图完整性、通用结构识别和跨相依赖增加针对性测试；其次通过清单摘要、引用集合和文件尺寸做产物静态校验；最后用注册表来源、10 步硬件日志和词元文件复核端到端执行。这样既能在本地定位生成问题，又能避免仅凭单元测试推断硬件可运行。

以下计数来自本次文档编写前已执行的命令。用户另确认全量测试已手动通过；本文未再重复执行耗时回归和 7 小时仿真。

| 验证项 | 命令 | 结果 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| 图编译器相关测试 | `python -m pytest tests/test_ceva_export.py ... tests/test_pwla.py -q` | 272 通过，0 失败 | 通过 | 覆盖导出、量化、写盘、相位 |
| CEVA 全量测试 | `python -m pytest tests/ -q` | 56 通过，0 失败 | 通过 | 含目录检查与归档一致性 |
| FlagTree 方言测试 | `lit build/flagtree-cmake/test --no-progress-bar` | 265 通过，0 失败 | 通过 | 含输出头正负例 |
| 自研包清单 | `python scripts/verify_gml_artifact.py ... --ceva-manifest ...` | 327/327 通过 | 通过 | 65 图、199353 个文件 |
| 现有硬件结果复核 | `TOKEN_CHECK=warn GML_BUNDLE=... python3 scripts/verify_run.py runs/ours_e2e baseline` | 21 通过、1 告警、0 失败/跳过 | 通过 | 10 步、11 个合法词元 |
| 第 7 步单独续跑 | `bash run_all.sh ... --orig-check external --from 7 --to 7` | 35627 处外部变化；可写变化 0 | 通过 | 28 秒，前 6 步未执行 |
| 图编译器全量回归 | `python -m pytest tests/ -q` | 用户确认全部通过 | 通过 | 本文未重复记录分项条数 |
| 完整硬件流程 | `bash scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10` | 65/65 图；10 步；硬件 PASS | 通过 | 复核已有日志与产物，未再次仿真 |

复现命令：

```bash
cd /data/fjg/pengcheng/flagOS/flagos-pim-compiler
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python -m pytest tests/test_ceva_export.py tests/test_quant_weights.py \
  tests/test_quant_calibration.py tests/test_runtime_files.py \
  tests/test_runtime_files_phases.py tests/test_gml_from_fx.py \
  tests/test_verify_layers.py tests/test_phase_plan.py \
  tests/test_phase_source.py tests/test_pwla.py -q
python scripts/verify_gml_artifact.py /data/fjg/ceva/bundles/ours_e2e \
  --ceva-manifest /data/fjg/ceva/bundles/ours_e2e/bundle.json

cd /data/fjg/ceva
python -m pytest tests/ -q
TOKEN_CHECK=warn GML_BUNDLE=/data/fjg/ceva/bundles/ours_e2e \
  python3 scripts/verify_run.py runs/ours_e2e baseline
bash run_all.sh -n ours_e2e --gml-bundle /data/fjg/ceva/bundles/ours_e2e \
  -s 10 --token-check warn --orig-check external --from 7 --to 7

source /data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/env-flagtree.sh
/data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/python-3.10.20/bin/lit \
  /data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/test \
  --no-progress-bar
```

### 6.3 验收标准对照

| 验收标准 | 验证方法 | 结论 | 证据 |
| --- | --- | --- | --- |
| 所有文件均为自研生成 | 来源、摘要、引用和尺寸检查 | 满足 | 327/327；199353 个 `generated` |
| 尺寸与参考同字段文件一致 | 逐图解析 GML 字段 | 满足 | 6.2 自研包清单 |
| 65 图完整且不得混入旧注册表 | 编译清单与 DDR 路径、摘要 | 满足 | 65/65；外部注册表 0 |
| 运行器接口正确 | 结构、形状、字段和输入输出核对 | 满足 | `export_ceva_graph()` 强制检查 |
| 动态缓冲不写标定样本 | 具名字段写零测试 | 满足 | R5-3、R8-4 回归用例 |
| 原始目录不受本流程污染 | 快照与写权限归因 | 满足 | 外部变化 35627；可写变化 0 |
| 正常结束且核心检查零跳过 | 复核硬件日志与判定 | 满足 | 21 通过、1 告警、0 失败/跳过 |
| 至少完成 1 步，目标 10 步 | 日志与词元文件交叉核对 | 满足 | 实际 10 步、11 个词元 |
| 自研包允许合法分叉 | 词表、EOS、步数独立检查 | 满足 | 前 7 个一致，之后合法分叉 |
| 性能偏差只告警 | 各阶段 cycles 对拍 | 满足 | 本运行各阶段均在 ±20% 内 |

### 6.4 评审情况

本需求共进行 11 轮评审，形成 58 条报告项，含跨轮重复与复核项。其中高优先级 12 条、中优先级 28 条、低优先级 18 条。各轮核心问题及最终状态如下，重复问题在后续轮次中按最终修复结果记录。

| 评审轮次 | 核心问题 | 修复状态 | 最终证据 |
| --- | --- | --- | --- |
| 第 1 轮 | 注册表复用遗漏 `bin`；跨相依赖被跳过；调试包误标完整；单层失败被放行 | 已修复 | 全输入摘要、强制读取关系校验、完整性判断和单层失败测试 |
| 第 2 轮 | 厂商原始文件曾被覆盖；续跑绕过来源；中断导出残留完整标记；算子编译可选；输出头缺少 `fixed2float` 约束 | 已修复 | 原始文件还原、续跑守卫、原子清单、强制算子编译和方言负例 |
| 第 3 轮 | DDR 成功标记可能过期；复用不含 GC 配置；厂商比较可选；失败测试只检查脚本文本 | 已修复 | `ddr_ok.json`、编译环境摘要、必填参考输出和行为测试 |
| 第 4 轮 | 通用化改动后缺少 65 图证据；精化产物档位混用；空 DDR map 被接受；特殊路径绕过检查 | 已修复并闭环 | 后续全模型运行、按档位命名、map 必需项和字面前缀检查 |
| 第 5 轮 | 局部续导混用标定；清单缺少引用与尺寸核对；动态缓冲写样本；告警摘要误称一致 | 已修复 | 生成摘要、327 项清单检查、动态缓冲写零和告警状态分离 |
| 第 6 轮 | V 投影按名称识别；提交号不能覆盖未提交源码；写零后缺少全链证据；运行归档被忽略 | 已修复并闭环 | 结构识别、工作树摘要、后续硬件运行和归档恢复 |
| 第 7 轮 | 验证未完成先标修复；摘要缺少 FlagTree；V 出口为投影本身时失效 | 已修复并闭环 | 状态按证据更新、算子编译器摘要和直接写入者测试 |
| 第 8 轮 | 中断续导混口径；当前全链未闭环；历史证据删除；动态字段漏零；空源码树有效化；多行 Softmax 与定点边界误判 | 已修复并闭环 | 逐图摘要、10 步硬件运行、归档测试及各项边界回归 |
| 第 9 轮 | 最新硬件证据尚未完成；告警档接受越界词元 | 已修复并闭环 | 65 图 10 步运行和独立词表范围检查 |
| 第 10 轮 | 非末尾 EOS 被接受；厂商模式也能使用告警档 | 已修复 | EOS 唯一且末尾、自研来源与 DDR 标记联合检查 |
| 第 11 轮 | 零步解码通过；旧归档覆盖新结果；方言索引未同步；第 10 轮问题复核 | 已修复 | 正步数守卫、归档一致性测试、文档索引和合并回归 |
| 后续补充 | 共享目录被其他用户写入导致总流程失败；第 7 步续跑覆盖原快照 | 已修复 | 写权限归因、原快照复用和 28 秒真实续跑 |

实施文档已逐项记录修复、后续验证关闭或文档口径修正。当前无未修复评审项；第十、十一轮的重复问题已合并验证，来源、EOS、零步和归档边界均有行为测试。

## 七、与设计的差异

| 设计条目 | 设计方案 | 最终实现 | 差异原因 | 是否经确认 |
| --- | --- | --- | --- | --- |
| design §4.11 | 新增 `compile_gml.sh` | `compile_gml.py` 加 `gc_worker.sh` | 并行、报告和摘要处理更直接 | 是，implement §5.2 |
| design §4.12 | 每图必须重新生成注册表 | 三类摘要一致时允许复用 | 减少重复 GC，同时防止旧产物 | 是，implement §5.2 |
| design §4.13、早期 P0-5 | 全部逐词元一致 | 自研包合法分叉只告警 | 自研标定与厂商不完全相同 | 是，request 2026-11-01 修订 |
| design §7.2 | 单层输出逐字节一致 | 自研 `bin` 改为数值误差比较 | 量化缩放允许不同 | 是，implement §8.5 |
| design §5.2～5.3 | 共享导出接口增加 `graph_spec` | 契约留在 `gml_bridge/ceva.py` 内部 | 共享导出无需感知 CEVA | 是，implement §5.2 |
| design 早期元数据键 | 使用 `pim_ceva_op` | 使用 `pim_static_quant` | 按通用数值语义而非平台命名 | 是，G-2 |
| design §4.12 后续接线 | 严格目录变化检查 | 另增显式 `external` 模式 | 区分共享目录的外部写入 | 是，implement §11 |
| design §4.14 | 删除自研 `prepare_out` | 已删除相关模块、脚本和测试 | 层内编排交给厂商 GC/L2A | 无差异 |

## 八、当前遗留问题

| 编号 | 问题描述 | 影响范围 | 来源 | 状态 |
| --- | --- | --- | --- | --- |
| L1 | 自研标定值与厂商不同，后段词元分叉 | 固定用例第 8 个词元起 | implement §8.6 K8-1 | 已规避：告警档允许，合法性与执行仍严格检查 |
| L2 | 通用性指规则不按层号或名称写死，不等于支持任意模型 | 其他模型、prompt、多核配置 | request §1.2、§2.3 | 边界明确：本期只验收 Llama2 7B 固定流程 |
| L3 | 厂商 `test_app` 会截断含 `data` 的路径，GC 也会写工作目录 | GC 输入路径与工作副本 | implement §5.1 | 已规避：使用 `/tmp/ceva_gc` 摘要软链和 `env/gc_w<N>` 隔离副本 |

上述事项均未阻断本需求验收。R1～R5 全部完成，11 轮评审无未修复项，本文列出的测试与硬件结果检查均通过。
