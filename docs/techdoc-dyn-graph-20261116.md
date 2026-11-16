# 技术文档：静态计算图 → host 驱动动态图（dyn-graph）

> 文档编号：techdoc-dyn-graph-20261116
> 创建日期：2026-11-16
> 关联文档：request-dyn-graph-20261116、design-dyn-graph-20261116、implement-dyn-graph-20261116、review-dyn-graph-20261116-round01/02/03、dynamic-graph-20261113（过程日志）
> 关联项目：flagos-pim-compiler（存算一体 PIM 大模型推理编译器）
> 代码基线：pengcheng @ 7fcf056，含 round01～round03 评审修复的未提交改动
> 核心目标：在静态路径保持默认与逐字节兼容的前提下，把编译期写死的静态蓝图改造为 host 驱动的动态图——运行期绑定实际形状、按计划序调度提交、按实际 T 收缩通信行数，单请求墙钟不高于静态路径的 105%。

## 一、概述

### 1.1 背景与问题

　　本需求的核心目标是把「编译期决策、运行期照表执行」的静态蓝图升级为 host 驱动的动态图。现有基础是一条完整的静态编译链：`torch.export` 用具体张量导出，prefill 每个桶一张图、decode 固定 T=1；分区与规格传播按静态形状计算；权重、KV、激活三区 offset 编译期写死；`ExecutionPlan` 的 `waits` 编译期预计算，执行器一次性全量提交后统一等待。但它缺少运行期建图、动态形状与动态内存三项能力——现有的「动态」只剩 MoE 的 `selected_mask` 加提交前 `gate` 跳过，图仍是全专家静态展开。本需求要求在不破坏「静态默认、逐字节兼容、可随时回退」边界的前提下完成改造；设备内不做数据依赖分支、不做 DPU 直连、不做逐 op JIT。

### 1.2 方案摘要

　　围绕该目标，总体采用「双轨制 + 分层落地」的技术路线：规划、地址与容量始终只用 T_max 包络整数，符号形状表达式随命令 payload 下发，运行期按实际 T 求值绑定。`contracts/runtime_graph.py` 与 `runtime/dependency.py` 提供契约与依赖判定（D0）；`runtime/scheduler.py` 以计划序全量提交并逐命令绑定（D1）；`runtime/compile.py` 导出单张符号 T 的 prefill 图并捕获表达式（D2-lite）；`runtime/executor.py` 与 `comm/lowering.py` 完成输入去填充与通信行收缩（D2-full）。`PIM_DYN_SHAPE=1` 控制绑定开关，`PIM_RUNTIME_MODE=static` 提供运行期整体回退，最终以 token 逐位一致和 102.3% 的单请求墙钟比达成验收。

### 1.3 核心目标完成度

　　对照需求文档的五条 Go/No-Go 验收判据，逐条状态、证据与缺口见表 1；证据细节按五段证据链在第三节展开，命令与出处见第六节。

<p align="center">表 1　核心目标完成度矩阵</p>

| 验收判据 | 状态 | 证据（详见第三节） | 缺口 / 外部阻塞 |
|----------|------|--------------------|-----------------|
| 1 静态回归绿、可随时回退 | 已完成 | 回归 2276 绿；回退开关落地并有行为测试 | 无 |
| 2 依赖 shadow、token 一致 | 已完成 | shadow ⊆ 用例；tiny 模型 7+5 种长度对拍 | 4L 数字未独立复跑 |
| 3 单图覆盖 1..max_seq、形状类有界 | 已完成 | 桶参数强制为空；绑定不动缓存键 | `kernel_classes` 计量已删 |
| 4 单请求 ≤105%、并发不劣化 | 已完成（复用） | 102.3% / 93.4%，同轮独占口径 | 跨轮墙钟波动 52–70s |
| 5 新增行为同次测试、文档一致 | 已完成 | 三轮评审补 16 条用例并经变异验证 | 无 |

　　五条判据全部达成。测试证据中，定向子集与全量回归复用 round03 修复后的记录并经本次核对适用（代码在最后一次验证后未再变动），两个核心单测文件本次抽查 14 passed；当前遗留 6 条显式未做或待确认项，均属需求边界内的登记项，详见第八节。

## 二、总体流程

### 2.1 总体流程图

　　本需求改造的对象是「torch.export 静态导出 → 图编译 → ExecutionPlan → 执行器」这条既有链路的形态。输入侧不再按桶导出多张具体形状图，而是导出一张符号 T 图；编译产物在包络规划之外携带形状表达式；运行期由调度器按实际 T 绑定后提交。首先，D0 层定义 `RuntimeGraph`/`ShapeExpr` 契约，并把地址依赖判定抽成唯一实现。然后，D1 层的 `EventScheduler` 按计划序全量提交、在提交线程内逐命令绑定形状。其次，D2-lite 在编译期捕获符号表达式并保证 payload 对齐，D2-full 进一步去掉输入填充并按实际 T 钳制通信行数。最后，token 对拍与墙钟对照完成验收。如图 1 所示，五个阶段串成「符号 T 从导出贯穿到执行」的主线，静态兼容约束全程在侧。

```mermaid
flowchart LR
  classDef stage fill:transparent,stroke:#fff,color:#fff
  classDef input fill:transparent,stroke:#fff,color:#fff,stroke-dasharray:5 3
  linkStyle default stroke:#fff,color:#fff

  A["符号 T 导出<br/>单张 prefill 图"]:::stage --> B["D0 契约<br/>RuntimeGraph+依赖判定"]:::stage
  B --> C["D1 调度<br/>EventScheduler 计划序提交"]:::stage
  C --> D["D2-lite 绑定<br/>按实际 T 求值形状"]:::stage
  D --> E["D2-full 收缩<br/>去填充+通信行钳制"]:::stage
  E --> F["HAL 执行<br/>NumpyBackend 多 DPU"]:::stage
  F --> G["验收<br/>token 对拍+墙钟对照"]:::stage
  W["全程约束<br/>静态默认·逐字节兼容"]:::input -.-> B
```

<p align="center">图 1　动态图改造总体流程</p>

### 2.2 模块关系

　　动态图改造横跨契约、编译、执行、通信与后端五组模块，各模块的职责、改动类型与依赖关系见表 2。

<p align="center">表 2　模块关系</p>

| 模块/文件 | 职责 | 改动类型 | 依赖的模块 |
|-----------|------|----------|------------|
| `contracts/runtime_graph.py` | 符号形状与控制流契约 | 新增 | `contracts/unified_ir` |
| `runtime/dependency.py` | 地址依赖判定与折叠 | 新增 | `contracts/exec_plan` |
| `runtime/scheduler.py` | 动态执行调度器 | 新增 | `runtime/shape_bind` |
| `runtime/shape_bind.py` | 运行期绑定与两级开关 | 新增 | `contracts/runtime_graph` |
| `runtime/compile.py` | 动态导出与表达式捕获 | 修改 | `graph/`、`memory/` |
| `runtime/exec_plan_gen.py` | `payload` 表达式与行钳制 | 修改 | `comm/plan`、`runtime/dependency` |
| `runtime/executor.py` | 模式分派与去填充 | 修改 | `runtime/scheduler` |
| `comm/plan.py`、`comm/lowering.py` | `t_axes` 契约与行收缩 | 修改 | `contracts/unified_ir` |
| `backend/hal_numpy.py` | `gate` 同步解析 | 修改 | `contracts/exec_plan` |

## 三、证据链：按五段组织

　　本节按「D0 契约 → D1 调度 → D2-lite 形状绑定 → D2-full 通信收缩 → 性能与并发验收」五段组织全部技术证据。每段先给出入口与契约，再以对照表逐机制列出「原理/不变式 → 代码位置 → 测试或产物证据」。表中证据状态分三档：「已实测通过」指存在真实执行过的测试或探针证据（命令与出处见 6.2，含本次抽查与复用轮次记录）；「仅代码推理」指经静态核对但无执行证据；「未验证/待确认」指当前无法核实。行号均为本次实际打开核对的位置，基线为 pengcheng@7fcf056 加未提交修复。

### 3.1 D0：契约层

　　D0 段的入口是编译期与运行期共用的两块地基：`contracts/runtime_graph.py` 定义符号形状与控制流的数据契约，`runtime/dependency.py` 把地址区间依赖判定从计划生成器中抽出为独立模块。契约要点有三：`ShapeExpr` 只表达常量、符号维与算术组合，求值时缺维直接抛错、不猜默认值；地址依赖的 RAW/WAW/WAR 判定与 `waits` 折叠全仓只有一份实现；静态导出不写任何表达式键，`payload` 新键只在动态导出时出现。`RegionRef`/`RTOp`/`RTControl`/`RuntimeGraph`（`contracts/runtime_graph.py:116`–`154`）是为 D3 控制流预留的契约结构，本阶段执行器不消费。关键产物与机制对照见表 3。

<p align="center">表 3　D0 契约层机制与证据</p>

| 机制 / 不变式 | 代码位置 | 测试或产物证据 | 证据状态 |
|---------------|----------|----------------|----------|
| ShapeExpr 求值缺维即抛，不猜默认 | `contracts/runtime_graph.py:60` | `test_runtime_graph_contract.py`（4 条） | 已实测通过 |
| 地址依赖 RAW/WAW/WAR 单实现 | `runtime/dependency.py:15`、`:32`、`:56` | shadow 子集用例（表 4） | 已实测通过 |
| `waits` 折叠「同 DPU 留最后前驱、host 逐条」唯一真源 | `runtime/dependency.py:41` + `runtime/exec_plan_gen.py:457` | 计划依赖与数值用例（随子集与回归） | 已实测通过 |
| 静态计划不携带表达式新键 | `runtime/exec_plan_gen.py:841`（仅节点带 meta 才写） | `test_static_plan_has_no_shape_expr_payload` | 已实测通过 |
| meta 键登记于契约真源 | `contracts/unified_ir.py:50`、`:80` | 契约键成对写入/消费（`grep` 核对） | 已实测通过 |

　　本段对应的验收判据是判据 1 的「静态兼容」半句：`payload` 新键仅动态写入、`execute_plan` 静态分支在提交区间 `diff` 中未被修改、`gate` 逻辑未动，三处经 round01～round03 逐 hunk 核对确认（复用已有验证），静态全量回归 2276 passed / 0 failed。

### 3.2 D1：调度器

　　D1 段的入口是 `execute_plan(plan, hal, ctx, mode)` 的动态分支（`runtime/executor.py:366`，分支判定 `:387`），核心产物是 83 行的 `EventScheduler`（`runtime/scheduler.py:26`）与 `tests/test_runtime_scheduler.py`（7 条用例）。契约有四条。其一，命令按计划序全量提交，「提交序等于计划序」这一 FIFO 前提是静态 `waits` 折叠成立的依据。其二，`waits` 引用校验按计划身份只做一次，且随对象回收自动清除。其三，提交异常与执行异常都从 `wait()` 抛出。其四，`gate` 由 HAL 在 `submit` 内同步解析，跳过的命令产生预完成 event 而不入执行队列。设计初版的「就绪驱动提交」经剖析证伪（保序约束下无收益且 65% 墙钟零并发），收敛为计划序全量提交，调度器保留的价值是逐命令形状绑定与统一动态入口。机制对照见表 4。

<p align="center">表 4　D1 调度器机制与证据</p>

| 机制 / 不变式 | 代码位置 | 测试或产物证据 | 证据状态 |
|---------------|----------|----------------|----------|
| 计划序全量提交，提交线程内逐命令绑定 | `runtime/scheduler.py:66`（`_loop`） | `test_scheduler_submits_all_commands_in_plan_order` | 已实测通过 |
| 异常传播：提交与执行异常都从 `wait()` 抛出 | `runtime/scheduler.py:51` | `test_submit_error_surfaces_from_wait` 等 2 条 | 已实测通过 |
| 校验缓存弱引用登记，计划回收后不误跳过 | `runtime/scheduler.py:13` | `test_validation_cache_drops_collected_plans` | 已实测通过 |
| 未知 `waits` 依赖 fail-fast 拒绝 | `runtime/scheduler.py:17` | `test_plan_with_unknown_wait_is_rejected` | 已实测通过 |
| shadow：运行期地址依赖 ⊆ 静态 `waits` | `runtime/dependency.py:56`、`:86` | `test_address_waits_are_a_subset_of_static_waits` | 已实测通过 |
| `gate` 在 `submit` 内同步解析，跳过产生预完成 event | `backend/hal_numpy.py:246`、`:327` | `test_moe_dispatch` 既有用例 | 已实测通过 |
| 动态与静态在附录 A 计划上结果一致 | `runtime/executor.py:387` | `test_dynamic_execution_matches_static_on_appendix_a` | 已实测通过 |

　　本段对应判据 2 的 shadow 半句。断言语义是「运行期地址依赖 ⊆ 折叠后的静态 `waits`」而非逐命令相等：host 闭包经 `resolve_node` 产生的数据依赖不在地址分析内，以计划 `waits` 为准；该收敛由 round01 澄清并写入实施文档（需求原文措辞的偏差见第七节）。调度器单测文件本次抽查 7 条全绿（本次实际验证）。

### 3.3 D2-lite：符号 T 单图与运行期形状绑定

　　D2-lite 段的入口是 `compile_llama2(dynamic=True)`（`runtime/compile.py:386`）：动态编译强制额外桶参数为空（`:427` `ValueError`），`seq_lens` 收敛为 `(prefill_seq_len,)` 单桶（`:440`），prefill 用 `Dim("T", min=1, max=seq_len)` 导出单张符号图（`:450`）。契约有四条。其一，只符号化 T：从 `program.range_constraints` 取唯一符号名（`:240`，多符号直接报错），形状环境辅助符号退包络常量。其二，捕获必须两遍——先读完全部 sympy 表达式再统一取整，否则取整污染会让其余表达式退化成常量。其三，绑定判据是行主序前缀兼容：T 是唯一符号维且排在全部非单位维之前才绑定，否则整条命令保持包络。其四，双轨制——绑定只改 `arg_kinds`/`arg_shapes`/`out_shape` 三组形状键，`mram_offset` 与内核缓存键永不动。关键产物有 `runtime/shape_bind.py`（134 行）、`tests/test_dynamic_shape_compile.py`（16 条）与 `tests/test_shape_binding.py`（7 条）。机制对照见表 5。

<p align="center">表 5　D2-lite 形状绑定机制与证据</p>

| 机制 / 不变式 | 代码位置 | 测试或产物证据 | 证据状态 |
|---------------|----------|----------------|----------|
| 单图强制：额外桶参数必须为空 | `runtime/compile.py:427` | `test_dynamic_compile_records_mode_and_single_bucket` | 已实测通过 |
| 两遍捕获：先读全部表达式再取整 | `runtime/compile.py:127` | `test_dynamic_export_concretizes_meta_and_keeps_exprs` | 已实测通过 |
| 符号名取自 `range_constraints` 真名 | `runtime/compile.py:240` | 7 种长度 token 对拍（`test_dynamic_tokens_match_static_baseline`） | 已实测通过 |
| 辅助符号退包络常量，只认 `dyn_symbol` | `runtime/compile.py:96` | 转换异常传播测试 + token 对拍 | 已实测通过 |
| 前缀布局判据：T 唯一且居非单位维之前 | `runtime/shape_bind.py:46` | `test_bind_skips_layouts_that_do_not_shrink_as_a_prefix` | 已实测通过 |
| 绑定只改三组形状键，不改地址 | `runtime/shape_bind.py:85` | `test_bind_rewrites_shapes_for_symbolic_dims`（断言原命令不动） | 已实测通过 |
| `payload` 表达式与形状逐位对齐 | `runtime/exec_plan_gen.py:744`–`845` | 对齐红测 + `test_arg_shape_exprs_align_with_payload_positions` | 已实测通过 |
| 融合随 `val` 复制形状表达式 | `graph/opt_passes.py:417` | `test_fuse_copies_shape_exprs` | 已实测通过 |
| 字面量形状列表逐元素绑定 | `runtime/compile.py:187`、`runtime/shape_bind.py:101` | `test_bind_rewrites_literal_shape_lists` 等 | 已实测通过 |

　　本段对应判据 2 与判据 3：tiny Llama 7 种长度、tiny MoE 5 种长度逐 token 与静态基线一致（复用 round03 子集记录）；「一张图覆盖 1..max_seq」由桶参数 `ValueError` 加 `prefill_seq_lens==(T_max,)` 单桶共同保证，关绑定回退包络正依赖该字段补 pad。形状类有界是结构性保证——绑定不触碰 `mram_offset` 与内核查找键（`backend/hal_numpy.py:264` 按内核名查找），不随请求数增长；计量字段 `kernel_classes` 按 round01 评审裁决删除，属判据 3 的字面缺口（见第八节）。

### 3.4 D2-full：prefill 去填充与通信行收缩

　　D2-full 段的入口有两个：执行侧是 `prefill_chunk`/`prefill_bucket`（`runtime/executor.py:224`、`:274`），通信侧是 `CommPlanEntry` 的 `t_axes` 契约（`comm/plan.py:49`）与行钳制原语（`comm/lowering.py:97`）。执行侧契约有两条：动态图开绑定时输入按实际长度构建、掩码保持包络（`(1,1,T,T)` 行宽随 T 变，不能前缀收缩，标准因果掩码的融合注意力不读值）；「token 数不超过最长桶」的防线先于模式分支执行，两种模式共用同一段 `ValueError`。通信侧契约也有两条：条目必须显式声明 `t_axes`，缺失抛 `ValueError`、多 T 轴抛 `NotImplementedError`，禁止启发式静默裁剪；行钳制门控与内核绑定同一判据，即单 T 轴且 T 在最外层非单位维。行收缩只做在「结果不进主机」的条目上，边界见表 6。

<p align="center">表 6　通信行收缩边界</p>

| 条目类别 | 收缩规则 | 原因 |
|----------|----------|------|
| d2h（结果回主机） | 不收缩，保持包络 | 主机闭包混用包络形状的兄弟实参，全链统一实际 T 前不可收缩 |
| 专家子图条目（`gate` 打标） | 不收缩，保持包络 | 路由缓冲按每专家 token 混排，行数非 T 的行前缀 |
| d2d/d2c（结果只进 MRAM） | 按实际 T 行收缩 | 下游绑定命令读实际 T 行，未绑定读包络尾行，语义一致 |

　　d2h 豁免的直接原因是小 MoE 的主机闭包：`where(selected_mask, expert_out, 0)` 一侧是 `launch` 绑定出的实际 T 行、另一侧是包络形状，收缩即形状错配崩溃；专家条目豁免还因为跳过专家的 `skip_value` 本身是包络零张量。边界判定落在 `runtime/exec_plan_gen.py:534`（`clamped = tag is None and edge.dst_loc.get("device") != DEVICE_HOST`）。本段机制对照见表 7。

<p align="center">表 7　D2-full 机制与证据</p>

| 机制 / 不变式 | 代码位置 | 测试或产物证据 | 证据状态 |
|---------------|----------|----------------|----------|
| prefill 去填充：输入按实际 T 建、掩码保包络 | `runtime/executor.py:244`–`252` | `test_dynamic_prefill_uses_actual_length` | 已实测通过 |
| 超包络防线先于模式分支，两模式同文报错 | `runtime/executor.py:280`–`284` | `test_dynamic_prompt_over_envelope_raises` + 探针两臂同文 | 已实测通过 |
| `CommPlanEntry` 显式 `t_axes`，缺失/多轴拒绝 | `comm/plan.py:49`、`:233` | `test_clamped_entry_rejects_missing_or_multi_t_axes` | 已实测通过 |
| 行钳制门控：单 T 轴且 T 在最外非单位维 | `runtime/exec_plan_gen.py:499` | `test_clamped_rows_requires_dyn_env_and_prefix_layout`（四否定态） | 已实测通过 |
| 收集原语按 `rows` 收缩，mean 除数不变 | `comm/lowering.py:97`、`:171` | `test_clamped_entry_drops_segments_beyond_actual_rows` 等 4 条 | 已实测通过 |
| d2h/专家条目豁免（表 6 边界） | `runtime/exec_plan_gen.py:522`–`534` | `test_dynamic_chunk_binds_each_segment_length`（锁定 d2h 包络行数） | 已实测通过 |
| 关绑定整体回退包络，无半开状态 | `runtime/shape_bind.py:17` | `test_dynamic_without_binding_falls_back_envelope` | 已实测通过 |
| 动态分块：每段绑定不同 T、KV 按绝对位置推进 | `tests/test_chunked_prefill.py:129`、`:145` | 「模式 × 分段形状」矩阵 4 参数 + 契约 1 条 | 已实测通过 |
| 动态多请求 wave：双泳道并发绑定不同 T | `tests/test_multi_request_decode.py:113` | `test_two_requests_wave_match_individual_runs_dynamic` | 已实测通过 |

　　本段对应判据 2 的 token 一致性在组合路径上的延伸：动态分块矩阵覆盖整段锚点、首段 1 token（[1,8,3]）、尾段非整桶（[5,7] 与 8+4），全部与静态基线逐 token 一致（复用 round03 修复后记录：分块文件 15 passed / 2 deselected）；该组合此前无测试可拦截，由变异验证证明新增用例专守该组合（见 6.2）。

### 3.5 性能与并发验收

　　性能与并发验收对应判据 4，全部数字复用 2026-11-15 的同轮独占记录：4L 真实 MoE（`/data2/pengcheng-work/Qwen3-30B-A3B-4L`，8 DPU，prefill 8，prompt 5 token），同轮各臂同机同负载各跑一次，机器独占。测量口径必须同轮独占的原因是跨轮绝对值波动大：同一代码的历史区间内单请求墙钟在 52–70s 之间波动，机器负载相关，跨轮比值不可比。验收数字见表 8。

<p align="center">表 8　性能与并发验收数字</p>

| 指标 | 数值 | 测量口径 | 证据状态 |
|------|------|----------|----------|
| 单请求墙钟比 | 102.3%（49.94s / 48.83s） | 同轮独占，4L 8 DPU，2026-11-15 | 复用已有验证 |
| 双请求 ×2 窗口墙钟比 | 93.4%（135.10s / 144.64s） | 同上 | 复用已有验证 |
| 4L 三臂 token | `[6722, 4849, 35579, 1197, 5434, 5434]` 逐位一致 | static / dynamic / dynamic_bind 三臂 | 复用已有验证 |
| 调度器剖析 | `__init__` 0.885s → 0.071s/步 | `cProfile`，decode 步，调优前后 | 复用已有验证 |
| 历史波动区间 | 52–70s | 同一代码历史区间，负载相关 | 复用已有验证 |

　　从 124% 到 102.3% 的调优链有两步，均在提交区间内：其一，调度器由就绪驱动收敛为计划序全量提交（提交 7f4bf92，剖析证伪就绪调度）；其二，等待路径由「逐命令完成回调 + 条件变量」改为等待提交线程结束后逐个等待 future（提交 7fcf056），消除每步万级锁操作与线程切换，并让执行异常从 `wait()` 抛出。多请求并发更优的原因是静态路径两请求 prefill 都补 pad 到桶 8、动态按实际长度建输入。探针 `/tmp/probe_4l_dynamic.py` 与 `/tmp/probe_4l_parallel.py` 本次核对仍在位未动；评审轮机器非独占（28 用户、load 15+），按约束未复跑，比值算术经 round02/round03 两轮核对一致（49.94/48.83=1.0227、135.10/144.64=0.9340）。

## 四、关键问题与解决方案

　　阻碍核心目标达成的问题集中在三类：需求点名的运行期回退机制缺位、动态分支绕过静态防线、绑定与收缩的正确性依赖脆弱的隐式约定。以下按对核心目标的阻塞程度排序，逐条说明问题、根因、手段与证据；来源均为三轮评审文档与实施记录，未虚构难点。

### 4.1 运行期回退开关缺位与半开状态（来源：round01 问题1、round02 问题1，高）

　　为了满足判据 1「`PIM_RUNTIME_MODE=static` 随时可回退」，需要给动态编译产物留运行期总开关。实施最初把「可回退」收敛为编译期 `dynamic=False` 默认加 `PIM_DYN_SHAPE` 关绑定回退，运行期总开关从未落地，而实施文档一度把判据 1 记为已完成。经用户 2026-11-16 裁决按方案 a 实施后，红测进一步暴露半开状态：只给 `execute_plan` 加旁路时，去填充输入配包络内核会读到垃圾索引。最终修复把 `runtime_static_mode()`（`runtime/shape_bind.py:12`，调用时读取、不缓存）并入 `dynamic_shape_enabled()`（`:17`）这一单一真源，`execute_plan` 动态分支入口（`runtime/executor.py:387`）复用既有静态全量提交分支，executor 去填充、通信钳制、调度器绑定三处门控统一失效，整体回退包络。由 `test_runtime_mode_static_routes_dynamic_to_static_path`（断言不实例化 `EventScheduler` 且 token 与默认动态一致）与 `test_dynamic_default_keeps_scheduler_without_override` 验证，已实测通过。

### 4.2 动态分支绕过包络上限防线（来源：round02 问题2，高）

　　为了不让越界形状进内核，两种编译模式必须在同一入口共享包络上限防线。静态路径对超长输入在 `prefill_bucket` 抛带处置建议的 `ValueError`，而动态早退分支只复用了「按原长选图」，未继承该校验；探针复现（tiny Llama、T_max=16、prompt 20）动态臂死于无关块表错误或深层 `IndexError`——numpy 镜像尚且越界，真硬件上即 MRAM 越界读写（后者为评审推断，未在真硬件执行）。根因是该校验藏在静态选桶循环的兜底路径，未被提升为两分支共享的入口约束。修复把校验提到模式分支之前（`runtime/executor.py:280`–`284`），超过 `prefill_seq_lens[-1]`（动态即 T_max）抛与静态同文同建议的 `ValueError`，原兜底 raise 成为死代码删除。由 `test_dynamic_prompt_over_envelope_raises`（单元加端到端）与探针复跑两臂同文报错验证，已实测通过。

### 4.3 `payload` 表达式索引错位（来源：round01 问题2，中）

　　为了运行期绑定正确配对，`arg_shape_exprs[i]` 必须对应 `arg_shapes[i]` 的同一实参。两组列表原在两个循环、两套过滤规则下构建：字面量与部分和在 `arg_shapes` 占 `None` 位、`cat` 未持有段被出栈左移，而 `arg_shape_exprs` 在循环外按 Node 实参压缩生成；一旦字面量混入张量实参之间，表达式会配到另一个实参的形状上。评审探针实测当前模型恰好对齐（1248 条 `launch`、64 条 `concat`），属潜伏缺陷而非现行错误。修复把构建移进实参编码循环（`runtime/exec_plan_gen.py:744`–`845`）：字面量/部分和占 `None`、`cat` 未持有段与 `arg_shapes` 同步出栈，两组列表逐位对齐成为构造性不变式。由先失败后通过的红测 `test_arg_shape_exprs_align_with_arg_positions_when_literals_mix_in` 与真实路径契约 `test_arg_shape_exprs_align_with_payload_positions` 验证，已实测通过。

### 4.4 符号绑定曾为空操作的三处缺陷（来源：dynamic-graph-20261113 修正记录，实施发现）

　　为了形状绑定真正生效，符号表达式必须在捕获与转换两个环节都保住 T。首次落地的 `PIM_DYN_SHAPE=1` 实际是空操作，67.7s 的「收益」实为包络执行噪声，靠剖析才查清。根因有三：torch 2.9 把 `Dim("T")` 导出成 `s30` 一类符号名，转换只认字面 `"T"` 导致除输入外全部退化成包络常量；对任一 SymInt 调 `int()` 会让图中其余 SymInt 的表达式一并退化；view/reshape 的字面量形状实参（如 `[1,16,-1,16]`）符号元素被折叠成常量。分别以「从 `range_constraints` 取唯一符号名」「两遍捕获：先读全部表达式再统一取整」「`_fold_symbolic_scalars` 记逐元素表达式并由 `arg_elem_exprs` 携带」修正（`runtime/compile.py:240`、`:127`、`:187`）。由 7 种长度 token 对拍与字面量绑定用例验证，已实测通过。

### 4.5 就绪调度证伪与等待路径调优（来源：dynamic-graph-20261113 末两节，实施记录）

　　为了达到判据 4 的 105% 上限，调度器开销必须压到静态路径的可比水平。事件驱动初版在 4L 上同轮测得 124%（62.22s 对 50.18s），超限。剖析定位两处开销：每步重建计划索引耗 0.885s（占 decode 步 9%）；逐命令完成回调加条件变量唤醒造成每步万级锁操作。改造分两步：就绪驱动提交经剖析证伪（保序约束下无收益且 65% 墙钟零并发），收敛为计划序全量提交；`wait()` 改为等待提交线程结束后逐个等待 future（`runtime/scheduler.py:51`–`64`），与静态路径等待方式同构，附带让执行异常从 `wait()` 抛出。修后单请求 102.3%，`__init__` 降至 0.071s/步；结果复用 2026-11-15 同轮独占记录（表 8）。

### 4.6 通信行收缩两次翻车与 `t_axes` 契约防线（来源：dynamic-graph-20261113 实验结论，实施记录）

　　为了通信行收缩不破坏 token，条目的 T 轴必须显式可知。前两次实验（仅 `gather` 方向、仅 `scatter` 方向）在独立后端的二分对拍中各自破坏 token，根因是 comm 条目不带形状表达式，单 T 轴启发式无法覆盖 `(1,1,T,T)` 这类四维形状，host 闭包与条目实际形状无法从 `CommPlanEntry.shape` 推导。错误接线在 a1417b8 回退干净，保留库层截断能力，并落契约防线：`CommPlanEntry` 新增显式 `t_axes`（`comm/plan.py:49`），`clamped_entry` 缺 `t_axes` 抛 `ValueError`、多 T 轴抛 `NotImplementedError`（`comm/lowering.py:106`–`113`），禁止再按启发式静默裁剪。最终接线（c9aeb0f、8d62ca0）按表 6 边界收缩，由 `test_comm_row_clamp.py`（7 条）与 token 对拍验证，已实测通过。

### 4.7 动态分块组合的测试空洞（来源：round03 问题1，中）

　　为了「模式 × 分段形状」的行为回归可见，分块入口必须进入动态测试矩阵。分块测试基线按静态桶语义建立，早于动态改造；D2 给 `prefill_chunk`/`prefill_bucket` 新增去填充与绑定行为时只把整段入口纳入测试，分块入口漏带，同一根因波及「动态 × 多请求 wave」。功能经评审探针实测正确（静态/动态 × 整段/分块 token 全为 `[58,124,53]`），定性为测试缺口。修复在 `tests/test_chunked_prefill.py` 补「模式 × 分段形状」矩阵（整段锚点、[5,7]、[1,8,3]、8+4 对拍静态基线）与返回契约用例（d2h 保持包络行数、本段 token 占前若干行、KV 按绝对位置推进），在 `tests/test_multi_request_decode.py` 补双泳道并发绑定不同 T 的 wave 用例。测试有效性以变异验证证明：把 `_shape_env` 的 T 临时改为段末绝对位置，3 个分段形态与契约用例立即翻红、整段锚点与既有动态整段用例仍绿，还原后全部通过（复用 round03 修复记录）。

## 五、改动说明

### 5.1 改动总览

　　本次需求分两批落地。首次实施（提交区间 9fc1b0d..7fcf056，共 18 个提交）：26 个文件、增加 2120 行、删除 93 行，其中新增 4 个运行时模块、1 个契约模块与 8 个测试文件。评审修复（round01～round03，工作区未提交）：15 个文件、增加 406 行、删除 80 行，其中代码与测试 13 个文件，另有 2 个 docs 改动（`merge-20261107.md` 与本需求无关，提交时需分开处理）。核心改动集中在 `runtime/` 与 `comm/`。

### 5.2 文件改动清单

　　首次实施区间的全部改动文件见表 9，评审修复的未提交改动见表 10；均与 `git diff --stat` 一致。

<p align="center">表 9　首次实施文件清单（9fc1b0d..7fcf056）</p>

| 文件路径 | 类型 | 改动内容 | 层 |
|----------|------|----------|-----|
| `contracts/runtime_graph.py` | 新增 | ShapeExpr/RuntimeGraph 契约（203 行） | D0 |
| `runtime/dependency.py` | 新增 | 地址依赖判定与折叠（94 行） | D0 |
| `runtime/scheduler.py` | 新增 | EventScheduler（83 行） | D1 |
| `runtime/shape_bind.py` | 新增 | 运行期绑定与开关（134 行） | D2 |
| `runtime/compile.py` | 修改 | 动态导出、表达式捕获、单桶强制 | D2-lite |
| `runtime/exec_plan_gen.py` | 修改 | `payload` 表达式、行钳制、折叠单实现 | D2 |
| `runtime/executor.py` | 修改 | 动态分派、去填充、包络防线 | D2-full |
| `comm/plan.py` | 修改 | `t_axes`/`shape_exprs` 契约 | D2-full |
| `comm/lowering.py` | 修改 | 行收缩原语与拒绝防线 | D2-full |
| `backend/hal_numpy.py` | 修改 | `gate` 同步解析入 `submit` | D1 |
| `graph/opt_passes.py` | 修改 | 融合复制形状表达式 | D2-lite |
| `contracts/unified_ir.py`、`contracts/ir_payloads.py` | 修改 | meta/`payload` 键登记 | D0 |
| `backend/architectures.py` | 修改 | MoE 执行开关 | 共用 |
| `tests/test_runtime_graph_contract.py` 等 8 个新增测试 | 新增 | 契约/调度/绑定/通信/分块用例 | 测试 |
| `tests/test_moe_dispatch.py`、`tests/test_comm_lowering.py`、`tests/test_opt_passes.py` | 修改 | FakeHAL 补字段、收缩等价、融合复制表达式 | 测试 |
| `docs/dynamic-graph-20261113.md` | 新增 | 分层改造过程日志 | 文档 |

<p align="center">表 10　评审修复文件清单（工作区未提交）</p>

| 文件路径 | 类型 | 改动内容 | 轮次 |
|----------|------|----------|------|
| `runtime/exec_plan_gen.py` | 修改 | 表达式同循环对齐、改调 `fold_waits` | round01 |
| `runtime/scheduler.py` | 修改 | 弱引用校验缓存、删 `drain` | round01 |
| `runtime/executor.py` | 修改 | 删 `drain`、回退分派、包络防线前移 | round01/02 |
| `backend/hal_numpy.py` | 修改 | 删 `make_skipped_event` 公共包装 | round01 |
| `runtime/dependency.py` | 修改 | `fold_waits` docstring 标注唯一真源 | round01 |
| `runtime/compile.py` | 修改 | 删兜底与死字段 `dynamic_axes` | round02/03 |
| `runtime/shape_bind.py` | 修改 | 新增 `runtime_static_mode` 并入开关 | round02 |
| `tests/test_exec_plan_gen.py` | 修改 | 新增对齐红测 | round01 |
| `tests/test_runtime_scheduler.py` | 修改 | 新增异常/校验/缓存 4 条并删死代码 | round01 |
| `tests/test_dynamic_shape_compile.py` | 修改 | 新增回退/防线/异常 4 条 | round02 |
| `tests/test_engine_static_compat.py` | 修改 | 同步字段断言 | round01/03 |
| `tests/test_chunked_prefill.py` | 修改 | 动态分块矩阵与契约 5 条 | round03 |
| `tests/test_multi_request_decode.py` | 修改 | 动态 wave 用例 1 条 | round03 |

### 5.3 核心函数清单

　　理解本次改动所需的核心函数见表 11。

<p align="center">表 11　核心函数清单</p>

| 函数/类 | 位置 | 类型 | 作用 | 调用方 |
|---------|------|------|------|--------|
| `EventScheduler` | `runtime/scheduler.py:26` | 新增 | 计划序提交+逐命令绑定 | `execute_plan` |
| `EventScheduler.wait()` | `runtime/scheduler.py:51` | 新增 | 等提交线程后逐个等 future | `run()` |
| `bind_command_shapes()` | `runtime/shape_bind.py:85` | 新增 | 按实际 T 求值三组形状键 | `EventScheduler._loop` |
| `_bindable()` / `_command_bindable()` | `runtime/shape_bind.py:46`/`:64` | 新增 | 前缀布局绑定判据 | `bind_command_shapes` |
| `runtime_static_mode()` / `dynamic_shape_enabled()` | `runtime/shape_bind.py:12`/`:17` | 新增 | 两级运行期开关单一真源 | `executor`/`scheduler`/`exec_plan_gen` |
| `capture_shape_exprs()` | `runtime/compile.py:127` | 新增 | 两遍捕获表达式并取整 | `compile_llama2` |
| `_sympy_to_shape_expr()` | `runtime/compile.py:96` | 新增 | sympy→ShapeExpr 退化规则 | `capture_shape_exprs` |
| `fold_waits()` | `runtime/dependency.py:41` | 新增 | `waits` 折叠唯一真源 | `_PlanBuilder.append`/`address_waits` |
| `address_waits()` / `address_wait_gaps()` | `runtime/dependency.py:56`/`:86` | 新增 | shadow 重算与差集 | 调度器测试 |
| `_clamped_rows()` | `runtime/exec_plan_gen.py:499` | 新增 | 通信行钳制门控 | `_emit_redistribute` |
| `clamped_entry()` | `comm/lowering.py:97` | 修改 | 条目段截断与 `t_axes` 防线 | `scatter`/`gather` 原语 |
| `prefill_bucket()` | `runtime/executor.py:274` | 修改 | 共用包络防线+动态按原长 | `prefill_chunk` |

## 六、验证

### 6.1 验证环境

代码基线：pengcheng @ 7fcf056 加工作区未提交修复（round01～round03，源码最后改动 2026-11-16 15:12，早于最后一轮验证记录）。运行环境：python 3.10.20、torch 2.9.1，numpy 后端多 DPU 模拟；pytorch 环境由 `paths.json` 的 `pytorch_env_script` 加载。性能测量环境：4L 真实 MoE、8 DPU、独占机器（2026-11-15）。

### 6.2 验证命令与结果

　　验证按四层组织：定向动态子集（14 文件）覆盖全部新增行为分支；受影响文件单跑确认评审修复；全量回归守护静态默认路径；端到端对拍与探针建立 token 与性能证据。复用证据的适用性本次逐项核对：源码在 round03 修复后未再变动（最后源码改动早于最后一轮验证记录）、回归日志尾部与记录数字一致、五个探针脚本在位未动。各验证项的结果与证据状态见表 12，未执行项均已注明原因，操作示例不作为实测结果。

<p align="center">表 12　验证命令与结果</p>

| 验证项 | 结果 | 结论 | 证据状态、来源、基线与环境 |
|--------|------|------|----------------------------|
| 定向动态子集 14 文件（命令见下方①） | 300 passed / 0 failed（206s） | 通过 | 复用已有验证；round03 修复后记录（implement §4）；基线同 6.1 |
| `tests/test_chunked_prefill.py`（命令②） | 15 passed / 2 deselected | 通过 | 复用已有验证；同上 |
| `tests/test_multi_request_decode.py`（命令②） | 5 passed | 通过 | 复用已有验证；同上 |
| `test_dynamic_shape_compile.py` + `test_engine_static_compat.py` | 24 passed | 通过 | 复用已有验证；round03 删字段断言后 |
| 全量回归（命令③，裸环境） | 2276 passed / 11 skipped / 0 failed / 52 deselected（1648s） | 通过 | 复用已有验证；日志 `/tmp/regress_round03fix.log` 本次核对尾部一致 |
| 抽查：调度器 + 形状绑定单测（命令④） | 14 passed（2.79s） | 通过 | 本次实际验证；最终树上执行 |
| 4L 三臂 token 逐位一致 + 102.3%/93.4% | tokens 六元组逐位一致；两比值达标 | 通过 | 复用已有验证；2026-11-15 同轮独占基线；评审轮非独占未复跑，比值算术两轮核对一致 |
| 超包络探针（`python /tmp/probe_dyn_overflow.py`） | 两臂同文 `ValueError` | 通过 | 复用已有验证；round02 修复后与 round03 复核记录 |
| 动态分块探针（`python /tmp/probe_dyn_chunked.py`） | 六种组合 token 全为 `[58,124,53]` | 通过 | 复用已有验证；round03 执行记录，探针在位 |
| 变异验证（改 `_shape_env` 的 T 为段末位置） | 3 个分段用例翻红、整段仍绿，还原后全绿 | 通过 | 复用已有验证；round03 修复时执行，还原后 `diff` 确认逐字节一致 |
| 静态路径逐字节兼容（`payload` 键/mode 分派/`gate`/静态分支） | 四处逐 hunk 确认未破坏 | 通过 | 仅核对实现但未执行（逐 hunk 静态核对，复用 round01～03）；行为级由全量回归覆盖 |

　　全量回归中 52 条 deselected 是按口径排除的 `llama2_7b` 慢用例（约 2.5 小时，历轮均不跑），11 条 skipped 恒定不变；历轮同口径通过数为 2260 → 2266 → 2270 → 2276（round01 前至 round03 修复后），增量与各轮新增用例数一致。

复现命令（在仓根目录执行）：

```bash
# 加载 pytorch 环境
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
# ① 定向动态子集（14 文件）
python -m pytest tests/test_runtime_graph_contract.py tests/test_engine_static_compat.py tests/test_runtime_scheduler.py tests/test_dynamic_shape_compile.py tests/test_shape_binding.py tests/test_dynamic_moe_binding.py tests/test_comm_shape_exprs.py tests/test_comm_row_clamp.py tests/test_comm_lowering.py tests/test_moe_dispatch.py tests/test_exec_plan_gen.py tests/test_exec_binding.py tests/test_prefill_buckets.py tests/test_opt_passes.py -q
# ② 受影响文件单跑
python -m pytest tests/test_chunked_prefill.py -q
python -m pytest tests/test_multi_request_decode.py -q
# ③ 全量回归（裸环境，不设 PIM_DYN_SHAPE）
python -m pytest tests/ -q -k "not llama2_7b"
# ④ 抽查（本次实际执行）
python -m pytest tests/test_runtime_scheduler.py tests/test_shape_binding.py -q
```

### 6.3 验收标准对照

　　逐条对照需求文档验收判据的结果见表 13。

<p align="center">表 13　验收标准对照</p>

| 验收标准（来自需求文档） | 验证方法 | 结论 | 证据 |
|--------------------------|----------|------|------|
| 静态模式全量回归绿、`PIM_RUNTIME_MODE=static` 可回退 | 全量回归 + 回退分派测试 | 满足 | 表 12 全量回归行；§3.2/§4.1 |
| 运行期依赖与静态 `waits` 相等、token 一致 | shadow ⊆ 断言 + 多长度对拍 | 满足（语义为 ⊆） | 表 4、表 12；偏差见第七节 |
| 一张 prefill 图覆盖 1..max_seq、形状类有界 | 桶参数 `ValueError` + 缓存键不动 | 满足 | §3.3；`prefill_seq_lens==(T_max,)` |
| 单请求 ≤105%、多请求不劣化 | 同轮独占墙钟对照 | 满足 | 表 8：102.3% / 93.4% |
| 新增行为有同次测试、文档一致 | 三轮评审清单核对 | 满足 | 表 14；文档同步见 §3、§7 |

### 6.4 评审闭环

　　本需求共进行 3 轮代码评审，形成 13 条问题（含跨轮延续 1 条）：高优先级 3 条、中 3 条、低 7 条，全部修复；跨轮延续的运行期回退开关问题以最终状态合并记录。各问题及最终证据见表 14。

<p align="center">表 14　评审问题与修复状态</p>

| 轮次 | 问题 | 严重性 | 修复状态 | 最终证据 |
|------|------|--------|----------|----------|
| round01 | 运行期回退开关未实现且文档记已完成 | 高 | 已修复（round02 按裁决落地） | §4.1；2 条行为测试 |
| round01 | `payload` 表达式索引错位（潜伏） | 中 | 已修复 | §4.3；红测转绿 |
| round01 | 调度器异常传播零测试 | 中 | 已修复 | 2 条异常用例 + 变异验证 |
| round01 | 死代码与投机接口四处 | 低 | 已修复 | `grep` 全仓零残留（round03 复核） |
| round01 | `_VALIDATED` 按 id 误命中且只增不减 | 低 | 已修复 | `test_validation_cache_drops_collected_plans` |
| round01 | 实施文档证据表述与代码不符 | 低 | 已修复 | 纯文档修正 |
| round01 | 等待折叠规则双实现 | 低 | 已修复 | `fold_waits` 单实现 + 既有等价性测试 |
| round02 | 运行期回退开关仍未落地（延续项） | 高 | 已修复并闭环 | §4.1；§6.3 判据 1 行 |
| round02 | 动态分支绕过包络上限防线 | 高 | 已修复 | §4.2；探针两臂同文报错 |
| round02 | 形状捕获防御性兜底掩盖缺陷 | 低 | 已修复 | 转换异常传播测试 |
| round02 | 设计文档引用已删接口 | 低 | 已修复 | 纯文档同步 |
| round03 | 动态 × 分块 prefill 零测试覆盖 | 中 | 已修复 | §4.7；矩阵 5 条 + wave 1 条 |
| round03 | `dynamic_axes` 字段写而不读 | 低 | 已修复 | 字段删除、全仓 `grep` 零残留 |

## 七、与设计的差异

　　最终实现与设计文档（2026-11-16 评审基准版）总体一致；设计文档本身已随实施演进更新（如分层表记录了就绪驱动证伪、不变式 3 同步了私有 `_skipped_event`）。仍存在的差异与原因见表 15，均为已确认或已记录的调整。

<p align="center">表 15　与设计的差异</p>

| 设计条目 | 设计方案 | 最终实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| 判据 2 措辞 | 运行期依赖与静态 `waits`「逐命令相等」 | 断言为 ⊆（host 闭包以 `waits` 为准） | 地址分析覆盖不了闭包数据依赖 | 是（round01 澄清） |
| 判据 3 措辞 | `prefill_seq_lens == ()` | 字段为 `(T_max,)`，桶参数为空 | 关绑定回退包络依赖该字段补 pad | 是（round01 澄清） |
| D1 调度模型 | 初版就绪驱动提交 | 计划序全量提交 | 剖析证伪（65% 墙钟零并发） | 是（设计分层表已同步） |
| 回退开关实现 | 裁决方案 a：入口旁路 | 开关并入 `dynamic_shape_enabled` | 只切提交路径会形成半开状态 | 是（round02 取舍记录） |
| 校验缓存结构 | 评审建议 `WeakSet` | `WeakValueDictionary` 按 id 键 | dataclass 自动生成 `__eq__` 不可哈希 | 是（round01 §6.2） |
| D3 契约预留 | 分层表「契约预留」 | 仅保留 contracts 结构，执行器字段已删 | 预留按需求只覆盖契约文件 | 是（round01-4 裁决） |

## 八、当前遗留问题

　　本节汇总尚未关闭的事项、影响与来源；前四条为需求边界内的显式未做项，后两条为证据口径限制，均非未修复缺陷。逐条状态见表 16。

<p align="center">表 16　当前遗留问题</p>

| 编号 | 问题描述 | 影响范围 | 来源 | 状态 |
|------|----------|----------|------|------|
| L1 | D3 动态控制流、D4 slot arena 未实施 | 动态控制流与任意 offset 分配不可用 | request 边界、implement §5 | 契约预留，多 batch 调度立项后实施 |
| L2 | d2h 全链收缩未做 | 结果回主机仍搬包络行 | implement §5、表 6 | 被专家路径阻塞：路由缓冲行数非 T 前缀，需真正动态专家分派，未立项 |
| L3 | 多 T 轴布局重映射未做 | `(1,1,T,T)` mask 不能收缩 | implement §5 | 收益仅 numpy 掩码构造，非缺陷 |
| L4 | `kernel_classes` 形状类计量字段已删 | 判据 3 的计量口径无运行时数字 | round01-4 | D4b 拆地址键前无消费者，需要时从 git 历史加回 |
| L5 | 4L token 与墙钟未在评审轮独立复核 | 性能结论依赖 2026-11-15 基线记录 | review round01～03 | 复用证据已核对适用；机器非独占不可比 |
| L6 | 跨轮墙钟绝对值波动 52–70s | 性能结论只取同轮独占对照 | dynamic-graph 末节 | 机器负载相关外部约束，无法在本环境消除 |
