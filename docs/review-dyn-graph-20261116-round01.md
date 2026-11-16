# 代码评审报告：静态计算图 → host 驱动动态图（dyn-graph）
> 文档编号：review-dyn-graph-20261116-round01
> 创建日期：2026-11-16
> 关联需求文档：request-dyn-graph-20261116
> 关联设计文档：design-dyn-graph-20261116
> 关联实施文档：implement-dyn-graph-20261116

## 一、评审概述

　　本轮对 9fc1b0d..7fcf056 的 18 个提交（工作区仅 docs 改动，无代码未提交项）按需求、设计、实施三份文档做了全面审计，覆盖静态兼容红线、调度器语义、形状绑定判据、通信行收缩边界、开关门控一致性与仓库纪律。整体结论：分层架构与设计六条不变式基本忠实落地，静态路径逐字节兼容红线守住（payload 新键只在动态导出写入、执行严格按 mode 分派、gate 逻辑未动），两次通信实验（23049f7）的回退干净、库层与 t_axes 防线自洽，验收数字（102.3% / 93.4% / 4L tokens）在同轮独占口径下记录一致且算术核对无误。本轮发现严重 0、高 1、中 2、低 4：高问题是需求点名的运行期回退开关 `PIM_RUNTIME_MODE=static` 全仓未实现而实施文档将判据 1 记为已完成；其余为 payload 表达式索引对齐缺乏结构性保证（潜伏）、调度器异常传播语义无测试，以及死代码残留与文档表述偏差类问题。八条核心目标：3、5、6 已完成（5 为复用证据），1、4、7、8 部分完成，详见第三节；高问题未清零。

## 二、修改建议

### 问题1（高）: 运行期回退开关未实现，导致验收判据1失守
**问题说明**：需求验收判据 1 要求「`PIM_RUNTIME_MODE=static` 随时可回退」，但全仓代码与文档（除需求文档本身）均无此变量：`grep -rn PIM_RUNTIME_MODE` 仅命中 request-dyn-graph-20261116.md:27。实际机制是编译期 `compile_llama2(dynamic=False)` 默认静态 + 运行期 `PIM_DYN_SHAPE` 只控制形状绑定/收缩（executor.py:251、280；exec_plan_gen.py:505）；动态编译产物在运行期永远走 `EventScheduler`（executor.py:388-398），无法整体切回静态全量提交路径。而实施文档第 1 节将判据 1 整条记为「已完成」，证据只覆盖「全量回归绿」这半句。
**根本原因**：实施把「可随时回退」收敛为「编译期开关 + 关绑定包络回退」双层机制，没有落需求点名的运行期总开关；收口文档沿用需求原始措辞记成已完成，未把机制偏差记录为偏差。同一根因影响所有以 `CompiledModel.compile_mode` 为准的执行入口（`step_request`、`prefill_chunk`、`execute_plan` 的 dynamic 分派），它们都没有给运行期留统一切换点。
**涉及文件**：runtime/executor.py:388-398（mode 分派无环境变量旁路）、runtime/compile.py:680（compile_mode 写死）；docs/implement-dyn-graph-20261116.md 第 1 节判据 1 行。
**修改思路**：二选一并闭环，属需求层面取舍、需回用户确认，不在评审里替做决定。其一：在 `execute_plan` 的 dynamic 分支入口读 `PIM_RUNTIME_MODE=static` 时改走静态全量提交路径，并补「动态编译 + 该变量 = 静态路径执行且 token 一致」的同次测试；其二：确认由「dynamic=False 默认 + PIM_DYN_SHAPE 关闭整体包络回退」替代该机制，把偏差写进实施文档并把判据 1 状态改为部分完成。无论哪种，实施文档不得把未实现机制记为已完成。

### 问题2（中）: payload表达式与形状索引错位，导致绑定配对错实参
**问题说明**：`arg_shape_exprs` 与 `arg_shapes` 是运行期绑定的配对输入（shape_bind.py:65-70 用 zip 做判据、113-122 按同一下标绑定），但两组列表按不同规则构建：`arg_shape_exprs` 在循环外按「flat_args 中的 Node 实参」压缩生成（exec_plan_gen.py:838-841），`arg_shapes` 在循环内按「payload 实参顺序」生成——字面量与部分和占 `None` 位（:764、:775），`cat` 未持有段被 pop（:789-790）后整体左移。一旦字面量出现在张量实参之前/之间，或 cat 段被 pop，`exprs[i]` 会配到第 i 个位置的**另一个实参**的形状上；`_bind_dims` 只有维数相同的守卫，同维数不同布局时会把错误表达式的求值结果写进错误实参的形状。本次用 /tmp 探针在 tiny Llama 动态编译上实测：1248 条携带表达式的 launch 当前全部对齐、64 条 concat 命令携带表达式，故为潜伏缺陷而非现行错误；但该对齐是「张量恰好都排在字面量前、cat 段恰好全持有」的巧合，没有任何代码或契约强制它。
**根本原因**：payload 两组平行列表分别在两个循环、两套过滤规则下构建，没有构造性对齐不变式。作者在 `arg_elem_exprs` 上已知道要在同一循环里对齐（pop 同步、逐位 append，:766、:776、:790），`arg_shape_exprs` 漏掉了同样的处理——这正是 a411403 修复的「字面量形状」缺陷的同族第四处（实参级对齐），且随后续引入新算子/新切分形态会持续暴露。
**涉及文件**：runtime/exec_plan_gen.py:838-841（构建）、runtime/shape_bind.py:65-70 与 113-122（按位消费）。
**修改思路**：把 `arg_shape_exprs` 的构建移进实参编码循环：张量实参 append 其表达式（无则 `None` 占位），字面量/部分和 append `None`，cat 未持有段与 `arg_shapes`/`arg_kinds` 同步 pop，使「两组列表逐位对齐」成为构造性不变式；`shape_bind` 已支持 `None` 条目，消费端无需改动。可加一条契约测试：凡携带 `arg_shape_exprs` 的命令，其长度等于 `arg_shapes` 长度且 None 位与 payload 实参位一致。

### 问题3（中）: 调度器异常传播无测试，导致不变式3缺覆盖
**问题说明**：设计不变式 3 明确「提交异常与命令执行异常都从 `wait()` 抛出」，过程日志也把「执行异常现在会从 `wait()` 抛出（旧路径只传播提交异常，执行异常静默）」记为 7fcf056 的附带修复。核对实现（scheduler.py:64-74：`_loop` 捕获提交异常存 `self.error`，`wait()` join 后先抛 `error`、再逐 `future.result()` 透传执行异常）语义正确，但全测试目录没有任何用例触发这两条异常路径——`tests/test_runtime_scheduler.py` 只有 3 个用例，`_FakeHAL` 无法注入失败。按「每个新增行为分支有同次测试」的要求，该行为分支为零覆盖；若未来有人把 `wait()` 改回静默（正是旧缺陷），回归不会变红。
**根本原因**：7fcf056 是性能调优提交，异常传播作为「附带修复」随性能改造一起上车，没有走先写失败测试的流程；`_FakeHAL` 只设计了记录提交顺序的能力，缺一个可控失败的 HAL 变体。同一根因也使 `run(drain=False)+wait()` 组合只在提交顺序用例里被顺带覆盖，异常时序（drain=False 下提交异常何时可见）完全无断言。
**涉及文件**：runtime/scheduler.py:49-74；tests/test_runtime_scheduler.py（缺用例）。
**修改思路**：在 `tests/test_runtime_scheduler.py` 补两个用例并复用可控失败的 FakeHAL：其一，`submit` 抛异常时 `wait()`（含 `run(drain=False)` 后再 `wait()`）抛同一异常；其二，命令 future `set_exception` 时 `wait()` 抛执行异常。断言异常类型与消息，保证静默化会立刻翻红。

### 问题4（低）: 死代码与投机接口未清理，导致违反仓库纪律
**问题说明**：CLAUDE.md 要求「删优于加」「不写以后可能用到的分支、参数」。范围内残留四处：(a) `NumpyBackend.make_skipped_event`（hal_numpy.py:180-188）自 7f4bf92 把 gate 判定移入 `submit` 后无任何调用方（submit 内部用私有 `_skipped_event`，hal_numpy.py:268）；(b) `tests/test_runtime_scheduler.py:60-64` 的 `_FakeHAL.make_skipped_event` 与 :139 的 `assert hal.skipped == []` 恒真（该方法无人调用，断言对象不存在）；(c) `CompiledModel.runtime_graph` 与 `kernel_classes` 字段（compile.py:65、67）全仓无写入方，`kernel_classes` 连设计文档都未提及用途，唯一「使用」是 test_engine_static_compat.py:34-36 断言其默认值的形式测试；(d) `execute_plan(drain=...)` 与 `EventScheduler.run(drain=False)`（executor.py:372、scheduler.py:40）无生产调用方传 `False`，是为测试提交顺序留的缝。request 允许的「契约预留」覆盖 contracts/runtime_graph.py 的 D3/D4 结构，不覆盖这些散落在执行器与后端的接口面。
**根本原因**：两次收敛（7f4bf92 调度器改全量提交、7fcf056 删回调等待）只删了各自直接替代的实现，没有回头清接口面；D0 加「动态缝」字段时按预留写入了本阶段无消费者的字段。测试侧则把「断言默认值」当成了字段存在的验证。
**涉及文件**：backend/hal_numpy.py:180-188；runtime/compile.py:63-67；runtime/executor.py:372、398；runtime/scheduler.py:40；tests/test_runtime_scheduler.py:60-64、139；tests/test_engine_static_compat.py:32-36。
**修改思路**：删除 `make_skipped_event` 公共包装与 FakeHAL 对应方法、`skipped` 断言（断言对象已不存在，应随语义一起删）；删除 `kernel_classes` 与 `drain` 参数（测试提交顺序可改为在 FakeHAL.submit 完成后立即轮询，或保留线程直测 `EventScheduler._loop`）；`runtime_graph` 字段若确为 D3 预留，在设计文档分层表标注归属，否则一并删。所有删除同步改对应测试。

### 问题5（低）: 计划校验缓存按id会误命中，导致校验可能被跳过
**问题说明**：`_validate_plan` 用模块级 `_VALIDATED: set[int]` 按 `id(plan)` 跳过重复校验（scheduler.py:10-20），注释依据是「计划常驻 CompiledModel，id 不会复用」。但 `execute_plan` 是公共入口，短命计划（测试、脚本、未来按请求建图的 D3 路径）会被垃圾回收，CPython 会把同一段内存（同一 id）分配给新计划，新计划因此被跳过 waits 引用校验；同时集合只增不减。校验是 fail-fast 防线，被跳过不会算错结果，但会把「依赖了不存在的命令」的暴露推迟到执行期甚至掩盖。
**根本原因**：用值语义的 `id()`（int）做对象身份缓存，把对象生命周期假设写进了跨调用共享的模块状态；该假设只在当前静态计划常驻的生产路径成立，与公共 API 的使用面不符。
**涉及文件**：runtime/scheduler.py:10-20。
**修改思路**：改为 `weakref.WeakSet` 按（弱）对象身份记录：计划被回收后条目自动消失，新对象即使复用同一地址也不会误命中，同时消除只增不减；语义与注释（计划身份只校验一次）保持不变。

### 问题6（低）: 实施文档证据表述与代码不符，导致文档可信度受损
**问题说明**：实施文档声称「与代码、测试、git 历史一致」，但两处证据表述与实物不符：(a) 第 1 节判据 3 证据「dynamic 编译时 `prefill_seq_lens=()`」——`CompiledModel.prefill_seq_lens` 实际是 `(T_max,)`（compile.py:675，`seq_lens = (prefill_seq_len,)`），`()` 的是 `compile_llama2` 的额外桶**参数**（由 :433-436 的 ValueError 强制为空），test_dynamic_shape_compile.py:99 断言的也是 `(MAX_SEQ,)`；该字段为 `(T_max,)` 恰恰是关绑定回退包络所依赖的（prefill_bucket 据此补 pad）。(b) 第 1 节判据 2 行写「依赖 shadow **相等**」，而测试断言的是「运行期地址依赖 ⊆ 静态 waits」（test_runtime_scheduler.py:76-83 的 `address_wait_gaps == []`，host 闭包依赖不在地址分析内），设计不变式与过程日志均按 ⊆ 表述。
**根本原因**：文档收口时沿用需求原始措辞（`prefill_seq_lens == ()`、「逐命令相等」）复述证据，没有跟随设计对语义的收敛（单图+桶参数为空、⊆）改写；实施文档与需求措辞混用，使证据列不可按字面核对。
**涉及文件**：docs/implement-dyn-graph-20261116.md 第 1 节判据 2、3 行；对照 docs/design-dyn-graph-20261116.md 不变式 2/4、tests/test_dynamic_shape_compile.py:95-99。
**修改思路**：修正两处表述为可实现语义并注明与需求措辞的对应关系：「额外桶参数必须为空、模型字段为 `(T_max,)` 单桶」；「shadow 断言 ⊆（host 闭包依赖以 waits 为准），非严格相等」。后续轮次实施文档的证据列一律写可核对的精确事实而非需求原文复述。

### 问题7（低）: 等待折叠规则双实现，导致规则漂移风险
**问题说明**：同一「同 DPU 只留最后一条前驱、host 命令逐条保留」的折叠规则存在两份实现：`runtime/dependency.fold_waits`（dependency.py:41-49）与 `_PlanBuilder.append` 的内联折叠（exec_plan_gen.py:447-457）。D1 抽取依赖判定到 `runtime/dependency.py` 时，`deps_of/war_waits/overlap` 抽走了，折叠逻辑留下了旧副本又在新模块重写一份。shadow 校验（`address_waits` 用 `fold_waits` 折叠运行期依赖，再与 `append` 折叠出的静态 waits 求差）的等价性依赖两份实现逐字一致；历史上两次性能重构翻车的根因正是这条 FIFO/折叠假设被破坏，双实现使「改一处漏一处」的漂移成为可能，仅靠附录 A 单计划用例间接兜底。
**根本原因**：抽取不彻底——新模块为 shadow 自建了折叠副本，而唯一真源的调用方（builder）没有切换到共享实现，形成按约定的重复而非按引用的复用。
**涉及文件**：runtime/dependency.py:41-49；runtime/exec_plan_gen.py:447-457。
**修改思路**：`_PlanBuilder.append` 改调 `dependency.fold_waits`，删除内联副本，折叠规则收敛为单实现；顺带在 `fold_waits` docstring 指明它同时服务静态计划生成与 shadow 校验，防止后续再被「优化」出分叉。

## 三、核心目标完成情况

| 需求目标 | 状态 | 证据 | 缺口或外部阻塞 |
| --- | --- | --- | --- |
| 1. 静态默认、逐字节兼容、可随时回退 | 部分完成 | 静态全量回归 2308 passed / 11 skipped / 0 failed（复用，基线 7fcf056=HEAD）；payload 新键仅动态导出写入（test_static_plan_has_no_shape_expr_payload，随本次定向子集绿）；static 模式 execute_plan 不走 EventScheduler（executor.py:388-398 逐 hunk 核对+契约测试） | `PIM_RUNTIME_MODE=static` 未实现且文档记为已完成（问题1） |
| 2. 运行期依赖 ⊆ 静态 waits（折叠后）逐命令，shadow 在位且真实断言 | 已完成 | test_address_waits_are_a_subset_of_static_waits：真实断言 gaps==[] 且运行期依赖非空；本次定向子集实跑通过 | 覆盖仅附录 A 单计划；「相等」措辞与实现 ⊆ 的差异见问题6 |
| 3. token 逐 token 一致 | 已完成（小模型本次实跑；4L 复用） | tiny Llama 7 种长度、tiny MoE 5 种长度逐 token 一致（用例自带 PIM_DYN_SHAPE，本次 290 passed 内）；4L tokens `[6722, 4849, 35579, 1197, 5434, 5434]` 复用实施文档记录 | 4L 本轮机器非独占（30 用户会话）未复跑，数字未独立复核 |
| 4. 一张 prefill 图覆盖 1..max_seq、形状类有界 | 已完成（判据字面有偏差） | 单图：compile_llama2 强制额外桶参数为空（compile.py:433-436）、test_dynamic_compile_records_mode_and_single_bucket 绿；形状类有界为结构性保证：绑定不改 mram_offset/缓存键（shape_bind.py:80-129 只改 arg_kinds/arg_shapes/out_shape） | `prefill_seq_lens == ()` 字面不成立（字段为 `(T_max,)`，问题6）；kernel_classes 计量字段未实现（问题4） |
| 5. 单请求 ≤105%、多请求不劣化 | 已完成（复用） | 102.3%（49.94/48.83s）、93.4%（135.10/144.64s），同轮独占口径记录于 dynamic-graph-20261113.md 末两节，比值算术核对一致 | 本轮非独占未复测；跨轮绝对值波动为已确认外部约束 |
| 6. 双轨制不变式（不改 MRAM offset/缓存键/KV 几何/容量） | 已完成 | bind_command_shapes 仅改三组形状键、`replace` 不改原命令（shape_bind.py 全文核对+test_bind_rewrites_shapes_for_symbolic_dims 断言原命令不动）；规划只用包络整数（capture_shape_exprs 先存表达式再取整，compile.py:127-160） | 无 |
| 7. 每个新增行为分支有同次测试 | 部分完成 | 关绑定回退包络（test_dynamic_without_binding_falls_back_envelope）、gate 否定分支（test_moe_dispatch 既有用例，本次实跑）、token 用例自带 PIM_DYN_SHAPE、行收缩门控否定分支（test_clamped_rows_requires_dyn_env_and_prefix_layout 四否定态）均在 | wait() 异常传播零测试（问题3）；scheduler skipped 断言恒真（问题4） |
| 8. 仓库纪律（最小实现、无死代码、契约同步） | 部分完成 | 注释简短中文、回退干净（23049f7→a1417b8 核对：接线全撤、库层保留并修正）、无调试残留（grep ready/probe/TODO 空）；meta 键登记 unified_ir.py:79-91、payload 键写入/消费成对、opt_passes 融合随 val 复制表达式（test_fuse_copies_shape_exprs）；genesim/scripts 不引用受改契约（grep 核对） | 死代码/投机接口四处（问题4）、_VALIDATED 缓存缺陷（问题5）、折叠双实现（问题7） |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
| --- | --- | --- | --- |
| 定向动态子集 14 文件（含契约/调度/绑定/通信/MoE/opt_passes） | 本次实际验证 | 290 passed / 0 failed（176s） | 本轮在 HEAD=7fcf056、工作区仅 docs 改动下执行；pytorch env（paths.json）；裸环境外仅用例自带的 PIM_DYN_SHAPE |
| 静态全量回归 `pytest tests/ -q -k "not llama2_7b"`（裸环境） | 复用已有验证 | 2308 passed / 11 skipped / 0 failed（7698s） | 来源 implement-dyn-graph §4 与 dynamic-graph-20261113.md 末节（2026-11-15，基线 7fcf056）；与 HEAD 同基线且工作区无代码改动、同为裸环境默认排除 slow，适用性核对一致，未重跑 |
| 4L 真实 MoE 对拍（tokens 六元组、102.3%、93.4%） | 复用已有验证 | 三臂 token 逐位一致；单请求 102.3%、双请求 93.4% 达标 | 来源 implement §4 与 dynamic-graph 末两节（探针 /tmp/probe_4l_dynamic.py、/tmp/probe_4l_parallel.py 存在）；本轮 30 用户会话、load≈10/112 非独占，按约束未复跑，墙钟未独立复核 |
| payload 对齐探针 /tmp/probe_align_dyn.py | 本次实际验证 | 携带 arg_shape_exprs 的 launch 1248 条全部索引对齐、concat 64 条带表达式；问题2 为潜伏缺陷的直接证据 | 本轮执行，tiny Llama（4 DPU、T_max=16）动态编译，基线同上 |
| `PIM_RUNTIME_MODE` 存在性 | 本次实际验证 | 全仓代码与 docs（除需求文档本身）均无该变量 | 本轮 `grep -rn` 全仓 |
| 静态路径逐字节兼容（payload 新键、mode 分派、gate 行为） | 仅核对实现但未执行（逐 hunk 核对；对应契约测试已随定向子集实跑） | 三处均确认：键仅动态写入、分派严格按 mode、gate 逻辑本区间未改动 | 9fc1b0d..7fcf056 逐 hunk 对照 + test_static_plan_has_no_shape_expr_payload（本次绿） |
| scheduler wait() 异常传播 | 仅核对实现但未执行 | 代码语义正确（error 先抛、future.result 透传），无测试覆盖（问题3） | 本轮静态核对 |

　　未覆盖范围：`llama2_7b` 慢用例与任何 CEVA/GeneSim 全流程按指令未跑，与本轮改动无关；4L 墙钟与 token 未独立复测，相关结论以文档记录为据并标注复用；性能数字的「同轮独占」口径无法在本轮事后复核，仅核对了记录一致性与算术。上述限制不影响问题 1-7 的判定：它们均由代码与文档静态证据或本轮实跑/探针直接支撑。
