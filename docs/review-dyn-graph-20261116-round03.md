# 代码评审报告：静态计算图 → host 驱动动态图（dyn-graph）
> 文档编号：review-dyn-graph-20261116-round03
> 创建日期：2026-11-16
> 关联需求文档：request-dyn-graph-20261116
> 关联设计文档：design-dyn-graph-20261116
> 关联实施文档：implement-dyn-graph-20261116

## 一、评审概述

　　本轮为多 batch 调度启动前的第 3 轮收口审计，范围是已提交区间 9fc1b0d..7fcf056（动态图全部 18 个提交）叠加工作区未提交修复（docs 除外，即 round01/round02 的全部代码修复）。评审先逐条复核前两轮共 11 个问题的修复是否落在根因、同类路径是否真正覆盖，再按设计六条不变式与重点检查清单全面审计静态兼容红线、调度器语义、形状绑定判据、通信行收缩边界、开关门控一致性、契约同步、测试质量与仓库纪律。复核结论：round01 七项与 round02 四项修复全部落实在根因位，未发现「只在探针或单例上生效」的复发；Q1 运行期回退开关已按用户裁决方案 a 落地且实现质量合格（调用时读环境变量、复用既有静态分支零改动、测试真断言分派路径与 token）。审计结论：静态默认路径逐字节兼容红线守住（`execute_plan` 静态分支在区间 diff 中未被修改、payload 新键仅动态写入、gate 逻辑本区间未动），两次通信实验回退干净，实施文档的提交链、文件清单与通信收缩边界表同 git 历史及代码一致，全量回归本轮在最终树上复跑 2270 passed / 0 failed。本轮新发现问题：严重 0、高 0、中 1、低 1——中问题为动态编译 × 分块 prefill 的行为组合零测试覆盖（功能经本轮探针实测正确，属测试缺口而非功能缺陷），低问题为 `dynamic_axes` 字段写而不读（round01 问题 4 的同类残留）；高、严重问题已清零。八条核心目标：1/2/3/4/5/6 已完成，7/8 部分完成，逐条证据见第三节。另注：工作区 docs 改动中含与本需求无关的 `merge-20261107.md`（CEVA 硬件仿真记录）与 `dynamic-graph-20261113.md` 的两行验收数字补记（与实施文档一致），不在本轮评审范围，提交时注意分开处理。

## 二、修改建议

### 问题1（中）: 动态分块prefill零测试覆盖，导致行为回归不可见
**问题说明**：需求判据 5 与核心目标 7 要求每个新增行为分支有同次测试；round02 问题 2 的评审已明确把「分块 prefill」列为 `prefill_bucket` 去填充行为的调用链之一（step_request 整段 prefill、分块 prefill、多请求 wave）。实际「动态绑定 × 分块」组合无任何自动化用例：`tests/test_chunked_prefill.py` 的 10 条用例全部静态编译（本轮单独实跑 10 passed 确认现状）；全仓设置 `PIM_DYN_SHAPE` 的测试文件仅 3 个（test_comm_row_clamp、test_dynamic_moe_binding、test_dynamic_shape_compile），均无分块场景。本轮新探针 `/tmp/probe_dyn_chunked.py` 实测功能正确（tiny Llama 4 DPU、prompt 12、分段 [5,7] 与 [1,8,3]：动态分块、动态整段、静态分块与静态整段 token 全为 [58,124,53] 逐位一致），故定性为测试缺口而非功能缺陷；但「每段绑定不同 T、pos 递增写 KV、返回 logits 按段内行偏移」是动态改造中行为组合最复杂的路径之一，一旦分段逻辑或 logits 行偏移语义被改动，现有回归全部无法拦截。
**根本原因**：分块 prefill 的测试基线按静态桶语义建立（早于动态改造），D2 给 `prefill_chunk`/`prefill_bucket` 新增去填充与绑定行为时只把整段入口纳入同次测试矩阵（round02-2 超包络防线的端到端测试也只取整段），分块入口没有随行为分支一起进入矩阵。同一根因还波及「动态 × 多请求 wave」组合——该组合有 4L 双请求探针的真实模型级证据（复用），分块则文档与探针均无任何记录，是矩阵中最完整的空洞。
**涉及文件**：tests/test_chunked_prefill.py（缺动态参数化用例）；对照 runtime/executor.py:244（prefill_chunk 调用 prefill_bucket）、runtime/executor.py:249-252（mask_len 去填充分支）。
**修改思路**：在 test_chunked_prefill.py 既有「分块 == 整段」参数化上叠加动态维：动态编译、用例自带 `PIM_DYN_SHAPE`，至少覆盖首段 1 token 与尾段非整桶两种分段形态，与静态基线对拍 token。按「模式 × 分段形状」矩阵组织用例，后续 D3 等新模式分支在同一矩阵扩展，一次补齐同一入口的全部模式组合，而不是逐轮按事故补单点。

### 问题2（低）: dynamic_axes字段写而不读，违反最小实现
**问题说明**：CLAUDE.md 要求「最小实现：不写以后可能用到的分支、参数、配置项」「删优于加」；round01 问题 4 已按此口径删除 `CompiledModel` 上无消费者的 `runtime_graph`/`kernel_classes` 字段。`dynamic_axes`（runtime/compile.py:65 定义、:676 动态编译写入 `{"T": (1, prefill_seq_len)}`）全仓无生产读取方——本轮 grep 覆盖本仓全部源码与 scripts/、genesim/scripts/，唯一「消费」是两条测试的值断言（tests/test_dynamic_shape_compile.py:98、tests/test_engine_static_compat.py:34 的 default_factory 断言），属写而不读的半个死代码；其信息与 `prefill_seq_lens[-1]` 冗余。
**根本原因**：round01 问题 4 的修复按「无写入方」标准筛选删除对象（实施文档 §3 记录的口径即「全仓无写入方」），漏掉了「有写入、无读取」的字段，`dynamic_axes` 因此遗留为同类残留。同类排查：`CompiledModel` 其余字段（moe_execution、prefill_seq_lens、kv_specs 等）均有运行时消费者，无第二个同类。
**涉及文件**：runtime/compile.py:65、runtime/compile.py:676；tests/test_dynamic_shape_compile.py:98；tests/test_engine_static_compat.py:34。
**修改思路**：按 round01 问题 4 先例二选一并回设计确认：删除该字段与对应断言（多 batch 调度立项时随首个真实消费者一起加回，git 历史可恢复）；或确认作为动态产物自描述保留，则在设计文档分层表显式登记其归属与首个预期消费者，避免与已删除字段口径不一致。

## 三、核心目标完成情况

| 核心目标 | 状态 | 证据 | 缺口或外部阻塞 |
|---------|------|------|----------------|
| 1. 静态路径默认、逐字节兼容、可随时回退 | 已完成 | 全量回归本轮最终树复跑 2270 passed / 11 skipped / 0 failed（本次实际验证，裸环境不设 PIM_DYN_SHAPE）；`execute_plan` 静态分支在区间 diff 中逐 hunk 未被修改，`mode` 仅新增参数与动态分支；static 模式不经 EventScheduler（唯一实例化点 runtime/executor.py:397 在动态分支内，test_runtime_mode_static_routes_dynamic_to_static_path 断言不实例化且 token 一致）；payload 新键仅动态导出写入（test_static_plan_has_no_shape_expr_payload）；Q1 已按裁决方案 a 落地：`PIM_RUNTIME_MODE` 调用时读取（shape_bind.py:12-14）、复用静态分支零改动、两测试真断言行为 | 无 |
| 2. 运行期依赖 ⊆ 静态 waits（折叠后），shadow 在位且真实断言 | 已完成 | test_address_waits_are_a_subset_of_static_waits 断言 gaps==[] 且运行期依赖非空（本轮定向子集内绿）；折叠规则收敛 dependency.fold_waits 单实现（本轮 grep 全仓确认 exec_plan_gen 内联副本已删、无第三份副本）；「逐命令相等」按设计收敛为 ⊆（host 闭包依赖以计划 waits 为准），round01 问题 6 已澄清 | 覆盖为附录 A 单计划口径 |
| 3. token 逐 token 一致 | 已完成 | tiny Llama 7 种长度、tiny MoE 5 种长度逐 token 一致（用例自带 PIM_DYN_SHAPE，本轮定向子集 300 passed 内实跑）；4L 真实 MoE tokens `[6722, 4849, 35579, 1197, 5434, 5434]` 复用 2026-11-15 基线（本轮 28 用户、load 21+ 非独占按约束未复跑） | 4L 数字未独立复核（非独占） |
| 4. 一张 prefill 图覆盖 1..max_seq，形状类有界 | 已完成 | 单图由「额外桶参数必须为空」强制（compile.py:428-431 ValueError）、`prefill_seq_lens==(T_max,)` 单桶（test_dynamic_compile_records_mode_and_single_bucket）；超包络防线两模式共用且先于模式分支（prefill_bucket 入口，本轮探针两臂同文 ValueError）；形状类有界为结构性保证：绑定只改 arg_kinds/arg_shapes/out_shape，mram_offset 与内核查找键不动（shape_bind.py 全文 + hal_numpy.lookup 按内核名查找） | 无 |
| 5. 单请求 ≤105%、多请求不劣化 | 已完成（复用） | 102.3%（49.94/48.83s）、93.4%（135.10/144.64s）同轮独占记录（dynamic-graph-20261113.md 末两节，含本轮工作区补记的两行）；比值算术本轮再核对一致（1.0227/0.9340） | 本轮非独占未复测；跨轮绝对值波动为已确认外部约束 |
| 6. 双轨制不变式（不改 MRAM offset/缓存键/KV 几何/容量） | 已完成 | bind_command_shapes 仅改三组形状键、replace 不改原命令（test_bind_rewrites_shapes_for_symbolic_dims 断言原命令不动）；规划只见包络整数（capture_shape_exprs 两遍捕获，先读全部 sympy 表达式再取整）；通信钳制只截 global_range 尾与 nbytes 不改段地址；区间 26 个改动文件不含 memory/ 与 contracts/exec_plan.py，规划结构未动 | 无 |
| 7. 每个新增行为分支有同次测试 | 部分完成 | 已覆盖：关绑定回退包络（test_dynamic_without_binding_falls_back_envelope）、门控否定分支（test_moe_dispatch、test_clamped_rows_requires_dyn_env_and_prefix_layout 四否定态）、运行期回退分派（2 条）、超包络防线（单元+端到端）、异常传播（2 条，断言异常实例同一）、payload 对齐契约（2 条）、弱引用缓存释放 | 动态编译 × 分块 prefill 组合零覆盖（问题1）；「动态 × 多请求 wave」仅有 4L 探针复用证据、无 tiny 级 pytest 用例 |
| 8. 仓库纪律（最小实现、无死代码、无防御兜底、契约同步） | 部分完成 | 无调试残留与旧实现尸体（grep ready/probe/TODO 空；新增模块仅 scheduler._loop 一处 except BaseException 作错误通道）；capture_shape_exprs 兜底已删（compile.py:158 直接调用）；meta 键登记 unified_ir META_KEYS、payload 键写入/消费成对、opt_passes 融合随 val 复制表达式（test_fuse_copies_shape_exprs）；genesim/scripts 不引用受改契约（本轮 grep）；注释简短中文；drain/make_skipped_event/runtime_graph/kernel_classes 无调用方残留 | dynamic_axes 写而不读（问题2） |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
|-------------|----------|------|----------------------|
| 定向动态子集 14 文件（每轮必跑） | 本次实际验证 | 300 passed / 0 failed（182s） | 本轮在 HEAD=7fcf056 + 工作区 round01/round02 修复树执行（代码最后改动 13:40，与实施文档基线一致）；pytorch env（paths.json）；裸环境外仅用例自带 PIM_DYN_SHAPE |
| 全量回归 `pytest tests/ -q -k "not llama2_7b"`（裸环境） | 本次实际验证 | 2270 passed / 11 skipped / 0 failed / 52 deselected（1436s） | 本轮在最终树后台复跑，日志 /tmp/regress_round03.log；与实施文档 round02 记录 2270/1446s 同数（代码未变，适用性核对一致）；2 条 RuntimeWarning 为 test_architectures 既有 fp16 告警 |
| 动态分块探针 /tmp/probe_dyn_chunked.py | 本次实际验证 | 静态整段=静态分块[5,7]=静态分块[1,8,3]=动态整段=动态分块[5,7]=动态分块[1,8,3]，token 全为 [58,124,53] | 本轮新写并执行，tiny Llama 4 DPU、T_max=16、prompt 12、PIM_DYN_SHAPE=1；问题1 的功能正确性直接证据（首版探针采样索引用错导致的两臂同错已修正后复核） |
| 超包络探针 /tmp/probe_dyn_overflow.py | 本次实际验证 | 静态臂与动态臂同文 ValueError（含处置建议） | 本轮执行，round02-2 修复现状复核 |
| test_chunked_prefill.py 单独实跑 | 本次实际验证 | 10 passed / 2 deselected（71s） | 本轮执行，确认分块测试现状全静态、无动态用例（问题1 依据） |
| 102.3% / 93.4% 算术核对 | 本次实际验证 | 49.94/48.83=1.0227、135.10/144.64=0.9340，与记录一致 | 本轮执行 |
| 4L 真实 MoE 对拍（tokens 六元组、单/双请求墙钟） | 复用已有验证 | 三臂 token 逐位一致；102.3%、93.4% 达标 | 来源 implement §4 与 dynamic-graph-20261113.md 末两节（2026-11-15，基线 7fcf056 同代码路径）；探针 /tmp/probe_4l_dynamic.py、probe_4l_parallel.py 本轮核对在位未动；本轮 28 用户、load 21+ 非独占按约束未复跑，token 与墙钟未独立复核 |
| 静态路径逐字节兼容（payload 新键、mode 分派、gate 行为、execute_plan 静态分支） | 仅核对实现但未执行（逐 hunk；行为级由本轮全量回归覆盖） | 四处均确认：键仅动态写入、分派严格按 mode、gate 逻辑本区间未改（hal_numpy 区间仅 +14 行且已删）、静态分支代码未动 | 9fc1b0d..7fcf056 + git diff HEAD 逐 hunk 对照；test_static_plan_has_no_shape_expr_payload 随子集实跑绿 |
| 前两轮修复复核（11 项：对齐不变式、异常测试、死代码、弱引用、文档表述、折叠单实现、runtime mode、超包络、兜底删除、设计文档同步） | 本次实际验证（静态核对 + 相关用例随子集/回归实跑） | 全部落实在根因位；arg_shape_exprs 四组平行列表同循环同点位（含 cat 未持有段 pop 与 literal_exprs 按 node.args 顺序的两次展开对齐，`_flatten_node_args` 只摊平全 Node 列表）；同类第四处遗漏未发现（tiling/融合子节点不携带表达式属保守包络，符合设计不变式 2） | 本轮逐项核对 exec_plan_gen.py:741-849、scheduler.py、shape_bind.py、executor.py、compile.py 与两轮评审记录 |
| 契约同步边界（genesim/scripts、meta/payload 键、折叠单实现、死代码残留） | 本次实际验证（grep 类静态核查） | genesim/scripts 无受改契约引用；fold_waits 全仓唯一；make_skipped_event/drain/runtime_graph/kernel_classes 无残留；dynamic_axes 为唯一写而不读字段（问题2） | 本轮 grep 全仓（含 ../genesim/scripts）执行 |

　　未覆盖范围：`llama2_7b` 慢用例与任何 CEVA/GeneSim 全流程按指令未跑，与本轮改动无关；4L 墙钟与 token 未独立复测，相关结论以文档记录为据并标注复用；性能数字的「同轮独占」口径无法在本轮事后复核，仅核对记录一致性与算术。上述限制不影响问题 1、2 的判定：问题 1 由测试目录静态证据与本轮探针实测直接支撑，问题 2 由全仓 grep 直接支撑。本轮探针与产物均在 /tmp，未写入任何仓库文件；评审过程未修改 docs 以外的任何文件。
