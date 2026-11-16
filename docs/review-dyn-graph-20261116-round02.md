# 代码评审报告：静态计算图 → host 驱动动态图（dyn-graph）
> 文档编号：review-dyn-graph-20261116-round02
> 创建日期：2026-11-16
> 关联需求文档：request-dyn-graph-20261116
> 关联设计文档：design-dyn-graph-20261116
> 关联实施文档：implement-dyn-graph-20261116

## 一、评审概述

　　本轮对已提交区间 9fc1b0d..7fcf056（动态图全部 18 个提交）叠加工作区未提交修复（docs 除外）做了第 2 轮审计：先逐条复核 round01 七个问题的修复是否落在根因上、同类路径是否真正覆盖（结论：问题 2/3/4/5/6/7 已修复到位，无「只在探针或单例上生效」的复发；问题 1 按用户 2026-11-16 裁决的方案 a 尚未落地，照常列出），再按设计六条不变式与重点检查清单逐项核对静态兼容红线、调度器语义、形状绑定判据、通信行收缩边界、开关门控一致性与仓库纪律。整体判断：分层架构与不变式忠实，静态路径逐字节兼容红线守住，两次通信实验回退干净，验收数字可复核（102.3%/93.4% 算术核对一致；全量回归本轮在最终树上复跑 2266 passed / 0 failed）。本轮发现问题：严重 0、高 2、中 0、低 2——高问题未清零（问题 1 为 round01 延续，问题 2 为本轮新发现的动态分支防线缺口，已用探针复现）；低问题为形状捕获处的防御性兜底残留与设计文档引用已删接口。八条核心目标：3/5/6/7 已完成，2/4 已完成（判据字面偏差已按实现语义澄清），1/8 部分完成，详见第三节。

## 二、修改建议

### 问题1（高）: 运行期回退开关仍未落地，导致验收判据1缺口延续
**问题说明**：需求验收判据 1 要求「`PIM_RUNTIME_MODE=static` 随时可回退」，用户已于 2026-11-16 裁决按方案 a 实施（`execute_plan` 动态分支入口读该变量改走静态全量提交路径）。本轮全仓 grep 确认代码仍无此变量（本次实际验证）；`execute_plan` 的 dynamic 分支（runtime/executor.py:386-396）无环境变量旁路，动态编译产物运行期永远走 `EventScheduler`；实施文档 §5 Q1（112-118 行）仍写「裁决前不实现」，与已下达的裁决之间出现实施空档。
**根本原因**：实施分层时把「可随时回退」收敛为「编译期 `dynamic=False` 默认 + `PIM_DYN_SHAPE` 关闭整体包络回退」双层机制，需求点名的运行期总开关从未排入任何提交；round01 后进入待裁决状态，裁决下达后未再启动实施。同一根因影响所有以 `compiled.compile_mode` 分派的执行入口（`step_request` 的 decode/prefill、`prefill_chunk`、`execute_plan`），它们都没有给运行期留统一切换点。
**涉及文件**：runtime/executor.py:386-396；docs/implement-dyn-graph-20261116.md 第 108-118 行（§5 Q1）。
**修改思路**：按已裁决方案 a 落地：`execute_plan` 动态分支入口在调用时读 `PIM_RUNTIME_MODE`（`os.environ.get`，不得 import 期缓存），为 `static` 时复用（而非复制）现有 static 分支的全量提交与等待路径，保证静态路径逐字节不变；同次补「动态编译 + `PIM_RUNTIME_MODE=static` → 走静态路径且 token 与默认动态一致」的行为测试（真断言分派路径与 token，不停留在字段检查）。落地后实施文档判据 1 由部分完成改已完成，并删除 §5 Q1 的待裁决表述。

### 问题2（高）: 动态分支绕过包络上限防线，导致越界执行与误诊
**问题说明**：静态路径对超过最长桶的输入在 `prefill_bucket` 抛带处置建议的 ValueError（runtime/executor.py:286-290，其报错信息本身说明该工况在实际运营中出现过）；动态分支（runtime/executor.py:279-281）在 `compile_mode=="dynamic"` 且开绑定时无条件 `return n_tokens`，不做任何上限校验。探针复现（本次实际验证，tiny Llama 4 DPU、T_max=16、prompt 20、`PIM_DYN_SHAPE=1`）：静态臂得到上述 ValueError；动态臂穿过选桶继续执行——块表无余量时死于无关的「块表长 2 超过块表项数 1」（误导定位），块表有余量（max_seq=32、kv_slot_tokens=4）时死于深层 `IndexError: index 19 is out of bounds for axis 1 with size 16`，即内核按 T=20 读 16 行包络缓冲。numpy 镜像尚且越界，真硬件（hal_vendor）上即 MRAM 越界读写。
**根本原因**：`prefill_bucket` 的动态早退只复用了「按原长选图」的语义，没有把静态分支附带的「token 数 ≤ 编译包络」校验一并继承；该校验在静态侧藏在选桶循环的兜底路径里，未被提升为两分支共享的入口约束。同一根因覆盖 `prefill_chunk` 的全部调用方——`step_request` 的整段 prefill、分块 prefill、多请求 wave 中的每个请求。
**涉及文件**：runtime/executor.py:279-281（缺口）、:286-290（静态侧既有校验）；唯一调用链 `prefill_chunk`（:244）。
**修改思路**：把「token 数不超过编译包络」的校验提到模式分支之前统一执行：动态包络取 `compiled.prefill_seq_lens[-1]`（动态下即 `prefill_seq_len`），超限抛与静态同文同建议的 ValueError；静态分支维持现状。这样两种模式在同一入口共享同一防线，后续再增模式分支也不会各自漏带。同次补红测：动态编译 + `PIM_DYN_SHAPE=1` + prompt > T_max → 断言抛 ValueError（而非 IndexError）。反向复核过：合法长度区间为 1..T_max（`Dim("T", min=1, max=seq_len)`），decode 不经 `prefill_bucket`，关绑定走静态分支仍有校验，收紧不误伤任何合法路径。

### 问题3（低）: 形状捕获留防御性兜底，导致缺陷被静默掩盖
**问题说明**：仓库纪律明确「不写防御性兜底……契约不满足就直接抛，让 bug 立刻暴露」。`capture_shape_exprs` 的逐维转换包了 `except Exception` 并回退包络常量（runtime/compile.py:159-161）。该兜底自相矛盾：try 体内最可能抛错的 `int(dim)` 在 except 分支里被原样重调，若真因 SymInt 无 hint 抛错，兜底自身会再抛同一异常；而若 `_sympy_to_shape_expr` 内部出错（如环境缺 sympy），则所有维静默退化为包络常量——绑定悄悄变空操作。过程日志记录过同类事故的代价：D2-lite 曾因绑定静默失效，67.7s 的「收益」实为包络执行噪声，靠剖析才查清。
**根本原因**：符号转换的合法退化规则（辅助符号退包络常量）已在 `_sympy_to_shape_expr` 内显式实现（`if not all(str(s) == dyn_symbol ...): return const`），捕获层又叠了一层无差别兜底，属设计外防御；无任何测试覆盖该分支。
**涉及文件**：runtime/compile.py:159-161。
**修改思路**：删除 try/except，让转换异常直接抛出（退化规则由 `_sympy_to_shape_expr` 显式承担）；若确有无 hint 维的合法场景，把它写成显式判断而非异常兜底，并补测试。同类排查：本区间新增模块（shape_bind/scheduler/dependency/lowering/exec_plan_gen 改动）grep 确认再无第二处 `except Exception`（本次实际验证）。

### 问题4（低）: 设计文档引用已删接口，导致基准漂移
**问题说明**：设计文档是本轮评审唯一基准之一，其不变式 3 写「gate 由 HAL 在 submit 内同步解析，跳过命令产生预完成 event（`make_skipped_event`）」（docs/design-dyn-graph-20261116.md:38）。round01 修复按评审决议删除了无调用方的 `NumpyBackend.make_skipped_event` 公共包装，现机制为 submit 内私有 `_skipped_event`（backend/hal_numpy.py:327）；设计文档（09:56 定稿，早于修复）未同步，基准文档指向不存在的接口。
**根本原因**：round01 修复只同步了代码与实施文档，没有把「公共接口面变更须回写三份基准文档」纳入修复动作，形成文档单向漂移；后续轮次若按设计文档核对代码会产生假偏差。
**涉及文件**：docs/design-dyn-graph-20261116.md:38。
**修改思路**：措辞改为「跳过命令产生预完成 event（submit 内私有 `_skipped_event`）」，并在实施流程中明确：凡删改公共接口/契约字段，设计文档与实施文档同次更新。

## 三、核心目标完成情况

| 需求目标 | 状态 | 证据 | 缺口或外部阻塞 |
|---------|------|------|----------------|
| 1. 静态默认、逐字节兼容、可随时回退 | 部分完成 | 静态兼容三处红线逐 hunk 核对（payload 新键仅动态导出写入、执行严格按 mode 分派、gate 逻辑未动），test_static_plan_has_no_shape_expr_payload 随本轮子集绿；静态全量回归本轮最终树上复跑 2266 passed / 0 failed（本次实际验证） | `PIM_RUNTIME_MODE=static` 未实现且裁决方案 a 未落地（问题1） |
| 2. 运行期依赖 ⊆ 静态 waits（折叠后）、shadow 在位且真实断言 | 已完成 | test_address_waits_are_a_subset_of_static_waits 断言 gaps==[] 且运行期依赖非空（本轮子集实跑）；折叠规则已收敛 `dependency.fold_waits` 单实现（grep 确认 exec_plan_gen 内联副本已删，本次实际验证） | 覆盖为附录 A 单计划；「相等」已按设计收敛为 ⊆（host 闭包依赖以 waits 为准），round01 已澄清、非缺口 |
| 3. token 逐 token 一致 | 已完成 | tiny Llama 7 种长度、tiny MoE 5 种长度逐 token 一致（用例自带 `PIM_DYN_SHAPE`，本轮 296 内实跑）；4L tokens `[6722, 4849, 35579, 1197, 5434, 5434]` 复用 2026-11-15 基线记录 | 4L 本轮机器非独占（28 用户、load≈12）未复跑，数字未独立复核 |
| 4. 一张 prefill 图覆盖 1..max_seq、形状类有界 | 已完成 | 单图由「额外桶参数必须为空」强制（compile.py ValueError），`prefill_seq_lens==(T_max,)` 单桶（test_dynamic_compile_records_mode_and_single_bucket）；形状类有界为结构性保证：绑定只改 `arg_kinds`/`arg_shapes`/`out_shape`，`mram_offset` 与内核查找键（按 kernel 名）不动（shape_bind.py 全文 + hal_numpy.lookup 核对） | `prefill_seq_lens==()` 判据字面不成立（round01 问题6 已澄清为「额外桶参数为空 + 字段 (T_max,)」）；超 max_seq 输入的拒绝防线缺失（问题2，属 1..max_seq 之外的行为） |
| 5. 单请求 ≤105%、多请求不劣化 | 已完成（复用） | 102.3%（49.94/48.83s）、93.4%（135.10/144.64s）同轮独占记录（dynamic-graph-20261113.md 末两节）；比值算术本轮核对一致（1.0227 / 0.9340，本次实际验证） | 本轮非独占未复测；跨轮绝对值波动为已确认外部约束 |
| 6. 双轨制不变式（不改 MRAM offset/缓存键/KV 几何/容量） | 已完成 | `bind_command_shapes` 仅改三组形状键、`replace` 不改原命令（test_bind_rewrites_shapes_for_symbolic_dims 断言原命令不动）；规划只用包络整数（capture 两遍捕获：先读全部 sympy 表达式再取整）；通信钳制只截 `global_range` 尾与 `nbytes`，不改段地址 | 无 |
| 7. 每个新增行为分支有同次测试 | 已完成 | round01 补齐后：提交/执行异常从 wait() 抛出 2 条（断言异常实例同一）、waits 校验与弱引用缓存 2 条、payload 对齐契约 2 条（本轮 296 内实跑）；关绑定回退包络（test_dynamic_without_binding_falls_back_envelope）、行收缩门控四否定态、gate 否定分支（test_moe_dispatch）均在 | 无 |
| 8. 仓库纪律（最小实现、无死代码、契约同步） | 部分完成 | round01 点名的死代码四处已删且 grep 无调用方残留（本次实际验证）；无调试残留与旧实现尸体（ready/probe/TODO grep 空）；meta 键登记于 unified_ir META_KEYS、payload 键写入/消费成对；`genesim/scripts` 无受改契约引用（grep，本次实际验证）；23049f7→a1417b8 回退干净、库层与 t_axes 防线自洽 | 形状捕获防御性兜底残留（问题3）；设计文档引用已删接口未同步（问题4） |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
|-------------|----------|------|---------------------|
| 定向动态子集 14 文件（每轮必跑） | 本次实际验证 | 296 passed / 0 failed（185.30s） | 本轮在 HEAD=7fcf056 + 工作区 round01 修复树执行；pytorch env（paths.json）；用例自带的 PIM_DYN_SHAPE 之外为裸环境 |
| 全量回归 `pytest tests/ -q -k "not llama2_7b"`（裸环境） | 本次实际验证 | 2266 passed / 11 skipped / 0 failed / 52 deselected（1400.62s） | 本轮在最终当前树（含 round01 修复与回归启动后才定稿的 test_runtime_scheduler.py 断言形态）后台复跑；与 round01 记录 2266/1384.89s 同数，消除其复用边界疑问；日志 /tmp/regress_round02.log |
| 4L 真实 MoE 对拍（tokens 六元组、102.3%、93.4%） | 复用已有验证 | 三臂 token 逐位一致；单请求 102.3%、双请求 93.4% 达标 | 来源 implement §4 与 dynamic-graph-20261113.md 末两节（2026-11-15，基线 7fcf056 同代码路径）；探针 /tmp/probe_4l_dynamic.py、probe_4l_parallel.py 本轮核对在位未动；本轮 28 用户、load≈12 非独占按约束未复跑，墙钟与 token 未独立复核 |
| 超包络探针 /tmp/probe_dyn_overflow.py | 本次实际验证 | 静态臂 ValueError（带处置建议）；动态臂块表无余量时死于无关块表错误、有余量时死于 IndexError 越界 | 本轮执行，tiny Llama 4 DPU、T_max=16、prompt 20、PIM_DYN_SHAPE=1，两档 kv_slot_tokens/max_seq 配置各一次 |
| `PIM_RUNTIME_MODE` 存在性 | 本次实际验证 | 全仓 `grep -rn` 无命中（代码与 docs 除需求文档本身） | 本轮执行 |
| 102.3% / 93.4% 算术核对 | 本次实际验证 | 49.94/48.83=1.0227、135.10/144.64=0.9340，与记录一致 | 本轮执行 |
| 静态路径逐字节兼容（payload 新键、mode 分派、gate 行为） | 仅核对实现但未执行 | 三处均确认：键仅动态写入、分派严格按 mode、gate 逻辑本区间未改 | 9fc1b0d..7fcf056 逐 hunk 对照；test_static_plan_has_no_shape_expr_payload 随本轮子集实跑通过 |
| 契约同步边界（genesim/scripts、meta/payload 键、折叠单实现、死代码残留） | 本次实际验证（grep 类静态核查） | genesim/scripts 无受改契约引用；折叠泳道规则仅 dependency.fold_waits 一份；make_skipped_event/字段/drain 无调用方残留；无调试残留 | 本轮 grep 全仓执行 |

　　未覆盖范围：`llama2_7b` 慢用例与任何 CEVA/GeneSim 全流程按指令未跑，与本轮改动无关；4L 墙钟与 token 未独立复测，相关结论以文档记录为据并标注复用；性能数字的「同轮独占」口径无法在本轮事后复核，仅核对记录一致性与算术。上述限制不影响问题 1-4 的判定：问题 1、3、4 由代码与文档静态证据直接支撑，问题 2 另有本轮探针实测复现；操作示例与文档引用均不作为实测结果表述。
