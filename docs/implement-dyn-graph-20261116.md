# 实施：静态计算图 → host 驱动动态图（dyn-graph）

日期：2026-11-16（随评审/修复就地更新的实施文档；详细过程日志见 `docs/dynamic-graph-20261113.md`）
关联需求文档：request-dyn-graph-20261116
关联设计文档：design-dyn-graph-20261116
关联评审文档：review-dyn-graph-20261116-round01、review-dyn-graph-20261116-round02、review-dyn-graph-20261116-round03
起始基线：main@7fcf056（提交链见第 2 节；评审后的修复为工作区未提交改动）

## 1. 核心目标完成度

| 判据（见 request-dyn-graph-20261116.md） | 状态 | 证据 |
| --- | --- | --- |
| 1. 静态模式全量回归绿、`PIM_RUNTIME_MODE=static` 随时可回退 | 已完成 | 静态全量回归绿（§4，裸环境不设 `PIM_DYN_SHAPE`，static 分支代码未动、`mode=="static"` 路径逐字节不变）；payload 形状表达式键仅动态导出写入（test_static_plan_has_no_shape_expr_payload）。运行期总开关已按用户 2026-11-16 裁决方案 a 落地：`execute_plan` 动态分支入口调用时读 `PIM_RUNTIME_MODE`（shape_bind.py `runtime_static_mode`），为 `static` 时复用既有静态全量提交与等待路径，且 `dynamic_shape_enabled()` 同步并入该开关，去填充/通信钳制/调度器绑定统一回退包络（无半开状态）；test_runtime_mode_static_routes_dynamic_to_static_path 断言不实例化事件调度器且 token 与默认动态一致，test_dynamic_default_keeps_scheduler_without_override 断言不设变量时动态行为不变（§6.1 问题1） |
| 2. 运行期依赖 ⊆ 静态 waits（折叠后）、token 逐 token 一致 | 已完成 | shadow 断言语义为「运行期地址依赖 ⊆ 静态 waits」：test_address_waits_are_a_subset_of_static_waits（gaps==[] 且运行期依赖非空；host 闭包依赖不在地址分析内、以计划 waits 为准，非严格相等）；tiny Llama 7 种长度、tiny MoE 5 种长度逐 token 一致（用例自带 `PIM_DYN_SHAPE`，§4 定向子集实跑）；4L 真实 MoE tokens `[6722, 4849, 35579, 1197, 5434, 5434]`（复用已有验证，§4） |
| 3. 一张 prefill 图覆盖 1..max_seq、形状类有界 | 已完成 | 单图由「额外桶参数必须为空」保证（compile.py `_compile` 入口 ValueError 强制），模型字段为 `prefill_seq_lens==(T_max,)` 单桶——关绑定回退包络正依赖它补 pad（test_dynamic_compile_records_mode_and_single_bucket 断言 `(MAX_SEQ,)`）。超 max_seq 输入的拒绝防线已提到模式分支之前两模式共用（prefill_bucket 入口，test_dynamic_prompt_over_envelope_raises）。形状类有界是结构性保证：运行期绑定只改 `arg_kinds`/`arg_shapes`/`out_shape`，不改 `mram_offset` 与内核缓存键；`kernel_classes` 计量字段未实现、round01 已删（§6 问题4，D4b 拆地址键前无消费者） |
| 4. 单请求墙钟 ≤105%、多请求不劣化 | 已完成（复用） | 同轮独占：单请求 static 48.83s / dynamic_bind 49.94s = 102.3%；双请求×2 窗口 144.64s / 135.10s = 93.4%（2026-11-15 记录，算术核对一致；跨轮绝对值波动为已确认外部约束） |
| 5. 新增行为有同次提交的测试、文档一致 | 已完成 | round01 补齐调度器测试 4 条（wait() 异常传播 2、waits 校验与缓存释放 2）、payload 表达式对齐 2 条；round02 修复同次补 4 条（运行期回退分派与 token 一致 2、超包络防线 1、转换异常传播 1，§6.1）；round03 补齐「动态 × 分块 prefill」矩阵与契约 5 条（首段 1 token、尾段非整桶、d2h 包络行契约）及「动态 × 多请求 wave」1 条（两泳道并发绑定不同 T），并经变异验证证明分块组合缺陷此前无测试可拦截（§4）；形状捕获的防御性兜底已删，设计文档接口引用已同步；验证命令与结果见 §4 |

## 2. 提交链（评审范围 9fc1b0d..7fcf056，共 18 个提交）

| 提交 | 层 | 内容 |
| --- | --- | --- |
| e81754f | D0 | RuntimeGraph 契约与静态兼容层 |
| 3a2fc2c | D1 | 事件驱动调度器与地址依赖 shadow |
| 0059a45 | D2-lite | 符号 T 单图 prefill 与形状表达式 |
| a884afc / 73deddd | 文档/格式 | 动态图分层改造记录 |
| 8e73dbc | D2a | 运行期按实际 T 绑定内核形状 |
| 4b0de02 | D1 修复 | 同道保序修复并接入解码循环 |
| 7f4bf92 | D1 收敛 | 调度器改计划序全量提交（就绪驱动证伪） |
| ce99ca9 | 清理 | 补回 `_shape_env`、清理调度器死代码 |
| 958dca8 | 文档 | 终版数字与提交链 |
| e03a22b → 23049f7 → a1417b8 | D2-full 实验 | 通信行裁剪三次实验（前两次破坏 token 已回退，保留库层与 t_axes 防线） |
| a1b5e7b | 测试 | FakeHAL 补字段 |
| a411403 | D2-lite 修复 | 形状绑定修正（符号名/取整污染/字面量形状）+ 前缀布局门控 |
| c9aeb0f | D2-full | 动态 prefill 去填充 + scatter 行数限制 |
| 8d62ca0 | D2-full | gather 收缩接线（d2h/专家条目保持包络） |
| 7fcf056 | 性能 | 调度器等待 join+逐 future 直等、计划校验只做一次 |

## 3. 主要改动清单

首次实施（9fc1b0d..7fcf056）：

- 新增：`contracts/runtime_graph.py`、`runtime/dependency.py`、`runtime/scheduler.py`、
  `runtime/shape_bind.py`；测试 `test_runtime_graph_contract.py`、`test_engine_static_compat.py`、
  `test_runtime_scheduler.py`、`test_shape_binding.py`、`test_dynamic_shape_compile.py`、
  `test_dynamic_moe_binding.py`、`test_comm_shape_exprs.py`、`test_comm_row_clamp.py` 等。
- 修改：`runtime/compile.py`（dynamic 编译、单 prefill 图、形状表达式捕获）、
  `runtime/exec_plan_gen.py`（payload 形状表达式、`_clamped_rows`、d2h/专家边界）、
  `runtime/executor.py`（dynamic 分派、去填充、`dynamic_shape_enabled()` 门控）、
  `comm/lowering.py`（行收缩收集原语、`t_axes` 契约）、`comm/plan.py`（t_axes/shape_exprs）、
  `backend/hal_numpy.py`（gate 同步解析）、`graph/opt_passes.py`（融合复制形状表达式）、
  `contracts/unified_ir.py`、`contracts/ir_payloads.py`（meta/payload 键）、
  `backend/architectures.py`（MoE 执行开关）。

评审 round01 修复（工作区未提交，逐条记录见 §6.1）：

- `runtime/exec_plan_gen.py`：`arg_shape_exprs` 构建移进实参编码循环，与 `arg_shapes`
  同点位（字面量/部分和占 `None`、cat 未持有段同步跳过），两组列表逐位对齐成为
  构造性不变式；`_PlanBuilder.append` 改调 `dependency.fold_waits`，删除内联折叠副本。
- `runtime/scheduler.py`：`_VALIDATED` 改弱引用登记（按 `id` 键的
  `WeakValueDictionary`），计划被回收后条目自动清除，地址复用不再误跳过校验；
  `run()` 删除 `drain` 参数。
- `runtime/executor.py`：`execute_plan` 删除 `drain` 参数（无任何调用方使用）。
- `backend/hal_numpy.py`：删除无调用方的 `make_skipped_event` 公共包装（submit 内部
  仍用私有 `_skipped_event`，语义不变）。
- `runtime/compile.py`：删除全仓无写入方的 `runtime_graph`、`kernel_classes` 字段
  （D3 契约预留按需求只覆盖 `contracts/runtime_graph.py` 的结构，不覆盖执行器字段）。
- `runtime/dependency.py`：`fold_waits` docstring 标注「静态计划生成与 shadow 校验
  共用的唯一实现」。
- 测试：`test_exec_plan_gen.py` 新增对齐红测（复现 K 向分块下 slice/cat 的长度错位）、
  `test_dynamic_shape_compile.py` 新增对齐契约测试、`test_runtime_scheduler.py` 新增
  提交异常/执行异常/未知 waits/缓存释放 4 条并删除死代码、`test_engine_static_compat.py`
  同步字段断言。

评审 round02 修复（工作区未提交，逐条记录见 §6.1 表尾 4 行）：

- `runtime/shape_bind.py`：新增 `runtime_static_mode()`（调用时读
  `PIM_RUNTIME_MODE`，不缓存）；`dynamic_shape_enabled()` 并入该开关，运行期回退
  静态时绑定、executor 去填充、通信钳制统一失效，整体回退包络。
- `runtime/executor.py`：`execute_plan` 动态分支入口按 `runtime_static_mode()` 回退
  静态全量提交路径（复用既有分支）；`prefill_bucket` 把「token 数 ≤ 最长桶」校验
  提到模式分支之前，两模式共用同一 ValueError 防线。
- `runtime/compile.py`：删除 `capture_shape_exprs` 载体转换的 try/except 兜底，
  退化规则由 `_sympy_to_shape_expr` 显式承担。
- `docs/design-dyn-graph-20261116.md`：不变式 3 接口引用同步为 submit 内私有
  `_skipped_event`（round01 删除公共包装后的文档收尾）。
- 测试：`test_dynamic_shape_compile.py` 新增 4 条（运行期回退分派 + token 一致、
  默认动态行为不变、超包络 ValueError、转换异常传播）。

评审 round03 修复（工作区未提交，逐条记录见 §6.1）：

- `tests/test_chunked_prefill.py`：`_build` 增 `dynamic` 参数并新增动态编译
  fixture（KV 几何与静态短句产物一致）；`_chunked` 支持按段长列表切段；新增
  「模式 × 分段形状」矩阵用例（整段锚点、[5,7]、[1,8,3]、8+4，对拍静态基线）
  与动态分块返回契约用例（logits 保持包络行数、本段 token 占前若干行、KV 按
  绝对位置推进）。
- `tests/test_multi_request_decode.py`：`_compile` 增 `dynamic` 参数；新增
  动态 × 多请求 wave 用例（两条泳道并发绑定不同 T，与单独运行逐 token 一致）。
- `runtime/compile.py`：删除写而不读的 `dynamic_axes` 字段、构造写入与
  `field` 导入（round01-4 的同类残留，信息与 `prefill_seq_lens[-1]` 冗余）；
  `test_dynamic_shape_compile.py`、`test_engine_static_compat.py` 的两条值
  断言同步删除。

## 4. 验证命令与结果

round03 修复后（均在仓根目录、先 source pytorch 环境）：

- 定向动态子集（14 文件，每轮必跑）：
  `python -m pytest tests/test_runtime_graph_contract.py tests/test_engine_static_compat.py tests/test_runtime_scheduler.py tests/test_dynamic_shape_compile.py tests/test_shape_binding.py tests/test_dynamic_moe_binding.py tests/test_comm_shape_exprs.py tests/test_comm_row_clamp.py tests/test_comm_lowering.py tests/test_moe_dispatch.py tests/test_exec_plan_gen.py tests/test_exec_binding.py tests/test_prefill_buckets.py tests/test_opt_passes.py -q`
  → 300 passed / 0 failed（206s；与 round03 评审基线同数，本轮对其中两文件
  仅删 `dynamic_axes` 断言、未增用例）。
- 本轮直接受影响文件：
  `python -m pytest tests/test_chunked_prefill.py -q`
  → 15 passed / 2 deselected（原 10 条全静态 + 新 5 条动态矩阵/契约）；
  `python -m pytest tests/test_multi_request_decode.py -q` → 5 passed
  （含新增动态 wave 1 条）；
  `python -m pytest tests/test_dynamic_shape_compile.py tests/test_engine_static_compat.py -q`
  → 24 passed（删除断言后全绿）。
- 测试有效性变异验证（替代红绿流程中的「先见红」）：把 scheduler
  `_shape_env` 的 T 临时改为 `pos+num_tokens`（段末绝对位置，整段 pos=0 不受
  影响），动态分块矩阵的 3 个分段形态与契约用例立即翻红（kernels_pim.py
  IndexError），整段锚点用例与既有 9 条动态整段用例（tokens_match、
  uses_actual_length、without_binding）全部仍绿——证明「动态 × 分块」组合的
  绑定缺陷在本轮之前无任何测试可拦截；还原后全部通过（diff 确认与变异前
  逐字节一致）。
- 全量回归（裸环境，不设 `PIM_DYN_SHAPE`，所有源码修改完成后后台启动）：
  `python -m pytest tests/ -q -k "not llama2_7b"`
  → 2276 passed / 11 skipped / 0 failed / 52 deselected（1648s，详见 §6.3）。
- 4L 真实 MoE 对拍：复用已有验证——本轮修复只新增测试并删除写而不读字段，
  未触及执行/绑定/通信语义，按约束未复跑；三臂 token 逐位一致与
  102.3%/93.4% 引用 2026-11-15 基线记录。探针 `/tmp/probe_4l_dynamic.py`、
  `/tmp/probe_4l_parallel.py` 在位未动。

round02 修复后（历史记录，命令同上定向子集）：

- 定向动态子集（14 文件，每轮必跑）：
  `python -m pytest tests/test_runtime_graph_contract.py tests/test_engine_static_compat.py tests/test_runtime_scheduler.py tests/test_dynamic_shape_compile.py tests/test_shape_binding.py tests/test_dynamic_moe_binding.py tests/test_comm_shape_exprs.py tests/test_comm_row_clamp.py tests/test_comm_lowering.py tests/test_moe_dispatch.py tests/test_exec_plan_gen.py tests/test_exec_binding.py tests/test_prefill_buckets.py tests/test_opt_passes.py -q`
  → 300 passed / 0 failed（181s；round02 评审基线 296 + 本轮新增 4 条全部在内）。
- 全量回归（裸环境，不设 `PIM_DYN_SHAPE`，所有源码修改完成后后台启动）：
  `python -m pytest tests/ -q -k "not llama2_7b"`
  → 2270 passed / 11 skipped / 0 failed / 52 deselected（1446s，详见 §6.3）。
- 超包络探针复跑（`python /tmp/probe_dyn_overflow.py`，tiny Llama 4 DPU、T_max=16、
  prompt 20、`PIM_DYN_SHAPE=1`）：两臂均抛同文
  `ValueError: token 数 20 超过最长 prefill 桶 16；建议调大 PIM_PREFILL_SEQ_LEN 或
  调小 --max-num-batched-tokens`（round02 评审时动态臂死于无关 IndexError，已消除）。
- 4L 真实 MoE 对拍：复用已有验证——本轮修复触及执行分派与绑定开关语义，但机器
  非独占（28 用户、load 15+，另有全量回归在跑）按约束未复跑；三臂 token 逐位一致
  与 102.3%/93.4% 引用 2026-11-15 基线记录。探针 `/tmp/probe_4l_dynamic.py`、
  `/tmp/probe_4l_parallel.py` 在位未动。
- 桶/分块相关既有路径复核：`tests/test_prefill_buckets.py tests/test_chunked_prefill.py`
  50 passed / 0 failed（含静态侧同错同文的 test_prompt_longer_than_longest_bucket_raises、
  关绑定包络回退、分块 prefill 全链）。

round01 修复后（历史记录，命令同上定向子集）：

- 定向动态子集（14 文件，每轮必跑）：
  `python -m pytest tests/test_runtime_graph_contract.py tests/test_engine_static_compat.py tests/test_runtime_scheduler.py tests/test_dynamic_shape_compile.py tests/test_shape_binding.py tests/test_dynamic_moe_binding.py tests/test_comm_shape_exprs.py tests/test_comm_row_clamp.py tests/test_comm_lowering.py tests/test_moe_dispatch.py tests/test_exec_plan_gen.py tests/test_exec_binding.py tests/test_prefill_buckets.py tests/test_opt_passes.py -q`
  → 296 passed / 0 failed（179s；评审基线 290，新增 6 条用例全部在内）。
- 全量回归（裸环境，不设 `PIM_DYN_SHAPE`）：
  `python -m pytest tests/ -q -k "not llama2_7b"`
  → 2266 passed / 11 skipped / 0 failed / 52 deselected（1385s）。
- 4L 真实 MoE 对拍：复用已有验证——本轮修复虽触及 payload 构造与调度器，但机器
  非独占（32 用户会话）未复跑；三臂 token 逐位一致与 102.3%/93.4% 引用 2026-11-15
  基线记录。探针 `/tmp/probe_4l_dynamic.py`、`/tmp/probe_4l_parallel.py` 在位未动。
- 评审对齐探针复跑（`python /tmp/probe_align_dyn.py`，tiny Llama 4 DPU 动态编译）：
  携带表达式的 launch 1248 条、concat 64 条全部携带，与评审记录同量。探针报
  536 条「misaligned」是其判据按**旧压缩格式**写的假阳性（要求「前 len(exprs) 个
  形状全为张量」）；修复后字面量位是「None 形状 ↔ None 表达式」的正确对位
  （`to.dtype`/`view` 的字面量尾随实参），新不变式由
  test_arg_shape_exprs_align_with_payload_positions 在同一模型上断言通过。
- 测试有效性验证（替代红绿流程中的「先见红」）：把 `scheduler.wait()` 临时改为
  吞异常的旧缺陷形态，两条异常传播用例立即失败；还原后通过；`_VALIDATED` 缓存
  用例同样以「变异回旧 id 集合实现 → 翻红 → 还原 → 通过」验证。
- 说明：全量回归启动后，`test_validation_cache_drops_collected_plans` 的断言形态
  从「全局清空」收紧为「本计划条目消失」（避免其他用例的存活计划干扰，旧实现在
  两种形态下均为红、新实现均为绿）；该文件最终态单独复验 7 passed，属测试断言
  调整、非主链代码改动，不触发回归重跑。

2026-11-15 基线（首次实施收口）：定向动态子集与静态基线子集共 138+25 passed；
全量回归记录 2308 passed / 11 skipped / 0 failed（7698s；该收集数在 HEAD 上无法
复现、且日志自注「机器有其他负载」，同日同命令的可比记录为 2260 passed / 1423s，
round01 已核对按不可比口径处理）；4L 真实 MoE
（`/data2/pengcheng-work/Qwen3-30B-A3B-4L`，8 DPU，prefill 8，prompt 5 token）
三臂 token 逐位一致，单/双请求墙钟见 §1 判据 4。

## 5. 待确认事项与遗留

待确认事项：

- Q1（round01 问题1，高）`PIM_RUNTIME_MODE=static` 运行期回退总开关：已关闭。用户
  2026-11-16 裁决按方案 a 实施，round02 已落地：`execute_plan` 动态分支入口调用时读
  `PIM_RUNTIME_MODE`，为 `static` 时复用静态全量提交路径，`dynamic_shape_enabled()`
  并入该开关保证整体包络回退（无半开）；测试证据见 §6.1 round02 问题1，判据 1 已改
  已完成。

遗留（非本轮缺陷，沿用首次实施记录）：

- D3（动态控制流）、D4（slot arena）：按需求边界只做契约预留，不实施；多 batch
  调度启动后再立项。
- d2h 全链实际 T：被专家路径阻塞（路由缓冲非 T 前缀），需真正的动态专家分派，未立项。
- 多 T 轴布局重映射（`(1,1,T,T)` mask 收缩）：未做，收益仅 numpy 掩码构造，非缺陷。
- 跨轮墙钟绝对值波动大（52–70s，机器负载相关）；性能结论只取同轮独占对照。

## 6. 评审问题修复记录（round01、round02、round03）

### 6.1 问题确认与修复清单

| 问题 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式与根因覆盖 | 对应测试 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 运行期回退开关未实现，判据 1 记为已完成 | 高 | 成立：全仓 grep 无 `PIM_RUNTIME_MODE`；实施文档证据只覆盖「回归绿」半句 | 实施把「可随时回退」收敛为编译期开关 + 关绑定包络回退，未落需求点名的运行期总开关，文档沿用需求措辞记成已完成 | 已修复（round01 先做文档修正：判据 1 改部分完成并登记 §5 Q1；用户裁决方案 a 后 round02 落地机制，见 round02-1 行） | 文档修正（§1 判据 1、§5 Q1）+ round02 机制落地 | 见 round02-1 行 |
| 2 `arg_shape_exprs` 与 `arg_shapes` 索引错位（潜伏） | 中 | 成立并复现：K 向分块图上 slice 命令 1 条表达式对 4 个形状位、cat 命令 4 对 5，红测失败于 `assert 1 == 4` | 两组平行列表在两个循环、两套过滤规则下构建，无构造性对齐不变式；`arg_elem_exprs` 已有的同循环对齐处理漏掉了 `arg_shape_exprs`（a411403 同族第四处）。全仓唯一生产方 exec_plan_gen、唯一消费方 shape_bind（按位 zip/枚举），修单点即根治 | 已修复 | `arg_shape_exprs` 构建移进实参编码循环：字面量/部分和占 `None`、cat 未持有段与 `arg_shapes` 同步跳过（exec_plan_gen.py:746,769,780,799,845） | test_exec_plan_gen.py::test_arg_shape_exprs_align_with_arg_positions_when_literals_mix_in（红→绿）；test_dynamic_shape_compile.py::test_arg_shape_exprs_align_with_payload_positions（真实路径契约） |
| 3 调度器异常传播零测试 | 中 | 成立：原 test_runtime_scheduler.py 仅 3 用例，无任何用例触发提交异常或 future 异常；实现语义核对正确 | 7fcf056 把异常传播作为性能调优的附带修复随车，未走先写失败测试流程；`_FakeHAL` 缺可控失败变体 | 已修复 | 新增 `_FailingHAL`（可控提交异常/失败 future）与两条用例，断言异常实例同一（`excinfo.value is boom`）；用变异验证（临时把 wait() 改回吞异常形态 → 两用例翻红 → 还原转绿）替代红绿流程 | test_runtime_scheduler.py::test_submit_error_surfaces_from_wait、::test_execution_error_surfaces_from_wait |
| 4 死代码与投机接口四处 | 低 | 成立：grep 证实 `make_skipped_event`（hal_numpy 与 FakeHAL）、`runtime_graph`/`kernel_classes` 字段、`drain` 参数均无生产调用方 | 两次收敛（7f4bf92、7fcf056）只删直接替代实现未清接口面；D0 按预留写入了本阶段无消费者的字段 | 已修复 | 删 `make_skipped_event` 公共包装、FakeHAL 镜像与恒真 `skipped` 断言及字段；删 `runtime_graph`/`kernel_classes` 字段（D3 预留按需求只在 contracts/runtime_graph.py，不覆盖执行器字段）；删 `execute_plan(drain)` 与 `run(drain)`，提交顺序用例改为直驱 `EventScheduler._loop` 线程，语义不变 | test_runtime_scheduler.py::test_scheduler_submits_all_commands_in_plan_order（直驱 _loop 重构后仍绿）；test_engine_static_compat.py::test_dynamic_entry_points_default_to_static（同步删除断言） |
| 5 `_VALIDATED` 按 id 误命中且只增不减 | 低 | 成立并复现：变异回旧 id 集合实现时，计划回收后其条目仍留在缓存（红测翻红）；弱引用实现下同一用例通过 | 用值语义的 `id()` 做对象身份缓存，把「计划常驻」假设写进跨调用模块状态；短命计划回收后地址复用会让新计划跳过 waits 校验 | 已修复 | 改弱引用登记：按 `id` 键的 `weakref.WeakValueDictionary`，条目随计划回收自动清除（scheduler.py:13-23）。评审建议的 `WeakSet` 不可用——`ExecutionPlan` 是自动生成 `__eq__` 的 dataclass（unhashable），弱值字典语义等价（见 §6.2） | test_runtime_scheduler.py::test_validation_cache_drops_collected_plans（旧实现红→弱引用绿）、::test_plan_with_unknown_wait_is_rejected（fail-fast 分支，原先同样零覆盖） |
| 6 实施文档证据表述与代码不符（判据 2「相等」、判据 3 `prefill_seq_lens==()`） | 低 | 成立：`CompiledModel.prefill_seq_lens` 动态时为 `(T_max,)`（compile.py，`seq_lens = ((prefill_seq_len,) if dynamic ...)`），`()` 的是额外桶参数（ValueError 强制为空）；shadow 断言为 ⊆ 不是相等 | 文档收口沿用需求原始措辞复述证据，未跟随设计对语义的收敛改写 | 已修复 | §1 判据 2/3 证据改为可核对事实：「额外桶参数必须为空、模型字段 `(T_max,)` 单桶」；「shadow 断言 ⊆（host 闭包依赖以 waits 为准），非严格相等」 | 无新增（纯文档修正） |
| 7 等待折叠规则双实现 | 低 | 成立：`dependency.fold_waits` 与 `_PlanBuilder.append` 内联折叠逐字重复，shadow 等价性依赖两份实现一致 | D1 抽取依赖判定时折叠逻辑留下旧副本，新模块又为 shadow 重写一份，形成按约定的重复 | 已修复 | `_PlanBuilder.append` 改调 `fold_waits(waits, self.lane_of)`，删内联副本（exec_plan_gen.py:451-457）；`fold_waits` docstring 标注唯一真源。红测试不可构造（两份实现当前逐字等价），以既有等价性测试守护（§6.2） | 既有 test_address_waits_are_a_subset_of_static_waits + 全部计划依赖/数值用例（本轮定向子集与全量回归全绿） |
| round02-1 运行期回退开关仍未落地（裁决方案 a 空档） | 高 | 成立：全仓 grep 无 `PIM_RUNTIME_MODE`（本轮再验证）；红测先证 `PIM_RUNTIME_MODE=static` 无效（事件调度器仍被实例化，AssertionError 翻红） | 「可随时回退」被实施收敛为编译期双层机制（`dynamic=False` 默认 + 关绑定包络回退），运行期总开关从未排入提交；round01 后进入待裁决、裁决下达后未启动。所有 `mode=compiled.compile_mode` 的执行入口（step_request decode、prefill_chunk 及其全部调用方）都汇聚在 `execute_plan`，且 `EventScheduler` 全仓唯一实例化点也在其中——单一分派点即根因位 | 已修复 | 按裁决方案 a：`execute_plan` 动态分支入口调用时读 `PIM_RUNTIME_MODE`（shape_bind.py `runtime_static_mode`，调用时读不缓存、monkeypatch 友好），为 `static` 时落入既有静态全量提交与等待分支（复用不复制，static 分支零改动）。红测同时暴露只切提交路径会形成「去填充输入 + 包络内核」半开状态（垃圾索引 IndexError），故把开关并入 `dynamic_shape_enabled()`，executor 去填充、通信钳制、调度器绑定三处门控统一失效，整体回退包络（设计不变式 5），见 §6.2 取舍说明 | test_runtime_mode_static_routes_dynamic_to_static_path（红→绿：断言不实例化事件调度器 + token 与默认动态一致）、test_dynamic_default_keeps_scheduler_without_override（不设变量时动态行为不变） |
| round02-2 动态分支绕过包络上限防线（越界执行与误诊） | 高 | 成立并复现：/tmp/probe_dyn_overflow.py 当前树重跑，静态臂 ValueError（带处置建议）、动态臂死于无关块表错误或深层 IndexError（内核按 T=20 读 16 行包络缓冲） | `prefill_bucket` 的动态早退只复用了「按原长选图」，未把静态侧附带的「token 数 ≤ 编译包络」校验一并继承；该校验藏在静态选桶循环的兜底路径，未提升为两分支共享的入口约束。唯一生产调用链 `prefill_chunk`（step_request 整段 prefill、分块 prefill、多请求 wave）全部经此入口，修单点即根治 | 已修复 | 校验提到模式分支之前统一执行：超过 `prefill_seq_lens[-1]`（动态即 T_max 包络）抛与静态同文同建议的 ValueError；静态侧同输入同错同文（test_prompt_longer_than_longest_bucket_raises 仍绿），原循环尾兜底 raise 提升后成为死代码删除；合法区间 1..T_max 不受影响、decode 不经此入口 | test_dynamic_prompt_over_envelope_raises（红→绿：prefill_bucket 单元 + step_request 端到端均断言 ValueError 而非 IndexError）；探针复跑两臂同文报错 |
| round02-3 形状捕获防御性兜底掩盖缺陷 | 低 | 成立：compile.py 载体转换包 `except Exception` 回退包络常量，except 分支还重调 `int(dim)`（真无 hint 时兜底自身再抛同一异常）；全仓区间内模块 grep 无第二处 `except Exception` | 合法退化规则（辅助符号退包络常量）已在 `_sympy_to_shape_expr` 内显式实现，捕获层又叠一层无差别兜底属设计外防御；同函数的标量路径本就直接调用无兜底，属同函数内不一致 | 已修复 | 删 try/except，转换异常直接抛出；真实动态导出全链用例（7 种长度 token 一致等）证明无合法场景依赖该兜底，无需显式判断替代 | test_capture_shape_exprs_propagates_conversion_errors（红→绿：monkeypatch `_sympy_to_shape_expr` 抛哨兵，断言 `capture_shape_exprs` 传播异常而非静默退化成包络常量） |
| round02-4 设计文档引用已删接口（基准漂移） | 低 | 成立：grep 全仓无 `make_skipped_event`（round01 已删公共包装），设计文档不变式 3 仍以该公共接口名描述机制 | round01 修复只同步了代码与实施文档，未把「公共接口面变更须回写基准文档」纳入修复动作，形成文档单向漂移 | 已修复 | 按评审措辞把不变式 3 改为「submit 内私有 `_skipped_event`」；机制语义未变（跳过命令产生预完成 event），属 round01 已裁决删除动作的文档同步收尾，非设计变更 | 无新增（纯文档同步；`_skipped_event` 行为由既有 test_moe_dispatch 用例守护） |
| round03-1 动态编译 × 分块 prefill 零测试覆盖（同类波及「动态 × 多请求 wave」无 tiny 级用例） | 中 | 成立：test_chunked_prefill.py 原有 10 条用例全部静态编译（实跑 10 passed 复核），全仓设 `PIM_DYN_SHAPE` 的测试仅 3 文件、无分块场景；wave 路径（run_one_step 并发窗口）测试全静态。功能经评审探针 /tmp/probe_dyn_chunked.py 实测正确，定性为测试缺口而非功能缺陷 | 分块测试基线按静态桶语义建立（早于动态改造），D2 给 `prefill_chunk`/`prefill_bucket` 新增去填充与绑定行为时只把整段入口纳入同次测试矩阵，分块入口没有随行为分支进矩阵；同一根因波及「动态 × 多请求 wave」（此前仅有 4L 探针复用证据） | 已修复 | 在 test_chunked_prefill.py 按「模式 × 分段形状」矩阵补动态维：整段锚点、首段 1 token（[1,8,3]）、尾段非整桶（[5,7] 与 8+4）对拍静态基线 token；另加动态分块返回契约用例（d2h 保持包络行数、本段 token 占前若干行、末行取段内偏移、KV 按绝对位置推进）。「动态 × wave」在 test_multi_request_decode.py 补 run_one_step 双请求用例（两 prompt 长度不同，两条泳道并发绑定不同 T）。后续新模式分支在同一矩阵扩展 | test_chunked_prefill.py::test_dynamic_chunked_prefill_matches_static（4 参数）、::test_dynamic_chunk_binds_each_segment_length、test_multi_request_decode.py::test_two_requests_wave_match_individual_runs_dynamic；变异验证见 §4 |
| round03-2 `dynamic_axes` 字段写而不读 | 低 | 成立：grep 全仓（含 scripts/、genesim/scripts）仅 compile.py 定义/写入与两条测试值断言，无生产读取方；设计文档无该字段登记（grep 零命中） | round01-4 按「全仓无写入方」口径筛选删除对象，漏掉「有写入、无读取」的字段，`dynamic_axes` 遗留为同类残留（round03 评审同类排查：`CompiledModel` 其余字段均有运行时消费者，无第二个同类） | 已修复 | 删字段定义、构造写入、失去使用者的 `field` 导入与两条测试断言；信息与 `prefill_seq_lens[-1]` 冗余，多 batch 调度立项时随首个真实消费者从 git 历史加回。删除与登记二选一的取舍见 §6.2 | test_dynamic_shape_compile.py、test_engine_static_compat.py 全绿（24 passed）；删除后全仓 grep 零残留 |

核心修复的「原理 → 位置 → 证据」：

- 对齐不变式（问题2）：运行期绑定按下标把 `arg_shape_exprs[i]` 配到 `arg_shapes[i]`
  （shape_bind.py `_command_bindable` zip、`bind_command_shapes` 枚举），因此两组列表
  必须在同一个循环里按同点位构建 → runtime/exec_plan_gen.py:746-845 → 红测
  `test_arg_shape_exprs_align_with_arg_positions_when_literals_mix_in` + 真实路径契约
  `test_arg_shape_exprs_align_with_payload_positions`。
- 异常传播（问题3）：设计不变式 3「提交异常与执行异常都从 wait() 抛出」→
  runtime/scheduler.py:49-74（error 先抛、逐 future 透传，本就正确，缺的是测试）→
  两条异常用例 + 变异验证。
- 弱引用校验缓存（问题5）：校验按计划身份只做一次，但身份登记必须随对象生命周期
  → runtime/scheduler.py:13-23 → `test_validation_cache_drops_collected_plans`、
  `test_plan_with_unknown_wait_is_rejected`。
- 折叠单实现（问题7）：折叠规则是「同 DPU 留最后前驱、host 逐条」的唯一真源 →
  runtime/dependency.py:41-49 + runtime/exec_plan_gen.py:451-457 → shadow 子集测试
  与全量回归。
- 运行期回退开关（round02 问题1）：需求判据「随时可回退」+ 设计不变式 5「整体回退
  包络、不允许半开」→ runtime/shape_bind.py:12-19（`runtime_static_mode` 唯一真源，
  `dynamic_shape_enabled` 并入）+ runtime/executor.py:387（`execute_plan` 动态分支
  入口）→ test_runtime_mode_static_routes_dynamic_to_static_path、
  test_dynamic_default_keeps_scheduler_without_override。
- 共享包络防线（round02 问题2）：两种编译模式的 prefill 入口共用「token 数 ≤ 最长桶」
  约束，防线在模式分派之前 → runtime/executor.py:274-290（`prefill_bucket`）→
  test_dynamic_prompt_over_envelope_raises + 既有 test_prompt_longer_than_longest_bucket_raises。
- 转换异常传播（round02 问题3）：仓库纪律「契约不满足直接抛」，退化规则只在
  `_sympy_to_shape_expr` 显式承担 → runtime/compile.py:158 →
  test_capture_shape_exprs_propagates_conversion_errors。
- 动态分块矩阵（round03 问题1）：分块与整段共用 `prefill_chunk`/
  `prefill_bucket` 入口，段级差异在「每段绑定不同 T + pos 递增写 KV + 段内
  行偏移读 logits」→ tests/test_chunked_prefill.py（矩阵 4 条 + 契约 1 条）+
  tests/test_multi_request_decode.py::test_two_requests_wave_match_individual_runs_dynamic
  （wave 双泳道不同 T）→ 变异验证（T 改段末绝对位置 → 3 分段形态翻红、
  整段与既有动态整段用例仍绿）。
- 死字段清理（round03 问题2）：仓库纪律「删优于加」，写而不读即死代码 →
  runtime/compile.py（字段/构造写入/`field` 导入三处）+ 两条测试断言 →
  两测试文件全绿、全仓 grep 零残留。

### 6.2 未修复问题与实现取舍说明

- 问题1（运行期回退开关）round01 未写代码：评审明确「二选一并闭环，属需求层面取舍、
  需回用户确认，不在评审里替做决定」。按当轮约定，把两个方案写入 §5 Q1 待用户裁决，
  仅完成无争议部分——实施文档不再把未实现机制记为已完成（判据 1 改部分完成）。
  该取舍已由用户 2026-11-16 裁决为方案 a，round02 落地（§6.1 round02-1）。
- 问题5 用 `WeakValueDictionary` 而非评审建议的 `WeakSet`：`ExecutionPlan` 是自动生成
  `__eq__` 的 dataclass（`__hash__` 为 None，实测 `TypeError: unhashable type`），
  无法入 WeakSet；按 `id` 键、计划本身作弱值的字典提供相同的「条目随对象回收清除」
  语义，且不改契约的可相等性。红测先在 set 实现上失败、修复后通过。
- 问题7 无红绿循环：两份折叠实现当前逐字等价，任何行为级测试在修复前后都通过，
  无法构造「先失败」的用例（除源码文本断言外，后者脆不可取）。守护改为构造性单
  实现（append 只调 fold_waits）+ 既有等价性测试（shadow ⊆ 用例、全部依赖与数值
  用例），并以本轮全量回归佐证重构无行为变化。
- 问题2 的「cat 未持有段被 pop」路径：本轮红测覆盖字面量混排（slice/cat 两类长度
  错位），未持有段在真实模型与评审探针中均未出现（评审亦定性为潜伏）；该路径与
  已测路径同一构造点（pop 时两组列表同步跳过 append），由对齐不变式测试与真实路径
  契约测试共同锁定。

round02 新增取舍说明：

- round02 问题1 按裁决方案 a 实施时的半开状态发现：只给 `execute_plan` 加旁路时，
  红测以「内核从输入缓冲未写入的行读到垃圾索引」（kernels_pim.py IndexError）失败——
  executor 去填充（`prefill_bucket` 早退、mask_len）与通信钳制仍按
  `dynamic_shape_enabled()` 生效，未填充输入配包络内核正是设计不变式 5 禁止的半开
  状态。修复因此把 `runtime_static_mode()` 并入 `dynamic_shape_enabled()`（单一真源
  shape_bind.py），三处门控统一失效、整体回退包络。这不是偏离裁决而是方案 a 的最小
  完整实现：裁决要求「= 静态路径执行且 token 一致」，token 一致只有在包络语义整体
  成立时才成立；`execute_plan` 入口旁路与裁决措辞一致，static 分支零改动。
- round02 问题2 修复顺带删除静态选桶循环尾的兜底 raise（防线提升后该分支不可达），
  属死代码清理而非行为变更；错误文本逐字保留。

round03 新增取舍说明：

- round03 问题2 选「删除」而非「设计文档登记」：评审给出二选一（删除字段与
  断言，或确认作为动态产物自描述保留并在设计文档分层表显式登记归属与首个
  预期消费者）。设计文档（2026-11-13 批准版）本就未登记该字段——保留并登记
  等于往基准设计里新增设计内容，超出本轮「不擅自改需求与设计文档」的权限；
  而删除是仓库纪律「删优于加/最小实现」与 round01-4 已接受先例的直接应用，
  且属本轮任务要求「前几轮同类问题的修复未覆盖根因的，视为本轮遗留一并
  补齐」的点名场景（round03 评审亦将其定性为 round01-4 同类残留）。信息与
  `prefill_seq_lens[-1]` 完全冗余，无行为影响（两测试文件全绿、全仓 grep 零
  残留）；若用户裁决保留，从未提交工作区恢复该字段为三行改动。
- round03 问题1 无传统红绿循环：问题本质是测试缺口而非功能缺陷（功能经评审
  探针实测正确），新增用例在当前树上直接通过。测试有效性以变异验证替代
  （把 T 绑定改成段末绝对位置 → 3 个分段形态与契约用例翻红、整段锚点与既有
  9 条动态整段用例仍绿，见 §4），证明用例专守「动态 × 分块」组合且非空洞；
  契约用例的首版断言（预期逐段收缩的 logits 行数）曾在真实契约（d2h 保持
  包络，8d62ca0 既有设计决策）上翻红，修正为锁定包络行数 + 段内行偏移。

### 6.3 回归测试结果

- 执行命令：`python -m pytest tests/ -q -k "not llama2_7b"`（裸环境，不设
  `PIM_DYN_SHAPE`，所有源码修改完成后后台启动，日志 /tmp/regress_round03fix.log）
- 执行结果：2276 passed / 11 skipped / 0 failed / 52 deselected（1648s）。
  2276 = round03 评审基线 2270 + 本轮新增 6 条用例（分块矩阵 4 + 分块契约 1 +
  动态 wave 1），0 failed；两条 RuntimeWarning 为 test_architectures 既有 fp16
  溢出告警，与本轮改动无关。本轮源码改动仅删 `dynamic_axes`（写而不读字段，
  裸环境静态默认路径不经过），静态默认路径行为不变即本红线证据。
- 历轮趋势（同口径 0 failed）：round01 前 2260 → round01 后 2266（+6）→
  round02 后 2270（+4）→ round03 后 2276（+6），skipped 始终 11。
