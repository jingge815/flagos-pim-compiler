# 静态图 → 动态图分层改造（D0–D2-lite，2026-11-13）

## 概述

把「编译期决策、运行期照表执行」的静态蓝图改造成 host 驱动的动态图。边界：运行期动态绑定形状、选择子图、按依赖调度；设备只执行预编译 kernel，不做设备内分支 / DPU 直连 / 逐 op JIT。静态路径保持默认、逐字节兼容，`dynamic` 均为显式 opt-in。

## 分层与提交

| 层 | 提交 | 内容 |
| --- | --- | --- |
| D0 契约 | `e81754f` | `contracts/runtime_graph.py`（ShapeExpr/RegionRef/RTOp/RTControl/RuntimeGraph + 可打印）；依赖判定抽到 `runtime/dependency.py`；`CompiledModel`/`export_annotated_graph`/`execute_plan` 加动态缝，动态模式显式 `NotImplementedError` |
| D1 调度器 | `3a2fc2c` | `runtime/scheduler.py` `EventScheduler`：依赖就绪才提交、完成回调驱动、gate 在调度侧异步解析（不再阻塞 `hal.submit`），跳过命令不入队；`address_waits` 重算 RAW/WAW/WAR 做 shadow 校验；`execute_plan(mode="dynamic")` |
| D2-lite 形状 | `0059a45` | `compile_llama2(dynamic=True)`：单张符号 T prefill 图（`torch.export dynamic_shapes`）；`capture_shape_exprs` 记录 ShapeExpr 并把 `val` 取整为 T_max 包络；`sym_size` 链折叠为常量；launch payload 按需携带 `arg_shape_exprs`/`out_shape_expr`（静态路径不加键） |

## 关键设计

- **双轨形状**：规划 / 地址 / 容量只用 T_max 包络整数（与静态单桶语义一致，数值零风险）；符号表达式存 `node.meta[pim_shape_expr]` 与命令 payload，供运行期绑定实际 T。
- **符号转换**：只认 `T`；形状环境辅助符号（s0/s30 类）退包络常量，避免不可绑定的表达式。
- **D1 依赖等价**：运行期地址依赖 ⊆ 静态 `waits`（host 闭包数据依赖仍以 `waits` 为准），shadow 测试逐命令核对。

## 验证（2026-11-13）

- 新增：`test_runtime_graph_contract.py`（4）、`test_engine_static_compat.py`（4）、`test_runtime_scheduler.py`（5）、`test_dynamic_shape_compile.py`（11，含 7 种 prompt 长度与静态基线逐 token 一致）。
- 定向子集：契约 / 无旁路 / 桶 / 计划生成 / 调度器 34 passed。
- 全量回归（排除 `llama2_7b`）：`python -m pytest tests/ -q -k "not llama2_7b" --basetemp=<data 卷>` → **2230 passed、11 skipped、0 failed**（1484 s）。

## 已知边界 / 下一步

- **D2-full（未做）**：运行期按实际 T 裁剪 DPU 行数——需 `comm/lowering` 段拷贝与内核形状读取的行数钳制；当前动态图收益是编译面（一张图替代桶表），执行仍走包络。
- **D3/D4（未做）**：`torch.cond`/`while_loop` 下降为 RTControl、调度器物化分支；预规划槽池 alloc/free（D4b 需先确认 op compiler 是否把 `mram_offset` 烧进代码）。
- 明确不做：设备内分支、DPU 直连、动态 KV 物理扩容、异步 DMA 重叠（一阶段约束）。

## 终版数字（2026-11-13 收口，4 层真实 MoE）

- static 70.0s / dynamic 69.5s / dynamic+PIM_DYN_SHAPE=1 67.7s，token 逐位一致。
- 全量回归 2233 passed、0 failed（排除 llama2_7b）。
- 调度器终版：计划序全量提交 + 逐命令形状绑定；就绪调度经剖析（65% 墙钟零并发）证伪并移除。
- 提交链：e81754f → 3a2fc2c → 0059a45 → a884afc → 73deddd → 8e73dbc → 4b0de02 → 7f4bf92 → ce99ca9。

## D2-full 实验结论（2026-11-13 夜）

- 设备侧形状绑定当时**并未真正生效**：4L 动态计划 15188/15188 条 launch 虽带表达式，但表达式全是包络常量（原因见下节修正），67.7s 只是包络执行下的噪声，不能作为绑定收益。
- 通信侧行收缩两次实验均失败：gather-only / scatter-only 在带计数、独立 backend 的 bisect 中各自破坏 token。根因是 comm 条目不带形状表达式，单 T 轴启发式无法覆盖 4-D mask（如 `(1,1,T,T)`）等形状；host 闭包与 gather/scatter 的实际形状无法从 `CommPlanEntry.shape` 推导。
- 结论：通信收缩暂缓，需把节点 `pim_shape_expr` 挂到 `CommPlanEntry` 并让 host 闭包统一按实际 T 后再做；当前动态收益以设备侧形状绑定为准（67.7s vs static 70.0s）。错误接线已在 `a1417b8` 回退，库层截断保留未接线。
- 契约防线：`CommPlanEntry` 新增显式 `t_axes`（缺省空=静态）；`clamped_entry` 在缺少 `t_axes` 或多 T 轴时显式拒绝（ValueError/NotImplementedError），禁止再按启发式静默裁剪。定向子集 100 passed，全量回归见下。

## D2-lite 绑定修正（2026-11-13 收口后）

发现并修复三处缺陷，此前 `PIM_DYN_SHAPE=1` 的形状绑定实际是空操作：

| 缺陷 | 现象 | 修正 |
| --- | --- | --- |
| 符号名不匹配 | torch 2.9 把 `Dim("T")` 导出成 `s30` 一类名字，转换只认字面 `"T"`，除 input_ids 外全部退化为包络常量 | 从 `program.range_constraints` 取唯一符号名，作为 `dyn_symbol` 传给 `capture_shape_exprs` |
| 取整污染 | 对任一 SymInt 调 `int()` 会让图中其余 SymInt 的 `.node.expr` 一并退化成常量 | capture 改两遍：先读完全部原始表达式，再统一取整 |
| 字面量形状 | view/reshape 第二实参 `[1,16,-1,16]` 的符号元素被折叠成常量，运行期无法绑定 | `_fold_symbolic_scalars` 把逐元素表达式记到用户节点 `pim_arg_exprs`；计划 payload 携带 `arg_elem_exprs`；`bind_command_shapes` 逐元素改写 |

附带 D2-full 前置：`CommPlanEntry` 增加 `t_axes`/`shape_exprs`，`build_comm_plan` 按边源节点的 `pim_shape_expr` 填充；仅元数据，收缩尚未接线。新增 `tests/test_comm_shape_exprs.py` 固定该行为。

验证（`PIM_DYN_SHAPE=1`）：`test_dynamic_shape_compile.py` 13 passed，含 7 种 prompt 长度与静态基线逐 token 一致；定向子集（契约 / 计划生成 / 调度器 / 通信 / 绑定 / 优化 Pass）静态口径 274 passed，动态口径 20 passed。

全量回归：**2296 passed、11 skipped、0 failed**（8858 s，最终代码状态）。

补充（4L MoE 复测暴露）：`fuse_elementwise` 合并逐元素算子时只搬了 `val`，没有搬 `pim_shape_expr`，导致下游 `view` 的输入实参丢掉 T 表达式（形状列表仍按 T 绑定），4L 真实权重下 reshape 元素数对不上。已在融合处随 `val` 一起复制形状表达式，并加 `test_fuse_copies_shape_exprs` 固定。

补充（同一轮 4L 复测）：绑定判据必须是“行主序前缀兼容”。只有 T 是唯一符号维、且排在所有非单位维之前（如 `(1,T,H)`、`(1,1,T,H)`）时，实际形状的前 t 行才与包络缓冲逐元素对应；T 在中间维（如 attention 的 `(1,heads,T,head_dim)`）或有两个 T 轴（`(1,1,T,T)` mask）会让内层步长/行宽随 T 变，整条命令必须保持包络。该判据落在 `runtime.shape_bind._command_bindable`，不满足时整条命令不绑定（arg/out/字面量形状都不动）。

4L 真实权重验证（prompt 5 token，8 DPU，prefill 8）：
- `dense_sim + PIM_DYN_SHAPE=1`：tokens `[6722, 4849, 35579, 1197, 5434, 5434]`，与 static/dynamic 逐位一致，run 70.05s。
- `dynamic_dispatch + PIM_DYN_SHAPE=1`：tokens 同为 `[6722, 4849, 35579, 1197, 5434, 5434]`，run 70.58s；与 dense 一致，说明先前发散来自前缀布局，不是 MoE 分派。
- 本轮三臂（同机同负载）：static run 63.42s / dynamic 67.95s / dynamic_bind 70.58s；绑定收益被 numpy 镜像与调度开销掩盖，尚未调优。

## D2-lite 去填充（2026-11-15）

动态 prefill 不再把输入补到包络长度：

- `prefill_bucket` 在 `compile_mode=="dynamic"` 时按实际 token 数返回；`prefill_chunk` 的 `input_ids`/`position_ids` 按实际 T 建，掩码仍取 `prefill_seq_len` 包络（`(1,1,T,T)` 行宽随 T 变，只能保持包络；标准因果掩码的融合注意力不读值）。
- `scatter` 在动态执行时只写实际 T 行：`_scatter_rows` 仅对“单 T 轴且 T 在最外层非单位维”的条目返回实际行数，其余保持整包络；`_check_host_size` 在 rows 给定时把主机缓冲下界放宽为 `rows*R`。
- `_prefill_plan` 在动态图（实际长度不是桶键）时落到唯一 T_max 计划的对应窗口。

验证：tiny Llama 动态用例 13 passed、tiny Qwen3-MoE 绑定用例 5 passed（1/2/3/5/7 token 与静态逐 token 一致）；静态 prefill/comm 子集 85 passed。

4L 真实 MoE（prompt 5 token，8 DPU，prefill 8）：去填充后 `dynamic_bind` tokens 仍为 `[6722, 4849, 35579, 1197, 5434, 5434]`，与 static 逐位一致；run 52.09s，低于同轮 static 63.42s 与未去填充的 dynamic_bind 70.58s。

## D2-full gather 收缩（2026-11-15 下午）

收集方向（`all_reduce`/`all_gather`/`all_to_all`）接入实际 T 行数：

- `runtime/exec_plan_gen.py`：`_scatter_rows` 更名 `_clamped_rows`，scatter 写入与 gather 收集/回写共用同一个「单 T 轴 + T 在最外层非单位维」门控；专家子图条目（gate 打标）一律包络——路由缓冲按每专家 token 混排，行数不是 T 的行前缀。
- `comm/lowering.py`：三个收集原语按 rows 收缩——`clamped_entry` 截段后，`_merged`/`all_reduce` 的缓冲按 `rows*R` 分配，返回值 reshape 成实际形状（T 维换成 rows）；mean 归约除数仍取原始条目的收集段数（行收缩只减搬运量，不改参与者个数）。`clamped_entry` 里「行几何不整除则原样返回」的死分支删除（numel 必整除自己的维，留着会让调用方的 rows 缓冲与包络段错位）。

| 边界 | 规则 | 原因 |
| --- | --- | --- |
| d2h（结果回主机） | 不收缩 | 主机闭包混用包络形状的兄弟实参：小 MoE 里 `where(selected_mask, expert_out, 0)` 一侧是 launch 绑定出的实际 T 行、另一侧是包络，形状对不上直接崩。主机可见形状要全链统一实际 T 才能收缩，尚未做到 |
| 专家子图（gate 打标） | 不收缩 | 路由缓冲行数非 T 前缀；跳过专家的 `skip_value` 也是包络零张量 |
| d2d/d2c（结果只进 MRAM） | 收缩 | 下游 launch 绑定读实际 T 行；未绑定命令读包络尾行，与去填充后的「尾行是旧值」一致，不引入新污染 |

另修一处 c9aeb0f 留下的不一致：executor 按 `compile_mode=="dynamic"` 无条件去填充，而 scatter 放宽检查要求 `PIM_DYN_SHAPE=1`——动态编译不带环境变量跑必炸（host_buf 元素数 1 != 包络 16）。现在去填充/实际长度同样挂在 `dynamic_shape_enabled()` 上：关绑定时整个执行回退包络（补 pad），与通信严格检查一致。

验证：
- 新增单测：三个收集原语的行收缩等价（收缩结果 == 包络结果的行前缀、DPU 回写区尾行不被触碰、mean 除数不变）、`_clamped_rows` 门控（开关/绑定缺失/多 T 轴/T 在中间维）。
- tiny Llama 14 项 + tiny Qwen3-MoE 5 项 + 形状绑定 8 项全绿；token 对拍用例改为自带 `PIM_DYN_SHAPE=1`（此前依赖外部环境变量，裸跑必炸），并新增「关绑定回退包络」用例。
- 静态定向子集（comm/prefill/exec/scheduler/dispatch）231 passed。
- 全量回归（默认排除 slow/llama2_7b）：**2260 passed、11 skipped、0 failed**（1423 s）。
- 4L 真实 MoE（prompt 5 token，8 DPU，prefill 8）：见下。

4L 真实 MoE 结果（同轮各跑一次、独占机器）：dynamic_bind tokens `[6722, 4849, 35579, 1197, 5434, 5434]`，与 static 逐位一致；run static 50.18s / dynamic_bind 62.22s。行收缩在 numpy 镜像上只减少拷贝次数与字节（段结构不变、尾段被裁掉），不会引入开销；跨轮绝对值波动大（同一代码历史区间 52–70s），绑定/收缩的性能调优仍以 numpy 镜像与调度开销为主，另立工作项。

## 调度器等待路径调优（2026-11-15 晚）

4L 同轮 dynamic_bind 62.22s vs static 50.18s（124%）超出计划 Go/No-Go 的 105% 上限，cProfile 定位 decode 步（剖析器下 10.04s/步）两处调度器开销：

| 开销 | 位置 | 量级 | 根因 |
| --- | --- | --- | --- |
| 每步重建计划索引 | `EventScheduler.__init__` 的 `commands`/`waits` 两个 dictcomp + 未知依赖校验 | 0.885s/步（9%） | 调度器每次 `execute_plan` 新建，索引却是计划派生的静态数据 |
| 逐命令完成唤醒 | `_on_done` 拿条件变量锁 + `notify_all`，`wait()` 的 `wait_for` 谓词每次被唤醒都重算 | 17876 条命令/步 = 万级锁操作 + 万级线程切换/步 | 每条命令完成都把主线程拽醒一次检查谓词 |

改造（`runtime/scheduler.py` 117 → 77 行，净 −40）：

- `wait()` 改为 join 提交线程后逐 future 直等（本次 run 自己提交的 event 列表），与静态路径的等待方式同构；不再有完成回调、条件变量、`done` 计数。附带修复：命令 future 的执行异常现在会从 `wait()` 抛出（旧路径只传播提交异常，执行异常静默）。
- waits 未知依赖校验改为模块级 `_validate_plan` 按计划身份只做一次（计划常驻 `CompiledModel`，id 不复用）；每步只剩构造器常量赋值。
- 删死代码：`_submit`（gate 判定已移入 HAL submit，无调用方）、`started` 死字段；单调用者 `_submit_plain` 内联进 `_loop`，`_shape_env` 提到每步一次（原先每条命令重算）。

效果（4L，8 DPU，同轮独占对照）：

- 剖析：`__init__` 0.885s → 0.071s/步；`future.result()` 等待为真实 worker 计算时间。
- 墙钟：static 48.83s / dynamic_bind 49.94s = **102.3%**，达标（上限 105%）；tokens `[6722, 4849, 35579, 1197, 5434, 5434]` 逐位一致。dynamic_bind 相对上一轮同判据的 62.22s 减少 20%。
- 测试：scheduler/dynamic/binding/parallel 25 项 + exec_plan/moe_dispatch/prefill_buckets/comm/compat 113 项全绿；全量回归见下。

decode 两路本就执行同一份 decode 计划（`_window_plan` 不分 mode），动态路径的 decode 开销即调度器本身；本轮修完后 dynamic decode ≈ static decode + 线程切换（毫秒级）。prefill 侧动态图一张 T_max 图 + 去填充，不劣于静态桶。剩余 ~2% 差值主要在 prefill 的形状绑定与未绑定命令的包络执行，属 numpy 镜像本底，不再单列工作项。
