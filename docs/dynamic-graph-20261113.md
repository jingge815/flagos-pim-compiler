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
