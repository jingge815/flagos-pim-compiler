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
