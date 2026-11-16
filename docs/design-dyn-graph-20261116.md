# 设计：静态计算图 → host 驱动动态图（dyn-graph）

日期：2026-11-16（评审基准文档；设计于 2026-11-13 批准，来自用户批准的分层改造方案）

## 目标架构

```text
编译期（一次）                                    运行期（每请求/每步）
──────────────────────────────────────────────────────────────────────
torch.export(dynamic_shapes)          runtime/compile.py
  → partition/spec_prop/fuse           graph/*
  → RuntimeGraph（符号形状+控制流）     contracts/runtime_graph.py  [新增]
  → DPUPlan（T_max 包络+slot 表）       memory/mem_planner.py
                                             │
                                   EventScheduler（运行期依赖/提交） runtime/scheduler.py [新增]
                                             │
                                   NumpyBackend.submit(Command, ctx)  backend/hal_numpy.py
```

新增层：`contracts/runtime_graph.py`（ShapeExpr/RegionRef/RTOp/RTControl/RuntimeGraph）、
`runtime/dependency.py`（区间依赖，从 exec_plan_gen 提取）、`runtime/scheduler.py`
（EventScheduler）、`runtime/shape_bind.py`（符号 T 求值与命令绑定）。复用 `Command`/
`Access`/`ExecutionPlan` 作为低层命令载体；gate 跳过语义保留。

## 关键设计不变式（评审以此为准）

1. **双轨制**：规划/地址/容量只用 T_max 包络整数（`TensorShardDetail`/`Access`/`DPUPlan`
   不变）；符号表达式存 `node.meta[pim_shape_expr]` 与命令 payload（`arg_shape_exprs`/
   `out_shape_expr`/`arg_elem_exprs`），供运行期按实际 T 绑定。运行期绑定不得改变
   MRAM offset、内核缓存键（`payload["mram_offset"]` 永不平移）、KV 几何。
2. **只符号化 T**：batch 固定 1；hidden/heads/head_dim/FFN/vocab/专家维保持静态。符号
   转换只认 `T`（从 `program.range_constraints` 取真实符号名）；形状环境辅助符号退包络
   常量。绑定判据是**行主序前缀兼容**：T 是唯一符号维且排在全部非单位维之前
   （`runtime/shape_bind._command_bindable`）；不满足则整条命令保持包络，不得猜测。
3. **调度语义**：计划序全量提交（就绪驱动经剖析证伪：保序约束下无收益且 65% 墙钟零
   并发）；同 `(lane_key, dpu)` 的静态 waits 折叠依赖「提交序=计划序」FIFO；gate 由 HAL
   在 submit 内同步解析，跳过命令产生预完成 event（submit 内私有 `_skipped_event`）。
   完成等待为 join 提交线程 + 逐 future 直等；提交异常与命令执行异常都从 `wait()` 抛出。
4. **通信行收缩边界**（D2-full）：收集方向（all_reduce/all_gather/all_to_all）与 scatter
   写入按实际 T 行钳制，仅限「dst 是 DPU 且非专家条目」。**d2h（结果回主机）一律不收缩**：
   主机闭包会混用包络形状的兄弟实参；**专家子图条目（gate 打标）一律包络**：路由缓冲行数
   非 T 前缀。`CommPlanEntry.t_axes` 显式声明；缺 t_axes 抛 ValueError、多 T 轴抛
   NotImplementedError，禁止启发式静默裁剪。mean 归约除数取原始条目收集段数。
5. **开关与回退**：`PIM_DYN_SHAPE=1` 是运行期绑定的唯一开关；关闭时动态编译整体回退
   包络执行（补 pad、包络通信）——executor 去填充、scatter 放宽、通信钳制全部挂
   `dynamic_shape_enabled()`，不允许半开状态。
6. **契约真源**：改 `contracts/` 字段、payload 键、`node.meta` 键必须 grep 调用方并同步
   全链路（含 `genesim/scripts/` 若涉对外接口）。

## 分层路线与状态

| 层 | 内容 | 状态 |
| --- | --- | --- |
| D0 契约 | `RuntimeGraph`/`ShapeExpr` 契约 + 静态兼容 + 摘要守护 | 已实施（e81754f） |
| D1 调度器 | EventScheduler、gate 非阻塞、地址依赖 shadow | 已实施（3a2fc2c，后经剖析收敛为计划序全量提交 7f4bf92，等待调优 7fcf056） |
| D2-lite | 符号 T 单图 prefill、形状表达式捕获、运行期绑定 | 已实施（0059a45、8e73dbc、a411403） |
| D2-full | 通信行收缩 + prefill 去填充 | 已实施（e03a22b→23049f7 回退→a1417b8 库层保留→c9aeb0f、8d62ca0 接线） |
| D3/D4 | 控制流、slot arena | 契约预留，不实施 |

## 风险与对策

- 调度器开销：Python 调度 + host lock 可能拖慢单请求；以 105% 上限验收（已达成 102.3%）。
- torch.export 动态形状：只符号化 T；SymInt 取整污染用两遍 capture 规避。
- kernel 缓存爆炸：D2 保持 offset 稳定，缓存键不动；D4b 前不拆地址键。
- 兼容：CEVA/GeneSim 的静态导出旁路保留；动态路径不替换 CEVA 产物。
