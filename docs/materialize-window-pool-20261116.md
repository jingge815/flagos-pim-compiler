# 窗口池与运行期物化（真动态图 M1）— 2026-11-16

## 改了什么

动态图路径（`dynamic=True`）引入「窗口池 + 运行期物化」：激活区/块表按**池深**规划地址
（默认 8），编译产物只留**窗口 0 模板**，其余窗口由执行器在运行期按需物化。并发宽度
默认 `min(4, 池深, num_request_slots)`。静态路径完全不走这套，行为逐字节不变。

```
编译期（动态）                      运行期（每一步）
─────────────────────              ─────────────────────────────
fit_activation_windows(池深)        _window_plan(compiled, phase, w)
  激活区 = 池深 × act_stride   →     template 存在？→ materialize(t, w)
_windows 只产窗口 0                 ├─ w=0：零拷贝返回模板计划
build_window_template：            ├─ 命中缓存：直接返回
  计划 + 激活区访问 RegionRef 表     └─ RegionRef.resolve({"W": w}) → Access
```

## 增删修改

| 文件 | 改动 |
| --- | --- |
| `runtime/materialize.py` | **新建**（约 110 行）：`WindowSpec`（池几何）、`AccessRef`/`RegionRef` 符号化访问、`WindowTemplate`、`build_window_template`、`materialize`。地址解析消费 `contracts/runtime_graph.py` 预留的 `ShapeExpr/RegionRef` 契约（此前全仓零使用）；payload 平移复用 `binding._bind_payload`，两条机制不漂移 |
| `runtime/compile.py` | 新参 `act_pool_slots`（静态路径传入直接抛错）；窗口池深 `window_pool` 与并发宽度 `parallel_slots` 分离（静态两者同一个数）；自动定容循环改为收池深（动态容量由池深决定，宽度不占额外激活区）；`_windows` 动态分支只产窗口 0；`CompiledModel` 增 `prefill/decode_window_template` 字段 |
| `runtime/executor.py` | `_prefill_plan`/`_window_plan` 动态分支先查模板、调 `materialize`；静态分支原样 |
| `backend/vllm_worker.py` | `PIM_DYN_SHAPE=1` 走动态编译；`PIM_ACT_POOL_SLOTS` 覆盖池深；`PIM_PARALLEL_SLOTS` 语义不变（宽度覆盖，编译后自动回读 `compiled.num_parallel_slots`） |
| `tests/test_runtime_materialize.py` | **新建** 25 条：物化==bind_plan 等价锚点（prefill/decode × 窗口 0..7）、零拷贝/缓存、越池与越界抛错、步长破坏抛错、静态路径不变式、`PIM_ACT_POOL_SLOTS` 覆盖、4 变长 prompt 动态 wave==逐个执行 |

## 关键决定

- **池深 ≠ 并发宽度**：地址空间是编译期资源（池深 8），并发是运行期决策（宽度 4）。
  批次超过宽度仍走 wave 串行，但编译产物不再封顶宽度——这是 M3 混批调度的前提。
- **等价锚点**：`materialize(t,w)` 与 `bind_plan(p, w)` 全窗口逐字节相等（payload、
  reads/writes、waits、任务片数全比），保证动态窗口和静态窗口是同一份语义。
- **fail-fast**：窗口号越池、步长破坏导致区间越池，一律 ValueError，不回退包络。
- 内核缓存键（`mram_offset`）不受窗口号影响：物化只动 reads/writes 偏移和
  `act_shift/block_slot` 附加字段，延续 D2 纪律。

## 已实施的测试

- 新增 25 条全绿；受影响面（动态形状/多请求/并行请求/绑定/分块/调度器）70 条全绿；
  全量回归 `-k "not llama2_7b"` 通过。

## 当前不足

- 4L 真实权重的 4 请求 probe 尚未跑（本轮只验到 tiny 4 请求）；CEVA 轨的窗口变体
  （P2 探路：mram_offset 是否烧死）未做。
- `serve_pim.py` 仍是静态编译入口，动态模式暂只走 `compile_llama2(dynamic=True)` 与
  vllm_worker 的 `PIM_DYN_SHAPE` 开关。
- 池深自动收缩只保证「装得下」，不保证达到 8；48L 权重近满 MRAM，池深按达成值登记。
