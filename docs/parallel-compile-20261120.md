# 算子编译并行化与冗余清理（2026-11-20）

> 追记一（同日，genesim 侧三处缓存优化）：仿真完整轮（tp2_pp4，10 请求）
> 30 分 37 秒 → **14 分 31 秒（2.11×）**，`Total time: 5551.468564 s` 与改前
> 逐位一致。三处：`config_loader.get` 记忆化、`_json_identity` 内容指纹、
> `_SHARED_PHASE_METRICS` FIFO 淘汰。教训：估热点占比一律 py-spy 原生采样，
> cProfile 对百万级调用计数既放大（config.get）又漏报（C 编码器里的 JSON）。
>
> 追记二（同日下午，测试修复轮）：genesim `./run.sh --test` 四个文件挂：
> ① `config.get` 记忆化只在 `__init__` 建属性，单测 `__new__` 直造实例炸
> → 改惰性 `_reader_cache()`；② `_json_identity` 按对象身份的缓存丢掉了
> 周期表原地改值契约（`test_vpu` 有测试钉着）→ 改精确分流指纹（平坦标量
> 表 `tuple(sorted(items))`、嵌套退 `json.dumps`），比 id 缓存慢 ~7%，契约
> 保住；③ `_SHARED_PHASE_METRICS` 跨用例泄漏 → 测试 setUp 清空；④
> `_resolve_pimir_path` 同样吃 `__new__` 绕过 → 惰性初始化；⑤
> `test_collective_edges` 配置与"用满校验"冲突 → 改非正式核数（2 核）。
> 修复后 `tests/sim` 49 个文件全绿；tp16_pp4 单请求 `Total time` 123.670581 s
> 逐位一致。
>
> 追记三：genesim 已切标准形态 **tp16_pp4（64 DPU 用满 64 核）**，加了
> 用满校验（tp2_pp4 会被拒）；本仓 `run_full_pipeline.py` 的 `--num-dpus`
> 缺省同步改为从 CEVA 画像取 64。

## 背景与动机

`scripts/run_full_pipeline.py --num-stages 4` 一轮约 31 分钟，实测分段耗时
（2026-11-20，缓存热）：

| 步骤 | 内容 | 耗时 | 占比 |
| ---- | ---- | ---- | ---- |
| [A]  | model_parser 图骨架 | 5 s | <1% |
| [0]  | PU 映射方案 | 4 s | <1% |
| [B+C+D] | 图编译 + 算子编译 + sidecar | 32 s | ~2% |
| [A2] | 精化 pimir 成本 IR | 14 s | <1% |
| [E]  | GeneSim 事件仿真 | 30.5 min | **~97%** |

结论：编译侧已不是瓶颈（缓存热时全套不到 1 分钟）；真正的耗时在仿真器
（单线程事件仿真，见文末"当前不足"）。并行化编译的价值在缓存冷、
`--force-compile` 严格档与调优扫描（`tune_pipeline.py --replay` 每个候选
全量重编）这几条路径上。

## 改动内容

### 1. 并行执行工具（新增）

`opcompiler_bridge/parallel.py`：

- `job_workers()`：读环境变量 `OPCOMPILER_JOBS`，默认 32，非法值退回默认，
  最小夹到 1。
- `run_jobs(fn, items)`：线程池按提交顺序返回结果（`pool.map` 语义，聚合
  逻辑与串行版一致）；任务数 ≤1 或并行度为 1 时直接串行不开池。

**为什么线程池安全（"彼此没有影响"）**：被并行化的只有子进程型编译
（`triton-opt` / `gcc`），线程只负责等待；每个编译键写各自的内容寻址缓存
文件，`compile_op` 对同 key 还有 `flock` 文件锁（driver 原有设计）。
唯一键先去重再派发，同 key 不会并发编译。GPU 捕获路径（FlagGems 实跑 +
全局补丁）不进池，保持串行。

### 2. 三处串行编译循环改为"收集 → 并行 → 回填"

| 位置 | 循环内容 | 唯一键数（llama2-7B tp2_pp4） |
| ---- | ---- | ---- |
| `runtime/compile.py::peak_kernel_mram_bytes` | 每个唯一 (x形状, w形状, dtype, 激活, 切分) 一次 compile_op 探测 | ~4-8 |
| `genesim_bridge/placement_export.py` GEMM 段 | 每个唯一 (本地形状, 激活) 一次 `_measure_kernel_tile_n` | ~4-8 |
| `genesim_bridge/placement_export.py::_attach_bpath_pimir` | 每个唯一 (mnemonic, 形状) 一次 `_bpath_pimir` | 17 |
| `genesim_bridge/cost_extractor.py::export_costs_to_genesim` | 每个唯一算子键 × 两个形状点（prefill/decode） | 21 × 2 |

等价性保障：

- 回填按提交顺序进行，`widths.setdefault` 等"首见优先"聚合与串行版一致；
- sidecar/IR 产物逐字节等价（实测 diff，见"验证"）；
- pimir 档的非 GEMM 键留在串行侧（`_measure_unique_keys` 的安全分类），
  有测试钉住。

### 3. 冗余清理

- `_get_attr_node`：原来每个 (层, 投影) 现扫全图（224 次全图遍历），改为
  先收一遍 `get_attr` 节点再在其中查找。
- `_attach_bpath_pimir`：同一条 pimir 被上千个算子引用时 sha256 逐个重读
  文件，改为按路径去重；每算子的编译键 json.dumps 两遍改为一遍。
- `_bpath_pimir`：删除函数内两段重复的 driver 导入。
- （此前已在工作区的未提交改动）`_source_name` 查 `SOURCE_NAMES` 表，
  不再为 3000+ 个桥接算子逐个建配方。

### 4. 入口参数

`scripts/run_full_pipeline.py --jobs N`（默认 32）：校验 ≥1 后写入
`OPCOMPILER_JOBS`，随环境传给导出/精化子进程。

## 关键结构

```python
run_jobs(fn, items)          # 线程池 map，按提交顺序返回
job_workers() -> int         # OPCOMPILER_JOBS，默认 32
```

调用链：

```
run_full_pipeline.py --jobs 32
  └─ OPCOMPILER_JOBS=32（环境变量）
       ├─ export_pp_placement.py
       │    ├─ compile_llama2 → peak_kernel_mram_bytes（并行探测）
       │    └─ export_placement_to_genesim
       │         ├─ GEMM 唯一形状并行测量 → 回填 kernel_tile_n/pimir_path
       │         └─ _attach_bpath_pimir：唯一键并行编译 → 串行回填
       └─ refine_ir_with_flagtree.py（genesim 侧薄入口）
            └─ export_costs_to_genesim → _measure_unique_keys
                 ├─ 并行：oplevel 全部、pimir 的 GEMM（子进程型）
                 └─ 串行：pimir/ttir 非 GEMM（GPU 捕获）
```

## 已实施的验证

- `tests/test_parallel_jobs.py`（新增 5 条）：并行度解析（默认/环境/非法/0）、
  提交顺序保持、空/单元素输入、并行档与串行档产物逐字节一致 + 每键只测一次、
  pimir 档非 GEMM 键不进线程池。
- `tests/test_placement_export.py`、`test_exec_plan_gen.py`、
  `test_layout_feedback.py`、`test_genesim_bridge.py`、`test_pimir_layout.py`、
  `test_ir_cost_eval.py` 全部通过。
- 真实数据等价性：改动前后各跑一轮 `--skip-simulation`，`placed.ir`、
  `pimir.ir`、refine sidecar 逐字节相同；placement sidecar 仅
  `compiler.toolchain.source_sha256` 一处不同（本仓源码摘要，理应变化）。
- 冷缓存 A/B（`OPCOMPILER_CACHE_DIR` 指向空目录）：jobs=1 与 jobs=32 的
  sidecar 在规范化缓存目录前缀后完全一致；两者产物各 78 个缓存文件。

## 当前不足

1. 仿真段经三处缓存优化后为 ~14.5 分钟（原 30.5），占完整轮比重仍最高。
   剩余热点的原生采样（py-spy speedscope，`/tmp/sim_native.json` 已随
   /tmp 清理，结论记录于此）：`dataclasses.replace` 合计 ~6%（每次算子
   分发复制 Operator，动它涉及调度器语义）、`_execute_pipeline` 事件簿记
   ~2%。要再快需算法层改造（事件批处理/向量化），缓存类手段已用尽。
2. `_json_identity` 按对象身份记忆化后，原地改周期表/模板不再换身份
   （全仓无此用法；配置重载产生新对象、自然换新身份）——契约边界写在
   函数 docstring 里。
3. 缓存热的标准流水线上，编译并行化对总时长无可测收益（编译子任务单个
   仅 ~0.15 s，冷缓存整体损失只有 ~5 s）；收益集中在 force-compile、
   调优扫描与工具链指纹变化后的全量重编。
4. `OPCOMPILER_JOBS` 只作用于编译子任务；GPU 机器上 pimir 档非 GEMM 测量
   仍串行（FlagGems 捕获不可并发，刻意保守）。
