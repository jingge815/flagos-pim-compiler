# B 路 PIMIR 携带 DPU/tasklet 参数

> 文档编号：tune-bpath-tasklet-dpu-20261113
> 创建日期：2026-11-13
> 关联模块：`genesim_bridge/placement_export.py`、`opcompiler_bridge/`、`genesim/`

## 一、目标

调优产物中的 B 路 PIMIR 也必须带上本候选的 `pim.num-dpus` 与 `pim.num-tasklets`。此前 B 路由 `_module()` 拼模块头，只写 `pim.target`，多 DPU 的 DPU/tasklet 参数没有进入 B 路产物，审计时无法证明 B 路算子按候选参数编译。

## 二、之前的问题

| 路径 | 生成方式 | 模块属性 |
| --- | --- | --- |
| A 路 GEMM | `_make_ttir` → `_run_triton_opt` → `a_path_passes` | `pim.num-dpus`、`pim.num-tasklets`、MRAM/WRAM/DMA、target |
| B 路算子 | `_bpath_pimir` → `_module()` + `_run_oplevel_triton_opt` | 只有 `pim.target`、MRAM/WRAM/DMA |

B 路 sidecar 条目的 `shards` 为空，表示算子按复制档进入每个 DPU；但 PIMIR 没有写出复制档与 DPU/tasklet，改 tasklet 时 B 路文本完全不变。

## 三、实现

`_bpath_pimir()` 不再直接 `_module(body)`，而是构造一个复制档请求并复用 A/B 共用的 `_module_text()`：

```python
request = OpCompileRequest(
    op=mnemonic,
    arg_shapes=[tuple(shape) for shape in ins],
    hardware=hw,
    shard=DpuShard(dim=-1, num_dpus=hw.num_dpus, kind="replicate"),
)
text = _module_text(body, request)
```

`_module_text()` 在多 DPU 时写入：

- `pim.num-dpus`
- `pim.num-tasklets`
- `pim.dma-align`
- `pim.placement = #pim.placement<kind = replicate, numDpus = N>`

目标改从 `hardware.compiler_target` 取，不再用 `op_classify` 里固定的 `pim:v1`。单 DPU（`num_dpus=1`）不写这些属性，文本与改动前保持一致。

`shard.kind` 用 `replicate` 而不是伪造 `shard`：B 路算子本来就不跨 DPU 切分，复制档是真实语义；伪造 shard 会让编译器以为张量被切开，产出错误编码。

## 四、为什么这样就能被审计

- `genesim_bridge/refine_profile.py` 与 `genesim/src/run_manifest.py` 已有 B 路可选属性核对：B PIMIR 写了 `pim.num-dpus`/`pim.num-tasklets` 时，必须与 `compiler.hardware` 完全一致；不写时仍按旧口径只核对 MRAM/WRAM/DMA/target。
- 现在新导出的 B PIMIR 会写出这两项，所以调优侧、精化侧和 GeneSim 启动侧都能证明 B 路吃的是同一套 DPU/tasklet。
- A 路契约不变；单 DPU 旧产物不变。

## 五、验证

| 测试 | 结果 |
| --- | --- |
| `tests/test_tune_compile_bridge.py` | 24 passed；新增 B 路 tasklet=4/8 文本必须不同、单 DPU 不写属性 |
| `tests/test_placement_export.py` | 64 passed；B 路条目现在必须写 DPU/tasklet 与复制档 |
| `gensim/tests/sim/test_refine_profile.py` | 48 passed；B 路可选 DPU/tasklet 与请求一致 |
| 真实 B 路编译 | `eltwise` 在 8 DPU/tasklet=4 下产出 `pim.num-dpus`、`pim.num-tasklets`、`#pim.placement<kind = replicate, numDpus = 8>`；tasklet=4 与 tasklet=8 的文本和 SHA 不同 |

真实产物示例：

```mlir
module attributes {"pim.dma-align" = 64 : i32, "pim.mram-bytes" = 8000000000 : i64,
"pim.num-dpus" = 8 : i32, "pim.num-tasklets" = 4 : i32,
pim.placement = #pim.placement<kind = replicate, numDpus = 8>,
pim.target = "pim:v1", "pim.wram-bytes" = 65536 : i64} {
```

## 六、tasklet 并行成本模型与代码降级

PIMIR 带参数只是第一步；本轮同时把“参数 → 代码 → 成本”接通。

### 6.1 FlagTree：B 路矩阵乘按 tasklet 切 M

`lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp` 的 B 路 `pim.matmul` 降级分为两条：

- 无 `tile` 属性：M 行循环按 `rowsPerTasklet = ceil(M / numTasklets)` 切成若干静态块；
- 有 `tile` 属性：按 M 方向 tile 数 `mTiles = M / tile_m` 切成若干静态块。

`numTasklets` 来自模块属性 `pim.num-tasklets`，缺失时为 1，因此旧 B PIMIR 的 C 文本不变。每条 tasklet 块只写自己的 M 行；`pim.matmul` 的输出块互不重叠。

### 6.2 GeneSim：矩阵算子按 tasklet 并行度缩放成本

- 编译 trace 时从 PIMIR 读 `pim.num-tasklets`，写进 `compile_signature` 与 trace 元数据；签名版本升到 13，旧 trace 自动失效重编。
- `PIMTrace.estimate_total_cycles()`（仿真执行走这里）与 `estimate_cycle_breakdown()`（效率指标走这里）对矩阵类算子（`GEMM`、`GEMV_SCORE`、`GEMV_CONTEXT`、`MATMUL`）按有效并行度缩放 cycles：
  - A 路 `GEMM` 按行切分，有效并行度 = `min(num_tasklets, M)`；
  - B 路矩阵乘按 tile 切分，有效并行度 = `min(num_tasklets, ceil(M / tile_m))`；
  - 向量/视图算子（ROPE、SOFTMAX、RMSNORM 等）不缩放。
- **只缩放计算周期，不缩放访存周期**：tasklet 并行做计算，DMA 是共享资源。`memory_bytes` 也不缩放。
- 结果里额外返回 `tasklet_parallelism`，方便审计每个算子实际用了多少并行度。

> 两个坑都在真实 quick 运行里暴露过：第一版只改 `estimate_cycle_breakdown`，
> 仿真执行走 `estimate_total_cycles`，分数与改动前逐位相同；第二版把访存周期
> 一起缩放，GeneSim 报 “PIM memory service time is below its configured
> peak-bandwidth lower bound”。现在两个入口同源，且访存周期保持原值。

### 6.3 现在的行为

- 改 tasklet 会改变 A/B 两路 PIMIR、算子编译缓存键、trace 成本与 `total_time_s`；
- 有效并行度被 M/tile 夹取：M 小于 tasklet 数时不会凭空加速；
- 单 tasklet（缺省）与改动前逐字节/逐周期一致。

### 6.4 验证

| 测试 | 结果 |
|---|---|
| FlagTree `LowerPIMToEmitC.cpp` 编译与 `triton-opt` 链接 | 通过 |
| `tests/test_tune_compile_bridge.py` | 25 passed；B 路 tasklet 1 vs 2 的 EmitC 循环数为 1 vs 2 |
| `tests/test_placement_export.py` | 64 passed；B 路条目写 DPU/tasklet 与复制档 |
| `tests/test_opcompiler_ops.py` | 新增 B 路矩阵乘回归：tasklet 1/2/4 的 M 循环块数为 1/2/4，三次结果与 numpy 镜像逐元素一致 |
| `tests/sim/test_pim_trace_loader.py`、`test_pimir_trace.py` | 101 passed；矩阵算子在 `estimate_total_cycles` 与 `estimate_cycle_breakdown` 两条入口都按有效并行度缩放，向量算子不缩放 |
| `tests/sim/test_scheduler.py`、`test_pipeline_execution.py` | 72 passed |

### 6.5 仍需注意

- 访存周期不随 tasklet 缩放，这是满足峰值带宽下界的保守口径；真实 DMA 若有独立并发度，需要实测数据再收紧。
- B 路 tasklet 切分只覆盖 `pim.matmul`；逐元素/视图算子仍不按 tasklet 并行，和当前 FlagTree 降级一致。
- 改动 tasklet 后 quick/strategy 的历史分数会变化，旧基准不能直接对比。
