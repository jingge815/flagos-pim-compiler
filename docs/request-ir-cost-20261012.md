# 需求文档：IR 驱动的端到端代价评估补全

> 文档编号：request-ir-cost-20261012
> 创建日期：2026-10-12
> 关联项目：flagos-pim-compiler、genesim（FlagTree 本轮不改）

## 一、需求背景与目标

1. 业务背景：Llama2 7B 已按 `genesim/docs/llama-2.md` 的工作流跑通（`model_parser.py` 出模型 IR，`run.sh --trace` 出请求 trace，`refine_ir_with_flagtree.py` 分别以 `ttir` 和 `pimir` 精化成本，最后 `./run.sh` 出仿真）。仿真前已经吃到算子级静态量（flops、搬运字节、分片形状），但图编译器的融合、通信、内存计划没有进入这套计算，换一种图优化或切分方案，仿真报告不随之变化。
2. 预期目标：设计并实现一套 IR 驱动的端到端代价评估，打通「图编译器 → 算子编译器 → GeneSim」的链路。算子编译器产出的 PIMIR 已经通过 `analyze_ir` 进入评估，本轮补的是图编译器的决策。做法是在 GeneSim 原有公式上融入这些信息，而不是另起一套评分；评估公式要科学合理，能够真实体现实际的运行效果，决策的变化要在仿真报告的对应字段上可见。具体满足四条：
   - 代价覆盖计算、访存、通信、并行性四项，每一项都落在现有公式的某个输入上，对应关系见第三节；
   - 不同的算子 IR 或图 IR，给出不同的评估值；
   - 评估值的相对趋势符合技术直觉与工程预期，无反常倒挂，判据见 2.4；
   - 改动可控，复用现有的 sidecar 与公式，不动核心编译流程和仿真内核。
3. 关联范围：pim-compiler 的 `genesim_bridge/` 与 `comm/`，GeneSim 的调度器读取侧、通信计时与 PIM 访存周期。FlagTree 不在本轮范围内。

## 二、核心功能需求

### 2.1 功能清单

**F1（P0）通信代价进入 GeneSim 原有通信公式**

- 现状：`comm/plan.py:360` 的 `plan_cost` 已从 `node.meta["redistribute"]`（`contracts/unified_ir.py:96`，类型 `list[RedistributeEdge]`）算出搬运次数和字节数，但这条结果没有下游。GeneSim 的通信时延用自己的链路公式另算，两条链路互不相通。
- 需求：把编译器生效的跨 DPU 通信字节数传进 GeneSim，由 GeneSim 现有的链路时延公式消费。不在 GeneSim 里新建第二套通信模型，也不把编译器侧的秒数直接加进去。
- 验收判据：见第五节判据表。`redistribute` 总字节数翻倍后，通信时延的增长比例须严格大于 1 且不超过 2。

**F2（P0）融合结果进入评估**

- 现状：融合标记（`fused_tail`、`pim_rms_norm`、`pim_rope`、`pim_absorbed`）登记在 `contracts/unified_ir.py:108-156`，消费者只有 GML 与算子编译器。被吸收节点省掉的中间搬运不体现在 `KernelCost` 里。
- 需求：被 `pim_absorbed` 折进主算子的节点不再单独计一次中间结果搬运，使融合开关反映到导出的搬运字节数，并经现有 sidecar 覆盖路径进入仿真。
- 验收判据：见第五节判据表。融合开启后省掉的搬运字节须与被吸收的中间张量大小相当。

**F3（P0）四项代价的合理性**

- 现状：`genesim_bridge/ir_cost.py:478` 的 `analyze_ir` 已产出 flops、搬运字节、DMA 次数，但 `tests/` 里没有断言这些量与第三节公式一致。只断言「不同的 IR 给出不同的值」不够：数值不变时单边不等式仍然成立，字段没被读取也会被判通过。
- 需求：合理性分三层，都打在导出量与逐算子时延上，不打在仿真墙钟上。
  - 复算：固定一个小 IR，按第三节公式手工算出 flops、搬运字节、`t_compute`、`t_memory` 与通信时延，程序输出须在容差内与手算值相等。
  - 幅度：通信量翻倍后，通信时延的增长比例须严格大于 1 且不超过 2，等于 1 视为该字段未被消费。融合开启后省掉的搬运字节须与被吸收的中间张量大小相当，而不是仅仅变小。
  - 反倒挂：构造一对 IR，后一个算子数更多但因融合而搬运更少，代价须随搬运下降，不得随算子数上升。
- 验收判据：见第五节判据表。

**F4（P0）内存区间进入评估并改变数值**

- 现状：`memory/mem_planner.py:279` 的 `plan_dpu` 把每台 DPU 的 MRAM 从 0 顺序排成权重区、KV 区、激活区，三者语义不同：权重常驻、加载一次；KV 按 `max_seq` 预留整段、按 token 反复读写；激活是临时区、按生命周期复用。GeneSim 的 `memory_service_time_s` 不读这个划分，PIM 访存用全局一套周期：`LOADN = 12 + 字节数 × 0.125`、`STOREN = 10 + 字节数 × 0.1`（`src/pim/pim_trace_loader.py`，默认表 `DEFAULT_CYCLE_COSTS`）。
- 需求：sidecar 带上每个分片的区间归属，GeneSim 按区间使用不同的每字节周期，使同一字节数落在不同区间时访存时间不同。周期数值的分档属于设计文档的内容，本需求只定方向：常驻且复用的区间，其每字节周期不得高于按 token 反复搬运的区间。
- 验收判据：同一算子、同样字节数，全部落在激活区时的访存周期不低于全部落在权重区时。区间归属与 `DPUPlan` 的划分一致。

**F5（P1）修正过期文档**

- 现状：`docs/spec-index.md:26` 写「代价模型……当前不实现、不预留抽象」，把它划在第 2/3 阶段。该文案属于第 1 阶段，而算子级代价（`ir_cost.py`、`cost_extractor.py`）已经实现，本轮要做的正是第 2 阶段的代价模型。
- 需求：改写该句，说明第 1 阶段的边界已过、代价模型已进入实现，并指向本需求文档。不改动其他文档。

### 2.2 核心流程

```
FX 图 (node.meta: spec / redistribute / fused_*)
        │  spec_prop、fuse、mem_planner（已有，不改其算法）
        ▼
genesim_bridge 导出
   ├─ 算子级：analyze_ir → flops / 搬运字节        （已有）
   ├─ 放置：shard_map → 本地分片形状 / pimir_path   （已有）
   ├─ 通信：plan_cost(redistribute) → 跨 DPU 字节数 （本轮新增）
   ├─ 融合：absorbed 节点不计中间搬运              （本轮新增）
   └─ 区间：每个分片的权重 / KV / 激活归属          （本轮新增）
        ▼  sidecar JSON
GeneSim（沿用原有公式，只换输入）
   ├─ 通信字节数进入链路时延公式
   └─ 区间归属选择每字节访存周期
        ▼
./run.sh → results/ 仿真报告
```

### 2.3 边界定义

包含：

- 通信字节数、融合后的搬运字节数、内存区间归属三类信息进入现有 sidecar。
- GeneSim 在原有公式上消费通信字节数与区间归属。
- IR 侧导出量的单调性断言。

不包含：

- 能耗。三个仓库均无能耗数据（GeneSim `src/` 中 `energy` 命中 0 次），本轮不建模。
- FlagTree。其 IR 已有布局与访存标注（`tasklet_tiled`、`contiguous_dim`、`elem_stride`、tile 尺寸），flops 由本仓扫描文本计算，本轮不把代价数值写回方言。
- 替换 GeneSim 的公式。本轮只改变公式的输入，不改公式本身。
- 以仿真壁钟做单调性断言。仿真固定写 `pim_traces/` 与 `results/`，不能并发，且 llama2_7b 那组不进单次验证。
- 命中率模型。区间只决定每字节周期的分档，不引入缓存命中率。
- 常量折叠。图 IR 上没有该 pass。
- KV cache 的 DMA。它是片内写回，已按搬运字节计入访存，再并入通信会双计。
- 新增综合评分或权重。

### 2.4 合理性要求

「不同的 IR 给出不同的值」不是合理性的判据，数值不变也能满足单边不等式。本轮以第三节公式为基准，分三层验收：

- 复算。固定一个小 IR，按公式手工算出 flops、搬运字节和各项时延，程序输出须在容差内与手算值相等。这是四项代价都生效的直接证据。
- 幅度。通信时延是基础时延加字节数除以带宽，字节数翻倍后时延增长比例须严格大于 1 且不超过 2。比例等于 1 说明字段没有被读取，判为不通过。
- 反倒挂。算子数更多但搬运更少的 IR，代价须随搬运下降，不得随算子数上升。

四项代价要体现在时延上而不只是中间量上。把算子调成计算受限（`t_compute` 大于 `t_memory`），放大 M、N、K 必须推动时延；调成访存受限，放大字节数必须推动时延。

这些断言落在导出量与逐算子时延上。`total_time_s` 含流水重叠与并发，比例会被淹没，只作为模型级验证的观察项，不做比例断言。

原始目标里还有三项本轮不做，因为没有数据来源：能耗、缓存命中率、连续与随机的访存模式。它们不作为本轮合理性的判据。

上面三层只证明程序与公式自洽。公式本身偏离实际运行时，自洽的实现同样是错的。所以再加四类校验，每类抓一种自洽发现不了的错误：

- 两路对照。同一算子走两条独立路径：IR 导出 flops 与字节数再套 roofline，以及 GeneSim 把 PIMIR 编译成指令 trace 再按周期表累加。两条路径的时延须在容差内一致。对不上说明 IR 这条路漏算或重算了，而复算发现不了，因为复算用的就是 IR 这条路自己的口径。
- 精确锚点。闭式量断言精确相等，不留容差。矩阵乘的运算数就是 `2MNK`，与 `ir_cost.py` 中 `_line_tile_flops`、`_line_operator_flops` 的算法一致；一段连续搬运的周期就是基数加字节数乘每字节周期。容差只留给拟合与浮点累加，用在闭式量上会把整数倍的错误藏起来。
- 隔离性。单变量变化时，不该动的项不得动。放大 K，`t_compute` 上升而通信时延不变；放大 `redistribute` 字节数，通信时延上升而 `t_compute` 不变。只要求该动的动了，四项代价可能耦在一起、互相抵消，表面仍是单调的。
- 账目。各分片字节数之和等于总量，权重、KV、激活三区之和等于总搬运，KV 写回不出现在通信项里。ir_cost 对 KV 写回按新值计 1 倍，GeneSim 的模型 IR 与 trace 按读加写计 2 倍，这个口径差只有对账能抓住。

这四类通过，证明的是评估与 GeneSim 自身的细粒度仿真自洽，不是与真实芯片一致。仓库里没有硅片数据，验收文字不得写成符合实际运行。


## 三、公式基准

本轮所有实现以 GeneSim 现有公式为准，编译器信息只替换公式的输入项。这些公式就是本轮对「科学合理」的定义：评估值的变化方向必须与它们一致，不允许出现规模更大、评估值反而更低的倒挂。下面每条都是源码原文的口径，设计与实现不得另立公式。

**计算与访存（GPU，`src/backend/gpu_backend.py:147`）**

```
t_compute = flops / (peak_flops × precision_factor × efficiency_compute)
t_memory  = data_bytes / (peak_bandwidth × efficiency_memory)
latency   = max(t_compute, t_memory) + 10µs
```

`data_bytes` 不超过 `l2_cache_mb` 时带宽再乘 `L2_SPEEDUP = 3`。CPU 路径（`src/backend/cpu_backend.py:131`）同结构，无效率系数，峰值按核数放大。

**PIM 访存（`src/pim/pim_trace_loader.py`，默认周期表）**

```
LOADN  周期 = 12 + 字节数 × 0.125
STOREN 周期 = 10 + 字节数 × 0.1
latency = 总周期 × clock_period_ns × 1e-9
```

`clock_period_ns` 取 `pim.clock_period_ns`，配置文件中为 0.63。当前不区分权重、KV、激活，这正是 F4 要补的输入：区间归属决定选用哪一档每字节周期，周期表的结构不变。

**通信（`src/backend/comm_lib.py:36`）**

```
时延 = latency_base_ns × 1e-9 + 字节数 / (bandwidth_gbps × 1e9 / 8)
```

带宽与基础时延来自 `comm_config.links`，例如片内 4096 Gbps、片间 1024 Gbps、主机侧 307.2 Gbps。F1 只提供字节数，时延仍由该式计算。

**字节数与 flops 的来源（`genesim_bridge/cost_extractor.py:257` 的 `_fit_coeffs`）**

prefill 与 decode 两个测量点拟合，GEMM 按 `Tq`、其余按 `Tq(Tp+Tq)`：

```
a = (y1 - y2) / (t1 - t2)
b = y1 - a × t1
```

斜率或截距为负时退化为过较大测量点的原点直线。得到的 `flops_coeffs` 与 `data_bytes_coeffs` 写回模型 IR。GeneSim 侧若 sidecar 的 `mram_traffic_bytes > 0`，则用它覆盖 `op.data_bytes`（`src/scheduler/gene_sim_scheduler.py:284`）。因此 F2、F4 改的是这个被覆盖的量，不用新通道。

**并行度**

并行性里由 IR 决定的只有一项：`tp_width` 改变单分片的形状，从而改变单分片的 `t_compute`、`t_memory` 与跨分片的通信字节数。本轮的并行判据只落在这一项上。

另外两个量不算数。`pipeline_bubble_time_s` 是仿真跑完才统计的后验量，不是从 IR 算出来的，用它充当并行性就退回了仿真后验，与 IR 驱动的目标相反。算子效率系数是按算子类型硬编码的常数，不随 IR 变化。指令级并行、同步开销、资源冲突没有公式、也没有数据来源，本轮不建模，不作为覆盖判据。

**对外指标**

`./run.sh` 的 `results/summary.json` 中，`throughput_tokens_per_s = processed_tokens / total_time_s`（`src/scheduler/gene_sim_scheduler.py:8175`），其中 `processed_tokens = prompt_tokens + generated_tokens`，`total_time_s` 是所有请求完成时刻的最大值。核心看这一项，同时看 `communication_time_s`、`memory_service_time_s`、`average_ttft_s` 和 `efficiency_metrics`，因为切分与放置的变化经常先出现在这几项上，而被墙钟掩盖。

## 四、技术约束与依赖

1. 改动范围


| 文件                                         | 现状                                                             | 本轮动作                                 |
| -------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| `genesim_bridge/placement_export.py`         | `export_placement_to_genesim`（341 行）写放置 sidecar            | 增补通信字节数与区间归属                 |
| `genesim_bridge/ir_cost.py`                  | `analyze_ir`（478 行）扫描 MLIR 文本                             | 被吸收节点不再单独计中间搬运             |
| `genesim_bridge/cost_extractor.py`           | `_fit_coeffs`（257 行）与导出                                    | 透传新增字段，不改拟合公式               |
| `comm/plan.py`                               | `plan_cost`（360 行）已算出 `CommCost` 但无下游                  | 作为通信字节数来源，不改算法             |
| `contracts/unified_ir.py`                    | `META_KEYS`（66 行）登记 16 个键                                 | 若新增键，必须登记 producer 与 consumers |
| GeneSim`src/scheduler/gene_sim_scheduler.py` | 已加载 sidecar，284 行起用`mram_traffic_bytes` 覆盖 `data_bytes` | 读取通信字节数，按区间选择每字节周期     |
| GeneSim`src/pim/pim_trace_loader.py`         | 全局一套`LOADN_PER_BYTE`、`STOREN_PER_BYTE`                      | 允许按区间取不同的每字节周期             |
| `docs/spec-index.md`                         | 26 行声称代价模型未实现                                          | 按 F5 改写                               |

2. 风险与应对

- sidecar 是跨仓约定。新增字段必须有默认值，旧 sidecar 仍能跑，缺字段按「无额外通信、使用默认周期」处理并显式记录。
- 编译器侧 `plan_cost` 用 host-star 模型算出的秒数，与 GeneSim 的链路时延不是同一口径，两者不可相加。本轮只传字节数。
- KV 写回已在访存里。ir_cost 按新值计 1 倍，GeneSim 的模型 IR 与 trace 按读加写计 2 倍。F4 不得把 KV 字节再加进通信，也不得把整块 cache 计入搬运字节，否则会放大几个数量级并双计。
- 融合是否真的减少搬运，取决于算子编译器是否不再单独发射被吸收节点。断言前先确认这一点。

## 五、验收标准

1. 回归命令与基线

- 本仓：`python -m pytest tests/ -q -k "not llama2_7b"`。该命令当前有既有失败（全量 34 条，源于 llama2_7b 用例污染 GML 导出），验收以新增失败数为零为准。
- GeneSim：`./run.sh --test`（unittest，45 个测试文件）。本轮改动不得新增失败。
- 模型级验证沿用 `genesim/docs/llama-2.md` 的工作流，不进单次回归，合入前单独跑。跑完后核对 `results/summary.json`：`completed_requests` 与请求数一致，`throughput_tokens_per_s = processed_tokens / total_time_s`，并记录 `communication_time_s`、`memory_service_time_s`、`average_ttft_s`、`efficiency_metrics`。该文档原文只要求核对 `completed_requests == 10`，其余字段是本轮补充的观察项。

2. 逐项判据


| 编号 | 操作 | 期望 |
| ---- | ---- | ---- |
| F1 | `redistribute` 总字节数翻倍 | 通信时延增长比例严格大于 1 且不超过 2 |
| F1 | 去掉所有 `redistribute` 边 | 通信时延不增大 |
| F2 | 融合开关开与关各导出一次 | 开启后省掉的搬运字节精确等于被吸收的中间张量字节数 |
| F3 | 固定小 IR，按第三节公式手算 | GPU roofline 口径的手算本轮豁免，见设计文档第八节第 12 条。PIM 侧的逐算子时延由下面两行覆盖 |
| F3 | 把矩阵乘的 K 放大为原来的 2 倍 | trace 计算周期的增长比例严格大于 1 且不超过 2 |
| F3 | 把搬运的每元素字节放大为原来的 2 倍 | trace 访存周期的增长比例严格大于 1 且不超过 2 |
| F3 | 算子数更多但因融合搬运更少 | 代价随搬运下降，不随算子数上升 |
| F3 | `tp_width` 放大为原来的 2 倍 | 单分片搬运字节数的下降比例严格大于 1 且不超过 2 |
| F4 | 同样字节数分别落在激活区与权重区 | 激活区的访存周期不低于权重区 |
| F4 | 导出 sidecar | 每个分片的区间归属与 `DPUPlan` 一致 |
| 两路对照 | 同一算子分别走手写模板 trace 与 PIMIR 编译 trace | 两条路径的计算与访存周期在容差内一致。roofline 那半条本轮豁免：PIM 时延由 trace 周期决定，不消费 flops 与字节套出的 roofline，见设计文档第八节第 12 条 |
| 精确锚点 | 形状为 M、K、N 的矩阵乘 | 运算数精确等于 `2MNK` |
| 精确锚点 | 一段连续搬运 | 周期精确等于基数加字节数乘每字节周期 |
| 隔离性 | 只放大 K | `t_compute` 上升，通信时延不变 |
| 隔离性 | 只放大 `redistribute` 字节数 | 通信时延上升，`t_compute` 不变 |
| 账目 | 导出分片与三区字节数 | 切分轴的本地宽度之和等于全局宽度，KV 写回不在通信项里。按字节求和本轮不成立：未切分一侧的激活在各片重复出现，分片字节之和本来就大于全局字节。「三区之和等于总搬运」本轮豁免：区间不再按算子汇总字节，而是逐段落在指令上，没有可对的汇总，见设计文档第八节第 15 条 |
| 四维覆盖 | 同一模型的一次评估 | 计算、访存、通信、并行四项各自非零且来源可追溯：运算数来自算子形状，搬运字节来自张量与区间，通信时延来自 `redistribute` 字节数，单分片字节数来自 `tp_width`。去掉任何一项的输入，只有该项归零，其余三项不变。并行项只认 `tp_width`，不认 `pipeline_bubble_time_s`。「`t_compute` 与 `t_memory` 相等时两项同时放大」本轮豁免：PIM 时延由 trace 周期决定，不消费这条 GPU roofline 输入，见设计文档第八节第 12 条 |
| IR 区分 | 未融合、融合后、加宽 TP、加重通信四种 IR 各评估一次 | 四次的逐算子时延不全相同。融合只改搬运字节，加宽 TP 只改单分片字节数，加重通信只改通信时延，其余项与未融合的那次相等 |
| 反倒挂 | 访存落在更慢区间但总字节数减半 | 访存时延不高于字节数减半前，即字节项的下降不少于区间变慢带来的上升 |
| 反倒挂 | 通信字节数减半，同时把基础时延加大到不超过字节项节省的时间 | 通信时延不高于原来 |

3. 不回归判据

- 不带新字段的旧 sidecar，GeneSim 仍能跑完并给出 `throughput_tokens_per_s`。
- `KernelCost` 既有字段含义不变，`tests/test_ir_cost.py` 与 `tests/test_genesim_bridge.py` 的现有断言保持通过。
- FlagTree 的方言与 pass 不变，`docs/pim-compiler-v0.0.7.md` 不变。
