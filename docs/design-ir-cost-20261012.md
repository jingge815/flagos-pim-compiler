# 技术设计文档：IR 驱动的端到端代价评估补全
> 文档编号：design-ir-cost-20261012
> 创建日期：2026-10-12
> 关联需求文档：docs/request-ir-cost-20261012.md
> 关联项目：flagos-pim-compiler、genesim

## 一、设计概述
### 1.1 需求背景回顾
Llama2 7B 已按 `genesim/docs/llama-2.md` 跑通仿真。算子级静态量经 `analyze_ir` 进入评估，但图编译器的通信与内存决策没有进入 GeneSim 的计时，换一种图决策仿真报告不变。

### 1.2 设计目标
让编译器的两类决策进入 GeneSim 现有的计时路径：把编译器图上有、而 GeneSim 图上没有的跨 DPU 通信落成依赖边；按权重、KV、激活三区拆开搬运字节，使访存时延随区间变化。不新增公式，不双计。

### 1.3 设计范围
- 包含范围：补齐 GeneSim 未展开的集合通信，按区间拆分搬运字节，以及需求文档第五节里不依赖已撤回设计的判据。
- 不包含范围：能耗、命中率、FlagTree、常量折叠、KV DMA 计入通信、`docs/pim-compiler-v0.0.7.md`，以及在 `ir_cost.py` 里按 `pim_absorbed` 扣搬运。

## 二、现状基线

PIM 的时延不读 flops。`PIMVPU.execute_profile`（`src/vpu/vpu.py:298`）的时延等于指令 trace 的周期乘时钟，flops 只进 GPU 与 CPU 的 roofline、放置粗估和跑完后的 MFU。sidecar 里的 `fitted_coeffs` 在 GeneSim 全仓没有读取点。所以只改 flops、不改 trace 的循环次数，PIM 时延不变。

通信已经有一条完整的计时路径。`enqueue_dependency_transfer`（`src/scheduler/gene_sim_scheduler.py:6165`）看依赖边两端算子的 `vpu_id`，不同就按边的字节数排队，时延由 `CommLink.latency` 计算，最终累加进 `communication_time_s`（`src/main.py:239`）。字节数来自边的 `always_bytes_coeffs` 与 `first_use_bytes_coeffs`（`src/scheduler/gene_sim_scheduler.py:6026` 的 `dependency_transfer_bytes`）。

TP 的 all-reduce 也已在这条路径上。`_expand_multi_shard_operators`（同文件 `1368`）把多副本算子展开，`_attach_reduce_node`（`1318`）为每个分片接一条入边，每条搬全量输出字节，归约节点落到 GPU。所以把 `redistribute` 边的 `nbytes` 再加一遍会双计。

`plan_cost`（`comm/plan.py:360`）对一条 all-reduce 的计法是收集加回写，共 `2 × N × edge.nbytes`，其中 `edge.nbytes` 是全局字节数（`contracts/pim_tensor_spec.py:136`）。GeneSim 现有展开只计了 N 条入边，出边只接到原下游的残差节点，不写回各分片。

llama2 一次传播的边是 all_gather 100 条、all_reduce 64 条、local_slice 32 条、scatter 6 条。all_to_all 只在测试夹具里出现。GeneSim 只展开了 all_reduce，其余跨设备边没有进入计时，local_slice 在 DPU 内部、不产生通信。

融合不需要在成本扫描里处理。`gml_bridge/from_fx.py:223` 的 `_is_emittable` 对 `pim_absorbed` 直接返回假，被吸收节点不进入发射集合。`analyze_ir` 的输入是孤立算子配方（`cost_extractor.py:153` 的 `_measure_oplevel`），文本里没有 FX 节点名，也没有被吸收节点。`emit_oplevel_mlir` 还要求图已融合，未融合的图会直接报错，且只发射五类算子。

区间落不到指令上。`loadn` 与 `storen` 只接收字节数（`src/pim/pim_isa.py:365`），编码时其余字段保持默认。`compile_from_pimir` 展开时同样只传字节数（`src/pim/pimir_trace.py:466`）。PIM 的 `memory_time_s` 由 trace 周期算出（`src/vpu/vpu.py:302`），`op.data_bytes` 的 roofline 只在没有 trace 时兜底。同一个 GEMM 的 trace 里权重搬运和激活搬运交替出现，整算子套一个区间必然算错。

## 三、总体设计
### 3.1 总体设计概述
问题是两类编译器决策没有进入计时：GeneSim 只展开了 all-reduce 的收集方向，且访存不分区间。方案是沿现有依赖边补齐缺失的通信，并在 trace 生成处按区间拆字节。

```
编译器 redistribute 边
  all_reduce 收集方向 ──原有──> _attach_reduce_node 的入边 ──> enqueue_dependency_transfer
  all_reduce 回写方向 ──新增──> 归约到各分片的出边          ──> 同一条路径
  all_gather / scatter / all_to_all ──新增──> 依赖边        ──> 同一条路径

搬运字节
  权重 / KV / 激活 ──新增──> 按区间拆分的字节数 ──> trace 生成时分段计周期
切分宽度
  tp_width ──原有──> 分片各钉一台 VPU ──> 流水线并行，墙钟不随片数线性增长
```

### 3.2 与现有逻辑的衔接
接入点是 `_expand_multi_shard_operators` 之后。新增的边用 `Dependency` 表达，字节数用 `always_bytes_coeffs` 写出，这样 `dependency_transfer_bytes`（`6026`）不用改就能取到值，时延仍由 `enqueue_dependency_transfer` 计算。不在 `communication_time_s` 上做任何加法。

### 3.3 模块划分与职责边界
- 通信的补齐放在 GeneSim 调度器，因为只有它知道展开后的 `vpu_id`。编译器只提供边的类型和全局字节数。
- 区间拆分放在 `pimir_trace.py` 的解析处：每段搬运带 `base_arg`，展开时按它取每字节周期。GeneSim 只消费这个倍数。
- `ir_cost.py` 只加一处：折进主算子的激活按结果张量记一次遍历。`purpose = absorbed` 的 transpose 仍照常计费（`ir_cost.py:417`），图层面的吸收走 `node.meta`，不在这里扣。
- `comm/plan.py` 的 `plan_cost` 只作字节数的对照，它的秒数不进入仿真。

## 四、模块详细设计

### 4.1 接入 all_gather 与 scatter（对应需求点：F1｜现有代码位置：`src/scheduler/gene_sim_scheduler.py:1368`｜改动类型：修改）
**功能点**：把编译器图上的 all_gather 与 scatter 落成依赖边，进入现有的 `enqueue_dependency_transfer`。
**原因**：llama2 上这两类共 106 条，GeneSim 没有展开，通信时延被漏计。all_reduce 已由 `_attach_reduce_node` 覆盖。
**代码对比**：
```python
# 原有：只展开 all_reduce 的收集方向
_expand_multi_shard_operators(placed_ir)

# 目标：按 sidecar 里的边补 all_gather 与 scatter
for edge in sidecar["collectives"]:
    placed_ir.add_dependency(edge.src_vpu, edge.dst_vpu, edge.bytes)
```
**解决思路**：sidecar 只带这两类边的参与 DPU 与全局字节数。不补归约写回各分片的边，llama2 上归约结果的消费者是残差，那条边不存在。all_to_all 本轮不处理。

### 4.2 解析时保留搬运的操作数身份（对应需求点：F4｜现有代码位置：`src/pim/pimir_trace.py:61`｜改动类型：修改）
**功能点**：`_DMA_LOAD_RE` 额外提取 `base_arg`，让每段搬运知道自己是激活、权重还是输出。
**原因**：正则只抓形状和类型，身份在这一层就丢了，后面无法按区间计周期。
**代码对比**：
```python
# 原有：只留形状
_DMA_LOAD_RE = re.compile(r"pim\.dma_load\b.*?!pim\.memdesc<([0-9x]+)x([a-z0-9]+)")

# 目标：带上 base_arg
_DMA_LOAD_RE = re.compile(r"pim\.dma_load\b.*?base_arg = (\d+).*?memdesc<([0-9x]+)x([a-z0-9]+)")
```
**解决思路**：0 归激活、1 归权重、2 归输出。解析结果从三元组改成带身份的四元组，向后传给展开。

### 4.3 按身份乘每字节周期（对应需求点：F4｜现有代码位置：`src/pim/pimir_trace.py:466`｜改动类型：修改）
**功能点**：展开 trace 时按 `base_arg` 取每字节周期，激活比权重贵。
**原因**：同一算子的权重搬运和激活搬运混在一条指令流里，只能逐段乘，不能整算子乘一个系数。
**代码对比**：
```python
# 原有：所有搬运用同一每字节周期
cycles = 12 + nbytes * 0.125

# 目标：按区间取周期，缺省回退
cycles = 12 + nbytes * per_byte[zone]
```
**解决思路**：默认表的每字节周期就是配置的峰值带宽速率，所以倍数不得低于 1——低于它，PIM 访存时延会掉到峰值带宽下界之下，`execute_profile` 直接报错。权重取 1 倍（跑在这条理想速率上），KV 与激活各取 2 倍（按 token 反复搬运的那一档），LOADN 与 STOREN 同比例。倍数写进指令的 `src2`（取两倍以便装进整数），没写时按默认周期。激活也不能再高：超过 2 倍，字节减半就盖不住区间变慢，评估值会随规模倒挂（需求第五节的访存反倒挂判据）。

### 4.4 并行性由分片占多台 VPU 体现（对应需求点：F3｜现有代码位置：`src/scheduler/gene_sim_scheduler.py:1094`｜改动类型：不改代码）
**功能点**：确认 `tp_width` 增大后，各分片钉在不同 VPU 上，整层墙钟低于各分片时延之和。
**原因**：单分片周期随形状变，属于计算维度。并行要看的是多分片同时跑，这件事由放置和流水线调度完成，缺的是断言。
**代码对比**：
```python
# 原有：分片已钉到不同 VPU，但没有校验墙钟
pinned[new_id] = dpu_ids[shard_index]

# 目标：测试侧比较墙钟与串行之和
assert layer_makespan(tp=4) < sum(shard_latency(tp=4))
```
**解决思路**：不新增代码。判据取流水线模拟的层墙钟，它必须小于同一层各分片时延相加，由 `genesim/tests/sim/test_pipeline_execution.py` 的 `test_tp_shard_layer_makespan_is_below_the_serial_sum` 覆盖：四个分片落在四台 VPU 上，时延 3/5/4/6 秒，层墙钟等于最慢的 6 秒而不是串行的 18 秒。

### 4.5 融合对比用 activation 参数（对应需求点：F2｜现有代码位置：`genesim_bridge/op_classify.py:320`｜改动类型：修改）
**功能点**：融合侧测带 `activation="silu"` 的矩阵乘，未融合侧测无激活矩阵乘再加一次 silu。
**原因**：现有配方不传 `activation`，测量结果与图是否融合无关。
**代码对比**：
```python
# 原有：配方固定，不带激活
pimir=lambda: _module(matmul_kernel("linear", m, k, n))

# 目标：融合侧带上激活
pimir=lambda: _module(matmul_kernel("linear", m, k, n, activation="silu"))
```
**解决思路**：`matmul_kernel` 已有 `activation` 参数，配方传入即可，省掉的字节等于单独一次 silu 的搬运。**融合与否由放置 sidecar 的 `fused_activations` 清单决定**：清单来自 `export_placement_to_genesim` 读图上的 `FUSED_TAIL_META_KEY`，没有清单就按未融合计——融合是图编译器的决策，不能从模型 IR 的形状猜。标记只在导出过程内部使用，写精化 IR 前去掉，否则 GeneSim 的 `Operator.from_dict` 会因未知字段抛错。放置导出在写 sidecar 前先跑一次 `fuse_for_pim`：真实链路上原本没有融合 pass，清单恒为空；加它之后仿真成本基线随之改变，10 请求基线按当前代码于 2026-10-15 重跑，见实施文档 7.3。

A 路 kernel 由 Triton 生成，带不出激活属性。生产链路走 `--ir-level pimir` 时，`mark_fused_activation` 把 `activation = #pim.act_spec<kind = silu>` 标到 `tt.dot` 上：放置导出另存一份 `*.silu.pimir.mlir` 给仿真编 trace，精化时对捕获到的 kernel 文本同样标注再交给 `analyze_ir`。属性只带 kind，给正则消费，不进 MLIR verifier。写出与识别收拢在 `ir_cost.fused_act_spec_attr` / `has_fused_act_spec`。分块 GEMM 的激活消费归约之后的写回，按 memdesc 计一次，不记在 K 循环里的 `tt.dot` 上。

### 4.6 修正阶段说明（对应需求点：F5｜现有代码位置：`docs/spec-index.md:26`｜改动类型：修改）
**功能点**：把「代价模型当前不实现」改为指向本设计与需求文档。
**原因**：该句是第 1 阶段文案，算子级代价已经实现。
**代码对比**：
```markdown
# 原有
代价模型、自动切分……当前不实现、不预留抽象。
# 目标
代价模型已进入实现，见 docs/request-ir-cost-20261012.md。
```
**解决思路**：只改这一句，其余文档不动。

## 五、改动汇总
### 5.1 文件级改动清单

| 文件路径 | 改动类型 | 改动内容简述 | 影响范围 | 优先级 |
|---|---|---|---|---|
| `genesim/src/scheduler/gene_sim_scheduler.py` | 修改 | 接入 all_gather 与 scatter | `communication_time_s` | P0 |
| `genesim/src/pim/pimir_trace.py` | 修改 | 解析保留 `base_arg`，展开时按身份乘周期 | 访存周期 | P0 |
| `genesim_bridge/placement_export.py` | 修改 | sidecar 写未展开的 all_gather 与 scatter | GeneSim 读取侧 | P0 |
| `genesim_bridge/op_classify.py` | 修改 | 配方传入 `activation`，并补上 A 路 `build` | 融合对比 | P0 |
| `genesim_bridge/cost_extractor.py` | 修改 | 按融合标记选配方，被吸收节点的系数清零 | 融合计费 | P0 |
| `genesim_bridge/ir_cost.py` | 修改 | 折进主算子的激活记一次遍历 | 融合计费 | P0 |
| `docs/spec-index.md` | 修改 | 改第 26 行 | 无 | P1 |
| `tests/test_ir_cost_eval.py` | 新增 | 第七节的判据 | 无 | P0 |
| `scripts/refine_ir_with_flagtree.py` | 修改 | 透传 `--placement` 的融合清单 | 融合计费 | P0 |
| `scripts/export_pp_placement.py` | 修改 | 导出前运行融合 pass；sidecar 带融合清单与 KV 区间 | 融合计费、访存区间 | P0 |
| `genesim/src/pim/pim_isa.py` | 修改 | 区间倍率表收拢到一处，生成端与求值端共用 | 访存周期 | P0 |
| `scripts/run_full_pipeline.py` | 修改 | 放置导出提到精化之前，精化带上放置 sidecar | 融合计费 | P0 |
| `opcompiler_bridge/oplevel_kernel.py` | 修改 | B 路矩阵乘改走激活属性 | 融合计费 | P0 |
| `genesim/src/backend/gpu_backend.py` | 修改 | 通信中转节点不计时延 | 通信时延 | P0 |
| `genesim/src/pim/pim_compiler.py` | 修改 | 手写模板的搬运带操作数身份 | 访存周期 | P0 |
| `genesim/src/pim/pim_trace_loader.py` | 修改 | 求值端按 src2 取每字节周期 | 访存周期 | P0 |
| `genesim/docs/llama-2.md` | 修改 | 精化步骤补上融合清单参数 | 融合计费 | P1 |
| `genesim/scripts/refine_ir_with_flagtree.py` | 修改 | 透传融合清单 | 融合计费 | P0 |

### 5.2 新增数据结构
未展开通信边：`{type, nbytes, dpu_ids, seq_len}` 的列表，只含 all_gather 与 scatter。`seq_len` 必有：带序列维的边写导出时的序列长度，不带的边显式写 null。缺这个键的是旧 sidecar，调度器直接报错。all_reduce 不在此列，避免与 `_attach_reduce_node` 双计。

搬运身份：解析结果由 `(rows, cols, elem)` 改为 `(base_arg, rows, cols, elem)`。缺 `base_arg` 且没有算子级区间时 `base_arg` 为 `None`，按单一周期计；有算子级区间时落到该区间对应的身份。

周期分档是相对默认每字节周期的倍数，写进指令的 `src2`：权重 1、KV 2、激活 2。指令里没有区间字段，配置表按段生效不了，所以不改 `conf/sim.yaml`。算子级区间（读 KV 的算子）由放置 sidecar 的 `zone` 给出，作为搬运不带操作数身份时的默认档。

### 5.3 接口变更
`export_placement_to_genesim` 要求导出图带 `input_ids` 且该节点有 `val`，否则分别报「没有 input_ids」与「没有 val」，导出序列长度取自它。`analyze_ir` 签名不变，`export_placement_to_genesim` 增一个可选的 `kv_specs`，`export_costs_to_genesim` 增一个可选的 `placement_sidecar`（取融合清单，并把被吸收的算子编号回写成 `absorbed_ops`）。不新增整图测量入口。GeneSim 对没有 `collectives` / `fused_activations` / `zone` / `absorbed_ops` 的旧 sidecar 分别回退到「不补边」「未融合」「单一周期」「不跳过任何算子」，与改动前逐条一致。

## 六、实施计划
### 6.1 阶段划分与依赖
1. 集合通信补边与 sidecar 的边描述。
2. 搬运身份与区间倍率的解析、展开。
3. 融合清单从图上的标记写进放置 sidecar，精化侧照单折算。
4. KV 算子的区间归属进 sidecar，GeneSim 按它取默认档。
5. 判据测试与 `docs/spec-index.md`。

### 6.2 每阶段的回归门槛
每步提交前跑 `python -m pytest tests/ -q -k "not llama2_7b"`，新增失败数为零。涉及 GeneSim 的步骤另跑 `./run.sh --test`，不得新增失败。

## 七、验证与测试方案
### 7.1 单元测试清单

交付的测试与所在文件如下，逐条对应需求第五节的判据。

| 判据 | 测试 | 文件 |
|---|---|---|
| 运算数精确等于 2MNK | `test_four_dimensions_come_from_distinct_inputs` | `tests/test_ir_cost_eval.py` |
| 搬运周期等于基数加字节乘每字节周期 | `test_missing_identity_matches_old_cost` | `genesim/tests/sim/test_zone_cycles.py` |
| 激活区周期不低于权重区 | `test_activation_zone_costs_more_than_weight` | 同上 |
| 区间倍率真的改变数值 | `test_zone_scale_changes_the_cost`、`test_compiled_trace_carries_the_factor` | 同上 |
| 访存反倒挂：字节减半盖得住区间变慢 | `test_halving_bytes_offsets_the_slower_zone` | 同上 |
| KV 区间由算子级默认档给出 | `test_kv_zone_comes_from_the_operator_not_the_identity`、`test_kv_default_zone_reaches_the_instructions`、`test_sidecar_zone_reaches_the_scheduler_trace` | 同上 |
| 混排 body 里不带身份的搬运仍计费 | `test_untagged_transfer_is_still_counted` | 同上 |
| 字节翻倍时延落在 (1, 2] | `test_doubling_bytes_stays_within_bound` | 同上 |
| 两路对照：模板与 PIMIR 逐项一致 | `test_agrees_with_the_handwritten_template_on_the_same_shape` | `genesim/tests/sim/test_pimir_trace.py` |
| 补边后跨设备字节变大、all_reduce 不双计 | `test_allgather_adds_cross_device_bytes`、`test_scatter_adds_cross_device_bytes`、`test_allreduce_is_not_added_again` | `genesim/tests/sim/test_collective_edges.py` |
| 通信字节翻倍比例在 (1, 2] | `test_comm_latency_grows_within_bound` | 同上 |
| 通信反倒挂 | `test_halving_bytes_beats_a_larger_base_latency` | 同上 |
| 隔离性：去掉边只改通信 | `test_dropping_edges_only_changes_communication` | 同上 |
| 隔离性：放大算子形状不改通信 | `test_larger_gemm_does_not_change_communication` | 同上 |
| 中转节点不留额外时延 | `test_relay_operators_cost_no_extra_time` | 同上 |
| 并行：层墙钟小于各分片之和 | `test_tp_shard_layer_makespan_is_below_the_serial_sum` | `genesim/tests/sim/test_pipeline_execution.py` |
| 融合开关开与关各导出一次 | `test_fusion_comes_from_the_placement_sidecar` | `tests/test_ir_cost_eval.py` |
| 融合清单来自图上的标记 | `test_fused_activations_follow_the_graph` | `tests/test_placement_export.py` |
| 融合省掉一次中间遍历 | `test_fusion_saves_the_silu_traversal` | `tests/test_ir_cost_eval.py` |
| 被吸收节点不计费 | `test_absorbed_operator_is_not_billed`、`test_absorbed_operator_is_not_billed_at_pimir_level` | 同上 |
| 折激活的投影不丢权值搬运 | `test_fused_projection_keeps_its_weight_traffic`、`test_fusion_measurement_does_not_leak_across_same_shape_ops`、`test_fused_projection_bills_one_result_traversal_at_pimir_level` | 同上 |
| 算子数不变、被吸收节点不计费 | `test_more_operators_with_less_traffic_costs_less` | 同上 |
| 四种 IR 变化落在各自的项上 | `test_four_ir_kinds_change_distinct_fields` | 同上 |
| 精化 IR 能被 GeneSim 读回 | `test_refined_ir_loads_back_in_genesim` | 同上 |
| 精化命令带上融合清单 | `test_refine_step_passes_fusion_from` | 同上 |
| 融合按清单里的激活名折 | `test_fusion_rejects_activation_name_mismatch`、`test_fusion_skips_an_unrelated_consumer_before_the_named_activation` | 同上 |
| KV 算子的区间归属与内存规划一致 | `test_kv_operators_are_attributed_to_the_kv_zone` | `tests/test_placement_export.py` |
| sidecar 只带未展开的跨 DPU 边 | `test_collectives_lists_only_unexpanded_cross_dpu_edges` | 同上 |
| 相位链 LOADN 与 STOREN 同档 | `test_phase_chain_store_uses_the_same_zone_as_load` | `genesim/tests/sim/test_zone_cycles.py` |
| 融合落到仿真：被吸收节点不消耗周期 | `test_absorbed_operator_costs_no_cycles` | `genesim/tests/sim/test_scheduler.py` |
| 吸收清单写进放置 sidecar | `test_absorbed_ops_are_listed_for_the_simulator` | `tests/test_ir_cost_eval.py` |
| 分片继承吸收标记 | `test_shards_inherit_the_absorbed_flag` | `genesim/tests/sim/test_compiler_placement.py` |
| 输入不变时复用缓存 trace | `test_unchanged_input_reuses_the_cached_trace` | `genesim/tests/sim/test_scheduler.py` |
| 账目：分片宽度之和等于全局 | `test_shard_widths_sum_to_the_global_width` | `tests/test_placement_export.py` |
| 账目：KV 写回不进通信项 | `test_kv_writeback_never_enters_the_communication_account` | 同上 |
| 四维覆盖：去掉一项输入只该项归零 | `test_removing_one_input_zeroes_only_its_dimension` | `tests/test_ir_cost_eval.py` |
| 尾块口径：K 不整除时两条路径一致 | `test_ir_bytes_match_the_trace_bytes_on_a_partial_k_tail` | `genesim/tests/sim/test_pimir_trace.py` |
| 尾块口径：融合分块 GEMM 两条路径一致 | `test_ir_bytes_match_the_trace_bytes_on_a_fused_tiled_gemm` | 同上 |
| 折激活只记最后一次写回 | `test_fused_activation_bills_only_the_last_store` | 同上 |
| 尾块口径：报告趟数与成本同口径 | `test_reported_trip_count_matches_the_billed_cost` | 同上 |
| 搬运字节必须整字节 | `test_bulk_transfer_rejects_fractional_bytes` | 同上 |
| 融合算子上 ttir 与 pimir 运算数一致 | `test_fused_activation_flops_agree_across_ir_levels` | `tests/test_ir_cost_eval.py` |

### 7.2 集成验证
按 `genesim/docs/llama-2.md` 跑一次，不进单次回归。核对 `completed_requests` 与请求数一致，`throughput_tokens_per_s` 等于 `processed_tokens / total_time_s`，并记录 `communication_time_s`、`memory_service_time_s`、`average_ttft_s`。

### 7.3 验收标准
7.1 全部通过，两个仓库的回归均无新增失败。不带新字段的旧 sidecar 访存周期与改动前逐条一致；模型级仿真按 `genesim/docs/llama-2.md` 跑通并核对 `results/summary.json` 的关键字段。

## 八、对需求文档的修正记录
1. 需求 F2 要求在成本里扣掉被吸收节点的搬运。实测 `gml_bridge/from_fx.py:223` 已跳过这些节点，且 `analyze_ir` 的输入对不回 FX 节点。本设计改为比较融合开与关的整图测量。
2. 需求 F1 把通信字节数直接加入 `communication_time_s`。实测该值已由依赖边累加得到，再加会双计，且按 `nbytes` 求和漏掉 `2N` 的放大。本设计改为补齐缺失的边，沿原路径计时。
3. 需求 F4 把区间周期放在读取指令处。实测 LOADN 不携带区间，且一个算子的 trace 混合多个区间。本设计把分档移到 trace 生成处。
4. 前一版要给 all_reduce 补写回各分片的出边并接入 all_to_all。实测归约结果的消费者是残差节点，不存在写回分片的边，all_to_all 只在夹具里出现。改为只接入 all_gather 与 scatter。
5. 前一版用 `emit_oplevel_mlir` 做整图测量。实测它要求图已融合且只覆盖五类算子。改为逐算子对比。
6. PIM 时延由 trace 周期决定，不读 flops。计算维度的判据落在 trace 的循环次数上，而不是 sidecar 的 flops。
7. 前一版把并行性判据写成单分片循环次数减半。那是计算维度的量，与并行无关。实测分片各自钉到不同 VPU（`gene_sim_scheduler.py:1094`），并行体现在层墙钟小于各分片时延之和。
8. 区间分档从 sidecar 的 `bytes_by_zone` 改为在 `pimir_trace.py` 保留 `base_arg`。原因是一个算子的 trace 混合多个区间，按算子整体乘会算错，而 `base_arg` 在文本里还在。**这一条只覆盖算子内部的细分**：读 KV 缓存的算子不带能区分 KV 的 `base_arg`，所以算子级区间另从放置 sidecar 的 `zone` 取，归属由内存规划的 KV 规格（`KVRegionSpec.layers`）判定，与 `DPUPlan` 的划分一致。
10. 倍数取值的两头都有硬约束，最终定为权重 1 倍、KV 与激活 2 倍。下界：默认表的每字节周期就是配置的峰值带宽速率，倍数低于 1 会让 PIM 访存时延掉到峰值带宽下界之下——模型级仿真第一次跑就撞上 `memory service time is below its configured peak-bandwidth lower bound`。上界：激活超过权重的 2 倍，字节减半就盖不住区间变慢，违反需求第五节的访存反倒挂判据。
11. 需求 F2 的判据「融合开关开与关各导出一次」在旧实现下不可构造（导出侧无条件打标）。改为放置 sidecar 带 `fused_activations` 清单，没跑融合 pass 的图清单为空，判据成立。
9. 融合从整图测量改为给 `matmul_kernel` 传 `activation`。该参数已存在，配方之前没传。
12. `t_compute` / `t_memory` 的手算、计算受限放大 M/N/K、访存受限放大字节，以及两项相等时同放大的判据不纳入本轮：PIM 时延由 trace 周期决定，不消费这些 GPU roofline 输入。通信时延与并行性不属于此豁免；通信时延的手算持平点、字节增长与移除通信输入后的隔离性由 GeneSim `test_collective_edges.py` 覆盖，并行性由 `test_tp_shard_layer_makespan_is_below_the_serial_sum` 覆盖。融合导出链路以非空 sidecar 验证吸收节点成本归零：放置导出在写 sidecar 前先跑一次 `fuse_for_pim`（真实链路上原本没有融合 pass，清单恒为空），代价是仿真成本基线随之改变——32 个 SILU 被吸收，10 请求基线按当前代码于 2026-10-15 重跑，数字见实施文档 7.3。被吸收节点进仿真靠精化回写放置 sidecar 的 `absorbed_ops`：只清系数改不了 PIM 时延，它的时延取自 trace 周期。开与关的对照落在 `memory_service_time_s`（-20.42 秒）与 64 条空 trace 上，见实施文档 7.3。
13. 缺 `base_arg` 且没有算子级区间时按单一周期计，不再默认落到激活档。有 `zone` 时仍按算子级区间。
14. 被吸收的算子由精化回写的 `absorbed_ops` 决定，仿真侧跳过：系数清零只改导出量，PIM 时延取自 trace 周期，不跳过 trace 融合就进不了仿真。跳过的算子仍留一条空 trace 占住编号，执行侧不会因为没有 trace 报错。
15. 需求第五节的「账目」三条只剩一条未闭环：「三区之和等于总搬运」随第 8 条失效（区间不再按算子汇总字节，而是逐段落在指令上），本轮明确豁免。另两条已补断言：「切分轴的本地宽度之和等于全局宽度」由 `test_shard_widths_sum_to_the_global_width` 按切分轴逐项对账（按字节求和不成立，未切分一侧的激活在各片重复出现），「KV 写回不在通信项里」由 `test_kv_writeback_never_enters_the_communication_account` 覆盖（片内搬运走 `local_slice`，图上确有此边而通信账目里没有）。「四维覆盖」里「去掉一项的输入只该项归零」由 `test_removing_one_input_zeroes_only_its_dimension` 覆盖，其 roofline 半条随第 12 条豁免。
16. 补边分两种口径。不带序列维的边（`seq_len` 为 null）每个参与者各发一份完整张量，即 `参与者数 × nbytes`，由 `test_allgather_adds_cross_device_bytes` 按 2 与 4 个参与者锁死。带序列维的边系数是 `nbytes / seq_len` 再挂到 `Tq` 上，decode 步只收 `1/seq_len` 份，由 `test_collective_bytes_scale_with_query_length` 锁死。`plan_cost` 的数不同：它按主机星型计，all_gather 是收集一份加向每个目标回写、scatter 是收集一份加回写各片，分别是 `(2N-1)` 与 `2N` 倍。仿真里中转节点落在 GPU 上、每个参与者直发一份，没有那一次主机汇合，所以不取 `plan_cost` 的数。
17. A 路 kernel 由 Triton 生成，带不出激活。生产链路走 `--ir-level pimir` 时，`mark_fused_activation` 把 `activation = #pim.act_spec<kind = silu>` 标到 `tt.dot` 上：放置导出另存一份 `*.silu.pimir.mlir` 给仿真编 trace，精化时对捕获到的 kernel 同样标注再交给 `analyze_ir`。属性只带 kind，给正则消费，不进 MLIR verifier。写出与识别收拢在 `ir_cost.fused_act_spec_attr` / `has_fused_act_spec`。分块 GEMM 的激活按写回张量计一次，不记在 K 循环里的 `tt.dot` 上——记在那里会按累加器类型乘内层趟数，比 trace 多两个数量级。
18. 需求 2.4 的「两路对照」在融合算子上只覆盖同一份文本：ttir 侧 A 路 kernel 没有 `dma_store`，折激活那笔运算数进不了 TTIR sidecar。`test_pimir_sidecar_agrees_with_ttir_on_flops` 跳过放置 sidecar 里带融合角色的算子；同一份带 `act_spec` 的分块 GEMM 文本由 `test_fused_activation_flops_agree_across_ir_levels` 锁死 ttir 与 pimir 的运算数相等。
19. 折激活的写回只记最后一次 `dma_store`，与 `pimir_trace` 取 `outer_stores[-1]` 同口径。多写回的 kernel 由 `test_fused_activation_bills_only_the_last_store` 覆盖。
20. B 路注意力矩阵乘不接收 `fused_activation`。该字段只在精化过程内部出现，注意力不算融合清单里的投影，放置导出看不到它。

## 九、待确认事项
1. 周期分档取权重 1 倍、KV 2 倍、激活 2 倍。数值没有实测依据，先取这组让区间差异进入时延；激活与 KV 同为「按 token 反复搬运」的区间，档位相同是需求只要求「常驻 ≤ 反复」的结果；有实测后再改这一处。倍数下限是硬约束（不得低于峰值带宽速率），上限由反倒挂判据给出（不得高于权重的 2 倍）。
