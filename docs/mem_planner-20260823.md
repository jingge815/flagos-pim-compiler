# 内存管理器（问题 8）——20260823

## 一、本次改动概述

本次实现了技术方案「问题 8：内存管理」的全部内容：编译期把归属某台 DPU 的
权重区、KV 区、激活区在一份统一的蓝图里排出 MRAM offset，做容量校验，并把
权重 offset 回填进问题 2 产出的 `PIMTensorSpec`（问题 3 的通信计划表一直在
读这个字段，此前恒为 0）。这是"图编译器"三件套（问题 1 拆分 / 问题 2 切分 /
问题 8 内存）里最后一块，打通后问题 1→2→7→8 在真实 Llama-2-7B 上端到端验证
通过。

一句话概括：**权重区顺序打包、KV 区调用问题 7 的布局器、激活区做 liveness
分析 + 贪心装箱，三区拼起来做一次容量校验；prefill/decode 两张图共用权重和
KV 的 offset，激活区各算一份、取峰值。**

问题 6（主机编排器）的双图执行机制尚未实现，本次的两图验证用两个不同
`SEQ_LEN` 的独立 `torch.export` 撑起"两图联合规划"的机制，不代表真实的
KV 感知 decode 图；这一局限在第七节详细说明。

## 二、修改了哪些文件

| 文件 | 改动 | 行数 |
| --- | --- | --- |
| `memory/mem_planner.py` | 新增实现（原为空文件） | +296 |
| `tests/test_mem_planner.py` | 新增单元测试（手搭小图，14 条） | +243 |
| `tests/test_mem_planner_llama2_7b.py` | 新增真实 Llama-2-7B 端到端测试（8 条） | +209 |
| `docs/mem_planner-20260823.md` | 本文档 | — |

未改动 `contracts/`、`graph/`、`comm/`、`memory/kv_layout.py`、`backend/` 任何
既有代码——纯新增，靠 `node.meta` 与函数参数解耦，对问题 1/2/3/7 零侵入。

## 三、新增了哪些结构体和函数

### 结构体（`memory/mem_planner.py`）

| 结构体 | 关键字段 | 含义 |
| --- | --- | --- |
| `HwBudget` | `mram_bytes`、`align`、`sys_reserve_bytes` | 硬件输入：单 DPU 容量、DMA 对齐、系统预留 |
| `DPUPlan` | `weight`、`kv_base`、`act_base`、`act_prefill`、`act_decode`、`total`、`pending_readers_prefill`、`pending_readers_decode` | 一台 DPU 的完整内存蓝图，字段名与方案骨架一致 |
| `TransientTensor` | `name`、`size`、`produced_at`、`last_read_at`、`readers` | 一个临时激活张量的 liveness 区间 |

### 函数分组

| 分组 | 函数 | 干什么 |
| --- | --- | --- |
| 通用 | `bytes_of(local_shape, itemsize)` | 本地分片 shape × dtype 字节数 |
| 权重区 | `weights_of(dpu_nodes, dpu_id)` | 挑出归属该 DPU 的权重常量（`get_attr` 且 `spec.device=="dpu"`），键为权重名 |
| 〃 | `_pack_weights(dpu_id, prefill_nodes, decode_nodes, align)` | 两图权重按名字去重、校验切分一致、顺序打包+对齐，回填两图的 `mram_offset` |
| 激活区 | `transient_tensors(dpu_nodes, dpu_id)` | 挑出该 DPU 的临时激活张量，算出 liveness 区间（含 redistribute 隐式读者） |
| 〃 | `greedy_reuse(tensors, base, align)` | liveness + 贪心装箱：不冲突的张量复用同一 offset |
| 〃 | `pending_readers_of_reused_addresses(tensors, offsets, dpu_id)` | 对每个被复用的地址，产出复用前必须等待的读者列表 |
| 主入口 | `plan_dpu(dpu_id, prefill_nodes, decode_nodes, kv_specs, hw)` | 串联三区、回填 `mram_offset`/`kv_base`、容量校验，产出 `DPUPlan` |
| 报告 | `format_mem_plan(plans, hw)` | 把各 DPU 的蓝图格式化为可读文本，定位问题用 |

### 每个函数的功能细节

**`bytes_of(local_shape, itemsize) -> int`**
纯算术：`prod(local_shape) * itemsize`。所有区的字节数计算都过这一个函数，
不各自重复写乘法。

**`weights_of(dpu_nodes, dpu_id) -> dict[str, Node]`**
遍历一张图的全部节点，筛出 `op=="get_attr"` 且其 `node.meta["spec"].device
== "dpu"` 且 `dpu_id` 在其 `shard_map` 里的权重节点。键用 `node.target`
（权重名字符串，如 `"model.layers.0.self_attn.q_proj.weight"`），不用 Node
对象——因为 prefill/decode 两张图是两次独立 `torch.export`，同一份权重在
两张图里是不同的 Node 对象，若用对象去重会把同一份权重错算成两份。

**`_pack_weights(dpu_id, prefill_nodes, decode_nodes, align) -> (offsets, total_bytes)`**
1. 分别调 `weights_of` 拿到两张图各自的权重表，取并集按名字排序（结果可
   复现）；
2. 对每个名字，取两图中存在的节点（可能只在一张图里出现），比较它们的
   `local_shape`/`shard_dim` ——不一致直接 `raise ValueError`，因为两图理论
   上共享同一份切分，不一致即 bug，不允许静默择一；
3. 顺序打包：offset 从 0 开始，每份权重占 `align_up(bytes_of(...), align)`
   字节；
4. **回填**：把算出的 offset 用 `dataclasses.replace` 写回两张图里所有同名
   `get_attr` 节点的 `spec.shard_map[dpu_id].mram_offset`（`TensorShardDetail`
   是 frozen dataclass，不能直接赋值属性，要整个替换）。

**`transient_tensors(dpu_nodes, dpu_id) -> list[TransientTensor]`**
只收 `spec.device=="dpu"` 且 `spec.residency=="transient"`（权重是
`"pinned"`，不算在内）且 `dpu_id` 在 `shard_map` 里的张量节点。对每个这样的
生产者节点，找两类读者：
- **本地读者**：`node.users` 中未被 redistribute 标注命中的同图消费者
  （同 DPU 直接消费，零 host 交互）；
- **隐式读者**：redistribute 边的 `copy_from`——`node.users` 覆盖不到这类
  读取，要另外扫一遍全图节点的 `node.meta["redistribute"]`，找 `edge.src`
  指向本张量、且本 DPU 在 `edge.src_loc["dpus"]` 内的边，读者记为
  `f"redist:e{edge_id}"`。

`last_read_at` 取全部读者里最大的图位置（下标）；无读者时退化为
`produced_at`（生产后立即可复用）。

**`greedy_reuse(tensors, base, align) -> (offsets, region_end)`**
借 ExecuTorch 的 `greedy` 思路。按 `(-size, produced_at, name)` 排序（大的
先放，结果确定性可复现）。对每个张量，线性扫描已经开出的 slot：如果某个
slot 的容量够大、且它历史上占用过的所有区间都跟当前张量的
`[produced_at, last_read_at]` 不冲突（一方的写不早于另一方的最后读），就
复用这个 slot；否则在当前顶端新开一个 slot（新开的 offset 按 `align` 对
齐）。返回逐张量的 offset 表和这块区域的末尾地址。

**`pending_readers_of_reused_addresses(tensors, offsets, dpu_id) -> dict`**
把 `greedy_reuse` 判定复用的地址找出来（同一个 offset 被 ≥2 个张量占用）：
按生产时间排序，除最后一个占用者外，其余占用者的 `readers` 全部并入该地址
的 pending 表。键的形状是 `(("dpu", dpu_id), offset)`，跟方案里问题 6 生成
`ExecutionPlan` 时的查表方式（`pending_readers.get((("dpu", dpu), offset),
[])`）对齐，供未来直接使用。

**`plan_dpu(dpu_id, prefill_nodes, decode_nodes, kv_specs, hw) -> DPUPlan`**
主入口，串联三区：
1. 调 `_pack_weights` 拿权重区 offset 表和占用字节数 `off`；
2. `kv_specs[dpu_id].kv_base = off`，调问题 7 的 `build_kv_layout` 就地
   重建该 DPU 的 KV 布局，`off += kv_allocated_bytes`（不用 `kv_bytes` 公式
   重算，避免对齐 padding 累积侵入激活区）；
3. `act_base = off`；对 prefill_nodes/decode_nodes 各跑一遍
   `transient_tensors` + `greedy_reuse`（共享同一个 `act_base`，各自独立
   装箱），`total = max(两图末尾地址)`；
4. 容量校验：`total <= hw.mram_bytes - hw.sys_reserve_bytes`，不满足直接
   抛 `ValueError`（不做方案里 `[阶段2]` 的自动重切反馈）；
5. 打包成 `DPUPlan` 返回。

**`format_mem_plan(plans, hw) -> str`**
按 DPU 编号顺序打印每台的权重/KV/激活字节数、总量、离预算上限的余量，方便
肉眼核对三区是否符合预期、余量是否合理。

## 四、整体应该怎么用

```python
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu, format_mem_plan

# 1. 分别导出 prefill / decode 两张静态图，各走一遍问题 1/2
shard_config = llama_shard_config(NUM_DPUS, num_heads=..., num_kv_heads=..., ...)
for gm in (prefill_gm, decode_gm):
    partition_graph(gm)
    propagate_specs(gm, shard_config)

# 2. 用问题 2 的 k_proj 切分结果建 KV 规格（kv_base 占位，plan_dpu 会覆写）
kv_specs = kv_specs_from_placement(k_proj_spec, num_layers=..., num_kv_heads=...,
                                    num_q_heads=..., head_dim=..., max_seq=...,
                                    dtype_bytes=2, kv_base=0)

# 3. 每台 DPU 各调一次 plan_dpu——副作用：回填两图权重节点的 mram_offset，
#    以及 kv_specs[dpu_id] 的 kv_base/kv_off/kv_allocated_bytes
hw = HwBudget(mram_bytes=8 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
plans = {d: plan_dpu(d, list(prefill_gm.graph.nodes), list(decode_gm.graph.nodes),
                      kv_specs, hw) for d in range(NUM_DPUS)}

print(format_mem_plan(plans, hw))  # 核对三区大小、总量、余量
```

调用之后：
- 两张图里所有权重节点的 `spec.shard_map[dpu_id].mram_offset` 已经是真实
  地址，问题 3 的 `build_comm_plan` 直接可用；
- `kv_specs[dpu_id]` 已经是完整的 `KVRegionSpec`（`kv_base`/`kv_off`/
  `kv_allocated_bytes` 都填好），可以直接喂给问题 7 的 `PIMStaticKVCache`；
- `plans[dpu_id].act_prefill` / `act_decode` 是两张图各自的激活 offset 表，
  `pending_readers_prefill` / `pending_readers_decode` 留给问题 6 未来生成
  `ExecutionPlan` 时查表转换成命令的 `waits`。

## 五、整体流程图

```
编译期（每台 DPU 各走一次）：

  两张导出图（prefill SEQ_LEN=P，decode SEQ_LEN=1）
        │ 各自过 partition_graph + propagate_specs（问题 1/2）
        ▼
  weights_of(图, dpu_id) ──┬── prefill 权重表 ──┐
                           └── decode  权重表 ──┤
                                                 ▼
                              按名字去重 + 校验两图切分一致
                                                 ▼
                              顺序打包 + 对齐 → weight offset 表
                                                 │ 回填两图 mram_offset
                                                 ▼
                              kv_specs[dpu_id].kv_base = off
                                                 ▼
                              build_kv_layout（问题 7）→ kv_allocated_bytes
                                                 ▼
                              act_base = off
                        ┌──────────────┴──────────────┐
                        ▼                              ▼
          transient_tensors(prefill图)        transient_tensors(decode图)
                        ▼                              ▼
              greedy_reuse（共享 act_base）    greedy_reuse（共享 act_base）
                        ▼                              ▼
              act_prefill, end_p             act_decode, end_d
                        └──────────────┬──────────────┘
                                       ▼
                          total = max(end_p, end_d)
                                       ▼
                    容量校验 total <= mram_bytes - sys_reserve
                                       ▼
                              DPUPlan（供问题 3/6/7 共用）
```

## 六、做了哪些测试和结果

### 单元测试：`tests/test_mem_planner.py`（14 条，手搭附录 A 小图，全部通过）

| 测试 | 验证点 |
| --- | --- |
| `test_bytes_of` | 字节数手推值 |
| `test_weights_of_picks_dpu_owned_get_attr` | 只挑 `get_attr` 权重，不含 placeholder |
| `test_pack_weights_offsets_and_alignment` | offset 手推值 + 对齐 padding + 两图回填一致 |
| `test_pack_weights_rejects_cross_graph_shape_mismatch` | 两图同名权重 shape 不一致直接抛错 |
| `test_transient_tensors_local_and_redistribute_readers` | 本地读者与 redistribute 隐式读者（`redist:eN`）都被正确识别 |
| `test_transient_tensors_no_readers_defaults_last_read_to_produced` | 无读者时 `last_read_at` 退化为 `produced_at` |
| `test_greedy_reuse_shares_offset_for_disjoint_lifetimes` | 生命周期不重叠的张量复用同一 offset，重叠的各开新 offset |
| `test_greedy_reuse_processes_largest_first_so_slots_always_fit_later_tensors` | 大小降序处理，后到的小张量能复用大 slot |
| `test_greedy_reuse_aligns_new_slots` | 新开 slot 的 offset 按 `align` 对齐 |
| `test_pending_readers_only_for_shared_addresses` | 只有真被复用的地址才产出 pending 表，独占地址不出现 |
| `test_plan_dpu_regions_do_not_overlap` | 权重/KV/激活三区地址范围互不重叠 |
| `test_plan_dpu_weight_offsets_shared_across_both_graphs` | 两图权重 offset 一致 |
| `test_plan_dpu_raises_when_over_budget` | 预算过小时抛 `ValueError` |
| `test_format_mem_plan_printable` | 报告可打印 |

**结果**：`14 passed`。

### 真实 Llama-2-7B 端到端测试：`tests/test_mem_planner_llama2_7b.py`（8 条，全部通过）

| 测试 | 验证点 |
| --- | --- |
| `test_weight_region_matches_hand_computed_bytes` | 权重区总字节 ≥ 该 DPU 全部本地分片字节和（对齐后）的手推交叉验证 |
| `test_weight_offsets_shared_between_prefill_and_decode` | 同名权重（`q_proj`）在两张图上的 `mram_offset` 相等 |
| `test_regions_are_disjoint_and_ordered` | 权重终点 < `kv_base`；`kv_base + kv_allocated_bytes == act_base`；激活 offset 都落在 `[act_base, total)` |
| `test_capacity_check_passes_with_realistic_budget` | 4GiB/DPU 预算下真实 7B 权重分片能装下 |
| `test_capacity_check_rejects_too_small_budget` | 1MiB 预算下装不下，正确抛错 |
| `test_weight_offset_is_a_real_usable_address_on_numpy_backend` | 按回填的 offset 在 `NumpyBackend` 上真实写入 `q_proj` 某台本地分片、读回，与 torch 参考逐字节一致——证明 offset 是可用的真实地址，不只是数字 |
| `test_kv_region_lands_at_planned_kv_base_and_stays_usable` | `plan_dpu` 回填的 `kv_base` 落地后，问题 7 的 `update`/`read_tile` 仍按预期地址正常工作 |
| `test_format_mem_plan_printable` | 报告可打印，打印了实际的三区字节数 |

**结果**：`8 passed`。实测每台 DPU（8 台均分 7B fp16 权重）：权重区
≈1575.76 MiB，KV 区 16.00 MiB（`max_seq=256`），激活区 ≈0.62 MiB，合计
≈1592.38 MiB；给 4096 MiB 预算（扣 64 MiB 系统预留后 4032 MiB 可用）时
余量 ≈2439.62 MiB。

### 全仓回归

```
python -m pytest tests/ -q --deselect ".../test_refined_ir_preserves_structure[llama2_7b_flagtree.ir]" \
                          --deselect ".../test_refined_ir_preserves_structure[llama2_7b_pimir.ir]"
144 passed, 2 deselected
```

（那两条 deselect 是本次改动之前就存在的失败，`ir_cost.py` 的 IR 夹具缺
`max_seq` 字段，与内存管理器无关。）

净增删：`memory/mem_planner.py` +296，`tests/test_mem_planner.py` +243，
`tests/test_mem_planner_llama2_7b.py` +209，本文档另计。不改动任何既有文件。

## 七、当前还存在哪些问题

- **两图机制未经真实 decode 图验证**：真正的 KV 感知 decode 图（每步只算
  一个新 token、复用 KV cache 的那种）要等问题 6 的编排器把 prefill/decode
  两条路径的 FX 图重写出来才存在。本次测试用 `SEQ_LEN=16` 和 `SEQ_LEN=1`
  两次独立 `torch.export` 模拟"两张不同形状的图"，只验证了"两图共用权重/KV
  offset、激活区各算一份"这套机制本身是对的，不代表真实解码场景下的激活
  区大小或 liveness 结果。
- **`pending_readers_prefill`/`pending_readers_decode` 尚无消费方**：问题 6
  的 `ExecutionPlan` 生成器（`runtime/exec_plan_gen.py`）还是空文件，这两张
  表按方案格式产出、结构已用单测验证，但还没有真实场景把它们转换成命令的
  `waits` 字段。
- **激活区按满预留，不做进一步紧凑复用**：`greedy_reuse` 已经做了生命周期
  复用，但第 1 阶段所有张量按 `max_seq`/固定 shape 的满尺寸走，没有按实际
  运行时长度动态收缩——这是方案明确写的 `[阶段2]` 范畴，当前不做。
- **不做自动重切反馈**：方案提到的"装不下就反馈给切分 pass 重切"的定点
  迭代（`[阶段2]`）未实现；容量超限时直接抛错，需要人工调整
  `ShardConfig`/`HwBudget` 重跑。
- **不处理 view/别名**：第 1 阶段 DPU 白名单（`linear`/`add.Tensor`/
  `mul.Tensor`/`tanh`）不产出 `transpose`/`reshape`/`permute` 之类的视图算
  子，所以 `transient_tensors` 没有实现方案里"未物化纯视图与源张量共享
  存储 id"的分支。等白名单扩宽出现 view 算子时需要补上。
- **`sys_reserve_bytes` 仍是外部传入的常量**：真实 kernel 二进制大小、
  runtime 栈、WRAM staging 余量、workspace 目前都由调用方在 `HwBudget` 里
  一次性给一个保守估计，未接入问题 5 的算子编译器产物做精确核算（方案里
  写明第 1 阶段本就如此，SDK 到位后再精确化）。
- **`_pack_weights` 的跨图校验只查 `local_shape`/`shard_dim`**：没有校验
  `placement.kind`（Shard/Replicate/Partial）是否一致；理论上两图共享同一
  `ShardConfig` 时不会出现这种不一致，但如果未来两图允许使用不同
  `ShardConfig`，这里需要补一条校验。
