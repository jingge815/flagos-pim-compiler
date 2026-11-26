# 技术设计文档：多 batch 全链路支持与性能评估

> 文档编号：design-multi-batch-20261121  
> 创建日期：2026-11-21  
> 关联需求文档：request-multi-batch-20261121（`docs/request-multi-batch-20261121.md`）  
> 关联项目：flagos-pim-compiler（主仓）、genesim（仿真器）；FlagTree 预计零改动

## 一、设计概述

### 1.1 需求背景回顾

　　从大模型到图编译、算子编译、genesim 仿真的全流程已走通（标准形态 tp16_pp4/64 核），但整条链路只考虑单 batch：图导出、KV 布局、内存规划、编排器、成本链、仿真调度都把 batch 写死为 1，跨请求执行线性叠加，PIM 多 batch 的权重复用收益在模型里结构性不存在。需求要求以 closed batch（等长、同相、整组进出）先行打通 B>1 全链路，建立收益与代价并计的评估模型，验证权重复用/流水/并行三项优化，并交付 batch 扫参与最优配置输出；本期不动 cascade 级联与 CEVA 厂商路径，抢占仅解析建模。

### 1.2 设计目标

　　让 batch 从编译入口贯穿到仿真报告：`compile_llama2(batch_size=B)` 导出 `[B,S]`/`[B,1]` 图，KV/内存按槽位规划，编排器跑通 B>1 并与 PyTorch 对齐，算子编译与成本链按 `(phase,B)` 出真实 M>1 产物，genesim 以「组即合并请求」执行 closed batch 并输出时间构成归因，扫参入口一条命令产出曲线与推荐。对应需求 R1–R9（R1–R7 P0，R8–R9 P1）。

### 1.3 设计范围

- 包含范围(1句话)：closed batch 的编译、执行、仿真、评估、扫参全链路（pim-compiler 与 genesim 两仓）。
- 不包含范围(1句话)：连续 batching/真抢占、cascade 级联、CEVA 厂商路径的 batch 化，以及 FlagTree kernel 改动（仅当前置实验证伪时另立项）。

## 二、现状基线

　　为把 batch 维贯穿三仓，现有代码在图导出、KV 布局、内存规划、执行编排、HAL 绑定、成本口径、仿真调度各层均按单请求闭环实现、batch 全部钉死为 1；同时容量台账、tile_m 折算、成本截距三项资产已经就位，可直接复用。核心调用链是：`scripts/run_full_pipeline.py` 的 (B/C/D) 步调 `scripts/export_pp_placement.py` 子进程，其内部经 `genesim_bridge/placement_export.py:920` 的 `export_placement_for_strategy` 调 `runtime/compile.py:214` 的 `compile_llama2`，产出 placement sidecar 后 (A2) 步按 sidecar 精化成本 IR，(E) 步把配置指向精化 IR 跑 genesim 仿真。逐层现状如下表。

| 层 | 现状（实测证据） | 就绪资产 |
| --- | --- | --- |
| 图导出 | `runtime/compile.py:78` `[1,seq]` 静态导出、无 `dynamic_shapes`；mask `[1,1,S,S]`（`:66-68`） | 每档 B 一次静态导出即可，无需动态轴 |
| KV 布局 | `memory/kv_layout.py:40` 键 `(layer,head,which)`；`:59-62` 容量无 batch 项；`:181-192` 单 pos 寻址 | `build_kv_layout` 线性分配结构可直接加槽位外层循环 |
| 内存规划 | `memory/mem_planner.py:317-321` 激活按 batch=1 图形状；`:344-358` 容量判据 | `:458-472` 逐请求 KV 峰值台账、`:527-591` `tensor_pu_usage(concurrency=)` |
| 执行器 | `runtime/executor.py:92-94` `_as_batch_tensor` 写死 `[1,len]`；`:41-89` 单请求循环；`DecodeState` 标量 `valid_len`（`:15-22`） | 执行计划形状自动跟随图（`runtime/exec_plan_gen.py` 从 spec `local_shape` 取形状），无需改本体 |
| HAL | `backend/hal_numpy.py:281-293` `bind_inputs(values, pos)` 单标量 pos | 绑定机制本身与形状无关，只需扩展 pos 类型 |
| 成本链 | `genesim_bridge/placement_export.py:257` 代表编译钉死 `M=1`；`cost_extractor.py:355-362` 符号表只认 `Tq/Tp`；`:611-612` 仅 prefill/decode 两个代表点 | 拟合截距=权重字节（`:393-395`）；kernel M 分块+掩码+tile 搜索 M 对称（FlagTree 零改动） |
| genesim | `src/config_loader.py:1391-1395` `execution_microbatch_size≠1` 抛错；`gene_sim_scheduler.py:9301` task_key 含 request_index；`:10374` FCFS 准入；`src/backend/pim_backend.py:61` 执行=单轨迹×repetitions | `src/vpu/vpu.py:841-851` 已按 `ceil(M/tile_m)` 折算重放；`_extract_query_count`（`:818-839`）天然把输入首维乘积当行数 |

　　两个关键机制决定设计走向。其一，genesim 的运行时形状由 `step_context` 绑定（`gene_sim_scheduler.py:8641-8670`：prefill `tq=query, tp=initial`，decode `tq=1, tp=initial+step`），算子形状按符号 `Tq/Tp` 解析——只要合并 decode 请求的 `tq` 取 B，decode 图 GEMM 的输入形状就变成 `[B,·]`，`_effective_repetitions` 自动得到 `ceil(B/tile_m)`，权重摊薄随之发生。其二，genesim 的容量按请求生命周期峰值求和（`src/pim/capacity.py:296-303` top-N），合并组必须按 B 倍计 KV，否则容量被低估。

## 三、总体设计

### 3.1 总体设计概述

　　要解决的问题是把「B 个等长请求」变成一条可编译、可执行、可计价、可归因的执行单元，当前不足是 batch 在六层各自断链，且直接在万行调度器里做跨请求任务合并侵入太深。技术路线是一句话：**closed batch 组在两条链路里都表现为「合并请求」——prefill 一次合批执行 [B,S]（M=B×S），decode 步合并为 Tq=B**。编译侧 `compile_llama2(batch_size=B)` 导出 `[B,S]`/`[B,1]` 图（batch 维 replicate，数值路径真实合批）；成本侧按相位出真实产物——decode 档逐 B 编 M=B（拿到 tile_m>1 的权重摊薄）、prefill 档逐 B 编 M=B×S（`prefill_b{B}` 产物，GEMM 权重装载份数 = ceil(B×S/tile_m)，修正当前 M=1 产物把 prefill 权重搬运放大 S 倍的问题）；仿真侧把 B 个 trace 请求在准入时合成一个组请求。prefill 合批（用户 2026-11-25 裁决，round18 问题 1）：与图编译/编排器一次执行 [B,S] 的真实语义一致——genesim 的 GEMM 按 M=B×S 计价，注意力按 B 个块对角序列分别计价（行轴 Tq=B×S 表达 GEMM 行数与 q 行总数，KV 长度轴保持每槽 S、KV 操作数按槽数放大；不能把全局 Tq 与 KV 长度都改成 B×S——那会把块对角的 B 份 S×S 放大成 (B×S)²、多算 B 倍）。早前「prefill 不合并」的依据 `ceil(B×S/tile_m)=B×ceil(S/tile_m)`（S≥tile_m）已被真实算子编译证伪：等式对非整尾不成立（round17 实测 S=17、tile_m=8 时合批 9.107 份 vs 逐槽 12.155 份）。decode 合并的口径不变：GEMM 走 M 维计费拿到摊薄，GEMV/SOFTMAX 按行数 B 线性放大（注意力本无跨请求复用）。步级绑定是算子与传输计费的单一真源（round25 R25-01 修正）：依赖边、集合通信与首次使用计费消费与算子绑定同一份（组行数、每槽 KV 长度）——行轴 Tq=组行数（合批 prefill 一步 B×S、合并 decode B、逐槽步 S）；与 Tq 轴配对的 KV 长度保持每槽（Tq(Tp+Tq) 项不放大 KV 轴）；不带 Tq 轴的 KV 项（Tp、Tp+Tq）按槽数放大（组 KV = B 份每槽，与轨迹层 kv_slots 同一条规则）。早前步级形状的 prefill 分支返回单槽 S、依赖传输少算 B 倍。

　　对于「genesim 的 batch 语义落在哪一层」，比较过两套候选：

| 候选 | 做法 | 侵入度 | B=1 兼容 | 结论 |
| --- | --- | --- | --- | --- |
| A. 组即合并请求（选定） | 准入时把 B 个等长请求合成一个组请求，占一个 request_index，decode 步 `tq` 取合并行数，容量按 B 倍计 | 低：准入与 step_context 两处改动，DAG/派发/指标机制全部复用 | 组大小 1 退化为现路径 | 采纳 |
| B. 派发时跨请求合并 | 保留 B 个独立请求，派发层把同 `(forward_step, op_id)` 的就绪任务合并成批任务 | 高：依赖跟踪、部分合并、尾延迟处理全要重写 | 双路径并存风险 | 落选：与 closed batch 语义错配，复杂度留给连续 batching 场景 |

　　对于「KV 多槽布局的键结构」，比较过三套候选：

| 候选 | 做法 | 结论 |
| --- | --- | --- |
| 槽位外层主序（选定） | `kv_off` 键加 slot 轴且 slot 为最外层循环，每槽整段 KV 连续 | 采纳：抢占建模要按槽迁移 KV，槽连续最友好；`kv_access` 加 slot 参数缺省 0，B=1 等价 |
| 每槽独立 KVRegionSpec | B 份 spec 各自 build | 落选：`kv_specs` 是 `dict[dpu_id]`，改键波及全部消费方 |
| 键加 request id | 与槽位同构但语义模糊 | 落选：closed batch 是编译期槽位，不是运行时请求 |

　　对于「成本链怎么表达 batch」，比较过两套候选：

| 候选 | 做法 | 结论 |
| --- | --- | --- |
| 相位产物 + 拟合求值（选定） | placement 侧逐相位逐档出产物（decode 档 M=B、prefill 档 M=B×S，合批口径见 3.1/§八.4），系数沿用两点拟合在 `Tq=B` 处求值，符号表不动 | 采纳：与「组即合并请求」同构，genesim 侧 `Tq` 语义与 extensions sidecar 结构零改动 |
| 新增 Tb 符号维 | IR 形状加 batch 维、符号表加 `Tb`、调度器按请求维解析 | 落选：要求 genesim IR 契约加维，波及 `model_ir`/trace/调度三处 |

　　按此路线，一次 `--batch 4` 运行的整体流程如下图（【新增】/【修改】/【原有】标注改动性质）。

```text
run_full_pipeline --batch 1,2,4,8（【新增】扫参循环）
  ├─(A)  model_parser 图骨架（原有）
  ├─(B)  compile_llama2(batch_size=B)（【修改】导出 [B,S]/[B,1]、KV B 槽、激活×B）
  │        └─ KV/内存规划（【修改】槽位布局 + 容量表输出）
  ├─(C)  算子编译（【修改】decode 档 M=B、prefill 档 M=B×S 逐档；FlagTree 零改动）
  ├─(D)  placement sidecar（【修改】pimir_by_phase 逐相位逐档产物索引）
  ├─(A2) refine 成本 IR（原有——两点拟合系数，Tq=B 处求值）
  └─(E)  genesim 仿真（【修改】closed batch 组准入 + 合并计费）
           └─ batch_metrics.json（【新增】时间构成归因 + 抢占解析项）
                └─ 扫参汇总报告（【新增】曲线 + 瓶颈 + 推荐）
```

### 3.2 与现有逻辑的衔接

　　接入点全部挂在既有流程的段边界上，不新增平行链路。第一处是 `export_placement_for_strategy`（`genesim_bridge/placement_export.py:920`）与 CLI `scripts/export_pp_placement.py`：加 batch 档列表参数，透传给 `compile_llama2(batch_size=B)` 并驱动 `_compile_tile` 出逐相位逐档产物（见 4.5）；调优入口 `tune/steps.py:1047` 不改（缺省 1 即旧行为）。第二处是成本精化步：不改——系数沿用两点线性拟合（`cost_extractor.py:405-445`），仿真在 `Tq=B` 处求值即覆盖合并行数，extensions sidecar 结构不变。第三处是 genesim 仿真配置：`scheduler.execution_microbatch_size` 从禁令参数变为组大小参数（B=1 语义不变，命名沿用现有配置项、其注释原意即张量批），`ceva_sim_config`（`run_full_pipeline.py:470-475`）在写 `compiler_placement_file` 的同一位置写入组大小。第四处是编排器数值路径：`run_decode_loop` 的输入从单 prompt 变 prompt 列表，HAL 的 `pos` 绑定从标量扩展为按槽元组。

### 3.3 模块划分与职责边界

　　涉及 pim-compiler 的 `contracts/`（契约先行）、`runtime/`（导出与执行）、`memory/`（KV 与内存）、`genesim_bridge/`（成本链与 sidecar）、`scripts/`（入口与扫参），以及 genesim 的 `src/config_loader.py`（组参数校验）、`src/scheduler/gene_sim_scheduler.py`（组准入与合并计费）、`src/pim/capacity.py`（组容量倍数）、`src/main.py`（batch 指标输出）。职责边界：batch 的形状真源只落在 `compile_llama2` 的导出图上，`exec_plan_gen`/`comm/plan` 不感知 batch（形状自动跟随）；槽位语义只落在 `KVRegionSpec`，内存规划经 `kv_allocated_bytes` 间接感知；「组」概念只落在 genesim 准入层，DAG 派发层不感知；抢占代价只出现在报告解析项里，不进事件引擎。容易重叠的一处：`mem_planner` 逐请求台账与 B 槽物理布局给出同一个数（B×S×per_token），约定以物理 `kv_allocated_bytes` 为容量判据真源，台账仅用于与 genesim 审计对账。

## 四、模块详细设计

　　整体问题按数据流拆成九个功能点：4.1–4.4 是 pim-compiler 侧的形状与执行链，4.5 是成本链，4.6–4.7 是 genesim 侧的执行与归因，4.8–4.9 是优化验证与扫参交付。上游功能点是下游的输入，依赖关系见 §六。

### 4.1 多 batch 图导出与编译入口

　　batch 维必须从导出源头进入，否则下游全部无从谈起。现有 `export_annotated_graph`（`runtime/compile.py:73-85`）用 `[1,seq]` 输入做 strict 静态导出，`compile_llama2`（`:214`）没有 batch 参数；需要给两者加 `batch_size`（缺省 1），每档 B 一次静态导出：input_ids `[B,S]`/`[B,1]`，position_ids `[B,S]`/`[B,1]`（各槽同长、值相同但显式展开），causal mask 保持 `[1,1,S,S]`/`[1,1,1,1]` 广播——各槽是独立等长序列，共享掩码语义正确且零开销；已在真实 `LlamaModel`（transformers 4.57）实测：`[B,S]` 输入配共享掩码的 batched 前向与逐样本逐位一致（diff=0.0）。`run_full_pipeline.py` 加 `--batch`（单值或逗号列表）传给 export 子进程（refine 步不改）。batch 维不参与 TP/PP 切分（`propagate_specs` 按现有维切，首维 replicate 自动成立）。该设计对应需求 R1，代码位于 `runtime/compile.py:73-85,214-239` 与 `scripts/run_full_pipeline.py:791`。

```python
# 原有（runtime/compile.py:78）
input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

# 目标
input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
```

### 4.2 KV cache 槽位化布局

　　多 batch 下每个槽位需要独立 KV 区域与有效长度，否则 decode 读写互相覆盖。执行期 KV 热路径（调用链核实）：`sdpa_kv_info`（`runtime/kernels.py:1238`）编译期把 per-(layer,head) 的 k_off/v_off 烘进命令 payload，`sdpa_kernel`（`:1332-1396`）运行时按 `hal.bound_pos`（`:1353`）写当前行、读整段历史、施加 `decode_mask/prefill_mask` 掩码（`:1391-1393`），K/V 索引硬编码槽 0（`q[0,head,t]`，`:1361-1364`）；`kv_access`/`PIMStaticKVCache` 仅被测试消费（`runtime/executor.py:11` 对其 import 是死引用，顺带清理）。布局层给 `KVRegionSpec`（`memory/kv_layout.py:28-41`）加 `num_slots: int = 1`（缺省 1 等价改动前），`build_kv_layout`（`:65-77`）最外层加 slot 循环，`kv_bytes` 乘 `num_slots`，`kv_access`/`PIMStaticKVCache.update`/`read_tile` 加 `slot: int = 0`（缺省 0 等价改动前），`decode_mask` 接受逐槽 `valid_lens` 返回 `[B, max_seq]`；热路径同步改：`sdpa_kv_info` 的 payload 记逐槽偏移（槽位主序下即首槽偏移+槽步长），`sdpa_kernel` 按 `q[b, head, t]` 逐槽写/读/掩码。`kv_specs_from_strategy` 透传 `num_slots=batch_size`。该设计对应需求 R2，代码位于 `memory/kv_layout.py` 与 `runtime/kernels.py:1238-1396`。

```python
# 原有（memory/kv_layout.py:59-62）
return (2 * len(spec.layers) * spec.max_seq
        * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes)

# 目标（乘槽位数）
return (2 * len(spec.layers) * spec.max_seq
        * len(spec.kv_heads) * spec.head_dim * spec.dtype_bytes * spec.num_slots)
```

### 4.3 内存规划 batch 缩放与容量表

　　规划层要能回答「这一档 B 装不装得下、卡在哪一区」。激活区大小取自导出图 transient 张量的 `local_shape`（`memory/mem_planner.py:317-321`），4.1 落地后形状自动变 `[B,·]`，规划随之缩放，无需改 `plan_dpu` 本体；KV 区字节数经 4.2 的 `kv_allocated_bytes` 自动乘 B。需要新增的是容量表输出：`format_capacity_table(compiled)` 按 DPU 打印「权重/KV/激活/内核 tile/合计/预算」六列与 (B, seq) 档位标注，挂在 `DPUPlan` 已有字段上（`:366-377`），沿用 `format_kv_layout` 惯例；超限仍走 `:354-358` 硬失败。逐请求台账（`:458-472`）在 closed batch 下天然收到 B 条目（trace 有 B 个请求），与物理布局对账。该设计对应需求 R3，代码位于 `memory/mem_planner.py`。

### 4.4 编排器批量执行与数值对齐

　　编排器要把一个 closed batch 组从头跑到尾并与 PyTorch 对齐，这是全链路数值锚点。现有 `run_decode_loop`（`runtime/executor.py:41-89`）单请求串行、`_as_batch_tensor`（`:92-94`）写死 1、`DecodeState` 只有标量 `valid_len`（`:15-22`）、HAL 每次绑单标量 pos（`backend/hal_numpy.py:281-284`）。需要把 `DecodeState` 改为 `valid_lens: tuple[int, ...] = (0,)` 并保留 `valid_len` 只读属性（返回首槽，B=1 调用方零改动）；`run_decode_loop` 的 `prompt_ids` 接受 `list[int]`（B=1 原行为）或等长 `list[list[int]]`（不等长直接抛 ValueError），prefill 喂 `[B, prompt_len]`，decode 每步喂按槽构造的 `[[t₀],[t₁],…]`（`[B,1]`）、position_ids 用 `torch.tensor(state.valid_lens).reshape(-1, 1)` 逐槽取值（closed batch 等长推进时各槽相同，但按槽取值保持语义一般性），logits 按槽采样（`np.asarray(logits)[b, -1]`）；EOS 逐槽判定，已结束槽喂 pad token 占位、输出丢弃（closed batch 整组进出、槽位不提前退出，各槽 `valid_len` 锁步推进）；`hal.bind_inputs` 的 `pos` 接受 `int | tuple[int, ...]`，`bound_pos` 透传类型，消费方即 `sdpa_kernel`（`runtime/kernels.py:1353`）按槽迭代。执行计划与通信计划不改（形状跟随图）。数值对齐沿用现有 fixture 对拍模式（`tests/test_executor_llama2_7b.py:126-141`），B=2/4 各建一例。该设计对应需求 R4，代码位于 `runtime/executor.py`、`backend/hal_numpy.py:281-293`、`runtime/kernels.py`。

```python
# 原有（runtime/executor.py:92-94）
def _as_batch_tensor(ids: list[int]):
    return torch.tensor([ids], dtype=torch.long)

# 目标（入参二选一：单序列 list[int] 保持原行为；B 行等长 list[list[int]]）
def _as_batch_tensor(ids):
    rows = [ids] if isinstance(ids[0], int) else ids
    return torch.tensor(rows, dtype=torch.long)
# decode 每步按槽构造：_as_batch_tensor([[t] for t in last_tokens_of_slot])
```

### 4.5 成本链 batch 代表点

　　成本必须按真实 M>1 编译产物计价，禁止线性外推。genesim 实际执行的产物由 placement sidecar 指定（`compiler_placement_file` → `gene_sim_scheduler.py:525-536` 装载进 `:348-375` 的 op_id 键控台账：pimir_path/pimir_sha256/zone 等，TP 展开算子按路径继承 `:361-363`），其代表编译钉死 M=1（`placement_export.py:254-259`），实测该产物 `"pim.tile-m" = 1`（抽查 `.opcompiler_cache`）——`ceil(M/tile_m)` 摊薄机制（`vpu.py:841-861`）从未在真实链路触发。做法是 `_compile_tile` 按相位出产物：decode 档逐 B 编 `M=B`（tile 搜索在 M>1 下可给出 tile_m>1，摊薄随之生效；静态形状无需占位维缩放），prefill 档逐 B 编 `M=B×S` 合批产物（与图编译一次执行 `[B,S]` 的真实语义一致，GEMM 权重装载份数 = `ceil(B×S/tile_m)`；当前 M=1 产物被 prefill 复用，权重搬运放大 S 倍）；缓存键 `(local_in, local_out, activation)` 加 M（`:243`）。sidecar 算子条目从单 `pimir_path` 扩为 `pimir_by_phase`：`{"prefill_b1": {path, sha256, kernel_tile_n, tile_m}, "prefill_b4": {...}, "decode_b1": {...}, "decode_b4": {...}}`，`prefill` 键保留为 prefill_b1 的同值副本（服务逐槽 micro-batch 策略与旧产物）、顶层旧字段保留为 decode_b1 的同值副本（旧消费方零改动）；genesim 侧台账键从 `op_id` 扩为 `(op_id, phase_key)`（装载循环与 TP 继承同改），执行分块按相位的 `kernel_tile_n` 绑定（`vpu.py:767-789` 校验所需）。成本系数无需逐档重测：两点线性拟合（`cost_extractor.py:405-445`，截距=权重字节）在 `Tq=B` 处求值即覆盖合并行数，`_resolve_dim` 符号表（`:355-362`）零改动。前置实验（同一 linear 编 M=1 与 M=8：断言 M=8 产物 tile_m>1 且权重搬运份数 ≤ `ceil(8/tile_m)`）作为本功能点的第一个测试落地。该设计对应需求 R5，代码位于 `genesim_bridge/placement_export.py:225-293` 与 genesim `src/scheduler/gene_sim_scheduler.py:348-375,525-536`。

```python
# 原有（genesim_bridge/placement_export.py:257-259）
request = OpCompileRequest(op="linear",
    arg_shapes=[(1, local_in), (local_out, local_in)], ...)

# 目标（按相位出产物：decode 档逐 B 的 m_rows=B，prefill 档逐 B 的 m_rows=B×S）
request = OpCompileRequest(op="linear",
    arg_shapes=[(m_rows, local_in), (local_out, local_in)], ...)
```

### 4.6 genesim closed batch 组执行

　　仿真器要让 B 个等长请求合成一个组、decode 步以合并行数计费，使权重摊薄出现在数字里。形状绑定的真实机制（调用链核实）：每个就绪任务经 `bind_runtime_op`（`gene_sim_scheduler.py:9364-9378`）调 `step_context`（`:8620-8670`）构造携带步级 tq/tp 的合成 `runtime_request`，再由 `_get_operator_with_runtime`（`:584-586` 的 `shape_dims`，`lkv = tp + tq`）解析符号形状——合并 decode 只需合成请求携带 `tq=B`，形状即变 `[B,·]`；注意力类算子（GEMV_SCORE/GEMV_CONTEXT/SOFTMAX/MASK）的 lkv 在组请求下按 `tp+1` 覆盖（落点 `:584`，否则 KV 搬运高估约 B/S，B=8、S=128 时约 5%）。生命周期同为局部函数：步链在 `:10016-10024`（`next_step < total_forward_steps(request)`），组请求把 `total_forward_steps`（`:8601-8603`，现值 `output_length`）改为 `prefill 步数 + output_length - 1`，`step_context` 按步号映射相位：缺省 prefill 合批一步（`Tq=B×S`、lkv 每槽 S、`kv_slots=B`，phase_key=`prefill_b{B}`——GEMM 按 M=B×S 计价、注意力按 B 个块对角序列分别计价，与 3.1/§八.4 的合批口径一致），`>=prefill 步数` 映射为合并 decode；`pipeline_overlap` 开启时切换为逐槽 micro-batch 策略（`slots` 个 prefill 步、`Tq=S、Tp=0`，消费 prefill_b1 同值产物）——两态是可切换的执行策略（2026-11-25 用户裁决，round18 问题 1，见 4.8）。具体改动四处：其一，`src/config_loader.py:1391-1395` 的禁令改为组大小校验（正整数；B>1 要求 trace 等长、等输出长度，违反直接抛；到达不设限，凑满 B 个等待请求成组、对齐最晚到达）；其二，准入层 `admit_waiting_requests`（`:10374-10397`）B>1 时一次取 B 个等待请求成组，组占用一个 request_index、携带 `slots=B`；其三，容量：`_validate_pim_capacity`（`:8595-8599`）按请求峰值 top-N 求和，closed batch 下有效并发 = 组数×B（等价于组请求峰值乘 slots，`src/pim/capacity.py:296-303`），否则容量低估 B 倍；其四，指标：`prefill_finish_times` 在 `:10013` 按最后一个 prefill 步完成记录——首 token 时点 = prefill 完成（真实编排器 prefill 返回即逐槽采样首 token，不等第一次 decode），所有 B 同一定义：单请求即第 0 步、合批组一步、逐槽组最后一个逐槽步；TPOT 以该时点起算（round25 R25-02 修正：早前「按首个合并 decode 步完成时刻计 TTFT」会把 output=2 的组算成 TPOT=0）。派发层、DAG 实例化、资源互斥排队全部复用——组就是一个请求，天然继承现有争用与流水重叠机制。该设计对应需求 R6。

```python
# 原有（gene_sim_scheduler.py:8656，decode 绑定）
tq = 1

# 目标（合并 decode 步的行数 = 组槽数；合批 prefill 步 tq=B×S、绑定 prefill_b{B}）
tq = 1 if slots == 1 else slots
```

### 4.7 batch 时间构成归因与抢占解析建模

　　评估模型要把「收益与代价各花在哪」从总时间里拆出来。收益侧不需要新机制——权重复用经 4.5/4.6 自动进数字，并行与流水经多组并发的事件仿真天然发生（不同组的算子在不同 PU 上并行、同 PU 上排队，`dispatch_ops` 的互斥队列就是争用模型）。需要新增的是归因输出：`src/main.py` 增加 `results/batch_metrics.json`，逐 (batch 档, 组) 记录：各 PU busy 区间并集、算子与传输就绪到派发的排队等待（算子经 `dispatch_ops`（`:10200-10227`）的堆键、传输经 `dispatch_transfers` 按组记 (ready,start,link) 等待区间，与 PU 同口径进区间台账——round18 问题 2 定案）、组间流水重叠时长、按 zone 的搬运字节（权重/激活/KV，`src/pim/pimir_trace.py:198` 的 zone 归属现成）与 `weight_bytes_per_token`。抢占是纯解析项（round18 问题 5 定案）：`estimate_s = swap_out_s + swap_in_s + switch_cost_s`——换出按整组 KV 迁移字节除以 PIM→HOST 方向链路带宽、换入同字节除以 HOST→PIM 方向带宽（两方向独立取该方向链路最小带宽，显式配置仍两方向共用）、切换常数缺省 0，分项随 `batch_metrics` 落盘、报告侧单列呈现并标注 `mode=analytical`、`calibrated=false`（系数未经实测校准），不进事件引擎。守恒判据（round05 四项 + round18 问题 2 链路份额 + round22 目标资源口径）：每组的 `wall = busy_pure（未被争用类别认领的忙时并集） + 组内争用（等待区间 ∩ 同组在同一目标资源上的忙时——PU 与链路同口径、按目标资源维度累计） + 组间争用（等待区间 ∩ 其它组在同一目标资源上的忙时，算子与链路等待都在内） + sync 余项（真实空档与未被任何争用类别认领的等待段）`，四项由事件流直接累计、在时间轴上按组内→组间→busy→sync 的优先级互斥分账（组内与组间交段重叠时组内优先、busy 让位），守恒负值直接抛；链路份额单列 `link_wait_s`（就绪→派发原始等待合计，与 `performance_metrics.total_transfer_queue_wait_time_s` 按组求和对账）、`intra_link_contention_s`（链路等待∩同组同链路忙时）、`inter_link_contention_s`（链路等待∩其它组同链路忙时——同样组内优先），sync 余项不再吞链路排队；`batch_metrics` 落 `result_schema=2` 口径锚点，报告侧按口径门禁（result_schema、forward_steps 合批语义、链路归因字段）识别旧口径产物并降级标注。该设计对应需求 R7，代码位于 genesim `src/main.py`、`src/scheduler/gene_sim_scheduler.py`（累加器）、pim-compiler 报告侧。

### 4.8 收益机制验证与三项优化的独立开关

　　三项优化各自真实实现并独立可开关（用户裁决 2026-11-23：以需求 R8 为准，本节早前「三个档位都是扫参维度、不引入新的执行机制」「DP 本期只做解析估算」的降级表述作废）。权重复用经 4.5/4.6 的 M=B 产物贯穿计费与执行：B 档曲线（扫 B 档画出 `weight_bytes_per_token` 并对照 `1/min(B, tile_m)` 界）之外，另有固定 B 的独立对照开关 `--weight-reuse`（genesim `scheduler.weight_reuse`，缺省开=合批现状逐位不变；关闭时同一 B/trace/资源下仅切换 M 合并与产物选择——prefill 拆 B 个 M=S 逐槽步、decode 合并步固定选 decode_b1 产物，与流水逐槽机制共用 `prefill_step_count`，避免两套路径）。流水优化是 genesim 事件循环内的 stage 间 micro-batch 重叠调度：配置项 `scheduler.pipeline_overlap`（缺省关闭）开启后，组内 prefill 拆回逐槽 micro-batch 步、下一个 prefill 步在当前步只剩最后一个 stage 时提前进管（后一槽的浅层 stage 与前一槽的深层 stage 并行），decode 步仍按整步栅栏（依赖全部 prefill 的 KV 与自回归输出）；关闭时 prefill 为合批一步（与真实执行一致、拿权重装载摊销）。两态是可切换的执行策略（round18 问题 1 定案）：开＝逐槽 micro-batch 拿流水重叠、关＝合批拿摊销，收益只表述为吞吐改善/步间重叠生效；B=1 无第二个 prefill 步可放行，两态行为一致。多 batch 空间并行（DP 副本）已真实落地（round06 切片）：`ShardStrategy` 增加副本维度 G（`replicas`，缺省 1 等价现状；权重复制 G 份、KV 与激活按副本划分），`export_placement_for_strategy` 按「副本 × 现有 TP/PP」导出副本放置进 sidecar（顶层 `dp_replicas` 载荷声明逐副本物理 DPU 块与每副本权重字节；`kv_layout` 键按物理 DPU 落，副本 g 偏移 g×N），genesim 配置 `scheduler.dp_replicas`（缺省 1 关闭）开启后钉扎放置建立逐算子副本路由表，执行引擎把组 i 的算子与依赖边路由到副本 (i mod G) 的物理 DPU 组，容量按逐副本视图校验（每副本权重全量、KV/激活只计本副本组）；sidecar 硬件契约的规模口径（round09 问题 1 定案）：`compiler.hardware.num_dpus`、`strategy.num_dpus` 与 `dpu_ids` 保持**每副本编译基数**，物理 DPU 总数 = 每副本基数 × `strategy.replicas`（与 `strategy.total_dpus` 互检），`dpu_to_cluster` 与 `kv_layout` 键按物理总规模落，genesim 严格身份核验按该口径放大期望映射数，扫参入口的场景选择（`sim_config_for_cores`）同用「num_dpus × replicas」选物理核数——物理规模单一真源，三处同步；导出侧另有执行侧 KV head 分片开关 `--kv-head-pin`（KV 读取算子按 head 归属钉到持有 DPU）。报告 DP 段按产物两态呈现（round08）：存在副本产物（`--replicas` 开启、逐档带 `dp_replicas`）时输出逐副本实测归因（逐副本吞吐、权重复制 ×G、KV 按副本划分），仅解析产物时维持估算口径并显式标注「--replicas 未开启」（见 §九.2）。主机编排器侧的对应能力是最小可执行切片（round22 用户裁决 3）：`compile_llama2(per_slot_prefill=True)` 额外导出 [1,S] 逐槽图并构建 B 份槽位平移 prefill 计划（每步 M=S、KV 落本槽段），`run_decode_loop` 按计划以 B 个 M=S 步顺序执行——数值与合批一致，时间维重叠仍在 genesim 事件侧（切片边界）；DP 副本由 `CompiledModel.mem_plans_of_replica(g)`、`replica_of_group`/`replica_dpu_window` 提供每副本 mem_plans、按副本物理块各装一份权重与按组路由（`execute_plan(dpu_window=)`）；逐槽机制同时服务 pipeline_overlap 与 weight_reuse 关档对照，开关独立、缺省关闭（per_slot_prefill=False/replicas=1）时与既有路径逐位一致。执行策略是统一编译输入（round25 R25-03）：`run_full_pipeline --pipeline-overlap/--weight-reuse` 的导出带 `--per-slot-prefill`，贯穿 `export_pp_placement → export_placement_for_strategy → compile_llama2`，导出结果交付 B 份逐槽主机计划（`PlacementExportResult.compiled`）、sidecar 记录 `per_slot_prefill` 策略声明，genesim 侧据此与自身执行策略互检。该设计对应需求 R8。

### 4.9 batch 扫参入口与汇总报告

　　最终交付是一条命令出全档对比。`scripts/run_full_pipeline.py` 的 `--batch` 接受逗号列表后进入扫参模式：export 一次拿到全部档位的逐相位产物，随后逐档执行「仿真（组大小=B、`num_requests` 取 B 的整倍数）→ 收集 `summary.json` 与 `batch_metrics.json`」；另有 `--groups`（默认 1）扫并发组数 G 档位（映射 `max_inflight_requests` 的组数口径，服务 R8 流水验证），B×G 两维笛卡尔积可裁剪。汇总报告的落盘约定（round09 问题 2/4 定案）：每轮扫参的聚合写专项文件 `report-<name>.md`（`--report-name` 显式命名，缺省按档位自动命名如 `b1-2-4-8g1`），报告头嵌机器可读的「扫参范围」与「产物根」；canonical 的 `report.md` 固定为全量扫参索引——有覆盖 `--sweep` 默认档的报告就指向它，没有就列各专项报告指针与覆盖范围，专项结果只写专项文件名，不再覆盖 canonical 名。逐档表带「产物指纹」列（该档 `batch_metrics.json`+`summary.json` 字节的 sha256 前 8 位）：报告行自证消费哪一代盘上产物；每轮扫参落盘后对盘上全部报告做一次过期自检，产物目录已被覆盖的行在行内显式标注「产物已被覆盖，本行过期」——从机制上消灭报告引用已消亡产物世代。报告内容：吞吐/TTFT/TPOT/容量占用/`weight_bytes_per_token` 逐档表、时间构成归因表、瓶颈标注（首个饱和的资源）、最优配置推荐（吞吐最优档、时延约束下最优档、容量墙位置）；DP 副本档的产物目录带 `dp` 后缀（如 `b4g2dp2`），与非副本同名档并存不互相覆盖。genesim 侧 `run_all.sh` 不改（单场景冒烟职责不变），扫参编排全在 pim-compiler 入口；每档复用现有步骤函数与指纹口径（同 trace 种子、同产物路径约定），保证档间可比。该设计对应需求 R9，代码位于 `scripts/run_full_pipeline.py`。

## 五、改动汇总

　　涉及两个代码仓库：pim-compiler 改 `runtime/`、`memory/`、`backend/`、`genesim_bridge/`、`scripts/` 五处与一个空文件清理，genesim 改 `src/config_loader.py`、`src/scheduler/gene_sim_scheduler.py`、`src/pim/capacity.py`、`src/main.py` 四处；FlagTree 零改动。

### 5.1 文件级改动清单

| 文件路径 | 改动类型 | 改动内容简述 | 影响范围 | 优先级 |
| --- | --- | --- | --- | --- |
| `runtime/compile.py` | 修改 | `export_annotated_graph`/`compile_llama2` 加 `batch_size`，`[B,S]`/`[B,1]` 导出 | 全部编译调用方（经缺省 1 兼容） | P0 |
| `memory/kv_layout.py` | 修改 | `KVRegionSpec.num_slots`、槽位主序布局、按槽寻址、逐槽 mask | `runtime/`、`memory/`、7B KV 测试断言 | P0 |
| `memory/mem_planner.py` | 修改 | `format_capacity_table` 输出；激活/KV 经形状与 `num_slots` 自动缩放 | 容量判据调用方（`placement_export.py:975`） | P0 |
| `runtime/executor.py` | 修改 | `DecodeState.valid_lens`、批量 decode 循环、逐槽采样 | `run_decode_loop` 调用方、`compile.py:297` | P0 |
| `backend/hal_numpy.py` | 修改 | `bind_inputs`/`bound_pos` 的 pos 类型扩展为 `int \| tuple` | `runtime/kernels.py` 的 pos 消费方 | P0 |
| `runtime/kernels.py` | 修改 | `sdpa_kv_info` 烘逐槽偏移、`sdpa_kernel` 按槽写/读/掩码（`:1238-1396`）；顺带清理 executor 死 import | 仅 batch 路径，B=1 行为不变 | P0 |
| `genesim_bridge/placement_export.py` | 修改 | `_compile_tile` 按相位出产物（decode 档 M=B 逐档、prefill 档 M=B×S 合批——`prefill_b{B}` 产物，口径同 §4.5）、缓存键加 M、sidecar `pimir_by_phase` 结构 | genesim `_compiler_pimir_paths` 消费、`tune/steps.py:1055`（缺省兼容） | P0 |
| `scripts/export_pp_placement.py` | 修改 | CLI 加 batch 档列表参数，透传 export_placement_for_strategy | run_full_pipeline (B/C/D) 步 | P0 |
| `scripts/run_full_pipeline.py` | 修改 | `--batch`/`--groups` 扫参循环与汇总报告 | 无破坏（缺省单档=现行为） | P0/P1 |
| `tests/assert_node_matches_ref.py` | 删除 | init 起即为空文件（`git log` 证实），按「删优于加」清理 | 无（无任何调用方） | P0 |
| tests/（新增用例） | 新增 | 见 §7.1 清单 | 仅测试 | P0/P1 |
| genesim `src/config_loader.py` | 修改 | `execution_microbatch_size` 改为组大小校验（等长/等输出） | 仿真配置校验 | P0 |
| genesim `src/scheduler/gene_sim_scheduler.py` | 修改 | 组准入（`:10374`）、`step_context` 合并行数（`:8641`）、组容量倍数、排队等待累加器、组指标展开 | 仿真主循环 | P0 |
| genesim `src/pim/capacity.py` | 修改 | 请求峰值按组 slots 倍计（`:296-303`） | 容量审计 | P0 |
| genesim `src/main.py` | 修改 | 输出 `batch_metrics.json` | 结果消费（报告侧） | P0 |

### 5.2 新增/修改的数据结构

| 结构 | 字段/默认值 | 语义 |
| --- | --- | --- |
| `KVRegionSpec`（修改） | `num_slots: int = 1` | KV 槽数；1 等价改动前行为；`kv_off` 键扩为 `(slot, layer, head, which)` |
| `DecodeState`（修改） | `valid_lens: tuple = (0,)`；保留只读 `valid_len` | 逐槽有效长度；单元素时与旧标量语义一致 |
| placement sidecar 算子条目（修改） | `pimir_by_phase: {"prefill_b{B}": {...}, "decode_b{B}": {...}}`（`prefill` 键为 prefill_b1 同值副本，服务逐槽 micro-batch 策略与旧产物），每项含 `path/sha256/kernel_tile_n/tile_m` | 逐相位逐档产物索引；顶层旧字段 `pimir_path/kernel_tile_n` 保留为 decode_b1 同值副本，旧消费方零改动 |
| genesim 组请求（运行时） | `slots: int`（请求对象新增可选字段，缺省 1） | 组内槽数；容量与指标按它倍计 |
| `batch_metrics.json`（新增） | 见 4.7 | 逐组时间构成与字节归因；抢占项带 `"mode": "analytical"` 标注 |

### 5.3 接口变更

　　`compile_llama2`、`export_annotated_graph`、`export_placement_for_strategy`、`_compile_tile`、`kv_access`、`PIMStaticKVCache.update/read_tile`、`decode_mask`、`run_decode_loop`、`hal.bind_inputs` 均为**加参数且缺省值等价旧行为**的向后兼容变更，既有调用方（含 `tune/steps.py`、CEVA 导出路径、全部现有测试）不改一行即可通过；`export_costs_to_genesim` 与 extensions sidecar 结构不变。破坏性变更仅两处：空文件 `tests/assert_node_matches_ref.py` 删除（无调用方）；genesim `execution_microbatch_size>1` 从 ValueError 变为合法值——配置校验从「拒绝」变「接受并校验前提」，属需求直接要求的行为变更。

## 六、实施计划

### 6.1 阶段划分与依赖

　　分四个里程碑：M1（pim-compiler 形状链，R1–R4）先行，是所有下游的形状真源；M2（成本链，R5）与 M1 后半并行（只动 `genesim_bridge/`，不碰 M1 文件）；M3（genesim，R6–R7）依赖 M2 的逐档 sidecar；M4（交付，R8–R9）依赖 M3。每阶段收口跑 L0，M2 收口加跑前置实验，M4 收口跑 L2 全量。

### 6.2 任务拆分与并行化分析

　　任务单元按文件/函数粒度拆分，与 tdd-implement 的任务单元定义一致；两个单元写同一文件必须串行。依赖 DAG 如下表（「依赖」列为空表示可立即开工）。

| 单元 | 内容（文件:函数） | 依赖 | 并行性 |
| --- | --- | --- | --- |
| T1 | `runtime/compile.py` batch 导出 + `KVRegionSpec.num_slots`（`memory/kv_layout.py` 布局/寻址/mask） | — | 与 T4/T5 并行 |
| T2 | `memory/mem_planner.py` 容量表 + 台账对账 | T1 | 串行（接 T1 的 `num_slots`） |
| T3 | `runtime/executor.py` 批量循环 + `backend/hal_numpy.py` pos + `runtime/kernels.py` 按槽 | T1 | 串行（接 T1） |
| T4 | `genesim_bridge/placement_export.py` 逐相位产物（decode 档 M=B、prefill 档 M=B×S 合批）+ 前置实验（M=1 vs M=8） | — | 与 T1-T3 并行（不同文件） |
| T5 | `scripts/export_pp_placement.py` batch 档列表 CLI + sidecar `pimir_by_phase` 结构 | T4 | 串行（接 T4 的产物结构） |
| T6 | genesim `config_loader.py` 组校验 + `gene_sim_scheduler.py` 组准入/合并行数 + `capacity.py` slots 倍计 | T5（逐档 sidecar 才能仿真） | genesim 仓内串行 |
| T7 | genesim `main.py` batch_metrics + 排队等待累加器 | T6 | 串行（同仓接 T6） |
| T8 | `scripts/run_full_pipeline.py --batch` 扫参 + 报告 | T3+T5+T7 | 串行（汇合点） |
| T9 | 优化验证（B/G 档位曲线 + DP 解析项） | T8 | 串行 |
| T10 | 删除空对拍文件 + 文档 | — | 任意时刻 |

　　共享可变状态检测：T1–T3 都触碰 `runtime/compile.py` 与 `contracts/` 消费链，标为串行链；T4/T5 只写 `genesim_bridge/` 与 `scripts/export_pp_placement.py`，T6/T7 只写 genesim 仓，三组之间无共享文件可并行；T8 消费三组产物，是串行汇合点。合并顺序按 DAG 拓扑序 T1→T2→T3、T4→T5、T6→T7、汇合 T8→T9；每个合并点的回归级别：T3 并入跑 L0，T5 并入跑 L0+前置实验判读，T7 并入跑 L1（`run_full_pipeline --num-stages 4` B=1 逐位对照），T8/T9 并入跑 L1（B=4 一轮）+ 最终 L2。

### 6.3 每阶段的回归门槛

　　L0 = `python -m pytest tests/ -q -k "not llama2_7b"`（232 用例全绿，B=1 产物不变）——每个单元合并点必跑。L1 = `python scripts/run_full_pipeline.py --num-stages 4`（B=1 与 `--batch 4` 各一轮，仿真出数且容量门禁通过）——M3/M4 收口跑。L2 = `python -m pytest tests/ -q -k "llama2_7b"` 与 genesim `./run_all.sh`——最终验收跑。任一层失败禁止合入下一单元。

## 七、验证与测试方案

　　验证分四层：单元测试钉住功能点契约（含 B=1 等价性断言），前置实验钉住权重复用的物理事实，集成验证钉住全链路出数与 B=1 逐位回归，端到端扫参钉住交付形态。所有用例与代码同一次改动完成。

### 7.1 单元测试清单

| 测试文件:用例 | 操作步骤 | 预期结果 |
| --- | --- | --- |
| `tests/test_compile_batch.py::test_b1_export_identical` | 以 batch_size=1 编译小型 Llama 配置，比对节点名/形状序列与现有基线 | 与 B=1 现状逐项一致 |
| `tests/test_compile_batch.py::test_b4_export_shapes` | batch_size=4 导出 | placeholder 形状 `[4,S]`/`[4,1]`，`graph break` 为零 |
| `tests/test_kv_layout_batch.py::test_slots_bytes` | `num_slots=4` 建 `build_kv_layout` | `kv_bytes` = 4×单槽公式；offset 互不重叠；`slot=0` 寻址与旧布局一致 |
| `tests/test_mem_planner_batch.py::test_capacity_table` | B=2 规划后打印容量表 | 三区+内核 tile 合计 ≤ 预算；超预算档位抛 ValueError 并指明卡在 KV 区 |
| `tests/test_executor_batch.py::test_b2_b4_match_torch` | B=2/4 执行 prefill+decode，与单卡 PyTorch batched forward 对 logits | `np.abs(result-ref).max() < 0.2`、`mean() < 0.02`（沿用 `test_executor_llama2_7b.py:137-139` 容差） |
| `tests/test_executor_batch.py::test_unequal_prompts_raise` | 传不等长 prompt 列表 | ValueError，消息含「等长」 |
| `tests/test_cost_batch.py::test_m8_weight_amortized` | 同一 linear 编 M=1 与 M=8，读产物 `pim.tile-m` 与 `mram_traffic_bytes` | M=8 产物 `tile_m > 1` 且权重搬运份数 ≤ `ceil(8/tile_m)`（失败即前置实验证伪，按 §九.1 处置） |
| `tests/test_cost_batch.py::test_sidecar_batch_entries` | `--batch 1,4` 导出 | sidecar 算子条目含 `pimir_by_phase.prefill_b1/prefill_b4` 与 `decode_b1/decode_b4`（`prefill` 键为 prefill_b1 同值副本），顶层旧字段=decode_b1 同值 |
| genesim `tests/sim/test_closed_batch.py::test_merged_attention_linear` | 组大小 1 与 4 各跑一步 decode，比较注意力类算子搬运字节 | 比值 = 4（严格线性；若按 tp+tq 绑定会出现约 `4×(S+4)/(S+1)≈4.1` 的偏大） |
| genesim `tests/sim/test_closed_batch.py::test_b1_bit_identical` | 组大小 1 跑固定 trace，比对 `summary.json` 全字段 | 与改前逐位一致 |
| genesim `tests/sim/test_closed_batch.py::test_b4_weight_per_token` | 组大小 4 跑同 trace | `weight_bytes_per_token` < B=1 档；GEMM 执行次数 = `ceil(4/tile_m)` |
| genesim `tests/sim/test_closed_batch.py::test_mixed_lengths_raise` | 组大小 4 + 不等长 trace | ValueError（closed batch 前提校验） |
| genesim `tests/sim/test_closed_batch.py::test_capacity_slots` | 组大小 4 + 小容量画像 | 容量按 4×请求峰值核算，超限 ValueError |

### 7.2 集成验证

　　`python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out` 确认 CEVA 路径 B=1 不受影响；`python scripts/run_full_pipeline.py --num-stages 4` 跑 B=1 与 `--batch 4` 两轮，检查逐档产物（placement sidecar 的 `decode_b4` 条目、仿真 `batch_metrics.json` 的守恒四项与链路份额、汇总报告的归因表）；genesim 侧 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh && ./run_all.sh` 确认单场景冒烟不回归。

### 7.3 验收标准

　　对应需求第五节：回归判据——L0 全绿且 B=1 逐位不变；快速端到端——§7.2 两命令通过；逐项——7.1 表覆盖 R1–R9 各判据（R8 的曲线与归因出自 `--batch 1,2,4,8` + `--groups 1,2,4` 扫参报告）；性能判据——`weight_bytes_per_token` 随 B 单调下降并饱和于 `1/tile_m`（见 §八修正 1），KV 容量墙位置与 `2×layers×B×max_seq×kv_heads×head_dim×dtype_bytes` 手推一致，吞吐曲线符合小 B 近线性→中 B 放缓→大 B 触墙的分档形态；不回归判据——容量检查仍为硬失败、无 try/except 兜底、batch 结论均出自真实编译产物与合并执行。

## 八、对需求文档的修正记录

1. 需求原文：「权重字节/token 随 B 单调下降并趋于 1/B 下界」。实测修正：摊薄由一条轨迹覆盖 `tile_m` 行决定（`vpu.py:841-861`；当前被执行产物实测 `"pim.tile-m" = 1`，摊薄未触发，M=B 产物落地后 tile_m 由其 tile 搜索决定），B≤tile_m 时界为 1/B，B>tile_m 后每多一份 `ceil(B/tile_m)` 重放，界饱和于 `1/tile_m`。依据：`_effective_repetitions` 的 `ceil(M/tile_m)` 折算与 `.opcompiler_cache` 产物抽查。验收判据相应改为「趋于 `1/min(B, tile_m)`」。
2. 需求原文：「对拍器 `tests/assert_node_matches_ref.py` 扩展 batch 维」。实测修正：该文件自 init 提交（`f8ac5c6`）起即为空文件，无任何调用方；真实对拍机制是 pytest fixture 执行计划后与单卡 PyTorch logits 对比（`tests/test_executor_llama2_7b.py:126-141`，容差 max<0.2/mean<0.02）。B>1 对齐按该模式新建 `tests/test_executor_batch.py`，空文件删除。
3. 需求原文（R4）：「执行计划按一张图一次执行的静态计划，无 batch 维度」暗示需要改 `exec_plan_gen`。实测细化：`build_execution_plan`（`exec_plan_gen.py:374`）的 `arg_shapes` 取自节点 spec `local_shape`，batch 进图后计划自动缩放，本体无需改动；实际要改的只有 HAL 的 `pos` 绑定与 host handler 的按槽迭代。
4. 需求原文（R5/R6 初稿）：「prefill 的 M=B×S 合并代表点、prefill 的 tq 改为 B*query」。初版设计推演曾以 `ceil(B×S/tile_m)=B×ceil(S/tile_m)`（S≥tile_m）与注意力口径为由改为 prefill 按槽独立——round17 以真实算子编译证伪该等式（S 非 tile_m 整数时不成立，合批装载严格更少），且注意力口径的正确做法是块对角绑定而非放弃合并（用户 2026-11-25 裁决，round18 问题 1）。终态：prefill 与图编译/编排器同语义合批（M=B×S 逐 B 产物、genesim GEMM 按 M=B×S 计价、注意力按 B 个块对角序列分别计价），decode 合并（Tq=B）不变；「prefill 档编 M=S、按槽独立」的中间口径作废。
5. 代码注释与实际路径不符的两处（以调用链实测为准）：`sdpa_kernel` 自述「拆头后不是主路径」，实测 `graph/opt_passes.py` 无任何 SDPA 拆分，`compile_llama2` 主路径的注意力与 KV 读写就在 `sdpa_kernel`；`runtime/executor.py:11` 对 `PIMStaticKVCache/decode_mask/prefill_mask` 的 import 是死引用（测试只 import `DecodeState/execute_plan/run_decode_loop`），随 4.2/4.4 清理。
6. 4.7 原文「算子就绪到派发的排队等待」按逐算子求和累计。实施推演证伪（实施记录 I9）：同一请求的算子并行等待时区间互相重叠，求和会重复计重叠部分，守恒判据 `wall = busy + 等待 + sync` 不成立（B=1 实测 Σwait≈17.7×总时长）。修正为区间台账 + 并集口径：忙时与等待逐算子记区间，聚合求并集；暴露等待 = |wait \ busy|，sync 为 wall 减 |busy ∪ wait| 的余项，构造上非负、负数直接抛。同时补充组间流水重叠测度（busy 区间两两交集之和，`inter_group_overlap_s`），作为 4.7 归因输出的正式口径而非后置项。round05 修正（2026-11-23）：单组（G=1）在多核上忙时并集恒等于 wall，暴露等待与 sync 构造性为零、守恒式同义反复——组内 PU 争用不可见。归因口径改为四项：`wall = busy_pure（无本组等待重叠的忙时并集） + 组内争用（等待期间目标 PU 被同组其他算子占用，按目标 PU 维度累计） + 组间争用（|wait \ busy| 暴露等待） + sync 余项`，四项由事件流直接累计且在时间轴上互斥；`queue_wait_total_s`（等待并集）随之删除，其可见部分由组内争用承接。round18 修正（2026-11-25，R17-02/05）：传输派发（`dispatch_transfers`）按组补记 (ready,start,link) 等待区间，与算子等待同入区间台账——组内/组间争用的累计维度从「目标 PU」推广为「目标资源（PU 与链路同口径）」；链路份额单列 `link_wait_s`（原始等待合计，与 `performance_metrics.total_transfer_queue_wait_time_s` 按组求和对账）、`intra_link_contention_s`（链路等待∩同组同链路忙时）、`inter_link_contention_s`（链路等待的暴露部分——链路被其它组占用），sync 余项不再吞链路排队。抢占解析项拆 `swap_out_s/swap_in_s/switch_cost_s` 三分项：换出按 PIM→HOST、换入按 HOST→PIM 方向链路最小带宽独立计价，维持 mode=analytical、calibrated=false。round22 修正（2026-11-25，R21-01/R21-04）：组间争用与 inter_link 的「暴露等待/暴露部分」判据被反例证伪（等待被本组无关资源忙时覆盖的段不构成组间争用，原口径却把真实组间链路等待挤进 sync 余项）——归因改为目标资源交段：组间争用=等待区间 ∩ 其它组同一目标资源忙时、组内争用=等待 ∩ 同组同资源忙时（原有）、busy_pure=忙时并集减被争用类别认领的段、sync=余项，时间轴按组内→组间→busy→sync 优先级互斥分账维持零误差守恒（守恒负值直接抛），`batch_metrics` 落 result_schema=2 口径锚点、报告侧按口径门禁识别旧产物；权重复用开关从「--batch 档位本身」证伪并改为独立 `--weight-reuse` 固定 B 对照开关（`scheduler.weight_reuse` 缺省开=合批现状逐位不变，关闭时 prefill 拆 B 个 M=S 步、decode 合并步固定选 decode_b1 产物），4.8 同步补主机侧逐槽 prefill 计划与每副本 mem_plans/按组路由的最小可执行能力边界。round25 修正（2026-11-25，R25-01/02/03）：其一，步级形状的 prefill 分支此前返回单槽 S，依赖边计费（普通依赖、集合通信、首次使用）与算子绑定使用两套行数——合批 prefill 通信量少算 B 倍；修正为步级绑定单一真源（组行数、每槽 KV 长度，见 3.1），不带 Tq 轴的 KV 项按槽数放大，result_schema 升 3。其二，多槽 TTFT/TPOT 早前按首个合并 decode 步完成时刻记录（output=2 时 TPOT=0），修正为所有 B 统一在 prefill 完成（最后一个 prefill 步）记首 token 时点、TPOT 以该时点起算（见 4.6 其四）。其三，执行策略（pipeline_overlap/weight_reuse 折算的 per_slot_prefill）作为统一编译输入贯穿run_full_pipeline → export_pp_placement → export_placement_for_strategy → compile_llama2，导出结果交付 B 份逐槽主机计划、sidecar 记录策略声明，genesim 侧与自身执行策略互检（合批消费逐槽产物为保守方向放行）。

## 九、待确认事项

1. 前置实验（`test_m8_weight_amortized`）若证伪——M=8 产物仍选 tile_m=1 或权重搬运未摊薄——说明 tile 搜索在真实形状下未利用 M 维，FlagTree kernel/搜索侧调整需另立项，R5/R6 的权重复用叙事随之降级为「如实计费」。  
2. DP 空间并行（真实副本放置导出）已于 round06 落地：`ShardStrategy.replicas`、sidecar `dp_replicas`/物理 `kv_layout` 载荷、genesim `scheduler.dp_replicas` 开关与逐副本路由（计算、依赖边与容量三处同口径）均已实现并有两侧用例（副本=1 逐位不变、G=2 路由块不重叠/权重全量/KV 按 1/G 划分）。round08 补齐 DP 开启态资源指标全覆盖（资源枚举以派发台账为真源）、扫参接线测试与报告 DP 段两态呈现；新口径副本形态的 7B 端到端实跑产物随 round08 受控重跑出数（`--replicas 2` 档），报告据此输出逐副本实测归因。  
3. 抢占解析项的系数取值无实测来源：换出/换入分方向取 PIM→HOST / HOST→PIM 各自方向链路的最小带宽（round18 问题 5 定案，分项 swap_out_s/swap_in_s/switch_cost_s、切换常数缺省 0），报告维持 mode=analytical、calibrated=false 标注，实测校准仍留待外部锚点（Q3）。  
4. 扫参默认档位：`--sweep` 入口给 `--batch 1,2,4,8,16,32` 与 `--groups 1,2,4`（用户裁决 2026-11-23），覆盖内核 tile_m 上限（BLOCK_M=16）与外推容量墙；显式 `--batch/--groups` 仍可覆盖。经典单档路径（`--batch` 缺省 1）不变。  
5. genesim 仓库存在并行开发会话，T6/T7 动工前核对 `git status` 与 mtime，提交时甄别归属（沿用既有约定）。  
6. 图按最大 B 档编译一次（round05 问题 11 的取舍，2026-11-23 定案）：sidecar `capacity.slots` 显式记编译档，非编译档的「容量占用」列标「编译期折算」，R3 的「B 档容量数值与手推一致」判据只对编译档逐位成立；逐档真算（图导出缓存 + 逐档 KV/内存规划）列为后续可选优化。  
7. closed batch 维持等长硬前提（round05 问题 10 的取舍，2026-11-23 定案）：变长 padding 属连续 batching 范围（需求 §2.3 边界），`batch_metrics` 不再有 `padding_ratio` 字段，R6 验收判据按「closed batch 不含 padding」理解；合成 trace 仍为等长复制。  
8. DP 副本 sidecar 硬件契约的规模口径（round09 问题 1 定案，2026-11-23）：`compiler.hardware.num_dpus`、`strategy.num_dpus` 与 `dpu_ids` 保持每副本编译基数，物理 DPU 总数 = 每副本基数 × `strategy.replicas`（与 `strategy.total_dpus` 互检），`dpu_to_cluster` 与 `kv_layout` 键按物理总规模落；genesim 严格身份核验按该口径放大期望映射数（在评审给出的两个方向中取「校验侧以 hardware.num_dpus × replicas 为期望」，sidecar 载荷格式零改动——dp_replicas 载荷本就以 dpus_per_replica 声明每副本基数，与该定义自洽），扫参场景选择（`sim_config_for_cores`）同用「num_dpus × replicas」。  
9. 扫参报告的落盘与世代一致性（round09 问题 2/4 定案，2026-11-23）：专项聚合写 `report-<name>.md`（`--report-name` 或按档位自动命名），`report.md` 固定为全量扫参索引；逐档表带产物指纹列（`batch_metrics+summary` 内容摘要），每轮扫参后对盘上报告做过期自检并标注被覆盖行（详见 §4.9）；DP 档产物目录带 `dp` 后缀。  
