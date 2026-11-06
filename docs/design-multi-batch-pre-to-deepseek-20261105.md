# 多 batch 并发评估与 CEVA 对齐：完整讨论记录与实施方案

> 文档编号：design-multi-batch-pre-to-deepseek-20261105
> 创建日期：2026-11-05
> 文档性质：完整讨论记录 + 评估模型设计 + 实施路线
> 涉及仓库：flagos-pim-compiler、genesim、FlagTree
> 目标阶段：流片（TO）之前
> 说明：本文是完整版；`docs/design-multi-batch-pre-to-20261105.md` 是同一主题的精简版方案。本文不受字数与格式限制，目的是把讨论中的事实、判断、证据、方案和待确认项全部保留下来。

---

## 目录

- [一、文档说明与阅读指南](#一文档说明与阅读指南)
- [二、一页结论](#二一页结论)
- [三、需求与目标](#三需求与目标)
- [四、术语与并发模型](#四术语与并发模型)
- [五、三仓职责与现有链路](#五三仓职责与现有链路)
- [六、现有命令、产物与指标](#六现有命令产物与指标)
- [七、代码级现状调研](#七代码级现状调研)
- [八、关键配置与容易误读的行为](#八关键配置与容易误读的行为)
- [九、关键发现：当前 batch 成本模型为何不成立](#九关键发现当前-batch-成本模型为何不成立)
- [十、评估模型的三个层级](#十评估模型的三个层级)
- [十一、评估契约设计](#十一评估契约设计)
- [十二、batch 语义与成本口径](#十二batch-语义与成本口径)
- [十三、资源、容量与共享带宽模型](#十三资源容量与共享带宽模型)
- [十四、CEVA 硬件画像与映射](#十四ceva-硬件画像与映射)
- [十五、指标与 PPA 代理](#十五指标与-ppa-代理)
- [十六、校准与客观性](#十六校准与客观性)
- [十七、评估模型调整清单](#十七评估模型调整清单)
- [十八、实施路线总览](#十八实施路线总览)
- [十九、P0：评估契约与基线](#十九p0评估契约与基线)
- [二十、P1：batch 成本口径](#二十p1batch-成本口径)
- [二十一、P2：Llama2-7B 端到端 batch](#二十一p2llama2-7b-端到端-batch)
- [二十二、P3：连续 batching 与 CEVA 对齐](#二十二p3连续-batching-与-ceva-对齐)
- [二十三、P4：PPA 与设计空间扫描](#二十三p4ppa-与设计空间扫描)
- [二十四、文件改动清单](#二十四文件改动清单)
- [二十五、实验矩阵](#二十五实验矩阵)
- [二十六、验收标准](#二十六验收标准)
- [二十七、风险与边界](#二十七风险与边界)
- [二十八、里程碑](#二十八里程碑)
- [二十九、附录 A：关键公式](#二十九附录-a关键公式)
- [三十、附录 B：数据结构](#三十附录-b数据结构)
- [三十一、附录 C：建议命令](#三十一附录-c建议命令)
- [三十二、附录 D：源码阅读索引](#三十二附录-d源码阅读索引)
- [三十三、附录 E：讨论问答记录](#三十三附录-e讨论问答记录)
- [三十四、附录 F：待确认事项](#三十四附录-f待确认事项)
- [三十五、附录 G：决策记录](#三十五附录-g决策记录)

---

## 一、文档说明与阅读指南

### 1.1 本文来源

本文整理自 2026-11-05 的完整讨论。讨论内容包括：

1. 对三个仓库（图编译器、模拟器、算子编译器）的源码与文档调研；
2. 对“多 batch 并发”概念的澄清；
3. 对当前 GeneSim 评估模型的代码级核查；
4. 对 PIM 硬件约束、CEVA 目标架构、评估参数与客观性的分析；
5. 面向流片前的分阶段实施方案；
6. 用户对 TO、目标架构和关注指标的确认。

### 1.2 阅读指南

| 读者关心的问题 | 建议阅读 |
| --- | --- |
| 结论是什么、下一步做什么 | §二、§十八 |
| 为什么不建议直接调 batch | §九 |
| 当前三仓到底支持到什么程度 | §五、§六、§七 |
| 评估模型要怎么改 | §十至 §十七 |
| 具体怎么落地、改哪些文件 | §十八至 §二十八 |
| 原始讨论中每个问题怎么回答的 | §三十三 |
| 还有哪些事实需要确认 | §三十四 |

### 1.3 证据约定

文中的代码引用格式为 `文件:行号`，均来自本次调研时的工作区。硬件参数若来自历史记录，会标注“待复核”。

---

## 二、一页结论

### 2.1 五条核心结论

1. **“多 batch”要拆成四层**：请求级并发、张量 batch、空间并行（多组计算单元各服务一个 batch）、流水 micro-batch。当前三仓只实现了第一层，第二层完全没有。
2. **现在直接调 `max_inflight_requests` 得不出 batch 结论**。GeneSim 的 GEMM 成本是“按 token 重放一次 `tile_m=1` 的 pimir”；B 个请求就是把同一份权重搬运重复 B 次。在这个模型里，batch 只会增加争用，不会出现权重复用收益。
3. **正确顺序是：先钉住成本口径，再扫参；执行器 B>1 的数值闭环可以后置**，但必须保留 B=1 回归锚点和至少一个 B>1 的算子级实测锚点。
4. **TO 前要交付的不是一个 tokens/s，而是曲线和敏感度**：吞吐-时延随 batch 的变化、容量墙位置、哪个架构参数最敏感、Pareto 前沿。
5. **GeneSim 必须与 CEVA 对齐**：核/engine、DDR/L2/L1/DTCM、带宽、量化格式都要映射到目标架构；默认的 8 设备 × 8 簇 × 16 TensorPU 不是 CEVA。

### 2.2 第一批实验只做一件事

用 FlagTree 对同一个 linear 分别按 M=1、M=8（N/K 相同）编译 pimir，比较：

- `tile_m` 是否随 M 变化；
- `mram_traffic_bytes` 是否接近“权重只读一次”；
- 权重 DMA 次数是否与 M 解耦。

如果 M=8 的权重搬运仍然接近 8×，说明算子编译器还没有生成“batch 复用权重”的 kernel，此时任何批量调度都是空谈。

### 2.3 明确不做的事

- 不先做连续 batching 的完整实现；
- 不先改执行器跑通 B>1 数值对齐；
- 不把 `max_inflight_requests` 调大当作“batch 优化”；
- 不用包含 GPU 的默认配置代表 PIM 芯片；
- 不承诺未校准的绝对性能数字。

---

## 三、需求与目标

### 3.1 甲方需求

甲方需要看清并调优多 batch 并发对 Llama2-7B 推理性能的整体影响，重点是流片之前。

已确认：

| 项 | 结论 |
| --- | --- |
| TO 含义 | 流片（tape-out） |
| 目标架构 | CEVA 单核 + 4 engine；后续可能连接多个核 |
| 验证路线 | 短期在 GeneSim 验证；最终回到 CEVA 验证 |
| GeneSim 要求 | 必须符合 CEVA 特征，不能是另一台机器的模型 |
| 关注指标 | 处理/生成的 token 数、吞吐率、TTFT、TPOT |

### 3.2 需求拆解

| 需求 | 含义 | 交付物 |
| --- | --- | --- |
| 端到端支持 | Llama2-7B 从图编译到算子编译到 GeneSim | 全链路 batch 可运行 |
| 多 batch 并发 | 能表达并评估 batch 对性能的影响 | batch 曲线、容量墙 |
| 符合实际 | 硬件、负载、成本来源都要对齐真实目标 | 硬件画像、负载画像、校准报告 |
| 客观 | 可复现、可校准、有误差条 | 指纹、校准阶梯、敏感度 |
| 指导芯片开发 | 能回答“改哪个架构参数、改多少” | Pareto、敏感度排序、推荐区间 |

### 3.3 用户最初的三个理解与回应

| 用户理解 | 回应 |
| --- | --- |
| 1. 首先要支持多 batch，从大模型推理到 AI 编译器到 GeneSim | 方向对，但不是“先把执行器跑通 B>1 再研究”。当前缺的是成本口径里的 batch 和权重复用；请求级并发已经存在，不要把它当成 batch |
| 2. 要考虑 PIM 架构本身的限制：尺寸、寄存器、带宽 | 对，但要分档：小 batch 看权重流量和利用率，中 batch 看 WRAM/tile，大 batch 看 KV 容量和带宽。不能把这些混成一个“有效算力” |
| 3. 要考虑多 batch 并行的优化方法 | 对，但优化对象是“权重复用 + KV 容量 + 并行映射 + 调度”，不是单纯调大 `max_inflight_requests` |

---

## 四、术语与并发模型

### 4.1 四种并发方式

| 并发方式 | 含义 | 当前支持 | 主要影响 |
| --- | --- | --- | --- |
| A. 请求级并发 | 多个请求同时在飞，按到达时间连续准入；每个 forward 的算子仍是独立任务，共享计算与链路 | GeneSim 支持（`max_inflight_requests`） | 排队、争用、尾延迟；不体现 batch 计算效率 |
| B. 张量 batch | 多个请求的 token 拼成一个 M 维，一次 kernel 算完，权重只读一次 | 不支持（`execution_microbatch_size` 强制 1） | PIM 上最大的收益来源，尤其是 decode |
| C. 空间并行 | 多组 DPU/engine 各服务一个 batch 或请求 | 只有全局放置，没有按 batch 分组 | 用容量换并行度；权重复制与 KV 分布权衡 |
| D. 流水 micro-batch | PP 的不同 stage 同时处理不同 micro-batch | 有 Stage 视图，无 micro-batch 重叠调度 | 减少流水气泡，与 B 正交 |

### 4.2 静态 batch、连续 batching 与张量 batch

- **静态/closed batch**：一批请求同时到达、形状相同、一起执行，整批结束后再换下一批。最容易建模，是 batch 收益的上界。
- **连续 batching**：请求按到达时间动态进出，空槽立即补新请求。更接近真实服务，但需要处理不同 KV 长度、padding、抢占。
- **张量 batch**：把多个请求的 token 拼成一个 M 维，一次 kernel 算完。它可以和静态或连续 batching 组合。

本方案的短期主线是 **A + B**：先用 A 建立基线，再把 B 的成本口径做对。C 和 D 放到后续阶段。

### 4.3 为什么“多 batch 并发”容易混淆

- 在 GeneSim 里，`max_inflight_requests` 表达的是“同时在飞的请求数”，不是“一次 kernel 里有多少 token”；
- 在图编译器里，batch 维目前是 1，导出图、KV 布局、执行器都是单序列；
- 在 FlagTree 里，kernel 形状可以带前导维，但成本链按 `Tq` 逐 token 重放，拿不到权重复用。

三个“batch”含义不同，必须分开说。

---

## 五、三仓职责与现有链路

### 5.1 仓库职责

| 仓库 | 路径 | 职责 |
| --- | --- | --- |
| 图编译器 | `/data/fjg/flagOS/flagos-pim-compiler` | 图拆分、切分传播、通信计划、内存规划、执行计划生成、运行时编排；同时维护 GeneSim 成本桥接 |
| 模拟器 | `/data/fjg/flagOS/genesim` | 请求 trace、算子成本、调度与离散事件仿真、吞吐/时延/利用率报告 |
| 算子编译器 | `/data/fjg/flagOS/flagOS-installers/FlagTree` | TTIR → pim mlir、tile 预算、显式 DMA、激活融合、相位展开、EmitC；提供算子成本元数据 |

### 5.2 现有端到端链路

`flagos-pim-compiler/scripts/run_full_pipeline.py` 覆盖 Llama2-7B 的完整链路：

```text
HF Llama-2-7b 权重 + config
  │
  ├─(A)─> GeneSim model_parser        生成图骨架 IR（semantic_role 标投影身份）
  │
  ├─(0)─> export_fixed_pu_mapping     GeneSim 给出固定 PU 映射 → PartitionPlan
  │
  ├─(B)─> 图编译器 compile_llama2     TP/PP 切分、放置、本地分片
  │
  ├─(C)─> FlagTree 算子编译           真实分块 + pim mlir
  │
  ├─(D)─> placement sidecar           dpu_id / local shapes / pimir_path / kernel_tile_n
  │
  ├─(A2)> refine_ir_with_flagtree     成本精化 IR（仿真加载）
  │
  └─(E)─> GeneSim 仿真                 照 pimir 生成 trace，出代价
```

### 5.3 现有链路的关键检查

`run_full_pipeline.py` 已经做了不少防止“静默退化”的检查：

- 图骨架里七种投影必须有 `semantic_role`；
- sidecar 必须有 `pimir_path`，并核对 `pimir_sha256`；
- 仿真 IR 与 sidecar 必须是同一套 `op_id`；
- trace 来源不能是 `template`；
- 融合清单非空时，折激活的 kernel 必须带 `act_spec`；
- 走 GeneSim 方案时，`dpu_to_cluster` 必须回传且一致。

这些检查是评估客观性的基础，后续应继续保留和扩展。

---

## 六、现有命令、产物与指标

### 6.1 常用命令

```bash
# 环境
source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh
export PATH="$HOME/.local/bin:$PATH"

# 生成模型 IR
python scripts/model_parser.py \
  --model_name /data/fjg/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --output models/llama2_7b.ir

# 生成 trace
./run.sh --trace --synthetic --seed 0 --num_requests 10 --output traces/llama2_7b.trace

# TTIR 对照成本
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_flagtree.ir \
  --sidecar models/llama2_7b_flagtree_extensions.json \
  --seq-len 128 --ir-level ttir

# PIM MLIR 最终成本
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_pimir.ir \
  --sidecar models/llama2_7b_pimir_extensions.json \
  --seq-len 128 --ir-level pimir

# 默认仿真
./run.sh

# 完整一轮：图编译 → 算子编译 → GeneSim
python scripts/run_full_pipeline.py --num-stages 4
```

### 6.2 现有结果文件

GeneSim 的 `results/` 目录包含：

- `summary.json`：吞吐、TTFT、TPOT、利用率等；
- `device_utilization.csv`：每设备利用率；
- `stage_metrics.json`：每 stage 的 busy/bubble/underfill；
- `pu_metrics.json`：每 TensorPU 利用率直方图与不均衡度；
- `communication_metrics.json`：总字节、传输时间、链路利用率；
- `memory_metrics.json`：本地内存服务时间；
- `pim_capacity_metrics.json`：每 TensorPU 常驻容量检查；
- `request_metrics.json`：逐请求时延与 stage 不均衡；
- `timeline.json`（可选）：事件时间线。

### 6.3 现有指标（实测样例）

一次默认运行的 `summary.json` 示例：

```json
{
  "result_schema_version": "pipeline_cost_v12",
  "request_execution_mode": "full_prefill_decode",
  "request_aggregation_mode": "continuous_admission",
  "completed_requests": 10,
  "processed_tokens": 8084,
  "total_time_s": 27.05,
  "average_request_time_s": 19.44,
  "average_ttft_s": 4.82,
  "p50_ttft_s": 3.07,
  "p95_ttft_s": 12.63,
  "average_mean_tpot_s": 0.1598,
  "p50_mean_tpot_s": 0.1418,
  "p95_mean_tpot_s": 0.3188,
  "throughput_requests_per_s": 0.3697,
  "throughput_tokens_per_s": 298.85,
  "throughput_prompt_tokens_per_s": 242.51,
  "throughput_output_tokens_per_s": 56.34
}
```

注意：这个结果里 GPU 0–3 都有 useful flops，说明默认配置是异构系统，不是纯 PIM。

### 6.4 现有指标缺少什么

- 没有能耗、面积、tokens/s/W；
- 没有共享 DDR/L2 带宽饱和标志；
- 没有结果指纹（IR/sidecar/trace/config/code 版本）；
- 没有“成本来源”字段（哪个 pimir、哪个 phase、哪个 batch、哪个 tile）；
- 没有容量墙表（`(batch, seq)` 最大并发）；
- 没有敏感度与 Pareto。

---

## 七、代码级现状调研

### 7.1 GeneSim

#### 7.1.1 配置

`conf/sim.yaml` 的关键项：

| 配置 | 值 | 含义 |
| --- | --- | --- |
| `system.num_gpus` | 4 | 默认启用 4 个 GPU |
| `system.num_pim_devices` | 8 | 8 个 PIM 设备 |
| `compute_config.gpu_compute` | true | GPU 参与计算 |
| `compute_config.pim_compute` | true | PIM 参与计算 |
| `compute_config.partition_strategy` | heuristic | 按算术强度阈值分配设备 |
| `compute_config.pim_arith_intensity_threshold` | 1.0 | FLOP/byte 低于阈值倾向 PIM |
| `scheduler.max_inflight_requests` | 128 | 同时在飞请求数 |
| `scheduler.execution_microbatch_size` | 1 | 真张量 batch 未实现 |
| `scheduler.prefill_decode_overlap` | false | prefill/decode 不重叠 |
| `scheduler.adaptive_chunking` | true | 自适应 chunk |
| `pim.clusters_per_device` | 8 | 每设备 8 个 ClusterPU |
| `pim.tensors_per_cluster` | 16 | 每 ClusterPU 16 个 TensorPU |
| `pim.cluster_memory_capacity_gb` | 8.0 | 每 ClusterPU 8GB |
| 每 TensorPU 容量 | 512 MiB | 8GB / 16 |
| `pim.compile.tile_size` | 32 | 默认 tile |
| `pim.compile.mac_width` | 8 | MAC 宽度 |
| `pim.compile.max_tiles_per_pu` | 4096 | 12-bit 循环寄存器限制 |
| `pim.compile.default_past_length` | 2048 | 符号形状默认 KV 长度 |
| `pim.compile.default_query_length` | 1 | 符号形状默认查询长度 |
| `model.ir_path` | `models/llama2_7b_pimir.ir` | 默认仿真 IR |
| `trace.path` | `traces/llama2_7b.trace` | 默认 trace |
| `trace.num_requests` | 10 | 默认请求数 |

#### 7.1.2 调度与执行

`src/scheduler/gene_sim_scheduler.py`（约 8814 行）的核心行为：

- `_get_operator_with_runtime`（:247）：
  - 解析符号形状与成本系数；
  - 支持的符号只有 `Tq`、`Tp`、`Tp+Tq`、`Tq(Tp+Tq)`；
  - **没有 batch 维**。
- `_execute_pipeline`（:5039）：
  - 按 DAG 依赖、传输、计算/链路可用性调度；
  - Stage 视图只用于统计，不门控执行；
  - 每个 forward 的每个算子是一个独立任务；
  - 计算资源键是 PIM TensorPU（`compute_resource_key`，:5329）；
  - 链路资源有仲裁，但内存带宽没有跨 PU 仲裁。
- `max_inflight_requests` 与 `execution_microbatch_size`（:5231-5259）：
  - `execution_microbatch_size != 1` 直接 `NotImplementedError`；
  - 注释写明 cross-request tensor batching 未实现。
- `step_context`（:5296）：
  - prefill：`Tq=seq_len`、`Tp=0`；
  - decode：`Tq=1`、`Tp=seq_len+forward_step-1`（prefill 请求）或 `+forward_step`（decode-only）。
- `admit_waiting_requests`（:7031）：
  - 请求按到达时间入队；
  - `active_request_count < max_inflight_requests` 就准入；
  - 完成的请求立即释放槽位，不是闭批屏障。
- 事件循环（:7063）：
  - 处理 `request_arrival`、`op_done`、`transfer_done`；
  - 每个时间戳先推进所有事件，再补充准入、派发传输、派发算子。
- 容量检查（:4521）：
  - 调 `validate_tensor_pu_capacity`；
  - 装不下直接 `ValueError`，是硬失败。
- sidecar 容量交叉核对（:550、:618）：
  - 只告警，不抛错；
  - 检查单内核 `placed_mram_bytes` 是否超过每 TensorPU 容量；
  - 检查 `shard_dpus` 与 `placed_shards` 是否一致。
- 权重部署统计（:5775-5833）：
  - 从依赖构建 `weight_deployment` 结果；
  - **不读取 `conf/sim.yaml` 的 `weight_deployment.mappings` 来驱动放置**。

#### 7.1.3 PIM 成本与重复次数

`src/vpu/vpu.py`：

- `_execute_runtime`（:448）：
  - `query_count = _runtime_repetitions(op)`；
  - SOFTMAX：`loop_count = ceil(kv_length / chunk_size)`；
  - GEMV_SCORE/GEMV_CONTEXT：`loop_count = ceil(kv_length / tile_size)`；
  - GEMM：`loop_count = ceil(output_width / tile_size)`；
  - 向量算子：`loop_count = ceil(input_width / chunk_size)`；
  - 最终 `backend.execute_runtime(loop_count=..., repetitions=query_count)`。
- `_extract_query_count`（:534）：
  - 取输入形状去掉最后一维后的乘积；
  - 对二维 `[Tq, N]`，结果就是 `Tq`。
- `_runtime_repetitions`（:557）：
  - GEMV 注意力用 `_extract_attention_query_count`；
  - 其余用 `_extract_query_count`。
- `execute_profile`（:290）：
  - latency 来自 trace；
  - memory time 来自 trace 的 memory cycle 比例；
  - `memory_bytes` 来自 trace 统计；
  - `ideal_memory_time` 只做下界检查，不做带宽仲裁。

#### 7.1.4 pimir trace

`src/pim/pimir_trace.py`：

- `PimirProgram` 字段：`tile_m/n/k`、`n_extent`、`k_extent`、`element_bytes`、`wram_bytes_used`、`inner_loads`、`outer_loads`、`outer_stores`。
- `n_iterations = ceil(n_extent / tile_n)`；
- `k_iterations = ceil(k_extent / tile_k)`；
- `parse_pimir` 只支持 1 或 2 层 `scf.for`（外层 N、内层 K），**没有 M 层循环**；
- 权重/激活 DMA 按内层/外层归属，避免把外层预取按 K 迭代重复计费；
- `compile_from_pimir` 把 `k_iterations` 写成循环计数。

#### 7.1.5 容量模型

`src/pim/capacity.py`：

- `resolve_pim_capacity_limits`：
  - `pim.cluster_memory_capacity_gb` 是权威值；
  - 每 TensorPU 容量 = cluster 容量 / `tensors_per_cluster`；
  - 默认 8GB / 16 = 512MiB。
- `resident_tensor_entries_by_destination_op`：
  - 按目标算子统计常驻张量；
  - 区分 `model` once 与 `request` lifetime；
  - 并发请求按“最大生命周期峰值”求和。
- `validate_tensor_pu_capacity`：
  - 超容量直接 `ValueError`；
  - 返回每个 TensorPU/Cluster 的占用与溢出。

#### 7.1.6 IR 与 trace

- `src/ir/model_ir.py:1077` `build_from_hf_config`：
  - 符号形状只有 `("Tq", hidden)` 等；
  - 没有 batch 维。
- `models/llama2_7b.ir`：
  - 6852 个算子；
  - 前几个算子形状是 `[Tq, 4096]`；
  - 没有 batch。
- `src/ir/trace_ir.py`：
  - `Request` 字段：`request_id`、`input_tokens`、`output_length`、`arrival_time_ns`、`is_prefill`、`seq_len`、`prompt`；
  - Trace v2 保存 prompt，兼容 v1。
- `scripts/trace_generator.py`：
  - 支持 synthetic、fixed、dataset；
  - `--fixed` 生成同形、同到达时间的请求；
  - `--is_prefill false` 时，`input_len` 是初始 KV 长度；
  - 默认到达间隔 10ms。

#### 7.1.7 输出指标

`src/main.py`：

- `summary.json` 包含 TTFT/TPOT 的 count、average、P50、P95；
- 吞吐包含 req/s、tokens/s、prompt tokens/s、output tokens/s；
- 有 `efficiency_metrics`（MFU 等）；
- 有 `pim_capacity_metrics`；
- **没有 energy/power/area**。

### 7.2 图编译器

#### 7.2.1 编译入口

`runtime/compile.py`：

- `compile_llama2(model, strategy, prefill_seq_len, max_seq, hw, hardware, kv_dtype_bytes=2, dtype=fp16)`（:199）；
- 导出 prefill 图：
  - `input_ids = torch.arange(seq_len).unsqueeze(0)`（:78）→ `[1, seq]`；
  - `position_ids = arange(seq_len).unsqueeze(0)` → `[1, seq]`；
  - mask `[1,1,S,S]`。
- 导出 decode 图：
  - `input_ids = torch.tensor([[0]])`（:217）→ `[1,1]`；
  - `position_ids = [[0]]`；
  - mask `[1,1,1,1]`。
- **没有 `batch_size` 参数**。

#### 7.2.2 KV 布局

`memory/kv_layout.py`：

- `KVRegionSpec`（:20）：
  - `dpu_id`、`layers`、`kv_heads`、`q_heads_by_kv`、`max_seq`、`head_dim`、`dtype_bytes`、`kv_base`、`kv_off`、`kv_allocated_bytes`；
  - **没有 batch 槽**。
- `kv_bytes = 2 × layers × max_seq × kv_heads × head_dim × dtype_bytes`（:52-54）；
- `build_kv_layout` 按 layer/head/which 分配偏移；
- `kv_specs_from_placement` 要求 `k_proj` 为列切 `Shard(0)`，按 KV head 分给 DPU；
- `decode_mask(valid_len, max_seq)` 是单序列的 mask。

#### 7.2.3 执行器

`runtime/executor.py`：

- `DecodeState.valid_len`（:15-23）是单序列；
- `run_decode_loop`（:41-89）：
  - prefill 输入 `[1, prompt_len]`；
  - decode 输入 `[1,1]`；
  - `position_ids = [[valid_len]]`；
  - mask `decode_mask(1, max_seq)`；
- `_as_batch_tensor([ids])` 固定生成 `[1, len(ids)]`。

#### 7.2.4 硬件契约

`contracts/op_contract.py`：

```python
DEFAULT_HARDWARE_CONFIG = PIMHardwareConfig(
    num_dpus=8,
    num_tasklets=16,
    mram_bytes_per_dpu=8 * 2**30,
    wram_bytes_per_dpu=65536,
    dma_align=8,
)
```

- `flatten_leading_dims`：把 `[B, S, H]` 压成 `(B*S, H)`；
- `flatten_shard_dim`：把图张量切分维换算到二维 kernel 坐标；
- `DpuShard`：`shard/replicate/partial` 三档。

#### 7.2.5 内存规划

`memory/mem_planner.py`：

- `plan_dpu` 规划权重区、KV 区、激活区；
- `kernel_mram_bytes` 是算子编译器回传的单台内核 tile 占用；
- 容量判据：三区 + 内核 tile ≤ MRAM 预算；
- 形状来自图节点，batch=1。

#### 7.2.6 成本桥接

`genesim_bridge/`：

- `op_classify.py:100`：
  - `ShapePoint(tq, tp)`；
  - prefill 代表点 `(Tq=seq_len, Tp=0)`；
  - decode 代表点 `(Tq=1, Tp=seq_len)`；
  - 没有 batch。
- `cost_extractor.py:173`：
  - linear 的编译形状 `[(point.tq, in_features), (out_features, in_features)]`；
  - 没有 batch。
- `placement_export.py`：
  - sidecar 记录 `dpu_id`、`local_in_features`、`local_out_features`、`kernel_tile_n`、`pimir_path`、`pimir_sha256`；
  - 没有 `phase`、`batch`、`kernel_tile_m`。
- `run_full_pipeline.py`：
  - 检查 sidecar、pimir hash、算子编号、trace 来源、融合清单；
  - 目前没有 batch 参数。

#### 7.2.7 CEVA 路径

- `scripts/run_ceva_pipeline.sh` 和 `scripts/export_gml.py`：
  - 导出 65 张 GML 子图；
  - 走厂商 GC、DDR 布局、CEVA 仿真；
  - 用 token 对拍判定。
- CEVA 设计/实施文档中没有 batch 或并发相关内容。

### 7.3 FlagTree

#### 7.3.1 PIM 编译流水线

`claude.md` 和 `docs/triton-pim-support-20260818.md`：

- `convert-triton-to-pim`：TTIR → PIM 布局/类型；
- `pim-tile-to-budget`：按 WRAM 预算搜索 tile，写 `pim.tile-m/n/k`；
- `pim-explicit-dma`：`tt.load/store` → 显式 DMA、WRAM 操作；
- `pim-fuse-activation`：激活折进主算子；
- `pim-expand-phases`：量化/softmax/RoPE 相位链；
- `pim-verify-gml-contract`：跨属性不变量；
- `pim-lower-to-emitc`：降 EmitC/C。

Python pass 参数：

```python
add_convert_to_pim(pm, target, num_dpus=1, num_tasklets=16,
                   wram_bytes=65536, mram_bytes=..., dma_align=8)
add_tile_to_budget(pm, full_m=-1, full_n=-1, full_k=-1)
add_explicit_dma(pm)
```

#### 7.3.2 tile 预算

`docs/tile-to-budget-flaggems-linear-20261002.md`：

- WRAM 预算 65536 B；
- 128×128×32 f16 的分块占 81920 B，超预算；
- FlagGems linear 是 grid 切分内核，`pim-tile-to-budget` 只回写实测分块，不重建；
- `pim.tile-wram-bytes` 可能大于 `pim.wram-bytes`，这是“放不下”的事实，不是错误；
- A 路只跑到 `pim-explicit-dma`，不走 EmitC（标量参数限制）。

#### 7.3.3 已知限制

`docs/triton-pim-support-20260818.md`：

- 还不是完整 PIM 后端；
- WRAM 超预算只 warning；
- DMA 指针分析故意很窄；
- barrier 插入策略粗；
- WRAM buffer 复用不够；
- `num-dpus` 当前基本没有真正参与 kernel 划分。

### 7.4 CEVA

来自历史记录与设计文档（**待复核**）：

- 目标配置 `m32k_vpu500_dtcm64_8m_1024k_4e`：
  - 1 个 NPM 控制器核 + 4 个 engine；
  - 每个 engine 有自己的 SP100 VPU；
  - L2 8MB、L1 1MB、DDR 约 3983MB、DTCM 64KB；
  - tcs 中 Core0 = npm_8k，Core1 = sp100_vpu；
  - `run_tools -n 4` 每个 engine 开一个 VPU 会话。
- engine 级切分由厂商 GC/L2A 完成；GML 没有 engine 字段。
- 多设备：`vbu_npm_system` 最多 8 个 NPM core，每 core 4GB DDR 窗口，mailbox + AllGather；Llama runner 用单核；厂商说并行仿真不准确。
- 编译器的 DPU（UPMEM 式、8GB 私有内存、host relay）对应 1 个 NPM core，不是 1 个 engine。
- CEVA 路径上 `llama_runner` 替换 `runtime/executor.py`，`tp=1`。
- GC 成本：prompt 每图约 2341s / 18GB，decode 约 25s / 1GB，lm_head 约 6s；完整 65 图约需 576GB 磁盘（历史记录）。
- 全流程约 10 小时，工作区至少 130GB 可用磁盘。

---

## 八、关键配置与容易误读的行为

### 8.1 默认配置是异构系统，不是 PIM 芯片

- `conf/sim.yaml` 默认 `num_gpus: 4`、`gpu_compute: true`；
- 现有 `results/summary.json` 里 GPU 0–3 都有 useful flops；
- `partition_strategy: heuristic` 会按算术强度把一部分算子放到 GPU；
- 因此默认结果是 CPU + GPU + PIM 的异构系统。

**芯片评估要求**：显式 PIM-only（`gpu_compute: false`、`cpu_compute: false`），或显式建模主机侧算子（embedding、lm_head 等）。

### 8.2 `weight_deployment` 不驱动放置

`conf/sim.yaml:342-351` 写了：

```yaml
weight_deployment:
  strategy: "layer_type"
  mappings:
  - layer_type: "qkv_proj"
    device: "gpu"
  - layer_type: "attention"
    device: "pim"
  - layer_type: "ffn"
    device: "gpu"
  pim_weight_replica: true
```

但调度器只在结果里构建 `weight_deployment` 统计对象（`src/scheduler/gene_sim_scheduler.py:5775-5833`），**不读取这份 mappings 来驱动放置**。实际放置由 partition heuristic 或编译器 sidecar 决定。

**结论**：不要依赖这份配置做设备映射；要么删掉避免误导，要么真正接入放置逻辑。

### 8.3 容量检查：硬失败与告警要分清

- 常驻容量检查 `validate_tensor_pu_capacity` 是**硬失败**（`src/pim/capacity.py:583-616`），装不下直接 `ValueError`；
- sidecar 的单内核 MRAM 交叉核对 `_check_placed_mram_against_capacity`（:550）和 `_check_placed_footprint`（:618）只**告警**。

之前讨论中“容量检查是 warn”的说法不准确；做 batch 研究时，硬失败要保留，告警项要按需升级为失败。

### 8.4 放置桥接的已知限制

`genesim/docs/llama-2.md` 6.5：

- 混合策略每段只标注首台 DPU；
- 被钉住的算子挤在每个 vpu 的第一个 PU 上；
- 换策略要手工同步改配置；
- 流水线还没有重叠调度；
- 跨段通信走主机中转而非点对点；
- 搜索空间只枚举 2 的幂段数。

这些限制会让并行度看起来偏低，做 batch/扩展性结论前要先修或至少标注。

### 8.5 时间和成本口径容易混

- pimir 的 `mram_traffic_bytes` 是算子编译器实测的搬运量；
- GeneSim 的 `data_bytes` 来自 IR 系数或实测；
- latency 来自 pimir trace；
- 三者可能来自不同口径（全局形状 vs 本地分片形状、A 路 vs B 路、prefill vs decode）。

必须统一到一个“成本来源”字段，否则数字不可比。

---

## 九、关键发现：当前 batch 成本模型为何不成立

### 9.1 实际 pimir 的样子

抽查 `.opcompiler_cache/143324fcfb1ff6b8.pimir.mlir`：

```text
pim.tile-m = 1
pim.tile-n = 512
pim.tile-k = 32
pim.wram-bytes-used = 33856
```

函数体：

- A 输入 `1×32`；
- B 权重 `512×32`；
- 输出 `1×512`；
- 外层 N 循环：`0..2048 step 512`；
- 内层 K 循环：`0..4096 step 32`；
- 内层每次迭代加载 A tile 和 B tile，做 `tt.dot`。

这是一个 **M=1 的 decode 单 token kernel**。

### 9.2 GeneSim 怎么用它

`PIMVPU._execute_runtime`：

- `query_count = _runtime_repetitions(op) = Tq`；
- `loop_count = ceil(output_width / tile_size)`；
- `latency = trace_latency(loop_count) × Tq`。

于是：

- decode（Tq=1）：执行 1 次；
- prefill（Tq=128）：执行 **128 次**；
- 每次执行都重新加载 B 权重 tile。

### 9.3 真实的 batch GEMM 应该怎样

一个 batch-B GEMM 可以做到：

- 外层 N：遍历输出宽度；
- 中层 M：遍历 batch 行；
- 内层 K：归约；
- 权重 tile 在每个 N/K 块只加载一次，在 M 行间复用；
- 激活 tile 每个 M 行加载一次；
- 输出每个 M 行写一次。

batching 的收益来自“权重搬运被 M 摊薄”，不是来自 MAC 数减少。

### 9.4 当前模型的两个后果

1. **batch 收益恒为零**：B 个请求 = B 份 per-token trace，权重流量随 B 线性增长；
2. **prefill 成本被高估**：prefill 用 decode pimir 重放 Tq 次，权重搬运被放大 Tq 倍。

### 9.5 为什么这不是“改个参数”能解决的

- GeneSim 的 pimir 解析器只支持 N/K 两层循环，没有 M 循环；
- `repetitions` 直接取 Tq，不知道 `tile_m`；
- placement sidecar 里 GEMM 的 pimir 全是 `tile_m=1`；
- 成本提取的 `ShapePoint` 没有 batch 维；
- 图编译器的导出图、KV、执行器都是 batch=1。

要修的是 **成本语义**，不是单个参数。

### 9.6 第一批实验

**实验 1：M=1 vs M=8 的同一个 linear**

- 用 FlagTree 分别编译 M=1、M=8；
- 比较 `tile_m`、`mram_traffic_bytes`、权重 DMA 次数；
- 如果 M=8 的权重搬运不是接近 1×，说明 kernel 还没有 batch 复用；
- 如果接近 1×，说明 GeneSim 侧需要按 `ceil(M/tile_m)` 执行，而不是按 M 重放。

**实验 2：prefill vs decode pimir**

- 确认 placement sidecar 是否应该按 phase 记录不同的 pimir；
- 如果只有 decode pimir，prefill 成本一定是错的。

**实验 3：B=1 回归**

- 在改任何东西之前，记录 B=1 的基线结果、指纹和成本来源；
- 后续所有改动都要过这个锚点。

---

## 十、评估模型的三个层级

### 10.1 可信性（必须改）

目标：让 Llama2-7B 全链路的数字在语义上自洽。

必须修：

1. batch 维与权重复用；
2. phase 对应的 pimir；
3. 目标硬件画像与 PIM-only；
4. 共享带宽/容量模型；
5. 成本来源统一；
6. 放置桥接限制。

### 10.2 客观性（应该加）

目标：让数字可校准、可复现、有误差条。

应该加：

1. 校准阶梯；
2. 参数不确定度与 Monte Carlo；
3. 结果指纹；
4. held-out 验证；
5. 不变量检查（roofline、容量、WRAM、循环次数）。

### 10.3 芯片指导力（必须补）

目标：从“跑分”变成“架构取舍依据”。

必须补：

1. 一阶 PPA（能耗/面积/能效）；
2. 设计空间扫描与敏感度；
3. Pareto 前沿；
4. “最敏感的三个参数 + 推荐区间”。

---

## 十一、评估契约设计

### 11.1 四份画像 + 一个指纹

| 契约 | 内容 | 落地位置 |
| --- | --- | --- |
| 硬件画像 | 核/engine 数、时钟、MAC 宽度、DTCM/L1/L2/DDR 容量与带宽、互连、量化格式 | `conf/hw_<target>.yaml` |
| 负载画像 | Llama2-7B、prompt/output 长度分布、prefill/decode 比例、到达率、batch 策略 | `conf/workload_<name>.yaml` + trace |
| 成本来源 | 每个 phase 用哪份 pimir、tile M/N/K、dtype、本地分片形状、batch | placement sidecar 扩展字段 |
| 结果指纹 | IR、sidecar、trace、配置、代码版本的哈希 | `results/summary.json` 增加 `evaluation_fingerprint` |

### 11.2 统一入口语义

```bash
python scripts/run_full_pipeline.py \
  --hw-profile ceva_single \
  --workload llama2_7b_batch_sweep \
  --batch 1,2,4,8,16 \
  --pim-only \
  --strict
```

`--strict` 的语义：

- 必须读到每个算子的 `pimir_path`，不允许退回手写模板；
- 必须记录 phase 与 batch 的成本来源；
- 容量装不下直接失败；
- 结果必须带指纹，不同指纹的结果禁止直接对比。

### 11.3 结果指纹字段

```python
@dataclass(frozen=True)
class EvaluationFingerprint:
    ir_sha256: str
    sidecar_sha256: str
    trace_sha256: str
    config_sha256: str
    code_commit: str
    hw_profile: str
    workload_profile: str
```

---

## 十二、batch 语义与成本口径

### 12.1 M 的定义

| 场景 | M | KV 长度 | 说明 |
| --- | --- | --- | --- |
| Prefill | 组内 prompt token 总数（或 padding 到 B×S_max） | 每个请求自己的 prompt 长度 | 初期用等长 closed batch |
| Decode | B | 每个槽位的 `valid_len` | 初期所有槽位等长 |

### 12.2 BatchGroup

```python
@dataclass
class BatchGroup:
    batch_id: int
    request_ids: list[str]
    phase: str                 # "prefill" / "decode"
    m_tokens: int              # 本步 M
    kv_lengths: list[int]      # 每个槽位的 KV 长度
    engine_ids: list[int]      # 本组使用的 engine
```

### 12.3 计费口径

- 权重：每个 N/K tile 在每个 batch group 内只搬一次；
- 激活：随 M 线性增长；
- KV：随组内各请求的 KV 长度之和增长；
- 输出：随 M 线性增长；
- 不变量：`weight_bytes_per_token` 随 B 上升而下降；总计算量随 M 近似线性。

### 12.4 cost provenance

建议 sidecar 条目增加：

```json
{
  "op_id": 4,
  "phase": "decode",
  "batch": 8,
  "m_tokens": 8,
  "pimir_path": "...",
  "pimir_sha256": "...",
  "kernel_tile_m": 1,
  "kernel_tile_n": 512,
  "kernel_tile_k": 32,
  "local_in_features": 4096,
  "local_out_features": 2048
}
```

GeneSim 按 `(op_id, phase, batch)` 选择 pimir；找不到就失败。

---

## 十三、资源、容量与共享带宽模型

### 13.1 容量公式

```text
KV bytes = 2 × layers × kv_heads × head_dim × seq_len × dtype_bytes × B
权重 bytes = num_parameters × weight_dtype_bytes
单核容量约束 = 权重 + KV + 激活 + 内核 tile ≤ DDR/core
DTCM 约束 = tile_m × tile_n + tile_k × tile_n + tile_m × tile_k ≤ DTCM
```

### 13.2 典型数值（用于建立直觉）

Llama2-7B fp16：

- 权重约 13.5GB；
- KV 每请求每 2048 token 约 1GiB：
  `2 × 32 × 32 × 128 × 2048 × 2B = 1,073,741,824B`。

W4A8：

- 权重约 3.5GB；
- KV 仍是 fp16 约 1GiB/请求/2K。

CEVA 单核 DDR 约 4GB（待复核）：

- W4A8 权重 3.5GB 后，剩余空间非常有限；
- 多 batch 在单核上可能首先被 DDR 容量卡住；
- 这也是“多核/多 device”成为后续方向的原因之一。

### 13.3 共享带宽

当前 GeneSim：

- 每个算子按自己的 pimir 周期算延迟；
- `peak_memory_bandwidth` 只用于理想下界检查（`src/vpu/vpu.py:314-329`）；
- 没有跨 engine/PU 的 DDR/L2 带宽仲裁；
- 高并发时多个 PU 各自按满带宽计时，总带宽可能超过物理峰值。

两步补：

1. **后处理式检查**：按时间窗口汇总并发算子的 memory bytes，计算聚合带宽利用率，超过峰值标记 `bandwidth_saturated`；
2. **事件式共享资源**：把 DDR/L2 建模成共享资源键，像链路资源一样参与调度。

### 13.4 分档结论

| 区间 | 主要瓶颈 | batch 的影响 |
| --- | --- | --- |
| 小 batch | 权重流量、tasklet/PU 利用率 | 吞吐随 B 近似线性上升 |
| 中 batch | WRAM/DTCM tile、compute、MRAM 带宽 | 增速变缓，权重复用收益接近上限 |
| 大 batch | KV 容量、DDR 带宽 | 排队、尾延迟上升，吞吐可能不升反降 |

甜点区在中段与大段的交界，由 WRAM、MRAM/DDR、带宽、PU/engine 数共同决定。

---

## 十四、CEVA 硬件画像与映射

### 14.1 参数（待复核）

来自目标配置 `m32k_vpu500_dtcm64_8m_1024k_4e` 的历史记录：

| 项 | 值 | 建模位置 |
| --- | --- | --- |
| 计算核 | 1 NPM core + 4 engine | 1 个 PIM device / 4 个 TensorPU |
| L2 | 8 MB | 共享缓存容量与带宽 |
| L1 | 1 MB | 每 engine 缓存 |
| DTCM | 64 KB | 对应 WRAM/tile 预算 |
| DDR | 约 3983 MB / core | 对应 core 级 MRAM 容量 |
| 多核 | 最多 8 核，4 GB DDR 窗口/核，mailbox + AllGather | 多 device / 节点与链路 |

**落地前必须用当前 `NPMEngine.xml`、`cnnha_upd_b.cfg`、tcs 复核。**

### 14.2 映射约定

- 图编译器的 1 个 DPU = 1 个 NPM core，不是 1 个 engine；
- 4 个 engine = 4 个 TensorPU，共享 core 的 DDR/L2；
- 短期 GML 没有 engine 字段，engine 级放置由厂商 GC/L2A 决定；
- GeneSim 先用“4 engine 并行 + 校准后的单 engine 吞吐”建模；
- engine 级映射待厂商接口确认后再细化；
- 多核场景先由 GeneSim 建模，CEVA 只做单核数值锚点。

### 14.3 与 GeneSim 默认配置的差异

| 项 | GeneSim 默认 | CEVA 目标 | 影响 |
| --- | --- | --- | --- |
| 计算单元 | 8 设备 × 8 簇 × 16 TensorPU = 1024 | 1 核 + 4 engine | 并行度、放置、容量完全不同 |
| 每单元内存 | 512 MiB/TensorPU | 4 GB/core | batch 容量墙不同 |
| 缓存 | 未显式分层 | L2 8MB / L1 1MB / DTCM 64KB | tile 与带宽模型不同 |
| 互连 | 多设备/节点模型 | mailbox/AllGather | 多核扩展不同 |

**结论**：不换配置，batch 曲线是另一台机器的。

---

## 十五、指标与 PPA 代理

### 15.1 指标定义

| 类别 | 指标 | 说明 |
| --- | --- | --- |
| token | prompt tokens、output tokens、total tokens | 区分输入与生成 |
| 吞吐 | req/s、output tokens/s、prompt tokens/s | 分别报告 |
| 时延 | TTFT P50/P95、TPOT P50/P95、端到端 P50/P95 | 尾延迟必须报 |
| 利用率 | engine/PU 利用率、DDR 带宽、L2、DTCM | 定位瓶颈 |
| 容量 | 权重/KV/激活占用、`(batch, seq)` 最大并发表 | 找容量墙 |
| 能效 | tokens/s/W、TOPS/W、energy/token | 一阶代理 |
| 面积 | MAC 阵列、SRAM、PHY 面积 | 一阶代理 |

### 15.2 PPA 一阶代理

不改事件引擎，先用后处理方式计算：

```text
E_token ≈ Σ_ops (MACs × e_mac + DRAM_bytes × e_dram + SRAM_bytes × e_sram)
        + P_leak × T
Area    ≈ N_mac × a_mac + SRAM_bytes × a_sram + PHY面积
```

系数从厂商数据或规格书获取；先保证相对比较可用，再逐步校准绝对值。

### 15.3 当前 GeneSim 缺什么

- 没有系统级 energy/power/area；
- DRAMsim3 内部有功耗统计，但没有接进 `summary.json`；
- 没有 tokens/s/W、TOPS/W、$/token；
- 没有带宽饱和标志。

---

## 十六、校准与客观性

### 16.1 校准阶梯

| 层级 | 对照对象 | 验收误差（建议） |
| --- | --- | --- |
| 单算子 | CEVA 单算子/单层 | 绝对误差 ≤ 20% |
| 单层 | CEVA 单层 | 绝对误差 ≤ 25% |
| 全模型 | CEVA 单核 B=1 | 绝对误差 ≤ 30%，趋势误差 ≤ 10% |
| 多请求 | CEVA 单核多请求 | 仅相对趋势，标注未校准 |

### 16.2 客观性要求

- 固定基准配置：Llama2-7B、B=1、固定 trace、固定 pimir；
- 留出验证：在一组 `(seq, batch)` 上校准，在另一组上验证；
- 参数不确定度：硬件参数给区间，用 Monte Carlo 或 Latin hypercube 报 P50/P90；
- 结果指纹：不同指纹的结果禁止直接对比；
- 不变量检查：roofline、容量、WRAM/DTCM、循环次数、带宽上限。

### 16.3 对甲方的表述

建议明确：

> 用相对趋势和敏感度指导流片前的架构选择，用校准锚点约束模型误差；绝对数字暂不作为规格承诺。

---

## 十七、评估模型调整清单

| # | 调整项 | 优先级 | 现状 | 调整后 | 对芯片决策的作用 |
| --- | --- | --- | --- | --- | --- |
| 1 | batch 权重复用 | P0 | pimir `tile_m=1`，按 Tq 重放 | phase×batch pimir；权重只读一次 | batch 收益是否真实 |
| 2 | phase 对应 pimir | P0 | prefill 复用 decode pimir | prefill/decode 各用各的 | TTFT 与 prefill 成本 |
| 3 | 目标硬件画像 | P0 | 默认 1024 PU、512MiB/PU | CEVA 1 核 + 4 engine、4GB DDR、L2/L1/DTCM | 结论是否属于目标芯片 |
| 4 | PIM-only | P0 | 默认 4 GPU 参与 | 关闭 GPU 或显式建模主机侧 | 结果是系统还是芯片 |
| 5 | 共享带宽 | P1 | 无跨 PU 仲裁 | 后处理检查 + 事件式资源 | 高并发是否被带宽卡住 |
| 6 | 成本来源 | P0 | A/B 路、全局/本地混用 | sidecar 记录 phase/batch/tile/hash | 数字可比性 |
| 7 | 放置桥接 | P1 | 每段只标首台 DPU | 多 PU 表达或标注下界 | 并行度是否被低估 |
| 8 | PPA | P2 | 无 | 一阶能耗/面积代理 | tokens/s/W、面积取舍 |
| 9 | 设计空间扫描 | P2 | 单点运行 | DoE + 敏感度 + Pareto | 改哪个参数、改多少 |
| 10 | 校准与指纹 | P1 | 无误差条 | 校准阶梯 + 指纹 | 数字可信度 |

---

## 十八、实施路线总览

| 阶段 | 周期 | 目标 | 关键交付 |
| --- | --- | --- | --- |
| P0 | 1 周 | 评估契约 + Llama2-7B B=1 基线 | 硬件/负载画像、指纹、基线报告 |
| P1 | 2 周 | batch 成本口径 + 单算子锚点 | `Tb` 维、phase pimir、B=8 锚点 |
| P2 | 4 周 | Llama2-7B 端到端 batch 闭环 | 图编译/算子编译/GeneSim 的 B>1 链路 |
| P3 | 4 周 | 连续 batching + CEVA 特征对齐 | 合批调度、4 engine、共享带宽 |
| P4 | 并行 | PPA + 设计空间扫描 | 敏感度排序、Pareto、推荐配置 |

---

## 十九、P0：评估契约与基线

### 19.1 工作内容

1. 新增 `conf/hw_ceva_single.yaml`、`conf/workload_llama2_7b.yaml`；
2. 在 `run_full_pipeline.py` 增加 `--hw-profile`、`--workload`、`--pim-only`、`--strict`；
3. 跑通 Llama2-7B B=1 全链路，记录 `summary.json`；
4. 增加 `evaluation_fingerprint`；
5. 明确 PIM-only：`compute_config.gpu_compute=false`、`cpu_compute=false`，或显式建模主机侧算子；
6. 记录 B=1 基线作为后续所有改动的回归锚点。

### 19.2 验收

- B=1 结果可复现；
- 无模板回退（`verify_trace_provenance` 通过）；
- 无 GPU 参与（或明确记录主机侧算子）；
- 容量检查通过；
- 输出包含 token 数、吞吐、TTFT、TPOT、利用率、容量；
- 结果带指纹。

### 19.3 风险

- 默认配置的 GPU 参与会让结果偏乐观；
- `weight_deployment` 不驱动放置，容易误读；
- 硬件画像参数需要复核。

---

## 二十、P1：batch 成本口径

### 20.1 工作内容

1. GeneSim IR/成本增加 batch 维（`Tb`，或约定 batch 组把 `Tq` 解释为组内 token 数）；
2. sidecar 按 `(op_id, phase, batch)` 记录 pimir；
3. FlagTree 按 `(phase, M)` 编译/量测，prefill 不再复用 decode pimir；
4. 修正 `_runtime_repetitions` 与 `tile_m` 的关系：`ceil(M/tile_m)` 或 pimir 覆盖全 M 时为 1；
5. 权重 DMA 按 N/K tile 计一次，激活 DMA 随 M 增长；
6. 用 FlagTree 对同一 linear 编 M=1 与 M=8，比较 `mram_traffic_bytes` 与权重 DMA 次数。

### 20.2 验收

- B=1 与现有结果逐位一致；
- B=8 的权重字节/token 明显低于 B=1；
- 至少一份 B>1 的 pimir 实测锚点；
- 单算子误差 ≤ 20%。

### 20.3 关键判定

如果 M=8 的权重搬运仍然接近 8×，停止仿真侧工作，先改 kernel 的循环顺序/tile_m。

---

## 二十一、P2：Llama2-7B 端到端 batch

### 21.1 图编译器

- `compile_llama2(batch_size=...)`；
- `export_annotated_graph` 支持 `[B, S]` 与 `[B, 1]`；
- `KVRegionSpec` 增加 batch 槽，`kv_bytes` 乘 B；
- `DecodeState` 改为 per-slot `valid_len`；
- 内存规划、通信计划、ExecutionPlan 形状乘 B；
- batch 维默认 replicate，不切开；空间并行作为单独开关。

### 21.2 算子编译器

- 支持 M=B（decode）与 M=sum tokens（prefill）的 kernel 生成；
- tile 选择在 WRAM/DTCM 预算内考虑权重复用；
- 输出 `tile_m/n/k`、`mram_traffic_bytes`、`wram_bytes` 等元数据；
- 增加 batch>1 的 lit 测试，固定“权重只读一次”的行为。

### 21.3 GeneSim

- 支持 `BatchGroup`：closed batch 先做，所有请求等长、同时到达；
- 按 batch 组调度的批任务；
- 4 engine 映射为 4 个 TensorPU；
- 容量检查按槽位 KV 计算；
- 输出 batch 曲线。

### 21.4 验收

- B=1/2/4 在 NumpyBackend 上与 PyTorch 数值对齐（小模型测试）；
- Llama2-7B B=1/2/4/8 全链路跑通；
- 批量曲线、容量墙、TTFT/TPOT 曲线；
- 无模板回退、无 GPU 参与、指纹一致。

---

## 二十二、P3：连续 batching 与 CEVA 对齐

### 22.1 工作内容

1. GeneSim 调度器把同一 stage、同一 op、同一时刻 ready 的多个请求合并为批任务；
2. 处理不同 KV 长度的 padding/mask；
3. prefill chunking 与 prefill/decode 混排；
4. 4 engine 的共享 DDR/L2 带宽模型；
5. 多核画像与 mailbox/AllGather 链路模型；
6. CEVA 单核 B=1 数值锚点，多核只做相对趋势。

### 22.2 验收

- 连续 batching 与 closed batch 的对比曲线；
- engine 利用率、DDR 带宽饱和标志；
- 多核扩展曲线（标注未校准）；
- CEVA B=1 数值一致。

---

## 二十三、P4：PPA 与设计空间扫描

### 23.1 工作内容

1. 一阶能耗/面积代理，消费现有结果；
2. 敏感度分析：WRAM、DDR 带宽、PU/engine 数、tile 策略、量化格式；
3. Pareto 前沿：吞吐-时延-能耗/面积；
4. 推荐配置区间与“最敏感的三个参数”。

### 23.2 验收

- 能给芯片团队输出敏感度排序和推荐区间；
- 报告区分“已验证趋势”和“未校准外推”。

---

## 二十四、文件改动清单

### 24.1 flagos-pim-compiler

| 文件 | 改动 |
| --- | --- |
| `runtime/compile.py` | `compile_llama2` 增加 `batch_size`；导出 `[B,S]` / `[B,1]` |
| `memory/kv_layout.py` | `KVRegionSpec` 增加 batch 槽；`kv_bytes` 乘 B |
| `runtime/executor.py` | `DecodeState` 改为 per-slot；batched decode 循环 |
| `memory/mem_planner.py` | 激活/KV 按 B 计算 |
| `genesim_bridge/op_classify.py` | `ShapePoint` 增加 `Tb` |
| `genesim_bridge/cost_extractor.py` | 按 `(phase, B)` 编译与量测 |
| `genesim_bridge/placement_export.py` | sidecar 增加 `phase`、`batch`、`kernel_tile_m` |
| `scripts/run_full_pipeline.py` | `--hw-profile`、`--workload`、`--pim-only`、`--strict`、batch sweep |
| `contracts/op_contract.py` | CEVA 硬件画像字段 |
| `docs/` | 本方案与实施记录 |

### 24.2 genesim

| 文件 | 改动 |
| --- | --- |
| `conf/hw_ceva_single.yaml` | 新增：1 核 + 4 engine、DDR/L2/L1/DTCM、带宽 |
| `conf/hw_ceva_multi.yaml` | 新增：多核与互连 |
| `conf/workload_llama2_7b.yaml` | 新增：负载与 batch 策略 |
| `src/ir/model_ir.py` | 支持 batch 维或 batch 组 |
| `src/vpu/vpu.py` | `repetitions` 与 `tile_m` 对齐；phase 选择 |
| `src/scheduler/gene_sim_scheduler.py` | `BatchGroup` 调度；phase pimir；共享带宽检查 |
| `src/pim/capacity.py` | 按槽位 KV 计算容量 |
| `src/main.py` | token/吞吐/TTFT/TPOT、指纹、PPA 输出 |
| `scripts/trace_generator.py` | batch trace（已有 `--fixed`，补参数） |
| `tests/sim/` | B=1 回归、B>1 成本、容量、共享带宽 |

### 24.3 FlagTree

| 文件 | 改动 |
| --- | --- |
| `lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp` | batch M 下的 tile 搜索与权重复用 |
| `python/triton/backends/pim_sidecar.py` | 输出 phase/batch 元数据 |
| `test/Dialect/TritonPIM/` | batch>1 的 lit 测试 |
| `docs/` | batch tiling 与权重 DMA 口径 |

---

## 二十五、实验矩阵

### 25.1 自变量

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| batch | 1, 2, 4, 8, 16 | closed batch 先做 |
| 序列长度 | 128, 512, 2048 | prefill 与 KV 长度 |
| 输出长度 | 1, 64, 128 | decode 步数 |
| 阶段 | prefill-only、decode-only、混合 | 分开报告 |
| 量化 | fp16、W4A8 | 改变算术强度 |
| 硬件 | 1 核、2 核、4 核、8 核 | 多核扩展 |
| 编译 | TP1/2/4、tile M/N/K、融合开关 | 区分编译瓶颈 |

### 25.2 方法

- 先用 fractional factorial 设计，控制总运行次数；
- 每个维度至少一个 held-out 点；
- 报告主效应与一阶交互；
- 所有结果带指纹与误差条。

### 25.3 建议的基准负载

| 名称 | 配置 |
| --- | --- |
| llama2_7b_prefill_s128 | B=1, prompt=128, output=1, fp16 |
| llama2_7b_decode_b1 | B=1, KV=2048, output=128, fp16 |
| llama2_7b_decode_batch | B=1/2/4/8, KV=2048, output=128, fp16 |
| llama2_7b_batch_w4a8 | 同上，W4A8 |
| llama2_7b_mixed | prefill/decode 混合，synthetic 或数据集 trace |

### 25.4 建议的第一个对照

```text
M=1  pimir vs M=8 pimir
指标：tile_m、mram_traffic_bytes、权重 DMA 次数、latency
判据：M=8 的权重搬运是否接近 1×
```

---

## 二十六、验收标准

| 类别 | 标准 |
| --- | --- |
| 正确性 | B=1 与现有结果逐位一致；B>1 在 NumpyBackend 上与 PyTorch 对齐 |
| 成本 | 每个算子有 phase/batch pimir；权重 DMA 只读一次；无模板回退 |
| 评估 | 结果带指纹；容量装不下直接失败；共享带宽饱和可见 |
| 校准 | 单算子/单层/全模型误差在约定范围；多核标注未校准 |
| 芯片指导 | 输出敏感度排序、容量墙、Pareto 与推荐配置区间 |

---

## 二十七、风险与边界

### 27.1 风险

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| CEVA 厂商并行仿真不准确 | 多核性能无法实测 | GeneSim 建模，CEVA 只做单核数值锚点；报告标注不确定性 |
| GML 没有 engine 字段 | engine 级放置不可表达 | 短期用 4 engine 均匀并行 + 校准；长期推动厂商接口 |
| 厂商编译器不支持 batch | 无法得到真实 batch kernel | 先在 GeneSim 做成本级 batch，再评估是否需要自研 kernel |
| 硬件参数不完整 | 画像不可信 | 参数给区间；敏感度分析；用 CEVA 单算子校准 |
| 绝对数字误差 | 不能作为规格承诺 | 只承诺相对趋势与敏感度；绝对数字标注误差范围 |
| 全流程运行时间长 | 迭代慢 | 用 `--max-requests`、缩小模型/序列做 scout；正式数据再全量跑 |
| 容量检查与 sidecar 交叉核对口径不一致 | 数字放不下却跑完 | 统一为硬失败；禁止跳过 |
| 放置桥接低估并行度 | batch/扩展性结论偏悲观 | 先修多 PU 表达或标注下界 |
| 共享带宽未建模 | 高并发结果偏乐观 | 后处理检查 + 事件式资源 |

### 27.2 现在能指导什么

- 相对架构取舍：多 PU/engine vs 大 WRAM/DTCM vs 高带宽；
- batch 甜点区和容量墙的位置；
- 编译策略（TP/PP、tile、融合、量化）对性能的影响；
- 目标负载下的瓶颈排序。

### 27.3 现在不能指导什么

- 绝对硅后性能、功耗、面积、热；
- 物理时序、真实内存系统效应（除非补争用模型）；
- 厂商编译器的实际代码质量；
- 多核性能的 CEVA 实测。

---

## 二十八、里程碑

| 时间 | 里程碑 | 判定 |
| --- | --- | --- |
| 第 1 周末 | P0 完成 | Llama2-7B B=1 基线可复现，契约文件齐全 |
| 第 3 周末 | P1 完成 | B=1 不变，B=8 权重/token 下降，单算子锚点通过 |
| 第 7 周末 | P2 完成 | 7B B=1/2/4/8 全链路跑通，批量曲线产出 |
| 第 11 周末 | P3 完成 | 连续 batching 与 4 engine 模型完成，CEVA 单核锚点通过 |
| 第 12 周末 | P4 初版 | 敏感度与 Pareto 报告交付 |

---

## 二十九、附录 A：关键公式

```text
KV bytes = 2 × layers × kv_heads × head_dim × seq_len × dtype_bytes × B
权重 bytes = num_parameters × weight_dtype_bytes
单核容量约束 = 权重 + KV + 激活 + 内核 tile ≤ DDR/core
DTCM 约束 = tile_m × tile_n + tile_k × tile_n + tile_m × tile_k ≤ DTCM
能耗 = Σ(MACs × e_mac + DRAM_bytes × e_dram + SRAM_bytes × e_sram) + P_leak × T
面积 = N_mac × a_mac + SRAM_bytes × a_sram + PHY面积
```

Llama2-7B fp16 典型值：

```text
权重 ≈ 7e9 × 2B = 14GB（实际 13.5GB 量级）
KV/请求/2K ≈ 2 × 32 × 32 × 128 × 2048 × 2B ≈ 1GiB
```

W4A8：

```text
权重 ≈ 7e9 × 0.5B = 3.5GB
```

---

## 三十、附录 B：数据结构

```python
@dataclass(frozen=True)
class HardwareProfile:
    name: str
    cores: int
    engines_per_core: int
    ddr_bytes_per_core: int
    l2_bytes: int
    l1_bytes_per_engine: int
    dtcm_bytes_per_engine: int
    clock_ghz: float
    mac_width: int
    ddr_bandwidth_gbs: float
    interconnect_bandwidth_gbs: float
    weight_dtype: str
    activation_dtype: str


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    model: str
    batch_sizes: list[int]
    prompt_lengths: list[int]
    output_lengths: list[int]
    prefill_decode_ratio: float
    arrival_interval_ns: int


@dataclass(frozen=True)
class CostProvenance:
    op_id: int
    phase: str
    batch: int
    m_tokens: int
    pimir_path: str
    pimir_sha256: str
    tile_m: int
    tile_n: int
    tile_k: int
    dtype: str


@dataclass(frozen=True)
class EvaluationFingerprint:
    ir_sha256: str
    sidecar_sha256: str
    trace_sha256: str
    config_sha256: str
    code_commit: str
    hw_profile: str
    workload_profile: str


@dataclass
class BatchGroup:
    batch_id: int
    request_ids: list[str]
    phase: str
    m_tokens: int
    kv_lengths: list[int]
    engine_ids: list[int]
```

---

## 三十一、附录 C：建议命令

```bash
# 环境
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
export PATH="$HOME/.local/bin:$PATH"

# B=1 基线（PIM-only、严格模式）
python scripts/run_full_pipeline.py \
  --hw-profile ceva_single \
  --workload llama2_7b_decode_b1 \
  --batch 1 --pim-only --strict

# batch 扫描（closed batch）
python scripts/run_full_pipeline.py \
  --hw-profile ceva_single \
  --workload llama2_7b_batch_sweep \
  --batch 1,2,4,8,16 --pim-only --strict

# 单算子锚点：同一 linear 的 M=1 vs M=8
python scripts/export_pp_placement.py --num-stages 1 --num-dpus 1 --batch-sizes 1,8

# 固定 shape trace（现有能力）
./run.sh --trace --fixed --num_requests 8 \
  --input_len 2048 --output_len 128 --is_prefill false \
  --fixed_arrival_time_ns 0 --output traces/fixed_b8.trace
```

---

## 三十二、附录 D：源码阅读索引

### 32.1 GeneSim

| 主题 | 文件:行号 |
| --- | --- |
| 运行时形状解析 | `src/scheduler/gene_sim_scheduler.py:247-295` |
| 连续准入 | `src/scheduler/gene_sim_scheduler.py:7031-7054` |
| 事件循环 | `src/scheduler/gene_sim_scheduler.py:7063-7184` |
| micro-batch 限制 | `src/scheduler/gene_sim_scheduler.py:5243-5259` |
| 计算资源键 | `src/scheduler/gene_sim_scheduler.py:5329-5341` |
| 容量硬失败 | `src/scheduler/gene_sim_scheduler.py:4521-4539`、`src/pim/capacity.py:583-616` |
| sidecar 容量告警 | `src/scheduler/gene_sim_scheduler.py:550-660` |
| 权重部署统计 | `src/scheduler/gene_sim_scheduler.py:5775-5833` |
| PIM 执行与重复次数 | `src/vpu/vpu.py:448-562` |
| memory 下界检查 | `src/vpu/vpu.py:314-329` |
| pimir 解析 | `src/pim/pimir_trace.py:220-424` |
| 12-bit 循环 | `src/pim/pim_trace_loader.py:64-121`、`src/pim/pim_isa.py:163-204` |
| 容量模型 | `src/pim/capacity.py:21-230` |
| 指标输出 | `src/main.py:141-165, 335-343` |
| IR 构建 | `src/ir/model_ir.py:1077-1200` |
| trace 生成 | `scripts/trace_generator.py` |
| 固定 trace | `src/ir/trace_ir.py` |

### 32.2 图编译器

| 主题 | 文件:行号 |
| --- | --- |
| 编译入口 | `runtime/compile.py:199-315` |
| batch=1 导出 | `runtime/compile.py:78, 216-218` |
| KV 布局 | `memory/kv_layout.py:20-164` |
| 单序列执行 | `runtime/executor.py:15-94` |
| 硬件契约 | `contracts/op_contract.py:9-62` |
| 内存规划 | `memory/mem_planner.py:274-338` |
| 成本代表点 | `genesim_bridge/op_classify.py:100-170` |
| linear 成本 | `genesim_bridge/cost_extractor.py:160-180` |
| 放置导出 | `genesim_bridge/placement_export.py:150-260` |
| 全流程 | `scripts/run_full_pipeline.py` |

### 32.3 FlagTree

| 主题 | 文件/文档 |
| --- | --- |
| PIM 方言与 pass | `claude.md`、`docs/triton-pim-support-20260818.md` |
| tile 预算 | `lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp` |
| grid 内核 tiling | `docs/tile-to-budget-flaggems-linear-20261002.md` |
| Python pass 参数 | `python/src/passes.cc` |
| sidecar 生成 | `python/triton/backends/pim_sidecar.py` |
| lit 测试 | `test/Dialect/TritonPIM/` |

---

## 三十三、附录 E：讨论问答记录

### 33.1 问：你是哪个版本？

答：Claude Code CLI；会话环境标注的模型是 `deepseek-flash[1m]`。当前工作目录 `/data/fjg/flagOS/flagos-pim-compiler`，分支 `main`，最新提交 `1226e96 support ceva Simulator`。

### 33.2 问：多 batch 是什么？从推理到编译器到 GeneSim 是否都要支持？

答：要拆成四层：

1. 请求级并发：GeneSim 已支持；
2. 张量 batch：不支持；
3. 空间并行：只有全局放置；
4. 流水 micro-batch：只有 Stage 视图。

不需要“先全链路支持 B>1 数值”才能研究 batch；先补成本口径，再做执行闭环。

### 33.3 问：PIM 架构限制怎么考虑？

答：分档看：

- 小 batch：权重流量、tasklet/PU 利用率；
- 中 batch：WRAM/DTCM tile、compute；
- 大 batch：KV 容量、DDR 带宽。

硬限制包括：DTCM/WRAM 64KB、12-bit 循环 4095、`max_tiles_per_pu=4096`、每核 DDR 容量、互连带宽。

### 33.4 问：多 batch 并行优化方法有哪些？

答：按 PIM 收益排序：

1. 权重复用 / batched kernel；
2. tile 目标函数升级；
3. KV 容量管理；
4. 并行映射（TP vs DP vs 混合）；
5. 调度（continuous batching、chunked prefill、prefill/decode 混排）；
6. 量化（fp16 vs W4A8）；
7. 长序列循环限制处理。

### 33.5 问：评估模型要不要调整？

答：要，分三层：

1. 可信性（必须改）：batch/phase/硬件画像/PIM-only/成本来源/共享带宽/放置桥接；
2. 客观性（应该加）：校准、误差条、指纹、held-out、不变量；
3. 芯片指导力（必须补）：PPA、设计空间扫描、敏感度、Pareto。

### 33.6 问：是否要支持 Llama2-7B 端到端？

答：链路已经存在（`run_full_pipeline.py`），但它是 batch=1。要做的是把 batch 语义和成本来源贯穿三仓，而不是重新搭一条链路。

### 33.7 问：TO 是什么？目标架构是什么？关注什么指标？

答：TO = 流片；目标 = CEVA 单核 + 4 engine，后续可能多核；短期 GeneSim、最终 CEVA；关注 token 数、吞吐、TTFT/TPOT。

### 33.8 关键提醒

- 不要用 `max_inflight_requests` 代替 batch；
- 不要用含 GPU 的默认配置代表 PIM 芯片；
- 不要依赖 `weight_deployment` 做放置；
- 不要把 decode pimir 重放当作 prefill 成本；
- 不要承诺未校准的绝对数字。

---

## 三十四、附录 F：待确认事项

1. CEVA 目标配置的当前实际参数（DDR 约 3983MB、L2 8MB、L1 1MB、DTCM 64KB、4 engine）需用最新厂商文件复核；
2. GML/GC 是否支持 engine 级放置或 batch>1 的字段；若支持，补充到画像与 sidecar；
3. 甲方对 TTFT/TPOT 的具体 SLO 目标；
4. W4A8 是否为唯一交付量化格式，fp16 是否仍需评估；
5. 多核互连的拓扑与带宽（mailbox/AllGather）的实测或厂商数据；
6. CEVA 单核 B=1 的可重复基线是否已有稳定的 token/时延数据；
7. 厂商 GC 是否会对 batch>1 的图产生不同的 engine 切分。

---

## 三十五、附录 G：决策记录

| 编号 | 决策 | 理由 | 备选方案 | 状态 |
| --- | --- | --- | --- | --- |
| D1 | 先做成本口径，再做执行器 B>1 | 当前 batch 收益结构上为零，先跑执行器会放大错误 | 先改执行器 | 已定 |
| D2 | 短期只做 closed batch | 等长、同时到达，复杂度最低，是 batch 收益上界 | 直接做连续 batching | 已定 |
| D3 | batch 维默认 replicate，不切分 | 先保证语义正确；空间并行后续单独评估 | 一开始就做 DP 切分 | 已定 |
| D4 | CEVA 1 DPU = 1 NPM core，4 engine = 4 TensorPU | 与现有编译器/硬件层级一致 | 1 DPU = 1 engine | 已定 |
| D5 | 多核性能先由 GeneSim 建模 | 厂商并行仿真不准确 | 等 CEVA 多核实测 | 已定 |
| D6 | PPA 先做一阶代理 | 不需要改事件引擎，能支撑相对比较 | 物理级功耗模型 | 已定 |
| D7 | 结果必须带指纹 | 避免不同配置/产物混比 | 只靠人工记录 | 已定 |
| D8 | `weight_deployment` 不作为放置依据 | 代码中未驱动放置 | 接入放置逻辑 | 待确认 |
| D9 | 容量检查保留硬失败 | 避免“跑完但放不下” | 降为告警 | 已定 |
| D10 | 绝对数字不作为规格承诺 | 未完成全链路校准 | 直接承诺 | 已定 |

---

## 结语

这份文档的核心不是“怎么把 batch 调大”，而是“怎么让 batch 的收益在模型里真实存在，并且能被 CEVA 校准”。

一句话总结：

> **先把“权重只读一次”这件事在成本模型里证明，再谈 batch 调度；先把 GeneSim 变成 CEVA 的模型，再谈流片决策。**
