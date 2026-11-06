用户：

背景：目前初步实现了图编译到算子编译到genesim的仿真流程。

目标：甲方需求能够看清和调优多batch并发对推理性能的整体影响，尤其是TO之前。

相关仓库：
/data/fjg/flagOS/flagos-pim-compiler  图编译器
/data/fjg/flagOS/genesim  模拟器
/data/fjg/flagOS/flagOS-installers/FlagTree  算子编译器

我的理解，应该是在genesim模拟器上验证。大概可能的相关命令：
# 1. 激活环境
source  /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh
export PATH="$HOME/.local/bin:$PATH"

# 3. 生成模型 IR
python scripts/model_parser.py \
  --model_name /data/fjg/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --output models/llama2_7b.ir

# 4. 生成 Trace
./run.sh --trace --synthetic --seed 0 --num_requests 10 --output traces/llama2_7b.trace

# 5. FlagGems + FlagTree 成本提取（TTIR 对照，再 PIM MLIR 最终产物）
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_flagtree.ir \
  --sidecar models/llama2_7b_flagtree_extensions.json \
  --seq-len 128 --ir-level ttir

python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_pimir.ir \
  --sidecar models/llama2_7b_pimir_extensions.json \
  --seq-len 128 --ir-level pimir

# 6. 运行默认模拟
./run.sh。

完整一轮 genesim->图编译/算子编译->genesim
python scripts/run_full_pipeline.py --num-stages 4

我的理解：
1.从大模型推理到AI编译器到genesim，首先要支持多batch，啥是多batch? 这个 多batch并发基本功能得考虑
2.为了达到一个好的性能优化效果，还得考虑pim架构本身的限制，尺寸，寄存器，带宽等限制；
3.为了达到一个好的性能优化效果，是不是得考虑多batch并行的优化方法，只有这样才能有好的效果？
我现在不太理解，还请你给我一些建议。需要仔细阅读上面相关参考源码以及文档。

Kimi：

我先把三个仓库现有的执行链路、batch/request 抽象和 PIM 成本模型梳理清楚，再判断哪些应在图编译、算子编译和 genesim 中实现，避免只从概念上给建议。

Kimi：

初步发现一个关键点：genesim 已经有多请求、固定长度 workload、TTFT/TBT、设备利用率和 PIM 容量统计，不需要从零实现“多 batch”。下一步要确认它的 `batch_size` 到底是请求并发、真正融合的大 batch，还是流水调度分组——这三者的性能含义不同。

Kimi：

现在可以明确区分：现有 `max_inflight_requests` 是“多个独立请求共享设备”的连续准入，代码明确禁止 `execution_microbatch_size != 1`，因此还没有“多个请求合成一个张量 batch、共享权重读取”的能力。你的第 1 点应先做前者扫描，再把后者作为独立功能，不能把两者混为一个 batch 数字。

Kimi：

还有两个会直接影响结论可信度的细节：当前 PIM GEMM trace 会按 `Tq` 重放整段（包括权重搬运），所以不能简单把 `Tq` 改成 batch 数来冒充融合 batch；另外，放置导出固定用 `M=1` 编 kernel，真正 batch 后必须重新评估 `M=B` 对 WRAM 分块和权重复用的影响。

Kimi：

源码梳理已经闭环。建议的主线不是“先把三个仓都改成动态 batch”，而是先用现有多请求并发建立可信基线，再单独实现融合 batch 的算子成本与调度语义；否则很容易得到吞吐上涨、但实际上只是重复计算或重复搬权重被重复计费的假结果。

Kimi：

## 核心结论

你的总体方向是对的，但需要先区分两种“多 batch”：

1. **多请求并发**：多个请求各自执行计算图，共享 DPU、流水线和通信链路。  
   **genesim 已经支持，可以立即做实验。**
2. **跨请求张量 batch**：多个请求合成一个 `[B, ...]` 张量，一次 kernel 共同计算，从而复用权重、改变 tile 和内存流量。  
   **目前尚未实现，不能用现有 `num_requests` 或 `max_inflight_requests` 代替。**

因此，甲方近期要“看清和调优多 batch 对整体性能的影响”，**不必先等三个仓库全部支持动态 batch**。建议先用现有请求并发建立基线，再把真正的张量 batch 作为第二阶段能力实现。

---

## 1. 当前代码到底支持什么

### 已支持：多请求连续并发

`genesim/conf/sim.yaml:28-33` 中：

- `max_inflight_requests`：最多允许多少个请求同时处于生命周期内。
- `execution_microbatch_size: 1`：每个执行项仍然只有一个请求。

调度器会按到达时间持续准入请求，请求完成后立即补入下一个，而不是等整批结束，见：

- `genesim/src/scheduler/gene_sim_scheduler.py:7031`
- `genesim/src/scheduler/gene_sim_scheduler.py:8333`

这可以模拟：

- 多请求竞争同一计算资源；
- PP 流水段之间的请求重叠；
- prefill/decode 重叠；
- 通信链路争用；
- 并发请求 KV 占用导致的容量限制。

容量检查也会按最大并发请求数累计请求级常驻内存，见 `genesim/src/pim/capacity.py:543`。

### 未支持：真正的跨请求张量 batch

代码已经明确限制：

```text
Only scheduler.execution_microbatch_size=1 is supported;
cross-request tensor batching is not implemented.
```

位置：`genesim/src/scheduler/gene_sim_scheduler.py:5243-5254`。

另外，旧的 `scheduler.batch_size` 会被迁移为 `max_inflight_requests`，它**不是闭合 batch，更不是张量 batch**，见 `genesim/src/config_loader.py:396-411`。

图编译器的数值执行路径也仍是单请求：

- 图导出固定 `[1, seq_len]`：`runtime/compile.py:78`
- 运行时输入固定 `[1, len(ids)]`：`runtime/executor.py:56-93`
- 只有一个 `DecodeState.valid_len`：`runtime/executor.py:15`
- KV 容量公式没有 batch 维：`memory/kv_layout.py:59`

所以，“多 DPU 并发”和“多请求并发”已经存在，但“一个请求一个 batch=1 的图”目前仍是基本执行模型。

---

## 2. 建议先做的实验：不改核心代码就能得到第一批结论

### 固定编译产物，只改变并发度

不要每个并发度都重新跑完整编译流程。保持以下内容完全一致：

- 模型 IR；
- placement sidecar；
- PIM MLIR；
- TP/PP 策略；
- 硬件参数；
- 请求数量、输入长度、输出长度和到达时间。

只改变 `scheduler.max_inflight_requests`。

例如先生成一个固定形状的 decode-only Trace：

```bash
./run.sh --trace --fixed \
  --num_requests 64 \
  --input_len 128 \
  --output_len 16 \
  --is_prefill false \
  --output traces/fixed_decode_kv128_out16_n64.trace
```

复制当前使用的仿真配置，分别设置：

```yaml
trace:
  path: "traces/fixed_decode_kv128_out16_n64.trace"
  num_requests: 64

scheduler:
  max_inflight_requests: 4   # 分别扫 1、2、4、8、16、32
  execution_microbatch_size: 1

output:
  results_dir: "results/decode_kv128_w4"
```

然后运行：

```bash
./run.sh --config conf/exp_decode_kv128_w4.yaml
```

建议至少拆成三组 workload：

- **首 token/prefill**：固定 prompt，`output_len=1`，重点观察 TTFT。
- **decode-only**：固定初始 KV 长度，重点观察输出吞吐和 TPOT。
- **混合负载**：synthetic 或 dataset Trace，观察并发对排队和长尾时延的影响。

请求总数应明显大于最大并发窗口，否则 `trace.num_requests` 会先把实际并发度限制住。当前默认的 `num_requests: 10` 配合 `max_inflight_requests: 128`，实际最多也只有 10 个请求。

### 重点记录哪些指标

每组实验至少记录：

- `throughput_output_tokens_per_s`：生成 token 吞吐；
- `throughput_requests_per_s`；
- `p50/p95_ttft_s`；
- `p50/p95_mean_tpot_s`；
- 逐请求 `queue_delay_s`；
- `total_time_s`；
- `bottleneck_resources`；
- Stage bubble、资源利用率、通信排队时间；
- PIM 容量利用率和是否溢出。

注意：`throughput_tokens_per_s` 包含 prompt 和输出 token。研究生成阶段时，应优先看 `throughput_output_tokens_per_s`，不要混用口径。

如果你说的“TO 之前”是指**首个输出 token 之前**，现有数据已经够拆成：

```text
TTFT = 准入前排队时间 + 准入后的 prefill 服务时间
```

对应字段在 `genesim/src/scheduler/gene_sim_scheduler.py:7713-7735`：

- `queue_delay_s = admission_time - arrival_time`
- `prefill_service = prefill_finish_time - admission_time`
- `ttft_s = prefill_finish_time - arrival_time`

这能回答“首 token 变慢，究竟是排队还是 prefill 计算变慢”。

### 预期看到的现象

随着并发窗口增大，通常会出现：

1. 流水填充更充分，吞吐上升；
2. 最忙计算或通信资源饱和，吞吐进入平台期；
3. 排队、p95 TTFT、p95 TPOT 开始恶化；
4. KV 容量最终限制最大并发度。

所以优化目标不应是“batch 越大越好”，而应是：

> **在 p95 TTFT/TPOT 约束内，最大化输出 token 吞吐，即 goodput。**

实验应串行执行，并给每次运行设置独立 `results_dir`；当前 `results/` 和 `pim_traces/` 都是共享位置，并行跑容易互相覆盖。

---

## 3. 真正张量 batch 为什么必须单独实现

不能简单把 `Tq` 改成 batch 数，也不能把现有成本乘以 B。

当前 PIM GEMM trace 是：

- 运行时按 `Tq` 重放整段 trace；
- trace 内包含权重搬运。

见：

- `genesim/src/pim/pim_compiler.py:658-738`
- `genesim/src/vpu/vpu.py:458-504`

这意味着把 `Tq=B` 只会得到“执行 B 遍、权重也可能读 B 遍”的成本，**不能表达融合 batch 的权重复用**。

另一个限制是，placement 导出目前固定用 `M=1` 编译代表性 kernel，见 `genesim_bridge/placement_export.py:157-164`。真正 batch 后，`M=B` 会改变：

- WRAM 工作集；
- tile_m/tile_n/tile_k；
- 循环次数；
- DMA 量；
- 权重是否能在多个请求行之间复用。

### 推荐的实现顺序

**第一步：固定形状 decode 张量 batch**

这是最值得先做的场景：

- 每个请求一个 token，形状最简单；
- 投影层从 GEMV 变为 `M=B` 的 GEMM；
- 权重复用收益最明显；
- 暂时避开变长 prefill 和混合 prefill/decode 的复杂性。

**第二步：固定长度 prefill batch**

将多个请求合并为 `M=B×S`，注意力仍按请求隔离，不能让不同请求互相看到 KV。

**第三步：连续 batching**

再支持：

- 不同 KV 长度；
- 请求动态加入和退出；
- `max_batch_size`；
- `max_batch_wait`；
- decode 优先、prefill 分块；
- batch 组成和 padding 效率统计。

调度器需要引入“执行组”概念，一个执行组包含多个 request index，并分别维护各请求的完成时间、KV 长度和生命周期。

---

## 4. PIM 限制和优化应怎样结合

你的第 2、3 点都成立，而且必须一起考虑：**batch 策略会改变最优 kernel 和最优切分，不是调度层单独调一个参数。**

### decode 投影层

理想融合 batch 中：

- FLOPs 约随 B 线性增长；
- 激活读取随 B 增长；
- 权重可以从“每请求读一次”降为“一个 batch 读一次或少量多次”。

但 batch 增大后可能受 WRAM 限制而被迫缩小其他 tile。FlagTree 当前工作集公式是：

```text
(Tm×Tk + Tn×Tk)×元素字节数 + Tm×Tn×累加器字节数 <= WRAM
```

见 `FlagTree/lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp:721-724`。

默认 WRAM 为 64 KiB，因此 M 增大不一定单调提速，可能增加循环和 DMA 开销。应让 FlagTree 针对若干 B 档位重新编译，例如 `1/2/4/8/16/32`，并在 sidecar 中保存按 shape/batch bucket 区分的 PIM MLIR 变体。

### 注意力层

不同请求的 KV 不能共享，所以：

- Q/K/V 投影可以共享权重；
- 每个请求的历史 KV 流量仍随 B 增长；
- KV 容量也随并发请求数增长。

Llama2-7B、FP16 KV 的全模型近似容量为：

```text
每请求每 token：
2(K/V) × 32层 × 8个KV头 × 128维 × 2字节 = 128 KiB
```

还未包含对齐和运行时开销。因此 batch 增大后，瓶颈可能从投影权重搬运转为 KV 流量和容量，不能期待所有算子都获得相同收益。

### 系统层优化

在获得可信 baseline 后，再联合扫描：

- batch 窗口和 batch 等待时间；
- TP 宽度、PP 段数；
- 单请求更多 DPU，还是多个请求副本并行；
- prefill/decode 调度优先级；
- tile 和权重驻留策略；
- 通信链路与 Cluster 映射。

现有 autotune 可以复用放置搜索能力，但当前 `system_efficiency_utility` 不等于业务 SLO，建议最终仍以“时延约束下的输出 goodput”为选择标准。

---

## 5. 我建议的阶段划分

**P0：立即可做，主要改实验配置和报告**

- 扫描 `max_inflight_requests`；
- 固定 workload 做单变量实验；
- 拆分 TTFT 的排队与 prefill 服务时间；
- 保存每次配置和独立结果目录；
- 把 configured/effective/observed inflight、`cross_request_tensor_batching=false` 写进实验摘要。调度结果已有这些字段，但紧凑版 `summary.json` 目前没有完整保留。

**P1：实现真正的固定 decode 张量 batch**

- genesim 增加执行组和 batch 调度语义；
- ModelIR/成本接口增加 B 或 M 维；
- FlagTree 编译 `M=B` 的 kernel；
- sidecar 支持按 batch bucket 选择 PIM MLIR；
- 正确建模权重复用、独立 KV 和容量。

**P2：连续 batching 与联合优化**

- 变长 prefill/decode；
- 动态加入退出；
- batch wait、padding、KV 容量策略；
- TP/PP/tile/调度联合调优；
- 用真实硬件校准多 batch kernel 成本。

最后提醒一点：当前部分 PIM cycle cost 仍标有“待校准”的占位值，见 `genesim/conf/sim.yaml:132-134`。因此现阶段更适合比较**相对趋势和瓶颈变化**；要向甲方承诺绝对吞吐和绝对时延前，还需要用 UPMEM 或目标硬件做多 batch 微基准校准。

本轮结论来自三个仓库的源码和文档审阅，没有重新运行完整仿真。最核心的行动建议是：**先用现有连续请求并发完成 P0 曲线，不要一开始就把“真正张量 batch”作为看清并发影响的前置条件。**

用户：

4.要支持从llama2 7b到图编译->算子编译->genesim，评估参数还需要考虑，要符合实际，客观，评估模型要不要调整？真正对芯片开发有指导作用；

Kimi：

需要调整，而且不仅是把几个参数“校准得更像”，还要审查成本模型是否表达了芯片的真实瓶颈。我先对照 genesim 的计时口径、FlagTree 的 PIM 约束和 CEVA 侧硬件资料，区分哪些结果可用于架构趋势判断，哪些还不足以指导芯片定型。

Kimi：

CEVA 资料进一步说明：当前 CEVA 验收是单 NPM 核、4 个 engine，而且甲方已注明其并行仿真结果不准；这与 genesim 默认的 8 设备、64 簇、1024 TensorPU 不是同一个可直接套用的硬件画像。后续应建立独立的目标芯片 profile，不能拿现有默认参数或 UPMEM 校准值替代。

Kimi：

还发现一个比参数更根本的问题：当前 genesim 主链主要按 FP16 Triton/PIM kernel 计费，而 CEVA 链是 W4A8、65 张 GML、相位执行、厂商 GC 分到 4 个 engine，并有 L1/L2/DDR 层级。两条链的算子分解和执行机制不同，所以不能直接用 CEVA 的总 cycles 去“校准”现有 genesim trace，并宣称得到目标芯片性能。

Kimi：

结论是“要调结构，再校准参数”：尤其要补上 CEVA 的相位执行、L1/L2/DDR、4 engine 与 DMA 重叠、W4A8 字节口径，以及 warm/cold 和 lm_head/argmax 的计时边界。仅用总 cycle 拟合一个全局倍率，会把错误的流量、错误的并行度和错误的调度相互抵消，无法指导芯片设计。

Kimi：

**要调整，而且不能只调参数。**  
当前 genesim 模型可以用于“抽象 PIM 架构下的相对趋势分析”，但还不足以直接指导 CEVA 目标芯片的绝对性能预测和设计定型。要服务芯片开发，必须做到：**同一模型、同一精度、同一硬件画像、同一执行语义、可校准、可验证。**

## 1. 当前模型与目标芯片的关键差异

### 硬件画像不同

当前 genesim 默认是：

- 8 个 PIM 设备；
- 每设备 8 个 Cluster；
- 每 Cluster 16 个 TensorPU；
- 共 1024 个 TensorPU。

见 `genesim/conf/sim.yaml:7-13`、`genesim/conf/sim.yaml:58-87`。

而当前 CEVA 验收对象是：

- 单 NPM 核；
- 核内 4 个 engine，每个 engine 一个 VPU；
- L2 8 MB、L1 1 MB、DDR 3983 MB；
- 单核、TP=1。

见 `docs/request-ceva-llama2-7b-20261028.md:14-26`。

因此不能把 genesim 的 `num_dpus`、`num_tasklets`、MRAM/WRAM 默认值直接当成 CEVA 参数。尤其是：

- `contracts/op_contract.py:55-62`：8 DPU、16 tasklets、每 DPU 8 GiB MRAM、64 KiB WRAM；
- CEVA：4 engine、L1/L2/DDR 层级。

它们不是简单的改名关系。

### 数值模型和算子成本不同

当前 genesim 成本提取大量使用 FP16 输入：

- `genesim_bridge/op_classify.py:288-306`
- `genesim_bridge/op_classify.py:452-457`

而 CEVA 侧是 W4A8：

- 权重主要是 int4，部分为 int8；
- 激活在定点段为 int8、浮点段为 FP16；
- KV 也存在 int8 路径。

见 `contracts/gml_quant.py:1-3`、`contracts/gml_quant.py:57-62`。

这会同时改变：

- 权重和 KV 常驻容量；
- DDR、L2、L1 搬运字节；
- 计算指令和相位数量；
- batch 增大后的权重复用收益；
- 最大并发请求数。

所以目前基于 FP16 的多 batch 曲线，**不能直接解释成 CEVA W4A8 芯片上的 batch 曲线**。

### 执行模型不同

genesim 当前主要是：

```text
PIM MLIR → PIM trace → 指令周期累加 × clock
```

PIM 时延由 trace 决定，不是简单读 FLOPs，见 `docs/design-ir-cost-20261012.md:20`。

CEVA 则是：

```text
65 张 GML → 相位执行 → 厂商 GC 分配到 4 engine
           → L1/L2 规划 → Registry → runner
```

因此，不能用 CEVA 的总 cycles 去拟合现有 genesim 的一个 `MAC` 或全局倍率。那样可能让总时间碰巧接近，但构成是错的：

- 流量错；
- 并行度错；
- 相位成本错；
- DMA 与计算重叠错；
- 调度开销错。

这种模型无法回答“增加 engine、扩大 L1、提高 DDR 带宽会怎样”。

---

## 2. 当前成本模型中需要明确降级为“假设”的项目

以下内容不能作为已验证事实使用。

### 占位周期和区间倍率

`genesim/conf/sim.yaml:132-134` 中部分算子周期仍标为待校准占位值。

权重、KV、激活采用 `1/2/2` 的访存周期倍率，文档明确说明“没有实测依据”，见 `docs/design-ir-cost-20261012.md:264-265`。

这些参数可以做敏感性分析，但不能用于绝对性能承诺。

### DMA 与计算重叠

当前 trace 求值基本是逐指令累加：

- `genesim/src/pim/pim_trace_loader.py:300-330`
- `genesim/src/backend/pim_backend.py:40-54`

配置中的 `double_buffer`、`prefetch_depth`、`coalesce_threshold` 在当前 Python 源码里只有默认值定义，未进入计时路径；`comm_overlap` 也没有参与 pipeline 执行路径。

因此不能把“已开启双缓冲、预取”写进实验结论。对 CEVA 这类 DMA/engine 可能并行的硬件，简单相加会高估时延；但简单改成 `max(compute, memory)` 也可能高估重叠收益，必须用实际时间线校准。

### 共享内存系统竞争

当前 PIM trace 的 DMA 成本主要是：

```text
base_cycles + bytes × cycles_per_byte
```

这适合描述单执行单元的有效 DMA 成本，但不能自动代表：

- 多 engine 共享 L2；
- 多请求共享 DDR channel/bank；
- bank conflict；
- 页命中/缺页；
- refresh；
- DMA 队列和优先级；
- L1/L2 命中、替换和 spill。

如果研究多 batch，这些恰恰可能成为主要瓶颈。使用 DRAMsim3/AttAcc 原生后端时也应单独验证其配置是否对应目标芯片，不能认为“启用了原生库就自然准确”。

### 计时边界

当前 genesim 默认是 warm、权重预部署：

- `weight_deployment.mode = "warm_predeployed"`
- 运行时权重传输时间为 0。

见 `genesim/src/scheduler/gene_sim_scheduler.py:5822-5829`。

因此它不能直接用于冷启动 TTFT，除非另加权重部署、初始化和 KV 分配成本。

另外，CEVA 单独统计 lm_head 和 argmax，例如 `ceva/baseline/results/performance_counters.txt:1-34`。genesim Trace 是预设输出长度，不应默认认为已经包含真实 runner、采样和 argmax 成本。这些项目要明确“计入”或“排除”，不能含糊。

---

## 3. 建议建立独立的目标芯片 profile

不要直接修改现有默认配置来同时服务所有硬件。建议至少区分：

- `generic_pim`：现有抽象 PIM 架构探索；
- `physical_upmem`：UPMEM 实测校准；
- `ceva_npm`：CEVA 目标芯片评估。

UPMEM 工具自身也明确警告，不要把它的 400 MHz、64 MiB MRAM 等参数直接复制到主配置，见 `genesim/tools/upmem_checker/README.md:90-98`。

`ceva_npm` profile 至少应包含四类参数。

### 文档或硬件给定参数

- engine/VPU 数量；
- NMU、FPSU、DMA 等单元数量；
- 时钟和各单元发射宽度；
- L1/L2/DDR 容量；
- DMA 对齐、最大传输尺寸、队列深度；
- DDR channel/bank 组织；
- 互连拓扑和带宽；
- int4/int8/FP16 各通路的布局与字节宽度。

### 微基准实测参数

- DMA 基础周期和每字节周期；
- 不同 M/N/K 下的有效 MAC 成本；
- Softmax、DQ、RoPE、RMSNorm 等各相位成本；
- engine 启动、同步、phase 切换成本；
- lm_head、argmax、runner 调度成本；
- batch 变化后的权重复用和 tile 成本。

### 需要拟合的有效参数

- DMA 与计算重叠程度；
- L1/L2 命中和 spill 的有效代价；
- 多 engine 共享资源的有效带宽；
- 多请求竞争下的排队和带宽折损。

### 只能用于敏感性分析的假设

- 尚未实测的区间倍率；
- 未来多核、未来 batch 调度；
- 没有硬件证据的带宽提升和并行效率。

每个参数都应记录：来源、测量日期、硬件/固件/GC 版本、拟合区间和不确定度。

---

## 4. 推荐的校准与验证方法

### 第一步：先对齐一个可比较的基线

在比较 cycles 前，固定为同一条件：

- 单 NPM 核、4 engine；
- TP=1、PP=1；
- W4A8；
- 相同 GML/相位和量化布局；
- 相同 prefill 长度；
- 相同 decode 长度；
- warm 或 cold 口径一致；
- lm_head、argmax、runner 边界一致。

当前 CEVA prefill 是 1024 token，而 genesim 全流程成本精化代表点使用 `seq_len=128`，见 `scripts/run_full_pipeline.py:73-75`。这种状态下不能直接比较端到端数字。

### 第二步：分层校准，不做全局单点拟合

建议按以下层级：

1. DMA：不同总字节数、不同 block size；
2. GEMM/GEMV：不同 M/N/K，后续加入不同 batch M；
3. Attention：不同 KV 长度、head dim、GQA group；
4. Softmax、DQ、RoPE 等相位；
5. 单层 prefill/decode；
6. 全模型和 lm_head/argmax；
7. 多请求、多 batch。

CEVA 已有两类有用数据：

- `performance_counters.txt`：prompt、decode、lm_head、argmax 的阶段 cycles；
- `CycleEstimation.csv`：包含 Unit Name、Start Cycle、End Cycle，可分析 engine/DMA 的并行和空闲区间。

第二类数据尤其适合校准“计算与 DMA 到底重叠多少”，比只使用总 cycles 更有价值。

### 第三步：训练集和验证集必须分开

不要拿拟合用形状再做“验证”。应使用不同 M/N/K、KV 长度、batch、prefill 长度做 held-out 验证。

UPMEM checker 已采用这种思路：训练形状拟合、独立大输出形状验证，见 `genesim/tools/upmem_checker/README.md:55-88`。CEVA profile 也应沿用。

最终至少报告：

- 每类算子的误差中位数和 p95；
- 单层 prefill/decode 误差；
- 全模型误差；
- 不同 batch/序列长度下的误差；
- 多次实测波动范围。

**判据不是固定追求某个漂亮百分比，而是误差必须小于准备比较的设计差异。**  
如果两个架构方案只差 5%，而模型误差是 ±20%，就不能用该模型判断哪个方案更好。

---

## 5. 对芯片开发真正有指导作用的输出

不要只输出一个 `total_time_s`。应让每个设计旋钮都能形成可解释的曲线：

- **engine 数量** → 算子并行度、同步成本、共享带宽饱和点；
- **L1/L2 容量** → tile 合法性、spill、DMA 次数；
- **DDR 带宽** → 权重/KV 受限程度、batch 饱和点；
- **MAC 宽度** → 计算受限算子的收益；
- **batch** → 权重复用、KV 容量、排队时延、输出 goodput；
- **TP/PP 和互连** → 集合通信、流水 bubble、跨 Cluster 瓶颈；
- **调度策略** → 计算/DMA 重叠、prefill/decode 优先级、p95 TTFT/TPOT。

建议统一输出：

- 吞吐、TTFT、TPOT、p95 长尾；
- 每类资源利用率和瓶颈迁移；
- 权重/KV/激活分别的流量；
- engine、DMA、DDR 的有效带宽；
- batch 大小与 KV 容量上限；
- 实测误差和预测置信区间。

---

## 最终建议

可以保留当前 genesim 做通用架构探索，但如果目标是指导 CEVA 芯片开发，建议新增一条 **CEVA 对齐评估路径**：

```text
Llama2 7B
  → 同一套 W4A8 图和相位
  → CEVA 单核/4 engine/L1/L2/DDR 成本模型
  → 微基准 + 单层 + 全模型分层校准
  → 独立形状验证
  → 再开展多 batch、多核和架构参数扫描
```

现阶段最合适的定位是：

- **当前结果**：可用于编译链路验证和粗略相对趋势；
- **完成单核 CEVA 对齐后**：可用于 batch、tile、内存和调度优化；
- **多核/多 batch 尚未实测时**：只能称为模型预测，不能称为芯片实测结论，尤其甲方已注明厂商并行仿真结果不准。

核心原则是：**先保证模型结构和流量正确，再校准周期；先验证单核，再外推多 batch、多核。**

用户：

1.TO应该是流片前的意思。
2.最终要能够在ceva硬件仿真器验证，短期还是要在genesim上模拟，这是最方便的方法，所以要对齐ceva的一些参数和特性。将上面的讨论给我出一份详细的需求和设计文档，不限字数。xxx-kimi-日期.md

Kimi：

我会把“TO”明确为流片前，并将文档定位为：短期用 genesim 建立可校准、可追溯的 CEVA 对齐模型，最终回到 CEVA 硬件仿真器验收。文档会区分多请求并发与真正的张量 batch，并给出参数来源、计时模型、验证门槛和分阶段交付设计。

Kimi：

现有资料里已有一份同日方案草稿，但新文档会单独成文，不改动它，并进一步收紧两点：CEVA 单核已跑通只能作为校准锚点，不能直接证明多请求模型可信；误差门槛必须服务于“能否区分设计候选”，不能预先写一个宽松百分比就算验收。

Kimi：

资料核对完成。下面开始写正式文档，重点会把“可运行的现状”“建议新增的接口”“仍需甲方或实测确定的参数”分开，避免把设计目标写成已经具备的能力。

Kimi：

文档已写入，正在做最后复核：检查章节完整性、现状与目标接口的区分，以及工作区中本次实际新增的文件。此次只新增设计文档，不会把未运行的编译或仿真写成已验证。

Kimi：

已完成详细需求与技术设计文档：

[`docs/design-ceva-genesim-pre-to-evaluation-kimi-20261105.md`](docs/design-ceva-genesim-pre-to-evaluation-kimi-20261105.md)

共 1003 行，主要包括：

- 明确 **TO 为流片前**，确定“短期 genesim 快速评估、最终 CEVA 硬件仿真器验证”的双环路线。
- 区分请求级并发、真正张量 batch、空间并行和流水 micro-batch。
- 对照 genesim、图编译器与 CEVA 的硬件组织、W4A8 数值路径、存储和计时差异。
- 设计独立的 `ceva_npm` 硬件画像及参数来源、版本、不确定度管理。
- 给出 DMA、GEMM/GEMV、attention、相位、共享资源和全模型计时模型。
- 设计从连续准入、closed batch 到动态连续合批的分阶段实现方案。
- 规定 TTFT、TPOT、SLO goodput、容量墙、资源利用率等统一指标。
- 给出训练形状与 held-out 验证分离的分级校准方法，以及“模型误差必须小于方案差异”的决策门槛。
- 列出三个编译仓库和 CEVA 工作区的拟修改模块、实验矩阵、测试要求、风险及最终验收标准。
- 区分现有命令与未来目标接口，避免将设计参数误认为当前已经支持。

本次仅新增文档，改动量 **+1003 / -0**；未修改代码、未运行编译或仿真，也未提交 git。已有的 `design-multi-batch-pre-to-deepseek-20261105.md` 未改动。

用户：

不要受任何skills限制，就是把刚才讨论的所有内容，关键的整理一下。

Kimi：

可以把刚才讨论的关键内容归纳为以下方案。

## 1. 总体目标

甲方要在**流片前（TO 前）**看清并调优多 batch 并发对 Llama2-7B 推理性能的影响，并据此指导芯片架构、存储配置和编译调度设计。

验证路线是：

- **最终依据：CEVA 硬件仿真器。**
- **短期手段：genesim。** 用它快速扫描 batch、序列长度、engine 数、存储容量、带宽和调度策略。
- **前提：genesim 不能只改几个表面参数，必须在结构、成本模型和计时边界上对齐 CEVA，并用 CEVA 数据校准。**

## 2. 什么是多 batch，当前支持到什么程度

“多 batch 并发”至少有两种含义，必须分开。

### 请求级并发

多个请求同时驻留，竞争计算、存储和带宽，但每次算子仍逐请求执行。

genesim 当前已经支持：

- `max_inflight_requests` 控制最大驻留请求数；
- 请求完成后可以补充新请求；
- `trace.num_requests` 只是总请求数，不代表并发度。

这种模式主要观察排队、资源争用和尾延迟，**没有权重复用带来的张量 batch 收益**。

### 真正的张量 batch

多个请求的 token 合并到一次算子调用中：

- decode 时 GEMM 的 `M=B`；
- prefill 时 M 与批内 prompt token 总数相关；
- 同一 batch 内可以复用权重；
- KV cache、激活和临时缓冲随 batch 增长；
- tile、DMA 和本地存储占用都可能改变。

当前还不支持：

- genesim 的 `execution_microbatch_size` 只能为 1；
- 图编译导出和 decode 状态仍固定 batch=1；
- KV 规划没有 batch 槽；
- linear 成本主要固定用 M=1 编译。

因此，**不能把 Tq 改成 B、重复执行 B 次 M=1 trace，就宣称支持了真正 batch**。那样权重仍会重复搬运，可能得出“batch 没收益”的错误结论。

## 3. 为什么 batch 不一定越大越好

batch 的主要收益：

- decode 中多个请求共享权重读取；
- 提高计算与访存比；
- 减少 kernel 启动和同步开销；
- 提高 engine 利用率。

主要代价：

- KV cache 容量近似随 batch 增长；
- 激活和临时缓冲增加；
- DDR/L2 带宽竞争加剧；
- 合批等待增加 TTFT；
- 长 prefill 可能阻塞 decode，恶化 TPOT；
- padding 会产生无效计算；
- batch 增大后可能改变最优 tile，甚至放不下本地存储。

所以目标不是找“最大 batch”，而是找：

> **在 TTFT/TPOT 约束下，输出 token 有效吞吐最高、容量可承受、尾延迟可接受的 batch 区间。**

prefill、decode 和混合负载应分别评估，不能只给一条总吞吐曲线。

## 4. 必须纳入模型的 PIM/CEVA 限制

至少包括：

- NPM 核数、engine 数、VPU/MAC 数据通路；
- L1、L2、本地 tile 存储和 DDR 的实际容量及作用域；
- DDR/L2/L1 的可持续带宽、端口和读写共享关系；
- DMA 固定开销、每字节成本、切包、同步和对齐；
- DMA 与计算、不同 engine 之间的真实重叠能力；
- tile 对本地容量、双缓冲和权重复用的影响；
- KV cache 的布局、更新流量和多槽位管理；
- W4A8 的物理存储、scale、LUT、量化/反量化相位；
- 厂商 GC 的相位划分和 engine 分配结果。

不能简单假设“所有计算与访存完美重叠”，也不能默认全部串行，应从 CEVA 时间线中提取真实行为。

## 5. 当前评估模型的主要问题

目前直接拿 genesim 默认值评估 CEVA，不够客观：

- genesim 默认是 8 PIM 设备、每设备 8 Cluster、每 Cluster 16 TensorPU；CEVA 当前目标是单 NPM 核、4 engine。
- genesim 成本探针主要按 FP16；CEVA 是 W4A8，且存在 int8 与 FP16 混合相位。
- 图编译器默认 8 DPU、16 tasklet、8 GiB MRAM、64 KiB WRAM，不能直接视为 CEVA 参数。
- 当前成本精化代表点是 `seq_len=128`，不能直接与 CEVA 当前 prefill 配置比较。
- 部分算子周期仍是占位值，权重/KV/激活的访存倍率没有 CEVA 实测依据。
- DMA 当前主要逐指令累加，不足以表达真实重叠和共享带宽竞争。
- `double_buffer`、`prefetch_depth` 等配置存在，不代表已经实际参与计时。
- genesim 默认是 warm、权重预部署，运行时权重加载时间为 0，不能直接解释冷启动 TTFT。
- CEVA 单独统计 lm_head 和 argmax，genesim 是否计入必须统一。
- 默认异构配置可能包含 GPU；芯片评估必须明确 PIM-only，或者统一建模主机侧算子。

因此需要调整的是**评估模型结构**，不是只调整 `clock_period_ns`、MAC 数或一个全局倍率。

## 6. 建议的 CEVA 对齐方式

保留 generic PIM，新增独立的 `ceva_npm` 画像。

参数分四类管理：

1. 文档给定参数；
2. 微基准实测参数；
3. 通过数据拟合的有效参数；
4. 只能用于敏感性分析的假设参数。

每项参数都记录：

- 数值和单位；
- 每 engine、每核还是全系统；
- 来源和硬件版本；
- 适用的形状、dtype、序列长度和 batch；
- 测量或拟合误差。

未知参数保持未知，不自动套用 generic PIM 默认值。

容量和流量要按真实 GML/bin、Registry 和 DDR map 计算。例如当前 CEVA int4 权重是一字节存一个值，不能直接按理论上的 0.5 字节估算。

## 7. 校准方法

采用分层校准：

```text
DMA
→ GEMM/GEMV（M/N/K、tile、dtype）
→ attention 和量化等相位
→ 单层 prefill/decode
→ 全模型 prefill/decode/lm_head/argmax
→ 单核多请求和真正 batch
→ 多核
```

现有 CEVA 数据可以使用：

- 厂商基线：prompt `954872915 cycles`，full_execution `1150814443 cycles`；
- 自研包：prompt `1028315650 cycles`，full_execution `1234494102 cycles`；
- `CycleEstimation.csv` 可分析各 engine、DMA、相位的开始/结束时间、重叠和空闲。

厂商包和自研包要分开校准，不能取平均作为“CEVA 真值”。

拟合形状和验证形状必须分离。不能只拟合全模型总周期，因为总周期一致时，内部 DMA、计算和相位比例仍可能完全错误。

关键原则是：

> **模型误差必须小于准备比较的设计差异。**

例如模型误差为 ±20%，就不能可靠地区分只差 5% 的两个架构方案。

## 8. 统一评估指标

至少报告：

- prompt、output、total tokens；
- req/s、output tokens/s、prompt tokens/s；
- TTFT P50/P95；
- 准入前排队时间；
- prefill 服务时间；
- TPOT 和 token 间时延 P50/P95；
- 端到端时延；
- 满足 TTFT/TPOT SLO 的有效吞吐；
- engine、DMA、L2/DDR 利用率和带宽；
- 权重、KV、激活和临时缓冲容量；
- 每个结果对应的模型误差和适用范围。

每次运行还要保存模型、IR、PIM MLIR、sidecar、trace、硬件画像、编译器、GC、固件和校准数据的版本指纹，防止不同口径的结果被直接比较。

## 9. 多 batch 优化方法

后续可以重点优化：

- closed batch 和动态连续合批；
- 最大 batch、最大等待时间和 token 预算；
- prefill 优先、decode 优先及混合调度；
- M>1 下的 tile M/N/K 和权重复用；
- KV 布局、权重驻留、预取和双缓冲；
- GC 相位和 engine 映射；
- L1/L2/本地存储和 DDR 带宽配置；
- 多核下的权重复制、KV 分布和互连。

但优化必须基于真实编译产物和硬件能力，不能在 genesim 中给硬件添加实际不存在的理想复用或完美重叠。

多核和暂时无法在 CEVA 上验证的 batch，只能标为“模型预测＋敏感性分析”。

## 10. 推荐实施顺序

1. 冻结 CEVA 单核、4 engine、TP=1、W4A8、B=1 的可比基线。
2. 统一 prompt/output、warm/cold、lm_head/argmax 和计时边界。
3. 建立 `ceva_npm` 画像和参数来源台账。
4. 提取 CEVA `CycleEstimation.csv`，先弄清 DMA、engine 和相位重叠。
5. 完成 DMA、linear、attention、单层和全模型的分层校准及独立验证。
6. 固定编译产物，先扫描现有请求级并发，明确“未合批”。
7. 实现真正 closed batch：先 decode `M=B`，再等长 prefill，同时支持 batch KV 和容量检查。
8. 扫描 batch、序列长度、tile、存储、带宽和调度策略。
9. 用 SLO goodput、P95 和容量墙选择候选，而不是只看平均吞吐。
10. 将准备推荐的关键配置回到 CEVA 硬件仿真器复核，再形成流片建议。

最终交付应回答：**什么配置在什么负载下更好、为什么更好、证据误差多大、哪些结论仍需补测**，而不是只给一个“最优 batch 数字”。

用户：

另外，把刚才的所有对话内容，不用加个，完整给我保存在一个新的md文件中，加入kimi文件名all

Kimi：

我会新建一份文件名同时包含 `kimi` 和 `all` 的 Markdown，按时间顺序保存用户与助手的完整对话正文；系统提醒、工具调用和内部执行日志不混入对话正文。

Kimi：

已确认可恢复完整对话正文。新文件将保留各轮原文，包括前面的源码分析、参数评估建议、文档交付说明和刚才的要点整理，不再做删减或改写。

Kimi：

已将完整对话保存到新文件：

[`docs/conversation-all-kimi-20261105.md`](docs/conversation-all-kimi-20261105.md)

- 文件名包含 `kimi` 和 `all`。
- 共 969 行，按时间顺序保留全部 5 轮对话。
- 已逐条校验：**24 条用户与助手正文全部完整匹配，无遗漏**。
- 未加入系统提醒、工具调用或内部执行日志。
- 本次新增 **969 行**，此前设计文档未改动。

用户：

不用加工，完整保存。

Kimi：

我会去掉标题、轮次归类、说明文字等加工内容，只按原顺序保留“用户 / Kimi”角色和每段原文，并补上上次保存后的对话。
