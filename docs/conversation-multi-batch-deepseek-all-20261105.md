# 会话记录（原始内容，未加工）

> 记录时间：2026-11-05
> 说明：以下按消息顺序原样保存用户与助手的可见对话内容，未做摘要、改写或删减。

---

## 消息 1（用户）

你是哪个版本

---

## 消息 2（助手）

我是 Claude Code（终端里的 CLI 工具）。这个会话的环境元数据里标明的模型是 `deepseek-flash[1m]`，我只能看到这个标识符，无法进一步验证底层模型的具体版本。

另外可以确认的上下文：当前工作目录是 `/data/fjg/flagOS/flagos-pim-compiler`（分支 `main`，工作区干净），最新提交是 `1226e96 support ceva Simulator`。

---

## 消息 3（用户）

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

---

## 消息 4（助手）

先给结论，再展开。我把 **TO 理解为流片（tape-out）**，如果不是这个意思请纠正，这会改变交付物形态。

## 结论先行

1. **"多 batch"不是一个开关，要拆成四层**：请求级并发、张量 batch、空间并行（多组 DPU 各服务一个 batch）、流水 micro-batch。当前三仓只实现了第一层，第二层完全没有。
2. **现在直接调 `max_inflight_requests` 得不出 batch 结论**：genesim 的 GEMM 成本是"按 token 重放一次 tile_m=1 的 pimir"，B 个请求就是把同一份权重搬运重复 B 次，batch 在模型里只会增加争用，不会体现权重复用收益。
3. **正确顺序是：先把成本口径钉住（batch 维 + 权重复用 + KV 容量），再扫参**。执行器跑通 B>1 的数值对齐可以后置，但必须留 B=1 回归锚点和至少一个 B>1 的算子级实测锚点。
4. **TO 前要交付的不是一个数，而是曲线和敏感度**：吞吐-时延随 batch 的变化、容量墙在哪、哪个架构参数最敏感。PIM 限制要分档看，不同 batch 区间由不同瓶颈主导。

---

## 1. "多 batch"到底有几种

| 并发方式 | 含义 | 当前支持 | 性能影响 |
| --- | --- | --- | --- |
| A. 请求级并发 | 多个请求同时在飞，按到达时间连续准入；每个 forward 的算子仍是独立任务，共享计算/链路 | **genesim 支持**（`max_inflight_requests`） | 反映排队、争用、尾延迟；**不反映 batch 的计算效率** |
| B. 张量 batch | 多个请求的 token 拼成一个 M 维，一次 kernel 算完，权重只读一次 | **不支持**（`execution_microbatch_size` 强制 1） | PIM 上最大的收益来源，尤其是 decode |
| C. 空间并行 | 把 DPU 分成多组，每组独立服务一个 batch/请求（数据并行） | 只有全局放置，没有“按 batch 分组” | 用容量换并行度；权重复制 vs KV 分布的权衡 |
| D. 流水 micro-batch | PP 的不同 stage 同时处理不同 micro-batch | 有 Stage 视图，但没有 micro-batch 重叠调度 | 减少流水气泡；与 B 正交 |

甲方说的"多 batch 并发"，通常同时包含 **A（并发在飞）和 B（合批执行）**，有时还包括 **C（多组并行）**。如果只做 A，得到的结论是"batch 越大越慢/越堵"；如果只做 B，得到的是"合批的效率上界"。两者要分开报告。

---

## 2. 三仓现状（已核对源码）

| 仓库 | 位置 | 事实 |
| --- | --- | --- |
| genesim | `src/scheduler/gene_sim_scheduler.py:5231-5259` | 连续准入窗口 `max_inflight_requests`；`execution_microbatch_size != 1` 直接 `NotImplementedError`，注释写明 cross-request tensor batching 未实现 |
| genesim | `src/scheduler/gene_sim_scheduler.py:7031-7054` | 请求按到达时间入队，空槽即准入；每个 forward 的每个算子仍是 `(request_index, forward_step, op_id)` 独立任务 |
| genesim | `src/vpu/vpu.py:458-473, 534-562` | GEMM 的 trace 执行次数 `repetitions = 输入前导维乘积 = Tq`；`loop_count = ceil(N/tile_size)` |
| genesim | `models/llama2_7b.ir` 和 `src/ir/model_ir.py:1077` | 符号形状只有 `Tq`/`Tp`，**没有 batch 维**；`build_from_hf_config` 直接产出 `("Tq", hidden)` |
| 图编译器 | `runtime/compile.py:78, 216-218` | prefill 导出 `input_ids=[1, seq]`，decode 导出 `[1, 1]`；`compile_llama2(...)` 没有 `batch_size` 参数 |
| 图编译器 | `memory/kv_layout.py:20-54` | `KVRegionSpec` 只有层、KV head、`max_seq`、`head_dim`，`kv_bytes = 2 × layers × max_seq × kv_heads × head_dim × dtype`，**没有 batch 槽** |
| 图编译器 | `runtime/executor.py:15-23, 41-89` | `DecodeState.valid_len` 是单序列；decode 循环一次只送 `[1,1]` |
| 算子编译器桥 | `genesim_bridge/op_classify.py:100`、`cost_extractor.py:173` | 成本代表点只有 `ShapePoint(tq, tp)`；linear 的编译形状是 `[(Tq, in), (out, in)]`，**没有 batch** |
| FlagTree | `docs/tile-to-budget-flaggems-linear-20261002.md` | tiling 预算按 WRAM 65536B；128×128×32 f16 已占 81920B，超预算；grid 切分内核只报分块不重建 |
| CEVA | 设计/实施文档 | GML 和 runner 都是 batch=1；厂商仿真单核运行，且并行仿真被标为不准确 |

**一句话**：请求级并发已经有了，而且默认 `./run.sh` 的 10 条请求就是同时在飞的；但张量 batch 从 IR、成本、KV、执行器到算子 kernel，整条链都没有。

---

## 3. 最关键的坑：现在的成本模型里 batch 收益恒为零

我抽了一份实际 GEMM 的 pimir（`.opcompiler_cache/143324fcfb1ff6b8.pimir.mlir`）：

- `pim.tile-m = 1`、`pim.tile-n = 512`、`pim.tile-k = 32`；
- 每次执行加载 A `1×32`、B `512×32`、输出 `1×512`；
- genesim 对 decode 的 Tq=1 执行 1 次，对 prefill 的 Tq=128 执行 **128 次**。

也就是说，prefill 被建模成 128 次单 token GEMV 顺序重放，**权重 tile 被搬运 128 次**。真实的 batch-B GEMM 可以做到：外层 N、内层 M/K，权重 tile 每个 N/K 块只搬一次，B 行复用。当前模型里这个收益**结构上不存在**。

后果很严重：

- 你扫 batch=1/2/4/8，得到的吞吐曲线只会由"资源争用 + KV 容量"决定，**不会出现权重复用带来的上扬**；
- 如果据此得出"batch 没收益"或"batch 越大越差"，对 TO 前的架构决策是误导；
- 反过来，如果只按解析公式把 `Tq` 改成 B，成本会线性乘 B，同样看不到收益。

**因此，第一批实验必须是"成本口径验证"，不是"batch 扫描"。** 具体做一件事就够：

> 用 FlagTree 对同一个 linear 分别按 M=1、M=8（N/K 相同）编译 pimir，比较 `tile_m`、`mram_traffic_bytes`、权重 DMA 次数。如果 M=8 的权重搬运不是接近 1×，说明算子编译器还没有生成"batch 复用权重"的 kernel，那么任何批量调度都是空谈；先改 kernel 的循环顺序/tile_m，再谈仿真。

这也解释了你第 3 个问题：**多 batch 优化不是"把 batch 调大"，而是"让 kernel 和调度真的利用 batch"**。

---

## 4. PIM 约束怎么进 batch 模型

约束要分档看，不要把所有限制塞进一个"有效算力"系数。

| 约束 | 当前配置/证据 | 对 batch 的影响 | 主导区间 |
| --- | --- | --- | --- |
| MRAM/DPU 容量 | 图编译器 `contracts/op_contract.py:56-62`：8 DPU × 8GiB，WRAM 64KiB；genesim `conf/sim.yaml:146`：每个 ClusterPU 8GB，16 TensorPU 均分 → 512MiB/TensorPU | 权重 + KV + 激活三区；batch 线性增加 KV 和激活 | **大 batch 的硬墙** |
| KV 容量 | Llama2-7B fp16：2×32层×32头×128×L×2B；L=2048 → **1GiB/请求**；W4A8 权重约 3.5GB，单 CEVA core DDR 约 4GB | 2K 上下文下，单核几乎放不下"权重 + 多个请求的 KV" | 大 batch / 长上下文 |
| WRAM | 64KiB/DPU；128×128×32 f16 = 80KiB 已超（FlagTree 文档实例） | batch 增大只能增加 M 方向 tile 数，**不能把 tile 变大**；权重复用受 WRAM 能同时驻留几个 tile 限制 | 中 batch 的核心 |
| 循环寄存器 | 12-bit，最大 4095；`max_tiles_per_pu=4096`（`conf/sim.yaml:149`） | `batch × seq / tile` 超过 4095 需要 long loop 或拆分 | 长序列 × 大 batch |
| 带宽 | intra-cluster 512 GB/s、inter 128 GB/s、host 38.4 GB/s（`conf/sim.yaml:159-213`） | batch 降低"权重字节/token"，但激活和 KV 字节/token 不变；跨段/跨设备搬运随 batch 增加 | 大 batch 的次生瓶颈 |
| tasklet | 16/DPU | decode 的 M=B 很小时，M 维并行度不足；batch 提高利用率 | 小 batch 的收益来源 |

**分档结论**（TO 前最该画的三段曲线）：

1. **小 batch**：权重复用收益大，但 M 太小、tasklet/PU 利用率低；吞吐随 B 近似线性上升。
2. **中 batch**：WRAM 能驻留的权重 tile 数决定复用上限，compute 和 MRAM 带宽接近饱和；吞吐增速变缓。
3. **大 batch**：KV 容量和带宽成为墙，出现排队、尾延迟上升；吞吐可能不升反降。

甜点区就是第 2 段和第 3 段的交界，**它由架构参数（WRAM、MRAM、带宽、PU 数）决定**——这正是 TO 前要回答的问题。

---

## 5. 建议路线（最小改动、逐步闭环）

### Phase 0：不改代码，先做请求级基线（1–2 天）

用 `--fixed` 生成同形、同时到达的请求，扫 `max_inflight_requests`：

```bash
for B in 1 2 4 8 16; do
  ./run.sh --trace --fixed --num_requests $B \
    --input_len 2048 --output_len 128 --is_prefill false \
    --fixed_arrival_time_ns 0 --output traces/fixed_b${B}.trace
  # 复制一份配置，把 scheduler.max_inflight_requests 设为 $B，trace.path 指向该文件
  ./run.sh --config conf/sim_b${B}.yaml
done
```

看 `results/summary.json` 的 `throughput_tokens_per_s`、`average_ttft_s`、`p95_mean_tpot_s`，以及 `pu_metrics.json` / `memory_metrics.json` / `pim_capacity_metrics.json`。

**这条曲线的定位要写清楚：它是"无张量 batch 的请求并发基线"，不是 batch 结论。** 它的价值是给出 B=1 回归锚点和容量墙位置。

### Phase 1：把 batch 进成本链（推荐首先做）

目标：让 closed static batch 的 B 扫描能反映"权重只读一次"的收益。需要三件事：

1. **IR/成本维**：给 GeneSim 的符号体系加 `Tb`（或约定 batch 组把 `Tq` 解释为"本步总 token 数"），并让 FlagTree 按 `(Tb, Tq, Tp)` 编译/量测。当前 `ShapePoint` 只有 `Tq/Tp`，这是入口。
2. **phase 对应的 pimir**：prefill 和 decode 各用各的 pimir，不能拿 tile_m=1 的 decode kernel 重放 Tq 次。`repetitions` 要和 `tile_m` 对齐（`ceil(M/tile_m)`，或 pimir 覆盖全 M 时为 1）。
3. **权重搬运口径**：确认 pimir 的 DMA 统计里权重按 N/K tile 只加载一次，而不是按 M 行重复。

验收：B=1 结果与现有结果逐位一致；B=8 的 GEMM 权重字节/token 明显低于 B=1；至少一份 FlagTree 实测 pimir 作为锚点。

### Phase 2：图编译器/执行器支持 B>1（数值闭环）

- `compile_llama2(batch_size=...)`，导出 `[B,S]` / `[B,1]`；
- `KVRegionSpec` 加 batch 槽，`DecodeState` 变成 per-slot `valid_len`，`kv_bytes` 乘 B；
- 内存规划、通信计划、ExecutionPlan 的形状乘 B；
- 分片策略：**batch 维默认 replicate，不切开**；如果要做空间并行（C），明确它是数据并行（权重复制），并把权重占用算进去。

先用"所有请求等长、同时到达"的 closed batch 对齐 PyTorch，再做变长 padding；不要把连续 batching 的复杂度混进这一步。

### Phase 3：连续 batching 与调度

- genesim 调度器把同一 stage/同一 op、同一时刻 ready 的多个请求合并成一个批任务（M=sum tokens）；
- 处理不同 `Tp` 的 padding/mask、prefill chunking、prefill/decode 混排；
- 这才是"多 batch 并发对整体性能影响"的完整模型。Phase 1 的 closed batch 是它的上界/基线。

**不要从 Phase 3 开始**：没有 Phase 1 的正确成本口径，合批调度只会把错误的 per-token 成本乘起来。

---

## 6. 多 batch 优化方法清单（按 PIM 收益排序）

1. **权重复用 / batched kernel**：让 `tile_m > 1` 或改变循环顺序（N 外层、M 内层），使权重 tile 在 M 行间复用。QKV 融合、gate/up 融合已经做了，下一步是"batch 内的权重驻留"。
2. **tile 目标函数升级**：FlagTree 的 `pim-tile-to-budget` 现在只保证"放进 WRAM"；应把目标改成 batch 下的 latency / MRAM traffic 最优（M/N/K 与 B 耦合）。
3. **KV 容量管理**：按 `(batch, seq_len)` 算最大并发表；paged/slot KV、按 SLO 准入控制。TO 前这张表比任何单个吞吐数都重要。
4. **并行映射**：TP（切权重）vs DP（切 batch、复制权重）vs 混合。判据是 MRAM 容量和带宽：权重放得下就 DP 提并发，放不下就 TP 省容量。
5. **调度**：continuous batching、chunked prefill、prefill/decode 分离或混排；用到达率扫描，而不是固定 B。
6. **量化**：W4A8 把权重流量减半，最优 batch 会右移；fp16 和 W4A8 要分别扫。
7. **长序列**：12-bit loop 和 `max_tiles_per_pu` 是硬限制，长上下文 × 大 batch 需要 long loop 或序列分块。

---

## 7. TO 前建议交付的东西

- **曲线**：吞吐 vs batch、TTFT/TPOT vs batch、MRAM traffic/token vs batch、PIM 利用率 vs batch；
- **容量墙**：`(batch, seq_len)` 的最大并发表，标出权重/KV/激活各占多少；
- **敏感度排序**：WRAM、MRAM 容量、带宽、PU 数、tile 策略，哪个对甜点区影响最大；
- **明确假设**：workload 到达率、prompt/output 分布、SLO（TTFT/TPOT 目标）、量化格式；
- **校准锚点**：至少一个 FlagTree 的 B>1 算子实测，以及 CEVA 单算子/单层数值锚点。

---

## 8. 对你三条理解的逐条回应

1. **"首先要支持多 batch"**：方向对，但不是"先把执行器跑通 B>1 再研究"。当前缺的是**成本口径里的 batch 和权重复用**；先补这个，才能判断值不值得改执行器。请求级并发已经存在，不要把它当成 batch。
2. **"要考虑 PIM 的尺寸、寄存器、带宽"**：对，但要分档。小 batch 看权重流量和利用率，中 batch 看 WRAM/tile，大 batch 看 KV 容量和带宽。把它们混成一个"有效算力"会掩盖真正的墙。
3. **"要考虑多 batch 并行的优化方法"**：对，但优化对象是"权重复用 + KV 容量 + 并行映射 + 调度"，不是单纯调大 `max_inflight_requests`。

---

## 9. 需要你确认的三点

1. **TO 是流片吗？** 如果是流片，上面的"架构参数敏感度 + 容量墙"就是主线；如果 TO 是 timeout/SLO，那主线会变成"满足 TTFT/TPOT 前提下的最大并发"。
2. **目标架构是 CEVA 单核 + 4 engine，还是 genesim 默认的 1024 TensorPU？** 两者差得很远：CEVA 单核 DDR 约 4GB、L2 8MB、L1 1MB；genesim 默认 8 设备 × 8 簇 × 16 PU、每 PU 512MiB。做 batch 结论前必须把仿真配置对齐到目标，否则曲线是另一台机器的。
3. **甲方最关心吞吐、TTFT/TPOT，还是 SLO 达标率？** 这决定扫描的自变量是 batch 还是到达率。

另外提醒两个已知偏差：当前放置桥接"每段只标首台 DPU、每个 vpu 只用第一个 PU"（`genesim/docs/llama-2.md` 6.5），会让并行度看起来偏低；容量检查目前是 warn 不是 error，做 batch 研究时应改成硬失败，避免"跑完但放不下"的数字混进结论。

---

## 消息 5（用户）

4.要支持从llama2 7b到图编译->算子编译->genesim，评估参数还需要考虑，要符合实际，客观，评估模型要不要调整？真正对芯片开发有指导作用；

---

## 消息 6（助手）

## 先回答：要调整，而且分三层

当前这版评估模型适合回答“相对趋势”，还不足以独立支撑流片前的芯片规格决策。建议按三层推进，优先级从高到低：

| 层级 | 目标 | 不做的后果 |
| --- | --- | --- |
| **可信性（必须改）** | 让 Llama2-7B 全链路的数字在语义上自洽 | batch 结论方向可能是错的 |
| **客观性（应该加）** | 让数字可校准、可复现、有误差条 | 甲方无法判断数字能不能信 |
| **芯片指导力（必须补）** | 从“跑分”变成“架构取舍依据” | 只能报一个 tokens/s，指导不了 TO |

下面分别展开。

---

## 一、端到端 Llama2-7B：链路已经通了，缺的是“评估契约”

`scripts/run_full_pipeline.py` 已经覆盖你要的链路：

```text
HF Llama-2-7b
  → (A) model_parser          图骨架 IR
  → (B) export_pp_placement   图编译：切分、放置、本地分片
  → (C) FlagTree              算子编译：真实分块 + pim mlir
  → (D) placement sidecar     dpu_id / local shapes / pimir_path / kernel_tile_n
  → (E) genesim               照 pimir 生成 trace，出代价
```

但它是 **batch=1 的链路**，而且“评估口径”散落在各步。要让它对芯片开发有指导作用，建议把评估口径固化成四份契约，而不是继续靠命令行参数口头约定：

| 契约 | 内容 | 落地位置建议 |
| --- | --- | --- |
| **硬件画像** | 核/engine 数、时钟、MAC 宽度、WRAM/L1/L2/DDR 容量与带宽、互连、量化格式 | 新增 `conf/hw_<target>.yaml`，同时喂给图编译器和 genesim |
| **负载画像** | Llama2-7B、prompt/output 长度分布、prefill/decode 比例、到达率、batch 策略 | 新增 `conf/workload_<name>.yaml` + trace |
| **成本来源** | 每个 phase 用哪份 pimir、tile、dtype、本地分片形状，禁止 A/B 路混用 | sidecar 里已有 `pimir_path`/`pimir_sha256`，补 `phase` 和 `batch` |
| **结果指纹** | IR、sidecar、trace、配置、代码版本的哈希 | `results/summary.json` 增加 `evaluation_fingerprint` |

建议的最终入口形态：

```bash
python scripts/run_full_pipeline.py \
  --hw-profile ceva_m32k \
  --workload llama2_7b_batch_sweep \
  --batch 1,2,4,8,16 \
  --pim-only \
  --strict
```

这样每次报告的数字都能回答“哪份硬件、哪份负载、哪份算子产物、哪版代码”。

---

## 二、评估参数怎么定才“符合实际、客观”

评估参数不是一个 batch 数，而是一个设计空间向量。建议至少覆盖下面四类：

| 类别 | 参数 | 建议取值/做法 | 目的 |
| --- | --- | --- | --- |
| **硬件** | 核/engine 数、PU 数、时钟、MAC 宽度、WRAM、L1/L2、DDR 容量/带宽、互连带宽 | 按目标芯片的配置档位给区间，不是单点 | 看架构参数的弹性 |
| **负载** | Llama2-7B、batch、prompt 长度、输出长度、prefill/decode 比例、到达率 | 固定一组基准 + 一到两组 held-out | 避免只对合成负载调优 |
| **编译** | TP/PP、tile M/N/K、融合开关、放置策略、KV 布局 | 与硬件画像联动，不手调常量 | 区分“架构瓶颈”和“编译瓶颈” |
| **指标** | TTFT、TPOT、吞吐、利用率、MRAM traffic/token、容量占用、能耗、面积、$/token | 全量输出，不只报吞吐 | 支撑 Pareto 取舍 |

**客观性的三条硬要求**：

1. **基准配置固定**：Llama2-7B + batch=1 + 固定 trace + 固定 pimir，作为回归锚点；任何改动先过这个锚点。
2. **校准阶梯**：单算子 → 单层 → 全模型 → 多请求。每一级都和 CEVA/真硬件对一次，报告相对误差和绝对误差。
3. **留出验证**：在一组 `(seq, batch)` 上校准，在另一组上验证；不要用同一组数据既调参又报结果。

---

## 三、评估模型必须改的 6 件事（可信性）

| # | 问题 | 现状证据 | 怎么改 | 对芯片决策的影响 |
| --- | --- | --- | --- | --- |
| 1 | **GEMM 没有 batch 权重复用** | pimir `tile-m=1`；`src/vpu/vpu.py:458-473, 534-562` 按 `repetitions=Tq` 重放；prefill 被算成 Tq 次单 token GEMV | 每个 phase 用对应 pimir；`repetitions` 与 `tile_m` 对齐；成本模型区分“权重字节”和“激活字节” | 不改的话，batch 曲线方向可能是错的 |
| 2 | **prefill/decode 共用 decode pimir** | placement sidecar 里 GEMM 的 pimir 全是 `tile_m=1`；prefill 只是重复执行 | sidecar 增加 `phase` 维，prefill 用 prefill 编译的 pimir | 直接影响 TTFT 和 prefill 成本 |
| 3 | **默认配置不是目标芯片** | genesim 默认 8 设备 × 8 簇 × 16 PU、每 PU 512MiB；CEVA 目标是单 NPM core + 4 engine、L2 8MB、L1 1MB、DDR 约 4GB/core | 新建目标硬件画像；核/engine 映射到 PU；内存层级和带宽按目标填 | 不换配置，曲线是另一台机器的 |
| 4 | **默认启用了 GPU，结果是异构系统** | `conf/sim.yaml` 中 `num_gpus: 4`、`gpu_compute: true`；现有 summary 里 GPU 0–3 都有 useful flops | 芯片评估时 `gpu_compute: false` / `pim_compute: true`，或显式建模主机侧算子 | 否则报的是“CPU+GPU+PIM 系统”，不是 PIM 芯片 |
| 5 | **`weight_deployment` 配置不驱动放置** | `conf/sim.yaml:342-351` 写了 qkv/ffn→GPU、attention→PIM，但调度器只在结果里记录，不用它做映射 | 要么删掉这个配置避免误导，要么真正接入放置逻辑 | 配置看起来像“已分工”，实际是启发式，容易误读 |
| 6 | **放置桥接低估并行度** | `genesim/docs/llama-2.md` 6.5：混合策略每段只标首台 DPU；每个 vpu 只用第一个 PU | 先修“算子→多 PU”表达，或至少在报告里标注这是下界 | 并行度低估会让 batch/扩展性结论偏悲观 |

补充一个更正：我上一条说“容量检查是 warn”不准确。**常驻容量检查 `validate_tensor_pu_capacity` 是硬失败**（`src/pim/capacity.py:583-616`）；只有 sidecar 的单内核 MRAM 交叉核对是 warn（`_check_placed_mram_against_capacity`）。做 batch 研究时，硬失败是对的，保留。

---

## 四、为“真正指导芯片开发”必须补的 3 件事

### 1. 功耗/面积/能效代理模型（当前完全缺失）

genesim 现在没有系统级的 energy/power/area 输出；DRAMsim3 内部有功耗统计，但没有接进 `summary.json`。芯片开发不能只看 tokens/s，建议先做一阶解析代理，不必上物理级模型：

```text
E_token ≈ Σ_ops (MACs × e_mac + DRAM_bytes × e_dram + SRAM_bytes × e_sram)
        + P_leak × T
Area    ≈ N_mac × a_mac + SRAM_bytes × a_sram + PHY面积
指标    ≈ tokens/s/W、TOPS/W、area efficiency、$/1M tokens
```

系数从厂商数据/规格书拿，先做相对比较。最小实现是新增一个 `ppa_model.py`，只消费现有 `results/*.json` 和硬件画像，不改事件引擎。

### 2. 共享资源争用模型（当前偏乐观）

现在每个算子按自己的 pimir 周期算延迟，`peak_memory_bandwidth` 只用于“理想下界检查”（`src/vpu/vpu.py:314-329`），没有跨 PU 的 DDR/L2 带宽仲裁。高并发时多个 PU 各自按满带宽计时，**总带宽会超过物理峰值**，batch 越大越乐观。建议：

- 先做“后处理式”检查：按时间窗口汇总并发算子的 memory bytes，算聚合带宽利用率，超过峰值就标红；
- 再进一步：把 DDR/L2 建模成共享资源键，像链路资源一样参与调度（改动较大，放第二阶段）。

### 3. 设计空间扫描与敏感度分析（当前是单点运行）

芯片团队需要的是 Pareto 前沿和敏感度排序，不是一组固定配置的跑分。建议输出：

- `batch_curve.csv`：B × seq × 吞吐/TTFT/TPOT/利用率/traffic；
- `capacity_wall.csv`：不同量化格式下 `(B_max, seq_len)` 的表；
- `sensitivity.csv`：WRAM、DDR 带宽、PU 数、tile 策略各自对吞吐/时延的弹性；
- `pareto.json`：吞吐-时延-能耗/面积的非支配配置；
- `calibration_report.md`：仿真 vs 实测误差、假设、适用范围。

---

## 五、建议的实施顺序

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **第 1 阶段（1–2 周）** | 评估契约 + Llama2-7B B=1 基线 + batch 成本口径 + 一份 B>1 的 FlagTree 实测锚点 | B=1 逐位回归；B>1 单算子误差在可接受范围；无模板回退 |
| **第 2 阶段（2–4 周）** | 目标硬件画像 + PIM-only + 全链 batch 扫描 + 共享带宽后处理 | 7B 端到端 batch 曲线；容量墙；带宽饱和标志 |
| **第 3 阶段（持续）** | PPA 代理 + DoE/敏感度 + Pareto + 留出验证 | 能给芯片团队输出“最敏感的三个参数 + 推荐配置区间” |

**如果只做一件事**：先做“同一 linear 的 M=1 vs M=8 pimir 对比”，确认 FlagTree 是否真的生成权重复用的 batch kernel。这是整条评估链是否成立的前提。

---

## 六、边界：现在能指导什么、不能指导什么

**现在可以指导**：

- 相对架构取舍：多 PU vs 大 WRAM vs 高带宽；
- batch 甜点区和容量墙的位置；
- 编译策略（TP/PP、tile、融合、量化）对性能的影响；
- 目标负载下的瓶颈排序。

**现在还不能指导**：

- 绝对硅后性能、功耗、面积、热；
- 物理时序、真实内存系统效应（除非补争用模型）；
- 厂商编译器的实际代码质量。

所以对甲方的表述建议是：**“用相对趋势和敏感度指导 TO 前的架构选择，用校准锚点约束模型误差；绝对数字暂不作为规格承诺。”**

---

## 七、结论

- **评估模型要调**：先改可信性（batch/phase/硬件画像/PIM-only/争用），再加客观性（校准/误差/指纹），最后补芯片指导力（PPA/Pareto/敏感度）。
- **Llama2-7B 全链路已经存在**，但要把“能跑通”升级为“评估契约固定、可复现、可校准”。
- **多 batch 不是单独一层的事**：图编译器、算子编译器、genesim 三处必须用同一套 batch 语义和同一份成本来源，否则 batch 结论会互相矛盾。

如果你确认 TO 是流片、目标架构是 CEVA 单核 + 4 engine，我可以下一步把“硬件画像 + 评估契约 + batch 扫描”的具体字段和命令列成一份可执行的实施清单。

---

## 消息 7（用户）

回答你的问题：1.TO是流片；
2.目标架构是 CEVA 单核 + 4 engine，后面有可能连接多个，最终要再ceva验证，短期在genesim验证，所以genesim要符合ceva特征；
3.甲方关注没有多少tokens,吞吐率，ttft/tpot都关注吧。 根据上面的讨论，给我出一份完整的md方案吧，放到doc中

---

## 消息 8（助手）

方案已写入：

`/data/fjg/flagOS/flagos-pim-compiler/docs/design-multi-batch-pre-to-20261105.md`（+598 / -0，未提交）

## 文档内容概览

| 章节 | 内容 |
| --- | --- |
| 一、背景与目标 | TO=流片；CEVA 单核+4 engine，后续多核；短期 GeneSim、最终 CEVA；指标为 token 数、吞吐、TTFT/TPOT |
| 二、并发模型 | 区分请求级并发、张量 batch、空间并行、流水 micro-batch；短期主线 A+B |
| 三、端到端链路与评估契约 | Llama2-7B 现有 A→E 链路；四份画像（硬件/负载/成本来源/结果指纹）+ `--pim-only --strict` 入口 |
| 四、现状与差距 | 三仓证据（file:line）；关键结论：GEMM pimir `tile_m=1`、prefill 按 Tq 重放、无 batch 权重复用；默认配置含 4 GPU、`weight_deployment` 不驱动放置 |
| 五、评估模型设计 | batch 语义（prefill M=总 token、decode M=B）；BatchGroup；phase×batch pimir；共享 DDR/L2 带宽；CEVA 画像映射；指标与 PPA 一阶代理；校准阶梯 |
| 六、实施方案 | P0 契约与基线 → P1 batch 成本口径 → P2 7B 端到端 batch → P3 连续 batching + CEVA 对齐 → P4 PPA/设计空间 |
| 七、实验矩阵 | batch、序列、输出长度、阶段、量化、核数、编译策略；fractional factorial + held-out |
| 八、文件改动清单 | 按 flagos-pim-compiler / genesim / FlagTree 三仓列出具体文件 |
| 九、风险与应对 | 厂商并行仿真不准、GML 无 engine 字段、厂商编译器 batch 支持未知、绝对数字误差等 |
| 十、验收标准 | 正确性、成本、评估、校准、芯片指导五类 |
| 十一、里程碑 | 12 周，P0–P4 各阶段判定 |
| 十二、附录 | KV/容量/能耗公式、HardwareProfile/WorkloadProfile/CostProvenance/EvaluationFingerprint 数据结构、建议命令、待确认事项 |

## 方案里最关键的三个决定

1. **先钉成本口径，再扫 batch**：第一批实验不是 batch 扫描，而是同一 linear 的 M=1 vs M=8 pimir 对比，确认 FlagTree 是否真的生成“权重只读一次”的 batch kernel。当前 `tile_m=1` + `repetitions=Tq` 的模型下，batch 收益恒为零。
2. **CEVA 映射约定**：图编译器的 1 个 DPU = 1 个 NPM core（不是 1 个 engine）；4 engine = 4 个 TensorPU，共享 core 的 DDR/L2；GML 目前没有 engine 字段，短期用“4 engine 并行 + 校准”建模，engine 级放置待厂商接口确认。
3. **评估输出形态**：不是单个 tokens/s，而是“batch 曲线 + 容量墙 + 敏感度排序 + Pareto 前沿”；绝对数字只作参考，芯片决策用相对趋势和校准锚点。

## 文档中标注的待确认项

- CEVA 参数（DDR 约 3983MB、L2 8MB、L1 1MB、DTCM 64KB、4 engine）需用最新厂商文件复核；
- GML/GC 是否支持 engine 级放置或 batch>1 字段；
- 甲方 TTFT/TPOT 的具体 SLO；
- W4A8 是否唯一交付量化格式；
- 多核互连拓扑与带宽。

需要的话，下一步我可以按文档的 P0 把 `conf/hw_ceva_single.yaml`、`conf/workload_llama2_7b.yaml` 和 `run_full_pipeline.py` 的 `--hw-profile/--pim-only/--strict` 参数先落成代码。

---

## 消息 9（用户）

这次不要考虑调用：任何skill，刚才咱们讨论挺好的，就是把刚才你完整的记录信息，整理一下存入到一个md文档中。不要受到任何skills字数或格式的限制。写入：docs/design-multi-batch-pre-to-deepseek-20261105.md。重新组织。

---

## 消息 10（助手）

已写入：

`/data/fjg/flagOS/flagos-pim-compiler/docs/design-multi-batch-pre-to-deepseek-20261105.md`

**+1637 / -0**（新文件，未提交）。未调用任何 skill。

## 文档组织方式

全文重新整理为五部分、三十五节：

| 部分 | 内容 |
| --- | --- |
| **一、背景与结论** | 文档说明、一页结论、需求与目标（TO=流片、CEVA 单核+4 engine、token/吞吐/TTFT/TPOT）、并发模型四分法 |
| **二、现状调研** | 三仓职责、现有命令与产物、代码级现状（GeneSim/图编译器/FlagTree/CEVA）、配置陷阱、关键发现 |
| **三、评估模型设计** | 可信性/客观性/芯片指导力三层、评估契约、batch 语义、资源与容量、CEVA 画像映射、指标与 PPA、校准 |
| **四、实施方案** | P0–P4 详细任务、文件改动清单、实验矩阵、验收标准、风险边界、里程碑 |
| **五、附录** | 公式、数据结构、建议命令、源码阅读索引、完整问答记录、待确认事项、决策记录 |

## 相比上一版的补充

- 把讨论中的**事实与证据**完整保留：GeneSim 调度器、PIMVPU 重复次数、pimir 解析、容量检查、KV 布局、执行器、成本桥接、FlagTree tiling 等，都带 `文件:行号`。
- 增加了**配置陷阱**专节：默认 4 GPU、`weight_deployment` 不驱动放置、容量硬失败 vs 告警、放置桥接限制。
- 增加了**问答记录**：把你最初的三个理解、后面的第 4 点、TO/CEVA/指标确认，逐条保留。
- 增加了**决策记录（ADR）**：D1–D10，说明每个关键取舍的理由和备选方案。
- 增加了**源码阅读索引**：按主题列出 GeneSim/图编译器/FlagTree 的具体位置，方便后续开发直接定位。

文档的核心结论仍然是：

> 先把“权重只读一次”在成本模型里证明，再谈 batch 调度；先把 GeneSim 变成 CEVA 的模型，再谈流片决策。

---

## 消息 11（用户）

我不要任何加工，把当前所有会话内容保存在一个新的md文件中。
