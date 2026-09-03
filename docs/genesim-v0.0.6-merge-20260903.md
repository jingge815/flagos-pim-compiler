# 合并 GeneSim v0.0.6 引入的问题与修复

日期：2026 年 9 月 3 日

## 一、背景

`fjg-dev` 分支落后 `origin/develop` 十三个提交，合并到 `v0.0.6` 后流水切分（PP）的全流程断了。本文记录合并引入的每个问题、根因、修法和验证。

上一轮把 PP 打通的记录在 `docs/pp-flagtree-genesim-20260902.md`，本文只写这次升级新出现的问题。

合并方式是 `git merge tag 'v0.0.6'`，冲突四处，提交 `9102caa`。冲突本身不难，真正的工作量在合并之后——上游换了图骨架，连带五处对不上。

## 二、上游改了什么

关键提交是 `ad22fcd`（作者 root，2026-08-25，`fix generation of model ir: add residual add, gqa, gate_proj-silu+up_proj+down_proj`）。它重写了 `src/ir/model_ir.py` 的 `build_from_hf_config`：

| 项目 | 旧图骨架 | v0.0.6 |
| --- | --- | --- |
| 算子总数（llama2-7b） | 3232 | 3491 |
| 每层 GEMM 数 | 4 | 7 |
| q/k/v | 合并成一个 GEMM（`4096 -> 12288`） | 三个独立 GEMM |
| MLP | 只有 `gate_proj` 和 `down_proj` | 补上了并列的 `up_proj` |
| 残差与归一化 | 无 | 新增 `VECTOR_ADD`、`RMSNORM`、`SILU`、`VECTOR_MUL` |
| 图边界 | 无 | 新增 `MODEL_INPUT`、`MODEL_OUTPUT` |

另外两处与本文相关的上游变化：

- `PIMVPU` 现在只代表**一个 TensorPU**（`_pim_pus_per_device()` 直接返回 1，注释写明 "scheduler code must never expand it again"）。此前它代表整个 PIM 设备、由调度器展开成 128 个条目。
- `pim_compiler.COMPILE_FUNCS` 新增了 `GEMM`、`GELU`、`SILU`、`RMSNORM`，`conf/sim.yaml` 的 `pim.supported_ops` 也跟着加了。

第二条让上一轮的两处修复变成了多余：跨段通信的 vpu 去重（编号塌缩的根因随资源模型消失）和 trace 准备阶段跳过被钉算子（GEMM 现在能正常编译出 trace）。合并时一并删掉了，细节见提交信息。

## 三、五处问题

按撞见的顺序排列，每一处都是修好前一处之后才暴露出来的。

### 1. 成本提取撞上未知算子类型

```text
File "genesim_bridge/cost_extractor.py", line 67, in _measure
    recipe = recipes[op_type](point)
KeyError: 'MODEL_INPUT'
```

新 IR 引入六种成本提取侧不认识的类型，共 195 个算子。报错只提了第一个撞上的：

| 类型 | 个数 | 是否自带模板系数 |
| --- | ---: | --- |
| `RMSNORM` | 65 | 有 |
| `VECTOR_ADD` | 64 | 有 |
| `SILU` | 32 | 有 |
| `VECTOR_MUL` | 32 | 有 |
| `MODEL_INPUT` | 1 | 无（零成本占位） |
| `MODEL_OUTPUT` | 1 | 无（零成本占位） |

**修法**：列入 `genesim_bridge/op_classify.py` 的 `UNCOVERED_OP_TYPES`，保留 `model_parser` 已写进 IR 的模板系数。

没有为它们硬编 FlagGems 配方，理由有两条：那四种逐元素算子在 `pim_compiler` 里本来就有各自的 trace 编译器，成本由那条路径负责；硬编一套配方等于引入一组未经校准的数值，比保留上游模板更差。两个边界节点 `flops` 和 `data_bytes` 都是 0，本来没什么可测。

**验证**：`Bridged 3296 operators from pimir; 195 kept template costs`。核对产物：算子总数、`subgraphs`、`dependencies`、所有输入输出形状与输入基线逐项一致；195 个 template 类算子的系数一个都没被改动；3296 个 bridged 算子的系数全部更新。

新增 `test_every_op_type_is_either_measurable_or_explicitly_uncovered`，对着真实 IR 检查每个类型都有归属（能测量，或明确列入模板）。下次上游再加新类型会立刻失败，而不是等跑精化脚本才发现。变异验证：退回旧的 `{"GELU"}` 集合后，它精确列出那六种类型并失败。

### 2. 放置导出的位置匹配失效

```text
ValueError: layer 0 期望 4 个 GEMM，实际 7 个：[2, 3, 4, 101, 104, 106, 108]
```

导出侧原先按固定顺序 zip 四个「代表权重」（`q_proj` 代表合并后的 qkv、`gate_proj` 代表整个 fc1），这依赖旧图骨架每层恰好四个 GEMM。

**修法**：改为按权重名匹配。IR 的 `dependencies` 给出每个 GEMM 消费的权重 `tensor_id`（形如 `layer.0.q_proj.weight`），据此确定身份，不再依赖出现顺序。

```text
op2    q_proj.weight       4096 -> 4096
op3    k_proj.weight       4096 -> 4096
op4    v_proj.weight       4096 -> 4096
op101  o_proj.weight       4096 -> 4096
op104  gate_proj.weight    4096 -> 11008
op106  up_proj.weight      4096 -> 11008
op108  down_proj.weight   11008 -> 4096
```

这一改顺带消掉了两个长期近似：qkv 不用再手工累加三份（GQA 下也不用分别取 k/v 的实际宽度），fc1 不再漏算并列的 `up_proj`。放置数从 128 涨到 224（32 层 × 7 个投影）。

`_GEMM_WEIGHT_SUFFIXES` 是那张名字表。遇到表里没有的权重名、或 GEMM 在 `dependencies` 里找不到权重，都直接抛错并提示同步，不静默跳过。

**验证**：真实 7B 导出 224 个 GEMM，七种投影各 32 个，层到 DPU 严格等于 `layer // 4`。

测试的 fixture 故意把 GEMM 的 op_id 顺序**打散**（`op0` 是 `down_proj`、`op10` 是 `q_proj`），与名字表的声明顺序不一致。这一点是必要的：第一版 fixture 顺序恰好与表一致，退回位置匹配也能通过——变异验证当时只失败了 1 个用例，说明测试没真正锚定「按名字匹配」。打散后退回位置匹配会失败 3 个，其中 2 个是宽度取错。

### 3. 分组逻辑与新图骨架不匹配（上游缺陷）

```text
ValueError: Initial placement units are not in topological order: operator 2->5.
```

**这一条与本仓改动无关，是上游自身的缺陷。** 断定前确认了三件事：

1. 不配 sidecar（关掉放置接入）同样失败
2. 用 develop 原有的 `conf/sim_llama2_7b_pimir.yaml` 同样失败
3. 在 `v0.0.6` 的纯净 git worktree 里同样失败——那份代码里连 `compiler_placement` 字样都不存在

**根因**在 `_build_adaptive_initial_placement`：它把每层非注意力算子拆成「第一个」和「其余」，后者整块排在所有注意力 unit **之后**。

```python
if non_attention_op_ids:
    placement_units.append([non_attention_op_ids[0]])   # 第一个
for attention_flow in attention_flows:
    placement_units.append(list(attention_flow))
if len(non_attention_op_ids) > 1:
    placement_units.append(non_attention_op_ids[1:])    # 其余，排在注意力之后
```

旧 IR 上成立：qkv 合并后每层恰好只有一个算子在注意力之前。新 IR 每层前面有五个：

```text
layer0 注意力之前: [0, 1, 2, 3, 4] = MODEL_INPUT, RMSNORM, q_proj, k_proj, v_proj
layer0 首个注意力: op5 (GEMV_SCORE)
```

于是 q/k/v（`op2/3/4`）落进了尾部 unit，而消费它们的注意力 unit 在前面，`op2 -> op5` 成了「后面的 unit 指向前面的 unit」。

IR 本身是拓扑有序的——层内和跨 subgraph 的逆序依赖都是 0，问题在分组，不在 IR。

**修法**：按「相对首个注意力算子的位置」切分，两侧各自成 unit。

```python
first_attention_position = min(layer_op_ids.index(op_id) for op_id in attention_members)
pre_attention_op_ids  = [op for op in non_attention_op_ids if pos[op] < first_attention_position]
post_attention_op_ids = [op for op in non_attention_op_ids if pos[op] > first_attention_position]
```

这是**本地补丁**。上游将来修这个缺陷时可能采用别的写法，届时会与本补丁冲突。建议同时把问题报给 develop 的维护者。

回归测试 `test_layer_with_several_ops_before_attention_stays_topological` 构造了一个层首有多个算子的最小 IR，且不配 sidecar——它守的是普通调度路径，不只是放置接入。变异验证退回旧分组后，它报出一模一样的 `operator 2->5`。这个缺陷此前完全没有测试覆盖，这也是它能溜进上游的原因。

### 4. 一台 DPU 应映射成 ClusterPU，不是 TensorPU

```text
ValueError: PIM resident-memory capacity exceeded
(limit=536870912 B; vpu=2082: +1082130432 B, ...)
```

分组修好后立刻撞上这个。根因是上游把 `PIMVPU` 改成只代表一个 TensorPU，而放置桥接还按「一台 DPU = 一个 PIMVPU」映射：

| 层级 | 容量 | 可容纳层数 |
| --- | ---: | ---: |
| TensorPU | 512 MiB | 1.3 |
| ClusterPU（16 个 TensorPU） | 8192 MiB | 21.2 |
| PIM 设备（8 个 ClusterPU） | 65536 MiB | 170 |

llama2-7b 每层七个投影约 386 MiB。`tp1_pp8` 每段四层就是 1544 MiB，钉到单个 512 MiB 的 TensorPU 上必然超限——报错里 `vpu2082` 需要的 1032 MiB 与这个推算吻合。

各策略的需求：

| 策略 | 每段层数 | 每段权重 | TensorPU | ClusterPU |
| --- | ---: | ---: | --- | --- |
| `tp1_pp8` | 4 | 1544 MiB | 超限 | 可容纳 |
| `tp2_pp4` | 8 | 3088 MiB | 超限 | 可容纳 |
| `tp8_pp1` | 32 | 12352 MiB | 超限 | 超限 |

**修法**：钉死逻辑改成按 ClusterPU 粒度分配——一台 DPU 对应一个 ClusterPU，段内算子在它的 16 个 TensorPU 之间轮转。

这样既分摊常驻内存，也让同段算子留在一个 cluster 内（走 `PIM_Intra_Cluster` 快链路），跨段才跨 cluster。

**验证**：8 个 dpu 一一对应 8 个不同 ClusterPU，每个用到全部 16 个 TensorPU。

```text
dpu0 -> device0 cluster0     每个 dpu 用到的 TensorPU 数: [16, 16, 16, 16, 16, 16, 16, 16]
dpu1 -> device0 cluster1
...
dpu7 -> device0 cluster7
```

注意 `tp8_pp1`（纯张量并行、单段 32 层）连 ClusterPU 都装不下，本轮没有验证它——它本来也是退化情形，见下文遗留问题。

### 5. 配置的链路表过时

```text
RuntimeError: No communication-reachable placement is available for operator 3480 (GEMM).
[ERROR] No communication path from GPU to HOST
```

上游给 `conf/sim.yaml` 新增了三条链路，带宽单位也改了（`64.0` → `512.0`）：

| 链路 | 用途 |
| --- | --- |
| `PCIe_Gen5_x16_GPU_to_HOST` | GPU 回写主机，缺了它 GPU 上的算子无处输出 |
| `PIM_Intra_Cluster` | ClusterPU 内 TensorPU 之间 |
| `PIM_Inter_Cluster` | ClusterPU 之间的直连 P2P |

而仓里八份 per-workload 配置都还是旧的六条——包括 develop 原有的五份（`sim_gpt2_*.yaml`、`sim_llama2_7b_flagtree.yaml`、`sim_llama2_7b_pimir.yaml`），不只是本仓新增的三份。

**修法**：用脚本把 `sim.yaml` 的九条链路统一同步到八份配置，避免手工改八遍出错。

内置默认值（`config_loader.py` 的 `DEFAULTS`）只有三条、也缺 GPU→HOST，所以每份配置必须自带完整的 `comm_config`，不能依赖继承。

## 四、全流程结果

`tp1_pp8`（纯流水）跑通：

```text
Built adaptive initial placement: 1408 atomic units, 2049 available resources, 357 used resources
Built 32 fixed layer pipeline Stages for 3491 operators
Prepared 908 PIM traces in pim_traces (833 compiled, 75 reused)
PIPELINE simulation finished. Total time: 2616.88 s
Completed requests: 10
Simulation completed successfully
```

顺带一个观察：放置搜索从合并前的约十八分钟降到七秒。资源列表从 262144 条缩到 2049 条——正是上游那个「一个 PIMVPU 代表一个 TensorPU」的重构效果。

`tp2_pp4`（混合切分）的本地成本口径在 IR 层面仍然正确，七个投影各自精确减半：

```text
q_proj / k_proj / v_proj / o_proj   4.295e+09 -> 2.148e+09   0.500
gate_proj / up_proj / down_proj     1.154e+10 -> 5.772e+09   0.500
224 个 GEMM 合计  1.6580e+12 -> 8.2903e+11   比值 0.5000
变化算子数 = 224，与被放置集合完全相等
```

但它**不再影响仿真时间**，见下一节第一条。

## 五、本轮新发现的遗留问题

### 本地分片成本口径在合并后不再影响仿真（重要更正）

上一轮的记录是：`tp2_pp4` 用本地口径跑出 7192 s、全局口径 15060 s，比值 0.4775，与 GEMM flops 减半吻合。合并 v0.0.6 后重做同策略 A/B 对照，两组结果**逐位相同**：

| 成本口径 | `model.ir_path` | Total time | comm_time |
| --- | --- | ---: | ---: |
| 全局 | `llama2_7b_pimir.ir` | 2630.510741 s | 1.2954 s |
| 本地分片 | `llama2_7b_tp2_pp4_local.ir` | 2630.510741 s | 1.2954 s |

先排除了操作失误：两份配置确实指向不同 IR；两份 IR 确实有 224 个算子的系数减半（`op2` 全局 `{Tq: 2.5367e7, constant: 1.0485e9}`，本地 `{Tq: 1.2684e7, constant: 5.2425e8}`）；两次运行独立（`summary.json` 的 MD5 不同、时间戳相隔九分钟）；292 个活跃 PU 的 `busy_time_s` 逐个相同。

**根因**是合并带来的语义变化。上游把 `GEMM` 加进了 `PIMVPU._RUNTIME_PARAMETERIZED_OPS`，于是 GEMM 在 PIM 上的延迟由 `_execute_runtime` 按 **shape** 算 tile 数：

```python
loop_count = max(1, math.ceil(output_width / tile_size))
```

它完全不读 `flops_coeffs`。而本地分片口径只改成本系数、没改 shape——两份 IR 的 `input_shapes` / `output_shapes` 完全一致——所以对执行时间零影响。

`flops` 现在只剩一处消费者：`estimate_op_service`（放置搜索的比价）。而被钉住的算子跳过比价直接钉死，所以连放置都不受影响，这解释了为什么 PU 负载也一模一样。

合并前 GEMM 走的是 roofline 公式（直接用 `flops` 和 `data_bytes`），那时这条路径是通的，上一轮的 0.4775 是真实测量。上游改走真实 trace 后，通路断了。

要让它重新生效，得把本地分片宽度写进 IR 的 `input_shapes` / `output_shapes`，让 tile 数跟着变。这是另一件事：会改变 IR 的形状语义，也要确认 `_execute_runtime` 的 tile 计算和 PIM trace 编译对新形状都成立。本轮没有做。

现阶段的准确表述是：**本地分片口径影响 sidecar 里记录的成本数值和放置搜索的比价输入，不影响仿真时间。**

### 注意力算子不受放置结果约束，跨段流量被记成跨设备

sidecar 只覆盖 GEMM（224 个），而 1024 个 `GEMV_SCORE`/`SOFTMAX`/`GEMV_CONTEXT` 从不被钉住，仍由调度器贪心分配。实测：

```text
算子按 pim_device_id 分布: {non-PIM: 2583, device0: 224, device1: 384, device2: 300}
  被钉住的 224 个 GEMM: 全在 device0（钉死生效）
  未被钉住的注意力算子: device1(384) + device2(300)
```

后果是跨段搬运被记成「跨物理设备」，走 `PIM_EXTERNAL_PAIR` 而不是本该的 `PIM_INTER`：

```text
PIM_EXTERNAL_PAIR:0:0:0:1   793,706,496 B   0.1116 s
PIM_EXTERNAL_PAIR:0:0:0:2   620,083,200 B   0.0872 s
PIM_INTER                             0 B   0.0000 s
PIM_INTRA                             0 B   0.0000 s
```

通信确实被计入了（总计 1.2954 s），但归类不对——流水段之间本该是同一设备内的 cluster 间搬运。

要修得让图编译器也把每层的注意力算子标上所属 DPU。当前 `placement_export.py` 只遍历 GEMM 权重（`_GEMM_WEIGHT_SUFFIXES`），做不到这一点；注意力算子没有权重，得靠层号来归属。

### 分组补丁是本地的

第三节那处修复改的是上游 `_build_adaptive_initial_placement`。上游修同一缺陷时若采用别的写法，会与本补丁冲突。

### 纯张量并行策略未验证

`tp8_pp1` 每段需要 12352 MiB，超出单个 ClusterPU 的 8192 MiB。它本来也是退化情形（`min(shard_map)` 使所有算子标到 dpu0，见上一轮文档），本轮没有验证。要跑它得让一台 DPU 映射到整个 PIM 设备（8 个 ClusterPU、64 GiB）。

### 上一轮的遗留问题仍然存在

段内张量切分失真、被钉算子集中在少数 PU、换策略要手工同步配置文件名等，见 `docs/pp-flagtree-genesim-20260902.md` 第七节。其中「被钉算子挤在每个 vpu 的第一个 PU 上」这条，因为本轮改成了 ClusterPU 内轮转，已经缓解——每台 DPU 现在用满 16 个 TensorPU。

## 六、怎么复现

合并后 IR 换了，所有下游产物都要重新生成，顺序不能颠倒。

```bash
# 一、GeneSim 侧：生成新图骨架（若尚未生成）
cd /media/disk/fengjingge/src/genesim
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
export PATH="$HOME/.local/bin:$PATH"
python scripts/model_parser.py \
    --model_name /media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
    --output models/llama2_7b.ir

# 二、图编译器侧：导出放置结果（约十分钟，要加载 7B 权重）
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
python scripts/export_pp_placement.py --num-stages 8      # tp1_pp8
python scripts/export_pp_placement.py --num-stages 4      # tp2_pp4

# 三、GeneSim 侧：精化成本（十几秒）
cd /media/disk/fengjingge/src/genesim
python scripts/refine_ir_with_flagtree.py \
    --ir models/llama2_7b.ir \
    --out-ir models/llama2_7b_pimir.ir \
    --sidecar models/llama2_7b_pimir_extensions.json \
    --seq-len 128 --ir-level pimir \
    --placement models/llama2_7b_tp1_pp8_placement.json

# 四、跑仿真
./run.sh --config conf/sim_llama2_7b_pp.yaml
```

`results/` 是固定路径且会被其它进程覆盖，跑完立刻归档再分析：

```bash
mkdir -p /tmp/myrun && cp results/summary.json results/communication_metrics.json \
    results/pu_metrics.json /tmp/myrun/
```

放置 sidecar 与 IR 的一致性有校验：sidecar 记录了导出时的算子总数和每个算子的 `op_type`，编号错位会直接抛错并提示重新导出。本轮实测到过一次——旧 sidecar（3232 口径）对上新 IR（3491）时被准确拦下。

## 七、改动清单

本仓（flagos-pim-compiler）：

| 文件 | 变化 |
| --- | --- |
| `genesim_bridge/op_classify.py` | `UNCOVERED_OP_TYPES` 增加六种新算子类型（+21 / -2） |
| `genesim_bridge/placement_export.py` | 从按顺序 zip 代表权重改为按权重 `tensor_id` 匹配；sidecar 增加 `weight` 字段（+109 / -50） |
| `tests/test_genesim_bridge.py` | 新增算子类型归属检查（+43） |
| `tests/test_placement_export.py` | fixture 改成七个投影且 op_id 顺序打散；测试改为逐投影核对宽度（+191 / -75） |
| `docs/genesim-v0.0.6-merge-20260903.md` | 新增，本文 |

GeneSim 仓：

| 文件 | 变化 |
| --- | --- |
| `src/scheduler/gene_sim_scheduler.py` | 分组按注意力位置切分；钉死改为 ClusterPU 粒度（+71 / -13） |
| `tests/sim/test_compiler_placement.py` | 新增分组拓扑序回归测试；样本从 GEMM 换成 `CONV2D`（+82 / -6） |
| `conf/sim_*.yaml`（八份） | 链路表统一同步为 `sim.yaml` 的九条 |

## 八、每处修复都做过变异验证

做法是把修复临时改回原样，确认对应测试真的失败——不这样做无法区分「测试通过」和「测试没测到」。

| 修复 | 变异方式 | 变异后的失败表现 |
| --- | --- | --- |
| 算子类型归属 | 退回 `{"GELU"}` | 精确列出那六种类型并失败 |
| 按权重名匹配 | 退回按 subgraph 顺序取 | 3 个用例失败，其中 2 个是宽度取错 |
| 分组按位置切分 | 退回「第一个 / 其余」 | 报出线上原样的 `operator 2->5` |

第二条的变异验证暴露过一次测试不足：第一版 fixture 的 op_id 顺序恰好与名字表一致，退回位置匹配只失败 1 个用例（而且是个不相关的用例），说明测试并没有真正锚定「按名字匹配」。把 fixture 顺序打散后才测到位。

上一轮也有过一次类似的教训：`_prepare_pim_traces` 的 skip，变异后 13 个测试全过——追查发现 GEMM 现在能正常编译 trace，那段 skip 已成死代码，于是删掉并重写了测试。
