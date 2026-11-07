# 需求文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：request-qwen3-moe-20261105
> 创建日期：2026-11-05
> 关联项目：flagos-pim-compiler（组件 pin 263fcda）

## 一、需求背景与目标

1. **业务背景**：链路已支持 Llama、Qwen2、Qwen3 dense 三类架构（`backend/architectures.py`）。
   Qwen3 的 MoE 线（`model_type="qwen3_moe"`，代表型号 Qwen3-30B-A3B：48 层全 MoE、
   128 专家 top-8、hidden 2048）是当前覆盖盲区：白名单启动期拒绝（实测确认，不会静默
   算错），绕过校验后 `torch.export` 也因 MoE top-k 分派的数据依赖形状直接失败。
2. **预期目标**：`qwen3_moe` 进入架构白名单，小随机 MoE 与真实 Qwen3-30B-A3B 都能
   「注册表加载 → 切分编译 → 执行」且贪心生成与 HF 逐 token 一致；蓝图模型新增
   数据依赖分派原语（性能路径）与专家并行放置维度。
3. **关联范围**：`backend/architectures.py`、`runtime/compile.py`、`graph/*`、
   `memory/*`、`runtime/kernels.py`、`runtime/executor.py`、`backend/hal_numpy.py`、
   `contracts/*`（P1）、`tests/*`、`paths.json`。

## 二、核心功能需求

### 2.1 功能清单

**P0-1 shape-static 的 MoE 前向导出与架构注册**
- 现状：`ARCHITECTURES` 只有 llama/qwen2/qwen3（`backend/architectures.py:48`）；
  `qwen3_moe` 被启动期拒绝（实测：`model_type='qwen3_moe' 不受支持`，已有防呆测试
  `tests/test_architectures.py::test_qwen3_moe_rejected`）。transformers 4.57.6 的
  `Qwen3MoeSparseMoeBlock.forward` 用 `expert_hit.nonzero()` + `hidden_states[None,
  top_x]` 做数据依赖分派，`torch.export` 报 `data-dependent expression`（实测）。
- 需求：注册 `qwen3_moe`（加载 fp16 + dense/MoE 共用策略入口）；提供静态形状的
  MoE 前向等价改写——`topk`（k 固定）/`softmax`/`one_hot` 形状本就静态，把
  「命中专家循环 + 按 token 子集 gather」改为「全专家 masked 计算 × 门控权重求和」，
  数值与 HF 严格一致（稠密模拟，计算量 ≈ 专家数/top-k 倍）。
- 验收判据：小随机 MoE（如 8 专家 top-2、GQA、head_dim 显式）`export_annotated_graph`
  + `propagate_specs` 通过；防呆测试改为「qwen3_moe 受支持」后仍守护其余未知架构。

**P0-2 图层规则支持 MoE 算子链**
- 现状：`graph/partition.py`、`graph/spec_prop.py` 无 `topk`/`one_hot`/masked 聚合
  的分区与规格规则；专家 linear 与 dense linear 同构。
- 需求：MoE 链路各节点获得设备标注与张量规格；专家的 gate/up/down 复用
  `LLAMA_WEIGHT_RULES` 列/行切规则（`graph/strategy.py:13`）；router（小 linear）
  与门控加权聚合获得明确放置（host 或 replicate）。
- 验收判据：MoE 图上每个张量节点都有 spec；无「被 linear 消费但策略未指定切法」
  类报错；专家权重分片无重叠、容量判据可用。

**P0-3 内存规划容纳专家权重**
- 现状：`memory/mem_planner.py` 按三区规划，容量判据以每 DPU MRAM 预算为准。
  实测 30B-A3B 几何：专家权重 fp16 54.0 GiB + 非专家约 3 GiB；KV 头=4 决定
  `tp_width ≤ 4`（`memory/kv_layout.py:158` 头对齐约束），8 DPU 首个可行策略为
  tp4×pp2 一类，权重约 7.1 GiB/DPU，8 GiB 预算下可行但紧。
- 需求：30B-A3B 在 8 DPU / fp16 / 张量并行切专家的形态下编译期容量判据通过；
  超预算时报出所需与可用（沿用现有报错形态）。
- 验收判据：真实 30B-A3B 编译期给出可行的三区规划且无重叠（对拍现有
  `test_mem_planner_llama2_7b` 的判据形态）。

**P0-4 MoE 执行路径（内核）**
- 现状：`runtime/kernels.py` 无门控加权聚合类内核；numpy 镜像路径按算子逐个执行。
- 需求：小模型与真实 30B 的稠密模拟路径都能执行（numpy 镜像优先；编译内核路径
  不支持的形状按既有机制退回镜像并计数）。
- 验收判据：小模型贪心生成与 HF `generate` 逐 token 一致；路由计数无异常退回
  （MoE 专属算子允许显式声明的退回路径）。

**P0-5 真实 Qwen3-30B-A3B 端到端**
- 现状：模型未下载；`paths.json` 无键。
- 需求：从 ModelScope 下载到 `/mnt/sda/butj/fjg/model/Qwen3-30B-A3B`（约 61 GB）；
  新增站点路径键（`qwen3_30b_model_dir` / 环境变量同型）；真实模型贪心生成短序列
  与 HF 逐 token 一致。
- 验收判据：新增 `tests/test_qwen3_30b_real.py` 通过（形态对齐
  `tests/test_qwen3_8b_real.py`）。

**P1-1 主机动态分派原语（性能路径）**
- 现状：执行计划为静态蓝图；`run_decode_loop` 已有步级数据依赖（读 logits 选 token），
  但层内专家分派不存在。
- 需求：蓝图新增数据依赖分派原语（主机读门控输出 → 只执行命中专家的子计划），
  执行计划契约与 hal 命令集相应扩展；语义与 P0 稠密模拟严格等价。
- 验收判据：同一小模型上 P1 与 P0 生成 token 完全一致；实测执行时间显著低于
  稠密模拟（记录加速比）。

**P1-2 专家并行放置维度**
- 现状：策略只有 TP/PP 两个维度（`graph/strategy.py` ShardStrategy）。
- 需求：新增专家并行维度（整专家到 DPU），与 TP/PP 可组合；专家按每层均分；
  16 DPU 组合（如 tp2×pp2×ep4 一类）在小模型上验证。
- 验收判据：16 DPU 专家并行编译通过、贪心生成与 HF 一致；专家到 DPU 的映射
  均衡（每 DPU 专家数差 ≤ 1）且可打印核查。

### 2.2 核心流程

注册表选架构 → 静态 MoE 前向导出（P0 稠密模拟形态）→ 分区/规格传播（专家权重
按规则切分）→ 三区内存规划（容量判据）→ 计划生成 → 执行（P0 全专家 masked / P1
动态分派）→ 贪心解码对拍 HF。

### 2.3 边界定义

**包含**：P0 全部、P1 全部（同一需求分阶段交付）；注册表接入后 vLLM 白名单自动
跟随属自然结果。**不包含**：genesim 的 MoE 稀疏计费（另立需求）、vLLM serve 实测、
Qwen3-235B-A22B、权重量化（int8/int4）、滑窗 MoE 变体、tied 词嵌入（30B-A3B 为
`tie=False`，不受影响；tied 缺口在 dense 文档已记录）。

## 三、技术约束与依赖

1. **代码影响范围**（逐文件）：`backend/architectures.py`（注册 + MoE 前向包装）、
   `runtime/compile.py`（导出入口适配）、`graph/partition.py`/`graph/spec_prop.py`
   （MoE 算子规则）、`graph/strategy.py`（P1 EP 维度）、`memory/mem_planner.py`/
   `memory/kv_layout.py`（容量与 KV 头约束）、`runtime/kernels.py`（聚合内核）、
   `runtime/executor.py` + `backend/hal_numpy.py` + `contracts/`（P1 原语）、
   `tests/test_architectures.py` 等、`paths.json`/`paths.local.json`。
2. **主要风险**：(a) 静态稠密展开使图规模膨胀约两个数量级（48 层 × 128 专家 ×
   每专家 3 linear + 激活 ≈ 数万节点，估算值），`spec_prop`/内存规划/编译耗时的
   可扩展性需在设计阶段压测，必要时对小模型先行基准；(b) 30B 权重 7.1 GiB/DPU
   贴近 8 GiB 预算，KV 与激活窗口参数须由容量判据反推；(c) P1 修改执行计划契约，
   波及 `contracts/` 全链路，需对拍既有 Llama/Qwen3 用例防回归。

## 五、验收标准

1. **回归判据**：`python -m pytest tests/ -q -k "llama2_7b"`（当前基线 48 passed）
   与 `tests/test_architectures.py` 全绿；新增 MoE 用例不改变既有架构行为。
2. **逐项验收**：P0-1/2/3/4 以小随机 MoE 全链路 + `test_qwen3_moe` 系列新用例通过
   为准；P0-5 以真实 30B 短序列贪心对拍 HF 一致为准；P1-1 以 token 一致 + 加速比
   记录为准；P1-2 以 16 DPU 专家并行用例通过 + 映射均衡断言为准。
3. **不回归判据**：Llama-2-7B 与 Qwen3-8B 既有端到端用例逐 token 结果不变。

## 六、调研补充信息

1. **transformers MoE 语义**（来源：本机 transformers 4.57.6
   `Qwen3MoeSparseMoeBlock.forward` 源码实测）：topk/softmax/one_hot 形状静态；
   数据依赖仅在「命中专家循环 + token 子集 gather」，静态化改写可行；Qwen3MoE
   与 Mixtral 的差别是 `norm_topk_prob`（门控重归一化）。
2. **真实 30B-A3B 配置**（来源：ModelScope `Qwen/Qwen3-30B-A3B/config.json` 实取）：
   见 2.1 P0-3；fp16 权重 57 GiB / 8 DPU×8 GiB 预算纸面可行。
3. 网络来源：模型经 ModelScope 下载（本机 huggingface 直连不稳定，此前 Qwen3-8B
   已验证 ~20 MB/s）。

## 七、本轮实测对既有结论的修正

1. 原结论（会话内曾表述）：「接入 Qwen3」——实测修正为「接入 Qwen3 dense 线」，
   MoE 线属另一类架构；已补防呆测试与文档澄清（`docs/qwen3-support-20261104.md`）。
2. 原结论（`docs/qwen2-support-20261102.md`）：「Qwen2 测试模型恰好走共享词嵌入
   路径」——实测 Qwen2Config 默认 `tie_word_embeddings=false`，tied 路径从未被测过
   （已在 Qwen3 文档记录）。

## 八、待确认事项

1. P1 动态分派原语的具体契约形态（plan 节点 vs hal 命令 vs 两者组合）留给技术设计。
2. 30B-A3B 的 KV 槽池 / 并行窗口参数：由容量判据反推，设计阶段定初值。
3. 小随机 MoE 的测试超参（专家数/头数/层数）在设计中与既有 `MODEL_KWARGS` 对齐微调。

## 九、附原始需求描述

「适配 Qwen3 MoE 模型（Qwen3MoeForCausalLM，model_type=qwen3_moe，代表型号
Qwen3-30B-A3B）到当前 PIM 推理编译链路。当前链路已支持 Qwen3 dense 线，MoE 线在
启动期被白名单拒绝；实测绕过校验后 torch.export 也过不去（top-k 分派是数据依赖
形状）。已知缺口：shape-static 的 MoE 前向导出、topk/gather/index_select 的图规则、
专家权重的放置策略（专家并行或复制）、MoE 分发/归约内核、30B 级专家权重（fp16 约
60GB）跨 DPU 的内存规划（8GB/DPU 预算）。模型可从 ModelScope 下载到
/mnt/sda/butj/fjg/model。」

澄清决策记录（2026-11-05）：运行时语义「两者都要，分阶段」；验收对象「小随机 MoE +
真实 30B-A3B」；范围「compile + executor 主链路」；部署形态「支持更大 DPU 数/专家并行
维度」，深度为「小模型验证专家并行、验证到 16 DPU」；30B 本体用张量并行切专家。
