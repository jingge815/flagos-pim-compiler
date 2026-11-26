# 需求文档：多 batch 全链路支持与性能评估

> 文档编号：request-multi-batch-20261121  
> 创建日期：2026-11-21  
> 关联项目：flagos-pim-compiler（图编译器，主工作区）、genesim（模拟器）、FlagTree（算子编译器）；/data/fjg/ceva 厂商工具链只读参考

## 一、需求背景与目标

　　从大模型到图编译、算子编译再到 genesim 仿真的全流程已经走通（`scripts/run_full_pipeline.py`，标准形态 tp16_pp4/64 核，配套 `conf/ceva_64core_4e.yaml`），但整条链路只考虑单 batch：图导出、KV cache 物理布局、内存规划、执行编排、算子成本口径、仿真调度六层全部把 batch 写死为 1，无法发挥 PIM 架构在多 batch 下的性能潜力。PIM 上多 batch 的最大收益来源是权重复用——一次权重装载摊薄给 M 维多行激活，这正是存算架构相对传统架构的核心优势，而当前 genesim 的跨请求执行是时间线性叠加（`pim_backend.py:60-61`），该收益在模型里结构性不存在。

　　本需求要把 batch 维贯穿三仓：其一，打通多 batch 全链路（图编译→算子编译→genesim，含内存分配、KV cache 管理与编排器）；其二，建立多 batch 评估模型，同时计价收益（并行、流水、权重复用）与代价（抢占、资源竞争、同步等待）；其三，落地能带来收益的优化（权重复用、流水优化、多 batch 并行）；其四，让 genesim 能按不同 batch 扫参，给出性能瓶颈与最优配置。核心思路是先做静态 closed batch（等长请求、整组进出）的执行闭环与成本口径，再在其上叠加评估与优化——这也是 2026-11-05 讨论记录（`docs/design-multi-batch-pre-to-deepseek-20261105.md` 决策 D1/D2）验证过的正确顺序，本期把当时后置的执行器 B>1 闭环正式纳入。已拍板的边界：硬件主口径为 ceva_64core_4e（单核作为容量墙极端档补充）；本期不动 cascade 多 CEVA 级联；抢占只建模不实现；多 batch 无外部实测锚点，验收采用内部一致性判据。

## 二、核心功能需求

### 2.1 功能清单

　　功能点按「从配置到模型、从单点到整体、从实施到验证」组织：R1–R4 打通图编译器侧的 batch 维（入口、KV、内存、编排器），R5–R6 打通算子编译成本链与 genesim 仿真执行，R7 在闭环之上建评估模型，R8 落地优化，R9 交付扫参与最优配置输出。R1–R7 为 P0（全链路与评估是甲方核心诉求），R8–R9 为 P1。

**R1（P0）多 batch 编译入口与图导出**

　　batch 维必须从编译入口进入整条链路，否则下游无从谈起。现状是导出形状的源头就钉死了 1：`runtime/compile.py:78-83` 用 `[1, seq]` 的 input_ids 做 torch.export 且不带 `dynamic_shapes`，mask 写死 `[1,1,S,S]`（`runtime/compile.py:66-68`），decode 图按 `[1,1]` 导出（`runtime/compile.py:237-239`）；入口脚本同样没有 batch 参数（`scripts/run_full_pipeline.py:794-831`、`scripts/export_gml.py:365-403`）。需要改成 `compile_llama2` 接受 `batch_size`，prefill/decode 分别导出 `[B,S]` 与 `[B,1]` 的静态图，mask 与 position_ids 按 batch 语义（B 条独立因果序列，掩码可共享广播）重新构造，`run_full_pipeline.py` 增加 `--batch` 贯通到编译与仿真两端；batch 维默认 replicate 不参与 TP/PP 切分，空间并行映射留给 R8 单独开关。本文以「B=1 时全部产物与现状逐位一致，B>1 时 torch.export 无 graph break、导出图含正确的 B 维形状」为验收判据。

**R2（P0）KV cache 多槽位布局与容量公式**

　　多 batch 下每个请求槽位需要独立的 KV 区域与有效长度，否则 decode 无法正确读写。现状是物理布局完全单请求：offset 键只有 `(layer, head, which)` 三元组（`memory/kv_layout.py:40`），容量公式无 batch 项（`:59-62`），读写接口只接受单序列位置（`:181-192` 的 `kv_access`、`:208` 的 `PIMStaticKVCache.update` 只收标量 pos）。需要给 `KVRegionSpec` 增加 batch 槽：每槽独立有效长度（per-slot `valid_len`）、KV 字节按槽位数线性放大（`kv_bytes = 2×layers×max_seq×kv_heads×head_dim×dtype_bytes×B`）、寻址接口带槽位轴。本文以「多槽布局 offset 互不重叠、容量数值与公式手推一致、`tests/test_kv_layout_llama2_7b.py` 断言更新后通过」为验收判据；容量装不下保持硬失败直接抛（11-05 决策 D9）。

**R3（P0）内存规划 batch 缩放与容量墙输出**

　　激活区与三区 offset 规划必须感知 batch，且要能回答「多 batch 首先被什么卡住」。现状是激活区大小取自导出图节点的 local_shape，batch=1 已烘进形状（`memory/mem_planner.py:317-321`），容量判据同样无 batch 项；已有的多请求容量台账（`memory/mem_planner.py:458-472` 逐请求 KV 峰值、`:527-591` `tensor_pu_usage(concurrency=)`）只影响 TensorPU 容量校验，是本功能点的现成起点。R1 落地后激活形状自动变为 `[B,·]`、KV 经 R2 乘 B，需要补的是容量判据输出：规划结果打印「(batch, seq) → 三区占用/上限」容量表（权重/KV/激活占比），超限硬失败并指明卡在哪一区。本文以「三区无重叠、B 档容量数值与手推一致、容量表可读输出」为验收判据。

**R4（P0）编排器多 batch 执行闭环与数值对齐**

　　编排器要能把一个 closed batch 组从头跑到尾，并与 PyTorch 对拍，这是全链路数值正确性的锚点。现状是执行器严格单请求串行：`runtime/executor.py:92-94` 的 `_as_batch_tensor` 把 batch 维写死为 1，`run_decode_loop`（`runtime/executor.py:41-89`）一次 prefill 后逐 token 串行 decode 且只采样单行 logits，`DecodeState` 只有标量 `valid_len`（`runtime/executor.py:15-22`），执行计划按一张图一次执行生成（`runtime/exec_plan_gen.py:374-380`），HAL 每次只绑定单个标量位置（`backend/hal_numpy.py:281-293`）。需要把 `DecodeState` 改为按槽位维护 `valid_len`，decode 循环每步喂 `[B,1]` 输入、按槽位推进有效长度、逐槽采样；执行计划与通信计划随 R1 的图形状自动跟随，不需要独立的 batch 字段（`comm/plan.py:316` 的 `_batch` 是 DMA 段合批，与模型 batch 无关，不得混淆）。本文以「B=1/2/4 在 NumpyBackend 上与单卡 PyTorch batched forward 对齐（沿用现有 fixture 对拍模式，见第七节修正 5）」为验收判据。

**R5（P0）算子编译 batch 代表点与成本口径**

　　成本链必须按真实编译产物计价 batch，禁止用「Tq×B 线性外推」冒充 batch 成本。现状是 genesim 实际执行的产物由 placement sidecar 的 `pimir_path` 指定（`scheduler._compiler_pimir_paths`），其代表编译钉死 M=1（`genesim_bridge/placement_export.py:254-259`），实测该产物 `pim.tile-m = 1`——摊薄机制（`ceil(M/tile_m)` 折算）虽已实现但真实链路从未触发；成本代表点只有 prefill/decode 两个 `(Tq,Tp)` 点（`genesim_bridge/op_classify.py:98-111`）。与之相对，算子编译侧对 M>1 已经就绪：前导维折进行数 M（`contracts/op_contract.py:125-130`），kernel 按 `BLOCK_M=16` 分块加掩码、tile 搜索对 M 对称，FlagTree 预计零改动。需要按相位出真实产物：decode 档逐 B 编 `M=B`；prefill 档逐 B 编 `M=B×S`（与图编译/编排器一次执行 `[B,S]` 的真实语义一致，2026-11-25 用户裁决修正——早前「合并无收益」依据的等式 `ceil(B×S/tile_m)=B×ceil(S/tile_m)` 已被真实算子编译证伪，S 非 tile_m 整数时合批装载严格更少；注意力的正确做法是按 B 个块对角序列计价而非放弃合并）；sidecar 按 `(phase, batch)` 记录产物路径与 tile 形状（`prefill_b{B}`/`decode_b{B}`），genesim 按此选取。本文以「B=1 成本逐位不变；sidecar 可按 `(op_id, phase, batch)` 查到产物与 tile_m；权重复用前置实验通过」为验收判据——前置实验：同一 linear 分别编译 M=1 与 M=8，确认 M=8 产物 `tile_m>1` 且权重搬运接近 1 份；若仍接近 8 份，说明 kernel 未做 batch 复用，必须先停下修 kernel 循环再谈调度。

**R6（P0）genesim 张量 batch 执行与计费**

　　仿真器要能把 B 个等长请求组成一个 batch 组，让权重复用收益真实出现在数字里。现状有三道闸：`execution_microbatch_size != 1` 在配置层直接抛错（`src/config_loader.py:1391-1395`）、每个请求独立实例化整张算子 DAG 且 task_key 含 request_index 结构上无法合并（`src/scheduler/gene_sim_scheduler.py:9300-9301`）、GEMM 执行按 `单轨迹周期 × repetitions` 线性缩放跨请求零共享（`src/backend/pim_backend.py:60-61`）。有利条件是 M 维计费机制现成（`src/vpu/vpu.py:841-861` 按 `ceil(M/tile_m)` 折算重放，当前因产物 tile_m=1 未生效）。需要引入 closed batch 组（等长 prompt、等输出长度、整组进出）：组请求的 prefill 一次合批执行（GEMM 按 M=B×S 计价、注意力按 B 个块对角序列分别计价，2026-11-25 用户裁决修正），随后每步 decode 以合并行数 `Tq=B` 执行——GEMM 走 M=B 产物，权重搬运份数为 `ceil(B/tile_m)`；注意力类算子的 KV 长度按每槽一条新查询绑定（tp+1），不随 tq 放大；KV 按槽位分区计费与容量核算（组容量按 B 倍计）；准入从 FCFS 槽位补位扩展出「凑满 B 个等待请求成组」的形态（连续准入语义保留不动）。本文以「B=1 全部指标逐位回归不变；B>1 时 decode 权重字节/token 随 B 单调下降并趋于 `1/min(B,tile_m)`；closed batch 输出真实执行 batch」为验收判据（2026-11-23 修正：closed batch 维持等长硬前提、不含 padding——变长 padding 属连续 batching 范围（§2.3 边界），不再要求输出 padding 比例）。

**R7（P0）多 batch 评估模型**

　　评估模型要在全链路闭环之上回答「多 batch 的收益和代价各是多少、分别花在哪」。收益侧计价三类：权重复用（B 摊薄权重装载，成本截距口径 R5 已就位）、多 batch 并行（batch 组与 TP/PP 的空间组合）、流水（PP 各 stage 处理不同 micro-batch 的重叠收益，现状 stage 视图只做统计不门控执行）。代价侧计价三类：资源竞争（现状仅有 PU/链路互斥排队，无共享带宽仲裁——至少要把排队等待时间从总时间里归因出来）、同步等待（组内栅栏与 stage 同步的等待时间）、抢占（本期仅建模：给出槽位换出/换入的 KV 迁移与重装代价的解析估算项，不在调度器实现）。评估落点以 genesim 仿真输出为主体（按 batch 组分解计算/DMA/等待/争用的时间构成），辅以解析口径互检（权重字节/token 下界、容量墙公式、流水气泡占比）。本文以「时间构成守恒（各归因项之和与总时间对账）、收益与代价随 B 的变化方向符合分档分析（小 B 近线性→中 B 放缓→大 B 被容量/带宽卡住）」为验收判据。

**R8（P1）多 batch 性能优化**

　　在评估模型能看清瓶颈之后，按 PIM 收益排序落地三项优化。第一是权重复用：让 M=B 的真实编译产物贯穿计费与执行（R5/R6 的直接收益，验证其曲线即可）。第二是多 batch 并行：batch 组到 DPU 组的空间映射（数据并行形态），与现有 TP16/PP4 正交组合，评估权重复制与 KV 分布的权衡。第三是流水优化：PP stage 间 micro-batch 重叠调度以消减流水气泡（现状 bubble 只做事后统计）。每项优化独立可开关，本文以「优化前后 batch 曲线对比 + 收益归因到具体机制、任一优化关闭时 B=1 与 closed batch 路径行为不变」为验收判据。

**R9（P1）genesim batch 扫参与最优配置输出**

　　最终交付形态是一套扫参入口与结论输出。现状 `run_all.sh` 只跑单场景（`conf/ceva_64core_4e.yaml`，`max_inflight_requests: 1`，`run_all.sh:132-134`），run.sh 的 `--predictor-sweep` 是 GNN 预测器扫描与 batch 无关（`run.sh:509-514`），扫 batch 只能手动改配置逐次跑。需要在 `run_full_pipeline.py` 提供 batch 扫参入口（如 `--batch 1,2,4,8`），一次运行产出各档吞吐、TTFT/TPOT（P50/P95）、容量墙表、时间构成归因，聚合出「吞吐-时延随 B 变化曲线 + 瓶颈归因 + 最优 batch/策略推荐」。本文以「单条命令完成扫参并输出同指纹口径的可比报告、曲线形态与分档分析一致、能明确指出瓶颈资源与最优配置」为验收判据。

### 2.2 核心流程

　　一次 closed batch 评估的执行顺序如下。首先在入口指定 batch 档位，`compile_llama2(batch_size=B)` 导出 `[B,S]` prefill 图与 `[B,1]` decode 图，KV 布局按槽位规划、内存规划按 B 缩放并输出容量表——这三步保证「装得下且地址不冲突」是编译期事实。随后编排器在 NumpyBackend 上跑通整组 prefill+decode 并与 PyTorch 对拍（数值锚点）；算子编译按相位出真实产物（decode 档逐 B 的 M=B、prefill 档逐 B 的 M=B×S，与一次执行 `[B,S]` 的真实语义一致），sidecar 按 `(phase, batch)` 记录产物与 tile 形状（`prefill_b{B}`/`decode_b{B}`），refine_ir 精化成本。最后 genesim 以 batch 组语义仿真，输出指标与时间构成归因；扫参入口对多个 B 档重复执行并聚合成曲线与推荐。前一步是后一步的输入：图形状定了内存与 KV，内存定了编排地址，编排对拍了才敢信仿真侧的成本数字。

### 2.3 边界定义

　　包含范围：closed batch（等长、同相、整组进出）的编译、执行、仿真全链路与数值对齐；评估模型的收益与代价计价（抢占仅解析建模）；三项优化；扫参输出。不包含范围及取舍原因：连续 batching（动态合批、变长 padding、槽位动态补位、真抢占）的执行侧实现不做——等长 closed batch 是收益上界且复杂度最低，动态形态建议后续单独立项；cascade 多 CEVA 级联（PP2/TP2 多进程蓝图）本期不动——其段式 runner 尚未实施且与 64 核 TP/PP 维度重叠，多核维度由现有 tp16_pp4 形态承载；CEVA 厂商路径（export_gml→GC→TCS）不做 batch 扩展，仅保持 B=1 回归通过；FlagTree 预计零改动，仅当 R5 前置实验发现 M=8 权重搬运未摊薄时才把 kernel 循环调整列入。术语约定：需求原文中「排布器」按「编排器」理解（`runtime/executor.py` 主机编排与 genesim 调度器），项目文档无「排布器」术语。

## 三、技术约束与依赖

1. **代码改动影响范围**

　　改动集中在前端六处写死 batch=1 的层与 genesim 调度/计费层，`contracts/` 作为全仓唯一真源先行：改任何契约字段先 grep 调用方，改完同步全链路。各文件现状与动作如下表。

| 文件 | 现状（实测） | 本轮动作 |
| --- | --- | --- |
| `runtime/compile.py:78-83,237-239` | `[1,seq]`/`[1,1]` 静态导出、无 dynamic_shapes | 增加 `batch_size`，导出 `[B,S]`/`[B,1]`，mask/position_ids 按 batch 重构 |
| `memory/kv_layout.py:40,59-62,181-208` | offset 键 `(layer,head,which)`、容量无 B、单 pos 读写 | batch 槽位、`kv_bytes` 乘 B、按槽寻址 |
| `memory/mem_planner.py:317-321` | 激活按 batch=1 图形状 | 形状自动跟随 `[B,·]`，容量表输出（复用 `:458-472/:527-591` 台账） |
| `runtime/executor.py:15-22,41-94` | 单请求循环、`_as_batch_tensor` 写死 1 | per-slot 状态、batched decode、逐槽采样 |
| `genesim_bridge/{op_classify,cost_extractor,placement_export}.py` | `ShapePoint(Tq,Tp)`、符号表无 batch、M=1 钉死 | `(phase,B)` 代表点、符号表/缓存键/sidecar 加 batch 与 tile_m |
| `scripts/run_full_pipeline.py:794-831` | 无 batch 参数 | `--batch` 贯通编译与仿真 |
| genesim `src/config_loader.py:1391-1395`、`src/scheduler/gene_sim_scheduler.py:9300,10374`、`src/backend/pim_backend.py:60-61`、`src/vpu/vpu.py:841-861` | microbatch 禁令、task_key 含 request_index、执行线性乘、tile_m 已折算 | BatchGroup 语义、M=B 合并执行、跨槽权重一次计费、per-slot KV |
| FlagTree | M 分块/掩码/tasklet 摊块已就绪 | 预计零改动（实验 1 触发才动） |

　　`contracts/compile_slots.py` 的 GML 尺寸真源只有 tokens 轴、`gml_bridge/from_fx.py:447-450` 的 decode token 轴恒 1，属于 CEVA 路径契约，本期不扩展。

2. **潜在技术风险与应对建议**

　　第一，R5 前置实验可能揭示 M=8 产物仍选 tile_m=1 或权重搬运未摊薄，此时「权重复用收益」叙事要先回 kernel/tile 搜索侧解决——该实验必须最先做，是全需求的第一个闸门。第二，genesim 有并行开发会话（11-20 曾有他人提交卷入未提交改动），改动前先看 git status/mtime，提交时甄别归属。第三，genesim 调度器是万行级大文件，batch 组应以最小侵入方式扩展准入与绑定，不重写事件循环。第四，B=1 逐位回归是所有改动的硬门禁：任何 batch 代码在 B=1 时必须走完全相同的路径与产物。第五，容量硬失败保留（装不下直接抛），不引入防御性兜底。第六，7B 全量测试很慢，遵循「快速回归多跑、全量最后跑」的分层验证纪律。

## 四、非功能需求

　　可复现性：扫参各档使用同一 trace 种子、同一编译产物口径与同一硬件画像，差异只来自 batch 档位本身。可审计性：batch 布局、容量表、batch 组组成与时间构成归因保持可 print 的可读文本，第一处异常能定位到具体算子或槽位。兼容性：B=1 全链路行为与产物逐位不变，现有全部测试与验证命令通过；batch 相关新配置缺省值为 1。代码量约束：遵循「最小实现」——batch 维只做 replicate 不切分、不做通用调度框架、抢占只出解析项；被替代路径直接删。

## 五、验收标准

1. **回归判据**

　　快速回归跑 `python -m pytest tests/ -q -k "not llama2_7b"`（约 16 秒，232 个用例），要求全绿且 B=1 产物逐位不变；7B 全量 `python -m pytest tests/ -q -k "llama2_7b"` 与 genesim `./run_all.sh` 仅在阶段末与最终验收执行。

2. **快速端到端验收**

　　小规模快速验证用 `python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out`（确认 CEVA 路径 B=1 未受影响）与 `python scripts/run_full_pipeline.py --num-stages 4`（batch 档位分别取 1 与 >1 各跑一轮），检查编译期产物（图、KV/内存计划、sidecar）结构正确、仿真正常出数。

3. **逐项验收条件**

　　对应功能清单：R1 验 B 维导出与入口贯通；R2 验多槽 offset 无重叠且容量与手推一致；R3 验三区无重叠与容量表输出；R4 验 B=1/2/4 与 PyTorch 逐元素对齐；R5 验 `(phase,B)` 代表产物与 tile_m 入 sidecar，且 M=1 vs M=8 前置实验给出明确结论；R6 验 B=1 逐位回归与 B>1 权重字节/token 下降；R7 验时间构成守恒与方向性；R8 验优化前后曲线对比与归因；R9 验单命令扫参报告。每个功能点的代码与测试在同一次改动内完成。

4. **性能判据**

　　多 batch 无外部实测锚点，采用内部一致性判据：其一，B=1 全链路逐位回归不变；其二，B>1 在 NumpyBackend 与 PyTorch 逐元素对齐；其三，decode 权重字节/token 随 B 单调下降并趋于 `1/min(B, tile_m)`（tile_m 取 M=B 产物实际值）；其四，KV 容量墙位置与公式 `2×layers×B×max_seq×kv_heads×head_dim×dtype_bytes` 手推一致；其五，吞吐曲线形态符合分档分析（小 B 近线性上升、中 B 增速放缓、大 B 触容量/带宽墙后不再上升甚至下降）。所有指标出自同一指纹口径的运行，报告区分「仿真实测」与「解析估算（抢占项）」。

5. **不回归判据**

　　以下情形视为验收无效：B=1 任一现有用例失败或产物变化；CEVA 厂商路径 B=1 流程（export_gml 快速验证）受影响；容量检查被降级为告警或绕过；batch 结论来自 `max_inflight_requests` 调大而非真实 M=B 合并执行；batch 成本来自线性外推而非编译产物；任何用 try/except 掩盖上游契约不满足的兜底。

## 六、调研补充信息

　　本需求借鉴 2026-11-05 的四份既有文档（均在本仓 `docs/`）：`design-multi-batch-pre-to-deepseek-20261105.md` 提供四种并发方式的拆解、batch 计费口径与决策记录 D1–D10；`design-ceva-genesim-pre-to-evaluation-kimi-20261105.md` 提供阶段 A–D 的功能推进设计与统一指标口径；两份 conversation 文档是原始讨论记录。这些文档的代码级断言基于 11-05 工作区，第七节列出的部分断言已过时、以本轮实测为准；其方法论（closed batch 先行、成本口径先行、容量硬失败、指纹可比）仍然适用并被继承。

## 七、本轮实测对既有结论的修正

1. 原结论（11-05 文档 §9）：「FlagTree 只出 tile_m=1 的 pimir、按 token 重放、M 维完全缺失」。实测修正：`flatten_leading_dims`（`contracts/op_contract.py:125-130`）已把前导维折进行数 M，kernel 按 `BLOCK_M=16` 分块加掩码（`opcompiler_bridge/kernel_src.py:8-38`）、tile 搜索对 M 对称（FlagTree `TileToBudget.cpp:836-840`）、genesim 已按 `ceil(M/tile_m)` 折算重放（`src/vpu/vpu.py:841-861`）。M>1 在算子编译侧基本就绪；但被执行的 placement 产物实测 `"pim.tile-m" = 1`（抽查 `.opcompiler_cache`），摊薄机制从未在真实链路触发——卡点是 `genesim_bridge/placement_export.py:254-259` 钉死 M=1。
2. 原结论（11-05 文档 §9.5）：「成本链拿不到权重复用」。实测修正：拟合截距即权重字节且与 M 无关（`genesim_bridge/cost_extractor.py:393-395`）、`_line_weight_residency_bytes` 的 stationarity 计费只记一次权重（`genesim_bridge/ir_cost.py:385-408`）——权重复用的成本结构已存在，缺的只是 batch 维进不了符号表（`cost_extractor.py:355-362`）。
3. 原结论（11-05 文档 §8.1）：「genesim 默认配置是含 4 GPU 的异构系统，不代表 PIM 芯片」。实测修正：当前默认仿真形态已切换为 `conf/ceva_64core_4e.yaml`（CEVA 画像、tp16_pp4/64 核、`max_inflight_requests: 1`），异构默认的告警对新形态不再适用，但「扫参必须显式锁定硬件画像与放置 sidecar」的原则继续有效。
4. 记忆断言（11-05）：「图编译器 batch=1 写死在 export/KV/executor」。本轮复核仍成立，但补充：容量核算层已新增多请求端口（`memory/mem_planner.py:458-472` 逐请求 KV 峰值、`:527-591` `tensor_pu_usage(concurrency=)`），可与 genesim 容量审计对账，是 R3 的现成起点。
5. 本仓 CLAUDE.md 与本文初稿引用的逐节点对拍器 `tests/assert_node_matches_ref.py` 自 init 提交（`f8ac5c6`）起即为空文件、无调用方；真实对拍机制是 pytest fixture 执行计划后与单卡 PyTorch logits 对比（`tests/test_executor_llama2_7b.py:126-141`，容差 max<0.2/mean<0.02）。B>1 对齐按该模式新建用例，空文件按「删优于加」清理。
6. 本文档初稿 R5/R6 曾写「prefill 的 M=B×S 合并代表点」。设计阶段推演曾以「合并 prefill 对 GEMM 权重搬运无收益（`ceil(B×S/tile_m)=B×ceil(S/tile_m)`，S≥tile_m 时恒等），且注意力计费按 `Tp+Tq=B×S` 会把工作量放大 B 倍（真实为块对角 B 份 S×S）」为由修正为 prefill 按槽独立执行（Tq=S）、仅 decode 合并（Tq=B）。2026-11-25 用户裁决再修正：round17 以真实算子编译证伪该等式（S 非 tile_m 整数时不成立，合批装载严格更少），且注意力的正确做法是按 B 个块对角序列计价而非放弃合并。终态：prefill 与图编译/编排器一次执行 `[B,S]` 的真实语义一致——合批（逐 B 导出 `M=B×S` 的 `prefill_b{B}` 产物、GEMM 按 M=B×S 计价、注意力按 B 个块对角序列分别计价），「prefill 按槽独立执行（Tq=S）」的设计阶段修正结论作废；仅 decode 合并（Tq=B）不变（同 design §八.4）。

## 八、待确认事项

1. R5 前置实验需确认 M=8 产物的 `pim.tile-m > 1` 且权重搬运接近 1 份（当前 M=1 产物 tile_m=1、摊薄未触发）；若 tile 搜索在 M=8 下仍选 tile_m=1 或搬运未摊薄，权重复用需先回到 kernel/tile 搜索侧解决。  
2. 抢占建模的深度：解析估算项（KV 迁移字节/带宽 + 上下文切换常数）还是带事件仿真的换出重演，建议设计阶段以前者为默认。  
3. batch 扫参档位（建议 1,2,4,8，饱和点由 M=B 产物的 tile_m 实测后定）与 7B 全量测试的触发时机（建议仅最终验收）。  
4. 连续 batching（动态合批/变长/真抢占）是否后续单独立项。  
5. 甲方对 TTFT/TPOT 的具体 SLO 目标仍未提供（11-05 文档附录 F 遗留），影响 R9「最优配置」的推荐准则定义，暂以吞吐-时延 Pareto 为准。  

## 九、附原始需求描述

　　下面是用户提供的原始需求，原样保留。

```
背景：
当前走通了从大模型到图编译到算子编译到模拟的全流程，然而当前只考虑单batch，对多batch支持不完善，无法发挥pim架构的性能。

核心目标：多batch支持
1.支持多batch下的全链路打通（图编译->算子编译->genesim，包括：内存分配、kv cache管理以及编排器）；
2.支持多batch下的评估模型设计和实现，考虑多batch带来的性能收益（并行，流水，权重复用等等），以及抢占、资源竞争、同步等待的代价；
3.支持多batch引入性能收益的优化，例如：多batch权重复用，流水优化，多batch并行等等；
4.多batch场景下的，支持kv cache，内存管理，排布器优化；
5.genesim可以进行不同batch性能评估，给出性能瓶颈或者最优的配置。

参考代码：
/data/fjg/flagOS/flagos-pim-compiler  图编译器（主工作区）                                                                                   
/data/fjg/flagOS/genesim  模拟器                                                                                              
/data/fjg/flagOS/flagOS-installers/FlagTree  算子编译器
/data/fjg/ceva`：CEVA 厂商工具链和实测结果，只读参考

思路：
可借鉴：/data/fjg/flagOS/flagos-pim-compiler/docs/中
-rw-rw-r-- 1 jgfeng jgfeng   39165 Nov  5 09:39 conversation-all-kimi-20261105.md
-rw-rw-r-- 1 jgfeng jgfeng   33924 Nov  5 09:28 conversation-multi-batch-deepseek-all-20261105.md
-rw-rw-r-- 1 jgfeng jgfeng   51839 Nov  5 09:20 design-ceva-genesim-pre-to-evaluation-kimi-20261105.md
-rw-rw-r-- 1 jgfeng jgfeng  63180 Nov  5 09:16 design-multi-batch-pre-to-deepseek-20261105.md
的部分思路。

验证：
1.支持多batch场景下图编译、算子编译、内存管理、kv-cache管理、排布器相关功能测试；
2.支持评估模型关于多batch考量的相关测试，包括：性能收益（并行，流水，权重复用等等），以及抢占、资源竞争、同步等待的代价，符合客观实际需求；
3.支持多batch引入性能收益的优化，例如：多batch权重复用，流水优化，多batch并行等的相关验证；
4.支持多batch场景下的，支持kv cache，内存管理，排布器优化相关验证；
5.genesim可以进行不同batch性能评估验证，符合客观实际；
6.原始的功能验证，包括：
pim-compiler:
# 快速回归，约 16 秒，232 个用例
python -m pytest tests/ -q -k "not llama2_7b" （建议多用这个，其他的只有最后阶段才做验证）

# 7B 全量，另开一轮
python -m pytest tests/ -q -k "llama2_7b"

# 快速验证：取 1 层，约 1 分钟
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out

python scripts/run_full_pipeline.py --num-stages 4

genesim：
# 1. 激活环境
source  /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh
./run_all.sh
```
