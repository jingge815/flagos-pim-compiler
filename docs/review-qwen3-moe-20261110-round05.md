# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261110-round05
> 创建日期：2026-11-10
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 5 项问题（2 严重、2 高、1 中）：四层真实权重数值对拍和双语义主链均实际跑通，但逐层核对真实执行计划发现动态分派只在第一层生效，GeneSim 又把设备侧加权累加错误抽象成主机共享算子；严格产物和容量契约仍有旁路，48 层当前代码主链尚未结束，不能关闭验收。
　　验证前已 `source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh`。临时镜像、探针、算子缓存、日志和产物均在 `/tmp/qwen3-review-round05`，未运行 CEVA 或 `-k llama2_7b`，未触碰另一个 tune-loop 工作区。实际执行结果如下：
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider`：**26 passed、1 skipped（318.11 秒）**；8B 无权重，SKIP 记为未覆盖。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`：默认 `dynamic_dispatch` 退出码 0，**88.483 s / 8.985 t/s**；显式 `--moe-execution dense_sim` 亦退出码 0，**91.012 s / 8.735 t/s**。独立审计确认两者均有 3681 个同编号算子、1557 个 GEMM、1536 个专家 GEMM、23 份有效哈希的 pim mlir，tiles 为 `[32,128,512]`；512 个专家 `VECTOR_MUL` 都是 `pim.eltwise<mul>`，系数为 `{Tq:768}` 和 `{Tq:4608}`，不是 hidden 2048 或 dense 6144；专家 0..127、四层三投影、4 条 lm_head 分片均覆盖。trace 均为 `{'pimir': 8858, 'absorbed': 2048}`，无 template/unknown。上述证据不覆盖问题 1、2 所指的错误门控和错误豁免。
- `cd genesim && ./run.sh --test sim`（同源码隔离镜像）：**45/45 个测试文件通过**。`python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider`：**1904 passed、14 skipped、52 deselected、0 failed（1361.44 秒）**；补齐隔离镜像的既有模型产物后，本轮额外跳过的 4 个桥接用例均单独复跑通过。另一个 `llama2_7b_flagtree.ir` 旧产物结构对拍失败不属于 Qwen3 改动，未据此运行 Llama2 长回归。
- 现有 FlagTree 安装执行 `triton-opt test/Dialect/TritonPIM/paged_attn_phases.mlir -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck ...`：通过，未重建源码树。
- 当前源码的 `full30b-pipeline-r5.service` 已生成 48 层骨架：**44117 个算子、18432 个专家 GEMM、54.0 GiB 专家权重、61064245248 字节完整参数驻留**；截至报告核对时仍在图编译/放置阶段，尚无 8 GiB 主链终态。既有 round04 数值与主链单元也仍运行且使用旧镜像，故本轮没有当前 48 层数值通过证据，历史 23 passed 和 round02 的 OOM 均不冒充本轮结果。
- 反向实测：两层 MoE 的两个 dispatch 均存在，但 97 条 gate 命令全部属于第 0 层，第 1 层为 0；把真实 4 层未融合的 op156（layer0/expert0/down_proj）加入 `absorbed_ops` 后覆盖审计仍通过；未知 `BRAND_NEW_OP` 伪标 absorbed 后，编译器和直接 GeneSim 均接受，去掉该字段则均拒绝；直接 GeneSim 还接受错误的专家号，而主链对同一错误按缺失 `(层,角色,专家)` 组合拒绝。真实 4 层 sidecar 缺 `hw_budget`、或写 `sys_reserve=0/align=1`，均通过严格加载和 pimir 覆盖校验，只有 16 GiB 总容量会被拒绝。
　　已确认的局部「不变式 → 实现 → 证据」为：真实配置 → `model_ir.build_from_hf_config` 的专家几何 → 4/48 层专家数与驻留实测；编译配置 → `resolve_moe_execution`、parser 和 sidecar → 两种 summary 的执行语义一致；本地分片形状 → `_measure_kernel_tile_n`/`_bpath_pimir` → 23 份真实 FlagTree 产物与正确乘法文本；原始 HF 稀疏前向 → `restore_hf_moe_forward` → 4 层逐 token 对拍通过。P1-1 的 4 层实测 prefill 1.148×、decode 1.202×已按用户确认关闭“显著加速”口径，但该确认不替代逐层只执行命中专家的功能验收。

## 二、修改建议
### 问题1（严重）: 动态分派跨层失效，导致后续专家全量执行
　　**问题说明**：request §2.1 P1-1 与 design §4.8 要求每层只执行命中专家的子计划，design §十.6 又要求 GeneSim 与该 any-hit 执行语义一致。`runtime/exec_plan_gen.py:84-108` 的 `_expert_provenance` 在所有输入上无层边界地传播专家归属：上一层累加结果已是 `_MULTI_EXPERT`，传到下一层后与下一层专家权重相遇，计算节点随即被判为多专家而失去 gate。两层真实编译探针中，`moe_dispatch_0`、`moe_dispatch_1` 都存在，但 **97 条 gate 命令全部属于第 0 层，第 1 层为 0**；第 1 层仅 24 个 `get_attr` 权重节点还带专家号，计算节点均失去归属。因而后续层仍执行全专家稠密路径，GeneSim 却对每层专家都按命中比例折算，执行与仿真同时偏离契约。数值仍可能因稠密路径正确而通过，不能据此证明分派生效。
　　**根本原因**：专家归属既表示“当前专家子图”又被允许穿过 MoE 块边界和残差链，缺少按层隔离的传播不变式。round01 问题 8 的拓扑传播修复只验证了 dispatch 数量、标签非空和聚合跳过比例：`tests/test_moe_dispatch.py:139-155` 没有逐层断言，`529-570` 在两层各跳一半时，仅首层生效也能得到约 50% 的合格比例；`scripts/bench_moe_dispatch.py:172-175` 只统计 dispatch 写出的跳过集合，不统计各 dispatch 实际拦下的命令。同一根因影响第二层起的 prefill/decode、分块、TP/PP 与通信门控；已接受的加速比只关闭时间口径，不能关闭该功能缺口。
　　**涉及文件**：`flagos-pim-compiler/runtime/exec_plan_gen.py:80-108,751-759`；`flagos-pim-compiler/tests/test_moe_dispatch.py:139-155,529-570`；`flagos-pim-compiler/scripts/bench_moe_dispatch.py:120-175`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.8、§十.6。
　　**修改思路**：把专家传播域限定为“本层、本专家的连续子图”，跨层激活和上一 MoE 块的多专家状态应作为中性输入，而不是污染下一层候选；在 MoE 块入口重新开始归属，在共享累加出口终止归属，不能用清空全部标签或只修第二层探针的方式处理。测试至少用三层及 PP 跨段模型，逐层断言计算、掩码、加权、视图和通信命令都挂到本层 dispatch，并在后端记录每个 dispatch 实际未提交/已提交的节点，覆盖 prefill、decode、强制分块与 EP 合法路径；基准同时报告逐层真实命令覆盖，不能只报告跳过集合和总加速比。

### 问题2（严重）: 加权累加错标主机，导致产物与计费失真
　　**问题说明**：design §3.1 的数据流明确要求路由加权 `mul` 和累加 `add` 走 DPU，§十.6 要求可门控专家工作按 any-hit、共享累加全量分别计费。实际静态前向在 `backend/architectures.py:40-48` 逐专家执行 `where → 路由 mul → 累加 add`；第 0 层计划实测这些操作均为 DPU `launch`，其中 `where` 和路由 `mul` 带 gate，后续 `add` 不带 gate，证明三类操作不能合成一种主机共享成本。`genesim/src/ir/model_ir.py:2453-2479` 却把每专家整段写成单个 `MOE_ACCUM`，同时设 `device_hint="cpu"`、`moe_cost_role="shared_reduce"`、`flops={"Tq":2*hidden}`；真实 4 层 dynamic/dense 产物中的 **512 个 MOE_ACCUM 全部按 host 豁免且全为 shared_reduce**。于是设备算子被排除在 pimir 覆盖之外，路由加权乘法也未按命中率缩放；问题 1 导致后续层未门控，更加说明不能用统一假设掩盖真实计划。design §十.3 目前把 MOE_ACCUM 写成 host，与 §3.1 和实际代码冲突，设计本身也需纠正。
　　**根本原因**：GeneSim 图构造把“路由选择/加权/共享累加”压成一个便于豁免的抽象节点，round01 增加 `where` 防溢出、round03 区分 any-hit 与共享累加时，都没有用真实 FX 计划逐算子校准设备落点和成本角色；round04 的覆盖修复又把该抽象产生的 host 豁免当成合法事实。相同根因影响 dense/dynamic、prefill/decode、sidecar 豁免、trace 完整性和首专家无加法这一边界。PP×EP 在 `graph/spec_prop.py:267-288` 中确有 host 聚合路径，不能反向一刀切把合法 EP 也判错，必须与本轮 TP 主链分策略表达。
　　**涉及文件**：`genesim/src/ir/model_ir.py:2453-2484`；`flagos-pim-compiler/backend/architectures.py:40-48`；`flagos-pim-compiler/runtime/exec_plan_gen.py:84-150,751-759`；`flagos-pim-compiler/genesim_bridge/placement_export.py:635-664`；`genesim/tests/sim/test_moe_execution_cost.py:36-60,100-108`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §3.1、§十.3、§十.6。
　　**修改思路**：先按真实执行计划统一设计，再把选择/路由加权与共享累加拆成有独立设备、门控和成本角色的节点：TP 主链的 `where` 与路由 `mul` 属可门控 `expert_work`，共享 `add` 才属 `shared_reduce`；首专家没有上一累加值，不得写死相同公式。设备侧节点必须进入 pimir 覆盖与 trace，若原语暂缺应显式失败而不是借 host 豁免；PP×EP 的 host 聚合则由实际切分语义单独声明。增加跨仓对拍，逐项比较 FX 命令的落点/gate、IR 角色、sidecar 豁免和最终 trace，覆盖 dense/dynamic、首/中/末专家及 TP/EP，不能用总时延或“trace 全为 pimir”替代算子级语义验证。

### 问题3（高）: 严格产物校验不闭合，导致缺算子错专家放行
　　**问题说明**：design §十.4 要求严格模式以源 IR 为准核对完整覆盖、显式豁免及 MoE 的 `(层,角色,专家)` 集合，未知设备算子不得回退。当前三处却直接信任 `absorbed_ops`：`gene_sim_scheduler.py:600-603,911-913`、`run_full_pipeline.py:468,481-482` 和 `1047-1054`。实测把真实 4 层未融合的 op156 加入该列表后，带哈希的覆盖审计仍通过；未知 `BRAND_NEW_OP` 伪标 absorbed 后，编译器和直接 GeneSim 也都接受，移除字段才拒绝。调度器随后在 `gene_sim_scheduler.py:2665-2671` 为这些节点生成空 trace，主链 trace 验收又把 absorbed 排除在必需集合外，可实际漏掉设备工作。另一条路径上，直接 GeneSim 没有主链 `_moe_expert_triples` 的等价校验：仅篡改 sidecar 专家号，主链拒绝而直接 GeneSim 接受。
　　**根本原因**：产物契约在编译器脚本和 GeneSim 中各自实现且规则不一致，同时把 sidecar 自报的“已吸收”当成无需证明的事实，未与 `fused_activations`、IR 依赖边和合法吸收类型交叉核对。round04 问题 2 只覆盖了 `pimir_exempt_ops`、落点和来源白名单，没有覆盖 absorbed 伪造及直接 GeneSim 的专家身份，因此换路复发。正常 512 个 SILU 吸收条目已由本轮独立脚本逐项验证合法，修复不得误伤这类真实融合。
　　**涉及文件**：`genesim/src/scheduler/gene_sim_scheduler.py:574-614,894-970,2518-2525,2664-2672`；`flagos-pim-compiler/scripts/run_full_pipeline.py:451-556,977-1061`；`flagos-pim-compiler/genesim_bridge/cost_extractor.py:701-729`；`genesim/tests/sim/test_compiler_placement.py:2486-2566`；`flagos-pim-compiler/tests/test_genesim_simulation.py:170-316`。
　　**修改思路**：为 absorbed 建立可验证的结构契约：每个被吸收编号必须能由融合清单中的层/角色/专家、相应生产者和 IR 依赖边唯一证明，且只能落在允许的吸收类型；主算子仍必须有真实 pimir，吸收节点只能生成对应的空 trace。编译器导出/A2、skip-simulation、直接 GeneSim 和 trace 验收共用同一覆盖规则，直接 GeneSim 也按源 IR 核对逐层、逐角色、逐专家及条目元数据。增加 GPU/CPU/PIM 三落点的伪 absorbed、错误专家号、越界/多余编号反例，以及真实 SILU 融合、host、边界和非严格旧产物的正向测试。

### 问题4（高）: 容量契约缺项仍放行，导致容量同源失效
　　**问题说明**：request §2.1 P0-3 的 8 DPU × 8 GiB 是硬约束，design §十.5 要求 `hw_budget` 随 sidecar 回传并由 GeneSim 核对同一预算。`gene_sim_scheduler.py:627-629` 在字段缺失时直接返回，631-641 行又只比较 `mram_bytes_per_dpu`，不校验 `sys_reserve_bytes` 与 `align`。本轮用真实 4 层 IR 和完整 pimir sidecar 实测：删除 `hw_budget`、或写 `sys_reserve=0/align=1`，严格加载与完整覆盖校验都通过；仅显式写 16 GiB 才拒绝。编译器侧 `step_bcd_export`、`verify_sim_ir_matches_sidecar` 和 skip-simulation 的覆盖审计也没有容量字段检查，所以“主链写出正确值”不能防止后续产物被替换或旧入口绕过。
　　**根本原因**：容量校验被实现成可选的单字段提示，而不是严格产物的必需 schema；兼容旧 sidecar 与接受本轮严格 Qwen3 产物缺项混为一谈。round03 问题 2 只新增“4/16 GiB 值不匹配拒绝”和“旧 sidecar 缺预算接受”测试，未覆盖新严格产物缺字段、错误预留/对齐及 skip-simulation，因而只收敛了单个数值探针。同一根因影响直接 GeneSim、仿真前复核和只到 sidecar 的验收路径。
　　**涉及文件**：`genesim/src/scheduler/gene_sim_scheduler.py:478-546,616-641`；`genesim/tests/sim/test_compiler_placement.py:2965-3004`；`flagos-pim-compiler/scripts/run_full_pipeline.py:593-721,899-974,1202-1211`；`flagos-pim-compiler/genesim_bridge/placement_export.py:723-730`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §十.5。
　　**修改思路**：把总容量、系统预留和对齐组成不可拆分的 sidecar 容量契约，在导出、skip-simulation、仿真前复核及直接 GeneSim 中统一校验。本轮 IR 带结构身份或严格模式启用时，字段缺失、类型错误和任一值偏离都必须失败；只有明确标识的旧式非严格产物才可走兼容分支。为三字段分别增加缺失/错值反例和各入口测试，并继续用 48 层实际每 DPU 所需/可用字节判定容量，不能借补齐字符串宣称 8 GiB 已通过。

### 问题5（中）: 全模型验收未闭环，导致完成状态失实
　　**问题说明**：本轮要求 48 层真实模型从图编译到 GeneSim 全链实测，implement §1.2 也把全链产物和测试通过列为完成标准。当前 §1.3 却写“全部完成”，同段承认 48 层未重跑，K7 又明确“48 层全模型主链未在 8 GiB 下实测跑通，未关闭”。本轮当前源码的 `full30b-pipeline-r5` 只完成真实骨架并仍在图编译/放置阶段，round04 的两个旧镜像任务也未结束；因此 4 层通过、round01 的历史 23 passed、round02 的 OOM 和 48 层元数据核算都不能支持“全部完成”。
　　**根本原因**：实施文档的完成汇总不是由未关闭验收项生成，历轮不断更新切片数字和历史状态，却保留同一个超前总标题，导致 round02 问题 7、round03 问题 5 的文档收敛再次只停留在局部文字。当前代码的 48 层数值、48 层主链容量、4 层切片和历史结果虽然已在后文分栏，顶部状态仍把它们合并成错误结论。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md:28-65,274-286,420-423,508-529`；`flagos-pim-compiler/docs/review-qwen3-moe-20261110-round04.md:44`；本轮 `/tmp/qwen3-review-round05/pipeline-full30b-r5.log`、`pipeline-full30b-r5-logs/bcd_export.log`。
　　**修改思路**：将“代码修复完成”和“需求验收完成”分成两个由任务清单推导的状态，K7 未关闭时不得写全部完成。待当前 systemd 单元结束后回填同一命令、退出码、当前源码身份和真实产物；若超 8 GiB，逐 DPU 记录实际所需/可用字节并保留遗留，不改成 16 GiB；若仍运行或被环境终止，明确记待验证/环境阻塞。48 层数值与主链、4 层与历史结果继续分开，不能用文档文字替代产物验证。
