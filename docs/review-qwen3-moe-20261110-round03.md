# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261110-round03
> 创建日期：2026-11-10
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 5 项问题（3 严重、1 高、1 中）：4 层真实权重数值对拍和两种执行语义的产物链均实际跑通，但图骨架缺少必要算子，容量契约未统一，动态分派计费与真实 any-hit 执行不一致；48 层主链仍不能宣称交付。
　　验证均使用当前源码，临时配置、探针、算子缓存和跨仓产物放在 `/tmp/qwen3-review-round03`；未启动 48 层全模型 pytest，未运行 `-k llama2_7b` 或 CEVA 流水线。实际命令与结果如下：
- 先 `source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，确认 torch 2.9.1+cpu、transformers 4.57.6。`QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider`：**23 passed、1 skipped（681.66 秒）**；8B 无权重，SKIP 记为未覆盖。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`（默认 dynamic_dispatch，GeneSim 根目录与日志重定向到 `/tmp`）：退出码 0；骨架与精化 IR 均为 3536 个同编号算子，sidecar 3018 条且路径/哈希全有效，专家 0..127 全覆盖，tiles 为 `[32,128,512]`，10758 条 trace 全部来自 pimir，`total_time_s=39.378`。显式 `--moe-execution dense_sim` 同样退出码 0、trace 全部来自 pimir，但实测 `total_time_s=76.052`，不是实施文档的 46.627。
- `cd genesim && ./run.sh --test sim`：**45/45 个测试文件通过**。现有 FlagTree 安装执行 `triton-opt test/Dialect/TritonPIM/paged_attn_phases.mlir -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck ...`：通过，未重建源码树。
- `python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider`：**1884 passed、10 skipped、52 deselected、13 warnings（1859.98 秒）**，失败 0；未设置两个真实模型路径，真实权重覆盖以上述独立目标命令为准。
- 反向探针实测：动态模式下 `Tq=16` 的专家 GEMM 按 0.0625 折算，而 any-hit 全张量执行比例应为 0.643925870；MOE_ACCUM 也被统一缩成 0.0625，尽管其共享累加未门控。不同词表、不同 `model_id` 的 3536 算子 IR 被真实 sidecar 接受，A2 也复用了旧精化 IR。48 层 safetensors 仅读头部元数据，未加载权重。
　　48 层历史日志 `/tmp/test_target_full_30b_systemd.log` 已核实为 **23 passed、1 skipped（25426.83 秒）**，其中 30B PASS、8B SKIP；`full30b-r2.service` 当前为 `Result=oom-kill`，按环境阻塞登记为本轮代码未覆盖，不据此判代码失败。已确认且不再列为问题的契约链为：编译配置 → `resolve_moe_execution` → parser/sidecar/summary 的模式传递；独立 HF 稀疏参考 → `restore_hf_moe_forward` → 4 层逐 token 对拍；严格 pimir 校验 → sidecar 全哈希与全 pimir trace；VECTOR_MUL 映射 → 512 份 `pim.eltwise<mul>`，768 宽度 → 每个 prefill 乘法 98304 FLOPs。P1-1 的 4 层实测 prefill 1.148×、decode 1.202× 已按用户确认关闭，不作为本轮失败项。

## 二、修改建议
### 问题1（严重）: 图骨架漏必要算子，导致全推理成本缺项
　　**问题说明**：request §2.1 P0-1/P0-2 与本轮要求图骨架真实表达编译器执行的 Qwen3 全推理图，不能跳过算子。`genesim/src/ir/model_ir.py:2531-2595` 在最后一层只生成 RMSNORM→MODEL_OUTPUT，没有 lm_head GEMM；Qwen3 的 q_norm/k_norm 也未生成。真实 4 层产物中词表维 GEMM 为 0，sidecar 与 trace 自然也没有这些算子。实测 `2×num_parameters-parameter_bytes=622331904` 字节，恰好等于 lm_head 的 622329856 字节（`[151936,2048]`、fp16）与 4 层 Q/K 归一化的 2048 字节；仅 lm_head 每 token 还漏计 622329856 FLOPs。主链只检查八种投影角色（`scripts/run_full_pipeline.py:151-180`），所以退出码 0 与“trace 全部为 pimir”不能证明全推理图完整。
　　**根本原因**：通用构造器沿用了简化 Llama 骨架，本轮只补 MoE 分支，没有建立“编译器导出图 ↔ GeneSim IR”的完整算子与权重覆盖契约；`model_parser.py:188-193` 又单独用完整 HF 模型统计 `num_parameters`，掩盖了图内 `parameter_bytes` 的缺项。round01 问题1只收敛了专家数、top-k 与 768 宽度，未做全图 parity，因而同类遗漏继续通过验收。相同根因影响 dense Qwen3 与 MoE、model_parser、精化、sidecar、容量和最终仿真，不是一个角色探针能覆盖的问题。
　　**涉及文件**：`genesim/src/ir/model_ir.py:1268-1290,2531-2595`；`genesim/scripts/model_parser.py:183-193`；`genesim/tests/sim/test_model_ir.py:1219-1278`；`flagos-pim-compiler/backend/architectures.py:56-64,99-107`；`flagos-pim-compiler/genesim_bridge/placement_export.py:41-67,774-843`；`flagos-pim-compiler/scripts/run_full_pipeline.py:151-180`。
　　**修改思路**：先按架构建立完整算子/权重对齐清单，再统一驱动图生成和验收：最终 RMSNORM 后补 lm_head GEMM、`lm_head.weight` 驻留和输出依赖，Q/K 投影后按 HF 顺序补 q_norm/k_norm 及权重；模型级输出头的放置不能只走现有“逐层权重”循环，应显式归到最后流水段并生成 FlagTree 产物。测试逐架构对拍编译器与 GeneSim 的角色、形状、权重名和数量，断言非 tied 模型 `parameter_bytes` 与完整执行 dtype 参数总量一致，并追踪 lm_head 的 sidecar、pimir、trace 与 FLOPs，避免只补当前两个探针。

### 问题2（严重）: 容量契约仍然分裂，导致全模型未交付
　　**问题说明**：request §2.1 P0-3 及本轮确认要求 8 DPU×8 GiB 硬约束，主链、真实测试和 GeneSim 必须使用同一容量契约。`scripts/export_pp_placement.py:154-162` 仍为 4 GiB，`tests/test_qwen3_30b_real.py:40-55` 与 `scripts/bench_moe_dispatch.py:63-72` 仍为 16 GiB，GeneSim 则取 8 GiB（`genesim/src/pim/capacity.py:17,49-118`）；design §4.7 的 16 GiB 也仍保留。4 层主链通过和 16 GiB 数值对拍均不能验收 48 层 8 GiB 容量。
　　只读 48 层权重元数据并按 tp4×pp2、64 字节对齐计算：每台专家分片为 7247757312 字节；stage0 的权重加最小 KV 下界为 **8100720704** 字节，stage1 为 **7633977408** 字节，均未含激活与内核。主链 4 GiB 扣 64 MiB 预留后仅 **4227858432** 字节，分别超 **3872862272** 和 **3406118976** 字节。8 GiB 扣同预留后为 **8522825728** 字节，两段只剩 **422105024** 和 **888848320** 字节给激活与内核；全量是否超过 8 GiB 尚未实测，必须保留为遗留，不能宣称通过或擅自改成 16 GiB。
　　**根本原因**：架构注册表只统一模型加载、宽度、角色和命名，没有统一硬件容量真源；各入口独立硬编预算，GeneSim 又独立取默认值。round01 问题7、round02 问题1仅以“待确认”保留，用户已经确认后代码仍未收敛；同一根因影响所有 PP/TP 策略、执行期基准、三区规划及全模型 sidecar。
　　**涉及文件**：`flagos-pim-compiler/scripts/export_pp_placement.py:154-182`；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:34-55`；`flagos-pim-compiler/scripts/bench_moe_dispatch.py:56-73`；`flagos-pim-compiler/memory/mem_planner.py:339-353`；`genesim/src/pim/capacity.py:17-118`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md:223-239,381-393`。
　　**修改思路**：用一份不可变容量契约统一 `HwBudget`、`PIMHardwareConfig`、真实测试、执行基准和 GeneSim 配置，并随 sidecar 回传供另一端核对，禁止各入口再写 4/8/16 GiB 字面量。按 8 GiB 反推请求窗口与槽池，逐 DPU 记录权重、KV、激活、内核的所需/可用字节及三区无重叠；环境允许后以同一契约实测 48 层导出、sidecar 与仿真，超限就如实保留所需字节和未通过状态，不用 4 层切片或 16 GiB 历史日志替代。

### 问题3（严重）: 动态计费背离执行，导致仿真严重低估
　　**问题说明**：本轮要求 GeneSim 代价与所选执行语义一致。编译器 dispatch 在 `runtime/exec_plan_gen.py:123-128` 用 `(routing!=0).any(dim=0)` 判断“是否有任意 token 命中”，命中后 `backend/architectures.py:30-48` 仍对完整 `Tq` 张量执行专家子图；GeneSim 却在 `gene_sim_scheduler.py:335-351` 对所有非 RELAY 专家算子固定使用 `top_k/E`，并把该比例传给 PIMVPU 的 trace 时延。在与现有静态模型相同的均匀路由期望下，每 token 无重复 top-k 的 any-hit 全张量执行比例应为 `1-(1-top_k/E)^Tq`，不是 `top_k/E`；文档中的通信公式 `1-(1-1/E)^(Tq·top_k)` 也按可重复独立抽样推导，不等于该概率。若改用真实路由，则应直接按实际命中集计费。
　　实测 `E=128,k=8,Tq=16` 时上述期望比例为 **0.643925870**，当前 GEMM 为 **0.0625**；本次首请求 `Tq=726` 时期望命中概率已约为 1，当前仍只计 1/16，真实命中集不在本探针中断言。`MOE_ACCUM` 同样带 `expert_id` 而被整体缩成 0.0625，但实施 §9.2 明确编译器仍执行全专家累加链，其中共享累加部分当前应为 1，不能与可门控的路由乘法共用一个比例。dense_sim 保持全量本身合理，但本轮 dense/dynamic 的 76.052/39.378 秒之比建立在错误折算上，不能作为加速证据。
　　**根本原因**：design §十.6 把理想的 token-专家对成本当成当前 any-hit 全张量内核的成本，实现又用 `expert_id` 同时表示“产物归属”和“可跳过的计算”，未区分专家子图、通信和共享主机累加。round02 问题2只修复模式字符串传递并增加 GEMM/RELAY 公式测试，没有与真实执行计划核对，故在 prefill 与 MOE_ACCUM 路径复发。
　　**涉及文件**：`flagos-pim-compiler/runtime/exec_plan_gen.py:84-150`；`flagos-pim-compiler/backend/architectures.py:30-48`；`genesim/src/ir/model_ir.py:2126-2130,2280-2305`；`genesim/src/scheduler/gene_sim_scheduler.py:304-351,6574-6638`；`genesim/src/vpu/vpu.py:276-320`；`genesim/tests/sim/test_moe_execution_cost.py:26-100`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md:383-390`。
　　**修改思路**：先修正设计中的执行—成本契约，再从同一语义派生所有入口：当前 any-hit 全张量专家计算及相应通信按 `1-(1-k/E)^Tq`，decode 自然为 `k/E`；MOE_ACCUM 的共享累加部分不按专家命中率缩放，路由加权部分则按真实门控口径计。不要用 `expert_id` 推断一切计费，应在 IR/计划中拆分可门控专家工作、命中并集通信和共享归约的成本角色；若以后改为按 token gather 或门控累加，必须同步改执行计划与成本。补 `Tq=1/16/726`、GEMM/RELAY/MOE_ACCUM、trace 时延及 dense 对照测试，并断言与真实计划的执行覆盖一致。

### 问题4（高）: 产物身份校验不全，导致跨模型复用
　　**问题说明**：design §十.4 与本轮要求 sidecar、精化 IR、仿真 IR 同源。`gene_sim_scheduler.py:1818-1868` 只核对算子总数、op_id 和 op_type，`_validate_moe_config`（600-632）只核对专家数/top-k/768/执行语义；`step_a2_refine_pimir_ir`（`run_full_pipeline.py:203-221`）也只按 op_id 序列与精化档位复用缓存。反向探针将真实 4 层配置的词表从 151936 改为 155648、`model_id` 改为另一模型后，严格模式仍接受原 sidecar；另一探针替换骨架的 `model_id` 与 `parameter_bytes` 后，A2 明确打印“跳过重新精化”，继续保留旧模型的精化 IR。实际主链的 IR/PartitionPlan 还均为 `model_id=unknown`，`export_pp_placement.py:139-143` 仅提示不失败。
　　**根本原因**：模型身份被缩减为少量 MoE 字段和算子编号，没有覆盖完整模型几何、算子形状与成本口径的结构指纹；缓存键也没有这份身份。round02 问题3只补“缺 moe_config”单例，未覆盖同算子序列不同模型的路径。该根因同时影响直接 GeneSim、主链 A2 缓存、skip-simulation 后的旧产物复用及跨模型仿真。
　　**涉及文件**：`genesim/src/scheduler/gene_sim_scheduler.py:471-632,1818-1868`；`genesim/src/ir/model_ir.py:2599-2619`；`flagos-pim-compiler/scripts/run_full_pipeline.py:190-242,686-729`；`flagos-pim-compiler/scripts/export_pp_placement.py:136-146`；`flagos-pim-compiler/genesim_bridge/placement_export.py:679-721`。
　　**修改思路**：定义规范化模型身份与结构指纹，覆盖模型类型、层数、hidden、词表、头数、专家几何、执行 dtype/语义以及完整算子编号/类型/形状/角色，由 parser 写入，精化原样保留，sidecar 回传，GeneSim 在生成 trace 前强制比对；`--model-dir` 应形成非 unknown 的可追溯身份，但不能依赖可移动的绝对路径。A2 缓存键同时包含该指纹、精化档位、融合清单和成本算法版本。用“同算子序列、不同词表/hidden”反例验证拒绝，用同指纹精化 IR 验证合法缓存不会被误伤。

### 问题5（中）: 实施证据未再收敛，导致验收状态失实
　　**问题说明**：implement §1.3、§5-§9 仍写 8 GiB 待确认、P1-1 显著加速待确认及 `full30b-r2`“仍在运行”；本轮事实是 8 GiB 已为硬约束，1.148×/1.202× 已被接受，服务已为 `oom-kill`。design §4.7 仍给 16 GiB、§九仍称 8 GiB 待确认。实施 §8.4/§9.2 的 dense_sim 对照写 46.627 秒，本轮同入口实测为 76.052 秒，且动态代价问题使该对照不能用于加速结论。文档已经把 4 层/48 层、历史/本轮分列，这一点不再判为“切片冒充全模型”，但旧状态与未复现数字仍会使 §7.1 的“已修复”结论失真。
　　**根本原因**：实施文档没有随用户确认、后台任务终态和本轮真实产物更新同一份证据矩阵，前两轮问题7及 round02 问题7的文档收敛只覆盖了当时状态。相同根因使 Q5/Q7、K6-K8、F11/F12/F16、§8.2-§9 相互矛盾。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md:29-41,61-71,207-244,246-296,324-417`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md:223-239,347-393`；`flagos-pim-compiler/tests/README.md`。
　　**修改思路**：待上述代码与产物验证完成后统一更新证据矩阵：关闭已接受的 Q7，保留 48 层容量与本轮数值未覆盖，登记 OOM 为环境阻塞，8B 始终记 SKIP；分别列 4 层/48 层、数值/主链、dense/dynamic、历史/本轮的命令、产物哈希和容量字节。dense 对照必须以可重放入口重测并解释差异，不能沿用 46.627 或把“文档已写”当作代码已验收。
