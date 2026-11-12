# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261109-round02
> 创建日期：2026-11-09
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 7 项问题（2 严重、4 高、1 中）：4 层真实权重切片已经真实走通图编译、FlagTree、sidecar 与 GeneSim，但 48 层主链容量、默认动态分派的仿真语义、非 GEMM 原语与成本仍未满足本轮验收，不能据切片通过而宣称全需求完成。
　　所有验证均在 `/tmp/qwen3-review-round02.HLEDEa` 的当前工作区镜像或独立探针中执行，未使用 monkeypatch 修改正式入口，未重建 FlagTree，未运行外部 CEVA 仿真，也未重复启动约 7 小时的 48 层目标测试。实际命令与结果如下：
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q`：**23 passed、1 skipped（315.75 秒）**；未设置两个模型路径时 `tests/test_architectures.py` 另有 **22 passed**。既有 `/tmp/test_target_full_30b_systemd.log` 与 `test-results/pytest-latest.md` 证明同一目标命令在 48 层全模型上为 **23 passed、1 skipped（25426.83 秒）**，其中真实 30B 用例通过；8B 因本机无权重而 skip，记为未覆盖而非通过。
- `python scripts/run_full_pipeline.py --num-stages 4 --skip-simulation`：默认 Llama2-7B 回归通过；骨架与精化 IR 均为 3780 个同编号算子，sidecar 含 224 个 GEMM 与 3554 个算子级条目，路径和哈希有效，实测分块为 512。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`：退出码 0；骨架与精化 IR 均为 3536 个同编号算子，sidecar 含 1556 个 GEMM、1462 个算子级条目和 518 条显式豁免，1536 个专家 GEMM 覆盖专家 0..127，全部 `pimir_path`/`pimir_sha256` 有效；FlagTree 实测分块为 `[32,128,512]`，GeneSim 生成 10758 条 trace，来源全部为 `pimir`，`total_time_s=76.052`。
- 真实 48 层配置运行 `scripts/model_parser.py --execution_dtype float16`：生成 42388 个算子、18432 个专家 GEMM，每层专家三件套宽度为 768，专家权重驻留合计 54 GiB，证明 round01 的 dense 6144 误建已经修复。`./run.sh --test sim`：**44/44 个测试文件通过**。现有 FlagTree 安装运行 `paged_attn_phases.mlir` 的 `triton-opt ... | FileCheck`：通过。
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/ -q -k "not llama2_7b"`：在包含 `.git` 的完整隔离镜像中为 **1875 passed、9 skipped、52 deselected、4 warnings（1621.98 秒）**，失败 0。首轮镜像因漏复制 `.git` 产生 6 个版本测试失败，补齐后 `tests/test_ceva_export.py` 单独复跑为 **72 passed**，不作为代码缺陷。
- 反向探针实测：显式 `require_compiler_pimir=true` 在 sidecar 缺失和未自报两条路径上仍各生成 1 条模板 trace；fresh sidecar 缺哈希仍被导出步骤接受；未知 `BRAND_NEW_OP` 被自动写入豁免清单。真实 4 层产物中，`VECTOR_MUL` 的 sidecar 与 trace 共同指向含 `pim.eltwise<add>` 的同一文件；其精化 FLOPs 在 Tq=128 时为 262144，而按 768 宽度应为 98304。无并发负载的 5 次执行期 A/B 中，dense/dynamic 中位时间分别为 4.267/4.294 秒，加速比仅 0.994×。
　　已核实的正确契约链为：真实配置的专家几何 → `genesim/src/ir/model_ir.py` → 48 层 IR 与 4 层 sidecar；独立 HF 稀疏参考 → `restore_hf_moe_forward` → 4 层实测及 48 层既有目标日志；MoE 默认关闭图级分块 → `runtime/compile.py` → 正式 4 层全链路无临时补丁通过；`config.head_dim=128`（而非 `hidden/heads=64`）→ 编译器配置读取 → 48 层数值对拍。动态子图、通信门控与专家并行聚焦回归另有 **43 passed**。下述问题均按相反路径复核，合法的 dense_sim、纯放置导出、host/边界豁免与稠密模型路径不应被一刀切误判。

## 二、修改建议
### 问题1（严重）: 全模型主链容量不可行，导致48层链路未交付
　　**问题说明**：request §2.1 P0-3 要求 8 DPU、每 DPU 8 GiB 的容量判据，本轮又要求 48 层 Qwen3-30B-A3B 完整走通图编译到 GeneSim；design §4.7、§七与 implement §8 要求真实全模型证据。`scripts/export_pp_placement.py:146-153` 却固定为 **4 GiB/DPU**，真实数值用例在 `tests/test_qwen3_30b_real.py:40-48` 使用 **16 GiB**，GeneSim 又按 design §十.5 使用 **8 GiB ClusterPU**。按真实配置计算，专家权重为 `48×128×3×2048×768×2=54 GiB`，tp4×pp2 的 8 DPU 上每台仅专家分片就需 **6.75 GiB**，已超过主链 4 GiB 预算 2.75 GiB，尚未计注意力、嵌入、KV 与激活。因此正式 48 层导出在原理上不可能通过其容量判据；本轮未重复启动注定超预算的全模型导出，4 层通过和 48 层 16 GiB 数值对拍都不能替代同口径的 48 层主链验收。
　　**根本原因**：架构注册表只统一了加载器、宽度、角色和产物名，没有统一硬件容量契约；主链、真实测试和 GeneSim 各自硬编预算，实施又用 4 层切片替代全模型主链验证。round01 问题 7 的 8/16 GiB 分歧尚未关闭，本轮进一步暴露出第三条 4 GiB 入口，同一根因影响所有 PP/TP 策略、prefill/decode 和全模型 sidecar 产物。
　　**涉及文件**：`flagos-pim-compiler/scripts/export_pp_placement.py:146-153`；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:34-53`；`flagos-pim-compiler/docs/request-qwen3-moe-20261105.md` §2.1 P0-3；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.7、§十.5；`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §5.2、§六 Q5、§8。
　　**修改思路**：先确认 8 GiB 是否仍为硬约束，再把同一份容量配置贯穿架构描述、图编译、真实测试和 GeneSim，禁止各入口独立改数。若保留 8 GiB，应按专家与非专家权重、KV、激活和内核分块反推序列/窗口并验证超预算错误；若调整硬件目标，须同步修订需求、设计和仿真配置。最后用同一容量口径实测 48 层导出、sidecar 与 GeneSim 仿真，并核对三区无重叠及每 DPU 所需/可用字节，不能只跑 4 层或只修改文档。

### 问题2（严重）: 仿真语义脱离动态分派，导致MoE代价失真
　　**问题说明**：本轮补充需求要求 GeneSim 的专家几何、驻留和通信与所选执行语义一致；design §十.1、§十.6 同时声明 `dense_sim` 与 `dynamic_dispatch` 两种契约。编译器在 `runtime/exec_plan_gen.py:25-26` 默认 `PIM_MOE_DISPATCH=1`，即默认只执行命中专家；GeneSim 却在 `genesim/src/ir/model_ir.py:1212` 无条件写入 `dense_sim`，并在 2096-2158 行为每层生成全部 128 个专家及完整累加链，没有动态分派入口。`genesim_bridge/placement_export.py:631-637` 只是照抄该标签。实际 4 层正式命令未设置 `PIM_MOE_DISPATCH=0`，其 sidecar 与 summary 仍均为 `dense_sim`，所以 decode 按 128 专家计费而编译器默认只命中 top-8；专家计算量相差 16 倍，通信门控也没有进入仿真。
　　**根本原因**：`moe_execution` 只是 IR 自描述字符串，不是从真实编译执行配置派生的共享契约；图构造只实现全专家一种语义，校验再比较两个同源字符串也只能证明“自洽”，不能证明与编译器一致。现有 GeneSim 测试只断言 `dense_sim`，`dynamic_dispatch` 虽被枚举接受却没有图生成、调度或成本实现，summary、sidecar 和 trace 因而共同固化错误假设。
　　**涉及文件**：`flagos-pim-compiler/runtime/exec_plan_gen.py:25-26`；`flagos-pim-compiler/genesim_bridge/placement_export.py:629-637`；`flagos-pim-compiler/scripts/run_full_pipeline.py:833-880`；`genesim/src/ir/model_ir.py:1191-1214,2096-2275`；`genesim/scripts/model_parser.py:84-92,173-176`；`genesim/tests/sim/test_model_ir.py:1219-1248`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §十。
　　**修改思路**：把执行语义设为正式编译配置，由同一入口传给模型解析、执行计划和 sidecar，缺省值也必须一致；GeneSim 加载时核对真实模式，不能只比较 IR 与 sidecar 的自抄标签。dense_sim 路径保留全专家计算，dynamic_dispatch 路径则按每个请求的命中专家执行 top-k 子图并跳过未命中专家的计算与通信，prefill 按实际命中并集、decode 按每 token top-k 计费。分别对两种模式做 IR—sidecar—summary—trace 验收，并断言默认模式一致；仅把字符串改成 dynamic 而保留全专家图不算修复。

### 问题3（高）: 产物契约仍可绕过，导致模板与错配混入
　　**问题说明**：本轮要求严格消费真实 pim mlir；design §十.2-4 要求模型身份、算子编号、路径、哈希和显式豁免完整，缺项必须失败。当前仍有四条绕过路径：其一，`genesim/src/scheduler/gene_sim_scheduler.py:2120` 只在 `_pimir_contract` 为真时预检，显式 `require_compiler_pimir=true` 在 sidecar 缺失或未自报时仍生成模板，两个探针各得到 1 条 `trace_source=template`；其二，`scripts/run_full_pipeline.py:415-421` 对 fresh sidecar 缺哈希仍跳过核对，探针证实 `--skip-simulation` 所依赖的导出步骤成功返回；其三，`genesim_bridge/placement_export.py:784-803` 把一切无路径的非 GEMM 自动豁免，未知 `BRAND_NEW_OP` 也得到“无原语”理由，随后可按 `gene_sim_scheduler.py:378-410` 落 GPU，绕开只检查 PIM 算子的预检；其四，`_validate_moe_config` 在 532-534 行对缺少 `moe_config` 直接放行，MoE 身份并非硬契约。
　　**根本原因**：严格模式被拆成配置开关与 sidecar 自报两套状态，产物完整性又分散在导出脚本和调度器；同时把旧产物兼容、合法无产物与新增算子漏接混为一谈，没有一份闭合的失败优先 schema。round01 问题 4 只修复了“自报 requires_pimir 且条目存在”的主路径，因此在本轮换用直接 GeneSim、skip-simulation、未知算子和缺元数据路径后复发。
　　**涉及文件**：`genesim/src/scheduler/gene_sim_scheduler.py:450-455,530-552,682-708,2116-2188,2246-2257`；`flagos-pim-compiler/scripts/run_full_pipeline.py:378-425`；`flagos-pim-compiler/genesim_bridge/placement_export.py:773-803`；`genesim/tests/sim/test_compiler_placement.py:1887-1893,2515-2545`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §十.2-4。
　　**修改思路**：建立供导出脚本与 GeneSim 共用的产物校验契约：显式严格开关或 sidecar 自报任一成立时，sidecar 必须存在，模型/MoE 身份和 IR 编号必须匹配，每个设备算子必须有可读路径与正确哈希，并在生成任何 trace 前一次验完。fresh 导出不得保留“旧产物缺哈希”豁免；豁免仅允许明确登记的 host、边界、被吸收和归约类型，并校验实际落点，未知设备算子应在导出时失败而不是自动豁免后落 GPU。保留非严格的纯放置用法，分别补直接缺失、宽松 sidecar、fresh 缺哈希、未知算子、缺 MoE 元数据及合法 host/边界的正反向测试。

### 问题4（高）: 向量乘法被降为加法，导致原语产物错误
　　**问题说明**：本轮要求每个 IR 算子都有真实且正确的 GeneSim 原语落点，专家 SwiGLU 的中间乘法不能用其他算子替代。`genesim_bridge/placement_export.py:338-352` 的 `_bpath_pimir` 对所有 `eltwise` 一律调用 `eltwise_kernel(..., "add")`，丢失了 IR 的 `VECTOR_MUL` 语义。实际 4 层产物 op119 类型为 `VECTOR_MUL`，其 sidecar 文件却是 `pim.eltwise<add>`；展开后的 op14362 trace 为 `trace_source=pimir`，且 trace 哈希与该加法文件完全一致，证明错误产物确实被 GeneSim 消费，而不是死代码或测试夹具。
　　**根本原因**：B 路构造器只接收 mnemonic 和形状，没有把完整算子类型/语义传入 lowering，测试又只检查文件存在、哈希和 trace 来源，没有断言生成的操作种类。该根因影响所有模型的 `VECTOR_MUL`，包括 Llama/Qwen dense 的 SwiGLU 与本轮 4 层 512 个原始专家乘法及其 TP 展开；即使某个硬件版本对加乘给出相同周期，也不能把“来源为 pimir”误当成“乘法语义与代价正确”。
　　**涉及文件**：`flagos-pim-compiler/genesim_bridge/placement_export.py:301-390,425-459`；`flagos-pim-compiler/opcompiler_bridge/oplevel_kernel.py:349-376`；`flagos-pim-compiler/tests/test_placement_export.py:458-493`；本轮 4 层 sidecar op119 与 trace op14362。
　　**修改思路**：让 B 路 lowering 接收完整算子描述，按 `VECTOR_ADD`/`VECTOR_MUL` 等明确映射生成对应 kind，未知类型直接拒绝；导出时同时校验 pimir 的操作种类、形状和哈希，GeneSim 加载时核对该语义身份。增加真实 FlagTree 测试，分别断言加法和乘法文本、对应 sidecar 与最终 trace 消费同一正确产物，并覆盖 MoE 专家乘法及 dense 回归。

### 问题5（高）: 精化乘法沿用隐层宽度，导致FLOPs虚高
　　**问题说明**：design §4.2 明确真正专家宽度是 `moe_intermediate_size=768`，本轮要求精化成本与 MoE 几何一致。`genesim_bridge/cost_extractor.py:526-531` 仍设置 `ffn_dim=4×hidden_size`，`genesim_bridge/op_classify.py:394-400` 对 `VECTOR_MUL` 又固定使用 `hidden_size`，而 `_measure` 在 78-79 行不给非 GEMM 传实际算子形状。实际 4 层 op119 输入宽度为 768，Tq=128 时精化 sidecar 测得 **262144 FLOPs**，正确值应为 `128×768=98304`；拟合系数为 16，正确值应为 6，偏差 8/3 倍。该错误进入精化 IR、summary 的有用 FLOPs 以及按 IR roofline 计费的入口。
　　**根本原因**：成本配方使用模型级代表维度而非逐算子真实形状，把残差加法、专家中间乘法和不融合激活错误地共享同一宽度假设；缓存键虽含形状，生成配方时却已经丢失该信息。现有测试主要覆盖同宽小模型或被吸收 SILU，没有覆盖未吸收的专家 `VECTOR_MUL`。当前 512 个 SILU 被融合吸收，故本问题不把 SILU 计入本次最终偏差，但未融合路径上的同类宽度假设也应一并消除。
　　**涉及文件**：`flagos-pim-compiler/genesim_bridge/cost_extractor.py:45-85,526-531,578-607`；`flagos-pim-compiler/genesim_bridge/op_classify.py:197-205,394-422`；`flagos-pim-compiler/tests/test_ir_cost_eval.py`；本轮 4 层 `qwen3_moe_oplevel_extensions.json` op119。
　　**修改思路**：让配方工厂接收当前算子的类型、输入/输出形状及 MoE 元数据，逐算子生成测量 IR；残差加法使用 hidden，专家乘法和未融合激活使用实际中间宽度，不能从 `4×hidden` 或 dense 残留配置推导。为未吸收的 `VECTOR_MUL`、未融合 SILU 和 dense 残差分别断言测量值及拟合系数，并检查精化 IR 与 summary 的 FLOPs；与问题 4 联动验证正确操作和正确形状同时成立。

### 问题6（高）: 动态分派未见执行加速，导致P1验收未闭合
　　**问题说明**：request §2.1 P1-1 要求动态分派实测执行时间显著低于稠密模拟并记录加速比，design §七也以此验收。implement §9 给出的总墙钟仅为 **1.09×**，且明确包含 57 GB 权重装载、两次图编译和 HF 参考等固定开销；随后用命令跳过比例替代计时，并在 288-289 行承认剔除固定成本的纯墙钟微基准仍是后续课题。`tests/test_moe_dispatch.py:443-484` 只按同一份计划和跳过集统计命令比例，没有测量实际执行时间。本轮在同一小模型、同一权重和相同请求上补做执行期 A/B，编译与装载均不计时；在无其他测试负载的复测中，5 次重复的中位数为 dense **4.267 秒**、dynamic **4.294 秒**，加速比 **0.994×**，token 完全一致。差异已处于噪声量级，且动态分派没有任何加速，不能支持“显著加速”，implement §1.3 的 P1-1 完成结论不成立。
　　**根本原因**：当前实现仍在静态计划中提交全部专家命令，跳过 launch 时还要逐块写零（`backend/hal_numpy.py:318-328`），多专家累加链又被 `_MULTI_EXPERT` 排除在门控之外（`runtime/exec_plan_gen.py:54-88`）；小模型节省的微小 GEMM 不足以抵消命令调度、写零和主机累加开销。实施再把需求规定的执行期时间替换成命令数量代理，没有仓库内隔离基准暴露这一问题；总流程固定成本又进一步掩盖了 decode 无收益的事实。
　　**涉及文件**：`flagos-pim-compiler/runtime/exec_plan_gen.py:54-106,703-709`；`flagos-pim-compiler/backend/hal_numpy.py:313-330`；`flagos-pim-compiler/docs/request-qwen3-moe-20261105.md` §2.1 P1-1、§五；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.8、§七；`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §1.3、§九；`flagos-pim-compiler/tests/test_moe_dispatch.py:443-484`；`/tmp/qwen3-review-round02.HLEDEa/probe_dispatch_perf.py`。
　　**修改思路**：先提供仓库内基准入口，在同一已编译计划、已装载后端和相同请求上分别运行 P0/P1，只测执行阶段并做足够重复，记录 decode/prefill 时间、加速比和波动，同时断言 token 一致及实际命中专家数。再在保持数值语义的前提下收敛为真正按命中专家执行的子计划与聚合，消除未命中命令提交、无效写零和全专家主机累加，而不是只给现有命令加跳过标签。命令比例可保留为诊断，但不能替代时间验收；若优化后仍无显著收益，应保留未完成或回到需求确认，而不是宣称 P1 完成。

### 问题7（中）: 实施证据未及时收敛，导致完成状态失实
　　**问题说明**：本轮要求 implement §1.3、§3、§4、§5 就地补齐新增范围与真实证据。当前 §1.3 宣称 P0/P1 全部完成，§4.1 仍写“全模型用例完成后回填”，§8.4 的 48 层仍为“待最终回填”，§六 Q6 也未关闭；但 `/tmp/test_target_full_30b_systemd.log` 和 `test-results/pytest-latest.md` 已记录 48 层目标测试 **23 passed、1 skipped**，8B 仍是未覆盖。与此同时，§1.3 的完成结论与本文前述 48 层主链容量、动态成本及 P1 计时缺口相矛盾。
　　**根本原因**：实施文档在长任务结束前写入完成状态，后续外部 systemd 日志没有映射回“契约→实现→测试/产物”矩阵；4 层切片、48 层数值对拍和 48 层图编译—仿真被混成同一验收层级。仅更新结论文本而不补真实入口与产物，会再次形成 round01 问题 10 的同类复发。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §1.3、§4.1、§4.3、§5、§六 Q5-Q6、§8.4、§九；`/tmp/test_target_full_30b_systemd.log`；`flagos-pim-compiler/test-results/pytest-latest.md`。
　　**修改思路**：先完成上述代码与验证修复，再按同一份证据矩阵更新实施文档：分别登记 4 层全链路、48 层数值对拍、48 层主链及 8B 未覆盖，附实际命令、退出码、日志位置、产物编号与哈希、容量口径和执行语义；保留未解决项，不得把 skip 或切片写成全模型通过。文档更新必须对应可重放入口与真实产物，不能只把“待回填”改成“完成”。
