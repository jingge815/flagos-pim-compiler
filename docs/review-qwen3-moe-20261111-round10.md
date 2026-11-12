# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261111-round10
> 创建日期：2026-11-11
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`（当前 HEAD 为 `b2f07ca`）、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 2 项问题（1 严重、1 中）：4 层双语义全链、skip-simulation、48 层真实产物与容量均有可复核证据，round09 的吸收和豁免问题已收敛；但动态分派的专家输出通信仍按全量计费，且实施文档誊错了 dynamic 有效 FLOPs，代价正确性与验收证据尚不能全部关闭。
　　验证前均加载 `paths.json` 的 PyTorch 环境；临时脚本、隔离 GeneSim、空算子缓存、日志和产物均在 `/tmp/qwen3-review-round10`。未运行 `-k llama2_7b` 或 CEVA，未触碰另一个 tune-loop 工作区，未重建 FlagTree，也未重复启动 48 层长任务。实际执行结果如下：
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q`（追加 `-p no:cacheprovider`）：**33 passed、1 skipped（291.73 秒）**；真实 4 层权重对拍通过，8B 无权重 SKIP，记为未覆盖。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`：默认 `dynamic_dispatch` 退出码 0，**88.483 s / 8.985 t/s**；显式 `--moe-execution dense_sim` 亦退出码 0，**91.012 s / 8.735 t/s**。追加 `--skip-simulation` 同样退出码 0，A/BCD/A2 与严格覆盖审计完成。两次全链均使用独立空算子缓存，A2 均实际完成。
- 独立结构化审计确认两种语义均为 3681 个同编号算子、1557 个 GEMM、1536 个专家 GEMM、3163 条有效 sidecar 路径/哈希、23 份真实 pim mlir，专家 0..127 全覆盖、tiles `[32,128,512]`；A/B 两路 MLIR 均为 8 GiB/64 字节，512 个 `VECTOR_MUL` 均为 `pim.eltwise<mul>` 且系数为 `{Tq:768}`。trace 均为 `pimir=8858 / absorbed=2048`，无 template/unknown，必需项缺失 0，融合与吸收各 512 条双向闭合；dense 的总时长和有效 FLOPs 均高于 dynamic。
- `cd genesim && ./run.sh --test sim`：**45/45 个测试文件通过**。`python -m pytest tests/ -q -k "not llama2_7b"`（追加 `-p no:cacheprovider`）：**1999 passed、11 skipped、52 deselected、0 failed（1319.66 秒）**；2 条 warning 为未选中专家溢出用例的预期数值告警。
- 使用安装内实际的 `build/flagtree-cmake/bin/triton-opt`：`paged_attn_phases.mlir` 与 `expand_phases.mlir` 的展开、契约验证及 FileCheck 均通过；`operator_ops_negative.mlir`、`phase_spec_negative.mlir`、`lower_to_emitc_ops_negative.mlir` 的 verify-diagnostics 也全部通过。
- 未重复启动 48 层长任务。round06 主链日志末尾已确认为 `Interrupted`，不能作为完成证据；round08 主链日志为 `exit=0`。当前代码重新执行完整 sidecar 校验、freshness、仿真 IR 与 trace 验收均通过：44117 个同编号算子、37923 条 sidecar 路径/哈希、18432 个专家 GEMM、23 份 pim mlir 全为 8 GiB/64 字节，6144 条融合与吸收闭合，trace 来源 `{'pimir': 106230, 'absorbed': 24576}`，**total_time_s=566.863、tokens/s=1.402**。dpu0..3 为 **8,258,576,576 / 8,522,825,728 B**，dpu4..7 为 **7,791,833,024 / 8,522,825,728 B**；GeneSim 最大 Cluster 常驻 **8,412,147,712 / 8,589,934,592 B**，均未超容量契约。round06 的 48 层数值日志 **26 passed、1 skipped（4:47:34）**只作历史数值证据，8B 仍未覆盖。
　　已核实且不因下述问题而否定的「不变式/契约 → 代码实现 → 测试或产物证据」为：
- 「真实配置与逐段产物同源」→ `build_from_hf_config`、注册表驱动的 parser/图编译/FlagTree/sidecar/GeneSim → 本轮 4 层及既有 48 层 IR、专家身份、哈希、真实分块和 trace 审计。
- 「8 DPU × 8 GiB 统一容量」→ `default_site_capacity`、sidecar `hw_budget` 与逐份 MLIR 属性 → 4/48 层容量实测及当前校验器通过，无 4 GiB、16 GiB 验收入口。
- 「真实权重与独立 HF 稀疏前向逐 token 对齐」→ `make_static_moe_forward`、`restore_hf_moe_forward` → 本轮 4 层 33 passed/1 skipped 及 round06 历史 48 层 26 passed/1 skipped。P1-1 的 prefill 1.148×、decode 1.202×已按用户确认关闭，不作为本轮失败项。
- 「round09 吸收与豁免闭合」→ 双仓消费者方向检查、阶段化吸收闭合、豁免与放置/host_gemms 交叉核对 → 真实产物接受、矛盾产物拒绝，45/45 与 1999 条回归通过；合法 BCD 中间态、host 和模型边界未被误伤。
- 「大规模图修改不逐次全量建索引」→ `ModelIR.batch_edit()` 及 TP/relay 批量调用 → 本轮 4 层约 1 秒展开，48 层日志约 19 秒展开，未再出现 O(N²) 的逐条重建。

## 二、修改建议
### 问题1（严重）: 专家输出通信未分派，导致动态代价虚高
　　**问题说明**：design §十.6 要求 `dynamic_dispatch` 的专家计算与通信都按 any-hit `1-(1-k/E)^Tq` 折算，只有共享累加始终保持全量。真实 IR 在 `genesim/src/ir/model_ir.py:2561-2614` 构造的是 `down_proj(expert_work) → MOE_ACCUM(shared_reduce)`；`_moe_dependency_fraction` 却只接收目标算子并读取目标角色（`genesim/src/scheduler/gene_sim_scheduler.py:340-349`），于是这条边被当成共享工作按 1.0 计费。该函数同时用于放置期通信估算（4288、4298 行）和正式管线（7106-7107 行），不是未使用的辅助逻辑。
　　本轮用 GeneSim 实际离散事件管线验证：E=8、top-1、decode 的 16 B 专家输出到共享主机累加边，dynamic 实计 **16 B**，按命中并集应为 **2 B**；相反路径中，dense 的同类边为 16 B、shared_reduce→shared_reduce 为 16 B、普通输入到 expert_work 为 2 B，均符合契约，说明不能一刀切缩放所有边。真实 4/48 层 IR 分别有 **512/6144 条**这种 PIM→CPU 边；按 top-8/128 的 decode 与源 IR 边系数计算，每个 decode 分别多计 **1,966,080 B / 23,592,960 B**。真实执行器对未命中专家的 host 收集命令直接返回主机零张量而不执行收集（`runtime/exec_plan_gen.py:551-564`、`backend/hal_numpy.py:242-256,336-351`），不存在“仍需把清零输出完整读回主机”的相反情况，故当前全量计费确属偏差。implement §1.3、F19 与 §8.3 所称动态通信语义已与执行一致，不能据此关闭验收。
　　**根本原因**：round03 引入成本角色时把“算子是否被门控”与“边所传输张量由谁生产”混为一体，实现误以目标角色决定整条边的命中比例；round04 又把它统一成所有依赖表示共用的出口，但没有改变方向假设。现有 `test_legacy_dependency_payload_is_scaled_by_target_role` 只构造“普通源 → 专家/共享目标”（`genesim/tests/sim/test_pipeline_execution.py:1970-2026`），没有真实“专家源 → 共享主机目标”，因此全量回归通过也未覆盖。相同根因还影响 TP 展开后的专家归约到 MOE_ACCUM、专家到 HOST 的出口，以及显式系数和旧式依赖两条路径。
　　**涉及文件**：`genesim/src/ir/model_ir.py:2561-2614`；`genesim/src/scheduler/gene_sim_scheduler.py:340-349,4275-4299,7044-7114`；`genesim/tests/sim/test_pipeline_execution.py:1970-2026`；`genesim/tests/sim/test_moe_execution_cost.py`；`flagos-pim-compiler/runtime/exec_plan_gen.py:508-575`；`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md:141,294,379,784`；本轮 `/tmp/qwen3-review-round10/probe-moe-edge.log`。
　　**修改思路**：把依赖折算统一为同时接收源、目的算子的边语义判定：专家生产的输出无论流向共享累加、另一设备还是 HOST，都随该专家的命中并集折算；HOST/共享输入到专家按消费者命中折算；shared_reduce→shared_reduce 与 dense_sim 保持全量。放置估算与正式管线、显式/旧式依赖共用这一判定，不在 down_proj 或 MOE_ACCUM 上加类型特判。补齐真实图及 TP 展开后的专家源→共享目标、专家源→HOST、共享源→共享目标、普通源→专家目标的 prefill/decode 正反向用例，并断言未命中 host 收集没有跨设备传输。修复后先用 4 层逐项核对 trace 元数据、算子成本、通信字节和 dense/dynamic summary，再按内存与 systemd 约束重跑 48 层，更新实施证据。
### 问题2（中）: 动态FLOPs誊写错误，导致验收证据失真
　　**问题说明**：implement §1.2、§4.1 与 §8.4 要求“全链产物可追溯”和逐位一致的验收证据。round09 原始 `/tmp/round09-audit-dynamic.json` 与本轮重新生成的 `/tmp/qwen3-review-round10/audit-dynamic.json` 都记录 dynamic `total_useful_flops=5,415,241,254,987.654`；实施文档却在 **349 行**写成 `5,414,125,498,987.654`，并在 **848-853 行**再次声称该值与 round08 逐位相等。文档值比两份真实产物少 **1,115,756,000**，据此计算的 round03 差值也不再成立。dense 的 `5,826,198,177,867.654`、两种语义的总时长、trace 分布及 fingerprint 均与本轮产物一致，故这不是代码回归或对历史结果的不同解释，而是 dynamic 证据誊写失实。
　　**根本原因**：实施证据矩阵中的数值由人工誊写，未与引用的审计 JSON 做机器一致性校验，错误值又复制到测试总览、真实 30B 证据和性能说明三处。round02 问题7、round03 问题5曾要求证据随产物收敛，本轮说明该根因仍存在于具体数值字段；反向核对表明 dense、时间、trace 与容量字段没有同类偏差，不扩大为整份实施文档不可信。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md:337-354,831-853`；`/tmp/round09-audit-dynamic.json`；本轮 `/tmp/qwen3-review-round10/audit-dynamic.json`、`/tmp/qwen3-review-round10/audit-dense.json`、`/tmp/qwen3-review-round10/audit-doc-flops.log`。
　　**修改思路**：以 summary/结构化审计 JSON 为唯一数值来源生成或回填实施文档中的 dynamic/dense 指标，增加文档记录值与产物的自动一致性检查，同时重算依赖这些值的差值和结论，不能只改 349 行一处。保留 4 层、48 层及历史数值的分栏口径；问题1修复并重跑后，再从新产物统一刷新相关通信与性能证据。
