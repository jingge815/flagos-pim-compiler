# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261111-round09
> 创建日期：2026-11-11
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`（当前 HEAD 为 `b2f07ca`）、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 2 项问题（1 高、1 中）：4 层双语义主链、48 层主链和容量均有真实产物证据，但 round05 的 absorbed 结构校验与 round07/round08 的豁免契约仍未覆盖同类路径，严格产物验收不能关闭。
　　验证前均已加载 `paths.json` 的 PyTorch 环境；当前源码隔离镜像、探针、缓存和产物均在 `/tmp/qwen3-review-round09.RGE6BP`，关键源码哈希已与当前工作区逐项核对一致。未运行 `-k llama2_7b` 或 CEVA，未触碰另一个 tune-loop 工作区，未重建 FlagTree。实际执行结果如下：
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q`（追加 `-p no:cacheprovider`）：**33 passed、1 skipped（470.63 秒）**；8B 无权重，SKIP 记为未覆盖。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`：默认 `dynamic_dispatch` 退出码 0，**88.483 s / 8.985 t/s**；显式 `--moe-execution dense_sim` 亦退出码 0，**91.012 s / 8.735 t/s**。同一命令追加 `--skip-simulation` 也退出码 0，A/BCD/A2 与覆盖审计完成。
- 独立结构化审计确认两种语义均为 3681 个同编号算子、1557 个 GEMM、1536 个专家 GEMM、3163 条有效 sidecar 路径/哈希、23 份真实 pim mlir，专家 0..127 全覆盖、tiles `[32,128,512]`；A/B 两路全部 MLIR 均为 8 GiB 和 64 字节对齐，512 个 `VECTOR_MUL` 均为 `pim.eltwise<mul>` 且系数为 `{Tq:768}`。trace 均为 `pimir=8858 / absorbed=2048`，无 template/unknown，必需项缺失为 0；dense 的总时长和有效 FLOPs 均高于 dynamic。
- `cd genesim && ./run.sh --test sim`：补齐隔离镜像中被 Git 忽略的 DRAMsim3 配置后，**45/45 个测试文件通过**。首次失败仅为该镜像缺配置，未计为代码失败。
- `python -m pytest tests/ -q -k "not llama2_7b"`：**1980 passed、13 skipped、52 deselected、0 failed（1486.76 秒）**。其中两个额外跳过是隔离镜像未复制被忽略的旧模型产物；补齐后 `tests/test_genesim_bridge.py` 整文件复跑为 **58 passed、2 skipped**，漏跑的结构用例单独核对为 **2 passed**，不把这些命令合并伪报成一次全量结果。
- 使用现有 FlagTree 安装执行 `triton-opt test/Dialect/TritonPIM/paged_attn_phases.mlir -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck ...`：通过。
- 未重复启动 48 层长任务。已核对 `full30b-pipeline-r8-qwen3` 主日志 `exit=0` 并只读审计其真实产物：44117 个同编号算子、37923 条 sidecar 路径/哈希、18432 个专家 GEMM、23 份 pim mlir 全为 8 GiB/64 字节；trace 来源 `{'pimir': 106230, 'absorbed': 24576}`，**total_time_s=566.863、tokens/s=1.402**。dpu0..3 为 **8,258,576,576 / 8,522,825,728 B**，dpu4..7 为 **7,791,833,024 / 8,522,825,728 B**，均未超 8 GiB 契约。round06 的 48 层数值日志为 **26 passed、1 skipped（4:47:34）**，作为历史数值证据保留，8B 仍未覆盖。
　　已核实且不因下述反例而否定的「不变式/契约 → 代码实现 → 测试或产物证据」为：
- 「128 专家/top-8/768 与完整执行图」→ `genesim/src/ir/model_ir.py::build_from_hf_config` → 本轮 4 层及既有 48 层 IR、参数字节与逐专家覆盖审计。
- 「执行语义源自编译配置，MoE 不做 dense 6144 近似」→ `runtime/exec_plan_gen.py::resolve_moe_execution`、parser、sidecar 与 `model_ir` 成本角色 → 两种 IR/sidecar/summary 同值，dense/dynamic 的总时长及 FLOPs 差异实际产生。
- 「真实原语与容量同源」→ `genesim_bridge/placement_export.py::_bpath_pimir`、`memory/mem_planner.py::default_site_capacity` → 正确乘法文本、`{Tq:768}`、全部哈希与 MLIR 属性，以及 48 层逐 DPU 所需/可用字节。
- 「大规模图修改不再逐次全量建索引」→ `ModelIR.batch_edit()` 及 TP/relay 批量调用 → 45/45 GeneSim 测试和 48 层日志中 30961 个 TP 算子约 19 秒展开为 130036 个分片。
- 「真实权重与独立 HF 稀疏前向逐 token 对齐」→ `make_static_moe_forward`、`restore_hf_moe_forward` → 本轮 4 层 33 passed/1 skipped 及 round06 历史 48 层 26 passed/1 skipped。P1-1 的 prefill 1.148×、decode 1.202×已按用户确认关闭，不作为本轮失败项。
　　反向探针则证实：合法原产物均能通过，但矛盾豁免、多消费者吸收和精化后缺失吸收编号仍会被部分或全部复核入口接受。下述问题均已按相反路径复核，不是把合法 host、模型边界或 BCD 阶段误判为失败。

## 二、修改建议
### 问题1（高）: 吸收校验漏查多消费者，导致非法融合放行
　　**问题说明**：design §十.4 要求 absorbed 由融合清单、主算子 pimir 与 IR 依赖边唯一证明；项目既有 `graph/fuse.py:31-37` 明确主算子只能有一个消费者，否则另一个读者取得的是激活后的值而不是其需要的激活前值。`scripts/run_full_pipeline.py:608-652` 与 `genesim/src/scheduler/gene_sim_scheduler.py:994-1031` 却只检查激活节点“只有一个生产者”，没有检查主 GEMM 是否只有该激活一个消费者。实测构造主 GEMM 同时连接 SILU 与另一设备 `VECTOR_ADD` 的 IR，并提供融合清单、主算子 pimir 和 `absorbed_ops=[SILU]` 后，编译器 `validate_sidecar_contract`、`verify_trace_provenance` 与直接 GeneSim **全部 ACCEPTED**。这会把本应拒绝的融合从必需 pimir/trace 集合中删除，破坏数值语义与成本口径。
　　同一校验还没有区分产物阶段：从真实 4 层精化产物中删除全部 512 个 `absorbed_ops`、保留 512 条 `fused_activations` 后，编译器与直接 GeneSim 仍 **全部 ACCEPTED**。BCD 阶段先产清单、A2 后补吸收编号是合法流程，但精化后、直接 GeneSim 与 trace 验收仍只检查“absorbed → 清单”单方向，会使主算子折激活与独立激活 trace 可以同时进入代价链。round05 问题3只覆盖了伪类型、错角色和单边集合，属同一根因未完整修复。
　　**根本原因**：实现把“唯一生产者”误当成“主算子的唯一消费者”，并用同一个无阶段参数的布尔校验同时服务 BCD、精化后复核和直接仿真，既没有完整验证依赖图的消费者方向，也没有闭合融合清单与吸收集合在不同阶段的双向关系。该缺陷同时存在于导出、`skip-simulation`、freshness、仿真 IR、trace 与直接 GeneSim；现有 `tests/test_genesim_simulation.py:680-740` 只有单链正例和类型/角色反例，GeneSim 侧也缺多消费者与阶段反例，所以全量回归通过不能覆盖这些路径。
　　**涉及文件**：`flagos-pim-compiler/scripts/run_full_pipeline.py:579-662,740-833,1445-1527`；`flagos-pim-compiler/graph/fuse.py:31-37`；`flagos-pim-compiler/tests/test_genesim_simulation.py:680-740`；`genesim/src/scheduler/gene_sim_scheduler.py:965-1036,1125-1213`；`genesim/tests/sim/test_compiler_placement.py`；本轮 `probe-absorbed-consumers.log`、`probe-absorbed-phase.log`。
　　**修改思路**：在两仓吸收校验中同时建立生产者与消费者索引，按 `graph/fuse_pim` 的透明节点规则计算主算子的真实消费者，要求被吸收激活是该主算子唯一真实消费者，并继续核对激活类型、融合三元组、主算子 pimir/act_spec 与依赖边，不能只在当前 SILU 探针上加一个特判。给共享校验显式传入产物阶段：BCD 允许“清单已产、吸收编号未补”，A2 后、`skip-simulation`、freshness、仿真 IR、trace 和直接 GeneSim 则要求清单与吸收编号按源 op_id 双向闭合。增加主算子多消费者、精化后删除吸收编号、BCD 合法中间态及真实 4 层 512 条融合正例，并重跑直接仿真证明既不漏计也不重复计费。

### 问题2（中）: 豁免未核放置矛盾，导致严格契约绕过
　　**问题说明**：design §十.4 要求只有模型边界与具有导出落点证据的 host 算子可以豁免，豁免不得与实际 PIM 放置矛盾；round07 问题2及 round08 问题4又要求导出、`skip-simulation`、freshness、仿真 IR、trace 与直接 GeneSim 使用同一完整契约。GeneSim 在 `gene_sim_scheduler.py:1151-1162` 会检查 sidecar 的 `operators` 放置条目并拒绝矛盾，编译器 `_exemption_is_legitimate`（`scripts/run_full_pipeline.py:446-471`）却只比较 IR 与豁免条目，`_invalid_exemptions`（665-681）完全不看同一算子的 PIM 分片或设备条目。
　　本轮以真实 4 层 host `MOE_ROUTE` op151 为反例，保留其合法 IR/豁免证据，仅在 sidecar 中增加 PIM shard、`device_hint=pim` 及一个哈希有效的 pimir 条目：原真实产物通过全部检查，而变异产物仍被 `validate_sidecar_contract`、`verify_sidecar_freshness`、`verify_sim_ir_matches_sidecar`、`verify_trace_provenance` **全部 ACCEPTED**；同一产物在直接 GeneSim 预检中被明确 **REJECTED**。因此 round08 新增的共享验证器并未与 GeneSim 使用同一规则，`--skip-simulation` 仍可给矛盾产物签发通过结论，独立 freshness/trace 复核也可放行。
　　**根本原因**：round07 的“豁免与放置矛盾”检查只落在 GeneSim，round08 将编译器各入口收拢到 `validate_sidecar_contract` 时没有移植这条不变式，导致“统一入口”统一的是不完整子集。相同根因影响 host 非 GEMM、host GEMM 和模型边界的逐条目证据核对，并贯穿编译器全部五个复核入口；现有反例只覆盖伪造 host 字符串，没有覆盖“IR/豁免确为 host，但 sidecar 同时钉到 PIM”的相反假设，因此测试未触发。
　　**涉及文件**：`flagos-pim-compiler/scripts/run_full_pipeline.py:446-471,531-542,665-681,740-833,1368-1437,1517-1527`；`genesim/src/scheduler/gene_sim_scheduler.py:933-959,1125-1213`；`flagos-pim-compiler/tests/test_genesim_simulation.py:224-299,1146-1278`；`genesim/tests/sim/test_compiler_placement.py:2682-2703`；本轮 `probe-exemption-contradiction.log`、`probe-direct-exemption.log`。
　　**修改思路**：把“源 IR 落点、豁免条目、sidecar 实际放置”组成一个逐 op_id 的闭合不变式，在编译器共享验证器中执行与 GeneSim 等价的放置交叉检查：host/边界豁免若同时存在 PIM shard、`device_hint=pim` 或其它设备执行证据必须失败；host GEMM 还需与 `host_gemms` 的角色、权重身份一致。五个生产复核入口全部调用该规则，直接 GeneSim 继续独立复核。为每个入口增加矛盾放置反例，同时保留真实 host、模型边界、合法融合与明确旧式非严格产物正例，避免用全局禁止豁免或只修 op151 的方式误伤合法路径。
