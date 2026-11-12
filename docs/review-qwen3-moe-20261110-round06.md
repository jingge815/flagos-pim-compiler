# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261110-round06
> 创建日期：2026-11-10
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`、`FlagTree 262adb3b7..6e34f0299` 及三仓当前工作区改动，行号按当前工作区定位。共发现 5 项问题（1 严重、3 高、1 中）：4 层真实权重、双语义主链和常规回归均已实测通过，48 层数值与 BCD 也已通过，但 top-k 极端数值语义、B 路容量对齐、专家身份审计仍有实质偏差，48 层 GeneSim 仿真尚未形成终态，不能关闭全模型验收。
　　验证前已加载 `paths.json` 的 PyTorch 环境；测试使用 `/tmp/qwen3-review-round06` 下包含当前未提交代码的隔离镜像，探针与产物均在 `/tmp`，未运行 `-k llama2_7b` 或 CEVA，也未触碰另一个 tune-loop 工作区。实际执行结果如下：
- `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider`：**26 passed、1 skipped（246.56 秒）**；8B 无权重，SKIP 记为未覆盖。
- `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`：默认 dynamic_dispatch 退出码 0，**88.483 s / 8.985 t/s**；追加 `--moe-execution dense_sim` 亦退出码 0，**91.012 s / 8.735 t/s**。两者均为 3681 个同编号算子、1557 个 GEMM、1536 个专家 GEMM、3163 条有效路径/哈希、专家 0..127 全覆盖、tiles `[32,128,512]`；trace 均为 `pimir=8858 / absorbed=2048`，无 template/unknown，输出头有 4 条分片 trace。
- `cd genesim && ./run.sh --test sim`（隔离镜像）：**45/45 个测试文件通过**。`python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider`：**1926 passed、10 skipped、52 deselected、0 failed（1643.97 秒）**。
- 使用现有 FlagTree 安装执行 `triton-opt test/Dialect/TritonPIM/paged_attn_phases.mlir -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck ...`：通过，未重建源码树。
- 48 层真实权重目标命令由 `full30b-r6-qwen3.service` 执行，结果为 **26 passed、1 skipped（17254.56 秒，4:47:34）**，30B 数值对拍通过、8B 未覆盖，systemd 峰值内存 63.1 GiB。`full30b-pipeline-r6-qwen3.service` 已完成骨架、BCD 与精化：44117 个同编号算子、18673 个 GEMM、18432 个专家 GEMM、37923 条有效路径/哈希，sidecar 为 8 GiB / 64 MiB / 64 字节，精化指纹为 `cf8e71b6…`；GeneSim 仿真仍在运行，summary 尚未生成，记为待验证。
　　已核实的「不变式/契约 → 代码实现 → 测试或产物证据」为：128 专家/top-8/768 → `model_ir.build_from_hf_config` → 4/48 层 IR；编译执行语义 → `resolve_moe_execution`、parser 与 sidecar → 两种 IR/sidecar/summary 同值；乘法与专家宽度 → `_bpath_pimir`、`_element_width` → 512 份 `pim.eltwise<mul>` 及 `{Tq:768}`；源 IR 覆盖与独立 HF 参考 → 覆盖/trace 校验、`restore_hf_moe_forward` → 必需 trace 无缺失及 4 层逐 token 对拍通过。上述证据不覆盖下述偏差。

## 二、修改建议
### 问题1（严重）: 零权重掩码偏离top-k，导致数值分派失真
　　**问题说明**：request §2.1 P0-1/P0-4 要求静态前向与 HF 严格等价，P1-1 要求只执行 top-k 命中专家。HF 用 `selected_experts` 的 one-hot 决定专家是否执行，而 `backend/architectures.py:34-48` 用 `routing[:,e] != 0` 代替选中状态，`runtime/exec_plan_gen.py:152-157` 也用非零路由权重计算动态命中集。top-k 选中但 softmax 权重下溢为零时，两者不再等价。实测 3 专家 top-2 配置中，选中专家为 `[0,1]`、归一化权重为 `[1,0]`，1 号专家产生非有限输出：HF 原始前向有 64 个 NaN，当前静态前向却全部掩成 0。有限输出时该差异可能被零权重掩盖，但严格 HF 数值语义及 top-k 执行语义均已改变。
　　**根本原因**：round01 的溢出修复把“top-k 选中”与“路由权重非零”混为一个条件，只覆盖了未选中专家溢出的单一路径，没有覆盖选中权重下溢；dispatch 复用了同一错误假设，GeneSim 的 `1-(1-k/E)^Tq` 又按每 token 恰有 k 个命中专家推导，因此同一根因还影响 P0/P1、prefill/decode 与动态成本。反向复核确认：未选中专家仍应掩零，不能为匹配 HF 的非有限结果而取消全部防护；真正需要区分的是选中索引和权重值。
　　**涉及文件**：`flagos-pim-compiler/backend/architectures.py:30-48`；`flagos-pim-compiler/runtime/exec_plan_gen.py:146-157`；`genesim/src/scheduler/gene_sim_scheduler.py:348-367`；`flagos-pim-compiler/tests/test_architectures.py:289-347`；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.1、§4.8、§十.6。
　　**修改思路**：由 `selected_experts` 生成独立的布尔选中掩码（可用 one-hot 或 scatter 常量 1），选择/跳过一次都使用该掩码，路由权重只参与加权，不再兼任命中条件；同步修正 design §十.6 的非零权重假设及 GeneSim 命中口径。增加选中权重下溢且选中专家产生 inf/nan、未选中专家产生 inf/nan、`norm_topk_prob` 两态、P0/P1、prefill/decode 的回归，并核对实际命中专家集合与成本公式，避免只修当前探针。

### 问题2（高）: 容量对齐仍然分裂，导致硬契约失效
　　**问题说明**：request §2.1 P0-3 与用户确认的验收口径要求 8 DPU × 8 GiB，design §十.5 要求总容量、64 MiB 预留、64 字节对齐三件套不可拆分。`contracts/op_contract.py:58-68` 却令 `DPU_DMA_ALIGN=64`、`DEFAULT_HARDWARE_CONFIG.dma_align=8`；`genesim_bridge/paths.py:35-41` 因而给 B 路默认传入 8。实测本轮 sidecar 自报 `align=64`，其中 **1557 条 GEMM 的 pim mlir 为 64，但 1606 条 B 路 pim mlir 全部为 8**，dense/dynamic 两份产物均如此。另一入口 `backend/vllm_worker.py:234-241` 的 `HwBudget.align=256`，同一函数硬件配置又为 64。即主链产物内部及生产入口都没有统一到同一契约。
　　**根本原因**：容量总字节与系统预留已收敛到工厂，但硬件配置、B 路 lowering 和 vLLM 仍各自构造对齐参数；`_attach_bpath_pimir`/`_run_oplevel_triton_opt` 没有接收本次导出的硬件配置，校验又只检查 sidecar 顶层三字段，不检查真实 MLIR 属性。round05 的 F30 因而只修复了自报字段，未覆盖产物生成与 vLLM 同类路径。部分 B 路算子可能不直接受对齐影响，但这不能使“自报 64、实产 8”成为合法状态，也不能证明相关分块/代价无偏差。
　　**涉及文件**：`flagos-pim-compiler/contracts/op_contract.py:55-69`；`flagos-pim-compiler/genesim_bridge/paths.py:34-42`；`flagos-pim-compiler/genesim_bridge/placement_export.py:430-440,475-509,693-707,941-958`；`flagos-pim-compiler/opcompiler_bridge/driver.py:837-869`；`flagos-pim-compiler/genesim_bridge/cost_extractor.py:181-189`；`flagos-pim-compiler/backend/vllm_worker.py:228-245`；本轮 `audit4l-dynamic.json` 与 `audit4l-dense.json` 的 `align_counts`。
　　**修改思路**：用一个工厂从同一容量三件套派生 `HwBudget` 与 `PIMHardwareConfig`，默认硬件、`pim_options`、vLLM、A/B 两路及成本精化均从该工厂取值；将本次硬件配置显式传入 B 路生成，禁止回退到另一默认值。导出和 GeneSim 加载除核 sidecar 字段外，还应核对每份 pim mlir 的容量/对齐属性，编译入口同时拒绝 `hw.align != hardware.dma_align`。测试须覆盖真实 A/B 产物、默认参数、vLLM 与直接 GeneSim，而不是只断言顶层 JSON 数值。

### 问题3（高）: 专家身份仅核对集合，导致错专家放行
　　**问题说明**：design §十.2/§十.4 要求按 `(层号, 投影身份, 专家号)` 唯一定位权重，直接 GeneSim、skip-simulation、仿真前复核与 trace 验收遵守同一规则。`scripts/run_full_pipeline.py:661-678` 却把各条目三元组收集为集合再求差，丢掉了它们与 `op_id` 的对应。实测将真实 4 层 sidecar 中同层同角色的 op152（expert0）与 op158（expert1）的专家号互换，`audit_sidecar_coverage`、`verify_sidecar_freshness`、`verify_sim_ir_matches_sidecar`、`verify_trace_provenance` 全部接受；直接 GeneSim 的 `_validate_moe_expert_identity` 正确拒绝这两条错配。合法的原产物逐项身份无误，因此该探针不是把合法路径误判为异常。
　　**根本原因**：round05 只在 GeneSim 增加逐 `op_id` 身份校验，没有收敛编译器侧的集合比较；freshness、skip-simulation、仿真前和 trace 又各自复用这份不完整审计，导致“缺一个专家组合”能被抓住、“两个专家互换”却绕过。相同根因会使后续按错误专家元数据追溯权重、分片和动态成本，集合相等不能证明产物同源。
　　**涉及文件**：`flagos-pim-compiler/scripts/run_full_pipeline.py:553-573,576-694,1027-1102,1105-1194,1335-1344`；`genesim/src/scheduler/gene_sim_scheduler.py:1030-1064`；`flagos-pim-compiler/tests/test_genesim_simulation.py`；`genesim/tests/sim/test_compiler_placement.py`。
　　**修改思路**：定义双仓一致的逐条目身份校验，按源 IR 的每个专家 GEMM `op_id` 对比 sidecar 的层、角色、专家号及对应权重身份，而不是比较集合；编译器导出、skip-simulation、仿真前、trace 和直接 GeneSim 全部调用同一规则。测试除缺失/多余组合外，必须加入同层同角色专家号互换、权重路径与专家号不一致，并保留合法 host、边界、吸收及真实 128 专家正例，避免再次只修直接 GeneSim 单点。

### 问题4（高）: 全模型验收未闭环，导致交付不能关闭
　　**问题说明**：本轮要求 48 层 Qwen3-30B-A3B 在 8 DPU × 8 GiB 下真实走通图编译、FlagTree、sidecar 与 GeneSim，并与 HF 逐 token 对齐。当前数值目标已实测 **26 passed、1 skipped**，BCD 与精化也已完成并生成 37923 条有效 sidecar 条目和 44117 个精化算子，但 `full30b-pipeline-r6-qwen3` 仍在 GeneSim 仿真阶段，summary 与最终 trace 覆盖尚未形成。因而 48 层数值、图编译和算子编译不再记为未验证，全链验收仍不能据这些中间产物提前关闭。实施文档的 K7/Q8 也需随本轮新证据更新，而不是继续只保留历史状态。
　　**根本原因**：前几轮以 4 层切片和历史结果支撑迭代，48 层长链路验证持续后置；本轮重跑已把缺口收敛到 GeneSim 仿真及最终 trace 验收。当前不能预判仿真是代码失败还是环境失败，只有主链单元终态和真实 summary 才能判定。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §1.3、§5.1 K7、§6 Q8、§8.4；本轮 `/tmp/qwen3-review-round06/target48l.log`、`pipeline48l.log`、`pipeline48l-logs/bcd_export.log`、`pipeline48l-logs/e_sim_pimir.log`。
　　**修改思路**：保留已通过的 48 层数值与 BCD 证据，继续等待主链 systemd 单元至终态，记录退出码、summary、trace 来源和源 IR 覆盖；若仿真暴露代码问题则按实际根因重新打开对应实现项，若环境终止则记环境阻塞。主链成功后以新结果回填实施文档并明确 8B 仍未覆盖，不能用历史日志或中间产物代替最终全链结论。

### 问题5（中）: 单专家误判非MoE，导致合法配置拒绝
　　**问题说明**：架构注册与静态前向面向 `qwen3_moe` 架构，并未把合法模型限定为专家数大于 1。`genesim/src/ir/model_ir.py:1327-1357` 与 `runtime/exec_plan_gen.py:42-49` 却以 `num_experts > 1` 判定 MoE：单专家时主链得到空执行语义，parser 随后以“非 MoE”处理，最终在 `_validate_moe_contract` 因空 `moe_execution` 报错。实测 HF 的 1 专家/top-1 模型能正常前向且静态改写逐元素相等，但 `build_from_hf_config` 无法生成 IR。相反假设复核确认它不是静默产出 dense IR，而是合法配置被错误拒绝。
　　**根本原因**：模型身份由专家数量阈值而非 `model_type` 与 MoE 配置共同决定，且同一 `>1` 假设散布在图构造、执行语义解析和 `runtime/compile.py:226-238` 的默认分块判断；因此单专家会同时失去 MoE 元数据、执行语义和 MoE 默认关分块策略。现有测试只覆盖 2/4/8/128 专家，没有守护该边界。
　　**涉及文件**：`genesim/src/ir/model_ir.py:1321-1357,2294-2297,2849-2857`；`flagos-pim-compiler/runtime/exec_plan_gen.py:36-53`；`flagos-pim-compiler/runtime/compile.py:226-238`；`genesim/tests/sim/test_model_ir.py`；`flagos-pim-compiler/tests/test_architectures.py`。
　　**修改思路**：建立统一的 MoE 配置判定：以 `model_type=qwen3_moe` 和有效 MoE 字段识别架构，专家数取正整数并校验 top-k 范围，不再用 `>1` 区分 dense/MoE；图骨架、编译配置、分块默认和 sidecar 共用该判定。增加 1 专家/top-1 的 HF 对拍、parser、编译执行语义与 sidecar 测试，同时验证普通 dense 配置不携带 MoE 字段的行为不变。
