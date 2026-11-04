# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-round8
> 创建日期：2026-11-02
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述
本轮审查图编译器、FlagTree 与 CEVA 工作区的全部未提交变更，GeneSim 无未提交变更；共发现 8 项问题：高 3 项、中 4 项、低 1 项，核心风险是中断续导混用生成口径、当前版本完整验收未闭环及历史证据删除。
图编译器快速全回归 1656 项通过、9 项跳过，针对性测试 243 项、CEVA 工作区 43 项、FlagTree lit 265 项均通过；现有 `r6_full` 65 图包的来源、引用、尺寸与摘要校验 327/327 通过，但这些检查未覆盖下述复现。`check_orig.sh` 报告共享原始目录与 10 月 28 日快照有 105510 处变化；本轮未新跑 GC 或仿真，差异归因仍需核查。

## 二、修改建议
### 问题1（高）：中断续导混用口径，导致整包量化失配
**问题说明**：设计 §5.3 要求生成口径变化时只允许整包重导，同一包的 prefill/decode 必须共用一致的 KV scale。`export_ceva_bundle` 在整包重导完成前就覆盖全局 `generation_digest`，而 `bundle_complete` 只检查图名是否齐全，不检查各图实际生成口径。用两张图、标定值从 1 改为 2 复现：整包重导在第 1 层失败后，只重试第 0 层，最终清单为 `complete=true`，但第 0 层是新口径 2、第 1 层仍是旧口径 1。现有用例只覆盖“未发生中断”的整包重导，未覆盖该路径。

**涉及文件**：`gml_bridge/ceva.py:919-928`（`bundle_complete`）、`gml_bridge/ceva.py:1028-1068`（`export_ceva_bundle`）、`tests/test_ceva_export.py:1539-1565`。

**修改思路**：为每张图记录生成摘要，或在清单中可靠记录本次重导尚未覆盖的旧图；只有全部图都与当前全局摘要一致时才允许 `complete=true`。整包重导中途失败后的任意局部续导，也必须证明未重导图符合本次口径，不能只比较已被提前覆盖的全局摘要。补“第一张图成功后失败，再局部重试”的回归用例，并在清单校验器中拒绝混合口径的完整包。

### 问题2（高）：当前全链验收未闭环，导致仿真可行性未证
**问题说明**：需求 P0-4、P0-5 要求当前我方 GML/bin 完成厂商 GC、DDR、CEVA 仿真的完整流程，至少完成全模型 1 步 decode，且原始目录无变化。当前 `r6_full_e2e` 只记录 32 张 decode、lm_head 与 2 张 prefill 共 35/65 张图的 GC 成功日志，未进入 DDR 与仿真，也没有判定报告；11 月 1 日 `own_bins_e2e` 的 3 步通过记录早于运行时缓冲写零及后续修复，不能替代当前代码验收。本次 `check_orig.sh` 还实际返回失败，共享原始目录存在 105510 处快照差异，当前无法确认其归属，更不能判定 P0-4 已通过。

**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md:16-20,181,228-232`；`/data/fjg/ceva/runs/r6_full_e2e/logs/04_step.log:1-36`、`run_all.log:23-48`；`/data/fjg/ceva/runs/own_bins_e2e/ARCHIVE.md`。

**修改思路**：修复本报告中的生成一致性问题后，按当前代码重导并校验 65 图，完成全部 GC、DDR 与至少 1 步全模型仿真，核心检查零 SKIP，并归档判定与耗时；默认跑 10 步，缩减时记录原因。先对照时间与运行日志查明共享目录差异来源，不要直接覆盖共享资料；后续 GC 只在隔离副本运行，并保留每次运行前后的快照证据。

### 问题3（高）：历史验收摘要被删除，导致结果不可复核
**问题说明**：需求第四章要求归档 GC 日志、CEVA 日志、token 与文本、判定报告及耗时，项目指引也要求 Git 保留这些摘要。当前工作区删除 `ours_e2e`、`ours_e2e2` 共 26 个已跟踪摘要文件，包括 `run.log`、`verdict.txt`、`argmax_results.bin` 和步骤日志；两个目录的 `ARCHIVE.md` 也都不存在，但实施文档仍引用它们作为两次 10 步 PASS 的归档索引。忽略规则测试只证明新文件可跟踪，不能防止既有证据被删除。

**涉及文件**：`/data/fjg/ceva/runs/ours_e2e/`、`/data/fjg/ceva/runs/ours_e2e2/`（`git status` 中 26 个 ` D`）；`docs/implement-ceva-llama2-7b-20261029.md:104-105`；`tests/test_gitignore.py`。

**修改思路**：从 HEAD 恢复这 26 个摘要文件，并按现有归档格式补齐两个 `ARCHIVE.md`；恢复前不要把删除纳入提交。增加对既有成功运行摘要存在性的检查，保留大产物忽略规则，不要用清理大产物的方式一并删除小型验收证据。

### 问题4（中）：动态定标仍写样本，导致运行缓冲失真
**问题说明**：需求 P0-3 与设计 §4.7 要求动态 scale 只按尺寸分配、写零，标定样本不得冒充运行时计算。`runtime_buffer_names` 只匹配数据缓冲、相位输入输出与 MatMul 权重通路，DQ 的 `output_sf` 和 `kantor_A_scale_buffer_file_phase_3` 被默认归为静态；写盘器分别写入由样本算出的 phase1、phase2。现有 `r6_full` 的 `output_sf_12.bin`（64 字节）及对应 Kantor scale 均为非零，二者按 `DynamicScalingPhases` 定义正是逐组动态 scale，不是 KV/输出头的静态标定 scale。测试只遍历同一判定函数挑出的文件，因此漏掉了这两类。

**涉及文件**：`gml_bridge/ceva.py:229-248`、`gml_bridge/export.py:526-552`、`gml_bridge/runtime_files.py:365-392`、`gml_bridge/phase_data.py:99-118`、`tests/test_ceva_export.py:1655-1692`。

**修改思路**：按字段语义把 DQ 的输出 scale 与相位 Kantor 动态 scale 纳入写零集合，同时复核 Softmax 的运行时定标落点；固定量化常量、KV/输出头静态 scale 不得误清零。测试应直接断言这些具名字段为零，并断言静态 scale 保持标定值，避免测试与实现共用同一个不完整的分类集合。

### 问题5（中）：空源码树版本有效化，导致局部混导放行
**问题说明**：设计 §5.3 要求读不到 FlagTree 源码树时将口径记为未知并拒绝局部续导。`opcompiler_revision` 只在源码根路径不存在时返回 `unknown`；若根目录存在但必需子目录或源文件缺失，函数对空输入求哈希并返回有效版本 `e3b0c44298fc`。本次用空目录实测得到该结果，局部续导的 `OPCOMPILER_UNKNOWN` 守卫因此被绕过。现有测试只覆盖“根路径为 None”，未覆盖空树或不完整树。

**涉及文件**：`gml_bridge/ceva.py:981-1003,1040-1047`、`tests/test_ceva_export.py:1792-1810`。

**修改思路**：校验必需源码目录及有效源文件，无法取得完整口径时返回 `unknown`，不要对空输入给出版本号。分别补空目录、缺少必需子目录、只有无关文件的局部续导拒绝用例。

### 问题6（中）：多行 Softmax 校验误判，导致合法 prefill 失败
**问题说明**：设计 §4.3 要求 prefill 的 Softmax phase0/2/3 每个查询行各有一个值。`check_softmax_reduction_encodings` 仍要求 phase0 文件总长恰为 4 字节，`check_softmax_is_normalised` 仍把整张输出求和后与 1 比较。用合法的 2 行×4 列 Softmax 实测，两行各自归一且编码正确，两项检查却分别报“低 2 字节非零”和 `Σ=2.00061`，会把合法 prefill 产物判失败。

**涉及文件**：`scripts/verify_gml_artifact.py:802-867`、`gml_bridge/phase_data.py:254-271`、`tests/test_verify_layers.py:245-323`。

**修改思路**：按 phase0 的 4 字节记录数确定行数，逐条检查高半 fp16 编码，并按输入行宽逐行检查 phase4 之和；同步更新仍写“4 字节标量”的注释。增加两行不同最大值的正例及逐行编码、归一化负例。

### 问题7（中）：FPSU 边界编码失真，导致缩放精度异常
**问题说明**：设计 §4.1 要求定点 FPSU 取不溢出的最大右移并正确生成 int16 乘数。当前判断使用 `mul < 32767`，排除了合法的 32767；同时未拒绝舍入后的零乘数。实测 `factor=32767/2^31` 返回 `(16384,30)` 而非可取的最大右移结果 `(32767,31)`，`factor=1e-20` 返回 `(0,31)`，后者会把正缩放静默变成零，不符合通用生成策略。现有测试只覆盖厂商给出的两个常规值。

**涉及文件**：`gml_bridge/runtime_files.py:292-305`、`tests/test_runtime_files.py:343-353`。

**修改思路**：允许 `mul == 32767`，并在所有候选右移都无法得到非零、不溢出的乘数时明确报错或按已核实的硬件规则处理，禁止静默写零。补 32767 上界与极小正因子的边界测试。

### 问题8（低）：反量化漏除保护倍数，导致校验误报
**问题说明**：量化器规定落盘 scale 已乘 `weight_sf_multiplier`，反量化必须除回；`QuantizedWeight.dequantize` 已按此实现，但 `check_dequantization_recovers_weights` 仍直接用码值乘落盘 scale。小权重实测倍数为 4 时，校验器还原值也放大 4 倍，会把正确产物误报为权重不匹配。

**涉及文件**：`scripts/verify_gml_artifact.py:488-560`、`quant/weights.py:38-45,71-81`。

**修改思路**：从同节点读取 `weight_sf_multiplier`（缺省为 1），在反量化时除回，并与 `QuantizedWeight.dequantize` 共用同一规则。补一个带倍数为 4 的小权重校验用例，同时保留输入维×输出维布局还原检查。
