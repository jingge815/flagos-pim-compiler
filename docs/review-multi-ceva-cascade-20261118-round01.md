# 代码评审报告：多 CEVA 级联仿真与 Llama2 7B 全流程性能评估
> 文档编号：review-multi-ceva-cascade-20261118
> 创建日期：2026-11-18
> 关联需求文档：request-multi-ceva-cascade-20261114
> 关联设计文档：design-multi-ceva-cascade-20261114
> 关联实施文档：implement-multi-ceva-cascade-20261114（第十三节已声明无效，第十六至十七节为后续修正）

## 一、评审概述

　　本轮覆盖四个仓库的未提交改动：pim-compiler（cascade 包三模块、gml_bridge/ceva.py、contracts/ceva.py、from_fx.py、五个脚本与四份测试，净增约 +1060 行）与 /data/fjg/ceva 工作区（段式 runner C++、ddr_manager、compile_gml、run_tools_step、run_cascade 及约 20 个新增脚本与三份测试，净增约 +530 行）；genesim 与 FlagTree 经 git status 核对确认本期零改动，符合设计声明。PP2 主链路的工程质量总体扎实：11 个 token 与单核基线逐位一致（本次直接复核归档 argmax_results.bin）、实施文档第十六至十七节自录的修复（TIMER 递减计数、失败段不覆盖 KV 快照、head 段编号、decode 交接补满、混合口径装载基线、加速比量纲、run_key 隔离等）经代码与测试逐项抽查均真实落地。核心问题集中在验收与证据侧：评估模型「标定与验证同源」使 0.8% 偏差属需求明文规定的无效验收（问题1），最大段 1.77 亿周期的手工扣减无溯源（问题2），一键入口从未端到端执行且实测绕开官方编排链路（问题3），单核默认行为在外壳入口被破坏（问题4），TP2 运行时缺失（问题5）。共列问题 10 个：严重 1、高 4、中 5。

## 二、修改建议

### 问题1（严重）: 评估标定与验证同源，导致偏差验收自证无效
**问题说明**：需求 §5.4 要求「标定与验证不能用同一组运行」，§5.5 将「评估模型用验证数据本身标定后再报告偏差」列为无效验收。实际 `cascade/layer_scaling_llama2_pp2.json` 与 `runs/pp2_v3/results/layer_scaling.json` 的 source 均自述系数 0.6804/0.9361 由「single_ours(32层单核) vs pp2_v3(16层分核)」拟合，其中分子 19,288,436 正是 pp2_v3 两核 prompt 实测之和 617,229,944÷32；`cascade/eval.py` 的 evaluate（约 L352-364）随后对**同一个 pp2_v3** 的实测总量报告偏差 0.8%（cascade_eval.json deviation=0.0092），而 prompt 项占总时间约 89%、其预测值按构造等于该 run 自身的分核实测，偏差被算术性压平。链路参数同样取自被评估 run 的 transfer_log 现场拟合（L326-327），通信项在预测与实测两侧取值相同、对偏差完全抵消。实施文档 17.16 以该 0.8% 宣称「达标（≤20%）」。  
**根本原因**：全期只有一次 PP2 实测运行，既当标定源又当验证对象；eval.py 的仓库级系数回退（_layer_scaling）与 calibrate_link 默认取本次运行日志，均没有「标定数据与被评估运行不相交」的机制约束或输出标注。同一根因使加速比 1.357× 也依赖问题2 的手工修正。  
**涉及文件**：cascade/eval.py（_layer_scaling/calibrate_link/evaluate）、cascade/layer_scaling_llama2_pp2.json、/data/fjg/ceva/runs/pp2_v3/results/{layer_scaling.json,cascade_eval.json}  
**修改思路**：把分离做成机制而非约定——evaluate 登记每个标定参数的来源 run 列表，与被评估 run 相交时 deviation 强制标注 `in_sample: true` 且不计入验收判据；取得第二次独立 PP2 运行做 out-of-sample 验证（可复用 GC 缓存与 state 断点把增量成本压到数小时）后再报偏差；仓库默认系数的 source 显式声明「对 pp2_v3 属同源验证」。

### 问题2（高）: 最大段手工扣减1.77亿周期无溯源记录
**问题说明**：非功能需求要求性能数字区分实测与外推、附参数来源。实际 `runs/pp2_v3/results/cycles_correction.json` 对 prompt:c0（占设备总时间约 38%）手工扣减 176,964,315 周期，文件无 source/推导字段，实施文档 16.10.6 只登记了机制未登记该数值的得出方式；扣减后 prompt:c0=310.68M 与 prompt:c1 的 TIMER 值 308.61M 恰好只差 0.7%，报告口径的「每核 busy 偏差 1.4%」结论由这个不可审计的手工值塑造。eval.py parse_state_cycles（L93-108）读到即扣。  
**根本原因**：prompt:c0 由旧 ELF（慢速逐头快照路径）跑出、与其余段口径不一致，正确做法是重跑该段，实际选择了无记录的手工修正；eval 消费 correction 时没有溯源强制。  
**涉及文件**：cascade/eval.py（correction 分支）、/data/fjg/ceva/runs/pp2_v3/results/cycles_correction.json、docs/implement §16.10  
**修改思路**：用当前 ELF 重跑 prompt:c0 一次性消除修正需求（断点续跑已支持，单段约 2.3h）；在此之前 eval 对缺 `source` 字段的 correction 拒绝采用并报错，correction 格式强制为 `{value, derived_from, measured_at}`，使任何扣减可复现。

### 问题3（高）: 一键入口未端到端实证，实测绕开官方编排链路
**问题说明**：R6 验收为「一条命令完成 2 CEVA PP2 全流程并产出判定报告」。实际 33 段实测由工作区脚本 `scripts/pp2_tokens.py` 手工驱动（硬编码 RUN=runs/pp2_v3、GMLS 路径、基线 tokens 与读宽），官方编排 `cascade.driver` 只在 17.14 复用其中 22 段、补跑 11 个 head 段；设计入口链 run_full_pipeline.py→run_cascade.sh 从未完整执行——runs/pp2_e2e 仅有 gmls/ 与 logs/（无 ddr_core*、无 results/verdict.txt），pim-compiler 侧 runs/multi 为空目录。17.12/17.14 的结果本身真实，但属「工作区脚本 + 部分官方编排」的拼装形态。  
**根本原因**：M0 架构转向（一段一进程 + 文件交接）先在工作区实验脚本上迭代成熟，未回灌官方入口即进入验收；一键链路只验证到干跑层。  
**涉及文件**：/data/fjg/ceva/scripts/pp2_tokens.py、/data/fjg/ceva/run_cascade.sh、cascade/driver.py（main）、scripts/run_full_pipeline.py（run_ceva 级联分支）、/data/fjg/ceva/runs/pp2_e2e、runs/multi  
**修改思路**：以官方链路完整执行一次并归档：`run_ceva_pipeline.sh --run-name <新名> --num-ceva 2 --strategy pp2`（GC 复用 pp2_e2e 缓存、段级断点续跑，增量成本可控），verdict 与 cascade_eval 落在统一 run 目录作为 R6 证据；随后将 pp2_tokens.py 的段后自检（KV 全量 diff、top5、指纹）以可注入 hook 并入 driver，消除双实现（与问题7 合并收敛）。

### 问题4（高）: 入口默认改双核并封禁基线名，单核默认行为被破坏
**问题说明**：需求 R6「不带多核参数时保持现有单核行为不变」、非功能需求「多核入口的默认行为等价现有单核行为」。实际 `scripts/run_ceva_pipeline.sh` 新逻辑在未给 --num-ceva 时自动注入 `--num-ceva 2 --strategy pp2`，并把 ours_e2e/pp2_e2e/pp2_v3/single_ours 列为受保护 run-name 直接拒绝——需求 §一引用的单核标准命令 `run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10` 现在要么变成双核级联、要么报错退出。  
**根本原因**：把「多核是本期主题」实现成了外壳入口的默认值；兼容性约束只在 run_full_pipeline.py 层保留（--num-ceva 默认 1），外壳层被改写且无默认值语义的回归测试。  
**涉及文件**：scripts/run_ceva_pipeline.sh（DEFAULTS 注入与保护名单）、tests/test_ceva_export.py::test_run_ceva_pipeline_sh_passthrough（未覆盖默认值）  
**修改思路**：默认恢复 `--num-ceva 1 --strategy single`，多核必须显式传入；保护名单仅对多核分支生效（单核重跑基线是合法操作）；补一条「外壳无参调用与旧版命令行等价」的回归测试。

### 问题5（高）: TP2 运行时缺失且残留已废弃机制脚本
**问题说明**：R5 要求 TP2 走完全流程并给出两核同层时间线重叠证据。实际段式 app 未实现 cmd=4/5/6（llama2_npm_app.cpp 的 switch 落入 default 直接跳过，out_size 亦无分片分支）；cascade/driver.py 的 run_tp2 仍是 M0 已否定的「长驻控制块会话」模型，CLI 入口明确 SystemExit 拒绝；`scripts/tp2_parallel.py` 是旧控制块机制的演示脚本——cmd 只写进控制块、未进 argv（当前 app 从 argv 取命令，argc=8 时 cmd=0 空跑），其归档 tp2_result.json 的「100% 重叠 / 加速 1.99×」无法用当前代码复现，且跑的是 PP2 布局的 decode 段而非 TP2 分片。TP2 现状 = 导出 + GC + 预测（assumption），R5 两项判据均无有效证据。  
**根本原因**：M0 架构转向后 TP2 运行时（一段一进程形态的 attn/mlp 并行段 + host 归约 + logits 拼接）未排期实现，旧演示脚本未清理，形成「看起来有 TP2 验证」的假象。  
**涉及文件**：/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp（switch cmd 分发）、cascade/driver.py（run_tp2/main）、/data/fjg/ceva/scripts/tp2_parallel.py、/data/fjg/ceva/runs/tp2_parallel/tp2_result.json  
**修改思路**：按一段一进程口径实现 TP2 运行时（app 增加 cmd=4/5/6 与分片输出尺寸；driver 按 plan 的 _tp2_segments 并行下发两核段进程、host 归约与拼接），时间线证据直接取双进程 resume 起止时刻；同时删除 tp2_parallel.py 或改造为对新机制的回归测试，归档的 tp2_result.json 标注「旧机制产物，不可复现」。

### 问题6（中）: eval 模块尾部缺陷：tp2 CLI 崩溃与不可达死代码
**问题说明**：本次实测 `python -m cascade.eval --strategy tp2` 直接崩溃：`NameError: name 'predict_tp2' is not defined`——该函数定义在 `if __name__ == "__main__": raise SystemExit(main())`（eval.py L432-433）**之后**（L436），以模块方式执行到 main() 时尚未定义；run_cascade.sh 因前置拦截未暴露此路径，属潜伏缺陷。另外 parse_state_cycles 在 return（L110）之后残留一段引用已删变量（clocks/source）的不可达旧实现（L112-125），属重构残留，违反仓内「删优于加」约束。  
**根本原因**：17.16 追加 predict_tp2 时直接贴在文件末尾、未整理模块结构；缺少「__main__ 块必须位于全部定义之后」的检查。  
**涉及文件**：cascade/eval.py L112-125、L432-456  
**修改思路**：把 predict_tp2 移到 main 之前并删除不可达块；补一条用假夹具跑 `python -m cascade.eval --strategy tp2` 的冒烟用例，锁定模块可 CLI 自举。

### 问题7（中）: 编排双真源与伪证据脚本堆积
**问题说明**：设计 3.3 约定切分与编排只在 pim-compiler、工作区不重复表达策略。实际 PP2 编排存在两份并行实现（cascade/driver.py 的 run_pp2_processes 与 scripts/pp2_tokens.py，段序/交接/补满/LUT/落盘逻辑重复且细节已分叉：counters 回退口径一处 `exec_cycles or cycles`、一处 `exec_cycles or device_cycles or cycles`）；入口并存三套（run_full_pipeline→run_cascade、run_all_goals.sh、pim scripts/run_all.sh）。其中 pim `scripts/run_all.sh` 的「[1/8] 编译调优空间选择」仅打印「✅ 策略选择完成」无任何实现——调优链路明确不在本期边界，伪造的完成痕迹会污染核心目标 1 的证据链；其第 5 步仍打印 core1 基址 0x300000000（与实现相悖），第 7 步实际驱动的又是硬编码 pp2_v3 的 pp2_tokens.py。另有 scripts/pp2_chain.py（第十三节已判定造假：抄基线 tokens、硬编码 baseline_match=True）与 run_eval.py/final_report.py（用本次运行自身均值预测自身再报「偏差≤20% ✅」，硬编码 LOAD=217 与历史单核 25260s）均可直接执行，易被误当验收证据。  
**根本原因**：实验期脚本未经收敛直接留在 scripts/，缺少「官方链路 vs 实验脚本」的目录与生命周期约束。  
**涉及文件**：/data/fjg/ceva/scripts/{pp2_tokens.py,pp2_chain.py,run_eval.py,final_report.py,run_all_goals.sh}、flagos-pim-compiler/scripts/run_all.sh、cascade/driver.py  
**修改思路**：确立 cascade/driver.py 为唯一编排真源后：删除 pp2_chain.py（造假脚本）、run_eval.py、final_report.py；run_all.sh 与 run_all_goals.sh 删除或收敛为 run_cascade 的薄封装；pp2_tokens.py 的自检能力以可注入 hook 并入 driver 后退役为测试工具。

### 问题8（中）: 工作区回归失败未登记，归档留档机制被移除
**问题说明**：本次实际运行工作区 `python -m pytest tests/ -q` 得 **62 passed / 1 failed**——test_gitignore.py::test_existing_success_archives_are_present 断言 runs/ours_full/ARCHIVE.md 存在而该归档已不在盘上（bundles/ours_full.log 显示该运行曾存在）；实施文档最近声明（16.14.6「工作区 62 passed」）未登记该失败，与现状不符。同时 .gitignore 把原来「runs 大产物忽略 + 判定摘要/argmax/counters 白名单留档」改成 runs/ 整体忽略，成功运行的判定摘要从机制上失去留档约束，与需求「任何性能数字都能追溯到具体的运行目录」的可审计性相冲突（ours_full 消失即为实例）。  
**根本原因**：忽略策略调整与磁盘清理时未同步测试基线，归档删除无记录；归档策略与可审计性要求未对齐。  
**涉及文件**：/data/fjg/ceva/.gitignore、/data/fjg/ceva/tests/test_gitignore.py、runs/ours_full（缺失）、docs/implement §16.14  
**修改思路**：恢复 ours_full 归档摘要（备份找回或由 bundle/log 重建 ARCHIVE.md）或修订测试清单并把删除原因记入实施文档；对成功运行的 verdict/argmax/counters 恢复白名单强制留档，仅忽略大体量中间产物。

### 问题9（中）: 设计与实现的口径漂移未回写设计文档
**问题说明**：设计 4.3 与 §8.3 钉死「核 N 基址 = 0x100000000 + N×8GiB、全局账面无重叠」，实现按 M0 实测改为两核同基址（llama2_ddr_manager.py 多核分支；pp2_v3 两核 control 均为 0x100001000，本次核对），R2 验收判据「两核 map.ini 地址空间无重叠」按现口径字面无法满足；设计 §5.1 的 verify_run.py/08_result.sh 修改项由独立 verify_cascade.py 替代（实施 5.2 已声明且标注「未经确认」）。这些修正只记录在实施文档 16.14 与第七轮，设计文档的修正记录节（§8）未追加，后续按设计文档验收会直接冲突。  
**根本原因**：M0 实测推翻设计假设后走了「只更实施文档」的路径，没有同步设计修正条目，三份文档的追溯链断裂。  
**涉及文件**：docs/design-multi-ceva-cascade-20261114.md（§4.3/§5.1/§8）、/data/fjg/ceva/src/ddr_manager/llama2_ddr_manager.py  
**修改思路**：在设计 §8 追加修正记录：同基址的实测依据与适用条件、R2 判据改写为「每核产物落在各自 TCS 装载窗口内且进程间无地址依赖」、verify_cascade 替代 verify_run 的决策；此后每处设计偏离都走该节登记。

### 问题10（中）: 计数产物混入重复段记录，host 汇总口径失真
**问题说明**：state.json 的 timing 是追加式日志：pp2_tokens 跑出 33 段后 driver 又重跑 11 个 head 段，形成 44 条记录（decode:k:lm 与 head:k+1 同物异名共存）。_write_pp2_artifacts 不去重直接写 performance_counters_core{N}.txt（44 行）与 segment_timeline（44 条），evaluate 的 host.wall_s 把重跑时间一并累加为 37,945s，而实施文档 17.12 手工按 33 段去重引用 34,777s，两个数字并存且无标注；eval 的分段 cycles 因字典后写覆盖才侥幸未双重计入。verify_cascade 的性能检查（每核至少一段）也拦不住重复。  
**根本原因**：timing 同时承担「执行日志」与「最终权威表」两个角色，落盘产物缺少唯一的权威投影。  
**涉及文件**：cascade/driver.py（execute/_write_pp2_artifacts）、/data/fjg/ceva/scripts/pp2_tokens.py、/data/fjg/ceva/runs/pp2_v3/results/state.json  
**修改思路**：落盘前按归一化段名（_seg_id 语义）取每个逻辑段的最后一次记录生成权威表，counters/timeline/host 汇总一律从权威表生成；重跑段的历史记录保留在 timing 日志里供审计。

## 三、核心目标完成情况

| 需求编号或目标 | 状态 | 证据 | 缺口或外部阻塞 |
|----------------|------|------|----------------|
| 目标1 多 CEVA 级联仿真底座与全链路 | 部分完成 | PP2 真机 33 段两核级联跑通，11 tokens 逐位一致（本次复核 runs/pp2_v3/run/argmax_results.bin）；R1 双进程底座（M0 结论 + runs/pp2_v3 分段进程记录）、R2 双核布局与容量核对（test_ddr_multi 本次通过）、R3 导出（pp2_e2e 65 图 complete=true、tp2_e2e 两核 129 图 complete=true，本次核对清单）均有实现与测试 | R6 一键形态未实证（问题3）；TP2 仿真未跑（问题5）；「编译调优空间选择」按边界本期不接入，但 run_all.sh 伪造完成步骤污染证据（问题7） |
| 目标2 面向多 CEVA 级联的编排器 | 部分完成 | cascade/plan.py 蓝图（段序列+通信计划表+可读文本）与 cascade/driver.py 运行时（段驱动/交接/LUT/断点续跑/产物落盘）实现，28 个单测本次全绿；PP2 编排（任务拆分、映射、通信、串行流水）经真机验证 | 并行编排（TP2）运行时缺失，run_tp2 为废弃机制（问题5）；编排真源分裂为 pp2_tokens.py 与 driver.py 两份、入口三套（问题3/7） |
| 目标3 多 CEVA 组合评估模型 | 部分完成 | cascade/eval.py 组合公式（PP 串行链/TP 层同步 max）、实测+预测双模式、参数溯源、三层抽象 [2,1,1] 与文本报告实现，27+ 用例绿；实测模式对 pp2_v3 输出完整结构分解（cascade_eval.json/txt 齐全） | ≤20% 偏差验收属自标定自验证、无效（问题1）；最大段手工修正无溯源（问题2）；TP2 预测为纯 assumption 且无实测可验证；tp2 评估 CLI 崩溃（问题6） |
| R1 多进程级联底座 | 已完成 | M0 探针（实施 §11）、双进程并行驱动与跨核搬运、各核独立 cycles（runs/pp2_v3 state.json 44 条分段记录） | 无 |
| R2 多核 DDR 布局 | 部分完成 | 同基址多核布局、层过滤、KV 重编号、embedding_row、控制块与容量核对实现并有测试 | 同基址偏离设计验收口径且未回写设计（问题9） |
| R3 多核切分导出 | 已完成 | pp2 bundle cascade 节 + 65 图过 GC（runs/pp2_e2e/gmls）；tp2 两核 129 图 complete=true、四类分片图过 GC（复用实施 §九/十差分证据链） | 无 |
| R4 PP2 token 对拍 | 已完成 | 11/11 token 逐位一致（本次复核归档 bin），verify_cascade PASS | 无（执行形态缺口计入 R6） |
| R5 TP2 并行编排与收益 | 未完成 | 仅导出 + GC + 预测；无运行时、无 tokens、无时间线 | 见问题5 |
| R6 一键命令入口 | 部分完成 | 参数链、干跑、防呆校验齐全（bash -n 与单测通过） | 从未端到端执行；外壳默认改双核破坏单核兼容（问题3/4） |
| R7 组合评估模型 | 部分完成 | 模型、溯源与产物齐全 | 偏差验收循环标定无效、手工修正无溯源、TP2 无实测（问题1/2/5/6） |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
|--------------|----------|------|----------------------|
| pim-compiler `pytest tests/test_cascade_{plan,driver,eval}.py -q` | 本次实际验证 | 28 passed | 工作区未提交改动；pytorch env（torch 2.9.1） |
| 工作区 `pytest tests/ -q`（/data/fjg/ceva） | 本次实际验证 | 62 passed / 1 failed（test_gitignore::test_existing_success_archives_are_present，见问题8） | 工作区未提交改动；pytorch env |
| `bash -n` run_cascade.sh、run_all_goals.sh、watch/auto_chain/launch×3/pp2_monitor、run_ceva_pipeline.sh、pim run_all.sh | 本次实际验证 | 全部通过 | 工作区未提交改动 |
| `python -m cascade.eval`（pp2 假夹具 /tmp/eval_fixture） | 本次实际验证 | pp2 正常产出报告；tp2 报 NameError（问题6） | 工作区未提交改动 |
| pp2_v3 tokens 复核（读 run/argmax_results.bin 比对基线） | 本次实际验证 | 11/11 逐位一致 | 归档 runs/pp2_v3（2026-11-17/18 运行），未重跑仿真 |
| 单核导出逐字节不变：md5 抽查 ours_e2e vs pp2_e2e（prompt L0 / decode L0 / lm_head 的 GML） | 本次实际验证 | 三处 SAME | ours_e2e 为改动前（2026-10-28）导出、pp2_e2e 为改动后导出；抽查非 65 图全量 |
| 65 图全量 md5、全量回归 `pytest -k "not llama2_7b"`（2190 passed） | 复用已有验证 | 声明通过 | 实施文档 17.13 / 16.14.6；其后代码仅有 17.x 小改，未重跑 |
| TP2 四类分片图过厂商 GC | 复用已有验证 | 声明通过（probe2/probe3 差分链） | 实施文档 §九/十；bundles/tp2_e2e complete=true 本次核对 |
| 2 CEVA PP2/TP2 仿真本体、llama2_7b 慢用例（48 个） | 仅核对实现但未执行 | 不重跑（单次约 10 小时，按纪律复用归档） | 归档 runs/pp2_v3、runs/single_ours |
| genesim / FlagTree 零改动 | 本次实际验证 | git status 干净 | 两仓库工作区 |
| 第十三节假数据事件复发面 | 本次实际验证 | pp2_chain.py 仍在且可执行（问题7），官方链路未受污染 | /data/fjg/ceva/scripts/pp2_chain.py |

　　未覆盖范围及限制：本轮未重跑任何 TCS 仿真与全量回归，所有性能与 token 结论均基于归档产物（runs/pp2_v3、single_ours、ours_e2e、pp2_e2e、tp2_e2e）的交叉核对，这些产物由实施文档第十六至十七节记录的运行产生、代码其后有小幅修改；单核导出逐字节不变仅抽查 3 张图，全量口径依赖实施文档 17.13 的声明。历史 review 轮次不存在（本轮为第 1 轮），「已修复问题复发抽查」以实施文档自录问题清单（I1-I7、16.x、17.x）替代，抽查项（TIMER 递减、失败段不覆盖快照、head 编号、交接补满、装载基线、加速比量纲、run_key 隔离）均确认真实落地且有测试锁定；未覆盖复发风险集中在标定链（问题1/2）与入口默认值（问题4），当时即以单例方式处理、未形成机制约束。
