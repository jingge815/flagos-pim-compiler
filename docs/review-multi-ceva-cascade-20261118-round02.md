# 代码评审报告：多 CEVA 级联仿真与 Llama2 7B 全流程性能评估（第 2 轮）
> 文档编号：review-multi-ceva-cascade-20261118-round02  
> 创建日期：2026-11-18  
> 关联需求文档：request-multi-ceva-cascade-20261114  
> 关联设计文档：design-multi-ceva-cascade-20261114  
> 关联实施文档：implement-multi-ceva-cascade-20261114（第十三节无效，第十六至十八节为修正与 round01 修复记录）  

## 一、评审概述

　　本轮覆盖四个仓库的未提交改动并复核 round01 十个问题的修复：pim-compiler 的 cascade 三模块（+hooks/权威投影/TP2 进程编排）、导出链与入口，/data/fjg/ceva 的段式 runner（cmd=4/5/6）、ddr_manager、run_cascade、.gitignore 留档机制与测试；genesim 与 FlagTree 经 git status 核对仍为零改动，符合设计声明。复核结论：round01 的 1、2、4、6、8、9、10 号问题已按根因修复并被新增测试锁定（标定同源强制 in_sample 标注、correction 溯源三字段、外壳默认恢复单核、eval 尾部缺陷清理、归档白名单留档、设计 §8.6-8.8 回写、权威投影去重，均有代码与产物证据）；问题 3（一键入口实证）未完成——官方链路在 GC 步被长跑纪律终止，verdict/cascade_eval 从未产生而实施文档声明已完成；问题 5（TP2 运行时）的修复存在功能性缺陷；问题 7 大体完成但残留一处旧归档未标注。本轮新列问题 7 个（严重 1、高 2、中 3、低 1），集中在 TP2 运行时与评估的未实证路径、以及证据与文档的一致性；测试侧本次补跑全量回归 2206 passed / 0 failed（25 分钟，填补实施文档 4.3 的「待回填」），工作区 67 passed，cascade 三件套 35 passed、导出 97 passed。PP2 主链路的质量判断与 round01 一致（11 token 逐位一致的结论仍成立），未跑任何小时级命令。

## 二、修改建议

### 问题1（严重）: TP2 驱动未传 layer，恒跑第 0 层图，导致【TP2 数值全错】
**问题说明**：设计 4.7/需求 R5 要求 TP2 两核在每个 decode/prompt 步内并行执行「同层」的分片算子，实施文档 18.1 问题5 声明该运行时「已修复（app 增 cmd=4/5/6，argv 传 stage/layer，按目录名定位注册表；driver 增 run_tp2_processes）」。实际 app 侧契约完整（`llama2_npm_app.cpp:520` 从 argv[14] 取层号、`:531-535` find_shard 按 `model_layers_{L}_{stage}_v2_{attn|mlp}` 定位注册表、`:437-457` 按层 KV 快照也用该层号），但 `cascade/driver.py:496-499` 的 execute_pair 调 `runner.run(..., stage=stage)` 时从未传 `layer`，`SegmentProcessRunner.run`（`driver.py:192-194`）在 stage 非空时追加 `str(layer)` 即字符串 `"None"`，app 侧 `atoi("None")=0`。本次实际验证：合成 2 层 TP2 蓝图离线跑 run_tp2_processes，所有 attn/mlp 段收到的 layer 均为 `'None'`——即 32 层真机运行时每一层都执行第 0 层的分片图与第 0 层 KV 张量，tokens 与性能数据必然错误且无告警。  
**根本原因**：driver↔app 的 argv[14] 层号契约只在 app 侧落地，Python 侧 run() 虽有 layer 形参但唯一的 TP2 调用点没有接线；测试 seam 掩盖了缺陷——`tests/test_cascade_driver.py` 的 FakeRunner.run 签名没有 layer 形参（若驱动传了会 TypeError），测试蓝图只有 1 层（layer 0 恰是缺陷路径会选中的层），18.1 记录的 I11 真机探针也只测了单层。同一根因覆盖 cmd=4 的按层 KV 快照/恢复路径（app 恒用 `v_key_cache[0]`）。  
**涉及文件**：cascade/driver.py（execute_pair L496-499、run L174-194）、/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp（L437-457、L519-535）、tests/test_cascade_driver.py（FakeRunner 签名、单层蓝图）  
**修改思路**：execute_pair 按 `seg.layer_lo` 传 layer（head 段不追加或显式 None），并把「argv 组装」收敛为 run() 内唯一出口、对 stage 有值而 layer 为 None 的 attn/mlp 调用直接抛错；FakeRunner 签名与真实 run 对齐；测试蓝图改 ≥2 层并断言每段 layer 随层递增；app 侧对 argv[14] 非数字返回 -1 使 find_shard 报错退出，而不是静默取 0——让契约违例在第一段就显式失败。  

### 问题2（高）: TP2 prompt 相中断续跑静默算错，导致【续跑结果不可信】
**问题说明**：实施文档 16.3/18.1 问题5 声明 TP2 支持断点续跑（test_tp2_processes_eos_and_resume 锁定），需求非功能项要求任何中断可恢复。实际 `run_tp2_processes` 的 prompt 相 x 初值读 `io[0]["inputs"]["hidden_states_file"]`（driver.py L573-575），而 `fanout`（L545-550）每层把归约结果写回两核的 hidden_states 输入文件——本次实际验证：离线跑 2 层 TP2 跑完一层后，原始全零的 h0.bin 首 8 元素已变为 12.0，即原始 prompt 输入被覆盖。若中断发生在 prompt 相中间（prompt 是小时级的最长阶段，是最可能的中断点），重启后已完成段虽复用输出，但 x 从被覆盖的文件起算，归约链全错、fanout 再把错误数据写进 pending 层的输入文件，token 静默算错。PP2 路径不受影响（core0 的输入文件从不被覆盖）。  
**根本原因**：原始 prompt 输入文件同时承担「设备装载源」与「host 重启推导源」两个角色，fanout 原地覆盖破坏了后者；断点状态只保存段输出，没有阶段级输入快照。现有续跑测试只覆盖「全部段完成后重入」与 decode 相（x 可由 embedding 行重推），未覆盖 prompt 相中断。  
**涉及文件**：cascade/driver.py（run_tp2_processes L545-575）、tests/test_cascade_driver.py（续跑用例缺口）  
**修改思路**：fresh 启动时把两核原始 hidden_states 快照到 results/（如 prompt_input_core{N}.bin），重启一律从快照取 x；或 fanout 写独立 staging 文件、由装载 ini 指向 staging，原始输入文件永不改写。补「prompt 相中断后续跑」离线测试：删去 prompt 中间层的 done 记录后重启，断言 pending 段读到的输入文件与首跑逐字节一致。  

### 问题3（高）: 一键入口证据缺失而文档称完成，导致【R6 验收无证据】
**问题说明**：R6 验收判据是「一条命令完成 2 CEVA PP2 全流程并产出判定报告」，可审计性要求性能数字可溯源到运行目录。实际 runs/pp2_r6 只完成了六步中的前两步：导出与清单校验有日志（test-results/ceva-pp2_r6/，清单 327/327 通过），GC 编到 prompt 图时（11-18 13:39）按长跑纪律终止（实施文档 17.17 记录了终止）；目录内无 ddr_core*、无蓝图、无 driver 日志、无 run/ 产物、无 verdict.txt、无 cascade_eval.*。但实施文档 1.3 声明「R6 一键链路经官方入口完整执行（runs/pp2_r6：导出→校验→GC 复用→布局→蓝图→driver→判定→评估）」，18.1 问题3 行声明「已修复」并把「官方链路端到端产物（verdict/cascade_eval）」列为证据——证据不存在。同根因的文档-事实偏差另有三处：18.2 称「GC 复用 pp2_e2e 编译产物（already_compiled 生效）」，实际 runs/pp2_r6/logs/04_gc.log 为冷编译（无 already 命中、约 25s/图）；I9 称「用当前 ELF 重跑 prompt:c0 直接测得 TIMER 口径消除修正需求」，实际该重跑已被 17.17 终止、state.json 的 prompt:c0 仍无 exec_cycles、correction 仍在生效；4.1 称「141+65=206 全过」，实测同口径四文件 132（35+97）加工作区 67 为 199。  
**根本原因**：官方链路执行被运行纪律中止后，文档回填先于事实收敛；R6 证据没有与 verdict/eval 产物的实际生成绑定（任何文本文件都能冒充判定），验收只能靠人工对盘核实——这正是 round01 问题 3/7 的根因（证据与产出脱钩）在文档维度的延续。  
**涉及文件**：docs/implement-multi-ceva-cascade-20261114.md（1.3、4.1、I9、18.1 问题3 行、18.2）、/data/fjg/ceva/runs/pp2_r6/（证据缺失本体）、scripts/run_full_pipeline.py（run_ceva 级联分支）  
**修改思路**：文档按事实改写：R6 证据缺口如实登记为待完成项（与 Q6/Q7 并列），18.1 问题3 的修复状态改为「部分修复（入口机制完成、端到端实证未完成）」；后续机时允许时以官方入口完整执行一次（段断点与 GC 清单可复用时增量可控），verdict、cascade_eval、driver 日志落在同一 run 目录作为 R6 证据；verify_cascade 写 verdict 时附带 run 指纹（state.json 段数与时间戳），使手工文件无法冒充运行产物；测试计数与回归数字以实际执行输出回填（本轮已代跑：2206 passed / 0 failed）。  

### 问题4（中）: TP2 加速比基线被截断，导致【加速比失真】
**问题说明**：设计 4.9 与 eval.py `_baseline_total` 自身 docstring 都要求基线「与实测范围等宽才能相除」（17.16 的 test_baseline_total_scoped_to_plan 正是为此而加）。实际 `_baseline_total`（cascade/eval.py L236-249）的步数推导只认 `kind=="decode_layers"` 段，TP2 蓝图的段类型是 attn_shard/mlp_shard/head_shard，推导恒得 steps=0。本次实际验证：steps=1 的 TP2 蓝图返回 1011=prompt+1 个 head，decode 相与后续 head 全部漏计；steps=10 时同样只剩 prompt+1 head。speedup_vs_single 的分母基线因此被截断，R7「TP2 体现并行加速」的方向判据在首次 TP2 评估时就会失真。  
**根本原因**：token 步数这一蓝图语义在 eval 内按 PP2 的段 kind 硬编码推导，缺少供 _baseline_total、predict_tp2、compose 共用的统一「相/步」抽象；限定范围的测试只写了 pp2 算例。  
**涉及文件**：cascade/eval.py（_baseline_total L236-249）、tests/test_cascade_eval.py（test_baseline_total_scoped_to_plan）  
**修改思路**：在 CascadeBlueprint 上提供统一步数推导（按 head 段或 phase 前缀分组），_baseline_total、predict_tp2 与 compose 的 tokens 计数一律改用它；补 TP2 算例测试（10 步蓝图基线=prompt+10 decode+11 head 的手推值）。  

### 问题5（中）: 单测把伪造 PASS 判定写进真实归档树，导致【伪证据残留】
**问题说明**：需求非功能项与 §5.5 要求性能/判定数字可溯源，「无法溯源到运行目录」属无效验收。实际 `tests/test_ceva_export.py::test_run_ceva_dispatches_cascade` 的 fake_run（L2425-2439 附近）把 verdict 写到 `Path(cwd)/runs/multi/results/verdict.txt`，而级联步的 cwd 是真实工作区 /data/fjg/ceva（`ceva_workspace()` 未被测试隔离）——测试每跑一次就在真实 runs/ 树留下一个 41 字节、无任何运行数据支撑的「结论: PASS」，其格式与 verify_cascade.py 的输出完全不符。本次实际验证：该文件 mtime 随本轮测试执行刷新（13:26→13:58），确认写入者就是该用例。这属 round01 问题 7（伪证据痕迹）同根因的复发：一个不可溯源的 PASS 判定躺在归档树里，极易被误当作 R6 证据。  
**根本原因**：单测对真实工作区路径的依赖未打桩（ceva_workspace/ceva_reference_dir 未 monkeypatch），fake 产物落进真实 runs/ 目录；runs/ 白名单机制只约束 git 忽略，不约束测试写入。  
**涉及文件**：tests/test_ceva_export.py（test_run_ceva_dispatches_cascade）、/data/fjg/ceva/runs/multi/results/verdict.txt（污染产物）  
**修改思路**：该用例 monkeypatch ceva_workspace（及 ceva_reference_dir）到 tmp_path，fake_run 只写临时目录；删除盘上已污染的 runs/multi/results/verdict.txt；同类排查所有引用 ceva_workspace 或向 runs/ 写入的测试，把「测试不得写真实工作区」固化为检查（可在 conftest 里对 workspace 路径写操作断言）。  

### 问题6（中）: 旧 TP2 归档未按声明标注不可复现，导致【误导性能结论】
**问题说明**：18.1 问题5 修复方式明确「删除 tp2_parallel.py，归档 tp2_result.json 标注不可复现」。实际 tp2_parallel.py 已删，但 runs/tp2_parallel/ 没有任何标注（无 notes.md、json 内无标注字段），tp2_result.json 仍裸呈「parallel: true、overlap 100%、speedup 1.99×」——该数据由已删除的旧控制块机制产生、当前代码不可复现，且跑的是 PP2 布局的 decode 段（round01 已指出），是 R5 并行收益最具误导性的假证据。同时 scripts/ 仍残留直接驱动已退役 pp2_tokens.py 的启动脚本 launch_pp2.sh、launch_resume.sh、pp2_resume.py（后者 import 路径笔误写成 /data/fjg/flagos-pim-compiler，已不可运行）与 m0_*.py 实验族，形成无真源指向的死入口。  
**根本原因**：round01 问题 7 的清理只处置了「脚本」维度，未同步处置脚本的历史产物与伴生启动器——「退役」缺少产物侧的落地动作，同一根因也会影响今后每次脚本清理。  
**涉及文件**：/data/fjg/ceva/runs/tp2_parallel/（tp2_result.json 等）、/data/fjg/ceva/scripts/{launch_pp2.sh,launch_resume.sh,pp2_resume.py,m0_*.py}  
**修改思路**：在 runs/tp2_parallel/ 落 notes.md（.gitignore 白名单已留档该名）声明「旧控制块机制产物、不可复现、不得作为 R5 证据」或直接删除该归档；删除三个退役启动脚本（m0_* 若保留须在文件头声明实验用途与真源指向）；把「退役脚本时同步清理或标注其历史输出与伴生启动器」固化为清理清单项。  

### 问题7（低）: 段后自检 hook 不适配 TP2 命名，导致【TP2 自检空转】
**问题说明**：18.1 问题7 声明「pp2_tokens.py 的段后自检（KV 指纹/diff/top5/app 日志解析）以可注入 hook 并入 driver（CLI 默认启用）」。实际 `_append_diagnostics`（cascade/driver.py L97-125）只认 `kv_core{N}/kv.bin` 与段名正则 `head:\d+|prompt:lm|decode:\d+:lm`；TP2 段的 KV 文件是 kv_L{n}.bin、head 段名是 head:{n}:c{m}，因此 KV 指纹链、diff、top5 在 TP2 路径全部不产生，diagnostics.jsonl 只剩 wall/md5 两类信息，自检能力在 TP2 上空转。  
**根本原因**：自检的段名与 KV 文件名约定按 PP2 口径硬编码，未从蓝图/段参数派生；execute_pair 手里已有 kv_layer 信息却没有传给 hook。  
**涉及文件**：cascade/driver.py（_append_diagnostics L97-125、execute_pair 的 diagnostics 调用 L529-534）、tests/test_cascade_driver.py（test_segment_diagnostics_hook 仅 PP2 用例）  
**修改思路**：diagnostics 增加 kv 文件名参数（execute_pair 按 kv_layer 传 kv_L{n}.bin），top5 的段名匹配覆盖 head_shard 命名；补一条 TP2 段的 diagnostics 断言用例，锁定两类策略的自检等价生效。  

## 三、核心目标完成情况

| 需求编号或目标 | 状态 | 证据 | 缺口或外部阻塞 |
|----------------|------|------|----------------|
| 目标1 多 CEVA 级联底座与全链路 | 部分完成 | R1-R4 实现并经 runs/pp2_v3 真机验证（11 token 与基线逐位一致；round01 直接复核、归档在盘，本轮未重跑仿真）；genesim/FlagTree 零改动（本轮 git status 核对） | R6 一键实证中断、证据缺失（问题3）；TP2 仿真未跑且运行时不可用（问题1/2）；「编译调优空间选择」按需求边界本期不接入（伪造步骤的 run_all.sh 已删） |
| 目标2 面向多 CEVA 级联的编排器 | 部分完成 | PP2 编排（任务拆分、映射、通信、串行流水、断点续跑）真机验证；编排真源唯一化（7 个伪证据脚本删除、pp2_tokens.py 退役声明）；TP2 编排代码与离线测试落地 | TP2 并行编排存在 layer 未接线的功能性缺陷（问题1）与 prompt 相续跑缺陷（问题2），无真机 tokens/时间线证据 |
| 目标3 多 CEVA 组合评估模型 | 部分完成 | 组合公式、实测+预测双模式、参数溯源、三层抽象齐全；标定同源已机制化（in_sample 强制标注，runs/pp2_v3/results/cascade_eval.json 实测带 in_sample=true 与限定语）；host 汇总按权威投影 33 段 | TP2 加速比基线截断（问题4）；out-of-sample 偏差验收需第二次独立 PP2 运行（Q7，长跑禁止，未实测不计缺陷）；TP2 预测仍为 assumption |
| R1 多进程级联底座 | 已完成 | M0 结论、一段一进程机制、双核分段 cycles（runs/pp2_v3 state.json） | 无 |
| R2 多核 DDR 布局 | 已完成 | 同基址多核布局、层过滤、KV 重编号、容量核对与测试；口径修正已回写设计 §8.6 | 无 |
| R3 多核切分导出 | 已完成 | pp2 bundle cascade 节、tp2 两核 129 图；清单校验 327/327（本轮核对 pp2_r6 导出日志）；97 个导出测试通过 | 无 |
| R4 PP2 token 对拍 | 已完成 | 11/11 token 逐位一致（复用 round01 对归档 argmax_results.bin 的直接复核） | 无（执行形态缺口计入 R6） |
| R5 TP2 并行编排与收益 | 未完成 | 导出+GC+离线测试（并行下发/归约/断点）就绪 | 运行时存在严重缺陷（问题1/2），未真机执行、无 tokens 与时间线证据（Q6，长跑禁止） |
| R6 一键命令入口 | 部分完成 | 参数链、干跑、单核默认恢复（本轮测试验证）、入口唯一化完成 | 官方链路端到端执行中断于 GC、verdict/cascade_eval 未产生而文档称完成（问题3） |
| R7 组合评估模型 | 部分完成 | 模型、溯源、同源标注机制、实测模式结构分解（pp2_v3 产物齐全） | out-of-sample 偏差无第二次独立运行数据（Q7）；TP2 基线截断（问题4）；预测模式对 pp2_v3 的 0.8% 偏差已正确标注为 in_sample 不可作验收依据 |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
|--------------|----------|------|----------------------|
| pim `pytest tests/test_cascade_{plan,driver,eval}.py -q` | 本次实际验证 | 35 passed | 工作区未提交改动；pytorch env（torch 2.9.1）；从仓库根目录运行 |
| pim `pytest tests/test_ceva_export.py -q` | 本次实际验证 | 97 passed（从非仓库目录运行时 3 个相对路径用例假失败，须从根目录跑） | 同上 |
| pim `pytest tests/ -q -k "not llama2_7b"`（全量回归） | 本次实际验证 | 2206 passed / 9 skipped / 0 failed（25:03） | 同上；填补实施文档 4.3「待回填」项；历史基线 2190 passed |
| 工作区 `pytest tests/ -q`（/data/fjg/ceva） | 本次实际验证 | 67 passed | 工作区未提交改动；pytorch env |
| TP2 layer 接线合成实验（2 层蓝图 + 记录型 Runner） | 本次实际验证 | 所有 attn/mlp 段收到 layer='None'（问题1） | /tmp 合成夹具，离线 |
| `_baseline_total` TP2 截断实验（steps=1 蓝图） | 本次实际验证 | 返回 1011=prompt+1 head，decode 相漏计（问题4） | /tmp 合成夹具，离线 |
| TP2 fanout 覆盖输入文件实验（2 层蓝图） | 本次实际验证 | 跑一层后 h0.bin 首 8 元素由全零变 12.0（问题2） | /tmp 合成夹具，离线 |
| runs/multi 伪 verdict 写入复现（mtime 跟踪） | 本次实际验证 | 文件 mtime 随本轮测试执行刷新（13:26→13:58），写入者为 test_run_ceva_dispatches_cascade（问题5） | /data/fjg/ceva/runs/multi/results/verdict.txt |
| runs/pp2_r6 产物与入口日志核对 | 本次实际验证 | 仅有导出/清单/GC 半程产物与日志；无 ddr_core*、蓝图、driver 日志、verdict、cascade_eval（问题3） | runs/pp2_r6、test-results/ceva-pp2_r6 |
| pp2_v3 评估产物核对（in_sample/host/修正溯源） | 本次实际验证 | cascade_eval.json 带 in_sample=true 与限定语、host 33 段；correction 带 value/derived_from/measured_at；仓库默认 scaling json 声明 source_runs | runs/pp2_v3/results |
| `bash -n` run_cascade.sh、run_ceva_pipeline.sh | 本次实际验证 | 通过 | 工作区未提交改动 |
| genesim / FlagTree 零改动 | 本次实际验证 | git status 干净 | 两仓库工作区 |
| 11 token 逐位一致、单核导出 md5 抽查（3 图）、65 图全量 md5 | 复用已有验证 | round01 直接复核/实施文档 17.13 声明通过 | round01 评审记录；代码其后仅 18.x 修复，未重跑仿真 |
| TP2 四类分片图过厂商 GC | 复用已有验证 | 实施文档 §九/十差分证据链；bundles/tp2_e2e complete=true（round01 核对） | 同上 |
| 2 CEVA PP2/TP2 仿真、llama2_7b 慢用例、第二次独立 PP2、TP2 端到端 | 仅核对实现但未执行 | 长跑禁止（用户 2026-11-18 纪律）；只核对代码机制与归档产物，不据此宣称通过 | — |

　　未覆盖范围及限制：本轮未执行任何 TCS 仿真与小时级命令，性能与 token 结论基于归档产物（runs/pp2_v3、single_ours、ours_e2e、pp2_e2e、tp2_e2e）与两轮评审的交叉核对；TP2 运行时与评估的问题（1/2/4/7）均以离线合成实验或代码机制核对取证，真机层面的最终确认待 Q6 排期；「未重跑长仿真」本身不作为问题列列据。round01 修复的抽查结论：7 项根因修复到位，问题 3 未完成、问题 5/7 部分完成，其缺口与复发原因已分别落在问题 3、1、2、6、7 中并给出完整修复范围。
