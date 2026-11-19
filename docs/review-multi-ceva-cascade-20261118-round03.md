# 代码评审报告：多 CEVA 级联仿真与 Llama2 7B 全流程性能评估（第 3 轮）
> 文档编号：review-multi-ceva-cascade-20261118-round03
> 创建日期：2026-11-18
> 关联需求文档：request-multi-ceva-cascade-20261114
> 关联设计文档：design-multi-ceva-cascade-20261114
> 关联实施文档：implement-multi-ceva-cascade-20261114（第十三节无效，第十六至十八节为修正与 round01/round02 修复记录）

## 一、评审概述
　　本轮覆盖四个仓库的未提交改动（pim-compiler 的 cascade 三模块、导出链与入口共约 +1150 行；/data/fjg/ceva 的段式 runner、ddr_manager、run_cascade 与脚本测试共约 +840 行），并复核 round02 七个问题与 round01 十个问题的修复是否覆盖根因。genesim 与 FlagTree 经 git status 核对仍为零改动，符合设计声明。复核结论：round02 的 R2-1（TP2 层接线）、R2-2（prompt 相断点快照）、R2-3（verdict 指纹绑定）、R2-4（步数口径统一）、R2-6（退役产物标注）、R2-7（自检适配 TP2）均已按根因修复并有真实断言的测试锁定，本次全部补跑通过；R2-5（测试写伪证据）的修复只收敛了工作区路径——同一条 fake_run 仍把 41 字节伪 PASS verdict 写进 pim-compiler 自身的 runs/ 归档树，本轮以 mtime 跟踪复现确认，列为问题1（复发）。此外新列 4 个问题：PP2 GC 缓存复用的设计口径未落地且未按设计 §8 登记偏离（问题2，round01 问题9 同根因类的残留）、评估外部标定缺 source_runs 强制形成自证旁路（问题3，round01 问题1 机制的一个缺口）、verify_cascade 指纹强度与声明不符（问题4）、tp2 核间一致性检查缺宽度互补校验（问题5）。共列问题 5 个：中 3、低 2，无严重/高级问题。PP2 主链路结论不变：11 token 与单核基线逐位一致（本次直接复核归档 argmax_results.bin），评估产物带 in_sample 标注，实施文档与代码、归档产物的对应关系经逐项核对一致。测试本次实际验证：pim 级联三件套+路径 51 通过、导出 97 通过、工作区 69 通过，未跑任何小时级命令。

## 二、修改建议

### 问题1（中）: 单测伪 PASS verdict 写进仓内 runs/，导致【伪证据复发】
**问题说明**：需求 §5.5 把「性能数字无法溯源到运行目录」列为无效验收，round02 问题5 已判定「测试写伪造 PASS 进真实归档树」为问题并声明修复（实施文档 I14：「fake_run 只写临时目录」）。实际 `tests/test_ceva_export.py::test_run_ceva_dispatches_cascade` 的 fake_run（tests/test_ceva_export.py:2442-2446）对**每一条** `_run` 调用都执行 `verdict = Path(cwd)/"runs"/"multi"/"results"/"verdict.txt"` 的写入，而级联分支的导出步与清单校验步以 `cwd=_REPO_ROOT` 调用（scripts/run_full_pipeline.py:751、757）——round02 修复只 monkeypatch 了 `ceva_workspace`，只改变了 run_cascade 步（cwd=工作区）的落点，`_REPO_ROOT` 的两次调用仍把伪 verdict 写进 pim-compiler 自身的 `runs/multi/results/verdict.txt`。本次实际验证：该文件现存 41 字节「结论: PASS」内容与 fake_run 写入串完全一致；单独执行该用例一次，文件 mtime 由 16:01:30 变为 16:07:06，写入者确认无误。且 pim 仓 .gitignore 未忽略 `runs/`（`git check-ignore` 不命中），`git add .` 会把伪 verdict 提交进仓。  
**根本原因**：round02 修复按「当时发现的路径」（工作区 runs/）打点，未对「fake_run 对任意 cwd 均落盘」这一行为本身收口；conftest 写守卫（tests/conftest.py:15-45）只保护 `paths.ceva_workspace()` 前缀，pim 仓自身的 runs/ 归档树不在守卫范围内，同一根因还使守卫漏掉 `os.replace`/`os.link`/`os.truncate` 等审计事件。实施文档 I14「fake_run 只写临时目录」的声明与事实不符，属修复声明未核对写入路径全集。  
**涉及文件**：tests/test_ceva_export.py:2442-2446（fake_run）、scripts/run_full_pipeline.py:751/757（cwd=_REPO_ROOT 的两步）、tests/conftest.py:15-45（守卫范围）、runs/multi/results/verdict.txt（污染产物）、docs/implement-multi-ceva-cascade-20261114.md（I14 行）  
**修改思路**：从行为根源收口——fake_run 不再无条件落盘，改为按命令内容判定（仅 `run_cascade.sh` 那次调用写 verdict，或完全不写：`ceva_verdict` 已打桩，对 verdict 文件的断言改为对 commands 列表的断言）；同时把 conftest 守卫的覆盖面从「工作区前缀」扩为「工作区与仓内 runs/ 两处真实归档树前缀」，并把审计事件补齐 `os.replace`/`os.link`/`os.truncate`，加一条「仓内 runs/ 写入即失败」的守卫生效用例（与既有 test_ceva_workspace_write_guard_fires 同构）；删除盘上已污染的 runs/multi 并更正实施文档 I14 的声明。

### 问题2（中）: PP2 GC 缓存复用未落地且未登记偏离，导致【设计判据失据】
**问题说明**：设计 3.2 承诺「PP2 两核指向同一 bundle 子集可全量复用单核 GC 缓存」、设计 7.2 把「pp2 路径 GC 缓存命中率 100%（65 图全部 already_compiled）」列为干跑验收判据。实际官方链路无法满足：runs/pp2_r6/logs/04_gc.log 为全量冷编译（本次核对：0 次 already 命中，decode 图约 25s/张、prompt 图冷编译达 1886s 才被终止），播种方案（软链复用 ours_e2e/gmls）因会写穿软链污染归档 manifest 已撤回（实施文档 §九.4），此后无替代机制，compile_gml 的缓存以 run 目录为键、跨 run 不共享。实施文档 18.2.3 已如实记录该事实，但设计 §8（round01 问题9 修复确立的「此后每处设计偏离走该节登记」机制）没有对应条目，按设计文档验收会直接冲突。同根因的次要实例：设计 5.1 列明的 `06_ddr_memory.sh` 透传改动项未实施（run_cascade.sh:82-87 直调 ddr_manager 绕开），同样未登记。  
**根本原因**：播种方案撤回时只做了「止血」（停止污染归档），没有回到设计登记点补记「GC 缓存复用」这一验收口径的失效与替代路径——round01 问题9 修复建立的偏离登记机制未被执行到位，设计 3.2/7.2 的承诺与实现脱节，后续 Q8 的「GC 清单可复用时增量可控」实际依赖手工拷贝而非机制。  
**涉及文件**：docs/design-multi-ceva-cascade-20261114.md（§3.2/§7.2/§8）、/data/fjg/ceva/run_cascade.sh:59-69（每 run 全量 GC）、/data/fjg/ceva/scripts/compile_gml.py（run_key 以 run_dir 为键）、/data/fjg/ceva/runs/pp2_r6/logs/04_gc.log（冷编译证据）  
**修改思路**：在设计 §8 追加修正条目（GC 缓存复用不可达的实测依据：播种写穿软链、run_key 按 run 目录隔离），把 7.2 的干跑判据改写为可实现口径；机制侧给 compile_gml 增加只读的跨 run 缓存复用入口（如 `--cache-from <run>` 校验 GML 摘要一致后复用清单与产物，或按 bundle generation_digest 共享 GC 缓存目录），使「复用」不依赖手工拷贝；06_ddr_memory.sh 的绕行决策一并写入该条目。

### 问题3（中）: 外部标定缺 source_runs 强制，导致【自证旁路】
**问题说明**：需求 §5.4/§5.5 与 round01 问题1 的修复要求「标定来源登记是机制不是约定」。实际 `cascade/eval.py` 的 `_layer_scaling`（cascade/eval.py:258-261）对 `--scaling-json` 外部标定分支执行 `data.setdefault("source_runs", [])`——外部文件不写 source_runs 时静默按空表处理，evaluate 随即判定 `in_sample=False`、偏差标注「out_of_sample 可作为验收」。也就是说，把某次运行自拟合的系数存成不带 source_runs 的 json 再用 `--scaling-json` 喂回对该运行评估，机制上会被宣称为 out-of-sample 验收依据——这正是 round01 问题1 要堵住的自证路径，在 override 分支重新打开。对照同文件 `_load_correction`（cascade/eval.py:112-126）对缺 provenance 字段直接拒绝的口径，两处同类输入的强制程度不一致；现有测试只覆盖 run 本地标定路径（test_evaluate_marks_in_sample_calibration），无 override 缺 source_runs 的用例。  
**根本原因**：in_sample 机制按「标定文件的三种来源路径」分别实现，run 本地与仓库默认两处默认登记来源，唯 override 分支把「未声明」默认为「无关」；缺少「凡标定输入必须声明来源，未声明即拒绝」的统一约束。  
**涉及文件**：cascade/eval.py:258-261（override 分支）、cascade/eval.py:112-126（correction 的对照口径）、tests/test_cascade_eval.py（缺用例）  
**修改思路**：与 `_load_correction` 对齐为同一原则——外部 scaling json 缺 `source_runs`（或显式 `assumption` 声明）时 raise，迫使操作者声明来源；补「override 无 source_runs 拒绝」「override 声明含被评估 run 时判 in_sample」两条用例。合法路径不受影响：真正的外部标定文件本就应携带来源（仓库默认文件即为范例）。

### 问题4（低）: verify_cascade 指纹强度弱于声明，导致【防伪口径失真】
**问题说明**：verify_cascade.py 文件头与测试注释声明「手工文件无法冒充运行产物」（scripts/verify_cascade.py:7-8），实际 run 指纹检查（scripts/verify_cascade.py:107-121）只校验 state.json 的存在性，段数仅打印不校验——测试 test_verify_cascade_pass_binds_state_fingerprint 本身就是用手工构造的目录（自写 tokens、counters、state.json）拿到 PASS，证明「带 state.json 的伪造目录」仍可通过全部检查。指纹机制（round02 问题3 修复引入）挡住的只是「裸 verdict 无运行目录」这一最弱形态，与声明的防伪强度不符。  
**根本原因**：指纹绑定停留在「存在性」而非「一致性」——verdict、tokens 与 state 三者互相之间的数量关系（head 段数应不少于 token 数、段数与 --steps 同量级）没有交叉校验，防伪声明按机制意图而非实际检查面书写。  
**涉及文件**：/data/fjg/ceva/scripts/verify_cascade.py:107-121、/data/fjg/ceva/tests/test_verify_cascade.py  
**修改思路**：把指纹从存在性升级为一致性：校验 state.done 中 head 段数与 argmax_results.bin 的 token 数相等、段数与传入 steps 同量级（容忍 EOS 提前停），不一致判 FAIL；同时把文件头与测试注释的声明改准确（「无 state 的目录判 FAIL」，不宣称全部伪造不可冒充）。

### 问题5（低）: tp2 核间一致性检查缺宽度互补，导致【对称错切漏检】
**问题说明**：设计 4.5 要求 tp2 的清单校验「改为核间一致性检查（两核同位置图的结构化字段对称、宽度之和等于全宽）」。实际 `_check_tp2_core_consistency`（scripts/verify_gml_artifact.py:1013-1036）只做两件事：双核 (stage, layer, shard_kind) 键集合相等、同位置图同名文件尺寸相等。若两核因切分 bug 拿到相同的半份（例如 core_id 传错使 core1 也切前半），两核产物完全对称、检查通过，而拼接结果宽度只有全宽的一半——设计明确要求的「宽度之和等于全宽」没有被任何检查覆盖。当前导出侧有 shard_model 形状测试兜底，但该检查的设计定位是清单校验层的独立防线。  
**根本原因**：核间一致性实现时把「对称」当成了充分条件，忽略了设计原文的第二个必要条件（互补）；缺少对「对称但不互补」这一错误形态的用例（现有 test_verify_tp2_core_asymmetry_fails 只测了不对称与缺图两种形态）。  
**涉及文件**：scripts/verify_gml_artifact.py:1013-1036、tests/test_ceva_export.py（tp2 校验用例）  
**修改思路**：在核间一致性检查中加入宽度互补断言：对 attn/mlp/head 的关键 IO 尺寸（可从 IO_info.txt 或清单记录读出），校验两核同位置图之和等于单核全宽（hidden 4096、head 词表 32000）；补一条「两核同错切一半」的构造用例断言判 FAIL。若清单不携带尺寸，可在导出时把 local 宽度写入 manifest 的 graphs 条目，避免校验层反向解析 GML。

## 三、核心目标完成情况

| 需求编号或目标 | 状态 | 证据 | 缺口或外部阻塞 |
|----------------|------|------|----------------|
| 目标1 多 CEVA 级联仿真底座与全链路 | 部分完成 | R1-R4 实现并经 runs/pp2_v3 真机验证；11 token 与基线逐位一致本次直接复核（读 run/argmax_results.bin 比对，11/11 相同）；genesim/FlagTree 零改动（本次 git status 核对） | R6 官方链路端到端实证缺（Q8：runs/pp2_r6 仅有导出/校验/GC 半程产物，无 run/、verdict、cascade_eval，本次核对）；TP2 仿真未跑（Q6）；「编译调优空间选择」按需求边界本期不接入 |
| 目标2 面向多 CEVA 级联的编排器 | 部分完成 | cascade/plan.py 蓝图（段+搬运表+plan_text+权威投影）与 driver（PP2 串行编排真机验证；TP2 并行下发/host 归约/断点/时间线机制，本次 51 用例含 layer 递增与续跑逐字节断言全绿）；编排真源唯一（pp2_tokens.py 已声明退役） | TP2 并行收益无真机 tokens 与时间线重叠证据（Q6，长跑禁止，未实测不计缺陷）；R5 判据未满足 |
| 目标3 多 CEVA 组合评估模型 | 部分完成 | 组合公式（PP 串行链/TP 层同步）、实测+预测双模式、三层抽象 [2,1,1]、参数溯源落盘；pp2_v3 归档评估带 in_sample=true 与限定语（本次直接读 cascade_eval.json 核对）；correction 带 value/derived_from/measured_at 且推导可由 state.json 复算（499910167−322945852=176964315，本次验算一致） | out-of-sample ≤20% 偏差验收需第二次独立 PP2 运行（Q7，长跑禁止）；TP2 预测为 assumption 待实测复核；--scaling-json 旁路缺来源强制（问题3） |
| R1 多进程级联底座 | 已完成 | M0 结论（一段一进程+argv 命令）、run_tools_step 受控会话、双核分段 cycles 归档（pp2_v3 state.json 44 条，本次抽查） | 无 |
| R2 多核 DDR 布局 | 已完成 | 同基址多核布局、层过滤、KV 重编号、embedding_row、容量核对（代码与 test_ddr_multi 本次全绿；判据已按设计 §8.6 改写） | 无 |
| R3 多核切分导出 | 已完成 | pp2 bundle cascade 节+65 图 complete=true；tp2 两核各 129 图 complete=true（本次读四个 bundle.json 核对）；97 个导出测试本次通过 | 无 |
| R4 PP2 token 对拍 | 已完成 | 11/11 token 逐位一致（本次直接复核归档 bin） | 无（执行形态缺口计入 R6） |
| R5 TP2 并行编排与收益 | 未完成 | 导出+GC+运行时代码与离线测试就绪（含 R2-1/R2-2 修复，本次测试锁定）；ELF 重编 815788B（本次核对，2026-11-18 14:58） | 真机 tokens/时间线证据未取得（Q6，长跑禁止）；R5 两项判据均无有效证据 |
| R6 一键命令入口 | 部分完成 | 参数链、干跑、单核默认恢复（test_run_ceva_pipeline_sh_defaults_to_single 本次通过）、入口唯一化；verdict 指纹绑定 | 官方链路端到端中断于 GC、verdict/cascade_eval 未产生（Q8）；GC 缓存复用设计判据不可达（问题2） |
| R7 组合评估模型 | 部分完成 | 模型、溯源、in_sample 机制、实测模式结构分解产物齐全 | out-of-sample 验收未取得（Q7）；标定旁路（问题3）；指纹强度（问题4） |

## 四、验证边界

| 验证项或命令 | 证据状态 | 结果 | 来源、代码基线与环境 |
|--------------|----------|------|----------------------|
| pim `pytest tests/test_cascade_{plan,driver,eval}.py tests/test_paths.py -q` | 本次实际验证 | 51 通过（含 R2-1/R2-2/R2-4/R2-7 的锁定用例与工作区写守卫生效用例） | 工作区未提交改动；pytorch env（torch 2.9.1），从仓库根目录运行 |
| pim `pytest tests/test_ceva_export.py -q` | 本次实际验证 | 97 通过（含入口分派、外壳默认单核、tp2 清单与核间一致性用例） | 同上 |
| 工作区 `pytest tests/ -q`（/data/fjg/ceva） | 本次实际验证 | 69 通过（含 test_verify_cascade 指纹 2 例、test_ddr_multi 5 例） | 工作区未提交改动；pytorch env |
| 问题1 复现（单跑 test_run_ceva_dispatches_cascade 跟踪 runs/multi mtime） | 本次实际验证 | mtime 16:01:30→16:07:06，伪 verdict 内容与 fake_run 串一致；pim .gitignore 不忽略 runs/ | 本仓工作区；/tmp 未产生中间物 |
| pp2_v3 tokens 复核（读归档 argmax_results.bin 比对基线） | 本次实际验证 | 11/11 逐位一致 | 归档 runs/pp2_v3（2026-11-17/18 运行），未重跑仿真 |
| pp2_v3 评估产物核对（in_sample/correction 溯源/host 33 段） | 本次实际验证 | in_sample=true 与限定语在盘；correction 推导可复算；host.wall_s=34669s/33 段 | runs/pp2_v3/results/*.json |
| bundle 清单核对（pp2_e2e 与 tp2_e2e core0/core1） | 本次实际验证 | complete 均为 true（65/129/129 图，cascade 节正确） | /data/fjg/ceva/bundles |
| `bash -n` run_ceva_pipeline.sh、run_cascade.sh；set -e 下 `&&` 循环行为实测 | 本次实际验证 | 全部通过；`[[ $c -eq 0 ]] && …` 在 set -euo pipefail 循环中不触发退出 | 工作区未提交改动 |
| pp2_r6 产物核对（logs/04_gc.log 冷编译、无 run//verdict/cascade_eval、state 44 段指向本 run） | 本次实际验证 | 全量冷编译（0 already，prompt 图 1886s 终止）；verdict/cascade_eval 不存在；与实施文档 Q8/18.2.3 声明一致 | runs/pp2_r6 |
| genesim / FlagTree 零改动 | 本次实际验证 | 两仓库 git status 干净 | 两仓库工作区 |
| round02 修复抽查（R2-1/R2-2/R2-3/R2-4/R2-6/R2-7 的代码与测试逐项读核） | 本次实际验证 | 全部按根因落地：execute_pair 传层+run() 缺层抛错+app 侧 -1 显式失败；prompt_input_core0.bin 快照；verdict 指纹；decode_steps 统一；tp2_parallel/notes.md 与 m0 头声明、三个死入口已删；diagnostics kv_name | 本轮评审记录；驱动/计划/评估/app 源码与测试文件 |
| pim 全量回归 `pytest -k "not llama2_7b"`（2209/1/9 后定向修复） | 复用已有验证 | 实施文档 18.3 声明通过；本轮未重跑（级联+导出+路径 148 例本次全绿，未改这些代码） | 实施文档 18.3；round02 后代码基线与本轮一致（本轮无代码改动） |
| 单核导出逐字节不变（md5 抽查 3 图、65 图全量） | 复用已有验证 | round01 抽查 3 图 SAME（含 from_fx 改名后的口径）；全量比对未执行 | round01 评审记录；其后仅 18.x 修复，export 主链未再改 |
| TP2 四类分片图过厂商 GC | 复用已有验证 | 实施文档 §九/十 probe2/probe3 差分链；tp2_e2e complete=true 本次核对 | 同上 |
| 2 CEVA PP2/TP2 仿真、llama2_7b 慢用例、第二次独立 PP2（out-of-sample）、TP2 端到端 | 仅核对实现但未执行 | 长跑禁止（用户 2026-11-18 纪律）；只核对代码机制与归档产物，不据此宣称通过，也不作为问题判据 | — |

　　未覆盖范围及限制：本轮未执行任何 TCS 仿真、全量 GC 与全量回归，性能与 token 结论基于归档产物（runs/pp2_v3、single_ours、ours_e2e、pp2_e2e、tp2_e2e、pp2_r6）的直接读取与交叉验算，这些产物由实施文档第十六至十七节记录的运行产生、其后代码有 18.x 修复（ELF 已重编）而未重跑对应段；单核导出「逐字节不变」仅有 3 图抽查加测试套件佐证，65 图全量比对未执行（沿用 round01 口径）；问题1 的复现依赖运行单个测试用例（1.2 秒），不属小时级运行。前两轮问题的复发抽查结论：round02 七项中六项根因修复到位，R2-5 在仓内侧路径复发（问题1）；round01 十项中标定溯源机制存在 override 旁路（问题3）、设计偏离登记机制存在 GC 缓存一项漏登（问题2），其余（in_sample 标注、correction 溯源、单核默认、权威投影、归档留档、设计 §8.6-8.8 回写）均确认在盘。
