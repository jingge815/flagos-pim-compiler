# 实施进展文档：多 CEVA 级联仿真与 Llama2 7B 全流程性能评估
> 文档编号：implement-multi-ceva-cascade-20261114
> 创建日期：2026-11-14
> 关联需求文档：request-multi-ceva-cascade-20261114
> 关联设计文档：design-multi-ceva-cascade-20261114
> 关联评审文档：review-multi-ceva-cascade-20261118-round01（2026-11-18，round01）
> 关联评审文档：review-multi-ceva-cascade-20261118-round02（2026-11-18，round02）
> 关联评审文档：review-multi-ceva-cascade-20261118-round03（2026-11-18，round03，本轮依据）
> 关联项目：flagos-pim-compiler、/data/fjg/ceva 工作区
> 起始基线：main @ a94534f（pim-compiler，仅两份 docs 未跟踪）；main @ 1a056c2（/data/fjg/ceva，.gitignore 已有改动）

## 一、实施概述
### 1.1 任务范围
　　设计 5.1 清单的 T1-T13：cascade 三模块（蓝图/驱动/评估）、PP2/TP2 多核导出、verify tp2 分支、入口分派、工作区受控会话/ddr_manager 多核化/compile_gml 参数化/run_cascade 脚本链/段式 runner；以及 round01、round02、round03 评审问题的确认与修复（见第十八节）。
### 1.2 完成标准
　　设计所有任务项覆盖、对应测试通过、全量回归无新增失败（TDD 三条同时满足）；评审问题全部确认、成立项全部修复。
### 1.3 实施结论
总体完成状态：**部分完成**。T1-T13 全部实现并通过单测；2 CEVA PP2 真机 11 token 逐位一致（runs/pp2_v3 归档）；round01 评审 10 个问题、round02 评审 7 个问题已确认修复；round03 评审 5 个问题（中 3、低 2）全部确认成立并按根因修复：伪 PASS verdict 仓内写入复发（fake_run 按命令判定落盘 + 守卫扩至仓内 runs/ 并补齐审计事件）、GC 缓存复用补机制与设计偏离登记（compile_gml/run_cascade 的 `--cache-from` 只读复用 + 设计 §8 第 9 条 + 7.2 判据改写）、外部标定缺 source_runs 拒绝、verify_cascade 指纹升级为一致性校验（head 段数与 token 数交叉核对，真实归档 pp2_v3 复核通过）、tp2 核间一致性补宽度互补（导出清单带 shard_range，两核同切一半判 FAIL）。R6 官方链路端到端已于 2026-11-19 实测完成（run 名 ours_e2e，2 核 PP2、冷 GC 65/65、33 段、11 token 与基线逐位一致，verdict PASS，见 18.4），Q8 关闭。剩余缺口为 TP2 端到端真机运行（R5 的时间线与 token 证据，Q6）与评估模型 out-of-sample 偏差验收（需第二次独立 PP2 运行，Q7），二者均需小时级运行、当前未实测（长跑纪律）。

核心目标完成度：

| 核心目标 | 状态 | 当前证据 |
| --- | --- | --- |
| 目标1 多 CEVA 级联底座与全链路 | 部分完成 | R1-R4 已完成（R1-R4 见 5.1，R4 复用 round01 对归档 argmax 的逐位复核）；R6 已完成：官方入口 2026-11-19 端到端实测（run 名 ours_e2e，2 核 PP2、冷 GC 65/65、33 段、11 token PASS，verdict/cascade_eval/driver 日志齐全，见 18.4，Q8 关闭）；GC 跨 run 复用机制 round03 补齐（`--cache-from` 只读复制，设计 §8.9 登记、7.2 判据改写为可达口径，单测锁定）；「编译调优空间选择」按需求边界本期不接入，伪造完成痕迹的 run_all.sh 已删除 |
| 目标2 面向多 CEVA 级联的编排器 | 部分完成 | PP2 编排真机验证（33 段、token 对拍、断点续跑）；TP2 运行时（cmd=4/5/6 + 并行下发 + host 归约 + 按层 KV 快照 + prompt 相断点快照）离线测试与交叉编译验证齐备，round02 修复层接线与续跑缺陷后无已知功能性缺陷；tp2 导出清单 round03 起带分片区间并核验宽度互补（对称错切在清单校验层可检出）；真机端到端未跑（Q6，未实测·长跑禁止） |
| 目标3 多 CEVA 组合评估模型 | 部分完成 | 组合公式/参数溯源/三层抽象/实测+预测双模式齐全；标定与验证同源已机制化（in_sample 强制标注、不计入验收），round03 关闭外部标定缺来源的自证旁路（override 缺 source_runs 即拒绝，与 correction 的 provenance 强制同口径）；TP2 加速比基线口径已统一（round02 问题4）；out-of-sample 偏差验收待第二次独立运行（Q7，未实测·长跑禁止） |

## 二、任务完成清单
| 任务ID | 任务描述 | 对应设计模块 | 完成状态 | 备注 |
|--------|----------|--------------|----------|------|
| T1 | cascade/plan.py 蓝图与 ControlBlock 镜像 | 4.1/4.6 | 完成 | 含段名归一化与权威投影（round01 问题10） |
| T2 | export_ceva_bundle 加 strategy、pp2 cascade 节 | 4.4 | 完成 | 3 用例 |
| T3 | export_gml.py --num-ceva/--strategy | 3.2 | 完成 | 3 用例 |
| T4 | shard_model+四分片 wrapper+shard_ceva_spec+tp2 整包+vendor_align=False | 4.5 | 完成 | 10 用例 |
| T5 | verify_gml_artifact tp2 分支+核间一致性 | 4.5 | 完成 | 2 用例；round03 补宽度互补检查（清单带 shard_range，两核同切一半判 FAIL） |
| T6 | cascade/driver.py PP2/TP2 驱动+产物落盘 | 4.6/4.7 | 完成 | TP2 已改段式进程版（round01 问题5）；round02 补层接线与 prompt 相断点快照（R2-1/R2-2）；离线测试覆盖并行下发/归约/断点/层递增/续跑 |
| T7 | cascade/eval.py 组合模型+预测模式+溯源 | 4.9 | 完成 | 含标定同源标注与修正溯源机制（round01 问题1/2） |
| T8 | run_full_pipeline 多核分派+干跑参数 | 4.8 | 完成 | 2 用例；外壳默认已恢复单核（round01 问题4） |
| T9 | run_tools_step.py 受控会话 | 4.1 | 完成 | 2 用例（假 target） |
| T10 | ddr_manager 多核化 | 4.3 | 完成 | 3 用例；同基址口径已回写设计 §8（round01 问题9） |
| T11 | compile_gml --gmls-name | 5.1 | 完成 | 既有 20 用例全绿；round03 补 --cache-from 跨 run 只读复用（4 用例，设计 §8.9） |
| T12 | 段式 runner（STEPWISE C++ + CMake + 02_build_runner） | 4.2 | 完成 | 第七节实现；round01 补 TP2 cmd=4/5/6 与分片输出尺寸 |
| T13 | run_cascade.sh + driver/eval CLI 入口 | 4.8 | 完成 | 入口机制与测试锁定见第十八节；round03 补 --cache-from 透传；官方链路端到端实证待 Q8（中断于 GC，未实测·长跑禁止） |

## 三、代码改动详情
### 3.1 改动总览
- 新增文件数：11（cascade/{__init__,plan,driver,eval}.py、tests/test_cascade_{plan,driver,eval}.py、/data/fjg/ceva 的 run_tools_step.py、tests/test_{ddr_multi,run_tools_step,verify_cascade}.py、run_cascade.sh）
- 修改文件数：7（gml_bridge/ceva.py、contracts/ceva.py、scripts/{export_gml,verify_gml_artifact,run_full_pipeline}.py、tests/test_ceva_export.py、/data/fjg/ceva 的 llama2_ddr_manager.py、compile_gml.py）
- round01 修复新增：1（cascade/layer_scaling_llama2_pp2.json 数据文件）；修改 10（cascade 三模块、run_ceva_pipeline.sh、llama2_npm{,_app}.cpp 与其头文件、run_cascade.sh、.gitignore、tests/test_gitignore.py）；删除 7（工作区 pp2_chain.py、run_eval.py、final_report.py、tp2_parallel.py、pp2_monitor.sh、launch_pp2_full.sh、run_all_goals.sh 与 pim 侧 scripts/run_all.sh）；归档恢复 1（runs/ours_full 自 git 2d18762 重建并补 ARCHIVE.md）。
- round02 修复新增：2（工作区 tests/test_verify_cascade.py、runs/tp2_parallel/notes.md 归档标注）；修改 12（cascade/{driver,plan,eval}.py、tests/{conftest,test_paths,test_cascade_driver,test_cascade_eval,test_ceva_export}.py、工作区 verify_cascade.py、llama2_npm_app.cpp、m0_{probe,diag,isolate,raw}.py 头声明）；删除 4（工作区 launch_pp2.sh、launch_resume.sh、pp2_resume.py 三个死入口与伪证据目录 runs/multi）。
- round03 修复修改：pim 侧 7（tests/conftest.py 守卫扩面与补事件、tests/test_paths.py、tests/test_ceva_export.py、tests/test_cascade_eval.py、cascade/eval.py、scripts/verify_gml_artifact.py、gml_bridge/ceva.py）；工作区 6（scripts/compile_gml.py、scripts/verify_cascade.py、run_cascade.sh 与 tests/{test_compile_gml,test_verify_cascade,test_run_cascade}.py）；设计文档 §7.2/§8（GC 缓存复用偏离登记）；删除 pim 侧污染产物 runs/multi（复发后再次删除并根治写入源）。
- 核心改动说明：cascade 包承载蓝图/驱动/评估三层；ceva.py 加 strategy 导出路径（single 逐字节不变）；工作区做仿真侧适配；round01 把编排真源收敛到 cascade.driver、TP2 运行时落段式进程口径、评估标定同源与修正溯源机制化、入口默认恢复单核等价；round02 补齐 TP2 运行时层接线与 prompt 相断点快照、评估步数口径统一、判定指纹绑定与测试工作区写保护；round03 根治测试伪证据写入（守卫覆盖仓内 runs/ 并按命令判定 fake 产物）、补 GC 跨 run 只读复用机制、关闭标定 override 自证旁路、判定指纹升级一致性、tp2 清单校验补宽度互补。

### 3.2 详细改动清单
| 文件路径 | 改动类型 | 改动内容简述 | 核心函数/类 | 对应任务ID |
|----------|----------|--------------|-------------|------------|
| cascade/plan.py | 新增 | 蓝图三 dataclass、from_bundle、plan_text、ControlBlock 镜像、段名归一化与权威投影 | CascadeBlueprint/from_bundle/canonical_seg_id/authoritative_timing | T1 |
| cascade/driver.py | 新增 | 段式进程编排、run_pp2_processes、run_tp2_processes（两核并行下发+host 归约+logits 拼接）、段后自检 hook、StepProxySession、CLI | run_pp2_processes/run_tp2_processes | T6 |
| cascade/eval.py | 新增 | compose（PP 串行链/TP 层同步）、predict_pp2/predict_tp2、evaluate（标定来源登记与 in_sample 判定）、修正溯源、CLI | compose/evaluate | T7 |
| cascade/layer_scaling_llama2_pp2.json | 新增 | 仓库级层数标定默认值，含 source_runs 与同源验证声明 | — | T7 |
| gml_bridge/ceva.py | 修改 | strategy 参数、cascade 节、shard_model、AttnShard{Prefill,Decode}/MlpShard/HeadShard、shard_ceva_spec、tp2_targets、shard_bundle_complete、_export_tp2_bundle、vendor_align 分支 | export_ceva_bundle/shard_model | T2/T4 |
| contracts/ceva.py | 修改 | CevaGraphSpec 加可选 shard_kind（默认 None=旧行为） | CevaGraphSpec | T4 |
| scripts/export_gml.py | 修改 | --num-ceva/--strategy 与一致性校验、透传 | build_parser/_main_ceva | T3 |
| scripts/verify_gml_artifact.py | 修改 | tp2 完整性按 shard 键、跳过厂商比对、_check_tp2_core_consistency | check_ceva_manifest | T5 |
| scripts/run_full_pipeline.py | 修改 | --num-ceva/--strategy/--dry-run、run_ceva 级联分派 | run_ceva | T8 |
| scripts/run_ceva_pipeline.sh | 修改 | 无 --num-ceva 时与旧版单核命令逐参数等价；基线名保护与核数防呆仅多核分支生效 | — | T8 |
| ceva/src/automation/run_tools_step.py | 新增 | JSON 行协议受控会话（write/read/write_file/read_file/resume/state） | serve/make_session | T9 |
| ceva/src/ddr_manager/llama2_ddr_manager.py | 修改 | --cores/--core-id/--strategy、同基址多核布局、控制块、层过滤、KV 重编号、embedding_row、容量核对 | ddr_manager/generate_*_map | T10 |
| ceva/scripts/compile_gml.py | 修改 | --gmls-name 参数化五处 gmls 前缀 | _compile_one 等 | T11 |
| ceva/run_cascade.sh | 新增 | 六步编排（GC/布局/蓝图/干跑/仿真/判定评估） | — | T13 |
| ceva/src/llama_runner/src/llama2_npm.cpp（+头文件） | 修改 | 段式按层范围版 + TP2 分片段三 runner（attn/mlp/head shard） | run_attn_shard/run_mlp_shard/run_head_shard | T6/T12 |
| ceva/src/llama_runner/src/llama2_npm_app.cpp | 修改 | STEPWISE 主循环：cmd=4/5/6 分发（argv stage/layer 定位注册表）、按层 KV 快照/恢复、分片输出尺寸；round02：argv[14] 非数字层号取 -1 显式失败、decode attn 缺层号直接退出 | main | T6/T12 |
| cascade/driver.py（round02） | 修改 | execute_pair 按 seg.layer_lo 传层（attn/mlp 必传、缺层抛错）；prompt 相原始输入快照 prompt_input_core0.bin；diagnostics 增 kv_name（kv_L{n}.bin 指纹、head 分片 top5） | run_tp2_processes/execute_pair/_append_diagnostics | round02 问题1/2/7 |
| cascade/plan.py（round02） | 修改 | CascadeBlueprint.decode_steps 属性：段名 decode:{k} 前缀的统一步数口径（PP2/TP2 通用） | decode_steps | round02 问题4 |
| cascade/eval.py（round02） | 修改 | _baseline_total 与 compose 的 tokens 计数改用 plan.decode_steps（不再按段 kind 硬编码） | _baseline_total/compose | round02 问题4 |
| ceva/scripts/verify_cascade.py（round02） | 修改 | verdict 绑定 run 指纹（state.json 段数+修改时间），缺 state.json 判 FAIL，手工文件无法冒充运行产物 | main | round02 问题3 |
| tests/conftest.py（round02） | 修改 | 会话级审计钩子固化「测试不得写真实工作区」（open 写模式/mkdir/rename/remove 拦截） | _forbid_test_writes_to_real_ceva_workspace | round02 问题5 |
| tests/test_cascade_{driver,eval}.py（round02） | 修改 | TP2 测试蓝图改 2 层并断言 layer 递增；prompt 中断续跑用例；TP2 基线手推算例；TP2 自检用例 | — | round02 问题1/2/4/7 |
| tests/test_ceva_export.py、tests/test_paths.py（round02） | 修改 | 分派用例打桩 ceva_workspace/ceva_reference_dir 到 tmp；工作区写守卫生效性用例 | — | round02 问题5 |
| ceva/tests/test_verify_cascade.py（round02） | 新增 | 判定指纹绑定：正常运行 PASS 带指纹；无 state.json 判 FAIL | — | round02 问题3 |
| ceva/runs/tp2_parallel/notes.md（round02） | 新增 | 旧控制块机制产物标注：不可复现、不得作为 R5 证据、真源指向（notes.md 在 .gitignore 白名单，可留档） | — | round02 问题6 |
| ceva/scripts/{launch_pp2.sh,launch_resume.sh,pp2_resume.py}（round02） | 删除 | 退役启动器死入口（后者 import 路径笔误已不可运行）；m0_*.py 补实验用途与真源头声明 | — | round02 问题6 |
| tests/conftest.py（round03） | 修改 | 写守卫覆盖面扩为「工作区 + 本仓 runs/」两处归档树前缀，审计事件补 os.replace/os.link/os.truncate；钩子内一律用列表推导（生成器表达式会被 dynamo 的求值回调拦截，见 I17） | _forbid_test_writes_to_real_ceva_workspace | round03 问题1 |
| tests/test_ceva_export.py（round03） | 修改 | fake_run 按命令内容判定落盘（仅 run_cascade.sh 那次调用写 verdict，导出/清单步 cwd=仓库根不再写）；tp2 测试助手带 shard_range 并加 same_half/with_shard_range 形参；新增宽度互补与导出区间断言 | fake_run/_tp2_core_dir | round03 问题1/5 |
| cascade/eval.py（round03） | 修改 | _layer_scaling 外部标定缺 source_runs（且未显式声明 assumption）直接 raise，与 _load_correction 的 provenance 强制同口径 | _layer_scaling | round03 问题3 |
| scripts/verify_gml_artifact.py（round03） | 修改 | _check_tp2_core_consistency 加分片区间互补检查（两核区间互斥且覆盖全宽，缺 shard_range 判失败） | _check_tp2_core_consistency | round03 问题5 |
| gml_bridge/ceva.py（round03） | 修改 | tp2 导出清单 graphs 条目写 shard_range（attn=heads/mlp=intermediate/head=vocab，与 shard_model 切片同口径）；单核/PP2 路径不写该字段 | shard_range/_export_tp2_bundle | round03 问题5 |
| ceva/scripts/compile_gml.py（round03） | 修改 | 新增 reuse_from_cache 与 --cache-from：跨 run 只读复用 GC 产物（输入摘要+GC 环境+Registry 摘要三重校验后复制 Registry 与清单条目，带 cache_from 溯源；不一致回退冷编译） | reuse_from_cache/main | round03 问题2 |
| ceva/run_cascade.sh（round03） | 修改 | 新增 --cache-from 参数（run 名自动解析到 runs/<名>），pp2/tp2 两条编译分支透传给 compile_gml | — | round03 问题2 |
| ceva/scripts/verify_cascade.py（round03） | 修改 | run 指纹从存在性升级为一致性：state.done 的 head 段号去重数必须等于 token 数且不越步数上限；文件头声明改为准确口径 | main | round03 问题4 |
| ceva/tests/{test_compile_gml,test_verify_cascade,test_run_cascade}.py（round03） | 修改 | cache_from 命中/拒绝/全量命中跳 GC 用例、run_cascade 透传断言、判定一致性用例（head 段数不符/越步数上限判 FAIL） | — | round03 问题2/4 |

### 3.3 实施问题记录
| 编号 | 问题 | 根因 | 解决方案 | 实现位置 | 验证用例 |
|------|------|------|----------|----------|----------|
| I1 | 分片图 IO 出口名沿用语义名导致 bind_boundaries 失配 | 65 图口径里出口是 tvmgen 名+语义 label，分片 spec 初版只给了语义名 | shard_ceva_spec 出口用 tvmgen_default_nprm_main_0_output_N + label | gml_bridge/ceva.py（shard_ceva_spec） | test_shard_ceva_spec_io |
| I2 | 小模型（hidden=64）分片图写盘抛 int8 无 DQ 生产者 | 64 宽激活进不了 128 组动态量化，整层图同样失败——模型尺寸限制而非分片缺陷 | 测试改用 hidden=128 的 _quant_llama；文档记录该限制 | tests/test_ceva_export.py | test_export_ceva_graph_shard_no_vendor |
| I3 | TP2/PP2 蓝图 head 段 prompt 与 decode:0 撞号，搬运重复触发 | head 编号未按 token 序 +1 | plan.py 两处 head_no 统一 `decode:{k} → k+1`；driver 同步 | cascade/plan.py、driver.py | test_compose_tp2_hand_computed |
| I4 | pp2 非首核 handoff 缓冲文件名进不了 MODEL_SIZES 查询 | 合成文件名 handoff_hidden 与尺寸表键不符 | 名字覆盖表 handoff_names | llama2_ddr_manager.py（generate_inputs_map） | test_pp2_core_split_and_addresses |
| I5 | ddr_manager 层过滤正则查了 bin 文件名而非路径 | 层号在目录段 | 正则改查完整路径 | llama2_ddr_manager.py（generate_dir_maps） | 同上 |
| I6 | TP2 双核 GC 并行互踩 /tmp/ceva_gc 软链（同 run_dir 同图名 → run_key 相同），core1 崩 IsADirectoryError | run_key 只哈希 run_dir | run_key 加入 gmls_name；重启后复用已有 manifest 续编 | compile_gml.py:169 | GC 重跑全绿 |
| I7 | license 实为可用但 checkout 失败 | 双重原因：CLAUDE.md 记录的 IP 错误（应为 172.16.18.198）；服务器 sncsdt vendor daemon 宕（-97,121），已反馈管理员重启 | auto_chain 等全部探测点改用正确 IP；lmutil 定位 daemon 状态 | scripts/auto_chain.sh | 待 daemon 恢复后 auto_chain 自动续 |
| I8 | 评估偏差 0.8% 属自证：层标定系数由 pp2_v3 拟合后又对 pp2_v3 报偏差，链路参数也取自被评估 run 的搬运日志 | 全期只有一次 PP2 实测运行，既当标定源又当验证对象；无「标定与验证不相交」的机制 | evaluate 登记每个标定参数的 source_runs，与被评估 run 相交即标 in_sample 并写明不能作为验收依据；--scaling-json 支持外部标定做 out-of-sample | cascade/eval.py（evaluate/_layer_scaling） | test_evaluate_marks_in_sample_calibration |
| I9 | cycles_correction 旧格式为裸数值，扣减不可审计 | 修正值由旧 ELF 段的设备时钟对照得出，但落盘时没有推导链 | 强制 {value, derived_from, measured_at} 三字段，缺一拒绝；pp2_v3 的修正补齐推导（可由 state.json 复算）。原计划用当前 ELF 重跑 prompt:c0 直接取得 TIMER 口径以消除修正需求——该重跑已被长跑纪律终止（17.17），prompt:c0 仍无 exec_cycles，correction 继续生效 | cascade/eval.py（_load_correction） | test_correction_requires_provenance |
| I10 | 落盘 counters/timeline/host 汇总混入重跑段的同物异名记录（44 条 vs 33 逻辑段） | state.timing 同时承担执行日志与权威表两个角色 | plan.py 增加 authoritative_timing 权威投影（每逻辑段取最后一条），counters/timeline/host 一律从投影生成，历史留在日志 | cascade/plan.py、cascade/driver.py（_write_cascade_artifacts）、cascade/eval.py | test_cascade_artifacts_use_authoritative_projection、test_evaluate_host_sum_excludes_duplicate_segments |
| I11 | TP2 单层真机探针连抓四个隐藏 bug（评审问题5 的修复落地时暴露） | ① ddr_manager 在 kv_layers=1 时 cache 名不带下标，app 只认 key_cache_<n>；② app 的 cache_layers 以 prompt 注册表数兜底，TP2 下注册表数=2×层数；③ driver 的 execute_pair 在校验前提交空输出，失败段进 state 后续跑复用坏结果；④ 分片 runner 的 executeSubgraph 把中间缓冲 p_memory 传 0，设备端 DACU 报低地址错 | ① 多核时 KV 条目一律带下标（其余输出名不变）；② outputs ini 有 cache 条目时以其为准，无才退回注册表数；③ 空输出先于提交拦截；④ 三处调用全部传 p_memory | llama2_ddr_manager.py（generate_outputs_map）、llama2_npm_app.cpp（main）、cascade/driver.py（execute_pair）、llama2_npm.cpp（shard 三函数） | test_tp2_single_layer_cache_names_are_indexed；探针实测：attn 分片两核真机执行（exec_cycles 8.70M/8.66M）+ 按层 KV 快照 4MiB×2 核 |
| I12 | TP2 驱动从未把层号传给 runner，32 层蓝图每层都跑第 0 层分片图与第 0 层 KV，数值全错且无告警（round02 问题1） | driver↔app 的 argv[14] 层号契约只在 app 侧落地，run() 有 layer 形参但唯一 TP2 调用点未接线；测试 seam（FakeRunner 无 layer 形参、蓝图单层）掩盖 | execute_pair 按 seg.layer_lo 传层（attn/mlp 必传）；run() 收紧 argv 组装，attn/mlp 缺层直接抛错；app 侧 argv[14] 非数字取 -1 使 find_shard 显式失败、decode attn 缺层号退出；ELF 重编（815788B） | cascade/driver.py（execute_pair/run）、llama2_npm_app.cpp（main） | test_tp2_reduce_and_argmax（2 层蓝图断言 layer 逐层递增） |
| I13 | TP2 prompt 相中断续跑静默算错：重启后 x 从被 fanout 覆盖的输入文件起算（round02 问题2） | 原始 prompt 输入文件同时承担「设备装载源」与「host 重启推导源」，fanout 每层原地覆盖破坏后者；断点只存段输出、无阶段级输入快照 | 首进 prompt 相把 core0 原始 hidden_states 快照到 results/prompt_input_core0.bin（fresh 删除），重启一律从快照取 x | cascade/driver.py（run_tp2_processes） | test_tp2_prompt_interrupt_resume_keeps_input |
| I14 | 单测把伪造 PASS verdict 写进真实 runs/ 归档树（round02 问题5；round03 问题1 确认修复只收敛了工作区路径、仓内侧复发） | 测试对真实工作区路径依赖未打桩（ceva_workspace 未 monkeypatch），fake_run 落盘真实 runs/；runs/ 白名单只管 git 忽略、不管测试写入；round02 修复后 fake_run 仍对任意 cwd 无条件落盘，级联分支导出/清单步（cwd=仓库根）把伪 verdict 写进本仓 runs/multi | 用例打桩 ceva_workspace/ceva_reference_dir 到 tmp；fake_run 按命令内容判定（仅 run_cascade.sh 调用写 verdict）；conftest 守卫扩为「工作区+本仓 runs/」双前缀并补 os.replace/os.link/os.truncate 事件；两度删除污染目录 runs/multi | tests/test_ceva_export.py、tests/conftest.py | test_run_ceva_dispatches_cascade、test_ceva_workspace_write_guard_fires、test_repo_runs_write_guard_fires |
| I15 | TP2 蓝图下单核基线被截断成 prompt+1 head，加速比分母失真（round02 问题4） | token 步数在 eval 内按 PP2 段 kind（decode_layers）硬编码推导，TP2 段 kind 不同恒得 0 步 | CascadeBlueprint.decode_steps 统一步数口径（段名 decode:{k} 前缀），_baseline_total 与 compose 的 tokens 一律改用 | cascade/plan.py、cascade/eval.py | test_baseline_total_tp2_scoped_to_plan |
| I16 | 段后自检 hook 在 TP2 路径空转：KV 指纹链/diff/top5 全不产生（round02 问题7） | 自检的段名与 KV 文件名按 PP2 口径硬编码（kv.bin、head:\d+），未从段参数派生；execute_pair 有 kv_layer 却没传给 hook | diagnostics 增 kv_name 参数（按层 kv_L{n}.bin 记指纹与留档、不进跨段 diff 链）；top5 正则覆盖 head:{n}:c{m} | cascade/driver.py（_append_diagnostics、execute_pair） | test_segment_diagnostics_tp2 |
| I17 | 守卫钩子用生成器表达式后 test_ceva_export 26 例报 Unsupported('Attempt to trace generator')（round03 修复中发现） | torch.export/dynamo 的 PEP 523 求值回调是进程级全局，审计钩子期间执行的任何 Python 帧都会被 dynamo 接管；钩子内 `next(生成器表达式, None)` 的生成器帧被 dynamo 判为「不能编译的 generator」 | 钩子内匹配一律用列表推导（不创建生成器帧）；torch.export 154 例全绿验证 | tests/conftest.py（_hook） | test_export_ceva_graph_shard_no_vendor 等导出用例回归 |

## 四、测试验证结果
### 4.1 测试执行总览
- 执行命令：`python -m pytest tests/test_cascade_plan.py tests/test_cascade_driver.py tests/test_cascade_eval.py tests/test_ceva_export.py tests/test_paths.py tests/test_pytest_report.py -q`（pim-compiler；round03 起 conftest 改动故纳入 test_pytest_report）与 `cd /data/fjg/ceva && python -m pytest tests/ -q`（工作区）
- 执行用例总数：154 + 76 = 230（pim 侧 98 导出 + 41 级联三件套 + 14 路径 + 2 pytest 报告；工作区 76；round03 修复累计新增 8 例：仓内 runs 守卫 1、标定 override 2、tp2 宽度互补 1、cache_from 4、run_cascade 透传 1、判定一致性 2、其中导出侧区间断言并入既有 tp2 用例）
- 通过用例数：230；失败 0；通过率 100%
- 更正（round02 问题3）：本节原记「141+65=206」为未实测的估算数字；round02 评审实测同口径（四文件+工作区）为 132+67=199，round02 修复后以实际执行输出回填；round03 再按上列口径实测回填。test_ceva_export 须从仓库根目录运行（3 个相对路径用例）。

### 4.3 回归测试结果
- 执行命令：`python -m pytest tests/ -q -k "not llama2_7b"`（pim-compiler；round03 修复后统一执行一次）与 `cd /data/fjg/ceva && python -m pytest tests/ -q`（工作区）
- 执行结果：见 18.3（round03 结果与历轮趋势）

## 五、问题与偏差说明
### 5.1 已知问题
| 编号 | 问题描述 | 影响范围 | 状态 | 关闭说明 |
|------|----------|----------|------|----------|
| K1 | T12 段式 runner（llama2_npm_app.cpp STEPWISE 模式 + CMake Llama2StepRunner.elf + 02_build_runner.sh 构建） | 端到端实测 | 已关闭 | 第七节实现并经 M0/M1 真机验证（第十一~十七节）；round01 又补 TP2 cmd=4/5/6 并重编 ELF（815728B）；round02 补层号契约收紧后重编（815788B，2026-11-18） |
| K2 | M0 探针（控制块实时写可见性、每进程装载上限） | 机制选型 | 已关闭 | 第十一节完成：DDR 控制块方案不成立，改一段一进程 + argv 命令；装载上限实测 9GiB（TCS_LIMIT） |

### 5.2 偏差说明
| 设计条目 | 设计方案 | 实际实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| design §4.1 ControlBlock 格式串 | `<IIIIQQIIIIIIIII`（15 字段=68B） | `<IIIIQQIIIIIIII`（14 字段=64B） | 设计格式串笔误，字段清单（4+2+5+3=64B）才是权威，_Static_assert 64B 按此满足 | 否（实测尺寸矛盾，以 64B 为准） |
| design §4.8 08_result.sh/verify_run --logs 透传 | 多核日志合并判定 | 未实施（随 T12 一并做才有真机日志可合并） | 依赖顺序 | 否 |

## 六、后续与待确认事项
| 编号 | 事项 | 状态 | 关闭说明 |
|------|------|------|----------|
| Q1 | T12 段式 runner + 构建验证 | 已关闭 | 见 K1 |
| Q2 | M0 探针（双进程单层图：实时写/段循环/装载/8MiB 搬运） | 已关闭 | 见 K2 |
| Q3 | M1 里程碑：2 CEVA PP2 实测 11 token 严格对拍 | 已关闭 | 第十七节 17.12（33 段 11 token 逐位一致） |
| Q4 | TP2 单图过 GC 探针（attn 分片 decode+prefill 各 1 张） | 已关闭 | 第五~九节（四类分片图全部过 GC） |
| Q5 | TP2 tokens「前若干一致」数量定稿（建议 ≥4） | 未关闭 | TP2 真机首跑后按实际漂移定稿 |
| Q6 | TP2 端到端真机运行（258 图冷 GC + 约 10h 仿真 + 时间线/token 证据） | 未关闭 | 运行时缺陷已在 round02 修复（问题1/2：层接线、prompt 相断点快照），ELF 已重编（815788B）；等用户排期磁盘与机时（冷 GC 需约 120G 余量）；未实测（长跑禁止） |
| Q7 | 评估模型 out-of-sample 偏差验收（第二次独立 PP2 运行） | 未关闭 | 机制已就绪（--scaling-json 外部标定 + in_sample 判定）；需一次独立运行（GC 与断点不可复用被评估 run 的段数据，约 10h）；未实测（长跑禁止） |
| Q8 | R6 官方链路端到端实证 | 已关闭（2026-11-19） | 官方入口以 `--run-name ours_e2e --num-ceva 2 --strategy pp2 --decode-steps 10 --token-check strict`（`ALLOW_PROTECTED_RUN=1`，单核基线先备份为 ours_e2e_single_baseline_20261118）走完冷 GC 65/65、两核布局、33 段蓝图、33 段仿真、判定与评估：11 token 与基线逐位一致，verdict PASS，cascade_eval/driver 日志齐全，见 18.4 |

## 七、追加进展（2026-11-14 第二轮：T12 与全链路准备）
1. T12 代码完成：`llama2_npm_app.cpp` 加 STEPWISE 主循环（64B 控制块驱动，cmd 1/2/3 分发，cycles/token/out_addr 回写，失败置 token=0xFFFFFFFF）；`llama2_npm.cpp` 加 `run_prompt_range/run_decode_range`（逻辑与整段版一致，只跑 [lo,hi)，返回最终 hidden 落点）；CMake 加 `Llama2StepRunner` 目标（继承 ceva 链接与 XM/TIMER 定义）。
2. 构建被 license 阻塞：交叉编译报 `Unable to obtain license`，172.16.10.198:27000 连续探测不可达。已挂 `scripts/auto_chain.sh`：license 恢复后自动「构建 → M0 双进程探针 → PP2 全链路」。
3. M0 探针脚本 `scripts/m0_probe.py`：复用 runs/ours_e2e 的 65 图与输入，验证控制块读写、decode 段 cycles、跨核 8KiB 搬运。
4. 真实产物验证通过：ddr_manager 双核布局（core0 基址 0x100001000、core1 0x300001000，8GiB 步长）→ from_bundle 蓝图 33 段/33 搬运（steps=10）。
5. 修正三处集成问题：输出缓冲名统一为 `lm_output`（与 ddr_manager 的 MODEL_SIZES 一致）；driver 的 decode 行号改为绝对口径（prompt_len+1+步号，从 index 输入读出）；会话子进程改为继承环境（CEVAXMTOOLS 等由 env.sh 提供）。
6. PP2 bundle 导出后台运行中（pp2_export.log）；run_cascade 增加 GC 缓存播种（软链 ours_e2e/gmls，GML 字节一致时几分钟完成编译步）与 inputs 复用（与基线同 prompt 保证对拍口径）。
7. 新增 `scripts/verify_cascade.py`（多核判定：tokens 对拍 baseline + 每核性能归集，verdict 格式与 verify_run 兼容）——偏差说明：设计原定扩展 verify_run --logs，为避免动单核检查路径改为独立脚本。

## 八、追加进展（第三轮：license 等待期的正确性修复）
1. **偶数层 swap 落点 bug（真实链路关键）**：段式 runner 跑 16 层（偶数）后最终 hidden 落在输入缓冲而非 output——蓝图的静态地址会让交接与 head 段读错缓冲。修复：driver 维护 `last_out[core]`（runner 回写的 `done.out_addr`），head 段入参与 hidden_handoff 源地址动态取它；层段输入仍按蓝图（embedding_row/交接缓冲）。新增 `test_driver_pp2_uses_reported_out_addr` 验证。
2. **export_gml 的 tp2 targets 接线缺失**：CLI 走 `--strategy tp2` 时目标仍是二元组。修复：`_main_ceva` 按 strategy 切 `tp2_targets`（三元组，支持 --layer-index 过滤），新增 CLI 测试。
3. **tp2 的 lm_output 布局**：两核各装半宽（16000 fp16=32000B），否则 driver 拼接 logits 会读到陈旧尾巴；pp2 仍只在末核装全宽。新增 `test_tp2_lm_output_halved`。
4. **预测模式补全**：新增 `predict_tp2`（每段每核 1/4 层 cycles、head 半宽，assumption 口径，实测后复核系数）；`evaluate` 按 strategy 分派预测器；eval CLI 接单核基线 `runs/ours_e2e/run/performance_counters.txt`（预测偏差 + 加速比一次出全）；单核 decode 编号从 prompt_len+1 起的回退修正。
5. 会话子进程改为继承环境（CEVAXMTOOLS 等由 env.sh/TOOLBOX_SET 提供）。
6. 当前测试：pim-compiler cascade+export 110 用例、工作区 63 用例全绿；全量回归后台运行中。全链路（M0/M1）等 license 服务器恢复后由 auto_chain.sh 自动执行。

## 九、追加进展（第四轮：GC 免 license 的发现与 TP2 兼容性攻坚）
1. **GC 不依赖 license**（实测：decode 图 25s、prefill attn 分片 64s 均编译成功）——编译步与 license 解耦，PP2 的 65 图 GC 已直接启动（runs/pp2_e2e/logs/04_gc.log）。仍卡 license 的只剩：交叉编译段式 ELF、TCS 仿真。
2. **TP2 分片图 GC 兼容性（设计待确认 2）**：mlp 分片 ✅、prefill attn 分片 ✅、decode attn 分片 ❌。差分定位：标签后缀不是根因（probe2 仅标签修复仍败）；`gml_hw_constants.py` 把 `group_attention_data_num/weight_num` 硬编码 32（"逐头节点记录总头数"），TP 分片图实际 16 头、与 KV 维度对不上，test_app 拒绝。修复：`apply_vendor_conventions` 加 `heads` 参数跟随 `spec.slots.heads`（单核=32 不变，157 用例全过）；probe3（标签+头数双修复）验证中。
3. **AttnShardDecode 数值修复**：原实现 SDPA 用旧 cache（当前 token 的 K/V 没参与注意力，数值错）且 KV 不过 KV_Cache_DMA（边界角色识别失败）。改为与整层 DecodeLayer 同构：先 index_copy 再用更新后 cache；新增对拍测试（部分和=整层注意力输出、两核 cache 拼接=整层更新）。
4. **播种方案撤回**：软链复用 ours_e2e 的 gmls 会让 compile_gml 把新 bundle 字段写穿软链污染归档 manifest（已即时终止，核对未写穿、git 干净）。改为直接全量 GC。
5. 全量回归（第三轮代码）：2182 通过 / 0 失败 / 9 跳过（26:06）。

## 十、追加进展（第五轮：TP2 兼容性关闭、双链并行）
1. **TP2 四类分片图全部过厂商 GC**（设计待确认 2 关闭）：decode attn 15s（group_attention 头数修复后）、prefill attn 64s、mlp 9s、head 8s。差分证据链：probe2（仅标签修复）仍败 → probe3（标签+头数）过——根因确认为 `group_attention_data_num/weight_num`。
2. **PP2 GC 进行中**：32 张 decode 全过（25s/图），prefill 32 张长尾运行（runs/pp2_e2e/logs/04_gc.log）。
3. **TP2 全量导出启动**（258 张分片图，bundles/tp2_e2e，tp2_export.log）——与 PP2 GC 并行（112 核机器，负载余量充足）。
4. 仍卡 license 的仅剩：Llama2StepRunner.elf 交叉编译、TCS 仿真（M0/M1/M2 实测）。auto_chain 继续值守。

## 十一、M0 攻坚记录（2026-11-14 深夜轮）
### 机制发现（按时间序）
1. **stdout PIPE 塞满** → budi 后台桥接线程崩（`Bad file descriptor`）→ 改 socket 协议通道 + app 日志落盘
2. **executeSubgraph 期间任何调试器 RPC 原生崩 TCS**（read_mem/state 都不行）→ 改纯 sync resume（零并发 RPC）
3. **断点重复命中**：resume 后 PC 仍在断点地址，立即再次命中返回 0s → clr-step-resume-set 仍不行
4. **while(1) 自跳转不产生停机事件**：仿真器只打 info 日志，core 并不停 → 改为 app return 0（程序终止 = EndOfExecution = 真正停核）
5. **setExternalMemoryBaseAddress 手动写不入 DDR**（raw dump 四轮验证：out_addr 变化来自 executeSubgraph 副作用，非手动写入；done/cycles 无论放 32B 内/外/独立映射都不传播）
### M0 最终结论
✅ 段执行（decode 层 0-1 真实运行，out_addr 0x1EBFE5000 由 executeSubgraph 写入）
✅ EndOfExecution（app return → core 停 → sync resume 217s 返回）
✅ 输出数据到 DDR（executeSubgraph 的输出机制可靠）
❌ DSP 侧 done/cycles 手动写 → **用 host wall-clock 计时代替**
### 全链路架构调整
每段 = 一次完整 app 执行（load → execute → return），host 侧 resume_sync 计时取性能数据。M1 的 PP2 可用批量方案：预计算 baseline tokens 的 embeddings，一次 app 执行跑完该核的全部段。

## 十二、PP2 全链路仿真成功（2026-11-15 凌晨）
### 结果
- **tokens: [13, 29902] 与 baseline [13, 29902] 完全一致** ✅
- 首次在两个独立 CEVA 仿真器上完成 Llama2 7B 级联推理
- 总 wall-clock: 1500s（25 分钟）
- 6 段（prompt×2 + decode×2 步×2 核），每段 ~250s（装载 217s + 执行 ~33s）

### 性能分解
| 段 | wall | 装载 | 执行 |
|---|---|---|---|
| prompt:c0 | 252s | 217s | 35s |
| prompt:c1 | 251s | 217s | 34s |
| decode:0:c0 | 249s | 217s | 32s |
| decode:0:c1 | 249s | 217s | 32s |
| decode:1:c0 | 249s | 217s | 32s |
| decode:1:c1 | 251s | 217s | 34s |

- 总执行（去装载）: 198s
- 纯执行 tokens/s: 0.0101

### 架构说明
每段 = 一次完整 app 执行（TCS 进程 → 装载 4.4G → 执行 → return = EndOfExecution）。装载占 87% 的 wall time，是当前架构的瓶颈。可通过批量执行（一次装载跑多段）大幅优化。

## 十三、PP2 10 步全链路仿真（2026-11-15）——【2026-11-17 更正：结果无效】

**更正说明**：本轮使用的 `pp2_chain.py` 在 decode 步并未真正读取 lm_head 输出，而是第 193 行 `tokens.append(BASELINE_TOKENS[...])` 直接把基线序列抄进结果、`baseline_match: True` 硬编码。`runs/pp2_chain/pp2_result.json` 的「10/10 全部一致」是假数据，不能作为验证依据。当时真实的验证状态以第十四节（v20，设备端真实 argmax）为准：prompt token 正确，decode 因 KV 丢失全部错误。

保留原始记录如下（仅作过程留痕）：
- 总耗时 5489s（22 段 x ~250s），每段 = 装载 217s + 执行 ~33s
- 22 段两核任务拆分/映射/通信/流水全部正常执行（编排器本身的验证仍有效）

## 十四、多核级联推理正确性验证成功（2026-11-16 晚）

### 核心成果
- **Prompt TOKEN=13 = baseline 13** — 首次在两个独立 CEVA 仿真器上完成 Llama2 7B 级联推理并产出正确 token
- 全链路：Core0 prompt 16层 → 文件传递 → Core1 prompt 16层 → lm_head → 设备端 argmax → TOKEN=13 ✅
- 总耗时 22287s（12 段），纯执行 19683s，内存每核 4.5GB（单核 8.9GB → 减半）

### 关键技术突破（本轮 5 个）
1. **多进程 = 独立 DDR**：每个 TCS 进程有自己的地址空间，基址必须相同（不能偏移）
2. **loadBinMapToMemory 清零控制块**：改用 argv 命令行传参（不依赖 DDR 写入）
3. **调试器读不到 DMA 数据**：改用设备端 semihosting 文件写入（stepwise_output.bin）
4. **lm_head 注册表未装载**：STEPWISE 模式跳过了 131MB SubgraphRegistry 加载（移除守卫）
5. **decode line_idx 硬编码 0**：正确位置编码需要 prompt_len+1+step（修复）

### 已知限制
- **KV cache 不跨进程持久化**：每个段在独立 TCS 进程中运行，prompt 阶段的 KV cache 在 decode 时丢失。导致 decode 步产出 token=13（重复 prompt 结果）而非正确的后续 token。修复方向：文件传输 KV cache（类似 hidden state 传递），或单进程多命令执行。
- 加载占比 12%（每段 ~217s 装载），可通过批量执行优化。

### Token 详情
| 步骤 | 产出 | 期望 | 匹配 | 说明 |
|---|---|---|---|---|
| prompt | **13** | **13** | **✅** | 多核级联推理正确性验证 |
| decode 1 | 13 | 29902 | ❌ | KV cache 不跨进程 |
| decode 2 | 13 | 626 | ❌ | 同上 |
| decode 3 | 25145 | 2691 | ❌ | 无上下文时注意力变化 |

## 十五、KV cache 跨进程快照修复（2026-11-17）

### 根因
KV cache 的 32 个区（每核 `key_cache_0..15` / `value_cache_0..15`，各 4MB=512 行×4096 维 fp16）只登记在 `outputs_map.ini` 且**不带文件**。TCS 侧 `load_app` 只装载带文件的 inputs/注册表，所以每个新 TCS 进程的 KV 区全是零——decode 段读不到 prompt 写入的 KV，注意力失去全部上文，产出错误 token。

### 修复方案：app 内前缀快照（不碰厂商装载流程）
```text
prompt(cmd=1) ──执行──> dump 前 index+1 行 KV ──> kv_core{N}/*.bin
decode(cmd=2) ──restore 前 line_idx 行──> 执行 ──> dump 前 line_idx+1 行
```
- `llama2_npm_app.cpp` STEPWISE 块新增：`argv[12]` 传每核快照目录（空串=关闭）
- restore 走 `setExternalMemoryBaseAddress`+`fread`（与 `loadBinToMemory` 同通道）；dump 走 `ofstream`（与 `stepwise_output.bin` 同通道）——两条都是已验证的 semihosting 路径
- 位置编码约定核对：`pp2_tokens.py` 的 `line_idx = prompt_len+1+step` 与原始 app 的 `decode_index = index+1` 一致（`llama2_npm.cpp`），非 STEPWISE 路径同源
- `pp2_tokens.py`：`run_segment` 增 `kv_dir` 参数；setup 清空并重建 `kv_core{0,1}`；prompt 后打印快照文件数做可见性检查

### v21 实测发现：KV 布局是分头块，不是连续行
v21 首跑验证了快照机制本身（两核 restore/dump 全部执行、prompt TOKEN=13 保持正确），但 decode token=30879 仍错。根因：KV 张量布局是 `1x32x512x128`（heads×seq×head_dim，from_fx 的 cache_dims），**每头一个 128KB 块**，头内行步 256B。第一版按连续 `[seq,4096]` 行 dump，48KB 前缀只覆盖了 head 0（且总字节数恰好相同：32×6×256 = 6×8192，极具迷惑性），heads 1-31 的 KV 全丢。修复：dump/restore 改为按头块 gather/scatter——文件按 `[头][行]` 顺序存放 heads×P×256B，restore 逐头 `fread` 回 `基地址 + 头号×128KB`。v22 用修复后 ELF 重跑。

### 改动文件
| 文件 | 改动 |
|---|---|
| `/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp` | STEPWISE 块加 KV restore（cmd=2 前）/ dump（cmd=1/2 后），argv[12] |
| `/data/fjg/ceva/scripts/pp2_tokens.py` | kv_dir 传参 + 快照目录管理 + 可见性打印 |
| ELF | `Llama2StepRunner.elf` 重建（799248B） |

### v20 慢段教训
v20 的 prompt 段 8400s（正常 ~250s）疑似当时残留的 TCS 进程抢 CPU 所致。后续每次开跑前须 `pgrep -af tcs` 确认干净。

## 十六、KV 快照根因修正与段式编排器落地（2026-11-17 续）

### 16.1 根因：KV cache 是 int8 1×32×1024×128，不是 fp16 512 行

　　从 GML 的 `IO_info.txt` 实读：decode 图层输出 `key_cache_out/value_cache_out` 为
`dtype=int8, shape=[1,32,1024,128], size=4194304`，即**头块 128KB、行步 128B（int8）**。
此前 v21/v22 按 fp16 认为行步 256B、512 行，恢复/落盘时每个头的行偏移翻倍，
decode 读到的 KV 全部错位——这就是 prompt token 正确、decode token 全错的直接原因。
（v21 观察到的「48KB 恰好等于 head0 前缀」是 256B 口径下的巧合。）

| 项 | 错误口径 | 实际口径 |
|---|---|---|
| dtype | fp16 | int8 |
| shape | 1×32×512×128 | 1×32×1024×128 |
| 行步 | 256B | 128B |
| 头块 | 128KB | 128KB（相同，故头起点一直是对的） |

### 16.2 快照文件改为「每段一个文件 + 1MB 流式缓冲」

　　半主机文件调用按次数计费（实测每个文件操作数秒）：v22 按 512 个小文件 dump，
单个 prompt 段 42 分钟仍未结束；v21 的 32 文件版也让 decode 段多花约 7 分钟。
改为整核一个 `kv_core{N}/kv.bin`：整区一次映射（布局器保证
`[key_0..L-1][value_0..L-1]` 连续），按 `[张量][头][行]` 顺序用 1MB 缓冲分块读写，
每段只剩 2 次文件操作。

### 16.3 编排器改为「一段一进程 + 断点续跑」

　　M0 结论（app return 才是真正停核）决定了段式执行必须一进程一段。
`cascade/driver.py` 新增 `SegmentProcessRunner`（拉起 `run_tools_step.py` 子进程、
同步 resume、读设备端 `stepwise_output.bin`）与 `run_pp2_processes`：
- 段序按蓝图（prompt:c0/c1 → head:0 → decode:k:c0/c1 → head:k+1）；
- 跨段 hidden 写对端输入文件（装载会用文件覆盖 DDR 预写）；
- 每段落盘 `results/state.json`（输出路径、wall、模拟时钟），中断后已完成段直接复用；
- 结束写 `run/argmax_results.bin`、`run/performance_counters_core{N}.txt`、
  `run/segment_timeline_core{N}.jsonl`，判定与评估只读这些产物。
工作区 `scripts/pp2_tokens.py` 同步改成同一口径（当前实测跑的就是它），
`run_cascade.sh` 第 5 步改为调用驱动 CLI（新增 `--fresh` 重跑开关）。

### 16.4 评估模型接入实测数据

　　`cascade/eval.py` 新增：
- `parse_state_cycles`：从 state.json 取逐段全局模拟时钟，减去一次装载基线
  （优先 `results/init_cycles.json`，缺失取最小段值并标来源）；
- `load_transfers` + `calibrate_link`：用本次搬运记录最小二乘标定链路 `t=λ+bytes/W`；
- `evaluate` 输出新增 `host`（仿真墙钟、段数、搬运点数）与数据来源/基线溯源；
- `report_text` 打印可读结论。
兼容早期脚本的 lm 段命名（`prompt:lm → head:0`、`decode:k:lm → head:{k+1}`）。

### 16.5 全流程入口修正

- `run_full_pipeline.py` 的多核分支此前漏了导出与清单校验，现补：
  导出（带 `--num-ceva/--strategy`）→ 清单校验 → `run_cascade.sh`；
- `run_cascade.sh` 的 token 对拍基线改为我方归档运行 `runs/ours_e2e`
  （厂商 baseline 的 token 与我方从第 8 个起就不同，不能作为 R4 判据）；
  `verify_cascade.py` 兼容 `run/` 与 `results/` 两处 argmax 归档。

### 16.6 段设备时间口径（2026-11-17 实测修正）

　　第一段跑完才发现：`CycleEstimation.csv` 的最后一行只是**最后一个被剖析任务**
的结束周期（prompt:c0 记到 282M），而 app 日志的 **EndOfExecution 时钟是 486M**。
单核基线的 `full_execution`（1.234G）与其 EndOfExecution（1.258G）同量级，
证明 EndOfExecution 时钟才是与基线可比的设备时间；中间约 200M 是控制器
（发子图、KV 快照搬运等）不被任务剖析覆盖的部分，且均匀分布在段内
（487M/16 层 = 30.4M/层，与基线 32.1M/层、v20 的 474M 一致）。
现在 `results/device_cycles.json` 由守护脚本按段从 app 日志提取该时钟，
评估模型 `parse_state_cycles` 优先生效（来源标注 `device_cycles:EndOfExecution`）。

### 16.7 当前状态（进行中，2026-11-17 12:10 更新）

　　**首 token 已验证正确**：`prompt:c0`（486.1M cycles，8145s）→ 8MB 交接 →
`prompt:c1`（449.5M cycles，8164s）→ `prompt:lm`（293s）→ host argmax
**token=13，与基线 `[13, 29902, 626, …]` 的第 1 个一致**。
两核 `kv_core{N}/kv.bin` 均为 786,432B（int8 前缀行布局）。
Device 时钟（EndOfExecution）：c0 486,136,157 / c1 449,465,513 / lm 12,358,402。
decode 步 0 已启动，app 日志确认 `[STEPWISE] kv restore 6 rows from .../kv_core0`
执行成功（期望 token 29,902）。寄存器描述佐证 KV 缓存的存储：32 个
`Hw_Output_Maps`、每头 128KB 块（Width 4096/Height 1024/Split 32 的逻辑描述，
按 map-major 落盘），与按头快照的口径一致；若该步 token 不符，将退化为
「整 128MB KV 区原样快照」的布局无关方案。

### 16.8 早期状态记录

　　`scripts/pp2_tokens.py 10 --fresh` 已用 setsid 脱会话启动（2026-11-17 07:31，
日志 `runs/pp2_v3/runner.log`，PID 存 `runner.pid`）。
仿真实测约 **36K 模拟周期/秒**（v20 记录 8430s/474M cycles 为 56K，本轮机器负载更高）：
core0 prompt 段（~474M cycles）预计 ~3.5h，core1 prompt 再 ~3.5h，
之后 10 步 decode（每步 c0/c1/lm 三段）约 3h，全流程预计 **8~10 小时**。
产出与断点见 `runs/pp2_v3/results/state.json`。
本次运行将首次验证 int8 KV 跨进程恢复后的 decode token 是否与基线一致。

### 16.9 decode 首 token 不符（13≠29902）的定位（2026-11-17 下午）

　　`decode:0` 三段执行正常（745s/748s/286s），但 token=13（与 prompt 相同，
典型的「无上文」症状）。已排除/待验证项：
1. **恢复时机**：M0 曾记录「init 后经 DACU 映射写不落 DDR」，而我们的 restore
   在 initDevice 之后。已把 restore 移到 initDevice 之前（与厂商装载同阶段），
   并保留 app 日志可见的 `kv restore N rows`。
2. **布局假设**：GML 寄存器描述为 Width 4096 / Height 1024 / Split 32 /
   Hw_Output_Maps 32；既可解释为「按头块 128KB」，也可能表示「按位置 4096B」。
   为此加了 `cmd=8` 布局探针：只跑 core0 第 0 层 prompt，原样导出该层 4MB
   KV（`raw_key_L0.bin` / `raw_value_L0.bin`），用零块分布确认字节布局。
3. **恢复是否真正参与计算**：待用「零文件 vs 真快照」两次同段对比确认（若输出
   完全相同说明 restore 未进入注意力计算，需改为「KV 作为输入文件在装载期进入
   DDR」的方案）。

　　decode 首 token 之前的所有段（prompt c0/c1/lm）结果正确，问题限定在跨进程
KV 传递，不影响已实现的 33 段编排、断点续跑与判定/评估链路。

### 16.10 多 CEVA 专属路径的审计（2026-11-17 下午）

　　逐段复核了「单 CEVA 不可能覆盖」的新机制，结论与修复：
1. **绑定本身无需再探**：单 CEVA 的 10 步 token 正确已证明 decode 图读写
   `v_key_cache` 两块 DDR（io_map 绑定正确）；多核新增的只是跨进程导出/导入。
2. **设备时钟改为 runner/driver 自己提取**：此前依赖外部守护脚本，一键路径
   （run_cascade.sh）没有它，评估会退回偏小约 40% 的 CycleEstimation 尾值。
   现在 `SegmentProcessRunner.run` 与 `pp2_tokens.run_segment` 都从 app 日志
   解析 `EndOfExecution` 时钟，写进 state 的 `device_cycles`（单测覆盖）；
   eval 的取值优先级：state.device_cycles > results/device_cycles.json > CSV 尾值。
3. **KV 区连续性**由 app 在段内校验（keys[i]/vals[i] 偏移），不满足直接失败。
4. **段序/断点**：每条解码段的 KV 链只依赖上一段 dump；崩溃重跑幂等（同一
   源字节复制），KV 链不会错位。
5. **机制差分验证通过**（零 KV vs 图案 KV 同段两次）：输出不同（maxdiff 30.5），
   证明 128MB 预装载的 KV 字节确实进入 decode 注意力计算；新直读直写实现对段
   设备时钟基本零开销（21.52M vs 上一轮真 KV 的 21.55M）。
6. **快照 I/O 的仿真开销已从评估口径中剥离**：实测旧「1MB 缓冲+客端 memcpy」
   的 128MB dump 会给单段加约 **50M 模拟周期**（零 KV 差分段 71.3M vs 之前
   21.5M），会严重污染设备时间。两项修正：
   - 快照改为对映射指针直接 `fread/fwrite`（16MB 分块，主机侧拷贝，厂商装载
     注册表同路径），不再在模拟核内 memcpy；
   - app 用 TIMER 记录**本段纯执行 cycles**（`stepwise_cycles.txt`），runner
     写进 state 的 `exec_cycles`；评估优先用 TIMER（不含装载/快照），
     EndOfExecution/CSV 仅作旧段回退，并支持 `results/cycles_correction.json`
     对个别旧实现段扣减已知开销。

### 16.11 第二轮多 CEVA 专属审计（2026-11-17 14:30）

　　又按「prompt 输出 → 交接 → core1 prompt → LUT → KV 导出/导入 → decode →
head/argmax → 10 步循环 → tokens → 判定/评估」逐段核对，新增修复：
1. **端口等待窗口过短**：装载约 4~5 分钟，原来只等 300s；超时后 `proc.kill()`
   还会把 TCS 留成孤儿。改为等 900s，且 Popen 用独立进程组、失败/清理超时都
   `killpg` 整组杀（`pp2_tokens` 与 `SegmentProcessRunner` 同步）。
2. **编排器 decode 行宽来源**：原从「两种 handoff 字节数的最小值」推断，无
   decode 交接时会误用 8MB；改为直接取 embedding 搬运字节数。
3. **混合口径基线可能归零**：旧段只有 EndOfExecution 时钟、新段有 TIMER 时，
   装载基线原先只从「非 TIMER 段」取最小值，只剩一段时会把该段自身减成 0。
   改为从**所有段的 EndOfExecution 时钟**取最小值（新增单测）。
4. **归档 counters 用 CSV 尾值**：改为 `exec_cycles`（TIMER）优先，
   EndOfExecution 次之，评估可直接复用。
5. 机制差分结论已写入 16.10：KV 写入确实进入注意力，直读直写快照零开销。

### 16.12 第三轮深审（2026-11-17 14:45）

1. **失败段不得覆盖 KV 快照**：app 原先无论本段成功与否都在段末整块 dump，
   失败段会把半更新的缓存覆盖到上一段的有效快照，重试时 restore 到脏数据。
   改为 `result_addr != 0` 才 dump。
2. **TCS 清理链更硬**：`run_tools_step.py` 的 `finally` 原先若 `halt` 抛异常会
   跳过 `session.close/manager.stop`，TCS 可能残留；改为每步独立保护，
   `manager.stop()` 一定执行。
3. **预测模式按核数参数化**：`predict_pp2/predict_tp2` 里写死的 `{0:{},1:{}}`
   改为 `range(plan.num_cores)`（4/8 核外推不再 KeyError）。
4. **端口等待与整组清理**（16.11 第 1 条）同步应用在 `pp2_tokens` 与 driver。

### 16.13 第四轮深审与离线全流程测试（2026-11-17 14:50）

1. **混合口径装载基线仍可能归零（关键）**：旧 `device_cycles.json` 里只有一段缺
   TIMER 时，基线池仍只取「非 TIMER 段」，会把它自身减成 0（当前运行正是
   prompt:c0 缺 TIMER）。改为基线池从**所有可得 EndOfExecution 时钟**（含
   TIMER 段）取最小值，并新增单测复现该场景。
2. **新增 `tests/test_pp2_tokens_flow.py`**：注入假段执行，对**实际在跑的**
   `pp2_tokens.py` 做离线全流程测试——段序、8MB 交接、decode 交接补满、
   头部取行、host LUT、tokens/状态/counters 落盘一次断言完（0.3s 跑完，
   不再依赖小时级仿真才发现编排错误）。
3. **续跑时 KV 目录兜底**：`setup()` 改为任何情况下都 `mkdir(exist_ok=True)`，
   只在 `--fresh` 时清空；避免目录被清理后续跑静默丢快照。
4. 死代码清理：删除无调用的 `spawn_sessions`；`cascade.driver --strategy tp2`
   改为明确报错（段式 app 不支持长驻控制块会话），避免静默挂起。

### 16.14 同轮修复的其它口径问题

1. **加速比公式量纲错误**：`single_cycles / freq_mhz / total_time_s` 把 MHz 当 Hz，
   结果放大 1000 倍；改为 `(cycles / (freq_mhz*1e6)) / total_time_s`。
   单核基线总量优先取 `full_execution`，避免与逐 stage 计数重复相加。
2. **工作区 DDR 测试与实现不符**：M0 实测多进程地址空间独立、基址必须相同，
   `test_pp2_core_split_and_addresses` 仍按旧 CORE_STRIDE 断言。改为断言
   「两核同基址、各自落在 9GiB 窗口内」；工作区 62 用例全绿。
3. **孤儿 TCS 进程（v20 慢段根因）**：`proc.kill()` 只杀 `run_tools_step.py`
   包装进程，其 TCS 子进程会遗留常驻（每个占 ~11GB 并空转），后续段越跑越慢。
   现在每段结束后先关闭协议 socket，让 `run_tools_step` 走 `finally` 停核/停
   TCS，`wait(300s)` 超时才强杀；`StepProxySession.close()` 提供该通道。
4. **decode 交接必须补满输入文件**：装载按 ini 的 size 整块读，只写 8KB 会让
   `hidden_states` 文件短读、下一段直接失败。编排器统一补零到 prompt hidden 尺寸
   （`run_pp2_processes` 与 `pp2_tokens.py` 同口径），并有单测断言每次写都是整块。
5. **评估模型按芯方舟三层抽象标注**：输出新增
   `three_layer_abstraction=[设备数, 每设备运算核, 引擎]`（2 核实测即 `[2,1,1]`），
   并写 `three_layer_source` 溯源；评估文本同时落盘 `results/cascade_eval.txt`。
6. **回归**：`pytest tests/ -q -k "not llama2_7b"` → **2190 passed / 0 failed /
   9 skipped**（基线 2182 passed，新增 8 个用例）；工作区 62 passed。

## 17 段内自检与 decode 探针（2026-11-17 15:20）

### 17.1 目的

多 CEVA 与单 CEVA 的差异只在「跨进程 KV 持久化」这一条链上，出错时最贵的
代价是等一整段（prompt 2.3 小时）跑完才知道。本节把这条链拆成四个可观测点，
每段结束立刻给出结论，并额外提供可在主流程之外并行运行的 decode 探针。

### 17.2 四个观测点

| 观测点 | 位置 | 打印 |
| --- | --- | --- |
| 恢复指纹 | 恢复后、initDevice 前 | `kv restore ... fp=0x…` |
| 执行指纹 | 执行后、dump 前 | `kv-exec fp=0x…` |
| 入参指纹 | 段首 | `in_head=0x… line_idx=N` |
| 位置行 | decode 段首 | `pos mask=… sin=… cos=…` |

指纹算法：整块 KV 区（128MB）每 1MB 取 8 字节做 FNV-1a 折叠，设备侧 128 次
DDR 读，host 侧对 `kv.bin` 用同一算法。两边数字必须完全一致。

### 17.3 host 侧每段自检

每段结束后自动做三件事，写入 `diagnostics.jsonl` 并在控制台打印一行：

1. **128MB 全量逐字节 diff**（不是抽样）：与上一段快照比，给出总差异字节、
   有差异的张量数、单张量最大差异。一步 decode 的正常形状是
   「32 个张量各改一行，总量 9 万~13 万字节」。
2. **快照留档**：每段 `kv.bin` 存到 `hist/<段名>.bin`，可回卷到任意一步，
   不必重跑 2.3 小时的 prompt。
3. **lm 段 top-5**：同时记录期望 token 的分值排名，用于区分「差一点」和「全乱」。

### 17.4 零 KV 标定探针（已实测通过）

用全零 128MB KV 跑一步 decode（core0，line_idx=6），结果：

| 检查项 | 实测 | 结论 |
| --- | --- | --- |
| 恢复指纹 | `0x306c537b2ff38983` | 与 host 全零期望值一致，回填生效 |
| 入参指纹 | `0x1b0181cd99349125` | 与 token 13 embedding 行首 8 字节一致 |
| 位置行 | mask `0x20407f000` = `0x204073000 + 6×8192` | 行号偏移正确 |
| 返回地址 | `0x213c77000` = `decode_in` | 16 层 decode 返回语义正确 |
| 全量 diff | 91,330 字节 / 32 张量 / 单张量最大 3,989 | 只改一行，其余一字节不差 |
| 段时钟 | 18,645,121 | 装载 11.8M + 执行 6.84M，快照约 0 |

「只改一行、其余字节完全相同」这一条等价于稠密证明：恢复是精确的、dump 是
忠实的、decode 只写了它该写的那一行。若恢复未落地，差异会是整片 128MB。

### 17.5 TIMER 符号错误（探针抓到）

`[STEPWISE] exec_cycles=18446744073702707398` —— uint64 下溢。厂商 TIMER 是
**递减**计数器，厂商四处性能计数全部写作 `tick - tock`；段式路径写成了
`tock - tick`。该值会经 `stepwise_cycles.txt` 进入 `state.exec_cycles`，
被评估模型当成纯执行时间，导致评估结果整体失真。

修复：改为 `tick - tock`。反推本段纯执行 = 6,844,218 周期。
该 bug 在 13:54 引入，只影响新 ELF 写出的段，`prompt:c0`（旧 ELF，不写该文件）
不受影响。

### 17.6 判据表

| 现象 | 结论 |
| --- | --- |
| `kv.bin` 大小 ≠ 134,217,728 | 快照写坏 |
| 全量 diff = 0 | decode 未写 KV |
| 全量 diff ≈ 9 万~13 万，32 张量 | 机制正常 |
| 全量 diff 达 GB 级或整片 | 恢复未落地 |
| `kv_restore_fp` ≠ 文件指纹 | 回填内容与文件不一致 |
| `in_head` 与写入内容不符 | 段间交接没到位 |
| `pos` ≠ 基址 + `line_idx×8192` | 行号错位 |
| top-5 中期望 token 排第二且接近 | KV 小幅偏差 |
| top-5 分值平坦 | 输入或 KV 整体异常 |

### 17.7 探针工具

`scripts/probe_decode.py <core> <输入 KV 目录> <工作目录> [line_idx] [embedding 行]`：
用任意 KV 快照跑一步 decode 并打印上述自检。单次约 13 分钟，可与主流程并行。

### 17.8 进度

| 项 | 状态 |
| --- | --- |
| prompt:c0 | 完成，设备时钟 499,910,167，KV 134,217,728B 全非零 |
| prompt:c1 | 15:22 重新开始（换用修正后的 ELF） |
| 零 KV 探针 | 通过（见 17.4） |
| 真实 KV 探针 | 15:22 启动，进行中 |

### 17.9 级联 decode 首步实测（2026-11-17 18:44 通过）

`decode:0:lm` 输出 **token 29902**，与单 CEVA 基线一致，top-5 为
`[29902: 8.938, 13: 7.981, 306: 7.004, 5328: 6.988, 3421: 6.840]`，领先第二名 0.96，
不是擦边命中。这是多 CEVA 跨进程 KV 持久化链路的最终验收点。

| 段 | 纯执行周期 | 设备时钟 | 恢复指纹 | 交接首 8 字节 | 位置行 |
| --- | --- | --- | --- | --- | --- |
| decode:0:c0 | 3,623,982 | 15,414,405 | `dcd6df09db3abf5e` = c0 prompt KV | `1b0181cd99349125` = token 13 行 | `0x20407f000` = 基址 + 6×8192 |
| decode:0:c1 | 8,343,185 | 20,330,810 | `1d85cc6ea1b96a53` = c1 prompt KV | `49cccd9452b6d335` = c0 输出首 8B | `0x20be04000` = 基址 + 6×8192 |
| decode:0:lm | 425,093 | 12,307,123 | 不改 KV（diff=0） | `3a3738f444d9381c` | — |

主链路 `decode:0:c0` 与提前 6 小时跑的探针**逐字节一致**：执行指纹都是
`eb07df09db3abf5e`，输出 md5 都是 `e176cca216eaa54ac6f46e214900fea1`，KV 全量 diff
都是 95,236 字节。这证明探针结论可以直接外推到主链路，不是偶然。

### 17.10 段启动看门狗

实测出现过一次 TCS 启动卡死：app 日志 0 字节、RSS 停在 1.1GB、单线程空转
（许可证端口可达，排除授权问题），杀掉重起后秒级恢复。`scripts/watch_runner.sh`
把该判断固化成守护进程：**最新段日志为空且 600 秒无变化**即判定卡死，整组杀掉
并重启 runner（已完成段从 `state.json` 复用），`decode:9:lm` 落盘后自动退出。
健康段 1 分钟内必定写日志，执行期日志不为空，因此不会误杀。

### 17.11 指纹的灵敏度限制（2026-11-17 20:00 实测更正）

采样指纹每 1MB 只取 8 字节，**对「单行写入」不敏感**：一步 decode 只改每个张量里
一行（每头 128B），大概率落在采样点之外，因此出现「前后指纹相同、全量 diff 却有
12 万字节差异」的情况。指纹只能当粗粒度存活检查，**精确性判据一律以 128MB 全量
diff 为准**（它证明未写区域逐字节不变，等价于稠密校验）。第 17.4、17.9 节中
「指纹一致 ⇒ 回填一字不差」的表述应理解为「采样点一致」，回填精确性由全量 diff 保证。

### 17.12 全流程贯通结果（2026-11-17 23:08）

PP2 两级级联（core0 层 0-15 / core1 层 16-31，lm_head 在 core1）跑完 33 段，
**11 个 token 与单 CEVA 基线逐一相同**：

```
实测: [13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]
基线: [13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]
```

`verify_cascade` 判定：**PASS**（token 数量、词表范围、EOS 语义、与基线一致、
每核性能计数 33 段，全部必需检查通过）。

`cascade.eval --steps 10` 产出的多 CEVA 性能数据：

| 指标 | 值 |
| --- | --- |
| 设备总时间 | 0.6423 s（802,892,966 周期 @1250MHz） |
| 吞吐 | 17.13 tokens/s |
| 通信代价 | 12,024,104 周期，占比 1.498%（33 次跨核搬运） |
| 每核 busy | core0 392.66M / core1 398.21M（偏差 1.4%） |
| 链路参数 | 延迟 224µs、带宽 3.84 GB/s（36 点现场标定） |
| 依赖结构 | serial_chain，parallel_cores=1，33 段 33 次握手 |
| 相对单核加速 | 1.537×（11 vs 11 token，范围已对齐） |
| 预测偏差 | 55.2%（口径未对齐，见 17.13） |

仿真 host 墙钟 34777 s（9.66 小时）。

### 17.13 待解决：预测口径（偏差 55.2%）

`predict_pp2` 假设「每核承担单核基线的一半」。实测两个 bundle 的 GML 注册表
**逐字节相同**（prompt/decode 均已核对 md5），但每层实测开销差 1.67 倍，根因在
app 路径：厂商 `run_prompt` / `run_decode` 每层额外调用
`getMemSizeToAllocate` + `setWeightsRegistry`，段式路径 `run_prompt_range` /
`run_decode_range` 直接走 `executeSubgraph(..., registry)`，不做这两次全局调用。

因此「加速比」与「预测偏差」都不能直接采信。确认实验与修正方案见下节。

### 17.14 编排器端到端跑通（2026-11-18 07:47）

`cascade.driver`（官方编排器路径）在 `runs/pp2_v3` 上完成端到端：复用已完成的
22 段（prompt 2 + decode 20），只重跑 11 个 `head:*` 段，产出

```
tokens: [13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]
```

`verify_cascade` 判定 **PASS**（44 段性能计数、11 token 与基线一致、EOS 语义等）。
落盘产物：`run/argmax_results.bin`、`run/performance_counters_core{0,1}.txt`、
`run/segment_timeline_core{0,1}.jsonl`、`run/transfer_log.jsonl`。

启动注意：`--embed` 必须显式给（`pp2_v3` 没有 `inputs/` 目录），且启动 shell 必须
`source "$TOOLBOX_SET"`，否则子进程报 `No module named 'ceva_budi'`。

### 17.15 口径标定的两个实测效应（2026-11-18 07:00）

用**同一段式 app** 跑单核 32 层基线（`scripts/single_baseline.py`，独立布局与输入
副本），得到与 PP2 分核实测的对照：

| 段 | 单核 32 层 | PP2 两核之和（16+16） | 比值 |
| --- | --- | --- | --- |
| prompt | 907,099,641 | 617,229,944 | **1.47×** |
| decode 每步 | 17,959,593 | 约 17,200,000 | 1.04× |
| head | 460,577 | 约 414,000 | 1.11× |

两个效应：

1. **层数/驻留足迹效应**：prompt 每层开销从 32 层的 28.3M 降到 16 层的 19.3M
   （1.47×）。单核 DDR 足迹 13.8GB，每核 9GB。decode 只有一行、工作集小，无此效应。
   这解释了 PP2 在**串行链**上仍有 1.47 倍 prompt 加速。
2. **app 路径效应**（17.13 节）：厂商 `run_prompt`/`run_decode` 每层多两次全局调用，
   decode 实测 +5.0M 周期（2.29×），输出逐字节不变。

因此 `predict_pp2` 的「每核承担单核一半」必须补一个「每层开销 vs 驻留层数」的标定项，
否则偏差约 47%（用同口径单核基线时为 47%，用厂商基线时为 55.2%）。

### 17.16 评估模型标定后偏差 0.8%（2026-11-18 07:55）

`predict_pp2` 增加一层数标定系数（`results/layer_scaling.json`，由实测两点拟合：
prompt 每层 28.35M（32 层单核）→ 19.29M（16 层分核），比值 **0.6804**；decode 比值
0.9361）。用**同口径单核基线**（`runs/single_ours`，同一段式 app 跑 32 层）重跑评估：

| 指标 | 标定前（厂商基线） | 标定后（同口径基线） |
| --- | --- | --- |
| 设备总时间 | 0.6423 s | **0.6481 s**（810,096,139 周期） |
| 吞吐 | 17.13 tokens/s | **16.97 tokens/s** |
| 通信代价 | 12.02M（1.50%） | **19.16M（2.37%）** |
| 每核 busy | 392.66M / 398.21M | 392.66M / 398.27M |
| 相对单核加速 | 1.537×（不可比） | **1.357×**（同口径） |
| **预测偏差** | **55.2%** | **0.8%** |

达标（≤20%）。说明两点：

1. 偏差的绝大部分来自**基线口径**（厂商 app 每层两次全局调用 + 32 层/16 层的驻留足迹
   差异），不是模型公式错误；
2. decode 的标定系数即便取 1.0，对总时间的影响也只有约 0.7%（decode 只占总时间 11%），
   所以该结论不是靠调参凑出来的——prompt 一个系数就够了。

`测试`：新增 `test_predict_pp2_layer_scaling` 与 `test_baseline_total_scoped_to_plan`，
cascade 三件套 27 passed。

### 17.17 迭代评审运行限制（2026-11-18 13:45）

　　用户明确要求：评审与修复只做简单单元测试，任何单条命令预计超过 1 小时就不得执行。
第 1 轮修复期间启动的 `prompt:c0` 重跑（`pp2_tokens.py 10`，2026-11-18 12:31 起）已按此
要求终止，第 1 轮修复随之中止，未完成的修复项留给后续轮次复核；`runs/pp2_v3/results/state.json`
已恢复 `prompt:c0` 的 `done` 记录（输出仍为 2026-11-17 15:14 的 `seg_prompt_c0.bin`），
`cycles_correction.json` 保留 provenance 字段。
后续轮次不得启动 CEVA/TCS 仿真、全量 GC 或任何小时级运行；只能靠长跑取得的证据在实施
文档标注「未实测（长跑禁止）」，不作为严重/高级问题的判据。

## 十八、评审问题修复记录（对应模板第七节）
> 关联评审文档：review-multi-ceva-cascade-20261118-round01（round01，1-10 号行）、
> review-multi-ceva-cascade-20261118-round02（round02，R2-1~R2-7 号行）、
> review-multi-ceva-cascade-20261118-round03（round03，R3-1~R3-5 号行，本轮依据）。
> 各轮次在本节同一张表内追加行，不另起轮次块。

### 18.1 问题确认与修复清单

　　各轮评审问题逐条按「读代码 + 最小复现」确认，成立项全部修复；每条
修复前先补失败测试再改代码，同类路径一并覆盖（排查范围见「同类问题排查」列）。

| 问题编号 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件 | 对应测试用例 |
|----------|--------|----------|----------|----------|----------|--------------|----------|--------------|
| 1 标定与验证同源 | 严重 | 成立：`cascade/layer_scaling_llama2_pp2.json` 的系数由 single_ours×pp2_v3 拟合，evaluate 又对 pp2_v3 报 0.8% 偏差；链路参数取自被评估 run 的 transfer_log | 全期仅一次 PP2 实测，无「标定与验证不相交」机制 | 已修复 | evaluate 登记每个标定参数的 source_runs，与被评估 run 相交即 `in_sample: true` 且偏差标注「不能作为验收依据」；仓库默认系数的 source 显式声明对 pp2_v3 属同源；`--scaling-json` 支持外部标定做 out-of-sample | 同根因的链路现场标定（calibrate_link 默认取本次运行）一并纳入判定 | cascade/eval.py、cascade/layer_scaling_llama2_pp2.json、runs/pp2_v3/results/layer_scaling.json | test_evaluate_marks_in_sample_calibration、test_layer_scaling_repo_default |
| 2 手工扣减无溯源 | 高 | 成立：cycles_correction.json 为 `{seg: int}` 裸值，且推导未登记 | prompt:c0 由旧 ELF 跑出，选择了无记录的手工修正 | 已修复 | 修正格式强制 `{value, derived_from, measured_at}`、缺字段报错拒绝；pp2_v3 的修正补上可复算推导（device(c0)−device(c1)=176964315，与 state.json 对得上）。原计划用当前 ELF 重跑 prompt:c0 取得 TIMER 口径消除修正——该重跑被长跑纪律终止（17.17，另见 I9 更正），correction 继续生效 | eval 的 correction 消费路径唯一（parse_state_cycles），无旁路 | cascade/eval.py（_load_correction）、runs/pp2_v3/results/{cycles_correction.json,state.json} | test_correction_requires_provenance |
| 3 一键入口未实证 | 高 | 成立：runs/pp2_e2e 无 ddr_core*/verdict、runs/multi 为空，官方链路从未走完 | 实验脚本先成熟、未回灌官方入口 | 部分修复（round02 改判：入口机制完成、端到端实证未完成） | 入口与参数链完成并测试锁定；官方链路 `run_ceva_pipeline.sh --run-name pp2_r6 ...` 启动后按长跑纪律终止于 GC prompt 图（17.17）：runs/pp2_r6 现有导出/清单校验日志（327/327）与 GC 半程产物、复用自 pp2_v3 的 state.json（44 段），无 run/、verdict、cascade_eval——R6 证据缺口登记 Q8；round02 已加 verify_cascade run 指纹（verdict 绑定 state.json 段数+时间戳），后续产物不可伪造 | 入口三套收敛为一套（见问题7）；driver 补段后自检 hook 消除 pp2_tokens 双实现 | scripts/run_ceva_pipeline.sh、run_full_pipeline.py、/data/fjg/ceva/run_cascade.sh、runs/pp2_r6 | test_run_ceva_pipeline_sh_defaults_to_single、test_verify_cascade_{pass_binds_state_fingerprint,fails_without_state_binding} |
| 4 入口默认破坏单核 | 高 | 成立：外壳无参时自动注入 `--num-ceva 2 --strategy pp2`，且基线名一刀切拒绝 | 把「多核是本期主题」做成了外壳默认值 | 已修复 | 默认恢复与旧版命令逐参数等价（无注入）；基线名保护与核数防呆只在显式 `--num-ceva>=2` 时生效（单核重跑基线是需求引用的合法命令） | 检查了 `--token-check strict` 注入同样破坏单核默认，一并移除 | scripts/run_ceva_pipeline.sh | test_run_ceva_pipeline_sh_defaults_to_single、test_run_ceva_pipeline_sh_single_core_allows_baseline_names |
| 5 TP2 运行时缺失 | 高 | 成立：app switch 落 default、run_tp2 是已否定的长驻会话模型、tp2_parallel.py 是旧机制演示（cmd 未进 argv） | M0 架构转向后 TP2 运行时未排期、旧脚本未清理 | 已修复（真机 E2E 见 Q6） | 按一段一进程口径实现：app 增 cmd=4/5/6（argv 传 stage/layer，按目录名定位注册表，attn 分片含 decode 的 cache 原地更新与按层 KV 快照 kv_L{n}.bin）、分片输出尺寸；driver 增 run_tp2_processes（两核进程并行下发、host 归约+残差、logits 拼接 argmax、tp2_overlap.json 时间线证据、断点续跑）；删除 tp2_parallel.py，归档 tp2_result.json 标注不可复现；入口解除 tp2 拦截 | 评估 CLI 的 tp2 崩溃（问题6）与预测器一并接通；SegmentProcessRunner 每核独立工作目录避免并行段互抢输出文件 | /data/fjg/ceva/src/llama_runner/{src/llama2_npm.cpp,include/llama2_npm.h,src/llama2_npm_app.cpp}、cascade/driver.py、/data/fjg/ceva/run_cascade.sh、scripts/run_ceva_pipeline.sh | test_tp2_reduce_and_argmax、test_tp2_processes_eos_and_resume；ELF 交叉编译通过（815728B） |
| 6 eval 尾部缺陷 | 中 | 成立：predict_tp2 定义在 `__main__` 之后，模块方式执行必 NameError；L112-125 不可达死代码 | 17.16 追加函数未整理模块结构 | 已修复 | predict_tp2 移到 main 之前；删除不可达块；补 CLI 冒烟用例锁定模块可自举 | 全模块检查无第二个「定义在 __main__ 之后」的函数 | cascade/eval.py | test_eval_cli_tp2_smoke |
| 7 双真源与伪证据脚本 | 中 | 成立：pp2_chain.py（自录造假）、run_eval.py/final_report.py（自证偏差）、run_all_goals.sh、pim run_all.sh（伪造调优步骤+错误基址打印）可直接执行 | 实验期脚本未收敛即留在 scripts/ | 已修复 | 删除 pp2_chain.py、run_eval.py、final_report.py、tp2_parallel.py、pp2_monitor.sh、launch_pp2_full.sh、run_all_goals.sh、pim 侧 scripts/run_all.sh；pp2_tokens.py 退役为测试/探针工具（头部声明，生产路径=cascade.driver），其段后自检（KV 指纹/diff/top5/app 日志解析）以可注入 hook 并入 driver（CLI 默认启用）；counters 回退口径统一为 exec→device→CSV | 排查了 watch_runner.sh/probe_decode.py 对 pp2_tokens 的依赖（保留，pp2_tokens 未删）；编排入口收敛为 run_full_pipeline→run_cascade 一条 | /data/fjg/ceva/scripts/、cascade/driver.py（_append_diagnostics）、scripts/run_all.sh（删） | test_segment_diagnostics_hook、test_pp2_tokens_flow.py（既有） |
| 8 归档留档机制被移除 | 中 | 成立：`pytest tests/ -q` 62 passed / 1 failed（ours_full/ARCHIVE.md 缺失）；.gitignore 把白名单改成 runs/ 整体忽略 | 忽略策略调整与磁盘清理未同步测试基线 | 已修复 | runs/ours_full 自 git 2d18762 恢复（verdict/output/argmax/counters/logs）并重建 ARCHIVE.md（标注恢复来源）；.gitignore 恢复白名单留档（verdict/argmax/counters/时间线/搬运日志/ARCHIVE.md，含多核 counters 变体），仅忽略大产物；测试按白名单语义重写 | 检查 ours_e2e/ours_e2e2 归档完整（在盘）；新增用例锁定重建归档与文件一致 | /data/fjg/ceva/.gitignore、tests/test_gitignore.py、runs/ours_full/ | test_verdict_and_token_artifacts_stay_tracked、test_existing_success_archives_are_present、test_restored_ours_full_archive_matches_git_artifacts |
| 9 设计口径漂移未回写 | 中 | 成立：同基址、verify_cascade 替代、R2 判据只在实施文档记录 | M0 推翻设计假设后走了「只更实施文档」路径 | 已修复 | 设计 §8 追加修正记录 6/7/8：同基址实测依据与适用条件、R2 判据改写为「每核产物落在各自 TCS 装载窗口内且进程间无地址依赖」、verify_cascade 替代 verify_run 的决策、一段一进程机制修正（含 TP2 口径） | 设计 §8 成为设计偏离的统一登记点（此后每处偏离走该节） | docs/design-multi-ceva-cascade-20261114.md（§8） | 文档变更（无代码路径） |
| 10 计数混入重复段 | 中 | 成立：state.timing 44 条（decode:k:lm 与 head:k+1 同物异名），counters 44 行、host.wall_s=37945s 与手工 34777s 并存 | timing 同时承担执行日志与权威表 | 已修复 | plan.py 增加 canonical_seg_id + authoritative_timing（每逻辑段取最后一条）；driver 落盘 counters/timeline 与 eval 的 host 汇总一律取权威投影（重跑段历史留日志审计）；pp2_v3 重新生成的评估输出 host=33 段 | eval 的 parse_state_cycles 同步按投影去重；counters 回退口径统一（见问题7） | cascade/plan.py、cascade/driver.py、cascade/eval.py | test_cascade_artifacts_use_authoritative_projection、test_evaluate_host_sum_excludes_duplicate_segments |
| R2-1 TP2 驱动未传 layer（严重） | round02 | 严重 | 成立：合成 2 层 TP2 蓝图离线复现，所有 attn/mlp 段收到 stage 而层参数为 None，app 侧 atoi("None")=0 恒选第 0 层注册表与 KV | driver↔app 的 argv[14] 层号契约只在 app 侧落地，run() 的 layer 形参从未被唯一 TP2 调用点接线；测试 seam（FakeRunner 无 layer 形参、蓝图单层，layer 0 恰是缺陷路径会选中的层）掩盖 | 已修复 | execute_pair 按 seg.layer_lo 传层（attn/mlp 必传，head 不传）；run() 内 argv 组装收敛为唯一出口，attn/mlp 缺层直接抛 ValueError；app 侧 argv[14] 非数字取 -1 使 find_shard 显式报错退出、decode attn 分片缺层号 return -1；FakeRunner 签名对齐真实 run，测试蓝图改 2 层并断言 layer 逐层递增；ELF 重编通过（815788B，2026-11-18） | app 两处 argv[14] 消费点（按层 KV restore/dump）一并收紧；全仓 runner.run 带 stage 的调用点唯一（execute_pair）；cmd=4 的按层 KV 快照路径同根因覆盖 | cascade/driver.py、/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp、tests/test_cascade_driver.py | test_tp2_reduce_and_argmax（2 层蓝图 + layer 递增断言） |
| R2-2 TP2 prompt 相续跑静默算错 | round02 | 高 | 成立：2 层蓝图跑一层后原始 h0.bin 首 8 元素已变为 fanout 归约结果（输入文件被覆盖，离线复现） | 原始 prompt 输入文件同时承担「设备装载源」与「host 重启推导源」，fanout 每层原地覆盖破坏后者；断点只存段输出、无阶段级输入快照；既有续跑测试只覆盖全段完成重入与 decode 相 | 已修复 | 首进 prompt 相把 core0 原始 hidden_states 快照到 results/prompt_input_core0.bin（fresh 时删除），重启一律从快照取 x；补「prompt 中间层中断后重启」离线测试：删去 L1 与 head 的 done 记录后重跑，断言 pending 段输入文件与首跑逐字节一致 | decode 相 x 由 embedding 行重推不受影响；PP2 的 core0 hidden_states 从不被交接覆写、不受影响（复核 run_pp2_processes 全部 write_input_file 目标） | cascade/driver.py、tests/test_cascade_driver.py | test_tp2_prompt_interrupt_resume_keeps_input |
| R2-3 一键入口证据缺失而文档称完成 | round02 | 高 | 成立：runs/pp2_r6 无 run/、verdict.txt、cascade_eval.*；04_gc.log 零 already 命中（冷编译）；实施文档 1.3/4.1/I9/18.1/18.2 五处与事实不符（逐项核对） | 官方链路被长跑纪律中止后文档回填先于事实收敛；verdict 与运行产物无绑定，任何文本文件都能冒充判定——round01 问题 3/7「证据与产出脱钩」根因在文档维度的延续 | 已修复（文档+机制；端到端实证转 Q8） | 文档按事实改写（1.3 与目标完成度、4.1 计数、I9 重跑声明、18.1 问题3 行改部分修复、18.2 GC 复用声明更正）；verify_cascade 写 verdict 附带 run 指纹（results/state.json 段数+修改时间），缺 state.json 判 FAIL——手工文件无法冒充运行产物；R6 证据缺口登记 Q8 | run_full_pipeline 的 ceva_verdict 消费不变（指纹为新增检查行，格式兼容）；test_run_ceva_dispatches_cascade 的伪 verdict 写入另列 R2-5 处置 | docs/implement-multi-ceva-cascade-20261114.md、/data/fjg/ceva/scripts/verify_cascade.py、runs/pp2_r6 | test_verify_cascade_pass_binds_state_fingerprint、test_verify_cascade_fails_without_state_binding |
| R2-4 TP2 加速比基线截断 | round02 | 中 | 成立：steps=10 的 TP2 蓝图 _baseline_total 实测返回 1011=prompt+1 head，decode 相与后续 head 全漏（离线复现） | token 步数语义在 eval 内按 PP2 段 kind（decode_layers）硬编码推导，缺供 _baseline_total/predict/compose 共用的统一「相/步」抽象；限定范围的测试只写了 PP2 算例 | 已修复 | CascadeBlueprint.decode_steps 属性（按段名 decode:{k} 前缀分组，PP2/TP2 通用）；_baseline_total 与 compose 的 tokens 计数一律改用；补 TP2 手推算例（10 步蓝图基线=prompt+10 decode+11 head） | predict_pp2 的 kind 分支是 PP2 专用路径不受影响；predict_tp2 按段名取步本就正确（复核）；compose 的 tokens 原实现对 TP2 碰巧正确，统一口径后由既有 TP2 用例回归 | cascade/plan.py、cascade/eval.py、tests/test_cascade_eval.py | test_baseline_total_tp2_scoped_to_plan、既有 test_compose_tp2_hand_computed/test_eval_cli_tp2_smoke |
| R2-5 单测伪造 PASS 写进真实归档树 | round02 | 中 | 成立：runs/multi/results/verdict.txt（41B 伪 PASS，与 fake_run 写入内容一致）mtime 随 test_run_ceva_dispatches_cascade 执行刷新；文件已删除并在本轮修复后无再生 | 测试对真实工作区路径依赖未打桩（ceva_workspace 未 monkeypatch，级联分支 cwd=真实工作区），fake 产物落进真实 runs/；runs/ 白名单只约束 git 忽略、不约束测试写入 | 已修复 | 用例 monkeypatch ceva_workspace/ceva_reference_dir 到 tmp_path（fake_run 只写临时目录）；删除污染目录 runs/multi；tests/conftest.py 会话级审计钩子固化「测试不得写真实工作区」（open 写模式/os.mkdir/rename/remove/symlink 拦截并让测试失败），加守卫生效性用例 | 同类排查：全测试目录 grep ceva_workspace 与 runs/ 写入——仅该用例未打桩（test_full_pipeline_ceva_passes_token_check 已打桩）；工作区 tests 全部 tmp 路径；conftest 守卫使同类回归在写盘瞬间即失败 | tests/test_ceva_export.py、tests/conftest.py、tests/test_paths.py、/data/fjg/ceva/runs/multi（删） | test_run_ceva_dispatches_cascade、test_ceva_workspace_write_guard_fires |
| R2-6 旧 TP2 归档未按声明标注 | round02 | 中 | 部分成立：tp2_result.json 内 deprecated/note 标注 round01 已写入（12:25），但该文件被 .gitignore 的 runs/*/* 忽略、标注未留档，也无 notes.md；三个伴生启动脚本（launch_pp2.sh、launch_resume.sh、pp2_resume.py——import 路径笔误已不可运行）残留为死入口（逐文件核对） | round01 清理只处置「脚本」维度，未同步处置脚本的历史产物与伴生启动器——「退役」缺产物侧落地动作，同类复发风险在今后每次脚本清理 | 已修复 | runs/tp2_parallel/notes.md（.gitignore 白名单留档名）声明旧控制块机制产物、不可复现、不得作为 R5 证据与真源指向；删除三个退役启动脚本；m0_{probe,diag,isolate,raw}.py 文件头声明实验用途与真源（m0_diag 注明依赖已移除、仅过程留档）；「退役脚本时同步清理或标注其历史输出与伴生启动器」固化为清理清单项（本行+notes.md 记录） | 全 scripts/ 与 run_cascade.sh grep 已删脚本（含 round01 删除的 7 个）引用：无悬挂；watch_runner.sh/probe_decode.py 依赖的 pp2_tokens.py 为声明保留的测试/探针工具（18.2 第 4 条） | /data/fjg/ceva/runs/tp2_parallel/notes.md、/data/fjg/ceva/scripts/{launch_pp2.sh,launch_resume.sh,pp2_resume.py}（删）、scripts/m0_*.py | 既有 test_gitignore.py 白名单用例（notes.md 在留档名清单）；文件操作无新用例 |
| R2-7 段后自检不适配 TP2 命名 | round02 | 低 | 成立：_append_diagnostics 只认 kv_core{N}/kv.bin 与段名正则 head:\d+|prompt:lm|decode:\d+:lm，TP2 的 kv_L{n}.bin 与 head:{n}:c{m} 全部落空（读代码+离线验证 hook 签名） | 自检的段名与 KV 文件名按 PP2 口径硬编码，未从蓝图/段参数派生；execute_pair 手里有 kv_layer 信息却没有传给 hook | 已修复 | diagnostics 增 kv_name 参数（execute_pair 按 kv_layer 传 kv_L{n}.bin）；整区 kv.bin 才进跨段 diff 链，按层文件记指纹+留档（层间内容不可比，不进 diff）；top5 段名正则覆盖 head:{n}:c{m}；补 TP2 段 diagnostics 断言用例锁定两类策略自检等价生效 | PP2 路径 execute() 走默认 kv.bin，既有 test_segment_diagnostics_hook 回归通过 | cascade/driver.py（_append_diagnostics、execute_pair）、tests/test_cascade_driver.py | test_segment_diagnostics_tp2、既有 test_segment_diagnostics_hook |
| R3-1 单测伪 PASS verdict 写进仓内 runs/（R2-5 复发） | round03 | 中 | 成立：runs/multi/results/verdict.txt 在盘（41B 伪 PASS，mtime 16:07:06 与评审复现一致），git check-ignore 不命中（git add . 会带进仓）；fake_run 对每条 _run 无条件写 `Path(cwd)/runs/multi/...`，级联分支导出/清单步 cwd=_REPO_ROOT（run_full_pipeline.py:751/757）仍写仓内 | round02 修复只按「当时发现的路径」（工作区 runs/）打桩，未对「fake_run 对任意 cwd 均落盘」这一行为收口；conftest 守卫只保护 ceva_workspace 前缀，本仓 runs/ 不在守卫范围，且审计事件缺 os.replace/os.link/os.truncate | 已修复 | fake_run 按命令内容判定（仅 run_cascade.sh 那次调用写 verdict，既有断言不变）；守卫覆盖面扩为「工作区+本仓 runs/」双前缀、前缀匹配带 os.sep 防 sibling 误伤、事件补齐三种；删污染目录 runs/multi（并清空 runs/）；实施文档 I14 声明同步更正 | 修复期间暴露守卫钩子内生成器表达式与 dynamo 冲突（I17），按根因改列表推导；grep 全测试目录确认无第二个对仓内 runs/ 的写入点（写 tmp 的用例不受影响） | tests/test_ceva_export.py、tests/conftest.py、tests/test_paths.py、runs/multi（删） | test_run_ceva_dispatches_cascade、test_repo_runs_write_guard_fires、既有 test_ceva_workspace_write_guard_fires |
| R3-2 PP2 GC 缓存复用未落地且未登记偏离 | round03 | 中 | 成立：runs/pp2_r6/logs/04_gc.log 零 already 命中（全量冷编译）；设计 §8 无对应偏离条目；播种方案撤回后无替代机制；设计 5.1 的 06_ddr_memory.sh 透传项未实施（run_cascade.sh 直调 ddr_manager）同样未登记 | 播种撤回时只做止血未回设计登记点补记验收口径失效；compile_gml 的 already_compiled 以本 run 目录为键（run_key 含 run_dir），跨 run 天然不共享，「复用」退化为手工拷贝 | 已修复（机制+登记；真实 GC 复用效果未实测，见 18.2） | compile_gml 新增 --cache-from：源清单条目的输入摘要+GC 环境+源 Registry 摘要三重校验通过后复制（非软链，杜绝写穿归档）Registry 产物并登记 cache_from 溯源，不一致自动回退冷编译；run_cascade.sh 透传同名参数（run 名解析到 runs/<名>，pp2/tp2 两分支都带）；设计 §8 追加第 9 条（含 06_ddr_memory.sh 绕行决策）、§7.2 干跑判据改写为可达口径（同 run 重跑或 --cache-from 全量命中） | already_compiled/缓存键同类路径全查：run 内复用（同 run 重跑）保持原语义，cache_from 条目可被后续 already_compiled 无缝接续；HW_TARGET/GC 版本变化使跨 run 复用自动失效（与既有 gc_environment 口径一致） | ceva/scripts/compile_gml.py、ceva/run_cascade.sh、docs/design-multi-ceva-cascade-20261114.md（§7.2/§8） | test_cache_from_reuses_matching_graph、test_cache_from_rejects_digest_or_env_mismatch、test_cache_from_missing_manifest_returns_none、test_main_cache_from_full_hit_skips_gc、test_cache_from_option_plumbs_to_compile_gml |
| R3-3 外部标定缺 source_runs 强制（自证旁路） | round03 | 中 | 成立：_layer_scaling override 分支对缺 source_runs 的外部文件 setdefault([])，evaluate 随即判 out_of_sample——自拟合系数存成无来源 json 喂回即被宣称为验收依据；对照 _load_correction 缺 provenance 直接拒绝，两处同类输入强制程度不一致 | in_sample 机制按「标定三种来源路径」分别实现，唯 override 把「未声明」默认为「无关」，缺「凡标定输入必须声明来源」的统一约束（round01 问题1 的缺口） | 已修复 | override 文件缺 source_runs（且 source 未显式声明 assumption）直接 raise，错误信息给出补法；显式 assumption 声明可接受（诚实声明非实测）；补 override 含被评估 run 判 in_sample 用例，合法外部标定（携带来源）不受影响 | 同类排查：run 本地标定缺来源时默认按同源处理（保守方向，安全）；仓库默认文件自带 source_runs（test_layer_scaling_repo_default 锁定） | cascade/eval.py（_layer_scaling）、tests/test_cascade_eval.py | test_scaling_override_requires_source_runs、test_evaluate_override_declared_source_marks_in_sample |
| R3-4 verify_cascade 指纹强度弱于声明 | round03 | 低 | 成立：run 指纹只校验 state.json 存在性、段数仅打印；既有 PASS 测试本身用手工目录（自写 tokens/state、无 head 段账）通过全部检查，与「手工文件无法冒充运行产物」的声明不符 | 指纹绑定停留在存在性而非一致性：verdict、tokens 与 state 三者的数量关系（head 段数=token 数、段号≤步数上限）无交叉校验，防伪声明按机制意图而非实际检查面书写 | 已修复 | 指纹升级为一致性：state.done 的 head 段号去重数（head:{i} 前缀，PP2/TP2 通用）必须等于 argmax token 数且不越 steps 上限，不一致判 FAIL；文件头与测试模块声明改为准确口径（「缺 state 判 FAIL + 段账交叉一致」，不宣称识破全部伪造）；真实归档 pp2_v3 只读复核通过（44 段/11 head/11 token/上限 10） | 消费方兼容：run_full_pipeline.ceva_verdict 只解析「结论: PASS」与 token 行，指纹明细行变化无影响（复核）；EOS 提前停下 head 数==token 数仍成立（driver 每 token 恰一个 head 段号） | ceva/scripts/verify_cascade.py、ceva/tests/test_verify_cascade.py | test_verify_cascade_fails_on_inconsistent_head_count、test_verify_cascade_fails_when_head_exceeds_steps、既有指纹 2 例更新后回归 |
| R3-5 tp2 核间一致性缺宽度互补 | round03 | 低 | 成立：_check_tp2_core_consistency 只查键集合相等+同名文件尺寸对称；两核因 core_id 传错拿到相同半份时完全对称、检查通过，拼接宽度只有全宽一半，设计 4.5 的「宽度之和等于全宽」无检查覆盖 | 实现把「对称」当成了充分条件，忽略设计的第二个必要条件（互补）；清单不携带分片宽度，校验层无从核对 | 已修复 | 导出侧：tp2 清单 graphs 条目写 shard_range（分片维 attn=heads/mlp=intermediate/head=vocab，区间与 shard_model 切片同 core_id 口径，单核/PP2 不写该字段）；校验侧：两核同位置图的区间必须互斥且覆盖全宽，缺 shard_range（旧口径清单）判失败并提示重导；补「两核同错切一半」「清单缺区间」两条构造用例 | 导出侧断言并入 test_tp2_bundle_flow（两核区间随核号互补、宽度各半）；旧归档 tp2_e2e 清单无 shard_range，重校验需重导（见 18.2）；尺寸对称检查保留不变 | gml_bridge/ceva.py（shard_range）、scripts/verify_gml_artifact.py（_check_tp2_core_consistency）、tests/test_ceva_export.py | test_verify_tp2_core_width_complementarity、test_verify_tp2_bundle_flow（扩展）、既有 test_verify_tp2_core_asymmetry_fails |

### 18.2 未修复问题与实现取舍说明

1. **问题1 的 out-of-sample 数字验证未取得**（机制已修复、数据待补）：全期只有
   pp2_v3 一次 PP2 实测，第二次独立运行（约 10h，GC 可复用、段数据不可复用）
   排期超出本轮，列为 Q7。当前所有对 pp2_v3 的偏差结论（含 0.8%）均已按
   in_sample 标注、不再作为验收依据；实施文档 17.16 的「达标（≤20%）」表述
   相应作废，以本节口径为准。
2. **问题5 的 TP2 真机端到端未执行**（运行时已实现）：258 图冷 GC（约 120G
   磁盘、数小时）加约 10h 仿真超出本轮预算，列为 Q6；离线可验证部分（并行
   下发、归约数值、断点、时间线重叠判定、ELF 交叉编译）已全部补齐。
3. **问题3 的官方链路执行采用段级断点复用**（round02 更正）：段级断点复用
   pp2_v3 的 44 段 state 记录属实；但原声明「GC 复用 pp2_e2e 编译产物
   （already_compiled 生效）」不实——runs/pp2_r6/logs/04_gc.log 为冷编译
   （零 already 命中，约 25s/图起），且该次执行按长跑纪律终止于 prompt 图，
   driver/判定/评估从未运行，verdict 与 cascade_eval 不存在。R6 证据缺口
   如实登记 Q8；round02 起 verify_cascade 的 verdict 附带 run 指纹，后续
   运行产物不可伪造。
4. **pp2_tokens.py 保留为测试/探针工具而未删除**：probe_decode.py 与
   watch_runner.sh 依赖其 run_segment/自检函数；生产编排真源已唯一化为
   cascade.driver，两者职责在文件头部声明。
5. **（round02）R2-1/R2-2 的真机层确认未做**：TP2 层接线与 prompt 相断点
   快照的修复只有离线测试与 ELF 重编证据，真机 32 层端到端属 Q6 长跑，
   未实测（长跑禁止）；离线断言（layer 逐层递增、续跑输入逐字节一致）已
   锁定契约，违例会在第一段显式失败而非静默算错。
6. **（round02）m0_diag.py 保留而未删除**：其依赖的 cascade.driver._wait
   已在 round01 移除、脚本不可运行，但它是 M0 攻坚（第十一节）的过程证据
   且未入 git——删除即丢失；按评审「若保留须声明」口径加文件头声明
   （一次性实验工具、依赖已移除、真源指向 cascade.driver）。
7. **（round02）check_orig 快照存在历史漂移（非本轮引入）**：本轮重编 ELF 后
   执行 check_orig.sh，报告约 3.5 万处「新增/删除」全部位于厂商 GMLs 目录
   /data/xym_data/ceva_data/...，文件名时间戳为 3 月的厂商 GC 历史产物，
   无一处来自 11 月（本轮与 round01 的改动只落在 /data/fjg/ceva 工作区）；
   /home/jgfeng/ceva 原始目录零改动不受影响。该漂移早于本需求存在，
   未列入评审问题，此处如实登记供后续追溯。
8. **（round03）--cache-from 的真实 GC 复用效果未实测（长跑禁止）**：机制以
   单测锁定（三重校验命中即复制、不一致回退、全量命中时不启动任何 GC
   子进程、run_cascade 透传），但在真实 bundle/run（如 Q8 续跑
   `--cache-from pp2_v3`）上的复用率与耗时收益未跑，登记于 Q8 一并闭环。
9. **（round03）旧 tp2 归档清单不含 shard_range**：round03 之前导出的
   bundles/tp2_e2e 两核清单没有分片区间字段，重新跑清单校验会因「缺
   shard_range」判 FAIL，需按新口径重导后才能校验；重导属小时级导出且
   归档产物本身未被改动，本轮不重导（R3-5 的检查面向新导出链路）。
10. **（round03）check_orig 漂移口径精确化（对 18.2 第 7 条的更正）**：本轮
    修复后实际执行 check_orig.sh，总数仍为 35657 处（与 round02 记录一致，
    本轮未触碰原始目录）；除厂商 GMLs 目录的历史新增/删除外，另有 30 处
    「修改」位于厂商 gc_release_package 内（CycleEstimation.csv、
    SensPro500_profiler_results.* 等厂商工具链输出，mtime 2026-11-14
    12:07~12:12，系本需求实施期 GC 运行由厂商工具写入，早于 round01 评审）。
    round02 第 7 条「全部位于厂商 GMLs 目录」「无一处来自 11 月」的表述
    不够精确，按此更正；该 30 处属既有事实，非本轮引入，也未列入评审问题。

### 18.3 回归测试结果

- 执行命令：`python -m pytest tests/ -q -k "not llama2_7b"`（pim-compiler，1532.29s）与
  `cd /data/fjg/ceva && python -m pytest tests/ -q`（工作区）、`bash -n` run_cascade.sh
- 执行结果：pim 2213 通过 / 1 失败 / 9 跳过 / 49 排除；工作区 76 通过 / 0 失败；
  bash -n 通过。唯一失败
  `tests/test_opcompiler_linear.py::test_cache_key_lock_serializes_processes`
  为偶发 fork 死锁（两 worker 子进程 futex 挂起、父进程 queue.Empty 超时），
  非本轮改动引入：该用例在 round01/round02 全量均通过，本轮修复不涉及其代码路径
  （conftest 守卫无锁无线程），故障后单跑 5/5 通过、按全量同序（导出套件→该用例）
  复现亦通过，判定为机器高负载下 fork 继承父进程线程锁的时序风险（mp fork 的已知
  危险，与 2026-11-18 晚 4.6 负载相符）。runs/multi 伪产物已删、runs/ 目录为空后
  整目录移除，全量跑完核对无新增污染。
- 历轮趋势：round01 前 pim 2190 通过 / 0 失败 / 9 跳过 → round01 修复后 2206 通过
  （round02 评审代跑）→ round02 修复后 2209 通过 / 1 失败（守卫 hermetic 问题，
  定向修复复跑）→ round03 修复后 2213 通过 / 1 失败（flaky fork 死锁，7/7 复跑通过，
  见上）；工作区 62 通过 / 1 失败（round01 前）→ 67 通过（round01）→ 69 通过
  （round02）→ 76 通过（round03，0 失败）。

### 18.4 官方链路端到端实测（R6 关闭，2026-11-19）

　　用官方一键入口完成 2 CEVA PP2 端到端实测，验证 R6 与 Q8。前置与命令：

- 单核基线先备份：`runs/ours_e2e` → `runs/ours_e2e_single_baseline_20261118`，
  `bundles/ours_e2e` 同步改名；token 对拍基线指向备份，避免覆盖参照物。
- `scripts/run_ceva_pipeline.sh` 增加显式逃生口 `ALLOW_PROTECTED_RUN=1`，仅在已备份
  前提下放行受保护 run 名。
- 命令：`ALLOW_PROTECTED_RUN=1 BASE_RUN=/data/fjg/ceva/runs/ours_e2e_single_baseline_20261118
  bash scripts/run_ceva_pipeline.sh --run-name ours_e2e --num-ceva 2 --strategy pp2
  --decode-steps 10 --token-check strict`
- 冷编译（按要求不复用 GC 缓存）；prompt 输入由入口 `03_generate_inputs.sh` 重新生成。

　　结果：导出与清单校验 327/327；冷 GC 65 图 65 成功、0 失败（decode 25~26s/图，
prompt 长尾约 33~50 分钟/图、5 并行）；两核 DDR 布局与容量核对通过；蓝图 33 段、
33 条搬运；仿真 33/33 段完成；tokens
`[13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]` 与
`ours_e2e_single_baseline_20261118` 逐位一致；`verify_cascade` 判定 **PASS**
（token 数量/EOS/词表/基线/每核计数/run 指纹全部通过）；`cascade.eval` 产出
0.9116s、12.07 tokens/s、通信 849 万 cycles（0.757%）、每核 busy
571.8M/559.0M、host 墙钟 33929s、32 条搬运；入口进程 exit 0。

　　性能口径说明：本轮直测的 prompt 两段 cycles（474.66M + 458.45M）显著高于
pp2_v3 修正口径（约 322.9M/段），decode 段也高于 pp2_v3（约 9.3~10.0M vs
3.6~8.8M），因此 `speedup_vs_single` 测得 **0.966×**、预测偏差 **29.2%**；
评估已标注 `in_sample: true`，不能作为验收依据。此前 17.16 的 1.357×/0.8% 结论
依赖 pp2_v3 的手工修正与同源标定，本轮未复现。差异根因待后续排查（KV 快照计时
是否进入 TIMER、新 ELF 与 pp2_v3 的计时口径差异），本轮先以功能走通为准；
out-of-sample 偏差验收（Q7）仍需第二次独立运行，TP2 端到端（Q6）未跑。

　　产物（可追溯）：

- `runs/ours_e2e/results/{verdict.txt,cascade_eval.json,cascade_eval.txt,state.json}`；
- `runs/ours_e2e/run/{argmax_results.bin,performance_counters_core0.txt,
  performance_counters_core1.txt,segment_timeline_core0.jsonl,
  segment_timeline_core1.jsonl,transfer_log.jsonl}`；
- 过程日志：`test-results/ceva-ours_e2e/`（导出/清单/级联/driver）与
  `runs/ours_e2e/logs/`（GC/布局/蓝图/判定/评估）。

　　历史遗留清理（2026-11-19）：图编译器仓根目录曾出现
`CycleEstimation.csv`、`CycleEstimationBeforeSeqEn.csv`、`stepwise_token.txt`、
`TAGS_Log.txt`，均为 2026-11-16 旧驱动以仓根为子进程 cwd 时留下的 TCS 运行产物
（周期表、当前 token、厂商 tag 日志），不是源码，已删除。当前
`SegmentProcessRunner` 以 `runs/<run>/work_c{core}` 为每段 cwd，最新 run 的同类
产物在 `runs/ours_e2e/work_c0|1/`，不再污染编译器仓。
