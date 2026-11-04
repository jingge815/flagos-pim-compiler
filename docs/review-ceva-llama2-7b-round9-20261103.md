# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-round9-20261103
> 创建日期：2026-11-03
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述
核查四仓未提交状态，重点审查图编译器生成与标定、FlagTree 相位变更、CEVA 接线和验收，以及第八轮修复记录；GeneSim 无未提交变更。本轮确认高优先级问题 2 项，当前版本尚不能据现有记录判定三个核心目标全部完成。
实跑生成、量化、相位、校验相关六组测试及 CEVA 工作区测试，共 217 passed（20.89 秒）；另用临时目录复现非法 token 被告警档放行。本轮未重跑全仓测试、FlagTree lit 或完整硬件仿真；以下运行状态截至 2026-11-03 06:58，现场另有 GC 续编进程，结论不代表其后续结果。

## 二、修改建议
### 问题1（高）：当前验收尚未完成，导致硬件通过未证
**问题说明**：需求 P0-4、P0-5 要求当前生成的全部 GML/bin 经 GC、DDR、CEVA 完成至少全模型 1 步 decode，并归档对应结果。实施文档第八轮记录仍为“最终判定结果待补”；现场 `ours_e2e/logs/run_all.log` 的最新一轮开始于 11 月 2 日 19:25，停留在步骤 4，且当前有 `prompt/model_layers_1_prompt_v2` 的 GC 续编进程。该目录现存 `results/verdict.txt` 却是历史 15 项 PASS，`ARCHIVE.md` 明确对应 10 月 29—30 日的运行，并不能证明最新自研 bin 已完成仿真。历史 `own_bins_e2e` 的自研 bin 三步通过同样早于后续写零等变更；这是第八轮问题2尚未闭环，不是已经观察到硬件执行失败。

**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md:244-249`；`/data/fjg/ceva/runs/ours_e2e/logs/run_all.log`、`logs/04_step.log`、`logs/resume_l1.log`、`ARCHIVE.md`、`results/verdict.txt`。

**修改思路**：完成当前续编后，使用同一生成包完成 DDR 与仿真，归档生成摘要、65 图编译清单、实际步数、当前判定及原始目录前后检查。历史记录继续保留在已有的历史归档目录；当前运行目录应明确标注未完成，避免展示历史 PASS 作为本次结论。默认完成 10 步，若缩减按需求记录原因；补齐证据前保持本项未完成状态。

### 问题2（高）：告警档放行非法 token，导致错误输出通过
**问题说明**：需求 P0-5 只放宽与厂商 token 的一致性，仍要求输出合理。新增 `TOKEN_CHECK=warn` 将所有 token 不匹配统一降为 WARN，而没有独立检查 token 是否在模型词表内。用既有 `tests/test_verify_run.py` 的 `_make` 构造日志和 bin 均为 `[65535, 65535]`、基线为 `[13, 29902]`，补齐性能记录与文本后调用告警档，实际返回 0、最终 PASS、零 FAIL、零 SKIP。65535 超出本期 Llama2 的 32000 词表范围，属于非法输出，不是量化导致的正常分叉；严格档原先会因不匹配拒绝，此次放宽后失去这道约束。

**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py:184-190`；`/data/fjg/ceva/tests/test_verify_run.py` 的告警档用例。

**修改思路**：将 token 合法性作为独立必需检查，依据实际模型或 tokenizer 的词表范围校验全部输出，严格档和告警档均遇非法值即失败；只有合法 token 与基线不同才允许 WARN。补充合法分叉通过、越界 token 失败两个用例，并断言最终退出码。该检查只能证明 token 合法，不能替代数值输出与硬件运行的验证。
