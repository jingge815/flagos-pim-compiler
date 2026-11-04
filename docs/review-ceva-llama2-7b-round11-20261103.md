# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-round11-20261103
> 创建日期：2026-11-03
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述
核查四仓未提交变更，重点复核第十轮问题、当前生成包与最新硬件证据；GeneSim 无未提交变更。本轮确认中优先级问题 4 项、低优先级问题 1 项，其中第十轮的厂商告警档与非末尾 EOS 问题仍成立，另有零步解码放行、旧归档未更新及方言索引未同步。
实跑图编译器快速全量回归 1664 passed、8 skipped、49 deselected，另完成相关测试 250 项、CEVA 工作区 45 项、FlagTree lit 265 项，均通过；当前包清单校验 327/327 通过，65 图共 199353 个文件全部为 `generated`，两端编译器摘要与当前工作树一致。最新 `ours_e2e` 已有 10 步 decode、17 项 PASS、2 项 WARN、零 FAIL/SKIP 的硬件记录，本轮未重跑完整硬件仿真；上述问题属于仍可复现的验收边界与归档缺陷，不否定该次运行结果。

## 二、修改建议
### 问题1（中）：厂商模式可告警通过，导致基线退化漏检
**问题说明**：需求 P0-5 明确只有我方生成的 bin 可以使用告警档，厂商 bin 的历史运行仍须严格逐 token 一致。当前 `TOKEN_CHECK=warn` 无条件把不一致降为 WARN，既不检查是否为我方包模式，`run_all.sh` 也允许厂商模式传入该档位。用既有测试夹具构造 `gml_bundle=False`、输出 `[13, 306]`、基线 `[13, 29902]`，告警档实际返回 0、结论 PASS，厂商数值退化因此会被我方量化差异的豁免掩盖。

**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py:178-195`；`/data/fjg/ceva/run_all.sh:49-57`；`/data/fjg/ceva/tests/test_verify_run.py:145-150`。

**修改思路**：依据可核验的自研生成包来源决定是否允许告警档；厂商模式及历史厂商 bin 运行保持严格档，误传 `warn` 时拒绝或仍按严格档判定。命令行与判定脚本都要执行该约束，避免直接调用判定脚本绕过；补充厂商模式分叉必失败、自研包合法分叉可告警通过的成对测试。

### 问题2（中）：非末尾EOS被放行，导致少跑误获通过
**问题说明**：需求 P0-5 只允许首次遇到 EOS 后正常提前结束，告警档没有放宽执行步数。当前数量检查只判断 token 序列中是否出现 `DELIMITER`，不要求它位于末尾；厂商 runner 则在首次 token 2 后停止 decode。构造 `steps=10`、日志与 bin 均为 `[13, 2, 306]`，告警档实际返回 0，并把仅完成的 2 步 decode 报成“遇到结束符提前停止”；`[13, 2, 2]` 也能通过。EOS 后仍有输出或重复 EOS 都不符合 runner 的正常终止语义。

**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py:152-160,192-195`；`/data/fjg/ceva/tests/test_verify_run.py:74-78,145-150`；runner 停止条件见 `/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp:394`。

**修改思路**：独立校验 EOS 终止语义，严格档和告警档均不得放宽：提前结束时 EOS 必须唯一且为最后一个 token，再核对实际 decode 数。补充 `[13, 2]` 正常提前结束、`[13, 2, 306]` 非末尾 EOS、重复 EOS 三类用例并断言最终退出码；保留无 EOS 的合法 token 分叉通过告警档。

### 问题3（中）：零步解码判定通过，导致全流程未达底线
**问题说明**：需求 P0-5 要求至少完成全 32 层的 1 步 decode，设计 7.3 采用相同判据。`run_full_pipeline.py` 已拒绝 `--decode-steps 0`，但直接入口 `run_all.sh` 只检查 `-s` 是非负整数，判定器又按 `steps+1` 和实际 decode 数核对，因此零步会形成自洽的假通过。用既有夹具构造 `steps=0`、token 与基线均为 `[13]`，实际返回 0，token 数量、decode 步数和最终结论均为 PASS，尚未执行 decode 却满足表面验收。

**涉及文件**：`/data/fjg/ceva/run_all.sh:42,55`；`/data/fjg/ceva/scripts/verify_run.py:152-160`；`/data/fjg/ceva/tests/test_verify_run.py` 的步数用例。

**修改思路**：在 `run_all.sh` 入口要求 `STEPS>=1`，判定器也对缺失或小于 1 的步数明确判 FAIL，防止绕过入口直接复核旧产物。补充命令行零步拒绝和判定器零步失败的回归用例，保留 EOS 正常提前结束的例外。

### 问题4（中）：当前归档仍是旧记录，导致验收证据失真
**问题说明**：需求第四章要求归档本次生成命令、来源清单、GC 日志、DDR map、CEVA 日志、token、判定与耗时。当前 `runs/ours_e2e/ARCHIVE.md` 与 `runs/ours_e2e_20261029/ARCHIVE.md` 逐字节相同，仍记录 10 月 29 日运行的“11 个 token 全一致、15 项 PASS、总耗时 10h19m44s”；同目录最新判定却是 11 月 3 日的“17 项 PASS、2 项 WARN、token 分叉、总耗时 7h09m14s”。旧归档已经备份到历史目录，继续留在当前运行目录会把两次不同产物与结果混为同一次验收。

**涉及文件**：`/data/fjg/ceva/runs/ours_e2e/ARCHIVE.md:5-14`；`/data/fjg/ceva/runs/ours_e2e/results/verdict.txt:14-22,39-40`；`/data/fjg/ceva/runs/ours_e2e/logs/run_all.log:187-207`；`/data/fjg/ceva/tests/test_gitignore.py:23-29`。

**修改思路**：按最新 `ours_e2e` 运行重写当前归档，旧内容只保留在 `ours_e2e_20261029` 历史目录；后续复用运行名时同步更新归档。现有测试只检查文件存在，应增加归档 token、判定项和实际步数与当前摘要一致的检查，避免旧 PASS 长期覆盖新结果。

### 问题5（低）：方言索引未同步，导致维护口径滞后
**问题说明**：FlagTree `claude.md` 第 3、112～116 行规定，修改代码后必须同步更新该文件与 `README.md`，属性变化还要更新检索指引。本次新增了 `fixed2float` FPSU 模式并放开 int8 模型权重，修改了 5 个方言实现文件和 3 个 lit 文件，但 `claude.md` 与 `README.md` 均无未提交变更，现有索引仍不能说明输出头的 int8 静态量化及显式 FPSU 保留规则，不符合仓库明确的文档维护约束。

**涉及文件**：`/data/fjg/flagOS/flagOS-installers/FlagTree/include/triton/Dialect/TritonPIM/IR/PIMAttrDefs.td`；`lib/Dialect/TritonPIM/IR/Ops.cpp`；`lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp`；`claude.md:3,39,49,89,112-116`；`README.md`。

**修改思路**：在方言索引中补充 `fixed2float`、int8 模型权重的配套约束及显式 FPSU 保留行为，并在 `README.md` 同步对应能力说明；只更新实际已实现的语义，不增加新的设计分支。
