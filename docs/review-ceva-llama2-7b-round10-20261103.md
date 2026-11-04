# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-round10-20261103
> 创建日期：2026-11-03
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述

核查四仓暂存区、工作区及新增文件，审查生成、量化、标定、FlagTree 相位、CEVA 接线与验收；GeneSim 无未提交变更。本轮确认中优先级问题 2 项，均为新增告警档带来的验收边界缺口；`ours_e2e` 生成包的两端编译器摘要与当前工作树一致，最新日志记录完整 10 步 decode、正常结束及合法 token 分叉，上一轮“当前硬件通过未证”和越界 token 问题已有对应证据与修复，本轮发现不表示该次运行失败。

实跑图编译器相关测试 468 passed、4 skipped，CEVA 工作区 45 passed，FlagTree TritonPIM 用例 68 passed；当前生成包校验 327/327 通过，65 张图的 GC 输入、Registry 与编译环境摘要均匹配；另用临时目录复现下述两项，均在严格档返回 1、告警档返回 0，且告警档零 FAIL、零 SKIP。本轮未重新执行完整硬件仿真或全仓 7B 慢测。

## 二、修改建议

### 问题1（中）：非末尾 EOS 被放行，导致少跑误获通过

**问题说明**：需求 P0-5 仅允许“遇到 EOS”提前结束，告警档只放宽与基线的 token 一致性；厂商 runner 在首次 token 2 后停止 decode。新增告警分支取消了基线不一致时的失败，但数量检查仍只判断 `DELIMITER in tokens`，没有要求 EOS 是最后一个 token。用 `tests/test_verify_run.py::_make` 构造 `steps=10`、日志与 bin 均为 `[13, 2, 306]`、基线 `[13, 29902, 626]`，补齐性能记录和输出文本，设置 `TOKEN_CHECK=warn` 后实际返回 0：只完成 2 步 decode，却以“遇到结束符提前停止”通过。该序列在 EOS 后仍有输出，不能作为正常提前终止的证据；新增词表范围检查也无法拦截，因为这些 token 均合法。

**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py:193` 新增降级分支与 `:152-160` 数量检查的交互；`/data/fjg/ceva/tests/test_verify_run.py:145` 的告警档用例。runner 停止条件可核对 `/data/fjg/ceva/src/llama_runner/src/llama2_npm_app.cpp:394`（仅用于确认契约）。

**修改思路**：独立校验 EOS 终止语义，严格档和告警档均不得放宽：出现 EOS 时必须是唯一且最后一个 token，提前结束时必须满足该条件，再核对实际 decode 数。补充 `[13, 2]` 正常提前结束、`[13, 2, 306]` 非末尾 EOS、重复 EOS 三类用例，断言最终退出码；保留无 EOS 的合法 token 分叉通过告警档。

### 问题2（中）：厂商模式也能告警通过，导致基线退化漏检

**问题说明**：需求 P0-5 明确“我方 bin 用告警档，厂商 bin 的历史运行仍按严格档逐 token 一致”，设计 7.3 采用相同边界。新增 `TOKEN_CHECK=warn` 分支却无条件将不一致降为 WARN，既不检查 `bundle_mode`，命令行也不限制告警档的使用模式。用既有测试夹具构造 `gml_bundle=False`、不设置 `GML_BUNDLE`，输出 `[13, 306]`、基线 `[13, 29902]`，补齐性能记录和文本后调用告警档，实际返回 0、结论 PASS，且不会执行自研包的 Registry 来源检查。因此厂商模式的数值退化同样能被我方量化差异的豁免掩盖。

**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py:178-194`；`/data/fjg/ceva/run_all.sh:49-57`；`/data/fjg/ceva/tests/test_verify_run.py:145-150`。

**修改思路**：依据可核验的自研生成包来源决定是否允许告警档；厂商模式及历史厂商 bin 运行保持严格档，误传 `warn` 时拒绝或仍按严格档判定。判定脚本本身也应执行该约束，避免直接调用绕过命令行检查。补充厂商模式分叉必失败、自研包合法分叉可告警通过的成对测试。
