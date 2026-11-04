# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理

> 文档编号：review-ceva-llama2-7b-round3-20261031
> 创建日期：2026-10-31
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029
> 关联上一轮评审：review-ceva-llama2-7b-round2-20261031

## 一、评审概述

本轮复核了第二轮评审 R2-1～R2-8 的修复：flagos-pim-compiler、FlagTree、CEVA 工作区均有对应改动，核心链路仍保持已记录的 PASS；genesim 工作区仍无未提交变更，未纳入代码审查。实际执行 `pytest tests/ -k "not llama2_7b and not test_pimir_sidecar_agrees_with_ttir_on_flops"` 为 1587 passed、8 skipped，`test_ceva_export.py` 42 passed，CEVA `tests/` 23 passed，FlagTree lit 265 passed；65 图全流程本轮未重跑，依据现存的 `ours_full` PASS 记录。仍发现 6 个问题：高 1、中 2、低 3，集中在 `--from` 复跑时的 Registry 新鲜度、单层验证默认口径和文档/测试一致性。

## 二、修改建议

### 问题1（高）

**问题说明**：设计 4.12 要求每张图都有本次新生成的 Registry，P0-2/P0-4 要求不得混入旧 Registry。R2-2 增加了 `--gml-bundle` 与 `--from > 4` 的检查，但该检查只看 `RUN_DIR/gml_bundle.txt` 的内容是否等于包路径、`RUN_DIR/gmls` 目录是否存在（`run_all.sh` 第 72–76 行），不检查该目录是否由当前包成功编译、是否覆盖全部 65 图、是否对应当前 `bundle.json` 内容。与此同时 `step_ddr` 在 `compile_gml.py` 之前就写入标记（第 185–197 行），编译失败后标记和部分 `gmls` 仍会保留。实测用临时 `bundle.json`、空 `gmls` 目录和匹配的 `gml_bundle.txt` 执行 `run_all.sh --from 5 --dry-run`，守卫直接放行到第 5 步。若同名 `RUN_DIR` 里已有上一次的 `ddr_output`，`07_run.sh` 只依赖 `ddr_output`，而 `verify_run.py` 只检查 Registry 路径是否以 `RUN_DIR/gmls/` 开头（第 153–163 行），不会发现这些 Registry 来自旧包或部分编译，仍可能给出 PASS。
**涉及文件**：`/data/fjg/ceva/run_all.sh` 第 72–76、185–197 行；`/data/fjg/ceva/scripts/verify_run.py` 第 153–163 行。
**修改思路**：在“编译成功、DDR 生成成功、`foreign` 检查通过”之后写一个成功标记（而不是在编译前写），并在 `FROM > 4` 时校验该成功标记、`compile_manifest.json` 覆盖当前 bundle 的全部图且 `problems` 为空、各 `input_sha256` 与当前包一致；或者直接禁止 `--gml-bundle` 与 `FROM > 4` 同用。`verify_run.py` 也应校验 Registry 的摘要/生成时间，而不只校验路径前缀。补“编译失败后 `--from 5` 必须拒绝”“同路径重导后 `--from 5` 必须拒绝”两个用例。

### 问题2（中）

**问题说明**：设计 4.11 要求单层独立运行后输出 hidden、K、V，并与厂商同层逐字节比较，不一致时报告首个差异。`standalone_layer.sh` 把厂商比较做成了可选的第四个参数（第 3、9 行），不传时第 31 行的 `${REF:+--compare "$REF"}` 不展开，脚本只检查本次输出是否存在、尺寸是否正确，然后打印“独立运行完成”（第 32 行），并不会做数值比较；这与设计 4.11 和任务 D2“单层独立运行与厂商同层逐字节比较”的默认口径不一致，也不在实施文档 5.2 的偏差清单里。另外第 29 行用 `grep -q "return value (return value): 0"` 只要求日志里出现任意一条 0，不能保证最后一个/全部 engine 的返回值都是 0。
**涉及文件**：`/data/fjg/ceva/scripts/standalone_layer.sh` 第 3、9、26–32 行；`docs/implement-ceva-llama2-7b-20261029.md` 第 5.2、7.1 节。
**修改思路**：把厂商同层输出目录设为单层验收的必填参数，或增加显式的“仅运行/含比较”模式并在 5.2 记录该偏差；检查日志中最后一个或全部 `return value` 都是 0，而不是任意一条为 0。补一个默认 3 参数调用必须拒绝（或明确打印“未做数值比较”）的用例。

### 问题3（中）

**问题说明**：设计 4.12 要求每张图有本次新生成的 Registry，修改时间晚于本次开始。`compile_gml.py` 的 `already_compiled`（第 66–83 行）只比较 `runtime_files` 全部文件名/内容摘要和 Registry 摘要；摘要不包含 GC 工具链版本、`HW_TARGET`、`-l/-u` 等 GC 参数。GC 或硬件目标变化后，只要 GML/bin 字节相同，仍会返回 True，跳过 `check_gc_result` 的新鲜度检查并复用旧 Registry。这样即使从第 4 步正常编译，得到的 Registry 也可能不是本次 GC 配置生成的，违反 4.12 的“新生成”口径。
**涉及文件**：`/data/fjg/ceva/scripts/compile_gml.py` 第 66–83 行；`docs/design-ceva-llama2-7b-20261028.md` 第 4.12 节。
**修改思路**：把 GC 版本、`HW_TARGET`、`-l/-u` 纳入 manifest/`input_digest`，环境变化时强制重编；或者删除复用分支，每次重新编译，完全回到 4.12。补“HW_TARGET 变化后不复用”“GC 版本变化后不复用”的测试。

### 问题4（低）

**问题说明**：测试没有覆盖 R2-2/R2-3 的关键失败路径。`tests/test_compile_gml.py::test_run_all_gml_bundle_from_late_step_needs_marker` 只验证“标记缺失”时拒绝（第 155–167 行），没有验证标记存在但编译失败、包在相同路径重导、或 `compile_manifest.json` 有 `problems` 时也必须拒绝；这正是问题 1 的复现条件。`tests/test_standalone_ini.py::test_standalone_layer_fails_when_run_fails_with_old_outputs`（第 125–135 行）只读取 `standalone_layer.sh` 的文本并断言几个字符串，没有实际执行脚本，也没有模拟 `run_tools.py` 失败后旧输出仍在的场景；即使脚本退出逻辑写反，该用例仍可能通过。
**涉及文件**：`/data/fjg/ceva/tests/test_compile_gml.py` 第 155–167 行；`/data/fjg/ceva/tests/test_standalone_ini.py` 第 125–135 行。
**修改思路**：为 `compile_gml` 构造“失败后 marker/gmls 残留”的目录，断言 `run_all.sh --from 5` 非零；为单层脚本引入可替换的 `run_tools`/环境变量，实际执行一次失败+旧输出的场景，断言非零且不打印成功。若脚本不便单测，可把“执行/校验/比较”抽成可测试的 Python 函数。

### 问题5（低）

**问题说明**：R2-6 已允许 int8 模型权重，但 `PIMAttrDefs.td` 的属性说明仍写「`elemBits` is 4 for a model weight and 8 for an activation used as one」（第 733 行），`Dialect.cpp` 的非法位宽错误信息也仍写「8 (activation used as a weight)」（第 558–560 行）。这会让后续 IR 作者误以为 int8 模型权重仍不合法，与当前 `MatmulOp::verify` 要求 int8 模型权重必须配 `fixed2float` 的新契约不一致。
**涉及文件**：`FlagTree/include/triton/Dialect/TritonPIM/IR/PIMAttrDefs.td` 第 733 行；`FlagTree/lib/Dialect/TritonPIM/IR/Dialect.cpp` 第 556–567 行。
**修改思路**：把说明和错误信息改为「模型权重 4 或 8 位；activation-as-weight 为 8 位」，并补充 int8 模型权重只在静态量化输出头使用、必须配 `fixed2float` 的说明。

### 问题6（低）

**问题说明**：实施文档 3.1 的改动计数仍未按实际状态更新：写「新增文件数 flagos-pim-compiler 5」「修改文件数 26」，但当前未跟踪文件为 8 个（代码 3 + `docs/request`、`docs/design`、本文档与两份评审文档），已跟踪修改为 31 个；5.2 的偏差清单也没有记录 `standalone_layer.sh` 把厂商比较改成可选参数这一接口偏差。R2-8 声称已统一数字，但 3.1 与 5.2 仍有遗漏。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 3.1、5.2 节；当前 `git status` 的 8 个未跟踪文件与 31 个修改文件。
**修改思路**：按当前 `git status` 重新统计 3.1；把可选厂商比较写入 5.2（或按问题 2 改成必填后删掉该偏差），并复核 4.1/4.2/7.3 的数字与本次实际执行结果一致。
