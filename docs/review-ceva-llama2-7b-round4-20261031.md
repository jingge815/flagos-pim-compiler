# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理

> 文档编号：review-ceva-llama2-7b-round4-20261031
> 创建日期：2026-10-31
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029
> 关联上一轮评审：review-ceva-llama2-7b-round3-20261031

## 一、评审概述

本轮复核了 R3-1～R3-6 的修复以及新增的 G-1、G-2：R3 的 `ddr_ok.json`、单层必填厂商参考、`gc_env` 复用条件、实际执行测试和 FlagTree 说明均已落地，当前工作树重新导出 prefill/decode/lm_head 第 0 层后与 `bundles/ours_e2e` 逐字节相同（分别 3104、3125、25 个文件，`diff` 为 0）。测试结果：`test_ceva_export.py` 45 passed，CEVA `tests/` 35 passed，FlagTree lit 265 passed；快速回归首次 1580 passed、8 skipped，另有 11 项因 `triton-opt` 瞬时 `PermissionError` 失败，单独重跑这 11 项全部通过。65 图全流程在 G-2 之后没有重跑，现有 `ours_full`/`ours_e2e` PASS 记录仍是 G-2 之前的产物；本轮共发现 6 个问题：中 1、低 5，未发现会改变第 0 层导出结果的代码缺陷。

## 二、修改建议

### 问题1（中）

**问题说明**：需求 P0-2/P0-5 要求 65 张图全部由我方生成并编译、全模型 token 与基线一致；设计 6.2/7.3 也以 65 图编译和单层/全模型运行作为完成判据。G-2 修改了共享的图编译、算子编译和 GML 序列化路径（`graph/fuse_rope.py`、`graph/kv_dma_pass.py`、`graph/quant_pass.py`、`gml_bridge/from_fx.py`、`gml_bridge/export.py`、`opcompiler_bridge/oplevel_emitter.py`、`contracts/unified_ir.py`），但实施文档 7.3 没有 G-2 之后的 65 图导出、GC、DDR 或仿真结果；现有 `ours_full`/`ours_e2e` 的 PASS 记录不能证明当前 G-2 工作树的 prefill 全 32 层产物。第 0 层三张图逐字节相同只能覆盖代表性样本，不能替代 65 图编译和至少一次全模型运行。
**涉及文件**：`gml_bridge/export.py` 第 519 行（注意力分数分组口径）；`graph/quant_pass.py` 第 130–146 行；`gml_bridge/from_fx.py` 第 160–174、2268–2290 行；`docs/implement-ceva-llama2-7b-20261029.md` 7.3 节。
**修改思路**：在当前 G-2 工作树上重新执行 65 图导出、GC、DDR 和至少 1 步全模型仿真，归档 GML 逐节点核对、GC success、token 判定；若时间不足，至少完成 prefill/decode 各一层的新旧逐字节比对并把结果补进 7.3。不能在未重跑的情况下把 65 图或全模型验收标为已验证。

### 问题2（低）

**问题说明**：G-1 要求无 GPU 时精化走 oplevel，sidecar 按实际档位命名，不能覆盖 pimir 那份。`step_a2_refine_pimir_ir` 的跳过条件（`scripts/run_full_pipeline.py` 第 165–168 行）只比较现有 `llama2_7b_pimir.ir` 的算子编号，不检查当前 `_refine_ir_level()`、目标 sidecar 是否存在及其 `ir_level`。在无 GPU 的机器上，如果上一次 GPU 运行留下了编号一致的 pimir IR/sidecar，本次会直接跳过精化，旧的 pimir sidecar 不会被 oplevel 结果替换；新增的 `test_refine_sidecar_is_named_after_ir_level`（`tests/test_ir_cost_eval.py` 第 972–991 行）只覆盖“没有旧 IR”的非跳过路径，无法发现这一点。
**涉及文件**：`scripts/run_full_pipeline.py` 第 160–176 行；`tests/test_ir_cost_eval.py` 第 972–991 行。
**修改思路**：把 `_refine_ir_level()` 与期望 sidecar 的存在/`ir_level` 纳入跳过条件，档位不一致时重新精化；补一个“已有 pimir IR、无 GPU”的用例，断言会生成/刷新 oplevel sidecar。

### 问题3（低）

**问题说明**：设计 4.13 要求 DDR map 中的 Registry 路径全部指向本次运行，P0-4 要求 Registry 来源可核对。`verify_run.py` 的 Registry 检查用 `read_text(...) or ""` 处理 map（第 193–197 行），map 文件缺失或存在但没有任何 `SubgraphRegistry` 路径时，`foreign` 为空、`registry_mismatches` 也返回空，随后仍会添加 PASS 项“Registry 全部来自本次编译/与清单摘要一致”。如果 `run` 和 token 文件已存在而 map 被删除或截断，判定可能放行一个无法证明 Registry 来源的运行。
**涉及文件**：`/data/fjg/ceva/scripts/verify_run.py` 第 189–202 行；`/data/fjg/ceva/scripts/verify_run.py` 第 75–99 行。
**修改思路**：在 bundle 模式下要求 `prompt_map.ini`、`decode_map.ini`、`lm_head_map.ini` 都存在且每个 map 至少解析出一条 Registry 路径，否则直接 FAIL；补一个删除/清空 map 的负例。

### 问题4（低）

**问题说明**：`gc_worker.sh` 用 `case "$2" in *data*)` 拒绝含 “data” 的 GC 路径，以规避厂商 test_app 在 “data” 子串处截断 net.ini。但 `compile_gml.py` 第 141 行把 `RUN_DIR` 的名字拼进 `/tmp/ceva_gc` 的软链名，而 `run_all.sh` 的 `-n` 允许字母、数字、`._-`；用户使用 `-n data_e2e`、`-n metadata` 等合法名字时，软链路径本身含 “data”，会被误判为非法路径而失败。路径检查应该只针对厂商实际收到的 `-d` 目标/其规范化路径，而不是运行名派生的软链名。
**涉及文件**：`/data/fjg/ceva/scripts/gc_worker.sh` 第 8 行；`/data/fjg/ceva/scripts/compile_gml.py` 第 141 行；`/data/fjg/ceva/run_all.sh` 第 52 行的运行名校验。
**修改思路**：软链改用只含槽号和图名的稳定名字，或对运行名做不含 “data” 的哈希/转义；补一个 `-n data_e2e` 的编译用例。

### 问题5（低）

**问题说明**：`run_all.sh` 第 197 行用 `grep -v "=$RUN_DIR/gmls/"` 过滤本次 Registry，`$RUN_DIR` 含运行名，而运行名允许 `.`；`grep` 的基本正则会把 `.` 当通配符，例如运行名 `ours.full` 时，路径 `oursXfull/gmls/` 也可能被当成“本次路径”过滤掉，从而漏报外部 Registry。`verify_run.py` 的 `startswith` 检查没有这个问题。
**涉及文件**：`/data/fjg/ceva/run_all.sh` 第 197 行。
**修改思路**：改成 `grep -F -v "=$RUN_DIR/gmls/"`（或对路径做 `realpath` 后按固定字符串比较），并补一个运行名含 `.`、map 指向另一个相似路径的测试。

### 问题6（低）

**问题说明**：文档与当前实现/测试仍不一致：实施文档 4.2 写 `test_ceva_export.py` 42 例、D1～D3 23 例，实际分别为 45 和 35 例；3.2 仍写“登记 `pim_ceva_op`”，而 G-2 已改为 `pim_static_quant`；设计 5.3 仍规定 `CEVA_OP_META_KEY="pim_ceva_op"`，代码/实施 7.1 已改为通用 `STATIC_QUANT_META_KEY`；设计 7.2 的 `standalone_layer.sh` 示例只有一个图参数，当前脚本要求 `<图> <输入> <输出> <厂商同层输出目录>` 四个参数；实施文档 5.2 还写“设计 4.12 允许复用”，但设计 4.12 的原文是“每张图都有新生成的 Registry，修改时间晚于本次开始”，复用只是本轮实现选择。数字和契约口径不统一会影响后续验收追溯。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 56、86、90、119、5.2 节；`docs/design-ceva-llama2-7b-20261028.md` 第 5.3、7.2 节。
**修改思路**：按当前实际测试数更新 4.2/7.3；把 3.2 的 `pim_ceva_op` 改成 `pim_static_quant`，并在设计 5.3 或实施 5.2 记录该键替换；把 7.2 的单层命令补成四参数；把 5.2 的“设计 4.12 允许复用”改为“实现选择复用，条件见清单摘要”，不要改写成设计原文。
