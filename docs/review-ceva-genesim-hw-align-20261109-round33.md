# 代码评审报告：GeneSim 对齐 CEVA 多核硬件与性能模型
> 文档编号：review-ceva-genesim-hw-align-20261109-round33
> 创建日期：2026-11-09
> 关联需求文档：request-ceva-genesim-hw-align-20261106
> 关联设计文档：design-ceva-genesim-hw-align-20261106
> 关联实施文档：implement-ceva-genesim-hw-align-20261106

## 一、评审概述
本次按实际 Git 状态合并评审 genesim 的 43 个已跟踪修改文件与 9 个新增实现/配置/测试文件、flagos-pim-compiler 的 4 个修改代码/测试文件与 1 个新增测试，以及 FlagTree 的干净工作区和已安装工具链；共发现 1 项高等级问题，即集合通信的任意精度整数及派生字节系数未纳入统一数值契约，可使正字节传输免费或在正式校验后崩溃。实施文档已登记的 W4A8 物理布局、分阶段成本、独立校准、秒级门控及其他开放偏差不重复计入；FlagTree 工作区无改动，未对其编造问题。

实际执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim`，46/46 个测试文件通过，累计 `Ran 1290`（其中 1 条跳过）、0 失败；`./run.sh --test predictor` 为 7/7 个文件、87 条通过，`./run.sh --test upmem_checker` 为 49 条通过。图编译器加载 `paths.json` 指定环境后，指定两文件测试为 36 通过、4 跳过，补跑 `tests/test_placement_export.py` 为 54 通过；三个仓库 `git diff HEAD --check` 均通过，测试前后未出现额外已跟踪代码改动。

执行 `/tmp/ceva_review_round31_inventory.py`，五份场景结构/正式校验均通过，实际创建核数为 1/64/64/128/128，GPU 均为 0，物理 engine 数为 4/256/64/512/128，模拟槽为 1/64/64/128/128，后端均为 `CEVANPMPIMCore`，容量均为 8,000,000,000 B、预留 0 B；真实 sidecar 的 6850 条记录引用 21 份 PIM MLIR，缺失和哈希错误均为 0，8 台 DPU 映射互不重复。以独立 `/tmp` 配置、结果和轨迹目录执行 `uv run python -u src/main.py --config /tmp/ceva-round33-review-smoke.yaml`，7138 条轨迹全部重新编译，其中 7074 条 `pimir`、64 条 `absorbed`、0 条模板；创建/使用/空闲核为 64/8/56，`processed_tokens=795`、`total_time_s=672.363454`、`result_maturity=model_prediction`，运行清单的 21 份 PIM MLIR 哈希复算全部一致。该秒级结果只是假设参数下的模型预测，不作硬件校准结论。

复核现有通用 `run_full_pipeline` 产物（本轮未重跑）：10 个请求、8084 个处理 token、`total_time_s=1812.037482`，7138 条轨迹同样为 7074 条 `pimir`、64 条 `absorbed`，通用摘要不含 `result_maturity` 或 `run_manifest` 字段。`test-results/full-pipeline/e_sim_pimir.log` 记录本次产物的 7138 条轨迹均重新编译，模型为 32 层 Llama2-7B。

FlagTree HEAD 为 `0d7273bac` 且工作区干净；实际使用已安装 `triton-opt` 对 `tile_to_budget_matmul.mlir` 执行 `-split-input-file -pim-tile-to-budget`，输出保留 `groupDequantAccum/groupSize`、`elemBits=4` 和 4×8×64 分块，再执行 `-pim-lower-to-emitc` 成功，二进制 SHA256 为 `ecd955ec…68d1d`。在图编译器仓执行 `python scripts/verify_gml_artifact.py /data/fjg/ceva/bundles/ours_e2e --ceva-manifest /data/fjg/ceva/bundles/ours_e2e/bundle.json`，327/327 项通过；该 W4A8 包仍属独立 CEVA 分支，不据此认定 GeneSim 已消费同一 W4A8 产物。未运行约 2.5 小时的 `-k llama2_7b` 和依赖外部 CEVA 仿真器的全流程。

复跑第 31 轮 sidecar 改写/恢复探针及稳定输入对照，TP 分片均保持 `[0,1]`，执行图均使用首次消费的 118 条集合通信记录并生成 254 个中转节点，快照复核通过。复跑第 32 轮同路径重载探针，两个装载入口均由 `5.28874e-05 s` 更新为正确的 `1.057478e-04 s`，原位修改 `LUT` 成本也由 `7.10506e-05 s` 更新为 `6.654826e-04 s`；NaN 权重探针按预期在写计划前以 `SystemExit` 终止。上述历史修复的同类路径本轮未再发现问题。

执行 `/tmp/ceva_review_round33_collective_extremes.py` 复现下述问题，并用 `/tmp/ceva_review_round33_collective_routes.py` 换用 `scatter` 及另一系数路径复核：大 `nbytes` 在常量与序列缩放路径分别抛出未包装的 `OverflowError`，大 `seq_len` 在两类集合通信中都把正字节下溢为 0 B。相反假设对照未被误判：合法常量 4096 B 的两条边各计 4096 B、`1.724 µs`；`seq_len=16` 时各计 256 B、`0.764 µs`，真实 sidecar 的集合通信契约检查也全部通过。

## 二、修改建议
### 问题1（高）: 集合通信数值域不闭合，导致正字节免费或崩溃
**问题说明**：需求 R6（需求文档第 70～76 行）要求正字节跨核传输具有预期有限成本，设计 §6、§9.1（设计文档第 161～167、241 行）要求通信字节、服务和共享路径可核对，合法零成本仅限纯视图、同核复用及无传输。实施文档 round32（实施文档第 798、1941～1943 行）宣称集合通信契约已在配置校验、首次消费、生产端和完整流水统一。实际 `collective_edge_errors()` 只检查 `nbytes`/`seq_len` 为非布尔正整数，没有检查它们进入浮点成本模型后的可表示性，也没有检查 `nbytes/seq_len` 的派生系数。探针中 `nbytes=10**1000` 的正式校验均为 True：`seq_len=null` 在 `float(nbytes)` 处抛 `OverflowError: int too large to convert to float`，`seq_len=16` 在除法处抛 `OverflowError: integer division result too large for a float`；`nbytes=4096、seq_len=10**1000` 则生成 `{"Tq": 0.0}`，两条边的求值字节和路由时延均为 0。完整流水对 0 B 的检查会通过，并在 `transfer_bytes <= 0` 分支直接完成传输，因此这不是仅影响诊断的边界问题。图编译器 `_collective_entry()` 也实测接受 `10**1000`，说明生产端可以生成消费端无法正确计费的产物。
**根本原因**：第 32 轮把“原始字段是正整数”误当成“字节量在成本模型中有效”，契约停留在字段类型和符号，未覆盖任意精度 JSON 整数到浮点的转换及派生系数下溢；生产端与消费端各自重复同一套不完整检查，完整流水又把 0 当作合法快速路径，无法区分真实零传输与数值下溢。同一根因影响 `all_gather`/`scatter`、常量/随序列缩放两种系数、配置校验、sidecar 导出、执行图生成及后续通信路由。现有测试只覆盖布尔、浮点、NaN/无穷、负数和零，未覆盖任意精度大整数及正系数下溢，因此全量回归通过仍发现不了该类问题，属于上一轮集合通信数值契约修复范围不完整的复发。
**涉及文件**：`genesim/src/config_loader.py:187-259`（原始字段契约未验证可表示性及派生系数）；`genesim/src/scheduler/gene_sim_scheduler.py:7076-7086`（直接除法和浮点转换）、`:9095-9134`（0 B 通过有限非负检查）、`:9364-9377`（0 B 直接完成传输）；`flagos-pim-compiler/genesim_bridge/placement_export.py:96-140`（生产端同款不完整契约）；`genesim/tests/sim/test_collective_edges.py:270-341` 与 `flagos-pim-compiler/tests/test_placement_export.py:416-447`（缺少大整数、下溢及派生系数的端到端反例）。依据为 `docs/request-ceva-genesim-hw-align-20261106.md:70-76`、`docs/design-ceva-genesim-hw-align-20261106.md:161-167,241`、`docs/implement-ceva-genesim-hw-align-20261106.md:798,1941-1943`。
**修改思路**：把“原始字节量—导出序列长度—每 token 系数—运行形状求值”作为一个统一数值契约，而不是只在当前两行转换外接异常处理。生产端、配置校验和调度器首次消费都应验证：`nbytes` 与非空 `seq_len` 不仅是非布尔正整数，还能被成本模型安全转换为有限正数；常量分支的字节量和序列分支的 `nbytes/seq_len` 必须有限且严格大于 0，溢出、下溢或无法表示时在写出 sidecar/修改执行图之前按字段失败，不得钳成 0 或留到路由阶段崩溃。消费端只使用已通过同一契约计算的系数，通用有限非负检查保留为最后防线。测试同时覆盖两类集合通信、两种序列模式、生产端导出、配置校验、执行图与完整流水，加入 `10**1000`、系数下溢、合法 `seq_len=None` 和普通正整数对照，并断言实际传输字节与通信服务时间，避免只验证辅助函数或只增加当前探针的特判。
