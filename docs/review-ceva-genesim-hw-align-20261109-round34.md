# 代码评审报告：GeneSim 对齐 CEVA 多核硬件与性能模型
> 文档编号：review-ceva-genesim-hw-align-20261109-round34
> 创建日期：2026-11-09
> 关联需求文档：request-ceva-genesim-hw-align-20261106
> 关联设计文档：design-ceva-genesim-hw-align-20261106
> 关联实施文档：implement-ceva-genesim-hw-align-20261106

## 一、评审概述
本次按实际 Git 状态合并评审 genesim 的 43 个已跟踪修改文件与 9 个新增实现/配置/测试文件、flagos-pim-compiler 的 4 个修改代码/测试文件与 1 个新增测试，并核对 FlagTree 的干净工作区和已安装工具链。本次评审未发现需求、设计、代码质量、测试完备性方面的新增或复发问题；实施文档已登记的 W4A8 物理布局、分阶段有效成本、独立校准、秒级结果门控及其他开放偏差不重复计入，此结论不代表这些未关闭项已通过完整性能验收。FlagTree 工作区无改动，未对其编造问题。

实际执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim`，46/46 个测试文件通过，累计 `Ran 1292`（其中 1 条跳过）、0 失败；`./run.sh --test predictor` 为 7/7 个文件、87 条通过，`./run.sh --test upmem_checker` 为 49 条通过。图编译器加载 `paths.json` 指定环境后，指定命令 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q` 为 36 通过、4 跳过，补跑 `tests/test_placement_export.py` 为 57 通过；三个仓库 `git diff HEAD --check` 均通过，测试前后未出现额外已跟踪代码改动。未运行约 2.5 小时的 `-k llama2_7b` 和依赖外部 CEVA 仿真器的 `run_ceva_pipeline.sh`。

执行 `/tmp/ceva_review_round31_inventory.py`，五份场景结构/正式校验均通过，实际创建核数为 1/64/64/128/128，GPU 均为 0，物理 engine 数为 4/256/64/512/128，模拟调度槽为 1/64/64/128/128，后端均为 `CEVANPMPIMCore`，容量均为 8,000,000,000 B、预留 0 B；真实 sidecar 的 6850 条记录引用 21 份 PIM MLIR，缺失和哈希错误均为 0，8 台 DPU 映射互不重复。以独立 `/tmp` 配置、结果和轨迹目录执行 `uv run python -u src/main.py --config /tmp/ceva-round34-review-smoke.yaml`，7138 条轨迹全部重新编译，其中 7074 条 `pimir`、64 条 `absorbed`、0 条模板；创建/使用/空闲核为 64/8/56，`processed_tokens=795`、`total_time_s=672.363454`、`result_maturity=model_prediction`，通信指标中的 PIM_INTER 正反向字节为 539,869,184/312,213,504 B。运行清单的配置、3 项模型/放置/请求输入、21 份 PIM MLIR 和实际 `triton-opt` 共 26 项哈希复算全部一致。该秒级结果只是假设参数下的模型预测，不作硬件校准结论。

复核现有通用 `run_full_pipeline` 产物（本轮未重跑）：10 个请求、8084 个处理 token、`total_time_s=1812.037482`；`test-results/full-pipeline/e_sim_pimir.log` 记录 7138 条轨迹全部重新编译，模型为 32 层 Llama2-7B。逐条解析当前通用与本轮 CEVA 轨迹，二者均为 7074 条 `pimir`、64 条 `absorbed`，模板版本均为 3，1024 条 SOFTMAX 均为 5 相且各有 1 条折叠工作项；通用摘要不含 `result_maturity` 或 `run_manifest` 字段。

FlagTree HEAD 为 `0d7273bac` 且工作区干净；实际使用已安装 `triton-opt` 对 `tile_to_budget_matmul.mlir` 执行 `-split-input-file -pim-tile-to-budget`，输出保留 `groupDequantAccum/groupSize`、`elemBits=4` 和 4×8×64 分块，再执行 `-pim-lower-to-emitc` 成功，二进制 SHA256 为 `ecd955ec…68d1d`。执行 `python scripts/verify_gml_artifact.py /data/fjg/ceva/bundles/ours_e2e --ceva-manifest /data/fjg/ceva/bundles/ours_e2e/bundle.json`，327/327 项通过；该 W4A8 包仍属独立 CEVA 分支，不据此认定 GeneSim 已消费同一 W4A8 产物。

复跑第 33 轮集合通信极值与替代路径探针：大 `nbytes` 和系数下溢在 `all_gather`/`scatter`、常量/序列缩放及 `direct`/`host_relay` 组合下均于首次消费前按字段拒绝；合法常量 4096 B 的两条边各计 4096 B、`1.724 µs`，`seq_len=16` 时各计 256 B、`0.764 µs`，`nbytes=4096、seq_len=10**100` 的有限正系数 `4.096e-97` 未被误判，完整流水按正字节向上取整，不会落入 0 B 快速路径。复跑第 32 轮同路径重载与原位改参探针，两个装载入口均更新为 `1.057478e-04 s`，原位修改 `LUT` 后更新为 `6.654826e-04 s`；复跑第 31 轮 sidecar 改写/恢复及稳定输入对照，TP 分片均保持首次消费的 `[0,1]`，均使用 118 条集合通信记录并生成 254 个中转节点，快照复核通过。相反假设与合法路径均已核对，未发现只修复当时探针或单例的复发。

## 二、修改建议
无
