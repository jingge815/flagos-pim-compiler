# 测试集精简与结果报告设计

## 目标

将 `python -m pytest tests/ -x -q` 从多个重复的真实 LLaMA-2-7B 完整推理收敛为一条代表性端到端链路，同时保留轻量模型对全部 TP/PP 组合的覆盖。每次测试结束生成覆盖每个收集项的 Markdown 结果日志，便于查看具体测试项目、状态和耗时。

## 范围与约束

- 默认命令保持为：

  ```bash
  source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
  python -m pytest tests/ -x -q
  ```

- 默认集中保留一条真实 LLaMA-2-7B 的 `tp4_pp2` 完整编译、加载权重、prefill、KV 常驻与自回归解码链路，并逐 token 对齐 HF 单卡参考。
- `tp4_pp2` 同时经过 Tensor Parallel 和 Pipeline Parallel 的边界，作为真实模型代表策略。
- 全部 TP/PP 组合继续由小模型 `tests/test_strategy_sweep.py` 覆盖；它保留 `tp4_pp1`、`tp2_pp2`、`tp1_pp4` 三种合法策略的完整 prefill/decode 数值校验和切分结构校验。
- 不新增用户关心领域的测试项目；只删除已被代表性真实 7B 测试和轻量策略遍历共同覆盖的重复真实 7B 用例。
- 不触碰当前工作区中用户已有的未提交修改。

## 测试精简

### 真实 7B 策略验证

修改 `tests/test_strategy_llama2_7b.py`，令参数化策略只包含 `tp4_pp2`。

保留的三条断言分别验证：

| 测试项 | 保留理由 |
| --- | --- |
| 端到端贪心解码结果逐 token 对齐 HF | GPU 上真实 7B 的编译、推理和仿真输出主链路 |
| PIM KV 区逐元素对齐 HF DynamicCache | KV 张量常驻、跨解码步读写和分区归属 |
| 各 DPU 内存预算与分层 KV 配置 | 多 DPU 内存布局和 PP 层归属 |

删除的仅是同一组三条断言在 `tp8_pp1`、`tp2_pp4`、`tp1_pp8` 下的重复运行。TP-only、TP+PP 和 PP-only 分支仍在 `test_strategy_sweep.py` 的全策略执行及单元测试中被覆盖；真实模型代表策略则覆盖 TP 与 PP 的组合交互。

### 不在本次删除范围内的真实 7B 测试

以下文件验证独立目标，不能视为策略测试的重复项，保持不变：图导出与标注、通信计划、内存规划、KV 布局、编排器、解码循环、并发、自然语言 prompt、算子编译端到端和 GeneSim 桥接。

## 结果日志

在 `tests/conftest.py` 实现 pytest 会话钩子。

1. `pytest_sessionstart` 创建 `test-results/`，删除旧的 `test-results/pytest-latest.md`，保证每次运行只留下本次结果。
2. `pytest_collection_modifyitems` 保存本次收集到的 node id 和测试名称；即使使用 `-x` 早停，也可标记未执行的收集项。
3. `pytest_runtest_logreport` 记录每项 call 阶段的 `PASS`、`FAIL`、`SKIPPED` 与耗时；setup/teardown 出错单独标为 `ERROR`。若一个测试在 setup 中被跳过，记录为 `SKIPPED`。
4. `pytest_sessionfinish` 生成 Markdown。表格列为“测试文件 / 测试项 / 状态 / 耗时（秒）”；未进入运行阶段的收集项标记为 `NOT RUN`。文首给出开始/结束时间、pytest exit code、总耗时和状态汇总，文末在失败或错误时附首段 traceback。
5. 控制台在会话结束打印报告绝对路径。`-q` 的进度点行为保持不变，详细结果以日志为准。

示例：

| 测试文件 | 测试项 | 状态 | 耗时（秒） |
| --- | --- | --- | ---: |
| `tests/test_strategy_llama2_7b.py` | `test_real_llama2_7b_inference_matches_hf_under_every_strategy[tp4_pp2]` | PASS | 312.48 |
| `tests/test_strategy_sweep.py` | `test_every_strategy_decodes_the_same_tokens_as_single_card_pytorch[tp1_pp4]` | PASS | 1.03 |
| `tests/test_*.py` | 未在 `-x` 前运行的收集项 | NOT RUN | — |

报告是运行产物，加入 `.gitignore`，不提交历史测试输出。

## 用户关心目标的现有覆盖审计

| 目标 | 现有测试 | 结论 |
| --- | --- | --- |
| GPU 上完整大模型编译、推理、仿真输出 | `test_strategy_llama2_7b.py`（保留 `tp4_pp2`）、`test_opcompiler_e2e_llama2_7b.py` | 有，代表策略保留 |
| 导出无 graph break、算子分组符合规则 | `test_partition.py`、`test_placement_export.py`、真实 7B 图测试 | 有 |
| 多 DPU 独立地址空间、无隐式共享 | `test_dpu_sdk.py`、`test_hal_numpy.py` | 有 |
| GeneSim 输出仿真数据 | `test_genesim_bridge.py` | 有桥接/成本数据测试；未见实际 GeneSim 二进制端到端调用 |
| 中间张量分布、重分布边、全图标注 | `test_spec_prop.py`、`test_spec_prop_llama2_7b.py` | 有 |
| 图计算按设备分割、TP/PP | `test_partition.py`、`test_strategy.py`、`test_strategy_sweep.py` | 有 |
| KV 无冗余重分布、常驻和掩码 | `test_kv_layout*.py`、`test_decode_loop_llama2_7b.py`、`test_strategy_llama2_7b.py` | 有 |
| 单 DPU 算子编译、PIM MLIR 解析 | `test_opcompiler_linear.py`、`test_opcompiler_e2e_llama2_7b.py`、`test_genesim_bridge.py` | 有；`ttir→pim mlir` 编译器本体按仓库边界在 FlagTree |
| IR dump / serialize / replay | 未找到对应测试或实现引用 | 缺失，本次不新增 |
| 内存区域、容量、多 DPU 调度 | `test_mem_planner*.py`、`test_exec_plan_gen.py`、`test_concurrency_llama2_7b.py` | 有 |
| LLaMA 2 解码与代价估计 | `test_decode_loop_llama2_7b.py`、`test_strategy_llama2_7b.py`、`test_genesim_bridge.py` | 有成本桥接；未见真实 GeneSim 性能评估执行 |
| PIM MLIR 与 PIM SDK 语义一致、PU 映射用于性能评估 | `test_dpu_sdk.py`、`test_genesim_bridge.py` | 部分覆盖；未找到针对 SDK 语义一致性或 PU 映射的专门端到端测试 |
| 算子语义/数据类型/Placement/Memory layout | `test_opcompiler_linear.py`、`test_op_contract.py`、`test_spec_prop.py`、`test_mem_planner.py` | 有主要覆盖 |
| 计算/访存/通信代价与不同 IR 评估 | `test_genesim_bridge.py`、`test_comm_plan.py` | 有静态估计与趋势测试；未见完整算子 IR 驱动的端到端多维度回归 |

## 错误处理与验证

- 日志写入必须在测试失败、跳过、`-x` 早停及 collection/setup 错误后仍执行，且不覆盖 pytest 原始退出码。
- 为报告钩子先写独立 pytester 测试：验证旧日志被替换、PASS/FAIL/NOT RUN 表格状态和 `-x` 早停状态。
- 先运行该测试确认新功能未实现时失败；再实现最小钩子并重跑。
- 最后运行轻量测试集、报告钩子测试和完整收集；真实 7B 代表用例按环境可用性单跑验证。完整默认集作为最终验证运行一次，但不以 40 分钟的旧基线判断精简效果。
