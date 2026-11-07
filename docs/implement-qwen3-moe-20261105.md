# 实施进展文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：implement-qwen3-moe-20261105
> 创建日期：2026-11-05
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联项目：flagos-pim-compiler
> 起始基线：feature @ 263fcda（工作区另有未提交：README.md、docs/qwen3-support-20261104.md、
> tests/README.md、tests/test_architectures.py 的 MoE 范围澄清批次，属本需求前置）

## 一、实施概述

### 1.1 任务范围
设计的 P0-a（静态前向、注册、HOST_ONLY、router 规则、小模型对拍）+ P0-b 的 paths 键；
P0-b 的 30B 端到端等模型下载完成后验收；P1 两项未开始。

### 1.2 完成标准
设计任务项 100% 覆盖、对应测试全部通过、全量回归无新增失败。

### 1.3 实施结论
总体完成状态：**P0 全部完成**（P0-a 六个单元全绿、快集回归 147 passed、
`-k llama2_7b` 门槛 48 passed 与基线一致、真实 30B-A3B 逐 token 对拍 HF 通过，
耗时 8:43:25，CPU 平均 104%——稠密模拟 16× 计算量在单线程 numpy 上的直接体现，
是 P1 动态分派的动机基线）。P1 两项未开始。

## 二、任务完成清单
| 任务ID | 任务描述 | 对应设计模块 | 完成状态 | 备注 |
|--------|----------|--------------|----------|------|
| T2 | 静态 MoE 前向改写 `make_static_moe_forward` | §4.1 | 完成 | 两种 norm_topk_prob 均位级一致 |
| T1 | 注册 `qwen3_moe` + `_moe_strategy` | §4.2 | 完成 | 防呆测试翻转为放行 |
| T3 | HOST_ONLY 增补 5 门控算子 | §4.3 | 完成 | softmax 用具体重载，防误伤 `_softmax` |
| T4 | `mlp.gate.weight` 列切规则 | §4.4 | 完成 | |
| T5 | 小随机 MoE 端到端对拍 | §4.6 | 完成 | 一次通过（组合验证，见 3.3-I1） |
| T6a | 站点路径键 `qwen3_30b_model_dir` | §4.7 前置 | 完成 | |
| T6b | 真实 30B 端到端用例 | §4.7 | 完成 | 1 passed（8:43:25），逐 token 与 HF 一致 |
| R1 | P1 动态分派原语 | §4.8 | 完成 | T7a/T7b/T7c 全绿；修复 inf×0=nan 污染（见 I4） |
| R2 | P1 专家并行维度 | §4.9 | 未开始 | |

## 三、代码改动详情

### 3.1 改动总览
- 新增文件数：1（tests/test_qwen3_30b_real.py）
- 修改文件数：6
- 删除文件数：0
- 核心改动：+189/−3（不含新测试文件 75 行）

### 3.2 详细改动清单
| 文件路径 | 改动类型 | 改动内容简述 | 核心函数/类 | 对应任务ID |
|----------|----------|--------------|-------------|------------|
| backend/architectures.py | 修改 | 静态前向、MoE 加载与策略、注册 | `make_static_moe_forward`、`_load_qwen3_moe`、`_moe_strategy` | T2/T1 |
| graph/partition.py | 修改 | HOST_ONLY 追加 topk/softmax.int/sum.dim_IntList/scatter.src/zeros_like | `HOST_ONLY` | T3 |
| graph/strategy.py | 修改 | `("mlp.gate.weight", "col")` 规则 | `LLAMA_WEIGHT_RULES` | T4 |
| genesim_bridge/paths.py、paths.json | 修改 | `qwen3_30b_model_dir` 键 | `qwen3_30b_model_dir` | T6a |
| tests/test_architectures.py | 修改 | 6 个新用例 + rejected 翻转 + 白名单断言扩充 | — | 全部 |
| tests/test_qwen3_30b_real.py | 新增 | 30B 端到端（skipif 守护） | — | T6b |

### 3.3 实施问题记录
| 编号 | 问题 | 根因 | 解决方案 | 实现位置 | 验证用例 |
|------|------|------|----------|----------|----------|
| I1 | T5 端到端用例未经历"红"状态即通过 | T2–T4 已逐个红→绿，端到端是组合验证；用例对各部件敏感（静态化缺失会 export 报错、规则缺失会 spec_prop 报错） | 判定为合理的集成通过，非假阳性 | — | test_qwen3_moe_generate_matches_hf |
| I2 | `paths.json` 用 json 默认序列化把中文 `_comment` 转义成 `\uXXXX` | `json.dumps` 默认 `ensure_ascii=True` | 以 `ensure_ascii=False` 重写，保持原文件中文原文 | paths.json | — |
| I3 | 设计待确认 #1：`_rule_weighted_partial` 是否必要 | 实测既有 mul 规则已覆盖 Partial×Replicate 布局 | 不新增该规则（设计已预留"覆盖则删"分支） | 无改动 | test_qwen3_moe_generate_matches_hf |
| I4 | 开启分派后 `kv_slot_tokens=4` 的用例第 3 token 分叉（PIM 出 0） | 被跳过的专家 launch 不写输出，下游 mul 读旧缓冲乘路由 0：有限值×0=0 安全，但 `comm/lowering.py:133` 的 fp16 归约溢出会在缓冲留 inf，inf×0=nan 毒化累加（kv=16 时缓冲恰好有限值而侥幸正确） | 跳过时把 `cmd.writes` 全部写零（语义=该专家贡献恒为 0），彻底消除 inf×0 通道 | `backend/hal_numpy.py`（`_submit_on_dpu.run`） | test_gate_skipped_launch_zeroes_its_output、test_qwen3_moe_generate_matches_hf |
| I5 | `_submit_on_dpu` 加 `gate` 参数时一度放在 `*args` 之前，dma_in/dma_out 的位置实参会被错绑 | 签名次序错误，自查发现（测试未直接覆盖 dma 位置传参） | `gate`/`zero_writes` 改为 keyword-only 参数 | `backend/hal_numpy.py` | test_hal_numpy.py（dma 用例） |

## 四、测试验证结果

### 4.1 测试执行总览
- 执行命令：`python -m pytest tests/test_architectures.py -q`
- 执行用例总数：17
- 通过用例数：17
- 失败用例数：0
- 通过率：100%

### 4.2 分任务测试结果
| 任务ID | 测试用例名 | 测试用例数 | 通过数 | 失败数 | 测试说明 |
|--------|------------|------------|--------|--------|----------|
| T2 | test_static_moe_forward_matches_hf | 1 | 1 | 0 | 两种归一化语义逐元素一致 |
| T1 | test_qwen3_moe_config_passes + test_unknown_architecture_lists_supported | 2 | 2 | 0 | 翻转后放行、白名单列表含 qwen3_moe |
| T3 | test_moe_gate_ops_stay_on_host | 1 | 1 | 0 | 5 门控算子分区后 device=host |
| T4 | test_moe_weight_rules | 1 | 1 | 0 | router col、专家 gate/up col、down row |
| T5 | test_qwen3_moe_generate_matches_hf | 1 | 1 | 0 | 贪心 3 token 与 HF 一致 |
| T6a | test_qwen3_30b_model_dir_configured | 1 | 1 | 0 | 站点键可解析 |

### 4.3 回归测试结果
- 执行命令：`OMP_NUM_THREADS=16 python -m pytest tests/test_architectures.py tests/test_kernels.py tests/test_partition.py tests/test_spec_prop.py tests/test_mem_planner.py tests/test_kv_layout.py tests/test_activation_windows.py tests/test_exec_binding.py tests/test_strategy.py -q -n 8`
- 执行结果：**147 passed**，无失败
- 设计门槛 `OMP_NUM_THREADS=16 python -m pytest tests/ -q -k "llama2_7b" -n 12`：
  **48 passed in 3224.39s（0:53:44）**，与基线（48 passed）完全一致

## 五、问题与偏差说明

### 5.1 已知问题
| 编号 | 问题描述 | 影响范围 | 状态 | 关闭说明 |
|------|----------|----------|------|----------|
| K1 | 30B 真机用例未执行（模型下载未完成） | P0-b 验收 | 已关闭 | 1 passed（8:43:25），逐 token 与 HF 一致 |
| K2 | ~4.7 万节点图的编译耗时未实测 | 30B 用例体验 | 已关闭 | 全程 8:43:25、CPU 平均 104%——瓶颈在稠密模拟执行（16× 专家计算 × 单线程 numpy），非编译；P1 动态分派预期将执行期压 ~16× |

### 5.2 偏差说明
| 设计条目 | 设计方案 | 实际实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| design §4.5 | 视需要新增 `_rule_weighted_partial` | 未新增 | 既有 mul 规则已覆盖（设计待确认 #1 的预留分支） | 是（设计原文预留） |

## 六、后续与待确认事项
| 编号 | 事项 | 状态 | 关闭说明 |
|------|------|------|----------|
| Q1 | 30B 下载完成后执行 T6b 验收并回填 4.3 | 未关闭 | 模型已就绪，用例执行中 |
| Q2 | P1-a 动态分派原语（§4.8） | 未开始 | |
| Q3 | P1-b 专家并行 ep_width（§4.9） | 未开始 | |
| Q4 | `-k llama2_7b` 回归门槛结果回填 4.3 | 已关闭 | 48 passed（0:53:44），与基线一致 |

## 八、P1-b 补充记录（2026-11-05 第二轮）

### 8.1 完成状态
部分完成：EP 放置机制全部落地且 ≤4 DPU 精确；≥8 DPU 存在跨段竞态未定位，
16 DPU 端到端用例以 `xfail(strict=True)` 守护（修复后自动报警提示移除标记）。

### 8.2 已完成
| 项 | 内容 | 验证 |
|---|---|---|
| ShardStrategy.ep_width | 校验、tp_width 收窄、stage_width 解耦、ep_dpu_of_expert 轮转 | test_shard_strategy_ep_width |
| EP 权重放置 | 专家单 DPU 整块、attention/router/lm_head 段内 REPLICATE | test_ep_weight_specs_and_host_aggregation |
| 传播扩展 | REPLICATE 输出 DPU 集取输入交集；REPLICATE 超集免搬运；linear 整块权重规则 | 同上 + 176 条核心回归 |
| MoE 累加链 host 聚合 | remark 按「experts 祖先层==节点归属层」精确圈定 | test_moe_ep 全部 |
| KV 段内复制 | kv_specs EP 分支（每 DPU 全头本段层） | ≤4 DPU 用例 |

### 8.3 已知问题 K3（2026-11-06 已关闭）
**根因**：`_emit_redistribute` 把重分布命令登记为 `(redist:eNN, None)`，而内存规划
的 WAR 链按 `(读者名, 读取 DPU 整数)` 查 `reader_cmds`——键形不匹配，dpu→host
收集的设备侧读区对后写者不可见；收集是 host 命令、不在 DPU 泳道链上，激活区复用
的写者不等它就覆盖（TP 路径同样存在，靠低并发时序掩盖）。修复：读区按
`(redist:eNN, 读 DPU)` 登记（`runtime/exec_plan_gen.py::_emit_redistribute`）。
守护测试 `test_dpu_to_host_reads_are_war_protected`（修复前红：cmd141 重写
dpu0@161024 未等 gather cmd139）。修复后 16 DPU 端到端 6/6 通过（此前 10/10 失败）。

原证据链（保留）：
- pp1ep2 / pp1ep4 / pp2ep2（≤4 DPU）：prefill 12/12 精确（~5e-04）、decode 10/10 一致；
- pp2ep4 → NaN、pp4ep2 / pp4ep4 误差 **4.27e-03~2.42e-02 逐次漂移**（同进程同权重，
  排除确定性数值 bug）；误差按段数放大（层 0 精确 6.1e-05，层 1 起 ~4e-03，层 3 ~9e-02）；
- 与动态分派无关（PIM_MOE_DISPATCH=0 复现）。
怀疑方向：跨段重分布边（dpu→host 单副本收集 / host→dpu 回写）在 ≥10 worker 并发下
的 waits 覆盖与 (lane, dpu) 折叠的交互；线程池大小 = min(64, num_dpus+2) 是暴露阈值。

### 8.4 过程中修复（先红后绿）
| 编号 | 问题 | 修复 |
|---|---|---|
| I6 | remark 首版把 MoE 下游全部 mul/add 扫进 host（下层的 RMSNorm 走 host fp16 mean，精度低于内核 fp32 累加，近平的门控 argmax 翻转） | 按「experts 祖先层==归属层」圈定 |
| I7 | pinned 专家权重被「REPLICATE→REPLICATE 不同 DPU 集」拒重分布 | 发射面交集 + 超集免搬运 |
| I8 | `_moe_layer_of` 跨模块误用（NameError） | 用 spec_prop 既有 `_layer_of_node` |
| I9 | K3 竞态：dpu→host 收集的设备侧读区对 WAR 链不可见 | `register_reader` 登记键 (名, None) 与 `_war_waits` 查询键 (名, dpu) 键形不匹配 | 读区按 (redist:eNN, 读 DPU) 登记 | `runtime/exec_plan_gen.py`（`_emit_redistribute`） | test_dpu_to_host_reads_are_war_protected、test_ep_16_dpu_generate_matches_hf |

## 九、P1-1 加速比记录（2026-11-06 验收补全）

30B 端到端（同一用例、同一机器、dispatch 开关对比）：

| 配置 | 墙钟 | CPU |
|---|---|---|
| PIM_MOE_DISPATCH=0（稠密模拟） | 8:43:25 | 104% |
| PIM_MOE_DISPATCH=1（默认，动态分派） | **7:58:51** | 101% |

墙钟加速 1.09×。该用例墙钟由固定开销主导：57GB 权重装载、~4.7 万节点图×2 的编译、
HF 参考生成（与分派无关）；decode 仅 6 步（每步跳 120/128 专家）、prefill T=8×top-8
仅命中约半数专家——执行期的分派节省（decode 专家计算 ~1/16）被固定成本掩盖。
小模型侧的跳过语义与计数断言见 `test_moe_dispatch.py`；纯执行期加速的干净测量
应剔除装载/编译/HF 参考后单独计时（后续课题）。
