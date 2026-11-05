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
