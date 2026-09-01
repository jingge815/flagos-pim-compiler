# CLAUDE.md

存算一体（PIM）大模型推理编译器：图层编译器（编译期产静态蓝图）+ 主机编排器（运行时照蓝图执行），在 NumpyBackend 上与单卡 PyTorch 逐元素对齐。

本仓覆盖问题 1/2/3/6/7/8 + 问题 4（GeneSim 成本桥接）。问题 5（算子编译器 `ttir→pim mlir`）在 FlagTree 仓，只通过 kernel 二进制 + 编译期算子契约对接。

## 环境

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
```

torch 2.9.1 / transformers 4.57.6 / python 3.10.20，CUDA 可用。**每个新 shell 都要先 source**，否则 `import torch` 失败。

## 技术方案

完整方案在 `docs/spec.md`（2494 行，287KB）。**禁止整篇读入**——按 `docs/spec-index.md` 的行号表用 `Read(offset, limit)` 只读当前任务对应的那一节。

## 目录职责


| 目录              | 内容                                                                                                                            | 问题 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---- |
| `contracts/`      | `PIMTensorSpec` / `ExecutionPlan` / `node.meta` 字段约定——全仓唯一真源                                                        | 共用 |
| `graph/`          | `partition.py` 图拆分打标、`strategy.py` 切分策略（TP/PP/混合）、`spec_prop.py` 切分传播 + redistribute 标注                    | 1、2 |
| `comm/`           | `plan.py` 编译期通信计划表、`lowering.py` 运行时 redistribute→DMA                                                              | 3    |
| `memory/`         | `kv_layout.py` KV 布局 + runtime、`mem_planner.py` 三区 offset 规划                                                             | 7、8 |
| `runtime/`        | `compile.py` 统一编译入口（模型+策略→蓝图）、`exec_plan_gen.py` 生成 ExecutionPlan、`executor.py` 解释 + 解码循环              | 6    |
| `backend/`        | `dpu_sdk.py` 厂商 SDK（pre-g-driver-api）的 numpy 镜像 / `hal_numpy.py` 异步 HAL（存储架在 dpu_sdk 上）/ `hal_vendor.py` 真硬件 | 底座 |
| `genesim_bridge/` | `ir_cost.py` 从 TTIR / pim mlir 抽 flops/data_bytes（评估旁支）                                                                 | 4    |
| `tests/`          | `assert_node_matches_ref.py` 逐节点对拍器 + 各模块单测                                                                          | 验证 |

`contracts/` 是地基：改任何字段先 grep 调用方，改完同步全链路。各组只通过 `node.meta` 解耦，不私自约定字段。

## 控制代码量（最重要）

历史教训：AI 辅助开发最容易堆出大量重复、投机、永不执行的代码。以下是硬约束。

- **先搜索再写**。写新函数前 grep 是否已有同类实现；优先改现有函数，而不是加一个并列的新函数。
- **最小实现**。只实现当前步骤需要的路径。不写「以后可能用到」的分支、参数、配置项。
- **不预造抽象**。工厂、注册表、插件机制、基类——只有当已经存在 2 个以上真实实现时才引入。`hal_numpy`/`hal_vendor` 是唯一预先允许的双实现抽象（方案要求「换后端不换数值」）。
- **不写防御性兜底**。不用 try/except 或默认值掩盖上游错误；契约不满足就直接抛，让 bug 立刻暴露在对拍器里。
- **单文件优先**。一个模块先写成一个文件，超过 ~400 行再考虑拆；不要一上来就铺开子包。
- **删优于加**。发现死代码、重复逻辑、被替代的旧路径，直接删掉，不要注释掉留着。
- 每次改动结束时报告净增删行数（如 `+120 / -35`），量级异常时说明原因。
- 第 2/3 阶段特性（序列变长、代价模型、自动切分、算子融合、异步 dispatch）**当前不实现、不预留接口**。

## 可读性

代码要让人和 AI 都能快速读懂、能下断点调试。

- 数据结构用 `@dataclass` + 类型标注，不用裸 dict 传结构化数据。
- 命名跟方案术语一致：`part_id`、`dpu_id`、`placement`、`mram_offset`、`valid_len`——不另起别名。
- 注释1，2句话简单描述功能，字数不要太多，不要中英文混合，除非英文专有名词，通俗易懂。
- 关键中间产物（标注图、通信计划表、`DPU_k.plan`、`ExecutionPlan`）要能 `print` 出可读文本，方便定位问题。
- 函数保持短、单一职责；避免深嵌套和长参数表。
- **关键结构体**（`contracts/` 里定义的、跨模块传递的 `@dataclass`）：字段名字已经自解释就不用注释；含义不直观的字段（如单位、取值范围、何时为 `None`）补一行说明。
- **关键函数**（模块对外接口、算法核心函数，即 `docs/<module>.md` 里列出接口签名的那些）：函数签名上方用几行说明功能、实现原理（依据方案哪一节/附录）、输入输出的含义和形状；普通内部小函数不强制。
- 函数超过 ~40 行时，内部按步骤分段落加一行短注释标出关键区域（如 `# 1. 按行切分权重` `# 2. 计算 mram_offset`），不用逐行注释。

## 测试

每个模块的代码和测试同一次改动内完成，不留「测试稍后补」。

- 单测放 `tests/test_<module>.py`，用 pytest，跑：`python -m pytest tests/ -x -q`。
- 编译期模块（问题 1/2/8）的判据是方案里的手推结果：附录 A 的 placement 推演、三区 offset 无重叠、容量 ≤ 8GB、torch.export 无 graph break。
- 运行时模块（问题 3/6/7）的判据是**与单卡 PyTorch 逐元素对齐**，用 `tests/assert_node_matches_ref.py`：按 placement 把分片合并回完整张量再比，第一处不匹配即定位到具体 node。
- `hal_numpy` 必须有「写 DPU_i 不影响 DPU_j」的独立地址空间测试。
- 不确定就跑测试，别靠读代码断言正确性。

## 文档

每次研发后在 `docs/<module>-日期.md` 留一份文档，概述修改内容，增删修改文件或函数，关键结构体，描述通俗易懂，拒绝中英文混合，除非英文专有名词，推荐画流程图，或者表格，描述已实施的测试，当前存在的不足。

## 提交

- 身份 `fengjg <fengjg@ios.ac.cn>`（已配为本仓 local identity）。不加 `Co-Authored-By`，不署 Claude/AI。
- 仅在用户明确要求时提交；用 `git add <具体文件>`，不用 `git add .`。
- 提交信息用简洁句，说明改了什么、为什么。
