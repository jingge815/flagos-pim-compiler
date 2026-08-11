# AGENTS.md

本文件为 Codex 及其他编码 agent 在本仓库工作时的指引。

## 项目定位

存算一体（PIM）大模型推理编译器，FlagOS 生态下的图层编译器 + 主机编排运行时。HuggingFace 模型经 `torch.export` 得到静态图后，编译期完成图拆分、切分传播、通信计划、三区内存规划，产出静态蓝图；运行时由主机编排器在 NumpyBackend（多 DPU 独立地址空间模拟）上驱动执行，与单卡 PyTorch 逐节点数值对齐。

本仓覆盖问题 1/2/3/6/7/8 + GeneSim 成本桥接（问题 4）。算子编译器（问题 5，`ttir→pim mlir`）在 FlagTree 中开发，通过 kernel 二进制 + 编译期算子契约与本仓对接。

## 目录职责

- `contracts/` — 共享数据契约（`PIMTensorSpec`、`ExecutionPlan`、`node.meta` 字段约定），全仓 import 的唯一真源（任务全组共用）
- `graph/` — 图拆分打标、切分传播 / redistribute 标注（任务 1、2）
- `comm/` — 通信库：编译期通信计划表 + 运行时 redistribute→DMA 下降（任务 3）
- `memory/` — KV 布局与 runtime、三区静态内存规划（任务 7、8）
- `runtime/` — 编排器：ExecutionPlan 生成 + 解码循环执行（任务 6）
- `backend/` — HAL 抽象：`hal_numpy` 假后端 / `hal_vendor` 真硬件
- `genesim_bridge/` — GeneSim 成本桥接（性能评估旁支，任务 4）
- `tests/` — 逐节点对拍器 + 各模块单测

## 关键约定

- `contracts/` 是全仓地基。改任何 schema 字段会波及所有下游模块——改动前确认调用方，改后同步全链路。各组只通过 `node.meta` 标注解耦。
- 正确性判据是"与单卡 PyTorch 逐元素对齐"。改动后用 `tests/` 的逐节点对拍器验证；换真硬件时"换后端不换数值"。
- 第 1 阶段固定 shape，模型为 GPT-2（`openai-community/gpt2`），窄算子白名单只放 A 类（GEMM/GEMV/逐元素），其余留 host 兜底。不要为第 2/3 阶段特性（序列变长、代价模型、算子融合）提前加复杂度。
- 改某模块前先读它依赖的 `contracts/` 定义和上游产物。

## 提交规范

- 所有提交以仓库所有者名义：`fengjg <fengjg@ios.ac.cn>`（已配置为本仓 local git identity）。
- 不添加任何 AI / agent 署名或 co-author 尾注。
- 仅在用户明确要求时提交；用 `git add <具体文件>` 精确暂存。
- 提交信息用简洁祈使句，说明改了什么、为什么。

## 验证

- 优先运行对应模块单测和逐节点对拍器，而非仅静态阅读代码。
- 提交前确认改动模块可 import、相关测试通过。
