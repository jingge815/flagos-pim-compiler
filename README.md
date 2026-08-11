# flagos-pim-compiler

存算一体（PIM）大模型推理编译器 —— FlagOS 生态下的图层编译器与主机编排运行时。

把 HuggingFace 模型经 `torch.export` 拿到静态图后，在编译期完成图拆分、切分传播、通信计划与三区内存规划，产出一组静态蓝图；运行时由主机编排器在 NumpyBackend（多 DPU 独立地址空间模拟）上驱动执行，并与单卡 PyTorch 逐节点数值对齐。

本仓覆盖方案中的问题 1/2/3/6/7/8 + GeneSim 成本桥接（问题 4）。算子编译器（问题 5，`ttir→pim mlir`）在 [FlagTree](https://github.com/) 中开发，只通过 kernel 二进制与编译期算子契约与本仓对接。

## 目录结构

| 目录 | 内容 | 对应任务 |
| --- | --- | --- |
| `contracts/` | 共享数据契约（`PIMTensorSpec`、`ExecutionPlan`、`node.meta` 字段约定）——全仓 import 的唯一真源 | 全组共用 |
| `graph/` | 图拆分打标、切分传播 / redistribute 标注 | 问题 1、2 |
| `comm/` | 通信库：编译期通信计划表 + 运行时 redistribute→DMA 下降 | 问题 3 |
| `memory/` | KV 布局与 runtime、三区静态内存规划 | 问题 7、8 |
| `runtime/` | 编排器：ExecutionPlan 生成 + 解码循环执行 | 问题 6 |
| `backend/` | HAL 抽象：`hal_numpy` 假后端 / `hal_vendor` 真硬件 | 运行时底座 |
| `genesim_bridge/` | GeneSim 成本桥接（从 IR 抽 flops/data_bytes，性能评估旁支） | 问题 4 |
| `tests/` | 逐节点对拍器 + 各模块单测 | 集成验证 |
| `examples/` | 端到端示例（如 GPT-2 全链路跑通） | — |
| `scripts/` | 开发与构建脚本 | — |
| `docs/` | 设计文档与接口契约 | — |

## 第 1 阶段目标

固定 shape 全链路打通，模型选定 GPT-2（`openai-community/gpt2`），在 NumpyBackend 上跑通 prefill + 完整 decode 自回归解码，与单卡 PyTorch 逐元素对齐。

## 开发约定

`contracts/` 是全仓地基，字段名须在开发初期钉死，之后各组在自己目录内并行推进，仅通过 `node.meta` 标注解耦。
