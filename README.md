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
| `genesim_bridge/` | GeneSim 成本桥接（从 TTIR / pim mlir 抽 flops/data_bytes，性能评估旁支） | 问题 4 |
| `tests/` | 逐节点对拍器 + 各模块单测 | 集成验证 |
| `examples/` | 端到端示例（默认 LLaMA2 全链路，GPT-2 仅作 legacy/debug） | — |
| `scripts/` | 开发与构建脚本 | — |
| `docs/` | 设计文档与接口契约 | — |

## 第 1 阶段目标

固定 shape 全链路打通，默认模型选定 `meta-llama/Llama-2-7b-hf`，例如，设置本地路径为 `/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/meta-llama-Llama-2-7b-hf`。编译器日常单测使用缩小的随机 LLaMA 配置避免下载权重，端到端验证加载官方 HuggingFace LLaMA2-7B 权重，在 NumpyBackend 上跑通 prefill + decode 路径并与单卡 PyTorch 逐元素对齐。GPT-2 只保留为显式 legacy/debug smoke。

## 环境

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -x -q
```

torch 2.9.1 / transformers 4.57.6 / python 3.10.20。每个新 shell 都要先 source。

### 站点相关路径

`genesim_bridge` 需要知道 flagTree 安装与 GeneSim 仓库的位置。统一到单一
`flagTree` 安装（2026-08-29 起）：这份安装重新编译后，已经把带 PIM pass 的
`libtriton.so`/`pim_sidecar.py`/nvidia backend 同步进了 pytorch 环境
（`0-install-flagtree.sh::sync_triton_to_pytorch`），任何时候 `import triton`
都自带 PIM 支持，不再需要维护第二份独立的 `flagTree-pim` 安装。默认值是当前
开发机路径，换机器时**不要改代码**，用环境变量或配置文件覆盖：

```bash
# 方式 1：环境变量
export FLAGTREE_PREFIX=/path/to/flagOS-installed/flagTree
export GENESIM_ROOT=/path/to/genesim

# 方式 2：仓库根建 paths.local.json（已被 .gitignore 忽略）
echo '{"flagtree_prefix": "...", "genesim_root": "..."}' > paths.local.json
```

PIM pass 的硬件参数（`FLAGTREE_PIM_TARGET` / `_NUM_DPUS` / `_NUM_TASKLETS` /
`_WRAM_BYTES`）走同一套优先级，键名去掉 `FLAGTREE_` 前缀改小写即为配置文件键。

优先级：环境变量 > `paths.local.json` > `genesim_bridge/paths.py` 里的默认值。
查看当前生效路径与参数：

```bash
python -c "from genesim_bridge.paths import describe; print(describe())"
```

GeneSim 侧的 `scripts/refine_ir_with_flagtree.py` 另有一个 `PIM_COMPILER_ROOT`
环境变量，指向本仓库根目录。

## 文档

- `docs/spec.md` — 完整技术方案
- `docs/spec-index.md` — 方案分节行号索引（按需定位，避免整篇读）
- `docs/<module>.md` — 各模块接口与设计决策

## 开发约定

`contracts/` 是全仓地基，字段名须在开发初期钉死，之后各组在自己目录内并行推进，仅通过 `node.meta` 标注解耦。

开发规范（代码量控制、可读性、测试、文档要求）见 [`CLAUDE.md`](./CLAUDE.md)，人和 agent 共用同一份。
