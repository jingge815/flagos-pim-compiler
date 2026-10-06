# 需求文档：Llama2 7B 推理算子全量走通存算一体编译链路

> 文档编号：request-llama2-7b-e2e-20261003
> 创建日期：2026-10-03
> 关联项目：flagos-pim-compiler（图编译器）、FlagTree（算子编译器）、genesim（仿真器）

## 一、需求背景与目标

1. 业务背景：Llama2 7B 推理测试当前能跑通并与单卡 HF 逐 token 对齐，但「跑通」只证明数值对。设备内核是「编译内核优先、numpy 兜底」，遇到 2 的幂、类型、尺寸一类限制就静默退回主机，而主机路径性能差。现有断言在整段推理退回主机时仍然是绿的。
2. 预期目标：Llama2 7B 推理所需的全部算子都在设备上编译运行，完整走通 numpy、genesim、gml 三条路径，并完成端到端仿真推理。限制一律打破，不许靠退回主机绕过。
3. 关联范围：`graph/partition.py`、`runtime/kernels.py`、`opcompiler_bridge/`、`gml_bridge/`、`genesim_bridge/`，以及 FlagTree、genesim 两个仓库。

## 二、核心功能需求

### 2.1 功能清单

**P0-1　确认推理算子的设备归属**

- 现状：`graph/partition.py:26` 的 `HOST_ONLY` 是黑名单，默认下设备。实测留主机的只有编译脚手架（`_assert_*`、`sym_size` 一类）、位置下标 `arange`、广播视图（`expand`、`repeat`、`contiguous`、`squeeze`）。注意力已拆进设备命令（`runtime/kernels.py:1045`）。
- 需求：核对导出图里每个算子都下设备，黑名单只允许上述项。
- 验收判据：对模型做一次 `torch.export`，图中 `aten` 算子与黑名单求差必须为空（除允许项）。

**P0-2　三条生成路径无错误**

- 现状：三条路径入口都在。numpy 走 `backend/hal_numpy.py`、`runtime/kernels.py`；genesim 走 `genesim_bridge/`；gml 走 `gml_bridge/export.py`、`scripts/export_gml.py`。真正调用 `compile_op` 的算子被 `tests/test_runtime_compiled_coverage.py:166` 钉死为 15 个。
- 需求：这 15 个算子各自走通 numpy、genesim、gml 三份产物，无编译与运行时错误。
- 验收判据：三份产物都生成，编译内核与 numpy 镜像逐元素相对误差 `<0.05`。

**P0-3　打破限制，推理全程走编译内核**

- 现状：`runtime/kernels.py:167` 要求 `k≥16` 且 `m`、`k`、`n` 都是 2 的幂，否则退回纯 numpy。实测 8 DPU 下 `q_proj`、`o_proj` 通过，而 `gate_proj`、`up_proj` 的 N=1376、`down_proj` 的 K=1376、`lm_head` 的 N=32000 全被拒。类型、尺寸也各有门槛：eltwise 要求同形 fp16，matmul 要求二维 fp16，convert 只收 fp16、fp32、int8。这些退回都落在主机上，性能差，且 `tests/test_opcompiler_e2e_llama2_7b.py:142` 只断言 token 对齐，退回了也判通过。
- 需求：2 的幂、类型、尺寸三类限制全部打开。因此超出 DPU 的 WRAM 容量时，改循环分块算法把计算拆进容量内，而不是退回主机。MRAM 容量不在本轮：硬件配置是每台 DPU 8GB，实测 llama2 各策略的峰值驻留约 206MB（利用率 38.5%，溢出为 0），没有超 MRAM 的形状，分批驻留没有判据可覆盖。
- 验收判据：Llama2 7B 推理中每个算子的兜底次数为 0，token 与文本仍与 HF `generate` 一致。任意形状、任意支持类型都走编译内核。唯一例外是 K 小于 16：Triton 的 `tl.dot` 要求 K 不小于 16，低于它编译器直接报错，这类形状退回镜像并计入兜底计数，判据看得见。

**P1　端到端仿真推理**

- 现状：`scripts/run_full_pipeline.py` 有 genesim 仿真步骤，但 llama2_7b 测试组验证的是 numpy 后端与 HF 的对齐，不是仿真结果。
- 需求：Llama2 7B 在 genesim 上完成一次端到端仿真推理并产出结果。仿真报错就改 genesim。
- 验收判据：仿真流程跑完无错误并给出输出 token。与 HF 的逐 token 对齐不作为本项判据。

### 2.2 核心流程

加载 `Llama-2-7b-hf` → `torch.export` → `partition_graph` 打设备标记 → 图编译产出 PIM MLIR → FlagTree 降到 C 并生成 numpy / genesim / gml 三份产物 → 运行时一律走编译内核，超容量由分块循环解决 → 解码并与 HF `generate` 逐 token 对比 → genesim 端到端仿真。

### 2.3 边界定义

包含：导出图算子归属核对、15 个算子的三路径产物与数值对齐、去掉 2 的幂与类型尺寸限制、超容量时的循环分块、端到端仿真。

不包含：把 `expand`、`repeat`、`contiguous`、`squeeze` 下设备，广播倍数在命令里没有存放位置；`layer_norm`（目标模型用 RMSNorm，它只在合成测试里作脚手架）；仿真侧把全部算子钉到 PIM（P1 判据只到仿真跑完并给出输出 token，模型出入口与视图算子留在 GPU 上不改变这条判据）；convert 的 bfloat16（llama2 推理不产生 bf16，理由是设计 4.5）；超 MRAM 的分批驻留（硬件每台 DPU 8GB，实测峰值驻留约 206MB，没有超容量的形状）。除此之外不设边界，算子编译的问题改 FlagTree，仿真的问题改 genesim，容量问题改分块算法。

## 三、技术约束与依赖

1. 代码改动影响范围

| 文件 | 本轮预期动作 |
| --- | --- |
| `graph/partition.py` | 只核对不改，除非导出图暴露黑名单漏项 |
| `runtime/kernels.py` | 去掉按形状提前退回的判断，退回点加命中计数 |
| `opcompiler_bridge/kernel_src.py` | M 维改为分块循环，不再要求整维是 2 的幂 |
| FlagTree `TileToBudget.cpp` | 分块搜索支持任意尺寸，超 WRAM 继续拆 |
| genesim | 仿真跑不通，或 sidecar 声明的放置未被遵守导致指标失真时改 |
| `tests/test_opcompiler_e2e_llama2_7b.py` | 补「兜底次数为 0」断言 |

2. 潜在风险：兜底是静默的，不加命中统计就无法验收 P0-3。llama2_7b 全量超过 40 分钟，只能在快速判据就绪后跑。放宽后单次计算可能超 DPU 容量，分块算法必须先于放宽落地。

## 五、验收标准

1. 回归判据：层级 1 为逐算子三路径生成，15 个算子全部生成且数值对齐。层级 2 为 `python -m pytest tests/ -q -k "not llama2_7b"`，要求无新增失败。层级 3 为 `python -m pytest tests/ -q -k "llama2_7b"`，耗时 40 分钟以上，仅前两级通过后执行，要求 token 与 HF 一致且兜底次数为 0。用例数以当次实测为准。
2. 逐项验收条件：见 2.1 各功能点的验收判据。
3. 不回归的判据：黑名单范围不扩大（不能靠把算子改判为主机来通过）；15 个算子的编译集合不缩小；HF 逐 token 对齐保持。

## 七、本轮实测对既有结论的修正

1. 「232 用例、16 秒」不成立：`def test_` 实测 1128 条。文档不写死数字。
2. tag 实名为 `pim-compiler-v0.0.5`、`pim-compiler-v0.0.6`，只在 FlagTree 与 genesim。
3. 「部分推理不走 PIM」在分区层不成立，在执行层成立：MLP 三个投影与 `lm_head` 因维度非 2 的幂退回 numpy，测试不检查。

## 八、待确认事项

1. P0-1 的 `torch.export` 全图核对尚未跑，黑名单有无漏项以那次结果为准。
2. 兜底计数加在 `runtime/kernels.py` 还是测试侧，留到设计阶段定。
