# 需求文档：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理

> 文档编号：request-ceva-llama2-7b-20261028
> 创建日期：2026-10-28
> 关联项目：flagos-pim-compiler、FlagTree、genesim、CEVA 工作区
> 状态：目标、验收口径及交付边界已由用户确认。2026-11-01 更新：bin 改为全部由我方生成，token 对拍放宽为告警档（见 P0-1、P0-5）。

## 一、需求背景与目标

现有图编译器能生成 GML 及 bin，CEVA 工作区已用厂商产物跑通 Llama2 7B。目标是让**我方编译器生成的 GML 与配套 bin** 经厂商工具，在 CEVA 完成 Llama2 7B 全流程推理。

交付边界（11-01 用户确认）：**浮点权重取自厂商（反量化值，已含旋转与 norm 折叠），GML 与全部 bin 由我方图编译、算子编译生成**；生成策略要通用。底线是 CEVA 正常跑完；token 与厂商基线不要求逐个一致。厂商 GC、runner、输入工具、SDK、固件和仿真器原样复用。

### 1.1 层级与分工

CEVA 仿真的是 1 个 NPM 核，核内有 4 个 engine，每个 engine 配 1 个 VPU。我方 1 个 DPU 在层级上大致对应 1 个 NPM 核，4 个 engine 相当于核内并行单元。本期只用 1 个仿真器，单核运行，tp=1。

| 环节 | GeneSim 链路（保留） | CEVA 链路（本期） |
| --- | --- | --- |
| 图编译 | TP/PP 切到多 DPU | 每层每阶段一张单设备 GML，共 65 张 |
| 算子编译 | FlagTree 生成 tasklet 内核 | FlagTree 只决定 GML 的相位和硬件字段 |
| 层内编排 | FlagTree + 编排器 | 厂商 GC：依据 GML 的边排序，分到 4 个 engine，规划 L1/L2，生成 Registry |
| 层间编排 | `runtime/executor.py` | 厂商 `llama_runner`：逐层执行、交换 hidden 地址、绑定 K/V、decode 循环、argmax |
| 硬件 | GeneSim 8 设备 × 8 簇 × 16 TensorPU | CEVA 1 核 4 engine，L2 8MB、L1 1MB、DDR 3983MB |

`runtime/executor.py` 不参与 CEVA 链路，继续服务 NumpyBackend 对齐和 GeneSim 评估。多核并行需要走厂商 `vbu_npm_system`，不属于本期范围；甲方也已说明并行仿真结果不准。

### 1.2 路径约定

| 前缀 | 根目录 |
| --- | --- |
| G（默认） | `/data/fjg/pengcheng/flagOS/flagos-pim-compiler` |
| F | `/data/fjg/pengcheng/flagOS/flagOS-installers/FlagTree` |
| S | `/data/fjg/pengcheng/flagOS/genesim` |
| C | `/data/fjg/ceva` |
| V | `/home/jgfeng/ceva/392111_E2E_Llama2_package/full_release_package/gc_release_package/vbu_ctp_apps` |
| R | `/home/jgfeng/ceva/gmls-device-1024/392111_E2E_Llama2_GMLs/GMLs`（厂商 65 图） |

## 二、核心功能需求

### 2.1 功能清单

**P0-1　固定基线，bin 全部由我方生成**

- 现状：厂商权重已追溯，结论如下（只读比对，证据见六、1）：
  - 底座是 HF 基础版 `Llama-2-7b-hf`。
  - 在底座上做了 4096 维 Hadamard 旋转，RMSNorm 的 γ 折进下游投影，v/o 另有逐头正交变换，最后用厂商量化器量化。
  - 我方 `gml_bridge/export.py:807` 从原始 HF 权重重新量化，与厂商数值不可比。
  - 10-31 前的实现把厂商 65 图里的权重、scale、LUT 与常量整字节复制进我方包，GML 是我方生成的，bin 却是厂商的。
- 需求（2026-11-01 更新，用户确认）：
  - GML 及其引用的全部 bin 由我方图编译、算子编译生成，不复制厂商任何 bin。
  - 旋转与折叠我方复现不了，所以浮点权重取厂商量化权重的反量化值，作为模型 checkpoint 装入；量化、scale、相位、LUT、定标与常量表全部由我方编译器重新生成。
  - KV cache 与输出头的静态 scale 由我方标定流程给出，标定输入走 runner 同源的已旋转 embedding。
  - tokenizer、embedding、cos/sin 和输入工具沿用厂商流程；冻结 prompt、贪心解码和硬件配置。
- 验收：
  - 清单 `bundle.json` 里每个文件的来源都是 `generated`，摘要与落盘一致；出现复制的厂商文件即判失败。
  - 每个文件与厂商同名文件的尺寸一致（GC 按尺寸分配缓冲）；内容不要求一致，逐族差异记录在实施文档。
  - 生成策略通用：量化与标定规则不按层号、节点名写死。

**P0-2　覆盖完整模型导出**

- 现状：`scripts/export_gml.py:626` 默认只导 decode 块，`runtime/compile.py:59` 导出时不带缓存。厂商有 32 张 prefill、32 张 decode、1 张 lm_head 子图（`C/runs/20261027_e2e_clean/run/run.log:25` 起加载）。
- 需求：生成全部 65 张子图。目录结构、子图划分和接口与厂商一致；embedding 查表和 argmax 仍由厂商 runner 完成。
- 验收：65 张图全部由我方生成，并经厂商 GC 编成 Registry。不得混入厂商旧 Registry。

**P0-3　符合 GML 与 runner 接口**

- 现状：
  - runner 按名字绑定地址，名字写死在 `C/src/llama_runner/src/llama2_npm.cpp:39–52,86–87,127–136`。
  - decode 的 K/V 原地更新：输入 `nprm_0_i3`/`nprm_0_i1` 经 `KV_Cache_DMA` 到输出 `key_cache_out`/`value_cache_out`，两端共用 runner 给的同一地址。
  - 写入行号由 `executeSubgraph(..., line_idx)` 传入。
  - 厂商 int4 权重按「输入维 × 输出维」存放，我方 `gml_bridge/runtime_files.py:73` 按 HF 的「输出维 × 输入维」写（11-01 已改为与厂商相同的布局）。
- 需求：
  - 节点、边、相位、融合、I/O 名称与顺序、形状、dtype、布局、量化、scale、LUT 均要可被 GC 接受，并能被 runner 正确绑定。
  - 动态缓冲只按尺寸分配，不得用标定样本冒充实际计算。
- 验收：
  - 我方静态校验通过，厂商 GC 逐图成功。
  - runner 用到的全部名字在每张图中存在，且形状一致。

**P0-4　接通厂商执行链**

- 现状：`C/run_all.sh` 没有 GC 编译步骤，`C/env.sh:22` 固定指向厂商目录。`scripts/export_gml.py:706` 的 `--orchestrate` 支线自行生成 `prepare_out`，未调用厂商 L2A。
- 需求：
  - 删除我方 `prepare_out` 生成支线。
  - 工作区新增「我方 GML → 厂商 GC → DDR 布局 → 仿真」的接线。
  - 每张图编完只保留 Registry 和日志。
  - 支持单层编译与单层独立运行，用于调试。
- 验收：
  - 编译器不再生成 `prepare_out`。
  - GC 失败或产物缺失时返回非零。
  - 原始资料目录前后无变化。

**P0-5　完整运行与 token 对拍**

- 现状：
  - 厂商基线跑 1 步 decode，输出 `[13, 29902]`，用时约 4.5 小时（`C/runs/20261027_e2e_clean/results/verdict.txt:14`）。
  - `C/scripts/verify_run.py:137,139` 存在 SKIP，且只比较两边较短的前缀。
- 需求：
  - 先跑 1 步调试，最终至少 10 步，遇到 EOS 可提前结束。
  - 若实测太慢，可在记录原因后缩减步数，但至少完成全 32 层的 1 步 decode。
- 验收：
  - prefill、decode、KV 更新、lm_head、argmax 全部执行，仿真正常结束，核心检查不得 SKIP。
  - token 对拍分两档（11-01 用户确认放宽）：bin 由我方生成后，量化口径与厂商不同，token 允许分叉，此时用 `--token-check warn`，不一致只告警；底线是 CEVA 正常跑完、输出有限且合理。厂商 bin 的历史运行仍按严格档逐 token 一致。
  - 缩减后的运行不得报告为「10 步通过」。

### 2.2 核心流程

厂商量化权重反量化成浮点 checkpoint → 整模型标定（KV 与静态入口 scale）→ 我方图编译和 FlagTree 相位 → 我方量化与写盘生成 GML、IO_info、全部 bin 并自检 → 厂商 GC 生成 Registry → DDR 布局 → CEVA 仿真 → token 对拍（严格或告警档）。

调试顺序如下：

1. 单层 decode：编译约 25 秒，用独立运行流程与厂商同层输出比对。
2. lm_head。
3. 单层 prefill：编译约 40 分钟。
4. 全部 65 图。
5. 全模型 1 步。
6. 全模型 10 步。

单层独立运行参照厂商 `gc_test/llama2_test`。

### 2.3 边界定义

- 包含：生成端修复、工作区接线、既有回归。首个验收用例为固定厂商 prompt。
- 不包含：自研量化以复现厂商数值、多核并行、其他模型与 prompt。
- GeneSim 只在共享接口受影响时同步；GeneSim 通过不能替代 CEVA 验收。

## 三、技术约束与依赖

### 3.1 影响范围

| 文件／函数 | 内容 |
| --- | --- |
| `scripts/export_gml.py`、`orchestrator/`、`scripts/diff_prepare_out.py` | CEVA 导出入口；删除旧 prepare_out 支线及其专用测试 |
| `gml_bridge/export.py`、`runtime_files.py`、`calib_data.py`、`phase_data.py`、`quant/` | 我方量化与写盘、权重布局、标定、prefill 相位数据 |
| `gml_bridge/from_fx.py`、`writer.py` | 三类子图几何、I/O 契约、head 模式 |
| `graph/kv_dma_pass.py`、`graph/fuse_rope.py`、`graph/quant_pass.py` | 逐层 KV、厂商 sin 编码、attention 分组 |
| `contracts/gml_quant.py`、`contracts/compile_slots.py` | 分组与尺寸；改动后同步调用方 |
| `opcompiler_bridge/phase_plan.py`、`phase_source.py`、`oplevel_emitter.py` | 相位读写关系、head 的 FPSU 模式 |
| `F/lib/Dialect/TritonPIM/IR/Dialect.cpp`、`Ops.cpp`、`Transforms/ExpandPhases.cpp` | int8 模型权重、显式 FPSU 模式 |
| `C/env.sh`、`run_all.sh`、`scripts/verify_run.py`、新增编译与独立运行脚本 | 工作区接线和严格验收 |

### 3.2 非功能约束

- `/home/jgfeng/ceva` 及其软链接目标只读。GC 一律在工作区副本上运行，运行前后用 `C/scripts/check_orig.sh` 核对。
- 资源（厂商日志数据，我方尚未实测）：

  | 子图 | GC 耗时 | 中间产物 |
  | --- | --- | --- |
  | 每层 prefill | 约 39 分钟 | 约 18GB |
  | 每层 decode | 约 25 秒 | 约 1GB |
  | lm_head | 约 6 秒 | — |

  65 图串行合计约 21 小时。CEVA 仿真 1 步约 4.5 小时，10 步按 cycles 推算约 5 小时。一次完整验收约 1 天。
- 磁盘：`/data` 剩 554G。估算如下，并行编译路数受磁盘限制。

  | 项 | 大小 |
  | --- | --- |
  | 我方 65 图包 | 约 70G |
  | 65 个 Registry | 约 9G |
  | 每次运行的 DDR 布局 | 约 27G |
  | 单路 GC 临时产物 | 约 18G |

  只要每张图编完就清理，空间够用。
- FlagTree 修改后用 `../flagOS-installers/0-install-flagtree.sh` 重建并同步。

## 四、验收标准

逐项满足 P0-1～P0-5。归档以下内容：生成命令、来源清单、GC 日志、DDR map、CEVA 日志、token 与文本、判定报告、耗时。

图编译器修改按以下命令回归：

```bash
cd /data/fjg/pengcheng/flagOS/flagos-pim-compiler
source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -q -k "not llama2_7b"
# 仅最终验证，预计 2～3 小时
python -m pytest tests/ -q -k "llama2_7b"
```

删除旧支线时，其专用测试一并删除，保留的功能不得新增失败。CEVA 必须正常结束且无推理错误。`EndOfExecution` 之后的关闭会话超时单独记录，不能用来掩盖之前的错误。

## 五、调研补充信息

- `V/docs/graph_compiler/GC_offline_guide.md:145` 支持从预生成的 GML/bin 生成 Registry，用法参照厂商脚本 `/home/jgfeng/ceva/gc_test/llama2_test/02_gc.sh`（`-hw` 传完整路径）：

  ```bash
  python3 -m nprm_backend.GC_offline_json -d <图目录> -hw <硬件目录完整路径> -l 3 -u 16
  ```

- `V/python/nprm_backend/processors.py:420` 读取 `parser_output/tvmgen_default_nprm_main_0/runtime_files/relay2gml_graph.gml`。L2A 生成 net.ini，test_app 生成 Registry，不需要我方提供 `prepare_out`。
- 厂商单层独立运行：`gc_test/llama2_test/03、04` 脚本加 `04_dsp_standalone_run/dsp_standalone.ini`，输出 hidden 和 K/V。注意 03 会在原始目录里构建，工作区需要复制源码后再构建。

原始prompt信息:
背景：
1.已初步完成面向存算一体编译器的AI编译器研发，版本在：
/data/fjg/pengcheng/flagOS/flagos-pim-compiler  图编译器
/data/fjg/pengcheng/flagOS/genesim  模拟器
/data/fjg/pengcheng/flagOS/flagOS-installers/FlagTree  算子编译器
/data/fjg/pengcheng/flagOS/flagOS-installers/ 0-install-flagtree.sh，安装脚本。
当前版本只是能够在genesim模拟器中运行，同时也能生成gml格式文件。
2.在甲方的需求是能够让上述AI编译器生成gml格式文件，走通他们的硬件模拟器ceva。ceva相关资料：/data/fjg/ceva。相关脚本在scripts以及run_all.sh。注意尽量不要动：/home/jgfeng/ceva中的原始资料，这个是需要比对的。目前初步运行/data/fjg/ceva，能够用厂商原来提供的gml原始文件走通llama2 7b在ceva硬件仿真器的模拟。

目标：
能够让存算一体编译器生成的gml以及配套文件，走通甲方的ceva硬件仿真器，走通llama2 7b的推理全流程。

思路：
1.可能当前的存算一体编译器生成的gml不符合当前gml规范，无法走通ceva硬件仿真器流程，为了符合规范可能需要调整flagos-pim-compiler，FlagTree，genesim。
2.厂商提供ceva相关资料：/home/jgfeng/ceva，这些资料多看看，可能有用。尤其是gml生成相关的；
3.根据厂商说我们只需要生成gml以及相关的bin就可以了，其他的应该用厂商提供的；
4.如果出现模拟器相关问题就看看/home/jgfeng/ceva相关文件，这里面也有正常执行的日志文件以及相关bin；
5.还可以比对厂商提供的gml相关配文件和bin与我们有什么不同，如果出现某些问题可以调整；
6.可以根据仿真器推理运行各种log信息，以及比对原来正确的进行调试，修改。
7.估计可能需要修改的地方：flagos-pim-compiler，FlagTree ，影响gml生成的，以及/data/fjg/ceva。

验证：
1.核心验证：能够让存算一体编译器生成的gml以及配套文件，走通甲方的ceva硬件仿真器，走通llama2 7b的推理全流程；
2.存算一体编译器验证：
cd /data/fjg/pengcheng/flagOS/flagos-pim-compiler
source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -q -k "not llama2_7b" , 图编译器修改，主要用这个验证；
python -m pytest tests/ -q -k "llama2_7b" ，这个运行时间要2-3个小时，只是最后验证，平常不验证。
