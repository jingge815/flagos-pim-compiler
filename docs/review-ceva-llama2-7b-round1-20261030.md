# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-20261030
> 创建日期：2026-10-30
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述
评审覆盖三处未提交改动：flagos-pim-compiler（30 个文件，含新增 `contracts/ceva.py`、`gml_bridge/ceva.py`、`tests/test_ceva_export.py`）、FlagTree（8 个文件）、CEVA 工作区（`run_all.sh`、`env.sh`、`verify_run.py` 及新增脚本与测试）。主体实现与设计一致，65 图全流程已经 PASS。快速回归只有 1 个失败，改动前就已存在。共发现 7 个问题：高 2、中 3、低 2。其中两项已用最小复现证实：编译复用会放过「GML 未变、bin 已变」的旧 Registry；相位依赖校验在 reads 全部丢失时也会放行。

## 二、修改建议
### 问题1（高）
**问题说明**：设计 4.12 要求每张图都有本次新生成的 Registry，修改时间晚于本次开始。`compile_gml.py` 新增了复用分支 `already_compiled`，判断依据只有 GML 文本和 Registry 两个摘要，不看同目录的 bin。导入的权重、scale、LUT 字节都会编进 Registry。所以重新导出一份「GML 文本相同、bin 不同」的包时，会直接复用旧 Registry，同时判定为 PASS，违反 P0-2「不得混入旧 Registry」。复现结果：只改 `weight_buffer_3.bin`，`already_compiled` 仍返回 True。这个分支也不在设计范围内。
**涉及文件**：`/data/fjg/ceva/scripts/compile_gml.py`，`already_compiled`（第 65 行）、`main` 中 `reused`（第 163 行）
**修改思路**：两种方式，二选一。删除复用分支，每次都重新编译，完全回到设计 4.12；或者把摘要范围扩大到整个 `runtime_files` 目录（全部文件名与内容），并补测试「bin 变化时不复用」。

### 问题2（高）
**问题说明**：设计 4.10 要求校验 DQ 与 Softmax 的跨相依赖。`cross_check` 只在 `any(p.reads for p in plan.phases)` 为真时才比较，所以 FlagTree 不再输出 reads、或者解析把 reads 全部丢掉时，依赖校验整段跳过，不报任何问题。复现结果：一个 DQ 计划四相 reads 全空，`.reads` 类不一致为 0。这与 CLAUDE.md「不写防御性兜底，契约不满足就直接抛」相悖。
**涉及文件**：`opcompiler_bridge/phase_source.py`，`cross_check` 中的 `expected_reads` 判断
**修改思路**：去掉 `any(p.reads ...)` 这个前置条件，DQ 与 Softmax 的计划一律逐相比较 reads。补一个用例：reads 全空时报出不一致。先确认现有 FlagTree 产物都带 reads，确保回归不受影响。

### 问题3（中）
**问题说明**：设计 4.3、5.2、5.3 规定 `phase_data.softmax(source, *, row_width=None)` 支持逐行归约，`SoftmaxPhases` 允许逐行数组，7.1 还列了 `test_softmax_rows`。实际 `gml_bridge/phase_data.py` 没有改动，`tests/test_phase_data.py` 也没有这个用例。CEVA 链路的运行时缓冲按尺寸写零，确实不经过这条路径，但实施文档 1.1 和 C1 写的是「prefill 几何与归约」已完成，5.2 偏差说明里也没有记录这一项。同类未记录的偏差还有：`fuse_for_gml`/`serialize_gml` 没有 `graph_spec` 参数，`GmlArtifact` 没有 `graph_spec` 字段，`export_ceva_subgraph` 返回 `(gm, report)` 而不是 `GraphModule`，`test_gml_depends_on_opcompiler.py` 没有扩展。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 1.1、2、5.2 节；`gml_bridge/phase_data.py`
**修改思路**：CEVA 链路不需要逐行 Softmax 数据，不应为了对齐设计而补实现。在实施文档 5.2 里如实记录这几项接口偏差和原因（运行时缓冲写零、契约改由 `CevaGraphSpec` 在 `gml_bridge/ceva.py` 内部传递），并修正 C1 的描述。

### 问题4（中）
**问题说明**：设计 5.2 规定 `bundle.json` 的 `complete` 表示 65 图齐全。`export_ceva_bundle` 用 `model.config.num_hidden_layers` 推算期望集合，而 `--layers 1` 会把它截成 1，于是只有 3 张图的调试包也被标为 `complete: true`（现存 `bundles/ours_l0/bundle.json` 就是这样）。`run_all.sh --gml-bundle` 只检查 `bundle.json` 是否存在，不看 `complete`，不完整的包要到 DDR 布局或仿真阶段才会暴露。
**涉及文件**：`gml_bridge/ceva.py`，`export_ceva_bundle`（第 927–929 行）；`/data/fjg/ceva/run_all.sh` 中 `--gml-bundle` 的前置检查
**修改思路**：期望集合按厂商参考目录里实际存在的子图来算（或者固定为 32 层），不依赖截层后的 config。`run_all.sh` 在 `--gml-bundle` 模式下要求 `complete` 为 true，否则非零退出。

### 问题5（中）
**问题说明**：设计 4.11 要求单层独立运行输出 hidden、K、V，P0-4 要求失败时返回非零。`standalone_layer.sh` 用 `|| true` 吞掉了运行退出码，事后只检查 `tvmgen_default_nprm_main_0_output_0` 一个输出。decode/prefill 缺 K/V 输出时仍会报「独立运行完成」。另外，`verify_gml_artifact.py --ceva-manifest` 在厂商来源文件不存在时会跳过来源比对（`source.is_file() and ...`），同样属于静默放行。
**涉及文件**：`/data/fjg/ceva/scripts/standalone_layer.sh`（第 23–27 行）；`scripts/verify_gml_artifact.py`，`check_ceva_manifest`
**修改思路**：从 ini 里按 `save=true` 的缓冲名逐个检查 `<名>_core0_output.bin`；manifest 校验中来源缺失直接判失败，并补对应用例。

### 问题6（低）
**问题说明**：CLAUDE.md 要求「删优于加」、不写防御性兜底，按现有代码风格组织 import。`gml_bridge/ceva.py` 有以下问题：模块中段 4 处 `import ... # noqa: E402`；函数内 `import dataclasses`；`fill_weight_hashes` 导入了但没用（`noqa: F401`）；`except OSError` 吞掉 git 调用异常。`contracts/ceva.py` 的 `STAGES` 没有调用方。`graph/kv_dma_pass.py::_transpose_key_once` 里 `anchor = copy if source is None else copy` 两个分支相同，是死逻辑。
**涉及文件**：`gml_bridge/ceva.py`；`contracts/ceva.py`；`graph/kv_dma_pass.py`
**修改思路**：import 统一放到文件头；删掉未使用的导入、`STAGES` 和无效条件；git 版本号取不到时直接报错，或者明确写成空串，但不要 try/except。

### 问题7（低）
**问题说明**：实施文档中的数字前后不一致。4.1 写「CEVA 工作区 11」，4.2 写 D1～D3 共 13 例（实际 `tests/` 下 6+5+2=13）；3.1 写 CEVA 工作区新增 9 个文件，实际只列了 8 项。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 3.1、4.1 节
**修改思路**：按实际情况统一为 13 例；新增文件逐一列出，并核对数量。
