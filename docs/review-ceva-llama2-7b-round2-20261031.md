# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理

> 文档编号：review-ceva-llama2-7b-20261031
> 创建日期：2026-10-31
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029

## 一、评审概述

本次评审覆盖 flagos-pim-compiler（30 个未提交文件，含新增 `contracts/ceva.py`、`gml_bridge/ceva.py`、`tests/test_ceva_export.py`）、FlagTree（8 个文件）与 CEVA 工作区（`run_all.sh`、`env.sh`、`verify_run.py` 及新增脚本和测试）；genesim 工作区没有未提交变更，未纳入代码审查。核心链路已有 `/data/fjg/ceva/runs/ours_full` 的 PASS 记录：11 个 token 与厂商基线逐 token 一致，Registry 全部来自本次编译，本次未重跑约 10 小时的 65 图全流程。实际执行结果：快速回归 1584 passed、8 skipped、1 个改动前已存在的失败（`test_pimir_sidecar_agrees_with_ttir_on_flops`），`test_ceva_export.py` 39 passed，CEVA `tests/` 17 passed，FlagTree lit 265 passed。共发现 8 个问题：高 2、中 4、低 2，集中在验收严格性、失败路径和文档口径；未发现会改变已通过 token 结果的代码缺陷。

## 二、修改建议

### 问题1（高）
**问题说明**：需求 P0-4 的验收口径是「原始资料目录前后无变化」，§3.2 还要求 `/home/jgfeng/ceva` 只读、运行前后用 `check_orig.sh` 核对。实施文档 5.1 明确记录 10-28 的 GC 探针覆盖了 7 个厂商运行日志与中间图（`TAGS_Log.txt`、`nprm_backend.log`、`g1b/g2/g3*.gml`、`L2_buffering_output/relay2gml_graph.html`、`sat_sbt_buffer_graph.gml`），且不在厂商 zip 与 `full_release_package.tar.gz` 中，无备份可还原，`check_orig.sh` 仍会报这 7 处；但实施文档 1.3 仍写「全部完成」，4.3 写判定 PASS。当前流程已改用 `env/gc_w<N>` 副本，`ours_full` 的第 7 步也通过，但这只能说明本次运行未再改动，不能改变原始目录相对交付起点已被改写的事实，P0-4 的「前后无变化」并未满足。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 18、105–109 行；`docs/request-ceva-llama2-7b-20261028.md` 第 86–89、142 行。
**修改思路**：把 P0-4 明确标为「未完全满足/有例外」，不要再用「全部完成」概括；若 7 个文件能从厂商其他介质恢复则恢复，若不能，取得甲方对「运行产物例外」的书面确认并归档，同时让 `check_orig.sh` 的已知差异在验收记录中逐项列明，而不是被一次本流程的干净快照掩盖。

### 问题2（高）
**问题说明**：设计 4.13 要求新模式下 prompt 不同直接 FAIL、核心检查不得 SKIP、DDR map 中的 Registry 必须来自本次运行；需求 P0-4/P0-5 也要求不得混入厂商旧 Registry。`verify_run.py` 判断「是否我方 GML 模式」的唯一依据是 `RUN_DIR/gml_bundle.txt` 是否存在，而该文件只在 `run_all.sh` 第 4 步 `step_ddr` 中写入。使用 `--gml-bundle` 同时指定 `--from 5` 或 `--from 6` 时，第 4 步被 `run_step` 跳过，标记文件不会生成；如果同名 `RUN_DIR` 里已有旧运行留下的 `ddr_output`（例如先用厂商 GML 跑过同名实验），`07_run.sh` 只依赖 `ddr_output`，仍可完成仿真。此时 `verify_run.py` 的 `bundle_mode` 为假，prompt 不一致或输入缺失会落到 SKIP，Registry 来源检查整段不执行，最终仍可能得到 PASS。这直接绕过了 P0-5 的「核心检查不得 SKIP」和「Registry 不得混入厂商旧产物」。
**涉及文件**：`/data/fjg/ceva/run_all.sh` 第 177–189、248–255 行；`/data/fjg/ceva/scripts/verify_run.py` 第 134–157 行。
**修改思路**：让 `--gml-bundle` 本身成为验收模式的权威来源，而不是依赖第 4 步的副作用文件。例如 `run_all.sh` 在 `GML_BUNDLE` 非空时无条件要求 `RUN_DIR/gml_bundle.txt` 与 `RUN_DIR/gmls` 存在，或在 `FROM > 4` 时直接拒绝并提示重新生成 DDR；`verify_run.py` 增加显式模式参数/环境变量，标记缺失时判 FAIL 而不是退回 SKIP。补一个 `--gml-bundle --from 6` 且无标记文件的负例。

### 问题3（中）
**问题说明**：设计 4.11 要求单层独立运行输出 hidden、K、V 并与厂商同层逐字节比较，P0-4 要求失败时返回非零。`standalone_layer.sh` 第 22–23 行仍在 `run_tools.py` 后接 `|| true`，吞掉了运行程序退出码；第 24 行只调用 `standalone_ini.py --check`，而 `missing_outputs`（第 98–108 行）只判断 `<名>_core0_output.bin` 是否存在，不检查文件是否本次生成、是否非空、尺寸是否正确，脚本也没有清空输出目录。重复使用同一个 `OUT` 时，本次运行失败但旧输出仍在，会被报告为「独立运行完成」；同时脚本内没有实现设计要求的厂商同层逐字节比较，只有存在性检查。
**涉及文件**：`/data/fjg/ceva/scripts/standalone_layer.sh` 第 22–24 行；`/data/fjg/ceva/scripts/standalone_ini.py` 第 98–108 行。
**修改思路**：去掉 `|| true`，保留 `run_tools.py` 退出码并作为失败条件；运行前清空/改用新的输出目录，或在 `missing_outputs` 中增加 `mtime > 开始时间` 与文件大小检查；补上厂商同层输出的逐字节比较（报告首个差异位置和误差），并增加「旧输出存在但本次运行失败」的测试。

### 问题4（中）
**问题说明**：设计 5.2 规定 `bundle.json` 的 `complete` 表示参考目录里的 65 张子图齐全，`run_all.sh` 也只看这个字段来决定能否进入 DDR/仿真。`export_ceva_bundle` 读取旧 `bundle.json` 后只在内存里 `manifest.update(..., complete=False)`，磁盘文件要等第一张图导出成功后才重写（`gml_bridge/ceva.py` 第 911–932 行）。如果本次导出在写第一张图之前失败（例如 `load_ceva_spec` 找不到参考图，或第一张 `export_ceva_graph` 报错），磁盘上仍保留上一次的 `complete: true` 和旧图清单；随后 `run_all.sh --gml-bundle` 只检查该字段，可能把不完整或过期的包当成完整包使用。
**涉及文件**：`gml_bridge/ceva.py` 第 911–932 行；`/data/fjg/ceva/run_all.sh` 第 59–64 行。
**修改思路**：读取旧清单后立即把 `complete=false` 原子写回磁盘（临时文件 + rename），或开始导出前删除/重命名旧 `bundle.json`；导出失败时保持 false。补一个「旧包 complete=true，第一张图导出失败」的测试，验证磁盘清单不再为 true。

### 问题5（中）
**问题说明**：设计 3.2 把 `phase_source_from_graph(gm)` 标为「FlagTree 必跑」，4.10 还要求用 `reads` 校验 DQ/Softmax 的跨相依赖。`scripts/export_gml.py` 的 `--use-opcompiler` 默认是 False（第 379 行），而 `_main_ceva` 直接把 `args.use_opcompiler` 传给 `export_ceva_bundle`（第 507 行）；`export_ceva_graph` 只有在该值为真时才执行 `phase_source_from_graph` 与 `cross_check`（`gml_bridge/ceva.py` 第 742–749 行）。因此用户只给 `--ceva-reference`、不加 `--use-opcompiler` 时，仍会导出 `complete` 包并进入 run_all，但相位来自静态表、跨相依赖没有校验，违反「FlagTree 必跑」的设计前提。
**涉及文件**：`scripts/export_gml.py` 第 379、498–507 行；`gml_bridge/ceva.py` 第 742–749 行。
**修改思路**：CEVA 模式下强制 `use_opcompiler=True`，或在 `--ceva-reference` 且未给 `--use-opcompiler` 时 `parser.error`；保留普通非 CEVA 路径的可选行为。补一个 CLI 用例，断言 CEVA 模式默认必须带算子编译器。

### 问题6（中）
**问题说明**：设计 4.9 要求 `Ops.cpp` 的 `MatmulOp::verify` 校验 int8 模型权重与显式 FPSU 模式的配套条件，输出头的口径是 int8 权重 + `fixed_point` NMU + `fixed2float` FPSU。当前 verifier 只做了两条检查：int8 模型权重必须 `nmuMode = fixed_point`，以及 `fixed2float` 必须 `nmuMode = fixed_point`（`lib/Dialect/TritonPIM/IR/Ops.cpp` 第 518–531 行），没有要求 int8 模型权重必须显式带 `fixed2float`。`stampMatmul` 在没有显式 FPSU 时会调用 `setFpsuSpec(op, 2)`，把模式补成 `floating_point_32`（`Transforms/ExpandPhases.cpp` 第 557–564 行）。实测用 `triton-opt -pim-expand-phases` 处理「int8 model weight + fixed_point NMU、无 fpsu」的 IR，输出为 `fpsu = #pim.fpsu_spec<mode = floating_point_32>`，verifier 不报错；这与输出头要求的 fixed2float 数值路径不一致，且现有负例只覆盖了「int8 + floating NMU」和「fixed2float + floating NMU」。
**涉及文件**：`FlagTree/lib/Dialect/TritonPIM/IR/Ops.cpp` 第 518–531 行；`FlagTree/lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp` 第 557–564 行；`FlagTree/test/Dialect/TritonPIM/operator_ops_negative.mlir`。
**修改思路**：在 `MatmulOp::verify` 中要求 int8 模型权重必须带 `fpsu = fixed2float`（或至少拒绝 `floating_point_32`），并让 `stampMatmul`/emitter 对 int8 模型权重只允许 fixed2float；补一条「int8 + fixed NMU + 无 fpsu/错误 fpsu」的负例，确保 `ExpandPhases` 不会静默补成错误模式。

### 问题7（低）
**问题说明**：设计 4.8 要求输出头的 FPSU 模式由算子编译器读回，实施文档 3.3 也写「GML 的该字段由算子编译器读回」；CLAUDE.md 要求不写防御性兜底。`gml_bridge/from_fx.py` 第 2140–2144 行在 `phase_source` 存在但 `plan` 或 `plan.fpsu_mode_name` 缺失时，静默退回硬编码的 `"fixed2float"`。这会把「FlagTree 没有产出 head_matmul 计划」这种契约失败伪装成正常导出；`cross_check` 只检查 `by_node` 中已有的计划，不会发现计划整体缺失。
**涉及文件**：`gml_bridge/from_fx.py` 第 2140–2144 行；`opcompiler_bridge/phase_source.py` 第 246–303 行。
**修改思路**：只允许在 `phase_source is None`（未启用算子编译器）时使用静态兜底；`phase_source` 存在但取不到 `head_matmul` 计划或 FPSU 名称时直接报错。补一个「phase_source 存在但缺 head 计划」的负例，避免回归时静默通过。

### 问题8（低）
**问题说明**：实施文档的状态与数字仍有多处不一致：1.3 写「总体完成状态：全部完成」，但 5.2 同时记录了 `phase_data.softmax(row_width=)`、`test_softmax_rows`、`GmlArtifact.graph_spec`、`fuse_for_gml(graph_spec=)`、`serialize_gml(graph_spec=)`、`export_ceva_subgraph` 返回类型和 `test_gml_depends_on_opcompiler.py` 扩展等未实现项；4.1 写 CEVA 工作区「13（评审修复后为 16）」，7.3 写 17；4.2 写 `test_ceva_export.py` 35 例，实际为 39 例；4.3 写快速回归 1578 passed，7.3 写 1584 passed，本次实际为 1584 passed、8 skipped、1 个改动前已存在的失败。数字和完成状态不统一，会影响验收时可追溯性。
**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` 第 16、78–81、95、144–146 行。
**修改思路**：按本次实际执行结果统一 4.1/4.2/4.3/7.3 的数字；把 1.3 改为「完成（含 5.2 记录的偏差）」，并在 5.2 中逐项标注「不实现/仅文档记录」的理由，避免「全部完成」与偏差说明并存。
