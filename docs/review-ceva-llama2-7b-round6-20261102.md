# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-round6-20261102
> 创建日期：2026-11-02
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029
> 关联上一轮评审：[review-ceva-llama2-7b-round5-20261102](review-ceva-llama2-7b-round5-20261102.md)

## 一、评审概述
本轮承接第五轮，覆盖编译器、FlagTree、CEVA 工作区三个仓库的未提交变更（genesim 无改动），围绕三条核心目标复核。R5-1～R5-4 的代码修复已存在：`test_ceva_export.py`、`test_pwla.py`、`test_quant_calibration.py` 共 82 项通过，CEVA `tests/` 41 项通过。本轮保留 1 项高、3 项中、1 项低问题。

目标 1（bin 全部由我方生成）在代码上已经满足。目标 2（自产包在 CEVA 跑通）的证据是 `own_bins_e2e`，这次运行早于 R5-3 的写零改动。按当前代码重导的第 0 层与 `own_bins` 相比，decode 有 844 个文件不同，prefill 有 839 个，目前只有 decode 和 lm_head 的 GC 有结果，prefill GC 和 65 图仿真都还没跑。目标 3（策略通用）有一处共享编译代码按 HF 参数名决定数值口径，与需求 P0-1 的"不按节点名写死"冲突。本轮没有重跑 GC、仿真和 7B 慢测，快速回归在评审期间仍在运行（`/data/fjg/tmp_ceva/r5/regress.log`）。

## 二、修改建议

### 问题1（高）：V 投影按参数名认定，导致换模型 V cache 失效
**问题说明**：需求 P0-1 要求"量化与标定规则不按层号、节点名写死"，实施 G-2 要求共享 pass 按图结构判定，I8-10 也已把写盘侧的 V 投影识别改为按结构。但 `from_fx.convert` 决定 Gemm 是否输出 int8、是否带 kantor 定标时，用的是 `"v_proj" in weight_param`，结构判据只认 KV_Cache_DMA 的消费者。prefill 没有 DMA，这一条完全退化成名字匹配；decode 的 V 投影和 DMA 之间隔着 Reshape/Split，结构判据也命中不了。本次复现用同一个小 Llama，只把 `v_proj` 改名为 `val_lin`：prefill 和 decode 图里的 V 投影都从 `int8` 加 kantor 字段退回 `float16`，不带 kantor 字段，也不报错。GC 和写盘器（`_output_scale_of` 按 `output_buffer_dtype` 识别）随后会把 V cache 的 requant 当成 fp16 处理，数值口径会静默出错。

**涉及文件**：`gml_bridge/from_fx.py:2152–2157`（`is_v` 判定）、`2173–2179`（kantor 字段）；测试只覆盖写盘侧（`tests/test_runtime_files.py::test_gemm_feeding_value_cache_is_found_by_structure`），序列化侧没有覆盖。

**修改思路**：改为结构判据。沿 Gemm 的输出穿过布局算子，找到写 V 的 KV_Cache_DMA（decode），或找到 `_graph_kv_outputs` 判定为 V 的出口（prefill），与 K 路的判定共用同一套遍历。新增用例：V 投影改名后，prefill 和 decode 仍输出 int8 并带 kantor 字段，且 GML 与改名前逐字节相同。

### 问题2（中）：生成摘要取 HEAD，导致改代码后仍可混导
**问题说明**：R5-1 的修复（设计 5.3：清单记 `generation_digest`，口径包含"编译版本"）用 `git rev-parse --short HEAD` 代表编译器版本。本期改动都没有提交，HEAD 一直是 `bf5b8a5`，改写盘规则不会改变摘要。实证：`bundles/own_bins`（R5-3 之前）和 `bundles/r5_zero_l0`（R5-3 之后）的 `compiler_revision` 都是 `bf5b8a5`，但第 0 层 decode 有 844 个文件内容不同。在旧包上只重导部分图，当前守卫会放行，旧图和新图按不同规则生成，包仍可能标成 `complete=true`。

**涉及文件**：`gml_bridge/ceva.py:962–972`（`generation_digest`）、`984–987`（`revision` 取值）；`tests/test_ceva_export.py:1539、1568`（只覆盖换标定值和换 checkpoint，没有覆盖换代码）。

**修改思路**：编译版本改为能反映工作树的值，例如把 HEAD 和影响生成的源码文件（`gml_bridge/`、`quant/`、`graph/`、`contracts/`、`opcompiler_bridge/`）的内容摘要拼起来，或者工作树有未提交改动时直接拒绝局部续导。补一个"同一 HEAD 下改动生成代码后局部重导被拒绝"的用例。

### 问题3（中）：写零后未重跑全流程，导致 CEVA 可运行无证据
**问题说明**：需求 P0-2/P0-5 要求当前产物经 GC 编译、在 CEVA 跑通。实施 §1.3 宣称"三条核心目标全部达成"，依据是 11-01 的 `own_bins_e2e`。R5-3 之后，运行时缓冲改为写零，第 0 层 decode 有 844 个文件、prefill 有 839 个、lm_head 有 3 个与那次运行的包不同。目前只验证了 decode 和 lm_head 的 Registry 与旧包一致（lm_head 只差第 222 字节）。prefill L0 的 GC 在评审时仍在运行，其余 62 张图和整模型仿真都没有用当前代码重跑。另外，实施 §7.1 的 R5-3 一行写着"GC 验证见 7.3"，但 §7.3 只记到第三轮，没有第五轮的结果。

**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` §1.3、§2、§7.1 R5-3 行、§7.3；运行产物 `/data/fjg/ceva/bundles/r5_zero_l0`、`/data/fjg/ceva/runs/r5_zero_l0`。

**修改思路**：等 prefill L0 GC 结束后，比较它与 `own_bins_e2e` 的 Registry，结果写入 §7.3。如果 65 张图的 Registry 都与旧运行相同（只差会随编译变化的字节），可以据此说明旧仿真结论仍然有效。否则在磁盘允许时重导 65 图，跑 GC 和至少 1 步 decode 的仿真。§1.3 的结论措辞改为只覆盖已经验证的范围。

### 问题4（中）：工作区取消跟踪运行记录，导致验收归档缺失
**问题说明**：需求第四章要求归档生成命令、判定报告、token 等，工作区 CLAUDE.md 原约定"git 只保留步骤日志、verdict、argmax 等摘要"。本次 `.gitignore` 改成整个 `runs/` 都忽略，并把 7 个已跟踪运行目录的 76 个摘要文件从索引中删除（已暂存，磁盘上的文件还在），包括厂商基线 `20261027_e2e_clean`，以及 `ours_e2e`、`ours_e2e2`、`ours_full` 的 verdict 和 run.log。这一改动不在设计和实施文档中，提交后，判定依据就不再受版本管理；`own_bins_e2e` 的结果也没有进入 git。

**涉及文件**：CEVA `.gitignore`、`CLAUDE.md`（目录职责一行），以及暂存区的 76 个 `runs/*` 删除项。

**修改思路**：由开发者决定是否保留这一策略。如果保留，在实施文档里写明归档改到哪里（例如每次运行目录的 `ARCHIVE.md`，以及哪些文件需要另外备份），并说明基线判定文件的来源。如果恢复原规则，取消暂存这些删除，并把 `own_bins_e2e` 的摘要按原规则纳入。

### 问题5（低）：设计文档清单接口未同步，导致文档与代码不符
**问题说明**：设计 5.3 只写了"清单记 `generation_digest`"，没有写摘要组成和 `check_ceva_manifest` 新增的引用集、参考尺寸和完整性检查（R5-2）。设计 4.1 的验收说明也只写"发现非 generated 即失败"。代码行为比设计多出了这些检查，设计文档没有同步。

**涉及文件**：`docs/design-ceva-llama2-7b-20261028.md` §4.1、§5.3；`scripts/verify_gml_artifact.py:942–1018`。

**修改思路**：在设计 4.1 和 5.3 中补充清单校验的四项检查（来源、摘要、引用集等于落盘集等于登记集、与参考同字段文件尺寸相同），以及摘要的组成，与问题 2 的修复一起更新。
