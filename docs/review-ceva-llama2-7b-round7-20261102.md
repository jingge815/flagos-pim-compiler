# 代码评审报告：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：review-ceva-llama2-7b-round7-20261102
> 创建日期：2026-11-02
> 关联需求文档：request-ceva-llama2-7b-20261028
> 关联设计文档：design-ceva-llama2-7b-20261028
> 关联实施文档：implement-ceva-llama2-7b-20261029
> 关联上一轮评审：[review-ceva-llama2-7b-round6-20261102](review-ceva-llama2-7b-round6-20261102.md)

## 一、评审概述
本轮复核第六轮 5 项问题的修复。R6-1、R6-2、R6-4、R6-5 已在代码和文档中落地，验证通过：`tests/test_ceva_export.py` 66 项、CEVA 工作区 `tests/` 43 项。R6-3 的 65 图全流程在 15:26 启动，评审时导出到 47/65 图，GC 和仿真尚未开始，实施文档也没有对应的结果记录。

本轮保留 2 项中、3 项低问题。快速回归在评审期间仍在运行（16%），本轮没有重跑 7B 慢测、GC 和仿真。

## 二、修改建议

### 问题1（中）：验证证据未落盘，R6-3 提前标已修复，导致结论无依据
**问题说明**：实施文档 §7.1 的 R6-3 行标为"已修复"，备注写"结果见 7.3 第六轮"，但 §7.3 只写到第三轮评审，没有第五、六轮的任何记录；R5-3 引用的"GC 验证见 7.3"同样落空。§1.3 的实施结论仍引用 11-01 的 `own_bins_e2e`，而 R5-3 写零之后第 0 层有 800 多个文件与该次运行的包不同，当前代码还没有全流程证据。截至评审时（15:37），`r6_full` 只导出到 47/65 图，GC、DDR、仿真和判定都未开始。

**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` §1.3、§7.1（R6-3 行）、§7.3（第 188 行起只有前三轮）；运行产物 `/data/fjg/ceva/bundles/r6_full`、`/data/fjg/tmp_ceva/r5/r6_full.sh`。

**修改思路**：运行结束后在 §7.3 补第六轮记录（导出与清单校验结果、GC 逐图结果、DDR、仿真步数与 token、判定结论），并据此更新 §1.3。运行完成前，R6-3 的状态改为"验证中"，不要写"已修复"和指向不存在章节的引用。

### 问题2（中）：口径摘要不含算子编译器，导致换 FlagTree 后可混导
**问题说明**：设计 §5.3（本轮更新）要求摘要覆盖影响生成结果的编译版本，保证同一包同一口径。`compiler_revision` 只对本仓 `gml_bridge/`、`quant/`、`graph/`、`contracts/`、`opcompiler_bridge/` 的 `.py` 求摘要，没有包含 FlagTree。而 GML 的 FPSU 模式是从 FlagTree 跑 pass 的输出读回的（`phase_source_from_graph` → `plan.fpsu_mode_name` 写进 `fpsu_mode`），相位也由算子编译器校验。FlagTree 在本任务中有 8 个未提交文件，只改 FlagTree 时 `compiler_revision` 和 `generation_digest` 都不变，旧包上局部续导会被放行，包内不同图可能按不同相位口径生成。

**涉及文件**：`gml_bridge/ceva.py:966–990`（`compiler_revision`、`generation_digest`）、`opcompiler_bridge/phase_source.py:171–189`（`phase_source_from_graph`）、`gml_bridge/from_fx.py:2161–2165`（读回 `fpsu_mode_name`）。

**修改思路**：把算子编译器的版本并入 `compiler_revision` 或单独并入 `generation_digest`，取 FlagTree 源码树摘要或安装目录里的版本文件；取不到时按"口径未知"拒绝局部续导。补一个"同一 HEAD、同一本仓源码，FlagTree 版本变化后局部重导被拒绝"的用例。

### 问题3（低）：V 出口就是 Gemm 本身时结构判据失效
**问题说明**：R6-1 把 V 投影识别改为结构判据，要求与名字无关、覆盖任意包装器。`_feeds_value_cache` 只遍历 `node.users`，不检查 writer 集合里是否就是该节点本身。实测传入 `kv_writers=[(node, False)]` 且节点无消费者时返回 `False`，V 投影会静默退回 fp16。当前两个包装器的 V 都经过 view/transpose 再成为图输出，所以现有 65 图不受影响；但直接返回 `v_proj` 输出的包装器会命中这个缺口。

**涉及文件**：`gml_bridge/from_fx.py:2281–2292`（`_feeds_value_cache`）。

**修改思路**：函数开头补 `if node in value_writers: return True`，并加一个 writer 就是节点本身的用例。

### 问题4（低）：实施文档用例数未更新，导致数字与实测不符
**问题说明**：`test_ceva_export.py` 现有 66 项，CEVA 工作区 `tests/` 现有 43 项；实施文档 §2 仍记 62 与 41，§4.2 的同一组数字也未更新，§4.1 的总数仍按第三轮统计。

**涉及文件**：`docs/implement-ceva-llama2-7b-20261029.md` §4.1、§2、§4.2。

**修改思路**：按本轮实测更新用例数与新增用例说明。

### 问题5（低）：设计 5.3 的 role 取值过时，导致文档与校验器矛盾
**问题说明**：设计 §5.3 仍写"`role` 取值为 `imported`、`generated`、`runtime`"，但 11-01 起清单只允许 `generated`（设计 4.1、4.7 与 `check_ceva_manifest` 都是这个口径）。R6-5 更新 5.3 时漏改这一行。

**涉及文件**：`docs/design-ceva-llama2-7b-20261028.md:545`。

**修改思路**：改为只允许 `generated`，与 4.1 的验收口径一致。
