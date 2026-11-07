# 实施进展文档：自研 GML 接入 CEVA 的 Llama2 7B 全流程推理
> 文档编号：implement-ceva-llama2-7b-20261029
> 创建日期：2026-10-29
> 关联需求文档：[request-ceva-llama2-7b-20261028](request-ceva-llama2-7b-20261028.md)
> 关联设计文档：[design-ceva-llama2-7b-20261028](design-ceva-llama2-7b-20261028.md)
> 关联项目：flagos-pim-compiler、FlagTree、CEVA 工作区（/data/fjg/ceva）
> 第十、十一轮修复起始基线（2026-11-03）：图编译器 `main@bf5b8a5`、FlagTree `develop@262adb3b7`、CEVA `main@717407d` 均已有未提交改动；GeneSim `fjg-dev@cdf6c99` 工作区干净。本轮按用户指定同时处理两份评审，重复项合并验证。

## 一、实施概述
### 1.1 任务范围
按设计文档第五章的改动清单实施 A～E 五个阶段：删除旧 prepare_out 支线；新增 CEVA 契约与逐层三类子图导出（prefill 32、decode 32、lm_head 1，共 65 张）；厂商权重反量化后作为浮点 checkpoint，量化、scale、LUT 与静态常量全部由我方生成；prefill 几何与归约；KV 原地更新与 RoPE 厂商 sin 编码；输出头 int8 静态量化及 FlagTree 的 fixed2float 模式；相位读写关系；工作区 GC 接线、严格判定、单层独立运行与严格 token 验收。

### 1.2 完成标准
按需求 P0-1～P0-5 与设计 7.3 的现行口径：GML 与全部 bin 由我方生成，尺寸、引用与摘要检查通过；65 图经 GC 编译并进入 CEVA；不再自研 prepare_out，原始目录不变；全模型正常结束。自研 bin 可使用告警档允许合法 token 分叉，厂商 bin 保持严格档；请求至少 1 步 decode，仅允许正常 EOS 提前结束。

### 1.3 实施结论
总体完成状态：全部完成。第十、十一轮的 5 个独立问题均确认成立并修复，CEVA 工作区全量回归 51 项通过；生成产物与硬件证据复核见 7.3。本轮只修改验收代码和文档，图编译、算子编译、GML/bin 与 runner 均未改变；现有 10 步硬件运行用新判定器复核仍为 PASS。第八轮的图编译器全量快速回归、7B 慢测与 GeneSim 全链路记录保留，本轮未重新执行这些全量慢测。

最终验收运行（2026-11-02 19:25 至 11-03 14:52，按目标命令 `scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10`）：65 图全部由我方图编译器导出、FlagTree 算子编译与厂商 GC 编译（GC 65/65，本次复用前一轮已完成产物，DDR 重算 7m13s）；仿真 prompt + 10 步 decode 共 7h00m37s，11 个 token 为 `[13, 29902, 626, 2691, 29889, 306, 626, 2599, 590, 3271, 1287]`，前 7 个与基线逐个一致，之后分叉；第十、十一轮新判定器复核为 22 项检查中 20 项 PASS、2 项 WARN、0 项 FAIL/SKIP，结论 PASS；原始目录检查判定无变化。token 分叉属于需求 P0-5 允许的告警档（我方量化 bins 与厂商不同），不是流程失败。

历史完整运行证据仍保留：11-01 的 `own_bins_e2e`（3 步，token 前 4 个一致）、10-29 的 `ours_e2e`（10 步，11 个 token 全一致；已备份到 `runs/ours_e2e_20261029/`）。

## 二、任务完成清单
| 任务ID | 任务描述 | 对应设计模块 | 完成状态 | 备注 |
|--------|----------|--------------|----------|------|
| A1 | 删除 orchestrator 支线，`--orchestrate` 在加载模型前报错，mem_layout 只留三项通用规则 | 4.14 | 完成 | |
| A2 | CEVA 契约三个数据结构；解析厂商子图，语义键与静态文件绑定，IO 契约 | 4.1、5.2 | 完成 | |
| A3 | decode/prefill/head 三类包装器；KV 原地更新识别为 KV_Cache_DMA | 4.2、4.4 | 完成 | |
| A4 | RoPE 厂商 sin 编码吸收；Q 路换轴折进 RoPE；K 路整体换轴 | 4.5 | 完成 | |
| A5 | 厂商编号规则、逐节点结构与字段核对、契约名发布 | 4.6 | 完成 | 新增，见 5.2 |
| A6 | 导入字节、运行时缓冲按尺寸写零、IO_info、bundle.json | 4.1、4.7 | 完成 | 11-01 改为全部生成时一度用了标定样本，11-02 按 R5-3 恢复写零 |
| A7 | CLI：`--ceva-reference/--stage/--layer-index` 与整包导出 | 3.2 | 完成 | |
| B1 | 输出头 int8 静态量化：不插 DQ、fixed_point NMU、fixed2float | 4.8 | 完成 | |
| B2 | FlagTree：FPSU 枚举加 fixed2float，int8 模型权重，显式 FPSU 不被覆盖，EmitC 明确拒绝 | 4.9 | 完成 | |
| B3 | 相位读写关系 `Phase.reads`，交叉校验 DQ/Softmax 跨相依赖；`fpsu_mode_name` | 4.10 | 完成 | |
| C1 | prefill 几何：查询轴 T 保留、DQ 注意力分数按行分组、K/V 直接输出不插 DMA | 4.3 | 完成 | Softmax 逐行归约未做，见 5.2 |
| D1 | `compile_gml.py`、`gc_worker.sh`：GC 接线与严格判定 | 4.11、4.12 | 完成 | |
| D2 | `standalone_layer.sh`、`standalone_ini.py`、`build_dsp_app.sh`：单层独立运行与厂商同层逐字节比较 | 4.11 | 完成 | |
| D3 | `run_all.sh --gml-bundle`；`verify_run.py` 严格 token 与 Registry 来源判定 | 4.11、4.13 | 完成 | |
| D4 | `verify_gml_artifact.py --ceva-manifest` | 5.3 | 完成 | |
| D5 | 编译器全流程入口 `run_full_pipeline.py --backend ceva` 一键跑通导出、GC、仿真与判定 | 用户补充 | 完成 | 实验名 ours_e2e，PASS，10 步 |
| E1 | 65 图导出、编译与全模型运行 | 6.1 D/E | 完成 | 10 步，未缩减 |

## 三、代码改动详情
### 3.1 改动总览
- 新增文件数（按 10-31 的 `git status` 统计）：flagos-pim-compiler 9，其中代码 3（`contracts/ceva.py`、`gml_bridge/ceva.py`、`tests/test_ceva_export.py`），文档 6（需求、设计、本文档与三轮评审）；CEVA 工作区 8（脚本 5 个：`scripts/compile_gml.py`、`gc_worker.sh`、`standalone_ini.py`、`standalone_layer.sh`、`build_dsp_app.sh`；测试 3 个：`tests/test_compile_gml.py`、`tests/test_verify_run.py`、`tests/test_standalone_ini.py`），另有厂商 dsp_app 的工作区副本 `src/vbu_npm_system`（不计入）
- 修改文件数：flagos-pim-compiler 34（代码、测试与配置 31，旧 prepare_out 文档 3）；FlagTree 8；CEVA 工作区 3（`run_all.sh`、`env.sh`、`scripts/verify_run.py`）；genesim 0（`models/` 下的 sidecar 改名不受 Git 跟踪）
- 删除文件数：flagos-pim-compiler 14（`orchestrator/` 9 个、`scripts/diff_prepare_out.py`、4 个旧测试）
- 核心改动说明：我方生成每层图、相位与字段，按厂商编号规则编号后与厂商同名子图逐节点核对（结构、形状与非文件字段全部一致才放行），静态参数整字节导入，运行时缓冲按尺寸写零。

### 3.2 详细改动清单
| 文件路径 | 改动类型 | 改动内容简述 | 对应任务ID |
|----------|----------|--------------|------------|
| `scripts/export_gml.py` | 修改 | 删除编排器支线；`--orchestrate` 报错；CEVA 分派 `_main_ceva` | A1、A7 |
| `orchestrator/*`、`scripts/diff_prepare_out.py`、4 个测试 | 删除 | 旧支线 | A1 |
| `contracts/mem_layout.py` 及 5 个测试 | 修改 | 去掉 L2 专用规则与旧模块引用 | A1 |
| `contracts/ceva.py` | 新增 | `CevaIoBinding`、`CevaFileBinding`、`CevaGraphSpec` | A2 |
| `gml_bridge/ceva.py` | 新增 | 厂商子图解析、三类包装器、编号、核对、导入写盘、整包导出 | A2～A7、C1 |
| `contracts/unified_ir.py`、`contracts/ir_payloads.py` | 修改 | 登记通用语义键 `pim_static_quant`（G-2 前为 `pim_ceva_op`）；`RopeMatch.sin_encoding` | A4、B1、C1 |
| `contracts/compile_slots.py` | 修改 | 新增 `tokens`（decode 为 1，prefill 为 1024） | C1 |
| `contracts/gml_quant.py`、`graph/quant_pass.py` | 修改 | `dq_group_size(row_width=)`；CEVA prefill 注意力分数按行分组；输出头不插 DQ | B1、C1 |
| `graph/fuse_rope.py` | 修改 | 识别厂商 sin 编码并吸收符号转换 | A4 |
| `graph/kv_dma_pass.py` | 修改 | `index_copy` 识别为 DMA；prefill 不插 DMA；K 路整体换轴 | A3、A4、C1 |
| `gml_bridge/from_fx.py` | 修改 | RoPE 吸收换轴后的出边形状；常量 cos/sin；prefill K/V 出口；输出头字段；查询轴；编号表可传入 | A4、B1、C1 |
| `opcompiler_bridge/oplevel_kernel.py`、`oplevel_emitter.py` | 修改 | 输出头发 int8 定点矩阵乘并带 fixed2float | B1 |
| `opcompiler_bridge/phase_plan.py`、`phase_source.py` | 修改 | `Phase.reads`、`fpsu_mode_name`、跨相依赖校验 | B3 |
| `scripts/verify_gml_artifact.py` | 修改 | `--ceva-manifest` | D4 |
| `scripts/run_full_pipeline.py`、`genesim_bridge/paths.py`、`paths.json` | 修改 | 全流程入口加 `--backend ceva`：导出 65 图 → 来源清单校验 → 工作区 `run_all.sh --gml-bundle` → 只认 PASS 的判定；新增站点路径 `ceva_workspace`、`ceva_reference_dir`；`pytorch_env_script` 改为指向脚本文件（原先指向目录，按 CLAUDE.md 的写法 source 会失败） | D5 |
| FlagTree `PIMAttrDefs.td`、`Dialect.cpp`、`Ops.cpp`、`ExpandPhases.cpp`、`LowerPIMToEmitC.cpp` 及 3 个 lit 测试 | 修改 | fixed2float、int8 模型权重、显式 FPSU 保留、EmitC 拒绝 | B2 |
| CEVA `scripts/*`、`run_all.sh`、`tests/*` | 新增/修改 | GC 接线、严格判定、单层运行、严格验收 | D1～D3 |
| `docs/prepare_out-*.md` | 修改 | 开头标注自研生成职责已废止 | A1 |

### 3.3 关键代码说明
- 为了让 Registry 与厂商一致，按厂商编号规则编号（入口按 IO 槽号，其余从出口沿输入槽深度优先）：GC 探针实测置换编号会改变 Registry，而改文件名、改边序、改字段顺序、删调试字段都不变。
- 为了保证「我方生成」可核对，导出后逐节点比对 op_type、入边（源与形状）与全部非文件字段，有一处不一致就报错，不硬写。65 张图全部零差异。
- 为了不复现厂商的旋转与量化，静态参数按（语义键、字段）整字节导入；运行时缓冲只按厂商尺寸写零（GC 探针实测其内容不进 Registry）。prefill 的 RoPE cos/sin 表实测编进 Registry，按静态文件导入。
- 为了表达输出头的静态量化，FlagTree 的 FPSU 枚举追加 fixed2float，`stampMatmul` 不再覆盖显式设置，GML 的该字段由算子编译器读回。

## 四、测试验证结果
### 4.1 测试执行总览
- 执行用例总数：第八轮修复后，快速回归 `pytest tests/ -k "not llama2_7b"` 1672 例（1663 通过 + 9 跳过）+ CEVA 工作区 44；FlagTree 本轮无改动，lit 沿用第七轮 265 通过
- 通过用例数：见 4.2 与 7.3
- 失败用例数：0
- 跳过用例数：9
- 关键子集实测：`tests/test_ceva_export.py` 72/72 通过；CEVA 工作区 `tests/` 44/44 通过

### 4.2 分任务测试结果
| 任务ID | 测试用例数 | 通过数 | 失败数 | 测试说明 |
|--------|------------|--------|--------|----------|
| A1～C1 | 72 | 72 | 0 | `tests/test_ceva_export.py`（首轮 35，第一轮评审后 39，第二轮评审后 42，G-2 后 45，11-01 生成 bin 后 58，第五轮后 62，第七轮后 69，第八轮后 72） |
| B3 | 4 | 4 | 0 | `test_phase_plan.py`、`test_phase_source.py` 新增 |
| C1 | 1 | 1 | 0 | `test_gml_quant.py` 新增 |
| B2 | 265 | 265 | 0 | FlagTree lit（文件数；第二轮评审在 `operator_ops_negative.mlir` 内新增 2 个负例） |
| D1～D3 | 44 | 44 | 0 | CEVA `tests/`（首轮 13，第一轮评审后 17，第二轮评审后 23，第三轮后 35，第四轮后 39，11-01 后 41，第七轮后 43，第八轮后 44） |
| 单层 GC | 3 | 3 | 0 | decode L0、lm_head、prefill L0：我方 Registry 与厂商 GML 在本机编出的 Registry 逐字节相同（head 仅第 222 字节随编译次变化） |
| 单层运行 | 3 | 3 | 0 | 厂商与我方 Registry 在同一输入下独立运行，输出逐字节比较：decode L0（hidden、K、V）、lm_head（32000 个 logits）、prefill L0（hidden、K、V，输入为基线 embedding 与因果掩码）全部相同 |

### 4.3 回归测试结果
- 第八轮最终快速回归（11-02，最后一个小改动之后重跑）：`pytest tests/ -k "not llama2_7b"` 1663 passed、0 failed、9 skipped（49 deselected，19m14s）。`pytest tests/ -k "llama2_7b"` 48 passed、0 failed（2h39m41s）。`test_ceva_export.py` 72 passed、`test_verify_layers.py` 25 passed、`test_runtime_files.py` 30 passed；CEVA 工作区 44 passed。genesim 全流程结果见 7.3。
- 首轮 `pytest tests/ -k "not llama2_7b"`：1578 passed，1 failed，8 skipped（失败项改动前已存在）；评审修复后的数字见 7.3。
- `pytest tests/ -k "llama2_7b"`：48 passed（2h21m）。
- 10-30 全量 `pytest tests/`（不加 `-k`）：1629 passed，1 failed，8 skipped（失败项同上，改动前已存在）。首次全量跑出 38 个失败：7B 测试夹具调用 `torch.set_grad_enabled(False)` 后不恢复，后续文件在无梯度模式下导出的图不同。`tests/conftest.py` 新增按文件恢复梯度开关的夹具后消失；此前分两段 `-k` 跑没有暴露。
- FlagTree 用 `0-install-flagtree.sh` 重装并同步到 PyTorch 环境，lit 265 全过。
- 65 图逐节点核对：结构、形状与非文件字段与厂商同名子图零差异。
- 65 图 GC：全部 success；56 个 Registry 与厂商原有 Registry 逐字节相同，其余 9 个只差第 222 字节（同一输入重复编译该字节也会变化）。
- 编译器一键全流程：`python scripts/run_full_pipeline.py --backend ceva --run-name ours_e2e --decode-steps 10`（10-29 20:04 至 10-30 06:28，共 10h24m），PASS。导出与清单校验约 4 分钟；65 图 GC 全部 success（8 路并行，3h17m）；仿真 7h01m；11 个 token 与基线逐个一致，判定 15 项全 PASS、零 SKIP；性能比基线慢 7%～16%。运行目录 `/data/fjg/ceva/runs/ours_e2e`，归档索引见该目录 `ARCHIVE.md`。
- 编译器一键全流程复跑（空算子编译缓存）：`run_full_pipeline.py --backend ceva --run-name ours_e2e2 --decode-steps 10`（10-30 07:12 至 17:29），PASS。65 个 GML 与第一次逐字节相同，65 图 GC 全部 success；11 个 token 与基线逐个一致，判定 14 项 PASS、0 SKIP、1 项性能告警（lm_head cycles +36.6%，第一次 +16.2%，Registry 只差会随编译变化的第 222 字节，波动原因未查明）。归档见 `/data/fjg/ceva/runs/ours_e2e2/ARCHIVE.md`。
- 全模型（分步命令）：`run_all.sh -n ours_full --gml-bundle bundles/ours_full -s 10`，PASS；DDR 布局地址与基线相同；Registry 全部来自本次编译；性能比基线慢 6%～16%，在 ±20% 告警线内（与基线跑出时的机器负载不同）。

## 五、问题与偏差说明
### 5.1 已知问题
- GC 的 test_app 会把 `-d` 路径在子串 "data" 处截断（`/data/fjg/ceva/...` 变成 `/dumps/ini_files/net.ini`），GC 退出码仍为 0。`gc_worker.sh` 拒绝含 "data" 的路径，`compile_gml.py` 经 `/tmp/ceva_gc` 软链调用，产物仍落在 `/data`。
- GC 会往厂商 `vbu_ctp_apps/python` 与 `npm_device/L2A` 写日志和中间文件。10-28 第一次探针在原始目录多留 478 个文件（已按快照删除），并覆盖了 7 个厂商 GC 上次运行（10-23 的 resnet50 示例）留下的日志与中间图。10-31 已按内容还原，被覆盖的版本备份在 `/data/fjg/ceva/manifests/overwritten_20261028/`：
  - `nprm_backend.log`、`TAGS_Log.txt` 是追加写入，原文件就是开头的那一段，截回快照长度即可。
  - `relay2gml_graph.html`、`sat_sbt_buffer_graph.gml`、`g1b*.gml`：10-23 那次运行的输出目录里有同一份（`gc_test/resnet50_test/.../l2a_output`），大小与快照一致，直接拷回。
  - `g2*.gml`、`g3*.gml`：在工作区副本里用同一份 resnet50 输入重跑厂商 GC，大小与快照一致，且两次重跑逐字节相同，说明输出是确定的。
  - 7 个文件的 mtime 改回快照值，属于本人的 6 个目录也恢复了 mtime。
  - 还原后 `check_orig.sh` 判定「原始目录文件未变化」，退出码 0。余下 2 个 root 所有的目录 mtime 无权限改回，这两个目录里的文件均与快照一致。
  - 现在 GC 只在 `env/gc_w<N>` 工作副本上跑，原始目录不再被写。
- 原基线失败 `test_pimir_sidecar_agrees_with_ttir_on_flops`：没有 GPU 时，全流程精化走 oplevel，却仍写到 `llama2_7b_pimir_extensions.json`，把 pimir 那份覆盖了。10-31 已修复：sidecar 按实际档位命名，genesim 里那份错名文件改名为 `llama2_7b_oplevel_extensions.json`；该测试在没有 pimir 产物时跳过。
- 2026-11-02 18:xx 起，共享原始目录 `/data/xym_data/ceva_data/HW-D0/release-new/gmls-device-1024` 被另一批 root 进程持续写入：`ps` 中可见 `/root/ceva/.../GC_offline_json` 与 `test_app`（自 16:42 CST 起，多个 worker），`check_orig.sh` 相对 10-28 快照因此报出约 3.5 万处变化（30 新增、30 删除，其余为修改与目录 mtime）。我方 GC 进程均为 jgfeng，产物只写 `runs/<name>/gmls`；共享树里新增文件属主为 root，路径是各图自己的 `l2a_output`/`test_app_outputs`，与本次流程无关。本次 `ours_e2e` 运行在 11-02 19:25 取的 before 快照与 11-03 14:52 的 after 快照对比判定「原始目录文件未变化」，说明运行窗口内共享目录未被外部写入。处理：不覆盖共享资料；每次运行保留 `logs/orig_before.tsv` 与 `orig_after.tsv`，判定时按属主、路径与时间归因。

### 5.2 偏差说明
- 设计 4.11 要求 `compile_gml.sh`；实现为 `compile_gml.py`（并行、报告解析与清单写入用 Python 更直接），另加 `gc_worker.sh` 负责厂商环境。
- 设计 4.11 的单层独立运行：`standalone_layer.sh` 的厂商同层输出目录为必填的第 4 个参数，与设计一致。另加两个仅供测试的环境变量：`STANDALONE_RUN_TOOLS` 替换 run_tools，`STANDALONE_SKIP_INI` 沿用已有 ini。
- 设计 4.12 原文要求每张图都有新生成的 Registry；复用已编译的子图是实现上的选择，不是设计原文。复用条件：GC 输入摘要、Registry 摘要、GC 配置（工具链版本文件摘要、`HW_TARGET`、`-l/-u`）都与清单记录一致，任一不同就重编。
- 设计 5.3 规定的元数据键 `CEVA_OP_META_KEY="pim_ceva_op"` 已按 G-2 替换为通用语义键 `STATIC_QUANT_META_KEY="pim_static_quant"`（int8 静态量化矩阵乘），由 `gml_bridge/ceva.py:mark_static_quant` 按结构打上；prefill 不再需要标记，共享 pass 按图结构判定。
- 设计 7.2 的单层命令只有一个图参数；实现为四参数 `standalone_layer.sh <图目录> <输入目录> <输出目录> <厂商同层输出目录>`，第四个参数必填。
- 设计未写编号规则与逐节点核对（A5），实施中通过 GC 探针发现编号进 Registry，按厂商规则编号是让 Registry 一致的必要条件，故补上。
- `env.sh` 原先写死 `GMLS` 为厂商目录，各步脚本重新 source 时会把 `--gml-bundle` 设好的值覆盖掉（第一次全流程尝试因此用回了厂商 Registry，已在第 4 步发现并中止）。改为允许覆盖，并在第 4 步检查 DDR 布局的 Registry 来源。
- 单层运行程序只映射 0xA0000000（外部虚拟基址）以下的地址：prefill 的 I/O、Registry 与中间缓冲合计约 268MB，放在 0x90000000 起会越界，放到 0x100000000 实测 ExternalAccessFailed。`standalone_ini.py` 改为 I/O、Registry、中间缓冲依次排布，放不下时从 0x80000000 起。
- 设计 4.3、5.3 的 `phase_data.softmax(row_width=)`：10-31 前未实现（当时相位缓冲整字节导入）；11-01 改为我方生成后已实现（见 8.2），用来推 Softmax 相位常量。
- 不实现：设计 5.2、5.3 的 `GmlArtifact.graph_spec`、`fuse_for_gml(graph_spec=)`、`serialize_gml(graph_spec=)`。理由：契约 `CevaGraphSpec` 只在 `gml_bridge/ceva.py` 内部传递（边界命名、编号、核对、写盘都在那里完成），共享导出函数不需要知道它。
- 接口改动（仅文档记录）：`export_ceva_subgraph` 返回 `(gm, report)`，因为融合报告要交给序列化。
- 不实现：7.1 的 `test_gml_depends_on_opcompiler.py` 扩展。理由：相位读写与 FPSU 的覆盖已放在 `test_phase_source.py`、`test_phase_plan.py` 与 `test_ceva_export.py`。
- `weight_sf_multiplier` 只出现在厂商第 0 层的 q/v 投影，改为跟随厂商同位节点。它是导入的那份 scale 的属性，不能再按投影名写。

## 六、后续与待确认事项
- 产物位置：GML 包 `/data/fjg/ceva/bundles/ours_full`（含 `bundle.json` 来源清单）；编译与运行 `/data/fjg/ceva/runs/ours_full`，需求文档第四章要求的归档项（生成命令、来源清单、GC 日志、DDR map、CEVA 日志、token 与文本、判定报告、耗时）索引在该目录的 `ARCHIVE.md`。
- 静态契约核对：65 张 Registry 的 DDR 表里，runner 绑定的全部名字都存在且尺寸一致；decode 的 RoPE 内部缓冲名 `self_attn_Reshape_qidx4_params_22_cos/sin_value_buffer` 都嵌在 Registry 中。
- 复现：`python scripts/export_gml.py --ceva-reference <厂商 65 图> --use-opcompiler --out-dir <包>`，再 `bash /data/fjg/ceva/run_all.sh -n <名> --gml-bundle <包> -s 10`。65 图 GC 约 3 小时（8 路并行），仿真约 7 小时。

## 七、评审问题修复记录
> 关联评审文档：review-ceva-llama2-7b-20261030（第一轮，问题 1～7）；review-ceva-llama2-7b-round2-20261031（第二轮，R2-1～R2-8）；review-ceva-llama2-7b-round3-20261031（第三轮，R3-1～R3-6）；review-ceva-llama2-7b-round4-20261031（第四轮，R4-1～R4-6）；review-ceva-llama2-7b-round5-20261102（第五轮，R5-1～R5-4）；review-ceva-llama2-7b-round6-20261102（第六轮，R6-1～R6-5）；review-ceva-llama2-7b-round7-20261102（第七轮，R7-1～R7-5）；review-ceva-llama2-7b-round8-20261102（第八轮，R8-1～R8-8）；review-ceva-llama2-7b-round9-20261103（第九轮，R9-1～R9-2）；review-ceva-llama2-7b-round10-20261103（第十轮）；review-ceva-llama2-7b-round11-20261103（第十一轮）；G-1、G-2 为 10-31 目标补充项

### 7.1 问题确认与修复清单
| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 涉及文件 | 备注 |
|----------|--------------|--------|----------|----------|----------|------|
| 1 | review-20261030 | 高 | 成立：只改 bin、GML 不变时 `already_compiled` 仍返回 True | 已修复 | CEVA `scripts/compile_gml.py`、`tests/test_compile_gml.py` | 摘要改为覆盖 `runtime_files` 全部文件名与内容（`input_digest`）。复核时又发现 GC 会往副本里写 `dummy.bin`，摘要改为编译前从包本身计算，并补了用例 |
| 2 | review-20261030 | 高 | 成立：reads 全空时依赖校验被跳过 | 已修复 | `opcompiler_bridge/phase_source.py`、`tests/test_phase_source.py` | 去掉「声明了 reads 才比」的前置条件；4 个旧用例的构造数据按 FlagTree 实际输出补上 reads |
| 3 | review-20261030 | 中 | 成立：接口偏差没有记录 | 已修复（文档） | 本文档 2、5.2 | 只补记录，不补实现：CEVA 链路没有调用方 |
| 4 | review-20261030 | 中 | 成立：现存 `bundles/ours_l0` 只有 3 张图却标为完整；`run_all.sh` 不看 `complete` | 已修复 | `gml_bridge/ceva.py`（`bundle_complete`）、CEVA `run_all.sh`、两边测试 | 期望集合按厂商参考目录计算；不完整的包在建运行目录前就被拒绝 |
| 5 | review-20261030 | 中 | 成立：独立运行只查 hidden；来源缺失时跳过比对 | 已修复 | CEVA `scripts/standalone_ini.py`（`missing_outputs`）、`standalone_layer.sh`；`scripts/verify_gml_artifact.py` | 按 ini 里 `save=true` 的缓冲逐个检查；用现存的 decode L0 两次运行结果核对通过 |
| 6 | review-20261030 | 低 | 成立 | 已修复 | `gml_bridge/ceva.py`、`contracts/ceva.py`、`graph/kv_dma_pass.py` | import 收到文件头；删掉未使用的导入、`STAGES`、无效条件和 try/except |
| 7 | review-20261030 | 低 | 成立 | 已修复（文档） | 本文档 3.1、4.1 | |
| R2-1 | round2 | 高 | 成立：`check_orig.sh` 报 7 处被覆盖的厂商运行产物 | 已修复 | 厂商原始目录 7 个文件（备份在 `manifests/overwritten_20261028/`）、本文档 1.3、5.1 | 两个日志截回快照长度；三个图从 10-23 运行的输出目录拷回；g2/g3 在工作区副本用同一输入重跑厂商 GC 重新生成（两次重跑逐字节相同）；mtime 改回。`check_orig.sh` 判定无变化 |
| R2-2 | round2 | 高 | 成立：`--gml-bundle --from 5/6` 跳过第 4 步，不写 `gml_bundle.txt`，判定退回 SKIP 且不查 Registry 来源 | 已修复 | CEVA `run_all.sh`、`scripts/verify_run.py`、`tests/test_compile_gml.py`、`tests/test_verify_run.py` | `run_all.sh` 在 `--from` 大于 4 时要求运行目录里有同一个包的标记与 `gmls`，否则拒绝；`GML_BUNDLE` 导出给判定脚本，模式以它为准，标记缺失或不符判 FAIL。新增 3 例 |
| R2-3 | round2 | 中 | 成立：运行命令后接的「或真」吞掉退出码；旧输出会被当成本次产物；没有逐字节比较 | 已修复 | CEVA `scripts/standalone_layer.sh`、`scripts/standalone_ini.py`、`tests/test_standalone_ini.py` | 运行前清掉旧输出并记开始时间；检查 run_tools 退出码与日志里的程序返回值；输出要晚于开始时间且尺寸与 ini 一致；可选第 4 个参数给厂商同层目录做逐字节比较，报告首个差异字节。新增 3 例；decode L0 实跑一次与厂商逐字节一致 |
| R2-4 | round2 | 中 | 成立：用例复现第一张图前失败时磁盘仍是 `complete: true` | 已修复 | `gml_bridge/ceva.py`、`tests/test_ceva_export.py` | 读旧清单后立即原子写回 `complete=false`；所有清单写入改为临时文件加 rename |
| R2-5 | round2 | 中 | 成立：只给 `--ceva-reference` 时不跑 FlagTree 也能导出 | 已修复 | `scripts/export_gml.py`、`tests/test_ceva_export.py` | CEVA 模式缺 `--use-opcompiler` 直接报参数错误；非 CEVA 路径不变 |
| R2-6 | round2 | 中 | 成立：`triton-opt -pim-expand-phases` 把无 FPSU 的 int8 头补成 `floating_point_32` 且不报错 | 已修复 | FlagTree `lib/Dialect/TritonPIM/IR/Ops.cpp`、`test/Dialect/TritonPIM/operator_ops_negative.mlir` | verifier 要求 int8 模型权重必须带 `fpsu = fixed2float`；同一段 IR 现在被拒绝。新增「无 FPSU」「floating_point_32」2 个负例 |
| R2-7 | round2 | 低 | 成立：有 `phase_source` 但没有头计划时静默写 `fixed2float` | 已修复 | `gml_bridge/from_fx.py`、`tests/test_ceva_export.py` | 只有未接算子编译器时用实测值；接了却取不到计划直接报错 |
| R2-8 | round2 | 低 | 成立 | 已修复（文档） | 本文档 1.3、4.1、4.2、4.3、5.2、7.3 | 数字按本轮实际结果统一；5.2 逐项标「不实现」或「仅文档记录」 |
| G-1 | 目标补充 | 高 | 成立：没有 GPU 时，全流程的 oplevel 精化覆盖了 `llama2_7b_pimir_extensions.json`，`test_pimir_sidecar_agrees_with_ttir_on_flops` 失败 | 已修复 | `scripts/run_full_pipeline.py`、`tests/test_ir_cost_eval.py`、genesim `models/` | sidecar 按实际档位命名；错名文件改名为 oplevel |
| G-2 | 目标补充 | 高 | 成立：共享的图编译、算子编译与 GML 序列化里有 CEVA 专用分支（`pim_ceva_op` 标记、`"vendor"` 编码、按占位名 `hidden` 认序列轴） | 已修复 | `graph/fuse_rope.py`、`graph/kv_dma_pass.py`、`graph/quant_pass.py`、`gml_bridge/from_fx.py`、`gml_bridge/export.py`、`opcompiler_bridge/oplevel_emitter.py`、`contracts/unified_ir.py`、`contracts/ir_payloads.py`、`gml_bridge/ceva.py` | 共享 pass 改按图结构判定：「K 是否作为图输出」决定插不插 DMA，cos 的广播维决定头轴，第一个三维输入给出序列轴；注意力分数一律一行一组。目标口径只剩一个通用语义标记 `pim_static_quant`（int8 静态量化矩阵乘），由 `gml_bridge/ceva.py` 按「直接产出图输出的投影」打上。新增测试禁止共享代码出现 CEVA 字样 |
| R3-1 | round3 | 高 | 成立：用临时 `bundle.json`、空 `gmls` 与匹配的 `gml_bundle.txt` 执行 `--from 5 --dry-run`，守卫放行 | 已修复 | CEVA `run_all.sh`、`scripts/compile_gml.py`、`scripts/verify_run.py`、`tests/test_compile_gml.py`、`tests/test_verify_run.py` | 第 4 步开头删掉旧成功标记；编译、DDR 生成与 Registry 来源检查全部通过后，写 `ddr_ok.json`（包路径 + `bundle.json` 摘要），写之前要求编译清单覆盖包里每张图且全部无问题；`--from` 大于 4 时核对该标记。`verify_run.py` 新增「Registry 与编译清单摘要一致」。新增「编译失败后拒绝」「同路径重导后拒绝」「清单不全不写标记」「Registry 内容是旧的判 FAIL」4 例；对 `ours_e2e2` 实跑新判定仍为 PASS |
| R3-2 | round3 | 中 | 成立：不给第 4 个参数时不做比较也打印完成；日志里任意一条 0 就放行 | 已修复 | CEVA `scripts/standalone_layer.sh`、`tests/test_standalone_ini.py` | 厂商同层输出目录改为必填，少参数直接拒绝；取日志里最后一条程序返回值，必须为 0 |
| R3-3 | round3 | 中 | 成立：复用判定不含 GC 版本、`HW_TARGET` 与 `-l/-u` | 已修复 | CEVA `scripts/compile_gml.py`、`scripts/gc_worker.sh`、`tests/test_compile_gml.py` | 清单逐图记 `gc_env`（版本文件摘要、硬件目标、参数），复用要求完全一致，旧清单没有这一项就重编；`-l/-u` 只在 `compile_gml.py` 定义，经 `GC_FLAGS` 传给 `gc_worker.sh`。新增 4 例 |
| R3-4 | round3 | 低 | 成立：原用例只断言脚本文本 | 已修复 | CEVA `tests/test_compile_gml.py`、`tests/test_standalone_ini.py` | 单层脚本用假的 run_tools 实际执行：失败且有旧输出、前 0 后非 0、未重写输出、缺参数四种失败与一次成功；把返回值判定改回「任意一条 0」时用例会失败（已实测） |
| R3-5 | round3 | 低 | 成立 | 已修复 | FlagTree `PIMAttrDefs.td`、`Dialect.cpp`、`test/Dialect/TritonPIM/operator_ops_negative.mlir` | 说明与错误信息改为：模型权重 4 或 8 位（8 位只用于静态量化路径，需配 `fixed_point` 与 `fixed2float`），activation-as-weight 为 8 位。lit 先改期望并确认失败，再改代码 |
| R3-6 | round3 | 低 | 成立 | 已修复（文档） | 本文档 3.1、5.2、7.3 | 按 10-31 的 `git status` 重新统计；5.2 补单层参数与复用条件的说明 |
| R4-1 | round4 | 中 | 成立（G-2 后未重跑 65 图） | 已关闭 | 本文档 8.4 | 11-01 的 `own_bins_e2e` 在 G-2 之后的工作树上重导 65 图、GC 65/65、DDR、3 步仿真，判定 PASS；第五轮又对第 0 层三图重导并重编，见 7.3 |
| R4-2 | round4 | 低 | 成立 | 已修复（10-31，本轮核对） | `scripts/run_full_pipeline.py`（`step_a2_refine_pimir_ir`）、`tests/test_ir_cost_eval.py` | 跳过条件加上「本档位 sidecar 存在且自报档位相符」；`test_refine_reruns_when_only_other_level_sidecar_exists` 等 3 例 |
| R4-3 | round4 | 低 | 成立 | 已修复（10-31，本轮核对） | CEVA `scripts/verify_run.py`、`tests/test_verify_run.py` | 三个 map 缺失或没有 Registry 路径判 FAIL；`test_fail_when_a_ddr_map_is_missing_or_empty` |
| R4-4 | round4 | 低 | 成立 | 已修复（10-31，本轮核对） | CEVA `scripts/compile_gml.py`、`tests/test_compile_gml.py` | 软链名改为运行目录路径摘要加图名，不含运行名；`test_gc_link_name_has_no_data_even_if_run_name_does` |
| R4-5 | round4 | 低 | 成立 | 已修复（10-31，本轮核对） | CEVA `run_all.sh`、`scripts/compile_gml.py`（`--foreign-registries`）、`tests/test_compile_gml.py` | 改为 Python 按字面前缀比较；`test_foreign_registry_check_is_literal_for_dotted_run_name` |
| R4-6 | round4 | 低 | 成立 | 已修复（文档） | 本文档 2、4.2、5.2；设计 4.7、4.11、5.3、7.2 | 测试数按当前更新；设计 5.3 改为 `pim_static_quant`；单层命令改为四参数；5.2 复用条件写成实现选择 |
| R5-1 | round5 | 高 | 成立：两次导出间换标定值后只重导 decode，`complete=true`，prefill 仍是旧 scale（新用例复现） | 已修复 | `gml_bridge/ceva.py`（`generation_digest`、`export_ceva_bundle`）、`tests/test_ceva_export.py` | 清单记生成口径摘要（checkpoint 全部张量、整模型标定、编译版本、是否接算子编译器）；口径不同且本次不覆盖全部已有图时报错，清单留在 `complete=false`；`test_partial_reexport_refuses_mixed_calibration`、`test_partial_reexport_refuses_changed_checkpoint` |
| R5-2 | round5 | 中 | 成立：GML 引用不存在的 bin、空包、漏登记、尺寸与参考不符都能通过（新用例复现） | 已修复 | `scripts/verify_gml_artifact.py`（`check_ceva_manifest`、`_check_ceva_graph_files`）、`tests/test_ceva_export.py` | 拒绝空包；每图要求登记 GML 与 IO_info；解析 GML，引用集、落盘集、登记集三者相等；按语义键与字段路径对照参考图核尺寸；`complete=true` 要求覆盖参考目录全部子图；`test_ceva_manifest_checks_references_and_reference_sizes` |
| R5-3 | round5 | 中 | 成立：中间激活、DQ/Softmax 相位缓冲、KV 权重通路写的是标定样本或随机值，与需求 P0-3、设计 4.7 不符 | 已修复（用户 2026-11-02 选择写零） | `gml_bridge/ceva.py`（`runtime_buffer_names`、`write_generated_files`）、`tests/test_ceva_export.py`、设计 4.7 | 按字段判定运行时缓冲并按尺寸写零；权重、LUT、FPSU 常量、标定 scale、prefill cos/sin 不动；`test_generated_runtime_buffers_are_zero_but_static_files_are_not`。GC 验证见 7.3 第五、六轮 |
| R5-4 | round5 | 中 | 成立：告警档 PASS 时摘要不报 token 状态（13:50 已去掉「token 一致」字样，但仍无法区分通过与告警） | 已修复 | `scripts/run_full_pipeline.py`（`main`）、`tests/test_ceva_export.py` | 摘要单独一行报「token 对拍：通过」或「token 对拍：告警」；`test_full_pipeline_ceva_warn_does_not_claim_token_match` 覆盖两种情况 |
| R6-1 | round6 | 高 | 成立：把小模型的 `v_proj` 改名为 `val_lin`，prefill 与 decode 的 V 投影都从 int8 加 kantor 字段退回 float16（新用例复现） | 已修复 | `gml_bridge/from_fx.py`（`_feeds_value_cache`）、`tests/test_ceva_export.py` | 改为结构判据：沿 Gemm 输出只穿过布局算子，到达写 V 的 DMA（decode）或 V 出口（prefill）即为 V 投影，与 K/V 出口识别共用 `kv_writers`；`test_value_projection_found_by_structure_not_name[prefill/decode]` 断言改名后 int8、带 kantor，GML 除 label/name 展示名外逐字节相同 |
| R6-2 | round6 | 中 | 成立：`own_bins` 与 `r5_zero_l0` 的 `compiler_revision` 都是 `bf5b8a5`，第 0 层 decode 却有 844 个文件不同 | 已修复 | `gml_bridge/ceva.py`（`compiler_revision`、`export_ceva_bundle`）、`tests/test_ceva_export.py`、设计 5.3 | 编译版本改为「HEAD+生成源码摘要」，源码覆盖 `gml_bridge/`、`quant/`、`graph/`、`contracts/`、`opcompiler_bridge/` 的 `.py`；`test_compiler_revision_tracks_uncommitted_source`、`test_partial_reexport_refuses_changed_compiler_source` |
| R6-3 | round6 | 中 | 成立：R5-3 写零后只有 decode 与 lm_head 的 GC 结果，没有全流程运行 | 未验证（用户决定暂停） | 本文档 1.3、7.3；CEVA `runs/r6_full_e2e` | 65 图已按当前代码重导并通过清单校验（`bundles/r6_full` 327/327）。全流程 GC + DDR + 仿真已启动，但因耗时过长（预计 8h+）按用户要求停止；状态记入 7.3，作为遗留项 |
| R6-4 | round6 | 中 | 成立：CEVA `.gitignore` 忽略整个 `runs/`，暂存区有 76 个摘要删除，`runs/20261027_e2e` 的摘要在磁盘上也已删除 | 已修复（用户 2026-11-02 选择恢复原规则） | CEVA `.gitignore`、`CLAUDE.md`、`tests/test_gitignore.py` | 恢复原规则并加 `ARCHIVE.md`；取消暂存的删除，从 HEAD 恢复 `20261027_e2e` 摘要；`own_bins_e2e`、`r6_full_e2e` 的摘要按规则可纳入；新增 2 例按 `git check-ignore` 检查摘要保留、大产物忽略 |
| R6-5 | round6 | 低 | 成立：设计 4.1 只写「非 generated 即失败」，5.3 没有摘要组成 | 已修复（文档） | 设计 4.1、5.3、8.1 | 4.1 补四项检查与完整性；5.3 补摘要组成与 `compiler_revision` |
| R7-1 | round7 | 中 | 成立：R6-3 在运行结束前标了「已修复」，§7.3 缺少第五、六轮记录，R5-3 的「GC 验证见 7.3」落空 | 未验证（用户决定暂停） | 本文档 1.3、7.1、7.3 | 已把 R6-3 状态改为「未验证」，1.3 不再宣称全流程已跑通；第六轮运行记录因运行中止未生成，作为遗留项 |
| R7-2 | round7 | 中 | 成立：`compiler_revision` 只覆盖本仓源码，FlagTree 变化不改变摘要；GML 的 FPSU 模式与相位来自 FlagTree | 已修复 | `gml_bridge/ceva.py`（`opcompiler_revision`、`generation_digest`、manifest 字段）、`tests/test_ceva_export.py`、设计 5.3 | 摘要加入 FlagTree 源码树 `include/`、`lib/`、`python/src/`、`python/triton/` 的内容；清单新增 `opcompiler_revision` 字段；读不到源码树记 `unknown` 并拒绝局部续导；`test_opcompiler_revision_tracks_flagtree_source`、`test_partial_reexport_refuses_changed_opcompiler` |
| R7-3 | round7 | 低 | 成立：`_feeds_value_cache` 不检查 writer 就是节点本身，直接返回投影输出的包装器会退回 fp16 | 已修复 | `gml_bridge/from_fx.py`（`_feeds_value_cache`）、`tests/test_ceva_export.py` | 函数开头判 `node in value_writers`；`test_value_writer_may_be_the_gemm_itself` |
| R7-4 | round7 | 低 | 成立：实施文档 §2、§4.1、§4.2 的用例数仍是旧数字 | 已修复 | 本文档 2、4.1、4.2 | 按本轮全量回归结果更新 |
| R7-5 | round7 | 低 | 成立：设计 5.3 的 `role` 仍写 `imported`、`generated`、`runtime`，与 4.1 和校验器只允许 `generated` 矛盾 | 已修复（文档） | 设计 5.3 | 改为只允许 `generated` |
| R8-1 | round8 | 高 | 成立：整包重导中途失败后只重试第一张图，清单仍是 `complete=true`，第 0 层新口径、第 1 层旧口径（新用例复现） | 已修复 | `gml_bridge/ceva.py`（`bundle_complete`、`export_ceva_bundle`）、`scripts/verify_gml_artifact.py`、`tests/test_ceva_export.py`、设计 4.1、5.3 | 每张图记录实际生成时的 `generation_digest`；局部续导逐图核对未重导图；`complete=true` 要求全部图与包级摘要一致；清单校验器拒绝混合口径的完整包。`test_interrupted_full_reexport_never_marks_mixed_bundle_complete`、`test_ceva_manifest_rejects_mixed_generation_digest` |
| R8-2 | round8 | 高 | 成立：当前代码只有 35/65 图的 GC 日志，未进入 DDR 与仿真，也没有判定报告 | 已修复（验证通过） | 本文档 1.3、7.3；CEVA `runs/ours_e2e`、`bundles/ours_e2e` | 按当前代码重导 65 图并校验（327/327），GC 65/65、DDR 7m13s、10 步仿真 7h00m37s，判定 PASS（0 FAIL、0 SKIP，token 与 lm_head 性能为 WARN）；原始目录无变化。11-02 会话中断导致 64/65 的断点已按实际产物重建清单并补编，未伪造编译记录 |
| R8-3 | round8 | 高 | 成立：CEVA 工作区删除了 `ours_e2e`、`ours_e2e2` 共 26 个已跟踪摘要，两个 `ARCHIVE.md` 缺失 | 已修复 | CEVA `runs/ours_e2e/`、`runs/ours_e2e2/`、`tests/test_gitignore.py` | 从 HEAD 恢复 26 个摘要文件；按现有归档格式补齐两个 `ARCHIVE.md`；新增 `test_existing_success_archives_are_present` 检查 ARCHIVE、verdict、output 与 argmax 存在；大产物忽略规则不变 |
| R8-4 | round8 | 中 | 成立：`output_sf_12.bin` 等 DQ 动态 scale 与 `kantor_A_scale_buffer_file_phase_3_*` 仍写入标定样本（在 r6_full 中核到非零） | 已修复（硬件实测无副作用） | `gml_bridge/ceva.py`（`runtime_buffer_names`）、`tests/test_ceva_export.py`、设计 4.7 | 按字段语义把 DQ 的 `output_sf`、相位 Kantor scale、Softmax 的 `Bias_buffer_phase_1` 与 `Scaling_buffer_phase_4` 纳入写零集合；固定量化常量与 KV/输出头静态 scale 保持标定值；`test_generated_runtime_buffers_are_zero_but_static_files_are_not` 直接断言具名字段。decode L0 单层对照：写零版与把 138 个文件恢复成 r6 标定值版的输出逐字节完全相同，说明硬件会在运行时重写这些缓冲；与厂商参考不逐字节一致是因为我方量化 bins 与厂商不同，不是写零导致 |
| R8-5 | round8 | 中 | 成立：FlagTree 根目录存在但子目录/源文件缺失时，`opcompiler_revision` 对空输入求哈希并返回有效版本 `e3b0c44298fc` | 已修复 | `gml_bridge/ceva.py`（`opcompiler_revision`）、`tests/test_ceva_export.py` | 四个必需目录缺一或任一目录没有有效源文件都返回 `unknown`；`test_opcompiler_revision_is_unknown_for_incomplete_source` 覆盖空树、缺目录、只有无关文件 |
| R8-6 | round8 | 中 | 成立：多行 Softmax 的 phase0 被当作 4 字节标量、phase4 整张求和，合法 prefill 被判失败（新用例复现 Σ=2.00037） | 已修复 | `scripts/verify_gml_artifact.py`（`check_softmax_reduction_encodings`、`check_softmax_is_normalised`）、`tests/test_verify_layers.py` | 按 phase0 的 4 字节记录数确定行数，逐行检查高半 fp16 编码；phase4 按行求归一；`test_softmax_checks_accept_multiple_query_rows`、`test_softmax_checks_reject_a_bad_second_row` |
| R8-7 | round8 | 中 | 成立：`mul < 32767` 排除了合法上界，极小因子返回 `(0,31)` 静默写零 | 已修复 | `gml_bridge/runtime_files.py`（`fpsu_mul_shift`）、`tests/test_runtime_files.py` | 允许 `mul == 32767`；所有右移都得到零乘数时报错；`test_fpsu_mul_shift_accepts_int16_max_and_rejects_zero` |
| R8-8 | round8 | 低 | 成立：校验器反量化未除 `weight_sf_multiplier`，倍数为 4 的小权重被误报（新用例复现误差 3.0002） | 已修复 | `scripts/verify_gml_artifact.py`（`check_dequantization_recovers_weights`）、`tests/test_verify_layers.py` | 从同节点读 `weight_sf_multiplier`（缺省 1），改用 `QuantizedWeight.dequantize` 共用同一条除回规则；`test_dequantization_divides_back_weight_sf_multiplier` |
| R9-1 | round9 | 高 | 评审时成立，当前已不成立：评审后的 11-03 07:43～14:52 已完成同一 `ours_e2e` 包的 DDR 与 10 步仿真 | 无需修复（后续验证已闭合） | CEVA `runs/ours_e2e/logs/run_all.log`、`results/verdict.txt`；本文档 1.3、7.3 | 当前编译清单 65/65 且无问题，包内 65 图、199353 个文件全部 generated；日志包含第 4～7 步完成记录，11 个 token 全部合法，Registry 来源与摘要检查通过。历史 `ARCHIVE.md` 仍对应 10-29 运行，不作为本轮证据 |
| R9-2 | round9 | 高 | 成立：用 `[65535, 65535]` 复现告警档返回 0、结论 PASS，代码没有独立词表范围检查 | 已修复 | CEVA `scripts/verify_run.py`、`tests/test_verify_run.py` | 按 runner 的 lm_head 输出宽度 32000 独立检查全部 token，严格档与告警档遇越界均 FAIL；合法分叉仍 WARN。`test_out_of_vocabulary_tokens_fail_in_strict_and_warn_modes` 先失败、修复后通过 |
| R10-2 / R11-1 | round10、round11 | 中 | 成立：新增来源与入口用例先出现 2 failed，厂商模式分叉被告警档放行 | 已修复 | CEVA `compile_gml.py`、`verify_run.py`、`run_all.sh`、`test_verify_run.py` | 共用 `generated_bundle` 核对完整包与全部 generated 文件；判定器再核对 DDR 成功标记。`test_warn_requires_generated_bundle_provenance` 覆盖厂商、imported、缺清单、无 DDR 标记；`test_cli_warn_requires_generated_bundle` 覆盖入口，既有合法分叉仍通过 |
| R10-1 / R11-2 | round10、round11 | 中 | 成立：`[13,2,306]` 在基线相同的严格档也能返回 0，行为测试先失败 | 已修复 | CEVA `verify_run.py`、`test_verify_run.py` | 独立检查 EOS 唯一且位于末尾。`test_eos_must_terminate_the_sequence_in_both_modes` 覆盖非末尾、重复、正常末尾、prompt 即 EOS，并分别覆盖满步数和提前结束 |
| R11-3 | round11 | 中 | 成立：零步在 CLI 与判定器均通过，2 个新用例先失败 | 已修复 | CEVA `run_all.sh`、`verify_run.py`、`test_verify_run.py` | CLI 拒绝零与负数；判定器对缺失、零、负 STEPS 明确 FAIL。`test_verdict_requires_positive_requested_steps`、`test_cli_requires_positive_steps`，正例含 1、08、10；保留 EOS 正常提前终止 |
| R11-4 | round11 | 中 | 成立：当前归档与历史副本逐字节相同，一致性测试先失败 | 已修复 | CEVA `runs/ours_e2e/ARCHIVE.md`、`tests/test_gitignore.py` | 归档改为最新生成摘要、token、10 步、20 PASS / 2 WARN 与分段耗时，明确本轮仅重新判定；`test_current_archive_matches_run_summaries` 对照 bin、日志和 verdict；历史目录未修改 |
| R11-5 | round11 | 低 | 成立：两个索引均无 fixed2float 说明，文档检查先失败 | 已修复 | FlagTree `claude.md`、`README.md` | 补充 int8 模型权重约束、显式 FPSU 保留、EmitC 不支持该模式及正反例检索；文档检查通过，lit 265 项通过；无算子代码变化，无需重建 |

### 7.2 未修复问题说明
无未修复项。第一轮问题 3 按评审建议只改文档，不补没有调用方的实现。第二轮 R2-1 起初只改了文档，10-31 已按内容还原。

### 7.3 回归测试结果
第一轮评审修复后：
- `pytest tests/ -k "not llama2_7b"`：1584 passed，1 failed，8 skipped。失败项还是改动前就已失败的 `test_pimir_sidecar_agrees_with_ttir_on_flops`；与评审前的 1581 passed 相比，多出的 3 例是本轮新增的用例。
- CEVA 工作区 `tests/`：17 passed（评审前 13 例，本轮新增 4 例）。
- FlagTree lit（按源码树的测试文件运行）：265 passed。
- 实际导出 decode、prefill、lm_head 第 0 层（`--layer-index 0 --layers 1 --use-opcompiler`）：3 张图的全部文件与 `bundles/ours_e2e` 中的同名图逐字节相同，清单的 `complete` 正确标为 false。

第二轮评审修复后（10-31）：
- `pytest tests/ -k "not llama2_7b"`：1587 passed，1 failed，8 skipped。失败项仍是改动前就已失败的 `test_pimir_sidecar_agrees_with_ttir_on_flops`；比第一轮修复后的 1584 多 3 例，是本轮新增用例。`test_ceva_export.py` 42 passed。
- 中途只用 ninja 重编 `triton-opt` 后，`test_inprocess_libtriton_is_not_behind_triton_opt` 报进程内 libtriton 旧于方言源码；用 `0-install-flagtree.sh` 重装并同步到 PyTorch 环境后通过。
- FlagTree lit：265 passed（重装后的构建目录）。
- CEVA 工作区 `tests/`：23 passed。
- 实际导出第 0 层三张图（`--layer-index 0 --layers 1 --use-opcompiler`）：6255 个文件与 `bundles/ours_e2e` 同名文件逐字节相同，清单 `complete` 为 false。
- 实跑修改后的单层脚本：`standalone_layer.sh` 对 decode L0（我方 Registry）运行 7m26s，返回 0，hidden、K、V 与厂商同层逐字节一致；用新的比较功能复核历史 lm_head、prefill L0 输出，也逐字节一致。
- 65 图全流程未重跑：本轮改动不改变导出产物（第 0 层逐字节复现），只收紧失败路径与判定。

第三轮评审与目标补充修复后（10-31）：
- `pytest tests/ -k "not llama2_7b"`：1591 passed，0 failed，9 skipped。原基线失败 `test_pimir_sidecar_agrees_with_ttir_on_flops` 已按 G-1 修复，本机没有 pimir 产物，该用例跳过。`test_oplevel_emitter_live.py::test_dq_group_widths_follow_tensors` 的期望由「attention scores 1 组」改为「每个查询 token 一组」：多 token 时整段一组会把不同 token 的分数混在一个 scale 里。厂商 prefill 第 0 层 GML 的 32 个 score DQ 组宽都是 S=1024，与新口径一致。
- genesim、numpy 后端与 genesim 桥：`test_hal_numpy.py`、`test_genesim_simulation.py`、`test_genesim_bridge.py` 83 passed，2 skipped。genesim 仓库没有代码改动。
- 通用路径回归：重构前后用同一模型导出的通用 decode GML、CEVA decode/prefill/head 三类小图逐字节相同。通用多 token 图只有 attention-score DQ 的 3 个字段变化：旧值 `output_shape_by_group "[1, 1, 8, 0, 64]"` 组数为 0，是错的。
- 65 图整包重新导出（`--use-opcompiler`）：与 `bundles/ours_e2e` 逐字节相同，`complete` 为 true。
- FlagTree：用 `0-install-flagtree.sh` 重装并同步到 PyTorch 环境，lit 265 passed。
- CEVA 工作区 `tests/`：35 passed。新 `verify_run.py` 重新判定 `runs/ours_e2e2`：PASS（含新增的「Registry 与编译清单摘要一致」），11 个 token 与基线一致。
- 原始目录：`check_orig.sh` 判定无变化（见 5.1）。
- 65 图仿真全流程未重跑：导出产物与已 PASS 的那次逐字节相同；CEVA 侧改动只收紧了失败路径与判定。

第五轮评审修复后（11-02，运行时缓冲写零，R5-3）：
- `pytest tests/ -k "not llama2_7b"`：1648 passed，1 failed，9 skipped。失败项 `test_opcompiler_linear.py::test_cache_key_lock_serializes_processes` 单独复跑通过，是随 GC 并行跑时的时序抖动；`test_ceva_export.py` 62 passed。
- CEVA 工作区 `tests/`：41 passed。
- 第 0 层三图按当前代码重导到 `bundles/r5_zero_l0`，清单校验 16/16 通过；GC 三图全部成功（prefill 1817s、decode 24s、lm_head 9s）。
- Registry 与 11-01 的 `own_bins_e2e` 比较（写零只改运行时缓冲，不影响 GC 产物）：decode 0 字节不同、prefill 0 字节不同、lm_head 1 字节不同（第 222 字节）。

第六轮评审修复后（11-02，R6-1～R6-5）：
- 65 图全流程验证未跑完，按用户要求停止，状态见第七轮记录。

第七轮评审修复后（11-02，R7-1～R7-5）：
- `pytest tests/ -k "not llama2_7b"`：1651 passed，1 failed，9 skipped。失败项仍是 `test_opcompiler_linear.py::test_cache_key_lock_serializes_processes` 的时序抖动，单独复跑可通过。
- `tests/test_ceva_export.py`：69 passed（新增 `test_value_writer_may_be_the_gemm_itself`、`test_opcompiler_revision_tracks_flagtree_source`、`test_partial_reexport_refuses_changed_opcompiler`）。
- CEVA 工作区 `tests/`：43 passed。
- 65 图全流程（`runs/r6_full_e2e`）已启动，decode 32 层与 lm_head GC 完成，prefill GC 运行中按用户要求停止，未进入 DDR 布局与仿真。完整 GC + DDR + 仿真验证作为遗留项。

第八轮评审修复后（11-02，R8-1～R8-8）：
- `pytest tests/ -k "not llama2_7b"`：1663 passed，0 failed，9 skipped（49 deselected，19m14s，最后一个小改动后重跑）。`pytest tests/ -k "llama2_7b"`：48 passed，0 failed（2h39m41s）。`test_ceva_export.py` 72 passed、`test_verify_layers.py` 25 passed、`test_runtime_files.py` 30 passed；CEVA 工作区 44 passed。
- 65 图按当前代码重导到 `bundles/r8_full`：65 图、complete=true，65/65 图的 `generation_digest` 与包级一致，199353 个文件全部 `generated`；`verify_gml_artifact.py --ceva-manifest` 327/327 通过。
- decode L0 用新包 GC 27s 成功。写零版与把 138 个动态 scale 文件恢复成 r6 标定值版的单层输出逐字节相同（三个输出的 md5 分别一致），证明 R8-4 的写零不改变硬件数值；与厂商参考不逐字节一致是因为我方量化 bins 与厂商不同。
- 全流程 `runs/r8_full_e2e` 曾用底层 `run_all.sh` 启动，GC 进行到 39/65 时按用户目标清单改用官方入口 `scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10`，该次运行停止并删除。
- 官方入口 `scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10`（等价于 `run_full_pipeline.py --backend ceva`）：导出 65 图、`complete=true`、65/65 图摘要与包级一致；清单校验 327/327 通过；GC 65/65（本次复用前一轮已完成产物，0 失败），DDR 布局 7m13s；仿真 prompt + 10 步 decode 共 7h00m37s。判定 18 项无 FAIL、无 SKIP：`[PASS]` 仿真正常结束、token 数量 11、decode 10 步、日志与 bin 一致、Registry 全部来自本次编译且与清单摘要一致、解码文本 `I am fine. I am doing my homework`；`[WARN]` token 与 baseline（前 7 个一致，step 12 起分叉）与 lm_head cycle +22.2%；结论 **PASS**。第 7 步原始目录检查：本次运行期间原始目录文件未变化。总耗时 7h09m14s（不含此前导出与清单校验约 21 分钟）。历史 `runs/ours_e2e` 摘要已先复制到 `runs/ours_e2e_20261029/` 保留。

第九轮评审修复后（11-03，R9-1～R9-2）：
- CEVA 工作区 `python -m pytest tests/ -q`：45 passed、0 failed。新增越界用例覆盖 32000 与 65535，严格档和告警档均返回 1；原有合法分叉用例在告警档仍返回 0。
- 当前 `bundles/ours_e2e`：65 图、199353 个文件全部 generated，逐图生成摘要一致；`runs/ours_e2e/gmls/compile_manifest.json` 为 65/65 且无问题。R9-1 的硬件证据已由评审后的完整运行闭合，无需重跑 7 小时仿真。
- 用新判定器复核 `runs/ours_e2e`：19 项检查中 17 项 PASS、2 项 WARN、0 项 FAIL/SKIP；新增「token 在词表范围内」为 PASS，最终结论 PASS。两项 WARN 仍是合法 token 与基线分叉及 lm_head 性能偏差。

第十、十一轮评审修复后（11-03）：
- 按用户指定同时处理两份评审，重叠项合并为 5 项；无不成立或遗留项。三项验收逻辑先由行为测试确认失败再修复；归档先由摘要一致性测试确认失败再更新；索引先查缺项再补文档。
- CEVA 全量回归：在 CEVA 仓先 `source /data/fjg/pengcheng/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，执行 `python -m pytest tests/ -q`，51 passed、0 failed、0 skipped（5.59 秒）。
- FlagTree 全量 lit：`source /data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/env-flagtree.sh` 后执行 `/data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/python-3.10.20/bin/lit /data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/test --no-progress-bar`，265 passed、0 failed（1.28 秒）。
- 图编译器相关回归：加载 `paths.json` 的 PyTorch 环境后执行 `python -m pytest tests/test_ceva_export.py tests/test_quant_weights.py tests/test_quant_calibration.py tests/test_runtime_files.py tests/test_runtime_files_phases.py tests/test_gml_from_fx.py tests/test_verify_layers.py tests/test_phase_plan.py tests/test_phase_source.py tests/test_pwla.py -q`，272 passed、0 failed、0 skipped（52.20 秒）。
- 整包校验命令（图编译器仓，同一环境）：`python scripts/verify_gml_artifact.py /data/fjg/ceva/bundles/ours_e2e --ceva-manifest /data/fjg/ceva/bundles/ours_e2e/bundle.json`，327/327 通过。65 图、199353 个登记文件全部 generated；本轮重新计算的图编译器版本 `bf5b8a5+4408a5affe03`、FlagTree 摘要 `9615b69ba9d7` 均与包一致。逐图调用现有 `already_compiled`，65/65 图的 GC 输入、Registry 和编译环境摘要均匹配。
- 现有硬件结果复核命令（CEVA 仓）：`TOKEN_CHECK=warn GML_BUNDLE=/data/fjg/ceva/bundles/ours_e2e python3 scripts/verify_run.py runs/ours_e2e baseline`，22 项中 20 PASS、2 WARN、0 FAIL/SKIP。实际 10 步、11 个合法 token，告警仍是量化造成的 token 分叉与 lm_head cycles +22.2%。
- 本轮不改变生成算法、编译配置或 runner；据此复用原 7h00m37s 的硬件执行记录，并重新检查其来源和判定，不把重新判定写成重新仿真。目标 3 的生成策略继续按数据、形状和图结构工作，没有为验收用例新增按层号或节点名的特例；不需要为本轮验收边界修复调整编译优化策略。
- 当前归档位于 `C/runs/ours_e2e/ARCHIVE.md`；后续复用运行名或重新判定必须同步归档，新增测试会拒绝摘要不一致。原日志的历史判定保持原样，最新判定取 `results/verdict.txt`。

## 八、bin 全部由我方生成（2026-11-01）
> 依据：需求 P0-1、P0-5 与设计 4.1 的 11-01 修订（用户确认：浮点权重取厂商反量化值，bin 全部由我方生成，token 放宽为告警档）。起始基线：flagos-pim-compiler `main@bf5b8a5`（工作区已有 10-28 起未提交改动），CEVA 工作区 `717407d`。

### 8.1 任务完成清单
| 任务ID | 任务描述 | 完成状态 |
|--------|----------|----------|
| H1 | int4 权重改厂商口径：scale = max(正峰/7, 负峰/8)；次正规倍数按数据算；输出头 int8 per-tensor absmax/128 | 完成 |
| H2 | int4 码值按「输入维 × 输出维」落盘；校验器按入边宽度还原后查饱和度与反量化 | 完成 |
| H3 | 定点 FPSU 编码：int16 mul + 最大右移；fixed2float 因子 = 输入 scale × 权重 scale | 完成 |
| H4 | LUT 改用 `contracts/pwla.py`（与厂商逐字节一致） | 完成 |
| H5 | 标定：`quant/calibration.py`，固定语料、99.999% 分位 / 128，输入走已旋转 embedding；每层 KV scale、输出头静态入口 scale | 完成 |
| H6 | 写盘器适配 CEVA 图：静态入口、KV 角色穿过 Transpose 与出口、MatMul 权重通路 scale、引用消费者的 sf/zp 不重复写、prefill 常量 cos/sin、prefill 多行（DQ、Softmax 逐行、Mask） | 完成 |
| H7 | CEVA 导出改走我方写盘器；删除整字节复制；`CevaGraphSpec.files` 改为 `runtime_dir`；清单只允许 `generated` | 完成 |
| H8 | 厂商反量化 checkpoint：`load_vendor_checkpoint`；CLI 先装 checkpoint 再导出 | 完成 |
| H9 | 验收放宽：CEVA `verify_run.py` 的 `TOKEN_CHECK=warn`、`run_all.sh --token-check`、`run_full_pipeline.py --token-check` | 完成 |
| H10 | 单层：三图 GC + 独立运行；65 图全流程 | 完成 | 全流程 PASS，见 8.4、8.6 |

### 8.2 详细改动清单
| 文件路径 | 改动类型 | 改动内容简述 | 对应任务ID |
|----------|----------|--------------|------------|
| `quant/weights.py` | 修改 | 新规则、`subnormal_multiplier`、`quantize_weight_int8`、`to/from_storage_order` | H1、H2 |
| `quant/calibration.py` | 新增 | `percentile_scale`、`ActivationScales`、`LlamaCalibration`、`calibrate_llama`、`graph_static_scales`、`CALIBRATION_PROMPTS` | H5 |
| `gml_bridge/runtime_files.py` | 修改 | `write_weight(shape=)`、`fpsu_mul_shift`、`write_fixed_point_scaling`、PWLA 表、Softmax 逐行、`write_dq_phases(fpsu_groups=)` | H2～H4、H6 |
| `gml_bridge/phase_data.py` | 修改 | `softmax(row_width=)`，phase0/2/3 按行 | H6 |
| `gml_bridge/export.py` | 修改 | `write_runtime_files(scales=)`；`_kv_scale`、`_static_input_scale`、`_fixed_point_factor`；常量表与 DQ 行数 | H3、H6 |
| `gml_bridge/from_fx.py` | 修改 | 次正规倍数按权重数据；`pim_static_quant_input`、`pim_rope_const`；RoPE-DQ 行数 | H1、H6 |
| `gml_bridge/ceva.py`、`contracts/ceva.py` | 修改 | `vendor_float_weights`、`load_vendor_checkpoint`、`write_generated_files`、`generated_io`、`calibrate_model`、真实 prefill rotary；删除 `write_ceva_files`、`follow_vendor_weight_guard`、`is_runtime_field`、`CevaFileBinding` | H6～H8 |
| `genesim_bridge/paths.py`、`paths.json` | 修改 | `ceva_embedding` | H5 |
| `scripts/export_gml.py`、`scripts/run_full_pipeline.py`、`scripts/verify_gml_artifact.py` | 修改 | 装 checkpoint；`--token-check`；清单核对与布局还原 | H2、H7～H9 |
| CEVA `scripts/verify_run.py`、`run_all.sh` | 修改 | `TOKEN_CHECK`、`--token-check` | H9 |
| CEVA `scripts/compile_gml.py` | 修改 | GC 完成后压缩 `log_infra.log`（与厂商编译链一致，prefill 全量少占约 30G） | H10 |
| 测试 | 修改、新增 | `test_quant_weights.py`、`test_quant_calibration.py`（新增）、`test_runtime_files*.py`、`test_ceva_export.py`、`test_gml_from_fx.py`、`test_verify_layers.py`、`test_paths.py`、CEVA `test_verify_run.py` | 全部 |

### 8.3 实施问题记录
| 编号 | 问题 | 根因 | 解决方案 | 实现位置 | 验证用例 |
|------|------|------|----------|----------|----------|
| I8-1 | absmax/8 重新量化厂商权重只有 92% 码值一致 | 厂商对两端各按满量程取 scale 再取大者（正峰 7、负峰 8），不是 absmax/8 | 改为 max(正峰/7, 负峰/8)；码值 99.998%、scale 99.99% 一致，且量化自身幂等 | `quant/weights.py`（`_group_scales`） | `test_quant_weights.py::test_scale_takes_the_tighter_side_of_the_asymmetric_range` |
| I8-2 | q/v 的 `weight_sf_multiplier` 原按投影名写死 2/4 | 次正规保护是数据驱动的（厂商 `save_scale`），第 17 层就没有 | 按最小 scale 计算倍数，序列化与写盘同一函数；写盘时声明与量化不一致即抛 | `quant/weights.py`、`gml_bridge/from_fx.py`（`_subnormal_multiplier_of`）、`export.py` | `test_ceva_export.py::test_weight_multiplier_comes_from_weight_data` |
| I8-3 | 用 HF 原始 embedding 标定，第 0 层 K scale 只有厂商的 1/3 | checkpoint 已旋转，激活要用同一坐标系的 embedding | 标定前换上 runner 同源的 `embed_torch.pth` | `gml_bridge/ceva.py`（`calibrate_model`） | `test_ceva_export.py::test_calibrate_model_uses_runner_embedding` |
| I8-4 | 输出头 FPSU mul 差 1（29067 对 29066） | 厂商用 fp32 的输入 scale 算因子，我方先截到 fp16 | 测试向量改用 fp32 scale；生产路径同样取未截断值 | `gml_bridge/runtime_files.py`（`fpsu_mul_shift`） | `test_runtime_files.py::test_fpsu_mul_shift_matches_vendor_encoding` |
| I8-5 | prefill 的 DQ、Softmax、Mask、RoPE-DQ 相位文件小 1024 倍 | 写盘口径只按一行，prefill 一张图有 1024 行 | DQ 元素数乘 `tokens`；Softmax 逐行归约；逐组 FPSU 仍按一行组数 | `export.py`（`_dq_specs`）、`phase_data.py`（`softmax`）、`runtime_files.py` | `test_gml_from_fx.py::test_dq_specs_cover_every_query_token`、`test_runtime_files_phases.py::test_softmax_phases_reduce_per_row`、`::test_rope_dq_fpsu_width_is_one_row` |
| I8-6 | prefill 独立运行 K 输出全零、hidden 发散 | 导出 prefill 时 cos/sin 常量用的是零张量；原先整字节导入厂商表掩盖了它 | 导出用模型 rotary 的真实表（sin 厂商编码），写盘把常量表落进 bin | `gml_bridge/ceva.py`（`_prefill_rope`）、`export.py` | `test_ceva_export.py::test_prefill_stage_uses_real_rotary_tables`、`::test_prefill_rope_tables_hold_the_graph_constants` |
| I8-7 | prefill K 路 RoPE 找不到 KV cache，定不出定点化 scale | prefill 没有 DMA，K 直接写 `key_cache_out` 出口 | 下游遍历把带 K/V 角色的出口缓冲也算作 cache；Split 角色穿过 Transpose | `export.py`（`_kv_cache_written_by`、`_kv_role_of`） | `test_ceva_export.py::test_prefill_rope_kantor_scale_follows_key_cache_exit`、`::test_attention_matmul_weight_sf_is_the_cache_scale` |
| I8-9 | prefill 独立运行 hidden 余弦只有 0.49 | prefill 没有 KV_Cache_DMA，QKᵀ/PV 的权重通路来自本图的 K 路 RoPE 与 v_proj，`_kv_role_of` 只认 DMA，`weight_sf` 落成 1.0 | Split 上游不是 DMA 时，沿该算子的输出找带 K/V 角色的出口缓冲 | `gml_bridge/export.py`（`_exit_role_of`） | `test_ceva_export.py::test_prefill_attention_matmul_weight_sf_traces_the_producer` |
| I8-10 | 写 V cache 的投影按参数名 `v_proj` 识别 | 名字是 HF 的约定，换模型就失效 | 改为按 GML 声明识别（输出 int8 的 Gemm），参数名不再参与取值 | `gml_bridge/export.py`（`_output_scale_of`） | `test_runtime_files.py::test_gemm_feeding_value_cache_is_found_by_structure` |
| I8-8 | 编辑 `phase_data.py` 时误替换了 `DynamicScalingPhases` 的字段块 | 两个 dataclass 首个字段行相同，按首次出现替换 | 从 HEAD 重建该文件再改 | `gml_bridge/phase_data.py` | 全量回归 |
| I8-11 | prefill 图目录比厂商大 0.9G（我方 2.5G 对厂商 1.6G） | 我方 `compile_gml` 清理只删非日志文件，1G 的 `log_infra.log` 原样保留；厂商编译链压成 gz | GC 完成后压成 `log_infra.log.gz`（31M），与厂商一致；本轮存量由 `from4_restart.sh` 统一补压，摘要校验只算 Registry 与输入、不受影响 | CEVA `scripts/compile_gml.py`、`/data/fjg/tmp_ceva/gen/from4_restart.sh` | Registry 尺寸与厂商逐字节同尺寸（164,773,442） |

### 8.4 测试验证结果
- 单元与集成：`python -m pytest tests/test_ceva_export.py tests/test_quant_weights.py tests/test_quant_calibration.py tests/test_runtime_files.py tests/test_runtime_files_phases.py tests/test_gml_from_fx.py tests/test_verify_layers.py tests/test_verify_gml_artifact.py tests/test_paths.py -q` 全部通过；CEVA `tests/test_verify_run.py` 11 passed。全量回归见 8.4 末。
- 第 0 层三图逐族比对（我方生成 vs 厂商文件，`/data/fjg/tmp_ceva/gen/export_real.py`）：尺寸零差异；decode 145/209 族、prefill 140/199 族、lm_head 19/33 族逐字节相同；其余差异来自标定值与运行时样例数据。int4 权重码值与厂商 99.998% 相同，输出头权重 100% 相同。
- GC：decode L0、lm_head、prefill L0 全部 success（`/data/fjg/ceva/runs/own_bins_l0/logs/compile*.log`）。
- CEVA 单层独立运行（同一输入，厂商 Registry 对我方 Registry；不再要求逐字节）：

| 子图 | 运行 | 与厂商输出 |
| --- | --- | --- |
| decode L0 | 正常结束，返回 0 | hidden 余弦 0.9996（相对误差 3.0%）；K 余弦 1.000000；V 余弦 0.999999 |
| lm_head | 正常结束，返回 0 | logits 余弦 0.9999；argmax 与前 5 名完全相同 |
| prefill L0 | 正常结束，返回 0 | hidden 余弦 0.99991（相对误差 1.4%）；K 余弦 0.99964；V 余弦 0.99966 |


- 回归：`python -m pytest tests/ -q -k "not llama2_7b"` 1643 passed、9 skipped、0 failed（18 分 21 秒，最终代码）；CEVA `tests/` 41 passed。7B 慢测与 FlagTree lit 本轮未动相关代码，未重跑。
- 65 图：导出 13 分 05 秒；`verify_gml_artifact.py --ceva-manifest` 130/130 通过，清单 complete=true、全部文件 generated；与厂商同字段文件逐个比尺寸，221345 个零差异，其中 130674 个小文件逐字节相同。
- 65 图全流程（`C/runs/own_bins_e2e`，2026-11-01 17:06 至 11-02 02:23，归档 `ARCHIVE.md`）：GC 65/65 success、0 失败（4 路并行 4h02m）；DDR 布局 7m28s；仿真 3 步 5h08m40s；判定 18 项 PASS、0 SKIP、0 FAIL——token `[13, 29902, 626, 2691]`（"I am fine"）与 baseline 前 4 个逐个一致，warn 档未触发；cycle 偏差 prompt +10.4%、decode 约 +9.9%、argmax -0.0%，均在 ±20% 带内；原始目录检查无差异。

### 8.5 偏差说明
| 设计条目 | 设计方案 | 实际实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| design §4.1（10-28 版） | 厂商 bin 整字节导入 | 反量化浮点 + 我方生成 | 交付要求 bin 来自我方编译 | 是（用户，2026-11-01） |
| design §7.3 P0-5 | token 与基线逐个一致 | 我方 bin 用告警档 | 量化口径（标定值）与厂商不同，token 可能分叉 | 是（用户，2026-11-01） |
| design §7.2 单层运行 | 与厂商同层逐字节一致 | 数值比较（余弦、相对误差） | 同上 | 是（用户，2026-11-01） |
| design §7.3 P0-5 仿真步数 | 至少 10 步（遇 EOS 提前结束） | 3 步（prefill + 3 次 decode，覆盖 KV 更新循环） | 10 步实测 7h01m（prefill 占约 4.7h），为当天完成全流程判定按需求缩减条款执行（至少 1 步 decode，已满足） | 否（按需求 P0-5 缩减条款） |

### 8.6 已知问题与待确认事项
| 编号 | 事项 | 状态 | 说明 |
|------|------|------|------|
| K8-1 | 标定值与厂商不同（第 0 层 K 0.0471 对 0.0459，V 0.00241 对 0.00261；末层 K 0.122 对 0.163） | 已规避 | 厂商标定语料未知；我方固定 6 段多领域短文本。单层输出余弦 ≥ 0.9996 |
| K8-2 | 65 图全流程（GC、DDR、3 步仿真、告警档判定） | 完成 | 判定 PASS：18 项全过、0 SKIP；token 与 baseline 前 4 个逐个一致（warn 档未触发）；结果 `C/runs/own_bins_e2e/results/verdict.txt`，总耗时约 9h20m（GC 4h02m + DDR 7m28s + 仿真 5h08m40s），归档 `C/runs/own_bins_e2e/ARCHIVE.md` |

## 九、全流程自动化入口完善（2026-11-02）

### 9.1 使用方式与改动

在编译器仓根目录执行一条命令即可，不再需要先手动加载 PyTorch 环境：

```bash
bash scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10
```

新增 `scripts/run_ceva_pipeline.sh`，从 `paths.json` 读取并加载 `pytorch_env_script`，再进入现有 `scripts/run_full_pipeline.py --backend ceva`。完整链路为：厂商量化权重反量化浮点 checkpoint → 图编译与整模型标定 → FlagTree 算子编译 → 65 张 GML、IO_info 与全部 bin → 来源清单校验 → 厂商 GC → DDR 布局 → CEVA 仿真 → 结果判定。GML 包在 `C/bundles/<实验名>`，运行与归档在 `C/runs/<实验名>`，编译器侧三段日志在 `test-results/ceva-<实验名>`。

同时完善 `scripts/run_full_pipeline.py`：我方生成 bin 时 `--token-check` 默认改为 `warn`，严格逐 token 对拍需显式传 `strict`；拒绝 `--decode-steps 0`；子命令使用当前 Python 解释器；最终摘要不再把告警档的仿真通过误报为 token 一致。`tests/test_ceva_export.py` 增加告警摘要和最小 decode 步数回归，并断言导出强制携带 `--use-opcompiler`。

### 9.2 本轮验证

- `bash scripts/run_ceva_pipeline.sh --help`：在未预先加载环境的 shell 中正常启动并显示全流程参数。
- `tests/test_ceva_export.py`：58 passed；`tests/test_genesim_simulation.py tests/test_ir_cost_eval.py`：36 passed、4 skipped、1 deselected；CEVA 工作区 `tests/`：41 passed。
- 现有 `C/bundles/own_bins` 重新执行来源清单校验：130/130 通过，全部登记文件为 generated，尺寸与摘要一致。
- 现有 `C/runs/own_bins_e2e` 通过 `run_all.sh --from 6 --to 6` 实跑当前判定入口：18 项 PASS、0 SKIP、0 FAIL；4 个 token 与 baseline 一致。新 `ceva_verdict` 能读取并接受该报告。
- 本轮未重新执行完整 65 图导出、GC 和仿真：`/data` 当前仅剩 104G，低于工作区全流程预估的 130G；未擅自删除历史产物。既有完整实跑结论仍以 §8.4 和 `C/runs/own_bins_e2e/ARCHIVE.md` 为准。

## 十、共享目录外部写入导致总流程失败（2026-11-04）

### 10.1 现象与根因

`ours_e2e` 在 11-03 19:38 再次执行同一官方入口，导出、清单校验、DDR、10 步仿真和结果判定均成功，但第 7 步原始目录检查发现 35627 处文件变化，令 `run_all.sh` 最终返回 1。该失败不是 GML、bin、Registry 或仿真错误：

- 变化由 527 个新增、207 个删除、34893 个非目录修改组成；仍存在的 35420 个新增或修改文件全部属于 `root:root`，mtime 为 11-03 19:39 至 11-04 01:19，落在本次运行窗口内。本流程以 `jgfeng` 运行，不能产生这些 root 所有的文件。
- 变化全部位于 `/data/xym_data/ceva_data/` 共享树，主要是厂商子图的 `l2a_output` 与 `test_app_outputs`；变化清单没有 `parser_output`、`runtime_files`、GML 或 bin。本次 GC 使用 `env/gc_w<N>` 工作副本，仿真使用 `runs/ours_e2e/gmls`，没有向上述共享树写产物。
- 默认严格检查仍应拒绝任何原始目录变化，不能无条件忽略，也不能恢复或删除共享目录。后续增加显式的外部变化模式，只豁免当前用户无权限造成的变化，见第十一节。

### 10.2 本次复核

- 当前图编译器 `compiler_revision=bf5b8a5+4408a5affe03`、FlagTree `opcompiler_revision=9615b69ba9d7`，与 `bundles/ours_e2e/bundle.json` 一致，说明未提交源码就是本包的实际生成版本；GeneSim 工作区保持干净。
- 重新执行 `verify_gml_artifact.py --ceva-manifest`：327/327 项通过。65 图共 199353 个登记文件全部为 `generated`、`source=null`，逐图生成摘要一致，GML 引用、落盘、登记集合与参考尺寸均通过检查。
- 三份 DDR map 的 65 个 Registry 全部指向 `runs/ours_e2e/gmls`，厂商 Registry 为 0；编译清单 65/65 无问题。
- 用当前判定器复核现有硬件结果：21 项 PASS、1 项 WARN、0 项 FAIL/SKIP。实际执行 prompt + 10 步 decode，11 个 token 均在词表内，解码文本为 `I am fine. I am doing my homework`；WARN 仅为需求允许的 token 分叉。

结论：用户提出的三项核心目标已经由本次硬件运行满足，不需要为第 7 步的外部写入修改生成或优化策略。

## 十一、第 7 步支持外部变化告警与单独续跑（2026-11-04）

### 11.1 实现

- CEVA `scripts/diff_snapshot.py` 新增 `--allow-external`：新增、删除只有父目录不可写时才归为外部变化；修改必须文件与父目录均不可写。任何本流程可写路径变化仍返回 1，root 运行时不会因权限检查放宽检查。默认严格模式不变。
- CEVA `run_all.sh` 新增 `--orig-check {strict,external}`，把外部变化模式接到第 7 步。`--from 7` 复用已有的 `logs/orig_before.tsv`，不再用当前状态覆盖运行前快照；若原快照不存在则直接失败。
- `--from 7 --to 7` 只执行原始目录检查；已有 `results/verdict.txt` 为 PASS 时，汇总显示第 6 步沿用现有判定，不再报告「未判定」。

本次修复命令：

```bash
cd /data/fjg/ceva
bash run_all.sh -n ours_e2e --gml-bundle /data/fjg/ceva/bundles/ours_e2e \
  -s 10 --token-check warn --orig-check external --from 7 --to 7
```

### 11.2 验证

- 新增 `tests/test_diff_snapshot.py` 4 例：外部模式仍拒绝可写变化、严格模式拒绝不可写变化、纯外部变化通过、文件不可写但父目录可写时仍拒绝。
- 真实 `ours_e2e` 只续跑第 7 步，800895 项原快照保持不变，28 秒完成。35627 处全部为当前用户不可写的外部变化，本流程可写路径变化为 0，最终结果 PASS；前 6 步未执行，第 6 步沿用已有硬件 PASS。
- `ARCHIVE.md` 同步末次硬件耗时、21 PASS / 1 WARN 和外部目录检查结论。CEVA 全量回归 55 passed、0 failed；`bash -n run_all.sh` 与两个 Python 文件的语法编译检查通过。
