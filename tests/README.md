# tests/ 测试指南与排错手册

本目录共 102 个测试文件、1589 条用例（其中 4 条 `slow` 默认不跑，需 `-m slow`）。本文按「依赖 → 分组 → 失败症状 → 排查入口」组织，目标是**测试报错后 1 分钟内知道该看哪个模块、哪份产物**。

> 先记住一件事：每次 pytest 结束都会把逐用例状态、耗时和失败首行写到
> **`test-results/pytest-latest.md`**（由 `conftest.py` 生成）。排错从这里开始。

---

## 1. 环境与常用命令

所有命令都必须在**仓库根目录**执行（部分用例扫描源码或使用相对路径，换目录会误报）。

```bash
# 每个新 shell 先激活 torch/transformers 环境，否则 import torch 失败
source "$(python3 -c 'import json; print(json.load(open("paths.local.json"))["pytorch_env_script"])')"

# 默认全集：跳过 slow
python -m pytest tests/ -x -q

# 单个文件 / 单个用例
python -m pytest tests/test_partition.py -x -q
python -m pytest tests/test_partition.py::test_attention_runs_on_the_device -x -q

# 按关键字筛选（定位同一主题的一组用例）
python -m pytest tests/ -q -k "raw or waw"

# 只看 skip 原因（依赖没配时会大量 SKIP，不是失败）
python -m pytest tests/ -q -rs

# 需要可读中间产物时关掉输出捕获
python -m pytest tests/test_spec_prop_llama2_7b.py -q -s

# 显式跑慢速用例（GeneSim 端到端仿真，十几分钟）
python -m pytest tests/test_genesim_simulation.py -q -m slow
```

`pytest.ini` 里只有一条全局约定：`addopts = -m "not slow"`。`slow` 标记目前只用在 `test_genesim_simulation.py::test_simulation_produces_a_summary`。

---

## 2. 失败后 60 秒定位流程

1. **先看状态**：`FAIL` 是断言不符；`ERROR` 多是 fixture/导入/路径问题（例如缺参考产物、缺模型）；`SKIPPED` 通常只是站点路径没配。
2. **打开 `test-results/pytest-latest.md`**，找到失败用例的「失败详情」首行。断言消息本身通常已经指出出错字段、`文件:行号` 或第一个不匹配的节点。
3. **按第 4 节的文件索引**找到该测试文件，看「验证内容」和「失败优先查」。
4. **单条复现**：`python -m pytest <file>::<test> -x -q -s`，必要时在断言前打印对应产物。
5. **区分两类失败**：
   - *契约/真源漂移*（字段名、常量、键集合、结构规则）——按报错改契约或同步调用方；
   - *数值/产物不符*（logits、KV、字节数、IR 文本）——先确认输入与布局，再查 kernel/规划。
6. **排除环境与缓存**：见第 5、6 节。

---

## 3. 外部依赖矩阵

`paths.local.json` 是站点路径唯一入口（环境变量优先级最高）。缺依赖时多数测试走 `skipif`，但**加粗的几项没有跳过保护，缺了会直接 ERROR**。

| 依赖 | 配置键 / 环境变量 | 使用它的测试 | 缺失表现 |
| --- | --- | --- | --- |
| torch/transformers 环境 | `pytorch_env_script` | 全部 | 导入即报错 |
| 真实 Llama-2-7B（fp16，约 13 GB） | `llama2_7b_model_dir` | 所有 `*_llama2_7b.py`、`test_phase_source.py`、`test_gml_depends_on_opcompiler.py`、`test_oplevel_emitter_live.py` | 一般 SKIP；`test_gml_depends_on_opcompiler` 有 1 条会 ERROR |
| ResNet50 GML 参考产物 | `gml_reference_dir` | `test_gml_reference.py`、`test_gml_names.py` | SKIP |
| llama2 decode-block 参考产物 | `gml_llama2_reference_dir` | `test_runtime_files_phases.py`、`test_gml_llama2_reference.py`、`test_gml_quant.py`、`test_gml_hw_table.py`、`test_gml_coverage.py`、**`test_gml_lut.py`**、**`test_phase_data.py`**、**`test_verify_layers.py`** | 前四个 SKIP；**后三个直接 ERROR** |
| 本机最近一次 decode-block 导出图 | `gml_ours_graph`（当前未配置） | `test_gml_node_parity.py` | SKIP（整文件） |
| FlagTree 源码树 | `flagtree_prefix` 下的源码目录 | `test_flagtree_ods_hygiene.py`、`test_genesim_bridge.py`（ODS 卫生）、`test_fusion_contract.py`（与 C++ 对齐） | SKIP |
| `triton-opt` + PIM pass | `flagtree_prefix` | `test_opcompiler_ops/linear.py`、`test_pimir_layout.py`、`test_phase_plan_live.py`、`test_three_path_generation.py`、`test_ir_cost_eval.py`、`test_genesim_bridge.py`（部分） | 对应文件/用例 SKIP；提示重跑 `0-install-flagtree.sh` |
| GeneSim 仓库与模型 IR | `genesim_root` | `test_genesim_simulation.py`、`test_genesim_bridge.py`（真实模型段）、`test_ir_cost_eval.py`（往返）、`test_placement_export.py`（ISA 解码一条） | 多数 SKIP；部分因 `required=True` 直接 ERROR |
| vLLM + CUDA | 系统环境 | `test_vllm_executor.py`、`test_vllm_compat.py` | 导入失败 / 用例失败 |
| networkx | 系统环境 | `test_verify_gml_artifact.py`（一条 `importorskip`） | 单条 SKIP |
| `flag_gems` | 系统环境 | `test_genesim_bridge.py::test_autotune_drops_tiles_over_the_wram_budget` | 单条 SKIP |
| 本机 `gcc` | 系统环境 | 所有真编译内核的用例 | `ToolchainUnavailable` |

**当前机器**（2026-10-23）：`llama2_7b_model_dir`、`flagtree_prefix`、`genesim_root`、`gml_reference_dir`、`gml_llama2_reference_dir` 均已配置；`gml_ours_graph` 未配置，因此 `test_gml_node_parity.py` 整体 SKIP。PIM pass、vllm、networkx、flag_gems 均可用。

### 3.1 未标 slow 但很重的测试

默认集里仍会实际加载 7B 权重或反复真编译，单跑很慢：

- 真实 7B：`test_executor_llama2_7b`、`test_strategy_llama2_7b`、`test_spec_prop_llama2_7b`、`test_comm_llama2_7b`、`test_kv_layout_llama2_7b`、`test_mem_planner_llama2_7b`、`test_natural_prompt_llama2_7b`、`test_concurrency_llama2_7b`、`test_opcompiler_e2e_llama2_7b`、`test_oplevel_emitter_live`、`test_phase_source`（live 段）、`test_gml_depends_on_opcompiler`（live 段）。
- 大量真编译：`test_opcompiler_ops.py`（59 条全部 `force=True`）、`test_opcompiler_linear.py`、`test_pimir_layout.py`（约 34 次 `triton-opt` 子进程）、`test_three_path_generation.py`（18 个算子各编一次）。

只改某个模块时，优先单文件或 `-k` 定向跑。

---

## 4. 测试文件索引（按模块分组）

图例：**7B** = 需要真实 Llama-2-7B；**REF** = 需要 `gml_reference`；**LREF** = 需要 `gml_llama2_reference`；**PASS** = 需要带 PIM pass 的 `triton-opt`；**GS** = 需要 GeneSim。

### 4.1 测试基础设施与站点路径

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `conftest.py` | — | 梯度开关逐用例复位；会话结束写 `test-results/pytest-latest.md` | 无 | 报告没生成 → conftest 未加载 / rootdir 不对；后续 `torch.export` 图异常先怀疑梯度泄漏 |
| `test_pytest_report.py` | 2 | 报告覆盖旧文件、PASS/FAIL/NOT RUN 标记、耗时列 | 无 | `conftest.py` 的 `pytest_runtest_logreport` 或报告表格格式 |
| `test_paths.py` | 11 | 路径优先级 env > `paths.local.json` > `paths.json`；FlagTree backend 发现；全仓禁止硬编码本机绝对路径 | 无 | `genesim_bridge/paths.py`；卫生检查会直接列出违规文件行号 |
| `test_diff_prepare_out.py` | 4 | 对拍器命名归一：节点号通配、槽/相/段/头号保留 | 无 | `scripts/diff_prepare_out.py::_normalise_naming_value` |

### 4.2 契约与「单一真源」

这一组多为静态扫描/一致性断言，几乎零耗时，但重构后最容易红。

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_unified_ir_contract.py` | 15 | `node.meta` 键登记表、裸字符串禁止、载荷类型存在、阶段协议可达 | 无 | `contracts/unified_ir.py::META_KEYS`；报错会列出「仅在用未登记 / 仅登记未在用」差集 |
| `test_op_semantics.py` | 12 | `OP_SEMANTICS` 四份派生视图与旧字面量逐项等价、登记表无重名/无重复 aten 映射 | 无 | `contracts/op_semantics.py`；新增算子只改登记表 |
| `test_op_contract.py` | 8 | `PIMHardwareConfig` 往返与取值校验、`OpCompileRequest` 必须带 hardware | 无 | `contracts/op_contract.py`；错误信息应含字段名 |
| `test_dtype_carrier.py` | 15 | dtype 单一真源 `contracts/dtypes.py`，quant 布局合法性，跨维一致性在 `propagate_specs` 出口执行 | 小模型 | `contracts/dtypes.py`、`graph/spec_prop.py` 出口校验 |
| `test_mem_layout.py` | 16 | 显式步幅、`align_up` 全仓唯一实现、`elem_strides/align_bytes` 有真实生产方 | 无 | 常见是又复制了一份 `align_up`，报错直接给 `文件:行号` |
| `test_stride_parity.py` | 39 | 排布规则搬家后数值不变、`orchestrator/` 不再保留重复实现 | 无 | `contracts/mem_layout.py` 公式；`orchestrator/` 是否重新写了本地实现 |
| `test_no_bypass.py` | 8 | 生产代码不得绕过 `dtype_bytes` 自行推导 dtype/字节宽 | 无 | 报错给出违规行号；白名单只允许 `graph/spec_prop.py` 与两个函数级例外 |
| `test_three_path_coverage.py` | 3 | 每个算子都同时有 numpy 对拍、GeneSim mnemonic、GML op_type 三条路径（`paged_attn` 只有 numpy 一条，按归属表免除另两条） | 无（扫源码） | 报 `{op: ["genesim","gml"]}`；看 `contracts/op_semantics.OP_SEMANTICS`、`_GENESIM_COVERED_BY`/`_NO_GML` 与两个 opcompiler 测试文件 |
| `test_flagtree_ods_hygiene.py` | 7 | 本仓 emitter 字段名/模块属性名与 FlagTree ODS/C++ 逐项对齐，无孤儿属性 | FlagTree 源码 | 字段顺序看 `contracts/mlir_layout.py` 与 `PIMAttrDefs.td`；`-rs` 看是否因源码缺失而 SKIP |

### 4.3 图切分、策略、传播

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_partition.py` | 14 | 「默认下设备 + 黑名单留主机」打标、DPU 连通分量编号、规则表/内核表正反向闭合 | 小模型 | 新算子没进 `RULE_TABLE` 或 `_KERNELS`，断言会列出算子名 |
| `test_strategy.py` | 25 | TP/PP/混合的 DPU↔stage↔layer 映射、非法参数、`llama_strategies` 枚举 | 无 | `graph/strategy.py`；断言匹配中文错误文案，改文案会误红 |
| `test_strategy_sweep.py` | 12 | 小 Llama 在全部合法策略下 token/KV/通信边数/权重字节不变量 | 小模型 | token 对但 KV 错 → 多 DPU 同写一行；边数不符 → `spec_prop` 位置编码复制规则 |
| `test_strategy_llama2_7b.py` | 10 | **7B** 三种代表策略的 token 与 KV 对拍、内存容量 | 7B | 先看断言里的 `layer/DPU/head` 与 `max diff`；`_run_strategy_cached` 可复用一次运行 |
| `test_spec_prop.py` | 27 | `propagate_specs`：Placement/shard_map、四类 redistribute 边、附录 A 手推对拍、幂等 | 小模型 | 附录 A 逐字段失败先 `format_spec_report`；拆头后 matmul/softmax 缺规则 |
| `test_spec_prop_llama2_7b.py` | 6 | **7B** 8 DPU 纯 TP 的节点规格、Megatron 列-行配对、logits all_gather | 7B | `inv_freq` 断言受「同进程是否先跑过 compile_op」影响，属已知耦合；`-s` 打印报告 |
| `test_split_heads.py` | 13 | SDPA 逐头展开结构、attention scale 落点、数值等价、GML 字段比例 | 小模型 | 最强判据是 `test_expansion_preserves_numerics`；字段计数不符查 `graph/split_heads.py` 与 `gml_bridge/from_fx.py` |
| `test_op_classify.py` | 3 | 评估旁支代表 IR 的形状口径（SiLU/FFN、Concat 32 头、Softmax 逐头） | 无 | `genesim_bridge/op_classify.py` 的形状公式 |

### 4.4 融合与量化

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_fuse.py` | 10 | 通用「主算子 + 激活/池化」折叠：该折的折、有第二读者不折、单激活 | 小模型（1 条） | `graph/fuse.py`、`contracts/fusion_contract.py`；负向用例误折查单消费者判定 |
| `test_fuse_pim.py` | 10 | RMSNorm 六合一、Gemm+SiLU、attention 定标吸收；折叠后数值不变 | 小模型 | `rms_norm_chain_ops_are_gone` 红 = 标记了但没删中间节点；数值红先查 mul 操作数顺序 |
| `test_fuse_rope.py` | 5 | RoPE 六算子链折叠 + 完整流水线 GML op_type 计数 | 小模型 | pass 顺序是硬约束；计数不符先查 `quant_pass` 与 `from_fx` 归属 |
| `test_fusion_contract.py` | 14 | 融合表唯一真源（`is` 同一性）、旧表已删、与 FlagTree `isFusionTarget` 对齐 | FlagTree 源码（3 条） | 有人把表复制回 `fuse.py`；FlagTree 正则不符需同步测试 |
| `test_quant_pass.py` | 12 | DQ 插入判据（每个激活矩阵乘上游、共享源共用一条、attention scores 整条一组）及 GML 4 相字段/bin | 小模型 | 插入位置看 `report.inserted/skipped` 与 `DQ_META_KEY`；字节数看 `artifact.dq_specs` |
| `test_quant_weights.py` | 10 | int4 权重量化值域、每组一 scale、误差、不可整除抛错 | 无 | `quant/weights.py::quantize_weight`；误差红多因分组轴取错 |
| `test_quant_activations.py` | 16 | int8 激活量化、absmax/128、饱和率、与 `phase_data.dynamic_scaling` 逐字节一致 | 无 | 不一致先对齐 `gml_bridge/phase_data.py`（fp16 截断位置） |
| `test_calib_data.py` | 17 | 标定常数形状/dtype/统计量、角色取数、KV scale | 无 | `gml_bridge/calib_data.py`；角色映射退化会回落到 `activation_for` |
| `test_layer_fields.py` | 22 | 24 类层字段编排：键集合、命名来源、mask 单/双输入、RoPE 段号、槽位宽度 | 无 | `orchestrator/layer_fields.py`；先打印 `fields` 与 `render_layer_txt` |

### 4.5 通信、内存与 KV

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_comm_plan.py` | 15 | all_reduce/all_gather/all_to_all/scatter/local_slice 的段展开、DMA 批序列、成本模型 | 无 | `comm/plan.py`；乱序分片看 `global_range`，多维按行展开 |
| `test_comm_lowering.py` | 9 | 通信原语在 `DmaEngine` 上的数值结果、非零 `mram_offset`、host staging | 无 | 先怀疑地址是否漏加基址、dtype、host 中转顺序 |
| `test_comm_llama2_7b.py` | 8 | **7B** 通信计划结构（2×层数条 all_reduce、logits all_gather、scatter）与数值 | 7B | 结构不符查 `propagate_specs` 与 `llama_shard_config`；数值容差已放宽到 fp16 口径 |
| `test_kv_layout.py` | 12 | 槽几何与 `kv_off` 精确偏移、块表补齐/越界、head 归属、注意力数值 | 无 | 偏移按手算整数核对；块表乱了查 `build_kv_layout` / `write_block_table` |
| `test_kv_layout_llama2_7b.py` | 3 | **7B** KV 分片跟随 k/v_proj placement，槽池 16 MiB/DPU 精确布局 | 7B | `graph/spec_prop.py` 的 k_proj 切分规则或 `align` |
| `test_kv_slots.py` | 9 | 槽几何与块表缓冲、默认池容量、块表乱序/重写补 -1、块号越界、池耗尽/复用 | 无 | `memory/kv_layout.py::build_kv_layout`、`runtime/executor.py` 的槽池 |
| `test_mem_planner.py` | 18 | 三区 offset、跨图权重共享、激活贪心复用不别名、内核 tile 计入容量 | 无 | 最危险是「同 step 读写别名」；打印 `plan.weight/act_prefill/act_decode/total` |
| `test_mem_planner_llama2_7b.py` | 9 | **7B** 两图内存规划、容量、地址可用、落地偏移贯通到通信 `dst_addr` | 7B | `dst_addr` 不符 = mem planner 回填与 `comm.plan` 的 itemsize/基址口径不一致 |
| `test_slot_isolation.py` | 3 | 两请求物理槽隔离、共享块表重写切换、`pos`/`num_tokens` 端到端透传 | 无 | 块表串了查 `_write_block_table`，透传断了查 `execute_plan`/`bind_inputs` |

### 4.6 后端、内核、执行计划与执行器

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_dpu_sdk.py` | 17 | 厂商 SDK numpy 镜像：DPU 枚举、地址空间隔离、broadcast/prepare/push 契约 | 无 | 先区分语义回归与错误文案变化（大量 `pytest.raises(match=...)`） |
| `test_hal_numpy.py` | 20 | 异步 HAL：submit/wait/query、同 DPU 串行、DMA 快照、tasklet hazard | 无 | 时序类用例在慢机器上可能抖动；buffer 互抢查 host 路径是否加锁 |
| `test_kernels.py` | 20 | 白名单内核数值、tasklet 行切分、编译内核路由、多块乱序块表的按块注意力、融合 `pim.paged_attn`（单遍在线 softmax）与主机相位链两遍在 f16 末位相符、块表装不下历史时报错 | 部分需 `triton-opt` + gcc | `mask/alias/view` 红先查工具链与 `.opcompiler_cache`；路由红查 `runtime/kernels.py`；注意力红查 `_iter_kv_blocks` 是否仍按块表查槽；`paged_attn` 红先看 `record_route("paged_attn", …)` 走的是 hit 还是 fallback |
| `test_exec_plan_gen.py` | 27 | 命令间 RAW/WAR/WAW、redistribute 落地、硬件参数、NumpyBackend 数值 | 小模型（1 条） | WAR 用例 `resolved==0` = 键不匹配导致 WAR 被丢；数值红先核对权重写入偏移 |
| `test_runtime_compiled_coverage.py` | 24 | 哪些内核真的走到算子编译器（打桩 + 一次真编译）、编译集合清单、退回计数 | 一次真编译 | 集合断言失败需同步文档与实现；`fallback` 计数查 `runtime/kernels.py` 的 route 记录 |
| `test_runtime_files.py` | 23 | `.bin` 字节宽度/命名、与 GML 引用交叉校验、零填充分类 | 无 | 「引用了没写盘」看文件名生成侧；「写了没引用」看 `contracts/gml_names.py` |
| `test_runtime_files_phases.py` | 18 | 新增 bin 族与 **LREF** 逐文件字节对拍 | LREF | 字节数不符查 group_size/dtype；非平局差异 = scale 算错而非舍入 |
| `test_executor_llama2_7b.py` | 2 | **7B** 整图 prefill logits 与贪心 token 对齐单卡 PyTorch | 7B | 依次查权重分片写入、SDPA KV 偏移、kernel 镜像 |
| `test_concurrency_llama2_7b.py` | 2 | **7B** 独立 launch 时间重叠、有 waits 的 launch 真等待 | 7B | 串行化查 DPU 线程池/异步 HAL；打印命令 `id/waits/op` |
| `test_multi_request_decode.py` | 4 | 多请求并发解码与单路一致、`slot_tokens=2` 与整段一致、槽复用、容量溢出报错 | 小模型 | 块表映射错或共享缓冲未按请求重写 |
| `test_activation_windows.py` | 7 | 每请求激活窗口的统一 `act_stride`、窗口不重叠、超预算报错 | 无 | 步长算错查 `memory/mem_planner.fit_activation_windows` |
| `test_exec_binding.py` | 6 | `bind_plan` 只平移激活地址与块表基址、窗口 0 返回原计划、`mram_offset` 不动 | 无 | 内核缓存失效先查是否误平移了 `mram_offset` |
| `test_parallel_requests.py` | 8 | 两/四请求并发的 token 与串行一致、混合 prefill/decode、确定性 | 小模型 | 激活窗口或块表窗口越界先查 `slot_index` 传递 |
| `test_parallel_overlap.py` | 1 | 同 DPU 上不同请求的命令真交错，允许后发先至 | 小模型 | 泳道是否仍按 `(lane_key, dpu_id)` 分开 |
| `test_natural_prompt_llama2_7b.py` | 1 | **7B** 自然语言 prompt 的逐 token 生成与 HF `generate` 一致 | 7B | 失败会打印 `route_counts()`；出现主机退回说明某算子未注册 |
| `test_vllm_executor.py` | 2 | vLLM 引擎经 `PIMExecutor` 跑两个并发请求（含被拆成 3+1 的分块 prefill）、同 seed 复现与 logprobs 回传，且槽池默认值多留一块 | vLLM + CUDA | 环境变量必须在 `import vllm` 前设置；worker 崩溃看 `backend/vllm_worker.py` |
| `test_vllm_compat.py` | 2 | 版本锁定：0.14.x 放行、0.31.0 报错且消息同时含当前版本与支持范围 | vLLM | 升级 vLLM 先看 `backend/vllm_compat.py` 与 `docs/vllm-031-gap-20261102.md` |
| `test_vllm_capabilities.py` | 18 | 启动期/请求期能力矩阵：TP、LoRA、投机、量化、dtype、非 llama、分块预算超图长逐项报错且点名参数 | 无 | 报错文案变了先看 `backend/vllm_worker.py` 的 `_unsupported` |
| `test_vllm_sampling.py` | 16 | 主机侧采样：温度、top-k/top-p、三种惩罚、seed、logprobs 与未实现参数报错 | 无 | `backend/sampling.py`；先查 `SamplingConfig.from_vllm` 的字段映射 |
| `test_vllm_block_table.py` | 7 | 块号以 vLLM 为唯一真源：新请求采用、cached 追加、resumed 替换、来源二选一、外部块号不改变 token | 无（1 条小模型） | 块表错乱先查 `KVBlockTable.adopt` 与 `_sync_cached` |
| `test_vllm_translate.py` | 7 | `execute_model` 翻译层：新请求注册与块号、分块 prefill 返回空 token、按窗口分 wave 且保序、结束清理、抢占重放、解码步多 token 报错、语法约束在跑之前拒绝 | vLLM | 桩掉 `_step` 后只测翻译逻辑；先看 `backend/vllm_worker.py::execute_model` |
| `test_prefix_cache.py` | 4 | 前缀缓存：命中段只读（共享块字节前后相同）、从命中长度为起点写、三请求共享前缀两窗口、vLLM 引擎命中数与关闭缓存时逐 token 一致 | 小模型 + vLLM | 共享块被改写先查 `_add_request` 的 `valid_len` 初值；`int()` 报错查 `_sync_cached` 是否按 KV 分组取 `[0]` |
| `test_preemption.py` | 6 | 抢占恢复：prefill 中途 / decode 中途被抢占整段重放、命中共享前缀从 `num_computed_tokens` 续算、真实 vLLM 池 3 块强制抢占后与不抢占逐 token 一致 | 小模型 + vLLM | 恢复后位置错先查 `_sync_cached` 的重置条件与 `RequestSlot.reset` 的 `computed`；没触发抢占就调小 `PIM_KV_POOL_SLOTS` |
| `test_serve_pim_preflight.py` | 11 | `--pim-*` 参数翻译与启动预检：槽池、MRAM、prefill 图长、分块预算、问题一次性列全；不设槽池时跳过 MRAM 估算 | 无 | 预检规则与 `scripts/serve_pim.py` 的 `preflight` 必须同步 |
| `test_auto_capacity.py` | 5 | KV 槽池与并行窗口自动定容：剩余 MRAM 求容量、超预算时报所需与可用、显式值装得下则保留、不设任何 `PIM_*` 环境变量时 vLLM 分到的块数等于定容结果 | 小模型 + vLLM | 槽数偏小先查 `memory.mem_planner.fit_pool_slots` 是否漏了内核 tile 或块表；块数对不上查 `determine_available_memory` |
| `test_prefill_buckets.py` | 21 | prefill 分桶：桶表排序、按最短可容纳原则选桶、跨桶 token 一致、各桶共享权重与 KV 布局、融合 vs 强制主机 | 小模型 | 桶几何查 `compile_llama2(prefill_seq_lens=)`；选桶查 `prefill_bucket` |
| `test_chunked_prefill.py` | 12（2 条 `slow`） | 分块 prefill：各种切法与整段逐 token 一致、主机两遍路径同样一致、交错请求互不干扰、中间段不采样、3000 token 的 64/512 切法（slow） | 小模型（slow 用 3072 长度图，约 8 分钟/用例） | 位置错先查 `prefill_chunk` 的 `pos`/`position_ids` 与 `_host_paged_attention` 的绝对位置掩码 |
| `test_profile_execution.py` | 6 | vLLM 调度采集：效率/并发/窗口冲突、空步不计均值、中途失败不串步、stop 还原方法 | 无 | 失败先看 `scripts/profile_execution.py` 的 `_VllmProfiler` 步号过滤；并发类用例靠线程，慢机器上慎调阈值 |
| `test_architectures.py` | 5 | 架构白名单：未知 `model_type` 列出受支持项、`rope_scaling`/`sliding_window` 点名拒绝、GQA（4 Q 头 / 2 KV 头）的 KV 布局与两请求 token 一致 | 小模型（GQA 用例） | 白名单改动看 `backend/architectures.py`；GQA 切分失败先查 `llama_strategy` 的 KV 头整除约束 |
| `test_step_stats.py` | 4 | 每步指标：wave 取整与最宽值、共用块去重占用、日志字段、真实 vLLM 下 `PIM_LOG_STEPS` 有输出且 `PIM_PROFILE` 两套采集器都有数据 | 小模型 + vLLM | 日志没有输出先查 `PIM_LOG_STEPS` 是否生效；采不到步查 `profile_vllm` 的 `_step` 签名是否跟 worker 对齐 |

### 4.7 GML 导出、格式与校验

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_gml_writer.py` | 14 | 文本序列化：重复键、缩进、嵌套块、端口 ≥10 的键名怪癖 | 无 | `gml_bridge/writer.py`；用 `scripts/gml_structure_check.py` 定位规则 |
| `test_gml_sharding.py` | 8 | 切分只通过形状体现，GML 中禁止设备/分片字段 | 无 | `gml_bridge/sharding.py`；GQA 用 kv 宽度而非 hidden |
| `test_gml_names.py` | 8 | 命名规则与真实 `.bin` 双向一致（1168 个文件） | REF | 差集非空 = 命名规则漏族或参考换版 |
| `test_gml_reference.py` | 8 | 用 ResNet50 参考产物校验五条结构规则与字段清单本身 | REF | 参考上就红说明校验器读错，查 `scripts/gml_structure_check.py` |
| `test_gml_llama2_reference.py` | 12 | 用 llama2 decode-block 实物钉住规则、量化特征、算子缺口、边形状分布 | LREF | `DEMO` 字段是实物证据；`operator_coverage_gap` 红 = 映射表或文档未同步 |
| `test_gml_hw_table.py` | 52 | 硬件字段由 `(op_type, phase)` 唯一决定，与参考逐格对拍 | LREF（无跳过保护） | 报错带「表里 X，实物 Y」，改 `contracts/gml_hw_table.py` |
| `test_gml_lut.py` | 20 | LUT 288 字节布局、倒数表切线精度、与参考字节对拍 | LREF（部分无保护） | 布局红查 `contracts/gml_lut.py` 打包顺序；硬件对拍注意第 31 段为空段 |
| `test_gml_quant.py` | 14 | 量化格式声明与参考交叉验证（int4 一字节一值、group=128、LUT=288B 等） | LREF | 参考版本（v1/v2）或 `contracts/gml_quant.py` 常量 |
| `test_gml_coverage.py` | 10 | 四张字段族声明表与参考/我方产物三方一致 | LREF + 小模型 | 未声明族先加到 `contracts/gml_coverage.py`；按算子缺族看 `absent` 列表 |
| `test_gml_export.py` | 63 | 导出入口：GML ↔ `.bin`/`IO_info.txt` 逐值自洽、scale/标定/槽位/rank | 小模型（1 条需 LREF + `gml_ours_graph`） | 先打开 `tmp_path` 下 `relay2gml_graph.gml`、`.bin`、`IO_info.txt`；分派逻辑在 `gml_bridge/export.py` |
| `test_gml_from_fx.py` | 49 | fx 图 → GML：算子映射、融合残留、dtype/Convert/Gather、decode 裁剪、1..N 重编号 | 小模型 | 结构规则、`A` 别名、`input_count` 恒等式、`idx` 端口一致性 |
| `test_gml_node_parity.py` | 1 | 我方 decode-block 与参考逐节点逐键对拍 | LREF + `gml_ours_graph` | 当前 `gml_ours_graph` 未配置 → SKIP；报错形如 `(op_type, 角色)[序号] 缺 [字段]` |
| `test_export_gml_cli.py` | 10 | `scripts/export_gml.py` 文件名、结构自检、落盘交叉校验、CLI 默认值 | 小模型 | `_check_structure` 与 `write_runtime_files`；参考配置用环境变量模拟 |
| `test_verify_gml_artifact.py` | 13 | 校验器自身有分辨力（每条检查都有反例）、networkx 可解析 | 小模型 | 反例失败 = 检查退化成恒真；正例失败看 `Report.summary()` 的 `✗` 行 |
| `test_verify_layers.py` | 21 | 身份/连接/残差、DQ 相位数学、Softmax 编码、归一化、零点、分组饱和度 | LREF（无跳过保护） | 正例看 `report.failed`；`-30.75` 相关是刻意设置的哨兵 |
| `test_gml_depends_on_opcompiler.py` | 10 | 相位字段真由算子编译器决定，且接入前后 GML 逐字节相同 | 7B + PASS | 「相同」失败 = 静态表与 pass 漂移；「依赖」失败 = 相位仍从常量表取 |

### 4.8 算子编译器与相位

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_opcompiler_ops.py` | 59 | B 路整算子：IR→EmitC→C→`.so`，ctypes 跑起来与 numpy 镜像逐元素对拍；ABI；路由不退回镜像 | PASS + gcc | 先看 `.opcompiler_cache/<key>.pimir.mlir`、`.so`、`.meta`；数值红再按算子定位 |
| `test_opcompiler_linear.py` | 32 | A 路 linear：数值、分块规则、编译指纹、跨进程锁、别名契约 | PASS（1 条需 CUDA） | 数值/分块/缓存指纹；`test_k_alias` 的「不一致」是设计信号 |
| `test_opcompiler_e2e_llama2_7b.py` | 4 | **7B** 三策略 decode 的 token 与 HF 完全一致，编译内核无 fallback | 7B + PASS | 先看打印的逐形状误差表与 `route_counts()`；monkeypatch 必须在 `register_all` 之前 |
| `test_oplevel_emitter.py` | 11 | FX 图 → 整算子 pim MLIR 的形状口径、分派、相位数、跳过规则 | 无 | 打印 `report.text`；`report.skipped` 多 = 形状或 DQ spec 不符 |
| `test_oplevel_emitter_live.py` | 10 | 真 7B 图的算子计数、相位、执行单元、flops、槽位边形状 | 7B + `triton-opt` | `triton-opt` stderr 已拼进断言；计数不符先看 lm_head 是否裁剪 |
| `test_phase_plan.py` | 17 | 固化 MLIR 文本 → `PhasePlan`：相位数/kind/unit/bytes 与静态表同口径 | 无 | 夹具可能过期；用 `test_phase_plan_live.py` 重跑真 pass |
| `test_phase_plan_live.py` | 4 | 真跑 `triton-opt -pim-expand-phases` 验证展开行为 | `triton-opt` | `plan.count/units` 不符；手跑 `triton-opt x.mlir -pim-expand-phases` 看 stderr |
| `test_phase_source.py` | 14 | pass 顺序、`cross_check` 双向对拍、真实图端到端（live 段） | 部分 7B + PASS | `cross_check` 失败打印 `PhaseMismatch`（算子值 vs 静态值） |
| `test_phase_data.py` | 60 | DQ 四相 / Softmax 五相公式与参考逐字节对拍及数学自洽 | LREF（无跳过保护） | 失败先看对应节点 `*_phase_*.bin`；Softmax p1–p4 与参考本就不同，勿误判 |
| `test_pimir_layout.py` | 97 | 四维决策写进 pimir：`#pim.tasklet_tiled`、`#pim.placement`、DMA 地址/步幅、回传消费 | PASS | 先打印 `compile_op(..., force=True).pimir`；shard 编码丢失查 A 路 `flatten_shard_dim` |
| `test_three_path_generation.py` | 20 | 18 个算子真实产出可加载 `.so` + GeneSim IR + GML op_type | PASS + gcc | 相对误差 >0.05 按算子名定位；重编入口 `opcompiler_bridge/driver.py::compile_op` |

> 已知缺口：`test_gml_depends_on_opcompiler.py::test_changing_an_opcompiler_phase_value_changes_gml` 缺少 `@requires_live`。未配置 7B 时它会 **ERROR**（`未配置站点路径 llama2_7b_model_dir`）而不是 SKIP，属测试自身问题。

### 4.9 GeneSim 桥接、放置导出与编排器

| 文件 | 用例 | 验证内容 | 依赖 | 失败优先查 |
| --- | ---: | --- | --- | --- |
| `test_genesim_bridge.py` | 52 | TTIR/PIMIR 的 flops、DMA/WRAM、本地分片、系数拟合、mnemonic 覆盖、ODS 卫生 | 部分 PASS / GS | `genesim_bridge/ir_cost.py`、`cost_extractor.py`、`op_classify.py`；`pass 可能没生效` 先验 PIM pass；`进程内 libtriton 比方言源码旧` = 重建 FlagTree 后没同步 `.so`，跑 `0-install-flagtree.sh` |
| `test_genesim_simulation.py` | 6 | GeneSim 端到端慢速冒烟 + 流水线参数/IR-sidecar 契约 | GS（1 条 slow） | 无 `summary.json` = GeneSim 未装或仿真崩溃；slow 用例会写真实 `genesim/results/` |
| `test_ir_cost.py` | 1 | `#pim.phase_spec<... bytes=N>` 计入 `mram_traffic_bytes` | 无 | `genesim_bridge/ir_cost.py` 的 phase_spec 解析 |
| `test_ir_cost_eval.py` | 30 | 融合成本可核算、导出时被吸收节点清零、ttir/pimir 口径一致、IR 可被 GeneSim 读回 | 部分 PASS / GPU / GS | 看 `out.ir`、成本 sidecar、放置 sidecar 的 `fused_activations/absorbed_ops` |
| `test_placement_export.py` | 40 | 放置结果写进 GeneSim IR/sidecar：分片宽度、collectives、B 路 pimir、trace 来源、KV 区 | 部分 PASS / GS | A 组先看 sidecar `shards.local_in/out_features` 与 `semantic_role`；B 组看 `.opcompiler_cache` |
| `test_layout_feedback.py` | 12 | 算子编译器布局决策回传给成本模型，且必须改变产物；无死字段 | 无 | `contracts/ir_payloads.LayoutFeedback`、`genesim_bridge/ir_cost.py`、`runtime/compile.peak_kernel_mram_bytes` |
| `test_orchestrator.py` | 33 | 层展开（422 层闭合）、Layer ID、L2 复用、`net.ini`/层 txt 渲染（CRLF） | 无 | `orchestrator/layer_expand.py`、`layer_id.py`、`l2_alloc.py`、`layer_render.py` |

---

## 5. 常见症状 → 排查路径

| 症状 / 报错片段 | 最可能原因 | 处理 |
| --- | --- | --- |
| 大量 `SKIPPED`，0 失败 | 站点路径或工具链未配置 | `-rs` 看原因；`python -c "from genesim_bridge.paths import describe; print(describe())"` |
| `未配置站点路径 <name>` | 对应 `paths.local.json` 键为空，或测试用了 `required=True` | 填 `paths.local.json` 或设同名环境变量；确认不是配置为 `None` |
| `ToolchainUnavailable` / `gcc 编译生成的 C 失败` | `triton-opt`/`mlir-translate`/`gcc` 缺失或缓存脏 | 检查 `flagtree_prefix`；清 `.opcompiler_cache`；重跑 `0-install-flagtree.sh` |
| `AssertionError: pass 可能没生效` | A 路产物缺 `pim.dma_*`，或 triton-opt 版本不对 | 验 `triton-opt --help | grep pim`；`genesim_bridge.env.assert_pim_passes_available()` |
| `这些算子退回了主机: [...]` | 某形状不支持编译（如 K<16），或路由判定与真实分片形状不符 | 看 `runtime.kernels.route_counts()`；查 `_compiled_linear_supports` |
| `GML 引用了但没写盘` / `写了没引用` | 命名规则与写盘侧不一致 | `contracts/gml_names.py` 与 `gml_bridge/runtime_files.py` |
| 结构规则 `规则 N ...` | GML 结构被破坏（独立激活、悬空 buffer、形状写在节点上等） | `python scripts/gml_structure_check.py <file.gml>` |
| 数值 `max|Δ|` 超阈值 | 权重写入偏移、KV 布局、kernel 镜像或 dtype 落点 | 先跑单策略/单节点；用 `-s` 打印 plan 与各 DPU 分片 |
| `内存超限` / 区域重叠 | 三区规划或内核 tile 未计入预算 | `memory/mem_planner.py`；打印 `plan.weight/kv_base/act_base/total` |
| `要求图已达到阶段 ...` | 跳过了前置 pass（partition/spec/fuse） | 查调用链是否漏了 `mark_stage` 对应的 pass |
| `断裂/重叠`、`nbytes` 不符 | 通信段全局区间有缺口，或字节数与 shape×dtype 不符 | `comm/plan.py`；打印 `format_comm_plan(...)` |
| 报告文件没生成 | 不是用仓库根目录跑，或 `conftest.py` 未被加载 | 从仓库根运行；检查 rootdir |
| 同一用例单独跑通过、整集失败 | 前序用例污染（梯度开关、进程级 aten 替换、编译缓存） | 先单跑；必要时在目标用例前加 `--forked` 或调整顺序验证。`inv_freq` 断言已知受 `compile_op` 影响 |

---

## 6. 调试技巧

### 6.1 先看可读产物，不要靠猜

项目里的关键结构都提供了 `format_*`/`__str__`，失败时优先打印：

```python
from graph.spec_prop import format_spec_report
from comm.plan import format_comm_plan
from memory.mem_planner import format_mem_plan
from graph.strategy import format_strategy
```

- 切分/通信：`format_spec_report`、`format_comm_plan`
- 内存：`format_mem_plan`
- 策略：`format_strategy`
- GML：`artifact.text`、`relay2gml_graph.gml`、`IO_info.txt`
- 算子编译器：`.opcompiler_cache/<key>.pimir.mlir`（展开后 IR）、`.so`、`.meta`

### 6.2 路由与退回主机

`runtime/kernels.py` 会记录每个算子走编译内核还是 numpy 镜像：

```python
from runtime import kernels as km
km.reset_route_counts()
# ... 跑用例 ...
print(km.route_counts())   # {("linear","hit"): ..., ("softmax","fallback"): ...}
```

出现 `fallback` 通常意味着工具链缺失或形状判定不接受该分片。

### 6.3 编译缓存

- 默认目录：仓库根 `.opcompiler_cache/`
- 可覆盖：`OPCOMPILER_CACHE_DIR`、`OPCOMPILER_TRITON_OPT`
- 怀疑缓存污染时：用 `compile_op(..., force=True)` 重编，或清空缓存目录后重跑
- 缓存键包含编译器指纹（kernel 源码 + libtriton）；改了 emitter 仍命中旧结果，先查指纹测试

### 6.4 pytest 小技巧

```bash
python -m pytest tests/ -q --maxfail=1          # 第一个失败就停
python -m pytest tests/test_x.py -q -k "not slow"  # 文件内排除慢用例
python -m pytest tests/test_x.py -q -rs         # 列出跳过原因
python -m pytest tests/test_x.py -q --tb=long   # 完整回溯
```

---

## 7. 维护约定（改代码前先看）

1. **测试与代码同一次改动完成**，不留下「测试稍后补」。
2. **不要删除断言来让测试变绿**：大多数断言是防退化守卫（结构规则、单一真源、无旁路、幂等）。要么修实现，要么同步契约并说明原因。
3. **跨文件复用的 helper 是隐式契约**，改名前先全局搜索：
   - `tests/test_partition.py::_export_random_llama`、`_FixedMaskLlama` — 被 spec_prop/dtype/placement/fuse/gml 等多组复用；
   - `tests/test_spec_prop.py::_appendix_a_graph/_appendix_a_config` — 被 exec_plan_gen/mem_planner/mem_layout 复用；
   - `tests/test_mem_planner.py::_two_appendix_a_graphs/_kv_specs` — 被 mem_layout 复用；`tests/kv_helpers.py` 的 `write_row/read_rows` 被 KV 相关用例复用；
   - `tests/test_comm_plan.py` 的 spec/edge 工厂 — 被 comm_lowering 复用；
   - `tests/test_placement_export.py::annotated_tiny_llama` — 被 spec_prop 的拆头用例复用。
4. **新增算子要同步三处**：`contracts/op_semantics.OP_SEMANTICS`、`graph/partition.RULE_TABLE`、`runtime/kernels._KERNELS`；并补 `test_three_path_coverage` 要求的三路覆盖（numpy 对拍、GeneSim mnemonic、GML op_type）。只有运行时一侧调用的算子（如 `paged_attn`）不进 `RULE_TABLE`，在 `_GENESIM_COVERED_BY` / `_NO_GML` 里写明归属。新增 `has_kernel` 算子还要在 `test_genesim_bridge` 补成本计费，否则 flops 按 0 计。
5. **基线表是刻意的**：`test_stride_parity` 的 `FROZEN_*`、`test_op_semantics` 的 `OLD_*`、`test_verify_layers` 的 `-30.75` 哨兵，改动数值前先确认是「有意变更」而非「顺手重算」。
6. **依赖外部资源的测试必须用 `skipif(required=False)`**；不要用 `@pytest.mark.slow` 掩盖普通依赖。
7. **`assert_node_matches_ref.py` 目前是 0 字节空文件**。CLAUDE.md 提到的「逐节点对拍器」实际由各测试内的辅助函数承担（如 `test_three_path_generation._assert_matches_numpy`、`test_opcompiler_ops._call`、`test_gml_export._nodes_by_op`）。新增逐节点对拍逻辑时，建议放在具体测试文件内，避免再造一个空壳。
