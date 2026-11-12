# 代码评审报告：Qwen3-MoE 接入与图编译—算子编译—GeneSim 全推理链路验收
> 文档编号：review-qwen3-moe-20261109
> 创建日期：2026-11-09
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联实施文档：implement-qwen3-moe-20261105

## 一、评审概述
　　本轮覆盖 `genesim cdf6c99..434e147`、`flagos-pim-compiler 3ce60b6..daef534`、`FlagTree 262adb3b7..6e34f0299` 及三仓工作区改动；三仓评审输入工作区均无代码改动（仅本报告为新增文档），行号按当前工作区定位。共发现 10 项问题（3 严重、5 高、2 中），本轮新增的「图编译 → FlagTree → GeneSim」链路、MoE 几何与代价、数值一致性验收均不能关闭，原需求中的 8 GiB 容量和动态分派加速验收也缺少有效证据。
　　实际验证结果如下；所有临时配置、探针和产物均在 `/tmp`，未重建 FlagTree，未运行外部 CEVA 仿真。
- `python scripts/model_parser.py --model_name /data2/pengcheng-models/Qwen3-30B-A3B --output /tmp/qwen3-review-30b.ir`：成功生成 10276 个算子，但仅有 336 个 GEMM，FFN 被写成 6144 宽稠密结构；`parameter_bytes=5.642 GiB`，未表达 54 GiB 专家权重。
- 设置 `QWEN3_30B_MODEL_DIR=/data2/pengcheng-models/Qwen3-30B-A3B` 后运行 `python scripts/run_full_pipeline.py --num-stages 4 --skip-simulation`（GeneSim、日志和算子缓存重定向到 `/tmp`）：退出码为 0，但实际执行的是 Llama2-7B。骨架与精化 IR 的 3780 个编号一致；sidecar 含 224 个 GEMM 和 3554 个算子级条目，`requires_pimir=true`，路径与哈希全部有效，实测分块为 512；无 GPU 时成本精化走 oplevel。该结果只证明旧链路，不是 Qwen3 证据。
- 用现有 FlagTree 安装运行 `triton-opt test/Dialect/TritonPIM/paged_attn_phases.mlir -split-input-file -pim-expand-phases -pim-verify-gml-contract | FileCheck ...`：通过；未重建源码树。
- `cd genesim && ./run.sh --test sim`：44/44 个测试文件通过，但仓内没有 Qwen3/MoE 用例。另用一层 Qwen3-MoE 配置和声明 `requires_pimir=true` 的空 sidecar 实测，`require_compiler_pimir=true` 下仍生成 24 条模板 trace 并成功结束；编译器 preflight 对同一产物则正确拒绝 48 个缺项。
- 真实权重验证使用 `/data2/pengcheng-work/Qwen3-30B-A3B-4L`，在 `/tmp` 按真实 MoE 几何局部关闭分块：独立恢复 HF 原始稀疏前向后，PIM 与 HF 均生成 `[6722, 4849]`，compile 133.2 秒、PIM 26.8 秒。目标命令 `python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q` 在同一临时辅助下为 **17 passed、1 skipped（239.57 秒）**；`python -m pytest tests/ -q -k "not llama2_7b"` 在相同真实模型局部辅助、GeneSim 根目录隔离到 `/tmp` 的条件下为 **1859 passed、9 skipped、52 deselected（1536.00 秒）**。
- 默认分块探针仅跑 4 层即从 7079 节点膨胀到 1390344 节点，优化 103.9 秒、峰值内存 24.29 GiB；48 层的 16684128 节点只是按层数线性外推，未作为实测结论。既有 `/data2/pengcheng-work/run_full_30b.log` 的 48 层结果为 `[12095, 13]`、compile 19290.6 秒、PIM 1205.1 秒；对应代码早于日志且未再改变，故未重跑全模型，但其 HF 参考同样经过静态化，且关闭分块来自外部 monkeypatch，证据边界列入问题 5、6。
- 8B 权重在本机不存在，目标用例结果为 skip，记为未覆盖而非通过。已核实的正确契约链为：`config.head_dim` → `runtime/compile.py:292-305` → 小模型及 4 层原始 HF 对拍；`moe_intermediate_size=768` → `backend/architectures.py:88-97` → 真实几何得到 `tp4_pp2`。历史 K3 修复位于通用重分布发射函数，`test_moe_dispatch.py`、`test_moe_ep.py`、`test_strategy.py` 实测 **38 passed（46.11 秒）**；I4 只修复动态跳过命令，静态稠密路径的同根因复发列入问题 3。

## 二、修改建议
### 问题1（严重）: GeneSim误建稠密MoE，导致几何代价失真
　　**问题说明**：本轮补充需求要求 48 层、128 专家、top-8、专家宽度 768 的图骨架及与执行语义一致的驻留和通信代价；design §4.2 也明确 6144 是配置残留。`genesim/src/ir/model_ir.py:1092-1132` 虽读取 `model_type`，却不按 `qwen3_moe` 分派，只取 `intermediate_size`；后续 FFN 构造（同文件 1990-2187）因此生成单个 dense FFN。实测 30B IR 每层只有 q/k/v/o 与 gate/up/down 七个 GEMM，没有 router、专家或 top-k，且只登记 5.642 GiB 驻留权重；一层 MoE IR 的 48 个非边界节点还能被 refine 正常桥接，说明错误几何会继续被下游消费而非被拦截。top-8 的每 token 激活 FLOPs 虽恰与 dense 6144 相同，但 128 专家驻留权重是其 16 倍；若选择全专家稠密模拟，计算量也必须是其 16 倍，不能用同一 dense 近似冒充两种语义。
　　**根本原因**：通用 Llama 图构造器把模型几何、执行语义和成本字段写死在一起，`ModelIR` 数据结构（266-291 行）也没有专家数、top-k、专家宽度或执行语义，下游无从校验和恢复配置，测试则没有真实 MoE 不变量。相同假设还使 parser 按原始 bf16 配置生成四节点注意力，而编译器按 fp16 执行走融合条件，跨段执行语义未对齐。
　　**涉及文件**：`genesim/src/ir/model_ir.py:266-291,1077-1132,1990-2187,2295-2310`；`genesim/scripts/model_parser.py:163-177`；`genesim/tests/sim/test_model_ir.py`、`genesim/tests/sim/test_model_parser.py`（缺少 MoE 覆盖）。
　　**修改思路**：先在 design 与 `ModelIR` 中补齐本轮 GeneSim 成本契约，记录并校验专家数、top-k、专家宽度和全专家稠密/top-k 动态分派语义，以及对应的路由、驻留、通信口径，再由这些元数据统一驱动图生成、精化与调度；不支持的语义必须显式报错，禁止退回 dense 模板。用真实 30B 配置断言专家数、top-k、768 宽度、每层专家权重 1207959552 字节、48 层专家权重 54 GiB，并验证这些字段被精化 IR、sidecar、summary 和 trace 逐段消费。

### 问题2（严重）: Qwen3主链写死Llama2，导致全链路未交付
　　**问题说明**：本轮要求 Qwen3 从 model_parser 开始，经精化、图切分、FlagTree、放置回传到 GeneSim 仿真，每段真实消费上段产物。`run_full_pipeline.py` 固定七种 Llama 投影、`llama2_7b.ir` 和 `sim_llama2_7b_*` 配置（62-75、113-136、811-848），`export_pp_placement.py:99-168` 固定加载 `LlamaForCausalLM`、dense `intermediate_size` 和 Llama 输出名。即使显式设置 30B 环境变量，实测仍打印并完成 Llama2 流程；把真实 30B IR 传给 `--ir` 也在 `layers.32.self_attn.q_proj.weight` 因 Llama 仅有 32 层而失败，不能作为等价入口。所得 sidecar 与哈希只能证明 Llama2。
　　**根本原因**：模型路径、加载类、策略、语义角色、文件名和仿真配置分散硬编码，主链没有架构描述贯穿 A/A2/B/C/D/E；单独给 `--ir` 换 Qwen3 文件仍会加载 Llama 权重，无法建立产物同源关系。FlagTree 区间没有 Qwen3 主链接入可弥补这一缺口。
　　**涉及文件**：`flagos-pim-compiler/scripts/run_full_pipeline.py:62-75,113-196,718-848`；`flagos-pim-compiler/scripts/export_pp_placement.py:99-168`；`flagos-pim-compiler/backend/architectures.py:88-105`（已有但未接入主链的 MoE 策略）；`flagOS-installers/FlagTree`（本区间无 Qwen3 调用入口）。
　　**修改思路**：以架构注册表派生统一的模型描述，包含加载器、策略、IR/sidecar/config 命名、router/专家角色、执行语义和必需原语，并贯穿所有阶段；校验模型身份、IR 编号、权重形状、sidecar、pimir 哈希和仿真配置的 lineage，任何一段缺失或回退都失败。交付仓库内 Qwen3 命令及端到端测试，分别核对 router/专家 GEMM 分类、真实 FlagTree 分块、sidecar 回传和 GeneSim 来源，禁止仅复用 Llama 产物作为通过证据。

### 问题3（严重）: 稠密前向溢出变NaN，导致数值偏离HF
　　**问题说明**：request §2.1 P0-1/P0-4 与 design §4.1 要求静态稠密模拟与 HF 稀疏前向严格等价。`backend/architectures.py:37-39` 对所有专家先计算再乘路由零；在 IEEE 浮点下，未选中专家若产生 `inf/nan`，`inf×0=nan`，并不等价于 HF 的“不计算该专家”。实测使用有限且合法的 fp16 输入和权重，令 router 只选专家 0、未选中的专家 1 内部溢出：HF 原前向输出全部有限（最大值 47.34375），静态前向产生 64 个 NaN。
　　**根本原因**：设计把代数上的零乘恒等误当成浮点恒等，缺少非有限值边界契约。实施 I4 只在 `backend/hal_numpy.py:318-329` 对“已跳过的 launch”写零，未覆盖静态前向、`PIM_MOE_DISPATCH=0`、prefill 或其他 host 回退，因而同一根因在前轮修复后仍复发；普通随机权重和真实权重未溢出使测试漏检。
　　**涉及文件**：`flagos-pim-compiler/backend/architectures.py:20-40`；`flagos-pim-compiler/backend/hal_numpy.py:318-329`；`flagos-pim-compiler/runtime/exec_plan_gen.py:25-26`；`flagos-pim-compiler/tests/test_architectures.py:264-286`、`tests/test_moe_dispatch.py:167-186`。
　　**修改思路**：在静态前向中先用路由掩码经 `torch.where` 选择“专家输出或零”，再做路由乘加，确保未选中分支的非有限值不会进入乘法和累加；该方向已用同一探针验证与原 HF 逐元素相等，选中专家若溢出仍应与 HF 一致暴露。同步为新增 `where` 补齐分区、规格传播和执行规则，并将防护覆盖所有静态/host 回退，避免数值修复引入无落点原语。增加有限输入导致未选专家 `inf` 与 `nan` 的回归，覆盖导出、prefill/decode、P0、P1 开关两态及 host/compiled 回退，并与未修改的 HF 稀疏前向对拍。

### 问题4（高）: 严格模式允许缺项回退，导致模板代价混入
　　**问题说明**：本轮要求 `require_compiler_pimir` 严格消费真实 pim mlir，禁止模板回退。`gene_sim_scheduler.py:751-756` 在 sidecar 没有某算子条目时无条件返回 `None`，`_warn_missing_pimir`（730-733）即使严格模式也只告警，随后 2180-2228 行生成模板 trace。实测 sidecar 自报 `requires_pimir=true` 但缺少全部条目时，日志先声明自动启用严格模式，随后仍生成 24/24 条模板 trace 并成功结束。`run_full_pipeline.py:372-379` 对新导出的缺哈希 sidecar 也允许 `--skip-simulation` 成功，进一步削弱同一契约。
　　**根本原因**：产物覆盖与完整性检查分散在编译器脚本和 GeneSim 两处，且把“合法无 trace 的节点”与“应有 pimir 但缺条目”混为同一种 `None`。反向核对表明编译器 preflight 能拒绝同一缺项产物，因此不是全路径无防护，而是直接 GeneSim 和只编译到 sidecar 的入口可绕过；合法 host、模型入口/出口、归约和被吸收节点必须保留显式豁免，不能一刀切误判。
　　**涉及文件**：`genesim/src/scheduler/gene_sim_scheduler.py:639-663,730-786,1942-2249`；`flagos-pim-compiler/scripts/run_full_pipeline.py:345-382,581-648`。
　　**修改思路**：在共享产物契约中逐类声明 `trace_required`/豁免原因，GeneSim 加载 IR 与 sidecar 时先核对模型身份、算子编号、每个必需条目的路径和哈希；严格模式遇到缺项、缺哈希、不可读或哈希不一致必须在生成 trace 前失败，新鲜导出不得沿用“旧产物”豁免。测试直接 GeneSim、完整脚本和 skip-simulation 三类入口，并同时验证合法 host/入口出口/归约/吸收节点不被误报。

### 问题5（高）: 默认分块令图失控，导致30B不可复现
　　**问题说明**：design §4.7、§七要求真实 30B 编译执行可验收，本轮又明确要求仓库内可复现入口。`runtime/compile.py:269-271` 未传 `tile`，`graph/opt_passes.py:1027-1044` 因而默认对专家 linear 按 64 KiB WRAM 分块；仓库没有 `PIM_DISABLE_TILE` 实现。实测仅 4 层、prefill=8 即从 7079 节点增至 1390344 节点，峰值 24.29 GiB；既有 48 层 PASS 依赖 `/data2/pengcheng-work/run_qwen3_30b.py:18-24` 的进程内 monkeypatch，正式 `tests/test_qwen3_30b_real.py` 没有该机制。
　　**根本原因**：通用线性分块策略没有 MoE 图规模约束或模型级开关，真实验收却在外部脚本改变核心编译行为，导致测试、文档和实际产物不来自同一入口。同一根因影响 prefill、decode、真实 pytest 和未来 Qwen3 主链；48 层约 1660 万节点仅为线性外推，不能冒充实测。
　　**涉及文件**：`flagos-pim-compiler/runtime/compile.py:221-271`；`flagos-pim-compiler/graph/opt_passes.py:984-1044`；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:34-51`；`flagos-pim-compiler/tests/README.md:195`；`/data2/pengcheng-work/run_qwen3_30b.py:18-24`（外部绕过证据）。
　　**修改思路**：把图优化选择纳入正式编译配置并逐模型/策略传递，MoE 应采用可扩展的专家子图复用或明确的安全分块策略，提供仓库内受支持的关闭入口及默认值，而不是环境变量被忽略或外部改函数。用正式入口实测 4 层与 48 层的节点数、内存和编译时间，确认 prefill/decode 均受控，并在 README 与实施文档记录同一命令和产物。

### 问题6（高）: 真实参考共用改写模型，导致对拍失去独立性
　　**问题说明**：request §2.1 P0-5、design §4.7/§7.1 要求真实 30B 与 HF 原始行为逐 token 一致。`tests/test_qwen3_30b_real.py:37` 通过注册表加载后，每层 forward 已被静态化；同文件 69 行却用同一个 `loaded.generate` 生成参考。外部 `/data2/pengcheng-work/run_qwen3_30b.py:93` 也如此，因此既有 48 层 PASS 只证明 PIM 与静态改写后的参考一致，不能独立发现静态改写自身的路由、溢出或归一化错误。
　　**根本原因**：加载器原地替换 forward，测试没有保存未修改执行语义或独立参考实例，编译结果与真值共享同一错误假设。小随机 MoE 使用另一个未静态化模型，说明只有真实模型和外部全模型日志缺失这一约束，不能以小模型通过替代真实独立对拍。
　　**涉及文件**：`flagos-pim-compiler/backend/architectures.py:68-73`；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:34-74`；`/data2/pengcheng-work/run_qwen3_30b.py:52,93`。
　　**修改思路**：在编译前保存各层原始 forward 供 HF 参考恢复，或独立加载未静态化参考模型，并在测试中断言参考路径未调用 `make_static_moe_forward`；真实权重对拍与静态等价性探针分离。保留本轮 4 层独立原始 HF 对拍通过的证据，同时重新形成 48 层独立参考证据，不能再引用共享 forward 的日志宣称 HF 验收完成。

### 问题7（高）: 容量验收抬高预算，导致8GiB未验证
　　**问题说明**：request §2.1 P0-3 明确要求 30B 在 8 DPU、每 DPU 8 GiB 下通过三区容量与无重叠检查。design §4.7 未记录需求确认就把测试预算改成 16 GiB，`tests/test_qwen3_30b_real.py:40-48` 的编译和后端也都使用 16 GiB；现有 4 层及 48 层日志同样不能证明 8 GiB 可行。因而“16 GiB 能运行”不能关闭 8 GiB 容量验收，也不能据此判断超预算报错是否给出所需与可用。
　　**根本原因**：设计阶段为解决 KV、激活和 tile 余量直接提高硬件假设，实施偏差表却没有登记该需求变更，测试只覆盖放宽后的环境。相同做法会掩盖真实权重、KV 槽池、并行窗口和内核 tile 叠加后的超预算路径。
　　**涉及文件**：`flagos-pim-compiler/docs/request-qwen3-moe-20261105.md` §2.1 P0-3；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.7；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:34-51`；`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §5.2。
　　**修改思路**：先回到需求/设计确认 8 GiB 是否仍为硬约束；若保留，则在 8 GiB 下由容量判据反推序列和窗口参数，断言权重/KV/激活/tile 的所需、可用及三区无重叠，并增加超预算错误文本用例。若确认改硬件目标，应同步修订需求、设计和验收证据，不能只把测试预算翻倍后宣称 P0-3 完成。

### 问题8（高）: 动态分派漏掉分块与子图，导致性能路径失效
　　**问题说明**：request §2.1 P1-1 要求只执行命中专家子计划并显著降低执行时间，design §7.1 要求实际 launch 数等于 top-k。`_moe_expert_of` 只认 linear 直接参数中的 `get_attr`（`runtime/exec_plan_gen.py:44-50`），分块插入 `_to_copy/slice` 后识别失败：实测 WRAM=1024 时 256 个专家 linear 中 0 个能取得 gate。即使不分块，641-647 行也只门控 linear；一层 4 专家的 decode 计划中 12 条 linear 有 gate，分支内 8 条 `mul` 以及 slice、归约仍无 gate，未命中专家仍做空计算并保留通信。现有测试只查 linear 标签和跳过集，implement §9 的 1.09× 总墙钟也未提供剔除固定成本后的执行期加速证据。
　　**根本原因**：专家识别与子图边界依赖未分块图的直接权重参数，既没有穿透布局中立节点，也没有把门控覆盖到激活、逐元素、重分布和归约命令；测试又只覆盖未分块小模型的 linear 元数据。同一根因使默认 tile 路径完全失去动态分派，并影响 prefill/decode、EP/host 回退和性能结论。
　　**涉及文件**：`flagos-pim-compiler/runtime/exec_plan_gen.py:44-68,641-665`；`flagos-pim-compiler/graph/opt_passes.py:984-1044`；`flagos-pim-compiler/backend/hal_numpy.py:217-245,313-330`；`flagos-pim-compiler/tests/test_moe_dispatch.py:103-186`；`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §9。
　　**修改思路**：沿 `_LAYOUT_NEUTRAL` 等布局中立链追溯专家权重，统一标记每个专家的完整子计划及通信，保持路由依赖、输出写零、泳道隔离和 WAR 等待。新增强制 tile 的 prefill/decode 用例，断言所有专家 linear 均可识别、未命中专家的计算与通信均不执行、命中结果与 HF 一致；再用剔除装载、编译和 HF 参考后的同一计划测量执行期加速比，避免只修当前未分块探针。

### 问题9（中）: 路径检查缺少跳过保护，导致默认回归失败
　　**问题说明**：request §五要求 `test_architectures.py` 全绿，项目 `tests/README.md:362` 要求外部资源用例使用 skip 保护。`test_qwen3_30b_model_dir_configured` 在 `paths.json` 默认 null 时仍无条件断言路径存在（`tests/test_architectures.py:364-370`）。实测未设置环境变量为 15 passed、1 failed；设置四层模型路径后为 16 passed。该结果使“真实权重可跳过”与“架构单测无需真实权重”的边界失效。
　　**根本原因**：把站点资源配置检查混入架构单元测试，且 30B 依赖未写入 `tests/README.md:60-72` 的依赖矩阵，和同文件真实用例的 skipif 策略不一致。同一问题会使完整回归在未配置站点路径时提前变红，也容易让 8B skip 被误读为全部真实路径通过。
　　**涉及文件**：`flagos-pim-compiler/tests/test_architectures.py:364-370`；`flagos-pim-compiler/tests/test_qwen3_30b_real.py:23-31`；`flagos-pim-compiler/tests/test_qwen3_8b_real.py:23-31`；`flagos-pim-compiler/tests/README.md:60-72,195,362`；`flagos-pim-compiler/paths.json:4-5`。
　　**修改思路**：路径优先级和键解析改用临时目录/monkeypatch 做无权重单测；真实目录存在性检查移入真实用例并统一 skipif，或按项目例外规范明确记录。补齐 30B 依赖矩阵和 skip 统计，报告中分别列出架构通过、30B 实测、8B 未覆盖，禁止用一个环境配置检查代替真实模型验收。

### 问题10（中）: 实施记录停留旧范围，导致新增验收无证据
　　**问题说明**：本轮要求 implement §1.3、§3、§4、§5 就地补充 GeneSim 全链路、执行语义取舍和验证证据。当前 §1.1-1.3 仍是编译器 P0 范围并宣称全部完成，§3 没有 model_parser、主流程或 GeneSim 改动，§4 没有 Qwen3 仿真与 sidecar/trace 证据，§5 未记录默认分块、模板回退、独立 HF 参考和 8B 未覆盖；Q1/Q2/Q3 还与后文完成状态矛盾。仅更新结论文字也不能替代本轮缺失的实现和产物。
　　**根本原因**：实施文档沿用旧 P0/P1 模板，没有按“契约 → 实现 → 产物/测试”维护可追溯验收矩阵，且历史状态未收敛。外部 `/data2` 脚本和日志没有映射到仓库内可复现命令，使临时绕过被误认为交付能力。
　　**涉及文件**：`flagos-pim-compiler/docs/implement-qwen3-moe-20261105.md` §1.1-1.3、§2-§6、§8-§9；`flagos-pim-compiler/docs/design-qwen3-moe-20261105.md` §4.7、§七；`flagos-pim-compiler/tests/README.md:195`。
　　**修改思路**：先完成上述代码与测试修复，再在 design 补充 GeneSim 执行语义和产物契约，在 implement 逐项记录真实改动文件、设计取舍、命令、通过/跳过/未运行状态、产物编号与哈希、资源峰值和遗留风险；同步收敛 Q1-Q3 与后续章节。不得只把实施状态改为完成，所有结论必须能由仓库内命令重放。
