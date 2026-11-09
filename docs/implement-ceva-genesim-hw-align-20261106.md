# 实施进展文档：GeneSim 对齐 CEVA 多核硬件与性能模型

> 文档编号：implement-ceva-genesim-hw-align-20261106
> 创建日期：2026-11-06
> 关联需求文档：[request-ceva-genesim-hw-align-20261106.md](request-ceva-genesim-hw-align-20261106.md)
> 关联设计文档：[design-ceva-genesim-hw-align-20261106.md](design-ceva-genesim-hw-align-20261106.md)
> 关联项目：flagos-pim-compiler（G）与 genesim（S）
> 起始基线：flagos-pim-compiler `f918586`、genesim `34481fe`；两个工作区起始时都只有未跟踪的需求/设计文档或空工作区。

## 一、实施概述

### 1.1 任务范围

本轮按设计第八节的 A～E 阶段落地结构、预算、映射、计时工作量和通信开销，主要改动在 genesim，图编译器只改了放置导出的预算入口。W4A8 物理布局、分阶段有效成本表、CEVA 锚点校准报告和秒级结果门控不在本轮交付范围内，见第五节。

### 1.2 完成标准

- CEVA 1/64/128 核 × 4e/1e 场景可加载，结构校验通过；正式运行校验拒绝 GPU、单核多槽、字节预算不一致和来源不全。
- CEVA-NPM 使用内置轨迹后端，找不到轨迹直接报错。
- 每核容量以整数 `ceva.local_memory_bytes` 为真源，扣除系统预留。
- CEVA 放置必须显式一对一映射；有哈希声明时核验 pim mlir 内容；计算节点缺算子编译产物即失败。
- 相位链计算量随 M/N/K、元素数和归约长度增长，超过 14 位立即数不截断。
- 区间倍率与 HOST 中转开销可从配置注入并进入 trace 缓存签名。
- CEVA 正式评估只接受 pipeline 后端和编译器本地分片；产物在加载时核对计算原语、形状、目标与放置字节；融合吸收必须绑定主算子及融合相位；验证样本绑定产物相位与运行形状范围。

### 1.3 实施结论

总体完成状态：部分完成。A、B、C 阶段的主要结构项和计时项已完成并通过测试；D 阶段的通信开销配置已完成，校准报告复用现有能力（`calibrate.py` / `pim_op_calibration.py` 已有拟合/验证分组和 median 误差）；E 阶段完成了运行期资源清点和结果成熟度标注。物理布局、分阶段有效成本表和秒级结果门控尚未实施。round02 已修复评审提出的 4 项问题：相位轨迹运行时重复缩放、严格产物模式可关闭、成熟度未绑定成本与精度、循环同步成本落入内置默认值。round03 已修复评审提出的 4 项问题：TP 分片丢失符号维占位值、精度校验只查类型集合成员、空验证证据可升级成熟度、中转开销缺来源。round04 已修复评审提出的 4 项问题：相位轨迹在运行时按实际形状逐操作数重建、验证报告绑定请求 trace 与请求边界、验证样本绑定产物并检查计算/DMA/prefill/decode 覆盖、纳秒断言精度不足。round05 已修复评审提出的 5 项问题：CEVA 正式模式只接受管线后端且结果成熟度绑定实际后端、产物加载时核对计算原语/形状/目标/放置字节、融合吸收绑定主算子与融合相位、强制按编译器本地分片执行、验证样本绑定产物相位与运行形状范围。round06 已修复评审提出的 3 项问题：相位产物按执行节点逐轴核对操作数与结果（同末维不同秩/前导轴、无法解析形状均拒绝）、结果成熟度绑定配置核数/实际创建核数/放置计划已用核数、验证样本形状按请求 trace 的 prefill/decode 运行时形状展开而不再使用编译占位值。round07 已修复评审提出的 3 项问题：输入操作数在位置一一对应时必须取得明确形状匹配（无结果写入类算子、结果正确但输入错误均拒绝）、SPLIT 产物的动态维/不可解析形状在加载时拒绝且相位解析对无法确定字节的张量报错、验证报告的 p90 按逐点绝对相对误差以最近秩复算。round08 已修复评审提出的 2 项问题：请求边界按 `is_prefill` 区分 decode 步数并让 `max_query_length` 反映实际前向 Tq、实际使用核数无法确定时不再跳过核对而是保持 `model_prediction`。round09 已修复评审提出的 3 项问题：相位产物按实际语句核对计算原语、最终结果绑定到真正的返回张量、验证报告的计时边界纳入并发上限并把契约升到 `schema_version=3`。round10 已修复评审提出的 2 项问题：产物语句按执行节点完整允许集合核对，搬运/布局原语不再绕过门禁；结果成熟度派生核对生效参数的来源等级与证据，假设来源不能靠报告文字升级。round11 已修复评审提出的 3 项问题：单行/单列内核视图豁免只对真实登记的 GATHER、QUANT 开放，未登记算子不能用 1×N 冒充 M×N 少计工作量；验证报告的通信链路比较纳入簇内/跨簇适用范围与全/半双工语义并把契约升到 `schema_version=4`；独立验证样本的预测周期必须为有限正数。round12 已修复评审提出的 3 项问题：CEVA 场景显式配置并绑定主机编排成本，主机出入口与归约开销单列并纳入独立验证，验证报告契约升到 `schema_version=5`；相位链编译透传 `max_bulk_transfer_bytes`，磁盘指令轨迹与运行时解析公式采用同一分段口径；五份 CEVA 场景改为纯 PIM 权重部署，正式校验拒绝非 PIM 权重映射。round13 已修复评审提出的 3 项问题：主机节点识别、验证证据与主机指标改读本次实际执行的 TP 展开后放置图，ALL_REDUCE 必须进入独立验证、合成中转不作为实测主机算子；整核峰值参数 `multipliers` 纳入成本模型摘要与逐项一致性检查并把契约升到 `schema_version=6`；CPU 执行路径对 `COLLECTIVE_RELAY` 采用与 GPU 一致的零计算/零存储语义，通信成本只在跨核边上计一次。round14 已修复评审提出的 2 项问题：大 M 分块产物的 M/N/K 三层循环与超过 14 位立即数的 MAC/K 工作量现在都能被解析、编译并完整计费（M 分块按运行时重复），不再拒绝合法的编译器分块；放置 sidecar 升到 version 4 并导出物理 MRAM、系统预留与可用字节，CEVA 严格模式逐项核对 `ceva.local_memory_bytes`/`ceva.sys_reserve_bytes`，分块产物同时核对 `pim.mram-bytes`，旧 sidecar 与另一套预算的产物在生成 trace 前即被拒绝。round15 已修复评审提出的 2 项问题：CEVA 严格模式要求 GEMM 放置条目提供正整数 `kernel_tile_n` 且与产物 `pim.tile-n` 完全一致，VPU 运行时再按轨迹元数据核对一次，缺失或不一致在生成 trace/执行前拒绝；验证报告的 `communication_model` 新增 `inter_cluster_communication` 并与本次生效配置逐项比较，路由模式同时纳入必需参数来源，报告契约升到 `schema_version=7`。round17 已修复评审提出的 2 项问题：CEVA 正式校验补充 CPU↔HOST 双向链路要求，缺少任一条不再通过；通用摘要仅在结果实际包含 `result_maturity` 时写入该字段，恢复通用键集合不变。round18 已修复评审提出的 3 项问题：CEVA 正式校验与加载端拒绝缺失或不可读的成本 sidecar；CEVA 结果目录归档带版本的运行清单（输入与 PIM MLIR 哈希、工具链身份、硬件预算），版本缺失或哈希变化在评估入口即拒绝；容量字段的正负无穷与 NaN 在校验和容量解析两处按契约返回失败或抛出带字段名的 `ValueError`，不再抛 `OverflowError`。round19 已修复评审提出的 2 项问题：验证形状范围改读 TP 展开后的实际执行图，全局 11008 维样本不再放行、真实本地分片样本可以正常核对；运行清单的解析配置应用 `GENESIM_*` 环境覆盖，与实际执行取值、硬件预算保持一致。round21 已修复评审提出的 5 项问题：CEVA 正式校验与调度器入口同时拒绝自动放置搜索；成本 sidecar 形成消费台账并绑定执行结果、运行清单和成熟度；循环编码开关与实际生效的循环原语纳入成本模型契约和来源台账；区间倍率校验下沉到编码与公共编译/求值入口；`override()` 生成的内存配置可归档运行清单。round22 已修复评审提出的 3 项问题：相位路径的 MAC 工作量改为每个输出分别按 `ceil(K/mac_width)` 计，与分块路径统一；成本 sidecar 覆盖集合只认真实搬运成本和显式模板路径，空条目与仅来源名条目不再冒充已覆盖，验证报告契约升到 10；CEVA 长循环备用接口统一按 `outer×inner` 求总次数并复用直接计数的升级/越界检查。round23 已修复评审提出的 3 项问题：模型 IR、放置 sidecar、请求 trace、成本 sidecar 和 PIM MLIR 在实际加载时统一登记输入快照，成熟度、消费台账与运行清单只引用该快照并在结束前逐项复核，运行中改写输入会显式失败而不是换绑到旧结果；CEVA 成本 sidecar 在统一解析边界校验搬运字节必须为有限非负数（`None` 表示未测得、零值保持既有语义），非法值在算子落值前拒绝整份输入；CEVA 主机参数校验改读生效配置（含 `GENESIM_*` 覆盖），CPU 核数必须为正整数、基础延迟/算力/带宽必须为有限正数，CPUModel 构造与 VPU 工厂复用同一契约，非法值在资源创建与执行前失败。round24 已修复评审提出的 3 项问题：模型 IR 与请求 trace 改为单次读取字节同时完成解析和快照登记，解析/登记窗口改写文件必然复核失败，请求边界证据改读已消费的 trace 对象；CEVA 资源入口把 `cpus` 列表与零/负核数收敛为配置校验、完整工厂和运行后清点共同遵守的契约；外部数值转换统一走 `finite_float()`，成本、配置、区间编码、验证报告、VPU/容量辅助与预测器入口不再抛未包装的 `OverflowError`。round25 已修复评审提出的 2 项问题：CEVA 正式校验、通信库/工厂构造与验证报告统一执行链路数值契约（带宽有限正数、基础延迟有限非负），非法类型与转换溢出按链路和字段失败，全双工零带宽并发不再除零，同时保留通用零带宽单路返回无穷与合法 UPMEM 行为；周期成本数值域下沉到 `resolve_cycle_costs()` 严格模式，指令轨迹、CEVA 核心与相位解析共用同一入口，非有限、负值与未放行的零值统一拒绝，配置校验覆盖成本表内全部键（含运行期直接读取的 `REDUCE_PARTIAL`），归约时长入口同步拒绝非法值。round26 已修复评审提出的 3 项问题：请求 trace 新增 `padded_query_length`/`initial_kv_length` 与统一 `shape_binding()`，有效 prompt 长度、padding 后执行查询长度和初始 KV 分别登记并贯穿实际执行、验证范围展开、样本核对、容量与报告，无法唯一确定三者的 CEVA 请求在运行入口显式拒绝，验证报告契约升到 `schema_version=11`；资源层级与 engine 数收敛为配置校验、完整工厂、运行后清点和运行清单共用的非布尔整数契约，`int()` 不再把布尔/浮点数量合法化，目标名 `_<n>e` 后缀与 engine 数交叉核对；三个链路语义标记纳入 `link_field_errors()`，字符串/整数/空值在配置、通信库、完整工厂和验证报告边界统一失败，合法布尔全/半双工与路由语义保持不变。round27 已修复评审提出的 3 项问题：CEVA PIM MLIR 首次按字节读取时建立不可变文本与哈希台账，数据类型、相位、验证样本、融合豁免、成熟度和 trace 生成共用该台账并禁止运行期重读磁盘，成熟度派生结束前统一复核消费快照，TP 展开出的分片算子按继承路径复用同一台账；CEVA 链路端点收敛为 HOST/PIM/CPU 纯 PIM 资源域，配置结构/正式校验、通信库构造（含直接 `add_link`）、完整工厂和验证报告统一拒绝 GPU 与未知域；CEVA 循环次数与整轨迹重复次数统一为非布尔整数契约，循环次数必须为正整数、重复次数必须为正、超过 24 位显式失败，高层后端、直接核心与 Python 轨迹绑定共用同一实现，通用/UPMEM 的 12 位兼容截断与零值既有语义保持不变。round28 已修复评审提出的 2 项问题：YAML 配置纳入第六类消费输入，加载时单次读取并登记原始字节身份，直接文件与无参数/带参数 `override()` 在运行结束前改写源文件均显式失败，运行清单与输入快照只引用消费时身份；CEVA 通信契约收敛到所有构造入口，直接 `add_link()` 复用完整字段契约，`CommLib` 构造与完整工厂共用 HOST 中转开销的有限非负检查，负时延、非法数值、字符串布尔标记与负中转开销不再被接受。round29 已修复评审提出的 4 项问题：相位账本不再只收集带 `phase_spec` 的行，同一文件里不带该属性、但有真实计算/搬运语义的语句按单相规则补计工作量与输入/输出字节，只有显式登记的零成本布局语句豁免，未知语句直接报错，模板语义版本升到 2 使旧缓存失效；配置来源种类在构造时固定，加载时缺失的路径在运行中新建文件不再被 `is_file()` 换绑为文件来源，内存配置始终按生效内容登记；映射导出脚本与 `build_mapping()` 一对一分支接入 CEVA 资源数量契约，浮点/布尔/字符串数量、非法核数与 GPU 资源在生成计划前失败，通用路径保持历史 `int()` 读取；运行清单统一调用共享容量解析结果，字节预算不再在清单层 `int()` 取整，浮点/布尔/字符串、负预留、预留吃光物理量和 GiB 视图不一致在任何清单输出前失败，有/无输入快照两条路径同口径。round30 已修复评审提出的 3 项问题：无相位 eltwise/lut helper 按 SSA 消费者归属折进相位，不再独立成相或产生 f16 中间往返（真实 SOFTMAX 恢复 5 个硬件相位），相位模板升到 v3；同编号轨迹重载、周期表、区间倍率和时钟全部纳入 VPU 相位结果缓存身份；切分参数校验收敛为共享前置契约，`main()` 与 `build_mapping(one_to_one=True)` 在除法前共同拒绝零/负/非整除/非 2 次幂/非法类型，另一条固定计划生产入口同步加固。round31 已修复评审提出的 1 项问题：放置 sidecar 的执行消费者（TP 展开与集合通信补边）收敛到首次消费视图，执行中途改写不再换版本。round32 已修复评审提出的 3 项问题：轨迹装载与周期表缓存改为按内容身份（文件 SHA256、周期表内容冻结），同路径覆盖与原字典原位改值不再返回旧成本；集合通信的 nbytes/seq_len/dpu_ids 契约在配置校验、调度器首次消费与图编译器导出三处统一，畸形 sidecar 在改图前失败；权重聚合与两个容量参数纳入有限数值域，NaN/无穷/负值不再绕过容量门禁。

## 二、任务完成清单

| 任务ID | 任务描述 | 对应设计模块 | 完成状态 | 备注 |
|--------|----------|--------------|----------|------|
| T1 | CEVA 场景配置、structure/simulation 两级校验、`--dry-run` 走结构校验 | §3.1、§3.2 | 完成 | 新增 5 份 `conf/ceva_*.yaml` |
| T2 | CEVA-NPM 内置 PIMTrace 后端与分派 | §5.1 | 完成 | 缺轨迹直接抛错 |
| T3 | 每核容量按整数字节真源、系统预留 | §4.1 | 完成 | GiB 字段与字节真源强制一致 |
| T4 | 显式 DPU 一对一映射、映射合法性、pim mlir 哈希、计算节点覆盖 | §3.3、§7.1 | 完成 | CEVA 禁止取模退化 |
| T5 | 相位链工作量（M/N/K、元素数、归约）与不截断循环编码 | §5.2 | 完成 | 缓存签名升到 8；round02 修正运行时按 Tq 重复整条静态轨迹的问题（round02 问题1），T5 结论恢复成立 |
| T6 | `pim.zone_cycle_factors` 注入、校验和签名 | §5.3 | 完成 | 支持 0.5 步长、上限 4095 |
| T7 | HOST 中转固定开销配置化 | §6 | 完成 | 缺省仍为 100 ns；round03 补齐五份 CEVA 场景的来源台账并纳入正式校验必需项 |
| T8 | 运行期核资源清点与结果成熟度 | §3.2、§7.3 | 完成 | 报告创建/使用/空闲核 |
| T9 | 图编译器放置导出 `--mram-bytes` / `--sys-reserve-bytes` | §4.1 | 完成 | 缺省 8,000,000,000 B / 0（CEVA 64 核 4e 画像）；round14 起 sidecar 以 version 4 导出同一份物理量/预留/可用量，CEVA 严格模式逐项核对（Q7 已关闭） |
| T10 | 多核一对一映射导出 | §3.3 | 完成 | `config.is_ceva()` 时强制独占 ClusterPU |

## 三、代码改动详情

### 3.1 改动总览

- 新增文件数：8（5 份 CEVA 场景配置、1 个映射测试、1 个 G 预算测试、1 份实施文档）
- 修改文件数：20（genesim 18 个，G 2 个）
- 删除文件数：0
- 核心改动说明：CEVA 画像与两级校验、内置 CEVA 后端、字节预算、显式映射与产物校验、相位工作量与区间倍率。

round02 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`、`src/scheduler/gene_sim_scheduler.py`、`src/vpu/vpu.py`、`src/pim/pim_core.py`、`src/pim/pim_trace_loader.py`、`conf/ceva_*.yaml` 5 份；测试 `tests/sim/test_vpu.py`、`test_compiler_placement.py`、`test_config_loader.py`、`test_pim_backend.py`。
- G：`genesim_bridge/placement_export.py`、`tests/test_placement_export.py`。
- F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round03 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（TP 占位值继承、精度绑定、独立验证证据）、`src/config_loader.py`（中转开销来源）、`conf/ceva_*.yaml` 5 份；测试 `tests/sim/test_compiler_placement.py`、`tests/sim/test_config_loader.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round04 增补（均为修改，未新增文件）：

- genesim：`src/pim/pimir_trace.py`（相位张量形状与运行时模板）、`src/vpu/vpu.py`（按实际形状重建轨迹并求值）、`src/scheduler/gene_sim_scheduler.py`（模板生成、请求边界、样本覆盖、trace schema 升 9）；测试 `tests/sim/test_vpu.py`、`tests/sim/test_pimir_trace.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_pim_backend.py`。
- G：本轮未修改代码；只核对 `genesim_bridge/placement_export.py` 的占位值生产端，符号维仍按 128 占位登记。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round05 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`（正式模式拒绝预测器后端与关闭本地分片）、`src/scheduler/gene_sim_scheduler.py`（产物语义/形状/目标/放置字节核对、融合归属核对、成熟度绑定实际后端、运行期护栏、验证样本相位与形状范围核对）；测试 `tests/sim/test_config_loader.py`、`tests/sim/test_compiler_placement.py`。
- G：本轮未修改代码；`genesim_bridge/placement_export.py` 已导出的 `fused_activations` 与 `absorbed_ops` 足以在仿真侧绑定融合归属，无需新增字段。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round06 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（相位产物逐轴形状核对、核数范围核对、按请求 trace 展开 prefill/decode 运行形状）；测试 `tests/sim/test_compiler_placement.py`。
- G：本轮未修改代码；验证报告新增的核数字段由独立校准产出方生成，G 不消费该报告。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round07 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（输入操作数位置对应时强制明确匹配、SPLIT 动态形状拒绝、p90 最近秩复算、最近秩函数上提为模块级复用）、`src/pim/pimir_trace.py`（相位解析拒绝无法解析静态形状的张量）；测试 `tests/sim/test_compiler_placement.py`、`tests/sim/test_pimir_trace.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round09 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（计算原语混入检查、最终返回张量绑定、计时边界纳入 `max_inflight_requests`、验证报告契约升到 3）；测试 `tests/sim/test_compiler_placement.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round10 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（产物语句按执行节点允许集合核对，保留 GEMM 必需的 barrier/DMA/WRAM 基础设施；成熟度派生核对参数来源等级与证据）、`tests/sim/test_compiler_placement.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round11 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（单行内核视图按算子登记、通信链路语义纳入规范化与比较、验证报告契约升到 4、预测周期必须为正）、`tests/sim/test_compiler_placement.py`（新增 7 条用例与 4 份产物夹具）。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round12 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`（主机成本来源路径、显式 `cpu_model` 校验、纯 PIM 权重部署校验）、`src/scheduler/gene_sim_scheduler.py`（主机成本绑定与验证、主机证据核对、主机指标单列、验证报告契约升到 5）、`src/main.py`（主机指标写入摘要）、`src/pim/pimir_trace.py`（相位链透传分段上限）、`conf/ceva_*.yaml` 5 份（主机成本与纯 PIM 权重部署）；测试 `tests/sim/test_config_loader.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_main.py`、`tests/sim/test_pimir_trace.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round13 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（主机节点识别/指标/证据读取实际执行图、合成中转从证据集合排除、指标逐类型计数与零成本标记、成本模型绑定 `multipliers`、验证报告契约升到 6）、`src/backend/cpu_backend.py`（`COLLECTIVE_RELAY` 零开销剖面）、`src/main.py`（主机逐类型计数与零成本类型写入摘要）；测试 `tests/sim/test_compiler_placement.py`、`tests/sim/test_collective_edges.py`、`tests/sim/test_cpu_backend.py`、`tests/sim/test_main.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round14 增补（均为修改，未新增文件）：

- genesim：`src/pim/pimir_trace.py`（M/N/K 三层循环与折叠形态识别、M 分块工作量、MAC/K 混合进制循环、`trace_metadata` 增加 M 字段、删除 `_capped`）、`src/vpu/vpu.py`（GEMM 按 `pim.tile-m` 计算 M 方向重复次数）；测试 `tests/sim/test_pimir_trace.py`、`tests/sim/test_vpu.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_collective_edges.py`。
- genesim：`src/scheduler/gene_sim_scheduler.py`（放置 sidecar 预算核对、GEMM 产物 `pim.mram-bytes` 核对）；测试 `tests/sim/test_compiler_placement.py`。
- G：`genesim_bridge/placement_export.py`（sidecar version 4 与顶层 `hardware`、`sys_reserve_bytes` 参数与校验）、`scripts/export_pp_placement.py`（透传 `HwBudget.sys_reserve_bytes`）；测试 `tests/test_placement_export.py`。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round15 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`（`pim.inter_cluster_communication` 纳入必需参数来源，正式校验只接受 `direct`/`host_relay`）、`src/scheduler/gene_sim_scheduler.py`（GEMM 产物核对 sidecar `kernel_tile_n`、验证报告核对路由模式、契约升到 7）、`src/vpu/vpu.py`（CEVA GEMM 运行时按轨迹 `tile_n` 复核分块）、`conf/ceva_*.yaml` × 5（路由模式来源）；测试 `tests/sim/test_compiler_placement.py`、`tests/sim/test_config_loader.py`、`tests/sim/test_vpu.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round17 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`（CEVA 正式校验增加 CPU→HOST 与 HOST→CPU 双向链路要求）、`src/main.py`（`result_maturity` 仅在结果实际包含该字段时写入摘要）；测试 `tests/sim/test_config_loader.py`、`tests/sim/test_main.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round18 增补（新增 2 个文件，其余为修改）：

- genesim：新增 `src/run_manifest.py`（CEVA 运行清单：输入产物与 PIM MLIR 哈希、图编译器/FlagTree 提交号、实际 `triton-opt` 路径与指纹、硬件预算、解析后配置）；`src/config_loader.py`（非有限容量按校验契约返回失败、新增 `resolved_config()`、CEVA 正式校验拒绝缺失成本 sidecar）；`src/pim/capacity.py`（非有限 GiB 换算抛带字段名的 `ValueError`）；`src/scheduler/gene_sim_scheduler.py`（CEVA 加载端拒绝缺失/不可读成本 sidecar、`_ceva_run_manifest()` 并把清单挂到结果）；`src/main.py`（创建 Scheduler 前核对工具链身份、结果目录写 `run_manifest.json`、摘要登记清单文件名）；新增测试 `tests/sim/test_run_manifest.py`，补充 `tests/sim/test_config_loader.py`、`tests/sim/test_pim_capacity.py`、`tests/sim/test_cost_sidecar.py`、`tests/sim/test_main.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round19 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（`_ceva_run_shapes_of`、`_ceva_run_matmul_stage_m`、`_ceva_run_matmul_shape_bounds` 改读 `_execution_ir_or_model()` 的 TP 展开后执行图，未运行时的兼容回退保持不变）；`src/config_loader.py`（`resolved_config()` 应用 `GENESIM_*` 环境覆盖，新增 `_apply_env_overrides`）；测试 `tests/sim/test_compiler_placement.py`（新增 3 条 TP2 范围与成熟度用例）、`tests/sim/test_run_manifest.py`（新增环境覆盖清单用例）。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round21 增补（均为修改，未新增文件）：

- genesim：`src/config_loader.py`（CEVA 正式校验拒绝 `autotune.enabled=true`、`pim.compile.use_long_loop_counts` 纳入必需来源、区间倍率校验改用共享实现、新增 `source_path`/`override_chain`）；`src/scheduler/gene_sim_scheduler.py`（`run()` 直接拒绝 CEVA 自动放置、成本 sidecar 消费台账、成熟度核对覆盖范围、成本模型新增 `loop_encoding`、运行期短→长循环升级登记）；`src/pim/pim_isa.py`（新增 `validate_zone_cycle_factors`，`zone_scale*` 共用）；`src/pim/pimir_trace.py`（两条公共编译入口与解析求值入口校验倍率）；`src/pim/pim_trace_loader.py`（短循环升级写入 `runtime_loop_upgrade` 元数据）；`src/run_manifest.py`（`config_entry`/`resolved_config_sha256`、清单记录成本消费台账）；`conf/ceva_*.yaml` 5 份（循环编码来源）；测试 `tests/sim/test_config_loader.py`、`test_cost_sidecar.py`、`test_compiler_placement.py`、`test_run_manifest.py`、`test_zone_cycles.py`、`test_pimir_trace.py`、`test_vpu.py`。
- G：本轮未修改代码，只运行回归。F：工作区保持干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round22 增补（均为修改，未新增文件）：

- genesim：`src/pim/pimir_trace.py`（抽出 `_matmul_dims_from_shapes`，新增 `_matmul_mac_groups_from_shapes`；`_phase_work_units` 的乘加分支改为按每个输出分别取整，编译与运行时解析共用）；`src/scheduler/gene_sim_scheduler.py`（成本 sidecar 覆盖集合改为“有效搬运成本 + 显式模板路径”，验证报告契约升到 `schema_version=10`）；`src/pim/pim_core.py`（`PIMTracePIMCore.set_tile_count_long` 按 `outer×inner` 求总次数并校验正整数，`execute` 统一走 `PIMTrace.set_tile_count` 的升级/越界检查）；测试 `tests/sim/test_pimir_trace.py`、`test_cost_sidecar.py`、`test_compiler_placement.py`、`test_pim_backend.py`。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round23 增补（均为修改，未新增文件）：

- genesim：`src/run_manifest.py`（新增 `verify_input_snapshot`/`_verify_snapshot_entry`/`_require_snapshot_inputs`；`build_ceva_run_manifest()` 新增 `input_snapshot` 参数，传入时只引用消费快照并先复核磁盘内容）；`src/scheduler/gene_sim_scheduler.py`（`load_model_ir`/`load_trace` 在加载时登记快照；`_compiler_sidecar` 首次读取后缓存并登记；`_load_compiler_placement` 改用缓存；`_load_cost_sidecar`/`_cost_sidecar_consumption` 按实际解析字节记录哈希并执行数值域校验；`_build_cost_sidecar_ledger` 记录消费内容哈希；`_ceva_validation_scope_error`、`_ceva_cost_sidecar_error`、`_ceva_run_manifest` 改读快照；`run()` 执行前锁定输入身份）；`src/config_loader.py`（新增 `ceva_host_parameter_errors`，`_validate_ceva` 改读生效配置并校验 `system.num_cpus`）；`src/backend/cpu_backend.py`（CEVA 构造边界复用同一数值契约，`_from_dict` 透传 `pim_type`）；`src/vpu/factory.py`（CEVA 工厂使用生效 `cpu_model` 构造并走同一契约）；测试 `tests/sim/test_cost_sidecar.py`、`test_compiler_placement.py`、`test_run_manifest.py`、`test_config_loader.py`、`test_cpu_backend.py`。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round24 增补（新增 1 个文件，其余为修改）：

- genesim 新增：`src/numeric_utils.py`（`finite_float()` 统一类型检查、转换与有限性判断，显式接住 `TypeError`/`ValueError`/`OverflowError`）。
- genesim 修改：`src/ir/model_ir.py`（新增 `ModelIR.from_bytes()`，`load()` 改为先读字节再解析）；`src/ir/trace_ir.py`（新增 `Trace.from_json_bytes()`/`from_binary_bytes()`，`load_json()`/`load_binary()` 复用）；`src/scheduler/base_scheduler.py`（`load_model_ir`/`load_trace` 一次读取字节、解析和登记共用，新增 `_register_loaded_input()` 钩子）；`src/scheduler/gene_sim_scheduler.py`（`_register_loaded_input` 按同一次读取的字节登记快照；`_load_configured_trace`/`_ceva_request_boundary` 优先使用已消费 trace；`_finite_number`、`_validate_ceva_mram_value` 和预测器入口的 `positive_float` 改用 `finite_float`）；`src/run_manifest.py`（新增 `artifact_entry_from_bytes()`）；`src/config_loader.py`（新增 `ceva_flat_resource_errors()`，`_validate_ceva` 拒绝 `cpus` 列表；主机参数、autotune、周期成本、时钟、中转开销与 legacy 容量转换改用安全数值边界）；`src/vpu/factory.py`（新增 `_ceva_resource_contract_errors()`，`create_vpus_and_topology` 在任何资源循环前统一拒绝）；`src/main.py`（`_ceva_resource_error` 核对实际 CPU 数量）；`src/pim/pim_isa.py`、`src/vpu/vpu.py`、`src/pim/capacity.py`、`src/pim/pim_compiler.py`、`src/performance_objective.py`、`src/autotune/dataset_observer.py`、`src/autotune/placement.py`、`src/predictor/dataset.py`、`src/predictor/benchmark.py`（同类外部数值转换复用共享边界或接住 `OverflowError`）。
- genesim 测试：`tests/sim/test_compiler_placement.py`、`test_config_loader.py`、`test_cpu_backend.py`、`test_main.py`、`test_cost_sidecar.py`、`test_zone_cycles.py`、`test_vpu.py`（新增失败用例，见 7.1 第 24 轮清单）。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round25 增补（均为修改，未新增文件）：

- genesim：`src/backend/comm_lib.py`（新增 `link_field_errors()`/`_link_number()`；`CommLink.from_dict()` 统一转换并按链路与字段报错；`concurrent_latency()` 复用零带宽保护；`CommLib` 对 CEVA 配置在 `add_links()` 构造边界执行同一契约）；`src/config_loader.py`（新增 `ceva_comm_link_errors()`；`_validate_ceva` 校验链路数值域并覆盖周期成本表内全部键）；`src/vpu/factory.py`（`_ceva_resource_contract_errors` 纳入链路契约，`_make_comm_link` 复用 `CommLink.from_dict`）；`src/scheduler/gene_sim_scheduler.py`（`_normalized_links` 复用链路契约；`_estimate_reduction_time` 拒绝非法 `REDUCE_PARTIAL`）；`src/pim/pim_trace_loader.py`（新增 `_strict_cycle_cost`，`resolve_cycle_costs(..., strict=)` 统一数值域检查，两个轨迹求值入口透传 strict）；`src/pim/pimir_trace.py`（相位解析把 strict 传给共享周期表解析器）。
- genesim 测试：`tests/sim/test_comm_lib.py`、`test_config_loader.py`、`test_cpu_backend.py`、`test_compiler_placement.py`、`test_pim_trace_loader.py`、`test_pimir_trace.py`、`test_pim_backend.py`、`test_scheduler.py`、`test_vpu.py`（新增失败用例，见 7.1 第 25 轮清单）。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round26 增补（均为修改，未新增文件）：

- genesim：`src/ir/trace_ir.py`（新增 `RequestShapeBinding` 与 `Request.shape_binding()`；请求新增 `padded_query_length`/`initial_kv_length`，JSON 与二进制 trace 升到 version 3 并保留旧版读取兼容；`__post_init__` 按显式绑定推断 `seq_len`）；`src/scheduler/gene_sim_scheduler.py`（`_ceva_request_boundary` 分别登记有效 prompt、执行查询和初始 KV；`_ceva_runtime_shape_contexts`、`_ceva_runtime_query_bounds`、`_get_operator_with_runtime`、`_execute_pipeline` 的 `step_context` 与自适应放置成本聚合统一改读 `shape_binding()`；CEVA 歧义请求在运行入口显式拒绝；请求指标新增有效长度与 padding 查询字段；验证报告契约升到 `schema_version=11`）；`src/pim/capacity.py`（峰值驻留形状复用同一绑定）；`src/config_loader.py`（新增 `ceva_resource_count_errors()`，资源层级、engine 数与 GPU 数按非布尔整数校验，目标名 `_<n>e` 后缀交叉核对；`get()` 的 `GENESIM_*` 覆盖按配置原值类型转换）；`src/vpu/factory.py`（完整工厂在任何资源循环前复用资源数量契约）；`src/main.py`（运行后清点复用同一契约，删除 `int()` 合法化）；`src/run_manifest.py`（硬件预算复用资源数量契约，删除 `int()` 合法化）；`src/backend/comm_lib.py`（`link_field_errors()` 增加端点与三个语义标记的布尔契约）。
- genesim 测试：`tests/sim/test_trace_ir.py`、`test_compiler_placement.py`、`test_config_loader.py`、`test_comm_lib.py`、`test_cpu_backend.py`、`test_main.py`、`test_run_manifest.py`、`test_scheduler.py`、`test_pu_pool.py`（新增失败用例，见 7.1 第 26 轮清单）。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

round27 增补（均为修改，未新增文件）：

- genesim：`src/scheduler/gene_sim_scheduler.py`（新增 `_ceva_pimir_texts()`，首次按字节读取 PIM MLIR 时保存文本、哈希和 sidecar 路径台账；`_ceva_artifact_texts_by_op_type()`、`_ceva_artifact_phases()`、`_ceva_run_compute_dtypes()`、`_ceva_validation_evidence_error()`、`_validate_ceva_fusion_exceptions()` 与 `_operator_pimir()` 全部改读台账，禁止运行期重读磁盘；`_ceva_validation_scope_error()` 在派生结论前复核输入快照；TP 展开分片按继承路径复用台账）；`src/backend/comm_lib.py`（新增 `CEVA_LINK_DOMAINS`/`link_endpoint_errors()`，`link_field_errors()` 与 `CommLib.add_link()` 复用端点契约）；`src/config_loader.py`（新增 `ceva_comm_link_endpoint_errors()`，结构校验与正式校验统一拒绝非纯 PIM 端点）；`src/backend/pim_backend.py`（`PIMBackend.execute_runtime()` 与 `UnifiedPIMBackend.execute_runtime()` 在任何轨迹绑定前校验计数类型、符号与 CEVA 正整数契约）；`src/pim/pim_core.py`（`PIMTracePIMCore.set_tile_count()`/`set_tile_count_long()`/`execute()` 共用计数契约，`CEVANPMPIMCore` 要求正整数）；`src/pim/pim_trace_loader.py`（新增 `require_loop_count()`，`PIMTrace.set_tile_count()`/`set_tile_count_long()` 拒绝布尔、浮点与非正 CEVA 次数，24 位上限显式失败）。
- genesim 测试：`tests/sim/test_compiler_placement.py`、`test_config_loader.py`、`test_comm_lib.py`、`test_cpu_backend.py`、`test_pim_backend.py`（新增失败用例与合法对照，见 7.1 第 27 轮清单）。
- G、F：本轮未修改代码，只运行回归与工作区核对。F 工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建。

### 3.2 详细改动清单

| 文件路径 | 改动类型 | 改动内容简述 | 核心函数/类 | 对应任务ID |
|----------|----------|--------------|-------------|------------|
| genesim/src/config_loader.py | 修改 | 新增 `ceva` 段、CEVA 常量、两级校验、区间倍率校验、中转开销校验 | `validate`、`_validate_ceva`、`_validate_zone_cycle_factors`、`is_ceva` | T1、T6 |
| genesim/src/main.py | 修改 | `--dry-run` 走结构校验；运行期核清点与使用/空闲报告 | `_ceva_resource_error`、`_ceva_core_inventory`、`_ceva_usage_report` | T1、T8 |
| genesim/conf/ceva_single_core_4e.yaml 等 5 份 | 新增 | 1/64/128 核 × 4e/1e 完整场景与来源台账 | — | T1 |
| genesim/src/pim/pim_core.py | 修改 | 抽出内置 PIMTrace 求值核心，新增 CEVA-NPM 分派 | `PIMTracePIMCore`、`CEVANPMPIMCore`、`create_pim_core` | T2 |
| genesim/src/backend/pim_backend.py | 修改 | CEVA-NPM 后端分派 | `UnifiedPIMBackend._create_core` | T2 |
| genesim/src/pim/capacity.py | 修改 | 字节真源、GiB 一致性、系统预留、可用量 | `resolve_pim_capacity_limits`、`_ceva_capacity_limits` | T3 |
| genesim/src/pim/pimir_trace.py | 修改 | 相位工作量字段与循环编码、区间倍率透传 | `PhaseStep`、`_matmul_mac_count`、`_work_units_for`、`_emit_work`、`compile_phase_chain` | T5、T6 |
| genesim/src/pim/pim_isa.py | 修改 | 区间编码函数接受外部倍率表 | `zone_scale`、`zone_scale_for` | T6 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | CEVA 映射校验、哈希核验、覆盖检查、倍率签名、成熟度 | `_validate_ceva_dpu_mapping`、`_operator_pimir`、`_ceva_result_maturity` | T4、T6、T8 |
| genesim/src/backend/comm_lib.py | 修改 | HOST 中转开销从配置读取 | `CommLib.__init__`、`_host_relay` | T7 |
| genesim/scripts/export_fixed_pu_mapping.py | 修改 | CEVA 模式一台 DPU 独占一个 ClusterPU | `build_mapping` | T10 |
| flagos-pim-compiler/scripts/export_pp_placement.py | 修改 | 新增 `--mram-bytes`、`--sys-reserve-bytes` 并报告物理/预留/可用量 | `build_hw_budget` | T9 |
| flagos-pim-compiler/tests/test_ir_cost_eval.py | 修改 | 融合激活的查表对比改为按循环次数统计有效工作量 | `test_fused_activation_costs_one_lookup` | T5 |
| genesim/tests/sim/test_*.py | 修改/新增 | 覆盖上述行为的单元测试 | — | 全部 |
| flagos-pim-compiler/genesim_bridge/placement_export.py（round02） | 修改 | B 路 sidecar 条目写 `symbolic_dim_placeholder`，登记符号维在 PIM MLIR 里的占位值 | `_attach_bpath_pimir`、`_SYMBOLIC_DIM` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round02） | 修改 | 登记占位值、给相位轨迹写 `work_binding`/`shape_scale`、CEVA 严格化、验证报告成本摘要与产物类型核对 | `_phase_trace_binding`、`_prepare_pim_traces`、`_register_compiler_artifacts`、`_ceva_cost_model_summary`、`_ceva_cost_model_error`、`_ceva_artifact_dtype_error`、`_operator_pimir` | 问题1/2/3 |
| genesim/src/vpu/vpu.py（round02） | 修改 | 整算子相位轨迹按实际/占位比例缩放一次，不再重复整条轨迹；profile 使用同一缩放因子 | `_static_trace_scale`、`_runtime_shape_dimension`、`execute`、`estimate`、`execute_profile` | 问题1 |
| genesim/src/config_loader.py（round02） | 修改 | 正式运行强制严格产物开关；循环/同步原语列入必需成本键和来源路径 | `_validate_ceva`、`CEVA_REQUIRED_CYCLE_COST_KEYS`、`CEVA_REQUIRED_SOURCE_PATHS` | 问题2/4 |
| genesim/src/pim/pim_trace_loader.py（round02） | 修改 | 严格模式按轨迹实际 opcode 检查周期表缺项，拒绝内置默认值 | `_require_cycle_cost_coverage`、`estimate_total_cycles`、`estimate_cycle_breakdown` | 问题4 |
| genesim/src/pim/pim_core.py（round02） | 修改 | CEVA 核心启用严格周期表；其他核心保持旧行为 | `PIMTracePIMCore.STRICT_CYCLE_COSTS`、`CEVANPMPIMCore`、`_trace_latency` | 问题4 |
| genesim/conf/ceva_*.yaml（round02） | 修改 | 5 份场景补 LOOP_BEGIN/LOOP_BEGIN_INDIRECT/LOOP_BEGIN_INDIRECT_LONG/LOOP_END/SYNC 单价与来源 | — | 问题4 |
| genesim/src/scheduler/gene_sim_scheduler.py（round03） | 修改 | TP 展开按原 op_id 把符号维占位值继承给每个分片 | `_expand_multi_shard_operators` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round03） | 修改 | 报告精度绑定产物主计算张量，核对元素宽度、逐 op_type 布局与主计算精度 | `_ceva_primary_compute_dtype`、`_ceva_run_compute_dtypes`、`_ceva_artifact_dtype_error`、`_CEVA_DTYPE_ELEMENT_BYTES`、`_CEVA_COMPUTE_OPCODES` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round03） | 修改 | 验证报告新增独立验证证据核对，逐点复算误差汇总 | `_ceva_validation_evidence_error`、`_ceva_validation_scope_error` | 问题3 |
| genesim/src/config_loader.py（round03） | 修改 | `comm_config.host_relay_overhead_ns` 纳入必需来源路径 | `CEVA_REQUIRED_SOURCE_PATHS` | 问题4 |
| genesim/conf/ceva_*.yaml（round03） | 修改 | 5 份场景补中转开销的 origin/evidence/unit/scope，数值仍为 100 ns | — | 问题4 |
| genesim/src/pim/pimir_trace.py（round04） | 修改 | 相位保留逐张量形状，新增运行时重建模板与解析周期计算；MAC 收缩维按两个操作数判定 | `TensorSpan`、`PhaseStep.operand_tensors`、`phase_runtime_template`、`resolve_phase_template`、`phase_cycle_breakdown`、`_bulk_cycle_count`、`_work_cycle_count`、`_matmul_mac_count_from_shapes`、`compile_from_pimir` | 问题1 |
| genesim/src/pim/pim_trace_loader.py（round04） | 修改 | 抽出周期表解析，供指令求值与相位解析计算共用 | `resolve_cycle_costs`、`estimate_total_cycles`、`estimate_cycle_breakdown` | 问题1 |
| genesim/src/vpu/vpu.py（round04） | 修改 | 静态形状轨迹按实际形状解析周期，替换整条比例缩放；按算子与形状缓存结果 | `_static_phase_metrics`、`_phase_dimensions`、`_phase_dimension_value`、`execute`、`execute_profile`、`estimate` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round04） | 修改 | 生成相位运行时模板、trace schema 升 9；报告绑定请求 trace 与边界；样本绑定产物并检查覆盖 | `_phase_runtime_metadata`、`_prepare_pim_traces`、`_ceva_request_boundary`、`_ceva_validation_scope_error`、`_ceva_validation_evidence_error` | 问题1/2/3 |
| genesim/tests/sim/test_vpu.py（round04） | 修改 | 用按实际形状直接编译的轨迹作对照，覆盖 GEMV_SCORE/CONTEXT、SOFTMAX、SILU | `TestStaticPhaseTraceRuntime` | 问题1 |
| genesim/tests/sim/test_compiler_placement.py（round04） | 修改 | 模板元数据、运行时执行、请求边界与样本覆盖用例 | — | 问题1/2/3 |
| genesim/tests/sim/test_pim_backend.py（round04） | 修改 | 纳秒断言改为可区分周期的精度，循环用例按 12 ns 计 | — | 问题4 |
| flagos-pim-compiler/genesim_bridge/placement_export.py（round04） | 核对 | 占位值生产端保持 128，无需修改 | `_bpath_pimir` | 问题1 |
| genesim/src/config_loader.py（round05） | 修改 | 正式校验强制管线后端、自动调参最终验证为管线、按编译器本地分片执行 | `_validate_ceva` | 问题1/4 |
| genesim/src/scheduler/gene_sim_scheduler.py（round05） | 修改 | 结果成熟度绑定实际后端与结果产出方；运行期拒绝预测器和关闭本地分片 | `_ceva_result_maturity`、`run`、`_apply_compiler_local_shapes` | 问题1/4 |
| genesim/src/scheduler/gene_sim_scheduler.py（round05） | 修改 | 加载产物时核对计算原语、激活种类、PIM 目标、分块/放置字节、逐操作数与结果的张量形状 | `CEVA_PIM_MLIR_TARGET`、`CEVA_ARTIFACT_PRIMITIVES`、`CEVA_ARTIFACT_ACTIVATIONS`、`_validate_ceva_compiler_artifact`、`_validate_ceva_gemm_artifact`、`_validate_ceva_phase_artifact_shape`、`_ceva_tensor_operand_dims`、`_ceva_bound_last_dim`、`_ceva_bound_element_count`、`_mlir_int_attribute` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round05） | 修改 | 融合吸收必须绑定主算子、融合相位、形状与哈希 | `_ceva_fused_owners`、`_validate_ceva_fusion_exceptions`、`_register_compiler_artifacts` | 问题3 |
| genesim/src/scheduler/gene_sim_scheduler.py（round05） | 修改 | 验证样本的相位必须在产物里存在，GEMM 形状按 M/N/K、非 GEMM 形状按本次执行形状核对 | `_ceva_artifact_texts_by_op_type`、`_ceva_artifact_phases`、`_ceva_run_shapes_of`、`_ceva_sample_shape_error`、`_ceva_validation_evidence_error`、`_ceva_validation_scope_error` | 问题5 |
| genesim/tests/sim/test_compiler_placement.py（round05） | 修改 | 产物语义/目标/形状/字节、融合归属、后端成熟度、样本相位与形状用例；正向夹具改为 `tt.dot` GEMM 产物 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2/3/5 |
| genesim/tests/sim/test_config_loader.py（round05） | 修改 | 正式校验拒绝预测器后端、预测器最终验证与关闭本地分片 | `TestCevaProfile` | 问题1/4 |
| genesim/src/scheduler/gene_sim_scheduler.py（round06） | 修改 | 相位产物按执行节点逐轴核对操作数与结果；前导广播维和单行内核视图之外的形状差异、形状无法解析、签名缺张量操作数均拒绝 | `_ceva_bound_shape`、`_ceva_tensor_operand_shapes`、`_ceva_artifact_axis_error`、`_ceva_artifact_shape_matches`、`_validate_ceva_phase_artifact_shape` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round06） | 修改 | 验证报告登记配置核数与已用核数，成熟度核对资源清点、放置计划与运行期实际使用核数 | `_ceva_created_core_count`、`_ceva_plan_used_core_count`、`_ceva_active_core_count`、`_ceva_validation_scope_error` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round06） | 修改 | 样本形状按请求 trace 展开 prefill/decode 真实执行形状，GEMM 的 M 也按阶段核对 | `_ceva_runtime_shape_contexts`、`_ceva_run_shapes_of`、`_ceva_run_matmul_stage_m`、`_ceva_sample_shape_error`、`_ceva_validation_evidence_error` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round06） | 修改 | 新增 13 条用例覆盖产物秩/前导轴/不可解析形状、核数范围、decode 运行时 KV 形状；正向夹具补核数字段并把 GEMV 产物改为逐轴一致 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2/3 |
| genesim/src/scheduler/gene_sim_scheduler.py（round07） | 修改 | `_ceva_artifact_axis_error` 增加 `strict`：末维不同在严格模式返回明确错误，结果列表以 `strict=False` 跳过不可比中间结果；输入操作数与执行节点输入个数一一对应时强制明确匹配，多出的参数/表格张量按 K18 约定不参与逐轴核对 | `_ceva_artifact_axis_error`、`_validate_ceva_phase_artifact_shape` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round07） | 修改 | SPLIT 产物的元素总数核对不再允许空形状列表或动态维跳过：预期形状无法绑定、无可解析静态形状、含未知维均直接拒绝 | `_validate_ceva_phase_artifact_shape`、`_MLIR_TENSOR_SHAPE_RE` | 问题2 |
| genesim/src/pim/pimir_trace.py（round07） | 修改 | 相位解析统计字节时，若类型文本里的 `tensor<` 数量与可解析静态形状数量不一致（如动态维），直接报错而不是按 0 计 | `_tensor_totals` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round07） | 修改 | 验证报告的绝对相对误差 p90 由逐点样本按最近秩复算；最近秩函数上提为模块级 `_nearest_rank`，运行指标的 p50/p95 共用同一实现；报告契约版本升到 2 | `_nearest_rank`、`_ceva_validation_evidence_error`、`_execute_pipeline`、`CEVA_VALIDATION_SCHEMA_VERSION` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round07） | 修改 | 新增 7 条用例：轴函数末维不同必须报错、MEM_COPY 输入形状不符、结果正确但输入错误、输入错误在 trace 生成前拒绝、SPLIT 动态形状、p90 中间值与线性插值值均拒绝、旧版报告拒绝 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2/3 |
| genesim/tests/sim/test_pimir_trace.py（round07） | 修改 | 新增 2 条用例：单相算子与带 phase_spec 的相位遇到动态维时都报错 | `TestParsePhaseChain` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round08） | 修改 | 请求边界按 `is_prefill` 区分 decode 步数：prefill 为 `output_length-1`，decode-only 为 `output_length`；`max_query_length` 改为各请求实际前向 Tq 的最大值，decode-only 为 1 | `_ceva_request_boundary` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round08） | 修改 | 实际使用核数无法确定时直接拒绝升级成熟度，不再跳过核对；实际值与放置计划已用核数不一致仍保持 `model_prediction` | `_ceva_validation_scope_error`、`_ceva_active_core_count` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round08） | 修改 | 新增 7 条用例：decode-only JSON/二进制边界、混合请求边界、decode-only 步数写错的报告不能升级、缺少运行期核映射、映射未落在 PIM 核、实际核数多于计划；正向夹具补与计划一致的 `op_to_vpu_id` | `TestCevaPlacementMapping` | 问题1/2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round09） | 修改 | 相位产物按实际语句解析计算原语，混入与执行节点矛盾的计算原语直接拒绝；SOFTMAX/QUANT 登记允许的融合相位 | `CEVA_ARTIFACT_COMPUTE_PRIMITIVES`、`CEVA_ARTIFACT_FUSED_PHASES`、`_MLIR_STATEMENT_PRIMITIVE_RE`、`_validate_ceva_compiler_artifact` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round09） | 修改 | 最终结果绑定到 `tt.return` 的操作数；无操作数时取最后一条带结果的语句；多结果按位置逐项核对，数量不一致直接拒绝 | `_MLIR_RETURN_RE`、`_MLIR_RESULT_STATEMENT_RE`、`_ceva_returned_shapes`、`_validate_ceva_phase_artifact_shape` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round09） | 修改 | 计时边界新增 `max_inflight_requests` 并逐项与本次配置比对；报告契约升到 `schema_version=3` | `CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_validation_scope_error` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round09） | 修改 | 新增 8 条用例：混入矛盾计算原语（含 trace 生成前拒绝）、最终返回错误（显式/无操作数返回）、多结果位置对应、写入类产物无返回、并发上限正反例、报告缺并发上限、旧版 schema；夹具升到 3 并补计时边界字段 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2/3 |
| genesim/src/scheduler/gene_sim_scheduler.py（round10） | 修改 | 产物语句按执行节点完整允许集合核对：语义原语、登记的融合相位和 GEMM 必需的 barrier/DMA/WRAM 基础设施之外一律拒绝，无关搬运不再进入计时 | `CEVA_ARTIFACT_SUPPORT_PRIMITIVES`、`_validate_ceva_compiler_artifact` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round10） | 修改 | 成熟度派生新增参数来源核对：必需来源路径缺失、证据为空、来源为假设或未确认时保持 `model_prediction` | `CEVA_CALIBRATED_SOURCE_ORIGINS`、`CEVA_REQUIRED_SOURCE_PATHS`、`_ceva_parameter_source_error`、`_ceva_result_maturity` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round10） | 修改 | 新增 7 条用例：SILU 混入 `pim.kv_cache` 拒绝、真实 GEMM 基础设施语句放行、全假设/单假设/缺来源/缺证据/未确认来源均保持 `model_prediction`；成熟度夹具补完整实测来源台账 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round11） | 修改 | 新增单行内核算子登记；单行/单列视图豁免只在登记算子（GATHER、QUANT）上生效，其余算子的位置对应输入与最终结果必须逐轴一致 | `CEVA_SINGLE_ROW_KERNEL_OPS`、`_ceva_artifact_axis_error`、`_ceva_artifact_shape_matches`、`_validate_ceva_phase_artifact_shape` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round11） | 修改 | 链路规范化纳入 `intra_cluster_only`、`inter_cluster_only`、`is_full_duplex`（缺省与运行时 `LinkConfig` 一致），相等判断逐项比较；验证报告契约升到 4 | `_normalized_links`、`_links_equal`、`CEVA_VALIDATION_SCHEMA_VERSION` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round11） | 修改 | 独立验证样本的预测周期必须为有限正数，负值和零值不能作为校准证据 | `_ceva_validation_evidence_error` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round11） | 修改 | 新增 7 条用例：SILU 单行输入/输出/加载前拒绝、登记后的 QUANT 单行内核与 GATHER 单行结果放行、链路三个语义字段逐一改动后保持 `model_prediction`、负/零预测保持 `model_prediction`；报告夹具补链路语义字段并升到 `schema_version=4` | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2/3 |
| genesim/src/config_loader.py（round12） | 修改 | 主机芯片参数纳入必需来源路径；正式校验要求显式 `cpu_model` 有限正数、拒绝非 PIM 权重部署 | `CEVA_REQUIRED_SOURCE_PATHS`、`_validate_ceva` | 问题1/3 |
| genesim/src/scheduler/gene_sim_scheduler.py（round12） | 修改 | 主机基础延迟、算力、带宽与核数纳入成本模型摘要并逐项比对，验证报告契约升到 5 | `CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_cpu_model_summary`、`_ceva_cost_model_summary`、`_ceva_cost_model_error` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round12） | 修改 | 主机编排节点类型识别、主机独立验证证据核对（覆盖、正数、逐点复算）、误差汇总核对抽为共用函数 | `_ceva_host_orchestration_op_types`、`_ceva_host_evidence_error`、`_ceva_error_summary_error`、`_ceva_validation_evidence_error` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round12） | 修改 | 主机出入口与归约开销按 CPU 设备单列，结果新增 `host_orchestration_metrics` | `_ceva_host_orchestration_metrics`、`_execute_pipeline` | 问题1 |
| genesim/src/main.py（round12） | 修改 | `summary.json` 仅对 CEVA 写入主机编排服务时间、存储时间、字节数与算子数，文本摘要单列主机服务时间 | `save_all_results`、`_existing_summary_lines` | 问题1 |
| genesim/src/pim/pimir_trace.py（round12） | 修改 | 相位链分支透传 `max_bulk_transfer_bytes`，不再固定用内置 16320 B | `compile_from_pimir` | 问题2 |
| genesim/conf/ceva_*.yaml（round12） | 修改 | 5 份场景显式声明 `cpu_model` 与纯 PIM `weight_deployment`，补 4 条主机参数来源 | — | 问题1/3 |
| genesim/tests/sim/test_config_loader.py（round12） | 修改 | 新增 6 条用例覆盖显式主机成本、数值校验、来源路径、五份场景主机来源与纯 PIM 权重部署 | `TestCevaProfile` | 问题1/3 |
| genesim/tests/sim/test_compiler_placement.py（round12） | 修改 | 新增 6 条用例覆盖主机成本绑定、主机证据缺失/不匹配/齐全、主机指标单列；报告夹具升到 `schema_version=5` 并补 `cpu_model` | `TestCevaPlacementMapping` | 问题1 |
| genesim/tests/sim/test_main.py（round12） | 修改 | 新增 3 条用例覆盖主机服务时间单列、通用摘要不变、`summary.json` 主机字段 | `TestHostOrchestrationSummary` | 问题1 |
| genesim/tests/sim/test_pimir_trace.py（round12） | 修改 | 新增 1 条用例：小分段上限下指令立即数与运行时解析公式完全一致，缺省上限行为不变 | `TestPhaseWorkload` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round13） | 修改 | 记录本次实际执行的放置图；主机节点识别、独立验证证据与主机指标读取该图而不是展开前的 `model_ir`；证据算子集合排除合成中转，指标新增逐类型计数与零成本类型标注 | `_execution_ir_or_model`、`_ceva_host_orchestration_op_types`、`_ceva_host_evidence_op_types`、`_ceva_host_evidence_error`、`_ceva_host_orchestration_metrics`、`_execute_pipeline` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round13） | 修改 | 成本模型摘要与逐项比对纳入整核峰值参数 `multipliers`；验证报告契约升到 `schema_version=6` | `CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_cost_model_summary`、`_ceva_cost_model_error` | 问题2 |
| genesim/src/backend/cpu_backend.py（round13） | 修改 | `COLLECTIVE_RELAY` 在 CPU 执行路径返回零计算、零存储、零开销剖面，与 GPU 语义一致 | `CPUModel.profile` | 问题3 |
| genesim/src/main.py（round13） | 修改 | `summary.json` 增加主机编排逐类型计数与零成本类型，便于区分模型出入口、归约和合成中转 | `save_all_results` | 问题1 |
| genesim/tests/sim/test_compiler_placement.py（round13） | 修改 | 新增 6 条用例：执行图主机节点识别与逐类型计数、缺 ALL_REDUCE 证据保持预测、覆盖全部真实主机节点可升级、合成中转不能作为实测样本、缺 `multipliers` 保持预测、摘要与峰值共同绑定 `multipliers`；报告夹具升到 `schema_version=6` 并补 `multipliers` | `TestCevaPlacementMapping` | 问题1/2 |
| genesim/tests/sim/test_collective_edges.py（round13） | 修改 | 新增 1 条纯 PIM 联合用例：中转节点落在 CPU 时执行剖面为零、跨核边成本仍非零 | `TestCollectiveEdges` | 问题3 |
| genesim/tests/sim/test_cpu_backend.py（round13） | 修改 | 新增 1 条用例：`COLLECTIVE_RELAY` 的 CPU 执行与剖面全为零 | `TestCPUModel` | 问题3 |
| genesim/tests/sim/test_main.py（round13） | 修改 | 断言 `summary.json` 持久化主机逐类型计数与零成本类型 | `TestHostOrchestrationSummary` | 问题1 |
| genesim/src/pim/pimir_trace.py（round14） | 修改 | 按步长与 `iter_args` 识别 M/N/K 循环及折叠形态；记录 `m_extent`/`m_iterations`；MAC 与 K 迭代用混合进制嵌套循环表达完整工作量；删除 `_capped` | `_ScfLoop`、`PimirProgram`、`parse_pimir`、`_emit_work_block`、`_emit_work`、`compile_from_pimir`、`trace_metadata` | 问题1 |
| genesim/src/vpu/vpu.py（round14） | 修改 | GEMM 的重复次数按 `ceil(Tq / pim.tile-m)` 计算，大 M 轨迹不再按 Tq 重复整条轨迹；执行剖面的 DMA 字节用同一重复次数 | `_execute_runtime`、`_effective_repetitions`、`_runtime_tile_m`、`execute_profile` | 问题1 |
| genesim/tests/sim/test_pimir_trace.py（round14） | 修改 | 新增三层 M/N/K、M+K 两层、全展开、非法步长、大 MAC/K、M/N/K 加倍与真实大 M 产物用例；替换超限拒绝用例 | `TestParsePimir`、`TestCompileFromPimir`、`_pimir_mnk`、`_pimir_unrolled`、`_pim_compiler_root` | 问题1 |
| genesim/tests/sim/test_vpu.py（round14） | 修改 | 新增大 M 轨迹按分块重复、缺元数据保持 Tq 重复、剖面 DMA 字节按分块重复三条用例 | `TestPIMVpu` | 问题1 |
| flagos-pim-compiler/genesim_bridge/placement_export.py（round14） | 修改 | sidecar 升到 version 4，顶层导出物理 MRAM、系统预留与可用字节；新增 `sys_reserve_bytes` 参数与范围校验 | `export_placement_to_genesim` | 问题2 |
| flagos-pim-compiler/scripts/export_pp_placement.py（round14） | 修改 | 把 `HwBudget.sys_reserve_bytes` 与 `PIMHardwareConfig` 的 MRAM 一起写进 sidecar | `main` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round14） | 修改 | CEVA 加载时核对 sidecar `hardware` 三项与画像逐项一致；GEMM 产物要求并核对 `pim.mram-bytes` | `_validate_ceva_hardware_budget`、`_load_compiler_placement`、`_validate_ceva_gemm_artifact` | 问题2 |
| flagos-pim-compiler/tests/test_placement_export.py（round14） | 修改 | 断言 version 4 与 `hardware` 三项；新增预留越界反例 | `test_sidecar_exports_the_same_byte_budget`、`test_sidecar_rejects_a_reserve_larger_than_capacity` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round14） | 修改 | 新增 sidecar 预算缺失/物理量/预留/可用量不一致与 GEMM 产物预算缺失/不一致用例；CEVA 夹具补预算字段 | `TestCevaPlacementMapping`、`TestCevaArtifactGate`、`TestPimirHashVerification`、`TestStaticPhaseTraceBinding` | 问题2 |
| genesim/tests/sim/test_collective_edges.py（round14） | 修改 | CEVA 用例的 sidecar 夹具补 version 4 与画像是同一份预算 | `_sidecar` | 问题2 |
| genesim/src/config_loader.py（round15） | 修改 | `pim.inter_cluster_communication` 纳入正式运行必需来源；路由模式只接受 `direct`/`host_relay` | `CEVA_REQUIRED_SOURCE_PATHS`、`_validate_ceva` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round15） | 修改 | GEMM 产物核对 sidecar `kernel_tile_n` 与 `pim.tile-n`；验证报告核对路由模式；契约升到 7 | `_validate_ceva_gemm_artifact`、`_ceva_validation_scope_error`、`CEVA_VALIDATION_SCHEMA_VERSION` | 问题1/2 |
| genesim/src/vpu/vpu.py（round15） | 修改 | CEVA GEMM 运行时按轨迹元数据 `pimir.tile_n` 复核执行分块 | `_check_gemm_artifact_tile`、`_execute_runtime` | 问题1 |
| genesim/conf/ceva_*.yaml × 5（round15） | 修改 | 五份场景登记 `pim.inter_cluster_communication` 的假设来源 | `ceva.parameter_sources` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round15） | 修改 | 新增 GEMM 分块缺失/不一致与验证报告路由模式互换/缺失/链路组合不一致用例；报告夹具升到 `schema_version=7` 并补路由字段 | `TestCevaArtifactGate`、`TestCevaPlacementMapping` | 问题1/2 |
| genesim/tests/sim/test_config_loader.py（round15） | 修改 | 新增五份场景路由来源、缺来源与非法路由模式用例 | `TestCevaProfile`、`_ceva_profile` | 问题2 |
| genesim/tests/sim/test_vpu.py（round15） | 修改 | 新增大 M 分块产物与真实大 M 产物的 VPU 端到端用例，断言 N 循环、MAC 与 DMA 按产物分块 | `TestStaticPhaseTraceRuntime` | 问题1 |
| genesim/src/config_loader.py（round17） | 修改 | CEVA 正式校验要求 CPU→HOST 与 HOST→CPU 双向链路，缺任一条即拒绝；结构校验仍允许先缺链路接线 | `_validate_ceva` | 问题1 |
| genesim/tests/sim/test_config_loader.py（round17） | 修改 | 新增分别删除 CPU→HOST/HOST→CPU 的正式校验反例，并断言合法配置的 CPU↔PIM 中转成本有限 | `TestCevaProfile` | 问题1 |
| genesim/src/main.py（round17） | 修改 | `result_maturity` 仅在结果实际包含该字段时写入摘要，通用摘要键集合保持不变 | `save_all_results` | 问题2 |
| genesim/tests/sim/test_main.py（round17） | 修改 | 新增通用摘要不含成熟度字段、CEVA 摘要保留 `model_prediction`/`validated_in_scope` 的回归用例 | `TestResultMaturitySummary` | 问题2 |
| genesim/src/config_loader.py（round18） | 修改 | 非有限容量在结构/正式校验中返回失败；新增 `resolved_config()`；CEVA 正式校验要求已声明的成本 sidecar 存在且可读 | `_validate_ceva`、`resolved_config` | 问题1/2/3 |
| genesim/src/pim/capacity.py（round18） | 修改 | CEVA GiB 换算先做 `math.isfinite`，非有限值抛带字段名的 `ValueError`，并把 `OverflowError` 纳入异常处理 | `_ceva_capacity_limits` | 问题3 |
| genesim/src/run_manifest.py（round18） | 新增 | 生成 CEVA 运行清单并核对工具链身份：输入产物哈希、唯一 PIM MLIR 哈希、图编译器与 FlagTree 提交号/脏标记、实际 `triton-opt` 路径与 SHA256、硬件预算、解析后配置 | `build_ceva_run_manifest`、`compiler_toolchain_identity`、`validate_ceva_toolchain`、`artifact_entry`、`pimir_resolver` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round18） | 修改 | CEVA 加载端对缺失/不可读成本 sidecar 直接报错；运行结束生成并挂载运行清单 | `_load_cost_sidecar`、`_ceva_run_manifest`、`run` | 问题1/2 |
| genesim/src/main.py（round18） | 修改 | 创建 Scheduler 前核对工具链身份与 PIM MLIR 哈希；结果目录写 `run_manifest.json`，摘要登记文件名 | `main`、`save_all_results` | 问题2 |
| genesim/tests/sim/test_config_loader.py（round18） | 修改 | 新增缺失/合法成本 sidecar 与正负无穷/NaN 容量校验用例 | `TestCevaProfile` | 问题1/3 |
| genesim/tests/sim/test_pim_capacity.py（round18） | 修改 | 新增非有限 GiB 容量解析反例 | `TestPIMCapacity` | 问题3 |
| genesim/tests/sim/test_cost_sidecar.py（round18） | 修改 | 新增 CEVA 缺失成本 sidecar 报错、通用路径保留告警、合法成本 sidecar 标注算子的用例 | `TestCostSidecar` | 问题1 |
| genesim/tests/sim/test_run_manifest.py（round18） | 新增 | 覆盖清单内容与实际文件一致、PIM MLIR 哈希变化、FlagTree 源码版本缺失、`triton-opt` 缺失、PIM MLIR 不在源码仓库内 | `TestCevaRunManifest` | 问题2 |
| genesim/tests/sim/test_main.py（round18） | 修改 | 新增 CEVA 清单落盘与通用结果不写清单的用例 | `TestResultMaturitySummary` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round19） | 修改 | 三个 CEVA 运行形状入口改读 TP 展开后的执行图，全局模板形状不再作为验证范围 | `_ceva_run_shapes_of`、`_ceva_run_matmul_stage_m`、`_ceva_run_matmul_shape_bounds` | 问题1 |
| genesim/src/config_loader.py（round19） | 修改 | 运行清单解析配置应用 `GENESIM_*` 环境覆盖，类型推断与 `get()` 一致 | `resolved_config`、`_apply_env_overrides` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round19） | 修改 | 新增 Llama2-7B TP2 夹具与三条用例：本地形状范围、全局 11008 样本拒绝、四种真实本地分片样本接受 | `TestCevaPlacementMapping` | 问题1 |
| genesim/tests/sim/test_run_manifest.py（round19） | 修改 | 新增时钟、路径、布尔与核数环境覆盖同时进入执行配置与清单的用例 | `TestCevaRunManifest` | 问题2 |
| genesim/src/pim/pimir_trace.py（round22） | 修改 | 抽出 M/K/N 形状解析，新增按输出取整的 MAC 分组；相位编译与运行时解析共用同一工作量公式 | `_matmul_dims_from_shapes`、`_matmul_mac_count_from_shapes`、`_matmul_mac_groups_from_shapes`、`_phase_work_units` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round22） | 修改 | 成本 sidecar 覆盖只认有效搬运成本与显式模板路径；空条目和仅来源名条目保持未覆盖；报告契约升到 10；指令发射语义变化使 trace 编译签名升到 12 | `_build_cost_sidecar_ledger`、`_ceva_cost_sidecar_error`、`CEVA_VALIDATION_SCHEMA_VERSION`、`base_compile_signature` | 问题1/2 |
| genesim/src/pim/pim_core.py（round22） | 修改 | CEVA 内置核心的长循环接口按 `outer×inner` 求总次数，复用短轨迹升级、长循环编码与 24 位越界检查 | `PIMTracePIMCore.set_tile_count_long`、`PIMTracePIMCore.execute` | 问题3 |
| genesim/tests/sim/test_pimir_trace.py（round22） | 修改 | 新增非整除 K、transposeB、带 scale 量化矩阵乘与运行时模板对拍用例 | `TestPhaseWorkload` | 问题1 |
| genesim/tests/sim/test_cost_sidecar.py（round22） | 修改 | 覆盖“有效成本 + 空条目”“仅来源名”两类台账，并把空条目的未覆盖期望改正 | `TestCostSidecar` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round22） | 修改 | 新增混合空条目、仅来源名不可升级的成熟度用例；报告夹具升到 `schema_version=10`，覆盖范围按有效成本复算 | `TestCevaPlacementMapping` | 问题2 |
| genesim/tests/sim/test_pim_backend.py（round22） | 修改 | 新增后端长循环接口 4095/4096/5000、非法因子、24 位越界与剖面一致性用例 | `TestCevaNpmBackend` | 问题3 |
| genesim/src/run_manifest.py（round23） | 修改 | 新增输入快照复核与完整性检查；清单在有快照时只引用消费时的路径与哈希，不再重读运行结束时的磁盘内容 | `verify_input_snapshot`、`_verify_snapshot_entry`、`_require_snapshot_inputs`、`build_ceva_run_manifest` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round23） | 修改 | 五个输入入口登记消费快照；放置 sidecar 缓存；成本台账按解析字节记账；成熟度/清单改读快照；成本 sidecar 统一数值域校验；运行前锁定输入身份 | `load_model_ir`、`load_trace`、`_record_input_snapshot`、`_record_pimir_snapshot`、`_ensure_input_snapshot`、`_compiler_sidecar`、`_load_compiler_placement`、`_register_compiler_artifacts`、`_load_cost_sidecar`、`_cost_sidecar_consumption`、`_build_cost_sidecar_ledger`、`_validate_ceva_cost_sidecar`、`_validate_ceva_mram_value`、`_ceva_cost_sidecar_error`、`_ceva_validation_scope_error`、`_ceva_run_manifest`、`run` | 问题1/2 |
| genesim/src/config_loader.py（round23） | 修改 | 新增主机参数共享数值契约；CEVA 正式校验读生效配置并检查 CPU 核数 | `ceva_host_parameter_errors`、`_validate_ceva` | 问题3 |
| genesim/src/backend/cpu_backend.py（round23） | 修改 | CEVA 的 CPUModel 构造边界复用同一契约并在读取参数前失败；最小配置对象透传目标类型 | `CPUModel.__init__`、`CPUModel._from_dict`、`_is_ceva_config` | 问题3 |
| genesim/src/vpu/factory.py（round23） | 修改 | CEVA 工厂使用生效 `cpu_model` 与核数构造 CPUModel，避免配置校验和执行使用两套取值 | `_create_cpu_model` | 问题3 |
| genesim/tests/sim/test_cost_sidecar.py（round23） | 修改 | 新增标量/分档/measurements 三种形态的非法搬运字节反例，以及零值、None、有限值正例 | `TestCostSidecar` | 问题2 |
| genesim/tests/sim/test_compiler_placement.py（round23） | 修改 | 新增成本改写后成熟度保持预测、五类输入快照覆盖、成本与 PIM MLIR 改写后清单失败用例 | `TestCevaPlacementMapping` | 问题1 |
| genesim/tests/sim/test_run_manifest.py（round23） | 修改 | 新增清单引用消费快照、成本/PIM MLIR 变化拒绝、快照缺项拒绝用例 | `TestCevaRunManifest` | 问题1 |
| genesim/tests/sim/test_config_loader.py（round23） | 修改 | 新增 YAML/override/环境覆盖三条入口的主机参数与 CPU 核数反例 | `TestCevaProfile` | 问题3 |
| genesim/tests/sim/test_cpu_backend.py（round23） | 修改 | 新增 CPUModel 直接构造与 VPU 工厂的 CEVA 数值契约、合法对照用例 | `TestCevaCPUModelContract` | 问题3 |
| genesim/src/numeric_utils.py（round24） | 新增 | 外部数值统一转换边界：类型不符、NaN/Inf 与任意精度整数溢出统一返回 None | `finite_float` | 问题3 |
| genesim/src/ir/model_ir.py（round24） | 修改 | 新增按字节解析入口，文件读取与解析拆开，供解析和哈希共用同一份字节 | `ModelIR.from_bytes`、`ModelIR.load` | 问题1 |
| genesim/src/ir/trace_ir.py（round24） | 修改 | JSON/二进制 trace 增加按字节解析入口，原文件加载接口复用 | `Trace.from_json_bytes`、`Trace.from_binary_bytes`、`Trace.load_json`、`Trace.load_binary` | 问题1 |
| genesim/src/scheduler/base_scheduler.py（round24） | 修改 | 模型 IR 与请求 trace 均只读一次字节，解析后通过钩子登记同一份内容；trace 格式探测改用内存字节 | `_register_loaded_input`、`load_model_ir`、`load_trace` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round24） | 修改 | CEVA 快照按同一次读取的字节登记；请求边界优先使用已加载 trace；报告数值、成本分档与预测器辅助转换复用安全边界 | `_register_loaded_input`、`_load_configured_trace`、`_ceva_request_boundary`、`_finite_number`、`_validate_ceva_mram_value`、`positive_float` | 问题1/3 |
| genesim/src/run_manifest.py（round24） | 修改 | 新增按已读字节生成产物身份，不再重新打开文件 | `artifact_entry_from_bytes` | 问题1 |
| genesim/src/config_loader.py（round24） | 修改 | CEVA 禁止 `cpus` 列表；主机参数、autotune、周期成本、时钟与中转开销改用安全数值边界；legacy 容量迁移接住溢出 | `ceva_flat_resource_errors`、`ceva_host_parameter_errors`、`_validate_ceva`、`_validate_zone_cycle_factors`、`validate` | 问题2/3 |
| genesim/src/vpu/factory.py（round24） | 修改 | CEVA 完整工厂在任何资源循环前统一检查资源入口与主机参数，零/负数量不再跳过校验 | `_ceva_resource_contract_errors`、`create_vpus_and_topology` | 问题2 |
| genesim/src/main.py（round24） | 修改 | 运行后清点同时核对实际 CPU 数量与 `system.num_cpus` | `_ceva_resource_error` | 问题2 |
| genesim/src/pim/pim_isa.py（round24） | 修改 | 区间倍率校验改用安全转换，并先判 2× 是否有限再取整 | `validate_zone_cycle_factors` | 问题3 |
| genesim/src/vpu/vpu.py（round24） | 修改 | VPU 峰值辅助参数转换接住大整数溢出并回落默认值 | `PIMVPU._positive_float` | 问题3 |
| genesim/src/pim/capacity.py（round24） | 修改 | 容量、TensorPU 数量与 legacy 限额改用安全转换，溢出按 `ValueError` 返回校验失败 | `resolve_pim_capacity_limits` | 问题3 |
| genesim/src/pim/pim_compiler.py（round24） | 修改 | RMSNORM epsilon 校验改用安全转换 | `compile_rmsnorm_phase` | 问题3 |
| genesim/src/performance_objective.py、src/autotune/dataset_observer.py、src/autotune/placement.py、src/predictor/dataset.py、src/predictor/benchmark.py（round24） | 修改 | 同根因旁路：性能目标数值、数据集观察值、放置权重、预测器特征与成功样本判断不再抛未包装的 `OverflowError` | `performance_objective._finite_number`、`dataset_observer._finite_or_none`、`PlacementSearchSpace.__init__`、`predictor.dataset._positive_number`、`_successful_evaluation` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round24） | 修改 | 新增模型 IR/请求 trace 解析-登记窗口改写反例、边界证据读消费 trace、报告数值大整数反例 | `TestCevaPlacementMapping` | 问题1/3 |
| genesim/tests/sim/test_config_loader.py（round24） | 修改 | 新增 `cpus` 列表拒绝、周期成本/倍率/时钟/中转开销大整数反例、容量字段大整数反例 | `TestCevaProfile`、`TestGeneSimConfig` | 问题2/3 |
| genesim/tests/sim/test_cpu_backend.py（round24） | 修改 | 新增完整工厂零/负核数、`cpus`/`gpus` 列表拒绝与合法单 CPU 对照 | `TestCevaCPUModelContract` | 问题2 |
| genesim/tests/sim/test_main.py（round24） | 修改 | 新增实际 CPU 数量与配置不一致的资源清点反例，既有夹具补充 CPU 资源 | `TestCevaResourceInventory` | 问题2 |
| genesim/tests/sim/test_cost_sidecar.py（round24） | 修改 | 非法搬运字节反例新增 `10**1000` 大整数 | `TestCostSidecar` | 问题3 |
| genesim/tests/sim/test_zone_cycles.py（round24） | 修改 | 区间倍率反例新增大整数与 `1e308`（2× 溢出） | `TestZoneCycles` | 问题3 |
| genesim/tests/sim/test_vpu.py（round24） | 修改 | 新增 VPU 峰值辅助转换的大整数反例与合法对照 | `TestPIMVpu` | 问题3 |
| genesim/src/backend/comm_lib.py（round25） | 修改 | 新增链路数值契约与统一转换；CEVA 构造边界拒绝非法链路；全双工零带宽返回无穷 | `link_field_errors`、`_link_number`、`CommLink.from_dict`、`CommLink.concurrent_latency`、`CommLib.add_links` | 问题1 |
| genesim/src/config_loader.py（round25） | 修改 | 新增 CEVA 链路契约聚合；正式校验校验链路数值域与成本表全部条目 | `ceva_comm_link_errors`、`_validate_ceva` | 问题1/2 |
| genesim/src/vpu/factory.py（round25） | 修改 | 工厂构造前复用链路契约；链路构造统一走 `CommLink.from_dict` | `_ceva_resource_contract_errors`、`_make_comm_link` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round25） | 修改 | 验证报告链路比较复用数值契约；归约时长拒绝非法 `REDUCE_PARTIAL` | `_normalized_links`、`_estimate_reduction_time` | 问题1/2 |
| genesim/src/pim/pim_trace_loader.py（round25） | 修改 | 严格模式的周期成本数值域下沉到共享解析器，轨迹求值入口透传 strict | `_strict_cycle_cost`、`resolve_cycle_costs`、`estimate_total_cycles`、`estimate_cycle_breakdown` | 问题2 |
| genesim/src/pim/pimir_trace.py（round25） | 修改 | 相位解析严格模式复用共享周期表数值契约 | `phase_cycle_breakdown` | 问题2 |
| genesim/tests/sim/test_comm_lib.py、test_config_loader.py、test_cpu_backend.py、test_compiler_placement.py、test_pim_trace_loader.py、test_pimir_trace.py、test_pim_backend.py、test_scheduler.py、test_vpu.py（round25） | 修改 | 新增链路与周期成本数值契约的失败/合法对照用例 | — | 问题1/2 |
| genesim/src/ir/trace_ir.py（round26） | 修改 | 新增三字段形状绑定；请求与 JSON/二进制 trace 保留 padding 查询与初始 KV；旧版 trace 读取兼容 | `RequestShapeBinding`、`Request.shape_binding`、`Request.__post_init__`、`Request.to_dict`、`Request.from_dict`、`Trace.save_binary`、`Trace.from_binary_bytes` | 问题1 |
| genesim/src/scheduler/gene_sim_scheduler.py（round26） | 修改 | 请求边界、验证展开、运行前向、容量与成本聚合统一读形状绑定；CEVA 歧义显式拒绝；报告契约升到 11；报告核对与 DPU 映射删除资源数量 `int()` 合法化 | `_ceva_request_boundary`、`_ceva_runtime_shape_contexts`、`_ceva_runtime_query_bounds`、`_get_operator_with_runtime`、`_execute_pipeline`（`step_context`）、`_build_adaptive_initial_placement`、请求指标、`_ceva_validation_scope_error`、`_validate_ceva_dpu_mapping` | 问题1/2 |
| genesim/src/pim/capacity.py（round26） | 修改 | 峰值驻留上下文复用同一形状绑定，padding 与初始 KV 不再混用 | `_request_runtime_contexts` | 问题1 |
| genesim/src/config_loader.py（round26） | 修改 | 新增资源数量共享整数契约与目标名 engine 后缀核对；环境覆盖按配置原值推断类型 | `ceva_resource_count_errors`、`_validate_ceva`、`GeneSimConfig.get`、`GeneSimConfig._raw_value` | 问题2 |
| genesim/src/vpu/factory.py（round26） | 修改 | 完整工厂在任何资源循环前复用资源数量契约 | `_ceva_resource_contract_errors` | 问题2 |
| genesim/src/main.py（round26） | 修改 | 运行后清点复用同一契约并删除 `int()` 合法化 | `_ceva_resource_error`、`_ceva_core_inventory` | 问题2 |
| genesim/src/run_manifest.py（round26） | 修改 | 运行清单硬件预算复用同一契约并删除 `int()` 合法化 | `build_ceva_run_manifest` | 问题2 |
| genesim/src/pim/pu_pool.py（round26） | 修改 | CEVA 的 PU 池在按资源层级相乘前复用整数契约 | `create_pu_pool` | 问题2 |
| genesim/src/backend/comm_lib.py（round26） | 修改 | 链路契约增加端点与三个语义标记的布尔校验 | `link_field_errors` | 问题3 |
| genesim/tests/sim/test_trace_ir.py（round26） | 修改 | 新增形状绑定正反例与 JSON/二进制往返用例 | `TestRequest` | 问题1 |
| genesim/tests/sim/test_compiler_placement.py（round26） | 修改 | 新增 padding prefill、历史 KV、混合请求、样本形状、运行绑定与报告契约用例 | `TestCevaPlacementMapping` | 问题1 |
| genesim/tests/sim/test_config_loader.py（round26） | 修改 | 新增资源数量 YAML/override/环境覆盖反例、合法组合与链路标记反例 | `TestCevaProfile` | 问题2/3 |
| genesim/tests/sim/test_cpu_backend.py（round26） | 修改 | 完整工厂资源数量类型反例 | `TestCevaCPUModelContract` | 问题2 |
| genesim/tests/sim/test_main.py（round26） | 修改 | 运行后清点资源数量类型反例 | `TestCevaResourceInventory` | 问题2 |
| genesim/tests/sim/test_run_manifest.py（round26） | 修改 | 运行清单资源数量类型反例与整数环境覆盖正例 | `TestCevaRunManifest` | 问题2 |
| genesim/tests/sim/test_comm_lib.py（round26） | 修改 | 链路标记/端点反例与合法布尔双工、路由对照 | `TestCevaLinkContract` | 问题3 |
| genesim/tests/sim/test_scheduler.py（round26） | 修改 | 通用歧义 trace 保持旧 `seq_len` 约定的对照用例 | `TestGeneSimScheduler` | 问题1 |
| genesim/tests/sim/test_pu_pool.py（round26） | 修改 | CEVA PU 池资源数量类型反例与合法单核对照 | `TestPUPool` | 问题2 |
| genesim/src/scheduler/gene_sim_scheduler.py（round27） | 修改 | PIM MLIR 首次读取台账；产物证据、融合豁免、成熟度与 trace 生成共用台账；成熟度前复核消费快照；TP 分片按路径复用 | `_ceva_pimir_texts`、`_register_compiler_artifacts`、`_ceva_artifact_texts_by_op_type`、`_validate_ceva_fusion_exceptions`、`_operator_pimir`、`_ceva_validation_scope_error` | 问题1 |
| genesim/src/backend/comm_lib.py（round27） | 修改 | 新增纯 PIM 链路端点域；完整链路契约与直接 `add_link` 入口复用 | `CEVA_LINK_DOMAINS`、`link_endpoint_errors`、`link_field_errors`、`CommLib.add_link` | 问题2 |
| genesim/src/config_loader.py（round27） | 修改 | 新增端点契约聚合，结构校验与正式校验对已出现的链路统一拒绝 GPU/未知域 | `ceva_comm_link_endpoint_errors`、`_validate_ceva` | 问题2 |
| genesim/src/backend/pim_backend.py（round27） | 修改 | 运行时入口在绑定轨迹前统一校验循环次数与重复次数类型/符号 | `PIMBackend.execute_runtime`、`UnifiedPIMBackend.execute_runtime` | 问题3 |
| genesim/src/pim/pim_core.py（round27） | 修改 | 直接核心的总次数与长循环接口共用正整数契约；CEVA 核心要求正数 | `PIMTracePIMCore.set_tile_count`、`set_tile_count_long`、`execute`、`CEVANPMPIMCore.REQUIRE_POSITIVE_TILE_COUNT` | 问题3 |
| genesim/src/pim/pim_trace_loader.py（round27） | 修改 | 新增执行次数共享契约；轨迹绑定拒绝布尔/浮点/非正 CEVA 次数并保留 24 位上限 | `require_loop_count`、`PIMTrace.set_tile_count`、`PIMTrace.set_tile_count_long` | 问题3 |
| genesim/tests/sim/test_compiler_placement.py（round27） | 修改 | 新增消费台账、直接/流水线成熟度、产物改写与恢复、TP 分片路径复用、GPU 端点与报告链路反例 | `TestCevaPlacementMapping` | 问题1/2 |
| genesim/tests/sim/test_config_loader.py（round27） | 修改 | 新增 GPU/未知域链路在 structure/simulation 的拒绝与合法链路联合正例 | `TestCevaProfile` | 问题2 |
| genesim/tests/sim/test_comm_lib.py（round27） | 修改 | 新增 CEVA 通信库与 `add_link` 的端点反例与合法对照 | `TestCevaLinkContract` | 问题2 |
| genesim/tests/sim/test_cpu_backend.py（round27） | 修改 | 工厂链路反例加入 GPU/未知域端点；合法工厂夹具改用纯 PIM 链路 | `TestCevaCPUModelContract` | 问题2 |
| genesim/tests/sim/test_pim_backend.py（round27） | 修改 | 新增直接/长循环、核心、轨迹绑定与执行入口的非法计数反例及合法整数对照 | `TestUnifiedPIMBackend`、`TestCevaNpmBackend` | 问题3 |

### 3.3 实施问题记录

| 编号 | 问题 | 根因 | 解决方案 | 实现位置 | 验证用例 |
|------|------|------|----------|----------|----------|
| I1 | 纯 PIM 场景下编译器钉住的 GEMM 单元被能力筛选拦截 | 能力判定只允许注意力算子，钉住的 GEMM 也要过同一判定 | 显式钉住的单元跳过能力筛选，按指定 DPU 执行 | `gene_sim_scheduler.py`（`_build_adaptive_initial_placement`） | `test_ceva_requires_pimir_for_every_compute_node` |
| I2 | 把区间倍率放进默认配置后，旧测试修改全局倍率表无法触发重编 | 配置默认值总是存在，签名不再读取全局表 | 倍率表改为可选覆盖，未配置时仍读内置全局表 | `config_loader.py`、`gene_sim_scheduler.py` | `test_template_trace_recompiles_when_zone_factors_change` |
| I3 | 大工作量逐单元展开会产生数万条指令 | 只用单层固定循环表达 | 按 14 位立即数做混合进制分段，每段嵌套循环，段数取决于工作量位数 | `pimir_trace.py`（`_emit_work`） | `test_large_workload_is_not_truncated` |
| I4 | W4A8 的 `pim.matmul %a, %b scales %s` 有三个张量操作数，MAC 提取误报错 | 旧逻辑要求恰好两个操作数 | 只取前两个矩阵类型算 M/N/K，scale 张量只进搬运字节 | `pimir_trace.py`（`_matmul_mac_count`） | `test_quantized_matmul_with_scales_uses_matrix_operands` |
| I5 | tasklet 分块类型 `tensor<2x64xf16, #pim.tasklet_tiled<...>>` 匹配不到元素数 | 正则要求类型以 `>` 闭合，尖括号内还有一层属性 | 形状正则只取 `tensor<AxBxTYPE` 前缀，不要求闭合 | `pimir_trace.py`（`_TENSOR_TYPE_RE`） | `tests/sim/test_pimir_trace.py` 全量 |
| I6 | 纯 PIM 场景的 MODEL_INPUT/OUTPUT、ALL_REDUCE、COLLECTIVE_RELAY 没有 GPU 可落，放置直接报错 | 能力筛选只认注意力算子，且 CPU 计算开关关闭后 CPU 不参与放置 | CEVA 模式把 CPU 加入候选资源，只承接主机编排节点；这些节点允许使用被关闭的 CPU VPU；场景补 CPU↔HOST 链路 | `gene_sim_scheduler.py`（`HOST_ORCHESTRATION_OP_TYPES`、`supports_unit`、`_validate_autotune_placement`） | `test_host_boundary_ops_land_on_cpu_in_pure_pim` |
| I7 | 相位轨迹的静态工作量按 PIM MLIR 的完整形状编码后，运行时又按查询行数重复整条轨迹，同形状下恰好放大 128 倍；KV 长度变化不进时延 | `compile_phase_chain()` 把完整元素/MAC 数编成固定循环，而 B 路 PIM MLIR 的符号维用 128 占位；`PIMVPU` 仍按 `repetitions` 和 `loop_count` 绑定，固定循环对 `loop_count` 不敏感 | 给相位轨迹写 `work_binding=static_shape` 和 `shape_scale`，运行时只按实际/占位比例缩放一次；分块/模板轨迹保持循环绑定。CEVA 缺少占位登记时直接拒绝 | `gene_sim_scheduler.py`（`_phase_trace_binding`）、`vpu.py`（`_static_trace_scale`）、`placement_export.py`（`_attach_bpath_pimir`） | `test_matching_shape_executes_the_whole_trace_once`、`test_single_row_scales_the_static_trace`、`test_kv_length_changes_the_softmax_latency`、`test_bpath_entry_records_symbolic_dim_placeholders` |
| I8 | 旧放置 sidecar 没有 `symbolic_dim_placeholder`，无法区分 128 占位形状与实际请求形状 | 新字段是本轮新增的跨仓契约，旧产物不含它 | CEVA 正式评估直接拒绝并提示重新导出；非 CEVA 保持改动前的运行时重复口径，避免通用旧配置行为突变 | `gene_sim_scheduler.py`（`_phase_trace_binding`） | `test_ceva_refuses_phase_trace_without_placeholders`、`test_non_ceva_keeps_the_old_binding_when_placeholders_are_absent` |
| I9 | CEVA 核心对未配置的 LOOP_BEGIN/LOOP_END/SYNC 静默采用内置 1/1/3 周期，来源台账缺项 | 必需周期键清单只覆盖计算与 DMA 原语，求值器始终从内置默认表起步 | 五份场景补齐单价与来源；严格模式按轨迹实际 opcode 检查配置缺项，缺一项直接报错 | `config_loader.py`、`pim_trace_loader.py`（`_require_cycle_cost_coverage`）、`pim_core.py`（`CEVANPMPIMCore`） | `test_ceva_core_refuses_missing_loop_and_sync_costs`、`test_loop_and_sync_costs_come_from_config` |
| I10 | 扫描产物全文的 `tensor<...>` 会把 f32 累加器、i32 索引和 i1 掩码混进“数据类型”集合，无法判断真实计算精度 | 旧实现扫描全文；GEMM 的 `tt.dot` 结果类型是 f32 累加器（落盘前截回 f16），量化注意力的 `pim.matmul` 是 i8 | 改为按计算原语取主计算张量：`tt.dot` 取第一个操作数，量化等原语取结果类型；搬运、布局、索引和掩码指令不参与 | `gene_sim_scheduler.py`（`_ceva_primary_compute_dtype`、`_CEVA_COMPUTE_OPCODES`） | `test_validation_report_dtype_must_be_the_main_compute_dtype` |
| I11 | 整条相位轨迹按符号维乘积缩放时，GEMV_SCORE 的查询、K 缓存和输出被绑成同一个比例，固定循环与同步成本也被一起缩小 | 编译期只保留汇总字节与工作量，丢弃了逐张量形状，运行时只能用总比例近似 | 相位解析保留每个操作数/结果的形状；调度器按算子执行形状的轴生成运行时模板；VPU 按实际维度重算字节、元素数和 MAC，并用同口径解析公式计算周期，固定成本只计一次 | `pimir_trace.py`（`TensorSpan`、`phase_runtime_template`、`resolve_phase_template`、`phase_cycle_breakdown`）、`gene_sim_scheduler.py`（`_phase_runtime_metadata`）、`vpu.py`（`_static_phase_metrics`） | `test_gemv_score_binds_query_and_kv_operands_separately`、`test_gemv_context_binds_the_probability_operand_to_kv`、`test_gemv_score_reads_a_non_square_kv_cache_as_n_by_k`、`test_view_op_tq_uses_the_bound_axis_not_the_batch_product` |
| I12 | 运行时模板重建 `pim.kv_cache` 这类没有结果的相位时，写回字节被算成 0 | 解析器对无结果算子把输出回退成输入，但模板重建只按结果张量求和 | 模板重建在结果张量为空时同样把输出回退成输入，与编译期口径一致 | `pimir_trace.py`（`resolve_phase_template`） | `test_runtime_template_keeps_result_less_store_bytes` |
| I13 | 全链路在 Tq=KV=726 的请求上重建 GEMV_SCORE 时误报 K 维不一致 | 相位模板只按 `[K,N]` 读第二操作数；K 缓存实际按 `[KV,D]` 存储，占位方阵下无法区分 | MAC 收缩维改为取与第一操作数 K 维相等的边；显式 `transposeB` 仍按 `[N,K]` 读 | `pimir_trace.py`（`_matmul_mac_count_from_shapes`） | `test_gemv_score_reads_a_non_square_kv_cache_as_n_by_k`、全链路回归 |
| I14 | 符号维只按名字解析时，视图算子（SPLIT/TRANSPOSE）的 Tq 被批量维乘积算成 `heads×Tq` | 模板只记维度名，运行时用 `_extract_query_count` 推 Tq；该函数按 `shape[:-1]` 求积，第二维是 Tq 时会把前导 heads 乘进去 | 模板记录每个符号维对应的算子形状轴，运行时按该轴直接取实际值；张量比算子形状多出的前导维按右对齐绑定 | `pimir_trace.py`（`phase_runtime_template` 的 `dimension_sources`、`_axis_binding`）、`vpu.py`（`_phase_dimension_value`） | `test_view_op_tq_uses_the_bound_axis_not_the_batch_product`、`test_runtime_template_right_aligns_broadcast_dims` |
| I15 | 全链路 10 个请求共 1524 个 decode 步，逐形状物化并逐条求值相位指令使仿真无法在可接受时间内完成 | decode 每步 KV 都变，按 (op, 形状) 缓存的指令仍要逐步重建；单算子一次求值约 1 ms，总量上千万次 | 新增 `phase_cycle_breakdown()`：按相位参数直接计算整块搬运分段、混合进制循环和同步周期，口径与指令求值一致；VPU 只缓存结果，不再物化指令。用直接轨迹求值做逐用例对拍 | `pimir_trace.py`（`phase_cycle_breakdown`、`_bulk_cycle_count`、`_work_cycle_count`）、`pim_trace_loader.py`（`resolve_cycle_costs`）、`vpu.py`（`_static_phase_metrics`） | `test_gemv_score_binds_query_and_kv_operands_separately`、`test_softmax_latency_matches_the_runtime_shape_trace`、`test_view_op_tq_uses_the_bound_axis_not_the_batch_product`、`test_static_phase_trace_honours_zone_factors`、`test_profile_memory_bytes_match_the_runtime_shape_trace` |
| I16 | 用真实 sidecar 验证新的产物形状核对时，SPLIT 产物按整头张量 `128x4096` 表达，而执行节点只表示单头 `[Tq,128]`，按末维比较会误报 | 视图算子的产物形态随头数变化，不一定与单个 IR 节点的输出形状一致 | SPLIT 改为核对输入的整份元素总数（`_ceva_bound_element_count`），不再比较末维；其他相位算子仍按结果末维核对 | `gene_sim_scheduler.py`（`_validate_ceva_phase_artifact_shape`、`_ceva_bound_element_count`） | 真实 sidecar 6850 条产物加载通过；`test_matching_artifacts_pass_the_ceva_gate` |
| I17 | 按执行节点逐轴核对相位产物形状时，真实产物里的量化单行内核、GATHER 查表与 RMSNORM 参数张量无法与执行节点输入一一对应 | 算子编译器的 B 路内核会压平、扩展或追加参数张量，函数签名并不等价于 IR 输入列表 | 逐轴核对只作用于能与执行节点轴绑定的张量：前导维必须为 1，单行/单列视图按 1 处理；绑定不上的张量按产物自身形状计，剩余边界记入 K18 | `gene_sim_scheduler.py`（`_ceva_artifact_axis_error`、`_ceva_artifact_shape_matches`） | 真实 sidecar 6850 条产物加载通过；`test_matching_artifacts_pass_the_ceva_gate` |
| I18 | 最终结果绑定首次实现要求返回张量数量等于执行节点输出数量，真实 MEM_COPY 产物被误拒（报“没有最终结果张量，执行节点却有 1 个输出”） | IR 节点有输出形状，但 `pim.kv_cache` 是写入 MRAM 的内核，没有 SSA 返回结果 | 最终结果为空时保留改动前的写入类算子语义：只核对操作数；新增正向用例锁定该行为 | `gene_sim_scheduler.py`（`_validate_ceva_phase_artifact_shape`） | `test_ceva_accepts_a_write_only_artifact_without_results`、真实 sidecar 6850 条产物加载 |
| I19 | 放置 sidecar 没有记录生产端工具链身份，GeneSim 无法直接知道“这份 PIM MLIR 是哪一个图编译器提交和哪个 FlagTree 二进制编出来的” | 设计 R5 要求保存编译版本和哈希，但 sidecar 契约只记录产物路径与单文件哈希；生产端也未嵌入提交号 | 从 sidecar 的 PIM MLIR 路径向上定位图编译器源码仓库并读 HEAD 与脏标记，再从该仓库 `paths.json` 找 FlagTree 安装目录、从 `env-flagtree.sh` 找源码树并读提交号，最后哈希实际安装的 `triton-opt`。任一环节缺失即拒绝正式评估；生产端嵌入身份作为后续跨仓契约改进（Q18） | `src/run_manifest.py`（`compiler_toolchain_identity`、`_find_repo_root`、`_flagtree_prefix`、`_flagtree_source`、`_triton_opt`） | `test_manifest_records_inputs_pim_mlir_and_toolchain`、`test_toolchain_identity_rejects_a_missing_flagtree_source`、`test_toolchain_identity_rejects_pimir_outside_a_source_repo` |

## 四、测试验证结果

### 4.1 测试执行总览

- 执行命令：`source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh` 后执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`（genesim，round27 修复后的最新结果）
- 执行用例总数：1231
- 通过用例数：1230
- 失败用例数：0
- 跳过：1
- 通过率：100%
- 说明：round01 首次实施时记录为 860 条（7.3 另记为 869 条，历史口径不一致）；round02 修复后实测 895 条；round03 修复后 906 条；round04 修复后 921 条（1 条跳过）；round05 修复后 941 条（1 条跳过）；round06 修复后 954 条通过（1 条跳过）；round07 修复后 963 条通过（1 条跳过）；round08 修复后 970 条通过（1 条跳过）；round09 修复后 978 条通过（1 条跳过）；round10 修复后 985 条通过（1 条跳过）；round11 修复后 992 条通过（1 条跳过）；round12 修复后 1008 条通过（1 条跳过）；round13 修复后 1016 条通过（1 条跳过）；round14 修复后 1033 条通过（1 条跳过），新增 17 条用例见 7.1 的第 14 轮清单；round15 修复后 1043 条通过（1 条跳过）；round16 修复后 1054 条通过（1 条跳过）；round17 修复后 1058 条通过（1 条跳过），新增 4 条用例见 7.1 的第 17 轮清单；round18 修复后 1075 条通过（1 条跳过），新增 17 条用例见 7.1 的第 18 轮清单；round19～round22 依次为 1079、1094、1120、1132 条通过（均为 1 条跳过）；round23 修复后 1148 条通过（1 条跳过），新增 16 条用例见 7.1 的第 23 轮清单；round24 修复后 1160 条通过（1 条跳过），新增 12 条用例见 7.1 的第 24 轮清单；round25 修复后 1182 条通过（1 条跳过），新增 22 条用例见 7.1 的第 25 轮清单；round26 修复后 1215 条通过（1 条跳过），新增 33 条用例见 7.1 的第 26 轮清单；round27 修复后 1230 条通过（1 条跳过），新增 15 条用例见 7.1 的第 27 轮清单。官方入口 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim` 按文件逐个运行，46/46 个文件通过，单文件 `Ran N test(s)` 合计 1225 条、1 条跳过、0 失败（round24 记录 1155 条时未计入一个 `Ran 1 test` 的单用例文件；与 pytest 的统计基数不同）。

### 4.2 分任务测试结果

| 任务ID | 测试用例名 | 测试用例数 | 通过数 | 失败数 | 测试说明 |
|--------|------------|------------|--------|--------|----------|
| T1 | `TestCevaProfile`、`TestZoneFactorValidation` | 30 | 30 | 0 | 场景/结构/正式校验与倍率约束 |
| T2 | `TestCevaNpmBackend` | 5 | 5 | 0 | 分派、时钟换算、缺轨迹报错 |
| T3 | `TestCevaCapacityBudget` | 5 | 5 | 0 | 字节真源、GiB 对齐、预留 |
| T4 | `TestCevaPlacementMapping`、`TestPimirHashVerification` | 12 | 12 | 0 | 一对一映射、越界/重复/覆盖、哈希 |
| T5 | `TestPhaseWorkload` | 10 | 10 | 0 | M/N/K、元素、归约、不截断、DMA 免费、W4A8 scales |
| T6 | 区间倍率相关用例 | 9 | 9 | 0 | 编码、校验、签名 |
| T7 | 中转开销用例 | 3 | 3 | 0 | 配置值、缺省 100 ns、校验 |
| T8 | `TestCevaResourceInventory`、成熟度用例 | 4 | 4 | 0 | 核清点与结果级别 |
| T9 | `tests/test_export_pp_placement_budget.py` | 6 | 6 | 0 | G 预算参数 |
| T10 | `tests/sim/test_export_fixed_pu_mapping.py` | 5 | 5 | 0 | 一对一映射导出 |

### 4.3 回归测试结果

以下为 round01 实施时的结果；round02～round32 修复后的最新回归命令与结果见 7.3。

- genesim 全量：`python -m pytest tests/sim -q`，859 通过、1 跳过、0 失败。
- genesim 官方入口：`./run.sh --test`，sim、predictor、DRAMsim3 wrapper、UPMEM checker 全部通过。
- 说明：在 PyTorch 环境里用 pytest 跑 `tests/` 会因该环境缺 `torch_geometric` 而让 predictor 用例报 ImportError；genesim 的 `.venv`（官方入口）具备该依赖，官方入口全绿。
- 图编译器快速回归：`python -m pytest tests/ -q -k "not llama2_7b"`，1670 通过、8 跳过、49 排除、0 失败（17 分 57 秒；本机为纯 CPU 环境）。
- 图编译器 7B 用例：`python -m pytest tests/ -q -k "llama2_7b"`，48 通过、0 失败（约 2 小时 22 分；本机为纯 CPU 环境，含 3 组 7B 编译加 16 步解码对拍）。
- 快速导出：`python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out`，结构自检、dtype 覆盖、引用一致性 3 项全部通过。
- 全流程：`python scripts/run_full_pipeline.py --num-stages 4`，通过。图编译 6852 个算子、224 个 GEMM，算子编译 6626 个算子级节点（21 种本地形状、分块 512），7138 条 trace 全部来自 pimir（`template` 为 0）；仿真 `total_time_s=4251.737`、`tokens/s=1.901`。
- CEVA-NPM 端到端冒烟：`python src/main.py --config /tmp/ceva_sim_onetoone.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射）。结果：7138 条 trace 全部来自 pimir；`total_time_s=2280.341`、`tokens/s=0.349`、`result_maturity=model_prediction`；峰值 TensorPU 常驻占用 24.82%（fits=True）；资源报告“创建 64、使用 8、空闲 56”。同一份旧计划（非 CEVA 生成、段内共用 ClusterPU）会被一对一校验拒绝，符合设计。
  该冒烟用的 sidecar 由本轮流水线产物复制后把 `dpu_to_cluster` 改为一对一；正式 CEVA 联调需用 CEVA 配置重新生成计划和 sidecar。
- CEVA 仿真全流程：`bash scripts/run_ceva_pipeline.sh --run-name ours_e2e_strict --decode-steps 10 --token-check strict` 需要外部 CEVA 仿真器且约 10 小时，本轮未执行，不作通过声明。

## 五、问题与偏差说明

### 5.1 已知问题

| 编号 | 问题描述 | 影响范围 | 状态 | 关闭说明 |
|------|----------|----------|------|----------|
| K1 | 未实现 sidecar 的物理布局描述（逻辑精度、存储类型、打包、scale/bias、布局来源） | W4A8 容量/通信/搬运不能按真实物理字节计量 | 未关闭 | 设计 §4.2 要求，当前只有 FP16 放置产物，按 GML 实物补齐后再接入 |
| K2 | 同一 opcode 在不同相位没有独立有效成本表 | FP16/W4A8 混合精度仍共用一份单价 | 未关闭 | 需按实际执行节点分表，先有分阶段测量数据 |
| K3 | 校准只复用现有 `calibrate.py` / `pim_op_calibration.py`，没有新增 CEVA 锚点报告 | 两包计数、控制器残差和 10 decode=11 输出未形成自动报告 | 未关闭 | 现有脚本已支持拟合/验证分组和 median 误差；锚点报告待校准数据确认 |
| K4 | 物理 engine 数只作画像记录，没有 engine 级调度模型 | 无法刻画核内跨作业重叠 | 已规避 | 设计首期选择整核等效模型；有 engine 调度证据后再扩展 |
| K5 | 多核互连仍是共享链路抽象，未按甲方拓扑建模 | 跨核争用和端口级拥塞不精确 | 未关闭 | 甲方拓扑资料未提供，仅做共享假设下的敏感性分析 |
| K6 | 未知物理时钟时仍输出秒级结果 | 秒级指标可能被误读为硬件实测 | 未关闭 | 设计 §5.4 的门控需要改动结果字段可空性和消费方；当前只输出模型预测标注 |
| K7 | 1e 场景成本表仍是假设值 | 1e 预测精度无独立证据 | 未关闭 | 场景中来源标为 assumption，只作预测，不声称已校准（round01 问题 4 已把 `engines_per_core` 与 `target_name` 统一改为 assumption） |
| K8 | 来源台账只逐项登记了 MAC、DMA 与循环/同步原语，其余原语单价只要求配置齐全 | 其余原语单价可追溯性弱于关键项 | 未关闭 | round02 问题4 已把 LOOP_BEGIN/LOOP_BEGIN_INDIRECT/LOOP_BEGIN_INDIRECT_LONG/LOOP_END/SYNC 纳入显式单价、来源台账和严格检查；其余原语仍需分阶段测量后逐项补来源 |
| K9 | 核数从 64 换到 128 时，没有自动校验切分维度整除和计划/本地分片重新生成 | 多核计划合法性依赖联调脚本 | 未关闭 | 由图编译器侧的计划校验补充 |
| K10 | 来源台账未记录目标文件摘要、时钟域和校准数据指纹 | 参数来源可追溯性弱于设计 §3.1 的完整要求 | 未关闭 | 等 CEVA 目标文件与校准数据确定后补字段 |
| K11 | 整算子相位轨迹按符号维比例整体缩放，SOFTMAX 中与 KV 无关的相位（如 reciprocal）也会乘上 KV 比例 | 相位级缩放粒度不足，KV 很大时该项会偏高 | 已关闭 | round04 问题1 改为按模板逐操作数重建轨迹：属于算子输入/输出的张量按实际轴绑定，reciprocal 等行向量不再乘 KV 比例；中间张量的剩余边界见 K15 |
| K15 | 相位链中不对应算子输入/输出、也不在单行内核行视图登记内的中间张量（如 SOFTMAX 的 `[Tq,1]` 行向量、SPLIT 的三维视图结果）在运行时模板里按占位形状保留，Tq 变化时不缩放 | Tq 远小于 128 的 decode 请求会高估这些相的字节与工作量 | 未关闭 | 本次占位请求 Tq=128 时无误差；逐相语义映射需结合算子编译器展开规则，待后续按相位语义补充轴绑定。round20 起 QUANT/GATHER 等单行内核的中间行视图张量已按运行行数展开，K15 仅剩无法绑定到算子轴且不在行视图登记内的中间张量 |
| K12 | 旧放置 sidecar 缺 `symbolic_dim_placeholder` 时，非 CEVA 配置仍按旧的运行时重复口径执行整算子轨迹 | 旧产物无法获得静态形状缩放，数字仍可能失真 | 未关闭 | CEVA 正式评估已拒绝执行（问题1 修复）；重新导出 sidecar 即消除，通用旧配置行为保持不变是刻意取舍 |
| K13 | `dtype_layout` 按 op_type 汇总；同一 op_type 在本次运行里出现两种主计算类型时无法用单值表述 | 该场景下结果成熟度只能保持 `model_prediction` | 未关闭 | 采用保守失败；需要更细粒度时再扩展为逐 op_id/相位布局（与 Q8 一起确认） |
| K14 | 验证证据的 `sample_set` 由产出方声明，程序只能核对非空和逐点数值自洽，不能证明样本确实来自独立验证集 | 独立性依赖校准流程按设计 §7.2 做分组隔离 | 未关闭 | 程序侧拒绝空证据、缺失字段和与样本不一致的汇总；独立性由产出方流程保证 |
| K16 | SPLIT 视图产物的形状核对只比较输入元素总数，不比较逐轴形状 | 同元素总数但排布不同的错误产物不会被拦下 | 未关闭 | 整头/单头两种产物口径无法统一到逐轴比较；其他相位算子仍按结果末维比较（I16）。round07 已要求动态维/不可解析形状显式失败，剩余边界仅剩“同元素总数但排布不同” |
| K17 | GEMM 的 A 路产物以指针传参，形状一致性用 `pim.tile-n/k` 与本地分片宽度、`pim.placed-mram-bytes` 与 sidecar 放置字节核对，不逐操作数比较张量形状 | 分块参数相同但整体形状不同的错误产物可能漏检 | 未关闭 | A 路产物没有张量级操作数形状；后续可要求导出端额外记录 A 路产物的完整操作数形状 |
| K18 | 相位产物中与执行节点形状绑定不上的张量不参与逐轴核对；单行内核的查表/参数张量仍按产物自身形状计工作量 | 这些路径的整份产物形状错误不会被形状核对拦下；GATHER 查表与 RMSNORM 参数等恒定张量不随运行形状缩放 | 未关闭 | round06 的逐轴核对只覆盖能与执行节点轴绑定的张量；round07 进一步要求操作数与执行节点输入个数一一对应时必须取得明确匹配，个数不等时仍按算子编译器既有约定保留，需在算子编译契约一侧约束。round11 起单行/单列视图豁免只对真实产物使用该契约的 GATHER、QUANT 开放；round20 起登记的单行内核把行视图张量按运行行数展开（QUANT 全部操作数与结果、GATHER 的索引与结果），GATHER 查表与 RMSNORM 参数等恒定张量仍按产物形状计，行视图张量首维不是 1 时显式报错。行视图登记集合仍是跨仓契约（Q19） |
| K19 | `CEVA_ARTIFACT_SUPPORT_PRIMITIVES` 为 GEMM 整体放行 barrier/DMA/WRAM 基础设施语句，没有逐条核对其中 DMA 的字节数与分块的关系 | 错误产物若只改动基础设施语句的搬运量，可能绕过语句级契约 | 未关闭 | GEMM 工作量由 `tt.dot` 分块属性、本地分片宽度与放置字节核对约束；round10 按真实算子编译缓存确认这七条语句是 GEMM 产物的固定基础设施，其他节点不登记。后续可在算子编译契约一侧补齐基础设施语句的字节核对 |
| K20 | 主机编排独立验证证据的样本独立性仍由产出方声明的 `sample_set` 保证；程序只核对算子覆盖、阶段覆盖、预测/观测数值自洽与主机成本参数绑定 | 主机出入口/归约的实测证据可能来自非独立验证集，程序无法识别 | 未关闭 | 与 K14 同类；独立性由校准流程按设计 §7.2 做分组隔离。round12 新增的是产物的机械核对与成本参数绑定；round13 把证据算子集合收窄到真实主机节点（不含合成中转） |
| K21 | B 路相位产物仍由 `_bpath_pimir` 按默认硬件生成 `pim.mram-bytes=8589934592`，round14 的预算门禁只核对分块 GEMM 产物的 `pim.mram-bytes` | 相位产物的硬件预算属性与 CEVA 画像不同，但没有参与门禁 | 未关闭 | 按评审范围只核对分块产物；B 路产物生成入口目前不接收 `PIMHardwareConfig`，需要时再扩展导出契约并同步消费方 |
| K22 | 通用 `pim.compile.use_long_loop_counts=false` 时，短间接循环轨迹超过 4095 次仍按 12 位截断（round16 起会告警） | 通用/UPMEM 配置下超过 4095 次的动态循环仍会少计时 | 未关闭 | round16 按“不改变通用 sim.yaml 与 UPMEM 路径既有行为”的约束保留兼容截断；CEVA 路径不再截断（长循环生成 + 短轨迹自动升级 + 24 位上限显式报错），因此评审缺陷在 CEVA 评估路径已关闭。通用场景需要时打开 `use_long_loop_counts` 即可，模板与 pimir 两条生成路径都会遵守；是否把通用默认改为长循环需另行确认 |
| K23 | 运行清单里的图编译器版本取自 PIM MLIR 所在源码仓库的当前 HEAD（附脏标记），不是产物生成时嵌入的生产端身份 | 若 sidecar 导出后图编译器源码又发生变化，清单记录的是评估时的工具链状态，无法精确证明“这份产物由该提交编译” | 未关闭 | 当前 sidecar 契约没有生产端身份字段，GeneSim 只能从产物路径反推；已把“生产端嵌入编译版本与工具指纹”列为跨仓契约改进（Q18）。清单同时记录 FlagTree 源码提交与实际 `triton-opt` 指纹，可识别安装二进制变化 |
| K24 | CEVA 正式评估要求 PIM MLIR 位于 git 源码工作区内，否则无法确定图编译器版本而拒绝 | 把 `.opcompiler_cache` 整目录复制到非 git 目录后不能直接做正式评估 | 未关闭 | 这是“版本缺失即拒绝正式评估”的刻意行为；可用 `scheduler.pimir_search_dirs` 指回源码仓库内的缓存，或先按 Q18 让生产端在 sidecar 中携带身份 |
| K25 | 通用 `DRAMsim3PIMCore`/`Ramulator2PIMCore` 的 Python 回退路径仍把 `set_tile_count_long(outer, inner)` 直接转给 `PIMTrace.set_tile_count_long(hi, lo)`，与后端公开契约的 `outer×inner` 语义不一致 | 通用后端的长循环备用接口在回退路径上可能错绑次数 | 未关闭 | round22 按评审范围只统一 CEVA 内置核心，通用后端既有行为保持不变（7.2 问题 3）；若后续确认通用核心也要支持长循环总次数，再按同一方式收敛 |

### 5.2 偏差说明

| 设计条目 | 设计方案 | 实际实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| design §3.1 | 场景文件由同一份解析画像生成完整 YAML | 手工维护 5 份完整 YAML | 加载器没有继承语义，生成脚本属于额外抽象 | 否（无用户确认，按最小实现取舍） |
| design §4.2 | sidecar 带版本化物理布局，CEVA 严格模式要求新字段 | 未实现 | 现有产物是 FP16，W4A8 实物映射未确认 | 否 |
| design §5.2 | 按执行节点选择有效成本表 | 同一 opcode 一份成本表 | 需要分阶段测量数据 | 否 |
| design §7.3 | 由程序根据产物、参数来源、独立验证结果和校准范围派生 `validated_in_scope` | 新增 JSON 验证报告契约（round26 起 `schema_version=11`），核对产物哈希、目标、engine 数、配置核数与已用核数、计时边界（含并发上限）、时钟、数据类型宽度、主计算类型与逐算子布局、独立验证证据（预测值为正）、成本模型摘要（含主机成本参数、整核 `multipliers` 和 `loop_encoding` 循环原语族）、主机编排独立证据、形状范围和通信模型（含路由模式、链路适用范围与双工语义）；round10 起成熟度派生同时核对 `ceva.parameter_sources`：必需来源缺失、证据为空、来源为假设或未确认一律保持 `model_prediction`；round20 起验证报告还必须绑定成本 sidecar 的使用状态与内容哈希，被环境覆盖的参数来源必须登记适用于生效值的 `value`；round21 起成本 sidecar 还必须给出与本次消费台账一致的 `coverage`（模型/条目/有效成本/模板/未覆盖五个计数），有效成本算子为零或存在未声明的未覆盖算子时保持 `model_prediction`；round22 起未覆盖算子只按“有效搬运成本 + 显式模板路径”判定，空条目和仅 `source_name` 条目不再算覆盖 | 设计未规定报告字段名，按评审建议定义最小契约；p90 口径收紧、计时边界新增并发上限、链路语义与预测值符号校验、成本模型新增主机参数/`multipliers`/`loop_encoding`、通信模型新增路由模式、成本 sidecar 新增覆盖范围与有效成本口径属于契约变化，按评审要求升级版本；来源证据的语义适用性（目标/形状/边界）仍由配置校验与校准流程人工保证 | 否（字段契约待独立校准产出方确认，见 Q8、Q12、Q15） |
| design §7.1 | 加载时核对产物形状与执行节点匹配 | 按函数签名解析产物操作数与结果的完整形状，符号维按占位值展开后逐轴核对；前导广播维必须为 1，单行/单列内核视图只对登记的 GATHER、QUANT 开放（round11 收紧）；round20 起单行内核的行视图张量按登记的运行行轴展开，查表/参数张量仍按产物自身形状计 | 真实算子编译产物存在量化单行内核、GATHER 查表和 RMSNORM 参数张量，无法与执行节点逐轴一一对应；已在 K18 记录剩余边界 | 否（按评审 round06 问题1 的最小修复取舍；round11 按真实产物把单行视图豁免限定到 GATHER、QUANT；round20 按评审要求把行视图张量与恒定张量分开计量） |
| design §5.2 | 运行时按循环/重复绑定形状，静态循环与运行时绑定不可重复缩放 | 整算子相位轨迹在运行时按模板逐操作数重算字节与工作量，并用与指令求值同口径的解析公式计算周期；分块 GEMM 与手写模板保持循环绑定 | 相位轨迹的固定循环已覆盖 PIM MLIR 完整形状，再按 Tq 重复会放大 128 倍；整条比例缩放会混淆不同操作数并缩小固定成本，故改为按运行时形状计算 | 是（评审 round04 问题1 明确允许“在运行时形状确定后生成对应轨迹”） |
| design §5.2 | 超过立即数范围时分段或用长循环，不截断实际计算量 | CEVA 路径按完整次数绑定：生成长循环，遗留短轨迹在绑定时升级，超过 24 位显式报错；通用/UPMEM 在 `use_long_loop_counts=false` 时仍保留 12 位兼容截断（仅告警） | 本轮约束要求通用 `sim.yaml` 与 UPMEM 路径既有行为不变，而该缺陷影响的是 CEVA 评估；通用场景可显式打开长循环配置（见 K22） | 否（按本轮用户约束保留兼容行为，记录为 K22） |

## 六、后续与待确认事项

| 编号 | 事项 | 状态 | 关闭说明 |
|------|------|------|----------|
| Q1 | 64/128 核交付优先级与 NPM 核到服务器的对应关系 | 未关闭 | 需求文档第五节待甲方确认 |
| Q2 | 每核 8 GB 的准确字节数与预留 | 未关闭 | 当前按十进制 8,000,000,000 B 假设 |
| Q3 | 4e/1e 目标文件、时钟域、MAC 作用域 | 未关闭 | 1e 只有假设成本 |
| Q4 | 核间互连、主机接口、共享 DDR 参数 | 未关闭 | 当前为共享链路假设；round15 起跨核路由模式（`direct`/`host_relay`）也纳入来源台账，五份场景按 `direct` 假设登记，需甲方确认 |
| Q5 | W4A8 真实产物映射与物理布局 | 未关闭 | 阻塞 K1/K2 |
| Q6 | 误差阈值与多核代表测点 | 未关闭 | 需双方确认后判定“性能达标” |
| Q7 | CEVA 联调时 G 侧放置导出需显式传 `--mram-bytes 8000000000 --sys-reserve-bytes <预留>` | 已关闭 | round14 起导出端把物理 MRAM、系统预留和可用量写入 sidecar（version 4），CEVA 加载时与 `ceva.local_memory_bytes`/`ceva.sys_reserve_bytes` 逐项核对，GEMM 产物核对 `pim.mram-bytes`；忘记传参会因预算不一致在生成 trace 前被拒绝，不会再被当成当前硬件产物。默认值仍是通用路径的 4 GiB / 64 MiB；8,000,000,000 B 本身仍是假设（Q2） |
| Q8 | 验证报告 JSON 契约（round26 起 `schema_version=11` 的 `package`/`scope` 字段）需与独立校准产出方对齐 | 未关闭 | round02 在 `scope` 增加 `cost_model`（MAC/DMA/分块/区间成本的规范化摘要），并改用 `scope.dtype` 与实际产物元素类型核对；round03 再增加 `scope.dtype_layout`（op_type → 主计算类型）和 `scope.validation`（`sample_set`、逐点 `operator/predicted/observed/error/relative_error`、`error_summary.signed_error.median` 与 `error_summary.abs_relative_error.median/p90/max`）。round04 再增加 `package.trace_sha256`、`scope.request_boundary`（`request_count`、`input_lengths`、`seq_lengths`、`output_lengths`、`decode_steps`、`prefill_flags`、`max_query_length`），要求 `scope.validation.samples` 每点带 `phase`、`kind`（compute/dma）、`stage`（prefill/decode）、`shape`、`dtype`，且整体覆盖 compute/dma 与 prefill/decode。产出方需按这些字段生成；若字段不同需适配或升级契约版本。`dtype_layout` 的 op_type 集合取自本次放置 sidecar 中带主计算原语的算子，`tt.dot` 取第一个操作数类型、量化原语取结果类型，搬运/布局/索引/掩码指令不参与。`operator` 必须是本次产物中出现的 op_type。round05 再要求样本 `phase` 是产物语句里的原语名、GEMM 形状按 `[M,N]`/`[M,N,K]` 核对运行范围，详见 Q14。round06 再增加 `scope.core_count`（配置核数）与 `scope.used_core_count`（放置计划引用的物理核数）；成熟度同时核对实际创建核数与运行期实际使用核数，核数或使用范围没有证据时保持 `model_prediction`。round07 因 p90 口径明确为最近秩而复算，契约版本升到 2，旧版报告按设计拒绝；round08 起 `used_core_count` 的核对要求运行期核映射存在且与放置计划一致，缺少使用证据时保持 `model_prediction`，`request_boundary` 的 `decode_steps`/`max_query_length` 按 prefill 与 decode-only 分别计算；round09 起 `timing_boundary` 必须登记本次配置的 `max_inflight_requests`（并发准入与排队上限），契约版本升到 3，旧版报告不再升级成熟度。round11 起 `communication_model.links` 的每项除了端点、带宽和基础延迟，还要能表达 `intra_cluster_only`、`inter_cluster_only`、`is_full_duplex`（省略时按运行时缺省：全双工 true、另两项 false），三个语义任一不同即保持 `model_prediction`；`scope.validation.samples` 的 `predicted` 必须为有限正数，零或负预测不再作为校准证据；契约版本升到 4，旧版报告不再升级成熟度。round12 起 `scope.cost_model` 增加 `cpu_model`（`base_latency_us`、`peak_flops`、`peak_bandwidth_gbs`、`num_cpus`），任一项与本次运行不一致即保持 `model_prediction`；运行包含 CPU 主机编排节点时，`scope.host_orchestration` 必须给出 `sample_set`、与本次运行一致的 `op_types`、逐点 `operator/stage/predicted/observed/error/relative_error` 和可复算的误差汇总，并覆盖全部真实主机编排算子与 prefill/decode 阶段；round13 起 `op_types` 按本次实际执行的 TP 展开后放置图核对（含 `ALL_REDUCE`），合成中转 `COLLECTIVE_RELAY` 不参与校准、报告列出它会因集合不一致而保持 `model_prediction`；`scope.cost_model` 还必须给出 `multipliers`，任一不符保持 `model_prediction`；契约版本升到 6，旧版报告不再升级成熟度。round15 起 `scope.communication_model` 还必须给出 `inter_cluster_communication` 并与本次运行的路由模式一致，缺失或不一致保持 `model_prediction`；契约版本升到 7，旧版报告不再升级成熟度。round20 起 `package.cost_sidecar` 必须登记 `used` 与内容 `sha256`，与本次加载的成本 sidecar 不一致（含未配置却声称使用、配置了却登记未使用）保持 `model_prediction`；被 `GENESIM_*` 覆盖的参数来源必须显式登记与生效值一致的 `value`；契约版本升到 8，旧版报告不再升级成熟度。round21 起 `scope.cost_model` 还必须给出 `loop_encoding`（`use_long_loop_counts`、`dynamic_loop_primitives`、`fixed_loop_primitives`），短循环在运行时升级为长循环时 `dynamic_loop_primitives` 必须包含 `LOOP_BEGIN_INDIRECT_LONG`；`package.cost_sidecar` 还必须给出 `coverage`（`model_operators`、`sidecar_operators`、`cost_operators`、`template_operators`、`uncovered_operators`），与本次消费台账逐项一致；契约版本升到 9，旧版报告不再升级成熟度。round22 起 `package.cost_sidecar.coverage.uncovered_operators` 只按“模型算子 − 有效搬运成本算子 − 显式模板算子”计算，空条目和仅 `source_name` 条目算未覆盖；契约版本升到 10，旧版报告不再升级成熟度。round26 起 `scope.request_boundary` 改为分别登记 `input_lengths`（有效 prompt token 数）、`query_lengths`（padding 后执行查询长度，prefill 为首个前向 Tq、decode-only 为 1）和 `initial_kv_lengths`（初始 KV），不再使用合并口径的 `seq_lengths`；三者必须与本次 trace 的统一形状绑定完全一致，缺失或歧义都会保持 `model_prediction`。契约版本升到 11，旧版报告不再升级成熟度 |
| Q9 | B 路放置 sidecar 新增 `symbolic_dim_placeholder` 字段，需与其它 sidecar 消费方确认没有契约冲突 | 未关闭 | 字段为附加可选字段，旧消费方忽略即可；CEVA 正式评估要求它存在。跨仓只涉及 G 导出与 S 仿真两端，本轮已同步 |
| Q10 | LOOP_BEGIN/LOOP_BEGIN_INDIRECT/LOOP_BEGIN_INDIRECT_LONG/LOOP_END/SYNC 的周期值仍是内置占位值（1/1/1/1/3，标为 assumption） | 未关闭 | round02 只把这些值显式化并纳入来源台账，没有编造新数值；需按 CEVA 目标实测或标定后替换 |
| Q11 | 整算子相位轨迹的运行时形状绑定方式 | 已关闭 | round04 问题1 改为在运行时按实际形状逐操作数计算周期，不再使用总时延浮点比例；中间张量不缩放的剩余边界见 K15 |
| Q12 | 验证报告的误差统计口径：`error` 约定为 `observed - predicted`，`median` 采用 `statistics.median`（偶数个样本取中间两值均值），`p90` 采用最近秩（有序绝对相对误差的第 `ceil(0.9·n)` 个，与运行指标 p50/p95 同一实现），程序逐点复算 median/p90/max 后逐项比对 | 未关闭 | round03 新增契约、round07 补齐 p90 复算并把契约升级到 `schema_version=2`；需独立校准产出方按同一口径生成，若约定其他分位数定义需再次升级版本或补充字段 |
| Q13 | PIM MLIR 目标字段契约：产物必须声明 `pim.target="pim:v1"`，与图编译器 `opcompiler_bridge/oplevel_emitter.py::PIM_TARGET` 一致；编译器升级目标串时需同步修改 `CEVA_PIM_MLIR_TARGET` | 未关闭 | 该常量是跨仓契约，不在本轮甲方硬件参数范围内；当前真实产物全部为 `pim:v1` |
| Q14 | 验证样本的相位与形状契约：`phase` 必须是产物语句里的原语名（`tt.dot` 或 `pim.*`）；GEMM 的 `shape` 解释为 `[M,N]` 或 `[M,N,K]`，必须落在 `scope.shape_range` 与本次运行范围内，且 `M` 必须属于对应阶段的实际取值；非 GEMM 的 `shape` 必须等于请求 trace 按运行时轴绑定展开到对应阶段（prefill/decode）的真实执行形状，编译占位形状只用于核对产物 | 未关闭 | round05 新增、round06 收紧的契约细节，需独立校准产出方按同一口径生成；与 Q8 一起确认。round26 起 prefill 的 `Tq` 取 `padded_query_length`（padding 后执行查询长度），`Tp` 取 `initial_kv_length`；decode 的 `Tq` 为 1，KV 从初始长度逐步增长。有效 prompt token 数只作为 `request_boundary.input_lengths` 证据，不能替代执行形状。round19 起“本次运行范围”按 TP 展开后的执行图本地分片计算，不再使用展开前模型图的全局形状 |
| Q15 | 参数来源升级路径：round10 起 `validated_in_scope` 要求 `CEVA_REQUIRED_SOURCE_PATHS` 全部具备非空证据且来源为甲方确认、CEVA 实测或拟合；round21 起该清单新增 `pim.compile.use_long_loop_counts`（循环编码开关）；当前五份 CEVA 场景的容量、时钟、周期成本、区间倍率、互连和主机 CPU 参数仍标为 `assumption`（见 Q1～Q4、Q10、Q16），因此实际运行只会输出 `model_prediction` | 未关闭 | 需要甲方确认或 CEVA 实测这些参数并更新来源台账后，结果才能升级为范围内已验证；本轮没有编造数值，也没有放宽判据。round20 起被 `GENESIM_*` 环境覆盖的参数还必须在来源台账登记与生效值一致的 `value`，否则保持 `model_prediction`；`value` 字段需独立校准产出方按同一约定维护 |
| Q16 | 主机 CPU 参数（`system.num_cpus`、`cpu_model.base_latency_us`、`cpu_model.peak_flops`、`cpu_model.peak_bandwidth_gbs`）尚无主机实测或甲方确认证据 | 未关闭 | round12 按现有 CPUModel 有效缺省值显式写入五份 CEVA 场景并登记为 `assumption`，没有编造新的主机规格；需主机实测或甲方确认后更新来源台账，结果才能升级为范围内已验证 |
| Q17 | 通用 `conf/sim.yaml` 是否把 `pim.compile.use_long_loop_counts` 的缺省值从 false 改为 true | 未关闭 | round16 为保持通用/UPMEM 既有行为，只让 CEVA 路径完整绑定超过 4095 次的动态循环，通用路径保留 12 位兼容截断（K22）。若确认通用场景也需要完整计时，可改为默认 true 后重跑通用回归；这会改变通用 trace 编码，属于行为变更，本轮未自行决定 |
| Q18 | 是否把图编译器提交号、FlagTree 源码提交号和实际 `triton-opt` 指纹由生产端写入放置 sidecar（跨仓契约，sidecar 版本需再升级） | 未关闭 | round18 为不改变 sidecar 契约，由 GeneSim 从 PIM MLIR 路径与已安装工具链反推身份（K23、K24）。若确认要精确绑定“产物—生产端工具链”，应由 `export_pp_placement.py` 在导出时写入身份并让 GeneSim 优先消费；这属于生产端契约变更，本轮未自行决定 |
| Q19 | 单行内核行视图契约（跨仓）：QUANT 的全部操作数/结果、GATHER 的索引与结果按运行行数展开，GATHER 查表操作数（下标 0）保持恒定；行轴取执行输入 0 的第 0 维 | 未关闭 | round20 按真实产物登记该契约，行视图张量首维不是 1 时显式报错。算子编译器若改变单行内核形态（多行内核、调换 GATHER 操作数顺序、新增其他单行内核算子），需要同步修改 `CEVA_SINGLE_ROW_KERNEL_CONTRACT`；这属于跨仓约定，需与算子编译契约一起确认 |
| Q20 | 请求 trace 三字段绑定契约：存在 padding 或前缀/历史 KV 时，trace 生产方必须显式登记 `padded_query_length` 与 `initial_kv_length`；只有 `seq_len` 的单字段旧 trace 仅在 token 数与 `seq_len` 相等时可唯一推断 | 未关闭 | round26 新增。仓库内真实 trace（10 个请求）token 数与 `seq_len` 全部相等，可继续按旧约定推断；若甲方或校准流程提供带 padding、chunked prefill 或 prefix cache 的 trace，需生产端同步输出两个字段，否则 CEVA 正式运行会显式拒绝而不是猜执行形状。二进制 trace 已升到 version 3，旧版 v1/v2 仍可读取 |

## 七、评审问题修复记录

> 关联评审文档：
> - review-ceva-genesim-hw-align-20261107-round01.md（第 1 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round02.md（第 2 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round03.md（第 3 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round04.md（第 4 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round05.md（第 5 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round06.md（第 6 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round07.md（第 7 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round08.md（第 8 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round09.md（第 9 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round10.md（第 10 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round11.md（第 11 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round12.md（第 12 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261107-round13.md（第 13 轮，2026-11-07）
> - review-ceva-genesim-hw-align-20261108-round14.md（第 14 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round15.md（第 15 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round16.md（第 16 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round17.md（第 17 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round18.md（第 18 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round19.md（第 19 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round20.md（第 20 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round21.md（第 21 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round22.md（第 22 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round23.md（第 23 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round24.md（第 24 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round25.md（第 25 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round26.md（第 26 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round27.md（第 27 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round28.md（第 28 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round29.md（第 29 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round30.md（第 30 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round31.md（第 31 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261108-round32.md（第 32 轮，2026-11-08）
> - review-ceva-genesim-hw-align-20261109-round33.md（第 33 轮，2026-11-09）

### 7.1 问题确认与修复清单

第 1 轮（round01）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round01 | 高 | 成立：写入空 JSON `{}` 后 `_ceva_result_maturity()` 返回 `validated_in_scope`，已用最小脚本复现 | 已修复 | 定义 `schema_version=1` 验证报告契约，逐项核对 `status=passed`、IR 与放置 sidecar 的 SHA256、目标型号、engine 数、执行边界、时钟、数据类型宽度、计时边界、通信链路和 GEMM 形状范围；任一证据缺失或不匹配保持 `model_prediction` | `src/scheduler/gene_sim_scheduler.py`（`_ceva_result_maturity`、`_ceva_validation_scope_error`、`_ceva_run_matmul_shape_bounds`、`_ceva_runtime_query_bounds`，另含 `_sha256_file`、`_finite_number`、`_normalized_links`、`_links_equal`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_empty_validation_report_keeps_model_prediction`、`test_unpassed_validation_report_keeps_model_prediction`、`test_validation_report_for_another_target_keeps_model_prediction`、`test_validation_report_shape_out_of_range_keeps_model_prediction`、`test_validation_report_package_hash_mismatch_keeps_prediction`、`test_validation_report_marks_the_result_validated` |
| 2 | round01 | 高 | 成立：`system.pim_type=CEVA-NPM`、`require_compiler_pimir=true` 下，sidecar 只有 `pimir_path` 没有 `pimir_sha256` 时仍返回文件内容，已复现 | 已修复 | CEVA 严格模式对已声明 PIM MLIR 的节点强制要求 `pimir_sha256`，缺失直接报错；非 CEVA 旧配置保留兼容路径。G 侧导出已同时写 `pimir_path` 与 `pimir_sha256`，本轮核对结论，未改 G 代码 | `src/scheduler/gene_sim_scheduler.py`（`_operator_pimir`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_strict_mode_requires_a_declared_hash`、`test_non_ceva_strict_mode_keeps_missing_hash_compatible`、既有 `test_matching_hash_loads`、`test_tampered_pimir_is_refused_in_strict_mode` |
| 3 | round01 | 中 | 成立：`A[2,3] × B^T[4,3] → C[2,4]` 被解析成 `K 维不一致: 3 vs 4`，已复现 | 已修复 | 保留并传递 `transposeB` 属性：该属性存在时第二操作数按 `[N,K]` 读取，否则维持 `[K,N]`；两条相位解析路径都透传该标记 | `src/pim/pimir_trace.py`（`_matmul_mac_count`、`_parse_single_phase_ops`、`parse_phase_chain`、`_TRANSPOSE_B_RE`）（S）；`tests/sim/test_pimir_trace.py`（S） | `test_transposed_matmul_reads_b_as_n_by_k`、`test_doubling_transposed_n_doubles_matmul_compute`、既有 `test_matmul_work_uses_m_n_k`、`test_doubling_k_doubles_matmul_compute`、`test_doubling_n_doubles_matmul_compute` |
| 4 | round01 | 中 | 成立：`ceva_64core_1e.yaml`、`ceva_128core_1e.yaml` 的 `ceva.engines_per_core` 标为 `ceva_measured`，evidence 却写明 4e 证据不适用、1e 独立证据缺失 | 已修复 | 两项来源改为 `assumption` 并改写 evidence；补充 `ceva.target_name` 假设来源；新增来源语义一致性测试 | `conf/ceva_64core_1e.yaml`、`conf/ceva_128core_1e.yaml`（S）；`tests/sim/test_config_loader.py`（S） | `test_one_engine_profiles_do_not_claim_measured_engine_count`、`test_measured_origin_does_not_contradict_evidence` |
| 5 | round01 | 中 | 成立：把 `pim.pim_trace_cycle_costs.MAC` 改为 `-1`、字符串、布尔值、NaN 或 Inf 后，`validate(mode="simulation")` 仍返回 True，已复现 | 已修复 | 正式校验逐项检查必需成本键：必须是有限数值且非负；除 `SCORE_TILE`、`CONTEXT_TILE` 和 setup 类占位项外必须为正值 | `src/config_loader.py`（`CEVA_ZERO_ALLOWED_CYCLE_COSTS`、`_validate_ceva`）（S）；`tests/sim/test_config_loader.py`（S） | `test_simulation_rejects_negative_cycle_cost`、`test_simulation_rejects_zero_for_non_setup_cost`、`test_simulation_rejects_non_numeric_cycle_cost`、`test_simulation_rejects_non_finite_cycle_cost` |

第 2 轮（round02）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round02 | 严重 | 成立：用真实 `bpath-fb65415debb6b9a5.pimir.mlir` 复现 SILU 同形状 `PIMVPU.execute()` 为轨迹时延的 128 倍（0.7358s vs 0.00575s 量级）；真实 SOFTMAX 相位轨迹在 KV=16/32/64 下时延完全相同 | 已修复 | 给整算子相位轨迹写 `work_binding=static_shape` 与符号维占位值 `shape_scale`；运行时只按实际/占位比例缩放一次，不再按 Tq 重复整条轨迹；G 侧 B 路 sidecar 登记 `symbolic_dim_placeholder`；CEVA 缺登记时拒绝执行 | `genesim_bridge/placement_export.py`（`_attach_bpath_pimir`、`_SYMBOLIC_DIM`）（G）；`src/scheduler/gene_sim_scheduler.py`（`_phase_trace_binding`、`_register_compiler_artifacts`、`_prepare_pim_traces`）（S）；`src/vpu/vpu.py`（`_static_trace_scale`、`_runtime_shape_dimension`、`execute`、`estimate`、`execute_profile`）（S） | `test_matching_shape_executes_the_whole_trace_once`、`test_single_row_scales_the_static_trace`、`test_kv_length_changes_the_softmax_latency`、`test_profile_memory_bytes_scale_with_the_runtime_shape`、`test_dynamic_indirect_loop_still_binds_at_runtime`、`test_phase_trace_records_static_binding_and_placeholders`、`test_ceva_refuses_phase_trace_without_placeholders`、`test_non_ceva_keeps_the_old_binding_when_placeholders_are_absent`、`test_bpath_entry_records_symbolic_dim_placeholders` |
| 2 | round02 | 严重 | 成立：64 核 4e 场景改 `require_compiler_pimir=false` 后 `validate(mode="simulation")` 返回 True；调度器侧缺产物检查也被该开关绕过，计算节点全部加载 `trace_source=template` | 已修复 | 正式运行校验强制 `scheduler.require_compiler_pimir=true`；调度器对 CEVA 的缺产物、路径不可读、缺哈希和哈希不符检查改为不依赖开关的兜底拒绝；非 CEVA 宽松模式保持兼容 | `src/config_loader.py`（`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_operator_pimir`、`_prepare_pim_traces`）（S） | `test_simulation_requires_the_strict_pimir_switch`、`test_ceva_requires_pimir_even_when_the_flag_is_off`、`test_ceva_requires_a_declared_hash_even_with_the_flag_off`、`test_ceva_refuses_a_missing_artifact_even_with_the_flag_off` |
| 3 | round02 | 高 | 成立：同一份通过报告下，把本次运行 MAC 成本 1 改为 1000、`mac_width` 8 改为 16，或把报告 `dtype` 改成 `int4`，结果仍为 `validated_in_scope` | 已修复 | 验证报告 `scope` 新增 `cost_model`（MAC/DMA/分块/区间成本规范化摘要），逐项与本次运行配置比对；`scope.dtype` 改为与本次放置 sidecar 引用产物中实际出现的元素类型比对；任一不符保持 `model_prediction` | `src/scheduler/gene_sim_scheduler.py`（`_ceva_cost_model_summary`、`_ceva_cost_model_error`、`_ceva_artifact_dtype_error`、`_ceva_run_artifact_dtypes`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_marks_the_result_validated`、`test_validation_report_missing_cost_model_keeps_prediction`、`test_validation_report_cost_model_mismatch_keeps_prediction`、`test_validation_report_dtype_outside_the_artifacts_keeps_prediction`、`test_validation_report_without_artifacts_keeps_prediction` |
| 4 | round02 | 中 | 成立：五份 CEVA YAML 与必需周期键清单都缺 LOOP_BEGIN/LOOP_END/SYNC，求值器静默采用内置 1/1/3；同一轨迹配置缺失时按内置默认值计 23 周期，显式配置后按配置值计（测试中为 46 周期） | 已修复 | 5 份场景补齐 LOOP_BEGIN/LOOP_BEGIN_INDIRECT/LOOP_BEGIN_INDIRECT_LONG/LOOP_END/SYNC 的单价和来源；必需成本键与来源路径同步扩展；CEVA 核心启用严格模式，按轨迹实际 opcode 检查配置缺项，缺一项直接报错 | `src/config_loader.py`（`CEVA_REQUIRED_CYCLE_COST_KEYS`、`CEVA_REQUIRED_SOURCE_PATHS`）（S）；`conf/ceva_*.yaml` × 5（S）；`src/pim/pim_trace_loader.py`（`_require_cycle_cost_coverage`、`estimate_total_cycles`、`estimate_cycle_breakdown`）（S）；`src/pim/pim_core.py`（`STRICT_CYCLE_COSTS`、`_trace_latency`）（S）；`src/vpu/vpu.py`（`execute_profile`）（S） | `test_simulation_requires_loop_and_sync_cycle_costs`、`test_repository_ceva_scenarios_declare_runtime_primitives`、`test_ceva_core_refuses_missing_loop_and_sync_costs`、`test_loop_and_sync_costs_come_from_config` |

第 3 轮（round03）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round03 | 严重 | 成立：用真实 sidecar（`/tmp` 改为 8 台 DPU 一对一映射）复跑，原代码在 TP 展开后失败于 op6867，报“整算子相位轨迹带符号维 ['Tq']，但放置 sidecar 没有登记 symbolic_dim_placeholder”；展开出的新 op_id 没有继承占位值 | 已修复 | 在 TP 展开继承 pim mlir 路径/哈希/区间/吸收标记的同一处，按原 op_id 把 `_compiler_symbolic_dims` 复制给每个分片；归约节点继续不继承 | `src/scheduler/gene_sim_scheduler.py`（`_expand_multi_shard_operators`）（S） | `test_shards_inherit_symbolic_dim_placeholders`；真实冒烟见 7.3 |
| 2 | round03 | 高 | 成立：真实产物类型集合含 `{f16,f32,i32,i8}` 时，报告 `dtype=int8`、`element_bytes=2` 仍返回 `validated_in_scope`；旧实现只查 dtype 是否为全产物类型集合的成员，也不核对元素宽度和主计算精度 | 已修复 | 改为从实际消费的产物里提取主计算张量类型：`tt.dot` 取第一个操作数、量化原语取结果类型，搬运/布局/索引/掩码不参与（替换旧的 `_ceva_run_artifact_dtypes` 全文扫描）；报告新增 `scope.dtype_layout` 逐 op_type 布局核对，`dtype` 必须是本次运行主计算精度（覆盖算子数最多），并按类型表核对 `element_bytes` 自洽 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_primary_compute_dtype`、`_ceva_run_compute_dtypes`、`_ceva_artifact_dtype_error`、`_CEVA_DTYPE_ELEMENT_BYTES`、`_CEVA_COMPUTE_OPCODES`、`_CEVA_TENSOR_TYPE_RE`）（S） | `test_validation_report_dtype_width_must_match_the_dtype`、`test_validation_report_dtype_must_be_the_main_compute_dtype`、`test_validation_report_dtype_layout_must_cover_every_compute_op`、既有 `test_validation_report_dtype_outside_the_artifacts_keeps_prediction`、`test_validation_report_without_artifacts_keeps_prediction` |
| 3 | round03 | 高 | 成立：完整报告删掉 `scope.validation` 后仍返回 `validated_in_scope`；旧实现只检查 `status="passed"` 与范围字段，不要求任何独立验证样本或误差统计，正向测试夹具同样没有证据 | 已修复 | 报告 `scope.validation` 要求独立验证集标识、非空逐点样本（算子标识、预测、观测、带符号误差、相对误差）、误差汇总（带符号误差 median，绝对相对误差 median/p90/max）；程序逐点复算误差并核对汇总，缺项、空集、非有限值或汇总对不上时保持 `model_prediction`；未自行发明误差阈值 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_validation_evidence_error`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（夹具补证据）（S） | `test_validation_report_without_evidence_keeps_prediction`、`test_validation_report_with_empty_samples_keeps_prediction`、`test_validation_report_without_error_summary_keeps_prediction`、`test_validation_report_error_summary_must_match_samples`、`test_validation_report_samples_need_predictions_and_observations`、既有正向 `test_validation_report_marks_the_result_validated` |
| 4 | round03 | 中 | 成立：64 核 4e 场景删除 `comm_config.host_relay_overhead_ns` 来源后正式校验仍返回 True；五份 CEVA YAML 都只登记了 `comm_config.links` | 已修复 | 把 `comm_config.host_relay_overhead_ns` 加入 `CEVA_REQUIRED_SOURCE_PATHS`；五份场景补 origin/evidence/unit/scope，数值仍为 100 ns；通用旧配置缺省 100 ns 的行为不变 | `src/config_loader.py`（`CEVA_REQUIRED_SOURCE_PATHS`）（S）；`conf/ceva_*.yaml` × 5（S）；`tests/sim/test_config_loader.py`（S） | `test_simulation_requires_host_relay_overhead_source`、`test_repository_ceva_scenarios_declare_host_relay_source`、既有 `test_host_relay_overhead_defaults_to_100ns`、`test_complete_profile_passes_simulation_validation` |

第 4 轮（round04）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round04 | 严重 | 成立：用真实 `bpath-f265519387e62064.pimir.mlir` 复现，Tq=1、KV=128 下旧实现为 1.480044 μs、384 B，按该形状直接编译的轨迹为 2.623194 μs、16,640 B | 已修复 | 相位解析保留逐张量形状；调度器生成运行时模板（轴绑定 + 编译参数）；VPU 按本次实际形状重算每相字节、元素数和 MAC，并用与指令求值同口径的解析公式计算周期，固定循环与同步只计一次；trace schema 升 9 | `src/pim/pimir_trace.py`（`TensorSpan`、`PhaseStep`、`phase_runtime_template`、`resolve_phase_template`、`phase_cycle_breakdown`、`_matmul_mac_count_from_shapes`）（S）；`src/pim/pim_trace_loader.py`（`resolve_cycle_costs`）（S）；`src/vpu/vpu.py`（`_static_phase_metrics`、`_phase_dimensions`、`execute`、`execute_profile`、`estimate`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_phase_runtime_metadata`、`_prepare_pim_traces`）（S） | `test_gemv_score_binds_query_and_kv_operands_separately`、`test_gemv_context_binds_the_probability_operand_to_kv`、`test_gemv_score_reads_a_non_square_kv_cache_as_n_by_k`、`test_view_op_tq_uses_the_bound_axis_not_the_batch_product`、`test_softmax_latency_matches_the_runtime_shape_trace`、`test_single_row_binds_each_operand_to_the_runtime_shape`、`test_static_phase_trace_honours_zone_factors`、`test_profile_memory_bytes_match_the_runtime_shape_trace`、`test_prepared_phase_trace_executes_at_the_runtime_shape`、`test_runtime_template_keeps_result_less_store_bytes`、`test_runtime_template_right_aligns_broadcast_dims` |
| 2 | round04 | 高 | 成立：报告 `package` 只绑定 IR 与放置 sidecar 哈希，`timing_boundary` 不含请求边界；同一报告在 `output_length=1/11/100` 下均返回 `validated_in_scope` | 已修复 | `package` 新增 `trace_sha256`；`scope.request_boundary` 绑定请求数、有效 prompt、初始序列长度、输出长度、decode 步数、prefill 标记和最大查询长度；任一与本次 trace 不符保持 `model_prediction` | `src/scheduler/gene_sim_scheduler.py`（`_ceva_request_boundary`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_with_another_output_length_keeps_prediction`、`test_validation_report_with_another_decode_boundary_keeps_prediction`、`test_validation_report_with_another_request_trace_keeps_prediction`、既有正向 `test_validation_report_marks_the_result_validated` |
| 3 | round04 | 高 | 成立：样本只要求 `operator` 非空与数值自洽；把三个样本改成 `unrelated_operation_*` 后仍返回 `validated_in_scope`，正向夹具也只用任意 `microbench_op*` | 已修复 | 样本的 `operator` 必须是本次产物中的 op_type，`dtype` 必须与该产物的主计算类型一致，并带 `phase`/`kind`/`stage`/`shape`；样本整体必须覆盖 compute、dma 与 prefill、decode，缺类或全部无关时保持 `model_prediction` | `src/scheduler/gene_sim_scheduler.py`（`_ceva_validation_evidence_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_with_unrelated_evidence_keeps_prediction`、`test_validation_report_sample_dtype_must_match_the_artifact`、`test_validation_report_with_single_stage_keeps_prediction`、`test_validation_report_without_dma_evidence_keeps_prediction`、既有正向 `test_validation_report_marks_the_result_validated` |
| 4 | round04 | 中 | 成立：`loop_count=10` 实际为 12 ns（begin 1 + MAC×10 + end 1），旧断言按 30 ns 且默认 7 位小数，12 ns 与 30 ns 都被舍入为 0.0 而通过；46 ns 断言同样失效 | 已修复 | 循环用例按当前求值语义改为 12 ns 并把精度提到 15 位，增加 `loop_count=20` 与 10 的差值断言；`46 ns` 与 `140 ns` 断言同样提高精度 | `tests/sim/test_pim_backend.py`（S） | `test_tile_count_binds_runtime_loop`、`test_loop_and_sync_costs_come_from_config`、`test_execute_uses_trace_cycles_and_clock` |

第 5 轮（round05）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round05 | 严重 | 成立：把真实 CEVA 配置的 `scheduler.simulation_backend` 改为 `predictor` 后，修复前 `validate(mode="simulation")` 仍返回 True；`_ceva_result_maturity()` 也不看实际后端与结果产出方 | 已修复 | 正式校验强制 `scheduler.simulation_backend=pipeline`，启用自动调参时强制 `autotune.validation_backend=pipeline`；Scheduler 入口对直接调用再兜底拒绝；成熟度同时核对 `simulation_backend` 与 `result_producer` | `src/config_loader.py`（`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_ceva_result_maturity`、`run`）（S） | `test_simulation_rejects_the_predictor_backend`、`test_simulation_rejects_predictor_autotune_validation`、`test_result_maturity_requires_the_pipeline_backend`、`test_result_maturity_requires_the_pipeline_producer`、`test_ceva_run_refuses_the_predictor_backend` |
| 2 | round05 | 严重 | 成立：把真实 sidecar 的 GEMM 条目改指向 SILU 的 `pim.lut` 产物后，修复前 `_apply_compiler_local_shapes()` 不报错；同一份 `tensor<8x64xf16>` 可以被 5 种算子共用 | 已修复 | 加载时核对计算原语与激活种类、`pim.target="pim:v1"`、GEMM 的 tile 与 `pim.placed-mram-bytes`、相位操作数与结果的末维及符号维（SPLIT 按元素总数）；任一不符直接拒绝 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_PIM_MLIR_TARGET`、`CEVA_ARTIFACT_PRIMITIVES`、`CEVA_ARTIFACT_ACTIVATIONS`、`_validate_ceva_compiler_artifact`、`_validate_ceva_gemm_artifact`、`_validate_ceva_phase_artifact_shape`、`_ceva_tensor_operand_dims`、`_ceva_bound_last_dim`、`_ceva_bound_element_count`、`_mlir_int_attribute`）（S） | `test_ceva_refuses_an_artifact_with_the_wrong_primitive`、`test_ceva_refuses_an_artifact_built_for_another_target`、`test_ceva_refuses_an_artifact_with_the_wrong_shape`、`test_ceva_refuses_a_symbolic_axis_mismatch`、`test_ceva_refuses_a_split_artifact_with_the_wrong_size`、`test_ceva_refuses_a_gemm_budget_mismatch`、`test_matching_artifacts_pass_the_ceva_gate` |
| 3 | round05 | 严重 | 成立：把真实 sidecar 的 `fused_activations` 删掉、只留 5 个计算节点到 `absorbed_ops` 后，修复前仍按吸收节点发送空 trace、周期为 0；旧实现不记录主算子归属 | 已修复 | 按 sidecar 的 `fused_activations` 与 IR 的 `subgraphs`/依赖把吸收节点绑定到唯一主算子与激活；核对主算子有产物和哈希、产物内含 `#pim.act_spec` 或 `#pim.activation` 相位、吸收节点形状等于主算子输出；证明不了的不予豁免 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_fused_owners`、`_validate_ceva_fusion_exceptions`、`_register_compiler_artifacts`）（S） | `test_fusion_exception_binds_to_its_main_operator`、`test_absorbed_operator_without_main_operator_is_refused`、`test_absorbed_operator_without_a_fused_phase_is_refused`、`test_arbitrary_operator_cannot_be_marked_absorbed`、既有 `test_shards_inherit_the_absorbed_flag` |
| 4 | round05 | 高 | 成立：把真实 CEVA 配置的 `scheduler.use_compiler_local_shapes` 改为 false 后，修复前正式校验仍返回 True，执行形状退回全局值 | 已修复 | 正式校验强制 `use_compiler_local_shapes=true`；`_apply_compiler_local_shapes()` 对 CEVA 关闭时直接抛错；测试核对执行形状等于 sidecar 的本地分片宽度 | `src/config_loader.py`（`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_apply_compiler_local_shapes`）（S） | `test_simulation_requires_compiler_local_shapes`、`test_ceva_requires_compiler_local_shapes` |
| 5 | round05 | 高 | 成立：把验证样本形状改为 `[999999,999999]`、相位改为 `unrelated_phase` 后，修复前仍返回 `validated_in_scope`；旧实现只检查相位非空、形状是正整数列表 | 已修复 | 样本相位必须是产物语句里的原语名；GEMM 形状按 `[M,N]`/`[M,N,K]` 同时核对 `scope.shape_range` 与本次运行范围；非 GEMM 形状必须等于本次运行该算子按占位值展开后的执行形状；正向夹具改用 GEMM 语义产物和范围内形状 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_artifact_texts_by_op_type`、`_ceva_artifact_phases`、`_ceva_run_shapes_of`、`_ceva_sample_shape_error`、`_ceva_validation_evidence_error`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_sample_shape_must_be_in_the_run_range`、`test_validation_report_sample_phase_must_exist_in_the_artifact`、既有正向 `test_validation_report_marks_the_result_validated` |

第 6 轮（round06）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round06 | 严重 | 成立：按评审复现脚本，SILU 执行节点 `[128,64]` 接受 `tensor<2x128x64xf16>` 产物，元素量由 8,192 增到 16,384；修复前同末维不同前导轴 `[64,64]` 与无法解析的 `tensor<?x...>` 也会跳过核对 | 已修复 | 从函数签名解析张量操作数与结果的完整形状，符号维按 sidecar 占位值展开后逐轴核对；前导广播维必须为 1，单行/单列内核视图只允许其余轴为 1；维度或轴向不一致、形状无法解析、签名缺张量操作数均直接拒绝；拒绝发生在 `_prepare_pim_traces()` 之前 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_bound_shape`、`_ceva_tensor_operand_shapes`、`_ceva_artifact_axis_error`、`_ceva_artifact_shape_matches`、`_validate_ceva_phase_artifact_shape`、`_ceva_bound_element_count`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_refuses_an_artifact_with_an_extra_leading_axis`、`test_ceva_refuses_an_artifact_with_another_leading_axis`、`test_ceva_refuses_an_unparseable_tensor_shape`、`test_ceva_refuses_before_any_trace_is_generated`、既有 `test_ceva_refuses_an_artifact_with_the_wrong_shape`、`test_ceva_refuses_a_symbolic_axis_mismatch`、`test_matching_artifacts_pass_the_ceva_gate` |
| 2 | round06 | 高 | 成立：同一份验证报告在 64 核与 128 核配置下均返回 `validated_in_scope`；旧实现不核对配置核数、实际创建核数与放置计划已用核数 | 已修复 | 报告 `scope` 新增 `core_count`（配置核数）与 `used_core_count`（放置计划引用的物理核数）；成熟度核对配置核数、实际创建 PIM 核数、放置计划引用的不同 DPU 数以及运行期实际使用的 PIM 核数，任一缺失或不一致保持 `model_prediction` | `src/scheduler/gene_sim_scheduler.py`（`_ceva_created_core_count`、`_ceva_plan_used_core_count`、`_ceva_active_core_count`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_for_another_core_count_keeps_prediction`、`test_validation_report_for_another_used_core_count_keeps_prediction`、`test_validation_report_without_core_scope_keeps_prediction`、`test_result_maturity_requires_the_created_core_count`、`test_used_core_count_includes_shard_dpus`、既有正向 `test_validation_report_marks_the_result_validated` |
| 3 | round06 | 高 | 成立：含 10 次 decode 的请求中，decode 样本写 `[128,128]` 仍返回 `validated_in_scope`；旧 `_ceva_run_shapes_of()` 把符号维替换成编译占位值，`stage="decode"` 只是标签 | 已修复 | 按请求 trace 展开每次前向的 `Tq/Tp/Tp+Tq`：prefill 的 Tq 与 KV 均为有效 prompt 长度，decode 的 Tq=1、KV 从初始长度逐步增长；非 GEMM 样本必须等于对应阶段的真实执行形状，GEMM 的 M 也按阶段核对；编译占位形状只用于核对产物 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_runtime_shape_contexts`、`_ceva_run_shapes_of`、`_ceva_run_matmul_stage_m`、`_ceva_sample_shape_error`、`_ceva_validation_evidence_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_runtime_shapes_expand_from_the_request_trace`、`test_validation_report_decode_sample_uses_the_runtime_kv_shape`、`test_validation_report_with_the_placeholder_decode_shape_keeps_prediction`、`test_validation_report_decode_sample_cannot_use_the_prefill_shape`、既有正向 `test_validation_report_marks_the_result_validated` |

第 7 轮（round07）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round07 | 严重 | 成立：按评审复现脚本，MEM_COPY 执行节点输入 `[128,64]`、产物操作数 `tensor<64x32xi8>` 时修复前通过加载核对；同秩不同末维（含结果正确但输入错误）也被放行。真实 sidecar 的输入侧只有 GATHER 查表张量属于“操作数与输入个数不等”的 K18 例外 | 已修复 | 输入操作数个数与执行节点输入个数一一对应时，`_ceva_artifact_axis_error(strict=True)` 对末维或轴向不同直接报错，不再把“无错误文本”当作形状一致；个数不等时（GATHER 表/参数张量）保留 K18 豁免；结果列表用 `strict=False` 保留中间结果跳过语义；拒绝发生在 `_prepare_pim_traces()` 之前 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_artifact_axis_error`、`_validate_ceva_phase_artifact_shape`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_axis_error_reports_a_different_last_dim`、`test_ceva_refuses_a_mem_copy_artifact_with_a_mismatched_input`、`test_ceva_refuses_a_wrong_input_even_when_the_result_matches`、`test_ceva_refuses_a_mismatched_input_before_any_trace_is_generated`、既有 `test_matching_artifacts_pass_the_ceva_gate`、真实 sidecar 加载核对 |
| 2 | round07 | 严重 | 成立：SPLIT 产物 `tensor<?x4096xf16>` 时正则取不到静态形状，`sizes=[]` 直接放行；随后 `parse_phase_chain()` 得到 `in_bytes=0/out_bytes=0`，`compile_phase_chain()` 只剩一条 SYNC（已用探针复现） | 已修复 | SPLIT 分支要求预期元素数可绑定、张量形状全部可解析为静态形状且包含预期元素总数，动态维或空形状列表直接拒绝；`_tensor_totals()` 对无法解析静态形状的非空张量报 `PimirTraceError`，相位解析不再退化为零成本 | `src/scheduler/gene_sim_scheduler.py`（`_validate_ceva_phase_artifact_shape`）（S）；`src/pim/pimir_trace.py`（`_tensor_totals`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_pimir_trace.py`（S） | `test_ceva_refuses_a_split_artifact_with_a_dynamic_shape`、`test_dynamic_tensor_shape_is_refused`、`test_dynamic_tensor_shape_in_a_phase_is_refused`、既有 `test_ceva_refuses_a_split_artifact_with_the_wrong_size` |
| 3 | round07 | 中 | 成立：逐点相对误差 `[0.001,0.002,0.003,0.004]`、报告 p90 填中位数 `0.0025` 时修复前仍返回 `validated_in_scope`；旧实现只检查 `median ≤ p90 ≤ max`，不复算 | 已修复 | p90 改为由逐点绝对相对误差按最近秩复算（与运行指标 p50/p95 同一实现），逐项比对；中位数 `0.0025` 与线性插值 `0.0037` 均被拒绝，最近秩值 `0.004` 通过。口径收紧属于契约变化，`CEVA_VALIDATION_SCHEMA_VERSION` 升到 2，旧版报告不再升级成熟度 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_nearest_rank`、`_ceva_validation_evidence_error`、`_execute_pipeline`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_p90_must_be_recomputed_from_samples`、`test_validation_report_with_an_old_schema_keeps_prediction`、既有 `test_validation_report_marks_the_result_validated`、`test_validation_report_error_summary_must_match_samples` |

第 8 轮（round08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round08 | 高 | 成立：按评审探针输入 decode-only 请求（`is_prefill=false`、`seq_len=128`、`output_length=10`），运行形状正确展开 10 个 decode 上下文，`_ceva_request_boundary()` 却返回 `decode_steps=[9]`、`max_query_length=128`；现有夹具只构造 prefill 请求 | 已修复 | decode 步数按 `is_prefill` 区分：prefill 为 `max(0, output_length-1)`，decode-only 为 `output_length`；`max_query_length` 改为各请求实际前向的最大 Tq（prefill 取有效 prompt 长度，decode-only 取 1），不再把 decode-only 的初始 KV 长度当作查询长度 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_request_boundary`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_request_boundary_counts_all_decode_only_steps`、`test_request_boundary_reads_a_decode_only_binary_trace`、`test_request_boundary_handles_mixed_requests`、`test_decode_only_report_with_wrong_decoding_steps_keeps_prediction` |
| 2 | round08 | 高 | 成立：完整有效验证报告、已创建核数正确但 `op_to_vpu_id` 为空时，`_ceva_active_core_count()` 返回 `None`，`_ceva_validation_scope_error()` 因只在非 `None` 时比较而跳过核对，最终仍返回 `validated_in_scope`；正向夹具没有放置运行期算子映射 | 已修复 | 实际使用核数无法确定改为明确的拒绝原因，不再跳过；实际使用核数仍须与放置计划已用核数一致。正向夹具补与计划一致的 `op_to_vpu_id`，新增空映射、映射未落在 PIM 核、实际核数多于计划三类反例 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_validation_scope_error`、`_ceva_active_core_count`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_result_maturity_requires_active_core_evidence`、`test_result_maturity_requires_pim_active_cores`、`test_result_maturity_rejects_more_active_cores_than_plan`、既有正向 `test_validation_report_marks_the_result_validated` |

第 9 轮（round09）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round09 | 严重 | 成立：评审探针把同时含 `pim.lut` 与 `pim.matmul` 的产物登记给 SILU，修复前 `_apply_compiler_local_shapes()` 返回成功，`parse_phase_chain()` 同时生成 8192 个查表工作单元和 524288 次 MAC；真实产物核对显示 SOFTMAX 合法相位为 `reduce_axis/eltwise/lut`、QUANT 为 `global_pool/kantor/lut` | 已修复 | 按语句解析产物实际出现的计算原语；允许集合为语义标记加登记的融合相位（SOFTMAX 补 `eltwise/lut`，QUANT 补 `kantor/lut`），混入其他计算原语直接拒绝；拒绝发生在相位解析和 trace 生成之前 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_ARTIFACT_COMPUTE_PRIMITIVES`、`CEVA_ARTIFACT_FUSED_PHASES`、`_MLIR_STATEMENT_PRIMITIVE_RE`、`_validate_ceva_compiler_artifact`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_refuses_a_contradictory_compute_primitive`、`test_ceva_refuses_a_mixed_primitive_before_any_trace_is_generated`、既有 `test_matching_artifacts_pass_the_ceva_gate`、真实 sidecar 6850 条产物加载 |
| 2 | round09 | 严重 | 成立：评审探针的 SILU 执行节点输出 `[128,64]`，产物中间结果为 `tensor<128x64xf16>`、最终返回 `tensor<128x32xf16>`，修复前加载核对返回成功；旧实现收集全文结果后只要任意一个匹配就通过，现有多结果用例没有绑定最终返回 | 已修复 | `tt.return` 带操作数时以返回行的类型为准；实产物是无操作数返回，取最后一条带结果的语句作为最终结果；与执行节点输出按位置逐项核对，返回数量不一致也拒绝，中间结果不能再代替最终结果；写入类算子没有 SSA 结果时保持按操作数核对 | `src/scheduler/gene_sim_scheduler.py`（`_MLIR_RETURN_RE`、`_MLIR_RESULT_STATEMENT_RE`、`_ceva_returned_shapes`、`_validate_ceva_phase_artifact_shape`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_binds_the_final_returned_result`、`test_ceva_binds_the_last_statement_for_a_bare_return`、`test_ceva_binds_multiple_returned_results_positionally`、`test_ceva_accepts_a_write_only_artifact_without_results`、既有 `test_ceva_refuses_an_artifact_with_the_wrong_shape`、真实 sidecar 6850 条产物加载 |
| 3 | round09 | 高 | 成立：两个真实 prefill 请求、其余字段匹配的完整报告，`scheduler.max_inflight_requests` 从 1 改为 10 时修复前后两种运行都返回 `validated_in_scope`；旧计时边界只核对执行范围、重叠与微批大小 | 已修复 | `timing_boundary` 新增 `max_inflight_requests` 并与本次配置逐项比对；契约变化使 `CEVA_VALIDATION_SCHEMA_VERSION` 升到 3，旧版报告不再升级成熟度；正向夹具补该字段 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_validation_scope_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_with_another_inflight_limit_keeps_prediction`、`test_validation_report_without_inflight_limit_keeps_prediction`、`test_validation_report_with_an_old_schema_keeps_prediction`、既有正向 `test_validation_report_marks_the_result_validated` |

第 10 轮（round10）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round10 | 严重 | 成立：按评审探针在形状、目标、激活均正确的 SILU 产物里追加 `pim.kv_cache`，修复前 `_apply_compiler_local_shapes()` 返回成功，`parse_phase_chain()` 得到 `activation`/`dma` 两相并额外产生 16,384 B 读和 16,384 B 写；旧检查只把实际语句与计算原语求交，搬运和布局原语被排除在外 | 已修复 | 产物语句改为按执行节点完整允许集合核对：语义原语、登记的融合相位和登记的 GEMM 执行基础设施（barrier、convert_layout、dma_load、dma_store、wram_alloc、wram_load、wram_store）之外一律拒绝，拒绝发生在相位解析与 trace 生成之前；按真实算子编译缓存核对 25,158 份产物的语句组合后确认其余节点没有基础设施语句 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_ARTIFACT_SUPPORT_PRIMITIVES`、`_validate_ceva_compiler_artifact`，删除只查计算原语的 `CEVA_ARTIFACT_COMPUTE_PRIMITIVES`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_refuses_an_unrelated_transfer_primitive`、`test_ceva_accepts_gemm_infrastructure_statements`、既有 `test_ceva_refuses_a_contradictory_compute_primitive`、既有 `test_matching_artifacts_pass_the_ceva_gate`、真实 sidecar 6850 条产物加载 |
| 2 | round10 | 高 | 成立：把全部必需参数来源改为 `assumption` 后，`validate(mode="simulation")` 仍返回 True，配合数值自洽的验证报告时 `_ceva_result_maturity()` 返回 `validated_in_scope`；旧范围核对只比较成本数值与报告字段，不读取 `ceva.parameter_sources`，正向夹具的来源台账也是空的 | 已修复 | 成熟度派生新增参数来源核对：`CEVA_REQUIRED_SOURCE_PATHS` 必须齐全，每条来源的证据非空且等级属于甲方确认、CEVA 实测或拟合，假设/未确认/缺证据/缺路径一律保持 `model_prediction`；正向夹具补齐完整实测来源台账，新增全假设、单假设、缺来源、缺证据、未确认五类反例 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_CALIBRATED_SOURCE_ORIGINS`、`CEVA_REQUIRED_SOURCE_PATHS`、`_ceva_parameter_source_error`、`_ceva_result_maturity`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_assumption_parameter_sources_keep_model_prediction`、`test_one_assumption_parameter_source_keeps_model_prediction`、`test_missing_parameter_source_keeps_model_prediction`、`test_missing_source_evidence_keeps_model_prediction`、`test_unconfirmed_parameter_source_keeps_model_prediction`、既有正向 `test_validation_report_marks_the_result_validated` |

第 11 轮（round11）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round11 | 严重 | 成立：SILU 执行节点输入、输出均为 `[128,64]` 时，修复前 `_apply_compiler_local_shapes()` 接受输入、输出都是 `1×64` 的产物；对同一 artifact 调用 `parse_phase_chain()` 得到 `in_bytes=128`、`out_bytes=128`、`work_units=64`，而 `128×64` 产物为 `16,384 / 16,384 / 8,192`，读写与计算均缩小 128 倍。现有反例只覆盖 `64×64`、`64×32` | 已修复 | 单行/单列内核视图豁免改为按算子登记：只有真实 B 路产物使用该契约的 GATHER、QUANT 放行；其他算子的位置对应输入与最终结果必须逐轴一致，`1×N` 冒充 `M×N` 直接拒绝；前导广播维（元素数不变）仍对所有算子成立 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_SINGLE_ROW_KERNEL_OPS`、`_ceva_artifact_axis_error`、`_ceva_artifact_shape_matches`、`_validate_ceva_phase_artifact_shape`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_ceva_refuses_a_single_row_input_artifact`、`test_ceva_refuses_a_single_row_output_artifact`、`test_ceva_refuses_a_single_row_artifact_before_any_trace_is_generated`、`test_ceva_accepts_the_registered_quant_single_row_kernel`、`test_ceva_accepts_the_registered_gather_single_row_result`、既有 `test_matching_artifacts_pass_the_ceva_gate`、真实 sidecar 6850 条产物加载 |
| 2 | round11 | 高 | 成立：同一 PIM→PIM 链路的 `inter_cluster_only` 与 `is_full_duplex` 由 True 改为 False 后，修复前 `_normalized_links()` 返回相同的四项摘要、`_links_equal()` 返回 True；新增用例把三个语义字段逐一改为 False 后仍为 `validated_in_scope` | 已修复 | 链路规范化加入 `intra_cluster_only`、`inter_cluster_only`、`is_full_duplex`，缺省值与运行时 `LinkConfig` 一致（全双工 true、另两项 false），字段类型不合法直接返回 None；相等判断逐项比较。报告契约升到 `schema_version=4`，旧版报告不再升级成熟度 | `src/scheduler/gene_sim_scheduler.py`（`_normalized_links`、`_links_equal`、`CEVA_VALIDATION_SCHEMA_VERSION`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_link_semantics_must_match_the_run`、既有正向 `test_validation_report_marks_the_result_validated`、既有 `test_validation_report_with_an_old_schema_keeps_prediction` |
| 3 | round11 | 中 | 成立：预测 `-100`、观测 `100`、误差 `200`、相对误差 `2` 的样本，以及预测 `0` 的同类样本，修复前均使 `_ceva_validation_evidence_error()` 返回 None，配合自洽汇总可升级为 `validated_in_scope`；旧实现只要求观测值为正 | 已修复 | 计算和 DMA 校准样本的预测周期都必须是有限正数，零或负值在逐点核对阶段直接拒绝；未对“合法零成本视图”开口子，避免按样本类型静默放宽 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_validation_evidence_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_validation_report_non_positive_prediction_keeps_prediction`、既有正向 `test_validation_report_marks_the_result_validated` |

第 12 轮（round12）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round12 | 高 | 成立：只改主机参数（基础延迟 10 µs → 1,000,000 µs、峰值算力 1 TFLOP/s → 2 TFLOP/s、带宽 100 GB/s → 200 GB/s）时，修复前 `_ceva_cost_model_summary()` 两次结果完全相等且不含主机参数，`_ceva_validation_scope_error()` 也不读取任何主机参数，因此报告结论不随主机参数变化（评审探针在同一完整报告下复现为 `validated_in_scope`）；本轮 CEVA 冒烟的主机存储服务时间 2.7480027955198816 s 恰为 274,800,279,552 B ÷ 100 GB/s，证明 CPUModel 缺省值已进入总时延 | 已修复 | 五份场景显式配置 `cpu_model` 并登记 4 条来源，主机参数纳入 `CEVA_REQUIRED_SOURCE_PATHS`；成本模型摘要与逐项比对增加 `cpu_model`，验证报告契约升到 5；运行结果新增 `host_orchestration_metrics` 单列 CPU 服务时间、存储时间、字节数与算子数；报告新增 `scope.host_orchestration` 独立验证证据，缺失、覆盖不符或数值不自洽均保持 `model_prediction` | `src/config_loader.py`（`CEVA_REQUIRED_SOURCE_PATHS`、`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_cpu_model_summary`、`_ceva_cost_model_summary`、`_ceva_cost_model_error`、`_ceva_host_orchestration_op_types`、`_ceva_host_evidence_error`、`_ceva_error_summary_error`、`_ceva_host_orchestration_metrics`、`_execute_pipeline`）（S）；`src/main.py`（`save_all_results`、`_existing_summary_lines`）（S）；`conf/ceva_*.yaml` × 5（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_main.py`（S） | `test_simulation_requires_explicit_cpu_model`、`test_simulation_rejects_non_positive_cpu_model`、`test_simulation_requires_cpu_model_sources`、`test_repository_ceva_scenarios_declare_cpu_model_sources`、`test_validation_report_missing_cpu_model_keeps_prediction`、`test_validation_report_cpu_model_mismatch_keeps_prediction`、`test_validation_report_without_host_evidence_keeps_prediction`、`test_validation_report_host_evidence_must_match_the_run`、`test_validation_report_with_host_evidence_marks_validated`、`test_host_orchestration_metrics_are_separate_from_pim`、`test_summary_reports_host_orchestration_separately`、`test_summary_omits_host_line_without_ceva_metrics`、`test_saved_summary_records_host_orchestration`、既有正向 `test_validation_report_marks_the_result_validated` |
| 2 | round12 | 中 | 成立：32 B 的 `pim.kv_cache` 相位产物传 `max_bulk_transfer_bytes=16` 时，修复前 LOADN/STOREN 立即数均为 `[32]`；同一产物按 16 B 调用 `compile_phase_chain()` 为 `[16,16]`，说明 `compile_from_pimir()` 的相位分支漏传参数 | 已修复 | 相位分支显式透传 `max_bulk_transfer_bytes`；新增用例核对小上限下立即数为 `[16,16]`、缺省上限仍为 `[32]`，并用 `PIMTrace.estimate_cycle_breakdown()` 与 `phase_cycle_breakdown()` 对拍总周期完全一致；`compile_signature` 原本已包含该上限，无需改动 | `src/pim/pimir_trace.py`（`compile_from_pimir`）（S）；`tests/sim/test_pimir_trace.py`（S） | `test_phase_trace_honours_the_configured_segment_limit`、既有 `test_prepare_gemm_trace_uses_configured_bulk_transfer_limit` |
| 3 | round12 | 低 | 成立：逐份解析五份 CEVA 场景，修复前 `weight_deployment.mappings` 都含通用缺省合入的 `qkv_proj→gpu` 与 `ffn→gpu`；实际资源清点 GPU=0 未受影响，但正式配置仍声明 GPU 权重部署 | 已修复 | 五份场景显式给出 qkv_proj/attention/ffn 全 PIM 映射；`_validate_ceva` 正式校验拒绝任何非 PIM 权重部署项；新增逐份解析断言，并保留实际 GPU 资源与执行算子为零的既有核对 | `src/config_loader.py`（`_validate_ceva`）（S）；`conf/ceva_*.yaml` × 5（S）；`tests/sim/test_config_loader.py`（S） | `test_simulation_rejects_gpu_weight_deployment`、`test_repository_ceva_scenarios_deploy_weights_on_pim_only`、既有 `test_repository_ceva_scenarios_validate_in_structure_mode`、既有 `test_gpu_resource_is_refused` |

第 13 轮（round13）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round13 | 高 | 成立：用真实 7B 放置探针（配置同 `/tmp/ceva-review-round12-smoke.yaml`，指向真实 IR、trace 与 8 DPU 一对一 sidecar）复现：展开后执行图的 CPU 主机节点为 `MODEL_INPUT` 1、`MODEL_OUTPUT` 1、`ALL_REDUCE` 64、`COLLECTIVE_RELAY` 254，修复前 `_ceva_host_orchestration_op_types()` 只返回 `MODEL_INPUT`/`MODEL_OUTPUT`，`_ceva_host_orchestration_metrics()` 的 `op_count=2`；两个函数都遍历未展开的 `self.model_ir`，而 TP 展开只改 `placed_ir` | 已修复 | `_execute_pipeline()` 记录本次实际执行的放置图；主机节点识别、独立证据核对与主机指标优先读取该图。证据要求的算子集合排除合成中转，覆盖 `ALL_REDUCE` 等真实主机节点；主机指标新增 `op_counts` 与 `zero_cost_op_types`，并写入 `summary.json`。新增多分片正反例：证据缺 `ALL_REDUCE` 保持 `model_prediction`，覆盖齐全可升级，把 `COLLECTIVE_RELAY` 当实测样本则拒绝 | `src/scheduler/gene_sim_scheduler.py`（`_execution_ir_or_model`、`_ceva_host_orchestration_op_types`、`_ceva_host_evidence_op_types`、`_ceva_host_evidence_error`、`_ceva_host_orchestration_metrics`、`_execute_pipeline`）（S）；`src/main.py`（`save_all_results`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_main.py`（S） | `test_host_orchestration_reads_the_expanded_execution_graph`、`test_validation_report_without_all_reduce_evidence_keeps_prediction`、`test_validation_report_with_all_reduce_evidence_marks_validated`、`test_host_evidence_must_not_measure_the_synthetic_relay`、`test_saved_summary_records_host_orchestration`、既有 `test_host_orchestration_metrics_are_separate_from_pim` |
| 2 | round13 | 高 | 成立：修复前 `_ceva_cost_model_summary()` 不含 `multipliers`，同一配置把 `pim.tensor_pu_config.multipliers` 从 4 改为 1000 后摘要完全相等；`PIMVPU.peak_flops_per_s()` 直接使用该参数计算整核峰值，报告因此可以在不同峰值假设间复用 | 已修复 | 成本模型摘要增加 `multipliers` 并纳入逐项一致性检查；验证报告契约升到 `schema_version=6`；补充仅改变该参数即保持 `model_prediction` 的反例，并用峰值断言证明摘要绑定的是实际生效值（倍数变化同时改变摘要与 `PIMVPU.peak_flops_per_s()`） | `src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_cost_model_summary`、`_ceva_cost_model_error`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_cost_model_summary_binds_the_effective_peak`、`test_validation_report_missing_multipliers_keeps_prediction`、`test_validation_report_cost_model_mismatch_keeps_prediction`、`test_validation_report_with_an_old_schema_keeps_prediction`、既有 `test_validation_report_marks_the_result_validated` |
| 3 | round13 | 中 | 成立：`CPUModel.profile()` 对 flops/字节均为零的节点返回 10 µs 基础延迟；真实 CEVA 探针中同一 `COLLECTIVE_RELAY` 节点 GPU 时延为 0、CPU 为 10 µs，7B 每次前向 254 个此类节点、69 次前向合计 0.17526 s，形成原 GPU 路径不存在的重复成本 | 已修复 | CPU 执行路径对 `COLLECTIVE_RELAY` 返回与 GPU 一致的零计算、零存储、零开销剖面；通信边仍按链路正常计费。新增 CPU 单元用例与纯 PIM 联合用例（中转执行剖面为零、跨核边成本非零）；真实 CEVA 冒烟的主机服务时间恰好减少 0.17526 s，验证了重复计费已消除 | `src/backend/cpu_backend.py`（`CPUModel.profile`）（S）；`tests/sim/test_cpu_backend.py`、`tests/sim/test_collective_edges.py`（S） | `test_collective_relay_costs_nothing`、`test_relay_on_cpu_host_is_free_while_the_edge_is_billed`、既有 `test_relay_operators_cost_no_extra_time` |

第 14 轮（round14）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round14 | 高 | 成立：真实产物 `.opcompiler_cache/cdb11ad10a30a818.pimir.mlir`（tile 8×512×32）调用 `parse_pimir()` 报“只支持一层或两层 scf.for……实际有 3 层”；同 tile 的两层循环输入 `mac_groups=16384` 又报“超出 14 位立即数上限”。对算子编译缓存里全部 2139 份 GEMM 产物逐份调用旧解析器，662 份失败（3 层 466 份、0 层 132 份、M+K 两层与 N 单层等 64 份），说明大 M 分块产物无法被 GeneSim 消费 | 已修复 | 按步长与 `iter_args` 识别 M/N/K 三层循环及编译器折叠后的 0/1/2 层形态；M 分块由运行时按 `ceil(Tq/tile_m)` 重复（`pimir.tile_m` 来自轨迹元数据，tile_m=1 时与原口径完全一致），trace 体仍覆盖一个 M 分块 × 一个 N 分块；MAC 次数与 K 迭代复用混合进制嵌套循环（`_emit_work_block`），超过 16383 不再拒绝；删除 `_capped`；`trace_metadata()` 增加 `m_extent`/`m_iterations` | `src/pim/pimir_trace.py`（`_ScfLoop`、`PimirProgram.m_extent`/`m_iterations`、`parse_pimir`、`_emit_work_block`、`compile_from_pimir`、`trace_metadata`，删除 `_capped`）（S）；`src/vpu/vpu.py`（`_execute_runtime`、`_runtime_tile_m`）（S）；`tests/sim/test_pimir_trace.py`、`tests/sim/test_vpu.py`（S） | `test_three_level_mnk_nest_recovers_every_extent`、`test_two_loop_mk_nest_recovers_collapsed_n`、`test_fully_unrolled_kernel_is_one_output_tile`、`test_rejects_a_loop_that_matches_no_tile_size`、`test_metadata_exposes_the_m_tiling`、`test_large_mac_group_count_is_fully_billed`、`test_large_k_iteration_count_is_fully_billed`、`test_three_level_nest_bills_every_m_n_k_tile`、`test_doubling_m_n_or_k_doubles_the_billed_macs`、`test_real_large_m_artifact_is_consumed_end_to_end`、`test_gemm_repetitions_follow_the_trace_m_tiling`、`test_gemm_without_m_tiling_metadata_keeps_tq_repetitions`、`test_gemm_profile_memory_bytes_follow_the_m_tiling` |
| 2 | round14 | 高 | 成立：放置 sidecar 顶层没有物理 MRAM/预留/可用量（version 3）；CEVA 画像 8,000,000,000 B、预留 0，而 sidecar 实际消费的 5 份 GEMM 产物声明 `pim.mram-bytes=4294967296`，`_validate_ceva_gemm_artifact()` 仍返回成功，完整 CEVA 冒烟照常完成 | 已修复 | G 的放置 sidecar 升到 version 4，顶层导出 `hardware{mram_bytes_per_dpu,sys_reserve_bytes,usable_mram_bytes}` 并校验预留小于物理量；导出脚本透传 `HwBudget.sys_reserve_bytes`。GeneSim CEVA 加载时逐项核对三者自洽、与 `ceva.local_memory_bytes`/`ceva.sys_reserve_bytes` 一致；GEMM 分块产物要求 `pim.mram-bytes` 存在且等于画像字节数。两类核对都在生成 trace 之前 | `genesim_bridge/placement_export.py`（`export_placement_to_genesim`）、`scripts/export_pp_placement.py`（`main`）（G）；`src/scheduler/gene_sim_scheduler.py`（`_validate_ceva_hardware_budget`、`_load_compiler_placement`、`_validate_ceva_gemm_artifact`）（S）；测试 `tests/test_placement_export.py`（G）、`tests/sim/test_compiler_placement.py`、`tests/sim/test_collective_edges.py`（S） | `test_sidecar_exports_the_same_byte_budget`、`test_sidecar_rejects_a_reserve_larger_than_capacity`、`test_ceva_requires_the_sidecar_hardware_budget`、`test_ceva_refuses_a_mram_budget_mismatch`、`test_ceva_refuses_a_reserve_mismatch`、`test_ceva_refuses_an_inconsistent_usable_budget`、`test_ceva_refuses_a_gemm_artifact_with_another_mram_budget`、`test_ceva_requires_pim_mram_bytes_in_the_gemm_artifact` |

第 15 轮（round15）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round15 | 严重 | 成立：用真实产物 `.opcompiler_cache/cdb11ad10a30a818.pimir.mlir`（tile 8×512×32、n_extent=32000）复现，sidecar `kernel_tile_n=32` 与缺失两种情况下 `_apply_compiler_local_shapes()` 均成功，运行时按 conf 的 32 得到 1000 次 N 循环，按产物分块应为 63 次，计算与搬运工作量约放大 15.87 倍 | 已修复 | CEVA 严格模式在 `_validate_ceva_gemm_artifact()` 末尾强制 GEMM 条目 `kernel_tile_n` 为正整数且等于产物 `pim.tile-n`，缺失或不一致在生成 trace 前拒绝；`PIMVPU._execute_runtime()` 对 CEVA 的 GEMM 再读取轨迹元数据 `pimir.tile_n` 复核运行分块，不一致直接报错；非 CEVA 路径保持原行为。新增合成与真实大 M 产物的 VPU 端到端用例，断言 N 循环 63 次、MAC 与 DMA 字节均按 512 分块计 | `src/scheduler/gene_sim_scheduler.py`（`_validate_ceva_gemm_artifact`）（S）；`src/vpu/vpu.py`（`_check_gemm_artifact_tile`、`_execute_runtime`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_vpu.py`（S） | `test_ceva_requires_the_sidecar_kernel_tile_n`、`test_ceva_refuses_a_kernel_tile_n_mismatch`、`test_gemm_binds_the_compiled_n_tile`、既有 `test_ceva_accepts_gemm_infrastructure_statements`、`test_matching_artifacts_pass_the_ceva_gate`、`test_ceva_requires_compiler_local_shapes`、`test_ceva_requires_pim_mram_bytes_in_the_gemm_artifact`、真实 sidecar 6850 条加载 |
| 2 | round15 | 高 | 成立：同一份完整验证报告在 `host_relay` 与 `direct` 下均返回 `validated_in_scope`；`_ceva_validation_scope_error()` 只比较链路参数与 HOST 固定开销，不核对 `pim.inter_cluster_communication` | 已修复 | `scope.communication_model` 新增 `inter_cluster_communication`，与本次生效配置逐项比较，不一致或缺失即保持 `model_prediction`；该参数纳入 `CEVA_REQUIRED_SOURCE_PATHS`，五份场景补来源；正式校验只接受 `direct`/`host_relay`；报告契约升到 `schema_version=7`。新增 direct/host_relay 互换、路由缺失、路由与链路组合不一致三类反例 | `src/config_loader.py`（`CEVA_REQUIRED_SOURCE_PATHS`、`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_validation_scope_error`）（S）；`conf/ceva_*.yaml` × 5（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_config_loader.py`（S） | `test_validation_report_routing_mode_must_match_the_run`、`test_validation_report_without_routing_mode_keeps_prediction`、`test_validation_report_routing_mode_must_match_the_links`、`test_repository_ceva_scenarios_declare_routing_source`、`test_simulation_requires_routing_source`、`test_simulation_rejects_unknown_routing_mode`、既有 `test_validation_report_marks_the_result_validated`、`test_validation_report_with_an_old_schema_keeps_prediction` |

第 16 轮（round16）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round16 | 严重 | 成立：用评审探针 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python /tmp/ceva_review_round16_loop_probe.py` 复现，tile_n=1、N=5000 时所需 N 循环为 5000，实际只绑定 4095；应得 `0.000327005 s`，实得 `0.000267818 s`，仅为正确值的 0.819。根因有三处：`compile_from_pimir()` 对分块 GEMM 固定发短间接循环，`compile_softmax()` 不接受长循环参数，`use_long_loop_counts` 只到达手写模板；`PIMTrace.set_tile_count()` 又用 `min(count, 0xFFF)` 把运行时绑定统一截断，CEVA 核心再次走同一入口。按 4095/4096/5000 三档写的边界用例在修复前全部失败：短循环超过 4095 被 `min()` 截断，长循环轨迹的 hi/lo 寄存器则从未被写入（保持 1 次），MAC、DMA 字节与总周期都停在错误值 | 已修复 | ① `compile_from_pimir()` 增加 `use_long_loop` 并按需发 24 位长循环；`compile_softmax()` 同样支持；调度器和两个 `pim_compiler` 前端把 `pim.compile.use_long_loop_counts` 传给 GEMM pimir、SOFTMAX 及所有接受该参数的模板；② `PIMTrace.set_tile_count()` 识别长循环轨迹并写 hi/lo 完整次数，超过 24 位显式报错；③ CEVA 核心/后端允许把遗留短循环轨迹在绑定时就地升级为长循环，非 CEVA 路径保持 12 位兼容截断但改为告警；④ trace 发射语义变化，`schema_version` 从 9 升到 10，旧短循环缓存自动失效重编 | `src/pim/pimir_trace.py`（`compile_from_pimir`）（S）；`src/pim/pim_compiler.py`（`compile_softmax`、CLI 参数透传）（S）；`src/pim/pim_trace_loader.py`（`set_tile_count`、`_uses_long_indirect_loop`、`_upgrade_indirect_loops_to_long`）（S）；`src/pim/pim_core.py`（`ALLOW_LONG_LOOP_UPGRADE`、`PIMTracePIMCore.execute`）（S）；`src/backend/pim_backend.py`（`execute_runtime`）（S）；`src/scheduler/gene_sim_scheduler.py`（`base_compile_signature`、`_prepare_pim_traces`）（S）；`scripts/pim_compiler.py`（SOFTMAX 编译参数）（S）；`tests/sim/test_pim_trace_loader.py`、`tests/sim/test_pimir_trace.py`、`tests/sim/test_pim_compiler.py`、`tests/sim/test_pim_backend.py`、`tests/sim/test_vpu.py`（S） | `test_long_indirect_loop_binds_counts_past_the_12_bit_limit`、`test_short_indirect_loop_keeps_the_legacy_12_bit_clamp`、`test_long_indirect_loop_rejects_counts_past_24_bits`、`test_long_tile_count_halves_must_fit_12_bits`、`test_outer_loop_uses_the_long_encoding_when_configured`、`test_softmax_loop_uses_the_long_encoding_when_configured`、`test_gemm_long_loop_binds_every_n_iteration_past_12_bits`、`test_gemv_long_loop_binds_every_kv_iteration_past_12_bits`、`test_softmax_long_loop_binds_every_kv_iteration_past_12_bits`、`test_ceva_upgrades_a_short_trace_to_bill_every_n_iteration`、`test_non_ceva_keeps_the_legacy_12_bit_clamp`、既有 `test_execute_runtime_binds_loop_count_and_repetitions` |

第 17 轮（round17）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round17 | 中 | 成立：先写反例测试，从完整 CEVA 画像中分别删除 `CPU_to_HOST`、`HOST_to_CPU` 后，修复前 `validate(mode="simulation")` 均返回 True；重跑评审探针 `/tmp/ceva_round17_boundary_probe.py`（真实 64 核场景与真实放置 sidecar）同样得到 `simulation_valid=True`，而 `CommLib.transfer("CPU","PIM",1)` 与反向传输均为 `inf`，日志为 `No relay path`。根因是正式校验只核对 HOST↔PIM（direct 时再加 PIM→PIM），没有核对 CPU↔HOST，而 CEVA 的 MODEL_INPUT/MODEL_OUTPUT/ALL_REDUCE 会落在 CPU 编排核上 | 已修复 | 正式校验（`mode="simulation"`）在 HOST↔PIM 要求之后追加 CPU→HOST 与 HOST→CPU 双向链路要求，缺任一条直接拒绝；结构校验（`mode="structure"`）仍允许先缺链路接线。新增反例分别删除两条链路，并断言合法配置的 CPU↔PIM 经 HOST 中转成本有限 | `src/config_loader.py`（`_validate_ceva`）（S）；`tests/sim/test_config_loader.py`（S） | `test_simulation_requires_cpu_host_links`、`test_valid_profile_has_a_finite_cpu_pim_route`、既有 `test_simulation_rejects_missing_links`、既有 `test_complete_profile_passes_simulation_validation` |
| 2 | round17 | 中 | 成立：先写通用摘要用例，修复前 `save_all_results(_base_results())` 生成的 `summary.json` 含 `"result_maturity": null`，与基线 `src/main.py` 的键集合不同；现有通用产物 `genesim/results/summary.json:4` 也是该空值的实证。根因是 `save_all_results()` 无条件写入 `results.get("result_maturity")`，而非 CEVA 结果没有该键 | 已修复 | 摘要字面量删除该字段，改为仅在 `"result_maturity" in results` 时写入；调度器只在 `_is_ceva()` 时设置该键，因此通用摘要恢复原键集合，CEVA 摘要保留 `model_prediction`/`validated_in_scope`。新增通用与 CEVA 两类回归断言 | `src/main.py`（`save_all_results`）（S）；`tests/sim/test_main.py`（S） | `test_generic_summary_omits_result_maturity`、`test_ceva_summary_keeps_result_maturity`、既有 `test_saved_summary_records_host_orchestration` |

第 18 轮（round18）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round18 | 高 | 成立：用评审复现脚本 `/tmp/round18_probe.py` 在真实 64 核冒烟配置上把 `scheduler.cost_sidecar_file` 指向不存在的文件，修复前 `validate(mode="simulation")` 返回 True，`_load_cost_sidecar()` 仅告警并返回 0，证明声明的成本输入完全未被消费 | 已修复 | 正式校验（`mode="simulation"`）对非空成本 sidecar 路径检查存在性与可读性，缺失即校验失败；调度器加载端在 CEVA 下对缺失或不可读文件直接抛 `RuntimeError`，通用配置保留原“告警后返回 0”的兼容行为。新增配置缺失反例、绕过校验直接加载反例、通用告警保持用例，以及合法成本 sidecar 确实标注 `source_name`/`mram_traffic_bytes` 的正向断言 | `src/config_loader.py`（`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_load_cost_sidecar`）（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_cost_sidecar.py`（S） | `test_simulation_rejects_a_missing_cost_sidecar`、`test_simulation_accepts_an_existing_cost_sidecar`、`test_ceva_missing_cost_sidecar_raises`、`test_non_ceva_missing_cost_sidecar_still_warns`、`test_ceva_cost_sidecar_annotates_the_real_operators` |
| 2 | round18 | 中 | 成立：本轮 CEVA 冒烟结果目录没有 `run_manifest.json`，`summary.json` 也没有 IR/放置/请求哈希、PIM MLIR 集合、图编译器或 FlagTree 版本；`/tmp/round18_probe.py` 对 round18 冒烟结果检查 `run_manifest.json` 不存在、摘要无清单字段，确认无法凭结果复现同一条编译链 | 已修复 | 新增 `src/run_manifest.py`：CEVA 运行清单记录配置文件名与 SHA256、合并默认值后的完整配置、IR/放置/成本/请求文件的路径与 SHA256、唯一 PIM MLIR 的路径与 SHA256、硬件预算、图编译器提交号与脏标记、FlagTree 源码提交号与脏标记、实际 `triton-opt` 路径与 SHA256。`main` 在创建 Scheduler 前核对工具链身份并校验全部 PIM MLIR 哈希，版本缺失或哈希变化即退出；调度器运行结束把清单挂到结果，`save_all_results()` 写 `run_manifest.json` 并在摘要登记文件名。通用结果不写清单、不改摘要键集合 | `src/run_manifest.py`（新增：`build_ceva_run_manifest`、`compiler_toolchain_identity`、`validate_ceva_toolchain`、`artifact_entry`、`pimir_resolver`、`sha256_file`）（S）；`src/config_loader.py`（`resolved_config`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_ceva_run_manifest`、`run`）（S）；`src/main.py`（`main`、`save_all_results`）（S）；`tests/sim/test_run_manifest.py`（新增）、`tests/sim/test_main.py`（S） | `test_manifest_records_inputs_pim_mlir_and_toolchain`、`test_manifest_rejects_a_changed_pim_mlir_hash`、`test_evaluation_entry_rejects_a_changed_pimir_hash`、`test_toolchain_identity_rejects_a_missing_flagtree_source`、`test_toolchain_identity_rejects_a_missing_triton_opt`、`test_toolchain_identity_rejects_pimir_outside_a_source_repo`、`test_toolchain_identity_rejects_a_mixed_cache`、`test_toolchain_identity_matches_the_installed_toolchain`、`test_ceva_run_manifest_is_saved_with_the_results`、`test_generic_results_do_not_write_a_run_manifest` |
| 3 | round18 | 中 | 成立：`/tmp/round18_probe.py` 把 `pim.cluster_memory_capacity_gb` 分别设为正负无穷，修复前 `validate(mode="structure")` 与 `resolve_pim_capacity_limits()` 都抛出未捕获的 `OverflowError: cannot convert float infinity to integer`；现有非有限值测试只覆盖周期成本 | 已修复 | `_validate_ceva` 与 `_ceva_capacity_limits` 在整数换算前先做 `math.isfinite` 检查，并把 `OverflowError` 纳入转换异常处理；结构与正式校验对正负无穷、NaN 返回 False，直接容量解析抛出带 `pim.cluster_memory_capacity_gb` 字段名的 `ValueError`。结构模式即 `--dry-run` 使用的校验入口，不再异常退出 | `src/config_loader.py`（`_validate_ceva`）（S）；`src/pim/capacity.py`（`_ceva_capacity_limits`）（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_pim_capacity.py`（S） | `test_non_finite_capacity_is_rejected_without_crashing`、`test_ceva_capacity_rejects_non_finite_gib` |

第 19 轮（round19）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round19 | 高 | 成立：用真实 `models/llama2_7b_pimir.ir` 与 `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 复现，展开前模型图 GEMM 的 `(N,K)` 为 `(4096,4096)`、`(4096,11008)`、`(11008,4096)`，TP2 展开后的执行图只有 `(2048,4096)`、`(4096,2048)`、`(5504,4096)`、`(4096,5504)`。修复前 `_ceva_run_matmul_shape_bounds()` 返回 `n/k=(4096,11008)` 且 `_ceva_sample_shape_error("GEMM",[726,11008,11008],"prefill")` 返回 None（放行不存在的全局形状），而真实本地样本 `[726,2048,4096]` 会因低于全局下界被拒；三处入口都读展开前的 `model_ir` | 已修复 | 三个入口统一改用 `_execution_ir_or_model()`：有本次执行图时取 TP 展开后的分片形状，未运行或预测器路径没有执行图时回退 `model_ir`。GEMM 的 M/N/K 包络与非 GEMM 的 prefill/decode 形状均按本地分片计算，SILU 从 `(726,11008)` 变为 `(726,5504)`。新增 Llama2-7B TP2 夹具与三条用例，覆盖本地范围、全局 11008 样本拒绝和四种真实本地分片样本接受，并走完整 `_ceva_result_maturity()` 派生 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_run_shapes_of`、`_ceva_run_matmul_stage_m`、`_ceva_run_matmul_shape_bounds`）（S）；`tests/sim/test_compiler_placement.py`（S） | `test_tp2_shape_ranges_come_from_the_execution_graph`、`test_tp2_validation_scope_rejects_the_global_11008_shape`、`test_tp2_validation_scope_accepts_every_real_local_shard` |
| 2 | round19 | 中 | 成立：`GENESIM_PIM_CLOCK_PERIOD_NS=1.26` 时 `config.get("pim.clock_period_ns")` 返回 1.26，但 `config.resolved_config()["pim"]["clock_period_ns"]` 仍为 0.63；布尔覆盖 `GENESIM_SCHEDULER_PREFILL_DECODE_OVERLAP=true` 同样只进入 `get()`，运行清单归档的仍是未覆盖副本。现有清单测试没有环境覆盖路径 | 已修复 | `resolved_config()` 在深拷贝后递归应用 `GENESIM_*` 覆盖，类型按当前值用与 `get()` 相同的 `_cast_env_value()` 推断；清单的 `resolved_config`、硬件预算与输入产物路径因此都读取同一份生效配置。新增清单用例同时覆盖时钟、请求 trace 路径、布尔字段与配置核数，断言实际执行取值与 `run_manifest.json` 一致 | `src/config_loader.py`（`resolved_config`、`_apply_env_overrides`）（S）；`tests/sim/test_run_manifest.py`（S） | `test_manifest_resolved_config_records_environment_overrides` |

第 20 轮（round20，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round20 | 严重 | 成立：用真实 `bpath-31471efb9ea37b19.pimir.mlir`（QUANT 单行内核）与 `bpath-90a38eac88f36ccb.pimir.mlir`（GATHER 单行内核）复现，修复前模板维度为空、所有张量轴绑定为 None（Tq=1/2/128 探测一致），Tq=2 的 QUANT 仍按 20,864 B 计；GATHER 的索引与结果同样恒定，查表字节与结果字节不随行数增长。修复后从真实冒烟轨迹读取模板：QUANT/GATHER 均带 `dimensions=['Tq']`、`dimension_sources={'Tq': ['in',0,0]}`；QUANT 在 Tq=1/2/726 下内存字节为 20,864/41,728/15,147,264 B，总周期 101,288/202,501/73,539,408；GATHER 为 1,056,772/1,064,968/6,998,872 B（查表 1,048,576 B 恒定，索引 4 B/行、结果 8,192 B/行随行数增长） | 已修复 | ① `phase_runtime_template()` 增加 `row_repeat` 契约：运行行轴名字、维度在算子执行形状中的来源、行视图操作数/结果下标；行视图张量首维绑定运行行轴，首维不是 1 时显式报错，查表/参数张量不登记行视图、保持编译期形状。② 调度器新增 `CEVA_SINGLE_ROW_KERNEL_CONTRACT`（QUANT 全部操作数与结果、GATHER 仅索引与结果；行轴取执行输入 0 第 0 维）与 `_single_row_kernel_repeat()`，在 `_phase_runtime_metadata()` 生成模板时传入。③ `compile_signature` 的 `schema_version` 从 10 升到 11，旧模板缓存自动失效重编 | `src/pim/pimir_trace.py`（`_row_view`、`_template_tensors`、`_row_repeat_spec`、`phase_runtime_template`）（S）；`src/scheduler/gene_sim_scheduler.py`（`CEVA_SINGLE_ROW_KERNEL_CONTRACT`、`_single_row_kernel_repeat`、`_phase_runtime_metadata`、`base_compile_signature`）（S）；`tests/sim/test_pimir_trace.py`、`tests/sim/test_vpu.py`、`tests/sim/test_compiler_placement.py`（S） | `test_single_row_kernel_repeats_the_row_view_tensors`、`test_gather_repeat_keeps_the_lookup_table_constant`、`test_row_view_tensor_must_start_with_one_row`、`test_single_row_quant_kernel_bills_every_runtime_row`、`test_single_row_gather_kernel_keeps_the_table_constant`、`test_quant_template_binds_every_row_view_tensor`、`test_gather_template_keeps_the_table_outside_the_row_axis` |
| 2 | round20 | 高 | 成立：修复前 `_ceva_validation_scope_error()` 的 `package` 只核对 IR、放置 sidecar 和请求 trace 哈希，不读取 `scheduler.cost_sidecar_file`；同一份完整报告分别指向两份内容不同的成本 sidecar 时都返回 `validated_in_scope`。修复后报告缺少 `package.cost_sidecar`、声明 `used=false` 但本次运行配置了成本 sidecar、哈希与磁盘文件不一致、或本次未配置却声明 `used=true`，均保持 `model_prediction`；登记正确哈希时恢复 `validated_in_scope` | 已修复 | 验证报告契约升级到 `schema_version=8`：`package.cost_sidecar` 必须显式登记 `used` 与 `sha256`；`_ceva_cost_sidecar_error()` 按本次 `scheduler.cost_sidecar_file` 的实际内容核对。运行清单与成熟度门禁共用新增的 `run_manifest.cost_sidecar_entry()`，报告哈希、清单 `inputs.cost_sidecar` 与实际加载文件由同一函数产生 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_ceva_cost_sidecar_error`、`_ceva_validation_scope_error`）（S）；`src/run_manifest.py`（`cost_sidecar_entry`、`build_ceva_run_manifest`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_run_manifest.py`（S） | `test_validation_report_cost_sidecar_must_match_the_run`、`test_validation_report_cannot_claim_an_unused_cost_sidecar`、`test_validation_report_unused_cost_sidecar_marks_validated`、`test_validation_report_with_an_old_schema_keeps_prediction`、`test_manifest_cost_sidecar_entry_matches_the_loaded_file`、既有正向 `test_validation_report_marks_the_result_validated` |
| 3 | round20 | 中 | 成立：修复前 `GENESIM_PIM_CLOCK_PERIOD_NS=1.26` 时执行值与清单值均为 1.26，来源证据仍是“沿用现有配置的 0.63 ns 假设”，成熟度仍返回 `validated_in_scope`；周期成本（MAC 1→5）与通信参数（HOST 中转 100→250 ns）的环境覆盖也未进入来源核对，其中周期成本的逐项覆盖甚至不会被 `get("pim.pim_trace_cycle_costs")` 读到。修复后上述三类覆盖在来源台账未登记匹配的 `value` 时均保持 `model_prediction`，登记 `value` 后恢复 `validated_in_scope`；运行清单新增 `env_overrides`，逐项记录环境变量、生效值、配置原值与 `evidence_applies`，旧证据不再默认适用于新值 | 已修复 | ① `GeneSimConfig` 新增 `env_overrides()`、`configured_value()`、`parameter_source_applicability()`；`get()` 返回整段映射时递归应用子键覆盖，周期成本逐项覆盖因此真正生效。② `_ceva_parameter_override_error()` 要求被覆盖参数在来源台账中登记与生效值一致的 `value`（覆盖值等于配置原值时原证据仍适用）。③ 运行清单新增顶层 `env_overrides`，同时覆盖配置树未声明但来源台账登记的参数 | `src/config_loader.py`（`_values_match`、`get`、`_apply_env_overrides`、`env_overrides`、`configured_value`、`parameter_source_applicability`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_ceva_parameter_override_error`、`_ceva_parameter_source_error`）（S）；`src/run_manifest.py`（`_env_override_entries`、`build_ceva_run_manifest`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_run_manifest.py`（S） | `test_environment_clock_override_needs_matching_source_value`、`test_environment_cycle_cost_override_needs_matching_source_value`、`test_environment_comm_override_needs_matching_source_value`、`test_manifest_lists_environment_overrides_with_source_applicability`、`test_manifest_resolved_config_records_environment_overrides`、既有 `test_validation_report_marks_the_result_validated` |

第 21 轮（round21，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round21 | 高 | 成立：给 `/tmp/ceva-round20-smoke.yaml` 补上本轮新增来源后，`validate(mode="simulation")` 为 True；`override(autotune={"enabled": True})` 与 `GENESIM_AUTOTUNE_ENABLED=true` 两条入口在修复前都仍为 True；两个新用例修复前分别报 `True is not false` 与 `VPUs not set.`（未在搜索前拦截） | 已修复 | CEVA 正式校验（simulation）直接拒绝 `autotune.enabled=true`，删除只限制最终验证后端的不可达分支；`run()` 对 CEVA 配置在进入 `_autotune_placement()` 前无条件拒绝。通用场景的自动调参路径不变 | `src/config_loader.py`（`_validate_ceva`）（S）；`src/scheduler/gene_sim_scheduler.py`（`run`）（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_compiler_placement.py`（S） | `test_simulation_rejects_autotune_placement_search`、`test_ceva_run_rejects_autotune_before_searching`、`test_ceva_fixed_placement_still_reaches_the_pipeline`、既有 `test_validation_accepts_enabled_autotune` |
| 2 | round21 | 高 | 成立：`{"operators": {}}` 的空成本文件修复前 `annotated=0`、有效 MRAM 覆盖 0，成熟度仍为 `validated_in_scope`；`{"operators": {"1": {}}}` 的空条目前修复前把 `annotated` 增到 1；只覆盖部分算子且未声明模板路径的 sidecar 同样可升级。四个新增/改写用例修复前失败（`validated_in_scope != model_prediction`），原正向用例把空文件当正例 | 已修复 | `_load_cost_sidecar()` 生成消费台账：实际落值的 `source_name`/`mram_traffic_bytes` 计数、`cost_operators`、`coverage` 五计数、未覆盖算子；空条目不再增加 `annotated`。台账进入运行结果与 `run_manifest.json`，成熟度要求有效成本算子非零、未覆盖算子显式登记在 `coverage.template`，并要求报告 `package.cost_sidecar.coverage` 与台账逐项一致；验证报告契约升到 `schema_version=9`。`used=false` 且未配置的合法路径保持不变 | `src/scheduler/gene_sim_scheduler.py`（`_load_cost_sidecar`、`_empty_cost_sidecar_ledger`、`_build_cost_sidecar_ledger`、`_cost_sidecar_consumption`、`_ceva_cost_sidecar_error`、`run`、`_ceva_run_manifest`）（S）；`src/run_manifest.py`（`build_ceva_run_manifest`）（S）；`tests/sim/test_cost_sidecar.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_run_manifest.py`（S） | `test_consumption_ledger_records_applied_fields_and_coverage`、`test_empty_sidecar_has_no_costs_landed`、`test_empty_entry_is_not_counted_as_annotated`、`test_declared_template_operators_are_not_uncovered`、`test_validation_report_cost_sidecar_must_match_the_run`、`test_cost_sidecar_without_any_cost_cannot_validate`、`test_cost_sidecar_with_only_empty_entries_cannot_validate`、`test_undeclared_partial_cost_coverage_keeps_prediction`、`test_declared_partial_cost_coverage_limits_the_scope`、`test_report_cost_coverage_must_match_the_run`、`test_manifest_records_cost_sidecar_consumption`、既有 `test_mram_bytes_replace_fitted_data_bytes_at_runtime` |
| 3 | round21 | 中 | 成立：同一份报告在 `pim.compile.use_long_loop_counts=true/false` 下修复前都返回 `validated_in_scope`，两份 `_ceva_cost_model_summary()` 完全相同且不含循环编码；短/长循环入口单价设为 1/100 时 10 次迭代实际周期为 12/111，说明切换编码会改变成本。新增用例修复前失败 | 已修复 | 成本模型新增 `loop_encoding`（开关、动态/固定循环原语名），`_ceva_cost_model_error` 逐项核对；`CEVA_REQUIRED_SOURCE_PATHS` 新增 `pim.compile.use_long_loop_counts`，五份场景登记来源；环境覆盖按生效值核对来源证据。`PIMTrace._upgrade_indirect_loops_to_long()` 写入 `runtime_loop_upgrade` 元数据，摘要把实际生效的 `LOOP_BEGIN_INDIRECT_LONG` 一并登记，覆盖运行时短→长升级 | `src/config_loader.py`（`CEVA_REQUIRED_SOURCE_PATHS`）（S）；`src/scheduler/gene_sim_scheduler.py`（`_ceva_cost_model_summary`、`_ceva_cost_model_error`、`_ceva_runtime_loop_upgraded`）（S）；`src/pim/pim_trace_loader.py`（`_upgrade_indirect_loops_to_long`）（S）；`conf/ceva_*.yaml` × 5（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_compiler_placement.py`、`tests/sim/test_pimir_trace.py`、`tests/sim/test_vpu.py`（S） | `test_loop_encoding_is_part_of_the_cost_model_contract`、`test_cost_model_summary_binds_the_emitted_loop_primitives`、`test_cost_model_summary_records_runtime_loop_upgrades`、`test_environment_loop_encoding_override_needs_matching_source`、`test_simulation_requires_loop_encoding_source`、`test_repository_ceva_scenarios_declare_loop_encoding_source`、`test_loop_encoding_price_lands_in_the_actual_cycles`、既有 `test_ceva_upgrades_a_short_trace_to_bill_every_n_iteration` |
| 4 | round21 | 中 | 成立：修复前 `zone_scale(1, {"weight": 1.25, ...})` 返回 src2=2（静默编成 1 倍）、`zone_scale_for("kv", {"kv": 0.5, ...})` 返回 1（0.5 倍），`compile_from_pimir()` 与 `phase_cycle_breakdown()` 对同一张非法表都不报错；合法 1.5 倍编码为 3。新增用例修复前 `ValueError not raised` | 已修复 | 在 `pim_isa` 新增 `validate_zone_cycle_factors()`，配置校验与 `zone_scale*`、`compile_from_pimir()`、`compile_phase_chain()`、`phase_cycle_breakdown()` 共用；拒绝非映射、非数值/布尔、非有限、低于 1、非 0.5 整数倍和 `2*factor>4095`，编码前不再 `round()` 外部值。配置入口额外要求三个区间齐全，直接编译接口允许只给实际使用的区间（保留既有部分表用法）；`zone=None` 仍返回 0 | `src/pim/pim_isa.py`（`ZONE_NAMES`、`validate_zone_cycle_factors`、`zone_scale`、`zone_scale_for`）（S）；`src/pim/pimir_trace.py`（`compile_from_pimir`、`compile_phase_chain`、`phase_cycle_breakdown`）（S）；`src/config_loader.py`（`_validate_zone_cycle_factors`）（S）；`tests/sim/test_zone_cycles.py`（S） | `test_encoding_rejects_invalid_external_factors`、`test_compile_and_eval_entries_reject_the_same_invalid_factors`、`test_legal_half_step_factor_still_encodes_exactly`、既有 `test_configured_factors_change_the_encoded_scale`、`test_compile_phase_chain_uses_configured_factors`、`TestZoneFactorValidation` 全部用例 |
| 5 | round21 | 中 | 成立：对真实 CEVA 配置构造 `GeneSimConfig(...).override(ceva={"engines_per_core": 4})` 后，`config_path` 指向已删除的临时 YAML；修复前 `build_ceva_run_manifest()` 报 `RuntimeError: CEVA 运行清单要求的 配置文件 不存在`，无参数 `override()` 同样复现 | 已修复 | `GeneSimConfig` 新增 `source_path`（原始持久文件）与 `override_chain`（逐层覆盖的顶层配置段），`override()` 把两者传给新实例；`run_manifest.config_entry()` 对持久文件仍按磁盘路径与 SHA256 归档，对内存配置按 `resolved_config()` 的规范化 JSON 计算 SHA256、`path` 置空、`origin=in_memory`，原始来源另存 `source`，覆盖链记 `override_sections`。输入产物哈希核对逻辑不变 | `src/config_loader.py`（`__init__`、`override`）（S）；`src/run_manifest.py`（`resolved_config_sha256`、`config_entry`、`build_ceva_run_manifest`）（S）；`tests/sim/test_run_manifest.py`（S） | `test_manifest_records_the_persistent_config_file`、`test_manifest_accepts_an_overridden_in_memory_config`、`test_manifest_records_the_override_chain`、`test_in_memory_config_identity_includes_environment_overrides`、既有 `test_manifest_records_inputs_pim_mlir_and_toolchain`、`test_manifest_resolved_config_records_environment_overrides` |

第 22 轮（round22，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 修复状态 | 修复方式 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|------------------|--------------|
| 1 | round22 | 严重 | 成立：用评审探针 `/tmp/ceva_review_round22_mac_probe.py` 复现，M=2、N=4、K=3、mac_width=2 时原始 MAC 次数 24，相位路径实得 12 次指令，分块路径 `PimirProgram.mac_groups()` 为 16 次，少计 25%；`compile_phase_chain()` 与 `phase_cycle_breakdown()` 共用 `_phase_work_units()` 的全局取整入口，属于运行时也生效的少计 | 已修复 | `_phase_work_units()` 的乘加分支改为调用新增的 `_matmul_mac_groups_from_shapes()`，按 `M×N×ceil(K/mac_width)` 逐输出取整；M/K/N 形状解析从 `_matmul_mac_count_from_shapes()` 抽成 `_matmul_dims_from_shapes()`，编译指令与解析周期共用同一实现。新增非整除 K、transposeB、带 scale 量化矩阵乘和运行时模板对拍四条用例，保留整除形状与大工作量不截断的既有用例。指令发射语义变化，`base_compile_signature` 的 trace `schema_version` 从 11 升到 12，旧缓存自动失效重编 | `src/pim/pimir_trace.py`（`_matmul_dims_from_shapes`、`_matmul_mac_groups_from_shapes`、`_phase_work_units`）（S）；`src/scheduler/gene_sim_scheduler.py`（`base_compile_signature`）（S）；`tests/sim/test_pimir_trace.py`（S） | `test_tail_k_bills_one_accumulate_per_output`、`test_tail_k_on_transposed_b_bills_per_output`、`test_runtime_template_tail_k_bills_per_output`、`test_quantized_matmul_tail_k_bills_per_output`、既有 `test_matmul_work_uses_m_n_k`、`test_doubling_k_doubles_matmul_compute`、`test_doubling_n_doubles_matmul_compute`、`test_doubling_transposed_n_doubles_matmul_compute`、`test_large_workload_is_not_truncated`、`test_quantized_matmul_with_scales_uses_matrix_operands` |
| 2 | round22 | 高 | 成立：用评审探针 `/tmp/ceva_review_round22_cost_probe.py` 复现，5 个算子中仅 1 个有正数 `mram_traffic_bytes`，其余 4 个条目只有 `op_type`，台账仍得到 `sidecar_operators=5、cost_operators=1、uncovered_operators=0`，成熟度返回 `validated_in_scope`；空条目被条目集合当作覆盖证据 | 已修复 | `_build_cost_sidecar_ledger()` 的未覆盖集合改为“模型算子 − 有效搬运成本算子 − 显式模板算子”：空条目和仅 `source_name` 条目不再算覆盖，`source_name` 只作为内核身份进入 `annotated_operators`，不与真实搬运成本混同。`_ceva_cost_sidecar_error()`、`run()` 结果与运行清单继续共用同一 `_cost_sidecar_consumption()` 台账；验证报告契约升到 `schema_version=10`，报告覆盖范围必须与台账逐项一致 | `src/scheduler/gene_sim_scheduler.py`（`CEVA_VALIDATION_SCHEMA_VERSION`、`_build_cost_sidecar_ledger`）（S）；`tests/sim/test_cost_sidecar.py`、`tests/sim/test_compiler_placement.py`（S） | `test_mixed_valid_and_empty_entries_leave_empty_ones_uncovered`、`test_source_name_only_entry_is_not_cost_coverage`、`test_empty_entry_is_not_counted_as_annotated`、`test_cost_sidecar_with_mixed_empty_entries_cannot_validate`、`test_cost_sidecar_with_only_source_names_cannot_validate`、既有 `test_consumption_ledger_records_applied_fields_and_coverage`、`test_declared_template_operators_are_not_uncovered`、`test_declared_partial_cost_coverage_limits_the_scope`、`test_validation_report_marks_the_result_validated`、`test_validation_report_with_an_old_schema_keeps_prediction` |
| 3 | round22 | 中 | 成立：用评审探针 `/tmp/ceva_review_round22_long_api_probe.py` 复现，按后端公开契约传 `outer=1、inner=5000` 时短轨迹和长轨迹都报 `ValueError: ... hi=1, lo=5000`，而直接 `set_tile_count(5000)` 正常得到 5002 周期；根因是 `PIMTracePIMCore` 把 `outer/inner` 当成低层 hi/lo 转发 | 已修复 | `PIMTracePIMCore.set_tile_count_long()` 校验两个因子为正整数并保存 `outer×inner` 为总次数；`execute()` 统一调用 `PIMTrace.set_tile_count(..., allow_long_upgrade=...)`，短轨迹升级、长轨迹 24 位 hi/lo 编码和越界检查全部复用直接计数入口，不再新增第二套绑定公式。低层 `PIMTrace.set_tile_count_long(hi, lo)` 保持独立；通用后端核心未改动 | `src/pim/pim_core.py`（`PIMTracePIMCore.set_tile_count_long`、`PIMTracePIMCore.execute`）（S）；`tests/sim/test_pim_backend.py`（S） | `test_tile_count_long_uses_the_total_of_outer_and_inner`、`test_tile_count_long_rejects_non_positive_factors`、`test_tile_count_long_rejects_counts_past_24_bits`、`test_profile_reads_the_same_long_tile_count`、既有 `test_tile_count_binds_runtime_loop`、`test_long_tile_count_halves_must_fit_12_bits` |

第 23 轮（round23，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round23 | 高 | 成立：复跑 `/tmp/ceva_review_round23_cost_snapshot_probe.py`，加载时为 1024，随后把成本文件改为 999999 并同步报告新哈希；修复前运行时仍持旧值 1024，成熟度仍为 `validated_in_scope`，缓存台账哈希 `2722ee…` 与当前输入哈希 `f44024…` 已不同 | 成本 sidecar、模型 IR、放置 sidecar、请求 trace 和 PIM MLIR 没有共同记录“消费时”的内容身份：`_load_cost_sidecar` 用加载时的值，`_ceva_cost_sidecar_error` 和 `build_ceva_run_manifest` 却在运行结束时重新读磁盘，旧执行结果可被静默绑定到新文件 | 已修复 | 新增输入快照（路径 + 消费时 SHA256），在 `load_model_ir`、`load_trace`、放置 sidecar 首次读取、成本 sidecar 实际解析、PIM MLIR 实际读入时登记；成本台账哈希改为实际解析字节的哈希；成熟度、消费台账和运行清单只引用快照；`_ceva_run_manifest()` 生成前逐项复核当前文件，任何变化或删除都抛 `RuntimeError`；CEVA 放置 sidecar 首次读取后缓存，后续放置/能力/清单共用同一内容。稳定输入的正例仍返回 `validated_in_scope` 并生成清单 | 同一根因的五个输入入口全部覆盖：模型 IR、放置 sidecar、请求 trace、成本 sidecar、PIM MLIR；`_ceva_validation_scope_error` 的 IR/放置/trace 哈希也改读快照，不只修成本探针；清单改为消费时身份后，运行结束重读的旁路不再存在 | `src/run_manifest.py`（`verify_input_snapshot`、`_verify_snapshot_entry`、`_require_snapshot_inputs`、`build_ceva_run_manifest`）；`src/scheduler/gene_sim_scheduler.py`（`load_model_ir`、`load_trace`、`_record_input_snapshot`、`_record_pimir_snapshot`、`_ensure_input_snapshot`、`_compiler_sidecar`、`_load_compiler_placement`、`_register_compiler_artifacts`、`_load_cost_sidecar`、`_cost_sidecar_consumption`、`_build_cost_sidecar_ledger`、`_ceva_cost_sidecar_error`、`_ceva_validation_scope_error`、`_ceva_run_manifest`、`run`）（S）；`tests/sim/test_compiler_placement.py`、`tests/sim/test_run_manifest.py`（S） | `test_cost_sidecar_rewrite_after_load_keeps_prediction`、`test_input_snapshot_covers_every_consumed_file`、`test_manifest_refuses_a_cost_sidecar_rewritten_after_consumption`、`test_manifest_refuses_a_pim_mlir_rewritten_after_consumption`、`test_manifest_uses_the_consumed_input_snapshot`、`test_manifest_rejects_each_rewritten_input`、`test_manifest_rejects_a_changed_pim_mlir_snapshot`、`test_manifest_rejects_an_incomplete_snapshot` |
| 2 | round23 | 高 | 成立：复跑 `/tmp/ceva_review_round23_boundary_probes.py`，op0 搬运字节设为 `Infinity` 时修复前台账仍计 5 个有效成本算子、运行时 `data_bytes=Infinity`、成熟度 `validated_in_scope`；有限值 4096 的对照正常且保持有限 | “有效成本”此前只实现为 `> 0` 判断，`_mram_bytes_of()` 仅做浮点转换，加载、台账和运行时共用这一不完整定义，没有有限性、非负和类型约束；顶层标量、prefill/decode 分档、measurements 回退都经过同一提取函数 | 已修复 | 新增 `_validate_ceva_cost_sidecar()`/`_validate_ceva_mram_value()`，在 `_load_cost_sidecar()` 与懒加载的 `_cost_sidecar_consumption()` 中、任何算子落值之前检查全部条目的 `mram_traffic_bytes`；`None` 表示未测得并保留回退语义，零值合法但不形成覆盖；±Infinity、NaN、负数、字符串、布尔、列表和嵌套非法类型使整份输入抛错。非 CEVA 路径保持既有行为。合法值下运行时字节、台账和成熟度一致 | 逐算子检查顶层标量、prefill/decode 分档和 measurements 回退三种形态，不限于探针的单个算子；校验发生在 `_load_cost_sidecar` 的统一解析边界，不在成熟度处事后过滤，执行不会消费异常值 | `src/scheduler/gene_sim_scheduler.py`（`_validate_ceva_cost_sidecar`、`_validate_ceva_mram_value`、`_load_cost_sidecar`、`_cost_sidecar_consumption`、`_build_cost_sidecar_ledger`）（S）；`tests/sim/test_cost_sidecar.py`（S） | `test_ceva_rejects_non_finite_or_negative_mram_values`（标量/分档/measurements 三种形态 × 正负无穷、NaN、负数、字符串、布尔、列表）、`test_ceva_accepts_zero_none_and_finite_mram_values`、既有 `test_ceva_cost_sidecar_annotates_the_real_operators`、`test_consumption_ledger_records_applied_fields_and_coverage` |
| 3 | round23 | 高 | 成立：复跑 `/tmp/ceva_review_round23_boundary_probes.py`，修复前 YAML `num_cpus=0` 与 `GENESIM_CPU_MODEL_PEAK_FLOPS=0`、`GENESIM_CPU_MODEL_PEAK_BANDWIDTH_GBS=nan`、`GENESIM_CPU_MODEL_BASE_LATENCY_US=-1` 下 `validate(mode="simulation")` 均为 True；直接构造 `CPUModel` 后剖面分别出现 `ZeroDivisionError`、NaN 存储时间和负时延；合法单核对照正常 | `_validate_ceva` 从 `raw_config["cpu_model"]` 读值，绕过了 `GENESIM_*` 覆盖后的生效配置，且完全没有校验 `system.num_cpus`；`CPUModel` 用生效值做除数和时延，配置校验与执行边界因此得出不同结论 | 已修复 | 新增共享 `ceva_host_parameter_errors()`：`cpu_model.base_latency_us`、`peak_flops`、`peak_bandwidth_gbs` 必须为有限正数，`system.num_cpus` 必须为正整数；`_validate_ceva` 改用 `get()` 读取生效配置；`CPUModel.__init__` 对 CEVA 配置在读取任何参数前复用同一契约并抛 `ValueError`；`_from_dict` 透传 `pim_type`；VPU 工厂的 CEVA 分支使用生效 `cpu_model` 与核数构造，通用路径保持原行为 | 覆盖 YAML、`override()`、`GENESIM_*` 环境覆盖、直接构造 `CPUModel(config)` 和 VPU 工厂 `_from_dict` 五个入口；失败发生在资源创建与执行之前；通用/UPMEM 配置不启用该契约 | `src/config_loader.py`（`ceva_host_parameter_errors`、`_validate_ceva`）；`src/backend/cpu_backend.py`（`_is_ceva_config`、`CPUModel.__init__`、`CPUModel._from_dict`）；`src/vpu/factory.py`（`_create_cpu_model`）（S）；`tests/sim/test_config_loader.py`、`tests/sim/test_cpu_backend.py`（S） | `test_simulation_rejects_non_positive_cpu_count`、`test_simulation_rejects_invalid_host_parameters_from_every_entry`、`test_invalid_effective_parameters_are_rejected_before_profile`、`test_environment_overrides_are_checked`、`test_valid_control_profiles_finite_latency`、`test_factory_uses_the_effective_ceva_host_parameters`、既有 `test_simulation_requires_explicit_cpu_model`、`test_simulation_rejects_non_positive_cpu_model` |

第 24 轮（round24，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round24 | 高 | 成立：复跑 `/tmp/ceva_review_round24_boundary_probes.py` 与 `/tmp/ceva_review_round24_trace_window.py`。修复前模型 IR 消费 `model_id=old` 但快照匹配未消费的新字节，最终复核仍 passed；请求 trace 消费 `old-request` 而快照匹配 `new-request`，复核同样 passed；稳定文件对照的快照与消费内容一致 | `load_model_ir()`/`load_trace()` 先由基类按路径解析，再由 `_record_input_snapshot()` 重新读盘计算哈希，解析与登记不是同一份字节；窗口内改写会被静默换绑。`_ceva_request_boundary()` 也在运行中重新读盘获取边界证据，属于同一根因的旁路 | 已修复 | 基类 `load_model_ir()`/`load_trace()` 改为一次 `read_bytes()`，新增 `ModelIR.from_bytes()`、`Trace.from_json_bytes()`、`Trace.from_binary_bytes()` 解析同一份字节；新增 `_register_loaded_input()` 钩子，CEVA 调度器用 `artifact_entry_from_bytes()` 登记该字节的路径与 SHA256。`_ceva_request_boundary()` 优先使用已加载的 `self.trace`，仅尚未调用 `load_trace` 的独立调用方回退配置路径。通用调度器钩子为空，非 CEVA 行为不变 | 模型 IR 与请求 trace 两个入口共用同一实现；放置 sidecar、成本 sidecar、PIM MLIR 已按同一次读取解析并哈希，保持原状；运行中重读 trace 的边界证据路径一并改读消费对象；稳定输入正例仍正常复核 | `src/scheduler/base_scheduler.py`、`src/ir/model_ir.py`、`src/ir/trace_ir.py`、`src/scheduler/gene_sim_scheduler.py`、`src/run_manifest.py`（S）；`tests/sim/test_compiler_placement.py`（S） | `test_model_ir_snapshot_binds_the_consumed_bytes`、`test_request_trace_snapshot_binds_the_consumed_bytes`、`test_request_boundary_uses_the_consumed_trace`；探针复跑后快照匹配旧消费字节、最终复核抛“内容已变化”，稳定对照复核通过 |
| 2 | round24 | 高 | 成立：复跑 `/tmp/ceva_review_round24_cpu_cost_controls.py`，`system.num_cpus=1` 且 `cpus=[{},{}]` 时 `simulation_valid=true`、工厂创建 2 个 CPU、每个模型仍按 1 核计费；`/tmp/ceva_review_round24_boundary_probes.py` 中 `num_cpus=0/-1` 虽被正式校验和 CPUModel 构造拒绝，完整 `create_vpus_and_topology()` 仍创建 64 个 PIM 核和 0 个 CPU | 第 23 轮的数值契约只落在 `_validate_ceva()` 与 CPUModel 构造器；工厂按数量循环创建，零/负数量时构造器根本不执行；`_validate_ceva()` 未处理可改变 CPU 数量的 `cpus` 列表；`_ceva_resource_error()` 只清点 GPU/PIM。配置校验、模型构造和实际主机资源仍是三套判断 | 已修复 | 新增 `ceva_flat_resource_errors()` 统一拒绝 `nodes/gpus/cpus` 列表并由 `_validate_ceva()` 复用；新增 `_ceva_resource_contract_errors()`，`create_vpus_and_topology()` 在任何资源循环前调用共享主机参数契约，零/负核数与替代列表直接抛 `ValueError`；`_ceva_resource_error()` 增加实际 CPU 数量与 `system.num_cpus` 的一致性核对 | YAML/override/环境覆盖配置校验、CPUModel 直接构造、`_create_cpu_model`、完整工厂、节点路径和运行后资源清点全部覆盖；通用/UPMEM 配置不启用该契约，行为不变 | `src/config_loader.py`（`ceva_flat_resource_errors`、`_validate_ceva`）、`src/vpu/factory.py`（`_ceva_resource_contract_errors`、`create_vpus_and_topology`）、`src/main.py`（`_ceva_resource_error`）（S）；`tests/sim/test_config_loader.py`、`test_cpu_backend.py`、`test_main.py`（S） | `test_ceva_rejects_alternate_cpu_resource_lists`、`test_factory_rejects_invalid_ceva_cpu_resources`（零/负核数、`cpus` 列表、`gpus` 列表）、`test_factory_creates_exactly_the_configured_host_cpu`、`test_cpu_resource_count_is_checked`；探针复跑后 `cpu_list_two` 为 `simulation_valid=false` 且工厂抛 `ValueError`，合法单核仍创建 1 CPU/1 PIM |
| 3 | round24 | 中 | 成立：复跑 `/tmp/ceva_review_round24_numeric_probes.py`，`10**1000` 使配置 MAC、配置区间倍率、直接 `zone_scale()`、验证报告 `_finite_number()` 与 VPU `_positive_float()` 全部抛未包装的 `OverflowError`；成本 sidecar 在 `_validate_ceva_mram_value()` 同样抛出。合法 1.5 倍仍编码为 3，真实 sidecar 6850 条记录、27400 个数值全部通过校验 | 各校验入口各自直接调用 `float()`，没有统一的“类型检查—转换—有限性”边界；JSON/YAML 可解析的任意精度整数在转换时溢出，绕过既定的返回 False、`ValueError`、`RuntimeError` 或默认值语义；区间倍率还缺少 2× 取整前的有限性判断 | 已修复 | 新增 `numeric_utils.finite_float()`，显式接住 `TypeError`/`ValueError`/`OverflowError` 并返回 None；成本三种形态、配置 MAC/时钟/倍率/中转开销/autotune、验证报告、VPU 辅助、`pim_isa` 区间编码、容量与 legacy 迁移、`pim_compiler` epsilon、性能目标/数据集观察/放置权重/预测器特征与成功样本判断全部复用或接住 `OverflowError`；区间倍率先判 2× 有限再取整。配置校验返回 False，直接编译接口抛 `ValueError`，成本 sidecar 抛带算子与字段路径的 `RuntimeError`，不把超大值当默认成本 | 按 grep 全仓收敛同类入口：成本标量/分档/measurements、配置校验全部 `math.isfinite(float(...))` 点、区间编码与编译/求值入口、验证报告、VPU 峰值、容量解析、预测器/autotune 旁路；真实 sidecar 与合法 1.5 倍对照保留 | `src/numeric_utils.py`（新增）、`src/scheduler/gene_sim_scheduler.py`、`src/config_loader.py`、`src/pim/pim_isa.py`、`src/vpu/vpu.py`、`src/pim/capacity.py`、`src/pim/pim_compiler.py`、`src/performance_objective.py`、`src/autotune/dataset_observer.py`、`src/autotune/placement.py`、`src/predictor/dataset.py`、`src/predictor/benchmark.py`（S）；`tests/sim/test_cost_sidecar.py`、`test_config_loader.py`、`test_zone_cycles.py`、`test_vpu.py`、`test_compiler_placement.py`（S） | `test_validation_report_number_rejects_conversion_overflow`、`test_validation_rejects_overflowing_external_numbers`、`test_validation_rejects_overflowing_capacity_values`、`test_source_applicability_survives_overflowing_values`、`test_positive_float_rejects_conversion_overflow`、`test_ceva_rejects_non_finite_or_negative_mram_values`（新增 `10**1000`）、`test_encoding_rejects_invalid_external_factors`（新增大整数与 `1e308`）；探针复跑后 `config_mac`/`config_zone` 为 false、`direct_zone` 抛 `ValueError`、报告数值为 null、VPU 回落默认值、成本加载抛字段化 `RuntimeError`，`legal_control=3` |

第 25 轮（round25，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round25 | 高 | 成立：复跑 `/tmp/round25_repro_issue1.py`（基于测试用 CEVA 画像和真实放置 sidecar）。修复前 `latency_base_ns=-1`、NaN 带宽、零带宽均使 `validate(mode="simulation")=True`，HOST→PIM 1 B 直连分别得到 `-8.75e-10 s`、`NaN`、`inf`；零带宽全双工 `concurrent_latency(1,1)` 抛 `ZeroDivisionError`；`bandwidth_gbps=10**1000` 在 `CommLib` 构造时抛未包装的 `OverflowError`；合法对照 HOST→PIM 直连与全双工均为 `5.00125e-7 s` | `_validate_ceva()` 只检查链路方向存在；`CommLink.from_dict()` 与工厂 `_make_comm_link()` 各自直接 `float()`；`concurrent_latency()` 未复用 `latency()` 的零带宽保护；报告 `_normalized_links()` 只查有限性不查正值。同一根因影响 HOST↔PIM、CPU 中转、跨核 direct/host_relay 以及全/半双工集合通信 | 已修复 | 新增 `link_field_errors()`（带宽有限正数、基础延迟有限非负）与 `_link_number()`（类型与溢出按链路和字段报错）；`CommLink.from_dict()` 成为唯一转换入口；`concurrent_latency()` 零带宽返回无穷；`CommLib` 对 CEVA 配置在 `add_links()` 构造边界拒绝非法值；新增 `ceva_comm_link_errors()` 供配置校验和完整工厂共用；`_normalized_links()` 复用同一契约。通用零带宽单路返回无穷与合法 UPMEM 路径保持 | `grep bandwidth_gbps` 全仓核对：删除工厂的第二份 `float()` 转换并复用 `from_dict`；`autotune_dataset.py`/`predictor/dataset.py` 的链路特征不参与 CEVA 正式运行，保持既有默认值语义；配置校验、通信库构造、完整工厂和验证报告四个入口全部走同一函数 | `src/backend/comm_lib.py`、`src/config_loader.py`、`src/vpu/factory.py`、`src/scheduler/gene_sim_scheduler.py`（S）；`tests/sim/test_comm_lib.py`、`test_config_loader.py`、`test_cpu_backend.py`、`test_compiler_placement.py`（S） | `test_full_duplex_zero_bandwidth_returns_inf`、`test_from_dict_rejects_invalid_fields_with_link_and_field`、`test_ceva_commlib_rejects_invalid_link_numbers`、`test_legal_ceva_links_still_build`、`test_simulation_rejects_invalid_link_numbers`（5 条必需链路 × 2 字段 × 反例）、`test_simulation_rejects_invalid_half_duplex_link`、`test_factory_rejects_invalid_ceva_comm_links`、`test_validation_report_links_require_valid_numbers`；探针复跑后全部非法场景 `validate=False`，合法对照 `5.00125e-7 s` |
| 2 | round25 | 中 | 成立：直接调用 CEVA 核心与相位解析入口。修复前 `PIMTrace([mac], {}).estimate_total_cycles({"MAC": NaN}, strict=True)` 返回 `nan`、`MAC=-1` 返回 `-1.0`、`MAC=10**1000` 抛 `OverflowError`；`phase_cycle_breakdown(..., strict=True)` 同样 NaN 透传、负值少计（`21.8` 对合法 `45.8`）、大整数抛 `OverflowError`；合法 `MAC=2` 得 `2.0`、相位合法 `REDUCE=2` 得 `45.8` | `resolve_cycle_costs()` 直接 `float()` 且没有 strict 参数；`estimate_total_cycles()`/`estimate_cycle_breakdown()` 的 strict 只检查键覆盖；`phase_cycle_breakdown()` 调用共享解析器时未传 strict，缺项检查放在计算之后；配置校验只遍历必需键，运行期直接读取的 `REDUCE_PARTIAL` 不受约束；`_estimate_reduction_time()` 直接把配置值乘进报告时延 | 已修复 | `resolve_cycle_costs(pu_config, *, strict=False)` 新增严格数值契约：非有限、负值、转换溢出按原语字段名报错，零值只放行 `CEVA_ZERO_ALLOWED_CYCLE_COSTS` 中的 setup/占位项，未知键保持原报错；两个轨迹求值入口和相位解析透传 `strict`；CEVA 配置校验遍历成本表全部条目；归约时长入口校验 `REDUCE_PARTIAL`。非严格通用路径与合法有限值行为不变 | `grep pim_trace_cycle_costs`/`estimate_*` 全仓核对：`pim_core._trace_latency`、`vpu._static_phase_metrics`、`vpu.execute_profile` 均经 `estimate_cycle_breakdown`/`phase_cycle_breakdown` 透传 strict；`_ceva_cost_model_summary` 读取的成本表现由配置全表校验保护；`vpu._positive_float`、`scheduler.positive_float` 与预测器特征转换是显式默认值语义，不属于轨迹/相位求值入口，保持原行为 | `src/pim/pim_trace_loader.py`、`src/pim/pimir_trace.py`、`src/config_loader.py`、`src/scheduler/gene_sim_scheduler.py`（S）；`tests/sim/test_pim_trace_loader.py`、`test_pimir_trace.py`、`test_pim_backend.py`、`test_config_loader.py`、`test_scheduler.py`、`test_vpu.py`（S） | `test_strict_total_cycles_rejects_invalid_mac_costs`（8 类反例）、`test_strict_breakdown_rejects_invalid_per_byte_costs`、`test_strict_rejects_zero_outside_setup_whitelist`、`test_strict_requires_explicit_cycle_table`、`test_non_strict_keeps_legacy_finite_values`、`test_strict_phase_rejects_invalid_costs`、`test_strict_phase_accepts_finite_positive_costs`、`test_strict_phase_zero_follows_setup_whitelist`、`test_execute_rejects_invalid_cycle_costs`、`test_static_phase_profile_rejects_invalid_cycle_costs`、`test_simulation_rejects_invalid_extra_cycle_costs`、`test_reduction_time_rejects_invalid_cycle_cost`、`test_reduction_time_uses_finite_cost`；探针复跑后所有非法值抛带字段名的 `ValueError`，合法对照数值不变 |

第 26 轮（round26，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round26 | 高 | 成立：复跑 `/tmp/ceva_review_round26_issues.py`。修复前 `input_tokens=5、seq_len=128、output_length=2` 的 prefill 边界为 `input_lengths=[5]、seq_lengths=[128]`，展开却是 prefill `Tq=128/Tp=0`、decode 初始 `Tp=128`，有效长度 5 没有进入任何执行形状；普通 5/5 请求正确展开为 5。修复后同一 padding 请求未显式登记时 `_ceva_request_boundary()` 返回 None、`_get_operator_with_runtime()` 抛带字段名的 `ValueError`；显式 `padded_query_length=128、initial_kv_length=0` 时边界分别登记 `[5]/[128]/[0]`，执行形状按 128 展开。decode-only `input_tokens=5、initial_kv_length=128` 展开为 `Tp=128/129`；普通 5/5 请求仍为 `Tq=5`，真实 10 请求 trace（token 数与 seq_len 全等）行为不变 | `Request` 只有 `seq_len`，第 4、6 轮把“读取真实 trace”误当成“trace 已包含全部边界”；`_ceva_runtime_shape_contexts()` 只读 `seq_len` 并同时当 prefill 有效 prompt 长度和 decode 初始 KV，`_ceva_request_boundary()` 的 `seq_lengths` 也是合并口径。相同根因继续影响 `_ceva_run_shapes_of()`、`_ceva_run_matmul_stage_m()`、`_ceva_sample_shape_error()`、`_get_operator_with_runtime()` 的 Tq/Tp 绑定、`_execute_pipeline()` 的 `step_context()`、自适应放置成本聚合、`pim.capacity._request_runtime_contexts()` 与请求指标；既有测试全部令 token 数等于 `seq_len`，等值探针掩盖了 padding 与历史 KV 场景 | 已修复 | 新增 `RequestShapeBinding` 与 `Request.shape_binding()`，请求新增 `padded_query_length`/`initial_kv_length`，JSON 与二进制 trace 升到 version 3 并保留 v1/v2 读取；`__post_init__` 在未给 `seq_len` 时按显式绑定推断。请求边界契约改为 `input_lengths`（有效 prompt）、`query_lengths`（padding 后执行 Tq，decode-only 为 1）、`initial_kv_lengths`（初始 KV）；实际运行、验证范围展开、独立样本核对、容量峰值上下文、自适应成本聚合与请求指标统一读同一绑定。无法唯一确定三者的 CEVA 请求在 `_execute_pipeline()`/`_get_operator_with_runtime()` 显式拒绝，边界返回 None 使成熟度保持 `model_prediction`；验证报告契约升到 `schema_version=11`。通用旧 trace 仍走原 `seq_len` 约定 | `grep seq_len` 全仓核对：`_get_operator_with_runtime`、`_ceva_runtime_shape_contexts`、`_ceva_runtime_query_bounds`、`_ceva_request_boundary`、`step_context`、`_build_adaptive_initial_placement`、`capacity._request_runtime_contexts`、请求指标 8 类入口全部改为同一绑定；`_execute_sequential` 缓存键加入两个新字段；predictor/autotune 不属于 CEVA 正式路径且保持既有语义；`_raises` 只在 CEVA 配置启用，通用歧义 trace 不受影响 | `src/ir/trace_ir.py`、`src/scheduler/gene_sim_scheduler.py`、`src/pim/capacity.py`（S）；`tests/sim/test_trace_ir.py`、`test_compiler_placement.py`（S） | `test_shape_binding_infers_plain_requests`、`test_shape_binding_requires_explicit_padding`、`test_shape_binding_rejects_inconsistent_values`、`test_padded_binding_roundtrips_json_and_binary`、`test_request_boundary_rejects_ambiguous_padding`、`test_request_boundary_records_effective_and_execution_lengths`、`test_decode_only_uses_the_declared_history_kv`、`test_request_boundary_handles_mixed_padded_and_history_requests`、`test_padded_prefill_sample_must_use_the_execution_shape`、`test_padded_prefill_gemm_sample_uses_the_execution_m`、`test_padded_prefill_runtime_uses_the_declared_query_length`、`test_decode_only_runtime_uses_the_declared_initial_kv`、`test_inconsistent_request_binding_is_rejected`、`test_padded_request_trace_can_still_validate_in_scope`、`test_version_2_binary_trace_still_loads`、`test_ambiguous_legacy_trace_keeps_seq_len_runtime_convention`、既有 `test_runtime_shapes_expand_from_the_request_trace`、`test_tp2_shape_ranges_come_from_the_execution_graph`；探针复跑结果见 7.3 |
| 2 | round26 | 高 | 成立：复跑 `/tmp/ceva_review_round26_issues.py`。修复前 `engines_per_core=True` 在 64 核 4e 场景下 structure/simulation 均为 true，工厂创建 64 核/64 engine；`clusters_per_device=True` 通过并只创建 1 核；三个 `1.0` 字段通过两种校验后分别在工厂 `range()` 抛未包装 `TypeError`；合法整数对照为 64 核/256 engine。修复后四个字段的布尔、浮点、字符串反例在 structure/simulation 均为 false，工厂在数量循环前抛带字段名的 `ValueError`，运行后清点与运行清单同样拒绝；合法 1/64/128 × 1e/4e 组合仍为 true，64 核 4e 仍为 64 核/256 engine | 资源数量与 engine 数没有与 CPU 数量相同的“非布尔正整数”契约：`_validate_ceva()` 用 `!=`/`in` 做数值相等判断，工厂共享契约只覆盖 CPU/链路，随后直接进入 `range()`，运行后清点与运行清单再用 `int()` 把布尔/浮点合法化；YAML、`override()`、`GENESIM_*` 覆盖和直调工厂因此可能得到不同结论，目标名 `_4e` 可与 engine 1 形成自相矛盾画像 | 已修复 | 新增 `ceva_resource_count_errors()`，对 `system.num_gpus`、`system.num_pim_devices`、`pim.clusters_per_device`、`pim.tensors_per_cluster`、`ceva.engines_per_core` 统一要求非布尔的整数并检查允许集合，同时核对目标名 `_<n>e` 后缀；`_validate_ceva()`、`create_vpus_and_topology()`、`_ceva_resource_error()`/`_ceva_core_inventory()`、`build_ceva_run_manifest()` 全部复用；删除三处 `int()` 合法化。`GeneSimConfig.get()` 在无 default 时按配置原值推断 `GENESIM_*` 覆盖类型，整数覆盖 `"128"` 仍可用，`"true"`/`"4.0"` 字符串被拒绝。通用 `system.num_pim_devices` 校验改为类型安全，不再对字符串抛 TypeError | 按 YAML、`override()`、环境覆盖、完整工厂、运行后清点、运行清单、PU 池、报告核对与 DPU 映射九类入口逐一覆盖；`grep clusters_per_device/tensors_per_cluster/engines_per_core/num_pim_devices` 核对运行清单、PU 池与调度器剩余 `int()` 点，运行清单与 PU 池改为共享契约，调度器两处删除 `int()` 并用整数类型契约守住报告核对与 DPU 映射；通用/UPMEM 合法配置行为不变 | `src/config_loader.py`（`ceva_resource_count_errors`、`_validate_ceva`、`GeneSimConfig.get`、`_raw_value`）、`src/vpu/factory.py`、`src/main.py`、`src/run_manifest.py`（S）；`tests/sim/test_config_loader.py`、`test_cpu_backend.py`、`test_main.py`、`test_run_manifest.py`（S） | `test_resource_counts_require_non_boolean_integers`（4 字段 × 布尔/浮点/字符串）、`test_resource_count_contract_covers_gpu_count`、`test_resource_count_contract_covers_override_and_environment`、`test_legal_resource_count_combinations_pass_both_modes`（1/64/128 × 1e/4e）、`test_target_name_engine_suffix_must_match`、`test_factory_rejects_non_integer_resource_counts`、`test_inventory_rejects_boolean_and_float_resource_counts`、`test_manifest_rejects_non_integer_resource_counts`、`test_manifest_reads_integer_environment_overrides`、`test_boolean_core_count_cannot_upgrade_maturity`、`test_create_pu_pool_rejects_non_integer_ceva_counts`；探针复跑后四字段反例均 false，合法对照 64/256 |
| 3 | round26 | 高 | 成立：复跑 `/tmp/ceva_review_round26_issues.py`。修复前 `is_full_duplex="false"` 使正式校验为 true，运行时按非空字符串真值按全双工计费，1 B 双向 `5.00125e-7 s`；布尔 `False` 为半双工 `1.00025e-6 s`，少计一半。`inter_cluster_only="false"` 仍选中跨簇链路，布尔 `False` 正确返回无链路。修复后三个标记的字符串/整数/空值在正式校验为 false、`CommLib` 构造抛带字段名的 `ValueError`；布尔全/半双工成本分别为 `5.00125e-7`/`1.00025e-6 s`，`inter_cluster_only` 真假正确控制路由 | 第 25 轮把链路契约收窄为数值域，`link_field_errors()` 只校验带宽/延迟；`CommLink.from_dict()` 原样保存三个语义标记，`config_loader`、`CommLib.add_links`、工厂与验证报告各自依赖另一套真值语义。相同根因影响 direct/host_relay 路由、同核/跨核筛选、全/半双工集合通信与报告执行一致性；新增测试只覆盖数值与合法布尔，属于上一轮未覆盖完整字段契约的复发 | 已修复 | `link_field_errors()` 增加 `src_domain`/`dst_domain` 非空字符串与三个语义标记的实际布尔校验；CEVA 配置校验、`CommLib.add_links`、完整工厂和 `_normalized_links()` 共用同一契约；缺省值仍按运行时规则（全双工 true、簇内/跨簇 false），合法布尔行为与通用路径保持不变 | 三个标记 × 字符串/整数/空值，以及端点缺失/空串/非字符串全覆盖；路由侧验证 `inter_cluster_only` 真假与全/半双工并发成本，断言选路和计费与报告一致；通用路径仍允许合法布尔链路，零带宽单路既有语义未改 | `src/backend/comm_lib.py`、`src/config_loader.py`、`src/vpu/factory.py`、`src/scheduler/gene_sim_scheduler.py`（S）；`tests/sim/test_comm_lib.py`、`test_config_loader.py`（S） | `test_ceva_commlib_rejects_non_boolean_route_flags`（3 标记 × 5 类反例）、`test_ceva_commlib_rejects_ambiguous_link_endpoints`、`test_boolean_duplex_flag_keeps_half_and_full_costs_distinct`、`test_boolean_inter_cluster_flag_selects_the_route`、`test_simulation_rejects_non_boolean_link_flags`（5 类反例）、既有 `test_validation_report_link_semantics_must_match_the_run`、`test_legal_ceva_links_still_build`；探针复跑后非法值全部 false 且构造抛错，合法布尔对照数值不变 |

第 27 轮（round27，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round27 | 高 | 成立：复跑 `/tmp/ceva_review_round27_probes.py`。修复前实际消费 i8 产物、报告声明 fp16 时成熟度为 `model_prediction`；仅把磁盘产物改写为未参与 trace 的 f16 后，同一快照、同一报告变为 `validated_in_scope`，`_ceva_validation_scope_error()` 返回空，`_ceva_run_compute_dtypes()` 重读出 f16；恢复 i8 后又回到预测，且消费快照哈希未变。真实 CEVA 冒烟还暴露 TP 展开出的分片 op_id 没有独立台账条目，修复初版在 op6852 误拒绝 | 第 23～24 轮把“记录哈希并在结束时复核”当成所有消费者都只读快照，但没有保存首次解析的不可变产物文本；`_register_compiler_artifacts()` 丢弃读取后的 `texts`，`_ceva_artifact_texts_by_op_type()` 重新打开放置 sidecar 和每份 PIM MLIR，成熟度、数据类型、相位、验证样本与 trace 生成分别消费不同内容。TP 展开生成的新 op_id 只继承路径/哈希，没有按路径关联台账 | 已修复 | 新增 `_ceva_pimir_texts()`，首次按字节读取 PIM MLIR 时保存文本、SHA256 和 sidecar 路径，按 op_id 与路径双键建台账；`_ceva_artifact_texts_by_op_type()` 只读缓存的放置 sidecar 与台账，数据类型/相位/验证样本随之同源；`_validate_ceva_fusion_exceptions()` 与 `_operator_pimir()` 改读台账，CEVA 在台账建立后拒绝重新打开磁盘；`_ceva_validation_scope_error()` 在返回 `None` 前调用 `verify_input_snapshot()`，输入在运行中变化时保持 `model_prediction`；TP 分片按继承路径复用主算子台账 | `grep read_bytes/read_text` 核对调度器剩余读盘点：放置 sidecar 仍按首次读取缓存，成本 sidecar 有独立台账，模型 IR/请求 trace 已有单次读取快照；PIM MLIR 的全部语义消费者（dtype、phase、validation evidence、fusion、trace）统一到同一台账，直接成熟度、流水线成熟度赋值与运行清单前的复核三类入口都复用 | `src/scheduler/gene_sim_scheduler.py`（`_ceva_pimir_texts`、`_register_compiler_artifacts`、`_ceva_artifact_texts_by_op_type`、`_validate_ceva_fusion_exceptions`、`_operator_pimir`、`_ceva_validation_scope_error`）；`tests/sim/test_compiler_placement.py`（S） | `test_maturity_uses_the_consumed_artifact_after_an_unconsumed_rewrite`、`test_maturity_refuses_an_artifact_rewritten_after_consumption`、`test_trace_text_comes_from_the_consumed_artifact_ledger`、`test_expanded_shards_reuse_the_ledger_by_artifact_path`、`test_pipeline_maturity_uses_the_trace_ledger_text`、既有 `test_manifest_refuses_a_pim_mlir_rewritten_after_consumption`；真实 CEVA 冒烟见 7.3 |
| 2 | round27 | 高 | 成立：复跑 `/tmp/ceva_review_round27_probes.py`。修复前分别追加 `GPU→HOST`、`HOST→GPU`、`PIM→GPU` 链路后，`validate(mode="structure")` 与 `validate(mode="simulation")` 均为 True，`CommLib` 实际注册这些链路，工厂 GPU 资源仍为 0；合法基础配置五份场景均通过。修复后同一探针在 `CommLib(config=config)` 构造时即抛“端点不属于纯 PIM 资源域” | 第 25～26 轮把链路契约收敛到数值和布尔标记时，把“端点非空”误当成“端点属于本目标资源集合”；资源计数、权重映射和通信链路分别校验，缺少统一的纯 PIM 资源域封闭约束。相同根因影响 direct/host_relay、GPU 双向出入口、未知域以及验证报告 `_normalized_links()` | 已修复 | `comm_lib` 新增 `CEVA_LINK_DOMAINS=(HOST,PIM,CPU)` 与 `link_endpoint_errors()`；`link_field_errors()` 纳入端点域，`CommLib.add_links()` 与直接 `add_link()` 在 CEVA 构造边界统一拒绝；`config_loader` 新增 `ceva_comm_link_endpoint_errors()`，结构校验对已出现的链路先拒绝，正式校验再走完整 `ceva_comm_link_errors()`；`_normalized_links()` 复用 `link_field_errors()`，报告带 GPU 端点时无法参与一致性比较 | 覆盖配置结构/正式校验、`CommLib` 批量与单条注册、完整工厂、报告链路比较四类入口；合法 CPU/HOST/PIM 链路、正字节有限成本与 GPU 资源为 0 的联合正例保留；通用 GPU/UPMEM 配置不调用 CEVA 端点契约，行为不变 | `src/backend/comm_lib.py`、`src/config_loader.py`、`src/scheduler/gene_sim_scheduler.py`（`_normalized_links` 经共享契约）；`tests/sim/test_comm_lib.py`、`test_config_loader.py`、`test_cpu_backend.py`、`test_compiler_placement.py`（S） | `test_ceva_commlib_rejects_gpu_and_unknown_link_endpoints`、`test_ceva_add_link_rejects_gpu_endpoints`、`test_rejects_gpu_and_unknown_link_endpoints`（structure/simulation）、`test_legal_cpu_host_pim_links_build_without_gpu_resources`、`test_factory_rejects_invalid_ceva_comm_links`（新增 GPU/未知域）、`test_normalized_links_reject_non_pim_domains`、`test_gpu_links_cannot_form_a_validated_scope` |
| 3 | round27 | 中 | 成立：复跑 `/tmp/ceva_review_round27_probes.py`。修复前 `loop_count=2` 为 4 周期、`loop_count=0` 仅 2 周期、`loop_count=1.5` 为 3.5 周期、`True` 按 1 次计、`repetitions=1.5` 把两次循环放大为 6 周期；负数被 `execute_runtime()` 拒绝。修复后上述非法次数在计算前抛带 `loop_count`/`repetitions`/`tile count` 字段名的 `ValueError`，合法 `loop_count=2` 仍为 4 周期 | 第 22 轮只在新增的 `set_tile_count_long()` 上补了非布尔正整数检查，没有把同一契约下沉到直接次数、整轨迹重复次数及 Python 轨迹绑定层；`PIMBackend.execute_runtime()` 只比较大小、`PIMTracePIMCore.set_tile_count()` 直接赋值、`PIMTrace.set_tile_count()` 只处理负数与 12/24 位编码，长短两个公开入口各自解释“有效次数” | 已修复 | `pim_trace_loader` 新增 `require_loop_count()`；`PIMBackend.execute_runtime()` 与 `UnifiedPIMBackend.execute_runtime()` 在轨迹绑定前校验非布尔整数、重复次数为正、CEVA 循环次数为正；`PIMTrace.set_tile_count()` 增加 `require_positive`，超 24 位在 CEVA 路径始终显式失败；`PIMTrace.set_tile_count_long()` 的两半也拒绝布尔/浮点；`PIMTracePIMCore.set_tile_count()`/`execute()` 共用契约，`CEVANPMPIMCore` 要求正整数 | 直接/长循环接口、执行与 `execute_profile()` 共用的 Python 轨迹、短轨迹升级、24 位上限全部走同一实现；`4095/4096/5000` 次完整计费测试保留；通用/UPMEM 的 12 位兼容截断、负数告警回落与零值既有语义未改，只拒绝布尔/浮点等非法类型 | `src/pim/pim_trace_loader.py`、`src/pim/pim_core.py`、`src/backend/pim_backend.py`；`tests/sim/test_pim_backend.py`（S） | `test_execute_runtime_rejects_non_integer_counts`（通用路径）、`test_execute_runtime_rejects_invalid_loop_binding`（CEVA）、`test_direct_core_rejects_invalid_tile_counts`、`test_trace_binding_rejects_invalid_tile_counts`；既有 `test_tile_count_binds_runtime_loop`、`test_tile_count_long_uses_the_total_of_outer_and_inner`、`test_tile_count_long_rejects_non_positive_factors`、`test_tile_count_long_rejects_counts_past_24_bits`、`test_profile_reads_the_same_long_tile_count`、`test_non_ceva_keeps_the_legacy_12_bit_clamp` |

第 28 轮（round28，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round28 | 高 | 成立：复跑 `/tmp/ceva_review_round28_manifest_probe.py` 与本地探针 `/tmp/round28_probe1.py`。修复前加载时钟 0.63 的 YAML 后改写为 1.26，`build_ceva_run_manifest()` 仍成功：`resolved_config` 保持 0.63，`config_file.sha256` 却等于改写后的文件；无参数/带参数 `override()` 的 `source.sha256` 同样在清单生成时重读原文件。修复后同一改写分别让 `config_entry()` 与完整清单抛“配置文件……内容已变化”，稳定输入的直接文件/无参覆盖/带参覆盖三类入口均通过 | 第 21 轮把配置身份实现为“生成清单时的当前文件哈希”；第 23～27 轮建立消费快照时按模型 IR、请求 trace、放置 sidecar、成本 sidecar、PIM MLIR 五类输入枚举边界，遗漏第六类 YAML 配置及其覆盖链来源。`GeneSimConfig._load_yaml()` 读文件但不保留字节，`override()` 只保留路径，`config_entry()` 的文件分支与内存配置 `source` 分支都在运行结束时重读磁盘 | 已修复 | `_load_yaml()` 改为单次读取字节，同一份字节用于解析并登记 `loaded_config_sha256`；`__init__` 建立 `source_identity`（真实源文件路径 + 消费时哈希），`override()` 继承最初来源而不是临时 YAML 身份；`config_entry()` 只引用消费身份，返回前复核磁盘，改写或删除源文件显式失败；`_ensure_input_snapshot()` 增加第六类 `config_source`，`_require_snapshot_inputs()` 在配置存在持久源文件时要求该登记，完整清单再核对快照与配置来源一致 | 配置身份的四个消费者（清单 `config_file`、内存配置 `source`、调度器输入快照、成熟度/清单前复核）统一到同一消费身份；`resolved_config` 与 `env_overrides` 本就来自内存，保持原样；直接文件、无参数覆盖、带参数覆盖、稳定输入、快照改写五类反例/正例齐备；通用路径不生成运行清单，行为不变 | `src/config_loader.py`（`__init__`、`_load_yaml`、`override`）；`src/run_manifest.py`（`_consumed_config_identity`、`config_entry`、`_require_snapshot_inputs`、`build_ceva_run_manifest`）；`src/scheduler/gene_sim_scheduler.py`（`_ensure_input_snapshot`）；`tests/sim/test_run_manifest.py`、`tests/sim/test_compiler_placement.py`（S） | `test_manifest_rejects_a_config_file_rewritten_after_load`、`test_manifest_rejects_a_deleted_config_file`、`test_manifest_rejects_a_source_rewritten_after_plain_override`、`test_manifest_rejects_a_source_rewritten_after_parameterized_override`、`test_stable_config_identity_comes_from_consumed_bytes`、`test_manifest_rejects_a_config_source_rewritten_after_snapshot`、`test_input_snapshot_covers_every_consumed_file`、`test_manifest_refuses_a_config_source_rewritten_after_consumption`；探针复跑见 7.3 |
| 2 | round28 | 高 | 成立：复跑 `/tmp/ceva_review_round28_comm_probe.py` 与本地探针 `/tmp/round28_probe2.py`。修复前直接 `add_link()` 接受 `latency_base_ns=-1`（1 B 时延为负）与 `is_full_duplex="false"`（按字符串真值执行）；负数 `host_relay_overhead_ns` 虽让 `validate()` 返回 False，`CommLib(config=...)` 仍保存负开销。修复后直接对象的零/负/非有限带宽与延迟、三个非布尔标记、负/非有限中转开销在构造边界统一抛带字段名的 `ValueError`；合法直接链路与合法零/正开销照常计费 | 第 25～27 轮持续扩展字典形态的 `link_field_errors()`：第 25 轮补数值域、第 26 轮补三个语义标记、第 27 轮补端点，但直接 `add_link()` 只补了当轮端点检查，没有把既有数值域与布尔契约应用到 `CommLink` 对象；HOST 中转开销又独立留在 `CommLib.__init__` 用裸 `float()` 转换，未进入同一通信参数边界。现有 `test_ceva_add_link_rejects_gpu_endpoints` 只证明端点修复，其他非法字段均走字典批量入口，属于前几轮同类修复未覆盖完整公开入口的复发 | 已修复 | `link_field_errors()` 接受 `CommLink` 并规范化为同一字段映射；直接 `add_link()` 与批量 `add_links()` 复用完整契约（端点、带宽有限正数、延迟有限非负、三个布尔标记）；新增 `host_relay_overhead_errors()`，`CommLib.__init__`（CEVA）、完整工厂 `_ceva_resource_contract_errors()` 与通用配置校验共用有限非负检查与安全转换；通用 `CommLib` 构造保持原 `float()` 与合法零带宽语义 | 批量字典、直接对象、完整工厂、验证报告四条链路入口统一到 `link_field_errors()`；中转开销在通信库构造与完整工厂两处共用同一实现；通用/UPMEM 的直接构造、零带宽与旧真值语义保持不变并新增对照用例；合法五份场景与真实 sidecar 复验无回归 | `src/backend/comm_lib.py`（`link_field_errors`、`CommLib.__init__`、`add_link`）；`src/config_loader.py`（`host_relay_overhead_errors`、`validate`）；`src/vpu/factory.py`（`_ceva_resource_contract_errors`）；`tests/sim/test_comm_lib.py`、`tests/sim/test_cpu_backend.py`（S） | `test_ceva_add_link_rejects_invalid_fields`、`test_ceva_commlib_rejects_invalid_host_relay_overhead`、`test_ceva_host_relay_overhead_reaches_the_transfer_cost`、`test_non_ceva_direct_links_keep_legacy_behavior`、`test_factory_rejects_invalid_ceva_host_relay_overhead`；既有 `test_ceva_add_link_rejects_gpu_endpoints`、`test_ceva_commlib_rejects_non_boolean_route_flags`、`test_boolean_duplex_flag_keeps_half_and_full_costs_distinct` 保持通过；探针复跑见 7.3 |

第 29 轮（round29，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round29 | 严重 | 成立：复跑 `/tmp/ceva_review_round29_unphased_ops.py`。修复前真实产物 `bpath-0858498cb704d901.pimir.mlir` 的 `parsed_phase_count=5`、运行时模板 5 相，漏掉第 2 行无 `phase_spec` 的 `pim.eltwise sub`（fpsu，16,384 工作单元、33,024 B 输入、32,768 B 输出）；该减法有独立 SSA 结果 `%1` 并被下一相 EXP 消费，真实 sidecar 有 1,024 条 SOFTMAX 记录复用这份产物。既有 `test_non_phase_ops_do_not_enter_the_chain` 只锁定跳过行为 | `parse_phase_chain()` 把“文件出现过 `phase_spec`”当成“整份算子工作已被属性覆盖”，对所有无属性语句直接 `continue`；运行时模板只复制筛选后的 `program.steps`，编译、周期求值和执行剖面消费同一份不完整账本。按属性有无而不是按 SSA 数据流与算子语义决定计费，缺少统一的逐操作覆盖分类 | 已修复 | 新增 `_op_matches()`（按 SSA 顺序返回含有/无结果的全部算子语句）与 `_oplevel_step()`（单相工作量规则）；`parse_phase_chain()` 遍历整份 IR：带 `phase_spec` 的语句按声明 index 展开，不带属性且不在豁免名单的语句按单相规则补计一次遍历，`_PHASED_LEDGER_EXEMPT_OPS` 显式登记的零成本语句（目前仅 NoMemoryEffect 的 `convert_layout`）才跳过，未知语句直接报错；账本 index 按执行顺序分配，同一 op 的多相仍按声明 index 排序；`PHASE_RUNTIME_TEMPLATE_VERSION` 从 1 升到 2，旧的盘上模板与轨迹缓存自动失效重编 | 全仓 grep `phase_spec`/`parse_phase_chain`/`phase_runtime_template`：调度器 trace 编译、运行时模板、`phase_cycle_breakdown` 与 `vpu._static_phase_metrics` 都消费同一 `PhaseProgram`，无第二套跳过逻辑。扫描 `.opcompiler_cache` 的 25,173 份真实产物：133 份带 `phase_spec` 的文件里无属性语句只有 64 处 `pim.eltwise`（均为 softmax 减法），另有 572 个声明相位，未发现第二类被跳过的计算/搬运语句；`tt.dot` 与 `phase_spec` 同时出现的产物为 0，不存在两条前端混用的旁路。编译器侧 `genesim_bridge/ir_cost.py` 的 `_line_operator_flops()` 本来就按输出元素数为无属性 `pim.eltwise` 计费，不属于同一“整条跳过”根因；其相位字节口径是独立成本 sidecar 模型，本轮不扩大改动。融合吸收在 pimir 里是主算子的 `activation = #pim.act_spec<...>` 属性，由 `activation_bytes` 计费而非独立语句，新增对照测试断言融合前后账本步数不变、查表指令只多一次，避免把“补计无属性语句”误扩成重复计费 | `genesim/src/pim/pimir_trace.py`（`_op_matches`、`_oplevel_step`、`_parse_single_phase_ops`、`parse_phase_chain`、`PHASE_RUNTIME_TEMPLATE_VERSION`）；`genesim/tests/sim/test_pimir_trace.py` | `test_unphased_compute_op_is_billed_from_its_own_ssa_results`、`test_unphased_compute_op_adds_cycles_on_both_paths`、`test_real_softmax_artifact_bills_the_unphased_subtraction`、`test_fused_activation_is_billed_once_without_an_extra_step`、`test_exempt_layout_op_is_the_only_free_statement`、`test_unknown_unphased_statement_fails`、改写后的 `test_every_declared_phase_is_parsed`、`test_phases_keep_their_order_and_units`、`test_softmax_ledger_keeps_each_steps_own_workload`、`test_chain_compiles_to_one_instruction_group_per_phase` |
| 2 | round29 | 高 | 成立：复跑 `/tmp/ceva_review_round29_config_identity.py`。修复前加载一个不存在的路径得到 `origin=in_memory`，随后新建该 YAML（时钟 1.26）但不重新加载，`config_entry()` 变为 `origin=file` 并采纳新文件哈希；完整清单 `resolved_config.pim.clock_period_ns=0.63`（实际内存值）而 `config_file` 指向未消费的新文件。修复后同一探针（补 `GENESIM_PIM_CLUSTER_MEMORY_CAPACITY_GB=7.450580596923828` 以满足新的 GiB 一致性）的 `before` 与 `after` 均为 `in_memory`、哈希不等于新建文件，完整清单 `config_file.origin=in_memory`、`resolved_config` 时钟保持 0.63；始终缺失的对照稳定保持内存身份 | 第 28 轮保存了“已读文件的内容身份”，却没有同时保存“加载时是否存在持久来源”这一状态；`config_entry()` 仍用清单生成时的 `config_path.is_file()` 补推来源，`source_identity=None` 被错误解释为可以读取当前文件 | 已修复 | `GeneSimConfig.__init__` 在加载时固定 `source_kind`（成功读取文件为 `file`，否则为 `in_memory`）；`override()` 固定为 `in_memory` 并继承最初的 `source_identity`；`config_entry()` 只按固定种类分支：`file` 仍复核加载时快照（改写/删除显式失败），`in_memory` 永不读取磁盘、按 `resolved_config()` 内容哈希登记；没有 `source_kind` 的旧适配器按消费身份回退。快照要求与调度器 `config_source` 登记继续只认 `source_identity`，缺失来源的内存配置不会要求快照项 | grep `source_identity`/`source_kind`/`config_path.is_file`：运行清单 `config_entry()`、清单生成、调度器 `_ensure_input_snapshot()` 与 `_require_snapshot_inputs()` 都经同一消费身份，无第二套来源推断；`src/predictor/benchmark.py` 的 `config_path.is_file()` 是基准数据文件的独立存在性检查，不是配置来源。始终缺失、缺失后出现、出现后删除、已有文件稳定/改写/删除、无参与带参 `override()` 均有正反例 | `genesim/src/config_loader.py`（`__init__`、`override`）；`genesim/src/run_manifest.py`（`config_entry`）；`genesim/tests/sim/test_run_manifest.py` | `test_missing_config_stays_in_memory_when_the_file_appears_later`、`test_missing_config_stays_in_memory_after_appearing_and_deleting`、`test_override_of_a_missing_config_stays_in_memory`、既有 `test_manifest_rejects_a_config_file_rewritten_after_load`、`test_manifest_rejects_a_deleted_config_file`、`test_manifest_rejects_a_source_rewritten_after_plain_override`、`test_manifest_rejects_a_source_rewritten_after_parameterized_override`、`test_stable_config_identity_comes_from_consumed_bytes` |
| 3 | round29 | 高 | 成立：复跑 `/tmp/ceva_review_round29_mapping_contract.py`。修复前 `clusters_per_device=64.5`、`num_pim_devices=1.5`、`tensors_per_cluster=1.0`、布尔核数和 `num_gpus=1` 的 `structure_valid` 均为 False，但导出脚本退出码均为 0 并生成一对一计划，浮点核数生成的映射与合法 64 核完全相同；合法 64 核对照正常。修复后五类反例全部退出码 1、不生成计划，诊断信息指出具体字段；合法 1/64/128 核仍生成一对一计划，通用配置仍按历史方式生成共享映射 | 第 26 轮把资源数量契约下沉到配置、完整工厂、运行清点和运行清单，却遗漏了切分计划的生产入口；`_hardware_hierarchy()` 又用独立 `int()` 把浮点/布尔数量合法化，`build_mapping(one_to_one=True)` 只信任已经转换的层级 | 已修复 | `_hardware_hierarchy()` 在 CEVA 路径先调用 `ceva_resource_count_errors()`（非布尔整数、1/64/128 核、单设备、单 TensorPU、1e/4e、GPU 为 0），不再做 `int()` 转换；`build_mapping()` 的一对一公开分支再校验层级为非布尔正整数、`num_pim_devices=1`、`clusters_per_device∈{1,64,128}`、`tensors_per_cluster=1`，可直接调用的入口也无法绕过；通用路径保持原有 `int()` 读取与共享映射行为 | 全仓 `build_mapping`/`_hardware_hierarchy` 只有脚本主入口与测试调用，主入口之外的分支已由公开函数自身校验；`--partition-plan` 消费端不需要重复拒绝，非法计划在生产端即无法生成；容量解析继续复用配置契约。YAML 与环境覆盖分别有反例，1/64/128 核与通用共享映射有正例 | `genesim/scripts/export_fixed_pu_mapping.py`（`_hardware_hierarchy`、`build_mapping`）；`genesim/tests/sim/test_export_fixed_pu_mapping.py` | `test_invalid_yaml_hierarchy_fails_before_writing_a_plan`、`test_invalid_environment_override_fails_before_writing_a_plan`、`test_one_to_one_rejects_invalid_hierarchy_directly`、`test_legal_core_counts_still_produce_one_to_one_plans`、`test_general_config_keeps_the_old_hierarchy_reading`、既有 `test_each_dpu_owns_a_cluster`、`test_64_core_plan_uses_64_clusters`、`test_insufficient_cores_raises`、`test_stage_shard_too_large_for_one_core_raises`、`test_general_mode_still_shares_a_cluster` |
| 4 | round29 | 中 | 成立：复跑 `/tmp/ceva_review_round29_budget_manifest.py`。修复前 `local_memory_bytes=8000000000.5` 与 `sys_reserve_bytes=0.5` 的正式校验为 False，但 `build_ceva_run_manifest()` 仍成功，`hardware.local_memory_bytes` 被写成 8,000,000,000、`sys_reserve_bytes` 写成 0，布尔字段写成 1，而 `resolved_config` 保留原值；合法整数配置正常。修复后四类非法值在清单输出前抛字段化 `ValueError`，合法整数的 `hardware` 三项与 `resolved_config` 一致；调度器侧原有的 sidecar 预算一致性核对保持通过 | 第 26 轮只清理了运行清单中的资源数量 `int()`，容量字段仍保留另一套转换，未与配置校验和 `resolve_pim_capacity_limits()` 共用“非布尔整数、GiB 一致、预留小于物理量”的契约；有/无输入快照两条路径汇合后都经过该转换 | 已修复 | `build_ceva_run_manifest()` 统一调用 `resolve_pim_capacity_limits(config)`，要求 `source_key=ceva.local_memory_bytes`（缺失字节真源时显式失败，不再 `int(None)`），直接登记 `total_capacity_bytes`、`sys_reserve_bytes`、`usable_bytes`，清单层不再做任何整数转换；有/无输入快照共用同一段硬件预算代码 | grep `int(...local_memory_bytes/sys_reserve_bytes)`：调度器 sidecar 预算核对在 `int()` 前已做严格类型检查（布尔/浮点/字符串/负值/超量预留均拒绝），本轮补充调度器反例测试；配置校验与容量解析本就共用 `resolve_pim_capacity_limits()`；运行清单是最后一条容量 `int()` 转换路径，已删除。清单正例断言 `hardware` 与 `resolved_config` 在直接与快照两条路径一致 | `genesim/src/run_manifest.py`（`build_ceva_run_manifest`）；`genesim/tests/sim/test_run_manifest.py`、`genesim/tests/sim/test_compiler_placement.py` | `test_manifest_rejects_invalid_memory_budgets`（浮点/布尔/字符串/负值/超量预留/GiB 不一致 × 直接与快照两条路径）、`test_hardware_budget_matches_resolved_config_on_both_paths`、`test_ceva_refuses_non_integer_config_memory_budget`（调度器）、既有 `test_manifest_records_inputs_pim_mlir_and_toolchain` |

第 30 轮（round30，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round30 | 严重 | 成立：复跑 `/tmp/ceva_review_round30_softmax_helper.py`。修复前真实产物 `bpath-0858498cb704d901.pimir.mlir` 有 5 个 `phase_spec` 却被解析成 6 步，`pim.eltwise sub` 独立成 fpsu 相（16,384 工作单元、33,024 B 读、32,768 B 写），比去掉该步骤的对照多 23,856.8 cycles、65,792 B 搬运；评审冒烟里 1,024 条 SOFTMAX 全部 6 步。修复后同一产物为 5 步，EXP 相携带折叠的 fpsu 16,384 工作单元与 33,024 B 输入（x + max），不存在独立 fpsu 相；相对 5 相基线只多 16,386 compute cycles（16,384 工作 + 2 循环开销）与 256 B 搬运（max 读入），指令轨迹与周期公式两条路径一致 | 第 29 轮只凭“无 `phase_spec` 但有 SSA 结果且被下一相消费”推断独立硬件工作，没有核对 FlagTree 的相位归属契约：`ExpandPhases.cpp` 明确 `x - max` 是折进 EXP 的 FPSU 仿射（`not a phase of its own`），`LowerPIMToEmitC.cpp` 的 helper（无 `phases` 的 Eltwise/Lut）不拥有独立 engine 遍历、中间值保持 f32 且不经过 f16 缓冲。同一根因影响混合相位账本、运行时模板、指令编译、解析周期与执行剖面，`PHASE_RUNTIME_TEMPLATE_VERSION=2` 还把错误语义固化进缓存；第 29 轮新增的 `test_real_softmax_artifact_bills_the_unphased_subtraction` 直接锁定了这一错误假设 | 已修复 | `parse_phase_chain()` 升级为带相位归属的执行账本：按 SSA 生产者/消费者识别无相位 eltwise/lut，沿 helper 链找到第一个带相位消费者，把 helper 的算术（`FoldedPhaseWork`：单元 + 输入/结果张量）并入消费者首个相位，消费者操作数展开为 helper 的真实输入（x 与 max），不产生独立 LOADN/STOREN，也不产生 f16 中间往返；没有相位消费者的 eltwise/lut（独立 LUT）以及 matmul、静态量化仍按单相完整计费，未知语句继续显式失败。运行时模板保存折叠张量并按实际形状重算工作量，指令编译与周期求值消费同一字段；模板版本升到 3，旧 v2 模板与轨迹缓存失效重编。整份没有 `phase_spec` 的单相 IR 路径保持不变 | 扫描 `.opcompiler_cache` 全部 25,173 份产物：2,154 份 `tt.dot`、133 份带 `phase_spec`（572 个声明相位），其中 64 个无相位 `pim.eltwise` 全部被 `pim.lut` 消费者折叠，独立无相位语句为 0，133 份文件的账本步数均等于“声明相位 + 独立无相位语句”；12 份解析失败均为空测试内核 `module { func.func @k() { return } }`。`_parse_single_phase_ops()` 对整份无 `phase_spec` 的文件继续逐 op 计费；融合吸收仍由主算子 `activation_bytes` 只计一次；编译器侧 `genesim_bridge/ir_cost.py` 是独立成本 sidecar 模型，不属于同一根因。重编后 7,138 条轨迹模板版本均为 3，1,024 条 SOFTMAX 均为 5 步且各带 1 条折叠 fpsu，其余 op 无折叠项 | `genesim/src/pim/pimir_trace.py`（`FoldedPhaseWork`、`_phase_statement`、`_phase_consumers`、`_first_phased_consumer`、`_folded_helper_indices`、`_expanded_operand_spans`、`_fold_helpers_into`、`parse_phase_chain`、`phase_runtime_template`、`resolve_phase_template`、`compile_phase_chain`、`phase_cycle_breakdown`、`PHASE_RUNTIME_TEMPLATE_VERSION`）；`genesim/tests/sim/test_pimir_trace.py` | `test_folded_helper_is_billed_inside_its_consumer_phase`、`test_folded_helper_adds_compute_without_extra_transfer`、`test_real_softmax_artifact_folds_the_subtraction_into_exp`、`test_independent_unphased_lut_keeps_its_own_step`、`test_unphased_matmul_in_a_phased_file_is_not_folded`、`test_unphased_static_quantize_in_a_phased_file_is_not_folded`、改写后的 `test_every_declared_phase_is_parsed`、`test_phases_keep_their_order_and_units`、`test_softmax_ledger_keeps_each_steps_own_workload`、`test_chain_compiles_to_one_instruction_group_per_phase`；既有 `test_fused_activation_is_billed_once_without_an_extra_step`、`test_exempt_layout_op_is_the_only_free_statement`、`test_unknown_unphased_statement_fails` 保持通过；探针复跑见 7.3 |
| 2 | round30 | 高 | 成立：复跑 `/tmp/ceva_review_round30_phase_cache.py`。修复前同一 VPU 先执行 128×64 SILU 得到 5.28874e-05 s，同 op_id 载入合法的 128×128 新轨迹后仍返回旧值，而直接编译与新 VPU 对照均为 1.057478e-04 s；`execute_profile()` 同时返回旧时延与 32,768 B，新形状应为 65,536 B。修复后重载即返回 1.057478e-04 s，剖面时延与搬运字节均与直接编译一致；新增反例覆盖同 op_id 换产物、改周期表、改区间倍率、改时钟，正例确认输入不变时仍命中缓存 | 第 4 轮为避免逐 decode 形状重复求值引入结果缓存，但缓存键只有 `(op_id, 模板已登记的符号维)`，没有绑定轨迹模板（含静态宽度与编译参数）、周期表、区间倍率与时钟，`load_pim_trace()` 替换同编号轨迹时也不清空缓存。同一根因影响所有 `work_binding=static_shape` 算子及 `execute()`、`estimate()`、`execute_profile()` 三条入口 | 已修复 | `vpu._static_phase_metrics()` 用完整输入身份作缓存键：本次装载的轨迹代号（`UnifiedPIMBackend` 在 `load_pim_trace()`/`load_tile_range_trace()` 每次装载时递增，绑定该次装载的完整模板、静态形状与 `compile` 参数）、运行时维度、周期表、时钟与严格模式标志；任一输入变化都重新求值，输入不变时命中；不按模板内容逐次冻结，周期表的冻结身份也按对象身份复用，避免逐算子构造大键拖慢整模型仿真。三条执行入口共用该缓存点；同编号轨迹重载后代号变化即自然失效，无需 VPU 侧失效通知 | grep `_phase_metrics_cache` 只有 VPU 这一处结果缓存；调度器 `latency_cache` 是 fallback 核心的每次运行内备忘录（以 core 标识与输入/输出形状为键），轨迹在仿真前一次性装载，不属于同一根因；`pim_trace_loader._trace_cache` 按文件路径缓存，调度器每次装载前调用 `clear_cache()`。两条装载入口（`load_pim_trace`/`load_tile_range_trace`）都会登记轨迹代号，均有重载反例；没有代号的后端回退为冻结模板身份，保证正确性 | `genesim/src/backend/pim_backend.py`（`_trace_serial`、`trace_serials`、`_record_trace`、`load_pim_trace`、`load_tile_range_trace`）；`genesim/src/vpu/vpu.py`（`_phase_cache_identity`、`_phase_metrics_cache`、`_static_phase_metrics`）；`genesim/tests/sim/test_vpu.py` | `test_phase_metrics_follow_a_reloaded_artifact`（两种装载入口）、`test_phase_metrics_react_to_cycle_cost_changes`、`test_phase_metrics_react_to_zone_factor_changes`、`test_phase_metrics_react_to_clock_changes`、`test_phase_metrics_reuse_the_cache_for_unchanged_inputs`；探针复跑见 7.3 |
| 3 | round30 | 中 | 成立：直接调用 `build_mapping(one_to_one=True)` 实测 `(8,4)` 返回 8 个一对一核，`(9,4)` 非整除仍返回 9 个映射，`(12,4)` 的 TP 宽度 3 非 2 次幂仍返回 12 个，`(0,1)` 返回空映射，`(8,0)` 抛未包装的 `ZeroDivisionError`；官方 CLI 与下游 `PartitionPlan` 均能拒绝这些参数。修复后上述参数全部在除法与映射生成前以字段化 `SystemExit` 失败，合法 `(8,4)`、`(1,1)`、`(64,1)`、`(64,8)`、`(128,1)` 仍生成一一对应计划，CLI 产物与 `PartitionPlan` 接受的映射一致 | 第 29 轮把硬件资源数量校验下沉到公开函数，却把同一入口必需的 DPU 数、流水段数、整除关系与 TP 宽度 2 次幂约束继续留在命令行 `main()`，`build_mapping()` 在 `one_to_one=True` 时先做 `num_dpus // num_stages` 再只校验硬件层级，直接调用可以绕过计划参数契约；同一根因影响零/负数、非整除、非 2 次幂与非法类型 | 已修复 | 抽出 `_validate_split(num_dpus, num_stages)`（非布尔整数、正数、整除、TP 宽度 2 次幂），`main()` 与 `build_mapping(one_to_one=True)` 在任何除法与映射生成之前共同调用；通用共享映射路径保持历史直接调用行为。同类复核发现另一条固定切分计划生产入口 `scripts/export_partition_plan.py::plan_from_fixed` 自带一套相同数值检查但缺少类型检查（布尔会写进计划 JSON、浮点抛 `TypeError`），本轮补上同一类型契约并保留合法计划输出 | grep `build_mapping`/`_hardware_hierarchy`/`num_dpus // num_stages`：`build_mapping()` 只被脚本主入口与测试调用，一对一分支已自行校验；`export_partition_plan.plan_from_fixed` 是第二条计划生产入口，已按同一规则加固；下游 `contracts/partition_plan.py::PartitionPlan.__post_init__` 独立复核四项约束，测试同时断言 CLI 产物、公开函数返回值与 `PartitionPlan` 接受的映射三者一致 | `genesim/scripts/export_fixed_pu_mapping.py`（`_validate_split`、`build_mapping`、`main`）；`genesim/scripts/export_partition_plan.py`（`plan_from_fixed`）；`genesim/tests/sim/test_export_fixed_pu_mapping.py` | `test_one_to_one_rejects_invalid_split_arguments_directly`、`test_one_to_one_accepts_legal_split_arguments_directly`、`test_invalid_split_arguments_fail_before_writing_a_plan`、`test_generated_plan_matches_the_downstream_partition_contract`、`TestLegacyFixedPlanContract.test_plan_from_fixed_rejects_invalid_arguments`、`test_plan_from_fixed_keeps_legal_plans`；探针复跑见 7.3 |

第 31 轮（round31，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round31 | 高 | 成立：复跑 `/tmp/ceva_review_round31_sidecar_consumers.py`。修复前首次缓存真实 sidecar 后交换 TP 分片 DPU 顺序，展开结果由 `[0,1]` 变为 `[1,0]`；把集合通信清单改成 1 条后执行图新增 2 个中转节点，而缓存仍记录 118 条；恢复原文件后 `verify_input_snapshot()` 通过。修复后同一探针为 `consumed_dpus_after_rewrite=[0,1]`、`relay_operators_added=254`、`snapshot_verify_after_restore=passed`；未修改 sidecar 的对照 `/tmp/ceva_review_round31_sidecar_control.py` 仍为 `[0,1]`、118 条、254 个中转 | 第 23～27 轮只把模型 IR、请求 trace、成本 sidecar、PIM MLIR 以及放置读取、产物证据与清单收敛到首次消费台账，没有枚举放置 sidecar 的全部执行消费者；`_expand_multi_shard_operators()` 与 `_attach_collective_edges()` 仍各自 `open()+json.load()` 独立读盘，同一文件的分片定义、集合通信定义可与快照/清单指向不同版本。同一根因影响本地分片形状与核映射、跨核边和中转节点数量、通信字节及最终时延 | 已修复 | 新增 `_execution_sidecar()` 作为放置 sidecar 的执行视图：CEVA 复用 `_compiler_sidecar()` 首次消费并登记的内容，未配置、文件不存在或尚未建账时显式抛错；TP 展开与集合通信生成改读该视图，不再打开文件；通用路径沿用 `_compiler_sidecar()` 的逐次读盘行为 | 按 `compiler_placement_file` 全仓 grep 盘点字段消费者：调度器内 `shards`、`dpu_to_cluster`、`collectives`、`fused_activations`、`absorbed_ops`、`pimir_path` 与 `pimir_sha256`、`symbolic_dim_placeholder`、`zone`、`hardware`、`ir_num_operators` 的读取均由 `_compiler_sidecar()` 缓存供数，本轮补齐仅剩的两个旁路；`main.py::validate_ceva_toolchain` 与 `run_manifest.build_ceva_run_manifest` 的无快照分支属于调度器创建前的工具链预检和独立工具入口，不参与执行图构建，保持读盘；非调度器模块没有放置 sidecar 内容消费者 | `genesim/src/scheduler/gene_sim_scheduler.py`（`_execution_sidecar`、`_expand_multi_shard_operators`、`_attach_collective_edges`）；`genesim/tests/sim/test_compiler_placement.py` | `test_tp_expansion_and_collectives_use_the_first_consumed_sidecar`、`test_missing_sidecar_fails_explicitly_for_execution_consumers`、`test_non_ceva_consumers_keep_reading_the_sidecar_from_disk`；探针复跑见 7.3 |

第 32 轮（round32，2026-11-08）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round32 | 高 | 成立：复跑 `/tmp/ceva_review_round32_same_path_reload.py` 与 `/tmp/ceva_review_round32_in_place_cost.py`。修复前两个装载入口把同一路径覆盖成 128 列轨迹后仍返回 64 列旧对象（`same_object_after_rewrite=true`），时延停在 `5.28874e-05 s`（正确 `1.057478e-04 s`）；在原周期表内把 `LUT` 从 6 改成 60 后仍返回 `7.10506e-05 s`（正确 `6.654826e-04 s`）。修复后同路径重载返回 `1.057478e-04 s`、`reloaded_template_width=128`、`same_object_after_rewrite=false`；原位改成本返回 `6.654826e-04 s`、`stale_after_in_place_mutation=false`，内容不变的对照仍命中缓存 | 第 30 轮把“装载次数”和“Python 对象身份”误当成“输入内容身份”：`load_trace_file()` 只按路径缓存，路径覆盖后继续返回旧对象；`_phase_cost_identity()` 按字典对象 `is` 复用冻结键，原位改值不可见。`trace_serials` 只能证明调用过装载，不能证明装载到了新字节；对象身份只能证明没换字典，不能证明字典值没变。同一根因影响两个装载入口、底层核心的同名轨迹缓存以及 VPU 的 `execute()`、`estimate()`、`execute_profile()` | 已修复 | `pim_isa.parse_trace_bytes()` 提供单次读取解析；`pim_trace_loader.load_trace_file()` 改为按文件字节的 SHA256 内容哈希判定缓存身份：内容变则重新解析并替换，内容不变仍返回同一对象；VPU 周期表缓存键每次按当前内容冻结，删除对象身份快捷缓存。后端与核心共用同一装载边界，装载到的新对象同时进入 `trace_cache` 与 `core.trace_cache`；模板、区间倍率、时钟本来就在内容键里，无需另加代号 | grep `_trace_cache`、`load_trace_file`、`_phase_metrics_cache`、`_phase_costs_identity`：`load_trace_file` 的其余调用（调度器 `use_cache=False` 的产物复核、`dump_trace`）共用新实现；`pim_core` 三个核心的 `load_trace` 都经该入口，随后端同时刷新；VPU 三条执行入口共用 `_static_phase_metrics()`；没有其他按路径缓存轨迹对象的地方。重写路径后 `backend.core.trace_cache` 与 `backend.trace_cache` 指向同一新对象，测试同时断言执行时延、剖面搬运字节与 `estimate()` | `genesim/src/pim/pim_isa.py`（`parse_trace_bytes`、`load_trace_file`）；`genesim/src/pim/pim_trace_loader.py`（`_trace_cache`、`load_trace_file`）；`genesim/src/vpu/vpu.py`（`_phase_cost_identity`、`_static_phase_metrics`）；`genesim/tests/sim/test_pim_trace_loader.py`、`genesim/tests/sim/test_vpu.py` | `test_same_path_rewrite_returns_new_content`、`test_phase_metrics_follow_a_same_path_rewrite`（两种装载入口）、`test_phase_metrics_react_to_in_place_cost_mutation`；既有 `test_phase_metrics_reuse_the_cache_for_unchanged_inputs` 保持通过；探针复跑见 7.3 |
| 2 | round32 | 高 | 成立：复跑 `/tmp/ceva_review_round32_collective_numeric.py`。修复前所有畸形 sidecar 的 `validate(mode="simulation")` 都为 true：`nbytes=true` 被当成 1 B 生成两条各 `0.70025 µs` 的边（合法 4096 B 为 `1.724 µs`），`nbytes=4096.9` 静默截成 4096，`seq_len=-16` 生成负系数并在完整流水求值时才被拒绝，`seq_len=0.5` 被转成 0 后在补边时抛未包装的 `ZeroDivisionError`。修复后四类畸形全部 `simulation_valid=false`，并在首次消费 sidecar 时抛字段化 `ValueError`；合法 4096 B 仍为两条 `1.724 µs` 边，`seq_len=null` 仍按常量计 | 集合通信缺少类似配置链路与成本 sidecar 的逐字段类型和取值约束：生产端 `int(edge.nbytes)`、消费端 `int(edge["nbytes"])`/`int(export_seq_len)` 直接用 `int()` 改写外部值；第 25～28 轮只收敛了 YAML 链路和 HOST 参数，没有覆盖放置 sidecar 内的 `collectives`。同一根因同时影响 `all_gather` 与 `scatter` 的字节量、序列缩放和后续 direct/HOST 路由计时，`dpu_ids` 的同款强制转换也一并遗漏 | 已修复 | `config_loader.collective_edge_errors()` 定义统一契约：`type` 必须是已知语义；`all_gather`/`scatter` 的 `nbytes` 为非布尔正整数，`seq_len` 键必须存在且为 `None` 或非布尔正整数，`dpu_ids` 为非空、唯一、非布尔整数列表；`all_reduce`/`local_slice` 按既有路径跳过，未知语义显式失败。`GeneSimConfig.validate(mode="simulation")` 与调度器 `_compiler_sidecar()` 首次消费共用该入口；`_attach_collective_edges()` 去掉所有 `int()` 改写，并在补边前拒绝映射不到执行资源的 DPU；生产端 `_collective_entry()` 在写 sidecar 前按同一约束拒绝非法边 | grep 两个仓库的 `collectives`、`nbytes`、`seq_len`：放置 sidecar 的集合通信只有 `gene_sim_scheduler` 一个消费者与 `placement_export` 一个生产者；`dependency_transfer_bytes()` 的有限/非负检查保留为通用防线；`all_reduce`/`local_slice` 仍不补边；`dpu_ids` 资源映射检查在修改执行图之前完成。合法 `seq_len=None`、同核复用与真正的零字节依赖语义保持不变 | `genesim/src/config_loader.py`（`COLLECTIVE_EDGE_TYPES`、`COLLECTIVE_SKIPPED_TYPES`、`collective_edge_errors`、`validate`）；`genesim/src/scheduler/gene_sim_scheduler.py`（`_compiler_sidecar`、`_attach_collective_edges`）；`flagos-pim-compiler/genesim_bridge/placement_export.py`（`_collective_entry`、`export_placement_to_genesim`）；`genesim/tests/sim/test_collective_edges.py`、`flagos-pim-compiler/tests/test_placement_export.py` | `test_malformed_collective_nbytes_is_rejected`、`test_malformed_collective_seq_len_is_rejected`、`test_malformed_collective_dpu_ids_is_rejected`、`test_unknown_collective_semantics_is_rejected`、`test_collective_on_unmapped_dpu_is_rejected`、`test_legal_collective_numbers_keep_exact_values`、`test_full_pipeline_sees_the_collective_edges`、`test_full_pipeline_refuses_malformed_collective`、`test_config_validation_rejects_malformed_collective`；`test_export_rejects_malformed_collective_fields`、`test_export_writes_legal_collective_fields_exactly`；探针复跑见 7.3 |
| 3 | round32 | 中 | 成立：复跑 `/tmp/ceva_review_round32_mapping_nan.py`。修复前 `_per_stage_weight_bytes()` 对 NaN 权重返回 NaN，`build_mapping(one_to_one=True)` 仍返回合法外观的 `[[0,0]]`，CLI 退出码 0 并写出 `model_id=nan-weight` 的计划；相同输入的有限权重正例与超容量 1 B 反例行为正确。修复后 NaN/正负无穷/负权重/布尔/字符串在聚合处 `SystemExit`，CLI 在写文件前失败；有限浮点权重、`per_stage_bytes=None`、恰好到限额与超 1 B 的对照保持原有结论 | 第 30 轮只把 DPU 数、流水段数、整除关系和 TP 宽度收敛为共享前置契约，没有把同一公开函数同时消费的权重载荷与容量参数纳入数值域；`_per_stage_weight_bytes()` 和 `build_mapping()` 都依赖大小比较，NaN 的比较恒为 False，`per_layer <= 0` 与 `per_dpu > capacity` 同时失效。同一根因影响公开函数与 CLI 两条 CEVA 生产路径，也影响通用分支的容量判断 | 已修复 | `_per_stage_weight_bytes()` 逐项经 `numeric_utils.finite_float()` 校验有限性并拒绝负权重，合计与每段乘积再查有限；`build_mapping()` 入口新增 `_validate_capacity_inputs()`，在任何比较或映射生成前统一校验 `per_stage_bytes`（None 或有限非负）与两个容量参数（有限正数），一对一与通用分支共用；保留合法浮点权重与 `per_stage_bytes=None` 的既有用途 | grep `per_stage_bytes`、`cluster_capacity_bytes`、`tensor_pu_capacity_bytes`：容量核算只有 `export_fixed_pu_mapping.py` 这一条生产路径；`resolve_pim_capacity_limits()` 已对配置容量做有限/正数校验；`export_partition_plan.py` 不做权重容量核算，无需改动；`build_mapping` 只被脚本主入口与测试调用 | `genesim/scripts/export_fixed_pu_mapping.py`（`_capacity_number`、`_validate_capacity_inputs`、`_per_stage_weight_bytes`、`build_mapping`）；`genesim/tests/sim/test_export_fixed_pu_mapping.py` | `test_weight_aggregation_rejects_non_finite_and_negative`、`test_weight_aggregation_keeps_legal_float`、`test_build_mapping_rejects_non_finite_per_stage_bytes`、`test_build_mapping_rejects_invalid_capacities`、`test_exact_limit_is_accepted_and_one_byte_over_is_rejected`、`test_general_mode_shares_the_same_capacity_contract`、`test_cli_rejects_non_finite_weight_before_writing`、`test_cli_accepts_legal_float_weight`；探针复跑见 7.3 |

第 33 轮（round33，2026-11-09）：

| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件（仓库） | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|------------------|--------------|
| 1 | round33 | 高 | 成立：复跑 `/tmp/ceva_review_round33_collective_extremes.py` 与 `/tmp/ceva_review_round33_collective_routes.py`。修复前 `collective_edge_errors()` 对 `nbytes=10**1000`（`seq_len=null` 与 `16`）和 `nbytes=4096、seq_len=10**1000` 均返回空错误列表，`validate(mode="simulation")` 均为 True；调度器补边时前两种分别抛未包装的 `OverflowError: int too large to convert to float` 与 `OverflowError: integer division result too large for a float`，第三种通过校验并生成两条各 0.0 B、路由时延 0 的边（合法对照 4096 B 两条边各 4096 B、`1.724 µs`，`seq_len=16` 各 256 B、`0.764 µs`）；图编译器 `_collective_entry()` 实测把 `10**1000` 原样写入 sidecar | 第 32 轮把“原始字段是非布尔正整数”误当成“字节量在浮点成本模型中有效”，契约只覆盖类型与符号，没有覆盖任意精度 JSON 整数到浮点的可表示性，也没有覆盖 `nbytes/seq_len` 派生系数的溢出与下溢；生产端与消费端各自重复同一套不完整检查，两处同时漏掉。调度器的 `transfer_bytes <= 0` 快速路径与通用“有限非负”检查保留为合法零传输与其他缺陷的最后防线，但无法区分真实零传输与正字节下溢。这是第 32 轮集合通信数值契约修复范围不完整的复发，同一根因覆盖 `all_gather`/`scatter`、常量/随序列缩放两种系数、配置校验、sidecar 导出、执行图补边与完整流水 | 已修复 | 把“原始字节量—导出序列长度—每 token 系数—运行形状求值”收敛为同一数值契约。消费端新增 `_collective_float_errors()`：`nbytes` 必须可转为有限浮点；非空 `seq_len` 也必须可表示；序列分支的 `nbytes/seq_len` 必须有限且严格大于 0；`collective_edge_errors()` 在类型检查通过后调用它，配置校验与调度器首次消费继续共用同一入口。生产端新增 `_collective_number_errors()`（与消费端同一契约），`_collective_entry()` 在写 sidecar 前调用并抛出带边号的 `ValueError`。`_attach_collective_edges()` 不重复校验，只使用已通过契约的系数；调度器通用有限非负检查与 0 B 快速路径保持不变，纯视图、同核复用与首次驻留的合法零字节语义不受影响 | 按 `collectives`、`_collective_entry`、`nbytes`、`seq_len` 全仓 grep：放置 sidecar 的集合通信生产者只有 `genesim_bridge/placement_export.py::_collective_entry`（`export_placement_to_genesim()` 唯一调用点）；消费者只有 `config_loader.collective_edge_errors()`（`validate(mode="simulation")` 与 `_compiler_sidecar()` 共用）和 `_attach_collective_edges()`（经 `_execution_sidecar()` 读同一份已校验缓存），没有其他入口、旁路或重复实现。复跑第 32 轮探针 `/tmp/ceva_review_round32_collective_numeric.py`，历史 6 个用例结论不变（合法 4096 B 仍为两条各 4096 B，布尔/浮点/负数/零 `seq_len` 仍被拒），本轮扩展没有放松既有契约 | `genesim/src/config_loader.py`（`_collective_float_errors`、`collective_edge_errors`）；`flagos-pim-compiler/genesim_bridge/placement_export.py`（`_collective_number_errors`、`_collective_entry`）；`genesim/tests/sim/test_collective_edges.py`、`flagos-pim-compiler/tests/test_placement_export.py` | `test_collective_numbers_outside_the_float_cost_model_are_rejected`、`test_representable_large_collective_numbers_are_still_billed`、`test_full_pipeline_refuses_malformed_collective`（扩展大整数与下溢两类）、`test_config_validation_rejects_malformed_collective`（扩展三类极值）、`test_export_rejects_malformed_collective_fields`（新增 `nbytes=10**1000` 参数）、`test_collective_number_contract_matches_the_cost_model`、`test_export_accepts_large_representable_collective_numbers`；探针复跑见 7.3 |

### 7.2 未修复问题说明

第 1 轮：无。5 项问题均按最小复现确认成立并修复。

第 2 轮：无。4 项问题均按当前代码和真实 PIM MLIR 复现确认成立并修复；没有判定不成立或本轮不修复的条目。

第 3 轮：无。4 项问题均按当前代码、真实放置 sidecar 和真实 PIM MLIR 复现确认成立并修复；没有判定不成立或本轮不修复的条目。

第 4 轮：无。4 项问题均按真实 PIM MLIR、真实报告夹具和实测周期复现确认成立并修复；没有判定不成立或本轮不修复的条目。

第 5 轮：无。5 项问题均按真实 sidecar、真实 CEVA 配置和真实 PIM MLIR 复现确认成立并修复；没有判定不成立或本轮不修复的条目。

第 6 轮：无。3 项问题均按评审复现脚本和真实 sidecar 确认成立并修复；没有判定不成立或本轮不修复的条目。round05 使用的末维辅助函数 `_ceva_tensor_operand_dims`、`_ceva_bound_last_dim` 在本轮被完整形状核对取代并删除；正向夹具 `_GEMV_ARTIFACT` 改为与执行节点逐轴一致，`test_absorbed_operator_without_a_fused_phase_is_refused` 的 SILU 条目补登 `symbolic_dim_placeholder`，使拒绝发生在原本要验证的融合相位检查上。

第 7 轮：无。3 项问题均按评审探针和真实 sidecar 确认成立并修复；没有判定不成立或本轮不修复的条目。两点实现取舍记录如下：

- 问题 1 的严格匹配只作用于“操作数个数与执行节点输入个数相等”的位置对应关系。真实 SOFTMAX/QUANT 产物有多个结果张量，`_ceva_artifact_axis_error()` 的“完全不可比则返回 None”仍是结果循环跳过中间结果的必要语义；真实 GATHER 产物把查表张量放在第一个操作数（2 个操作数对 1 个执行输入），属于 K18 已登记的不可绑定张量。若把“完全不同”一律改成错误，会拒绝这两类真实产物。本轮改为在位置对应时要求明确匹配，效果上覆盖了评审的 MEM_COPY 与“结果正确但输入错误”反例，且真实 sidecar 的 6850 条记录仍全部通过加载。
- 问题 3 的 p90 采用最近秩口径（有序样本第 `ceil(0.9·n)` 个，与 `_execute_pipeline` 已有的 p50/p95 实现相同），没有另造算法。该口径已写入 `Q12`；独立校准产出方若约定其他分位数定义，需要同步升级 `schema_version` 或补充字段，暂列待确认事项。

第 8 轮：无。2 项问题均先写失败测试复现后确认成立并修复；没有判定不成立或本轮不修复的条目。两点实现取舍记录如下：

- 问题 1 的 `max_query_length` 取“各请求单次前向最大 Tq”的最大值：prefill 请求取有效 prompt 长度，decode-only 请求每次前向只处理 1 个 token，故取 1。混合请求（同时含 prefill 与 decode-only）按逐请求最大值汇总，例如 `[prefill 128, decode-only 64]` 得到 128。
- 问题 2 的“实际使用核数”仍按运行期放置映射引用的不同 PIM 核统计；本轮只改变证据缺失时的处理，不改变核数统计口径。正向夹具的放置计划只引用 dpu 0，因此测试补 `op_to_vpu_id={0: 0}`。

第 9 轮：无。3 项问题均先用评审探针和真实 sidecar 复现确认成立，再按 TDD 修复；没有判定不成立或本轮不修复的条目。三点实现取舍记录如下：

- 问题 1 的“允许融合相位”按真实产物登记，而不是把同文件出现的所有计算原语一概放行：SOFTMAX 额外允许 `pim.eltwise`、`pim.lut`，QUANT 额外允许 `pim.kantor`、`pim.lut`。真实 sidecar 的 6850 条产物全部通过这些集合的核对；其他计算原语仍会被拒绝。
- 问题 2 对实产物的无操作数 `tt.return` 采用“最后一条带结果的语句”作为最终结果。真实 sidecar 的 6562 条相位产物在最后一条语句上都只有 1 个结果，与执行节点的 1 个输出一一对应；写入类算子（如 `pim.kv_cache`）没有结果张量，保持本轮之前按操作数核对的语义。
- 问题 3 的并发上限取配置值 `scheduler.max_inflight_requests`，不取按请求数裁剪后的有效值；请求数量本身已由 `request_boundary.request_count` 绑定。契约从 2 升到 3 后，旧版报告一律保持 `model_prediction`。

第 10 轮：无。2 项问题均先用失败测试复现后确认成立并修复；没有判定不成立或本轮不修复的条目。三点实现取舍记录如下：

- 问题 1 的允许集合按“执行节点”分别登记。真实分块 GEMM 产物固定包含 `pim.barrier`、`pim.convert_layout`、`pim.dma_load`、`pim.dma_store`、`pim.wram_alloc`、`pim.wram_load`、`pim.wram_store` 七条基础设施语句，因此为 GEMM 单独登记；其余节点（含 SILU、MEM_COPY 等）只允许语义原语与融合相位。若算子编译器后续给其他节点增加基础设施语句，需要同步扩展 `CEVA_ARTIFACT_SUPPORT_PRIMITIVES`，否则会被拒绝（与 Q13 同类契约）。
- GEMM 的基础设施语句仍按产物协议整体放行，没有逐条核对 DMA 字节；GEMM 的工作量由 `tt.dot` 分块属性、本地分片宽度与放置字节核对约束。该边界写入 5.1 的 K19。
- 问题 2 的来源核对只读取 `ceva.parameter_sources` 的等级与证据字段，不自动判定证据文字是否覆盖本次目标、形状和边界；这部分仍由正式配置校验（证据非空、`ceva_measured` 不得自述证据缺失或不适用）和报告范围核对（目标、数据类型、形状范围、计时边界）共同约束，语义适用性需人工评审。

第 11 轮：无。3 项问题均先写失败测试复现后确认成立并修复；没有判定不成立或本轮不修复的条目。三点实现取舍记录如下：

- 问题 1 的单行内核登记集合来自真实放置 sidecar 的产物核对：对 6850 条算子记录逐条调用形状核对，只有 GATHER 的最终结果（`1x1x4096` 对 `128x4096`）和 QUANT 的输入/结果（`1x4096` 对 `128x4096`）依赖单行视图豁免；ROPE 的 `1x1x128x128` 属于前导广播维（元素数不变），不依赖该豁免。因此登记集合取 `{"GATHER", "QUANT"}`，没有按“所有 B 路算子”放宽。
- 问题 2 的链路字段缺省值直接沿用运行时 `LinkConfig.from_dict()`：`is_full_duplex` 缺省 True，`intra_cluster_only`/`inter_cluster_only` 缺省 False。报告省略字段时按同一缺省值解释，语义与本次运行一致才通过、不一致仍拒绝，没有引入第二套默认规则。
- 问题 3 只拒绝非正预测，没有新增“零成本视图”算子白名单；设计 §7.2 明确样本观测为 cycles，当前没有合法零周期样本。若后续算子语义确实需要零成本视图，应按算子显式登记而不是放开全部样本。

第 12 轮：无。3 项问题均先写失败测试复现后确认成立并修复；没有判定不成立或本轮不修复的条目。四点实现取舍记录如下：

- 问题 1 的主机参数取 CPUModel 现有有效缺省值（10 µs、1 TFLOP/s、100 GB/s）并在五份场景显式化，没有编造新的主机规格；来源统一标为 `assumption`，需主机实测或甲方确认后才能升级（Q16）。YAML 中 `1.0e12` 缺少指数符号会被 PyYAML 解析成字符串，场景文件统一写成 `1.0e+12`，用例同时断言取值为数值类型。
- 主机证据要求按“本次运行实际落在 CPU 编排核上的节点类型”启用：没有主机编排节点时不额外要求，避免与无 CPU 放置的运行冲突；有节点时 `scope.host_orchestration.op_types` 必须与运行一致，并覆盖每个编排算子与 prefill/decode 两个阶段。
- 主机指标 `service_time_s` 取 CPU 设备在流水线中的忙碌时间，`memory_service_time_s`/`memory_bytes` 取同一设备的存储服务口径；PIM 设备指标不变，主机开销只是从原总时延中单列出来，不改变计时路径（冒烟 `total_time_s` 与 round11 完全一致）。
- 问题 2 只改 `compile_from_pimir()` 的相位分支透传；`compile_signature` 从 round04 起已包含 `max_bulk_transfer_bytes`，既有 `test_prepare_gemm_trace_uses_configured_bulk_transfer_limit` 已锁定该口径，因此缓存复用不会绕过分段上限。

第 13 轮：无。3 项问题均先写失败测试并复现后确认成立再修复；没有判定不成立或本轮不修复的条目。四点实现取舍记录如下：

- 问题 1 的主机证据算子集合取“执行图里落在 CPU 编排核上的真实主机节点减去合成中转”。`COLLECTIVE_RELAY` 没有可校准的主机工作量（修复后执行剖面为零），不能要求也不接受它的实测样本；报告的 `op_types` 必须与真实主机节点集合完全一致，因此报告若把合成中转列为实测主机算子会保持 `model_prediction`。该约定已写入 Q8，需独立校准产出方同步。
- 主机指标的逐类型计数与零成本标记同时写入运行结果 `host_orchestration_metrics` 与 `summary.json`；真实冒烟中 `op_count` 从 2 变为 320，`op_counts` 为 `{ALL_REDUCE: 64, COLLECTIVE_RELAY: 254, MODEL_INPUT: 1, MODEL_OUTPUT: 1}`，`zero_cost_op_types` 为 `[COLLECTIVE_RELAY]`，可以直接看出此前只统计原始模型图的问题。
- 问题 2 的 `multipliers` 是成本模型里的标量项，校验口径与既有 `mac_width`/`tile_size` 等整数项一致（相等比较）；配置侧已有 `pim.tensor_pu_config.multipliers` 的数值与来源校验，因此不存在非法值落入缺省 4.0 后仍被报告绑定的路径。
- 问题 3 的零成本判断只对 `COLLECTIVE_RELAY` 生效，不改变其他零工作量 CPU 节点的既有基础延迟语义，避免通用/旧配置行为突变；通信成本仍由跨核依赖边计费。真实冒烟的端到端总时延不变（PIM 是瓶颈），但主机服务时间恰好减少 0.17526 s，证明重复成本已从主机路径移除。

第 14 轮：无。2 项问题均先写失败测试/真实产物探针复现后确认成立并修复；没有判定不成立或本轮不修复的条目。四点实现取舍记录如下：

- 问题 1 的 M 方向没有编成轨迹内的固定循环，而是由运行时按 `ceil(Tq/pim.tile-m)` 重复一个 M 分块：轨迹体仍是一个 M 分块 × 一个 N 分块，N 仍由 `set_tile_count` 绑定。这样既让大 M 产物的 `tile_m` 行工作量只计一次，也避免和运行时原有的 Tq 重复口径叠加；执行剖面里的 DMA 字节沿用同一重复次数。`tile_m=1` 的现有产物重复次数不变，通用/UPMEM 路径的既有结果不受影响。若把 M 循环固定进轨迹，运行时再按 Tq 重复就会把 M 方向放大 `tile_m` 倍。
- 问题 1 的循环识别按“最内层带 `iter_args` 的是 K 循环，其余按步长对应 tile_m/tile_n”实现；K 折叠（0 层或只剩 M/N 层）时最内层是 N/M 循环。真实 2139 份 GEMM 产物中 `tile_n == tile_k` 的 210 份全部由 `iter_args` 正确区分；`tile_m == tile_n` 且只剩一层循环时无法按步长区分，显式报错，当前真实产物没有这种组合。K 折叠时写回本来就在分块体内，因此“K 循环内不得有 dma_store”的检查只在 K 循环存在时执行。
- 问题 2 的预算核对放在 `_load_compiler_placement()`，GEMM 产物的 `pim.mram-bytes` 核对放在 `_apply_compiler_local_shapes()` 的产物登记里；两者都在 `_prepare_pim_traces()` 之前。旧 sidecar 冒烟先命中产物检查（`op4` 的 4,294,967,296 B 对 8,000,000,000 B），说明两条路径任意一条都能拦住跨预算产物。B 路相位产物仍由 `_bpath_pimir` 按默认硬件生成 8 GiB 属性，本轮按评审范围只核对分块 GEMM 产物，未把相位产物的 `pim.mram-bytes` 纳入门禁。
- 问题 2 的 sidecar version 从 3 升到 4，只新增顶层 `hardware` 字段；非 CEVA 消费者忽略该字段，GeneSim 只在 `_is_ceva()` 分支核对，通用路径行为不变。默认导出仍是 4 GiB / 64 MiB，CEVA 联调必须显式传 `--mram-bytes 8000000000 --sys-reserve-bytes 0`（Q7 已关闭，8,000,000,000 B 本身仍是假设，见 Q2）。

第 15 轮：无。2 项问题均先用复现脚本和真实产物确认成立再修复；没有判定不成立或本轮不修复的条目。四点实现取舍记录如下：

- 问题 1 的 `kernel_tile_n` 门禁放在 `_validate_ceva_gemm_artifact()` 的预算与本地分片宽度核对之后：新反例能命中分块检查，既有 mram/放置字节/目标反例仍按原失败原因报告。真实放置 sidecar 的 224 条 GEMM 产物逐条核对 `kernel_tile_n == pim.tile-n` 全部一致，6850 条记录加载通过。
- 问题 1 的运行时复核只对 CEVA 的 GEMM 生效，且只在轨迹元数据带 `pimir.tile_n` 时比较，复核放在计算 `loop_count` 之前；非 CEVA 与手写模板路径保持原有兼容行为。
- 问题 2 的路由模式按本次生效配置的原始取值比较，不做别名归一；正式校验额外把取值限定为 `direct`/`host_relay`。报告省略字段、与运行不一致，或链路集合不一致，均保持 `model_prediction`；两者共同决定通信模型是否匹配。
- CEVA 放置 sidecar 必须由 `export_pp_placement.py --measure-kernel-tiles` 生成，GEMM 条目才会带 `kernel_tile_n`；不带该选项导出的 sidecar 会在加载时被拒绝，避免退回 conf 常量静默计费。

第 16 轮：无。1 项问题先用评审探针对照复现、再按 4095/4096/5000 边界写失败测试确认成立并修复；没有判定不成立或本轮不修复的条目。五点实现取舍记录如下：

- 评审把 `src/vpu/vpu.py` 列为涉及文件，但核对后 `_execute_runtime()` 已经算出并透传完整 `loop_count`，截断发生在下游的 `UnifiedPIMBackend.execute_runtime()`、CEVA 核心和 `PIMTrace.set_tile_count()`。本轮没有在 VPU 再写一套绑定逻辑，而是在共享绑定入口统一修复，保证 VPU、核心、Python 侧轨迹与 `execute_profile()` 读同一份次数。
- CEVA 路径不允许任何 12 位截断：调度器按 `pim.compile.use_long_loop_counts=true` 生成长循环轨迹；即使手写模板或遗留轨迹仍是短循环，`CEVANPMPIMCore`/`UnifiedPIMBackend` 也会在绑定超过 4095 次时就地把短间接循环升级为 24 位长循环。升级只改内存中的轨迹对象，不重写磁盘文件；下次加载再按需升级。超过 24 位（16,777,215 次）或长循环两半任一超过 12 位都直接报错，不再有可编码的循环形式。
- 通用 `conf/sim.yaml`（`use_long_loop_counts=false`）与 UPMEM 路径保持既有行为：短循环轨迹超过 4095 仍按 12 位截断，只把原来的静默 `min()` 改成告警；没有把“自动升级”下放到 `PIMTrace` 默认路径。需要完整次数的通用场景可显式打开该配置项，模板与 pimir 两条生成路径都会遵守。
- 长循环轨迹的 `_tile_count` 记录完整次数（不再压到 4095），仅用于检查/打印口径；求值仍按 `_tile_count_hi/lo` 的 24 位组合。短循环轨迹的 `_tile_count` 语义不变，既有断言保持通过。
- trace 发射语义变化必须让磁盘缓存失效，因此 `compile_signature` 的 `schema_version` 从 9 升到 10。CEVA 冒烟首轮 7138 条 trace 全部重编，重跑后 7138 条全部复用且 `total_time_s` 与 round15 完全相同（670.469079 s），说明编码切换没有改变真实模型（N 循环 63 次，远小于 4095）的计时。

第 17 轮：无。2 项问题均先写失败测试复现后确认成立并修复；没有判定不成立或本轮不修复的条目。两点实现取舍记录如下：

- 问题 1 的 CPU↔HOST 链路要求放在正式校验（`mode="simulation"`）中，结构校验不受影响：CEVA 场景的主机编排是固定语义（R1 要求保留主机编排和必要 I/O），因此没有另加“是否启用主机编排”的开关；只要进入正式运行就要求两条链路。五份仓库场景都已带 `CPU_to_HOST`/`HOST_to_CPU`，新增要求不改变它们的校验结论。
- 问题 2 按结果实际是否包含键来决定写入，而不是按 `_is_ceva()` 判断：`result_maturity` 本就在调度器的 CEVA 分支才设置，按键存在性写入最贴近数据来源，也不会让其他路径误带空值。CEVA 的两种成熟度取值都保留在摘要中。

第 18 轮：无。3 项问题均先用探针复现、再写失败测试确认成立并修复；没有判定不成立或本轮不修复的条目。三点实现取舍记录如下：

- 问题 2 的编译版本身份由 GeneSim 在评估时反推，而不是修改放置 sidecar 契约：为不改变跨仓 sidecar 格式、避免重导现有产物，清单从 PIM MLIR 路径向上定位图编译器源码仓库，再从该仓库 `paths.json` 找 FlagTree 安装目录、从 `env-flagtree.sh` 找源码树，最后哈希实际安装的 `triton-opt`。该方式记录的是评估时的工具链状态（附脏标记），生产端身份嵌入留作 Q18；PIM MLIR 不在 git 工作区内时按“版本缺失”拒绝正式评估（K23、K24）。
- 问题 2 的运行清单只在 CEVA 路径生成；通用结果目录保持原文件集合与摘要键集合，不写 `run_manifest.json`，避免改变通用/UPMEM 行为。
- 问题 3 的容量有限性检查同时放在 `_validate_ceva` 与 `_ceva_capacity_limits`：前者保证结构校验和 `--dry-run` 只返回 False，后者保证直接调用容量解析时抛出带字段名的 `ValueError`；两处都保留原有“GiB 与字节真源一致”核对。

第 19 轮：无。2 项问题均先用真实 TP2 产物和失败测试确认成立再修复；没有判定不成立或本轮不修复的条目。两点实现取舍记录如下：

- 问题 1 只把“取哪张图”改为执行图，没有改变验证样本的核对口径：GEMM 仍按 `scope.shape_range` 与本次运行各轴 `min/max` 包络核对，非 GEMM 仍按阶段执行形状集合核对。逐样本精确匹配到某个分片组合会超出本轮评审范围，剩余边界见 7.3 末尾。
- 问题 2 的环境覆盖通过遍历已加载配置的键路径实现，与 `get()` 的查找键一致；未出现在配置文件中的键不会被补进清单。该取舍保持 `resolved_config()` 的“解析后配置”语义，不把无关环境变量写进结果。

第 20 轮：无。3 项问题均先用真实 B 路产物、评审所述输入和失败测试复现后确认成立再修复；没有判定不成立或本轮不修复的条目。四点实现取舍记录如下：

- 问题 1 的“恒定查表/参数张量”按算子显式登记：GATHER 的操作数 0 是查表，只有索引（操作数 1）与结果随行数展开；QUANT 的全部操作数与结果都是行视图。登记为行视图的张量首维必须是 1，否则直接报错，避免把多行产物静默当成单行重复。该契约是跨仓约定，写入 Q19。
- 问题 1 通过张量形状展开而不是把整条轨迹乘行数：`phase_cycle_breakdown()` 的固定 SYNC 仍只计一次，各相固定循环开销随各自工作量计；测试逐相复算后核对总周期只多一次 SYNC。
- 问题 2 的成本 sidecar 绑定采用内容哈希（`package.cost_sidecar.sha256`），不把文件路径写进报告：换到内容相同的另一路径仍视为同一成本输入，清单的 `inputs.cost_sidecar` 记录实际路径与哈希；换内容或换到内容不同的文件均保持 `model_prediction`。
- 问题 3 的 `value` 字段只在环境覆盖值不等于配置原值时必需；覆盖值与原值相同时原证据仍适用。周期成本的逐项覆盖此前不会被 `get("pim.pim_trace_cycle_costs")` 读到，本轮让 `get()` 在返回整段映射时递归应用子键覆盖，清单值与实际执行值同源；`resolved_config()` 仍只包含配置树中已声明的键，配置树未声明但来源台账登记的覆盖由清单 `env_overrides` 单列（round19 记录的这一边界至此在清单层面关闭）。运行清单版本仍为 1，`env_overrides` 是附加字段，旧消费方忽略即可。

第 21 轮：无。5 项问题均先用修复前探针和失败测试确认成立再修复；没有判定不成立或本轮不修复的条目。

根因分析与受影响入口：

- 问题 1：`autotune.enabled` 在 CEVA 校验里只约束最终验证后端，没有禁止搜索本身；两条配置入口（YAML、`GENESIM_AUTOTUNE_ENABLED`）都经 `GeneSimConfig.get()` 进入同一判断，`override()` 只是构造另一份同语义配置；调度器侧 `_autotune_placement()` 的唯一调用点是 `run()`。本轮在配置校验和 `run()` 两处同时收敛，覆盖直接调用。round05 只收敛“最终验证不能用预测器”，没有禁止搜索本身，属于同类修复未覆盖根因的复发。
- 问题 2：`_load_cost_sidecar()` 的返回值没有消费者，`annotated` 对空条目照样自增，成熟度侧只比较文件哈希；运行清单的 `cost_sidecar_entry()` 与报告校验各自算哈希，没有共同的消费证据。本轮把“实际写入的字段、覆盖范围、未覆盖算子”统一为 `_build_cost_sidecar_ledger()` 的台账，由 `run()` 结果、运行清单和成熟度共用。round20 只完成了文件身份绑定，没有验证成本是否真正落值，属于同一类门禁只在文件层生效的复发。
- 问题 3：成本模型摘要和 `CEVA_REQUIRED_SOURCE_PATHS` 按固定键逐项列举，`use_long_loop_counts` 既不在摘要也不在来源清单，报告边界因此无法识别编码切换；同类的“配置说一套、实际跑另一套”还有运行时短→长升级。本轮把开关、编译期原语族和运行期升级结果一并纳入同一契约，环境覆盖仍走同一 `get()` 生效值。round20 的来源修复只检查已登记来源的探针参数，没有盘点所有会改变执行成本的生效参数，属于同类遗漏的复发。
- 问题 4：倍率校验只存在于 `config_loader._validate_zone_cycle_factors()`，而共享编码函数 `zone_scale*` 和两条公共编译入口直接 `round()` 外部值，直接调用、解析求值和校准脚本都能绕过。本轮把校验下沉到 `pim_isa`，所有接收外部倍率表的边界共用；配置入口只是其中一个调用方，并额外要求三个区间齐全。该规则从 round01 起只在 YAML 入口生效，属于校验只守配置、未守编码边界的复发。
- 问题 5：`build_ceva_run_manifest()` 把 `config.config_path` 当作所有配置对象的唯一持久来源，而 `override()` 用临时 YAML 构造新实例后立即删除；直接文件配置、带参数/无参数覆盖、环境覆盖都走同一个 `build_ceva_run_manifest()`。本轮让配置对象保留 `source_path`/`override_chain`，清单按“持久文件路径+哈希”或“内存生效配置哈希”二选一归档。

五点实现取舍记录如下：

- 问题 1 只在 CEVA 正式校验（`mode="simulation"`）拒绝自动放置搜索，结构模式仍允许，便于 `--dry-run` 做接线联调；调度器 `run()` 对所有 CEVA 配置无条件拒绝，覆盖绕过配置校验的直接调用。通用场景的自动调参路径和既有测试保持不变。
- 问题 2 的覆盖判定把“sidecar 条目”和 `coverage.template` 都视为已声明，只有既无条目又未被模板列表声明的算子才算未覆盖。`source_name` 单独存在不算有效成本，必须至少有一个算子真正写入正数 `mram_traffic_bytes`；这正是“文件存在/哈希一致”之外的消费证据。部分覆盖只有在 sidecar 显式声明模板路径且报告复算同一组五计数时才可升级，验证范围因此被限制在声明的覆盖内。
- 问题 3 除配置开关外还登记运行时升级：`PIMTrace._upgrade_indirect_loops_to_long()` 在短循环超过 12 位被就地升级时写 `runtime_loop_upgrade` 元数据，成本摘要据此把 `LOOP_BEGIN_INDIRECT_LONG` 加入实际原语族。报告必须同时匹配配置开关和运行期升级结果，不能只按 YAML 取值复用。直接编译接口允许部分区间表（如只给 `kv`），配置入口仍要求三个区间齐全，避免改变既有调用方。
- 问题 4 的共享校验函数放在 `pim.pim_isa`（编码定义处），配置层调用时传 `require_all=True`；`zone_scale*` 和三条公共编译/求值入口对同一张表给出同一 `ValueError`，不再出现“YAML 拒绝、直接调用静默舍入”的两套结论。
- 问题 5 不为内存配置单独写生效配置副本：清单直接归档规范化 `resolved_config()` 并按其 SHA256 建立内容身份，可从 `run_manifest.json` 逐字节复算；`config_file.path` 在内存配置时为空，原始文件路径与哈希放在 `source`，覆盖链以 `override_sections` 记录。持久文件配置的 `path`/`sha256` 字段保持不变，旧消费方无需适配。

第 22 轮：无。3 项问题均先用评审探针与失败测试确认成立再修复；没有判定不成立或本轮不修复的条目。

根因分析与受影响入口：

- 问题 1：相位路径把 `mac_count = M×K×N` 交给 `_phase_work_units()` 做一次全局 `ceil(mac_count/mac_width)`，而分块路径 `PimirProgram.mac_groups()` 对每个输出分别取整。同一个矩阵乘在 K 有尾量时因此给出两种指令数。受影响的入口只有 `compile_phase_chain()`（编译成轨迹）与 `phase_cycle_breakdown()`（运行时解析周期），两者共用 `_phase_work_units()`；`PhaseStep.operand_tensors` 已保留逐操作数形状，运行时模板解析也会重建形状，因此按形状重算即可，不需要新增字段。之前各轮只覆盖 K 整除 `mac_width` 的形状（如 K=16/32/4096），尾量从未进入测试，问题因此长期未暴露。本轮把形状解析抽成公共函数，分块与相位两条路径使用同一逐输出公式，并补齐非整除、transposeB、量化 scale 与运行时模板四条用例。
- 问题 2：round21 的台账已经区分 `annotated_ids`/`cost_ids`，但计算 `uncovered` 时仍用 `entry_ids`，成熟度门禁也只要求全局 `cost_operators > 0`；因此“条目存在”被当成了逐算子的成本覆盖。实际加载入口、成熟度前按磁盘重建台账、运行清单三处都调用 `_build_cost_sidecar_ledger()`/`_cost_sidecar_consumption()`，共用同一错误集合。本轮把覆盖集合改为 `cost_ids`（正数 `mram_traffic_bytes`，含 measurements 回退）与 `coverage.template` 的并集，`source_name` 只保留在 `annotated_operators`，不与真实成本混同；报告契约升到 10，覆盖范围仍与台账逐项比较。round21 只验证了“全空文件/全空条目”和“部分覆盖未声明”，没有验证“一个有效成本 + 多个空条目”，属于同类门禁只在整体层面生效的复发。
- 问题 3：round16 统一了 `execute_runtime()/set_tile_count()` 主路径，但新增的 `PIMTracePIMCore.set_tile_count_long(outer, inner)` 把公开契约的嵌套次数直接转发给低层 `PIMTrace.set_tile_count_long(hi, lo)`，绕过了短轨迹升级、完整次数编码和 24 位越界检查；现有测试只直接检查轨迹对象的两半 12 位边界，没有经过后端接口。本轮在 CEVA 核心内先求 `outer×inner`，再走直接计数入口；低层 hi/lo 接口和通用后端保持原语义，`execute_profile()` 仍读取同一 Python 轨迹。`PIMTrace.set_tile_count_nested()` 没有任何调用方，也不参与求值，不构成旁路，本轮未改动。

三点实现取舍记录如下：

- 问题 1 只改 `_phase_work_units()` 的乘加分支，不把逐输出分组写进 `PhaseStep` 新字段：`operand_tensors` 已保存完整形状，运行时模板解析也会重建它们，重算即可避免静态编译与缓存模板出现两套公式。
- 问题 2 的“有效成本”沿用 `_mram_bytes_of()` 的既有口径（顶层标量/分档字典/measurements 回退），没有把 `source_name` 计入覆盖：只说明内核身份的条目不能证明成本已经量到。显式模板路径仍由 `coverage.template` 声明，`coverage` 五个计数与未覆盖列表继续由同一台账产生。
- 问题 3 的长循环接口要求两个因子为正整数；`outer` 或 `inner` 为 0/负数时不再乘出 0 次或负数次数，直接报错。总次数仍受 24 位上限制约，超过时由 `PIMTrace.set_tile_count()` 抛出原有错误。

第 2 轮补充核对结论：

- 问题 1 的缩放边界：相位轨迹按符号维比例整体缩放，SOFTMAX 中与 KV 无关的相位会被 KV 比例放大（K11），本轮按最小改动保留该边界。
- 问题 1 的兼容边界：旧 sidecar 没有 `symbolic_dim_placeholder` 时，非 CEVA 配置保持改动前的运行时重复口径（K12）；CEVA 正式评估已直接拒绝。
- FlagTree 工作区保持干净，`git status --porcelain` 无输出，HEAD 为 `0d7273bac`；本轮未修改 F 源码，无需重新构建。

第 3 轮补充核对结论：

- 评审把首个失败节点 op6867 记为 SILU，按修复后生成的轨迹文件名核对实际是 VECTOR_MUL 分片（`op_6867_VECTOR_MUL.pim_trace`）；两者在同一条 TP 展开链路上，继承逻辑相同，不影响结论和修复范围。
- round02 问题 1 此前未覆盖的 TP 展开边界在本轮补齐：展开后的 VECTOR_MUL 分片轨迹登记为 `work_binding=static_shape`、`shape_scale={'Tq': 128}`，真实冒烟可跑完（见 7.3）。
- FlagTree 工作区保持干净，`git status --porcelain` 无输出，HEAD 为 `0d7273bac`；本轮未修改 F 源码，无需重新构建。

第 4 轮补充核对结论：

- G `genesim_bridge/placement_export.py:229-233` 的占位值生产端保持 128，是运行时模板把符号维对回实际形状的必要输入；本轮 G 未修改代码，只核对该结论。
- 相位链中不对应算子输入/输出的中间张量仍按占位形状保留（K15），属于 Tq 远小于 128 时的剩余高估；占位请求 Tq=128 下无误差。
- 全链路回归首次运行时在 `Tq=KV=726` 的 GEMV_SCORE 上暴露 K 缓存读取朝向问题，已按收缩维推断修复并重新跑通（见 I13、7.3）。
- FlagTree 工作区保持干净，`git status --porcelain` 无输出，HEAD 为 `0d7273bac`；本轮未修改 F 源码，无需重新构建。

第 5 轮补充核对结论：

- 真实 sidecar（6850 条算子记录、32 条融合清单、32 个吸收节点）通过新的加载核对，吸收节点全部绑定到 gate_proj 主算子；本轮 G 未修改代码，`fused_activations` 与 IR `subgraphs` 已足够支撑归属核对。
- SPLIT 视图产物按整头张量表达，与单头执行节点无法逐轴比较，改为元素总数核对（I16）；GEMM A 路产物以指针传参，用分块和放置字节代理形状核对（K17）。
- FlagTree：本轮未修改 F 源码，只核对评审结论（`triton-opt`、4-bit 分组反量化矩阵乘与 `0d7273bac` 工作区干净），无需重新构建；W4A8 的 GML/bin 仍由 `run_full_pipeline.py` 的独立 CEVA 分支产出，不能据此认定 GeneSim 已消费同一 W4A8 产物。

第 1 轮补充核对结论：

- FlagTree 工作区保持干净（`git status --porcelain` 无输出），本轮未修改 F 源码，无需重新构建。
- G 侧 `genesim_bridge/placement_export.py` 的 GEMM 条目与 `_attach_bpath_pimir` 的 B 路条目都同时写 `pimir_path` 和 `pimir_sha256`（约第 343～344 行、第 618～624 行），满足问题 2 对导出端的要求，第 1 轮未改 G 代码。

第 23 轮：无。3 项问题均先用评审探针和失败测试确认成立再修复；没有判定不成立或本轮不修复的条目。

根因分析与受影响入口：

- 问题 1：round20～22 分别绑定了“当前磁盘哈希”和“消费覆盖计数”，但没有把实际消费的不可变快照作为成熟度、台账和清单的共同真源。读取输入的入口共有五个：`load_model_ir`（模型 IR）、`load_trace`（请求 trace）、`_compiler_sidecar`/`_load_compiler_placement`（放置 sidecar）、`_load_cost_sidecar`（成本 sidecar）、`_register_compiler_artifacts`（PIM MLIR）。修复前这五个入口没有统一的记录点，`_ceva_cost_sidecar_error` 用 `cost_sidecar_entry()` 重读磁盘，`build_ceva_run_manifest()` 在运行结束时重读 IR/放置/trace/成本，`_pim_mlir_entries()` 也重读产物；因此只要文件在加载后被改写，旧执行结果就会与新文件身份绑定。修复在五个入口统一登记 `{path, sha256}`，`_compiler_sidecar` 首次读取后缓存，成本台账哈希改为实际解析字节的哈希，成熟度和清单只读快照，`_ceva_run_manifest()` 在生成前逐项复核。`build_ceva_run_manifest()` 无快照参数的旧调用路径只保留给独立工具，正式调度器始终传入快照。
- 问题 2：round21～22 把“有效成本”实现为正数判断，遗漏数值域。`_mram_bytes_of()` 是标量、分档字典和 measurements 回退三种形态共用的提取函数，加载端用它落值、台账用它算覆盖、运行时用 `mram_traffic_bytes > 0` 替换 `data_bytes`；三处共用不完整的定义，任何一处单独过滤都挡不住执行消费非法值。修复把校验放在 `_load_cost_sidecar()`/`_cost_sidecar_consumption()` 的统一解析边界，逐 operator 检查三种形态，校验通过后才允许落值。CEVA 路径新增契约，非 CEVA 保持既有行为。
- 问题 3：round12 把“显式 YAML 存在”当成“生效参数合法”，round20 只补了覆盖后的来源适用性。实际数据流是 `_validate_ceva` 读 `raw_config`，`CPUModel` 用 `config.get()`（含 `GENESIM_*` 覆盖）计算，VPU 工厂又通过 `_from_dict` 只传 `num_cores`、忽略 `cpu_model` 段，三条入口各取一套值。修复把数值契约收敛到 `ceva_host_parameter_errors()` 一个实现，`_validate_ceva`、`CPUModel.__init__` 和工厂 `_create_cpu_model` 共用；工厂在 CEVA 下显式传入生效 `cpu_model` 段与 `pim_type`，使配置校验、摘要和实际执行使用同一组主机参数。通用配置不启用契约，`conf/sim.yaml` 与 UPMEM 路径行为不变。

复发原因说明：问题 1 是 round18 引入清单、round20 绑定成本哈希、round21～22 细化覆盖口径后，三轮都只在“当前磁盘哈希/覆盖计数”上打补丁，没有建立统一的消费快照；问题 2 是 round21～22 把“有效成本”定义为正数判断，未补数值域；问题 3 是 round12 只校验原始 YAML、round20 只补来源适用性，没有统一生效值契约。本轮均按共同边界一次性收敛，并用多入口反例覆盖。

三点实现取舍记录如下：

- 问题 1 的快照只记录路径与内容哈希，不复制输入文件本身：这是评审要求的“内容身份”语义，输入被删除或改写时正式评估显式失败；若需要离线重放，可在此之上另行归档内容副本。
- 问题 2 允许 `mram_traffic_bytes` 为 `None`（生产方未测得时写 `None`）并保持 measurements 回退；零值合法但不计入“有效成本”覆盖。这与生产方 `cost_extractor._measurement_dict()` 的实际形态一致，避免误伤真实 sidecar。
- 问题 3 的工厂改动只对 CEVA 生效：通用配置仍走原有 `_from_dict(cpu_cfg)`，不读取 `cpu_model` 段、不启用数值契约，保证 `conf/sim.yaml` 与 UPMEM 路径行为不变。

第 24 轮：无。3 项问题均先用评审探针和失败测试确认成立再修复；没有判定不成立或本轮不修复的条目。

根因分析与受影响入口：

- 问题 1：round23 已经建立“消费时快照”，但模型 IR 与请求 trace 的快照仍由加载后的第二次读盘生成，解析用的字节和登记用的字节不是同一份；评审测试只覆盖“完整加载结束后的改写”，没有覆盖解析与登记之间的窗口。`_ceva_request_boundary()` 在成熟度派生时还会重新读盘取边界证据，属于同一根因的旁路，因此本轮一并改为优先使用已加载的 trace 对象。放置 sidecar、成本 sidecar 与 PIM MLIR 已按同一次读取解析并哈希，不是本问题的受影响路径。修复把读取、解析、哈希收敛到同一份字节，并在 CEVA 调度器的统一钩子登记，两个入口共用同一实现。
- 问题 2：round23 把主机参数契约放在配置校验与 CPUModel 构造器内，但工厂按数量循环创建资源，零/负数量时构造器不会执行；`cpus` 列表能直接替换 `system.num_cpus`；运行后清点只看 GPU/PIM。根因是“配置通过、模型构造通过、实际资源一致”三套判断并存。修复把 `nodes/gpus/cpus` 替代入口、主机参数和实际 CPU 数量收敛为配置校验、完整工厂和运行后清点共同遵守的契约，工厂在任何资源循环前失败。
- 问题 3：round23 的成本数值域只覆盖 NaN、正负无穷、负数、字符串、布尔和列表，没有覆盖 Python/JSON/YAML 可解析但不能转换为有限浮点的大整数；其他校验入口又各自直接调用 `float()`。本轮按 grep 全仓结果建立 `finite_float()` 统一边界，覆盖成本、配置、区间编码、验证报告、VPU 辅助、容量解析、`pim_compiler` epsilon 以及预测器/autotune 的同类转换；配置校验返回 False，直接编译接口抛 `ValueError`，成本 sidecar 抛字段化 `RuntimeError`，VPU 辅助回落默认值，语义与各入口既有契约一致。真实 sidecar 与合法 1.5 倍对照保持通过。

复发原因说明（对前几轮同类修复的复核）：

- 问题 1 是 round23 把“加载后重新哈希”当成“消费时快照”，只修了“加载完成后改写”这一种时序，未覆盖解析与登记之间的第二次读取；同类问题在 round18～round23 反复出现，都是因为身份登记与内容解析不是同一次读取。本轮改为单次读取后，该窗口不复存在。
- 问题 2 与 round12/round23 同源：契约只放在某一层（配置文件或单个模型构造器），而资源创建还有按数量循环、列表入口和运行后清点等旁路；只在单例上补校验，换个入口仍能绕过。本轮把契约放到工厂循环之前并复用同一函数，补充了完整工厂与资源清点用例。
- 问题 3 与 round18/round23 同源：每次只给被点名的字段补 NaN/Inf/类型检查，没有统一“转换可能溢出”的边界；本轮按 grep 结果一次性收敛所有校验/默认值入口，并用大整数与 `1e308` 反例覆盖。

扩展理由（评审修改思路之外的同类路径）：

- `_ceva_request_boundary()` 的运行中重读是本轮问题 1 的同一根因（边界证据不是消费内容），评审未点名但属于同一入口链，故一并修复。
- `pim/capacity.py`、legacy 容量迁移、`pim_compiler` epsilon、性能目标、数据集观察、放置权重和预测器特征/成功样本判断中的 `float()` 与评审点名的校验入口同根：外部大整数会抛未包装 `OverflowError`。这些改动只影响非法输入的异常路径，有限值的返回值与计时不变，已用全量回归和全链路结果确认。
- 非 CEVA 成本 sidecar 保持既有行为：`_mram_bytes_of()` 不做新增过滤，CEVA 在提取前已由成本数值域拒绝非法值；通用 `conf/sim.yaml`、预测器与 UPMEM 路径的有限输入行为不变。

实现取舍：

- `BaseScheduler` 对非 CEVA 也只读一次字节，但只在 CEVA 钩子里计算 SHA256；解析结果与原路径一致，通用计时不变。
- `Trace`/`ModelIR` 的按路径加载接口保留给其它调用方（编译器、脚本），调度器改用按字节接口；`_ceva_request_boundary()` 保留“尚未调用 `load_trace` 时回退配置路径”的兼容分支，正式运行始终使用已加载 trace。
- `finite_float()` 只接受 `int/float` 且排除 `bool`，与原校验的类型语义一致；`None`、空串等非数值仍按各入口原契约处理。

第 25 轮：无。2 项问题均先用失败测试和复现脚本确认成立再修复；没有判定不成立或本轮不修复的条目。

根因分析与受影响入口：

- 问题 1：链路实现只覆盖“路由存在”和“参数有来源”，没有建立 CEVA 通信数值契约；配置校验、`CommLink.from_dict()`、工厂 `_make_comm_link()` 和报告 `_normalized_links()` 各自处理字段，后三者没有统一的正值/有限性检查。同一根因影响 HOST↔PIM、CPU 中转、跨核 direct/host_relay 以及全/半双工集合通信；本轮把契约收敛到 `link_field_errors()`，并在配置校验、通信库构造、完整工厂和报告比较四个入口复用。
- 问题 2：第 24 轮引入的 `finite_float()` 只落在配置校验和被点名的辅助函数，没有下沉到指令轨迹与相位解析共用的周期表解析器；`strict` 的“完整”仍然只是键集合完整，且配置校验只遍历必需键。同一根因影响 `PIMTrace.estimate_total_cycles()`、`estimate_cycle_breakdown()`、CEVA 核心和 `phase_cycle_breakdown()`，普通原语与每字节成本均可触发；运行期直接读取的 `REDUCE_PARTIAL` 也是同一根因的旁路。本轮把数值域检查放进 `resolve_cycle_costs()` 的 strict 分支，配置校验遍历全表，并补齐归约时长入口。

复发原因说明（对前几轮同类修复的复核）：

- 问题 1 与 round03/round17/round24 的“链路/参数契约”同源：每次都只补链路方向、来源台账或某个数值点，没有把带宽/延迟作为一个字段契约统一实现；工厂一直保留第二份 `float()` 转换。本轮删除重复转换并把契约放进 `from_dict()`、构造边界、配置校验和报告比较。
- 问题 2 与 round01 问题 5、round02 问题 4、round24 问题 3 同源：每次只修配置校验或被点名的函数，直接核心、公共解析器和相位入口仍能绕过。第 24 轮虽然统一了 `finite_float()`，但没有把约束下沉到共享解析器，属于“只在探针入口生效”的复发；本轮把 strict 语义放进 `resolve_cycle_costs()` 并让所有求值入口透传。

扩展理由（评审修改思路之外的同类路径）：

- `_estimate_reduction_time()` 直接读取 `REDUCE_PARTIAL` 并乘进报告时延，是周期成本数值域的旁路；评审未点名，但与问题 2 同一根因，故一并校验。
- `_validate_ceva()` 原先只校验 `CEVA_REQUIRED_CYCLE_COST_KEYS`；`_ceva_cost_model_summary()` 与 `_estimate_reduction_time()` 会读取非必需键，故改为遍历成本表全部条目。
- 报告 `_normalized_links()` 与配置/构造共用 `link_field_errors()`，避免报告比较放行零/负带宽。

实现取舍：

- 通用 `CommLink` 直接构造仍允许零带宽（单路返回无穷），只为 CEVA 配置与构造边界启用正数契约；这是评审明确要求的既有行为保留。
- `resolve_cycle_costs()` 的非严格路径仍保持内置默认值与原有 `float()` 语义；数值域契约只在 strict（CEVA）路径启用，通用/UPMEM 合法有限值行为不变。
- `_positive_float`/`positive_float` 与预测器特征中的成本转换维持“非法值回落默认”的既有语义，它们不是轨迹/相位求值入口；CEVA 正式配置在进入这些路径前已由配置校验拒绝非法值。

第 26 轮：无。3 项问题均先用失败测试和复现脚本确认成立再修复；没有判定不成立或本轮不修复的条目。3 项都是前几轮同类修复未覆盖完整契约的复发：

根因分析与受影响入口：

- 问题 1 与 round04/round06/round08/round19 的“请求边界/运行形状”同源：前几轮把验证形状改为按请求 trace 展开，却一直沿用 `seq_len` 单字段同时表示有效 prompt、padding 后执行长度和初始 KV；每轮补充的测试又都令 token 数等于 `seq_len`，等值夹具让合并口径看起来正确。同一根因影响 `_ceva_runtime_shape_contexts()`、`_ceva_run_shapes_of()`、`_ceva_run_matmul_stage_m()`、`_ceva_sample_shape_error()`、`_ceva_runtime_query_bounds()`、`_get_operator_with_runtime()`、`_execute_pipeline()` 的 `step_context()`、自适应放置成本聚合、`pim.capacity._request_runtime_contexts()` 与请求指标 10 类入口；本轮把它们统一到 `Request.shape_binding()`，并让 CEVA 歧义请求在运行入口显式失败。
- 问题 2 与 round24 的“CPU 数量契约”同源：第 24 轮只把 `cpus/gpus` 替代入口和主机参数下沉到工厂，没有把 PIM 层级与 engine 数纳入同一“非布尔整数”契约；配置校验用相等比较、工厂直接进入 `range()`、运行后清点和运行清单再用 `int()` 合法化。同一根因影响 YAML、`override()`、`GENESIM_*` 覆盖、完整工厂、运行后清点和运行清单 6 类入口；本轮新增 `ceva_resource_count_errors()` 并全部复用，同时核对目标名 `_<n>e` 后缀。
- 问题 3 与 round11/round25 的“链路语义契约”同源：第 25 轮把链路契约收窄为数值域，三个语义标记仍靠 Python 真值执行；配置校验、`CommLink.from_dict()`、工厂、路由与验证报告各自使用不同判断。同一根因影响 direct/host_relay 路由、同核/跨核筛选、全/半双工计费与报告比较；本轮把端点与三个标记并入 `link_field_errors()`，配置、通信库、工厂和报告共用。

扩展理由（评审修改思路之外的同类路径）：

- 请求形状绑定不只改 `_ceva_runtime_shape_contexts()`：实际执行 `_get_operator_with_runtime()`/`step_context()`、容量峰值上下文、自适应放置成本聚合与请求指标都读取 `seq_len`，只改验证展开会让执行形状与报告继续分叉，因此按根因一并收敛。
- 资源数量契约不只改配置校验与工厂：`main._ceva_core_inventory()` 的 `int()`、`build_ceva_run_manifest()` 的 `int()` 和 `PUPool.from_config()` 的直接相乘会把同一非法值重新合法化或生成错误的池大小，因此运行清单与 PU 池入口一并复用共享契约；`GeneSimConfig.get()` 的环境覆盖类型推断也一并修正，否则合法整数环境覆盖会被误判为字符串。
- 链路标记不只改 `is_full_duplex`：`intra_cluster_only`/`inter_cluster_only` 同时决定同核/跨核筛选和 direct/host_relay 路由，三者在同一 `CommLink` 结构中必须同批校验，否则换个标记仍可绕过。

实现取舍：

- 旧请求 trace 只在 `seq_len == len(input_tokens)` 时自动推断三字段；不相等且未显式登记时 CEVA 正式运行拒绝，通用路径仍按原 `seq_len` 约定执行（用户约束要求通用 `sim.yaml`/UPMEM 行为不变）。二进制 trace 升到 version 3，v1/v2 读取保持兼容。
- `_ceva_request_boundary()` 的 `seq_lengths` 被 `query_lengths`/`initial_kv_lengths` 取代而不是并存，避免报告继续保留合并口径；报告契约因此升到 `schema_version=11`，旧报告一律保持 `model_prediction`。
- 资源数量契约只在 CEVA 配置启用；通用 `system.num_pim_devices` 只做类型安全比较（避免字符串抛 `TypeError`），不引入 CEVA 的允许集合。
- 目标名 `_<n>e` 后缀核对只在名称以 `_<数字>e` 结尾时生效；其它自由命名不做推断，避免把未约定的命名规则强加给外部配置。

第 27 轮：无。3 项问题均先用评审探针复现确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。3 项都是前几轮同类修复未覆盖完整消费链或资源域的复发：

根因分析与受影响入口：

- 问题 1 与 round23/round24 的“输入快照”同源：前两轮把模型 IR、请求 trace、放置 sidecar、成本 sidecar 和 PIM MLIR 的**身份**登记成快照，却没有保存 PIM MLIR 首次读取后的**内容**；成熟度、数据类型、相位和验证样本仍通过 `_ceva_artifact_texts_by_op_type()` 重新读盘，于是快照只能拦住“结束时仍是改写状态”的运行，拦不住“成熟度直调”和“改写后恢复”窗口。同一根因影响 `_ceva_run_compute_dtypes()`、`_ceva_artifact_phases()`、`_ceva_validation_evidence_error()`、`_validate_ceva_fusion_exceptions()`、`_operator_pimir()`、直接 `_ceva_result_maturity()` 与流水线成熟度赋值；本轮统一到首次读取台账，并在成熟度返回前复核输入快照。
- 问题 2 与 round25/round26 的“链路契约”同源：前两轮分别补了数值域和布尔标记，端点只要求非空字符串；GPU 端点因此在配置校验、通信库、工厂和报告比较中全部被当成合法链路。同一根因影响 direct/host_relay 路由、GPU 双向出入口、未知资源域和报告 `_normalized_links()`；本轮把端点集合并入共享 `link_field_errors()`，并让直接 `add_link()` 与结构校验也走同一契约。
- 问题 3 与 round22 的“长循环契约”同源：第 22 轮只修了新增的 `set_tile_count_long()` 备用接口，直接次数、整轨迹重复次数和 Python 轨迹绑定仍各自解释有效次数。同一根因影响 `PIMBackend.execute_runtime()`、`PIMTracePIMCore.set_tile_count()`、`PIMTrace.set_tile_count()`/`set_tile_count_long()` 与 `execute_profile()` 共用的轨迹；本轮把契约下沉到 `require_loop_count()`，所有入口复用。

扩展理由（评审修改思路之外的同类路径）：

- 台账不只服务成熟度：`_operator_pimir()` 是 trace 生成的唯一文本入口，若它继续读盘，成熟度即便改为读台账，trace 与报告仍可能消费不同内容；融合豁免同样直接读盘。因此按根因把这三类消费者一并收敛，并用测试断言 trace 文本与证据文本完全相等。
- TP 展开分片不在放置 sidecar 的 op_id 集合里：只按 op_id 建台账会让真实 6850 条 sidecar、7138 条 trace 的 CEVA 运行在第一个分片就失败（真实冒烟已复现）；台账增加 sidecar 路径键，分片按继承路径复用主算子内容。
- 成熟度复核放在 `_ceva_validation_scope_error()` 末尾而不是 `_ceva_result_maturity()` 开头：数据类型与相位检查会懒加载台账并登记 PIM MLIR 快照，只有这些读取完成后复核才能覆盖本轮实际消费的全部文件；直接调用 `_ceva_validation_scope_error()` 的入口也同时获得同一保护。
- 链路端点不只改 `_validate_ceva()`：`CommLib(config=...)` 的批量注册、直接 `add_link()` 和 `_normalized_links()` 是三个独立入口，只改配置校验仍可被 `CommLib` 或报告比较绕过；结构校验也拒绝已出现的 GPU 链路，避免“结构上合法、正式校验才失败”的分裂结论。
- 计数契约不只改 `execute_runtime()`：`PIMTracePIMCore.set_tile_count()` 与 `PIMTrace.set_tile_count()` 是公开/内部绑定的两个底层入口，只改高层后端仍可绕过；`execute_profile()` 与执行共用同一条已绑定轨迹，因此轨迹层必须拒绝非法次数。

实现取舍：

- CEVA 在台账建立后不再重新解析 PIM MLIR 路径：台账建立时读不到、之后才出现的文件会让运行显式失败，而不是等运行期补读；这与“CEVA 正式评估拒绝静默退化”一致。
- 成熟度派生前的快照复核失败时返回 `model_prediction` 而不是抛错：成熟度函数的契约是给出可信级别，运行清单仍会在生成前对同一快照抛 `RuntimeError`，两层分工不变。
- CEVA 链路端点集合固定为 HOST/PIM/CPU；如果后续目标新增资源域，需要显式扩展 `CEVA_LINK_DOMAINS`，不能靠配置自然放行。
- CEVA 的循环次数要求正整数，与长循环接口一致；通用/UPMEM 路径保留零值、负数告警回落和 12 位截断的既有语义，只拒绝布尔/浮点/字符串等非法类型。
- 台账按 sidecar 原始路径字符串建第二索引，而不是重新解析后的绝对路径：TP 展开继承的是 sidecar 原字符串，按原字符串匹配不会因搜索目录或工作目录变化而失配。

第 28 轮：无。2 项问题均先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。2 项都是前几轮同类修复未覆盖完整输入类别或构造入口的复发：

根因分析与受影响入口：

- 问题 1 与 round23/round24 的“输入快照”同源：前几轮把模型 IR、请求 trace、放置 sidecar、成本 sidecar 和 PIM MLIR 五类输入登记为消费快照，却始终没有把配置对象本身当作第六类输入。`GeneSimConfig.__init__`/`_load_yaml()` 只按路径解析，不保存读取的字节；`override()` 只把最初路径带到新实例；`config_entry()` 的文件分支用 `artifact_entry()` 重读当前文件，内存配置的 `source` 分支用 `sha256_file()` 重读原文件；`_ensure_input_snapshot()` 与 `_ceva_run_manifest()` 的结束前复核也只覆盖五类输入。同一根因影响直接文件配置、无参数 `override()`、带参数 `override()`，以及依赖 `config_entry()` 的完整 `build_ceva_run_manifest()`；`resolved_config` 与 `env_overrides` 本身来自内存，不受影响。本轮把配置源文件纳入“单次读取—解析—登记”边界，并让清单、快照与复核共用同一身份。
- 问题 2 与 round25～27 的“链路契约”同源：契约函数只处理字典形态，`add_links()` 会在 `from_dict()` 前调用 `link_field_errors()`，而直接 `add_link()` 在第 27 轮只补了端点检查，数值域与三个布尔标记仍可绕过；HOST 中转开销不在链路字典里，单独留在 `CommLib.__init__` 用裸 `float()` 转换。同一根因影响直接 `CommLink` 注册、CEVA 通信库构造与完整工厂的 HOST 中转开销；验证报告 `_normalized_links()` 一直走字典契约，不受影响。本轮把契约函数扩展为同时接受 `CommLink`，并新增共享的中转开销检查。

扩展理由（评审修改思路之外的同类路径）：

- 配置来源不只 `config_entry()` 一处：正式清单前还有 `_ensure_input_snapshot()` + `verify_input_snapshot()` 的统一复核。只改 `config_entry()` 仍会让调度器快照缺项，因此把配置作为 `config_source` 纳入第六类输入，并要求 `build_ceva_run_manifest()` 在传入快照时交叉核对配置对象与快照登记一致。
- `config_entry()` 的复核不只用于正式清单：独立工具也会直接调用它生成清单。若只在调度器路径复核，独立工具仍会用当前磁盘文件替换消费身份；因此把复核下沉到 `config_entry()` 本身，两条路径得到同一结论。
- 中转开销的数值域不只属于通信库：完整工厂在 `create_vpus_and_topology()` 中会先执行 `_ceva_resource_contract_errors()`，若只在 `CommLib` 构造函数检查，工厂要到创建通信库时才失败。本轮把共享检查加入工厂契约，使配置校验、工厂与通信库构造在同一个错误集合上失败。
- 配置校验的通用中转开销检查原本是 `validate()` 内的一份重复实现；本轮改为调用 `host_relay_overhead_errors()`，数值域与报错字段名完全一致，避免配置校验、工厂和通信库出现三套判断。

实现取舍：

- `config_source` 只进入内部消费快照，不重复写入清单 `inputs`：清单已有 `config_file` 作为配置的规范记录，重复字段会让消费方难以判断哪一份是权威来源；`inputs` 的四个历史键保持不变，旧消费方无需适配。
- 原配置文件的改写与删除都显式失败，而不只是“保留旧哈希继续出清单”：与五类输入快照的语义一致，运行结果不能与无法证明来源的配置拼在一起。若确实需要在运行后修改配置，应在启动下一次运行前完成。
- `source_path` 属性保留：既有测试和调用方仍使用它表示最初的真实文件；新增的 `source_identity` 才是清单与快照的权威身份，两者在 `GeneSimConfig` 构造时一致。
- 通用 `CommLib` 的 HOST 中转开销仍走原 `float()` 转换（含可转换字符串与整数的旧语义），CEVA 走 `host_relay_overhead_errors()`；这是评审要求保留的通用行为，不把 CEVA 的严格数值域扩散到 UPMEM/通用路径。
- `link_field_errors()` 在 `CommLink` 类定义之前引用该类名，但只在运行期调用，模块导入阶段不会求值；这样不必为了参数类型调整函数定义顺序。

本轮无新增待确认事项：两项修复都不改变 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码，工作区保持干净，构建与验证方式沿用既有 `triton-opt`，无需重新构建。

第 28 轮回归测试结果见 7.3。

第 29 轮：无。4 项问题均先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。4 项都是前几轮同类修复未覆盖完整账本或生产入口的复发：

根因分析与受影响入口：

- 问题 1 与 round04/round20 的“相位轨迹运行时重建”同源：前几轮把 `PhaseProgram` 当作完整工作账本，在它上面补了形状绑定、行视图和 MAC 分组，却没有检查 `parse_phase_chain()` 收集账本时是否覆盖了整份 IR。`parse_phase_chain()` 用“文件是否出现过 `phase_spec`”选择分支，进入相位分支后对所有无属性行 `continue`；`phase_runtime_template()` 只序列化筛选后的 `program.steps`；`compile_phase_chain()`、`phase_cycle_breakdown()`、`vpu._static_phase_metrics()` 与 `trace_metadata()` 都消费同一份不完整账本，因此同一根因同时影响指令编译、运行时模板、解析周期和执行剖面。本轮把 `_parse_single_phase_ops()` 的逐 op 规则抽成 `_oplevel_step()`，让 phased 与 unphased 语句共用同一实现，并用 `_PHASED_LEDGER_EXEMPT_OPS` 明确登记唯一可豁免的零成本布局语句。
- 问题 2 与 round21/round23/round28 的“配置消费身份”同源：前几轮保存了加载时文件字节哈希、把配置纳入快照并在结束前复核，但来源种类仍由清单生成时的 `config_path.is_file()` 推断。缺失路径在加载时没有 `source_identity`，运行中一旦出现文件就被当作新的文件来源读取，属于同一消费快照根因在“加载时缺失”这一时序上的复发。本轮在构造时固定 `source_kind`，让内存身份与文件身份都不再依赖后续磁盘状态。
- 问题 3 与 round26 的“资源数量契约”同源：前几轮把 `ceva_resource_count_errors()` 接到配置校验、完整工厂、运行后清点与运行清单，却没有枚举切分计划的生产入口 `export_fixed_pu_mapping.py`；该脚本的 `_hardware_hierarchy()` 又自建 `int()` 转换，浮点/布尔/字符串数量在到达共享契约之前已被合法化。`build_mapping(one_to_one=True)` 作为公开函数也只信任转换后的元组。本轮把共享契约接到层级读取之前，并让公开分支自行校验。
- 问题 4 与 round26 的“运行清单整数契约”同源：上一轮只把 `engines_per_core`/`clusters_per_device`/`tensors_per_cluster` 等资源数量改为共享整数契约，容量字段 `ceva.local_memory_bytes`/`ceva.sys_reserve_bytes` 仍由 `build_ceva_run_manifest()` 自己 `int()`；`pim/capacity.py` 已有完整容量契约却未被复用。同一根因影响有/无输入快照两条路径，本轮统一调用 `resolve_pim_capacity_limits()`。

扩展理由（评审修改思路之外的同类路径）：

- 问题 1 不只补一条特判：把无属性语句的计费规则抽成与单相 IR 共用的 `_oplevel_step()`，避免同一套“一次遍历”口径在 phased/unphased 两条前端各写一份。除 `phase_spec` 外还纳入无结果语句（`_op_matches()` 同时使用两个正则），否则 `pim.kv_cache` 这类写回在混合文件里仍会消失。未知 op 在没有 `unit`、也没有 `_OPLEVEL_FALLBACK_UNIT` 时直接报错，而不是退回 0。
- 问题 2 不只修 `config_entry()` 的分支：`override()` 也必须把来源种类固定为内存身份，否则临时 YAML 路径在运行中重新出现时仍会被 `is_file()` 换绑。`source_kind` 与 `source_identity` 一起构成来源契约，调度器输入快照仍只登记真实文件身份，二者不冲突。
- 问题 3 不只修脚本主入口：`build_mapping()` 是公开函数，测试和后续工具可能直接调用；一对一分支重复校验层级类型与 CEVA 允许集合，避免“绕过 main 即可出计划”。容量解析仍在层级校验之后执行，非法层级不会先算容量。
- 问题 4 不只修探针点名的浮点：共享容量解析同时覆盖布尔、字符串、负预留、预留吃光物理量与 GiB 视图不一致；清单要求 `ceva.local_memory_bytes` 作为字节真源，避免通用容量字段顶替 CEVA 预算。调度器 sidecar 核对已有严格类型检查，本轮只补反例测试，不重复改写其实现。

实现取舍：

- `_PHASED_LEDGER_EXEMPT_OPS` 目前只登记 `convert_layout`（类型转换器插入的 NoMemoryEffect 布局修正）。相位文件里出现的视图/搬运语句（`reshape`/`transpose`/`gather` 等）不默认豁免，而是按单相规则计真实字节；宁可保守计费，也不把未验证的“纯视图”语义当免费。算子编译器若新增真正零成本的语句，需要显式登记，否则会显式失败。
- 账本 index 改为执行顺序序号，同一 op 的多个相位仍按声明的 index 排序；当前真实产物每个 op 一相且声明 index 连续，未出现需要同时保留“声明索引”和“执行序号”的形态。
- 加载时缺失的配置在运行中出现后保持内存身份，而不是报“来源变化”错误：清单要证明本次实际消费的内容，内存身份是真实结论；是否改用后来出现的文件应由调用方重新加载配置决定。运行中改写/删除已消费的原文件仍保持显式失败。
- 运行清单要求 CEVA 字节真源：只有通用 `pim.cluster_memory_capacity_gb` 而没有 `ceva.local_memory_bytes` 的配置会被字段化 `ValueError` 拒绝，不再以 `int(None)` 的 `TypeError` 失败。
- 映射导出脚本只执行资源数量与容量契约，不重复 `validate(mode="simulation")` 的时钟、来源台账等主流程校验；独立导出工具的职责是生成合法计划，不代替正式评估入口。

本轮无新增待确认事项：4 项修复都不改变 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码，工作区保持干净，构建与验证方式沿用既有 `triton-opt`，无需重新构建。

第 29 轮回归测试结果见 7.3。

第 30 轮：无。3 项问题均先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。3 项分别是前几轮同类修复的复发或未覆盖完整路径：

根因分析与受影响入口：

- 问题 1 是第 29 轮修复的反向复发：第 29 轮正确识别到“无 `phase_spec` 的真实语句不能免费”，但用“有独立 SSA 结果且被下一相消费”推断存在独立硬件遍历，没有核对 FlagTree 的折叠契约。`ExpandPhases.cpp` 把 SOFTMAX 的 `x - max` 明确生成为折进 EXP 相的 FPSU 仿射（`not a phase of its own`），`LowerPIMToEmitC.cpp` 又把无 `phases` 的 Eltwise/Lut 定义为 helper：不拥有独立 engine 遍历，中间值保持 f32、不经过 f16 缓冲。`parse_phase_chain()` 因此为它生成独立的 LOADN→MUL→STOREN；`phase_runtime_template()` 复制了同一账本，`compile_from_pimir()`/`compile_phase_chain()`、`phase_cycle_breakdown()`、调度器 `_phase_runtime_metadata()` 与 `vpu._static_phase_metrics()` 全部消费这份错误归属。第 29 轮新增的 `test_real_softmax_artifact_bills_the_unphased_subtraction` 把错误假设写成断言，全量测试通过反而掩盖了语义回归。本轮把账本从“逐操作计费”升级为“带相位归属的执行账本”：按 SSA 生产者/消费者区分折叠 helper、独立无相位语句与零成本布局，折叠项只并入消费者相的算术，不产生独立搬运。
- 问题 2 是第 4 轮引入相位结果缓存时的遗留：当时缓存键只覆盖“算子编号 + 模板已登记的符号维”，后续轮次不断增加轨迹模板、静态形状绑定、区间倍率与周期表，却没有同步扩展缓存身份；`load_pim_trace()`/`load_tile_range_trace()` 替换同编号轨迹时也没有失效通知。同一根因同时影响 `execute()`、`estimate()`、`execute_profile()` 三条入口，以及所有 `work_binding=static_shape` 的算子；重复评估、A/B 对照、同调度器重编产物或修改成本后都会复用旧结果。现有测试只覆盖同一轨迹下不同 Tq/KV，未覆盖同符号维但不同静态宽度、不同产物或不同成本配置。
- 问题 3 是第 29 轮公开入口修复不完整：第 29 轮把硬件资源数量契约下沉到 `build_mapping()`，却把同一入口必需的 DPU 数、流水段数、整除关系与 TP 宽度约束留在命令行 `main()`。`build_mapping(one_to_one=True)` 在参数校验之前就执行 `num_dpus // num_stages`，零流水段直接抛 `ZeroDivisionError`，非整除/非 2 次幂/零核则生成非法计划。另一条固定计划生产入口 `export_partition_plan.plan_from_fixed` 也维护同一套数值检查，但缺少布尔/浮点/字符串类型检查，布尔数量会被写进计划 JSON。

扩展理由（评审修改思路之外的同类路径）：

- 问题 1 不只把探针的 6 步改回 5 步：新增 `_phase_statement()`/`_phase_consumers()`/`_first_phased_consumer()` 做 SSA 归属，`FoldedPhaseWork` 保存 helper 的输入与结果张量而不是固定工作量，运行时模板按实际形状重算；没有相位消费者的 eltwise/lut（独立 LUT）、matmul、静态量化仍完整计费，未知语句继续失败。否则换一份产物、换一个形状或换一个无属性算子仍会错。
- 问题 2 不只给缓存键补一个宽度维度：在两条装载入口登记轨迹代号（绑定该次装载的模板、静态形状与编译参数），并连同运行时维度、周期表、时钟与严格模式一起纳入身份，三条执行入口共享同一缓存点；输入不变时仍命中，避免把缓存退化成逐次求值。不按模板内容逐次冻结，避免每次执行都构造大键拖慢整模型仿真。
- 问题 3 不只修 `build_mapping()`：`main()` 与公开的一对一分支调用同一份 `_validate_split()`，保证命令行与直接调用结论一致；同时复核到第二条固定计划生产入口并按同一类型/数值契约加固，避免同类参数在另一个公开函数上以布尔或浮点形态漏过。

实现取舍：

- helper 归属用 SSA 消费者链而不是“所有无相位 eltwise/lut 一律折叠”：评审明确要求完全独立的无属性 LUT 仍计 1 步、8192 个工作单元，因此只有被带相位消费者（可跨 helper 链）使用的 eltwise/lut 才折叠；matmul、静态量化与无消费者的 LUT 保持独立计费。一个 helper 结果被多个带相位消费者使用时，工作量按 SSA 顺序并入第一个消费者只计一次；当前 25,173 份真实产物中未出现多消费者形态。
- 折叠工作量用张量而不是整数表达：`FoldedPhaseWork` 保留 helper 的输入与结果张量，`phase_runtime_template()` 按轴绑定序列化、`resolve_phase_template()` 按实际维度重算，避免把占位形状（如 128×128）固化进运行时模板；v2 模板因语义错误并不兼容，版本升到 3 后统一重编。
- 折叠只加算术、不加搬运：消费者相的操作数展开为 helper 的真实输入（x 与 max），EXP 相直接读两者并写回一次，不读写 helper 的 f16 中间缓冲，与 `LowerPIMToEmitC` 的 f32 helper 语义一致。
- 公开映射入口只对 `one_to_one=True` 分支新增切分校验，通用共享映射保持原有直接调用行为（评审要求保持不变）；`--partition-plan` 消费端不重复校验，非法计划在生产端即无法生成。
- 第二条计划生产入口采用就地类型检查而不是跨脚本抽共享模块：两个脚本是独立 CLI 工具，为十几行校验引入导入耦合不划算；两条入口的规则与错误语义已对齐，后续若新增计划生产者应复用同一实现。

本轮无新增待确认事项：3 项修复都不改变 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码，工作区保持干净，构建与验证方式沿用既有 `triton-opt`，无需重新构建。

第 30 轮回归测试结果见 7.3。

第 31 轮：无。1 项问题先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。该问题是第 23～27 轮输入快照修复的同类路径遗漏，不是新增需求：

根因分析与受影响入口：

- 问题 1 的根因不是两个函数各少一次缓存调用，而是放置 sidecar 没有像模型 IR、请求 trace、成本 sidecar 与 PIM MLIR 那样建立统一的“首次消费内容”边界。第 23～27 轮按已点名的入口（放置读取、产物证据、成熟度、运行清单、配置来源）逐个收敛，`_expand_multi_shard_operators()`（shards → 本地分片形状与核映射）和 `_attach_collective_edges()`（collectives → 跨核边与中转节点）从未被枚举，因此继续按磁盘读取。快照只在结束时核对哈希，无法区分“执行时读新文件、清单仍指旧文件”和“改写后恢复原文”，探针复现的正是这两种时序。
- 受影响入口按字段盘点：`shards`（`_load_compiler_placement`、`_apply_compiler_local_shapes`、`_validate_ceva_dpu_mapping`、`_ceva_plan_used_core_count`、`_expand_multi_shard_operators`）、`dpu_to_cluster`（`_load_compiler_placement`）、`collectives`（`_attach_collective_edges`）、`fused_activations` 与 `absorbed_ops`（`_register_compiler_artifacts`、`_ceva_fused_owners`、`_validate_ceva_fusion_exceptions`）、`pimir_path` 与 `pimir_sha256`（`_register_compiler_artifacts` → `_ceva_pimir_texts` 文本台账）、`symbolic_dim_placeholder`、`zone`、`kernel_tile_n`、`hardware`、`ir_num_operators`、`placed_mram_bytes`。除本轮两个入口外，其余读取均已通过 `_compiler_sidecar()` 的 CEVA 缓存与首次消费快照；没有发现第三个执行旁路。

扩展理由（评审修改思路之外的同类路径）：

- 不只给两个探针追加结束前复核：结束前哈希只能发现“改写后仍未恢复”，对“执行时换版本、结束前恢复”无效，而且不改变执行图来源。本轮把两个入口改为消费同一份已登记内容，并让 CEVA 读不到时显式失败；测试同时覆盖首次建账后改写、执行完成后恢复原文的时序，以及未修改 sidecar 的 `[0,1]`、118 条、254 个中转正例。
- 不只修改 TP 展开：集合通信清单同样决定通信字节与中转节点，若只修展开，改写 collectives 仍可绕过台账；两处共用新增的 `_execution_sidecar()`，并核对 `_compiler_dpu_to_cluster` 仍来自同一份内容。
- 通用路径不做统一缓存：`conf/sim.yaml` 与 UPMEM 路径的既有行为是按次读盘，本轮用非 CEVA 用例锁定“改写后第二次展开看到新顺序”，避免把 CEVA 的快照语义扩散到通用场景。

实现取舍：

- 新增 `_execution_sidecar()` 薄封装而不是把 `_compiler_sidecar()` 直接改成抛错：后者在 `_ensure_input_snapshot`、`_apply_compiler_local_shapes` 等入口以“返回 None 表示未配置或尚未就绪”为正常语义，直接加严格失败会改变既有处理；执行消费者需要显式失败，因此用独立薄封装复用同一缓存。
- 执行视图只读：`_compiler_sidecar()` 返回的字典被放置、能力核对、成熟度与清单共同引用，本轮两个新消费者只读取字段，不修改缓存内容，避免一个消费者影响其余消费者。

本轮无新增待确认事项：该修复不改变 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码，工作区保持干净（HEAD `0d7273bac`），无需重新构建。

第 31 轮回归测试结果见 7.3。

第 32 轮：无。3 项问题均先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。3 项分别是前几轮同类修复的复发或未覆盖完整路径：

- 问题 1 是第 30 轮缓存身份修复的范围不完整复发：第 30 轮用“装载代号 + 字典对象身份”代替内容身份，只覆盖了“换路径重新装载”和“换新配置对象”，没有覆盖“同一路径覆盖文件”和“原字典原位改值”。本轮把身份判定下沉到共享装载边界（文件内容哈希）与周期表内容冻结，两个装载入口、底层核心与 VPU 三条执行入口同时受益。
- 问题 2 是第 25～28 轮数值契约修复的范围不完整复发：此前只收敛 YAML 链路和 HOST 参数，没有把放置 sidecar 的 `collectives` 纳入同一契约。本轮在配置校验、调度器首次消费与图编译器导出三处共用同一组字段规则。
- 问题 3 是第 30 轮公开入口修复的范围不完整复发：第 30 轮只收敛切分参数，没有覆盖同一公开函数消费的权重载荷与容量参数。本轮把数值域校验放在任何大小比较之前，并同时覆盖一对一与通用分支。

扩展理由（评审修改思路之外的同类路径）：

- 问题 1 不只让两个装载入口清缓存：评审说“显式重载时读取并核验当前文件字节，或使对应路径缓存失效”。本轮选择在 `load_trace_file()` 本身按内容哈希判定，因为 `PIMTracePIMCore`/`DRAMsim3PIMCore`/`Ramulator2PIMCore` 的 `load_trace()` 也走同一入口；只改后端入口会让核心继续持有旧对象。周期表则按内容冻结，避免只覆盖 `override()` 换对象而漏掉原位改值。
- 问题 2 不只在 `_attach_collective_edges()` 加判断：若只改消费端，畸形 sidecar 仍会先进入 `_expand_multi_shard_operators()` 并完成 TP 展开；本轮把校验提前到 `_compiler_sidecar()` 首次消费处，并在导出处同步拒绝，避免畸形产物跨仓库传播。`dpu_ids` 的资源映射检查放在补边前，保留同核复用不插边的既有语义。
- 问题 3 不只堵 NaN：`build_mapping()` 的两个容量参数与 `per_stage_bytes` 同样参与大小比较，本轮统一纳入有限性/正负校验；`_per_stage_weight_bytes()` 的逐项聚合与每段乘积也分别检查，防止有限输入在求和后溢出成无穷。

本轮无新增待确认事项：3 项修复均不涉及 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码。

第 32 轮回归测试结果见 7.3。

第 33 轮：无。1 项问题先用评审探针与失败测试确认成立，再按 TDD 写失败测试并修复；没有判定不成立或本轮不修复的条目。该问题是第 32 轮集合通信数值契约修复的范围不完整复发：

- 问题 1 是第 32 轮“字段类型与符号”契约的范围不完整复发：第 32 轮只验证 `nbytes`/`seq_len` 是非布尔正整数，把“原始字段是整数”当成“字节量在浮点成本模型中有效”。本轮把可表示性与派生系数纳入同一契约，配置校验、首次消费、生产端导出三个入口同时补齐；`_attach_collective_edges()` 的 0 B 快速路径与 `dependency_transfer_bytes()` 的有限非负检查保持为最后防线，不再承担区分“合法零传输”和“正字节下溢”的职责。

扩展理由（评审修改思路之外的同类路径）：

- 不只在外层包 `try/except OverflowError`：那样只能把崩溃换成报错，仍会让“除以下溢成 0”的正字节边通过。本轮在契约入口同时检查 `nbytes`、非空 `seq_len` 的可表示性和 `nbytes/seq_len` 的有限正数性，使“通过校验”等价于“进入成本模型不会溢出或下溢”。
- 生产端与消费端使用同一组规则：若只修消费端，图编译器仍会导出消费端无法正确计费的 sidecar。本轮在 `_collective_entry()` 写文件前用 `_collective_number_errors()` 拒绝同一批值，避免畸形产物跨仓库传播。
- 测试不只断言辅助函数：正例走真实补边与路由，断言每个参与者的字节数、`nbytes/seq_len` 系数和路由服务时间为正；反例覆盖配置校验、调度器首次消费、完整流水与生产端导出四个入口，并同时覆盖 `all_gather`/`scatter` 与常量/序列两种系数。
- 第 32 轮同类契约复核：复跑 `/tmp/ceva_review_round32_collective_numeric.py`，历史 6 个用例结论不变，说明本轮扩展没有放松既有类型与符号约束。

本轮无新增待确认事项：修复不涉及 64/128 核对应关系、8 GB 精确字节数、时钟域、互连带宽或 W4A8 真实产物；FlagTree 未修改源码。

第 33 轮回归测试结果见 7.3。

### 7.3 回归测试结果

第 1 轮（round01）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `./run.sh --test sim`，45/45 个测试文件通过，合计 869 条用例、1 条跳过、0 条失败（约 90 秒）。
- genesim 单文件（uv 虚拟环境）：`uv run python tests/sim/test_compiler_placement.py`（81 条 OK）、`uv run python tests/sim/test_config_loader.py`（74 条 OK）、`uv run python tests/sim/test_pimir_trace.py`（50 条 OK）。
- 五份 CEVA 场景：依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，在临时放置 sidecar 下 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 `source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"`，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `python scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；`total_time_s=4251.737`、`tokens/s=1.901`，与修复前一致。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

第 2 轮（round02，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `./run.sh --test sim`，45/45 个测试文件通过，0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `python -m pytest tests/sim -q`，894 通过、1 跳过、0 失败（约 28 秒）。
- genesim 受影响文件：`python -m pytest tests/sim/test_vpu.py tests/sim/test_compiler_placement.py tests/sim/test_config_loader.py tests/sim/test_pim_backend.py -q`，218 通过、0 失败。
- 问题 1 真产物复现与修复核对：用 `bpath-fb65415debb6b9a5.pimir.mlir` 生成 SILU 轨迹后，Tq=128 时延等于轨迹时延（不再 ×128），Tq=64/1 分别为 1/2、1/128；用 `bpath-0858498cb704d901.pimir.mlir` 生成 SOFTMAX 轨迹，KV=16/32/64 时延分别为轨迹的 1/8、1/4、1/2，KV=128 时等于轨迹。
- 真实流水线 trace 元数据核对：7138 条 trace 中 6562 条为 `work_binding=static_shape`（4096 条带 `Tq`/`Tp+Tq` 占位，2466 条只带 `Tq`）、512 条 GEMM 为 `runtime_tiles`、64 条被吸收算子为 `absorbed`；样例 `op_1000_SOFTMAX.pim_trace`、`op_1004_GEMV_SCORE.pim_trace` 的 `shape_scale` 为 `{'Tq': 128, 'Tp+Tq': 128}`，`op_1002_ROPE.pim_trace`、`op_1065_RMSNORM.pim_trace` 为 `{'Tq': 128}`。
- 问题 2 复现与修复核对：64 核 4e 场景改 `require_compiler_pimir=false` 后 `validate(mode="simulation")` 返回 False（`true` 时返回 True）。
- 问题 3 复现与修复核对：MAC 1→1000、`mac_width` 8→16、区间倍率变化、报告 `dtype=int4` 均保持 `model_prediction`；报告缺 `cost_model` 或本次运行没有产物时同样保持 `model_prediction`。真实 `/tmp/ceva_placement_onetoone.json` 引用的算子编译产物元素类型集合为 `{f16, f32, i32, i8}`，报告写 `fp16` 可通过、写 `int4` 被拒。
- 问题 4 复现与修复核对：五份 CEVA 场景现包含 LOOP/SYNC 单价与来源；删除任一键后正式校验返回 False；CEVA 核心遇到未配置的 LOOP_BEGIN/LOOP_END/SYNC 直接报错，显式配置后周期按配置值计算（begin 7 + 2×MAC 10 + end 11 + MAC 5 + sync 13 = 46 周期）。
- 五份 CEVA 场景复验：在临时放置 sidecar 下依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。
- G 受影响文件：`python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py tests/test_placement_export.py tests/test_genesim_bridge.py -q`，137 通过、5 跳过、0 失败。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `python scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；sidecar 的 B 路条目均带 `symbolic_dim_placeholder`；仿真成功结束，`total_time_s=1865.332`、`tokens/s=4.334`。与 round01 的 4251.737/1.901 不同，原因是整算子相位轨迹不再按 Tq 重复整条轨迹（问题1 修复），属于预期。
- CEVA-NPM 端到端冒烟本轮未重跑：round01 使用的 `/tmp/ceva_sim_onetoone.yaml` 放置 sidecar 没有 `symbolic_dim_placeholder`，按 round02 的严格口径会被拒绝；需用 CEVA 配置重新导出计划和 sidecar（Q7、Q9）后再跑。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round02 改动无直接关系。

仍存在的不足（round02 增补）：

- 1e 目标文件与独立周期证据仍缺失，1e 画像与成本表仍只作预测（K7、Q3）。
- 整算子相位轨迹按符号维比例整体缩放，SOFTMAX 中与 KV 无关的相位仍会被 KV 比例放大；待分阶段成本表落地后改为逐相绑定（K11）。
- 旧放置 sidecar 没有 `symbolic_dim_placeholder` 时，非 CEVA 配置保持旧的运行时重复口径；重新导出 sidecar 后消除（K12）。
- LOOP/SYNC 周期仍是 assumption 占位值，需 CEVA 实测或标定后替换（Q10）。
- 验证报告字段契约是本轮定义的最小版本，需与独立校准产出方对齐（Q8）。
- 新增的产物哈希/形状/通信核对只覆盖 GEMM 形状范围；1e、多核与未测通信仍不能据此标为已校准。

第 3 轮（round03，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `./run.sh --test sim`，45/45 个测试文件通过，合计 900 条用例、1 条跳过、0 条失败（约 90 秒）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `python -m pytest tests/sim -q`，905 通过、1 跳过、0 失败（29.85 秒）。
- genesim 受影响文件：`uv run python tests/sim/test_compiler_placement.py`（100 条 OK）、`uv run python tests/sim/test_config_loader.py`（79 条 OK，日志中的 ERROR 全是校验用例的预期输出）。
- 五份 CEVA 场景复验：在临时放置 sidecar 下依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- 问题 1 真产物复现与修复核对：用 `/tmp/ceva-review-round03-placement.json`（真实 sidecar 改为 8 台 DPU 一对一映射）和补登记中转开销来源的配置执行 `uv run python src/main.py --config /tmp/ceva-round03-fixed-config.yaml`。修复前在展开后 op6867（VECTOR_MUL 分片）报缺少 `symbolic_dim_placeholder`；修复后跑完 1 个请求，`total_time_s=666.879261`、`tokens/s=1.192`、`result_maturity=model_prediction`，实际创建 64 核。轨迹核对：`op_6867_VECTOR_MUL.pim_trace`、`op_6988_VECTOR_MUL.pim_trace` 等分片为 `work_binding=static_shape`、`shape_scale={'Tq': 128}`；被融合吸收的 SILU 分片按设计登记 `runtime_tiles`、空轨迹。
- 问题 2/3 真产物复现与修复核对：用真实 IR、真实 trace 和真实 sidecar 构造报告。`dtype=int8, element_bytes=2` 从 `validated_in_scope` 变为 `model_prediction`（元素宽度不符）；`dtype=fp16` 且只有逐算子布局、没有验证证据时从 `validated_in_scope` 变为 `model_prediction`（缺独立验证证据）；补全 `dtype_layout`（`{GATHER:f16, GEMM:f16, GEMV_CONTEXT:i8, GEMV_SCORE:i8, MASK:f16, QUANT:i8, RMSNORM:f16, ROPE:f16, SILU:f16, SOFTMAX:f16, VECTOR_ADD:f16, VECTOR_MUL:f16}`，f16 覆盖 4514 个算子、i8 覆盖 2080 个）和逐点证据后恢复 `validated_in_scope`。
- 问题 4 复现与修复核对：64 核 4e 场景在临时放置 sidecar 下先确认 `validate(mode="simulation")` 为 True，删除 `comm_config.host_relay_overhead_ns` 来源后变为 False；五份场景均登记该来源。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `python scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1762.411`、`tokens/s=4.587`。与 round02 的 1865.332/4.334 不同，原因是 TP 展开出的 VECTOR_MUL 分片也继承了符号维占位值，改用静态形状缩放（问题 1 修复），属于预期。轨迹核对：448 条 GEMM 为 `runtime_tiles`、64 条被吸收 SILU 为 `runtime_tiles`、64 条 VECTOR_MUL 分片为 `static_shape`（`shape_scale={'Tq': 128}`），其余 6562 条为 `static_shape`。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round03 改动无直接关系。

仍存在的不足（round03 增补）：

- 精度布局按 op_type 汇总，同一 op_type 混用两种主计算类型的运行无法通过核对，只能保持 `model_prediction`（K13）。
- 验证证据的独立性由产出方声明的 `sample_set` 保证，程序只能核对非空与数值自洽（K14）。
- `dtype_layout` 与 `validation` 是新增的报告字段，需独立校准产出方按同一契约生成（Q8、Q12）。
- 主计算精度按“覆盖算子数最多的类型”判定，未按工作量加权；形状范围仍只核对 GEMM 的 M/N/K。

第 4 轮（round04，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `./run.sh --test sim`，45/45 个测试文件通过，合计 916 条用例、0 失败（约 90 秒）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `python -m pytest tests/sim -q`，921 通过、1 跳过、0 失败（31 秒）。
- genesim 受影响文件：`python -m pytest tests/sim/test_vpu.py tests/sim/test_compiler_placement.py tests/sim/test_pim_backend.py tests/sim/test_pimir_trace.py -q`，215 通过、0 失败。
- 问题 1 真产物复现与修复核对：用真实 `bpath-f265519387e62064.pimir.mlir` 在 Tq=1、KV=128 下，旧实现为 1.480044 μs、384 B；修复后运行时按实际形状重建轨迹，与按该形状直接编译的轨迹一致：2.623194 μs、16,640 B。全链路 10 个请求、1524 个 decode 步下，运行时改为与逐条指令求值同口径的解析周期计算；CEVA 冒烟结果与逐条求值完全一致（`total_time_s=670.469079`）。G 侧占位值生产端 `_bpath_pimir` 只核对，无需修改。
- 问题 2/3 复现与修复核对：旧报告在同一夹具的 `output_length=1/11/100` 下均返回 `validated_in_scope`，三个样本改成无关算子名后也仍返回该级别；修复后改输出长度、decode 步数或请求 trace 都保持 `model_prediction`；正向夹具使用 10 次 decode、11 个输出并返回 `validated_in_scope`；样本改为无关算子、单阶段或缺 DMA 时同样保持 `model_prediction`。
- 问题 4 复现与修复核对：`loop_count=10` 的实际周期为 12（begin 1 + MAC×10 + end 1），旧断言按 30 ns 且默认精度通过；修复后按 12 ns、15 位精度断言，并验证 `loop_count=20` 比 10 多 10 个周期；46 ns 与 140 ns 断言同样提高精度。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `python -m pytest tests/test_placement_export.py -q`，43 通过、0 失败。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `python scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`。与 round03 的 1762.411/4.587 不同，原因是整算子相位轨迹改为按实际形状逐操作数解析计算（问题 1 修复），属于预期；首次回归在 Tq=KV=726 的 GEMV_SCORE 上暴露 K 缓存朝向问题，已修复（I13）。
- CEVA-NPM 冒烟：在 `/data/fjg/flagOS/genesim` 执行 `uv run python src/main.py --config /tmp/ceva-review-round04-config.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射）。7138 条 trace 按新模板编译（最终代码重跑时全部复用），仿真成功结束，`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`；资源报告“创建 64、使用 8、空闲 56”。与 round03 的 666.879 s 不同，原因是相位轨迹改为按实际形状逐操作数重建，属于预期。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round04 改动无直接关系。

仍存在的不足（round04 增补）：

- 相位链中不对应算子输入/输出的中间张量仍按占位形状计字节，Tq 远小于 128 的 decode 请求会高估这些相（K15）。
- 验证报告的请求边界绑定依赖 `trace.path` 与二进制/JSON trace；没有请求 trace 的运行按设计只能保持 `model_prediction`。
- 验证样本的独立性仍由产出方声明的 `sample_set` 保证；程序新增的是产物关联与覆盖检查，不能证明样本来自独立验证集（K14）。
- 新增报告字段（`package.trace_sha256`、`scope.request_boundary`、样本的 `phase/kind/stage/shape/dtype`）需独立校准产出方按同一契约生成（Q8）。

第 5 轮（round05，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 935 条、0 失败（round04 同一口径为 915 条，本轮新增 20 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，941 通过、1 跳过、0 失败（32.66 秒）。
- genesim 受影响文件：`python -m pytest tests/sim/test_compiler_placement.py tests/sim/test_config_loader.py -q`，209 通过、0 失败。
- 通用预测器路径：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败，说明 CEVA 的后端限制没有改变通用配置的预测器能力。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向 `/tmp/ceva-review-round04-placement.json` 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- 问题 1/4 复现与修复核对：把 `/tmp/ceva-review-round04-config.yaml` 的 `scheduler.simulation_backend` 改为 `predictor` 或把 `scheduler.use_compiler_local_shapes` 改为 false 后，修复前正式校验返回 True，修复后两者都返回 False；直接调用 `GeneSimScheduler.run()` 也在进入仿真前拒绝预测器。
- 问题 2/3 复现与修复核对：用 `/tmp/ceva-review-round04-placement.json`（真实 6850 条产物、32 个吸收节点）做最小复现，修复前把 GEMM 条目指向 SILU 的 `pim.lut` 产物、或删掉 `fused_activations` 只留 5 个 `absorbed_ops` 都不报错；修复后分别报“计算原语与算子语义不一致”和“没有所属主算子及融合相位”。原 sidecar 不改动时加载通过，32 个吸收节点全部绑定到 gate_proj 主算子。
- 问题 5 复现与修复核对：把正向夹具的样本形状改为 `[999999,999999]` 或把相位改为 `unrelated_phase` 后，修复前仍为 `validated_in_scope`，修复后保持 `model_prediction`；正向夹具改用 `tt.dot` GEMM 产物、样本形状 `[1,768,256]` 和相位 `tt.dot` 后恢复 `validated_in_scope`。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `python -m pytest -p no:cacheprovider tests/test_placement_export.py -q`，43 通过、0 失败。本轮 G 未修改代码。
- 全链路：本轮未在当轮时限内跑完，未取得结果；第 6 轮已重新执行并通过，命令与结果见第 6 轮记录。
- CEVA-NPM 冒烟：在 `/data/fjg/flagOS/genesim` 执行 `python -u src/main.py --config /tmp/ceva-review-round04-config.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射；输出与轨迹目录分别为 `/tmp/ceva-review-round04-results`、`/tmp/ceva-review-round04-traces`）。新的加载核对在真实 sidecar（6850 条产物、32 个吸收节点）上通过，7138 条 trace 全部复用，仿真成功结束：`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`、`result_producer=pipeline_simulator`；资源报告“创建 64、使用 8、空闲 56”。结果与 round04 完全相同，说明本轮只增加校验，没有改变计时路径。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round05 改动无直接关系。

仍存在的不足（round05 增补）：

- SPLIT 视图产物只按元素总数核对，同元素总数但排布不同的错误产物不会被拦下（K16）。
- GEMM 的 A 路产物以指针传参，形状核对依赖分块参数和放置字节，没有逐操作数张量形状（K17）。
- 验证样本的相位名与形状口径是本轮新定义的契约，需独立校准产出方按同一口径生成（Q8、Q14）。
- `pim.target="pim:v1"` 是跨仓常量，编译器升级目标串时需要同步修改（Q13）。

第 6 轮（round06，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 948 条、1 条跳过、0 失败（round05 同一口径为 935 条，本轮新增 13 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，954 通过、1 跳过、0 失败（33.56 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`，139 条 OK、0 失败。
- 问题 1 复现与修复核对：按评审复现脚本，执行节点 `[128,64]` 的 SILU 产物 `tensor<2x128x64xf16>` 修复前被接受（元素量 16,384 对 8,192），修复后报“操作数形状 (2, 128, 64) 与执行节点输入 (128, 64) 不一致：前导轴 (2,) 不是广播维 1”；同末维不同前导轴 `[64,64]`、不可解析的 `tensor<?x...>` 均被拒绝；用例同时确认拒绝发生在 `_prepare_pim_traces()` 之前。本轮全链路新生成的 sidecar（6850 条算子记录、32 条融合、32 个吸收节点）与真实 IR 在新核对下仍全部通过加载，放置计划引用 8 个 DPU。
- 问题 2 复现与修复核对：同一份报告在 64 核与 128 核配置下修复前均为 `validated_in_scope`；修复后把报告改写为 64 核、已用 1 核，在 64 核配置下为 `validated_in_scope`，在 128 核配置下为 `model_prediction`；配置核数相同但报告已用核数改为 8 时同样为 `model_prediction`；实际创建核数与配置不符、报告缺核数字段时也保持 `model_prediction`。
- 问题 3 复现与修复核对：夹具请求为 1 个 prefill（seq_len=128）加 10 次 decode，`_ceva_run_shapes_of('SOFTMAX')` 修复后给出 prefill `{(128,128)}`、decode `{(1,129)…(1,138)}`；decode 样本用 `[128,128]` 占位形状时为 `model_prediction`，改用首步真实 KV 形状 `[1,129]` 后恢复 `validated_in_scope`；decode 样本改用 prefill 形状也保持 `model_prediction`。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_placement_export.py tests/test_genesim_bridge.py -q`，101 通过、1 跳过、0 失败（6 分 41 秒）。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04 记录完全一致，说明本轮 CEVA 分支的校验改动没有影响通用链路。新生成的放置 sidecar 为 6850 条算子记录、32 条融合、32 个吸收节点。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round06 改动无直接关系；本轮改动只落在 CEVA 分支，通用与 UPMEM 路径行为不变。

仍存在的不足（round06 增补）：

- 相位产物中与执行节点形状绑定不上的参数/表格/单行内核张量按产物自身形状计工作量，整份产物形状错误无法被逐轴核对拦下（K18）。
- 新增的 `scope.core_count`、`scope.used_core_count` 与运行期阶段形状口径属于验证报告契约，需独立校准产出方按同一口径生成（Q8、Q14）。
- 相位链中间张量仍按占位形状计的边界未在本轮改变（K15）。

第 7 轮（round07，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 958 条、1 条跳过、0 失败（round06 同一日志口径记录为 948 条，本轮新增 9 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，963 通过、1 跳过、0 失败（34.91 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`（146 条 OK）、`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_pimir_trace.py`（54 条 OK）。
- 问题 1 复现与修复核对：按评审探针，MEM_COPY 执行节点输入 `[128,64]`、产物操作数 `tensor<64x32xi8>` 时修复前 `wrong_input_shape_accepted=True`；修复后报“操作数形状 (64, 32) 与执行节点输入 (128, 64) 不一致：末维 32 与执行节点末维 64 不同”，直接调用 `_ceva_artifact_axis_error((64,32),(128,64))` 也返回该错误（`strict=False` 时为 `None`，仅供结果循环跳过中间结果）。结果正确但输入错误的 SILU 产物同样被拒绝；用例确认拒绝发生在 `_prepare_pim_traces()` 之前。真实放置 sidecar（6850 条算子记录）在 CEVA 配置下仍全部通过加载；输入侧唯一无法按位置绑定的是 GATHER 查表张量（2 个操作数对 1 个执行输入），按 K18 保留。
- 问题 2 复现与修复核对：修复前 `tensor<?x4096xf16>` 的 SPLIT 产物通过加载核对，`parse_phase_chain()` 得到 `in_bytes=0/out_bytes=0`，指令只剩一条 SYNC；修复后加载时直接报“含无法解析静态形状的张量（如动态维）”，`parse_phase_chain()` 对单相算子与带 phase_spec 的相位都抛 `PimirTraceError`。真实 SPLIT 产物（`32x128x128`、`128x4096`，元素总数均为 524,288）在 6850 条记录的加载核对中仍全部通过。
- 问题 3 复现与修复核对：逐点相对误差 `[0.001,0.002,0.003,0.004]` 时，报告 p90 填中位数 `0.0025` 或线性插值 `0.0037` 修复前均返回 `validated_in_scope`，修复后均保持 `model_prediction`；最近秩值 `0.004` 恢复 `validated_in_scope`。报告契约版本升到 2，旧版报告保持 `model_prediction`。
- 五份 CEVA 场景复验：在评审探针里把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建 PIM 核数分别为 1/64/64/128/128，GPU 均为 0。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round06 记录完全一致，说明本轮改动只影响 CEVA 校验与动态形状拒绝，没有改变通用链路的计时。新生成的放置 sidecar 为 6850 条算子记录、32 条融合、32 个吸收节点。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round07 改动无直接关系。

仍存在的不足（round07 增补）：

- 相位产物中与执行节点形状绑定不上的参数/表格/单行内核张量仍按产物自身形状计工作量（K18），只在操作数与输入个数一致时执行严格匹配。
- SPLIT 仍只核对输入元素总数，同元素总数但排布不同的错误产物不会被拦下（K16）。
- p90 最近秩口径与 `schema_version=2` 属于验证报告契约，需独立校准产出方按同一口径生成（Q8、Q12）。
- 相位链中间张量仍按占位形状计的边界未在本轮改变（K15）。

第 8 轮（round08，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 964 条、1 条跳过、0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，970 通过、1 跳过、0 失败（36.32 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`（153 条 OK，round07 为 146 条）；`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`（7/7 个测试文件通过）。
- 问题 1 复现与修复核对：修复前新增用例 `test_request_boundary_counts_all_decode_only_steps` 失败，decode-only 请求（`is_prefill=false`、`seq_len=128`、`output_length=10`）的边界为 `decode_steps=[9]`、`max_query_length=128`；修复后为 `decode_steps=[10]`、`max_query_length=1`。decode-only 二进制 trace 与 prefill+decode-only 混合请求用例通过，混合边界为 `decode_steps=[10,5]`、`max_query_length=128`。报告把 decode-only 的 `decode_steps` 写成 `[9]` 时保持 `model_prediction`，写回正确边界后恢复 `validated_in_scope`。
- 问题 2 复现与修复核对：修复前新增用例 `test_result_maturity_requires_active_core_evidence` 在 `op_to_vpu_id` 为空时返回 `validated_in_scope`；修复后返回 `model_prediction`。映射值全部不落在 PIM 核（`{0: -1}`）时同样保持 `model_prediction`；实际使用 2 个核而放置计划只用 1 个核时保持 `model_prediction`；正向夹具补 `op_to_vpu_id={0: 0}` 与计划一致后仍返回 `validated_in_scope`。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py tests/test_placement_export.py -q`，79 通过、4 跳过、0 失败。本轮 G 未修改代码。
- 全链路：（全链路结果待补）
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round08 改动无直接关系。

仍存在的不足（round08 增补）：

- decode-only 的 `decode_steps` 与 `max_query_length` 口径属于验证报告契约，需独立校准产出方按同一约定生成（Q8、Q14）。
- 实际使用核数按运行期放置映射静态统计；当前 CEVA 路径不做执行期动态重映射，暂无实际影响。
- round07 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约与相位链占位形状边界未在本轮改变（K15、K16、K18、Q8、Q12）。

第 9 轮（round09，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 972 条、1 条跳过、0 失败（round08 同一日志口径记录为 964 条、1 条跳过，本轮新增 8 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，978 通过、1 跳过、0 失败（36.61 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim/test_compiler_placement.py -q`，161 通过、0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败，说明 CEVA 产物核对没有改变通用配置的预测器能力。
- genesim UPMEM 检查入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败，UPMEM 路径行为不变。
- 问题 1 复现与修复核对：评审探针把同时含 `pim.lut` 与 `pim.matmul` 的产物登记给 SILU，修复前 `accepted=true`，相位解析同时得到查表 8192 个工作单元和 524288 次 MAC；修复后报“混入了与算子语义矛盾的计算原语 ['pim.matmul']”。新增用例同时确认拒绝发生在 `_prepare_pim_traces()` 之前。真实放置 sidecar 的 6850 条产物在新核对下全部通过，`_apply_compiler_local_shapes()` 与 `_load_compiler_placement()` 合计约 1 秒完成，映射 6626 条。
- 问题 2 复现与修复核对：评审探针的 SILU 执行节点输出 `[128,64]`，产物中间结果 `tensor<128x64xf16>`、最终返回 `tensor<128x32xf16>`，修复前 `accepted=true`；修复后报“最终结果形状 (128, 32) 与执行节点输出 (128, 64) 不一致：末维 32 与执行节点末维 64 不同”。无操作数 `tt.return` 的实产物按最后一条带结果的语句核对；多结果返回按位置对应执行节点输出，顺序不符时拒绝；写入类算子（`pim.kv_cache`，无结果张量）仍按操作数核对。
- 问题 3 复现与修复核对：两个真实 prefill 请求、其余字段匹配的完整报告，修复前 `max_inflight_requests` 为 1 与 10 都返回 `validated_in_scope`；修复后报告记 1 时，配置 1 返回 `validated_in_scope`、配置 10 返回 `model_prediction`；把报告改写为 10 后配置 10 恢复 `validated_in_scope`。报告缺少该字段或 `schema_version=2` 时均保持 `model_prediction`。
- 五份 CEVA 场景复验：在评审探针里把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建 PIM 核数分别为 1/64/64/128/128，GPU 均为 0。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，43 通过、0 失败。本轮 G 未修改代码。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 uv run python -u src/main.py --config /tmp/ceva-review-round09-smoke.yaml`，退出码 0。7138 条 trace 全部来自 `pimir` 且全部复用（0 编译），`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。结果与 round04/round05 及本轮评审记录完全一致，说明新增的产物核对只拦截错误产物，没有改变正确产物的计时。
- 全链路 `run_full_pipeline.py --num-stages 4` 本轮未重跑：改动只落在 `Scheduler._is_ceva()` 分支，通用与 UPMEM 路径行为不变；真实 sidecar 与 CEVA 冒烟已覆盖修改范围。评审本轮记录已复现通用产物（6850 条 sidecar、21 份唯一 PIM MLIR、7138 条 `pimir` trace、1815.611866 秒摘要）。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round09 改动无直接关系。

仍存在的不足（round09 增补）：

- 计算原语的允许集合是显式登记的跨仓契约：算子编译器新增融合相位时，需要同步更新 `CEVA_ARTIFACT_FUSED_PHASES`，否则会被拒绝（与 Q13 同类）。
- 无操作数 `tt.return` 的最终结果按“最后一条带结果的语句”取；若后续产物在最后追加无结果的辅助语句，仍会回退到上一条带结果的语句。
- 计时边界新增并发上限与 `schema_version=3` 属于验证报告契约，需独立校准产出方按同一口径生成（Q8）。
- round07～round08 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约与相位链占位形状边界未在本轮改变（K15、K16、K18、Q8、Q12）。

第 10 轮（round10，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 980 条、1 条跳过、0 失败（round09 同一日志口径记录为 972 条，本轮新增 7 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，985 通过、1 跳过、0 失败（41.42 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`，168 条 OK（round09 为 161 条）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败，说明产物语句契约与来源核对没有改变通用配置的预测器能力。
- genesim UPMEM 检查入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败，UPMEM 路径行为不变。
- 问题 1 复现与修复核对：按评审探针构造形状、目标、激活均正确的 SILU 产物并追加 `pim.kv_cache`，修复前新增用例 `test_ceva_refuses_an_unrelated_transfer_primitive` 失败（`RuntimeError not raised`），直接对同一份产物调用 `parse_phase_chain()` 得到 `activation 16384/16384` 与 `dma 16384/16384` 两相，证实无关搬运确实会进入计时；修复后加载时直接报“混入了与算子语义无关或未登记的原语 ['pim.kv_cache']”。新增正例 `test_ceva_accepts_gemm_infrastructure_statements` 确认带 barrier/dma/wram/convert_layout 的真实 GEMM 产物结构仍被接受。
- 问题 1 真实产物核对：统计 `.opcompiler_cache` 的 25,158 份 PIM MLIR，语句组合只有 17 种；`tt.dot` 产物固定为 `barrier/convert_layout/dma_load/dma_store/wram_alloc/wram_load/wram_store/tt.dot` 八条，其余节点没有基础设施语句。真实 sidecar（6850 条算子记录、21 份唯一 PIM MLIR，其中 5 份 GEMM）在 CEVA 配置下调用 `_apply_compiler_local_shapes()` 全部通过，注册 `pimir_paths=6850`、吸收节点 32 个，耗时 0.775 秒，合法搬运相位（`pim.kv_cache`、`pim.dma_load` 等）没有被误杀。
- 问题 2 复现与修复核对：修复前新增用例 `test_assumption_parameter_sources_keep_model_prediction`、`test_one_assumption_parameter_source_keeps_model_prediction`、`test_missing_parameter_source_keeps_model_prediction`、`test_missing_source_evidence_keeps_model_prediction`、`test_unconfirmed_parameter_source_keeps_model_prediction` 均失败（返回 `validated_in_scope`），与评审探针一致；修复后全部返回 `model_prediction`，既有正向 `test_validation_report_marks_the_result_validated` 在补齐完整实测来源台账后仍返回 `validated_in_scope`。评审烟雾配置（沿用 `ceva_single_core_4e` 的来源台账，容量、时钟、周期成本和互连等为 assumption）在真实 sidecar 下 `validate(mode="simulation")` 仍为 True，`_ceva_result_maturity()` 返回 `model_prediction`。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，`ceva_single_core_4e`、`ceva_64core_4e`、`ceva_64core_1e`、`ceva_128core_4e`、`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q` 后合计 79 通过、4 跳过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 26 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round07 记录完全一致，说明本轮改动只影响 CEVA 产物核对与成熟度派生，没有改变通用链路的计时。新生成的放置 sidecar 为 6850 条算子记录、21 份唯一 PIM MLIR、32 条融合、32 个吸收节点，与评审记录的产物规模一致。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round10 改动无直接关系。

仍存在的不足（round10 增补）：

- GEMM 产物允许的 barrier/DMA/WRAM 基础设施语句只按执行节点整体放行，没有逐条核对 DMA 字节（K19）；其他节点新增基础设施语句时需要同步扩展 `CEVA_ARTIFACT_SUPPORT_PRIMITIVES`（与 Q13 同类契约）。
- 来源等级只核对 origin 与证据非空，证据文字是否覆盖本次目标、形状和边界仍需人工或独立校准流程保证（Q15）。
- 当前五份 CEVA 场景的容量、时钟、周期成本、区间倍率和互连来源仍为 assumption，因此正式运行只能输出 `model_prediction`（Q1～Q4、Q10、Q15）。
- round07～round09 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约与相位链占位形状边界未在本轮改变（K15、K16、K18、Q8、Q12）。

第 11 轮（round11，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 987 条、1 条跳过、0 失败（round10 同一日志口径记录为 980 条，本轮新增 7 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，992 通过、1 跳过、0 失败（41.67 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`，175 条 OK（round10 为 168 条）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败，说明产物形状核对与校验逻辑没有改变通用配置的预测器能力。
- genesim UPMEM 检查入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败，UPMEM 路径行为不变。
- 问题 1 复现与修复核对：新增用例 `test_ceva_refuses_a_single_row_input_artifact`、`test_ceva_refuses_a_single_row_output_artifact`、`test_ceva_refuses_a_single_row_artifact_before_any_trace_is_generated` 修复前均失败（`RuntimeError not raised`）；对同一份 `1×64` SILU 产物调用 `parse_phase_chain()` 得到 `in_bytes=128`、`out_bytes=128`、`work_units=64`，而 `128×64` 产物为 `16,384 / 16,384 / 8,192`，与评审探针一致。修复后三种错产物都在 `_prepare_pim_traces()` 之前报“操作数/最终结果形状不一致”；登记后的 QUANT 单行内核与 GATHER 单行结果正例通过。真实放置 sidecar（6850 条算子记录、32 个吸收节点、21 份唯一 PIM MLIR）调用 `_apply_compiler_local_shapes()` 全部通过。
- 问题 2 复现与修复核对：新增用例 `test_validation_report_link_semantics_must_match_the_run` 修复前三个子用例（`is_full_duplex`、`intra_cluster_only`、`inter_cluster_only`）均返回 `validated_in_scope`；修复后三者都返回 `model_prediction`，链路字段一致的报告仍返回 `validated_in_scope`。直接探针：同一条 PIM→PIM 链路把 `inter_cluster_only` 与 `is_full_duplex` 由 True 改为 False 后，`_normalized_links()` 结果不同且 `_links_equal()` 返回 False（修复前两者都相同/True）。
- 问题 3 复现与修复核对：新增用例 `test_validation_report_non_positive_prediction_keeps_prediction` 覆盖预测 -100 与 0 两种样本（观测 100、误差与相对误差自洽、汇总按同口径重算），修复前均返回 `validated_in_scope`，修复后均保持 `model_prediction`；正向夹具（预测 99.95/50.02/199.90/80.04）仍返回 `validated_in_scope`。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，`ceva_single_core_4e`、`ceva_64core_4e`、`ceva_64core_1e`、`ceva_128core_4e`、`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 uv run python -u src/main.py --config /tmp/ceva-review-round11-smoke.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0。7138 条 trace 全部复用（0 编译），`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。结果与 round04/05/09 完全一致，说明新增的链路语义与预测值核对没有改变正确输入的计时。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py tests/test_genesim_bridge.py -q`，101 通过、1 跳过、0 失败（6 分 41 秒）。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round10 记录完全一致，说明本轮改动只影响 CEVA 产物门禁与验证报告核对，没有改变通用链路的计时。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round11 改动无直接关系。

仍存在的不足（round11 增补）：

- 单行内核视图的登记集合是跨仓契约（GATHER、QUANT）；算子编译器若给其他算子改用单行内核，需要同步扩展 `CEVA_SINGLE_ROW_KERNEL_OPS`，否则会被拒绝（与 Q13 同类）。
- 链路语义字段省略时按运行时缺省值解释，程序无法区分“显式声明缺省值”和“漏填”，只能保证语义比较不因漏填而误判为相同；产出方仍需按 Q8 完整登记。
- 预测值必须为正，当前没有合法的零周期样本；若后续引入零成本视图，需要按算子显式登记而不是放开全部样本。
- round07～round09 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约与相位链占位形状边界未在本轮改变（K15、K16、K18、Q8、Q12）。

第 12 轮（round12，2026-11-07）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1003 条、1 条跳过、0 失败（round11 同一日志口径记录为 987 条，本轮新增 16 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1008 通过、1 跳过、0 失败（46.34 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim/test_config_loader.py tests/sim/test_compiler_placement.py tests/sim/test_main.py tests/sim/test_pimir_trace.py -q -p no:cacheprovider`，343 通过、0 失败（28.68 秒）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败，说明新增的 CEVA 校验没有改变通用配置的预测器能力。
- genesim UPMEM 检查入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败，UPMEM 路径行为不变。
- 问题 1 复现与修复核对：修复前只改主机参数（基础延迟 10 µs → 1,000,000 µs、峰值算力 1 TFLOP/s → 2 TFLOP/s、带宽 100 GB/s → 200 GB/s）时，`_ceva_cost_model_summary()` 两次结果完全相等且不含 `cpu_model`，`_ceva_validation_scope_error()` 也不读取主机参数，评审探针在同一完整报告下复现为 `validated_in_scope`；修复后摘要包含 `cpu_model` 且两次不相等。新增用例逐项改坏基础延迟、算力、带宽和核数后均返回 `model_prediction`；报告缺 `cpu_model`、运行含主机编排节点但报告缺 `scope.host_orchestration`、主机算子集合不一致时同样保持 `model_prediction`，补齐覆盖 MODEL_INPUT 的 prefill/decode 样本后恢复 `validated_in_scope`。
- 问题 1 冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 uv run python -u src/main.py --config /tmp/ceva-review-round12-smoke.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0；7138 条 trace 全部复用（0 编译）。`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round11 完全一致。`summary.json` 新增主机编排单列指标：`host_orchestration_service_time_s=5.789119323847103`、`host_orchestration_memory_service_time_s=2.7480027955198816`、`host_orchestration_memory_bytes=274800279552.0`、`host_orchestration_op_count=2`；其中存储时间恰好等于 274,800,279,552 B ÷ 100 GB/s，与评审指出的缺省 CPUModel 计算一致，说明主机开销现在单独可见且可被报告绑定。
- 问题 2 复现与修复核对：32 B 的 `pim.kv_cache` 相位产物传 `max_bulk_transfer_bytes=16` 时，修复前 LOADN/STOREN 立即数为 `[32]`，修复后为 `[16,16]`；缺省上限仍为 `[32]`。`PIMTrace.estimate_cycle_breakdown()` 与 `phase_cycle_breakdown()` 在同一上限下的总周期完全一致；显式传入已解析相位程序的调度器路径也得到同一指令序列。`compile_signature` 早已包含该上限，缓存复用不会绕过分段口径。
- 问题 3 复现与修复核对：修复前逐份解析五份 CEVA 场景，`weight_deployment.mappings` 均含通用缺省合入的 `qkv_proj→gpu` 与 `ffn→gpu`；修复后五份均为 qkv_proj/attention/ffn 全 PIM，正式校验在出现任何非 PIM 映射时返回 False。实际 GPU 资源与执行算子为零的既有核对保持不变。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，`ceva_single_core_4e`、`ceva_64core_4e`、`ceva_64core_1e`、`ceva_128core_4e`、`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True，解析后的权重部署均无 GPU 项。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 28 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round07～round11 记录完全一致，说明主机指标、报告契约与相位分段修复没有改变通用链路的计时。新生成的放置 sidecar 为 6850 条算子记录、21 份唯一 PIM MLIR、32 条融合、32 个吸收节点，引用 8 个 DPU。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与 round12 改动无直接关系。

仍存在的不足（round12 增补）：

- 主机 CPU 参数（核数、基础延迟、峰值算力、带宽）仍为 `assumption`，需主机实测或甲方确认后才能升级为范围内已验证（Q16）。
- 主机编排证据的样本独立性仍由产出方声明的 `sample_set` 保证，程序只核对算子/阶段覆盖、数值自洽和成本参数绑定（K20）。
- 验证报告契约升到 `schema_version=5`，新增 `scope.cost_model.cpu_model` 与 `scope.host_orchestration`，需独立校准产出方按同一口径生成（Q8）。
- 真实 CEVA 冒烟没有验证报告，结果仍为 `model_prediction`；主机开销现在虽已单列，但没有独立主机实测数据可供对比。
- round07～round11 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合与链路语义漏填边界未在本轮改变（K15、K16、K18、Q8、Q12）。

第 13 轮（round13，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1011 条、1 条跳过、0 失败（round12 同一口径记录为 1003 条，本轮新增 8 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1016 通过、1 跳过、0 失败（46.86 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`（187 条 OK）、`tests/sim/test_collective_edges.py`（21 条 OK）、`tests/sim/test_cpu_backend.py`（10 条 OK）、`tests/sim/test_main.py`（18 条 OK）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败；UPMEM 检查入口 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向真实 sidecar 后，`ceva_single_core_4e`、`ceva_64core_4e`、`ceva_64core_1e`、`ceva_128core_4e`、`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- 问题 1 复现与修复核对：真实 7B 放置探针（`/tmp/ceva-review-round12-smoke.yaml` 同款配置，指向真实 IR、trace 与 8 DPU 一对一 sidecar）显示展开后的执行图有 CPU 主机节点 `MODEL_INPUT` 1、`MODEL_OUTPUT` 1、`ALL_REDUCE` 64、`COLLECTIVE_RELAY` 254。修复前 `_ceva_host_orchestration_op_types()` 只返回 `MODEL_INPUT`/`MODEL_OUTPUT`，主机指标 `op_count=2`；修复后返回四类，`op_counts={'ALL_REDUCE': 64, 'COLLECTIVE_RELAY': 254, 'MODEL_INPUT': 1, 'MODEL_OUTPUT': 1}`，`zero_cost_op_types=['COLLECTIVE_RELAY']`。单测中报告证据缺 `ALL_REDUCE` 时保持 `model_prediction`，覆盖 `MODEL_INPUT`/`MODEL_OUTPUT`/`ALL_REDUCE` 后恢复 `validated_in_scope`，把 `COLLECTIVE_RELAY` 列为实测样本则保持 `model_prediction`。
- 问题 2 复现与修复核对：修复前 `_ceva_cost_model_summary()` 不含 `multipliers`，同一配置把 `pim.tensor_pu_config.multipliers` 从 4 改为 1000 后两次摘要完全相等（探针输出 `multipliers in summary: False`、`equal: True`）；修复后摘要包含 `multipliers`，4 与 8 两种配置分别为 4.0/8.0，`PIMVPU.peak_flops_per_s()` 随倍数同比翻倍。报告缺 `multipliers`、仅改该参数或使用旧版 schema（2/4/5）均保持 `model_prediction`。
- 问题 3 复现与修复核对：修复前 CPUModel 对 `COLLECTIVE_RELAY` 返回 10 µs 基础延迟，GPUModel 返回 0；修复后 CPU 执行与剖面全为零。纯 PIM 联合用例确认中转节点落在 CPU、执行剖面为零，同时跨核依赖边的链路时延大于 0。
- CEVA-NPM 冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 uv run python -u src/main.py --config /tmp/ceva-review-round13-smoke.yaml`（round12 冒烟配置的副本，仅改结果目录；64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0，7138 条 trace 全部复用（0 编译）。`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round12 相同（PIM 是瓶颈，主机路径不在关键路径上）。`summary.json` 的主机编排指标：`host_orchestration_op_count=320`（round12 为 2）、`host_orchestration_op_counts={'ALL_REDUCE': 64, 'COLLECTIVE_RELAY': 254, 'MODEL_INPUT': 1, 'MODEL_OUTPUT': 1}`、`host_orchestration_zero_cost_op_types=['COLLECTIVE_RELAY']`；`host_orchestration_service_time_s=5.613859324285841`（round12 为 5.789119323847103，恰好减少 0.17526 s = 254 个中转 × 10 µs × 69 次前向），存储服务时间 2.7480027955198816 s 与字节数 274,800,279,552 B 不变。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0）；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round07～round12 记录完全一致，说明本轮主机指标与中转成本修复没有改变通用链路的计时。新生成的放置 sidecar 为 6850 条算子记录、21 份唯一 PIM MLIR、32 条融合、32 个吸收节点，引用 8 个 DPU。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，无需重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round13 增补）：

- 主机 CPU 参数仍为 `assumption`，需主机实测或甲方确认后才能升级为范围内已验证（Q16）。
- 主机证据的样本独立性仍由产出方声明的 `sample_set` 保证（K20）；合成中转不参与校准的约定需要独立校准产出方按 Q8 同步。
- 验证报告契约升到 `schema_version=6`，新增 `scope.cost_model.multipliers` 与主机证据算子集合的收窄语义，需独立校准产出方按同一口径生成（Q8）。
- 真实 CEVA 冒烟没有验证报告，结果仍为 `model_prediction`；主机逐类型指标虽已可核对，但没有独立主机实测数据可供对比。
- round07～round12 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义漏填与主机证据独立性边界未在本轮改变（K15、K16、K18、K20、Q8、Q12）。

第 14 轮（round14，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1027 条、0 失败（round13 同一口径为 1011 条，本轮新增 17 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1033 通过、1 跳过、0 失败（47.31 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_pimir_trace.py`（63 条 OK）、`tests/sim/test_vpu.py`（40 条 OK）、`tests/sim/test_compiler_placement.py`（193 条 OK）、`tests/sim/test_collective_edges.py`（22 条 OK）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败；UPMEM 检查入口 `PYTHONDONTWRITEBYTECODE=1 ./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 问题 1 复现与修复核对：真实产物 `.opcompiler_cache/cdb11ad10a30a818.pimir.mlir` 在旧代码上报“只支持一层或两层 scf.for……实际有 3 层”；同 tile 的两层循环输入 `mac_groups=16384` 报“超出 14 位立即数上限”。对算子编译缓存里全部 2139 份 GEMM 产物逐份调用旧解析器，662 份失败；修复后 2139/2139 解析成功且全部能编译成指令。真实产物解析得到 `m_extent=128`、`n_extent=32000`、`k_extent=4096`、`tile 8×512×32`、`mac_groups=16384`，按 M/N/K 分块计费 `16×63×128×16384 = 2,114,289,664` 次 MAC。新增用例覆盖三层循环、M+K 两层、全展开、非法步长、M/N/K 各加倍、16384 MAC 与 16384 K 迭代。
- 问题 2 复现与修复核对：旧放置 sidecar 顶层没有 `hardware`，5 份 GEMM 产物声明 `pim.mram-bytes=4294967296`，修复前 `_validate_ceva_gemm_artifact()` 返回成功；修复后同一份旧 sidecar 在 CEVA 冒烟配置下于 trace 生成前报 `op4（GEMM）的 PIM MLIR pim.mram-bytes=4294967296 B 与本次 CEVA 画像 8000000000 B 不一致`，退出码 1。
- 真实 CEVA 预算 sidecar 重新生成：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/export_pp_placement.py --num-stages 4 --mram-bytes 8000000000 --sys-reserve-bytes 0 --measure-kernel-tiles --out-dir /tmp/ceva-round14-export`，退出码 0。sidecar 为 `version=4`、`hardware={mram_bytes_per_dpu: 8000000000, sys_reserve_bytes: 0, usable_mram_bytes: 8000000000}`，6850 条算子记录、21 份唯一 PIM MLIR；5 份分块 GEMM 产物（被 224 条 GEMM 引用）全部声明 `pim.mram-bytes=8000000000`，16 份 B 路相位产物仍是默认硬件的 8 GiB（K21）。随后用 `scripts/refine_ir_with_flagtree.py --ir-level oplevel --fusion-from <sidecar>` 回写 32 个 `absorbed_ops`。
- CEVA 端到端冒烟（新 sidecar）：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 uv run python -u src/main.py --config /tmp/ceva-round14d-smoke.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0。7138 条 trace 全部按新产物重编（7074 条 `pimir`、64 条 `absorbed`）；`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。与 round04～round13 完全一致，说明按 CEVA 预算重新生成的产物被同一套计时路径消费。作为对照，同一 sidecar 未回写 `absorbed_ops` 时会多算 64 条 SILU 轨迹（`total_time_s=683.162018`），差异来自缺少融合回写而非预算核对。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败；`tests/test_genesim_bridge.py -q`，58 通过、1 跳过、0 失败（6 分 38 秒）。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 34 分钟，其中 GeneSim 仿真约 29 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round13 记录完全一致，说明 M/N/K 解析、混合进制循环和 sidecar 预算字段没有改变通用链路的计时。新生成的放置 sidecar 为 `version=4`（默认 4 GiB / 64 MiB 预算）、6850 条算子记录、21 份唯一 PIM MLIR、32 条融合、32 个吸收节点，引用 8 个 DPU。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round14 增补）：

- B 路相位产物仍按默认硬件生成 `pim.mram-bytes=8589934592`，本轮预算门禁只核对分块 GEMM 产物；需要时再扩展 `_bpath_pimir` 的导出契约（K21）。
- M 方向按 `ceil(Tq/tile_m)` 重复；编译分块大于运行 Tq 时只执行一个分块，与编译器分块语义一致，尾块不足部分不额外计费。
- `pim.tile-m == pim.tile-n` 且只剩一层循环的产物无法按步长区分 M/N，程序显式报错；当前 2139 份真实产物没有这种组合。
- round07～round13 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义漏填、主机证据独立性与 B 路预算边界未在本轮改变（K15、K16、K18、K20、K21、Q8、Q12）。

第 15 轮（round15，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1038 条、0 失败（约 90 秒）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1043 通过、1 跳过、0 失败（49.55 秒）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 uv run python tests/sim/test_compiler_placement.py`（198 条 OK）、`tests/sim/test_config_loader.py`（91 条 OK）、`tests/sim/test_vpu.py`（42 条 OK）。
- genesim 通用预测器入口：`./run.sh --test predictor`，7/7 个测试文件通过、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 问题 1 复现与修复核对：真实产物 `.opcompiler_cache/cdb11ad10a30a818.pimir.mlir`（tile 8×512×32、n_extent=32000）在修复前 `kernel_tile_n=32` 与缺失两种输入都能通过严格加载，运行时按 conf 的 32 得到 1000 次 N 循环，按产物分块应为 63 次。修复后两种输入都在 `_apply_compiler_local_shapes()` 的产物登记阶段被拒（早于 trace 生成）：`kernel_tile_n=32` 报“与 PIM MLIR pim.tile-n=512 不一致”，缺失报“缺少正整数 kernel_tile_n”。新增合成与真实大 M 产物两份 VPU 端到端用例：`_tile_count=63`、M 方向重复 16 次，MAC 指令数与 DMA 字节与 `_rollup` 展开结果完全一致，执行剖面 DMA 字节再乘 M 分块重复次数；运行时把分块换成 32 会在执行前报错。
- 问题 1 真实 sidecar 核对：`/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 的 224 条 GEMM 产物逐条核对 `kernel_tile_n == pim.tile-n` 全部一致；CEVA 冒烟将该 sidecar 的 6850 条算子记录全部通过新门禁，7138 条 trace 中 448 条 GEMM 的 `pimir.tile_n` 均为 512。
- 问题 2 复现与修复核对：修复前同一份完整验证报告在 `pim.inter_cluster_communication=host_relay` 与 `direct` 下均返回 `validated_in_scope`；修复后报告为 `direct` 时仅 `direct` 运行升级，`host_relay` 运行保持 `model_prediction`，报告改写为 `host_relay` 后结论对调；删除报告路由字段、或路由一致但链路集合不符时也保持 `model_prediction`。
- 五份 CEVA 场景复验：在真实放置 sidecar 与真实 IR/trace 下依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；五份场景均登记 `pim.inter_cluster_communication` 的假设来源，正式校验只接受 `direct`/`host_relay`。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round15b-smoke.yaml`（round15 评审冒烟配置补登路由来源），退出码 0。7138 条 trace 全部重编（7138 compiled、0 reused），`total_time_s=670.469079`、`tokens/s=1.186`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。与 round14/round15 评审记录完全一致，说明新增门禁与运行时复核没有改变真实产物链路的计时。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败（合计 81 通过、4 跳过）；`tests/test_genesim_bridge.py -q`，58 通过、1 跳过、0 失败（6 分 37 秒）。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `python scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round14 记录完全一致，说明本轮 CEVA 分块门禁与路由模式校验没有改变通用链路的计时。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round15 增补）：

- `kernel_tile_n` 门禁只覆盖带可读 PIM MLIR 的 GEMM 条目；缺少产物本身的节点仍由 `_prepare_pim_traces()` 在生成 trace 前统一拒绝，两条路径都在执行前失败。
- CEVA 运行时复核依赖轨迹元数据 `pimir.tile_n`；非 CEVA 或手写模板轨迹保持原兼容口径，不做该核对。
- 验证报告新增的路由模式字段属于契约变化（`schema_version=7`），需独立校准产出方按同一口径生成（Q8）。
- round07～round14 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界未在本轮改变（K15、K16、K18、K20、K21、Q8、Q12）。

第 16 轮（round16，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1048 条、1 条跳过、0 失败（本轮新增 11 条边界、编码与兼容性用例）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1054 通过、1 跳过、0 失败（49.36 秒）。
- genesim 受影响文件：`uv run python tests/sim/test_pim_trace_loader.py`（8 条 OK）、`tests/sim/test_pimir_trace.py`（64 条 OK）、`tests/sim/test_pim_compiler.py`（25 条 OK）、`tests/sim/test_pim_backend.py`（18 条 OK）、`tests/sim/test_vpu.py`（46 条 OK），合计 161 条、0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、86 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 问题 1 复现与修复核对：
  - 修复前按评审探针 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python /tmp/ceva_review_round16_loop_probe.py` 复现：`{'required_n_iterations': 5000, 'actual_n_iterations': 4095, 'expected_latency_s': 0.000327005, 'actual_latency_s': 0.000267818, 'actual_over_expected': 0.8190027675417807}`。
  - 修复后同一探针输出：`{'required_n_iterations': 5000, 'actual_n_iterations': 5000, 'expected_latency_s': 0.000327005, 'actual_latency_s': 0.000327005, 'actual_over_expected': 1.0}`；`execute_profile()` 的 `latency_s=0.000327005`、`memory_bytes=650000`，与执行路径一致。
  - 边界用例覆盖 4095/4096/5000：GEMM 的 N 循环、GEMV_SCORE 的 KV 循环、SOFTMAX 的 KV 循环均按完整次数增长，MAC 次数、DMA 字节与总周期同步断言；长循环两半超过 12 位、总次数超过 24 位均显式报错；通用（非 CEVA）后端对 5000 次仍保持 4095 截断（探针输出 `4095->4095 65520`、`4096->4095 65520`、`5000->4095 65520`，并打印告警）。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round16-smoke.yaml`（round15 冒烟配置副本，仅改结果/轨迹目录；64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0。首轮 7138 条 trace 全部重编（`schema_version=10`），重跑 0 编译、7138 复用；`total_time_s=670.4690789662563`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round15 的 `670.469079` 完全一致，说明长循环编码切换没有改变真实模型（N 循环 63 次）的计时。
- 轨迹编码核对：冒烟生成的 7138 条 trace 中，448 条运行时绑定轨迹全部使用 `LOOP_BEGIN_INDIRECT_LONG`，短间接循环为 0；其余 6690 条为固定循环的静态相位/吸收轨迹。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round15 记录完全一致，说明长循环编码切换与 trace schema 升级没有改变通用链路（`use_long_loop_counts=false`，短循环）的计时。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round16 增补）：

- 通用 `use_long_loop_counts=false` 配置仍保留 12 位兼容截断（仅告警），超过 4095 次的通用场景需要显式打开该配置；这是本轮“不改变通用 sim.yaml/UPMEM 既有行为”约束下的取舍（K22、Q17）。
- 24 位长循环上限为 16,777,215 次；当前 7B 模型的 N 循环 63 次、KV 循环远低于上限，本轮没有真实硬件计数器可对照，只能证明模型内部工作量按完整次数增长。
- GEMV/SOFTMAX 的生产轨迹当前走静态相位轨迹（固定循环），4095 边界用例覆盖的是模板/手写轨迹路径；真实 7B 模型没有超过 4095 次的 KV 循环，冒烟只证明编码切换不改变现有计时。
- round07～round15 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界未在本轮改变（K15、K16、K18、K20、K21、Q8、Q12）。

第 17 轮（round17，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，45/45 个测试文件通过，单文件 `Ran N tests` 合计 1053 条、1 条跳过、0 失败（约 90 秒）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1058 通过、1 跳过、0 失败（49.72 秒；round16 同一口径为 1054 通过、1 跳过，本轮新增 4 条）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python tests/sim/test_config_loader.py`（93 条 OK）、`tests/sim/test_main.py`（20 条 OK）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 问题 1 复现与修复核对：
  - 修复前新增反例 `test_simulation_requires_cpu_host_links` 在分别删除 `CPU_to_HOST`、`HOST_to_CPU` 后，`validate(mode="simulation")` 均返回 True（两条子用例失败）；修复后均返回 False，且 `mode="structure"` 仍返回 True。
  - 重跑评审探针 `/tmp/ceva_round17_boundary_probe.py`（真实 `conf/ceva_64core_4e.yaml` 与真实放置 sidecar）：修复前输出 `NO_CPU_LINKS {'simulation_valid': True, 'cpu_to_pim_latency': inf, 'pim_to_cpu_latency': inf}`；修复后为 `simulation_valid=False`，日志明确报告缺少 `CPU→HOST` 与 `HOST→CPU`。
  - 合法配置的中转成本断言 `test_valid_profile_has_a_finite_cpu_pim_route` 通过：`CommLib.transfer("CPU","PIM",1)` 与反向传输均小于无穷大。
  - 五份仓库 CEVA 场景在真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下复验：`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建核数分别为 1/64/64/128/128，GPU 均为 0。
- 问题 2 复现与修复核对：
  - 修复前 `test_generic_summary_omits_result_maturity` 失败，通用摘要实际生成 `"result_maturity": null`；修复后通用摘要不含该键。
  - `test_ceva_summary_keeps_result_maturity` 覆盖 `model_prediction` 与 `validated_in_scope`，修复后两种取值均保留。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round17-smoke.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射，评审 round17 使用的冒烟配置），退出码 0。7138 条 trace 全部复用（0 编译、7138 复用，说明本轮配置/摘要改动不影响编译签名），`total_time_s=670.4690789662563`、`throughput_tokens_per_s=1.1857`、`simulation_backend=pipeline`，与 round16/review round17 的 `670.469079` 完全一致；资源报告“创建 64、使用 8、空闲 56”；`summary.json` 仍包含 `result_maturity=model_prediction`，证明通用摘要条件写入没有影响 CEVA 摘要。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round16 记录完全一致，说明本轮校验与摘要改动没有改变通用链路的计时。
- 通用摘要核对：全流程重写 `genesim/results/summary.json` 后，`result_maturity` 键不再出现（`result_maturity_present=False`），同一份结果文件的 `total_time_s=1815.611866053922`、`tokens/s=4.4525` 与历史一致。重跑 `/tmp/ceva_round17_verify.py` 同时复验：五份 CEVA 场景 `validate(mode="structure")`/`validate(mode="simulation")` 全部 True，实际创建核数 1/64/64/128/128、GPU=0；放置 sidecar 为 version 4、6850 条算子记录、21 份唯一 PIM MLIR、缺失与哈希错误均为 0；通用 trace 为 7074 条 `pimir` + 64 条 `absorbed`、0 条 `template`。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮评审只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round17 增补）：

- CPU↔HOST 链路校验只核对两条链路存在，并只以测试断言中转成本有限；不核对链路带宽、延迟数值是否为甲方确认值。五份场景的 CPU 主机链路参数仍为 `assumption`（Q4）。
- 通用摘要键集合按“结果是否包含 `result_maturity`”恢复；若后续其他非 CEVA 路径也写入该键，需要同步约定其语义。
- round07～round16 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断未在本轮改变（K15、K16、K18、K20、K21、K22、Q8、Q12、Q17）。

第 18 轮（round18，2026-11-08）：

- 问题确认探针 `/tmp/round18_probe.py`（真实 64 核冒烟配置）修复前输出：`P1 validate(simulation) = True`（成本 sidecar 路径不存在）；`P3 capacity=inf/-inf validate(structure) raised OverflowError` 且 `resolve raised OverflowError`；round18 冒烟结果目录没有 `run_manifest.json`，摘要也不含清单字段。修复后同一探针输出：`P1 validate(simulation) = False`；`P3` 两种取值在结构校验下均返回 False，容量解析抛出 `ValueError: pim.cluster_memory_capacity_gb must be finite and positive.`。探针的 P2 检查固定指向旧 round18 结果目录，清单生成改用本轮 `/tmp/ceva-round19-smoke-results` 核对（见下文）。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1069 条、1 条跳过、0 失败（本轮新增一个测试文件）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1075 通过、1 跳过、0 失败（50.98 秒；round17 同一口径为 1058 通过、1 跳过，本轮新增 17 条）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python tests/sim/test_config_loader.py`（96 条 OK）、`tests/sim/test_pim_capacity.py`（17 条 OK）、`tests/sim/test_cost_sidecar.py`（15 条 OK）、`tests/sim/test_run_manifest.py`（8 条 OK）、`tests/sim/test_main.py`（22 条 OK）。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、86 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 问题 1 复现与修复核对：
  - 修复前 `test_simulation_rejects_a_missing_cost_sidecar` 失败（断言 `True is not false`），`test_ceva_missing_cost_sidecar_raises` 失败（`RuntimeError not raised`）；修复后均通过。
  - 通用路径保持原告警行为：`test_non_ceva_missing_cost_sidecar_still_warns` 断言返回 0 且日志包含 `cost_sidecar_file not found`。
  - 合法成本 sidecar 的正向断言 `test_ceva_cost_sidecar_annotates_the_real_operators` 通过：两个算子的 `source_name` 与 `mram_traffic_bytes` 均落到 `Operator` 上。
- 问题 2 复现与修复核对：
  - 修复前同一探针确认结果目录无清单；修复后 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round19-smoke.yaml`（round18 冒烟配置副本，仅改结果/轨迹目录）退出码 0。首轮 7138 条 trace 全部重编并写出清单；用最终代码重跑为 0 编译、7138 复用，`total_time_s=670.4690789662563`、`throughput_tokens_per_s=1.1857370085220755`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round17/review round18 的 `670.469079` 完全一致。
  - 清单内容核对：`/tmp/ceva-round19-smoke-results/run_manifest.json` 的 `schema_version=1`；`inputs` 记录 `model_ir`/`placement_sidecar`/`request_trace` 三个文件的绝对路径与 SHA256（本轮冒烟 `cost_sidecar_file` 留空，故不写成本输入）；`pim_mlir` 列出 21 份唯一产物的路径与 SHA256；`hardware` 记录 64 核、每核 4 engine、0 GPU、8,000,000,000 B 物理量与 0 B 预留；`compiler_identity` 记录图编译器 `/data/fjg/flagOS/flagos-pim-compiler` 提交 `f918586c9ff5…`、FlagTree 源码 `/data/fjg/flagOS/flagOS-installers/FlagTree` 提交 `0d7273bac131…`、实际 `triton-opt` 路径与 SHA256 `ecd955ec25e6…`；`resolved_config` 为合并默认值后的完整配置；`summary.json` 含 `run_manifest: "run_manifest.json"`。
  - 清单与实际文件一致：外部复算 `inputs` 三个文件、`config_file` 与 21 份 PIM MLIR 的 SHA256，全部与清单记录一致；真实 sidecar 的工具链身份先用 `validate_ceva_toolchain("/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json")` 复验通过。
- 问题 3 复现与修复核对：
  - 修复前 `test_ceva_capacity_rejects_non_finite_gib` 以 `OverflowError` 报错、`test_non_finite_capacity_is_rejected_without_crashing` 同样在结构校验阶段抛 `OverflowError`；修复后前者得到带 `cluster_memory_capacity_gb` 字段名的 `ValueError`，后者对 ±inf 与 NaN 在 `mode="structure"`/`mode="simulation"` 下均返回 False。
  - `--dry-run` 实跑核对：把冒烟配置的 `pim.cluster_memory_capacity_gb` 改为 `inf` 后执行 `uv run python src/main.py --config /tmp/ceva-round19-inf.yaml --dry-run`，退出码 1，日志为 `pim.cluster_memory_capacity_gb 必须是有限数值，收到 inf。`，无 `Traceback`/`OverflowError`；同一配置未改容量时 `--dry-run` 退出码 0 并输出 `structural_only`。
- 五份仓库 CEVA 场景复验：重跑评审脚本 `/tmp/ceva_round18_verify.py`，在真实放置 sidecar 下 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建核数 1/64/64/128/128，GPU 均为 0；sidecar 6850 条记录、21 份唯一 PIM MLIR、缺失与哈希错误均为 0，32 个吸收节点。
- 通用链核对：同一评审脚本复验通用 trace 为 7074 条 `pimir` + 64 条 `absorbed`、schema 10，通用摘要 `total_time_s=1815.611866053922`、`result_maturity` 键不存在。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 27 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round17 记录完全一致，说明本轮配置校验、成本 sidecar 与运行清单改动没有改变通用链路的计时。通用结果目录（`/tmp/full_pipeline_pimir` 与 `genesim/results`）均未生成 `run_manifest.json`，摘要也不含 `result_maturity`/`run_manifest` 键，确认清单只作用于 CEVA 路径。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只读取 FlagTree 源码提交号与已安装 `triton-opt` 指纹，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round18 增补）：

- 运行清单的图编译器版本取自 PIM MLIR 所在源码仓库的当前 HEAD 与脏标记，不是产物生成时嵌入的生产端身份；sidecar 导出后若源码再变化，清单无法精确证明“该产物由该提交编译”（K23、Q18）。
- CEVA 正式评估要求 PIM MLIR 位于 git 工作区内；复制到非 git 目录的算子编译缓存会被当作“版本缺失”拒绝（K24）。
- 运行清单只覆盖 CEVA 路径；通用/UPMEM 结果目录不生成清单，这是为保持既有行为不变的取舍。
- 清单只记录文件路径与哈希，不复制输入产物本身；输入文件被删除或移动后无法仅凭结果目录重放。
- round07～round17 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断未在本轮改变（K15、K16、K18、K20、K21、K22、Q8、Q12、Q17）。

第 19 轮（round19，2026-11-08）：

- 问题 1 确认（真实 Llama2-7B TP2 产物）：
  - 用 `models/llama2_7b_pimir.ir` 与 `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 把 288 个 TP 算子展开为 640 个分片（其中 576 个分片钉到 DPU）；展开前 GEMM 的 `(N,K)` 为 `(4096,4096)`、`(4096,11008)`、`(11008,4096)`，展开后的执行图只有 `(2048,4096)`、`(4096,2048)`、`(5504,4096)`、`(4096,5504)`。
  - 修复前按 `model_ir` 统计的运行范围是 `m/n/k=(1,2338)/(4096,11008)/(4096,11008)`，`_ceva_sample_shape_error("GEMM",[726,11008,11008],"prefill")` 返回 None；真实本地样本 `[726,2048,4096]` 因低于全局下界被拒。修复后运行范围为 `(1,2338)/(2048,5504)/(2048,5504)`，全局 11008 样本返回“超出本次运行范围：n=[2048,5504]”，四个真实本地分片形状均返回 None（接受）；`SILU` 的 prefill/decode 形状从 `(726,11008)/(1,11008)` 变为 `(726,5504)/(1,5504)`。
  - 新增三条用例修复前分别失败：范围断言 `None != {...}`、全局样本断言 `validated_in_scope != model_prediction`、本地样本断言 `model_prediction != validated_in_scope`；修复后三条全部通过，成熟度派生走完整验证报告核对，不只测试辅助函数。
- 问题 2 确认与端到端核对：
  - 修复前探针：`GENESIM_PIM_CLOCK_PERIOD_NS=1.26` 时 `get()` 为 1.26、`resolved_config()` 为 0.63；`GENESIM_SCHEDULER_PREFILL_DECODE_OVERLAP=true` 时 `get()` 为 True、`resolved_config()` 为 False。新增用例修复前在 `resolved["pim"]["clock_period_ns"]` 上失败（0.63 != 1.26），修复后通过。
  - 环境覆盖实跑：`GENESIM_PIM_CLOCK_PERIOD_NS=1.26 GENESIM_TRACE_PATH=/tmp/round19-trace-copy/llama2_7b.trace PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round19-smoke.yaml` 退出码 0；`summary.json` 的 `total_time_s=1334.9978145016437`、`tokens/s=0.595506592867925`、`result_maturity=model_prediction`；`run_manifest.json` 的 `resolved_config.pim.clock_period_ns=1.26`、`resolved_config.trace.path` 与 `inputs.request_trace.path` 均为 `/tmp/round19-trace-copy/llama2_7b.trace`，`hardware.clusters_per_device=64`。
  - 无环境覆盖重跑同一配置：`total_time_s=670.4690789662563`、`tokens/s=1.1857370085220755`，清单记录 `0.63`，与 round18 基线一致；两次运行的 4 个输入文件与 21 份 PIM MLIR 的 SHA256 外部复算全部一致。清单 `cost_model` 含 `chunk_size`/`cpu_model`/`cycle_costs`/`mac_width`/`max_bulk_transfer_bytes`/`multipliers`/`tile_size`/`zone_cycle_factors`，与本次仿真成本摘要同源。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1074 条、1 条跳过、0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1079 通过、1 跳过、0 失败（51.57 秒；round18 同一口径为 1075 通过、1 跳过，本轮新增 4 条）。
- genesim 受影响文件：`PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim/test_compiler_placement.py -q`（201 通过）、`tests/sim/test_run_manifest.py`（9 通过）、`tests/sim/test_config_loader.py`（97 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 五份仓库 CEVA 场景复验：`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True，实际创建核数 1/64/64/128/128、GPU 均为 0；真实放置 sidecar 6850 条记录、21 份唯一 PIM MLIR、缺失与哈希错误均为 0，32 个融合与 32 个吸收节点。
- 通用链核对：全链路重跑后通用 trace 为 7074 条 `pimir` + 64 条 `absorbed`、schema 10，通用摘要 `total_time_s=1815.611866053922`、`tokens/s=4.452493482304611`、`result_maturity` 键不存在。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0（约 26 分钟）。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1815.612`、`tokens/s=4.452`，与 round04～round18 记录完全一致，说明本轮形状范围与清单配置改动没有改变通用链路的计时。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；已安装的 `/data/fjg/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/bin/triton-opt` 仍在位，本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round19 增补）：

- 验证样本的 GEMM 形状仍按本次运行各轴的 `min/max` 包络核对，不是逐分片组合白名单：例如 `N=3000` 落在 `[2048,5504]` 内、但不对应任何真实分片组合的样本仍可通过。逐组合匹配需要验证报告契约增加分片标识或允许集合，属设计外扩展，本轮未做（与 Q14 一起确认）。
- `resolved_config()` 只应用已加载配置文件中存在的键路径上的 `GENESIM_*` 覆盖；运行时通过 `get(key, default)` 读取但配置文件未声明的键，其环境覆盖不会出现在清单中。若需要记录任意环境变量，应在清单里单独登记覆盖列表。
- 预测器后端不设置 `_execution_ir`，三个形状入口回退到展开前的 `model_ir`；但预测器结果因后端不是 `pipeline` 本来就无法升级为 `validated_in_scope`，不影响本轮修复的验证门禁。
- round07～round18 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断、运行清单工具链身份反推等未在本轮改变（K15、K16、K18、K20、K21、K22、K23、K24、Q8、Q12、Q17、Q18）。

第 20 轮（round20，2026-11-08）：

- 问题 1 确认（真实 B 路产物）：
  - 修复前用真实 `bpath-31471efb9ea37b19.pimir.mlir`（QUANT）与 `bpath-90a38eac88f36ccb.pimir.mlir`（GATHER，逐字嵌入测试）生成模板，`dimensions=[]`、所有张量 `axes` 为 None（Tq=1/2/128 探测一致）；Tq=2 的 QUANT 仍按 20,864 B 计，GATHER 的索引与结果字节恒定。
  - 新增用例先失败后通过：模板层 `test_single_row_kernel_repeats_the_row_view_tensors`、`test_gather_repeat_keeps_the_lookup_table_constant`、`test_row_view_tensor_must_start_with_one_row`；VPU 执行层 `test_single_row_quant_kernel_bills_every_runtime_row`、`test_single_row_gather_kernel_keeps_the_table_constant`（Tq=1/2/128，断言字节、计算量与周期增长，并逐相复算确认固定 SYNC 只计一次）；调度器层 `test_quant_template_binds_every_row_view_tensor`、`test_gather_template_keeps_the_table_outside_the_row_axis`（走完整 `_prepare_pim_traces()`）。
  - 真实冒烟轨迹核对：`/tmp/ceva-round20-smoke-traces/op_1072_QUANT.pim_trace` 与 `op_1_GATHER.pim_trace` 均带 `dimensions=['Tq']`、`dimension_sources={'Tq': ['in',0,0]}`；Tq=1/2/726 时 QUANT 为 20,864/41,728/15,147,264 B、101,288/202,501/73,539,408 周期，GATHER 为 1,056,772/1,064,968/6,998,872 B（查表 1,048,576 B 恒定，索引 4 B/行、结果 8,192 B/行）。
- 问题 2 确认：修复前成本 sidecar 内容不同或未配置却声明已使用时仍返回 `validated_in_scope`；修复后报告缺少 `package.cost_sidecar`、`used=false` 但本次配置了成本 sidecar、哈希不符、未配置却 `used=true` 均保持 `model_prediction`，哈希正确时恢复 `validated_in_scope`。旧版 `schema_version`（2/4/5/6/7）不再升级；正向夹具升级到 8 并显式登记 `used=false`。运行清单 `inputs.cost_sidecar` 与成熟度门禁共用 `run_manifest.cost_sidecar_entry()`（路径 + SHA256）。
- 问题 3 确认：时钟、周期成本、通信参数三个环境覆盖反例修复前均返回 `validated_in_scope`；修复后来源台账未登记匹配 `value` 时保持 `model_prediction`，登记后恢复。周期成本的逐项覆盖此前不会进入 `get("pim.pim_trace_cycle_costs")` 的返回值，本轮一并修正为与 `resolved_config()` 同源。运行清单 `env_overrides` 逐项记录 `env/value/configured_value/source_value/evidence_applies`。
- 覆盖冒烟：在 `/data/fjg/flagOS/genesim` 执行 `GENESIM_PIM_CLOCK_PERIOD_NS=1.26 PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round20-override.yaml`（round19 冒烟配置副本，仅改结果/轨迹目录），退出码 0；7138 条 trace 全部重编（schema 11），`total_time_s=1337.959593`、`tokens/s=0.594`、`result_maturity=model_prediction`。`run_manifest.json` 的 `env_overrides.pim.clock_period_ns` 为 `{env: GENESIM_PIM_CLOCK_PERIOD_NS, value: 1.26, configured_value: 0.63, source_value: null, evidence_applies: false}`，`resolved_config.pim.clock_period_ns=1.26`；作为对照，本轮无环境覆盖的 CEVA 冒烟 `run_manifest.json` 的 `env_overrides` 为空、`resolved_config.pim.clock_period_ns=0.63`。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests`/`Ran 1 test` 合计 1089 条、0 失败（round19 同一日志口径为 1074 条，本轮新增 15 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1094 通过、1 跳过、0 失败（54.76 秒；round19 同一口径为 1079 通过、1 跳过，本轮新增 15 条）。
- genesim 受影响文件：`test_compiler_placement.py`（209 通过）、`test_run_manifest.py`（11 通过）、`test_config_loader.py`（96 通过）、`test_pimir_trace.py`（67 通过）、`test_vpu.py`（48 通过）、`test_cost_sidecar.py`（15 通过）、`test_main.py`（22 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 五份仓库 CEVA 场景复验：真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建核数 `ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 分别为 1/64/64/128/128，GPU 均为 0。
- CEVA 端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round20-smoke.yaml`（64 核画像、1 个请求、8 台 DPU 一对一映射），退出码 0。7138 条 trace 全部按 schema 11 重编（7138 compiled、0 reused），`total_time_s=671.949968`、`tokens/s=1.183`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。与 round19 的 670.469079 s 相差 +1.480889 s，恰好等于 32 个 QUANT 算子各多计 725 行（725 × 101,287.8 周期 × 0.63 ns × 32 ≈ 1.4805 s）加 1 个 GATHER 算子的 0.000377 s，属于问题 1 修复的预期变化。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q -p no:cacheprovider`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败；`tests/test_genesim_bridge.py -q`，58 通过、1 跳过、0 失败（6 分 38 秒）。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部按 schema 11 重编（元数据核对为 7074 条 `pimir` + 64 条 `absorbed`、`template` 为 0；流水线汇总打印 `{'pimir': 7138}`），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1817.367954`、`tokens/s=4.448`。与 round19 的 1815.611866 s 相差 +1.756088 s（0.097%）：10 个请求各有一次 prefill，按请求长度逐条复算的单行内核增量为 13.382989 s（32 个 QUANT 13.379585 s + GATHER 0.003403 s），这些工作量分布在 8 台 DPU 并与各请求的 prefill/decode 重叠，流水线关键路径只体现其中一部分；CEVA 单请求冒烟（同一模型与 trace、726 token）给出完整的 +1.480889 s，与逐条轨迹复算一致。新生成的放置 sidecar 为 `version=4`（默认 4 GiB / 64 MiB 预算）、6850 条算子记录、21 份唯一 PIM MLIR、32 条融合、32 个吸收节点，引用 8 个 DPU。通用 trace 中 32 条 QUANT 与 1 条 GATHER 均带 `dimensions=['Tq']`、`dimension_sources={'Tq': ['in',0,0]}`，证明单行内核契约也进入通用链路。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round20 增补）：

- 单行内核的行视图登记是跨仓契约：QUANT 的全部操作数与结果、GATHER 的索引与结果随行数展开，GATHER 查表操作数恒定；算子编译器若改为多行内核或调换操作数顺序，需要同步修改 `CEVA_SINGLE_ROW_KERNEL_CONTRACT`，否则行视图首维不是 1 会显式报错（Q19）。
- GATHER 查表与 RMSNORM 参数等恒定张量仍按产物自身形状计，不随运行形状缩放（K18）；GATHER 查表产物按 Tq 占位当 vocab 的形态由算子编译产物决定，不在本轮范围。
- 验证报告契约升到 `schema_version=8`，新增 `package.cost_sidecar` 与来源 `value` 字段，需独立校准产出方按同一约定生成（Q8、Q15）。
- 成本 sidecar 采用内容哈希绑定；换到内容相同的另一路径仍视为同一成本输入，清单记录实际路径与哈希（round20 取舍）。
- `resolved_config()` 仍只包含配置树中已声明的键；配置树未声明但来源台账登记的覆盖由清单 `env_overrides` 单列（round19 记录的这一边界在清单层面关闭）。
- 通用（非 CEVA）全链路因 QUANT/GATHER 计时修正由 1815.611866 s 变为 1817.367954 s（+1.756088 s，0.097%），属于预期修复；增量按 8 台 DPU 的流水线关键路径体现，不等于逐请求增量之和（13.382989 s）。
- round07～round19 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、链路语义与主机证据独立性、B 路预算边界、通用短循环截断、运行清单工具链身份反推等未在本轮改变（K15、K16、K20、K21、K22、K23、K24、Q8、Q12、Q17、Q18）；K18 的单行内核计量边界本轮已按上面第一条收窄。

第 21 轮（round21，2026-11-08）：

- 探针脚本：修复前 `/tmp/probe21_issue1b.py`（问题 1）、`/tmp/probe21_issue23.py`（问题 2/3）、`/tmp/probe21_issue4.py`（问题 4）、`/tmp/probe21_issue5.py`（问题 5）；修复后统一用 `/tmp/probe21_after.py` 复验，输出见下。
- 问题 1 确认与修复核对：
  - 给 `/tmp/ceva-round20-smoke.yaml` 补上本轮新增的循环编码来源后，修复前 `validate(mode="simulation")` 为 True；`override(autotune={"enabled": True})` 与 `GENESIM_AUTOTUNE_ENABLED=true` 两条入口也均为 True。修复后同一探针输出 base=True、override=False、env=False，日志为“CEVA 正式运行要求 autotune.enabled=false”。
  - 新增用例修复前：`test_simulation_rejects_autotune_placement_search` 报 `True is not false`；`test_ceva_run_rejects_autotune_before_searching` 因未在搜索前拦截而落到 `VPUs not set.`。修复后前者对 YAML 与环境两条入口都断言 False，后者断言 `_autotune_placement()` 未被调用并抛出“自动放置”；`test_ceva_fixed_placement_still_reaches_the_pipeline` 证明禁用搜索后仍进入固定放置的流水线入口。
- 问题 2 确认与修复核对：
  - 修复前探针：`{"operators": {}}` 的空成本文件 `annotated_operators=0`、有效 MRAM 覆盖 0，成熟度仍为 `validated_in_scope`；`{"operators": {"1": {}}}` 的空条目前把 `annotated` 增到 1。修复后空文件为 `model_prediction`，有效成本 sidecar 为 `validated_in_scope`。
  - 新增/改写用例修复前全部失败：空文件、空条目、未声明部分覆盖、覆盖计数不一致均返回 `validated_in_scope`；原 `test_validation_report_cost_sidecar_must_match_the_run` 把空文件当正例。修复后空文件/空条目/未声明部分覆盖/覆盖计数不一致均保持 `model_prediction`，五份夹具模型算子全部有正数 `mram_traffic_bytes` 的正向 sidecar 恢复 `validated_in_scope`；显式声明 `coverage.template` 的部分覆盖按同一五计数进入验证。
  - 台账内容：`operators`、`annotated_operators`、`cost_operators`、`applied_fields.source_name`、`applied_fields.mram_traffic_bytes`、`coverage` 五项与 `uncovered_operators` 均进入运行结果 `cost_sidecar_consumption` 和 `run_manifest.json` 的同名字段；真实成本改变运行时 `data_bytes` 的既有用例继续通过。
- 问题 3 确认与修复核对：
  - 修复前同一份报告在 `pim.compile.use_long_loop_counts=true/false` 下都返回 `validated_in_scope`，两份 `_ceva_cost_model_summary()` 完全相同且不含循环编码。修复后短/长循环摘要分别为 `['LOOP_BEGIN_INDIRECT','LOOP_END']` 与 `['LOOP_BEGIN_INDIRECT_LONG','LOOP_END']`，切换后旧报告保持 `model_prediction`。
  - 短/长循环入口单价 1/100、10 次迭代的实际周期为 12/111 的既有行为由新增 `test_loop_encoding_price_lands_in_the_actual_cycles` 锁定；短循环超过 4095 次时的完整计数验收（`test_ceva_upgrades_a_short_trace_to_bill_every_n_iteration`）继续通过，并新增 `runtime_loop_upgrade` 元数据断言。
  - 环境覆盖：`GENESIM_PIM_COMPILE_USE_LONG_LOOP_COUNTS=true` 时来源台账未登记 `value` 的报告保持 `model_prediction`，登记 `value=true` 后恢复 `validated_in_scope`。
- 问题 4 确认与修复核对：
  - 修复前探针：`zone_scale(1, {"weight": 1.25, ...})=2`、`zone_scale_for("kv", {"kv": 0.5, ...})=1`，`compile_from_pimir()` 与 `phase_cycle_breakdown()` 对同一非法表都不报错；合法 1.5 倍编码为 3。修复后 1.25 与 0.5 都抛出 `ValueError: 区间倍率必须有限、≥1、为 0.5 的整数倍且满足 2*factor<=4095`，两条公共编译/求值入口同样拒绝；`zone=None` 仍返回 0，1.5 倍仍精确编码为 src2=3。
  - 配置侧既有 `TestZoneFactorValidation` 全部通过：缺区间仍返回 False（配置入口 `require_all=True`），直接编译接口只给 `kv` 的既有用法保持可用。
- 问题 5 确认与修复核对：
  - 修复前探针：真实 CEVA 配置 `override(ceva={"engines_per_core": 4})` 后 `config_path` 指向已删除的临时 YAML，`build_ceva_run_manifest()` 报 `RuntimeError: CEVA 运行清单要求的 配置文件 不存在`；无参数 `override()` 同样复现。
  - 修复后同一探针输出 `config_file.origin=in_memory`、`path=None`、`sha256 == resolved_config_sha256(cfg)`、`source` 指向原始 CEVA 配置文件、`override_sections=[['ceva']]`；二次覆盖累加为 `[['ceva'],['scheduler']]`，源文件路径保持。实体配置文件仍记录原始路径与 SHA256，输入产物哈希核对不变。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests`/`Ran 1 test` 合计 1115 条、0 失败（round20 同一日志口径为 1089 条，本轮新增 26 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q -p no:cacheprovider`，1120 通过、1 跳过、0 失败（58.22 秒；round20 同一口径为 1094 通过、1 跳过，本轮新增 26 条）。
- genesim 受影响文件：`test_config_loader.py`（98 通过）、`test_cost_sidecar.py`（19 通过）、`test_compiler_placement.py`（220 通过）、`test_run_manifest.py`（16 通过）、`test_zone_cycles.py`（23 通过）、`test_pimir_trace.py`（68 通过）、`test_vpu.py`（48 通过）、`test_pim_trace_loader.py`（8 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变。
- 五份仓库 CEVA 场景结构校验：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python src/main.py --config <scenario> --dry-run` 对 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml` 全部退出码 0，输出 `structural_only`。把 `scheduler.compiler_placement_file` 指向真实 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 后，五份场景的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；追加 `tests/test_placement_export.py -q` 后合计 81 通过、4 跳过、0 失败；补充 `tests/test_genesim_bridge.py -q`，58 通过、1 跳过、0 失败（约 6 分 39 秒）。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`，`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1817.368`、`tokens/s=4.448`，与 round20 的 1817.367954 s 一致（差异仅在打印精度），说明本轮配置校验、成本台账、循环编码和运行清单改动没有改变通用链路的计时。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round21 增补）：

- 验证报告契约升到 `schema_version=9`，新增 `scope.cost_model.loop_encoding` 与 `package.cost_sidecar.coverage`，需独立校准产出方按同一字段生成（Q8）。
- 成本 sidecar 的“未覆盖”判定依赖 sidecar 自带的 `coverage.template` 列表：生产方若改变覆盖结构（例如不再输出 template 列表），未进入 `operators` 的算子会被判为未声明并保持 `model_prediction`。该结构是跨仓约定，需与算子编译器成本导出一起确认。
- `runtime_loop_upgrade` 只记录本次是否发生过短→长循环升级，不记录升级的轨迹数或次数；若需要逐轨迹的成本证据，应扩展元数据与报告字段。
- 内存配置的运行清单不单独落盘生效配置副本，内容身份依赖 `run_manifest.json` 中的 `resolved_config` 与规范化 JSON 哈希；若消费方需要独立 YAML 副本，需另行归档。
- round07～round20 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断、运行清单工具链身份反推等未在本轮改变（K15、K16、K18、K20、K21、K22、K23、K24、Q8、Q12、Q17、Q18）。

第 22 轮（round22，2026-11-08）：

- 探针脚本：评审提供的 `/tmp/ceva_review_round22_mac_probe.py`（问题 1）、`/tmp/ceva_review_round22_cost_probe.py`（问题 2）、`/tmp/ceva_review_round22_long_api_probe.py`（问题 3）。修复前分别输出：相位 12 次对分块 16 次 MAC、`uncovered_operators=0` 且 `validated_in_scope`、`ValueError: Long tile count halves must fit in 12 bits, got hi=1, lo=5000`；修复后分别为：两侧均 16 次、`uncovered_operators=4` 且 `model_prediction`、短/长轨迹与直接计数对照均为 5002 周期。
- 问题 1 真产物核对：从 64 核冒烟轨迹 `op_1001_GEMV_CONTEXT.pim_trace` 读取相位模板，按 Tq=726、KV=726 解析后 `mac_count=67,465,728`，编译轨迹滚动求和为 8,456,448 次 MAC，等于 `726×128×ceil(726/8)`；旧口径为 `ceil(726×726×128/8)=8,433,216` 次，每算子少计 23,232 次。`phase_cycle_breakdown()` 与指令求值在非整除形状下对拍一致。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests`/`Ran 1 test` 合计 1127 条（含 1 条跳过）、0 失败（round21 同一日志口径为 1115 条，本轮新增 12 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1132 通过、1 跳过、0 失败（60.71 秒；round21 为 1120 通过、1 跳过，本轮新增 12 条）。
- genesim 受影响文件：`tests/sim/test_pimir_trace.py`（72 通过）、`tests/sim/test_cost_sidecar.py`（21 通过）、`tests/sim/test_compiler_placement.py`（222 通过）、`tests/sim/test_pim_backend.py`（22 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变，`conf/sim.yaml` 未修改。
- 五份仓库 CEVA 场景复验：真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下，`ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-review-round22-smoke.yaml`，退出码 0。7138 条 trace 全部按新签名重编（`schema_version=12`，7138 compiled、0 reused）；`total_time_s=671.967412`、`tokens/s=1.183`、`result_maturity=model_prediction`、`simulation_backend=pipeline`；资源报告“创建 64、使用 8、空闲 56”。与修复前同配置的 671.949968 s 相比增加 0.017444 s：按 1024 个 GEMV_CONTEXT 算子在 prefill（Tq=KV=726）与 69 个 decode 步（KV=727～795）复算的尾量分组增量约为 17.5 ms，实测增量与其一致，属于问题 1 修复的预期变化。同一配置重跑时 7138 条 trace 全部复用（0 compiled、7138 reused），总时延仍为 671.967412 s，签名升级后的缓存复用与计时稳定。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部按 `schema_version=12` 重编（trace 来源 `{'pimir': 7138}`；元数据核对为 7074 条 `pimir` + 64 条 `absorbed`），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1818.417206`、`tokens/s=4.446`、`processed_tokens=8084`、`simulation_backend=pipeline`。与 round21 的 1817.368 s 相比增加 1.049 s（0.058%）：按 10 个请求的 prefill/decode 形状逐条复算，1024 个 GEMV_CONTEXT 的尾量分组 MAC 增量约 0.34 s，流水线阶段不均衡与关键路径使实际增量更大；方向与原因同问题 1 修复一致，属于预期变化。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round22 增补）：

- 通用 `DRAMsim3PIMCore`/`Ramulator2PIMCore` 的 Python 回退仍把长循环接口的 `outer/inner` 当 hi/lo 转发，未在本轮统一（K25）；CEVA 评估路径已不受影响。
- 相位乘加工作量改为逐输出取整后，KV 不是 `mac_width` 整数倍的 GEMV_CONTEXT 成本会上调；64 核冒烟的总时延变化与逐算子复算的尾量增量一致，全链路总时延由 1817.368 s 变为 1818.417 s（+0.058%），属于预期修复而不是回归。
- `source_name` 仍写入 Operator 作为内核身份，但 GeneSim 运行期没有读者；本轮只把它从覆盖证据中剔除，没有删除字段或新增消费逻辑（最小改动）。
- 成本 sidecar 的未覆盖判定仍依赖生产端提供 `coverage.template` 列表和正数 `mram_traffic_bytes`；若独立校准产出方改变 sidecar 结构，需要同步契约（Q8）。
- round07～round21 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断、运行清单工具链身份反推等未在本轮改变（K15、K16、K18、K20、K21、K22、K23、K24、K25、Q8、Q12、Q17、Q18）。

第 23 轮（round23，2026-11-08）：

- 探针脚本：评审提供的 `/tmp/ceva_review_round23_cost_snapshot_probe.py`（问题 1）与 `/tmp/ceva_review_round23_boundary_probes.py`（问题 2、3）。修复前分别输出：成本文件改写后 `maturity_after_rewrite=validated_in_scope`、`cached_ledger_matches_consumed_snapshot=false`；正无穷搬运字节下 `ledger_cost_operators=5`、`runtime_data_bytes=Infinity`、成熟度 `validated_in_scope`；YAML `num_cpus=0` 与环境覆盖 `peak_flops=0`、`peak_bandwidth_gbs=nan`、`base_latency_us=-1` 下 `simulation_valid=True`，剖面分别 `ZeroDivisionError`、NaN 存储时间和负时延。修复后统一用 `/tmp/round23_after_probe.py` 复验（该脚本捕获构造异常并逐项打印），输出：四条非法主机参数 `simulation_valid=False` 且 `CPUModel` 抛 `ValueError`；正无穷成本加载时抛 `成本 sidecar ... mram_traffic_bytes 必须是有限非负数，收到 inf。`；有限值 4096 的运行时 `data_bytes=4096.0`、台账有效成本算子 5、成熟度 `validated_in_scope`。
- 问题 1 复现与修复核对：成本探针修复前 `loaded_mram_before_rewrite=1024.0`、`runtime_mram_still_old_value=1024.0`、`maturity_after_rewrite=validated_in_scope`，修复后 `maturity_after_rewrite=model_prediction`；新增 placement 用例 4 条、manifest 用例 4 条，修复前分别报 `validated_in_scope != model_prediction`、`ImportError: verify_input_snapshot`、清单错误地落到工具链校验；修复后全部通过。CEVA 64 核冒烟的 `run_manifest.json` 三个输入与 21 份 PIM MLIR 哈希由外部脚本复算，`hash_mismatches=[]`。
- 问题 2 复现与修复核对：`test_ceva_rejects_non_finite_or_negative_mram_values` 覆盖标量、分档字典、measurements 回退三种形态 × 正负无穷、NaN、负数、字符串、布尔、列表七类非法值，修复前全部 `RuntimeError not raised`，修复后全部抛出且失败加载不修改算子（保持 0.0）；`test_ceva_accepts_zero_none_and_finite_mram_values` 覆盖零值、`None`/measurements 回退和有限值，运行时 `data_bytes=128.0` 且有限，台账有效成本算子 2、未覆盖为空，显式模板算子不计未覆盖。真实 `/tmp/ceva-round14-export/oplevel_extensions.json` 的 20,550 个 `mram_traffic_bytes` 值全部通过新校验。
- 问题 3 复现与修复核对：`test_simulation_rejects_non_positive_cpu_count`、`test_simulation_rejects_invalid_host_parameters_from_every_entry`、`test_invalid_effective_parameters_are_rejected_before_profile`、`test_environment_overrides_are_checked` 在修复前分别以 `True is not false` 或 `ValueError not raised` 失败，修复后通过；`test_valid_control_profiles_finite_latency` 与 `test_factory_uses_the_effective_ceva_host_parameters` 锁定合法对照（含工厂使用环境覆盖值与工厂非法值拒绝）。五份仓库 CEVA 场景 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部 True。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1143 条、1 条跳过、0 失败（round22 同一日志口径为 1127 条，本轮新增 16 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1148 通过、1 跳过、0 失败（62.03 秒；round22 为 1132 通过、1 跳过，本轮新增 16 条）。
- genesim 受影响文件：`tests/sim/test_cost_sidecar.py`（23 通过）、`tests/sim/test_compiler_placement.py`（226 通过）、`tests/sim/test_run_manifest.py`（20 通过）、`tests/sim/test_config_loader.py`（100 通过）、`tests/sim/test_cpu_backend.py`（14 通过）、`tests/sim/test_main.py`（22 通过）、`tests/sim/test_vpu.py`（48 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变，`conf/sim.yaml` 未修改；`conf/sim.yaml` 的 `validate(mode="simulation")` 仍为 True。
- 五份仓库 CEVA 场景复验：真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下，`ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；`system.num_cpus=1`、`cpu_model.peak_flops=1e12` 与场景文件一致。五份场景 `uv run python src/main.py --config <scenario> --dry-run` 均退出码 0 并输出 `structural_only`。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round23-fix-smoke.yaml`，退出码 0。首次运行 7138 条 trace 全部重编（新轨迹目录，7138 compiled、0 reused），随后重跑均为 0 compiled、7138 reused；`total_time_s=671.9674122015546`、`throughput_tokens_per_s=1.1830930869033605`、`result_maturity=model_prediction`、`simulation_backend=pipeline`、`processed_tokens=795`；资源报告“创建 64、使用 8、空闲 56”，物理 engine 256。清单 `inputs` 记录模型 IR、放置 sidecar 与请求 trace，`pim_mlir` 21 份，外部复算哈希全部一致；`cost_sidecar_consumption.configured=false`（冒烟未配置成本 sidecar）。多次运行的 `total_time_s` 完全相同，说明本轮快照与校验改动没有改变计时。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码，`contracts/` 与图编译主链未变。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部按 `schema_version=12` 重编（trace 来源 `{'pimir': 7138}`；元数据核对为 7074 条 `pimir` + 64 条 `absorbed`），`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真成功结束，`total_time_s=1818.417206`、`tokens/s=4.446`、`processed_tokens=8084`、`simulation_backend=pipeline`。与 round22 的 1818.417206 s 完全一致，说明本轮快照与校验改动没有改变通用链路的计时；通用 `summary.json` 不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

第 24 轮（round24，2026-11-08）：

- 评审探针：`/tmp/ceva_review_round24_boundary_probes.py`（问题1 模型 IR 窗口与问题2 CPU 入口、问题3 成本大整数）、`/tmp/ceva_review_round24_trace_window.py`（问题1 请求 trace 窗口）、`/tmp/ceva_review_round24_cpu_cost_controls.py`（问题2 `cpus` 列表与真实成本 sidecar）、`/tmp/ceva_review_round24_numeric_probes.py`（问题3 六类入口）。修复前：模型 IR 消费 `old` 但快照匹配未消费的新字节且最终复核 `passed`，请求 trace 消费 `old-request` 而快照匹配 `new-request` 且复核 `passed`；`cpu_list_two.simulation_valid=true`、工厂创建 2 个 CPU；`10**1000` 在配置 MAC、配置区间倍率、直接 `zone_scale()`、报告 `_finite_number()`、VPU `_positive_float()` 与成本加载处抛 `OverflowError`。修复后：两个窗口的快照均匹配旧消费字节，最终复核抛“CEVA 输入快照登记的 ... 内容已变化”；`cpu_list_two.simulation_valid=false` 且工厂抛 `ValueError`，合法单核仍创建 1 CPU/1 PIM；`config_mac=false`、`config_zone=false`、`direct_zone` 抛 `ValueError`、报告数值为 `null`、VPU 回落默认值、成本加载抛带算子与字段路径的 `RuntimeError`，`legal_control=3`；真实成本 sidecar 6850 条记录、27400 个数值校验通过。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1155 条、1 条跳过、0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1160 通过、1 跳过、0 失败（61.76 秒；round23 为 1148 通过、1 跳过，本轮新增 12 条用例）。
- genesim 受影响文件：`tests/sim/test_compiler_placement.py`（230 通过）、`test_config_loader.py`（104 通过）、`test_cpu_backend.py`（16 通过）、`test_main.py`（23 通过）、`test_cost_sidecar.py`（23 通过）、`test_vpu.py`（49 通过）、`test_zone_cycles.py`（23 通过）、`test_run_manifest.py`（20 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变，`conf/sim.yaml` 未修改。
- 五份仓库 CEVA 场景复验：真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下，`ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建核数 1/64/64/128/128，GPU 均为 0，物理 engine 数为 4/256/64/512/128，容量均为 8,000,000,000 B、预留 0 B；真实 sidecar 6850 条记录引用 21 份 PIM MLIR，缺失与哈希错误均为 0。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round24-fix-smoke.yaml`，退出码 0。`total_time_s=671.9674122015546`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，资源报告“创建 64、使用 8、空闲 56”，与 round23 完全一致；运行清单 `inputs` 为模型 IR、放置 sidecar、请求 trace 三项，21 份 PIM MLIR 外部复算哈希 0 不一致，`result_maturity` 为 `model_prediction`。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`，`(tile_n, k_iterations)` 分布为 `{(None, None): 6626, (512, 128): 320, (512, 64): 64, (512, 172): 64}`；仿真 `total_time_s=1818.4172063463388`、`tokens/s=4.4456`、`processed_tokens=8084`、`simulation_backend=pipeline`。与 round23 的 `1818.417206` 完全一致，说明单次读取加载与安全数值转换没有改变通用链路的计时；通用 `summary.json` 不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

第 25 轮（round25，2026-11-08）：

- 评审探针：`/tmp/round25_repro_issue1.py`（真实 CEVA 画像下的链路数值契约）与 `/tmp/round25_repro_issue2.py`（CEVA 核心与相位解析的周期成本数值契约）。修复前：`latency_base_ns=-1`、NaN 带宽、零带宽均使 `validate(mode="simulation")=True`，HOST→PIM 1 B 直连分别为 `-8.75e-10 s`、`NaN`、`inf`，零带宽全双工抛 `ZeroDivisionError`，超大整数带宽构造抛未包装 `OverflowError`；`MAC=NaN` 返回 `nan`、`MAC=-1` 返回 `-1.0`、`MAC=10**1000` 抛 `OverflowError`，相位入口同样 NaN 透传、负值少计（`21.8` 对合法 `45.8`）。修复后：所有非法链路使 `validate=False`，`CommLib`/工厂构造抛带链路与字段名的 `ValueError`；核心与相位入口所有非法成本抛带原语名的 `ValueError`，合法对照 `MAC=2` 得 `2.0`、相位 `REDUCE=2` 得 `45.8`；真实 `ceva_64core_4e.yaml` 加真实放置 sidecar 的合法配置仍为 `simulation=True`。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1177 条、1 条跳过、0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1182 通过、1 跳过、0 失败（round24 为 1160 通过、1 跳过，本轮新增 22 条用例）。
- genesim 受影响文件：`tests/sim/test_comm_lib.py`（36 通过）、`test_config_loader.py`（108 通过）、`test_cpu_backend.py`（17 通过）、`test_compiler_placement.py`（231 通过）、`test_pim_trace_loader.py`（13 通过）、`test_pimir_trace.py`（75 通过）、`test_pim_backend.py`（23 通过）、`test_scheduler.py`（19 通过）、`test_vpu.py`（50 通过），均 0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `./run.sh --test upmem_checker`，49 条用例通过、0 失败。通用与 UPMEM 路径行为不变，`conf/sim.yaml` 未修改。
- 五份仓库 CEVA 场景复验：真实放置 sidecar `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 下，`ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 的 `validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True；实际创建核数 1/64/64/128/128，GPU 均为 0，物理 engine 数为 4/256/64/512/128，容量均为 8,000,000,000 B、预留 0 B；真实 sidecar 6850 条记录引用 21 份 PIM MLIR，缺失与哈希错误均为 0。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round24-fix-smoke.yaml`，退出码 0。`total_time_s=671.9674122015546`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，资源报告“创建 64、使用 8、空闲 56”，与 round23/round24 完全一致；7138 条轨迹全部复用。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。7138 条 trace 全部来自 `pimir`，仿真 `total_time_s=1818.4172063463388`、`tokens/s=4.4456`、`processed_tokens=8084`、`simulation_backend=pipeline`，与 round23/round24 完全一致，说明本轮数值契约没有改变通用链路的计时；通用 `summary.json` 不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

第 26 轮（round26，2026-11-08）：

- 评审探针：`/tmp/ceva_review_round26_issues.py`（资源类型、链路标记、请求边界三组复现）与修复后复验脚本 `/tmp/ceva_review_round26_postfix.py`。修复前 `engines_per_core=True`、`clusters_per_device=True/1.0`、`tensors_per_cluster=1.0`、`num_pim_devices=1.0` 在 structure/simulation 下均为 true，工厂分别创建 64/1 核或抛未包装 `TypeError`；`is_full_duplex="false"` 正式校验为 true 且按全双工计 `5.00125e-7 s`；padded 请求展开为 `Tq=128`、有效长度 5 未进入执行形状。修复后四个资源字段的布尔/浮点/字符串反例均 false 且工厂抛字段化 `ValueError`，合法对照仍为 64 核/256 engine；三个链路标记的字符串/整数/空值均 false 且 `CommLib` 构造抛错，布尔全/半双工成本分别为 `5.00125e-7`/`1.00025e-6 s`，`inter_cluster_only` 真假正确控制路由；歧义请求边界为 None 且运行入口抛错，显式登记后边界分别登记 `[5]/[128]/[0]` 并按 128 执行，普通 5/5 请求仍为 5。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1210 条、1 条跳过、0 失败。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1215 通过、1 跳过、0 失败。
- genesim 受影响文件：`python -m pytest tests/sim/test_trace_ir.py tests/sim/test_compiler_placement.py tests/sim/test_config_loader.py tests/sim/test_comm_lib.py tests/sim/test_cpu_backend.py tests/sim/test_main.py tests/sim/test_run_manifest.py tests/sim/test_scheduler.py tests/sim/test_pim_capacity.py tests/sim/test_pu_pool.py -q`，571 通过、0 失败。
- genesim 通用预测器入口：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor`，7/7 个测试文件通过、87 条用例、0 失败；UPMEM 检查入口 `PYTHONDONTWRITEBYTECODE=1 bash tools/upmem_checker/run.sh --test`，49 条通过、0 失败。
- 五份仓库 CEVA 场景与真实 sidecar：执行 `/tmp/ceva_review_round26_verify.py`，`ceva_single_core_4e`/`ceva_64core_4e`/`ceva_64core_1e`/`ceva_128core_4e`/`ceva_128core_1e` 的 structure 与 simulation 均为 true，实际创建 PIM 核 1/64/64/128/128、GPU 均为 0、物理 engine 数 4/256/64/512/128；真实 sidecar 6850 条记录引用 21 份 PIM MLIR，缺失与哈希错误均为 0。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round26-smoke.yaml`，7138 条轨迹全部复用（7074 条 `pimir`、64 条 `absorbed`、0 条模板），创建/使用/空闲核为 64/8/56，`total_time_s=671.967412`、`result_maturity=model_prediction`，与修复前完全一致（真实 trace 的 token 数与 `seq_len` 全等，三字段绑定走等值推断）。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；补充 `tests/test_placement_export.py` 后为 81 通过、4 跳过、0 失败。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`，仿真 `total_time_s=1818.417`、`tokens/s=4.446`，与修复前的 `1818.4172063463388`/`4.4456` 一致，说明通用请求形状推断与追踪编码改动没有改变既有链路结果。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时），与本轮改动无直接关系。

仍存在的不足（round23 增补）：

- 输入快照只记录路径与 SHA256，不复制输入文件本身；输入被删除或改写后正式评估显式失败，无法仅凭结果目录重放（与 round18 的清单边界一致）。
- `build_ceva_run_manifest()` 在没有 `input_snapshot` 参数时仍按当前磁盘读取，该回退只供独立工具和测试使用；正式调度器始终传入消费快照。
- 配置文件（YAML）不在输入快照内，`config_file` 仍按清单生成时的磁盘内容登记；实际执行值由清单 `resolved_config` 记录。若需精确绑定运行开始时的 YAML 内容，可把 `config_entry` 也纳入快照（本轮未实现，避免改变配置身份语义）。
- 非 CEVA 路径保持既有行为：成本 sidecar 的非有限值不会被新校验拒绝，CEVA 契约只作用于正式评估；VPU 工厂也只对 CEVA 使用生效 `cpu_model`，通用配置仍走原路径。
- round07～round22 列出的相位产物绑定边界、SPLIT 元素总数边界、p90 契约、单行内核登记集合、链路语义与主机证据独立性、B 路预算边界、通用短循环截断、运行清单工具链身份反推等未在本轮改变（K15、K16、K18、K20、K21、K22、K23、K24、K25、Q8、Q12、Q17、Q18）。

仍存在的不足（round24 增补）：

- 输入快照仍只记录路径与 SHA256，不复制输入文件；单次读取保证了“解析与登记同一份字节”，但输入被删除或改写后正式评估仍会显式失败，无法仅凭结果目录离线重放。
- `_ensure_input_snapshot()` 的“未调用加载接口时按配置路径补登记”回退仍然保留，供独立工具/测试使用；若调用方绕过 `load_model_ir()`/`load_trace()` 直接给 `model_ir`/`trace` 赋值，快照只能登记配置路径的当前文件，不能证明对象来源。正式 `main.py` 始终走加载接口。
- `_ceva_request_boundary()` 对尚未调用 `load_trace()` 的调用方仍回退配置路径；正式运行始终使用已加载的 `self.trace`，该回退不参与正式评估。
- 非 CEVA 成本 sidecar 不在本轮数值域契约内：`_mram_bytes_of()` 保持既有行为，超大整数仍按原路径处理；CEVA 在提取前已由 `_validate_ceva_cost_sidecar()` 拒绝非法值。
- `finite_float()` 只接受 `int/float` 且排除 `bool`；字符串形式的数值仍由各入口原有类型契约处理，不做隐式转换。
- `_execute_pipeline` 内部对 VPU 返回剖面值的 `float()` 转换未纳入本轮统一边界：这些值由后端按自身契约计算，不属于外部产物输入；若后续允许第三方后端返回任意 JSON 数值，需要再收敛。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round25 增补）：

- 通用 `CommLink` 直接构造仍允许零带宽与 NaN 带宽（单路零带宽返回无穷，NaN 透传），链路正数契约只在 CEVA 配置校验、通信库构造、完整工厂和报告比较中启用；这是评审明确要求保留的通用既有行为，非 CEVA 调用方直接构造非法链路不会被拒绝。
- `resolve_cycle_costs()` 的非严格路径保持原有 `float()` 与内置默认表语义；数值域契约只对 strict（CEVA）求值生效。`vpu._positive_float`、`scheduler.positive_float` 与预测器特征中的成本转换仍按显式默认值回落，本轮未把它们纳入共享求值入口（它们不是轨迹/相位计时路径）。
- `_estimate_reduction_time()` 只校验 `REDUCE_PARTIAL` 的有限非负，时钟仍由配置校验负责；通过 `GeneSimScheduler.__new__` 之类绕过初始化的直接调用不在正式运行保护范围内。
- CEVA 配置校验遍历 `pim.pim_trace_cycle_costs` 全部条目，但未知键名仍由 `resolve_cycle_costs()` 在求值时拒绝；配置阶段不会提前报未知键（沿用原有分层）。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round26 增补）：

- 旧请求 trace 只在 `seq_len == len(input_tokens)` 时自动推断三字段；带 padding、chunked prefill 或 prefix cache 的 trace 必须由生产方显式登记 `padded_query_length`/`initial_kv_length`，否则 CEVA 正式运行会显式拒绝（Q20）。仓库内现有真实 trace 全部是等值场景，未提供带 padding 的真实产物做端到端对照。
- 通用路径遇到单字段歧义 trace 仍按原 `seq_len` 约定执行，这是为保持 `conf/sim.yaml` 与 UPMEM 既有行为而保留的回退；它不参与 CEVA 正式评估，但通用场景若存在 padding 语义仍会被当成执行长度。
- `request_metrics` 保留原有 `prompt_length` 字段（prefill 仍记 `seq_len`），另加 `valid_prompt_length`/`padded_query_length` 作为独立证据；现有数据集消费者未改读新字段，若需要用有效长度重算吞吐，需另行约定。
- 目标名 engine 后缀核对只在名称以 `_<数字>e` 结尾时生效；外部目标命名若采用其它约定，仍需人工保证 `target_name` 与 `engines_per_core` 一致。
- 通用 `CommLink` 直接构造与非 CEVA 配置仍不启用链路标记的布尔契约（保持旧真值语义）；CEVA 配置、通信库构造、完整工厂与报告比较已全部走严格契约。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 27 轮（round27，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1225 条、1 条跳过、0 条失败（round26 同一口径为 1210 条，本轮新增 15 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1230 通过、1 跳过、0 失败（69.84 秒）。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_compiler_placement.py`（249 条 OK）、`test_pim_backend.py`（27 条 OK）、`test_config_loader.py`（116 条 OK）、`test_comm_lib.py`（43 条 OK）、`test_cpu_backend.py`（18 条 OK）、`test_pim_trace_loader.py`（13 条 OK）。
- 通用预测器与 UPMEM 路径：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过，均 0 失败，说明通用链路未受 CEVA 端点契约影响。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向 `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True（1/64/64/128/128 核、4e/4e/1e/4e/1e）。
- 问题 1 复现与修复核对：复跑 `/tmp/ceva_review_round27_probes.py`，修复前实际消费 i8 时成熟度为 `model_prediction`，把磁盘改成未参与 trace 的 f16 后同一快照变为 `validated_in_scope`、`_ceva_validation_scope_error()` 返回空、`_ceva_run_compute_dtypes()` 重读出 f16；恢复 i8 后又回到预测。修复后复跑同一探针（`/tmp/ceva_review_round27_probes_after.py`）：`maturity_after_unconsumed_f16_rewrite=model_prediction`、原因是“报告的数据类型布局 GEMM 与产物 i8 不一致”、`dtype_reread_after_rewrite={"GEMM":"i8"}`，恢复 i8 后快照复核通过且结论仍为预测。新增测试覆盖：消费 i8 + 报告 fp16 时改写 f16 仍保持预测；稳定 f16 正例为 `validated_in_scope`，磁盘改成 i8 后保持预测且台账仍为 f16，恢复后回到已验证；trace 文本与证据文本完全相等；TP 分片按继承路径复用台账；完整 `scheduler.run()` 流水线在稳定输入下为 `validated_in_scope`，改写后再次运行保持 `model_prediction` 且 trace 仍用消费文本。
- 问题 2 复现与修复核对：修复前追加 `GPU→HOST`、`HOST→GPU`、`PIM→GPU` 后 structure/simulation 均为 True、`CommLib` 实际注册、GPU 资源为 0；修复后同一探针三类附加链路的 `structure=false`、`simulation=false`，`CommLib` 与完整工厂均抛“端点不属于纯 PIM 资源域，只允许 HOST/PIM/CPU”。新增测试覆盖 GPU 双向、`GPU→GPU`、未知域 `NPU`/小写域在结构/正式校验、`CommLib.add_links`、直接 `add_link`、完整工厂与报告 `_normalized_links()` 的统一失败，并保留合法 CPU/HOST/PIM 链路 + 正字节有限成本 + GPU 资源为 0 的联合正例。
- 问题 3 复现与修复核对：复跑循环探针，修复前 `loop_count=2/0/1.5/True` 分别为 4/2/3.5/3 周期、`repetitions=1.5` 为 6 周期；修复后 `loop_count=0/-3` 报“必须为正整数”，`loop_count=1.5/True` 与 `repetitions=1.5` 报“必须是非布尔整数”，合法 `loop_count=2` 仍为 4 周期。新增测试覆盖通用与 CEVA 运行时入口、直接核心、Python 轨迹绑定、长循环两半的布尔/浮点/字符串/零值/负值反例；4095/4096/5000 次完整计费、短轨迹升级与通用 12 位截断既有用例全部保持通过。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round27-smoke.yaml`，退出码 0。7138 条轨迹全部复用（0 编译），创建/使用/空闲核为 64/8/56，`total_time_s=671.967412`、`tokens/s=1.183`、`result_maturity=model_prediction`，与修复前完全一致；生成的 `run_manifest.json` 含 3 项输入与 21 份 PIM MLIR，逐份复算 SHA256 与放置 sidecar 声明一致（缺失 0、哈希不符 0）。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（模板 0）；仿真 `total_time_s=1818.417`、`tokens/s=4.446`，与 round26 完全一致，说明通用链路的台账未受本轮 CEVA 会话影响。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round27 增补）：

- 台账只保存本轮首次读取的 PIM MLIR 文本与哈希；台账建立时读不到、之后才出现的产物会让 CEVA 运行显式失败，不会补读新文件。这是刻意的内容封闭语义。
- 成熟度前的快照复核会重新哈希全部 PIM MLIR（真实 sidecar 为 21 份唯一文件），大缓存目录下会增加一次 IO；本轮接受该成本以换取结论可追溯。
- CEVA 链路端点集合固定为 HOST/PIM/CPU；若后续目标新增资源域，需要显式扩展 `CEVA_LINK_DOMAINS` 并同步报告契约。
- CEVA 循环次数必须为正整数；通用/UPMEM 路径保留零值、负数告警回落与 12 位兼容截断，两类口径仍不同（K22、Q17）。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 28 轮（round28，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1237 条、1 条跳过、0 条失败（round27 同一口径为 1225 条，本轮新增 12 条）。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_run_manifest.py`（28 条 OK）、`test_comm_lib.py`（47 条 OK）、`test_cpu_backend.py`（19 条 OK）、`test_config_loader.py`（116 条 OK）、`test_compiler_placement.py`（250 条 OK；round27 为 249 条，本轮新增 1 条）。
- 通用预测器与 UPMEM 路径：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过，均 0 失败，说明通用链路的配置加载与通信库构造未受 CEVA 契约影响。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向 `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True（实际创建核数 1/64/64/128/128，物理 engine 4/256/64/512/128，GPU 均为 0，容量 8,000,000,000 B、预留 0 B）。
- 真实 sidecar 复验：`/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 的 6850 条算子引用 21 份唯一 PIM MLIR，缺失 0、哈希不符 0。
- 问题 1 复现与修复核对：修复前执行本地探针 `/tmp/round28_probe1.py`，加载时钟 0.63 的 YAML 后改写为 1.26，`build_ceva_run_manifest()` 仍成功，`resolved_config` 保持 0.63、`config_file.sha256` 却等于改写后的文件；无参数/带参数 `override()` 的 `source.sha256` 同样在清单生成时重读原文件。修复后同一探针在 `config_entry()` 与完整清单处抛“配置文件……内容已变化”；评审探针 `/tmp/ceva_review_round28_manifest_probe.py` 因快照缺少 `config_source` 被显式拒绝（`CEVA 输入快照缺少 ['config_source']`）。新增测试覆盖直接文件改写/删除、无参覆盖改写、带参覆盖改写、稳定输入正例、快照内配置改写与调度器 `_ceva_run_manifest()` 复核。
- 问题 2 复现与修复核对：修复前执行本地探针 `/tmp/round28_probe2.py`，直接 `add_link()` 接受 `latency_base_ns=-1` 的 CPU→HOST 链路、接受 `is_full_duplex="false"` 并按字符串真值计费；`host_relay_overhead_ns=-10000` 虽让 `validate(mode="simulation")` 返回 False，`CommLib(config=...)` 仍保存 `-1e-5 s`。修复后同一探针中负延迟、字符串布尔标记、零带宽与负中转开销均抛带字段名的 `ValueError`，合法直接链路 1 B 时延为 `1.00125e-7 s`、布尔半双工并发为 `2.0025e-7 s`，零/正中转开销按同一公式进入两段成本；评审探针 `/tmp/ceva_review_round28_comm_probe.py` 在第一条非法直接链路处即被拒绝。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round28-smoke.yaml`，退出码 0。7138 条轨迹全部复用（0 编译），`total_time_s=671.967412`、`tokens/s=1.183`、`processed_tokens=795`、`result_maturity=model_prediction`，与修复前完全一致；生成的 `run_manifest.json` 的 `config_file.sha256=109f17233fdbb248cf959893bb221e3928ade3de6570d37f487d1f753a0e3e01` 与本次加载的 `/tmp/ceva-round28-smoke.yaml` 字节一致，说明配置源文件已按消费身份登记并复核通过。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，45 通过、0 失败。本轮 G 未修改代码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`；仿真 `total_time_s=1818.4172063463388`、`tokens/s=4.4456`、`processed_tokens=8084`，与 round23～27 完全一致；通用 `summary.json` 仍不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round28 增补）：

- 配置源快照只记录路径与 SHA256，不复制文件内容；运行中改写或删除源配置会显式失败，但无法仅凭结果目录离线重放配置。
- 从不存在路径构造的 `GeneSimConfig` 没有持久源文件身份，运行清单继续按 `resolved_config` 内容哈希登记为 `in_memory`；该对象不能证明文件来源，这是“无文件配置”的既有语义。
- `override()` 只记录每层覆盖的顶层段名与最终生效配置；每层覆盖的原始输入值未单独归档，需要逐层审计时应由调用方保存覆盖输入。
- 清单与成熟度派生前会重新哈希一次配置源文件；文件很小，IO 成本可忽略，但与其他输入快照一样要求原文件在运行期间仍可访问。
- 通用/UPMEM `CommLib` 仍保留原 `float()` 转换与直接非法链路接受语义（评审要求保留）；CEVA 的严格数值域不扩散到通用路径，两类口径仍不同。
- CEVA 拒绝字符串形式的数值中转开销，而通用路径 `float("25")` 仍可转换；这是共享检查只用于 CEVA 构造与工厂的边界。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 29 轮（round29，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1253 条、1 条跳过、0 条失败（round28 同一口径为 1237 条，本轮新增 16 条）。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_pimir_trace.py`（80 条 OK）、`test_run_manifest.py`（33 条 OK）、`test_export_fixed_pu_mapping.py`（10 条 OK）、`test_compiler_placement.py TestCevaPlacementMapping.test_ceva_refuses_non_integer_config_memory_budget`（1 条 OK）。
- 通用预测器与 UPMEM 路径：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过，均 0 失败，说明通用 `sim.yaml` 与 UPMEM 链路未受逐操作账本、来源种类和容量契约影响。
- 五份 CEVA 场景复验：把 `scheduler.compiler_placement_file` 指向 `/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 后，依次加载 `ceva_single_core_4e.yaml`、`ceva_64core_4e.yaml`、`ceva_64core_1e.yaml`、`ceva_128core_4e.yaml`、`ceva_128core_1e.yaml`，`validate(mode="structure")` 与 `validate(mode="simulation")` 全部返回 True（配置核数 1/64/64/128/128、engine 数 4/4/1/4/1、GPU 0、容量 8,000,000,000 B、预留 0 B）。
- 真实 sidecar 复验：`/tmp/ceva-round14-export/llama2_7b_tp2_pp4_placement.json` 的 6850 条算子引用 21 份唯一 PIM MLIR，缺失 0、哈希不符 0。
- 问题 1 复现与修复核对：复跑 `/tmp/ceva_review_round29_unphased_ops.py`。修复前真实产物 `bpath-0858498cb704d901.pimir.mlir` 的 `parsed_phase_count=5`、`runtime_template_phase_count=5`，`skipped_compute_ops` 列出 `pim.eltwise`（fpsu、16,384 工作单元、33,024 B 输入、32,768 B 输出）；修复后同一探针两项计数均为 6，说明逐操作账本与运行时模板都包含该减法。真实 64 核冒烟重新编译的 7138 条轨迹中，`phase_template_version` 全部为 2，1024 条 SOFTMAX 的模板均为 6 步（`pim.reduce_axis, pim.eltwise, pim.lut, pim.reduce_axis, pim.lut, pim.eltwise`），第 2 步即原先被跳过的减法。
- 问题 2 复现与修复核对：复跑 `/tmp/ceva_review_round29_config_identity.py`，并补 `GENESIM_PIM_CLUSTER_MEMORY_CAPACITY_GB=7.450580596923828` 使字节真源与 GiB 视图一致。修复前加载缺失路径后新建 YAML 会让 `config_entry()` 从 `in_memory` 变为 `file` 并采纳新文件哈希，完整清单的 `resolved_config` 时钟 0.63 与 `config_file` 指向的 1.26 文件矛盾；修复后 `before == after`（均为 `in_memory`、`path=null`）、`after_matches_unconsumed_file=false`、完整清单 `config_file.origin=in_memory`、`resolved_config.pim.clock_period_ns=0.63`；始终缺失的对照 `control_stable_in_memory=true`。
- 问题 3 复现与修复核对：复跑 `/tmp/ceva_review_round29_mapping_contract.py`。修复前 `fractional_clusters`、`boolean_clusters`、`fractional_devices`、`fractional_slots`、`gpu_enabled` 的 `structure_valid=false` 但 `export_returncode=0` 且生成一对一计划；修复后五类反例全部 `export_returncode=1`、`plan_exists=false`，诊断分别指出 `pim.clusters_per_device`、`system.num_pim_devices`、`pim.tensors_per_cluster`、`system.num_gpus` 的具体非法值；`legal` 仍为 `structure_valid=true`、退出码 0 并生成 8 台 DPU 的一对一计划。新增测试另覆盖 1/64/128 核合法计划与通用配置共享映射。
- 问题 4 复现与修复核对：复跑 `/tmp/ceva_review_round29_budget_manifest.py`。修复前 `fractional_capacity`、`boolean_capacity`、`fractional_reserve`、`boolean_reserve` 的 `simulation_valid=false` 但清单仍生成（`local_memory_bytes` 取整为 8,000,000,000、`sys_reserve_bytes` 取整为 0），`legal_integer` 正常；修复后四类非法值均返回 `ValueError: ceva.local_memory_bytes must be a positive integer.` 或 `ValueError: ceva.sys_reserve_bytes must be a non-negative integer.`，`legal_integer` 的 `hardware.local_memory_bytes/sys_reserve_bytes/usable_memory_bytes` 为 8,000,000,000/0/8,000,000,000，与 `resolved_config.ceva` 一致。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round29-fix-smoke.yaml`（结果目录 `/tmp/ceva-round29-fix-results`、轨迹目录 `/tmp/ceva-round29-fix-traces` 均为本轮新建），退出码 0。7138 条轨迹全部重新编译（0 复用，模板版本 2），`total_time_s=672.533367`、`tokens/s=1.182`、`processed_tokens=795`、`result_maturity=model_prediction`；round28 为 `671.967412 s`，本轮增加 0.566 s（+0.084%），与 1024 条 SOFTMAX 每次执行补计 16,384 个工作单元、33,024 B 读和 32,768 B 写的方向和量级相符（具体执行次数由流水调度决定），属于预期增长而非链路退化。生成的 `run_manifest.json` 中 `config_file.origin=file`、SHA256 与本轮消费的 `/tmp/ceva-round29-fix-smoke.yaml` 一致，`hardware` 为 8,000,000,000/0/8,000,000,000 B，与 `resolved_config.ceva` 相同。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，45 通过、0 失败，确认合法 `--partition-plan` 仍可被图编译器消费端读取。本轮 G 未修改代码。
- G 扩展回归（可选；本轮未改动 `contracts/` 与图编译主链）：执行 `python -m pytest tests/ -q -k "not llama2_7b"`，运行至 94%（约 1593/1681 条、耗时约 17 分钟）后剩余用例单条长时间无输出，主动停止（退出码 143）；已执行部分无失败（输出只有通过点和跳过标记 `s`）。必需的两文件回归与放置导出消费测试均已完成，见上一条。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对结论，未涉及 F 源码。
- 本轮未重跑 `run_full_pipeline.py --num-stages 4`：G/F 源码未改动，且已用现有 21 份真实 PIM MLIR 的放置 sidecar 完成 E 段冒烟与 7138 条轨迹重编；A～D 段的产物新鲜度检查不涉及本轮改动。未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round29 增补）：

- 相位文件里的无属性视图/搬运语句按真实字节保守计费，只有 `convert_layout` 显式豁免；算子编译器若新增真正 NoMemoryEffect 的视图或合成通信语句，需要同步登记 `_PHASED_LEDGER_EXEMPT_OPS`，否则会显式失败（不会静默免费）。
- 账本 index 改为执行顺序序号，同一 op 的多相按声明 index 排序；当前真实产物每个 op 一相且声明 index 连续，没有出现两种索引需要并存或声明 index 乱序的产物。
- 加载时缺失的配置在运行中出现后保持内存身份，不会报“来源变化”；无法从清单判断运行者是否本来想使用后来创建的文件，需要调用方重新加载配置。
- 映射导出只执行资源数量与容量契约，不重复 `validate(mode="simulation")` 的时钟、来源台账等主流程校验；独立导出工具与正式评估入口的校验范围仍有分工。
- 运行清单要求 `ceva.local_memory_bytes` 作为 CEVA 字节真源；只有通用 `pim.cluster_memory_capacity_gb` 的配置会以字段化 `ValueError` 被拒绝（此前为 `int(None)` 的 `TypeError`）。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 30 轮（round30，2026-11-08）：

- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N test(s)` 合计 1267 条、1 条跳过、0 条失败（round29 同一口径为 1253 条，本轮新增 14 条：`test_pimir_trace.py` +3、`test_vpu.py` +5、`test_export_fixed_pu_mapping.py` +6）。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_pimir_trace.py`（83 条 OK）、`uv run python tests/sim/test_vpu.py`（55 条 OK）、`uv run python tests/sim/test_export_fixed_pu_mapping.py`（16 条 OK）。另用 PyTorch 环境复核全量：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再 `python -m pytest tests/sim -q`，1272 passed、1 skipped（pytest 对 subTest 的计数口径与 unittest 不同）。
- 通用预测器与 UPMEM 路径：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过，均 0 失败，说明通用 `sim.yaml` 与 UPMEM 链路未受相位折叠与缓存身份改动影响。
- 问题 1 复现与修复核对：复跑 `/tmp/ceva_review_round30_softmax_helper.py`。修复前 `bpath-0858498cb704d901.pimir.mlir` 的 `phase_spec_count=5` 但 `ledger_steps=6`（第 2 步为独立 `pim.eltwise` fpsu，16,384 工作单元、33,024 B 读、32,768 B 写），相对去掉该步骤的 5 相基线对照多 `23,856.8` cycles、`65,792 B` 搬运。修复后同一产物为 5 步（`pim.reduce_axis, pim.lut, pim.reduce_axis, pim.lut, pim.eltwise`），EXP 相 `folded_work=(fpsu, 16,384)`、`in_bytes=33,024`、`out_bytes=32,768`，无独立 fpsu 相；相对 5 相基线只多 `16,386` compute cycles（16,384 工作 + 2 循环开销）与 `256 B` 搬运（max 读入），指令轨迹（`compile_phase_chain` + `PIMTrace.estimate_cycle_breakdown`）与周期公式（`phase_cycle_breakdown`）两条路径数值一致。另有独立无相位 LUT（1 步、8,192 工作单元）、同文件无相位 matmul（nmu、独立计费）与未知语句报错对照用例。
- 问题 1 全产物归属复核：执行 `/tmp/ceva_round30_fix_scan.py` 扫描 `.opcompiler_cache` 的 25,173 份 PIM MLIR：2,154 份 `tt.dot`、22,874 份单相文件、133 份带 `phase_spec`（572 个声明相位）；133 份中 64 个无相位 `pim.eltwise` 全部被 `pim.lut` 消费者折叠（如 `a709a3cbd12c31a1.pimir.mlir`），独立无相位语句为 0，所有文件的账本步数均等于“声明相位 + 独立无相位语句”；12 份解析失败全部是空测试内核 `module { func.func @k() { return } }`，与评审结论一致，不是模型计算遗漏。
- 问题 1 三条路径一致性复核：执行 `/tmp/round30_phase_consistency.py`，对全部 133 份带 `phase_spec` 的真实产物比较周期公式（`phase_cycle_breakdown`）、指令轨迹（`compile_phase_chain` + `PIMTrace.estimate_cycle_breakdown`）与运行时模板重建（`phase_runtime_template` + `resolve_phase_template`）三条路径，`total_cycles` 与 `memory_bytes` 全部一致（0 条不一致），共覆盖 64 条折叠项。
- 问题 1 真实轨迹复核：64 核冒烟重编的 7,138 条轨迹中 `compile_signature.phase_template_version` 全部为 3（round29 为 2）；`trace_source` 为 `pimir` 7,074、`absorbed` 64、`template` 0；`work_binding` 为 `static_shape` 6,626、`runtime_tiles` 512。1,024 条 SOFTMAX 轨迹均为 5 步且各带 1 条折叠 fpsu（结果张量 `[128, 128]`），其余 op（GEMV 1 步、ROPE 3 步、QUANT 4 步等）无折叠项。
- 问题 2 复现与修复核对：复跑 `/tmp/ceva_review_round30_phase_cache.py`。修复前 `width_64_actual=5.28874e-05`，同 op_id 载入 128×128 新轨迹后 `width_128_after_reload=5.28874e-05`，而直接编译与新 VPU 对照均为 `1.057478e-04`；`stale_after_artifact_reload=true`，剖面仍报 `32768.0 B`。修复后 `width_128_after_reload=1.057478e-04`、`stale_after_artifact_reload=false`，剖面时延 `1.057478e-04`、搬运字节 `65536.0`，与直接编译一致。`UnifiedPIMBackend` 在两条装载入口登记递增的 `trace_serials`，同编号重载即换代号；新增反例覆盖两种装载入口（`load_pim_trace`/`load_tile_range_trace`）、周期表 `LUT` 单价变化、区间倍率 `kv` 1.0→2.5、时钟 1.0→2.0 ns；正例断言同一输入连续 `execute()` 与 `execute_profile()` 只求值一次缓存，且 `_phase_cache_identity` 只在小规模周期表上使用、不逐次冻结整份模板。
- 问题 3 复现与修复核对：直接调用 `build_mapping(one_to_one=True)` 的探针。修复前 `(9,4)` 返回 9 个映射、`(12,4)` 返回 12 个、`(0,1)` 返回空映射、`(8,0)` 抛 `ZeroDivisionError`；修复后这些参数与 `(8.0,4)`、`(True,1)`、`("8",4)` 等非法类型全部在除法前抛字段化 `SystemExit`（如 `num_stages=4 不能整除 num_dpus=9`、`tp_width=3 不是 2 的整数次幂`、`num_stages=0 必须为正`、`num_dpus=True 必须是整数`），合法 `(8,4)` 仍返回 8 个一一对应核。命令行反例同样退出非零且不生成计划；1/64/128 核合法计划经 CLI 生成后与公开函数返回值、下游 `contracts/partition_plan.py::PartitionPlan` 三者一致。
- 五份 CEVA 场景与真实 sidecar 复验：复跑 `/tmp/ceva_review_round30_inventory.py`，`structure`/`simulation` 全部返回 True，实际创建核数 1/64/64/128/128、物理 engine 4/256/64/512/128、GPU 均为 0、容量 8,000,000,000 B、预留 0 B、后端均为 `CEVANPMPIMCore`；21 份唯一 PIM MLIR 无缺失、无哈希不符，8 台 DPU 映射互不重复。本轮未修改配置校验与容量路径，复验结果与评审一致。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round30-fix-smoke.yaml`（结果目录 `/tmp/ceva-round30-fix-results`、轨迹目录 `/tmp/ceva-round30-fix-traces` 均为本轮新建），退出码 0。7,138 条轨迹全部重新编译（0 复用，模板版本 3），`total_time_s=672.363454`、`throughput_tokens_per_s=1.182396`、`processed_tokens=795`、`result_maturity=model_prediction`、`simulation_backend=pipeline`。修复前（round30 评审冒烟）为 `672.533367 s`，本轮减少 `0.169913 s`（-0.025%）：折叠移除了每条 SOFTMAX 执行的 33,024 B 读与 32,768 B 中间写回，只增加 256 B max 读入与 16,384 个 FPSU 工作单元，方向和量级相符。该结果同时纠正了 round29 把 `0.566 s` 增量解释为“预期增长”的结论：那部分增量来自错误的独立相位计费，折叠后净时延回落。生成的 `run_manifest.json` 中 `config_file.origin=file`、SHA256 `f846229b…5fbb1` 与本轮消费的 `/tmp/ceva-round30-fix-smoke.yaml` 一致，`hardware` 为 8,000,000,000/0/8,000,000,000 B。缓存身份改为轨迹代号后用最终代码重跑 `/tmp/ceva-round30-final-smoke.yaml`（独立结果目录），`total_time_s=672.363454`、`throughput_tokens_per_s=1.182396`、`processed_tokens=795`、`result_maturity=model_prediction`，与上述完全一致，证明缓存键优化未改变代价语义。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），模板版本均为 3，1,024 条 SOFTMAX 为 5 步且各带 1 条折叠 fpsu；仿真 `total_time_s=1812.037482`、`tokens/s=4.461276`、`processed_tokens=8084`、`simulation_backend=pipeline`。round28 同一链路为 `1818.417206 s`，本轮为 `1812.037482 s`（-6.380 s，-0.351%）；该差值是 round28→round30 多次口径变化（round29 补计无相位语句、round30 折叠修正）后的综合结果，流水线 makespan 对单相成本不是线性关系。两条链路均通过产物新鲜度、trace 来源与分块分布检查，通用 `summary.json` 仍不含 `result_maturity`/`run_manifest` 键。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，45 通过、0 失败。本轮未修改 compiler 仓库源码。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮只核对 `ExpandPhases.cpp` 的折叠语义和 `LowerPIMToEmitC.cpp` 的 helper 契约，未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round30 增补）：

- 折叠 helper 的工作量按 SSA 顺序并入第一个带相位消费者；真实产物均为单消费者，若将来出现一个 helper 结果被多个带相位消费者使用，成本只计一次并归属第一个消费者，分相展示可能不体现其余消费者。
- `FoldedPhaseWork` 只按 helper 的张量重算工作量；当前 FlagTree helper 契约限定为无 `phases` 的 Eltwise/Lut，若将来给 helper 配上 nmu 等依赖 MAC 的单元，需要扩展折叠表示（现有代码会显式失败，不会静默计 0）。
- 相位模板版本升到 3 会使盘上 v2 模板整体重编一次；这是修正折叠语义的必要代价，不是长期开销。
- 两条固定切分计划生产入口（`export_fixed_pu_mapping` 与 `export_partition_plan`）仍各自维护同一份切分契约；本轮已对齐规则与错误语义，但尚未抽成跨脚本共享模块，后续新增计划生产者应复用同一实现。
- 调度器 fallback 核心的 `latency_cache` 仍是每次运行内的 core+shapes 备忘录，不随轨迹模板变化失效；轨迹在仿真前一次性装载，当前没有运行中重载轨迹的路径，因此不属于本轮缓存根因。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 31 轮（round31，2026-11-08）：

- 探针脚本：评审提供的 `/tmp/ceva_review_round31_sidecar_consumers.py`（问题 1）与对照 `/tmp/ceva_review_round31_sidecar_control.py`。修复前改写探针输出 `original_dpus=[0,1]`、`consumed_dpus_after_rewrite=[1,0]`、`cached_collective_count=118`、`collectives_consumed_from_rewrite=1`、`relay_operators_added=2`，恢复原文件后 `snapshot_verify_after_restore=passed`；未修改 sidecar 的对照为 `consumed_dpus=[0,1]`、118 条、254 个中转。修复后改写探针输出 `consumed_dpus_after_rewrite=[0,1]`、`cached_collective_count=118`、`relay_operators_added=254`、`snapshot_verify_after_restore=passed`；对照仍为 `[0,1]`、118 条、254 个中转。
- 问题 1 的 TDD 失败复现：新增 `TestCevaSidecarExecutionView` 后先运行 `uv run python tests/sim/test_compiler_placement.py TestCevaSidecarExecutionView`，修复前 2 条失败（`test_tp_expansion_and_collectives_use_the_first_consumed_sidecar` 报 `[1,0] != [0,1]`，`test_missing_sidecar_fails_explicitly_for_execution_consumers` 报 `RuntimeError not raised`），非 CEVA 正例通过；修复后 3 条全部通过。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_compiler_placement.py`（254 条 OK）、`uv run python tests/sim/test_collective_edges.py`（21 条 OK）。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1270 条（含 1 条跳过）、0 失败（round30 同一口径为 1267 条，本轮新增 3 条）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1275 通过、1 跳过、0 失败（74.18 秒；round30 为 1272 通过、1 跳过，本轮新增 3 条）。
- genesim 通用预测器与 UPMEM：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过；均 0 失败，说明通用 `sim.yaml` 与 UPMEM 路径按次读盘的既有行为未变。
- 五份 CEVA 场景与真实 sidecar 复验：复跑 `/tmp/ceva_review_round31_inventory.py`，`structure`/`simulation` 全部为 true，实际创建核数 1/64/64/128/128、物理 engine 4/256/64/512/128、GPU 均为 0、容量均为 8,000,000,000 B、预留 0 B、后端均为 `CEVANPMPIMCore`；真实 sidecar 版本 4、6,850 条算子记录引用 21 份 PIM MLIR，缺失 0、哈希不符 0，8 台 DPU 映射互不重复。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round31-fix-smoke.yaml`（结果目录 `/tmp/ceva-round31-fix-results`、轨迹目录 `/tmp/ceva-round31-fix-traces` 为本轮新建），退出码 0。7,138 条轨迹全部重新编译（0 复用），创建/使用/空闲核 64/8/56，`total_time_s=672.363454`、`throughput_tokens_per_s=1.182396`、`processed_tokens=795`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round30 冒烟完全一致（固定输入下执行视图与直接读盘内容相同）。`run_manifest.json` 的 3 项输入与 21 份 PIM MLIR 逐份复算 SHA256 全部一致（缺失 0、不符 0）。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，45 通过、0 失败。本轮未修改 compiler 仓库源码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None,None):6626,(512,128):320,(512,64):64,(512,172):64}`；仿真 `total_time_s=1812.037482`、`tokens/s=4.461276`、`processed_tokens=8084`、`completed_requests=10`、`simulation_backend=pipeline`，与 round30 完全一致，说明普通非 CEVA 路径的按次读盘行为未被改变；通用 `summary.json` 不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round31 增补）：

- `_execution_sidecar()` 是调度器内新增的执行边界，后续新增放置 sidecar 消费者必须走它；本轮没有做静态约束禁止直接打开该路径，仍依靠代码审阅与用例守卫。
- `main.py::validate_ceva_toolchain` 在调度器创建前按磁盘读取 sidecar 与 PIM MLIR 做工具链预检，`run_manifest.build_ceva_run_manifest` 的无快照分支供独立工具使用；两处不参与执行图构建。若预检与调度器首次读取之间文件被改写，预检放行的是旧版本，但调度器快照与结束复核会拒绝本次执行结果。
- 本轮未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 32 轮（round32，2026-11-08）：

- 探针脚本：评审提供的 `/tmp/ceva_review_round32_same_path_reload.py`、`/tmp/ceva_review_round32_in_place_cost.py`、`/tmp/ceva_review_round32_collective_numeric.py`、`/tmp/ceva_review_round32_mapping_nan.py`。问题 1 修复前同路径重载时延停在 `5.28874e-05 s`（正确 `1.057478e-04 s`）、`same_object_after_rewrite=true`，原位把 `LUT` 从 6 改成 60 后仍为 `7.10506e-05 s`（正确 `6.654826e-04 s`）；修复后分别为 `1.057478e-04 s`、`reloaded_template_width=128`、`same_object_after_rewrite=false` 与 `6.654826e-04 s`、`stale_after_in_place_mutation=false`。问题 2 修复前 `nbytes=true` 生成两条各 `0.70025 µs` 的边（合法 4096 B 为 `1.724 µs`）、`4096.9` 被截成 4096、`seq_len=-16` 生成负系数、`seq_len=0.5` 抛未包装的 `ZeroDivisionError`，且全部 `simulation_valid=true`；修复后四类反例均 `simulation_valid=false` 并在首次消费 sidecar 时抛出字段化 `ValueError`，合法 4096 B 与 `seq_len=null` 的对照保持。问题 3 修复前 NaN 权重使 `_per_stage_weight_bytes()` 返回 NaN、`build_mapping(one_to_one=True)` 返回合法外观的 `[[0,0]]`、CLI 退出码 0 并写出 `model_id=nan-weight`；修复后探针在聚合处以“权重常驻量必须是有限数值，收到 nan”退出，CLI 不写计划，有限权重与超容量 1 B 的对照不变。
- 问题 1 的 TDD 失败复现：`uv run python tests/sim/test_pim_trace_loader.py` 修复前 1 条失败（`test_same_path_rewrite_returns_new_content` 仍返回同一旧对象）；`uv run python tests/sim/test_vpu.py` 修复前 3 条失败（两种装载入口的同路径重写与原位成本均返回旧值）。修复后分别为 14 条、57 条全部通过；同路径用例同时断言 `execute()`、`estimate()`、`execute_profile()` 的时延、memory_time 与搬运字节，并断言后端与核心指向同一新轨迹。
- 问题 2 的 TDD 失败复现：`uv run python tests/sim/test_collective_edges.py` 修复前新增用例出现 34 个失败与 5 个错误（畸形 nbytes/seq_len/dpu_ids、未知语义、未映射 DPU 未被拒绝，配置校验仍返回 true）；`python -m pytest tests/test_placement_export.py -q -k collective` 修复前 7 个参数化反例未抛错。修复后 30 条与 10 条全部通过；覆盖布尔/浮点/NaN/Inf/负数/字符串、两类集合通信、配置校验、图生成与完整流水。
- 问题 3 的 TDD 失败复现：`uv run python tests/sim/test_export_fixed_pu_mapping.py` 修复前 NaN/Inf/字符串权重未被拒绝（`SystemExit not raised` 或未包装的 `TypeError`）；修复后 24 条全部通过，覆盖公开函数、一对一与通用分支、CLI 以及“非法输入在写文件前失败”。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_pim_trace_loader.py`（14 条 OK）、`test_vpu.py`（57 条 OK）、`test_collective_edges.py`（30 条 OK）、`test_export_fixed_pu_mapping.py`（24 条 OK）。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1290 条、1 条跳过、0 失败（round31 同一口径为 1270 条，本轮新增 20 条：test_pim_trace_loader +1、test_vpu +2、test_collective_edges +9、test_export_fixed_pu_mapping +8）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1295 通过、1 跳过、0 失败（75.58 秒；round31 为 1275 通过、1 跳过，本轮新增 20 条）。
- genesim 通用预测器与 UPMEM：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过；均 0 失败，说明通用 `sim.yaml` 与 UPMEM 路径的既有行为未变。
- 五份 CEVA 场景与真实 sidecar 复验：复跑 `/tmp/ceva_review_round31_inventory.py`，`structure`/`simulation` 全部为 true，实际创建核数 1/64/64/128/128、物理 engine 4/256/64/512/128、GPU 均为 0、容量均为 8,000,000,000 B、预留 0 B、后端均为 `CEVANPMPIMCore`；真实 sidecar 版本 4、6,850 条算子记录引用 21 份 PIM MLIR，缺失 0、哈希不符 0，8 台 DPU 映射互不重复。本轮新增的集合通信契约对该真实 sidecar 未报任何错误。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round32-fix-smoke.yaml`（结果目录 `/tmp/ceva-round32-fix-results`、轨迹目录 `/tmp/ceva-round32-fix-traces` 为本轮新建），退出码 0。7,138 条轨迹全部重新编译（0 复用），`total_time_s=672.363454`、`throughput_tokens_per_s=1.182396`、`processed_tokens=795`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round31/round30 冒烟完全一致，证明内容哈希缓存与集合通信契约没有改变合法输入的代价语义。`run_manifest.json` 的 3 项输入与 21 份 PIM MLIR 逐份复算 SHA256 全部一致（缺失 0、不符 0），`hardware` 为 64 核 / 4 engine / GPU 0 / 8,000,000,000 B。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，54 通过、0 失败（其中 `-k collective` 为 10 条）。
- G 扩展回归：执行 `python -m pytest tests/ -q -k "not llama2_7b"`，1681 通过、8 跳过、49 排除、0 失败（18 分 21 秒）。全量运行开始后只再收紧两处同类校验（生产端 `dpu_ids` 先做类型检查再排序、生产端与配置契约拒绝负 DPU 编号）并追加相应参数化用例；受影响文件已在最终代码上重跑：`tests/test_placement_export.py` 54 条、genesim `tests/sim/test_collective_edges.py` 30 条均通过，未再执行第二次全量。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7,138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None,None):6626,(512,128):320,(512,64):64,(512,172):64}`；仿真 `total_time_s=1812.037482`、`tokens/s=4.461276`、`processed_tokens=8084`、`completed_requests=10`、`simulation_backend=pipeline`，与 round31 完全一致，说明内容哈希装载与集合通信契约没有改变通用链路的代价语义。本轮重导出的 placement sidecar 含 118 条集合通信记录（`all_gather`/`scatter`），`nbytes` 全为正整数、`seq_len` 全为 null 或正整数、`dpu_ids` 全部唯一，通用调度器首次消费与补边阶段未报未映射 DPU；通用 `summary.json` 仍不含 `result_maturity`/`run_manifest` 键。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round32 增补）：

- `load_trace_file()` 每次命中缓存都要读取并哈希整份轨迹文件；当前 7,138 条约 29 MB，开销可忽略，但超大规模轨迹库会多一次顺序读。内容哈希换来的是“同路径覆盖必然换身份”，不再依赖 mtime 或调用次数。
- 集合通信契约在配置校验阶段只能校验字段类型与取值范围；`dpu_ids` 到执行资源的映射需要实际 VPU 集合，仍在调度器建图阶段核对，配置校验无法单独完成该项。
- 生产端只对实际写入 sidecar 的 `all_gather`/`scatter` 做契约校验；`all_reduce`/`local_slice` 在镜像过滤阶段就被排除。将来若 sidecar 增加新语义，需要同步扩展 `COLLECTIVE_EDGE_TYPES`/`COLLECTIVE_SKIPPED_TYPES`，未登记的语义会显式失败而不是静默跳过。
- 容量契约允许有限浮点容量参数用于门禁比较；映射计划本身只写整数 DPU 映射，容量数值不下发，因此浮点容量的精度不影响产物格式。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

第 33 轮（round33，2026-11-09）：

- 探针脚本：评审提供的 `/tmp/ceva_review_round33_collective_extremes.py` 与 `/tmp/ceva_review_round33_collective_routes.py`（后者换用 `scatter` 及 `direct`/`host_relay` 路由复核）。修复前：`nbytes=10**1000、seq_len=null` 在补边的 `float(nbytes)` 处抛未包装的 `OverflowError: int too large to convert to float`；`nbytes=10**1000、seq_len=16` 在除法处抛 `OverflowError: integer division result too large for a float`；`nbytes=4096、seq_len=10**1000` 被接受并生成两条各 0.0 B、路由时延 0 的边；三类场景的 `validate(mode="simulation")` 均为 true。修复后：合法对照不变——`4096` 常量两条边各 `4096 B`、系数 `{"constant": 4096.0}`，`4096/16` 两条边各 `256 B`、系数 `{"Tq": 256.0}`，均 `simulation_valid=true`、`zero_byte_fast_path=false`；三类极值全部 `simulation_valid=false`，首次消费 sidecar 时按字段抛 `ValueError`（`nbytes 超出浮点可表示范围`、`seq_len 超出浮点可表示范围`）；`all_gather`/`scatter` × `direct`/`host_relay` 四类组合复核结论一致。`nbytes=4096、seq_len=10**100` 的对照仍合法：系数 `4.096e-97` 是有限正数，实际传输向上取整为 1 B，不会免费。
- 问题 1 的 TDD 失败复现：`uv run python tests/sim/test_collective_edges.py` 修复前新增用例出现 6 个失败、5 个错误（大整数路径抛未包装 `OverflowError`、下溢路径未报错、配置校验仍返回 true、完整流水未在仿真前拒绝）；修复后 32 条全部通过。`python -m pytest tests/test_placement_export.py -q -k collective` 修复前 2 个失败、11 个通过（`nbytes=10**1000` 未抛错、`_collective_number_errors` 尚不存在）；修复后 13 条全部通过。
- genesim 受影响文件（uv 虚拟环境）：`uv run python tests/sim/test_collective_edges.py`（32 条 OK）、`test_config_loader.py`（116 条 OK）、`test_compiler_placement.py`（254 条 OK）。
- genesim 官方入口：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 TMPDIR=/tmp ./run.sh --test sim`，46/46 个测试文件通过，单文件 `Ran N tests` 合计 1292 条、1 条跳过、0 失败（round32 同一口径为 1290 条，本轮新增 2 条：`test_collective_edges.py` +2）。
- genesim 全量（PyTorch 环境）：先 `source /data/fjg/flagOS/flagOS-installed/pytorch/env-pytorch.sh`，再执行 `PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/sim -q`，1297 通过、1 跳过、0 失败（77.36 秒；round32 为 1295 通过、1 跳过，本轮新增 2 条）。
- genesim 通用预测器与 UPMEM：`PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 ./run.sh --test predictor` 为 7/7 个文件、87 条通过；`./run.sh --test upmem_checker` 为 49 条通过；均 0 失败，说明通用 `sim.yaml` 与 UPMEM 路径的既有行为未变。
- 五份 CEVA 场景与真实 sidecar 复验：复跑 `/tmp/ceva_review_round31_inventory.py`，5 份场景 `structure`/`simulation` 全部为 true，实际创建核数 1/64/64/128/128、物理 engine 4/256/64/512/128、GPU 均为 0、容量均为 8,000,000,000 B、预留 0 B、后端均为 `CEVANPMPIMCore`；真实 sidecar 版本 4、6,850 条算子记录引用 21 份 PIM MLIR，缺失 0、哈希不符 0，8 台 DPU 映射互不重复；118 条集合通信记录 `nbytes` 范围 64～1,024,000 B、`seq_len` 全为 null 或 16，新契约未报错。
- CEVA 64 核端到端冒烟：在 `/data/fjg/flagOS/genesim` 执行 `PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1 uv run python -u src/main.py --config /tmp/ceva-round33-fix-smoke.yaml`（结果目录 `/tmp/ceva-round33-fix-results`、轨迹目录 `/tmp/ceva-round33-fix-traces` 为本轮新建），退出码 0。7,138 条轨迹全部重新编译（0 复用），创建/使用/空闲核 64/8/56，`total_time_s=672.363454`、`throughput_tokens_per_s=1.182396`、`processed_tokens=795`、`result_maturity=model_prediction`、`simulation_backend=pipeline`，与 round32/31/30 冒烟完全一致。`run_manifest.json` 的 3 项输入（模型 IR、放置 sidecar、请求 trace）与 21 份 PIM MLIR 逐份复算 SHA256 全部一致（缺失 0、不符 0），`hardware` 为 64 核 / 4 engine / GPU 0 / 8,000,000,000 B；`communication_metrics.json` 的 PIM_INTER 正向/反向字节为 539,869,184 / 312,213,504 B、`total_transfer_time_s=0.776152`，说明真实 sidecar 的集合通信仍按正字节计入服务时间。
- G 快速回归：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `python -m pytest tests/test_export_pp_placement_budget.py tests/test_ir_cost_eval.py -q`，36 通过、4 跳过、0 失败；另跑 `python -m pytest tests/test_placement_export.py -q`，57 通过、0 失败（round32 为 54 条，本轮新增 3 条：`nbytes=10**1000` 参数化 1 条、`test_collective_number_contract_matches_the_cost_model`、`test_export_accepts_large_representable_collective_numbers`）。
- G 扩展回归：执行 `python -m pytest tests/ -q -k "not llama2_7b"`，1685 通过、8 跳过、49 排除、0 失败（18 分 04 秒）。round32 全量同一口径为 1681 通过，差值 4 = 本轮新增 3 条 + round32 在全量开始后追加、未计入其 1681 基线的参数化用例 1 条（round32 文档已注明未重跑第二次全量）。本轮全量在最终代码状态上执行，未再改动源码。
- 全链路：在 `/data/fjg/flagOS/flagos-pim-compiler` 先 source `paths.json` 指定的环境，再执行 `PYTHONDONTWRITEBYTECODE=1 python -u scripts/run_full_pipeline.py --num-stages 4`，退出码 0，输出“全流程验证通过”。图编译 6852 个算子、224 个 GEMM；算子编译 6626 个算子级节点（21 种本地形状、分块 512）；7,138 条 trace 全部来自 `pimir`（`template` 为 0），`(tile_n, k_iterations)` 分布为 `{(None,None):6626,(512,128):320,(512,64):64,(512,172):64}`；仿真 `total_time_s=1812.037482`、`tokens/s=4.461276`、`processed_tokens=8084`、`completed_requests=10`、`simulation_backend=pipeline`，与 round31/32 完全一致，说明生产端新契约没有改变合法产物的代价语义。本轮重导出的 placement sidecar 含 118 条集合通信记录（`all_gather`/`scatter`，`nbytes` 范围 64～1,024,000 B、`seq_len` 全为 null 或 16），新契约全部通过；E 段 `communication_metrics.json` 的 `total_bytes=479,072,045,432 B`、`total_transfer_time_s=19.667897`，其中 PIM_INTER 字节 396,853,248 B、PIM_EXTERNAL 收发 4,233,133,608 / 23,134,397,520 B，集合通信仍按正字节计入服务时间。通用 `summary.json` 仍不含 `result_maturity`/`run_manifest` 键。
- 历史同类修复复核：复跑第 32 轮三个探针，结论保持——`/tmp/ceva_review_round32_same_path_reload.py` 两个装载入口同路径重载均为 `1.057478e-04 s`（`reloaded_template_width=128`、`same_object_after_rewrite=false`）；`/tmp/ceva_review_round32_in_place_cost.py` 原位改 `LUT` 后为 `6.654826e-04 s`（`stale_after_in_place_mutation=false`）；`/tmp/ceva_review_round32_mapping_nan.py` 按预期以“权重常驻量必须是有限数值，收到 nan”退出，未写出计划。复跑第 31 轮 sidecar 改写探针 `/tmp/ceva_review_round31_sidecar_consumers.py`：`consumed_dpus_after_rewrite=[0,1]`、`cached_collective_count=118`、`relay_operators_added=254`、`snapshot_verify_after_restore=passed`；稳定输入对照 `/tmp/ceva_review_round31_sidecar_control.py` 为 `[0,1]`、118 条、254 个中转。第 31 轮 inventory 与真实 sidecar 复验见上。
- FlagTree：工作区干净（`git status --porcelain` 无输出，HEAD `0d7273bac`），未修改源码，不需要重新构建；本轮未涉及 F 源码。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。

仍存在的不足（round33 增补）：

- 数值契约只保证单个系数在进入成本模型时可表示为有限正数；运行期 `系数 × Tq` 仍可能溢出为 `inf`，由调度器 `dependency_transfer_bytes()` 的有限非负检查拒绝（不会静默计 0），这是最后防线而不是新的契约入口。
- 契约约束的是可表示性与正数性，不是物理合理性：例如 `nbytes=4096、seq_len=10**100` 的系数 `4.096e-97` 合法并按 1 B 向上取整计费；没有为 `seq_len` 增加与导出张量形状或真实序列上限相关的上界，避免编造甲方未确认的限制。
- 非法极值的错误消息会带完整任意精度整数，极端输入下日志较长；当前错误路径只用于拒绝，不影响正常产物。
- 未执行 `-k llama2_7b`（约 2.5 小时）和 `run_ceva_pipeline.sh`（依赖外部 CEVA 仿真器且约 10 小时）。
