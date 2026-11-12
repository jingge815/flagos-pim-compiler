# 实施进展文档：接入 Qwen3 MoE（Qwen3MoeForCausalLM）到 PIM 推理链路
> 文档编号：implement-qwen3-moe-20261105
> 创建日期：2026-11-05
> 关联需求文档：request-qwen3-moe-20261105
> 关联设计文档：design-qwen3-moe-20261105
> 关联评审文档：review-qwen3-moe-20261109-round01、review-qwen3-moe-20261109-round02、
> review-qwen3-moe-20261110-round03、review-qwen3-moe-20261110-round04、
> review-qwen3-moe-20261110-round05、review-qwen3-moe-20261110-round06、
> review-qwen3-moe-20261111-round07、review-qwen3-moe-20261111-round08、
> review-qwen3-moe-20261111-round09、review-qwen3-moe-20261111-round10、
> review-qwen3-moe-20261111-round11、review-qwen3-moe-20261111-round12、
> review-qwen3-moe-20261111-round13、review-qwen3-moe-20261111-round14、
> review-qwen3-moe-20261112-round15、review-qwen3-moe-20261112-round16、
> review-qwen3-moe-20261112-round17、review-qwen3-moe-20261112-round18、
> review-qwen3-moe-20261112-round19、review-qwen3-moe-20261112-round20（本轮依据）
> 关联技术文档：techdoc-qwen3-moe-20261112、qwen3-moe-final-20261112
> 关联项目：flagos-pim-compiler（改动跨 flagos-pim-compiler、genesim、
> flagOS-installers/FlagTree；FlagTree 修复 paged_attn 降级并重建安装前缀）
> 起始基线：feature @ 263fcda（工作区另有未提交前置批次：README.md、
> docs/qwen3-support-20261104.md、tests/README.md、tests/test_architectures.py
> 的 MoE 范围澄清，属本需求前置）

## 一、实施概述

### 1.1 任务范围
本轮覆盖需求 P0-1～P0-5、P1-1、P1-2，以及评审 round01 的 10 项、round02 的 7 项、
round03 的 5 项、round04 的 4 项、round05 的 5 项、round06 的 5 项、round07 的
7 项、round08 的 5 项、round09 的 2 项、round10 的 2 项、round11 的 3 项、
round12 的 2 项、round13 的 5 项、round14 的 6 项、round15 的 5 项、round16 的
4 项、round17 的 4 项、round18 的 6 项、round19 的 5 项（高 2、中 2、低 1）与
round20 的 2 项（高 1、低 1）问题。
范围跨三仓：
- flagos-pim-compiler：静态 MoE 前向（top-k 选中掩码与未选中专家非有限值掩码）、
  架构注册表驱动的主链、MoE 放置导出、动态分派子图与通信门控、图级分块开关、
  真实 30B 验收、执行语义统一契约、提交前分派、仓库内执行期基准、逐算子成本
  宽度与 B 路原语语义、逐条目豁免与专家身份校验、单一容量/对齐工厂、
  仿真阶段绑定本次放置产物、架构感知的切分校验、统一的严格产物验证器、
  配置与产物自报取并集的严格性上下文（贯穿五个复核入口）、启动预检复用
  真实 KV 布局、正文证据按运行标签与块级语境绑定；round16 把 B 路 pimir 固定为
  「1 token × 本地宽」编译单位并回传 sidecar，trace 验收按单位核对 token 数、
  形状与字节；MoE 身份（专家数/top-k/专家宽）改为从 HF config 派生后逐项核对；
  round17 把单位完整性接入共享严格验证器（导出、skip-simulation、freshness、
  仿真 IR 共用），单位形状保留符号维并在两仓只比较特征维；round18 把 A/B 路
  分类收回到源 IR（`op_type == GEMM` 才是 A 路，`shards` 自报不得改写类别），
  并让 B 路单位的数值直接与所引用 pimir 的 `pim.traceTokens`/特征末维绑定；
  round19 把 A 路 GEMM 的 pimir 语义/形状绑定到分片实产（分块 GEMM 原语、
  tile 属性与 K/N 循环、融合 `act_spec`），把 host 判定收成「源 IR 落点 +
  豁免登记 + `host_gemms` 证据」一条链，并把条目 `op_type`/`semantic_role`
  与源 IR 逐条绑定；round20 给 B 路补一张按真实编译实产校准的闭合
  `op_type → pimir 语义指纹`表，严格分支逐条核对顶层原语与关键属性，
  未登记类型直接拒绝，五入口共用；
- genesim：MoE 图骨架（router／专家／top-k／驻留／执行语义）、全图覆盖
  （lm_head/q_norm/k_norm）、严格 pimir 产物契约、MoE sidecar 元数据与结构指纹
  核对、TP 展开按专家分组并继承成本角色、批量图修改接口、summary/trace 携带
  MoE 元数据与身份、dense_sim／dynamic_dispatch 两种执行语义的 any-hit 代价
  模型与容量契约校验、依赖边按两端的专家角色折算通信；round16 增 B 路编译单位
  登记与校验、运行期按 trace token 单位折算重复次数、TP 展开只在分片本地形状
  与单位一致时继承 pimir，并让相位解析优先读显式 `pim.traceTokens`；round17
  把重复次数的唯一来源改为调度器绑定的实际 Tq（删除形状启发式），B 路单位形状
  保留 `Tq`/`Tp+Tq` 符号维以区分 token/历史/特征维，并刷新独立入口的共享产物、
  修正模板说明与输出目录；round18 在 GeneSim 按源 IR 校验条目类别（B 路假分片、
  严格产物设备 GEMM 空分片直接失败），并在生成 trace 时把 sidecar 单位与 pimir
  实产逐值核对；round19 在覆盖校验与 trace 单位校验复用同一 A 路规则，分页
  注意力主机回退显式落 f16 相位边界，容量利用率改成与 `fits` 同源并单列总量比；
  round20 在覆盖校验与 trace 加载用同一张 B 路语义指纹表（与编译器逐值钉住），
  并让 `summary.json` 显式输出单 TensorPU、集群常驻与诊断比三个 scope 字段；
- FlagTree：修复 `paged_attn` 的 PV 相位存储边界与块表槽号契约（主机预检 +
  设备兜底），更新相位表说明并补 EmitC lit 用例，重建 `triton-opt`/`libtriton`
  并同步到 FlagTree 与 PyTorch 安装前缀；round16/round19/round20 没有改
  FlagTree 源码。

### 1.2 完成标准
设计与需求任务项 100% 覆盖、对应测试全部通过、全量回归无新增失败；全链路产物
（IR → sidecar → pimir mlir → GeneSim trace）互相可追溯，任何静默回退都视为失败。

### 1.3 实施结论
代码修复状态：**完成**。round01～round20 的评审问题已逐条确认；round20 的
2 项（高 1、低 1）全部成立并修复，无未修复项；因达到用户设定的 20 轮上限，
round20 的修复尚未经第 21 轮独立评审复核（详见
techdoc-qwen3-moe-20261112 与 qwen3-moe-final-20261112）。round20 把 B 路 pimir 的语义
绑定从 round19 的 add/mul 扩成闭合的实产指纹表（18 类：必需顶层原语多重集、
允许集合与关键属性），两仓各持一份并用跨仓用例逐值钉住；编译器共享验证器
五入口与 GeneSim 覆盖校验/trace 加载同判，未登记类型直接拒绝，`phase_count`
与 `_UNIT_*` 集合不再是跳过语义核对的出口；`summary.json` 增加单 TensorPU
常驻、集群常驻与诊断比三个显式 scope 字段，文档统一按单 TensorPU scope
描述 `max_resident_bytes`。round19 问题3 的局部修复因此收敛；同时把
round18 问题3 遗留的缓存锁测试 fork/Queue 抖动改成 spawn + 模块级 worker
（本轮全量回归复现 `_queue.Empty`，只改等待上限没有根治）。复发原因与
其余取舍见 §7.2。
需求验收状态：**完成（4 层）**——4 层真实权重 **36 passed、1 skipped**
（242.71 s；8B 无权重 SKIP）、round20 重跑 4 层全链 dynamic_dispatch
**60.785 s / 13.079 t/s**（精化 IR 与 round19 产物逐字节相同，sha256
`43634aac…`）、dense_sim **63.637 s / 12.493 t/s**（1.047×，本轮 dense
重跑）、GeneSim **45/45** 个测试文件、编译器全量回归 **2207 passed、
11 skipped、52 deselected、失败 0**。48 层
数值沿用 round06 历史
**26 passed、1 skipped**；48 层主链 A/BCD/A2 为 round08 产物、E 为 round10
重跑。round16/17 的单位契约使旧 48 层 B 路条目缺少 `trace_tokens`/
`local_*_shapes` 与符号维，当前严格验证器在 freshness、仿真 IR、trace
三入口都拒绝该产物，48 层成本证据（`total_time_s=566.863`、
`tokens/s=1.402`）失效；按用户要求不重跑 48 层主链，保留为 K16/Q12。
8B 无权重记为未覆盖。
当前关键事实：
- A/B 路分类只由源 IR 决定：`op_type == GEMM` 进 A 路，其余算子都是 B 路；
  `shards` 是被验产物自报字段，不得反过来定义类别。B 路带假分片、设备 GEMM
  空分片在编译器共享验证器与 GeneSim 两仓直接失败；4L 真实产物 1557 条
  GEMM + 1606 条算子级条目分类不变。
- B 路单位以 pimir 为唯一真源：sidecar 的 `trace_tokens` 必须等于 pimir 的
  `pim.traceTokens`，`local_*_shapes` 的具体特征末维必须等于对应参数/结果
  末维（SPLIT 纯视图只比输入）；缺 `pim.traceTokens` 的 pimir 在严格产物中
  直接失败。4L 1606 条 B 路条目全部与 15 份 pimir 逐值一致。
- A 路 GEMM 与 B 路对称绑定：pimir 必须是该分片的实测分块 GEMM（单一
  `tt.dot`/`pim.matmul`、无 traceTokens/相位链），M/K/N 与 tile 属性、
  `kernel_tile_n`、分片本地宽、K/N 循环一致，融合 `act_spec` 与清单双向
  一致。round19 探针（全部 GEMM 指向 B 路产物、换分片宽度、删分块、融合
  缺 act_spec）在共享验证器与直接 GeneSim 全部 REJECTED；4L/共享 dense/
  llama2 真实 A 路产物 ACCEPTED。
- 条目按 host 跳过 A 路检查必须同时有源 IR `device_hint=cpu`、
  `pimir_exempt_ops` 登记与 GEMM 的 `host_gemms` 角色/权重证据；
  `operators` 自称 host 缺证据直接失败。round19 的 lm_head/router 探针在
  两仓全部 REJECTED，真实产物 0 条 host 条目不被误判。
- 条目 `op_type` 必须等于源 IR 类型，设备 GEMM 的 `semantic_role` 必须是
  投影身份；trace 字节判据按 IR 类型选择。B 路 pimir 的顶层原语与关键属性
  还必须与算子类型对应（round20）：两仓闭合表覆盖 18 类，ROPE 必须是带
  `rotateHalf`/`sourceSubBlocks` 的三相 `pim.eltwise` 链，四节点注意力按
  实际编译产物登记；未登记类型直接拒绝。编译器五入口（导出、
  skip-simulation、freshness、仿真 IR、trace）共用同一份判定；round18/19/20
  探针（假分片、`trace_tokens=128`、本地宽 384、旧 128-token pimir、无
  traceTokens pimir、设备 GEMM 空分片、A 路指向 B 路产物、host 自报、
  `op_type` 互换、B 路语义互换）全部 REJECTED，4L 新产物 ACCEPTED，
  48L/llama2 真实 B 路产物语义 0 问题。
- 主机分页注意力在 QK 点积、乘 scale 与每块 PV 乘积三处显式落 f16，再进
  f32 累加器；不依赖 `matmul` 返回 dtype。同时关闭 `_compiled_paged_attn`
  与 `_compiled_matmul` 的用例与相位表参考逐元素一致。
- 缓存锁测试的等待上限改为 60 秒死锁检测，teardown 显式 `join`/
  `terminate`/`kill`；新增慢调度用例固定复现旧 5 秒超时的 `_queue.Empty`。
- 实施证据 checker 的声明改为实际命令：只核 dynamic、dense 两份 current
  summary；alt/48l 是 history 正文引用。新增 CI 用例实际执行文档命令并断言
  退出码 0。
- 容量指标按 scope 拆开：判定利用率 `max_utilization` 与 `fits` 同源（模型常驻
  按 ClusterPU 校验时分子是 `request_bytes`，4L 为 0.30%），请求期、
  集群常驻 `max_cluster_resident_bytes=3,140,834,304 / 8,589,934,592`（36.6%）
  与诊断比 `max_total_resident_ratio` 分开报告；
  `max_resident_bytes=779,673,600` 是**单 TensorPU 常驻最大值**（含该
  TensorPU 的模型分片与请求期数据），`max_total_resident_ratio` 是它与
  单 TensorPU 上限 512 MiB 的诊断比，不参与 `fits`；集群总量只引用
  `max_cluster_resident_bytes`/`max_cluster_utilization`。
- 两个真实测试的 `HwBudget` 与 `PIMHardwareConfig` 都由
  `default_site_capacity()` 派生；新增静态守卫扫描手写 8 GiB/64 MiB/对齐/
  WRAM 字面量。
- 执行语义从编译配置统一派生，IR/sidecar/summary 三处独立核对；两种语义的
  结构指纹各自一致，trace 来源只有 `pimir`/`absorbed`。
- P1-1 用户已接受 4 层真实权重的 prefill 1.148×、decode 1.202×（旧实现
  0.994×），Q7 关闭；剩余执行后端优化另立需求。
- 剩余事项：48 层成本待在新单位契约下重跑（K16/Q12）；Llama2 旧 sidecar
  需用当前主链重导后跑慢速用例（K12）；8B 无权重（K4）；单专家真实 BCD 受
  FlagTree N=1 分块限制（K11/Q10）。

## 二、任务完成清单
| 任务ID | 任务描述 | 对应设计模块 | 完成状态 | 备注 |
|--------|----------|--------------|----------|------|
| T2 | 静态 MoE 前向改写 `make_static_moe_forward` | §4.1 | 完成 | 两种 norm_topk_prob 均位级一致；未选中专家溢出经 where 掩码 |
| T1 | 注册 `qwen3_moe` + `_moe_strategy` | §4.2 | 完成 | 防呆测试翻转为放行 |
| T3 | HOST_ONLY 增补 5 门控算子 | §4.3 | 完成 | softmax 用具体重载，防误伤 `_softmax` |
| T4 | `mlp.gate.weight` 列切规则 | §4.4 | 完成 | |
| T5 | 小随机 MoE 端到端对拍 | §4.6 | 完成 | 组合验证，见 3.3-I1 |
| T6a | 站点路径键 `qwen3_30b_model_dir` | §4.7 前置 | 完成 | 解析改为无权重单测，真实用例统一 skipif |
| T6b | 真实 30B 端到端用例 | §4.7 | 完成 | 4 层切片目标命令 round18 **36 passed/1 skipped（256.34s，8B SKIP）**；48 层结果见 8.4/K7/K16 |
| R1 | P1-1 动态分派原语 | §4.8 | 完成 | 子图/通信整条门控；跳过写零防 inf×0=nan |
| R2 | P1-2 专家并行维度 | §4.9 | 完成 | PP×EP（设计 §九.4 路线）；16 DPU 用例与 K3 守护 |
| F1 | GeneSim MoE 图骨架与元数据 | 评审问题1 | 完成 | router + 每专家三件套；54 GiB 专家驻留 |
| F2 | 架构注册表驱动主链 | 评审问题2 | 完成 | `--model-dir`；Qwen3 入口可复现 |
| F3 | 未选中专家溢出掩码 | 评审问题3 | 完成 | P0／P1 两态与 HF 稀疏前向对拍 |
| F4 | 严格 pimir 产物契约 | 评审问题4 | 完成 | 缺项/缺哈希/不可读/哈希不符均在编 trace 前失败 |
| F5 | MoE 图级分块开关 | 评审问题5 | 完成 | MoE 默认关闭；`PIM_DISABLE_TILE` 为受支持入口 |
| F6 | 独立 HF 参考 | 评审问题6 | 完成 | 保存并恢复 HF 原始稀疏 forward + 断言 |
| F7 | 8 GiB 容量口径 | 评审问题7 | 完成 | 用户确认 8 DPU × 8 GiB 硬约束；见 F20 |
| F8 | 动态分派子图/通信门控 | 评审问题8 | 完成 | provenance 穿透分块与布局中立节点 |
| F9 | 站点路径跳过保护 | 评审问题9 | 完成 | 架构单测不再依赖真实权重 |
| F10 | 实施文档收敛 | 评审问题10 | 完成 | 本文就地重写并补全证据 |
| F11 | 全模型容量口径统一 | round02 问题1 | 完成 | 统一到 `default_hw_budget()`（8 GiB），见 F20 |
| F12 | 执行语义统一契约 | round02 问题2 | 完成 | 编译配置 → 图骨架 IR、sidecar、GeneSim 同一份值；两种语义分别计费 |
| F13 | 产物契约闭合 | round02 问题3 | 完成 | 显式严格开关、fresh 缺哈希、未知算子、缺 MoE 元数据四条绕过全部失败 |
| F14 | VECTOR_MUL 原语语义 | round02 问题4 | 完成 | B 路按算子类型发 add/mul；导出与仿真两侧核对 kind |
| F15 | 逐算子成本宽度 | round02 问题5 | 完成 | 专家乘/激活按实际 768，残差加按 hidden |
| F16 | 动态分派执行期加速 | round02 问题6 | 完成 | 提交前跳过 + 仓库内基准；4 层真实权重 1.148×/1.202×，用户已接受（Q7 关闭） |
| F17 | 实施证据收敛 | round02 问题7 | 完成 | 本文就地重写；48 层历史结果与 4 层本轮结果分栏登记 |
| F18 | 图骨架全图覆盖（lm_head / q_norm / k_norm） | round03 问题1 | 完成 | IR 3681 算子、1557 GEMM；parameter_bytes 与 HF 参数逐字节一致；跨仓权重对拍 |
| F19 | 动态分派成本与执行语义一致 | round03 问题3 | 完成 | 专家计算/通信按 any-hit `1-(1-k/E)^Tq`；共享累加不缩放；成本角色进 IR；依赖边同时按源、目标角色折算（round10 补齐） |
| F20 | 容量契约单一真源（8 DPU × 8 GiB） | round03 问题2 | 完成 | `contracts/op_contract.py` + `default_hw_budget()`；sidecar 回传，GeneSim 核对；48 层余量记 K7 |
| F21 | 模型身份与结构指纹 | round03 问题4 | 完成 | IR/sidecar/精化 IR/仿真四处比对；A2 缓存键入指纹/档位/融合清单/成本版本 |
| F22 | 实施证据随确认与终态收敛 | round03 问题5 | 完成 | §1/§3-§9 就地更新；Q5/Q7 关闭；OOM 与 48 层未覆盖分栏登记 |
| F23 | 非 GEMM 成本依赖项按配方声明 | round04 问题1 | 完成 | 当前 token 类用 Tq、注意力类用 Tq(Tp+Tq)；ROPE/MASK 按逐头形状测量；成本版本 1.3 |
| F24 | 源 IR 驱动的产物覆盖与 trace 验收 | round04 问题2 | 完成 | 覆盖与落点无关；豁免按 IR 语义核对；MoE 三件套完整集合；trace 只收 pimir/absorbed 且按应覆盖集合对拍 |
| F25 | 结构指纹补拓扑与层归属 | round04 问题3 | 完成 | schema 2：依赖端点/张量身份 + subgraphs；成本/设备/本地分片字段继续排除 |
| F26 | 旧式依赖按专家端角色折算 | round04 问题4 | 完成 | 旧式 (src,dst) 与显式系数共用同一折算出口，逐值等价；round10 补齐按源专家角色折算 |
| F27 | 专家归属按层隔离 | round05 问题1 | 完成 | (层, 专家)/(层, None) 域；decode/prefill/分块/PP/EP 逐层断言；后端按 dispatch 记录放行/拦下 |
| F28 | MoE 尾部选择与设备侧专家子图对齐 | round05 问题2 | 完成 | `ne`/`where.ScalarOther` 归 host；设备侧只剩可门控专家子图；MOE_ACCUM 计入选择并区分首专家；跨仓角色对拍 |
| F29 | absorbed 结构契约与专家身份 | round05 问题3 | 完成 | 融合清单 + 主算子 pimir + IR 依赖边唯一证明；直接 GeneSim 按 (层,角色,专家) 核对 |
| F30 | 容量契约三件套统一校验 | round05 问题4 | 完成 | 总容量/系统预留/对齐三字段；导出、审计、trace 验收、直接 GeneSim 同一规则 |
| F31 | 实施结论按验收项生成 | round05 问题5 | 完成 | §1.3 拆分代码修复与需求验收；48 层历史/环境阻塞分栏 |
| F32 | top-k 选中掩码与路由权重分离 | round06/07 问题1 | 完成 | `selected_mask` 管命中与掩蔽，routing 只参与加权；dispatch 只读布尔掩码 |
| F33 | 逐条目豁免落点证据 | round07 问题2 | 完成 | 导出按 `spec.device` 写 IR/sidecar 证据，五处入口逐条核对 |
| F34 | 容量/对齐单一工厂 | round06/07 问题2/3 | 完成 | `default_site_capacity()` 派生 HwBudget+硬件；A/B 两路与 pimir 属性核验 |
| F35 | 逐 op_id 专家身份与权重 | round06/07 问题3/4 | 完成 | 双仓按 op_id 核对 (层, 角色, 专家, 权重)，集合比较废弃 |
| F36 | 批量图修改接口 | round07 问题5 | 完成 | `ModelIR.batch_edit()`；TP 展开与 relay 插入一次重建索引 |
| F37 | 单专家 MoE 判定 | round06/07 问题5/7 | 完成 | model_type+有效字段；1 专家/top-1 全链通过 |
| F38 | 全链证据回填 | round06/07 问题4/6 | 完成 | 48 层数值/BCD 与 4 层本轮结果分栏；主链终态按实际状态登记 |
| F39 | 仿真绑定本次放置产物 | round08 问题1 | 完成 | E 阶段显式改写并复核 `compiler_placement_file`；策略/DPU/段数/Cluster/分片本地形状不符即失败；运行目录按架构/策略隔离 |
| F40 | 48 层当前产物主链闭环 | round08 问题2 | 完成 | 当前源码重跑 48 层 A→0→BCD→A2→E，重新生成 sidecar/pim mlir，逐段核对豁免/对齐/哈希/容量/指纹/summary/trace |
| F41 | 架构感知切分校验 | round08 问题3 | 完成 | `Architecture.strategy_check` 统一自动选择、直接段数与 PartitionPlan；专家整除不合法在加载后、图编译前拒绝 |
| F42 | 编译器侧统一严格产物验证器 | round08 问题4 | 完成 | `validate_sidecar_contract` 统一核对 MoE 元数据/指纹/容量/豁免/吸收/专家身份；导出、skip-simulation、freshness、仿真 IR、trace 共用 |
| F43 | 启动预检复用真实 KV 布局 | round08 问题5 | 完成 | 删除 `ALIGN=256`，预检直接调用 `build_kv_layout`；示例不再给 4 GiB 预算 |
| F44 | 吸收契约补消费者方向与阶段闭合 | round09 问题1 | 完成 | 主算子必须只有被吸收激活一个真实消费者；BCD 允许清单先行，A2 后所有复核入口与直接 GeneSim 要求清单/吸收编号双向闭合 |
| F45 | 豁免与 sidecar 放置交叉核对 | round09 问题2 | 完成 | host/边界豁免不得与 PIM 分片、pimir 或 `device_hint=pim` 并存；host GEMM 的 `host_gemms` 角色/权重身份逐项核对；五入口共用 |
| F46 | GeneSim 测试入口守卫修正 | round09 同类路径 | 完成 | 5 个测试文件的 `unittest.main()` 移到文件末尾，后续测试类恢复在 `./run.sh --test sim` 中执行 |
| F47 | 依赖边按两端的专家角色折算 | round10 问题1 | 完成 | 专家产出到共享/HOST 随源命中并集，共享输入到专家随消费者，shared→shared 全量；放置估算与正式管线共用；四方向与 TP 展开用例 |
| F48 | 实施证据自动一致性检查 | round10 问题2 | 完成 | `scripts/check_implement_evidence.py` 以 summary 为唯一数值来源核对文档；历史值显式声明；5 条单测 |
| F49 | 直接 GeneSim 容量按需求常量验收 | round11 问题1 | 完成 | `DPU_MRAM_BYTES=8 GiB` 独立常量；严格路径同时核对 sidecar、有效配置与 MLIR 属性；旧式非严格产物保留其它容量兼容 |
| F50 | 全流程运行身份与输出命名空间 | round11 问题2 | 完成 | 运行目录按架构/策略/语义/运行号派生；A/A2/BCD 产物、E 的 trace/results/log 与临时配置全部落本次目录；删除共享 `pim_traces` 清空 |
| F51 | 实施证据结构化归属与派生重算 | round11 问题3 | 完成 | 文档嵌入证据块；检查器逐标签绑定 summary、核对层数/语义、重算派生值；history/measurements 独立命名空间 |
| F52 | 编译器复核入口的严格性上下文 | round12 问题1 | 完成 | 共享验证器新增 `strict_pimir`，与 sidecar 自报取并集；导出、skip-simulation、freshness、仿真 IR、trace 五入口统一传入；配置显式严格时删自报/改 4/16 GiB 全部拒绝 |
| F53 | 正文证据归属与比较结论 | round12 问题2 | 完成 | 证据块 `prose_aliases` + 块级前置标签绑定；必需值只扫正文；带数值的大小结论按实际值判定；只换正文、层级冒充、历史冒充、错误比较全部失败 |
| F54 | PV 相位存储边界 | round13/14 问题1 | 完成 | 每块 PV 乘积先落 f16 再加进 f32 累加器；相位边界表统一 lowering/主机参考/测试；单块、多块尾块与不同 RNE 数据逐元素对拍 |
| F55 | 块表槽号不变式 | round13/14 问题2 | 完成 | 主机按 `0 <= slot < pool_slots` 预检本次历史实际读取的项；写 KV 与主机兜底共用 `_physical_slot`；设备 `pim_block_slot` 兜底中止；边界与非法项全覆盖 |
| F56 | 严格入口必需输入 | round13/14 问题3 | 完成 | 配置解析与严格性判定先于兼容返回；配置/IP 键与文件、sidecar 键与文件任一缺失在三个入口直接失败；非严格旧式用法保留 |
| F57 | trace 编译签名绑定 | round13/14 问题4 | 完成 | 逐条核对 `(parent_op_id, op_type, pimir_sha256)` 与源 IR/sidecar；absorbed 零指令且由融合结构证明；真实 4L/48L 产物正例 |
| F58 | 证据逐值归属与 facts | round13/14 问题5 | 完成 | 最近前置标签归属、前者/后者身份、必需值运行语境；`facts` 声明容量/通信/trace 计数并绑定运行或独立命名空间 |
| F59 | round13 登记与 §8.3 收敛 | round14 问题6 | 完成 | round13 五项与 round14 六项进 §7.1；§1.3 由未关闭项生成；§8.3 改为实际规则；GeneSim 命中口径注释同步 |
| F65 | B 路 pimir 成本单位契约 | round16 问题1 | 完成 | B 路 pimir 按「1 token × 本地宽」编译，sidecar 回传 `trace_tokens`/`local_*_shapes`，pimir 带 `pim.traceTokens`；GeneSim 校验单位、TP 展开同形继承、运行期 `ceil(Tq/tokens)` 重复；trace 验收与 SPLIT 视图/TRANSPOSE 维序一并覆盖 |
| F66 | MoE 身份从 HF config 派生 | round16 问题2 | 完成 | A 段与 BCD 段逐项核对专家数/top-k/专家宽；非 MoE 配置要求全零 |
| F67 | bool 类型注释与命中口径一致 | round16 问题3 | 完成 | `selected_mask` 语义写进 `contracts/dtypes.py` 并加静态测试 |
| F68 | B 路重复次数绑定实际 Tq | round17 问题1 | 完成 | 调度器写 `runtime_query_tokens`，VPU 删除形状推断；TRANSPOSE/SPLIT/GATHER/PAGED_ATTN 用例覆盖 |
| F69 | B 路单位保留符号维 | round17 问题2 | 完成 | sidecar 单位形状保留 `Tq`/`Tp+Tq`；两仓只比较具体特征维，字节按 trace 的 token/块 token 解析符号 |
| F70 | 单位完整性进共享严格验证器 | round17 问题3 | 完成 | `audit_sidecar_coverage` 严格分支核对完整性/符号维角色；导出、skip-simulation、freshness、仿真 IR 共用 |
| F71 | 独立入口共享产物与模板收敛 | round17 问题4 | 完成 | 刷新 `genesim/models` 为当前 dynamic 产物；模板改为说明 + 独立输出目录；产品一致性用例与预检提示 |
| F72 | A/B 路分类取自源 IR | round18 问题1 | 完成 | 两仓统一「GEMM 走 A 路、其余算子走 B 路」；`shards` 自报不能改写类别；B 路假分片、设备 GEMM 空分片直接失败；六处消费点全部改用同一规则 |
| F73 | B 路单位与 pimir 实产绑定 | round18 问题2 | 完成 | 共享验证器按路径解析一次 `pim.traceTokens` 与参数/结果形状；`trace_tokens`、`local_*_shapes` 特征末维必须与 pimir 一致；GeneSim 生成 trace 时同规则复核 |
| F74 | 缓存锁测试移除固定短超时与残留子进程 | round18 问题3 | 完成 | 队列等待上限改为 60 秒死锁检测；`finally` 中 join/terminate/kill 子进程；新增慢调度回归用例 |
| F75 | 实施证据 checker 声明按实际命令书写 | round18 问题4 | 完成 | 只声明 dynamic、dense 两份 current summary；alt/48l 标为 history 正文引用；新增执行文档命令的 CI 用例 |
| F76 | 4L 容量引用按集群口径统一 | round18 问题5 | 完成 | 三处引用改用 `max_cluster_resident_bytes`/`max_cluster_utilization`；per-TensorPU 最大值单列；证据块补 4L 集群/单 PU 容量 facts |
| F77 | 真实测试容量三件套单一工厂 | round18 问题6 | 完成 | 8B/30B 测试改用 `default_site_capacity()`；新增静态守卫扫描手写容量/对齐/WRAM 字面量 |
| F78 | A 路 GEMM 的 pimir 语义/形状绑定 | round19 问题1 | 完成 | 共享验证器与 GeneSim 按源 IR 逐条核对分块 GEMM 原语、tile/循环与分片宽度、融合 `act_spec`；B 路产物不能冒充 A 路 |
| F79 | host 判定收成证据链 | round19 问题2 | 完成 | 条目按 host 跳过 A 路检查需源 IR `device_hint=cpu` + 豁免登记 + `host_gemms` 角色/权重证据；`operators` 自称 host 缺证据直接失败 |
| F80 | 条目字段与源 IR 逐条绑定 | round19 问题3 | 完成 | `op_type` 必须等于源 IR 类型，设备 GEMM 的 `semantic_role` 必须是投影身份；trace 字节判据按 IR 类型选择；逐元素 pimir kind 与 IR 类型对应 |
| F81 | 主机分页注意力显式落 f16 边界 | round19 问题4 | 完成 | QK 点积与乘 scale 后落 f16、每块 PV 乘积落 f16 再进 f32 累加；不依赖 `matmul` 返回 dtype；补双编译内核缺席用例 |
| F82 | 容量利用率与判定同源 | round19 问题5 | 完成 | `max_utilization` 用与 `fits` 相同的判定分子；请求期利用率与集群常驻利用率分开输出，总量比单列 `max_total_resident_ratio` |
| F83 | B 路 pimir 语义指纹 | round20 问题1 | 完成 | 两仓各持一张按真实实产校准的闭合 `op_type → 顶层原语/关键属性` 表；编译器五入口与 GeneSim 覆盖/trace 加载同判，未登记类型拒绝；跨仓测试逐值钉住 |
| F84 | 容量 scope 字段与文档口径 | round20 问题2 | 完成 | summary 显式输出单 TensorPU 常驻、集群常驻与诊断比三字段；实施/设计文档统一按 scope 描述，不再把单 TensorPU 最大值写成跨 TensorPU 总量 |
| F85 | 缓存锁测试改用 spawn | round20 复发复核 | 完成 | 全量回归中 `test_cache_key_lock_serializes_processes` 以 `_queue.Empty` 超时；round19 的 60 秒上限只改了等待时间，没有消除 fork 继承套件状态的根因。两个锁用例改用 spawn + 模块级 worker，子进程自行设置缓存目录 |

## 三、代码改动详情

### 3.1 改动总览
- 新增文件数：8（`genesim/conf/sim_qwen3_moe_pimir.yaml`；`scripts/bench_moe_dispatch.py`；
  `genesim/tests/sim/test_moe_execution_cost.py`；`genesim_bridge/ir_fingerprint.py`；
  `tests/test_ir_fingerprint.py`；`scripts/check_implement_evidence.py`；
  `tests/test_implement_evidence.py`；`FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir`）
- 修改文件数：78（代码/测试 72：flagos-pim-compiler 50、genesim 20、
  FlagTree 2；文档 6：设计、实施、`tests/README.md` 与 3 份 vllm 默认值文档。
  其中 `tests/test_implement_evidence.py` 是前几轮新增文件的本轮修订，
  计入修改数是为了让本轮触达文件可追溯）
  round16 新触达 `genesim/tests/sim/test_pimir_trace.py` 与 `test_vpu.py`；
  round18 新触达 `tests/test_qwen3_8b_real.py`，并修改设计文档的 A/B 分类与
  单位真源契约；round20 新触达 `genesim/tests/sim/test_scheduler.py`、
  `tests/test_implement_evidence.py` 与 `tests/test_opcompiler_linear.py`）
- 删除文件数：0
- 核心改动：静态 MoE 前向与注册表主链、严格产物契约、执行语义贯穿三处、
  提交前动态分派与仓库内执行期基准、B 路原语语义与逐算子成本宽度；
  round03 增加全图覆盖（lm_head/q_norm/k_norm）、单一容量契约、any-hit 成本
  语义、模型结构指纹与精化缓存键；round04 把成本依赖项收进配方声明、覆盖
  校验改为源 IR 驱动、指纹补依赖拓扑与层归属、旧式依赖统一折算出口；
  round05 把专家归属限定到本层、把路由选择归回主机共享工作、为 absorbed
  与容量三件套补结构契约；round06/07 把 top-k 选中掩码与路由权重分离、
  逐条目核对豁免落点与专家权重身份、用单一工厂派生容量与对齐、为 TP 展开
  增加批量图修改接口、把单专家按合法 MoE 处理；round08 把仿真阶段绑定到
  本次 BCD 返回的 placement（策略/DPU/段数/Cluster/本地形状逐项复核）、
  把架构切分约束扩展到直接段数与 PartitionPlan、用统一的严格产物验证器
  收口五个复核入口、让启动预检直接复用真实 KV 布局，并重跑 48 层当前源码
  主链；round09 给吸收契约补上「主算子唯一消费者」方向与阶段化双向闭合，
  给豁免契约补上「与 sidecar 实际放置及 `host_gemms` 证据交叉核对」，
  并修正 5 个 GeneSim 测试文件的入口守卫，使后置测试类重新进入回归；
  round10 把依赖边折算改为同时看源、目的算子的专家角色，并新增实施证据
  checker，把文档中的时间/吞吐/有效 FLOPs 与 summary 产物做机器核对；
  round11 把 GeneSim 的严格容量验收固定到需求常量（不再从可配置容量反推），
  给全流程建立按架构/策略/语义/运行号派生的输出命名空间并删除共享
  `pim_traces` 清空，把实施证据改成结构化块（逐标签绑定 summary、重算
  派生值，history/measurements 独立命名空间）；round12 把严格性上下文下沉到
  编译器共享验证器（配置与产物自报取并集，五个入口共用），并给正文证据补
  运行标签绑定、必需值正文扫描与数值比较判定；round13/14 在 FlagTree 修复
  `paged_attn` 的 PV 相位取整与块表槽号兜底，在编译器补主机块表预检与
  「配置/IR/sidecar 是严格入口必需输入」，在 trace 验收补编译签名哈希与
  absorbed 空 trace 的结构闭合，在证据 checker 把正文数值改成最近标签
  归属、前者/后者身份核对、必需值运行语境与 `facts` 独立命名空间；
  round15 把 pimir 产物身份（可读 + 哈希）变成严格入口的固定规则并删除
  可选开关，把 trace 验收的配置改为必需输入，把请求分布 trace 纳入主链
  生成与回读，修正 §8.4 的证据来源分栏，并清理 `where` 注册表的错位注释
  与不可达分支。
  round16 把 B 路 pimir 固定为「1 token × 本地宽」编译单位（含 MLP act/mul
  的 TP 本地宽），sidecar 回传 `trace_tokens`/`local_*_shapes`，GeneSim 按单位
  校验 trace 并在 TP 展开时只继承同形 pimir，运行期按 `ceil(Tq/trace_tokens)`
  重复；MoE 身份（专家数/top-k/专家宽）改为从 HF config 派生后逐项核对，
  并修正 bool 类型注释。
  round17 把 B 路 trace 的重复次数统一到调度器绑定的实际 Tq（删除形状推断，
  4L 的 TRANSPOSE/SPLIT/GATHER 不再把头数或 1 行当 token），sidecar 的编译
  单位形状保留 `Tq`/`Tp+Tq` 符号维，GeneSim 与编译器只比较两侧具体的特征维，
  字节核对按 trace 的 token/块 token 解析符号；单位完整性接入共享严格验证器，
  使 freshness、仿真 IR、skip-simulation 与 trace 四个入口对同一份 sidecar
  给出一致判定；刷新独立入口的共享 models 产物并修正模板说明与输出目录。
  round18 把 A/B 路分类收回到源 IR（`op_type == GEMM` 才是 A 路，`shards`
  只是被验产物字段），两仓的覆盖审计、单位检查、放置加载、TP 展开与 trace
  验收都改用同一规则；B 路单位数值与 pimir 的 `pim.traceTokens` 和特征末维
  绑定，GeneSim 生成 trace 时再核对一次；缓存锁测试改用 60 秒死锁检测上限并
  回收 fork 子进程；证据 checker 的声明改为只核两份 current summary；4L 容量
  引用改按集群口径，per-TensorPU 值单列；真实测试容量三件套收回到
  `default_site_capacity()` 并加静态守卫。
  round19 给 A 路 GEMM 补上与 B 路对称的编译单位（分块 GEMM 原语、tile 属性与
  K/N 循环、分片本地宽、融合 `act_spec`），把 host 判定收成「源 IR 落点 +
  豁免登记 + `host_gemms` 证据」，把条目 `op_type`/`semantic_role` 与源 IR
  逐条绑定，给主机分页注意力补显式 f16 相位边界，并把容量利用率改成与 `fits`
  同源、总量比单列。
  round20 把 B 路 pimir 的语义绑定从 add/mul 扩成闭合的实产指纹表（顶层原语
  多重集、允许集合与关键属性），两仓共用一张表并跨仓钉住；编译器五个入口与
  GeneSim 的覆盖校验/trace 加载同判，未登记类型直接拒绝；`summary.json`
  增加单 TensorPU 常驻、集群常驻与诊断比三个显式 scope 字段。
  具体行数随各轮累计，见各文件条目。

### 3.2 详细改动清单
| 文件路径 | 改动类型 | 改动内容简述 | 核心函数/类 | 对应任务ID |
|----------|----------|--------------|-------------|------------|
| backend/architectures.py | 修改 | 静态前向用 top-k 布尔选中掩码掩蔽并保留零权重选中专家、HF 原始 forward 保存/恢复、注册表描述符与 `architecture_of`；必需投影身份增 `lm_head`；MoE 策略要求 TP 宽度整除专家数；新增 `strategy_check`/`check_strategy` 供自动选择、直接段数与 PartitionPlan 共用 | `make_static_moe_forward`、`restore_hf_moe_forward`、`architecture_of`、`_moe_strategy`、`_moe_strategy_check`、`check_strategy` | T2/F2/F3/F6/F18/F32/F37/F41 |
| backend/hal_numpy.py | 修改 | 提交前等 dispatch 结果再决定是否入队；被跳过命令按 `skip_zero` 清零、主机命令返回零张量；`moe_dispatch` 运行期开关；按 (泳道, dispatch) 记录放行/拦下命令数 | `submit`、`_skipped_event`、`gate_stats`、`NumpyBackendConfig.moe_dispatch` | F8/F12/F16/F27 |
| contracts/dtypes.py | 修改 | 谓词张量 bool 类型（1 字节）；round16 把注释从旧口径 `routing != 0` 改为 top-k 选中索引生成的布尔 `selected_mask` | `BOOL_DTYPES`、`dtype_bytes` | F3/F67 |
| graph/opt_passes.py | 修改 | 图级分块独立开关 | `optimize_graph(graph_tiling=)` | F5 |
| graph/spec_prop.py | 修改 | 删除 `where.ScalarOther` 设备切分规则（选择归 host，只留 host 布局传播） | `_require_host_inputs`、RULE_TABLE | F3/F28 |
| runtime/compile.py | 修改 | `graph_tiling` 传参；按 MoE 身份（非专家数阈值）默认关分块、`PIM_DISABLE_TILE` 入口；`moe_execution` 编译配置与 `CompiledModel` 回传；拒绝 `hw.align != hardware.dma_align` | `graph_tiling_enabled`、`compile_llama2`、`CompiledModel.moe_execution` | F5/F12/F34/F37 |
| runtime/exec_plan_gen.py | 修改 | 专家归属按层隔离、整条子图与通信打标、主机跳过零值；执行语义解析与缺省规则；dispatch 只读 top-k 选中掩码；`is_moe_config` 统一 MoE 判定（1 专家合法）；按图边判定 `skip_zero` | `_expert_provenance`、`_node_layer_of`、`_emit_redistribute`、`_emit_moe_dispatch`、`is_moe_config`、`resolve_moe_execution`、`_skip_zero_nodes` | F8/F12/F16/F27/F32/F37 |
| runtime/kernels.py | 修改 | 删除设备侧 `ne.Scalar`/`where.ScalarOther` 内核注册（选择归 host），`where_kernel` 只服务 `where.self` | `where_kernel` | F3/F28 |
| genesim_bridge/op_classify.py | 修改 | MoE host 算子保留模板成本；逐元素/激活/RMSNorm 配方按算子实际宽度生成；配方声明成本依赖项（Tq / Tq(Tp+Tq)）；ROPE/MASK 按逐头节点形状测量 | `UNCOVERED_OP_TYPES`、`_binary`、`_activation`、`_normalize_ir`、`_rope_ir`、`_mask_ir` | F1/F15/F18/F23 |
| genesim_bridge/cost_extractor.py | 修改 | 融合清单按专家匹配；逐算子按真实形状测量（专家 768/残差 hidden/QK 归一化 head_dim）；拟合按配方声明的成本项选择依赖变量；成本版本 1.4（host 累加计入选择，强制重精化）；成本 sidecar 回传结构指纹与融合签名 | `_fused_pairs`、`_mark_fused_activations`、`_element_width`、`_fit_coeffs`、`export_costs_to_genesim` | F1/F2/F15/F18/F21/F23/F28 |
| genesim_bridge/placement_export.py | 修改 | 专家/router/输出头权重定位、expert_id、moe_config、B 路 add/mul 语义、通信边 expert_id；模型级 lm_head 导出、结构指纹与容量契约回传；host GEMM 按 `spec.device` 记录 `host_gemms` 落点证据（不混入设备条目），设备 GEMM 缺 pimir 拒绝；B 路接收本次硬件配置并核验每份 pimir 的容量/对齐属性；round16 把 B 路 pimir 固定为「1 token × 本地宽」单位：符号维 `Tq=1`、历史维仍取一块 KV 占位，MLP act/mul 从相邻多分片 GEMM 的 sidecar 分片取本地宽，sidecar 回传 `trace_tokens` 与 `local_*_shapes`，pimir 函数带 `pim.traceTokens`；round17 侧车单位形状保留 IR 的符号维（`Tq`/`Tp+Tq` 标注角色，特征维为本地宽），具体占位只在编译时展开 | `_weight_pattern`、`_role_of_weight`、`export_gemm`、`_mlp_elementwise_local_width`、`_pimir_unit_shapes`、`_bpath_pimir`、`_attach_bpath_pimir`、`_build_pimir_exempt_ops`、`_hw_budget_payload`、`_assert_pimir_hardware_contract` | F1/F2/F4/F13/F14/F18/F20/F21/F24/F30/F33/F34/F65/F69 |
| scripts/export_pp_placement.py | 修改 | 从注册表加载模型与命名产物；`--moe-execution` 与 sidecar 回传；容量/硬件配置从 `default_site_capacity()` 派生；切分方案身份硬校验；抽出 `resolve_strategy` 让两条入口共用架构约束；打印每台 DPU 的权重/KV/激活/内核所需与可用字节 | `main`、`resolve_strategy`、`report_capacity` | F2/F12/F20/F21/F34/F40/F41 |
| scripts/run_full_pipeline.py | 修改 | 注册表驱动主链、MoE 身份/专家号/配置核对、豁免清单；`--moe-execution` 贯穿 A/B/E；fresh sidecar 缺哈希失败；A2 从放置 IR 精化并校验落点证据；源 IR 驱动的覆盖审计、逐 op_id 专家身份与权重路径、逐条目豁免落点、trace 来源白名单与应覆盖集合对拍；absorbed 清单证明与容量三件套校验；E 阶段显式绑定本次 placement 并按策略/DPU/段数/Cluster/本地形状复核；运行与结果目录按架构/策略派生；统一严格产物验证器 `validate_sidecar_contract`；round09 补吸收的消费者方向与 `require_absorbed_closure` 阶段参数，豁免与 sidecar 放置条目/`host_gemms` 证据交叉核对；round12 给共享验证器增 `strict_pimir` 上下文并与 sidecar 自报取并集，导出/skip-simulation 显式严格，freshness/仿真 IR/trace 从配置解析 `require_compiler_pimir` | `step_a_model_ir`、`step_a2_refine_pimir_ir`、`step_bcd_export`、`step_e_simulate`、`_resolve_moe_execution`、`prepare_simulation_config`、`run_directory`、`_assert_placement_matches_run`、`_validate_sidecar_identity`、`validate_sidecar_contract`、`audit_sidecar_coverage`、`_config_requires_pimir`、`_invalid_exemptions`、`_validate_absorbed_ops`、`_validate_sidecar_hw_budget`、`verify_trace_provenance`；round16 增 `_configured_moe_identity`/`_validate_moe_identity`（A 段与 BCD 段都用 HF config 派生值核对 top-k/专家宽）与 `_bpath_unit_problems`（trace 的 token 数/本地形状/字节按 B 路编译单位验收）；round17 增 `_bpath_unit_contract_problems`（单位完整性/符号维角色）挂到 `audit_sidecar_coverage` 严格分支，导出、skip-simulation、freshness、仿真 IR 共用；trace 侧 `_unit_shape_conflict` 只比较两侧具体维，字节按 `trace_tokens`/`loop_unit_tokens` 解析符号维 | F2/F4/F5/F12/F13/F18/F20/F21/F24/F29/F30/F33/F35/F39/F42/F44/F45/F52/F65/F66/F69/F70 |
| tests/test_architectures.py | 修改 | 溢出回归、分块策略、注册表、无权重站点键解析；两态语义用后端开关切换；编译器导出图 ↔ GeneSim 骨架的权重对拍（llama/qwen3/qwen3_moe）；单专家直接段数/PartitionPlan 与自动选择共用架构约束；round16 增 bool 类型注释必须描述 `selected_mask` 的静态用例 | — | F3/F5/F9/F16/F18/F41/F67 |
| tests/test_moe_dispatch.py | 修改 | 分块专家识别、通信门控与次数下降、主机跳过零值；提交前跳过、`skip_zero` 取舍、同一后端 A/B、基准执行循环 | — | F8/F16 |
| tests/test_opt_passes.py | 修改 | `graph_tiling` 开关单测 | — | F5 |
| tests/test_placement_export.py | 修改 | MoE 权重模式/专家号校验；VECTOR_MUL 原语、未知算子拒绝、fresh sidecar 缺哈希失败（含源 IR 覆盖前置）；模型级 lm_head 导出与容量契约回传；round16 增 B 路单位契约用例（1 token、MLP 本地宽、视图算子保持全局宽、kv_cache/matmul 非退化形状）；round17 单位断言改为保留符号维，并增四节点注意力历史维（MASK/SOFTMAX/GEMV_SCORE/GEMV_CONTEXT）单位模板用例 | — | F1/F2/F13/F14/F18/F20/F24/F65/F69 |
| tests/test_serve_pim_preflight.py | 修改 | KV 估算与 `build_kv_layout` 逐字节一致；契约对齐下实际装得下的配置放行；超预算 1 字节仍拒绝；入口无 256 对齐/4 GiB 示例 | — | F43 |
| tests/test_qwen3_30b_real.py | 修改 | 恢复 HF 原始参考、节点数上限；预算改用 `default_hw_budget()`（8 GiB 契约） | — | F6/F5/F20 |
| tests/README.md | 修改 | 30B 依赖矩阵、用例计数与新行为说明；结构指纹、lm_head 导出与容量契约用例；round18 把 30B 容量口径改为 `default_site_capacity()` 派生 | — | F9/F10/F18/F20/F21/F77 |
| genesim/src/ir/model_ir.py | 修改 | MoE 元数据与校验、router/门控/专家图构造、expert_id；`moe_execution` 参数与缺省规则、专家 SILU/MUL/ACCUM 标号；host 累加计入选择并区分首专家（2/3×hidden）；lm_head 与 q_norm/k_norm、模型身份字段、结构指纹（schema 2 含依赖拓扑与层归属）、`moe_cost_role`；切分算子继承成本角色；round17 增运行期字段 `runtime_query_tokens`（调度器绑定实际 Tq，不落 IR） | `build_from_hf_config`、`_validate_moe_contract`、`structural_fingerprint`、`default_moe_execution`、`Operator.runtime_query_tokens` | F1/F12/F18/F19/F21/F25/F28/F68 |
| genesim/src/main.py | 修改 | summary 携带模型身份、结构指纹与 MoE 契约 | `save_all_results` | F1/F21 |
| genesim/src/pim/capacity.py | 修改 | 编译器放置下模型常驻按 ClusterPU 校验（KV 仍按 TensorPU）；系统预留与 MRAM 对齐常量（与编译器契约逐值一致） | `tensor_pu_capacity_metrics`、`validate_tensor_pu_capacity`、`DPU_SYS_RESERVE_BYTES`、`DPU_DMA_ALIGN` | F2/F4/F30 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | 严格 pimir 契约（显式开关与自报统一）、MoE 元数据核对、专家分组 TP 展开、trace expert_id/parent_op_id；动态分派计算/通信计费、pimir 逐元素语义核对；结构指纹与容量契约三件套校验、模型级输出头 TP 展开、成本角色计费；源 IR 驱动的覆盖校验（先于放置与 trace）、absorbed 结构契约与专家身份核对、豁免语义核对、旧式依赖统一折算出口、展开后重算指纹；依赖边按两端的专家角色折算（专家产出到共享/HOST 随源命中并集，共享输入到专家随消费者），放置估算与正式管线共用；round09 吸收校验补主算子唯一消费者与清单双向闭合，豁免校验补放置矛盾与 `host_gemms` 身份核对（含模型边界）；round17 在请求绑定时写 `runtime_query_tokens`（带符号 Tq 才随请求增长），单位登记拒绝丢失符号维的旧式形状，单位比较只核两侧具体维并按 trace 的 tokens/块 token 解析符号做字节核对 | `_prepare_pim_traces`、`_validate_pimir_source_coverage`、`_validate_absorbed_ops`、`_validate_moe_expert_identity`、`_validate_moe_config`、`_validate_model_identity`、`_validate_hw_budget`、`_expand_multi_shard_operators`、`_moe_dependency_fraction`、`_moe_expert_fraction`、`dependency_transfer_bytes`、`_assert_pimir_operator_kind`、`_get_operator_with_runtime`、`_register_pimir_unit`、`_unit_shape_conflict`、`_resolved_unit_shape` | F1/F4/F12/F14/F16/F18/F19/F20/F21/F24/F25/F26/F29/F30/F44/F45/F47/F68/F69 |
| genesim/tests/sim/test_model_ir.py | 修改 | 真实 30B 几何/驻留/语义/序列化用例；执行语义缺省/显式/非法值；全图覆盖（lm_head/q_norm/k_norm）与结构指纹用例；换边/换张量身份/跨层移动必须变指纹 | — | F1/F12/F18/F21/F25 |
| genesim/tests/sim/test_model_parser.py | 修改 | MoE CLI 端到端元数据用例；`--moe-execution` 显式传参 | — | F1/F12 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | 严格契约 6 条、专家分组展开用例；显式严格开关、缺 sidecar、豁免落 PIM、缺 moe_config、pimir 语义核对；结构指纹/容量契约与模型级输出头展开用例；GPU/CPU 落点缺项、全图落 GPU、未知算子豁免、合法 host/边界豁免 | — | F1/F4/F13/F14/F20/F21/F24 |
| genesim/tests/sim/test_pipeline_execution.py | 修改 | 旧式依赖与显式系数边的实际传输字节等价；覆盖 expert_work/expert_comm/shared_reduce 与 dense_sim；round10 补专家源→共享累加、专家源→HOST、共享源→共享目标、普通源→专家目标的 decode/prefill 正反向用例 | — | F26/F47 |
| genesim/tests/sim/test_main.py | 修改 | summary MoE 字段用例 | — | F1 |
| genesim/tests/sim/test_pim_capacity.py | 修改 | 编译器放置容量口径用例 | — | F2/F4 |
| tests/test_ir_cost_eval.py | 修改 | 融合清单专家号契约、精化步骤新签名；专家 768/残差 hidden/QK 归一化 head_dim 的逐算子成本；精化缓存键含指纹/融合清单/成本版本；Tp=0/128/726 三个 decode 点上当前 token 类成本不变、注意力类随历史增长 | — | F1/F2/F15/F18/F21/F23 |
| tests/test_genesim_simulation.py | 修改 | 源 IR 覆盖审计（GPU/CPU 落点无关、未知豁免、合法 host/边界/非 DPU GEMM、MoE 三件套）；trace 来源白名单、应覆盖集合对拍与 parent_op_id；skip-simulation 覆盖校验；仿真绑定本次 placement/策略形状核对/运行目录隔离；完整产物验证器在五个入口的正反向用例；round09 增主算子多消费者、BCD 清单先行、精化后缺吸收编号、豁免与 PIM 放置矛盾、host GEMM 证据缺失/角色/权重不符，覆盖 freshness/sim_ir/trace/skip-simulation 四个入口；round12 增严格配置 + 删自报/4 GiB/16 GiB、非严格兼容、导出与 skip-simulation 容量契约用例；round17 增四节点注意力单位正反向用例、三入口缺/半缺单位与丢符号维用例、skip-simulation 缺单位用例、独立入口共享产物一致性用例 | — | F24/F39/F42/F44/F45/F52/F69/F70/F71 |
| genesim/conf/sim_qwen3_moe_pimir.yaml | 新增 | Qwen3-MoE 仿真配置（pimir IR + sidecar + 严格模式）；注明主链会用本次 BCD 产物改写并复核放置路径；round17 改为「模板 + 说明」：共享 models 产物由主链刷新，可直接运行的配置是 run 目录里绑定本次 IR/sidecar 的那份；独立输出目录避免覆盖共享 results/pim_traces，并写明 trace 生成命令 | — | F2/F39/F71 |
| runtime/executor.py | 修改 | 后端未显式配置时按 `CompiledModel.moe_execution` 决定是否分派 | `_apply_moe_execution` | F12/F16 |
| genesim/scripts/model_parser.py | 修改 | CLI 新增 `--moe-execution`，把编译配置原样写进图骨架；`parameter_bytes` 与完整 HF 参数交叉校验 | `main` | F12/F18 |
| scripts/bench_moe_dispatch.py | 新增 | 同一计划/后端/权重上比较两种语义的执行期基准，断言 token 一致并报告命中专家数与逐层放行/拦下命令数 | `benchmark`、`_run_steps`、`_dispatch_coverage` | F16/F27 |
| genesim/tests/sim/test_moe_execution_cost.py | 新增 | 动态分派专家计算/通信按 any-hit `1-(1-k/E)^Tq`、共享累加不缩放、PIM trace 时延折算；round10 补依赖边方向（专家源/消费者/HOST 出口）单元用例 | — | F12/F16/F19/F47 |
| genesim_bridge/ir_fingerprint.py | 新增 | 编译器侧结构指纹与融合签名（与 GeneSim 算法逐字段一致）；schema 2 补依赖端点/张量身份与层归属 | `fingerprint_ir`、`assert_ir_fingerprint`、`fusion_signature` | F21/F25 |
| tests/test_ir_fingerprint.py | 新增 | 跨仓指纹一致、词表/hidden 变化、改内容不改指纹、A2 缓存键；换边/换张量身份/跨层移动必须变指纹，成本与放置字段不变 | — | F21/F25 |
| contracts/op_contract.py | 修改 | 8 DPU × 8 GiB 容量契约常量；默认硬件配置取它 | `DPU_MRAM_BYTES`、`DPU_SYS_RESERVE_BYTES` | F20 |
| memory/mem_planner.py | 修改 | 单一容量契约的 `HwBudget` 工厂 | `default_hw_budget` | F20 |
| backend/vllm_worker.py | 修改 | 运行期 MRAM/系统预留默认值取容量契约；HwBudget 与 PIMHardwareConfig 由同一工厂派生（对齐统一为 64） | — | F20/F34 |
| scripts/serve_pim.py | 修改 | 启动预检默认值取容量契约；删除独立 `ALIGN=256`，KV 估算直接复用 `build_kv_layout`；示例不再给 4 GiB 预算 | `_kv_bytes_per_dpu`、`preflight` | F20/F43 |
| scripts/profile_execution.py | 修改 | 执行期剖析预算取容量契约 | — | F20 |
| graph/partition.py | 修改 | `ne.Scalar`/`where.ScalarOther` 列入 HOST_ONLY：PIM 原语集没有逐元素 select；选中掩码的 `scatter.value` 同属门控链留在主机 | `HOST_ONLY` | F28/F32 |
| tests/test_moe_dispatch.py | 修改 | 逐层 gate 断言、层域传播不变式、PP 跨段、后端逐 dispatch 计数、跨仓 plan↔IR 角色对拍 | — | F27/F28 |
| tests/test_moe_ep.py | 修改 | EP 合法路径逐层门控断言 | — | F27 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | absorbed 结构契约正反向用例、直接 GeneSim 专家身份、容量三件套缺失/偏值用例 | — | F29/F30 |
| genesim/tests/sim/test_model_ir.py | 修改 | host 尾部（选择+加权累加）结构与首专家公式用例；批量修改索引次数/扫描量/与逐条等价用例；单专家 MoE 图骨架用例 | — | F28/F36/F37 |
| contracts/op_contract.py | 修改 | `default_hardware_config()` 从容量三件套派生硬件；默认 `dma_align` 改为契约值 64 | `default_hardware_config`、`DEFAULT_HARDWARE_CONFIG` | F34 |
| memory/mem_planner.py | 修改 | `default_site_capacity()` 同时派生 HwBudget 与 PIMHardwareConfig；对齐只取契约 | `default_site_capacity` | F34 |
| opcompiler_bridge/driver.py、pipeline.py、genesim_bridge/flagtree_driver.py | 修改 | B 路 `_run_oplevel_triton_opt` 接收本次硬件配置；模块已有冲突属性时拒绝而非保留；只拿到文本时优先沿用模块头预算 | `_run_oplevel_triton_opt`、`_options_hardware`、`with_budget`、`module_budget`、`lower_oplevel_to_pimir` | F34 |
| genesim/src/ir/model_ir.py | 修改 | `_BatchEdit`/`batch_edit()` 批量增删与一次索引重建；MoE 身份改由 model_type+有效字段判定 | `_BatchEdit`、`batch_edit`、`build_from_hf_config` | F36/F37 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | TP 展开与 relay 插入改批量；豁免读落点证据；专家身份加权重路径；逐份 pimir 核容量/对齐；round16 登记 sidecar 的 B 路编译单位（`_register_pimir_unit`），trace 生成前核对 tokens/本地形状/单相字节（`_validate_pimir_unit`），TP 展开只在分片本地形状与单位一致时继承 pimir，严格覆盖校验新增「B 路缺单位」类别 | `_expand_multi_shard_operators`、`_attach_collective_edges`、`_exemption_is_legitimate`、`_validate_moe_expert_identity`、`_validate_pimir_hardware_contract`、`_register_pimir_unit`、`_require_pimir_unit`、`_validate_pimir_unit`、`_unit_shape_conflict`、`_validate_pimir_source_coverage` | F33/F34/F35/F36/F65 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | 设备 router 豁免/伪造 host 证据/权重互换/单专家 sidecar/容量对齐用例；批量展开与逐条等价、索引只建一次；round09 增吸收的其它消费者、清单缺吸收编号、host GEMM 证据与权重身份、边界豁免落 PIM 用例；round10 增 TP 展开后专家分片→共享累加边仍按源角色折算的用例；round16 增编译单位校验（tokens/末维/字节/SPLIT 视图）与 TP 分片继承守卫用例，严格夹具补单位；round17 单位形状改符号模板，增四节点注意力 prefill/decode 接受、错误特征宽拒绝、旧式固化历史维拒绝用例 | — | F33/F34/F35/F36/F37/F44/F45/F47/F65/F69 |
| genesim/tests/sim/test_collective_edges.py、test_config_loader.py、test_cost_sidecar.py、test_gpu_backend.py | 修改 | `unittest.main()` 从文件中段移到末尾，后置测试类恢复在 `./run.sh --test sim` 中执行 | — | F46 |
| flagos-pim-compiler/tests/test_architectures.py | 修改 | 零权重选中专家的 HF 对拍（两态）、未选中溢出两态、单专家生成与执行语义/分块默认 | — | F32/F37 |
| flagos-pim-compiler/tests/test_moe_dispatch.py | 修改 | decode/prefill dispatch 必须读布尔选中掩码；提交前跳过与逐层 gate 断言沿用；round10 增未命中 host 收集命令不触发跨设备搬运、命中专家照常收集的用例 | — | F32/F47 |
| flagos-pim-compiler/tests/test_placement_export.py | 修改 | host GEMM 落点证据、设备 GEMM 缺 pimir 拒绝、默认对齐工厂、B 路冲突属性、pimir 属性核验 | — | F33/F34 |
| flagos-pim-compiler/tests/test_genesim_simulation.py | 修改 | 逐 op_id 专家身份/权重路径/跨层互换、router 豁免拒绝、trace 验收豁免校验、单专家执行语义 | — | F32/F33/F34/F35/F37 |
| scripts/check_implement_evidence.py | 新增 | 从 summary 提取 total_time_s、tokens/s 与有效 FLOPs 并核对实施文档；文档里千分位高精度数字必须能溯源到产物或显式声明的历史值 | `summary_metrics`、`check_document`、`main` | F48 |
| tests/test_implement_evidence.py | 新增 | 当前值通过、抄错 FLOPs 被拒绝、历史值放行、time-only 不要求 FLOPs | — | F48 |
| genesim/src/pim/capacity.py | 修改 | 需求常量 `DPU_MRAM_BYTES=8 GiB`；默认容量由常量派生；跨仓逐值相等 | `DPU_MRAM_BYTES`、`DEFAULT_CLUSTER_MEMORY_CAPACITY_GB` | F49 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | 严格路径同时核对有效配置与 sidecar；MLIR 容量属性固定对需求常量；旧式非严格产物保留兼容 | `_validate_hw_budget`、`_validate_pimir_hardware_contract` | F49 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | 配置与 sidecar 同步 4/16 GiB、单侧偏离、MLIR 16 GiB、旧产物兼容用例 | — | F49 |
| scripts/run_full_pipeline.py | 修改 | 运行目录含语义/运行号；A/A2/BCD 产物落本次 artifacts；E 临时配置绑定 IR/trace/results/log 并回读；删除共享 `pim_traces` 清空与 results 复制；trace 验收按配置目录 | `run_directory`、`prepare_simulation_config`、`step_e_simulate`、`step_a_model_ir`、`step_a2_refine_pimir_ir`、`step_bcd_export`、`step_zero_pu_mapping`、`_configured_dir`、`verify_trace_provenance` | F50 |
| tests/test_genesim_simulation.py | 修改 | 预置哨兵、交错语义、配置输出目录回读与 trace 来源目录用例 | — | F50 |
| scripts/check_implement_evidence.py | 修改 | 解析证据块；current 逐标签绑定 summary 并核对层数/模型/语义；派生 subtract/ratio/greater_than 重算；正文高精度数值按 history/measurements 命名空间溯源；round12 增 `prose_aliases` 与块级前置标签绑定、必需值只扫正文、带数值的大小结论判定 | `parse_evidence_block`、`check_document`、`_check_derived`、`_prose_context_problems`、`_check_comparisons`、`summary_identity` | F51/F53 |
| tests/test_implement_evidence.py | 修改 | 标签互换、4 层冒充 48 层、历史冒充 current、错误差值/比值/结论、科学计数法正文与合法分栏；round12 增只换正文、48 层借值、历史冒充、错误比较、正文缺失必需值与 measurements 放行 6 条 | — | F51/F53 |
| tests/test_placement_export.py | 修改 | 跨仓容量常量增加 `DPU_MRAM_BYTES` 逐值相等 | — | F49 |
| FlagTree/lib/Dialect/TritonPIM/Transforms/LowerPIMToEmitC.cpp | 修改 | paged_attn 的每块 PV 乘积先 `roundF16` 再加进 f32 累加器；块表项改走 `pim_block_slot`（负值/上界中止，不再夹成槽 0）；补相位存储边界表注释 | `lowerOperatorOp`（`PagedAttnOp` 分支） | F54/F55 |
| FlagTree/lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp | 修改 | PV 相说明改为「f32 累加器 + f16 乘积临时值」，与算子契约一致；`bytes` 仍记流式累加器 | `stampPagedAttn` | F54 |
| FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir | 新增 | lit 约束块表兜底调用与 5 个 f16 取整点（QK 点积、QK 分数、exp、PV 乘积、最终输出） | — | F54/F55 |
| runtime/kernels.py | 修改 | `sdpa_kv_info` 回传 `pool_slots`；新增 `_physical_slot`/`_validate_block_table`，启动前按 `0 <= slot < pool_slots` 预检、写 KV 与主机兜底共用；主机参考说明与相位边界对齐 | `_paged_attention`、`_iter_kv_blocks`、`sdpa_kernel`、`_physical_slot` | F54/F55 |
| scripts/run_full_pipeline.py | 修改 | `_config_text` 缺失即失败；freshness/仿真 IR/trace 先解析严格性，严格时配置键与 IR/sidecar 文件为必需输入；trace 逐条核对父算子/类型/`pimir_sha256` 与 absorbed 零指令 | `verify_sidecar_freshness`、`verify_sim_ir_matches_sidecar`、`verify_trace_provenance`、`_trace_source_lineage`、`_strict_sidecar_context` | F56/F57 |
| scripts/check_implement_evidence.py | 修改 | 正文数值改为最近前置标签归属；前者/后者按最近两个运行解析；必需值要求运行语境；新增 `facts` 绑定非 summary 证据值 | `_effective_labels`、`_nearest_labels`、`_check_comparisons`、`_fact_bindings`、`_value_in_run_context` | F58 |
| tests/test_opcompiler_ops.py | 修改 | PV 参考按契约落 f16；新增多块/尾块用例，并断言数据能区分取整口径 | — | F54 |
| tests/test_kernels.py | 修改 | 槽 0/最后合法槽放行、负值/上界拒绝、表尾哨兵不读、启动前预检；融合/主机对拍容差说明改为只吸收 exp/顺序差 | — | F54/F55 |
| tests/test_genesim_simulation.py | 修改 | 严格入口缺配置/IR/sidecar 的六种组合；trace 错哈希/缺哈希/错类型/错父编号/非空 absorbed；真实哈希夹具与分片正例；round16 增 config 派生 MoE 身份（top-k/专家宽）用例、B 路 trace 单位（tokens/末维/字节/SPLIT 视图）正反向用例，夹具补编译单位；round17 增共享验证器的单位完整性/符号维角色反例（三入口 + skip-simulation）与独立入口共享产物一致性用例 | — | F56/F57/F65/F66/F69/F70/F71 |
| tests/test_implement_evidence.py | 修改 | 同块前者/后者对调、身份正确正例、无标签前言必需值、facts 运行绑定与未知运行标签 | — | F58 |
| genesim/tests/sim/test_moe_execution_cost.py | 修改 | 命中口径说明从 `routing != 0` 改为 top-k 布尔 `selected_mask.any(dim=0)` | — | F59 |
| runtime/kernels.py | 修改 | round15：`where_kernel` 删除不可达的显式 `broadcast_to` 分支（`where.ScalarOther` 是 HOST_ONLY，不经 `_KERNELS`）；MoE 门控选择说明移到 `_KERNELS` 的「故意不在表里」段 | `where_kernel` | F64 |
| scripts/run_full_pipeline.py | 修改 | round15：严格上下文下 pimir 产物（可读 + 哈希）成为固定规则并删除 `check_artifacts` 参数；`stale_pimir_entries` 缺路径/读失败转 `StepFailed`；trace 验收要求配置；`prepare_simulation_config` 在运行目录生成缺失的请求分布 trace 并回读绑定 | `validate_sidecar_contract`、`audit_sidecar_coverage`、`stale_pimir_entries`、`verify_trace_provenance`、`_generate_synthetic_trace`、`prepare_simulation_config`、`step_e_simulate` | F60/F61/F62 |
| tests/test_genesim_simulation.py | 修改 | round15：三入口 × 五类 pimir 产物破坏矩阵、自报严格路径、无配置 trace 反例、缺失 trace 生成与已有 trace 保留正反例；容量正例改用真实 pimir 文件 | — | F60/F61/F62 |
| tests/test_placement_export.py | 修改 | round15：模板/unknown 用例改传配置；trace 头部字节用例直接测 `_trace_metadata` | — | F61 |
| tests/test_kernels.py | 修改 | round15：`where.self` 广播对拍、`where.ScalarOther`/`ne.Scalar` 主机闭包与未注册断言 | — | F64 |
| genesim/src/pim/pimir_trace.py | 修改 | round16：解析函数属性 `pim.traceTokens`（优先于首维启发式），相位链与单相算子都回传 trace 覆盖的查询 token 数 | `_declared_trace_tokens`、`parse_phase_chain`、`_parse_single_phase_ops`、`phase_metadata` | F65 |
| genesim/src/vpu/vpu.py | 修改 | round16：运行期重复次数改为 `ceil(实际 Tq / trace_tokens)`，块循环重复次数同口径；`execute_profile` 的访存字节按同一重复次数计。round17：实际 Tq 只读调度器写入的 `runtime_query_tokens`，删除按输入形状推断的 `_extract_query_count`/GQA 启发式（缺字段直接报错）；GATHER 的模板循环宽改取输出隐藏维，不再把 rank-1 token 维当特征宽 | `_trace_tokens`、`_bound_query_tokens`、`_unit_repetitions`、`_block_loop_binding`、`_execute_runtime`、`execute_profile` | F65/F68 |
| genesim/tests/sim/test_pimir_trace.py | 修改 | round16：无块循环也回传 tokens；显式 `pim.traceTokens` 优先于形状启发式（TRANSPOSE 维序）；4 个本地宽分片的 trace 合计等于全局宽 | — | F65 |
| genesim/tests/sim/test_vpu.py | 修改 | round16：B 路单位 trace 按实际 Tq 重复、旧宽 token trace 折算、部分 prefill 向上取整、块循环按 trace token 单位折算；round17 增 3-D TRANSPOSE、头优先 SPLIT、rank-1 GATHER、PAGED_ATTN prefill/decode 的绑定 Tq 用例，以及缺字段拒绝形状猜测的用例；既有直调夹具补 `runtime_query_tokens` | — | F65/F68 |
| scripts/run_full_pipeline.py | 修改 | round18：A/B 分类只取源 IR（GEMM 走 A 路，其余走 B 路），B 路假分片与严格产物设备 GEMM 空分片直接失败；共享验证器按路径解析一次 pimir 的 `pim.traceTokens` 与参数/结果形状，要求 sidecar `trace_tokens`、`local_*_shapes` 特征末维与实产一致；trace 入口按 IR 分类并拒绝 B 路假分片；`check_sidecar_entries` 增加源 IR 参数 | `_entry_is_host`、`_entry_class_problems`、`check_sidecar_entries`、`_split_top_level`、`_pimir_unit_facts`、`_pimir_unit_problems`、`_bpath_unit_contract_problems`、`_bpath_unit_problems`、`audit_sidecar_coverage` | F72/F73 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | round18：新增条目类别校验（B 路假分片、严格产物设备 GEMM 空分片直接失败）；单位检查、放置加载、本地形状标注、覆盖审计、TP 展开与继承一律按源 IR 的 `op_type` 分类；`_validate_pimir_unit` 增加 pimir 文本入参，核对 sidecar 单位与 `pim.traceTokens`/特征末维 | `_validate_entry_classes`、`_load_compiler_placement`、`_require_pimir_unit`、`_apply_compiler_local_shapes`、`_expand_multi_shard_operators`、`_validate_pimir_source_coverage`、`_validate_pimir_unit`、`_validate_unit_against_pimir` | F72/F73 |
| genesim/src/pim/pimir_trace.py | 修改 | round18：新增 `pimir_unit_facts`，解析函数参数/最后一条带结果 op 的张量形状与 `pim.traceTokens`，供 sidecar 单位与实产核对 | `pimir_unit_facts`、`_split_top_level` | F73 |
| tests/test_genesim_simulation.py | 修改 | round18：夹具 pimir 补真实函数/`traceTokens`；新增假分片、设备 GEMM 空分片、`trace_tokens`/本地宽与 pimir 不符、旧 128-token pimir、无 traceTokens pimir、skip-simulation 反例 | — | F72/F73 |
| tests/test_placement_export.py | 修改 | round18：`check_sidecar_entries` 改传源 IR；新增 B 路假分片、设备 GEMM 空分片分类用例 | — | F72 |
| scripts/run_full_pipeline.py | 修改 | round19：`_pimir_unit_facts` 扩展解析分块 GEMM 事实（原语、tile 属性、`tt.dot` 的 M/K/N、循环嵌套、`act_spec`、逐元素 kind）；新增 `_apath_unit_problems`/`_apath_facts_problems`，严格产物逐条核对设备 GEMM 的 pimir 是分片实测分块 GEMM、tile/循环与分片宽度一致、融合激活与清单双向一致；`_entry_is_host` 改为「源 IR 落点 + 豁免登记 + `host_gemms` 证据」三件套，类检查把 `op_type`/`semantic_role` 与源 IR 逐条绑定；`_bpath_unit_contract_problems` 核对逐元素 pimir kind；trace 字节判据改用源 IR 类型 | `_entry_is_host`、`_entry_class_problems`、`check_sidecar_entries`、`_pimir_unit_facts`、`_apath_facts_problems`、`_apath_unit_problems`、`_bpath_unit_contract_problems`、`_bpath_unit_problems`、`_assert_placement_matches_run`、`audit_sidecar_coverage` | F78/F79/F80 |
| runtime/kernels.py | 修改 | round19：`_host_paged_attention` 在主机代码里显式实现相位表——QK 点积落 f16、乘 scale 再落 f16，每块 PV 乘积落 f16 后加进 f32 累加器；不再依赖 `matmul` 走编译内核时返回 f16 | `_host_paged_attention` | F81 |
| tests/test_kernels.py | 修改 | round19：新增同时关闭 `_compiled_paged_attn` 与 `_compiled_matmul` 的用例，按相位表参考逐元素对拍，并断言数据能区分是否落 PV 边界 | — | F81 |
| tests/test_genesim_simulation.py | 修改 | round19：契约夹具把 A 路 GEMM 与 B 路算子分开引用同形产物（分块 GEMM / 1-token 单相），补 `kernel_tile_n`；新增全部 GEMM 指向 B 路产物、换分片宽度、删 `kernel_tile_n`、融合缺 `act_spec`、host 自报缺证据（三入口 + skip-simulation）、`op_type` 互换、缺 `semantic_role`、逐元素 kind 互换、trace 字节按 IR 类型选择等用例 | — | F78/F79/F80 |
| tests/test_placement_export.py | 修改 | round19：`check_sidecar_entries` 夹具的源 IR 补 `semantic_role`，与真实 GEMM 的投影身份一致 | — | F80 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | round19：`_validate_entry_classes` 的 host 判定改为源 IR + 豁免 + `host_gemms` 证据，`op_type`/`semantic_role` 与源 IR 逐条比较；新增 `_gemm_pimir_problems` 并在 `_validate_pimir_source_coverage` 与 `_validate_pimir_unit` 逐条核对 A 路分块 GEMM 原语、tile/循环、分片宽度与融合激活；登记每个 GEMM 的期望融合激活 | `_entry_is_host`、`_validate_entry_classes`、`_gemm_pimir_problems`、`_validate_pimir_source_coverage`、`_validate_pimir_unit`、`_register_compiler_artifacts` | F78/F79/F80 |
| genesim/src/pim/pimir_trace.py | 修改 | round19：新增 `pimir_gemm_facts`，解析分块 GEMM 原语、`tt.dot` 三维、`pim.tile-k`/`pim.tile-n`、K/N 循环嵌套、`act_spec` 与 `pim.traceTokens`，供 A 路语义核对 | `pimir_gemm_facts` | F78 |
| genesim/src/pim/capacity.py | 修改 | round19：`max_utilization`/`max_tensor_pu_utilization` 改用与 `fits` 相同的判定分子；新增 `max_tensor_pu_request_bytes`/`max_tensor_pu_request_utilization` 与 `max_total_resident_ratio`，含模型权重的总量不再与 512 MiB 上限配成利用率 | `tensor_pu_capacity_metrics` | F82 |
| genesim/src/main.py | 修改 | round19：summary 分开记录 TensorPU 请求期读数与集群常驻利用率；控制台改为「TensorPU 判定口径利用率」+「ClusterPU 模型常驻利用率」两行 | `save_all_results`、`format_summary` | F82 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | round19：A 路/ B 路夹具分开引用同形产物并补 `kernel_tile_n`；新增分块 GEMM 原语、分片宽度、缺分块、host 自报缺证据、trace 单位校验 A 路规则用例 | — | F78/F79/F80 |
| genesim/tests/sim/test_pim_capacity.py | 修改 | round19：新增判定利用率与 `fits` 同源、请求期/集群/总量比三个 scope 的用例 | — | F82 |
| genesim/tests/sim/test_main.py | 修改 | round19：summary 渲染改为判定口径利用率与集群利用率两行 | — | F82 |
| docs/design-qwen3-moe-20261105.md | 修改 | round19：§十.4 补 A 路编译单位、host 证据链与条目字段绑定；§十.5 补利用率与判定同源、总量比单列 | — | F78/F79/F80/F82 |
| tests/test_implement_evidence.py | 修改 | round18：新增 CI 用例，按证据块 current 标签实际执行文档中的 checker 命令并断言退出码 0，同时禁止「四份 summary」声明 | — | F75 |
| tests/test_opcompiler_linear.py | 修改 | round18：缓存锁测试等待上限改 60 秒、子进程 `finally` 回收；新增首次 put 晚于 5 秒的慢调度回归用例 | — | F74 |
| tests/test_architectures.py | 修改 | round18：新增真实测试容量工厂静态守卫（禁用 8 GiB/64 MiB/对齐/WRAM 字面量与手写 `HwBudget`/`PIMHardwareConfig`） | — | F77 |
| tests/test_qwen3_8b_real.py | 修改 | round18：`HwBudget` 与 `PIMHardwareConfig` 改由 `default_site_capacity()` 派生 | — | F77 |
| tests/test_qwen3_30b_real.py | 修改 | round18：`PIMHardwareConfig` 并入 `default_site_capacity()` 的返回值 | — | F77 |
| docs/design-qwen3-moe-20261105.md | 修改 | round18 §十.4 补 A/B 分类规则与「pimir 是单位唯一真源」契约 | — | F72/F73 |
| scripts/run_full_pipeline.py | 修改 | round20：`_pimir_unit_facts` 增解析顶层原语、激活/逐元素种类、块循环与 KV 固定开销事实；新增闭合表 `_BPATH_PIMIR_SEMANTICS` 与 `_bpath_pimir_semantic_problems`；`_bpath_unit_contract_problems` 逐条核对 B 路语义指纹，替换只覆盖 add/mul 的旧判定 | `_pimir_unit_facts`、`_bpath_pimir_semantic_problems`、`_bpath_unit_contract_problems` | F83 |
| genesim/src/pim/pimir_trace.py | 修改 | round20：`pimir_unit_facts` 扩解析与编译器同名的事实；新增 `BPATH_PIMIR_SEMANTICS` 与 `bpath_pimir_semantic_problems`，表内容与编译器逐值一致 | `pimir_unit_facts`、`BPATH_PIMIR_SEMANTICS`、`bpath_pimir_semantic_problems` | F83 |
| genesim/src/scheduler/gene_sim_scheduler.py | 修改 | round20：覆盖校验对每个 B 路 pimir 核对语义指纹；`_assert_pimir_operator_kind` 扩成 `_assert_pimir_operator_semantics`（全部 B 路类型，GEMM 仍走 A 路规则）；`_validate_pimir_unit` 在相位/字节分支之前先做语义核对 | `_validate_pimir_source_coverage`、`_assert_pimir_operator_semantics`、`_validate_pimir_unit` | F83 |
| genesim/src/main.py | 修改 | round20：summary 增加 `pim_capacity_max_tensor_pu_resident_bytes`、`pim_capacity_max_cluster_resident_bytes` 与 `pim_capacity_max_total_resident_ratio`，容量 scope 写进字段名 | `save_all_results` | F84 |
| tests/test_genesim_simulation.py | 修改 | round20：SILU 夹具补 `#pim.activation<silu>`；新增 5 类语义互换反例、未登记类型反例、18 类实产正例与两仓语义表逐值相等用例 | — | F83 |
| genesim/tests/sim/test_compiler_placement.py | 修改 | round20：覆盖/签名/GEMM 夹具改为按类型构造语义正确 pimir；新增 5 类互换反例、实产正例与覆盖闸门反例 | — | F83 |
| genesim/tests/sim/test_scheduler.py | 修改 | round20：GELU 用例的 pimir 激活种类从 silu 改为 gelu，与源 IR 类型一致 | — | F83 |
| genesim/tests/sim/test_main.py | 修改 | round20：新增 summary 容量 scope 三字段用例 | — | F84 |
| docs/design-qwen3-moe-20261105.md | 修改 | round20：§十.4 补 B 路闭合语义指纹契约（按实产校准、未登记拒绝、两仓钉住）；§十.5 明确 `max_resident_bytes` 为单 TensorPU scope、集群总量用集群字段 | — | F83/F84 |
| tests/test_implement_evidence.py | 修改 | round20：补 `import pytest`（I95），让 current summary 缺失时的 skip 分支可用 | — | F84 |
| tests/test_opcompiler_linear.py | 修改 | round20：两个缓存锁用例从 fork 改为 spawn，worker 提到模块级并显式传入缓存目录；消除全量回归中 fork 继承线程/队列状态导致的事件丢失（I96） | `_lock_worker`、`test_cache_key_lock_serializes_processes`、`test_cache_key_lock_waits_for_a_slow_schedule` | F85 |
### 3.3 实施问题记录
| 编号 | 问题 | 根因 | 解决方案 | 实现位置 | 验证用例 |
|------|------|------|----------|----------|----------|
| I1 | T5 端到端用例未经历“红”状态即通过 | T2–T4 已逐个红→绿，端到端是组合验证 | 判定为合理的集成通过 | — | test_qwen3_moe_generate_matches_hf |
| I2 | `paths.json` 中文被转义 | `json.dumps` 默认 `ensure_ascii=True` | `ensure_ascii=False` 重写 | paths.json | — |
| I3 | 设计待确认 #1：`_rule_weighted_partial` 是否必要 | 既有 mul 规则覆盖 Partial×Replicate | 不新增 | 无改动 | test_qwen3_moe_generate_matches_hf |
| I4 | 开启分派后第 3 token 分叉 | 被跳过的专家 launch 不写输出，旧缓冲的 inf 与路由 0 相乘得 nan | 跳过时把输出区写零 | `backend/hal_numpy.py` | test_gate_skipped_launch_zeroes_its_output |
| I5 | `gate` 参数放到 `*args` 之前导致错绑 | 签名次序错误 | 改为 keyword-only | `backend/hal_numpy.py` | test_hal_numpy.py |
| I6 | `remark` 首版把 MoE 下游全部 mul/add 扫进 host | 圈定条件过宽 | 按「experts 祖先层==归属层」圈定 | `graph/spec_prop.py` | test_moe_ep 全部 |
| I7 | pinned 专家权重被 REPLICATE→REPLICATE 拒重分布 | 发射面未取交集 | 发射面交集 + 超集免搬运 | `runtime/exec_plan_gen.py` | test_moe_ep 全部 |
| I8 | `_moe_layer_of` 跨模块误用（NameError） | 自造辅助函数 | 用 spec_prop 既有 `_layer_of_node` | `runtime/exec_plan_gen.py` | test_moe_ep |
| I9 | K3 竞态：dpu→host 收集读区对 WAR 链不可见 | `register_reader` 的键形 (名, None) 与查询键 (名, dpu) 不匹配 | 读区按 (redist:eNN, 读 DPU) 登记 | `runtime/exec_plan_gen.py`（`_emit_redistribute`） | test_dpu_to_host_reads_are_war_protected |
| I10 | 默认图级分块把 MoE 图放大到 139 万节点（4 层） | 通用 tile 按 fp32 工作集拆专家 linear；MoE 无模型级开关 | 新增 `graph_tiling` 参数与 `PIM_DISABLE_TILE`，MoE 默认关闭；算子层 tt.dot 分块仍在 | `runtime/compile.py`、`graph/opt_passes.py` | test_optimize_graph_graph_tiling_switch |
| I11 | 门控通信命令早期在 submit 时判跳过，dispatch 尚未运行 | 提交异步、dispatch 是依赖 | 判断移入 worker，等完 deps 再查；被跳过的主机命令返回零张量 | `backend/hal_numpy.py` | test_gate_skipped_host_command_returns_zero |
| I12 | prefill 图里 `_moe_layer_of` 找不到层号 | 分块后 `cat([a,b])` 的上游在列表实参里，只扫 `args` 会跳过 | 改用 `all_input_nodes` | `runtime/exec_plan_gen.py` | test_tiled_expert_subgraph_is_recognized_and_gated |
| I13 | MoE sidecar 校验把稠密 gate/up/down 当专家 | 角色语义未按 IR 的 MoE 元数据区分 | `_assert_every_gemm_has_known_role` 按 `moe_num_experts` 分流；专家号必填、稠密不得带 | `genesim_bridge/placement_export.py` | test_moe_gemm_must_carry_expert_id、test_dense_llama_gemm_rejects_expert_id |
| I14 | GeneSim TP 展开假设每层每角色唯一 | `roles` 是 `{role: op_id}`，MoE 一层 128 组三件套 | 按 expert_id 分组；router 走单列切展开 | `genesim/src/scheduler/gene_sim_scheduler.py` | TestExpertAwareTensorParallelExpansion |
| I15 | 融合清单按 (层, 角色) 匹配在 MoE 上匹配到 128 个 | 清单无专家号 | 清单增 `expert` 字段，匹配带 expert_id | `genesim_bridge/placement_export.py`、`genesim_bridge/cost_extractor.py` | 全链路融合步骤 |
| I16 | Qwen3 全链路仿真在 30B embedding（622 MiB）处报单 TensorPU 容量超限 | 图编译器把 DPU 当 ClusterPU（8 GiB），GeneSim 却对模型常驻按单 TensorPU（512 MiB）判超限 | 编译器放置下模型常驻按 ClusterPU 校验，KV/激活仍按 TensorPU；非编译器放置保持双限 | `genesim/src/pim/capacity.py`、`gene_sim_scheduler._validate_pim_capacity` | TestCompilerPlacementCapacityScope（2 条）、Qwen3 全链路仿真 |
| I17 | 检查点 bf16 的 30B 图骨架生成四节点注意力，与编译器 fp16 融合 SDPA 不一致 | builder 用 config 的 `torch_dtype` 决定融合分支，而加载器统一 fp16 | `build_from_hf_config(execution_dtype=)` + parser `--execution_dtype`，主链显式传 float16 | `genesim/src/ir/model_ir.py`、`genesim/scripts/model_parser.py`、`scripts/run_full_pipeline.py` | TestExecutionDtypeAlignment（2 条）、TestParserExecutionDtype |
| I18 | 未命中专家仍要占线程池任务：每条被跳过的命令都提交后在 worker 里等 deps、查跳过集、逐块清零，4 层实测动态/稠密耗时 0.994× | 跳过判定放在执行期，提交面没有收敛；清零也对整条子图逐条执行 | 改为提交前判定：`submit` 先等 dispatch 事件，未命中的命令不进执行队列；`moe_dispatch` 运行期开关供 A/B | `backend/hal_numpy.py`（`submit`、`_skipped_event`）、`runtime/executor.py`（`_apply_moe_execution`） | `test_skipped_launch_is_never_submitted`、`test_benchmark_harness_flips_modes_on_one_backend` |
| I19 | 按地址区间找「读者」会把激活区复用的早期命令误当成消费者，清零范围失控（392 条门控命令里 388 条被误标） | 计划里的读写区间会跨层复用同一块 MRAM，地址重叠不等于数据依赖 | 改沿 FX 图边判定：输出的使用者若与自身专家归属不同才标 `skip_zero`（`_skip_zero_nodes`） | `runtime/exec_plan_gen.py` | `test_tiled_dispatch_tokens_match_hf`、`test_gate_skipped_launch_without_skip_zero_leaves_the_buffer` |
| I20 | GeneSim 的专家通信字节挂在依赖边上，只缩放中转算子会漏掉通信 | 通信代价由 `Dependency.always_bytes` 求值，算子自身的 data_bytes 为 0 | 在两个依赖字节求值点按目标算子的 expert_id 折算（计算按 top_k/E、通信按命中并集） | `genesim/src/scheduler/gene_sim_scheduler.py`（`_moe_dependency_fraction`） | `TestDynamicDispatchCost` |
| I21 | VECTOR_MUL 的 B 路 pimir 被一律发成加法 | 构造器只拿 mnemonic 和形状，VECTOR_ADD/VECTOR_MUL 共用 add 分支 | 按 op_type 映射 kind（add/mul、silu/gelu），未知类型抛错；导出与仿真两侧核对文本里的 kind | `genesim_bridge/placement_export.py`（`_bpath_pimir`）、`gene_sim_scheduler._assert_pimir_operator_kind` | `test_bpath_eltwise_keeps_operator_kind`、`TestPimirOperatorKindIsVerified` |
| I22 | 专家 VECTOR_MUL/SILU 的 FLOPs 按 hidden 2048 计，Tq=128 时 98304 被记成 262144，系数 6 记成 16 | 配方工厂用模型级代表维度（hidden、4×hidden），没有接收逐算子形状 | 配方接收宽度参数，`_measure` 从当前算子输出形状取宽度；残差加仍用 hidden | `genesim_bridge/op_classify.py`、`genesim_bridge/cost_extractor.py`（`_element_width`） | `test_eltwise_and_activation_costs_follow_the_operator_shape` |
| I23 | 显式 `require_compiler_pimir=true` 与 sidecar 自报是两套状态，前者在缺项时仍生成模板 | 严格性只由 `_pimir_contract` 驱动，配置开关只影响错误信息 | `_strict_pimir()` 统一两个来源；缺 sidecar、缺条目、缺哈希在编 trace 前失败 | `genesim/src/scheduler/gene_sim_scheduler.py` | `test_explicit_strict_prechecks_without_self_declaration`、`test_missing_sidecar_file_raises_when_required` |
| I24 | 本轮 fresh 导出的 sidecar 缺哈希仍被 `--skip-simulation` 接受 | 沿用「旧产物」豁免，而该步骤自己刚产出 sidecar | 缺哈希直接失败；旧产物应重新导出 | `scripts/run_full_pipeline.py`（`step_bcd_export`） | `test_export_step_rejects_a_fresh_sidecar_without_hashes` |
| I25 | 未知设备算子被自动写进豁免清单，随后可被钉到设备上 | 豁免理由对所有非 GEMM 缺产物算子一律放行 | 豁免只允许 host/模型边界/非 DPU GEMM 三类，其余在导出时失败 | `genesim_bridge/placement_export.py`（`_build_pimir_exempt_ops`） | `test_unknown_device_op_is_not_silently_exempted`、`test_legitimate_exemptions_are_registered` |
| I26 | MoE IR 缺 sidecar `moe_config` 时直接放行 | 校验函数把缺失当成了非 MoE | 当前 IR 有专家但 sidecar 缺 `moe_config` 直接失败 | `genesim/src/scheduler/gene_sim_scheduler.py`（`_validate_moe_config`） | `test_sidecar_without_moe_config_is_rejected` |
| I27 | 执行语义由各入口各自读环境变量或硬编 dense_sim，仿真按 128 专家计费 | 语义只是 IR 自描述字符串，没有从编译配置派生 | `compile_llama2(moe_execution=)` 统一解析，经 sidecar 与 parser 传至 GeneSim；三处独立核对 | `runtime/exec_plan_gen.py`、`runtime/compile.py`、`genesim_bridge/placement_export.py`、`genesim/scripts/model_parser.py` | `TestMoeExecutionContract`、`TestMoeConfigLineage`、Qwen3 全链路 |
| I28 | 只在 `_get_operator_with_runtime` 缩放 flops/data_bytes 不改变 PIM 仿真时间：PIM 的时延取自 trace 周期 × 形状循环次数 | 代价模型按算子字段缩放，而 PIMVPU 的执行路径不看 flops | 运行期算子携带 `moe_cost_fraction`，`PIMVPU.execute`/`execute_profile` 把 trace 时延与 memory_bytes 同步折算；该字段不落 IR | `genesim/src/ir/model_ir.py`（`Operator.moe_cost_fraction`）、`genesim/src/vpu/vpu.py` | `TestDynamicDispatchPimLatency`（3 条） |
| I29 | 4 层真实 IR 没有词表维 GEMM，也没有 q_norm/k_norm；`parameter_bytes` 比完整 HF 参数少 622,331,904 字节 | 通用构造器沿用简化 Llama 骨架，只补了 MoE 分支；parser 单独统计 `num_parameters`，图内缺项没有判据 | 补 lm_head GEMM 与逐头 q/k norm、权重驻留按 tensor_id 去重；parser 与主链都断言 `parameter_bytes == num_parameters × dtype 字节`；放置导出补模型级 lm_head 映射与分片 | `genesim/src/ir/model_ir.py`、`scripts/model_parser.py`、`genesim_bridge/placement_export.py`、`scripts/run_full_pipeline.py` | `TestFullGraphCoverage` 4 条、`test_compiler_graph_and_genesim_ir_cover_the_same_weights` 3 条、`test_model_level_lm_head_is_exported_with_its_own_weight` |
| I30 | Q/K 归一化的成本按模型级 hidden=2048 计，实际宽度是 head_dim=128，放大 16 倍 | RMSNORM 配方没有接收逐算子宽度，成本抽取也没走宽度分支（与 round02 I22 同一根因的另一条路径） | `_normalize_ir` 接收 width，成本抽取把 RMSNORM 并入逐算子宽度分支 | `genesim_bridge/op_classify.py`、`genesim_bridge/cost_extractor.py` | `test_rmsnorm_cost_follows_the_operator_shape` |
| I31 | GeneSim 的 TP 展开只遍历层子图，模型级 lm_head 不会被展开成每台 DPU 一份 | 逐层循环天然看不到不属于任何层的算子 | 展开循环之后补模型级多分片分支（只允许 lm_head），按 shards 列切展开并继承 pimir/分块 | `genesim/src/scheduler/gene_sim_scheduler.py` | `test_lm_head_is_expanded_across_tensor_parallel_shards` |
| I32 | 动态仿真对专家计算统一用 top_k/E，MOE_ACCUM 也被缩放；与编译器 any-hit 全张量执行语义不符 | 用 `expert_id` 同时表达「产物归属」与「可门控计算」，共享累加没有独立成本角色 | IR 增 `moe_cost_role`；专家计算/通信按 `1-(1-k/E)^Tq`，共享累加保持 1.0；Relay 运行期标 `expert_comm` | `genesim/src/ir/model_ir.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `TestDynamicDispatchCost` 5 条、`TestDynamicDispatchPimLatency` 3 条 |
| I33 | A2 只按算子编号序列跳过精化；骨架被改内容（词表/参数量）后仍复用旧精化 IR | 模型身份只有少量 MoE 字段，没有结构指纹，缓存键也没有融合清单与成本版本 | IR 增模型身份与结构指纹（双仓同一算法），sidecar 回传、GeneSim 重算比对；A2 缓存键含指纹/档位/融合清单/成本版本 | `genesim_bridge/ir_fingerprint.py`、`scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `test_ir_fingerprint.py` 6 条、`TestModelIdentityFingerprint` 4 条 |
| I34 | 4 层 dense_sim 与 dynamic_dispatch 的 GeneSim 总时长逐位相同（76.1225576142813 s），动态计费实际没有生效 | `ModelIR.split_operator` 四个分支复制了 `semantic_role`/`expert_id` 却没复制 `moe_cost_role`；TP 展开把每个专家算子都换成新分片，角色在这一步丢失 | 四个分支统一复制 `moe_cost_role`；补分片继承角色与管线级「动态 < 稠密」用例 | `genesim/src/ir/model_ir.py`（`split_operator`） | `test_split_operator_keeps_expert_id_and_role`、`test_every_split_dimension_keeps_the_moe_cost_role`、`TestExpertAwareTensorParallelExpansion`、`test_dynamic_dispatch_shortens_expert_compute_on_the_critical_path` |
| I35 | 非 GEMM 成本全部挂 `Tq(Tp+Tq)`，decode 长历史下 VECTOR_MUL/RMSNORM/ROPE 放大 5.7–90 倍 | 拟合按「GEMM / 非 GEMM」一刀切选公式，与配方实际依赖的编译代表点变量脱节 | 成本项进 `OpRecipe` 并随 `Measurement` 传到拟合；`_fit_coeffs` 只按声明项求解；成本版本升到 1.3 | `genesim_bridge/op_classify.py`、`genesim_bridge/cost_extractor.py` | `test_current_token_costs_ignore_kv_history_and_attention_tracks_it` |
| I36 | ROPE/MASK 的单节点成本被放大 `num_heads` 倍（ROPE 实测 12288 vs 逐头 384） | 代表 IR 用了整层所有头，而真实图里每个头一个节点；与 round02 问题5 的「代表形状不跟随算子」同一根因 | `_rope_ir`/`_mask_ir` 改为逐头形状 `[1, Tq, ...]`；测试同时钉住系数 384/1 | `genesim_bridge/op_classify.py` | `test_current_token_costs_ignore_kv_history_and_attention_tracks_it` |
| I37 | 缺 pimir 的算子改放 GPU/CPU 就绕过严格预检；无 trace_source 的 trace 以 unknown 放行 | 覆盖集合由「已落到 PIM 的算子」反推，缺条目会改变落点、让校验对象消失 | 覆盖校验改由源 IR 定义并在放置前执行；豁免按 IR 语义核对；trace 按应覆盖集合与来源白名单对拍 | `gene_sim_scheduler._validate_pimir_source_coverage`、`run_full_pipeline.audit_sidecar_coverage`、`verify_trace_provenance` | `test_missing_op_placed_on_gpu_is_still_rejected`、`test_all_ops_on_gpu_are_still_checked`、`test_audit_sidecar_coverage_*`、`test_trace_provenance_*` |
| I38 | 仅翻转一条依赖方向，两侧结构指纹 SHA256 完全相同 | 指纹只哈希算子清单，没有依赖端点、张量身份与层归属 | 双仓指纹 schema 升到 2，补 dependencies 与 subgraphs；成本系数/设备提示/本地分片继续排除 | `genesim/src/ir/model_ir.py`、`genesim_bridge/ir_fingerprint.py` | `test_fingerprint_tracks_topology_and_layer_ownership`、`test_fingerprint_tracks_dependency_topology_and_layer_ownership` |
| I39 | 真实 4L 主链里 lm_head（op3679）从未展开成分片，也没有 pimir trace，实际落在 GPU 上；round03 的单测用的是「lm_head 不在任何层子图」的夹具，没覆盖真实骨架 | `_expand_multi_shard_operators` 以 subgraph 归属判断模型级算子；model_parser 把最后一层区间一直切到图尾，lm_head 被归进最后一层，逐层循环既不处理它、模型级循环又因 `handled` 跳过它 | 模型级展开改为「层内循环没有替换的多分片算子一律在这里处理」，非 lm_head 直接报错；单测夹具改成真实布局（lm_head 在最后一层子图内）。新的 trace 覆盖校验会把它当成缺 trace 抓出来 | `genesim/src/scheduler/gene_sim_scheduler.py`、`genesim/tests/sim/test_compiler_placement.py` | `test_lm_head_is_expanded_across_tensor_parallel_shards`（真实布局）、4L 全链路 trace 覆盖 |
| I40 | 第 1 层起的专家子图全部失去 gate：2 层探针的 97 条门控命令全在第 0 层 | `_expert_provenance` 没有层域，上一层累加的多专家状态与下一层专家权重相遇后被判成多专家；模块路径这一现成的层号信号没用上 | provenance 改为 (层, 专家)/(层, None)，节点层号取 `nn_module_stack`/权重名并回退到输入；跨层输入按中性，层内多专家汇合在累加出口终止 | `runtime/exec_plan_gen.py`（`_node_layer_of`、`_expert_provenance`、`_expert_tag_of`） | `test_every_layer_gates_its_own_experts`、`test_pp_stages_each_gate_their_own_layers`、`test_ep_16_dpu_generate_matches_hf`、`test_tiled_dispatch_tokens_match_hf` |
| I41 | `where`/`ne` 被下发到设备，却没有对应 PIM 原语；设备 kernel 内部把 `where` 折成加性 `pim.mask` 才能跑 | PIM 原语集（FlagTree 的 eltwise 加减乘除与加性 mask）没有逐元素 select；这两个目标不在 HOST_ONLY 里 | 把 `ne.Scalar`/`where.ScalarOther` 列入 HOST_ONLY，选择与随后的加权累加同为主机共享工作；删除设备侧 dead 内核与切分规则 | `graph/partition.py`、`graph/spec_prop.py`、`runtime/kernels.py` | `test_moe_tail_is_host_shared_work`、`test_plan_and_genesim_ir_agree_on_moe_roles` |
| I42 | GeneSim 的 host `MOE_ACCUM` 只计加权乘与累加的 2×hidden FLOPs，选择本身没计 | 抽象节点把选择并进主机累加时没有同步成本公式；首专家的差异只体现在输入/字节上 | 首专家 2×hidden（选择+加权乘）、其余 3×hidden（再加累加）；成本版本升 1.4 强制重精化 | `genesim/src/ir/model_ir.py`、`genesim_bridge/cost_extractor.py` | `test_host_tail_covers_select_and_weighted_accumulation`、4L 精化 IR 首/中专家系数 4096/6144 |
| I43 | `absorbed_ops` 只要写进 sidecar 就能让任意设备算子从覆盖校验里消失，还能拿到一条空 trace | 两侧都把「已吸收」当成无需证明的事实，没有与融合清单、主算子 pimir 和 IR 依赖边交叉核对 | 两侧新增 `_validate_absorbed_ops`：只允许激活、唯一生产者必须是带 pimir 的主算子 GEMM、三元组必须在融合清单里；trace 验收同样调用 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | compiler 4 条、genesim 4 条 absorbed 用例 |
| I44 | 容量契约只比对单 DPU 总容量：删掉 `hw_budget`、写 `sys_reserve=0/align=1` 都能进仿真 | `_validate_hw_budget` 缺字段直接返回、只比较 mram；导出与覆盖审计没有容量字段检查 | 三件套不可拆分：严格产物缺字段/类型错误/任一值偏离即失败；导出、审计、trace 验收、直接 GeneSim 共用同一规则；旧非严格产物保留兼容 | `scripts/run_full_pipeline.py`、`genesim_bridge/placement_export.py`、`genesim/src/pim/capacity.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | compiler 7 条、genesim 7 条容量用例 |
| I45 | 首版吸收校验要求「融合清单每条都必须有 absorbed 编号」，在 B+C+D 阶段误报 512 条 | `absorbed_ops` 由精化步骤（A2）回写，BCD 审计时还没生成；两个字段由不同阶段产出 | 审计只校验 absorbed → 清单方向；清单自身的真实性由既有的「折激活 kernel 必须带 act_spec」核对；GeneSim 同样只要求 absorbed → 清单 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | 4L 主链 BCD 阶段通过 |
| I46 | 4L 主链 step E 在 TP 展开后第二次跑覆盖校验时报 2048 条「absorbed 编号不在源 IR 里」 | `_expand_multi_shard_operators` 把展开出的分片编号加进 `_compiler_absorbed_ops`（运行期集合），新加的 absorbed 结构校验直接读这个集合，把展开编号当成 sidecar 声明的源编号 | sidecar 声明的源编号单独存 `_compiler_source_absorbed_ops`，结构校验只读它；登记时从当前 sidecar 重建两个集合，避免换产物后残留 | `genesim/src/scheduler/gene_sim_scheduler.py`（`_register_compiler_artifacts`、`_validate_absorbed_ops`） | `test_expanded_shard_ids_do_not_poison_the_source_check`、4L 主链 step E |
| I47 | top-k 选中但权重下溢为 0 的专家被静态前向与 dispatch 同时跳过；选中专家输出 inf 时 HF 得 nan、静态得 0 | round01 的溢出修复用 `routing != 0` 兼任命中条件，选中索引与权重值没有分开 | 静态前向新增 `selected_mask` scatter，掩蔽与命中只用它；dispatch 只读布尔掩码 scatter（`scatter.value` 列入 HOST_ONLY），路由权重只参与加权 | `backend/architectures.py`、`runtime/exec_plan_gen.py`、`graph/partition.py` | `test_static_moe_forward_zero_weight_selected_expert_matches_hf`、`test_decode_and_prefill_dispatch_read_the_selected_mask` |
| I48 | 任何 `expert_id<0` 的 GEMM 都能自动写进 `pimir_exempt_ops`，设备 router/注意力/输出头因此从覆盖与 trace 必需集合消失 | 导出在 host 权重处直接返回，没有保存落点证据；后续只能从专家号猜测 | `export_gemm` 按 `spec.device` 把 host 落点写进 IR 与 sidecar（角色/权重身份），`_build_pimir_exempt_ops` 只复制证据；审计、skip-simulation、仿真前、trace 与直接 GeneSim 逐条核对 | `genesim_bridge/placement_export.py`、`scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | compiler 5 条、genesim 3 条豁免用例；4L 真实产物 router 豁免探针 |
| I49 | sidecar 自报 `align=64`，但 1606 条 B 路 pim mlir 实际是 8；vLLM 又用 256 | `DEFAULT_HARDWARE_CONFIG.dma_align=8` 与 `DPU_DMA_ALIGN=64` 分裂；B 路生成不接收本次硬件配置，`with_budget` 只补缺失属性、保留旧值 | 新增 `default_hardware_config()`/`default_site_capacity()` 单一工厂；B 路显式传硬件，冲突属性直接失败；导出与 GeneSim 逐份核验 pimir 的 `dma-align`/`mram-bytes`；全部 `compile_llama2` 测试夹具改用契约对齐（256/8 的两处另行保留为负例） | `contracts/op_contract.py`、`memory/mem_planner.py`、`genesim_bridge/placement_export.py`、`opcompiler_bridge/driver.py`、`opcompiler_bridge/pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `test_default_hardware_and_budget_share_one_alignment`、`test_vllm_compile_capacity_uses_the_site_factory`、`test_with_budget_rejects_conflicting_module_attributes`、`test_pimir_hardware_attributes_must_match_the_contract`、4L 产物 align 统计 |
| I50 | 同层两个专家的专家号互换后覆盖审计、freshness、trace 验收仍通过 | 编译器把三元组收成集合再求差，丢掉了与 op_id 的对应；freshness 只核哈希、trace 只核来源 | 按源 IR 的每个 op_id 逐条核对角色/专家号/权重路径；`verify_sidecar_freshness` 从配置加载 IR、`verify_trace_provenance` 也调用同一规则；GeneSim 侧同样加权重身份 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | compiler 6 条、genesim 2 条身份用例；4L 真实 sidecar 互换探针在三处均 REJECTED |
| I51 | 48 层 TP 展开耗时 5:14:33；4 层 2581 算子展开 99 秒 | `add_operators`/`remove_operators` 每次全量 `_build_index`，TP 展开逐组调用 | 新增 `ModelIR.batch_edit()`：增量分配编号、暂存增删，退出时一次扫描落实；TP 展开与 relay 插入改批量。修复后 4 层展开 ≤1 秒、48 层只读基准 **6.24 秒**（130036 分片/143192 算子） | `genesim/src/ir/model_ir.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `test_batch_edit_builds_the_index_once`、`test_batch_edit_matches_sequential_result`、`test_expansion_rebuilds_the_index_once`、`test_batch_expansion_matches_sequential_expansion`、48 层展开基准 |
| I52 | 1 专家/top-1 模型被 parser 以「非 MoE」拒收 | 图构造、执行语义与默认分块都用 `num_experts > 1` 当 MoE 阈值 | `is_moe_config`（model_type+有效字段）贯穿编译器与 GeneSim；专家数正整数、top-k 范围校验；`_moe_strategy` 要求 TP 宽度整除专家数 | `runtime/exec_plan_gen.py`、`runtime/compile.py`、`backend/architectures.py`、`genesim/src/ir/model_ir.py` | `test_single_expert_moe_keeps_moe_execution_semantics`、`test_single_expert_moe_generate_matches_hf`、`test_single_expert_is_still_moe`、`test_cli_accepts_a_single_expert_moe` |
| I53 | 多文件跑回归时 vLLM 子进程报 `No module named 'backend.vllm_executor'` | 测试把 GeneSim 的 `src` 插到 `sys.path[0]`，两仓都有 `backend` 包，遮住本仓的；spawn 子进程按 sys.path 重新导入 | 三处 GeneSim 导入改为追加到 `sys.path` 末尾，编译器根保持优先 | `tests/test_architectures.py`、`tests/test_ir_fingerprint.py`、`tests/test_moe_dispatch.py` | 跨文件组合运行 preemption/prefix/chunked 用例 |
| I54 | 全量回归里 `test_pimir_sidecar_agrees_with_ttir_on_flops` 报覆盖集合不同；`test_refined_ir_preserves_structure` 报结构不同 | 工作区 `genesim/models` 的 llama2 精化产物来自不同日期的骨架版本：flagtree 产物重跑后与当前骨架同源，旧 pimir 产物（version 1.1、无指纹）仍是旧编号 | 重新生成 flagtree 产物；sidecar 对比用例在 `ir_fingerprint` 缺失/不符时跳过并说明是旧产物，不把产物陈旧误判为代码回归 | `tests/test_genesim_bridge.py` | 全量回归；`test_refined_ir_preserves_structure` 2 条通过 |
| I55 | 全量回归里 `test_no_production_code_reads_dtype_through_a_val_variable` 报 `runtime/exec_plan_gen.py::build_execution_plan` 通过中间变量读 `val.dtype` | 新增的选中掩码判定用 `node.meta['val'].dtype == torch.bool`，绕过统一 IR 的 dtype 入口 | 改读 `node.meta[SPEC_META_KEY].dtype`；dtype 仍只从 spec_prop 的单入口进入 IR | `runtime/exec_plan_gen.py`（`build_execution_plan`） | `test_no_production_code_reads_dtype_through_a_val_variable`、dispatch 用例 |
| I56 | 全量回归里 4 GiB 硬件的 B 路文本被默认 8 GiB 预算判成冲突，量化布局 3 条用例失败 | 只拿到 IR 文本的调用方没有「本次请求」，`with_budget` 却拿 `pim_options()` 的默认值去比对模块头已有的预算属性 | 新增 `module_budget()` 读出模块头预算；文本调用优先沿用已写属性、缺失项才用默认值，显式传硬件时仍严格拒绝冲突 | `opcompiler_bridge/pipeline.py`、`opcompiler_bridge/driver.py`（`_options_hardware`、`_run_oplevel_triton_opt`）、`genesim_bridge/flagtree_driver.py` | `test_text_only_b_path_uses_the_budget_written_in_the_module`、`test_the_quant_layout_survives_into_the_delivered_pimir` 3 条 |
| I57 | round05/round08 已加的 GeneSim absorbed/host 豁免契约用例在标准命令 `./run.sh --test sim` 中从未执行 | 5 个测试文件把 `if __name__ == "__main__": unittest.main()` 写在文件中段，`unittest.main()` 调用时后面的测试类尚未定义，文件仍以 exit 0 通过 | 把入口守卫移到文件末尾；12 条此前从未执行的用例（`test_compiler_placement.py` 的 6 条 absorbed/专家契约 + 另外 4 个文件各 1–2 条）恢复执行并通过，round09 新用例也随之进入标准回归 | `genesim/tests/sim/test_collective_edges.py`、`test_compiler_placement.py`、`test_config_loader.py`、`test_cost_sidecar.py`、`test_gpu_backend.py` | `./run.sh --test sim` 45/45 个文件（详见 §4.3） |
| I58 | dynamic_dispatch 下专家输出到共享累加的真实边按全量计费：E=8/top-1 的 decode 16 B 边计 16 B，按命中并集应为 2 B | round03 引入成本角色时把「算子是否被门控」与「边传输的张量由谁生产」混为一体，`_moe_dependency_fraction` 只读目标角色；round04 把旧式依赖并入同一出口，但方向假设没变。现有测试只构造「普通源→专家目标」，没有「专家源→共享目标」 | 依赖边同时接收源、目的算子：专家生产的输出无论到共享累加、其它设备还是 HOST 都随源专家命中并集；共享/HOST 输入到专家随消费者；shared→shared 全量。放置估算、正式管线、显式与旧式依赖共用同一判定；补专家源→共享/HOST、共享→共享、普通源→专家四方向的 decode/prefill 用例和 TP 展开后用例 | `genesim/src/scheduler/gene_sim_scheduler.py`（`_moe_dependency_fraction`、`dependency_transfer_bytes`、`dependency_transfer_stats`）、`genesim/tests/sim/test_pipeline_execution.py`、`test_moe_execution_cost.py`、`test_compiler_placement.py` | `test_dependency_payload_follows_the_expert_endpoint`、`test_expert_host_egress_is_scaled_by_the_hit_union`、`TestMoeDependencyDirection` 5 条、`test_expanded_expert_output_edges_keep_the_source_role`；4L `tp1_pp4` 实跑通信字节减少 133,693,440 = 68 decode × 1,966,080 B |
| I59 | 实施文档把 dynamic 有效 FLOPs 写成 `5.414125498987654e12`，产物是 `5,415,241,254,987.654`，并据此把与 round03 的差值写成 1.66e10/1.63e10 | 证据数值由人工誊写，没有与引用的 summary/审计 JSON 做机器一致性校验，错误值被复制到测试总览、真实 30B 证据和性能说明三处 | 以 summary 为唯一来源回填全部引用位置；新增 `scripts/check_implement_evidence.py`，从 summary 提取三项指标核对文档，并扫描千分位高精度数字是否可溯源（历史值显式声明）；差值重算为 1.55e10 | `docs/implement-qwen3-moe-20261105.md`、`scripts/check_implement_evidence.py`、`tests/test_implement_evidence.py` | `test_mistranscribed_flops_value_is_rejected` 等 5 条；对当前文档与 4 层三份 summary 运行 checker 通过 |
| I60 | 直接 GeneSim 在配置与 sidecar 同时改成 16 GiB 时 ACCEPTED；配置 16 GiB + sidecar 8 GiB 也会被“配置即契约”放行；MLIR 的 `pim.mram-bytes` 同样从可配置容量反推 | round03/round05 的容量收敛只统一了各入口的取值，没有把需求常量作为独立验收上限；`_validate_hw_budget` 与 `_validate_pimir_hardware_contract` 都从 `resolve_pim_capacity_limits()` 取期望值 | 新增 `DPU_MRAM_BYTES=8 GiB` 常量；严格路径同时断言有效配置、sidecar 三件套与每份 MLIR 属性等于常量；只有明确标识的旧式非严格产物继续按运行配置兼容 | `genesim/src/pim/capacity.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `test_strict_product_rejects_a_raised_runtime_capacity`、`test_strict_product_rejects_a_config_raised_on_one_side`、`test_pimir_attributes_are_checked_against_the_requirement_constant`、`test_legacy_product_keeps_other_capacity_configurations`、`test_contract_matches_the_compiler_values` |
| I61 | `step_e_simulate` 删除共享 `genesim/pim_traces`、读取共享 `genesim/results/summary.json`；A/A2/BCD 固定文件名；同目录两次 E 的临时配置同名覆盖 | round08 只把“本次 sidecar”和日志副本命名绑定到策略，没有覆盖 A/A2/BCD/E 的运行身份；trace 全量重编被实现成删除全局目录 | 运行目录按架构/策略/执行语义/运行号派生；A/0/A2/BCD 产物写本次 `artifacts/`；E 临时配置按 label 命名，并把 `model.ir_path`、`output.pim_traces_dir/results_dir/log_file` 改写为本次目录并回读；trace 目录必须是本次空目录；不再删除共享目录、不再从共享 results 复制 | `scripts/run_full_pipeline.py` | `test_simulation_step_never_deletes_shared_outputs`、`test_interleaved_runs_keep_their_own_traces_and_summaries`、`test_prepare_simulation_config_binds_the_run_placement`、`test_run_directory_is_derived_from_artifact_and_strategy`、`test_trace_provenance_reads_the_configured_trace_dir` |
| I62 | dynamic 与 dense 的时间/吞吐/FLOPs 对调、并写入 `difference=9.99e99` 后 checker 仍返回通过；4 层可冒充 48 层、历史值可冒充 current | checker 把全部产物值放进一个全篇 `allowed` 集合，只检查“文档任意位置是否出现”；没有标签→summary→指标的结构，也不重算派生值；高精度扫描只覆盖千分位格式 | 实施文档嵌入机器可读证据块；checker 逐 current 标签加载同名 summary，交叉核对层数/模型/执行语义并逐值比较；派生 subtract/ratio/greater_than 由同一组指标重算；history 与 measurements 独立命名空间；正文扫描补科学计数法、比值与时间格式 | `scripts/check_implement_evidence.py`、`docs/implement-qwen3-moe-20261105.md` | `test_swapped_dynamic_and_dense_values_are_rejected`、`test_four_layer_summary_cannot_pose_as_forty_eight`、`test_history_value_cannot_be_declared_as_current`、`test_wrong_derived_difference_is_rejected`、`test_wrong_derived_ratio_is_rejected`、`test_wrong_derived_conclusion_is_rejected`、`test_scientific_notation_outside_the_evidence_block_is_rejected` |
| I63 | 配置显式 `require_compiler_pimir=true` 时，删除 sidecar 的 `requires_pimir` 与 `hw_budget`（或删除自报后写 4/16 GiB），freshness、仿真 IR、trace 与共享审计仍 ACCEPTED | 严格性只由产物自报推导：`validate_sidecar_contract` 没有严格性入参，`audit_sidecar_coverage` 只在 `sidecar.get("requires_pimir")` 为真时核对容量；三个复核入口读了配置却不传状态 | 共享验证器新增 `strict_pimir` 入参，与 sidecar 自报取并集；五个入口显式传入，严格时按 8 GiB 契约核对三件套，只有配置与产物都明确非严格才走旧式兼容 | `scripts/run_full_pipeline.py`（`validate_sidecar_contract`、`audit_sidecar_coverage`、`_config_requires_pimir`） | `test_strict_config_requires_the_capacity_contract` 等 14 条；探针 `probe-strict-capacity.log` |
| I64 | 证据块正确、只交换正文的 dynamic/dense 时间/吞吐/FLOPs 后 checker 仍返回 `problems=[]、unknown=[]`；4 层冒充 48 层、历史冒充 current、正文错误比较都不报 | 正文仍用全篇 `allowed` 集合；必需值检查搜索整个 `doc_text`（含证据块），没有「运行标签 → summary → 指标 → 正文锚点」绑定，也不核对大小结论 | 证据块增 `prose_aliases`；正文数值按所在块的前置标签绑定，块内前置标签未声明过该值即失败；必需值只扫去掉证据块的正文；带数值的比较结论按实际值判定 | `scripts/check_implement_evidence.py`（`_run_aliases`、`_label_spans`、`_token_context`、`_prose_context_problems`、`_check_comparisons`） | `test_prose_only_swap_of_dynamic_and_dense_is_rejected` 等 6 条；探针 `probe-evidence-prose.log` |
| I65 | paged_attn 的 PV 相位与主机/契约相差一次 f16 取整；单块探针最大差 0.00048828125 | 相位存储边界没有统一表驱动，PV 手写循环漏掉 `roundF16`；测试参考与 lowering 共享同一未取整假设 | 在 lowering 的 PV 累加前落 f16；补相位边界表；测试参考独立按契约实现，并断言数据能区分两种口径 | `FlagTree/.../LowerPIMToEmitC.cpp`、`runtime/kernels.py`、`tests/test_opcompiler_ops.py` | `test_paged_attn_matches_the_online_reference`、`test_paged_attn_multi_block_tail_matches_the_online_reference`、`paged_attn_emitc.mlir` |
| I66 | 块表只判 `slot < poolSlots`，`-1` 直接算出负偏移；上界非法项被静默换成槽 0；主机兜底不查上界 | 块表有效性没有共享契约，低位只是注释与单侧 if；写 KV、融合内核、主机兜底三条路径各写一套 | 抽出 `_physical_slot`/`_validate_block_table`，主机启动前逐项预检；设备侧 `pim_block_slot` 中止非法项；三条路径共用 | `runtime/kernels.py`、`FlagTree/.../LowerPIMToEmitC.cpp` | `test_physical_slot_*`、`test_block_table_tail_sentinel_is_not_read`、`test_paged_attention_rejects_invalid_slots_before_addressing`、`paged_attn_emitc.mlir` |
| I67 | 配置显式严格、但删 sidecar 键/IR 键/sidecar 文件/IR 文件/配置本身时，freshness、仿真 IR 与 trace 仍 ACCEPTED | 兼容分支「无对象就跳过」在读取严格性之前执行；`_config_text` 把缺文件当成没有可核对对象 | 先读配置并算出严格性（配置 ∪ 自报），再判断输入是否齐全；严格时任一缺失直接失败，非严格才允许旧式无 sidecar | `scripts/run_full_pipeline.py`（`_config_text`、`_strict_sidecar_context`、`_require_simulation_input`、三个 verify 入口） | 严格入口 12 条矩阵用例、非严格旧式正例、自报严格缺 IR 键反例 |
| I68 | trace 只凭来源标签与父编号即可通过；改错哈希、删哈希、换类型、换父编号、非空 absorbed 都不报 | sidecar 文件哈希与 trace 编译签名分属两条未连接的链；absorbed 只检查来源标签 | 逐条核对父算子、类型与 `compile_signature.pimir_sha256`；absorbed 要求零指令且编号在 sidecar 吸收集合；真实 4L/48L 产物作正例 | `scripts/run_full_pipeline.py`（`_trace_source_lineage`、`verify_trace_provenance`） | 错哈希/缺哈希/错类型/错父编号/非空 absorbed/未知吸收编号 6 条反例与分片正例 |
| I69 | 同块两个运行都在值前时，正文把 dense/dynamic 的语义对调仍通过；必需指标只写无标签前言也通过；容量/通信/trace 计数完全不在绑定范围 | 归属用「块内任一前置标签集合」而非最近标签；必需值只做全篇字符串存在；数值模型只覆盖 summary 三项指标 | 最近标签归属、前者/后者身份解析、必需值运行语境检查；新增 `facts` 声明非 summary 证据值（按运行绑定或独立命名空间） | `scripts/check_implement_evidence.py`（`_effective_labels`、`_nearest_labels`、`_value_in_run_context`、`_fact_bindings`）、实施文档证据块 | 同块对调/身份正例/无标签前言/facts 绑定/未知运行标签 5 条新用例 |
| I70 | FlagTree 重建后既有 4L/48L sidecar 的 pimir 哈希是否仍有效需要确认 | 重建会改变 `triton-opt`/`libtriton` 的指纹，但 sidecar 哈希只覆盖 pimir MLIR 文本；ExpandPhases 的相位表只改注释，不改输出 | 只改 lowering 与注释，不改 `phases` 属性；重建后重跑 4L/48L 的 freshness/仿真 IR/trace，全部 ACCEPTED，无需重跑 48 层导出 | `FlagTree/.../ExpandPhases.cpp`、`scripts/run_full_pipeline.py` | 4L/48L 三入口重跑 ACCEPTED |
| I71 | 新加的 trace 哈希校验让 3 条 E 阶段夹具与严格契约夹具失败 | 测试夹具只写来源标签，sidecar 有真实哈希而 trace 没有；`_fake_sim_run` 没有把 sidecar 哈希带进 trace | 夹具按 sidecar 写入 `compile_signature` 与 `num_instructions`；契约夹具按每个算子条目生成哈希 | `tests/test_genesim_simulation.py`（`_write_trace`、`_write_pimir_traces`、`_fake_sim_run`） | 314 条相关用例通过 |
| I72 | 真实 4L sidecar 改一个条目的 `pimir_path` 为不存在文件、删哈希或换内容后，仿真 IR 与 trace 入口仍 ACCEPTED；freshness 对缺失文件抛 `FileNotFoundError` 而不是 `StepFailed` | 产物校验被建模成共享验证器的可选参数 `check_artifacts=False`，三个复核入口各自决定不打开；`stale_pimir_entries` 直接 `read_bytes`，没有把读失败纳入错误口径 | 产物校验改为严格上下文的固定规则（自报 ∪ 调用方显式严格即逐条核对可读性与哈希），删除 `check_artifacts` 参数；`stale_pimir_entries` 把缺路径/`OSError` 转成带 op 与路径的 `StepFailed`；补三入口 × 五类破坏矩阵 | `scripts/run_full_pipeline.py`（`validate_sidecar_contract`、`audit_sidecar_coverage`、`stale_pimir_entries`）、`tests/test_genesim_simulation.py` | `test_strict_entry_requires_a_valid_pimir_artifact`（15）、`test_sidecar_self_report_requires_a_valid_pimir_artifact`（3）、真实 4L 破坏探针 |
| I73 | 不传配置调用 `verify_trace_provenance(root)`，对一条 `pimir` 且编译签名为 64 个零的 trace 返回 ACCEPTED | 严格性被设计成「调用方可选传配置」，函数保留 `config is None` 的旧式返回；现有测试把旧行为固化成期望 | 配置改为 trace 验收的必需输入，缺失直接 `StepFailed`；模板/unknown 用例改传最小配置，头部解析用例直接测 `_trace_metadata` | `scripts/run_full_pipeline.py`（`verify_trace_provenance`）、`tests/test_genesim_simulation.py`、`tests/test_placement_export.py` | `test_trace_entry_requires_a_configuration`、模板/unknown 用例、真实 4L 配置下 ACCEPTED |
| I74 | 干净检出首次运行 E 阶段报 `File not found: traces/llama2_7b.trace`；补文件后同一命令 exit 0 | Qwen3 配置沿用 llama 开发环境里生成、且被 `.gitignore` 排除的 trace；主链只改写请求数/IR/sidecar/输出目录，没有生成或校验请求分布输入 | `prepare_simulation_config` 接收 GeneSim 根：配置引用的 trace 缺失时，用官方 `trace_generator.py --synthetic --seed 0` 在本次运行目录生成并绑定绝对路径、回读复核；已存在的 trace 原样使用 | `scripts/run_full_pipeline.py`（`_generate_synthetic_trace`、`prepare_simulation_config`、`step_e_simulate`）、`tests/test_genesim_simulation.py` | `test_prepare_simulation_config_generates_a_missing_trace`、`test_prepare_simulation_config_keeps_an_existing_trace`、生产 Qwen3 配置隐藏 trace 后生成成功 |
| I75 | §8.4 把 round08 主链写成「当前代码的 48 层证据」，与 §1.3 的「48 层数值 round06、主链 round08/round10」矛盾 | 完成状态与证据归属仍由人工写在文档里；round14 收敛了 §1.3/§7.1，§8.4 的旧表述没有随之收敛 | §8.4 改为「证据—源码/产物来源—时间—层数—本轮是否重跑」表：48 层数值 round06、主链 A/BCD/A2 round08、E round10、本轮只复核；明确当前工作区只重跑 4 层；8B 未覆盖与 P1-1 用户确认单列 | `docs/implement-qwen3-moe-20261105.md` | checker 对当前文档 + 两份 current summary（dynamic、dense）退出码 0；alt/48l 是 history 正文引用，不由本轮 summary 核验 |
| I76 | MoE 的两行注释挂在 `split.Tensor` 注册项上方；`where_kernel` 的标量广播注释指向 HOST_ONLY 的 `where.ScalarOther`，该分支经 `_KERNELS` 不可达 | 新增注释与镜像分支时没有同步注册表接线，注释与真实调用路径脱节 | 注释移到 `_KERNELS` 后的「故意不在表里」说明段；删除不可达的显式 `broadcast_to`（numpy 与 `kernels_pim.mask` 本就支持广播）；补 `where.self` 广播对拍与主机闭包断言 | `runtime/kernels.py`、`tests/test_kernels.py` | `test_where_kernel_matches_torch_with_a_broadcast_operand`、`test_scalar_where_is_host_only_and_not_a_device_kernel` |
| I77 | B 路 pimir 按源 IR 的全局形状与 128 token 占位编译：4L 的 VECTOR_MUL 分片 `[Tq,192]` 执行的是 128 token×768 宽的 trace（393216/196608 B），VECTOR_MUL 65.244 s、48L 782.930 s 与 dense/dynamic 1.029× 全部建立在这条错误成本上 | 编译器按「代表形状」生成 pimir，运行期按实际 Tq 与本地宽绑定，两套单位没有共同契约；sidecar 不带编译单位，TP 展开无条件把源 pimir 继承给本地分片，严格验收只核路径/哈希，测试夹具又只用 1 token | 定义 B 路编译单位为「1 token × 本地宽」：`Tq` 折 1、历史维仍取一块 KV 占位；MLP act/mul 从相邻多分片 GEMM 的分片取本地宽；sidecar 回传 `trace_tokens` 与 `local_*_shapes`，pimir 函数带 `pim.traceTokens`；GeneSim 在 trace 生成前核 tokens/末维/单相字节，TP 展开只在分片本地形状一致时继承，运行期按 `ceil(Tq/trace_tokens)` 重复；trace 验收（编译器侧）按同一单位核对；SPLIT 的 IR 输出只是代表切片，按纯视图核读写相等 | `genesim_bridge/placement_export.py`、`scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、`genesim/src/vpu/vpu.py`、`genesim/src/pim/pimir_trace.py` | `test_bpath_uses_one_token_and_the_mlp_local_width`、`test_bpath_view_ops_keep_the_global_width_next_to_tp_gemms`、`TestPimirUnitContract` 6 条、`test_bpath_shards_inherit_pimir_only_for_matching_local_shapes`、`test_bpath_shard_shape_conflict_is_rejected_in_strict_mode`、`test_vector_trace_token_unit_scales_with_the_query_rows`、`test_legacy_wide_token_trace_is_not_multiplied_again`、`test_explicit_trace_tokens_attribute_wins_over_shape_heuristic`、`test_trace_lineage_accepts_a_matching_bpath_unit` 等 6 条、`test_trace_lineage_checks_split_as_a_pure_view`；4L `op_type_metrics` VECTOR_MUL 65.244→0.151 s、SPLIT 15.317→0.121 s、4L dynamic 总时长从 88.483 s 降到 round16 的 61.015 s |
| I78 | A 段只比较专家数与专家 GEMM 数，top-k 不进入权重字节：把 4L IR 的 `moe_top_k` 改成 7 并重算指纹后，A/BCD/E 全部放行 | MoE 身份按字段枚举校验，没有从 HF config 派生完整期望元组；sidecar 的 `moe_config` 也只与 IR 自洽 | 新增 `_configured_moe_identity`/`_validate_moe_identity`：A 段与 BCD 段都用 config 的 `num_experts`/`num_experts_per_tok`/`moe_intermediate_size` 逐项核对 IR/sidecar；非 MoE 配置要求三项全零 | `scripts/run_full_pipeline.py`（`step_a_model_ir`、`step_bcd_export`、`_validate_moe_identity`） | `test_moe_identity_rejects_a_top_k_that_disagrees_with_the_config`、`test_moe_identity_rejects_a_wrong_expert_width`、`test_dense_config_rejects_moe_scalars_and_accepts_an_empty_identity` |
| I79 | `contracts/dtypes.py` 的 bool 类型注释仍写「MoE 门控掩码（`routing != 0`）」，与实现的 `selected_mask` 命中口径相反 | round14 问题6 的同类路径扫描只覆盖 GeneSim 测试与设计/实施文档，没有扫到 `contracts/` 的新增注释 | 注释改为 top-k 选中索引 scatter 出的布尔 `selected_mask`；新增静态测试防回退；全仓代码注释不再出现 `routing != 0` | `contracts/dtypes.py` | `test_bool_dtype_comment_describes_the_selected_mask` |
| I80 | 把请求 Tq 无条件写入 `runtime_query_tokens` 后，`test_scheduler` 的全具体形状夹具（IR 里没有符号 Tq）从 0.0202 s 变成 0.0602 s | 夹具 IR 是固定规模，请求 seq_len 与算子形状无关；无条件按请求绑定会把固定算子重复执行 | 只有 IR 形状里出现符号 `Tq` 的算子才随请求增长，全具体形状按一次执行；真实 IR 的 B 路算子都带 Tq | `genesim/src/scheduler/gene_sim_scheduler.py`（`_get_operator_with_runtime`） | `tests/sim/test_scheduler.py` 17 条、新的绑定 Tq 用例 |
| I81 | 单位形状改符号模板后，trace 字节核对与测试夹具里的 `int(dim)` 直接抛 `ValueError` | 旧的字节口径默认单位形状全具体，符号模板需要按 trace 的 `tokens` 与 `loop_unit_tokens` 解析后才能乘 | 两仓各增 `_resolved_unit_shape`：`Tq`→trace tokens、`Tp+Tq`→一块 KV token；缺块循环元数据时明确报错 | `genesim/src/scheduler/gene_sim_scheduler.py`、`scripts/run_full_pipeline.py`、测试夹具 | 单位/四节点/字节用例 |
| I82 | 共享单位检查最初按 `entry["shards"]` 区分 GEMM，导致 `_capacity_sidecar` 等最小夹具里的 GEMM（没有 shards 键）被误判成 B 路并要求单位 | GEMM 的判据被表达在 sidecar 字段上，最小夹具不必写空 shards | 按**源 IR** 的 `op_type == "GEMM"` 判定 A 路；sidecar 字段只作辅助 | `scripts/run_full_pipeline.py`（`_bpath_unit_contract_problems`、`_bpath_unit_problems`） | `test_audit_accepts_the_contract_hw_budget` 等容量用例 |
| I83 | 独立入口的共享 `models/` 产物仍是 round15 的 dense_sim 且 1606 条 B 路无单位，`./run.sh --config conf/sim_qwen3_moe_pimir.yaml` 启动即失败 | 模板引用共享产物，产物由主链 run 目录生成后没有回写；模板注释的单步导出不产出 A2 `_pimir.ir` | 用本轮主链产物刷新 6 个共享文件并回填产品一致性测试；模板写明 run 目录配置才是可直接运行的产物，独立输出到 `results/qwen3_moe_standalone/` | `genesim/models/`、`genesim/conf/sim_qwen3_moe_pimir.yaml` | `test_checked_in_qwen3_standalone_products_match_the_compiler_default`、独立入口实跑 |
| I84 | 给 B 路条目加一个假 shard 并删掉单位后，共享验证器、`check_sidecar_entries` 与直接 GeneSim 全部接受；GEMM 置空 shards 也会被当成 B 路 | A/B 路分类被实现成「被验产物自己的 `shards` 自报」，不是源 IR 属性；六个消费点各自读分片字段 | 两仓统一分类规则：源 IR `op_type == GEMM` 才是 A 路，其余算子都是 B 路；B 路非空 shards、严格产物设备 GEMM 空分片直接失败；覆盖审计、单位检查、trace 入口、放置加载、本地形状、TP 展开与继承全部改用该规则 | `scripts/run_full_pipeline.py`（`_entry_class_problems`）、`genesim/src/scheduler/gene_sim_scheduler.py`（`_validate_entry_classes` 等） | `test_bpath_entry_with_a_fake_shard_is_rejected`（3 入口）、`test_device_gemm_without_shards_is_rejected`（3 入口）、`test_skip_simulation_rejects_a_fake_shard_on_a_bpath_entry`、`test_bpath_entry_with_a_fake_shard_fails_classification`、`test_device_gemm_without_shards_fails_classification`、两仓 GeneSim 分类用例 |
| I85 | 把 `trace_tokens` 改成 128、本地宽改成 384、或把 `pimir_path` 指向旧 128-token/无 traceTokens 的 pimir 后，freshness 与仿真 IR 仍 ACCEPTED | 共享验证器只核单位的「存在性与符号维角色」，从不读 pimir 实产；单位是 sidecar 自报字段，trace 入口与 GeneSim 各自再读一次，形成两套口径 | pimir 成为单位唯一真源：按路径解析一次 `pim.traceTokens` 与参数/结果形状，要求 sidecar 单位数值与之一致；GeneSim 生成 trace 时对同一份单位再核一次；SPLIT 纯视图只比输入末维 | `run_full_pipeline._pimir_unit_facts`/`_pimir_unit_problems`、`gene_sim_scheduler._validate_unit_against_pimir`、`pimir_trace.pimir_unit_facts` | `test_unit_tokens_must_match_the_pimir`（2）、`test_unit_feature_width_must_match_the_pimir`（2）、`test_unit_must_match_the_pimir_actually_referenced`（2）、`test_pimir_without_trace_tokens_is_rejected`（2）；4L 真实产物 ACCEPTED |
| I86 | 全量回归在缓存锁测试上偶发 `_queue.Empty`，且 pytest 打印 summary 后进程被 fork 子进程挂住 | 串行化判据用「5 秒内收到四条队列消息」表达，超时是机器负载的函数；teardown 没有回收子进程 | 等待上限改为 60 秒死锁检测；`finally` 中 join/terminate/kill；新增首次 put 晚于 5 秒的慢调度用例（旧写法实测 `_queue.Empty`） | `tests/test_opcompiler_linear.py`（`_drain_lock_events`、`_reap_processes`） | `test_cache_key_lock_waits_for_a_slow_schedule`、`test_cache_key_lock_serializes_processes` |
| I87 | 实施文档声称 checker 以四份 summary 通过，实际 alt/48l 是 history，按该命令执行返回 exit 1 | 文档把「current 可核验」与「history 只做正文引用」混写成一个命令，且没有测试执行文档声明 | 声明改为实际命令（dynamic/dense 两份 current summary）；新增 CI 用例按证据块 current 标签执行文档命令并断言退出码 0 | `docs/implement-qwen3-moe-20261105.md`、`tests/test_implement_evidence.py` | `test_documented_checker_command_only_uses_current_summaries` |
| I88 | 4L 容量把 per-TensorPU 的 `max_resident_bytes=779,673,600` 与 per-cluster 上限 8,589,934,592 配成一对，集群利用率被读成约 9% | 从 summary 里挑了同名不同 scope 的两个字段，没有用 `pim_capacity_metrics.json` 的集群字段 | 三处引用改按集群口径 `max_cluster_resident_bytes=3,140,834,304 / 8,589,934,592`（36.6%）；per-TensorPU 最大值与 512 MiB 上限单列；证据块补 4L 集群/单 PU 容量 facts | `docs/implement-qwen3-moe-20261105.md` | checker 退出码 0；4L 容量 facts |
| I89 | 8B 测试手写 8 GiB/64 MiB/对齐，30B 测试手写 `PIMHardwareConfig`，契约变化不会被测试发现 | round03 只改了 30B 的 `HwBudget`，没有覆盖真实测试的整套三件套，也没有静态守卫 | 两处都改为 `default_site_capacity()` 派生；新增静态守卫扫描手写容量/对齐/WRAM 字面量与手写构造 | `tests/test_qwen3_8b_real.py`、`tests/test_qwen3_30b_real.py`、`tests/test_architectures.py` | `test_real_weight_tests_derive_capacity_from_the_single_factory`（2 参数）、4 层真实目标命令 |
| I90 | A 路 GEMM 的 `pimir_path` 全部指向一份 B 路逐元素产物、或换成另一分片宽度、或删掉 `kernel_tile_n` 后，编译器五入口与直接 GeneSim 全部接受，仿真总时长从 60.785 s 降到 1.696 s | round18 只定义了 B 路编译单位，A 路只绑定路径与哈希，没有绑定「这份产物必须是该 GEMM 的实测分块 GEMM」；共享验证器对 GEMM 直接跳过解析，GeneSim 在 `unit is None` 时提前返回；契约测试夹具还把同一份 LUT pimir 同时挂在 GEMM 与 B 路条目上 | 两仓各自按源 IR 逐条解析一次 GEMM 的 pimir：必须是 `tt.dot`/`pim.matmul` 单原语、不带 `traceTokens`/相位链；`tt.dot` M=1，K/N 等于 pimir tile 属性，`pim.tile-n` 等于 `kernel_tile_n`，K/N 循环与各分片本地宽对应、N 不大于本地输出宽；融合清单声明的激活必须与 `tt.dot` 的 `act_spec` 双向一致。共享验证器、GeneSim 覆盖校验与 trace 单位校验共用同一规则 | `scripts/run_full_pipeline.py`（`_apath_unit_problems`、`_apath_facts_problems`、`_pimir_unit_facts`）、`genesim/src/scheduler/gene_sim_scheduler.py`（`_gemm_pimir_problems`）、`genesim/src/pim/pimir_trace.py`（`pimir_gemm_facts`） | 编译器 5 条 A 路用例；GeneSim 6 条；round19 探针 `tampered_allgemm`/`tampered_gemm_pimir`/`tampered_notile` 两仓 REJECTED，4L/共享 dense/llama2 真实 A 路产物 ACCEPTED |
| I91 | 把 `lm_head` 这类设备 GEMM 标成 `device_hint=cpu` 并删掉分片/角色/分块后，freshness、仿真 IR 与 skip-simulation 接受，直接 GeneSim 以 45.895 s 跑完并声明 `dynamic_dispatch` | `device_hint` 是条目自报的开关，`_entry_is_host` 只读它；round18 只把 `shards` 从分类依据里删掉，host 判定没有完成 | host 判定改成三件套证据：源 IR `device_hint=cpu`、`pimir_exempt_ops` 登记、GEMM 的 `host_gemms` 角色/权重证据；`operators` 条目自称 host 却缺任一证据直接失败。统一用于 `check_sidecar_entries`、`_entry_class_problems`、`_assert_placement_matches_run`、GeneSim `_validate_entry_classes` 与放置加载 | `scripts/run_full_pipeline.py`（`_entry_is_host`）、`genesim/src/scheduler/gene_sim_scheduler.py`（`_entry_is_host`、`_validate_entry_classes`） | 编译器 host 自报 4 条（含三入口与 skip-simulation）；GeneSim 1 条；`tampered_lmhead`/`tampered_hostgemm`/`tampered_router` 两仓 REJECTED |
| I92 | `SILU` 条目改成 `VECTOR_ADD` 后 freshness/仿真 IR/skip-simulation 接受、直接 GeneSim 才拒绝；删设备 GEMM 的 `semantic_role` 共享验证器仍接受；同形同字节的 add/mul 互换与 SILU/GELU 互换可签发 | 共享验证器只把 `op_type` 当「mnemonic 是否认识」的弱校验，从不与源 IR 比较；trace 入口反而用条目自报的 `op_type` 选字节判据；`semantic_role` 与逐元素 kind 没有纳入源 IR 契约 | 类检查逐条比较 `op_type` 与源 IR 类型、设备 GEMM 的 `semantic_role` 与投影身份；B 路逐元素算子解析 pimir 核对 add/mul；trace 字节判据改用源 IR 类型；两仓用同一张 op_type→kind 表 | `scripts/run_full_pipeline.py`（`_entry_class_problems`、`_bpath_unit_contract_problems`、`_bpath_unit_problems`）、`genesim/src/scheduler/gene_sim_scheduler.py`（`_validate_entry_classes`） | 编译器 `op_type` 互换、缺 `semantic_role`、kind 互换、trace 字节判据 4 条；`tampered_optype` 两仓 REJECTED |
| I93 | 工具链整条缺席（`matmul` 也回退）时，主机分页注意力既不在 QK×scale 后落 f16，也不在 PV 乘积后落 f16，与注释/不变式不符；现有用例只关 `_compiled_paged_attn`，`matmul` 仍走编译内核 | 主机回退把取整委托给 `matmul` 的返回 dtype，而工具链缺席时 numpy 返回 f32；测试夹具只强制 paged_attn 走主机 | `_host_paged_attention` 显式落边界：QK 点积与乘 scale 后各落一次 f16，每块 PV 乘积落 f16 再加进 f32 累加器；新增同时关闭两个编译内核的用例，按相位表参考逐元素对拍并验证数据能区分取整 | `runtime/kernels.py`（`_host_paged_attention`）、`tests/test_kernels.py` | `test_sdpa_host_fallback_rounds_at_the_phase_boundaries` |
| I94 | 4L 产物出现 `max_utilization=145.23%` 而 `fits=True`；文档把 `max_resident_bytes=779,673,600` 与 512 MiB 上限配成「per-TensorPU 最大值」 | 一个字段同时承载「常驻总量」与「按上限判定的量」两种 scope；round18 只改文档引用，没把指标按 scope 拆开 | `max_utilization`/`max_tensor_pu_utilization` 改用与 `fits` 相同的判定分子；新增请求期利用率与集群常驻利用率，总量比单列 `max_total_resident_ratio`；summary/控制台/文档同步新口径 | `genesim/src/pim/capacity.py`、`genesim/src/main.py`、`docs/implement-qwen3-moe-20261105.md` | 容量用例 1 条（判定/请求期/集群/总量比四口径）、`test_main` 渲染用例 |
| I95 | 证据 checker 的 CI 用例在 current summary 缺失时抛 `NameError: name 'pytest' is not defined` | 测试文件用 `pytest.skip` 但没导入 `pytest`；历轮 current summary 都存在，该分支从未执行，缺口一直没暴露 | 补 `import pytest`；round20 的 dense 新 summary 在写完前会触发该分支 | `tests/test_implement_evidence.py` | `test_documented_checker_command_only_uses_current_summaries` |
| I96 | 全量回归中 `test_cache_key_lock_serializes_processes` 在 `_drain_lock_events` 等满 60 秒后以 `_queue.Empty` 失败；同一用例单独运行 2/2 通过 | 测试用 fork 创建子进程，父进程刚跑完整套回归、带着多线程与队列状态；fork 子进程的 Queue feeder 线程可能丢事件。round19 只把等待上限从 5 秒改成 60 秒，等待时间不是根因 | 两个锁用例改用 spawn 上下文，worker 提到模块级（spawn 只按模块名导入），缓存目录随参数传入并在子进程内设置 `driver._CACHE_DIR`；串行化断言保持不变 | `tests/test_opcompiler_linear.py`（`_lock_worker`、两个 `cache_key_lock` 用例） | `test_cache_key_lock_serializes_processes`、`test_cache_key_lock_waits_for_a_slow_schedule`；全量回归 0 失败 |

## 四、测试验证结果

### 4.1 测试执行总览
- 目标命令：`QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider`
  - 4 层真实权重切片（当前代码，本轮 round20 重跑）：
    **36 passed、1 skipped、2 warnings in 242.71s（0:04:02）**，日志
    `/tmp/r20-fix-4l-tests.log`；参考为编译后恢复的 HF 原始稀疏前向；8B
    无权重 SKIP。2 条 warning 是未选中专家溢出用例里 `exp`/cast 的预期数值
    告警。用例集合与 round19 相同（round20 只改 B 路 pimir 复核与容量字段
    口径，不改真实权重生成），结果逐项一致，证明语义指纹与容量 scope 改动
    没有改变数值路径。
  - 48 层全模型：round06 源码数值 **26 passed、1 skipped in 17254.56s（4:47:34）**，
    30B 通过、8B SKIP；round01 历史日志 23 passed、1 skipped（25426.83s）；
    round02 的 `full30b-r2` 为 oom-kill（环境阻塞）。round08 用当轮源码重跑
    48 层主链（A→0→BCD→A2→E）**exit 0**：total_time_s=566.863、
    tokens/s=1.402、trace `{'pimir': 106230, 'absorbed': 24576}`；round10 在
    round08 的 48 层 A/BCD/A2 产物上用当轮 GeneSim 代码重跑 E 阶段，
    **与 round08 逐位一致（total_time_s=566.863、tokens/s=1.402、通信字节
    166,702,607,112）**，审计 ACCEPTED。round11～round16 沿用该 summary 作为
    48 层既有证据，不重复启动长任务；当前 round20 工作区只重跑了 4 层。
    round16/17 的单位契约使旧 48 层 B 路条目缺少 `trace_tokens`/`local_*_shapes`
    与符号维，当前验证器在 freshness、仿真 IR 与 trace 三入口都 REJECTED
    （13106 条 B 路条目），48 层成本证据失效，见 8.4/K16/Q12；
    48 层数值用例保留 round06 历史。
- 本机无 Qwen3-8B 权重：8B 用例 SKIP，如实记为未覆盖，不作为通过。
- 全链路历史记录（Qwen3 入口，4 层切片，round11 重跑；`tp1_pp4` 对照继续
  沿用该轮）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`
  → dynamic_dispatch 退出码 0，默认日志目录下运行目录
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261111-190527-2004830/`
  （按架构/策略/语义/运行号派生）：A/0/A2/BCD 产物在 `artifacts/`，E 使用
  `pim_traces_pimir/` 与 `results_pimir/summary.json`；trace
  `{'pimir': 8858, 'absorbed': 2048}`、**total_time_s=88.483、tokens/s=8.985**，
  E 阶段绑定本次 BCD 的 `tp4_pp2` sidecar；该轮同模型的 dense_sim 对照是历史值，已被
  round16 的单位修正与 round17 的重复次数修正作废（当前口径见下一条）。
  两种语义的结构化审计（`/tmp/qwen3-review-round11-fix/audit-dynamic.log`、
  `audit-dense.log`、`audit-alt.log`）：
  3681 算子、3163 条 sidecar、1536 个专家 GEMM、23 份 pim mlir 全为
  `(dma-align=64, mram-bytes=8 GiB)`、哈希全部有效（dynamic/dense 的
  tiles `[32,128,512]`，`tp1_pp4` 为 `[128,512]`）、
  豁免 518、融合/吸收 512/512 双向闭合、512 个 VECTOR_MUL 全为
  `pim.eltwise<mul>`、必需 trace 缺失 0；dense 有效 FLOPs
  `5,826,198,177,867.654` 高于 dynamic 的 `5,415,241,254,987.654`。
  同一模型的 `--num-dpus 4 --num-stages 4 --log-dir
  /tmp/qwen3-review-round11-fix/pipeline-alt` → 退出码 0：E 阶段绑定
  `tp1_pp4` sidecar，op152 本地输出 768（tp4 为 192）、lm_head 输出
  `[Tq,151936]`，trace `{'pimir': 2651, 'absorbed': 512}`、
  **total_time_s=138.589、tokens/s=5.736**；通信字节
  **8,218,497,128**，比修复前的 8,352,190,568 少 133,693,440
  （68 个 decode × 1,966,080 B），直接验证专家输出边按命中并集折算。
  两次策略结果不同，证明 E 消费的是各自 BCD 的产物。结果见 8.2/8.4。
- 4 层全链路（含仿真）当前代码（本轮 round20 重跑，`tp4_pp2`，验证 B 路
  语义指纹与容量 scope 字段；校验改动不改变 trace 生成路径）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`
  → dynamic_dispatch 退出码 0（PIPELINE_EXIT=0）；运行目录
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/`
  （A/0/A2/BCD 在 `artifacts/`，E 在 `pim_traces_pimir/`、`results_pimir/`）；
  E 阶段临时配置带 `scheduler.require_compiler_pimir: true`，依次通过
  freshness（含 pimir 产物可读与哈希）、仿真 IR、完整产物契约（含 B 路
  语义指纹）与逐条 trace lineage（含 B 路单位）核对；trace 来源
  `{'pimir': 8858, 'absorbed': 2048}`、
  **total_time_s=60.785、tokens/s=13.079**、有效 FLOPs
  `5,415,241,254,987.654`（重复次数修正不改变 IR 系数）、结构指纹
  `4519c683…`；1606 条 B 路条目全部带 `trace_tokens` 与符号形状模板，
  15 份 B 路 pim mlir 带 `pim.traceTokens`；容量 `fits=True`，
  `pim_capacity_metrics.json` 的集群常驻
  `max_cluster_resident_bytes=3,140,834,304 / 8,589,934,592`（36.6%）；
  summary 的 `max_utilization` 是判定口径利用率（模型常驻按 cluster 时分子是
  `request_bytes`，与 `fits=True` 同源）；`max_resident_bytes=779,673,600` 是
  单 TensorPU 常驻最大值（含该 TensorPU 的模型分片与请求期数据），
  `max_total_resident_ratio=1.452` 是该值与单 TensorPU 上限的诊断比，只报告、
  不参与 `fits`，也不与集群上限配成利用率；请求期利用率
  `max_tensor_pu_request_utilization=0.30%`（分子 1,626,112 B）与
  `max_cluster_utilization=36.6%` 分开输出。本轮精化 IR 与 round19 产物
  逐字节相同（sha256 `43634aac…`），sidecar 仅 `source_ir` 路径不同，说明
  校验改动没有改变产物本身。日志 `/tmp/r20-fix-pipeline.log`。新 4L 产物的
  freshness/仿真 IR/trace 三入口 ACCEPTED；从同一侧车删掉 1606 条 B 路条目
  的 4818 个单位字段后三入口全部 REJECTED（`/tmp/r17-probes/sim-stripped.yaml`）。
- 执行期 A/B：`python scripts/bench_moe_dispatch.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --new-tokens 2 --repeat 3 --warmup 1 --num-dpus 8`
  → prefill 1.148×、decode 1.202×（token 完全一致；用户已接受为 P1-1 验收）。
- GeneSim 单元测试：`cd genesim && ./run.sh --test sim` → **45/45 个测试文件通过**
  （本轮 round20 重跑，exit 0；日志 `/tmp/r20-fix-genesim-sim2.log`；
  `test_compiler_placement` 149 条、`test_pim_capacity` 14 条、
  `test_main` 14 条全过）。本轮新增 4 条 GeneSim 用例：B 路语义互换反例、
  实产正例与覆盖闸门反例 3 条，容量 scope 字段 1 条；另把覆盖/签名/GEMM/
  GELU 夹具改成按类型构造的语义正确产物。
- 实施证据一致性：`python scripts/check_implement_evidence.py --doc
  docs/implement-qwen3-moe-20261105.md --summary
  dynamic=test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/results_pimir/summary.json
  --summary dense=/tmp/r20-fix-dense/qwen3_moe_tp4_pp2_dense_sim_20261112-182507-3141801/results_pimir/summary.json`
  → 退出码 0（输出「实施文档数值与产物一致」）。证据块（本节末尾）是唯一取值
  来源：current 条目逐标签回读同名 summary，核对层数/模型/执行语义并逐值比较；
  派生比值由同一组指标重算；48 层 summary、round16 对照与 round11 `tp1_pp4`
  改用 history 命名空间登记，不能冒充 current。
- 编译器全量回归：`python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider --basetemp=/data/fjg/pengcheng/flagOS/.pytest-tmp-r21`
  → **2207 passed、11 skipped、52 deselected、2 warnings in 1373.42s**，失败 0，日志
  `/tmp/r20-fix-regression.log`，见 §4.3。


- FlagTree 重建（round14 源码修复后，round15/round16 未重建）：
  `cd /data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake && ninja -j 32 triton-opt libtriton.so`
  → exit 0（426 个构建步骤）；重建的
  `FlagTree/build/lib.linux-x86_64-cpython-310/triton/_C/libtriton.so`
  原子同步到 `flagOS-installed/flagTree/python/.../site-packages/triton/_C/` 与
  `flagOS-installed/pytorch/python/.../site-packages/triton/_C/`。
- FlagTree 方言回归（round15 执行，round16 未改源码、不重建）：安装内 `triton-opt` 与 `FileCheck`
  按各 mlir 的 RUN 行执行（`triton-opt <file> -split-input-file
  -pim-expand-phases [-pim-lower-to-emitc|-pim-verify-gml-contract] |
  FileCheck <file>`；negative 用 `-verify-diagnostics`）跑
  `paged_attn_emitc.mlir`、`paged_attn_phases.mlir`、`expand_phases.mlir`
  与四份 negative 全部 exit 0；`tests/test_opcompiler_ops.py -k paged_attn`
  **2 passed**、`tests/test_kernels.py -k "physical_slot or block_table or
  paged_attention"` **6 passed**。本轮没有改 FlagTree 源码，沿用 round14 的
  源码对象/安装对象逐字节一致结论，不重建安装前缀；round16 只改 B 路 pimir 形状，
  paged_attn 的 FlagTree lowering 与相位表不变。

机器可读证据块（由 `scripts/check_implement_evidence.py` 逐标签核验：current 条目必须与同名 summary 逐值一致，history/measurements 只允许正文引用，`facts` 按运行绑定或独立命名空间登记）：
<!-- evidence-block:start -->
```evidence
runs:
- label: dynamic
  stratum: current
  artifact: qwen3_moe
  layers: 4
  semantics: dynamic_dispatch
  prose_aliases: [dynamic_dispatch, dynamic]
  command: python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L
    --num-stages 2 --max-requests 1
  summary: test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/results_pimir/summary.json
  metrics:
    total_time_s: 60.784631503664286
    throughput_tokens_per_s: 13.078963881718604
    total_useful_flops: 5415241254987.654
- label: dense
  stratum: current
  artifact: qwen3_moe
  layers: 4
  semantics: dense_sim
  prose_aliases: [dense_sim, dense]
  command: python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L
    --num-stages 2 --max-requests 1 --moe-execution dense_sim --log-dir /tmp/r20-fix-dense
  summary: /tmp/r20-fix-dense/qwen3_moe_tp4_pp2_dense_sim_20261112-182507-3141801/results_pimir/summary.json
  metrics:
    total_time_s: 63.63723146811222
    throughput_tokens_per_s: 12.492686775639573
    total_useful_flops: 5826198177867.654
- label: alt
  stratum: history
  artifact: qwen3_moe
  layers: 4
  semantics: dynamic_dispatch
  prose_aliases: [tp1_pp4]
  command: round11 当轮源码的 tp1_pp4 主链（本轮不重跑）
  summary: /tmp/qwen3-review-round11-fix/pipeline-alt/qwen3_moe_tp1_pp4_dynamic_dispatch_20261111-191713-2087074/results_pimir/summary.json
  metrics:
    total_time_s: 138.5891210116591
    throughput_tokens_per_s: 5.736380995829528
- label: 48l
  stratum: history
  artifact: qwen3_moe
  layers: 48
  semantics: dynamic_dispatch
  prose_aliases: ["48 层", "48L"]
  command: round08 当轮源码主链 + round10 当轮 GeneSim E 重跑（本轮不重复启动）
  summary: /tmp/round10-fix/genesim-48l/results/summary.json
  metrics:
    total_time_s: 566.8630033645501
    throughput_tokens_per_s: 1.402455258645156
- label: round03
  stratum: history
  artifact: qwen3_moe
  layers: 4
  prose_aliases: [round03]
  metrics:
    total_time_s: 73.593
    total_useful_flops: 5430713177457.654
- label: round16_dynamic
  stratum: history
  artifact: qwen3_moe
  layers: 4
  semantics: dynamic_dispatch
  prose_aliases: [round16]
  metrics:
    total_time_s: 61.014502967924784
    throughput_tokens_per_s: 13.029689030129937
    total_useful_flops: 5415241254987.654
derived:
- id: round03_minus_dynamic_flops
  op: subtract
  left: round03.total_useful_flops
  right: dynamic.total_useful_flops
  value: 15500000000.0
- id: dense_over_dynamic_time
  op: ratio
  left: dense.total_time_s
  right: dynamic.total_time_s
  value: 1.046929625036486
- id: round16_minus_round17_time
  op: subtract
  left: round16_dynamic.total_time_s
  right: dynamic.total_time_s
  value: 0.229871464260498
- id: dense_flops_above_dynamic
  op: greater_than
  left: dense.total_useful_flops
  right: dynamic.total_useful_flops
  value: true
facts:
- id: dynamic_comm_bytes
  run: dynamic
  source: E 阶段依赖边累计
  value: 14692832968
- id: dense_comm_bytes
  run: dense
  source: E 阶段依赖边累计
  value: 15464703688
- id: alt_comm_bytes
  run: alt
  source: E 阶段依赖边累计
  value: 8218497128
- id: dynamic_parameter_bytes
  source: 4 层 A 阶段 IR（与执行语义无关）
  value: 6229628928
- id: 48l_parameter_bytes
  source: 48 层 A 阶段 IR
  value: 61064245248
- id: 48l_comm_bytes
  run: 48l
  source: round10 E 阶段 summary 与审计
  value: 166702607112
- id: 48l_capacity_dpu0_3
  run: 48l
  source: round08 BCD report_capacity
  value: 8258576576
- id: 48l_capacity_dpu4_7
  run: 48l
  source: round08 BCD report_capacity
  value: 7791833024
- id: 4l_cluster_resident_bytes
  source: 4L dynamic/dense E 阶段 pim_capacity_metrics.json（集群常驻）
  value: 3140834304
- id: 4l_cluster_limit_bytes
  source: pim_capacity_metrics.json（8 GiB 集群上限）
  value: 8589934592
- id: 4l_tensor_pu_max_resident_bytes
  source: 4L summary / pim_capacity_metrics.json（per-TensorPU 最大值）
  value: 779673600
- id: 4l_tensor_pu_limit_bytes
  source: pim_capacity_metrics.json（512 MiB per-TensorPU 上限）
  value: 536870912
- id: 4l_total_resident_ratio_dynamic
  run: dynamic
  source: 4L dynamic pim_capacity_metrics.json（单 TensorPU 常驻最大值 / 512 MiB 上限，诊断比、不参与 fits）
  value: 1.4522552490234375
- id: 4l_total_resident_ratio_dense
  run: dense
  source: 4L dense pim_capacity_metrics.json（单 TensorPU 常驻最大值 / 512 MiB 上限，诊断比、不参与 fits）
  value: 1.4522552490234375
- id: 4l_tensor_pu_request_bytes
  run: dynamic
  source: 4L dynamic pim_capacity_metrics.json（单 TensorPU 请求期最大值）
  value: 1626112
- id: 4l_tensor_pu_request_utilization
  run: dynamic
  source: 4L dynamic pim_capacity_metrics.json（request_bytes / 512 MiB，百分比）
  value: 0.302886962890625
- id: 4l_cluster_utilization
  run: dynamic
  source: 4L dynamic pim_capacity_metrics.json（集群常驻 / 8 GiB，百分比）
  value: 36.56412363052368
- id: 48l_trace_total
  run: 48l
  source: round10 E 阶段 trace
  value: 130806
- id: 48l_trace_pimir
  run: 48l
  source: round10 E 阶段 trace
  value: 106230
- id: dynamic_trace_total
  source: 4 层 E 阶段 trace（dynamic 与 dense 共用 10906 条）
  value: 10906
- id: dynamic_trace_pimir
  source: 4 层 E 阶段 trace（dynamic 与 dense 同为 8858 条 pimir）
  value: 8858
measurements:
- id: bench_prefill_speedup
  stratum: history
  source: scripts/bench_moe_dispatch.py 4L 真实权重
  value: 1.148
- id: bench_decode_speedup
  stratum: history
  source: scripts/bench_moe_dispatch.py 4L 真实权重
  value: 1.202
- id: old_dense_over_dynamic_1
  stratum: history
  source: round03 旧计费口径（已作废）
  value: 1.034
- id: old_dense_over_dynamic_2
  stratum: history
  source: round03 旧计费口径（已作废）
  value: 1.184
- id: round15_dynamic_time
  stratum: history
  source: round15 B 路单位错配口径（已作废）
  value: 88.48313625886303
- id: round15_dynamic_tps
  stratum: history
  source: round15 B 路单位错配口径（已作废）
  value: 8.984762900742771
- id: round15_dense_time
  stratum: history
  source: round15 B 路单位错配口径（已作废）
  value: 91.01234662163839
- id: round15_dense_tps
  stratum: history
  source: round15 B 路单位错配口径（已作废）
  value: 8.735078585601341
- id: old_dense_over_dynamic_round15
  stratum: history
  source: round15 B 路单位错配口径（已作废）
  value: 1.029
- id: dense_minus_dynamic_time_round16
  stratum: history
  source: round16 dense/dynamic summary 时间差
  value: 2.852
- id: round16_dense_time
  stratum: history
  source: round16 dense_sim 对照 summary
  value: 63.86710293237272
- id: round17_repetition_time_delta
  stratum: history
  source: round16 与 round17 dynamic summary 时间差（重复次数修正）
  value: 0.229871464260498
- id: round17_dense_minus_dynamic_time
  stratum: history
  source: round17 dense/dynamic summary 时间差
  value: 2.8525999644479327
- id: reviewer_old_ab
  stratum: history
  source: round02 评审旧实现 A/B
  value: 0.994
- id: round10_wrong_flops
  stratum: history
  source: round10 错误誊写（已更正）
  value: 5414125498987.654
- id: round10_wrong_flops_mantissa
  stratum: history
  source: round10 错误誊写的尾数写法
  value: 5.414125498987654
- id: round10_wrong_delta_1
  stratum: history
  source: round10 错误派生差值（已更正）
  value: 16600000000.0
- id: round10_wrong_delta_2
  stratum: history
  source: round10 错误派生差值（已更正）
  value: 16300000000.0
- id: round11_review_probe_delta
  stratum: history
  source: round11 评审探针的错误派生值
  value: 9.99e+99
- id: old_dense_time_round03
  stratum: history
  source: round03 旧产物（已作废）
  value: 76.1225576142813
- id: old_dynamic_time_round03
  stratum: history
  source: round03 旧产物（已作废）
  value: 39.378
- id: old_dense_time_round03b
  stratum: history
  source: round03 旧产物（已作废）
  value: 46.627
- id: 48l_ttft
  stratum: history
  source: round10 48L summary
  value: 518.314
- id: 48l_mean_tpot
  stratum: history
  source: round10 48L summary
  value: 0.714
- id: round14_pv_probe_single_delta
  stratum: history
  source: round14 评审探针（修复前，单块）
  value: 0.00048828125
- id: round14_pv_probe_multi_delta
  stratum: history
  source: round14 评审探针（修复前，多块）
  value: 0.0001220703125
- id: round19_allgemm_probe_time
  stratum: history
  source: round19 评审探针（全部 GEMM 指向 B 路产物，直接 GeneSim）
  value: 1.696
- id: round19_lmhead_probe_time
  stratum: history
  source: round19 评审探针（lm_head 自报 host，直接 GeneSim）
  value: 45.895
- id: op_metrics_vector_mul_round15
  stratum: history
  source: round15 4L op_type_metrics（B 路单位错配，已作废）
  value: 65.244
- id: op_metrics_vector_mul_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.151
- id: op_metrics_vector_mul_dense
  stratum: history
  source: round16 4L dense op_type_metrics
  value: 0.164
- id: op_metrics_split_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 15.317
- id: op_metrics_split_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.121
- id: op_metrics_transpose_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 15.198
- id: op_metrics_transpose_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.12
- id: op_metrics_rmsnorm_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 1.077
- id: op_metrics_rmsnorm_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.016
- id: op_metrics_rope_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 4.442
- id: op_metrics_rope_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.045
- id: op_metrics_concat_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 0.475
- id: op_metrics_concat_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.004
- id: op_metrics_vector_add_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 0.739
- id: op_metrics_vector_add_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 0.006
- id: op_metrics_paged_attn_round15
  stratum: history
  source: round15 4L op_type_metrics（已作废）
  value: 0.694
- id: op_metrics_paged_attn_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics
  value: 6.282
- id: op_metrics_gemm_dynamic
  stratum: history
  source: round16 4L dynamic op_type_metrics（与 round15 相同）
  value: 596.365
- id: op_metrics_vector_mul_48l
  stratum: history
  source: round10 48L op_type_metrics（B 路单位错配，已作废）
  value: 782.93
- id: op_metrics_transpose_round17
  stratum: history
  source: round17 4L dynamic op_type_metrics
  value: 0.0037380440159999768
- id: op_metrics_split_round17
  stratum: history
  source: round17 4L dynamic op_type_metrics
  value: 0.004760093519999858
- id: op_metrics_gather_round17
  stratum: history
  source: round17 4L dynamic op_type_metrics
  value: 0.00047375836199999824

```
<!-- evidence-block:end -->

### 4.2 分任务测试结果
| 任务ID | 测试用例名 | 测试用例数 | 通过数 | 失败数 | 测试说明 |
|--------|------------|------------|--------|--------|----------|
| T2/F3 | test_static_moe_forward_matches_hf、test_static_moe_forward_masks_unselected_expert_overflow、test_qwen3_moe_unselected_overflow_matches_hf | 3 | 3 | 0 | 含 inf/nan 溢出与 P0/P1 两态 |
| T1/F2 | test_qwen3_moe_config_passes、test_architecture_registry_drives_moe_pipeline 等 | 4 | 4 | 0 | 注册表驱动的主链描述 |
| T3/T4 | test_moe_gate_ops_stay_on_host、test_moe_weight_rules | 2 | 2 | 0 | 门控留主机、router/专家切分 |
| T5 | test_qwen3_moe_generate_matches_hf | 1 | 1 | 0 | 贪心 3 token 与 HF 一致 |
| T6b/F6 | test_qwen3_30b_generate_matches_hf | 1 | 1 | 0 | 4 层切片真实权重；参考为恢复后的 HF 稀疏前向 |
| F5 | test_moe_graph_tiling_policy、test_moe_compile_does_not_tile_experts、test_optimize_graph_graph_tiling_switch | 3 | 3 | 0 | MoE 默认关分块、显式开关生效 |
| F8 | test_tiled_expert_subgraph_is_recognized_and_gated、test_tiled_dispatch_tokens_match_hf、test_tiled_dispatch_skips_expert_communication、test_gate_skipped_host_command_returns_zero、test_dispatch_skips_expert_subgraph_commands_ratio 等 | 14 | 14 | 0 | 分块 + 通信门控 + 零值 + 子图命令跳过比例 |
| F1 | genesim TestQwen3MoeBuild（6 条）、TestExecutionDtypeAlignment（2 条）、TestSplitPreservesMoeIdentity（1 条）、TestParserQwen3Moe（1 条）、TestParserExecutionDtype（1 条） | 11 | 11 | 0 | 128 专家/top-8/768/54 GiB、执行 dtype 对齐、分片继承专家号 |
| F4 | TestPimirContractCompleteness（6 条）、TestMoeConfigLineage（3 条） | 9 | 9 | 0 | 缺项/缺哈希/不可读/哈希不符/豁免；sidecar MoE 元数据与 IR 一致 |
| F2/F4 | TestCompilerPlacementCapacityScope（test_compiler_placement.py 2 条 + test_pim_capacity.py 2 条） | 4 | 4 | 0 | 编译器放置的 ClusterPU 容量口径与 KV 仍按 TensorPU |
| F1 | TestExpertAwareTensorParallelExpansion（1 条） | 1 | 1 | 0 | 专家分组与 router 展开；TestMoeConfigLineage 见 F4 行 |
| F12 | TestMoeExecutionContract（2 条）、test_explicit_dense_sim_is_recorded、test_unknown_execution_mode_is_rejected、test_dense_model_rejects_moe_execution、test_cli_honours_explicit_execution_mode | 6 | 6 | 0 | IR 缺省/显式/非法执行语义与 parser CLI |
| F12/F16/F19 | TestDynamicDispatchCost（5 条）+ TestMoeExecutionContract 序列化与角色校验（3 条）+ TestDynamicDispatchPimLatency（3 条）+ `test_dynamic_dispatch_shortens_expert_compute_on_the_critical_path` + `test_split_operator_keeps_expert_id_and_role` / `test_every_split_dimension_keeps_the_moe_cost_role` | 14 | 14 | 0 | any-hit `1-(1-k/E)^Tq`（Tq=1/16/726）、通信命中并集、共享累加不缩放、dense_sim 全量、PIM trace 时延折算、管线级执行语义生效 |
| F13 | TestPimirContractCompleteness 新增 3 条、TestPimirConsumptionIsNotSilent 2 条、test_sidecar_without_moe_config_is_rejected、test_mismatched_execution_mode_is_rejected | 7 | 7 | 0 | 显式严格开关、缺 sidecar、豁免落 PIM、缺 MoE 元数据 |
| F13 | test_unknown_device_op_is_not_silently_exempted、test_legitimate_exemptions_are_registered、test_export_step_rejects_a_fresh_sidecar_without_hashes、test_pipeline_resolves_one_moe_execution_for_all_stages | 4 | 4 | 0 | 豁免白名单、fresh 缺哈希、主链语义解析 |
| F14 | test_bpath_eltwise_keeps_operator_kind、test_bpath_rejects_unknown_eltwise_operator、TestPimirOperatorKindIsVerified（2 条） | 4 | 4 | 0 | 导出侧 add/mul、未知拒绝、仿真侧 kind 核对 |
| F15 | test_eltwise_and_activation_costs_follow_the_operator_shape | 1 | 1 | 0 | 专家 768 的乘法/激活与 hidden 残差加 |
| F16 | test_skipped_launch_is_never_submitted、test_gate_skipped_launch_without_skip_zero_leaves_the_buffer、test_benchmark_harness_flips_modes_on_one_backend | 3 | 3 | 0 | 提交前跳过、skip_zero 取舍、同一后端 A/B |
| F18 | TestFullGraphCoverage 4 条、`test_compiler_graph_and_genesim_ir_cover_the_same_weights` 3 条、`test_model_level_lm_head_is_exported_with_its_own_weight`、`test_lm_head_with_expert_id_is_rejected`、`test_lm_head_is_expanded_across_tensor_parallel_shards`、`test_rmsnorm_cost_follows_the_operator_shape` | 11 | 11 | 0 | lm_head/q_norm/k_norm 覆盖、跨架构权重身份/形状对拍、模型级输出头导出与 TP 展开、QK 归一化按 head_dim 计费 |
| F20 | test_sidecar_carries_the_single_capacity_contract、TestHwBudgetContract 8 条 | 9 | 9 | 0 | 8 GiB 契约、sidecar 回传三件套、4/16 GiB 偏差被拒绝、缺字段/0 预留/1 字节对齐被拒绝、旧非严格 sidecar 兼容、跨仓常量相等 |
| F21 | tests/test_ir_fingerprint.py 6 条、TestModelIdentityFingerprint（model_ir 4 条）、TestModelIdentityFingerprint（compiler_placement 4 条） | 14 | 14 | 0 | 跨仓指纹一致、词表/hidden 变化、改内容不改指纹、A2 缓存键、sidecar 指纹拒绝与旧产物规则 |
| F23 | `test_current_token_costs_ignore_kv_history_and_attention_tracks_it`、`test_eltwise_and_activation_costs_follow_the_operator_shape`、`test_rmsnorm_cost_follows_the_operator_shape` | 3 | 3 | 0 | Tp=0/128/726 三点不变式；VECTOR_MUL/ADD、SILU、RMSNORM、ROPE 用 Tq；MASK/SOFTMAX/GEMV/PAGED_ATTN 用 Tq(Tp+Tq) |
| F24 | genesim `test_missing_op_placed_on_gpu_is_still_rejected`、`test_missing_op_placed_on_cpu_is_still_rejected`、`test_all_ops_on_gpu_are_still_checked`、`test_unknown_device_op_cannot_be_exempted`、`test_legitimate_host_and_boundary_exemptions_pass_the_gate`；compiler `test_audit_sidecar_coverage_*` 4 条、`test_trace_provenance_*` 3 条、`test_skip_simulation_runs_the_coverage_audit` | 13 | 13 | 0 | 落点无关覆盖、未知豁免拒绝、合法 host/边界、MoE 三件套、trace 来源/应覆盖集合、skip-simulation |
| F25 | `test_fingerprint_tracks_dependency_topology_and_layer_ownership`、`test_fingerprint_ignores_refinement_and_placement_fields`、`test_fingerprint_tracks_topology_and_layer_ownership` | 3 | 3 | 0 | 换边/换张量身份/跨层移动改变指纹；成本/设备/本地分片字段不改变 |
| F26 | `test_legacy_dependency_payload_is_scaled_by_target_role` | 1 | 1 | 0 | 旧式与显式系数等价；expert_work/comm/shared_reduce 与 dense_sim |
| F27 | `test_every_layer_gates_its_own_experts`、`test_pp_stages_each_gate_their_own_layers`、`test_backend_records_per_dispatch_execution`、`test_tiled_dispatch_tokens_match_hf`（增逐层断言）、`test_ep_16_dpu_generate_matches_hf`（增逐层断言） | 5 | 5 | 0 | decode/prefill/强制分块/PP 跨段/EP 五条路径逐层断言 gate 层号与归属；后端按 dispatch 记录放行/拦下 |
| F28 | `test_moe_tail_is_host_shared_work`、`test_plan_and_genesim_ir_agree_on_moe_roles`、genesim `test_host_tail_covers_select_and_weighted_accumulation` | 3 | 3 | 0 | 选择/加权累加归 host；IR expert_work 集合 == 计划带 gate 的 (层,专家) 集合；首专家 2×hidden、其余 3×hidden |
| F29 | compiler `test_audit_accepts_a_manifest_backed_fused_activation`、`test_audit_rejects_absorbed_ops_without_a_fusion_manifest`、`test_audit_rejects_absorbing_a_non_activation_op`、`test_audit_rejects_absorption_into_a_role_that_was_not_fused`、`test_audit_rejects_an_absorbed_id_outside_the_source_ir`；genesim `test_manifest_backed_absorption_passes`、`test_absorbed_without_a_manifest_is_rejected`、`test_absorbing_a_non_activation_is_rejected`、`test_absorbed_id_outside_the_source_ir_is_rejected`、`test_sidecar_expert_id_must_match_the_source_ir`、`test_expanded_shard_ids_do_not_poison_the_source_check` | 11 | 11 | 0 | absorbed 只能由融合清单 + 主算子 pimir + IR 依赖边证明；直接 GeneSim 按 (层,角色,专家) 核对；TP 展开编号不参与源校验 |
| F30 | compiler `test_audit_accepts_the_contract_hw_budget`、`test_audit_rejects_a_strict_sidecar_with_a_missing_budget_field`（3 参数）、`test_audit_rejects_a_strict_sidecar_without_hw_budget`、`test_audit_rejects_zero_reserve_and_unit_align`、`test_audit_accepts_a_legacy_sidecar_without_a_budget`、`test_hw_budget_payload_rejects_weakened_contract`；genesim `TestHwBudgetContract` 8 条 | 16 | 16 | 0 | 三件套缺失/偏值/0 预留/1 字节对齐全部拒绝；旧非严格产物兼容；导出写出前校验 |
| F32 | compiler `test_static_moe_forward_zero_weight_selected_expert_matches_hf`（2 态）、`test_static_moe_forward_masks_unselected_overflow_both_norms`（2 态）、`test_decode_and_prefill_dispatch_read_the_selected_mask` | 5 | 5 | 0 | 选中权重下溢仍执行、未选中溢出掩零、两态归一化、decode/prefill dispatch 均读布尔掩码 |
| F33 | compiler 10 条豁免/落点入口用例（含 router/q_proj/lm_head 三种设备 GEMM）；genesim `test_exempted_router_gemm_without_host_evidence_is_rejected`、`test_exempted_router_with_pim_placement_entry_is_rejected`、`test_legitimate_host_and_boundary_exemptions_pass_the_gate` | 13 | 13 | 0 | router/伪造 host 证据/无导出证据的 host GEMM 全部拒绝；合法 host 与边界放行 |
| F34 | compiler `test_default_hardware_and_budget_share_one_alignment`、`test_vllm_compile_capacity_uses_the_site_factory`、`test_compile_rejects_misaligned_budget`、`test_with_budget_rejects_conflicting_module_attributes`、`test_pimir_hardware_contract_reads_the_real_module_attributes`；genesim `test_pimir_hardware_attributes_must_match_the_contract` | 6 | 6 | 0 | 默认/pim_options/vLLM/两路派生一致；冲突属性与错误 MLIR 属性拒绝；4L 产物 64 逐份核验 |
| F35 | compiler `test_audit_sidecar_coverage_rejects_swapped_expert_identity`、`..._rejects_a_cross_layer_expert_swap`、`..._accepts_matching_expert_weights`、`test_audit_sidecar_coverage_requires_every_moe_expert_role`、`test_trace_provenance_rejects_swapped_expert_identity`、`test_sidecar_freshness_rejects_swapped_expert_identity`、`test_audit_rejects_exempting_any_device_dense_gemm`（3 参数）；genesim `test_expert_identity_checks_the_weight_path_per_op`、`test_sidecar_expert_id_must_match_the_source_ir` | 11 | 11 | 0 | 同层/跨层互换、权重路径不符、缺组合全部拒绝；trace 验收也逐 op_id 核对；合法 128 专家通过 |
| F36 | genesim `test_batch_edit_builds_the_index_once`、`test_batch_edit_does_not_scan_the_operator_list_per_addition`、`test_batch_edit_matches_sequential_result`、`test_expansion_rebuilds_the_index_once`、`test_batch_expansion_matches_sequential_expansion` | 5 | 5 | 0 | 批量与逐条结果等价；索引只建一次；扫描量不随 add 次数线性放大 |
| F37 | compiler `test_single_expert_moe_keeps_moe_execution_semantics`、`test_single_expert_moe_generate_matches_hf`；genesim `test_single_expert_is_still_moe`、`test_cli_accepts_a_single_expert_moe`、`test_single_expert_moe_sidecar_config_is_accepted` | 5 | 5 | 0 | 1 专家/top-1 的 HF 对拍、生成、parser、IR、sidecar 全通过；dense 行为不变 |
| F39 | `test_placement_check_*` 4 条、`test_prepare_simulation_config_binds_the_run_placement`、`test_run_directory_is_derived_from_artifact_and_strategy`、`test_simulation_step_consumes_the_current_placement`、`test_simulation_step_rejects_a_sidecar_from_another_strategy` | 8 | 8 | 0 | 正例放行、错策略/错本地形状/错层段全部拒绝；E 绑定本次 sidecar |
| F40 | 48L 主链实跑：A→0→BCD→A2→E；4L `tp4_pp2`/`tp1_pp4` 对照 | 3 | 3 | 0 | 48L 重新生成产物并形成 summary/trace；两种 4L 策略各自绑定 |
| F41 | `test_arch_strategy_constraints_cover_all_entry_points`、单专家生成/parser/sidecar 用例、真实主链非法分支 | 3 | 3 | 0 | 合法 TP=1 放行、非法 TP=2 在加载后报错；dense 行为不变 |
| F42 | `test_full_contract_*` 9 条、`test_freshness_*` 5 条、`test_sim_ir_check_uses_the_full_contract`、`test_trace_provenance_uses_the_full_contract`、`test_export_step_uses_the_full_contract` | 17 | 17 | 0 | 五个入口缺 MoE/容量/伪造 host/非法吸收/专家互换/指纹全部拒绝；合法产物放行 |
| F43 | `test_kv_estimate_reuses_the_real_layout`、`test_preflight_accepts_a_layout_that_fits_under_the_contract_alignment`、`test_preflight_rejects_one_byte_over_the_budget`、`test_serve_pim_has_no_independent_alignment_or_old_budget` | 4 | 4 | 0 | 与真实布局逐字节一致；契约对齐放行、超 1 字节拒绝 |
| F44 | compiler `test_audit_rejects_absorbing_an_activation_with_another_consumer`、`test_audit_accepts_a_manifest_without_absorbed_ids_during_export`、`test_audit_rejects_a_refined_sidecar_without_absorbed_ids`、`test_post_refine_entries_require_absorbed_closure`（3）、`test_skip_simulation_requires_the_absorbed_ids`、`test_export_step_validates_the_bcd_phase`；genesim `test_absorbing_an_activation_with_another_consumer_is_rejected`、`test_manifest_without_absorbed_ids_is_rejected` | 10 | 10 | 0 | 多消费者拒绝；BCD 清单先行放行；A2 后 freshness/sim_ir/trace/skip 与直接 GeneSim 要求双向闭合 |
| F45 | compiler `test_audit_rejects_an_exempt_op_with_a_pim_placement_entry`、`test_audit_rejects_a_boundary_with_a_pim_placement_entry`、`test_audit_rejects_a_host_gemm_exemption_without_evidence`、`test_audit_rejects_a_host_gemm_exemption_with_a_mismatched_role`、`test_audit_rejects_a_host_gemm_exemption_with_a_forged_weight`、`test_post_refine_entries_reject_exemption_placement_contradictions`（3）、`test_skip_simulation_rejects_exemption_placement_contradictions`；genesim `TestHostGemmExemptionEvidence` 6 条 | 15 | 15 | 0 | host 非 GEMM/边界/host GEMM 三类矛盾与证据不符全部拒绝；合法 host/边界/真实产物放行 |
| F46 | 5 个 GeneSim 测试文件的后置测试类 | 12 | 12 | 0 | 守卫移到末尾后，此前未执行的 12 条用例全部通过 |
| F47 | `TestMoeDependencyDirection` 5 条、`test_dependency_payload_follows_the_expert_endpoint`、`test_expert_host_egress_is_scaled_by_the_hit_union`、`test_expanded_expert_output_edges_keep_the_source_role`、`test_gate_skipped_host_gather_does_not_transfer` | 9 | 9 | 0 | 四方向 decode/prefill、TP 展开后专家分片→共享累加、专家→HOST、未命中 host 收集不搬运 |
| F48/F51 | `tests/test_implement_evidence.py` 13 条 | 13 | 13 | 0 | round11 起改为证据块：标签互换、4 层冒充 48 层、历史冒充 current、无 summary 的 current、错误差值/比值/结论、科学计数法正文、合法分栏 |
| F49 | genesim `test_strict_product_rejects_a_raised_runtime_capacity`、`test_strict_product_rejects_a_config_raised_on_one_side`、`test_pimir_attributes_are_checked_against_the_requirement_constant`、`test_legacy_product_keeps_other_capacity_configurations`、`test_contract_matches_the_compiler_values`；compiler 跨仓常量用例 | 6 | 6 | 0 | 配置与 sidecar 同步 4/16 GiB、单侧偏离、MLIR 16 GiB 全部拒绝；旧式非严格产物兼容；两仓常量逐值相等 |
| F50 | `test_simulation_step_never_deletes_shared_outputs`、`test_interleaved_runs_keep_their_own_traces_and_summaries`、`test_trace_provenance_reads_the_configured_trace_dir`、扩展的 `test_prepare_simulation_config_binds_the_run_placement`/`test_run_directory_is_derived_from_artifact_and_strategy`/`test_simulation_step_consumes_the_current_placement` | 6 | 6 | 0 | 共享 trace/summary 哨兵保留；两次语义交错执行各自 summary/trace 独立；临时配置回读 IR/trace/results/log |
| F52 | `test_strict_config_requires_the_capacity_contract`（3）、`test_strict_config_rejects_a_sidecar_with_a_wrong_capacity`（4/16 GiB × 3）、`test_audit_honours_an_explicit_strict_context`、`test_audit_keeps_self_reported_strictness_when_the_caller_is_not_strict`、`test_non_strict_config_keeps_the_legacy_compatibility_branch`、`test_export_step_requires_the_capacity_contract`、`test_skip_simulation_requires_the_capacity_contract` | 14 | 14 | 0 | 配置显式严格时删自报/缺容量/4/16 GiB 在三个复核入口、导出与 skip 全部拒绝；非严格配置保留旧式兼容；自报严格不被显式非严格关掉 |
| F53 | `test_prose_only_swap_of_dynamic_and_dense_is_rejected`、`test_prose_cannot_borrow_a_forty_eight_layer_value`、`test_prose_history_value_cannot_pose_as_current`、`test_prose_comparison_conclusion_is_checked`、`test_required_values_must_appear_in_the_prose`、`test_measurements_are_allowed_in_a_run_context` | 6 | 6 | 0 | 只换正文、48 层借值、历史冒充、错误比较、正文缺失必需值全部失败；合法分栏与独立 measurements 放行 |
| F54 | `test_paged_attn_matches_the_online_reference`、`test_paged_attn_multi_block_tail_matches_the_online_reference`、`paged_attn_emitc.mlir`、`paged_attn_phases.mlir` | 4 | 4 | 0 | PV 乘积落 f16 的逐元素对拍（单块/多块尾块）、数据能区分取整口径、EmitC 取整点与槽号兜底、相位表不变 |
| F55 | `test_physical_slot_accepts_both_legal_bounds`、`test_physical_slot_rejects_negative_and_upper_invalid`（2）、`test_block_table_tail_sentinel_is_not_read`、`test_paged_attention_rejects_invalid_slots_before_addressing`（2）、`test_sdpa_fused_paged_attn_tracks_the_host_path` | 7 | 7 | 0 | 槽 0/最后合法槽放行；负值/上界拒绝；表尾哨兵不读；启动前预检先于寻址；融合/主机对拍 |
| F56 | `test_strict_entry_requires_config_and_artifacts`（12）、`test_strict_entry_rejects_a_missing_config_file`（3）、`test_sidecar_self_report_requires_the_ir_key`（3）、`test_non_strict_entry_keeps_the_no_sidecar_legacy_path` | 19 | 19 | 0 | 六种缺输入 × 三入口全部失败；自报严格缺 IR 键失败；非严格旧式用法放行 |
| F57 | `test_trace_entry_accepts_hash_bound_shards`、`test_trace_entry_rejects_unbound_pimir_traces`（5）、`test_trace_entry_accepts_a_proven_empty_absorbed_trace`、`test_trace_entry_rejects_a_non_empty_absorbed_trace`、`test_trace_entry_rejects_an_absorbed_id_outside_the_sidecar` | 9 | 9 | 0 | 分片回指正例；错哈希/缺哈希/错类型/错父编号/错 expert_id；空 absorbed 正例与非空/无编号反例 |
| F58 | `test_same_block_former_latter_swap_is_rejected`、`test_former_latter_identity_must_match_the_declared_runs`、`test_required_values_in_an_unlabeled_preamble_do_not_count`、`test_fact_values_are_bound_to_their_run`、`test_facts_with_an_unknown_run_are_rejected`、`test_facts_with_a_non_numeric_value_are_rejected` | 6 | 6 | 0 | 同块对调失败、身份正确正例、无标签前言不算引用、facts 运行绑定/未知运行标签/非数值失败 |
| F59 | `python scripts/check_implement_evidence.py`（当前文档 + dynamic/dense 两份 current summary） | 1 | 1 | 0 | 退出码 0；alt/48l 是 history 正文引用，不由本轮 summary 核验；§8.3 与 GeneSim 注释按实际规则更新 |
| F60 | `test_strict_entry_requires_a_valid_pimir_artifact`（3 入口 × 5 类）、`test_sidecar_self_report_requires_a_valid_pimir_artifact`（3） | 18 | 18 | 0 | 缺路径/不存在/不可读/缺哈希/错哈希 × freshness/sim_ir/trace 全部拒绝；自报严格路径同样拒绝；真实 4L 破坏探针 REJECTED、合法产物 ACCEPTED |
| F61 | `test_trace_entry_requires_a_configuration`、模板/unknown 用例改传配置、`test_trace_metadata_survives_a_header_byte_of_0x7b` | 4 | 4 | 0 | 无配置不能签发；模板/unknown 仍拒绝；头部字节解析直接对拍 |
| F62 | `test_prepare_simulation_config_generates_a_missing_trace`、`test_prepare_simulation_config_keeps_an_existing_trace` | 2 | 2 | 0 | 缺失时生成 `GSTR` trace 并绑定绝对路径；已存在则原样使用 |
| F63 | `python scripts/check_implement_evidence.py`（当前文档 + dynamic/dense 两份 current summary） | 1 | 1 | 0 | 退出码 0；alt/48l 是 history 正文引用，不由本轮 summary 核验；§8.4 按来源/时间/层数/是否重跑分栏 |
| F64 | `test_where_kernel_matches_torch_with_a_broadcast_operand`、`test_scalar_where_is_host_only_and_not_a_device_kernel` | 2 | 2 | 0 | `where.self` 广播与 torch 一致；`ScalarOther`/`ne.Scalar` 在 HOST_ONLY 且未注册 |
| F65 | compiler `test_bpath_uses_one_token_and_the_mlp_local_width`、`test_bpath_view_ops_keep_the_global_width_next_to_tp_gemms`、`test_trace_lineage_accepts_a_matching_bpath_unit`、`test_trace_lineage_rejects_a_wide_token_bpath_trace`、`test_trace_lineage_rejects_a_global_width_bpath_trace`、`test_trace_lineage_rejects_bpath_bytes_that_ignore_the_token_unit`、`test_trace_lineage_requires_the_unit_fields_in_strict_mode`、`test_trace_lineage_checks_split_as_a_pure_view`；genesim `TestPimirUnitContract` 6 条、`test_bpath_shards_inherit_pimir_only_for_matching_local_shapes`、`test_bpath_shard_shape_conflict_is_rejected_in_strict_mode`、`test_vector_trace_token_unit_scales_with_the_query_rows`、`test_legacy_wide_token_trace_is_not_multiplied_again`、`test_block_loop_repetitions_scale_by_the_trace_token_unit`、`test_partial_prefill_is_rounded_up_to_whole_trace_units`、`test_plain_phase_chain_reports_the_trace_token_unit`、`test_explicit_trace_tokens_attribute_wins_over_shape_heuristic`、`test_four_tp_shards_cost_the_same_bytes_as_the_global_width` | 23 | 23 | 0 | B 路单位按「1 token × 本地宽」编译/验收；TP 分片只在同形时继承；运行期按 `ceil(Tq/tokens)` 重复；TRANSPOSE 维序与 SPLIT 代表形状两条同类路径一并覆盖 |
| F66 | `test_moe_identity_rejects_a_top_k_that_disagrees_with_the_config`、`test_moe_identity_rejects_a_wrong_expert_width`、`test_dense_config_rejects_moe_scalars_and_accepts_an_empty_identity` | 3 | 3 | 0 | top-k 不进入权重字节；A 段与 BCD 段都从 HF config 派生身份逐项核对 |
| F67 | `test_bool_dtype_comment_describes_the_selected_mask` | 1 | 1 | 0 | bool 注释描述 `selected_mask`，不再出现 `routing != 0` |
| F68 | `test_view_embed_and_eltwise_ops_repeat_by_the_bound_tokens`、`test_paged_attn_prefill_and_decode_repeat_by_the_bound_tokens`、`test_unbound_query_tokens_are_rejected_instead_of_guessed` | 3 | 3 | 0 | TRANSPOSE/SPLIT/GATHER/VECTOR_MUL 重复实际 Tq、总字节=tq×单位字节；PAGED_ATTN prefill/decode；缺绑定字段直接拒绝 |
| F69 | `test_bpath_kv_ops_keep_symbolic_history_dims`、`test_four_node_attention_accepts_prefill_and_decode_history`、`test_four_node_gemv_context_rejects_a_wrong_feature_width`、`test_a_concrete_history_placeholder_loses_the_dimension_role`、`test_trace_lineage_accepts_four_node_attention_units`（4 参数）、`test_trace_lineage_rejects_a_history_dim_folded_into_a_placeholder`、`test_trace_lineage_rejects_a_wrong_feature_width_in_four_node_units` | 10 | 10 | 0 | 单位保留符号维；四节点四类算子 prefill/decode 接受；特征宽错误与固化历史维拒绝 |
| F70 | `test_strict_entries_require_a_complete_bpath_unit`（3 入口 × 2 形态）、`test_strict_entries_reject_units_that_drop_symbolic_dims`（2 入口）、`test_skip_simulation_requires_a_complete_bpath_unit` | 9 | 9 | 0 | freshness/仿真 IR/trace/skip-simulation 对缺单位、半缺、丢符号维一致拒绝 |
| F71 | `test_checked_in_qwen3_standalone_products_match_the_compiler_default` | 1 | 1 | 0 | 共享 IR/sidecar 执行语义=编译器默认、指纹一致、严格契约通过 |
| F72 | `test_bpath_entry_with_a_fake_shard_is_rejected`（3 入口）、`test_device_gemm_without_shards_is_rejected`（3 入口）、`test_skip_simulation_rejects_a_fake_shard_on_a_bpath_entry`、`test_bpath_entry_with_a_fake_shard_fails_classification`、`test_device_gemm_without_shards_fails_classification`、genesim `test_bpath_entry_with_a_fake_shard_is_rejected`、`test_strict_sidecar_rejects_a_device_gemm_without_shards` | 11 | 11 | 0 | 假分片与空分片在导出、skip、freshness、仿真 IR、trace 与直接 GeneSim 全部拒绝；4L 真实分类 1557/1606 不变 |
| F73 | `test_unit_tokens_must_match_the_pimir`（2）、`test_unit_feature_width_must_match_the_pimir`（2）、`test_unit_must_match_the_pimir_actually_referenced`（2）、`test_pimir_without_trace_tokens_is_rejected`（2）、genesim `test_sidecar_unit_must_match_the_referenced_pimir`、`test_pimir_without_trace_tokens_is_rejected_for_a_unit` | 10 | 10 | 0 | 单位数值与 pimir 的 `traceTokens`/特征末维逐值绑定；缺 traceTokens 的旧 pimir 在严格产物中拒绝；4L 真实产物 ACCEPTED |
| F74 | `test_cache_key_lock_waits_for_a_slow_schedule`、`test_cache_key_lock_serializes_processes` | 2 | 2 | 0 | 旧 5 秒写法在慢调度用例上复现 `_queue.Empty`；修复后 60 秒死锁检测 + 子进程回收，无残留进程 |
| F75 | `test_documented_checker_command_only_uses_current_summaries` + 原有 25 条证据用例 | 26 | 26 | 0 | 文档不再出现「四份 summary」；CI 按证据块 current 标签执行文档命令，退出码 0 |
| F76 | checker 对 current dynamic/dense 两份 summary | 1 | 1 | 0 | 4L 容量按集群口径引用；per-TensorPU 最大值单列；证据块补 4 个容量 facts |
| F77 | `test_real_weight_tests_derive_capacity_from_the_single_factory`（2 参数）、4 层真实目标命令 | 3 | 3 | 0 | 8B/30B 真实测试均由 `default_site_capacity()` 派生；手写字面量与手写构造被静态守卫拒绝 |
| F78 | `test_audit_rejects_gemms_pointing_at_a_bpath_pimir`、`test_audit_rejects_a_gemm_pimir_from_another_shard_width`、`test_audit_rejects_a_device_gemm_without_kernel_tile_n`、`test_audit_rejects_a_fused_gemm_without_act_spec`、`test_strict_entries_reject_an_invalid_apath_pimir`（3 入口 × 2 形态）、`test_skip_simulation_rejects_an_invalid_apath_pimir`（2）；genesim `test_gemm_pointing_at_a_bpath_pimir_is_rejected`、`test_gemm_pimir_from_another_shard_width_is_rejected`、`test_device_gemm_without_kernel_tile_n_is_rejected`、`test_validate_pimir_unit_checks_gemm_semantics`、`test_complete_sidecar_passes_the_gemm_rule` | 17 | 17 | 0 | A 路产物必须是分片实测分块 GEMM；B 路产物、别的分片宽度、缺分块、融合缺 `act_spec` 在共享入口与直接 GeneSim 同判；4L/共享 dense/llama2 真实产物 ACCEPTED |
| F79 | `test_audit_rejects_a_host_gemm_claim_without_source_ir_evidence`、`test_strict_entries_reject_a_host_gemm_claim`（3 入口）、`test_skip_simulation_rejects_a_host_gemm_claim`；genesim `test_host_claim_without_source_ir_evidence_is_rejected`；原有 host 豁免正例 | 6 | 6 | 0 | `operators` 自称 host 缺源 IR/豁免/`host_gemms` 证据全部拒绝；真实产物 0 条 host 条目不被误判；合法 host 豁免仍放行 |
| F80 | `test_audit_rejects_a_bpath_op_type_that_disagrees_with_the_source_ir`（VECTOR_ADD/GELU）、`test_audit_rejects_a_device_gemm_without_semantic_role`、`test_audit_rejects_a_bpath_eltwise_kind_that_disagrees_with_the_ir`、`test_trace_lineage_checks_bytes_by_the_source_ir_op_type`；`test_sidecar_checks_split_by_entry_class` 的源 IR 补投影身份 | 5 | 5 | 0 | `op_type`/`semantic_role` 与源 IR 逐条一致；逐元素 kind 与 IR 类型对应；trace 字节按 IR 类型选判据 |
| F81 | `test_sdpa_host_fallback_rounds_at_the_phase_boundaries`；`tests/test_kernels.py` 共 31 条 | 31 | 31 | 0 | 双编译内核缺席时主机路径与相位表参考逐元素一致；数据能区分 PV 边界；既有融合/主机对拍与槽号用例不回归 |
| F82 | `test_judged_utilization_matches_the_fits_basis`；`test_main` 的 summary 渲染断言同步新口径 | 1 | 1 | 0 | 判定利用率与 `fits` 同源；请求期/集群/总量比四口径；summary 渲染两行 |
| F83 | compiler `test_audit_rejects_a_bpath_pimir_with_another_operators_semantics`（7）、`test_audit_rejects_a_bpath_operator_type_without_a_calibrated_fingerprint`、`test_bpath_semantic_table_accepts_each_calibrated_product`（18）、`test_bpath_semantic_table_is_pinned_across_repos`、`test_genesim_pimir_facts_agree_on_each_calibrated_product`、`test_strict_entries_reject_a_bpath_semantic_swap`（3 入口 × 2）、`test_skip_simulation_rejects_a_bpath_semantic_swap`；genesim `test_calibrated_forms_pass_their_own_fingerprint`、`test_pimir_with_another_operators_semantics_is_rejected`、`test_bpath_pimir_with_another_operators_semantics_is_rejected` | 38 | 38 | 0 | 五类互换 + 两次注意力矩阵乘互换在共享验证器、三后置入口、skip-simulation 与直接 GeneSim 同判；18 类实产正例、未登记类型拒绝、两仓表逐值相等且解析逐键一致；4L/48L/llama2 真实产物 0 问题 |
| F84 | `test_capacity_scope_is_explicit_in_summary`；文档四处措辞与设计 §十.5 | 1 | 1 | 0 | summary 显式输出单 TensorPU/集群常驻/诊断比三个 scope 字段；文档不再把单 TensorPU 最大值写成跨 TensorPU 总量 |
| F85 | `test_cache_key_lock_serializes_processes`、`test_cache_key_lock_waits_for_a_slow_schedule` | 2 | 2 | 0 | 两个锁用例改用 spawn + 模块级 worker；单独连跑 3 次均通过，全量回归不再出现 `_queue.Empty` |

### 4.3 回归测试结果
- 执行命令：`python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider --basetemp=/data/fjg/pengcheng/flagOS/.pytest-tmp-r21`
  （用例集合与用户要求的默认命令一致；临时目录放工作区所在磁盘，`/tmp` 根文件系统当时仅剩约 2 GB）
- 执行结果：**2207 passed、11 skipped、52 deselected、2 warnings in 1373.42s**，失败 0，日志
  `/tmp/r20-fix-regression.log`。与 round19 的 2172 passed 相比，通过数增加
  35，全部来自 round20 的 compiler 新增用例（B 路语义互换/实产正例/
  未登记类型/跨仓表与解析一致/三入口与 skip-simulation）；跳过、deselected
  与失败数不变。另 4 条新增用例在 GeneSim 套件（语义 3 条 + 容量 scope 1 条）。
- GeneSim：`cd genesim && ./run.sh --test sim` **45/45 个测试文件通过**
  （exit 0）；日志 `/tmp/r20-fix-genesim-sim2.log`；其中
  `test_compiler_placement.py` 149 条全过（round19 为 146 条）。
- 4 层真实权重目标命令：**36 passed、1 skipped、2 warnings in 242.71s
  （0:04:02）**，日志 `/tmp/r20-fix-4l-tests.log`，见 4.1/8.4。
- 全链路回归：Qwen3 4 层入口 `tp4_pp2` dynamic **60.785 s / 13.079 t/s**、
  dense **63.637 s / 12.493 t/s**（本轮两语义均重跑，`PIPELINE_EXIT=0`）；
  独立入口同源配置的 dynamic **60.785 s / 13.079 t/s**（round18 历史，
  本轮未重跑）；编译器
  五入口与直接 GeneSim 对 4L 新产物 ACCEPTED，分类/单位/A 路/host/字段/
  B 路语义互换探针全部 REJECTED；48 层主链 A/BCD/A2 沿用 round08 产物、E
  沿用 round10 的 summary；round16/17 的单位契约使旧 48 层 B 路条目在
  freshness、仿真 IR 与 trace 三入口都 REJECTED，48 层成本证据失效
  （K16/Q12），本轮未重跑；48L/llama2 的 B 路语义指纹复核 0 问题。
  本轮第一次全量回归在 `test_cache_key_lock_serializes_processes` 上以
  `_queue.Empty` 超时（单跑 2/2 通过）；按 I96/F85 改用 spawn 后重跑，
  该用例通过，结果见上。
  Llama2-7B 慢速链路回归不在快速回归范围内（`-k "not llama2_7b"` 与 slow
  标记排除）。

## 五、问题与偏差说明

### 5.1 已知问题
| 编号 | 问题描述 | 影响范围 | 状态 | 关闭说明 |
|------|----------|----------|------|----------|
| K1 | 30B 真机用例未执行（模型下载未完成） | P0-b 验收 | 已关闭 | 4 层切片与全模型用例见 4.3 |
| K2 | ~4.7 万节点图的编译耗时未实测 | 30B 用例体验 | 已关闭 | 全模型编译耗时为分钟到小时级，见 4.3 |
| K3 | ≥8 DPU 跨段竞态（dpu→host 读区 WAR 缺失） | PP×EP 验收 | 已关闭 | 读区按 (redist, 读 DPU) 登记；16 DPU 用例 6/6 |
| K4 | 本机无 Qwen3-8B 权重 | 8B 真实用例 | 已规避 | 如实记为 SKIP，不作为通过 |
| K5 | 48 层全模型编译耗时约数小时 | 真实验收节奏 | 已规避 | 迭代用 4 层切片；全模型用例后台执行 |
| K6 | 30B 单 DPU 容量口径曾三处不一致：主链导出 4 GiB、真实用例 16 GiB、GeneSim 8 GiB | 全模型主链容量验收、所有 PP/TP 策略 | 已关闭 | 用户确认 8 DPU × 8 GiB 硬约束；`default_hw_budget()` 统一主链/测试/基准/运行期入口，sidecar 回传 `hw_budget` 由 GeneSim 核对（F20） |
| K7 | 48 层全模型主链（图编译 → sidecar → GeneSim）未在 8 GiB 下形成最终 summary | 全模型交付 | 已关闭 | round08 当轮源码 `full30b-pipeline-r8-qwen3` 已完成 A/0/BCD/A2/E：A **44117 算子、18673 GEMM、18432 专家 GEMM**；BCD **18673 GEMM + 19250 算子级**、23 份 pim mlir 全 `dma-align=64`、豁免 **6194 条**（6192 条 `device_hint=cpu` + 2 条边界）、`hw_budget={8 GiB, 64 MiB, 64}`、`moe_config={128,8,768,dynamic_dispatch}`、指纹 `cf8e71b6…`；48 层容量实测 dpu0..3 **8,258,576,576 / 8,522,825,728 B**（余 264,249,152 B）、dpu4..7 **7,791,833,024 / 8,522,825,728 B**（余 730,992,704 B）；A2 44117 算子；48 层 E 展开 30961 → 130036 分片、130806 条 trace、195676 个算子，trace `{'pimir': 106230, 'absorbed': 24576}`、**total_time_s=566.863、tokens/s=1.402**；round10 用当轮 GeneSim 代码在隔离根重跑 E，130806 条 trace 全重编、0 复用，total_time_s/tokens/s/48 层通信字节（166,702,607,112）与 round08 逐位一致，审计 ACCEPTED、容量 `max_cluster_resident_bytes=8,412,147,712 / 8,589,934,592`。round16/17 的 B 路单位契约使这批旧条目的 freshness、仿真 IR 与 trace 三入口都无法通过严格复核，结构证据保留、成本结论转 K16/Q12 |
| K8 | 48 层全模型数值目标用例在历史代码上的重跑被环境杀死 | 48 层数值验收 | 已关闭 | `full30b-r2.service` 已核实为 `Result=oom-kill`（systemd-oomd），不是代码失败。round06 的 `full30b-r6-qwen3` 数值目标已完成：**26 passed、1 skipped（17254.56 秒，4:47:34）**，30B 对拍通过、8B 无权重 SKIP，峰值内存 63.1 GiB（日志 `/tmp/qwen3-review-round06/target48l.log`） |
| K9 | 旧代码的 48 层 `run_full_pipeline` 任务与 4 层主链曾共用 `genesim/models` 下的同名产物 | 4 层主链产物的完整性 | 已规避 | round06 的 48 层任务使用独立的 `GENESIM_ROOT=/tmp/qwen3-review-round06/genesim-full48`，不写本仓 `genesim/models`；该任务已在 round08 前被外部停止（无 summary）。round08 的 4 层主链使用 `genesim-4l-r8`/`genesim-alt-r8`、48 层主链使用 `genesim-full48-r8` 隔离根；round09 的 4 层 dynamic/dense 主链写本仓 `genesim/models`（运行时无其它并发主链任务），48 层仍用隔离根 |
| K10 | 选择归 host 后，路由掩码的 `slice` 仍被 partition 按算子名判到设备，host 的 `ne`/`where` 通过 512 条 64B `all_gather` 取回掩码（约 32KB/forward） | 每次 forward 多 512 条小命令；不在关键路径（4 层 GeneSim 总时长与 round04 逐位相同 88.483s） | 已规避（记录） | 根因是 `graph/partition.py` 只看 target、不看数据流；改成通用「输入全 host 则 host」会连带改变 Llama2 注意力掩码链的落点，超出本轮根因范围，另立需求 |
| K11 | 1 专家/top-1 小模型无法走完真实 BCD：router 的 GEMM 输出宽 N=1，FlagTree 分块搜索报 `no legal tile fits: M=1 N=1 K=64`（最小 M=1 N=1 K=1 也不通过） | 单专家的正向全链用例与单专家生成用例；不影响 128 专家主链 | 已规避（记录） | 策略入口的合法/非法判定已由 parser、`resolve_strategy` 与真实主链非法分支覆盖；N=1 分块属工具链能力，另立需求（Q10） |
| K12 | `genesim/models` 的 Llama2-7B sidecar 生成于严格契约之前，缺 `hw_budget`/`ir_fingerprint`；按当前规则属于应重新导出的旧产物 | `pytest -m slow` 的 Llama2 仿真冒烟 | 未关闭 | 本轮只处理 Qwen3-MoE；旧 Llama2 产物需用当前主链重新导出后再跑慢速用例（Q11） |
| K13 | 证据块的 `measurements`（bench 加速比与历史淘汰值）与 `facts`（容量/通信/trace 计数等）来自执行期日志、历史记录或旁路审计，checker 只做声明与语境绑定，不复算这些值 | 非 summary 证据 | 已规避（记录） | 这些值没有 summary 产物可复算；独立命名空间禁止它们冒充 current，带 `run` 的 facts 还会按最近标签核对运行语境；current 运行时仍逐值由 summary 核验 |
| K14 | 正文的『某运行更高/更低』若不带具体数值，checker 无法判定结论真假 | 实施证据检查 | 已规避（记录） | 没有数值锚点就无法机器比较；证据性结论统一写成带数值的形式（如 `dynamic 60.785 s 低于 dense 63.637 s`），带数值的比较已由 `_check_comparisons` 逐条判定 |
| K15 | 当前 round16～19 工作区都没有重跑 48 层数值与主链；48 层证据是 round06 数值、round08/round10 主链历史产物 | 48 层「当前代码」结论的适用范围 | 已规避（记录） | §8.4 按「源码/产物来源—时间—层数—本轮是否重跑」分栏，明确历史产物不等于当前代码重跑；round16/17 的 B 路单位契约使旧 48 层三入口都无法通过严格复核，成本证据转 K16/Q12 |
| K16 | round16/17 的 B 路编译单位契约要求 sidecar 回传 `trace_tokens`/`local_*_shapes` 且保留 `Tq`/`Tp+Tq` 符号维；round08/10 的 48 层 19,250 条 B 路条目没有这些字段，当前验证器在 freshness、仿真 IR、trace 三入口都拒绝该产物（13106 条 B 路条目的单位不完整），48 层成本结论（total_time_s=566.863、tokens/s=1.402）在新契约下失效 | 48 层代价验收 | 未关闭 | 用户明确要求直接引用 48 层历史产物、不重跑主链；结构证据（算子数/容量/trace 计数）保留为历史，成本证据待用当前主链重跑 A/BCD/A2/E 后恢复（Q12） |

### 5.2 偏差说明
| 设计条目 | 设计方案 | 实际实现 | 差异原因 | 是否经确认 |
|----------|----------|----------|----------|------------|
| design §4.5 | 视需要新增 `_rule_weighted_partial` | 未新增 | 既有 mul 规则已覆盖 | 是（设计预留） |
| design §4.7 | 30B 测试预算 16 GiB | 统一为 8 GiB 契约（`default_hw_budget()`） | 用户确认 8 DPU × 8 GiB 硬约束（2026-11-10）；设计同步改为 8 GiB | 是（用户确认） |
| design §4.9 | 原设想 TP×EP | PP×EP（tp=1） | 与既有契约冲突，设计 §九.4 已记录并经用户确认 | 是（2026-11-05） |
| 评审问题1 | genesim 原需求把 MoE 计费排除 | 本轮纳入 MoE 图骨架与代价 | 评审 round01 明确要求 GeneSim 表达专家几何 | 是（评审要求） |
| 评审问题5 | 专家子图复用或安全分块 | 提供图级分块关闭入口并默认对 MoE 关闭 | 专家 linear 由算子层 tt.dot 分块；图级拆分放大两个数量级且无收益 | 是（评审要求提供仓库内入口） |
| 评审问题1/2（容量口径） | GeneSim 对所有常驻按单 TensorPU 512 MiB 校验 | 编译器放置下模型常驻按 ClusterPU 8 GiB 校验，KV/激活仍按 TensorPU | 编译器把一台 DPU 当一个 ClusterPU；30B embedding 622 MiB 无法放进单 TensorPU，属两仓容量模型对齐 | 是（与设计 §二.3 的 DPU 语义一致） |
| round02 问题1 | 统一容量配置贯穿架构描述、图编译、真实测试与 GeneSim | 统一为 `contracts/op_contract.py` 的 8 DPU × 8 GiB 契约；sidecar 回传 `hw_budget` 由 GeneSim 核对 | 用户已确认 8 GiB 为硬约束 | 是（用户确认 2026-11-10） |
| round02/03 问题2 | dynamic_dispatch 按每个请求的实际命中专家执行 top-k 子图 | 编译器 any-hit 全张量执行；GeneSim 按 `1-(1-k/E)^Tq` 折算专家计算与通信（decode 退化为 k/E），共享累加不缩放 | 仿真是无 routing 数据的静态代价模型；公式与执行计划的 `selected_mask.any(dim=0)` 一致，已写入 design §十.6 | 是（用户确认 8 GiB；评审要求成本与执行同源） |
| round02 问题6 | 在同一已编译计划、已装载后端上比较 P0/P1 | 一份计划 + 一个后端，切换运行期 `moe_dispatch` 开关 | 满足评审「同一计划」要求；执行语义仍由编译配置记录并交叉核对 | 是（评审要求） |
| round02 问题6 | 消除未命中的全专家累加 | 计算、视图、通信命令在提交前跳过；每层 128 个专家的加权/累加尾部仍全量执行（host `shared_reduce`，不门控） | 累加链汇合了多专家输出，按专家门控需要额外的零初始化语义；4 层真实权重实测 1.15–1.20×，累加不在剩余热路径上，不做超出根因的改动 | 是（记录取舍） |
| round04 问题1 | 理论 FLOPs 已确定的算子保留解析系数 | 仍由 B 路测量，但依赖项由配方声明；系数与解析值一致（VECTOR_MUL=768、ROPE=384、MASK=1） | 测量值是确定性算术结果，依赖项正确后无需再维护第二套公式；模板系数继续服务 A 路未覆盖算子 | 是（记录取舍） |
| round04 问题2 | 覆盖校验要求显式登记全部豁免 | 模型入口/出口按 op_type 隐式放行；host 与非 DPU 稠密 GEMM 必须显式登记（round07 起 GEMM 豁免必须带导出落点证据，不再按 `expert_id<0` 推断） | 边界是 IR 的结构事实、没有设备算子；host/GEMM 豁免会影响落点语义，必须逐条核对 | 是（记录取舍；round07 按评审收紧） |
| round04 问题2 | trace 只接受 pimir | 另接受 `absorbed`（被折进主算子的空 trace） | 融合后的被吸收节点按设计保留空 trace 占位，禁止它会把融合开关重新打成失败 | 是（记录取舍） |
| round04 问题3 | 指纹 schema 升级 | schema 1 → 2，旧指纹与旧 sidecar 全部失效，需要重导 | 拓扑入指纹必然改变哈希；A2 缓存键与 sidecar 校验会强制刷新，旧产物不能复用 | 是（记录取舍） |
| design §3.1/§十.3 | 路由加权 `mul` 与累加 `add` 走 DPU | 选择、加权乘与累加整段落在主机（`ne`/`where` 在 HOST_ONLY，加权乘与加由通用逐元素融合折成 host 闭包）；设备侧只留可门控专家子图 | PIM 原语集没有逐元素 select；实测计划中加权乘与累加本来就是 host 闭包，保留设备侧只能借语义不同的加性 mask。设计按真实计划纠正（§3.1/§十.3/§十.6 已同步） | 是（评审 round05 要求按真实计划统一设计） |
| round05 问题3 | 融合清单与被吸收编号一一对应 | 只校验 absorbed → 清单方向；清单真实性由“折激活 kernel 必须带 act_spec”守住 | `fused_activations` 在 B+C+D 产出、`absorbed_ops` 由精化步骤回写，反向闭合在 BCD 阶段必然报 512 条；两字段由不同阶段产出 | 是（记录取舍） |
| round06/07 问题1 | design §4.1/§十.6 用 `routing != 0` 描述专家命中 | 改为 top-k 选中索引生成的独立布尔 `selected_mask`；路由权重只参与加权 | `routing != 0` 把选中索引与权重值混为一个条件，权重下溢时与 HF 语义不符；设计已同步纠正（§3.1/§4.1/§十.6） | 是（评审要求按真实语义纠正设计） |
| round07 问题2 | 豁免允许「非 DPU 稠密 GEMM」 | 只有带导出落点证据（IR 与 sidecar 的 `device_hint=cpu`、角色、权重身份）的 host 算子可以豁免 | `expert_id<0` 无法区分 host GEMM 与设备 router/注意力/输出头；设计 §十.4 已改为逐条目证据核对 | 是（评审要求） |
| round06/07 问题2/3 | 容量与对齐由各入口分别构造 | `default_site_capacity()` 统一派生；B 路显式接收硬件配置；pimir 模块属性逐份核验 | 只有 sidecar 自报同一套数值不够，真实 MLIR 也必须同源；设计 §十.5 已同步 | 是（评审要求） |
| round06/07 问题3/4 | 专家身份按三元组集合核对 | 按源 IR 每个 `op_id` 核对角色、专家号与权重路径；覆盖审计、skip-simulation、仿真前 freshness、trace 验收与直接 GeneSim 用同一规则 | 集合相等不能证明产物同源；freshness 只核哈希、trace 只核来源同样会漏掉互换；设计 §十.4 已写明逐条目规则 | 是（评审要求） |
| round06/07 问题5/7 | 用 `num_experts > 1` 区分 dense/MoE | 用 `model_type` 与有效 MoE 字段判定，1 专家/top-1 合法 | 专家数不是架构身份；设计 §十.1 已写明正整数与 top-k 范围校验 | 是（评审要求） |
| round07 问题5 | 设计未约束图修改复杂度 | TP 展开与 relay 插入走 `ModelIR.batch_edit()`，循环内只记录、退出时一次重建 | 逐条 `add/remove` 各自全量重建是 O(N²) 的根因；设计补第 8 条 | 是（评审要求） |
| round08 问题1/2 | design §十.4 只写产物契约 | 主链 E 阶段只消费本次 BCD 返回的 placement；临时配置显式改写并回读复核，再按策略/DPU/段数/Cluster/本地形状核对；运行与结果目录按架构/策略派生 | 静态仿真配置会让另一种合法策略消费旧 sidecar；结构指纹按设计排除放置字段，无法靠指纹发现错配 | 是（评审要求） |
| round08 问题3 | design §4.2 只在注册表自动选择里要求专家整除 | `Architecture.strategy_check` 让自动选择、直接段数与 PartitionPlan 共用同一约束 | 直接段数与方案转换绕过注册表；设计 §十.1 的合法 MoE 范围需要一致判定 | 是（评审要求） |
| round08 问题4 | design §十.4/§十.5 要求导出、skip-simulation、仿真前、trace 与直接 GeneSim 同一规则 | 新增 `validate_sidecar_contract`，`audit_sidecar_coverage` 作为其覆盖组件；编译器侧五个入口共同调用 | 各入口各自实现导致缺量/缺豁免证据仍放行；共享验证器是根因修复 | 是（评审要求） |
| round08 问题5 | design §十.5 要求容量/对齐唯一真源 | `serve_pim` 预检直接调用 `build_kv_layout`，删除独立 `ALIGN=256` 与 4 GiB 示例 | 预检与实际规划两套公式、两套对齐是根因 | 是（评审要求） |
| round09 问题1 | design §十.4 要求 absorbed 由融合清单、主算子 pimir 与 IR 依赖边唯一证明 | 补「主算子只有被吸收激活一个真实消费者」方向；BCD 导出允许清单先行，A2 后所有复核入口与直接 GeneSim 要求清单/吸收编号双向闭合 | 只查生产者方向、同一布尔校验无阶段参数，会让多消费者融合与「清单在、吸收编号被删」两种产物放行 | 是（评审要求；设计已同步） |
| round09 问题2 | design §十.4 要求豁免不得与实际 PIM 放置矛盾 | 豁免逐条交叉核对 sidecar `operators` 放置条目与 `host_gemms` 角色/权重身份，模型边界同样适用；两仓同一规则 | round07 的放置矛盾检查只落在 GeneSim，round08 收拢共享验证器时未移植，且 GeneSim 的边界豁免分支被提前跳过 | 是（评审要求；设计已同步） |
| round10 问题1 | design §十.6 要求专家计算与通信都按 any-hit 折算 | 依赖边同时接收源、目的角色：专家产出的边随源命中并集，共享输入到专家随消费者，shared→shared 全量；两处出口共用 | 只按目标角色判定会把专家输出到共享累加的边当成共享工作；执行器对未命中专家直接返回零张量、不做收集 | 是（评审要求；设计 §十.6 已补边方向） |
| round10 问题2 | 实施证据必须与产物一致且可追溯 | 数值以 summary 为唯一来源回填；新增 `scripts/check_implement_evidence.py` 做机器一致性检查，历史值显式声明 | 人工誊写缺少校验，错误值会复制到多处；只在文档里改一处不能防止复发 | 是（评审要求） |
| round11 问题1 | design §十.5 要求导出、仿真前复核与直接 GeneSim 使用同一容量规则 | 严格 compiler-placement 路径的期望容量固定为需求常量 `DPU_MRAM_BYTES=8 GiB`；有效配置、sidecar 三件套与每份 MLIR 属性都对该常量核对 | “配置与产物自洽”不等于“符合需求硬约束”；只有明确标识的旧式非严格产物保留其它硬件容量兼容 | 是（用户确认 8 GiB 硬约束） |
| round11 问题2 | design §十.4 要求运行与结果目录按架构与策略派生，禁止跨任务删除共享固定目录 | 运行目录再加入执行语义与唯一运行号；A/A2/BCD 产物、E 的 trace/results/log 与临时配置全部落本次目录；共享 `pim_traces` 不再删除，共享 results 不再读取 | 只绑定 sidecar 与日志副本不足以隔离 A/A2/BCD/E 全链；唯一空 trace 目录本身就能保证全量重编 | 是（评审要求） |
| round11 问题3 | design §十.4/§十.5 要求实施数值与产物一致且可追溯 | 实施文档嵌入机器可读证据块：current 条目逐标签绑定 summary，核对层数/模型/执行语义并重算派生值；history 与 measurements 独立命名空间 | 全篇允许值集合无法表达“哪个值属于哪次运行”，也无法证明派生结论；结构化块才能在错配时失败 | 是（评审要求） |
| round12 问题1 | design §十.4/§十.5 要求导出、skip-simulation、仿真前复核与直接 GeneSim 用同一份容量规则 | 严格性 = 仿真配置的 `require_compiler_pimir` ∪ sidecar 自报；五个编译器入口共用一份判定，MLIR 属性仍在导出与 GeneSim 加载时逐份解析 | 配置是调用方的明确声明，自报是被验对象可删除的状态；只信自报等于把严格性交给产物自己决定 | 是（评审要求；设计 §十.4/§十.5 已补） |
| round12 问题2 | design §十.4/§十.5 要求实施数值与产物一致且可追溯 | 正文按块级前置运行标签绑定数值，必需值只扫去掉证据块的正文，带数值的大小结论按两侧实际值判定 | 全篇允许集合只能证明「值在文档某处出现过」，不能证明它出现在对应运行语境；绑定到块级标签才能拒绝只改正文的错配 | 是（评审要求） |
| round13/14 问题1 | design §十.3 与 `PIMOps.td` 要求每块 PV 乘积先落 f16 | lowering 在 PV 累加前调用 `roundF16`；相位表 `bytes` 仍记 f32 累加器，f16 乘积是块内临时值 | `PhaseSpecAttr.bytes` 的语义是「该相流式读写的缓冲区」，PV 相流式读写的是 f32 累加器；f16 乘积不是主缓冲，不能改 `bytes`，否则会把容量计费算错 | 是（评审要求；与 `PIMOps.td` 契约一致） |
| round13/14 问题2 | 块表项合法集合 `0 <= slot < pool_slots` | 主机预检逐项失败；设备侧 `pim_block_slot` 中止非法项 | 主机预检给出可读错误且不触发进程终止；设备兜底只防直接调 kernel 的旁路，正常路径不会走到 | 是（评审要求） |
| round13/14 问题4 | trace 必须绑定 sidecar 的 `pimir_sha256` | 仅当 sidecar 条目带哈希（或严格上下文要求哈希）时强制；非严格旧产物没有哈希时保留来源检查 | `pimir_sha256` 是当前导出产物的字段，旧式纯放置 sidecar 没有该字段；严格性上下文与 round12 的并集规则一致 | 是（评审要求；旧式兼容边界） |
| round13/14 问题5 | 容量、通信、trace 计数等证据值纳入同一绑定或独立命名空间 | 通信字节按运行绑定；参数、容量、trace 计数等按 `facts` 独立命名空间登记并写明来源 | 这些值不在 summary 三项指标内，无法由 checker 复算；独立命名空间保证可追溯且禁止冒充 current summary 指标，逐运行绑定用于最易错配的通信字节 | 是（评审要求允许二选一） |
| round15 问题3 | design §十.4 规定 trace 验收按源 IR 对拍，没有规定请求分布输入由谁生成 | `prepare_simulation_config` 在配置 trace 缺失时用 GeneSim 官方生成器在本次运行目录生成并绑定绝对路径；已有 trace 原样使用 | 干净检出里 `genesim/traces/` 被 `.gitignore` 排除；评审问题3 要求主链纳入生成或校验，而不是等到 E 阶段才失败 | 是（评审要求；不改变请求分布语义，生产运行仍用已有 trace） |
| round17 问题1 | design §十.7 要求视图/嵌入按 `Tq` 计费 | sidecar 与 trace 都只认调度器绑定的实际 Tq；IR 里没有符号 `Tq` 的全具体形状按一次执行 | 固定形状的旧式夹具/IR 与请求长度无关，无条件按请求重复会把固定算子执行 N 遍（I80）；真实 B 路 IR 都带 `Tq` | 是（记录取舍） |
| round17 问题2 | design §十.4 要求 B 路单位可核对 | sidecar 的 `local_*_shapes` 由「具体占位」改为「保留 `Tq`/`Tp+Tq` 的符号模板」；具体占位只在编译时展开 | 具体占位无法区分历史维与本地宽；改符号模板后两仓比较语义才一致。旧式具体单位在严格模式被拒绝，需要当前主链重导 | 是（评审要求；schema 语义收紧） |
| round17 问题3 | design §十.4 要求五个编译器入口共用同一判定 | 单位完整性挂到 `audit_sidecar_coverage` 严格分支，导出、skip-simulation、freshness、仿真 IR 与 trace 共用；trace 入口另做 trace-vs-unit 字节核对 | 单位只接 trace 消费点会让前三者签发 GeneSim 必然拒绝的产物；共享验证器是根因修复 | 是（评审要求） |
| round17 问题4 | 评审建议 GeneSim 入口核对「与编译器默认」的执行语义 | GeneSim 只核对 IR/sidecar 自身一致；「等于编译器默认」由共享产物刷新 + 编译器产品测试保证 | 仿真器不 import 编译器（两仓独立），运行时跨仓依赖会把 GeneSim 绑到编译器安装；产品层测试同样能阻止过期产物 | 是（记录取舍） |
| round18 问题1 | design §十.4 要求覆盖集合以源 IR 为准 | A/B 路分类也收回到源 IR：`op_type == GEMM` 才是 A 路，其余算子都是 B 路；`shards` 只作被验产物字段 | 分片自报既是被验对象又是分类依据，假分片能同时骗过验证器与运行期绑定；两仓各自实现同一规则 | 是（评审要求；设计 §十.4 已补） |
| round18 问题2 | design §十.4 要求 B 路编译单位可核对 | sidecar 单位数值必须与所引用 pimir 的 `pim.traceTokens`/特征末维一致；SPLIT 纯视图只比较输入末维 | 只验单位结构与符号维仍让 sidecar 自证；pimir 是编译实产，必须作为唯一真源 | 是（评审要求；设计 §十.4 已补） |
| round18 问题3 | 全量回归必须可重复 | 缓存锁测试等待上限改 60 秒死锁检测，teardown 回收 fork 子进程；新增慢调度回归用例 | 5 秒是机器负载的函数，不是锁语义；子进程残留会挂住 pytest 进程 | 是（评审要求） |
| round18 问题4 | 实施证据必须可复现 | checker 声明改为只核 dynamic/dense 两份 current summary；alt/48l 标为 history 正文引用 | history 条目不能用 `--summary`/`--time-only` 核验，混写会让历史值冒充机器核验 | 是（评审要求） |
| round18 问题5 | design §十.5 容量按 ClusterPU 校验 | 4L 引用 `max_cluster_resident_bytes`/`max_cluster_utilization`；per-TensorPU 最大值与 512 MiB 上限单列 | 同名不同 scope 字段不能配成一对 | 是（评审要求） |
| round18 问题6 | design §十.5 要求容量三件套单一真源 | 8B/30B 真实测试统一 `default_site_capacity()`；静态守卫禁用容量/对齐/WRAM 字面量 | round03 只覆盖了 30B 的 `HwBudget`，没有整套三件套与守卫 | 是（评审要求） |
| round19 问题1 | design §十.4 要求 A 路 GEMM 绑定 pimir 语义/形状 | 接受 `tt.dot` 与 `pim.matmul` 两种分块 GEMM 原语；K/N 循环嵌套核对只对 `tt.dot` 生效，`pim.matmul` 只核 M/K/N、tile 属性与分片宽度 | 当前 linear kernel 实产都是 `tt.dot`；`pim.matmul` 是相位展开后的合法形态，不应判死，但两种原语都必须与 `kernel_tile_n`、本地宽、融合激活一致 | 是（记录取舍） |
| round19 问题3 | design §十.4 要求设备 GEMM 的 `semantic_role` 是源 IR 投影身份 | 只有源 IR 带角色时才要求条目必填并一致；条目多带源 IR 没有的角色仍失败 | 真实 30B 骨架每个 GEMM 都有角色（导出侧保证）；合成/旧夹具没有身份时不能凭空要求，否则会把无身份 IR 判成产物错误 | 是（记录取舍） |
| round19 问题4 | design §十.3 的相位存储边界只描述设备 f16 路径 | 主机回退对 f32 KV（只有测试会造）也按同一张表落 f16 | 该函数是设备相位表的镜像，设备只有 f16 存储；既有 f32 用例容差 0.05，取整不改变结论 | 是（记录取舍） |
| round19 问题5 | design §十.5 要求利用率与判定同源 | 保留 `max_utilization`/`max_tensor_pu_utilization` 键名但语义改为判定利用率；总量比新增 `max_total_resident_ratio` | 下游 autotune 数据集与控制台在用旧键名；改语义而不改键名可避免 schema 断裂，同时消除 145% 与 fits=True 的矛盾 | 是（记录取舍） |
## 六、后续与待确认事项
| 编号 | 事项 | 状态 | 关闭说明 |
|------|------|------|----------|
| Q1 | 30B 下载完成后执行 T6b 验收并回填 4.3 | 已关闭 | 4 层切片通过；全模型结果见 4.3 |
| Q2 | P1-a 动态分派原语（§4.8） | 已关闭 | 子图/通信整条门控，见 F8 |
| Q3 | P1-b 专家并行 ep_width（§4.9） | 已关闭 | PP×EP 落地，16 DPU 用例守护 |
| Q4 | `-k llama2_7b` 回归门槛结果回填 4.3 | 已关闭 | 48 passed（0:53:44），与基线一致（历史记录） |
| Q5 | **8 GiB 容量口径**：需求 P0-3 写 8 DPU×8 GiB，设计 §4.7 与测试曾用 16 GiB；主链导出曾硬编 4 GiB | 已关闭 | 用户确认 8 GiB 为硬约束（2026-11-10）；统一契约与 sidecar 回传见 F20；48 层全量余量未实测，转 K7 |
| Q6 | 48 层全模型目标用例的最终结果 | 已关闭 | round01 历史 23 passed、1 skipped；round02 的 `full30b-r2` 为 oom-kill（环境阻塞）；round06 用当时源码完成 **26 passed、1 skipped（4:47:34）**，30B 通过、8B SKIP。本轮改动（选中掩码/豁免/对齐/批量图修改）不改变正常权重的数值路径，但当前代码的 48 层数值重跑与主链终态合并到 Q8 跟踪 |
| Q7 | **P1-1「显著加速」验收口径**：提交前跳过已消除未命中命令的调度与清零，4 层真实权重执行期加速比为 prefill 1.148×、decode 1.202×（旧实现 0.994×） | 已关闭 | 用户接受 1.148×/1.202× 作为加速验收（2026-11-10）；剩余执行后端优化另立需求，不阻塞本需求 |
| Q8 | 当前代码的 48 层数值目标用例与 8 GiB 主链终态 | 已关闭 | 48 层主链已用 round08 当轮源码重跑并完成（`full30b-pipeline-r8-qwen3`，exit 0）：44117 算子、18673 GEMM、18432 专家 GEMM；sidecar 18673+19250、23 份 pim mlir 全 64、豁免 6194 条带证据；容量 dpu0..3 8,258,576,576/8,522,825,728 B、dpu4..7 7,791,833,024/8,522,825,728 B；trace `{'pimir': 106230, 'absorbed': 24576}`、total_time_s=566.863；round10 的当轮 GeneSim 代码 E 重跑与 round08 逐位一致（tokens/s=1.402、通信字节 166,702,607,112、审计 ACCEPTED）；48 层数值保留 round06 历史 26 passed/1 skipped（K7/K8）；round16/17 的单位契约使旧 48 层 B 路条目在三个入口都无法通过严格复核，成本证据失效并转 K16/Q12 |
| Q9 | 旧代码的 48 层主链任务与 4 层主链共用 `genesim/models` 同名产物 | 已规避 | round06 的 48 层任务使用独立的 `GENESIM_ROOT=/tmp/qwen3-review-round06/genesim-full48`，不写本仓 `genesim/models`；round09 的 4 层 dynamic/dense 主链已用当前代码完整重跑并逐段核对同源（K9） |
| Q10 | FlagTree 对 N=1 的 GEMM 输出没有合法分块，1 专家/top-1 模型无法走完真实 BCD | 未关闭 | 属工具链能力（K11）；合法/非法策略入口已由 parser、`resolve_strategy` 与真实主链非法分支覆盖，不通过放宽专家约束绕过 |
| Q11 | `genesim/models` 的 Llama2-7B 旧 sidecar 缺 `hw_budget`/`ir_fingerprint`，严格契约下需重导 | 未关闭 | 本轮只处理 Qwen3-MoE；用当前主链重新导出 Llama2 产物后再跑 `pytest -m slow`（K12） |
| Q12 | 在新 B 路编译单位契约下重跑 48 层 A/BCD/A2/E，恢复 48 层成本证据 | 未关闭 | round16～19 只重跑 4 层；48 层旧 B 路条目缺单位且无符号维，严格验证器在 freshness/仿真 IR/trace 三入口都拒绝该产物（K16）。内存可用（`free -g` ≥ 120 GB）时用 systemd 重跑，并回填 §8.4/§9.2 |
| Q13 | 独立入口的共享 `genesim/models/` 产物需随主链刷新 | 已关闭 | round17 用当前主链 4L dynamic 产物刷新 6 个文件；模板说明与输出目录已修正，编译器产品测试钉住执行语义=默认、指纹与严格契约（F71） |

## 七、评审问题修复记录
> 关联评审文档：review-qwen3-moe-20261109-round01、review-qwen3-moe-20261109-round02、
> review-qwen3-moe-20261110-round03、review-qwen3-moe-20261110-round04、
> review-qwen3-moe-20261110-round05、review-qwen3-moe-20261110-round06、
> review-qwen3-moe-20261111-round07、review-qwen3-moe-20261111-round08、
> review-qwen3-moe-20261111-round09、review-qwen3-moe-20261111-round10、
> review-qwen3-moe-20261111-round11、review-qwen3-moe-20261111-round12、
> review-qwen3-moe-20261111-round13、review-qwen3-moe-20261111-round14、
> review-qwen3-moe-20261112-round15、review-qwen3-moe-20261112-round16、
> review-qwen3-moe-20261112-round17、review-qwen3-moe-20261112-round18、
> review-qwen3-moe-20261112-round19

### 7.1 问题确认与修复清单
| 问题编号 | 来源评审文档 | 严重性 | 确认结果 | 根本原因 | 修复状态 | 修复方式 | 同类问题排查 | 涉及文件 | 对应测试用例 |
|----------|--------------|--------|----------|----------|----------|----------|--------------|----------|--------------|
| 问题1 | round01 | 严重 | 成立：30B IR 只有 336 GEMM、FFN 6144 稠密、parameter_bytes 5.642 GiB | 通用 Llama 构造器把几何/语义/成本写死；ModelIR 无专家元数据 | 已修复 | ModelIR 增专家数/top-k/专家宽度/执行语义并校验；qwen3_moe 建 router+门控+每专家三件套；不支持的 MoE 类型显式报错；expert_id 贯穿 sidecar | dense Llama/Qwen 路径字段全 0，行为不变 | genesim/src/ir/model_ir.py 等 | genesim TestQwen3MoeBuild 6 条、TestParserQwen3Moe |
| 问题2 | round01 | 严重 | 成立：主链固定 Llama 七投影与 llama2 命名 | 模型路径/加载类/策略/角色/文件名/配置分散硬编码 | 已修复 | `Architecture` 增 artifact/required_roles/widths/sim_config；`architecture_of` 读 config；pipeline/放置导出全走注册表 | 所有模型入口统一；Qwen2/Qwen3 dense 仍走同一路径 | backend/architectures.py、scripts/* | test_architecture_registry_drives_moe_pipeline、Qwen3 全链路 |
| 问题3 | round01 | 严重 | 成立：未选中专家 inf 经 0×inf=nan 污染 | 静态前向先算全专家再乘路由零 | 已修复 | 路由掩码经 `ne`/`where` 先选零；补 spec_prop 规则、bool dtype、numpy 内核；P0/P1 两态回归 | 所有静态/host 回退共用同一前向 | backend/architectures.py、graph/spec_prop.py、runtime/kernels.py | test_*overflow* 2 条 |
| 问题4 | round01 | 高 | 成立：sidecar 自报 requires_pimir 但缺条目仍生成模板 trace | 合法无 trace 与应有缺项混为 None；产物完整性检查分散 | 已修复 | 逐类豁免清单 `pimir_exempt_ops`；严格模式先核模型身份/编号/路径/哈希/缺项再编 trace | llama2 既有 sidecar 与纯放置 sidecar 行为保持 | genesim scheduler、placement_export、run_full_pipeline | TestPimirContractCompleteness 6 条 |
| 问题5 | round01 | 高 | 成立：4 层默认分块到 139 万节点；外部 monkeypatch 才算关闭 | 通用 tile 无 MoE 约束，仓库无受支持关闭入口 | 已修复 | `graph_tiling` 参数 + MoE 默认关闭 + `PIM_DISABLE_TILE`；算子层 tt.dot 分块保留 | prefill/decode 共用同一编译入口 | runtime/compile.py、graph/opt_passes.py | test_moe_graph_tiling_policy 等 3 条 |
| 问题6 | round01 | 高 | 成立：30B 用例参考与被测共用静态化模型 | 加载器原地替换 forward，测试未保留原实现 | 已修复 | 加载时保存 `_hf_sparse_forward`；编译后 `restore_hf_moe_forward`；断言参考未走静态实现 | 4 层真实权重对拍通过 | backend/architectures.py、tests/test_qwen3_30b_real.py | test_qwen3_30b_generate_matches_hf（4 层 288.08s） |
| 问题7 | round01 | 高 | 成立：16 GiB 预算未记录需求变更，8 GiB 未验证 | 设计阶段直接提高硬件假设 | 已修复（round03） | 用户确认 8 GiB 硬约束；统一契约并随 sidecar 回传，见 round03 问题2 | 所有 PP/TP 与容量入口 | 需求/设计/测试 | TestHwBudgetContract 等 |
| 问题8 | round01 | 高 | 成立：分块后专家识别失败；通信与分支内算子无 gate | 识别依赖未分块图的直接权重参数；门控只覆盖 linear | 已修复 | 拓扑序专家归属传播；整条子图与通信打标；主机跳过返回零张量 | 覆盖 prefill/decode 与 EP/host 回退 | runtime/exec_plan_gen.py、backend/hal_numpy.py | test_tiled_* 3 条、test_gate_skipped_host_command_returns_zero |
| 问题9 | round01 | 中 | 成立：架构单测无条件断言站点路径存在 | 资源配置检查混入架构单测，依赖矩阵未收录 30B | 已修复 | 改为 monkeypatch 解析单测；真实目录由真实用例 skipif 守护；README 补依赖矩阵 | 8B skip 与 30B skip 语义一致 | tests/test_architectures.py、tests/README.md | test_qwen3_30b_model_dir_resolution |
| 问题10 | round01 | 中 | 成立：实施文档停留旧 P0 范围 | 未按“契约→实现→产物/测试”维护可追溯矩阵 | 已修复 | 就地重写 §1-§7，新增 §8 全链路验收 | — | docs/implement-qwen3-moe-20261105.md | 本文 |
| round02 问题1 | round02 | 严重 | 成立：主链 4 GiB / 真实用例 16 GiB / GeneSim 8 GiB 三处独立预算；48 层在 4 GiB 下按专家权重即超预算 | 没有统一硬件容量契约，8 GiB 是否硬约束未确认 | 已修复（round03） | 统一到 `default_hw_budget()`；sidecar 回传 `hw_budget`；48 层余量记 K7 | 所有 PP/TP 策略与全模型 sidecar | export_pp_placement.py、test_qwen3_30b_real.py、design §十.5 | round03 问题2 行 |
| round02 问题2 | round02 | 严重 | 成立：编译器默认动态分派，GeneSim 固定 dense_sim，专家计费差 16 倍 | `moe_execution` 只是 IR 自描述，未从编译配置派生；GeneSim 只有全专家一套图 | 已修复 | `compile_llama2(moe_execution=)` 统一解析，经 parser、sidecar、仿真三处传递并交叉核对；GeneSim 按 top_k/E 与命中并集计费，并把比例下发到 PIMVPU 的 trace 时延 | 非 MoE 模型为空串，Llama 路径行为不变；dense_sim 仍全专家计费 | runtime/exec_plan_gen.py、runtime/compile.py、genesim_bridge/placement_export.py、genesim/scripts/model_parser.py、genesim/src/ir/model_ir.py、genesim/src/scheduler/gene_sim_scheduler.py | TestMoeExecutionContract、TestDynamicDispatchCost、TestMoeConfigLineage、Qwen3 全链路 summary |
| round02 问题3 | round02 | 高 | 成立：显式严格开关、fresh 缺哈希、未知算子自动豁免、缺 moe_config 四条绕过仍在 | 严格性拆成配置与自报两套状态；豁免理由过宽；缺项检查不完整 | 已修复 | `_strict_pimir()` 统一；sidecar 缺失/缺条目/缺哈希在编 trace 前失败；豁免白名单四类；MoE IR 缺 moe_config 失败 | 纯放置导出保持宽松；合法 host/边界不受影响 | gene_sim_scheduler.py、run_full_pipeline.py、placement_export.py | test_explicit_strict_prechecks_without_self_declaration、test_missing_sidecar_file_raises_when_required、test_export_step_rejects_a_fresh_sidecar_without_hashes、test_unknown_device_op_is_not_silently_exempted、test_sidecar_without_moe_config_is_rejected |
| round02 问题4 | round02 | 高 | 成立：4 层 sidecar 里 512 个 VECTOR_MUL 的 pimir 全部是 `pim.eltwise<add>` | B 路构造器只接收 mnemonic 和形状，VECTOR_ADD/VECTOR_MUL 共用 add 分支 | 已修复 | 按 op_type 映射 add/mul 与 silu/gelu，未知类型拒绝；导出侧校验 IR 文本里的 kind，仿真侧加载时再核对 | llama2 的 VECTOR_MUL 仍映射为 mul，dense 回归不变 | placement_export.py、gene_sim_scheduler.py | test_bpath_eltwise_keeps_operator_kind、test_bpath_rejects_unknown_eltwise_operator、TestPimirOperatorKindIsVerified |
| round02 问题5 | round02 | 高 | 成立：op119 在 Tq=128 时 262144 FLOPs、系数 16；实际输入宽 768 | 配方工厂用模型级 hidden/4×hidden 代表维度，未接收逐算子形状 | 已修复 | 配方接收宽度；`_measure` 从算子输出形状取宽度；专家乘/激活 768、残差加 hidden | llama2 同宽路径不变；精化 IR 与 summary 同步修正 | op_classify.py、cost_extractor.py | test_eltwise_and_activation_costs_follow_the_operator_shape |
| round02 问题6 | round02 | 高 | 成立：旧实现未命中专家仍逐条提交、逐块清零，A/B 0.994× | 跳过判定放在执行期，提交面与清零面没有收敛 | 已修复（1.148×/1.202×，用户已接受；Q7 关闭） | 提交前等 dispatch 再决定是否入队；`skip_zero` 沿图边只标跨子图的输出；同一计划/后端用 `moe_dispatch` 开关做 A/B；新增仓库内基准 | token 与稠密完全一致；命中专家数 120/128 可逐层核对 | backend/hal_numpy.py、runtime/exec_plan_gen.py、runtime/executor.py、scripts/bench_moe_dispatch.py | test_skipped_launch_is_never_submitted、test_gate_skipped_launch_without_skip_zero_leaves_the_buffer、test_benchmark_harness_flips_modes_on_one_backend、bench_moe_dispatch 4L 输出 |
| round02 问题7 | round02 | 中 | 成立：§1.3 宣称为全部完成，§8.4 仍待回填，与容量/加速缺口矛盾 | 实施文档未按证据矩阵在长任务结束后收敛 | 已修复 | 就地重写 §1.3、§3、§4、§5、§6、§8、§9；4 层/48 层、历史/本轮分栏 | — | docs/implement-qwen3-moe-20261105.md | 本文 |
| 问题1 | round03 | 严重 | 成立：4 层 IR 无词表维 GEMM、无 q_norm/k_norm，`parameter_bytes`=5,607,297,024，比完整 HF 参数少 622,331,904 字节 | 通用构造器沿用简化 Llama 骨架，只补了 MoE 分支；parser 单独统计 `num_parameters`，图内缺项无判据 | 已修复 | 补 lm_head GEMM（tied 去重）与逐头 q/k norm 及权重；`parameter_bytes` 按 tensor_id 去重并与 HF 参数交叉校验；主链核对 role/形状/参数量；放置导出补模型级 lm_head 映射与分片；GeneSim TP 展开覆盖输出头 | 同类：qwen3 dense 与 llama 骨架同样补齐；对拍覆盖 llama/qwen3/qwen3_moe | genesim/src/ir/model_ir.py、genesim/scripts/model_parser.py、backend/architectures.py、genesim_bridge/placement_export.py、scripts/run_full_pipeline.py、genesim/src/scheduler/gene_sim_scheduler.py | `TestFullGraphCoverage` 4 条、`test_compiler_graph_and_genesim_ir_cover_the_same_weights` 3 条、`test_model_level_lm_head_is_exported_with_its_own_weight`、`test_lm_head_is_expanded_across_tensor_parallel_shards`、4 层全链路 |
| 问题2 | round03 | 严重 | 成立：主链 4 GiB、真实测试/基准 16 GiB、GeneSim 8 GiB 三套预算 | 没有统一容量真源，各入口独立硬编 | 已修复 | `contracts/op_contract.py` 增 8 DPU × 8 GiB 契约与 `default_hw_budget()`；主链/真实测试/基准/运行期入口全部改用；sidecar 回传 `hw_budget`，GeneSim 启动前核对同值 | 所有 PP/TP 策略、三区规划、执行基准；48 层余量按用户要求保留为 K7 | contracts/op_contract.py、memory/mem_planner.py、scripts/export_pp_placement.py、scripts/bench_moe_dispatch.py、tests/test_qwen3_30b_real.py、backend/vllm_worker.py、scripts/serve_pim.py、scripts/profile_execution.py、genesim/src/scheduler/gene_sim_scheduler.py | `test_sidecar_carries_the_single_capacity_contract`、`TestHwBudgetContract` 3 条、4 层真实目标命令（8 GiB 下 26 passed） |
| 问题3 | round03 | 严重 | 成立：动态仿真对专家计算统一 top_k/E（Tq=16 时 0.0625 vs 实际 0.6439），MOE_ACCUM 也被缩放；修复中又发现 TP 展开丢 `moe_cost_role`，动态计费根本没进仿真（dense/dynamic 逐位相同） | 用 `expert_id` 同时表达产物归属与可门控计算；共享累加没有独立成本角色 | 已修复 | IR 增 `moe_cost_role`；专家计算/通信按 `1-(1-k/E)^Tq`（decode 退化为 k/E），共享累加保持 1.0；Relay 运行期标 `expert_comm`；补成本角色校验；TP 展开的分片在 `split_operator` 四个分支继承角色（I34） | 非专家、共享累加、通信、TP 分片与专家归约节点、trace 时延路径逐条核对 | genesim/src/ir/model_ir.py（`split_operator` 继承角色）、genesim/src/scheduler/gene_sim_scheduler.py | `TestDynamicDispatchCost` 5 条、`TestDynamicDispatchPimLatency` 3 条、`test_dynamic_dispatch_shortens_expert_compute_on_the_critical_path` |
| 问题4 | round03 | 高 | 成立：sidecar 只核总数/op_id/op_type，A2 只按编号序列复用；改词表/model_id 的骨架仍被接受 | 模型身份只有少量 MoE 字段，没有覆盖几何与算子结构的指纹，缓存键也不含融合清单与成本版本 | 已修复 | IR 增 model_type/vocab_size/num_kv_heads/execution_dtype/tie/fingerprint（双仓同一算法）；sidecar 回传 `ir_fingerprint`，GeneSim 在生成 trace 前重算比对；A2 缓存键含指纹/档位/融合清单/成本版本；model_id 改为路径无关的仓库名或 `模型类型:指纹前缀`，切分方案身份不一致直接失败 | 旧 sidecar 无指纹时：IR 无身份按旧行为、IR 有身份则严格模式失败；llama2 产物重导后一致 | genesim_bridge/ir_fingerprint.py、genesim_bridge/placement_export.py、scripts/run_full_pipeline.py、scripts/export_pp_placement.py、genesim/src/ir/model_ir.py、genesim/src/scheduler/gene_sim_scheduler.py | tests/test_ir_fingerprint.py 6 条、`TestModelIdentityFingerprint` 4 条（genesim）、`TestModelIdentityFingerprint`（model_ir 4 条）、`test_a2_cache_key_covers_fingerprint_and_fusion`、`test_a2_rejects_a_refined_ir_from_another_model` |
| 问题5 | round03 | 中 | 成立：实施文档仍写 8 GiB 待确认、P1-1 待确认、`full30b-r2` 仍在运行、dense_sim 46.627 与本轮实测不符 | 证据矩阵未随用户确认与后台任务终态收敛 | 已修复 | 就地更新 §1/§3/§4/§5/§6/§8/§9：Q5/Q7 关闭；OOM 记环境阻塞；48 层/4 层、历史/本轮分栏；dense/dynamic 用同入口重测并解释差异 | — | docs/implement-qwen3-moe-20261105.md、docs/design-qwen3-moe-20261105.md、tests/README.md | 本文 §4/§7.3/§8/§9 |
| 问题1 | round04 | 严重 | 成立：4L 精化 IR op155 写成 `6×Tq(Tp+Tq)`；Tp+Tq=727 时 272.625 FLOPs/26172 B，按 768 宽度应为 48 FLOPs/288 B；q_norm 3635 vs 640 | 拟合按「GEMM / 非 GEMM」一刀切选公式，配方没有声明成本依赖项；round02 问题5 只修宽度未修符号项，测试还把错误项固化成期望 | 已修复 | `OpRecipe.cost_term` + `Measurement.cost_term` 驱动测量与拟合；当前 token 类 Tq、注意力类 Tq(Tp+Tq)；ROPE/MASK 改逐头形状；`SIDECAR_VERSION` 升 1.3；三个 Tp 点与注意力对照用例 | 全仓 19 个算子级配方逐类核对依赖项；真实 4L 精化 IR 复核：VECTOR_MUL `{Tq:768}`（decode Tq=1、top-8/128 下有效 48 FLOPs/288 B）、q_norm `{Tq:640}`、ROPE `{Tq:384}`、1994 个当前 token 类非 GEMM 算子无历史项、128 个注意力算子保留 `Tq(Tp+Tq)` | genesim_bridge/op_classify.py、cost_extractor.py、tests/test_ir_cost_eval.py、genesim_bridge/ir_fingerprint.py（文档）、docs/design | `test_current_token_costs_ignore_kv_history_and_attention_tracks_it`、`test_eltwise_and_activation_costs_follow_the_operator_shape`、`test_fusion_saves_exactly_the_absorbed_tensor_bytes` |
| 问题2 | round04 | 高 | 成立：缺 pimir 的专家 GEMM 与 `BRAND_NEW_OP` 改放 GPU 后严格预检直接返回；无 `trace_source` 的 trace 以 unknown 计数后主链放行；`step_bcd_export` 只查专家号并集 | 覆盖集合由「最终落点及已生成文件」反推，缺条目会改变落点、让校验对象消失；豁免无语义核对；trace 验收只拒绝 template | 已修复 | 源 IR 驱动的覆盖校验在放置与 trace 之前执行（直接 GeneSim、主链导出、skip-simulation 共用）；豁免只允许模型边界/host/非 DPU 稠密 GEMM；MoE 按 (层,角色,专家号) 核对；trace 只接受 pimir/absorbed，按应覆盖集合对拍，分片用 `parent_op_id` 回指 | 直接 GeneSim、主链、skip-simulation、GPU/CPU 落点、未知算子、纯放置五条路径逐条覆盖；复核发现真实骨架的 lm_head 从未展开、没有 trace（I39），本轮一并修复 | gene_sim_scheduler.py、run_full_pipeline.py、placement_export.py、test_compiler_placement.py、test_genesim_simulation.py、test_placement_export.py | `test_missing_op_placed_on_gpu_is_still_rejected`、`test_missing_op_placed_on_cpu_is_still_rejected`、`test_all_ops_on_gpu_are_still_checked`、`test_unknown_device_op_cannot_be_exempted`、`test_audit_sidecar_coverage_*`、`test_trace_provenance_*`、`test_skip_simulation_runs_the_coverage_audit`、`test_lm_head_is_expanded_across_tensor_parallel_shards` |
| 问题3 | round04 | 高 | 成立：仅把一条依赖 `0→1` 改为 `1→0`，双仓指纹 SHA256 完全相同 | round03 问题4 把「模型身份」实现成算子清单摘要，没有把依赖拓扑与层归属纳入结构定义；测试只覆盖词表/hidden | 已修复 | 双仓指纹 schema 升 2，补依赖端点/张量身份与 subgraphs；精化/放置字段继续排除；TP 展开前完成同源校验、展开后重算自身指纹 | 直接 GeneSim sidecar 校验、A2 缓存、精化 IR、展开图四处核对 | genesim/src/ir/model_ir.py、genesim_bridge/ir_fingerprint.py、gene_sim_scheduler.py、tests/test_ir_fingerprint.py、genesim/tests/sim/test_model_ir.py | `test_fingerprint_tracks_dependency_topology_and_layer_ownership`、`test_fingerprint_tracks_topology_and_layer_ownership`、`test_fingerprint_ignores_refinement_and_placement_fields` |
| 问题4 | round04 | 中 | 成立：E=8、top_k=1、Tq=1 时同一条 16 字节边，显式系数路径计 2 B，旧式 `(src,dst)` 路径计 16 B | `dependency_transfer_bytes` 在旧式分支提前返回完整载荷，折算只写在显式分支里 | 已修复 | 先求原始 always 载荷（显式系数或目标形状），再在唯一出口按目标成本角色折算一次；first-use 保持原口径（round10 扩展为按边两端的专家角色判定） | expert_work/expert_comm/shared_reduce、dense_sim 与显式/旧式等价性逐项覆盖 | genesim/src/scheduler/gene_sim_scheduler.py、genesim/tests/sim/test_pipeline_execution.py | `test_legacy_dependency_payload_is_scaled_by_target_role` |
| 问题1 | round05 | 严重 | 成立：2 层/4 层真实编译探针都只有第 0 层有 gate 命令，第 1 层起为 0 | `_expert_provenance` 没有层域：上一层累加的多专家状态与下一层专家权重相遇后整层被判多专家；round01 问题8 的修复只验证了 dispatch 个数与总比例 | 已修复 | provenance 值改为 (层, 专家)/(层, None)，层号取 `nn_module_stack`/权重名并回退输入；跨层输入按中性，层内多专家汇合在累加出口终止；后端按 dispatch 记录放行/拦下；基准报告逐层覆盖 | decode/prefill、强制分块、PP 跨段、EP 四条路径逐层断言；`_expert_provenance` 的两个消费者（执行计划、放置导出的通信 expert_id）同源 | runtime/exec_plan_gen.py、backend/hal_numpy.py、scripts/bench_moe_dispatch.py、tests/test_moe_dispatch.py、tests/test_moe_ep.py | `test_every_layer_gates_its_own_experts`、`test_moe_tail_is_host_shared_work`、`test_pp_stages_each_gate_their_own_layers`、`test_backend_records_per_dispatch_execution`、`test_plan_and_genesim_ir_agree_on_moe_roles`、`test_ep_16_dpu_generate_matches_hf` |
| 问题2 | round05 | 严重 | 部分成立：设备侧 `where`（2048 条/步）确实被并入 host `MOE_ACCUM` 并排除在 pimir 覆盖外；但“路由加权 mul 是设备 launch”不成立——真实 4L 计划里加权乘与累加已被逐元素融合折成 126 个/层 host 闭包，`aten.mul` 的设备 launch 都是专家内乘与注意力乘 | PIM 原语集没有逐元素 select，`ne`/`where` 未被 HOST_ONLY 接住；GeneSim 的选择+加权累加与真实计划落点不一致，成本也没计选择 | 已修复 | `ne.Scalar`/`where.ScalarOther` 归 host，选择与加权累加同为主机共享工作；设备侧只留可门控专家子图；MOE_ACCUM 成本计入选择并区分首专家（首专家 2×hidden、其余 3×hidden），成本版本 1.4 强制重精化；设计 §3.1/§十.3/§十.6 按实测纠正 | dense/dynamic、首/中/末专家、TP/PP 逐项对拍：IR 的 expert_work 集合必须等于计划里带 gate 的 (层,专家) 集合；host 选择节点必须是 host 命令 | graph/partition.py、graph/spec_prop.py、runtime/kernels.py、runtime/exec_plan_gen.py、genesim/src/ir/model_ir.py、genesim_bridge/op_classify.py、genesim_bridge/cost_extractor.py、docs/design-qwen3-moe-20261105.md | `test_moe_tail_is_host_shared_work`、`test_plan_and_genesim_ir_agree_on_moe_roles`、genesim `test_host_tail_covers_select_and_weighted_accumulation` |
| 问题3 | round05 | 高 | 成立：真实 4L 未融合的 op156 加入 `absorbed_ops` 后覆盖审计仍通过；`BRAND_NEW_OP` 伪标 absorbed 后两侧都接受；直接 GeneSim 接受被篡改的专家号 | 两侧把 sidecar 自报的“已吸收”当成无需证明的事实，没有与融合清单、主算子 pimir、IR 依赖边交叉核对；直接 GeneSim 缺少主链的专家三元组校验 | 已修复 | 两侧新增 absorbed 结构契约（只允许激活、唯一生产者是带 pimir 的主算子 GEMM、三元组在融合清单里、编号在源 IR 里）；GeneSim 新增 (层,角色,专家) 逐项核对；trace 验收也走 absorbed 校验 | 编译器导出/skip-simulation/仿真前复核/trace 验收/直接 GeneSim 共用同一规则；真实 512 条 SILU 融合正例通过 | scripts/run_full_pipeline.py、genesim/src/scheduler/gene_sim_scheduler.py、tests/test_genesim_simulation.py、genesim/tests/sim/test_compiler_placement.py | compiler 5 条、genesim 5 条 absorbed/专家身份用例 |
| 问题4 | round05 | 高 | 成立：真实 4L sidecar 删掉 `hw_budget`、或写 `sys_reserve=0/align=1`，严格加载与完整覆盖校验都通过；只有 16 GiB 总容量被拒绝 | 容量校验被实现成可选的单字段提示，而不是严格产物的必需 schema；导出与覆盖审计没有容量字段检查 | 已修复 | 三件套不可拆分：严格产物缺字段、类型错误或任一值偏离即失败；导出写出前校验、覆盖审计（含 skip-simulation）、trace 验收与直接 GeneSim 用同一规则；旧非严格产物保留兼容 | 三个字段分别做缺失/错值反例；两仓常量逐值相等由跨仓测试钉住 | scripts/run_full_pipeline.py、genesim_bridge/placement_export.py、genesim/src/pim/capacity.py、genesim/src/scheduler/gene_sim_scheduler.py | compiler 7 条、genesim 7 条容量用例、`test_sidecar_carries_the_single_capacity_contract`（含跨仓常量） |
| 问题5 | round05 | 中 | 成立：§1.3 写“全部完成”，同段承认 48 层未重跑、K7 未关闭 | 完成汇总不是由未关闭验收项生成，历轮只更新切片数字却保留超前总标题 | 已修复 | §1.3 拆成“代码修复完成”与“需求验收部分完成”，明确 K7 未关闭；48 层历史结果、本轮 4 层结果、环境阻塞分栏登记 | 全文按多轮更新规则就地收敛，不新增轮次块 | docs/implement-qwen3-moe-20261105.md | 本文 §1.3/§4/§5/§6/§7 |
| 问题1 | round06 | 严重 | 成立：选中 `[0,1]`、权重 `[1,0]` 时 HF 产生 NaN、静态前向全 0，dispatch 跳过专家 1 | round01 的溢出修复用 `routing != 0` 兼任命中条件，未区分选中索引与权重值；round06 未收敛，round07 复现同一根因 | 已修复（round07 根治） | 静态前向新增布尔 `selected_mask` scatter，掩蔽与命中只用它；dispatch 只读布尔掩码；`scatter.value` 列入 HOST_ONLY | prefill/decode、两态归一化、选中/未选中 inf/nan 全覆盖 | backend/architectures.py、runtime/exec_plan_gen.py、graph/partition.py、design §3.1/§4.1/§十.6 | `test_static_moe_forward_zero_weight_selected_expert_matches_hf`（2 态）、`test_decode_and_prefill_dispatch_read_the_selected_mask` |
| 问题2 | round06 | 高 | 成立：sidecar 自报 align=64，1557 条 GEMM 为 64，1606 条 B 路全部为 8；vLLM 用 256 | `DEFAULT_HARDWARE_CONFIG.dma_align=8` 与 `DPU_DMA_ALIGN=64` 分裂；B 路不接收本次硬件配置，`with_budget` 保留已有属性 | 已修复（round07 补 A/B 两路与产物核验） | `default_site_capacity()` 单一工厂；B 路显式传硬件；冲突属性拒绝；导出与 GeneSim 逐份核验 pimir 的 dma-align/mram-bytes | 默认参数、vLLM、A/B 两路、直接 GeneSim | contracts/op_contract.py、memory/mem_planner.py、genesim_bridge/placement_export.py、opcompiler_bridge/*、backend/vllm_worker.py、genesim scheduler | `test_default_hardware_and_budget_share_one_alignment`、`test_with_budget_rejects_conflicting_module_attributes`、`test_pimir_hardware_attributes_must_match_the_contract`；4L 产物 1557/1606 份均为 64 |
| 问题3 | round06 | 高 | 成立：同层 op152/op158 专家号互换后覆盖审计、freshness、仿真前复核、trace 验收全部接受 | 编译器把三元组收成集合再求差，丢掉与 op_id 的对应；round06 未收敛 | 已修复（round07 补权重身份与 trace 验收） | 按源 IR 每个 op_id 核对角色/专家号/权重路径；GeneSim 同规则；trace 验收也校验豁免 | 同层互换、跨层互换、权重路径不符、合法 128 专家正例 | scripts/run_full_pipeline.py、genesim scheduler | compiler 4 条身份用例、genesim 1 条；4L 真实 sidecar 互换探针 REJECTED |
| 问题4 | round06 | 高 | 成立：48 层数值与 BCD 已通过，GeneSim 主链未形成 summary，实施文档仍写旧状态 | 长链路验证后置；证据矩阵未随现有任务更新 | 代码与文档已修复；主链终态待 Q8 | 保留 48 层数值/BCD 证据；round06 五项确认状态回填 §7；主链单元继续运行，不重复启动 | 48 层与 4 层、历史与本轮分栏 | docs/implement-qwen3-moe-20261105.md、round06 日志 | 本文 §4/§5/§6/§7 |
| 问题5 | round06 | 中 | 成立：1 专家/top-1 的 HF 前向正常，parser 以空执行语义拒收 | 图构造、执行语义与默认分块都用 `num_experts > 1` 当阈值 | 已修复（round07 补策略约束与全链用例） | `is_moe_config` 统一判定；专家数正整数、top-k 范围校验；`_moe_strategy` 要求 TP 整除专家数 | 1 专家生成、parser、IR、sidecar 与 dense 行为不变 | runtime/exec_plan_gen.py、runtime/compile.py、backend/architectures.py、genesim/src/ir/model_ir.py | `test_single_expert_moe_generate_matches_hf`、`test_single_expert_is_still_moe`、`test_cli_accepts_a_single_expert_moe`、`test_single_expert_moe_sidecar_config_is_accepted` |
| 问题1 | round07 | 严重 | 成立：同 round06 问题1，当前代码仍用非零路由权重判命中 | round01 的混淆条件在 round06 修复中未真正引入独立掩码，属同一根因复发 | 已修复 | 同 round06 问题1；新增 scatter.value 绑定与 prefill/decode 图级断言 | 设计 §4.1/§十.6 与 GeneSim 命中口径同步 | 同 round06 问题1 | 同 round06 问题1 + `test_static_moe_forward_masks_unselected_overflow_both_norms` |
| 问题2 | round07 | 高 | 成立：删除 router op150 的 pimir 并写豁免后，覆盖、freshness、仿真前、trace 与直接 GeneSim 全部接受 | `expert_id<0` 被当成「非 DPU GEMM」；导出未保存真实落点证据 | 已修复 | 导出按 `spec.device` 写 IR/sidecar 落点证据；`_build_pimir_exempt_ops` 只复制证据；五处入口逐条核对；设备 GEMM 缺 pimir 直接失败 | router/q_proj/lm_head 删除 pimir 均拒绝；合法 host GEMM/边界仍放行 | placement_export.py、run_full_pipeline.py、genesim scheduler | compiler 6 条、genesim 3 条；4L router 豁免探针 REJECTED |
| 问题3 | round07 | 高 | 成立：同 round06 问题2，当前 4L 产物 B 路仍为 8 | round05 只统一自报字段；round06 修复未覆盖 B 路生成 | 已修复 | 同 round06 问题2；新增 4L 真实产物核对 1557 条 GEMM=64、1606 条 B 路=64 | 默认硬件、pim_options、vLLM、A/B、成本精化 | 同 round06 问题2 | 同 round06 问题2 + 4L 产物统计 |
| 问题4 | round07 | 高 | 成立：同 round06 问题3，真实 sidecar 专家互换仍被接受 | round05 只修 GeneSim 单点；round06 修复未覆盖权重身份与 trace | 已修复 | 同 round06 问题3；新增 trace 验收与跨层/权重路径反例 | 五处入口共用 | 同 round06 问题3 | 同 round06 问题3 |
| 问题5 | round07 | 高 | 成立：4 层 2581 算子展开 99 秒；48 层 30961 算子展开 5:14:33；微基准平方增长 | `add_operators`/`remove_operators` 每次全量 `_build_index`，TP 展开逐组调用 | 已修复 | `ModelIR.batch_edit()` 批量增删、一次重建；TP 展开与 relay 插入改批量；`_pinned_op_on_dpu` 预计算。4 层展开 ≤1 秒、48 层 6.24 秒 | 4 层 trace 元数据与 total_time 与修复前逐项相等；批量/逐条等价用例；48 层展开 130036 分片/143192 算子 | genesim/src/ir/model_ir.py、genesim scheduler | `test_batch_edit_*` 4 条、`test_expansion_rebuilds_the_index_once`、`test_batch_expansion_matches_sequential_expansion` |
| 问题6 | round07 | 高 | 成立：`full30b-pipeline-r6-qwen3` 在报告时仍在 GeneSim 阶段；§1.3/K7/K8/Q8/§8.4 仍写旧状态 | 证据矩阵未随 round06 数值/BCD 证据与主链进度收敛 | 代码与文档已修复；主链终态待 Q8 | 回填 round06 数值（26 passed/1 skipped）与 BCD 证据；48 层/4 层、历史/本轮分栏；不重复启动主链 | 48 层数值、BCD、主链三栏登记 | docs/implement-qwen3-moe-20261105.md | 本文 §1.3/§4/§5/§6/§7 |
| 问题7 | round07 | 中 | 成立：同 round06 问题5，单专家仍被拒收 | 同一 `>1` 阈值散布在图构造、执行语义与默认分块 | 已修复 | 同 round06 问题5 | 单专家全链 | 同 round06 问题5 | 同 round06 问题5 |
| 问题1 | round08 | 严重 | 成立：tp1_pp4 的 BCD 产物已生成，E 仍读静态配置里的 tp4_pp2；op152 本地输出 192、展开 2581 算子、总时长与旧策略逐位相同；结果复制到固定 `/tmp` 并先删后建 | 主链只在 BCD 用策略名，E 从静态配置读另一份策略产物，没有统一运行身份；结构指纹按设计排除放置字段 | 已修复 | BCD placement 作为 E 唯一输入；`prepare_simulation_config` 改写并回读；`_assert_placement_matches_run` 核对策略/DPU/段数/Cluster/分片数与本地形状/层段归属；`run_directory` 按架构/策略隔离 | 同一规则核对 Llama2 与 Qwen3 真实 sidecar；两种策略各跑一次 | scripts/run_full_pipeline.py、genesim/conf/sim_qwen3_moe_pimir.yaml、tests/test_genesim_simulation.py | `test_placement_check_*` 4 条、`test_prepare_simulation_config_binds_the_run_placement`、`test_run_directory_is_derived_from_artifact_and_strategy`、`test_simulation_step_consumes_the_current_placement`、`test_simulation_step_rejects_a_sidecar_from_another_strategy`、4L tp4_pp2/tp1_pp4 实跑 |
| 问题2 | round08 | 高 | 成立：round06 的 48L sidecar 6192 条豁免缺 `device_hint`、19250 条 B 路 pim mlir 为 8 字节；主链在 06:36 被外部停止，无 summary/最终 trace | 旧产物生成于 round07 收紧之前；长任务终态未随真实状态收敛 | 已修复 | 当前源码用 systemd `full30b-pipeline-r8-qwen3` 重跑 48L A→0→BCD→A2→E：BCD 18673 GEMM + 19250 B 路、23 份 pim mlir 全 64、豁免 6194 条带落点证据；容量 dpu0..3 8,258,576,576/8,522,825,728 B、dpu4..7 7,791,833,024/8,522,825,728 B；E trace `{'pimir': 106230, 'absorbed': 24576}`、total_time_s=566.863、tokens/s=1.402；`report_capacity` 记录每 DPU 权重/KV/激活/内核所需与可用字节 | 4L 全链回归 | scripts/run_full_pipeline.py、scripts/export_pp_placement.py、docs/implement-qwen3-moe-20261105.md | 48L 主链日志 `/tmp/qwen3-review-round08/pipeline48l-r8.log`、`bcd_export.log`、`report_capacity` 输出、`full30b-pipeline-r8-qwen3`（exit 0） |
| 问题3 | round08 | 中 | 成立：单专家四层配置自动选择 tp1_pp4，但 `strategy_from_partition_plan(plan(4,2))` 与 `llama_strategy(--num-stages 2)` 都得到 tp2_pp2，`num_experts=1` 不能被 TP=2 整除 | 架构注册表只传宽度字典，专家整除约束只存在于自动选择入口 | 已修复 | `Architecture.strategy_check`/`_moe_strategy_check`；`export_pp_placement.resolve_strategy` 两条入口统一调用；非法方案在加载后、图编译前失败 | 稠密模型无专家约束、行为不变；真实主链非法分支已跑 | backend/architectures.py、scripts/export_pp_placement.py、tests/test_architectures.py | `test_arch_strategy_constraints_cover_all_entry_points`、单专家生成/parser/sidecar 用例、真实主链 `--num-dpus 4 --num-stages 2` 退出失败并报 num_experts |
| 问题4 | round08 | 中 | 成立：删除 `moe_config` 后 audit/freshness/仿真 IR/trace 全接受；删 `hw_budget` 或伪造 router host 豁免 freshness 也接受；只有直接 GeneSim 拒绝 | 各复核函数各自实现子集，没有共享完整验证器；audit 不比较 `moe_config` | 已修复 | 新增 `_validate_sidecar_identity` + `validate_sidecar_contract`；五个入口共同调用；`audit_sidecar_coverage` 作为覆盖组件；直接 GeneSim 保留独立复核 | 每个入口分别覆盖缺 MoE 元数据/容量/伪造 host/非法吸收/专家互换/指纹；合法 host/边界/融合/旧非严格产物正例 | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | `test_full_contract_*` 9 条、`test_freshness_*` 5 条、`test_sim_ir_check_uses_the_full_contract`、`test_trace_provenance_uses_the_full_contract`、`test_export_step_uses_the_full_contract` |
| 问题5 | round08 | 中 | 成立：同一 KV 几何经 `build_kv_layout` 需要 4224 B，小于 4300 B 可用预算，预检却因 256 字节对齐算得更大而拒绝；示例仍给 4 GiB | 预检持有独立 `ALIGN=256` 与独立估算公式，未复用真实布局 | 已修复 | 删除 `ALIGN`，`_kv_bytes_per_dpu` 直接调用 `build_kv_layout` 并取契约对齐；示例去掉 4 GiB；补边界正反例 | 其它启动入口（`vllm_worker`、`profile_execution`）已走同一工厂；`verify_cpu_only.sh` 的 4 GiB 是历史产物逐字节复现输入，不是容量入口 | scripts/serve_pim.py、tests/test_serve_pim_preflight.py | `test_kv_estimate_reuses_the_real_layout`、`test_preflight_accepts_a_layout_that_fits_under_the_contract_alignment`、`test_preflight_rejects_one_byte_over_the_budget`、`test_serve_pim_has_no_independent_alignment_or_old_budget` |
| 问题1 | round09 | 高 | 成立：复跑探针，主 GEMM 同时连 SILU 与 VECTOR_ADD、`absorbed_ops=[SILU]` 时编译器 validator/trace 与直接 GeneSim 全部 ACCEPTED；真实 4L 精化产物删掉 512 个 `absorbed_ops` 后两侧仍 ACCEPTED | `_validate_absorbed_ops` 只建生产者索引，把「唯一生产者」当成「主算子唯一消费者」；同一布尔校验无阶段参数，BCD 与精化后共用；GeneSim 的 absorbed 契约用例因 I57 从未在标准入口执行 | 已修复 | 两仓建立消费者索引，要求主算子只有被吸收激活一个真实消费者；共享校验新增 `require_absorbed_closure`：BCD 导出显式放宽，A2 后、skip-simulation、freshness、仿真 IR、trace 与直接 GeneSim 要求清单/吸收编号双向闭合；补多消费者、阶段反例与 BCD 正例 | 主算子多消费者、精化后删吸收编号、BCD 合法中间态、真实 4L 512/48L 6144/Llama2 32 条融合正例；五入口 + skip-simulation + 直接 GeneSim 逐项复核 | scripts/run_full_pipeline.py（`_validate_absorbed_ops`、`audit_sidecar_coverage`、`validate_sidecar_contract`、`step_bcd_export`）、genesim/src/scheduler/gene_sim_scheduler.py（`_validate_absorbed_ops`）、tests/test_genesim_simulation.py、genesim/tests/sim/test_compiler_placement.py | `test_audit_rejects_absorbing_an_activation_with_another_consumer`、`test_audit_accepts_a_manifest_without_absorbed_ids_during_export`、`test_audit_rejects_a_refined_sidecar_without_absorbed_ids`、`test_post_refine_entries_require_absorbed_closure`（3）、`test_skip_simulation_requires_the_absorbed_ids`、`test_export_step_validates_the_bcd_phase`、`test_absorbing_an_activation_with_another_consumer_is_rejected`、`test_manifest_without_absorbed_ids_is_rejected`、真实 4L/48L 产物回归 |
| 问题2 | round09 | 中 | 成立：真实 4L host `MOE_ROUTE` op151 保留合法 IR/豁免证据，仅加 PIM 分片、`device_hint=pim` 与有效 pimir 条目后，validator/freshness/仿真 IR/trace 全部 ACCEPTED，直接 GeneSim REJECTED | round07 的「豁免与放置矛盾」只落在 GeneSim；round08 收拢共享验证器时没有移植该不变式，`_invalid_exemptions` 只比较 IR 与豁免条目；GeneSim 覆盖循环在查豁免前先跳过模型边界 | 已修复 | 两仓共用同一规则：`_placement_contradicts_exemption`（PIM 分片、`pimir_path` 或 `device_hint=pim`）+ `_host_gemm_evidence_matches`（`host_gemms` 的 device_hint/角色/权重身份，豁免条目权重不得伪造）；编译器 `_invalid_exemptions` 传入放置与 host 证据，五入口自动继承；GeneSim 同时登记 `host_gemms` 并让边界豁免进入校验 | host 非 GEMM、host GEMM、模型边界三类逐条覆盖；证据缺失/角色不符/权重伪造/仅 pimir 落点四种反例；真实 4L/48L host 豁免正例 | scripts/run_full_pipeline.py（`_invalid_exemptions`、`_exemption_is_legitimate`）、genesim/src/scheduler/gene_sim_scheduler.py（`_exemption_is_legitimate`、`_register_compiler_artifacts`、`_validate_pimir_source_coverage`）、tests/test_genesim_simulation.py、genesim/tests/sim/test_compiler_placement.py | `test_audit_rejects_an_exempt_op_with_a_pim_placement_entry`、`test_audit_rejects_a_boundary_with_a_pim_placement_entry`、`test_audit_rejects_a_host_gemm_exemption_without_evidence`、`test_audit_rejects_a_host_gemm_exemption_with_a_mismatched_role`、`test_audit_rejects_a_host_gemm_exemption_with_a_forged_weight`、`test_post_refine_entries_reject_exemption_placement_contradictions`（3）、`test_skip_simulation_rejects_exemption_placement_contradictions`、`TestHostGemmExemptionEvidence`（6）、`test_boundary_exemption_with_a_pim_placement_is_rejected` |
| 问题1 | round10 | 严重 | 成立：复跑探针，`dynamic_dispatch` 下 `expert_work → shared_reduce` 的 16 B 专家输出边实计 16 B（按 E=8/top-1 命中并集应为 2 B）；dense 同类边 16 B、shared→shared 16 B、普通源→专家 2 B，说明不能一刀切；真实 4/48 层 IR 分别有 512/6144 条专家输出边，每 decode 多计 1,966,080 / 23,592,960 B | round03 引入成本角色时把「算子是否被门控」与「边传输张量由谁生产」混为一体，`_moe_dependency_fraction` 只读目标角色；round04 统一出口时保留了方向假设；既有测试只覆盖「普通源→专家目标」 | 已修复 | 依赖边同时接收源、目的算子：专家生产的输出无论流向共享累加、其它设备还是 HOST，都随源专家的命中并集；共享/HOST 输入到专家随消费者命中；shared→shared 全量。放置期估算与正式管线、显式与旧式依赖共用同一出口；复跑同一探针为 2 B | 四方向 decode/prefill、TP 展开后专家分片→共享累加、专家→HOST、未命中 host 收集不搬运；真实 4L `tp1_pp4` 通信字节减少 133,693,440 = 68 decode × 1,966,080 B；`tp4_pp2` 因归约中转落在 host 同一 VPU 而不进 `dependency_transfer_bytes`，总时长不变（见 §7.2） | genesim/src/scheduler/gene_sim_scheduler.py（`_moe_dependency_fraction`、`dependency_transfer_stats`、`dependency_transfer_bytes`）、genesim/tests/sim/test_pipeline_execution.py、test_moe_execution_cost.py、test_compiler_placement.py、flagos-pim-compiler/tests/test_moe_dispatch.py | `test_dependency_payload_follows_the_expert_endpoint`、`test_expert_host_egress_is_scaled_by_the_hit_union`、`TestMoeDependencyDirection` 5 条、`test_expanded_expert_output_edges_keep_the_source_role`、`test_gate_skipped_host_gather_does_not_transfer`；探针 `/tmp/round10-fix/probe-moe-edge.log` |
| 问题2 | round10 | 中 | 成立：当前文档 dynamic 有效 FLOPs 写 `5.414125498987654e12`，两份真实产物均为 `5,415,241,254,987.654`，少 1,115,756,000；依赖该值的 round03 差值 1.66e10 与 §9.2 的 1.63e10 同样失真；dense、时间、trace、容量字段无同类偏差 | 实施证据由人工誊写，未与引用的审计 JSON 做机器一致性校验，错误值被复制到 §4.1 与 §8.4（两处），派生差值出现在 §8.4 与 §9.2 | 已修复 | 以 summary 为唯一来源回填全部引用位置；新增 `scripts/check_implement_evidence.py`：从 summary 提取时间/吞吐/有效 FLOPs 核对文档，并扫描文档里的千分位高精度数字，必须属于本次产物或 `--history` 显式声明的历史值；dynamic 与 round03 的差值重算为 1.55e10 | 4L dynamic/dense/alt 三份 summary 与 48L 时间字段全部机器核对；历史 round03 值单独声明，避免把历史证据当成本轮产物 | docs/implement-qwen3-moe-20261105.md、scripts/check_implement_evidence.py、tests/test_implement_evidence.py | `test_summary_metrics_are_extracted`、`test_document_with_current_values_passes`、`test_mistranscribed_flops_value_is_rejected`、`test_declared_history_values_are_allowed`、`test_time_only_metrics_do_not_require_flops`；checker 对当前文档退出码 0 |
| 问题1 | round11 | 高 | 成立：复跑探针，配置与 sidecar 同时为 16 GiB 时直接 GeneSim ACCEPTED；4 GiB 同理；MLIR 属性核验也从可配置容量反推 | 容量验收标准来自 `resolve_pim_capacity_limits()`，与运行配置同源；配置与产物同步改大后自洽，需求常量没有独立出现在严格路径 | 已修复 | `pim/capacity.py` 增 `DPU_MRAM_BYTES=8 GiB`；`_validate_hw_budget` 严格分支同时核对有效配置与 sidecar；`_validate_pimir_hardware_contract` 固定对常量；仅旧式非严格产物按运行配置兼容 | 同类路径：sidecar 三字段、每份 pimir MLIR、直接 GeneSim 与主链导出；跨仓常量由 `test_placement_export.py` 与 GeneSim 用例逐值钉住 | genesim/src/pim/capacity.py、genesim/src/scheduler/gene_sim_scheduler.py、genesim/tests/sim/test_compiler_placement.py、flagos-pim-compiler/tests/test_placement_export.py | `test_strict_product_rejects_a_raised_runtime_capacity`、`test_strict_product_rejects_a_config_raised_on_one_side`、`test_pimir_attributes_are_checked_against_the_requirement_constant`、`test_legacy_product_keeps_other_capacity_configurations`、`test_contract_matches_the_compiler_values` |
| 问题2 | round11 | 高 | 成立：`step_e_simulate` 删除共享 `genesim/pim_traces` 并读取共享 `genesim/results/summary.json`；A/A2/BCD 固定文件名；同一日志目录两次 E 的临时配置同名 | round08 的修复只把本次 sidecar 和日志副本命名绑定到策略，没有建立覆盖 A/A2/BCD/E 的运行身份；为保证重编仍清空全局目录 | 已修复 | 运行目录按架构/策略/执行语义/唯一运行号派生；A/0/A2/BCD 写本次 `artifacts/`；E 临时配置按 label 命名并绑定 `model.ir_path`、`output.pim_traces_dir/results_dir/log_file`，回读复核；trace 目录必须是本次空目录，不再删除共享目录或复制共享 results | 同类路径：A/A2/BCD 产物、A/B 第二次 E 的配置/trace/results、trace 验收读取目录；预置哨兵与交错执行用例守护 | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | `test_simulation_step_never_deletes_shared_outputs`、`test_interleaved_runs_keep_their_own_traces_and_summaries`、`test_prepare_simulation_config_binds_the_run_placement`、`test_run_directory_is_derived_from_artifact_and_strategy`、`test_trace_provenance_reads_the_configured_trace_dir` |
| 问题3 | round11 | 中 | 成立：复跑探针，dynamic/dense 时间、吞吐与 FLOPs 全部对调并写入 `difference=9.99e99`，checker 仍返回 `missing=[]、unknown=[]`；4 层可冒充 48 层、历史值可冒充 current | 证据检查以整篇文本的数值集合为模型，不绑定运行标签与 summary，不重算派生结论；高精度扫描只覆盖千分位格式 | 已修复 | 实施文档嵌入机器可读证据块；checker 逐 current 标签加载同名 summary，核对层数/模型/执行语义并逐值比较；派生 subtract/ratio/greater_than 重算；history/measurements 独立命名空间；正文扫描补科学计数法、比值与时间格式 | 同类路径：4L/48L 分栏、history/current、动态/静态语义、差值/比值/大小结论；合法历史值与 bench 测量用独立命名空间放行 | scripts/check_implement_evidence.py、docs/implement-qwen3-moe-20261105.md、tests/test_implement_evidence.py | `test_swapped_dynamic_and_dense_values_are_rejected`、`test_four_layer_summary_cannot_pose_as_forty_eight`、`test_history_value_cannot_be_declared_as_current`、`test_current_label_without_a_summary_is_rejected`、`test_wrong_derived_difference_is_rejected`、`test_wrong_derived_ratio_is_rejected`、`test_wrong_derived_conclusion_is_rejected`、`test_scientific_notation_outside_the_evidence_block_is_rejected` |
| 问题1 | round12 | 高 | 成立：复跑探针，配置显式 `require_compiler_pimir=true` 时，删除 sidecar 的 `requires_pimir` 与 `hw_budget`，或删除自报后写 4/16 GiB，编译器 freshness、仿真 IR、trace 与共享审计均 ACCEPTED；直接 GeneSim 拒绝 | round02「配置严格开关与 sidecar 自报分裂」在编译器复核层未收敛：`validate_sidecar_contract` 没有严格性入参，`audit_sidecar_coverage` 只在 sidecar 自报为真时核对容量；三个调用方读了配置却不把状态传给验证器 | 已修复 | 共享验证器新增 `strict_pimir` 入参，与 sidecar 自报取并集；导出与 skip-simulation 显式要求严格，freshness/仿真 IR/trace 从配置解析 `require_compiler_pimir`；严格时无论自报是否存在都按 8 GiB 契约核对总容量/系统预留/对齐；只有配置与产物都明确非严格才走旧式兼容 | 同类路径：五个编译器入口、共享审计组件、直接 GeneSim（round11 已收敛）、4/16 GiB、缺三字段、自报缺失；真实 4L 三入口 ACCEPTED、48L 三入口 ACCEPTED | scripts/run_full_pipeline.py（`validate_sidecar_contract`、`audit_sidecar_coverage`、`_config_requires_pimir`、`_validate_sidecar_hw_budget` 及五个调用点）、tests/test_genesim_simulation.py | `test_strict_config_requires_the_capacity_contract`（3）、`test_strict_config_rejects_a_sidecar_with_a_wrong_capacity`（4/16 GiB × 3）、`test_audit_honours_an_explicit_strict_context`、`test_audit_keeps_self_reported_strictness_when_the_caller_is_not_strict`、`test_non_strict_config_keeps_the_legacy_compatibility_branch`、`test_export_step_requires_the_capacity_contract`、`test_skip_simulation_requires_the_capacity_contract`；探针 `/tmp/qwen3-review-round12-fix/probe-strict-capacity.log` |
| 问题2 | round12 | 中 | 成立：复跑探针，只交换正文 dynamic/dense 的时间、吞吐与 FLOPs、证据块不变，checker 返回 `problems=[]、unknown=[]`；4 层冒充 48 层、历史冒充 current、正文错误比较结论同样放行 | round11 只把证据块内部做成结构化，正文仍用全篇 `allowed` 集合；必需值检查搜索整个 `doc_text`（含证据块），没有「运行标签 → summary → 指标 → 正文锚点」绑定，也不核对大小结论 | 已修复 | 证据块运行条目增 `prose_aliases`；正文数值按所在块（表格行/列表项/段落）的前置标签绑定，块内前置标签未声明过该值即失败；必需值只扫去掉证据块的正文；带数值的「高于/低于」结论按两侧实际值判定；history/measurements 保持独立命名空间 | 同类路径：只换正文、4 层冒充 48 层、历史冒充 current、错误比较、正文缺失必需值、删除证据块、合法分栏与 measurements；当前文档 + dynamic/dense 两份 current summary 通过（alt/48l 为 history 引用） | scripts/check_implement_evidence.py（`_run_aliases`、`_label_spans`、`_token_context`、`_block_start`、`_prose_context_problems`、`_check_comparisons`）、docs/implement-qwen3-moe-20261105.md、tests/test_implement_evidence.py | `test_prose_only_swap_of_dynamic_and_dense_is_rejected`、`test_prose_cannot_borrow_a_forty_eight_layer_value`、`test_prose_history_value_cannot_pose_as_current`、`test_prose_comparison_conclusion_is_checked`、`test_required_values_must_appear_in_the_prose`、`test_measurements_are_allowed_in_a_run_context`；探针 `/tmp/qwen3-review-round12-fix/probe-evidence-prose.log` |
| 问题1 | round13 | 严重 | 成立：当前 lowering 的 PV 循环在 f32 里累加 `dot` 后直接 `acc*alpha+dot`；合法块表探针与“不取整”参考逐元素相同，与契约参考有 16 个元素不同、最大差 0.00048828125 | 相位存储边界没有统一表驱动，PV 手写循环漏掉 `roundF16`；`test_opcompiler_ops` 的参考与实现共享同一未取整假设 | 已修复（round14 一并根治） | lowering 在 PV 累加前落 f16；补相位边界表；测试参考独立按契约实现并断言数据能区分两种口径；新增多块/尾块用例与 EmitC lit 取整点约束 | 相位表 `bytes` 保持 f32 累加器语义，f16 乘积是块内临时值；主机 `_host_paged_attention` 的 `matmul` 本来就落 f16，修复后两侧边界一致 | FlagTree/LowerPIMToEmitC.cpp、FlagTree/ExpandPhases.cpp、runtime/kernels.py、tests/test_opcompiler_ops.py、tests/test_kernels.py | `test_paged_attn_matches_the_online_reference`、`test_paged_attn_multi_block_tail_matches_the_online_reference`、`paged_attn_emitc.mlir` |
| 问题2 | round13 | 严重 | 成立：设备侧只判 `slot < poolSlots`，`-1` 生成负偏移；上界非法项被静默换成槽 0；主机兜底不查上界 | 块表有效性没有共享契约，写 KV、融合内核、主机兜底三路径各写一套单侧判断 | 已修复（round14 一并根治） | 抽出 `_physical_slot`/`_validate_block_table`，主机启动前按 `0 <= slot < pool_slots` 逐项预检；设备 `pim_block_slot` 中止非法项；写 KV 与主机兜底共用 | 槽 0、最后合法槽、负值、上界、表尾哨兵、多块尾块全覆盖；K/V 两个基址共用同一槽值 | runtime/kernels.py、FlagTree/LowerPIMToEmitC.cpp、FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir、tests/test_kernels.py | `test_physical_slot_accepts_both_legal_bounds`、`test_physical_slot_rejects_negative_and_upper_invalid`、`test_block_table_tail_sentinel_is_not_read`、`test_paged_attention_rejects_invalid_slots_before_addressing` |
| 问题3 | round13 | 高 | 成立：配置显式严格时，删 sidecar 键/IR 键/sidecar 文件/IR 文件仍 ACCEPTED；配置本身缺失时三个入口也 ACCEPTED | 兼容分支在读取严格性之前执行；`_config_text` 把缺文件当成无可核对对象 | 已修复（round14 一并根治） | `_config_text` 缺失即失败；先解析严格性（配置 ∪ 自报），严格时配置键与 IR/sidecar 文件为必需输入；三个入口共用同一判定 | 六种缺输入组合 × 三入口；自报严格 + 配置非严格缺 IR 键；非严格无 sidecar 旧式正例保留 | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | `test_strict_entry_requires_config_and_artifacts`（12 条）、`test_strict_entry_rejects_a_missing_config_file`（3 条）、`test_sidecar_self_report_requires_the_ir_key`（3 条）、`test_non_strict_entry_keeps_the_no_sidecar_legacy_path` |
| 问题4 | round13 | 高 | 成立：正确哈希的 trace 改成 64 个零、删哈希或改错 op_type 后仍 ACCEPTED；缺哈希夹具同样通过 | sidecar 文件哈希与 trace 编译签名是两条未连接的链；absorbed 只看来源标签 | 已修复（round14 一并根治） | 逐条核对父算子、op_type 与 `compile_signature.pimir_sha256`；TP 分片回指同一源条目；absorbed 要求零指令且编号在 sidecar 吸收集合 | 错哈希/缺哈希/错类型/错父编号/非空 absorbed/未知吸收编号 6 类反例；真实 4L（10906 条）与 48L（130806 条）trace 三入口 ACCEPTED | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | `test_trace_entry_accepts_hash_bound_shards`、`test_trace_entry_rejects_unbound_pimir_traces`（4 条）、`test_trace_entry_accepts_a_proven_empty_absorbed_trace`、`test_trace_entry_rejects_a_non_empty_absorbed_trace`、`test_trace_entry_rejects_an_absorbed_id_outside_the_sidecar` |
| 问题5 | round13 | 中 | 成立：追加把 dense/dynamic 时间与吞吐对调的中文比较句仍通过；必需指标放无标签前言也通过；容量/通信/trace 计数不在绑定范围 | 归属用块内任一前置标签集合而非最近标签；必需值只做全篇存在；数值模型只覆盖 summary 三项指标 | 已修复（round14 一并根治） | 最近前置标签归属；前者/后者按最近两个不同运行解析并核对两侧身份；必需值要求出现在自己的运行语境；`facts` 声明非 summary 证据值（按运行绑定或独立命名空间） | 同块对调、身份正确正例、无标签前言、facts 运行绑定、未知运行标签；当前文档 + dynamic/dense 两份 current summary 通过（alt/48l 为 history 引用） | scripts/check_implement_evidence.py、tests/test_implement_evidence.py、docs/implement-qwen3-moe-20261105.md | `test_same_block_former_latter_swap_is_rejected`、`test_former_latter_identity_must_match_the_declared_runs`、`test_required_values_in_an_unlabeled_preamble_do_not_count`、`test_fact_values_are_bound_to_their_run`、`test_facts_with_an_unknown_run_are_rejected` |
| 问题1 | round14 | 严重 | 成立：同 round13 问题1；本轮复跑安装内 FlagTree，编译输出与未取整参考逐元素相同，与契约参考差 16 个元素、最大差 0.0001220703125（单块 0.00048828125） | 同 round13 问题1；测试参考与被测 lowering 共享错误假设，容差掩盖边界差 | 已修复 | 同 round13 问题1；`test_sdpa_fused_paged_attn_tracks_the_host_path` 的容差说明改为只吸收 exp 实现与累加顺序差，PV 边界由逐元素用例钉住 | 相位表核对结论：`bytes` 是 f32 累加器，f16 乘积为块内临时值，不改相位计费；主机参考与设备 lowering 共用同一张边界表 | FlagTree/LowerPIMToEmitC.cpp、FlagTree/ExpandPhases.cpp、runtime/kernels.py、tests/test_opcompiler_ops.py、tests/test_kernels.py | 同 round13 问题1 + `test_sdpa_fused_paged_attn_tracks_the_host_path` |
| 问题2 | round14 | 严重 | 成立：主机 `negative=REJECTED`、`zero/last_legal=ACCEPTED`、`upper_invalid=ACCEPTED`；设备对 `-1` 生成负偏移 | 同 round13 问题2；合法集合只做单侧判断，两条执行路径互补错位 | 已修复 | 同 round13 问题2；写 KV、融合内核、主机兜底共用 `_physical_slot`，设备侧由 `pim_block_slot` 兜底 | 同 round13 问题2；主机与设备两侧合法集合一致 | runtime/kernels.py、FlagTree/LowerPIMToEmitC.cpp、tests/test_kernels.py | 同 round13 问题2 |
| 问题3 | round14 | 高 | 成立：配置显式严格时，删 sidecar 配置键后 freshness/仿真 IR/trace 均 ACCEPTED；sidecar 文件缺失时 trace ACCEPTED；删 IR 键三者 ACCEPTED；IR 文件缺失 freshness ACCEPTED；配置文件缺失三者 ACCEPTED | 同 round13 问题3；旧式兼容返回在严格性判定之前，`_config_text` 缺文件返回 `None` | 已修复 | 同 round13 问题3；另把配置文件本身缺失视为错误 | 同 round13 问题3；合法旧产物、纯放置与合法 host/边界正例保留 | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | 同 round13 问题3 |
| 问题4 | round14 | 高 | 成立：正确哈希 trace 接受后，改 64 个零、删哈希或改错 op_type 均 ACCEPTED；缺哈希夹具同样接受 | 同 round13 问题4；没有核对 trace 的 `op_type`、`expert_id` 或 `num_instructions` | 已修复 | 同 round13 问题4；补 `num_instructions` 与 absorbed 集合核对 | 同 round13 问题4；真实 4L/48L 产物正例 | scripts/run_full_pipeline.py、tests/test_genesim_simulation.py | 同 round13 问题4 |
| 问题5 | round14 | 中 | 成立：同块语义对调句、无标签前言、非 summary 证据值三处均可绕过 | 同 round13 问题5；比较器丢失左右操作数身份，证据模型是数值集合而非值→运行→指标→正文位置 | 已修复 | 同 round13 问题5；新增 `facts` 机制覆盖容量/通信/trace 计数 | 同 round13 问题5 | scripts/check_implement_evidence.py、tests/test_implement_evidence.py、docs/implement-qwen3-moe-20261105.md | 同 round13 问题5 |
| 问题6 | round14 | 中 | 成立：§1.3 仍写「round01～round12 完成」「需求验收完成」，§7.1 无 round13 行，§8.3 声称正文按运行标签绑定；`genesim/tests/sim/test_moe_execution_cost.py` 仍用 `routing != 0` 描述命中 | 完成状态是人工写在 §1.3 的结论，不由 §7.1/§5.1 的未关闭项生成；round13 评审后文档没有随代码状态收敛 | 已修复 | 本轮先修代码，再就地更新 §1.3/§3/§4/§5/§7/§8；round13 五项与 round14 六项逐条登记；§8.3 改为实际验收规则；GeneSim 命中口径注释同步 | 同根因的陈旧描述：§1.3 总标题、§7.1 头部关联文档、§8.3 checker 描述、GeneSim 测试注释；全部按当前事实重写 | docs/implement-qwen3-moe-20261105.md、genesim/tests/sim/test_moe_execution_cost.py | 本文 §1.3/§7.1/§8.3；`python scripts/check_implement_evidence.py` 退出码 0 |
| 问题1 | round15 | 高 | 成立：真实 4L sidecar 把某条目 `pimir_path` 改成不存在文件后，freshness 以 `FileNotFoundError` 失败而 sim_ir/trace ACCEPTED；换内容保留旧哈希、删 `pimir_sha256` 后 sim_ir/trace 仍 ACCEPTED | 产物校验是共享验证器的可选参数 `check_artifacts=False`，三个复核入口各自决定不打开；`stale_pimir_entries` 直接 `read_bytes`，缺失文件不在 `StepFailed` 口径内 | 已修复 | 严格上下文（sidecar 自报 ∪ 调用方显式）下逐条核对非豁免/非吸收算子的 pimir_path 可读且哈希匹配，删除 `check_artifacts` 参数；`stale_pimir_entries` 缺路径/读失败转 StepFailed；补三入口 × 五类破坏矩阵与自报严格路径 | 同类路径：导出、skip-simulation、freshness、仿真 IR、trace 五个入口全部经同一验证器；真实 4L 破坏探针 REJECTED、合法产物 ACCEPTED；非严格旧式 sidecar 仍放行 | `scripts/run_full_pipeline.py`、`tests/test_genesim_simulation.py` | `test_strict_entry_requires_a_valid_pimir_artifact`（15）、`test_sidecar_self_report_requires_a_valid_pimir_artifact`（3） |
| 问题2 | round15 | 中 | 成立：不传配置调用 `verify_trace_provenance(root)`，对一条 `pimir` 且编译签名为 64 个零的 trace 返回 ACCEPTED；现有测试还断言无 IR/无 sidecar 的 pimir trace 可以通过 | 严格性被设计成「调用方可选传配置」，函数保留 `config is None` 的旧式返回；测试固化了旧行为 | 已修复 | 配置改为 trace 验收的必需输入，缺失直接 StepFailed；无配置不再有签发旁路；模板/unknown 用例改传最小配置，头部字节用例改测 `_trace_metadata` | 同类路径：离线工具或新入口复用该函数同样被强制；配置 + 完整 lineage 正例保留 | `scripts/run_full_pipeline.py`、`tests/test_placement_export.py`、`tests/test_genesim_simulation.py` | `test_trace_entry_requires_a_configuration`、模板/unknown 用例、真实 4L 配置下 ACCEPTED |
| 问题3 | round15 | 中 | 成立：干净检出（`traces/` 被 gitignore）首次 E 阶段报 `File not found: traces/llama2_7b.trace`；补文件后同一命令 exit 0 | Qwen3 配置沿用 llama 开发环境里生成且被忽略的 trace；主链只改写请求数/IR/sidecar/输出目录，没有生成或校验请求分布输入 | 已修复 | `prepare_simulation_config` 接收 GeneSim 根：配置引用的 trace 缺失时用官方生成器在本次运行目录生成（`--synthetic --seed 0`）并绑定绝对路径、回读复核；已存在则原样使用 | 同类路径：任何复用缺失 trace 的配置、干净检出与 CI；生产 Qwen3 配置隐藏 trace 后生成成功 | `scripts/run_full_pipeline.py`、`tests/test_genesim_simulation.py` | `test_prepare_simulation_config_generates_a_missing_trace`、`test_prepare_simulation_config_keeps_an_existing_trace` |
| 问题4 | round15 | 中 | 成立：§8.4 写「当前代码的 48 层证据由下面的主链提供」，而该主链是 round08 产物、E 是 round10 重跑；当前工作区没有重跑 48 层 | 完成状态与证据归属仍由人工写在文档；round14 收敛了 §1.3/§7.1，§8.4 旧表述残留 | 已修复 | §8.4 改为「证据—源码/产物来源—时间—层数—本轮是否重跑」表；明确当前工作区只重跑 4 层，48 层仅由当前验证器复核；8B 未覆盖与 P1-1 用户确认单列；§1.3/§4.1/§7.3 同步当前状态 | 同类路径：后续轮次不得把旧 run 目录的 summary 当作 current 证据；checker 对两份 current summary（dynamic、dense）重跑通过 | `docs/implement-qwen3-moe-20261105.md` | `check_implement_evidence.py` 退出码 0 |
| 问题5 | round15 | 低 | 成立：两行 MoE 注释挂在 `split.Tensor` 注册项上方；`where_kernel` 的标量广播注释指向 HOST_ONLY 的 `where.ScalarOther`，该分支经 `_KERNELS` 不可达 | 新增注释与镜像分支时没有同步注册表接线，注释与真实调用路径脱节 | 已修复 | 注释移到 `_KERNELS` 后的「故意不在表里」说明段，明确 `where.ScalarOther`/`ne.Scalar` 是 HOST_ONLY；删除不可达的显式 `broadcast_to`（numpy 与 `kernels_pim.mask` 本就广播）；补广播对拍与主机闭包断言 | 同类路径：`_KERNELS` 反向闭合测试 `test_no_kernel_is_registered_for_a_host_only_op` 守护全部主机算子 | `runtime/kernels.py`、`tests/test_kernels.py` | `test_where_kernel_matches_torch_with_a_broadcast_operand`、`test_scalar_where_is_host_only_and_not_a_device_kernel` |
| 问题1 | round16 | 严重 | 成立：4L dynamic 的 `op_10010_VECTOR_MUL.pim_trace` 输入形状是分片 `[Tq,192]`，pimir 字节却是 393216/196608（128 token×768 宽）；`op_type_metrics` VECTOR_MUL 65.244 s、48L 782.930 s；PAGED_ATTN trace `tokens=128` 使 prefill 少算约 5.7×、decode 多算 128× | B 路 pimir 按源 IR 全局形状与 128 token 占位编译，运行期按实际 Tq 与本地宽绑定；sidecar 不带编译单位，TP 展开无条件继承源 pimir，严格验收只核路径/哈希，测试夹具全是 1 token | 已修复 | 定义 B 路编译单位为「1 token × 本地宽」：`Tq` 折 1、历史维取一块 KV 占位；MLP act/mul 取相邻多分片 GEMM 的本地宽；sidecar 回传 `trace_tokens`/`local_*_shapes`，pimir 带 `pim.traceTokens`；GeneSim 生成 trace 前核 tokens/末维/单相字节，TP 展开只继承同形 pimir，运行期按 `ceil(Tq/tokens)` 重复；编译器 trace 验收按同一单位核对，SPLIT 按纯视图核读写相等 | 同类路径：全部 B 路算子（VECTOR_MUL/SILU/RMSNORM/ROPE/视图/PAGED_ATTN）、Llama2 MLP 分片、直接 GeneSim、PAGED_ATTN 的 `tokens>1` 分支；修复中另发现 TRANSPOSE 首维启发式误读 token、SPLIT 代表输出形状两处同类问题并一并修复 | genesim_bridge/placement_export.py、scripts/run_full_pipeline.py、genesim/src/scheduler/gene_sim_scheduler.py、genesim/src/vpu/vpu.py、genesim/src/pim/pimir_trace.py | `test_bpath_uses_one_token_and_the_mlp_local_width`、`test_bpath_view_ops_keep_the_global_width_next_to_tp_gemms`、`TestPimirUnitContract`（6）、`test_bpath_shards_inherit_pimir_only_for_matching_local_shapes`、`test_bpath_shard_shape_conflict_is_rejected_in_strict_mode`、`test_vector_trace_token_unit_scales_with_the_query_rows`、`test_legacy_wide_token_trace_is_not_multiplied_again`、`test_explicit_trace_tokens_attribute_wins_over_shape_heuristic`、`test_trace_lineage_*`（6）；4L 实测 VECTOR_MUL 65.244→0.151 s、4L 的 round16 总时长从 88.483 s 降到 61.015 s |
| 问题2 | round16 | 中 | 成立：把 4L IR 的 `moe_top_k` 改成 7 并重算指纹/回写 sidecar 后，A/BCD/E 全部通过；权重字节与专家 GEMM 数都不受 top-k 影响 | MoE 身份按字段枚举校验，没有从 HF config 派生完整期望元组；sidecar 的 `moe_config` 也只与 IR 自洽 | 已修复 | 新增 `_configured_moe_identity`/`_validate_moe_identity`：A 段与 BCD 段都用 config 的 `num_experts`/`num_experts_per_tok`/`moe_intermediate_size` 逐项核对；非 MoE 配置要求三项全零 | 同类路径：新增 MoE 标量字段走同一份期望元组；A 段无条件执行；BCD 读 `model_dir/config.json` 独立复核 | scripts/run_full_pipeline.py（`step_a_model_ir`、`step_bcd_export`、`_validate_moe_identity`） | `test_moe_identity_rejects_a_top_k_that_disagrees_with_the_config`、`test_moe_identity_rejects_a_wrong_expert_width`、`test_dense_config_rejects_moe_scalars_and_accepts_an_empty_identity` |
| 问题3 | round16 | 低 | 成立：`contracts/dtypes.py` 的 bool 注释仍写「MoE 门控掩码（`routing != 0`）」，与 `backend/architectures.py` 的 `selected_mask` 实现相反 | round14 问题6 的同类路径扫描只覆盖 GeneSim 测试与设计/实施文档，没有扫到 `contracts/` 新增注释 | 已修复 | 注释改为 top-k 选中索引 scatter 出的布尔 `selected_mask`；新增静态测试防回退；代码注释中不再有 `routing != 0` | 同类路径：`grep -rn "routing != 0"` 只余设计里的否定式说明与实施文档历史记录，均为刻意保留 | contracts/dtypes.py、tests/test_architectures.py | `test_bool_dtype_comment_describes_the_selected_mask` |
| 问题4 | round16 | 低 | 成立：实施文档 §1.3/§4.3/§7.3 仍写 2076 passed，round15 新增用例后实际为 2099 passed；本轮同样的全量命令又会因 F65/F66/F67 新增用例继续增长 | 回归计数与时长是人工誊写的，没有随新增用例更新；§7.3 还保留「新增 F54/F55/F56/F57/F58 共 40 条」的旧说明 | 已修复 | 以本轮实际回归输出回填 §1.3/§4.3/§7.3，历史计数只保留一行趋势；新增用例在 §4.2 按任务登记 | 同类路径：§4.1 的 GeneSim/真实权重计数同步本轮重跑结果；48 层历史计数与 4 层本轮计数分栏 | docs/implement-qwen3-moe-20261105.md | 本文 §1.3/§4.3/§7.3；`python -m pytest tests/ -q -k "not llama2_7b"` |

| 问题1 | round17 | 高 | 成立：复跑探针，Tq=7 时 3-D TRANSPOSE 与头优先 SPLIT 重复 224 次（7×32 头）、rank-1 GATHER 重复 1 次；4L `op_1846_TRANSPOSE` 单位 trace 8192 B 只按 1 个 token 计 | trace 的 token 数与运行期实际 Tq 是两套口径，`_runtime_repetitions` 仍按「除末维以外各维相乘」猜查询行数 | 已修复 | 调度器在 `_get_operator_with_runtime` 绑定 `runtime_query_tokens`（IR 带符号 Tq 才随请求增长）；VPU 删除 `_extract_query_count`/GQA 启发式，缺绑定直接报错；GATHER 模板循环宽改取输出隐藏维 | 同类路径：全部 `_VECTOR_RUNTIME_OPS`、GEMM/GEMV 与块循环；PAGED_ATTN prefill/decode 一并覆盖 | `genesim/src/ir/model_ir.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、`genesim/src/vpu/vpu.py`、`genesim/tests/sim/test_vpu.py` | `test_view_embed_and_eltwise_ops_repeat_by_the_bound_tokens`、`test_paged_attn_prefill_and_decode_repeat_by_the_bound_tokens`、`test_unbound_query_tokens_are_rejected_instead_of_guessed`；4L TRANSPOSE 0.1196→0.0037 s、SPLIT 0.1206→0.0048 s，GATHER 由「只按 1 个 token 计」变为按 726 行计（§8.4），round16 的 61.015 s 降到 dynamic 的 60.785 s |
| 问题2 | round17 | 中 | 成立（契约不一致）：单位把 Tq→1、Tp+Tq→128 固化；按绑定形状复核时 MASK prefill `[726,726]` 与 decode `[1,69]` 均 REJECTED，而编译器对同一份单位 ACCEPTED | B 路单位丢失 token/历史/特征维角色；GeneSim 按绑定形状、编译器按符号形状，两个消费方口径不同 | 已修复 | sidecar 单位形状保留 IR 符号维；两仓 `_unit_shape_conflict` 只比较两侧具体特征维，token 维交给重复次数、历史维交给块循环；字节核对按 trace 的 `tokens`/`loop_unit_tokens` 解析符号；严格模式拒绝丢失符号维的旧式单位 | 四节点四类算子 × prefill/decode、错误特征宽、旧式固化单位、TP 分片继承守卫 | `genesim_bridge/placement_export.py`、`scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、两仓测试 | `test_bpath_kv_ops_keep_symbolic_history_dims`、`test_four_node_attention_accepts_prefill_and_decode_history`、`test_four_node_gemv_context_rejects_a_wrong_feature_width`、`test_a_concrete_history_placeholder_loses_the_dimension_role`、`test_trace_lineage_accepts_four_node_attention_units`（4 参数）、`test_trace_lineage_rejects_a_history_dim_folded_into_a_placeholder`、`test_trace_lineage_rejects_a_wrong_feature_width_in_four_node_units`；修正探针 MASK/SOFTMAX prefill/decode 全部 ACCEPTED |
| 问题3 | round17 | 中 | 成立：删除 4L sidecar 1606 条 B 路条目的 4818 个单位字段后，freshness 与仿真 IR 仍 ACCEPTED，只有 trace REJECTED；48L 旧 sidecar 同样只有 trace 拒绝 | 单位证据只接到 trace 消费点，没有进共享验证器；五个入口的严格性不是由一份 schema 生成 | 已修复 | `audit_sidecar_coverage` 严格分支新增 `_bpath_unit_contract_problems`：单位完整性、正整数、符号维角色；导出、skip-simulation、freshness、仿真 IR 与 trace 共用；trace 入口保留 trace-vs-unit 的字节核对 | 同类路径：缺/半缺/丢符号三种反例 × 三入口 + skip-simulation；48L 旧产物三入口全部 REJECTED | `scripts/run_full_pipeline.py`、`tests/test_genesim_simulation.py` | `test_strict_entries_require_a_complete_bpath_unit`（3×2）、`test_strict_entries_reject_units_that_drop_symbolic_dims`（2）、`test_skip_simulation_requires_a_complete_bpath_unit`；4L 新产物三入口 ACCEPTED、剥单位探针三入口 REJECTED（1094 条）、48L 旧产物三入口 REJECTED（13106 条） |
| 问题4 | round17 | 中 | 成立：共享 `models/` 是 round15 的 dense_sim 产物、1606 条 B 路无单位；`./run.sh --config conf/sim_qwen3_moe_pimir.yaml` exit 1（1094 条缺单位），模板注释的单步导出不产出 A2 `_pimir.ir` | 架构注册表模板指向共享旧产物，产物只在 run 目录生成、没有回写，也没有启动前兼容检查 | 已修复 | 用本轮主链产物刷新共享 IR/sidecar 等 6 个文件（dynamic_dispatch、符号单位）；模板改为「说明 + 独立输出目录」，写明 run 目录配置才是可直接运行的产物；启动预检错误补「重跑主链/用 run 目录配置」提示；产品一致性用例把执行语义=编译器默认、指纹与严格契约钉住 | 同类入口：llama2 共享旧产物走同一严格预检（K12）；主链 run 目录配置已可直接运行 | `genesim/models/`（产物）、`genesim/conf/sim_qwen3_moe_pimir.yaml`、`genesim/src/scheduler/gene_sim_scheduler.py`、`tests/test_genesim_simulation.py` | `test_checked_in_qwen3_standalone_products_match_the_compiler_default`；模板通过启动预检并进入仿真，同源 1 请求配置 `./run.sh --config` exit 0 |
| 问题1 | round18 | 高 | 成立：复跑探针，B 路 op3 加假 shard 并删单位后 `validate_sidecar_contract`、`check_sidecar_entries` 与直接 GeneSim 全部 ACCEPTED；设备 GEMM 置空 shards 也放行 | A/B 路分类取自被验产物自己的 `shards` 自报，六个消费点各自读分片字段；源 IR 类别没有唯一入口 | 已修复 | 两仓统一分类规则：源 IR `op_type == GEMM` 才是 A 路，其余算子都是 B 路；B 路非空 shards、严格产物设备 GEMM 空分片直接失败；覆盖审计、单位检查、trace 入口、放置加载、本地形状、TP 展开与继承全部使用同一规则 | 同类路径：`_bpath_unit_contract_problems`、`_bpath_unit_problems`、`check_sidecar_entries`、GeneSim `_require_pimir_unit`/`_apply_compiler_local_shapes`/`_expand_multi_shard_operators`/`_validate_pimir_source_coverage`/`_load_compiler_placement`；4L 真实产物 1557/1606 分类不变 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、两仓测试 | `test_bpath_entry_with_a_fake_shard_is_rejected`（3 入口）、`test_device_gemm_without_shards_is_rejected`（3 入口）、`test_skip_simulation_rejects_a_fake_shard_on_a_bpath_entry`、`test_bpath_entry_with_a_fake_shard_fails_classification`、`test_device_gemm_without_shards_fails_classification`、genesim 2 条 |
| 问题2 | round18 | 中 | 成立：`trace_tokens` 改 128、本地宽改 384、或把 `pimir_path` 指向旧 128-token/无 traceTokens pimir 后，freshness 与仿真 IR 仍 ACCEPTED；共享验证器只验单位存在性与符号维角色 | 单位被复制成 sidecar 自报字段，共享验证器只校验声明结构，不读 pimir 实产；trace 入口与 GeneSim 各自再读一次，形成两套口径。round17 只补了「存在性」，数值维未收敛 | 已修复 | pimir 成为单位唯一真源：按路径解析一次 `pim.traceTokens` 与参数/结果形状，要求 `trace_tokens` 与特征末维逐值一致；SPLIT 纯视图只比输入；GeneSim 生成 trace 时对同一份单位再核一次；缺 `pim.traceTokens` 的严格产物直接拒绝 | 同类路径：五入口共享 `_bpath_unit_contract_problems`；GeneSim `_validate_unit_against_pimir`；SPLIT 代表输出形状不参与输出末维比较；4L 1606 条 B 路条目与 15 份 pimir 逐值一致 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、`genesim/src/pim/pimir_trace.py`、两仓测试 | `test_unit_tokens_must_match_the_pimir`（2）、`test_unit_feature_width_must_match_the_pimir`（2）、`test_unit_must_match_the_pimir_actually_referenced`（2）、`test_pimir_without_trace_tokens_is_rejected`（2）、genesim `test_sidecar_unit_must_match_the_referenced_pimir`、`test_pimir_without_trace_tokens_is_rejected_for_a_unit`；4L 真实产物 ACCEPTED |
| 问题3 | round18 | 中 | 成立：同一全量命令得到 1 failed（`_queue.Empty`），单跑 0.41s 通过；pytest 打印 summary 后被 fork 子进程挂住 | 串行化判据用「5 秒内收到四条队列消息」表达，超时随机器负载抖动；teardown 只 `join(timeout=5)`，未 terminate 残留子进程 | 已修复 | 等待上限改为 60 秒死锁检测；`finally` 中 join/terminate/kill；新增首次 put 晚于 5 秒的慢调度用例，旧写法实测复现 `_queue.Empty` | 同类路径：任何等待多进程结果的测试；本仓仅此一处固定 5 秒多进程等待 | `tests/test_opcompiler_linear.py` | `test_cache_key_lock_waits_for_a_slow_schedule`、`test_cache_key_lock_serializes_processes` |
| 问题4 | round18 | 低 | 成立：`--summary dynamic --summary dense --time-only alt --time-only 48l` exit 1（history 不能核验）；只有两份 current summary 时 exit 0 | 文档把 current 可核验与 history 只做正文引用混写成一个命令，且没有测试执行文档声明 | 已修复 | 声明改为实际命令（dynamic/dense 两份 current summary）；alt/48l 明确为 history 正文引用；新增 CI 用例按证据块 current 标签执行文档命令并断言退出码 0 | 同类路径：I75/F59/F63/§7.2 round12/15/§8.3 全部同步；checker 对当前文档退出 0 | `docs/implement-qwen3-moe-20261105.md`、`tests/test_implement_evidence.py` | `test_documented_checker_command_only_uses_current_summaries` |
| 问题5 | round18 | 低 | 成立：4L 文档把 `max_resident_bytes=779,673,600`（per-TensorPU）与 8,589,934,592（per-cluster）配对，实际集群常驻是 3,140,834,304（36.6%） | 从 summary 挑了同名不同 scope 的字段，没有用 `pim_capacity_metrics.json` 的集群字段 | 已修复 | 三处引用统一为 `max_cluster_resident_bytes`/`max_cluster_utilization`；per-TensorPU 最大值与 512 MiB 上限单列；证据块补 4 个容量 facts | 同类路径：§4.1、§8.4 dynamic 与 dense 三处；48L 的 `max_cluster_resident_bytes=8,412,147,712` 原本就是集群口径，无需改 | `docs/implement-qwen3-moe-20261105.md` | checker 对两份 current summary 退出 0；4L 容量 facts |
| 问题6 | round18 | 低 | 成立：8B 测试手写 `HwBudget(8*2**30, 64, 64*2**20)` 与 `PIMHardwareConfig(dma_align=64)`；30B 测试手写 `PIMHardwareConfig` | round03 只改了 30B 的 `HwBudget`，没有覆盖真实测试整套三件套，也没有静态守卫 | 已修复 | 两处改用 `default_site_capacity(num_dpus=NUM_DPUS, num_tasklets=4)`；新增静态守卫扫描手写容量/对齐/WRAM 字面量与手写构造 | 同类路径：`tests/test_architectures.py` 的微小模型夹具仍用小预算，不属于 8 GiB 契约入口；真实测试由守卫覆盖 | `tests/test_qwen3_8b_real.py`、`tests/test_qwen3_30b_real.py`、`tests/test_architectures.py` | `test_real_weight_tests_derive_capacity_from_the_single_factory`（2 参数）、4 层真实目标命令 36 passed |
| 问题1 | round19 | 高 | 成立：复跑探针，1557 条 GEMM 全指向一份 `pim.gather` 产物（同步哈希）后五入口 ACCEPTED、直接 GeneSim exit 0、`total_time_s` 60.785→1.696；换另一分片宽度、删 `kernel_tile_n` 同样接受 | round18 只定义了 B 路编译单位，A 路只绑路径与哈希；共享验证器对 GEMM 直接 `continue`，GeneSim 在 `unit is None` 时提前返回；测试夹具把同一份 LUT pimir 同时给 GEMM 与 B 路 | 已修复 | 两仓按源 IR 逐条解析 GEMM pimir：单一 `tt.dot`/`pim.matmul`、无 traceTokens/相位链；M=1，K/N 与 tile 属性、`kernel_tile_n`、各分片本地宽、K/N 循环一致；融合 `act_spec` 与清单双向一致；共享验证器、GeneSim 覆盖与 trace 单位校验共用 | 同类路径：`_apath_unit_problems`、`_gemm_pimir_problems`、`pimir_gemm_facts`、`_pimir_unit_facts`；契约夹具 A/B 路分开引用同形产物；4L/共享 dense/llama2 真实 A 路产物 ACCEPTED | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、`genesim/src/pim/pimir_trace.py`、两仓测试 | `test_audit_rejects_gemms_pointing_at_a_bpath_pimir`、`test_audit_rejects_a_gemm_pimir_from_another_shard_width`、`test_audit_rejects_a_device_gemm_without_kernel_tile_n`、`test_audit_rejects_a_fused_gemm_without_act_spec`；genesim `test_gemm_pointing_at_a_bpath_pimir_is_rejected` 等 4 条；探针两仓 REJECTED |
| 问题2 | round19 | 高 | 成立：`lm_head` op3679 改成 `device_hint=cpu` 并删分片/角色/分块，freshness/仿真 IR/skip-simulation ACCEPTED，直接 GeneSim 以 45.895 s 跑完；同改法用于 q_proj 时两仓判定分叉；真实侧车 `operators` 里 host 条目为 0 | `device_hint` 是被验条目自报开关，`_entry_is_host` 只读它；round18 只把 `shards` 从分类依据删掉，host 判定没有完成 | 已修复 | host 判定收成三件套证据：源 IR `device_hint=cpu` + `pimir_exempt_ops` 登记 + GEMM 的 `host_gemms` 角色/权重；`operators` 自称 host 缺任一证据直接失败；五入口与放置加载共用 | 同类路径：`check_sidecar_entries`、`_entry_class_problems`、`_assert_placement_matches_run`、GeneSim `_validate_entry_classes`/`_load_compiler_placement`；真实 4L/48L 的 0 条 host 条目不被误判 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py` | `test_audit_rejects_a_host_gemm_claim_without_source_ir_evidence`、`test_strict_entries_reject_a_host_gemm_claim`（3）、`test_skip_simulation_rejects_a_host_gemm_claim`；genesim `test_host_claim_without_source_ir_evidence_is_rejected`；`tampered_lmhead`/`tampered_hostgemm`/`tampered_router` REJECTED |
| 问题3 | round19 | 中 | 成立：op153（SILU）改为 `VECTOR_ADD` 后 freshness/仿真 IR/skip-simulation ACCEPTED、直接 GeneSim 才 exit 1；删设备 GEMM 的 `semantic_role` 共享验证器仍接受；同形 add/mul 与 SILU/GELU 互换可签发 | 条目 `op_type`/`semantic_role` 是冗余自报字段，共享验证器只查 mnemonic 认识与否；trace 入口还反过来用条目自报类型选判据 | 已修复 | 类检查逐条比较 `op_type` 与源 IR 类型、设备 GEMM 的 `semantic_role` 与投影身份（源 IR 有角色时必填）；B 路逐元素 pimir 核对 add/mul；trace 字节判据改用 IR 类型；两仓同一张 kind 表 | 同类路径：`_entry_class_problems`、`_bpath_unit_contract_problems`、`_bpath_unit_problems`、GeneSim `_validate_entry_classes`；真实产物条目全部带类型与角色 | `scripts/run_full_pipeline.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、两仓测试 | `test_audit_rejects_a_bpath_op_type_that_disagrees_with_the_source_ir`、`test_audit_rejects_a_device_gemm_without_semantic_role`、`test_audit_rejects_a_bpath_eltwise_kind_that_disagrees_with_the_ir`、`test_trace_lineage_checks_bytes_by_the_source_ir_op_type`；`tampered_optype` REJECTED |
| 问题4 | round19 | 中 | 成立：只关 `_compiled_paged_attn` 时取整由编译 matmul 兜住；同时关掉 matmul 后主机路径与相位表参考差 1 个 f16 ulp，注释与 §8.3 的边界不成立 | 主机回退把取整委托给 `matmul` 的返回 dtype，工具链缺席时 numpy 返回 f32；测试夹具只强制 paged_attn 走主机 | 已修复 | `_host_paged_attention` 显式落 f16 边界（QK、乘 scale、每块 PV 乘积）；新增双内核缺席用例按相位表参考逐元素对拍并验证数据能区分取整；§8.3 表述改为实际规则 | 同类路径：`add_mask`/`softmax` 已各自按 f16 落盘，主机路径只在 QK 与 PV 两处缺边界 | `runtime/kernels.py`、`tests/test_kernels.py`、实施文档 §8.3 | `test_sdpa_host_fallback_rounds_at_the_phase_boundaries`；`tests/test_kernels.py` 31 passed |
| 问题5 | round19 | 低 | 成立：4L 产物 `max_utilization=145.23%` 而 `fits=True`；文档把 `max_resident_bytes=779,673,600` 与 512 MiB 上限配成「per-TensorPU 最大值」 | 一个字段同时承载总量与判定量两种 scope；round18 只改了文档引用，未拆指标 | 已修复 | `max_utilization`/`max_tensor_pu_utilization` 改用与 `fits` 同源的判定分子；输出请求期、集群常驻与总量比三个独立字段；summary/控制台/文档同步 | 同类路径：`validate_autotune_dataset` 要求利用率 ≤1，旧值在编译器放置产物上会失败；now 与 fits 同源 | `genesim/src/pim/capacity.py`、`genesim/src/main.py`、实施文档 | `test_judged_utilization_matches_the_fits_basis`、`test_main` 渲染用例 |
| 问题1 | round20 | 高 | 成立：复跑探针，256 条 ROPE 全指向 128 宽 RMSNORM pimir、128 条 PAGED_ATTN 全指向同一 RMSNORM pimir 后，编译器共享验证器与直接 GeneSim 全部 ACCEPTED（ROPE 时延 0.0448→0.0173 s）；修复后同一探针编译器 REJECTED（1024/512 条 B 路语义不符）、直接 GeneSim exit 1（仿真前拒绝），4L/48L/llama2 真实 B 路产物 0 问题 | round19 只把条目 `op_type`/`semantic_role` 与源 IR 绑定，没有把「pimir 内容本身」定义为契约；两仓逐元素表只覆盖 add/mul，且 GeneSim 在 `phase_count != 1`、算子不在 `_UNIT_*` 集合时提前返回，同宽不同算子的产物可以互换 | 已修复 | 两仓各持一张按真实编译实产校准的闭合表：18 个 B 路类型登记必需顶层原语多重集、允许集合与关键属性（ROPE 用 `rotateHalf`/`sourceSubBlocks` 三相 eltwise 链；四节点注意力按实际编译产物登记 MASK/softmax 链/two matmul 的固定读写端）；编译器共享验证器严格分支逐条核对，五入口继承；GeneSim 覆盖校验与 trace 加载用同一张表，语义核对移到相位/字节分支之前；未登记类型直接拒绝；跨仓测试钉住两表逐值相等 | 同类路径：`_bpath_unit_contract_problems`、`_pimir_unit_facts`、`_bpath_pimir_semantic_problems`、`_validate_pimir_source_coverage`、`_assert_pimir_operator_semantics`、`_validate_pimir_unit`；两仓覆盖/签名/GEMM 夹具全部改为按类型构造；被吸收节点的 pimir 不参与 B 路代价（激活由主 GEMM 的 `act_spec` 与融合清单证明） | `scripts/run_full_pipeline.py`、`genesim/src/pim/pimir_trace.py`、`genesim/src/scheduler/gene_sim_scheduler.py`、两仓测试 | `test_audit_rejects_a_bpath_pimir_with_another_operators_semantics`（5）、`test_audit_rejects_a_bpath_operator_type_without_a_calibrated_fingerprint`、`test_bpath_semantic_table_accepts_each_calibrated_product`（18）、`test_bpath_semantic_table_is_pinned_across_repos`、`test_genesim_pimir_facts_agree_on_each_calibrated_product`；genesim `test_pimir_with_another_operators_semantics_is_rejected`、`test_bpath_pimir_with_another_operators_semantics_is_rejected`、`test_calibrated_forms_pass_their_own_fingerprint`；`/tmp/r20-fix-probes/` 两探针两仓 REJECTED |
| 问题2 | round20 | 低 | 成立：文档四处把 summary 的 `max_resident_bytes=779,673,600`（单 TensorPU 最大值）写成「跨 TensorPU 的常驻总量」，同时又引用 `max_total_resident_ratio`；按字面与 8 GiB 集群预算相比会读成约 9% | 代码 scope 已正确，但 summary 只暴露 `max_resident_bytes` 一个键，文档沿用了旧表述；checker 只核数值与运行语境、不核措辞 | 已修复 | 四处改为「单 TensorPU 常驻最大值（含该 TensorPU 的模型分片与请求期数据）」；`max_total_resident_ratio` 明确为与单 TensorPU 上限的诊断比、不参与 `fits`；集群总量统一引用 `max_cluster_resident_bytes`/`max_cluster_utilization`；summary 新增三个显式 scope 字段并加用例 | 同类路径：design §十.5 同步改为单 TensorPU scope；§1.3、§8.4 dynamic、§8.4 dynamic 证据块与 §8.4 dense 四处；48L 的 `max_cluster_resident_bytes=8,412,147,712` 原本就是集群口径 | `genesim/src/main.py`、`genesim/tests/sim/test_main.py`、实施文档、设计文档 | `test_capacity_scope_is_explicit_in_summary`；设计/实施文档措辞与 `pim_capacity_metrics.json` 字段一致 |
### 7.2 未修复问题与实现取舍说明
- **问题7**：评审要求先确认 8 GiB 是否仍为硬约束。该决定涉及甲方硬件参数，当时无人值守，
  按约定写入 Q5 并跳过，不擅自把测试预算从 16 GiB 改为 8 GiB，也不宣称 8 GiB 已验收。
  （round03 已获用户确认并修复，见 round03 问题2；本条保留为历史取舍记录。）
- **问题5 的“专家子图复用”方案未采用**：评审给出“专家子图复用或明确的安全分块策略”二选一。
  专家 linear 的本地形状本来由算子层 `tt.dot` 分块压进 WRAM，图级拆分只会把每个专家拆成
  上百个子节点（4 层实测 >130 万），没有执行收益；因此选择“显式关闭图级分块并默认对 MoE
  生效”，同时保留 `PIM_DISABLE_TILE` 与 `compile_llama2(graph_tiling=)` 两个仓库内入口。
- **问题4 的“新鲜导出不得沿用旧产物豁免”**：实现为没有 `requires_pimir` 的自报就不启用严格
  完整性检查（纯放置导出仍合法），自报后缺项/缺哈希一律失败；没有引入额外的“新旧产物”判定。
- **round02 问题1**：8 GiB 是否仍为硬约束属甲方硬件参数，当时无人值守，按约定写入 Q5 并跳过；
  不改主链 4 GiB、真实用例 16 GiB、GeneSim 8 GiB 中任何一处，也不宣称 48 层主链容量通过。
  （round03 已获用户确认并修复，见 round03 问题2；本条保留为历史取舍记录。）
- **round02 问题2 的 GeneSim 动态计费**：仿真是没有 routing 数据的静态代价模型，无法知道
  每个请求实际命中的专家集合，因此按期望值折算：计算按 `top_k/E`（token-专家对），
  通信按命中并集 `1-(1-1/E)^(Tq·top_k)`；公式只依赖模型元数据与请求长度，不引入随机数。
  该口径已写入 design §十.6，待有真实 routing trace 时可替换为实测。
  （round03 已按执行计划的 any-hit 语义修正为 `1-(1-k/E)^Tq`，见 round03 问题3；
  本条保留为历史取舍记录。）
- **round02 问题6 的验收口径**：评审在旧实现上用同一小模型测得 0.994×；本轮提交前跳过在
  4 层真实权重上测得 prefill 1.148×、decode 1.202×（模型不同，不作直接可比）。剩余执行期
  由注意力/门控等非专家命令主导，继续提高需要优化通用执行后端，超出本需求根因。按评审
  “若优化后仍无显著收益，应保留未完成或回到需求确认”，当时转 Q7 待确认，不宣称
  P1-1 已满足“显著加速”。（用户已接受 1.148×/1.202×，Q7 关闭；本条保留为历史记录。）
- **round02 问题6 的累加链**：每层 128 个专家的加权/累加尾部仍不门控（round05 起
  与选择一起在主机闭包里执行）。汇合多专家输出，按专家门控需要额外的零初始化
  语义；实测其开销不在剩余热路径上，本轮不做超出根因的改动。

- **round03 问题2 的 48 层容量核算**：按 48 层权重元数据（tp4×pp2、64 字节对齐）：
  每台专家分片 7,247,757,312 B；stage0 权重+16 token 最小 KV 为 8,100,720,704 B、
  stage1 为 7,633,977,408 B；8 GiB 扣 64 MiB 预留后可用 8,522,825,728 B，两段余
  422,105,024 / 888,848,320 B 给激活与内核。round06 的 48 层 BCD 已按 8 GiB
  完成（44117 算子、37923 条 sidecar 条目、23 份 pim mlir），未报超预算；
  全量激活+内核峰值与 GeneSim 终态未完成，保留 K7；不改成 16 GiB。
- **round03 问题2 的预留口径**：编译器按 `8 GiB − 64 MiB` 判「三区 + 内核 tile」，
  GeneSim 按 ClusterPU 总量 8 GiB 判常驻；总量同源，编译器更严格（先报
  `dpuX 内存超限: peak=... > 可用预算=...`），不会放过 GeneSim 能接受的超限产物。
- **round03 问题3 的 dense/dynamic 对照**：旧的 46.627 s / 39.378 s 建立在错误折算上，
  本轮作废；同一入口重测的 dynamic/dense 数字与解释见 §9.2。仿真没有 routing 数据，
  动态计费只能按均匀路由期望 `1-(1-k/E)^Tq`；有真实 routing trace 时可替换为实测命中集。
- **round03 问题3 的 TP 展开缺口（本轮新发现）**：即使 `_get_operator_with_runtime`
  算出了比例，TP 展开会在 `split_operator` 里丢掉 `moe_cost_role`，导致放置后的
  所有专家分片按非专家全量计费——4 层 dense 与 dynamic 的总时长逐位相同就是这个
  症状。修复后新增管线级用例，确保动态语义真的进入事件调度。
- **round03 问题1 的模型级输出头**：lm_head 不属于任何层，只补 IR 不够——放置导出
  不遍历模型级 GEMM，GeneSim 的 TP 展开也只遍历层子图，两处都必须显式处理，否则
  输出头停在一个全局算子上、没有真实分片与 trace（I29/I31）。
- **round03 问题4 的旧产物兼容**：无指纹 sidecar 只在 IR 自身也没有身份时按旧行为
  放行；真实 IR 由 parser 写入指纹后，严格模式缺 `ir_fingerprint` 即失败。
  Llama2 等既有 sidecar 需用当前主链重新导出一次。
- **round04 问题1 的扩展：ROPE/MASK 逐头形状**（评审未点名）：根因排查发现
  两者的测量代表形状乘了 `num_heads`，单节点成本被放大 32 倍（ROPE 实测
  12288 vs 逐头 384）。这与 round02 问题5「代表形状不跟随算子」同一根因，
  前一轮修复只覆盖 VECTOR_ADD/MUL/SILU/RMSNORM，属同类路径遗漏，本轮一并
  改为 `[1, Tq, ...]` 并在同一用例里钉住系数。
- **round04 问题1 的系数来源**：没有额外保留模板解析系数，而是让测量 IR 与
  拟合共用配方声明的依赖项。B 路的测量值是真实算子 IR 的确定性算术结果，
  依赖项正确后拟合系数与解析值一致（VECTOR_MUL=768、ROPE=384、MASK=1），
  不需要再维护第二套公式；模板系数仍用于 A 路未覆盖算子。
- **round04 问题3 的 lineage 口径**：同源校验在 TP 展开前对未展开 IR 完成；
  展开后的图必须重算自身指纹，避免把展开前的旧值留在可落盘 IR 上形成假证明。
- **round04 问题2 的扩展：lm_head 真实布局**（评审未点名）：按源 IR 对拍 trace
  时发现真实图骨架把 lm_head 归进最后一层 subgraph，而
  `_expand_multi_shard_operators` 只用「不在任何层里」筛选模型级算子——lm_head
  因此从未展开，既没有分片也没有 pimir trace，整块词表 GEMM 留在 GPU 上执行。
  round03 的单测夹具是「lm_head 不在层里」，所以没暴露。这是上一轮修复只覆盖
  探针布局的同类路径，本轮改为按「层内循环未替换」判定并补真实布局用例（I39）。
- **round04 问题2 的 trace 来源口径**：顶层 `trace_source` 记录实际来源
  （pimir/absorbed），`compile_signature.trace_source` 只是编译输入口径；
  校验优先读顶层，旧 trace 才退回签名。4L 实跑顶层分布为 pimir 8858 / absorbed
  2048，而签名口径仍是 10906 全 pimir，两者不能混用。
- **round05 问题2 的“路由加权乘下设备”子结论不成立**：修复前的真实 4L 计划
  decode 图里，`where` 2048 条是设备 launch（全部带 gate），而路由加权乘与
  累加已被 `fuse_elementwise` 折成 504 个 host 闭包（每层 126 个，覆盖 128 个
  专家的尾部），`aten.mul` 的设备 launch 全部出现在专家内部（silu×up）与注意力
  链；本轮把选择归 host 后 where/ne 的设备 launch 为 0。
  因此“加权 mul 应按命中率缩放”在现有执行计划下没有对象；真正的问题是设备侧
  的 `where` 被并进 host `MOE_ACCUM` 并排除在 pimir 之外，这一半已按根因修复
  （选择归 host、设备侧只剩可门控专家子图、MOE_ACCUM 成本计入选择）。
- **round05 问题1 的断言口径**：逐层测试不断言“每层命令数完全相等”（分块/
  切分会改变数量），而是断言每条门控命令的专家归属层号等于其 dispatch 层号，
  且每层至少覆盖 8 专家 × 7 类命令；这样“只有首层生效”会被直接抓出。
- **round05 问题4 的 GeneSim 侧常量来源**：两仓是两个独立仓库，GeneSim 无法
  直接 import 编译器契约，因此在 `genesim/src/pim/capacity.py` 持有同一组
  `DPU_SYS_RESERVE_BYTES/DPU_DMA_ALIGN`，由跨仓测试逐值钉住相等；只改一侧会
  在测试里失败，而不是等仿真启动才发现。
- **round05 问题2 的 EP 路径**：PP×EP 的 host 聚合由
  `graph.spec_prop.remark_moe_aggregation_for_ep` 按实际切分语义单独声明，
  与 TP 主链“选择/加权累加在 host”是两件事；本轮没有改动 EP 的标 host 逻辑，
  EP 用例只补了“每层专家计算仍按本层 dispatch 门控”的断言。
- **48 层状态**：round05 的代码改动涉及专家归属门控、host 选择落点与容量
  校验，round01 的历史数值与 round02 的 OOM 结果都不能代表当前代码；本轮
  未重跑 48 层目标用例与主链（单次约 7 小时，且两个旧代码 48 层任务仍在写
  同名产物），保留 K7/K8/Q8 并按历史/环境阻塞分栏登记，不宣称 48 层通过。

- **round07 问题2 的信任边界**：豁免证据由放置导出按 `spec.device` 写入 IR 与
  sidecar，审计/仿真逐条核对。这条链能阻止流水线自动生成错误豁免，也能抓住
  「删条目再手写豁免」；但产物本身没有签名，能同时伪造 IR、sidecar 与 pim mlir
  的对手仍可自洽。本轮按评审要求闭合自动路径，不引入签名机制（超出需求）。
- **round07 问题1 的端到端非有限值用例**：选中专家输出 inf 时 HF 结果本身是
  nan，token 生成无法用 argmax 对拍；因此数值等价用静态前向逐元素对拍
  （含两态归一化），执行路径用 decode/prefill 计划断言 dispatch 读布尔掩码，
  P0/P1 数值由既有 overflow 用例覆盖。没有为 nan 场景再造一个不可判定的
  生成用例。
- **round06/07 问题2 的小形状单测**：`test_fused_gate_kernel_carries_the_activation`
  的夹具 K=16（fp16 一行 32 字节），装不下 64 字节对齐的分块，因此显式传
  `dma_align=8` 的硬件配置；生产契约 64 与本轮 4L 真实产物由主链用例和
  align 统计覆盖，不用放宽契约迁就夹具。
- **round07 问题5 的 48 层耗时**：4 层展开从 99/102 秒降到 1 秒内、trace 与
  total_time 逐项等价；48 层在 round06 的只读产物上用当前代码跑展开基准：
  **6.24 秒**（旧实现 5:14:33），30961 个 TP 算子 → 130036 个分片、44117 →
  143192 个算子。未启动第二条 48 层主链（round06 单元仍在运行，用户约束
  不重复启动）。

- **round08 问题1 的产物身份**：结构指纹按设计排除放置字段，所以另一份合法
  策略的 sidecar 也能通过指纹。修复后 E 阶段只消费 BCD 返回的 placement，
  并在启动前核对策略标签、DPU/段数、Cluster 映射、逐 GEMM 分片数与本地形状、
  逐层所属流水段；这比把放置字段塞进指纹更直接，也不破坏指纹的用途。
  同一模型连续跑 `tp4_pp2` 与 `tp1_pp4` 的实跑证据见 §8.4；两种策略各跑一次
  真实主链（不是 pytest 内重复两条数十分钟的编译+仿真），单测覆盖配置改写、
  错策略/错本地形状/错层段拒绝。
- **round08 问题3 的单专家真实 BCD**：1 专家/top-1 小模型的 router GEMM 输出宽
  N=1，FlagTree 的 `pim-tile-to-budget` 搜索报 `no legal tile fits`（最小
  M=1 N=1 K=1 也不通过），因此无法在现有工具链上跑出单专家 BCD 的正向产物。
  策略入口的合法/非法判定已由 parser、`resolve_strategy` 单测和真实主链的
  非法分支覆盖；N=1 分块列为 K11/Q10，不通过放宽专家约束绕过。
- **round08 问题4 的验证器定位**：生产入口统一调用
  `validate_sidecar_contract`；`audit_sidecar_coverage` 仍只负责覆盖/豁免/
  吸收/专家身份，是验证器的组件，测试可以直接调用它验证覆盖规则。这样
  「五处入口同一契约」由唯一生产入口保证，同时保留覆盖审计的单元测试。
- **round08 问题5 的边界**：预检与真实规划共用 `build_kv_layout` 后，
  「实际装得下但旧预检拒绝」的正例与「超预算 1 字节仍拒绝」的反例都已固化；
  预检仍只拦显式给定槽池的配置，自动定容仍交给编译期 `plan_dpu()`。
- **round08 问题2 的 48 层分栏**：48 层主链用当前源码重新生成全部产物并在
  GeneSim 形成 summary/trace（total_time_s=566.863、trace
  `{'pimir': 106230, 'absorbed': 24576}`）；48 层数值目标用例仍保留 round06
  历史结果（当前代码的 4 层数值已通过），两者分栏登记，不把历史数值当作
  当前产物。
- **round09 问题1 的复发原因**：round05 的问题3只校验了「absorbed → 清单」
  单方向，round08 又把同一个无阶段参数传给 B+C+D 与精化后复核，多消费者
  方向和「清单在、吸收编号被删」两态都没有对象覆盖；同时 GeneSim 侧已写的
  absorbed 契约用例因 I57 从未在标准入口执行，所以前几轮的「全量回归通过」
  不能证明这条契约被保护。本轮同时修代码与测试入口，并按根因补两仓反例。
- **round09 问题1 的消费者口径**：GeneSim 的图骨架不含 `to`/`assert`/`alias`
  这类透明节点（`graph/fuse_pim.py` 的 `_real_users` 过滤发生在 FX 层），
  因此 IR 依赖边上的直接消费者就是 `graph/fuse.py::_single_consumer` 语义下
  的真实消费者；校验按依赖边的消费者集合与 `{被吸收激活}` 严格相等实现，
  不为不存在的节点类型预造透明规则。
- **round09 问题2 的复发原因**：round07 只把「豁免与放置矛盾」落在 GeneSim，
  round08 把编译器各入口收拢到共享验证器时没有移植这条不变式，统一的是
  不完整子集；且 GeneSim 的覆盖循环在查豁免之前先跳过模型边界，边界豁免
  在该侧也从未被核对。本轮两仓使用同一组函数（放置矛盾 + `host_gemms`
  身份），并让边界豁免进入覆盖校验。
- **round09 的 48 层口径**：本轮改动只发生在产物校验与测试入口，不改变
  图骨架、放置、成本与 trace 生成路径；48 层主链沿用 round08 当轮源码的
  完整产物与 summary（A 44117 算子、E total_time_s=566.863、
  trace `{'pimir': 106230, 'absorbed': 24576}`），48 层数值保留 round06
  历史结果，本轮不重复启动长任务（用户约束）。
- **round09 的探针与审计留存**：两仓反例复跑见
  `/tmp/round09-probe-current.log`（多消费者、删除吸收编号、矛盾放置三条
  探针在编译器五个入口与直接 GeneSim 全部 REJECTED，真实产物 ACCEPTED）；
  4 层 dynamic/dense 产物审计见 `/tmp/round09-audit-dynamic.json` 与
  `/tmp/round09-audit-dense.json`；全量回归日志 `/tmp/round09-regression.log`，
  GeneSim 套件日志 `/tmp/round09-genesim-suite.log`。

- **round10 问题1 的复发原因**：round03 引入成本角色时用目标算子的角色代表
  整条边的语义，round04 把旧式依赖并入同一出口但没有改方向假设；既有用例只
  构造「普通源→专家目标」，恰好与目标角色假设同向，所以全量回归一直通过。
  本轮把判定改为同时看两端，并补「专家源→共享/HOST」的反向用例；四方向、
  TP 展开与显式/旧式两种表示都由同一出口覆盖。修复后复跑评审探针
  `/tmp/round10-fix/probe-moe-edge.log`：dynamic 的专家源→共享累加边为 2 B
  （修复前 16 B），shared→shared 仍为 16 B，dense 保持 16 B。另在真实 4L E
  阶段对 `_moe_dependency_fraction` 打计数探针：放置估算与正式管线合计
  1,818,150 次调用，其中 1,401,280 次命中专家边、1,381,760 次按比例折算，
  证明两处出口都走了新判定（`/tmp/round10-fix/probe_fraction_calls.py`）。
- **round10 问题1 的 `tp4_pp2` 口径**：TP 展开为每个专家 `down_proj` 分片插入
  `ALL_REDUCE` 中转，该中转在放置后与 `MOE_ACCUM` 同属 host VPU，因此
  `ALL_REDUCE → MOE_ACCUM` 边不产生跨设备传输；被计费的
  `down_proj 分片 → ALL_REDUCE` 边以专家为中转目标、本来就按命中并集折算。
  所以 `tp4_pp2` 的通信字节与总时长不变，`tp1_pp4` 的真实 PIM→host 边减少
  正好 68 decode × 1,966,080 B = 133,693,440 B。两策略都按同一规则计费，
  不是保留特例。
- **round10 问题2 的历史值口径**：round03 的 `5,430,713,177,457.654` 是历史
  对照，不是本轮产物；round10 用 `--history` 显式声明后才放行，round11 起
  改为证据块的 `history` 命名空间。当前 dynamic
  `5,415,241,254,987.654`、dense `5,826,198,177,867.654` 与两份 summary 逐位
  相等，dynamic 与 round03 的差值重算为 1.55e10。
- **round10 的 48 层口径**：本轮只改 GeneSim 依赖边折算，不改变图骨架、放置、
  编译与 trace 生成；48 层 A/BCD/A2 沿用 round08 当轮源码产物，E 阶段用当轮
  GeneSim 代码在隔离根重跑：130806 条 trace 全重编、0 复用，
  `total_time_s=566.863`、`tokens/s=1.402`、通信字节 166,702,607,112，
  与 round08 逐位一致；审计 ACCEPTED，容量最大值
  8,412,147,712 / 8,589,934,592 B（`fits=True`）。

- **round11 问题1 的复发原因**：round03/05/06/07 的容量修复都把“期望值”取成
  `resolve_pim_capacity_limits()` 的结果，统一的是各入口之间的取值，而不是
  “取值等于需求常量”。配置与 sidecar 同步改大后两边仍自洽，验收标准就跟着
  被抬高。本轮在 GeneSim 容量契约里新增独立常量 `DPU_MRAM_BYTES`，严格路径
  同时核对有效配置、sidecar 三件套与每份 MLIR 的属性；只有明确标识的旧式
  非严格产物继续按运行配置兼容其它硬件容量，避免误伤通用仿真。
- **round11 问题2 的复发原因**：round08 只把“本次 sidecar”与日志副本命名绑定
  到策略，A/A2/BCD 仍用固定文件名，E 仍以“清空共享 `pim_traces`、读取共享
  `results`”为运行前提。根因是缺少贯穿 A→A2→BCD→E 的运行身份，而不是删除
  动作选错了目录。本轮按架构/策略/执行语义/唯一运行号派生运行目录，A/0/A2/
  BCD 产物写本次 `artifacts/`；E 的临时配置按 label 命名并绑定本次 IR、
  trace、results 与 log 路径，回读复核；trace 目录必须是本次的空目录，删除
  共享目录与复制共享结果两条路径一并移除。该扩展覆盖了评审点名的 E 阶段之外
  的同类路径。
- **round11 问题3 的复发原因**：round10 的 F48 把“文档里出现过 summary 的格式化
  数值”当成一致性证明，没有“运行身份 → summary → 指标 → 派生结论”的结构；
  测试只有数值抄错一个反例，标签互换、层级冒充、历史冒充 current 与派生值
  错误都不在覆盖范围。本轮用证据块把标签、层数、语义、命令、summary 与指标
  绑在一起，由 summary 反向核对身份与数值，派生差值/比值/结论由同一组指标
  重算；历史值与 bench 测量使用独立命名空间，不再用 `--history` 放行裸数字。
- **round11 的 48 层口径**：按用户约束不重复启动 48 层长任务。48 层数值保留
  round06 历史结果（26 passed、1 skipped，4:47:34）；48 层主链沿用 round08
  round08 当轮源码产物与 round10 用当轮 GeneSim 代码重跑的 E 阶段 summary
  （`total_time_s=566.863`、`tokens/s=1.402`、trace
  `{'pimir': 106230, 'absorbed': 24576}`），本轮以该 summary 作为 current
  证据登记，并在证据块中与 4 层运行分栏。本轮没有需要用户确认后跳过的评审项。
- **round11 问题2 的实跑证据**：本轮 dynamic/dense/alt 三次运行分别写入
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261111-190527-2004830/`
  与 `/tmp/qwen3-review-round11-fix/pipeline-{dense,alt}/<运行号>/` 的独立目录；
  A/0/A2/BCD 在 `artifacts/`，E 在 `pim_traces_*/` 与 `results_*/`。
  共享 `genesim/pim_traces` 与 `genesim/results` 的 mtime 保持运行前的
  16:38/16:40，没有被删除、清空或改写。
- **round12 问题1 的复发原因**：round02 问题3 与 round11 问题1 都只把
  「配置严格开关与 sidecar 自报」的并集落在 GeneSim 的 `_strict_pimir()`；
  编译器侧从 round08 收拢共享验证器起，容量核对一直以
  `sidecar.get("requires_pimir")` 为唯一入口，三个复核入口虽然读了
  `require_compiler_pimir: true`，却没有把状态传进验证器。产物自报是
  『可被删除』的一侧，配置是调用方『明确声明』的一侧，只信前者等于把
  严格性交给被验对象自己决定。本轮把并集下沉到共享验证器的 `strict_pimir`
  入参，五个入口都显式传入，旧式兼容只在配置与产物都明确非严格时成立。
- **round12 问题1 的 MLIR 口径**：严格容量契约在编译器侧仍按 sidecar 三字段
  核对；每份 pim mlir 的 `pim.dma-align` / `pim.mram-bytes` 继续由导出步骤
  （`placement_export._assert_pimir_hardware_contract`）与直接 GeneSim 加载时
  逐份解析，符合 design §十.5「导出与 GeneSim 加载逐份解析」。本轮没有在
  freshness/仿真 IR/trace 里重复解析 3163 条算子条目对应的 23 份唯一 MLIR，
  避免把同一规则做成三套 I/O 路径；这条边界不影响本轮两个绕过分支的判定。
- **round12 问题2 的复发原因**：round10 的 F48 以「全篇出现过 summary 数值」
  为一致性证明，round11 的 F51 把证据块内部做成了结构化，但正文仍以
  `_allowed_tokens` 的全局集合判定，必需值检查也搜索整个 `doc_text`，
  于是「证据块正确、正文对调」仍能通过。本轮把运行标签与正文锚点绑定：
  数值必须由所在块的前置标签声明过；只交换正文、4 层冒充 48 层、历史冒充
  current、错误大小结论都会失败。
- **round12 问题2 的绑定口径**：正文里一行/一个列表项/一个表格行视为一个
  语义块，块内任一前置标签声明过该值即放行，最近的标签只用于报错定位。
  这是为了同时容纳「同一段落先提 dynamic、再提 round03 对照」这类写法；
  代价是「dynamic 与 dense 分别为 61.015 和 63.867」这种两个标签都在前的
  歧义句不会按位置拆开，只会在值不属于任何前置标签时失败（K14）。
  带数值的「高于/低于/大于/小于/超过」结论按两侧实际值判定；没有数字的
  比较句不在机器判定范围。派生差值/比值先由证据块重算，出现在正文时按其
  参与标签放行；measurements 是独立命名空间，全局放行。
- **round12 的 48 层口径**：按用户约束不重复启动 48 层长任务。48 层数值保留
  round06 历史结果，主链沿用 round08 当轮源码产物与 round10 的 E 阶段
  summary；本轮用当前代码对该产物重跑 freshness/仿真 IR/trace 三入口，
  均 ACCEPTED（trace 来源 `{'pimir': 106230, 'absorbed': 24576}`），
  容量仍为 `max_cluster_resident_bytes=8,412,147,712 / 8,589,934,592`。
- **round12 的探针与日志**：严格容量探针见
  `/tmp/qwen3-review-round12-fix/probe-strict-capacity.log`（缺自报+缺容量、
  缺自报+4/16 GiB 在 freshness/sim_ir/trace/audit 全部 REJECTED，真实 4L
  产物 ACCEPTED）；正文证据探针见
  `/tmp/qwen3-review-round12-fix/probe-evidence-prose.log`（只换正文被逐条
  报出、删证据块、错误比较、48 层借值、历史冒充 current 全部失败）。
- **round13 未登记与复发的根因**：round13 评审产物已生成，但上一轮只处理到
  round12，§1.3 的「完成」是人工维护的结论，不由 §7.1/§5.1 的未关闭项生成，
  因此 round13 的 5 项严重/高/中问题既未修复也未登记。本轮把 round13 与
  round14 的同类问题合并确认：两项 FlagTree 数值/槽号问题、两项严格复核/trace
  lineage 问题与一项证据归属问题都按根因修复，并在 §7.1 逐条登记，避免下一轮
  继续以「完成」状态掩盖未关闭项。
- **FlagTree 重建的哈希口径**：本次只改 `LowerPIMToEmitC` 的 PV 取整、槽号
  兜底与 `ExpandPhases` 的注释，没有改 `phases` 属性或 pimir MLIR 文本。
  重建 `triton-opt`/`libtriton` 并同步两个安装前缀后，用当前代码对既有 4L
  dynamic/dense（各 10906 条 trace）与 48L（130806 条 trace）产物重跑
  freshness、仿真 IR 与 trace 验收，全部 ACCEPTED；本轮新跑的 4L dynamic
  主链也以 exit 0 通过同一套逐条 lineage 校验。因此不重跑 48 层导出，48 层
  主链、容量与 summary 沿用 round08/round10 证据。重建命令与范围见 §4.1。
- **PV 相位表的 `bytes` 语义**：`PhaseSpecAttr.bytes` 记该相流式读写的缓冲
  区，PV 相读写的是 f32 累加器，所以仍记 `accBytes`；契约要求的 f16 乘积是
  块内临时值，不能记进 `bytes`（那会把容量计费算成 f16）。本轮修正的是
  lowering 的取整点与相位注释，不是相位表字段。
- **严格入口的非严格兼容边界**：主机预检、pimir 产物身份与 trace lineage 只在
  「配置或产物自报严格」时要求 IR/sidecar/哈希齐全；配置与产物都明确非严格
  时，无 sidecar 的旧式用法仍放行。这样既堵住严格探针的绕过，也不误伤纯放置
  导出与旧产物。
- **trace 哈希的旧产物边界**：sidecar 条目没有 `pimir_sha256` 时，非严格
  上下文只做来源与父编号校验；条目带哈希时无论严格与否都逐条比对。旧式
  Llama2 产物（K12/Q11）因此仍可加载，严格契约下的新产物必须带哈希。
- **`facts` 的覆盖范围**：通信字节按运行绑定（最易发生 dynamic/dense/alt
  错配）；参数、容量与 trace 计数用独立命名空间并写明来源，因为同一数值会
  出现在架构表、K7/Q8 与历史说明中，强行逐运行绑定会要求把大段架构描述改
  写成运行语境，超出本轮根因。两类值都由 checker 纳入允许集合，未声明的
  高精度数字仍会被拒绝。
- **round14 的 48 层口径**：按用户约束不重复启动 48 层长任务。FlagTree 重建
  后既有 48L sidecar/pimir 哈希保持有效，三入口重跑 ACCEPTED；48 层数值保留
  round06 历史，主链与容量沿用 round08/round10 证据。
- **根因扩展修复范围**：评审点名的位置之外，本轮按同一根因补了三处。
  其一，槽号不变式不只影响注意力读路径：`sdpa_kernel` 写 KV 的
  `physical_slot` 与主机兜底 `_iter_kv_blocks` 同样只判负值，本轮统一到
  `_physical_slot`；其二，严格入口的绕过不只来自删 sidecar 键：配置文件
  本身缺失时 `_config_text` 返回 `None`，三个入口同样空跑，本轮改为缺失即
  失败；其三，trace lineage 不只需要哈希：`op_type`、`expert_id` 与
  `num_instructions` 同属「这条 trace 是不是这套产物」的证据，本轮一并核对。
  这些扩展都在同一根因（契约没有单一入口 / 证据链未闭合）范围内，没有新增
  设计外功能。
- **round15 问题1 的复发原因**：round14 问题3 把「严格入口的必需输入」只
  枚举为配置键与 IR/sidecar 文件，产物校验仍是共享验证器的可选参数
  `check_artifacts`；同一根因（严格性由入口自行决定）在 pimir 产物这条
  路径上继续存在，round14 的矩阵测试也只覆盖配置键与文件缺失。本轮把产物
  校验并入严格性判定并删除参数，五个入口无法再各选一条路径；测试矩阵同步
  覆盖 pimir 产物的缺项/不可读/缺哈希/错哈希。
- **round15 的入口级包装取舍**：评审建议新增
  `verify_complete_product(genesim, config)` 统一打开产物校验。实现没有加
  这层包装：导出、skip-simulation、freshness、仿真 IR 与 trace 五个入口
  本来就只经 `validate_sidecar_contract` → `audit_sidecar_coverage` 这一条
  验证链，产物校验的开关放进 `audit_sidecar_coverage` 的严格判定即可；
  再加一层包装只会多一个可被绕过的入口，不改变不变式。
- **round15 问题1 的严格边界**：产物校验只在严格上下文（sidecar 自报
  `requires_pimir` 或调用方显式严格）打开。配置与产物都明确非严格的旧式
  sidecar 不做可读性与哈希核对——这些产物没有当前 pimir 身份字段，强行
  核对会把纯放置导出与 Llama2 旧产物误伤；这与设计 §十.4 的严格性并集
  规则一致。
- **round15 问题2 的诊断模式**：评审给出「无配置只返回未验收/仅诊断」或
  「显式 `legacy_no_lineage=True` 的独立诊断函数」两种做法。仓库内没有
  需要无配置扫描的调用方，因此直接把配置设为必需输入，不新增诊断函数；
  来源白名单本身仍保留在验收路径里，只是不再能脱离配置签发通过。
- **round15 问题3 的生成策略**：trace 只描述请求数与 token 长度、与模型
  结构无关，因此缺文件时在本次运行目录生成合成 trace（`--synthetic
  --seed 0`）并绑定绝对路径，而不是让 E 阶段在编译数分钟后才失败。配置里
  已有 trace 时原样使用，生成路径只影响干净检出/CI，不改变既有 4 层与
  48 层数值基线。
- **round15 问题5 的分支取舍**：`where_kernel` 的显式 `broadcast_to` 分支
  是为 `where.ScalarOther` 的标量操作数写的，但该算子在
  `graph/partition.HOST_ONLY`、从未注册进 `_KERNELS`，分支按构造不可达；
  按设计 §十.3 选择删除分支而不是给主机算子注册设备内核。`where.self`
  的张量广播由 numpy 与 `kernels_pim.mask` 原生支持，新用例钉住结果。
- **round15 问题4 的复发原因**：round14 问题6 只收敛了 §1.3 与 §7.1，
  §8.4 的旧证据归属句没有按同一根因清理；完成状态与证据归属仍靠人工誊写。
  本轮把 §8.4 改成「来源—时间—层数—是否重跑」分栏，并同步 §1.3/§4.1/§7.3，
  避免后续再把旧 run 目录的 summary 当作当前代码证据。

- **round16 问题1 的根因与扩展范围**：B 路成本链存在两套互不相干的单位——
  编译器按源 IR 的全局形状与 128 token 占位生成 pimir，GeneSim 运行期把 trace
  当成「单 token/本地宽」单位再乘实际 Tq；TP 展开又把源 pimir 无条件继承给本地
  分片。修复落在契约层而不是逐个算子打补丁：所有 B 路 pimir 统一为
  「1 token × 本地宽」，sidecar 回传 `trace_tokens`/`local_*_shapes`，pimir 函数
  显式写 `pim.traceTokens`；GeneSim 在 trace 生成前核 tokens/末维/单相字节，
  TP 展开只在分片本地形状一致时继承，运行期按 `ceil(Tq/trace_tokens)` 重复。
  同一根因的 Llama2 MLP 分片、直接 GeneSim 入口、PAGED_ATTN 的 128 行占位与
  KV 块循环的 `tokens>1` 分支一并覆盖。真实 4L sidecar 把一条 B 路条目的
  `trace_tokens` 改成 128 后，`verify_trace_provenance` 直接 REJECTED
  （tokens=1 vs 128、输出字节 4096 vs 524288）。4L 实测：VECTOR_MUL 65.244→0.151 s、
  SPLIT 15.317→0.121 s、TRANSPOSE 15.198→0.120 s、RMSNORM 1.077→0.016 s、
  ROPE 4.442→0.045 s；PAGED_ATTN 因按真实查询行重复整份单位 trace 由
  0.694→6.282 s（此前 prefill 少算约 5.7×、decode 多算 128×）；
  round16 的总时长从 88.483 s 降到 61.015 s。
- **round16 问题1 修复中发现的两条同类路径**：其一，`phase_metadata`/单相解析
  原先按「结果首维」猜 token 数，TRANSPOSE 把头维挪到最前（`[32,1,128]`）后被
  读成 32 个 token；改为优先读编译器显式写入的 `pim.traceTokens`，旧 IR 才回退
  形状启发式。其二，SPLIT 的 IR 只记录一个代表输出形状（`[Tq,128]`），pimir 却
  一次搬完整份输入；验收改为按纯视图核对「读写字节相等」，不拿代表形状算输出
  字节，否则会把 32 个头的搬运量少算 32 倍。
- **round16 问题1 对 48 层既有产物的影响（待确认，未自行重跑）**：新单位契约
  要求 sidecar 回传 `trace_tokens`/`local_*_shapes`，round08/10 的 48 层 B 路
  条目没有这些字段，当前严格验证器会拒绝其 31,538 条 B 路 trace。用户本轮明确
  要求「直接引用 48 层历史产物、不重跑主链」，因此没有重新生成 48 层 A/BCD/A2/E；
  48 层结构证据（算子数、容量、trace 计数）保留为历史，成本结论
  （`total_time_s=566.863`、`tokens/s=1.402`）在新契约下**失效**，登记为 K16/Q12，
  待资源允许时用当前主链重跑后再宣称 48 层代价正确。
- **round16 问题2 的校验位置**：A 段与 BCD 段都从 HF config 派生
  `{num_experts, top_k, intermediate_size}` 后逐项比较；只比较 IR 与 sidecar
  自洽不够，因为 top-k 不进入权重字节，写错的 IR 会让两者一起错。非 MoE 配置
  要求三项全零，防止 parser 把稠密模型写成 MoE 标量。评审建议的「链级反例」
  由 `_validate_moe_identity` 的 top-k/专家宽反例直接覆盖；该调用在
  `step_a_model_ir` 中是无条件的，4L 主链的正例也走同一入口，因此没有再造
  一份需要伪造 parser 子进程的夹具。
- **round16 问题3 的扫描边界**：代码注释只保留 `contracts/dtypes.py` 一处旧口径，
  已改为 `selected_mask` 并加静态测试；设计 §117 的「不能用 `routing != 0`
  兼任命中条件」是否定式说明，实施文档中的历史记录保留原文，不属于需清零的
  现行语义描述。
- **round16 问题4 的计数来源**：§1.3/§4.3/§7.3 的 2076 是 round14 的回归数，
  round15 新增 F54～F58 后没有随用例更新；本轮以实际回归输出回填，并在 §7.3
  保留一行历轮趋势，不再人工累加。
- **round16 全量回归的环境抖动**：第一次运行到 `test_cache_key_lock_serializes_processes`
  时 5 s 多进程队列超时（该用例单跑 5.81 s 通过），且 pytest 打印 summary 后
  卡在 teardown，疑似 fork 子进程残留；这是与 B 路改动无关的时序抖动。第二次
  全量重跑 **2111 passed、11 skipped、52 deselected、exit 0**。期间 root 盘
  （`/dev/sdc2`）一度写满，pytest 临时目录被重定向到
  `/data/fjg/pengcheng/flagOS/.pytest-tmp`；用例集合与默认命令一致，仅临时
  目录位置不同。
- **round17 问题1 的复发原因与扩展范围**：round16 只把 trace 的 `tokens` 显式
  写进 `pim.traceTokens`，重复次数仍由 `_extract_query_count` 从输入形状乘出；
  `SPLIT`/`TRANSPOSE` 的头数维与 rank-1 `GATHER` 没有放进同一批用例，所以
  45/45 全绿没有暴露。本轮把「运行期实际 Tq」变成唯一来源：调度器绑定字段，
  VPU 缺字段直接报错。带符号 `Tq` 的 IR 才随请求增长，全具体形状的旧夹具按
  一次执行，避免把固定规模算子重复 N 遍（I80）。VIEW 类的手写模板
  `loop_count` 仍按特征宽计算，但相位链 trace 没有间接循环、不消费该值；
  rank-1 GATHER 的模板宽改取输出隐藏维，避免把 token 数当特征宽。
- **round17 问题2 的确认口径**：评审所述「四节点模型启动即失败」在当前
  trace 生成调用点不成立——那里传入的是符号形状，末维为符号时旧比较会跳过；
  但根因（单位把符号维固化成占位、两个消费方一个按符号一个按绑定形状各判
  一套）成立：任何按绑定形状复核的消费者都会误拒 MASK/SOFTMAX，编译器侧则
  对历史维完全不做校验。本轮按根因让侧车保留符号维，两仓只比较具体特征维，
  并按 trace 的 `tokens`/`loop_unit_tokens` 解析符号做字节核对。
- **round17 问题3 的复发原因**：round16 把单位只接进 trace 入口与 GeneSim，
  共享验证器仍只看 pimir_path 与哈希；round08 问题4 与 round12 问题1 的
  「新证据只接消费点、不接共享验证器」在同一位置再次复发。本轮把单位完整性
  与符号维角色做成 `audit_sidecar_coverage` 的严格分支，导出、skip-simulation、
  freshness、仿真 IR 与 trace 五个入口自动继承，不再按入口逐个补检查。
- **round17 问题3 对 48 层既有产物的影响**：48L 旧 sidecar 的 19,250 条 B 路
  条目没有单位字段，新检查使其在 freshness、仿真 IR、trace 三入口全部
  REJECTED（不再是只有 trace 拒绝）。按用户要求不重跑 48 层主链，结构/容量
  证据保留为历史，成本证据继续登记 K16/Q12。
- **round17 问题4 的修复边界**：GeneSim 不 import 编译器默认语义（两仓独立，
  运行时加跨仓依赖会把仿真器绑到编译器安装上），因此「执行语义与编译器默认
  一致」在产品层保证：共享 `models/` 用本轮主链产物刷新，并由编译器测试
  `test_checked_in_qwen3_standalone_products_match_the_compiler_default` 钉住
  默认语义、指纹与严格契约；GeneSim 启动预检继续核对 IR/sidecar 语义一致、
  容量三件套与 B 路单位，并在错误信息里指向「重跑主链或用 run 目录配置」。
  模板的 10 请求 trace 会跑完整 8084 token，验证时用同源配置把请求数收成 1。
- **round18 问题1 的复发原因与扩展范围**：round17 问题3 只把「单位存在性与
  符号维角色」接进共享验证器，A/B 分类仍取自 sidecar 自报的 `shards`，等于
  让被验对象决定验什么；同一个错误在六个消费点各写了一遍。本轮把分类收回
  源 IR，并沿同一根因补了三处评审未逐条点名的入口：GeneSim 的放置加载
  （`_load_compiler_placement` 的钉死）、覆盖审计的缺单位判断
  （`_validate_pimir_source_coverage`）与 TP 展开的 pimir 继承
  （`_expand_multi_shard_operators`）。GeneSim 的非严格旧式 sidecar 仍允许
  只写 `dpu_id` 的单 DPU GEMM，这是旧格式的放置表达，不改变 A/B 分类；
  严格产物与 B+C+D 导出按 4L/48L 真实导出格式要求非空 `shards`。
  主链 E 阶段的 `_assert_placement_matches_run` 本来就是按 IR 分类；本轮把
  它之外的五入口补齐，并把同一规则移植到 GeneSim。
  `placement_export.count_sidecar_classes` 仍按「条目是否带分片」做打印统计，
  它只描述导出产物、不参与验收判定，真实产物的分类与源 IR 一致。分类只判
  `op_type == GEMM`，不叠加「有投影角色」：真实 IR 的 q_norm/k_norm 是
  RMSNORM 且带 `semantic_role`，按角色并集会把它们误判成 A 路。
- **round18 问题2 的字段保留取舍**：没有删除 sidecar 的 `trace_tokens`/
  `local_*_shapes`。TP 展开与放置期成本绑定发生在生成 trace 之前，此时还
  没有 pimir 元数据；删掉字段会让 GeneSim 在展开阶段无法判断分片是否与
  编译单位同形。修复把这两个字段降级为「pimir 的同源副本」，由编译器共享
  验证器与 GeneSim 生成 trace 时的 `_validate_unit_against_pimir` 逐值核对。
- **round18 问题2 的 pimir 解析边界**：共享验证器只按位置比较参数/结果的
  特征末维，不要求整秩相等：ROPE 的 pimir 是四维、GATHER 的 token 输入是
  一维、PAGED_ATTN 的 KV 操作数是 `memdesc`（不是张量，按 None 跳过）。
  SPLIT 的 pimir 是纯视图，输出类型记完整输入形状，而 IR 只记一个代表切片，
  所以只比较输入末维；其余算子输出末维都参与比较。旧 pimir 没有
  `pim.traceTokens` 时严格产物直接拒绝，不再回退形状启发式。
- **round18 问题3 的测试取舍**：没有为锁机制引入 `Barrier` 之外的同步协议；
  `Barrier` 只用于让两个子进程同时开始竞争，保证「不重叠」断言有对象。
  慢调度用例固定等待 6 秒以复现旧 5 秒上限的 `_queue.Empty`，是唯一一条
  刻意放慢的用例；60 秒上限只做死锁检测，不影响正常路径耗时。
- **round18 问题4 的声明口径**：history 值（alt/48l）本身仍与各自 summary
  一致，问题在「四份 summary 通过」这句话不成立；没有给 checker 新增
  history 核验模式，因为仓库内没有需要机器核验历史 summary 的调用方。
  CI 用例在 summary 产物不存在时跳过，避免干净检出依赖 `/tmp` 历史目录。
- **round18 问题5 的容量口径**：GeneSim summary 同时有 per-TensorPU 的
  `max_resident_bytes`/`max_utilization` 与 `pim_capacity_metrics.json` 的
  `max_cluster_resident_bytes`/`max_cluster_utilization`；后者才是 design
  §十.5 的集群常驻检查。per-TensorPU 最大值 779,673,600 超过 512 MiB 上限
  是因为「模型常驻按 ClusterPU 校验、KV/激活按 TensorPU 校验」，两者不能
  同式比较，已在三处引用中单列说明。
- **round18 问题6 的守卫边界**：`tests/test_architectures.py` 的微小随机模型
  仍用 `1 << 28` 的小预算，它们不是 8 DPU × 8 GiB 契约入口，也不加载真实
  权重；守卫只扫描两个真实测试文件。契约一旦变化，真实测试经工厂自动跟随，
  手写字面量会由守卫直接报错。
- **round18 的 48 层口径**：按用户约束不重复启动 48 层长任务；本轮分类与
  单位校验改动只影响产物复核，不改变 48 层历史产物结构。48L 旧 sidecar 的
  19,250 条 B 路条目仍缺单位，三入口继续 REJECTED，成本证据继续登记
  K16/Q12；结构/容量证据保留。
- **round18 的探针证据**：编译器共享验证器对 4L 真实 sidecar 的六个探针
  全部 REJECTED（B 路假分片、`trace_tokens=128`、本地宽 384、旧 128-token
  pimir、无 `pim.traceTokens` pimir、设备 GEMM 空分片），基线 4L 产物
  ACCEPTED（`/tmp/r18-probes/compiler-validator.log`）。直接 GeneSim 用同一
  份真实产物做两个探针：给 op3（RMSNORM）加
  假 shard 并删单位 → exit 1「op3（RMSNORM）是非 GEMM 却带分片」
  （`/tmp/r18-probes/probe-a.log`）；把 op1 的 `trace_tokens` 改成 128 →
  exit 1「pimir trace 覆盖 1 个 token，sidecar 编译单位是 128」
  （`/tmp/r18-probes/probe-b.log`）。未篡改的 4L 产物在 freshness、仿真 IR、
  trace 与直接 GeneSim 四个入口均 ACCEPTED。
- **round19 问题1 的 `pim.matmul` 分支**：A 路核对接受 `tt.dot` 与
  `pim.matmul` 两种分块 GEMM 原语。当前 `_measure_kernel_tile_n` 走 linear
  kernel，实产都是 `tt.dot`；K/N 循环嵌套的核对只对 `tt.dot` 生效，
  `pim.matmul`（相位展开后的形态）只核 M/K/N、tile 属性与分片宽度。
  这是为不把另一种合法原语判死，而不是放宽分支：两种原语都必须与
  `kernel_tile_n`、分片本地宽、融合激活逐项一致。
- **round19 问题3 的 `semantic_role` 必填条件**：只有源 IR 该 GEMM 带
  `semantic_role` 时才要求条目必填并与之一致。真实 30B 骨架的每个 GEMM
  都有投影身份（导出侧 `_assert_every_gemm_has_known_role` 保证），合成
  IR/旧夹具没有身份时不应凭空要求；条目多带一个源 IR 没有的角色仍然失败。
- **round19 问题4 的测试取舍**：主机回退的 f16 边界对 f32 KV（只有测试会造）
  也生效，因为这条函数是设备相位表的镜像，设备只有 f16 存储；既有 f32
  用例的容差是 0.05，取整不改变结论。新用例同时关闭 `_compiled_paged_attn`
  与 `_compiled_matmul`，并用独立实现的相位表参考逐元素对拍，而不是只与
  融合内核做 allclose。
- **round19 问题5 的指标命名**：保留 `max_utilization`/`max_tensor_pu_utilization`
  两个键（下游 autotune 数据集与控制台在用），但语义改成「与 fits 同源的
  判定利用率」；新增 `max_total_resident_ratio` 明确是总量比。没有改
  `pim_capacity_max_utilization` 的 summary 键名，避免下游 schema 断裂。
- **round19 的 48 层口径**：按用户约束不重跑 48 层主链与数值。本轮的 A 路
  核对不要求 sidecar 新增字段（48L 旧产物已有 `shards`/`kernel_tile_n`/
  `semantic_role`），因此 A 路本身能过；但 48L 旧产物的 19,250 条 B 路条目
  仍缺编译单位，三入口继续 REJECTED，K16/Q12 状态不变。48 层成本证据继续
  失效，只保留结构/容量证据。
- **round19 问题3 的覆盖范围不完整（round20 复发原因）**：round19 只把条目
  `op_type`/`semantic_role` 与源 IR 绑定，并给 add/mul 加了 kind 核对，没有
  把「pimir 内容本身是不是该算子的实产」定义成通用契约；ROPE、PAGED_ATTN、
  RMSNORM 等 16 类仍可互换。round20 把同一根因收敛成 18 类闭合表，属于
  round19 问题3 的补完，不是新需求。
- **round20 问题1 的 `silu_as_gelu`/`silu_as_mul` 探针不是有效反例**：这两个
  探针改的是 op153，而 op153 在 `absorbed_ops` 里。被吸收节点的 pimir 不参与
  B 路代价（trace 置空，激活由主 GEMM 的 `act_spec` 与融合清单证明），所以
  它们在旧代码上 ACCEPTED 并不能证明 B 路语义缺口。根因验证改用非吸收 SILU
  单元用例与真实 ROPE/PAGED_ATTN 探针；这不是放宽——非吸收的 B 路 SILU 会被
  同一张表拒绝。
- **round20 问题1 的四节点注意力表项按实产入表**：评审要求「先按真实产物
  取样再入表」。本轮用 `_bpath_pimir` 实际编译出 MASK/SOFTMAX/GEMV_SCORE/
  GEMV_CONTEXT 四份产物并据此登记（SOFTMAX 展开为 reduce/exp/reciprocal 链，
  两次 matmul 用 `kvFixedIn`/`kvFixedOut` 的读写端区分得分与上下文），没有
  凭 mnemonic 猜；SOFTMAX/MASK 的块循环标注同时是必需事实。
- **round20 问题2 的 checker 取舍**：容量三值已在证据块 `facts` 里登记并带
  运行语境；本轮把 scope 写进 summary 字段名并加用例，没有新增「facts 对
  `pim_capacity_metrics.json` 逐值复算」的独立输入——`facts` 本来就不在
  summary 指标集内，再加一条平行通道会与现有命名空间重复。
- **round18 问题3 的缓存锁测试修复只改了等待上限（round20 复发）**：本轮
  全量回归再次在 `test_cache_key_lock_serializes_processes` 上等满 60 秒后
  以 `_queue.Empty` 失败，而单跑 2/2 通过。根因是 fork 子进程继承了跑完整套
  回归的父进程线程/队列状态，Queue feeder 线程丢事件；60 秒上限只决定等多久，
  不改变事件会不会丢。两个锁用例改用 spawn + 模块级 worker，worker 自己设置
  缓存目录；这是同一根因的收敛，不是新需求。

### 7.3 回归测试结果
- 执行命令：`python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider --basetemp=/data/fjg/pengcheng/flagOS/.pytest-tmp-r21`
- 执行结果：**2207 passed、11 skipped、52 deselected、2 warnings in 1373.42s**，失败 0，日志
  `/tmp/r20-fix-regression.log`。
- GeneSim：`cd genesim && ./run.sh --test sim` **45/45 个测试文件通过**
  （exit 0，日志 `/tmp/r20-fix-genesim-sim2.log`）；4 层真实权重目标命令
  **36 passed、1 skipped in 242.71s**；4 层全链 `tp4_pp2` dynamic **exit 0、
  60.785 s / 13.079 t/s**、dense **exit 0、63.637 s / 12.493 t/s**；本轮新增
  F83 35 条 compiler 用例与 3 条 genesim 用例、F84 1 条 genesim 用例，
  并修正 4 个测试夹具；F85 把 2 条既有缓存锁用例从 fork 改成 spawn
  （用例数不变，消除全量回归的 `_queue.Empty` 复发）。
- 历轮趋势：round01 通过 1859→1873、round02 →1884、round03 →1897、round04
  →1908、round05 →1926、round07 →1952、round08 →1982、round09 →1999、
  round10 →2005、round11 →2016、round12 →2036、round14 →2076、round15 →2099、
  round16 →2111、round17 →2128、round18 →2149、round19 →2172、round20 →
  2207（round06/13 未单列回归数；历轮失败始终为 0）。

## 八、Qwen3 全链路验收（评审 round01/round02）

### 8.1 入口与命令
```bash
# 0) 环境
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"

# 1) Qwen3 入口（4 层真实权重切片；全模型同命令换 --model-dir）
python scripts/run_full_pipeline.py \
    --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
    --num-stages 2 --max-requests 1
# 仅验证到 sidecar：追加 --skip-simulation
# 执行语义缺省跟随 PIM_MOE_DISPATCH（默认 dynamic_dispatch）；显式指定：
#   --moe-execution {dense_sim,dynamic_dispatch}
# 同一次运行里 A（图骨架）、B（放置 sidecar）、E（仿真 summary）三处独立核对。

# E 阶段的请求分布输入由主链保证：配置引用的 trace 不存在时（干净检出里
# genesim/traces/ 被 .gitignore 排除），prepare_simulation_config 调用
# GeneSim 的 trace_generator（--synthetic --seed 0）在本次运行目录生成，
# 把 trace.path 改写为绝对路径并回读复核；已有 trace 原样使用。

# 1b) 执行期 A/B（同一计划、同一后端、同一权重；只计执行阶段）
python scripts/bench_moe_dispatch.py \
    --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
    --new-tokens 2 --repeat 3 --warmup 1 --num-dpus 8

# 2) 真实 30B 目标用例
QWEN3_30B_MODEL_DIR=/data2/pengcheng-models/Qwen3-30B-A3B \
  python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q
```

### 8.2 每段产物与核对
| 段 | 产物 | 本轮证据 |
|----|------|----------|
| A | `<run_dir>/artifacts/qwen3_moe.ir` | **3681 算子、1557 GEMM**（1536 专家 GEMM + 16 注意力投影 + 4 router + **1 lm_head**）、128 PAGED_ATTN（与编译器 fp16 融合 SDPA 一致）、**128 q_norm + 16 k_norm**（逐头，head_dim=128）、128 专家 top-8、专家宽 768、投影身份 9 类齐全；`parameter_bytes=6,229,628,928` 与 `num_parameters=3,114,814,464`×2 逐字节相等；`model_id=qwen3_moe:4519c683a2b3e1d3`、`model_fingerprint=4519c683…`（schema 2）；7689 条依赖、4 个层子图；`moe_execution=dynamic_dispatch`（与编译配置同源），3072 个专家算子带 `expert_id`（GEMM 1536 + SILU 512 + VECTOR_MUL 512 + MOE_ACCUM 512），成本角色 expert_work 2560 / shared_reduce 512 |
| 0 | `<run_dir>/artifacts/qwen3_moe_tp4_pp2_plan.json` | 8 DPU 两段，各自同 Cluster 快链路 |
| B/C/D | `<run_dir>/artifacts/qwen3_moe_tp4_pp2_placement.json`、`..._placed.ir` | **GEMM 1557 + 算子级 1606**；专家 0..127 与 (层, 角色, 专家号) 全覆盖；tiles [32,128,512]（FlagTree 实测，23 种本地形状）；requires_pimir=true；**3163 条 pimir_path 与 sha256 全部有效**；豁免 518 条（MOE_ROUTE 4 / MOE_ACCUM 512 / 边界 2，全部 host，条目带 `device_hint=cpu` 与角色/专家证据）；融合激活 512 条、被吸收 512 条；`moe_config.execution=dynamic_dispatch`（来自编译配置）；1102 条通信边中 24 条专家边带 `expert_id`（逐层 provenance 修复后）；512 个 VECTOR_MUL 的 pimir 全部为 `pim.eltwise<mul>`（8 个 VECTOR_ADD 为 add）；`ir_fingerprint=4519c683…`；`hw_budget={8 GiB, 64 MiB, align 64}`；**lm_head op3679 落 dpu4..7，local_out=37984、tile_n=512**；1557 份 GEMM 与 1606 份 B 路 pim mlir 的 `pim.dma-align` 全部为 64（修复前 B 路为 8）；**round16 起 1606 条 B 路条目全部回传 `trace_tokens=1` 与 `local_*_shapes`，15 份 B 路 pim mlir 带 `pim.traceTokens`，MLP act/mul 的本地宽为 192（tp4）**；豁免条目带 `device_hint=cpu`/角色/专家号证据，router op150 条目 `device_hint=pim` 且有 pimir |
| A2 | `<run_dir>/artifacts/qwen3_moe_pimir.ir`、`..._oplevel_extensions.json` | **3681 算子**，编号与结构指纹（`4519c683…`）均与骨架一致；sidecar 版本 1.4，3163 个算子桥接 oplevel 成本、518 个保留模板；融合清单 512 条按专家吸收 SILU；**host 累加 `MOE_ACCUM` 首专家 `{Tq:4096}`、其余 `{Tq:6144}`**（选择+加权乘，再加共享累加；字节 `8194/12290`）；**VECTOR_MUL 系数 `{Tq: 768}`、字节 `{Tq: 4608}`**（原 `Tq(Tp+Tq): 6/36`），VECTOR_ADD `{Tq: 2048}`，RMSNORM(hidden) `{Tq: 10240}`，**q_norm/k_norm `{Tq: 640}`**（5×128），**ROPE `{Tq: 384}`**（逐头 3×128），注意力 PAGED_ATTN 保持 `Tq(Tp+Tq)`；全 IR 1994 个当前 token 类非 GEMM 算子、以及全部 1557 个 GEMM 均不携带历史项；精化输入改为 `..._placed.ir`，pimir IR 携带落点证据（pim 1557 / cpu 516 / auto 1608），供直接 GeneSim 核对豁免 |
| E | `<run_dir>/pim_traces_pimir/op_*.pim_trace`、`<run_dir>/results_pimir/summary.json` | trace 来源（顶层口径）`{'pimir': 8858, 'absorbed': 2048}`，无 template/unknown；其中 4 条 lm_head 分片 trace 带 `parent_op_id=3679`、输出 `[Tq, 37984]`（I39 修复后真实展开）；tile 分布 {(128,64):4128, (None,None):2630, (512,6):2048, (32,64):16, (512,64):**20**, (512,32):16}；dynamic_dispatch 下 **total_time_s=60.785、tokens/s=13.079**（round17 把 TRANSPOSE/SPLIT/GATHER 的重复次数改成实际 Tq 后比 round16 的 61.015 s 减少 0.230 s；round18 重跑逐位相同，分类与单位校验不改变计费路径；有效 FLOPs 与通信字节不变）；1606 条 B 路条目带 `trace_tokens` 与保留 `Tq`/`Tp+Tq` 符号的 `local_*_shapes`，15 份 B 路 pim mlir 带 `pim.traceTokens`；summary.model={qwen3_moe:4519c683…, vocab 151936, moe {128,8,768,dynamic_dispatch}}；`parameter_bytes=6,229,628,928` 不受执行语义影响 |
| E-alt | `<alt_run_dir>/pim_traces_pimir/op_*.pim_trace`、`<alt_run_dir>/results_pimir/summary.json` | `--num-dpus 4 --num-stages 4` 的 `tp1_pp4` 运行（本轮 round11 重跑）；E 阶段临时配置绑定 `qwen3_moe_tp1_pp4_placement.json`；**没有 TP 展开**（3699 个算子，tp4 为 16376），op152 单分片 `local_out=768`、trace `output_shapes=[Tq,768]`/`n_extent=768`；lm_head 单分片 `local_out=151936`、trace `[Tq,151936]`；trace 来源 `{'pimir': 2651, 'absorbed': 512}`；**total_time_s=138.589、tokens/s=5.736**；通信字节 **8,218,497,128**，比修复前的 8,352,190,568 少 **133,693,440**（68 个 decode × 1,966,080 B），验证专家输出边按命中并集折算。与 tp4_pp2 的 88.483/8.985 不同，证明仿真消费的是本次 BCD 的产物 |
| E' | dense_sim 对照（独立 `<dense_run_dir>`，同一 4 层模型、同一入口、当前代码） | IR/sidecar/summary 的 `moe_config.execution=dense_sim`；结构指纹 `5c777fae…`（执行语义进指纹）；3163 条 sidecar 的 pimir_path/哈希全部有效；豁免 518 条带落点证据；融合/吸收 512/512 双向闭合；trace 来源 `{'pimir': 8858, 'absorbed': 2048}`；**total_time_s=63.637、tokens/s=12.493**；有效 FLOPs `5,826,198,177,867.654`；summary.model.moe={128,8,768,dense_sim}。两种语义共用同一张图与同一份专家驻留，只有计费不同；精化 IR 的 op155 系数与 dynamic 完全一致（`{Tq:768}`/`{Tq:4608}`），证明成本项不随执行语义变化；dynamic 60.785 → dense 63.637（1.047×）见 §9.2 |

图级分块规模（评审问题5，正式入口 `compile_llama2`）：4 层真实切片 decode/prefill 各 **7048 节点、20924 条命令**，编译 **155.0s**（load 1.5s）；评审查到的默认分块同规模为 1,390,344 节点——默认关闭后回到与原非分块基线同量级。48 层全模型见 8.4。

资源峰值：4 层切片编译期进程 RSS ≈7 GiB；48 层全模型用例编译期进程峰值 RSS ≈ 66 GiB（`ps` 观测），未触发换页。

### 8.3 原理/不变式 → 代码位置 → 证据
| 不变式 | 代码位置 | 证据 |
|--------|----------|------|
| 30B 配置必须建 128 专家/top-8/768，每层专家权重 1207959552 字节、48 层 54 GiB | `genesim/src/ir/model_ir.py`（`build_from_hf_config`、`_validate_moe_contract`） | `tests/sim/test_model_ir.py::TestQwen3MoeBuild` 精确字节断言；4L IR 1536 专家 GEMM |
| 门控链留 host、专家 GEMM 走 DPU，不被写成 dense FFN | `model_ir` 的 MOE_ROUTE/MOE_ACCUM `device_hint=cpu`；`graph/partition.HOST_ONLY` | sidecar 豁免 518 = 4 + 512 + 2 |
| 专家权重按 (层, 角色, 专家号) 唯一定位 | `genesim_bridge/placement_export.py`（`_weight_pattern`、`_role_of_weight`、`_assert_every_gemm_has_known_role`） | sidecar gate/up/down 各 512 条、专家 0..127 全覆盖 |
| pimir 分块来自 FlagTree 实测而非 conf 常量 | `placement_export._measure_kernel_tile_n` | tiles [32,128,512]；trace 的 `(tile_n, k_iterations)` 分布可逐项对应 |
| 严格模式缺项/缺哈希/不可读/哈希不符在生成 trace 前失败 | `gene_sim_scheduler._prepare_pim_traces`、`_operator_pimir` | `TestPimirContractCompleteness` 6 条 |
| sidecar 的 MoE 元数据必须与 IR 一致，防跨模型错配 | `gene_sim_scheduler._validate_moe_config` | 全链路加载通过；trace 元数据覆盖 expert 0..127 |
| 动态分派覆盖分块后的整条专家子图与通信 | `runtime/exec_plan_gen.py`（`_expert_provenance`、`_emit_redistribute`） | `test_moe_dispatch.py` 14 条 |
| 未选中专家的非有限输出不得毒化累加 | `backend/architectures.py`、`graph/spec_prop.py`、`runtime/kernels.py` | `test_*overflow*` 2 条 |
| 参考独立于静态改写 | `backend/architectures.restore_hf_moe_forward` | 30B 用例 fixture 断言；4 层真实对拍通过 |
| 主链命名/角色/配置从注册表派生 | `backend/architectures.architecture_of` | `test_architecture_registry_drives_moe_pipeline`；Qwen3 入口全链路 |
| trace 可逐专家追溯 | `genesim/src/ir/model_ir.py`（`split_operator` 继承 expert_id）、`gene_sim_scheduler._prepare_pim_traces` | 最终 trace 元数据覆盖 expert 0..127 |
| 执行语义由编译配置统一派生，IR/sidecar/summary 三处独立核对 | `runtime/exec_plan_gen.resolve_moe_execution`、`genesim_bridge/placement_export.py`、`scripts/run_full_pipeline.py` | `TestMoeExecutionContract`、`TestMoeConfigLineage`、`step_a_model_ir`/`step_bcd_export`/`step_e_simulate` 三段核对 |
| dynamic_dispatch 的专家计算/通信按 any-hit `1-(1-k/E)^Tq` 计费，共享累加不缩放；dense_sim 保留全量 | `moe_cost_role`（`model_ir`，切分与归约继承）+ `gene_sim_scheduler._moe_expert_fraction`、`_moe_dependency_fraction`；`PIMVPU.execute` 按 `moe_cost_fraction` 折算 trace 时延 | `TestDynamicDispatchCost` 5 条、`TestDynamicDispatchPimLatency` 3 条、管线级语义用例 |
| 依赖边按两端的专家角色折算：专家产出到共享/其它设备/HOST 随源命中并集，共享输入到专家随消费者，shared→shared 全量 | `gene_sim_scheduler._moe_dependency_fraction`（同时接收源与目标）、`dependency_transfer_stats`、`dependency_transfer_bytes` | `TestMoeDependencyDirection` 5 条、`test_dependency_payload_follows_the_expert_endpoint`、`test_expert_host_egress_is_scaled_by_the_hit_union`、`test_expanded_expert_output_edges_keep_the_source_role`、`test_gate_skipped_host_gather_does_not_transfer` |
| 实施文档的数值证据按运行标签绑定 summary，派生结论由同一组指标重算；历史值与外部测量独立命名空间 | `scripts/check_implement_evidence.py`（证据块、`summary_identity`、`_check_derived`） | `tests/test_implement_evidence.py` 13 条；本轮 dynamic/dense/alt 三份 current summary 与 48 层 history 值分栏核对通过 |
| 严格 compiler-placement 的容量验收固定为需求常量，配置与产物同步抬高也会失败 | `pim/capacity.py::DPU_MRAM_BYTES`、`gene_sim_scheduler._validate_hw_budget`、`_validate_pimir_hardware_contract` | genesim 5 条容量用例、compiler 跨仓常量用例；4/16 GiB 同步改值全部拒绝，旧式非严格产物兼容 |
| A/A2/BCD/E 的运行产物按架构/策略/语义/运行号隔离，共享 trace/results 不被删除或读取 | `run_full_pipeline.run_directory`、`prepare_simulation_config`、`step_e_simulate`、`verify_trace_provenance` | `test_simulation_step_never_deletes_shared_outputs`、`test_interleaved_runs_keep_their_own_traces_and_summaries`、`test_trace_provenance_reads_the_configured_trace_dir`；本轮实跑目录含语义与运行号 |
| B 路 pimir 的顶层原语/关键属性必须真是该算子的编译实产（round20 起覆盖 18 类，闭合表） | `run_full_pipeline._bpath_pimir_semantic_problems`、`pimir_trace.bpath_pimir_semantic_problems`、`gene_sim_scheduler._assert_pimir_operator_semantics` | `placement_export` add/mul 用例、`TestPimirOperatorKindIsVerified`、`test_audit_rejects_a_bpath_pimir_with_another_operators_semantics`（7）、`test_bpath_semantic_table_accepts_each_calibrated_product`（18）、`test_bpath_semantic_table_is_pinned_across_repos`；4L/48L/llama2 真实产物 0 问题 |
| 专家乘法/激活成本按实际 768，残差加按 hidden | `cost_extractor._element_width`、`op_classify._binary/_activation` | `test_eltwise_and_activation_costs_follow_the_operator_shape` |
| 未命中的专家命令在提交前就被拦下，不占执行队列 | `backend/hal_numpy.py`（`submit`、`_skipped_event`） | `test_skipped_launch_is_never_submitted`、`test_gate_skipped_launch_without_skip_zero_leaves_the_buffer` |
| 只有输出跨出专家子图的命令才在跳过时清零 | `runtime/exec_plan_gen._skip_zero_nodes` | `test_tiled_dispatch_tokens_match_hf`、`test_tiled_dispatch_skips_expert_communication` |
| 显式严格开关与 sidecar 自报统一成一份硬契约 | `gene_sim_scheduler._strict_pimir` | 新增 8 条正反向用例 |
| 图骨架覆盖完整执行图：lm_head、q_norm/k_norm 与权重逐字节对齐 | `model_ir.build_from_hf_config`、`model_parser`（参数量交叉校验）、`run_full_pipeline.step_a_model_ir` | `TestFullGraphCoverage` 4 条、跨仓权重对拍 3 条；4L IR 3681 算子/1557 GEMM/parameter_bytes 6,229,628,928 |
| 模型级输出头有独立权重、分片与 pimir（含真实骨架的「在最后一层子图内」布局） | `placement_export.export_gemm`、`gene_sim_scheduler._expand_multi_shard_operators` | `test_model_level_lm_head_is_exported_with_its_own_weight`、`test_lm_head_is_expanded_across_tensor_parallel_shards`（真实布局）、4L sidecar op3679 与 trace parent 3679 |
| IR/sidecar/精化 IR/仿真四处用同一结构指纹 | `ModelIR.structural_fingerprint`、`genesim_bridge/ir_fingerprint.py`、`step_a2_refine_pimir_ir`、`_validate_model_identity` | `test_ir_fingerprint.py` 8 条、`TestModelIdentityFingerprint` 9 条；4L 三段产物指纹均为 `4519c683…`（schema 2，含依赖拓扑与层归属） |
| 容量预算只有一个真源（8 DPU × 8 GiB） | `contracts/op_contract.py`、`memory.mem_planner.default_hw_budget`、sidecar `hw_budget`、GeneSim `_validate_hw_budget` | `test_sidecar_carries_the_single_capacity_contract`、`TestHwBudgetContract` 3 条 |
| 逐算子成本宽度跟随真实形状（专家 768 / 残差 hidden / QK 归一化 head_dim） | `op_classify._binary/_activation/_normalize_ir`、`cost_extractor._element_width` | `test_eltwise_and_activation_costs_follow_the_operator_shape`、`test_rmsnorm_cost_follows_the_operator_shape` |
| 成本依赖项由配方声明，当前 token 类只随 Tq、注意力类随 Tq(Tp+Tq) | `op_classify.OpRecipe.cost_term`、`cost_extractor._fit_coeffs` | `test_current_token_costs_ignore_kv_history_and_attention_tracks_it`（Tp=0/128/726） |
| ROPE/MASK 单节点按逐头形状测量，不乘 num_heads | `op_classify._rope_ir`、`_mask_ir` | 同上用例钉住 ROPE 系数 384、MASK 系数 1 |
| pimir 覆盖由源 IR 定义，与最终落点无关 | `gene_sim_scheduler._validate_pimir_source_coverage`、`run_full_pipeline.audit_sidecar_coverage` | `test_missing_op_placed_on_gpu_is_still_rejected`、`test_missing_op_placed_on_cpu_is_still_rejected`、`test_all_ops_on_gpu_are_still_checked`、`test_audit_sidecar_coverage_rejects_missing_device_op` |
| 豁免只允许模型边界与带落点证据的 host 执行（round07 收紧，不再按 `expert_id<0` 推断） | 两侧 `_exemption_is_legitimate`、`placement_export._build_pimir_exempt_ops`、`run_full_pipeline._invalid_exemptions` | `test_unknown_device_op_cannot_be_exempted`、`test_audit_sidecar_coverage_rejects_an_unknown_exemption`、`test_legitimate_exemptions_are_registered`、`test_audit_sidecar_coverage_rejects_an_exempted_router_gemm`、`test_exempted_router_gemm_without_host_evidence_is_rejected` |
| MoE 按 (层, 角色, 专家号) 全覆盖 | `run_full_pipeline._moe_expert_triples`、`audit_sidecar_coverage` | `test_audit_sidecar_coverage_requires_every_moe_expert_role` |
| trace 只接受 pimir/absorbed，且源 IR 应覆盖集合必须全部拿到 pimir | `run_full_pipeline.verify_trace_provenance`、trace 元数据 `parent_op_id` | `test_trace_provenance_rejects_a_missing_required_op`、`test_trace_provenance_accepts_expanded_shards_as_parent_coverage`、`test_trace_without_a_source_field_is_rejected` |
| 结构指纹覆盖依赖端点/张量身份/层归属，精化与放置字段不变 | `ModelIR.structural_fingerprint`、`ir_fingerprint.fingerprint_ir` | `test_fingerprint_tracks_topology_and_layer_ownership`、`test_fingerprint_tracks_dependency_topology_and_layer_ownership` |
| 旧式依赖与显式系数边按边两端的专家角色同口径折算 | `gene_sim_scheduler` 的 `dependency_transfer_bytes` | `test_legacy_dependency_payload_is_scaled_by_target_role`、`test_dependency_payload_follows_the_expert_endpoint` |
| 专家归属限定在本层，层内多专家汇合点终止、跨层激活中性 | `runtime/exec_plan_gen._expert_provenance`、`_node_layer_of` | `test_every_layer_gates_its_own_experts`、`test_pp_stages_each_gate_their_own_layers`；4L 计划每个 dispatch 2060 条门控命令（四层相同） |
| 选择/加权累加是主机共享工作，设备侧只剩可门控专家子图 | `graph/partition.HOST_ONLY`（`ne`/`where.ScalarOther`）、`model_ir` 的 `MOE_ACCUM` | `test_moe_tail_is_host_shared_work`、`test_plan_and_genesim_ir_agree_on_moe_roles`；4L decode 计划 where/ne 设备 launch = 0、fused host 504 |
| 被吸收节点由融合清单 + 主算子 pimir + IR 依赖边唯一证明，且激活是主算子的唯一真实消费者 | `run_full_pipeline._validate_absorbed_ops`、`gene_sim_scheduler._validate_absorbed_ops` | compiler 8 条、genesim 8 条 absorbed/阶段用例；4L 侧车 512 条融合与 512 条吸收双向闭合，48L 6144/6144、Llama2 32/32 |
| 吸收清单的阶段语义：BCD 允许清单先行，A2 后所有复核入口与直接 GeneSim 要求双向闭合 | `validate_sidecar_contract(require_absorbed_closure=)`、`step_bcd_export`、`_validate_absorbed_ops` | `test_audit_accepts_a_manifest_without_absorbed_ids_during_export`、`test_audit_rejects_a_refined_sidecar_without_absorbed_ids`、`test_export_step_validates_the_bcd_phase`、`test_post_refine_entries_require_absorbed_closure`（3）、`test_skip_simulation_requires_the_absorbed_ids`；round09 两仓探针 REJECTED |
| 容量三件套在导出、覆盖审计、trace 验收与直接 GeneSim 用同一规则 | `placement_export._hw_budget_payload`、`run_full_pipeline._validate_sidecar_hw_budget`、`gene_sim_scheduler._validate_hw_budget` | F30 的 15 条用例；4L 侧车 8 GiB / 64 MiB / 64 字节 |
| 专家命中只由 top-k 选中索引决定，路由权重只参与加权 | `backend/architectures.make_static_moe_forward`、`exec_plan_gen._emit_moe_dispatch` | 零权重选中专家 inf/nan 两态对拍、decode/prefill dispatch 读布尔掩码 |
| 豁免必须有导出时写下的 host 落点证据，且与 sidecar 实际放置及 `host_gemms` 身份不矛盾 | `placement_export.export_gemm`、`_build_pimir_exempt_ops`、`run_full_pipeline._invalid_exemptions`、`_placement_contradicts_exemption`、`_host_gemm_evidence_matches`、`gene_sim_scheduler._exemption_is_legitimate` | compiler 9 条、genesim 9 条；4L op151 矛盾放置探针在五入口与直接 GeneSim 全部 REJECTED，真实 4L/48L host 豁免 ACCEPTED |
| 专家身份按 op_id 逐条核对角色/专家号/权重路径 | `run_full_pipeline.audit_sidecar_coverage`、`gene_sim_scheduler._validate_moe_expert_identity` | 同层/跨层互换与权重路径反例；4L 真实互换探针 REJECTED |
| 容量与对齐由同一工厂派生，真实 pimir 属性也必须一致 | `memory.mem_planner.default_site_capacity`、`opcompiler_bridge.pipeline.with_budget`、`placement_export._assert_pimir_hardware_contract`、`gene_sim_scheduler._validate_pimir_hardware_contract` | 默认参数/冲突属性/错误属性用例；4L 1557 条 GEMM 与 1606 条 B 路条目对应的 pim mlir 全为 64 |
| TP 展开与 relay 插入一次重建索引 | `ModelIR.batch_edit`、`gene_sim_scheduler._expand_multi_shard_operators`、`_attach_collective_edges` | 批量/逐条等价、索引只建一次、扫描量不随 add 次数增长；4L trace 元数据与 total_time 与逐条展开一致（round07/08 口径）；48L 展开 5:14:33 → 6.24 s |
| 1 专家/top-1 也是 MoE，且 TP 宽度必须整除专家数 | `exec_plan_gen.is_moe_config`、`backend/architectures._moe_strategy`、`model_ir.build_from_hf_config` | HF 对拍、生成、parser、IR、sidecar 五类用例 |
| 仿真只消费本次 BCD 的 placement；策略/DPU/段数/Cluster/本地形状不符即失败；运行产物按运行号隔离 | `run_full_pipeline.prepare_simulation_config`、`_assert_placement_matches_run`、`run_directory`、`step_e_simulate` | 单测 11 条；4L `tp4_pp2` 与 `tp1_pp4` 实跑各自绑定并得到不同展开结果；共享 trace/results 哨兵与交错执行用例 |
| 导出、skip-simulation、freshness、仿真 IR 与 trace 共用完整产物契约 | `validate_sidecar_contract`、`_validate_sidecar_identity`、`audit_sidecar_coverage` | 五个入口的缺 MoE 元数据/容量/伪造 host/非法吸收/专家互换/指纹反例 |
| 架构切分约束覆盖自动选择、直接段数与 PartitionPlan | `Architecture.strategy_check`、`export_pp_placement.resolve_strategy` | 单专家合法 tp1/非法 tp2 单测与真实主链非法分支；稠密模型行为不变 |
| 启动预检与真实 KV 布局同一公式与对齐 | `serve_pim._kv_bytes_per_dpu`、`memory.kv_layout.build_kv_layout` | 逐字节一致、契约对齐放行、超预算 1 字节拒绝 |
| 编译器侧严格性 = 仿真配置 `require_compiler_pimir` ∪ sidecar 自报，严格时按 8 GiB 契约核对容量三件套 | `run_full_pipeline.validate_sidecar_contract`、`audit_sidecar_coverage`、`_config_requires_pimir` | `test_strict_config_requires_the_capacity_contract`（3）、`test_strict_config_rejects_a_sidecar_with_a_wrong_capacity`（6）、`test_export_step_requires_the_capacity_contract`、`test_skip_simulation_requires_the_capacity_contract`；探针四个失败分支与真实 4L 正例（48L 旧产物容量字段合规，但在 B 路单位契约处被拒绝，见 K16） |
| 正文数值绑定到最近的前置运行标签；前者/后者按块内最近两个不同运行解析；必需值必须在自己的运行语境出现；`facts` 覆盖非 summary 证据值 | `check_implement_evidence._effective_labels`、`_nearest_labels`、`_prose_context_problems`、`_check_comparisons`、`_value_in_run_context`、`_fact_bindings` | `tests/test_implement_evidence.py` 12 条正文归属/比较/facts 用例；同块对调、无标签前言、错哈希证据与错误比较全部失败，当前文档 + dynamic/dense 两份 current summary 通过（alt/48l 为 history 引用） |
| `paged_attn` 每块 PV 乘积先落 f16 再加进 f32 累加器，相位存储边界在 lowering、主机参考与测试中一致；主机路径在 QK、乘 scale 与 PV 乘积三处显式取整，不依赖 `matmul` 是否走编译内核 | `FlagTree/LowerPIMToEmitC.cpp`（`PagedAttnOp` 分支）、`ExpandPhases.stampPagedAttn`、`runtime/kernels._host_paged_attention` | `test_paged_attn_matches_the_online_reference`、`test_paged_attn_multi_block_tail_matches_the_online_reference`、`test_sdpa_host_fallback_rounds_at_the_phase_boundaries`、`paged_attn_emitc.mlir` |
| 块表项合法集合 `0 <= slot < pool_slots`；主机预检、写 KV 与主机兜底共用，设备侧非法项中止 | `runtime/kernels._physical_slot`、`_validate_block_table`、`FlagTree/LowerPIMToEmitC.cpp`（`pim_block_slot`） | `test_physical_slot_*`、`test_block_table_tail_sentinel_is_not_read`、`test_paged_attention_rejects_invalid_slots_before_addressing`、`paged_attn_emitc.mlir` |
| 严格模式下配置键、IR/sidecar 文件与 pimir 产物（可读 + 哈希）是 freshness、仿真 IR 与 trace 的必需输入；trace 验收没有配置不能签发；非严格才允许旧式无 sidecar 用法 | `run_full_pipeline._config_text`、`_strict_sidecar_context`、`_require_simulation_input`、`audit_sidecar_coverage`、`verify_trace_provenance`、三个 verify 入口 | `test_strict_entry_*`（15 条）、`test_sidecar_self_report_requires_the_ir_key`（3 条）、`test_strict_entry_requires_a_valid_pimir_artifact`（15 条）、`test_sidecar_self_report_requires_a_valid_pimir_artifact`（3 条）、`test_trace_entry_requires_a_configuration`、`test_non_strict_entry_keeps_the_no_sidecar_legacy_path` |
| trace 逐条绑定 `(parent_op_id, op_type, pimir_sha256)`；吸收 trace 必须零指令且编号在 sidecar 吸收集合 | `run_full_pipeline._trace_source_lineage`、`verify_trace_provenance` | `test_trace_entry_*` 7 条；真实 4L/48L 产物三入口 ACCEPTED |
| 配置引用的请求分布 trace 缺失时，主链在本次运行目录生成并绑定绝对路径；已有 trace 原样使用 | `run_full_pipeline._generate_synthetic_trace`、`prepare_simulation_config`、`step_e_simulate` | `test_prepare_simulation_config_generates_a_missing_trace`、`test_prepare_simulation_config_keeps_an_existing_trace`；生产 Qwen3 配置隐藏 trace 后生成 `GSTR` 文件 |
| B 路 pimir 的编译单位是「1 token × 该 DPU 的本地宽」；sidecar 回传 `trace_tokens`/`local_*_shapes`，pimir 带 `pim.traceTokens`，运行期按 `ceil(实际 Tq / tokens)` 重复整份 trace | `placement_export._pimir_unit_shapes`/`_mlp_elementwise_local_width`/`_attach_bpath_pimir`、`pimir_trace._declared_trace_tokens`、`gene_sim_scheduler._validate_pimir_unit`、`vpu.PIMVPU._unit_repetitions`、`run_full_pipeline._bpath_unit_problems` | compiler `test_bpath_uses_one_token_and_the_mlp_local_width`、`test_trace_lineage_*`（6）；genesim `TestPimirUnitContract`（6）、`test_vector_trace_token_unit_scales_with_the_query_rows`；4L trace `[Tq,192]` 对应 pimir 768/384 B、VECTOR_MUL 65.244→0.151 s |
| TP 展开出的 B 路分片只在本地形状与 sidecar 编译单位一致时继承同一份 pimir；严格产物缺单位直接失败 | `gene_sim_scheduler._expand_multi_shard_operators`、`_require_pimir_unit`、`_validate_pimir_source_coverage` | `test_bpath_shards_inherit_pimir_only_for_matching_local_shapes`、`test_bpath_shard_shape_conflict_is_rejected_in_strict_mode`；4L 1606 条 B 路条目全部带单位、15 份单位 pimir |
| MoE 身份（专家数/top-k/专家宽）由 HF config 派生后在 A 段与 BCD 段逐项核对；非 MoE 配置要求三项全零 | `run_full_pipeline._configured_moe_identity`、`_validate_moe_identity`、`step_a_model_ir`、`step_bcd_export` | `test_moe_identity_rejects_a_top_k_that_disagrees_with_the_config`、`test_moe_identity_rejects_a_wrong_expert_width`、`test_dense_config_rejects_moe_scalars_and_accepts_an_empty_identity`；4L 主链 top-8/768 正例 |
| B 路 trace 的重复次数只读调度器绑定的实际 Tq；带符号 `Tq` 的 IR 随请求增长，全具体形状按一次执行 | `gene_sim_scheduler._get_operator_with_runtime`、`vpu.PIMVPU._bound_query_tokens`/`_unit_repetitions` | `test_view_embed_and_eltwise_ops_repeat_by_the_bound_tokens`、`test_paged_attn_prefill_and_decode_repeat_by_the_bound_tokens`、`test_unbound_query_tokens_are_rejected_instead_of_guessed` |
| B 路单位形状保留 `Tq`/`Tp+Tq` 符号维；两仓只比较具体特征维，历史维交给块循环、token 维交给重复次数 | `placement_export._attach_bpath_pimir`、`gene_sim_scheduler._unit_shape_conflict`/`_resolved_unit_shape`、`run_full_pipeline._unit_shape_conflict`/`_resolved_unit_shape` | `test_bpath_kv_ops_keep_symbolic_history_dims`、`test_four_node_attention_accepts_prefill_and_decode_history`、`test_four_node_gemv_context_rejects_a_wrong_feature_width`、`test_trace_lineage_accepts_four_node_attention_units` |
| 严格产物的每条 B 路 pimir 必须带完整编译单位且保留符号维；导出、skip-simulation、freshness、仿真 IR 与 trace 同一判定 | `run_full_pipeline._bpath_unit_contract_problems`、`audit_sidecar_coverage`、`gene_sim_scheduler._register_pimir_unit` | `test_strict_entries_require_a_complete_bpath_unit`（3 入口 × 2 形态）、`test_strict_entries_reject_units_that_drop_symbolic_dims`、`test_skip_simulation_requires_a_complete_bpath_unit`；4L 新产物三入口 ACCEPTED、48L 旧产物三入口 REJECTED |
| 独立入口模板引用的共享产物必须与编译器默认语义、结构指纹和单位契约一致 | `genesim/conf/sim_qwen3_moe_pimir.yaml`、`genesim/models/`（主链刷新）、`prepare_simulation_config` | `test_checked_in_qwen3_standalone_products_match_the_compiler_default`；run 目录 `sim_qwen3_moe_pimir_pimir.yaml` 可直接 `./run.sh --config` |
| A/B 路分类只看源 IR：`op_type == GEMM` 走 A 路，其余算子走 B 路；`shards` 是被验产物自报字段，B 路非空分片与严格产物设备 GEMM 空分片直接失败 | `run_full_pipeline._entry_class_problems`/`check_sidecar_entries`/`_bpath_unit_contract_problems`/`_bpath_unit_problems`；`gene_sim_scheduler._validate_entry_classes`/`_require_pimir_unit`/`_apply_compiler_local_shapes`/`_expand_multi_shard_operators`/`_validate_pimir_source_coverage`/`_load_compiler_placement` | F72 的 11 条用例；4L 真实产物 1557 条 GEMM + 1606 条算子级分类不变；假分片/空分片在五入口与直接 GeneSim 全部 REJECTED |
| B 路编译单位的数值以 pimir 实产为唯一真源：`trace_tokens` = `pim.traceTokens`，特征末维 = 对应参数/结果末维（SPLIT 只比输入），缺 `pim.traceTokens` 直接拒绝 | `run_full_pipeline._pimir_unit_facts`/`_pimir_unit_problems`；`gene_sim_scheduler._validate_unit_against_pimir`、`pimir_trace.pimir_unit_facts` | F73 的 10 条用例；4L 1606 条 B 路条目与 15 份 pimir 逐值一致，真实产物 ACCEPTED |
| A 路 GEMM 的 pimir 是该分片的实测分块 GEMM：单一 `tt.dot`/`pim.matmul`、无 `traceTokens`/相位链；M=1，K/N 与 tile 属性、`kernel_tile_n`、分片本地宽、K/N 循环一致；融合 `act_spec` 与清单双向一致 | `run_full_pipeline._apath_unit_problems`/`_apath_facts_problems`、`gene_sim_scheduler._gemm_pimir_problems`、`pimir_trace.pimir_gemm_facts` | F78 的 11 条用例；4L 1557 条 GEMM 与共享 dense/llama2 A 路产物 ACCEPTED；`tampered_allgemm`/`tampered_gemm_pimir`/`tampered_notile` 两仓 REJECTED |
| 条目按 host 跳过 A 路检查必须有源 IR `device_hint=cpu` + `pimir_exempt_ops` 登记 + GEMM 的 `host_gemms` 角色/权重证据；自称 host 缺证据直接失败 | `run_full_pipeline._entry_is_host`、`gene_sim_scheduler._entry_is_host`/`_validate_entry_classes`、`_assert_placement_matches_run` | F79 的 8 条用例；4L/48L 真实产物 `operators` 中 host 条目为 0，不被误判；`tampered_lmhead`/`tampered_hostgemm`/`tampered_router` REJECTED |
| 条目 `op_type` 等于源 IR 类型，设备 GEMM 的 `semantic_role` 是源 IR 投影身份；trace 字节判据按 IR 类型选择；逐元素 pimir kind 与 IR 类型对应 | `run_full_pipeline._entry_class_problems`/`_bpath_unit_contract_problems`/`_bpath_unit_problems`、`gene_sim_scheduler._validate_entry_classes` | F80 的 7 条用例；`tampered_optype` 两仓 REJECTED，真实产物 ACCEPTED |
| 主机分页注意力显式落 f16 相位边界（QK、乘 scale、每块 PV 乘积），不依赖 `matmul` 的返回 dtype | `runtime/kernels._host_paged_attention` | `test_sdpa_host_fallback_rounds_at_the_phase_boundaries`（双编译内核缺席 + 相位表参考逐元素对拍 + 取整可区分断言） |
| 容量利用率与 `fits` 同源：模型常驻按 cluster 时分子是 `request_bytes`；请求期、集群常驻与总量比分开输出 | `genesim/src/pim/capacity.tensor_pu_capacity_metrics`、`genesim/src/main.py` | `test_judged_utilization_matches_the_fits_basis`；4L 产物判定利用率与 `fits=True` 自洽，集群利用率 36.6% |

### 8.4 真实 30B 证据

证据按「源码/产物来源—时间—层数—本轮是否重跑」分栏。当前 round20 工作区
只重跑了 4 层；48 层没有用当前工作区重新生成。round16/17 的 B 路编译单位
契约使 round08/10 的 48 层 B 路 trace 缺少 `trace_tokens`/`local_*_shapes`，
当前严格验证器拒绝这些 trace（31,538 条），因此 48 层结构证据保留为历史、
成本证据在新契约下失效（K16/Q12），不能当作「当前代码重跑过」。

| 证据 | 源码/产物来源 | 时间 | 层数 | 本轮是否重跑 |
|------|----------------|------|------|----------------|
| 48 层数值目标用例 | round06 源码与日志 | round06 | 48 | 否，历史 |
| 48 层主链 A/0/BCD/A2 | round08 当轮源码，systemd `full30b-pipeline-r8-qwen3` | round08 | 48 | 否，历史产物 |
| 48 层主链 E | round10 当轮 GeneSim 代码在 round08 产物上重跑 | round10 | 48 | 否 |
| 48 层既有产物静态复核 | 当前验证器（freshness/仿真 IR/trace + B 路语义指纹） | round20 | 48 | 否；A 路分块 GEMM 与 B 路语义指纹在旧产物上 0 问题，但 B 路单位契约要求 `trace_tokens`/`local_*_shapes` 与符号维，旧产物缺字段，三入口全部 REJECTED（K16） |
| 4 层数值目标用例 | 当前工作区 | round20 | 4 | 是 |
| 4 层全链（dynamic） | 当前工作区 | round20 | 4 | 是 |
| 4 层 dense 对照 | 当前工作区 | round20 | 4 | 是（同一入口 dense_sim） |

- 4 层真实权重目标命令（当前代码，本轮 round20 重跑）：
  `QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py tests/test_qwen3_30b_real.py -q -p no:cacheprovider`
  → **36 passed、1 skipped、2 warnings in 242.71s（0:04:02）**，日志
  `/tmp/r20-fix-4l-tests.log`；8B 无权重，SKIP 记为未覆盖；30B 用例的参考为
  编译后恢复的 HF 原始稀疏前向。用例集合与 round19 相同，结果证明 B 路语义
  指纹与容量 scope 字段没有改变真实权重生成，仍与 HF 逐 token 一致。
- 48 层全模型数值结果（round06 历史，**不是当前代码重新生成**，
  `/tmp/qwen3-review-round06/target48l.log`）：
  **26 passed、1 skipped in 17254.56s（4:47:34）**，`test_qwen3_30b_generate_matches_hf`
  PASS、8B SKIP，systemd 峰值内存 63.1 GiB。round01 历史日志为 23 passed、
  1 skipped（25426.83s）。当前工作区没有重跑 48 层数值；48 层主链见下方
  round08/round10 历史产物，本轮只用当前验证器复核既有产物。
- `--skip-simulation`：round16 当轮跑过同一入口（`/tmp/r16-pipeline-skip.log`，
  退出码 0）；round17 没有重跑整条 BCD/A2，只把新产物交给同一共享验证器：
  `validate_sidecar_contract(..., strict_pimir=True)` 对 4L 新产物通过，对
  剥掉 4818 个单位字段的侧车 REJECTED（`/tmp/r17-probes/sim-stripped.yaml`）。
  单测 `test_skip_simulation_requires_a_complete_bpath_unit` 覆盖
  `--skip-simulation` 的实际调用路径。
- 干净检出 E 阶段探针（round15 问题3）：临时隐藏 `genesim/traces/llama2_7b.trace`
  后直接调用 `step_e_simulate`（消费当前 4L dynamic 产物）→ exit 0；主链在
  `/tmp/r15-clean-e/traces/` 生成 `sim_qwen3_moe_pimir.trace`（`GSTR`，1 个
  请求、726 输入 token、69 输出 token）并把 `trace.path` 绑定为绝对路径；
  trace 来源与基线一致，运行结束后原 trace 已恢复。日志
  `/tmp/r15-clean-e.log`。
- 4 层全链路（含仿真）当前代码（本轮 round20 重跑，`tp4_pp2`，验证 B 路
  语义指纹与容量 scope 字段；校验改动不改变 trace 生成路径）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`
  → dynamic_dispatch 退出码 0（PIPELINE_EXIT=0）；运行目录
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-181818-3093375/`
  （A/0/A2/BCD 在 `artifacts/`，E 在 `pim_traces_pimir/`、`results_pimir/`）；
  E 阶段临时配置带 `scheduler.require_compiler_pimir: true`，依次通过
  freshness（含 pimir 产物可读与哈希）、仿真 IR、完整产物契约（含 B 路
  语义指纹）与逐条 trace lineage（含 B 路单位）核对；trace 来源
  `{'pimir': 8858, 'absorbed': 2048}`、
  10906 条；**total_time_s=60.785、tokens/s=13.079**、有效 FLOPs
  `5,415,241,254,987.654`（重复次数修正不改变 IR 系数）、结构指纹
  `4519c683…`、通信字节 14,692,832,968；1606 条 B 路条目全部带编译单位与
  符号形状模板，15 份 B 路 pim mlir 带 `pim.traceTokens`；容量 `fits=True`，
  `pim_capacity_metrics.json` 的集群常驻
  `max_cluster_resident_bytes=3,140,834,304 / 8,589,934,592`（36.6%）；
  summary 的 `max_utilization` 是判定口径利用率（与 `fits` 同源），
  `max_resident_bytes=779,673,600` 是单 TensorPU 常驻最大值（含该 TensorPU
  的模型分片与请求期数据），`max_total_resident_ratio=1.452` 是该值与单
  TensorPU 上限的诊断比，只报告、不参与 `fits`；请求期利用率
  `max_tensor_pu_request_utilization=0.30%`（分子 1,626,112 B）与
  `max_cluster_utilization=36.6%` 分开输出。日志 `/tmp/r20-fix-pipeline.log`。
- 4 层 B 路成本修正的直接证据（同一 4L 模型、同一 trace 输入，round17 口径）：
  `results_pimir/op_type_metrics.json` 里 TRANSPOSE 3.74e-03 s（round16 为
  0.120 s，7 个 token 都按 1 次计）、SPLIT 4.76e-03 s（round16 为 0.121 s，
  3-D 头优先分片按 224 次重复）、GATHER 4.74e-04 s（round16 只按 1 个 token
  计）；VECTOR_MUL 0.151 s、PAGED_ATTN 6.282 s、GEMM 596.365 s 与 round16
  相同（重复次数本来就是 Tq 或历史块数）；round16 的 61.015 s 因此降到
  dynamic 的 60.785 s（差值 0.230 s）。
  `op_10010_VECTOR_MUL.pim_trace` 的 `input_shapes=[['Tq',192],['Tq',192]]`
  与 `pimir.total_in_bytes=768`/`total_out_bytes=384` 一致。
- 4 层独立入口（round17 问题4）：共享 `genesim/models/` 用本轮主链产物刷新后，
  `cd genesim && ./run.sh --config <同源配置>` **exit 0**。模板请求数是 10
  （8084 token），验证时用同源配置把 `trace.num_requests` 收成 1，与主链 E
  的 1 请求口径一致（`/tmp/r17-standalone/sim_qwen3_moe_pimir_1req.yaml`，
  round18 以当前 GeneSim 代码重跑，日志 `/tmp/r18-standalone.log`，exit 0）；
  summary 的 dynamic 为
  **total_time_s=60.785、tokens/s=13.079**、`moe.execution=dynamic_dispatch`，
  与主链 E 逐位相同，证明独立入口消费的正是刷新后的同一套产物。编译器产品
  测试 `test_checked_in_qwen3_standalone_products_match_the_compiler_default`
  钉住共享产物的执行语义=编译器默认、IR/sidecar 指纹一致与严格契约通过。
- 4 层全链路历史记录（round11 重跑，`tp4_pp2`）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`
  → 退出码 0；运行目录
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261111-190527-2004830/`：
  A/0/A2/BCD 产物写 `artifacts/`，E 的 trace/results/log 写
  `pim_traces_pimir/`、`results_pimir/`；E 阶段临时配置绑定本次 BCD 的
  `artifacts/qwen3_moe_tp4_pp2_placement.json`；
  trace 来源（顶层）`{'pimir': 8858, 'absorbed': 2048}`；**total_time_s=88.483、
  tokens/s=8.985**；summary.moe.execution=dynamic_dispatch；
  summary.model.model_fingerprint=`4519c683…`；`tp4_pp2` dynamic 的通信字节
  14,692,832,968 与修复前逐位相同（原因见 §7.2）。
  本轮的结构化审计（`/tmp/qwen3-review-round11-fix/audit-dynamic.log`）：
  3681 算子、3163 条 sidecar、1536 个专家 GEMM、23 份 pim mlir 全为
  `(dma-align=64, mram-bytes=8 GiB)`、1557+1606 条哈希全部有效、
  tiles `[32,128,512]`、豁免 518 条、`fused_activations` 512 与
  `absorbed_ops` 512 双向闭合并与 IR 主算子签名逐一相等、512 个 VECTOR_MUL
  全为 `pim.eltwise<mul>`、trace 无可疑来源、必需项缺失 0。
  与 round03 的 73.593 s 相比，round05 起把从未展开的 lm_head 真正落到 4 台 DPU
  并生成 pimir trace（+4 条），词表 GEMM 的代价从 GPU roofline 回到 PIM 事件
  调度，因此总时长上升；成本依赖项修正则让非 GEMM 的 FLOPs/字节回到真实几何
  （见 A2）。dynamic 的 summary 有效 FLOPs 合计数 `5,415,241,254,987.654`，比 round03 的
  `5,430,713,177,457.654` 少 `1.55e10`（非 GEMM 历史项虚高被扣除）。
  round11 的容量/运行隔离/证据改动不改变 trace 生成路径：本轮仍是 10906 条
  trace、来源 `{'pimir': 8858, 'absorbed': 2048}`、tile 分布
  {(128,64):4128, (None,None):2630, (512,6):2048, (32,64):16, (512,64):20,
  (512,32):16}，与 round10 相同；dynamic 的 total_time_s、tokens/s 与有效
  FLOPs 逐位相等（88.483 / 8.985 / `5,415,241,254,987.654`）。
- 4 层全链路历史记录（round12 复跑，验证严格容量上下文贯通，`tp4_pp2`）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-stages 2 --max-requests 1`
  → dynamic_dispatch 退出码 0；运行目录
  `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261111-203432-2657277/`：
  A/0/A2/BCD 产物写 `artifacts/`，E 的 trace/results/log 写 `pim_traces_pimir/`、
  `results_pimir/`；E 阶段临时配置带 `scheduler.require_compiler_pimir: true`，
  依次通过 freshness、仿真 IR 与 trace 的严格容量核对；trace 来源（顶层）
  `{'pimir': 8858, 'absorbed': 2048}`；**total_time_s=88.483、tokens/s=8.985**；
  有效 FLOPs `5,415,241,254,987.654`、结构指纹 `4519c683…` 与 round11 逐位
  一致（round12 只改校验与证据检查，不改变数值路径）。
- 同一模型连续跑第二种策略（round08 问题1 的实跑证据，本轮 round11 重跑）：
  `python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L --num-dpus 4 --num-stages 4 --max-requests 1 --log-dir /tmp/qwen3-review-round11-fix/pipeline-alt`
  → 退出码 0；运行目录 `/tmp/qwen3-review-round11-fix/pipeline-alt/qwen3_moe_tp1_pp4_dynamic_dispatch_20261111-191713-2087074`；E 阶段临时配置绑定本次 BCD 的
  `artifacts/qwen3_moe_tp1_pp4_placement.json`；
  sidecar op152 只有 1 个分片、`local_out_features=768`（tp4 为 4 片 192），
  lm_head 1 片 `local_out_features=151936`；trace 的 op152 为
  `output_shapes=[Tq,768]`、pimir `n_extent=768`，lm_head 为 `[Tq,151936]`；
  trace 来源 `{'pimir': 2651, 'absorbed': 512}`；**total_time_s=138.589、
  tokens/s=5.736**；通信字节 **8,218,497,128**，比修复前少
  **133,693,440**（68 个 decode × 1,966,080 B）。结果与 tp4_pp2 的
  88.483/8.985 不同，证明仿真消费的是第二次 BCD 的产物而不是模板里的旧
  sidecar。产物留存 `/tmp/qwen3-review-round11-fix/pipeline-alt/qwen3_moe_tp1_pp4_dynamic_dispatch_20261111-191713-2087074` 与审计输出
  `/tmp/qwen3-review-round11-fix/audit-alt.log`。
- 48 层 TP 展开（当前代码，只读 round06 的 pimir IR 与 sidecar）：**6.24 秒**，
  30961 个 TP 算子 → 130036 个分片、44117 → 143192 个算子、pinned 123844；
  旧实现同一区间耗时 **5:14:33**。4 层展开 ≤1 秒（旧 99/102 秒）。
- 执行计划（decode，4 层真实权重，round05 代码）：`where.ScalarOther`/`ne.Scalar`
  的设备 launch 均为 0，主机命令 1024 条；每个 `moe_dispatch_L` 拦下本层专家
  子图命令 **2060 条，四层相同**（round04 只有第 0 层 97 条）；fused host 累加
  504 条（每层 126 个，覆盖 128 个专家的尾部）。计划探针见
  `/tmp/round05-plan-probe.log`。
- dense_sim 对照（本轮 round20 重跑；同一 4 层模型、同一入口显式
  `--moe-execution dense_sim --log-dir /tmp/r20-fix-dense` → 退出码 0
  （PIPELINE_EXIT=0）；运行目录
  `/tmp/r20-fix-dense/qwen3_moe_tp4_pp2_dense_sim_20261112-182507-3141801/`，
  A/0/A2/BCD 在 `artifacts/`、E 在 `pim_traces_pimir/` 与 `results_pimir/`；
  IR/sidecar/summary 三处 `execution=dense_sim`、结构指纹
  `5c777fae6c9f9f64`；trace 10906 条、来源 `{'pimir': 8858, 'absorbed': 2048}`；
  容量口径 `8 GiB / 512 MiB`，`fits=True`，
  `pim_capacity_metrics.json` 的 `max_cluster_resident_bytes=3,140,834,304 /
  8,589,934,592`（36.6%）；summary 的 `max_utilization` 是判定口径利用率
  （与 `fits` 同源），`max_resident_bytes=779,673,600` 是单 TensorPU 常驻
  最大值（含该 TensorPU 的模型分片与请求期数据），`max_total_resident_ratio`
  是该值与单 TensorPU 上限的诊断比，只报告、不参与 `fits`；
  **total_time_s=63.637、
  tokens/s=12.493**、有效 FLOPs `5,826,198,177,867.654`、通信字节
  15,464,703,688；sidecar 1606 条 B 路条目带编译单位与符号形状模板，
  VECTOR_MUL 0.164 s、PAGED_ATTN 6.282 s。两种语义的
  `parameter_bytes`、专家驻留与精化成本系数相同，只有专家命中计费不同
  （见 §9.2）。
- 48 层全模型历史重跑：2026-11-09 启动的 `full30b-r2`（日志
  `/tmp/test_target_full_30b_round02.log`）被 systemd-oomd 杀死（`Result=oom-kill`），
  登记为**环境阻塞**，不作代码失败；round06 重新执行的数值目标为 26 passed、
  1 skipped（4:47:34）。round06 的主链 `full30b-pipeline-r6-qwen3` 在报告时被
  外部停止、无 summary；round08 当轮源码的 48 层主链见下一条。
- 48 层主链（round08 当轮源码的历史产物，systemd `full30b-pipeline-r8-qwen3`，8 DPU ×
  8 GiB，`tp4_pp2`）：`python scripts/run_full_pipeline.py --model-dir
  `/data2/pengcheng-models/Qwen3-30B-A3B --num-stages 2 --max-requests 1`，运行
  目录 `/tmp/qwen3-review-round08/pipeline48l-r8/qwen3_moe_tp4_pp2/`（结果副本
  `results_pimir/`，主日志 `pipeline48l-r8.log`，systemd 服务 exit 0）。
  - A：**44117 算子、18673 GEMM、18432 专家 GEMM**；128 专家/top-8/768。
  - 0：8 DPU 两段，stage0 dpu0..3、stage1 dpu4..7，Cluster 映射 8 项。
  - B/C/D：全新 sidecar **18673 个 GEMM + 19250 个算子级节点**，23 份 pim mlir
    全部 `pim.dma-align=64`；tiles `[32,128,512]`；豁免 **6194 条**
    （6192 条 `device_hint=cpu` + 2 条模型边界）；`fused_activations` 6144 条；
    `hw_budget={8 GiB, 64 MiB, align 64}`；`moe_config={128,8,768,
    dynamic_dispatch}`；结构指纹 `cf8e71b6…`。容量实测（`report_capacity`）：
    48 层 dpu0..3 权重 8,099,934,464 + KV 983,104 + 激活 1,920,512 + 内核 tile
    155,738,496 = **8,258,576,576 / 8,522,825,728 B**（余 264,249,152 B）；
    dpu4..7 权重 7,633,190,912 + 同三项 = **7,791,833,024 / 8,522,825,728 B**
    （余 730,992,704 B），均在 8 GiB 硬约束内。
  - A2：`qwen3_moe_pimir.ir` **44117 个算子**，编号与结构指纹 `cf8e71b6…` 均
    与图骨架一致；放置 IR 作为精化输入，融合清单 6144 条随 sidecar 带入，
    精化后回写 `absorbed_ops` 6144 条（与融合清单一一对应）。
  - E：round08 的 48 层生成 **130806 条 PIM trace**（TP 展开 30961 个算子 →
    130036 个分片）；本轮在隔离根 `/tmp/round10-fix/genesim-48l` 用当前
    GeneSim 代码重跑 E 阶段（`cd /tmp/round10-fix/genesim-48l && ./run.sh
    --config sim48l.yaml`，exit 0；日志
    `/tmp/round10-fix/genesim-48l-rerun.log`）：130806 条 trace 全部重编、0 复用；
    管线以 48 个 reporting stage、**195676 个算子**启动；trace 来源
    `{'pimir': 106230, 'absorbed': 24576}`，无 template/unknown；
    **total_time_s=566.863、tokens/s=1.402**，请求完成 69 个 output token
    （ttft=518.314 s、mean_tpot=0.714 s）；通信字节 **166,702,607,112**，
    与 round08 逐位相同（`tp4_pp2` 的专家输出边经 host 中转，规则修复不
    改变计费字节）；summary.model=`qwen3_moe:cf8e71b6…`、
    `parameter_bytes=61,064,245,248`、MoE `{128,8,768,dynamic_dispatch}`；
    PIM 常驻上限 8 GiB / 512 MiB（`fits=True`），本地内存服务占比 72.43%；
    结构化审计（`/tmp/round10-fix/audit48l-new.py` → `audit48l.json`）
    ACCEPTED：44117 算子、37923 条 sidecar、18432 专家 GEMM、23 份 pim mlir
    全为 `(dma-align=64, mram-bytes=8 GiB)`、融合/吸收 6144/6144、
    `max_cluster_resident_bytes=8,412,147,712 / 8,589,934,592`。

- 48 层既有产物静态复核（round19 首次复核，round20 补 B 路语义指纹）：用
  当前验证器对 `/tmp/qwen3-review-round08/genesim-full48-r8/models` 的当前
  产物复核：A 路分块 GEMM 规则 **0 问题**（旧产物已有 `shards`/
  `kernel_tile_n`/`semantic_role`，18,673 条 GEMM 条目的 pimir 全部与
  tile/循环/分片宽一致）；round20 的 B 路语义指纹对 15 份 pimir 全部
  **0 问题**（llama2-7B 的 14 份同样 0 问题），说明拒绝原因只剩单位字段
  缺失，不是语义错配；
  freshness、仿真 IR 与 trace 三入口仍全部 **REJECTED**：round08/10 的
  19,250 条 B 路条目没有 `trace_tokens`/`local_*_shapes`（其中 13,106 条在
  吸收/豁免过滤后由单位契约直接抓到），31,538 条 B 路 trace 也因此无法证明
  与运行期形状同源。这是 round16/17 单位契约的预期结果，不是代码回归；
  48 层结构/容量证据保留为历史，成本证据失效，待用当前主链重跑后恢复
  （K16/Q12）。本条不代表当前工作区重新生成了 48 层产物。

- FlagTree 重建与既有产物复核（round14）：在
  `/data/fjg/pengcheng/flagOS/flagOS-installed/flagTree/build/flagtree-cmake`
  执行 `ninja -j 32 triton-opt libtriton.so`（exit 0），把重建的
  `triton/_C/libtriton.so` 原子同步到 FlagTree 与 PyTorch 两个安装前缀；
  `paged_attn_emitc.mlir`、`paged_attn_phases.mlir`、`expand_phases.mlir`、
  `phase_spec*.mlir`、`lower_to_emitc_ops*.mlir` 用安装内 `triton-opt` 与
  `FileCheck` 全部通过。重建只改 lowering 与注释，不改 `phases` 属性，因此
  既有 4L（10906 条 trace）与 48L（130806 条 trace）的 sidecar/pimir 哈希
  保持有效：用当前代码对既有 4L dynamic/dense 与 48L 产物重跑 freshness、
  仿真 IR 与 trace 三入口均 ACCEPTED，不重跑 48 层导出。4 层数值探针与真实
  权重目标命令见 §4.1。
- 8B 证据状态（单列）：本机无 Qwen3-8B 权重，`test_qwen3_8b_real.py` 为
  SKIP，如实记为未覆盖，不作为通过。
- P1-1 证据状态（单列）：用户已确认接受 4 层真实权重的 prefill 1.148×、
  decode 1.202×（旧实现 0.994×），Q7 关闭；本轮不重测，剩余执行后端优化
  另立需求，不阻塞本轮。

## 九、动态分派与性能记录

### 9.1 执行期 A/B（本轮口径）

“稠密模拟”与“动态分派”跑**同一份已编译计划、同一个已装载后端、同一份权重**，
只切换后端配置 `moe_dispatch`；计时只覆盖执行阶段（编译与权装载不计），
每种模式 1 次预热 + 3 次重复取中位数。命令：

```bash
python scripts/bench_moe_dispatch.py \
    --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
    --new-tokens 2 --repeat 3 --warmup 1 --num-dpus 8
```

4 层真实权重结果（`tp4_pp2`，无其他测试负载；round04 代码测得。round05 只把
选择/掩码移到主机、把专家归属逐层补齐，未改提交前跳过与专家子图门控，用户已
接受该口径并关闭 Q7）：

| 配置 | prefill 中位 | decode 中位 | token |
|---|---:|---:|---|
| 稠密模拟（`moe_dispatch=False`） | 17.9414 s | 15.9717 s | [6722, 4849] |
| 动态分派（`moe_dispatch=True`） | 15.6322 s | 13.2858 s | [6722, 4849] |
| 加速比 | **1.148×** | **1.202×** | 完全一致 |

每层跳过 120/128 个专家（top-8），跳过集逐 lane/逐 dispatch 可打印核对。
评审在旧实现上用同一小模型测得的 A/B 为 0.994×（未命中命令仍逐条提交、逐块
清零）；本轮把跳过前移到提交前，未命中的命令不再进入执行队列。两轮使用的
模型不同，数值不作直接可比，仅说明方向与量级。用户已接受 1.148×/1.202×
作为 P1-1 加速验收（Q7 关闭，2026-11-10）；容量契约改为 8 GiB 只影响编译预算，
不改变执行期数据路径，基准脚本已同步改用 `default_hw_budget()`。

### 9.2 加速比边界与遗留

剩余执行期由注意力、门控与视图类等非专家命令主导（小模型上门控命令只占
约 8%），继续提高需要优化通用执行后端，超出本需求根因。用户已接受 4 层真实
权重的 prefill 1.148×、decode 1.202× 作为 P1-1 加速验收（Q7 关闭，2026-11-10）；
执行后端优化另立需求，不阻塞本需求。

每层 128 个专家的加权/累加尾部仍不门控：选择、路由加权与累加在主机闭包里
一体执行（`ne`/`where` 已在 HOST_ONLY），按专家门控需要额外的零初始化语义，
且实测不在剩余热路径上；GeneSim 按 `shared_reduce` 保持全量计费，与执行语义
一致。

GeneSim 侧的 dense/dynamic 对照在 round20 用同一代码重跑
（同一 4 层模型、同一入口；round20 的语义指纹与容量字段不改变计费路径，
dynamic 与 round17/18 逐位相同，dense 与 round17 产物逐位相同）：

| 语义 | total_time_s | tokens/s |
|---|---:|---:|
| dense_sim | 63.637 | 12.493 |
| dynamic_dispatch | **60.785** | **13.079** |
| 比值 | 1.047× | — |

解释：本次请求 prefill 有 795 个 token，任一专家被命中的概率
`1-(1-8/128)^795 ≈ 1`，prefill 的专家计算在两种语义下几乎相同；decode 的
Tq=1 只执行 8/128 个专家，但这部分被内存、注意力与门控链掩盖。round17 的
重复次数修正把 TRANSPOSE 0.120→0.0037 s、SPLIT 0.121→0.0048 s，GATHER 由
只按 1 个 token 计改为 726 行；这些非 GEMM 成本在两种语义下同样存在。
round16 的 61.015 s 降到 dynamic 的 60.785 s（dense 从 63.867 s 降到
63.637 s）；decode 少执行专家带来的绝对收益仍为 2.853 s，两种语义同步下降
后比值仍为 1.047×。旧的 dense 46.627 s / dynamic 39.378 s（1.184×）来自
prefill 按 token-专家对折算（0.0625）的错误口径，更早的 1.029× 来自 B 路
单位错配，均已作废。

依赖边方向修复（round10 问题1）对两种策略的影响不同：`tp4_pp2` 的专家输出
先经过 TP 展开插入的 `ALL_REDUCE` 中转，该中转被放在 host 的同一 VPU 上，
`ALL_REDUCE → MOE_ACCUM` 边因两端同 VPU 不进入 `dependency_transfer_bytes`；
被计费的 `down_proj 分片 → ALL_REDUCE` 边本来就按目标（专家）角色折算，
所以 dynamic 的通信字节 14,692,832,968 与修复前逐位相同、总时长不变。`tp1_pp4` 没有
TP 展开，专家 `down_proj → MOE_ACCUM` 是真实的 PIM→host 边：修复后通信字节
从 8,352,190,568 降到 8,218,497,128，差值正好是 68 个 decode ×
1,966,080 B（512 条边 × 4096 B × (1-8/128)），总时长仍为 138.589 s
（这部分传输不在关键路径上）。

历史数值变化来自三处修复：其一，I39 让 lm_head 真正按 4 台 DPU 展开并生成
pimir trace，词表 GEMM 从 GPU roofline 回到 PIM 事件调度；其二，round04 问题1
修正成本依赖项后，非 GEMM 的 FLOPs/字节不再按 `Tq(Tp+Tq)` 虚高（dynamic 的
有效 FLOPs 比 round03 少 1.55e10）；其三，round16 修正 B 路 trace 的 token
单位与本地宽度。三处都是让产物更接近真实执行几何，而不是性能回退。

修复过程中还发现 TP 展开在 `split_operator` 里丢掉 `moe_cost_role`（I34）：
修复前 dense 与 dynamic 的仿真总时长逐位相同，动态计费根本没有进入事件调度。
`test_lm_head_is_expanded_across_tensor_parallel_shards` 使用真实布局后，两种
语义都覆盖 4 条 lm_head 分片 trace（`parent_op_id=3679`）。
