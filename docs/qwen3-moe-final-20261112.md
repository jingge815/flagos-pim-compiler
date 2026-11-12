# Qwen3 MoE 接入最终汇总：文档-代码一致性核对与当前状态
> 文档编号：qwen3-moe-final-20261112
> 创建日期：2026-11-12
> 关联文档：request-qwen3-moe-20261105、design-qwen3-moe-20261105、implement-qwen3-moe-20261105、review-qwen3-moe-20261109-round01 至 review-qwen3-moe-20261112-round20、techdoc-qwen3-moe-20261112
> 代码基线：flagos-pim-compiler `pengcheng@b2f07ca`、genesim `pengcheng@434e147`、FlagTree `feat/ceva-with-paged-attn@6e34f0299`；三仓均含未提交改动，未 commit。

## 一、本文定位

　　本文是 Qwen3 MoE 接入的**当前状态入口**，用于替代散落在需求、设计、实施与二十轮评论文档中的过时表述。过程文档仍然保留，但其角色是历史记录：需求与设计记录当时的决定，实施文档记录逐轮修复，评论文档是每轮的不可变快照。三者与当前代码冲突时，以本文、`techdoc-qwen3-moe-20261112.md` 与当前工作区代码为准。

　　本次整理做三件事：第一，逐份核对文档中可被代码或产物验证的陈述（路径、函数/行号、默认值、测试名、计数、验证结果）；第二，修正或标注过时内容；第三，把当前代码状态、验证证据与遗留问题汇总成本文。整理**只改文档，不改代码，不提交**；也**没有重跑** 4 层全链、GeneSim 套件、编译器全量回归等耗时测试，数值引用最近一次记录（第 20 轮评审与第 20 轮修复自测），并在第四节标明来源与证据等级。

## 二、文档与代码状态核对结论

| 文档 | 定位 | 核对结果 | 本次处理 |
|------|------|----------|----------|
| `request-qwen3-moe-20261105.md` | 历史需求基线 | §2.3 把 GeneSim MoE 计费列为不包含，但该能力后来已交付；P0-5 模型路径 `/mnt/sda/...` 与当前站点不符；§五 48 passed 为历史值；header 只写 pin 263fcda | 加「实施状态（2026-11-12）」注记；修正 §2.3 边界与 §2.1 P0-5 路径注记；§五 标注历史基线；header 补 3ce60b6..daef534 |
| `design-qwen3-moe-20261105.md` | 当前设计（含 §十 契约） | §1.3 仍写 genesim MoE 计费不包含，与 §十 矛盾；§4.7 模型路径过时；§5.1 把已存在的测试文件写成「新增」；§7.1 的 P1 测试名与实际文件不符；§六/§7.3 的 48 passed 未标历史 | 修正 §1.3 边界、§4.7 路径、§5.1/§7.1 测试名；48 passed 标注历史；header 补基线区间 |
| `implement-qwen3-moe-20261105.md` | 当前实施记录（逐轮） | 主体与代码/产物一致；§1.3「无未修复项」未体现第 20 轮修复未经独立复核；header 未链接技术文档 | §1.3 补「第 20 轮修复未经第 21 轮独立评审复核」；header 加关联技术文档 |
| `review-qwen3-moe-*-round01..20.md`（20 份） | 历史评审快照 | 每轮问题与当轮产物一致；其中对旧代码的批评已被后续修复，属于审计轨迹 | 不修改（审计轨迹），由本文与 techdoc 说明其历史属性 |
| `techdoc-qwen3-moe-20261112.md` | 当前技术文档 | 结构、引用与结果抽查一致（评审计数 20/37/29/8 与 20 份报告逐份吻合；抽查的代码行号全部命中）；§4.2 把 round01 的 7,079 节点与当前 7,048 节点混写 | 修正 §4.2 节点数表述并注明两个测量口径 |
| `qwen3-support-20261104.md` | Qwen3 dense 历史文档 | 「当前不足」写 Qwen3 MoE 未适配，已被本需求推翻 | 加状态注记，改写 MoE 条目并指向 techdoc |
| `README.md` 文档索引 | 仓库文档入口 | 未列出任何 Qwen3 文档 | 补 qwen3-support / qwen3-moe-final / techdoc / design / implement 五条 |
| `merge-20261107.md` | 合并历史记录 | 「Qwen3-30B 模型未配置，对应路径检查未运行」已过时 | 加后续状态注记并指向本文与 techdoc |
| `genesim/docs/IR.md` | GeneSim IR 规范 | 代码已有 MoE 字段与 `expert_id`/`moe_cost_role`/`semantic_role`，规范表未记录 | 补 Operator 3 个字段、ModelIR 9 个字段 |
| `docs/vllm-backend-20261023.md` 等 3 份 | vLLM 默认值文档 | 单 DPU MRAM 默认值仍写 4 GiB | 已在容量契约修复轮改为 8 GiB（本次核对确认） |
| 其余 `docs/`（约 140 份） | 其他模块/历史交付 | 不在 Qwen3 需求范围；抽查未涉及 Qwen3 陈述 | 不修改 |

## 三、当前代码状态

### 3.1 仓库与改动

| 仓库 | HEAD | 未提交改动（git diff --stat） |
|------|------|------------------------------|
| flagos-pim-compiler | `b2f07ca`（pengcheng） | 60 个已跟踪文件 `+12,398 / -881`（代码/测试 50、文档 10） |
| genesim | `434e147`（pengcheng） | 21 个已跟踪文件 `+6,975 / -519`（代码/测试 20、文档 1） |
| FlagTree | `6e34f0299`（feat/ceva-with-paged-attn） | 2 个已跟踪文件 `+33 / -12` |

　　另有 8 个新增源文件（`genesim_bridge/ir_fingerprint.py`、`scripts/bench_moe_dispatch.py`、`scripts/check_implement_evidence.py`、`tests/test_implement_evidence.py`、`tests/test_ir_fingerprint.py`、`genesim/conf/sim_qwen3_moe_pimir.yaml`、`genesim/tests/sim/test_moe_execution_cost.py`、`FlagTree/test/Dialect/TritonPIM/paged_attn_emitc.mlir`）与 22 份未跟踪文档（20 份评审 + techdoc + 本文）。改动全部未提交。

### 3.2 已落地的能力与默认值

| 能力 | 代码位置 | 默认值/契约 |
|------|----------|-------------|
| `qwen3_moe` 注册与静态 MoE 前向 | `backend/architectures.py:22`（`make_static_moe_forward`）、`:210`（注册） | `selected_mask` 决定命中，`routing` 只参与加权 |
| 门控链留主机 | `graph/partition.py:55` 起（HOST_ONLY） | `topk`/`softmax`/`sum`/`scatter`/`zeros_like` |
| router 列切与专家并行 | `graph/strategy.py:40`（`ep_width`）、`:84`（`ep_dpu_of_expert`） | `ep_width=1` 等价现状；16 DPU 用 pp4×ep4 |
| 图级分块 | `runtime/compile.py:227`（`graph_tiling_enabled`） | MoE 默认关闭；`PIM_DISABLE_TILE=1` 为显式入口 |
| 动态分派与提交前跳过 | `runtime/exec_plan_gen.py:122`、`:169`；`backend/hal_numpy.py:233` | 层域专家归属；未命中命令不入队 |
| 容量契约 | `contracts/op_contract.py:58`（`DPU_MRAM_BYTES`）；`memory/mem_planner.py:32`、`:44` | 8 GiB / 64 MiB 预留 / 64 字节对齐，单一工厂派生 |
| 严格产物契约 | `scripts/run_full_pipeline.py:742`（`validate_sidecar_contract`）、`:1493`（`audit_sidecar_coverage`） | 五入口共用；严格性 = 配置 ∪ sidecar 自报 |
| A/B 路 pimir 绑定 | `scripts/run_full_pipeline.py:1181`、`:1274`、`:1415`；`genesim/src/pim/pimir_trace.py:732`、`:830` | A 路必须是分块 GEMM；B 路 1 token × 本地宽 + 18 类语义指纹表 |
| GeneSim MoE 骨架与成本 | `genesim/src/ir/model_ir.py:409` 起；`genesim/src/scheduler/gene_sim_scheduler.py:1121` 起 | `moe_execution` 缺省 `dynamic_dispatch`，与编译器同规则 |
| 分页注意力相位与槽号 | `FlagTree/.../LowerPIMToEmitC.cpp:275`、`:3069`；`runtime/kernels.py:1405` | 每块 PV 先落 f16；`0 <= slot < pool_slots` |

### 3.3 文档与代码不一致但未改代码的说明

- `request` 与 `design` 中「48 passed」是 2026-11-05 对 `-k llama2_7b` 慢速子集的历史基线；本轮按约定未重跑该子集，不能当作当前代码结果。
- `design §4.8` 的 `exec_plan_gen.py:498` 是设计阶段的示例行号，当前实现位置以本文 §3.2 与 techdoc §5.3 为准。
- `genesim/docs/ARCH.md` 未描述 MoE（它只讲通用架构）；MoE 字段已补进 `IR.md`，架构级说明以 `design §十` 与 techdoc 为准。

## 四、验证证据（按证据等级）

　　证据等级：**已实测**＝有本次或记录在案的运行命令与结果；**仅代码推理**＝只核对代码路径；**未验证**＝缺当前代码产物或独立复核。

| 项目 | 结果 | 来源/日志 | 等级 |
|------|------|-----------|------|
| 4 层真实权重（架构/30B 目标用例） | 36 passed、1 skipped（8B 无权重 SKIP） in 242.71 s | round20 修复自测 `/tmp/r20-fix-4l-tests.log` | 已实测 |
| 4 层全链 dynamic_dispatch | exit 0，`total_time_s=60.785`、`tokens/s=13.079`；A 3681 算子/1557 GEMM/1536 专家 GEMM；sidecar 3163；23 份 pimir 全 64 B/8 GiB；trace 10906（pimir 8858 + absorbed 2048） | round20 修复自测运行目录 `test-results/full-pipeline/qwen3_moe_tp4_pp2_dynamic_dispatch_20261112-162719-2262904/` | 已实测 |
| 4 层全链 dense_sim | exit 0，`63.637 s / 12.493 t/s` | round20 修复自测（dense） | 已实测 |
| GeneSim 套件 | 45/45 个测试文件通过 | round20 修复自测 `/tmp/r20-genesim-sim.log` | 已实测 |
| 编译器全量回归（排除 llama2_7b） | 2207 passed、11 skipped、52 deselected、0 failed in 1,373.42 s | round20 修复自测 `/tmp/r20-regression.log` | 已实测 |
| 证据一致性 checker | exit 0（「实施文档数值与产物一致」） | round20 修复自测 | 已实测 |
| FlagTree 源码=安装对象 | sha256 `068d2624…`、`518cd1e2…`；`triton-opt` FileCheck exit 0（1 处 `pim_block_slot`、5 处 `pim_f32_to_f16`） | round20 评审 | 已实测 |
| 严格契约探针 | round18–20 的假分片、错单位、A 路指向 B 路产物、host 自报、`op_type`/语义互换等全部 REJECTED；4L 真实产物 ACCEPTED | round20 评审与修复自测 | 已实测 |
| 48 层结构/容量 | round08 历史：44117 算子/18673 GEMM/18432 专家 GEMM；容量 dpu0..3 `8,258,576,576 / 8,522,825,728 B`、dpu4..7 `7,791,833,024 / 8,522,825,728 B` | round08 产物与日志 | 历史 |
| 48 层成本 | round10 历史 `566.863 s / 1.402 t/s`；当前严格契约拒绝旧 B 路条目（缺 `trace_tokens`/`local_*_shapes`），成本证据失效（K16/Q12） | round10 summary、round20 静态复核 | 失效 |
| 48 层数值 | round06 历史 26 passed、1 skipped；当前代码未重跑 | round06 日志 | 历史 |
| 8B 真实权重 | SKIP，未覆盖 | 4 层目标用例 | 未覆盖 |
| P1-1 加速比 | prefill 1.148×、decode 1.202×（历史实测，用户已接受） | 实施文档 Q7 | 历史 |
| A 路 `pim.matmul` 形态 | 只有代码路径核对，无真实产物样例 | techdoc §4.9 | 仅代码推理 |
| 第 20 轮修复（18 类语义指纹表等） | 修复者自测通过；未进入第 21 轮独立评审 | round20 fix.log | 待独立复核 |

## 五、本次清理的过时信息清单

| 位置 | 过时内容 | 处理 |
|------|----------|------|
| `request §2.3` | 「不包含：genesim 的 MoE 稀疏计费（另立需求）」 | 删除该不包含项，注明 round01 后追加到设计 §十并交付 |
| `request §2.1 P0-5`、`design §4.7` | 模型路径 `/mnt/sda/butj/fjg/model/Qwen3-30B-A3B` | 改为 `paths.json` 的 `qwen3_30b_model_dir`（本工作区 `/data2/pengcheng-models/Qwen3-30B-A3B`） |
| `request §五`、`design §六/§7.3` | 48 passed 写成当前基线 | 标注为 2026-11-05 历史值、本轮未重跑 |
| `request/design header` | 只写基线 263fcda | 补 Qwen3 MoE 实际区间 3ce60b6..daef534 |
| `design §1.3` | genesim MoE 计费列为不包含，与 §十 矛盾 | 改为已纳入交付 |
| `design §5.1` | 把 `test_moe_dispatch.py`/`test_moe_ep.py` 写成「新增」 | 改为「修改」并列出两个实际文件 |
| `design §7.1` | P1 测试名 `test_expert_parallel::test_16_dpu_tokens_match_tp` 等不存在 | 改为实际用例：`test_moe_dispatch.py::test_gate_skipped_launch_runs_no_kernel`、`::test_dispatch_skips_and_matches_hf`、`test_moe_ep.py::test_ep_16_dpu_generate_matches_hf` |
| `qwen3-support` header 与「当前不足」 | Qwen3 MoE 未适配 | 加状态注记，改写为已接入并指向 techdoc |
| `implement §1.3` | 「无未修复项」未说明第 20 轮未经独立复核 | 补复核状态说明 |
| `techdoc §4.2` | 7,079 节点与当前 7,048 节点混写 | 注明 round01 探针 7,079、当前非分块基线 7,048 |
| `merge-20261107.md` | 「Qwen3-30B 模型未配置，对应路径检查未运行」 | 加后续状态注记并指向本文与 techdoc |
| `README 文档索引` | 无 Qwen3 条目 | 补 5 条 Qwen3 文档链接 |
| `genesim/docs/IR.md` | 缺 MoE 字段 | 补 Operator `semantic_role`/`expert_id`/`moe_cost_role` 与 ModelIR 9 个字段 |

　　评论文档中的旧问题描述属于不可变审计轨迹，不删除；其结论以对应后续轮次的修复与本文第四节为准。

## 六、遗留问题与未覆盖项

| 编号 | 问题 | 影响 | 状态 |
|------|------|------|------|
| K16/Q12 | 48 层旧 B 路 sidecar 缺 `trace_tokens`/`local_*_shapes`，当前严格验证器拒绝 | 48 层成本证据（566.863 s/1.402 t/s）失效 | 未闭环；需用当前代码重跑 48 层 A/BCD/A2/E |
| Q4 | 48 层数值为 round06 历史结果（26 passed、1 skipped） | 当前代码未重跑 48 层数值 | 未覆盖 |
| — | 8B 真实权重无权重 | 8B 未覆盖 | SKIP |
| — | P1-1 加速比未重测 | 1.148×/1.202× 为历史实测 | 用户已接受 |
| — | A 路 `pim.matmul` 无真实产物 | 该分支仅代码推理 | 待验证 |
| — | TP×EP 复合放置未实现 | 仅 PP×EP（tp=1） | 设计已记录 |
| — | 第 20 轮修复未经独立评审 | 18 类语义指纹表等为修复者自测 | 待复核 |
| — | 根文件系统曾因 /tmp 占满触发 ENOSPC | 长跑需 `--basetemp` 指向 /data；历史探针已归档到 `/data/fjg/pengcheng/review-evidence-archive-20261112` | 环境已恢复（可用约 44 GB） |

## 七、复核命令（可重跑）

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"

# 4 层真实权重数值
QWEN3_30B_MODEL_DIR=/data2/pengcheng-work/Qwen3-30B-A3B-4L \
  python -m pytest tests/test_architectures.py tests/test_qwen3_8b_real.py \
                   tests/test_qwen3_30b_real.py -q

# 4 层全链（图编译 → 算子编译 → GeneSim）
python scripts/run_full_pipeline.py --model-dir /data2/pengcheng-work/Qwen3-30B-A3B-4L \
  --num-stages 2 --max-requests 1

# GeneSim 套件
cd ../genesim && ./run.sh --test sim

# 编译器全量回归（慢，约 23 分钟；basetemp 指向 /data 避免根盘占满）
python -m pytest tests/ -q -k "not llama2_7b" -p no:cacheprovider \
  --basetemp=/data/fjg/pengcheng/.pytest-tmp

# 文档数值与产物一致性
python scripts/check_implement_evidence.py --doc docs/implement-qwen3-moe-20261105.md \
  --summary dynamic=<run>/results_pimir/summary.json \
  --summary dense=<dense_run>/results_pimir/summary.json
```

　　注意：`-k llama2_7b` 慢速子集（历史约 2.5 小时）按用户约定不重跑；CEVA 流水线不在本需求范围。

## 八、文档权威关系

```
request（历史需求基线，含实施注记）
   └─ design（当前设计 + §十 契约，已按代码修订）
        └─ implement（逐轮实施记录，含验证命令与证据）
             ├─ review-round01..20（不可变审计轨迹）
             └─ techdoc（技术文档：原理、证据链、遗留问题）
                  └─ qwen3-moe-final（本文：当前状态入口与文档核对结论）
```

　　阅读顺序建议：先本文第三节与第四节了解当前状态，再按需查阅 techdoc 的证据链细节、design §十的契约原文、implement §8 的逐项证据，以及对应轮次的评审文档。
