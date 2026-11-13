# 四维调优搜索：优化开关 × tasklet × DPU × PP

> 文档编号：tune-search4d-20261113
> 创建日期：2026-11-13
> 关联模块：`tune/`、`scripts/tune_pipeline.py`、`examples/tune/quick.yaml`

## 一、目标

在原有「图策略 + 算子参数 + 硬件画像」三层候选空间上增加第四层「编译优化开关」，让一轮调优同时搜索：

| 维度 | 配置位置 | 示例取值 | 作用阶段 |
| --- | --- | --- | --- |
| 编译优化开关 | `space.optimization` | `cse: true / false` | 算子编译器 A/B 两路的 pass 列表 |
| 每 DPU tasklet 数 | `space.op.num_tasklets` | `4 / 8` | 算子编译参数、PIMIR、仿真成本 |
| DPU 数 | `space.strategy.num_dpus` | `4 / 8` | 图切分、放置、算子编译预算 |
| 流水段数（PP） | `space.strategy.num_stages` | `2 / 4` | 图切分；TP 宽度 = DPU 数 / 段数 |

四个维度各取两个值，共 2×2×2×2 = 16 个候选；`search.algorithm: enumerate` 与 `max_evaluations: 16` 一次跑完。

## 二、从配置到仿真的流程

```text
examples/tune/quick.yaml
  │ load_config
  ▼
SpaceConfig(strategy, op, optimization, hardware)
  │ CandidateSpace 展开合法组合
  ▼
16 个 TuneState(strategy, op, optimization, profile)
  │ run_tune(enumerate)
  ▼
每个候选：
  ├─ compile_key = 模型/IR/轨迹 + 策略 + tasklet + 优化开关 + 编译画像 + 源码/工具链
  ├─ 图编译器：按 num_dpus / num_stages 切分并生成放置 sidecar
  ├─ 算子编译器：A/B 两路按 state.optimization 过滤 pass；PIMIR 带 DPU/tasklet
  ├─ 成本精化：同一份硬件契约（含优化开关）生成 cost sidecar
  └─ GeneSim：hardware_contract schema 3 绑定编译键与开关，PIM trace 按 tasklet 并行度计成本
  ▼
rounds.jsonl / best.json / summary.md
```

四层取值任一变化都会改变 `compile_key`，不会复用别的候选的编译产物。

## 三、代码改动

| 文件 | 改动 |
| --- | --- |
| `tune/config.py` | 新增 `OptimizationSpace(enabled, field, values)`；解析 `space.optimization`；生效配置原样落盘 |
| `tune/candidate.py` | `TuneState` 增加 `optimization`；`resolved_optimization(cfg)`；枚举、身份键、邻居、交叉与排序都带上开关 |
| `tune/steps.py` | `compile_key` 与四处 `to_pim_hardware_config` 改用候选自己的开关 |
| `tune/report.py` | 候选状态 JSON 与 `candidate_id` 带上开关位 |
| `tune/sweep.py` | 硬件扫描保留最优候选的开关 |
| `scripts/tune_pipeline.py` | dry-run 每个候选打印 `opt=1111` 四位布尔串 |
| `examples/tune/quick.yaml` | 四维 16 候选的一键配置 |
| `docs/tune-usage.md` | 增加 2.4 节与 3.1 节的空间说明 |

关键结构：

```python
@dataclass(frozen=True)
class OptimizationSpace:
    enabled: bool
    field: str | None      # None 表示不扫描，冻结在顶层 optimization
    values: tuple[bool, ...]
```

`TuneState.optimization` 为 `None` 时沿用顶层 `optimization`，因此旧代码与旧记录不需要改；候选空间展开时一定填入具体值。

## 四、使用方法

先只看 16 个组合，不编译：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python scripts/tune_pipeline.py --config examples/tune/quick.yaml --dry-run
```

输出中每行是一个候选：

```text
合法候选 16 个：
  tp2_pp2 dpu=4 tasklets=4 opt=1011 clock=0.63
  tp2_pp2 dpu=4 tasklets=4 opt=1111 clock=0.63
  ...
```

正式跑 16 轮：

```bash
python scripts/tune_pipeline.py --config examples/tune/quick.yaml
```

产物在 `test-results/tune-quick-4d/`：`effective_config.yaml`、`rounds.jsonl`、`responses.jsonl`、`best.json`、`summary.md`。需要换目录用 `--output-dir`；换搜索算法用 `--algorithm`（此时要给 `search.layer_weights.optimization` 一个正权重，算法才会提案这一层）。

## 五、验证

### 5.1 真实 16 轮

用 `scripts/tune_oneclick.sh examples/tune/quick.yaml` 在干净目录跑完 16 个候选，全部成功，停止原因为 `search_space_exhausted`。真实搜索覆盖：16 次新 pipeline 评估、0 次编译缓存命中、0 次仿真缓存命中、16 次最终映射重放，`real_search=true`。

| 策略 | tasklet 4（秒） | tasklet 8（秒） |
| --- | --- | --- |
| `tp4_pp2 dpu8` | 7.982036 | **7.595039** |
| `tp2_pp4 dpu8` | 14.546637 | 13.803242 |
| `tp2_pp2 dpu4` | 14.546581 | 13.803186 |
| `tp1_pp4 dpu4` | 27.505506 | 26.018721 |

最优候选为 `tp4_pp2_dpu8_tl8_opt1011`，`total_time_s=7.5950386510624055`。每行两个 tasklet 取值在 `cse` 开/关下分数相同，原因见第六节。

随后用 `--evaluate-candidate ... --cache-mode strict` 严格复跑最优候选，强制重新编译并新开仿真尝试目录，分数仍为 `7.5950386510624055`，相对容差 1e-9 内一致。

### 5.2 身份与契约核对

独立脚本逐候选核对通过：

- 16 个四维组合两两不同，16 个 `compile_key` 与 16 个 `sim_key` 两两不同；
- 放置 sidecar 的 `num_dpus`、`num_tasklets`、`optimizations`、策略名与候选状态逐项一致；
- `hardware_contract.json` 的 `compile_key`、硬件五元组、优化开关、策略与记录一致；
- A/B 两路 PIMIR 都带与候选相同的 `pim.num-dpus` 和 `pim.num-tasklets`；
- 仿真 PIM trace 的 `num_tasklets` 与候选一致，GEMM 的有效并行度等于 tasklet 数；
- `best.json` 的分数等于 16 个成功候选的最小分数。

### 5.3 单维变化

同一其余三维下对比 8 对样本：

| 维度变化 | 平均时延变化 | 范围 |
| --- | --- | --- |
| `cse` 开→关 | 0 秒 | 0 |
| tasklet 4→8 | −0.840143 秒 | −1.486785 ～ −0.386998 |
| DPU 4→8 | −9.486760 秒 | −12.958869 ～ −6.208148 |
| PP 2→4 | +9.486816 秒 | +6.208203 ～ +12.958925 |

tasklet 与 DPU 维度对分数有明确影响，PP 段数增加在本负载上变慢，与 `tp1_pp4` 最慢的结果一致。

## 六、当前不足

- 优化开关层一次只扫描一个字段：当前需求是「开关特定优化」，多字段笛卡尔积未实现。
- 示例选 `cse` 而不是 `loop_tiling`：关闭 tiling 后 PIMIR 没有 `pim.tile-*`，当前本地形状推导会读不到输出分块。
- 优化开关在某些 workload 上可能不改变 PIMIR 与分数（当前 quick 的 cse 开/关就是这种情况），但 `compile_key`、侧车开关与候选编号仍然不同，搜索与审计不受影响。
- TP 宽度不作为独立维度，由 `num_dpus / num_stages` 派生，避免出现互相矛盾的 TP/PP 组合。
- 访存周期不随 tasklet 缩放（共享 DMA 的保守口径），tasklet 收益只体现在计算周期。
- 每个候选都要完整跑图编译、算子编译与仿真，16 轮是真实成本；先用 dry-run 确认组合，再用 `--max-evaluations` 控制预算。
