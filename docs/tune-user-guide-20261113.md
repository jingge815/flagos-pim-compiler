# PIM 编译调优用户手册

　　本手册说明如何使用本仓库的编译调优入口，把「搜索空间配置、图编译、算子编译、成本精化、GeneSim 仿真、结果记录」串成一轮完整的调优。读者可以据此决定要搜索哪些参数、每个参数能取什么值、如何写配置文件，以及如何用一条命令跑完并查看结果。

　　手册先介绍整体流程和一键入口，再逐一说明当前支持的全部搜索空间，最后给出配置示例、结果解读、验证边界和常见报错。第一次使用时，建议直接按第三章运行默认配置；需要定制搜索空间时，再阅读第四至第九章。

## 一、手册用途与阅读方式

　　本手册面向两类读者。第一类是只想快速得到一轮调优结果的用户，关心如何用一条命令跑完 16 个候选并拿到最优配置。第二类是要自行设计搜索空间的用户，关心哪些参数可以搜索、每个参数允许什么取值、如何写 YAML、如何确认组合合法。

　　两类读者都从第三章的快速开始入手即可。快速开始使用仓库自带的默认配置，不需要改任何文件。需要增删搜索维度时，再对照第四至第九章修改配置；需要解释结果或排查错误时，阅读第十三至第十五章。

## 二、调优流程总览

　　一轮调优从一个 YAML 配置开始。配置先被解析成候选空间，再逐个候选执行图编译、算子编译、成本精化和 GeneSim 仿真，最后把每个候选的分数写入轮次日志并选出最优候选。整体流程如下图所示。

```text
调优配置
  │ 解析搜索空间
  ▼
候选列表（每个候选包含图策略、算子参数、优化开关、硬件画像）
  │ 逐个候选执行
  ├─ 图编译：按 DPU 数与流水段数切分计算图，生成放置 sidecar
  ├─ 算子编译：A/B 两路按候选开关过滤 pass，生成带 DPU/tasklet 的 PIMIR
  ├─ 成本精化：按同一份硬件契约生成成本 sidecar
  └─ GeneSim 仿真：加载 PIM trace，计算 total_time_s
  ▼
轮次日志、最优候选、摘要
```

　　每个候选的四层取值都会进入编译键。只要其中一层变化，编译键就变化，该候选会重新编译，不会复用其他候选的产物。仿真开始前，调优器还会核对放置 sidecar 与仿真契约里的硬件参数、优化开关、图策略、编译键和产物摘要，任何一项不一致都会直接失败，避免把不同候选的产物混在一起。

## 三、快速开始：一键调优

　　默认配置是 `examples/tune/quick.yaml`，它同时搜索四个维度，共 16 个候选。在仓库根目录执行如下命令即可完成一轮调优。

```bash
scripts/tune_oneclick.sh
```

　　脚本按顺序做三件事：先从 `paths.json` 找到并激活 Python 环境，再执行一次 dry-run 检查 16 个候选组合，最后正式跑图编译、算子编译、成本精化和 GeneSim 仿真。结果默认写到 `test-results/tune-quick-4d/`。

　　如果只想看候选组合、不启动编译和仿真，可以加 `--dry-run`：

```bash
scripts/tune_oneclick.sh --dry-run
```

　　dry-run 会逐行列出候选，并在组合非法时给出拒绝原因。确认组合无误后，再去掉 `--dry-run` 执行正式调优。脚本支持的参数如下表所示。

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `--config PATH` | 指定调优配置，默认 `examples/tune/quick.yaml` | `--config examples/tune/strategy.yaml` |
| `--output-dir DIR` | 指定输出目录；不填时用配置里的 `output.dir` | `--output-dir /data/fjg/tmp_test` |
| `--dry-run` | 只检查候选组合，不编译不仿真 | `--dry-run` |
| `--help` | 显示帮助 | `--help` |

　　位置参数仍然兼容：第一个位置参数是配置，第二个位置参数是输出目录。例如：

```bash
scripts/tune_oneclick.sh examples/tune/strategy.yaml /tmp/my-tune
```

　　不使用一键脚本时，等价的手工命令是先激活环境，再调用统一入口：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python scripts/tune_pipeline.py --config examples/tune/quick.yaml
```

## 四、当前支持哪些搜索空间

　　当前调优器支持的空间可以分成三组：第一组是外层候选空间，包含图策略、算子参数、编译优化开关和硬件画像四层；第二组是外层搜索算法；第三组是由专门入口消费的内层放置搜索、硬件参数扫描和代价模型校准。各组空间如下表所示。

| 搜索空间 | 配置段 | 搜索内容 | 使用方式 |
| --- | --- | --- | --- |
| 外层候选：图策略 | `space.strategy` | DPU 数、流水段数（TP 宽度派生） | 多个值参与枚举 |
| 外层候选：算子参数 | `space.op` | 每 DPU 的 tasklet 数 | 多个值参与枚举 |
| 外层候选：编译优化开关 | `space.optimization` | 四类优化中的某一个开关 | 多个值参与枚举 |
| 外层候选：硬件画像 | `space.hardware` | 20 个硬件字段 | 多个值参与枚举 |
| 外层搜索算法 | `search` | 枚举或随机算法、预算与权重 | 决定候选如何被挑选 |
| 内层放置搜索 | `placement` | 算子到 DPU 的放置方案 | 仅非 CEVA 画像可用 |
| 硬件参数扫描 | `sweep` | 沿硬件字段逐点扫描 | 由扫描入口消费 |
| 代价模型校准 | `calibration` | 实测报告与结果成熟度 | 仿真阶段消费 |

　　需要说明的是，默认配置只用到前四类中的一部分取值；调优器本身支持表中列出的全部空间，用户可以按第五至第九章自行组合，不限于默认的 16 个候选。

## 五、外层候选空间

　　外层候选空间由四层组成：图策略、算子参数、编译优化开关和硬件画像。每一层可以写一个或多个取值；写多个值时参与枚举，只写一个值时该层实际被冻结。四层的组合数是各层取值数的乘积，候选总数在 dry-run 中给出。

### 5.1 图策略：DPU 数与流水段数

　　图策略决定计算图如何切分到多个 DPU 上。它有两个配置字段：`space.strategy.num_dpus` 控制 DPU 数量，`space.strategy.num_stages` 控制流水段数（PP）。TP 宽度不需要单独配置，由 DPU 数除以流水段数得到。

　　DPU 数必须是 2 的整数次幂，并且不能超过硬件画像的物理簇总数。流水段数必须能整除 DPU 数；由此得到的 TP 宽度也必须是 2 的幂。此外，层数要能被流水段数均分，注意力头数、KV 头数、中间维和词表大小都要能被 TP 宽度整除。这些约束不满足时，dry-run 会把组合列为非法并给出原因。

　　图策略字段的含义与约束如下表所示。

| 字段 | 含义 | 可以取什么值 | 约束 |
| --- | --- | --- | --- |
| `space.strategy.num_dpus` | 逻辑 DPU 数 | 2 的幂，如 1、2、4、8、16 | 不超过画像的物理簇总数 |
| `space.strategy.num_stages` | 流水段数（PP） | 能整除 DPU 数的正整数 | 段数整除 DPU 数；TP 宽度是 2 的幂；层数与模型维度满足整除关系 |

　　默认配置中的两个 DPU 数和两个流水段数会派生出四种策略，如下表所示。

| DPU 数 | 流水段数 | TP 宽度 | 策略名 |
| --- | --- | --- | --- |
| 4 | 2 | 2 | `tp2_pp2` |
| 4 | 4 | 1 | `tp1_pp4` |
| 8 | 2 | 4 | `tp4_pp2` |
| 8 | 4 | 2 | `tp2_pp4` |

### 5.2 算子参数：tasklet 数

　　`space.op.num_tasklets` 控制每台 DPU 上的 tasklet 数。它会被写进 A 路和 B 路 PIMIR 的模块属性，参与算子编译，并进入 GeneSim 的矩阵算子成本模型。取值是正整数，常用值为 1、2、4、8。

　　tasklet 数的配置方式和作用如下表所示。

| 项目 | 说明 |
| --- | --- |
| 配置字段 | `space.op.num_tasklets` |
| 取值 | 正整数列表，如 `[4, 8]` |
| 约束 | 要与目标硬件的每 DPU tasklet 数一致 |
| 影响 | A/B 两路 PIMIR、算子编译缓存键、矩阵算子计算周期 |

### 5.3 编译优化开关

　　`space.optimization` 用来把某一个编译优化开关变成搜索维度。它由两个字段组成：`field` 指定要扫描的开关，`values` 给出该开关的候选布尔值。一次只扫描一个开关，其余三个开关始终取顶层 `optimization` 的值。

　　当前可以扫描的开关如下表所示。融合、相位展开和契约校验属于目标格式的硬性要求，不受这四个开关控制。

| 开关 | 控制的 pass | 说明 |
| --- | --- | --- |
| `constant_fold` | `-pim-constant-fold` | 常量折叠 |
| `cse` | `-cse`、`-pim-cse` | 公共子表达式消除；开和关都能编过，建议作为搜索维度。某些负载上没有可消除的公共子表达式，两个候选的 PIMIR 和分数会相同，但编译键仍然不同 |
| `dce` | `-pim-dce` | 死代码消除 |
| `loop_tiling` | `-pim-tile-to-budget` | 按 WRAM 预算分块；关闭后 PIMIR 没有 `pim.tile-*`，当前本地形状推导会读不到输出分块，因此不建议作为搜索维度 |

　　一个开启该层的例子如下所示。

```yaml
space:
  optimization:
    enabled: true
    field: cse
    values: [true, false]
```

### 5.4 硬件画像字段

　　`space.hardware.fields` 用来把硬件画像里的字段变成搜索维度。每个字段给出一个取值列表，调优器会为每个取值派生一份合法画像，再与其他层做笛卡尔积。取值列表为空或 `enabled` 为 `false` 时，该层不参与搜索。

　　当前可以搜索的硬件字段如下表所示。每个字段的每个取值都会在加载配置时尝试派生画像，派生失败会立即报错。

| 字段 | 含义 | 取值类型与示例 |
| --- | --- | --- |
| `num_pim_devices` | PIM 设备数 | 正整数，如 `[1, 2]` |
| `clusters_per_device` | 每设备物理簇数 | 正整数，如 `[32, 64]` |
| `tensors_per_cluster` | 每簇 TensorPU 数 | 正整数，如 `[1, 2]` |
| `cluster_memory_bytes` | 每簇容量（字节） | 正整数，如 `[8000000000]` |
| `sys_reserve_bytes` | 系统预留（字节） | 非负整数，如 `[0, 1073741824]` |
| `wram_bytes_per_dpu` | 每 DPU 的 WRAM（字节） | 正整数，如 `[65536, 131072]` |
| `dma_align` | DMA 对齐（字节） | 2 的幂，如 `[32, 64, 128]` |
| `clock_period_ns` | 时钟周期（纳秒） | 正数，如 `[0.63, 0.5]` |
| `multipliers` | 乘法器数量 | 正整数，如 `[4, 8]` |
| `cycle_costs` | 各原语周期表 | 映射；通常给完整周期表，这里只列三项作示例 `[{"LOAD": 12, "STORE": 10, "MAC": 1}]` |
| `zone_cycle_factors` | 分区周期系数 | 映射；通常给完整系数表，这里只列三项作示例 `[{"activation": 2.0, "kv": 2.0, "weight": 1.0}]` |
| `tile_size` | 分块大小 | 正整数，如 `[16, 32]` |
| `mac_width` | MAC 位宽 | 正整数，如 `[8]` |
| `chunk_size` | 批处理块大小 | 正整数，如 `[16]` |
| `max_bulk_transfer_bytes` | 单次批量传输上限 | 正整数，如 `[16320]` |
| `use_long_loop_counts` | 是否使用长循环计数 | 布尔，如 `[true, false]` |
| `inter_cluster_communication` | 跨簇通信方式 | 字符串，如 `["direct", "shared_l2"]` |
| `host_relay_overhead_ns` | 主机中继开销（纳秒） | 非负数，如 `[100.0]` |
| `links` | 链路参数 | 链路字典列表，每项含名称、源域、目的域、带宽、基础延迟与三个布尔位 |
| `cpu_model` | CPU 侧模型参数 | 完整映射，如 `[{"base_latency_us": 10.0, "peak_flops": 1000000000000.0, "peak_bandwidth_gbs": 100.0}]` |

### 5.5 组合数、冻结与预算

　　四层的组合数等于各层取值数之积。例如两个 DPU 数、两个流水段数、两个 tasklet 数和两个优化开关相乘，得到 16 个候选。候选总数可以用 dry-run 查看，输出中每个候选会带上派生出的策略名和四位开关标记。

　　`search.max_evaluations` 控制一轮最多评估多少个候选。使用 `enumerate` 时，该值至少要等于合法组合数；小于组合数时只会跑前几个候选。使用随机算法时，该值是搜索预算，不一定等于组合数。

　　只搜索一个维度时，其余层各写一个值即可。只写一个值的层在效果上被冻结，不会产生额外候选。

## 六、外层搜索算法

　　外层搜索算法决定候选如何被挑选。小空间用 `enumerate` 按顺序完整枚举；空间较大时可以用模拟退火、遗传算法或禁忌搜索，由算法在层之间提案新的候选。无论使用哪种算法，每个候选仍要走完整的编译和仿真流程。

　　外层搜索的配置字段如下表所示。

| 字段 | 可以取什么值 | 说明 |
| --- | --- | --- |
| `search.algorithm` | `enumerate`、`simulated_annealing`、`genetic_algorithm`、`tabu_search` | 小空间建议用 `enumerate` |
| `search.seed` | 非负整数 | 随机算法的种子 |
| `search.max_evaluations` | 正整数 | 最多评估多少个候选，包含基线 |
| `search.max_time_s` | 非负数，0 表示不限 | 墙钟预算，只限制发起新评估 |
| `search.max_proposals` | 正整数或空 | 提案数上限 |
| `search.layer_weights` | 策略、算子、硬件三层必填，优化开关层可选 | 各层被提案的权重；权重大于 0 才会提案该层 |
| `search.algorithm_params` | 映射 | 算法自己的参数；`enumerate` 不接受参数 |

　　三种随机算法的参数如下表所示。未写出的参数使用默认值。

| 算法 | 参数 | 类型与约束 | 默认值 |
| --- | --- | --- | --- |
| `simulated_annealing` | `initial_temperature_ratio` | 有限正数 | 0.05 |
| | `minimum_temperature_ratio` | 有限正数，且小于初始温度比 | 1e-4 |
| | `cooling_rate` | 大于 0 且小于 1 | 0.90 |
| | `steps_per_temperature` | 正整数 | 4 |
| | `patience` | 正整数或空；也可写别名 `stagnation_limit`，两者只能给一个 | 空 |
| | `min_improvement` | 非负数 | 0.0 |
| `genetic_algorithm` | `population_size` | 正整数 | 16 |
| | `elite_size` | 非负整数，且小于种群大小 | 2 |
| | `tournament_size` | 正整数 | 3 |
| | `crossover_rate` | 大于等于 0 且小于等于 1 | 0.80 |
| | `mutation_rate` | 大于等于 0 且小于等于 1 | 0.20 |
| | `stagnation_generations` | 正整数 | 20 |
| | `min_improvement` | 非负数 | 0.0 |
| `tabu_search` | `neighborhood_size` | 正整数 | 8 |
| | `tabu_tenure` | 正整数 | 5 |
| | `stagnation_iterations` | 正整数 | 30 |
| | `aspiration` | 布尔 | true |
| | `min_improvement` | 非负数 | 0.0 |

　　如果希望随机算法也提案编译优化开关，必须在 `search.layer_weights` 中写 `optimization`，并且权重为正。否则这一层只会在完整枚举时展开。

## 七、内层放置搜索

　　内层放置搜索决定每个算子落到哪个 DPU 和哪个 TensorPU。它与外层候选空间相互独立：外层先确定图策略、算子参数、优化开关和硬件画像，GeneSim 再在该候选内部搜索放置方案，并用真实 pipeline 验证最终映射。

　　CEVA 画像是一对一映射，不支持内层放置搜索，必须把 `placement.enabled` 设为 `false`。通用 PIM 画像可以开启该功能。配置字段如下表所示。

| 字段 | 可以取什么值或有什么约束 | 说明 |
| --- | --- | --- |
| `placement.enabled` | 布尔；CEVA 画像必须为 `false` | 是否开启内层放置搜索 |
| `placement.algorithm` | `simulated_annealing`、`genetic_algorithm`、`tabu_search` | 不支持 `enumerate` |
| `placement.evaluation_backend` | `pipeline`、`predictor` | 搜索时用真实 pipeline 还是预测器 |
| `placement.validation_backend` | 只能是 `pipeline` | 最终排序必须用真实 pipeline 重放 |
| `placement.objective_metric` | `total_time_s`、`system_efficiency_utility` | 后者只允许评估与验证都用 pipeline |
| `placement.objective_weights` | 必须定义 `compute`、`communication`、`memory`，均非负 | 加权目标 |
| `placement.max_evaluations` | 正整数 | 放置搜索评估上限 |
| `placement.max_time_s` | 非负数，0 表示不限 | 放置搜索墙钟预算 |
| `placement.max_proposals` | 正整数或空 | 放置提案上限 |
| `placement.seed` | 非负整数 | 随机种子 |
| `placement.moves.max_segment_ops` | 正整数 | 连续段移动的最大算子数 |
| `placement.moves.max_layer_block_layers` | 正整数 | 层块移动的最大层数 |
| `placement.moves.weights` | 八类移动各一个非负权重，至少一个为正 | 各类移动的抽样权重 |
| `placement.algorithm_params` | 与第六章算法参数相同 | 放置算法的参数 |

　　八类放置移动分别是：`op_relocate`、`op_swap`、`connected_segment_relocate`、`attention_branch_relocate`、`kv_group_pack_or_spread`、`layer_block_relocate`、`layer_boundary_shift`、`pim_rebalance_or_swap`。权重只写其中几类也可以，但总和必须为正。

　　如果使用 `predictor` 作为评估后端，画像与扫描里的 `host_relay_overhead_ns` 必须保持 100 纳秒，并且必须存在预测器检查点；否则加载配置时就会报错。

## 八、硬件参数扫描

　　硬件参数扫描不是外层候选空间的一部分，而是沿指定硬件字段逐点求值，用来观察容量、时钟或带宽变化对结果的影响。它由扫描入口使用，配置写在 `sweep` 段。

　　扫描字段与第五章的硬件字段是同一份清单；每个字段给出一个取值列表。扫描配置如下表所示。

| 字段 | 可以取什么值或有什么约束 | 说明 |
| --- | --- | --- |
| `sweep.fields` | 键必须是硬件字段，值是非空列表 | 逐点扫描的字段与取值 |
| `sweep.core_policy` | `fixed_dpus`、`use_all_clusters` | 固定 DPU 数，或使用画像全部物理簇 |
| `sweep.knee_improvement_threshold` | 小于 1 的数 | 容量拐点判据 |
| `sweep.pareto.resource` | `total_physical_memory_bytes`、`num_physical_cores` | Pareto 资源轴 |
| `sweep.pareto.direction` | `minimize`、`maximize` | Pareto 方向 |

　　硬件扫描需要以一个已有最优候选为基准。例如先跑完默认快速调优，再执行如下命令。

```bash
python scripts/tune_sweep.py --config examples/tune/hardware_sweep.yaml \
    --best test-results/tune-quick-4d/best.json
```

## 九、代价模型校准

　　代价模型校准用一份实测报告校准仿真结果的成熟度。配置段只有一个字段 `calibration.validation_report`。该字段为空时表示不校准；给出路径时，文件必须存在，否则加载配置时报错。

　　校准不改变候选空间和编译过程，只影响仿真结果如何被标注为模型预测或经过实测校准。需要校准时，把报告路径写入配置即可。

## 十、每轮固定、不参与搜索的配置

　　除了搜索空间，配置里还有若干固定项。它们每轮只取一个值，不参与候选枚举，但会决定输入、硬件基线和输出位置。这些固定项如下表所示。

| 配置段 | 字段 | 含义与取值 |
| --- | --- | --- |
| `model` | `model_dir` | 模型目录；为空时使用 `paths.json` 里的默认模型 |
| `model` | `ir` | 已导出的图骨架；为空时按模型现场生成 |
| `model` | `prefill_seq_len` | 预填充长度，必须覆盖请求的最长提示 |
| `model` | `max_seq` | 上下文上界；为空时按请求推导 |
| `model.trace` | `path` | 请求轨迹文件；与合成轨迹二选一 |
| `model.trace` | `num_requests` | 从轨迹里取多少个请求 |
| `model.trace` | `synthetic` | 合成轨迹，例如 `{seed: 0, num_requests: 10}` |
| `profile` | `path` | 硬件画像文件 |
| `optimization` | 四个开关 | 所有候选的基线开关，默认全部开启 |
| `cache` | `mode` | `fast` 复用完整产物；`strict` 强制重编 |
| `simulation` | `use_cost_sidecar` | 是否让仿真消费成本 sidecar |
| `output` | `dir` | 输出目录，相对配置文件所在目录解析 |

## 十一、配置示例

　　本章给出几种常见改法。每个例子只展示需要修改的片段，其余字段保持默认配置即可。修改后先跑 dry-run，确认候选数和拒绝原因，再执行正式调优。

### 11.1 只搜索一个维度

　　只希望搜索 tasklet 数时，把其他层各写一个值。此时组合数为 2。

```yaml
space:
  strategy:
    enabled: true
    num_dpus: [8]
    num_stages: [2]
  op:
    enabled: true
    num_tasklets: [4, 8]
  optimization:
    enabled: false
    field: cse
    values: [true]
```

### 11.2 搜索三个维度

　　希望同时搜索 DPU 数、tasklet 数和优化开关时，把流水段数固定为一个值。此时组合数为 8。

```yaml
space:
  strategy:
    enabled: true
    num_dpus: [4, 8]
    num_stages: [2]
  op:
    enabled: true
    num_tasklets: [4, 8]
  optimization:
    enabled: true
    field: cse
    values: [true, false]
```

### 11.3 加入硬件维度

　　希望比较不同时钟或不同 WRAM 容量时，把字段写入 `space.hardware.fields`。组合数会在原有基础上再乘以硬件取值数。

```yaml
space:
  hardware:
    enabled: true
    fields:
      clock_period_ns: [0.63, 0.5]
      wram_bytes_per_dpu: [65536, 131072]
```

### 11.4 换用随机搜索算法

　　候选空间较大时，可以改用模拟退火并给出预算与各层权重。注意要给优化开关层权重，算法才会提案它。

```yaml
search:
  algorithm: simulated_annealing
  seed: 0
  max_evaluations: 32
  max_time_s: 0
  layer_weights: {strategy: 0.3, op: 0.3, hardware: 0.1, optimization: 0.3}
  algorithm_params: {initial_temperature_ratio: 0.05, cooling_rate: 0.9}
```

### 11.5 开启内层放置搜索

　　CEVA 画像保持 `placement.enabled: false`。通用 PIM 画像可以按如下方式开启，并给出预算、目标权重和移动权重。

```yaml
profile:
  path: ../../contracts/profiles/generic.json
placement:
  enabled: true
  algorithm: simulated_annealing
  evaluation_backend: pipeline
  validation_backend: pipeline
  objective_metric: total_time_s
  objective_weights: {compute: 0.5, communication: 0.25, memory: 0.25}
  max_evaluations: 4
  moves:
    max_segment_ops: 8
    max_layer_block_layers: 4
    weights: {op_relocate: 1.0, op_swap: 1.0}
  algorithm_params: {initial_temperature_ratio: 0.05, cooling_rate: 0.9}
```

## 十二、每个候选实际执行了什么

　　一个候选确定后，调优器依次执行图编译、算子编译、成本精化和 GeneSim 仿真。四层取值决定编译参数，编译产物再决定仿真成本。执行顺序如下图所示。

```text
候选
  ├─ 图编译：按 DPU 数与流水段数切分计算图，生成放置 sidecar
  ├─ 算子编译：A/B 两路按候选开关过滤 pass，PIMIR 带 DPU 与 tasklet 属性
  ├─ 成本精化：按同一份硬件契约生成成本 sidecar
  └─ GeneSim：加载 PIM trace，按 tasklet 并行度计算周期，输出 total_time_s
```

　　四层取值都会进入编译键，因此不同候选不会复用彼此的编译产物。仿真前，调优器会核对侧车与仿真契约中的硬件五元组、优化开关、图策略、编译键和产物摘要；任何一项不一致都会直接失败。内层放置搜索开启时，GeneSim 会在该候选内部继续搜索放置方案，并用真实 pipeline 验证最终映射。

## 十三、结果与复现

　　一轮调优结束后，输出目录里会留下若干文件。轮次日志记录每个候选的状态与分数，最优文件记录最终选择和复现命令，摘要文件给出人类可读的结论。这些文件如下表所示。

| 文件 | 内容 |
| --- | --- |
| `effective_config.yaml` | 合并命令行覆盖后的生效配置 |
| `rounds.jsonl` | 每个候选一行：状态、分数、四层状态、编译键、仿真键、耗时和失败原因 |
| `responses.jsonl` | 续跑使用的调用日志 |
| `best.json` | 最优候选与复现命令 |
| `summary.md` | 人类可读摘要与真实搜索覆盖 |

　　需要严格复现最优候选时，使用最优文件里的路径并加上严格缓存档位。该命令会强制重新编译并重新仿真，再与记录比较分数。

```bash
python scripts/tune_pipeline.py --evaluate-candidate test-results/tune-quick-4d/best.json --cache-mode strict
```

　　需要按历史顺序复跑整轮时，使用轮次重放入口。它会重算每个候选的编译键和仿真键，核对产物后再重放。

```bash
python scripts/tune_pipeline.py --replay-rounds --output-dir test-results/tune-quick-4d
```

## 十四、实现与验证边界

　　手册中的搜索空间都已经接入配置解析和实际执行链路。外层四层会进入候选枚举、编译键和仿真契约；随机算法的参数会在构造算法时校验；内层放置搜索和硬件扫描分别由各自入口消费。

　　本轮的验证重点是默认配置的 16 个候选。它们已经用一键脚本在干净目录完整跑完：16 个全部成功，0 次缓存命中，最优候选为 `tp4_pp2_dpu8_tl8_opt1011`，仿真时延 7.595039 秒。其余空间通过配置加载、dry-run 和单元测试核对可配置性，本轮没有重新跑长时间的内层放置搜索和硬件扫描。具体边界如下表所示。

| 能力 | 当前状态 |
| --- | --- |
| 外层四层：DPU 数、流水段数、tasklet 数、优化开关 | 已实现；默认配置的 16 个候选做真实端到端验证 |
| 20 个硬件字段 | 已实现；每个字段都能派生合法画像，改值会产生不同画像摘要 |
| 四种外层算法与参数 | 已实现；三种随机算法可构造，枚举算法拒绝参数 |
| 内层放置搜索 | 已实现；通用画像可加载，CEVA 画像强制关闭 |
| 硬件扫描字段 | 已实现；与候选空间的硬件字段共用同一份清单 |
| 代价模型校准 | 已实现；路径和文件存在性在加载配置时校验 |

## 十五、常见报错与处理

　　配置错误和运行错误大多会在编译前或仿真前报出，并指出具体字段或候选。常见现象、原因和处理方式如下表所示。

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `num_stages 不能整除 num_dpus` | 流水段数与 DPU 数不匹配 | 改成能整除的组合 |
| `tp_width 不是 2 的整数次幂` 或 `不能被 tp_width 整除` | 模型维度不支持该 TP 宽度 | 减少 DPU 数或增加流水段数 |
| `space.xxx.enabled=false 表示冻结，取值必须恰好一个` | 关闭的层写了多个值 | 关闭层只保留一个值 |
| `space.optimization.field 不在支持列表` | 开关名写错 | 使用 `constant_fold`、`cse`、`dce`、`loop_tiling` |
| `从 pim mlir 里没读出输出分块` | 关闭了 `loop_tiling` | 换用 `cse` 等开关，保持分块开启 |
| `space.hardware.fields.xxx 派生不出合法画像` | 硬件取值自相矛盾 | 检查容量、DPU 数和对齐等约束 |
| `placement.algorithm=enumerate` 报错 | 内层放置不支持枚举 | 改用三种随机算法之一 |
| CEVA 画像下开启放置搜索报错 | CEVA 是一对一映射 | 把 `placement.enabled` 设为 `false` |
| `sweep.fields 有未知字段` | 扫描字段名写错 | 使用第五章的硬件字段清单 |
| 输出目录已有调优日志 | 目录里已有 `responses.jsonl` | 换一个输出目录，或加 `--resume` 续跑 |
| 候选一致性校验失败 | 运行期间改了源码、工具链或输入 | 用干净目录重跑，运行期间不要修改 `tune/`、`genesim/` 或 `paths.json` |

## 十六、推荐使用流程

　　初次使用时，建议先复制默认配置，改一个全新的输出目录，避免与历史日志冲突。然后按需要保留搜索维度，其余层各写一个值冻结。配置写好后先跑 dry-run，确认候选数和非法组合的拒绝原因。

　　确认组合无误后，执行一键脚本或统一入口。运行结束后先看摘要和最优文件，再按需严格复现最优候选。需要观察硬件参数拐点时，再使用硬件扫描入口。按照这个顺序，用户可以在不了解内部实现的情况下完成一轮可复现的编译调优。
