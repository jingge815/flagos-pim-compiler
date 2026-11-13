# PIM 编译调优使用说明

　　本轮闭环把「候选配置 → 图编译 → 算子编译 → 成本精化 → GeneSim 仿真 → 结果记录」做成一个入口。配置是唯一状态：命令行覆盖先合并进生效配置并落盘，所有后续阶段读同一份。

## 一、环境与输入

　　每个新 shell 先激活环境：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
```

　　模型默认取 `paths.json` 的 `llama2_7b_model_dir`；请求轨迹用 GeneSim 的 JSON 轨迹格式。示例 `examples/tune/trace_quick.json` 是一个真实的 16 token 请求。图骨架 `ir: null` 时由 GeneSim 的 `scripts/model_parser.py` 现场生成，编号与仿真加载的那份同源。

## 二、三份示例

### 2.1 快速闭环

```bash
python scripts/tune_pipeline.py --config examples/tune/quick.yaml
```

　　CEVA 64 核画像、一个真实请求、四维 16 个候选（DPU 数 × PP 段数 × tasklet 数 × 编译优化开关）。真实执行图编译、算子编译、成本精化与 pipeline 仿真，不使用 `--skip-simulation`，也不使用假目标。产物在 `test-results/tune-quick-4d/`。

### 2.2 策略搜索

```bash
python scripts/tune_pipeline.py --config examples/tune/strategy.yaml
python scripts/tune_pipeline.py --config examples/tune/strategy.yaml --algorithm tabu_search
```

　　通用 PIM 画像，搜索 TP/PP 组合与 tasklet，内层放置搜索开启并配置八类移动权重。三种算法（`simulated_annealing` / `genetic_algorithm` / `tabu_search`）共用外层候选协议；`enumerate` 用于小空间确定枚举。

　　每种算法做真实搜索验收时必须各用一个干净输出目录：同一目录里的编译与仿真缓存按内容键共享，后跑的算法会直接命中前一个算法的 `sim_key`，只能证明缓存一致性，不能当作本算法的新搜索证据。每次运行都会在 stdout、`summary.md` 与 `best.json` 的 `search_coverage` 中记录逻辑/实际调用、新 pipeline 评估数、编译/仿真缓存命中数、新后端执行数与最终映射重放数；只有至少一次本算法发起的新 pipeline 评估且完成最终映射重放时 `real_search` 才为真。用同一目录重复运行验证缓存复用时，`real_search` 应为假、`sim_cache_hits` 应等于成功候选数。

### 2.3 硬件扫描

```bash
python scripts/tune_sweep.py --config examples/tune/hardware_sweep.yaml \
    --best test-results/tune-quick-4d/best.json
```

　　扫描每设备核数（`clusters_per_device`）、每核容量、时钟与跨簇路由（`inter_cluster_communication`）。改核数或容量的点会按新画像重编，只改时钟或路由的点复用编译产物。`--output-dir` 在输入快照、编译与仿真前合并进生效配置，输出目录里同时写 `effective_config.yaml`、`sweep.jsonl`、`sensitivity.json`、`pareto.csv`、`pareto.svg` 与 `report.md`。CEVA 画像固定单设备与每簇一个 TensorPU；通用 PIM 可把 `num_pim_devices` 放进字段。

　　扫描开始前先核对 `best.json` 的绑定：必须是成功且有有限分数的记录，模型/IR/轨迹内容、`max_seq`、请求数与基线画像必须和本次扫描一致。`best.json` 保存生成时生效配置的内容摘要 `effective_config_sha256`，扫描前置会用 best 自己的生效配置与本次输入重算历史 `compile_key`/`sim_key`，再与记录逐项比较：同路径原地改写、配置被移动或替换、放置/仿真/校准语义不同都会在发布输入快照、写生效配置与求值之前拒绝。仅移动 best 与其生效配置文件、内容不变时允许；候选空间、搜索预算、扫描字段、缓存档位与输出目录的变化只作用于新扫描求值，不改写历史 best 的语义。任何不符都退出码 3。`sensitivity.json` 的 `binding` 字段与报告头部保留 best 候选、分数、生成时摘要、当前文件摘要、编译/仿真键与输入身份，供独立复算。

### 2.4 四维搜索（16 轮）

```bash
python scripts/tune_pipeline.py --config examples/tune/quick.yaml
```

　　一个配置同时扫描四个维度：

| 维度 | 配置位置 | 取值 |
| --- | --- | --- |
| 编译优化开关 | `space.optimization.field` / `values` | `cse` 开 / 关 |
| 每 DPU tasklet 数 | `space.op.num_tasklets` | 4 / 8 |
| DPU 数 | `space.strategy.num_dpus` | 4 / 8 |
| 流水段数（PP） | `space.strategy.num_stages` | 2 / 4 |

　　2×2×2×2 = 16 个候选，`search.algorithm: enumerate`、`search.max_evaluations: 16` 一次跑完。TP 宽度不单独配置，由 `num_dpus / num_stages` 派生：4 台 DPU、2 段是 TP2；8 台 DPU、2 段是 TP4；8 台 DPU、4 段是 TP2。先加 `--dry-run` 可以只看 16 个组合与派生出的 `tp*_pp*`，不编译不仿真。

　　每个候选的 `compile_key` 都包含这四层取值，任一维变化都会重编。轮次日志与候选编号用 `opt=1111` 这样的四位布尔串记录开关（顺序 `constant_fold`、`cse`、`dce`、`loop_tiling`），例如 `opt=1011` 表示关闭 `cse`。

## 三、命令行

| 参数 | 含义 |
| --- | --- |
| `--config` | 版本化调优配置，必填（单候选/重放入口除外） |
| `--max-evaluations` | 外层候选评估数上限 |
| `--max-time-s` | 外层墙钟预算；只限制发起新评估，正在执行的允许完成 |
| `--algorithm` | 外层算法：enumerate / simulated_annealing / genetic_algorithm / tabu_search |
| `--profile` | 覆盖硬件画像路径 |
| `--cache-mode` | `fast` 复用完整产物；`strict` 强制重编并新开仿真尝试目录 |
| `--dry-run` | 只列合法候选、拒绝原因与预算，不加载权重、不编译、不仿真；输出目录已有响应或生效配置时不改写 |
| `--resume` | 沿用输出目录已有的响应日志继续，已完成调用不重复执行 |
| `--output-dir` | 覆盖输出目录 |
| `--evaluate-candidate` | 按 `best.json` 严格重编并重跑该候选，比较 `total_time_s` |
| `--replay-rounds` | 按轮次日志首次真实评估顺序复跑并比较；用 `--config` 或 `--output-dir`（读取其中 `effective_config.yaml`）确定原生效配置 |

### 3.1 编译优化开关

　　YAML 顶层 `optimization` 段控制四类编译优化，缺省全开；旧配置没有该段时行为不变。它是本轮所有候选的基线：

```yaml
optimization:
  constant_fold: true
  cse: true
  dce: true
  loop_tiling: true
```

　　`cse` 同时控制上游 `-cse` 与 `-pim-cse`，`loop_tiling` 控制 `-pim-tile-to-budget`。上游 `-canonicalize` 与 `-loop-invariant-code-motion` 始终执行，开关只控制 PIM 专用优化 pass。融合 `-pim-fuse-activation`、相位展开 `-pim-expand-phases` 与契约校验 `-pim-verify-gml-contract` 是目标格式的硬要求，不能关闭。关闭 loop tiling 后大 kernel 可能因 WRAM 超预算在后续 pass 失败，这是显式选择；当前图编译的本地形状推导依赖 `pim.tile-*`，关掉 tiling 会直接读不到输出分块，所以示例用 `cse` 作为搜索维度。开关进入 `compile_key`、算子编译缓存键与放置 sidecar 的 `compiler.hardware.optimizations`，换开关必然重编；A/B 两路、成本精化与仿真前校验消费同一份开关。需要对比优化收益时，复制 YAML 改开关并换 `output.dir`，或直接用 `--output-dir` 隔离产物。

　　要把某一个开关作为搜索维度，在 `space.optimization` 里声明字段与取值：

```yaml
space:
  optimization:
    enabled: true
    field: cse
    values: [true, false]
```

　　只扫描声明的这一个字段，其余开关取顶层值；不写该段表示冻结在顶层配置，旧配置行为不变。候选的开关进入 `compile_key`、算子缓存键与 sidecar，也写进 `rounds.jsonl`/`best.json` 的候选状态，`--evaluate-candidate`、扫描与续跑都能原样复现。`search.layer_weights` 可以额外给 `optimization` 一个权重，让模拟退火等算法也提案这一层；不写则只有完整枚举会展开它。

## 四、预算与续跑

　　外层预算由候选评估数与墙钟共同约束，计数包含基线；内层放置搜索用自己的预算单独记账，不会被外层一轮掩盖。墙钟预算统计历史活动耗时，回放与离线等待不计。预算耗尽后不再发起新评估，已完成记录正常汇总。

　　续跑以原种子、原总评估与原提案预算从头运行算法，按调用序号回放已提交响应；调用顺序与日志不一致直接拒绝。续跑必须保持同一生效配置、源码、工具、基础 GeneSim 配置、环境变量与输入：只放宽外层候选数/墙钟/提案预算是合法的预算扩展，收缩预算、改缓存档位、仿真开关、画像、空间、环境或输入会直接拒绝，不会复用旧响应。运行身份按内容覆盖画像与校准报告，同一路径原地修改也会在写输入与生效配置之前拒绝；每次回放前还会按当前源码、工具、画像、基础配置、环境、轨迹和放置输入重算编译/仿真键，与日志不符即拒绝。纯续跑或纯回放是独立调用：`summary.md` 与 `best.json` 的 `search_coverage` 描述本次调用，实际执行 0 次时 `real_search=false`；原始搜索证据以 `rounds.jsonl` 的逐轮缓存命中状态与首次运行日志为准。

　　输入处理分三步：只读解析与身份计算、续跑/严格重放身份核对、先在独立暂存目录写全并复核、再提交整个输入集合。轨迹、外置 IR、现场生成的 IR 与 `inputs.json` 共用一个提交边界：暂存字节逐项与 prepare 阶段身份一致后才提交；首次发布用目录整组换名（原子），已有发布目录时内容一致即不写任何文件、内容不同则整组拒绝。任何失败只清理暂存区，已发布快照的 IR、轨迹与身份清单保持原字节。外置 `model.ir` 会复制进输入快照，原文件后续变化不影响已发布输入；严格复跑先逐项核对 `best.json` 的模型/IR/轨迹/max_seq/请求数以及编译键、仿真键，任何一项不符都在编译前退出码 3。轨迹请求数少于配置的 `num_requests` 在 dry-run 与普通入口都直接拒绝。

　　输入快照发布后并不会被“记住摘要”替代：候选编译开始前与编译产物发布前、仿真前校验、仿真配置生成、完成清单发布、快速缓存读取、严格复跑与轮次重放都会按磁盘当前字节重新核对模型完整指纹、原始 IR 与请求轨迹窗口（含逐请求 KV 峰值与 `max_seq`）。同一输入对象指向的文件在 prepare 后被原地改写（包括编译期间）即拒绝，不能用新文件内容重绑旧编译/仿真键；同内容搬迁不算变化，合法 padding/decode 动态形状按 GeneSim 同一口径重算请求窗口与处理 token 数。完成清单发布还会交叉核对 manifest 的模型/IR/轨迹身份、画像契约登记的轨迹摘要与窗口、仿真配置实际消费的 `trace.path` 内容与摘要的实际 `processed_tokens`/`completed_requests`，目录内部自洽但属于另一条轨迹的结果不能发布或被缓存复用。

　　响应日志是唯一提交点：轮次记录带相同 `call_index`，崩溃留下的“有轮次、无响应”记录在恢复时移入 `orphan_rounds.jsonl` 并明确重跑，排序与轮次数只认已对账记录，同一候选不会在轮次日志里重复出现。恢复前还会按调用序号核对响应与轮次的状态、分数（含 `inf` 特殊编码）、编译/仿真键、运行身份与失败原因；响应序号必须连续且每条有对应轮次，任何冲突或不连续都在发起评估前拒绝，不能一边回放失败响应、一边从轮次日志选出成功候选。

　　仿真键声明本次实际使用的计时后端：原生库按 GeneSim 运行期同一套解析与加载规则探测，文件存在但加载或入口符号不满足时声明为内置轨迹回退，并记录库内容身份。仿真摘要同时记录运行期逐算子台账（原生与回退次数、回退阶段、是否混合）与原生准备台账：每个算子的轨迹内容身份、装载返回值、循环绑定结果和实际使用的循环次数都会登记；DRAMsim3 装载返回非零、AttAcc 缺少 `load_trace`/`run_kernel` 入口、循环次数绑定异常都会在任何原生计时前关闭该次原生路径并转入已声明的 PIMTrace 回退，不允许先原生计时再静默混用。发布完成清单、快速缓存命中、严格复跑都要求摘要里的实际后端与键声明逐项一致，否则候选失败或缓存不命中。续跑身份包含同一后端探测结果，环境变化会在回放前失效。这样原生初始化、轨迹装载或执行中途回退不会继续复用旧缓存身份。

　　编译缓存按 `compile_key` 发布，每次编译写入独立 `attempt-<id>` 目录：快速档命中任一完整 attempt，`strict` 总是新建 attempt 且保留旧目录与旧 PIM MLIR。编译完成判据要求设计产物角色齐全：`partition_plan.json`、`placed.ir`、`placement.json`、`refined.ir`、`cost.json`，角色必须与规范相对路径一一对应，禁止重复角色、重复路径与越界符号链接；placement sidecar 引用的全部 PIM MLIR 都必须登记在清单里、逐文件哈希一致。写入、快速读取与两种重放共用这一判据。编译键除模型/IR/形状/策略/算子/画像/工具链外，还包含本轮轨迹的逐请求 KV 峰值与 `scheduler.max_inflight_requests` 的有效并发上限：相同 `max_seq` 下换请求分布或改并发准入都会重新编译，不能复用旧请求的容量复核结果。仿真按 `sim_key` 与独立尝试目录隔离，`strict` 不改写旧结果。不完整的编译 attempt（缺角色、角色别名到其他文件、空清单、PIM MLIR 与清单不一致等）会改名为 `invalid-attempt-*` 隔离后重建，不参与快速命中。

## 五、输出目录

```text
<output>/ effective_config.yaml, inputs/, responses.jsonl, rounds.jsonl,
          orphan_rounds.jsonl, best.json, summary.md
  cache/compile/<compile_key>/attempt-<id>/ partition_plan.json, placed.ir,
    placement.json, refined.ir, cost.json, pimir/<sha256>.mlir, artifacts.json
  sim/<sim_key>/<attempt>/ config.yaml, hardware_contract.json, manifest.json,
    effective_placement.json, completion.json, pim_traces/,
    results/{summary,autotune_metrics,efficiency_metrics,run_manifest}.json
```

　　`rounds.jsonl` 每行记录候选三层取值、调用序号、产物路径与哈希、编译/仿真缓存命中状态、仿真时延、效率与容量指标、原生后端/轨迹回退身份、结果成熟度、实际耗时、失败阶段与失败原因。`orphan_rounds.jsonl` 保存崩溃窗口里未提交的轮次及隔离原因，不参与排序。`summary.md` 与 `best.json` 的 `search_coverage` 记录本次运行的真实搜索覆盖：逻辑/实际调用、新 pipeline 评估数、编译与仿真缓存命中数、新后端执行数、最终映射重放数与 `real_search` 判定；回放和缓存命中不计入新评估。`best.json` 含完整候选、输入身份、编译/仿真键、生成时 `effective_config_sha256` 与复现命令。仿真 attempt 在全部后置校验通过后写 `completion.json`；完成清单必须登记 `results/summary.json`、`results/autotune_metrics.json`、`config.yaml`、`manifest.json`、`hardware_contract.json`、`effective_placement.json` 与全部 `pim_traces/op_*.pim_trace`，trace 的算子集合必须与最终挂 PU（`attached_pu_id >= 0`）的目标集合精确相等：缺 trace、图外 trace 与搜索后迁离 PIM 的残留 trace 都会被拒绝。键必须与本次 `manifest.json` 同源，并记录 `execution_mode`（`search`/`fixed`/`replay`）。`autotune_metrics.json` 按模式校验：原生搜索必须有算法、搜索后端、逻辑评估数、一次最终后端执行与总数自洽；固定映射与严格重放必须显式记录零搜索和一次最终执行。缺任一角色、内层指标缺失或计数矛盾、trace 与最终 PIM 目标不一致或文件哈希不符的目录不会被快速档复用。完成清单还必须与当前已复核输入交叉一致：manifest 的模型/原始 IR/轨迹输入身份、配置摘要与画像契约摘要逐项与磁盘对账，画像契约登记的轨迹摘要与窗口、仿真配置实际消费的 `trace.path` 内容以及摘要里的 `processed_tokens`/`completed_requests` 必须与请求窗口一致；目录内部自洽但绑定另一条轨迹的结果既不能发布，也不能被快速档命中。`--replay-rounds` 逐轮重算编译/仿真键并核对编译完成角色/PIM MLIR 覆盖、产物清单与最终放置，成功轮次的编译目录缺失或不完整直接拒绝；第二十一轮起还会按当前输入重算完整候选身份，核对清单中的模型完整指纹/原始 IR/轨迹输入角色、五个产物路径与服务端内容、精化 IR 的源模型/原始 IR 身份，任何一项不符都在发布输入与评估前退出码 3。第二十二轮起，模型/IR/轨迹身份不只在重放前置比较一次：编译发布前后、仿真配置生成、完成清单发布、快速缓存读取与两种重放都会按磁盘当前字节重新核对，完成清单还会交叉核对 manifest 输入身份、画像契约轨迹摘要/窗口、配置实际消费轨迹与摘要 `processed_tokens`/`completed_requests`。

## 六、一致性校验与结果分级

　　仿真启动前核对候选画像摘要、容量、图策略（TP/PP、DPU 集合）、切分方案与 DPU 映射、PIM MLIR 哈希、精化 IR 与放置 sidecar 的实际内容身份、模型与原始 IR 的完整指纹、请求轨迹内容与窗口、仿真 IR 的算子编号与类型；同硬件不同图策略、同策略异模型/异原始 IR/异请求轨迹以及混合候选产物一律拒绝，任一不符即失败，不沿用旧结果目录。这些输入身份不只在启动前比较一次内存摘要：编译发布前后、仿真配置生成、完成清单发布、快速缓存读取与两种重放都会按磁盘当前字节重新核对，文件在任一阶段被改写都会让本轮候选失败，而不是把新内容绑到旧键上。契约由画像覆盖项递归生成，环境变量覆盖后的周期、倍率、链路、主机模型与来源台账也会逐项核对。严格产物模式下按最终执行图逐算子核对 PIM MLIR 与轨迹，被融合吸收的节点必须有带证据的宿主。CEVA 保持编译器一对一映射；通用 PIM 才启用原生放置搜索，搜索结果与严格重放共用同一物化入口并重建集合通信与 relay。内层搜索的 `autotune_metrics.json` 必须与本次生效配置一致：算法、搜索后端与验证后端逐项比对；`predictor` 只有在 `placement.evaluation_backend` 明确配置时才允许做筛选，最终验证仍必须是 `pipeline` 的一次执行。写入完成清单与快速缓存命中都会重做同样的比对，改算法/后端或伪造指标不能命中旧结果。

　　契约同时覆盖区间倍率：画像里的 `zone_cycle_factors` 在加载时复用 GeneSim 的 12 位编码约束（有限、≥1、0.5 的整数倍），非法值在编译前报错。内层放置预算的 `max_proposals` 会真正传入 GeneSim 的 `Budget`，不是只写入配置。路由只允许 `direct`/`host_relay`，链路三个语义标记必须是布尔值，CEVA 主机参数必须严格为正；这些与 GeneSim 原生契约同宽严，在画像加载与候选展开时前置拒绝。

　　编译目标是请求级编译契约的一部分：正式调优从画像的 `compiler_target`（固定 `pim:v1`）派生，A 路 `convert-triton-to-pim` 的 `target=`、B 路模块的 `pim.target`、算子缓存键与放置 sidecar 的 `compiler.hardware.compiler_target` 全部由它决定；`FLAGTREE_PIM_TARGET` 只保留给不使用画像的旧单次进程内捕获入口，不能再改变正式链路的产物。仿真前再核对 sidecar 的编译目标与候选画像一致，`refine_ir_with_flagtree.py --profile` 也拒绝缺少该声明或目标不符的旧 sidecar（需用新导出器重新生成）。

　　放置 sidecar 的工具链身份是可核对证据，不是自报声明：调优侧在每次候选仿真前把 `compiler.toolchain` 的版本、图编译源码摘要、`compiler_fingerprint` 与每个实际使用工具的内容摘要同当前 `compiler_identity()` 逐项比较，路径不参与身份；声明缺失、工具摘要为空或全零、源码摘要/编译指纹/工具内容不符都在仿真前拒绝。GeneSim 对 `triton-opt`、`libtriton`、`mlir_translate`、`gcc` 四类实际工具按角色逐个核对内容：每类先按“进程环境（`OPCOMPILER_TRITON_OPT`/`FLAGTREE_PREFIX`/`CC`）→ 图仓库 `paths.json` 的实际安装/当前进程加载的库 → sidecar 声明的前缀与工具路径”选定本机权威路径，首个可解析工具就是本次消费的工具；内容不符或读不到都失败关闭，匹配后不再让低优先级旧路径否决。**本机存在可解析工具时，声明必须给出有效内容摘要；把路径与摘要同时置空不能豁免，会按“未声明内容摘要”拒绝。**只有本机完全无法解析某个角色时，才按外置产物的声明格式校验（例如产物搬到没有对应工具的机器）。`mlir_translate` 在产物迁移后按声明前缀下的相对布局在本机前缀里找回。严格重放与快速缓存的输入摘要只消费通过该校验的身份。

　　GeneSim 的通用与 CEVA 启动校验、独立/快照运行清单与最终放置重放统一调用同一产物契约入口 `verify_compiled_artifact()`：一次完成画像绑定、工具链身份、逐引用哈希、A/B 路由与必需预算属性、映射范围核验，调用方不能再只做结构检查；通用 PIM 的共享 ClusterPU 仍按非一对一语义放行，通用启动不再绕过 PIM MLIR 的实际预算核对；通用 PIM 配置了 `scheduler.compiler_placement_file` 时同样在创建调度器前核验，文件缺失或不可解析直接退出码 1，不再告警后退回启发式放置；未配置 sidecar 的旧命令行为不变。**调优候选写出的 `hardware_contract.json` 从 schema 3 起绑定完整候选编译身份**：除 `compile.compile_profile_sha256`、`compile.hardware` 的 DPU/tasklet/MRAM/WRAM/DMA/编译目标与 `physical_mram_bytes`、`reserved_mram_bytes`、`usable_mram_bytes` 外，还携带 `compile_key`、`strategy`（名称、DPU 数、段数、TP 宽度与 DPU 集合）、`dpu_to_cluster` 计划映射，以及实际消费的 refined IR/放置 sidecar（使用成本 sidecar 时另含成本 sidecar）内容摘要，以及本次仿真实际消费的请求轨迹内容摘要 `trace_sha256` 与请求窗口 `trace_num_requests`。调优侧 `verify_compiled()` 从同一 `TuneState` 与 partition plan 派生期望策略/映射并逐项核对，精化 IR 还记录产出它的放置 sidecar 语义身份与源模型/原始 IR 身份（忽略可搬运的物理路径）；`verify_compiled()`、`build_sim()`、快速缓存读取、严格复跑与轮次重放共用 `verify_candidate_identity()`，按当前 cfg/输入/候选重算 compile key 并核对产物对象与清单的编译键、清单输入角色与五个产物路径/内容，单独交换任一产物或输入都会被拒绝。模型完整指纹、原始 IR 与请求轨迹窗口由 `verify_input_snapshot()` 按磁盘当前字节复核；`build_sim()` 写出的画像契约、`manifest.json` 与 `sim_key` 都只消费这一份已复核快照，不再各自重读磁盘。完成清单发布时再交叉核对 manifest 输入身份、画像契约轨迹摘要/窗口、配置实际消费的轨迹与摘要 `processed_tokens`/`completed_requests`。GeneSim 把这份绑定、生效容量与实际产物/轨迹身份一起传入统一入口，在创建调度器或写运行清单前逐项核对 sidecar 的 `compiler`、`compiler.strategy`、`compiler.hardware`、顶层 `dpu_to_cluster`/`hardware` 副本、生效容量、本次实际消费的文件与请求轨迹窗口；只比较画像摘要不够——同硬件不同 TP/PP、同策略异模型/异原始 IR/异请求轨迹的候选必须区分，字段被改而摘要未同步时同样失败。因此异编译画像、异 MRAM/WRAM/DMA/目标/容量块、异图策略、异模型/异原始 IR/异请求轨迹与混合候选产物都不能在通用启动、CEVA 启动、独立/快照清单或最终重放中入库。缺少轨迹摘要/窗口的旧 schema 3 契约因结构不合法被拒绝，需用当前调优入口重新生成；schema 1/2 旧契约按可推导的画像摘要与字节容量做部分绑定；schema 2 契约（只有硬件五元组与容量）按显式旧格式保留部分绑定，不会被说成“完整候选身份已核验”；显式旧 sidecar（`version<5`）与无画像契约的旧命令保留原有兼容边界。严格与旧产物的分流只看放置 sidecar 顶层的显式 `version`：当前 schema（`version=5`）的产物必须同时带 `compiler.toolchain`、`compiler.hardware`（DPU/tasklet/MRAM/WRAM/DMA/编译目标）、`compiler.strategy`（`num_dpus` 与硬件一致，`dpu_ids` 必须覆盖全部 DPU）与可解析的 PIM MLIR 引用，CEVA 还要求顶层 `dpu_to_cluster` 显式、一对一，正式严格运行还要求 `requires_pimir=true`。**删除整个 `compiler` 块、只删 `compiler.toolchain`、删 `dpu_ids` 或清空算子产物引用都会在创建调度器前失败**——分类由版本号决定，不能再靠删字段退回旧兼容。每个 PIM MLIR 引用必须声明 `pimir_route`：A 路（GEMM）必须写全 `pim.num-dpus`、`pim.num-tasklets`、`pim.mram-bytes`、`pim.wram-bytes`、`pim.dma-align`、`pim.target` 六项并与 `compiler.hardware` 相同；B 路（其余算子级节点）只豁免 DPU/tasklet 编码，MRAM/WRAM/DMA/目标仍须逐项相同，写了 DPU/tasklet 也要一致，路由与算子类型不符直接拒绝。**同一 PIM MLIR 文件被多个算子引用时，缓存只复用文件字节、摘要与模块属性；路由、算子类型与预算属性逐条引用核对，JSON 条目顺序不影响结论**，合法共享的 A/A、B/B 与混合 A/B 引用不受影响。映射编号还必须落在本次生效配置的 `system.num_pim_devices` × `pim.clusters_per_device` 范围内：CEVA 启动校验、运行清单与最终放置重放共用同一范围判据，设备或簇越界在创建调度器前拒绝，CEVA 允许只使用部分物理核，通用 PIM 的共享 ClusterPU 映射不套用一对一限制。显式旧 schema（`version<5`）的产物没有编译契约，沿用原有的路径反推身份与“写了才核对”的预算校验；把当前产物改写成旧版本号等于主动声明旧格式，离线无法与真正的旧产物区分（与工具内容核验的离线边界一致），不能据此声称严格契约已核对。

　　`refine_ir_with_flagtree.py --profile` 在给出放置 sidecar 时要求完整的 `compiler` 与 `compiler.hardware` 块（DPU、tasklet、MRAM/WRAM/DMA、编译目标与非空编译画像摘要），并把 sidecar 引用的 PIM MLIR 当作同一编译契约的一部分：每个 `operators.*.pimir_path` 必须能解析到实际文件（原路径、sidecar 同级 `pimir/` 或显式搜索目录），`pimir_sha256` 必须与内容一致。当前 schema（`version=5`）还要求 `compiler.strategy.dpu_ids` 覆盖全部 DPU，并按每条引用的 `pimir_route` 核对预算属性：A 路六项齐全且与 `compiler.hardware`/画像一致，B 路只豁免 DPU/tasklet 编码，MRAM/WRAM/DMA/目标仍须一致；同一文件被多次引用时，缓存只避免重复读盘，每条引用仍独立核对路由与属性；`strategy.num_dpus` 必须等于请求的 DPU 数。显式旧 schema 保留原有“只核对已写出属性”的兼容行为。映射合法性按后端区分：CEVA 画像要求顶层 `dpu_to_cluster` 显式给出与 DPU 数相同的一对一映射，且每项落在画像的“设备数 × 每设备簇数”范围内；通用 PIM 允许缺省映射，也允许原生生成器 `export_fixed_pu_mapping.py` 的“同段逻辑 DPU 共用一个 ClusterPU、分配不同 TensorPU”共享映射，仍逐项核对数量、类型、非负与物理范围，由 GeneSim 既有的 TensorPU 轮转与容量校验判定合法性。任何矛盾都在读取本地形状与成本编译前拒绝。正式调优的仿真前校验、独立精化与 GeneSim 的 CEVA 启动前校验共用上述判据：GeneSim 启动时同样核对编译策略、硬件请求、一对一物理映射、PIM MLIR 的编译路由与预算属性，缺块或矛盾即退出码 1；启动校验与运行清单、最终放置重放还共用生效配置的物理范围，越界映射不再等到调度器创建与 TP 展开后才拒绝。没有放置 sidecar 的旧命令仍按单 DPU 默认测量。

　　Pareto 资源轴只有 `total_physical_memory_bytes` 与 `num_physical_cores` 两种：配置解析时即校验 `sweep.pareto.resource`，非法取值在 dry-run、扫描与命令行入口都于任何候选求值前失败，不会先编译/仿真再在报告阶段报错。

　　物理容量、系统预留与可用容量从同一画像派生：`sys_reserve_bytes` 同时进入图编译预算与 GeneSim 的 `pim.sys_reserve_bytes`，通用与 CEVA 后端都按“物理 − 预留”得到可用容量，画像导出的单 TensorPU 限额是 `(物理 − 预留) // 每簇 PU 数`，图编译前置与 GeneSim 容量审计都由这同一组物理量与预留派生，逐字节一致。图编译的常驻台账按执行算子归属权重；同一层同一 KV head 的 K/V 两块共用一个 attention 执行单元，必须落在同一 TensorPU，tensor 身份只负责共享去重。KV 的逐请求峰值取本轮轨迹的上下文上界，并发上限与 GeneSim 的 `scheduler.max_inflight_requests` 相同：同一 TensorPU 上逐请求峰值按请求下标相加后，再取最大的若干个请求求和，与 GeneSim 最终容量审计的算法一致。

　　`result_maturity` 只有独立验证报告通过（schema 11、绑定本轮 IR/放置/trace/成本 sidecar、报告自带用户阈值与训练集拟合输入可重算、逐点来源合法且形状落在声明的校准范围内）、参数来源为实测/拟合/甲方确认、画像为 CEVA 且适用范围的时钟、核数、目标型号、成本模型与通信链路逐项匹配时才是 `validated_in_scope`，否则为 `model_prediction`。仿真摘要必须经本轮运行清单把 IR、放置 sidecar、请求轨迹与成本 sidecar 的哈希与报告 package 交叉绑定，只有两个字段的任意 JSON 不算后端证据。训练集与留出集按完整测点身份互斥：目标、特征、阶段、类别、形状、数据类型、单位、来源、算子与产物相位完全相同（改名不算不同测点）时，报告在任何拟合前被判为改名两用，GeneSim 运行前核对与离线成熟度都不会升级。训练与留出还必须与报告适用范围同一目标配置、逐算子数据类型布局、单位与完整成本模型身份：样本的 `target` 必须逐项声明目标型号、核/engine 配置、执行边界、精度布局与完整成本模型，只写任意标签不算同一目标；任一测点缺目标、跨目标、跨精度或单位不符时，报告只能保持 `report_only`，不得先混合拟合再升级。通用 PIM 画像不套用 CEVA 已验证等级；缺少实测数据与阈值时，默认画像只能用于趋势与排序，不能宣称硬件准确。严格复跑的 `1e-9` 只验证确定性，不代表硬件精度。

　　硬件扫描的来源降级精确到实际变化的生效叶子：通用画像的容量/预留写 `pim.*`，CEVA 写 `ceva.*`，周期表写对应原语叶子。父键（如整表 `pim.pim_trace_cycle_costs` 或整组 `comm_config.links`）只表示整组语义字段本身；扫描某个原语时新增该原语的 assumption 叶子，父键下未变化的原语继续保留原有 customer/fitted 证据，不会被整组覆盖。

## 七、代价模型校准

```bash
python scripts/calibrate_profile.py \
    --samples observations.jsonl \
    --training-dataset calib --validation-dataset heldout \
    --profile contracts/profiles/ceva_64core_4e.json \
    --simulation-summary test-results/tune-quick-4d/sim/<sim_key>/attempt-*/results/summary.json \
    --thresholds thresholds.json \
    --package package.json \
    --artifacts artifacts.json \
    --scope scope.json \
    --profile-out /tmp/profile_calibrated.json \
    --out /tmp/calibration_report.json
```

　　样本按阶段与类别（prefill/decode × compute/dma/host/communication）拟合，并保留测点身份：compute 用 `operator` + `artifact_phase`、host 用主机算子类型、communication 用链路名。计算、DMA、主机与通信的计时字段在画像里是全局的，同一类别用两个阶段的联合样本得到一个冻结参数组；写回前先按最终画像字段汇总所有身份的冻结值，同一字段只以原画像值为基数写一次：同值系数合并逐身份证据，不同系数在任何画像写入前拒绝，绝不逐身份连乘或后写覆盖。DMA 样本必须显式用 `unit: cycle`（纳秒先按画像时钟换算）。DMA 样本若带 `features.operator`（`MRAM_READ`/`MRAM_WRITE`）、`features.transfer_count` 与 `features.total_bytes`，直接复用 GeneSim `scripts/calibrate.py` 的相对误差 WLS 拟合，只把训练集出现的方向写回 `LOADN`/`LOADN_PER_BYTE` 或 `STOREN`/`STOREN_PER_BYTE`；其余分组只把缩放系数写回该身份能归属的字段（`artifact_phase` 对应的周期原语、`cpu_model.base_latency_us` 或该链路的基础延迟）。单点 compute 不会改动未测的 SILU，单链路不会改动其他链路；无法归属到画像字段的身份跳过写回，全部身份都不可写时整次写回拒绝，不会把单点结论外推成全局校准。写回的是实际计时字段与逐字段证据，不是只加一个来源标签。报告状态不是 `passed` 时校准入口拒绝写回画像。

　　报告由 GeneSim 的 `ceva_validation` 唯一契约构造与校验：计算/DMA 样本要带 `operator` 与 `artifact_phase` 才能绑定算子编译产物；`kind=host` 的样本用 `operator` 填主机编排算子，`kind=communication` 的样本用 `operator` 填链路名，并覆盖 prefill/decode 两个阶段。每个训练与留出样本的 `target` 必须是规范化的目标配置身份，不能只写任意标签：至少包含 `target_name`、`engines_per_core`、`core_count`、`used_core_count`、`execution_scope`、`dtype_layout` 与完整 `cost_model`，并与 `--scope` 的同名字段逐项相同。生成报告前可用 GeneSim `ceva_validation.target_config(scope)`（调优侧同名 `tune.calibration.target_config`）得到该映射；只写 `{"profile": "ceva"}` 之类的标签会被判为缺少目标配置，训练与留出同步换目标同样保持 `report_only`。`--artifacts` 要求每个角色登记 `path` 与 `sha256`，报告会按当前文件字节复核；`model_ir` 角色填**本次仿真实际消费的 IR**（通常是 `refined.ir`）。`--package` 是 schema 11 的产物绑定，其中的 IR/放置/trace 哈希必须与这些文件一致；`--simulation-summary` 必须是本轮真实 pipeline 仿真的 `summary.json`（`result_producer=pipeline_simulator`），且同一结果目录下的 `run_manifest.json` 记录的 IR/放置/trace/成本哈希必须与 `package` 一致。`--scope` 必须给出 `dtype_layout` 逐算子精度映射、`cost_model`（含 `loop_encoding`、`cycle_costs`、`zone_cycle_factors`）、`timing_boundary`、`request_boundary` 与 `shape_range`；GEMM 样本按 `shape_range` 的 `M/N/K` 轴区间核对，其他算子没有统一轴语义，必须在 `shape_range.operator_shapes` 里按算子声明本次运行实际执行过的形状列表（如 `{"SILU": [[8, 16]]}`），缺声明或样本不在其中都只能保持 `report_only`。报告生成、离线成熟度与 GeneSim 运行前复核共用同一形状契约。报告会保存用户阈值与训练集逐点输入，升级判定时重新拟合、复算统计并重做阈值判定。缺产物绑定、缺适用范围键、缺主机或通信证据、缺任一分组阈值、来源伪造、形状越界或没有真实仿真运行清单时，报告状态保持 `report_only`，不能升级。

## 八、常见失败

| 现象 | 原因与处理 |
| --- | --- |
| 未知字段/非法算法参数 | 配置在编译前被拒；按报错字段修正 |
| `prefill_seq_len` 覆盖不了提示长度 | 提高 prefill 或缩小轨迹窗口 |
| `max_seq` 小于上下文上界 | 显式给足或保持 null 由请求推导 |
| 输出目录已有日志 | 加 `--resume` 或换 `--output-dir` |
| 续跑配置与日志不一致 | 续跑必须保持原生效配置、源码、工具、环境与输入；只允许放宽外层预算。换配置请换输出目录 |
| 续跑预算小于日志 | 外层预算只允许保持或扩大；收缩会被显式拒绝，恢复原预算或换输出目录 |
| 续跑/严格复跑提示输入身份不一致 | `best.json` 或响应日志记录的模型/IR/轨迹/max_seq/请求数与本次不一致；恢复原输入或换输出目录 |
| 轨迹请求数不足 | 已有轨迹少于配置的 `num_requests`，补齐请求或把 `num_requests` 调到可用数量 |
| 输入快照已存在且内容不同 | 输出目录里已有另一份输入快照，系统拒绝覆盖旧证据；换一个 `--output-dir` 或恢复原输入 |
| 提示输入快照在读取后发生变化 | 同一轮运行期间模型/IR/轨迹文件被原地改写（含编译窗口内改写）；恢复原输入后换输出目录重跑，不能沿用旧编译/仿真键 |
| 严格重放提示 `targets_include_relay` | 修复前生成的最终放置文件需用新版本重新生成 |
| 缓存不完整 | 编译目录缺 `artifacts.json`、缺 partition plan/placed.ir/放置 sidecar/refined IR/成本 sidecar、清单为空、sidecar 引用的 PIM MLIR 未登记或哈希不符，或仿真目录没有 `completion.json`、缺 summary/autotune 指标/配置/manifest/契约/最终放置、trace 与最终执行图不一致、键与 manifest 不同源或文件哈希不符，本轮会隔离或重建 |
| 内层指标不完整 | 启用原生搜索但 `autotune_metrics.json` 缺失，或搜索/最终执行计数矛盾，或算法/搜索后端/验证后端与生效配置不一致；固定映射与严格重放必须记录零搜索和一次最终执行 |
| 续跑提示响应与轮次冲突 | 同序号响应的状态、分数、编译/仿真键或运行身份与轮次日志不一致，或响应序号不连续；日志已损坏，换输出目录重跑 |
| 轮次重放提示编译目录缺失 | 成功轮次的编译产物目录被删除或清单不完整；恢复原输出目录或重新生成该轮产物 |
| 轮次重放被拒绝 | 某轮记录的编译/仿真键与当前输入、源码或配置重算结果不符，或产物清单/最终放置失效；恢复原环境或重新生成该轮产物 |
| predictor 配置被拒 | `host_relay_overhead_ns` 不是 100 ns，或检查点不存在/schema 版本不符；predictor 特征只支持固定 100 ns 中转 |
| 扫描点 `invalid` | 画像构造或物理资源组合（如 `num_dpus` 超过物理簇数）在编译前被拒；原因写入 `sweep.jsonl` 与报告 |
| 校准报告 `report_only` | 缺真实 pipeline 运行清单、产物 path+sha256、package 绑定、逐算子 dtype 布局、主机/通信逐点样本、任一分组阈值，来源/形状/阈值无法复算，训练点与留出点按完整测点身份相同（改名两用），或样本 `target` 只是标签/与 `--scope` 目标配置不一致（含全部测点同步换目标） |
| 仿真完成清单拒绝 trace | trace 算子集合必须与最终挂 PU 的目标集合精确相等；缺 trace、图外 trace 与搜索后迁离 PIM 的残留 trace 都会被拒绝，需要清理残留或重新仿真 |
| 仿真实际原生后端与键声明不一致 | 原生库在探测时可加载，但 GeneSim 初始化、轨迹装载（DRAMsim3 返回非零、AttAcc 缺装载入口）或循环绑定失败后回退到了内置轨迹；修复库/配置后重跑，或移除不可用的原生库让键明确声明回退 |
| 硬件扫描 best 被拒 | `best.json` 不是成功记录、分数缺失，或其模型/IR/轨迹/请求数/画像与本次扫描不一致；生效配置被原地改写、移动或替换导致内容摘要/历史编译键/仿真键对不上；换用同一负载且内容未变的 best，扫描配置只改扫描字段、预算与输出目录 |
| 候选一致性校验报编译目标不一致 | 放置 sidecar 的 `compiler.hardware.compiler_target` 与画像不符，或旧 sidecar 缺少该字段；用当前导出器重新生成编译产物 |
| 候选一致性校验报工具链不一致 | sidecar 的 `compiler.toolchain` 缺失、工具摘要为空/全零/被改写，或图编译源码摘要、编译指纹、实际工具内容与当前编译器不符；`triton-opt`、`libtriton`、`mlir_translate`、`gcc` 四类工具都会按本机权威路径核对内容，本机环境、图仓库 `paths.json` 或当前进程能解析到实际工具时按实际内容比对，**本机存在工具但声明把路径与摘要同时置空同样按未声明拒绝**；声明前缀存在但为空/缺二进制不再被当作已核实，工具内容读不出来也拒绝；不得手改 sidecar，重新编译或恢复本轮实际工具链 |
| 精化入口报缺少 compiler 契约 | `refine_ir_with_flagtree.py --profile` 同时给了放置 sidecar，但 sidecar 缺 `compiler`/`compiler.hardware`、`compiler.strategy`、DPU/tasklet、编译目标或非空编译画像摘要；或策略 `num_dpus`/`dpu_ids`、顶层 `dpu_to_cluster` 数量/类型/范围与 hardware 的 DPU 数矛盾；或 CEVA 画像缺显式 `dpu_to_cluster`、`dpu_ids` 为空、多个逻辑 DPU 映射到同一物理核（通用 PIM 的共享 ClusterPU 映射是合法的，不会报此错）；或 sidecar 引用的 PIM MLIR 读不到、哈希不符、缺 `pimir_route`/路由与算子类型不符、A 路缺六项预算属性、B 路缺 MRAM/WRAM/DMA/目标，或写出的预算属性与 `compiler.hardware` 不一致（schema 5 起；显式旧 schema 只核对已写出属性）；用当前导出器与当前算子编译产物重新生成，或去掉 `--placement` 走旧单 DPU 测量 |
| 候选一致性校验报 schema 版本 | 放置 sidecar 的顶层 `version` 不是当前导出器的 5（缺版本、旧版本或未来版本）；旧产物需用当前导出器重新生成，不能靠删改契约字段继续 |
| `sweep.pareto.resource` 被拒 | 资源轴只允许 `total_physical_memory_bytes` / `num_physical_cores`；在配置解析阶段报错，不会先执行候选 |
| 候选失败 | 该候选被淘汰并保留诊断，其余候选继续 |

## 九、交付环境实测记录

　　下表记录本轮及历史版本的实测口径；标注第十三轮的行对应当前源码，第十二轮及以前的行保留作对照。换机器或换数据需要重新测量，实施文档第四节给出完整命令与产物。

| 项目 | 配置 | 实测 |
| --- | --- | --- |
| 机器 | 112 核 CPU、无可用 CUDA（成本精化走 oplevel）、503 GB 内存 | 本轮 quick/strategy 与多个验证任务并发执行，耗时含并发影响 |
| quick 首轮 | CEVA 64 核、8 DPU、2 个候选、1 个请求、prefill 16 | 第十五轮实测 282.3 秒 + 263.4 秒（干净目录 `/tmp/tune-quick-fix-round15`，两个候选各含一次完整图/算子编译与仿真，机器同时有 G 主链回归并发）；时延 10.304023382370014 / 19.00700623227144 秒；`search_coverage` 为 2 次新 pipeline、0 编译/仿真缓存命中、2 次最终映射重放；PIM trace 7714 / 7138；sidecar 工具链声明通过 `verify_compiled`（四类工具内容核验、PIM MLIR 预算属性与映射）与 GeneSim 启动前严格契约（工具链、硬件、策略、一对一映射、PIM MLIR 引用）；续跑 0 执行/2 回放、严格复跑 1e-9 一致、轮次重放两候选一致 |
| quick 编译缓存命中 | 同配置，2 个候选复用编译产物 | 同目录续跑为 0 次实际执行、2 次回放；`rounds.jsonl` 的编译/仿真缓存命中状态保持首次记录值 |
| quick 续跑 | 同一输出目录 `--resume` | 第十三轮实测 0 次实际执行、2 次回放，最优分数不变；纯回放调用 `search_coverage.real_search=false`，原始搜索证据以 `rounds.jsonl` 与首次日志为准 |
| quick 严格复跑 | `--evaluate-candidate best.json --cache-mode strict` | 第十一轮约 5 分 20 秒；强制重编（新 attempt `attempt-5bc86450`）并新开仿真 attempt 后分数 10.304023382370014 与首次一致（1e-9），旧 attempt 保留；完成清单记录 `execution_mode=replay` |
| 真实 strategy | 通用 PIM 8 设备 × 8 簇、1 个候选、2 次内层搜索评估、1 个请求 | 首轮图/算子编译约 4 分 41 秒；缓存命中后搜索 + 最终重放 3 分 50 秒 |
| 放置严格重放 | 从原 IR 重建并应用保存映射，关闭搜索 | 第十轮约 5 分 20 秒（重编 + 重放），分数 10.304023382370014 与搜索最优一致（1e-9） |
| 三种外层算法 | 第十一轮最小真实搜索：三种算法各用独立输出目录（干净编译/仿真缓存），8/16 dpu 两候选空间，SA/GA/Tabu 各 2 次逻辑/2 次实际评估 | SA 903.0 秒、GA 899.6 秒、Tabu 885.3 秒；三者均 2 次新 pipeline（`sim_cache_hit=false`）、0 编译/仿真缓存命中、6 次新后端执行、2 次最终映射重放，`real_search=true`；最优一致（3.4946278379822013，初始 6.317389525660534） |
| 硬件扫描 | 第十一轮内容绑定对照：2 个时钟点（0.63/0.5 ns），真实 best 与真实编译/仿真 | 2/2 成功（10.304023382370014 / 8.199983301026025 秒），同一 compile_key、不同 sim_key；binding 生成时与当前摘要一致；同路径原地改写按生成时摘要拒绝，去掉摘要后仍按历史仿真键拒绝，仅移动内容不变接受。历史 16 点扫描结果见实施文档第四节 |
