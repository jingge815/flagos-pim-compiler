# 补齐 tp4_pp2，并让两个交付指标由产物自身保证

## 一、这次要解决什么

校对两个交付指标时发现：目标一（PU 映射进模拟器代价）已经打通，目标二（TP/PP 从
图编译到模拟器的评估验证）主体已通，但有四处缺口。前三处让「四种切分策略都验证过」
这句话不成立，第四处更根本——**两个指标是否满足，取决于配置文件写没写对**。

| 缺口 | 现象 | 危害 |
| --- | --- | --- |
| tp4_pp2 完全没有产物 | 有配置文件，但方案、sidecar 都不存在 | 这一档跑不起来，宽度 4 从未实测 |
| tp4_pp2 配置漏写严格模式 | 缺 `require_compiler_pimir: true` | 读不到 pim mlir 时静默退回手写模板，数字悄悄偏掉 |
| 两份 sidecar 没有内容哈希 | tp1_pp8、tp8_pp1 的 `pimir_sha256` 全为空 | 无法确认 sidecar 与当前算子编译产物是同一套 |
| **两条静默退化路径** | 见下表 | 「跑通」不等于「按指标要求跑的」 |

第二条此前文档没有记录。它恰好撞上 `docs/pimir-cost-integrity-20260905.md` 第 6.4
节自己写下的隐患：严格模式默认关闭，新配置忘记打开就会静默退化。tp4pp2 就是那个
「忘记打开」的实例——隐患从假设变成了事实。

### 1.1 两条静默退化路径（本轮的主要工作）

| 指标 | 退化点 | 退化后的后果 |
| --- | --- | --- |
| 一、PU 映射 | sidecar 缺 `dpu_to_cluster` 时按 `dpu_id % len(clusters)` 取模，**无任何日志** | 仿真跑完，但同段 DPU 可能落在不同 ClusterPU、段内通信走慢链路，等于没按 PU 映射评估 |
| 二、TP/PP 评估 | `require_compiler_pimir` 默认 `false`，配置漏写就退回手写模板 | 分块换成 conf 常量 32，实测分块完全不进代价链 |

两条都违反 CLAUDE.md 的「不写防御性兜底，契约不满足就直接抛」。修法不是把兜底删掉
（旧 sidecar 还要兼容），而是**让产物自己声明它要什么**，这样漏写配置也不会退化。
这正是 6.4 节指出的根治方向。

## 二、改了什么

### 2.1 配置：补上严格模式开关

`genesim/conf/sim_llama2_7b_pp_tp4pp2_globalcost.yaml` 增加四行（含注释）：

```yaml
scheduler:
  compiler_placement_file: "models/llama2_7b_tp4_pp2_placement.json"
  require_compiler_pimir: true
```

改完之后四份 `*_globalcost.yaml` 口径一致。这个开关的作用点在
`genesim/src/scheduler/gene_sim_scheduler.py:501` 和 `:517`——两处 pim mlir 读取
失败的分支，开了就抛错，不开就只打一条警告然后退回模板。

### 2.2 产物：生成 tp4_pp2 全套，重导两份旧 sidecar

三条命令，都走既有脚本，没有新增代码：

```bash
# 1) GeneSim 定 PU 映射
cd genesim
python scripts/export_fixed_pu_mapping.py --num-stages 2 --num-dpus 8 \
    --ir models/llama2_7b.ir --out models/llama2_7b_tp4_pp2_plan.json

# 2) 图编译 + 算子编译，产 sidecar（另两份同法，换 plan 文件）
cd flagos-pim-compiler
python scripts/export_pp_placement.py \
    --partition-plan .../llama2_7b_tp4_pp2_plan.json --measure-kernel-tiles
```

### 2.3 让指标由产物自身保证（本轮主要改动）

补配置只解决了 tp4pp2 这一份，下一份新配置漏写还会退化。所以让 sidecar 自报家门。

**导出侧**：`genesim_bridge/placement_export.py` 在 sidecar 顶层增加一个字段，
值就是 `--measure-kernel-tiles` 传了没传：

```python
"requires_pimir": bool(measure_kernel_tiles),
```

**消费侧**：`gene_sim_scheduler.py` 新增 `_apply_sidecar_strict_pimir`，在加载
sidecar 时读这个声明，带产物就把严格模式抬起来：

```
sidecar 声明 requires_pimir=true
        ↓
require_compiler_pimir 自动置 true
        ↓
pim mlir 读不到 → 报错（而不是退回手写模板）
```

**配置是下限，不是上限**。这一点走了一次弯路，值得记下来：第一版设计成「配置显式
写过就以配置为准」，想给 A/B 对照留个手动关闭的余地。写完测试立刻挂了——
`conf/sim.yaml:46` 有一行 `require_compiler_pimir: false` 作为基线被所有策略配置
继承，于是「配置是否显式写过」恒为真，sidecar 的声明永远被压制，整个机制形同虚设。

改成「配置只能抬高、不能压低」之后测试通过。A/B 对照本来也不需要这个余地：
`ab_global.yaml` 用的是 `use_compiler_local_shapes: false`（改形状口径），与 pim
mlir 读不读得到是两件事。这也印证了 CLAUDE.md 的「不预造抽象」——那层让步是想象出
来的需求。

**取模退化改为可见**：PU 映射缺失时仍然退回取模（要兼容旧 sidecar），但现在会打一条
告警说明「性能评估不反映声明的 PU 映射」，整轮只提示一次、不按算子刷屏。

**全流程脚本加一道闸**：`scripts/run_full_pipeline.py` 核对 sidecar 必须自报
`requires_pimir=true`——本步骤明明编了算子，产物却不声明，说明导出侧退化了。

### 2.4 代码量

| 仓库 | 净增删 | 其中测试 |
| --- | ---: | ---: |
| flagos-pim-compiler | +41 / -0 | 27 |
| genesim | +150 / -3 | 97 |
| 合计 | **+191 / -3** | 124 |

产物文件（四份 plan、四份 sidecar、pim mlir 缓存）不计入。测试占比 65%，因为这轮
改的是「防止静默退化」的机制，而机制本身是否生效只能靠测试锁住——第一版设计的缺陷
就是测试抓出来的。

## 三、验证结果

### 3.1 四份 sidecar 现在口径一致

```
策略      requires 分片/算子 kernel_tile_n       无哈希 缺文件 哈希不符 cluster项
tp1_pp8   True     1         {512:160, 256:64}   0      0      0        8
tp2_pp4   True     2         {512:160, 128:64}   0      0      0        8
tp4_pp2   True     4         {512:160,  64:64}   0      0      0        8
tp8_pp1   True     8         {512:160,  32:64}   0      0      0        8
```

四点值得注意：

- **`requires_pimir` 四份都是 true**，所以四种策略无论配置怎么写都会进严格模式，
  pim mlir 读不到直接报错。这是本轮机制落地的证据。
- **`dpu_to_cluster` 四份都有八项**，PU 映射不会退回取模换算。
- **分片数严格等于 TP 宽度**，每台 DPU 都有实际任务（tp4_pp2 是八台各 112 个
  分片），不存在早期「只导出编号最小那台」的失真。
- **分块随宽度单调收窄**：256 → 128 → 64 → 32。这是算子编译器按 WRAM 预算真实
  搜出来的，不是配置里拍的常量（GeneSim 默认常量是 32）。

### 3.2 宽度 4 确实触发了新的分块组合

`docs/genesimsupporTp-20260905.md` 第 8.5 节记录过一个担心：宽度 4 的本地宽度是
1024/2752，落在哪个分块值上未知，而且这个映射是**非单调**的，不能从宽度 2 和 8
都通过推断宽度 4 一定通过。

实跑结果：`tile_n = 64`，是宽度 1、2、8 三档都没出现过的值，能正常编译并进入
代价链。这条担心现在有了实测答案。

### 3.3 tp4_pp2 的 trace 全部来自算子编译器

```
GEMM trace 文件数: 896        （= 224 算子 × 4 分片）
trace_source:      {'pimir': 896}    ← 零个退回手写模板
(tile_n, k_iterations): {(512,128):384, (512,32):128, (64,128):256, (512,86):128}
```

896 这个数字本身就是 TP 展开生效的证据：图上真的按分片展开成了四份，而不是把一个
算子钉在一台设备上。仿真结束后复查仍是 896/896 pimir，说明严格模式全程生效、没有
中途退回模板。

### 3.4 tp4_pp2 端到端仿真结果（首次实测）

`conf/sim_llama2_7b_pp_tp4pp2_globalcost.yaml`，十个请求，跑了约十分钟：

| 指标 | 数值 |
| --- | ---: |
| total_time_s | 1181.24 |
| tokens/s | 6.84 |
| 完成请求 / 处理 token | 10 / 8084 |
| 平均首 token 延迟 | 1014.09 s |
| 平均每 token 延迟 | 0.807 s |
| 累计通信时间 | 2.785 s（占比 0.02%） |
| 峰值常驻内存占用 | 75.39%（fits=True） |

结果已存到 `/tmp/full_pipeline_tp4pp2`。**注意 `genesim/results/` 是固定路径、会被
下一轮仿真覆盖**，跑完要立刻拷走再分析——本轮核对时就遇到过 `results/summary.json`
还是七点多旧产物的情况，容易误当成本次结果。

容量这一项值得单独说：段内四台共用一个 ClusterPU，每段权重 6176 MiB ≤ 8192 MiB，
峰值占用 75.39% 通过。这是 `export_fixed_pu_mapping.py` 挑选映射时就按容量约束算过
的，实测与预期一致。

### 3.5 回归测试

| 范围 | 本轮 | 改动前基线 |
| --- | --- | --- |
| 图编译器快速回归（不含 7B 全量） | 247 passed, 42 deselected, 18.03s | 246 passed |
| 图编译器放置导出测试 | 8 passed, 5.92s | 7 passed |
| GeneSim 放置逻辑测试 | 53 passed, 6.03s | 49 passed |

新增五个用例，全部针对本轮机制：

| 用例 | 锁住什么 |
| --- | --- |
| `test_requires_pimir_declares_whether_opcompiler_products_are_carried` | 纯放置导出声明 false，且不写 pimir_path/kernel_tile_n |
| `test_sidecar_requires_pimir_enables_strict_mode` | 声明带产物时自动进严格模式 |
| `test_config_false_cannot_disable_sidecar_requires_pimir` | 配置是下限不是上限（第一版设计的缺陷就在这里） |
| `test_placement_only_sidecar_stays_lenient` | 纯放置导出不被拖进严格模式 |
| `test_modulo_fallback_warns_once` | 取模退化必须留告警，且只提示一次 |

## 四、链路全景

```
GeneSim 定 PU 映射                    export_fixed_pu_mapping.py
  dpu_to_cluster: 段内共用一个 Cluster
        │
        ↓
图编译器按方案切分                     export_pp_placement.py
  TP: q/k/v/gate/up 列切, o/down 行切
  PP: 32 层分成 num_stages 段
        │
        ↓
算子编译器编本地分片形状               FlagTree: pim-tile-to-budget
  每种本地形状编一次, 搜出真实分块
        │
        ↓
sidecar 回传                          llama2_7b_<策略>_placement.json
  shards / local_*_features / kernel_tile_n
  pimir_path / pimir_sha256 / dpu_to_cluster
        │
        ↓
GeneSim 按 sidecar 评估                gene_sim_scheduler.py
  行切后插 ALL_REDUCE 节点 (:1099)
  同 Cluster 512 GB/s, 跨 Cluster 128 GB/s (node.py:143)
```

## 五、当前不足

以下几条是既有的，本轮没有改变，交付时不宜声称已解决。

### 5.1 集合通信按点对点近似

归约节点的入边字节数是准确的，但耗时是把一次 all-reduce 当成 N 条独立点对点传输
相加，没有建模环形/树形归约的优化，也没有建模多条传输之间的带宽竞争。方向上偏
保守（倾向高估）。详见 `genesimsupporTp-20260905.md` 第 8.1 节。

因此**绝对耗时数字适合同类配置之间的相对比较，不宜当作绝对性能预测**。

### 5.2 归约节点落在 GPU 上

`_attach_reduce_node` 把归约节点标成 `device_hint = "gpu"`
（`gene_sim_scheduler.py:1121`），因为 PIM 侧没有 ALL_REDUCE 的 trace 编译器。
物理含义是「经主机归约」，不是 PIM 原生归约，其计算开销是估算值。

### 5.3 算子覆盖范围

IR 共 3491 个算子，进入切分导出和算子编译的只有 224 个 GEMM。注意力内部的 3072 个
算子（GEMV_SCORE / SOFTMAX / GEMV_CONTEXT）不拆分，改为按 head 归属重连上游
（`gene_sim_scheduler.py:967-1096`）——设计上成立（GeneSim 的 IR 本来每个 q_head
一条独立链，边字节数已按 head_dim 算好），但它们的代价不来自算子编译器。

### 5.4 严格模式的默认值仍是 false（但已不再是隐患）

`require_compiler_pimir` 的配置默认值仍是 `false`，不过本轮之后它只是「不主动
要求」：带算子编译产物的 sidecar 会自己把它抬成 true。所以新配置漏写那一行**不再**
导致静默退化——这是 2.3 节的机制。

仍然存在的边界：只有**带产物的 sidecar** 才有这个保护。如果将来出现「配置指向了
一份纯放置 sidecar，但使用者以为它带产物」的情况，依然只会拿到手写模板的数字。
这种误用现在至少是可查的（sidecar 里 `requires_pimir: false` 写得很明确）。

### 5.5 缺 dpu_to_cluster 时仍会取模，但不再静默

sidecar 没有 `dpu_to_cluster` 时，`gene_sim_scheduler.py` 仍退回
`dpu_id % len(cluster_keys)`——这是为兼容旧产物保留的，不是错误。本轮的改动是让它
**打一条告警**，说明「同段 DPU 可能落在不同 ClusterPU、性能评估不反映声明的 PU
映射」，整轮只提示一次。

四份 sidecar 现在都带映射，正常路径不会走到这里。

### 5.6 本轮未核查的项

- PP 的流水气泡口径。已确认 `stage_view` 是观测性的、不控制执行
  （`gene_sim_scheduler.py:4592-4596`），算子就绪只由 DAG 依赖、传输和资源可用性
  决定，气泡是测出来的而非人为施加。本轮 tp4_pp2 的产出进一步印证了这一点：
  `pipeline_bubble_definition` 字段自己声明 `reporting_diagnostic_only`，且
  `blocking_bubble` 为 0、25.77% 全部记在 `resource_wait` 名下。也就是说这些气泡
  数字是诊断信息，不宜直接当成"流水效率"对外报。micro-batch 调度本身仍未核对。
- prefill / decode 的建模细节。已知仿真按 `full_prefill_decode` 模式跑
  （summary 里的 `request_execution_mode`），且分别产出 TTFT 与 TPOT 两组指标，
  说明两个阶段是分开计时的；但 KV cache 增长如何影响逐 token 延迟仍未追到代码。
- 图编译器侧 TP 切分逐元素对拍的具体断言。注意一个错位：单测把 tp4_pp2 选作
  **唯一**代表策略做真实模型对拍（`tests/test_strategy_llama2_7b.py:46-52`），而
  仿真侧 tp4_pp2 恰恰是本轮之前唯一跑不起来的一档——数值正确性和性能评估两条路径
  的覆盖点是错开的。
