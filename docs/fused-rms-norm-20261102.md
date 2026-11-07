# RMSNorm 链融合（2026-11-02）

## 目的

G1 的测量把瓶颈定在单进程 GIL 与命令数上：单请求 decode 一步 934 条命令，
约 0.97 ms CPU/条，时间花在逐条 Python 调度而不是数学运算。这次把每条 RMSNorm
的 7 条命令压成 1 条，作为「减少命令数」的第一步。

## 改法

| 位置 | 改动 |
|---|---|
| `graph/fuse_pim.py` | 新增 `fuse_rms_norms(gm)`，走 `_fuse_rms_norms(for_exec_plan=True)` |
| `runtime/compile.py` | `propagate_specs` 之后对 prefill 各桶与 decode 图各调一次 |
| `runtime/exec_plan_gen.py` | 跳过被吸收的节点；锚点改发融合命令 |
| `runtime/kernels.py` | 新增 `rmsnorm_kernel`，按图上的顺序算整条链 |
| `contracts/op_semantics.py` | `RMS_NORM_KERNEL = "pim.normalize"` |

1. **标注**沿用 GML 侧那一套：锚点仍是链尾乘缩放的那个 `mul`，链首到链尾之间
   的节点打 `ABSORBED_META_KEY`。执行计划这一版多两条门槛：
   链上不能有重分布边（被吸收的节点不发命令，边就没有落点），必须带缩放权重
   （没有权重那条链折了只剩归一化，与图的语义不符）。
2. **时机**必须在 `propagate_specs` 之后：第一条门槛要看传播出来的重分布边。
3. **发命令**：被吸收的节点跳过，但它名下的重分布边照发（边挂在目的节点上）；
   锚点的实参换成 `[链首输入, 缩放权重]`，`kernel` 换成 `pim.normalize`，
   `epsilon` 从图上读出来放进 payload。
4. **算**：`pow → mean → 加 ε → rsqrt → 乘 → 落 f16 → 乘缩放` 在一次调用里做完。

## 关键决定：不调 `pim.normalize` 的编译内核

C 侧那个内核在 f32 里把缩放一起乘完，只在最后落一次 f16；而图上 `to` 节点在
乘缩放**之前**就已经落过一次 f16。少这一次取整，解码 token 会分叉——实测
4 条请求 8 步，第 3 条从 `355, 842, 842, 690, 690, 875 …` 变成
`557, 765, 875, 875, 875 …`。

所以融合命令在 numpy 里按图上的顺序算，名字仍叫 `pim.normalize`（它是 GML 侧
同一个助记符），但不进算子编译器的通路。代价是这条链在模拟器上没有 `.so`；
收益是逐位对齐，token 与融合前完全相同。

## 结果

单请求一步的命令数（4 台 DPU、4 层、hidden 1024）：

| 计划 | 融合前 | 融合后 | 变化 |
|---|---|---|---|
| decode | 934 | **718** | −23.1% |
| prefill（桶 8） | 934 | **718** | −23.1% |

每条链在每台 DPU 上少 6 条（`pow` / `mean` / `add` / `rsqrt` / `mul` / `to`），
9 条链 × 4 台 = 216 条，正好是差值。

`scripts/bench_parallel.py`（4 请求 × 8 步，中位数 3 次；同一台机器前后脚各跑一次）：

| P | 墙钟前 ms | 墙钟后 ms | 变化 | 内核调用次数 |
|---|---|---|---|---|
| 1 | 25 116 | 21 047 | −16.2% | 28 544 → 21 632（−24.2%） |
| 2 | 20 990 | 18 058 | −14.0% | 同上 |
| 4 | 20 525 | 16 880 | −17.8% | 同上 |

两组的生成 token 逐字相同：
`[[243 …], [434, 434, 434, 639, …], [355, 842, 842, 690, 690, 875 …], [995 …]]`。
墙钟在这台机器上抖动 ±20%，只有命令数与内核调用次数是可复现的口径。

## 为什么没到 650

原验收写的 934 → ≤650 是把「残差 `add` 也折进来」算进去的，这条不成立：
Llama 里每个残差的结果有两个消费者——下一个 RMSNorm 的 f16→f32 提升，以及
下一个残差。吸收它等于让其中一个读不到值。实测 8 层链里只有末尾那个 norm 的
残差是单消费者，折它不值一提。所以这一版只到 −23%。

## 已实施的测试

| 文件 | 用例 | 判据 |
|---|---|---|
| `tests/test_fuse_pim.py` | `test_exec_plan_fusion_tags_the_chain_tail_and_its_inputs` | 锚点是链尾的 `mul`，实参是链首输入与缩放张量 |
| `tests/test_fuse_pim.py` | `test_exec_plan_fusion_skips_a_chain_with_a_redistribute_edge` | 带重分布边的链不折 |
| `tests/test_exec_plan_gen.py` | `test_the_plan_emits_one_command_per_rms_norm` | 计划里每条链每台 DPU 一条 `pim.normalize`，`pow`/`mean`/`rsqrt` 不再出现 |
| `tests/test_kernels.py` | `test_fused_rms_norm_rounds_before_the_scale` | 乘缩放前先落 f16，逐元素相等 |
| `tests/test_runtime_compiled_coverage.py` | 三处调整 | 编译集合里去掉 `normalize`；计划里的内核允许 `pim.normalize` 不进编译器 |

## 当前不足

- 没折链首那次 f16→f32 提升（`to_7`）。它每台 DPU 多出一条 `convert`，
  9 条链共 36 条，折掉能到 682（−27%）。
- 带重分布边的链一律不折。tp4 的 decode 图里没有被拦下的链，别的策略没有验证。
- 命令数 718 里，视图类仍是最大一块：`slice` 64、`transpose` 64、`view` 48、
  `neg` 32、`cat` 32、`unsqueeze` 32，共 272 条。下一步该看 RoPE 与拆头那一段
  能不能整段压掉。
