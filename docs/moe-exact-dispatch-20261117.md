# MoE exact_dispatch 第三模式落地（M2③④⑤，2026-11-17）

> **2026-11-19 更新：exact_dispatch 已转默认**（用户拍板）。两仓缺省规则同步
> 翻转：`PIM_MOE_DISPATCH` 缺省值 "1"→"2"，即不带环境时 MoE 走精确行数；
> `=1` 显式回 dynamic_dispatch、`=0` 回 dense_sim。CEVA 链路不受影响
> （当前产物非 MoE；MoE 走 CEVA 未指定时 fail-fast 提示选 dense/dynamic，
> 不静默切换）。守护测试反向：缺省即 exact 是被锁定的行为。

## 做了什么

`exact_dispatch` 成为 MoE 第三种执行语义（与 `dense_sim`、`dynamic_dispatch` 并列，
`PIM_MOE_DISPATCH=2|exact` 启用，默认不变）：MoE 不再走「每 token 过全部专家 +
where 掩码清零」的稠密尾链，而是按运行期实际命中行数精确执行——dispatch 读两个
scatter 的结果算出每专家的行索引/权重/行数写进设备路由表，专家链改读紧凑缓冲，
down_proj 输出用 scatter_add 加权散写进累加器，最后跨 DPU 归约写回落地缓冲。

```
dense（改动前）                          exact（现在）
──────────────────────                  ──────────────────────────────
gate linear → softmax/topk (host)       同左
scatter(布尔掩码) → dispatch(记跳过)     scatter(布尔掩码) ─┐
scatter(路由权重) → to copy             scatter(路由权重) ─┴→ dispatch v2 (host)
                                                           │ 写 idx/w 表段 + 清零 acc
per-expert: gate/up/down proj            per-expert: gather_rows → gate/up/down proj
  (读 xs 整份包络)                          (读该专家紧凑缓冲, k_e 行)
per-expert: all_reduce (host)            scatter_add ×E (WAW 链按专家号串行)
where×E + 加权融合闭包 (host)            ONE all_reduce: acc → acc_land (host)
出口 add 残差                            出口 add 残差（改读 acc_land）
```

tiny 模型（2 层 4 专家）decode 计划从 430 条命令缩到 356 条；`skip_value`/`skip_zero`
在 exact 分支为 0（M2⑤ 随尾链退役自动达成），dense 两模式逐字节不动。

## 文件与函数

| 文件 | 改动 |
| --- | --- |
| `graph/moe_exact.py`（新，221 行） | `MoELayerTopology` + `moe_exact_topology`：专家链（down_proj 反向闭包）、待退役尾链（前向闭包减链减种子减出口）、出口投影（exit_consumer 反向闭包，只沿值透传节点下行）。纯图分析，编译与内存规划共用 |
| `memory/mem_planner.py`（+81/-2） | `MoEBuffer`；`_moe_exact_tensors` 按拓扑产 idx/w/compact/acc/acc_land 五类命名瞬态（读者名与 exec_plan_gen 的合成命令登记名一一对应，WAR 链靠它们闭合）；`plan_dpu` 新参 `moe_*_topology`，`DPUPlan.moe_prefill/moe_decode` 带回地址 |
| `runtime/exec_plan_gen.py`（+291/-9） | 模式表 + `_emit_moe_exact_dispatch`（dispatch v2）、`_emit_moe_scatter_add`、`_emit_moe_acc_reduce`；主循环：尾链抑制、边作废、链上 xs 改读紧凑缓冲、出口改读 acc_land、down_out 落地即散写、布尔 scatter 不再触发 dense dispatch |
| `runtime/compile.py`（+25/-4） | 每张导出图各分析一次拓扑（节点名随桶不同），传给 `plan_dpu` 与两处 `build_execution_plan` |
| `tests/test_moe_exact_dispatch.py`（新，591 行） | 12 个测试：模式解析、尾链退役+行级命令数、链首改读紧凑缓冲、moe_rows 写表+行数守恒、exact==dense 逐位、exact==HF 稀疏参考，外加 M2⑥ 矩阵（见下节） |

## 关键决策（为什么这样做）

1. **内核缓存键不含 k_e**（全程不变式 #2）：专家链命令保持包络形状，行级精确性只走
   `hal.moe_rows` 行数通道；行级内核入口按 `arange < rows` 钳位，尾行读第 0 行，不出 NaN。
2. **acc 清零折进 dispatch v2** 而不是独立 dma_in：dma_in 的 payload 带绝对 offset，
   `bind_plan` 只平移 Access 不平移 payload——写进 `cmd.writes` 的 Access 窗口平移才安全。
3. **acc 归约复用 dense 专家分片同一条 `all_reduce` 原语**：`DmaEngine._addr` 按窗口
   平移激活区地址，`CommSegment.src_addr/dst_addr` 在多窗口下天然正确。
4. **dispatch v2 挂在路由权重 scatter**（不是布尔 scatter）：一次主机命令同时完成
   算表、写表、清零 acc、登记 `moe_rows`/`gate_skips`；零命中专家仍走既有 gate 跳过。
5. **WAW 链保证逐位一致**：同 DPU 的 scatter 按 `_deps_of` 生成 waits，发射序=专家号
   升序，每 token 加法序与 dense 的升序累加一致；未选中贡献 dense 侧恰为 +0.0。

## E=2 数值发散修复（M2⑥ 红测揪出的真 bug）

E=4 全部测试通过后，M2⑥ 的 `top-k 全命中` 用例（E=2, topk==E）暴露 exact 与 HF
发散（exact=[24,101,101] vs HF=[41,2,62]），而 dense_sim 同配置正确——问题锁定在
exact 专属路径。分阶段探针定位（gather ✓ → scatter acc ✓ → reduce ✓ → 出口乘法 ✗）：

**根因**：dense 建模里每专家的加权项 `expert_out(e) * routing[:, e:e+1]` 是图节点，
E≥3 时多项求和融合成主机闭包（出口反向闭包下行的天然边界），E=2 时两项求和只是
普通设备 add——`moe_exact_topology` 的出口反向闭包一路下行穿过它进入 term mul，
把 MoE 表达式本体误划进「出口投影」，发射后改读 acc_land → **acc 被路由权重再乘
一遍**（双重加权），且 term mul 缓冲与路由列缓冲复用冲突，logits 全乱。

**修复**（`graph/moe_exact.py`）：出口反向闭包只沿**值透传节点**（view/reshape/
slice/to 等只搬值不改值的节点）下行，碰到算术节点（term mul、求和 add）即停步——
那是 MoE 表达式本体，exact 里整体被行级 scatter+归约替代，留在尾链退役。

## M2⑥ 测试矩阵（新增 6 个测试，共 12/12 绿）

| 测试 | 覆盖 |
| --- | --- |
| `test_exact_skip_set_matches_moe_rows` | 零命中专家集 == `gate_skips`；零命中专家内核 0 次执行、命中专家 4 次（内核 spy 按 gate×专家计数，不用 gate_stats——它把专家链的带 gate 命令也计进去） |
| `test_exact_routing_tables_bitwise` | 路由表位级核对：从 dispatch 闭包 int 提取 mask/routing 命令 id，result_of spy 捕获张量，按专家段（`t_rows * dtype_bytes` 步长）读回 idx/w 与重算 mask 对比、token 升序。**坑：decode 与 prefill 是两份计划**（同 id 不同偏移，T=1/T=4 包络），`max_new_tokens=1` 只跑 prefill 步——必须跑 2 步验证末步（decode 计划） |
| `test_exact_scatter_waw_chain` | scatter 发射序升序；`fold_waits` 会折叠同 DPU 依赖（per-lane FIFO 是文档化保证），WAW 链要在 `deps_of` 原始层断言（writes 重放含前一条 scatter） |
| `test_exact_kernel_cache_bounded` | 内核缓存条目数不随步数增长（先暖 T=4 与 T=1 两种形状绑定再断言——首个 run 只编译 T=4 形状，缓存 58→117 是新形状而非泄漏） |
| `test_exact_topk_equals_experts` | E=2 topk==E 全命中：token==HF、8 次 gather、无跳过、k_e==1（暴露上面 E=2 bug 的用例） |
| `test_exact_norm_topk_prob_false` | norm_topk_prob=False 两语义：token==HF |

## 测试结果

- 新测试 12/12 绿（M2③④⑤ 6 个 + M2⑥ 矩阵 6 个），含三 token 解码
  exact==dense==HF 逐位一致、行数守恒（Σk_e == topk×T）。
- 受影响回归全绿：three_path/moe_rows/kernels/exec_binding/dynamic_moe_binding（78）；
  moe_dispatch/op_semantics/comm_row_clamp/shape_binding/genesim_bridge（108+2skip）；
  vllm_executor/genesim_simulation（245）。
- 探针：静态多窗口（parallel=2）、dynamic 编译（窗口模板）+ exact 均与 HF 逐位一致。

## 4L 真实权重探针揪出的两个真 bug（M2⑥ 收尾）

tiny 测试全是 TP×PP=1（全复制），4L Qwen3-30B 的合法策略是 **TP=4×PP=2**
（num_kv_heads=4 不能被 8 整除，纯 TP=8 非法）——层 0-1 与层 2-3 宿主在不同
DPU 子集。三臂探针一跑就暴露了两个 tiny 覆盖不到的真 bug：

### 修复 1：PP 下 MoE 只在宿主子集规划与归约

**症状**：exact 臂编译即崩 `KeyError: 0`（`mem_planner.py` 的
`_moe_exact_tensors` 对每层按全部 8 台查 `shard_map[dpu_id]`，层 2-3 不宿主
在 dpu0-3）。

**修复**（三处对齐，宿主集 = xs 节点 `shard_map` 的键集）：

| 文件 | 改动 |
| --- | --- |
| `memory/mem_planner.py` | `_moe_exact_tensors` 按层跳过不宿主 DPU——不宿主的台不规划该层路由缓冲 |
| `runtime/exec_plan_gen.py` | `_emit_moe_exact_dispatch` 的 `dpus` 改从 xs placement 取（原 `buffers_by_dpu` 在 PP 下混入非宿主台）；`_emit_moe_acc_reduce` 签名加 `dpus`，归约只在宿主集内收集段 |

锁定测试 `test_exact_pp_hosting_subset_matches_hf`：4 DPU × 2 段，断言 4 层
gather 各宿主 2 台、层 0/2 宿主集不相交、token == HF。

### 修复 2：跳过事件竞态（归约读到未完成的 scatter）

**症状**：exact 臂 token 从中途开始发散且不可复现（同命令同数据，重跑结果
不同）；tiny 多窗口探针稳定复现 exact 7/8 失败（dense 8/8 正确）。

**根因**：`_skipped_event` 对零命中专家的命令即时返回预完成 Future。exact 的
acc 归约只等 `deps_of` 给出的每 DPU **末位 scatter**（`fold_waits` 折叠同 DPU
等待，依赖 per-lane FIFO）；当末位 scatter 恰是零命中（跳过）专家时事件瞬时
兑现，而同 DPU 流上更早的**命中** scatter 还在执行 → 归约读到旧累加值。
dense 免疫：where 主机闭包按值读专家输出，不吃内存等待的折叠。

**修复**（`backend/hal_numpy.py`）：设备命令（dpu_id 非 None）的跳过事件改挂
`_stream_tail[(lane, dpu)]` 队尾（`add_done_callback`），等价 FIFO 到位且不占
执行队列——跳过优化完整保留；主机命令（dpu_id None）仍即时兑现，dense 行为
逐字节不变。

### 验证

- tiny 多窗口（2 窗口 × 2 请求 × 5 步 × 8 次）：修复前 exact 7/8，修复后
  exact 8/8、dense 8/8。
- `test_moe_exact_dispatch.py` 13/13（12 + 新增 PP 测试）；受影响回归
  102 + 301 = 403 条全绿。
- 4L 三臂探针（8 DPU、4 提示、7 步，同轮独占）**通过**：三臂 token 逐位一致
  （r0=[6722,4849,35579,1197,5434,5434,5434]、r1=[12824,102642,124264,93418,
  1940,5434,103511]、r2=[108535,13447,86740,105139,135220,9794,48316]、
  r3=[1632,21309,15245,22270,22270,22270,22270]）；墙钟 exact 93.44s vs
  static 235.08s / dynamic 243.72s（39.7%，≤105% 达标，exact 反而快 2.5×——
  零命中专家整支跳过 + 每专家只算命中行的双重收益）；`moe_rows_sum=128`
  （=4 泳道 × 4 层 × Σk_e=8，行数守恒），零命中专家 16 个全部跳过。

## 契约对齐（全量回归揪出，M2 遗留）

全量回归两条契约测试失败，均指向 `graph/moe_exact.py`：`device` 键裸字符串
读取（:34/:68）与两步式 `val.dtype` 取法（:73）——定向回归批次的关键字没
覆盖这两个契约测试文件，漏网。修复对齐仓内先例（`exec_plan_gen.py` 同一批
scatter 节点的写法）：device 走 `DEVICE_META_KEY`/`DEVICE_DPU`/`DEVICE_HOST`
常量，dtype 走 `SPEC_META_KEY` 的 `spec.dtype == "bool"`（统一 IR 字符串，
spec_prop 是 dtype 进入 IR 的唯一入口）。契约测试 23 passed、exact 测试
13 passed，行为等价。

## 已落地：紧凑缓冲共享化（2026-11-18）

立项时的定位（4L 实测账本，T_max=512）：exact 比 dense 多出的 260MiB 激活
中 **98% 来自「每专家一份全宽紧凑缓冲」**——E=128 份 `moe:compact` 共
256MiB，块级生命周期（routing_scatter→exit_consumer）致 128 份互不能叠址。

**实现**（改动 +147/-53，三文件）：

| 文件 | 改动 |
| --- | --- |
| `memory/mem_planner.py` | `_moe_exact_tensors` 的 compact 生命周期从块级改为**本专家链 xs 读者的节点区间**——专家链按序执行，区间互不重叠，`greedy_reuse` 自动把全部专家叠到同一地址（本体未动） |
| `runtime/exec_plan_gen.py` | gather 从「dispatch 后整批预发」改为**随链首惰性发射**（`_emit_moe_gather`，挂在主循环首个链节点处）——整批预发 + 共享地址会让 WAR 边指向更晚提交的同流命令，单工作流下死锁；惰性化后 WAR 边与计划序同向、被 fold_waits 折叠零开销。附带编译期交错防护：前序专家读者未发射即抛（`war_waits` 对未发射读者静默跳过，不能靠它兜） |
| `tests/test_moe_exact_dispatch.py` | 新增 `test_exact_compact_buffers_shared`（同层 compact 单地址、gather 惰性分布、WAR 边含前序专家命令）；`chain_reads_compact` 断言改共享语义；`routing_tables_bitwise` 改为 scatter 执行时快照表段（表生命周期到块尾，块后缓冲合法复用其地址，终态读回会读到复用者） |

**实测**（4L 真实权重，8×8GiB，池 8）：

| 每台激活总占用 | 共享前 | 共享后 |
| --- | --- | --- |
| exact T_max=128 | 637 MiB | **126 MiB** |
| exact T_max=512 | 2552 MiB | **505 MiB** |
| exact / dense 比 | 5.4× | **1.07×**（差额 4.2MiB = idx/w/acc） |

exact 臂 token 与三臂基线**逐位一致**、行数守恒；单测 14/14、受影响回归
223 绿。执行序与逐位一致性不受影响（同一 DPU 流内命令集合与时序不变，
只改地址复用与发射位置）。

**48L 全模型账本**（plan-only 探针 `/tmp/probe_48l_exact_plan.py`：真代码
路径跑到内存规划、发射三件打桩，79 分钟 vs 全编译 5h21m；4L 冒烟与全编译
账本逐字节一致验证桩不扰动数字）：

| Qwen3-30B-A3B 48L @ T_max=512，每台 8GiB | 值 |
| --- | --- |
| 权重 | 7724.7 MiB（94.9%） |
| stride | 63.16 MiB（与 4L 相同——层间串行复用，峰值不随层数涨） |
| 自动收缩 | 池 8→**2**，并发宽度 2 |
| 激活合计 / KV | 126.3 MiB / 52.3 MiB（279 槽 = 4464 token） |
| 余量 | 224.7 MiB |

推断（未实测旧代码）：共享化前 exact stride ~319MiB > 全部可用余量，
48L exact 编译会直接失败——共享化把 48L exact 从「装不下」救成「池 2 宽 2」。

dense 对照臂（同探针 PROBE_MODE=dense）：stride 59.09MiB、池/宽同为 2/2
——**48L 上 exact 零并发代价**；代价转移到 KV：322→279 槽（−13%，约少
0.8 个满长请求），换来 MoE 计算 ~16× 缩减。exact 转默认的拍板数据至此
齐备（4L：激活比 1.07、墙钟快 2.5×、token 逐位一致；48L：同池深、
−13% KV、~16× 计算）。

## 当前不足 / 已知边界

- **段偏移、写表偏移是手工算术**：`expert * t_rows * dtype_bytes(...)` 在 dispatch v2 /
  gather / scatter 三处重复出现（曾因漏乘元素宽度出过 IndexError）。M2⑥ 路由矩阵测试
  覆盖后可考虑提一个小工具函数，现在先不预造。
- **EP（专家并行）未覆盖**：跨 DPU 路由表广播是 stretch 目标，本期只交付 TP。
- **环境观察**（非本改动引入，dense 同样受影响）：`.opcompiler_cache/` 是仓内共享缓存，
  两个进程同时编译内核时曾观察到 token 漂移（并发 pytest + 探针同跑，事后无法复现；
  独跑/双探针并行/人工 CPU 负载均不复现）。跑对拍类测试时避免与其他编译任务并发。
- GeneSim/CEVA 接线是 M4a/b：sidecar 枚举、按 Σk_e 行计费、设备路径落地或降级登记。

净增删：M2 主体与收尾已全部落库（cafdb35 + 3321efb/3afc0d3/2138cf4）；
紧凑缓冲共享化批次 +147/-53（mem_planner +5/-2、exec_plan_gen +72/-38、
test +70/-13）另文档增量，等用户下令提交。
