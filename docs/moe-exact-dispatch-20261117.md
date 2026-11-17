# MoE exact_dispatch 第三模式落地（M2③④⑤，2026-11-17）

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

## 当前不足 / 已知边界

- **段偏移、写表偏移是手工算术**：`expert * t_rows * dtype_bytes(...)` 在 dispatch v2 /
  gather / scatter 三处重复出现（曾因漏乘元素宽度出过 IndexError）。M2⑥ 路由矩阵测试
  覆盖后可考虑提一个小工具函数，现在先不预造。
- **EP（专家并行）未覆盖**：跨 DPU 路由表广播是 stretch 目标，本期只交付 TP。
- **环境观察**（非本改动引入，dense 同样受影响）：`.opcompiler_cache/` 是仓内共享缓存，
  两个进程同时编译内核时曾观察到 token 漂移（并发 pytest + 探针同跑，事后无法复现；
  独跑/双探针并行/人工 CPU 负载均不复现）。跑对拍类测试时避免与其他编译任务并发。
- GeneSim/CEVA 接线是 M4a/b：sidecar 枚举、按 Σk_e 行计费、设备路径落地或降级登记。

净增删：+397 / -15（三个已跟踪文件），新文件 graph/moe_exact.py 221 行、
tests/test_moe_exact_dispatch.py 591 行、本文档 99 行。未提交，等用户下令。
