# prefill 走融合注意力内核（2026-11-01）

## 改了什么

`pim.paged_attn` 融合内核的可见性按**位置**计算（第 t 行只看 0..pos+t），
图上只要传了掩码，`_paged_attention` 就判成「自定义掩码」回退主机两遍路径。
prefill 恰好满足「掩码长度 == 历史长度」，于是每次 prefill 都回退。

现在执行器显式声明「本次掩码就是 `causal_mask_of()`」，内核据此放行融合路径。

| 文件 | 改动 |
| --- | --- |
| `runtime/executor.py` | `execute_plan(..., causal_mask_standard=)`；`step_request` 两处传 True |
| `backend/hal_numpy.py` | 新增 `bound_flag(name)`，读绑定时附加的布尔开关 |
| `runtime/kernels.py` | `sdpa_kernel` 读开关；`_paged_attention` 只在未声明时才把掩码当自定义 |
| `tests/test_kernels.py` | 新增路由与数值用例 |
| `tests/test_prefill_buckets.py` | 新增融合/主机 token 一致性用例 |

## 为什么用绑定开关，而不是在编译期打标

图上的第 4 个实参只是一个 placeholder，编译期看不到它的值；用节点名判断
「它就是 `causal_mask`」并不可靠——测试与调用方可以往里塞任意掩码，那会让
自定义掩码静默走融合路径（正是老注释里警告的失真）。

所以由**调用方**声明：`step_request()` 每次都用 `causal_mask_of()` 生成掩码，
它在 `execute_plan` 上置位 `causal_mask_standard=True`；其他调用方（测试里
手工给掩码）默认不置位，行为与改动前一致。

```text
step_request ──execute_plan(causal_mask_standard=True)──► values["causal_mask_standard"]=True
                                                     │
                                        sdpa_kernel ─┴─► hal.bound_flag(...)
                                                     │
                                     _paged_attention ─┴─► standard_mask
                                        uses_graph_mask = mask 存在 且 长度匹配 且 未声明标准
```

## 数值口径

融合内核是单遍在线 softmax（f16 分数、f32 累加），主机路径是相位链两遍，
两者数学等价、取整位置不同：实测最大绝对误差 < 3e-4，判据取
`allclose(rtol=1e-3, atol=1e-3)`，生成 token 逐 token 相同。

## 已实施的测试

| 测试 | 验证点 |
| --- | --- |
| `test_sdpa_standard_causal_mask_routes_to_fused_kernel` | 声明的标准掩码 → `paged_attn hit`；未声明 → `fallback`；两条路径数值一致 |
| `test_prefill_uses_fused_kernel_and_matches_host_path` | 整个 prefill + 解码过程无 fallback，生成 token 与强制主机路径完全相同 |
| 既有 `test_sdpa_uses_the_mask_operand_not_a_recomputed_one` | 自定义掩码仍被真正使用（f32 回退路径） |

实测：`tests/test_kernels.py` 21 passed（2.28s）；`tests/test_prefill_buckets.py`
21 passed（53.04s）。

## 实测墙钟

同一份微基准（4 DPU、4 层、hidden 512、100 token 的 prompt、桶 128、取 3 次最小值）：

| 注意力路径 | prefill 墙钟 | 路由计数 |
| --- | --- | --- |
| 融合内核（声明标准因果掩码） | 2846.6 ms | `paged_attn hit=384`，无 fallback |
| 强制主机两遍（屏蔽融合） | 6585.0 ms | `paged_attn fallback=384`，另有 `matmul/mask/softmax` 各 5376/49152/49152 |

融合路径比主机路径快 56.8%；相对 B1 分桶后的 3102.7 ms 再降 8.3%。
单步采样延迟（`test_kernels.py`）最大绝对误差 2.4e-4，生成 token 逐 token 相同。

## 当前不足

- 混合精度/自定义掩码仍回退主机路径；滑动窗口需要一个能声明窗口宽度的内核变体。
- `causal_mask_standard` 是调用方声明，没有二次校验（例如抽样比对掩码值）；
  声明错了会静默算错——所以只允许 `step_request` 这条既定路径置位。
- f32 KV（只有测试会造）仍走主机路径。
