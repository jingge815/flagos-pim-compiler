# MoE exact_dispatch 专家并行（EP×exact）

2026-11-20。让 `exact_dispatch` 支持 `ep_width>1`：不同专家宿主在不同 DPU
子集，专家权重 ÷ep_width，专家链天然并行。此前 exact×EP 在内存规划期
`min() arg is an empty sequence` 直接崩（2026-11-19 转默认时实证）。

## 改了什么

| 文件 | 改动 |
| --- | --- |
| `graph/spec_prop.py` | `remark_moe_aggregation_for_ep` 收窄：只把**跨专家** mul/add 改标 host；`_experts_ancestor_layers`（层粒度）删除，并入新 `_expert_origins`（(层, 专家) 粒度） |
| `graph/moe_exact.py` | `MoELayerTopology` 新增 `hosts: dict[专家号, 宿主 DPU 元组]`，取自 down_proj 分片键集 |
| `memory/mem_planner.py` | `_moe_exact_tensors`：紧凑缓冲 `moe:compact:L:e` 只规划在宿主该专家的台；idx/w/acc/acc_land 仍按 xs 键集（段内每台）规划 |
| `runtime/exec_plan_gen.py` | `_emit_moe_exact_dispatch` 写表按专家宿主分段：每台只收自己宿主专家的 idx/w 段 |
| `tests/test_moe_ep.py` | 16 DPU 生成对拍参数化到 exact（含行数守恒 + 零命中跳过断言）；新增宿主缓冲结构断言与 pp2ep4 exact 对拍 |
| `tests/test_architectures.py` | 单专家两用例补上 02c87cc 默认转 exact 时漏掉的断言更新（基线就是红的） |

## 关键设计

**宿主集口径**（PP 宿主集修复 3321efb 的思想推广到专家粒度）：

```
层粒度（PP, ep=1）:  本台规划该层 ⟺ dpu ∈ xs.shard_map
专家粒度（EP）:      本台规划该专家紧凑缓冲 ⟺ dpu ∈ down_proj_e.shard_map
                     idx/w/acc/acc_land 仍按 xs 键集（归约跨全段）
```

**remark 收窄的判据**：mul/add 的每个张量实参沿上游收集 (层, 专家) 祖先，
按本层过滤（残差流会把上一层专家带进祖先集）；全部实参出自本层同一专家
（如 silu·up 乘法）留设备侧随链执行，否则改标 host。原先不区分，silu·up
也被标 host，专家链被主机打断，`_chain_of` 走不进 gate/up linear，
`xs_readers[e]` 为空——这就是崩点。

**acc 归约不发明新机制**：EP 下每台 acc 只含本台宿主专家的部分和（scatter
只在宿主台发射），`_emit_moe_acc_reduce` 的 all_reduce 集仍取 xs 键集
（= 段内全台），求和落 acc_land 每台一份，出口投影本地读。dense EP 的
host 聚合链在 exact 里整支退役（尾链抑制），这条 all_reduce 就是替代机制。

## 已实施测试

- `tests/test_moe_exact_dispatch.py` 14 绿（ep=1 逐字节不变：TP/PP、共享化、
  惰性 gather、WAW 链全未动）。
- `tests/test_moe_ep.py` 6 绿：16 DPU pp4ep4 生成对拍 HF（dynamic+exact 双
  参数，exact 含行数守恒 Σk_e==topk×T 与零命中跳过 =E-topk）；8 DPU pp2ep4
  exact 对拍；紧凑缓冲只规划在宿主台的结构断言。
- `tests/test_architectures.py` 35 绿（含单专家 exact 对拍 HF）。
- 广域 `pytest -k "moe or three_path or ceva"` 193 绿。
- verify-act-reuse 技能 check.py：384,950 输入三档全 PASS（改了
  `_moe_exact_tensors` 必跑）。

## 当前不足

- 路由表仍按 (E, T) 全宽分配，只按宿主分段**写**；分段分配需要第二套段内
  偏移口径，表本身很小（int64×E×T），不值得。
- 策略自动枚举（`llama_strategies`）不产 EP 组合，EP 仍只能手工
  `ShardStrategy(ep_width=N)`（tp=1 限定）；自动枚举是另一个话题。
- 零命中台（E < ep_width 时段内有台不宿主任何专家）仍参与 acc 归约与
  acc_land 规划，贡献为零；无害但浪费。

## 48L 账本探针（如实登记，未强行通过）

对 Qwen3-30B-A3B（48 层 × 128 专家）跑 `/tmp/probe_48l_ep4_plan.py`
（plan-only，桩掉 comm/execution/template，手工 pp2ep4，T=512）：**编译期
容量 fail-fast**——dpu0 三区 8462MiB + 内核 tile 892MiB > 预算 8128MiB。
参照 tp8（ep=1）基线：weight 7724.7MiB/DPU、stride 63.16MiB、pool 2。
原因：8 DPU 上 pp2ep4 每台装整专家 ÷4（2× tp8 的 ÷8），专家主导的权重
直接超 8GiB；任何 8-DPU EP 组合的每台专家字节都不优于同规模纯 TP
（E/(段×ep) vs E/总台数），池深问题在此规模无从谈起。EP×exact 代码本身
在 48L 图上全链路跑通（拓扑、规划、容量判据按设计拒绝）。
