# M4a GeneSim 接线：exact_dispatch 进仿真链路（2026-11-18）

## 做了什么

两仓枚举锁步加 `exact_dispatch`，GeneSim 的 MoE 计费从「命中集×T 包络」
升级为「按运行期精确行数 Σk_e」，跨仓角色对拍扩到三模式。

| 仓 | 改动 |
| --- | --- |
| 编译器（+80/−27） | `placement_export` 报错列三模式；`export_pp_placement.py`/`run_full_pipeline.py` 的 `--moe-execution` choices 接 `MOE_EXECUTION_MODES`（原硬编码两模式，exact 进不了主链）；`test_plan_and_genesim_ir_agree_on_moe_roles` 参数化三模式（26 passed） |
| GeneSim（+315/−31） | `model_ir` 枚举 + `PIM_MOE_DISPATCH=2\|exact` 缺省规则 + 运行期字段 `moe_exec_rows`；调度器 `_apply_exact_dispatch_billing`：flops 的 Tq 项×k_e、权重常数项×命中并集（命中专家整份权重要读，与行数无关）、依赖边按行占比；10 个 exact 计费用例 |

## 关键决策

1. **k_e 来源 = 均匀路由期望**：调度器写 E[k_e]=Tq·top_k/E（Σk_e=top_k·Tq
   是 top-k 恒等式，与路由分布无关）——编译器零导出，只声明语义，与
   「运行期信息写缓冲」的路线一致。
2. **计费拆项**：行相关项按 k_e，权重读取按命中并集——decode 下 exact 与
   dynamic 收敛同值（frac=8/128），prefill T=128 实测 down_proj Σflops
   5.15e10 → 3.22e9（16×，恰 E/topk）。
3. **通信口径（如实登记）**：专家端边按行占比 k/E 折算，未为 exact 单建
   「每层一条 acc 归约」的结构模型——与 dynamic 包络口径同族；要严格口径
   只需在 `_moe_exact_row_share` 调用处对 ALL_REDUCE 特判一行。

## 测试

编译器：`test_moe_dispatch` 26 passed（三模式角色对拍）、
`test_genesim_bridge`+`test_placement_export` 129 passed。
GeneSim：model_ir 83、compiler_placement 149、moe_execution_cost 27、
model_parser 11、partition 3 全绿（主线复跑 moe_execution_cost OK）。

两仓均未提交。CEVA 侧按 M0 结论登记降级（exact 不落 CEVA，见
docs/m0-ceva-probe-20261118.md）。
