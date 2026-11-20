# exact×PP 在 no_grad 生产导出图下的编译期崩溃修复（2026-11-20）

## 问题

`export_pp_placement.py`（内部 `torch.set_grad_enabled(False)`）+ 任意 PP 策略 +
`exact_dispatch` 在编译期崩：`KeyError: 'moe:acc_land:3'`（runtime/exec_plan_gen.py
出口投影读 `moe_buffers[dpu_id]` 时 `dpu_id` 在消费者 shard_map 里、但该层
acc_land 未规划在彼）。同策略、grad 开着（单测路径）编译正常，2385 绿回归
拦不住它。

## 根因

两种梯度开关导出的图结构不同（仓内既有知识：vllm_worker / architectures 注释
「不改全局梯度开关：关闭时 torch.export 会导出另一张图」）：

| | grad 开（单测） | no_grad（生产） |
| --- | --- | --- |
| RoPE 前段 | 包成 `wrap_with_set_grad_enabled`（实参含 placeholder，与 MoE 无关） | 展开成裸节点 `unsqueeze ← inv_freq(get_attr) → … → to_1`，后接 `wrap_with_autocast` |

`moe_exact_topology` 的前向闭包把「实参只有 get_attr/常量」的节点也吸进闭包：
`unsqueeze`（只吃 inv_freq 权重）达标，随后 `unsqueeze_1 / to / expand / to_1`
级联入闭包；游走在 `wrap_with_autocast`（实参 = 闭包内 `to_1` + 外部 `to_2`）
处提前停步，把它误判为**每一层**的出口消费者。层 3（宿主段 dpu{2,3}）的出口
于是落在 stage0 的 `to_1`（shard {0,1}），acc_land 只按 xs 宿主集 {2,3} 规划，
出口读跨段 → KeyError。

`to_1` 既不是 MoE 出口投影、也不是真实跨段消费者——跨段是分叉图 + 闭包误吸
收的假象。

## 修改

| 文件 | 内容 |
| --- | --- |
| `graph/moe_exact.py` | 前向闭包只吸收「至少消费一个闭包成员」的节点：只吃权重/常量的节点不在 MoE 种子下游（4 行守卫）。修后 no_grad 图与 grad 图的每层拓扑一致（exit_consumer = 本层残差加，exits = {残差加, acc view}，落本层宿主段） |
| `tests/test_moe_exact_dispatch.py` | 新增 `test_exact_pp_no_grad_export_acc_land_on_consumers`：走生产同款 no_grad 导出链编译 tp2×pp2，断言每层出口投影 shard_map 全集都规划了 acc_land，且 no_grad 与 grad 两种导出图的贪心 token 均与 HF 参考逐位一致 |

内存规划（acc_land 落地集）、出口重定向与归约发射机制**未动**：拓扑修正后出口
天然落回层宿主段，与既有 all_reduce 落地集一致。

## 测试

- 新红测复现原 KeyError（守卫临时关闭时 `KeyError: 'moe:acc_land:3'`），修后绿。
- `pytest tests/test_moe_exact_dispatch.py tests/test_moe_ep.py -q` → 21 passed。
- `pytest tests/ -q -k "moe or three_path or placement"` → 218 passed。
- 复现命令编译段通过；sidecar 导出需图骨架 IR 的 `moe_execution` 与编译配置一致
  （既有守卫），IR 用 `model_parser --moe-execution exact_dispatch` 重新生成后
  导出成功（run_full_pipeline 本就按此生成每运行一份 IR）。

## 不足与顺带观察

- dynamic/dense×PP（tp2×pp2 tiny）贪心 token 与 HF 不一致（两种梯度开关下都
  如此），属既有行为、不在本修复范围；exact×PP 两种开关下均与 HF 一致。
- 站点共享 IR（`genesim/models/qwen3_moe.ir`）仍是 30B/dynamic 旧骨架，直接跑
  `export_pp_placement` 默认路径会命中一致性守卫，需先重生成对应骨架。
