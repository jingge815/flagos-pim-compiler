# 需求：静态计算图 → host 驱动动态图（dyn-graph）

日期：2026-11-16（评审基准文档；需求于 2026-11-13 批准实施，2026-11-15 首期收口）

## 背景

当前链路是「编译期决策、运行期照表执行」的静态蓝图：`torch.export` 用具体张量导出、无
`dynamic_shapes`；prefill 每个桶一张图、decode 固定 T=1；分区/规格传播按 `meta["val"].shape`
静态计算；三区（权重/KV/激活）offset 编译期写死；`ExecutionPlan` 是静态线性命令表，`waits`
在编译期预计算；执行器一次性全量 submit 后统一 wait。现有「动态」只有 MoE 的 `selected_mask`
+ 提交前 gate 跳过：图仍是全专家静态展开，没有运行期建图、动态形状、动态内存。

用户目标：**全动态图**，并接受本仓可落地的边界：host 驱动的动态图 + 预编译设备 kernel，
即运行期动态绑定形状、选择子图、按依赖调度和分配地址；设备内不做数据依赖分支、不做 DPU
直连、不做逐 op JIT。静态路径保持默认、逐字节兼容、可随时回退。后续多 batch 调度将在此
基础上继续（动态控制流为后续阶段）。

## 明确不做（边界）

- 设备内分支/循环、DPU 直连通信、动态 KV 物理扩容、逐 op JIT。
- 第 1 阶段异步 DMA 重叠；任意 Python 控制流 eager 执行。
- D3（`torch.cond`/`while_loop` 下降为 RTControl、调度器物化分支）与 D4（slot arena /
  任意 offset arena）本阶段只做契约预留，不实现；没有真实带控制流的模型时不做投机实现。

## 验收判据（Go/No-Go）

1. 静态模式全量回归绿（默认模式，不设 `PIM_DYN_SHAPE`）；`PIM_RUNTIME_MODE=static` 随时可回退。
2. 运行期依赖集与静态 `waits`（折叠后）逐命令相等（shadow 校验）；token 逐 token 一致。
3. 一张 prefill 图覆盖 1..max_seq，`prefill_seq_lens == ()`；kernel 形状类数量有界且不随请求数增长。
4. 单请求墙钟不高于静态路径 105%；多请求并发不劣化。
5. 新增行为必须有同次提交的测试；文档 `docs/dynamic-graph-20261113.md` 与实现一致。
