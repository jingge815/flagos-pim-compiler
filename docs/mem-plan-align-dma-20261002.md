# 内存规划的对齐必须等于硬件 DMA 对齐（2026-10-02）

## 现象

修好 partial 缺 `reduce` 之后（见 `docs/placement-partial-reduce-20261002.md`），
端到端用例的 tp8_pp1 / tp2_pp4 又挂在下一个点上：

```
error: no legal power-of-two tile fits: M=1 N=2048 K=4096 dtype_bytes=2
       wram_bytes=65536 mram_bytes=4294967296 dma_align=1024;
       smallest tried was M=1 N=1 K=1
```

## 排查

- FlagTree 的 `-pim-tile-to-budget` 把「分片的起始对齐」当成**每块 WRAM 缓冲**
  的对齐用：`fitsBudget` 要求 x / w / out 三块缓冲的字节数都是 `dma` 的整数倍，
  而 `dma` 取 `max(模块属性 pim.dma-align, #pim.placement 的 alignBytes)`。
- 这个形状可见的分片是 M=1、K=32 —— 一行只有 64 字节，任何分块的缓冲都凑不出
  1024 的整数倍，搜索必然失败（报错里的 "smallest tried was M=1 N=1 K=1"）。
- 1024 的来源：`HwBudget.align`（内存规划的对齐）被回填进分片的 `align_bytes`，
  随命令下发成 `#pim.placement<alignBytes = 1024>`。而同一份配置里
  `PIMHardwareConfig.dma_align` 是 64 —— 两个「对齐」不是同一个数。

实测（同一形状、同一个 partial 分片，只改 `align_bytes`）：

| align_bytes | 结果 | 选出的分块 |
| ---: | --- | --- |
| 0 | 编得过 | m=1 n=512 k=32 |
| 64 | 编得过 | m=1 n=512 k=32 |
| 1024 | **失败** | — |

## 根因

`HwBudget.align` 的字段语义就是「DMA 对齐边界」，设计里也确实假设它与
`pim.dma-align` 是同一个值（`docs/implement-unified-ir-20260930.md`：
「`align_bytes` 由内存规划回填，实测与模块属性 `pim.dma-align` 是同一个值」）。
但 llama2 组用例与 `scripts/export_pp_placement.py` 里写的是 `align=1024`
配 `dma_align=64`，这个假设被破坏了：规划把偏移按 1024 对齐，算子编译器却
拿它当缓冲对齐，于是小分片编不出内核。

小测试（`test_mem_planner.py` / `test_exec_plan_gen.py`）本来就是 `align=64`
配 `dma_align=64`，只有 llama2 组是特例。

## 修复

恢复这条不变式：**内存规划的对齐 = 硬件 DMA 对齐**。

| 文件 | 改动 |
| --- | --- |
| `memory/mem_planner.py` | `HwBudget.align` 的注释写明这条不变式与违反后果（字段语义本来就写着「DMA 对齐边界」） |
| `tests/test_opcompiler_e2e_llama2_7b.py` | `align=1024` → `64`，并说明为什么不能比 `dma_align` 大 |
| `tests/test_mem_planner_llama2_7b.py` / `test_strategy_llama2_7b.py` / `test_executor_llama2_7b.py` / `test_natural_prompt_llama2_7b.py` / `test_concurrency_llama2_7b.py` | 同上，各一处（strategy 两处） |
| `scripts/export_pp_placement.py` | 同上 |

三区偏移会随对齐变小而变紧，但这些用例的断言都是不变式（无重叠、≤ 容量、
offset 与 spec 自洽），没有钉死 1024 的数字。

## 测试

| 测试 | 结果 |
| --- | --- |
| `tests/test_opcompiler_e2e_llama2_7b.py`（三档策略全跑） | **4 passed**（28 分 17 秒）；改动前 tp8_pp1 / tp2_pp4 两档 FAIL |
| `pytest tests/ -q -k "llama2_7b"`（全量，含配置改过的 5 个用例） | **PASS 42 / FAIL 0**，exit 0（40 分 47 秒）；改动前 PASS 28 / FAIL 2 / 被中断 |

## 余留问题

1. `docs/mem_planner-20260823.md` 的例子还是 `align=1024`，历史文档没有回改。
2. 对齐层仍在往下发（`alignBytes`），只是不再比 `pim.dma-align` 更严。将来如果
   真有「比 DMA 更严的起始对齐」要求，会再撞上同一个冲突：FlagTree 把**起始**
   对齐当**缓冲**对齐用，而小分片不可能满足更粗的缓冲粒度。
3. `tests/test_mem_layout.py` 用 `align=256` 只做规划侧的记账断言（不编内核），
   保持原样。
