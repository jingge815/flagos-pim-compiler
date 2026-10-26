# 命令计划补上写后写依赖（2026-10-22）

## 现象

`test_real_llama2_7b_kv_region_matches_hf_cache_under_every_strategy[tp8_pp1]` 偶发失败：
某台 DPU 某个头的 K 区有一两行和 HF cache 对不上（例如第 0 列被写成 0），token 却常常还对得上。
小模型 `tests/test_strategy_sweep.py` 的 tp4_pp1 能稳定复现，几次运行里坏的 DPU 不固定，属于并发竞争。

## 根因

`runtime/exec_plan_gen.py` 生成命令时只记两类依赖：

| 依赖 | 记录方式 | 修改前 |
| --- | --- | --- |
| 写后读 | `_deps_of(reads, writers)` | 有 |
| 读后写 | `_war_waits(...)` | 有 |
| 写后写 | 无 | **缺失** |

decode 计划里多条命令复用同一块激活缓冲（tp4 小模型是偏移 109056）：

```
cmd 33  launch dpu3 to_2     写 (dpu3, 109056, 4)
cmd 39  host_slice 广播 cos   写 (dpu0..3, 109056, 64)   不等 cmd 33
cmd 43  launch dpu3 to_5     读 (dpu3, 109056, 64)
```

cmd 39 不等 cmd 33。如果 dpu3 的流比 host 慢，`to_2` 晚一步写进来，就把广播好的 RoPE cos 头 4 字节盖掉。
后面 K 的 RoPE 在那台 DPU 上就会算错，所以写进 KV 区的是错误的 K 行。

## 修改

`_PlanBuilder.append` 把「与新写区间相交的历史写者」并入 `waits`。改了一处，所有类型的命令（launch、host、redistribute）都会走到。

| 文件 | 改动 |
| --- | --- |
| `runtime/exec_plan_gen.py` | `_PlanBuilder.append` 加写后写依赖 |
| `tests/test_exec_plan_gen.py` | 新增 `test_later_writer_waits_for_earlier_writer_of_same_range` |

之前调试时改的 `runtime/kernels.py`（只让 owner 写 KV、写 KV 时去掉 `mram_offset`）以及对应的单测都没有解决这个问题，已经撤回。
`tests/test_strategy_sweep.py` 里新增的 KV 区对拍测试保留，用来在小模型上兜住这类问题。

## 测试

- 新单测：不改代码时失败，改后通过。
- `tests/test_strategy_sweep.py` 连跑 3 次，12 项全部通过（修改前 tp4_pp1 的 KV 对拍大多数次失败）。
- 除 `llama2_7b` 以外的全量测试：1350 通过，6 跳过。

## 不足

- 写后写依赖会让部分命令多一些等待，稍微降低并发度，数值不变。
- 两个多小时的 `llama2_7b` 全量测试还没在修复后重跑。
