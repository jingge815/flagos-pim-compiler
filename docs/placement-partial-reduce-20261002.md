# partial 放置决策缺 `reduce` 导致算子编不出来（2026-10-02）

## 现象

`python -m pytest tests/ -q -k "llama2_7b"` 里，端到端那两条失败：

```
FAILED tests/test_opcompiler_e2e_llama2_7b.py::test_compiled_linear_end_to_end_matches_hf_generate[tp8_pp1]
FAILED tests/test_opcompiler_e2e_llama2_7b.py::test_compiled_linear_end_to_end_matches_hf_generate[tp2_pp4]
```

`tp1_pp8` 那一档没跑到（上一次运行被 Ctrl-C 中断，报告里是 NOT RUN）。

## 排查

先把三种策略的蓝图摊开，数每个命令带的是什么放置决策：

| 策略 | 未下发 | replicate | shard | partial | partial 出现在哪 |
| --- | ---: | ---: | ---: | ---: | --- |
| tp8_pp1 | 476 | 22816 | 3600 | **1024** | `aten.linear.default` |
| tp2_pp4 | 584 | 5704 | 900 | **256** | `aten.linear.default` |
| tp1_pp8 | 4046 | 0 | 0 | 0 | — |

失败的两档正好是**有 partial** 的两档，纯流水那档一条 partial 都没有。
partial 是行并行 `linear` 的输出（权重按收缩维切，每台 DPU 拿到一份全形状的
局部和，要跨 DPU 归约才完整），见 `graph/spec_prop.py` 的
`Placement("Partial", reduce_type="sum")`。

再看下发出去的文本：

```
error: a partial placement must say how its pieces combine; set reduce to sum or mean
module attributes {pim.placement = #pim.placement<kind = partial, numDpus = 8>}
```

FlagTree 的 verifier 见到不带 `reduce` 的 partial 直接拒收整份 pimir，于是
**每一个 partial 的 linear 都编不出来**，RuntimeError 挂在解码循环里。

## 根因

`contracts/mlir_layout.placement_attribute` 有意不发 `reduce`，理由写在注释里：
「PIMMLIR 侧 partial 改变决策只靠档位与 DPU 数，归约方式没有任何读者，下发就是
只写不读」。这个判断对这版 FlagTree 不成立：`#pim.placement` 的 verifier 要求
partial 必须写明归约方式（FlagTree 自己的用例
`test/Dialect/TritonPIM/placement_negative.mlir` 就是钉这条的）。

本仓设计文档里的写法本来就是带 `reduce` 的（见 `docs/pim-compiler-v0.0.6.md`
第 192 行、`docs/implement-unified-ir-20260930.md`），是实现时把它省掉了。

## 修复

`contracts/mlir_layout.py`：partial 档补发 `reduce`（值取自 `DpuShard.reduce`，
`DpuShard` 已经保证它是 `sum` / `mean` 之一）。

```
# 修改前
#pim.placement<kind = partial, numDpus = 4>
# 修改后
#pim.placement<kind = partial, numDpus = 4, reduce = sum>
```

`shard` / `replicate` 两档的文本一个字节都没变（它们本来就不带 `reduce`），
单 DPU 口径仍然一个字不发。

## 测试

| 测试 | 位置 | 说明 |
| --- | --- | --- |
| `test_a_partial_placement_names_its_reduce` | `tests/test_pimir_layout.py` | 断言模块属性文本 |
| `test_a_partial_op_actually_compiles` | 同上 | 判据落在编译结果上：带 partial 的真编一次，编不过就失败 |
| `test_all_three_placement_kinds_are_downlinked` | 同上 | 原来断言「不带 reduce」，按新契约改成带 |

两条新测试都验证过**在旧代码上会失败**（第二条报的就是 FlagTree 那句
`must say how its pieces combine`）。`tests/test_pimir_layout.py`、
`test_flagtree_ods_hygiene.py`、`test_unified_ir_contract.py`、`test_no_bypass.py`、
`test_placement_export.py`、`test_mem_layout.py` 合计 162 passed。

端到端那两条（`[tp8_pp1]` / `[tp2_pp4]`）在修完本条 + 下一条对齐问题后一起变绿：
`test_opcompiler_e2e_llama2_7b.py` 4 passed，全量 `-k llama2_7b` PASS 42 / FAIL 0。

## 余留问题

1. 修好这条之后，端到端用例又在**下一个点**上失败（分片对齐 1024 与 DMA 对齐 64
   不自洽，分块搜索失败）——那是另一个问题，见
   `docs/mem-plan-align-dma-20261002.md`。
2. `#pim.placement` 的 `reduce` 只在下发侧写、回读侧（`contracts/ir_payloads.py`
   的 `placement_of_module`）不认它。当前没有消费者，暂不补。
3. 单 DPU 口径（`num_dpus <= 1`）依然一个字不发 —— 这条「产物逐字节不变」的
   判据没被这次改动破坏，但它也意味着单卡下 partial 无处表达。
