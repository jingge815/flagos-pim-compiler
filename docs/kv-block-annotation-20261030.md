# 给四节点注意力盖 KV 块循环标注

## 背景

`docs/paged-attn-bridge-20261030.md` 让融合算子 `pim.paged_attn` 拿到了 B 路
产物，它的 pimir 带 `perBlock` 相位，GeneSim 按历史块数计费。拆开的四个注意力
节点（GEMV_SCORE / MASK / SOFTMAX / GEMV_CONTEXT）没有相位链，一份 trace 写死
一份字节量，历史多长都是同一个数——同口径对照里它们被算便宜了一个量级。

这一篇说的是编译器这一侧怎么补：不改 FlagTree、不改算子本身，只在放置导出时
给这几份 B 路 IR 盖三个**函数属性**，由 GeneSim 沿用已有的块循环机制计费。
GeneSim 侧的实现见 `genesim/docs/kv-block-billing-20261030.md`。

## 标注

| 属性 | 含义 |
|---|---|
| `pim.kvBlockTokens` | 这份内核按多少个 token 历史编的（`_KV_BLOCK_TOKENS = 128`，与 `_SYMBOLIC_DIM` 同源） |
| `pim.kvFixedIn` | 循环外只读一次的字节 |
| `pim.kvFixedOut` | 循环外只写一次的字节 |

四个节点各标什么：

| 节点 | `kvFixedIn` | `kvFixedOut` | 理由 |
|---|---|---|---|
| GEMV_SCORE | 第一个操作数（q）的字节 | 0 | q 对整段历史只用一份 |
| GEMV_CONTEXT | 0 | 结果（累加）的字节 | 累加结果在整段历史跑完后写一次 |
| MASK | 0 | 0 | 整份算子逐块跑 |
| SOFTMAX | 0 | 0 | 同上 |

写在 `tt.func` 上而不是逐个 op 上：softmax 会被 `triton-opt` 展开成六个 op，
算子级标注会随展开丢掉。实测函数级属性会原样留到最终产物里。

## 改动

| 文件 | 改动 |
|---|---|
| `genesim_bridge/placement_export.py` | `_kv_block_attrs` 按 op_type 算三个属性值，`_with_func_attrs` 把 `attributes {...}` 插进 `tt.func` 签名；`_bpath_pimir` 增加 `op_type` 形参并在打哈希前进文本；`_attach_bpath_pimir` 的缓存键带上 op_type |
| `tests/test_placement_export.py` | 新增 4 条 |

缓存键必须带 op_type：GEMV_SCORE 与 GEMV_CONTEXT 的 mnemonic（`matmul`）与形状
完全相同，只有标注不同，不分键会互相串用。

`op_type` 默认空串：不传就是「没有标注」，与改动前逐字节相同。

## 产物

| 算子 | 文件 | 标注 |
|---|---|---|
| GEMV_SCORE | `bpath-52eddb800edfc5e2.pimir.mlir` | 128 / 16384 / 0 |
| GEMV_CONTEXT | `bpath-793902f6a72813d3.pimir.mlir` | 128 / 0 / 16384 |
| MASK | `bpath-e180f436487c1741.pimir.mlir` | 128 / 0 / 0 |
| SOFTMAX | `bpath-8269cd906431ef0b.pimir.mlir` | 128 / 0 / 0 |

四份共用同一份内核：1024 个注意力节点各自的 `max_seq` 不同，但块数在运行期
按请求历史绑定，编译期只有占位形状。

`llama2_7b_tp2_pp4_legacy_placement.json` 已按新产物重钉 4096 条 `pimir_path`。

## 测试

`tests/test_placement_export.py` 44 个用例全过，新增 4 条：

- `test_attention_matmul_declares_what_does_not_repeat_over_the_history`：
  两次矩阵乘形状相同、标注不同；
- `test_mask_and_softmax_repeat_whole_over_the_history`：两个固定量都是 0；
- `test_ops_without_kv_history_carry_no_block_annotation`：rope 不带标注；
- `test_block_annotation_lands_on_the_function_not_the_op`：softmax 展开后
  标注仍在函数上，且全文只出现一次。

## 不足

- **标的是编译期占位**。一块 128 个 token、形状按 `_SYMBOLIC_DIM` 取，运行期
  只按块数循环，块内的形状不再随真实 Tq 变。decode（Tq=1）时 MASK 与 SOFTMAX
  每块仍按 128×128 记，见 GeneSim 侧文档的「不足」。
- **只有这四个 op_type**。别处若出现同样「读一段 KV 历史」的节点，要在这里
  加一条，否则它的代价仍与历史无关。
