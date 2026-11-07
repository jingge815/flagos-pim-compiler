# 放置导出认识 `PAGED_ATTN`

## 背景

GeneSim 的 ModelIR 把 fp16 注意力折成了一个 `PAGED_ATTN` 节点（见
`genesim/docs/ir-paged-attn-20261030.md`），对应编译器的融合内核 `pim.paged_attn`。
但导出侧完全不认识这个 op_type：`MNEMONIC_OF` 里没有条目，`_attach_bpath_pimir`
也没有分支，于是 1024 个注意力节点导出时被跳过，拿不到 `pimir_path`，仿真只能
退回模板系数。

## 改动

### 1. `genesim_bridge/op_classify.py`

- 新增 `paged_attn_geometry(head_dim, kv_len)`：按 16 token 一块推出
  `(cache_elems, max_blocks, slot_tokens, slot_elems, scale)`，与
  `opcompiler_bridge.oplevel_kernel.paged_attn_kernel` 的形参一一对位。
  scale 取 `head_dim ** -0.5`，与运行时 `1/sqrt(head_dim)` 一致。
- `build_recipes` 里注册 `PAGED_ATTN`，用现成的 `bpath()` ——**只有 B 路**。
  A 路留空是有代价依据的：它的 A 路只能是 PyTorch 的
  `scaled_dot_product_attention`，在 FlagGems 下落到 flash 内核，那份 TTIR
  `lower_ttir_to_pimir` 直接报 `PassManager::run failed`，降不出 pim mlir。
- `UNCOVERED_OP_TYPES` 加上 `PAGED_ATTN`：A 路保留模板成本，B 路正常桥接。
- `MNEMONIC_OF` 的类型元组加上 `PAGED_ATTN`。
- `flash_attention_probe` 挪到 `build_recipes` 之前（`MNEMONIC_OF` 是 import
  时构建的字典推导，配方工厂用到的函数必须先定义）。

### 2. `genesim_bridge/placement_export.py`

- `_bpath_pimir` 的 `builders` 加 `"paged_attn"` 分支：q 取 `[Tq, head_dim]`、
  KV 取 `[Tp+Tq, head_dim]`，槽几何由 `paged_attn_geometry` 推。
- `_KV_OP_TYPES` 加上 `PAGED_ATTN`，这个节点同样占 KV 区（zone 标 `kv`）。

## 产物

| 产物 | 数字 |
|---|---|
| `llama2_7b.ir` | 3780 算子 / 11303 依赖，其中 `PAGED_ATTN` 1024 |
| `llama2_7b_tp2_pp4_placement.json` | 3778 条；`PAGED_ATTN` 1024 条全部带 `pimir_path`，死路径 0 |
| `.opcompiler_cache/bpath-*.pimir.mlir` | 一份，6 条相位（相 0-4 `perBlock = true`，相 0/4 `poolBytes = 4096`） |

1024 个注意力节点共用同一份内核产物：块数由 GeneSim 在运行期按请求历史长度绑定，
编译期只要求槽池覆盖代表形状。

## 测试

`tests/test_genesim_bridge.py` 52 个用例全过。两处断言按融合后的形状更新：

- `test_oplevel_sidecar_records_mnemonics`：允许 `pim.paged_attn`。它是方言算子，
  不在表 1.2.4 的 14 个 GML mnemonic 里，sidecar 的 `source_name` 只能记它自己。
- `test_real_model_bridge_coverage_is_recorded`：整模型覆盖从「14 个全到」变成
  「12 个 + `pim.paged_attn`」——融合节点把 `pim.mask` 和 `pim.softmax` 吞掉了。

## 不足

- **A 路没有实测成本**。pimir / ttir 两种精化下 `PAGED_ATTN` 都留在模板成本，
  数值来自 ModelIR 的线性系数（`4h+3` flops、`6eb` 字节），不是内核实测。
  要用实测值得先让 flash 内核降出 pim mlir，或者单独给这条算子走 oplevel。
- **符号维用 128 占位**。`_bpath_pimir` 把 `Tq` / `Tp+Tq` 一律取 128，所以生成的
  trace 是按 128 个查询行量的：prefill（Tq=128）对得上，decode（Tq=1）的
  查询侧字节偏大。KV 侧的槽池字节不受影响，因为它是每块固定 4 KiB。
