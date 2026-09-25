# 词嵌入与类型转换真的发 GML 节点（20260924）

## 一、改了什么

`pim.gather`（词嵌入查表）与 `pim.convert`（纯类型转换）以前在 `_WALK_THROUGH`
里被**无条件跨过**：全模型导出里查表这一步整块消失，真实位宽转换也不成节点。
本轮把「跨过」改成有条件的发射。

| 算子 | 发节点的条件 | 不发的场景 |
| --- | --- | --- |
| `aten.embedding.default` → `Gather` | 图里真的带词嵌入（图入口是 `input_ids`） | decode 块口径导出（入口是隐藏态、嵌入在块外） |
| `aten.to.dtype` / `to.dtype_layout` → `Convert` | 两侧元素类型真的不同 | 同 dtype 的恒等 `to`（`export` 自己塞进来的那种） |

```
convert(decode_block_only=False)         convert(decode_block_only=True)
  emittable 全量（含 embedding）            emittable 先裁掉 embedding
        │                                        │
        ├─ Gather 节点 ──► 参考无此节点           └─ 参考产物逐字节不变
        └─ Convert 节点（位宽真变才发）
```

## 二、关键实现点（`gml_bridge/from_fx.py`）

1. `_cast_dtypes(node)`：读 `to.dtype` 两侧的元素类型（源, 目标），任何一侧读不到
   就返回 None——此时不发射，不猜。
2. **裁剪必须在 `convert()` 里、节点编号之前**：节点 id 按 `len(ordered)` 逆拓扑算，
   少一个节点全图编号都会平移，所以不能留到收尾再删。
3. `Gather` 的字段：`table`（编译期常量词表，走权重通路，与通用分支写的
   `weight_buffer` 是同一个文件）、`indices`（图入口那一路 token id，即
   `input_buffer`）。**参考产物无此节点，字段名待求证**，这里按方言 `pim.gather`
   的两个操作数名发。
4. `Convert` 的两侧位宽由图侧元素类型定，**不能按 `_stamp_dtypes` 的沿边传播写**
   ——它自己就是改位宽的那一步，传播会把两侧写成同一个值。所以 `_stamp_dtypes`
   对它只把目标位宽记进传播表供下游用，其余推导跳过。扩展位只覆盖 fp16 / int8
   （实测编码表 3 / 1），其余（如 f32）没有编码可写，那一份就不写。
5. 只声明产物认识的缓冲元素类型（`_BUFFER_DTYPES` = int8 / int16 / float16 /
   float32）：`int64` 这类是索引 / 主机侧类型，声明了校验器没有宽度可核文件大小
   （`verify_gml_artifact` 的宽度表只认这四种）。实测 `torch.set_grad_enabled(False)`
   下导出的图里有一个 `int64 -> float32` 的真转换，这条规则正是为它加的。
6. `_BUFFER_DTYPES` 之外的图侧类型（如 `int64`）两侧都不声明，缓冲尺寸交给写盘
   的缺省口径，产物自洽。

## 三、修改文件

| 文件 | 改动 |
| --- | --- |
| `gml_bridge/from_fx.py` | `OP_TYPES` 加 `embedding` → `Gather`；`_WALK_THROUGH` 只剩两个 `to.dtype` 重载；新增 `_cast_dtypes` / `_BUFFER_DTYPES`；`convert()` 加 decode 块裁剪 + `Gather` / `Convert` 两个发射分支；`_stamp_dtypes` 加 `Convert` 分支；`EltwiseAdd` 逐槽 `input_<槽>_sf_dtype` 按参考补齐 |
| `contracts/gml_coverage.py` | 字段家族登记补 `table` / `indices` |
| `tests/test_gml_from_fx.py` | 新增 `test_embedding_lookup_is_emitted_as_gather`、`test_decode_block_export_omits_gather`、`_cast_graph` + `test_only_real_dtype_changes_become_convert_nodes`；替换原来的 embedding 记账用例 |

## 四、已实施的测试

- 含词嵌入的图恰好发 1 个 `Gather`，`table` 指到 `weight_buffer`、`indices` 指到
  `input_buffer`，权重参数以 `embed_tokens.weight` 结尾。
- 同一条图按 decode 块口径导出时**一个 `Gather` 都不发**。
- 手搭 `f32 -> to(f16) -> to(f16)`：只发 1 个 `Convert`（恒等那一步跨过），
  `input_buffer_dtype "float32"` / `output_buffer_dtype "float16"`，f32 那侧不写扩展位。
- `tests/test_gml_coverage.py` / `tests/test_gml_node_parity.py`：逐算子对拍参考
  字段族（`pim.gather` 的字段家族含 `table` / `indices`）。
- 回归：`python -m pytest tests/ -q -k "not llama2_7b"` → **852 passed, 1 skipped,
  42 deselected**。

## 五、实测结果

- `--layers 1 --seq-len 16`：204 节点 / 335 边 / 3235 个缓冲，`Gather` **1** 个、
  `Convert` **0** 个（该导出 6 个 `to.dtype` 全是恒等 f32→f32），3 项 CLI 检查全过。
- `--decode-block-only`：与 `/media/disk/fengjingge/tmp/gml_dbo/relay2gml_graph.gml`
  **逐字节相同**（498509 字节），参考产物零改动。

## 六、当前不足

1. `Gather` / `Convert` 的字段名没有参考产物可对（参考里这两个节点都不存在），
   是按方言名写的，拿到实物后要核一遍。
2. `Convert` 的 f32 侧只写 dtype、不写扩展位——编码表只有 fp16 / int8 两档，
   宁缺勿猜。
3. 本轮之前那 3 条 `test_strategy_sweep` 失败是 `runtime/kernels.py` 的「编译内核
   优先」路径造成的：它对**恒等**类型转换也去编 `pim.convert` 内核，被方言校验器以
   「两侧同为 f16 不是转换」拒绝（`opcompiler_bridge/driver.py:305`）。该链路上没有
   `gml_bridge.from_fx`，与本轮改动无关。跑回归时那条路径上已有别人加的
   `source == target` 守卫与一处 `TEMP-DIAG` 短路（`runtime/kernels.py:807`），
   所以本次全绿是在那两个短路生效的状态下取得的。
