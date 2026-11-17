# MoE 行级路由内核（exact_dispatch 地基）— 2026-11-17

真动态图 M2 的第 ①+② 步：给「精确行数分派」补上两个行级设备内核和契约登记。
此时 `exact_dispatch` 模式还没接入执行计划生成（那是第 ③ 步），本步只交付
内核、行数通道与语义钉子。

## 架构位置

```
dispatch（第③步，还没写）
  ├─ 算每专家行数 k_e、行索引表 idx_e、权重列 w_e
  ├─ push_xfer 写设备路由表（地址编译期定死）
  └─ 写 hal.moe_rows[(泳道, dispatch, 专家)] = k_e   ← 本步新增的通道
        ↓
gather_rows_kernel：把命中 token 的行从 hidden_states 抄进紧凑输入区
        ↓
专家子图（matmul 链，包络形状编译，行数不进缓存键）
        ↓
scatter_add_kernel：专家输出按命中行加权加回累加器（fp16，专家升序）
```

## 改了什么

| 文件 | 内容 |
| --- | --- |
| `contracts/op_semantics.py` | 新常量 `GATHER_ROWS_KERNEL = "pim.gather_rows"`、`SCATTER_ADD_KERNEL = "pim.scatter_add"`；登记表加两条 `has_kernel=False` 条目（同 `paged_attn` 口径：运行时下发、图上无节点、GML 无对应） |
| `runtime/kernels_pim.py` | 镜像 `scatter_add(acc, out, weights, indices, rows)`：`acc[indices[i]] += weights[i]*out[i]`，i < rows；权重按 `(rows,1)` 广播按行缩放；行数越界直接抛 |
| `runtime/kernels.py` | `_moe_rows_of`（行数通道，缺了抛）；`gather_rows_kernel`（复用 `gather` 编译入口 + 镜像）；`scatter_add_kernel`（镜像）；`_KERNELS` 登记两条非 aten 名字 |
| `backend/hal_numpy.py` | `moe_rows: dict[(lane_key, dispatch_id, expert) -> k_e]` 运行期行数表，类比 `gate_skips` |
| `tests/test_moe_rows_kernels.py` | 新增 10 条单测 |

净增删：+132 / -2（四个已有文件），新增测试约 240 行。

## 关键决定

**1. gather_rows 复用 `gather` 的编译入口（行数不进缓存键）。**
opcompiler 没有「行级 gather」的新 mnemonic，也不需要：查表就是
`table[indices]`。按**包络形状**（索引表 T_max 行）编译一次，缓存键 =
（上下文，"gather_rows"，表形状，索引形状，索引类型）——行数是运行期值，
进键每步都要重编。

**2. 行数之外的索引在本内核钳到 0 行。**
编译内核按包络整块处理时，尾部是行 0 的复制；而 scatter 走镜像只取前
rows 行，这些复制根本不会被读。dispatch 不必预钳位，约定只在这一处。
两条路由（编译/镜像）产出逐字节相同的包络输出。

**3. scatter_add 先走镜像，没有编译路径。**
设备侧没有「按索引累加」的编译入口（`kv_cache` 的散写不带累加）。等 M0
探路给出厂商行 gather/gather-DMA 结论后再定第 ③ 步之后是否需要设备路径。

**4. 行数通道 = `hal.moe_rows`，缺了直接抛。**
与 `gate_skips` 同一个登记模式：dispatch 运行期写入，行级内核读取；键含
泳道，多请求互不串行数。exact 模式不猜包络——回退包络等于悄悄丢正确性。

**5. fp16 语义对齐稠密 where 路径。**
`make_static_moe_forward` 的累加顺序是：routing 先转 fp16、乘专家输出、
专家号升序逐项加。镜像严格照此：fp16 乘、fp16 加、升序由第③步的 WAW
waits 链（发射序 = 专家号升序）保证。未选中专家在稠密侧贡献恰为 +0.0，
不加它数值相同（±0.0 位级差异不改变任何比较与 token）。

## 已实施的测试（10 条全绿）

- 镜像：手写循环逐位对拍；稠密 where 累加对拍；行数越界抛错。
- gather_rows 内核：前缀取行正确 + 尾部钳到行 0（含越界哨兵索引不触雷）；
  编译/镜像两条路由逐字节一致（注入伪编译内核强制两条路都走到）；缓存键
  不随行数增长；行数缺失抛错；泳道隔离。
- scatter_add 内核：两专家 waits 链串行累加 == 专家升序手写累加。
- 登记：两个非 aten 名字可被 `register_all` 挂上后端。

受影响面：`test_op_semantics` / `test_three_path_coverage` /
`test_three_path_generation` / `test_partition`（反向闭合）/
`test_kernels` / `test_opcompiler_ops` / `test_runtime_compiled_coverage`
/ `test_moe_dispatch` / `test_shape_binding` / `test_auto_capacity` /
`test_runtime_materialize` / `test_multi_request_decode` 共 240 条全过；
全量回归见本次提交说明。

## 当前不足

- `exact_dispatch` 还没有发射方：第③步 `exec_plan_gen` 接入前，这两个内核
  只有单测触达。
- scatter_add 无设备路径（镜像-only），真实设备收益依赖 M0 探路结论。
- 索引表内容的有效性（entries < rows 且 < 表行数）由 dispatch 契约保证，
  内核不查——与 `kernels_pim.gather` 「不钳越界」的既有口径一致。
