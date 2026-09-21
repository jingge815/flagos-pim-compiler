# prepare_out 生成代码评审 · 第三轮（2026-09-21）

评审对象：当前工作区全部未提交改动（19 个 modified + 5 个未跟踪新文件）与
6 份 `prepare_out-*` 文档。方法：通读代码与文档；按文档给的组合命令实跑一次
GML+prepare_out（`--use-opcompiler --orchestrate --decode-block-only`）；
跑官方对拍与全量快速回归；另写 4 个独立脚本做官方对拍器**没做**的核对：

1. GML 逐节点字段覆盖（我方 vs 参考 `relay2gml_graph.gml`）；
2. `parser_output` 文件族计数（数字归一后）；
3. 槽位/相位敏感命名比对（官方 `_normalise_naming_value` 把数字全改成 `#`，
   槽位/相位差被吃掉）；
4. L2 双槽声明尺寸 vs 分配器分配尺寸。

所有结论都带可复现命令与实测数字（附录 A）。

---

## 0. 结论摘要

上一轮修复（`docs/prepare_out-P0修复2-20260921.md`）的核心技术方向**成立**，
但「只有 82 个既有缺失引用」这一叙事把更大的问题盖住了。按严重度：

| 级别 | 问题 | 实测依据 |
| --- | --- | --- |
| **P0** | bmm（64 个 MatMul）的 `weight_buffer`/`weight_sf`/`weight_zp` **整族没进 GML**，runtime 不落盘；txt 的 64 个 `weight_sf_*` 悬空，parser_output 比参考少 60/62/62 个文件 | §2.1；参考 GML 73 个节点带 `weight_buffer`（7 Gemm int4 + 64 MatMul int8 + 2 RMSNorm），我方 11 个 |
| **P0** | GML 仍是完整图（带模型末尾 final RMSNorm + lm_head + DQ + 2×Reshape），`--decode-block-only` 只裁编排器；parser_output 多出一族文件、GML 契约与参考节点集合不同 | §2.2；参考 2/7/36/2 个 RMSNorm/Gemm/DS/Reshape，我方 3/8/37/4 |
| **P0** | `DDR Input/Output Orig Buffer Name` 是**占位值**：输入恒为 `buffer0`/`buffer1`（槽号），输出恒为 `buffer<自己>`；参考是生产者/消费者编号或语义名，32 个 bmm2 共享 `buffer4` | §2.3；官方对拍把数字全归一，只报 43+37+6 处，逐值比对实际 183+41+220 = 444 处全不同 |
| **P0** | 两个 cache/定标引用悬空：`Original cache file`（v_proj、rope_add_k）指向 GML 不存在的 `input_buffer_0_181/186.bin`；RoPE mul 的 Kantor/Scaling 段号错（Q mul_cos 应为 5、mul_sin 应为 4，我方都写 6），8 个引用悬空 | §2.4 |
| **P1** | 值来源仍是「参考反推静态表 + 硬编码」，验收点「从图编译→算子编译（pimmlir）映射」未满足；PhaseSource 只进相位数与 L2 字节 | §3.1 |
| **P1** | 对拍器的 `_normalise_naming_value` 仍把所有数字抹成 `#`，104 不是非纯编号差异的下界（本轮独立比对得到 117 处命名结构差，官方 104 之外的 12 处 RoPE 定标段号差被记为 ALLOC） | §3.2 |
| **P1** | 残差旁路的域是**伪造**的：`input_count=1` 却写双槽 `Datain file 1`/`Residual input buffer 1`（填成槽 0 同值），mask 的 `Residual input buffer 1` 填字面量 `0` | §3.3 |
| **P1** | `l2_alloc._dual_slot1_size` 对 rope_add 也返回 `HD*2=256`，而 txt 声明 `L2 input buffer size 1 = 8192`；分配器按错尺寸记账 | §3.4 |
| **P1** | GML 的 dtype/extension 注解大范围缺失（`input_buffer_dtype` 153→77、`output_buffer_dtype` 197→39、`input_data_extensions` 193→79、`weight_buffer_dtype` 73→3、`flp_*` 1→0） | §3.5 |
| **P2** | Softmax/DQ 相位 bin 尺寸按导出图（`--seq-len 16` → 512B）而不是 txt 的 S=1024（参考 2048B）；内容全是占位 | §4.1 |
| **P2** | `net.ini` 末行仍多一个 LF；`_order_like_reference` 仍吞异常；`_drop_tail` 按 FX 字符串；`HwRow.pending_q` 文档与实现不符；`layer_fields.py` 1493 行；`layer_kinds`/`_weight_role` 死代码；根目录 509KB 会话记录 | §4.2 |

**该保留的结论**：Mask 双输入、Softmax 5 入 + 5 出、L2 双槽不重叠、
bmm2→Concat 命名收敛、对拍白名单三分类、陈旧 bin 清理这几项修复**核实为真**
（§1）。组合命令能一条命令稳定产出两侧且退出码如实为 1；`pytest` 686 全过。

---

## 1. 已核实为真的修复（不要回退）

1. **Mask 第二路输入（causal mask）确实补进了 GML**：我方 32 个 mask 的
   `input_buffer_1` 全部指向同一个文件，且该文件真实落盘；
   `_data_slot_count`/`_slotted_dataout` 的联动也对上了。
2. **Softmax 5 入 + 5 出对称模型是真的**：`parser_output` 里
   `input_buffer_phase_0..4`/`output_buffer_phase_0..4` 每族都比上一轮多 32 个
   （我方相位六族各 312 个 = 参考 308 + 多出的那条尾部 DQ 的 4 个）。
   参考 GML node 18 与 `write_softmax_phases` 的逐相写入核对一致。
3. **L2 双输入层 offset 0/1 不再重叠**：新闸门实测 `41 层双输入，0 层重叠`。
4. **对拍白名单拆三类 + `counts_mine` 参与判定**：`VALUE_DIFF 0` 的假象纠正为
   104，退出码如实非 0。
5. **写盘前清空 `out_dir/*.bin`、反向多余 bin 信息项**：陈旧文件不会再造成
   「引用闭合」假通过。
6. **回归**：`python -m pytest tests/ -q -k "not llama2_7b"` → **686 passed,
   42 deselected**（本次实跑，123s）。

---

## 2. P0：会导致仿真器读不到数据 / 产物与参考契约不符

### 2.1 bmm 的 `weight_buffer`/`weight_sf`/`weight_zp` 整族缺失（最大一处）

**证据（独立脚本逐节点统计）**：

| GML 字段 | 参考 | 我方 | 说明 |
| --- | --- | --- | --- |
| `weight_buffer` | 73 | 11 | 参考 = 7 Gemm(int4) + **64 MatMul(int8)** + 2 RMSNorm；我方只有 Gemm/RMSNorm |
| `weight_sf` | 73 | 11 | 同上 |
| `weight_zp` | 73 | 11 | 同上 |
| `weight_buffer_dtype` | 73 | 3 | |
| `DEBUG_weight_buffer_float` | 73 | 0 | 溯源字段，可选 |

参考 node 20（bmm1 head0）逐字声明：

```
input_sf "output_buffer_phase_1_22.bin"
weight_sf "weight_sf_20.bin"        # 2 字节 = 1 个 fp16
weight_zp "weight_zp_20.bin"        # 4 字节 0
weight_buffer_dtype "int8"
weight_buffer "weight_buffer_20.bin"  # 131072 字节 = S×hd = 1024×128
```

我方的 MatMul 节点只有 `input_sf/input_zp/output_sf/output_zp`，**没有 weight 三族**。
对应的 `parser_output`：`weight_buffer` 73→13、`weight_sf` 73→11、
`weight_zp` 73→11。txt 侧 `weights scaling buffer file` 走
`layer_fields.py:887-888` 的回退 `names.weight_scale(nid)`，落盘从不存在 ——
这就是新闸门那 82 个缺失里 **64 个**的来源。

**根因**：`gml_bridge/from_fx.py:963-967` 只在
`_weight_param_of(node)`（`get_attr` 二维权重）命中时写 weight 三族。
bmm 的“权重”是 KV cache（第二个 operand 走 weight 通路），不会命中；
`gml_bridge/export.py:437-452` 的写盘分支自然也从不执行。

**影响**：除 64 个 txt 悬空引用外，参考 GML 契约里每个 head 的 int8 权重缓冲
（仿真器/DMA 可能会按 `Split weight index` 去寻址）完全没有；`parser_output`
比参考少 272 个文件（按族归一后统计，其中 weight 三族就占 60+62+62）。

**建议**（按此顺序）：

1. `from_fx.convert`：当 `MatMul_input_as_weight=1` 且第二个 operand 是
   KV_Cache_DMA 输出时，补 `weight_buffer/weight_sf/weight_zp/weight_buffer_dtype`；
   `weight_buffer` 尺寸按 `S×hd×1`（1024×128），`weight_sf` 单个 fp16，`weight_zp` 4B 0。
2. `write_runtime_files`：给这三个字段加写盘分支（结构轮内容可用占位 0，
   但**尺寸与 dtype 必须按声明**，同 DQ/Softmax 的既有约定）。
3. 加一个单测：对 bmm1/bmm2 节点断言三个字段存在且落盘尺寸正确。
4. `weights scaling buffer file` 不再需要回退（`_gml_str(node,"weight_sf",...)`
   会拿到真名）。

### 2.2 GML 未裁掉模型尾部，`--decode-block-only` 只作用于编排器

**证据（GML op_type 计数）**：

| 节点类 | 参考 | 我方 | 差异 |
| --- | --- | --- | --- |
| `RMSNorm_vpu` | 2 | 3 | 多了 final norm |
| `Gemm` | 7 | 8 | 多了 lm_head |
| `DynamicScaling` | 36 | 37 | 多了 lm_head 前那条 DQ |
| `Reshape` | 2 | 4 | |
| 边界 buffer 节点 | 10 | 5 | |

`--decode-block-only` 只在 `orchestrator/plan.py:154-169` 的 `_drop_tail` 里生效，
GML 文本与 `parser_output` 在 `gml_bridge/export.py` 阶段就已经按完整图写完。
结果：`parser_output` 多出一族文件（`output_sf/output_zp` 各 +37、
`input_zp` +47、`input_buffer_#_sf/#_zp` +68 等），而缺少的 weight 族把总数
拉回来，最终 3257 vs 参考 3231（+26）——**总数看着接近，文件集合差得很远**
（按族归一：缺 272、多 298）。

另外 `_drop_tail` 靠 FX 名字符串（`"mul_12"`、`"linear_7"`）丢层，
`_order_like_reference` 里还有 `except Exception` 静默降级（`plan.py:184-187`），
换模型/换图即失效。

**建议**：把“本 decode block 之外”的裁剪放到图级别（`export_annotated_graph`
或 `serialize_gml` 之前），按结构信息（是否属于 `model.model.layers`、
消费者是否在块外）判定，而不是字符串；同时给闸门加一项
「`parser_output` 文件族计数 == 参考文件族计数」——现在的新闸门只查
「txt 引用 → 盘上有」，不查「盘上文件集合」，这个缺口正是本问题的藏身处。

### 2.3 `DDR Input/Output Orig Buffer Name` 是占位值，不是推导值

**证据**：

- `DDR Input Orig Buffer Name 0`：我方**全部 183 层**写成字面量 `buffer0`
  （`layer_fields.py:1475-1476` 的 `f"buffer{slot}"`）；参考是生产者的
  Relay buffer id 或语义名：
  - `dq_p1` → `buffer185`/`buffer16`/`self_attn_Reshape_qidx4_params_22_mid_buf`
  - `bmm1` 全部头 → `buffer13`；`bmm2` 全部头 → 各自生产者的 buffer id
  - `mlp_mul` → `buffer186`（gate 输出）/`buffer189`（up 输出）
- `DDR Input Orig Buffer Name 1`：我方是 `buffer1`，只有 mask 写 `mask`；
  参考是 `nprm_182_i168`（TVM 名，待 Q-A）或生产者的 `..._mul_cos_buffer` /
  `..._sin_value_buffer` 等语义名。
- `DDR Output Orig Buffer Name`：我方是 `buffer<自己的 node_id>`
  （`layer_fields.py:1270-1277`），参考按**消费者/共享缓冲**编号：
  32 个 bmm2 全部写 `buffer4`（Concat 下游）、sm_p5 全部写 `buffer9`；
  `rope_add_k` 参考是 `key_cache_out`，我方是 `buffer182`（只 special-case 了
  `gemm_v → value_cache_out`）。

**为什么之前没暴露**：官方对拍的 `_normalise_naming_value`
（`scripts/diff_prepare_out.py:116-126`）把所有数字换成 `#`，
`buffer0` 与 `buffer13` 归一后相同，整族被记为 ALLOC。104 里只有 43+37+6 处
（参考是语义名、归一后仍不等的那部分）被看到；逐值比对这三族共
183 + 41 + 220 = 444 处全部不同。

**影响**：这是验收点「域的值能对上」「不是硬算/绕开」的直接违反，
也是仿真器若把这些域当索引就会出错的地方（域确认表 Q48 明确问过命名约束）。

**建议**：

- 输入 Orig 名 = 该槽生产者：`input{slot}_node_id` / `residual_input_buffer[slot]`
  对应节点的 `buffer<node_id>`，相位生产者用其语义 label（同
  `_upstream_dq` 已能取到的东西）；
- 输出 Orig 名 = 消费者/共享缓冲：单消费者用 `buffer<消费者 node_id>`，
  bmm2→Concat 用共享名，cache 写出用 `value_cache_out`/`key_cache_out`；
- 对拍工具先修（§3.2），否则改完也看不见。

### 2.4 两个 cache 引用悬空 + RoPE 定标段号错

**a) `Original cache file` 悬空**（新发现，不在任何文档的清单里）：

```
v_proj（node 189）      : Original cache file: input_buffer_0_181.bin   ← 盘上不存在
rope_add_k（node 182）  : Original cache file: input_buffer_0_186.bin   ← 盘上不存在
```

我方 GML 的 KV_Cache_DMA 节点只有单数的 `input_buffer "input_buffer_181.bin"` /
`"input_buffer_186.bin"`，**没有参考那样的 `input_buffer_0`**：

```
参考 node 28（K ScatterND）:
  input_buffer_0 "input_buffer_0_28.bin"    ← txt 的 Original cache file 取它
  input_buffer_1 "input_buffer_1_28.bin" dtype int16 / use_input_buffer_1 "L2A_ignore"
  input_buffer_2 "input_buffer_2_28.bin"
  output_buffer  "input_buffer_199.bin"
```

`layer_fields.py:196-211` 的 `_cache_dma_buffer` 用 `names.data_buffer(pick.node_id, 0)`
**自己拼** `input_buffer_0_<DMA节点>`，而不是读 GML 里已声明的槽 0 字段。

**b) RoPE 定标段号错**（复核 2 已提，仍未修）：参考 K 路 mul_cos 用
`Scaling_buffer_file_6_..._Cos_30.bin`、**Q 路 mul_cos 用 `_5_..._Cos_22.bin`**、
两条 mul_sin 都用 `_4_..._Sin_*`；我方 `layer_fields.py:935` 对 mul 一律 `idx=(6,6)`：

```
参考 Q mul_cos : Scaling_buffer_file_5_Llama2Activation_Cos_22.bin
我方 Q mul_cos : Scaling_buffer_file_6_Llama2Activation_Cos_184.bin   （文件存在但段号错）
参考 mul_sin   : Scaling_buffer_file_4_Llama2Activation_Sin_30.bin
我方 mul_sin   : Scaling_buffer_file_6_Llama2Activation_Sin_182.bin   （文件不存在 → 悬空）
```

官方对拍同样因为数字归一没看见 Q cos 的错（Sin 的缺失被 82 清单抓到了）。

**建议**：`_cache_dma_buffer` 改为读 GML 节点的 `input_buffer_0`（并让
from_fx 按参考补齐 `input_buffer_0/1/2`、dtype、`use_input_buffer_1`）；
mul 的 `idx` 拆成 `{K-cos:6, Q-cos:5, sin:4}`，由 IR/相位属性或 Q/K 路径决定，
不要再写 `(6,6)`；给这两个引用各加一个夹具测试。

---

## 3. P1：正确性/可维护性，不立刻阻断但影响验收

### 3.1 值不是从 pimmlir 映射的（验收点 3）

现状与证据：

| 来源 | 位置 | 例子 |
| --- | --- | --- |
| 参考 422 层反推的静态表 | `orchestrator/layer_hw_table.py:94-174` | `l2_fpsu_size` 28672/57344/77824、`flp=(10,17,3)`、`transpose_type`、`l2_weights_off0/1` |
| 硬编码 | `orchestrator/layer_fields.py:22` | `H,I,HD,S=4096,11008,128,1024`；`DDR Weight Width=4096`、`offset 2112/8208`、`buffer24_map0`（`:1282-1304`）；`Num Output Heads=32` |
| 静态 Task 邻接表 | `orchestrator/layer_id.py:34-46` | `_DQ_LINKS`/`_SOFTMAX_LINKS` 写死扇出；`assign_ids` 没接 PhaseSource |
| 算子编译器只提供了 | `opcompiler_bridge/` | 相位数、`phase-bytes`（只进 L2 尺寸） |

`from_fx` 已经把 FlagTree 的 `pim.phase`、`pim.phase-bytes`、`unit`、`kind`
等属性拿在手里，但 `layer_fields` 一个都没读。`create方案` §1.3.1 承诺的
“有 PhaseSource 时按 SSA 使用-定义连”没有实现（`_task_links` 只有静态分支）。

**建议**（与评审 1 §4.1 一致，但给出可验收判据）：

1. FlagTree 相位 op 上打全 `flp-min-exp/max-exp/mantisa`、`activation-mode`、
   `kantor-mode`、`fpsu-mode`、`transpose-type`、`lut-kind`；
2. `phase_plan.py` 解析这些 attr，`layer_fields` 优先读，静态表降级为 fallback；
3. 加**反证测试**：改一处 IR attr，txt 对应域必须变（现在只有 GML 侧反证，
   txt 侧没有）；
4. Task 扇出改从相位 op 的 SSA 使用-定义推导；
5. `H/I/HD/S/nh` 从 `config.json` + GML 边读，不写死。

### 3.2 对拍器仍把槽位/相位数字抹掉

`scripts/diff_prepare_out.py:116-126` 的 `_normalise_naming_value` 是
`re.sub(r"\d+", "#", value)`，与复核 2 §2.2 的建议（只归一节点号、保留
槽位/相位）相比没有变化。我用「保留 ≤8 的小数字（槽位/相位/Kantor 段号）、
只抹掉 qidx/params/节点号」的口径复测：

```
槽位/相位敏感命名差异 117 处（官方 104 之外的 12 处新增）
  新增暴露：12 处 RoPE 定标段号（Q mul_cos 5 vs 6、Sin 4 vs 6，各 3 处）
            6 处 Dataout 语义名（rope_mul/mul→self_attn_Reshape_*_cos/sin.bin）
            2 处 rope_add Datain 0/1 语义名
            4 处 Datain file 0（残差旁路）
            1 处 output scale factor buffer
```

**建议**：解析出 `(family, slot/phase, node)` 再把 node 当通配、slot/phase
当闭合量；给对拍器本身加单测（现在没有 `test_diff_prepare_out.py`）。

### 3.3 残差旁路：GML 缺边，txt 伪造域

- 我方 `add_1`（GML node 14）`input_count=1`，参考是 2（`input0_node_id=1`
  是图入口、`input1_node_id=11` 是 o_proj）；
- `layer_fields.py:1019-1028` 在只有一条边时把 `Datain file 1` /
  `Residual input buffer 1` 填成与槽 0 **相同**的值；
- mask 缺第二邻居时 `Residual input buffer 1` 填字面量 `0`（`:1019-1020`）；
- 副作用：`o_proj` 的 `Dataout file` 槽位判成 0（参考槽 1，`input_buffer_1_10.bin`），
  残差 `output scale factor buffer` 也少槽（参考 `input_0_sf_9.bin`）。

`from_fx.py:444-490` 已经留了 `entry_bypass` 的结构（入口缓冲多记一个 reader、
补边），但字典从未被填充，是**死代码**。建议按注释里说的先定命名契约
（生产者 `output_buffer` 按第一个消费者；第二个消费者在自己的
`input_buffer_0/1` 里声明共享边），补完边后写针对 `add_1` 的闭合测试；
在补齐之前，宁可让这类层**报错**也不要填伪造值（CLAUDE.md 的
「不写防御性兜底」同样适用）。

### 3.4 `#1` 槽的分配尺寸与声明尺寸不一致

`orchestrator/l2_alloc.py:86-102` 的 `_dual_slot1_size`：

```python
if op_type in ("Llama2Activation", "Llama2ActivationDQ"):
    return _ROPE_HD * 2          # = 256
return slot0_size
```

`rope_add_k`/`rope_add_q` 也命中这个分支，但它们的两个槽都是数据宽度 H
（8KB）：生成的 txt 实测

```
self_attn_Reshape_1_..._add_params_182.txt:
  L2 input buffer offset 1: 307200
  L2 input buffer size   1: 8192      ← 声明 8192
（分配器只按 256 记这块槽，后续可把别的缓冲放到 307200+256）
```

复核 2 §2.5 的本意是让 `#1` 按真实尺寸分配，实现时只改对了 mul 的表槽，
把 add 也归进了同一个特例。新闸门只查「两个 offset 不重叠」，不查
「分配尺寸 ≥ 声明尺寸」，所以没拦住。

**建议**：`_dual_slot1_size` 只对 `rope_mul`（phase 0/1）返回 `HD*2`；
`rope_add`（phase 2）返回 `slot0_size`；`_check_dual_slot_offsets_disjoint`
增加「size1 必须 ≥ txt 声明」的校验（两处判据必须同源，不能再各写一套）。

### 3.5 GML 的 dtype/extension 注解大范围缺失

| 字段 | 参考 | 我方 |
| --- | --- | --- |
| `input_buffer_dtype` | 153 | 77 |
| `output_buffer_dtype` | 197 | 39 |
| `input_data_extensions` | 193 | 79 |
| `input_sf_dtype` | 110 | 3 |
| `weight_buffer_dtype` | 73 | 3 |
| `output_sf_dtype` | 146 | 183 |
| `flp_min_exp/max_exp/mantisa` | 1/1/1 | 0/0/0 |
| `activation_lut_file` | 1 | 0 |

这些注解是对方的 GML schema 的一部分（写盘侧还要用
`input_buffer_dtype` 决定元素宽度，见 `export.py:409-412`）。
`activation_lut_file` 的缺失直接造成 `mlp_gate` 的 `Activation LUT file:
activation_lut_file_11.bin` 悬空（`write_identity_lut` 分支永不触发）。

**建议**：在 `from_fx` 里按“输入/输出张量的量化分析结果”统一补
dtype/extension 注解（Gemm/MatMul=输出 fp16、DQ=入 fp16 出 int8 等），
`gate/up` 补 `flp_*`、`activation_lut_file`；加一个「GML 字段覆盖」对拍
（参考字段集合 - 我方字段集合）作为回归指标。

---

## 4. P2：非阻断，但应按节奏清理

### 4.1 Softmax/DQ 相位 bin 的尺寸与内容

- `--seq-len 16` 导出时，相位 bin 按边形状 16×16=256 个元素写（Softmax
  `input_buffer_phase_0` = 512B），而层卡声明的 S=1024（参考 2048B）。
  结构轮“尺寸要准”的约定在相位族上不成立，建议 `export_gml.py` 接受
  编译期槽位数（S=1024）并用它定相位缓冲尺寸，或把对拍口径写成
  「相位 bin 尺寸按导出图、与 txt 的 S 无关」。
- DQ/Softmax 的相位内容来自 `np.zeros`/`softmax(zeros)`（`_dq_source`、
  `export.py:300-305`），exp LUT 是 `synth_exp` 占位。与“结构轮”约定一致，
  但应在文档里写明，并保证 `output_sf` 等“闭合公式域”的取值来自真实
  张量（域确认表 Q6 已给公式）。

### 4.2 其它

- `net.ini` 末行多一个 LF：参考 EOF 无换行（`xxd` 末字节 `39`），
  我方 `net_ini.py:40` 末行用 `LF`，整文件 +1 字节。
- `plan.py:184-187` `except Exception` 静默降级；`_drop_tail` FX 字符串；
  `_kv_cache_buffer`/`_cache_dma_buffer` 按 `node_id` 大小猜 K/V
  （`layer_fields.py:189-191`），应从 `Cache idx`/KV_DMA meta 取。
- `layer_hw_table.py:6` docstring 说“每个条目可带 `pending_q`”，
  `HwRow` 没有这个字段；`orchestrator/l2_alloc.py:4-5` 的
  “`L2 input buffer offset` 全图出现 0 次”与参考/自身实现不符。
- `_is_second_rope` docstring 仍写“K 走 Llama2ActivationDQ”，
  实际代码已改成 Q 走 DQ（`from_fx.py:171-178`），注释需同步。
- `layer_fields.py` 的 `_ROPE_ABSENT` 注释说 `skip compare` “照常写成独立
  一行”，但 `layer_render.py:42-47` 实际是**粘连**（且与参考 5/6 文件一致），
  两处注释互相矛盾。
- 死代码：`OrchestrationPlan.layer_kinds`、`layer_fields._weight_role`；
  `layer_fields.py` 1493 行，远超仓规约软上限。
- 仓库根的 509KB 会话记录仍未清理；`.gitignore` 建议加
  `*-local-command-*.txt`。

---

## 5. 文档需要更正的点

| 文档 | 说法 | 实测 |
| --- | --- | --- |
| `P0修复2` §验证 | “唯一未过的还是那 82 个既有缺失引用” | 82 只是 **txt 引用缺失**；`parser_output` 另有 272 个文件缺（按族归一，weight 三族 184 个）、298 个文件多；文件集合差异远大于 82 |
| `P0修复2` §增删 | “parser_output bin 数 3259” | 应写清我方 3257 个 `.bin`、参考 3231 个，并给出「missing 272 / extra 298」的构成；不要让 +26 的总数掩盖文件集合差异 |
| `P0修复2` §2.2 | “mul_cos/mul_sin 的槌位差异已修掉，被同数量的 rope_add/residual 顶替” | 修的是一次**槽位**差异；Q mul_cos 的**段号**（5 vs 6）与 mul_sin 的段号（4 vs 6）仍未修，靠数字归一隐藏 |
| `P0修复` / `生成方案` | 顶部无警示、正文仍可读到「VALUE_DIFF 0」「结构语义 78 处」 | 应像 `prepare_out-txt` 那样在顶部加“数字已过期，以评审 2/3 为准”的警示 |
| `prepare_out-txt` | 已加警示（做对了） | 但“当前不足”第 1 条仍写 78 处，应改成官方 104 + 独立口径 117 的分类 |
| `生成方案` §1.3.1 | “有 PhaseSource 时按 SSA 使用-定义连” | 未实现（`layer_id.py` 只有静态表） |
| `层字段/生成方案` | `Residual input buffer 1`“域已补齐” | 是伪造值（填槽 0 同值/字面量 0），不是补齐 |

---

## 6. 建议的修复顺序（按依赖与收益）

1. **先修对拍工具**（半天）：槽位/相位敏感归一 + 给对拍器加单测。
   否则后面每修一处都可能被 `#` 归一掩盖。
2. **P0-1 bmm 权重三族**（1 天）：from_fx 声明 + runtime 落盘 + 单测；
   闸门缺失数应从 82 降到 18 上下（其余 8 Sin、4 残差、1 LUT、4 Q-DQ、6 Kantor）。
3. **P0-2 裁剪尾部 + 文件族闸门**（1 天）：把 `_drop_tail` 结构化并前移到
   GML 生成前；加「parser_output 文件族 == 参考」闸门。
4. **P0-4 cache 引用 + RoPE 段号**（半天）：`_cache_dma_buffer` 读字段、
   from_fx 补 DMA 槽字段；mul 段号拆 K/Q。
5. **P0-3 DDR Orig 名**（1 天）：输入取生产者、输出取消费者/共享名。
6. **P1-3 残差旁路**（1–2 天）：补边 + 命名契约 + 测试；在此之前禁止填伪造值。
7. **P1-4 `#1` 尺寸 + 闸门尺寸校验**（半天）。
8. **P1-5 GML dtype/extension 覆盖**（1 天）。
9. **P1-1/F1–F3 值来源迁移**（数天）：IR attr → phase_plan → layer_fields，
   加 txt 侧反证测试。
10. **P2 清理**：net.ini EOF、注释、死代码、拆文件、仓库卫生。

**CI 验收清单**（在现有基础上新增）：

```bash
python -m pytest tests/ -q -k "not llama2_7b"
rm -rf /tmp/ci && python scripts/export_gml.py --layers 1 --seq-len 16 \
    --out-dir /tmp/ci --use-opcompiler --orchestrate --decode-block-only
python scripts/diff_prepare_out.py --mine /tmp/ci/prepare_out --ref <参考>
# 新增：
#   a) parser_output 文件族计数 == 参考（数字归一后逐族）
#   b) GML 字段覆盖：参考有、我方无的字段列表为空（或白名单）
#   c) txt 引用 bin 全部存在（已是闸门）
#   d) 每个 #1 槽：分配尺寸 ≥ 声明尺寸，offset 不重叠
#   e) 改一处 IR attr → txt 对应域必须变（txt 侧反证）
```

---

## 附录 A：本次实跑（可复现）

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler

rm -rf /tmp/opencode/review3
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/opencode/review3 \
    --use-opcompiler --orchestrate --decode-block-only
# 16/17 项通过；未过：txt 引用 bin 缺失 82（引用 2008）；3257 个 .bin

python scripts/diff_prepare_out.py --mine /tmp/opencode/review3/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out
# MATCH 45854 / VALUE_DIFF 104 / ALLOC 5846 / MISSING 0 / EXTRA 0 / 退出码 1

python -m pytest tests/ -q -k "not llama2_7b"
# 686 passed, 42 deselected in 123.43s
```

关键独立核对结果：

- GML：参考 73 节点带 `weight_buffer`（7 Gemm + 64 MatMul + 2 RMSNorm），
  我方 11（8 Gemm + 3 RMSNorm）；参考 2/7/36/2 个 RMSNorm/Gemm/DS/Reshape，
  我方 3/8/37/4。
- `parser_output`：我方 3257 `.bin`，参考 3231；按族归一缺 272、多 298；族差异
  `weight_buffer 73→13`、`weight_sf 73→11`、`weight_zp 73→11`、
  `activation_lut_file 1→0`、`self_attn_Reshape_*_cos/sin 各 1→0`；
  多出 `output_sf/output_zp 146→183`、`input_sf 110→118`、`input_zp 110→157`、
  `input_#_sf/#_zp 38→106`，相位六族各 308→312。
- 槽位/相位敏感命名比对：117 处（官方 104；另有 12 处 RoPE 定标段号 + 1 处 Dataout 被官方归一吃掉）。
- `add_1` 的 `L2 input buffer size 1` 声明 8192，分配器按 256 记账
  （`_dual_slot1_size` 对 `Llama2Activation*` 一律 `HD*2`）。
- `net.ini`：`[general]` 与参考逐字节相同；422 行执行序数字归一后一致；
  末行我方 `...params_6\n`，参考 `...ms_9`（无换行）。
