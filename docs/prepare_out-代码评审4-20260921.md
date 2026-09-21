# prepare_out 生成代码评审 · 第四轮（2026-09-21）

评审对象：当前工作区全部未提交改动（20 个 modified + 未跟踪的
`orchestrator/layer_fields.py`、`layer_hw_table.py`、`layer_render.py`、
`scripts/diff_prepare_out.py`、`tests/test_layer_fields.py` 与 9 份
`prepare_out-*` 文档）。

方法（全部结论都带可复现命令与实测数字，见附录 A）：

1. 通读 diff 与全部 `prepare_out-*` 文档；
2. 按文档给的组合命令实跑 GML+prepare_out
   （`--use-opcompiler --orchestrate --decode-block-only`，产物在 `/tmp/opencode/review4`）；
3. 跑官方对拍（`scripts/diff_prepare_out.py`）与全量快速回归；
4. 写独立脚本核对官方对拍器**没做**的四项：
   - GML 逐 op_type 字段覆盖（我方 vs 参考 `relay2gml_graph.gml`）；
   - `parser_output` 文件族计数与**每个文件的字节数**；
   - 运行期 `.bin` 尺寸（MatMul 权重、KV cache、相位缓冲）；
   - L2 分配器尺寸 vs txt 声明尺寸（instrument `buffers_from_layers`，
     重跑产物与正式产物逐字节相同后取数）。

---

## 0. 结论摘要

上一轮（`prepare_out-代码评审3修复-20260921.md`）声称的 P0 修复，
**方向成立、但没有一项完全闭环**：声明补上了，落盘尺寸不对；裁剪做了，
文件族仍不一致；闸门加了，但恰好不检查出问题的那几处。按严重度：

| 级别 | 问题 | 实测依据 |
| --- | --- | --- |
| **P0** | 运行期 `.bin` 按**导出图 seq_len=16** 写，与 txt/GML/参考契约的编译期 S=1024 不同源：MatMul `weight_buffer` 64 个全是 65536B（参考 131072B）；KV cache 平面 65536B（参考 4MB）、位置下标 2B（参考 192B）、新值 1B（参考 4096B）；DQ/Softmax 相位缓冲 512B（参考 2048B） | §2.1；文件字节数逐族比对 |
| **P0** | L2 分配尺寸与 txt 声明尺寸**不同源**：32 个 Mask 分配 64B 却声明 2048/2080B；Q 路 mul_cos `#1` 槽分配 256B 却声明 8192B；41 个双输入层 33 个欠分配。闸门只查「size1>0」和「两槽不重叠」，不查「分配 ≥ 声明」 | §2.2；instrument 实测表 |
| **P0** | KV cache 的初始输入没进 GML 契约：参考 4 个边界 buffer（K cache 4MB、V cache 4MB、位置下标 int16、输出）我方没有，节点 198 vs 200、边 326 vs 331、边界 buffer 6 vs 10 | §2.3 |
| **P0** | 12 个悬空引用里至少 3 个是**代码 bug 不是结构缺口**：`add_1` 的 `Datain file 0/1` 用合成名（盘上真实名是 `input_buffer_14.bin`）；Q 路 DQ 的 p1/p4 `Datain file` 写 `input_buffer_184.bin`，而盘上已有 `input_buffer_phase_0_184.bin` | §2.4 |
| **P0** | GML 注解字段大范围缺/多，`parser_output` 文件族与参考不同：参考独有 11 族/我方 0 族；`input_zp_#` +43、`input_#_zf/#_zp` 各 +72、`output_sf/output_zp` 各 +34；`input_buffer_#` −37、`input_buffer_#_#` −35。裁剪前移到 GML 后**没有**补评审 3 要求的「文件族 == 参考」闸门 | §2.5 |
| **P1** | 域值仍不是从 pimmlir 映射：`flp`/`Kantor`/`Fpsu`/`Transpose`/L2 切片 全在 `layer_hw_table.py` 静态表；`H/I/HD/S=4096/11008/128/1024` 写死在 `layer_fields.py:22`；PhaseSource 只贡献相位数与逐相字节数，`unit`/`force_consecutive` 解析后从未使用 | §3.1 |
| **P1** | `DDR Weight Orig Buffer Name` 仍写死 `buffer23_map0`/`buffer24_map0`（62 处，参考 `buffer19_map<h>`）；residual、mask、RoPE 的语义 Orig 名与槽位仍不对 | §3.2 |
| **P1** | 对拍归一化把 266 里 198 处判成「纯数字差异」——其中约 62 处是真实的 `map<头号>` 差异，约 50 处是小号节点号假阳性。**magnitude 判据无法区分节点号与槽位/相位/头号**，对拍器本身仍无单测 | §3.3 |
| **P1** | GML 多出 2 个 Reshape（4 vs 2），导致 residual 的输出 scale 名、Dataout 名与参考结构不同（`input_sf_13.bin` vs `input_0_sf_9.bin`） | §3.4 |
| **P2** | `net.ini` 末行多一个 LF 未修，且 `net_ini.py` docstring 与测试都写成「参考最后一行是 LF」——参考实测**无结尾换行**；`test_dual_slot1_size` 把 Q mul_cos 的错误行为钉死；仓库根 509KB 会话记录未清 | §4 |

**已核实为真、不要回退的修复**（§1）：Mask 第二路、RoPE cos/sin 表边界节点、
KV_Cache_DMA 三槽字段、RoPE 定标段号 5/6/4、Softmax 5 入 + 5 出、bmm 权重三族
**声明**、decode-block 尾部裁剪（RMSNorm/Gemm/DS 计数 2/7/36 与参考相等）、
写盘前清空陈旧 bin、`txt 引用闭合` 闸门、`pytest` 692 passed。

---

## 1. 已核实为真的修复（不要回退）

1. **Mask 第二路输入（causal mask）**：GML node 202 是唯一共享边界节点，
   32 个 Mask 的 `input_buffer_1` 指向它，文件真实落盘。
2. **RoPE cos/sin 表节点**：node 200/201 按 K 路锚点命名，
   Q/K 消费者读同一份共享名（`input_buffer_1_182.bin` / `input_buffer_2_182.bin`）。
3. **KV_Cache_DMA 三槽**：`input_buffer_0/1/2` + 槽 1 `int16` + `L2A_ignore`
   已声明（**但尺寸不对，见 §2.1**）。
4. **RoPE 定标段号**：sin→4、Q cos→5、K cos→6、add→1/2，
   官方对拍里这族已无差异；文件都在盘上。
5. **Softmax 5 入 + 5 出**：`write_softmax_phases` 与 GML 字段逐相一一对应，
   相位六族文件数与参考一致（各 308+4）。
6. **bmm 权重三族声明**：`weight_buffer/weight_sf/weight_zp` 在 64 个 MatMul
   与 7 个 Gemm + 2 个 RMSNorm 上共 73/73/73，与参考相等（**但尺寸不对，见 §2.1**）。
7. **decode-block 裁剪**：`op_type` 计数 RMSNorm 2 / Gemm 7 / DynamicScaling 36
   与参考相等；422 层、23 类计数、net.ini 数字归一后的 422 行执行序全部相等。
8. **闸门与回归**：写盘前清空 `out_dir/*.bin`；`pytest` 692 passed /
   42 deselected（124s，本次实跑）。

---

## 2. P0：会让仿真器读不到数据 / 数据尺寸错误 / 契约不一致

### 2.1 运行期 bin 按导出图 seq_len=16 生成，尺寸与 txt 声明/参考差 2~4096 倍

**证据（逐文件字节数）**：

| 文件族 | 参考尺寸 × 个数 | 我方尺寸 × 个数 | 差 |
| --- | --- | --- | --- |
| `weight_buffer_#.bin`（MatMul） | 131072 × 64 | **65536 × 64** | 少一半 |
| `input_buffer_0_<KV_DMA>.bin`（cache 平面） | 4194304 × 2 | **65536 × 2** | 少 64 倍 |
| `input_buffer_1_<KV_DMA>.bin`（位置下标 int16） | 192 × 2 | **2 × 2** | 少 96 倍 |
| `input_buffer_2_<KV_DMA>.bin`（新值） | 4096 × 2 | **1 × 2** | 少 4096 倍 |
| `input_buffer_phase_#_#.bin`（DQ/SM 相） | 8192×8 / 2048×192 / 22016×2 … | 1024×8 / **512×192** / 2752×2 … | 约 1/4 |
| `output_buffer_phase_#_#.bin` | 2048×64 / 1024×32 … | **512×64** / 256×32 … | 约 1/4 |

**根因**（代码位置）：

- 导出图是 `--seq-len 16`，所有按张量元素数落盘的路径都拿到了 16 token 的
  形状：`gml_bridge/export.py:526 _matmul_weight_elements` 取
  `elements_between[(input1_node_id, node)]`（KV cache 整张量 16×32×128=65536），
  只有取不到边时才回退 `1024*128`——回退值才是对的；
- KV_DMA 的槽 0/1/2、DQ/Softmax 相位缓冲同理来自
  `elements_between` / `Layer.phase_bytes`（见 `export.py` 写盘循环与
  `runtime_files.py:229 write_dq_phases`、`:296 write_softmax_phases`）；
- 而 txt 侧（`layer_fields.py`）用编译期常量 S=1024 算 L2 尺寸，
  `DDR Weight buffer size=131072`、`L2 output size=2080` 就是这么来的。
  **两边不同源**：GML/bin 是 16 token 的，txt 是 1024 槽位的。

**影响**：这是「结构轮尺寸必须准」这条自定契约（`P0修复2` §1、本仓
`AGENTS/CLAUDE` 约定）的直接违反。仿真器按 txt 声明去 DMA 时，MatMul 权重
越界一倍、KV cache 越界 64 倍、位置下标直接是错的。`Original cache file`
引用的 `input_buffer_0_181.bin` 只有 64KB，装不下一个 head 的 1024 token 历史。

**建议**（按此顺序）：

1. 定一条硬规则：**prepare_out 相关的一切尺寸以编译期槽位（S=1024）为准，
   导出图只提供拓扑与相位结构**。在 `gml_bridge` 里加一个
   `compile_slots()`（H/I/HD/S/nh 从 model config + CLI 传入），
   所有 `elements_between` 取数的地方改成「按槽位重算」；
2. MatMul：`_matmul_weight_elements` 不再优先取边，直接用 `S×hd`；
3. KV_DMA：槽 0 = `nh×S×hd×1`、槽 1 = `nh×3×2`、槽 2 = `nh×hd×1`；
4. 相位缓冲：`phase_bytes` 只在算子编译器里做结构校验，落盘尺寸按
   `(编译期 Width × elem_bytes)`；或让 FlagTree 以 `--pim-seq-len` 参数
   展开相位模板（本轮不必，见 §3.1 的 F1）；
5. 回归闸门（新增，见 §5 清单）：对每个文件族断言
   「盘上尺寸 == txt 声明尺寸」——现在这条恰好没人查。

### 2.2 L2 分配尺寸与声明尺寸不同源，33/41 双输入层欠分配

instrument `l2_alloc.buffers_from_layers` 后拿到的分配器尺寸 vs txt 声明：

| 层 | 声明 size0/size1 | 分配 size0/slot1 | 判定 |
| --- | --- | --- | --- |
| `mha_masking_head0..31`（32 层） | 2048 / 2048（输出 2080） | **64 / 64** | **欠分配 32 倍** |
| `self_attn_Reshape_qidx36_params_184_mul_cos`（Q 路） | 256 / **8192** | 131072 / **256** | **`#1` 槽欠分配 32 倍** |
| `self_attn_Reshape_1_..._mul_cos/sin`（K 路） | 8192 / 256 | 131072 / 256 | OK |
| `mlp_mul` | 22016 / 22016 | 22048 / 22048 | OK |
| `add_1` | 8192 / 8192 | 8224 / 8224 | OK |
| `add_2` | 8192 / 8192 | **无分配**（offset1 走 `offset0+size0` 回退） | 需查明 |

根因有两处：

1. **尺寸来源不同源**：`l2_alloc.buffers_from_layers` 用
   `layer.phase_bytes`（Mask 为 0 → 退到边宽 16 → `(16+16)*2=64`；
   DQ/SM 用算子编译器给的导出图相位字节），而 txt 的
   `L2 input/output buffer size` 用 `layer_fields._l2_out_size`（S=1024）。
   即 §2.1 的同一根因在地址分配上的表现。
2. **`_dual_slot1_size` 把「槽 1 是表」当成了 mul 的通用规则**
   （`l2_alloc.py:86-105`）：Q 路 mul_cos 的广播表在**槽 0**
   （`Eltwise broadcast input index=0` 时 `layer_fields.py:1355-1360`
   把 `w0,w1` 设成 `HD,H`），槽 1 是数据（8192B）。分配器不读这个字段，
   一律给 `#1` 返回 `HD*2=256`。评审 3 §3.4 的本意是「按真实尺寸分配」，
   这次修成了「一律按表尺寸」，把 Q 路反过来弄错了。

**影响**：当前布局**侥幸**没有实际重叠（把分配器的 buffer 列表重放后，
没有发现与「声明的超界区间」时间重叠的活缓冲），但分配器的 liveness
判据是建立在错误尺寸上的：Mask 实际要用 2048B、Q mul 实际要用 8192B，
任何一次布局变化都可能踩到相邻缓冲。闸门
（`export_gml.py:319-357`）只查 `size1>0` 和两槽地址不重叠，
**不查「分配 ≥ 声明」**，所以这个错误一路绿灯。

**建议**：

1. 让分配器和 `layer_fields` 共用同一个尺寸函数（`_l2_out_size` /
   `_l2_in_size` / `_l2_dual_in_size`），不要在 `l2_alloc` 里另算；
2. `_dual_slot1_size` 增加 `bcast` 参数：`bcast==1` → `HD*2`，
   `bcast==0` → `slot0_size`（Q mul_cos）；由 `buffers_from_layers`
   从 GML `Eltwise broadcast input index` 读取；
3. 闸门加一条：每层每个 `L2 input/output buffer size N` 必须 ≤
   分配器给同一 offset 的 slot 尺寸；并给 `test_dual_slot1_size`
   补 Q 路用例（现在它断言 Q 路 256，把错误钉死了）。

### 2.3 KV cache 初始输入没进 GML 契约（节点/边/边界 buffer 都对不上）

| 项 | 参考 | 我方 |
| --- | --- | --- |
| GML 节点 / 边 | 200 / 331 | **198 / 326** |
| `is_buffer` 边界节点 | 10（hidden、cos、sin、K cache、位置、V cache、mask、output…） | 6（hidden、cos、sin、mask、output、exit…） |
| K/V cache 初始平面 | node 4 / node 7，4MB int8 `input_buffer_0_28/33.bin` | 无节点、无声明 |
| 位置下标 | node 5，int16 [1,32,1,3] `input_buffer_1_28.bin` | 只在 DMA 槽字段里有个 2B 文件 |
| KV_DMA `residual_input_buffer` | `[4,5,29]`（cache、下标、新值） | `[182]`（只有 RoPE 新值） |

参考 `IO_info.txt` 明确列出 7 个输入（hidden / cos / sin / K cache / position /
mask / V cache）；我方缺 3 个。**没有初始 cache 的输入声明**，
仿真器起步时没有 KV 历史可读；这也是边数差 35 的主要来源。

**建议**：与 RoPE 表/Mask 同构，给 K/V cache 和位置下标各建一个
`is_buffer` 边界节点并入边；DMA 的 `residual_input_buffer` 按三个真实槽写
（现在是 `[182]` 一条）。尺寸按 §2.1 修完后一起回归。

### 2.4 12 个悬空引用里 3 个是代码 bug；另有 2 类文件没写

本次实跑未过项仍是「txt 引用的 bin 全部存在：缺失 12」。逐个核对后：

| 悬空名 | 性质 | 正确值 / 修法 |
| --- | --- | --- |
| `input_buffer_0_14.bin`、`input_buffer_1_14.bin` | **实现 bug**：`layer_fields.py:825-828` 对 `dual` 层直接 `names.data_buffer(nid, slot)` 合成；GML 节点 14 明明声明了 `input_buffer: input_buffer_14.bin`（盘上有，65536B） | 槽 0 先读 `_gml_str(node, "input_buffer")`，读不到再谈合成；槽 1 只有在真有第二条边时才写，否则报错（现在两槽都是凭空的） |
| `input_buffer_184.bin` | **实现 bug**：Q 路 DQ 节点在 `from_fx` 里没有裸 `input_buffer`（被 `pop` 换成了三槽），`layer_fields.py:760` 的 fallback `names.data_buffer(nid)` 拼出了不存在的名字 | p1/p4 的 `Datain` 对 `Llama2ActivationDQ` 用 `names.phase_input_buffer(nid, 0)`（盘上有 `input_buffer_phase_0_184.bin`，131072B，与参考 `input_buffer_phase_0_22.bin` 语义一致） |
| `activation_lut_file_11.bin` | GML 缺字段 + 缺写盘：`gemm_gate` 在 `layer_fields.py:734` 生成了引用，但 `from_fx` 没给门控 Gemm 写 `activation_lut_file`（参考 node 195 有） | from_fx 补 `activation_lut_file`；`write_runtime_files` 补写 `synth_silu()`（`runtime_files.py` 已有函数） |
| `kantor_A/B_*_189/9`（8 个） | GML 缺字段 + 缺写盘：`gemm_v` 的 Kantor A 三件（参考 node 36）、`mlp_mul` 的 A/B 五件（参考 node 194） | 同 KV_DMA 的做法：from_fx 声明字段 + runtime 写真实字节（scale 2B、bias 4B、Shift 1B） |

**注意**：`P0修复2` 把这 12 个统一归为「既有结构缺口」。核对后其中
3 个（add 的两个 + Q-DQ 一个）是本次改动引入的实现 bug——`add_1` 的
「不伪造」只停了复制槽 0 的老做法，却换成了**新造的**悬空名，
连原本能闭合的槽 0 都失去了；这不是遗留，是回归。

### 2.5 GML 注解与 `parser_output` 文件族不一致，且没有闸门

**GML 字段覆盖（逐 op_type，mine/ref）**：

| 字段 | 缺失方 | 实测 |
| --- | --- | --- |
| `input_buffer_dtype` | 我方 | MatMul 32/64、Softmax/dmask 0/32、Transpose/Split/Reshape 0/9 |
| `output_buffer_dtype` | 我方 | 全图 38 vs 197；Gemm 0/7、MatMul 0/64、Softmax 0/32、Mask 0/32、KV_DMA 0/2 |
| `input_data_extensions` | 我方 | 全图 77 vs 193 |
| `input_sf_dtype` | 我方 | 0 vs 64（MatMul）、0 vs 7（Gemm） |
| `weight_buffer_dtype` | 我方 | **Gemm 0/7**（只有 MatMul/RMSNorm 有） |
| `flp_min/max_exp`、`activation_lut_file` | 我方 | 0 vs 1（`mlp_gate`） |
| `input_sf/input_zp` | **我方多写** | DynamicScaling 36 vs 0、Split 3 vs 0、Transpose 4 vs 0；MatMul 0 vs 64、KV_DMA 0 vs 2 |

**`parser_output` 文件族**（数字归一后）：

| 方向 | 内容 |
| --- | --- |
| 参考独有 | `activation_lut_file_#`(1)、`kantor_{A,B}_*`(6)、`self_attn_Reshape[_#]_qidx#_params_#_{cos,sin}`(4) |
| 数量多 | `input_zp_#` +43、`input_#_{sf,zp}_#` 各 +72、`output_{sf,zp}_#` 各 +34、`input_sf_#` +5 |
| 数量少 | `input_buffer_#` −37、`input_buffer_#_#` −35 |
| 总数 | 参考 3231，我方 3409 |

评审 3 §2.2 要求的两件事都没做：
(a) 把裁剪按结构信息前移（做了，但 Reshape 4 vs 2 还在，见 §3.4）；
(b) **给闸门加「`parser_output` 文件族计数 == 参考」**——没加。
现在的 `_check_bin_references_closed`（`export_gml.py:284`）只查
「txt 引用 → 盘上有」，不查「盘上文件集合」，所以 `input_zp` 多 43 个、
`input_buffer` 少 37 个这种差异完全不可见。

**建议**：from_fx 按「输入/输出张量的量化分析结果」统一补 dtype/extension
（MatMul/Gemm=输出 fp16、DQ=入 fp16 出 int8、KV_DMA=槽 dtype 等），
删掉 DQ/Split/Transpose 上不该有的 `input_sf/zp`；加回归项
「`parser_output` 分族计数（数字归一）与参考逐族相等，缺/多都失败」。

---

## 3. P1：正确性 / 可维护性，不立刻阻断但影响验收

### 3.1 域值不是从 pimmlir 映射的（验收点 3）

现状与证据：

| 来源 | 位置 | 例子 |
| --- | --- | --- |
| 参考 422 层反推的静态表 | `orchestrator/layer_hw_table.py:94-174` | `l2_fpsu_size` 28672/57344/77824、`flp=(10,17,3)`、`transpose_type`、`l2_weights_off0/1`、`kantor_mode` |
| 硬编码 | `orchestrator/layer_fields.py:22` | `H,I,HD,S=4096,11008,128,1024`；`DDR Weight Width/Height=4096/1024`（`:1306-1307`）、`Num Output Heads=32`（`:920,1152,1323`）、`buffer23/24_map0`（`:1313-1314`） |
| 静态 Task 邻接表 | `orchestrator/layer_id.py:34-46` | `_DQ_LINKS`/`_SOFTMAX_LINKS` 写死扇出 |
| 算子编译器实际贡献 | `opcompiler_bridge/` | 相位数、`phase_bytes`（只进 L2 分配）；`unit`/`force_consecutive` 在 `plan.py:84-85` 赋值后**全仓无人读**（grep 实证） |

也就是说，`--use-opcompiler` 的反证（砍相位数 GML 会变）只证明了
「GML 的相位字段套数」依赖 IR；`prepare_out` 的数字域几乎全部来自静态表。
这与目标「从算子编译器生成出来的产物得到」不符。

**建议**（与评审 3 §3.1、生成方案 F1–F6 一致，给出可验收的最小步）：

1. FlagTree `ExpandPhases.cpp` 给相位 op 补 `pim.flp-min-exp/max-exp/mantisa`、
   `activation-mode`、`kantor-mode`、`fpsu-mode`、`transpose-type`、`lut-kind`；
2. `phase_plan.py` 解析这些 attr，`layer_fields` 优先读，静态表降级 fallback；
3. 加**反证测试**：改一处 IR attr → 对应 txt 域必须变（现在只有 GML 侧反证）；
4. `H/I/HD/S/nh` 从 `config.json` + GML 边读，删掉 `layer_fields.py:22` 的常量；
5. Task 扇出从相位 op 的 SSA 使用-定义推导。

### 3.2 `DDR Weight Orig Buffer Name` 与语义 Orig 名仍是占位

- 64 个 MatMul 的 `DDR Weight Orig Buffer Name` 我方恒为
  `buffer23_map0`/`buffer24_map0`，参考是 `buffer19_map<h>`（bmm1）/
  `buffer24_map<h>`（bmm2），**head 下标是真实语义值**。修法：`map{head_index}`，
  base 名按 KV_Cache_DMA 的语义缓冲取。
- `residual` 的 `output scale factor buffer`：参考 `input_0_sf_9.bin`
  （消费者 add_2、槽 0），我方 `input_sf_13.bin`（取了 `outs[0]` 的
  RMSNorm 消费者且无槽）。`layer_fields.py:981-989` 应优先选双输入消费者，
  与 `Dataout file` 的取法同源。
- RoPE mul/add 的 `Dataout`/`DDR Output Orig` 参考用
  `..._cos.bin`/`..._sin.bin`/`..._mul_cos_buffer` 语义名，我方是
  `input_buffer_N_x.bin`/`bufferN`。这条生成方案标了 PENDING（Q27），
  但参考文件都是**可推导的语义名**（不是 TVM 内部符），建议本轮一起修。

### 3.3 对拍器：magnitude 归一化区分不了节点号与槽位/相位/头号

对 266 个 VALUE_DIFF 做二级分类（把「纯数字差异」再按全数字通配是否相等切）：

- 198 处是「数字不同但非数字骨架相同」。其中：
  - **真差异**：`DDR Weight Orig` 62 处（`map0` vs `map5` 是头号，语义值）；
  - **假阳性**：`Original name` 7（qidx4 vs qidx36）、`Bias/Scaling/Dataout`
    等约 40+ 处（node 8 vs 193、qidx4 vs qidx36，单元号 ≤8 被当成槽位保留）。
- 68 处是真实结构差异：`DDR Input Orig 1` 37（mask `mask` vs `nprm_182_i168`）、
  `Datain`（Q-DQ `input_buffer_184.bin`）、`Datain 0/1`（残差）、
  `DDR Output Orig` 6 等。

结论：`_normalise_naming_value`（`diff_prepare_out.py:122-134`）的
「≤8 保留」把两类完全不同的数字混在了一起——既漏判（真差异被算成纯数字）
又误判（节点号被算成槽位）。评审 3 §3.2 建议的
「解析 `(family, slot/phase, node)`，只把 node 当通配」仍未实现，
对拍器也仍无单测（`tests/` 里没有 `test_diff_prepare_out.py`）。

**建议**：按 key 的 schema 归一（`qidx`/`params`/节点号 → `#`；
`_phase_N`/槽号/段号/`mapN` 保留），并给对拍器加夹具测试。

### 3.4 多出的 2 个 Reshape 影响下游命名

我方 GML 有 4 个 Reshape（`reshape`/`view`/`view_1`/`view_2`），
参考只有 2 个（`self_attn_Reshape_2/_7`）。其中一个后果已经在实测里：
`add_1` 的输出消费者顺序变成 `[RMSNorm 13, add_5 6]`，于是
`output scale factor buffer` 取了 RMSNorm 13 的裸名
（参考取残差 add_2 的槽 0）。Reshape 不占层，但会改变
`residual_output_buffer[0]`，所以不是「图结构不同、无影响」。

**建议**：查明参考把哪两个 reshape 折进了相邻算子（TVM Relay 侧隐式折叠），
在 `fuse_for_gml` 里加对应的折叠 pass；折不掉时，至少让
`_output_orig_name`/scale 选择按「双输入消费者优先」而不是 `outs[0]`。

### 3.5 残差旁路与死代码

- `from_fx.py:458` 的 `entry_bypass` 仍是**永远为空**的死代码；
  `plan.py:154-190` 的 `_drop_tail` 与 `from_fx._trim_decode_block` 是两套
  结构判据，同一件事写了两遍，容易漂移；
- `OrchestrationPlan.layer_kinds`、`layer_fields._weight_role` 无调用方；
- `layer_fields.py` 1556 行，超仓规约软上限，建议按
  DQ/Softmax/Gemm/Eltwise 拆四个模块。

---

## 4. P2：清理项

1. **`net.ini` 末行 LF**：参考 EOF **无换行**（`xxd` 末字节是 `39`），
   我方多一个 LF。`net_ini.py:9-15` 的 docstring 和
   `test_net_ini_line_endings_match_reference` 都写「参考最后一行是 LF」，
   实测是错的——应改完代码同时改测试与注释。
2. **测试把错误行为钉死**：`test_dual_slot1_size` 断言
   `_dual_slot1_size("Llama2ActivationDQ", 0, 8192) == 256`，
   而 Q 路槽 1 是数据、应为 8192；缺「MatMul 权重尺寸」「KV cache 尺寸」
   「相位缓冲尺寸」「分配 ≥ 声明」四类断言。
3. **文档与实现互相矛盾**：
   - `P0修复2` §「`#1` 槌用错误尺寸分配」说修完是「按真实表宽」，
     实际只对 K 路成立，Q 路反向；
   - `P0修复3` §验证表说「GML `weight_buffer` 73」对，但没验尺寸
     （实测 65536 vs 131072）；
   - `l2_alloc.py:4-5` 的「`L2 input buffer offset` 全图出现 0 次」
     与参考/自身实现不符；
   - `layer_hw_table.py:6` 说「每个条目可带 `pending_q`」，
     `HwRow` 没有这个字段。
4. **交付物布局**：参考是 `llama2_w4a8_decode_block_0/{parser_output, prepare_out}`
   两个**兄弟**目录；我方 `--out-dir` 下同时放 `.bin` 和 `prepare_out/`，
   而 net.ini 写的是 `dumps_bin_path = llama2_w4a8_decode_block_0/parser_output`。
   建议 `--out-dir` 语义改成 block 目录（bin 写 `parser_output/`，
   txt 写 `prepare_out/`），或提供 `--layout=reference`。
5. 仓库根 509KB 会话记录仍未清理；`.gitignore` 已加 `*-local-command-*.txt`，
   建议直接删除。

---

## 5. 建议修复顺序 + CI 闸门

按依赖与收益排序：

1. **尺寸同源**（P0-2.1）：所有 bin 落盘尺寸改用编译期槽位；
   先修 MatMul 权重、KV cache 三槽、相位缓冲。回归：文件族「尺寸 == 声明」。
2. **L2 分配与声明同源**（P0-2.2）：合并尺寸函数；`_dual_slot1_size` 按
   `Eltwise broadcast input index` 判；闸门加「分配 ≥ 声明」。
3. **补齐 KV cache 输入边界节点**（P0-2.3）：3 个 `is_buffer` + DMA 三槽
   真实入边；目标节点 200、边 331、边界 buffer 10。
4. **闭合 12 个悬空引用**（P0-2.4）：3 个命名 bug + LUT/Kantor 字段与写盘。
5. **GML 注解 + 文件族闸门**（P0-2.5）。
6. **对拍器 schema 归一 + 单测**（P1-3.3），否则后面每修一处都可能被
   数字通配掩盖或误报。
7. **语义 Orig 名 + map<头号>**（P1-3.2）。
8. **Reshape 折叠 / 消费者优先**（P1-3.4）。
9. **pimmlir 映射迁移**（P1-3.1，需要 FlagTree 改动，数天）。
10. P2 清理。

CI 验收清单（在现有基础上新增）：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
rm -rf /tmp/ci && python scripts/export_gml.py --layers 1 --seq-len 16 \
    --out-dir /tmp/ci --use-opcompiler --orchestrate --decode-block-only
python scripts/diff_prepare_out.py --mine /tmp/ci/prepare_out --ref <参考>
python -m pytest tests/ -q -k "not llama2_7b"

# 新增闸门：
#  a) 每个 bin 的字节数 == 它被引用处声明的尺寸（DDR Weight/L2/文件族）
#  b) parser_output 分族计数（数字归一）与参考逐族相等
#  c) 每个 #1 槽：分配尺寸 ≥ 声明尺寸，且 offset 不重叠
#  d) GML 字段覆盖：参考有、我方无的字段列表为空（或白名单）
#  e) 改一处 IR attr → txt 对应域必须变（txt 侧反证）
#  f) 对拍器的归一化有单元测试（节点号通配、槽位/相位/头号保留）
```

---

## 附录 A：本次实跑（可复现）

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler

rm -rf /tmp/opencode/review4
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/opencode/review4 \
    --use-opcompiler --orchestrate --decode-block-only
# 17/18 项通过；未过：txt 引用 bin 缺失 12
# 422 层；L2 461 块 → 6 槽，数据区 476768 字节；bin 3409 个

python scripts/diff_prepare_out.py --mine /tmp/opencode/review4/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out
# MATCH 45855 / VALUE_DIFF 266 / ALLOC 5683 / MISSING 0 / EXTRA 0 / 退出码 1

python -m pytest tests/ -q -k "not llama2_7b"
# 692 passed, 42 deselected in 124.20s
```

关键独立核对结果（脚本在 `/tmp/opencode/review4_tools/`，全部只读）：

- 文件族计数：参考 3231 `.bin` vs 我方 3409；参考独有 11 族、
  我方独有 0 族；`input_zp` +43、`input_#_{sf,zp}` 各 +72、
  `output_{sf,zp}` 各 +34、`input_buffer_#` −37、`input_buffer_#_#` −35。
- 字节数：MatMul `weight_buffer` 65536×64（参考 131072×64）；
  KV cache 65536/2B/1B（参考 4194304/192B/4096B）；
  相位缓冲 512×192（参考 2048×192）。
- GML：节点 198/边 326 vs 参考 200/331；op_type 计数除 Reshape
  （4 vs 2）外全部相等；字段覆盖表见 §2.5。
- L2：Mask 32 层 64 vs 2048/2080；Q mul_cos `#1` 256 vs 8192；
  41 个双输入层 33 个欠分配。
- `net.ini`：`[general]` 与参考逐字节相同；数字归一后 422 行执行序一致；
  末行我方 `...params_6\n`，参考 `...params_9`（无换行）。
- VALUE_DIFF 分解：198 处纯数字（含 62 处真实 `map<头号>` + 约 50 处
  ≤8 数字假阳性）+ 68 处非数字结构差异。
