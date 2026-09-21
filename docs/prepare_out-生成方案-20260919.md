# prepare_out 生成方案：每个文件、每个域、三方职责、FlagTree 遗留

日期：2026-09-19

配套：`docs/prepare_out-域确认表-20260918.md`（字段含义与待确认项）、
`docs/gml-pipeline-20260919.md`（GML 链路已打通、编排器停在清单）。
本文是「怎么从已有 GML + PhaseSource 生成 net.ini 与 422 个层 txt」的实现方案。

---

## 0. 目标与现状（短）

参考产物：
`xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out/`
- `net.ini`：`[general]` 10 个键 + `[layers]` 422 行 `layer = <stem>`
- `txt_files/`：422 层参数卡 + `gml_version.txt` + `l2a_version.txt`
- 穷举：23 类层、248 个字段族、每层 89–187 个键

开工前的本仓（本文写作时的起点）：
```
Llama2-7B → 图编译 6 个 pass → FlagTree -pim-expand-phases
         → GML 200 节点 + 2411 bin
         → orchestrator 层展开 / 发号 / L2 贪心
         → 只写出 net.ini 骨架 + layers.tsv，没有逐层 txt
```

**已实施完成**（结果与实测记录见 `docs/prepare_out-txt-20260919.md`）：

```
Llama2-7B → 图编译（q/k/v 共用 DQ、cos/sin 建表节点、语义 label）
         → FlagTree -pim-expand-phases
         → GML 199 节点 + 2330 bin
         → orchestrator 层展开 / 发号 / L2 贪心 / 逐层字段 / 渲染
         → net.ini + txt_files/ 422 层参数卡 + 2 个版本戳
```

对拍状态：文件名模式、**每个文件的域集合 422/422 与参考一致**，
闭合域 `VALUE_DIFF 0`。余 78 处为 TVM Relay 内部符号名（改不动，
等甲方确认），5872 处为片上地址与节点号（结构性）。

1 层导出含模型末尾 = 428 层；参考是纯 decode block = 422。
加 `--decode-block-only` 丢掉末尾 RMSNorm + lm_head + 它们的 DQ。

---

## 1. 三方现在有什么、这次加什么、FlagTree 遗留什么

### 1.1 图编译器（本仓 `graph/` + `gml_bridge/`）

已有 pass（`fuse_for_gml` 固定顺序，不重排）：

| 顺序 | 函数 | 文件 | 已产出、能直接喂 txt 的 |
| --- | --- | --- | --- |
| 1 | `fuse_rope` | `graph/fuse_rope.py` | `ROPE_META_KEY`：Q/K 两套、cos/sin 源 |
| 2 | `fuse_for_pim` | `graph/fuse_pim.py` | RMSNorm 六合一 + eps；Gemm+SiLU；attention 1/√hd 折进 Scaling |
| 3 | `fuse_graph` | `graph/fuse.py` | 通用主算子+尾激活（llama 路径基本已被 2 吃掉） |
| 4 | `split_attention_heads` | `graph/split_heads.py` | `HEAD_INDEX_META_KEY`、`HEAD_ROLE_META_KEY`（QK/mask/sm/PV） |
| 5 | `insert_kv_dma_and_split` | `graph/kv_dma_pass.py` | K/V cache 写出点、cache idx |
| 6 | `insert_dynamic_scaling` | `graph/quant_pass.py` | `DQ_META_KEY`：group_size、numel、是否分数 DQ |

`gml_bridge/from_fx.py` 已把这些变成 GML 节点字段：`op_type`、边 `dims`、
`input_buffer`/`output_buffer`/`weight_buffer` 文件名、dtype 字符串、
RoPE 子块、DQ/Softmax 的 `*_phase_N` 字段套数（套数来自 PhaseSource）。

**这次图编译器要加的（原语/pass，都在本仓，不改 FlagTree）：**

1. **语义层名原语**（不是新 pass，是 `from_fx` / `layer_fields` 读已有 meta）
   - 输入：`HEAD_ROLE` / `DQ_META` / `ROPE_META` / `RMS_NORM_META` / FX 参数名
   - 输出：参考风格前端名
     `self_attn_q_proj_MatMul`、`mha_batch_matmul1_head{h}`、
     `mha_softmax_head{h}`、`mha_masking_head{h}`、`RMSNorm`、
     `dynamic_quantization_params_{node}`、`mlp_gate_proj_MatMul` …
   - 文件名：
     `<前端>_qidx<GML id>_params_<主LayerID>[_gp|act_<dq|sm>_phase<P>_params_<相位LayerID>].txt`
   - **qidx 不与 TVM 对齐**（两边节点号不同）。对拍按 `(层类, phase, head)`。

2. **残差邻居原语**（`gml_bridge/from_fx.py` 补全，不新开 pass）
   - 已有 `residual_input_buffer` / `residual_output_buffer` = GML 节点号
   - 要保证：DQ 扇出到 q/k/v 三条时写出 `Residual output buffer 0/1/2`；
     eltwise 双输入写成 `Residual input buffer 0/1`
   - 数据来源：GML 边的 source/target，不是猜

3. **不新开图 pass 的理由**：拓扑、切头、DQ 插入、RoPE 融合、KV DMA
   已经在 6 个 pass 里。缺的是「GML 字段 → prepare_out 键名」的投影，
   那是编排器的工作。图侧只补「语义名」和「残差多槽」两处缺口。

### 1.2 算子编译器（FlagTree + `opcompiler_bridge/`）

已有：

| 项 | 位置 | 已产出 |
| --- | --- | --- |
| 整算子 MLIR | `opcompiler_bridge/oplevel_emitter.py` | `pim.quantize` / `pim.softmax` / `pim.rope` / `pim.matmul` |
| `-pim-fuse-activation` | FlagTree `FuseActivation.cpp` | 校验图侧融合，幂等 |
| `-pim-expand-phases` | FlagTree `ExpandPhases.cpp` | DQ 4 相、SM 5 相、RoPE 3 相；`pim.phase`、`pim.phase-bytes`、`unit`、`kind`、`pim.force-consecutive`、`pim.rotate-half` |
| 读回 | `opcompiler_bridge/phase_plan.py` | `PhasePlan` |
| 接线证明 | `export_gml.py` 反证 | 砍 DQ 相位数 GML 必须变 |

GML 字段**取值**仍在 `contracts/gml_hw_table.py`（文档 8.1）。

**本次不动 FlagTree 源码。** 下面是 FlagTree **必须记进遗留** 的缺口——
这些不补，txt 的「算」侧取值就只能继续查静态表，不能说「从 pimmlir 映射」。

#### 遗留 F1. 相位属性扩到 prepare_out 数字域（新 attr，挂在现有 expand 上）

`ExpandPhases.cpp` 的 `emitLut` / `emitReduce` / `QuantizeOp` 上现在只有
`pim.phase`、`pim.phase-bytes`、`unit`、`kind`。缺的、静态表在替它填的：

| PIM IR 应新增的属性 | 对应 txt 域 | 谁用 |
| --- | --- | --- |
| `pim.flp-min-exp` / `max-exp` / `mantisa` | `Flp min/max exp`、`Flp mantisa` | DQ p2=10/17/3，SM p2=9/16/3，倒数相=15/15/0，SiLU=10/17/3 |
| `pim.activation-mode` | `Activation mode` | 已有一处 DQ p1 写了 `activation_mode=1`，没推广、没进 PhasePlan 解析 |
| `pim.activation-special` | `Activation special operators` | 倒数=4，其余=0 |
| `pim.pooling-type` | `Pooling Type` | absmax/max→4，sum→3 |
| `pim.kantor-mode` 数字 | `Kantor mode` | 已有 `KantorMode` 枚举，没投影成 0/3/5 |
| `pim.fpsu-mode` 数字 | `Fpsu mode` | GML 是字符串，txt 是 1/2，映射未定义（Q3） |
| `pim.transpose-type` | `Transpose type` | SM p2=1，倒数相=2 |
| `pim.lut-kind` 稳定名 | 选哪张 288B 表 | identity / reciprocal / exp / silu |

做法（遗留，不在本轮）：在 `PIMAttrDefs.td` 加这些 attr；`ExpandPhases.cpp`
按相位模板写上；`phase_plan.py` 的 `_KIND_RE` 一类正则读回来；
`gml_hw_table.phase_fields` 改为「PhaseSource 有则用，无则静态表」。
判据仍是接上后 GML/txt 与静态表路径逐字节相同，再退役静态表。

#### 遗留 F2. LUT 张数与文件名由 IR 声明

现在 LUT 文件名在 `from_fx.py` 按相位号拼。IR 应声明
`pim.lut-file-count` 和 kind，编排器再调用 `gml_names.phase_lut`。

#### 遗留 F3. Kantor / FPSU 缓冲由 IR 声明张数

`kantor_A_scale` 等文件名同理。DQ p4 的 `Kantor A scale buffer file`
= p3 的 Dataout，这是边，不是常量，IR 里已经是 SSA 使用-定义，
`phase_plan` 应把「p4 的 Kantor A 读 p2 的结果」读出来
（注意 DQ 是 p1/p2 都读 p0，p4 的 Kantor 读 p2）。

#### 遗留 F4. Gemm/MatMul 不走 ExpandPhases

线性层、bmm、eltwise、vpu **没有**相位展开，所以
`Weight Format`、`Fpsu mode=2`、`L2 fpsu buffer size` 查表
目前只能放编排器 `layer_hw_table.py`。若以后要「从 pimmlir 映射」，
需要新 pass（例如 `-pim-emit-layer-params`）给 `pim.matmul` 打
`pim.weight-format` / `pim.fpsu-mode` / 切片字节。这是更大一块，
本轮明确不做。

#### 遗留 F5. RoPE 半旋转

只打了 `pim.rotate-half`，没拆成 split/neg/concat（文档 8.5）。
txt 里表现为 mul_sin 的 `rotary window size=64` 和 Kantor 系数。
本轮编排器按文档 B5 填 `rotary window size = hd/2`；真拆留给 FlagTree。

#### 遗留 F6. 静态表退役判据

补 F1–F3 后：同一份融合图，`phase_source=None` 与有 PhaseSource
的 txt 闭合域必须逐字段相同，然后删掉 `gml_hw_table` 里对应相位行。

### 1.3 编排器（本仓 `orchestrator/`）—— 本轮主体

**开工前**已有（下表的「200 节点」等数字是那时的状态；现状见 §0）：

| 模块 | 函数 | 已产出 |
| --- | --- | --- |
| `layer_expand.py` | `expand_layers` | 200 节点 → 层列表（phase、head） |
| `layer_id.py` | `assign_ids` | Layer ID、Task ID；**Prev/Next 误建成线性链** |
| `l2_alloc.py` | `allocate` | L2 offset 贪心，16 对齐 |
| `net_ini.py` | `render` | 只有 `layers_count`/`gml_version`，格式也不对 |
| `plan.py` | `orchestrate` | 串四步，不写 txt |

**这次编排器要加的：**

1. **修 Task 图（改 `assign_ids`，不是新文件）**
   - DQ：p1(Task0) next=1 和 2；p2 prev=0；p3 prev=0 next=3；p4 prev=2
     （参考 `dynamic_quantization_params_24_*`：p1 `Next task 0/1 = 1,2`）
   - Softmax：p2 next=2 和 4；p5 prev=1 和 3
   - 有 PhaseSource 时按 SSA 使用-定义连；没有时抄这两张静态邻接表
   - 单层：Task=0，两个 count=0

2. **新 `orchestrator/layer_hw_table.py`**
   - 66 个全层恒定域（常/硬）
   - B7 查表：`L2 fpsu buffer size`、`Fpsu mode`、`Kantor mode`、
     `L2 weights buffer size`、`Data scale buffer size`、
     `L2 weight scale buffer size`、`L2 * buffer id`、Format 共现、
     Flp 三元组、Transpose type
   - 每个条目：`source ∈ {const, hw, model, op, orch, pending}`，
     可选 `pending_q=N`

3. **新 `orchestrator/layer_fields.py`**
   - `build_layer_fields(...) -> OrderedDict`
   - 23 类层各一个填充函数，**只写该类该有的键**（见第 2 节清单）
   - 顺序按文档 E7：形状 → 卷积壳 → 量化/Kantor/FPSU → dump 名 →
     layer type → DDR → Task → L2 → Layer ID → 硬件口

4. **新 `orchestrator/layer_render.py`**
   - `键: 值` 文本；`Datain file` 双写；双输入展开 `xxx 0` / `xxx 1`
   - RoPE 的 `force consecutive execution` 与 `skip compare` **拆成两行**
     （参考有粘连，文档 Q27，我方不复现粘连）

5. **改 `net_ini.py`**
   - `[general]` 全套（见 2.0）
   - `[layers]`：`layer = <stem>`，stem 不含 `.txt`

6. **改 `plan.py` / `export_gml.py`**
   - 写出 `txt_files/*.txt` + 两个 version 戳
   - `--decode-block-only`
   - `--reference-dir` 时跑全量对拍（第 3 节）

---

## 2. 每个域怎么算（按 23 类覆盖全部 248 族）

符号（模，来自 `config.json` + 编译期槽位，图侧已有）：
```
H=4096  I=11008  nh=32  hd=128  S=1024  G=128
elem_bytes = {0:1, 1:2, 3:4}[Data Type]
align16(w) = ((w+15)//16)*16
```

来源代号：常 / 硬 / 模 / 算 / 编 / 待。
「待」仍填文档「我方当前算法」，对拍进 PENDING 栏，不混 MATCH。

### 2.0 `net.ini`

`[general]`（常/硬/运，全网一份）：

| 键 | 值 | 来源 |
| --- | --- | --- |
| is_seq_test / seq_tunneling / test_update_buffer | 0 / 0 / 0 | 常，Q32 |
| input_line_stride / input_map_stride | 8 / 4 | 待 Q10，样例恒定，照抄 |
| output_line_stride / output_map_stride | 12 / 5 | 同上 |
| dumps_bin_path | `<out-dir 相对>/` 或参考风格路径 | 运 |
| dumps_txt_path | `.../prepare_out/txt_files` | 运 |
| seq_output_bin_file | `/net.bin` | 常 |

`[layers]`：按 expand 顺序，一行 `layer = <stem>`，422 行。

`gml_version.txt` = `26.2.1`（`contracts.gml_quant.GML_VERSION`）。
`l2a_version.txt` = 本仓标识（不是对方 L2A hash，文档写明）。

### 2.1 422 层都有的 66 族（每层都写）

| 域 | 来源 | 公式 |
| --- | --- | --- |
| Number of frames | 常 | 1 |
| Input/Output Maps, Height | 模 | 1（压平，Q8 待确认 L2A 是否接受） |
| Input Width, Stride X | 模 | Width=numel，StrideX=Width；各类取值见 2.2 |
| Output Width, Stride X | 模 | 同左 |
| Output Stride Z | 模→编 | 终相 `align16(W)+15`；中间相 `=W`；Width=1 的 DQ p2 用 16（Q52） |
| Input/Output Data Type | 模 | 1B→0，2B→1，4B→3（Q1 完整表待确认） |
| Input/Output data extension | 模 | 整数 1，浮点 3 |
| Weights Data Type | 模 | 有 int4 权重的 Gemm=2，其余=0 |
| Is Winograd, Sparsity, Padding×4 | 常 | false, 0.0, 0 |
| Weight Compression Rate | 硬 | 1.0（Q45） |
| Kernel W/H, Filter stride | 算 | Gemm/MatMul=1 且 stride=1；其余 0 |
| Raster, Macro Tile*, Use Clipping, LeakyReLU, Fraction bits | 常 | 0 / False,1,1 / 0 / 0 / 0 |
| After/Before concat | 常 | false |
| skip compare | 常 | 1（Q31；RoPE 参考有粘连，我方仍单独一行） |
| Bytes in cycle read/write | 硬 | 64（手册 Table 7-11） |
| L2 qman offset/size | 硬 | `0x1FFF0000` / 65536（Q11 与手册不是一套） |
| Input/Output data order | 常 | 0（Q39） |
| Input Format | 算 | 常规 0；DQ p2/p3=6（Q2） |
| Quant_source | 算 | Gemm/MatMul 用 DQ scale=1，其余=0（Q51） |
| Activation Type | 算 | 走 LUT=13，否则 0 |
| Pooling Type / Filter / Stride / Pad | 算 | pooling 相见 2.3/2.4，其余 Type=0 窗口=0 |
| Fpsu source | 算 | softmax 链=1，其余=3（Q25） |
| Weights source | 算 | cache MatMul=0，其余=3 |
| L2 weights buffers per engine | 编 | Gemm/MatMul=2，RMSNorm=1，无权重=4 |
| L2 fpsu buffer id | 编 | 按同类层抄 f1..f5（Q35） |
| layer type, number of inputs | 算 | 见各类；eltwise=2 |
| Layer ID, Task ID, Prev/Next count | 编 | 见 1.3.1 |
| Sys virtual in/out | 编 | 中间相 true，落盘相 false |
| Virtual Input/Output 行 | 编 | 把 stem 拼进那行名字，true/false 同上 |
| Dataout file | 编 | 抄 GML `output_buffer` / phase 名；常双写 |

`Datain file` 在 381 层出现（RoPE add 等把 Datain 放在后段），不是 66 里的
「每层都有」，但几乎都有。生成规则：有输入就写，双输入写成 `Datain file 0/1`。

### 2.2 形状（模，从 GML 边 `dims` 读，禁止在编排器里写死 4096）

| 层类 | InW | OutW | 终相? |
| --- | --- | --- | --- |
| rmsnorm, residual, gemm_qko, gemm_v | H | H | 是 |
| gemm_gate, gemm_up | H | I | 是 |
| gemm_down | I | H | 是 |
| mlp_mul | I | I | 是 |
| bmm1 | hd | S | 是 |
| mask, sm_p2, sm_p5 | S | S | p5 是，p2 否（SZ=S） |
| sm_p1, sm_p3 | S | 1 | 否（标量 SZ=1） |
| sm_p4 | 1 | 1 | 否 |
| dq_p1 | W | W/G | 否（SZ=W/G） |
| dq_p2, dq_p3 | W/G | W/G | 否 |
| dq_p4 | W | W | 是 |
| rope_* | H | H | 是 |
| 分数 DQ（attn） | S | p1 出 1 | p4 是 |

W 是该 DQ 看到的激活长：hidden=H，MLP=I，分数=S。
`L2 input size`：int8=Width，fp16=Width×2；bmm 再 +16（hd→144，S→1040）。
`L2 output size`：终相 `(align16(W)+16)*elem_bytes`；bmm2 例外按 H 占位=8224（Q19）。

### 2.3 DQ 四相 ×37（算闭合 + 编文件名）

线性 5 条（24/12/196/193/22）×4 + 分数 32×4 = 148。组：线性 G=128，分数 G=S。

公共额外域（四相都有）：`dynamic quantization phase`、`Kantor mode`、
`Fpsu mode=1`、`Pooling data type=2`、`Use FPSU=1`、`Scale axis=1`、
`Bias/Scaling/Scaling PS buffer file`（`gml_names.phase_*`）、
`output scale factor buffer`、`Residual input/output`、`skip compare`、
`L2 fpsu size=1024`。

| 相 | 仅该相有的域 | 值 |
| --- | --- | --- |
| p1 | Output Format=6, Pooling Type=4, Filter W=G, Group data axis=3, Group data size=G, DDR Input*（形状=W）, Graph Input=1, Next task=1 和 2, L2 input size=W×2, Datain=原始 input_buffer_{id} | |
| p2 | Output Format=7, LUT=identity, Activation mode=1, special=0, Flp=10/17/3, DDR Output* 宽=Gn, Graph Output=1, Prev=0, Datain=p1 dump, L2 output size 按 Gn | |
| p3 | Output Format=4, LUT=1/x, mode=0, special=4, Flp=15/15/0, Transpose=2（样例有的是 1，以参考同类为准，Q24）, Datain=p1 dump（不是 p2）, Prev=0, Next=3 | |
| p4 | Output Format=1 或 0, Kantor=3, Kantor A source=1, Group kantor A size=G, Kantor A scale=p3 dump, Kantor A bias/shift 文件, DDR Output 宽=W, L2 input num=2, L2 output=(align16(W)+16)×1, Prev=2, Datain=原始输入再读 | |

Q 上那条 DQ（Llama2ActivationDQ）文件名带 `Reshape_qidx*_dynamic_quantization_*`，
并带 `Original name`、`Scale per tensor=1`。

### 2.4 Softmax 五相 ×32（算闭合；Task 扇出）

公共：`softmax phase`、`softmax axis=3`、`Split Head Index=h`、
`Kantor=0`、`Pooling data type=2`、`Scale axis=1`、残差邻居=mask 节点、
`input/output scale factor buffer`。

| 相 | 关键域 |
| --- | --- |
| p1 | layer=pooling, Pooling Type=4, Filter=S, Out dt=3, OutW=1, Output Format=4, Fpsu=1, Next=1, DDR Input 宽=S, 无 L2 fpsu 域, Datain=mask 输出 |
| p2 | layer=activation, Act=13, LUT=exp, Flp=9/16/3, special=0, Transpose=1, Fpsu=1, L2 fpsu=512, Bias=p1 Dataout, Prev=0, Next=2 和 4, Datain=与 p1 同一份原始分数, SZ=S（不对齐） |
| p3 | layer=pooling, Pooling Type=3, Filter=S, Out dt=3, OutW=1, Fpsu=1, 无 L2 fpsu, Prev=1, Next=3, Datain=p2 dump（input_buffer_phase_2） |
| p4 | layer=activation, Act=13, LUT=1/x, Flp=15/15/0, special=4, Transpose=2, Fpsu=**2**, Output Format=4, L2 fpsu=512, Prev=2, Next=4, In dt=3 |
| p5 | layer=activation, Act=0, Scaling=p4 Dataout, Fpsu=1, DDR Output 宽=S, Graph Output=1, L2 output=2080, Prev=1 和 3, Datain=p2 dump, 终相对齐 SZ=1039 |

### 2.5 Gemm 7 条

公共额外（qko/up/down 92 族，gate 再加 LUT，v 再加 cache/Kantor）：
`Group data axis=3, Group data size=128, Data scale*` 整组、
`DDR data scale*`、`Runtime data scale=false, Registry=true`、
`Weights buffer / weight_sf / Bias / Scaling / Scaling_PS`、
`input scale factor = 上游 DQ p2 dump`、`L2 weights double/partial=1`、
`L2 weights buffers per engine=2`。

| 层 | 差异 |
| --- | --- |
| q/k/o | Kantor=0, Fpsu mode=2, L2 fpsu=28672, L2 weights=32768, L2 wscale=131072, Data scale buf=32, Out dt=fp16, DDR Out H=1 |
| v | Kantor=3, Kantor A source=0 + 三个 Kantor A 文件, Fpsu=2, L2 fpsu=57344, Out dt=int8, Output Format=2, DDR Out H=S strideZ=H×S Orig=`value_cache_out`, Cache output=true, Cache idx=1, Num Output Heads=nh, Original cache file |
| gate | Act=13, LUT=activation_lut_file_{id}, Flp=10/17/3, special=0, L2 fpsu=77824, L2 weights=16384, L2 wscale=176128, Data scale buf=16, OutW=I |
| up | 同 gate 但不带 LUT, L2 fpsu=77312 |
| down | InW=I, Data scale width=86, Data scale buf=30, L2 weights=30720, L2 wscale=122880, L2 fpsu=28672 |

闭合：`weight_buffer 字节=N*K*1`，`weight_sf=N*(K/128)*2`，
`Data scale width=K/G`，`DDR data scale size=width×2`（q=64；bmm 分数=16）。

### 2.6 MatMul 64

bmm1 与 bmm2 额外族几乎相同（115 vs 116），bmm2 多 `Head output`。

| | bmm1 | bmm2 |
| --- | --- | --- |
| Weight Format | 3 | 2 |
| InW / OutW | hd / S | S / hd |
| Group data size | hd | S |
| Data scale width/buf | 1 / 2 | 1 / 2 |
| Weights buffer | K cache | V cache |
| Head input / output | 1 / 无 | 0 / 1 |
| Cache idx | 0 | 1 |
| DDR Weight offset | 2112（Q20 待，32 头相同，抄） | 8208 |
| DDR Weight size | S×hd=131072 | 同 |
| L2 input | 144 | 1040 |
| L2 output | 2080 | 8224（按 H） |
| L2 fpsu / weights / wscale | 7168 / 4096 / 2048 | 1024 / 32768 / 256 |
| Split Head/Weight index | h / 32 | 同 |
| Datain | Q 的 DQ p4 | 该头分数 DQ p4 |
| input scale factor | Q 的 DQ p2 | 该头 DQ p2 |

### 2.7 mask ×32

`Eltwise mode=2`，`Mask Input/DataType/Buffer Index=1`，Use FPSU 0/1=0/0，
InW=S，Datain 0=bmm1 出，Datain 1=共享 mask，`DDR Input TVM Orig Buffer Name 1`
可写语义名（不必追 TVM 的 `nprm_*`，对拍这族进 PENDING/ALLOC 类文件名差），
L2 fpsu=7168，L2 in/out=2048/2080，Output Format=1，Split Head Index=h。

### 2.8 RoPE 6 层

三连都有：`Llama2Activation=True`，`force consecutive execution=1`，
`Original name`=reshape 后 Q/K 名，`Scale per tensor=1`，L2 fpsu=57344，
双输入 Datain/DDR/L2。

| | mul_cos | mul_sin | add(K) | add(Q) |
| --- | --- | --- | --- | --- |
| Eltwise mode | 1 | 1 | 0 | 0 |
| Kantor | 5 | 5 | 3 | 0 |
| broadcast dim/factor/index/strideX | 3 / nh / 1 / hd | 同 | 无 | 无 |
| rotary window size | 无 | hd/2=64 | 无 | 无 |
| Out dt | fp16 | fp16 | int8 | fp16 |
| Cache output / Num Output Heads | 无 | 无 | true / nh | 无 |
| Runtime input 0/1 | Q 的 mul 为 1 | 同 | 无 | 无 |
| Kantor A source | 0（常量 bin） | 0 | 3 的 add 用 0 或 3 | 0 |

### 2.9 残差 Add ×2、mlp_mul ×1、RMSNorm ×2

残差：mode=0，Kantor=0，Width=H，Fpsu mode 0/1=1/1，Use FPSU=1/1，
Pooling data type 0/1=2/2，L2 fpsu=28672，L2 in=8192，L2 out=8224。
图入口 RMSNorm 的 Datain、图出口 add_2 的 Dataout 带
`DDR * TVM Orig Buffer Name`（语义名即可）。

mlp_mul：mode=1，Kantor=5，Kantor A/B 各一套文件，Width=I，
L2 fpsu=154624，L2 in/out=22016/22048，Output Format=1。

RMSNorm：`layer type=vpu`，`sublayer type=rmsnorm`，`Vpu Axis=-1`，
Use FPSU=0，Bias=`RMSNorm_Add_Const_{id}.bin`=fp32(eps)，
weight_buffer=4096B（Q21 布局待确认，按 4096 字节写），
weight_sf=fp32(1/127)，L2 weights=4096，double/partial=0，
buffers per engine=1，L2 fpsu=512，无 Kantor/Fpsu mode 域。
RMSNorm 有一组**大小写重复键**（`Bias Buffer File` 与 `bias buffer file`），
生成时按参考双写。

### 2.10 查表禁止临场凑（B7）

`L2 fpsu buffer size`、`Data scale buffer size`、`L2 weights buffer size`、
`L2 weight scale buffer size`、`Fpsu mode`、`Input/Output Format`、
`L2 * buffer id`、`Flp` 三元组：全部进 `layer_hw_table.py`，
按 `(层类, phase)` 查。换 H/I 要重测，文档写明。

L2 offset 数值：用已有贪心分配。对拍不要求等于参考（文档 8.3），
只要求 16 对齐、双缓冲间距=size、不重叠。栏位 `ALLOC`。

DDR Orig Buffer Name：线性层 `bufferN` 不追 TVM 编号；KV 用
`value_cache_out` / K cache 语义名（Q48）。

---

## 3. 验证：每个文件、每个域

> **本节是设计。实际命令、实测输出、以及「对拍器一度放水」的教训，
> 见 `docs/prepare_out-txt-20260919.md` 的「怎么验证」一节。**
>
> 落地后补的一条判据：白名单只允许放**结构性**不可对齐的三类（节点编号、
> 片上/DDR 地址、TVM 专属名）。尺寸、模式、枚举、槽位 id 一律当闭合域 ——
> 第一版把它们一起放进白名单，报了假的 `VALUE_DIFF 0`，绕过白名单的原始
> 比对实际有 6582 处不等。所以对拍之外还要有**不看白名单的原始核对**。

对拍单位 `(层类, phase, head_index)`，**不用文件名字符串**
（qidx 两边不同是预期的）。

`scripts/diff_prepare_out.py` 分五步，缺一步都不算过：

**步 A — 文件集合**
- 我方 `--decode-block-only` 必须 422 个层 txt + 2 个 version
- 23 类计数必须与参考完全相同：
  `dq_p1..p4` 各 37，`sm_p1..p5` 各 32，`bmm1/bmm2/mask` 各 32，
  `gemm_qko` 3，`gemm_v/gate/up/down` 各 1，RoPE 三类各 2，
  `rmsnorm` 2，`residual` 2，`mlp_mul` 1
- 多/少的文件进 `UNMATCHED_FILE`，非 0 退出

**步 B — 配对**
- 从字段读 `layer type`、`dynamic quantization phase` / `softmax phase`、
  `Split Head Index`、`Eltwise mode`、`Llama2Activation`、InW/OutW、
  是否 `Cache output`，映射到 23 类
- 同类同 phase 同 head 一对一；配不上进 `UNMATCHED_PAIR`

**步 C — 域集合（每一对）**
- 参考有我方无 → `MISSING`（实现漏，必须修）
- 我方有参考无 → `EXTRA`（警告；默认不挡，人工看是不是多写）
- 键名归一化：`Datain file` 与 `Datain file 0` 视为不同；
  RMSNorm 大小写两套都要对上

**步 D — 域值（每一对的每一个共有键）**
- source∈{常,硬,模,算} 且值等 → `MATCH`
- 同上且值不等 → `VALUE_DIFF`（闭合域零容忍）
- source=编 且键是 L2 offset / L2 buffer id / DDR Orig Buffer Name /
  TVM Orig Buffer Name / qidx 出现在文件名 → `ALLOC`
  （offset 只查 16 对齐、双缓冲差=size；名字不追 TVM）
- source=待 → `PENDING_MATCH` 或 `PENDING_DIFF`
- 多值键（Datain 双写）：两边出现次数都要相等

**步 E — 报告与退出码**
- 打印：23 类各多少对、每类 MATCH/VALUE_DIFF/MISSING/PENDING_DIFF/ALLOC 计数
- 每个 VALUE_DIFF / MISSING 打出：层类、head、phase、键、参考值、我方值
- 退出码：有 `MISSING` 或 `VALUE_DIFF` 或 `UNMATCHED_*` → 1
- `PENDING_DIFF` / `ALLOC` / `EXTRA` 打印，不挡（清单进 docs「当前不足」）

测试命令（用户要求全跑）：

1. `python -m pytest tests/ -q -k "not llama2_7b"`
   新增：`test_layer_fields.py`（23 类各一个夹具，断言该类**应有键都在、
   不应有键不在**、闭合域数值）、`test_layer_id.py` 扇出图、
   `test_diff_prepare_out.py` 对拍器自己的夹具。
2. `python -m pytest tests/ -q -k "llama2_7b"` —— 原 numpy 链路回归。
3. `python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_full
   --use-opcompiler --orchestrate --decode-block-only`
4. `python scripts/diff_prepare_out.py --mine /tmp/gml_full/prepare_out
   --ref .../llama2_w4a8_decode_block_0/prepare_out`
   **全量 422 文件 × 每文件全部键**。闭合域 VALUE_DIFF 必须 0。

32 层全模型不对拍 1-block 参考（层数 32 倍）。prepare_out 对拍固定 1 层
decode block。

---

## 4. 代码落点（已全部完成）

| 顺序 | 文件 | 做什么 | 状态 |
| --- | --- | --- | --- |
| 1 | 本文 | 方案 | 完成 |
| 2 | `orchestrator/layer_id.py` | Task 扇出（DQ p1→p2/p3，Softmax p2→p3/p5） | 完成 |
| 3 | `orchestrator/layer_hw_table.py` | 恒定域 + B7 查表，带 source / pending_q | 完成 |
| 4 | `orchestrator/layer_fields.py`、`layer_render.py` | 23 类填充 + txt 文本 | 完成 |
| 5 | `orchestrator/net_ini.py`、`plan.py`、`scripts/export_gml.py` | `[general]`、落盘 txt、`--decode-block-only` | 完成 |
| 6 | `scripts/diff_prepare_out.py` + 测试 | 全量文件×域对拍 | 完成 |
| 7 | `docs/prepare_out-txt-20260919.md` | 研发记录 | 完成 |

### 计划外但必须做的改动

写方案时没预见、落地时被参考产物逼出来的（明细见研发记录）：

| 文件 | 为什么非改不可 |
| --- | --- |
| `graph/quant_pass.py` | 参考里 q/k/v 共用一条 DQ、gate/up 共用一条。原先「每个 Gemm 各插一条」会多出 12 层，层数永远对不上 422 |
| `gml_bridge/from_fx.py` | 三处：① 语义名写进 GML `label`（参考的 label 就是 txt 文件名主干，两边各拼一套必然发散）；② FX 名移到内部键 `pim_fx_name`，写盘取 DQ spec、取 eps、层展开认 `headN` 都改读它；③ cos/sin 建成 `is_buffer` 边界节点并连边进 RoPE（否则 `input_count` 记 1 而参考记 3，下游 DQ 缺 `Residual input buffer 0/1/2`） |
| `opcompiler_bridge/oplevel_emitter.py` | Q/K 两条 RoPE 的 DQ 挂反了：参考是 **Q** 带 4 相 DQ、K 的 add 写 cache |
| `orchestrator/layer_render.py` | 行尾必须 CRLF；`force consecutive execution` 与 `skip compare` 在 5 个 RoPE 文件里**粘成一行**（Q27），K 路 add 不粘 |

`layer_fields.py` 约 1300 行，超过方案里定的约 400 行软上限，**待按
DQ / Softmax / Gemm / Eltwise 拆**（记在研发记录的「其它」里）。

---

## 5. 明确不做（落地后逐条复核）

| 项 | 说到做到？ | 说明 |
| --- | --- | --- |
| 不改 FlagTree 源码 | 是 | 只改了 `opcompiler_bridge/oplevel_emitter.py`（本仓侧的 Q/K 分派），FlagTree 的 `.cpp` 一行未动。遗留 F1–F6 见 §1.2 |
| 不启用多卡 | 是 | `gml_bridge/sharding.py` 仍默认单卡 |
| 不追 L2 offset / qidx / TVM 名 | 是 | 这三类是最终 5872 + 78 处差异的全部来源，结构性不可对齐 |
| 不把 bin 内容纳入对拍 | 是 | 只对 `.txt`；bin 由 `write_runtime_files` 落盘，交叉校验只查「引用集 == 落盘集」 |
| 不实现序列变长 / 自动切分 / 异步 dispatch | 是 | |
| 不退役 `gml_hw_table` | 是 | 等 F1–F3 把属性打进 PIM IR |

### 落地后新增的「不做」

- **残差旁路的边不接**。域已补齐（域集合 422/422 对齐），但 GML 里那条边仍缺。
  补边与规则 2「缓冲按第一个消费者编号」冲突，要改命名契约、波及 2330 个 bin，
  影响面只有 `input_count` 一个域。已回退，`from_fx.py` 的 `entry_bypass`
  注释记了复现路径。
- **不猜 TVM buffer name 的语义**。参考把 `Orig Buffer Name` 与
  `TVM Orig Buffer Name` 写成同值，无法判断哪个是索引键。猜错代价不对称
  （当标签而实为索引 → 仿真器找不到缓冲），所以等甲方回 Q-A 再动。
