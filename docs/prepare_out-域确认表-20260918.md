# prepare_out 域确认表（给甲方填空）

面向：Ceva-NeuPro-M × Llama2-7B W4A8 decode 单层（block_0）  
样例：`llama2_w4a8_decode_block_0/prepare_out`（`net.ini` + `txt_files/` 424 个文件）  
配套：`parser_output/` 二进制常量、`relay2gml_graph.gml`（200 个节点）、`Ceva-NeuPro-M High-Level ArchSpec V1.6.6.GA`、`VBU-GML Structure`  
日期：2026-09-18  
用途：甲方逐行确认「含义 / 取值 / 计算公式 / 责任方」。我方按确认结果在图编译器、算子编译器、编排器里补原语和 pass，最终由 AI 编译器生成同类文件。

---

## 0. 怎么用这张表

### 0.1 怎么读这张表

对每一行：

需要对方拍板的项全部收在第 15 章，每条都写了样例文件名和域名。第 1–14 章只讲已经能从模型、手册或 422 层统计对上的规则；正文里不再夹确认问句。

### 0.2 置信度

| 标记 | 含义 |
| --- | --- |
| 已核实 | 手册有定义，或全量 422 层统计无反例，或 `.bin` 字节数对得上 |
| 高 | 多条独立证据一致，但手册没有逐字段定义 |
| 中 | 样例规律清楚，枚举或边界仍可能有第二种读法 |
| 待确认 | 第 15 章列出的项，生成该域前需要对方书面口径 |

### 0.3 责任方（我方落地时的模块）

| 代号 | 谁 | 在本工具链里对应什么 |
| --- | --- | --- |
| 图 | 图编译器 | 拓扑、形状、dtype、量化标注、head 切分、融合、命名 |
| 算 | 算子编译器 | 相位模板、LUT / Scaling / Kantor 内核参数、tiling 切片尺寸 |
| 编 | 编排器（参考工具链里是 L2Analyzer） | Layer ID、Task 图、L2 地址、buffer 直通、执行序 |
| 运 | 运行时 / 仿真器装载 | DDR 实地址、KV cache / cos / sin / mask 物化、dumps 路径 |
| 硬 | 硬件能力库（编译期查表，不随模型变） | 位宽、对齐、L2 窗口、QMAN 尺寸、引擎数 |
| 常 | 本样例观察到的常量（是否永远恒定，见第 15 章） | 例如本样例 `skip compare=1`、`Number of frames=1` |

一条域可以有两个责任方：图给出形状，编排器给出 offset。表里写「图→编」表示上游提供输入、下游算出最终值。

### 0.4 链路方向（先对齐，再看域）

```
HuggingFace Llama2-7B
        │
        ▼
图编译器 + 算子编译器  ──►  GML（算子图，本样例 200 节点）
        │
        ▼
L2Analyzer（L2A）      ──►  prepare_out（层参数，本样例 422 层 + net.ini）
        │
        ▼
底层编译器 / 仿真器
```

要点：

1. **一个前端算子可以展开成多个层。** GML 200 节点 → prepare_out 422 层。差出来的 222 层几乎全是 DynamicQuantization 的 4 相位和 Softmax 的 5 相位。
2. **GML 描述算子；层描述一次硬件遍历。** 硬件最小可编程单位是「一个引擎在存储层级之间的一次流式遍历」，不是 PyTorch op。
3. 目标：图编译器 + 算子编译器生成 GML（以及 L2A 能接收的等价信息）。若对方要求直接生成 `prepare_out`，表里标「编」的域也要由我方实现。交付边界见 Q9。

---

## 1. 样例对应的模型与硬件约定

### 1.1 Llama2-7B（decode 一步、一层）

来源：`Llama-2-7b-hf/config.json`。

| 符号 | 值 | 在 prepare_out 里的直接体现 |
| --- | --- | --- |
| `hidden_size` H | 4096 | RMSNorm / qkv/o_proj / residual 的 Width=4096 |
| `intermediate_size` I | 11008 | gate/up 输出、down 输入、mlp_mul Width=11008 |
| `num_attention_heads` | 32 | 32 套 bmm1/mask/softmax/bmm2；`Total Split Head Num=32` |
| `head_dim` | 128 = 4096/32 | bmm1 输入宽 128，bmm2 输出宽 128 |
| `rms_norm_eps` | 1e-5 | `RMSNorm_Add_Const_*.bin` 4 字节，按 fp32 解 = `1e-5` |
| `hidden_act` | silu | 仅 `mlp_gate` 的 `Activation Type=13` + `activation_lut_file_195.bin` |
| 本样例 KV 槽位数 S | 1024 | IO_info：K/V cache shape `[1,32,1024,128]`；mask/softmax Width=1024 |
| 本样例 batch / seq | 1 / 1（decode 一步） | 几乎所有层 Height=1、Maps=1 |
| 权重量化组大小 G | 128 | `Group data size=128`（线性层）；scale 组数 = K/128 |

量化方案 **W4A8**：

- 权重逻辑 int4，一字节存一个值（符号扩展，值域 −8..7），`Weights Data Type=2`
- 激活线性段 int8，`Input Data Type=0`
- 非线性 / 残差 / RMSNorm / Softmax 工作在 fp16，`Data Type=1`
- Softmax 归约标量走 fp32，`Data Type=3`

### 1.2 形状压平（几乎所有形状域的根）

本工具链把张量看成「单 map 平面图」：

```
Maps   = 1
Height = 1
Width  = numel(该层看到的张量)
StrideX = Width
```

例外（保留多维视图，只出现在 DDR 侧，不出现在 Input Width）：

- KV cache：`DDR Weight/Output` 为 `Width=4096, Height=1024, strideZ=4096×1024=4194304`
- RoPE 的 cos/sin 表：逻辑形状 `(1,1,1,128)`，通过 `Eltwise broadcast factor=32` 播到 32 头

L2A 是否接受这种压平，见第 15 章 Q8。

### 1.3 本样例观察到的硬件常量

| 量 | 本样例值 | 我方理解 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- |
| 内部存储器每周期读写字节 | 64 / 64 | 全 422 层恒定。手册 Table 7-11：NPM4K 及以上 L2MSS SysDMA 内部口 64 字节/周期 | 已核实 |  |
| `L2 qman buffer size` | 65536 | DMA Queue Manager 队列区，全层恒定。手册 7.3：DMA Manager 含 DMA Task Queue Manager | 已核实 |  |
| `L2 qman buffer offset` | 536805376 = `0x1FFF0000` | 样例全层同一值，`+65536 = 0x20000000` | 高 | 与手册地址表的关系见 Q11 |
| L2 容量（手册） | 1 MB–32 MB | 手册 7.3 / Table 7-4：L2M 从 1 MB 到 32 MB。Table 12-1：NPM2K=1MB，NPM4K/8K=2MB。图 2-2 仍写 0.75MB–32MB，但 1.6.1.GA 变更记录写明「删除 L2MSS 的 0.75MB」。**768 KB 是 L1 选项**（7.2.1：NPM4K 及以上 L1 = 512/768/1024 KB），不是 L2 | 已核实（手册） |  |
| L2 内部地址（手册 Table 7-12） | 起点 `0x05000000`，空间 32 MB | 到 `0x07000000`。样例里的 `0x1FFF0000`、`0x1FF9F1C0` **落在这份表外面**，更像 4 GB 虚拟地址（第 8 章 DACU 把 32 位 VA 翻成 64 位 PA） | 手册已写清，样例对不上 | 见 Q11 |
| 对齐粒度 | 16 个元素 | `Output Stride Z = align16(W)+15` 对终相平面成立 | 高 | 例外见 Q52 |
| PWL LUT | 32 段 × (slope+intercept) fp16，文件 288 字节 | 手册 4.3.3；后 80 个 fp16 在恒等表里全 0 | 已核实（前 128 项布局） | 后 80 项见 Q26 |
| 动态量化最小组 | 手册最小 16；本样例线性 G=128 | 手册 1.6.x：16/32/64/96/128 | 已核实 | 分数 DQ 的 G=1024 见 Q6 |
| 权重未压缩 | `Weight Compression Rate=1.0` | 手册 7.3.4 WDM 可选；本样例未开。`weight_buffer_23.bin` = 4096×4096 字节 | 已核实（本样例） | 交付形态见 Q45 |

### 1.4 由 Llama2-7B 和手册直接算出的量

模型：`Llama-2-7b-hf/config.json`。手册：ArchSpec 3.4 / 4.2 / 4.3.4 / 10.5。

```
H  = hidden_size            = 4096
I  = intermediate_size      = 11008
nh = num_attention_heads    = 32
nkv= num_key_value_heads    = 32          # 等于 nh，本样例是 MHA 不是 GQA
hd = H / nh                 = 128         # head_dim
eps= rms_norm_eps           = 1e-5
act= hidden_act             = silu
G  = 权重组大小              = 128         # 手册 3.4 / 10.5 允许 16/32/64/96/128
S  = KV 槽位数               = 1024        # 本样例 IO_info，不是 max_position_embeddings（4096）
```

`I % G = 11008 % 128 = 0`，所以 down_proj 的组数是整数 86，不用 ceil。

| 要生成的量 | 公式 | 样例落点 |
| --- | --- | --- |
| q/k/v/o_proj 的 Input/Output Width | H = 4096 | `self_attn_q_proj_MatMul_qidx2_params_23.txt` |
| gate/up_proj 的 Output Width | I = 11008 | `mlp_gate_proj_MatMul_qidx397_params_195.txt` |
| down_proj 的 Input Width | I = 11008 | `mlp_down_proj_MatMul_qidx405_params_192.txt` |
| RMSNorm / 残差 Width | H = 4096 | `RMSNorm_params_197.txt`、`add_1_Add_qidx394_params_10.txt` |
| bmm1 Input Width | hd = 128 | `mha_batch_matmul1_head0_qidx30_params_20.txt` |
| bmm1 Output Width、mask/softmax Width、bmm2 Input Width | S = 1024 | 同上 + `mha_masking_head0_...` + softmax 各相 |
| bmm2 Output Width | hd = 128 | `mha_batch_matmul2_head0_qidx45_params_16.txt` |
| `Total Split Head Num` / `Total Split Weight Num` | nh = 32 | 64 个 matmul |
| `Split Head Index` | 0 .. nh−1 | 文件名 `headN` |
| `Eltwise broadcast factor`（RoPE） | nh = 32 | `..._mul_cos_params_201.txt` |
| `Eltwise broadcast Input Stride X`（RoPE） | hd = 128 | 同上 |
| 线性 DQ 的 `Group data size` | G = 128 | `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` |
| 线性 DQ 的 `Output Width`（p1） | H/G = 32 或 I/G = 86 | 节点 24 → 32；节点 193 → 86 |
| 分数 DQ 的 `Group data size` | S = 1024 | 手册 10.5：「group size … up to token size」。token = 本样例槽位 1024 |
| 分数 DQ 的 `Output Width`（p1） | S/S = 1 | `dynamic_quantization_params_17_gp_dq_phase1_params_210.txt` |
| q/k/v/o `weight_buffer` 字节 | H×H×1 = 16777216 | `weight_buffer_23.bin` |
| gate/up/down `weight_buffer` 字节 | I×H×1 = 45088768 | `weight_buffer_195.bin` |
| q/k/v/o `weight_sf` 字节 | H×(H/G)×2 = 262144 | `weight_sf_23.bin` |
| gate/up `weight_sf` 字节 | I×(H/G)×2 = 704512 | `weight_sf_195.bin` |
| down `weight_sf` 字节 | H×(I/G)×2 = 704512 | `weight_sf_192.bin` |
| `Data scale width`（线性 Gemm） | 输入宽 / G | q=4096/128=32；down=11008/128=86 |
| `DDR data scale buffer size` | `Data scale width` × 2 | q=64；分数 DQ 的 1 组 → 2 或 16（见 Q16） |
| `RMSNorm_Add_Const` | eps 的 fp32 | `RMSNorm_Add_Const_197.bin` = 1e-5 |
| softmax / mask 文件套数 | nh = 32 | 32 套 bmm1/mask/sm/bmm2 |
| `Activation Type=13` 出现在 gate | hidden_act=silu | `mlp_gate_proj_..._195.txt` |

手册对生成规则的直接约束（不再含糊的几条）：

| 手册出处 | 约束 | 落到哪个域 / 哪个文件 |
| --- | --- | --- |
| 4.3.4 末句 | 量化按对称动态范围 **max(\|x\|)** | DQ p1 的 `Pooling Type=4`：组内 MaxAbs。见 Q5 |
| 4.3.4 | 全局池化也可算 MAX−MIN（非对称）或 abs(max(x))（对称） | 本样例走对称；`weight_zp` 恒 0 |
| 10.5 | 组大小 16/32/64/96/128，**可到 token 长度** | 线性 G=128；分数 G=S=1024 |
| 10.5 | 非线性 FP16，量化成 int4/int8，线性 MAC 在 int32，scale 为 FP16 | 本样例数值模式。`Data Type` 0=int8、1=fp16、2=int4 权重 |
| 10.1 / 10.3 | int5/6/7 存在 8-bit 容器里；int4 是独立档 | 本样例 int4 也按 1 字节一值落地（`weight_buffer_23.bin` 元素数=字节数，值域 −8..7） |
| 4.2 Softmax | 分子 e^x、分母 Σe^x、归一化 各一层；分母把内存当 **flattened vector** | 压平 Width=S；本样例再拆出减 max 和取倒数，变成 5 层。见 Q41 |
| 4.3.3 | PWL 32 段，slope+intercept；另支持 1/x、e^x | LUT 288 字节；`Activation special operators=4` 对应 1/x |
| 4.3.2 FPSU | 32 位输入：加 bias，乘 16 位 scale，四舍五入，右移，饱和到 16 位 | `Bias_buffer` + `Scaling_buffer` + `Scaling_PS`（右移位数） |
| 4.3.6 Kantor | 加 zp、乘 scale、舍入、右移、饱和；或从指数生成定点 scale 做 fp↔int | DQ p4 / v_proj 的 `Kantor mode=3` |
| 4.2 | CSTL **只做有符号** 运算 | `data extension` 本样例没有 2（unsigned） |
| 7.2.1 / 7.3 | L1 可选 512/768/1024 KB；L2 为 1–32 MB | 768KB 不是 L2。见 Q11 |
| Table 7-12 | L2 内部起点 `0x05000000` | 与样例 `0x1FFF0000` 不是同一套数。见 Q11 |

S=1024 的来源：不是 `max_position_embeddings`。IO_info 里 K/V cache 形状 `[1,32,1024,128]`，是本工具链编译期选定的 decode 槽位数。换槽位时 softmax/mask/bmm 的 Width 和分数 DQ 的 G 一起改。

### 1.5 编译器怎么从模型和手册推出这些文件（可实现的推导）

输入只有三样：HuggingFace 的 `config.json`、手册里的硬件常量、本工具链选定的 decode 槽位 `S`。输出是 GML 节点集合，以及（若直接出 prepare_out）每层一份 txt。

下面按「谁算、算什么、公式是否闭合」写。**闭合** = 代入模型参数就能得到唯一整数，422 层无反例。**查表** = 按层类型取固定值，全层同类相同。**对方填** = 编号表或分配器，见第 15 章。

#### 步骤 A. 图编译器：从前端图得到节点集合和张量几何

```
读 config.json
    H, I, nh, nkv, hd=H/nh, eps, act
读编译期常量
    S          # KV 槽位，本样例 1024
    G_w = 128  # 权重组，手册 3.4 / 10.5
    G_a = 128  # 线性激活动态量化组
    G_score = S  # 分数 DQ 组 = token 长，手册 10.5「up to token size」

对 Llama decoder 一层，展开成下面这张节点表（decode 一步、batch=1、seq=1）：
```

| 顺序 | 节点 | 输入几何 | 输出几何 | dtype 路径 | 拆成几层 |
| --- | --- | --- | --- | --- | --- |
| 1 | RMSNorm | [H] fp16 | [H] fp16 | 全程 fp16 | 1 × vpu |
| 2 | DQ | [H] fp16 | [H] int8，scale[H/G_a] fp16 | 4 相 | 4 |
| 3 | v_proj Gemm | [H] int8 × W[H,H] int4 | [H] int8 写 V cache | Kantor fp2int | 1 |
| 4 | k_proj Gemm | 同 | [H] fp16 | 输出 fp16 | 1 |
| 5 | RoPE_K | [H] fp16 × cos/sin[hd] | [H] int8 写 K cache | 3 个 eltwise | 3 |
| 6 | q_proj Gemm | [H] int8 × W[H,H] int4 | [H] fp16 | 输出 fp16 | 1 |
| 7 | RoPE_Q | [H] fp16 × cos/sin[hd] | [H] fp16 | 3 个 eltwise | 3 |
| 8 | DQ_Q | [H] fp16 | [H] int8 | 4 相 | 4 |
| 9 | 对 h=0..nh−1 | | | | |
| 9a | bmm1 | Q切片[hd] int8 × Kcache[S,hd] int8 | [S] fp16 | Weight Format=3 | 1 |
| 9b | mask | [S] fp16 + mask[S] fp16 | [S] fp16 | Eltwise mode=2 | 1 |
| 9c | softmax | [S] fp16 | [S] fp16 | 5 相 | 5 |
| 9d | DQ_score | [S] fp16 | [S] int8 | G=S，4 相 | 4 |
| 9e | bmm2 | [S] int8 × Vcache[S,hd] int8 | [hd] fp16 | Weight Format=2 | 1 |
| 10 | DQ | 拼回 [H] fp16 | [H] int8 | 4 相 | 4 |
| 11 | o_proj Gemm | [H] int8 × W[H,H] int4 | [H] fp16 | | 1 |
| 12 | residual Add | 层输入[H] fp16 + o_proj[H] fp16 | [H] fp16 | | 1 |
| 13 | RMSNorm | [H] fp16 | [H] fp16 | | 1 |
| 14 | DQ | [H] fp16 | [H] int8 | 4 相 | 4 |
| 15 | gate_proj Gemm+SiLU | [H] int8 × W[I,H] int4 | [I] fp16 | Activation Type=13 | 1 |
| 16 | up_proj Gemm | [H] int8 × W[I,H] int4 | [I] fp16 | | 1 |
| 17 | mlp_mul | [I]×[I] fp16 | [I] fp16 | Kantor mode=5 | 1 |
| 18 | DQ | [I] fp16 | [I] int8 | G=128，组数 I/G=86 | 4 |
| 19 | down_proj Gemm | [I] int8 × W[H,I] int4 | [H] fp16 | | 1 |
| 20 | residual Add | 12 的输出 + down | [H] fp16 | | 1 |

层数（闭合，应等于 422）：

```
RMSNorm         2
DQ 线性（24,12,196,193 四条 ×4）= 16
Q 上 DQ（节点 22）×4 = 4
Gemm 7
RoPE 3+3 = 6
Add 2
mlp_mul 1
循环内每头：bmm1 + mask + sm5 + DQ4 + bmm2 = 12，×32 = 384
合计 2+16+4+7+6+2+1+384 = 422
```

```
elem_bytes(dt) = {int8:1, fp16:2, fp32:4, int4_stored:1}
Maps = 1
Height = 1
Width  = numel
StrideX = Width
终相 Output Stride Z = align16(Width) + 15
    align16(w) = ((w + 15) // 16) * 16
中间相 Output Stride Z = Width
    例外：Width=1 的部分中间相为 16，见 Q52
```

dtype 填写（闭合，本样例）：

```
进 NMU 的激活（Gemm/MatMul 的 Input Data Type）     = 0  # int8
Gemm 默认输出 / RMSNorm / 残差 / softmax 工作值      = 1  # fp16
写 KV cache 的输出（v_proj、RoPE_K add、DQ 终相）    = 0  # int8
Softmax p1/p3 归约标量                               = 3  # fp32
Weights Data Type：有 int4 权重的 Gemm               = 2
其余层 Weights Data Type                             = 0
Input/Output data extension：整数 → 1，浮点 → 3
```

编号 0/1/3/2 的官方全表仍见 Q1。上面是「本样例生成时填这些数」。

缓冲区命名（闭合）：

```
生产者写出的 Dataout 文件名 = 消费者读入的 Datain 文件名
相位 dump：output_buffer_phase_{P-1}_{node_id}.bin
权重：weight_buffer_{LayerID}.bin
scale：weight_sf_{LayerID}.bin
```

#### 步骤 B. 算子编译器：相位模板和内核参数

**B1. DQ 四相（闭合，手册 4.3.4 + 10.5 + 实测 1/256）**

```
输入：x 长度 W，组大小 G（线性 128，分数 S），组数 Gn = W/G   # 本样例都能整除
p1 pooling
    Input Width = W, Output Width = Gn, dt 入出 = fp16
    Pooling Type = 4, Pooling Filter Width = G, Filter Height = 1
    dump[g] = max_{i=0..G-1} |x[g*G+i]|
p2 activation
    Input = Output = Gn, dt = fp16
    Activation mode = 1, LUT = 恒等表（A[0]=1）
    Scaling_buffer = fp16(1/256)
    dump[g] = dump_p1[g] * (1/256)          # = amax[g]/128
p3 activation
    Input = Output = Gn, dt = fp16
    Activation Type = 13, special operators = 4, LUT = 1/x 表
    dump[g] = 1 / dump_p2[g]
p4 activation
    Input = Output = W, 入 fp16 出 int8
    Kantor mode = 3, Kantor A source = 1
    Kantor A scale buffer = p3 dump
    q[i] = sat_int8( x[i] * dump_p3[i//G] )
Task ID = 相位号 - 1
Layer ID：p4 = GML node_id；p1..p3 另发辅号
```

**B2. Softmax 五相（闭合，手册 4.2 三步 + 样例多拆两步）**

```
输入：该头 mask 后的分数，长度 S，fp16
p1 pooling    Output Width=1, dt 出=fp32, Pooling Type=4, Filter Width=S
              dump = max(x)                    # softmax 减 max，数学是 max(x)
p2 activation Input=Output=S, dt=fp16, Activation Type=13, LUT=exp
              Bias buffer 文件名 = p1 的 Dataout
              dump = exp(x - max)
p3 pooling    Input=S 出 1, dt 出=fp32, Pooling Type=3
              dump = Σ dump_p2
p4 activation Input=Output=1, 入 fp32 出 fp16
              Activation Type=13, special operators=4, LUT=1/x
              dump = 1 / Σ
p5 activation Input=Output=S, dt=fp16
              Scaling buffer 文件名 = p4 的 Dataout
              dump = dump_p2 * dump_p4
Prev/Next：p1→p2；p2→p3 且 p2→p5；p3→p4；p4→p5
```

p1 与 DQ p1 的 `Pooling Type` 都是 4，数学不同。生成时按算子分流：DQ→MaxAbs，Softmax→Max。编号能否这样用见 Q5。

**B3. Gemm（闭合的几何 + 查表的内核）**

```
权重形状 W[N, K]，本样例：
    q/k/v/o : N=K=H
    gate/up : N=I, K=H
    down    : N=H, K=I
Input Width  = K
Output Width = N
Weights Data Type = 2
weight_buffer 字节 = N * K * 1
weight_sf 字节     = N * (K / G_w) * 2
Data scale width   = K / G_a          # 输入激活的组数
Quant_source = 1
Datain = 上游 DQ 终相 dump
input scale factor buffer = 上游 DQ 第二相 dump
gate 额外：Activation Type=13，LUT=SiLU，contraction 融进本层
v_proj 额外：Output Data Type=0，Kantor mode=3，DDR Output Height=S、strideZ=H*S
```

**B4. MatMul 32 头（闭合）**

```
对 h in 0..nh-1:
    bmm1:
        Input Width = hd, Output Width = S, Weight Format = 3
        Weights buffer = K cache, Split weight index = h
        Group data size = hd, Data scale width = 1
        Cache idx = 0, Head input = 1
        L2 input size = hd + 16          # 128→144
        L2 output size = (align16(S)+16)*2   # 2080
    bmm2:
        Input Width = S, Output Width = hd, Weight Format = 2
        Weights buffer = V cache, Split weight index = h
        Group data size = S, Data scale width = 1
        Cache idx = 1, Head output = 1
        L2 input size = S + 16           # 1024→1040
        L2 output size = (align16(H)+16)*2   # 按整条 hidden 占位，8224
```

**B5. Eltwise / RMSNorm（闭合）**

```
残差 Add：mode=0，Kantor=0，Width=H，两路 fp16
mlp_mul：mode=1，Kantor=5，Width=I
mask：mode=2，Width=S，第二输入 = 图输入 mask[S]
      Mask Input Index = Mask Data Type = Mask Buffer Index = 1
RoPE：
    Llama2Activation = True
    Original name = reshape 后的 Q 或 K 名
    mul_cos / mul_sin：mode=1，Kantor=5
        broadcast dim=3, factor=nh, input index=1, strideX=hd
        mul_sin 另写 rotary window size = hd/2 = 64
        Q 的 mul_cos/mul_sin：Runtime input 0/1 = 1（cos/sin 运行时装载）
    add：mode=0
        K 的 add：Kantor=3，输出 int8，Dataout = K cache
                  Cache output = true，Num Output Heads = nh
        Q 的 add：Kantor=0，输出 fp16，进入 DQ_Q
    六个文件 force consecutive execution = 1
RMSNorm：layer type=vpu, sublayer=rmsnorm, Vpu Axis=-1
    Bias = fp32(eps)
    Width=H，入出 fp16
```

**B6. LUT（闭合三张，exp 一张拷贝）**

```
288 字节 = 144 个 fp16
[0:32] slope，[32:64] intercept，[64:144] 写 0
恒等：A[0]=1，其余 0                    # DQ p2，Activation mode=1
倒数：切线 1/x，p_i = 1+(i+0.5)/32      # DQ p3、softmax p4
SiLU：切线族                             # gate
exp：本阶段拷贝 LUT_phase_1_18.bin      # softmax p2，见 Q14
```

**B7. 按层类型查表（422 层同类相同，不是几何公式）**

这些域不能从 H/I/S 算出来，生成时按层类型抄：

| 层类型 | `L2 fpsu buffer size` | `Fpsu mode` | `Kantor mode` | `L2 weights buffer size` |
| --- | --- | --- | --- | --- |
| RMSNorm | 512 | （无此域） | （无） | 4096 |
| DQ 任一层 | 1024 | 1 | p4=3，其余 0 | （无） |
| softmax p2/p4 | 512 | p4=2，其余 1 | 0 | （无） |
| softmax 其它相 | 无或 512 | 1 | 0 | （无） |
| q/k/o_proj、down、残差 Add | 28672 | Gemm=2 | 0 | q/k/o=32768；down=30720 |
| v_proj、RoPE（K 的三连、Q 的 mul） | 57344 | Gemm=2 | v_proj/RoPE_K add=3；mul=5 | v_proj=32768 |
| Q 的 RoPE add | 28672 | — | 0 | （无） |
| gate | 77824 | 2 | 0 | 16384 |
| up | 77312 | 2 | 0 | 16384 |
| mlp_mul | 154624 | — | 5 | （无） |
| bmm1、mask | 7168 | bmm1=2 | 0 | bmm1=4096 |
| bmm2 | 1024 | 2 | 0 | 32768 |

`L2 fpsu buffer size` 全是 512 的倍数，但倍数（1/2/14/56/112/151/152/302）对不上单一的 `k×Width×elem_bytes`。生成时用上表，不要临时凑公式。换 hidden/intermediate 之后这张表要重测，见 Q17。

`Data scale buffer size`（不是 width×2）：

| 层 | width | size |
| --- | --- | --- |
| q/k/v/o | 32 | 32 |
| gate/up | 32 | 16 |
| down | 86 | 30 |
| bmm1/bmm2 | 1 | 2 |

按层类型抄。闭合公式未见，见 Q16。

#### 步骤 C. 编排器：Layer ID、Task、L2、net.ini

```
Layer ID
    单层算子、多相的最后一相 = GML node_id
    多相的前几相、RoPE 三连：从 201 起按出现顺序 +1
    文件名最后一个 params_N = 该文件 Layer ID

Task ID / Prev / Next
    相位链：Task ID = 相位号 - 1，Prev/Next 只连本链（见 Q13 的五相表）
    单层：Task ID=0，两个 count=0
    跨算子依赖不写进 Prev/Next，写进 Datain/Dataout 文件名

net.ini [layers]
    按步骤 A 的执行骨架展开：RMSNorm → DQ → v/k/RoPE_K → q/RoPE_Q/DQ_Q
    → for h in 0..nh-1: bmm1, mask, sm×5, DQ×4, bmm2
    → DQ → o_proj → residual → RMSNorm → DQ → gate, up, mul → DQ → down → residual

L2 输入/输出字节（闭合）
    L2 input size  = Width * elem_bytes(Input Data Type)
                    bmm 另加 16：hd→144，S→1040
    L2 output size = (align16(Width) + 16) * elem_bytes(Output Data Type)
                    bmm2 例外：按 H 而不是 hd 占位

L2 offset
    QMAN offset = 0x1FFF0000，size = 65536     # 本样例全层相同，与手册 Table 7-12 不是同一套
    数据区大 offset / 小槽：本阶段没有闭合分配器，见 Q11
    若走对方 L2A：这些域全部不填，由 L2A 写

恒定硬件口（查表）
    Bytes in cycle read/write = 64     # 手册 Table 7-11，NPM4K+
    Number of frames = 1
    Maps = Height = 1
    skip compare = 1                   # 见 Q31
```

#### 步骤 D. 一张「域 → 谁填 → 公式种类」总表

| 域类 | 谁填 | 公式种类 | 出处 |
| --- | --- | --- | --- |
| Width / StrideX / 组数 / 头数 / 权重字节 / sf 字节 | 图 | 闭合，§1.4 | config.json |
| dtype 编号、extension | 图 | 本样例闭合；全表见 Q1 | 手册 10.5 + dump 字节 |
| DQ / Softmax 相位链、LUT、Scaling=1/256、Kantor=3 | 算 | 闭合，步骤 B1/B2 | 手册 4.2 / 4.3.4 + 实测 bin |
| Gemm/MatMul 几何、Weight Format、Cache idx、head 切分 | 图+算 | 闭合，B3/B4 | 模型 + 样例无反例 |
| Eltwise mode、RoPE broadcast、RMSNorm eps | 图+算 | 闭合，B5 | config + 样例 |
| Fpsu mode、L2 fpsu size、L2 weights 切片、Data scale buffer size | 算/编 | 查表 B7 | 422 层同类相同，换形状要重测 |
| Layer ID、Task ID、net.ini 序、L2 输入输出字节 | 编 | 闭合，步骤 C | 422 层无反例 |
| L2 offset、Format 0..7、net.ini 四个 stride | 编 | 对方填 | Q2 / Q10 / Q11 |
| exp LUT 切点 | 算 | 对方填或拷贝 | Q14 |

按 A→B→C 实现，除第 15 章列出的编号表和 L2 分配器外，可以生成与本样例结构一致的 422 个层文件。换一套 hidden/heads/S，闭合部分自动跟着变；查表部分要按新形状重新从样例或仿真标定。

#### 步骤 E. 全量域怎么填（422 层出现过的键，按生成规则分组）

层 txt 里归一化后约 250 个键。带 `0/1` 后缀的是双输入槽，生成时按 `number of inputs` 展开。下面按「闭合公式 / 本样例恒定 / 按层类型 / 对方填」四类列全。未出现在某层的键：该层不写。

**E1. 422 层都有，本样例恒定（生成时原样抄）**

| 域 | 值 | 说明 |
| --- | --- | --- |
| Number of frames | 1 | decode 一步 |
| Input Maps / Output Maps | 1 | 压平 |
| Input Height / Output Height | 1 | 压平 |
| Is Winograd | false | 非卷积 |
| Weight Compression Rate | 1.0 | WDM 未开 |
| Sparsity | 0.0 | |
| Padding Left/Right/Top/Bottom | 0 | |
| Raster mode | 0 | |
| Is Macro Tile | False | |
| Macro Tile ID | 1 | |
| Total number of MT | 1 | |
| Use Clipping | 0 | 381 层有此域，全 0 |
| LeakyReLU Negative Slope | 0 | |
| Input Fraction bits / Output Fraction bits | 0 | |
| After concat / Before concat | false | |
| skip compare | 1 | 见 Q31 |
| Bytes in cycle internal memory read/write | 64 | 手册 Table 7-11 |
| L2 qman buffer offset | 536805376 | 见 Q11 |
| L2 qman buffer size | 65536 | |
| Output data order | 0 | 见 Q39 |
| Input data order | 0 | 381 层有；双输入另有 `Input data order 1=0` |

**E2. 422 层都有，由步骤 A/B/C 算出**

| 域 | 公式 |
| --- | --- |
| Input/Output Width, Stride X, Stride Z | 步骤 A 压平 + 终相对齐 |
| Input/Output Data Type, data extension | 步骤 A dtype |
| Weights Data Type | Gemm 有 int4 权重 → 2，否则 0 |
| Kernel Width/Height | Gemm/MatMul=1，其余 0 |
| Filter Horizontal/Vertical Stride | 有核=1，无核=0 |
| Pooling Type / Filter Width/Height / Stride / Pad | DQ/Softmax pooling 相按 B1/B2；其余 Type=0、窗口=0 |
| Activation Type | 走 LUT 的相=13，其余 0 |
| Quant_source | 使用 DQ scale 的 Gemm/MatMul=1，其余 0 |
| layer type / sublayer type / number of inputs | 步骤 A 节点表 |
| Layer ID / Task ID / Prev/Next task / count | 步骤 C |
| Fpsu source / Weights source | 查表：softmax 链 Fpsu source=1；cache 路径 Weights source=0；其余 3 |
| L2 weights buffers per engine | Gemm/MatMul=2，RMSNorm=1，无权重层=4 |
| L2 fpsu buffer id | 按相/槽抄 f1..f5，见 Q35 |
| Kantor mode | 见 B7 |
| Input Format | 常规 0；DQ p2/p3 内部口 6。完整表见 Q2 |
| Sys virtual input/output、Virtual Input/Output ... is | 中间相 true，落盘相 false；与 Datain/Dataout 是否 DDR 一致 |

**E3. 按是否双输入展开（eltwise / mask / 残差）**

`number of inputs=2` 时，下列键写成 `... 0` 和 `... 1` 两套：

| 域 | 公式 |
| --- | --- |
| Datain file N / Input buffer file N | 第 N 路生产者的 Dataout 名 |
| Residual input buffer N | 第 N 路上游 GML 节点号 |
| Fpsu mode N / Use FPSU N / Pooling data type N | 残差 Add：两路都是 1 / 1 / 2；mask：两路都是无 FPSU（0） |
| L2 fpsu buffer offset N | 两路各一块 |
| Sys virtual input N / Graph Input N | 图上真实边 → Graph Input=1 |
| DDR Input * N | 与该路形状相同；本样例 offset/size=0 |
| Input data order N | 0 |

`Dataout file` 本样例常写两遍（同一文件名出现两次）。生成时按样例双写。

**E4. 量化 / FPSU / Kantor 文件引用（有则写）**

| 域 | 何时写 | 公式 |
| --- | --- | --- |
| Bias buffer file / Scaling buffer file / Scaling PS buffer file | 379 层 | 文件名带 LayerID 或 `phase_{P-1}_{node}` |
| output scale factor buffer | 416 层 | `output_sf_{id}.bin`；线性层常为标量 fp16 1.0 |
| input scale factor buffer | Gemm/MatMul | 上游 DQ **第二相** dump，见 Q44 |
| weights scaling buffer file / Weights buffer file | 73 层有权重 | `weight_sf_{id}.bin` / `weight_buffer_{id}.bin` |
| Activation LUT file / Activation mode / special operators / Flp min/max exp / mantisa | 139 层带 LUT | 见 B6 与 Q22 |
| Scale axis | 379 层 | 1 |
| Group data axis | 108 层 | 3（最后一维） |
| Group data size | 108 层 | 线性 128；分数 DQ / bmm2 = S 或 hd，见 B1/B4 |
| Data scale maps/height/width/stride X/Z / buffer size | 71 层 | maps=height=1；width=组数；strideX=strideZ=width；buffer size 查表 B7 |
| Data scale format | 71 层 | 7，见 Q15 |
| Data scale source | 71 层 | 0 |
| Runtime data scale / Registry data scale | 71 层 | false / true |
| DDR data scale * | 71 层 | Width=组数；size=组数×2（q=64；分数 DQ 的 bmm2=16） |
| L2 data scale buffer id / offset engine 0 | 71 层 | 见 Q35 / Q11 |
| L2 weight scale buffer size / offset | 71 层 | 切片，查表；q=131072，gate/up=176128，down=122880，bmm1=2048，bmm2=256 |
| Kantor A source / scale axis / group axis / Group kantor A size | DQ p4 等 | source=1；axis=2 或 1；group axis=3；size=G（128 或 1024） |
| Kantor A/B scale/bias/shift buffer file | 有 Kantor 的层 | 文件名带语义后缀或 phase |
| Scale per tensor | RoPE 三连等 10 层 | 1 |
| Transpose type | softmax p2=1；倒数相=2 | 见 Q24 |
| Pooling data type | 379 层 | 2，见 Q30 |

**E5. DDR / KV / head 切分（有则写）**

| 域 | 何时写 | 公式 |
| --- | --- | --- |
| DDR Input * / Graph Input | 有 DDR 输入的层 | 形状=压平后的 Width；offset/size 本样例 0 |
| DDR Output * / Graph Output | 有 DDR 输出的层 | 普通层 Height=1；**写 KV** 时 Height=S、strideZ=H×S、Orig Name=`value_cache_out` / K cache |
| DDR Weight * | 64 个 matmul | Width=H=4096，Height=S=1024，strideZ=H×S，size=S×hd=131072；offset 见 Q20（bmm1=2112，bmm2=8208） |
| Weights input / weights from ddr | 64 个 matmul | 1 / true |
| Weight Format | 64 个 matmul | bmm1=3，bmm2=2 |
| Head input | 64 个 matmul | bmm1=1，bmm2=0 |
| Head output | 32 个 bmm2 | 1 |
| Split Head Index | 384 层（循环内） | 0..nh−1，等于文件名 headN |
| Total Split Head Num / Total Split Weight Num | 64 个 matmul | nh=32 |
| Split weight index | 64 个 matmul | = head 序号 |
| Cache input | 64 个 matmul | true |
| Cache idx | 66 层 | bmm1=0（K），bmm2=1（V）；RMSNorm 也有 0/1 |
| Cache output | 仅 K 写出：RoPE_K add | true |
| Num Output Heads | 同上 2 层 | nh=32 |
| Original cache file | v_proj、RoPE_K add | 该 cache 的原 buffer 名 |
| rotary window size | RoPE mul_sin 2 层 | hd/2 = 64 |
| Llama2Activation | RoPE 6 层 | True |
| Original name | RoPE + Q 的 DQ | 回溯到 reshape 后的 Q/K 名 |
| force consecutive execution | RoPE 6 层 | 1（写成独立一行） |
| Mask Input Index / Mask Data Type / Mask Buffer Index | 32 个 mask | 全是 1 |
| Eltwise mode / broadcast * | eltwise | 见 B5 |
| Vpu Axis / sublayer type | RMSNorm | -1 / rmsnorm |
| Residual input/output buffer | 有邻居的层 | GML 节点号 |
| L2 input num of buffers | 381 层 | 单输入 1；eltwise 与 DQ p4 为 2 |
| L2 input buffer offset/size/id / for DMA width/height / slice maps | 有 L2 输入 | size 见步骤 C；DMA width=Input Width，height=1，maps=1，slice offset=0 |
| L2 output buffer offset/size/id | 有落盘输出 | size 见步骤 C；无此域 = virtual 相，见 Q34 |
| L2 weights buffer offset 0/1、size、double/partial、id | 73 层有权重 | 查表 B7；双缓冲两槽差 = size |
| L2 fpsu buffer offset / size | 有 FPSU | size 查表 B7 |
| Output Format | 250 层 | 见 Q2 |
| softmax phase / softmax axis | 160 层 | 1..5 / 3 |
| dynamic quantization phase | 148 层 | 1..4 |
| DDR Input TVM Orig Buffer Name | 图入口 RMSNorm | IO_info 的 node_name，如 `nprm_182_i12` |
| DDR Output TVM Orig Buffer Name | 图出口 add_2 | IO_info 输出名 |
| Runtime input 0/1 | Q 的 RoPE mul 各 1 层 | 1（cos/sin 运行时装载） |

**E6. 写 KV / 写图 I/O 时多出来的几何**

```
v_proj、RoPE_K add 写出：
    Output Data Type = 0
    DDR Output Width = H, Height = S, strideX = H, strideZ = H*S
    Dataout 文件 = input_buffer_200.bin（V）或 input_buffer_199.bin（K），4MB = 32*S*hd*1

图入口 RMSNorm_25：
    Datain = 图输入 hidden [H] fp16
    DDR Input TVM Orig Buffer Name = IO_info 输入 1 的 node_name

图出口 add_2：
    Dataout = 图输出 hidden [H] fp16
    DDR Output TVM Orig Buffer Name = IO_info 输出 0 的 node_name
```

**E7. 生成一层 txt 的固定顺序**

参考工具链按「形状 → 卷积壳 → 量化/Kantor/FPSU → dump 文件名 → layer type → DDR → Task → L2 → Layer ID → 硬件口」往下写。生成时保持这个顺序，避免解析器按行号假设。没有的键跳过，不要填空值掩盖。

---



## 2. 产物文件全景

```
prepare_out/
├── net.ini                          # 全局配置 + 422 层执行序
├── net.ini.orig
└── txt_files/                       # 424 个文件 = 422 层 cfg + 2 个版本戳
    ├── gml_version.txt              # 26.2.1
    ├── l2a_version.txt              # 0.0.0-c45e54f
    └── <layer_name>.txt             # 一层一份，文件名 = net.ini 里的 layer 值
```

`net.ini` 的 `[layers]` 有 422 条，与 422 个层 cfg **一一对应**（版本戳不进执行序）。

### 2.1 422 层按前端算子分类

| 前端算子 | GML op_type | 展开层数 | layer type 序列 | 文件数 |
| --- | --- | --- | --- | --- |
| RMSNorm ×2 | `RMSNorm_vpu` | 1 | `vpu` / `sublayer type=rmsnorm` | 2 |
| 线性层 q/k/v/o/gate/up/down | `Gemm` | 1 | `gemm` | 7 |
| 残差 Add ×2 | `EltwiseAdd` | 1 | `eltwise` mode=0 | 2 |
| MLP 的 SiLU 后乘 | `EltwiseMul` | 1 | `eltwise` mode=1 | 1 |
| RoPE（Q 一套 + K 一套） | `Llama2Activation` | 3 | `eltwise` mul_cos → mul_sin → add | 6 |
| Q 上的 DQ（RoPE 之后） | `Llama2ActivationDQ` | 4 | pooling → act → act → act | 4 |
| 其它 DQ（含 32 头 attn 分数） | `DynamicScaling` | 4 | 同上 | 36×4=144 |
| 32 头 QK^T | `MatMul` | 1 | `matmul`，Weight Format=3 | 32 |
| 32 头因果 mask | `Mask` | 1 | `eltwise` mode=2 | 32 |
| 32 头 Softmax | `Softmax` | 5 | pooling → act → pooling → act → act | 160 |
| 32 头 PV | `MatMul` | 1 | `matmul`，Weight Format=2 | 32 |
| 合计 | 200 个 GML 节点 | — | — | 422 |

GML 里还有 Split / Concat / Transpose / Reshape / KV_Cache_DMA / Lut(Silu 融进 gate) 等节点，**不单独成 prepare_out 层**（融合进邻居或被 L2A 消化）。

### 2.2 文件名规则

```
<前端算子名>_qidx<GML节点号>_params_<主LayerID>[_<gp|act>_<dq|sm>_phase<P>_params_<相位LayerID>].txt
```

| 段 | 含义 | 谁生成 |
| --- | --- | --- |
| 前端算子名 | 与 GML `name` / Relay 名同源，如 `self_attn_q_proj_MatMul` | 图 |
| `qidxN` | GML 节点号。相位文件沿用**所属算子**的节点号，不是相位自己的 Layer ID | 图（GML id） |
| 第一个 `params_X` | 该算子「主层」的 Layer ID。Softmax/DQ 的主层 = 最后一相 | 编 |
| `gp_*_phaseP` / `act_*_phaseP` | `gp` = pooling 相位，`act` = activation 相位；**P 是 1-based**（cfg 里的 `softmax phase` / `dynamic quantization phase`） | 算 |
| 第二个 `params_Y` | **这一相自己的 Layer ID**。最后一相 Y = 主 Layer ID，文件名里两段 params 相同 | 编 |

缓冲区文件名里的 `phase_N` 是 **0-based = cfg 相位号 − 1**。这是最容易写错的地方：

| cfg 相位号 | 文件名 `phase_N` | dump 例子 |
| --- | --- | --- |
| 1 | `phase_0` | `output_buffer_phase_0_18.bin` |
| 2 | `phase_1` | `LUT_phase_1_18.bin` |
| 3 | `phase_2` | `input_buffer_phase_2_18.bin` |
| 4 | `phase_3` | `output_buffer_phase_3_18.bin` |
| 5 | `phase_4` | Softmax 第五相的 Bias 用 `Bias_buffer_phase_4_18.bin` |

cfg 与 dump 文件名是否允许改成同一套编号，见第 15 章 Q12。

### 2.3 Layer ID 发号（实测，422 层零反例）

Layer ID **不等于** `net.ini` 的执行序，也不等于 GML `id`。

观察到两段号：

| 号段 | 谁占用 | 例子 |
| --- | --- | --- |
| 主号 = GML 节点号 | 每个算子「最后一相」或唯一层 | Softmax 第五相 `Layer ID=18`；DQ 第四相 `Layer ID=24`；q_proj `Layer ID=23` |
| 辅号 ≥ 201 | 同一算子的前几相、RoPE 三连 | Softmax phase1–4 为 318–321；RoPE mul_cos/sin/add 为 201–206 |

规则（422 层无反例，生成时按此写；发号是否可当硬规则见 Q56）：

1. 单层算子：`Layer ID = GML node_id`。
2. 多相算子：最后一相沿用 `node_id`；前面的相从 201 起按出现顺序另发。
3. 文件名最后一个 `params_N` **恒等于** 该文件的 `Layer ID`。

### 2.4 执行序 vs Layer ID vs Task ID

| 概念 | 本样例事实 | 谁决定 |
| --- | --- | --- |
| 执行序 | `net.ini` `[layers]` 从上到下 | 编 |
| Layer ID | 层的身份，不随执行序变 | 编 |
| Task ID | **相位链内部的局部序号，不是全局调度 id**。全文件只有 0..4 | 算→编 |

Task ID 分布（422 层穷举）：

| Task ID | 出现位置 |
| --- | --- |
| 0 | 所有单层算子；DQ/Softmax 的 phase1 |
| 1 | DQ/Softmax phase2 |
| 2 | DQ phase3；Softmax phase3 |
| 3 | DQ phase4；Softmax phase4 |
| 4 | Softmax phase5 |

`Prev task` / `Next task` 只在同一条相位链内部有效。跨算子依赖不走这组域，走 `Datain`/`Dataout` 文件名和 `Residual input/output buffer`。

**算法（422 层无反例，生成时按此写）：**

```
若该层属于某条相位链（DQ 4 相或 Softmax 5 相）：
    Task ID          = 该层 cfg 相位号 − 1          # 即 0..3 或 0..4
    Prev task i      = 本链中直接前驱的 Task ID
    Next task i      = 本链中直接后继的 Task ID
    Prev/Next count  = 前驱/后继个数
否则（单层算子：gemm / matmul / eltwise / vpu）：
    Task ID = 0，Prev task count = 0，Next task count = 0
```

head0 Softmax 五相实例（文件在 `txt_files/`）：

| 文件 | `softmax phase` | `Task ID` | `Prev task` | `Next task` | `Layer ID` |
| --- | --- | --- | --- | --- | --- |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | 1 | 0 | （无，count=0） | 1 | 318 |
| `..._act_sm_phase2_params_319.txt` | 2 | 1 | 0 | 2 和 4 | 319 |
| `..._gp_sm_phase3_params_320.txt` | 3 | 2 | 1 | 3 | 320 |
| `..._act_sm_phase4_params_321.txt` | 4 | 3 | 2 | 4 | 321 |
| `..._act_sm_phase5_params_18.txt` | 5 | 4 | 1 和 3 | （无，count=0） | 18 |

读法：p2（Task 1）扇出到 p3 和 p5；p5（Task 4）等 p2 的 e^x 和 p4 的 1/Σ。两条 head 的 Softmax 都从 Task 0 数起，**编号会重复**。q_proj 这种单层算子三个 count 都是 0。

因此「全局唯一 Task ID」和「相位链内部 0-based」是两种不同的编号法。样例用的是后者。对方是否要求保持这种写法，见第 15 章 Q13。

---

## 3. `net.ini` 逐域

### 3.1 `[general]`

| 域 | 本样例值 | 含义（我方理解） | 计算公式 / 取值来源 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- |
| `is_seq_test` | 0 | 是否序列测试模式 | 本样例恒 0。decode 单步不走 seq test | 常 / 运 | 中 |  |
| `seq_tunneling` | 0 | 本样例恒 0 | 本样例恒 0 | 常 | 见 Q32 |  |
| `test_update_buffer` | 0 | 测试时是否回写 buffer | 本样例恒 0 | 常 / 运 | 中 |  |
| `input_line_stride` | 8 | 输入行步长（单位？） | 全网络一个值，与任何 Width 都对不上 | 硬 / 运 | 待确认 | **P0：物理含义与单位** |
| `input_map_stride` | 4 | 输入 map 步长 | 同上，全网恒定 | 硬 / 运 | 待确认 | **P0** |
| `output_line_stride` | 12 | 输出行步长 | 同上 | 硬 / 运 | 待确认 | **P0** |
| `output_map_stride` | 5 | 输出 map 步长 | 同上 | 硬 / 运 | 待确认 | **P0** |
| `dumps_bin_path` | `llama2_w4a8_decode_block_0/parser_output` | 二进制相对路径 | 工程目录约定 | 运 | 已核实 |  |
| `dumps_txt_path` | `.../prepare_out/txt_files` | 层 cfg 相对路径 | 工程目录约定 | 运 | 已核实 |  |
| `seq_output_bin_file` | `/net.bin` | 序列测试输出 | 本样例未用 | 运 | 中 | 见 Q32 |

这四个 stride **不随层变化**，也不等于任何 `Input Stride X`（那些是 128/1024/4096/11008）。更像仿真器/DMA 打包参数或 CDNN 遗留字段。

### 3.2 `[layers]`

| 域 | 含义 | 计算公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| `layer = <stem>` | 按执行序排列的层名，等于 `txt_files/<stem>.txt` 的文件名（无后缀） | 拓扑序 + 相位展开序。本样例：RMSNorm → DQ(4) → v/k/q 投影与 RoPE → 32 头 (bmm1, mask, sm×5, DQ×4, bmm2) → o_proj → residual → RMSNorm → DQ → gate/up → mul → DQ → down → residual | 编 | 已核实 |  |

本样例执行骨架（decode 一层）：

```
RMSNorm_25
  → DQ_24 (p1..p4)          # 输入激活动态量化
  → v_proj, k_proj
  → RoPE_K (mul_cos, mul_sin, add)     # add 输出写 key_cache
  → q_proj
  → RoPE_Q (mul_cos, mul_sin, add)
  → DQ_Q (p1..p4)           # Q 量化后供 32 头 QK^T 共用
  → for head h in 0..31:
        bmm1_h              # QK^T，K 来自 cache，按 head 切片
        mask_h
        softmax_h (p1..p5)
        DQ_score_h (p1..p4) # 分数量化后供 bmm2
        bmm2_h              # PV，V 来自 cache
  → DQ_12 (p1..p4)          # 拼回头后的激活量化
  → o_proj
  → add_1                   # 残差
  → RMSNorm_197
  → DQ_196
  → gate_proj（融 SiLU）, up_proj
  → mlp_mul
  → DQ_193
  → down_proj
  → add_2                   # 残差，图输出
```

---

## 4. 层 cfg 公共域（422 层都会出现的部分）

下面按功能分组。样例值取自 `self_attn_q_proj_MatMul_qidx2_params_23.txt`，除非另注。

### 4.1 形状与 dtype

| 域 | 样例 | 含义 | 公式 | 取值/范围 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Number of frames` | 1 | 帧（子 tile）数 | 本样例不做多帧切分 ⇒ 1 | ≥1；本样例恒 1 | 算 / 常 | 已核实 |  |
| `Input Maps` | 1 | 输入通道数 | 压平约定 ⇒ 1 | ≥1；本样例恒 1 | 图 | 已核实 | 见 Q8 |
| `Input Height` | 1 | 输入高 | 压平约定 ⇒ 1 | 本样例恒 1 | 图 | 已核实 |  |
| `Input Width` | 4096 | 输入平面元素数 | 见 §1.4：线性层 H 或 I；softmax/mask = S；bmm1 = hd；bmm2 = S；DQ 归约相 = W/G | 见 §1.4 | 图 | 已核实 |  |
| `Input Stride X` | 4096 | 行内步长 | `= Input Width`（本样例无 padding 行） | ≥ Width | 图 | 已核实 |  |
| `Output Maps` | 1 | 输出通道数 | 压平 ⇒ 1 | 恒 1 | 图 | 已核实 |  |
| `Output Height` | 1 | 输出高 | 压平 ⇒ 1 | 恒 1 | 图 | 已核实 |  |
| `Output Width` | 4096 | 输出平面元素数 | 见 §1.4。DQ p1：`W/G`（H/128=32，I/128=86，S/S=1，都能整除） | — | 图 | 已核实 |  |
| `Output Stride X` | 4096 | 输出行内步长 | `= Output Width` | — | 图 | 已核实 |  |
| `Output Stride Z` | 4111 | 输出 map 步长（含对齐空洞） | 终相：`align16(Width)+15`。4096→4111，11008→11023，1024→1039，128→143 | 例外见下 | 图→编 | 高 | 例外见 Q52 |
| `Input Data Type` | 0 | 输入元素类型 | 见 §12。本样例 0/1/3 | 0/1/3 | 图 | 已核实（0/1/3） | 完整表见 Q1 |
| `Output Data Type` | 1 | 输出元素类型 | 见 §12 | 0/1/3 | 图 | 已核实 | 见 Q1 |
| `Input data extension` | 1 | 有符号性 | GML：1=signed，2=unsigned，3=float。手册 4.2：CSTL 只做有符号，本样例无 2 | 1 或 3 | 图 | 已核实 |  |
| `Output data extension` | 3 | 有符号性 | 同上 | 1 或 3 | 图 | 已核实 |  |
| `Weights Data Type` | 2 | 权重元素类型 | 有 int4 权重的 gemm = 2；其余填 0 | 0 或 2 | 图 | 已核实 | 见 Q1 |
| `Input data order` | 0 | 输入维序 | 本样例恒 0 | 0 | 图 / 常 | 中 | 见 Q39 |
| `Output data order` | 0 | 输出维序 | 本样例恒 0 | 0 | 图 / 常 | 中 | 见 Q39 |

`Output Stride Z` 的**已观察到的例外**（不能套 `align16(W)+15`）：

| 层 | Width | 实测 SZ | 备注 |
| --- | --- | --- | --- |
| DQ / Softmax 归约输出（Width=1） | 1 | 1 | 标量，不对齐 |
| Softmax phase2 输出 | 1024 | 1024 | 中间相，不对齐 |
| DQ phase2（组数=1 的 head 分数） | 1 | 16 | 16 而不是 31，像按 16 对齐而不是 +15 |
| DQ phase3 | 1 或 32 或 86 | = Width | 中间相，不对齐 |
| reshape DQ phase1 输出 | 32 | 32 | 中间相 |

终相按 `align16(W)+15` 写；相位链内部按 `SZ=W` 写。Width=1 的中间相有时为 16。例外清单见 Q52。

- **最后一相 / 要写回给下一算子的平面**：`SZ = align16(W)+15`
- **相位链内部标量或中间向量**：`SZ = W`，或对极小 Width 用 16

对应 L2 输出字节（高，仍有例外）：

```
elem_bytes(dt) = {0:1, 1:2, 3:4}[Data Type]
L2 output buffer size ≈ (align16(W) + 16) * elem_bytes
```

核对：

| 输出 | dt | W | 预测 | 实测 |
| --- | --- | --- | --- | --- |
| RMSNorm / q_proj / residual | fp16 | 4096 | (4096+16)×2=8224 | 8224 |
| gate/up / mlp_mul | fp16 | 11008 | (11008+16)×2=22048 | 22048 |
| DQ phase4 输出 int8 | int8 | 4096 | (4096+16)×1=4112 | 4112 |
| DQ phase4 输出 int8 | int8 | 1024 | (1024+16)×1=1040 | 1040 |
| mask / softmax 终相 | fp16 | 1024 | (1024+16)×2=2080 | 2080 |

**例外**：`mha_batch_matmul2` 输出 Width=128、dt=fp16，按公式应是 288，实测 `L2 output buffer size=8224`（按整条 4096 fp16 平面分配）。读法：32 头的 PV 输出在 L2 里按 hidden=4096 的整平面占位，每个 head 写其中 128 个元素。是否如此见 Q19。

dtype 与 extension 的配对（422 层无反例）：

| Data Type | extension | 字节/元素 | 含义 |
| --- | --- | --- | --- |
| 0 | 1 | 1 | int8 signed。DQ 终相输出、gemm 输入、v_proj 输出（写 cache）、RoPE_K add 输出 |
| 1 | 3 | 2 | fp16。RMSNorm、残差、softmax 工作值、gemm 输出 |
| 3 | 3 | 4 | fp32。仅 Softmax phase1/3 的归约标量 |

---

## 5. 公共域：卷积壳、宏块、硬件口

这些域对 LLM 线性层多数是「卷积引擎的壳」，填 0 / false。

| 域 | 本样例 | 含义 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- |
| `Is Winograd` | false | Winograd 卷积 | LLM 不用 ⇒ false | 常 | 已核实 |  |
| `Weight Compression Rate` | 1.0 | WDM 压缩比 | 未启用 WDM ⇒ 1.0 | 硬 / 常 | 已核实 | `.bin` 是否保持 1B/元素 |
| `Sparsity` | 0.0 | 非结构化稀疏率 | 不用 ⇒ 0 | 常 | 已核实 |  |
| `Padding Left/Right/Top/Bottom` | 0 | 卷积 pad | 不用 ⇒ 0 | 图 | 已核实 |  |
| `Kernel Width/Height` | gemm/matmul=1；其余=0 | 卷积核；FC 视为 1×1 | 线性层 1；eltwise/pooling/vpu 0 | 图 / 算 | 已核实 |  |
| `Filter Horizontal/Vertical Stride` | 1 或 0 | 卷积步长 | 与 Kernel 同现：有核为 1，无核为 0 | 图 | 已核实 |  |
| `Raster mode` | 0 | 光栅读模式 | 恒 0 | 硬 / 常 | 中 | 其它取值 |
| `Is Macro Tile` | False | 是否宏块切分 | 恒 False | 算 / 常 | 已核实 |  |
| `Macro Tile ID` | 1 | 宏块序号 | 恒 1 | 算 | 中 |  |
| `Total number of MT` | 1 | 宏块总数 | 恒 1 | 算 | 中 |  |
| `Use Clipping` | 0 | 是否 clip | 恒 0 | 算 | 已核实 |  |
| `Bytes in cycle internal memory read` | 64 | 每周期读带宽 | 硬件口宽，恒 64 | 硬 | 已核实 |  |
| `Bytes in cycle internal memory write` | 64 | 每周期写带宽 | 恒 64 | 硬 | 已核实 |  |
| `skip compare` | 1 | 仿真跳过黄金比对 | 恒 1 | 运 / 常 | 中 | 见 Q31 |
| `After concat` / `Before concat` | false | 是否紧挨 concat | 恒 false（concat 被 L2A 消化） | 图 | 已核实 |  |
| `LeakyReLU Negative Slope` | 0 | LeakyReLU 负坡 | 本网络无 LeakyReLU | 算 | 已核实 |  |
| `Input/Output Fraction bits` | 0 | 定点小数位 | 本样例走 fp16/int，恒 0 | 算 | 中 | 定点网络时怎么填 |
| `Activation Type` | 0 或 13 | 激活类别 | 0=直通；13=走 LUT 的非线性（SiLU / exp / 1/x） | 算 | 高 | **完整枚举** |
| `Pooling Type` | 0 / 3 / 4 | 池化 / 归约类别 | 0=不做；3=Sum（Softmax phase3 的 Σe^x）；4=Max 或 MaxAbs（DQ phase1 与 Softmax phase1） | 算 | 高 | **3/4 的官方名；phase1 是 Max 还是 MaxAbs** |
| `Pooling Filter Width/Height` | 0 或归约窗 | 归约窗口 | DQ/SM 的 pooling 相：Width=`Group data size` 或 `Input Width`，Height=1 | 算 | 已核实 |  |
| `Pooling Horizontal/Vertical Stride` | 0 或 1 | 池化步长 | 有 pooling 时为 1 | 算 | 已核实 |  |
| `Pooling Pad *` | 0 | 池化 pad | 恒 0 | 算 | 已核实 |  |

---

## 6. 动态量化 4 相位（线性层的输入来自这里）

手册 §4.3.4 / §4.2：Pooling 块可复用做「按组求动态范围」；Kantor 做 fp→int。GML 把 4 相写在**同一个** `DynamicScaling` 节点的 `*_phase_N` 字段里；L2A 拆成 4 个层。

### 6.1 三条 DQ 实例，宽度不同

| 实例 | 输入 W | G | 组数 Gn=`W/G`（本样例都能整除） | 出现次数 | 下游 |
| --- | --- | --- | --- | --- | --- |
| 残差后 / RMSNorm 后（节点 24、196、12） | 4096 | 128 | 32 | 3 + Q 上那条 | q/k/v/o_proj、gate/up |
| MLP mul 后（节点 193） | 11008 | 128 | 86 | 1 | down_proj |
| 32 头 attn 分数（节点 17,38,…） | 1024 | **1024** | **1** | 32 | 该头的 bmm2 |
| Q 的 RoPE 后（节点 22） | 4096 | 128 | 32 | 1 | 32 头 bmm1 共用 |

组大小不是全局常数。手册 10.5：组大小 16/32/64/96/128，**可到 token 长度**。线性激活沿最后一维 G=128（H/G=32，I/G=86）；注意力分数把整条 S=1024 当成一组（`Group data size=1024`，等于本样例 token/槽位数）。

### 6.2 四相在做什么（与 dump 字节互证）

以节点 24（W=4096, G=128, Gn=32）为例，dump 在 `parser_output/`：

| 相 | cfg 相位号 | 文件名 phase | layer type | 输入 W | 输出 W | 输出 dt | dump 文件 | dump 字节 | 数学 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | pooling | 4096 | 32 | fp16 | `output_buffer_phase_0_24.bin` | 64=32×2 | 每组 max(\|x\|)。手册 4.3.4：量化按对称动态范围 max(\|x\|） |
| 2 | 2 | 1 | activation | 32 | 32 | fp16 | `output_buffer_phase_1_24.bin` | 64 | 由 absmax 生成 scale（÷128 的定点表示）；LUT 是恒等表 |
| 3 | 3 | 2 | activation | 32 | 32 | fp16 | `output_buffer_phase_2_24.bin` | 64 | `1/scale`，LUT 是倒数表 |
| 4 | 4 | 3 | activation | 4096 | 4096 | int8 | `output_buffer_phase_3_24.bin` | 4096 | Kantor fp16→int8：`x / scale[g]` |

节点 103（head 分数，W=1024, G=1024, Gn=1）：

| 相 | 输出元素 | dump 字节 |
| --- | --- | --- |
| 1 | 1×fp16 | 2 |
| 2 | 1×fp16 | 2 |
| 3 | 1×fp16 | 2 |
| 4 | 1024×int8 | 1024 |

`output_sf_24.bin` = 64 字节 = 32 个 fp16，实测范围约 `[0.017, 0.030]`，与「按组 absmax/128」同量级。`output_sf_193.bin` = 172 字节 = 86 个 fp16。

### 6.3 四相关键域

| 域 | p1 pooling | p2 act | p3 act | p4 act | 公式 / 规则 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dynamic quantization phase` | 1 | 2 | 3 | 4 | 1-based | 算 | 已核实 |  |
| `layer type` | pooling | activation | activation | activation | 相位模板 | 算 | 已核实 |  |
| `Pooling Type` | 4 | 0 | 0 | 0 | 4 = 组内 MaxAbs。手册 4.3.4 量化用 max(\|x\|） | 算 | 已核实（手册，DQ） | Softmax 同号见 Q5 |
| `Pooling Filter Width` | G 或 W | 0 | 0 | 0 | 线性 DQ：128；分数 DQ：1024 | 算 | 已核实 |  |
| `Output Width` | Gn | Gn | Gn | W | `Gn = W/G`（H/128=32，I/128=86，S/S=1） | 图 / 算 | 已核实 |  |
| `Output Data Type` | 1 | 1 | 1 | 0 | 前三相 fp16，终相 int8 | 算 | 已核实 |  |
| `Kantor mode` | 0 | 0 | 0 | 3 | 3 = fp2int_converter | 算 | 高 | 见 §8 枚举对齐 |
| `Kantor A source` | — | — | — | 1 | 1 = 取上游 buffer（phase3 的 1/sf） | 算 / 编 | 高 |  |
| `Group kantor A size` | — | — | — | G 或 W | 与 `Group data size` 相同 | 算 | 已核实 |  |
| `Activation Type` | 0 | 0 | 13 | 0 | 13 = 走 LUT（倒数） | 算 | 高 |  |
| `Activation mode` | — | 1 | 0 | — | 1=even（恒等表）；0=regular | 算 | 已核实（GML 文档） |  |
| `Activation LUT file` | — | `LUT_phase_1_<id>.bin` | `LUT_phase_2_<id>.bin` | — | 288 字节；p2=恒等，p3=倒数 | 算 | 已核实 |  |
| `Activation special operators` | — | 0 | 4 | — | 4 与「取倒数」绑定（32+37 层全是 phase3/4 的 1/x） | 算 | 高 | **4 的官方含义** |
| `Flp min/max exp, mantisa` | — | 10/17/3 | 15/15/0 | — | 按 LUT 种类抄三张表 | 算 | 中 | 见 Q22 |
| `Input Format` | 0 | 6 | 6 | 0 | 6 = 相位链内部虚拟口 | 编 | 中 | **Format 完整枚举** |
| `Output Format` | 6 | 7 | 4 | 1 或 0 | 见 §8 | 编 | 待确认 | **P0** |
| `Use FPSU` | 1 | 1 | 1 | 1 | 动态量化走 FPSU | 算 | 已核实 |  |
| `Fpsu mode` | 1 | 1 | 1 | 1 | 见 §8（**不是简单的 fixed/float**） | 算 | 待确认 | **P0** |
| `Task ID` | 0 | 1 | 2 | 3 | 相位号−1 | 编 | 已核实 |  |
| `Layer ID` | 辅号 | 辅号 | 辅号 | = GML id | 见 §2.3 | 编 | 已核实 |  |
| `Datain file` | `input_buffer_<id>.bin` | `output_buffer_phase_0_<id>.bin` | `output_buffer_phase_0_<id>.bin` | 原始输入（再读一遍） | p3 的输入仍是 p1 的 absmax，不是 p2 | 编 | 已核实 |  |
| `Dataout file` | `output_buffer_phase_0_<id>.bin` | `output_buffer_phase_1_<id>.bin` | `output_buffer_phase_2_<id>.bin` | `output_buffer_phase_3_<id>.bin` | 0-based | 编 | 已核实 |  |
| `Virtual Input/Output` | in F, out **T** | in T, out F | in T, out T | in T, out F | 中间相不落 DDR | 编 | 高 |  |
| `Sys virtual input/output` | 同 Virtual | 同 | 同 | 同 | 与 Virtual 同步 | 编 | 高 | 二者是否永远相同 |
| `L2 fpsu buffer size` | 1024 | 1024 | 1024 | 1024 | DQ 全相恒 1024 | 编 / 硬 | 中 | 公式 |

节点 24 上能直接量到的量：

| 文件 | 内容 |
| --- | --- |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `Pooling Type=4`，`Group data size=128`，`Output Width=32` |
| `parser_output/output_buffer_phase_0_24.bin` | 64 字节 = 32×fp16（p1 输出） |
| `parser_output/Scaling_buffer_phase_1_24.bin` | 2 字节，fp16 值 = **0.003906 = 1/256**（p2 的 Scaling） |
| `parser_output/output_buffer_phase_1_24.bin` | 64 字节 = 32×fp16（p2 输出，下游 Gemm 的 `input scale factor buffer`） |
| `parser_output/output_sf_24.bin` | 64 字节，32 个 fp16，范围约 0.017–0.030 |

生成时按下面这条写（与权重 int8 分母 128 一致；若不对，在 Q6 改正文）：

```
对每组 128 个 fp16：
    amax[g]  = max_i |x[g,i]|                         # p1，Pooling Type=4
    scale[g] = amax[g] * (1/256)                      # p2，乘 Scaling_buffer_phase_1
             = amax[g] / 128                          # 与 int8 满量程 128 对齐
    inv[g]   = 1 / scale[g]                           # p3，倒数 LUT
    q[i]     = sat_int8( x[i] * inv[g] )              # p4，Kantor mode=3
```

完整逐步公式、以及分数 DQ 的 G=1024，见第 15 章 Q6。

---


## 7. Softmax 5 相位

手册 4.2 节 Self-Attention：分子 e^x、分母 Σe^x、归一化 e^x/Σe^x 各为独立层。本样例拆成 5 层，32 头各一套。

输入永远是该头 mask 之后的分数，Width = S = 1024（KV 槽位数），dtype = fp16。

### 7.1 五相在做什么（与 dump 字节互证，head0 节点 18）

| 相 | cfg 相位号 | 文件名 phase | layer type | 输入 | 输出 | dump | dump 字节 | 数学 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | pooling | 1024 fp16 | 1 fp32 | output_buffer_phase_0_18.bin | 4 | max(x)，沿 softmax axis=3 |
| 2 | 2 | 1 | activation | 1024 fp16 | 1024 fp16 | input_buffer_phase_2_18.bin | 2048 | e^(x − max)，LUT = exp 表 |
| 3 | 3 | 2 | pooling | 1024 fp16 | 1 fp32 | input_buffer_phase_3_18.bin | 4 | Σ e^(x−max) |
| 4 | 4 | 3 | activation | 1 fp32 | 1 fp16 | output_buffer_phase_3_18.bin | 2 | 1 / Σ |
| 5 | 5 | 4 | activation | 1024 fp16 | 1024 fp16 | input_buffer_17.bin | 2048 | e^(x−max) × (1/Σ) |

手册只写了 3 步（分子 / 分母 / 归一化）。本工具链把「减 max」和「取倒数」也拆成层，所以是 5 相。

手册写 3 步，本样例拆成 5 层。是否允许改成 3 相，见 Q41。

### 7.2 五相关键域

| 域 | p1 | p2 | p3 | p4 | p5 | 规则 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| softmax phase | 1 | 2 | 3 | 4 | 5 | 1-based | 算 | 已核实 |  |
| softmax axis | 3 | 3 | 3 | 3 | 3 | 压平后的最后一维 = seq/slot | 图 | 已核实 |  |
| layer type | pooling | activation | pooling | activation | activation | 相位模板 | 算 | 已核实 |  |
| Pooling Type | 4 | 0 | 3 | 0 | 0 | 3=Sum（32 层全是 softmax p3）；4 在 softmax p1 与 DQ p1 都出现 | 算 | 高 | 4 的 Max / MaxAbs 见 Q5 |
| Pooling Filter Width | 1024 | 0 | 1024 | 0 | 0 | = Input Width = S | 算 | 已核实 |  |
| Input Data Type | 1 | 1 | 1 | 3 | 1 | p4 输入是 fp32 标量 | 算 | 已核实 |  |
| Output Data Type | 3 | 1 | 3 | 1 | 1 | p1/p3 产出 fp32 标量 | 算 | 已核实 |  |
| Activation Type | 0 | 13 | 0 | 13 | 0 | 13 = 走 LUT | 算 | 高 |  |
| Activation LUT file | — | LUT_phase_1_18.bin | — | LUT_phase_3_18.bin | — | p2=exp 表；p4=倒数表（与 DQ p3 同一张） | 算 | 已核实 |  |
| Activation special operators | — | 0 | — | 4 | — | 4 只出现在倒数相 | 算 | 高 | 官方名见 Q23 |
| Flp min/max exp, mantisa | — | 9/16/3 | — | 15/15/0 | — | 按 LUT 种类抄表 | 算 | 中 | 见 Q22 |
| Transpose type | — | 1 | — | 2 | — | 相位链里的 on-the-fly 转置 | 算 | 待确认 | 1 和 2 的官方含义 |
| Output Format | 4 | 无此域 | 无此域 | 4 | 无此域 | 4 与「标量输出」共现 | 编 | 待确认 | 见枚举节 |
| Kantor mode | 0 | 0 | 0 | 0 | 0 | Softmax 不走 Kantor | 算 | 已核实 |  |
| Fpsu mode | 1 | 1 | 1 | 2 | 1 | 仅 p4 为 2 | 算 | 待确认 | 见枚举节 |
| Fpsu source | 1 | 1 | 1 | 1 | 1 | 1 = 内部生成（无 DDR 常量表） | 编 | 中 | 完整枚举 |
| Task ID | 0 | 1 | 2 | 3 | 4 | 相位号减 1 | 编 | 已核实 |  |
| Layer ID | 辅号 | 辅号 | 辅号 | 辅号 | = GML id | 最后一相沿用节点号 | 编 | 已核实 |  |
| Datain file | input_buffer_18.bin（mask 输出） | 同 p1 的原始分数 | input_buffer_phase_2_18.bin | input_buffer_phase_3_18.bin | input_buffer_phase_2_18.bin | p5 再读 p2 的 e^x，不读 p4 | 编 | 已核实 |  |
| Dataout file | output_buffer_phase_0_18.bin | input_buffer_phase_2_18.bin | input_buffer_phase_3_18.bin | output_buffer_phase_3_18.bin | input_buffer_17.bin | p5 写出给下游 DQ | 编 | 已核实 |  |
| Bias buffer file（p2 特例） | 普通 Bias | p1 的 max 输出 | 普通 | 普通 | 普通 | p2 的 Bias 文件名等于 p1 的 Dataout | 算 / 编 | 高 | 见 Q42 |
| Scaling buffer file（p5 特例） | 普通 | 普通 | 普通 | 普通 | p4 的 1/Σ 输出 | p5 的 Scaling 文件名等于 p4 的 Dataout | 算 / 编 | 高 | 见 Q43 |
| Prev / Next task | next=1 | prev=0, next=2 和 4 | prev=1, next=3 | prev=2, next=4 | prev=1 和 3 | p2 扇出到 p3 和 p5；p5 等 p2 和 p4 | 编 | 已核实 |  |
| L2 fpsu buffer size | 无此域 | 512 | 无此域 | 512 | 无此域 | 有 LUT 的相为 512，比 DQ 的 1024 小 | 编 / 硬 | 中 | 公式 |
| Split Head Index | 0..31 | 同 | 同 | 同 | 同 | 等于文件名 headN | 图 | 已核实 |  |

p2 的 Output Stride Z = 1024（等于 Width，不对齐）。p5 的 Output Stride Z = 1039 = align16(1024)+15（终相要对齐）。内部相不对齐、终相对齐，与 DQ 同一条规则。


## 8. GEMM / MatMul 线性层

7 个 Gemm（q/k/v/o/gate/up/down）+ 64 个 MatMul（32 头 QK^T + 32 头 PV）。

共同点：都走 NMU，输入是 DQ 终相的 int8，权重 int4（Gemm）或另一份激活/cache（MatMul）。

### 8.1 七条 Gemm 对照（Llama2-7B 对得上）

| 层 | Input W | Output W | 权重逻辑形状 | weight_buffer 字节 | weight_sf 字节 | Activation Type | 输出 dt | Kantor | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_proj | 4096 | 4096 | 4096×4096 | 16777216 | 262144=4096×32×2 | 0 | fp16 | 0 | 输入来自 DQ_24 |
| k_proj | 4096 | 4096 | 4096×4096 | 16777216 | 262144 | 0 | fp16 | 0 | 同输入 |
| v_proj | 4096 | 4096 | 4096×4096 | 16777216 | 262144 | 0 | **int8** | **3** | 直接写 value cache，fp→int |
| o_proj | 4096 | 4096 | 4096×4096 | 16777216 | 262144 | 0 | fp16 | 0 | 输入来自 DQ_12 |
| gate_proj | 4096 | 11008 | 11008×4096 | 45088768 | 704512=11008×32×2 | **13 SiLU** | fp16 | 0 | LUT = silu 表，融进主算子 |
| up_proj | 4096 | 11008 | 11008×4096 | 45088768 | 704512 | 0 | fp16 | 0 | 与 gate 同输入 |
| down_proj | 11008 | 4096 | 4096×11008 | 45088768 | 704512=4096×86×2 | 0 | fp16 | 0 | 输入来自 DQ_193，组数 86 |

权重公式（已核实，字节数对得上）：

```
逻辑 int4，一字节存一个值（符号扩展，值域 -8..7）
weight_buffer 字节 = N * K * 1
沿 K 维每 G=128 一组：
  组数/行 = K / 128
  weight_sf 字节 = N * (K/128) * 2     # fp16
q/k/v/o: K=N=4096 → sf = 4096*32*2 = 262144
gate/up: N=11008, K=4096 → sf = 11008*32*2 = 704512
down:    N=4096, K=11008 → sf = 4096*86*2 = 704512
```

本样例 `Weight Compression Rate=1.0`，`weight_buffer_23.bin` 字节数 = 元素数。交付形态见 Q45。

### 8.2 Gemm 专有域

| 域 | q_proj 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| layer type | gemm | 权重是编译期常量 → gemm；权重是另一激活/cache → matmul | 图 | 已核实 |  |
| number of inputs | 1 | 权重不占 input 槽 | 图 | 已核实 |  |
| Quant_source | 1 | 1 = 取动态量化 scale（上游 DQ phase2 的输出） | 图 | 已核实 | 0/1 官方名 |
| Group data axis | 3 | 最后一维 | 图 | 已核实 |  |
| Group data size | 128 | = G | 图 | 已核实 |  |
| Data scale source | 0 | 0 = registry / 上游 DQ 产物，不是本层现场算 | 图 | 中 |  |
| Data scale format | 7 | 本样例恒 7；scale 是 fp16 | 硬 / 算 | 待确认 | **7 的官方含义** |
| Data scale width | 32 | = K / G。down_proj = 11008/128 = 86；bmm 见下 | 图 | 已核实 |  |
| Data scale buffer size | 32 | 引擎本地段。q/k/v/o=32；gate/up=16；down=30。**不是**简单的 width×2 | 编 | 待确认 | 单位是字节还是元素？为何 gate 是 16 不是 64 |
| DDR data scale Width | 32 | 与 Data scale width 相同（线性层） | 图 | 已核实 |  |
| DDR data scale buffer size | 64 | = 组数 × 2B。4096/128=32 → 64；分数 DQ 的 1 组 → 16 | 编 | 已核实 |  |
| DDR data scale Orig Buffer Name | dynamic_quantization_params_24_dequant_buff | 上游 DQ 的 dequant 缓冲名 | 图 / 编 | 已核实 |  |
| Runtime data scale | false | 本样例全走 registry | 图 | 已核实 |  |
| Registry data scale | true | 与 Runtime 互斥 | 图 | 已核实 |  |
| Scale axis | 1 | 输出通道轴 | 算 | 中 |  |
| Kantor mode | 0（v_proj=3） | 0=off；3=fp2int（写 cache） | 算 | 高 |  |
| Fpsu mode | 2 | 输出 fp16 的线性层为 2；见枚举节 | 算 | 待确认 |  |
| Use FPSU | 1 | 线性层都用 FPSU 做 dequant+rescale | 算 | 已核实 |  |
| Weights source | 3 | 3 = 编译期常量。MatMul 的 cache 路径为 0 | 编 | 中 | 完整枚举 |
| Fpsu source | 3 | 3 = registry | 编 | 中 |  |
| Datain file | output_buffer_phase_3_24.bin | 上游 DQ 终相 | 编 | 已核实 |  |
| Dataout file | input_buffer_0_22.bin | 按消费者命名（GML 规则：buffer 跟消费者走） | 图 | 已核实 |  |
| Weights buffer file | weight_buffer_23.bin | weight_buffer_<LayerID>.bin | 图 | 已核实 |  |
| weights scaling buffer file | weight_sf_23.bin | 见上面字节公式 | 图 | 已核实 |  |
| input scale factor buffer | output_buffer_phase_1_24.bin | 文件名指向上游 DQ 第二相输出 | 编 | 高 | 该文件是 scale 还是 1/scale，见 Q44 |
| Activation LUT file | 仅 gate：activation_lut_file_195.bin | 288B，SiLU 表。手册：融合进 contraction，不是独立层 | 算 | 已核实 |  |
| L2 weights buffer size | 32768 | 双缓冲切片，不是全权重。见 L2 节 | 编 / 算 | 中 | 切片公式 |
| L2 weights use double buffer | 1 | 线性层恒 1 | 编 | 已核实 |  |
| L2 weights use partial buffer | 1 | 线性层恒 1 | 编 | 已核实 |  |
| L2 weights buffers per engine | 2 | 双缓冲 → 2 | 编 / 硬 | 已核实 |  |
| L2 weight scale buffer size | 131072 | q/k/v/o=131072；gate/up=176128；down=122880。与 N×(K/128)×2 的全量 sf **不相等**，是切片 | 编 | 待确认 | 公式 |
| L2 fpsu buffer size | 28672 | q/k/o/down=28672；v=57344；up=77312；gate=77824（多了 SiLU LUT） | 编 / 硬 | 待确认 | 公式 |
| L2 input buffer size 0 | 4096 | = Input Width × 1B（int8） | 编 | 已核实 |  |
| L2 output buffer size | 8224 | (align16(W)+16)×2B，fp16 | 编 | 已核实 |  |

v_proj 与其它 Gemm 的三个关键差异（写 KV cache）：

1. Output Data Type = 0（int8），Output data extension = 1
2. Kantor mode = 3（fp2int），带 Kantor A scale/bias/shift 三个 bin
3. DDR Output 保留多维：Width=4096, Height=1024, strideZ=4194304，Orig Buffer Name = value_cache_out

k_proj 的 RoPE add 才写 key_cache（input_buffer_199.bin，4MB）。v_proj 直接写 value_cache（input_buffer_200.bin，4MB）。


### 8.3 32 头 MatMul（bmm1 = QK^T，bmm2 = PV）

layer type = matmul。第二个操作数走权重通路（GML: MatMul_input_as_weight = 1），所以 number of inputs 仍是 1。

| 项 | bmm1 QK^T | bmm2 PV |
| --- | --- | --- |
| 文件 | mha_batch_matmul1_headH_... | mha_batch_matmul2_headH_... |
| 出现次数 | 32 | 32 |
| Input Width（Q 或 attn） | 128 = head_dim | 1024 = S |
| Output Width | 1024 = S | 128 = head_dim |
| Output Stride Z | 1039 | 143 |
| Weight Format | 3 | 2 |
| Group data size | 128 | 1024 |
| Data scale width / buffer size | 1 / 2 | 1 / 2 |
| Weights buffer file | input_buffer_199.bin（K cache） | input_buffer_200.bin（V cache） |
| Datain file | output_buffer_phase_3_22.bin（Q 的 DQ 终相，32 头共用） | 该头 Softmax 之后 DQ 的终相 |
| Weights source | 0（从 DDR 动态取） | 0 |
| weights from ddr | true | true |
| Split weight index / Total | 0..31 / 32 | 0..31 / 32 |
| Head input / Head output | 1 / 无此域 | 无此域 / 1 |
| Cache input / Cache idx | true / 0 | true / 1 |
| DDR Weight Width × Height | 4096 × 1024 | 4096 × 1024 |
| DDR Weight stride Z | 4194304 = 4096×1024 | 同 |
| DDR Weight buffer size | 131072 | 131072 |
| DDR Weight buffer offset | 2112 | 8208 |
| L2 weights buffer size | 4096 | 32768 |
| L2 input buffer size 0 | 144 | 1040 |
| L2 output buffer size | 2080 | **8224** |
| L2 fpsu buffer size | 7168 | 1024 |
| L2 weight scale buffer size | 2048 | 256 |

计算核对：

```
bmm1:
  输入 Q 切片 int8，head_dim=128
  L2 input = 144 = 128 + 16     # 16 字节对齐空洞，不是 128
  输出 attn fp16，S=1024
  L2 output = 2080 = (1024+16)×2
  L2 weights = 4096             # 一个 head 的 K 切片：1024 slot × 4? 或 128×32?
  DDR Weight size = 131072 = 1024×128×1   # 一个 head 的 K，int8，S×head_dim

bmm2:
  输入 attn int8，S=1024
  L2 input = 1040 = 1024 + 16
  输出 128 fp16，按公式 L2 应 = (128+16)×2 = 288
  实测 L2 output = 8224 = (4096+16)×2
  → 按整条 hidden=4096 的 fp16 平面占位，head 只是其中 128 元素
  DDR Weight size = 131072 = 1024×128     # 一个 head 的 V
```

这四项分别见 Q7、Q19、Q20、Q46。

### 8.4 MatMul 相对 Gemm 多出来的域

| 域 | bmm1 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| Weight Format | 3 | 见上，2 或 3 | 算 | 待确认 | 见 Q7 |
| Weights input | 1 | 第二个操作数当权重 | 图 | 已核实 |  |
| weights from ddr | true | 与 Weights source=0 同义 | 编 | 已核实 |  |
| Head input | 1（仅 bmm1） | 输入按 head 切 | 图 | 已核实 |  |
| Head output | 1（仅 bmm2） | 输出按 head 切 | 图 | 已核实 |  |
| Split Head Index | 0..31 | = 文件名 headN | 图 | 已核实 |  |
| Total Split Head Num | 32 | = num_attention_heads | 图 | 已核实 |  |
| Split weight index | 0..31 | 与 head 对齐 | 编 | 已核实 |  |
| Total Split Weight Num | 32 | 同总头数（本样例无 GQA） | 图 | 已核实 | GQA 时是否 = num_kv_heads |
| Cache input | true | 权重来自 KV cache | 图 | 已核实 |  |
| Cache idx | 0 或 1 | 样例：bmm1=0，bmm2=1 | 编 | 高 | 见 Q46 |
| DDR Weight * | 见上表 | cache 的多维视图，不压平 | 图 / 运 | 已核实 |  |
| L2 weights buffer offset 0/1 | 2112 / 6208 | 双缓冲两个槽。bmm2 为 8256 / 41024 | 编 | 中 | 公式 |

GQA 注意：Llama2-7B 是 MHA（kv heads=32）。若换成 GQA，Total Split Weight Num 和 cache 切片都要改。本样例没有证据。


## 9. Eltwise：残差、RoPE、mask、MLP 乘法

全部 layer type = eltwise，number of inputs = 2。用 Eltwise mode 区分运算。

### 9.1 Eltwise mode（422 层穷举，无反例）

| mode | 运算 | 出现 | 文件 |
| --- | --- | --- | --- |
| 0 | Add | add_1、add_2、RoPE 的 add ×2 | 4 |
| 1 | Mul | mlp_mul、RoPE mul_cos ×2、RoPE mul_sin ×2 | 5 |
| 2 | MaskAdd（加因果 mask） | mha_masking ×32 | 32 |

`Eltwise mode=2` 的官方名见 Q47。

### 9.2 残差 Add（add_1 / add_2）

| 域 | add_1 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| Input/Output Width | 4096 | = H | 图 | 已核实 |  |
| Input/Output Data Type | 1 / 1 | 残差在 fp16 | 图 | 已核实 |  |
| Kantor mode | 0 | 纯加，不转换 | 算 | 已核实 |  |
| Eltwise mode | 0 | Add | 算 | 已核实 |  |
| Use FPSU 0 / 1 | 1 / 1 | 两个输入各一套 FPSU | 算 | 已核实 |  |
| Fpsu mode 0 / 1 | 1 / 1 | 双槽 | 算 | 待确认 | 编码见枚举节 |
| Datain file 0 / 1 | input_buffer_25.bin（层输入） / input_buffer_1_10.bin（o_proj 输出） | 两个残差源 | 图 / 编 | 已核实 |  |
| Dataout file | input_buffer_197.bin | 下一 RMSNorm 的输入 | 编 | 已核实 |  |
| L2 input buffer size 0 | 8192 | 4096×2B | 编 | 已核实 |  |
| L2 output buffer size | 8224 | (4096+16)×2 | 编 | 已核实 |  |
| L2 fpsu buffer size | 28672 | 与 q_proj 同类 | 编 | 中 |  |
| L2 fpsu buffer offset 0 / 1 | 两个槽 | 双输入各一块 FPSU 参数 | 编 | 已核实 |  |
| Graph Input 0 / 1 | 1 / 1 | 两个输入都是图上的真实边 | 图 | 已核实 |  |

add_2 同结构，输入是 down_proj 与 add_1 的输出，Dataout = 图输出。

### 9.3 RoPE（Llama2Activation 拆成 3 层）

GML 里是一个 Llama2Activation 节点。L2A 拆成 mul_cos、mul_sin、add。Q 一套（节点 22）+ K 一套（节点 30）。

数学：

```
x1, x2 = split(x)                         # 每 head 的 128 维劈成两半
out = concat(x1*cos - x2*sin, x1*sin + x2*cos)
本工具链实现成：
  t_cos = x * cos                         # eltwise mul + broadcast
  t_sin = x * sin                         # 含旋转后的符号，LUT/Kantor 里
  out   = t_cos + t_sin                   # eltwise add
```

sin 支路的半头对换和符号写在哪，见 Q29。

| 域 | mul_cos | mul_sin | add（K） | add（Q） | 规则 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama2Activation | True | True | True | True | 三连都打这个标记 | 图 | 已核实 |  |
| Eltwise mode | 1 | 1 | 0 | 0 | mul / add | 算 | 已核实 |  |
| Kantor mode | 5 | 5 | 3 | 0 | 5=逐元素乘（定点）；K 的 add 要 fp2int 写 cache；Q 的 add 保持 fp16 给后续 DQ | 算 | 高 | mode 5 对应 GML 的哪一档 |
| Eltwise broadcast dim | 3 | 3 | 无 | 无 | 沿最后一维播 | 图 | 已核实 |  |
| Eltwise broadcast factor | 32 | 32 | 无 | 无 | = num_heads。cos/sin 表 (1,1,1,128) 播到 32 头 | 图 | 已核实 |  |
| Eltwise broadcast input index | 1 | 1 | 无 | 无 | 第二个输入是广播源 | 图 | 已核实 |  |
| Eltwise broadcast Input Stride X | 128 | 128 | 无 | 无 | = head_dim | 图 | 已核实 |  |
| Output Data Type | 1 | 1 | 0 | 1 | 仅 K 的 add 产出 int8 cache | 图 / 算 | 已核实 |  |
| Dataout file | ..._cos.bin | ..._sin.bin | input_buffer_199.bin（K cache，4MB） | input_buffer_phase_0_22.bin（给 Q 的 DQ） | K 写 cache；Q 进 DQ | 编 | 已核实 |  |
| force consecutive execution | 1 | 1 | 1 | 1 | 仅这 6 个 RoPE 层为 1 | 编 | 高 | 见 Q27 |
| L2 fpsu buffer size | 57344 | 57344 | 57344 | 57344 | 比残差 Add 的 28672 大一倍（broadcast + Kantor） | 编 | 中 | 公式 |
| Original name | self_attn_Reshape_... | 同 | 同 | 同 | 回溯到 reshape 后的 Q/K | 图 | 已核实 |  |

部分文件该行写成 `force consecutive execution: 1skip compare: 1`（与下一域粘在一起）。生成时拆成两行。解析器是否容忍粘连，见 Q27。

### 9.4 因果 mask（32 头）

| 域 | 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| Eltwise mode | 2 | MaskAdd | 算 | 已核实 |  |
| Input Width | 1024 | = S | 图 | 已核实 |  |
| Kantor mode | 0 | 不加转换 | 算 | 已核实 |  |
| Use FPSU 0 / 1 | 0 / 0 | mask 不做 FPSU rescale | 算 | 已核实 |  |
| Mask Input Index | 1 | 第二个输入是 mask | 图 | 已核实 |  |
| Mask Data Type | 1 | mask 是 fp16 | 图 | 已核实 |  |
| Mask Buffer Index | 1 | 与 Input Index 相同 | 编 | 中 | 是否永远等于 Mask Input Index |
| Datain file 0 | input_buffer_0_19.bin | bmm1 输出 | 编 | 已核实 |  |
| Datain file 1 | input_buffer_1_190.bin | 全局 mask，IO_info 里 shape [1,1,1,1024]，dtype fp16 | 运 | 已核实 | 32 头是否共享同一份 mask |
| Dataout file | input_buffer_18.bin | 该头 Softmax 的输入 | 编 | 已核实 |  |
| Split Head Index | 0..31 | = head | 图 | 已核实 |  |
| L2 fpsu buffer size | 7168 | 与 bmm1 相同 | 编 | 中 |  |
| L2 input / output | 2048 / 2080 | 1024×2 与 (1024+16)×2 | 编 | 已核实 |  |

IO_info 输入 6：shape [1,1,1,1024]，`mask: True`。mask 数值约定见 Q28。

### 9.5 mlp_mul（gate 经 SiLU 之后 × up）

| 域 | 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| Eltwise mode | 1 | Mul | 算 | 已核实 |  |
| Input/Output Width | 11008 | = I | 图 | 已核实 |  |
| Kantor mode | 5 | 逐元素乘 | 算 | 高 |  |
| Kantor A/B scale axis | 1 / 1 | 双输入各一套 Kantor 系数 | 算 | 已核实 |  |
| Datain 0 / 1 | gate 输出 / up 输出 | input_buffer_0_194.bin / input_buffer_1_194.bin | 编 | 已核实 |  |
| Dataout | input_buffer_193.bin | 下游 DQ_193 | 编 | 已核实 |  |
| L2 fpsu buffer size | 154624 | 本样例最大。11008×fp16 双输入 + Kantor | 编 | 待确认 | 公式 |
| L2 input / output | 22016 / 22048 | 11008×2 与 (11008+16)×2 | 编 | 已核实 |  |


## 10. RMSNorm（VPU）

手册：RMSNormalization 走 VPU + CSTL，不走 NMU。本样例 2 层（attn 前、MLP 前），都不拆相位。

| 域 | RMSNorm_197 样例 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| layer type | vpu | 走向量单元 | 算 | 已核实 |  |
| sublayer type | rmsnorm | 仅这 2 层出现 | 算 | 已核实 | 其它 sublayer 取值 |
| Input/Output Width | 4096 | = H | 图 | 已核实 |  |
| Input/Output Data Type | 1 / 1 | fp16 进 fp16 出 | 图 | 已核实 |  |
| Input/Output data extension | 3 / 3 | float | 图 | 已核实 |  |
| Weights Data Type | 0 | 权重是 fp16 的 gamma，不走 int4 通路 | 图 | 已核实 |  |
| Kernel Width/Height | 0 | 非卷积 | 图 | 已核实 |  |
| Use FPSU | 0 | 不用 FPSU | 算 | 已核实 |  |
| Vpu Axis | -1 | 沿最后一维做 RMS | 图 | 已核实 | -1 是否永远等于最后一维 |
| Weights buffer file | weight_buffer_197.bin | 4096 字节。按 fp16 解是 2048 个数，对不上 4096；按 int8/raw 是 4096 字节。**布局待确认** | 图 / 算 | 待确认 | gamma 是 fp16 还是 fp32？为何 4096B 而不是 8192B |
| weights scaling buffer file | weight_sf_197.bin | 4 字节，按 fp32 解 ≈ 0.007874 = 1/127 | 算 | 待确认 | 这个 scale 的用途 |
| bias buffer file | RMSNorm_Add_Const_197.bin | 4 字节，按 fp32 解 = 1e-5 = rms_norm_eps | 图 | 已核实 |  |
| Datain file | input_buffer_197.bin | 8192B = 4096×fp16 | 编 | 已核实 |  |
| Dataout file | input_buffer_196.bin | 下游 DQ | 编 | 已核实 |  |
| L2 input buffer size 0 | 8192 | 4096×2 | 编 | 已核实 |  |
| L2 output buffer size | 8224 | (4096+16)×2 | 编 | 已核实 |  |
| L2 weights buffer size | 4096 | 与 weight_buffer 同大 | 编 | 已核实 |  |
| L2 weights use double/partial buffer | 0 / 0 | 小权重一次装完 | 编 | 已核实 |  |
| L2 weights buffers per engine | 1 | 非双缓冲 | 编 | 已核实 |  |
| L2 fpsu buffer size | 512 | Use FPSU=0 仍分配 512 | 编 | 中 | 见 Q17 |
| Cache idx | 0 或 1 | 两层各一个 | 编 | 中 | RMSNorm 为何有 Cache idx |

手册写 Normalization 是 3 阶段。本样例 `RMSNorm_params_197.txt` 仍是 1 个 `layer type: vpu`。是否对外 1 层、对内 3 阶段，见 Q36。


## 11. 编排器域：L2、DDR、Task、邻居

这些域 GML 里没有，是 L2A 填的。若对方要求直接出 prepare_out，需要单独做编排器。边界见 Q9。

### 11.1 恒定硬件口

| 域 | 全 422 层值 | 含义 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| L2 qman buffer offset | 536805376 = 0x1FFF0000 | 样例全层同一值 | 硬 | 已核实（样例） | 与手册 Table 7-12 对不上，见 Q11 |
| L2 qman buffer size | 65536 | 64KB，顶到 0x20000000 | 硬 | 已核实 |  |
| Bytes in cycle internal memory read/write | 64 / 64 | 数据口宽 | 硬 | 已核实 |  |

大 offset 最小值 0x1FF9F1C0，QMAN 在 0x1FFF0000。手册 Table 7-12 的 L2 起点是 0x05000000（32MB）。768KB 是 L1 选项，不是 L2。样例地址与手册内部地址表的关系见 Q11。

### 11.2 L2 输入 / 输出

| 域 | 公式（已核对的部分） | 例外 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| L2 input num of buffers | 单输入=1；eltwise=2；DQ p4=2（再读原始输入 + scale） | 中间相可能没有这组域 | 编 | 高 |  |
| L2 input buffer size 0 | int8: Width；fp16: Width×2。bmm 另加 16 字节对齐：128→144，1024→1040 | RoPE 有一条 L2in=256，不是 8192 | 编 | 高 | 256 那条的含义 |
| L2 input for DMA width 0 | = Input Width | — | 编 | 已核实 |  |
| L2 input for DMA height 0 | = 1 | — | 编 | 已核实 |  |
| L2 input num of maps 0 | 1 | — | 编 | 已核实 |  |
| L2 input slice maps offset/num | 0 / 1 | 无 map 切分 | 编 | 已核实 |  |
| L2 output buffer size | 终相平面：(align16(W)+16)×elem_bytes | bmm2 输出 128 fp16 却分配 8224；部分中间相无此域 | 编 | 中 | 见 8.3 问 2 |
| L2 output buffer offset | 有输出段的层才有。小 offset（0, 2112, 8256, ...）或大 offset（0x1FFxxxxx） | 308 个相位层里只有一部分有 | 编 | 待确认 | 没有 offset 的相是 virtual 还是原地复用 |
| L2 input buffer offset 0 | 同上，小槽或大窗口 | 出现 252 次 | 编 | 待确认 |  |

小 offset 集合（16 个值）：0, 320, 2112, 4160, 6208, 8208, 8256, 8320, 8448, 16384, 22080, 36928, 38464, 38976, 41024, 44032。像双缓冲固定槽，不是线性堆。分配算法见 Q11。

### 11.3 L2 权重 / FPSU / scale

| 域 | 观察 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- |
| L2 weights buffer size | Gemm 是切片不是全量：q=32768（全量 16MB）；RMSNorm=4096（全量） | 编 / 算 | 中 | 切片公式（与 tiling 的关系） |
| L2 weights buffer offset 0 / 1 | 双缓冲两个槽。q_proj: 8256 与 41024，差 32768 = size | 编 | 高 |  |
| L2 weights use double buffer | Gemm/MatMul=1；RMSNorm=0；eltwise 无此域 | 编 | 已核实 |  |
| L2 weights use partial buffer | 同上，Gemm=1 | 编 | 已核实 |  |
| L2 weights buffers per engine | Gemm/MatMul=2；RMSNorm=1；其它层填 4（像默认） | 编 / 硬 | 中 | 非权重层为什么填 4 |
| L2 fpsu buffer size | 见各算子表，无唯一公式 | 编 / 硬 | 待确认 | 见 Q17 |
| L2 fpsu buffer offset | 大窗口内。eltwise 有 offset 0 和 1 两个槽 | 编 | 中 |  |
| L2 weight scale buffer size | 切片，小于全量 sf | 编 | 待确认 | 公式 |
| L2 data scale buffer offset engine 0 | 仅 Quant_source=1 的层有 | 编 | 已核实 |  |
| L2 * buffer id | 4/5/6 或 w1/w2/f1/f2/f3 | 编 | 待确认 | id 编码规则 |

### 11.4 DDR 视图

DDR * 描述「这一层在外部内存里的逻辑窗口」，offset/size 在本样例几乎全是 0（运行时再绑）。

| 域 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- |
| DDR Input Maps/Width/Height | 与 Input Maps/Width/Height 相同（压平后） | 图 | 已核实 |  |
| DDR Input stride X / Z | = Width / Width（压平） | 图 | 已核实 |  |
| DDR Input start Col/Row/Map | 0 | 编 | 已核实 |  |
| DDR Input Orig Buffer Name | bufferN 或语义名（value_cache_out） | 图 / 运 | 中 | 见 Q48 |
| DDR Input buffer offset / size | 本样例 0 | 运 | 已核实 | 运行时谁写 |
| Graph Input 0 | 1 = 该输入是图的真实入口或上游输出 | 图 | 已核实 |  |
| DDR Output * | 同构。KV 写出时 Height=1024、strideZ=4194304 | 图 | 已核实 |  |
| Graph Output | 1 = 该输出被下游或图出口消费 | 图 | 已核实 |  |
| DDR Weight * | 仅 matmul 的 cache 路径。见 8.3 | 图 / 运 | 已核实 |  |
| DDR data scale * | 仅 Quant_source=1 的层。Width=组数，size=组数×2 | 图 / 编 | 已核实 |  |

### 11.5 Task 与邻居

| 域 | 公式 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- |
| Task ID | 相位链内 0-based。单层算子恒 0 | 编 | 已核实 | 能否改成全局唯一 |
| Prev task count / Next task count | 链内前驱/后继个数 | 编 | 已核实 |  |
| Prev task i / Next task i | 链内 Task ID | 编 | 已核实 |  |
| Residual input buffer | GML 的 residual_input_buffer = 上游节点号 | 图 | 已核实 |  |
| Residual output buffer | 下游节点号 | 图 | 已核实 |  |
| Layer ID | 见 2.3 | 编 | 已核实 | 发号规则是否可当硬规则 |
| Virtual Input/Output for ... is | true = 不落 DDR 的相位口 | 编 | 高 |  |
| Sys virtual input/output | 本样例与 Virtual 同步 | 编 | 中 | 能否不同 |

### 11.6 Dump 文件命名

| 模式 | 谁用 | 规则 |
| --- | --- | --- |
| input_buffer_<id>.bin | 层输入 | id 常等于消费者 Layer ID 或 GML 节点号 |
| input_buffer_<slot>_<id>.bin | 多输入的第 slot 路 | slot=0/1 |
| output_buffer_phase_<p>_<id>.bin | 相位中间 | p 是 0-based |
| input_buffer_phase_<p>_<id>.bin | 相位中间（有时用 input_ 前缀） | 同 |
| weight_buffer_<id>.bin | Gemm / RMSNorm 权重 | id = Layer ID |
| weight_sf / weight_zp | 权重量化参数 | zp 恒 4 字节（本样例 0） |
| LUT_phase_<p>_<id>.bin / activation_lut_file_<id>.bin | PWL | 288 字节 |
| Scaling_buffer(_phase_*) / Scaling_PS / Bias_buffer | FPSU 参数 | PS 恒 1 字节 |
| Kantor_A/B_* | Kantor 的 scale/bias/shift | 有语义后缀（Llama2Activation_Cos 等） |

GML 规则：缓冲区按消费者编号，不按生产者。prepare_out 的 Dataout file 经常等于下游的 Datain file。是否硬规则见 Q49。


## 12. 枚举字典（本样例出现过的值；完整表见第 15 章）

本样例只出现下面这些取值。未出现的档位我方不知道，不能猜。

### 12.1 已核实（有手册或字节互证）

| 枚举 | 本样例取值 | 我方理解 | 证据 | 甲方确认（请补未出现的档） |
| --- | --- | --- | --- | --- |
| Input / Output Data Type | 0, 1, 3 | 0=int8（1B）；1=fp16（2B）；3=fp32（4B） | dump 字节 / 元素数，422 层无反例 | 有无 int4 / int16 / bf16 / fp8？编号是什么 |
| Input / Output data extension | 1, 3 | 1=signed；2=unsigned（未出现）；3=float | VBU-GML 文档原文 |  |
| Weights Data Type | 0, 2 | 2=int4（值域 -8..7，1B/元素）；0=本层无 int4 权重 | weight_buffer 值域 + 字节数 | 1 是否 = int8 权重 |
| layer type | vpu, gemm, matmul, pooling, activation, eltwise | 硬件引擎类别，不是前端 op 名 | 422 层穷举 | 还有 conv 等吗 |
| sublayer type | rmsnorm | 仅 vpu 层 | 2 层 | 还有哪些 |
| Eltwise mode | 0, 1, 2 | 0=Add；1=Mul；2=MaskAdd | 与算子一一对应 | 2 的官方名 |
| Pooling Type | 0, 3, 4 | 0=不做；3=Sum（32 层全是 softmax p3）；4=Max 或 MaxAbs（69 层 = 37 DQ p1 + 32 softmax p1） | 计数与语义对得上 | **4 在 DQ 是 MaxAbs、在 softmax 是 Max？还是都是 MaxAbs** |
| Activation Type | 0, 13 | 0=直通；13=走 LUT 的非线性 | 所有带 LUT 的层都是 13（exp / 1/x / SiLU） | 13 的官方名；其它激活编号 |
| Activation mode | 0, 1 | GML：0=regular，1=even，2=odd。恒等表配 1 | GML 文档 + 恒等 LUT |  |
| softmax phase | 1..5 | 1-based | 160 层 |  |
| dynamic quantization phase | 1..4 | 1-based | 148 层 |  |
| number of inputs | 1 或 2 | eltwise=2，其余=1 | 穷举 |  |

### 12.2 Kantor mode（GML 是字符串，prepare_out 是数字）

VBU-GML 文档给 6 档（文档按 1 开始列）：

1. off
2. elementwise_mul_fp16
3. float_elt_wise_and_scale
4. fp2int_converter
5. elementwise_mul_fixed_point
6. scalar

本样例 prepare_out 只出现 0 / 3 / 5：

| prepare_out 值 | 出现 | 出现位置 | 我方对应到 GML | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- |
| 0 | 376 | 大多数层 | off | 高 |  |
| 3 | 39 | DQ p4 的 37 层 + v_proj + RoPE_K add | fp2int_converter | 高（都在 fp16→int8 的点上） | 请书面确认 3=fp2int |
| 5 | 5 | mlp_mul + RoPE 四个 mul | elementwise_mul_fp16 或 elementwise_mul_fixed_point | 中 | **5 对应哪一档** |

GML 参考图里 kantor_mode 字符串只出现 off / fp2int_converter / elementwise_mul_fp16。0/3/5 与字符串的对应见 Q4。

Kantor A source：本样例 0 或 1。仅 DQ p4 为 1，其 `Kantor A scale buffer file` 指向 phase3 输出。对应关系见 Q50。

### 12.3 Fpsu mode（旧文档可能写反了，请以本表为准）

旧分析写「1=fixed_point，2=floating_point」。这与本样例**对不上**：

| 层 | 输出 dtype | Fpsu mode | 若旧说成立则矛盾点 |
| --- | --- | --- | --- |
| q_proj 等 Gemm | fp16 | 2 | 按旧说 2=float，还说得通 |
| DQ 四相 | 前三 fp16、终相 int8 | 全是 1 | 终相是 int8，1 像 fixed；但前三相是 fp16 也是 1 |
| Softmax p1/p2/p3/p5 | fp16/fp32 | 1 | 浮点输出却是 1 |
| Softmax p4 | fp16 标量（1/Σ） | 2 | 同样浮点，却是 2 |
| RMSNorm | fp16 | 无此域 | Use FPSU=0 |

GML 里 fpsu_mode 是字符串：floating_point_32（71 次）、floating_point（6）、fixed_point（2）。prepare_out 压成了 1/2，与「1=fixed、2=float」对不上。对应关系见 Q3。

Pooling data type 恒为 2（379 层）。与 Input Data Type 的 0/1/3 不同源。GML 对应 pooling_dtype = floating_point / fixed_point。2 的含义见 Q30。

### 12.4 Format / source 类（待确认优先）

| 枚举 | 本样例取值 | 共现规律 | 对应问题 |
| --- | --- | --- | --- |
| Input Format | 0, 6 | 6 只出现在 DQ p2/p3（相位链内部虚拟口） | 0/6 含义 |
| Output Format | 0, 1, 2, 3, 4, 5, 6, 7 | 6=DQ p1 出口；7=DQ p2 出口；4=标量（softmax p1/p4、DQ p3）；1=写回 int8（DQ p4、多数 eltwise）；2=仅 v_proj；5=两条 eltwise；0=Q 的 DQ p4；3=一条 eltwise | **P0 完整表** |
| Quant_source | 0, 1 | 1 = 该层使用动态 scale（Gemm/部分 MatMul）；0 = 离线或不使用 | 见 Q51 |
| Data scale source | 0 | 仅 71 层出现 |  |
| Data scale format | 7 | 仅 71 层，scale 是 fp16 | 7=? |
| Weights source | 0, 3 | 0=DDR 动态（cache）；3=编译期常量 | 完整枚举 |
| Fpsu source | 1, 3 | 1=内部（softmax）；3=registry | 完整枚举 |
| Weight Format | 2, 3 | 仅 64 个 matmul。3=bmm1，2=bmm2 | **P0** |
| Transpose type | 1, 2 | 仅 101 层：softmax p2=1，softmax p4 与 DQ p3=2 | 1/2 含义 |
| Activation special operators | 0, 4 | 4 只出现在「取倒数」相（DQ p3、softmax p4） | 4 的官方名 |
| Raster mode | 0 | 恒 0 | 其它取值 |
| Input / Output data order | 0 | 恒 0 | 其它取值 |


## 13. LUT 与 FPSU 常量表

手册 4.3.3：Activation 用 32 段 PWL，每段 slope + intercept，y = A[i]*x + B[i]。

### 13.1 288 字节布局（已与 139 个文件对拍）

```
144 个 fp16 = 288 字节
[0:32]    slope A[i]        第 32 项恒 0，实际用 31 段
[32:64]   intercept B[i]    第 32 项恒 0
[64:144]  填充 / 残留        恒等表这里全 0；exp/SiLU/倒数表这里有非参数残留
```

全图 139 张表只有 4 种内容：

| 变体 | 张数 | 用途 | 我方能否自造 |
| --- | --- | --- | --- |
| 恒等：A[0]=1，其余 0 | 37 | DQ p2（真正的 /256 在 Scaling_buffer，LUT 只是通路） | 能，字节可对上 |
| 倒数 1/x 切线族 | 69 | DQ p3 + softmax p4 | 能合成，精度优于参考产物 |
| exp | 32 | softmax p2 | 段点规则未反推出，目前只能拷贝参考表 |
| SiLU | 1 | gate_proj 融合 | 可按切线族合成 |

后 80 项能否写 0 见 Q26；exp 段规则见 Q14；交付是否要求字节级一致见 Q54。

### 13.2 FPSU 小表

| 文件 | 本样例字节 | 实测值 | 含义 | 责任方 | 置信度 | 甲方确认 |
| --- | --- | --- | --- | --- | --- | --- |
| Scaling_buffer_file_23.bin | 2 | fp16 = 0.5 | 线性层一条全局 FPSU scale | 算 | 中 | 见 Q55 |
| Scaling_PS_buffer_file_23.bin | 1 | 0x00 | 恒 1 字节 | 算 | 待确认 | 见 Q33 |
| Bias_buffer_file_23.bin | 4 | 0 | 线性层 bias 本样例为 0 | 图 | 已核实 |  |
| Scaling_buffer_phase_0_*.bin | 2 | fp16 = 1.0 | DQ p1 直通 | 算 | 高 |  |
| Scaling_buffer_phase_1_*.bin | 2 | fp16 = 1/256 = 0.003906 | DQ p2 的乘数 | 算 | 高 | 与 absmax 的组合见 Q6 |
| Bias_buffer_phase_0_*.bin | 4 | 非 0 的极小 fp 值 | DQ p1 偏置 | 算 | 待确认 | 见 Q53 |
| output_sf_<id>.bin | 2 或 64 或 172 | 1 个或 Gn 个 fp16 | 该层输出 scale。线性层常为标量 1.0（输出已是 fp16）；DQ 终相是 per-group scale | 图 / 算 | 高 |  |
| weight_zp_*.bin | 4 | 0 | 对称量化 | 图 | 已核实 |  |

Flp min exp / max exp / mantisa 与 LUT 覆盖域绑定：

| 场景 | min | max | mantisa | Activation mode |
| --- | --- | --- | --- | --- |
| DQ p2 恒等 | 10 | 17 | 3 | 1 |
| softmax p2 exp | 9 | 16 | 3 | 0 |
| 取倒数（DQ p3、softmax p4） | 15 | 15 | 0 | 0 |
| gate SiLU | 10 | 17 | 3 | 0 |

三元组编码规则见 Q22。


## 14. 确认之后，我方编译器要补什么

按责任方拆。当前图编译器已经能出一部分 GML；prepare_out 层文本还没有出口。

### 14.1 图编译器（新原语 / 新 pass）

| 项 | 为何需要 | 对应域 / 文件 |
| --- | --- | --- |
| 注意力按 head 展开 | 参考产物 32 套 bmm1/mask/softmax/bmm2，不是一个 batched matmul | Split Head Index、文件个数、执行序 |
| 动态量化作为图节点 | 每个要进 NMU 的 fp16 张量前面插 DQ；Q 的 RoPE 后再插一条 | DynamicScaling 节点、4 个相位层 |
| W4A8 量化标注 | 权重 per-group G=128；激活 DQ per-group 或 per-tensor | Group data size/axis、weight_sf 字节、Quant_source |
| 融合：SiLU 折进 gate | GML contraction；prepare_out 仍是一层 gemm，Activation Type=13 | gate_proj 的 LUT |
| 融合：RoPE 作为 Llama2Activation | GML 一个节点；prepare_out 三层 eltwise | Llama2Activation=True、broadcast |
| 形状压平 | Maps=1, Height=1, Width=numel | 几乎所有形状域 |
| KV cache 多维视图 | 压平只作用于 Input Width；DDR Weight/Output 保留 [32,1024,128] | DDR Weight *、v_proj/RoPE_K 写出 |
| 因果 mask 作为图输入 | IO_info 第 6 号输入 | mha_masking 的第二输入 |
| 缓冲区按消费者命名 | Dataout 名 = 下游 Datain 名 | Dump 文件名 |
| 残差边 | Residual input/output buffer = GML 节点号 | 邻居域 |
| net.ini [layers] 拓扑序 | 执行序不是 Layer ID | net.ini |

暂不做（方案阶段约束）：序列变长、自动切分、异步 dispatch。KV 槽位 S=1024 当编译期常量。

### 14.2 算子编译器（相位模板 + 内核参数）

| 项 | 产出 |
| --- | --- |
| DQ 4 相模板 | pooling(MaxAbs) → act(恒等 LUT + ×1/256) → act(倒数 LUT) → act(Kantor fp2int) |
| Softmax 5 相模板 | pooling(Max) → act(exp LUT, bias=max) → pooling(Sum) → act(倒数 LUT) → act(×1/Σ) |
| RMSNorm 不拆相 | 一个 vpu 层，eps 进 Add_Const，gamma 进 weight_buffer |
| PWL LUT 合成 | 恒等 / 倒数 / SiLU 可自造；exp 等甲方给段规则或允许拷贝 |
| FPSU / Kantor 小表 | Scaling、Bias、Shift、PS |
| Flp min/max/mantisa | 按算子查表，等甲方给编码 |
| 线性层 tiling 切片尺寸 | L2 weights buffer size、双缓冲、partial buffer（目前公式未闭合） |
| Weight Format 2/3 | QK^T 转置 vs PV 不转 |

相位在 GML 里是同一节点的 phase_N 字段族，不是 4/5 个节点。拆层是 L2A 的事。若我方直接出 prepare_out，算子编译器要同时给出「GML 字段族」和「拆开后每层 cfg」。

### 14.3 编排器（L2A 同职，当前参考工具链里不是我方）

| 项 | 产出 |
| --- | --- |
| 相位实例化 | 200 节点 → 422 层，发 Layer ID |
| Task 图 | 链内 Task ID、Prev/Next |
| L2 分配 | offset/size/id，双缓冲槽 |
| virtual 口 | 中间相不落 DDR |
| 执行序 | net.ini [layers] |
| force consecutive | 仅 RoPE 三连 |
| 文件写出 | 每层一份 txt + net.ini |

交付停在 GML 还是连 prepare_out 一起出，见 Q9。

### 14.4 运行时

| 项 | 产出 |
| --- | --- |
| DDR 实地址 | 本样例 offset/size 为 0，运行时绑定 |
| KV cache 物化 | input_buffer_199/200.bin，4MB，int8 |
| cos / sin / mask 物化 | broadcast 源、mask 输入 |
| dumps 路径 | net.ini [general] |

### 14.5 硬件能力库（查表，不随模型变）

Bytes in cycle=64、QMAN 64KB、L2 窗口、对齐 16、LUT 32 段、G 最小 16。换 NPM 型号只改这张表。


## 15. 请甲方按条确认的问题清单

路径约定（下面只写相对路径）：

- 层参数：`prepare_out/txt_files/<文件名>`
- 全局清单：`prepare_out/net.ini`
- 二进制：`parser_output/<文件名>`

每一问请在「甲方填写」栏写：同意 / 改正文 / 官方枚举。空着 = 我方不能生成该域。

---

### 15.1 P0（不填就不能正确生成该域）

#### Q1. `Input Data Type` / `Output Data Type` 编号表

| 项 | 内容 |
| --- | --- |
| 样例文件 | `txt_files/self_attn_q_proj_MatMul_qidx2_params_23.txt` |
| 域 | `Input Data Type` = **0**；`Output Data Type` = **1** |
| 对照文件 | `txt_files/mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` 的 `Output Data Type` = **3** |
| 我方当前算法 | 看该层元素的存储宽度：1 字节→填 0，2 字节→填 1，4 字节→填 3。已用 dump 字节对过：`output_buffer_phase_3_24.bin` = 4096B / 4096 元素 → 0=int8；`input_buffer_197.bin` = 8192B / 4096 元素 → 1=fp16；`output_buffer_phase_0_18.bin` = 4B / 1 元素 → 3=fp32 |
| 请确认 | 0 / 1 / 3 是否就是 int8 / fp16 / fp32？还有没有 int4、int16、bf16、fp8？编号分别是多少？ |
| 甲方填写 |  |

同一套编号还出现在全部 422 层的 `Input Data Type`、`Output Data Type`。

#### Q2. `Input Format` / `Output Format` 编号表

| 项 | 内容 |
| --- | --- |
| 样例文件 | 见下表四行 |
| 我方当前算法 | 没有闭合公式。只能按共现规律猜，未出现的 3/5 不会填 |
| 请确认 | 0..7 每一档的官方名。我方生成时按「层类别 + 相位」查表 |
| 甲方填写 |  |

| 文件 | 域 | 值 | 我方猜测 |
| --- | --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `Input Format` | 0 | 常规平面 |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `Output Format` | 6 | DQ 第一相出口（虚拟） |
| `dynamic_quantization_params_24_act_dq_phase2_params_214.txt` | `Input Format`=6，`Output Format`=7 | 相位内部口 |
| `dynamic_quantization_params_24_act_dq_phase3_params_215.txt` | `Output Format` | 4 | 标量/中间向量 |
| `dynamic_quantization_params_24_act_dq_phase4_params_24.txt` | `Output Format` | 1 | 写回 int8 |
| `self_attn_v_proj_MatMul_qidx38_params_36.txt` | `Output Format` | 2 | 写 KV cache |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | `Output Format` | 4 | 标量 max |

#### Q3. `Fpsu mode` 的 1 和 2 分别是什么

样例对不上「1=定点、2=浮点」这种读法。

| 文件 | 域 | 值 | 该层输出 dtype |
| --- | --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `Fpsu mode` | **2** | fp16（Output Data Type=1） |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `Fpsu mode` | **1** | fp16 |
| `dynamic_quantization_params_24_act_dq_phase4_params_24.txt` | `Fpsu mode` | **1** | int8 |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | `Fpsu mode` | **1** | fp32 标量 |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase4_params_321.txt` | `Fpsu mode` | **2** | fp16 标量（1/Σ） |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无。GML 里是字符串 `floating_point_32` / `floating_point` / `fixed_point`，prepare_out 压成 1/2，映射未知 |
| 请确认 | 1 对应哪条 GML 字符串？2 对应哪条？还有没有别的编号？ |
| 甲方填写 |  |

#### Q4. `Kantor mode` 的 0 / 3 / 5 对应 GML 哪一档

GML 文档六档：off、elementwise_mul_fp16、float_elt_wise_and_scale、fp2int_converter、elementwise_mul_fixed_point、scalar。

| 文件 | 域 | 值 | 该层在做什么 |
| --- | --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `Kantor mode` | **0** | 普通 Gemm，输出 fp16 |
| `self_attn_v_proj_MatMul_qidx38_params_36.txt` | `Kantor mode` | **3** | 输出 int8 写 value cache |
| `dynamic_quantization_params_24_act_dq_phase4_params_24.txt` | `Kantor mode` | **3** | fp16→int8 |
| `self_attn_Reshape_1_qidx18_params_30_mul_cos_params_201.txt` | `Kantor mode` | **5** | RoPE 逐元素乘 cos |
| `mlp_mul_Mul_qidx403_params_194.txt` | `Kantor mode` | **5** | gate×up |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 0→off；输出要变成 int8 时填 3；eltwise 乘法填 5。3 当作 fp2int_converter。5 对应「fp16 乘」还是「定点乘」不确定 |
| 请确认 | 0=?  3=?  5=?  （请写 GML 档名） |
| 甲方填写 |  |

#### Q5. `Pooling Type` 的 3 和 4

手册 4.3.4：量化按对称动态范围 **max(\|x\|)**；全局池化也可算 MAX−MIN 或 abs(max(x))。手册 4.2 Softmax 分母是 Σ，分子前要减 max(x)。

| 文件 | 域 | 值 | 窗口 | 数学（手册 + 该层用途） |
| --- | --- | --- | --- | --- |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `Pooling Type` | **4** | `Pooling Filter Width`=128，`Group data size`=128 | DQ：对称动态范围 → **max(\|x\|)**。生成 p1 dump 按此写 |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | `Pooling Type` | **4** | `Pooling Filter Width`=1024 | Softmax 减 max：数学上是 **max(x)**，不是 max(\|x\|） |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase3_params_320.txt` | `Pooling Type` | **3** | 1024 | Softmax 分母 → **Sum** Σe^(x−max) |

生成时：DQ 的 4 按 MaxAbs 写（手册已写清）；Softmax 的 4 按 Max(x) 写。编号相同、数学不同。

| 项 | 内容 |
| --- | --- |
| 请确认 | 3 的官方名是否就是 Sum。4 在 DQ 和 Softmax 是否同一硬件模式？若是，靠哪个域区分 Max 与 MaxAbs（例如 `Output Data Type`、`softmax phase`）？ |
| 甲方填写 |  |

#### Q6. 动态量化四相：每个文件写出什么

以节点 24 为例（RMSNorm 之后、q/k/v_proj 之前）。四个层文件和四个 dump：

| 相 | 层文件 | 关键域 | dump | dump 字节 |
| --- | --- | --- | --- | --- |
| 1 | `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `layer type=pooling`，`Pooling Type=4`，`Group data size=128`，`Input Width=4096`，`Output Width=32`，`Dataout file=output_buffer_phase_0_24.bin` | `output_buffer_phase_0_24.bin` | 64 = 32×fp16 |
| 2 | `dynamic_quantization_params_24_act_dq_phase2_params_214.txt` | `layer type=activation`，`Activation mode=1`，`Scaling buffer file=Scaling_buffer_phase_1_24.bin`，`Dataout file=output_buffer_phase_1_24.bin` | `Scaling_buffer_phase_1_24.bin` = fp16 **1/256**；`output_buffer_phase_1_24.bin` | 2 + 64 |
| 3 | `dynamic_quantization_params_24_act_dq_phase3_params_215.txt` | `Activation Type=13`，`Activation special operators=4`，`Dataout file=output_buffer_phase_2_24.bin` | `output_buffer_phase_2_24.bin` | 64 = 32×fp16 |
| 4 | `dynamic_quantization_params_24_act_dq_phase4_params_24.txt` | `Kantor mode=3`，`Kantor A scale buffer file=output_buffer_phase_2_24.bin`，`Output Data Type=0`，`Dataout file=output_buffer_phase_3_24.bin` | `output_buffer_phase_3_24.bin` | 4096 = 4096×int8 |

下游 `self_attn_q_proj_MatMul_qidx2_params_23.txt` 的 `input scale factor buffer` = `output_buffer_phase_1_24.bin`（第二相输出，不是第三相）。

生成这四个文件时按下面公式写。p1 用手册 4.3.4 的对称动态范围 max(\|x\|）；p2 的乘数来自 `Scaling_buffer_phase_1_24.bin` 实测 fp16=1/256，合起来 scale=amax/128（int8 满量程，手册 10.3 量化到 int8）。`output_sf_24.bin` 的 32 个 fp16 约 0.017–0.030，量级符合 amax/128。

```
输入 x：4096 个 fp16，沿最后一维每 G=128 一组，组数 Gn=32

p1  output_buffer_phase_0_24.bin[g]  = max_i |x[g*128 + i]|          # 32 个 fp16
p2  output_buffer_phase_1_24.bin[g]  = p1[g] * (1/256)               # = amax[g]/128
p3  output_buffer_phase_2_24.bin[g]  = 1 / p2[g]
p4  output_buffer_phase_3_24.bin[i]  = sat_int8( x[i] * p3[i//128] )
```

分数那条组大小不同，同一套公式，只换 G。手册 10.5：「group size … up to token size」。本样例 token/槽位 S=1024，所以分数 DQ 的 G=1024、组数=1。

| 文件 | `Group data size` | `Input Width` | `Output Width` |
| --- | --- | --- | --- |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | 128 | 4096 | 32 |
| `dynamic_quantization_params_17_gp_dq_phase1_params_210.txt` | 1024 | 1024 | 1 |

| 项 | 内容 |
| --- | --- |
| 请确认 | 上面四行公式是否就是这四个文件的生成规则。若 p1 不是 max\|x\|，或 p2 不是乘 1/256，请直接改公式 |
| 甲方填写 |  |

#### Q7. `Weight Format` 2 和 3

| 文件 | 域 | 值 | 该层 |
| --- | --- | --- | --- |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | `Weight Format` | **3** | QK^T，权重来自 K cache（`Weights buffer file` = `input_buffer_199.bin`） |
| `mha_batch_matmul2_head0_qidx45_params_16.txt` | `Weight Format` | **2** | PV，权重来自 V cache（`Weights buffer file` = `input_buffer_200.bin`） |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 3 = K 按 head 转置后与 Q 点积；2 = V 不转置。64 个 matmul 无反例。Gemm 没有这个域 |
| 请确认 | 2、3 的官方名。还有没有 0/1？Gemm 为何不写这个域？ |
| 甲方填写 |  |

#### Q8. 形状是否按「单 map 平面」写

| 文件 | 域 | 值 |
| --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `Input Maps`=1，`Input Height`=1，`Input Width`=4096，`Input Stride X`=4096 | 逻辑张量是 hidden=4096 的向量，不是 1×4096 图像 |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase5_params_18.txt` | `Input Width`=1024 | 逻辑是 [1,1,1024] 的注意力分数 |
| 例外：`self_attn_v_proj_MatMul_qidx38_params_36.txt` | `DDR Output Width`=4096，`DDR Output Height`=1024，`DDR Output stride Z`=4194304 | 只有 DDR 侧保留 KV 的 [32,1024,128] 视图 |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 层 cfg 的 Input/Output Width = numel；Maps=Height=1。KV 只在 DDR Weight/Output 保留多维 |
| 请确认 | L2A / 底层编译器是否接受这种压平？官方 CDNN 前端对 LLM 是否也这样写？ |
| 甲方填写 |  |

#### Q9. 我方交付停在哪一步

| 项 | 内容 |
| --- | --- |
| 涉及文件 | 整份 `prepare_out/`（`net.ini` + 422 个 txt）以及上游 `relay2gml_graph.gml` |
| 背景 | `gml_version.txt` = 26.2.1；`l2a_version.txt` = 0.0.0-c45e54f。GML 没有 Task ID / Layer ID / L2 offset，这些是 L2A 填的 |
| 请确认 | A) 我方只出 GML，prepare_out 仍由你们 L2A 生成；B) 我方连 prepare_out 一起出。选 B 则第 11 节全部 L2/Task 域都要我方实现 |
| 甲方填写 |  |

#### Q10. `net.ini` 四个 stride 的含义和单位

| 文件 | 域 | 值 |
| --- | --- | --- |
| `prepare_out/net.ini` `[general]` | `input_line_stride` | **8** |
| 同 | `input_map_stride` | **4** |
| 同 | `output_line_stride` | **12** |
| 同 | `output_map_stride` | **5** |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无。这四个值全网络恒定，不等于任何层的 `Input Stride X`（那些是 128/1024/4096/11008） |
| 请确认 | 物理含义、单位（字节？元素？寄存器编号？）。生成时是否原样抄这四个数即可？ |
| 甲方填写 |  |

#### Q11. L2 地址：手册一张表，样例另一套数

手册（ArchSpec 7.5 Table 7-12，内部 27 位地址）：

| 空间 | 起点 | 大小 |
| --- | --- | --- |
| L2 Memory | `0x05000000` | 32 MB（到 `0x07000000`） |
| L2 Programming Model | `0x07000000` | 1 MB |

手册 7.3 / Table 7-4：L2M 容量 **1 MB–32 MB**。Table 12-1：NPM4K/8K 的 L2 Size = 2 MB。变更记录 1.6.1.GA 写明删除了 L2 的 0.75MB；**768 KB 是 L1 选项**（7.2.1），不是 L2。

样例层文件里的数对不上 Table 7-12：

| 文件 | 域 | 值 | 十六进制 |
| --- | --- | --- | --- |
| 全部 422 层 | `L2 qman buffer offset` | 536805376 | `0x1FFF0000` |
| 全部 422 层 | `L2 qman buffer size` | 65536 | 64 KB，顶到 `0x20000000` |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `L2 input buffer offset 0` | 536641440 | `0x1FFC81A0` |
| 同 | `L2 weights buffer offset 0` / `1` | 8256 / 41024 | 小槽，差 = `L2 weights buffer size` 32768 |
| `dynamic_quantization_params_103_act_dq_phase2_params_256.txt` | `L2 output buffer offset` | 2112 | 小槽 |

手册第 8 章 DACU：控制器把 32 位虚拟地址翻成 64 位物理地址。样例的 `0x1FFxxxxx` 落在 4 GB 虚拟空间高端，不在 Table 7-12 的 `0x05000000` 段。

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无分配器。生成时若走 L2A，这些域由 L2A 填。若直接出 prepare_out：大 offset 按样例抄 QMAN=`0x1FFF0000`；小 offset 目前只有 16 个观测值，没有闭合公式 |
| 请确认 | 1) `0x1FFF0000` 是 DACU 虚拟地址还是物理地址？2) 与 Table 7-12 的 `0x05000000` 如何对应？3) 小槽 8256/41024/2112 是否必须复现，还是窗口内不重叠即可？ |
| 甲方填写 |  |

#### Q12. 相位号：cfg 从 1 起，dump 文件名从 0 起

| 层文件 | 域 | dump 文件 |
| --- | --- | --- |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | `softmax phase` = **1** | `Dataout file` = `output_buffer_phase_0_18.bin` |
| `..._act_sm_phase2_params_319.txt` | `softmax phase` = **2** | `Activation LUT file` = `LUT_phase_1_18.bin` |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | `dynamic quantization phase` = **1** | `Dataout file` = `output_buffer_phase_0_24.bin` |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | cfg 相位号 P 是 1-based；所有 `*_phase_N_*` 文件名 N = P−1 |
| 请确认 | 这是官方约定吗？我方必须遵守这套分裂吗？ |
| 甲方填写 |  |

#### Q13. `Task ID` / `Prev task` / `Next task` 怎么填（请对照文件看）

`Task ID` 出现在每一个层 txt 里，和 `Layer ID` 不是一回事。

- `Layer ID`：这一层在全图里的身份。softmax 第五相 `Layer ID=18`，第一相 `Layer ID=318`，互不相同。
- `Task ID`：这一层在**自己那条相位链里的序号**。head0 的 softmax 第一相是 0，第五相是 4；head1 的 softmax 第一相也是 0。所以全文件只有 0、1、2、3、4 五种值，32 头会重复使用同一套号。

生成算法（422 层无反例）：

```
若文件名带 gp_sm / act_sm / gp_dq / act_dq（相位链）：
    Task ID         = 该文件「softmax phase」或「dynamic quantization phase」的值 − 1
    Prev task i     = 本链中直接前驱的 Task ID
    Next task i     = 本链中直接后继的 Task ID
    Prev/Next count = 前驱/后继个数
否则（gemm / matmul / eltwise / vpu 单层）：
    Task ID = 0
    Prev task count = 0
    Next task count = 0
```

head0 Softmax 五个文件的实际值：

| 文件 | `softmax phase` | `Task ID` | `Prev task` | `Next task` | `Layer ID` |
| --- | --- | --- | --- | --- | --- |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | 1 | **0** | count=0 | 1 | 318 |
| `..._act_sm_phase2_params_319.txt` | 2 | **1** | 0 | 2 和 4 | 319 |
| `..._gp_sm_phase3_params_320.txt` | 3 | **2** | 1 | 3 | 320 |
| `..._act_sm_phase4_params_321.txt` | 4 | **3** | 2 | 4 | 321 |
| `..._act_sm_phase5_params_18.txt` | 5 | **4** | 1 和 3 | count=0 | 18 |

单层对照：`self_attn_q_proj_MatMul_qidx2_params_23.txt` 的 `Task ID=0`，`Prev task count=0`，`Next task count=0`。它和 softmax 第一相的 Task ID 都是 0，靠 `Layer ID`（23 vs 318）区分。

「改成全局唯一」的意思：如果我方写成 Task ID=0,1,2,…,421（一层一个号），`Prev task` / `Next task` 也改成指向这些全局号——底层编译器还认不认。样例用的是相位链内部 0-based。

| 项 | 内容 |
| --- | --- |
| 请确认 | 生成时是否必须按上表写（相位链内部 0-based，跨链重复）？若底层要求全局唯一，请给出编号规则 |
| 甲方填写 |  |

#### Q14. softmax 指数 LUT 怎么造

| 文件 | 域 | 值 |
| --- | --- | --- |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt` | `Activation LUT file` | `LUT_phase_1_18.bin` |
| 同 | `Activation Type`=13，`Flp min exp`=9，`Flp max exp`=16，`Flp mantisa`=3 |  |
| 二进制 | `parser_output/LUT_phase_1_18.bin` | 288 字节，32 头共用同一张 exp 表 |

对照：DQ 第二相的 `LUT_phase_1_24.bin` 是恒等表（第一个 fp16=1，其余 0），我方能自造且字节一致。DQ 第三相的倒数表也能自造。

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | exp 表的 32 个切点规则没有反推出来，目前只能拷贝参考产物这一张 |
| 请确认 | 段边界 / 采样公式。或者：允许我方拷贝这张 288 字节的表，不自己采样 |
| 甲方填写 |  |

---

### 15.2 P1（结构可先做，数值要对齐必须填）

#### Q15. `Data scale format = 7`

| 文件 | 域 | 值 |
| --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | `Data scale format` | **7** |
| 同文件相关 | `Data scale width`=32，`DDR data scale buffer size`=64 | 32 个 fp16 = 64 字节，所以 7 很像 fp16 |

71 个带 `Data scale *` 的层全是 7。请给 7 的官方名，以及其它编号。

甲方填写：

#### Q16. `Data scale buffer size` 的单位和公式

| 文件 | `Data scale width` | `Data scale buffer size` | 全量 scale 字节（width×2） |
| --- | --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | 32 | **32** | 64 |
| `mlp_gate_proj_MatMul_qidx397_params_195.txt` | 32 | **16** | 64 |
| `mlp_down_proj_MatMul_qidx405_params_192.txt` | 86 | **30** | 172 |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | 1 | **2** | 2 |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无。它不是 width，也不是 width×2。像引擎本地切片长度 |
| 请确认 | 单位是字节还是元素？公式是什么？ |
| 甲方填写 |  |

#### Q17. `L2 fpsu buffer size` 公式

| 文件 | 值 |
| --- | --- |
| `RMSNorm_params_197.txt` | 512 |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | 28672 |
| `self_attn_v_proj_MatMul_qidx38_params_36.txt` | 57344 |
| `mlp_gate_proj_MatMul_qidx397_params_195.txt` | 77824 |
| `mlp_up_proj_MatMul_qidx401_params_198.txt` | 77312 |
| `mlp_mul_Mul_qidx403_params_194.txt` | 154624 |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | 7168 |
| `mha_batch_matmul2_head0_qidx45_params_16.txt` | 1024 |
| DQ 各相 | 1024 |
| softmax 带 LUT 的相 | 512 |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无唯一公式。粗看与输出字节、是否双输入、是否带 LUT 有关 |
| 请确认 | 公式，或「按层类型查表」的官方表 |
| 甲方填写 |  |

#### Q18. 权重大小和 L2 切片

| 文件 | `L2 weights buffer size` | 全量 `weight_buffer_*.bin` |
| --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | **32768** | `weight_buffer_23.bin` = 16777216B（4096×4096×1，int4 一字节一值） |
| 同 | `L2 weight scale buffer size` = **131072** | `weight_sf_23.bin` = 262144B（4096×32×2） |
| `mlp_gate_proj_MatMul_qidx397_params_195.txt` | 16384 | `weight_buffer_195.bin` = 45088768B |
| `mlp_down_proj_MatMul_qidx405_params_192.txt` | 30720 | 同 45088768B |
| `RMSNorm_params_197.txt` | 4096 | `weight_buffer_197.bin` = 4096B（一次装完） |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 全量权重字节 = N×K×1；全量 sf 字节 = N×(K/128)×2。L2 里是双缓冲切片，切片公式未知。`L2 weights use double buffer`=1、`L2 weights buffers per engine`=2 |
| 请确认 | 切片公式。交付 `weight_buffer` 是否保持 1 字节/元素（不 nibble 打包）？ |
| 甲方填写 |  |

#### Q19. bmm2 的 `L2 output buffer size` 为何按 4096 占位

| 文件 | `Output Width` | 按公式 (align16(W)+16)×2 | 实测 `L2 output buffer size` |
| --- | --- | --- | --- |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | 1024 | 2080 | **2080**（对得上） |
| `mha_batch_matmul2_head0_qidx45_params_16.txt` | 128 | 288 | **8224** = (4096+16)×2 |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | bmm2 按整条 hidden=4096 的 fp16 平面占 L2，32 头各写其中 128 个元素 |
| 请确认 | 是否如此？32 头是否必须共用这块 8224 字节？ |
| 甲方填写 |  |

#### Q20. `DDR Weight buffer offset` 32 头都相同

| 文件 | 域 | 值 |
| --- | --- | --- |
| 全部 32 个 `mha_batch_matmul1_head*_params_*.txt` | `DDR Weight buffer offset` | **2112** |
| 全部 32 个 `mha_batch_matmul2_head*_params_*.txt` | `DDR Weight buffer offset` | **8208** |
| 同组 | `DDR Weight buffer size` | 131072 = 1024×128（一个 head 的 int8 KV） |
| 同组 | `Split weight index` | 0..31（头序号，这个是变的） |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | offset 是 L2/DMA 窗口里的固定槽，真正按头切片靠 `Split weight index`，不靠 offset |
| 请确认 | 2112 / 8208 怎么来的？生成时可否原样抄？ |
| 甲方填写 |  |

#### Q21. RMSNorm 权重 4096 字节是什么布局

| 文件 | 域 / 二进制 | 值 |
| --- | --- | --- |
| `RMSNorm_params_197.txt` | `Weights Buffer File` | `weight_buffer_197.bin` |
| 二进制 | 大小 | **4096** 字节 |
| 同 cfg | `bias buffer file` = `RMSNorm_Add_Const_197.bin` | 4 字节，按 fp32 解 = **1e-5**（即 `rms_norm_eps`） |
| 同 cfg | `weights scaling buffer file` = `weight_sf_197.bin` | 4 字节，按 fp32 解 ≈ **0.007874 = 1/127** |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | eps 已确认。gamma 若是 4096 个 fp16，应是 8192B，实际 4096B，所以不是 fp16 逐元素，或只存了半精度以外的打包 |
| 请确认 | `weight_buffer_197.bin` 的 dtype 和元素数。`weight_sf_197.bin` 的 1/127 用在哪一步？ |
| 甲方填写 |  |

#### Q22. `Flp min exp` / `Flp max exp` / `Flp mantisa`

| 文件 | 三元组 | 用途 |
| --- | --- | --- |
| `dynamic_quantization_params_24_act_dq_phase2_params_214.txt` | 10 / 17 / 3 | 恒等 LUT |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt` | 9 / 16 / 3 | exp LUT |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase4_params_321.txt` | 15 / 15 / 0 | 倒数 LUT |
| `mlp_gate_proj_MatMul_qidx397_params_195.txt` | 10 / 17 / 3 | SiLU LUT |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 按「LUT 种类」查这三张表，不会算 |
| 请确认 | 编码规则，或允许按上表抄 |
| 甲方填写 |  |

#### Q23. `Activation special operators = 4`

| 文件 | 值 | 该相数学 |
| --- | --- | --- |
| `dynamic_quantization_params_24_act_dq_phase3_params_215.txt` | **4** | 1/scale |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase4_params_321.txt` | **4** | 1/Σ |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt` | **0** | exp |
| `mlp_gate_proj_MatMul_qidx397_params_195.txt` | **0** | SiLU |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 4 = 取倒数；0 = 普通 LUT。139 层无反例 |
| 请确认 | 4 的官方名。还有哪些编号？ |
| 甲方填写 |  |

#### Q24. `Transpose type` 1 和 2

| 文件 | 值 |
| --- | --- |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt` | **1** |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase4_params_321.txt` | **2** |
| `dynamic_quantization_params_24_act_dq_phase3_params_215.txt` | **2** |

| 项 | 内容 |
| --- | --- |
| 我方当前算法 | 无。只出现在带 LUT 的中间相 |
| 请确认 | 1、2 的官方含义 |
| 甲方填写 |  |

#### Q25. `Weights source` / `Fpsu source`

| 文件 | `Weights source` | `Fpsu source` | 含义猜测 |
| --- | --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | **3** | **3** | 编译期常量 / registry |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | **0** | 3 | 0 = 从 DDR 取 KV；该文件另有 `weights from ddr: true` |
| softmax 各相 | 3 | **1** | 1 = 内部生成，没有 DDR 常量表 |

请给 0/1/3 的官方表。

甲方填写：

#### Q26. LUT 后 80 个 fp16 能否写 0

| 二进制 | 布局 |
| --- | --- |
| 全部 `LUT_phase_*.bin`、`activation_lut_file_195.bin` | 288B = 144 个 fp16：[0:32] slope，[32:64] intercept，[64:144] 填充 |

恒等表 `LUT_phase_1_24.bin` 在 [64:144] 全是 0，而这张表是能工作的（DQ 第二相）。

请确认：我方自造 LUT 时后 80 项是否允许全 0？exp 表是否必须字节级一致？

甲方填写：

#### Q27. `force consecutive execution` 是否只用于 RoPE

仅这 6 个文件为 1：

- `self_attn_Reshape_1_qidx18_params_30_mul_cos_params_201.txt`
- `..._mul_sin_params_202.txt`
- `..._add_params_203.txt`
- `self_attn_Reshape_qidx4_params_22_mul_cos_params_204.txt`
- `..._mul_sin_params_205.txt`
- `..._add_params_206.txt`

部分文件该行写成了 `force consecutive execution: 1skip compare: 1`（和下一域粘在一起）。

请确认：只有 RoPE 三连必须连跑吗？解析器是否容忍粘连？我方生成时会拆成两行。

甲方填写：

#### Q28. mask 怎么填数值

| 文件 | 域 | 值 |
| --- | --- | --- |
| `mha_masking_head0_qidx33_params_19.txt` | `Eltwise mode`=**2**，`Mask Input Index`=1，`Mask Data Type`=1 |  |
| 同 | `Datain file 1` | `input_buffer_1_190.bin` |
| `parser_output/IO_info.txt` 输入 6 | shape `[1,1,1,1024]`，dtype fp16，`mask: True` |  |

请确认：mask 是 0/−inf 还是加性 bias？32 头是否共享同一份 `input_buffer_1_190.bin`？

甲方填写：

#### Q29. RoPE 的 sin 支路，半头对换和符号写在哪

| 文件 | 相关域 |
| --- | --- |
| `self_attn_Reshape_1_qidx18_params_30_mul_sin_params_202.txt` | `Eltwise mode`=1，`Kantor mode`=5，`Eltwise broadcast factor`=32，`Eltwise broadcast Input Stride X`=128 |
| 同 | `Datain file 1` = `input_buffer_1_30.bin`（sin 表）；`Kantor A/B * buffer file` 一组 |

数学上 RoPE 需要把 128 维劈成两半并对其中一半变号。cfg 里看不到 split。请确认：这件事编进 sin 常量表，还是编进 Kantor shift？

甲方填写：

---

### 15.3 P2（有默认就能先往前走）

#### Q30. `Pooling data type` 恒为 2

379 层都是 2，例如 `self_attn_q_proj_MatMul_qidx2_params_23.txt` 的 `Pooling data type: 2`。GML 对应字符串 `floating_point` / `fixed_point`。请给 2 的含义。

甲方填写：

#### Q31. `skip compare`

全部 422 层 = 1。交付给我方仿真器时是否必须为 1？

甲方填写：

#### Q32. `net.ini` `[general]` 三个开关

`is_seq_test=0`，`seq_tunneling=0`，`test_update_buffer=0`。decode 单步是否永远填 0？

甲方填写：

#### Q33. `Scaling_PS_buffer_file`

`self_attn_q_proj_MatMul_qidx2_params_23.txt` 引用 `Scaling_PS_buffer_file_23.bin`，文件 **1 字节**，值 0x00。PS 是什么？取值含义？

甲方填写：

#### Q34. 没有 `L2 output buffer offset` 的相

例如 `mha_softmax_head0_qidx34_params_18_gp_sm_phase3_params_320.txt` 没有该域，且 `Virtual Output ... is: true`。是 virtual 不落 L2，还是原地复用上一相的槽？

甲方填写：

#### Q35. `L2 * buffer id`

`self_attn_q_proj_MatMul_qidx2_params_23.txt`：`L2 input buffer id 0`=4，`L2 data scale buffer id`=5，`L2 output buffer id`=6，`L2 weights buffer id`=w2，`L2 fpsu buffer id`=f2。请给编码规则，或允许按层类型抄样例。

甲方填写：

#### Q36. RMSNorm 是否拆成 3 个 prepare_out 层

手册 Normalization 是 3 阶段。本样例 `RMSNorm_params_197.txt` 仍是 **1** 个 `layer type: vpu`。请确认对外就是 1 层、3 阶段在硬件内部完成。

甲方填写：

#### Q37. GQA 时 `Total Split Weight Num`

`mha_batch_matmul1_head0_qidx30_params_20.txt` 里 `Total Split Head Num`=32、`Total Split Weight Num`=32（Llama2-7B 是 MHA）。换成 GQA 时后者是否改成 `num_key_value_heads`？

甲方填写：

#### Q38. `gml_version`

`txt_files/gml_version.txt` 内容为 **26.2.1**。我方不走 TVM，从 HuggingFace 构图。该填 26.2.1，还是自己的版本号？

甲方填写：

#### Q39. `Input data order` / `Output data order`

422 层几乎全是 0。还有别的取值吗？非 0 时 Width/Stride 怎么改？

甲方填写：

#### Q40. `Bytes in cycle internal memory read/write`

样例：全部 422 层，例如 `self_attn_q_proj_MatMul_qidx2_params_23.txt`：

- `Bytes in cycle internal memory read: 64`
- `Bytes in cycle internal memory write: 64`

手册 Table 7-11：NPM4K 及以上 L2MSS SysDMA 内部口 64 字节/周期。生成时按硬件型号查表，本样例抄 64。换 NPM2K（手册该口 32）是否改成 32？

甲方填写：

---

### 15.3 正文已指向、上面尚未单列的项

#### Q41. Softmax 是否固定 5 相

手册 4.2 写 3 步（分子 / 分母 / 归一化）。本样例每个 head 5 个层文件，例如 head0：

- `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt`（`softmax phase=1`）
- `..._act_sm_phase2_params_319.txt`（`softmax phase=2`）
- `..._gp_sm_phase3_params_320.txt`（`softmax phase=3`）
- `..._act_sm_phase4_params_321.txt`（`softmax phase=4`）
- `..._act_sm_phase5_params_18.txt`（`softmax phase=5`）

生成时按这 5 个文件各写一份。是否允许改成 3 个层文件？相位顺序能否改？

甲方填写：

#### Q42. softmax 第二相的 `Bias buffer file` 指向第一相输出

文件：`mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt`

- `Bias buffer file: output_buffer_phase_0_18.bin`

该文件正是第一相 `..._gp_sm_phase1_params_318.txt` 的 `Dataout file`（max 标量，4 字节）。生成时按「p2 的 Bias = p1 的 Dataout」写。是否就是用 max 做 x−max 的官方接法？

甲方填写：

#### Q43. softmax 第五相的 `Scaling buffer file` 指向第四相输出

文件：`mha_softmax_head0_qidx34_params_18_act_sm_phase5_params_18.txt`

- `Scaling buffer file: output_buffer_phase_3_18.bin`

该文件正是第四相 `..._act_sm_phase4_params_321.txt` 的 `Dataout file`（1/Σ，2 字节）。生成时按「p5 的 Scaling = p4 的 Dataout」写。是否就是用 1/Σ 做逐元素缩放的官方接法？

甲方填写：

#### Q44. Gemm 的 `input scale factor buffer` 指向 DQ 第二相还是第三相

文件：`self_attn_q_proj_MatMul_qidx2_params_23.txt`

- `input scale factor buffer: output_buffer_phase_1_24.bin`

`output_buffer_phase_1_24.bin` 是 DQ 节点 24 第二相的输出（scale），不是第三相的 `output_buffer_phase_2_24.bin`（1/scale）。生成时按第二相文件名写。该文件内容是 scale 还是 1/scale？

甲方填写：

#### Q45. `weight_buffer` 一字节一个 int4，WDM 开不开

文件：`self_attn_q_proj_MatMul_qidx2_params_23.txt`

- `Weights Data Type: 2`
- `Weight Compression Rate: 1.0`
- `Weights buffer file: weight_buffer_23.bin`

对照：`parser_output/weight_buffer_23.bin` = 16777216 字节 = 4096×4096，值域 −8..7。手册 7.3.4 WDM 可选，本样例未开。

生成 `weight_buffer_*.bin` 时按「1 字节存一个符号扩展的 int4」写。交付是否保持这种形态？nibble 打包或 WDM 是否由 L2A 之后做？

甲方填写：

#### Q46. `Cache idx`

| 文件 | 域 | 值 | 权重来源 |
| --- | --- | --- | --- |
| `mha_batch_matmul1_head0_qidx30_params_20.txt` | `Cache idx` | **0** | `Weights buffer file=input_buffer_199.bin`（K） |
| `mha_batch_matmul2_head0_qidx45_params_16.txt` | `Cache idx` | **1** | `Weights buffer file=input_buffer_200.bin`（V） |

生成时：读 K 填 0，读 V 填 1。是否官方约定？

甲方填写：

#### Q47. `Eltwise mode = 2` 的官方名

文件：`mha_masking_head0_qidx33_params_19.txt`

- `Eltwise mode: 2`
- `Mask Input Index: 1`
- `Datain file 1: input_buffer_1_190.bin`

同文件 `mode=0` 是残差 Add，`mode=1` 是 Mul。2 是独立 MaskAdd，还是 Add 与 mask 融合？

甲方填写：

#### Q48. `DDR Input Orig Buffer Name` 命名是否约束

文件：`self_attn_q_proj_MatMul_qidx2_params_23.txt`

- `DDR Input Orig Buffer Name 0: buffer15`

`self_attn_v_proj_MatMul_qidx38_params_36.txt`：

- `DDR Output Orig Buffer Name: value_cache_out`

生成时线性层用 `bufferN`，KV 用语义名。名称是否必须与参考产物一致？

甲方填写：

#### Q49. Dataout 文件名是否必须等于下游 Datain

文件：`dynamic_quantization_params_24_act_dq_phase4_params_24.txt`

- `Dataout file: output_buffer_phase_3_24.bin`

下游 `self_attn_q_proj_MatMul_qidx2_params_23.txt`：

- `Datain file: output_buffer_phase_3_24.bin`

GML 按消费者编号。生成时 Dataout 与下游 Datain 写成同一个文件名。是否硬规则？

甲方填写：

#### Q50. `Kantor A source`

| 文件 | 域 | 值 | `Kantor A scale buffer file` |
| --- | --- | --- | --- |
| `dynamic_quantization_params_24_act_dq_phase4_params_24.txt` | `Kantor A source` | **1** | `output_buffer_phase_2_24.bin`（上游相位输出） |
| `self_attn_Reshape_1_qidx18_params_30_mul_cos_params_201.txt` | `Kantor A source` | **0** | `Kantor_A_Shift_Llama2Activation_Cos_30.bin`（常量） |

生成时：scale 来自上一相填 1，来自常量 bin 填 0。0/1 的官方名？

甲方填写：

#### Q51. `Quant_source`

文件：`self_attn_q_proj_MatMul_qidx2_params_23.txt`

- `Quant_source: 1`
- 同时有完整 `Data scale *` 一组，`input scale factor buffer` 指向 DQ 第二相

无动态 scale 的层（如 `RMSNorm_params_197.txt`）为 0。生成时：该层使用 DQ 产物填 1，否则填 0。官方名？

甲方填写：

#### Q52. `Output Stride Z` 的例外

终相平面：`self_attn_q_proj_MatMul_qidx2_params_23.txt` 的 `Output Width=4096`，`Output Stride Z=4111` = align16(4096)+15。

例外：

| 文件 | `Output Width` | `Output Stride Z` |
| --- | --- | --- |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase2_params_319.txt` | 1024 | **1024**（等于 Width，不对齐） |
| `dynamic_quantization_params_103_act_dq_phase2_params_256.txt` | 1 | **16**（不是 31） |
| `dynamic_quantization_params_24_gp_dq_phase1_params_213.txt` | 32 | **32** |

生成规则：终相 `SZ=align16(W)+15`；相位链内部 `SZ=W`；Width=1 的中间相有时为 16。这三条例外是否官方？

甲方填写：

#### Q53. DQ 第一相的 `Bias buffer file`

文件：`dynamic_quantization_params_24_gp_dq_phase1_params_213.txt`

- `Bias buffer file: Bias_buffer_phase_0_24.bin`

对照：`parser_output/Bias_buffer_phase_0_24.bin` 为 4 字节，按 fp32 解是约 2^−63 量级的非零值。生成该 bin 的公式？

甲方填写：

#### Q54. LUT 交付是否要求与参考产物字节级相同

涉及：`parser_output/LUT_phase_1_18.bin`（exp，softmax p2）、`LUT_phase_2_24.bin`（倒数）、`LUT_phase_1_24.bin`（恒等）、`activation_lut_file_195.bin`（SiLU）。

恒等和倒数我方能自造；exp 目前只能拷贝。交付是字节级一致，还是数值误差在某阈值内即可？

甲方填写：

#### Q55. 线性层 `Scaling_buffer_file` 为何是 0.5

文件：`self_attn_q_proj_MatMul_qidx2_params_23.txt`

- `Scaling buffer file: Scaling_buffer_file_23.bin`

对照：`parser_output/Scaling_buffer_file_23.bin` = 2 字节，fp16 值 = **0.5**。该 0.5 从哪来？生成时是否所有 Gemm 都写 0.5？

甲方填写：

#### Q56. `Layer ID` 发号能否当硬规则

样例：

| 文件 | `Layer ID` | 文件名最后一个 `params_N` |
| --- | --- | --- |
| `self_attn_q_proj_MatMul_qidx2_params_23.txt` | 23 | 23 = GML 节点号 |
| `mha_softmax_head0_qidx34_params_18_act_sm_phase5_params_18.txt` | 18 | 18 = 主节点号 |
| `mha_softmax_head0_qidx34_params_18_gp_sm_phase1_params_318.txt` | 318 | 318 = 辅号 |

规则：单层和多相的最后一相 `Layer ID = GML node_id`；前面的相从 201 起另发；文件名最后一个 `params_N` 等于该文件 `Layer ID`。生成时按此写。是否可当硬规则？

甲方填写：

---

## 16. 请这样回

对 Q1–Q56 每一条，三种回法之一：

1. **同意**（按该条当前算法 / 公式实现）
2. **改正文**（直接写公式，或「编号 = 官方名」）
3. **给手册页码**

枚举类写成：

```
0 = int8（有符号，1 字节）
1 = fp16
3 = fp32
```

只回「是浮点」不够用来填域。

优先回 Q1–Q14。这 14 条不齐，对应域无法生成。

生成时按第 1.5 节步骤 A（图）→ B（算子）→ C（编排）→ E（全量域）走。闭合部分代入 Llama2-7B 的 H/I/nh/S/G 即可；查表部分按层类型抄 B7 / E1；编号表和 L2 分配器等第 15 章回完再填。

---

## 附录 A. 本文件章节

| 节 | 内容 |
| --- | --- |
| 0 | 怎么读、置信度、责任方、链路 |
| 1 | 模型与手册约定；1.4 闭合量；1.5 编译器推导（A–E） |
| 2 | 文件全景、命名、Layer ID、Task ID |
| 3 | net.ini |
| 4–5 | 公共形状 / 卷积壳 |
| 6 | DQ 4 相 |
| 7 | Softmax 5 相 |
| 8 | Gemm / MatMul |
| 9 | Eltwise |
| 10 | RMSNorm |
| 11 | L2 / DDR / Task / dump 名 |
| 12 | 枚举 |
| 13 | LUT 与 FPSU 小表 |
| 14 | 确认后要补的原语和 pass |
| 15 | 待填清单 Q1–Q56 |
| 16 | 回法 |

## 附录 B. 证据来源

| 来源 | 用在哪 |
| --- | --- |
| Ceva-NeuPro-M High-Level ArchSpec V1.6.6.GA | 相位、PWL、Kantor/FPSU/Pooling、QMAN、数值模式 10.5、地址表 7-12 |
| VBU-GML Structure | extension、kantor_mode 六档、phase 字段、Llama2Activation |
| prepare_out 422 层穷举 | 枚举取值、Task ID、Layer ID、形状、L2 尺寸、全量键 |
| parser_output dump 字节 | dtype、DQ/Softmax 相位数学、权重 sf、LUT |
| Llama-2-7b-hf/config.json | H=4096，I=11008，heads=32，eps=1e-5，SiLU |
| IO_info.txt | 7 输入 3 输出，KV [1,32,1024,128]，mask [1,1,1,1024] |
