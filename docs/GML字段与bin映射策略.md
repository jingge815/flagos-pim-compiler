# relay2gml_graph.gml 字段与 .bin 映射策略

日期：2026-09-16（2026-09-16 二次修订：接入硬件规范与真实模型）
对象：`llama2_w4a8_decode_block_0/parser_output/`（参考产物，Llama2-7B W4A8 decode block 0）

**四份依据**：

| 依据 | 文件 | 作用 |
| --- | --- | --- |
| GML 字段规范 | `VBU-GML Structure-281025-031239.pdf`（20 页） | 字段语义的官方定义 |
| **硬件架构规范** | **`Ceva-NeuPro-M_High_Level_ArchSpec_V1.6.6.GA.pdf`（147 页）** | **解释了字段背后的硬件单元；本次修订的主要来源** |
| 模型 | `flagOS-installed/model-inference/models/Llama-2-7b-hf` | 所有形状与超参的真源 |
| 对照产物 | `/tmp/gml_full/`（pim-compiler 当前输出） | 差距分析 |

**目标硬件已确认**：GML 规范里的「NPM device」就是 **Ceva-NeuPro-M**（1.4.1.EA 起，
NeuPro-M11/12/14/18/116 改名为 **NPM4K / NPM8K / NPM16K / NPM32K / NPM64K**）。
GML 里的 `nmu_*`、`fpsu_*`、`kantor_*`、`global_pooling_*`、`activation_*`、`vpu_params`
六族字段，逐一对应硬件规范里的 NMU、CSTL/FPSU、KANTOR、Pooling、Activation、VPU 六个单元。
**这把「字段取什么值」从猜测变成了查规范**，见 §2.0。

本文的用途：**为图编译器 + 算子编译器从 Llama2-7B 模型生成 GML 及其附属 bin 提供逐字段实现依据**。
每个字段给出「含义 / 功能 / 取值范围 / 计算策略 / 由谁计算」五列，每个 bin 族给出「字节格式 / 元素数公式 / 数值来源 / 计算方法 / 由谁计算」。

所有数值结论均为对实物逐字节解码的实测结果，复现脚本见 §12。**推测**二字标出未能实测确认的部分。

---

## 0. 速览

### 0.1 规模

| 项 | 值 |
| --- | ---: |
| GML 行数 / 节点 / 边 | 16012 / 200 / 331 |
| 顶层不同键名（未归一化） | **617** |
| 归一化字段族（折叠槽号/相号/单元号后） | **201** |
| 引用的 `.bin` 名字 | 3464 |
| 落盘 `.bin` | 3231（缺 379 个全为 `DEBUG_*`/`lut_debug`） |
| `.bin` 文件族 | **76** |

> 既有文档记 258 族，本文归一化后为 201 族。差异全在归一化口径（本文额外折叠了
> `fpsu_<单元号>_*`、`Scaling_buffer_file_<单元号>_Llama2Activation_<块>`、
> `input_<槽>_sf` 等编号变体）。两者不矛盾，§3 的分类表逐族列出，可直接核对。

### 0.2 本文相对既有文档的修正

这些项直接影响能否生成**字节正确**的 bin，是本文最重要的部分。带 **[硬件规范]** 标记的是
本次二次修订新增或由硬件规范独立佐证的。

| # | 既有文档的说法 | 实测结论 | 影响 |
| --- | --- | --- | --- |
| 1 | `Bias_buffer_*` 是 4 字节 fp32 = 0.0 | **是 fp32，但不恒为 0**。DQ phase0 恒为 **2⁻⁶³**（=1.0842e-19），Softmax phase1 为 **-30.75**。**[硬件规范]** 独立佐证：FPSU「加 **32 位** bias，乘 **16 位**有符号 scale，round 后**右移**」——4B/2B/1B 三个文件正是这三个操作数 | 写 0 会让 DQ 的 absmax 求解与 Softmax 的 exp 段失效 |
| 2 | Softmax phase0/phase2 输出是 2 个 fp16 | **是 1 个 fp32**。phase2 的 fp32 = 84.875 = Σexp（精确吻合） | 按 2×fp16 写会写出错误字节 |
| 3 | `Scaling_buffer_file` 恒为 fp16 1.0 | **有 5 种值**：1.0(55)、**0.08838(32)=1/√128**、0.5、0.25、2.0(2) | 1/√128 是注意力缩放，写 1.0 等于丢掉 attention scale |
| 4 | `weight_sf_multiplier` 含义未明 | 实测 **= 1/`Scaling_buffer_file`**（节点 23：mult=2↔scale=0.5；节点 36：mult=4↔scale=0.25） | 次正规数修正的补偿路径可闭合 |
| 5 | GML 的 96 个硬件字段族须由算子编译器提供 | 实测 **100% 由 `(op_type, phase)` 决定**，无一按节点变化 → 图编译器用常量表即可填全 | 解除对算子编译器的阻塞（详见 §7） |
| 6 | Softmax 的 phase 语义未展开 | 五级全部实测反推：`-max → exp(x-max) → Σ(fp32) → 1/Σ → ×(1/Σ)`，phase4 与 p1×p3 **1024/1024 逐元素吻合**，Σ(p4)=0.99975。**[硬件规范]** 佐证：规范 §4.2「Self-Attention (Softmax Support)」明列 numerator/denominator/final 三层 | Softmax 可完整生成 |
| **7** | **LUT 是唯一硬阻塞，`[64:103]` 编码未知** | **[硬件规范]** 规范 §4.3.3：「PWL LUT 支持 **32 段**，每段由一个 **slope** 和一个 **intercept** 定义」。实测布局 `[0:32]`=slope、`[32:64]`=intercept 正好吻合。**倒数表完整破解**：每段是 `1/x` 的**切线**（`A[i] = -B[i]²/4`，31 段全满足），段索引取 fp16 尾数高 5 位；自行合成的表与硬件输出平均差 **0.04%**，**优于参考产物的表（0.55%）**。**恒等表可字节级复现**。**`[64:104]` 是未初始化残留而非参数**（恒等表该区全 0 仍能工作） | **LUT 不再是阻塞**：4 张表 3 张可合成，仅 exp 表需拷贝一次，见 §4.8 |
| **8** | **DQ 的 `2×absmax` 语义不明** | **[硬件规范]** 规范 §4.3.4：Pooling 块「算 abs(max(x))（对称 DR）」后「算前导零并据此**左移** DR」。左移 1 位 = ×2。实测 32/32 组 `p0/absmax` 恒为 **2.0**。**×2 的目的**：使固定的 `×256` 右移后 absmax 正好落到 **128** 满量程（`output_sf = absmax/128`，32/32 组吻合） | 量化链路的每一步都有硬件依据 |
| **9** | 本文一次修订称 Softmax `Bias_buffer_phase_1` 是**编译期常量 -30.75** | **更正**：它与 `output_buffer_phase_0`（phase0 求出的 `-max`）**32/32 节点逐字节相同**。它不是常量，而是 **phase0 归约结果的落盘位置**——运行时由硬件写入，编译期只需分配 | 不必也不应把 -30.75 硬编码；参考产物的值来自其合成输入 |

> 第 9 项是对本文自身前一版的更正。判据：`Bias_buffer_phase_1_<id>` 与
> `output_buffer_phase_0_<id>` 在全部 32 个 Softmax 节点上字节相同，且该值恰为
> `-max(x)`。FPSU 的 bias 操作数就是前一相的归约输出——这与硬件规范 §4.3.2
> 「FPSU 加 32 位 bias」以及 §4.2「动态量化通过测量运行时动态范围来修正 scale/shift」
> 一致。DQ 的 `Bias_buffer_phase_0` = 2⁻⁶³ 则确认是常量（36/36 节点字节相同，
> 且它是流水线**首**相，没有前序归约可承接）。

### 0.3 四类归属（本文的「由谁计算」取值）

| 记号 | 归属 | 判据 |
| --- | --- | --- |
| **G** | 图编译器 | 由计算图拓扑、形状、量化契约、模型权重决定 |
| **K** | 算子编译器（FlagTree） | **算子内部**的执行方式：分块结果、片上/主存地址、步幅、缓冲区编号 |
| **C** | 常量表（图编译器内置） | 由 `(op_type, phase)` 唯一确定的硬件模板值，与具体节点无关 |
| **?** | 待对方确认 | 编码规则未能从数据反推 |

**C 与 K 的区分是本文的核心判断**，它把既有文档划给 K 的 96 个 GML 字段族里的绝大部分移到了 C。
论证见 §7.1。用户对「不是所有硬件相关的都该算子编译器干」的质疑，实测支持。

---

## 1. 语法与解析前提

写生成器之前必须知道这四条，否则产出的文件对方读不了。

| # | 规则 | 实测证据 |
| --- | --- | --- |
| 1 | **同名键在一个 node 内可重复**。`residual_input_buffer` 每个输入槽出现一次。标准 GML 解析器只保留最后一个，会丢数据 | `residual_input_buffer` 出现 310 次 / 193 个节点 |
| 2 | **数组被拆成多行同名键**（PDF 第 17 页明确要求）：`pads [1,2]` → `pads 1` + `pads 2` | `kernel_shape` 108 次、`pads` 216 次 |
| 3 | **`id` ≡ `node_id`，`name` ≡ `label`**。前者是 GML 保留字，后者供 L2Analyzer 用，两份都要写 | 200/200 相同 |
| 4 | **`id` 顺序不是执行顺序**。算子链多为「大 id → 小 id」。判先后必须走 edge 或 `inputN_node_id` | 节点 195→194→193 为正向数据流 |

**两个必须复现的键名 bug**（对方产物自带，我方若要字节对齐就得照写）：

| bug | 实测次数 | 正确形式 |
| --- | ---: | --- |
| 端口号 ≥10 时 `residual_input_buffer_` / `residual_output_buffer_`（末尾多下划线、端口号丢失） | 22 / 88 | `residual_input_buffer` |
| `fpsu_<n>_scale_axisLlama2Activation_<块>` 缺下划线 | 6 | `..._scale_axis_Llama2...` |

---

## 2. 硬件单元与字段的对应（本次修订新增）

### 2.0 GML 字段 ← 硬件单元

目标硬件已确认为 **Ceva-NeuPro-M**（NPM 系列）。GML 的字段前缀直接对应硬件单元，
知道这层映射后，「某字段该填什么」多数情况下可以直接查硬件规范而不必猜。

| GML 字段前缀 | 硬件单元 | 规范章节 | 单元职能（规范原文要点） |
| --- | --- | --- | --- |
| `nmu_*` | **NMU** | §3 | 矩阵乘累加阵列。支持 FP4/FP8/BF16 等 |
| `fpsu_*`、`Scaling_*`、`Bias_*` | **FPSU**（在 CSTL 内） | §4.3.2 | 逐元素重定标：**加 32 位 bias → 乘 16 位有符号 scale → round → 右移 → 饱和到 16 位**。另负责定点/FP4/FP8/FP16/BF16/TF32/FP32 → FP16/BF16/INT16 的格式转换 |
| `activation_*`、`LUT_*`、`flp_*` | **Activation**（在 CSTL 内） | §4.3.3 | **PWL LUT 逼近非线性函数，32 段，每段一个 slope 与一个 intercept**。另支持 `1/x`、`x²`、`1/sqrt(x)`、`e^x`、标准差等复杂运算 |
| `global_pooling_*`、`pooling_dtype` | **Pooling**（在 CSTL 内） | §4.3.4 | 全局池化（按 tensor/通道/通道组/元素组）。**支持动态量化：算 abs(max(x)) 对称 DR，再算前导零并左移 DR** |
| `kantor_*` | **KANTOR** | §4.3.6 | 逐元素乘、浮点↔定点转换（从指数生成定点 scale、转换尾数）、FP16/BF16/INT16 → 更窄格式 |
| `vpu_params` | **VPU** | §4 | 独立向量通路（RMSNorm 走这条） |
| `residual_*` | **Residual Add/Concat** | §4.3.5 | 残差加与拼接 |

**这解释了三个此前只能靠实测归纳的现象**：

1. **为什么定标是三个文件**（`Bias_buffer` 4B / `Scaling_buffer` 2B / `Scaling_PS_buffer` 1B）：
   规范 §4.3.2 的 FPSU 操作序列就是「加 32 位 bias、乘 16 位 scale、右移」。三个文件
   一一对应三个操作数，字节宽度精确吻合（4B=32bit bias、2B=16bit scale、1B=移位量）。
   规范 §10.4 另给出定点模式的 dtype 规则：「Scale dtype is int16, bias dtype is int32」。
   → **`Scaling_PS` 的 `PS` 是 Post-Shift（后置右移量），不是 Partial Sum。**
   佐证：393/395 个 `Scaling_PS` 文件为 0（浮点 FPSU 不需右移），唯一非零的两个
   （值 14）正是唯一 `fpsu_mode="fixed_point"` 的 KV_Cache_DMA 节点。

2. **为什么 DQ phase0 输出 `2×absmax` 而不是 `absmax`**：规范 §4.3.4 说 Pooling 在算完
   对称 DR 后「算前导零并据此左移」。左移 1 位即 ×2。目的是让后续固定的 `×256`
   右移之后，absmax 正好落在 **128**（int8 满量程），即 `output_sf = absmax/128`。
   实测 32/32 组吻合，且量化结果用满了 `[-128,127]` 全域。

3. **为什么 LUT 是 288 字节 = 144 个 fp16**：规范 §4.3.3 说 PWL LUT 是 **32 段，
   每段一个 slope + 一个 intercept**。实测布局 `[0:32]` 是 slope、`[32:64]` 是 intercept，
   正好 64 个 fp16 = 2×32。详见 §4.8。

### 2.1 顶层与骨架

| 字段 | 含义 | 功能 | 取值范围 | 计算策略 | 谁算 |
| --- | --- | --- | --- | --- | --- |
| `graph [` | 根块 | 容器 | 固定 | 固定输出 | G |
| `directed` | 有向图标志 | networkx 属性 | 恒 `1` | 常量 | G |
| `relay2gml_version` | 生成器版本 | 后端据此选解析分支 | `"26.2.1"`（参考产物）/ `"26.10.1"`（runtime_files 的 ResNet 样本） | **须与对方约定**；我方自定版本号有兼容风险 | ? |
| `node [` ×200 | 节点块 | 算子或缓冲 | 10 buffer + 190 算子 | 见 §3 | G |
| `edge [` ×331 | 边块 | 数据流 | — | 见 §3.2 | G |

**节点两种形态**，序列化器都要支持：

| 形态 | 数量 | 特征 | node_id |
| --- | ---: | --- | --- |
| 图输入 buffer | 7 | `is_buffer 1` + `from_tvm 1` + `output_*` | 1–7 |
| 图输出 buffer | 3 | `is_buffer 1`，无 `from_tvm`，用 `input_*` | 8, 199, 200 |
| 算子节点 | 190 | 有 `op_type`，无 `is_buffer` | 其余 |

### 2.2 前端替换：我方不走 TVM/Relay

参考产物由对方的 `relay2gml` 工具从 **TVM Relay** 子图翻译而来，所以它的
`raw_relay_mod.txt` / `quantize_relay_mod.txt` 与若干字段带着 Relay 痕迹。
**我方不使用 TVM，也没有 Relay IR** —— pim-compiler 直接从 HF 模型
（`transformers` + `torch.fx`）构图。本节把每一处「原本依赖 Relay」的信息
换成我方的等价来源，是后续改代码的直接依据。

| 参考产物依赖 Relay 的信息 | 我方等价来源 | 落点 |
| --- | --- | --- |
| 算子拓扑与形状 | **`torch.fx` 图**（`gml_bridge/from_fx.py`，`node.meta["val"].shape`） | 节点/边/`dims` |
| 算子类型映射 | **`OP_TYPES` 表**（aten op → GML `op_type`），已存在 | `op_type` |
| 算子路径命名（`<算子路径>`） | **HF 模块路径**（`model.layers.0.mlp.gate_proj` → `mlp_gate_proj`） | `label` / `name` |
| `qidx<N>` 访问序号 | **我方量化 pass 的遍历计数器**（自行编号） | `label` / `name` |
| 图 I/O 张量名（`original_name`） | **HF forward 的形参名**（`hidden_states`、`attention_mask`、`past_key_value`…） | `original_name` |
| 量化契约（`simulated_quantize` 注解） | **我方量化配置**：int8 激活 / int4 权重、对称、per-group 128（见 §6.2） | 各 `*_dtype` / `*_sf` / `*_zp` |
| 权重数值 | **safetensors 直接加载**（`model-0000{1,2}-of-00002.safetensors`） | `weight_buffer` / `weight_sf` |
| 超参（hidden/heads/eps…） | **`config.json`**（见 §6.1） | 形状、`num_heads`、`RMSNorm_Add_Const` |
| RoPE 频率 | **检查点里的 `rotary_emb.inv_freq`**（`[64]` fp32） | RoPE cos/sin 表 |
| 算子融合边界 | **我方融合 pass**（`graph/fuse.py`，`FUSED_TAIL_META_KEY`） | `contraction` 块 |
| `axes` 等算子属性 | **fx 节点的 args**（`aten.permute` 的 dims 等） | `axes` / `axis` |

**三个字段名带 tvm/relay 但必须照写**（它们是对方 GML 的既定契约，与我方前端无关）：

| 字段 | 处理 |
| --- | --- |
| `relay2gml_version` | 版本头，照写 `"26.2.1"`。**它标识 GML 格式版本，不表示我方用了 relay**（须与对方确认我方该填什么） |
| `from_tvm 1` | 实际语义是「这是图输入 buffer」。参考产物里 7 个图输入全为 1，我方照置 1 |
| 文件名 `relay2gml_graph.gml` | 对方后端按这个固定名字找文件，必须照用 |

> **本文其余各节凡引用 `raw_relay_mod.txt` / `quantize_relay_mod.txt` 的地方，
> 都只作为「参考产物长什么样」的证据**，不构成我方的实现依赖。我方对应的信息源
> 见上表。

---

## 3. 字段总表（按功能分域）

列含义：**含义** = 语义；**功能** = 后端拿它干什么；**范围** = 实测取值域；**计算策略** = 从 Llama2-7B 出发怎么算出来；**谁** = G/K/C/?。

### 3.1 标识与溯源（8 族）

| 字段 | 出现 | 含义 | 功能 | 范围 | 计算策略 | 谁 |
| --- | ---: | --- | --- | --- | --- | --- |
| `id` | 200 | GML 主键 | networkx 连通性 | 1–200 连续 | 拓扑排序后顺序分配 | G |
| `node_id` | 200 | 语义节点号 | L2Analyzer 索引、bin 文件名后缀 | ≡ `id` | 同 `id` | G |
| `label` | 200 | 算子名 | 可视化；**net.ini 层名由此派生** | 见下 | 见下 | G |
| `name` | 200 | 同 `label` | L2Analyzer | ≡ `label` | 同 `label` | G |
| `original_name` | 10 | 图级 I/O 张量名 | 运行时 I/O 绑定 | 参考产物用 `nprm_182_i<N>` / `tvmgen_default_nprm_main_182_output_<k>`（对方工具链的命名） | **我方自定，与 TVM 无关**：直接用 HF 的语义名（`hidden_states`、`key_cache`…），见 §2.2 | G |
| `is_buffer` | 10 | 缓冲节点标志 | 后端跳过计算单元分配 | 恒 `1` | 图输入/输出节点置 1 | G |
| `from_tvm` | 7 | 图输入来源标记 | 后端区分两类前端来源 | 恒 `1` | **字段名带 tvm 但语义只是「图输入」**：仅图**输入** buffer 置 1。我方照置 1（见 §2.2） | G |
| `is_mask` | 1 | attention mask 标记 | 后端特殊搬运 | 恒 `1` | 识别 causal mask 输入张量 | G |

**`label` 命名规则**（参考产物实测 190/190 成立）：

```
算子节点：  <算子路径>_qidx<N>_params_<node_id>
             例 mlp_gate_proj_MatMul_qidx397_params_195
                self_attn_o_proj_MatMul_qidx392_params_11
                mha_batch_matmul1_head0_qidx30_params_20   <- 逐头，含 head<h>
                mha_softmax_head0_qidx34_params_18
                dynamic_quantization_params_12             <- DQ 无 qidx
                RMSNorm_params_25                          <- 融合算子无 qidx
图输入 buffer： nprm_<subgraph_id>_i<idx>      <- 对方前端的内部命名
图输出 buffer： output / key_cache_out / value_cache_out
```

三段信息：`<算子路径>`、`qidx<N>`（对方量化 pass 的访问序号）、`params_<node_id>`（本节点 id）。

**我方的构造方式（不依赖 TVM/Relay）**：`<算子路径>` 直接来自 **HF 模块路径**，
`qidx` 由我方量化 pass 自行编号。对照关系一目了然：

| 参考产物 label | HF 模块路径 | 我方构造 |
| --- | --- | --- |
| `mlp_gate_proj_MatMul_qidx397_params_195` | `model.layers.0.mlp.gate_proj` | `mlp_gate_proj_MatMul_qidx<n>_params_195` |
| `self_attn_o_proj_MatMul_qidx392_params_11` | `model.layers.0.self_attn.o_proj` | `self_attn_o_proj_MatMul_...` |
| `RMSNorm_params_25` | `model.layers.0.input_layernorm` | `RMSNorm_params_25` |
| `mha_batch_matmul1_head0_qidx30_params_20` | 逐头展开产物，无对应模块 | `mha_batch_matmul1_head<h>_...` |

规则：取 HF 模块路径去掉 `model.layers.<L>.` 前缀、`.` 换 `_`，接算子类型名。
逐头展开与 DQ/Softmax 这类无模块对应的节点，按参考产物的固定串（`mha_batch_matmul1_head<h>`、
`mha_softmax_head<h>`、`dynamic_quantization`）构造。

> `qidx` 的递增规则未能反推（分布不连续：2,4,16,18,26,30,33,34,38,45,…），它是对方
> 量化 pass 的内部计数器。**只影响 label 字符串与 net.ini 层名，不影响任何数值**。
> 我方按自己的量化 pass 遍历序编号即可，需与对方确认是否要求严格一致。

### 3.2 数据流连接（10 族）

数据流被**冗余记录三次**，三者实测逐边相等（集合差为空），生成时必须三份同步。

| 记录方式 | 边数 |
| --- | ---: |
| `edge [source/target]` | 331 |
| 生产者 `outputN_node_id` | 331 |
| 消费者 `inputN_node_id` | 331 |

| 字段 | 出现 | 含义 | 功能 | 范围 | 计算策略 | 谁 |
| --- | ---: | --- | --- | --- | --- | --- |
| `source` / `target` | 331 各 | 边的生产者 / 消费者 node_id | 拓扑 | 1–200 | 由计算图边直接得 | G |
| `dims` | 331 | 张量 shape | 后端算搬运量 | 11 种，`AxBxCxD` | `"x".join(shape)`，四维定长 | G |
| `label`(edge 内) | 331 | 同 `dims` | 可视化 | ≡ `dims` | 同 `dims`（331/331 相同） | G |
| `inputN_node_id` | 331 | 第 N 输入的生产者 | 调试/连通 | N=0..31 | 按算子 operand 顺序 | G |
| `outputN_node_id` | 331 | 第 N 输出的消费者 | 调试/连通 | N=0..31 | 按消费者顺序 | G |
| `residual_input_buffer` | 310+22 | 与同位 `inputN_node_id` **数值相同** | PDF 注为 irrelevant | 同上 | 与 `inputN_node_id` 同值重复写 | G |
| `residual_output_buffer` | 243+88 | 与同位 `outputN_node_id` 相同 | 同上 | 同上 | 同上 | G |
| `input_count` | 193 | 登记的输入数 | 调试 | 1(153)/2(35)/3(4)/32(1) | **注意 MatMul 少记 1**，见下 | G |
| `idx` | 197 | **本节点输出挂在消费者的第几个输入端口** | 端口绑定 | 0(155)/1(9)/2(4)/3/4 | `consumer.input<idx>_node_id == self.node_id`（197/197 成立） | G |
| `A` | 71 | 矩阵 A 操作数来源 | 单元操作数选择 | 恒 `= input0_node_id`（71/71） | 直接复制 `input0_node_id` | G |

**两个必须注意的坑**：

1. **`idx` 不是「第几个输出」而是「在下游的哪个端口」**。Split 26(k) 与 Split 32(v) 都是
   `idx=1`，因为都挂在 MatMul 的 `input1`。
2. **`input_count` 对 MatMul 少记 1**。全图 Σ`input_count`=267，Σedge=331，差 **64 = MatMul 数**。
   因为 64 个 attention MatMul 各有两个 operand（`input0/1_node_id`），但 `input_count` 只记 1
   —— 第二个走**权重通路**（`MatMul_input_as_weight 1`）。**生成时不能仅凭 `input_count` 分配 operand。**

### 3.3 缓冲区命名契约（最关键，最易错）

核心反直觉点：**数据缓冲按消费者编号，不按生产者编号**。因为缓冲代表的是**边**，不是「某节点的输出」。

| 模式 | 含义 | 例 | 谁 |
| --- | --- | --- | --- |
| `input_buffer_<consumer>.bin` | 单输入算子的数据缓冲 | `input_buffer_25.bin` | G |
| `input_buffer_<slot>_<consumer>.bin` | 多输入算子第 slot 个输入 | `input_buffer_1_19.bin` | G |
| `input_sf_<consumer>.bin` / `input_<slot>_sf_<consumer>.bin` | 输入 scale | `input_0_sf_15.bin` | G |
| `input_zp_<consumer>.bin` / `input_<slot>_zp_<consumer>.bin` | 输入 zero-point | | G |
| `weight_buffer_<self>.bin` | 权重，按**本节点**编号 | `weight_buffer_195.bin` | G |
| `weight_sf_<self>.bin` / `weight_zp_<self>.bin` | 权重量化参数 | | G |
| `output_sf_<self>.bin` / `output_zp_<self>.bin` | 输出 requant 参数，按**本节点** | | G |
| `Scaling_buffer_file_<self>.bin` | FPSU per-channel scale（**S 大写**） | | C |
| `Scaling_PS_buffer_file_<self>.bin` | FPSU post-shift | | C |
| `Bias_buffer_file_<self>.bin` | FPSU bias（**B 大写**） | | C |
| `Scaling_buffer_file_<slot>_<self>.bin` | 双输入 eltwise 逐槽 FPSU | `Scaling_buffer_file_0_9.bin` | C |
| `output_buffer_<self>.bin` | **仅 phase 型算子**用自己编号命名输出 | `output_buffer_12.bin` | G |
| `<in\|out>put_buffer_phase_<k>_<self>.bin` | phase 中间张量 | | G |
| `Scaling\|Scaling_PS\|Bias_buffer_phase_<k>_<self>.bin` | 各 phase 的 FPSU 配置 | | C |
| `LUT_phase_<k>_<self>.bin` / `activation_lut_file_<self>.bin` | 激活查找表 | | ? |
| `kantor_A_{scale,bias,Shift}_buffer_file_phase_3_<self>.bin` | Kantor 单元配置 | | G/C |
| `RMSNorm_Add_Const_<self>.bin` | RMSNorm epsilon | | G |
| `updates_{sf,zp}_<self>.bin` | KV_Cache_DMA 写入值量化参数 | | G |
| `{sin,cos}_mul_output_<self>.bin` | RoPE 中间结果 | | G |
| `DEBUG_*_<self>.bin` | 调试副本，**可选，可不落盘** | | G |

**三条一致性规则（实测）**：

1. **生产者/消费者一致**：194/197 满足「生产者 `output_buffer` == 某消费者的 `input_buffer[_slot]`」。
   3 个例外：节点 22（phase 型自命名）、节点 26/32（Split 的输出被绑为下游 MatMul 的
   `weight_buffer_<matmul_id>.bin`，走权重通路）。
2. **phase 型自命名**：37 个带 `rtl_version` 的节点（36 DynamicScaling + 1 Llama2ActivationDQ）
   **37/37** 用 `output_buffer_<self_id>.bin`，不用消费者编号。
3. **动态量化的 scale 跨节点引用**：当上游是 DynamicScaling 时，下游 `input_sf` **直接引用
   上游的 phase_1 输出文件**，不是自己的 sf 文件：

```
node 195 (Gemm gate_proj):
  input_buffer "output_buffer_196.bin"             <- 上游 DQ 的 int8 输出
  input_sf     "output_buffer_phase_1_196.bin"     <- 上游 DQ 的 per-group scale
```

这是动态量化的核心：**scale 是运行时算出来的，作为张量传递，不是编译期常量**。
实测 `input_sf` 的 110 次引用中，有 37 次指向上游的 `output_buffer_phase_1_*.bin`。

**跨语言真源**：这套规则在 pim-compiler 落在 `contracts/gml_names.py`，FlagTree(C++) 侧照抄，
两边必须字节一致，否则后端读到悬空引用。

### 3.4 量化参数类（18 族）

命名规律：`{input|weight|output|updates|bias}[_<slot>]_{sf|zp}[_dtype]`。

| 字段 | 出现 | 含义 | 功能 | 范围 | 计算策略 | 谁 |
| --- | ---: | --- | --- | --- | --- | --- |
| `input_sf` | 110 | 输入激活 scale 文件名 | 反量化 | 文件名 | 静态：本节点自算；动态：**引用上游 `output_buffer_phase_1_*`** | G |
| `input_zp` | 110 | 输入 zero-point | 反量化偏移 | 文件名 | 对称量化，恒 0（文件仍须存在） | G |
| `input_<slot>_sf` / `_zp` | 各 38 | 多输入算子逐槽量化参数 | 同上 | slot 0–31 | 逐槽独立算；Concat 最多 32 组 | G |
| `weight_sf` | 73 | 权重 scale | 反量化 | 文件名 | 见 §4.2 权重量化 | G |
| `weight_zp` | 73 | 权重 zp | — | 恒 0 | 对称量化 | G |
| `output_sf` | 183 | 输出 requant scale | 下游反量化 | 文件名 | 静态算子=下游 `input_sf`；DQ=**逐元素等于本节点 phase_1 输出** | G |
| `output_zp` | 183 | 输出 zp | — | 恒 0 | 对称量化 | G |
| `updates_sf` / `_zp` | 各 2 | KV_Cache_DMA 写入值的量化参数 | KV cache 写入 | 文件名 | 等于被写入 cache 的 sf（实测 0.04589=key、0.00261=value） | G |
| `bias_sf` / `bias_zp` / `bias_buffer` / `bias_buffer_dtype` | 0（本图）| 偏置量化 | — | — | Llama2 **无 bias**，本图不出现；ResNet 样本里有 | G |
| `*_sf_dtype` | 110/73/146 | scale 文件的元素类型 | 解码 bin | `float16`(主) / `float32`(仅 RMSNorm 系列 2 处) | **RMSNorm_vpu → float32，其余 → float16** | G |
| `use_dynamic_quantization` | 72 | 输入走动态量化 | 后端等待运行时 scale | 恒 `1` | MatMul 64 + Gemm 7 + Split 1 置 1 | G |
| `weight_sf_multiplier` | 2 | 权重 scale 乘数 | 次正规数修正 | `2`(节点23) / `4`(节点36) | **实测 = 1/`Scaling_buffer_file`**（见 §4.3） | G |
| `DEBUG_sub_normal_weights_sf` | 2 | 次正规数处理标记 | 调试 | 恒 `1` | 与 `weight_sf_multiplier` 同时出现 | G |
| `DEBUG_weight_buffer_spc` / `_spc_axis` / `_spg` / `_spg_axis` / `_spg_group_size` | 各 7 | 权重侧定标粒度（**是数值不是文件名**） | 记录分组方式 | `1 / 2 / 1 / 3 / 128` | int4 权重固定这组值 | G |

**W4A8 量化契约**。参考产物把它记在 Relay 注解里（我方不走 Relay，这里只作为契约的
证据来源；我方的等价配置见右列）：

| 项 | 参考产物的 Relay 注解 | 我方量化配置 |
| --- | --- | --- |
| 激活 | `simulated_dynamic_quantize(%4, -128f, 127f, spc=True, spg=True, axis=1, group_axis=-1, group_size=128)` | int8 对称、**动态**、per-channel + per-group、group_size=128、group_axis=-1 |
| 权重 | `simulated_quantize(..., -8f, 7f, kind=2, spc=True, spg=True, group_axis=-1, group_size=128)` | int4 对称、**静态**、per-channel + per-group、group_size=128、group_axis=-1 |

即：激活 int8 对称 `[-128,127]`、权重 int4 对称 `[-8,7]`、**两侧都是 per-channel + per-group、
group_size=128**（= Llama2-7B 的 head_dim，非巧合）。

> **注意除数不是 127/7 而是 128/8**：见 §4.5 与 §4.2 —— 满量程分母取 `2^(bits-1)`
> 而非 `2^(bits-1)-1`。这一条 pim-compiler 当前实现与实物不一致，见 §8.6。

### 3.5 数据通路与硬件配置（归属 C，见 §7.1 论证）

这一域是既有文档划给算子编译器的部分。实测**全部由 `(op_type, phase)` 唯一决定**，
无一按节点变化 —— 所以图编译器用一张常量表就能填全。

**顶层（非 phase）算子的配置表**（实测全覆盖，无例外）：

| `op_type` | `nmu_mode` | `fpsu_mode` | `pooling_dtype` | `kantor_mode` | `fpsu_spc/spc_axis/spg` | `transpose` |
| --- | --- | --- | --- | --- | --- | --- |
| `Gemm` (7) | floating_point | floating_point_32 | floating_point | off ×6, fp2int_converter ×1 | 1 / 1 / 0 | — |
| `MatMul` (64) | floating_point | floating_point_32 | floating_point | off | 1 / 1 / 0 | — |
| `KV_Cache_DMA` (2) | — | **fixed_point** | **fixed_point** | off | 1 / 1 / 0 | — |
| `EltwiseAdd` (2) | — | `fpsu_mode_<slot>`=floating_point | `pooling_dtype_<slot>`=floating_point | off | `fpsu_<slot>_spc`=1, `_spg`=0 | — |
| `EltwiseMul` (1) | — | 同上 | 同上 | **elementwise_mul_fp16** | 同上 | 1 |
| `Mask` (32) | — | — | — | off | — | 1 |
| `DynamicScaling` (36) | — | 见 phase 表 | 见 phase 表 | 见 phase 表 | 见 phase 表 | 1 |
| `Softmax` (32) | — | 见 phase 表 | 见 phase 表 | 见 phase 表 | 见 phase 表 | — |
| `Llama2ActivationDQ` (1) | — | 见 phase 表 | 见 phase 表 | 见 phase 表 | 见 phase 表 | 1 |
| `Llama2Activation` (1) | — | 逐子块 | 逐子块 | 逐子块 | 逐子块 | **0** |
| `Split`/`Concat`/`Transpose`/`Reshape` (10) | — | — | — | off | — | — |
| `RMSNorm_vpu` (2) | — | — | — | — | — | — |

| 字段 | 出现 | 含义 | 功能 | 范围 | 计算策略 | 谁 |
| --- | ---: | --- | --- | --- | --- | --- |
| `nmu_mode` | 71 | 累加单元数值模式 | NMU 配置 | 恒 `floating_point` | 上表按 op_type 查 | C |
| `fpsu_mode` | 73 | 定标单元模式 | FPSU 配置 | `floating_point_32`(71) / `fixed_point`(2) | 上表 | C |
| `fpsu_spc` / `_spc_axis` | 73 各 | 是否按通道定标 / 沿哪轴 | 定标粒度 | `1` / `1` | 恒定 | C |
| `fpsu_spg` | 73 | 是否按组定标 | 同上 | 恒 `0` | 恒定（顶层不分组） | C |
| `pooling_dtype` | 73 | 池化通路数据类型 | 通路配置 | `floating_point`(71)/`fixed_point`(2) | 上表 | C |
| `kantor_mode` | 118 | 后置重定标模式 | Kantor 配置 | `off`(116)/`fp2int_converter`(1)/`elementwise_mul_fp16`(1)；PDF 定义 6 种 | 上表 | C |
| `transpose` | 71 | 算子是否以转置起始 | 搬运方式 | `1`(70)/`0`(1，仅 Llama2Activation) | 上表 | C |
| `input_data_extensions` | 193 | 输入数据通道规格 | 有符号性/浮点性 | `1`(82)/`3`(111) | **由 dtype 推**：int8→1，float16→3 | G |
| `output_data_extension` | 193 | 输出同上 | 同上 | `1`(50)/`3`(143) | 同上 | G |
| `weight_format` | 64 | 权重布局 | 矩阵单元操作数布局 | `weights_transpose`(32) / `weight`(32) | **实测完全由角色决定**：matmul1(QK^T)→transpose 32/32，matmul2(PV)→weight 32/32 | C |
| `MatMul_input_as_weight` | 64 | input1 走权重通路 | 标记第二操作数为 B | 恒 `1` | 两操作数都是网络节点时置 1 | G |
| `group_attention_data_num` / `_weight_num` | 64 各 | GQA 分组数 | 注意力分组 | 恒 `"32"` | = `num_key_value_heads`（7B MHA 下 = 头数 32） | G |
| `rtl_version` | 37 | RTL 版本 | 后端选硬件行为 | 恒 `"1.4"` | 仅 phase 型算子写；**须与对方确认目标 RTL** | 先按照当前参照的版本 |
| `flp_min_exp` / `_max_exp` / `_mantisa` | 各 1（顶层） | 窄浮点动态范围 | FLP 单元配置 | `(10,17,3)` | 见 phase 表 | C |
| `activation_mode` | 1 | LUT 激活模式 | LUT 配置 | `0`（regular；1=even，2=odd） | 见 phase 表 | C |
| `activation_special_operators` | 1 | LUT 特殊算子 | 选倒数等特殊通路 | `0` / `4`(=倒数) | 见 phase 表 | C |

**`data_extension` 与 dtype 的绑定**（PDF：1=signed，2=unsigned，3=float）：

| dtype | extension | 出现 |
| --- | ---: | ---: |
| `int8` | 1 | 127 |
| `float16` | 3 | 215 |
| `float16` | **1** | **1 个例外**（Split 21） |

生成时按 dtype 直接推导即可，保留那一个例外的可能性。

### 3.6 phase 流水线（69 个节点，占全部 bin 的 57%）

phase 是硬件把一个大算子拆成的**顺序子流水级**，每级有独立的输入/输出缓冲、FPSU 配置、
Kantor 配置、LUT。**这是文件数量的主要来源**：308 个 `*_phase_*` 文件 × 5 族。

| 算子 | 节点数 | phase 数 | 各级职能（实测反推） |
| --- | ---: | ---: | --- |
| `DynamicScaling` | 36 | 0–3 | p0 求组统计量 → p1 ×1/256 → p2 取倒数 → p3 Kantor fp2int |
| `Llama2ActivationDQ` | 1 | 0–3 | 与 DynamicScaling 同构（RoPE 后接动态量化） |
| `Softmax` | 32 | 0–4 | p0 求 max → p1 exp → p2 求和 → p3 取倒数 → p4 归一化 |

**phase 配置常量表**（实测：同一 `(op_type, phase)` 的所有节点取值完全相同，**无一例外**）：

#### DynamicScaling / Llama2ActivationDQ（37 节点）

| 字段 | phase 0 | phase 1 | phase 2 | phase 3 |
| --- | --- | --- | --- | --- |
| `fpsu_mode_phase_#` | floating_point | floating_point | floating_point | floating_point |
| `fpsu_spc_phase_#` / `_spc_axis_phase_#` | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `fpsu_spg_phase_#` | 0 | 0 | 0 | 0 |
| `pooling_dtype_phase_#` | floating_point | floating_point | floating_point | floating_point |
| `kantor_mode_phase_#` | off | off | off | **fp2int_converter** |
| `input_data_extensions_phase_#` | 3 | 3 | 3 | 3 |
| `output_data_extensions_phase_#` | 3 | 3 | 3 | **1** |
| `input_buffer_dtype_phase_#` | float16 | float16 | float16 | float16 |
| `output_buffer_dtype_phase_#` | float16 | float16 | float16 | **int8** |
| `flp_min_exp/max_exp/mantisa_phase_#` | — | **10 / 17 / 3** | **15 / 15 / 0** | — |
| `LUT_phase_#` | — | 有 | 有 | — |
| `activation_mode_phase_#` | — | **1** | 0 | — |
| `activation_special_operators_phase_#` | — | 0 | **4**（=倒数） | — |
| `global_pooling_spc/spc_axis/spg/spg_axis_phase_0` | **1 / 2 / 1 / 3** | — | — | — |
| `global_pooling_group_size_phase_0` | **128 或 1024**（见下） | — | — | — |
| `kantor_A_spc/spg/scale_axis/spg_axis_phase_3` | — | — | — | **1 / 1 / 2 / 3** |
| `kantor_A_spg_group_size_phase_3` | — | — | — | **= group_size** |

`global_pooling_group_size_phase_0` 是**唯一按节点变化**的 phase 字段，取值由被量化张量决定：

| 节点 | 张量 | group_size | 组数 |
| --- | --- | ---: | ---: |
| 12, 24, 196 | hidden `[1,1,1,4096]` | **128** | 32 |
| 193 | MLP 中间态 `[1,1,1,11008]` | **128** | **86** |
| 22 | RoPE 后的 Q（逐头） | **128** | 32 |
| 17,38,43,…（32 个） | attention scores `[1,1,1,1024]` | **1024** | **1** |

规则：`group_size = 128`（= 量化契约的 group_size），**但当张量最后一维 ≤1024 且是 attention
scores 时整条当一组**（`group_size=1024`）。组数 = `numel / group_size`（11008/128=86，非整除的
边界情况本图未出现）。

#### Softmax（32 节点）

| 字段 | phase 0 | phase 1 | phase 2 | phase 3 | phase 4 |
| --- | --- | --- | --- | --- | --- |
| `nmu_output_type_phase_#` | floating_point | floating_point | floating_point | floating_point | floating_point |
| `fpsu_mode_phase_#` | floating_point | floating_point | floating_point | **floating_point_32** | floating_point |
| `fpsu_spc_phase_#` / `_spc_axis_phase_#` | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| `fpsu_spg_phase_#` / `_spg_axis` / `_spg_group_size` | 0 / -1 / -1 | 0 / -1 / -1 | 0 / -1 / -1 | 0 / -1 / -1 | 0 / -1 / -1 |
| `pooling_dtype_phase_#` | floating_point | floating_point | floating_point | floating_point | floating_point |
| `kantor_mode_phase_#` | off | off | off | off | off |
| `input/output_data_extensions_phase_#` | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| `flp_min_exp/max_exp/mantisa_phase_#` | — | **9 / 16 / 3** | — | **15 / 15 / 0** | — |
| `LUT_phase_#` | — | 有 | — | 有 | — |
| `activation_mode_phase_#` | — | 0 | — | 0 | — |
| `activation_special_operators_phase_#` | — | 0 | — | **4**（=倒数） | — |

注意 Softmax **没有** `global_pooling_*`（求 max 走的是另一条通路），且 `fpsu_spg_axis` /
`_spg_group_size` 显式写 `-1`（DynamicScaling 侧不写这两个）。这两处差异必须照做。

#### phase 字段族清单（完整，10 族 × 5 相）

```
input_buffer_phase_#          output_buffer_phase_#
input_buffer_dtype_phase_#    output_buffer_dtype_phase_#
input_data_extensions_phase_#  output_data_extensions_phase_#
fpsu_mode_phase_#  fpsu_spc_phase_#  fpsu_spc_axis_phase_#
fpsu_spg_phase_#   fpsu_spg_axis_phase_#  fpsu_spg_group_size_phase_#
Scaling_buffer_phase_#  Scaling_PS_buffer_phase_#  Bias_buffer_phase_#
kantor_mode_phase_#     pooling_dtype_phase_#
nmu_output_type_phase_#           （仅 Softmax）
global_pooling_*_phase_0          （仅 DQ）
LUT_phase_#  flp_min_exp_phase_#  flp_max_exp_phase_#  flp_mantisa_phase_#
activation_mode_phase_#  activation_special_operators_phase_#
kantor_A_*_phase_3                （仅 fp2int 的 phase 3）
```

### 3.7 算子专属字段

| 字段 | 所属 | 含义 | 范围 | 计算策略 | 谁 |
| --- | --- | --- | --- | --- | --- |
| `axis` | Concat/Split/Softmax | 操作轴 | `3`(32, Softmax) / `1`(4) | Softmax 沿最后轴=3；Split/Concat 按头=1 | G |
| `axes` | Transpose | 轴置换 | `"[0, 2, 1, 3]"`(3) / `"[0, 1, 3, 2]"`(1) | 取 fx 节点 `aten.permute`/`aten.transpose` 的 dims 参数，按 `"[a, b, c, d]"` 格式化 | G |
| `num_heads` | Split / RoPE | 注意力头数 | 恒 `"32"` | = config.json `num_attention_heads` | G |
| `split_channel_number` | 32 个 matmul1 | **head 编号** | `0`–`31` | 逐头展开时的头下标；= 层参数文本的 `Split Head Index` | G |
| `original_shape` | DynamicScaling | 量化前形状 | `"[1, 1, 1, 4096]"` / `[...11008]` / `[...1024]` | 输入张量 shape 的字符串 | G |
| `output_shape_by_group` | DynamicScaling | 分组后形状 | `"[1,1,1,32,128]"` / `[...86,128]` / `[...1,1024]` | 把最后一维拆成 `(组数, group_size)` | G |
| `dq_contraction` | 节点 22 | DQ 融合标记 | 恒 `1` | RoPE 与 DQ 融合时置 1 | G |
| `use_input_buffer_1` | KV_Cache_DMA | 索引张量不走 L2A | 恒 `"L2A_ignore"` | scatter 的 indices 槽固定此值 | C |
| `RMSNorm_Add_Const` | RMSNorm_vpu | epsilon 文件名 | 文件名 | 见 §4.6 | G |
| `Use_Scaling` | RMSNorm_vpu | 是否启用缩放 | 恒 `0` | 常量 | C |
| `Vpu_Axis` | RMSNorm_vpu (vpu_params 内) | VPU 归约轴 | 恒 `-1` | 常量（-1=最后轴） | C |
| `sin_mul_output` / `cos_mul_output` (+`_dtype`) | RoPE | 中间结果缓冲 | 文件名，8192B | 见 §4.7 | G |
| `activation_op_type` | contraction 内 | 融合的激活类型 | `"Silu"` | 由被融合算子决定 | G |
| `activation_lut_file` / `lut_debug` | Gemm 195 | 融合 SiLU 的 LUT | 文件名 | 见 §4.8 | ? |
| `weights_scaling_buffer_file` / `Weights_buffer_file` / `bias_buffer_file` / `input_scale_factor_buffer` / `output_scale_factor_buffer` | vpu_params 内 | **指向已有文件的别名** | 文件名 | 直接复用 `weight_sf_<id>` / `weight_buffer_<id>` / `RMSNorm_Add_Const_<id>` / `input_sf_<id>` / `output_sf_<id>` | G |

**两个嵌套子块**（GML 里唯一的层级结构，序列化器必须支持）：

```
vpu_params [                                  contraction [
  Vpu_Axis -1                                   fused_Silu_act [
  input_scale_factor_buffer "input_sf_25.bin"      name "Silu_act"
  output_scale_factor_buffer "output_sf_25.bin"    op_type "Lut"
  Weights_buffer_file "weight_buffer_25.bin"       activation_op_type "Silu"
  weights_scaling_buffer_file "weight_sf_25.bin"   residual_input_buffer 195
  bias_buffer_file "RMSNorm_Add_Const_25.bin"    ]
]                                             ]
```

`contraction` 的内层块名格式为 `fused_<算子名>`（PDF 示例中为 `fused_layer4_..._Relu_qidx167`，
本图为 `fused_Silu_act`）。**融合是强制的**：不融合的图对方后端读不了。

### 3.8 RoPE 子块（Llama2Activation / DQ，2 节点 67 文件）

RoPE 用一组**固定命名子块**描述四路运算，PDF 第 11–13 页有官方定义。子块名：
`Llama2Activation_Add_Cos`、`Add_Sin`、`Sin`、`Cos`（Sin/Cos 再分 `_Broadcast`），另有 `_add`。

每路的字段族（`<U>` 为单元号 1–6，与子块一一绑定）：

| 单元号 | 子块 | 字段前缀 |
| ---: | --- | --- |
| 1 | `Add_Cos` | `Scaling_buffer_file_1_Llama2Activation_Add_Cos`、`fpsu_mode_1_...`、`fpsu_1_spc_...`、`pooling_dtype_1_...` |
| 2 | `Add_Sin` | 同构 |
| 3, 4 | `Sin` | 同构 + `kantor_mode_Llama2Activation_Sin` = `elementwise_mul_fp16` + `Kantor_A/B_*` |
| 5, 6 | `Cos` | 同构 + `kantor_mode_Llama2Activation_Cos` = `elementwise_mul_fp16` + `Kantor_A/B_*` |

| 字段族 | 出现 | 范围 | 谁 |
| --- | ---: | --- | --- |
| `Llama2Activation_<块>_sf` / `_zp` / `_sf_dtype` | 12 各 | 文件名 / `float16` | G |
| `Scaling_buffer_file_<U>_Llama2Activation_<块>` | 12 | 文件名（2B fp16 = 1.0） | C |
| `Scaling_PS_buffer_file_<U>_Llama2Activation_<块>` | 12 | 文件名（1B = 0） | C |
| `fpsu_mode_<U>_Llama2Activation_<块>` | 12 | 恒 `floating_point` | C |
| `fpsu_<U>_spc_...` / `_spg_...` / `_spg_axis_...` / `_spg_group_size_...` | 12 各 | `1 / 0 / -1 / -1` | C |
| `fpsu_<U>_scale_axisLlama2Activation_<块>` | 12 | 恒 `1`（**注意缺下划线的 bug**） | C |
| `pooling_dtype_<U>_Llama2Activation_<块>` | 12 | 恒 `floating_point` | C |
| `kantor_mode_Llama2Activation_<块>` | 6 | `elementwise_mul_fp16`(4) / `off`(1) / `fp2int_converter`(1) | C |
| `Kantor_A/B_Llama2Activation_<块>_{scale,bias}_buffer_file` | 5+4 / 1+4 | 文件名 | C |
| `Kantor_A/B_Shift_Llama2Activation_<块>` | 5 / 4 | 文件名（1B int8） | C |
| `Kantor_A/B_{spc,spg,spg_axis,spg_group_size}_Llama2Activation_<块>` | 各 4–5 | `1 / 0 / -1 / -1` | C |

**`kantor_mode_Llama2Activation_add` 区分两个 RoPE 节点**：节点 22（DQ 版，Q 路）为
`fp2int_converter` 且额外带 `Kantor_A_Llama2Activation_add_{scale,bias}_buffer_file`；
节点 30（非 DQ 版，K 路）为 `off`。这是 Q 路要量化、K 路不量化的直接体现。

---

## 4. `.bin` 文件族：字节格式与计算方法

这一章是生成器必须精确复现的部分。所有格式均用 `struct` 逐字节解码验证。

### 4.0 元素格式总表（76 族，按字节宽度分类）

| 元素类型 | 字节/元素 | struct | 用在哪些族 |
| --- | ---: | --- | --- |
| `float16` | 2 | `<e` | 所有 `*_sf`、`Scaling_buffer_*`、`kantor_*_scale_*`、`LUT_*`、fp16 数据缓冲 |
| `float32` | 4 | `<f` | `Bias_buffer_*`、`kantor_*_bias_*`、`RMSNorm_Add_Const`、RMSNorm 的 `*_sf`、**Softmax 的归约相输出** |
| `int32` | 4 | `<i` | 所有 `*_zp` |
| `int8` | 1 | `<b` | int8/int4 权重、int8 数据缓冲、`kantor_*_Shift_*` |
| `uint8` | 1 | `<B` | `Scaling_PS_buffer_*` |

**三个最容易搞错的点**：

1. **int4 权重不打包**。4096×4096 的 int4 权重占 **16777216 字节 = 每元素整整 1 字节**，
   值域限制在 `[-8,7]` 但存储用 int8。**不是每字节两个 nibble**。
2. **`Bias_buffer_*` 是 fp32 且不恒为 0**。见 §4.4 —— 这是既有文档的错误。
3. **`*_zp` 恒为 4 字节的 0**，但文件必须存在（否则 GML 悬空引用）。W4A8 是对称量化，
   zp 不承载信息。

### 4.1 数据缓冲区族

| 族 | 数量 | 元素格式 | 元素数公式 | 数值来源 | 谁 |
| --- | ---: | --- | --- | --- | --- |
| `input_buffer_<c>` | 153 | 由 `input_buffer_dtype` 定 | `prod(shape)` | 上游算子的输出（运行时数据；编译期落盘的是**参考输入**） | G |
| `input_buffer_<slot>_<c>` | 114 | 由 `input_buffer_<slot>_dtype` 定 | 同上 | 同上 | G |
| `output_buffer_<self>` | 37 | 由 `output_buffer_dtype` 定 | 同上 | 仅 phase 型算子；= `output_buffer_phase_<末相>` 的内容 | G |
| `input_buffer_phase_<k>_<self>` | 308 | 见 §4.5 | 见 §4.5 | 上一相的输出 | G |
| `output_buffer_phase_<k>_<self>` | 308 | 见 §4.5 | 见 §4.5 | 本相的计算结果 | G |

**实测尺寸对照（验证形状推导正确）**：

| 文件 | 字节 | 解释 |
| --- | ---: | --- |
| `input_buffer_25.bin` | 8192 | 4096 fp16 = hidden_size × 2 |
| `input_buffer_0_28.bin` | 4194304 | 32×1024×128 int8 = KV cache 全量 |
| `input_buffer_18.bin` | 2048 | 1024 fp16 = attention scores 单头 |
| `output_buffer_12.bin` | 4096 | 4096 int8 = 量化后 hidden |
| `output_buffer_193.bin` | 11008 | 11008 int8 = MLP 中间态 |

> **重要**：这些数据缓冲的**内容**是运行时张量，参考产物里装的是对方的合成测试数据
> （证据见 §8）。我方生成时应装**真实 Llama2-7B 的一次前向的中间值**，或按对方约定装占位。
> **字节数与 dtype 必须对**，内容按用途选。

### 4.2 权重族

| 族 | 数量 | 元素格式 | 元素数 | 计算方法 | 谁 |
| --- | ---: | --- | ---: | --- | --- |
| `weight_buffer_<id>` (int4) | 7 | int8，1 字节/元素，值域 `[-8,7]` | `out×in` | 见下 | G |
| `weight_buffer_<id>` (int8, MatMul) | 64 | int8 | `128×1024` | KV cache 的一头切片 | G |
| `weight_buffer_<id>` (int8, RMSNorm) | 2 | int8 | `4096` | RMSNorm weight per-tensor 量化 | G |
| `weight_sf_<id>` (int4) | 7 | float16 | `out×in/128` | 见下 | G |
| `weight_sf_<id>` (int8 MatMul) | 64 | float16 | 1 | per-tensor，= KV cache 的 sf | G |
| `weight_sf_<id>` (RMSNorm) | 2 | **float32** | 1 | 实测 `0.007874 = 1/127` | G |
| `weight_zp_<id>` | 73 | int32 | 1 | 恒 0 | G |

**int4 per-group 权重量化（W4 的核心，实测验证）**：

```python
# W: [out_features, in_features]，group_size=128 沿 in_features（最后轴）
G = in_features // 128
absmax[o, g] = max(|W[o, g*128:(g+1)*128]|)
weight_sf[o, g] = absmax[o, g] / 8                    # fp16 落盘
q[o, i]         = clamp(round(W[o,i] / weight_sf[o, g]), -8, 7)   # int8 落盘，1B/元素
```

实测验证（节点 195，`mlp_gate_proj`）：

| 检查 | 结果 |
| --- | --- |
| `weight_buffer_195` 大小 | 45088768 = 11008×4096 ✓ |
| 值域 | `[-8, 7]` ✓ |
| `weight_sf_195` 元素数 | 352256 = 11008×4096/128 ✓ |
| 每组 `max|q| ∈ {7,8}` | **125/128** 组满足（3 组不满足 → 该组权重分布使 round 未触边界，正常） |
| `sf × 8` 与组 absmax | 逐组吻合 |

**Llama2-7B 维度对照（7/7 全部吻合，这是「从模型算出来」的依据）**：

| 权重节点 | 模型张量 | 形状 | 实测字节 | 公式 | ✓ |
| --- | --- | --- | ---: | --- | --- |
| 23 | `self_attn.q_proj.weight` | 4096×4096 | 16777216 | `H×H` | ✓ |
| 31 | `self_attn.k_proj.weight` | 4096×4096 | 16777216 | `H×H` | ✓ |
| 36 | `self_attn.v_proj.weight` | 4096×4096 | 16777216 | `H×H` | ✓ |
| 11 | `self_attn.o_proj.weight` | 4096×4096 | 16777216 | `H×H` | ✓ |
| 195 | `mlp.gate_proj.weight` | 11008×4096 | 45088768 | `I×H` | ✓ |
| 198 | `mlp.up_proj.weight` | 11008×4096 | 45088768 | `I×H` | ✓ |
| 192 | `mlp.down_proj.weight` | 11008×4096 | 45088768 | `I×H`（转置前） | ✓ |
| 25 | `input_layernorm.weight` | 4096 | 4096 | `H` | ✓ |
| 197 | `post_attention_layernorm.weight` | 4096 | 4096 | `H` | ✓ |

其中 `H=hidden_size=4096`、`I=intermediate_size=11008`、`heads=32`、`head_dim=128=group_size`。

**64 个 MatMul 的 int8 「权重」不是模型权重**，而是 **KV cache 的单头切片**（
`128×1024 = head_dim × max_seq_len`）：matmul1 吃 Kᵀ（`weight_format=weights_transpose`），
matmul2 吃 V（`weight_format=weight`）。其 `weight_sf` 是 per-tensor 单值，实测
matmul1 = 0.04589（= key cache sf）、matmul2 = 0.00261（= value cache sf），与 `IO_info.txt` 一致。

### 4.3 FPSU 定标三族（最易错，既有文档有误）

三个文件一组，挂在每个计算算子上，是 FPSU（定标单元）的配置。**本次修订：三个文件的
字节宽度由硬件规范 §4.3.2 独立确认** —— FPSU 的操作序列是「加 **32 位** bias → 乘
**16 位**有符号 scale → round → **右移** → 饱和到 16 位输出」，三个文件正是这三个操作数：

| 族 | 数量 | 字节 | 元素格式 | FPSU 操作数（规范 §4.3.2） | 实测取值 | 谁 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `Bias_buffer_file_<id>` | 73+6 | **4** | **float32** | **32 位 bias**（定点模式下 int32） | 恒 `0.0`（79/79） | C |
| `Scaling_buffer_file_<id>` | 73+6 | **2** | float16 | **16 位有符号 scale**（定点模式下 int16） | **5 种值，见下** | C/G |
| `Scaling_PS_buffer_file_<id>` | 73+6 | **1** | uint8 | **右移位数（Post-Shift）** | `0`(393) / **`14`(2)** | C |

> **`Scaling_PS` 的 `PS` = Post-Shift（后置右移量），不是 Partial Sum。**
> 判据：393/395 个文件为 0，而浮点 FPSU 通路不需要右移；唯一非零的两个（值 14）
> 正是全图唯一 `fpsu_mode="fixed_point"` 的两个 KV_Cache_DMA 节点。规范 §10.4
> 对定点模式明确写「Scale dtype is int16, bias dtype is int32」，与 2B/4B 完全一致。

**`Scaling_buffer_file` 的 5 种取值（这是关键修正）**：

| 值 | 数量 | 出现在 | 含义与来源 |
| --- | ---: | --- | --- |
| `1.0` | 55 | 大多数算子 | 无额外缩放 |
| **`0.08837890625`** | **32** | 全部 matmul1（QK^T） | **= 1/√128 = 1/√head_dim**，即 attention 的 scale。实测与 `1/sqrt(128)=0.0883883` 吻合 |
| `0.5` | 1 | 节点 23（q_proj） | = 1/`weight_sf_multiplier`(=2) |
| `0.25` | 1 | 节点 36（v_proj） | = 1/`weight_sf_multiplier`(=4) |
| `2.0` | 2 | 节点 28/33（KV_Cache_DMA） | 与 `Scaling_PS=14` 配对，定点通路的定标 |

**计算策略**：

```python
if 是 matmul1(QK^T):        Scaling_buffer_file = 1/sqrt(head_dim)   # = 0.088388 for 7B
elif 有 weight_sf_multiplier: Scaling_buffer_file = 1/weight_sf_multiplier
elif op_type == KV_Cache_DMA: Scaling_buffer_file = 2.0 ; Scaling_PS = 14
else:                        Scaling_buffer_file = 1.0
Scaling_PS_buffer_file = 0    # 除 KV_Cache_DMA
Bias_buffer_file       = 0.0  # fp32，全部
```

> **`1/√d` 落在 `Scaling_buffer_file` 里，是本文最实用的发现之一。**
> 参考产物的 Relay 里 QK^T 后有 `/11.3137`（=√128）。这个除法**没有单独成节点**，
> 而是被折叠进 matmul1 的 FPSU 定标系数。既有文档把这一族当成恒 1.0 的常量，
> 那样生成的图会丢掉 attention scale，数值全错。
> 旁证：32 个 matmul1 节点各有一个 `DEBUG_div_value_<id>` 字段引用（未落盘，属 DEBUG 族）。

**`weight_sf_multiplier` 的作用闭环**：当某层权重的 per-group scale 落进 fp16 次正规数区间
（< 6.1e-5）时，把 `weight_sf` 整体乘 `multiplier` 抬进正规数区，再在 FPSU 用
`Scaling_buffer_file = 1/multiplier` 补偿回来。实测两处：

| 节点 | `weight_sf_multiplier` | `Scaling_buffer_file` | 乘积 |
| --- | ---: | ---: | ---: |
| 23 (q_proj) | 2 | 0.5 | 1.0 ✓ |
| 36 (v_proj) | 4 | 0.25 | 1.0 ✓ |

### 4.4 phase 版定标三族（308×3 个文件，含关键常量）

| 族 | 数量 | 字节 | 元素格式 | 元素数 | 谁 |
| --- | ---: | ---: | --- | --- | --- |
| `Scaling_buffer_phase_<k>_<id>` | 308 | 2 或 64 | float16 | 1 或 32 | C |
| `Scaling_PS_buffer_phase_<k>_<id>` | 308 | 1 或 32 | uint8 | 1 或 32 | C |
| `Bias_buffer_phase_<k>_<id>` | 308 | 4 或 128 | **float32** | 1 或 32 | C |

**取值表**（实测按 `(op_type, phase)` 确定）。**关键区分：编译期常量 vs 运行时归约落点**
—— 后者不是常量，是硬件在运行时写入的位置，编译期只需按正确字节宽度分配：

| 算子 | phase | `Scaling_buffer_phase` | `Scaling_PS_buffer_phase` | `Bias_buffer_phase` | 性质 |
| --- | ---: | ---: | ---: | --- | --- |
| DynamicScaling / DQ | 0 | `1.0` | `0` | **`2⁻⁶³`** = 1.0842021724855044e-19 | **常量**（36/36 节点字节相同） |
| DynamicScaling / DQ | 1 | **`0.00390625`** = 1/256 | `0` | `0.0` | 常量 |
| DynamicScaling / DQ | 2 | `1.0` | `0` | `0.0` | 常量 |
| DynamicScaling / DQ | 3 | **`256.0`** | `0` | `0.0` | 常量 |
| Softmax | 0 | `1.0` | `0` | `0.0` | 常量 |
| Softmax | 1 | **`0.5`** | `0` | **= phase0 的输出（`-max`）** | **运行时归约落点** |
| Softmax | 2 | `1.0` | `0` | `0.0` | 常量 |
| Softmax | 3 | `1.0` | `0` | `0.0` | 常量 |
| Softmax | 4 | **= phase3 的输出（`1/Σexp`）** | `0` | `0.0` | **运行时归约落点** |

**五点说明**：

1. **`Bias_buffer_phase_0` = 2⁻⁶³（DQ）是真常量**：36/36 个 DQ 节点字节完全相同
   （`00000020`），且它位于流水线**首**相，没有前序归约可承接。作用是给 absmax 加一个
   极小正数，避免全零组在 phase2 取倒数时溢出成 inf。**写 0.0 会让全零组产生 inf。**
   旁证：这个值**只有按 fp32 解码才是 2⁻⁶³**，按 2×fp16 解出来是 `(0.0, 0.0078125)` 无意义
   —— 这是 `Bias_buffer_*` 为 fp32 的决定性证据，且与硬件规范「bias dtype is int32 / 32-bit
   bias」的宽度一致。

2. **`Bias_buffer_phase_1`（Softmax）不是常量 -30.75，而是 phase0 的归约输出**。
   判据：它与 `output_buffer_phase_0_<id>.bin` 在 **32/32 个 Softmax 节点上逐字节相同**，
   且该值恰为 `-max(x)`。参考产物里显示为 -30.75 只是因为其合成输入的 max 使然。
   硬件依据：FPSU 的 bias 操作数就是前一相的归约结果（规范 §4.3.2），而规范 §4.2 说
   动态量化「通过测量运行时动态范围来修正 scale/shift」。
   **生成时不要硬编码 -30.75**；按 4 字节分配、让 phase0 写入即可。
   （本文一次修订误判为常量，此处更正。）

3. **`Scaling_buffer_phase_4`（Softmax）同理**：逐字节等于 `output_buffer_phase_3_<id>.bin`
   （倒数相的输出 `1/Σexp`），32/32 节点吻合。归一化因子在运行时算出后直接作为
   phase4 的 FPSU scale。

4. **`Scaling_buffer_phase_1` = 1/256、`_phase_3` = 256.0（DQ）**：对应量化公式里的
   `/256` 与 `×256`（见 §4.5）。256 = 2⁸，由 `kantor_A_Shift = -8` 编码，硬件用移位实现
   （规范 §4.3.2 的「round and right-shift」）。

5. **`Scaling_PS_buffer_phase` 全为 0**：308/308 个文件。因为所有 phase 的 `fpsu_mode`
   都是浮点（`floating_point` / `floating_point_32`），浮点通路不需要后置右移。
   非零的后置右移只出现在唯一的定点 FPSU 节点（KV_Cache_DMA，值 14），见 §4.3。

> **对生成器的实际影响**：这三族在 GML 里都必须有文件名引用，且文件必须落盘（否则悬空引用）。
> 「常量」行按表里的值写死；「运行时归约落点」行按正确字节宽度写占位即可（写什么值都会被
> 运行时覆盖）——但为了与参考产物对齐、也为了让离线对拍可跑，建议写入我方自己前向算出的
> 对应值。

节点 22（RoPE DQ）的这三族是 **32 元素向量**而非标量（64B/32B/128B），因为它逐头（32 头）
各有一份配置；值仍是上表的常量，只是重复 32 次。

### 4.5 phase 中间缓冲：完整数学（本文的核心推导）

#### DynamicScaling（4 相），以节点 12 为例（输入 `[1,1,1,4096]` fp16，group_size=128 → 32 组）

| 文件 | 字节 | 元素 | 含义 | 实测关系 |
| --- | ---: | ---: | --- | --- |
| `input_buffer_phase_0_12` | 8192 | 4096 fp16 | 原始输入 | = `input_buffer_12` |
| `output_buffer_phase_0_12` | 64 | **32 fp16** | 每组统计量 | **= 2 × per-group absmax**（32/32 精确） |
| `input_buffer_phase_1_12` | 64 | 32 fp16 | = phase0 输出 | |
| `output_buffer_phase_1_12` | 64 | 32 fp16 | 反量化 scale | **= phase_0 / 256** |
| `output_buffer_phase_2_12` | 64 | 32 fp16 | 量化倒数 scale | **= 1 / phase_0** |
| `input_buffer_phase_3_12` | 8192 | 4096 fp16 | 原始输入（再次） | |
| `output_buffer_phase_3_12` | 4096 | **4096 int8** | 量化结果 | **= round(x × inv × 256)** |
| `output_buffer_12` | 4096 | 4096 int8 | 最终输出 | = phase_3 输出 |
| `output_sf_12` | 64 | 32 fp16 | 输出 scale | **逐字节等于 phase_1** |
| `kantor_A_scale_buffer_file_phase_3_12` | 64 | 32 fp16 | Kantor 量化 scale | **逐字节等于 phase_2** |
| `kantor_A_Shift_buffer_file_phase_3_12` | 32 | 32 int8 | 移位量 | 恒 **-8** |
| `kantor_A_bias_buffer_file_phase_3_12` | 128 | 32 **fp32** | Kantor bias | 恒 0.0 |

**完整公式**（g 为组号，实测 4096/4096 元素在 ±1 内吻合）：

```python
absmax[g] = max(|x[i]|)  for i in group g       # phase 0：Pooling 块，对称 DR
p0[g]     = 2 * absmax[g]                       # phase 0 输出（fp16）：左移 1 位
scale[g]  = p0[g] / 256                         # phase 1 输出 == output_sf
inv[g]    = 1 / p0[g]                           # phase 2 输出 == kantor_A_scale
q[i]      = clamp(round(x[i] * inv[g] * 256), -128, 127)   # phase 3：Kantor fp2int
```

等价形式 `q[i] = round(x[i] / scale[g])`（实测同样 4096/4096 通过）。

**`×2` 与 `/256` 的由来（本次修订用硬件规范解释清楚）**：

规范 §4.3.4 说 Pooling 块做动态量化时「算 abs(max(x))（对称 DR），再算前导零并据此
**左移** DR」。左移 1 位就是 ×2。把两步代入：

```
q[i] = x[i] × (1/p0[g]) × 256 = x[i] × 256/(2·absmax[g]) = x[i] × 128/absmax[g]
```

**即 absmax 正好映射到 128 —— int8 的满量程。** `×2` 存在的目的就是让固定的 `×256`
（硬件移位，`kantor_A_Shift = -8`）之后落在 128 而不是 256。等价地：

```
output_sf[g] = p0[g]/256 = absmax[g]/128
```

实测 **32/32 组**吻合 `output_sf == absmax/128`。量化结果用满了 `[-128, 127]` 全域
（实测 min=-128、max=127，15/32 组的正峰到 127，其余组的负峰到 -128）。

> 所以此前「分母是 256 而不是 127，牺牲约 0.5 bit 动态范围」的说法**不准确**：
> 真实的满量程分母是 **128**，`×2` 与 `/256` 合起来正好等价于除以 `absmax/128`。
> 与理论最优的 127 相比只差 1 个 LSB，那是二补数 int8 固有的不对称（负侧多一格），
> 不是精度损失。实测反量化相对误差 0.70%，符合 int8 per-group 的预期量级。

单组情形（32 个 attention scores 的 DQ，group_size=1024 → 1 组）实测同样成立：
节点 17 的 `output_buffer_phase_0` 仅 2 字节（1 组 × fp16），
`p0 = 0.0230560 = 2×absmax(0.0115280)` ✓，`p1 = p0/256` ✓，`p2 = 1/p0` ✓，
`q = round(x×p2×256)` **1024/1024 吻合** ✓。

**元素数公式**：

```
n_groups = numel / group_size
output_buffer_phase_0/1/2 : n_groups 个 fp16          -> 2*n_groups 字节
input_buffer_phase_0/3    : numel 个 fp16             -> 2*numel 字节
output_buffer_phase_3     : numel 个 int8             -> numel 字节
kantor_A_scale_phase_3    : n_groups 个 fp16          -> 2*n_groups 字节
kantor_A_Shift_phase_3    : n_groups 个 int8          -> n_groups 字节
kantor_A_bias_phase_3     : n_groups 个 fp32          -> 4*n_groups 字节
```

实测核对：节点 193（11008 元素、86 组）→ `output_sf` 172B = 86×2 ✓、
`output_buffer_phase_0` 172B ✓、`kantor_A_Shift_phase_3` 86B ✓、`kantor_A_bias` 344B = 86×4 ✓。

#### Softmax（5 相），以节点 18 为例（`[1,1,1,1024]` fp16 单头 attention scores）

| 文件 | 字节 | 元素 | 含义 | 实测值/关系 |
| --- | ---: | ---: | --- | --- |
| `input_buffer_phase_0_18` | 2048 | 1024 fp16 | 输入 scores | max = 2.98047 |
| `output_buffer_phase_0_18` | **4** | 归约结果 | `-max(x)` | fp16 读高半 = **-2.98047 = -max(x)** 精确吻合 |
| `input_buffer_phase_1_18` | 2048 | 1024 fp16 | 输入（再次） | |
| `output_buffer_phase_1_18` | 2048 | 1024 fp16 | **exp(x - max)** | max(p1)=1.0（在 max 元素处 exp(0)=1）；相对误差 mean 4.8%（LUT 近似） |
| `input_buffer_phase_2_18` | 2048 | 1024 fp16 | = phase1 输出 | |
| `output_buffer_phase_2_18` | **4** | 归约结果 | **Σexp** | **fp32 读 = 84.875，Σ(p1) = 84.8706 精确吻合** |
| `input_buffer_phase_3_18` | 4 | 同上 | = phase2 输出 | 字节相同 |
| `output_buffer_phase_3_18` | 2 | 1 fp16 | **1/Σexp** | `0.0117798`，`1/84.875 = 0.0117820`，差 1 fp16 ULP 内 ✓ |
| `input_buffer_phase_4_18` | 2048 | 1024 fp16 | = phase1 输出 | |
| `output_buffer_phase_4_18` | 2048 | 1024 fp16 | **softmax 结果** | **= p1 × p3，1024/1024 逐元素吻合**；Σ = 0.99975 ✓ |

**完整公式**：

```python
m    = max(x)                          # phase 0 → 落盘 -m
e[i] = exp(x[i] - m)                   # phase 1（FPSU 仿射 0.5x-30.75 + LUT）
S    = Σ e[i]                          # phase 2（fp32 归约）
r    = 1 / S                           # phase 3（LUT special_operators=4 取倒数）
y[i] = e[i] * r                        # phase 4（FPSU scale = r）
```

`Scaling_buffer_phase_4` 逐字节等于 `output_buffer_phase_3`，即归一化因子直接作为 FPSU 定标喂进 phase4。

**归约相的字节宽度规则（实测）**：

| 算子 | 归约相 | 字节 | 解释 |
| --- | --- | ---: | --- |
| DynamicScaling | phase0 / 2 | `2 × n_groups` | 每组一个 fp16 |
| Softmax | phase0 / 2 | **4** | 单个 fp32 |

Softmax 的 4 字节按 **fp32** 解：phase2 = 84.875 = Σexp，**数值上决定性吻合**。
与之一致的是 `fpsu_mode_phase_3 = floating_point_32`（唯一一处 fp32 模式）。

> **一处残留歧义（不影响生成）**：Softmax `output_buffer_phase_0` 的 4 字节
> `0000f6c1` 同时满足两种读法 —— 按 fp32 = **-30.75**（恰好等于 `Bias_buffer_phase_1` 的值，
> 字节完全相同），按 fp16 取高半 = **-2.98047**（恰好等于 `-max(x)`）。
> 巧合来自 fp32 与 fp16 的位布局：`fp32(-30.75)` 的高 2 字节正是 `fp16(-2.98047)`。
> 单一合成样本无法区分。**生成时的安全做法**：按 fp32 写 `-max(x)` 的 fp16 位模式左移 16 位
> （即写 `struct.pack('<HH', 0, fp16_bits(-max))`），这样两种读法都得到正确语义。
> 建议向对方确认归约相的字节宽度定义。

### 4.6 标量常量族

| 族 | 数量 | 字节 | 格式 | 实测值 | 计算策略 | 谁 |
| --- | ---: | ---: | --- | --- | --- | --- |
| `*_zp`（全部） | 501 | 4 | int32 | **恒 0** | 对称量化，写 0 | G |
| `input_sf` / `output_sf`（per-tensor，静态算子） | 多数 | 2 | float16 | 各异 | 见 §4.1/§4.5 | G |
| `input_sf` / `output_sf`（**per-group，DQ 节点**） | 5 各 | **2×组数** | float16 | `output_sf_12`=64B(32 组)、`output_sf_193`=172B(86 组) | 见 §4.5 | G |
| `input_sf_25` / `output_sf_25`（RMSNorm） | 2 各 | 4 | **float32** | `1.0` | RMSNorm 系列用 fp32 | G |
| `RMSNorm_Add_Const_<id>` | 2 | 4 | float32 | **1e-05** | **= config.json 的 `rms_norm_eps`** | G |
| `weight_sf_25/197`（RMSNorm） | 2 | 4 | float32 | `0.007874` | **= 1/127**（int8 对称满量程） | G |
| `kantor_A_Shift_<id>` / `_phase_3_<id>` | 39 | 1×n | int8 | **恒 -8** | = -log2(256)，量化移位 | C |
| `kantor_A_bias_buffer_file*` | 39 | 4×n | float32 | 恒 0.0 | 常量 | C |
| `kantor_A_scale_buffer_file_36` | 1 | 2 | float16 | `383.25` | = 1/absmax 类的量化倒数 | G |
| `Kantor_A/B_Shift_Llama2Activation_*` | 5/4 | 1 | int8 | `0` | RoPE 的 fp16 逐元素乘不移位 | C |
| `Kantor_B_Llama2Activation_*_scale_buffer_file` | 4 | 2 | float16 | `1.0` | 常量 | C |
| `Llama2Activation_*_sf` | 12 | 2 | float16 | `1.0` | cos/sin 表为 fp16 原值，sf=1 | G |
| `updates_sf_28` / `_33` | 2 | 2 | float16 | `0.04589` / `0.00261` | = key/value cache 的 sf | G |

### 4.7 RoPE 中间缓冲

| 族 | 数量 | 字节 | 元素 | 含义 | 计算策略 | 谁 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `sin_mul_output_<id>` | 2 | 8192 | 4096 fp16 | `x_rot × sin` 的中间结果 | `heads×head_dim = 32×128 = 4096` fp16 | G |
| `cos_mul_output_<id>` | 2 | 8192 | 4096 fp16 | `x × cos` 的中间结果 | 同上 | G |
| `<label>_cos` / `_sin` | 2 各 | 8192 | 4096 fp16 | RoPE 的 cos/sin 表 | 由 `position_ids` 与 `rope_theta=10000` 算出 | G |

`<label>_cos` 的文件名用**节点 label 全名**而非 node_id（例
`self_attn_Reshape_qidx4_params_22_cos.bin`），是命名规则的一个例外。

### 4.8 LUT 族（本次修订：结构已破解，仅剩 exp 的段索引待定）

139 个 LUT 文件（138 `LUT_phase_*` + 1 `activation_lut_file`）实测**只有 4 种不同内容**：

| md5 前缀 | 文件数 | 出现在 | `activation_mode` | `special_op` | `flp(min,max,mantisa)` | 语义 | 状态 |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| `17d2387467` | 69 | DQ phase2 / Softmax phase3 | 0 | **4** | 15/15/0 | **倒数 1/x** | **已破解** |
| `9e077580ba` | 37 | DQ phase1 | **1** | 0 | 10/17/3 | **恒等表** | **已破解**（单项） |
| `4fc44231e4` | 32 | Softmax phase1 | 0 | 0 | 9/16/3 | **exp** | 结构已知，段索引待定 |
| `c133ac07eb` | 1 | Gemm 195 融合 | 0 | 0 | 10/17/3 | **SiLU** | 结构已知，定域 `[-4,4)` |

#### 表结构（由硬件规范 §4.3.3 确定）

规范原文：「PWL LUT 支持 **32 段**，每段由一个 **slope** 和一个 **intercept** 定义」。
288 字节 = 144 个 fp16，实测布局与之精确吻合：

```
[0:32]     32 个 fp16 —— slope     A[i]      （第 32 项恒 0，实际用 31 段）
[32:64]    32 个 fp16 —— intercept B[i]      （第 32 项恒 0）
[64:104]   40 个 fp16 —— 未初始化残留，非参数（见下）
[104:144]  40 个 fp16 —— 填充，通常为 0（倒数表在 index 104 有 1 项 1e-05 残留）
```

**求值形式：`y = A[i]·x + B[i]`**，其中 `i` 是段索引。

**`[64:104]` 不是参数，是未初始化内存残留。** 三条证据：

1. **恒等表（`LUT_phase_1_12`）的这 40 项全为 0**，而它是一张能正常工作的表
   （DQ phase1 走它）。若该区承载必需参数，全零的表不可能工作。
2. **exp 表的这一区是 `00bd ffff ffff ffff …` 的长串 `0xff`/`0xbd`/`0xde`/`0x7b`**
   —— 典型的填充/未初始化字节模式（69% 的字节 ≥ 0xbd），周期 4 字节的自相关达 46%。
3. **三张表的这一区互不相同且无规律**，数值范围横跨 `-31696 … 64448`，
   既不单调、也不落在任何合理的段边界域内。

> 这更正了本文一次修订的判断（当时把该区当作「段边界/指数参数编码」并列为待确认项）。
> **实际含义**：生成时这 40 项**写 0 即可**，不需要拷贝参考产物。

#### 倒数表：已完整破解（切线族，非弦线）

段索引取自 **fp16 尾数高 5 位**，指数部分单独处理：

```python
def reciprocal_lut(v, A, B):                # v > 0, fp16
    bits = fp16_bits(v)
    e    = (bits >> 10) & 0x1F              # 指数字段
    m    = bits & 0x3FF                     # 尾数字段（10 bit）
    mant = 1.0 + m / 1024.0                 # 归一化尾数 ∈ [1,2)
    i    = m >> 5                           # 尾数高 5 bit → 32 段
    r    = A[i] * mant + B[i]               # ≈ 1/mant ∈ (0.5,1]
    return r * 2.0 ** (15 - e)              # 还原指数：1/(mant·2^(e-15))
```

**关键发现：每段是 `1/x` 的切线，不是弦线。** 判据 —— 过点 `p` 的 `1/x` 切线为
`y = -x/p² + 2/p`，故 `A = -1/p²`、`B = 2/p`，两者满足 **`A = -B²/4`**：

| 检查 | 结果 |
| --- | --- |
| `A[i] == -B[i]²/4`，31 段全检 | 平均偏差 **0.000167**，最大 0.000469 —— **在 fp16 分辨率内即精确** |
| 反解切点 `p[i] = 2/B[i]` | `1.0079 → 1.9807`，恰好覆盖尾数域 `[1,2)` |
| 切点间距是否均匀 | **否**：间距 `0.0156 → 0.0554`，比值 3.6。与任何均匀网格的偏差都在 0.06 以上 |
| 用切线式重算 5 个 DQ 节点（n=1/32/86）的倒数，对比实测硬件输出 | 平均偏差 **0.032%–0.049%** |

**另一个反直觉的结论：按均匀网格自行合成的切线表，比参考产物的表更接近硬件实际输出。**

| 表 | 与硬件输出的平均相对偏差（5 个节点） |
| --- | ---: |
| 我方合成（切线 @ 均匀中点 `p_i = 1+(i+0.5)/32`） | **0.032% – 0.049%** |
| 参考产物的表 | 0.368% – 0.666% |

硬件实测结果与真值 `1/x` 只差 **0.019%**，说明硬件内部精度高于 32 段 PWL 的理论极限
（均匀切线约 0.04%、弦线约 0.39%）。参考产物的表用了**非均匀切点**，反而偏离更多
—— 推测其切点由对方工具链按某种等误差准则选取（实测 `width/p^1.5` 的离散度最小、
为 0.37，但仍不足以确认具体准则）。

**结论：倒数表按均匀切点自行合成即可，无需拷贝，且精度更好。**

#### 恒等表（DQ phase1）：可字节级复现

只有 `A[0] = 1.0`，其余 143 项全 0，配 `activation_mode=1`。DQ 的 phase1 走 LUT 通路但不做
非线性变换（真正的 `/256` 由 `Scaling_buffer_phase_1` 完成）。
**实测：按此合成的 288 字节与参考产物 `LUT_phase_1_12.bin` 完全字节相同。**

#### 恒等表（DQ phase1）

只有 `index 0` 非零且为 `1.0`，配 `activation_mode=1`。DQ 的 phase1 走 LUT 通路但不做
非线性变换（真正的 `/256` 由 `Scaling_buffer_phase_1` 完成）。生成时照此写即可。

#### SiLU 表

`A` 从 0 升到 1.0967 再回落到 1.0068（正是 SiLU 导数的形状：先 0、过冲到约 1.1、
再收敛到 1）；`B` 下探到 -0.4319 再回 0。按 `y = A[i]·x + B[i]`、32 段均匀覆盖
**`[-4, 4)`** 拟合真值 `x·sigmoid(x)`，平均绝对误差 **0.0196**（次优定域 `[-5,5)` 为 0.0230）。

#### exp 表（Softmax phase1）：结构已知，段索引未定

`B` 从 0 单调升到 1.0（29 段），`A` 从 0 升到 1.9688 —— 形状与 `exp` 的切线族一致。
段索引的推导未能确定：LUT 的输入是 FPSU 变换后的值 `u = (x - max) × 0.5`
（`Scaling_buffer_phase_1 = 0.5`），实测 `u ∈ [-2.994, 0]`，需要 LUT 实现 `exp(2u)`。
按均匀定域拟合最好只到平均误差 0.045（定域 `[-1.5,0)`），说明**段索引不是对 `u` 均匀切分**，
而与 `flp_min_exp=9 / flp_max_exp=16 / flp_mantisa=3` 有关（8 个 binade × 2³ = 64 个候选槽，
但仅 31 段可用）。与倒数表一样，它的段点也是非均匀的，具体准则未能从单一合成样本反推。

**注意 `A[0] = B[0] = 0`**：exp 与 SiLU 的第 0 段全零（这两个函数在 `-∞` 侧趋于 0，
段 0 承担「饱和到 0」），而倒数表的第 0 段是实值（`1/x` 在 `[1,2)` 内不衰减）。
合成 exp/SiLU 表时应把有效段放在 `1..30`，第 0 段留 0。

#### 落地方案

| 表 | 文件数 | 方案 | 阻塞 |
| --- | ---: | --- | --- |
| 恒等 | 37 | **合成**：`A[0]=1.0`，其余全 0 —— **与参考产物字节完全相同** | 无 |
| 倒数 | 69 | **合成**：切线族 `B[i]=2/p_i`、`A[i]=-1/p_i²`，`p_i=1+(i+0.5)/32`。**实测比参考表更接近硬件（0.04% vs 0.55%）** | 无 |
| SiLU | 1 | **合成**：段 1..30 覆盖 `[-4,4)`，段 0 留 0。定域为拟合最优值，非实测确认 | 低 |
| exp | 32 | **拷参考产物 `LUT_phase_1_18.bin`**（32 个 Softmax 共用同一张，与模型/输入无关） | 低（拷贝即可） |
| `[64:104]` | 全部 | **写 0**（实测为未初始化残留，非参数） | 无 |

```python
LUT_TABLES = {
    ('DynamicScaling', 1):     synth_identity(),      # 字节级复现
    ('DynamicScaling', 2):     synth_reciprocal(),    # 优于参考表
    ('Llama2ActivationDQ', 1): synth_identity(),
    ('Llama2ActivationDQ', 2): synth_reciprocal(),
    ('Softmax', 1):            copy('ref/LUT_phase_1_18.bin'),   # exp，暂拷
    ('Softmax', 3):            synth_reciprocal(),    # 与 DQ phase2 同一张
    ('Gemm', 'fused_silu'):    synth_silu(),
}
```

**注意**：表与 `flp_*` 三元组强绑定，写表时必须同时写对应的
`flp_min_exp / flp_max_exp / flp_mantisa`（见 §3.6 的常量表），否则查表定址错位。

> **阻塞程度的变化**：一次修订时这一族是「唯一硬阻塞、整族只能拷贝」。
> 接入硬件规范后：**4 张表里 3 张可自行合成**（其中恒等表字节级复现、倒数表精度反超
> 参考产物），只有 exp 表需要拷贝一次；`[64:104]` 段确认为残留、写 0 即可。
> 拷贝也是安全的——这些表与模型无关，全模型 32 层共用同一批。

### 4.9 DEBUG 族（可跳过）

**379 个被引用的 DEBUG 文件在参考产物中不存在** —— 它们是**可选生成**的浮点对拍副本，
后端编译器不依赖。参考产物自己就留着这些悬空引用。

| 缺失族 | 数量 |
| --- | ---: |
| `DEBUG_input_buffer_float` / `_<slot>_float` | 153 + 112 |
| `DEBUG_weight_buffer_float` | 73 |
| `DEBUG_div_value` | 32 |
| 其余 `DEBUG_*` / `lut_debug` | 9 |

**我方策略**：完全跳过这一族的落盘。GML 里写不写引用两种做法都可（参考产物写了且悬空）。
建议**写引用但不落盘**，与参考产物行为一致。

例外：`DEBUG_weight_buffer_spc/spg/*_axis/_group_size`（各 7 处）**不是文件名而是数值字段**，
必须写（见 §3.4）。

---

## 5. 逐算子字段模板（生成器可直接照此发射）

16 种 `op_type` 的分布与字段清单。**「层」列指 `prepare_out/txt_files` 里展开成几个层**
（本文不展开层参数文本，见 `prepare_out生成规范.md`）。

「对应来源」列给的是**我方 fx 图里的 aten 算子**（`gml_bridge/from_fx.py` 的 `OP_TYPES`），
不是 Relay 算子 —— 我方不走 TVM，见 §2.2。「已映射」标出 `OP_TYPES` 里是否已有该条。

| `op_type` | 数量 | 对应来源（aten / 图变换） | 已映射 | 层 | 备注 |
| --- | ---: | --- | :---: | ---: | --- |
| `MatMul` | 64 | `aten.bmm` / `aten.mm` | ✓ | 1 | 32 头 ×（QK^T、PV） |
| `DynamicScaling` | 36 | **量化 pass 插入**（无 aten 对应） | ✗ | 4 | phase 型 |
| `Softmax` | 32 | `aten._softmax` | ✓ | 5 | phase 型，逐头 |
| `Mask` | 32 | `aten.masked_fill` / `aten.where` | ✓ | 1 | 逐头 |
| `Gemm` | 7 | `aten.linear` / `aten.addmm` | ✓ | 1 | q/k/v/o_proj + gate/up/down |
| `Transpose` | 4 | `aten.transpose` / `aten.permute` | ✓ | **0** | 布局类 |
| `Split` | 3 | `aten.split` / `aten.split_with_sizes` | ✓ | **0** | Q/K/V 按头拆 |
| `RMSNorm_vpu` | 2 | **融合 pass**：`pow`+`mean`+`add`+`rsqrt`+`mul`+`mul` 六合一 | 部分（现挂 `aten.rsqrt`） | 1 | VPU 类 |
| `Reshape` | 2 | `aten.view` / `aten.reshape` | ✓ | **0** | 布局类 |
| `KV_Cache_DMA` | 2 | `aten.index_put` / `aten.slice_scatter`（KV cache 写入） | ✗ | **0** | 布局类 |
| `EltwiseAdd` | 2 | `aten.add.Tensor`（残差） | ✓ | 1 | 双输入 |
| `Concat` | 1 | `aten.cat` | ✓ | **0** | 32 头拼接 |
| `EltwiseMul` | 1 | `aten.mul.Tensor`（MLP gate×up） | ✓ | 1 | 双输入 |
| `Llama2ActivationDQ` | 1 | **融合 pass**：RoPE(q) + 动态量化 | ✗ | 7 | phase + 子块 |
| `Llama2Activation` | 1 | **融合 pass**：RoPE(k) | ✗ | 3 | 子块 |
| `Lut` | (1) | `aten.silu` 折进 Gemm 的 `contraction` | 部分（现映射为独立 `Silu`） | 0 | **不是顶层节点**，嵌在 Gemm 195 的 `contraction` 里 |

**`OP_TYPES` 需要改的四处**（这是代码层面的直接待办）：

1. **`aten.silu` 现映射为独立 `op_type "Silu"`** —— 但实物里 SiLU 是折进 Gemm 的
   `contraction`，没有独立 `Silu` 节点。应改为融合而非独立节点。
2. **缺 `KV_Cache_DMA`** —— fx 侧对应 `aten.index_put` / `aten.slice_scatter`。
3. **缺 `DynamicScaling`** —— 它由量化 pass 插入，不来自 aten。
4. **`RMSNorm_vpu` 现挂在 `aten.rsqrt` 上** —— 实物是六个算子融合成一个节点，
   应由融合 pass 识别整个 RMSNorm 模式，而不是单挂 rsqrt。

另外 `OP_TYPES` 里的 `EltwiseSub` / `EltwiseDiv` / `MaxPool` / `AveragePool`
在 llama2 decode block 中不出现（属 ResNet 类模型），保留无害。

**布局类算子（Reshape/Transpose/Split/Concat/KV_Cache_DMA，共 12 个）不产生层参数文本**，
因为不占计算单元。其中 8 个（Reshape/Transpose/Concat/Split）实测**也没有 `output_sf`/`output_zp`**
—— 量化信息只挂在计算类算子上。

### 5.1 Gemm 模板（以节点 195 `mlp_gate_proj` 为例，带融合 SiLU）

字段发射顺序（参考产物的实际顺序，建议照抄以便 diff 比对）：

```
id, use_dynamic_quantization, node_id, label, name
input_data_extensions, output_data_extension
input_sf, input_sf_dtype, input_zp                     <- 动态量化时 input_sf 指向上游 phase_1
weight_sf, weight_sf_dtype, weight_zp
output_sf, output_sf_dtype, output_zp
fpsu_spc, fpsu_spc_axis, fpsu_spg, fpsu_mode
Scaling_buffer_file, Bias_buffer_file, Scaling_PS_buffer_file
kantor_mode
input_buffer_dtype, input_buffer, DEBUG_input_buffer_float
DEBUG_weight_buffer_spc/_spc_axis/_spg/_spg_axis/_spg_group_size    <- 仅 int4
weight_buffer_dtype, weight_buffer, DEBUG_weight_buffer_float
op_type, nmu_mode
[融合时] DEBUG_silu_input_dtype/_input/_float, contraction[...], DEBUG_silu_input_sf/_dtype/_zp,
         flp_min_exp, flp_max_exp, flp_mantisa, activation_mode,
         activation_special_operators, activation_lut_file, lut_debug
idx, A, residual_input_buffer, input0_node_id, input_count
residual_output_buffer, output0_node_id
pooling_dtype, output_buffer, output_buffer_dtype
```

未融合的 6 个 Gemm 省掉 `contraction` 及其 LUT 相关字段。
`kantor_mode` 在 6 个 Gemm 上是 `off`，在 1 个上是 `fp2int_converter`（该 Gemm 后接量化）。

### 5.2 MatMul 模板（64 个，attention 的两个矩阵乘）

与 Gemm 的差异：

| 字段 | matmul1 (QK^T) | matmul2 (PV) |
| --- | --- | --- |
| `weight_format` | **`weights_transpose`** | **`weight`** |
| `Scaling_buffer_file` 的值 | **1/√128 = 0.088388** | `1.0` |
| `weight_sf` 值 | key cache sf (0.04589) | value cache sf (0.00261) |
| `split_channel_number` | **有**，= head 编号 0–31 | 无 |
| `A` | = `input0_node_id`（Split 节点） | = `input0_node_id`（Softmax 节点） |
| `output_buffer` | `input_buffer_0_<mask_id>.bin` | `input_buffer_<slot>_<concat_id>.bin` |

共同字段：`MatMul_input_as_weight 1`、`group_attention_data_num "32"`、
`group_attention_weight_num "32"`、`use_dynamic_quantization 1`、
`nmu_mode floating_point`、`fpsu_mode floating_point_32`、`pooling_dtype floating_point`、
`kantor_mode off`、`fpsu_spc 1`、`fpsu_spc_axis 1`、`fpsu_spg 0`。

**注意 `input_count` 恒为 1（不是 2）**，第二个 operand 走 `input1_node_id` + 权重通路。

### 5.3 RMSNorm_vpu 模板（2 个）

特点：**唯一使用 fp32 量化参数的算子**，且带 `vpu_params` 嵌套块。

```
input_sf_dtype  "float32"      <- 不是 float16
weight_sf_dtype "float32"
output_sf_dtype "float32"
weight_buffer_dtype "int8"     <- RMSNorm weight 是 int8 per-tensor（不是 int4 per-group）
weight_sf 值 = 1/127 = 0.007874
RMSNorm_Add_Const → 4B fp32 = 1e-05 = config.json 的 rms_norm_eps
Use_Scaling 0
vpu_params [ Vpu_Axis -1 + 5 个文件名别名 ]
```

**没有** `fpsu_*` / `nmu_mode` / `pooling_dtype` / `kantor_mode` —— VPU 是独立通路。

### 5.4 EltwiseAdd / EltwiseMul 模板（双输入，逐槽配置）

双输入 eltwise 的 FPSU 配置**逐槽各一套**，用 `_<slot>` 后缀（不是 `_phase_`）：

```
input_<slot>_sf / _sf_dtype / _zp                     slot = 0, 1
fpsu_<slot>_spc, fpsu_<slot>_spg                      恒 1, 0
fpsu_mode_<slot>                                       恒 floating_point
Scaling_buffer_file_<slot>, Bias_buffer_file_<slot>, Scaling_PS_buffer_file_<slot>
pooling_dtype_<slot>                                   恒 floating_point
input_buffer_<slot>, input_buffer_<slot>_dtype
```

`EltwiseMul`（节点 194）额外带 `kantor_mode "elementwise_mul_fp16"` 与完整的
`kantor_A_*` / `kantor_B_*` 配置（scale/bias/Shift/spc/spg/scale_axis），`transpose 1`；
`EltwiseAdd` 是 `kantor_mode "off"`，无 kantor 配置。

`EltwiseAdd` 节点 9（最终残差）**无 `output_sf`/`output_zp` 文件**，而是直接写
`output_sf "input_sf_8.bin"` 指向图输出 buffer 的 sf —— 输出侧复用消费者的文件名。

### 5.5 Mask 模板（32 个，逐头）

最简单的算子：双输入、无量化参数、无 FPSU。

```
id, node_id, label, name, transpose 1
input_data_extensions 3, output_data_extension 3
kantor_mode "off"
input_buffer_0 / _dtype / DEBUG_...  (scores)
input_buffer_1 / _dtype / DEBUG_...  (mask，来自 is_mask 的图输入节点 6)
op_type "Mask"
idx, residual_input_buffer ×2, input0/1_node_id, input_count 2
residual_output_buffer, output0_node_id
output_buffer, output_buffer_dtype, output_sf, output_zp
```

`output_sf` / `output_zp` 直接写下游 Softmax 的 `input_sf_<softmax_id>.bin` / `input_zp_...`。

### 5.6 KV_Cache_DMA 模板（2 个）

```
input_sf / _dtype / _zp                 被写入的 cache 的量化参数
updates_sf / _dtype / _zp               写入新值的量化参数
output_sf / _dtype / _zp
fpsu_mode "fixed_point"                 <- 唯一用定点 FPSU 的算子
pooling_dtype "fixed_point"
Scaling_buffer_file → 2.0 ; Scaling_PS_buffer_file → 14   <- 唯一非 0 的 PS
input_buffer_0 (cache, int8) / input_buffer_1 (indices, int16) / input_buffer_2 (updates, int8)
use_input_buffer_1 "L2A_ignore"         <- indices 不走 L2A 通路
input_count 3
residual_output_buffer ×2 → 下游算子 + 图输出 buffer(199/200)
```

**双输出**：既给下游用，又作为图输出（`key_cache_out` / `value_cache_out`）。

### 5.7 buffer 节点模板

图输入（7 个，node_id 1–7）：

```
id, node_id, label, name, is_buffer 1, from_tvm 1, original_name
[is_mask 1]                             仅 node 6
idx
residual_output_buffer + output<k>_node_id   每个消费者一对（可多对）
output_buffer, output_buffer_dtype
[output_sf, output_zp]                  仅部分节点有
```

图输出（3 个，node_id 8/199/200）：

```
id, original_name, node_id, label, name
input_data_extensions, output_data_extension
input_sf, input_sf_dtype, input_zp
input_buffer_dtype, input_buffer, DEBUG_input_buffer_float
is_buffer 1
residual_input_buffer, input0_node_id, input_count 1
```

`label` 用语义名（`output` / `key_cache_out` / `value_cache_out`）。
`original_name` 在参考产物里是对方前端的内部名（`tvmgen_default_nprm_main_182_output_<k>`）；
**我方直接用 HF 的语义名**（`hidden_states` / `key_cache` / `value_cache`），见 §2.2。

### 5.8 图级 I/O 与 `IO_info.txt`

`IO_info.txt` 是 Python dict 字面量（单行，无换行），记录 7 输入 3 输出：

```python
{'inputs': {'<node_id>': {'sf': array(<val>, dtype=float32), 'dtype': '<dt>',
                          'node_name': '<original_name>', 'input_idx': <k>,
                          'shape': [...], 'size': <numel>, ['mask': True]}, ...},
 'outputs': {'<node_id>': {'sf': ..., 'dtype': ..., 'node_name': ...,
                           'previous_name': '<语义名>', 'output_idx': <k>,
                           'shape': [...], 'size': <numel>}, ...}}
```

Llama2 decode block 的 I/O 契约（**这是从 7B 模型出发的入口**）：

| node_id | 张量 | shape | dtype | sf | 说明 |
| ---: | --- | --- | --- | ---: | --- |
| 1 | hidden_states | `[1,1,4096]` | float16 | 1.0 | = `[batch, seq=1, hidden]` |
| 2 | sin_position_embedding | `[1,1,1,128]` | float16 | 1.0 | = head_dim |
| 3 | cos_position_embedding | `[1,1,1,128]` | float16 | 1.0 | |
| 4 | key_cache | `[1,32,1024,128]` | int8 | 0.04589 | `[b, heads, max_seq, head_dim]` |
| 5 | cache_position | `[1,32,1,3]` | int16 | 1.0 | scatter 索引 |
| 6 | attention_mask | `[1,1,1,1024]` | float16 | 1.0 | `mask: True` |
| 7 | value_cache | `[1,32,1024,128]` | int8 | 0.00261 | |
| 8 | output (hidden) | `[1,1,4096]` | float16 | 1.0 | |
| 199 | key_cache_out | `[1,32,1024,128]` | int8 | 0.04589 | 与输入同 sf |
| 200 | value_cache_out | `[1,32,1024,128]` | int8 | 0.00261 | |

`max_seq_len=1024` 是**这份产物的配置**，不是 7B 的固有值（7B 支持 4096）。
从 7B 生成时须按目标 KV cache 长度调整，届时 `dims` 里的 `1024`、
attention scores 的元素数、`global_pooling_group_size` 的 `1024` 都随之变化。

整个图就是标准的 Llama2 decode layer（参考产物的 `raw_relay_mod.txt` 可读到同一结构；
我方从 HF `LlamaDecoderLayer` 的 fx 图得到）：

```
RMSNorm → q/k/v_proj → RoPE → ScatterND(KV写入) → QK^T → /√128
        → +mask → softmax → PV → o_proj → 残差
        → RMSNorm → gate/up_proj → SiLU → mul → down_proj → 残差
```

---

## 6. 从 Llama2-7B 出发的取值来源（「取值范围」的最终依据）

本章回答「这个值到底从哪来」。四类来源：

### 6.1 来自 `config.json` 的模型超参

**本次修订：以下取值已用真实模型 `flagOS-installed/model-inference/models/Llama-2-7b-hf/config.json`
逐项核对。**

| 超参 | 实际值 | 落进哪些字段/bin |
| --- | ---: | --- |
| `hidden_size` | 4096 | `dims` 的 `4096`；`original_shape`；权重形状；hidden 缓冲元素数 |
| `intermediate_size` | 11008 | `dims` 的 `11008`；MLP 权重形状；节点 193 的 86 组 |
| `num_attention_heads` | 32 | `num_heads`、`group_attention_data_num/_weight_num`、`split_channel_number` 上界、逐头展开的份数 |
| `num_key_value_heads` | **32**（= 上者，7B 是 MHA 非 GQA） | `group_attention_weight_num` |
| `head_dim`（= hidden/heads，config 不直接给） | 128 | `group_size`、`dims` 的 `128`、**`Scaling_buffer_file` 的 1/√128** |
| `rms_norm_eps` | **1e-05** | **`RMSNorm_Add_Const_*.bin` 的值**（实测参考产物为 9.99999975e-06 = fp32(1e-05) ✓） |
| `max_position_embeddings` | 4096 | KV cache 长度上界（本产物用 1024） |
| `num_hidden_layers` | 32 | 全模型的块数（决定总产物规模，见 §6.5） |
| `hidden_act` | `"silu"` | `activation_op_type "Silu"`、融合 LUT 的函数 |
| `torch_dtype` | `"float16"` | 激活缓冲的 `float16`、`data_extension 3` |
| `vocab_size` | 32000 | 本 block 不涉及（embedding/lm_head 在 block 外） |
| **`rope_theta`** | **config.json 里没有这个键** | RoPE 的 cos/sin 表 |

**`rope_theta` 需要注意**：这份 7B 的 `config.json` **不含 `rope_theta`**
（`transformers 4.31` 时代的检查点），推理时由 transformers 取默认值 **10000.0**。
但更可靠的做法是**直接用检查点里存好的 `inv_freq` 张量**：

```
model.layers.0.self_attn.rotary_emb.inv_freq   F32   shape [64]   256 bytes
```

`64 = head_dim/2`，正是 RoPE 的频率表。直接读它可以避免 `rope_theta` 默认值假设。
`rope_scaling` 为 `null`，无需缩放修正。

### 6.1.1 层 0 权重张量（已用真实模型核对，10/10 吻合）

从 `model-00001-of-00002.safetensors` 的头部读出的实际形状，与参考产物的
`weight_buffer` / `weight_sf` 字节数交叉验证：

| 模型张量 | dtype | 形状 | 量化后 | 参考产物 | ✓ |
| --- | --- | --- | ---: | ---: | --- |
| `self_attn.q_proj.weight` | F16 | `[4096, 4096]` | int4 → 16777216 B | 16777216 | ✓ |
| `self_attn.k_proj.weight` | F16 | `[4096, 4096]` | 16777216 | 16777216 | ✓ |
| `self_attn.v_proj.weight` | F16 | `[4096, 4096]` | 16777216 | 16777216 | ✓ |
| `self_attn.o_proj.weight` | F16 | `[4096, 4096]` | 16777216 | 16777216 | ✓ |
| `mlp.gate_proj.weight` | F16 | `[11008, 4096]` | 45088768 | 45088768 | ✓ |
| `mlp.up_proj.weight` | F16 | `[11008, 4096]` | 45088768 | 45088768 | ✓ |
| **`mlp.down_proj.weight`** | F16 | **`[4096, 11008]`** | 45088768 | 45088768 | ✓ |
| `input_layernorm.weight` | F16 | `[4096]` | int8 → 4096 B | 4096 | ✓ |
| `post_attention_layernorm.weight` | F16 | `[4096]` | 4096 | 4096 | ✓ |
| `weight_sf`（4096×4096，group 128） | — | — | 262144 B | 262144 | ✓ |
| `weight_sf`（11008×4096，group 128） | — | — | 704512 B | 704512 | ✓ |

**注意 `down_proj` 的形状是 `[4096, 11008]`**，与 gate/up 的 `[11008, 4096]` 相反。
字节数相同所以容易漏掉，但**分组轴不同**：per-group 量化沿最后一维切 128，
`down_proj` 的最后一维是 11008（86 组/行 × 4096 行），gate/up 是 4096（32 组/行 × 11008 行）。
两者的 `weight_sf` 元素数都是 352256，但**排布顺序不同**。搞错会让反量化整体错位。
这也解释了 GML 里 `weight_format` 需要区分布局。

### 6.2 来自量化契约（我方量化配置；参考产物记在 Relay 注解里）

| 契约项 | 值 | 落点 |
| --- | --- | --- |
| 激活位宽/范围 | int8 `[-128,127]` | `*_buffer_dtype "int8"`、`data_extension 1`、量化公式的 clamp |
| 权重位宽/范围 | int4 `[-8,7]` | `weight_buffer_dtype "int4"`、权重 clamp |
| 量化方式 | 对称（zero-point=0） | **所有 `*_zp` 恒为 0** |
| `group_size` | 128 | `global_pooling_group_size_phase_0`、`kantor_A_spg_group_size_phase_3`、`DEBUG_weight_buffer_spg_group_size`、`weight_sf` 元素数 |
| `spc` / `spg` | True / True | `fpsu_spc`、`global_pooling_spc/spg`、`kantor_A_spc/spg` |
| `group_axis` | -1（最后轴） | `kantor_A_spg_axis_phase_3 = 3`、`DEBUG_weight_buffer_spg_axis = 3` |
| 动态量化 | 激活侧 | `use_dynamic_quantization 1`、DQ 节点的存在、`input_sf` 跨节点引用 |

### 6.3 来自模型权重张量（须真实加载 7B）

| 目标 | 来源张量 | 计算 |
| --- | --- | --- |
| `weight_buffer_<id>` (int4) | `layers.0.{self_attn,mlp}.*.weight` | §4.2 的 per-group 对称量化 |
| `weight_sf_<id>` (int4) | 同上 | `absmax_per_group / 8` |
| `weight_buffer_25/197` (int8) | `layers.0.{input,post_attention}_layernorm.weight` | `round(w × 127 / absmax)`，per-tensor |
| `weight_sf_25/197` | 同上 | `1/127`（实测值），即按满量程归一 |
| `weight_buffer_<matmul>` (int8) | **不是模型权重**，是 KV cache 切片 | 运行时数据 |
| `activation_lut_file_195` | SiLU 函数 | 采样规则待确认（§4.8） |
| RoPE `_cos`/`_sin` | `position_ids` + `rope_theta` | 标准 RoPE 公式 |

> **既有文档已验证参考产物用的是合成数据**（RMSNorm 权重解出恒 1.0、RoPE cos/sin 超出
> `[-1,1]`、causal mask 有 968 个不同值且 50.2% 为正）。所以**不能拿参考产物的 bin 内容
> 做数值对拍**，只能对格式、尺寸、字段。数值正确性要靠我方自己的量化流程 + 反量化误差校验。

### 6.4 来自硬件模板（常量表，与模型无关）

即 §3.5 / §3.6 的两张表。这些值**不依赖模型也不依赖分块**，是 `(op_type, phase)` → 配置的
固定映射。建议在 pim-compiler 里落成一份声明式表格（与 `contracts/gml_names.py` 并列），
例如 `contracts/gml_hw_defaults.py`，并在 FlagTree 侧提供同源的 C++ 常量以防漂移。

### 6.5 全模型规模推算

单个 decode block（本产物）→ 32 层全模型：

| 项 | 单块 | 32 层 | 说明 |
| --- | ---: | ---: | --- |
| GML 节点 | 200 | ~6400 | 线性 |
| GML 边 | 331 | ~10592 | 线性 |
| `.bin` 文件 | 3231 | ~103392 | 线性 |
| `.bin` 体积 | 246 MB | **~7.9 GB** | 主要是 int4 权重（每块 ~180 MB） |
| GML 行数 | 16012 | ~512K | 线性 |

**权重占绝对主导**：每块 4×16.7MB + 3×45MB = 202 MB，×32 = 6.5 GB。
int4 不打包（1 字节/元素）使体积翻倍 —— 若对方后端支持 nibble 打包可省一半，
**建议确认**（§9 问题 3）。

---

## 7. 责任划分：G / K / C 的判据与论证

### 7.1 核心论证：GML 里的硬件字段应归 C（常量表），不是 K（算子编译器）

用户对「不是所有硬件相关的都需要算子编译器干，而是算子内的事情由算子编译器干」的质疑，
**实测支持**。论证如下。

既有文档把 GML 里 96 个字段族（`fpsu_*`、`kantor_*`、`nmu_*`、`flp_*`、`weight_format`、
`global_pooling_*`、`pooling_dtype`、`transpose`、`rtl_version` 等，5531 次出现）划给算子编译器，
结论是「图编译器填不出来，这是真正的瓶颈」。

**实测检验：这些字段是否按节点变化？**

| 字段族 | 是否按节点变化 | 实测 |
| --- | --- | --- |
| `nmu_mode` | **否** | 71/71 全为 `floating_point` |
| `fpsu_mode` | **否**（仅按 op_type） | Gemm/MatMul 71 个全 `floating_point_32`；KV_Cache_DMA 2 个全 `fixed_point` |
| `fpsu_spc` / `_spc_axis` / `_spg` | **否** | 73/73 全为 `1 / 1 / 0` |
| `pooling_dtype` | **否**（仅按 op_type） | 同 `fpsu_mode` 的分布 |
| `kantor_mode` | **否**（仅按 op_type） | 见 §3.5 表，每个 op_type 一个固定值 |
| `flp_*` 三元组 | **否**（仅按 op_type+phase） | DQ p1=(10,17,3)、DQ p2=(15,15,0)、SM p1=(9,16,3)、SM p3=(15,15,0)，**36/36、32/32 全一致** |
| `global_pooling_spc/spg/*_axis` | **否** | 37/37 全为 `1/1/2/3` |
| `kantor_mode_phase_#` | **否** | (DQ,p3)=fp2int 36/36；其余全 off |
| `transpose` | **否**（仅按 op_type） | DQ 36/36=1、Mask 32/32=1、Llama2Activation 1/1=0 |
| `weight_format` | **否**（仅按矩阵乘角色） | matmul1 32/32=transpose、matmul2 32/32=weight |
| `rtl_version` | **否** | 37/37 全为 `"1.4"` |
| `data_extension` | **否**（按 dtype） | int8→1、float16→3，仅 1 例外 |
| **`global_pooling_group_size_phase_0`** | **是** | **唯一按节点变化的**：128 或 1024，由被量化张量决定（§3.6） |

**结论：GML 里的「硬件字段」是硬件的算子模板配置，不是分块/排布的结果。**
它们由 `(op_type, phase)` 唯一确定，图编译器用一张常量表就能填全 —— 无一需要知道
分块方案、L2 地址或步幅。唯一按节点变化的那个字段（`group_size`）也由**量化契约**
（而非硬件排布）决定，同样归图编译器。

`weight_format` 是最值得单独说的一个。既有文档称「这是算子编译器根据矩阵单元的操作数
要求决定的，图编译器不知道哪种更优」。实测它 100% 由矩阵乘在 attention 里的角色决定
（QK^T 要转置、PV 不要），**这是数学决定的，不是优化决定的** —— Kᵀ 本来就要转置。
图编译器完全知道哪个 matmul 是 QK^T。

### 7.2 修正后的三方归属表

| 信息 | 归属 | 落点 | 依据 |
| --- | --- | --- | --- |
| 计算图拓扑、形状、边 | **G** | GML 节点/边/`dims` | 图结构 |
| `label` / `qidx` 命名 | **G** | GML `label`/`name` | 量化 pass 的遍历序 |
| 量化位宽选择 | **G** | 各 `*_dtype` | 量化契约 |
| 量化参数数值（sf/zp） | **G** | `*_sf`/`*_zp` bin | 量化流程 |
| 权重量化与 per-group scale | **G** | `weight_buffer`/`weight_sf` | 模型权重 |
| 算子融合（Gemm+SiLU、RMSNorm 六合一） | **G** | `contraction` 块 | 融合 pass |
| 注意力逐头展开 | **G** | 节点集合（32 头 × 4 节点） | 图变换 |
| phase 拆分与 phase 内数学 | **G** | phase 字段族 + phase bin | §4.5 的公式 |
| `1/√head_dim` 缩放 | **G** | `Scaling_buffer_file` | 模型超参 |
| RMSNorm epsilon | **G** | `RMSNorm_Add_Const` | `config.json` |
| **单元模式（nmu/fpsu/kantor/pooling）** | **C** ← 原 K | GML | §7.1：按 op_type 固定 |
| **定标粒度（spc/spg/axis）** | **C** ← 原 K | GML | §7.1：全图恒定 |
| **浮点指数范围 `flp_*`** | **C** ← 原 K | GML | §7.1：按 (op,phase) 固定 |
| **`weight_format`** | **C** ← 原 K | GML | §7.1：由矩阵乘角色决定 |
| **phase 定标常量（含 2⁻⁶³、-30.75、1/256、256）** | **C** | phase bin | §4.4 |
| **`rtl_version`** | **?** | GML | 须与对方约定目标 RTL |
| **LUT 表内容** | **?** → 过渡用 C（拷参考字节） | LUT bin | §4.8 |
| L2 缓冲区偏移/大小/编号 | **K** | **仅层参数文本** | 分块与片上分配的结果 |
| DDR 偏移、切片起始行列 | **K** | 仅层参数文本 | 主存排布 |
| 步幅对齐（Stride X/Z） | **K** | 仅层参数文本 | 分块的结果 |
| 搬运块高宽、切片映射偏移 | **K** | 仅层参数文本 | 分块的结果 |
| 每引擎权重缓冲数、double buffer | **K** | 仅层参数文本 | 片上复用策略 |
| 权重/定标存储来源标识 | **K** | 仅层参数文本 | 存储分配 |
| 循环分块本身 | K | **不外露**（体现为步幅） | — |
| 片上复用与搬运调度 | K | **不外露** | — |
| 切分方案、设备映射 | 仿真器 | 不进交付物 | — |
| 张量并行、内存三区、KV cache 管理、通信计划 | 主机运行时 | 不进交付物 | — |

### 7.3 这个修正的直接后果

**GML + bin 这份交付物（`parser_output/`）不再阻塞于算子编译器。**

| 交付物 | 阻塞状态（既有文档） | 阻塞状态（本文修正后） |
| --- | --- | --- |
| `parser_output/` GML | 阻塞：96 族等算子编译器 | **不阻塞**：C 表可自填；仅 exp 表段索引与 `rtl_version` 待确认（均可拷/照填绕过） |
| `parser_output/` bin | 阻塞 | **不阻塞**：全部公式已实测确认；LUT 拷参考字节过渡 |
| `prepare_out/` 层参数文本 | 阻塞：127 族等算子编译器 | **仍然阻塞，但面收窄到约 34 族**：DDR/L2 的 offset/size（22）与 stride（12） |

算子编译器需要做的事**收窄到层参数文本里的 DDR/L2 地址与步幅**。
**任务调度不在其中**：实测 `Task ID` / `Prev-Next task` / `Sys virtual` 共 10 族在
`(layer type, phase)` 内恒为单值，且 `Task ID` = phase 序号 − 1（269 个文件全中），
图编译器可算。详见 `prepare_out生成规范.md` §3.1。
这正是「算子内的事情由算子编译器干」—— 地址与步幅是算子降级到硬件指令时的内部决策结果。

FlagTree 侧的对接建议不变：复用已有的 MLIR 模块属性通道（`pim.l2-bytes`、`pim.l1-bytes`、
`pim.tile-m/n/k` 等已在最新提交中），新增承载 L2/DDR 偏移与步幅的属性。
**但这条通道现在只服务于层参数文本，不再是 GML 的前置条件。**

---

## 8. 与我方当前产物（`/tmp/gml_full`）的差距

### 8.1 规模对照

| 对象 | 参考产物 | 我方 | 差距性质 |
| --- | ---: | ---: | --- |
| GML 行数 | 16012 | 26778 | 更长但更空 |
| GML 节点 | **200** | **1159** | **粒度错位** |
| GML 边 | 331 | 1445 | |
| 顶层键名 | **617** | **29** | 覆盖率 4.7% |
| `.bin` 文件族 | **76** | **7** | |
| `.bin` 文件数 | 3231 | 3339 | 数量近似但族数差 11 倍 |

### 8.2 最根本的差距：粒度，不是字段

我方是**未融合的 aten 算子级**表达，参考产物是**融合后的硬件算子级**表达。
两边的前端不同（我方 `torch.fx`，对方 TVM Relay），但差距不在前端 —— 在这三步图变换：

```
aten 算子级 (1159, 未融合)   ← 我方当前（torch.fx 直出）
    │ ① 融合：RMSNorm 六合一、Gemm+SiLU、Mask+Add
    ▼
硬件算子级 (~100，批量注意力)
    │ ② 逐头展开：批量 attention → 32 头 × 4 节点
    ▼
硬件算子级 (200)  ← 参考产物的 GML
    │ ③ phase 拆分（GML 内为字段族；层参数文本里才成为独立层）
    ▼
层级 (422)        ← 参考产物的 net.ini + txt_files
```

我方节点样例 vs 参考产物：

| 我方（aten/fx 名） | 参考产物 |
| --- | --- |
| `mul_2`/`rsqrt`/`add`（3 个 aten 算子） | 融进 1 个 `RMSNorm_vpu` |
| `silu_13` + `linear_95`（2 个） | 融进 1 个 `Gemm` 的 `contraction` |
| label `linear_95`（fx 节点名） | label `mlp_gate_proj_MatMul_qidx397_params_195` |

**最后一行是 label 命名的待办**：fx 节点名（`linear_95`）不带模块路径信息，
必须改从 HF 模块路径构造（§3.1 的对照表）。fx 的 `node.meta["nn_module_stack"]`
里有模块路径，这是现成的来源。

### 8.3 我方已有的 7 族

`input_buffer`(1445)、`input_sf`(902)、`input_0/1/2_sf`(542)、`weight_sf`(225)、`weight_buffer`(225)
—— 即**只有输入侧数据与 scale、以及权重**，恰好是**只依赖量化流程**的部分。

### 8.4 缺失字段族按可实现性分档（本文修正了归档）

#### 甲档：立即可补，纯规则推导（约 30 族，1500+ 文件）

| 缺失项 | 参考数量 | 补法 |
| --- | ---: | --- |
| `*_zp`（input/weight/output） | 501 | **4 字节 int32 的 0** |
| `output_sf` | 183 | 静态算子=下游 `input_sf`；DQ=本节点 phase_1 |
| `Scaling_buffer_file` | 79 | **§4.3 的五值规则（含 1/√128）** |
| `Scaling_PS_buffer_file` | 79 | 1 字节 = 0（KV_Cache_DMA 为 14） |
| `Bias_buffer_file` | 79 | **4 字节 fp32 = 0.0** |
| `output_buffer` / `_dtype` | 197 各 | 命名规则已知（§3.3） |
| `input_buffer_dtype` / `weight_buffer_dtype` | 153 / 73 | 由量化位宽直接得 |
| `input_data_extensions` / `output_data_extension` | 193 各 | int8→1、float16→3 |
| `idx`/`residual_*`/`A`/`input_count` | 各 ~200 | 纯拓扑推导（§3.2，注意 MatMul 的 `input_count` 坑） |
| edge 的 `label`/`dims` | 331 各 | 由形状推导 |
| `RMSNorm_Add_Const` | 2 | fp32 = `rms_norm_eps` = 1e-05 |
| **全部硬件配置字段（原丙档）** | **~5531 次** | **§3.5/§3.6 的常量表** ← 本文的修正 |

#### 乙档：需实现算子/图变换（规则已全部明确）

| 缺失项 | 参考数量 | 依赖 | 状态 |
| --- | ---: | --- | --- |
| 算子融合（RMSNorm 六合一、Gemm+SiLU） | 3 处 | 融合 pass | 规则明确 |
| 注意力逐头展开 | 32 头 × 4 = 128 节点 | `TTPIM_SplitHeadsOp` 已有 | 规则明确 |
| `DynamicScaling` 节点 + 4 phase | 36 节点 / ~1789 文件 | **§4.5 公式已实测** | 可实现 |
| `Softmax` 5 phase | 32 节点 | **§4.5 公式已实测** | 可实现 |
| `Mask` 逐头 | 32 节点 | 简单 eltwise | 可实现 |
| RoPE（`Llama2Activation`/`DQ`） | 2 节点 / ~67 文件 | `TTPIM_RopeOp` 已有 + §3.8 子块命名 | 可实现 |
| `KV_Cache_DMA` | 2 节点 | `TTPIM_KvCacheOp` 已有 | 可实现 |
| `label` 命名对齐 | 190 | `qidx` 规则待定（不影响数值） | 可近似 |

#### 丙档：仅层参数文本需要算子编译器（不阻塞 GML）

L2/DDR 地址、步幅、缓冲区编号、任务调度 —— 见 §7.2 的 K 行。

#### 丁档：已无硬阻塞（本次修订）

一次修订时这一档是「LUT `[64:103]` 的编码规则」，判为唯一硬阻塞。接入硬件规范后：

| 表 | 状态 |
| --- | --- |
| 恒等（37 文件） | **可字节级复现** |
| 倒数（69 文件） | **可合成，且精度优于参考产物**（0.04% vs 0.55%） |
| SiLU（1 文件） | 可合成（定域 `[-4,4)` 为拟合最优，非实测确认）；也可直接拷 |
| exp（32 文件） | 拷参考产物一次即可（与模型无关，全模型 32 层共用） |
| `[64:104]` | **确认为未初始化残留，写 0 即可** |

详见 §4.8 与 §13 的合成代码（已验证）。

### 8.5 实现顺序（依赖序）

```
① 甲档字段 + bin（含 C 常量表）    图编译器，无阻塞  ← 建议立刻做
        │
        ├──► ② 算子融合                 图编译器，无阻塞
        │            │
        │            ▼
        │       ③ 注意力逐头展开        图编译器，原语已有
        │            │
        │            ▼
        │       ④ DQ/Softmax phase 节点  图编译器，公式已确认
        │            │
        ▼            ▼
   ⑤ label 命名对齐                   图编译器
        │
        ▼
   【GML + bin 交付物在此闭合】        ← 不再等算子编译器
        │
        ▼
   ⑥ 算子编译器产出 L2/DDR 地址与步幅   算子编译器 ← 当前空缺
        │
        ▼
   ⑦ 层参数文本序列化 → ⑧ net.ini → ⑨ 三方一致性校验
```

### 8.6 pim-compiler 现有代码与实物不符之处（代码级待办）

按本文结论核对 `flagos-pim-compiler` 的量化契约与桥接代码，发现以下不符。
**这些都是已核实的，不是推测**；前两项会直接产出**字节数错误**的 bin。

| # | 位置 | 现状 | 实物 | 影响 |
| --- | --- | --- | --- | --- |
| 1 | `contracts/gml_quant.py:63` `ACTIVATION_LAYOUT = QuantLayout("per_tensor")` | 激活 per-tensor，`scale_count()` 恒返回 1 | **per-group、group_size=128**：`output_sf_12` 有 **32** 个 scale、`output_sf_193` 有 **86** 个 | **`*_sf` 文件字节数错**（2B vs 64B/172B），后端读到的 scale 数组长度不对 |
| 2 | `contracts/gml_quant.py:64` `WEIGHT_LAYOUT = QuantLayout(..., axis=0)` | 对 `(4096,4096)` 返回 **32** 个 scale | **131072**（= numel/128）。`quant/weights.py` 自己按 numel/128 算，**是对的** | 两处口径矛盾；若按 `WEIGHT_LAYOUT` 分配缓冲会严重偏小 |
| 3 | `quant/weights.py:56` `peak / INT4_MAX`（=7）<br>`quant/activations.py:61` `peak / INT8_MAX`（=127） | 除数取 `2^(bits-1) − 1` | **除数是 `2^(bits-1)`**：int4 → **8**，int8 → **128** | 见下方证据。scale 偏大约 14%（int4）/ 0.8%（int8），且少用一个码点 |
| 4 | `contracts/gml_quant.py:95` `DQ_PHASE_COUNT = 5` | 统一 5 相 | **DynamicScaling 是 4 相**（36/36 节点），**Softmax 才是 5 相**（32/32） | 按 5 相给 DQ 分配会多出一整套 phase 文件与字段 |
| 5 | `contracts/gml_quant.py:98` `DQ_REDUCTION_WIDTH = 1024` | 归约宽度恒 1024 | **`global_pooling_group_size_phase_0` 为 128（5 个节点）或 1024（32 个）** | 对 hidden/MLP 的 DQ 节点分组数算错（应 32/86 组，非 1 组） |
| 6 | `contracts/gml_quant.py` `DTYPES["bias"] = ("int32",)` | 偏置只有 int32 | 浮点通路下 `Bias_buffer_*` 是 **fp32**（DQ phase0 = 2⁻⁶³ 只有按 fp32 解才成立） | 宽度都是 4 字节，**不影响落盘**；但类型注释误导，且写值时若按 int32 解释会写错 |
| 7 | `gml_bridge/from_fx.py` `OP_TYPES` | `aten.silu → "Silu"` 独立节点；缺 `KV_Cache_DMA` / `DynamicScaling`；`RMSNorm_vpu` 挂在 `aten.rsqrt` 单算子上 | SiLU 折进 Gemm 的 `contraction`；RMSNorm 是六算子融合 | 见 §5 的 `op_type` 表 |
| 8 | `gml_bridge/from_fx.py` label | 用 fx 节点名（`linear_95`） | `mlp_gate_proj_MatMul_qidx397_params_195` | 见 §3.1；用 `node.meta["nn_module_stack"]` 取模块路径 |

**第 3 项的证据**（除数 8 而非 7）：除数为 7 时 `|q| = 8` **在数学上不可能出现**
（`round(absmax / (absmax/7)) = 7`）。实测 5 个权重张量共 15000 个组：

| 张量 | 组数 | 含 `q = -8` 的组 |
| --- | ---: | ---: |
| gate_proj / up_proj / down_proj / o_proj / q_proj | 15000 | **7801（52.0%）** |

`max|q|` 的分布是 `{7: 5538, 8: 7801, ...}` —— 出现 8 就排除了除数 7。
激活侧同理：`output_sf == absmax/128` 在 32/32 组成立（而非 `/127`），
且 int8 值域用满 `[-128, 127]`。两条路径都是除以 `2^(bits-1)`，符合 §4.5 推导的
「absmax 映射到满量程 128」。

> **`lut_identity()` 是对的**：实测与参考产物 `LUT_phase_1_12.bin` 字节完全相同。
> `contracts/gml_names.py` 的命名规则也与本文 §3.3 一致。

**已过时的代码注释**（内容不错，但结论已被本文推进，建议同步）：

| 位置 | 过时说法 | 现结论 |
| --- | --- | --- |
| `contracts/gml_quant.py:108` | 「LUT 采样规则未确认，所以不提供生成函数」 | 32 段 PWL 已确认；倒数/恒等/SiLU **可合成**（§4.8、§13） |
| `contracts/gml_quant.py:71-79` | 「Scaling 都是标量」列了 5 种取值 | 正确，但漏了 `Scaling_buffer_phase_*` 的向量情形（节点 22 为 32 元素） |
| `quant/activations.py` 开头 | 「激活公式无法与实物字节级对照」 | DQ 四相公式已完整反推并验证（§4.5） |

---

## 9. 校验清单（生成器必须内置）

按成本从低到高排，前三层不需要真实模型。

### 9.1 第一层：结构自洽（最便宜，必须全过）

| # | 检查 | 判据 | 参考产物实测 |
| --- | --- | --- | --- |
| 1 | `id ≡ node_id` | 全部相等 | 200/200 ✓ |
| 2 | `name ≡ label` | 全部相等 | 200/200 ✓ |
| 3 | 三份连接信息一致 | `edge` 集合 == `outputN` 集合 == `inputN` 集合 | 331 == 331 == 331 ✓ |
| 4 | `residual_input_buffer` ≡ `inputN_node_id` | 逐项相等 | 193/193 ✓ |
| 5 | `residual_output_buffer` ≡ `outputN_node_id` | 逐项相等 | 197/197 ✓ |
| 6 | `A ≡ input0_node_id` | 全部相等 | 71/71 ✓ |
| 7 | `idx` 语义 | `consumer.input<idx>_node_id == self.node_id` | 197/197 ✓ |
| 8 | `dims ≡ edge.label` | 逐边相等 | 331/331 ✓ |
| 9 | Σ`input_count` + MatMul 数 == Σedge | `267 + 64 == 331` | ✓ |
| 10 | 无非 DEBUG 悬空引用 | 所有非 `DEBUG_*` 引用都落盘 | 3085/3085 ✓（唯一例外 `lut_debug`） |

### 9.2 第二层：命名契约

| # | 检查 | 判据 | 实测 |
| --- | --- | --- | --- |
| 11 | 数据缓冲按消费者编号 | 生产者 `output_buffer` ∈ 某消费者的 `input_buffer[_slot]` | 194/197 ✓（3 例外：phase 型自命名 + 2 个 Split 走权重通路） |
| 12 | phase 型自命名 | 有 `rtl_version` → `output_buffer_<self>.bin` | 37/37 ✓ |
| 13 | 权重按本节点编号 | `weight_buffer_<self_id>.bin` | 73/73 ✓ |
| 14 | label 格式 | 算子节点匹配 `<op>_qidx<N>_params_<node_id>` | 190/190 ✓ |
| 15 | 动态量化 scale 引用 | 上游为 DQ 时 `input_sf` 指向上游 `output_buffer_phase_1_*` | 37 处 ✓ |

### 9.3 第三层：bin 字节格式（最容易出错，必须逐族查）

| # | 检查 | 判据 |
| --- | --- | --- |
| 16 | 元素宽度 | 按 §4.0 的表逐族核对 `filesize % elem_size == 0` |
| 17 | 元素数 | `filesize / elem_size == prod(shape)` 或 `n_groups` |
| 18 | `*_zp` 全零 | 501 个文件均为 4 字节 int32 的 0 |
| 19 | **`Bias_buffer_*` 为 fp32（4 字节）** | DQ p0 = **2⁻⁶³**（常量）；Softmax p1 = phase0 归约落点（**不硬编码 -30.75**）；其余 0.0 |
| 20 | **`Scaling_buffer_file` 五值** | matmul1 = 1/√head_dim；有 multiplier 的 = 1/mult；KV_DMA = 2.0；其余 1.0 |
| 21 | **`Scaling_PS_buffer_*` 全零，除定点节点** | 浮点 FPSU 不需后置右移；仅 KV_Cache_DMA 两节点为 14 |
| 22 | int4 权重值域 | `[-8, 7]`，1 字节/元素，**不打包** |
| 23 | int4 per-group 自查 | 每组 `max|q| ∈ {7,8}`（应 >95%；沿错误的轴分组则不满足） |
| 24 | **int8 激活 per-group 自查** | 每组 `max|q| ∈ {127,128}`（`absmax` 映射到满量程）。实测参考产物 32/32 组满足 |
| 25 | `weight_sf` 元素数 | `= numel / 128` |
| 26 | **`down_proj` 分组轴** | 形状 `[4096,11008]`，沿最后一维（11008）切 86 组/行；与 gate/up 相反（§6.1.1） |
| 27 | LUT 大小与结构 | 288 字节 = 144 fp16；`[0:32]`=slope、`[32:64]`=intercept、`[64:144]`=0 |
| 28 | **倒数 LUT 切线签名** | `A[i] == -B[i]²/4`（31 段全满足，fp16 精度内） |
| 29 | **恒等 LUT 字节级** | `A[0]=1.0`，其余 143 项为 0 —— 应与参考产物字节相同 |
| 30 | `kantor_A_Shift` | int8 恒 -8（RoPE 的为 0） |

第 23、24 项是**不需要真实模型的分组轴自查**，很有价值：对称量化下每组 max|q| 必须触边界，
沿错误的轴分组则不满足。实测参考产物 int4 为 125/128 组、int8 为 32/32 组满足。

第 28 项可在**不依赖参考产物**的前提下验证倒数表正确性：切线关系 `A = -B²/4` 是 `1/x`
切线族的代数签名，写错任何一项都会破坏它。

### 9.4 第四层：量化数学（用自己的数据验证）

```python
# DQ 四相（§4.5），每个 DQ 节点都应通过
assert p0 == 2 * absmax_per_group              # 逐组
assert p1 == p0 / 256                          # 逐组
assert p2 == 1 / p0                            # 逐组，相对误差 < 3e-3（fp16）
assert output_sf  == p1                        # 逐字节
assert kantor_A_scale == p2                    # 逐字节
assert kantor_A_Shift == -8
assert q == clamp(round(x * p2 * 256), -128, 127)   # 逐元素，允许 ±1

# Softmax 五相（§4.5）
assert phase0 编码 -max(x)
assert phase1 ≈ exp(x - max)                   # LUT 近似，相对误差可到 5%
assert fp32(phase2) == sum(phase1)             # 相对误差 < 1e-3
assert phase3 == 1 / fp32(phase2)              # 1 fp16 ULP 内
assert phase4 == phase1 * phase3               # 逐元素
assert abs(sum(phase4) - 1.0) < 1e-3           # softmax 归一性

# 权重量化（§4.2）
assert weight_sf == absmax_per_group / 8
assert q_weight == clamp(round(W / weight_sf), -8, 7)
```

### 9.5 第五层：与 Llama2-7B 的对应（需真实模型）

| # | 检查 | 判据 |
| --- | --- | --- |
| 26 | 9 个权重张量的形状与字节数 | §4.2 的对照表，9/9 |
| 27 | `RMSNorm_Add_Const` == `rms_norm_eps` | 1e-05 |
| 28 | `Scaling_buffer_file`(matmul1) == 1/√head_dim | 0.088388 |
| 29 | `num_heads` == `num_attention_heads` | 32 |
| 30 | 反量化误差 | 用**峰值相对误差**判据，不是逐元素；**必须同时做阳性对照** |

第 30 项的方法学要点（既有文档已建立，此处强调）：判据须用峰值相对误差，
且**必须做阳性对照**（自行按 W4A8 量化真实权重再用同一指标测，应得高相关性），
否则无法区分「指标失效」与「产物错误」。

---


## 11. 附：实现要点速查（写代码时的 checklist）

**必须做对否则后端读不了**：

1. 同名键在 node 内重复输出（`residual_*_buffer` 每槽一次），不能用 dict 存
2. 数组拆成多行同名键（`pads 1` × 4，不是 `pads [1,2,3,4]`）
3. `id`/`node_id` 与 `label`/`name` 各写两份
4. 融合是强制的（RMSNorm 六合一、Gemm+SiLU 进 `contraction`）
5. 数据缓冲按**消费者**编号；权重/输出 requant 参数按**本节点**编号；phase 型输出**自命名**
6. 三份连接信息（edge / `outputN` / `inputN`）同步且一致
7. `input_count` 对 MatMul 少记 1（第二 operand 走权重通路）
8. 端口号 ≥10 时复现 `residual_*_buffer_` 的键名 bug
9. `vpu_params` / `contraction` 两个嵌套块的缩进与层级

**必须算对否则数值错**：

10. **`Scaling_buffer_file` 在 matmul1 上是 1/√head_dim，不是 1.0**（attention scale 在这里）
11. **`Bias_buffer_*` 是 fp32（4 字节）**；DQ phase0 = **2⁻⁶³**（真常量，不能写 0）
12. **Softmax 的 `Bias_buffer_phase_1` 与 `Scaling_buffer_phase_4` 不是常量**，
    是 phase0 / phase3 的运行时归约落点 —— **不要硬编码 -30.75**（§4.4 第 2 点）
13. **`Scaling_buffer_phase` 在 DQ p1 = 1/256、p3 = 256**；Softmax p1 = 0.5
14. DQ 量化的等效满量程分母是 **128**（`×2` 与 `/256` 合起来），`output_sf = absmax/128`
15. int4 权重 **1 字节/元素，不打包**，值域 `[-8,7]`
16. **`down_proj` 的形状是 `[4096, 11008]`**，与 gate/up 相反 —— 分组轴不同，
    `weight_sf` 元素数相同但**排布顺序不同**（§6.1.1）
17. RMSNorm 系列的 sf 是 **fp32**，其余是 fp16
18. `weight_sf_multiplier` 与 `Scaling_buffer_file` 互为倒数（次正规数补偿）
19. `*_zp` 全零但文件必须存在
20. 动态量化时 `input_sf` 指向上游的 `output_buffer_phase_1_*`，不是自己的文件
21. RoPE 优先用检查点里的 `rotary_emb.inv_freq`（`[64]` fp32），
    而不是假设 `rope_theta=10000` —— 这份 config.json 没有该键

**常量表（照 §3.5 / §3.6 / §4.4 三张表填，不需要算子编译器）**：

22. 单元模式、定标粒度、`flp_*`、`weight_format`、`transpose`、`data_extension`
23. LUT：倒数 / 恒等 / SiLU **三张可自行合成**（§4.8），exp 表拷参考字节；
    写表时必须同时写对应的 `flp_min_exp / max_exp / mantisa`，否则定址错位

---

## 12. 复现本文结论的方法

本文所有实测数字均可用以下方式复现（工作目录 `parser_output/`，仅需 Python 标准库）：

```python
import struct, os, re, collections

def rd(f, fmt):                      # 按元素格式解码整个 bin
    b = open(f,'rb').read(); sz = struct.calcsize(fmt)
    return struct.unpack('<'+fmt*(len(b)//sz), b[:sz*(len(b)//sz)])
# 'e'=fp16  'f'=fp32  'b'=int8  'B'=uint8  'i'=int32

# 1. 字段清单与取值域（→ §0.1 的 617 / §3 各表）
#    逐行正则抓 key/value，统计出现次数与 distinct 值

# 2. 节点切分（GML 的 node 块可嵌套，必须按括号深度切）
#    depth += s.count('[') - s.count(']')

# 3. 关键数值验证
rd('Bias_buffer_phase_0_12.bin','f')      # (1.0842021724855044e-19,) == 2**-63   → §4.4
rd('Bias_buffer_phase_1_18.bin','f')      # (-30.75,) —— 但它是 phase0 归约落点，非常量 → §4.4
rd('Scaling_buffer_file_20.bin','e')      # (0.08837890625,) == 1/sqrt(128)       → §4.3
rd('Scaling_buffer_phase_1_12.bin','e')   # (0.00390625,) == 1/256                → §4.4
rd('Scaling_buffer_phase_3_12.bin','e')   # (256.0,)                              → §4.4
rd('RMSNorm_Add_Const_25.bin','f')        # (1e-05,) == rms_norm_eps              → §4.6
rd('weight_sf_25.bin','f')                # (0.007874,) == 1/127                  → §4.6
rd('kantor_A_Shift_buffer_file_phase_3_12.bin','b')   # 全 -8                     → §4.5

# 4. DQ 四相公式（→ §4.5）
p0 = rd('output_buffer_phase_0_12.bin','e'); p1 = rd('output_buffer_phase_1_12.bin','e')
p2 = rd('output_buffer_phase_2_12.bin','e'); osf = rd('output_sf_12.bin','e')
ka = rd('kantor_A_scale_buffer_file_phase_3_12.bin','e')
assert all(abs(p1[i]-p0[i]/256) < 1e-8 for i in range(32))       # p1 == p0/256
assert all(abs(p2[i]-1/p0[i]) < abs(1/p0[i])*3e-3 for i in range(32))  # p2 == 1/p0
assert list(osf) == list(p1) and list(ka) == list(p2)

# 5. Softmax 五相（→ §4.5）
x  = rd('input_buffer_phase_0_18.bin','e')
q1 = rd('output_buffer_phase_1_18.bin','e')
q3 = rd('output_buffer_phase_3_18.bin','e')[0]
q4 = rd('output_buffer_phase_4_18.bin','e')
assert abs(rd('output_buffer_phase_2_18.bin','f')[0] - sum(q1)) < 0.01   # fp32 == Σexp
assert rd('output_buffer_phase_0_18.bin','e')[1] == -max(x)              # 高半 == -max
assert all(abs(q1[i]*q3-q4[i]) <= max(1e-6, abs(q4[i])*0.01) for i in range(1024))
assert abs(sum(q4)-1.0) < 1e-3

# 6. 7B 维度对照（→ §4.2）
assert os.path.getsize('weight_buffer_195.bin') == 11008*4096      # I*H
assert os.path.getsize('weight_sf_195.bin')     == 11008*4096//128*2
assert os.path.getsize('weight_buffer_23.bin')  == 4096*4096       # H*H

# 7. 硬件字段是否按节点变化（→ §7.1 的核心论证）
#    对每个字段族统计 (op_type, phase) → distinct 值集合；
#    除 global_pooling_group_size_phase_0 外，全部应为单值

# 8. 「运行时归约落点」判据（→ §0.2 第 9 项、§4.4）
#    Softmax 的 Bias_buffer_phase_1 与 phase0 输出字节相同 → 不是常量
raw = lambda f: open(f,'rb').read()
sm = [18,39,44,49,54,59,64,69,74,79,84,89]     # 32 个 Softmax 节点的前 12 个
assert all(raw(f'Bias_buffer_phase_1_{n}.bin') == raw(f'output_buffer_phase_0_{n}.bin')
           for n in sm)                                   # 32/32 成立
assert all(raw(f'Scaling_buffer_phase_4_{n}.bin') == raw(f'output_buffer_phase_3_{n}.bin')
           for n in sm)                                   # 32/32 成立
# 对照：DQ 的 Bias_buffer_phase_0 在 36 个节点上字节相同 → 是真常量
assert len({raw(f'Bias_buffer_phase_0_{n}.bin') for n in [12,17,24,193,196]}) == 1

# 9. 量化满量程是 128（→ §4.5、§2.0 第 2 点）
x  = rd('input_buffer_phase_0_12.bin','e')
p0 = rd('output_buffer_phase_0_12.bin','e')
p1 = rd('output_buffer_phase_1_12.bin','e')
q  = rd('output_buffer_phase_3_12.bin','b')
for g in range(32):
    absmax = max(abs(v) for v in x[g*128:(g+1)*128])
    assert abs(p0[g] - 2*absmax) < 1e-5                   # p0 == 2*absmax（左移 1 位）
    assert abs(p1[g] - absmax/128) < absmax/128*0.01      # output_sf == absmax/128
assert min(q) == -128 and max(q) == 127                   # int8 全域用满

# 10. LUT 是 32 段 PWL；倒数表是切线族（→ §4.8）
L = rd('LUT_phase_2_12.bin','e'); A, B = L[0:32], L[32:64]
assert all(abs(A[i] - (-B[i]**2/4)) < 5e-4 for i in range(31))   # A == -B^2/4，切线签名

# 恒等表是唯一 [64:144] 全零的表，且它能正常工作 → 该区非参数
I = rd('LUT_phase_1_12.bin','e')
assert I[0] == 1.0 and all(v == 0.0 for v in I[1:])
# 对照：其余三张表该区有残留（倒数 39 项、exp 40 项、silu 39 项非零），
# 且倒数表在 [104:144] 还有 1 项非零（index 104 = 1e-05）→ 残留不止于 [64:104]
assert sum(1 for v in L[64:104] if v != 0) == 39
```

**本文所有断言均已实际运行通过**（含 §13 的合成代码）。`Bias_buffer_phase_1` 那一项
在参考产物里解出 -30.75，但第 8 组断言证明它是 phase0 的落点而非常量 —— 这是本文
二次修订对自身一次修订的更正。

---

## 13. LUT 合成代码（已验证，可直接用）

三张表可自行合成，不需要参考产物。每张表 288 字节 = 144 个 fp16。
**以下代码已实际运行验证**，结果见末尾的验证表。

```python
import struct, math

def pack_lut(A, B):
    """按硬件规范 §4.3.3 打包 PWL 表：32 段，每段一个 slope + 一个 intercept。

    布局: [0:32]=slope A[i]  [32:64]=intercept B[i]
          [64:104]=未初始化残留（写 0）  [104:144]=填充 0
    求值: y = A[i]*x + B[i]
    """
    return struct.pack('<' + 'e'*144, *(list(A) + list(B) + [0.0]*80))


def synth_reciprocal():
    """倒数表：切线族。段索引 = fp16 尾数高 5 位，指数由硬件单独处理。

    过点 p 的 1/x 切线为 y = -x/p² + 2/p，故 A=-1/p²、B=2/p（满足 A=-B²/4）。
    切点取均匀中点 p_i = 1 + (i+0.5)/32，覆盖尾数域 [1,2)。
    """
    A = [0.0]*32
    B = [0.0]*32
    for i in range(31):
        p = 1.0 + (i + 0.5) / 32
        A[i] = -1.0 / (p*p)
        B[i] = 2.0 / p
    return pack_lut(A, B)


def synth_pwl(fn, lo, hi, nuse=30, base=1):
    """exp / SiLU 一类在 -∞ 侧衰减到 0 的函数：段 0 留 0（饱和段），
    有效段放在 base .. base+nuse-1，均匀覆盖 [lo, hi)。每段取弦线。
    """
    A = [0.0]*32
    B = [0.0]*32
    w = (hi - lo) / nuse
    for k in range(nuse):
        x0, x1 = lo + k*w, lo + (k+1)*w
        y0, y1 = fn(x0), fn(x1)
        a = (y1 - y0) / (x1 - x0)
        A[base+k] = a
        B[base+k] = y0 - a*x0
    return pack_lut(A, B)


def synth_silu():
    return synth_pwl(lambda x: x / (1.0 + math.exp(-x)), -4.0, 4.0)


def synth_identity():
    """DQ phase1：只有 A[0]=1.0，配 activation_mode=1。字节级复现参考产物。"""
    return struct.pack('<' + 'e'*144, *([1.0] + [0.0]*143))


def reciprocal_lut(v, A, B):
    """解码侧参考实现：验证合成表时用它对拍硬件输出。"""
    bits = struct.unpack('<H', struct.pack('<e', v))[0]
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    return (A[m >> 5] * (1.0 + m/1024.0) + B[m >> 5]) * 2.0 ** (15 - e)
```

**验证结果**（实际运行上述代码，对比参考产物与硬件实测输出）：

| 检查 | 结果 |
| --- | --- |
| 三张表均为 288 字节 | ✓ |
| `synth_identity()` vs `LUT_phase_1_12.bin` | **字节完全相同** ✓ |
| `synth_reciprocal()` 自检 `A[0]·1.0 + B[0]` | ≈ 1.0（切点 1.0156 的切线在 x=1 处的值） |
| `synth_reciprocal()` vs 真值 `1/x`，32 段中点 | 平均 **0.043%**，最大 0.091% |
| `synth_reciprocal()` 重算 5 个 DQ 节点，对比硬件 `output_buffer_phase_2` | 平均 **0.032% – 0.049%** |
| 参考产物的倒数表做同样对拍 | 平均 0.368% – 0.666%（**我方合成更准**） |

**`[64:104]` 写 0**：实测该区为未初始化残留而非参数 —— 恒等表这 40 项全为 0 且能正常工作，
exp 表该区是 `00bd ffff ffff …` 的填充模式。详见 §4.8。

**SiLU 定域 `[-4,4)` 是拟合最优值，非实测确认**：按系数距离比对参考表时 `[-3,3)` 略优，
但两者都与参考表有可见差异（平均系数差 0.05 量级）。SiLU 只有 1 个文件，
若对精度有疑虑，直接拷参考产物的 `activation_lut_file_195.bin` 最稳妥。

---

## 14. 相关文档

| 文档 | 内容 | 关系 |
| --- | --- | --- |
| `VBU-GML Structure-281025-031239.pdf` | 对方官方字段定义（20 页） | **字段语义**的规范来源 |
| **`Ceva-NeuPro-M_High_Level_ArchSpec_V1.6.6.GA.pdf`** | **目标硬件架构（147 页）** | **字段取值**的规范来源。§3 NMU、§4.2 CSTL、§4.3.2 FPSU、§4.3.3 Activation/LUT、§4.3.4 Pooling、§4.3.6 KANTOR、§7.2/7.3 L1/L2MSS、§10.4/10.5 量化模式 |
| `Llama-2-7b-hf/config.json` + `model.safetensors.index.json` | 模型超参与张量形状 | 所有形状与超参的真源（§6.1、§6.1.1 已逐项核对） |
| 本文 | GML 字段 + bin 的映射策略与计算方法 | `parser_output/` 的生成依据 |
| `prepare_out生成规范.md` | 层参数文本与 net.ini 的格式与归属 | `prepare_out/` 的生成依据 |
| `交付物结构与生成方案分析.md` | 两份交付物的整体结构与差距 | 上层视图；本文修正了其中若干处（§0.2） |
| `flagos-pim-compiler/docs/gml-lowering-20260914.md` | 图编译器侧实现方案 | 本文 §4.5 解决了其「公式无法确认」 |
| `flagos-pim-compiler/docs/gml-responsibility-20260915.md` | 三方职责划分 | 本文 §7 修正了其对 GML 硬件字段的归属判断 |
| `flagos-pim-compiler/docs/pim-compiler-v0.0.4.md` | 编译器安装与全流程闭环 | 模型加载与 `scripts/export_gml.py` 的调用方式 |

### 14.1 硬件规范里还可深挖的部分

本次只用到了与 GML 字段直接相关的章节。以下部分在实现层参数文本（`prepare_out/`，
即算子编译器的那部分职责）时会用到：

| 章节 | 内容 | 用途 |
| --- | --- | --- |
| §7.2 L1MSS | L1M 容量：NPM2K 为 512KB（16 块）；NPM4K 及以上为 1MB（16 块）或 0.75MB（12 块） | 层参数文本的 L2/L1 地址分配 |
| §7.3 L2MSS | L2M 为 0.75MB–32MB 可配置、多引擎共享；支持 64 位物理地址转换 | DDR/L2 偏移分配 |
| §7.3.4 / §7.4.1 DMA | 两类 DMA 的搬运配置 | 搬运块高宽、步幅 |
| §10.5 | group size 可为 16/32/64/96/128…（16 需 NPM8K 以上） | 校验 `group_size=128` 的合法性 |
| §5 Sequencer | 单元同步与配置 | 任务调度字段（Task ID 等） |
