# prepare_out P0 修复复核（2026-09-21）

复核对象：`docs/prepare_out-P0修复-20260921.md` 所声称的修复，以及当前工作区全部
未提交代码。方法：通读本轮改动；按修复文档的命令实跑生成 + 对拍 + 回归；
另写 3 个独立脚本做「修复文档没做」的核对（引用闭环的**逐名**清单、
命名模式的**相位/槽位**敏感比对、parser_output 与参考的**文件族**比对）。

## 0. 复核结论摘要

**修复方向正确、数字基本诚实**，评审列的 P0 确有实质改善：

| 指标 | 修复前 | 修复后 | 参考 |
| --- | --- | --- | --- |
| txt 引用缺失的 bin（distinct） | 1051（引用 3405 处） | **82** | 0 |
| 双输入层 L2 offset 重叠 | 41 层 | **0 层** | 0 |
| 对拍 VALUE_DIFF | 0（假象） | **104（如实）** | — |
| parser_output bin 数 | 2330 | 3163 | 3235 |
| 快速回归 | 683 | **683 passed** | — |

但复核发现 **1 个新的 P0 级问题**（Softmax 相位模型与参考 GML 不一致，
修复文档的关键论据是错的）、**新增闸门自身的两个漏洞**、以及一批仍存在的
命名/取值差异。按严重度列在下面。

---

## 1. 已核实为真的修复

### 1.1 Mask 第二路输入（评审误判已确认）

参考 `parser_output/input_buffer_1_190.bin` 确实存在，修复文档对评审误判的
更正是对的。核对我们自己的 GML：边界节点 202 `in_causal_mask` 已建，
32 个 Mask 的 `input_buffer_1` 全部指向 `input_buffer_1_178.bin`，且该文件
512B 已落盘。此条**通过**。

### 1.2 Softmax 五相接入写盘

`write_softmax_phases` 已在 `export.py` 调用，GML 也补了相位字段。txt 引用的
Softmax 相位 bin 已不再悬空（gate 的 82 里没有 sm 相关的）。此条**部分通过**，
但相位模型不对——见 §2.1。

### 1.3 bmm2→Concat 命名收敛

`Dataout file` 差异从 39 处降到 7 处。参考 32 个 bmm2 头确实共享
`input_buffer_14.bin` 一个名字；修复方向正确。

### 1.4 L2 双输入槽不重叠

41 层 0 重叠，新增闸门实测 `41 层双输入，0 层重叠`。分配器为双输入层多开
`<stem>#1` buffer，`layer_fields` 读 `#1` key，设计合理。

### 1.5 对拍工具收紧

白名单拆成 `ALLOC_KEYS` / `_NAMING_KEYS`（按模式比）/ `_STRUCTURAL_KEYS`，
`counts_mine` 也参与判定；104 与非零退出码都是真实数字。旧文档顶部警示已加。

---

## 2. 新发现的问题

### 2.1 P0：Softmax 相位声明与参考 GML 不一致，修复文档的论据是错的

修复文档（§2 修法-2）写：

> Softmax phase1（exp）**没有** `output_buffer_phase_1` 这个名字……
> `output_buffer_phase_1` 这个名字只留给 DQ 用，Softmax 不写。

**这与参考产物矛盾**。参考 GML
`xinfangzhou-resource/.../parser_output/relay2gml_graph.gml` 的 Softmax 节点
（node_id 18）逐字如下：

```
input_buffer_phase_0 / output_buffer_phase_0
input_buffer_phase_1 / LUT_phase_1 / output_buffer_phase_1
input_buffer_phase_2 / output_buffer_phase_2
input_buffer_phase_3 / LUT_phase_3 / output_buffer_phase_3
input_buffer_phase_4 / output_buffer_phase_4
```

参考 `parser_output/` 里也确实有 `output_buffer_phase_1_18.bin`（2048B）、
`input_buffer_phase_1_18.bin`（2048B）、`input_buffer_phase_4_18.bin`（2048B）。
我方 GML 只声明 input 0/2/3、output 0/2/3/4，且 `write_softmax_phases` 用
`input_buffer_phase_2` 承载 phase1 的输出、额外写 `input_buffer_phase_3`：

```
ref: 5 in + 5 out  |  mine: 3 in + 4 out   → 每节点缺 3 个 bin
```

32 个 Softmax 节点共缺 96 个文件，这是 parser_output 比参考少 72 个的主因。
由于 txt 里没有任何字段引用这三个名字，官方对拍器（按 txt 域比较）**永远
看不到**这个问题；新 gate 也只查「txt 引用 → 盘上有没有」，不查
「参考 GML 声明的文件族我方有没有」。也就是说：

- 这个缺口不会让现有对拍变红；
- 但参考的 GML 契约里每个相位都有独立的输入/输出缓冲，仿真器若按相位链
  读写这些缓冲，我方的缺失/错名会影响执行。

**建议**：

1. 把 Softmax 的相位字段/writer 改成参考的 5 in + 5 out 对称模型
   （`input_buffer_phase_i`、`output_buffer_phase_i` 各 5 个，LUT 只有 1/3），
   `phase1` 的逐元素输出写 `output_buffer_phase_1`（不是顶替 phase2 的输入）；
2. 修正修复文档里「参考没有 output_buffer_phase_1」的表述；
3. 新增 gate：把参考 GML 的 `.bin` 字段名集合与自己的 parser_output 对照，
   或在 CI 里比对「本产物文件族集合 == 参考文件族集合」。

### 2.2 P1：收紧后的对拍器仍隐藏「槽位/相位」差异（104 不是下界）

`_normalise_naming_value` 把**所有**数字换成 `#`，于是
`input_buffer_2_30.bin` 与 `input_buffer_1_182.bin` 归一后相同，被当成 ALLOC。
独立比对（保留小数字，只归一化节点号）测出 **4 处真实的槽位差异**：

| 文件 | 键 | 参考 | 我方 | 性质 |
| --- | --- | --- | --- | --- |
| `self_attn_o_proj_MatMul_*` | `Dataout file` | `input_buffer_1_10.bin` | `input_buffer_0_14.bin` | 槽 1 vs 0（o_proj 下游两路） |
| `self_attn_Reshape_1_*_mul_cos` | `Datain file 1` | `input_buffer_2_30.bin` | `input_buffer_1_182.bin` | 表槽 2 vs 1 |
| `self_attn_Reshape_*_mul_cos` | `Datain file 0` | `input_buffer_2_22.bin` | `input_buffer_0_184.bin` | 表槽 2 vs 0 |
| `self_attn_Reshape_*_mul_cos` | `Datain file 1` | `input_buffer_0_22.bin` | `input_buffer_1_184.bin` | 数据槽 0 vs 1 |

Q 路 `Eltwise broadcast input index: 0`（表在槽 0）两边一致，但参考的文件名是
`input_buffer_2_22.bin`（RoPE 节点声明的第 2 槽），我方是
`input_buffer_0_184.bin`——广播槽位与文件名槽位是两套规则，现在被数字归一
掩盖了。**建议**：归一化只替换节点号段（按 `gml_names` 的命名规则解析出
`(family, slot/phase, node)` 再比），把槽位、相位当闭合量。

另外，官方 104 把 37 处 `DDR Input TVM Orig Buffer Name`（前缀在
`_STRUCTURAL_KEYS`）整体排除；加上这 4 处，**「非纯编号差异」实际是
104 + 4 = 108 处**（不含 TVM 的 37 处）。修复文档写 104 可以，但应说明这是
「排除 TVM 名 + 数字全归一」后的口径。

### 2.3 P1：剩余 82 个缺失引用，文档分类不全，且根因都清楚

gate 的 `82` 与我独立核对一致（distinct 名字）。按族分类（含根因）：

| 处数 | 名字族 | 根因（可定位） |
| --- | --- | --- |
| 64 | `weight_sf_*.bin` | bmm 的 `weights scaling buffer file` 走 `_gml_str(node, "weight_sf", 回退 names.weight_scale(nid))`；bmm 的 GML 根本没有 `weight_sf` 字段，runtime 也不写。参考 parser_output 有 73 个 weight_sf（含 64 个 bmm 的）。需要：GML 声明 + 落盘（KV cache 的量化 scale） |
| 8 | `input_buffer_0_14/1_14`（add_1）、`input_buffer_0_6/1_6`（add_2）、K 表槽、Q DQ | residual 双槽缓冲仍按自己编号，没有指向真实生产者；K/Q RoPE 表槽命名（同 §2.2） |
| 8 | `Scaling[_PS]_buffer_file_6_Llama2Activation_Sin_*` | `layer_fields` 对 mul 固定 `idx=(6,6)`，但 runtime 写的是 3/4/5/6 六套（Cos 用 5/6，Sin 用 3/4）；`mul_sin` 应取 Sin 的段号，不能照抄 Cos |
| 6 | `kantor_A_*_189`、`kantor_B_*_9` 等 | gemm_v / mlp_mul 的 Kantor 文件：GML 没有这些字段（node 189/9 实测无 `kantor_*`），`layer_fields` 手拼了名字，runtime 不写 |
| 1 | `activation_lut_file_11.bin` | gate Gemm 的 `contraction` 里有 `Lut/Silu`，但 GML 没有 `activation_lut_file` 字段，`write_identity_lut`/`write_fused_silu_lut` 分支永不触发 |
| 1 | `input_buffer_184.bin` | Q RoPE DQ 的输入仍指 `input_buffer_<dq>`，参考是 `input_buffer_phase_0_<rope>.bin` |

我的独立清单与 gate 的 82 完全一致；文档「仍存在的差异」表写的是
`weight_sf_*/input_sf_*/RoPE 表/残差旁路`，其中 **`input_sf_` 不缺失**，
漏了 **Kantor 6 处、activation_lut 1 处、`Dataout file` 7 处**。建议按上表重写。

### 2.4 P1：`Dataout file` 还剩 7 处，文档表里没有

官方 diff 明示 7 处（文档「仍存在的差异」表总和 97≠104，就是漏了这 7 处）：

- `k_proj` / `q_proj`：参考 `input_buffer_0_<RoPE节点>.bin`（带槽），
  我方 `input_buffer_<折叠节点>.bin`（无槽）——生产者按**直接消费者**（Transpose/
  Reshape 折叠算子）命名，而参考按折叠后真正的层节点（RoPE）的槽位命名；
  和 `_upstream_dq` 是同一问题的下游版本。
- Q/K 的 `add`、`mul_cos`、`mul_sin`：参考用语义名
  （`..._cos.bin`/`..._sin.bin`/`input_buffer_phase_0_<rope>.bin`），
  我方用通用 `output_buffer`/`input_buffer`。

建议做一个「下游穿透」版本（与 `_upstream_dq` 对称），把生产者输出名统一到
真正层节点的槽位；语义名表缓冲复用 `gml_names` 已有的函数。

### 2.5 P2：新增两项闸门的漏洞

1. **陈旧 bin 会让「引用闭合」假通过**。`export_gml.py` 只清理
   `txt_files/*.txt`，不清理 `out_dir/*.bin`。如果复用同一个 `--out-dir`，
   上一轮遗留的 `weight_sf_99.bin` 等会让 gate 误判为已闭合。
   建议开始时清理 `out_dir/*.bin`（或先写临时目录再原子替换），
   并在 gate 里同时检查「盘上有但没人引用」。
2. **双槽重叠判据用错了尺寸**：

   ```python
   if o0 == o1 or (o0 < o1 < o0 + s0) or (o1 < o0 < o1 + s0):
   ```

   第二、三句都用 `s0`，当 `s1 != s0` 时漏判；且只查两槽互不重叠，
   不查 `#1` 分配块（`l2_alloc` 用 base `size` 开的）是否 ≥ 声明
   `L2 input buffer size 1`。当前数据恰好
   （mask 2048≤2080、mlp_mul 22016≤22048、rope 8192≈8192）没炸，但属于
   脆弱点。建议：`#1` 按 `_l2_dual_in_size` 的真实尺寸分配，
   gate 用各自 size 检查。

### 2.6 P2：Mask 的「双槽」判定仍是按层类，不是按边

`from_fx` 里已经做了「只有真的接上 causal mask 的 Mask 才登记
`mask_boundary_of`」，但编排器没有同步：

- `layer_fields.build_layer_fields`：`kind == "mask"` 一律走 dual 分支，
  写 `input_buffer_1`/`L2 offset 1`；
- `l2_alloc._is_dual_input`：`Mask` 一律返回 True。

`seq_len==1`（没有 causal mask）时，GML 是单槽，txt/分配器却按双槽写，
会产生新的悬空引用。当前导出总是带 mask，所以没暴露。建议 dual 判定改成读
GML 节点实际有没有 `input_buffer_1`，并加一个 `seq_len=1` 的单测兜底。

### 2.7 P2：bin 的尺寸/内容仍是占位，且与声明的几何不一致

- 参考 Softmax `input_buffer_phase_0_18.bin` = 2048B（1024×fp16），
  我方 node 177 同族文件 = **512B**（256×fp16）——因为导出用
  `--seq-len 16`（边 dims `1x1x16x16`），而 txt 的宽度域写死 S=1024。
  即：相位 bin 的尺寸按导出图，字段按参考图，两者差 4 倍。
- `write_softmax_phases` 的输入是 `softmax(zeros)`，phase4 = 1/numel 的占位值；
  `synth_exp` 也是占位表。这与项目「结构轮用零张量占位、bin 内容不对拍」的约定
  一致，但修复文档只说了「尺寸要准」——当前尺寸其实不准（见上条），
  建议明确写成「尺寸按导出图，内容占位；对拍参考不代表内容可比」。

### 2.8 其它（上一轮已提、本轮未动）

- `net.ini` 末行仍比参考多一个 LF（参考 EOF 无换行）。
- Task 扇出仍是 `layer_id.py:34-48` 的静态邻接表；生成方案承诺的
  「有 PhaseSource 时按 SSA 使用-定义连」仍未实现。
- `H/I/HD/S = 4096/11008/128/1024`、头数 32 等硬编码仍在
  （`layer_fields.py:22` 等）。
- `layer_fields.py` 已 1432 行；`_weight_role` 与 `OrchestrationPlan.layer_kinds`
  是死代码；`layer_hw_table.HwRow` 仍没有 docstring 里说的 `pending_q` 字段。
- `plan.py:182` `_order_like_reference` 的 `except Exception` 静默降级仍在。
- 仓库根仍留着 509KB 的未跟踪会话记录
  `2026-09-19-165121-local-command-caveatcaveat-the-messages-below.txt`。
- `scripts/export_gml.py` 现在会因 gate 未过返回 1（这是对的），
  建议在文档里把「生成命令退出码=1 属预期」写清楚，避免 CI/脚本误判。

### 2.9 工作区里的无关改动：`域确认表` 被删了 60 行

`docs/prepare_out-域确认表-20260918.md` 在工作区里是 `M`，删掉的正好是
§16（请这样回）与附录 A/B，共 60 行。这份文件是**已跟踪**的，和本轮修复
无关。研发同学说不是他所为——那更应该处理：要么
`git checkout -- docs/prepare_out-域确认表-20260918.md` 恢复，要么单独说明
删除理由再提交。**不要**把这 60 行删除夹带在 P0 修复的提交里。

---

## 3. 修复文档本身需要更正的点

| 文档说法 | 实测 |
| --- | --- |
| 「Softmax phase1 没有 output_buffer_phase_1，参考产物里从不存在」 | 参考 GML 与 parser_output **都有** `output_buffer_phase_1_18.bin`（2048B）；还有 `input_buffer_phase_1/4` |
| 「剩余差异 = weight_sf/input_sf/RoPE 表/残差旁路」 | 实际 82：weight_sf 64、input_buffer 8、Sin scaling 8、Kantor 6、activation_lut 1、Q DQ 1；`input_sf` 没有缺 |
| 剩余差异表总计 97，但 diff 报 104 | 漏了 `Dataout file` 7 处 |
| 「43 处 DDR Output Orig Buffer Name 是 TVM 溯源名」 | 37 处是 `dynamic_quantization_params_N_dequant_buff`（可从自己的 label 推），6 处是 `key_cache_out` / `..._mul_[cos|sin]_buffer`（语义名，可推），没有一处是 TVM 名 |
| 「Mask 相关 608+32 处已经全部清零」 | 引用层面清零属实；但 Mask/RoPE 的槽位与语义命名差异仍在（§2.2/§2.4） |

---

## 4. 建议的下一步（按依赖排序）

1. **Softmax 相位按参考 GML 补齐 5 in + 5 out**（P0，§2.1），并把
   「参考 GML 文件族 vs 本产物文件族」加进 gate。
2. **修 `_normalise_naming_value` 的槽位/相位敏感比对**（§2.2），
   否则后续每修一处都可能被「数字归一」掩盖。
3. **按 §2.3 的六类清单逐个闭合 82 个引用**（weight_sf、Kantor、activation_lut、
   Sin scaling、residual 双槽、Q DQ 相位名）；每关一类，gate 数字下降并留记录。
4. **补闸门自身**：清理陈旧 bin、用各自 size 检查重叠、`#1` 按真实尺寸分配、
   加 `seq_len=1` 的 Mask 单槽回归。
5. **补针对性单测**：本次三个大修复（Mask 边界、Softmax 五相、L2 双槽、
   Concat 命名）没有任何直接单测，只有端到端自检；建议在
   `tests/test_layer_fields.py` / `test_orchestrator.py` 加夹具。
6. 清理：恢复被误删的域确认表段落、删根目录会话记录、`net.ini` EOF、
   静态 Task 图/硬编码/死代码按上一轮节奏处理。

---

## 附：本次复核的实跑结果（可复现）

```bash
source .../env-pytorch.sh
rm -rf /tmp/review2
python3 scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/review2 \
    --use-opcompiler --orchestrate --decode-block-only
# → GML 200 节点 / 3163 bin；15/16 项通过；
#   未过：txt 引用 bin 缺失 82；双输入层 41 层 0 重叠
# → 退出码 1（gate 未过，符合"如实报告"的预期）

python3 scripts/diff_prepare_out.py --mine /tmp/review2/prepare_out --ref <参考>
# → MATCH 45854 / VALUE_DIFF 104 / ALLOC 5846 / UNMATCHED 0 / 退出码 1

python3 -m pytest tests/ -q -k "not llama2_7b"
# → 683 passed, 42 deselected
```

三个独立脚本的结论已在正文各节标注数字；`llama2_7b` 那组（约 33 分钟）
本次未复跑，沿用修复文档的「未实测」口径。
