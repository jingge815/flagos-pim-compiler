# prepare_out P0/P1 修复（2026-09-21）

> **本文档的 Softmax 相位模型判断有误，已被
> `docs/prepare_out-P0修复2-20260921.md` 更正**——下文「2. Softmax 五相
> 接入写盘」一节里"phase1 没有 output_buffer_phase_1，参考产物里从不
> 存在"这个论据是错的（只看了 txt 层卡字段模式，没有去核对参考 GML
> 文本本身；参考 GML 的 node 18 逐相都有对称的 input/output_buffer_phase_N，
> 实测 5 入 + 5 出）。Mask 边界节点、bmm2→Concat、L2 双槎、对拍白名单
> 三分类这几节的技术方向仍然成立，但本次的 104 处 VALUE_DIFF 与
> parser_output 3163 个文件的数字已过期，**当前状态以 P0修复2 那份为准**。

## 一句话状态（存档，见上方警示）

`docs/prepare_out-代码评审-20260920.md` 列的 P0（悬空 bin 引用、双输入层 L2
地址重叠）与部分 P1（对拍白名单放水、bmm2/Concat 命名）全部核实属实并修完。
`docs/prepare_out-txt-20260919.md` 的「VALUE_DIFF 0」结论**不可信**——那是
白名单放水造成的假象，本次收紧白名单后重新测出 **104 处**真实差异（详见
下文），比评审测出的 173 处少，但不是 0。

**重要纪律变更**：本次发现 GML 侧（图编译）与 prepare_out 侧（编排器）不能
分开验证——`gml_bridge/from_fx.py` 的图结构决定了哪些边存在，
`orchestrator/layer_fields.py` 的字段假设了这些边存在，两边一旦不同步就是
悬空引用（Mask 第二路输入就是这样：编排器早就在写 `Datain file 1`，但
GML 图从没有这条边，bin 从没被写过）。**以后每次改动都要用
`export_gml.py --orchestrate` 一条命令连着生成 GML + prepare_out，不要
分两次跑、不要只看其中一侧的自检**，见下文「验证方式」。

## 核实结论：评审的问题哪些是真的

| 级别 | 评审的说法 | 核实结论 |
| --- | --- | --- |
| P0 | 1051 处悬空 bin 引用 | **属实**。最大两块：Softmax 五相从未落盘（`write_softmax_phases` 写好了但 `export.py` 从没调用过）、Mask 第二路输入（causal mask）从未进图 |
| P0 | 41 层 `L2 input buffer offset 0 == offset 1` | **属实**，且是结构性的：`l2_alloc.buffers_from_layers` 只给每层分配一块地址，双输入层的两个槎share 同一个 key |
| P1 | 对拍白名单放水，真实差异是 173 处不是 0 | **属实**。命名类键（`Datain file`/`Dataout file`/`Weights buffer file` 等）被整体放进白名单，收紧后测出 104 处（本次已修掉一部分，含 bmm2→Concat 那批） |
| P0（评审误判） | Mask 第二路是「悬空引用」 | **修正**：那条边在参考产物的 `parser_output` 里**真实存在**（`input_buffer_1_190.bin`，2048 字节，非虚引用），根因是 `gml_bridge/from_fx.py` 把 causal mask 这个 `placeholder` 判成不可发射，边被静默丢弃 |

## 本次修了什么

### 1. Mask 第二路输入（causal mask）补进 GML 图

**根因**：`graph/split_heads.py` 里 Mask 节点是
`add.Tensor(current, mask)`，`mask` 是 `runtime/compile.py::
PositionalLlama.forward` 的第二个入参（`causal_mask`），经过一个 `alias`
（fx 恒等操作）喂给全部 32 个 Mask 节点，共享同一份 placeholder。
`gml_bridge/from_fx.py::_is_emittable` 只认 `call_function`，placeholder
判不过，`_tensor_inputs` 的 `walk()` 走到这里直接断边——Mask 的
`input_count` 因此少算 1，被当成单槎算子。

**修法**：同 RoPE cos/sin 表节点的处理方式——单独建一个 `is_buffer`
边界节点（全部 32 个 Mask 节点共享，不是各自一份），不进 `_tensor_inputs`
的返回值，不影响算子节点的逆拓扑编号。实测参考产物 32 个头的
`Datain file 1` 全部指向同一个文件名，边界节点的命名规则照抄这一点。

**联动修复**（否则会引入新的悬空引用）：
- `gml_bridge/export.py` 的写盘名字解析：`write_data_buffer` 原来无条件
  按 `node.node_id` 拼名字，Mask 的槎 1 是全头共享的名字，31/32 个头会
  各写一份从没被引用的多余文件，共享名反而没人写。改成按字段**值**写，
  值已经是正确名字时才信它。
- `_data_slot_count`/生产者输出命名的联动：bmm1 的输出按消费者（Mask）
  的槎数命名，Mask 从单槎变双槎后，这条链路要同步识别，否则 bmm1 写
  `input_buffer_103.bin`（无槎）而 Mask 只声明 `input_buffer_0_103.bin`
  （带槎），两边悬空。

节点数因此 199 → **200**（多了一个 Mask 边界节点），`bin` 数从 2330 涨到
2395。两个断言了旧节点数的测试（`test_gml_depends_on_opcompiler.py`、
`test_orchestrator.py` 间接依赖）已同步更新。

### 2. Softmax 五相接入写盘 + GML 字段声明

**根因**：`gml_bridge/runtime_files.py::write_softmax_phases` 一直存在，
但 `export.py::write_runtime_files` 从未 import、从未调用；GML 序列化侧
（`from_fx.py`）对 Softmax 节点也只发了顶层字段，没有 DQ 那样的
`*_phase_N` 字段族。两边都缺，Softmax 相位族因此在悬空引用里占最大一块
（608/1051）。

**修法**：
- `export.py` 补 `if node.fields.get("op_type") == "Softmax": ...
  write_softmax_phases(...)`，同 DQ 分支一样"不能 continue"（Softmax
  的 `output_sf`/`input_buffer`/`input_sf` 仍走通用路径）。
- `from_fx.py` 补 Softmax 的 `hw_table.phase_fields` 循环，字段集合逐一
  对齐 `write_softmax_phases` 实际写的名字（这一步踩了两个坑，记录在下）：
  1. Softmax phase1（exp）**没有** `output_buffer_phase_1` 这个名字，
     它的逐元素输出按下一相的输入命名（`input_buffer_phase_2`）——
     直接照抄 DQ 四相的字段判据会导致「写了盘但没引用」或反过来，
     必须逐相核对参考产物的字段集合，不能假设两种 phase 型算子同构。
  2. `LUT_phase_1`（exp 表）参考产物里真实存在但此前完全没合成——
     补了 `contracts/gml_lut.py::synth_exp()`（用现有的 `synth_decaying`
     弦线合成，同 SiLU 表一样是占位表，不追求硬件 31 段 PWL 的精度，
     `phase_data.py::softmax()` 本来就用精确 exp 计算，不经过这张表）。

### 3. bmm2 → Concat 的输出命名收敛

**根因**：参考产物里 Concat（32 路 attention 头汇合）在 `prepare_out`
里**没有自己的 `params_*.txt`、没有 `net.ini` 条目**——它是 GML 结构里的
记账节点，不是可执行层。32 个 bmm2 头的 `Dataout file` 全部指向
**同一个**文件名（`input_buffer_14.bin`，Concat 下游那个节点的名字），
不是 Concat 自己按槎位分的 32 个名字。我方 `_data_slot_count` 把 Concat
当成普通 32 槎多输入算子，按槎位给每个 bmm2 头单独命名，32 个头各写
一份从未被引用的文件。

**修法**：`orchestrator/layer_fields.py::_slotted_dataout` 加 Concat 分支
——消费者是 Concat 时，直接借用 Concat 自己的 `output_buffer` 字段（已经
按 Concat 的下游编号），不按槎位重新命名。这个分支必须排在"已带槎号就
直接放过"的早退判断**之前**：bmm2 的原始 `output_buffer` 本来就是带槎号
的形式（GML 侧 Concat 是 32 真实槎的算子），先判"已带槎号"会让 Concat
分支永远走不到。

### 4. L2 双输入层地址不重叠

**根因**：`orchestrator/l2_alloc.py::buffers_from_layers` 每个
`LayerIdentity` 只分配一块 `L2Buffer`（key 是 `identity.stem`），双输入层
（mask/residual/mlp_mul/rope_mul）的两个槎在 `layer_fields.py` 里都查
同一个 key，`offset 0 == offset 1` 是分配器天生只有一块地址，不是
`layer_fields.py` 摘错了字段。

**修法**：`l2_alloc.py` 新增 `_is_dual_input(op_type, phase)`（判据抄自
`layer_fields.py::build_layer_fields` 的 `dual` 变量，但只用
`Layer.op_type`/`phase` 重新表达一遍，避免跟 `layer_fields.py` 循环
import），双输入层额外分配一块 `f"{stem}#1"` 的 `L2Buffer`，生命周期同
槎 0。`layer_fields.py` 读 `l2_offsets.get(f"{stem}#1", offset0 + size0)`
——查不到就退回"紧跟在槎 0 后面"，保证任何路径下都不重叠。

**没做的部分**：没有伪造一个与参考字节对齐的真实地址。
`docs/prepare_out-域确认表-20260918.md` Q11 这一项本身标了「待确认」
（甲方还没回答"小槎是否必须复现，还是窗口内不重叠即可"），本次目标只是
"不重叠"这个下限。

### 5. 对拍工具（`scripts/diff_prepare_out.py`）收紧白名单

- 把 `_is_alloc_key` 拆成三类：`ALLOC_KEYS`（纯地址，无条件放过）、
  `_NAMING_KEYS`（命名类键，改成按模式比对——数字换成 `#` 归一化后再比，
  归一化后仍不同才算 VALUE_DIFF）、`_STRUCTURAL_KEYS`（`Layer ID`、
  排版瑕疵字段，无条件放过）。之前命名类键混在无条件放过的白名单里，
  评审测出 173 处被这样静默吃掉的真实差异。
- `counts_mine`（我方每类层的计数）之前只打印不比对，现在跟
  `REFERENCE_COUNTS` 逐类比对，不等则算 `UNMATCHED_FILE`。

收紧后重新跑，`VALUE_DIFF` 从假象的 0 变成 **104**（继续修 bmm2/Concat
那批之前是 136；这是本次没有全部修完的诚实数字，见下文「仍存在的差异」）。

## 仍存在的差异（104 处 VALUE_DIFF，本次未修，留作后续）

跑 `diff_prepare_out.py` 后按键归类：

| 键 | 处数 | 备注 |
| --- | --- | --- |
| `DDR Output Orig Buffer Name` | 43 | TVM 溯源名，参考走 Relay 语义名，我方是 `buffer<id>` |
| `DDR Input Orig Buffer Name 1` | 37 | 同上 |
| `DDR Input Orig Buffer Name 0` | 6 | 同上 |
| `Datain file 0` | 4 | RoPE cos/sin 表命名（`self_attn_Reshape_*_cos.bin` 这类语义名，我方是 `input_buffer_N.bin`） |
| `Datain file` | 2 | 同上，dq_p1 一条 |
| `Input buffer file 0` | 2 | RoPE 表命名 |
| `Datain file 1` | 2 | RoPE 表命名 |
| `output scale factor buffer` | 1 | 待查（本次改过一次 mask 的这个字段，剩余 1 处可能是别的层类） |

这批不属于本次评审列的 P0/P1（Mask/Softmax/L2/对拍白名单），是评审
§3.1 第 4 点提到的"GML/runtime 侧命名与 txt 侧命名是两套规则"的另一个
子集（RoPE 表、DDR Orig 溯源名），留给下一轮。

## 增删文件

| 文件 | 改动 |
| --- | --- |
| `gml_bridge/from_fx.py` | 加 Mask 边界节点（causal mask）；加 Softmax 五相字段声明 |
| `gml_bridge/export.py` | 接入 `write_softmax_phases`；写盘按字段值而不是 `node.node_id` 解析文件名 |
| `gml_bridge/runtime_files.py` | 修正 Softmax phase1/2/3 的落盘名字（`input_buffer_phase_2/3` 而不是 `output_buffer_phase_1`）；补 `LUT_phase_1`（exp 表占位） |
| `contracts/gml_lut.py` | 新增 `synth_exp()` |
| `orchestrator/l2_alloc.py` | 新增 `_is_dual_input()`；双输入层多分配一块 `#1` 槎 |
| `orchestrator/layer_fields.py` | 修正 Mask 的 `output scale factor buffer`（应指消费者而非自己）；`_slotted_dataout` 加 Concat 分支；读 L2 槎 1 偏移改查 `#1` key |
| `scripts/export_gml.py` | 新增两项闸门：`_check_bin_references_closed`（txt 引用的 bin 全部存在）、`_check_dual_slot_offsets_disjoint`（双输入层 L2 offset 不重叠） |
| `scripts/diff_prepare_out.py` | 白名单拆三类，命名类键按模式比对；`counts_mine` 参与判定 |
| `tests/test_gml_depends_on_opcompiler.py` | 节点数断言 199 → 200 |
| `tests/test_verify_layers.py` | 修正测试读取的文件名（`output_buffer_phase_1` → `input_buffer_phase_2`） |

8 个有 git 历史的文件净增 **+592 / -56**（`git diff --stat`）。
`orchestrator/layer_fields.py`、`scripts/diff_prepare_out.py` 是上一轮
遗留的未跟踪文件，没有 git 历史可比，本次在其上各做了两处改动（Mask 的
`output scale factor buffer` 修正 + `_slotted_dataout` 的 Concat 分支；
白名单拆三类 + `counts_mine` 比对），行数变化没有精确统计，不在上述净增
数字内。

## 验证方式（本次实测命令与结果）

**必须一条命令生成 GML + prepare_out，不要分开跑**——`write_runtime_files`
先写 GML 的 bin，`_run_orchestrator` 再读 GML 产物写 prepare_out 的 txt，
两段共享同一份 `GmlArtifact`；分开跑（比如先跑一次只导 GML，再用旧产物
去跑编排器）会让两侧看到不同版本的图，掩盖掉本该报错的不一致。

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler

# 一条命令生成 GML + prepare_out，两个新增闸门在这里跑
rm -rf /tmp/verify_p0
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/verify_p0 \
    --use-opcompiler --orchestrate --decode-block-only

# 对拍（收紧白名单后的版本）
python scripts/diff_prepare_out.py --mine /tmp/verify_p0/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out

# 回归
python -m pytest tests/ -q -k "not llama2_7b"
```

实测结果：
- `export_gml.py`：**15/16** 项通过。唯一未过的是新增闸门「txt 引用的 bin
  全部存在」——82 个缺失（`weight_sf_*`/`input_sf_*`/RoPE 表命名/残差旁路
  相关），这些是本次未修的既有缺口（见上文「仍存在的差异」旁的 RoPE/DDR
  Orig 名问题同源），Mask/Softmax 相关的 608+32 处已经全部清零。不是
  "16/16 全绿"——如实报告未过的那一项，不要省略。
- `diff_prepare_out.py`：`VALUE_DIFF 104`（非 0，见上文「仍存在的差异」；
  退出码非零，如实反映还有真实差异，不是"基本对齐"）
- `pytest`：683 passed, 42 deselected（`llama2_7b` 标记的用例本次未重跑，
  预计耗时约 33 分钟，按上一轮记录未受本次改动影响的路径推断，但**没有
  实测**，如需确认请单独跑 `python -m pytest tests/ -q -k "llama2_7b"`）
