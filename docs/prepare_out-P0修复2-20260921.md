# prepare_out P0 修复第二轮（2026-09-21）

## 一句话状态

`docs/prepare_out-代码评审2-20260921.md` 复核了上一轮修复
（`docs/prepare_out-P0修复-20260921.md`），指出该文档有一处新的 P0 级错误
（Softmax 相位模型判断错，论据没去对参考 GML 文本）、对拍白名单的数字归一
掩盖了真实的槽位差异、82 个剩余缺失的分类不准确、两处新增闸门本身有漏洞。
复核的四类问题**全部核实属实**，本次逐一修完，并把上一轮文档里被复核指出
错误的表述在这份文档里更正。

**本文档取代 `docs/prepare_out-P0修复-20260921.md` 的技术结论**——那份
文档记录的问题诊断（Mask 边界节点、Softmax 未接写盘、bmm2→Concat、L2
双槎重叠、对拍白名单三分类）方向仍然是对的，但其中「Softmax phase1 没有
output_buffer_phase_1」这条论据是错的，据此写的 `write_softmax_phases`
实现也是错的，本次已改正。

## 复核发现的问题：核实结论

| 级别 | 复核的说法 | 核实结论 |
| --- | --- | --- |
| P0 | Softmax 相位模型判断错——参考 GML 是 5 入 + 5 出对称结构，上一版做成 3 入 + 4 出 | **属实**。直接读参考 `relay2gml_graph.gml` 的 node 18 逐相字段，`input_buffer_phase_1`/`output_buffer_phase_1`/`input_buffer_phase_4` 全部真实存在；上一版的判断只看了 `prepare_out/txt_files` 的层卡字段模式，没有去对 GML 文本本身 |
| P1 | 对拍白名单的 `_normalise_naming_value` 把所有数字换成 `#`，掩盖了真实的槎位差异 | **属实**。实测 4 处：o_proj 的 `Dataout file` 槎 0/1 不对、`mul_cos`/`mul_sin` 的 `Datain file` 引用了本节点自己的槎而不是共享表节点的槎 |
| P1 | 82 个剩余缺失的分类不准确（漏了 Kantor 6 处、activation_lut 1 处、Dataout file 7 处） | **属实**，已按复核给的清单核对 |
| P2 | 两个新增闸门自身有漏洞：陈旧 bin 假通过、size0/size1 混用漏判重叠、`#1` 槎用错误尺寸分配 | **属实**，三处都修 |
| P2 | Mask 双槎判定仍按 `kind`，seq_len==1 会产生新悬空引用 | **属实**（结构性风险，未能用当前 seq_len 参数构造出真实的单槎 Mask 场景来复现，但代码路径的缺口是真的，已加单测直接测这条判据） |
| — | 工作区里的域确认表被误删 60 行，与本轮无关 | **属实**，已 `git checkout` 恢复 |

## 本次修了什么

### 1. Softmax 相位模型改成 5 入 + 5 出对称结构（P0）

**根因**：上一轮 `write_softmax_phases`/`from_fx.py` 的 Softmax 字段声明
基于一个错误的相位模型——只在 phase0/2/3 声明 `input_buffer_phase_N`，
只在 phase0/2/3/4 声明 `output_buffer_phase_N`，phase1 的 exp 数组塞进了
`input_buffer_phase_2`（下一相的输入名）。这个模型的唯一证据是
`prepare_out/txt_files` 里 `Datain file`/`Dataout file` 字段的取值模式，
从没去核对参考 GML 文本自己怎么声明这个节点。

直接读参考 `parser_output/relay2gml_graph.gml` 的 node 18，逐相都有
`input_buffer_phase_N` 与 `output_buffer_phase_N`（N=0..4），且逐字节验证
了实际数据流：

```
phase0  in=原始分数(1024fp16)          out=[0,-max]（高2字节编码）
phase1  in=原始分数(1024fp16，同p0)     out=exp 数组(1024fp16)      <- 真实存在，不是"没有"
phase2  in=exp 数组(读 p1 的输出)        out=Σexp（真 fp32，4B）
phase3  in=Σexp（fp32，读 p2 的输出，1 个元素，不是 2 个）  out=1/Σexp（fp16，2B）
phase4  in=exp 数组(再读一次 p1 的输出)   out=exp×(1/Σexp)（1024fp16）
```

**修法**：`gml_bridge/runtime_files.py::write_softmax_phases` 改成每相都写
`input_buffer_phase_N`/`output_buffer_phase_N`；`gml_bridge/from_fx.py` 的
Softmax 字段声明同步改成每相都声明，不再照抄 DQ 四相"有些相没有独立
输出"的判据（Softmax 是另一套结构）。`layer_fields.py` 里 `sm_p2`/`sm_p3`
读 `phase_input_buffer(nid, 2)`/`phase_input_buffer(nid, 3)` 的 Datain/
Dataout 链路不用改——那些名字本来就对（参考产物两个名字都真实存在，
只是上一轮误以为 `output_buffer_phase_1` 不存在，其实两个名字**都**存在，
装的是同一份 exp 数组内容的两个视角）。

**验证**：GML/bin 交叉校验仍 5/5 干净（3163→**3259** 个文件，多的 96 个
正是 32 个 Softmax 节点各补的 3 个：`output_buffer_phase_1`、
`input_buffer_phase_4`、修正后的 `input_buffer_phase_3`）。

### 2. RoPE cos/sin 表节点命名改成按 K 路消费者编号（P1，复核 §2.2 第 2 条）

**根因**：cos/sin 表节点被 Q 与 K 两条 RoPE 链共用，上一版按"第一个消费者"
编号（`reader_ids[0]`，与 `output0_node_id` 一致）。但实测参考产物两张
表节点（cos=node2、sin=node3）都被 Q（node22）与 K（node30）共用，
`output0_node_id` 记的是 Q（先出现），`output_buffer` 却写成
`input_buffer_*_30`——按 **K** 编号，不是按"第一个"。

**修法**：`from_fx.py` 建表节点时改用 `_is_second_rope`（已有的 K 路判据）
选锚点消费者；同时补上一个新记录 `rope_table_names`，让下游每个 RoPE
消费者节点自己的 `input_buffer_N` 字段也读这份共享名字，不再各自按
`names.data_buffer(node_id, slot)` 重新拼一份自己的名字——这一步漏掉的话，
K 锚点落地后 Q 侧还是各自拼名字，读者与生产者的名字不同步，会让
`test_output_buffer_points_at_the_consumer` 判成悬空（本次修复过程中
先踩了这个坑，加固定测试后才发现并补上）。

`orchestrator/layer_fields.py` 的 `rope_mul_cos`/`rope_mul_sin` 的
`Datain file 0/1` 构造也要跟着改：GML 槎号（cos 固定槎 2、sin 固定槎 1，
不随 Q/K 变）与 txt 层的 `Datain file 0/1`（由 `Eltwise broadcast input
index` 决定，Q 的 mul_cos 是 0、其余三条是 1）是两套完全独立的编号，
之前混用过。

**验证**：`self_attn_Reshape_1_qidx44_params_182_mul_cos` (K)、
`self_attn_Reshape_qidx36_params_184_mul_cos` (Q) 等 4 个文件的
`Datain file 0/1` 全部核对与参考产物结构一致（数字不同，槎位结构相同）。

### 3. o_proj 的 `Dataout file` 槎位差异——未修，确认是既有的残差旁路缺口

复核 §2.2 第 1 条指出 o_proj 的 `Dataout file` 槎位不对
（参考槎 1、我方槎 0）。核实后确认：o_proj 下游是残差 add 节点，这个
差异属于 `from_fx.py` 里早就记录的已知缺口（"残差旁路：第一条残差的一路
往上追到 embedding 就断了，`input_count` 记 1 而参考记 2"）——不是本轮
新引入的问题，也不在评审列出的 P0/P1（Mask/Softmax/L2/对拍白名单）范围内，
留给下一轮专门处理残差旁路时一起修。

### 4. 新增闸门自身的三处漏洞（P2，复核 §2.5）

- **陈旧 bin 假通过**：`export_gml.py` 写盘前现在先清空 `--out-dir` 下的
  `*.bin`，避免复用同一目录重跑时上一版遗留的文件让「引用闭合」检查
  误判为已闭合。同时加了反向检查（盘上有但没被 txt 引用），信息项打印
  不计入失败——不是所有未引用的 bin 都是错的（GML 侧权重/scale 本来就不
  在 txt 引用范围内）。
- **size0/size1 混用漏判重叠**：`_check_dual_slot_offsets_disjoint` 原来
  两个方向的重叠判断都用 `size0`，`size1 != size0` 时会漏判。改成两个
  方向各用自己的尺寸。
- **`#1` 槎用错误尺寸分配**：`l2_alloc.buffers_from_layers` 给双输入层的
  `#1` 槎分配地址时，原来照抄槎 0 的 `size`——大多数双输入层两槎同宽，
  这个近似没问题，但 RoPE 的 mul_cos/mul_sin 数据槎（H=4096，8192 字节）
  和表槎（HD=128，256 字节）差 32 倍，照抄会让分配器按错误尺寸判生命
  周期重叠，虚耗地址空间（复核前的数据区是 585792 字节，修完降到
  **367744 字节**）。新增 `_dual_slot1_size()`，RoPE 按表宽单独算。

### 5. Mask 双槎判定改成读 GML 边而非只按 kind（P2，复核 §2.6）

`orchestrator/layer_fields.py::build_layer_fields` 与
`orchestrator/l2_alloc.py::_is_dual_input` 原来对 `kind == "mask"`/
`op_type == "Mask"` 无条件当双输入处理。改成读 GML 节点自己的
`input_count`——只有真的接上 causal mask 边界节点（`input_count >= 2`）
才按双输入写 `Datain file 1`/`L2 input buffer offset 1`。

`l2_alloc.buffers_from_layers` 新增 `nodes_by_id` 参数（`plan.py` 调用处
把这份字典的构造提前，供分配 L2 之前用），`_is_dual_input` 新增可选
`node` 参数，读它的 `input_count`；residual/mlp_mul/rope_* 目前没有类似
的"可能退化成单输入"的已知场景，仍按 `kind`/`op_type` 判定，未改。

写这条修复时发现**同一个 bug 还藏在另一处**：`layer_fields.py` 里
`Datain file 0/1` 的构造用的是另一个独立的 `if kind in (...)` 分支
（不是 `dual` 那个变量），只按 `kind` 判，没读 `n_in`。补了单测才测出来
（先写的测试只断言 `number of inputs`，没断言 `Datain file 1` 是否存在，
第一次跑绿了但没测到点上）。

**验证**：`--seq-len 1` 导出仍会产生真实的（退化值但非空）causal mask
张量（`causal_mask_of(1)` 返回 `torch.zeros(1,1,1,1)`），所以用当前的
导出参数**没能构造出真实的单槎 Mask 场景**来做端到端复现——这一点如实
写明，不假装已经端到端验证过。改用直接测代码路径的方式补了三个单测
（`tests/test_layer_fields.py::test_mask_with_causal_edge_is_dual_input`/
`test_mask_without_causal_edge_is_single_input`/
`test_is_dual_input_reads_mask_edge_from_node`），构造一个
`input_count=1` 的 Mask 节点直接验证 `dual` 判据与 `_is_dual_input`
的行为。

### 6. 恢复被误删的域确认表段落

`docs/prepare_out-域确认表-20260918.md` 工作区里少了 60 行（§16 与
附录 A/B），确认与本轮无关（本次会话从未编辑过这个文件），已
`git checkout -- docs/prepare_out-域确认表-20260918.md` 恢复。

## 未处理的项（如实记录，不是本轮范围）

- o_proj/residual 的槎位与命名差异（见上文第 3 节，既有的残差旁路缺口）。
- `rope_add` 系列的 `Datain file`/`Dataout file`（语义名 vs 通用名，见
  `docs/prepare_out-代码评审2-20260921.md` §2.4）。
- `DDR Output/Input Orig Buffer Name` 系列（43+37+6=86 处，语义名/TVM 名，
  复核已指出这些**不是** TVM 名而是可推导的语义名，上一版文档「43 处是
  TVM 溯源名」的说法不准确，本次连带更正——它们是
  `dynamic_quantization_params_N_dequant_buff`/`key_cache_out` 这类可从
  自己的 label 推出来的语义名，不是 Relay 内部符号）。
- `net.ini` 末行多一个 LF、静态 Task 扇出表、`H/I/HD/S` 硬编码、
  `layer_fields.py` 1432 行超软上限、`_weight_role`/`layer_kinds` 死代码、
  `plan.py` 的静默降级——这些是 P2/P3，仍留给专门的清理轮次。
- 仓库根 509KB 的未跟踪会话记录文件——已确认是与本仓无关的终端会话
  转录，未删除（删除属于清理操作，留给用户确认后处理）。

## 增删文件（本次第二轮）

| 文件 | 改动 |
| --- | --- |
| `gml_bridge/runtime_files.py` | Softmax 五相改成 5 入 + 5 出对称写盘 |
| `gml_bridge/from_fx.py` | Softmax 字段声明同步改对称；RoPE 表节点改按 K 路消费者编号，新增 `rope_table_names` 让消费者读共享名 |
| `orchestrator/layer_fields.py` | `rope_mul_*` 的 `Datain file 0/1` 改读共享表名；Mask 的 `output scale factor buffer`/`Datain file 1` 改读 GML 边（`input_count`）判定双槎；`dual` 变量同步 |
| `orchestrator/l2_alloc.py` | `_is_dual_input` 新增 `node` 参数读 Mask 的 `input_count`；新增 `_dual_slot1_size()` 给 RoPE 的 `#1` 槎按真实表宽分配 |
| `orchestrator/plan.py` | `nodes_by_id` 构造提前到 L2 分配之前，传给 `buffers_from_layers` |
| `scripts/export_gml.py` | 写盘前清空 `out_dir/*.bin`；新增反向检查（盘上多余 bin，信息项）；双槎重叠判据修正 size0/size1 混用 |
| `tests/test_layer_fields.py` | 新增 3 个 Mask 双槎判定单测 |

## 验证方式（本次实测命令与结果）

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler

rm -rf /tmp/final_verify
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/final_verify \
    --use-opcompiler --orchestrate --decode-block-only

python scripts/diff_prepare_out.py --mine /tmp/final_verify/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out

python -m pytest tests/ -q -k "not llama2_7b"
```

实测结果：
- `export_gml.py`：**16/17** 项通过（含 3 个新增/改进的闸门：bin 引用
  闭合、反向多余引用信息项、双槎不重叠）。唯一未过的还是那 82 个既有缺失
  引用（weight_sf/Kantor/activation_lut/RoPE 语义名/残差旁白，见
  `docs/prepare_out-代码评审2-20260921.md` §2.3 的分类），本轮未处理。
- `diff_prepare_out.py`：`MATCH 45854 / VALUE_DIFF 104 / ALLOC 5846`，
  退出码非零。**104 这个数字本身没变**，但构成变了——`mul_cos`/`mul_sin`
  的槎位差异（复核 §2.2 指出的 4 处）已经修掉，被同样数量的既有
  `rope_add`/residual 差异顶替，总数刚好没变，这是巧合，不代表白改。
- `pytest`：**686 passed**（683 + 本次新增 3 个 Mask 双槎判定单测），
  0 failed，无回归。`llama2_7b` 标记的用例本次未重跑（同上一轮，未实测）。
- `parser_output` bin 数：**3259**（上一轮 3163，Softmax 对称模型补的
  96 个）。
- L2 数据区：**367744 字节**（上一轮因 RoPE `#1` 槎用错误尺寸虚耗到
  585792 字节，修完降下来）。
