# prepare_out 生成代码评审（2026-09-20）

评审对象：本仓当前**未提交**改动（`orchestrator/layer_fields.py`、`layer_hw_table.py`、
`layer_render.py`、`plan.py`、`layer_id.py`、`net_ini.py`、`graph/quant_pass.py`、
`gml_bridge/from_fx.py`、`export.py`、`scripts/export_gml.py`、
`scripts/diff_prepare_out.py`、`tests/test_layer_fields.py`，以及两份新文档
`docs/prepare_out-生成方案-20260919.md`、`docs/prepare_out-txt-20260919.md`）。

评审方式：通读全部新代码与文档；实跑文档中的生成/对拍命令；另写 4 个独立脚本绕过
官方对拍器的白名单与配对方式，直接核对「文件名、域集合、域值、引用闭环」。
所有结论都给了可复现命令与实测数字。

---

## 0. 结论摘要

**完成度**：422 个文件 + `net.ini` 的**骨架**（文件集合、23 类计数、每文件域集合、
执行顺序、`[general]` 字节）确实对齐了，这是实打实的进展。

**但「基本对齐」这个结论不能采信**，问题按严重度排序：

| 级别 | 问题 | 实测依据 |
| --- | --- | --- |
| **P0** | prepare_out 的 txt 引用了 **1051 个不存在的 `.bin`**（3405 个引用中）；参考物同口径 **0/3400 缺失**。产物在仿真器上会因找不到缓冲而失败 | 见 §3.1 |
| **P0** | parser_output 比参考少 **905 个 bin**：Softmax 5 相的相位缓冲族整族没落盘（`write_softmax_phases` 写了但**生产路径从没调用**）；mask 双槽/外部 mask、权重族也缺 | 见 §3.1 |
| **P0** | **41 个双输入层的 `L2 input buffer offset 0/1` 完全相同**（地址重叠）；参考两槽不同地址 | 见 §3.2 |
| **P1** | 对拍「VALUE_DIFF 0 / 结构差异只有 78 处」是白名单 + 统计口径造成的假象。按仓库对拍器自己的配对方式逐对分类，**结构差异是 173 处**；`skip compare`、`Datain/Dataout/Input buffer`、`Residual *` 等语义键被整体放进 ALLOC 白名单 | 见 §3.3 |
| **P1** | 值的主要来源是**参考产物反推的静态表**（`layer_hw_table.py`）＋编排器硬编码（`H/I/HD/S=4096/11008/128/1024`），不是从 pimmlir 映射；Task 扇出是抄的静态邻接表，生成方案承诺的「有 PhaseSource 时按 SSA 连」没有实现 | 见 §4.1 |
| **P1** | GML/runtime 侧命名与 txt 侧命名是**两套规则**，这是 1051 处悬空引用的根因 | 见 §3.1 |
| **P2** | `net.ini` 末行比参考多一个 LF；文档多处与实现/实测互相矛盾；`layer_kinds`、`_weight_role` 是死代码；`layer_fields.py` 1395 行（超过仓规约 400 行软上限的 3.5 倍） | 见 §5 |
| **P3** | 仓库根多了一个 7096 行的未跟踪会话记录文件；`export_gml.py` 无条件 `unlink` 输出目录里的 `*.txt` | 见 §5.4 |

建议：**先修 P0 的三个闭环问题，再把对拍口径收紧并补测试，最后再谈「从 pimmlir 映射」的迁移**。
在这之前不宜把「422/422、VALUE_DIFF 0」当作可交付的证据。

---

## 1. 我跑了什么（可复现）

环境：`source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh`

```bash
# 1. 文档里的生成命令（约 2 分钟）
rm -rf /tmp/review_gml
python3 scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/review_gml \
    --use-opcompiler --orchestrate --decode-block-only
# 实测：14 项自检全过，GML 199 节点 / 2330 bin，编排 428 层 → 422 txt

# 2. 官方对拍
python3 scripts/diff_prepare_out.py --mine /tmp/review_gml/prepare_out \
    --ref /media/.../llama2_w4a8_decode_block_0/prepare_out
# 实测：MATCH 45854 / VALUE_DIFF 0 / MISSING 0 / EXTRA 0 / ALLOC 5950 / 退出码 0

# 3. 域集合逐文件核对（官方配对）：422/422 完全相同 ✅
# 4. 独立另写脚本做四件官方没做的事：
#    a) 文件名模式集合（数字归一后）：24 种模式完全相同，394 对直接同名模式配对成功
#    b) 逐对逐键再分类（不按 key 的“第一个例子”归并）：结构差异 173 处，不只是 78
#    c) txt → parser_output 引用闭环：1051/3405 悬空（参考 0/3400）
#    d) net.ini 顺序：422/422 模式一致；[general] cmp 逐字节一致

# 5. 快速回归
python3 -m pytest tests/ -q -k "not llama2_7b"
# 实测：683 passed（与文档一致）
```

`python3 -m pytest -k "llama2_7b"`（文档称 42 passed / 33 分钟）本轮未复跑，
不做结论；但注意这条回归只证明 NumPy 执行链路没坏，**不覆盖 prepare_out**。

---

## 2. 做对了的部分（应当保留）

1. **文件集合与顺序**：424 个 txt（422 层 + 2 版本戳）、23 类计数、`net.ini [layers]`
   422 行、按模式归一后的执行顺序 422/422 与参考一致。`_order_like_reference` 虽然
   是启发式，但当前配置下结果正确。
2. **每文件域集合 422/422**：用官方配对实测确认，无 MISSING/EXTRA。
3. **`[general]` 逐字节一致**（含 CRLF 混排），`gml_version.txt` 逐字节一致，
   422 个层文件全 CRLF。
4. **闭合公式域的数值**：`Output Stride Z`、`Data scale width`、各 `L2 * buffer size`、
   dtype/枚举类键逐对比较均为数值相同；只有「只差编号」的键在变。
5. **图侧两处修正是真进步**：DQ 按激活源共享（`graph/quant_pass.py:179-187`）、
   cos/sin 建成边界节点（`from_fx.py` convert 尾部），有对拍数据支撑。
6. **文档诚实**：Q-A/Q-B/Q-C、FlagTree F1–F6、`gml_hw_table` 未退役都写清楚了；
   这比「假全绿」有价值得多。

---

## 3. 阻断级问题（P0）

### 3.1 txt 引用的 bin 有 1051 个不存在，parser_output 比参考少 905 个文件

**证据**（脚本逻辑：把 txt 里所有以 `.bin` 结尾的域值取出，和 `<out-dir>/*.bin` 求差）：

```
我方  : 引用 3405 个 .bin，缺失 1051
参考  : 引用 3400 个 .bin，缺失 0
parser_output: 我方 2330 个 bin，参考 3235 个
```

缺失按层类（缺失/引用）：`sm_p2 160/256`、`sm_p3 160/224`、`sm_p4 192/256`、
`sm_p5 128/224`、`sm_p1 128/224`、`mask 128/224`、`bmm1 96/352`、
`bmm2 32/352`、`residual/mlp_mul/gemm_v` 各数处。典型：

```
mha_softmax_head10_..._act_sm_phase4....txt  引用 Bias_buffer_phase_3_127.bin
                                              实际盘上只有 input_buffer_127.bin
mha_masking_head0_...txt                      引用 input_buffer_0_178.bin / input_buffer_1_178.bin
                                              实际盘上只有 input_buffer_178.bin
mha_batch_matmul2_head0_...txt                引用 Dataout input_buffer_0_19.bin
                                              参考同位置是 input_buffer_14.bin（不带槽）
```

参考的 `parser_output` 里这些文件都有（`Bias_buffer_phase_N_18.bin` 等），
而且**参考 txt 的引用 0 缺失**——这就是「可在仿真器上正常执行」的最低要求。

**根因**（都是可定位的代码问题）：

1. **Softmax 相位族整族没落盘**。`gml_bridge/runtime_files.py:291` 有
   `write_softmax_phases()`，但 `gml_bridge/export.py:227-243` 的
   `write_runtime_files` **没有 import、也没有调用它**；`from_fx.py:801-848`
   只给 `DynamicScaling` / `Llama2ActivationDQ` 写 `*_phase_N` 字段，
   Softmax 节点一个相位字段都没有。编队器却在
   `layer_fields.py:742-771` 按 Softmax 节点号拼出了整套 `input_buffer_phase_N`、
   `Bias_buffer_phase_N`、`Scaling*_phase_N`、`LUT_phase_N`——**下游名对不上游写盘**。
   注意：即使接上 `write_softmax_phases`，其内部 exp 表也还没实现
   （`runtime_files.py:333`「exp 表尚需拷贝」），要做到参考的 138 个 LUT 二进制文件
   还差一块。
2. **mask 的第二个输入（外部 causal mask）在图里就是缺的**（与文档只提的
   「残差旁路边未接」同类）。`from_fx._data_slot_count("Mask", ...)` 给 1 个槽，
   GML 写 `input_buffer_178.bin`；编队器在 `layer_fields.py:776-787`
   却按双槽发明 `input_buffer_0/1_<mask>.bin`。于是 bmm1 的 `Dataout`（编队器补槽后）
   与 mask 的 `Datain`（编队器补槽后）两两自洽，但和 GML 实际写出的
   `input_buffer_178.bin` 对不上。
3. **bmm2 输出**：`from_fx` 给 Concat 消费的 32 路输入按 `first_slots=32` 编号，
   写成 `input_buffer_<h>_19.bin`；参考 32 个头**都写同一个 `input_buffer_14.bin`**。
   编队器 `_slotted_dataout` 只强化了这个差异。
4. **K/V cache、RoPE 表、相位引用**：`key_cache_out`/`value_cache_out` 只实现了 v；
   K add 仍写 `buffer<id>`（`layer_fields.py:1179`），参考是 `key_cache_out`；
   RoPE 表缓冲参考用 `<semantic_label>_cos.bin/_sin.bin`，我方用
   `input_buffer_<n>_<m>.bin`；Q-RoPE DQ 的 `Datain` 参考是
   `input_buffer_phase_0_<rope>.bin`/`..._mid_buf`，我方写 `input_buffer_<dq>.bin`。
5. **bmm 的 `weights scaling buffer file` 32+32 缺失**：该字段没有对应的落盘
   （GML 里 bmm 没有 `weight_sf`），编队器回落到 `names.weight_scale(nid)` 凭空生成。

**建议修法**（顺序很重要）：

1. **先加闭环校验，再修 bug**：在 `export_gml.py` 的 14 项自检里加一项
   「prepare_out txt 引用的 `.bin` 集合 ⊆ parser_output 落盘集合」，不通过就退出 1。
   这一步做完，1051 会立刻变成可见的红灯，后面每修一类就减一类。
2. **命名单一真源**：现在 GML 写盘（`from_fx` + `runtime_files`）与 txt 拼名
   （`layer_fields`）各有一套 slot 规则，必然漂移。短期让 `layer_fields` **只读
   GML 节点已有的 `input_buffer*` / 相位字段**，不再自己拼；长期把命名函数统一到
   `contracts/gml_names.py` 一处，两个消费方都调它。
3. **Softmax 相位补全**：给 Softmax 节点在 GML/IR 里打上 5 相字段（真源是
   FlagTree `-pim-expand-phases`，它已经展开 Softmax 5 相），`export.py` 调
   `write_softmax_phases` 落盘；补上 exp 表。
4. **mask 外部输入补边**：与残差旁路是同一类问题。要在命名契约上先定
   「一个入口缓冲喂多个消费者时 `output_buffer` 指谁、其余槽怎么命名」，
   这会影响全部 bin，必须先设计再动（文档 §4 已记，但影响面不止 `input_count`）。

> 补充：文档 §「当前不足 2」说「我方 `weight_buffer_195.bin` 确实落盘了，引用不悬空」——
> 这一句只对**部分**键成立。全量口径是 1051/3405 悬空，必须更正。

### 3.2 双输入层的两个 L2 输入槽地址重叠

**证据**：41 个双输入层（mask 32 + residual 2 + mlp_mul 1 + rope 6）
全部是 `L2 input buffer offset 0 == L2 input buffer offset 1`。例：

```
mine mask head0 : offset0 = offset1 = 176128
ref  mask head0 : offset0 = 0, offset1 = 536788928
```

而这个键在官方对拍器里属于 `ALLOC_KEYS`（`diff_prepare_out.py:30-31`），于是「地址不同」
被当成正常的分配差放过了。但**同一层两个输入槽指向同一段 L2** 不是「分配策略不同」，
而是布局冲突：两块输入缓冲互相覆盖。单输入层同样可疑：`L2 input buffer offset 0`
填的是 `l2_offsets[identity.stem]`，即**本层输出**的分配偏移，不是上游输出的位置
（`layer_fields.py:1251/1258/1272` 与 `1282/1287` 用了同一个 `l2_offsets` 条目）。

**建议**：

- `l2_alloc` 的分配单位应从「层」改成「缓冲」，双输入层分配两块，`layer_fields`
  用 `offsets[(stem, slot)]` 取；
- 或者至少 `offset 1 = offset 0 + size 0`，保证不重叠；
- 顺带把 ALLOC 白名单里的 `L2 input buffer offset` 拆成「只允许值不同，不允许相等」
  的检查（相等直接判错）。

### 3.3 「VALUE_DIFF 0」是口径问题，真实的域值差异是 173 处

官方对拍器本身没有算错，它按白名单把 5950 处归入 ALLOC（`diff_prepare_out.py:28-92`）。
问题在**白名单里塞进了语义键**：`Datain file`、`Dataout file`、`Weights buffer file`、
`Input buffer file`、`Scaling*/Bias*/Kantor* buffer file`、`output scale factor buffer`、
`Original name`、以及所有 `Residual *` / `Virtual *` / `DDR * Orig Buffer Name`。

文档 §4 的「原始核对」脚本又用 `ex.setdefault(k, ...)` **按 key 的第一个例子**
决定整族算「只差编号」还是「结构语义」，于是 173 处里有 95 处被误记成「只差编号」，
得到「结构语义 78 处」。复现：

```bash
# 用同一份 load_dir，但逐对分类（不按 key 的 first example）：
# 结构语义 173 处；只差编号 5777 处
```

173 处的完整分类（`norm(digits->#)` 后仍不同）：

| 处数 | 键 | 参考 | 我方 | 性质 |
| --- | --- | --- | --- | --- |
| 43 | `DDR Output Orig Buffer Name` | `<dq label>_dequant_buff`、`key_cache_out`、`..._mul_cos_buffer` | `buffer<id>` | P0 命名错误，可从 label 推导 |
| 39 | `Dataout file` | bmm2 `input_buffer_14.bin`；k_proj `input_buffer_0_30.bin` | bmm2 `input_buffer_<h>_19.bin`；k_proj `input_buffer_191.bin` | P0 槽位规则不一致 |
| 37 | `DDR Input Orig Buffer Name 1` | `nprm_182_i168`（TVM）/ `..._sin_buffer` | `mask` / `buffer1` | 一部分 TVM，一部分命名错 |
| 34 | `DDR Input TVM Orig Buffer Name 1` | TVM 名 | 语义名 | 等甲方 Q-A，可接受 |
| 6 | `DDR Input Orig Buffer Name 0` | `..._mid_buf`、`..._cos_value_buffer`、`..._mul_cos_buffer` | `buffer0` | P0 命名错误 |
| 4 | `Datain file 0` | 入口 `input_buffer_25.bin`；K add `..._cos.bin` | `input_buffer_0_14.bin`；`input_buffer_0_182.bin` | 残差旁路 / 表命名 |
| 2 | `Datain file` | `input_buffer_phase_0_22.bin` | `input_buffer_184.bin` | Q-RoPE 相位命名 |
| 2 | `Input buffer file 0` | 同 `Datain file 0` | 同 | |
| 2 | `DDR Input TVM Orig Buffer Name 0` | TVM | 语义 | 可接受 |
| 2 | `Datain file 1` | `..._sin.bin` | `input_buffer_1_182.bin` | 表命名 |
| 1 | `output scale factor buffer` | `input_0_sf_9.bin` | `input_sf_13.bin` | residual 的 sf 编号规则没落对 |
| 1 | `DDR Output TVM Orig Buffer Name` | TVM | TVM 格式差个 `_182` | 等甲方 |

另外还有两类**被白名单藏起来的伪造值**（不是编号差，是编造）：

- `residual`（`add_1`）的 `Residual input buffer 1`（`layer_fields.py:936-937`）
  被填成和槽 0 相同的值，参考是两个不同的上游（入口旁路 vs RMSNorm）；
- `mask` 的 `Residual input buffer 1` 缺边时被填成字面量 `0`（`layer_fields.py:925-929`），
  参考是真实节点号。域在、可解析，但语义是错的。

**建议**：

1. 文档 §「当前不足 #1」的 78 处改成 173 处的口径，并把 173 按上表分类；
   即使短期不修，也要如实进「当前不足」，不能只留 TVM 名一类。
2. `diff_prepare_out.py` 的白名单只保留「纯地址 + TVM 名 + 纯节点号」；
   `Datain/Dataout/Input buffer/*state*/buffer file/Original name` 等**命名类键**
   应从白名单移出，改为「模式同构则 PASS，否则 FAIL」——因为命名模式本身是
   可推导、可对齐的。
3. `diff_prepare_out.py` 中 `counts_mine` 只打印不判定（`:289-299`），
   建议 `counts_mine` 也和 `REFERENCE_COUNTS` 比对。

---

## 4. 高优先问题（P1）

### 4.1 值不是从 pimmlir 映射的，是「参考产物反推 + 静态查表 + 编排器硬编码」

需求原文要求「保证是从图编译 → 算子编译（pimmlir）映射得到的，而不是硬算的或者绕开」。
当前实际：

| 来源 | 位置 | 例子 |
| --- | --- | --- |
| 参考 422 层实测反推的常量表 | `orchestrator/layer_hw_table.py:94-174` | `l2_fpsu_size` 28672/57344/77824、`l2_wscale_size`、`l2_weights_off0/off1`、`flp=(10,17,3)`、`transpose_type` |
| 编排器硬编码 | `orchestrator/layer_fields.py:22` | `H, I, HD, S = 4096, 11008, 128, 1024`；bmm 宽直接 `= HD, S`（`:611-626`）；`Num Output Heads=32`（`:833/1044`）；`DDR Weight Width=4096`（`:1194-1200`）；`Group data size=1024/128`（`:1085-1088`） |
| 参考产物穷举的邻接表 | `orchestrator/layer_id.py:34-48` | `_DQ_LINKS` / `_SOFTMAX_LINKS` 写死 fan-out |
| 算子编译器只提供了两样 | `opcompiler_bridge/` | 相位数（决定展开层数）与 `phase-bytes`（只进 L2 分配） |

而 `from_fx.py` 里已经拿到了 FlagTree 展开后的相位模板（`pim.phase`、
`pim.phase-bytes`、`unit`、`kind`、`force-consecutive`、`rotate-half`），
但 `layer_fields` **一行都没读**：Flp、Activation mode、Transpose type、Kantor、
LUT kind 全部来自静态表。文档 F1–F6 承认了这一点，但**这正是不满足验收点 3 的地方**，
不应该只作为「遗留」。

**更具体的缺口**：生成方案 §1.3.1 承诺「有 PhaseSource 时按 SSA 使用-定义连，
没有时抄静态邻接表」；代码里 `_task_links` 只有静态表分支，`PhaseSource` 参数
根本没传到 `assign_ids`。所以 Task 扇出也是抄的。

**建议路线**（与档 F1–F3 一致，但要求给出可验收判据）：

1. `FlagTree/PIMAttrDefs.td` + `ExpandPhases.cpp`：把 `flp-min-exp/max-exp/mantisa`、
   `activation-mode`、`activation-special`、`pooling-type`、`kantor-mode`（数字）、
   `fpsu-mode`（数字）、`transpose-type`、`lut-kind` 打在相位 op 上；
2. `phase_plan.py` 解析这些 attr（现在只有 `_KIND_RE` 能读 kind）；
3. `layer_fields` 优先读相位数据，静态表只做 fallback，并加**反证测试**：
   改 IR 里某个 attr，txt 对应域必须变（类似现有 GML 反证，但作用在 txt 上）；
4. Task 扇出改从相位 op 的 SSA 使用-定义推导，`_DQ_LINKS/_SOFTMAX_LINKS` 降级为
   PhaseSource 不可用时的 fallback；
5. `H/I/HD/S/32` 一律从 `config.json` + GML 边读，不在编排器写死。

在 1–3 完成前，「从 pimmlir 映射」的验收点无法通过，文档标题不要写成已打通。

### 4.2 命名契约分裂（P0-3.1 的根因）

`contracts/gml_names.py` 是唯一命名真源，但消费方式分裂：

- GML 写盘：`from_fx` 按「消费者的槽数」命名 `output_buffer`（`_data_slot_count`），
  `runtime_files` 按 GML 字段名写文件；
- txt 生成：`layer_fields` 又按自己的规则重算 `names.data_buffer(nid, slot)`、
  `names.phase_*`、`_slotted_dataout`。

两边对「Mask 是几槽、Concat 怎么编号、Softmax 相位归谁」的答案不同，就产生 1051 处悬空。
**验收时必须能归因到「哪个函数写错了」**，所以建议把命名收敛为一个函数，
并给它加单测：给定图 → 同一函数输出 GML 字段名与 txt 域值，断言逐字节相等。

### 4.3 启发式与静默降级

- `plan.py:182-185` `_order_like_reference` 用 `try/except Exception` 吞掉分类异常，
  静默退回 `op_type`；排序错了不会报错，只会让顺序和文件名悄悄漂移。
  建议：异常必须抛出（CLAUDE.md 也要求「不写防御性兜底」）。
- `plan.py:150-166` `_drop_tail` 靠 FX 名字符串（`"mul_12"`、`"linear_7"`）丢层，
  换模型/换图就失效；应由「是否属于本 decode block」的结构信息决定。
- `layer_fields.py:185-193` `_kv_cache_buffer` 靠 `node_id` 大小猜哪个 DMA 是 K；
  应从 `KV_DMA_META_KEY` 的 meta（已经有）或 IR 角色标记取。
- `layer_fields.py:153-174` `_upstream_dq` 有 `_depth>6` 截断与折叠算子白名单；
  这是命名契约不完整的补丁，建议并入统一命名函数后删除。

### 4.4 测试与验收缺口

- 没有 `test_diff_prepare_out.py`（生成方案 §3 计划要加），对拍器本身无测试；
- `_order_like_reference`、`_drop_tail`、`--decode-block-only` 无单测；
- `tests/test_layer_fields.py` 只有 7 个用例，23 类里大量类没有夹具，
  且只断言「键在不在」，不断言**关键闭合值**（更难发现槽位/命名回归）；
- 14 项自检里没有「txt 引用 → parser_output」的闭环检查（本次评审发现 1051 处悬空）。
  建议至少加这三项：
  1. txt 引用 `.bin` 全部存在；
  2. 双输入层 `offset 0 != offset 1`（且区间不重叠）；
  3. 层执行序与参考骨架同构（可直接用本仓 `diff_prepare_out` 的 `net_ini_order`）。

---

## 5. 中低优先问题（P2/P3）

### 5.1 `net.ini` 末尾多一个 LF

参考 `net.ini` 最后一个 `layer = ...` **没有换行**（`xxd` 末字节是 `39`）；
我方 `render_layers_section` 给末行加了 `LF`（`net_ini.py:38-41`），因此整文件比
参考多 1 字节（LF 计数 436 vs 435）。文档「参考最后一行是 LF」的描述不成立，
`test_net_ini_line_endings_match_reference` 也没覆盖 EOF。建议末行不加换行，
或至少把这个字节差异记成已知项。

### 5.2 文档与实现/实测的矛盾

| 文档 | 文档说 | 实际 |
| --- | --- | --- |
| `prepare_out-生成方案 §1.3.4` | RoPE 的粘连**拆成两行**，「我方不复现粘连」 | `layer_render.py:42-47` **粘成一行**（研发记录里写的是粘，方案没同步） |
| `prepare_out-txt §当前不足` | 结构语义只剩 **78** 处 | 173 处（统计口径缺陷，见 §3.3） |
| `prepare_out-txt §2.1` | 已对齐的几何/尺寸「值内部自洽、引用不悬空」 | 1051/3405 悬空（§3.1） |
| `prepare_out-生成方案 §1.3.6` | `--reference-dir` 时跑全量对拍 | CLI 没有这个参数 |
| `prepare_out-txt §增删文件` | 净增 +2140/-142、新文件约 1720 行 | `layer_fields.py` 1395 + `layer_hw_table` 180 + `layer_render` 53 + `diff` 379 + 测试 132 ≈ 2139 行（代码），另有 1028 行 md；`layer_fields` 从 1161 改到 1395 |
| `layer_expand.py:62` 注释 | `label` 是 GML label 即 FX 名 | 现在 `Layer.label` 已经是 FX 名，GML label 是语义名，注释误导 |
| `plan.py` 模块 docstring | 「200 节点 → 422 层」 | 现状是 199 节点 → 428 层（`--decode-block-only` 后 422） |

建议发版前把两份文档与实测数字对齐；「当前不足」按 §3.3 分类重写。

### 5.3 代码质量

- `orchestrator/layer_fields.py` 1395 行，硬约束是 ~400 行；方案里也承认要拆。
  建议按 DQ / Softmax / Gemm-BMM / Eltwise-RoPE-Misc 拆四个模块，共用 `_base`。
- 死代码：`OrchestrationPlan.layer_kinds` 只写不读（`plan.py:141/147`）；
  `layer_fields._weight_role` 无调用方；`semantic_stem` 里
  `dq_` 的两个分支返回同一字符串（`:280-284`）。
- `layer_hw_table.py` 的每个常量建议标注来源（手册哪张表/Q 编号/参考层类），
  并给「换 hidden 要重测」的值加 `pending_q`（现在只有注释，字段 `pending_q`
  在 dataclass 里定义了吗？`HwRow` 没有 `pending_q` 字段，docstring 说「每个条目可带
  `pending_q`」——又是一处文档与实现不符）。
- `layer_fields.py:1325/1326` 在末尾重复写了 `Bytes in cycle...`（`CONSTANTS` 里已写过），
  无害但属于冗余。

### 5.4 仓库卫生与破坏性操作

- 未跟踪文件 `2026-09-19-165121-local-command-caveatcaveat-the-messages-below.txt`
  （7096 行、约 509KB 的 AI 会话记录）留在仓库根目录。它不属于交付物，包含完整
  环境路径与内部讨论，建议删除，并考虑加进 `.gitignore`（例如 `*-local-command-*.txt`）。
- `scripts/export_gml.py:324-325` 无条件删除 `out_dir/prepare_out/txt_files/*.txt`。
  若用户误把 `--out-dir` 指到参考目录或已有产物目录，会直接删掉别人的文件。
  建议：只删本次将要写出的名字，或要求目录里有本工具写的标记文件才允许清理。
- `.gitignore` 新增 `gml-artifacts/` 是合理的（产物很大）。

---

## 6. 建议的修改顺序（按依赖）

1. **加闸**（半天）：在 `export_gml.py` 自检里加「txt→bin 引用闭环」「双槽不重叠」
   「顺序同构」三项；把 §3.3 的 173 处作为已知清单落到 `docs`。
2. **统一命名**（1–2 天）：抽出唯一命名函数，`from_fx`/`runtime_files`/`layer_fields`
   都走它；先消除 txt 侧自行发明名字的路径（`_slotted_dataout`、
   `layer_fields.py:776-787`、`:1251-1290`）。
3. **补落盘**（2–3 天）：Softmax 5 相（接 `write_softmax_phases` + exp 表）、
   mask 第二槽、bmm 权重 scale、`key_cache_out`、RoPE 表/相位命名。
   每修一类，第 1 步的闸门数字必须下降。
4. **修对拍口径**（1 天）：白名单只留地址/TVM/纯节点号；命名类键按模式比对；
   `counts_mine` 参与判定；修正文档 §「原始核对」的统计方法。
5. **迁移值来源**（按 F1–F3，数天）：IR attrs → `phase_plan` → `layer_fields`，
   静态表退化为 fallback，加 txt 级反证测试；Task 扇出改 SSA。
6. **收尾**：文档数字与实现对齐、拆 `layer_fields`、删死代码、清仓库根的大文件。

---

## 7. 建议加进 CI 的验收清单

```bash
# 1. 现有
python -m pytest tests/ -q -k "not llama2_7b"
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/ci \
    --use-opcompiler --orchestrate --decode-block-only
python scripts/diff_prepare_out.py --mine /tmp/ci/prepare_out --ref <参考>

# 2. 新增（本次缺的）
# a) txt 引用的 bin 必须都在 parser_output 里（当前 1051 缺失）
# b) 双输入层 offset0 != offset1 且区间不重叠（当前 41 层重叠）
# c) 每文件的关键闭合值抽查（不止键存在）
# d) net.ini 末行字节、LF/CRLF 计数
# e) 改一处 pim.phase 属性 → txt 对应域必须变（真正的“从 IR 映射”反证）
```

---

## 附：评审中未发现问题的项（避免误伤）

- `[general]` 段、`gml_version.txt`、CRLF、文件数、23 类计数、每文件域集合、
  执行序、以及所有尺寸/枚举/模式类键的值——这些经独立脚本核对，确实对齐；
- `graph/quant_pass.py` 的 DQ 共享逻辑有正确的早退保护（`:147-148`），
  不会有双重 DQ；`consumers` 列表边改边用是安全的；
- `layer_render.py` 的「粘连」选择有参考实测支撑（6 个 RoPE 文件里 5 个粘连，
  K add 无 `skip compare`），虽然与方案文字矛盾，但实现本身是对的；
- 683 个快速测试确实全过，说明改动没有打坏既有链路。
