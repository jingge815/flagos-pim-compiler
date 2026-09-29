# 实施记录：GML 与 bin 产物对齐甲方参考格式

> 文档编号：impl-gml-align-20260928
> 完成日期：2026-09-29
> 关联需求：`docs/request-gml-align-20260928.md`
> 关联设计：`docs/design-gml-align-20260928.md`
> 关联评审：`docs/review-gml-align-20260928.md` 及 `-r2` ~ `-r8`（共八轮）
> 实施方式：TDD（先写测试、确认失败、再编码、再确认通过）

## 一、实施概述

把我方 GML 图与 bin 运行时文件对齐甲方参考产物（`model_layers_0_decode_v2`）的格式。
设计文档的 10 个任务单元全部落地，八轮评审提的 41 项全部处置完毕。

| 项 | 结果 |
| --- | --- |
| 任务单元 | 10/10 完成 |
| 评审项 | 41 项全部处置（八轮：9 + 4 + 4 + 5 + 5 + 7 + 5 + 2，其中 6 项的根因比评审定位的更深，按实测修正） |
| 验收判据 | 20 条**全部通过**（含 r8 新增的 A18） |
| 全量回归 | `981 passed, 1 skipped, 42 deselected` |
| 真实导出 | 200 节点 / 331 边 / 3122 个 bin，`op_type` 分布与参考 **diff 为空** |
| 文件族对账 | 72 个文件族，字节数**全部一致**，「仅我方」「仅参考」两个差集都为空 |
| 代码增量 | 源码 **+946 / -162** + 新增 `gml_bridge/calib_data.py` 796 行；测试 **+1691 / -0** + 新增 `tests/test_calib_data.py` 168 行 |

核心成果分三类：

1. **标定数据接通**。原先大量 bin 写的是零张量或写死的 1.0，表示「这一步还没算」。
   现在图入口的 5 个张量嵌成源码常数，四条入口缓冲与参考**逐字节相同**。
2. **取值口径归一**。同一件事原先有多处各算一份（`A` 与 `input0_node_id`、Kantor 的
   scale 与 Shift、`output_sf` 的两个写者），改为单一真源，由构造保证一致。
3. **判据可核对**。每条验收判据都落成能跑的测试或命令；原先「措辞比证据强」的
   A1/A3/A16 按实际核对范围改写，新增 A18 守住评审发现的配对盲区。

## 二、实施原理

### 2.1 两层产物、三条数据源

导出产物分两层：一份 `relay2gml_graph.gml`（图结构 + 每个节点的字段），以及字段
指向的几千个 `.bin`（权重、激活、量化定标）。GML 里的字段名就是 bin 的文件名，
两者必须闭合——引用了必须落盘，落盘了必须被引用。

```
PyTorch 模型
    │ torch.export
    ▼
  FX 图 ──► gml_bridge/from_fx.py ──► GmlArtifact（节点 + 边 + 字段）
                 建节点、打 dtype、标角色、                │
                 裁剪 decode block、重编号 1..N           │
                                                          ▼
                              gml_bridge/export.py ──► relay2gml_graph.gml
                                 按字段名分派谁来写         + 3122 个 .bin
                                        │                  + IO_info.txt
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            calib_data.py         phase_data.py      runtime_files.py
            标定常数（源码内嵌）    动态量化四相公式     落盘 + 分族写值
```

三条数据源各管一段，互不越界：

| 模块 | 职责 | 关键接口 |
| --- | --- | --- |
| `calib_data.py` | 标定常数的唯一真源 | `calibration_for_role(role, numel)`、`activation_for(numel)`、`kv_cache_scale(is_key=)` |
| `phase_data.py` | 动态量化四相公式（本次只加可读打印，公式未动） | `dynamic_scaling`、`DynamicScalingPhases.__repr__` |
| `runtime_files.py` | 按族决定宽度与内容，落盘并记账 | `write_data_buffer`、`write_dq_phases`、`write_rope_buffer` |

### 2.2 标定数据为什么内嵌成源码常数

需求 P0-1 要求导出不读甲方目录下任何 `.bin`——参考目录改名后导出仍要成功。所以
5 个小张量全量嵌进 `calib_data.py`：`HIDDEN_STATES` 4096、`ATTENTION_MASK` 1024、
`COS_EMBEDDING` 128、`SIN_EMBEDDING` 128、`CACHE_POSITION` 96，合计 5472 个值。

KV cache 各 419 万个元素，全量嵌入要 112MB，而标定链路只用到 absmax，所以只嵌两个
标量（`KEY_CACHE_ABSMAX = 35.222347`、`VALUE_CACHE_ABSMAX = 38.633411`）。

**按角色取常数，不按位置猜**。`from_fx` 建边界缓冲节点时打一个内部标记
`pim_calib_role`（取值 cos / sin / mask / kv_position），`write_data_buffer` 照这个
标记去取对应常数。角色只有建节点的地方知道（RoPE 匹配结果决定哪张表是 cos），
所以在那里标；靠槽号或名字子串反推都会在改名后静默取错。`pim_` 前缀的字段会被
`writer.py` 从 GML 文本里滤掉，只在编译期传值。

整数常数**按值落盘、不做量化**：位置索引是 96 个 0，若按 absmax 归一到满量程会
变成噪声（评审 r7 实测改前是 min −32767 / max 29129）。

### 2.3 四相公式与它落到哪些文件

动态量化把一段激活按 `group_size` 分组，逐组算出四个相：

| 相 | 含义 | 落盘文件 |
| --- | --- | --- |
| phase0 | 2 × 逐组 absmax | `output_buffer_phase_0_<id>.bin` |
| phase1 | requant scale（= `output_sf`） | `output_buffer_phase_1_<id>.bin` |
| phase2 | kantor scale（= 1 / phase0） | `output_buffer_phase_2_<id>.bin` |
| phase3 | 量化后的 int8 | `output_buffer_phase_3_<id>.bin` == `output_buffer_<id>.bin` |

公式本身经独立复算成立：拿甲方自己的 `input_buffer_12` / `_193` 喂进我方公式，
phase0 与 phase1 与甲方文件**逐字节相同**（节点 12 的 32/32 组、节点 193 的 86/86 组）。

### 2.4 零值分类：哪些该改、哪些必须留

原先几千个 bin 里有大量全零。不能一律改成非零——有些零是硬件语义，有些零表示
「还没算」。判定原则是后者才改：

| 族 | 处置 | 依据 |
| --- | --- | --- |
| DQ 四相的源张量 | **改**：吃标定激活 | 参考此族逐元素非零 |
| Softmax 五相的源张量 | **改**：吃标定激活 | 同上（评审 r1 发现的漏项） |
| 数据缓冲 `input_buffer_*` | **改**：按角色取标定常数 | 参考与常数逐字节相同 |
| RoPE 定标族 / cos·sin 乘积 | **改**：定标发 1.0、乘积取标定乘积 | 参考逐族实测 |
| 零点 `*_zp_*` | **保留全 0** | 参考 137 个逐字节相同 |
| Kantor / FPSU 的 bias、post_shift | **保留全 0** | 参考实测恒 0 |

### 2.5 定标取值：按判据分派，识别不到就抛

`output_sf` 等定标字段原先大量写死 1.0 或 0.0。改为按算子类型与角色分派，**白名单外
直接抛，不回落 1.0**——回落会让「新增算子忘了接定标」这类 bug 永远静默。

```
_output_scale_of(node, by_id)
  ├─ _kv_role_of(node) 非空 ──► kv_cache_scale(is_key=)   # DMA 看自身 pim_kv_is_key；
  │                                                       # Split 回溯上游那块 cache
  ├─ Gemm 的 v_proj ──────────► value cache 的 scale
  ├─ 白名单内的 fp16 域算子 ──► 1.0（真值，不是兜底）
  └─ 其余 ───────────────────► raise
```

Q 路上那个 Split 的上游是 RoPE 而不是 DMA，回溯不到 cache 角色，于是落到白名单
拿到 1.0——这是正确取值，不是兜底。

`input_sf` 同构，但走到的分支不同——定点数据的逐组 scale 属于产生它的那个 DQ，
由 DQ 自己写盘、消费者只引用所引用的文件名，所以 `_input_scale_of` 只管两类
自命名来源：KV cache 那一路取那块 cache 的 scale；fp16 域的边发 1.0（真值）。
**int8 却还是自命名的直接抛**——那说明这条边的缓冲与定标出自两个不同口径
（r8 修掉的配对错位），报错消息里点名该引用哪个文件。这条取值路径已删，换成不变量。

Kantor 四族的 scale 与 Shift 共用同一个判据——节点上的 `kantor_mode` 字段：

| `kantor_mode` | 含义 | scale | Shift |
| --- | --- | ---: | ---: |
| `fp2int_converter` | 做定点化 | 那次量化 scale 的倒数 | −8（左移 8 位，×256） |
| `elementwise_mul_fp16` | 不改变量化域 | 1.0 | 0 |

原先 scale 按 `kantor_mode` 判、Shift 按文件名前缀判，今天巧合等价，但一族改名就
会分叉。改为同一个真源后，两者是同一件事的两半（×256 与 1/scale），不会各改一半。

### 2.6 一致性由构造保证，不靠两处平行修改

八轮评审里最严重的一项（r3）是这类问题的典型：`A` 字段在建节点时写成
`fields["A"] = fields["input0_node_id"]`，而节点重编号只认三类名字（`*_node_id` 结尾、
`residual_*_buffer`、`.bin` 结尾），单字母键 `A` 三类都不匹配。结果 `input0_node_id`
压到新号、`A` 留旧号——旧号区间 3..206 与新号 1..200 重叠，改出来的 `A` **仍是合法
id**，不悬空不报错，只是指向了另一个算子。5 条结构规则与当时全部单测都拦不住。

改法不是「把 `A` 加进重映射表」，而是在重编号收尾处从 `input0_node_id` **重新派生**：
两个名字只有一个来源，下次再加别名字段也不会漏。同一原则用在另外三处：

| 场景 | 原先 | 改为 |
| --- | --- | --- |
| hidden 入口 rank | 取「节点列表里第一个」带 rank 的缓冲 | 认 `pim_is_hidden_entry` 角色标记，标了两个就抛 |
| 边界缓冲的位宽 | `{e.source: e.target}` 字典推导，后写覆盖先写 | 按缓冲名认槽位，多个声明必须一致、不一致就抛 |
| int8 入边的名字与取值 | `from_fx` 只看直接上游定名、`export` 穿过布局算子取值 | 两侧共用 `_quant_origin_of`，同一个判据 |

最后一条是 r8 修掉的配对错位：名字按「这是我自己的边」发（128 元素的自命名缓冲）、
数值按「这是上游 DQ 的整张张量」取（32 组 scale），配到一起等于宣称 group_size = 4，
而上游 DQ 实际是 128。两侧统一后，那 32 个节点与参考同构地引用
`output_buffer_179.bin`(4096B) + `output_buffer_phase_1_179.bin`(64B)。

## 三、修改的文件与函数

### 3.1 源码

| 文件 | 增删 | 关键改动 |
| --- | ---: | --- |
| `gml_bridge/calib_data.py` | 新增 796 | 5 个标定常数内嵌 + KV absmax 两个标量；`calibration_for_role` 按角色取、`activation_for` 按元素数平铺/截断、`kv_cache_scale` 算 KV 量化 scale；`numel <= 0` 与未登记角色都抛 |
| `gml_bridge/export.py` | +488 / -66 | 新增 `_output_scale_of` / `_input_scale_of` / `_kantor_scale_of` / `_kantor_shift_of` / `_kantor_mode_key` / `_kv_role_of` / `_kv_cache_written_by` / `_quant_producer_of` 与 `_FP16_DOMAIN_OP_TYPES` / `_LAYOUT_OP_TYPES`；DQ 与 Softmax 的相源接标定；`write_io_info` 加 `scale_of` / `shape_of`；自命名判据改按文件尾号；KV_Cache_DMA 的定标发 (2.0, post_shift 14)；**删** `_rope_elements` 与 `_input_scale_of` 的 int8 取值分支（换成不变量） |
| `gml_bridge/from_fx.py` | +318 / -51 | 新增 `_renumber_from_one`（id 压到 1..N，`A` 重新派生）、`_quant_origin_of`（穿过布局算子找 DQ）、`_peer_dtype_for`（按缓冲名认槽位）；`_stamp_dtypes` 位宽改按需递归解析（不再假设 `nodes` 是拓扑序）；四处边界节点打 `pim_calib_role`，另打 `pim_io_rank` / `pim_is_hidden_entry` / `pim_kv_is_key`；cache 出口缓冲补 sf 三字段与 dtype / extension；**删**写死的 Split dtype 覆盖 |
| `gml_bridge/runtime_files.py` | +88 / -29 | `write_data_buffer` / `calibration_buffer` 加 `calib_role`；`write_phase_output_buffer` 的 `content` 改必填、`element_count` 退化为校验；`write_dq_phases` 加 `per_group_fpsu`；`write_rope_buffer` 标量四族不看 `element_count`、乘积族按边宽 |
| `gml_bridge/phase_data.py` | +24 / -0 | 只加 `DynamicScalingPhases.__repr__`（报组数与四相首尾 3 项，不整表转储）；四相公式未动 |
| `contracts/gml_quant.py` | +4 / -3 | `GML_VERSION` `"26.2.1"` → `"19.2.0"` |
| `scripts/export_gml.py` | +19 / -8 | 拆出 `build_parser`（默认值要能被断言）；`--decode-block-only` 换成 `--no-decode-block-only`；接 `--verbose` 打印标定摘要 |
| `orchestrator/plan.py` | +1 / -2 | 删 `orchestrate` 的 `gml_version` 死参数（版本号单一真源） |
| `orchestrator/net_ini.py` | +2 / -2 | 删 `render` 的同名死参数，docstring 同步 |
| `genesim_bridge/paths.py` | +2 / -1 | 改掉「我方 `GML_VERSION` 不随参考改」这句已失效的注释 |

合计源码 **+946 / -162**。量级说明：`calib_data.py` 的 796 行里 680 行是标定常数
字面量；`export.py` 的增量主要是分派函数与它们的判据注释。删掉的都是被替代的旧路径
（`_rope_elements`、`activation_kantor_scale`、写死的 Split 覆盖、int8 取值分支、
三处无效的 None 防护），按「删优于加」没有注释留存。

### 3.2 测试（8 个文件，93 个新用例）

| 文件 | 新增用例 | 覆盖面 |
| --- | ---: | --- |
| `tests/test_gml_export.py` | 46 | 定标分派、Kantor 四族、IO_info 三项、int8 入边配对、入口缓冲内容、声明与上游位宽一致 |
| `tests/test_gml_from_fx.py` | 12 | id 连续 1..N、`A` 别名、hidden 角色取 rank、边界位宽不随边序漂 |
| `tests/test_runtime_files.py` | 12 | 零值分类逐族、RoPE 标量族宽度、FPSU 逐组判据、相长度校验 |
| `tests/test_calib_data.py` | 新增文件 17 | 常数形状/统计量/确定性、四角色各取自己那份、非正入参与未登记角色都抛 |
| `tests/test_phase_data.py` | 2 | 非零输入 → phase0/phase1 非零 |
| `tests/test_export_gml_cli.py` | 2 | `decode_block_only` 是默认值 |
| `tests/test_gml_coverage.py` | 1 | P1-3 的 4 个字段声明不被删 |
| `tests/test_orchestrator.py` | 1 | 版本号只有一个真源 |

**每条新用例都做过变异验证**：把被修的 bug 改回去，确认对应用例转红。两条如实
记录为「今天不红」的：边界位宽那条（本夹具上两种取法重合，它守的是将来的漂移）、
早期一条 int8 回落变异（小图上每条边都能找到生产者，那条路径走不到——已在 r8 改成
不变量，由新用例的变异验证覆盖）。

## 四、测试与验收结果

### 4.1 验收判据 20 条

真实导出：`--layers 1 --seq-len 16`，产物 `/tmp/gml_doc_a`。

| # | 判据 | 结果 |
| --- | --- | --- |
| A1 | phase 型算子的 phase0 逐组非零 | **通过**，37 个 DQ 节点共 0 个含零组。Softmax 也已接通（32 个节点；节点 17 实测 phase0 = −19.31、phase1 是真实 exp 数组——1024 个元素里 374 个非零，不再是全 1.0） |
| A2 | 所有 `output_sf` > 0 | **通过**。按各节点声明的 dtype 读；RMSNorm 那 4 个是 fp32 `0000803f`，与参考逐字节相同，按 fp16 读会误判 |
| A3 | 四相公式与甲方逐元素相同 | **通过 4/4**。用甲方自己的 `input_buffer_12` / `_193` 重算；内置 `HIDDEN_STATES` 与甲方 `input_buffer_25.bin` 逐字节相同 |
| A3b | 导出不读甲方任何 bin | **通过**。`sys.addaudithook` 记录全部 `open`，甲方 `.bin` 命中 0 个 |
| A4 | KV / Split / v_proj 的 `output_sf != 1.0` | **通过**，取到 0.2773 / 0.3042，int8 饱和 0.0244% |
| A4b | 7 个 Gemm 中恰 1 个非 1.0 | **通过**（v_proj） |
| A5 | IO_info 的 int8 sf == 对应 `output_sf` bin | **通过**，4 个 int8 项逐值相等 |
| A6 | IO_info 可 `eval` + numpy 读出 | **通过**，`sf` 是 numpy 标量 |
| A7 | hidden 入口 rank 3、其余六项 4 | **通过**，`[3,4,4,4,4,4,4]` |
| A8 | 悬空 0、孤儿 0 | **通过**，引用集 == 落盘集 3122 |
| A9 | 文件名模式类两边相等 | **通过**，72 类，缺 0 多 0 |
| A10 | 需求 2.4 的已对齐项逐项复测 | **通过**，node/edge = 200/331、`op_type` 分布 diff 为空。⚠️ 口径边界：这一条只证明**各类算子个数**一致，不证明**边接得一样**（见 §六 不足 3） |
| A11 | 同角色 bin 字节数相同 | **通过**。72 个共有族**全部一致**，两个差集都为空（经 r6 → r8 两轮整改，不符族 33 → 2 → **0**） |
| A12 | 全量回归全绿 | **通过**，`981 passed, 1 skipped, 42 deselected` |
| A13 | 出口 shape `[1,1,4096]`、无 lm_head | **通过**，无 32000 词表出口 |
| A14 | id == 1..N 连续无空洞 | **通过**，200 个节点、min 1、max 200 |
| A15 | `relay2gml_version "19.2.0"` | **通过** |
| A16 | Kantor scale / bias / Shift | **通过**。(a) `phase_3` 族 scale 逐元素等于 phase2（37/37）；(b) 非 phase 的 scale 非 0 且形式合法；(c) bias 全 0、Shift 全族 `{-8: 39, 0: 10}` 与对账表一致 |
| A17 | 结构校验器 5/5 | **通过** |
| A18 | int8 入边的「缓冲元素数 / scale 组数」== 上游 group_size | **通过**（r8 新增）。我方与参考逐项相同：Gemm 128、Split 128、matmul1 128、matmul2 1024 |

### 4.2 额外核对

| 项 | 结果 |
| --- | --- |
| 可复现（非功能需求 1） | 两次独立导出 `diff -rq` **逐字节相同** |
| 整网路径未被打破 | `--no-decode-block-only` → 3173 个 bin，三项自检全通过 |
| 四条入口缓冲 | cos / sin / mask / kv_position 与参考**逐字节相同**（hidden 为对照组，同样相同） |
| 逆拓扑方向未被改反 | 331 条边里 327 条 `source > target`，与参考同向（参考 190/331 亦为逆拓扑） |

### 4.3 八轮评审的处置分布

| 轮次 | 项数 | 最高级别 | 处置落点 |
| --- | ---: | --- | --- |
| r1 | 9 | 高 | 标定接通（DQ 四相 / Softmax / 9 个零 scale）+ 定标白名单 + 失效注释 |
| r2 | 4 | 中 | KV 三个 sf 字段同值 + Kantor 按 `kantor_mode` 分族 + hidden 角色标记 |
| r3 | 4 | **严重**（`A` 指向错误节点） | `A` 由 `input0_node_id` 重新派生 + 白名单收敛到实测面 |
| r4 | 5 | 中 | 标定中间产物可读打印 + A3 口径改写 + `output_sf` 单写者 |
| r5 | 5 | 中 | 出口槽序按角色 + 出口缓冲补 dtype/extension + 版本号死参数清理 |
| r6 | 7 | 高（A11 未达标） | A11 重统计（两个根因：RoPE 标量族宽度、FPSU 逐组判据）+ `input_sf` 显式分派 |
| r7 | 5 | 高（入口缓冲装错常数） | 入口缓冲按角色取常数 + Split 位宽 + KV DMA 定标发 2.0 |
| r8 | 2 | 中 | int8 入边的名字与取值统一 + 边界位宽按槽位取 |
| **合计** | **41** | — | — |

其中 **6 项的根因比评审定位的更深**，按实测修正了方向而不是照评审的判断改：

| 轮次 | 评审定位 | 实测根因 |
| --- | --- | --- |
| r1 | Cos/Sin 的 scale 写死在 kantor 分支的 else | `elif "kantor"` 排在 `_is_rope_file` **之前**，先把这几个键截住了 |
| r2 | `input_sf` 按 dtype 分派 | 不能按 dtype——参考里其余 int8 输入的 `input_sf` 是**引用上游 DQ 的 phase1**，按 dtype 分派会误改 32 个 MatMul。判据应落在 KV 角色上 |
| r6 | `input_sf` 非 KV 时「潜在」回落 1.0 | 不是潜在，**已经实际发生**：32 个 MatMul + 1 个 Split 的 int8 入边落盘就是 1.0 |
| r7 | Split 的 dtype「照抄旧样本」 | 写死覆盖只是表象，根因是 `_stamp_dtypes` 假设 `nodes` 是拓扑序而实测不是，位宽静默退回 fp16；新断言由此又抓出第 4 条边 |
| r7 | cos / sin 的文件配对 | 评审把两槽对交叉了；参考约定是 slot 1 = sin、slot 2 = cos，同槽比才是对的口径（不影响结论，改前两槽都错） |
| r8 | `_input_scale_of` 照抄上游整条向量 | 那是后果；真正的分叉是 `from_fx` **定名**只看直接上游、`export` **取值**穿过布局算子，两侧口径不一致 |

## 五、如何验证

先 source 环境，每个新 shell 都要做一次：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
```

### 步骤 1：全量回归

```bash
python -m pytest tests/ -q -k "not llama2_7b"
```

预期 `981 passed, 1 skipped, 42 deselected`。`llama2_7b` 那组要加载 7B 权重、
单次验证不必跑。

### 步骤 2：真实导出（decode block，默认路径）

```bash
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_check
```

预期尾部输出 200 节点 / 331 边 / **3122** 个 bin，以及三项自检全通过：
结构 5/5、dtype 覆盖 15 类 0 缺、引用集 == 落盘集。
加 `--verbose` 可逐个 DQ 节点打印标定四相摘要（默认关闭，37 个节点会刷屏）。

### 步骤 3：图结构逐项核对（A10 / A14 / A15）

```bash
python - <<'EOF'
import re
t = open('/tmp/gml_check/relay2gml_graph.gml').read()
ids = [int(m) for m in re.findall(r'^\s+id (\d+)$', t, re.M)]
edges = re.findall(r'edge \[\s*source (\d+)\s*target (\d+)', t)
print("node/edge =", len(ids), "/", len(edges))
print("id 连续 1..N:", sorted(ids) == list(range(1, len(ids)+1)))
print("version:", re.search(r'relay2gml_version "([^"]+)"', t).group(1))
blocks = t.split('node [')
ne = sum(1 for b in blocks
         if (a := re.search(r'\n\s+A (\d+)', b)) and (i := re.search(r'input0_node_id (\d+)', b))
         and a.group(1) != i.group(1))
print("A != input0_node_id 的节点数:", ne)
EOF
```

预期 `200 / 331`、`True`、`19.2.0`、`0`。

### 步骤 4：文件族字节数对账（A11）

```bash
python - <<'EOF'
import os, re, glob, collections, sys
sys.path.insert(0, '.')
from genesim_bridge import paths
ref = str(paths.gml_llama2_reference_dir())
fam = lambda n: re.sub(r'\d+', 'N', n)
def collect(d):
    m = collections.defaultdict(set)
    for p in glob.glob(os.path.join(d, '*.bin')):
        m[fam(os.path.basename(p))].add(os.path.getsize(p))
    return m
a, b = collect('/tmp/gml_check'), collect(ref)
common = set(a) & set(b)
print("共有族", len(common), " 不符族", len([k for k in common if a[k] != b[k]]))
print("仅我方", len(set(a) - set(b)), " 仅参考", len(set(b) - set(a)))
EOF
```

预期 `共有族 72  不符族 0`、两个差集都为 0。

### 步骤 5：可复现与整网路径

```bash
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_check2
diff -rq /tmp/gml_check /tmp/gml_check2          # 预期无输出
python scripts/export_gml.py --layers 1 --seq-len 16 --no-decode-block-only \
    --out-dir /tmp/gml_full                       # 预期 3173 个 bin、自检全通过
```

### 验证顺序与关系

```
步骤 1 回归 ──► 步骤 2 导出 ──┬──► 步骤 3 图结构（A10/A14/A15）
   代码级判据      产物级判据   ├──► 步骤 4 族对账（A9/A11）
                               └──► 步骤 5 可复现 + 整网路径
```

步骤 1 守代码级不变量（分派逻辑、抛异常的边界、配对关系）；步骤 2~5 守产物级判据。
两层都要过：单测绿而产物错过去出现过（A5 那条测试原先只比常量集合、从不打开
`output_sf` bin），产物对而单测缺也出现过（`A` 指向错误节点时 5 条结构规则全绿）。

## 六、当前仍存在的不足

以下 5 项都已核实、已入册需求 2.4 或已拍板接受，不是待修缺陷：

| # | 差异 | 状态 |
| --- | --- | --- |
| 1 | 非标定类 `output_sf` 的**文件名口径**与参考不同：我方按本节点自命名，参考按消费者编号，连带 `output_sf_*` 文件数 182 vs 146 | r5 拍板维持现状。两边都自洽（A8 悬空 0 孤儿 0）。入边侧已在 r8 改成与参考同的「引用生产者」，只剩出边侧 |
| 2 | `IO_info` 的 **inputs 槽序**按节点号升序；参考是 relay 子图签名顺序（`nprm_0_i3` 排槽 3、`nprm_0_i15` 排槽 5），推不出规则 | r5 拍板维持现状。outputs 侧已按角色对齐（槽 0 = hidden、1 = key、2 = value） |
| 3 | **图拓扑**的 `Transpose` 位置不同：Q 路径我方多一个、K 缓存读路径少一个 | r7 核实为**等价改写**（边宽证据：我方 `DMA→Split` 与参考 `Transpose→Split` 同为 `1x32x128x1024`），已入册 2.4。多一个与少一个恰好相抵，所以 `op_type` 计数仍相等——这也是 A10 口径边界的来由 |
| 4 | **KV scale 数值**不复现甲方：我方 0.2773 / 0.3042（`absmax / 127`），甲方 0.0458939 / 0.00261151 | 需求 P0-2 已接受。甲方那两个是真实模型标定定值，无法从标定数据反推；我方取值自洽、非 1.0、不除零、不饱和，符合判据 4 |
| 5 | **同名文件 235 个逐字节不同**（559 个同名里 324 个相同） | 判据 4 已豁免数值差异（权重与标定数据本就不同）。但这个比例没有逐族归因过，不排除其中混有形式问题——是目前最值得下一轮查的一条 |

另有一项留待向甲方确认：`kantor_B_scale_buffer_file`（EltwiseMul）甲方是 5.96e-08
（= 2⁻²⁴，fp16 最小非规格化数），我方按「不做定点化发 1.0」的口径是 1.0。该族的
`kantor_mode` 是 `elementwise_mul_fp16`，按判据应为 1.0；2⁻²⁴ 的来由不在标定数据里，
判不出。
