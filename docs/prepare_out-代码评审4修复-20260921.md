# prepare_out 评审 4 修复（2026-09-21）

评审对象：`docs/prepare_out-代码评审4-20260921.md`。上一轮声称的 P0
方向成立但没闭环：声明补上了，落盘尺寸不对；闸门加了，恰好不检查出
问题的那几处。本轮按依赖修尺寸同源、L2 分配、KV 边界节点、悬空引用。
**已核实为真的修复（Mask 第二路、RoPE 表、DMA 三槽声明、段号、Softmax
5 入 5 出、bmm 权重三族声明、decode-block 裁剪）全部保留，没有回退。**

## 核实结论：评审 4 哪些是真的

| 级别 | 评审的说法 | 核实结论 |
| --- | --- | --- |
| P0-2.1 | 运行期 bin 按导出图 seq_len=16 写，MatMul 权重 65536 vs 131072，KV cache 64KB vs 4MB | **属实**。`_matmul_weight_elements` 优先取边；DQ/Softmax `spec.numel` 来自导出形状 |
| P0-2.2 | L2 分配与 txt 声明不同源；Q mul_cos `#1` 256 vs 8192；Mask 64 vs 2048 | **属实**。`buffers_from_layers` 用 `phase_bytes`/边宽；`_dual_slot1_size` 一律按表宽 |
| P0-2.3 | KV cache 初始输入没进 GML：节点 198 vs 200、边界 6 vs 10 | **属实**。图是 `use_cache=False` 导出的，cache/位置不在 FX 里 |
| P0-2.4 | 12 个悬空里 3 个是代码 bug：add 合成 `input_buffer_0_14`、Q-DQ 合成裸 `input_buffer_184` | **属实**。`layer_fields` 缺字段时自己拼名字 |
| P0-2.5 | GML 注解缺/多、文件族无闸门 | **部分属实**。本轮补了 gate LUT、v_proj/mlp_mul Kantor、部分 dtype。文件族逐族相等的闸门未加（P1 范围） |
| P1-3.1 | 域值不是从 pimmlir 映射 | **属实**，本轮不做（需 FlagTree 改 attr，数天） |
| P1-3.2 | `DDR Weight Orig` 写死 `buffer23_map0` | **属实**，本轮未改（对拍 VALUE_DIFF 仍 62 处） |
| P2 | `net.ini` 末行多一个 LF | **属实**。参考 EOF 无换行 |

## 本次修了什么

### 1. 编译期槽位（P0-2.1）

新增 `contracts/compile_slots.py`：`H/I/hd/S/nh` 从模型 `config.json`
来，默认 llama2-7B 的 4096/11008/128/1024/32。

硬规则：**prepare_out 相关尺寸以编译期槽位为准，导出图只提供拓扑。**
小图测试（hidden≠4096）不改写，避免夹具被 7B 几何污染。

落盘：

- MatMul 权重：`S×hd = 131072`（含生产者先按权重通路写的那 2 个）
- KV 三槽：`nh×S×hd` / `nh×3` / `nh×hd` → 4MB / 192B / 4096B
- Softmax 相位：`S=1024` → 2048B
- DQ 相位：hidden 8192B、MLP 22016B、分数 2048B

CLI `export_gml.py` 用 `CompileSlots.from_config(model.config)` 传入。

### 2. L2 分配与声明同源（P0-2.2）

`l2_alloc.buffers_from_layers` 按编译期宽度算，不再用导出图边宽。
Mask 分配 2048/2080，不再是 64。

`_dual_slot1_size` 增加 `bcast`：Q 路 mul_cos（`bcast=0`）槎 1 是数据
8192B，K 路仍是表 256B。原来的测试把 Q 路 256 钉死，已改。

闸门：MatMul 权重尺寸、KV cache 平面尺寸。双槽不重叠 / size1>0 保留。

### 3. KV cache 边界节点（P0-2.3）

与 RoPE 表/Mask 同构：K cache、V cache、位置下标各一个 `is_buffer`。
DMA 的 `residual_input_buffer` 写成三槽真实入边
`[cache, position, 新值]`，`input_count=3`。

本轮实测：节点 201、边 331（参考 200/331），`is_buffer` 9（参考 10，
缺的是 KV 写回那两个出口之一，入口侧 7 个输入已齐）。

### 4. 闭合悬空引用（P0-2.4）

- 残差 `Datain`：读 GML 已声明的 `input_buffer` / `input_buffer_0`，
  禁止合成 `input_buffer_0_14.bin`。真正接上 `entry_bypass`（第一条
  add 只有一路可映射上游时，槽 0 接入口缓冲）。
- Q 路 DQ p1/p4：裸 `input_buffer` 被 pop 成三槽后，读
  `input_buffer_phase_0`。
- gate Gemm：`from_fx` 声明 `activation_lut_file`，写盘走 `synth_silu()`。
- v_proj Kantor A 三件、mlp_mul Kantor A/B 五件：GML 声明 + 落盘
  （scale 2B、bias 4B、Shift 1B）。

实测 txt 引用 2008 个，**缺失 0**（评审 4 是 12）。

### 5. net.ini 末行（P2，顺手）

参考 EOF 无换行。`render_layers_section` 最后一行不再加 LF。
测试与注释一起改。

## 增删文件

| 文件 | 改动 |
| --- | --- |
| `contracts/compile_slots.py` | **新增**。编译期槽位真源 |
| `contracts/gml_coverage.py` | Kantor / LUT / 分槽 FPSU 从 PENDING 挪到 EMITTED |
| `gml_bridge/from_fx.py` | KV 边界节点；残差旁路；Gemm/EltwiseMul Kantor+LUT；边 dims 按槽位 |
| `gml_bridge/export.py` | 落盘按槽位；SiLU LUT；Kantor 写盘 |
| `orchestrator/l2_alloc.py` | 分配尺寸与声明同源；`_dual_slot1_size(bcast=)` |
| `orchestrator/layer_fields.py` | Datain 读声明；常量从 CompileSlots 取 |
| `orchestrator/net_ini.py` | 末行无 LF |
| `scripts/export_gml.py` | 传入 slots；权重/KV 尺寸闸门 |
| `tests/test_layer_fields.py` | 残差 Datain、Q-DQ phase0、槽位尺寸、bcast |
| `tests/test_orchestrator.py` | net.ini 末行 |
| `tests/test_gml_depends_on_opcompiler.py` | DMA 三槽入边、KV 边界节点 |
| `tests/test_gml_export.py` | Kantor 命名白名单 |

`layer_fields.py` 现 1572 行（仓规约软上限 ~400，拆文件仍是 P2）。

## 验证

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/ -q -k "not llama2_7b"
# 696 passed, 42 deselected（134s）

rm -rf /tmp/review4_fix
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/review4_fix \
    --use-opcompiler --orchestrate --decode-block-only
# 20/20 通过。txt 引用缺失 0。422 层。

python scripts/diff_prepare_out.py --mine /tmp/review4_fix/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out
# MATCH 45855 / VALUE_DIFF 263 / ALLOC 5686 / MISSING 0 / EXTRA 0
```

**P0 已对上的实测：**

| 项 | 参考 | 本轮 |
| --- | --- | --- |
| MatMul `weight_buffer` 尺寸 | 131072 × 64 | **131072 × 64** |
| KV cache 平面 | 4194304 × 2 | **4194304 × 2** |
| 位置下标 int16 | 192 × 2 | **192 × 2** |
| 新值 | 4096 × 2 | **4096 × 2** |
| Softmax `input_buffer_phase_0` | 2048 × 32 | **2048**（相位族 2048×64 含 DQ 分数） |
| Mask L2 size0/1 / 输出 | 2048 / 2048 / 2080 | **2048 / 2048 / 2080** |
| Q mul_cos size0/size1 | 256 / 8192 | **256 / 8192** |
| txt 引用缺失 | 0 | **0**（12→0） |
| `activation_lut_file` | 1 | **1**（288B） |
| Kantor A/B（v_proj + mlp_mul） | 8 | **8** |
| 层类 23 类计数 | 422 | **422 全等** |
| net.ini 末行 | 无 LF | **无 LF** |
| 导出闸门 | — | **20/20** |

**VALUE_DIFF 263**（评审 4 是 266）：主要仍是 P1 的
`DDR Weight Orig Buffer Name`（`map0` vs `map<头号>`，62 处）和语义
Orig 名。不是回归。

## 仍缺（不在本轮 P0）

- 边界 `is_buffer` 9 vs 10（缺 KV 写回出口之一；Reshape 4 vs 2 仍在）
- `DDR Weight Orig` 的 `map<头号>`（P1-3.2）
- 域值从 pimmlir 映射（P1-3.1，需 FlagTree）
- `parser_output` 分族计数闸门（P0-2.5 后半，本轮只补了会让仿真器读不到的文件）
- 对拍器 schema 归一 + 单测（P1-3.3）
- `layer_fields.py` 拆文件（P2）
