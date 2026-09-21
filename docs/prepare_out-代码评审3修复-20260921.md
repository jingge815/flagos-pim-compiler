# prepare_out 评审 3 修复（2026-09-21）

评审对象：`docs/prepare_out-代码评审3-20260921.md`。上一轮
（`docs/prepare_out-P0修复2-20260921.md`）的 Mask 双输入、Softmax 5 入 + 5 出、
L2 双槽不重叠、bmm2→Concat、对拍白名单三分类**全部保留，没有回退**。

本轮只处理评审 3 核实为真的 P0，以及修 P0 时顺手能做的 P1 小项。P1-1
（值从 pimmlir 映射）和残差旁路边仍未做。

## 核实结论：评审 3 哪些是真的

| 级别 | 评审的说法 | 核实结论 |
| --- | --- | --- |
| P0 | bmm 的 `weight_buffer`/`weight_sf`/`weight_zp` 整族没进 GML | **属实**。`from_fx` 只在 `_weight_param_of`（`get_attr` 二维权重）命中时写三族；MatMul 的第二个 operand 是 KV cache，抓不到。参考 64 个 MatMul 全有 `weight_buffer_dtype "int8"` + 131072B 平面 |
| P0 | `--decode-block-only` 只裁编排器，GML 仍带 final RMSNorm + lm_head | **属实**。`_drop_tail` 还靠 FX 名 `mul_12`/`linear_7` |
| P0 | `DDR Input/Output Orig Buffer Name` 是占位值 | **属实**。输入恒 `buffer{slot}`，输出恒 `buffer{自己}`；参考按生产者/消费者编号，32 个 bmm2 共享 Concat 下游 |
| P0 | `Original cache file` 悬空 + RoPE 定标段号错 | **属实**。`_cache_dma_buffer` 自己拼 `input_buffer_0_<DMA>`，GML 侧 DMA 只有单数 `input_buffer`；mul 一律 `idx=(6,6)`，Q cos 应为 5、sin 应为 4 |
| P1 | 对拍 `_normalise_naming_value` 把所有数字抹成 `#` | **属实**。段号 5 vs 6 被吃掉 |
| P1 | 残差旁路填伪造域 | **属实**。缺边时把槽 0 复制到槽 1 / 填字面量 0。本轮只停伪造，不补边 |
| P1 | `_dual_slot1_size` 对 rope_add 也返回 `HD*2=256` | **属实**。txt 声明 8192 |
| P2 | `_order_like_reference` 吞异常、`.gitignore` 缺会话记录 | **属实**，已顺手清 |

**不是本轮范围（评审自己也标了数天）**：P1-1 值来源迁到 pimmlir、残差旁路边、
GML dtype/extension 全覆盖、相位 bin 尺寸按 S=1024、`net.ini` 末行 LF。

## 本次修了什么

### 1. 对拍归一保留槽位/相位/段号（P1，先做）

`scripts/diff_prepare_out.py::_normalise_naming_value`：个位数（≤8）原样保留，
两位数以上的节点号/qidx 才通配成 `#`。否则后面每修一处段号都会被 `#` 掩盖。

### 2. bmm 权重三族进 GML + 落盘（P0-1）

`gml_bridge/from_fx.py`：`MatMul_input_as_weight=1` 时声明
`weight_buffer`/`weight_sf`/`weight_zp`/`weight_buffer_dtype int8`。
这是 KV cache 走权重通路，不是 `get_attr`。

`gml_bridge/export.py`：没有 f32 张量时按边元素数写 int8 占位平面 + 单个
fp16 scale（值 1.0，校验器要求非零）。**必须与**「上游已按权重通路写过
`weight_buffer`」**分开处理**——`weight_buffer` 已在盘上时仍要补 `weight_sf`，
否则 64 个 scale 悬空。

### 3. KV_Cache_DMA 三槽 + Original cache file 读字段（P0-4a）

参考 node 28：`input_buffer_0/1/2`，槽 1 是 int16 位置下标、`L2A_ignore`。
`from_fx` 按这个声明；生产者给 DMA 的输出命名顺移到槽 2（新值）。
`layer_fields._cache_dma_buffer` 改读 GML 已声明的 `input_buffer_0`，
K/V 优先看 `pim_kv_is_key`，不再按 node_id 大小猜。

### 4. RoPE 定标段号拆 K/Q（P0-4b）

段号来自 `contracts.gml_hw_table.ROPE_UNITS`：sin 乘 → 4，Q cos → 5，
K cos → 6，加法 → 1/2。不再写死 `(6,6)`。

### 5. decode-block 裁剪前移到 GML（P0-2）

`convert(..., decode_block_only=True)` 按结构丢掉：
- 权重不是七个投影之一的 Gemm（lm_head）
- 喂它的 DynamicScaling
- 没有块内计算消费者的 RMSNorm（final norm）
- 只连向这些节点的边界缓冲 / Reshape

`scripts/export_gml.py --decode-block-only` 在序列化时就裁，编排器的
`_drop_tail` 改成同一套结构判据，不再靠 `mul_12`/`linear_7` 字符串。
没有权重参数的测试夹具 Gemm 不裁。

### 6. DDR Orig 名从生产者/消费者推导（P0-3）

- 输入：该槽 `residual_input_buffer` / `inputN_node_id` 的 `buffer<生产者>`
- 输出：消费者 `buffer<id>`；bmm2→Concat 用 Concat 下游的共享名；
  cache 写出仍是 `value_cache_out` / `key_cache_out`

### 7. 其它

- `_dual_slot1_size` 只对 mul（phase 0/1）返回 `HD*2`，rope_add 照抄槎 0
- 缺残差第二邻居时不再填伪造值
- `_order_like_reference` 不再 `except Exception` 静默降级
- `.gitignore` 加 `*-local-command-*.txt`

## 增删文件

| 文件 | 改动 |
| --- | --- |
| `gml_bridge/from_fx.py` | MatMul 权重三族；KV DMA 三槽；decode-block 裁剪；DMA 消费者槽 2 |
| `gml_bridge/export.py` | MatMul 权重落盘；`decode_block_only` 传入 convert；int16 缓冲 |
| `orchestrator/layer_fields.py` | cache 读字段；RoPE 段号；DDR Orig 名；停伪造残差域 |
| `orchestrator/l2_alloc.py` | rope_add 槎 1 按数据宽分配 |
| `orchestrator/plan.py` | `_drop_tail` 结构化；去掉静默降级 |
| `scripts/export_gml.py` | 裁剪前移；双槽 size1 闸门 |
| `scripts/diff_prepare_out.py` | 槽位/相位敏感归一 |
| `tests/test_layer_fields.py` | 段号、cache、DDR Orig、L2 尺寸、对拍归一 |
| `tests/test_gml_from_fx.py` | MatMul 三族断言 |
| `tests/test_gml_depends_on_opcompiler.py` | DMA 三槽断言 |
| `.gitignore` | 会话记录 |

有 git 历史的文件净增约 **+1223 / -135**（含本轮之前未提交的 GML/编排器改动）。
`layer_fields.py` 仍是未跟踪文件，本轮在其上改了 cache / 段号 / Orig 名 /
停伪造，现 1556 行（仓规约软上限 ~400，P2 拆文件仍未做）。

## 验证

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
python -m pytest tests/ -q -k "not llama2_7b"
# 692 passed, 42 deselected（123s）

rm -rf /tmp/review3_fix
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/review3_fix \
    --use-opcompiler --orchestrate --decode-block-only
# 17/18 通过。未过：txt 引用 bin 缺失 12（从 82 降到 12）

python scripts/diff_prepare_out.py --mine /tmp/review3_fix/prepare_out \
    --ref /media/disk/fengjingge/src/xinfangzhou-resource/llama2_w4a8_decode_block_0/prepare_out
# MATCH 45855 / VALUE_DIFF 266 / ALLOC 5683 / MISSING 0 / EXTRA 0 / 层类 422=422
```

**P0 已对上的实测：**

| 项 | 参考 | 本轮 |
| --- | --- | --- |
| GML `weight_buffer`/`weight_sf`/`weight_zp` | 73 | **73**（7 Gemm + 64 MatMul + 2 RMSNorm） |
| 落盘 `weight_buffer_*.bin` | 73 | **73** |
| RMSNorm / Gemm / DynamicScaling | 2 / 7 / 36 | **2 / 7 / 36**（尾部已裁） |
| 编排层数 | 422 | **422**，23 类计数全等 |
| DMA `input_buffer_0/1/2` | 有 | **有**（槽 1 int16 + L2A_ignore） |
| `Original cache file` | `input_buffer_0_<DMA>` | **读声明字段，文件在盘上** |
| Q/K mul 段号 | 5 / 6 / 4 | **5 / 6 / 4，对应 bin 存在** |
| bmm2 `DDR Output Orig` | 32 头共享同一 buffer | **共享 `buffer18`**（参考是 `buffer4`，编号体系不同） |
| bmm1 `DDR Input Orig 0` | 生产者 | **`buffer183`（上游）不是 `buffer0`** |
| txt 引用缺失 | 0 | **12**（82→12；剩 LUT / Kantor / 残差旁路） |

**VALUE_DIFF 从 104 涨到 266**：对拍归一不再抹个位数，评审 3 §3.2 要的就是这个。
新增暴露的主要是 `DDR Weight Orig Buffer Name`（参考带 `map<头下标>`，我方还写死
`buffer23_map0`）和语义 Orig 名（`nprm_*` / `*_dequant_buff`）。不是回归，是口径变严。

**仍缺的 12 个 txt 引用**（不是本轮 P0）：`activation_lut_file_11.bin`（gate LUT）、
残差旁路 `input_buffer_{0,1}_14.bin`、RoPE 表语义名 `input_buffer_184.bin`、
v_proj/down 的 Kantor 六件。对应评审 3 的 P1-3 / P1-5。

Reshape 仍是 4（参考 2）、边界缓冲 6（参考 10）——图结构不同，不是漏裁 lm_head。
`llama2_7b` 标记的用例本次未重跑。
