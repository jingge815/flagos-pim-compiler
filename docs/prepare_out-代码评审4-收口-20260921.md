# prepare_out 评审 4 收口：P0-2.5 dtype + P1-3.1 全量（2026-09-21）

承接 `docs/prepare_out-代码评审4-P1修复-20260921.md`。上一轮结束时还剩两块没做完，
本轮按评审做完，**FlagTree 真的改了、也按 `0-install-flagtree.sh` 重编了**。

不回退已对上的修复（200/331/10、Reshape 2、422 层、悬空 0、尺寸同源）。

---

## 1. P0-2.5：GML dtype 覆盖

上一轮只给 MatMul/Softmax/Mask 盖了一部分，参考里 15 类算子有 dtype、我方 7 类缺。

### 做法

`gml_bridge/from_fx.py` 新增 `_stamp_dtypes(nodes)`，在全部节点发射完之后统一盖章。

**关键发现：布局算子的 dtype 是沿边传播的，不是按 op_type 固定。** 第一版我按
`op_type -> (in, out)` 写了张静态表，实测对不上：参考里同一个 `Transpose` 既有
fp16 的（node 14，concat 之后那条）也有 int8 的（node 27/29/34，KV 与 QK 那条），
`Reshape` 同样两种。按类型写死会让一半节点位宽错一倍，而这在我们这侧不报错。
改成按 `residual_input_buffer` 取上游已定的 `output_buffer_dtype` 传播。

固定的那几类仍按实测写死：

| 规则 | 算子 |
| --- | --- |
| 输出恒 int8（落定点） | DynamicScaling、Llama2Activation(DQ)、KV_Cache_DMA、Split |
| 输出恒 fp16（浮点 FPSU 累加） | MatMul、Softmax、Mask、EltwiseAdd、EltwiseMul、RMSNorm |
| 输入恒 int8（吃定点激活） | Gemm、MatMul |
| Gemm 输出 | 按 `kantor_mode == fp2int_converter` 判（v_proj 落 cache 走 int8） |
| `weight_buffer_dtype` | Gemm 的 get_attr 二维权重 **int4**；MatMul 的 KV cache 权重通路仍 int8 |

一处例外照抄参考、没自己推：吃 `Llama2ActivationDQ` 的那个 Split，参考声明
**fp16**（node 21），而生产者自己的 `output_buffer_dtype` 是 int8 —— 那个节点是
「RoPE 3 连 + DQ 4 相」的融合体，RoPE 侧 fp16、DQ 侧 int8，两种声明各自都讲得通。

同时删掉不该有的：DynamicScaling 与布局算子的顶层 `input_sf`/`input_zp`
（参考 0 个，我方曾多写 36 个）。KV_DMA 反过来要补顶层那一份（参考 node 28 有）。

### 闸门

`export_gml.py` 加 `_check_dtype_coverage`：按 op_type 比「字段在不在」，
参考有、我方无的列表必须空。不比编号（两边 node_id 体系不同）。

### 实测

```
GML dtype 覆盖（参考有则我方有）: 15 类算子，0 类缺 dtype
dtype 取值逐 op 比对：值不一致 0 处
落盘 3335 → 3266（少写的正是删掉的多余 sf/zp）
```

---

## 2. P1-3.1a：编排器吃 `artifact.slots`

`CompileSlots` 之前只到 GML 侧，编排器仍读模块级 `DEFAULT_SLOTS`。现在
`orchestrate()` 取 `artifact.slots`，下传 `classify` / `build_layer_fields` /
`buffers_from_layers`。

替换的字面量（`layer_fields.py`）：`Num Output Heads`、`Total Split Head/Weight Num`、
`Eltwise broadcast factor` → `slots.heads`；`broadcast Input Stride X`、
`Group data size` → `head_dim`；`DDR Weight Width/Height/stride`、`ddr_h`、
bmm2 的 `Group data size` → `hidden`/`seq`；`classify` 的 `11008` → `intermediate`。

`build_layer_fields` 不传 `slots` 时回退 llama2-7B，单测的 7B 数字断言不用改。

---

## 3. P1-3.1b：FlagTree 改 + 重编

### 改了什么

`ExpandPhases.cpp`：

1. **`stampMatmul`**：矩阵乘是单相、不展开，但层卡仍需要 FPSU / FLP / 激活模式。
   以前这些只能由本仓查表，现在 pass 盖在 `pim.matmul` 上。这样消费者读矩阵乘
   与读展开相位走同一条路，不必单独为矩阵乘族留一张硬编码表。
2. **RoPE add 区分 Q/K**：K 路写 cache 前要重定标（层卡 3），Q 路留 fp16 给紧随的
   DQ（层卡 0）。pass 自己分不出在展开哪条链，所以由 emitter 在 `pim.rope` 上标
   `pim.kantor-mode`，pass 有则照抄、无则 0。
3. **`transpose-type` 按组数算**：第一版写了常量 2，实测压掉了「线性 DQ 走 1、
   分数 DQ 走 2」的正确逻辑（对拍多出 5 处 `dq_p3` 差异）。改成
   `elementCount(groupsTy) <= 1 ? 2 : 1` —— pass 自己就知道组数。

`kantor-mode` 用**层卡取值**而不是 dialect 枚举序：多个不同的卡值共享同一个
`KantorMode`，消费者从 `#pim.datapath` 反推不出来。

### 本仓侧

- `phase_plan.PhasePlan.op_attrs`：收单相算子的域。**不能进 `phases`** ——
  进了 `count` 会把单相算子算成一相，`cross_check` 立刻判不符。
- `plan._attach_op_attrs`：把 `op_attrs` 填进 `Layer`（Gemm 的 `phase is None`，
  走不到原来那条按相位号取的路径）。
- `layer_fields`：IR 有则用、无则回退 TABLE（上一轮已接）。

### 编译

`build_flagtree()` 会 `rm -rf` 整个 build 目录（7.1G，全量重编约 11 分钟），
所以先备份了可用的 `triton-opt`。脏源码要显式放行：

```bash
ALLOW_DIRTY_FLAGTREE_SOURCE=1 MAX_JOBS=8 bash 0-install-flagtree.sh \
    --prefix .../flagOS-installed/flagTree \
    --source-dir .../flagOS-installers/FlagTree --skip-test
```

`--skip-test` 跳过 matmul 示例（要 GPU），不影响 `triton-opt`。

---

## 4. 顺手修的 Orig 语义名

对拍在 dtype / slots 修完后暴露出几处**可修**的命名（都能从 GML label 推出来，
不需要 TVM）：

| 层 | 域 | 参考 | 原来 |
| --- | --- | --- | --- |
| Q 路 `dq_p1` | `DDR Input Orig Buffer Name 0` | `..._params_22_mid_buf` | `buffer192` |
| Q 路 `dq_p2` | `DDR Output Orig Buffer Name` | `..._params_22_dynamic_quantization_dequant_buff` | `dynamic_quantization_params_184_...` |
| `rope_add` | `DDR Input Orig Buffer Name 0/1` | `..._mul_cos_buffer` / `_mul_sin_buffer` | `buffer190` / `buffer198` |

这几个中间缓冲不对应任何 GML 节点编号，所以只能按子块语义命名。新增
`_ddr_input(semantic=)` 这条通路：**块内**语义名只写 `Orig`，不写 TVM 名
（参考那几层确实只有一项，多写就是多域）。

---

## 验证

```
pytest -k "not llama2_7b"                708 passed
export_gml --use-opcompiler --orchestrate --decode-block-only
                                         23/23 通过
GML dtype 覆盖                            15 类，0 类缺；取值逐 op 比对 0 处不一致
接算子编译器前后 GML 逐字节相同            节点 200
算子编译器真的决定 GML（反证）              DQ 4→2 相，GML 少 58815 字节
txt 引用缺失                              0
L2 分配 ≥ 声明                            513 处，0 欠分配
对拍                                      MATCH 45886 / VALUE_DIFF 48 / MISSING 0 / EXTRA 0
```

IR 域真的到了 txt，且**不是查表巧合**：

| txt 域 | 值 | 凭什么说是 IR 来的 |
| --- | --- | --- |
| K 路 `rope_add` Kantor mode | 3 | Q 路同一层是 0，查表给不出两个值 |
| Q 路 `rope_add` Kantor mode | 0 | 同上 |
| `dq_p3` Transpose type | 32 个 2、5 个 1 | pass 按组数算，改 `groupSize` 就变 |
| `gemm_gate` Flp / Fpsu mode | 10/17/3、2 | `test_single_phase_op_attrs_really_drive_txt` 把 IR 改成 (1,2,0)/7，txt 跟着变 |

最后一条是**反证**：`gemm_gate` 的查表值恰好也是 10/17/3，所以「导出后是
10/17/3」本身证明不了接线为真 —— 单测把 IR 值改成不可能来自查表的数，
断言 txt 跟着变。

---

## 仍未对齐（如实记录）

对拍剩 48 处，全部集中在命名：

- **35 处 mask 的 `DDR Input Orig Buffer Name 1`**：参考是 TVM Relay 内部符
  `nprm_182_i168`，我方不走 TVM，编不出这个名字（写 `mask`）。
- **3 处 `nprm_182_i12`**：同上，图入口 hidden（我方写 `hidden_states`）。
- **10 处 residual 的 `Datain` / `Dataout` / `Input buffer file`**：两边缓冲
  编号体系不同（我方逆拓扑，参考来自 Relay）。

这些要对上得先拿到对方的 Relay 符号表，不是本仓能推的。

Gemm/bmm/mask 的 L2 fpsu 切片仍查表：那是布局而非相位语义，`ExpandPhases`
没有对应的整算子模板，硬造会是假映射。
