# prepare_out 评审 4：剩余 P0 + 全部 P1（2026-09-21）

评审对象：`docs/prepare_out-代码评审4-20260921.md`。上一轮 P0 尺寸/悬空引用
已闭环，本轮把**没闭环的 P0**和**全部核实为真的 P1**一次修完。不回退
Mask 第二路、RoPE 表、DMA 三槽、段号 4/5/6、Softmax 5 入 5 出、bmm 权重
三族、decode-block 裁剪、CompileSlots。

## P0 残留：核实与结果

| 项 | 上一轮缺口 | 本轮 |
| --- | --- | --- |
| 2.1 运行期尺寸闸门 | 只查 MatMul 权重 / KV 平面 | 保留；相位/槽 1/2 已按 CompileSlots 落盘 |
| 2.2 分配 ≥ 声明 | 闸门只查不重叠 | **新增闸门**：513 处声明，0 处欠分配 |
| 2.3 KV 边界 9 vs 10 | 缺写回出口 | **10 个 is_buffer**（K/V 入 + 位置 + mask + cos/sin + hidden + output + key/value_cache_out） |
| 2.5 文件族闸门 | 没有 | **参考独有族缺 0** |
| 节点/边 | 201/333 | **200 / 331**，与参考相等 |
| Reshape | 4 vs 2 | **2**（q/k 喂 RoPE 的 view 跨过去） |

## P1：核实全部属实，本轮修了什么

### 3.3 对拍器 schema 归一（先做）

`≤8` 保留会把头号 `map5` 当成纯数字、把节点号 8 当成槽位。改成按键
schema：`qidx` / `params` / 节点号通配；槽号 / `_phase_N` / RoPE 段号 /
`mapN` 保留。新增 `tests/test_diff_prepare_out.py`。

修好后 `DDR Weight Orig map<头号>` 不再被吃掉——3.2 的 `map{head}`
才能在对拍里消失。

### 3.2 Orig 名

- `DDR Weight Orig Buffer Name` = `buffer23_map{head}` / `buffer24_map{head}`
  （头号来自 `Split Head Index`）
- 残差 `output scale factor buffer`：优先双输入消费者（add_2 槽 0），
  不取 RMSNorm
- RoPE Dataout / DDR Output Orig：`{label}_cos.bin` / `{label}_mul_cos_buffer`
- q/k Gemm Dataout：穿透 Transpose 找到 RoPE，写成 `input_buffer_0_<rope>.bin`

### 3.4 Reshape 折叠

`_is_emittable` 对喂 RoPE 的 view/reshape 返回 False，边跨过去。
参考两处语义 Reshape（v 拆头前、concat 后）保留。

### 3.1 域值从 pimmlir 映射（最小切片）

FlagTree `ExpandPhases.cpp` 给已 stamp `pim.phase` 的 op 再 stamp：

- `"pim.flp-min-exp/max-exp/mantisa"`
- `"pim.kantor-mode"`（**txt 值**：DQ p3=3、RoPE mul=5，不是 dialect 枚举序）
- `"pim.fpsu-mode"` / `"pim.transpose-type"` / `"pim.activation-mode"`

`phase_plan.Phase` 解析这些键，`_attach_phase_data` 填进 `Layer`，
`layer_fields` **IR 有则用、没有回退 TABLE**。反证：
`test_ir_flp_overrides_hw_table`、`test_dq_p1_parses_flp_and_activation_mode`。

Gemm/bmm/mask 仍查表（ExpandPhases 不展开它们）。L2 fpsu 切片仍查表。

### 3.5 死代码

删 `_drop_tail`（GML 已裁）、`_weight_role`、`OrchestrationPlan.layer_kinds`。
`entry_bypass` 上一轮已接上，不回退。`layer_fields.py` 不拆（P2）。

## 验证

```
pytest -k "not llama2_7b"     704 passed
export_gml.py --decode-block-only   22/22 通过
节点 200 / 边 331 / is_buffer 10 / Reshape 2
txt 引用缺失 0
L2 分配 ≥ 声明 513/0
参考独有文件族 缺 0
对拍 MATCH 45886 / VALUE_DIFF 54 / MISSING 0
```

VALUE_DIFF 54 主要是 TVM `nprm_*`（mask 的 DDR Input Orig 1，37 处）和
编号体系不同的 Orig 名。`map<头号>` 已对齐。

FlagTree 改动：`ExpandPhases.cpp` + `expand_phases.mlir`。本仓未重编
FlagTree 时，固化 IR 夹具已带新 attr，解析与 txt 反证不依赖现场 pass。
