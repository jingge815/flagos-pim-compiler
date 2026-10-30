# 让 GeneSim 按 KV 块循环计费 `pim.paged_attn`（2026-10-30）

## 问题

`pim.paged_attn` 是 FlagTree 里直接降 C 的融合算子（QKᵀ + 单遍 online softmax + PV），
在 pim mlir 里只有一行。它没有 `unit`，也没有相位链，GeneSim 读它只有两条路：

- 报 `PimirTraceError`（单元未知，按 0 计会让它免费）；
- 或者退化成「整算子一次遍历」，历史长度完全没进去。

前一步已给它盖上 6 条 `#pim.phase_spec`（`ExpandPhases.cpp::stampPagedAttn`），
GeneSim 能读成 6 步 19 条指令，但**每步的循环次数还是 1，KV 的搬运字节也没进账**，
所以代价仍然错一两个数量级。本次把这块补齐。

## 设计：让 IR 自己说明「这一段按块重复」

`#pim.phase_spec` 增加两个可选参数，默认值让旧 IR 文本逐字节不变：

| 参数 | 含义 | 谁写 |
|---|---|---|
| `perBlock`（bool，默认 false） | 这一相在 KV 块循环体内，每个历史块跑一遍 | 相 0-4 |
| `poolBytes`（int64，默认 0） | 每次遍历从 KV 槽池读入的字节（一块 K 或一块 V） | 相 0（K）、相 4（V） |

两者合起来的含义是「这条相位链里有一段按块重复、每次都读池子」。
GeneSim 因此不需要认算子名，也不需要为它新增 `_OPLEVEL_FALLBACK_UNIT` 条目；
谁的 `pimir_path` 指向这份产物，谁就按块循环计费。

块数本身**不写进 IR**：它是运行期的历史长度，编译期写一个上界会读成真实规模
（这也是 `stampPagedAttn` 当初不肯把块数折进 `bytes` 的原因）。

## 改动

### 1. FlagTree（`flagOS-installers/FlagTree`）

| 文件 | 改动 |
|---|---|
| `include/triton/Dialect/TritonPIM/IR/PIMAttrDefs.td` | `PhaseSpecAttr` 追加 `perBlock` / `poolBytes`，description 补两段 |
| `lib/Dialect/TritonPIM/IR/Dialect.cpp` | `PhaseSpecAttr::verify` 加两参，新增两条不变量：`poolBytes >= 0`；`poolBytes > 0` 必须 `perBlock` |
| `lib/Dialect/TritonPIM/Transforms/ExpandPhases.cpp` | `makePhase` 加两个带默认值的形参；`stampPagedAttn` 给相 0-4 打 `perBlock`、给相 0/4 打 `poolBytes = slotTokens × headDim × elemBytes` |
| `test/Dialect/TritonPIM/paged_attn_phases.mlir` | 新增：`-pim-expand-phases -pim-verify-gml-contract` 后 CHECK 六条相位 |
| `test/Dialect/TritonPIM/phase_spec_negative.mlir` | 新增两条反例：池读不在块循环里、`poolBytes` 为负 |

打印顺序随声明序，只打非默认值：

```
#pim.phase_spec<index = 0, bytes = 32, unit = nmu, perBlock = true, poolBytes = 4096>
#pim.phase_spec<index = 5, bytes = 256, unit = combiner, reads = [3, 4]>
```

`pim-lower-to-emitc` 只看 `hasAttr("phases")`，不读新字段，所以生成的 C 逐字节不变
（实测 `EMITC_IDENTICAL`）。

### 2. 编译器成本侧（`genesim_bridge/ir_cost.py`）

`_line_phase_movement_bytes` 从「按正则取 `bytes` 求和」改成「按 `键 = 值` 拆参数」：

- 带 `perBlock` 的相按**块数**乘，块数取同一行块表操作数的首维
  （`tensor<4xi32>` → 4），与 `_line_operator_flops` 取历史上界的口径一致；
- `poolBytes` 每块加一次；
- 没有 `perBlock` 的相（softmax/lut/... 的全部相位链）行为与改动前相同。

`opcompiler_bridge/phase_plan.py` 没有改：它按 `键 = 值` 字典解析，新字段被忽略，
眼下也没有消费者需要它们。

### 3. GeneSim 解析与发射（`src/pim/pimir_trace.py`）

- 相位解析改成「先抓 `<...>` 整段，再按 `键 = 值` 拆字典」，不再依赖参数顺序；
  新增字段不会让整相从账本里消失。
- `PhaseStep` 新增 `per_block` / `pool_bytes`；`in_bytes` 在原口径（该行全部操作数）
  之外加上 `pool_bytes`——cache 是 `!pim.memdesc`，类型正则抓不到它。
- `compile_phase_chain` 把**连续的 perBlock 前缀**包进
  `OPCODE_LOOP_BEGIN_INDIRECT … OPCODE_LOOP_END`，其后的相发在循环外；
  夹在中间的 perBlock 相直接报错（一条循环只能包前缀）。
- `phase_metadata` 增加 `has_block_loop` / `loop_unit_tokens`（IR 的 `slotTokens`）/
  `tokens`（q 的行数），随 trace metadata 走。

### 4. GeneSim 运行时绑定（`src/vpu/vpu.py`）

`_execute_runtime` 在按算子类型分派**之前**先看 trace metadata：

```
有 has_block_loop → loop_count = ceil(历史长度 / loop_unit_tokens)
                    repetitions = 1（tokens > 1 时），否则按查询行数
否则             → 老规则（SOFTMAX / GEMV_* / GEMM / 向量算子）
```

`历史长度` 仍取 `_extract_pim_length`（GEMV 类取共享 KV 的倒数第二维）。
`tokens > 1` 的那一条是 prefill：trace 已经覆盖整段 prompt，再乘一遍查询行数
会把代价放大 T 倍（T² 而不是 T）。`execute_profile` 用同一个绑定，避免内存账与
时延账各按一个重复次数算。

## 验证

| 项 | 结果 |
|---|---|
| FlagTree | 重建 `triton-opt` + `libtriton.so` 并原子替换进 PyTorch 前缀；PIM 方言 lit 全绿（46/46，含新用例） |
| EmitC 不变 | 同一份 paged_attn 降级出的 C 与改动前逐字节相同 |
| 编译器 | `tests/test_ir_cost.py`、`test_genesim_bridge.py`、`test_phase_plan.py`、`test_opcompiler_ops.py` 共 130 项全过 |
| 成本算例 | 同一份 paged_attn：flops 不变 33088，`mram_traffic_bytes` 从 840 变成 35360 |
| GeneSim | `tests/sim` 758 项通过、1 项跳过；新增 7 项（块循环解析、指令流、块数缩放、KV 每位置字节、元数据、两条 vpu 绑定） |
| 端到端 | `./run.sh --config conf/sim.yaml` 跑通 |

成本算例的算式（tokens=1 / head=128 / slotTokens=16 / 块表 4 项）：

```
工作区相 0-4 = 32 + 4 + 32 + 4 + 512 = 584 B，每块一次 → ×4 = 2336
相 5（归一化写回）= 256 B，整算子一次      → 256
KV 流 = (K 4096 + V 4096) × 4 块            → 32768
合计                                         35360
```

KV 的量级对照：一块 16 个位置搬 K+V 共 8192 B，即每位置 512 B，
与手写 `GEMV_SCORE + GEMV_CONTEXT` 模板的 512 B/位置一致；
多出来的是每相 272 B 的操作数记账（q 256 + 块表 16），约 30%。

## 已知限度

- 每相仍把该行**全部**操作数当 `in_bytes`（paged_attn 是 272 B/相），相对 4 KiB 的
  KV 块是小量，但 prefill 大 token 数下会累积。修法是让相位自己声明操作数，
  属于下一步。
- `set_tile_count` 上限 0xFFF（4095 块）：slot 16 时是 65520 个 token，超出会静默截断。
  本改动没走 `LOOP_BEGIN_INDIRECT_LONG`。
- 单元 opcode 的周期是每指令一个常数，`imm` 只带规模信息，所以「字节算对」不等于
  「周期精确」。
- 「哪个 ModelIR 节点该拿 paged_attn 的 pimir」不在本次范围：ModelIR 仍按
  `GEMV_SCORE / MASK / SOFTMAX / GEMV_CONTEXT` 分解，放置 sidecar 也没动。
  utils 侧一旦把 `pimir_path` 指向这份产物，上面的规则立即生效。
- 块循环绑定挂在 `PIMVPU._execute_runtime` 上，只有 `_RUNTIME_PARAMETERIZED_OPS`
  里的算子类型才走得到。将来若给 ModelIR 新增 `PAGED_ATTN` 类型，必须同时登记
  进那张表，否则循环次数停在 1，代价又退回到「一块当全部历史」。
- 重建 `libtriton.so` 改变了 `_compiler_fingerprint`，`.opcompiler_cache` 的全部键
  失效；`genesim/models/llama2_7b_tp2_pp4_placement.json` 里记的 `pimir_path`
  因此会解析失败，需要重新导出放置表才能走 pimir 路径。
