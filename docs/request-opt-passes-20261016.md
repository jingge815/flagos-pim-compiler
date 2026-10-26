# 需求文档：两层编译器五类通用优化 Pass

> 文档编号：request-opt-passes-20261016
> 创建日期：2026-10-16
> 关联项目：flagos-pim-compiler、FlagTree

## 一、需求背景与目标

图编译器负责模型图、分区和 PIMMLIR 生成，FlagTree 负责 PIMMLIR 到 C 的 lowering，GeneSim 只作验证载体。本需求在两层补齐五类 Pass，使 A、B 两路可触发，适配不同计算图且保持结果等价。

## 二、核心功能需求

### 2.1 功能清单

五项均为 P0，图层和算子层各有实现、注册、触发和测试。

1. **常量折叠**

   - 现状：图层没有独立实现；`runtime/compile.py:72-84` 仅导出并分区。FlagTree 的 PIM 变换目录目前只有融合、分块、DMA、相位等 Pass；`PIMOps.td` 未检出 `hasFolder`。
   - 需求：对输入全为编译期常量且无副作用的纯表达式求值；跳过动态输入、随机/写内存操作和超大结果。图层产出合法常量节点，算子层在 PIMMLIR 上可触发。
   - 验收：合成图和 TTIR/PIMMLIR 中常量链节点数减少；含动态输入的同型表达式不折；变换前后逐元素相等。
2. **分块**

   - 现状：图层没有分块入口。FlagTree `Passes.td:112-152` 的 `pim-tile-to-budget` 有 `full-m/full-n/full-k`，实现主要围绕 `tt.dot`；A 路在 `opcompiler_bridge/driver.py:308-321` 和 `genesim_bridge/flagtree_driver.py:65-70` 条件触发，B 路没有对应 Pass。
   - 需求：图层按静态形状和硬件预算产生 tile 约束并沿算子请求传递；算子层扩展为按循环/张量形状处理可分块算子，保留现有 `tt.dot` 结果。
   - 验收：图层 tile 满足预算且可复现；A 路 PIMMLIR 能看到合法 tile 属性或循环改写；非 `tt.dot` 合成算子也能触发；数值不变。B 路按张量形状给超预算的整算子矩阵乘选 tile，降到 C 时按块执行，数值与不分块逐位一致。
3. **算子融合**

   - 现状：`graph/fuse.py:60-101` 只按 `contracts/fusion_contract.py:30-43` 白名单匹配主算子和激活，并直接删除尾节点；`gml_bridge/export.py:122-150` 只在 GML 路按固定顺序调用专用融合。FlagTree `FuseActivation.cpp` 只识别 `pim.matmul/conv/eltwise + pim.lut (+ pool)`。
   - 需求：保留 RMSNorm、RoPE、门控激活等专用规则，同时增加按纯逐元素、形状不变、唯一消费者等性质判定的通用规则。执行路径必须保持 FX 数值语义，GML 路保持 contraction 编码。
   - 验收：白名单外的合成组合可触发；多消费者、有副作用或无法表达的链不融合；旧融合测试和现有 GML 结构不变；前后数值相等。
4. **DCE**

   - 现状：图层没有独立 DCE；`graph/fuse_pim.py:299-313` 明确其融合不删节点。FlagTree `TileToBudget.cpp` 的局部清理只服务自身重写，非通用 DCE。
   - 需求：按可达性删除无输出用户且无副作用的节点，保留输出、动态副作用和仍有用户的节点；两层均注册独立入口或等价通用 Pass。
   - 验收：无用户算子链被删，有用户同型链保留；图输出不变；A/B PIMMLIR 均能触发（B 路真实产物没有可删对象，见第五节第 3 条的例外）。
5. **CSE**

   - 现状：图层未实现 CSE。FlagTree Python 绑定已有通用 `add_cse`（`python/src/passes.cc:35-36`），但当前 PIM 管线未统一调用；`opcompiler_bridge/driver.py:804-808` 的 B 路只有融合、相位展开、契约校验。
   - 需求：对纯操作、相同属性和相同 SSA 输入的重复表达式保留一个定义；区分动态值、不同布局和副作用。
   - 验收：相同输入的重复表达式合并，不同输入不合并；两条算子路径和图层结构、数值均正确。

### 2.2 核心流程

图层在 `torch.export` 后进入统一优化入口，再进入执行路径和 GML 路：

```text
torch.export → 通用五类图 Pass → partition/spec 或 GML 专用融合与序列化
```

建议顺序：常量折叠 → CSE → 通用融合分析 → DCE → 分块。GML 专用融合必须保留既有顺序（`fuse_rope → fuse_for_pim → fuse_graph → split_heads → KV/DQ`）。算子层 A/B 两路五类 Pass 的相对顺序一致；A 路保留 convert、tile、explicit-DMA，B 路保留 phase-expand、verify 和 EmitC lowering。B 路的融合放在相位展开之前，其余几类放在展开之后。

### 2.3 边界定义

包含：两层 Pass 的分析、转换、注册、触发、A/B 管线统一和合成图/IR 测试。

不包含：GeneSim 核心逻辑、按分块结果反向改变图切分、非必要的新硬件算子、破坏现有 GML 字段的融合框架重写。

## 三、技术约束与依赖

1. 图层影响 `graph/`、`runtime/compile.py`、`gml_bridge/export.py`、`contracts/unified_ir.py` 和算子请求桥接；新增 `node.meta` 键必须登记在统一契约表。
2. FlagTree 影响 `include/triton/Dialect/TritonPIM/Transforms/Passes.td`、`lib/Dialect/TritonPIM/Transforms/`、对应 CMake 和 `python/src/passes.cc`。
3. PIM 算子没有已验证的通用 folder 接口时，不能假设 `canonicalize` 自动完成常量折叠；需补折叠接口或实现方言 Pass。CSE/DCE 必须尊重 MemoryEffects。
4. 两条算子路径的 Pass 顺序不能继续分别硬编码；至少覆盖 `opcompiler_bridge/driver.py:308-321,:804-808`、`phase_source.py:41-42`、`genesim_bridge/flagtree_driver.py:55-70` 和 FlagTree sidecar。

## 五、验收标准

1. 回归命令：`python -m pytest tests/ -q -k "not llama2_7b"` 全部通过；其后运行 `python -m pytest tests/ -q -k "llama2_7b"` 并通过。FlagTree 新增 lit/FileCheck 用例。
2. 每个 Pass 至少有图层、算子层、A 路 TTIR、B 路 PIMMLIR 三类结构断言和数值/语义验证；覆盖现有白名单之外的合成算子或循环。
3. 需要进行FlagTree功能单元验证，验证每个新增pass的功能正确性，可有效触发。常量折叠是例外：2026-10-20 实测三条默认通路的真实输入里没有可折对象（图层导出图的可折叠节点输入全是激活或模型参数，A 路 linear kernel 没有两个操作数都是常量的运算，B 路展开后没有 `arith.constant`），所以它在合成 IR 上验证，默认通路上改动数为 0 是预期，不是漏触发；
   图层分块是第二个例外：2026-10-23 实测默认 65536 字节预算下，7B 四种投影要切 2048 到 32000 块，超过 512 块上限，图层不切，交给算子层 A 路 `tt.dot` 分块；回归用小模型的投影都在预算内。图层分块在合成图和压小的预算上验证，默认通路上改动数为 0 是预期。A 路、B 路分块在默认通路上照常触发；
   B 路死代码消除是第三个例外：2026-10-25 实测展开后的相位都是张量结果，按内核输出保留，softmax、动态量化、matmul 默认通路上改动数为 0，所以它在真实展开产物里夹入无用户标量的合成 IR 上验证。这一例外由需求方 2026-10-25 确认接受：一层 Llama 的默认 B 路产物共 54 个算子（decode）与 74 个（prefill），`-pim-dce` 前后逐字节相同，真实展开产物里本来就没有可删对象，不是 Pass 漏触发。A 路死代码消除在默认通路的门控 linear 上触发；
   B 路公共子表达式消除是第四个例外：2026-10-25 实测一层 Llama 的发射器真实产物（prefill 20 个函数、decode 15 个函数）展开后，同一函数内没有结构相同的算子，`-pim-cse` 前后完全相同。所以它在合成 IR 上验证，含整算子函数里的合并与降到 C 后的数值对拍，默认通路上改动数为 0 是预期。这一例外由需求方 2026-10-25 确认接受。A 路公共子表达式消除在 64×64、256×256 真实 linear 上默认触发；
   图层公共子表达式消除在 GML 路上改动数为 0：真实图里重复的只有各层的 rope 形状类算子，形状类只在执行路径、没有流水并行时合并，GML 路不合并。执行路径无流水并行时默认触发，有流水并行时为 0，不跨 stage 合并。这一口径由需求方 2026-10-25 确认接受；
4. CSE、DCE、常量折叠和融合不能误删输出或副作用；分块不能超 WRAM/MRAM 预算；现有 `test_fuse*.py`、`test_fusion_contract.py`、`test_opcompiler_linear.py` 和 GML 闭合检查保持通过。
5. 验收记录应包含每个 Pass 的触发日志、变换前后 IR/图摘要、输出对拍结果和 A/B 管线顺序。

## 六、调研补充信息

本轮未做外部调研。MLIR canonicalizer/CSE 仅作参考，是否适用于 PIM 方言须以折叠接口、纯性和副作用为准。

## 七、本轮实测对既有结论的修正

1. 需求描述把融合概括为通用优化；实测 GML 融合是目标格式要求，且 `graph/fuse.py:63-64,92-95` 会删除 FX 尾节点。因此执行路径不能直接复用该删除行为，必须另设保持数值语义的入口。
2. 需求描述把两层视为同一条管线；实测 A/B 当前列表不同：A 路在 `driver.py:308-321` 只有转换、分块、DMA，B 路在 `driver.py:804-808` 只有融合、相位、校验。
3. FlagTree 已有通用 Python `add_canonicalizer`/`add_cse` 包装（`python/src/passes.cc:35-36`），但不等于 PIM 管线已经触发五类 Pass。

## 八、待确认事项

1. A/B 是否仅指 TTIR 入口与整算子级 PIMMLIR 入口；若另有业务定义，需补充对应触发点。
2. 五类 Pass 是否默认启用，还是由配置显式启用？本文按“统一管线默认启用”编写。
3. 分块验收是否有固定 tile 尺寸、WRAM 预算或性能目标？本文按“预算内、功能等价、无固定尺寸”编写。
4. 常量折叠是否包括 shape/index 等标量表达式，还是只包括张量值？本文暂按纯值表达式，并要求对 shape/index 单独补测试。
