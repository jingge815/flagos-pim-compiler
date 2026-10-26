# 技术设计文档：两层编译器五类通用优化 Pass

> 文档编号：design-opt-passes-20261016
> 创建日期：2026-10-16
> 关联需求文档：request-opt-passes-20261016
> 关联项目：flagos-pim-compiler、FlagTree

## 一、设计概述

### 1.1 需求背景回顾

图编译器产出模型图与 PIMMLIR，FlagTree 把 PIMMLIR 降到 C。两层目前只有按算子名硬编码的融合和分块，常量折叠、死代码消除、公共子表达式消除都没有。

### 1.2 设计目标

在图编译器和算子编译器中都实现常量折叠、分块、算子融合、死代码消除和公共子表达式消除，覆盖 A、B 两路；上述优化功能验证正确且是通用的，并且可以被有效触发。五类优化在执行路径、A 路、B 路上默认生效。GML 路有两处例外：分块与通用融合都不跑。GML 按整块算子编码 contraction 与缓冲名，切成子块会改掉这些字段，所以 `fuse_for_gml` 传 `tile=False`；通用融合换出的合并节点是普通 Python 闭包，`from_fx` 不认它，被吃掉的计算会从产物里静默消失，融合由 `fuse_graph` 与专用 Pass 负责。折叠、CSE、DCE 在 GML 路照常跑。B 路分块的对象是整算子 `pim.matmul`：按形状选 tile 写在算子上，降到 C 时按它生成分块循环，详见四·附 B。

### 1.3 设计范围

- 包含范围：五类优化在两层的实现与触发、A/B 两路统一顺序、按层按路拆开的结构断言和数值对拍。
- 不包含范围：GeneSim 核心逻辑、按分块结果反向改图切分、非 2 的幂 shape 的 padding、改写现有 GML 字段。

## 二、现状基线

**图层有两条消费路径，优化对它们的影响方式不同。**

执行路径 `runtime/compile.py:194` 的 `compile_llama2` 把 FX 图交给 `build_execution_plan`（`runtime/exec_plan_gen.py:321`）。它按 `node.target` 逐节点发 `launch` 命令，payload 只有形状、dtype、shard（`:435-465`），不读 `fused_tail` 等融合标记。`runtime/executor.py:25` 只按命令执行。因此图层上只有删节点、并节点才会改变执行结果，只写 meta 不会。

GML 路 `gml_bridge/export.py:142` 的 `fuse_for_gml` 依次跑 `fuse_rope`、`fuse_for_pim`、`fuse_graph`、`split_attention_heads`、`insert_kv_dma_and_split`、`insert_dynamic_scaling`。前两个只打标不删节点（`graph/fuse_rope.py:132`、`graph/fuse_pim.py:310`），`fuse_graph` 会 `erase_node`（`graph/fuse.py:87`）。

**算子层两条路的输入形态不同。**

A 路 `_run_triton_opt`（`opcompiler_bridge/driver.py:276`）吃 TTIR，来源是 `kernel_src.py:18` 的 `linear_kernel`：循环嵌套加 `arith` 加 `tt.dot`，是多算子 IR。顺序为 `convert-triton-to-pim → pim-tile-to-budget → pim-explicit-dma`。

B 路 `_run_oplevel_triton_opt`（`driver.py:792`）吃算子级 IR，来源是 `oplevel_kernel.py` 的 16 个函数，每个函数体只有一条 `pim.*`（如 `matmul_kernel:159` 只有一条 `pim.matmul`）。顺序为 `pim-fuse-activation → pim-expand-phases → pim-verify-gml-contract`。`pim-expand-phases` 会把 `quantize`、`softmax`、`rope` 展开成多条相位链。

同一份列表还抄在 `phase_source.py:41`、`genesim_bridge/flagtree_driver.py:55`、FlagTree `python/triton/backends/pim_sidecar.py:72`。

**FlagTree 已有的通用优化不作用在 PIM 算子上。** `python/src/passes.cc:32` 有 `add_canonicalizer`、`add_cse`、`add_symbol_dce`、`add_sccp`，`passes.ttir` 有 `add_combine`、`add_loop_aware_cse`。但 `PIMOps.td` 中 `hasFolder` 为 0，`matmul`、`eltwise`、`lut` 都没标 `Pure`，这些 Pass 对它们不生效。

## 三、总体设计

### 3.1 总体设计概述

- 解决的问题：五类优化在两层缺失或不通用，且触发点各写各的。
- 现状：图层优化只在 GML 路且按算子名；算子层优化只覆盖 `tt.dot` 和激活尾，B 路输入是单算子。
- 方案：图层五类都做成会改图的 Pass，执行路径因此真正受益；算子层五类都做成 PIM Pass，插在 B 路相位展开之后，让单算子展开出的多算子 IR 也得到优化。

```
图层  torch.export
        │  新增 optimize_graph
        │    常量折叠 → CSE → 融合 → DCE → 分块
        ├── 执行路径 compile_llama2      （原有，读到的是优化后的图）
        └── GML 路 fuse_for_gml          （原有六个专用 Pass 保留，不分块不通用融合）

算子层  五类顺序 OPT_PASSES = 折叠 → CSE → 融合 → DCE → 分块
        ├── A 路  convert → canonicalize → cse → LICM → OPT_PASSES → explicit-dma（TTIR，多算子）
        └── B 路  融合 → expand-phases → 折叠 → CSE → DCE → 分块 → verify
                  （分块按形状切整算子 pim.matmul，降到 C 时生成块循环）
```

### 3.2 与现有逻辑的衔接

`compile_llama2` 在两次 `export_annotated_graph` 之后、`propagate_specs` 之前调 `optimize_graph`，这样切分传播看到的就是优化后的图。没有流水并行时它传 `merge_shape_ops=True`，这是 `optimize_graph` 唯一读策略的地方。`fuse_for_gml` 开头先调 `optimize_graph`，再跑原有六个 Pass。

算子层新增 `opcompiler_bridge/pipeline.py` 导出 `a_path_passes(hardware)` 和 `b_path_passes()`。`driver.py`、`phase_source.py`、`flagtree_driver.py` 改为调用它。FlagTree 的 `pim_sidecar.py` 不引本仓，按同一顺序内联，由测试钉住顺序。

### 3.3 模块划分与职责边界

- `graph/opt_passes.py`：图层五个 Pass，都改 FX 图，不写新 meta。
- `graph/fuse.py` 三个专用融合：保持原样，只服务 GML 编码。
- `opcompiler_bridge/pipeline.py`：两路 Pass 顺序的唯一真源。
- FlagTree `Transforms/`：算子层五个 Pass，三个新增、融合与分块各改一处。
- 分块的循环改写只由 `-pim-tile-to-budget` 做；图层只负责算尺寸并下发。

## 四、模块详细设计

### 4.1 图层优化入口（对应需求点：2.2｜现有代码位置：`runtime/compile.py:207`｜改动类型：新增）

**功能点**：新增 `optimize_graph(gm)`，执行路径和 GML 路都调它。
**原因**：图层优化没有统一入口，执行路径一个优化都不跑。
**代码对比**：

```python
# 原有（runtime/compile.py:207）
prefill_gm = export_annotated_graph(...)
prefill_edges = propagate_specs(prefill_gm, strategy)

# 目标
prefill_gm = export_annotated_graph(...)
optimize_graph(prefill_gm)
prefill_edges = propagate_specs(prefill_gm, strategy)
```

**解决思路**：顺序为常量折叠、CSE、融合、DCE、分块，每步返回改动数。

### 4.2 图层常量折叠（对应需求点：Pass 1｜现有代码位置：无｜改动类型：新增）

**功能点**：输入全是常量的纯算子换成一个常量节点。
**原因**：图层没有常量折叠，能确定的子图仍逐节点执行。
**代码对比**：

```python
# 原有：无

# 目标（graph/opt_passes.py）
def fold_constants(gm):
    for node in list(gm.graph.nodes):
        if node.target in _PURE and _all_constant(node):
            node.replace_all_uses_with(_const(gm, _eval(node)))
```

**解决思路**：常量指 `get_attr` 结果；随机和写内存算子排除。

### 4.3 图层公共子表达式消除（对应需求点：Pass 5｜现有代码位置：无｜改动类型：新增）

**功能点**：目标与参数都相同的纯算子只留第一次。
**原因**：重复子图各算各的，没有按结构合并的步骤。
**代码对比**：

```python
# 原有：无

# 目标
def eliminate_common(gm):
    seen = {}
    for node in list(gm.graph.nodes):
        key = (node.target, tuple(a.name for a in node.args))
        if key in seen and node.target in _PURE:
            node.replace_all_uses_with(seen[key])
        else:
            seen[key] = node
```

**解决思路**：键取结构不取值，参数不同的同型算子不合并；字面量连类型一起入键，`1` 与 `1.0` 不合并。默认只合并有实际计算的算子；没有流水并行时，`compile_llama2` 传 `merge_shape_ops=True`，各层重复的形状类算子也合并。

### 4.4 图层算子融合（对应需求点：Pass 3｜现有代码位置：`graph/fuse.py:60`｜改动类型：新增）

**功能点**：把单消费者的逐元素算子替换成一个合并节点。
**原因**：现有融合按算子名匹配且只在 GML 路跑，执行路径不读 meta。
**代码对比**：

```python
# 原有（graph/fuse.py:92）：删尾节点，语义只留在 meta，执行路径读不到
last.replace_all_uses_with(main)
gm.graph.erase_node(node)

# 目标：尾节点换成一个真正的合并算子，node.target 即语义
fused = graph.call_function(_fused_op(producer, consumer), producer.args)
consumer.replace_all_uses_with(fused)
```

**解决思路**：判定按算子性质。合并算子是普通 Python 闭包，`partition` 判为主机节点，执行计划发 `host_op` 在主机上求值——闭包调 aten 原语，数值与 FX 语义一致；占位符经 `_as_torch` 转成 torch 张量，numpy 绑定不会抛「Expected Tensor」。设备融合由 `fuse_linear_activation` 做：切分传播之后，把 linear 后的单用户尾部激活（按 `LINEAR_TAIL_ACTIVATIONS` 判定，现为 silu、relu、sigmoid）折进 linear 节点，A 路编译内核按 `activation` 常量把激活算在累加器上，见 4.11 的 A 路形态。这类激活通用融合不碰。

### 4.5 图层死代码消除（对应需求点：Pass 4｜现有代码位置：无｜改动类型：新增）

**功能点**：删除没有用户的纯算子。
**原因**：折叠和融合留下无用户节点，图层没有清理步骤。
**代码对比**：

```python
# 原有：无

# 目标：复用 torch.fx 已有实现
def eliminate_dead(gm):
    gm.graph.eliminate_dead_code(is_impure_node=_impure)
    gm.graph.lint()
```

**解决思路**：`is_impure_node` 沿用 FX 的 `Node.is_impure()`，保住写内存、随机和运行期断言，其余按无用户删除。唯一例外是 `_assert_tensor_metadata`：它只核对编译期已知的形状与类型，按死代码删。

### 4.6 图层分块（对应需求点：Pass 2｜现有代码位置：`runtime/exec_plan_gen.py:435`｜改动类型：新增）

**功能点**：超预算的算子拆成多个预算内的子节点。
**原因**：执行路径按节点发命令，只写标记不改变执行。
**代码对比**：

```python
# 原有：一个 linear 节点对应一条 launch 命令

# 目标：超预算时换成若干个子块节点，每个都在预算内
for tile in _tiles(shape, budget):
    graph.call_function(node.target, _sliced_args(node, tile))
node.replace_all_uses_with(_concat(parts))
```

**解决思路**：子块沿用原算子，结果用 `concat` 拼回。

### 4.7 算子层统一管线（对应需求点：2.2｜现有代码位置：`opcompiler_bridge/driver.py:308`｜改动类型：新增）

**功能点**：两路 Pass 顺序由 `pipeline.py` 给出，调用点改用它。
**原因**：管线抄在四处且不一致，加 Pass 必然漏改。
**代码对比**：

```python
# 原有（opcompiler_bridge/phase_source.py:41）
PASS_PIPELINE = ("-pim-fuse-activation", "-pim-expand-phases",
                 "-pim-verify-gml-contract")

# 目标
from opcompiler_bridge.pipeline import b_path_passes
PASS_PIPELINE = b_path_passes()
```

**解决思路**：五类 Pass 的相对顺序两路一致。A 路在五类之前先跑上游 `canonicalize`、`cse` 与 `loop-invariant-code-motion`：前两个覆盖 TTIR 里 PIM 三个 Pass 不认的 `arith`；循环不变量外提不属于五类，放在分块之前，让分块看到的循环体只剩随迭代变化的部分。B 路融合放在相位展开之前，折叠、CSE、DCE、分块放在展开之后。B 路模块头由 `pipeline.with_budget` 补上 WRAM、MRAM 预算与 DMA 对齐；A 路分块读这三个，B 路分块只读 WRAM 与 MRAM，B 路没有 DMA，候选不检查对齐。

### 4.8 算子层常量折叠（对应需求点：Pass 1｜现有代码位置：`python/src/passes.cc:35`｜改动类型：新增）

**功能点**：新增 `-pim-constant-fold`，折叠常量输入的算子。
**原因**：算子无 `hasFolder`，上游 `canonicalize` 折不动。
**代码对比**：

```cpp
// 原有：PIM 管线无常量折叠

// 目标（ConstantFold.cpp），参照 triton-combine 的模式匹配
if (isPurePim(op) && allOperandsConstant(op))
    rewriter.replaceOp(op, evalConstant(op));
```

**解决思路**：覆盖 `pim.eltwise` 的 `add`、`sub`、`mul`、`div`、`max`、`min`，以及 `pim.lut` 的 `relu`、`exp`、`sqrt`、`reciprocal`、`silu`。`pim.param` 只有名字没有值，不在范围内。

### 4.9 算子层公共子表达式消除（对应需求点：Pass 5｜现有代码位置：`python/src/passes.cc:36`｜改动类型：新增）

**功能点**：新增 `-pim-cse`，合并重复的 `pim.*` 表达式。
**原因**：上游 CSE 只处理标 `Pure` 的算子，这三个都没标。
**代码对比**：

```cpp
// 原有：PIM 管线不调用 CSE

// 目标（PimCSE.cpp）：按性质表判定，不看 Pure
if (isPurePim(op) && seen.count(key(op)))
    rewriter.replaceOp(op, seen.lookup(op));
```

**解决思路**：B 路放在相位展开之后，重复子表达式才合并得了。

### 4.10 算子层死代码消除（对应需求点：Pass 4｜现有代码位置：`python/src/passes.cc:33`｜改动类型：新增）

**功能点**：新增 `-pim-dce`，删除结果无用户的算子。
**原因**：上游 `symbol-dce` 只删符号，删不了函数体内的死算子。
**代码对比**：

```cpp
// 原有：PIM 管线无死代码消除

// 目标（PimDCE.cpp）
if (isPurePim(op) && op->use_empty())
    rewriter.eraseOp(op);
```

**解决思路**：与 CSE 共用 `isPurePim`，放在融合之后。

### 4.11 算子层融合通用化（对应需求点：Pass 3｜现有代码位置：`FuseActivation.cpp:38`｜改动类型：修改）

**功能点**：融合判定从三个算子名改为按性质。
**原因**：`isFusionTarget` 只认三个算子，表外的不触发。
**代码对比**：

```cpp
// 原有（FuseActivation.cpp:38）
return isa<MatmulOp, ConvOp, EltwiseOp>(op);

// 目标：主算子按是否带 activation 属性判定，激活按 pim.lut 判定
return op->hasTrait<OpTrait::ZeroRegions>() && acceptsActivation(op);
```

**解决思路**：能带 `activation` 属性的算子都可作主算子。A 路形态另开一条识别路径：`tt.dot` 没有 activation 属性位，Triton 对 `acc * sigmoid(acc)` 展开出的五连（`0-x → exp → 1+e → 1/(1+e) → x*s`）又保持 arith/math 形态（`convert-triton-to-pim` 不碰浮点算子，实测），所以按链的结构判定种类：链根（dot 结果，或经单携带值 K 循环拿到的循环结果）唯一用户是与 0 取大即 relu；唯一用户是 `0-x → exp → 1+e → 1/(1+e)` 四连即 sigmoid；两个用户、四连之后再乘回链根即 silu。常数是 0.0/1.0 的 dense splat，中间结果单用户。匹配上就把对应的 `activation = #pim.act_spec<kind = ...>` 写进 tt.dot、删掉整条链；`-pim-tile-to-budget` 重建 dot 时把属性带过去，`-pim-lower-to-emitc` 在累加器写回前补对应的 `pim_lut_*`。

### 4.12 算子层分块通用化（对应需求点：Pass 2｜现有代码位置：`TileToBudget.cpp:889`｜改动类型：修改）

**功能点**：没有 `tt.dot` 时按循环结构分块。
**原因**：现在没有 `tt.dot` 直接报错，B 路因此没有分块。
**代码对比**：

```cpp
// 原有（TileToBudget.cpp:892）
if (dots.empty())
    return signalPassFailure();

// 目标：没有 tt.dot 时改走通用循环分块
if (dots.empty())
    return tileGenericLoops(mod);
```

**解决思路**：`tt.dot` 路径保留；通用路径按 `scf.for` 边界选 tile。

## 四·附、Pass 设计细则

每个 Pass 写清四项：适用范围（支持哪些算子、明确不支持哪些）、分析（判定依据）、
转换（具体改什么）、顺序依据（为什么放在这个位置）。评估不单列，因为它决定一个
其中评估只用于分块，因为只有分块有多个合法结果需要取舍；其余四个 Pass 满足分析条件就必须变换，不设收益模型。
IR 空转，测试要断言它确实改动了 IR 而不是只跑了一遍。

### A. 图层（操作 torch.fx 图）

**常量折叠**

- 适用范围：输入全部可追溯到 `get_attr` 或常量、且有实际计算的逐元素算子，即算术（`add`、`mul`、`sub`、`div`）和激活。不支持：形状类（`reshape`、`transpose`、`squeeze`，它们不产生新数值，折掉会改变切分传播看到的节点，跨段通信边随之变化）、参数含占位符的、随机算子、写内存算子、`linear`（权重虽是常量但输入是激活）。
- 分析：从常量节点向前传播，一个算子的所有输入都已折叠才可折。无收益取舍：输入全是常量就折，否则不折。
- 转换：用求值结果新建一个常量节点，`replace_all_uses_with` 接过去，原节点留给后续 DCE。
- 顺序依据：放第一个。它产生的常量节点是 CSE 的合并对象，也让融合看到的输入更简单。

**公共子表达式消除**

- 适用范围：同一图内 `target` 与参数都相同的有计算的纯算子；没有流水并行时（`num_stages == 1`）也包括形状类算子。纯性按 aten schema 判定，不查算子名单：不写输入、不随机（与 DCE 同用 `Node.is_impure()`）；形状类指返回值是输入别名的算子。不支持：有副作用的算子、参数里含张量值而非节点引用的、跨图的、流水并行时的形状类算子（各层在不同 stage，合并会让后面的层跨 stage 读第一层的结果）、结果或其视图会被原地写的（如 `clone` 后接 `mul_`，合并会让两个用户共用一块缓冲）、读了被原地写过的缓冲的（写前写后同一个键取到的值不同）。原地写按 schema 的写别名标记判定。字面量按类型和值比较。
- 分析：以 `(target, 参数节点名序列)` 为键建表，后出现的命中先出现的即重复。无收益取舍：结构相同就合并。折叠产生的 `_folded_*` 常量按类型、形状与值入键，相同的合并；模型参数不按值合并。原地写的参数是张量列表时（如 `_foreach_mul_`），列表里每个节点都算被写。
- 转换：重复节点的用户改接到第一次出现的节点，重复节点留给 DCE。
- 顺序依据：放在折叠之后，折叠新增的常量节点才能参与合并；放在融合之前，避免同一子图被融合两次。

**算子融合**

- 适用范围：生产者是纯算子、消费者是纯逐元素算子、且生产者只有一个用户。不支持：多用户（会改变另一个用户拿到的值）、消费者含归约或形状变化、已有专用融合覆盖的 `silu`/`rope`/`rsqrt` 链（那些走 GML 专用路径）。
- 分析：沿单消费者链找满足上述性质的相邻对。无收益取舍：单消费者且两边都是纯算子就合并。
- 转换：新建一个合并算子节点承载两者的计算，消费者的用户改接过去。合并算子需要在 `runtime/kernels.py` 有实算实现，否则执行路径缺内核。
- 顺序依据：放在折叠和 CSE 之后，输入已化简；放在 DCE 之前，被替换的原节点由 DCE 清理。

**死代码消除**

- 适用范围：通用，不按算子名单。从输出和有副作用节点（写内存、随机、运行期断言，以 FX `Node.is_impure()` 为准）反向走不到的都删；有副作用的节点本身是可达性的根，不删。导出器插入的 `_assert_tensor_metadata` 例外，按死代码删。
- 分析：复用 `Graph.eliminate_dead_code` 的可达性，`is_impure_node` 排除有副作用的。只看使用：从输出和有副作用节点反向标可达，不可达才删。
- 转换：直接删节点。
- 顺序依据：放在折叠、CSE、融合之后，三者都只改用户不删节点，死节点集中在这里清。

**分块**

- 适用范围：工作集超过 `wram_bytes_per_dpu` 的 `linear` 与 `matmul`。工作集是输入、权重、输出三块加上 bias 与其余常驻输入，判断整块是否超预算与给候选计价用同一个函数。不支持：逐元素算子（工作集等于输出本身，拆了不减）、已在预算内的、右操作数带 batch 维的 `matmul`（各批权重不同，按左操作数切行窗口会把批次对错）。
- 与切分策略的衔接：`optimize_graph(gm)` 不需要策略，按列切与按行切的权重都可以沿 K 切。按行切的权重沿 K 切开后，一段 K 只与部分台的分片相交，切分传播按权重切片在各台的区间给激活提要求，`linear` 的部分和只标在这些台上；没有这段 K 的台贡献为 0，相加时按 0 参与，最后归约一次。按列切的权重沿 K 切开后各段输出区间相同、持有者是部分台，相加的输出沿用这组持有者，不再插收集。
- 块数上限：一个节点最多拆成 `MAX_TILE_BLOCKS = 512` 块，超过时图层不切。真实 7B 投影在默认 65536 字节下要拆 2048 到 32000 块（`down_proj` 单个就让 `plan_dpu` 跑 262 秒），这类算子留给算子层 A 路的 `tt.dot` 分块压进预算。A 路在默认 8 DPU、16 tasklet 下按分片后的本地形状（如 `gate_proj` 的 5504、1376，`lm_head` 的 4000 与整块 32000，`down_proj` 的部分和）验证过：tile 不整除维度时尾块不越界、不改输入、数值一致。
- 分析：见下方评估模型。
- 转换：一个节点换成若干个子块节点加一个 `concat`。每个子块仍是原算子，执行路径会据此各发一条命令。
- 顺序依据：放最后。前面的 Pass 会改变节点形状，预算比较要针对最终形状。

### 分块评估模型（图层与算子层共用）

分块是个 Pass 里唯一有多个合法结果的，所以只有它需要评估模型。候选是各维上能把单块工作集压进 `wram_bytes_per_dpu` 的 2 的幂切分。模型对每个候选算两项：工作集字节，必须不超预算；切分块数，越少越好。选满足预算且块数最少的候选；没有候选能压进预算、原样已在预算内，或图层最少块数超过上限，则不改。

图层用它决定把一个 `linear` 拆成几块，算子层用它决定 `scf.for` 的步长。两边输入不同（一个是 FX 节点的形状，一个是循环跨度），比较规则相同，避免两层对同一个算子选出冲突的 tile。

### B. 算子层（操作 MLIR）

**常量折叠 `-pim-constant-fold`**

- 适用范围：操作数全是 `arith.constant` 的 `pim.eltwise`、`pim.lut`，以及 A 路里的 `arith` 整数加减乘。`pim.eltwise` 覆盖 `add`、`sub`、`mul`、`div`、`max`、`min`，`pim.lut` 覆盖 `relu`、`exp`、`sqrt`、`reciprocal`、`silu`，浮点覆盖 f16 与 f32，整数 dense 张量只覆盖 i32。不支持：`pim.param`（只有名字、没有值，折不了）、`absmax`（是归约不是逐元素）、`sigmoid`、`tanh`、`gelu`、`rsqrt`、`identity`、`leaky_relu`、`relu_x`（编译期没有与硬件表对齐的求值）、操作数含加载结果的、两侧形状不同的（有广播，对齐规则这里不猜）、结果类型与操作数不同的、`pim.matmul`（一个操作数是整块权重，折叠等于把矩阵乘搬进编译期）、带折入激活或池化的、`combineMode` 不是 `straightforward` 的、带收缩编码的、带逐槽定标的、带半旋转的、带 kantor 配置的、`datapath` 带缩放规格、kantor 块或分组反量化的、fpsu 是定点模式或改过粒度的、带 `table` 操作数的、查表寻址不是常规的（`mode` 不是 regular，或 `activationMode` 不是 0）、表选择与 `kind` 对不上的、带缩放或截断的。相位记录整份只记账，`forceConsecutive` 与 `reads` 不在排除之列；`fpsu` 的两种浮点模式只记账；`specialOperators = 4` 与 `kind = reciprocal` 是同一个选择，两者一致时不排除。
- 分析：操作数都有常量值属性，且属性不改变数值，才可折。无收益取舍：满足就折。A 路的对象在 `linear_kernel` 展开的循环边界里，B 路的对象要等 `pim-expand-phases` 把相位链展开后才出现。
- 转换：算出结果，替换成一个常量算子。
- 顺序依据：放在 `pim-expand-phases` 之后，否则 B 路只有一条 `pim.*`，没有可折的对象。

**公共子表达式消除 `-pim-cse`**

- 适用范围：同一基本块内算子名、属性、操作数、结果类型都相同的纯算子，单结果与多结果（如 `pim.split`）都算。不支持：跨基本块的、标了 `MemoryEffects` 的（`dma_load`、`kv_cache`、`wram_alloc`）、带区域的（`tt.reduce`、`scf.if` 的语义在区域体里，交给上游 `-cse`）。
- 分析：以 `(算子名, 属性, 操作数)` 为键。无收益取舍：同块内结构相同就合并。B 路的重复表达式来自 `pim-expand-phases` 展开 `softmax`、`quantize` 时生成的相同子表达式。整算子函数里结果没人用的张量算子是内核输出（与 DCE 同一判据），不参与合并。相位算子按数据流并成链（与 `-pim-verify-gml-contract` 同一判据），只整条合并到另一条等长的链上；只有前缀相同的两条链（如 Q、K 两路 rope）一相都不合并，否则两条链共用前缀，契约校验报相位重号。链的配对先于合并定下来：每条链单独与每条更早的链试合并，只有整条都对得上才记成一对，相位算子只能合并到配对链上。别的链只与它共享前缀，不影响它与整条相同的链配对，所以结果不随链的排列变化。
- 转换：重复算子的用户改接第一次出现的结果并当场删除，后面的算子按合并后的操作数比较，链式重复一遍合并完。
- 顺序依据：紧跟常量折叠，折叠产生的相同常量才能合并。

**算子融合 `-pim-fuse-activation`（改造）**

- 适用范围：从只认 `matmul`、`conv`、`eltwise` 改为凡是带 `activation` 属性位的算子后接单用户 `pim.lut`。不支持：`pim.lut` 有第二个用户、主算子没有 `activation` 属性位。
- 分析：沿定义-使用链找「主算子 → 唯一用户是 `pim.lut`」。无收益取舍：唯一用户是 `pim.lut` 就折。
- 转换：把 `pim.lut` 的 `kind` 写进主算子的 `activation` 属性，删除 `pim.lut`。这是现有 `FuseActivation.cpp:128` 的行为，只放宽主算子的判定。
- 顺序依据：B 路把它放在相位展开之前，而且只跑一次。展开后的相位链里也有 `pim.lut`，再融合一次会拆掉一条相位，契约校验因相位编号空档而失败。

**死代码消除 `-pim-dce`**

- 适用范围：结果全部没有用户、且没有内存副作用的纯算子，多轮删除。纯性按副作用接口判定，标了 `Pure` 的 `pim.tasklet_id`、`pim.dpu_id`、`pim.reshape` 等也算纯。整算子函数里（第一个参数是整块张量或片上缓冲，由图编译器发射）产出张量的 `pim.*` 算子保留，因为没人用的张量可能是内核输出或带外读取的中间值；内核输出本身按 `pim-lower-to-emitc` 的 `findOperatorOutputs` 认定，是「最后一批没人用的张量结果」，产出它的算子不论方言都保留（输出窄化这类非 `pim.*` 算子也在内）。A 路内核的头部参数是标量指针或指针张量，结果经 `tt.store` 或 `tt.return` 交出，没人用的纯算子不论方言、不论是否产出张量都删。`dma_store`、`kv_cache` 有内存副作用，所以留着，不是靠反向标可达保住的。
- 分析：结果全部没有用户、算子是纯的、且不在上条的保留范围内才删。不做从返回值反向的可达性标注。整算子函数的判据与 `pim-lower-to-emitc` 共用 `isOperatorLevelFunc`：第一个参数是元素类型不是指针的整块张量，或 `!pim.memdesc`。
- 默认通路的触发：A 路真实门控 linear 上，silu 链折进 `tt.dot` 后留下的常量 1 由它删除；不带激活的 linear 改动数为 0。B 路相位都是张量结果，按上条保留，真实展开后的 softmax、动态量化、matmul 改动数为 0 是预期，B 路这一格在合成 IR 上验证。这条例外已由需求方 2026-10-25 确认（见需求验收第 3 条）。
- 转换：删算子。
- 顺序依据：放在融合之后，融合留下的无用户 `pim.lut` 在这里清。

**分块 `-pim-tile-to-budget`（改造）**

- 适用范围：保留现有 `tt.dot` 路径；新增对 `scf.for` 嵌套、循环体含张量运算的通用路径。寻址覆盖两种形态：归纳变量经 `index_cast` 再 `splat`，以及真实 kernel 的 `i32` 归纳变量直接 `splat`，之后都可经过 `expand_dims`、`muli`、`broadcast`、`addi` 才到 `addptr`。不支持：循环边界不是编译期常量的、带 `iter_args` 的累加循环、认不出的寻址形态、各条地址链行距不同的、行距乘法落在加列偏移之后的（列偏移也被乘，重建会把行距多乘一次）、循环体含归约变形或控制流等拉长首维会改语义的算子（只放行地址链、加载存储与逐元素运算）、循环体内产生且不由归纳变量驱动的首维为 1 的张量（如地址不变的额外 `tt.load`，可能与循环内的写有别名）、偏移首维已大于 1 的（一轮处理多行，改首维会换掉这些行的地址）。循环外的不变张量（如 bias 常量）在新循环里广播到 tile 长；重建出的循环验证不过就整体撤回并给诊断。非 2 的幂边界只挡「只改步长」那条路径。重建路径不生成尾块：从预算内最大的 2 的幂 tile 向下回退，取第一个能整除跨度的；都不整除时保持原循环并给诊断。原步长不是 1、加载或存储带掩码、循环体含嵌套 `scf.for` 的，也保持原循环并给诊断。读 `tt.get_program_id` 的 grid 内核与 `tt.dot` 路径同一条约定：一个 program 只算一块，不改写循环。先算工作集再决定跳过：超预算却不改写的每条路径（只跑一轮、带 iter_args、grid 内核、非 2 的幂且不读归纳变量、有副作用且不读归纳变量）都给 `left this loop untiled` 诊断，预算内的保持静默。单轮放得下但没有更大 tile 时，诊断与单轮超预算分开写。
- 分析：见下方评估模型，与图层用同一个。通用循环的「原样」是整段循环：整段搬运量（单轮加载存储字节乘迭代次数）在预算内不切；超预算时按单轮工作集选预算内块数最少的 tile，所以单轮已能放下的循环也会合并成更少的块。
- 转换：`tt.dot` 路径维持 `LinearKernelRewriter`，写 `pim.tile-m/n/k` 三个属性；通用路径只有一个循环跨度，没有 m、n、k 之分，重建 `scf.for` 的边界与步长后只写 `pim.tile-m`。一个模块只有这一个属性，所以模块内已切的循环之后，tile 不同的循环保持原样并给诊断，属性与每条已切循环一致。超预算却切不了时给出诊断，与预算内不切区分开。
- B 路整算子矩阵乘：B 路没有循环，`pim.matmul` 按形状分块。评估模型与 `tt.dot` 路径相同（`searchTile`），另加两条：tile 必须整除 M、N 与组宽，输出块按两个 f32 累加器（总数与当前组）各 4 字节计；B 路没有 DMA，候选不检查 DMA 对齐。选中的 `[M, N, K]` 写在算子的 `tile` 属性上，模块同时写 `pim.tile-m/n/k` 与 `pim.tile-wram-bytes`。`-pim-lower-to-emitc` 见到 `tile` 时按块生成循环，组内按 K 块分段累加，同一输出元素的加法顺序与不分块相同，结果逐位一致。整除不了又超预算时保持原样并给诊断。模块头只有一组 tile 属性：只有模块里每个矩阵乘都切成同一个 tile、工作集字节数也相同时才写；有一个不同（含预算内不切的）就不写模块头并给诊断，各算子照用自己的 `tile`。矩阵乘写过 `pim.tile-m` 后，tile 不同的通用循环保持原样；矩阵乘切开却没写模块头时，通用循环也不切。B 路的 WRAM 用量写在 `pim.tile-wram-bytes`，布局回传读它作为 `wram_bytes_used`。
- 顺序依据：放最后，tile 大小要按前面各 Pass 化简后的循环与形状计算。A、B 两路都含这个 Pass。

## 五、改动汇总

### 5.1 文件级改动清单


| 文件路径                                        | 改动类型 | 改动内容简述                                                | 影响范围         | 优先级 |
| ----------------------------------------------- | -------- | ----------------------------------------------------------- | ---------------- | ------ |
| `graph/opt_passes.py`                           | 新增     | 五个通用 Pass 与`optimize_graph`                            | 执行路径、GML 路 | P0     |
| `runtime/compile.py`                            | 修改     | `compile_llama2` 在 `propagate_specs` 前调 `optimize_graph` | 执行路径数值     | P0     |
| `gml_bridge/export.py`                          | 修改     | `fuse_for_gml` 开头调 `optimize_graph`                      | GML 产物         | P0     |
| `contracts/op_semantics.py`                     | 修改     | 增加纯算子、逐元素两个性质集合                              | 融合与折叠判定   | P0     |
| `runtime/kernels.py`                            | 修改     | linear 镜像按 payload 的 activation 补 silu；A 路编译请求带上 activation | 执行路径数值     | P0     |
| `runtime/exec_plan_gen.py`                      | 修改     | payload 下发 activation；占位符经 `_as_torch` 转 torch | 执行路径         | P0     |
| `opcompiler_bridge/kernel_src.py`               | 修改     | linear 内核加 `ACTIVATION` 常量，累加器上算 silu          | A 路内核         | P0     |
| `graph/opt_passes.py`（融合）                   | 修改     | 新增 `fuse_linear_activation`，切分传播后把 linear 尾部激活折进 linear | 执行路径         | P0     |
| `opcompiler_bridge/pipeline.py`                 | 新增     | `a_path_passes`、`b_path_passes`                            | 四处调用点       | P0     |
| `opcompiler_bridge/driver.py`                   | 修改     | 两路改用 pipeline                                           | A、B 两路产物    | P0     |
| `opcompiler_bridge/phase_source.py`             | 修改     | `PASS_PIPELINE` 改调用                                      | B 路相位抽取     | P0     |
| `genesim_bridge/flagtree_driver.py`             | 修改     | 两路改用 pipeline；抓 A 路内核前把 autotune 候选限制在 WRAM 预算内，口径与 `fitsBudget` 相同 | 成本抽取的 IR    | P0     |
| FlagTree`Transforms/Passes.td`                  | 修改     | 声明三个新 Pass                                             | triton-opt       | P0     |
| FlagTree`Transforms/ConstantFold.cpp`           | 新增     | `-pim-constant-fold`                                        | A、B 两路        | P0     |
| FlagTree`Transforms/PimCSE.cpp`                 | 新增     | `-pim-cse`                                                  | A、B 两路        | P0     |
| FlagTree`Transforms/PimDCE.cpp`                 | 新增     | `-pim-dce`                                                  | A、B 两路        | P0     |
| FlagTree`Transforms/PimPure.h`                  | 新增     | 三个 Pass 与降级共用的纯性、整算子函数判定                  | A、B 两路        | P0     |
| FlagTree`Transforms/FuseActivation.cpp`         | 修改     | 判定改为按性质；新增 tt.dot + silu 五连的 A 路形态          | A、B 两路融合    | P0     |
| FlagTree`Transforms/TileToBudget.cpp`           | 修改     | 无`tt.dot` 时按形状切整算子矩阵乘，再走通用循环分块         | A、B 两路        | P0     |
| FlagTree`Transforms/LowerPIMToEmitC.cpp`        | 修改     | 带`tile` 的 `pim.matmul` 生成块循环；融合可折的激活（relu_x 除外）都有 C 实现；整算子函数的判据改用 `PimPure.h` 的 `isOperatorLevelFunc` | B 路 C 产物      | P0     |
| `genesim_bridge/cost_extractor.py`              | 修改     | pimir 级 GEMM 用本仓 linear 内核经 `compile_op` 编，激活随请求下发，由 `-pim-fuse-activation` 折进 tt.dot | GeneSim 成本     | P0     |
| FlagTree`IR/PIMOps.td`                          | 修改     | `pim.matmul` 声明可选 `tile` 属性                           | B 路             | P0     |
| FlagTree`Transforms/CMakeLists.txt`             | 修改     | 加入三个 cpp                                                | 构建             | P0     |
| FlagTree`python/src/passes.cc`                  | 修改     | 三个 Pass 的 Python 包装                                    | 进程内调用       | P0     |
| FlagTree`python/triton/backends/pim_sidecar.py` | 修改     | 按同一顺序插入五类 Pass                                     | GPU sidecar      | P0     |
| `opcompiler_bridge/pipeline.py`（A 路）         | 修改     | 五类之前加 `canonicalize`、`cse`、`loop-invariant-code-motion` | A 路产物         | P0     |

### 5.2 新增/修改的数据结构

不新增 `node.meta` 键。图层五类优化都通过改图生效，因为执行路径不读 meta。`OpCompileRequest` 不增字段，tile 尺寸仍由 `-pim-tile-to-budget` 在算子层决定。

### 5.3 接口变更

`compile_llama2`、`fuse_for_gml` 签名不变，内部多一步。`a_path_passes`、`b_path_passes` 为新接口，四个调用点从字面量改为调用。三个新 Pass 无选项，由 TableGen 收进 `registerTritonPIMPasses`。

## 六、实施计划

### 6.1 阶段划分与依赖

1. `pipeline.py` 与 `optimize_graph` 骨架，Pass 先空实现，接通全部调用点。
2. 图层五个 Pass，每个带合成图测试。
3. 算子层三个新 Pass 与两处改造，每个带 lit 测试。
4. A、B 两路接入与跨层数值对拍。
   后一阶段依赖前一阶段测试全绿。

### 6.2 每阶段的回归门槛

每阶段结束跑 `python -m pytest tests/ -q -k "not llama2_7b"`。FlagTree 改动另跑 `test/Dialect/TritonPIM` 的 lit。全部通过后跑 `python -m pytest tests/ -q -k "llama2_7b"`。

## 七、验证与测试方案

### 7.1 单元测试清单

每个 Pass 的用例对应「四·附」里的适用范围：一条落在支持范围内断言变换发生，一条落在不支持范围内断言不动。图层仿 `tests/test_fuse.py:25` 手写 FX 图。


| 用例                               | 操作                         | 预期                             |
| ---------------------------------- | ---------------------------- | -------------------------------- |
| `test_fold_constant_chain`         | 图含常量输入的`add`          | 该节点替换为常量，输出值等于手算 |
| `test_fold_skips_dynamic`          | 同型`add` 输入是占位符       | 节点保留                         |
| `test_cse_merges_identical`        | 两处`mul` 参数相同           | 只剩一处                         |
| `test_cse_keeps_distinct`          | 两处`mul` 参数不同           | 两处都留                         |
| `test_fuse_unlisted_elementwise`   | 白名单外的逐元素算子单消费者 | 合并为一个节点                   |
| `test_fuse_blocked_by_second_user` | 同算子有两个用户             | 不合并                           |
| `test_dce_drops_unused`            | 一段结果无人使用             | 该段被删                         |
| `test_dce_keeps_output`            | 同型算子连到输出             | 保留                             |
| `test_tile_splits_oversized`       | 形状超过 WRAM 预算           | 拆成多个子节点，每个在预算内     |

算子层每个 Pass 两个 lit，输入分别是 TTIR 片段和展开后的 `pim.*` 片段：

```
// RUN: triton-opt %s -pim-expand-phases -pim-cse | FileCheck %s
```

### 7.2 集成验证

A 路用 `tests/test_opt_passes_ir.py` 的 `_opt` 把合成 TTIR 跑完 `a_path_passes()`：真实 kernel 上断言分块与融合确实改动 IR；折叠与 CSE 由各自在合成 IR 上的用例验证（真实 kernel 上没有作用对象，见 7.3），DCE 在合成 IR 与真实门控 linear 上都触发，不带激活的 kernel 上改动数为 0。B 路同样用 `_opt` 跑展开后的 `pim.*`：重复子表达式合并与无用户标量删除由合成 IR 的用例验证；真实展开产物没有可作用对象，相位按内核输出保留，见 7.3 与需求验收第 3 条。数值对拍沿用 `np.testing.assert_allclose`，容差与现有用例一致。

### 7.3 验收标准

- `python -m pytest tests/ -q -k "not llama2_7b"` 全部通过，其后 `llama2_7b` 通过。
- 下表每格一个用例，共 15 格，缺一格不通过。每格同时含结构断言和数值对拍，且用的算子不在现有白名单内。


| Pass     | 图层                         | A 路（TTIR）                 | B 路（展开后的 pim.*）         |
| -------- | ---------------------------- | ---------------------------- | ------------------------------ |
| 常量折叠 | 常量`add` 被替换             | 循环边界的`arith` 常量被折   | 展开后的常量`pim.eltwise` 被折 |
| CSE      | 两处相同`mul` 合并           | 循环内重复`arith` 合并       | 合成相位上的重复子表达式合并   |
| 融合     | 表外逐元素算子合并成一个节点 | `tt.dot` 后的 silu 五连折进 dot 的 activation | 表外主算子后的`pim.lut` 折进   |
| DCE      | 无用户链被删                 | 无用户张量`arith` 与地址链被删 | 展开后相位链里的无用户标量被删，相位一条不少 |
| 分块     | 超预算`linear` 拆成子节点    | 非`tt.dot` 的循环被改写 tile | 超预算`pim.matmul` 写 tile 并生成块循环 |

常量折叠三格在真实 IR 上都没有可折对象，只在合成 IR 上验证 Pass 本身。
2026-10-20 实测：一层 Llama 导出图的可折叠节点输入全是激活或模型参数；
A 路 linear kernel 在 16、64、128、256 四个尺寸上，两个操作数都是常量的
`arith` 运算都是 0 条；B 路 softmax、rope、matmul、eltwise、lut 展开后
`arith.constant` 都是 0 条。默认通路上改动数为 0 是预期。

图层分块一格在真实模型上也是 0。2026-10-23 实测默认 65536 字节下，7B 四种投影要切
2048 到 32000 块，超过 `MAX_TILE_BLOCKS = 512`，留给 A 路 `tt.dot` 分块；回归用小模型
（hidden 64、intermediate 128、vocab 128）的投影都在预算内。这一格在合成图和 4096 字节
预算的小模型上验证，`test_tile_leaves_real_projections_whole` 与
`test_tile_leaves_small_model_projections_whole` 钉住默认通路改动数为 0。

另有三格在真实 IR 上没有作用对象，覆盖关系如下。

- A 路融合：`convert-triton-to-pim` 保留 `tt.dot`，它没有 `activation` 属性位，
  融合因此不是按属性位，而是按 Triton 展开出的激活链结构识别（见 4.11）。
  触发条件是 linear 编译请求带 activation——由执行路径的 `fuse_linear_activation`
  在切分传播之后写入 `FUSED_TAIL_META_KEY`。默认通路上 `compile_llama2` 的内核
  占用探针、执行计划的编译内核和 GeneSim 旁支的分块测量都按这个标记下发
  activation，所以不开编译内核时，真实门控投影每层也触发一次。
- B 路融合：发射器按图上的融合标记把激活发成主算子唯一用户的独立 `pim.lut`，
  不预写 activation，`-pim-fuse-activation` 把它折回主算子。真实模型每层门控
  投影触发一次，两层口径按「发射 LUT 数 = 折入 activation 数」核对。
- B 路 CSE：`pim-expand-phases` 对 softmax、rope、动态量化展开出的相位链里
  没有结构相同的算子，实测展开后再跑 `-pim-cse` 输出与输入逐字节相同。
  2026-10-25 复测一层 Llama 的发射器真实产物（prefill 20 个函数、decode 15 个函数），
  同一函数内也没有结构相同的算子。
  这一格由合成 IR 覆盖，验证 Pass 本身；真实重复结构出现时同一 Pass 生效。
  合成用例含整算子函数形态，降到 C 后与不跑 CSE 的结果逐位对拍。这一例外已由需求方
  2026-10-25 确认（见需求验收第 3 条）。图层 CSE 在 GML 路为 0，同条确认。
- A 路通用循环分块：2026-10-23 实测 FlagGems 缓存里 179 个无点积真实内核，
  177 个没有 `scf.for`，剩下 2 个循环边界是运行期值，通用路径在真实内核上
  没有作用对象，只在合成 IR 上验证。A 路真实触发的是 `tt.dot` 路径。

- 每格另有一条反例：动态输入不折、多用户不融合、有内存副作用的不删、预算内的不拆。
- 触发断言：前四个 Pass 只要输入落在适用范围内，改动数必须大于 0。分块另断言选出的 tile 与评估模型手算的结果一致，预算内的输入改动数为 0。
- `tests/test_fuse.py`、`test_fuse_pim.py`、`test_fuse_rope.py`、`test_fusion_contract.py`、`test_opcompiler_linear.py` 原断言保持通过。

## 八、对需求文档的修正记录

1. 需求第七节第 3 条把 `add_canonicalizer`、`add_cse` 列为可复用基础。实测 `PIMOps.td` 无 `hasFolder`，且 `matmul`、`eltwise`、`lut` 未标 `Pure`，这三个上游 Pass 对 PIM 算子不生效。算子层改为手写，上游 Pass 只作写法参考。
2. 需求验收第 2 条要求 20 组。实测 A、B 两路就是算子层的全部入口，没有独立于两路之外的算子层，收敛为 15 组。
3. 需求把融合描述为折进主算子并写 meta。实测执行路径 `build_execution_plan` 不读 meta，只写标记不会改变执行。图层融合改为替换成真正的合并节点。

## 九、本设计自身的修正记录

1. 初稿把图层融合设计成写 `folded_into` 标记。错在没核对执行路径的消费点：`exec_plan_gen.py:435` 只读 `node.target`。改为替换节点。
2. 初稿把图层分块设计成写 `tile` 字段下发。同样的原因，只写字段不改变执行。改为在图上拆节点。
3. 初稿把算子层三个新 Pass 放在 B 路相位展开之前。错在没核对 B 路输入：`oplevel_kernel.py` 每个函数体只有一条 `pim.*`，展开前没有可优化的对象。改为放在 `pim-expand-phases` 之后。
5. 通用循环分块原先写「不支持非 2 的幂边界」，并只认 `index` 归纳变量经 `index_cast` 再 `splat` 的寻址。实测真实 linear kernel 的循环边界是 `i32`，归纳变量直接 `splat`，而且 K、N 常取 11008、32000 这种非 2 的幂。改为两种寻址都认，非 2 的幂只挡「只改步长」那条路径。重建路径后来改为不生成尾块，按整除回退选 tile，见四·附 B。

4. 初稿把常量折叠写成在三条默认通路上都能有效触发。实测真实输入里没有可折对象：图层导出图的可折叠节点输入全是激活或模型参数；A 路 linear kernel 的 `arith` 运算都有一个操作数来自 `tt.splat`、`tt.make_range` 或 `tt.broadcast`；B 路展开后的相位操作数全是函数参数或上游相位的结果，`arith.constant` 一条都没有。常量折叠的验收改为在合成 IR 上验证 Pass 本身，默认通路上改动数为 0 是预期，不是漏触发。

## 十、待确认事项

1. 默认通路生效（已定）。常量折叠、CSE、融合、DCE 在执行路径、GML 路、A 路、B 路都默认跑，不需要配置开启。分块有一处例外：GML 按整块算子编码 contraction 与缓冲名，切成子块会改掉这些字段，所以 GML 路关闭分块，在执行路径、A 路、B 路上跑。`optimize_graph` 的 `tile` 参数就是这处例外，不是一个通用开关。
2. 图层融合替换出的合并算子需要在 `runtime/kernels.py` 有实算实现，否则执行路径会缺内核。本设计按新增一个逐元素合并内核处理，需确认可接受。
3. B 路的常量折叠、CSE、DCE 作用在 `pim-expand-phases` 展开之后的 IR 上，展开前的单条 `pim.*` 不在其处理范围内。需确认这符合「覆盖 B 路」的预期。
