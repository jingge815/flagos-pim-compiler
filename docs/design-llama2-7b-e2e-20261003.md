# 技术设计文档：Llama2 7B 推理算子全量走通存算一体编译链路
> 文档编号：design-llama2-7b-e2e-20261003
> 创建日期：2026-10-03
> 关联需求文档：request-llama2-7b-e2e-20261003
> 关联项目：flagos-pim-compiler、FlagTree、genesim

## 一、设计概述

### 1.1 需求背景回顾

Llama2 7B 推理能与 HF 逐 token 对齐，但设备内核遇到 2 的幂、类型、尺寸限制就退回主机，主机路径性能差，而断言在整段退回时仍通过。需求是全部算子在设备上走通三条路径并完成仿真。限制要打开，超容量就改分块算法，不许退回主机。

### 1.2 设计目标

1. 去掉 2 的幂、类型、尺寸三类限制，推理中每个算子都走编译内核，兜底次数为 0。
2. 超过 WRAM 或 MRAM 时由分块循环拆开，不拒绝编译。
3. 三路径产物按算子汇总，缺一即失败。
4. genesim 仿真能跑完，跑不通就改 genesim。

### 1.3 设计范围

- 包含范围：放宽支持判断、分块循环的尾块处理、FlagTree 分块搜索去掉 2 的幂前提、退回计数与断言、设备归属核对、三路径汇总、仿真入口。
- 不包含范围：把 `expand`、`repeat`、`contiguous`、`squeeze` 下放到设备，广播倍数在命令里没有存放位置；把 bfloat16 加入 convert 的类型表（理由见 4.5）。

## 二、现状基线

### 2.1 推理怎么跑起来

`compile_llama2`（`runtime/compile.py:194`）是入口：78 行导出，83 行打设备标记，图存 `prefill_gm`、`decode_gm`（30 行），`register_all`（`runtime/kernels.py:1268`）注册内核后逐条执行。

### 2.2 设备归属没有漏标

`graph/partition.py:26` 的 `HOST_ONLY` 是黑名单，默认下设备。留主机的只有不动张量的脚手架、位置下标 `arange`、广播视图 `expand`、`repeat`、`contiguous`、`squeeze`。注意力已拆成设备命令（`runtime/kernels.py:1045`）。问题不在分区，在执行时的退回。

### 2.3 三道限制把计算退回主机

`runtime/kernels.py:167` 的 `_compiled_linear_supports` 要求 m、k、n 都是 2 的幂且 k≥16，否则 182 行直接退回纯 numpy。实测 8 DPU 下 `q_proj`、`o_proj` 通过，`gate_proj`、`up_proj` 的 N=1376、`down_proj` 的 K=1376、`lm_head` 的 N=32000 全被拒。

这道判断比编译器更严。`opcompiler_bridge/driver.py:163` 已放宽为「M 必须是 2 的幂，K、N 只要有能整除的 2 的幂分块」，我调用 `_kernel_launcher`，上面三个被拒的形状它都接受。但 M 的 2 的幂限制还在，来自 `kernel_src.py:30` 的 `tl.arange(0, M)` 一次铺满整维。

类型和尺寸的限制同样是提前退回：eltwise:269 要求同形 fp16，matmul:634 要求二维 fp16，convert:1150 只收 fp16、fp32、int8。

### 2.4 分块算法本身不接受任意尺寸

FlagTree 的 `TileToBudget.cpp` 把 2 的幂写死了两处：`validateDot`（706 行）要求每个维都是 2 的幂，否则报错；`searchTile`（838 行）只在 2 的幂格点上找分块。

循环也依赖整除：`buildOuterDim`（544 行）用分块尺寸当步长走到整维，没有尾块，所以维度必须能被分块整除，这也是 `kernel_src.py:41` 的 `_pick` 反复折半的原因。允许任意尺寸后，最后一块会越界，必须单独处理。

### 2.5 退回看不见

`tests/test_opcompiler_e2e_llama2_7b.py:112` 在不支持时走 numpy 且不计数，206 行只断言至少命中一次，MLP 全走主机时仍是绿的。其余退回点 softmax:530、mask:696、transpose:806、reshape:843、concat:883、normalize:409、rope:456、lut:358、split_heads:907、kv_cache:1002 都是内核没编出来才退回。gather:731 无退回，失败直接抛。

### 2.6 三条路径对不齐

numpy 对拍在 `tests/test_opcompiler_ops.py`，但 5 个算子在 `test_runtime_compiled_coverage.py:154` 只是正则扫源码。genesim 的 `MNEMONICS` 有 14 项，不含 `linear`（并入 `matmul`）和 `convert`。gml 按字段族断言，不按算子名。仿真在 `scripts/run_full_pipeline.py:325`，故意留在 pytest 之外。

## 三、总体设计

### 3.1 总体设计概述

目标是全程走编译内核，超容量靠分块而不是退回。三处要改：`runtime/kernels.py` 去掉按形状提前退回；`kernel_src.py` 把 M 维也改成分块循环；`TileToBudget.cpp` 放宽 `validateDot`、让 `searchTile` 接受任意尺寸，并给循环加尾块。

```
任意形状进入编译
  ├─ 旧：整维是 2 的幂？否则退回主机          （删除）
  └─ 新：一律编译
        ├─ 容量内：一次算完
        └─ 超 WRAM：searchTile 找分块
              ├─ 整除：满块循环
              └─ 不整除：满块循环 + 一块尾块   （新增）
超 MRAM：本轮不处理。硬件每台 DPU 8GB，实测峰值驻留约 206MB（利用率 38.5%，
        溢出为 0），没有超容量的形状，分批驻留没有判据可覆盖。
```

退回路径保留为编译器真失败时的可见信号，但测试断言推理中它的次数为 0。一旦出现退回，就是还有限制没打开，继续改，而不是接受它。

### 3.2 与现有逻辑的衔接

判断放宽后所有形状都会调用 `compile_op`，所以 `driver.py:168` 那个 M 必须是 2 的幂的报错要先去掉，否则放宽后编译直接失败。顺序是先改内核和分块，再放开判断。

计数放在 `runtime/kernels.py` 模块内。退回判断在函数内部，monkeypatch 替换整个函数会把判断一起换掉，统计不到。

### 3.3 模块划分与职责边界

- `kernel_src.py` 负责循环形状，含 M 维分块和尾块掩码。
- `TileToBudget.cpp` 负责在容量内选分块并生成尾块，不决定退回。
- `runtime/kernels.py` 负责不再提前退回并计数。`driver.py` 去掉冲突的校验。
- genesim 在仿真跑不通，或 sidecar 声明的放置未被遵守导致指标失真时改。

## 四、模块详细设计

### 4.1 M 维改为分块循环（对应需求点：P0-3｜现有代码位置：`opcompiler_bridge/kernel_src.py:30`｜改动类型：修改）

**功能点**：linear 内核对 M 维也分块遍历，使任意 M 都能编译。

**原因**：`tl.arange(0, M)` 一次铺满整维，Triton 要求这个范围是 2 的幂，于是 M 被限制死。

**代码对比**：

```python
# 原有
offs_m = tl.arange(0, M)
for n0 in range(0, N, BLOCK_N):
    ...

# 目标
for m0 in range(0, M, BLOCK_M):
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    for n0 in range(0, N, BLOCK_N):
        ...
        tl.load(x_ptr + x_off, mask=mask_m[:, None])
```

**解决思路**：M 取固定的 2 的幂分块，最后一块用掩码丢掉越界元素。K、N 的循环保持原样，尾块同样加掩码，于是三个维都不再需要被分块整除。

### 4.2 分块搜索去掉 2 的幂前提（对应需求点：P0-3｜现有代码位置：`TileToBudget.cpp:838`｜改动类型：修改）

**功能点**：让 `searchTile` 能给任意尺寸找到不超 WRAM 的分块。

**原因**：它只在 2 的幂格点上搜索，且 `validateDot`（706 行）先把非 2 的幂的维度拒掉，任意尺寸进不来。

**代码对比**：

```cpp
// 原有
static SmallVector<int64_t, 8> powerOfTwoDivisorsDesc(int64_t full) {
  for (int64_t v = full; v >= 1; v /= 2)
    vals.push_back(v);
}

// 目标
static SmallVector<int64_t, 8> candidateTilesDesc(int64_t full) {
  for (int64_t v = std::min(full, maxTile); v >= 1; v /= 2)
    vals.push_back(v);
  vals.push_back(full);   // 整维本身也是候选
}
```

**解决思路**：候选改为「不超过上限的 2 的幂，外加整维本身」。2 的幂候选保证容量内总有得选，整维候选保住不超容量时一次算完。`validateDot` 删掉 2 的幂那行检查，只保留形状为正、K 一致。

### 4.3 循环补尾块（对应需求点：P0-3｜现有代码位置：`TileToBudget.cpp:544`｜改动类型：修改）

**功能点**：维度不被分块整除时，最后一块单独算，不越界。

**原因**：`buildOuterDim` 用分块当步长直接走到整维，前提是整除，否则最后一趟读出边界。

**代码对比**：

```cpp
// 原有
Value ub = constI32(b, loc, dim.full);
Value step = constI32(b, loc, dim.tile);

// 目标
int64_t tail = dim.full % dim.tile;
int64_t mainUb = dim.full - tail;
// 主循环上界取 mainUb；tail > 0 时再补一段长度为 tail 的块，加载带掩码
```

**解决思路**：主循环只走满块，余数单独成一块，该块的加载和存储加掩码。余数为 0 时不生成这段，输出与现在一致。

### 4.4 去掉提前退回（对应需求点：P0-3｜现有代码位置：`runtime/kernels.py:167`｜改动类型：修改）

**功能点**：`_compiled_linear_supports` 不再按形状拒绝，所有形状都进编译。

**原因**：它比 `driver.py` 更严，把 MLP 和 `lm_head` 挡在编译器外面退回主机。

**代码对比**：

```python
# 原有
return k >= 16 and _is_pow2(m) and _is_pow2(k) and _is_pow2(n)

# 目标
def _compiled_linear_supports(arg_shapes, dtype="float32") -> bool:
    _m, k = flatten_leading_dims(arg_shapes[0])
    return dtype in ("float16", "float32") and k >= 16
```

**解决思路**：形状不再是拒绝理由，超容量交给 4.2、4.3 的分块。dtype 仍限定 fp16、fp32，因为这是内核的累加类型；其他类型走 convert 后再进来。K 的下限保留：Triton 的 `tl.dot` 要求 K 不小于 16，低于它编译器直接报错，这类形状退回镜像并计入 fallback，判据看得见。`driver.py:168` 的 M 校验同步删除，K 的下限与这里保持一致。

### 4.5 类型与尺寸限制打开（对应需求点：P0-3｜现有代码位置：`runtime/kernels.py:269`｜改动类型：修改）

**功能点**：eltwise 的广播、matmul 的非二维、convert 的更多类型都进编译内核。

**原因**：这三处在形状或类型不符时直接退回 numpy，和 linear 一样落在主机上。

**代码对比**：

```python
# 原有（eltwise）
if isinstance(y, np.ndarray) and x.shape == y.shape and x.dtype == np.float16:
    fn = _compiled_eltwise(kind, tuple(x.shape))

# 目标
x, y = _align_broadcast(x, y)        # 先广播到同形
fn = _compiled_eltwise(kind, tuple(x.shape))
```

**解决思路**：广播在进内核前展开成同形，内核仍按同形计算。matmul 只覆盖三维且两侧都是 fp16 的 bmm，逐批拆成二维再折回；三维 fp32、四维、两侧类型不同的形状退回镜像并计入 fallback。convert 的类型表加上 int32 与 int64，内核按位宽生成。int64 是 attention 掩码的位置索引类型，不进类型表就会在每个 decode step 退回主机；坐标值域远小于 2^31，f32 存得下。

**bfloat16 划出本轮范围**：本轮加 int32 与 int64，bf16 不加。理由是 llama2-7b-hf 的权重与激活都是 float16，整条推理链不产生 bfloat16，加进来没有判据能覆盖；而要让 bf16 真的编得过，降级侧（FlagTree 的 EmitC）得补一套 bf16 的位操作转换，属于另一个仓库的改动。本轮的开销花在 llama2 全算子的三路径与端到端仿真上。

### 4.6 退回计数与断言（对应需求点：P0-3｜现有代码位置：`runtime/kernels.py:182`｜改动类型：新增）

**功能点**：14 处退回点各记命中或退回，推理后断言退回为 0。

**原因**：退回无返回值差异，不记录就无法证明没有退回主机。

**代码对比**：

```python
# 原有
assert len(stats) > 0, "没有任何调用走到编译产物路径"

# 目标
counts = km.route_counts()
bad = [op for (op, route), n in counts.items() if route == "fallback" and n]
assert not bad, f"这些算子退回了主机：{bad}"
```

**解决思路**：`record_route(op, route)` 放模块顶层，`hit` 记在调用编译内核前，`fallback` 记在调用镜像前。断言不再分档，所有算子一视同仁。token 对齐断言保留。

记录点按内核实际落点计，不止原先列的 14 处：`linear`、`eltwise`、`matmul`、`softmax`、`reduce`、`lut`、`reshape`、`transpose` 记命中或退回，`pow` 指数不为 2 时记 `pow` 的退回，`slice` 步长不为 1 时记 `slice` 的退回，`relu`、`sigmoid`、`exp`、`sqrt`、`reciprocal` 没有编译形态，各记自己的退回。`rsqrt` 与指数为 2 的 `pow`、取负都复用已有入口，分别记在 `lut` 与 `eltwise` 上。

### 4.9 五个算子改走已有编译入口（对应需求点：P0-3｜现有代码位置：`runtime/kernels.py`｜改动类型：修改）

**功能点**：`mean.dim`、`pow`、`rsqrt`、`unsqueeze`、`transpose.int` 从 numpy 镜像改走编译内核。

**原因**：这五个算子在 llama 的推理计划里占了二十多条命令，原先直接跑 numpy，既不计命中也不计退回，「兜底次数为 0」看不见它们。

**解决思路**：不新增编译器能力，复用已有入口。`mean.dim` 走新增的 `reduce` 入口，沿单轴求和再除以轴长，求和在 fp32 上做完再存，4096 宽的平方和不会溢出成 inf。`pow` 指数为 2 时就是自己乘自己，走 `eltwise` 的乘。`rsqrt` 走 `lut` 的 rsqrt 种类。`unsqueeze` 插轴不改元素，走 `reshape`。`transpose.int` 把两个轴号拼成全轴序，走 `transpose`。

`slice` 步长为 1 时也能走编译：被切的轴先转到轴首，目标区间就成了连续前缀，`reshape` 按元素数拷这段前缀，再转回原轴序。步长不是 1 的切片没有对应形态，退回镜像并计入退回。`neg` 走 `eltwise` 的减法，`0 - x` 就是取负。

### 4.7 设备归属核对（对应需求点：P0-1｜现有代码位置：`runtime/compile.py:30`｜改动类型：新增测试）

**功能点**：遍历导出图，断言设备标记与黑名单一致。

**原因**：黑名单有无漏项没有测试守着，算子被静默留在主机会直接让 P0-3 失真。

**代码对比**：

```python
# 原有
# 无此断言

# 目标
for node in compiled.prefill_gm.graph.nodes:
    if node.op != "call_function":
        continue
    assert (node.meta["device"] == "host") == _is_host_only(node.target)
```

**解决思路**：复用 `_is_host_only`，不重写规则。`decode_gm` 同样遍历一次。

### 4.8 三路径汇总与仿真入口（对应需求点：P0-2、P1｜现有代码位置：`tests/test_opcompiler_ops.py:77`｜改动类型：新增测试）

**功能点**：按算子汇总三路径产物，缺一即失败；另加仿真慢速测试。

**原因**：三套测试各用各的命名，缺产物看不出来；仿真是手动脚本，回归跑不到。

**代码对比**：

```python
# 原有
# 三份测试各自断言；仿真只在 scripts/run_full_pipeline.py 里手动跑

# 目标
def test_three_paths_cover_every_compiled_op():
    missing = [op for op in COMPILED_OPS if op not in covered()]
    assert not missing, missing
```

**解决思路**：genesim 侧 `linear` 由 `matmul` 覆盖、`convert` 不是 mnemonic，这两项在汇总里注明归属而不是算缺失。仿真测试调用 `step_e_simulate`（`scripts/run_full_pipeline.py:325`），断言产出 `summary.json`，用 `slow` 标记排除出快速回归。仿真失败时改 genesim，不改断言去迁就。

## 五、改动汇总

### 5.1 文件级改动清单

| 文件路径 | 改动类型 | 改动内容 | 影响范围 | 优先级 |
| --- | --- | --- | --- | --- |
| `opcompiler_bridge/kernel_src.py` | 修改 | M 维分块循环，三维都加尾块掩码 | linear 内核 | P0 |
| `opcompiler_bridge/driver.py` | 修改 | 删除 M 必须是 2 的幂的校验 | 编译入口 | P0 |
| FlagTree `TileToBudget.cpp` | 修改 | 放宽 `validateDot`，`searchTile` 支持任意尺寸，循环补尾块 | 全部分块 | P0 |
| `runtime/kernels.py` | 修改 | 去掉按形状退回，逐元素支持广播与 fp32，五个算子改走编译，退回计数覆盖到每个内核 | 全部设备内核 | P0 |
| `contracts/op_semantics.py` | 修改 | 新增 `reduce` 条目，承载 `aten.mean.dim` | 算子语义表 | P0 |
| `opcompiler_bridge/oplevel_kernel.py` | 修改 | 新增 `reduce_kernel`，沿单轴求均值 | 算子级内核 | P0 |
| `opcompiler_bridge/driver.py` | 修改 | 新增 `reduce` 分支，`group_size` 复用为归约轴 | 编译入口 | P0 |
| `tests/test_opcompiler_e2e_llama2_7b.py` | 修改 | 断言退回次数为 0 | llama2 端到端 | P0 |
| `tests/test_partition.py` | 修改 | 补 decode 图的设备标记断言 | 图分区 | P0 |
| `tests/test_three_path_coverage.py` | 新增 | 三路径汇总 | 三个桥接 | P0 |
| `tests/test_genesim_simulation.py` | 新增 | 慢速仿真冒烟 | genesim | P1 |
| `scripts/run_full_pipeline.py` | 修改 | 仿真前核对 sidecar 哈希与 trace 来源；`--max-requests` 收短仿真，默认跟随配置 | 仿真入口 | P1 |
| genesim | 视情况 | 仿真跑不通，或 sidecar 声明的放置未被遵守导致指标失真时改 | 仿真器 | P1 |

### 5.2 新增数据结构

`_KERNEL_ROUTE: dict[tuple[str, str], int]`，键是 `(算子名, "hit" 或 "fallback")`，默认空，等价于改动前不统计，不落盘。

### 5.3 接口变更

`_compiled_linear_supports` 签名不变，但不再因形状返回 False，唯一调用方 `compiled_linear_kernel` 对所有 fp16、fp32 形状都走编译。`pick_blocks` 的整除要求随之消失，调用方 `driver.py` 不再收到它的报错。`register_all`、`compile_llama2`、`CompiledModel` 签名不变。

## 六、实施计划

### 6.1 阶段划分与依赖

1. 先改 `kernel_src.py` 和 `TileToBudget.cpp`，让任意尺寸能编译，旧形状行为不变。
2. 再放宽支持判断和三处类型尺寸判断，加计数，跑层级 2。
3. 加归属测试和退回断言，跑层级 3，出现退回就回到对应仓库改。
4. 导出 gml 填 `GML_UNMAPPED`，启用汇总测试。
5. 跑仿真，失败就改 genesim。

顺序不能反：先放宽再补分块，超容量的形状会在编译期直接报错。

### 6.2 每阶段的回归门槛

阶段 1 后 `-k "not llama2_7b"` 无新增失败，且原有 2 的幂形状的编译产物与改动前一致。阶段 3 后 `-k "llama2_7b"` 通过，token 与 HF 一致，退回次数为 0。仿真不计入前两级。

## 七、验证与测试方案

### 7.1 单元测试清单

`test_arbitrary_shape_compiles`：用 N=1376、K=1376、N=32000 三个形状编译 linear，断言都产出内核并与 numpy 对齐。退回计数由端到端断言覆盖，这条用例直接调 `compile_op`，不经过记录点。

`test_odd_m_compiles`：M=3 的 linear 编译通过并与 numpy 逐元素一致。这是 4.1 的直接判据，M=3 不是 2 的幂。

`test_tail_block_matches_numpy`：构造不被分块整除的维度，比对编译内核与 numpy 的最后一块，确认尾块掩码没有算多也没有算少。

`test_fallback_is_recorded`：拿掉 PIM pass 时断言退回计数增加，确认计数本身有效。

`test_placement_matches_blacklist`：见 4.7。`test_three_paths_cover_every_compiled_op`：见 4.8。

### 7.2 集成验证

端到端测试保留 token、文本对齐和误差小于 0.05，加退回断言。`num_stages` 取 1、4、8，三种都要退回为 0。

### 7.3 验收标准

层级 1：15 个算子三路径都有产物，误差小于 0.05。层级 2 无新增失败。层级 3：token 与 HF 一致，全部算子退回次数为 0。P1：仿真产出 `summary.json` 和输出 token。

## 八、对需求文档的修正记录

1. P0-2「15 个算子各走通三份产物」里，genesim 的 `MNEMONICS` 不含 `linear` 和 `convert`。汇总时按归属注明，不把它们算成缺失，也不要求 genesim 新增这两个名字。

## 九、本设计自身的修正记录

1. 前两版把 2 的幂当成要保留的限制，一版据此缩小验收，一版只改一道过时判断，都把任务打折了。本版把三类限制全部打开，并补上尾块处理。
2. 前一版写「FlagTree 不改」。复核 `TileToBudget.cpp:706` 和 `:838` 后确认两处都写死了 2 的幂，本版将该文件列为 P0。
3. 4.4 的目标代码原先写成「只看 dtype」，实施时确认 Triton 的 `tl.dot` 对 K 有不小于 16 的硬约束，低于它编译器直接报错。目标代码改为保留 `k >= 16`，这类形状退回镜像并计入 fallback。
4. 4.5 原先只写加 int32。实施时发现 attention 掩码的位置索引是 int64，每个 decode step 都有一批退回，类型表同时加了 int64。
5. 3.1 的流程图原先把「超 MRAM 分批驻留」标为新增。实测硬件每台 DPU 8GB、峰值驻留约 206MB（利用率 38.5%，溢出为 0），没有超容量的形状，本轮划出范围，理由见需求 2.3。
7. 4.3 的尾块最终落在降级侧而不是 `buildOuterDim`。`TileToBudget.cpp` 的循环仍是
   `scf.for 0 to full step tile`，维度不被分块整除时最后一趟含越界部分；
   `LowerPIMToEmitC.cpp` 的 `tripCountOf` 把循环次数向上取整，`emitDotLoops` 把
   M、K、N 收回到真实维度，`snapshotToLocal` 把拷贝行数夹到真实行数，越界元素
   不进累加也不落盘。`elementOffset` 里的 `clampIndex` 是第二道保险：循环上界
   已经是真实维度时它是恒等变换，现有用例钉的是产物一致而不是夹取被走到。
   在 `buildOuterDim` 里另起一段下界不为 0 的尾块循环走不通——降级器只收下界为
   0 的分块循环，尾块循环会被直接拒掉。

6. 编译算子集合由 16 个扩到 17 个，新增 `reduce` 承载 `aten.mean.dim`。`pow`、`rsqrt`、`unsqueeze`、`transpose.int`、步长为 1 的 `slice`、`neg` 不新增入口，改走已有的 `eltwise`、`lut`、`reshape`、`transpose`。4.6 的「14 处」据此改为按内核实际落点计，名单见 4.6。

## 十、待确认事项

1. `GML_UNMAPPED` 要跑一次 gml 导出才能确定，此前汇总测试会全报缺口，故排在阶段 4。
2. 超 MRAM 的分批驻留本轮不做，理由见 3.1 与需求 2.3：硬件每台 DPU 8GB，实测峰值驻留约 206MB，没有超容量的形状。
3. 仿真依赖 genesim 的 `uv` 与 `.venv`，环境缺失时慢速测试跳过还是失败，未定。
