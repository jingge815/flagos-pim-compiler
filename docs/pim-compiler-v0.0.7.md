# 存算一体大模型推理编译器 v0.0.7


| 项目     | 内容                                           |
| -------- | ---------------------------------------------- |
| 版本     | v0.0.7                                         |
| 日期     | 2026-10-05                                     |
| 目标模型 | Llama-2-7B                                     |
| 基线     | `pim-compiler-v0.0.6`（`09ff9ae`，2026-10-02） |
| 本文范围 | v0.0.6 → v0.0.7 的变化、原理、安装与逐步验证  |

本文档接续 `pim-compiler-v0.0.6.md`，只写 **v0.0.6 → v0.0.7 的变化**。第 6、7 章是
面向新机器的完整操作清单：**安装命令与 v0.0.6 逐条相同**，验证命令也逐条相同，只有
预期数字跟着本版变了（快速回归 1222 → 1289，见 7.2；全流程仿真的耗时与分块分布变了，
见 7.6）。技术方案的完整描述（图编译、算子编译、主机编排、内存管理的设计原理）仍以
`pim-compiler-v0.0.3.md` 为准。

一条主线：

**Llama-2-7B 推理的全部算子走通编译内核**。v0.0.6 的设备内核遇到「不是 2 的幂、类型
不在表里、尺寸对不上」就退回主机 numpy，而端到端断言只要求「至少命中一次编译产物」，
MLP 与 `lm_head` 整段退回时测试照样是绿的。本版把这三类限制全部打开：任意形状进编译，
超 WRAM 由分块循环拆开，余数由尾块处理，退回被逐点计数，端到端断言退回次数为 0。

配套的两件事：

1. **三路径按算子汇总**。numpy 对拍、genesim 助记符、GML 字段族原先各用各的命名，缺
   一份产物看不出来。本版按算子编译器的内核入口逐个核对三份归属，缺一即失败。
2. **仿真侧的放置修复**。拆分放置单元时层面与注意力序号两张表没有跟着重建，3840 个
   注意力算子的 trace 退回手写模板。本版修了下标，并把「产物必须是一套」做成仿真前的
   硬检查：trace 来源与 sidecar 哈希对不上就拒绝开跑。

## 1. 版本与代码量

### 1.1 三个仓库


| 仓库                | v0.0.6（tag `pim-compiler-v0.0.6`） | v0.0.7               | 提交数 | 代码量                                                        |
| ------------------- | ----------------------------------- | -------------------- | -----: | ------------------------------------------------------------- |
| flagos-pim-compiler | `09ff9ae`（2026-10-02）             | 工作区改动（未提交） |      0 | 17 个受控文件，+1860 / -185；新增 4 个文件 419 行（不含本文档） |
| genesim             | `e281b1c`（2026-10-02）             | 工作区改动（未提交） |      0 | 2 个文件，+140 / -8                                           |
| FlagTree            | `4a56daf49`（2026-10-02）           | 工作区改动（未提交） |      0 | 7 个受控文件，+166 / -53；新增 1 个文件 63 行                 |

三点口径说明：

- **三个仓库都有 `pim-compiler-v0.0.6` 这个 tag**，分别指向 `09ff9ae`、`e281b1c`、
  `4a56daf49`，都是 2026-10-02 v0.0.6 交付当天的状态（v0.0.6 文档里写的三个「工作区
  未提交」在交付时提交并打了 tag）。本版的全部改动都还在工作区，三个仓库都是 0 个新提交。
- 代码量相对各自的 tag 统计，含测试、不含文档。flagos-pim-compiler 另有 12 份过程文档
  （需求、设计、实施各 1 份，评审 8 轮）未计入；genesim 的 `.venv` 是环境目录，未计入。
- FlagTree 改的是方言 pass（`TileToBudget.cpp`、`LowerPIMToEmitC.cpp`），改完必须重编
  并同步两份 `libtriton.so`，否则进程内绑定比方言源码旧，回归里的探针会失败（见 8.3）。

### 1.2 测试面


| 项                | v0.0.6          | v0.0.7          |
| ----------------- | --------------- | --------------- |
| 快速回归          | 1222 passed     | **1289 passed** |
| 跳过              | 1 skipped       | 1 skipped       |
| 排除（llama2_7b） | 42 deselected   | 43 deselected   |
| 7B 全量           | 本文档未重跑    | **42 passed**   |
| 端到端仿真        | 1510.447 秒     | **1438.882 秒** |

快速回归多出的 67 条来自本版新增的判据：任意形状与尾块（含真实硬件配置下的
N=1376 / K=1376 / N=32000）、广播与 bmm、int32 / int64 转换、退回计数、三路径
汇总与生成、decode 图的设备标记、仿真产物的新鲜度。排除数从 42 变成 43，是因为
新增的仿真冒烟测试带 `slow` 标记，被 `pytest.ini` 默认排除（见 7.2）。

## 2. 能力变化

### 2.1 任意形状进编译，超容量靠分块而不是退回

v0.0.6 的 `_compiled_linear_supports` 要求 M、K、N 都是 2 的幂且 K ≥ 16。实测
8 DPU 下 `q_proj`、`o_proj` 通过，`gate_proj`、`up_proj` 的 N=1376、`down_proj`
的 K=1376、`lm_head` 的 N=32000 全部被拒，直接退回纯 numpy。这道判断比编译器更严，
而端到端测试只断言「至少命中一次」，整段退回时仍是绿的。

本版把形状从拒绝理由里删掉：

```python
# v0.0.6
return k >= 16 and _is_pow2(m) and _is_pow2(k) and _is_pow2(n)

# v0.0.7
def _compiled_linear_supports(arg_shapes, dtype="float32") -> bool:
    _m, k = flatten_leading_dims(arg_shapes[0])
    return dtype in ("float16", "float32") and k >= 16
```

K 的下限保留：Triton 的 `tl.dot` 要求 K 不小于 16，低于它编译器直接报错。这类形状
退回镜像并计入 fallback，判据看得见，不是静默退回。

形状放开之后，分块算法本身也得接受任意尺寸。三处一起改：

| 位置 | v0.0.6 | v0.0.7 |
| --- | --- | --- |
| `kernel_src.py` | `tl.arange(0, M)` 一次铺满整维，M 必须是 2 的幂 | M 维也按 `BLOCK_M` 分块遍历，三维都加尾块掩码 |
| FlagTree `TileToBudget.cpp` | `validateDot` 要求每维都是 2 的幂，`searchTile` 只在 2 的幂格点上找 | 删掉 2 的幂检查；候选改为「不超过上限的 2 的幂，外加整维本身」 |
| FlagTree `LowerPIMToEmitC.cpp` | 循环用分块当步长走到整维，前提是整除 | 循环上界收回真实维度（`realM/realK/realN`），越界下标夹到最后一个合法位置 |

尾块的实现与最初的设计写法不同，说明一下。设计写的是「主循环只走满块，余数单独成
一块并加掩码」。降级侧（EmitC）没有掩码机制，改成循环向上取整、越界下标夹到边界。
真正让尾块算对的是把循环上界收回到真实维度，夹取是兜底：被测形状下循环上界已是真实
维度，夹取比出来的下标恒等于原下标。`test_triton_and_emitc_agree_on_the_tail_block`
钉的是两种口径的产物一致——数据里埋了陷阱，被夹取反复读到的元素设成 100，一旦它进了
累加，结果会偏 8 万倍量级。

### 2.2 类型与尺寸限制打开

三处提前退回一并去掉，推理里真实出现的类型都能进编译内核：

| 内核 | v0.0.6 | v0.0.7 |
| --- | --- | --- |
| `eltwise` | 只收同形 fp16，标量与广播走 numpy | 先按 `out_shape` 广播到同形，fp16 与 fp32 各编一种；计算类型取输出契约，不取第一槽 |
| `matmul` | 只收二维 fp16 | 三维且两侧都是 fp16 的 bmm 逐批拆成二维再折回；其余形状退回镜像并计入 fallback |
| `convert` | 类型表只有 fp16、fp32、int8 | 加上 int32 与 int64，降级侧补双向转换 |

int64 是补出来的，不是预先设计的。层级 3 首跑时 `convert` 退回 16 次，入参全是
`int64 -> float32`，形状 `(1,1,1)` 与 `(1,1,6)`——attention 掩码的位置索引是 int64，
每个 decode step 都有一批。坐标值域远小于 2^31，fp32 存得下。

bfloat16 明确不做。llama2-7b-hf 的权重与激活都是 float16，整条推理链不产生 bf16，
加进来没有判据能覆盖；而要让它真的编得过，降级侧得补一套 bf16 的位操作转换。

### 2.3 退回被逐点计数

v0.0.6 的退回没有返回值差异，不记录就无法证明没有退回主机。本版在 `runtime/kernels.py`
模块顶层加了一张计数表：

```python
_KERNEL_ROUTE: dict[tuple[str, str], int] = {}

def record_route(op: str, route: str) -> None:
    key = (op, route)
    _KERNEL_ROUTE[key] = _KERNEL_ROUTE.get(key, 0) + 1
```

`hit` 记在调用编译内核前，`fallback` 记在调用镜像前。记录点按内核实际落点计：
`linear`、`eltwise`、`matmul`、`softmax`、`reduce`、`lut`、`reshape`、`transpose`
记命中或退回；没有编译形态的 `relu`、`sigmoid`、`exp`、`sqrt`、`reciprocal` 各记
自己的退回。端到端断言不再分档：

```python
bad = [op for (op, route), n in counts.items() if route == "fallback" and n]
assert not bad, f"这些算子退回了主机：{bad}"
```

计数放在模块内而不是包装层，是因为 monkeypatch 替换整个函数会把判断一起换掉。
对拍包装 `_wrap_with_numpy_cross_check` 有一处例外：它在不支持的形状下直接调镜像，
绕过了 `compiled_linear_kernel` 里的记录点，本版在包装函数退回前补记一次，否则线性
算子的退回对这条判据不可见。

### 2.4 五个算子改走已有编译入口

`mean.dim`、`pow`、`rsqrt`、`unsqueeze`、`transpose.int` 在 llama 的推理计划里占了
二十多条命令，原先直接跑 numpy，既不计命中也不计退回，「兜底次数为 0」看不见它们。
本版不新增编译器能力，复用已有入口：

| 算子 | 走的入口 | 做法 |
| --- | --- | --- |
| `mean.dim` | 新增的 `reduce` | 沿单轴求和再除以轴长，求和在 fp32 上做完再存 |
| `pow`（指数为 2） | `eltwise` 的乘 | 自己乘自己 |
| `rsqrt` | `lut` 的 rsqrt | 直接复用 |
| `unsqueeze` | `reshape` | 插轴不改元素 |
| `transpose.int` | `transpose` | 两个轴号拼成全轴序 |

另外两个顺手接上：`neg` 走 `eltwise` 的减法（`0 - x`）；`slice` 步长为 1 时把被切的
轴先转到轴首，目标区间成了连续前缀，`reshape` 按元素数拷这段再转回原轴序。步长不是
1 的切片没有对应形态，退回镜像并计入退回。

`reduce` 是本版唯一新增的编译算子，编译算子集合由 16 个扩到 17 个。它落在三处：
`contracts/op_semantics.py` 加条目，`oplevel_kernel.py` 加 `reduce_kernel`，
`driver.py` 加分支（`group_size` 复用为归约轴）。

### 2.5 累加器必须在 fp32 上做完

这一条是层级 3 暴露的，不在最初的设计里。真实提示词的生成从第 7 个 token 起与 HF
分叉，文本退化成「The capital of France is the capital of France is…」。逐层比对
隐藏状态：第 0 层每行都对，第 30 层第 0 行差到 633。沿这一行往回追，`rsqrt` 的输入
是 `inf`，归一化整行归零。`inf` 来自 RMSNorm 的两步：

1. `pow_kernel` 把平方收成 fp16。隐藏状态到几百时，单个平方就超过 fp16 上限 65504。
2. `mean_dim_kernel` 把 4096 个平方的和也收成 fp16。累加器本身是 fp32，但结果写回
   fp16 缓冲时溢出成 `inf`。

两处都改成在 fp32 上算完再按命令声明的 dtype 存。`test_mean_of_wide_squares_stays_finite`
用 4096 宽、平方和超过 fp16 上限的输入钉住：修复前均值全是 `inf`，修复后与 fp32 参照一致。

### 2.6 三路径按算子汇总

v0.0.6 的三套测试各用各的命名：numpy 对拍在 `test_opcompiler_ops.py`，5 个算子在
`test_runtime_compiled_coverage.py` 只是正则扫源码，genesim 的助记符不含 `linear`
与 `convert`，GML 按字段族断言不按算子名。缺一份产物看不出来。

本版新增两份测试：

- `tests/test_three_path_generation.py`：每个编译算子真实编译一次，numpy 拿到可加载
  的内核文件，genesim 拿到 pim mlir，并与 numpy 镜像逐元素对拍（相对误差小于 0.05）。
  GML 用一张单层 llama 导出图核对，除 `convert` 外每个算子的 `op_type` 都出现。
  `convert` 只在位宽真变化时才成节点，恒等转换按设计被跨过，测试核对的是这个判定。
- `tests/test_three_path_coverage.py`：按内核入口汇总三份归属，缺一即失败。numpy
  的判据按真实调用形式提取（`op="..."` 与 `_compile("...")`），先按 `#` 截断每行，
  注释、docstring、错误信息里的名字不算覆盖。

两处归属按现状注明，不算缺失：genesim 侧 `linear` 并入 `matmul`，`convert` 与
`reduce` 不是助记符；GML 侧 `reduce` 只出现在 RMSNorm 链里，折进 `RMSNorm_vpu`，
不单发节点。

### 2.7 仿真的放置与产物核对

两处都会让仿真「跑完但数字是错的」，本版都做成了硬检查。

**放置下标错位。** 拆分放置单元时，编译器钉过的算子被摘出来单独成 unit，
`placement_units` 变长了，而 `unit_layer_positions`、`unit_attention_indices`、
`unit_attention_segment_indices` 三张表仍按拆分前的下标记录。下标张冠李戴之后，
GEMM 单元继承了注意力流的 `attention_index`，真正的注意力单元掉出层面分桶，被通用
贪心分到 GPU。实测 3840 个注意力算子的 trace 因此是手写模板产物。修复是拆分每一片时
把原单元的三张属性按新下标抄过去（genesim 的 `gene_sim_scheduler.py`）。同一处还修了
另一个钉不死的口子：B 路算子的 sidecar 给的是 `"shards": []`，空列表不是缺席，顶层
`dpu_id` 被跳过，这些算子只能靠贪心碰巧落 PIM。加载时把空列表当缺席，回落到顶层
`dpu_id`。

**产物不是一套。** sidecar 的 `pimir_sha256` 与磁盘文件不符时，仿真照样开跑，代价
按旧产物计。实测一份 10-02 导出的 sidecar 里 224 个 GEMM 有 96 个哈希对不上，重导
之后 q/k/v 投影的分块从 `(128, 128)` 变成 `(512, 128)`，总时间少了约 41 秒——旧
sidecar 给出的代价确实偏了。本版把两道检查都挂到 `step_e_simulate` 的开头：

1. `verify_sidecar_freshness` 核对 sidecar 全部条目的哈希，对不上就抛 `StepFailed`，
   仿真跑十几分钟之前先拦下来。配置里没有 sidecar 时跳过。
2. `verify_trace_provenance` 的检查面从 `op_*_GEMM.pim_trace` 扩到全部 `op_*.pim_trace`，
   出现模板产物即拒绝。

另外修了一处 trace 元数据的读取：头部是 `PIMT` 时按固定偏移切 JSON，不再用
`raw.index(b"{")` 定位——头部 12 个字节里任意一个等于 `0x7B` 就会切到 JSON 之前；
元数据长度超出文件时直接抛 `StepFailed`，报错带上文件名、声明长度与实际大小，不再
退回按字节扫描。

## 3. 技术原理

### 3.1 为什么「至少命中一次」守不住整网

退回与命中的返回值没有差异，两者都产出数值正确的张量——主机 numpy 镜像本来就是
数值参照。所以「结果与 HF 对齐」这条判据对退回完全不敏感：MLP 的三个投影与
`lm_head` 全走主机时，token 仍然逐个一致，测试是绿的。

要让退回可见，只能在退回发生的那个点上计数，并断言次数为 0。计数必须落在内核函数
内部而不是调用方：调用方可以被测试用 monkeypatch 整个换掉，换掉之后判断逻辑一起
消失，统计永远是 0，又变成一条不可能失败的断言。

### 3.2 2 的幂限制是分块算法的前提，不是硬件的前提

`tl.arange(0, M)` 一次铺满整维，Triton 要求这个范围是 2 的幂，于是 M 被限制死。
K、N 的限制来自另一处：循环用分块尺寸当步长走到整维，没有尾块，所以维度必须能被
分块整除，`_pick` 才反复折半去找一个能整除的 2 的幂。

两处都是算法前提。硬件侧没有这个要求——WRAM 只关心一块放不放得下。把整维改成分块
循环、给最后一块加边界处理之后，前提消失，任意尺寸都能编。候选分块保留 2 的幂是
为了容量内总有得选，另加整维本身是为了不超容量时一次算完，不退回小分块。

### 3.3 尾块为什么用夹取而不是掩码

两条路径的边界处理不一样。Triton 侧有掩码：`mask = offs < DIM`，越界元素不参与
加载和存储，最后一块可以不满。降级侧产出的是 C 代码，没有掩码加载，只能保证下标
不越界。

两道保险叠在一起。第一道把循环上界收回到真实维度：余数块的循环次数按真实剩余元素
算，正常情况下根本走不到边界外。第二道 `clampIndex` 把下标夹到最后一个合法位置，
兜住第一道没覆盖到的路径。被测形状下第一道已经足够，夹取比出来的下标恒等于原下标，
所以夹取本身没有被数值用例打到——`tile_to_budget_odd_extent.mlir` 的 TAIL 行用
FileCheck 钉住它的存在（上界常量 87、`cmp gt`、`conditional`），这是它唯一的守卫。

### 3.4 累加宽度为什么不能跟着存储宽度走

fp16 的上限是 65504。RMSNorm 先平方再求均值：隐藏状态到几百，单个平方就溢出；
即便单个不溢出，4096 个平方的和也溢出。溢出的结果是 `inf`，`rsqrt(inf)` 是 0，
这一行的归一化整体归零，误差从这一层开始逐层放大，到第 30 层已经差出几百。

累加器保持 fp32 并不够。`mean_dim_kernel` 的累加循环本来就是 fp32，溢出发生在
**写回**：结果按存储宽度收成 fp16 时才变成 `inf`。所以「在 fp32 上算完」的含义是
算完之后再按命令声明的 dtype 存，而不是中途每步都收一次。第 30 层那种几百量级的
隐藏状态，平方和在 fp32 里放得下，最终均值也在 fp16 里放得下，只有中间的和不行。

### 3.5 放置表为什么必须跟着单元一起重建

放置单元是调度的粒度，层面位置、注意力序号、注意力分段序号是单元的三个属性。
拆分把一个单元切成多个，三张属性表却留在原下标上，于是新单元继承了别人的属性：
GEMM 单元带上了注意力流的序号，被当成注意力去分桶；真正的注意力单元没有序号，掉出
层面分桶，落到通用贪心，贪心把它分到 GPU。

这一步的后果在数值上不可见——GPU 上算出来的结果是对的，只有 trace 的来源变了。
所以它躲过了所有对拍，只在「trace 必须来自 pim mlir」这条来源检查里暴露。修复不是
给拆分后的单元重新计算属性，而是按拆分后的新下标把原属性抄到对应的新位置上。

### 3.6 为什么要在仿真前核对哈希

仿真跑十几分钟，而 sidecar 与磁盘上的 pim mlir 是两次独立的产物。任何一次只重编了
其中一方的操作都会让它们不是一套：仿真照样跑完，`summary.json` 照样产出，数字却是
旧分块的代价。实测的偏差是 41 秒（1479.698 → 1438.882），来自 128 条 GEMM 的分块
从 `(128, 128)` 变成 `(512, 128)`。

哈希核对放在仿真命令之前而不是之后：产物不是一套时，花十几分钟跑出来的数字没有
意义，越早拒绝越好。配置里没有 sidecar 时跳过，因为那条路径本来就不宣称自己用了
编译产物。

## 4. 这一版修掉的问题

### 4.1 静默退回


| 问题 | 改法 |
| --- | --- |
| 形状不是 2 的幂就退回主机，端到端断言不看退回次数 | 删掉 2 的幂判断，14 处退回点逐点计数，断言退回为 0 |
| 对拍包装在不支持的形状下直接调镜像，绕过记录点 | 包装函数退回前补记一次 |
| `eltwise` 只收 fp16，RMSNorm 的 eps 加法与门控乘在 fp32 域，退回 2080 次 | 内核支持 fp32，标量按第一槽展开 |
| `convert` 没有 int64，attention 掩码的位置索引每个 decode step 退回 | 类型表加 int32 与 int64，降级侧补双向转换 |
| 五个算子直接跑 numpy，不计命中也不计退回 | 改走已有编译入口，`mean.dim` 新增 `reduce` |
| 汇总测试按子串扫源码，注释里的名字也算覆盖 | 按真实调用形式提取，先截掉注释 |

### 4.2 尾块与降级


| 问题 | 改法 |
| --- | --- |
| 维度不被分块整除时最后一趟读出边界 | 循环上界收回真实维度，越界下标夹到边界 |
| `pim.split_heads` 在拆分轴前面还有维度时算错，每份只有最后一块外层的数据 | 写回下标改为「外层序号 × 每份元素数 + 内层序号」 |
| 两侧类型不同时按第一槽的类型编内核，结果偏差 6.4e-4 还被记成 hit | 计算类型改取输出契约 |
| `pow` 与 `mean` 把中间结果收成 fp16，平方和溢出成 `inf`，归一化整行归零 | 在 fp32 上算完再按声明的 dtype 存 |
| `_compiled_normalize` 传了两个形状，契约要求恰好一个，运行期抛 `ValueError` | 改成单个二维形状，epsilon 作为第三个指针传入 |

### 4.3 仿真与放置


| 问题 | 改法 |
| --- | --- |
| 拆分放置单元后下标错位，3840 个注意力算子的 trace 退回手写模板 | 拆分时按新下标抄层面、注意力序号、分段序号三张表 |
| B 路算子的 sidecar 给 `"shards": []`，顶层 `dpu_id` 被跳过，只能靠贪心碰巧落 PIM | 空列表当缺席，回落到顶层 `dpu_id` |
| sidecar 哈希与磁盘文件不符时仿真照样开跑 | `verify_sidecar_freshness` 挂到仿真开头，对不上就拒绝 |
| trace 来源检查只覆盖 GEMM | 扩到全部 `op_*.pim_trace`，模板产物即拒绝 |
| trace 元数据用 `index(b"{")` 定位，头部字节碰巧是 `0x7B` 就切错 | 头部是 `PIMT` 时按固定偏移切，长度越界直接抛错 |
| 编译缓存只看 `kernel_src.py`，改了 IR 生成器旧产物仍被复用 | 指纹纳入三份生成 IR 的源文件，以及 `triton-opt` 和进程内 `libtriton.so` 的大小、mtime |

### 4.4 判据自身不可能失败


| 问题 | 改法 |
| --- | --- |
| 设备标记测试从 `_is_host_only` 反推期望值，与写入同源，增删黑名单都不会失败 | 期望值写死为允许留主机的 aten 名字集合，另加一条把 `aten.add` 标成主机必须失败的用例 |
| `split_heads` 的 lit 断言钉的是源侧下标，写回下标改回错误形态照样通过 | 写回下标挪到读取之后，源侧取常量、写回侧取归纳变量，FileCheck 单独绑出并钉到输出指针 |
| 放置测试的 fixture 没有 `dpu_id`，被测分支根本不执行 | loader 对空 `shards` 回落到顶层 `dpu_id`，测试断言实际加载结果 |
| 非 2 的幂分块没有 lit 守卫 | 新增 `tile_to_budget_odd_extent.mlir`，40x88x48 的 `tt.dot`，容量够与超 WRAM 两条都钉住 |

## 5. 环境与容量要求

**与 v0.0.6 完全相同**，逐条抄录如下（无变化）。


| 项目     | 要求                                                                                                              |
| -------- | ----------------------------------------------------------------------------------------------------------------- |
| 操作系统 | Ubuntu 22.04 或 24.04，x86_64                                                                                     |
| GPU      | **可选**。有 NVIDIA GPU（驱动 570+）用 CUDA 版 torch；无卡用 CPU 版                                               |
| 磁盘     | `flagTree` 约 23 GB、`pytorch` 约 13 GB；模型权重约 14 GB（由甲方提供，放任意目录，见第 6 章第四步）                |
| 内存     | FlagTree 是完整 LLVM/Triton CMake 构建，`MAX_JOBS` 默认 8，低内存机器请调小                                       |
| 网络     | 需访问 GitHub、`oaitriton.blob.core.windows.net`（LLVM）、PyPI、`download.pytorch.org`                            |
| root     | **全程不需要 root**。系统命令需已由管理员预装（清单与自检命令见第 6 章第一步），三个安装脚本与全部验证都不需要 root |

## 6. 安装：一步一步做下来

**本章与 v0.0.6 的第 6 章逐条相同**：三个安装脚本、`paths.json` 的六个键、环境脚本
的 source 方式、以及「不要拷贝已装好的目录」这条限制都没有变。下面只列出照做时要
看的命令，细节与卡点说明请看 `pim-compiler-v0.0.5.md` 第 6 章。

```bash
# 第一步：系统命令自检（不需要 root）
for c in git tar gzip dpkg-deb apt-get awk sed find make cc c++ ar ld curl; do
  command -v "$c" >/dev/null 2>&1 || echo "缺: $c"
done

# 第二步：网络设置（跨境网络建议）
export UV_HTTP_TIMEOUT=600
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# 第三步：三个安装脚本（直接跑，不需要 root、不需要 GPU）
git clone https://github.com/jingge815/flagOS-installers.git
cd flagOS-installers
bash 0-install-flagtree.sh       # 最慢的一步：编译 LLVM/Triton/PIM pass
bash 1-install-flaggems.sh
bash 2-install-pytorch.sh        # 无卡机器加 --torch-cpu

# 第四步：图编译器与 GeneSim
cd /path
git clone https://github.com/jingge815/flagos-pim-compiler.git
git clone https://github.com/pimtools/genesim.git
cd /path/genesim && ./install.sh --skip-attacc
# 然后按 v0.0.5 文档配好 paths.json 的六个键
source /path/flagOS-installed/pytorch/env-pytorch.sh
python -c 'from genesim_bridge.paths import describe; print(describe())'
```

## 7. 验证：一步一步做下来

### 7.1 验证顺序与判据


| 序 | 验证项                | 覆盖什么                       | 耗时       |
| -- | --------------------- | ------------------------------ | ---------- |
| 1  | pim-compiler 快速回归 | 除 7B 端到端外的全部契约与单测 | 约 6 分钟  |
| 2  | GML 导出（两次）      | GML 产物链与参考产物比对       | 各约 20 秒 |
| 3  | genesim 全套          | 仿真器、预测器、UPMEM checker  | 约 1 分钟  |
| 4  | 全流程闭环            | 三仓串起来的端到端             | 约 15 分钟 |
| 5  | 7B 全量（可选）       | 真实 7B 的逐 token 对拍        | 约 80 分钟 |

与 v0.0.6 的差别只有耗时：快速回归从约 3 分钟涨到约 6 分钟，是因为
`test_arbitrary_shape_compiles` 去掉 `slow` 标记后进入了这一级（三个 MLP 形状在
真实硬件配置下真编译、真对拍，单条 127.63 秒）。

### 7.2 pim-compiler 快速回归

```bash
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh

python -m pytest tests/ -q -k "not llama2_7b" -rs
#    预期: 1289 passed, 1 skipped, 43 deselected in 366.20s
#    唯一一条跳过: tests/test_gml_node_parity.py:67  需要参考产物与一次 decode-block 导出
```

本版新增 `pytest.ini`，注册了 `slow` 标记并默认加 `-m "not slow"`，所以命令行不需要
再手写排除。那条跳过与 v0.0.6 相同：`test_gml_node_parity.py` 把「我方产物」的路径
写死成开发机上的 `/media/disk/fengjingge/tmp/gml_dbo/relay2gml_graph.gml`，新机器上
不存在，整组跳过；它比对的两个场景由 7.4 的导出验证覆盖。另外，刚装完、还没跑过
7.6 全流程的机器上，`tests/test_genesim_bridge.py` 里可能**再多**一条因缺 refine
产物而跳过；跑过 7.6 之后就不跳了。**两种状态都是正常的**，`1 skipped` 或
`2 skipped` 都符合预期。

排除数从 v0.0.6 的 42 变成 43：多出的一条是 `tests/test_genesim_simulation.py` 的
仿真冒烟，标了 `slow`，单条要 15 分钟，不进快速回归。要跑它用
`python -m pytest tests/test_genesim_simulation.py -q -m slow`。

> 数字说明：这条命令在本版开发期间随评审修复逐轮上涨，121 条之外的增量全部是新增
> 用例——1222（v0.0.6）→ 1257 → 1263 → 1264 → 1266 → 1268 → 1272 → **1289**。
> 最后一跳含第八轮补上的运行时改动所解锁的用例。**判据是「1 skipped、0 failed」**，
> 绝对值以 1289 为准。

### 7.3 7B 全量（可选）

```bash
python -m pytest tests/ -q -k "llama2_7b"
```

命令与 v0.0.6 的第 7.3 节完全相同，判据多了一条：**退回次数为 0**。v0.0.6 只要求与
单卡 PyTorch 逐元素对齐；本版在此之上断言 `route_counts()` 里没有任何 `fallback`。

实测（约 80 分钟）：

```text
42 passed, 0 failed（4794 秒到 4843 秒）
```

三条端到端（tp8_pp1、tp2_pp4、tp1_pp8）全部通过，生成文本与单卡 HF 一致。修复之前
的首跑是 3 failed / 39 passed：`eltwise` 退回 2080 次、`convert` 退回 16 次，而
linear 侧 3600 次对拍的最大相对误差只有 9.2e-4，文本已经一字不差——退回确实不影响
文本对齐，只有计数看得到。修复后同一诊断复跑：`eltwise` 命中 7728、`convert` 命中
2128，全部算子退回 0 次。

两点要说明：

**（一）这条记录早于后两轮修复。** 42 passed 是第一轮评审修复之后跑的。此后第七轮
修了 fp16 溢出（真实提示词从第 7 个 token 起分叉），第八轮重写了 `kernels.py` 的
运行时改动。这两轮之后全套 42 条没有重跑，重跑过的是失败点所在的三条：
`test_executor_llama2_7b.py` 2 条、`test_natural_prompt_llama2_7b.py` 1 条，修复前
logits 最大差 22，修复后 3 passed（426.87 秒）。

**（二）有一次不可复现的失败。** 42 passed 那一轮里同一命令连跑 5 次，4 次全绿、
1 次 `test_strategy_llama2_7b.py` 的两条断言失败：解码序列从 `[17..24]` 变成
`[11, 12]` 交替，而 11 是参考 logits 的第 5 名（与首位差 2.8），属于数值漂移而非
随机噪声。该文件单独连跑 8 次全部通过。失败那次没有打印 `route_counts()`，无从判断
兜底次数是否为 0。机器上当时有 4 个 vLLM worker 各占约一个核，疑为资源争用，留待在
空闲机器上复核。

`paths.json` 没配模型目录或目录不存在时整组跳过（不是失败）。

### 7.4 GML 导出验证

**命令行与 v0.0.6 完全相同**（decode block 裁剪是默认行为，不再传
`--decode-block-only`）。

第一次，只导结构（不跑算子编译器）：

```bash
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/gml_out
```

预期尾部（约 20 秒）：

```text
节点 200 个，边 331 条
检查:
  [通过] GML 结构自检（5 条规则）: 5/5 通过
  [通过] GML dtype 覆盖（参考有则我方有）: 15 类算子，0 类缺 dtype
  [通过] GML 引用集 == 落盘集: 3122 个文件

图: /tmp/gml_out/relay2gml_graph.gml
运行时文件: 3122 个, 240.36 MB

==============================================================
验证全部通过（3 项）
==============================================================
```

第二次，加算子编译器与编排器：

```bash
python scripts/export_gml.py --layers 1 --seq-len 16 --out-dir /tmp/b \
  --use-opcompiler --orchestrate
```

预期尾部（**24 项全部通过**，约 20 秒）：

```text
节点 200 个，边 331 条
融合 4 处，带 contraction 的节点 1 个
检查:
  [通过] GML 结构自检（5 条规则）: 5/5 通过
  [通过] GML dtype 覆盖（参考有则我方有）: 15 类算子，0 类缺 dtype
  [通过] GML 引用集 == 落盘集: 3122 个文件
  [通过] 接算子编译器前后 GML 逐字节相同: 492676 字节，节点 200
  [通过] 算子编译器真的决定 GML（反证）: 38 个 DQ 相位数 4→2，GML 少 58827 字节
编排器:
  [通过] 层展开: 422 层（非逐头 38、逐头 384）
  [通过] 全部 op_type 都能展开: 无未识别
  [通过] Layer ID 唯一: 422 个，唯一 422 个
  [通过] L2 offset 16 字节对齐: 463 块全对齐
  [通过] L2 地址分配: 463 块 → 6 槽（复用率 98.7%），数据区 258720 字节
  [通过] net.ini 列出全部层: 422 行 vs 422 层 txt
  [通过] txt_files 文件数: 422 层 + 2 个版本戳
  [通过] 编排器产物落盘: /tmp/b/prepare_out/net.ini、txt_files/
  [通过] txt 引用的 bin 全部存在: 2010 个引用，缺失 0 个
  [通过] 盘上 bin 全部被 txt 引用（信息项，不计入失败）: 1116 个未被 txt 引用
  [通过] 双输入层 L2 offset 0/1 不重叠: 41 层双输入，0 层重叠
  [通过] 双输入层 L2 size1 为正: 41 层，0 层 size1<=0
  [通过] MatMul 权重按 S×hd 落盘: 64 个 131072B，0 个仍 65536B
  [通过] KV cache 平面按 nh×S×hd 落盘: 2 个 4194304B
  [通过] L2 分配 ≥ 声明尺寸: 513 处声明，0 处欠分配
  [通过] 参考独有文件族都已产出: 缺 0 族，多 4 族
  [通过] 非白名单文件族数量与参考一致: 0 族数量不同
验证全部通过（24 项）
```

这组数字沿用 v0.0.6 的实测（本版没有重跑导出）：本版改的是编译内核与退回路径，
GML 的结构、编号、标定因子都没有动。四条说明与 v0.0.6 相同，这里保留要点：

**（一）两次导出的 GML 必须逐字节相同**，这是本仓的硬不变量。检查里报的
`492676 字节` 是检查用的文本，权重指纹在检查之后才补进 `artifact.text`，落盘文件
因此比它大 6570 字节（499246 字节）。

**（二）反证也要成立**：把 DQ 的相位数从 4 砍成 2，GML 必须随之变化（少 58827 字节）。

**（三）三件用产物本身就能查的事**：

```bash
python - <<'PY'
import re, pathlib
t = pathlib.Path("/tmp/b/relay2gml_graph.gml").read_text()
ids = sorted({int(m) for m in re.findall(r'^\s+node_id (\d+)', t, re.M)})
assert ids == list(range(1, len(ids) + 1)), "编号必须 1..N 连续"
assert re.findall(r'relay2gml_version "([^"]+)"', t) == ["19.2.0"], "版本号必须是 19.2.0"
io = pathlib.Path("/tmp/b/IO_info.txt").read_text()
sf = re.findall(r"'sf': np\.float32\(([\d.]+)\)", io)
assert sf and all(float(x) > 0 for x in sf), "标定因子不得为 0"
print("节点", len(ids), "；非 1 标定因子", [x for x in sf if x != "1.0"])
PY
#    预期: 节点 200 ；非 1 标定因子 ['0.30420008', '0.2773413', '0.2773413', '0.30420008']
#    （KV 各两次：入边一次、出边一次）
```

**（四）`output_buffer`、`LUT_phase` 这些族来自算子编译器的相位回读**，必须加
`--use-opcompiler` 才会发。

### 7.5 GeneSim 验证

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /path/genesim
./run.sh --test
#    预期尾部: [SUCCESS] All test suites passed.
```

`--test` 依次跑三组，全部通过才打上面的 `[SUCCESS]`：


| 组            | 实测                   |
| ------------- | ---------------------- |
| sim           | 38 个文件 / 686 个用例 |
| predictor     | 7 个文件 / 86 个用例   |
| upmem_checker | 49 个用例              |

这组数字沿用 v0.0.6 的实测。本版对 genesim 的改动另有一条直接覆盖：

```bash
python3 -m unittest tests.sim.test_compiler_placement
#    预期: 60 条全部通过
```

它钉的是本版的两处修复：空 `shards` 回落到顶层 `dpu_id`，以及拆分单元时层面与
注意力序号两张表按新下标重建。

**前置产物不需要**：`./run.sh --test` 不读 `--config`，不要求 `models/*.ir`，不要求
`traces/*.trace`，也不要求联网。测试文件都是自建微型 IR。

按 `docs/llama-2.md` 的默认工作流走一遍仿真（**需要模型目录**）：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
export PATH="$HOME/.local/bin:$PATH"
export PIM_COMPILER_ROOT=/media/disk/fengjingge/src/flagOS/flagos-pim-compiler

# 1. 生成模型 IR
python scripts/model_parser.py \
  --model_name /media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --output models/llama2_7b.ir

# 2. 生成请求 trace
./run.sh --trace --synthetic --seed 0 --num_requests 10 --output traces/llama2_7b.trace

# 3. 成本精化（PIM MLIR 阶段，conf/sim.yaml 默认指向它的输出）
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir --out-ir models/llama2_7b_pimir.ir \
  --sidecar models/llama2_7b_pimir_extensions.json \
  --seq-len 128 --ir-level pimir

# 4. 跑默认仿真
./run.sh
```

> 第 3 步会 import flag_gems；共享机器上 `/dev/shm` 被占满时 import 会报
> `OSError: [Errno 28]`（FlagGems 在 import 期为每个算子建 POSIX 命名信号量，
> 固定落在 `/dev/shm`，不看 TMPDIR）。refine 脚本已自带处理：剩余不足 256 MB
> 时在 user+mount namespace 里把它重定向到脚本目录下的 `dev-shm/`，再重启本脚本。
>
> **不要并发跑两个会写同一份 IR 的进程**。v0.0.6 给 GeneSim 的 `ModelIR.save()` 加了
> 原子写，读者不会再拿到半个文件，但「读到的是哪一次 run 的产物」仍然不保证——
> 并发跑的人请各用各的 IR 路径。
>
> `./run.sh --clean --force` 会**连 `models/*.ir` 一起删掉**，删之前想清楚。

### 7.6 全流程闭环

这是「链路通没通」的唯一判据：

```bash
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python scripts/run_full_pipeline.py --num-stages 4
```

本版把两道检查加进了这条链路：仿真开始前先核对 sidecar 全部条目的哈希
（`verify_sidecar_freshness`），仿真结束后核对全部 trace 的来源
（`verify_trace_provenance`），任何一条是模板产物都直接失败。

实测（约 15 分钟，`tests/test_genesim_simulation.py` 调 `step_e_simulate` 跑的同一次）：

```text
[E] GeneSim 仿真（pimir）
    sidecar 哈希: 6850 条全部一致，仿真才开跑
    total_time_s = 1438.882
    处理 token   = 8084（10 个请求全部完成）
    tokens/s     = 5.618
    执行图 trace : 6720 条全部 trace_source=pimir，模板 0 条
```

与 v0.0.6 的 1510.447 秒 / 5.352 tokens/s 不同，差了约 71 秒。差别来自分块：
v0.0.6 的 sidecar 里 q/k/v 投影有 128 条 GEMM 的分块是 `(128, 128)`，本版重导之后
这 128 条变成 `(512, 128)`，分块更大、循环更少。这正是 2.7 那条哈希检查要抓的
情况——旧 sidecar 与新产物不是一套时，代价数字是偏的。

三行最关键的：

- **sidecar 哈希 6850 条全部一致**——仿真用的 pim mlir 与编译器刚产出的是同一套。
- **6720 条 trace 全部来自 pim mlir，模板 0 条**——v0.0.6 这里只覆盖 448 条 GEMM
  trace（224 个 GEMM × 2 个分片）；本版把检查面扩到全部算子，3840 个注意力算子
  不再退回手写模板。
- `total_time_s = 1438.882`——绝对耗时不作性能预测，适合同类配置横向比较。

脚本退出码非 0 即失败，会把失败步骤的日志尾部直接贴出来。

### 7.7 实测结果汇总


| 验证项                | 命令                                                  | 结果                                                                 |
| --------------------- | ----------------------------------------------------- | -------------------------------------------------------------------- |
| 快速回归              | `pytest tests/ -q -k "not llama2_7b" -rs`             | **1289 passed, 1 skipped, 43 deselected**（366.20s）                 |
| 三路径                | `pytest tests/test_three_path_generation.py tests/test_three_path_coverage.py -q` | **20 passed**（7.70s）                              |
| 任意形状（真实硬件）  | `test_arbitrary_shape_compiles`                       | **1 passed**（127.63s），N=1376 / K=1376 / N=32000 相对误差 < 0.05   |
| GML 导出（结构）      | `export_gml.py --layers 1 --seq-len 16`               | 沿用 v0.0.6：**3/3 通过**，200 节点 / 3122 文件 / 240.36 MB（本版未重跑） |
| GML 导出（完整）      | `export_gml.py ... --use-opcompiler --orchestrate`    | 沿用 v0.0.6：**24/24 通过**（本版未重跑）                            |
| genesim 全套          | `./run.sh --test`                                     | 沿用 v0.0.6：**38 文件 / 686 用例、7 文件 / 86 用例、49 用例**（本版未重跑） |
| genesim 放置          | `python3 -m unittest tests.sim.test_compiler_placement` | **60 条通过**                                                      |
| FlagTree lit          | `tile_to_budget` 与 `lower_to_emitc` 的全部用例       | **全部通过**（含新增的 `tile_to_budget_odd_extent.mlir` 两条）       |
| 全流程闭环            | `run_full_pipeline.py --num-stages 4`                 | **通过**，`total_time_s = 1438.882`，`tokens/s = 5.618`，模板 0 条   |
| 7B 全量               | `pytest tests/ -q -k "llama2_7b"`                     | **42 passed, 0 failed**（约 80 分钟）；后两轮修复后未重跑全套，见 7.3 |

FlagTree 的 lit 有一个环境限制：`/dev/shm` 被占满时 lit 起进程池申请信号量失败。
本版的 lit 结果是按每条 RUN 行用 `triton-opt | FileCheck` 逐条核对的，覆盖的文件是
`lower_to_emitc_ops.mlir`、`tile_to_budget_negative.mlir`、`tile_to_budget_small_wram.mlir`
与新增的 `tile_to_budget_odd_extent.mlir`。

## 8. 限制与注意事项

### 8.1 明确取舍（有意为之，非缺陷）


| 条目 | 说明 |
| --- | --- |
| K < 16 仍退回镜像 | Triton 的 `tl.dot` 硬约束，编译器直接报错；退回计入 fallback，判据看得见 |
| bfloat16 不进 convert 的类型表 | llama2 推理不产生 bf16，降级侧也没有 bf16 的位操作转换，加进来没有判据能覆盖 |
| 超 MRAM 的分批驻留不做 | 硬件每台 DPU 8GB，实测峰值驻留 206509056 字节（利用率 38.47%），溢出为 0，没有超容量的形状 |
| bmm 只覆盖三维且两侧 fp16 | 三维 fp32、四维、两侧类型不同的形状退回镜像并计入 fallback；llama2 的导出图里没有批量矩阵乘 |
| `slice` 只覆盖步长为 1 | 其余步长没有对应的编译形态，退回镜像并计入退回 |
| `reduce` 在 GML 侧不单发节点 | 归约只出现在 RMSNorm 链里，折进 `RMSNorm_vpu` |
| 尾块用夹取而不是掩码 | 降级侧没有掩码机制；循环上界已收回真实维度，夹取是兜底 |
| KV 标定因子不复现甲方 | 与 v0.0.6 相同，未动 |
| SDPA 仍是单一设备节点 | 与 v0.0.6 相同，未动 |

### 8.2 待立项


| 条目 | 现状 |
| --- | --- |
| 层级 3 的全套复跑 | 42 passed 的记录早于第七轮（fp16 溢出）与第八轮（`kernels.py` 重写）。失败点所在的 3 条已单独通过，其余 39 条没有在这两轮修复之后重跑 |
| 一次不可复现的层级 3 失败 | 5 次里 1 次 `test_strategy_llama2_7b.py` 两条断言失败，失败时没有兜底计数。疑为 vLLM worker 争用，需在空闲机器上复核，并在失败分支打印 `route_counts()` |
| 仿真侧 484 个算子仍在 GPU 上 | 模型出入口、层宽视图算子一类。各自都有 pim mlir，但放置策略没把它们钉到 PIM 上。不影响「trace 来自 pim mlir」这条判据 |
| 本仓产物的原子写 | 与 v0.0.6 相同：本仓的 `llama2_7b_*_placed.ir`、placement sidecar、精化后的 IR 仍是 `Path.write_text` 式非原子写 |
| 编译缓存没有清理机制 | 缓存按编译器指纹失效，改内核源码或重编 FlagTree 会让已有缓存整体作废，旧目录随改动累积 |
| GeneSim 的补丁未上游 | 放置表重建改的是独立仓（origin `pimtools/genesim`），对方合并前只存在于本机工作区，pull 时留意被覆盖 |
| FlagTree 的补丁未上游 | 分块放宽与尾块夹取同样只在本机工作区 |

### 8.3 环境陷阱（会制造假结果）


| 陷阱 | 现象 | 规避 |
| --- | --- | --- |
| `/dev/shm` 被占满 | FlagGems 的 `LibEntry` 报 `ENOSPC`；lit 起进程池申请信号量失败 | refine 脚本已内置重定向（见 7.5）；lit 可改用 `triton-opt \| FileCheck` 按 RUN 行逐条核对 |
| 两份 `libtriton.so` 不同源 | 改了 FlagTree 只重编 `triton-opt`，进程内绑定比方言源码旧，`test_inprocess_libtriton_is_not_behind_triton_opt` 失败 | 本版开发中踩了两次。重编后要把产物同步进 PyTorch 环境的 `triton/_C/libtriton.so` |
| 并发跑测试与全流程 | 两方会读写同一份 `genesim/models/llama2_7b.ir` | 分开跑；原子写只保证读不到半个文件，「读到哪次 run 的产物」仍不保证 |
| 旧 sidecar 配新产物 | 仿真照样跑完，耗时与分块分布是旧的 | 本版已在仿真前核对哈希，对不上直接拒绝；手动跑时先重导 sidecar |

### 8.4 文档待同步项


| 位置 | 问题 |
| --- | --- |
| `README.md` | 第 34 行仍写「纯 CPU 的 Ubuntu 22.04」；`gml_llama2_reference_dir` 示例仍指向 v1（与 v0.0.6 相同，未动） |
| `docs/pim-compiler-v0.0.6.md` 第 7.2 节 | 预期仍是 1222 passed / 42 deselected，本版为 1289 / 43 |
| `docs/pim-compiler-v0.0.6.md` 第 7.6 节 | 全流程数字仍是 1510.447 秒 / 5.352 tokens/s，本版重导 sidecar 后为 1438.882 秒 / 5.618 |
| `docs/pim-compiler-v0.0.5.md` 第 7.3 节 | `--decode-block-only` 现在是默认行为，老参数会被 argparse 拒绝（与 v0.0.6 相同，未动） |
| `scripts/verify_cpu_only.sh` | 仍固定 `IMAGE=pim-cputest:22.04`（与宿主机系统版本无关，属已知遗留） |
