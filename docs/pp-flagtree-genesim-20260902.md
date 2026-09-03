# 流水切分打通算子编译与 GeneSim

日期：2026 年 9 月 2 日

## 一、这次做了什么

目标是让流水切分（PP）和混合切分的 Llama2-7B 走通完整链路：图编译 → 算子编译（pim mlir 到 C）→ GeneSim。

改动前的状况是两件事各自缺一半：

| 事项 | 改动前 | 改动后 |
| --- | --- | --- |
| PP 策略过 FlagTree 真实编译 | 从未验证。唯一开启编译产物的 7B 测试是按纯张量并行硬编码的 | 张量、混合、流水三种策略都验证 |
| 放置结果导出给 GeneSim | 桥接函数存在但无人调用，只有单层小模型单测 | 有导出脚本和 GeneSim 配置 |
| GeneSim 跑完整仿真 | 放置接入特性从未跑过完整 `run()`，撞上 trace 准备阶段的硬报错 | 改一处后跑通，详见第四节 |

结论先说：FlagTree 确实不用改，这一点有实测支撑；但 GeneSim 需要改一处，这是实跑才暴露的，之前按读代码得出的「GeneSim 完全不用改」是错的。

## 二、PP 到底需不需要改 FlagTree

不需要。原因可以从切分数学直接看出来。

算子编译器收到的只有本地分片形状、数据类型、任务数和硬件参数。而本地形状只由段内张量并行宽度 `tp_width` 决定，跟流水段数 `num_stages` 无关：

```text
graph/spec_prop.py 的 _shard_map：
  布局是 Shard 时   local_shape 的切分维长度 = 全局长度 / len(dpu_ids)
  布局是 Replicate 时  local_shape = 全局形状

纯流水（tp_width=1）时 len(dpu_ids)=1，除完还是原长度，
所以本地形状等于单卡形状。
```

也就是说流水切分只改变「这份分片属于哪台 DPU」，不改变「这份分片有多大」。算子契约不变，已有编译入口可以直接复用。

这一点有实测支撑，不只是推论。直接读编译后图上的 `PIMTensorSpec.shard_map`，四台 DPU、8 层的小模型：

| 策略 | `tp_width` | `q_proj` 全局到本地 | `o_proj` 全局到本地 |
| --- | ---: | --- | --- |
| `tp4_pp1` | 4 | (256,256) 到 (64,256) | (256,256) 到 (256,64) |
| `tp2_pp2` | 2 | (256,256) 到 (128,256) | (256,256) 到 (256,128) |
| `tp1_pp4` | 1 | (256,256) 到 (256,256) | (256,256) 到 (256,256) |

纯流水那一行本地形状与全局形状完全相同，且列切、行切的方向正确：`q_proj` 切输出维，`o_proj` 切规约维。

真实 7B 上也能看到同样的规律。三种策略的编译产物调用形状分别是 `tp8_pp1` 的 N=512、`tp2_pp4` 的 N=2048，宽度随 `tp_width` 变化而与段数无关。

## 三、验证结果

测试文件 `tests/test_opcompiler_e2e_llama2_7b.py`，真实 7B 权重，开启 `use_compiled_linear=True`。

每次调用编译产物时都与 NumPy 内核在同一份输入上对拍，再用 NumPy 结果继续往下生成，避免误差累积干扰判断。

全部 4 个用例通过，耗时 26 分 17 秒。

| 策略 | 类别 | 编译产物调用次数 | 最大相对误差 | 生成文本与单卡 HF |
| --- | --- | ---: | ---: | --- |
| `tp8_pp1` | 张量 | 15360 | 9.50e-04 | 一致 |
| `tp2_pp4` | 混合 | 3840 | 8.66e-04 | 一致 |
| `tp1_pp8` | 流水 | 1920 | 8.86e-04 | 一致 |

三种策略实际编译的形状如下，这组数据本身就是「本地形状只由 `tp_width` 决定」的最直接证据：

| 策略 | `tp_width` | 编译产物形状 | 形状种类数 |
| --- | ---: | --- | ---: |
| `tp8_pp1` | 8 | (1,1,4096)×(512,4096)、(1,1,512)×(4096,512) | 2 |
| `tp2_pp4` | 2 | (1,1,4096)×(2048,4096)、(1,1,2048)×(4096,2048) | 2 |
| `tp1_pp8` | 1 | (1,1,4096)×(4096,4096) | 1 |

纯流水那一行的 4096 就是未切分的全局宽度。它只有一种形状，因为段内不切分时 `q`、`k`、`v`、`o_proj` 的本地形状完全相同，四者共用同一份编译产物。调用次数随 `tp_width` 减小而下降也是同一个原因：参与计算的 DPU 数从 8 降到 1。

判据有三条，缺一条即失败：

1. `len(stats) > 0`，证明真的走进了编译产物，而不是静默回退 NumPy。
2. 每种形状的最大相对误差小于百分之五。
3. 生成的 token 序列和文本与 `model.generate` 完全相同。

### 哪些算子真的过了 FlagTree

这一点必须说清楚，否则容易把结论读得过宽。

算子编译桥要求矩阵乘的三个维度都是 2 的幂（`opcompiler_bridge/driver.py` 的 `_kernel_launcher`），不满足就回退 NumPy。而 Llama2-7B 的维度是：

| 权重 | 维度来源 | 是否 2 的幂 | 结论 |
| --- | --- | --- | --- |
| `q/k/v/o_proj` | `hidden_size` 4096 | 是，任何 2 的幂宽度切完仍是 | 过 FlagTree |
| `gate/up/down_proj` | `intermediate_size` 11008 = 2^8 × 43 | 否，任何宽度切完仍带因子 43 | 回退 NumPy |
| `lm_head` | `vocab_size` 32000 = 2^8 × 5^3 | 否，仍带因子 5^3 | 回退 NumPy |

所以「过了 FlagTree」只覆盖注意力投影。这是改动前就存在的限制，跟流水切分无关，也不会因为换策略变好或变差。

另外，测试用的提示词是 6 个 token，不是 2 的幂，所以预填充阶段全部回退 NumPy，真正走编译产物的是解码阶段（M=1）。

## 四、放置结果导出

### 数据流

```text
Llama2 权重 + ShardStrategy
        │
        ▼
   compile_llama2          图编译，给每个权重标注 PIMTensorSpec
        │
        ▼
 export_placement_to_genesim   读 shard_map，取 min(shard_map) 作为代表 DPU
        │
        ├──► llama2_7b_<策略名>_placed.ir        GEMM 的 device_hint 改为 pim
        └──► llama2_7b_<策略名>_placement.json   {op_id: dpu_id}
                     │
                     ▼
        GeneSim scheduler.compiler_placement_file
                     │
                     ▼
     _build_adaptive_initial_placement 把这些算子钉死，不进贪心搜索
```

### 新增文件

| 文件 | 作用 |
| --- | --- |
| `scripts/export_pp_placement.py` | 加载 7B、构造策略、编译、导出 sidecar |
| `<genesim>/conf/sim_llama2_7b_pp.yaml` | 指向 sidecar 的 GeneSim 配置 |
| `<genesim>/docs/llama-2.md` 第六步 | 使用说明，写在 GeneSim 的工作流文档里 |

### 这一步是手工离线的，没有自动触发

`export_pp_placement.py` 没有任何调用点，不在任何测试或流程里被自动执行。原因是它跨两个仓库：GeneSim 不认识 `LlamaForCausalLM` 也不认识 `ShardStrategy`，只读一个 JSON；而加载 7B 权重、编译整张图要十分钟量级，不适合塞进任何自动流程。

完整链路是三个手工步骤：

```bash
# 一、产出 sidecar（flagos-pim-compiler 仓，约十分钟）
python scripts/export_pp_placement.py --num-stages 8
#    → <genesim>/models/llama2_7b_tp1_pp8_placement.json

# 二、在 GeneSim 配置里指向它（一次性）
#    conf/sim_llama2_7b_pp.yaml:
#      scheduler:
#        compiler_placement_file: "models/llama2_7b_tp1_pp8_placement.json"

# 三、跑仿真，读到该配置项才生效
./run.sh --config conf/sim_llama2_7b_pp.yaml
```

生效点在 GeneSim 内部：`_load_compiler_placement` 读该配置项得到 `{op_id: dpu_id}`，`_build_adaptive_initial_placement` 据此把这些算子从贪心搜索里摘出来单独钉死。不设置这个键时 GeneSim 行为完全不变，所以对既有流程零侵入。

面向使用者的操作步骤写在 `<genesim>/docs/llama-2.md` 的「第六步（可选）：按图编译器的切分策略放置算子」，与该仓原有的五步工作流接续，包含怎么确认放置真的生效（看 `pu_metrics.json` 的负载分布）。

### sidecar 与 IR 的一致性校验

sidecar 里的 `op_id` 是导出时那份 IR 的编号。如果之后重新跑过 `scripts/model_parser.py` 导致编号变化，或者配置指向了另一个模型的 sidecar，`op_id` 就会错位钉到别的算子上——仿真照样跑完，只是结果悄悄不对。这类「配错了不报错」的失效比普通崩溃更危险，所以加了校验。

导出侧在 sidecar 里多写两样东西（`version` 升到 2）：IR 的算子总数，以及每个被放置算子的 `op_type`。消费侧读取时比对当前 Model IR，任一项不符直接抛错，不做兜底。

不能用 sidecar 已有的 `source_ir` 字段来校验，这一点是查证后否掉的：它是导出时的绝对路径，换机器就失效；而正常用法本来就允许加载同一批 op_id 的另一个 IR——导出针对 `llama2_7b.ir`，仿真加载的是成本精化后的 `llama2_7b_pimir.ir`，两者路径不同但 op_id 完全一致。拿路径比对会误报。

真实 7B 数据上验证了三种错配都被拦下：算子总数不符、`op_id` 不存在于当前 IR、`op_id` 错位（拿一个真实的 SOFTMAX 算子编号冒充 GEMM）。旧版没有这两个字段的 sidecar 仍然照常加载，不会因为字段缺失而报错。

这个校验拦不住同一份 IR 下不同策略之间的混用（op_id 完全相同，内容校验分辨不出来），那种情况只能靠配置里的文件名对上。sidecar 文件名带策略名正是为此，配置注释里也写明了换策略必须同时改这一项。

### GeneSim 侧需要改一处

配置本身不用改校验：`scheduler.compiler_placement_file` 走运行时点路径查找，任意键都能透传，不设置这个键时行为与改动前完全一致。

但真正跑完整仿真时会撞上一处 GeneSim 自身的前后矛盾，必须改代码，这一点是实跑才暴露的：

```text
2026-09-02 16:43:10 [INFO] Built adaptive initial placement: 1184 atomic units,
                            262145 available resources, 1019 used resources
2026-09-02 16:43:11 [INFO] Built 32 fixed layer pipeline Stages for 3232 operators
2026-09-02 16:43:11 [ERROR] Simulation failed:
                            PIM operator 0 (GEMM) has no registered trace compiler
```

放置阶段是成功的，失败在执行前的 `_prepare_pim_traces`。矛盾在于：

- `vpu/vpu.py` 的 `PIMVPU.execute` 里有专门一支处理 `GEMM`，注释明确写了图编译器放置的 GEMM 在 `COMPILE_FUNCS` 里没有指令 trace，所以改走 roofline 公式。
- `_validate_autotune_placement` 也专门为被编译器标注过的算子放开了类型白名单。
- 但执行前的 `_prepare_pim_traces` 对任何不在 `COMPILE_FUNCS` 里的 PIM 算子无条件抛错，而 `COMPILE_FUNCS` 只登记了 attention 类算子，没有 GEMM。

也就是说执行阶段和校验阶段都为「GEMM 被钉在 PIM 上」做了准备，只有 trace 准备阶段没有同步。这说明放置接入这个特性从未真正跑过完整的 `run()`：`scripts/test_compiler_placement.py` 用 mock 直接测 `execute()`，绕过了这道关卡。

改法是在收集待编译算子时先排除被编译器钉住的算子，它们本来就不需要 trace：

```python
compiler_placed_op_ids = set(self._load_compiler_placement())
for op in placed_ir.operators:
    ...
    if op.op_id in compiler_placed_op_ids:
        continue
```

改之前确认了两件事，避免改出新问题：

一是静态 IR 里这些 GEMM 的 `flops` 和 `data_bytes` 都是 0，看着会让 `execute` 的 roofline 分支（条件是两者大于 0）落空。但运行时的 `_get_operator_with_runtime` 会用 `flops_coeffs` 重新求值，op0 的系数是 `{Tq: 7.6e7, constant: 3.1e9}`，解析出来必然为正，所以 roofline 分支会正确命中。静态的 0 不是问题。

二是配置里有 `split_ops: true`，如果 GEMM 会被拆成子算子，子算子的 op_id 不在 sidecar 里就会漏掉。实际查证 `split_ops` 只在 `config_loader.py` 有一个默认值、从未被消费，`split_operator` 也没有任何调用点，所以按 `op.op_id` 匹配是安全的。

真实 `llama2_7b.ir` 的结构也正好满足导出函数的假设，这一点是读实际产物确认的：32 层、3232 个算子、`subgraphs` 恰好 32 段、每层恰好 4 个 GEMM、op_id 全局唯一且不复用。因为 op_id 不复用，sidecar 才能给不同层写不同的 DPU，流水语义得以完整表达。

### 导出逻辑的验证

用 8 层和 16 层的小模型验证过三件事：每层 4 个 GEMM 都被放置、同层的 GEMM 落在同一台 DPU、层到 DPU 的映射与 `strategy.dpus_of_layer` 完全一致。16 层是关键用例，因为同时存在 `layers.1` 和 `layers.1x`，能暴露权重名子串误匹配。

真实 7B 的导出产物也逐项核对过，纯流水和混合两种策略都跑了：

| 策略 | GEMM 数 | 每台 DPU | 层到 DPU 映射 | 核对结果 |
| --- | ---: | --- | --- | --- |
| `tp1_pp8` | 128 | dpu0 到 dpu7 各 16 个 | 严格等于 `layer // 4` | 通过 |
| `tp2_pp4` | 128 | dpu0、2、4、6 各 32 个 | 严格等于 `(layer // 8) * 2` | 通过（段内失真见第六节第 3 条） |

两种策略的输出 IR 与输入基线对比都是同一个结论：只有那 128 个算子的 `device_hint` 从 `gpu` 变成 `pim`，算子总数、`subgraphs`、`dependencies`、所有输入输出形状都没有变化。

### 端到端仿真结果

`tp1_pp8` 的完整仿真跑通：

```text
Built adaptive initial placement: 1184 atomic units, 262145 available resources,
                                  1019 used resources, 1524 aggregated forwards
Built 32 fixed layer pipeline Stages for 3232 operators
Pipeline simulation finished. Total time: 11889.65 s
Completed requests: 10        processed tokens: 8084
Simulation completed successfully
```

放置钉死确实生效，这一点从每个 PU 的负载分布可以直接看出来，不只是「没有报错」：

```text
PIM_PU:17:0 … 17:7    busy=5323.59s  util=0.4477   ← 恰好 8 个，数值完全相同
其余 1010 个 PU        busy=2.63s     util=0.0002   ← 注意力算子
```

恰好 8 个 PU 承担完全相同的负载，正是被钉住的 8 台 DPU（dpu0 到 dpu7 映射到 vpu17 的 pu0 到 pu7）。

同时 `communication_ratio` 只有 0.00017，近乎为零，从结果侧印证了下一节说的编号塌缩问题：跨流水段的搬运没有被计入。所以这份输出只能用来说明流程走通、放置正确，不能当作流水切分的性能评估。

## 五、顺带修掉的一个真实缺陷：读写地址重叠

这个缺陷是这轮验证暴露出来的，跟流水切分无关，改动前就存在。已修复。

### 症状

内存规划器做激活区生命周期复用时，会把某个节点的输出缓冲区分配到它自己输入的基址上。NumPy 内核先把输入整块读进数组再写回，对这种别名安全；编译产物直接把裸指针交给 C 函数，逐块边读边写，写输出的前几行就覆盖了尚未读取的输入行。

触发条件是输出字节数大于输入字节数且行数大于一。实测误差：

| M | K | N | 输出/输入字节 | 无别名 | 有别名 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 64 | 256 | 2048 / 512 | 2.4e-04 | 5.9 |
| 4 | 128 | 256 | 2048 / 1024 | 2.3e-04 | 1.1e+02 |
| 8 | 64 | 256 | 4096 / 1024 | 2.4e-04 | 7.8 |
| 1 | 64 | 256 | 512 / 128 | 3.4e-04 | 3.4e-04 |

最后一行是 M=1，输入已经读完，覆盖无害，属于侥幸正确。

为什么一直没被发现：现有测试的提示词是 6 个 token，不是 2 的幂，预填充直接回退 NumPy；解码阶段 M=1 又恰好安全。也就是说编译产物路径从来没有在 M 大于一的情况下跑过。用 2 的幂长度的提示词做小模型冒烟才暴露出来。

### 根因

`memory/mem_planner.py` 的 `greedy_reuse` 判断能否复用一个槽位时，两个生命周期判据都用了非严格不等号：

```text
start >= t.last_read_at  或  t.produced_at >= end
```

`node_step` 给每个节点唯一编号，所以取等只可能是同一个节点既读旧张量又写新张量。实测在 8 层小模型的预填充图上，同 step 读写别名有 172 处，其中输出比输入大、真正会算错的有 16 处，例如：

```text
off=4096
  旧: redist_dst:e19  size= 512  produced=158  last_read=159
  新: linear_3        size=2048  produced=159   ← 输出更大，覆盖未读的输入
```

### 修法

选择让规划器保证不产生别名，把两个判据都改成严格不等号。理由是契约应当在规划阶段就成立，而不是依赖内核的分块顺序碰巧安全——实测 `out` 与 `x` 尺寸相同时结果正确，纯粹是行主序的巧合，不该依赖。

两个判据必须一起改。第一次只改了第二个，结果「会算错」的数量从 16 涨到 57，方向反了：`greedy_reuse` 按尺寸降序插入，大输出先进槽、小输入后进时，检查走的是第一个判据，等号同样放行——而这恰好就是危险场景。两个都改严格后别名归零：

| 场景 | 修复前激活区 | 修复后 | 增幅 | 别名对 | 会算错 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tp4_pp1` 预填充 | 11264 B | 15360 B | +36% | 172 → 0 | 16 → 0 |
| `tp4_pp1` 解码 | 5120 B | 6144 B | +20% | 172 → 0 | 16 → 0 |
| `tp1_pp4` 预填充 | 13312 B | 17408 B | +31% | 35 → 0 | 0 → 0 |
| `tp1_pp4` 解码 | 5120 B | 6144 B | +20% | 35 → 0 | 0 → 0 |

代价是激活区增大百分之二三十，绝对值都是千字节量级，相对 GB 级的 MRAM 预算可以忽略。

### 验证

小模型冒烟里原先 `max_rel` 等于 1.0（结果全零）的那几个形状，修复后全部回到 1e-04 量级：

| 形状 | 修复前 | 修复后 |
| --- | ---: | ---: |
| `(1,4,64),(256,64)` | 1.000e+00 | 3.509e-04 |
| `(1,4,128),(256,128)` | 1.000e+00 | 1.905e-04 |
| `(1,4,256),(512,256)` | 1.000e+00 | 0.000e+00 |

测试放在两层：

- `tests/test_mem_planner.py` 的 `test_greedy_reuse_never_aliases_same_step_read_and_write` 是主测试，直接固定规划器的不变量，并覆盖两个判据分支各自的危险场景，同时验证真正隔开一步的生命周期仍然可以复用。
- `tests/test_opcompiler_linear.py` 的 `test_compiled_linear_requires_non_aliasing_output_buffer` 记录内核这一侧的契约方向：无别名时两条内核一致，有别名时编译产物确实算错。如果哪天内核改成先把输入读进暂存区，这个测试会失败，提示可以把规划器的判据放宽回去、把激活区省回来。

两个测试都做过变异验证：把判据改回非严格后，它们确实失败（规划器那条报 `assert 0 != 0`，两个张量拿到同一地址），确认覆盖是真实的而不是恰好通过。

## 六、把切分真正贯穿到成本与通信

前面两节把「层段归属」送进了 GeneSim，但仿真报出的数字还有两处不反映切分：跨段搬运的延迟恒为零，每个算子的计算量仍是未切分的口径。这一节修掉这两处。

### 1. 跨流水段的通信延迟原本计为零

GeneSim 各条执行路径在算通信延迟前都有一道判断，源和目标在同一个 vpu_id 上就跳过。而放置结果里的 `dpu_id` 原先是对整个 PIM 资源列表取模映射的：

```text
PIMVPU 实例数=2048  每实例展开 128 个 PU  资源列表长度=262144
dpu0..dpu7 全部映射到 vpu17，只是 pu_id 不同
```

`resources` 把每个 PIMVPU 展开成 `clusters_per_device × tensors_per_cluster` 个条目，所以列表前 128 项共享同一个 vpu_id，8 个 dpu_id 全落在里面。后果是跨段搬运压根不调用带宽模型。

**改法**：`_build_adaptive_initial_placement` 里构造 `pim_resources` 时按 vpu_id 去重，每个 PIMVPU 只保留它的第一个条目，这样 `dpu_id % len(pim_resources)` 落到互不相同的 PIMVPU 上。

改之前确认了三件事，避免改出更糟的结果：

一是 pipeline 路径（真实仿真走的就是它，日志里的 `Pipeline simulation started`）确实会算通信：`_execute_pipeline` 的 `enqueue_transfer` 里有 `topology.transfer`。如果它压根不算，修编号也看不到效果。

二是那里遇到无链路时是**直接抛错**（`No finite communication path from VPU ... to VPU ...`），不像顺序路径那样静默跳过。所以必须先确认这 8 个 vpu 之间的链路存在，否则修完仿真直接崩。

三是实测这 8 个 PIMVPU（vpu17 到 vpu24）的 `cluster_id` **全为 0**——`tensors_per_cluster` 是 16，前 16 个 PIMVPU 都在 cluster 0 里。所以走的是 intra-cluster 链路，正是本轮配置里已经补上的 `PIM_Intra_Cluster`，不会撞上上面那个抛错。

**验证**：真实 7B、`tp1_pp8` 下，8 个流水段一一对应 8 个不同的 vpu_id：

```text
stage0 (layers 0..3)   -> vpu 17
stage1 (layers 4..7)   -> vpu 18
...
stage7 (layers 28..31) -> vpu 24
覆盖 cluster_id: [0] -> 同一 cluster，走 intra 链路
```

完整仿真跑完后，`results/communication_metrics.json` 里出现了修复前不可能存在的跨段链路：

```text
跨流水段的 PIM-PIM 链路: 20 条，2,080,868,608 字节，累计传输 0.0645 s
  LINK:18:17   164,982,528 B   0.004102 s
  LINK:19:17   164,982,528 B   0.004102 s
  ...（另 18 条）
涉及 GPU 的链路: 16 条，14,612,206,080 字节（原本就有）
```

修复前 8 个段全在 vpu17，`src_vpu_id == dst_vpu_id` 直接跳过，`LINK:17:18` 这类条目压根不会产生。总时间也从 11889 s 涨到 12008 s（约百分之一），与多出来的搬运开销一致。

**验收判据要看链路，不要看 `communication_ratio`。** 这一点我起初判断错了：本以为该比值会明显上升，实测反而从 1.7e-04 降到 4.2e-05。原因是它的定义是

```text
communication_ratio = 传输服务时间 / (计算服务时间 + 传输服务时间)
```

分母里的计算服务时间是 45245 s，传输只有 1.9 s。修复后分子分母同时增加，而新增的跨段传输（0.06 s 量级）远小于同时增加的计算服务时间，比值就被摊薄了。这个指标在计算密集型负载上本来就极小，不适合当"通信是否被计入"的判据。正确的判据是 PIM 之间是否存在带流量的链路。

回归测试 `test_distinct_dpu_ids_land_on_distinct_vpus` 用两个 PIM 设备、每个展开 2 个 PU 的最小配置复现这个塌缩场景。变异验证：把去重逻辑改回原样后它确实失败（`1 == 1`，两个 dpu_id 挤在同一个 vpu 上）。

### 2. 算子成本原本是单卡口径

放置导出和成本提取原本是两条互不感知的桥接。成本提取只用模型级全局维度（`ir["hidden_size"]` 这种），sidecar 里 `shard_participants` 直接写 `None`。所以即使算子被钉到了正确的 DPU，它的运算量仍是整个未切分算子的量，只是记账到某一台 DPU 上。

**改法**分两侧：

导出侧在 sidecar 里多写每个算子的本地宽度 `local_in_features` / `local_out_features`，直接取 `spec.shard_map[dpu_id].local_shape`——放置导出的循环里本来就有它。

消费侧 `export_costs_to_genesim` 增加可选参数 `local_shapes: {op_id: (in, out)}`，命中的算子按本地宽度编译测量，未命中的沿用全局形状。`load_local_shapes` 负责从放置 sidecar 里把它读出来。GeneSim 的薄入口脚本加 `--placement` 参数把两者接起来。

**qkv 的宽度要累加，不能拿代表权重乘份数**。GeneSim 的 IR 把 q/k/v 合并成一个 GEMM（真实 7B 上是 `4096 -> 12288 = 3 × 4096`），而代表权重只是 `q_proj`。只拿 `q_proj` 的本地 out_features 当这个 GEMM 的输出宽度，成本会被低估到三分之一。

第一版按「乘 3」实现，事后复查发现这在分组查询注意力（GQA）下是错的：k/v 的头数少于 q，三者宽度并不相同。实测一个 8 个 q 头、4 个 kv 头、`tp_width=2` 的模型：

```text
q_proj  global=(256,256)  local=(128,256)
k_proj  global=(128,256)  local=( 64,256)
v_proj  global=(128,256)  local=( 64,256)

真实本地宽度之和 = 128 + 64 + 64 = 256
按 q 乘三        = 3 × 128       = 384   高估 1.5 倍
```

不报错、不告警，成本直接高估五成。`graph/strategy.py` 的 `llama_strategy` 明确接受 `num_kv_heads` 参数并独立校验它能否被 `tp_width` 整除，也就是说 GQA 是被支持的输入；只是当前用的 llama2-7b 恰好是 32/32 的多头注意力，掩盖了这个缺陷。Llama2-70B 和 Llama3 全系都是 GQA。

现在的实现是 `_GEMM_OUT_FEATURE_SOURCES` 逐个累加 q/k/v 的实际本地宽度，对 GQA 和非 GQA 都正确。真实 7B 上重新导出，qkv 本地宽度仍是 6144，与「乘 3」的旧结果一致——非 GQA 下两种算法等价，所以这个修复是向后兼容的。

测试 `test_qkv_local_width_sums_actual_kv_shards_under_gqa` 用 GQA fixture 固定这条；变异验证退回旧算法后它确实失败（`96 == 64`）。

**缓存键也要带上本地形状**。`export_costs_to_genesim` 按 `(op_type, input_shapes, output_shapes)` 缓存测量结果，同一个全局形状在不同切分宽度下对应不同本地形状，漏掉它会让后面的算子错误复用前面的测量值。

**成本精化这条路径也要校验 op_id 没错位**。GeneSim 侧读放置结果时有一致性校验，但成本精化是另一条路径，第一版漏了：传入一份编号已经错位的放置结果，会把某个算子的本地宽度套到别的算子上，不报错、只是成本悄悄不对。现在 `validate_local_shapes_against_ir` 在开始编译之前就核对两件事——op_id 是否都存在于目标 IR，以及它们在 IR 里是否都是 GEMM（本地宽度只对 GEMM 有意义，落到别的类型上就说明编号错位了，即使算子总数恰好相同也能被抓出来）。

真实 7B 上验证过三种情形：正常放置结果通过；`op_id` 不存在时抛错；把一个真实 SOFTMAX 的 op_id 塞进去时抛错。变异验证移除校验调用后，对应测试失败，而且从警告能看到它真的走到了 FlagGems 编译——印证了「拦在编译之前」这一点。

**验证**：真实 7B、`tp2_pp4`（`tp_width=2`）下每个 GEMM 的 flops 精确减半，与切分数学吻合：

| 算子 | 全局口径 | 本地口径 | 比值 |
| --- | ---: | ---: | ---: |
| op0（qkv） | 1.289e+10 | 6.443e+09 | 0.500 |
| op97（o_proj） | 4.295e+09 | 2.148e+09 | 0.500 |
| op98（fc1） | 1.154e+10 | 5.772e+09 | 0.500 |
| op100（fc2） | 1.154e+10 | 5.772e+09 | 0.500 |

切分方向也逐个核对过：qkv 和 fc1 按输出维切（out 减半），o_proj 和 fc2 按规约维切（in 减半）。未被放置的注意力算子（`GEMV_SCORE` / `SOFTMAX` / `GEMV_CONTEXT`）数值逐位不变，确认改造没有溢出到不该动的地方。

纯流水 `tp1_pp8` 下本地形状**等于**全局形状（`tp_width=1` 不切分），所以成本不变——这不是没生效，而是切分数学的必然。要看出 C2 的效果必须用带张量切分的策略。

产出的 IR 也核对过结构完整性：GeneSim 的 loader 正常加载，算子总数、`subgraphs`、`dependencies`、所有输入输出形状与输入基线逐项一致。

改动范围精确可控，这一点是直接比对两份 IR 得到的，不依赖仿真：

```text
flops_coeffs 变化:      128 个，全部是 GEMM
data_bytes_coeffs 变化: 128 个，全部是 GEMM
变化集合 == 被放置集合: True
```

按 `Tq=128` 展开求值，全模型的量级关系是：

| 部分 | 全局口径 | 本地口径 | 比值 |
| --- | ---: | ---: | ---: |
| 128 个 GEMM 的 flops | 1.2886e+12 | 6.4433e+11 | 0.5000 |
| 128 个 GEMM 的 data_bytes | 1.0515e+10 | 5.3247e+09 | 0.5064 |
| 非 GEMM 部分 | 9.1417e+09 | 同左 | 1.0000 |
| 全模型 flops | 1.2978e+12 | 6.5347e+11 | 0.5035 |

GEMM 的 flops 比值精确等于 `1 / tp_width`。`data_bytes` 是 0.5064 而不是正好 0.5，因为权重字节按分片减半、而激活字节里的 `Tq × in_features` 那一项在按输出维切的算子上并不减少——这是切分语义本身，不是误差。全模型比值 0.5035 略高于 0.5，是因为占 0.7% 的非 GEMM 部分没有变。

**做仿真层面的对照时要注意两件事**，我自己先踩过：

一是必须**同策略对照**。拿 `tp1_pp8` 的运行结果和 `tp2_pp4` 的比是没有意义的：策略和成本口径两个变量同时变了。要隔离成本口径的影响，得用同一份放置结果、同一个策略，只把 `model.ir_path` 在全局口径的 `llama2_7b_pimir.ir` 和本地口径的 `llama2_7b_tp2_pp4_local.ir` 之间切换。

二是 `results/` 目录**会被覆盖**。它是固定路径，这台机器上别的进程也在写，实测出现过 `summary.json` 的时间戳比我这次运行还新、内容是上一轮策略的情况。跑完要立刻把 `summary.json`、`communication_metrics.json`、`pu_metrics.json` 复制到别处再分析，不要隔一段时间回来读它。

## 七、遗留问题

### 1. 放置导出只能表达一台代表 DPU，段内张量切分会失真

放置导出取 `min(shard_map)` 作为代表 DPU，一个算子只能对应一个 dpu_id，表达不了「这个算子切在多台 DPU 上」。三种策略受影响的程度完全不同，这一点用真实 7B 导出逐个确认过：

| 策略 | 段划分 | sidecar 实际标注 | 失真情况 |
| --- | --- | --- | --- |
| `tp1_pp8` | 8 段 × 1 台 | dpu0 到 dpu7，各 16 个 GEMM | 无，每段只有一台 DPU，`min` 就是它 |
| `tp2_pp4` | 4 段 × 2 台 | 只有 dpu0、2、4、6，各 32 个 | 段级正确，段内失真 |
| `tp8_pp1` | 1 段 × 8 台 | 全部标到 dpu0 | 完全退化 |

混合策略的具体表现是每段只标首台，另一半 DPU 完全缺席：

```text
stage0: dpus=[0, 1]  →  sidecar 只有 dpu0，承担 32 个 GEMM
stage1: dpus=[2, 3]  →  只有 dpu2
stage2: dpus=[4, 5]  →  只有 dpu4
stage3: dpus=[6, 7]  →  只有 dpu6
```

后果是 GeneSim 会以为每段的计算全压在一台 DPU 上，该段负载虚高一倍，段内 TP 的并行度体现不出来。所以当前这条桥接只适合纯流水；混合策略可以跑通、层段归属也对，但段内的并行度在仿真里看不到。

要修得改 sidecar 的表达能力，让一个算子能带一组 dpu_id，GeneSim 侧也要能把一个算子摊到多个 PU 上——这已经超出「放置提示」的范畴，牵涉到 GeneSim 怎么建模一个被切分的算子。

### 2. 被钉住的算子挤在每个 vpu 的第一个 PU 上

这是修跨段通信时按 vpu_id 去重带来的代价。去重让每个 PIMVPU 只在 `pim_resources` 里留一个条目，即 `first_pu_id + 0`，所以同一个 dpu_id 上的所有 unit 都绑到那一个 PU。真实 7B `tp1_pp8` 下每台 DPU 有 16 个 GEMM，它们全在同一个 PU 上，而同 vpu 的另外 127 个 PU 空闲。

修复前的行为是另一种失真：8 个 dpu_id 落在同一个 vpu 的 pu0 到 pu7 上，跨段通信被整条跳过。两者都不理想，但通信被计入比 PU 分布更接近真实——PIM 设备内部本来就有多个 TensorPU 协同算一个算子，而当前的放置桥接表达不了这种协同（见上一条）。

要同时修好这两件事，需要 sidecar 能表达「一个算子用哪几个 PU」，跟上一条是同一个根因。

### 3. 换策略要手工同步改配置里的文件名

放置结果的文件名带策略名（`llama2_7b_tp1_pp8_placement.json`），而 GeneSim 配置里的 `compiler_placement_file` 是写死的。换策略时如果只改了导出命令、忘了改配置，仿真会静默用上一个策略的放置结果。

op_id 层面的错配现在会被 GeneSim 侧和成本精化侧的一致性校验拦下，但同一份 IR 下不同策略之间的混用它们分辨不出来——两份 sidecar 的 op_id 完全相同，只有 dpu_id 不同。当前只能靠"一种策略一份配置文件"这个约定，配置注释和 `<genesim>/docs/llama-2.md` 第六步都写明了这一点。

要根治得让 sidecar 记下策略名、配置里也声明期望的策略，读取时比对。

### 4. 其余沿用上一轮的结论

流水线还没有重叠调度、跨段通信走主机中转而非点对点、搜索空间只枚举 2 的幂段数、层号识别依赖 Llama 导出结构。详见 `docs/split-pp-20260830.md` 第八节。

注意「分组查询注意力未覆盖」这条已经部分收敛：本轮修掉了 qkv 本地宽度在 GQA 下算错的问题（见第六节第 2 条），成本口径对 GQA 是正确的。但 GQA 模型本身没有跑过端到端验证，`llama_strategy` 对 `num_kv_heads` 只做整除校验、KV 缓存布局在 GQA 下是否正确也未验证，所以整体仍算未覆盖。

## 八、怎么复现

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh

# 一、三种策略过 FlagTree 编译并与单卡 HF 对拍（需要 GPU，约半小时）
python -m pytest tests/test_opcompiler_e2e_llama2_7b.py -q -s

# 二、导出放置结果（含本地分片宽度），纯流水用 8，混合用 4
python scripts/export_pp_placement.py --num-stages 8

# 三、只跑放置：直接用已有的 pimir 成本
cd /media/disk/fengjingge/src/genesim
./run.sh --config conf/sim_llama2_7b_pp.yaml
```

要让成本也按本地分片形状测量（第六节第 2 条），在第二、三步之间插一步重新精化。注意纯流水下本地形状等于全局形状、成本不会变，所以这一步要配合带张量切分的策略才看得出差别：

```bash
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler
python scripts/export_pp_placement.py --num-stages 4        # tp2_pp4

cd /media/disk/fengjingge/src/genesim
python scripts/refine_ir_with_flagtree.py \
    --ir models/llama2_7b.ir \
    --out-ir models/llama2_7b_tp2_pp4_local.ir \
    --sidecar models/llama2_7b_tp2_pp4_local_ext.json \
    --seq-len 128 --ir-level pimir \
    --placement models/llama2_7b_tp2_pp4_placement.json    # ← 关键参数

./run.sh --config conf/sim_llama2_7b_pp_tp2pp4.yaml
```

精化那步约十几秒（去重后只需编译七个形状），日志会打印 `Using local shard shapes for 128 operators`，没有这一行就说明 `--placement` 没生效。

## 九、改动清单

本仓（flagos-pim-compiler）：

| 文件 | 变化 |
| --- | --- |
| `memory/mem_planner.py` | `greedy_reuse` 的两个生命周期判据改为严格不等号，消除同 step 读写别名（+13 / -2） |
| `genesim_bridge/placement_export.py` | sidecar 升到版本 2：写入 `ir_num_operators`、`op_type`，以及本地分片宽度（qkv 按 q/k/v 实际分片累加，兼容 GQA）（+58 / -6） |
| `genesim_bridge/cost_extractor.py` | `export_costs_to_genesim` 增加 `local_shapes` 参数；新增 `load_local_shapes` 与 `validate_local_shapes_against_ir`；缓存键带上本地形状（+125 / -9） |
| `genesim_bridge/__init__.py` | 导出 `load_local_shapes`、`validate_local_shapes_against_ir`（+8 / -1） |
| `tests/test_mem_planner.py` | 新增规划器不变量测试，覆盖两个判据分支各自的危险场景（+29） |
| `tests/test_genesim_bridge.py` | 新增 8 个测试：本地宽度算字节、缺省退回全局、读放置 sidecar、旧版返回空，以及 4 个 op_id 错位校验（+115） |
| `tests/test_placement_export.py` | 新增本地宽度测试和 GQA fixture 下的 qkv 累加测试（+125） |
| `tests/test_opcompiler_e2e_llama2_7b.py` | 删掉内联重复的编译流程，改调 `compile_llama2` 和 `load_weights`；策略参数化为三种 |
| `tests/test_opcompiler_linear.py` | 新增内核侧的读写别名契约测试（+70） |
| `scripts/export_pp_placement.py` | 新增，放置结果导出入口 |
| `docs/pp-flagtree-genesim-20260902.md` | 新增，本文 |

净增删 `+600 / -134`，其中 7B 测试文件因删除内联重复代码而净减 116 行。

GeneSim 仓：

| 文件 | 变化 |
| --- | --- |
| `src/scheduler/gene_sim_scheduler.py` | `_prepare_pim_traces` 跳过被钉住的算子；新增 `_validate_compiler_placement`；`pim_resources` 按 vpu_id 去重（+95 / -8） |
| `tests/sim/test_compiler_placement.py` | 新增 8 个回归测试：trace 准备 2 个、sidecar 一致性 5 个、vpu 去重 1 个（+211）。合并 develop 后随该仓测试布局从 `scripts/` 迁到 `tests/sim/` |
| `scripts/refine_ir_with_flagtree.py` | 新增 `--placement` 参数，把本地分片宽度接进成本提取（+21 / -3） |
| `conf/sim_llama2_7b_pp.yaml` | 新增，`tp1_pp8` 的仿真配置，补上 `PIM_Intra_Cluster` 链路 |
| `conf/sim_llama2_7b_pp_tp2pp4.yaml` | 新增，`tp2_pp4` 的配置，指向按本地分片形状精化的 IR |
| `conf/sim_llama2_7b_pp_tp2pp4_globalcost.yaml` | 新增，A/B 对照用：与上一份只差 `model.ir_path`（全局成本口径） |
| `docs/llama-2.md` | 新增第六步，把 `export_pp_placement.py` 的用法接到该仓原有的五步工作流后面（+153） |

净增删 `+480 / -11`。

三份配置的注释头都写明了各自的用途、产出命令、以及当前仍存在的局限。这一点是复查时补的：最初三份都是从第一版复制的，注释里还留着"通信不计入""本配置写死 tp1_pp8"这类已经过时或对不上号的说法，会误导使用者。

该仓的测试入口是 `./run.sh --test compiler_placement`（对应 `tests/sim/test_compiler_placement.py`），不是 pytest；`./run.sh --list-tests` 列出全部目标。开发时也可以直接 `python -m pytest tests/sim/test_compiler_placement.py`（同一批用例、输出更详细），但提交前建议用 `./run.sh --test` 跑一遍，那才是该仓在 `uv` 环境下的正式入口。

### 每处修复都做过变异验证

做法是把修复临时改回原样，确认对应测试真的失败——不这样做就无法区分"测试通过"和"测试没测到"。五处修复的结果：

| 修复 | 变异方式 | 变异后的失败表现 |
| --- | --- | --- |
| 读写别名 | 判据改回非严格 | 规划器不变量测试报 `assert 0 != 0`（两个张量拿到同一地址） |
| trace 准备跳过被钉算子 | 移除两行 skip | 报出线上原样的 `PIM operator 0 (GEMM) has no registered trace compiler` |
| sidecar 一致性校验 | 移除校验调用 | 3 个断言抛错的用例失败，2 个正向用例仍通过 |
| dpu_id 映射到不同 vpu | 退回按整个资源列表取模 | 报 `1 == 1`（两个 dpu_id 挤在同一个 vpu 上） |
| qkv 宽度累加 | 退回按代表权重乘份数 | GQA 测试报 `96 == 64`，非 GQA 测试仍通过 |
| 成本精化侧的 op_id 校验 | 移除校验调用 | 对应测试失败，且从警告可见它真的走到了 FlagGems 编译 |

这一步不是形式主义：第一版 trace 测试用 `MagicMock` 当 backend 就是**假通过**——`_prepare_pim_traces` 在 backend 的**类**上查 `load_pim_trace`，而 MagicMock 只在实例上自动造属性，于是收集循环直接跳过、函数提前返回，根本没走到被测的检查。改用一个最小的真实类才真正覆盖到。

改造测试文件时删掉的内联代码里有一处流水切分下的真实错误：它取第 0 层的 KV 规格套用到全部 32 层。纯张量并行时所有层的 DPU 集合相同，这样做碰巧正确；流水切分下不同层在不同 DPU 上，必须按段分别取。`compile_llama2` 内部用的 `kv_specs_from_strategy` 已经处理对了，所以改成调用它同时也修掉了这个隐患。
