# flagos-pim-compiler

存算一体（PIM）大模型推理编译器 —— FlagOS 生态下的图层编译器与主机编排运行时。

把 HuggingFace 模型经 `torch.export` 拿到静态图后，在编译期完成图拆分、切分传播、通信计划与三区内存规划，产出一组静态蓝图；运行时由主机编排器在 NumpyBackend（多 DPU 独立地址空间模拟）上驱动执行，并与单卡 PyTorch 逐节点数值对齐。

本仓覆盖方案中的问题 1/2/3/6/7/8 + GeneSim 成本桥接（问题 4）。算子编译器（问题 5，`ttir→pim mlir`）在 [FlagTree](https://github.com/) 中开发，只通过 kernel 二进制与编译期算子契约与本仓对接。

## 在 FlagOS 工作区中的位置

本仓是 PIM 工作区（`pim/` 元仓，`manifest.yaml` 逐组件钉 commit）的核心组件，上下游关系如下：

| 组件 | 职责 | 与本仓的接口 |
| --- | --- | --- |
| **flagos-pim-compiler**（本仓） | 图层编译 + 主机编排运行时 | — |
| **FlagTree** | 算子编译器（问题 5）：`ttir → pim mlir → EmitC → C → .so` | 本仓 `opcompiler_bridge/driver.py::compile_op` 调 `triton-opt` / `mlir-translate` / gcc，拿回 `.so` 与编译期回传属性（tile MRAM 占用、相位数、元素宽度） |
| **genesim** | 离散事件仿真器：吞吐、延迟、利用率 | 本仓写出 `.ir` 成本系数、placement 文件、pim mlir；GeneSim 不回写本仓 |
| **gml_reference** | GML 参考产物（结构与数值真源） | 只读，供测试与导出脚本对拍 |
| **flagOS-installers** | FlagTree / FlagGems / PyTorch PIM 栈的安装脚本 | 提供 `flagtree_prefix` 等站点路径 |

反方向的唯一接口是 `contracts/partition_plan.py`：GeneSim 侧做完 placement 搜索后，用 `graph/strategy.py::strategy_from_partition_plan` 把结果变成 `ShardStrategy`。其余跨仓数据都是本仓单向产出。

## 整体架构

一张带标注的 FX 图，三条下游支线：

```
LlamaForCausalLM
   │  runtime/compile.py::export_annotated_graph
   │    torch.export.export(strict=True).module() + graph/partition.py::partition_graph
   ▼
统一 IR：FX GraphModule + node.meta 四维（device / part_id / spec / stage）
   │
   ├── A. 运行时蓝图 ──────────────────────────────────────────────
   │     graph/spec_prop.py::propagate_specs        切分传播 + redistribute 标注
   │     comm/plan.py::build_comm_plan              每个 redistribute 边 → DMA 计划
   │     memory/kv_layout.py::build_kv_layout       KV 物理槽池 + 块表几何
   │     memory/mem_planner.py::plan_dpu            三区 offset（权重/KV/激活）
   │     runtime/exec_plan_gen.py::build_execution_plan → ExecutionPlan（prefill + decode）
   │     执行：executor.step_request → hal.submit → runtime/kernels.py 的 aten 内核
   │
   ├── B. GML 导出 ────────────────────────────────────────────────
   │     gml_bridge/export.py::fuse_for_gml        六个融合/拆头 pass
   │     gml_bridge/from_fx.py::convert            FX → GML 节点/边/权重参数
   │     gml_bridge/writer.py::write_gml           relay2gml_graph.gml + *.bin + IO_info.txt
   │     orchestrator/plan.py::orchestrate         prepare_out/net.ini + txt_files/
   │
   └── C. GeneSim 桥接 ────────────────────────────────────────────
         genesim_bridge/flagtree_driver.py         TTIR / 算子级 mlir → pim mlir
         genesim_bridge/ir_cost.py::analyze_ir     抽 flops/data_bytes
         genesim_bridge/cost_extractor.py          回填 .ir 系数 + sidecar
         genesim_bridge/placement_export.py        导出 placement 文件
```

三条支线共享同一张图，但**用途不同**：A 跑在 NumpyBackend 上做数值对齐，B 交给 GML 工具链，C 给 GeneSim 做性能评估。A 走的是**未融合**的 aten 图（SDPA 仍是一个节点，RMSNorm 仍是一条 aten 链），融合只在 B 支线做——GML 没有独立激活节点的表达能力，融合是目标格式的硬要求。

### 编译期主线：带标注的统一 IR

`contracts/unified_ir.py` 是字段约定的唯一真源：`META_KEYS`（:66）登记 `device` / `part_id` / `spec` / `stage` 四维，`spec_of`（:184）取切分说明，阶段常量与推进顺序在 :198-225，`validate_node_dimensions`（:271）校验每个节点四个维度齐全。

| 步骤 | 做什么 | 代码 |
| --- | --- | --- |
| 1. 导出 | `torch.export` 出 FX 图；按 `HOST_ONLY` 黑名单把节点标成 host 或 device，再按 DPU 做并查集连通分量分组 | `graph/partition.py::partition_graph`（:107） |
| 2. 定策略 | 权重按 `LLAMA_WEIGHT_RULES` 切分（`q/k/v/gate/up/lm_head` 列切，`o/down` 行切），得出每层的 `Shard`/`Replicate` | `graph/strategy.py::llama_strategy`（:115） |
| 3. 传播 | 沿数据流推 placement，产出 `PIMTensorSpec`（局部形状、`mram_offset`、步长、对齐）与 `RedistributeEdge` | `graph/spec_prop.py::propagate_specs`（:587） |
| 4. 通信计划 | 每个 redistribute 边展开成 DMA 收发序列 | `comm/plan.py::build_comm_plan`（:298） |
| 5. KV 布局 | 全局物理槽池 + 每 DPU 每窗口一份块表缓冲 | `memory/kv_layout.py::build_kv_layout`（:71） |
| 6. 内存规划 | 三区 offset、激活区按生命周期复用、多窗口步长 `act_stride` | `memory/mem_planner.py::plan_dpu`（:265）、`fit_activation_windows`（:358） |
| 7. 出蓝图 | 逐节点发 `Command`：`launch` / `dma_in` / `dma_out` / `host_reduce` / `host_concat` / `host_permute` / `host_slice` / `host_op` | `runtime/exec_plan_gen.py::build_execution_plan`（:353） |
| 8. 装权重 | 权重按策略切片写进各 DPU 的权重区 | `runtime/compile.py::write_weight_shards`（:365）、`load_weights`（:385） |

`compile_llama2`（`runtime/compile.py:202`）串起 1–7，产出 `CompiledModel`：`prefill` / `decode` 两个阶段的计划表（每阶段 P 份窗口计划）、`kv_specs`、`act_stride`、`block_table_bytes`、`kv_pool_slots`。第 8 步由调用方 `load_weights(compiled, hal)` 完成，权重区 offset 两张图共用。

几个关键约束：

- **地址全部烘死**。`mram_offset` 在编译期算定，运行时 `MRAMAllocator` 只做越界检查，不做分配。
- **未知设备算子直接抛**。`spec_prop` 遇到没规则可推的 device 节点就报错，不会安静退回 host（`graph/spec_prop.py:638` 附近）。静默回退会让编译器少算一块，也不会有任何提示。
- **依赖剪枝**。`_deps_of`（`runtime/exec_plan_gen.py:52`）对同一 `(loc, offset, length)` 只保留最后一个写者；`_PlanBuilder.append`（:212）对 DPU 泳道只保留最大命令号（同 DPU 命令天然按提交序执行），host 命令则每条依赖都留。
- **并行窗口**。`bind_plan`（`runtime/binding.py:39`）只平移激活地址与 `sdpa.dpus[*].bt_base`，永不平移 `payload["mram_offset"]`——那个字段是内核编译缓存键，靠它让所有窗口共享同一份 `.so`。

### 运行时：蓝图 → 设备命令

`runtime/executor.py` 维护请求、KV 槽与解码循环：

- `RequestTable` / `RequestSlot`：请求槽是 host 侧并发记账，不参与设备寻址。
- `KVSlotPool` / `KVBlockTable`：全局物理槽池；每请求一张逻辑块 → 物理槽的映射。
- `step_request`（:138）：写块表 → `execute_plan` → 等输出命令 → 采样，一次只出一个 token。
- `run_one_step`（:214）：活跃请求按 `parallel_slots` 分 wave，wave 内各用各的 `slot_index` 窗口并发，wave 之间串行。
- `execute_plan`（:258）：按计划序把所有命令都提交出去，主线程只在最后统一 `wait`。

## 五份产物：“IR”到底指哪一个

仓里“IR”被用在五种不同的东西上，混起来会看不懂代码：

| 名称 | 形态 | 生产者 | 消费者 | 用途 |
| --- | --- | --- | --- | --- |
| **统一 IR** | FX `GraphModule` + `node.meta` 四维 | `runtime/compile.py::export_annotated_graph` + `graph/partition.py` + `graph/spec_prop.py` | 本仓全部下游 pass、GML 桥、GeneSim 桥 | 编译期主干，字段约定在 `contracts/unified_ir.py` |
| **ModelIR** | `.ir` JSON（算子/边/形状/成本系数） | GeneSim 的 `--parse-model`；本仓 `genesim_bridge` 回填 `flops_coeffs` / `data_bytes_coeffs` | GeneSim 调度器 | 性能仿真的模型输入 |
| **pimir** | pim mlir 文本 | FlagTree（本仓 `genesim_bridge/flagtree_driver.py` 驱动） | GeneSim 的 PIM 指令编译与成本估算 | 设备侧算子的中间表达 |
| **蓝图** | `ExecutionPlan`（`Command` 列表） | `runtime/exec_plan_gen.py::build_execution_plan` | `runtime/executor.py` + `backend/hal_numpy.py` | 运行时命令流；内存对象，host 命令带 Python 闭包，不能跨进程序列化 |
| **GML** | `relay2gml_graph.gml` + `*.bin` + `IO_info.txt`（另加 `prepare_out/net.ini`、`txt_files/`） | `gml_bridge/export.py` + `orchestrator/` | GML 工具链、参考产物对拍 | 交付给硬件侧的目标格式 |

GML 导出（单层 decode block，`--layers 1 --seq-len 16`）的参考规模：200 节点 / 331 边 / 3122 个运行时文件（`docs/pim-compiler-v0.0.9.md:579`），编排后展开成 422 层（`orchestrator/layer_expand.py::expand_layers`）。

文档里常见的 `DPU_k.plan` **不是文件**：它指 `memory/mem_planner.py::plan_dpu` 返回的 `DPUPlan`（问题 8 的内存蓝图，可用 `format_mem_plan` 打印）。同理 `format_comm_plan`、`format_kv_layout` 分别打印通信计划与 KV 布局。

## 算子：两种口径

“算子”在本仓有两个层次，`contracts/op_semantics.py` 是两层共用的登记表：`OP_SEMANTICS`（:40）共 26 条，其中 17 条有算子编译器内核入口（`has_kernel=True`），14 条进 GeneSim 助记符统计（`is_mnemonic=True`）。四份视图全部由它派生：`oplevel_ops()`、`mnemonics()`、`aten_to_gml()`、`role_to_gml()`；模块导入时 `_validate_registry()` 会检查重名与 aten 目标冲突。

**第一层：aten 目标。** `torch.export` 出来的图里，节点就是 `aten.xxx`。运行时按 aten 目标查 `runtime/kernels.py::_KERNELS`（:1664，登记 34 个目标），找到内核函数在 DPU 线程里跑；`register_all`（:1715）把这张表注册进 HAL。缺一个名字，`hal.submit` 当场找不到内核。

**第二层：算子级助记符。** 编译到硬件时，一串 aten 会被折成一个 PIM 算子（如 `pow/mean/add/rsqrt/mul` → `normalize`），助记符就是 `pim.<name>`。这层是算子编译器的输入、也是 GeneSim 的成本口径。

| aten 目标 | 语义条目 | 运行时内核 | GML op_type | 助记符 |
| --- | --- | --- | --- | --- |
| `rsqrt.default` | `normalize` | `rsqrt_kernel` | `RMSNorm_vpu` | `pim.normalize` |
| `scaled_dot_product_attention.default` | `matmul` | `sdpa_kernel`（内部走 `paged_attn`） | `MatMul` | `pim.matmul` |
| `add.Tensor` | `eltwise_add` | `add_kernel` | `EltwiseAdd` | `pim.eltwise` |
| `linear.default` | `gemm` | `linear_kernel`（`use_compiled_linear=True` 时换 `compiled_linear_kernel`） | `Gemm` | — |

三个容易踩的点：

- `eltwise` 一条条目服务 GML 的四种逐元素（Add/Sub/Mul/Div），`silu` 走 `lut` 内核 —— GML 名与内核入口不是一一对应。
- `reduce` / `convert` / `paged_attn` 有内核入口但**不进**助记符统计：`reduce` 折进 `RMSNorm_vpu`，`paged_attn` 折进 GML 的 MatMul/Softmax 链，`convert` 是 GML 里的独立节点（类型转换是真实的数据运动）。
- `runtime/kernels_pim.py` 是算子级助记符的 numpy 镜像，公式只认 `gml_bridge/phase_data.py`，供编译内核逐元素对拍。

## NumpyBackend：HAL 是什么

它不是硬件模拟器（性能交给 GeneSim），而是一个**带硬件约束的 x86 执行底座**：用 numpy 数组冒充每台 DPU 的 MRAM/WRAM，把“编译期定地址、运行期只搬数据”这条约束真的执行一遍。换后端不换数值，`hal_vendor.py` 是留给真硬件的同接口桩（当前抛 `NotImplementedError`）。

三层栈：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 设备 SDK 镜像 | `backend/dpu_sdk.py` | `dpu_alloc` 造出每台 DPU 的 `mram` / `wram` 字节数组（独立地址空间），提供 `dpu_copy_to/from`、`prepare_xfer`/`push_xfer`、`broadcast_to` |
| 异步 HAL | `backend/hal_numpy.py` | `NumpyBackend`（:110）把 `Command` 提交到线程池并返回 `Event`（:23）；地址只做越界检查 |
| 设备内核 | `runtime/kernels.py` | 用 HAL 原语写的 aten 算子实现；多 tasklet 内核用 `record_access` / `barrier` |

HAL 的四条语义决定了蓝图长什么样：

1. **泳道**。流按 `(lane_key, dpu_id)` 分键（`_stream_locks` / `_stream_tail`）：同一请求在同一 DPU 上严格 FIFO，不同请求（`ExecutionContext.lane_key` 不同）互不排队。`ExecutionContext`（:30）同时装输入值、`pos`、`num_tokens` 和事件表，让多个请求同时在飞。
2. **依赖**。每条命令在自己泳道里先等前一条，再等 `waits` 列出的命令（`_submit_on_dpu`，:281）。所以主机可以一次性把所有命令提交完，不必逐条前置等待。
3. **host 命令串行**。`host_reduce` 等带闭包的命令共用机器级 `xfer_buffers`，必须持 `_host_lock`（:145）逐条跑，否则两个并发 host op 会互相偷走对方的缓冲区。
4. **冲突检测**。`HazardTracker`（:65）在一次 `launch` 内记录各 tasklet 的 WRAM/MRAM 访问，不同 tasklet 在同一同步区间内重叠且至少一方写就抛 `TaskletHazardError` —— 缺 barrier 或区间算错都会当场暴露。

结论：`NumpyBackend` 保证的是**数值与约束**，不是性能。想测吞吐/延迟，走 GeneSim。

## 目录结构

| 目录 | 内容 | 对应任务 |
| --- | --- | --- |
| `contracts/` | 共享数据契约：`PIMTensorSpec`、`ExecutionPlan`、`node.meta` 四维、`OP_SEMANTICS`、`PartitionPlan` —— 全仓 import 的唯一真源 | 全组共用 |
| `graph/` | 图拆分打标、切分策略、切分传播 / redistribute 标注；GML 侧六个融合与拆头 pass | 问题 1、2 |
| `comm/` | 编译期通信计划表 + 运行时 redistribute→DMA 下降 | 问题 3 |
| `memory/` | KV 布局与块表、三区静态内存规划、激活窗口 | 问题 7、8 |
| `runtime/` | 编译入口、ExecutionPlan 生成、解码循环、aten 内核、窗口绑定 | 问题 6 |
| `backend/` | `dpu_sdk` SDK 镜像 / `hal_numpy` 异步 HAL / `hal_vendor` 真硬件桩；`vllm_worker`、`vllm_executor` 接 vLLM V1 | 运行时底座 |
| `opcompiler_bridge/` | 算子编译器桥：TTIR（A 路）/ 算子级 mlir（B 路）→ `triton-opt` → EmitC → gcc → `.so`，带 `.opcompiler_cache` | 问题 5 的下游 |
| `gml_bridge/` | FX → GML：融合、转换、序列化、`.bin` 与运行时文件、相位数据 | GML 交付 |
| `orchestrator/` | GML 产物编排：层展开、L2 分配、`net.ini` 与 `txt_files/` | GML 交付 |
| `quant/` | GML 要的 int4 权重 / int8 激活量化 | GML 交付 |
| `genesim_bridge/` | GeneSim 成本桥接与 pimir / placement 导出 | 问题 4 |
| `scripts/` | 导出与验证脚本：`export_gml.py`、`run_full_pipeline.py`、`serve_pim.py` 等 | — |
| `tests/` | 各模块单测与逐节点对拍 | 集成验证 |
| `examples/` | 预留 | — |
| `docs/` | 设计文档、接口说明、逐轮交付文档 | — |

## 一次 decode 的走读

以 vLLM 后端（`backend/vllm_worker.py::PIMWorker`）跑一步解码为例：

1. **编译**。`load_model`（:87）算出 `ShardStrategy`，调 `compile_llama2` 编两张图（prefill 固定 `prefill_seq_len`、decode 长度 1），得到 P 份窗口计划与 KV 几何；`load_weights` 把权重切片写进各 DPU 权重区。
2. **接请求**。`execute_model`（:169）把 `SchedulerOutput` 翻译成 `add_request` / `free_slot`，并按 `PIM_PARALLEL_SLOTS` 分 wave。
3. **备 KV**。`step_request` 先 `KVBlockTable.ensure` 向全局槽池要物理槽，再把逻辑块 → 物理槽写进每台 DPU 的第 `slot_index` 份块表。
4. **提交**。`ExecutionContext(lane_key=slot_index)` 带本请求的 `input_ids` / mask / `position_ids` / `pos` / `num_tokens`，`execute_plan` 按计划序 `hal.submit` 全部命令。
5. **调度**。HAL 把 `launch` 与 `dma_*` 投进 `(lane_key, dpu_id)` 泳道（先等同泳道前一条、再等 `waits`），host 命令进公共线程池并持 `_host_lock`。
6. **执行**。DPU 线程里调 `runtime/kernels.py` 的内核：从 MRAM 读分片、numpy 算、写回；`linear` 在 `register_all(..., use_compiled_linear=True)` 时换成算子编译器编出的 `.so`（`compiled_linear_kernel`，形状不支持时仍回退 numpy）。
7. **注意力**。`sdpa_kernel`（:1321）先按 `pos` / `num_tokens` 把本步 K/V 逐行写进物理槽，再逐头跑 `_paged_attention`（:1394）：KV 是 f16 且图上没有可用的 mask 时用融合内核 `pim.paged_attn`（单遍在线 softmax，逻辑块 → 物理槽在设备上解析），否则退回主机相位链路径。
8. **收结果**。`hal.wait(events[decode.output_cmd_id])` 取 logits，`argmax` 采样，`valid_len += 1`。
9. **回收**。请求结束 `free_slot` 归还 request slot 与全部物理槽。

同一 wave 内的其他请求在第 4–8 步同时进行：激活区与块表窗口互不相交，物理 KV 槽由槽池全局分配。超并行窗口的内存代价是激活区与块表各 ×P，超预算时编译期按 DPU 报错。

## 关键文件导航

| 想弄清/改动 | 看这里 |
| --- | --- |
| 节点四个维度怎么约定的 | `contracts/unified_ir.py` |
| 切分说明、redistribute 边的数据结构 | `contracts/pim_tensor_spec.py` |
| 命令与计划的字段 | `contracts/exec_plan.py` |
| 算子登记表（aten / 内核入口 / 助记符 / GML 名） | `contracts/op_semantics.py` |
| 哪些算子走 host、哪些走 device | `graph/partition.py`（`HOST_ONLY` 黑名单） |
| Llama 权重切分规则 | `graph/strategy.py`（`LLAMA_WEIGHT_RULES`） |
| placement 推导规则表 | `graph/spec_prop.py`（`RULE_TABLE`） |
| 三区 offset、激活复用、并行窗口 | `memory/mem_planner.py` |
| 物理槽池、块表、KV 区几何 | `memory/kv_layout.py` |
| 编译总入口 | `runtime/compile.py::compile_llama2` |
| 命令计划怎么生成、依赖怎么剪 | `runtime/exec_plan_gen.py` |
| 请求/槽/解码循环 | `runtime/executor.py` |
| aten 算子的设备实现 | `runtime/kernels.py` |
| 助记符算子的 numpy 镜像 | `runtime/kernels_pim.py` |
| HAL 语义（泳道、事件、冲突检测） | `backend/hal_numpy.py` |
| 算子编译器调用与缓存 | `opcompiler_bridge/driver.py::compile_op` |
| GML 导出总入口 | `gml_bridge/export.py::export_graph` / `export_llama2` |
| GML 六个 pass 的顺序与理由 | `gml_bridge/export.py::fuse_for_gml` |
| `prepare_out/` 编排 | `orchestrator/plan.py::orchestrate` |
| GeneSim 成本抽取 | `genesim_bridge/ir_cost.py::analyze_ir` |
| vLLM 接入 | `backend/vllm_worker.py`、`scripts/serve_pim.py` |

## 第 1 阶段目标

固定 shape 全链路打通，默认模型选定 `meta-llama/Llama-2-7b-hf`。编译器日常单测使用缩小的随机 LLaMA 配置避免下载权重，端到端验证加载官方 HuggingFace LLaMA2-7B 权重，在 NumpyBackend 上跑通 prefill + decode 路径并与单卡 PyTorch 逐元素对齐。GPT-2 只保留为显式 legacy/debug smoke。

## 环境

### GPU 不是必需的

算子编译产出的是 **pim mlir**（再往下是 EmitC → C → .so），全程在 CPU 上跑，不需要 GPU
硬件。纯 CPU 的 Ubuntu 22.04 上编译、安装、测试、运行全部可用，产出的 pim mlir 与有卡
机器**逐字节相同**。

纯 CPU 环境下唯一不可用的是"需要真实执行 Triton kernel"的那类步骤（详见
[`docs/pim-compiler-v0.0.4.md`](./docs/pim-compiler-v0.0.4.md) 第 7.1 节）：

从零搭建的完整步骤（含系统包、四个安装脚本、五项测试的预期输出）见
[`docs/pim-compiler-v0.0.4.md`](./docs/pim-compiler-v0.0.4.md) 第 5.0 节。

### 站点相关路径

本仓根目录的 `paths.json` 是入库模板；站点路径写到 `paths.local.json`（已被
gitignore）。甲方拿到代码后先复制再改：

```bash
cp paths.json paths.local.json
vim paths.local.json
```

```json
{
  "pytorch_env_script": "/path/to/env-pytorch.sh",
  "llama2_7b_model_dir": "/path/to/Llama-2-7b-hf",
  "flagtree_prefix": "/path/to/flagTree",
  "genesim_root": "/path/to/genesim",
  "gml_reference_dir": "/path/to/gml-reference",
  "gml_llama2_reference_dir": "/path/to/gml-reference/llama2_w4a8_decode_block_0/parser_output",
  "gml_ours_graph": "/path/to/our-decode-block/relay2gml_graph.gml"
}
```

配置完成后加载 PyTorch 环境并运行测试：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.local.json"))["pytorch_env_script"])')"
python -m pytest tests/ -x -q
```

`llama2_7b_model_dir` 只供真实 7B 端到端测试使用；未配置时这些测试会跳过，常规
单元测试仍可运行。`flagtree_prefix` 供算子编译器与 Triton 环境使用，`genesim_root`
供 GeneSim 产物读写使用。

后两个 `gml_*` 是 **GML 参考产物**（GML 结构参考与 llama2 W4A8 decode block 的
parser 输出），它不在四个安装脚本的产物里，需要单独放置。它们影响两类结构校验：

| 未配置时的现象 | 影响的检查 |
| --- | --- |
| `RuntimeError: 未配置站点路径 gml_llama2_reference_dir` | `tests/test_gml_hw_table.py`、`tests/test_runtime_files_phases.py` 等一组结构校验 |
| 打印「无参考产物，跳过」或「无参考目录，跳过」，校验静默通过 | `scripts/export_gml.py` 的 dtype 覆盖检查、`--orchestrate` 的文件族检查 |

也就是说，**不配这两个键，测试会报错，而导出脚本会悄悄少做两项校验**——后者更隐蔽，
建议一并配好。

`gml_ours_graph` 是本机一次 decode-block 导出，给逐节点对拍用；不配则相关测试跳过。

需要临时覆盖时，可设置同名环境变量：`PYTORCH_ENV_SCRIPT`、
`LLAMA2_7B_MODEL_DIR`、`FLAGTREE_PREFIX`、`GENESIM_ROOT`、`GML_OURS_GRAPH`。
优先级为环境变量 > `paths.local.json` > `paths.json` 模板。PIM 硬件参数（如 `FLAGTREE_PIM_NUM_DPUS`）
也沿用该优先级。
查看当前生效路径与参数：

```bash
python -c "from genesim_bridge.paths import describe; print(describe())"
```

GeneSim 侧的 `scripts/refine_ir_with_flagtree.py` 用 `PIM_COMPILER_ROOT` 指向本仓库
根目录；未设置时按常见相对布局探测（`../flagOS/flagos-pim-compiler` 或
`../flagos-pim-compiler`）。

### vLLM 版本

PIM 后端只支持 **vLLM 0.14.x**（当前基线与回归都是 0.14.0）。`PIMExecutor` 启动时会
校验 `vllm.__version__`，不在范围内直接失败并说明当前版本与支持范围，避免在接口不兼容
的版本上跑出难查的错。升级前先看
[`docs/vllm-031-gap-20261102.md`](./docs/vllm-031-gap-20261102.md)，那里列了 0.14 → 0.31
的接口差异与适配顺序。

```bash
pip install "vllm==0.14.*"
python -m pytest tests/test_vllm_compat.py -q
```

## 文档

- `docs/spec.md` — 完整技术方案
- `docs/spec-index.md` — 方案分节行号索引（按需定位，避免整篇读）
- `docs/pim-compiler-v0.0.3.md` — v0.0.3 交付文档（技术方案的完整描述以此为准）
- `docs/pim-compiler-v0.0.4.md` — v0.0.4 交付文档：TP/PP + PU 映射，以及纯 CPU 部署
  （第 5.0 节是从零搭建的完整操作清单）
- `docs/pim-compiler-v0.0.9.md` — v0.0.9 交付文档：图编译决策（融合/切分/内存排布）进入 GeneSim 计时
- `docs/hal_numpy.md` — HAL 接口与 tasklet 模型
- `docs/kv-paged-attn-online-20261026.md` — 融合分页注意力（单遍在线 softmax）
- `docs/executor-parallel-phase1-20261026.md` — 跨请求并行执行
- `docs/vllm-integration-20261102.md` — vLLM 接入现状索引（能力 → 代码 → 文档 → 测试）
- `docs/vllm-scheduling-20261102.md` — vLLM 调度超参影响表（预算、块大小、前缀缓存、分块预填充）
- `docs/executor-parallel-bottleneck-20261102.md` — 并行瓶颈测量（GIL 与命令数），G1.5 收口
- `docs/fused-rms-norm-20261102.md` — RMSNorm 链融合（934 → 718 条命令，token 逐字不变）
- `docs/runtime-windows-20261102.md` — 运行期窗口选择与设备端分配器接口设计
- `docs/qwen2-support-20261102.md` — 接入 Qwen2（偏置链路）与 GQA 缓存覆盖修复
- `docs/bench-regression-20261102.md` — 基准回归门限：`verify.sh --bench` 与基线文件
- `docs/weekly-20261102.md` — 周报（10-27 → 11-02）：与 10-26 汇报逐条对照的进度
- `docs/<module>.md` — 各模块接口与设计决策

## 开发约定

`contracts/` 是全仓地基，字段名须在开发初期钉死，之后各组在自己目录内并行推进，仅通过 `node.meta` 标注解耦。

开发规范（代码量控制、可读性、测试、文档要求）见 [`CLAUDE.md`](./CLAUDE.md)，人和 agent 共用同一份。
