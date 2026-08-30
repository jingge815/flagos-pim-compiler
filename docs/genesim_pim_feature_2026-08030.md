# GeneSim 与 PIM 编译链路修改总结

日期：2026 年 8 月 30 日

本文总结以下三个仓库当前工作区中的相关未提交修改：

1. /media/disk/fengjingge/src/flagOS/flagOS-installers/FlagTree
2. /media/disk/fengjingge/src/genesim
3. /media/disk/fengjingge/src/flagOS/flagos-pim-compiler

本文以三个仓库执行 git diff 得到的内容为准。GeneSim 中的 .venv 是本地环境目录，不属于本次技术修改，以下不作记录。

## 一、修改概述

本次修改把“图层编译器的张量放置”和“算子编译器的 PIM 成本”进一步接入 GeneSim，主要解决以下问题：

- 真实 FlagGems 线性算子的完整矩阵尺寸 M/N/K 是运行期参数，无法只从 tt.dot 外层循环推断。
- GPU 自动调优得到的切分尺寸不一定满足 PIM 的 WRAM 容量和 DMA 对齐约束。
- GeneSim 原有模型把大部分 GEMM 默认放在 GPU，而图层编译器已经能够判断部分线性层应下沉到 DPU。
- GeneSim 的 PIM 指令表没有通用 GEMM 指令轨迹，编译器放置到 PIM 的 GEMM 不能继续走原来的轨迹查找路径。
- 三个仓库原先分别维护部分硬件默认值，且曾经依赖两份 FlagTree 安装，容易产生参数或二进制不一致。

本次形成的主链路如下：

~~~mermaid
flowchart TD
    A[真实模型与运行期算子参数] --> B[FlagGems 捕获真实启动参数]
    B --> C[FlagTree 转换为 PIM 中间表示]
    C --> D[读取完整 M N K]
    D --> E[按 WRAM 与 DMA 约束搜索 tile]
    E --> F[生成 PIM 中间表示与 tile 元数据]
    F --> G[分析浮点运算量与内存搬运量]
    G --> H[回填 GeneSim IR 成本系数]
    B --> I[图层编译器传播张量放置]
    I --> J[导出 GEMM 放置文件与侧车文件]
    J --> K[GeneSim 读取放置侧车文件]
    K --> L[固定算子到指定 PIM 资源]
    H --> M[GeneSim 运行时调度]
    L --> M
    M --> N[GEMM 使用屋顶线模型估算延迟]
    M --> O[注意力算子继续使用原有 PIM 轨迹]
~~~

本次修改不是把所有算子都改成 PIM。当前放置导出只覆盖图层编译器白名单中的 GEMM，GEMV_SCORE、SOFTMAX、GEMV_CONTEXT 仍沿用 GeneSim 原有的注意力算子路径。

## 二、跨仓库接口

| 数据 | 产生位置 | 使用位置 | 含义 |
| --- | --- | --- | --- |
| M/N/K | FlagGems 启动时的标量参数 | FlagTree 的 pim-tile-to-budget | 未切分的完整矩阵尺寸 |
| pim.tile-m/n/k | FlagTree | 成本分析和侧车文件 | 按 PIM WRAM 预算选出的 tile 尺寸 |
| pim.tile-wram-bytes | FlagTree | 成本分析和侧车文件 | 一个 tile 的 WRAM 占用 |
| flops_coeffs | 本仓 cost_extractor.py | GeneSim 调度器 | 按 Tq 或 Tq(Tp+Tq) 求值的运算量系数 |
| data_bytes_coeffs | 本仓 cost_extractor.py | GeneSim 调度器 | 算子对外净读写字节数系数 |
| mram_traffic_bytes | 本仓 ir_cost.py | PIM 成本侧车文件 | tile 重复装载产生的 MRAM 与 WRAM 搬运量 |
| device_hint=pim | 本仓 placement_export.py | GeneSim .ir | 图层编译器判定该 GEMM 使用 PIM |
| dpu_id | 本仓放置侧车文件 | GeneSim 调度器 | 图层编译器内部的 DPU 编号 |
| scheduler.compiler_placement_file | GeneSim 配置 | GeneSim 调度器 | 编译器放置侧车文件路径 |

放置编号存在两个编号空间：图层编译器的 dpu_id 是 0..num_dpus-1，GeneSim 的 PIM 资源还包含设备、VPU 和 TensorPU 层级。本次 GeneSim 侧先把图层编译器的 DPU 编号按可用 PIM 资源数量取模，转换为实际资源。

## 三、FlagTree 修改

### 3.1 文件变化

本仓库 git diff 统计：4 个已修改文件，新增 100 行，删除 18 行。

| 文件 | 状态 | 修改内容 |
| --- | --- | --- |
| include/triton/Dialect/TritonPIM/Transforms/Passes.td | 修改 | 为 pim-tile-to-budget 增加 full-m、full-n、full-k 选项及说明 |
| lib/Dialect/TritonPIM/Transforms/TileToBudget.cpp | 修改 | 支持完整尺寸覆盖，按 WRAM 和 DMA 约束搜索合法 tile |
| python/src/passes.cc | 修改 | 暴露带三个完整尺寸参数的 add_tile_to_budget Python 接口 |
| python/triton/backends/pim_sidecar.py | 修改 | 增加 MRAM、DMA 参数，并接入预算切分 pass |

### 3.2 函数和接口变化

| 函数或接口 | 变化 | 说明 |
| --- | --- | --- |
| inferFullShape(dot) | 修改为 inferFullShape(dot, overrideM, overrideN, overrideK) | 参数非负时覆盖对应维度；仍保留未提供维度的结构推断 |
| createTritonPIMTileToBudget({fullM, fullN, fullK}) | 新增调用参数 | 由 pass 选项承接 full-m、full-n、full-k |
| passes.pim.add_tile_to_budget(pm, full_m, full_n, full_k) | 修改 | Python 接口增加三个默认值为 -1 的完整尺寸参数 |
| pim_options() | 修改 | 增加 mram_bytes 和 dma_align |
| make_pimir(ttir_mod) | 修改 | 先执行预算切分，再执行显式 DMA；只对含 tt.dot 的模块执行预算切分 |
| emit_pim_ir(...) | 间接修改 | 继续调用 make_pimir，因此自动使用新的预算切分链路 |

### 3.3 原理

FlagTree 的预算切分流程是：

~~~mermaid
flowchart LR
    A[TTIR 中的 tt.dot] --> B{是否提供完整 M N K}
    B -- 是 --> C[使用运行期捕获的尺寸]
    B -- 否 --> D[从 scf.for 结构推断]
    C --> E[检查完整算子是否超过 MRAM]
    D --> E
    E -- 超过 --> X[报错]
    E -- 未超过 --> F[枚举不大于可见 tile 的二次幂候选]
    F --> G{WRAM 总量和三个缓冲区是否满足 DMA 对齐}
    G -- 否 --> F
    G -- 是 --> H[选 tile 元素数最多者]
    H --> I[必要时改写 M N K 三层循环]
    I --> J[写出 tile 元数据]
    J --> K[pim-explicit-dma]
~~~

核心约束如下：

~~~text
tile_wram_bytes = (M_tile*K_tile + N_tile*K_tile + M_tile*N_tile) * 元素字节数

tile_wram_bytes <= WRAM 字节预算
x_bytes % DMA 对齐 == 0
w_bytes % DMA 对齐 == 0
out_bytes % DMA 对齐 == 0
~~~

候选 tile 的三个维度都从二次幂除数中枚举。选择时优先使 M_tile*N_tile*K_tile 最大，也就是尽量减少循环次数；元素数相同的情况下优先保留更大的 K，其次是 N，最后是 M。MRAM 检查使用完整算子的 footprint，WRAM 检查使用单个 tile 的 footprint，两者职责不同。

pim-tile-to-budget 必须排在 pim-explicit-dma 之前：前者负责先把 tile 切小，后者根据最终 tile 建立 WRAM 暂存区并检查预算。如果顺序相反，显式 DMA 会先因原始大 tile 超预算而失败。

对于没有 tt.dot 的逐元素算子，预算切分 pass 会直接报“至少需要一个 tt.dot”。因此 pim_sidecar.py 只在 TTIR 文本包含 tt.dot 时加入该 pass，SOFTMAX、GELU 等算子仍可正常生成 PIM 中间表示。

## 四、本仓 flagos-pim-compiler 修改

### 4.1 文件变化

本仓已跟踪文件的 git diff 统计：11 个文件，新增 284 行，删除 140 行。另有 2 个本次指定纳入的未跟踪文件，共 252 行；.venv 不纳入。

| 文件 | 状态 | 修改内容 |
| --- | --- | --- |
| README.md | 修改 | 更新为单一 FlagTree 安装方式，删除 FLAGTREE_PIM_PREFIX 配置说明 |
| contracts/op_contract.py | 修改 | 增加全仓唯一的 DEFAULT_HARDWARE_CONFIG |
| docs/genesim_bridge.md | 修改 | 同步桥接流程、PIM 中间表示和单安装说明 |
| genesim_bridge/cost_extractor.py | 修改 | 记录预算切分元数据，调整融合注意力交叉验证探针 |
| genesim_bridge/env.py | 修改 | 删除运行时切换第二份 FlagTree 的逻辑，保留 CUDA 工具链准备和 PIM pass 检查 |
| genesim_bridge/flagtree_driver.py | 修改 | 传递完整 M/N/K，接入预算切分开关和 WRAM 覆盖参数 |
| genesim_bridge/ir_cost.py | 修改 | 解析 MRAM、DMA 对齐和 tile 元数据 |
| genesim_bridge/paths.py | 修改 | 统一单一 FlagTree 路径；tasklet、WRAM、MRAM、DMA 对齐默认值改由契约派生，桥接层 DPU 数默认值仍为 1 |
| opcompiler_bridge/driver.py | 修改 | 使用统一 FlagTree 路径，self-test 从统一硬件配置派生参数 |
| tests/test_genesim_bridge.py | 修改 | 只有两侧都存在时才比较 max_seq、vocab_size，兼容不同 IR 形态 |
| tests/test_opcompiler_e2e_llama2_7b.py | 修改 | 端到端测试只保留默认的 4 个 tasklet，小形状测试覆盖边界 |
| genesim_bridge/placement_export.py | 新增 | 导出图层编译器判定的 GEMM 放置结果 |
| tests/test_placement_export.py | 新增 | 验证 GEMM 放置导出和结构错误检查 |

### 4.2 函数新增和修改

| 文件 | 函数 | 变化 |
| --- | --- | --- |
| contracts/op_contract.py | DEFAULT_HARDWARE_CONFIG | 新增统一硬件配置：8 个 DPU、16 个 tasklet、每个 DPU 8 GiB MRAM、64 KiB WRAM、8 字节 DMA 对齐 |
| genesim_bridge/flagtree_driver.py | lower_ttir_to_pimir | 增加 full_m、full_n、full_k、tile_to_budget、wram_bytes 参数 |
| genesim_bridge/flagtree_driver.py | capture_kernels | 从启动参数提取 M/N/K，并将新参数传给 PIM 降级流程 |
| genesim_bridge/flagtree_driver.py | run_and_capture | 透传预算切分和 WRAM 覆盖参数 |
| genesim_bridge/cost_extractor.py | _pim_kernel_dict | 增加 MRAM 预算、DMA 对齐、tile 尺寸和 tile WRAM 占用 |
| genesim_bridge/cost_extractor.py | _cross_validate | 融合 flash attention 探针跳过线性 tile 选择，并使用 256 KiB 探针 WRAM 预算 |
| genesim_bridge/ir_cost.py | KernelCost | 增加 mram_bytes_budget、dma_align、tile_m/n/k、tile_wram_bytes 字段 |
| genesim_bridge/ir_cost.py | analyze_ir | 从 PIM 中间表示的模块属性解析上述字段 |
| genesim_bridge/paths.py | pim_options | 从统一硬件契约派生 tasklet、WRAM、MRAM、DMA 对齐参数，并支持 MRAM、DMA 环境变量；桥接层 DPU 数仍保留 1 的默认值 |
| genesim_bridge/paths.py | flagtree_pim_prefix、flagtree_pim_site_packages、flagtree_pim_nvidia_backend | 删除第二份 FlagTree 安装相关函数 |
| genesim_bridge/env.py | prepare_triton_env | pim 参数保留兼容旧调用，但不再切换 sys.path，只准备 cuda.h 和 ptxas |
| genesim_bridge/placement_export.py | _get_attr_node | 新增，按权重路径严格查找唯一 get_attr 节点 |
| genesim_bridge/placement_export.py | export_placement_to_genesim | 新增，按每层 4 个 GEMM 的固定顺序，把图层编译器放置写入 GeneSim .ir 和放置侧车文件 |
| opcompiler_bridge/driver.py | _triton_opt、_mlir_translate | 修改为使用统一 FlagTree 安装路径 |
| opcompiler_bridge/driver.py | _selftest | 修改为从 DEFAULT_HARDWARE_CONFIG 派生测试硬件参数 |

### 4.3 成本提取原理

成本提取默认从 PIM 中间表示进行：

~~~mermaid
flowchart TD
    A["GeneSim 图骨架"] --> B["按算子类型和形状缓存测量"]
    B --> C["Prefill：Tq=seq_len，Tp=0"]
    B --> D["Decode：Tq=1，Tp=seq_len"]
    C --> E["FlagGems + FlagTree 捕获与分析"]
    D --> E
    E --> F{"算子类型"}
    F -- GEMM --> G["拟合 a*Tq+b"]
    F -- 注意力算子 --> H["拟合 a*Tq*(Tp+Tq)+b"]
    G --> I["写入 flops_coeffs 和 data_bytes_coeffs"]
    H --> I
    E --> J["记录 PIM tile 和 MRAM 搬运元数据"]
    I --> K["改写 GeneSim IR"]
    J --> L["写入扩展侧车 JSON"]
~~~

具体规则如下：

- 不只回填标量 flops 和 data_bytes，而是回填 GeneSim 实际使用的符号系数；标量字段置零，避免被误用。
- GEMM 使用两个代表点拟合 a*Tq+b，常数项主要包含与查询长度无关的权重读取。
- 注意力算子使用 a*Tq*(Tp+Tq)+b，沿用 GeneSim 原有符号项集合。
- 如果两点线性拟合得到负斜率或负常数，改用“大项点与原点拟合”，保证 GeneSim 后续屋顶线计算不会得到负成本。
- data_bytes 表示算子对外净读写量；GEMM 会补上权重字节，不能把 tile 内部重复搬运混入该字段。
- PIM 中间表示额外提供 mram_traffic_bytes、WRAM 使用量、DMA 数量和 tile 元数据，这些写入侧车文件，不改变 GeneSim 的对外 data_bytes 口径。

融合 flash attention 不是单个线性算子，包含多个 tt.dot，不符合 pim-tile-to-budget 当前的单线性算子模型。交叉验证时跳过 tile 选择，并把探针 WRAM 预算临时放宽到 256 KiB，只用于取得运算量和 MRAM 搬运量的量级基线，不代表真实硬件能把融合 kernel 放入 64 KiB WRAM。

### 4.4 放置导出原理

export_placement_to_genesim 对每层 GeneSim 子图要求恰好 4 个 GEMM，并按以下顺序与图层编译器权重节点对应：

| GeneSim GEMM 顺序 | 图层编译器权重路径 | 说明 |
| --- | --- | --- |
| 第 1 个 | self_attn.q_proj.weight | 代表合并的 QKV 投影 |
| 第 2 个 | self_attn.o_proj.weight | 输出投影 |
| 第 3 个 | mlp.gate_proj.weight | 代表 GeneSim 的 FC1，属于近似映射 |
| 第 4 个 | mlp.down_proj.weight | FC2 |

如果权重的设备是 DPU，就把对应 GeneSim 算子的 device_hint 改为 pim，并把 shard_map 中最小的 DPU 编号写入放置侧车文件。图层编译器一个 GEMM 实际可能分布在多个 DPU，但 GeneSim 当前一个算子只能绑定一个 PU，因此本次选最小 DPU 作为代表。

输出有两份：

- 改写后的 GeneSim .ir，供 GeneSim 直接加载。
- 放置侧车 JSON，格式为 operators -> op_id -> {device_hint, dpu_id}，供调度器读取。

## 五、GeneSim 修改

### 5.1 文件变化

GeneSim 已跟踪文件的 git diff 统计：2 个文件，新增 128 行，删除 6 行。另有本次指定纳入的未跟踪测试文件：scripts/test_compiler_placement.py。

| 文件 | 状态 | 修改内容 |
| --- | --- | --- |
| src/scheduler/gene_sim_scheduler.py | 修改 | 读取编译器放置侧车文件、拆分放置单元、固定算子到 PIM 资源、放宽被编译器明确放置的算子类型检查 |
| src/vpu/vpu.py | 修改 | PIM 上的有真实成本 GEMM 改用屋顶线延迟估算 |
| scripts/test_compiler_placement.py | 新增 | 测试放置侧车门控、类型校验和 GEMM 屋顶线执行路径 |

### 5.2 函数新增和修改

| 文件 | 函数 | 变化 |
| --- | --- | --- |
| src/scheduler/gene_sim_scheduler.py | GeneSimScheduler.__init__ | 增加 scheduler.compiler_placement_file 配置读取，默认空字符串时保持旧行为 |
| src/scheduler/gene_sim_scheduler.py | _load_compiler_placement | 新增，读取 {operators: {op_id: {dpu_id}}} 并返回算子到 DPU 的映射 |
| src/scheduler/gene_sim_scheduler.py | _build_adaptive_initial_placement | 新增编译器放置单元拆分和固定资源逻辑；未标注算子继续使用原有贪心放置 |
| src/scheduler/gene_sim_scheduler.py | _validate_autotune_placement | 编译器明确标注的算子可以绕过普通 PIM 类型白名单，但仍保留 VPU、节点和 PU 范围检查 |
| src/vpu/vpu.py | PIMVPU.execute | 注意力算子继续走参数化 PIM 轨迹；有正成本的 GEMM 走屋顶线模型；其余类型继续走后端执行 |
| src/vpu/vpu.py | _estimate_roofline | 新增，统一执行阶段和放置搜索阶段的 PIM 延迟公式 |

GeneSim 调度器的放置流程如下：

~~~mermaid
flowchart TD
    A[GeneSim 图与算子分组] --> B{是否配置 compiler_placement_file}
    B -- 否 --> C[保持原有放置单元和贪心搜索]
    B -- 是 --> D[读取 op_id 到 dpu_id 的映射]
    D --> E[把被标注算子从组合单元中拆成单独单元]
    E --> F[按可用 PIM 资源数量转换 dpu_id]
    F --> G[直接固定资源，不参与贪心比较]
    E --> H[未标注单元继续正常搜索]
    G --> I[校验 VPU、节点、PU 范围]
    H --> I
    I --> J[生成放置后的 GeneSim IR]
~~~

### 5.3 GEMM 延迟原理

被编译器固定到 PIM 的 GEMM 没有可用的原生 PIM 指令轨迹，所以 PIMVPU.execute 使用与调度搜索相同的屋顶线公式：

~~~text
clock_s = clock_period_ns * 10^-9
peak_flops = 2 * multipliers / clock_s
peak_bandwidth = 1 / (LOADN_PER_BYTE * clock_s)
base_latency = 22 * clock_s

latency = max(flops / peak_flops, data_bytes / peak_bandwidth) + base_latency
~~~

这样可以保证“放置时估算的 GEMM 时间”和“实际执行阶段返回的 GEMM 时间”使用同一套口径。注意力三类算子仍然使用原有的参数化轨迹和后端接口。

## 六、测试与验证

### 6.1 本次实际执行

| 仓库 | 命令 | 结果 |
| --- | --- | --- |
| flagos-pim-compiler | source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_op_contract.py tests/test_genesim_bridge.py tests/test_placement_export.py -q | 26 项通过 |
| flagos-pim-compiler | source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_opcompiler_linear.py -q | 13 项通过 |
| flagos-pim-compiler | source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/test_placement_export.py -q | 2 项通过 |
| genesim | source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python scripts/test_compiler_placement.py | 5 项通过 |
| 三个仓库 | git diff --check | 均通过，无空白错误 |

放置导出测试覆盖：

- 一层 4 个 GEMM 的 device_hint 被正确改写为 pim。
- 导出的 dpu_id 位于 [0, num_dpus)。
- 注意力算子和非 GEMM 算子不被改写。
- 每层 GEMM 数量不符合约定时直接抛出错误。

GeneSim 新增测试覆盖：

- 未设置放置侧车文件时，默认启发式行为不变。
- 被侧车文件标注的 GEMM 固定到指定 PIM VPU。
- 未标注、手动放到 PIM 的 GEMM 仍会被类型白名单拒绝。
- PIM GEMM 不调用没有轨迹的后端执行，而是使用屋顶线公式。
- 注意力算子仍调用原有运行时轨迹。

### 6.2 已由开发者完成的完整验证

根据 GeneSim 文档 /media/disk/fengjingge/src/genesim/docs/llama-2.md，Llama-2-7B 的完整流程已经验证通过，本次不重复执行。该流程包括：

~~~bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
./install.sh --skip-attacc
python scripts/model_parser.py \
  --model_name /media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf \
  --output models/llama2_7b.ir
./run.sh --trace --synthetic --seed 0 --num_requests 10 \
  --output traces/llama2_7b.trace
~~~

成本精化和结果检查命令如下：

~~~bash
python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_flagtree.ir \
  --sidecar models/llama2_7b_flagtree_extensions.json \
  --seq-len 128 --ir-level ttir

python scripts/refine_ir_with_flagtree.py \
  --ir models/llama2_7b.ir \
  --out-ir models/llama2_7b_pimir.ir \
  --sidecar models/llama2_7b_pimir_extensions.json \
  --seq-len 128 --ir-level pimir
~~~

侧车检查：

~~~bash
python - <<'PY'
import json
from pathlib import Path

pimir = json.loads(Path("models/llama2_7b_pimir_extensions.json").read_text())
assert pimir["ir_level"] == "pimir"
assert pimir["coverage"]["bridged"]
entries = [pimir["operators"][str(i)] for i in pimir["coverage"]["bridged"]]
assert any(e["measurements"]["prefill"].get("pim_kernels") for e in entries)
assert any((e["mram_traffic_bytes"] or {}).get("prefill", 0) > 0 for e in entries)
print("PIM 中间表示元数据检查通过")
PY
~~~

默认模拟和结果检查：

~~~bash
./run.sh

python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("results/summary.json").read_text())
assert summary["completed_requests"] == 10
print("模拟完成 10/10 个请求")
PY
~~~

### 6.3 本仓完整测试命令

本仓完整测试命令由开发者已验证通过，本次不重复执行：

~~~bash
cd /media/disk/fengjingge/src/flagOS/flagos-pim-compiler && \
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && \
python -m pytest tests/ -x -q
~~~

该命令是本仓默认的全量测试入口，覆盖契约、图拆分、规格传播、通信、内存规划、运行时执行器、算子编译桥接和放置导出等测试。

### 6.4 FlagTree 回归测试命令

FlagTree 的预算切分相关回归用例位于以下文件：

- test/Dialect/TritonPIM/tile_to_budget_linear.mlir
- test/Dialect/TritonPIM/tile_to_budget_negative.mlir
- test/Dialect/TritonPIM/tile_to_budget_small_wram.mlir
- test/Dialect/TritonPIM/tile_to_budget_m_split.mlir

构建 FlagTree 后，可在 FlagTree 仓库根目录执行：

~~~bash
cmake --build build/flagtree-cmake --target triton-opt -j
cmake --build build/flagtree-cmake --target check-triton-lit-tests -j
~~~

也可以使用 triton-opt 直接验证预算切分用例。以下参数分别覆盖正常路径、M 维切分、显式 DMA 联合路径和超预算报错路径：

~~~bash
TRITON_OPT=/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree/build/flagtree-cmake/bin/triton-opt

$TRITON_OPT test/Dialect/TritonPIM/tile_to_budget_linear.mlir \
  -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=65536 mram-bytes=4294967296 dma-align=64' \
  -pim-tile-to-budget

$TRITON_OPT test/Dialect/TritonPIM/tile_to_budget_m_split.mlir \
  -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=32 mram-bytes=4294967296 dma-align=8' \
  -pim-tile-to-budget -pim-explicit-dma

$TRITON_OPT test/Dialect/TritonPIM/tile_to_budget_small_wram.mlir \
  -convert-triton-to-pim='target=pim:v1 num-dpus=1 num-tasklets=4 wram-bytes=65536 mram-bytes=4294967296 dma-align=64' \
  -pim-tile-to-budget -pim-explicit-dma
~~~

## 七、已知不足和风险

1. 放置导出尚未接入默认端到端入口。当前 export_placement_to_genesim 和 GeneSim 的 compiler_placement_file 读取分支均已实现并通过单元测试，但仓库搜索未发现默认脚本自动调用导出函数或自动写入该配置。要在完整工作流启用放置接管，还需要增加明确的编排步骤。

2. 一个 GEMM 多 DPU 分片被简化为一个代表 DPU。图层编译器的 shard_map 能表示一个算子分布在多个 DPU，但 GeneSim 当前的 Operator.attached_pu_id 只能绑定一个 PU。本次选最小 DPU，未使用 GeneSim 的 split_operator 精确还原张量并行，也未建模分片间通信。

3. GEMM 对应关系包含模型结构近似。GeneSim 把 QKV 合并为一个 GEMM，把真实的 gate_proj、up_proj 两路结构近似为单个 FC1。因此放置导出使用 q_proj.weight 和 gate_proj.weight 作为代表，并不等于完整还原真实 Llama 计算图。

4. 预算切分当前只支持固定的二维线性 tt.dot 模式。pim-tile-to-budget 要求二维输入、输出和累加结构，tile 维度为二次幂，并要求不同 tt.dot 使用一致的 tile。融合 flash attention 等复杂 kernel 只能作为成本交叉验证探针，不能直接使用该预算切分路径。

5. 完整尺寸依赖参数命名。捕获逻辑按 M、N、K 读取运行期标量。命名不同的 kernel 会回退到 -1，再依赖中间表示结构推断；如果结构也无法推断，预算切分会报错。后续若扩大算子范围，需要建立更明确的算子参数契约。

6. GeneSim 的 DPU 编号转换使用取模。当图层编译器的 DPU 数量与 GeneSim 可用 PIM 资源数量不一致时，当前采用 dpu_id % len(pim_resources)。这能避免越界，但不是物理拓扑的一一映射，正式硬件评估前应改为显式设备映射并校验编号范围。

7. 两点拟合不能表达自动调优的阶跃成本。真实 kernel 可能因为 tile padding 在短序列上产生远大于理论值的运算量。当前使用两点线性拟合，并把原始测量值保存在侧车文件；中间序列长度的成本仍可能有误差。

8. 全量验证依赖外部环境。Llama-2-7B 流程依赖本地模型、CUDA、FlagGems、已同步 PIM pass 的 FlagTree 和 GeneSim 依赖。当前会话没有重复执行该大模型流程，文档中的结论依据开发者已完成的 docs/llama-2.md 验证记录。

9. FlagTree 安装同步必须保持一致。修改 FlagTree pass 后需要重新构建并同步到统一安装；如果 triton-opt、Python 侧 libtriton.so 和 NVIDIA 后端来自不同构建，可能出现 pass 不存在、cuda.h 缺失或 ptxas 不匹配等问题。

## 八、结论

本次修改完成了三项关键能力：FlagTree 能使用真实启动尺寸按 PIM 硬件预算选择 tile；本仓能提取 PIM 中间表示中的成本和切分元数据，并导出图层编译器的 GEMM 放置；GeneSim 能在配置侧车文件后固定编译器指定的 GEMM，并以统一屋顶线模型执行成本估算。

当前最重要的后续工作是把放置导出正式接入 Llama-2-7B 默认精化和模拟入口，并将“一个 GEMM 对应多个 DPU”的简化映射替换为真实张量并行与通信建模。
