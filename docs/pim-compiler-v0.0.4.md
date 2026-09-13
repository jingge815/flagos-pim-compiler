# 存算一体大模型推理编译器 v0.0.4

本文档接续 `pim-compiler-v0.0.3.md`，只写 **v0.0.3 → v0.0.4 的变化**。技术方案的完整
描述（图编译、算子编译、主机编排、内存管理、成本桥接的设计原理）仍以 v0.0.3 为准，
此处不重复。

两条主线：

1. **能力上**：GeneSim 升级到 v0.0.6，新增 TP/PP 切分与 PU 映射支持，打通
   `GeneSim → 图编译 → 算子编译 → GeneSim` 完整闭环，同时保持 NumPy 后端逐元素对拍。
2. **部署上**：去掉 GPU 硬件依赖，可在纯 CPU 的 Ubuntu 22.04 服务器上编译、安装、
   测试、运行。

## 1. 版本与代码量

| 仓库 | v0.0.3 | v0.0.4 | 提交数 | 代码量 |
| --- | --- | --- | ---: | --- |
| flagos-pim-compiler | `0517bac`（2026-09-01） | `b892298`（2026-09-06） | 9 | 26 文件，+5261 / -233 |
| genesim | `657b51b`（v0.0.3, 2026-08-04） | `281ebc2`（含 v0.0.6 合并） | 30 | 167 文件，+61910 / -5765 |
| FlagTree | `dc35b24df` | 同（未改动） | — | PIM pass 是 CPU 上的 MLIR 变换 |
| FlagGems | `41a26a35` | 同（未改动） | — | 只改调用方式，不改其源码 |

GeneSim 的 61910 行里约六成是 `tools/upmem_checker/`（真机核对工具，新增 DPU kernel
和测试生成器）。与本编译器直接相关的是 `src/` 下这几处：

```
src/scheduler/gene_sim_scheduler.py   +8934   TP 展开、PU 映射消费、pimir trace
src/ir/model_ir.py                    +1578   图骨架重写（见 2.1）
src/pim/pimir_trace.py                 +402   照 pim mlir 生成 PIM trace
src/pim/capacity.py                    +618   常驻内存容量检查
src/vpu/vpu.py                         +470   PIMVPU 语义改为单 TensorPU
tests/sim/test_compiler_placement.py  +1888   放置与映射的回归
```

## 2. 能力变化

### 2.1 GeneSim 升级到 v0.0.6：图骨架重写

上游提交 `ad22fcd` 重写了 `build_from_hf_config`，图骨架从"简化版"变成"贴合真实
Llama 结构"：

| 项目 | v0.0.3 的图 | v0.0.6 的图 |
| --- | --- | --- |
| 算子总数（7B） | 3232 | **3491** |
| 每层 GEMM 数 | 4 | **7** |
| q/k/v | 合并成一个 `4096→12288` | **三个独立 GEMM** |
| MLP | 只有 gate_proj / down_proj | **补上并列的 up_proj** |
| 残差与归一化 | 无 | 新增 VECTOR_ADD / RMSNORM / SILU / VECTOR_MUL |
| 图边界 | 无 | 新增 MODEL_INPUT / MODEL_OUTPUT |

这次升级连带断了五处（成本提取撞未知算子类型、放置错位、容量检查等），逐个修复的
记录在 `docs/genesim-v0.0.6-merge-20260903.md`。

另一处影响资源模型的变化：`PIMVPU` 现在只代表**一个 TensorPU**（512 MiB），不再代表
整个 PIM 设备。这直接决定了 PU 映射为什么必须按 ClusterPU 粒度分配——一段流水的权重
（tp1_pp8 下每段四层约 1544 MiB）钉到单个 TensorPU 上必然触发容量超限。

### 2.2 PU 映射：编译器与模拟器双向对齐

v0.0.3 时 GeneSim 靠 `dpu_id % len(cluster_keys)` 取模决定逻辑 DPU 落在哪个
ClusterPU——能跑，但落到哪几个是碰巧的。v0.0.4 引入声明式映射：

```
GeneSim 按物理约束挑 ClusterPU        scripts/export_fixed_pu_mapping.py
  ├─ 带宽：同 Cluster 内 512 GB/s，跨 Cluster 128 GB/s → 同段尽量同 Cluster
  └─ 容量：一段权重不能超过单个 ClusterPU 的 8 GiB
        │  dpu_to_cluster
        ↓
图编译器读方案、原样写回 sidecar      contracts/partition_plan.py（新增 195 行）
        │
        ↓
GeneSim 按声明分配资源                gene_sim_scheduler.py:3768
  声明的 ClusterPU 不存在 → 直接报错，不静默折回
```

**代价模型真的按它区分带宽**：`src/node/node.py:143-164`，同 cluster 走
`get_intra_cluster_link()`（4096 Gbps），跨 cluster 走 `get_inter_cluster_link()`
（1024 Gbps）。不是只写进 log。

### 2.3 TP/PP 切分：从图编译到模拟器代价

**图编译器侧**：`graph/strategy.py` 支持 TP×PP 混合策略，TP 宽度必须整除注意力头数
（GQA 下还要整除 KV 头数）。

**导出侧**：`genesim_bridge/placement_export.py`（+271 行）把 TP 组内**每台** DPU 都
写进 sidecar。v0.0.3 时只取编号最小的那台，导致 tp2_pp4 下 8 台 DPU 有 4 台在仿真里
完全空闲——那是失真，不是优化。

**模拟器侧**：`_expand_multi_shard_operators` 把多分片算子在图上展开成每台 DPU 一份，
行切（o_proj / down_proj）之后插入 ALL_REDUCE 节点（`gene_sim_scheduler.py:1132`），
入边搬全量输出字节，即真实的 all-reduce 流量。

四种策略的实测产物：

```
策略      分片/算子  kernel_tile_n        每台DPU分片数  dpu_to_cluster
tp1_pp8   1         {512:160, 256:64}    各 28          8 个 Cluster 各一台
tp2_pp4   2         {512:160, 128:64}    各 56          两台配一个 Cluster
tp4_pp2   4         {512:160,  64:64}    各 112         四台配一个 Cluster
tp8_pp1   8         {512:160,  32:64}    各 224         8 个 Cluster 各一台
```

`kernel_tile_n` 随 TP 宽度单调收窄（256→128→64→32），是算子编译器按 WRAM 预算实搜
出来的，不是配置里的常量（GeneSim 默认常量是 32）。

### 2.4 算子编译产物进入代价链

v0.0.3 时 GeneSim 用手写模板生成 PIM trace，分块取 `conf/sim.yaml` 里拍的 32。
v0.0.4 起照算子编译器产出的 pim mlir 生成 trace（`src/pim/pimir_trace.py` 新增
402 行），真实的 DMA 量和循环嵌套进入周期数。

sidecar 为此新增三个字段：`kernel_tile_n`（实测分块）、`pimir_path`（pim mlir 路径）、
`pimir_sha256`（内容哈希，进 trace 缓存签名，避免"同形状同分块但 pim mlir 换了"复用
旧 trace）。

**防静默退化**：sidecar 顶层新增 `requires_pimir` 字段自报"我带了算子编译产物"，
GeneSim 据此自动进入严格模式——pim mlir 读不到直接报错，不退回手写模板。此前这依赖
配置文件写对 `require_compiler_pimir: true`，而 tp4pp2 的配置就漏写过。

### 2.5 全流程闭环脚本

新增 `scripts/run_full_pipeline.py`（416 行），一条命令跑完三仓链路并逐段核对产物：

```
HF 权重 + config
  ├─(A)─> GeneSim model_parser   → 图骨架 IR（semantic_role 标好投影身份）
  ├─(B)─> 图编译器 compile_llama2 → PIMTensorSpec（每个权重的切分与归属）
  ├─(C)─> 算子编译器 FlagTree     → .so + pim mlir（真实分块由 WRAM 预算定）
  ├─(D)─> placement sidecar      → dpu_id / local_*_features / kernel_tile_n
  └─(E)─> GeneSim 仿真            → 照 pim mlir 生成 PIM trace，出代价
```

每一步跑完都核对产物，不满足就非零退出——这个脚本的用途是回答"链路通没通"，任何一段
静默退化都必须变成失败。

### 2.6 NumPy 后端对拍保持不变

NumpyBackend 仍是数值正确性的唯一判据：7B 在四种切分策略下的 logits 与单卡 HF 推理
逐元素对齐，KV cache 也对拍。这条路径本来就在 CPU 上跑，不受本次 GPU 改造影响。

## 3. 纯 CPU 部署（v0.0.4 新增）

### 3.1 结论

**可以在纯 CPU 服务器上跑完整流程。** 关键判据：

```
同一形状（4096→4096）、同一硬件参数下
  有 GPU 时的 pim mlir   sha256 = 26a1a970d0c648d8...
  无 GPU 时的 pim mlir   sha256 = 26a1a970d0c648d8...   ← 逐字节相同
```

不是"降级版本"。这一点关键：GeneSim 的 trace 缓存签名里记的就是这个哈希，两边产物
不同就会全量重编、代价数字也不可比。

### 3.2 为什么 GPU 本来不是必需的

这条链的产物是 **pim mlir → EmitC → C → .so**，全在 CPU 上跑。GPU 从不参与生成，
只是 Triton 前端的既有实现顺手依赖了它：

```
v0.0.3 的做法（opcompiler_bridge/driver.py）
    torch.empty(..., device="cuda") × 3  →  真实 launch  →  compiled.asm["ttir"]
                                             ↑ 只是为了拿到 TTIR 文本
```

TTIR 是 AST → IR 的纯前端产物，不需要设备。FlagTree 的
`triton/backends/pim_sidecar.py` 开头就写着 "PIM branches off TTIR rather than
sitting on the path to the GPU binary"——设计上本就支持。

### 3.3 无 GPU 时真正的卡点

不是"编译需要算力"，而是两处**设备探测**：

| 卡点 | 位置 | 现象 |
| --- | --- | --- |
| driver 注册 | `triton/runtime/driver.py:7` 要求恰好一个 active driver，而 `backends/nvidia/driver.py:762` 的 `is_active()` 直接返回 `torch.cuda.is_available()` | `RuntimeError: 0 active drivers` |
| import 期探测 | Triton hint manager（`compiler/hint_manager.py:95`）、FlagGems 的 `utils/triton_driver_helper.py:22` 在 import 期就读 `driver.active` | 连 `import flag_gems` 都过不去 |

两处都只是"问一下当前设备是什么"，没有一处真要跑 kernel。注入一个只回答这几个问题的
driver 就够了。

### 3.4 改动清单

| 仓库 | 文件 | 改动 |
| --- | --- | --- |
| flagos-pim-compiler | `opcompiler_bridge/cpu_host.py`（新增） | 编译期 driver 注入 + 纯前端 TTIR 生成 |
| flagos-pim-compiler | `opcompiler_bridge/driver.py` | `_make_ttir`：有卡走原生 launch，无卡走前端路径 |
| flagos-pim-compiler | `genesim_bridge/flagtree_driver.py` | 先注入 driver 再 import flag_gems；`cuda.synchronize()` 仅有卡时调用 |
| flagos-pim-compiler | `genesim_bridge/op_classify.py` | 9 处 `device="cuda"` → `device=_probe_device()` |
| flagos-pim-compiler | `tests/test_opcompiler_linear.py` | 门禁从"有没有 GPU"改为"triton 带不带 PIM pass" |
| flagos-pim-compiler | `tests/test_opcompiler_e2e_llama2_7b.py` | 同上 |
| flagos-pim-compiler | `genesim_bridge/cost_extractor.py` | 无卡时报错明确指向可用的编译路径（见 7.1） |
| flagos-pim-compiler | `scripts/run_full_pipeline.py` | 核对 sidecar 自报 `requires_pimir` |
| flagos-pim-compiler | `scripts/verify_cpu_only.sh`（新增） | 真无 GPU 容器里的一键验证（见 6.0） |
| flagOS-installers | `cpu-host-driver.py`（新增） | 无卡时的编译期 driver 注入，三处验证段共用一份 |
| flagOS-installers | `0-install-flagtree.sh` | `nvidia-smi` 从硬性要求改为可选检测 |
| flagOS-installers | `1-install-flaggems.sh` | 同上；验证段与 smoke test 载入共享 shim，无卡时只验 import |
| flagOS-installers | `2-install-pytorch.sh` | wheel 索引按机器选（cu128 / cpu）；驱动 570+ 检查只在装 CUDA 版时做；smoke test 双路径 |
| flagOS-installers | `3-install-model-inference.sh` | 五处 `cuda.is_available()` 硬退出改条件分支；三处 `import flag_gems` 前载入共享 shim |
| flagOS-installers | `matmul_sm80.py` | 脚本 0 的安装后验证示例：无卡时改为验证 PIM pass 而不执行 kernel |
| flagOS-installers | `model-inference/examples/run_llm_with_flaggems.py` | 8 处 GPU 硬依赖改为设备无关；无卡时退回 PyTorch 原生算子 |
| flagOS-installers | `README.md` | 补齐前置 apt 包、GPU 可选说明、"不要拷贝已装目录"的警告 |
| genesim | `install.sh` | torch 索引按机器选（见 5.2）；两步装避免 nvidia 包被拖进来 |
| genesim | 无需改动（其余） | 仿真主链路不依赖 GPU；`predictor/` 的 cuda 引用本就 `auto` 回落 cpu |
| FlagTree | 无需改动 | PIM pass 是 CPU 上跑的 MLIR 变换 |
| FlagGems | 无需改动 | 改的是调用方式（注入 driver + `GEMS_VENDOR`），不改其源码 |

FlagGems 这一行值得说明：它的 `utils/libentry.py:874-878` 已经有 CPU 分支，注释明写
"This branch is CPU-generic"，说明上游有 CPU 意图；卡住的只是
`runtime/backend/device_finder.py:136` 的设备探测。用 `GEMS_VENDOR=nvidia` 越过即可
（`_arm` vendor 虽然 `device_name="cpu"`，但其数学 shim 缺 `asin`，反而不能用）。

### 3.5 核心机制

```python
# opcompiler_bridge/cpu_host.py
class CpuHostDriver(DriverBase):          # 注意：不继承 CudaDriver，原因见下
    """无 GPU 机器上的编译期 driver：只回答设备探测，不碰 CUDA 运行时。"""
    def __init__(self):
        self.utils = _CpuHostUtils()      # 只回答 get_device_properties
        self._target = GPUTarget("cuda", 80, 32)
    def get_active_torch_device(self): return torch.device("cpu")
    def get_current_device(self): return 0
    def get_current_stream(self, device=None): return 0
    def get_current_target(self): return self._target
    # ... map_python_to_cpp_type / current_arch_id 等

driver_config.set_active(CpuHostDriver())   # set_active 是 Triton 公开接口
```

**为什么不继承 `CudaDriver`**：它的 `__init__` 第一行就是 `CudaUtils()`，后者会编译一个
C 扩展并链接 `libcuda.so.1`。纯 CPU 机器上那个库不存在，构造直接 assert 失败：

```
AssertionError: libcuda.so cannot found!
  at CudaDriver.__init__ → CudaUtils() → library_dirs() → libcuda_dirs()
```

所以改成继承 `DriverBase`，只实现编译期真正会被读到的成员。这份清单是从 Triton 和
FlagGems 源码里逐个 grep 出来的：

| 成员 | 谁读它 |
| --- | --- |
| `get_active_torch_device` | Triton hint manager |
| `utils.get_device_properties` | Triton compiler / FlagGems |
| `get_current_device` / `get_current_stream` | Triton compiler |
| `get_current_target` | Triton compiler / autotuner |
| `map_python_to_cpp_type` | `DriverBase` 抽象方法 |
| `current_arch_id` | FlagGems |
| `launcher_cls` / `get_benchmarker` | 只在真正启动 kernel 时读 |

最后两项保留为**显式报错**而不是不实现——纯编译路径不会走到它们，一旦走到说明有人试图
在无 GPU 的机器上启动 Triton kernel，报清楚的话比 `AttributeError` 好排查。

**必须在 `import flag_gems` 和任何 `triton.compile` 之前注入**——那些模块在 import 期
就读 `driver.active`，一旦读到就会缓存住结果。

### 3.6 两个容易漏掉的一致性细节

**其一**，第一次比对时 CPU 路径的产物与 GPU 路径差**一行**：

```
-  tt.func public @linear_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, ...)
+  tt.func public @linear_kernel(%arg0: !tt.ptr<f16>, ...)
```

`tt.divisibility = 16` 来自真实张量指针的对齐推断（`torch.empty` 返回 16 字节对齐的
指针）。无卡时没有真实张量，必须显式声明同样的特化，否则产物差这一行——而 sidecar 记
的是内容哈希，差一行就对不上，属于会静默偏掉的那类问题。

**其二**，TTIR 必须跑完 ttir stage 的 pass（含 inliner）才能作为 PIM pass 的输入。
`make_ir` 的裸输出还留着未内联的 `tt.call`，`convert-triton-to-pim` 会报
`tensor<1x512xf32, #pim.tasklet_tiled<...>>` 与函数返回类型不匹配。

## 4. 硬件与容量要求

### 4.1 实测数据（关闭 GPU 的环境下测得）

| 项目 | 实测值 | 说明 |
| --- | ---: | --- |
| 7B fp16 权重常驻 | 13 GiB | 磁盘 26 GB |
| **7B 推理峰值 RSS** | **13.0 GiB** | prefill 128 + decode 4，8.4 秒 |
| 7B 全量测试 | 7 分 48 秒 | 38 passed |
| 算子编译单形状 | 约 10~20 秒 | llama2 共 7 种本地形状 |

### 4.2 服务器规格

**最低可跑**：

| 资源 | 要求 | 依据 |
| --- | --- | --- |
| 内存 | **32 GiB** | 推理峰值 13 GiB + 仿真 + 编译进程 + OS 余量 |
| 磁盘 | **80 GiB** 可用 | 模型 26 GB + FlagTree/LLVM 构建约 20 GB + Python 环境与缓存 |
| CPU | 8 核 | 能跑通，但 7B 测试与 LLVM 编译明显变慢 |
| 系统 | Ubuntu 22.04 x86_64 | 安装脚本硬性检查这两项 |

**推荐**：内存 64 GiB、磁盘 200 GiB、CPU 32 核以上。

**内存是唯一硬约束**：7B fp16 权重必须整个放进内存（当前实现不做权重分片加载），
13 GiB 是下限，低于约 20 GiB 有 OOM 风险。

**本文档实测环境**：128 核 / 1007 GiB。上面"最低"一栏是按峰值实测加余量推算，
**未在 32 GiB 机器上实跑**。若乙方机器接近下限，建议分步确认：先跑单条 7B 推理，
再跑全量测试。

### 4.3 网络访问

安装脚本要下载 LLVM 预编译包、Triton 构建依赖、Python standalone 发行版和 PyTorch
wheel。离线环境需先在有网机器上准备 `downloads/` 目录再整体拷贝。

## 5. 安装步骤（与 v0.0.3 的差异）

步骤不变，仍是四个脚本按顺序执行（见 v0.0.3 第 2.2 节）。

### 5.0 纯 CPU 环境从零开始：完整操作清单

这一节是纯 CPU 机器上从零搭建的**完整步骤**，照着做即可。

#### 第一步：装系统包（唯一需要 root 的一步）

纯净的 Ubuntu 22.04 缺 7 个必需命令（`git`、`make`、`cc`、`c++`、`ar`、`ld`、`curl`），
必须先补：

```bash
sudo apt-get update
sudo apt-get install -y build-essential git curl tar gzip python3
```

装完之后四个安装脚本本身不需要 root。**这一步在 v0.0.3 文档里没有写明**，纯净环境下
不做会在第一个脚本就报 `缺少命令 git`；漏掉 `python3` 则第三个脚本报
`缺少命令 python3`（脚本 0/1/2 用的是自带的独立 Python，只有脚本 3 需要系统 python3）。

这个清单是在纯净 `ubuntu:22.04` 容器里逐个试出来的，四个脚本 `require_command` 的完整
并集：`git`、`tar`、`gzip`、`dpkg-deb`、`apt-get`、`awk`、`sed`、`find`、`make`、`cc`、
`c++`、`ar`、`ld`、`curl`（或 `wget`）、`python3`。上面一条 apt 命令覆盖全部。

#### 第二步：网络设置（跨境网络建议）

安装要从 GitHub 和 PyPI 拉几个 GB，默认超时在跨境网络下偏短：

```bash
export UV_HTTP_TIMEOUT=600      # GeneSim 的 install.sh 用 uv，默认仅 30 秒
```

国内网络建议同时配 PyPI 镜像（**只影响普通 pip 包**，脚本装 torch 时用的是
`--index-url https://download.pytorch.org/whl/...`，不受这个变量影响）：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

**PyTorch 官方索引没有国内镜像**（清华的 `pytorch-wheels` 路径已下线，返回 404），
torch wheel 只能从 `download.pytorch.org` 拉，CPU 版 184 MB、CUDA 版约 2.5 GB。

实测遇到过这几类网络失败，**重跑同一条命令即可继续**（脚本每一步都有幂等判断）：

| 报错 | 发生在 |
| --- | --- |
| `GnuTLS recv error (-110): The TLS connection was non-properly terminated` | clone FLIR 子模块 |
| `Failed to download rich==15.0.0 ... network timeout` | GeneSim 装 Python 依赖 |
| `Connection interrupted while downloading` / `Attempting to resume incomplete download` | pip 拉包，会自动重试 |
| `ERROR: No matching distribution found for torch==2.9.1+cpu` | 瞬时故障，**不是版本不存在**（见下） |

最后一条值得说明：该版本在 PyTorch 官方 CPU 索引里确实存在
（`torch-2.9.1+cpu-cp310-cp310-manylinux_2_28_x86_64.whl`），实测用 `--dry-run` 能正常
解析。报这个错时不要去改脚本里钉的版本号，**直接重跑**。

如果反复失败，可以先把 Python 发行版 tarball 预置到 `<prefix>/downloads/`，脚本检测到
就跳过下载（`install_python` 有幂等判断）。

#### 第三步：四个安装脚本

```bash
git clone https://github.com/jingge815/flagOS-installers.git
cd flagOS-installers

bash 0-install-flagtree.sh       # 自动探测：无 GPU → 纯 CPU 模式
bash 1-install-flaggems.sh
bash 2-install-pytorch.sh        # 无 GPU 时装 torch 2.9.1+cpu

# 模型用本机已有的目录，不需要重新下载、不需要 HF 授权
bash 3-install-model-inference.sh \
  --model-path /path/to/Llama-2-7b-hf
```

四个脚本都会自动探测 GPU；**装 torch 的那两个**（`2-install-pytorch.sh`、
`3-install-model-inference.sh`）额外接受 `--torch-cpu` / `--torch-cuda` 强制指定：

```bash
bash 2-install-pytorch.sh --torch-cpu     # 强制 CPU 版
bash 2-install-pytorch.sh --torch-cuda    # 强制 CUDA 版
```

脚本 0 和 1 不装 torch，所以没有这两个开关——它们只是把原来的"没有 nvidia-smi 就退出"
改成了可选检测。

#### 第四步：图编译器与 GeneSim

先克隆两个仓库（它们不在 flagOS-installers 里）：

```bash
cd /path                     # 与 flagOS-installers 同级即可
git clone https://github.com/jingge815/flagos-pim-compiler.git
git clone https://github.com/pimtools/genesim.git
```

**先装 GeneSim 依赖**——它同时补齐了图编译器测试要用的 `scipy`、`datasets`
（flagOS 的 PyTorch 环境里没有这两个），所以这一步不能放到最后：

```bash
cd /path/genesim
./install.sh --skip-attacc          # 装 uv、建 .venv，并补齐 Python 依赖
```

然后配置图编译器的站点路径。这些是环境变量，不改任何源码：

```bash
cd /path/flagos-pim-compiler
source /path/flagOS-installed/pytorch/env-pytorch.sh

export PYTORCH_ENV_SCRIPT=/path/flagOS-installed/pytorch/env-pytorch.sh
export LLAMA2_7B_MODEL_DIR=/path/flagOS-installed/model-inference/models/Llama-2-7b-hf
export FLAGTREE_PREFIX=/path/flagOS-installed/flagTree
export GENESIM_ROOT=/path/genesim

# 确认路径都解析正确
python -c 'from genesim_bridge.paths import describe; print(describe())'
```

`genesim/scripts/refine_ir_with_flagtree.py` 里的 `DEFAULT_BRIDGE_ROOT` 写的是开发机的
绝对路径，**不要改源码**——用环境变量覆盖：

```bash
export PIM_COMPILER_ROOT=/path/flagos-pim-compiler
```

#### 第五步：跑测试确认装成功

装完之后按这个顺序验证。每一条都在纯 CPU 环境实测过，预期结果写在右边。

```bash
cd /path/flagos-pim-compiler
source /path/flagOS-installed/pytorch/env-pytorch.sh
# 第四步的那几个 export 也要在当前 shell 里生效

# 1. 快速回归（约 20 秒）
python -m pytest tests/ -q -k "not llama2_7b"
#    预期: 247 passed, 42 deselected

# 2. 算子编译单元测试（约 3 秒）——这一组在改动前的无卡机器上会全部 skip
python -m pytest tests/test_opcompiler_linear.py -q
#    预期: 19 passed

# 3. 7B 全量（纯 CPU 约 34 分钟，含真实编译 224 个 GEMM）
python -m pytest tests/ -q -k "llama2_7b"
#    预期: 42 passed

# 4. GeneSim 全套
cd /path/genesim && ./run.sh --test
#    预期: All test suites passed
#    注: tests/predictor/ 需要可选依赖，见下面「可选：GNN 性能预测器」

# 5. 全流程闭环（约 10 分钟）——这是"链路通没通"的唯一判据
cd /path/flagos-pim-compiler
python scripts/run_full_pipeline.py --num-stages 4
#    预期尾部:
#      算子编译器选出的分块: [128, 512]（GeneSim 默认常量是 32）
#      total_time_s = 1418.109
#      GEMM trace 来源: {'pimir': 448}      ← 零退回手写模板
#      全流程验证通过：模型加载 → 图编译切分 → 算子编译 → GeneSim 代价
```

第 5 条的 `{'pimir': 448}` 是最关键的一行：448 = 224 个 GEMM × 2 个分片，全部照算子
编译器产出的 pim mlir 生成 trace。如果这里出现 `template`，说明算子编译产物没进代价链。

##### 可选：GNN 性能预测器

`tests/predictor/` 共 87 个用例，其中 9 个需要可选依赖 `torch-geometric`（默认不装）。
不装时那 9 个会失败，报错自己会指明原因：

```
ModuleNotFoundError: No module named 'torch_geometric'
ImportError: the GAT predictor backbone requires torch-geometric;
  install predictor dependencies with: ./install.sh --predictor
```

需要它时：

```bash
cd /path/genesim
./install.sh --predictor        # 装 torch-geometric==2.8.0.post1
./run.sh --test predictor
```

性能预测器是独立特性，**不参与 PIM 编译链路**——不装它，上面五条验证全部照常通过。

#### 一条重要限制：不要拷贝已装好的目录

`flagOS-installed/` **不能整体打包拷到另一台机器或另一个路径**。pip 生成的 32 个命令
包装脚本（`cmake`、`ninja`、`lit` …）把解释器绝对路径写进了 shebang：

```
$ head -1 flagTree/python-3.10.20/bin/cmake
#!/media/disk/.../flagOS-installed/flagTree/python-3.10.20/bin/python
```

换路径后这些命令全部失效，FlagTree 编译报
`RuntimeError: CMake must be installed to build the following extensions: triton`。

**正确做法**：在目标机器上跑一遍安装脚本。各步骤都有幂等判断（`install_llvm` 检查
`llvm-config` 是否可执行、`checkout_flir` 检查目录是否存在），已就位的 LLVM、Python、
下载缓存都会自动跳过，不会重复下载。

如果确实要复用已下载的大件（LLVM 约 4.6 GB），可以把它们放到目标机器的**安装前缀下的
同名位置**再跑脚本，脚本会跳过下载而照常重新编译 FlagTree。

### 5.1 四个脚本都不再硬性要求 nvidia-smi

v0.0.3 的 2.1 节要求"必须安装可用的 NVIDIA 驱动并能执行 `nvidia-smi`"。v0.0.4 起
四个脚本都自行检测，两种环境都兼容：

```
==> 检测到 NVIDIA GPU（驱动 570）：装 CUDA 版 torch。                    # 有卡
==> 未检测到 NVIDIA GPU（或指定了 --torch-cpu）：装 CPU 版 torch。        # 无卡
```

**有卡时行为与 v0.0.3 完全一致，无任何回退**——这是设计前提，不是巧合：所有改动都是
"多一条分支"，不是"改掉原路径"。实测在本机（A800）上自动判定仍走 CUDA 路径。

四个脚本各自的改动点：

| 脚本 | 改了什么 | 没改什么 |
| --- | --- | --- |
| `0-install-flagtree.sh` | `nvidia-smi` 改为可选检测 | 编译流程（本来就全在 CPU 上） |
| `1-install-flaggems.sh` | 同上；smoke test 无卡时只验 import | 装包流程 |
| `2-install-pytorch.sh` | wheel 索引按机器选；驱动 570+ 检查只在装 CUDA 版时做；smoke test 双路径 | **Triton 同步整段** |
| `3-install-model-inference.sh` | 五处 `cuda.is_available()` 硬退出改条件分支；preflight 注入 driver | 模型下载与推理对拍流程 |

新增两个开关（四个脚本口径一致）：

```bash
--torch-cpu     # 强制 CPU 版
--torch-cuda    # 强制 CUDA 版
# 都不给 = 按机器实际情况自动判断
```

#### 关于 `2-install-pytorch.sh` 的 Triton 同步（一行都不用改）

这个脚本**本来就不编译 PyTorch**，只下载官方 wheel + 配置环境。它的
`sync_triton_to_pytorch()`（`2-install-pytorch.sh:194` 起）是纯文件拷贝：

```
FlagTree 的 triton  →  PyTorch 环境的 triton
  _C/libtriton.so                          ← 带 PIM pass 的核心库
  backends/pim_sidecar.py                  ← PIM 从 TTIR 分叉的入口
  backends/nvidia/compiler.py
  backends/nvidia/{bin,include,lib/cupti}   ← ptxas / cuda.h
```

**那三个 `backends/nvidia/` 目录在纯 CPU 上也必须同步**，不能当作"GPU 相关"删掉：
图编译器的 `genesim_bridge/env.py` 需要其中的 `cuda.h` 和 `ptxas`。它们是随 pip 包
分发的**文件**，不是驱动——实测无卡下 `prepare_triton_env()` 与
`assert_pim_passes_available()` 都正常通过。

这也回答了"能不能把不必要的 GPU 东西删掉"：**不建议删**。这些文件是算子编译链的组成
部分，删了纯 CPU 环境反而跑不起来。真正该动的只是"硬性要求有 GPU 硬件"那些门禁。

### 5.2 GeneSim 的 install.sh 默认改用 CPU-only torch

`genesim/install.sh` 原来无条件装默认 PyPI 的 torch。纯 CPU 机器上那会额外拉
**18 个 nvidia-*/cuda-* 包**（`cuda-toolkit`、`cudnn`、`nccl`、`cublas` 等，共数 GB），
一个都用不到——GeneSim 是纯 CPU 仿真器，算子编译也只需要 TTIR → pim mlir。

实测两个索引的依赖数量：

```
默认 PyPI:  29 个包，其中 18 个是 nvidia/cuda
CPU-only :  10 个包，其中  0 个是 nvidia/cuda
```

本轮给它加了自动判断（探测不到 `nvidia-smi` 就用 CPU-only 索引），并提供两个开关：

```bash
./install.sh --skip-attacc              # 自动判断：无卡→CPU-only，有卡→CUDA
./install.sh --skip-attacc --torch-cpu  # 强制 CPU-only
./install.sh --skip-attacc --torch-cuda # 强制 CUDA 版
```

实测生效：`torch: 2.14.0+cpu`，venv 里 nvidia 包 **0 个**。

这一条对甲方影响很直接：不加这个改动，纯 CPU 机器上装依赖会白下几 GB，且在网络不佳
时长时间卡住（本轮验证中曾卡到 26 分钟只下了 112 MB，另一次因 `rich` 下载超时整个
安装失败）。建议同时设 `export UV_HTTP_TIMEOUT=600`，默认的 30 秒在跨境网络下太短。

**一个值得记下的坑**：第一版实现写成

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu \
               --extra-index-url https://pypi.org/simple \
               torch transformers ...        # ← 错的
```

看起来合理（CPU 索引优先、其余包回落 PyPI），但实测 uv 仍从 PyPI 解析出**普通版**
`torch==2.14.0`（而不是 `2.14.0+cpu`），18 个 nvidia 包照样被拖进来。正确做法是分两步、
第一步只给 CPU 索引：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # 先钉死 torch
uv pip install pyyaml numpy scipy transformers ...                      # 其余走正常 PyPI
```

这个错误当时被另一个现象掩盖了：安装在 `rich` 下载超时时失败，venv 没装完，于是
"nvidia 包 0 个"看起来像是修好了——实际上只是还没装到那一步。**判断依据要看
`torch.__version__` 是否带 `+cpu` 后缀**，而不是数 venv 里的包。

### 5.3 纯 CPU 环境下 `genesim/docs/llama-2.md` 怎么走

`llama-2.md` 的六步里，**第四步在纯 CPU 上不可用**（原因见 7.1 节：它需要真实执行
FlagGems 算子）。可行的走法是跳过第四步，用第六步的放置导出替代：

| 步骤 | 命令 | 纯 CPU |
| --- | --- | --- |
| 一、环境准备 | `./install.sh --skip-attacc` | 可用（见 5.2） |
| 二、图骨架 | `python scripts/model_parser.py --model_name <模型目录> --output models/llama2_7b.ir` | **可用**，实测产出 3491 算子 |
| 三、请求 trace | `./run.sh --trace --synthetic --seed 0 --num_requests 10 --output traces/llama2_7b.trace` | **可用**，实测 10 条请求 |
| 四、FlagTree 成本精化 | `python scripts/refine_ir_with_flagtree.py ... --ir-level pimir` | **不可用**，改走下面第六步 |
| 五、默认仿真 | `./run.sh` | 可用 |
| 六、按切分策略放置 | `python scripts/export_pp_placement.py --partition-plan <plan> --measure-kernel-tiles` | **可用**，这是纯 CPU 下的成本来源 |

最省事的做法是直接跑全流程脚本，它已经把第二、六、五步串好了：

```bash
cd flagos-pim-compiler
python scripts/run_full_pipeline.py --num-stages 4
```

另外 `llama-2.md` 第二步说要改 `scripts/refine_ir_with_flagtree.py` 里的
`DEFAULT_BRIDGE_ROOT`——**其实不用改**，代码已支持环境变量覆盖
（`refine_ir_with_flagtree.py:54`）：

```bash
export PIM_COMPILER_ROOT=/path/to/flagos-pim-compiler
```

改源码反而会把本地路径写进仓库。这条文档建议已过时，交付时应一并更新。

#### GeneSim 的 uv 依赖：是前置步骤，但不是唯一路径

`genesim/run.sh` 的每个子命令都先 `check_uv` + `check_venv`，所以第一步 `install.sh`
不能跳过——它负责装 `uv` 并建 `.venv`。

值得说清 `.venv` 到底是什么：它是一套**与 flagOS 完全独立**的 Python 环境。

```
GeneSim .venv:        Python 3.11.15 / torch 2.13.0+cu130 / triton 无 PIM pass
flagOS pytorch 环境:  Python 3.10.20 / torch 2.9.1+cu128 / triton 有 PIM pass
```

注意 `.venv` 的 triton **没有 PIM pass**——这说明**仿真本身不需要算子编译能力**，它读
的是已经生成好的 pim mlir 文本，不自己编译。实测 flagOS 环境导入 GeneSim 主程序只缺
`scipy` 一个包。

于是有两条路径，实测**结果完全一致**：

| 路径 | 命令 | 实测 `total_time_s` |
| --- | --- | --- |
| A（推荐，上游支持） | `./install.sh --skip-attacc` 后 `./run.sh --config ...` | 1418.1092459087413 |
| B（无 uv 时的备选） | 用 flagOS 的 python 直接跑 `src/main.py` | 1418.1092459087413 |

路径 B 的做法：

```bash
source .../flagOS-installed/pytorch/env-pytorch.sh
pip install --target /tmp/extra scipy          # flagOS 环境唯一缺的包
PYTHONPATH=/tmp/extra:src python3 src/main.py --config conf/xxx.yaml
```

两条路径小数位都一样，说明 uv 只是包管理工具，对仿真代价零影响。

**交付建议用路径 A**：它是上游设计的用法，不动 GeneSim 代码。路径 B 混用了两套 Python
环境（flagOS 的 3.10 跑 GeneSim 代码），实测能跑通仿真，但**未覆盖 `run.sh` 的全部
子命令**（predictor、upmem_checker 等），只适合"临时验证仿真结果"这一个用途。

### 5.4 纯 CPU 机器上的环境变量

`genesim_bridge` 会自动注入编译期 driver 并设好 `GEMS_VENDOR`，正常使用无需手工设置。
若直接调用 FlagGems（不经本仓桥接）：

```bash
export GEMS_VENDOR=nvidia    # 越过 FlagGems 的设备探测；数学函数走 libdevice
```

选 `nvidia` 而不是看起来更贴切的 `arm`（后者 `device_name` 恰好是 `"cpu"`）：`arm`
vendor 的数学 shim 缺 `asin`，`import flag_gems` 会在 `ops/arcsin.py` 挂掉。

## 6. 验证结果

### 6.0 验证方法：必须用真无 GPU 的环境，`CUDA_VISIBLE_DEVICES=""` 不够

这一条是本轮最重要的方法论修正。`CUDA_VISIBLE_DEVICES=""` 只让
`torch.cuda.is_available()` 返回 `False`，但**驱动、`libcuda.so`、`/dev/nvidia*` 都还
在**。甲方的机器上这些根本不存在，行为不同。

实测差异：第一版 driver shim 在 `CUDA_VISIBLE_DEVICES=""` 下全部测试通过，但在真正
没有 GPU 的容器里**直接挂掉**：

```
AssertionError: libcuda.so cannot found!
  at CudaDriver.__init__ → CudaUtils() → library_dirs() → libcuda_dirs()
```

这个缺陷用环境变量怎么测都不会暴露。所以本仓提供一个容器验证脚本：

```bash
bash scripts/verify_cpu_only.sh          # 全部项目
bash scripts/verify_cpu_only.sh quick    # 只跑不需要 7B 模型的项（约 1 分钟）
```

它在 `ubuntu:22.04` 容器里跑，容器内**没有设备节点、没有 libcuda、没有 nvidia-smi**，
与甲方机器一致。脚本第一步就是自检这三项，确认隔离真的生效——否则后面的结论不成立。

不会改动宿主环境：源码以只读挂载到 `/src`，容器内复制成可写副本；`OPCOMPILER_CACHE_DIR`
和 `HOME` 都指向容器内的 `/tmp`。

没有 docker 时，在真实的纯 CPU 机器上直接跑脚本里的那些命令即可，判据完全相同。

### 6.1 各项结果

以下在**真无 GPU 的 Ubuntu 22.04 容器**里实测（部分项另在宿主机
`CUDA_VISIBLE_DEVICES=""` 下复核过，结果一致）。

| 验证项 | 命令 | 结果 |
| --- | --- | --- |
| **安装脚本 0（编译 FlagTree）** | `bash 0-install-flagtree.sh` | **通过**，`SCRIPT_EXIT=0`；新编 wheel 285 MB，`PIM pass 可用: True`、`triton-opt` 有 3 个 pim pass |
| **安装脚本 1（FlagGems）** | `bash 1-install-flaggems.sh` | **通过**，`SCRIPT_EXIT=0`；smoke test `max error: 0.0` |
| **安装脚本 2（PyTorch + PIM Triton 同步）** | `bash 2-install-pytorch.sh` | **通过**，`SCRIPT_EXIT=0`；`torch: 2.9.1+cpu`、`PIM Triton passes: OK`，CPU wheel 只下 184 MB（CUDA 版约 2.5 GB） |
| **安装脚本 3（模型推理）** | `bash 3-install-model-inference.sh --model-path <本机模型目录> --skip-download` | **通过**，`SCRIPT_EXIT=0`；`flag_gems_preflight: ok`、7B 真实生成 8 个 token、`inference_status: ok` |
| 快速回归 | `pytest tests/ -q -k "not llama2_7b"` | **247 passed, 42 deselected**，17.7 秒（与有卡基线一致） |
| 7B 全量 | `pytest tests/ -q -k "llama2_7b"` | **42 passed**，33 分 37 秒（容器内；宿主机 38 passed + 4 skip，多出的 4 个正是原先被 GPU 门禁 skip 的算子编译测试） |
| 算子编译单元 | `pytest tests/test_opcompiler_linear.py -q` | **19 passed**，3.1 秒（改动前无卡全 skip） |
| 算子编译 7B 端到端 | `pytest tests/test_opcompiler_e2e_llama2_7b.py -q` | **4 passed**，26 分 09 秒（改动前无卡全 skip；已含在上面的 42 里） |
| 安装脚本 preflight | `3-install-model-inference.sh` 的 preflight 段 | **通过**，无卡走 CPU 分支、有卡报 `NVIDIA A800-SXM4-80GB` |
| GeneSim 全套 | `cd genesim && ./run.sh --test` | **All test suites passed** |
| FlagGems smoke | 安装脚本 `run_smoke_test` 段 | **通过**，max error 0.0 |
| pim mlir 一致性 | 与有卡产物比对 sha256 | **逐字节相同** |
| **全流程闭环** | `python scripts/run_full_pipeline.py --num-stages 4` | **通过**，约 10 分钟（见 6.1） |
| 7B CPU 推理 | prefill 128 + decode 4 | **通过**，峰值 13.0 GiB / 8.4 秒 |

那 4 个 skip 是 `paths.json` 未配置可选项导致的，与 GPU 无关。

### 6.2 用安装脚本新编出的产物复跑一遍

上表部分项目最初是借用机器上已有的 `flagOS-installed` 跑的。为确认"脚本编出来的东西
真的能用"，把 FlagTree 重新编译一遍（`Successfully built flagtree`），用 2 号脚本把新
triton 同步进 PyTorch 环境（`PIM Triton passes: OK`），然后用**这套新产物**复跑：

```
确认用的是新产物：
  triton:   .../pytorch/python/lib/python3.10/site-packages/triton/__init__.py
  PIM pass: True
  torch:    2.9.1+cpu | cuda: False
```

| 测试 | 结果 |
| --- | --- |
| pim-compiler 快速回归 | **247 passed** / 16.69 秒 |
| pim-compiler 7B 全量 | **42 passed** / 33 分 59 秒 |
| GeneSim 全套 | **734 passed**, 9 failed, 1 skipped（9 个失败见下） |
| 全流程闭环 | **通过**，`total_time_s = 1418.109`、`{'pimir': 448}`、`PIPELINE_EXIT=0` |

`total_time_s` 与借用已有环境时完全一致（`1418.109`）——新编产物与原有产物等效。

### 6.3 照本文档从零走一遍（最终校对）

上面两轮验证是分步做的（每步挂载上一步的产物）。为确认**本文档本身可照做**，最后在一个
只装了第一步那几个 apt 包的纯净 `ubuntu:22.04` 容器里，严格按 5.0 节的顺序重跑一遍。

容器自检确认真的没有 GPU：无 `/dev/nvidia*` 设备节点、`ldconfig` 里 libcuda 条目为 0、
无 `nvidia-smi`。

**四个安装脚本**：

| 脚本 | 退出码 | 关键输出 |
| --- | --- | --- |
| `0-install-flagtree.sh` | 0 | `未检测到 NVIDIA GPU，按纯 CPU 模式安装`、`Successfully built flagtree`、验证示例 `PIM passes: OK` |
| `1-install-flaggems.sh` | 0 | `cpu-host-driver: 已注入编译期 driver`、smoke test `max error: 0.0` |
| `2-install-pytorch.sh` | 0 | `torch: 2.9.1+cpu`、`PIM Triton passes: OK`、`device: cpu（未检测到 GPU…）` |
| `3-install-model-inference.sh` | 0 | `flag_gems_preflight: ok`、`flaggems_generated_tokens: 8`、`inference_status: ok` |

**五项测试**（全部用这一轮新编出的产物；先确认 `triton` 指向新同步的路径、
`PIM pass: True`、`torch: 2.9.1+cpu | cuda: False`）：

| 测试 | 结果 |
| --- | --- |
| 1. 快速回归 | **247 passed, 42 deselected** / 17.20 秒 |
| 2. 算子编译单元 | **19 passed** / 1.52 秒 |
| 3. 7B 全量 | **42 passed, 247 deselected** / 33 分 33 秒 |
| 4. GeneSim 全套 | **734 passed**, 9 failed（全在 `tests/predictor/`，缺可选依赖）, 1 skipped |
| 5. 全流程闭环 | **通过**，`total_time_s = 1418.109`、`GEMM trace 来源: {'pimir': 448}`、`ALL_EXIT=0` |

第 4 项那 9 个失败逐项核实过：**全部在 `tests/predictor/`，其他位置零失败**，原因是可选
依赖 `torch-geometric` 未安装（见 5.0 节第五步的说明）。

第 5 项的 `total_time_s = 1418.109` 与前两轮完全一致，`{'pimir': 448}` 表示 448 个 GEMM
trace 全部来自算子编译器产出的 pim mlir，零退回手写模板。

这一轮校出四处文档与实际不一致，都已修正：

| 问题 | 修正 |
| --- | --- |
| 说"四个脚本都接受 `--torch-cpu`/`--torch-cuda`" | 实际只有装 torch 的脚本 2、3 有；脚本 0、1 不装 torch |
| genesim 仓库 URL 写成 `jingge815/genesim` | 实际是 `pimtools/genesim` |
| 五处代码行号引用过时（`gene_sim_scheduler.py:1121` 等） | 逐个核对当前源码并更正 |
| 缺"装完之后怎么跑测试"一节 | 补上 5.0 节第五步，五条命令都标了预期输出 |

还发现两个**脚本本身**的遗漏，也一并修了：

| 问题 | 修正 |
| --- | --- |
| 脚本 0 的安装后验证 `matmul_sm80.py` 硬编码 `device="cuda"`，纯 CPU 上必然失败 | 无卡时改为验证 PIM pass 而不执行 kernel |
| 脚本 2 装 torch 不幂等：重跑会重新下载 184 MB / 2.5 GB | 已装同版本就跳过；新增 `--wheel-dir` 支持从本地目录装 |

#### 那 9 个 GeneSim 失败：缺可选依赖，与纯 CPU 无关

9 个全在 `tests/predictor/`（GNN 性能预测器），报错是：

```
ModuleNotFoundError: No module named 'torch_geometric'
ImportError: the GAT predictor backbone requires torch-geometric;
  install predictor dependencies with: ./install.sh --predictor
```

`torch-geometric` 是 GeneSim 的**可选**依赖，只有 `./install.sh --predictor` 才装
（`install.sh:282` 装 `torch-geometric==2.8.0.post1`）。性能预测器是独立特性，**不参与
PIM 编译链路**——上表其余项目全部通过就说明这一点。

对照证据：机器上的 `.venv` 装过这个包（`torch_geometric 2.8.0.post1`），跑同样的测试

```
$ ./run.sh --test predictor
Ran 3 tests ... OK
[SUCCESS] All predictor tests passed (7/7 test files)
```

**所以要跑 predictor 测试，纯 CPU 环境同样需要先 `./install.sh --predictor`。**

⚠️ 一处如实说明：我**没能在纯 CPU 容器里直接验证** `--predictor` 装完后这 9 个测试转
绿——两次尝试都卡在网络（`No matching distribution found for requests`）。上面的结论
基于两条证据：报错信息自己指明了安装方法，以及有该包的环境里同样测试通过。交付前建议
在目标机器上实跑一次 `./install.sh --predictor && ./run.sh --test predictor` 确认。

#### 口径差异：`pytest tests/` 比 `./run.sh --test` 更严格

容器里没有 uv，所以我用 `pytest tests/` 代替 `./run.sh --test`。两者范围不同：
`run.sh --test` 分套件执行，而 pytest 全量收集会把可选特性（predictor）也算进来。所以
"734 passed / 9 failed" 与机器上的 "All test suites passed" 不是同一口径——前者更严格，
正是它暴露了可选依赖缺失。

### 6.1 全流程闭环的实测输出

这一项是"链路通没通"的唯一判据，五段逐个核对产物，任何一段静默退化都会非零退出。
纯 CPU 下的实际输出：

```
[A] HuggingFace config → GeneSim 图骨架 IR
    算子 3491 个，GEMM 224 个，七种投影身份齐全

[0] GeneSim 固定 PU 映射 → PartitionPlan
    stage0: [(0,0), (0,0)]   stage1: [(0,1), (0,1)]
    stage2: [(0,2), (0,2)]   stage3: [(0,3), (0,3)]   （每段同 Cluster 快链路）

[B+C+D] 图编译 → 算子编译（真实分块）→ placement sidecar
    放置 224 个 GEMM，本地形状 4 种 pim mlir
    算子编译器选出的分块: [128, 512]     ← GeneSim 默认常量是 32
    Cluster 映射已随 sidecar 回传，与方案一致（8 项）

[E] GeneSim 仿真（pimir）
    total_time_s = 1418.109
    tokens/s     = 5.701
    GEMM trace 来源: {'pimir': 448}      ← 零个退回手写模板
    (tile_n, k_iterations): {(512,128):192, (512,64):64, (128,128):128, (512,172):64}

全流程验证通过：模型加载 → 图编译切分 → 算子编译 → GeneSim 代价
```

两个数字值得单独指出：

- **`{'pimir': 448}`**：448 = 224 算子 × 2 分片（tp2），全部照算子编译器的 pim mlir
  生成 trace，**零个退回手写模板**。这是纯 CPU 环境下算子编译真的进了代价链的直接证据。
- **分块 [128, 512]**：算子编译器按 WRAM 预算实搜的值，不是 conf 里的常量 32。

`total_time_s` 与有卡环境的可比性：仿真代价只取决于 pim mlir 的内容和硬件参数，而
pim mlir 在两种环境下逐字节相同（3.1 节），所以代价数字也相同——纯 CPU 不是"估算模式"。

## 7. 限制与注意事项

### 7.1 两条成本路径：一条纯 CPU 可用，一条不可用

这是纯 CPU 环境下**最需要注意的一条**。仓里有两条给 GeneSim 提供算子成本的路径，
机制完全不同：

| 路径 | 入口 | 机制 | 纯 CPU |
| --- | --- | --- | --- |
| 编译路径 | `scripts/export_pp_placement.py` → `opcompiler_bridge.compile_op` | 只做 TTIR → pim mlir 的**编译** | **可用**，产物与有卡逐字节相同 |
| 执行路径 | `genesim/scripts/refine_ir_with_flagtree.py` → `cost_extractor.run_and_capture` | **真的把 FlagGems 算子跑一遍**，顺手抓下发的 kernel | **不可用** |

执行路径在纯 CPU 上必然失败：Triton kernel 无处可跑，抓不到内核，成本无从测量。
实测报错：

```
RuntimeError: op_type=GEMM 在 Tq=128,Tp=0 未捕获到任何 kernel：
  当前机器没有 GPU，而本路径需要真实执行 FlagGems 算子才能测量成本。
  纯 CPU 环境请改用编译路径（不需要执行 kernel）：
    python scripts/export_pp_placement.py --partition-plan <plan.json> --measure-kernel-tiles
```

#### 为什么不能"只编译不执行"

直觉上"抓 kernel 只是为了拿 pim mlir，编译一下就够了、不必真跑"是对的——Triton 本身
也支持（`JITFunction.run` 在 `if not warmup:` 之后才 launch）。但这条**具体路径**做不到。
本轮逐层试过，每修一层就冒出新的运行时假设：

| 尝试 | 撞上什么 |
| --- | --- |
| 强制 `warmup=True`（只编译不 launch） | 通过，但捕获 0 个 kernel |
| `GEMS_VENDOR=nvidia` → dispatch_key=CUDA | CPU 张量走原生实现，`use_gems()` 拦不到 |
| dispatch_key 改成 CPU | FlagGems `ops/linear.py` 内部 `torch_device_fn` 硬绑 `torch.cuda` |
| `GEMS_VENDOR=arm`（`device_name="cpu"`，走 FlagGems 自己的 CPU 分支） | 数学 shim 缺 `asin` 等符号 |
| 从 `cuda.libdevice` 补齐符号 | 撞上 autotune |
| 覆盖 `LibTuner.policy` 跳过 benchmark | 撞上 `get_empty_cache_for_benchmark` |
| 补上该方法 | 撞上 `Event() takes no arguments`（CUDA Event） |

根因不是"编译 vs 执行"，而是 **`run_and_capture` 建立在 FlagGems 的运行时派发 +
autotune 路径上**——autotune 必须实测计时才能挑 block size。要让它在纯 CPU 上工作，需要
改 FlagGems 上游、或为 CPU 预置一份 `tune_configs`，那是独立的工作量。

**而这条路径本来就是可选的**：`compile_op` 已经在做同一件事（TTIR → pim mlir），产物
与有卡 sha256 一致。`run_and_capture` 的唯一额外价值是覆盖 `GEMV_SCORE` / `SOFTMAX` /
`GEMV_CONTEXT` 三类非 GEMM 算子——而那三类在 GeneSim 里本来就用模板成本
（`UNCOVERED_OP_TYPES`），不进入 224 个 GEMM 的代价链。

#### 对测试的影响：零

全仓只有 `tests/test_genesim_bridge.py` 引用 `cost_extractor`，而它测的是**校验逻辑**
（"错位的 op_id 要在编译前拦下"，断言 `pytest.raises`），不会走到真实 kernel 捕获。

实证：真无 GPU 容器里 7B 全量 **42 passed**、快速回归 **247 passed** 全部通过，而那时
`run_and_capture` 在纯 CPU 上就是不可用的。

**好消息是主链路不受影响**：`scripts/run_full_pipeline.py` 走的就是编译路径
（`run_full_pipeline.py:159` 调 `export_pp_placement.py`），所以"GeneSim → 图编译 →
算子编译 → GeneSim"这个闭环在纯 CPU 上完整成立。

受影响的只有 `genesim/docs/llama-2.md` 的第四步（4.1 TTIR 对照 / 4.2 PIM MLIR 精化）
——那两步纯 CPU 上跑不了，要改走第六步的放置导出路径。详见 5.3 节。

#### FlagGems 的算子在纯 CPU 上不执行（推理仍然正确）

同一个原因还影响 `3-install-model-inference.sh` 跑的推理示例。无卡时它会打印：

```
note: 未检测到 GPU，本轮用 PyTorch 原生算子（FlagGems 的 Triton kernel 需要 GPU 才能执行）
```

**推理结果照样正确**，只是不经过 FlagGems 的算子——实测 7B 在纯 CPU 容器里正常生成
8 个 token，`inference_status: ok`。这一项的用途是确认"模型能加载、能推理"，而不是
"FlagGems 算子快不快"。

FlagGems 的 smoke test 同理，在无卡时退化为"验证 import 成功"。数值正确性由
NumpyBackend 对拍 PyTorch 保证，那条路径本就在 CPU 上。

### 7.2 集合通信按点对点近似

TP 的归约节点入边字节数准确，但耗时是把一次 all-reduce 当成 N 条独立点对点传输相加，
未建模环形/树形归约的优化，也未建模带宽竞争。方向上偏保守（倾向高估）。

因此**绝对耗时数字适合同类配置横向比较，不宜当作绝对性能预测**。

### 7.3 归约节点落在 GPU/CPU 上

`_attach_reduce_node` 把归约节点标成 `device_hint="gpu"`
（`gene_sim_scheduler.py:1154`），因为 PIM 侧没有 ALL_REDUCE 的 trace 编译器。物理
含义是"经主机归约"，其计算开销是估算值。

### 7.4 算子覆盖范围

IR 共 3491 个算子，进入切分导出和算子编译的只有 **224 个 GEMM**。注意力内部的 3072 个
算子（GEMV_SCORE / SOFTMAX / GEMV_CONTEXT）不拆分，改为按 head 归属重连上游
（`gene_sim_scheduler.py:967-1096`）——设计上成立（GeneSim 的 IR 本就每个 q_head 一条
独立链，边字节数已按 head_dim 算好），但它们的代价不来自算子编译器。

### 7.5 7B CPU 推理速度

128 核机器上 prefill 128 token 约 0.4 秒、decode 每步 0.1 秒量级。核数少的机器显著
变慢，7B 全量测试的 7 分 48 秒会相应拉长。只影响测试耗时，不影响结论。

### 7.6 GPU 路径保留

有卡时 `_make_ttir` 仍走原生 launch 取 `asm["ttir"]`。保留的理由是那是 Triton 自己
维护的口径，跟着上游演进最稳妥；前端路径要自己拼 signature 和特化，是无卡机器的补偿
实现。两条路径产物一致（3.1 节的 sha256 比对）。

### 7.7 一个既有的潜伏缺陷（与本次改动无关）

`genesim/src/scheduler/gene_sim_scheduler.py` 的 `_partition_attention_with_tiles`
引用了未定义的 `MAX_BULK_TRANSFER_BYTES`（该常量只在另一个函数内 import，作用域不
覆盖）。目前在死路径上——调用点被 `pim.enable_tile_range_partitioning` 门住，而
`conf/sim.yaml` 里该项为 `false`，流水执行路径本身也明确拒绝这个特性。一旦打开那个
开关就会 `NameError`。属于该单独处理的事，本次未动。
