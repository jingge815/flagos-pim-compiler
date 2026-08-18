# genesim_bridge：TTIR / pim mlir 接入说明

这次改动把 GeneSim 的成本回填，从“只支持 TTIR”扩展成了“同时支持 TTIR 和 pim mlir”。  
前者用于对照，后者用于把 PIM 侧新增的搬运信息也记录下来。

## 改动概览

```mermaid
flowchart LR
  A[GeneSim .ir / trace] --> B[refine_ir_with_flagtree.py]
  B --> C[prepare_triton_env]
  C --> D[FlagGems 触发 kernel]
  D --> E[捕获 TTIR]
  E --> F[需要时就地降成 pim mlir]
  F --> G[analyze_ir]
  G --> H[两点拟合 + 写回 .ir / sidecar]
  H --> I[run.sh 仿真]
```

核心变化：

| 变化点 | 结果 |
| --- | --- |
| `ir_level` 可切换 | 同一套流程支持 `ttir` / `pimir` |
| PIM triton 可切换 | 用 `sys.path` 前插切到带 PIM 支持的那份安装 |
| 成本分析升级 | `flops` 之外新增 `mram_traffic_bytes`、`wram_*` |
| 侧车文件升级 | 记录 PIM 参数、kernel 级信息、校验信息 |

## 修改了哪些文件

| 文件 | 主要改动 |
| --- | --- |
| `genesim_bridge/env.py` | 支持 `pim=True`，新增 `assert_pim_passes_available()` |
| `genesim_bridge/flagtree_driver.py` | 捕获 kernel 后可就地降 `pimir` |
| `genesim_bridge/ir_cost.py` | 原 `ttir_cost.py` 改名并扩展为统一 IR 分析器 |
| `genesim_bridge/cost_extractor.py` | 支持 `ir_level`，回填 PIM 侧新增字段 |
| `genesim_bridge/paths.py` | 增加 PIM 安装路径和 PIM 参数配置 |
| `genesim_bridge/__init__.py` | 导出 `assert_pim_passes_available` |
| `tests/test_genesim_bridge.py` | 补 TTIR / PIM 两层对齐测试 |
| `README.md`、`docs/genesim_bridge.md`、`CLAUDE.md` | 同步更新说明和使用入口 |

## 修改了哪些函数

| 文件 | 函数 | 作用 |
| --- | --- | --- |
| `env.py` | `prepare_triton_env()` | 负责切换普通 triton / PIM triton |
| `env.py` | `assert_pim_passes_available()` | 缺 PIM pass 直接报错，不静默降级 |
| `flagtree_driver.py` | `lower_ttir_to_pimir()` | 把捕获到的 TTIR 就地降成 pim mlir |
| `flagtree_driver.py` | `capture_kernels()` / `run_and_capture()` | 支持额外捕获 `pimir` |
| `ir_cost.py` | `analyze_ir()` | 统一分析 TTIR / pim mlir 成本 |
| `cost_extractor.py` | `_measure()` / `export_costs_to_genesim()` | 按 `ir_level` 回填成本并写 sidecar |
| `paths.py` | `pim_options()` / `describe()` | 输出 PIM 参数与来源 |

## 原理

| 项目 | 说明 |
| --- | --- |
| `flops` | 两层 IR 都从 `tt.dot` 和 `scf.for` 还原，PIM pass 不改计算，所以结果应一致 |
| `data_bytes` | 仍然表示对外净读写，不统计 tile 内重复搬运 |
| `mram_traffic_bytes` | 只在 `pimir` 下统计，表示 MRAM 和 WRAM 间真实搬运量 |
| 两点拟合 | 用 `prefill` 和 `decode` 两个代表点解系数，保证回填到 GeneSim 的是表达式而不是单标量 |

## 复现方法

```bash
cd /media/disk/fengjingge/src/genesim
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
export PATH="$HOME/.local/bin:$PATH"   # run.sh 需要 uv

python scripts/generate_builtin_gpt2_ir.py --output models/gpt2_builtin.ir
./run.sh --trace --synthetic --num_requests 10 --output traces/gpt2_builtin.trace

python scripts/refine_ir_with_flagtree.py \
  --ir models/gpt2_builtin.ir \
  --out-ir models/gpt2_builtin_flagtree.ir \
  --sidecar models/gpt2_builtin_flagtree_extensions.json --seq-len 128

python scripts/refine_ir_with_flagtree.py --ir-level pimir \
  --ir models/gpt2_builtin.ir --out-ir models/gpt2_builtin_pimir.ir \
  --sidecar models/gpt2_builtin_pimir_extensions.json --seq-len 128

./run.sh --config conf/sim_gpt2_flagtree.yaml
```

生成物：

| 文件 | 说明 |
| --- | --- |
| `models/gpt2_builtin_flagtree.ir` | TTIR 路精化结果 |
| `models/gpt2_builtin_flagtree_extensions.json` | TTIR 路侧车 |
| `models/gpt2_builtin_pimir.ir` | PIM 路精化结果 |
| `models/gpt2_builtin_pimir_extensions.json` | PIM 路侧车 |

## 当前问题

| 问题 | 现状 |
| --- | --- |
| PIM 侧只进 sidecar | 还没有把 `mram_traffic_bytes` 喂给 GeneSim 运行时 |
| `num-dpus` / `wram-bytes` 仍是记录值 | 目前不会反向改变 tile 切分 |
| WRAM 超预算不自动修复 | 只记录警告，可能仍不可执行 |
| 只验证了 fp16 | bf16 / int8 还没测 |
