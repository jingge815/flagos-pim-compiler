# 站点路径可移植性（2026-10-15）

纯 CPU 的 Ubuntu 上跑本仓时，机器相关的绝对路径只允许出现在 `paths.json`。
测试、脚本、操作文档不再写 `/media/disk/fengjingge/...`。

## 怎么配

根目录 `paths.json` 是唯一配置文件，环境变量同名覆盖（`PYTORCH_ENV_SCRIPT` 等）。

| 键 | 用途 |
| --- | --- |
| `pytorch_env_script` | `env-pytorch.sh` |
| `llama2_7b_model_dir` | Llama-2-7B 权重 |
| `flagtree_prefix` | FlagTree 安装根 |
| `genesim_root` | GeneSim 仓库根 |
| `gml_reference_dir` | GML 结构参考 |
| `gml_llama2_reference_dir` | llama2 decode-block 参考 |
| `gml_ours_graph` | 本机一次 decode-block 导出（可选，不配则对拍测试跳过） |

FlagTree **源码树**不另配：从 `$flagtree_prefix/env-flagtree.sh` 读 `FLAGTREE_SOURCE`。

加载环境：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
```

## 改了什么

| 位置 | 原先 | 现在 |
| --- | --- | --- |
| `tests/test_fusion_contract.py` | 写死 FlagTree 源码路径 | `flagtree_source()` |
| `tests/test_pimir_layout.py` | 写死 GeneSim 调度器路径 | `genesim_root()` |
| `tests/test_gml_node_parity.py` / `test_gml_export.py` | 写死 `/tmp` 或本机导出 | `gml_ours_graph` |
| `scripts/verify_cpu_only.sh` | 写死 GeneSim / 安装前缀 | 读 `paths.json` |
| `scripts/refine_ir_with_flagtree.py` | 用法里写死 `.ir` 路径 | `$GENESIM_ROOT` |
| genesim `scripts/refine_ir_with_flagtree.py` | `DEFAULT_BRIDGE_ROOT` 写死本机 | `PIM_COMPILER_ROOT`，否则按相对布局探测 |
| genesim `docs/llama-2.md` / `gpt2.md` | 可执行命令里的本机路径 | 从 `paths.json` 读 |

`tests/test_paths.py` 增加守卫：除 `paths.json` 外，本仓可执行代码不得再出现 `/media/disk/fengjingge`。

## 测试

- `tests/test_paths.py`：配置读取 + 守卫
- `tests/test_fusion_contract.py`：融合表仍与 FlagTree 源码对齐
- `tests/test_pimir_layout.py::test_the_simulator_actually_consumes_the_returned_fields`：经 `genesim_root()` 读调度器

## 未改

- 历史文档 `docs/`、genesim `docs/superpowers/` 里的过期命令，不当作可执行入口。
- FlagTree 安装脚本本身按 `SCRIPT_DIR` 相对定位，没有本机绝对路径。
