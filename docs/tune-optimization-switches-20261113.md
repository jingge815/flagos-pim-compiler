# 调优编译优化开关

> 文档编号：tune-optimization-switches-20261113
> 创建日期：2026-11-13
> 关联模块：`contracts/`、`opcompiler_bridge/`、`genesim_bridge/`、`tune/`

## 一、目标

调优配置新增 `optimization` 段，允许用户分别开关常量折叠、CSE、DCE 与 loop tiling。开关必须同时作用于 A 路、B 路与成本精化，并进入编译缓存键与放置 sidecar，保证“编译、精化、仿真”消费同一份优化配置。

## 二、设计决定

| 决定 | 内容 | 理由 |
| --- | --- | --- |
| 开关位置 | YAML 顶层 `optimization`，不是候选搜索维度 | 一次调优使用一套优化配置；不同配置换输出目录即可对比 |
| 可开关项 | `constant_fold`、`cse`、`dce`、`loop_tiling` | 对应五类优化中可安全关闭的四类 |
| 强制项 | `-pim-fuse-activation`、`-pim-expand-phases`、`-pim-verify-gml-contract` | 融合与相位展开是目标格式硬要求，关掉会产出失真产物 |
| 缺省值 | 四类全部 `true` | 旧配置、旧命令与改动前逐字一致 |
| 身份 | 开关进入 `compile_key`、算子缓存键与 sidecar | 换开关必须重编，不能复用旧产物 |

## 三、数据流

```text
tune YAML
  optimization: {constant_fold, cse, dce, loop_tiling}
        │
        ▼
tune.config.load_config ──► TuneConfig.optimization (PimOptimizations)
        │
        ├── compile_key() 加 optimization.to_payload()
        │
        └── HardwareProfile.to_pim_hardware_config(optimizations=...)
                 │
                 ▼
          PIMHardwareConfig.optimizations
                 │
     ┌───────────┼───────────────────────────┐
     ▼           ▼                           ▼
 A 路 pass    B 路 pass                算子缓存键 / sidecar
 a_path_passes b_path_passes           compiler.hardware.optimizations
     │           │                           │
     └───────────┴──────────────┬────────────┘
                                ▼
                    GeneSim 仿真前契约核对
```

## 四、关键结构体

```python
@dataclass(frozen=True)
class PimOptimizations:
    constant_fold: bool = True
    cse: bool = True
    dce: bool = True
    loop_tiling: bool = True
```

`PimOptimizations` 放在 `contracts/op_contract.py`，作为 `PIMHardwareConfig` 的字段。这样 A/B 两路、成本精化、算子缓存键和放置 sidecar 都从同一个编译请求取开关，不另立第二份状态。

## 五、改动文件

| 文件 | 改动 |
| --- | --- |
| `contracts/op_contract.py` | 新增 `PimOptimizations`；`PIMHardwareConfig` 增加 `optimizations` 字段，payload 读写兼容旧产物 |
| `contracts/hardware_profile.py` | `to_pim_hardware_config(..., optimizations=...)`，缺省全开 |
| `opcompiler_bridge/pipeline.py` | `a_path_passes`/`b_path_passes` 按开关剔除 pass；顺序与强制项不变 |
| `opcompiler_bridge/driver.py` | A/B 两路把请求里的开关传给 pass 列表 |
| `genesim_bridge/flagtree_driver.py` | 进程内 A/B 精化路径按同一开关加减 pass |
| `genesim_bridge/placement_export.py` | sidecar 的 `compiler.hardware.optimizations` 记录生效开关 |
| `genesim_bridge/refine_profile.py` | 成本精化从放置 sidecar 读取开关，旧 sidecar 缺块时按全开 |
| `tune/config.py` | 解析可选 `optimization` 段；未知字段编译前拒绝；进入生效配置与运行身份 |
| `tune/steps.py` | `compile_key`、候选编译、成本精化、仿真相对于同一份开关；`verify_compiled` 核对 sidecar 开关 |
| `runtime/kernels.py` | 硬件 payload 含嵌套开关，已编译内核缓存键改为稳定 JSON 串 |
| `examples/tune/*.yaml` | 三份示例增加 `optimization` 段 |
| `docs/tune-usage.md` | 增加“编译优化开关”使用说明 |

## 六、行为说明

- `cse` 同时控制上游 `-cse` 与 `-pim-cse`；`loop_tiling` 控制 `-pim-tile-to-budget`。
- `constant_fold` 只控制 `-pim-constant-fold`；`dce` 只控制 `-pim-dce`。
- 关闭 loop tiling 后，`full_m/full_n/full_k` 形状提示不再有分块 pass 可下发；A/B 两路仍会完成转换，但大 kernel 可能因 WRAM 超预算在后续 pass 失败，这是显式选择。
- 开关改变会使 `compile_key`、算子 `.so`/PIM MLIR 缓存键与 `sim_key` 全部变化，旧产物不会命中。
- 侧车 `compiler.hardware.optimizations` 与候选配置不一致时，`verify_compiled` 在仿真前拒绝。

## 七、测试

| 测试 | 覆盖 |
| --- | --- |
| `tests/test_op_contract.py` | 默认全开、payload 往返、旧 payload 兼容、非法开关 |
| `tests/test_tune_config.py` | 缺省全开、显式关闭、未知开关拒绝、生效配置与运行身份 |
| `tests/test_tune_compile_bridge.py` | A/B 两路按开关剔除 pass、强制项保留、算子缓存键随开关变化 |
| `tests/test_tune_steps.py` | `compile_key` 随开关变化、侧车开关不一致时 `verify_compiled` 拒绝 |
| `tests/test_hardware_profile.py` | 画像派生入口传递开关 |
| `tests/test_placement_export.py` | sidecar 记录 `compiler.hardware.optimizations` |

本轮执行结果：`test_op_contract`、`test_tune_config`、`test_tune_compile_bridge`、`test_hardware_profile` 共 101 条通过；`test_tune_steps` 129 条通过；`test_tune_loop`、`test_tune_report`、`test_tune_sweep` 共 107 条通过；`test_tune_candidate`、`test_tune_inputs`、`test_tune_calibration` 共 96 条通过。算子编译/放置回归 `test_opcompiler_linear`、`test_opcompiler_ops`、`test_placement_export` 为 188 通过、1 跳过。另用一条真实 A 路编译对比确认：默认配置产出 `pim.tile-*`，四类优化全关后同一请求仍能编译且不写分块属性。

执行：

```bash
source "$(python3 -c 'import json; print(json.load(open("paths.json"))["pytorch_env_script"])')"
python -m pytest tests/test_op_contract.py tests/test_tune_config.py tests/test_tune_compile_bridge.py tests/test_hardware_profile.py -q
python -m pytest tests/test_tune_steps.py tests/test_tune_loop.py tests/test_tune_report.py tests/test_tune_sweep.py -q
```

## 八、当前不足

- 开关是运行级配置，不是候选搜索维度；要比较不同开关组合需要多次运行并隔离输出目录。
- 目前只开放四类优化；`enable-source-remat` 等 FlagTree 选项仍未接入。
- 关闭 CSE/DCE/常量折叠后产物更大，GeneSim 的 trace 与成本可能随之变化；这是预期行为，但还没有逐开关的性能对照数据。
- B 路 tasklet/DPU 编码与多 DPU 验收缺口仍需单独处理，不属于本次开关范围。
