# 融合条件表单点维护（2026-09-22）

对应方案 `docs/flagtree-pimmlir-primitives-impl-20260922.md` §4.10、§1.2.3。

## 改了什么

原先三份融合条件表各写各的：`graph/fuse.py` 一份、`graph/fuse_pim.py` 一份、
FlagTree `FuseActivation.cpp` 一份。改一处另外两处不会跟着变，于是把前两份合并
成 `contracts/fusion_contract.py`，两个 pass 直接 import；FlagTree 的 C++ 不读
Python，靠注释声明同步关系。

| 文件 | 改动 |
| --- | --- |
| `contracts/fusion_contract.py` | 新增。四份常量（见下表） |
| `graph/fuse.py` | 删本地 `FUSION_TARGETS` / `ACTIVATIONS`，改 import；逻辑不动 |
| `graph/fuse_pim.py` | 删本地 `FUSABLE_ACTIVATIONS` / `ACTIVATION_HOSTS`，改 import `GATE_ACTIVATIONS` / `GATE_TARGETS`；逻辑不动 |
| `tests/test_fusion_contract.py` | 新增。10 条断言 |
| FlagTree `lib/Dialect/TritonPIM/Transforms/FuseActivation.cpp` | `isFusionTarget` 上方加一行同步注释，行为不变 |

## 表怎么合的

两份原表有三处不一致，合的依据是「取覆盖面更广、且与 PIM 语义一致的那份」：

| 项 | `fuse.py` | `fuse_pim.py` | 合并结果 |
| --- | --- | --- | --- |
| 主算子 | `addmm/linear/mm/add/mul`（5 个） | `linear/addmm/mm`（3 个） | 取 5 个的（matmul 类 + eltwise 类），GML 里 Gemm/MatMul 与 EltwiseAdd/Mul 都带 contraction 块 |
| `silu` | **不在**表里 | 在表里 | 只进 `GATE_ACTIVATIONS`（门控表），通用表仍不含它 |
| `relu` / `gelu` 的拼写 | 小写（`"relu"`） | 首字母大写（`"Relu"`） | 保留小写；下游 `from_fx._ACTIVATION_NAMES` 与 `oplevel_emitter` 都按小写归一，GML 产物不变 |

`rsqrt` 两份原表都没有，现在也没有：RMSNorm 由 `fuse_pim.py` 的
`RMS_NORM_CHAIN` 折成自己的锚点，保持独立节点。

契约里四份常量，分成两组是因为「silu 只折 gate」这条语义必须靠表结构表达：

```
FUSION_TARGETS   主算子（matmul 类 + eltwise 类）        ← fuse.py 用
ACTIVATIONS      通用可折激活（不含 silu）                ← fuse.py 用
GATE_TARGETS     门控主算子 = 主算子里的 matmul 类        ← fuse_pim.py 用
GATE_ACTIVATIONS 门控可折激活（含 silu）                  ← fuse_pim.py 用
```

`silu` 只出现在门控表、而门控主算子只有 matmul 类，所以它只会折进 gate 投影
（llama2 里就是 `mlp.gate_proj` 那个 `Gemm`），不会折进 eltwise；反过来，
`fuse.py` 的通用表没有 `silu`，走通用路径的图仍旧把 `silu` 当独立节点发
（`OP_TYPES[silu] = "Silu"` 那条兜底不变）。

```
FX 图 ──fuse_for_pim──▶ 门控表：linear(addmm/mm) + {Silu, Relu, Gelu}
      └─fuse_graph────▶ 通用表：linear/addmm/mm/add/mul + 7 个激活（无 silu）
```

## 测试

`tests/test_fusion_contract.py`：

- `is` 同一性：`fuse.FUSION_TARGETS is fusion_contract.FUSION_TARGETS` 等四条，
  证明两个 pass 拿到的是契约里的那个对象而不是各自的副本；另断言旧名字
  `ACTIVATION_HOSTS` / `FUSABLE_ACTIVATIONS` 已删。
- 逐元素比对：四份表与合并前原表（硬编码在测试里）完全相等。
- `silu 只折 gate`：`silu` 在门控表、不在通用表；门控表的主算子只有 matmul 类
  （eltwise 不在其中）；真实 llama2 小图上唯一的 `silu` 折进主算子 `linear_4`，
  其权重是 `model.model.layers.0.mlp.gate_proj.weight`；反证是同一张图跑
  `fuse_graph`（通用表）时 `silu` 仍是独立节点。
- `rms 独立`：`rsqrt` 不在任何可折激活表里；图上 3 条 RMSNorm 折成 3 个锚点，
  所有 `rsqrt` 都被标记为已吸收。

回归：`python -m pytest tests/ -q -k "not llama2_7b"` 全绿（767 passed）；单独跑
`tests/test_fuse.py tests/test_fuse_pim.py tests/test_gml_from_fx.py` 37 passed。

## 不足

- FlagTree 侧仍是第二份真源：C++ 不读 Python，只能靠注释 + lit
  `fuse_activation.mlir` 与 `tests/test_gml_from_fx.py` 对拍同一组用例。
- 方案 §1.2.3 的清单里还有 `identity`（两条原表都没有），本轮按「原样抄」没有
  加；`sqrt` 在通用表里、方案清单没列，同样保留原状。
- FlagTree 的 `isFusionTarget` 认 `MatmulOp/ConvOp/EltwiseOp`，本仓主算子表没有
  `aten.convolution`（ResNet 路径也没折过 Conv 激活），两边口径待后续对齐。
