"""融合条件表的唯一真源。

主算子与可折激活原先在 `graph/fuse.py`、`graph/fuse_pim.py` 各存一份，改一处
另一处不会跟着变；这里合并成一份，两个 pass 直接 import，FlagTree 侧
`FuseActivation.cpp` 的 `isFusionTarget` 必须同步（见该函数上方注释）。

合并依据：两份表不一致处，取覆盖面更广、且与 PIM 语义一致的那份。
- 主算子：取 `fuse.py` 的 5 个（matmul 类 + eltwise 类），它是 `fuse_pim.py`
  那 3 个 matmul 类的超集——GML 里 Gemm/MatMul 与 EltwiseAdd/Mul 都带
  contraction 块，都能持有折入的激活。
- 可折激活：按主算子分两张表。通用表取 `fuse.py` 那 7 个（覆盖面更广），门控表
  保留 `fuse_pim.py` 那 3 个——它多了 `silu`，而 `silu` 只折 gate 投影，不能进
  通用表。**两张表的值一律小写**，GML 那边的首字母大写由
  `from_fx._ACTIVATION_NAMES` 统一转，`oplevel_emitter` 则统一取小写当 mnemonic
  ——两处都只认小写键，这里写大写会静默绕过归一化。`rsqrt` 两张表都没有，
  RMSNorm 因此保持独立节点。
"""

from __future__ import annotations

import torch

# 主算子：能持有折入激活的算子。
FUSION_TARGETS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
    }
)

# 通用可折激活 → GML 名。折进主算子 contraction 块，块名由 GML 名拼出。
ACTIVATIONS = {
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.gelu.default: "gelu",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.reciprocal.default: "reciprocal",
}

# gate 投影的主算子 = 主算子表里的 matmul 类（比它少 eltwise）。
GATE_TARGETS = frozenset(
    {
        torch.ops.aten.addmm.default,
        torch.ops.aten.linear.default,
        torch.ops.aten.mm.default,
    }
)

# 门控可折激活 → 激活名。`silu` 只在这里，所以只折进 gate 投影。
#
# 值统一小写（评审 20260923 的 P2-1）。原来这三个写成 `Silu` / `Relu` / `Gelu`，
# 与上面那张表的小写不一致，而下游 `from_fx._ACTIVATION_NAMES` 是一张**小写键**
# 的表：`get("Silu")` 不命中，`from_fx.py` 那行 `get(name, name)` 让它原样透出。
# `Silu` 恰好就是 GML 要的拼写，所以结果对——**靠的是巧合**，归一化那一层
# 实际上被绕过了。多一个不巧合的（比如 `LeakyRelu`）就会静默写出对方解析器
# 找不到的块名。
GATE_ACTIVATIONS = {
    torch.ops.aten.silu.default: "silu",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.gelu.default: "gelu",
}
