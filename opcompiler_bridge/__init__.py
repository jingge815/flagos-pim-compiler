"""图层编译器 → 算子编译器桥接（本仓问题 4/5 数值闭环的第一期垂直切片）。

跟 `genesim_bridge`（问题 4 成本桥接，只读 TTIR/pim mlir 做静态估算，不产出可
执行代码）是两条独立、互不依赖的桥：这里产出的是能在 `backend/hal_numpy.py`
上真正被 `ctypes` 加载执行的 `.so`，替换 `runtime/kernels.py` 里对应算子的手写
NumPy 实现。当前只覆盖 `linear`——范围收窄的理由见 `contracts/op_contract.py`
与 `docs/opcompiler_bridge-20260825.md`。

用法见 `driver.py` 顶部的模块 docstring；`kernel_src.py` 是驱动编译的最小
Triton kernel 源码。
"""

from .driver import compile_op

__all__ = ["compile_op"]
