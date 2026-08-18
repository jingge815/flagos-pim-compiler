"""GeneSim 成本桥接（方案问题 4）。

把 FlagGems + FlagTree 的编译产物（TTIR 或 pim mlir）测出的算子成本回填进
GeneSim 的 ModelIR，不改 GeneSim 的图结构、schema 与动态 trace 两条原路。

用法见 genesim/scripts/refine_ir_with_flagtree.py。

注意调用顺序：`prepare_triton_env(pim=True)` 必须在任何 `import triton` 之前
调用，否则切不到带 PIM 支持的那份 triton 安装。本包各模块的 `import triton`
/ `import torch` 一律写在函数体内，所以 `import genesim_bridge` 本身安全；
但调用方自己不要在 prepare_triton_env 之前 import triton。
"""

from .cost_extractor import export_costs_to_genesim
from .env import assert_pim_passes_available, prepare_triton_env
from .op_classify import ShapePoint

__all__ = [
    "assert_pim_passes_available",
    "export_costs_to_genesim",
    "prepare_triton_env",
    "ShapePoint",
]
