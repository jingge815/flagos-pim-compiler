"""GeneSim 成本桥接（方案问题 4）。

把 FlagGems + FlagTree 的编译产物（TTIR）测出的算子成本回填进 GeneSim
的 ModelIR，不改 GeneSim 的图结构、schema 与动态 trace 两条原路。

用法见 genesim/scripts/refine_ir_with_flagtree.py。
"""

from .cost_extractor import export_costs_to_genesim
from .env import prepare_triton_env
from .op_classify import ShapePoint

__all__ = ["export_costs_to_genesim", "prepare_triton_env", "ShapePoint"]
