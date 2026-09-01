"""将图层算子编译为可由运行时加载的共享库。"""

from .driver import compile_op

__all__ = ["compile_op"]
