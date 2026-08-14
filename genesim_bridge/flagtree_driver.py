"""FlagTree 编译驱动：按目标 shape 触发 FlagGems 算子，捕获 TTIR + grid。

方案依据：spec.md 问题 4 三.(2)。不手搓 ASTSource——那会绕过 FlagGems 的
autotune，编出的 tile 配置与实跑不符（实测 linear 在 Tq=1 选 BLOCK_M=64、
Tq=128 选 BLOCK_M=32，成本差 64 倍）。因此这里用目标 shape 的真张量调
FlagGems 算子，在 `LibEntry.run` 处捕获真实 grid、autotune 选中的 constexprs
与 `CompiledKernel.asm['ttir']`。

不改 FlagGems / FlagTree 源码，只在运行时包一层 LibEntry.run。
环境要求见 env.py。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class CapturedKernel:
    """一次 launch 捕获到的编译产物。"""
    name: str
    grid: Tuple[int, ...]
    ttir: str
    constexprs: Dict[str, object]
    arg_values: Dict[str, float]   # 标量实参，供 ttir_cost 求循环次数


@contextmanager
def capture_kernels() -> List[CapturedKernel]:
    """在 with 块内捕获所有 FlagGems kernel launch。

    产出列表在退出 with 后仍可读。包装 LibEntry.run 而非 JITFunction.run：
    FlagGems 命中自身 kernel_cache 后直接 `kernel[grid](...)` 发射，
    不再经过 JITFunction.run，那一层挂钩子会漏掉所有热路径 launch。
    """
    from flag_gems.utils.libentry import LibEntry

    captured: List[CapturedKernel] = []
    original = LibEntry.run

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        kernel, constexprs = result

        grid = kwargs["grid"]
        bound = {**dict(zip(self.arg_names, args)), **kwargs}
        if callable(grid):
            grid = grid({**bound, **constexprs})
        grid = tuple(int(g) for g in tuple(grid)[:3])

        # 只留标量实参：张量参数在 TTIR 里是指针，对循环次数没有贡献
        arg_values = {
            name: float(value)
            for name, value in bound.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        arg_values.update({
            name: float(value) for name, value in constexprs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })

        captured.append(CapturedKernel(
            name=kernel.name,
            grid=grid,
            ttir=kernel.asm["ttir"],
            constexprs=dict(constexprs),
            arg_values=arg_values,
        ))
        return result

    LibEntry.run = patched
    try:
        yield captured
    finally:
        LibEntry.run = original


def run_and_capture(fn) -> List[CapturedKernel]:
    """跑 fn 两遍，返回第二遍捕获的 kernel。

    第一遍让 autotune 定下最优 config（此间会 bench 多个候选配置，捕获到的
    是噪声）；第二遍走 FlagGems 缓存，只发射选中的那个 kernel。
    """
    import torch
    import flag_gems

    with flag_gems.use_gems():
        fn()
    torch.cuda.synchronize()

    with capture_kernels() as captured:
        with flag_gems.use_gems():
            fn()
        torch.cuda.synchronize()
    return list(captured)
