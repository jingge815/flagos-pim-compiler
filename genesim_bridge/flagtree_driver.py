"""捕获 FlagGems 内核的 TTIR、PIM IR、启动网格和标量参数。"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .paths import pim_options


@dataclass
class CapturedKernel:
    """一次 launch 捕获到的编译产物。"""
    name: str
    grid: Tuple[int, ...]
    ttir: str
    constexprs: Dict[str, object]
    arg_values: Dict[str, float]   # 标量实参，供 ir_cost 求循环次数
    pimir: Optional[str] = None    # 仅 emit_pimir=True 时有值


def lower_ttir_to_pimir(
    ttir: str,
    full_m: int = -1,
    full_n: int = -1,
    full_k: int = -1,
    *,
    tile_to_budget: bool = True,
    wram_bytes: Optional[int] = None,
) -> str:
    """将 TTIR 文本转换为 PIM IR，并可按预算选择矩阵乘分块。"""
    from triton._C.libtriton import ir, passes

    options = pim_options()
    if wram_bytes is not None:
        options = {**options, "pim_wram_bytes": int(wram_bytes)}
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as handle:
        handle.write(ttir)
        path = handle.name
    try:
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(path, context)
        # 在同一 MLIR 上下文中运行编译 pass。
        module.context = context
        manager = ir.pass_manager(context)
        # 根据硬件预算和 DMA 对齐参数生成 PIM IR。
        passes.pim.add_convert_to_pim(
            manager,
            options["pim_target"],
            num_dpus=options["pim_num_dpus"],
            num_tasklets=options["pim_num_tasklets"],
            wram_bytes=options["pim_wram_bytes"],
            mram_bytes=options["pim_mram_bytes"],
            dma_align=options["pim_dma_align"],
            enable_source_remat=False,
        )
        # 为矩阵乘选择符合 WRAM 预算的分块。
        if tile_to_budget and "tt.dot" in ttir:
            passes.pim.add_tile_to_budget(
                manager, full_m=full_m, full_n=full_n, full_k=full_k
            )
        passes.pim.add_explicit_dma(manager)
        manager.run(module)
        return str(module)
    finally:
        os.unlink(path)


@contextmanager
def capture_kernels(
    emit_pimir: bool = False,
    *,
    tile_to_budget: bool = True,
    wram_bytes: Optional[int] = None,
) -> List[CapturedKernel]:
    """在上下文中捕获 FlagGems 内核启动及其可选 PIM IR。"""
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

        # 标量实参用于推导循环和分块大小。
        arg_values = {
            name: float(value)
            for name, value in bound.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        arg_values.update({
            name: float(value) for name, value in constexprs.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })

        ttir = kernel.asm["ttir"]
        pimir = None
        if emit_pimir:
            # 提供 M、N、K 的实际分块尺寸。
            full_m = int(arg_values.get("M", -1))
            full_n = int(arg_values.get("N", -1))
            full_k = int(arg_values.get("K", -1))
            pimir = lower_ttir_to_pimir(
                ttir, full_m=full_m, full_n=full_n, full_k=full_k,
                tile_to_budget=tile_to_budget, wram_bytes=wram_bytes,
            )
        captured.append(CapturedKernel(
            name=kernel.name,
            grid=grid,
            ttir=ttir,
            constexprs=dict(constexprs),
            arg_values=arg_values,
            pimir=pimir,
        ))
        return result

    LibEntry.run = patched
    try:
        yield captured
    finally:
        LibEntry.run = original


def run_and_capture(
    fn,
    emit_pimir: bool = False,
    *,
    tile_to_budget: bool = True,
    wram_bytes: Optional[int] = None,
) -> List[CapturedKernel]:
    """执行预热调用，并捕获后续调用发射的内核。"""
    import torch
    import flag_gems

    with flag_gems.use_gems():
        fn()
    torch.cuda.synchronize()

    with capture_kernels(
        emit_pimir=emit_pimir,
        tile_to_budget=tile_to_budget,
        wram_bytes=wram_bytes,
    ) as captured:
        with flag_gems.use_gems():
            fn()
        torch.cuda.synchronize()
    return list(captured)
