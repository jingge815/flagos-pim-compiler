"""FlagTree 编译驱动：按目标 shape 触发 FlagGems 算子，捕获 TTIR / pim mlir + grid。

方案依据：spec.md 问题 4 三.(2)。不手搓 ASTSource——那会绕过 FlagGems 的
autotune，编出的 tile 配置与实跑不符（实测 linear 在 Tq=1 选 BLOCK_M=64、
Tq=128 选 BLOCK_M=32，成本差 64 倍）。因此这里用目标 shape 的真张量调
FlagGems 算子，在 `LibEntry.run` 处捕获真实 grid、autotune 选中的 constexprs
与 `CompiledKernel.asm['ttir']`。

**pim mlir（第 2 步）在同一个 hook 里就地降。** 捕获到 TTIR 文本后重新 parse
成 module，再跑 `convert-triton-to-pim` + `pim-explicit-dma` 两个 pass，与
FlagTree 的 `pim_sidecar` 走的是同一条 pass 序列。这样做而不是读 sidecar dump
的 `.pimir` 文件，有三个理由：

- 不依赖 `FLAGTREE_EMIT_PIM` 与 `TRITON_DUMP_DIR`；
- 不依赖编译缓存 miss——热路径 launch 命中缓存时 `make_ttir()` 根本不跑，
  sidecar 也就不会 dump；
- `pim_sidecar.emit_pim_ir` 吞掉所有异常只发 warning，pass 失败会静默变成
  「没有 pim mlir」，而这里直接抛，让问题暴露在测试里。

不改 FlagGems / FlagTree 源码，只在运行时包一层 LibEntry.run。
环境要求见 env.py：pim mlir 路要先 `prepare_triton_env(pim=True)`。
"""

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


def lower_ttir_to_pimir(ttir: str) -> str:
    """把一份 TTIR 文本降到 pim mlir 文本。

    走 `ir.parse_mlir_module` 而不是直接对 `CompiledKernel` 里的 module 对象
    动手：`asm['ttir']` 是文本，拿不到那个 module；而 pass 会原地改 module，
    在主编译路径的 module 上跑会污染 GPU 编译（FlagTree 自己的 sidecar 因此
    先 clone 一份）。从文本重新 parse 天然隔离，且实测与 sidecar 的产物结构
    等价（`wram-bytes-used` 逐位相同）。

    `parse_mlir_module` 只接受文件路径，故先落一个临时文件。
    """
    from triton._C.libtriton import ir, passes

    options = pim_options()
    with tempfile.NamedTemporaryFile("w", suffix=".ttir", delete=False) as handle:
        handle.write(ttir)
        path = handle.name
    try:
        context = ir.context()
        ir.load_dialects(context)
        module = ir.parse_mlir_module(path, context)
        # `context` 是 Python 侧的动态属性，用于让 MLIRContext 与 module 同生命周期；
        # parse 出来的 module 没有它，后面 pass_manager 要用，必须显式挂上
        module.context = context
        manager = ir.pass_manager(context)
        passes.pim.add_convert_to_pim(
            manager,
            options["pim_target"],
            options["pim_num_dpus"],
            options["pim_num_tasklets"],
            options["pim_wram_bytes"],
            False,
        )
        passes.pim.add_explicit_dma(manager)
        manager.run(module)
        return str(module)
    finally:
        os.unlink(path)


@contextmanager
def capture_kernels(emit_pimir: bool = False) -> List[CapturedKernel]:
    """在 with 块内捕获所有 FlagGems kernel launch。

    产出列表在退出 with 后仍可读。包装 LibEntry.run 而非 JITFunction.run：
    FlagGems 命中自身 kernel_cache 后直接 `kernel[grid](...)` 发射，
    不再经过 JITFunction.run，那一层挂钩子会漏掉所有热路径 launch。

    emit_pimir=True 时每个捕获到的 kernel 额外降一份 pim mlir。
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

        ttir = kernel.asm["ttir"]
        captured.append(CapturedKernel(
            name=kernel.name,
            grid=grid,
            ttir=ttir,
            constexprs=dict(constexprs),
            arg_values=arg_values,
            pimir=lower_ttir_to_pimir(ttir) if emit_pimir else None,
        ))
        return result

    LibEntry.run = patched
    try:
        yield captured
    finally:
        LibEntry.run = original


def run_and_capture(fn, emit_pimir: bool = False) -> List[CapturedKernel]:
    """跑 fn 两遍，返回第二遍捕获的 kernel。

    第一遍让 autotune 定下最优 config（此间会 bench 多个候选配置，捕获到的
    是噪声）；第二遍走 FlagGems 缓存，只发射选中的那个 kernel。第一遍不开
    emit_pimir——那些候选配置的 pim mlir 是纯浪费。
    """
    import torch
    import flag_gems

    with flag_gems.use_gems():
        fn()
    torch.cuda.synchronize()

    with capture_kernels(emit_pimir=emit_pimir) as captured:
        with flag_gems.use_gems():
            fn()
        torch.cuda.synchronize()
    return list(captured)
