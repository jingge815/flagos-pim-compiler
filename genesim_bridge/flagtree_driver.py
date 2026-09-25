"""捕获 FlagGems 内核的 TTIR、PIM IR、启动网格和标量参数。"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
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

    _check_inprocess_matches_triton_opt()

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


# 进程内 libtriton 与 `triton-opt` 是否同源（评审 20260923 的 P1-2）。
# None = 还没查过；True/False = 查过的结论。
_INPROCESS_CHECKED: bool | None = None

# 探针属性：选一个**后期加进方言**的属性。它在两份二进制之间最容易出现差异，
# 而早期就有的属性（`#pim.datapath` 之类）两边都认，探不出落后。
_PROBE_ATTR = "stationarity"

# 方言源码：这些文件变了，两份二进制就该重建。
_PIM_SOURCE_GLOBS = (
    "include/triton/Dialect/TritonPIM/**/*.td",
    "include/triton/Dialect/TritonPIM/**/*.h",
    "lib/Dialect/TritonPIM/**/*.cpp",
    "lib/Conversion/TritonToTritonPIM/**/*.cpp",
)


def _newest_pim_source_mtime() -> float | None:
    """方言源码里最新的修改时刻。取不到返回 None（不阻断，交给属性探针）。"""
    from .paths import flagtree_source

    root = flagtree_source(required=False)
    if root is None:
        return None
    stamps = [p.stat().st_mtime
              for pattern in _PIM_SOURCE_GLOBS
              for p in root.glob(pattern)]
    return max(stamps) if stamps else None


def _check_binding_is_not_older_than_the_dialect() -> None:
    """进程内那份 `libtriton.so` 不能比方言源码旧。

    这是「同源」唯一判得准的口径：它问的是「绑定是不是用当前源码建的」，
    而不是「绑定认不认我挑的那一个属性」。属性探针探不出的落后正是后者——
    挑中的属性加得早，两边都认，探针就不响了。
    """
    import triton

    from .paths import flagtree_source

    binding = Path(triton.__file__).parent / "_C" / "libtriton.so"
    if not binding.is_file():
        return
    newest = _newest_pim_source_mtime()
    if newest is None:
        return
    behind = newest - binding.stat().st_mtime
    if behind > 0:
        from .paths import describe

        raise RuntimeError(
            f"进程内 libtriton 比方言源码旧 {behind / 3600:.1f} 小时：\n"
            f"  绑定: {binding}\n"
            f"  方言源码最新: {flagtree_source()} 下 {newest:.0f}\n"
            f"A 路成本抽取用的是这份绑定，继续跑会量出与 B 路不一致的数字"
            f"而不报错。\n重装绑定：flagOS-installers/0-install-flagtree.sh\n"
            f"当前生效路径：\n{describe()}"
        )


def _check_inprocess_matches_triton_opt() -> None:
    """进程内 `libtriton` 与 `triton-opt` 认同一组 pim 属性，否则抛。

    A 路走进程内 `libtriton`（`from triton._C.libtriton import ir, passes`），
    B 路走 `triton-opt` 可执行文件。两者是**分别构建**的：实测某一轮里
    `triton-opt` 是 20:05 的产物而两份 `libtriton.so` 都停在 16:01，后者对
    `#pim.stationarity` 报 `unknown attribute`。

    A 路当时不报错，因为它只用 tile 级 pass，撞不到新属性——**那是巧合，
    不是设计**。哪天 A 路的 pass 开始读新属性，进程内那份会解析失败或者
    静默忽略，量出来的成本就对不上 B 路，而没有任何地方会说这件事。

    所以在这里查一次：不一致直接抛，让「该重装绑定了」变成一句错误信息，
    而不是一份数字对不上的 sidecar。

    **两道检查，缺一不可**：

    1. 先比时间戳——`libtriton.so` 必须比方言源码新。属性探针只能探到**它探的
       那一个**属性，落后得不够久就探不出来：实测 `libtriton.so` 落后九个多
       小时、已经缺 `pim.dynamic_quant` 算子、也不认新增的值域校验，探针却
       照样返回成功。时间戳判的是「这份绑定是不是用当前源码建的」，与探什么
       无关。
    2. 属性探针留着兜时间戳判不出的情形——源码没动但绑定没重编（例如构建
       中途失败后只补了 `triton-opt`）。
    """
    global _INPROCESS_CHECKED
    if _INPROCESS_CHECKED is not None:
        return

    from triton._C.libtriton import ir

    _check_binding_is_not_older_than_the_dialect()

    probe = (f'module {{\n'
             f'  tt.func @probe(%a: tensor<4x8xf16>, %b: tensor<8x4xf16>) {{\n'
             f'    %0 = pim.matmul %a, %b '
             f'{{datapath = #pim.datapath<nmuMode = floating_point, '
             f'scaleMode = floating_point>, '
             f'weightBinding = #pim.weight_binding<format = weight, '
             f'role = model_weight, elemBits = 4>, '
             f'{_PROBE_ATTR} = #pim.{_PROBE_ATTR}<weight>}}'
             f' : tensor<4x8xf16>, tensor<8x4xf16> -> tensor<4x4xf16>\n'
             f'    tt.return\n  }}\n}}\n')

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(probe)
        path = handle.name
    try:
        context = ir.context()
        ir.load_dialects(context)
        try:
            ir.parse_mlir_module(path, context)
        except Exception as exc:  # noqa: BLE001 - 任何解析失败都算落后
            _INPROCESS_CHECKED = False
            raise RuntimeError(
                f"进程内 libtriton 不认 `#pim.{_PROBE_ATTR}`，说明它比 "
                f"triton-opt 落后（两者分别构建）。A 路成本抽取用的是进程内那"
                f"份，继续跑会量出与 B 路不一致的数字而不报错。\n"
                f"重装绑定：flagOS-installers/0-install-flagtree.sh\n"
                f"原始错误: {exc}"
            ) from exc
    finally:
        os.unlink(path)
    _INPROCESS_CHECKED = True


def lower_oplevel_to_pimir(mlir_text: str) -> str:
    """将**整算子级** PIM IR 展开成相位 SSA，返回 pimir 文本。

    与 `lower_ttir_to_pimir` 是两条不同的入口：那条从 Triton kernel 出发，
    经布局转换、分块与显式 DMA；这条从图编译器发的整算子级 IR 出发，没有
    DMA 可显式化，走的是融合 → 展开 → 校验三步。

    顺序不能反：展开后主算子已经变成相位链，`-pim-fuse-activation` 认的
    `pim.matmul + pim.lut` 模式就不在了；而校验查的跨属性不变量（相位数连续、
    动态量化的相 1/2 扇出读相 0）展开前根本不存在。

    **跑的是 `triton-opt` 可执行文件，不是进程内的 `triton._C.libtriton`。**
    两者不是同一份构建：算子编译器（`driver._run_oplevel_triton_opt`）走的是
    前者，而 pytorch 环境里那份 libtriton 常常落后于 FlagTree 源码，认不出
    `stationarity` / `pim.gather` / transpose 的 `purpose`。用进程内那份展开，
    genesim 量到的就不是算子编译器真正产出的 IR——成本看着正常，对的是另一份
    产物。pass 列表与 `driver._run_oplevel_triton_opt` 逐字相同。
    """
    from opcompiler_bridge.driver import _run_passes, _triton_opt

    return _run_passes(
        _triton_opt(),
        mlir_text,
        ["-pim-fuse-activation", "-pim-expand-phases",
         "-pim-verify-gml-contract"],
        "整算子级 pim mlir",
    )


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
    """执行预热调用，并捕获后续调用发射的内核。

    无 GPU 机器上先注入编译期 driver（见 `opcompiler_bridge/cpu_host.py`），否则
    `import flag_gems` 在 import 期就会去读 `triton.runtime.driver.active` 而抛
    "0 active drivers"。同样地，`torch.cuda.synchronize()` 只在真有卡时才有意义——
    无卡时 CPU 执行本来就是同步的，调用它会直接抛 "No CUDA GPUs are available"。
    """
    from opcompiler_bridge.cpu_host import ensure_compile_driver, gpu_hardware_present

    ensure_compile_driver()

    import torch
    import flag_gems

    has_gpu = gpu_hardware_present()

    def _sync() -> None:
        if has_gpu:
            torch.cuda.synchronize()

    with flag_gems.use_gems():
        fn()
    _sync()

    with capture_kernels(
        emit_pimir=emit_pimir,
        tile_to_budget=tile_to_budget,
        wram_bytes=wram_bytes,
    ) as captured:
        with flag_gems.use_gems():
            fn()
        _sync()
    return list(captured)
