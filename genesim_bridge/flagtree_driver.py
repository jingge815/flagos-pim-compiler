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


def lower_ttir_to_pimir(
    ttir: str,
    full_m: int = -1,
    full_n: int = -1,
    full_k: int = -1,
    *,
    tile_to_budget: bool = True,
    wram_bytes: Optional[int] = None,
) -> str:
    """把一份 TTIR 文本降到 pim mlir 文本。

    走 `ir.parse_mlir_module` 而不是直接对 `CompiledKernel` 里的 module 对象
    动手：`asm['ttir']` 是文本，拿不到那个 module；而 pass 会原地改 module，
    在主编译路径的 module 上跑会污染 GPU 编译（FlagTree 自己的 sidecar 因此
    先 clone 一份）。从文本重新 parse 天然隔离，且实测与 sidecar 的产物结构
    等价（`wram-bytes-used` 逐位相同）。

    `parse_mlir_module` 只接受文件路径，故先落一个临时文件。

    `full_m`/`full_n`/`full_k`：算子真实的（未切分的）M/N/K，由调用方从
    launch 实参里按名字（`"M"`/`"N"`/`"K"`）取出后传入，默认 -1（不提供）。
    `add_tile_to_budget` 在 IR 结构推断不出这些值时需要它们——真实 FlagGems
    kernel（如 `linear_kernel`）用 2-D launch grid 切分 M/N、K 是运行期标量
    kernel 参数，两者都不是 IR 里能看到的编译期常量，纯靠 IR 结构走
    `scf.for`/`tt.dot` 是推断不出来的（见 TileToBudget.cpp::inferFullShape
    的说明）。

    `tile_to_budget=False`：跳过 `pim-tile-to-budget`。给**不是单个线性算子**
    的 kernel 用——该 pass 的模型是「一个 tt.dot 加最多两层 void tiling 循环
    加一层 K reduction 循环」，融合 flash attention（`flash_fwd_splitkv_kernel`
    有 4 个 tt.dot，QK^T 与 PV 两段串在一条 online-softmax 链上）根本不符合，
    跑它只会得到 `could not infer the linear tile shape`。这类 kernel 只取
    flops / mram_traffic_bytes 作交叉验证基线，本来就不需要 tile 选择结果。

    `wram_bytes`：覆盖 `pim_options()["pim_wram_bytes"]`。同样是给融合
    flash kernel 用的——它的 WRAM staging（实测 131584 B）超过真实硬件的
    65536 B 预算，而 `pim-explicit-dma` 在超预算时 `signalPassFailure()`
    直接失败（FlagTree commit 80757547f 把这里从 warning 改成了 error）。
    交叉验证要的是「融合路径的 flops / MRAM 搬运量」这个量级对照，不是
    「融合 kernel 能不能塞进一个 DPU 的 WRAM」——后者的答案已知是不能，
    这正是 GeneSim 把 attention 拆成 96 个分离算子的原因。放宽这一个探针的
    预算让对照量测得出来，实测 flops 与 mram_traffic_bytes 不受该值影响。
    """
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
        # `context` 是 Python 侧的动态属性，用于让 MLIRContext 与 module 同生命周期；
        # parse 出来的 module 没有它，后面 pass_manager 要用，必须显式挂上
        module.context = context
        manager = ir.pass_manager(context)
        # mram_bytes/dma_align 必须按关键字传，不能沿用旧的 5 参数位置调用——
        # 旧调用把 `False` 落进了 mram_bytes 形参位（等价于 mram_bytes=0），
        # add_tile_to_budget 会拿它去比较 tile footprint，0 会导致任何张量都
        # 判定超预算。
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
        # 用真实预算选 tile（写出 pim.tile-m/n/k + pim.tile-wram-bytes），
        # 而不是沿用 FlagGems 的 GPU autotune tile。full_m/n/k 未提供
        # （仍是 -1）时这几个参数不改变行为，pass 照常走 IR 结构推断。
        #
        # **必须排在 pim-explicit-dma 之前**：tile-to-budget 的职责是把超出
        # WRAM 预算的 tile 切小，而 explicit-dma 负责按最终 tile 建 WRAM
        # staging buffer 并在超预算时 `signalPassFailure()`。反过来排的话，
        # explicit-dma 先按未切分的大 tile 建 buffer 就直接失败了，tile 切分
        # 根本没机会跑。FlagTree 自己的 lit 测试（tile_to_budget_m_split.mlir、
        # tile_to_budget_small_wram.mlir）用的都是 `-pim-tile-to-budget
        # -pim-explicit-dma` 这个顺序。
        #
        # 只对含 tt.dot 的算子（GEMM/GEMV_SCORE/GEMV_CONTEXT）跑这个 pass——
        # pim-tile-to-budget 硬性要求至少一个 tt.dot（"requires at least one
        # tt.dot"），纯逐元素算子（SOFTMAX/GELU）没有矩阵乘、也没有 tile 概
        # 念可选，跑这个 pass 只会报错。tile_m/n/k 等字段在 ir_cost.py 里本
        # 就是 Optional，这类算子保持 None 是预期行为，不是遗漏。
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
    """在 with 块内捕获所有 FlagGems kernel launch。

    产出列表在退出 with 后仍可读。包装 LibEntry.run 而非 JITFunction.run：
    FlagGems 命中自身 kernel_cache 后直接 `kernel[grid](...)` 发射，
    不再经过 JITFunction.run，那一层挂钩子会漏掉所有热路径 launch。

    emit_pimir=True 时每个捕获到的 kernel 额外降一份 pim mlir。

    `tile_to_budget` / `wram_bytes` 原样转给 `lower_ttir_to_pimir`，供融合
    flash attention 这类不是单个线性算子的探针放宽约束用（见那里的说明）。
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
        pimir = None
        if emit_pimir:
            # FlagGems 的 linear_kernel（以及大多数真实 autotuned kernel）
            # 把 M/N/K 当运行期标量参数按这三个名字传入——不是 IR 里能看到
            # 的编译期常量，也不是靠 scf.for 循环切分的（M/N 靠 2-D launch
            # grid 切），所以 add_tile_to_budget 单靠 IR 结构推断不出来
            # （见 lower_ttir_to_pimir 的说明）。这里直接从已经拿到的
            # arg_values 里按名字取真实值传下去；命名不是这三个的算子
            # （attention 类）拿不到，仍然回退到 IR 结构推断（-1）。
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
    """跑 fn 两遍，返回第二遍捕获的 kernel。

    第一遍让 autotune 定下最优 config（此间会 bench 多个候选配置，捕获到的
    是噪声）；第二遍走 FlagGems 缓存，只发射选中的那个 kernel。第一遍不开
    emit_pimir——那些候选配置的 pim mlir 是纯浪费。

    `tile_to_budget` / `wram_bytes` 原样转给 `lower_ttir_to_pimir`。
    """
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
