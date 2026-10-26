"""A、B 两路 pass 顺序的唯一真源。

五类优化的内部顺序固定为 常量折叠 → CSE → 融合 → DCE → 分块：折叠产出的
常量是 CSE 的合并对象，融合留下的无用户算子由 DCE 清理，分块要按前面化简后的
形状计算。

两路插的位置不同。A 路吃的是多算子 TTIR，五类放在转 PIM 之后、显式 DMA 之前。
B 路吃的是单算子 IR，融合必须在相位展开之前——展开后主算子已成相位链，
`pim.matmul + pim.lut` 的模式不在了，再跑一遍融合会把相位链里的 `pim.lut`
折进它的生产者，相位编号出现空档，契约校验直接失败。折叠、CSE、DCE 作用在
展开后的相位链上，分块按形状切整算子矩阵乘，校验放最后。

`pim_sidecar.py` 不引本仓，按同一顺序内联，由测试钉住两边一致。
"""

from __future__ import annotations

import re

# 五类优化的固定顺序。
OPT_PASSES = (
    "-pim-constant-fold",
    "-pim-cse",
    "-pim-fuse-activation",
    "-pim-dce",
    "-pim-tile-to-budget",
)

# 能安全地跑在相位展开之后的那几类。融合会改写相位链，不在这里。
_POST_EXPAND = ("-pim-constant-fold", "-pim-cse", "-pim-dce",
                "-pim-tile-to-budget")


def a_path_passes(hardware, *, target: str) -> list[str]:
    """A 路：TTIR 转 PIM，五类优化，再显式化 DMA。

    `convert-triton-to-pim` 的预算参数来自硬件配置，分块 pass 读的就是
    这里写进模块的 WRAM 与 MRAM 预算。`target` 沿用原来的硬件参数口径。
    """
    convert = (
        f"-convert-triton-to-pim=target={target} "
        f"num-dpus={hardware.num_dpus} "
        f"num-tasklets={hardware.num_tasklets} "
        f"wram-bytes={hardware.wram_bytes_per_dpu} "
        f"mram-bytes={hardware.mram_bytes_per_dpu} "
        f"dma-align={hardware.dma_align}"
    )
    # canonicalize 与 cse 是上游 pass，先清一遍 A 路 TTIR 里的 arith：
    # 折循环边界的整数运算，合并循环内重复的算术。pim 的折叠只认 pim 与整数
    # arith，CSE 与 DCE 按纯性处理所有方言。
    # 循环不变量外提放在分块之前：真实 kernel 的地址与布局转换不随内层
    # 迭代变化，外提后分块看到的循环体更小。分块会重建循环，所以必须在它前面。
    return [convert, "-canonicalize", "-cse",
            "-loop-invariant-code-motion",
            *OPT_PASSES, "-pim-explicit-dma"]


def b_path_passes() -> list[str]:
    """B 路：先融合再展开，折叠、CSE、DCE、分块作用于展开后的 IR，最后校验。

    融合只跑一次且必须在展开之前：展开后的相位链里也有 `pim.lut`，再融合一次
    会拆掉一条相位，`-pim-verify-gml-contract` 因相位编号空档而失败。

    分块的对象是整算子 `pim.matmul`：按形状选 tile 写在算子上，降到 C 时
    按它生成分块循环。模块要带预算属性，见 `with_budget`。
    """
    return [
        "-pim-fuse-activation",
        "-pim-expand-phases",
        *_POST_EXPAND,
        "-pim-verify-gml-contract",
    ]


def with_budget(mlir_text: str, hardware) -> str:
    """给 B 路模块头补上分块要读的 WRAM、MRAM 预算与 DMA 对齐，已有的不动。"""
    head = re.search(r"module attributes \{([^}]*)\}", mlir_text)
    if head is None:
        bare = re.search(r"module \{", mlir_text)
        if bare is None:
            raise ValueError(f"整算子 IR 里找不到模块头：\n{mlir_text[:200]}")
        # 没有属性字典时补一个空的，再按下面统一追加。
        mlir_text = (mlir_text[:bare.start()] + "module attributes {} {"
                     + mlir_text[bare.end():])
        head = re.search(r"module attributes \{([^}]*)\}", mlir_text)
    present = head.group(1)
    extra = [
        f'"{name}" = {value} : i64'
        for name, value in (
            ("pim.wram-bytes", hardware.wram_bytes_per_dpu),
            ("pim.mram-bytes", hardware.mram_bytes_per_dpu),
            ("pim.dma-align", hardware.dma_align),
        )
        if f'"{name}"' not in present
    ]
    if not extra:
        return mlir_text
    joined = ", ".join([p for p in (present.strip(),) if p] + extra)
    return mlir_text[:head.start(1)] + joined + mlir_text[head.end(1):]
