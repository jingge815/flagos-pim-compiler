"""Memory Layout 维度的通用排布规则。

1. **对齐**：`align_up` 的唯一实现，`memory/kv_layout.py` 复用它。
2. **步幅**：行主序步幅与步幅合法性校验，统一 IR 的分片校验与下发入口共用。

L1 / L2 / DDR 级的排布由厂商 GC/L2A 负责，本仓不再自研。
"""

from __future__ import annotations


def align_up(n: int, align: int) -> int:
    """向上对齐到 `align` 的整数倍。

    非正的对齐粒度直接报错。
    """
    if align <= 0:
        raise ValueError(f"align 必须为正，got {align}")
    return (n + align - 1) // align * align


def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """行主序紧密排列的逐维步幅（元素数）。

    这是 `bytes_of()` 原先隐含假设的显式形式：分片构造时按它产出
    `TensorShardDetail.elem_strides`，让「怎么摆」由字段承载而不是靠默认值。
    """
    if not shape:
        raise ValueError("形状的秩必须为正，标量没有排布可言")
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def check_elem_strides(shape: tuple[int, ...],
                       elem_strides: tuple[int, ...]) -> None:
    """步幅必须与形状同秩、全为正、且相邻行不重叠。

    统一 IR 的分片校验与下发入口共用这一份，免得两处宽严不一。
    """
    if not elem_strides:
        return
    if len(elem_strides) != len(shape):
        raise ValueError(
            f"elem_strides 秩 {len(elem_strides)} 与形状秩 {len(shape)} 不符")
    if any(stride <= 0 for stride in elem_strides):
        raise ValueError(f"elem_strides 必须全为正，got {elem_strides}")
    # 每个元素独占一段地址，长度为「最内层步幅」。某一维的步幅小于它内侧
    # 各维已占用的地址跨度，这一维上相邻的元素就会落到同一段地址里。
    extent = min(elem_strides)
    for axis in sorted(range(len(shape)), key=lambda d: elem_strides[d]):
        if shape[axis] == 1:
            continue  # 这一维只有一个元素，不占新地址
        if elem_strides[axis] < extent:
            raise ValueError(
                f"第 {axis} 维步幅 {elem_strides[axis]} 小于其内侧已占用的 "
                f"{extent}，相邻元素会重叠")
        extent += (shape[axis] - 1) * elem_strides[axis]
