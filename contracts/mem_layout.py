"""Memory Layout 维度的排布规则。

本模块承接两类内容：

1. **对齐**：`align_up` 的唯一实现。原先 `memory/kv_layout.py` 与
   `orchestrator/l2_alloc.py` 各有一份，算法相同、校验不同，这里取严不取宽。
2. **步幅规则**（见下）：L2/DDR 级的排布与尺寸规则，原先只活在编排器支线里。

内存层次的分管边界：WRAM / MRAM 级归图编译器与算子编译器（经 PIMMLIR），
L1 / L2 / DDR 级归编排器（**不经 PIMMLIR**）。本文件里的规则属后者。
"""

from __future__ import annotations

from contracts.compile_slots import DEFAULT_SLOTS

L2_ALIGN = 16          # 对齐粒度。L2 output size 按 align16(Width)+16 算。
L2_OUTPUT_PAD = 16

# `net.ini [general]` 的四个步幅，全网恒定（域含义见
# `docs/prepare_out-域确认表-20260918.md` Q10）。它们是排布规则而非文件格式，
# 所以真源在这里，编排器只负责按 net.ini 的格式渲染出来。
NET_INI_STRIDES = {
    "input_line_stride": 8,
    "input_map_stride": 4,
    "output_line_stride": 12,
    "output_map_stride": 5,
}

# 编译期槽位。bmm2 的输出段按 hidden 占位，不是导出时的 seq_len。
H = DEFAULT_SLOTS.hidden


def align_up(n: int, align: int) -> int:
    """向上对齐到 `align` 的整数倍。

    保留 `memory/kv_layout.py` 那份的参数校验 —— `orchestrator/l2_alloc.py`
    那份没有校验，合并时取严不取宽。
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


def align16(width: int) -> int:
    return align_up(width, L2_ALIGN)


def l2_output_bytes(width: int, elem_bytes: int) -> int:
    """输出段字节数（闭合公式，文档步骤 C）。

        L2 output size = (align16(Width) + 16) * elem_bytes
    """
    return (align_up(width, L2_ALIGN) + L2_OUTPUT_PAD) * elem_bytes


def stride_z(width: int, *, final: bool, scalar_align16: bool = False) -> int:
    """终相 align16(W)+15；中间相 =W；Width=1 的中间相有时 16。

    照搬 `orchestrator/layer_fields.py` 的原实现，逻辑一字不改。
    `scalar_align16` 只在 dq_p2 且 out_w==1 时为真。
    """
    if final:
        return align16(width) + 15
    if scalar_align16 and width == 1:
        return 16
    return width


def _elem_bytes(dt: int) -> int:
    return {0: 1, 1: 2, 3: 4}[dt]


def l2_in_size(kind: str, in_w: int, in_dt: int) -> int:
    """L2 输入段字节数。

    参考实测（422 层）：
      int8 平面        = Width          （q_proj 4096、mask 2048 是 fp16 的 W×2）
      bmm              = Width + 16     （hd→144、S→1040）
      DQ p4 终相       = align16(W)+16 再 ×2  （4096→8224、1024→2080）
    """
    if kind in ("bmm1", "bmm2"):
        return in_w + 16
    if kind == "dq_p4":
        return (align16(in_w) + 16) * 2
    if kind == "sm_p2":
        # 实测 W=1024 → 2080 = (align16(W)+16)×2：exp 相要多留一行。
        return (align16(in_w) + 16) * 2
    return in_w * _elem_bytes(in_dt)


def l2_dual_in_size(kind: str, width: int) -> int:
    """双输入层某一路的 L2 输入段字节数。

    参考实测：两路都是 fp16 平面，`Width × 2`
    （H=4096→8192、hd=128→256、I=11008→22016、S=1024→2048）。
    """
    return width * 2


def l2_out_size(kind: str, out_w: int, out_dt: int) -> int:
    """L2 输出段字节数。

    参考实测（422 层）：

      bmm2      按 hidden 占位 8224
      DQ p2     Gn=32→96、Gn=1→32、Gn=86→208，即 `(align16(Gn)+16)×2`，
                Gn=1 例外取 32（16×2）
      DQ p4     终相 int8 平面 `align16(W)+16`：4096→4112、1024→1040、
                11008→11024（**不乘 2**，它已是 int8）
    """
    if kind == "bmm2":
        return (align16(H) + 16) * 2
    if kind == "dq_p2":
        # 实测 Gn=32→96、86→208，即 `align16((Gn+16)×2)`；Gn=1 例外取 32
        # （按公式会得 48，参考是 32 —— 单组时不留那一行余量）。
        return 32 if out_w <= 1 else align16((out_w + 16) * 2)
    if kind == "dq_p4":
        return align16(out_w) + 16
    return l2_output_bytes(out_w, _elem_bytes(out_dt))
