"""把量化产物写成 GML 引用的 `.bin` 文件。

文件名一律经 `contracts.gml_names`，不在这里拼字符串——那是跨语言真源，
两侧靠它保持一致（见文档第 10 节）。

写盘用 `tofile`，不做任何字节序转换：实物是小端，本机也是小端，恒等 LUT 的
字节级比对（26.4）已经确认这条路径对了。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contracts import gml_names as names
from contracts.gml_quant import lut_identity
from quant.weights import QuantizedWeight


@dataclass
class WrittenFiles:
    """一次写盘的结果，供交叉校验用。

    `names_written` 是实际落盘的文件名集合。它要与 GML 里引用的名字集合完全相等——
    多一个是垃圾文件，少一个是悬空引用，两者都会让底层编译器读不下去。
    """

    directory: Path
    names_written: set[str] = field(default_factory=set)
    total_bytes: int = 0

    def _write(self, name: str, data: np.ndarray | bytes) -> None:
        path = self.directory / name
        if isinstance(data, bytes):
            path.write_bytes(data)
            size = len(data)
        else:
            data.tofile(path)
            size = data.nbytes
        self.names_written.add(name)
        self.total_bytes += size


def write_weight(
    files: WrittenFiles, node_id: int, quantized: QuantizedWeight
) -> None:
    """写一个节点的权重与它的 per-group scale。

    权重按**本节点**编号——它属于节点自己，不属于某条边（见规则 2）。
    """
    files._write(names.weight_buffer(node_id), quantized.values)
    files._write(names.weight_scale(node_id), quantized.scales)


def write_activation_scale(
    files: WrittenFiles, consumer_id: int, scale: float,
    slot: int | None = None,
) -> None:
    """写一条边上的激活 scale。

    按**消费者**编号：缓冲区代表边，编号取读它的那个节点。
    """
    files._write(
        names.scale(consumer_id, slot),
        np.array([scale], dtype=np.float16),
    )


def write_activation_zero_point(
    files: WrittenFiles, consumer_id: int, zero_point: int = 0,
    slot: int | None = None,
) -> None:
    """写零点。实物里恒为 0（对称量化），但字段必须在。"""
    files._write(
        names.zero_point(consumer_id, slot),
        np.array([zero_point], dtype=np.int32),
    )


def write_data_buffer(
    files: WrittenFiles, consumer_id: int, element_count: int,
    slot: int | None = None, dtype=np.int8,
) -> None:
    """写一条边上的数据缓冲区。

    按**消费者**编号——缓冲区代表边，编号取读它的那个节点（规则 2）。

    结构层写全零：这些是运行时才有内容的激活缓冲区，实物里它们带的是某次推理的
    真实数据。尺寸必须对，因为对方按形状读取；内容在结构验证阶段不参与。
    """
    files._write(
        names.data_buffer(consumer_id, slot),
        np.zeros(element_count, dtype=dtype),
    )


def write_identity_lut(files: WrittenFiles, node_id: int) -> None:
    """写一张恒等 LUT。

    在采样规则确认之前只能发恒等表——它是实物里的合法值（37 个），
    而且我们生成的与实物字节完全一致（见 26.4）。
    """
    files._write(names.activation_lut(node_id), lut_identity())


def write_scaling(
    files: WrittenFiles, node_id: int, scaling: float, post_shift: int = 0
) -> None:
    """写定标系数。

    `scaling` 是**算子自身的数学缩放**，不是量化因子——attention 的 `1/√d` 就写在
    这里（见第 19 节）。实物里它是标量。
    """
    files._write(
        names.fpsu_scale(node_id), np.array([scaling], dtype=np.float16))
    files._write(
        names.fpsu_post_shift(node_id), np.array([post_shift], dtype=np.int8))


def verify_against_graph(files: WrittenFiles, referenced: set[str]) -> None:
    """交叉校验：落盘的文件名与 GML 引用的名字必须完全相等。

    这是架构 C 的那道防线（第 10 节）。不相等就抛，因为悬空引用不会在我们这侧
    报错，要到对方的解析器才炸。
    """
    missing = referenced - files.names_written
    extra = files.names_written - referenced

    problems = []
    if missing:
        problems.append(f"GML 引用了但没写盘: {sorted(missing)[:5]}")
    if extra:
        problems.append(f"写了盘但 GML 没引用: {sorted(extra)[:5]}")
    if problems:
        raise ValueError("；".join(problems))
