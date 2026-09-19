"""`net.ini` 的 `[layers]` 执行序。

GML 是纯 DAG，不带执行序（`prev_task` / `next_task` 在 GML schema 里存在但
对方 PDF 注明 "L2A will determine the execution flow"）。所以执行序在这里定。

顺序就是 `layer_expand.expand_layers` 的产出顺序——那来自 FX 图的拓扑序，与
文档步骤 C 的执行骨架一致：

    RMSNorm → DQ → v/k/RoPE_K → q/RoPE_Q/DQ_Q
    → for h in 0..nh-1: bmm1, mask, sm×5, DQ×4, bmm2
    → DQ → o_proj → residual → RMSNorm → DQ → gate, up, mul → DQ → down → residual

本模块只做序列化，不重排：重排会破坏 `force_consecutive`（RoPE 三连必须连续，
中间结果不落 DDR）。
"""

from __future__ import annotations

from orchestrator.l2_alloc import QMAN_OFFSET, QMAN_SIZE
from orchestrator.layer_id import LayerIdentity

# 全层恒定的硬件口，来自文档步骤 C 的「恒定硬件口（查表）」。
CONSTANTS: dict[str, object] = {
    # 手册 Table 7-11，NPM4K+
    "Bytes in cycle internal memory read": 64,
    "Bytes in cycle internal memory write": 64,
    "Number of frames": 1,
    "Input Maps": 1,
    "Output Maps": 1,
    "Input Height": 1,
    "Output Height": 1,
    # 见文档 Q31：本样例全层为 1。
    "skip compare": 1,
    "L2 qman buffer offset": QMAN_OFFSET,
    "L2 qman buffer size": QMAN_SIZE,
}


def render_layers_section(identities: list[LayerIdentity]) -> str:
    """`[layers]` 段：一行一个层参数文件名，按执行序。"""
    lines = ["[layers]"]
    for identity in identities:
        lines.append(identity.filename)
    return "\n".join(lines) + "\n"


def render_general_section(*, num_layers: int, gml_version: str) -> str:
    """`[general]` 段。

    `input_line_stride` 这类域的物理含义尚未获对方确认（文档 §12 问题 1 列为
    P0），所以这里只写已核实的项，不猜。
    """
    lines = ["[general]",
             f"layers_count={num_layers}",
             f"gml_version={gml_version}"]
    return "\n".join(lines) + "\n"


def render(identities: list[LayerIdentity], *, gml_version: str) -> str:
    """整份 `net.ini`。"""
    return (render_general_section(num_layers=len(identities),
                                  gml_version=gml_version)
            + "\n"
            + render_layers_section(identities))
