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

from orchestrator.layer_hw_table import NET_INI_GENERAL
from orchestrator.layer_id import LayerIdentity

# 行尾符照参考产物：整份 net.ini 用 CRLF，**只有两条 dumps 路径用 LF**。
# 文件末行没有换行（评审 4 §4：末字节是层名最后一个字符，不是 LF）。
CRLF = "\r\n"
LF = "\n"


def render_layers_section(identities: list[LayerIdentity],
                          *, stems: list[str] | None = None) -> str:
    """`[layers]` 段：一行 `layer = <stem>`，stem 不含 `.txt`。

    最后一行没有换行（参考实测 EOF 无 LF）。
    """
    names = stems if stems is not None else [
        identity.filename.removesuffix(".txt") for identity in identities]
    text = "[layers]" + CRLF
    for index, name in enumerate(names):
        last = index == len(names) - 1
        if last:
            text += f"layer = {name}"
        else:
            text += f"layer = {name}" + CRLF
    return text


def render_general_section(
    *,
    dumps_bin_path: str = "llama2_w4a8_decode_block_0/parser_output",
    dumps_txt_path: str = "llama2_w4a8_decode_block_0/prepare_out/txt_files",
) -> str:
    """`[general]` 段。四个 stride 物理含义见域确认表 Q10，值全网恒定。"""
    g = NET_INI_GENERAL
    crlf_lines = [
        "[general]",
        f"is_seq_test = {g['is_seq_test']}",
        f"seq_tunneling = {g['seq_tunneling']}",
        f"test_update_buffer = {g['test_update_buffer']}",
        f"input_line_stride = {g['input_line_stride']}",
        f"input_map_stride = {g['input_map_stride']}",
        f"output_line_stride = {g['output_line_stride']}",
        f"output_map_stride = {g['output_map_stride']}",
        "",
    ]
    text = CRLF.join(crlf_lines) + CRLF
    # 这两行参考是 LF。
    text += f"dumps_bin_path = {dumps_bin_path}" + LF
    text += f"dumps_txt_path = {dumps_txt_path}" + LF
    text += CRLF
    text += f"seq_output_bin_file = {g['seq_output_bin_file']}" + CRLF
    return text


def render(identities: list[LayerIdentity], *, gml_version: str = "",
           stems: list[str] | None = None,
           dumps_bin_path: str = "llama2_w4a8_decode_block_0/parser_output",
           dumps_txt_path: str = (
               "llama2_w4a8_decode_block_0/prepare_out/txt_files")) -> str:
    """整份 `net.ini`。`gml_version` 不进 net.ini（版本走 gml_version.txt）。"""
    return (render_general_section(dumps_bin_path=dumps_bin_path,
                                  dumps_txt_path=dumps_txt_path)
            + render_layers_section(identities, stems=stems))
