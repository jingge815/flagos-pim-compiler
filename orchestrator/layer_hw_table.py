"""prepare_out 查表：全层恒定域 + 按层类型的 B7 值。

几何（Width 等）不在这里，从 GML 边算。这里只放 422 层同类相同、
对不上单一 `k×Width×elem_bytes` 的量。换 hidden 要重测。

每个条目可带 `pending_q`，对拍时进 PENDING 栏，不混进 MATCH。
依据：`docs/prepare_out-生成方案-20260919.md` §2.1 / §2.10。
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.l2_alloc import QMAN_OFFSET, QMAN_SIZE


@dataclass(frozen=True)
class HwRow:
    """一层类型在查表区的值。缺的键表示该类不写该域。"""

    fpsu_mode: int | None = None
    kantor_mode: int | None = 0
    l2_fpsu_size: int | None = None
    l2_weights_size: int | None = None
    l2_wscale_size: int | None = None
    data_scale_buf: int | None = None
    input_format: int = 0
    output_format: int | None = None
    fpsu_source: int = 3
    weights_source: int = 3
    l2_weights_per_engine: int = 4
    l2_fpsu_id: str = "f1"
    l2_weights_id: str | None = None
    l2_input_id: str = "4"
    l2_output_id: str | None = None
    flp: tuple[int, int, int] | None = None  # min, max, mantisa
    transpose_type: int | None = None
    pooling_data_type: int | None = 2
    use_fpsu: int = 1
    # 片上双缓冲基址。参考 422 层同类相同，换形状要重测。
    l2_weights_off0: int | None = None
    l2_weights_off1: int | None = None


# 全层恒定。生成时每层都写。
CONSTANTS: dict[str, object] = {
    "Number of frames": 1,
    "Input Maps": 1,
    "Input Height": 1,
    "Output Maps": 1,
    "Output Height": 1,
    "Is Winograd": "false",
    "Weight Compression Rate": 1.0,
    "Sparsity": 0.0,
    "Padding Left": 0,
    "Padding Right": 0,
    "Padding Top": 0,
    "Padding Bottom": 0,
    "Raster mode": 0,
    "Is Macro Tile": "False",
    "Macro Tile ID": 1,
    "Total number of MT": 1,
    "Use Clipping": 0,
    "LeakyReLU Negative Slope": 0,
    "Input Fraction bits": 0,
    "Output Fraction bits": 0,
    "After concat": "false",
    "Before concat": "false",
    "skip compare": 1,
    "Bytes in cycle internal memory read": 64,
    "Bytes in cycle internal memory write": 64,
    "L2 qman buffer offset": QMAN_OFFSET,
    "L2 qman buffer size": QMAN_SIZE,
    "Output data order": 0,
    "Pooling Pad Left": 0,
    "Pooling Pad Right": 0,
    "Pooling Pad Top": 0,
    "Pooling Pad Bottom": 0,
}

# net.ini [general]，四个 stride 物理含义见 Q10。
NET_INI_GENERAL: dict[str, object] = {
    "is_seq_test": 0,
    "seq_tunneling": 0,
    "test_update_buffer": 0,
    "input_line_stride": 8,
    "input_map_stride": 4,
    "output_line_stride": 12,
    "output_map_stride": 5,
    "seq_output_bin_file": "/net.bin",
}

# 按层类查。键是 `docs/prepare_out-生成方案-20260919.md` 的 23 类名。
TABLE: dict[str, HwRow] = {
    # 相位链的 fpsu 槽按相位号递增：p1->f1 ... p5->f5（参考 422 层无反例）。
    "dq_p1": HwRow(fpsu_mode=1, l2_fpsu_size=1024, output_format=6,
                   l2_fpsu_id="f1", l2_output_id=None),
    "dq_p2": HwRow(fpsu_mode=1, l2_fpsu_size=1024, input_format=6,
                   output_format=7, flp=(10, 17, 3),
                   l2_fpsu_id="f2", l2_output_id="5"),
    "dq_p3": HwRow(fpsu_mode=1, l2_fpsu_size=1024, input_format=6,
                   output_format=4, flp=(15, 15, 0), transpose_type=2,
                   l2_fpsu_id="f4", l2_output_id=None),
    "dq_p4": HwRow(fpsu_mode=1, kantor_mode=3, l2_fpsu_size=1024,
                   output_format=1, l2_fpsu_id="f5", l2_output_id="6"),
    "sm_p1": HwRow(fpsu_mode=1, l2_fpsu_size=None, output_format=4,
                   fpsu_source=1, l2_fpsu_id="f1", l2_output_id=None),
    "sm_p2": HwRow(fpsu_mode=1, l2_fpsu_size=512, flp=(9, 16, 3),
                   transpose_type=1, fpsu_source=1, l2_fpsu_id="f2"),
    "sm_p3": HwRow(fpsu_mode=1, l2_fpsu_size=None, fpsu_source=1,
                   l2_fpsu_id="f3"),
    "sm_p4": HwRow(fpsu_mode=2, l2_fpsu_size=512, output_format=4,
                   flp=(15, 15, 0), transpose_type=2, fpsu_source=1,
                   l2_fpsu_id="f4"),
    "sm_p5": HwRow(fpsu_mode=1, l2_fpsu_size=None, fpsu_source=1,
                   l2_fpsu_id="f5", l2_output_id="5"),
    "gemm_qko": HwRow(fpsu_mode=2, l2_fpsu_size=28672, l2_weights_size=32768,
                      l2_wscale_size=131072, data_scale_buf=32,
                      l2_weights_per_engine=2, l2_weights_id="w2",
                      l2_fpsu_id="f2", l2_output_id="6", l2_input_id="4",
                      l2_weights_off0=8256, l2_weights_off1=41024),
    "gemm_v": HwRow(fpsu_mode=2, kantor_mode=3, l2_fpsu_size=57344,
                    l2_weights_size=32768, l2_wscale_size=131072,
                    data_scale_buf=32, output_format=2,
                    l2_weights_per_engine=2, l2_weights_id="w2",
                    l2_fpsu_id="f2", l2_output_id="6",
                    l2_weights_off0=4160, l2_weights_off1=36928),
    "gemm_gate": HwRow(fpsu_mode=2, l2_fpsu_size=77824, l2_weights_size=16384,
                       l2_wscale_size=176128, data_scale_buf=16, flp=(10, 17, 3),
                       l2_weights_per_engine=2, l2_weights_id="w2",
                       l2_fpsu_id="f2", l2_output_id="6",
                       l2_weights_off0=22080, l2_weights_off1=38464),
    "gemm_up": HwRow(fpsu_mode=2, l2_fpsu_size=77312, l2_weights_size=16384,
                     l2_wscale_size=176128, data_scale_buf=16,
                     l2_weights_per_engine=2, l2_weights_id="w2",
                     l2_fpsu_id="f2", l2_output_id="6",
                     l2_weights_off0=22080, l2_weights_off1=38464),
    "gemm_down": HwRow(fpsu_mode=2, l2_fpsu_size=28672, l2_weights_size=30720,
                       l2_wscale_size=122880, data_scale_buf=30,
                       l2_weights_per_engine=2, l2_weights_id="w2",
                       l2_fpsu_id="f2", l2_output_id="6",
                       l2_weights_off0=8256, l2_weights_off1=41024),
    "bmm1": HwRow(fpsu_mode=2, l2_fpsu_size=7168, l2_weights_size=4096,
                  l2_wscale_size=2048, data_scale_buf=64, weights_source=0,
                  l2_weights_per_engine=2, l2_weights_id="6",
                  l2_fpsu_id="f3", l2_output_id="7",
                  l2_weights_off0=2112, l2_weights_off1=6208),
    "bmm2": HwRow(fpsu_mode=2, l2_fpsu_size=1024, l2_weights_size=32768,
                  l2_wscale_size=256, data_scale_buf=16, weights_source=0,
                  l2_weights_per_engine=2, l2_weights_id="6",
                  l2_fpsu_id="f3", l2_output_id="7",
                  l2_weights_off0=8256, l2_weights_off1=41024),
    "mask": HwRow(fpsu_mode=None, l2_fpsu_size=7168, output_format=1,
                  use_fpsu=0, pooling_data_type=None,
                  l2_fpsu_id="f2", l2_output_id="6"),
    "residual": HwRow(fpsu_mode=1, l2_fpsu_size=28672,
                      l2_fpsu_id="f2", l2_output_id="6"),
    "mlp_mul": HwRow(fpsu_mode=1, kantor_mode=5, l2_fpsu_size=154624,
                     output_format=1, l2_fpsu_id="f2", l2_output_id="6"),
    "rope_mul_cos": HwRow(fpsu_mode=1, kantor_mode=5, l2_fpsu_size=57344,
                          l2_fpsu_id="f2", l2_output_id="6"),
    "rope_mul_sin": HwRow(fpsu_mode=1, kantor_mode=5, l2_fpsu_size=57344,
                          output_format=5, l2_fpsu_id="f2", l2_output_id="6"),
    # K 路的 add 写 int8 cache（Kantor 重定标，L2 fpsu 57344）；
    # Q 路的 add 出 fp16 给后面的 DQ（L2 fpsu 28672）。实测各 1 个。
    "rope_add_k": HwRow(fpsu_mode=1, kantor_mode=3, l2_fpsu_size=57344,
                        output_format=3, l2_fpsu_id="f2", l2_output_id="6"),
    "rope_add_q": HwRow(fpsu_mode=1, kantor_mode=0, l2_fpsu_size=28672,
                        output_format=1, l2_fpsu_id="f2", l2_output_id="6"),
    "rmsnorm": HwRow(fpsu_mode=None, kantor_mode=None, l2_fpsu_size=512, use_fpsu=0,
                     pooling_data_type=None, l2_weights_size=4096,
                     l2_weights_per_engine=1, l2_weights_id="w1",
                     l2_output_id="5", l2_fpsu_id="f1"),
}


def lookup(kind: str) -> HwRow:
    if kind not in TABLE:
        raise KeyError(f"查表没有层类 {kind!r}")
    return TABLE[kind]
