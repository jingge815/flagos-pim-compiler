"""ir_cost 的搬运列要算进展开后相位链里 `phase_spec` 的 bytes。"""
from __future__ import annotations

from genesim_bridge.ir_cost import analyze_ir


def test_phase_spec_bytes_count_toward_mram_traffic() -> None:
    """计算类算子（softmax/lut/...）展开后每一相的 `bytes` 字段是真实搬运量，
    不能因为 mnemonic 不属于访存类就整列记 0。

    以前 `_PHASES_ATTR_RE` 只用来判断"这一行是不是已经展开过"，从不解析
    `#pim.phase_spec<...>` 里的 `bytes` 字段，结果 softmax/lut/pool/eltwise/
    normalize/rope/mask/dq 的 `mram_traffic_bytes` 全是 0。
    """
    text = """\
module {
  tt.func @softmax(%x: tensor<128x64xf16>) {
    %0 = pim.eltwise %x {phases = [#pim.phase_spec<index = 0, bytes = 2048, unit = vpu>]} : tensor<128x64xf16> -> tensor<128x64xf16>
    %1 = pim.eltwise %0 {phases = [#pim.phase_spec<index = 1, bytes = 1024, unit = cstl, reads = [0]>]} : tensor<128x64xf16> -> tensor<128x64xf16>
    tt.return
  }
}
"""
    cost = analyze_ir(text, "softmax", grid=(1,), arg_values={}, ir_level="pimir")
    assert cost.mram_traffic_bytes == 2048 + 1024, cost.mram_traffic_bytes


def test_inner_loop_bounds_defined_in_outer_body_are_folded() -> None:
    """分块后内层循环的边界常量定义在外层循环体里，也要折出循环次数。

    `-pim-tile-to-budget` 重建循环时把 `%c4096_i32_8` 这类常量放进外层循环体，
    以前只在最外层收常量，内层次数按 1 次计，GEMM 成本低估约千倍。
    """
    text = """\
module {
  tt.func @k() {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    scf.for %i = %c0_i32 to %c128_i32 step %c8_i32  : i32 {
      %c0_i32_1 = arith.constant 0 : i32
      %c4096_i32_2 = arith.constant 4096 : i32
      %c32_i32_3 = arith.constant 32 : i32
      scf.for %k = %c0_i32_1 to %c4096_i32_2 step %c32_i32_3  : i32 {
        %d = tt.dot %a, %b, %acc : tensor<8x32xf16> * tensor<32x512xf16> -> tensor<8x512xf32>
      }
    }
    tt.return
  }
}
"""
    cost = analyze_ir(text, "k", grid=(1,), arg_values={}, ir_level="ttir")
    # 16 趟外层 × 128 趟内层 × 每趟 2*8*32*512。
    assert cost.flops == 16 * 128 * 2 * 8 * 32 * 512, cost.flops
    assert not any("循环次数" in n for n in cost.notes), cost.notes
