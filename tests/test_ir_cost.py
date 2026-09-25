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
