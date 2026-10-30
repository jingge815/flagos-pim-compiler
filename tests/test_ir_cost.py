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


def test_per_block_phases_multiply_by_the_block_count() -> None:
    """`pim.paged_attn` 的相 0-4 每个 KV 块跑一遍，块数取块表首维。

    以前相位字节只按 `bytes` 求一次和，KV 槽池那 2×4×4096 B 一点没进账，
    整算子只剩工作区的 840 B。
    """
    text = """\
module {
  tt.func @kernel(%q: tensor<1x128xf16>, %cache: !pim.memdesc<8192xf16, #pim.mram>, %bt: tensor<4xi32>) {
    %0 = pim.paged_attn %q, %cache, %bt {phases = [#pim.phase_spec<index = 0, bytes = 32, unit = nmu, perBlock = true, poolBytes = 4096>, #pim.phase_spec<index = 1, bytes = 4, unit = vpu, reads = [0], perBlock = true>, #pim.phase_spec<index = 2, bytes = 32, unit = activation, reads = [0, 1], perBlock = true>, #pim.phase_spec<index = 3, bytes = 4, unit = vpu, reads = [2], perBlock = true>, #pim.phase_spec<index = 4, bytes = 512, unit = nmu, reads = [2], perBlock = true, poolBytes = 4096>, #pim.phase_spec<index = 5, bytes = 256, unit = combiner, reads = [3, 4]>], slotElems = 2048 : i64, slotTokens = 16 : i64} : tensor<1x128xf16>, !pim.memdesc<8192xf16, #pim.mram>, tensor<4xi32> -> tensor<1x128xf16>
    tt.return
  }
}
"""
    cost = analyze_ir(text, "kernel", grid=(1,), arg_values={}, ir_level="oplevel")
    # 工作区相 0-4 = 32+4+32+4+512 每块一次，相 5 = 256 只一次；
    # KV 流 = (4096 K + 4096 V) × 4 块。
    assert cost.mram_traffic_bytes == (32 + 4 + 32 + 4 + 512) * 4 + 256 + 2 * 4096 * 4
