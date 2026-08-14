"""genesim_bridge 单测。

TTIR 分析器的判据是理论值：bmm 的 flops 必须等于 2*M*N*K。
拟合器的判据是非负性——GeneSim 的 scheduler 直接消费求值结果，
负的 flops / data_bytes 会让 roofline 与分区判据失效。

不需要 GPU：TTIR 用 tests/data 下的固化样本，拟合用构造的 Measurement。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.cost_extractor import Measurement, _fit_coeffs, _net_data_bytes
from genesim_bridge.op_classify import ShapePoint, gemm_features
from genesim_bridge.paths import genesim_models_dir
from genesim_bridge.ttir_cost import analyze_ttir

DATA = Path(__file__).parent / "data"

# 一个最小的 bmm TTIR 骨架：tile 32x32x32，循环次数 ceil(K/32)
BMM_TTIR = """
module {
  tt.func public @bmm_kernel(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %O: !tt.ptr<f16>, %K: i32) {
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %n0 = arith.addi %K, %c31_i32 : i32
    %n1 = arith.divsi %n0, %c32_i32 : i32
    %o:1 = scf.for %i = %c0_i32 to %n1 step %c1_i32 iter_args(%acc = %cst) -> (tensor<32x32xf32>) : i32 {
      %a = tt.load %ap : tensor<32x32x!tt.ptr<f16>>
      %b = tt.load %bp : tensor<32x32x!tt.ptr<f16>>
      %d = tt.dot %a, %b, %acc : tensor<32x32xf16> * tensor<32x32xf16> -> tensor<32x32xf32>
      scf.yield %d : tensor<32x32xf32>
    }
    tt.store %op, %o#0 : tensor<32x32x!tt.ptr<f16>>
    tt.return
  }
}
"""


def test_bmm_flops_matches_theory():
    """M=128, N=128, K=64 -> grid (4,4,1), flops 应等于 2*M*N*K。"""
    cost = analyze_ttir(BMM_TTIR, "bmm_kernel", (4, 4, 1), {"K": 64})
    assert cost.loop_trip_counts == [2]       # ceil(64/32)
    assert cost.tile_flops_per_program == 2 * (2 * 32 * 32 * 32)
    assert cost.flops == 2 * 128 * 128 * 64
    assert cost.dtype == "f16"
    assert cost.element_bytes == 2
    assert not cost.notes


# 两个顺序排列的顶层循环，次数不同（flash_fwd 的结构：masking 段 + 非 masking 段）
TWO_LOOP_TTIR = """
module {
  tt.func public @two_loop(%A: !tt.ptr<f16>, %P: i32, %Q: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %l1:1 = scf.for %i = %c0_i32 to %P step %c1_i32 iter_args(%a = %cst) -> (tensor<8x8xf32>) : i32 {
      %d = tt.dot %x, %y, %a : tensor<8x8xf16> * tensor<8x8xf16> -> tensor<8x8xf32>
      scf.yield %d : tensor<8x8xf32>
    }
    %l2:1 = scf.for %j = %c0_i32 to %Q step %c1_i32 iter_args(%b = %l1#0) -> (tensor<8x8xf32>) : i32 {
      %e = tt.dot %x, %y, %b : tensor<8x8xf16> * tensor<8x8xf16> -> tensor<8x8xf32>
      scf.yield %e : tensor<8x8xf32>
    }
    tt.return
  }
}
"""


def test_sequential_top_level_loops_keep_own_trip_counts():
    """两个顺序顶层循环各用自己的次数，不能共用一个 trip。

    早期实现只保留一个 trip，后一个循环的次数会覆盖前一个并被同时应用到
    两个循环体上（P=3,Q=5 会算成 (1+1)*5=10 个 dot 而不是 3+5=8 个）。
    """
    dot = 2 * 8 * 8 * 8
    cost = analyze_ttir(TWO_LOOP_TTIR, "two_loop", (1,), {"P": 3, "Q": 5})
    assert cost.loop_trip_counts == [3, 5]
    assert cost.tile_flops_per_program == (3 + 5) * dot
    assert not cost.notes


def test_dot_with_attributes_is_counted():
    """带属性的 tt.dot 必须照样计入。

    flash_fwd 的 dot 形如 `tt.dot %a, %b, %c, inputPrecision = tf32 : ...`。
    早期正则要求「三个操作数后紧跟冒号」，遇到属性直接匹配失败，把整个
    flash kernel 的矩阵乘静默算成 0 flops，交叉验证结果因此偏低 37 倍。
    """
    line = ("      %S = tt.dot %Q, %K, %cst, inputPrecision = tf32 : "
            "tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>")
    ttir = "module {\n  tt.func public @k(%A: !tt.ptr<f16>) {\n" + line + "\n    tt.return\n  }\n}\n"
    cost = analyze_ttir(ttir, "k", (1,), {})
    assert cost.flops == 2 * 128 * 64 * 128


def test_unresolved_loop_is_flagged_not_silently_zero():
    """循环次数折不出来时必须留 note，绝不静默当 0 flops。"""
    cost = analyze_ttir(BMM_TTIR, "bmm_kernel", (1, 1, 1), {})   # 不给 K
    assert cost.flops > 0
    assert any("循环次数" in n for n in cost.notes)


def _gemm_op():
    return {
        "op_id": 0,
        "op_type": "GEMM",
        "input_shapes": [["Tq", 512]],
        "output_shapes": [["Tq", 2048]],
        "flops_coeffs": {"Tq": 2097152},
        "data_bytes_coeffs": {"constant": 2097152, "Tq": 5120},
    }


def _score_op():
    return {
        "op_id": 1,
        "op_type": "GEMV_SCORE",
        "input_shapes": [["Tq", 64], ["Tp+Tq", 64]],
        "output_shapes": [["Tq", "Tp+Tq"]],
        "flops_coeffs": {"Tq(Tp+Tq)": 127},
        "data_bytes_coeffs": {"Tq": 128, "Tp+Tq": 128, "Tq(Tp+Tq)": 2},
    }


def test_gemm_net_bytes_includes_weight():
    """GEMM 的 input_shapes 不含权重，必须补上，否则 decode 访存量低估两个量级。"""
    op = _gemm_op()
    decode = ShapePoint(tq=1, tp=512)
    got = _net_data_bytes(op, decode, element_bytes=2)
    # 激活 (1*512 + 1*2048) + 权重 (512*2048)，按 fp16
    assert got == float((512 + 2048 + 512 * 2048) * 2)
    # 权重项占绝对主导
    assert got > 100 * float((512 + 2048) * 2)


def test_gemm_features_from_symbolic_shapes():
    assert gemm_features(_gemm_op()) == (512, 2048)


def _m(point, flops, data_bytes=1.0):
    return Measurement(point=point, flops=flops, data_bytes=data_bytes,
                       dtype="f16", element_bytes=2)


def test_fit_two_point_linear():
    """成本单调随 term 增长时走标准两点解。"""
    prefill, decode = ShapePoint(128, 0), ShapePoint(1, 128)
    f, b, mode = _fit_coeffs(
        _gemm_op(),
        _m(prefill, 201_523_200.0, 100.0),
        _m(decode, 50_380_800.0, 10.0),
        seq_len=128,
    )
    assert mode["flops"] == "two_point_linear"
    assert f["Tq"] > 0
    assert all(v >= 0 for v in f.values())
    assert all(v >= 0 for v in b.values())


def test_fit_rejects_negative_slope_from_padding():
    """padding 主导的小 term 点会导致负斜率，必须降级为过原点解。

    实测 GEMV_SCORE：decode 点 term 更小(129 vs 16384) 但实测 flops 更大
    (2.62e6 vs 2.10e6)，因为 bmm 对 1 行输入 padding 到 32 行 tile。
    无约束两点解会给出负斜率，在 Tq=512 的 prefill 上算出负 flops。
    """
    prefill, decode = ShapePoint(128, 0), ShapePoint(1, 128)
    f, b, mode = _fit_coeffs(
        _score_op(),
        _m(prefill, 2_097_152.0, 100.0),
        _m(decode, 2_621_440.0, 10.0),
        seq_len=128,
    )
    assert mode["flops"] == "origin_through_large_term"
    assert f["Tq(Tp+Tq)"] > 0
    assert "constant" not in f          # 过原点，无常数项

    # 关键判据：在真实 trace 长度上求值必须非负
    for tq, tp in ((512, 0), (1024, 0), (1, 512), (1, 2048)):
        term = tq * (tp + tq)
        value = f["Tq(Tp+Tq)"] * term + f.get("constant", 0.0)
        assert value >= 0, f"Tq={tq},Tp={tp} 算出负 flops: {value}"

    # 过原点解应还原理论标度 2*head_dim = 128
    assert f["Tq(Tp+Tq)"] == pytest.approx(128.0, rel=0.02)


def test_refined_ir_preserves_structure():
    """精化只改成本，不动图结构；且保留 ModelIR.to_dict() 不含的字段。"""
    ir_dir = genesim_models_dir()
    base_path, ref_path = ir_dir / "gpt2_builtin.ir", ir_dir / "gpt2_builtin_flagtree.ir"
    if not (base_path.is_file() and ref_path.is_file()):
        pytest.skip("需要先跑 scripts/refine_ir_with_flagtree.py 生成产物")

    base, ref = json.loads(base_path.read_text()), json.loads(ref_path.read_text())

    assert ref["dependencies"] == base["dependencies"]
    assert ref["subgraphs"] == base["subgraphs"]
    # ModelIR.to_dict() 丢这两个字段，故走原始 JSON 改写
    assert ref["max_seq"] == base["max_seq"]
    assert ref["vocab_size"] == base["vocab_size"]

    assert len(ref["operators"]) == len(base["operators"])
    for new_op, old_op in zip(ref["operators"], base["operators"]):
        assert new_op["op_id"] == old_op["op_id"]
        assert new_op["op_type"] == old_op["op_type"]
        assert new_op["device_hint"] == old_op["device_hint"]
        assert new_op["input_shapes"] == old_op["input_shapes"]
        assert new_op["output_shapes"] == old_op["output_shapes"]

    terms = lambda tq, tp: {"constant": 1, "Tq": tq, "Tp": tp,
                            "Tp+Tq": tp + tq, "Tq(Tp+Tq)": tq * (tp + tq)}
    for tq, tp in ((512, 0), (2048, 0), (1, 512), (1, 2048)):
        table = terms(tq, tp)
        for op in ref["operators"]:
            for field in ("flops_coeffs", "data_bytes_coeffs"):
                value = sum(c * table[k] for k, c in op[field].items())
                assert value >= 0, f"op {op['op_id']} {field} 负值 @Tq={tq},Tp={tp}"
