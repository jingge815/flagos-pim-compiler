"""验证 GeneSim IR 成本分析、拟合和结果回填。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.cost_extractor import (
    Measurement,
    _fit_coeffs,
    _net_data_bytes,
    export_costs_to_genesim,
    load_local_shapes,
    validate_local_shapes_against_ir,
)
from genesim_bridge.op_classify import ShapePoint, gemm_features
from genesim_bridge.paths import genesim_models_dir
from genesim_bridge.ir_cost import analyze_ir

DATA = Path(__file__).parent / "data"

# 最小的 BMM TTIR 样例。
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
    """验证 BMM 浮点运算量与理论值一致。"""
    cost = analyze_ir(BMM_TTIR, "bmm_kernel", (4, 4, 1), {"K": 64})
    assert cost.loop_trip_counts == [2]       # ceil(64/32)
    assert cost.tile_flops_per_program == 2 * (2 * 32 * 32 * 32)
    assert cost.flops == 2 * 128 * 128 * 64
    assert cost.dtype == "f16"
    assert cost.element_bytes == 2
    assert not cost.notes


# 含两个不同迭代次数的顶层循环。
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
    """验证顺序顶层循环使用各自的迭代次数。"""
    dot = 2 * 8 * 8 * 8
    cost = analyze_ir(TWO_LOOP_TTIR, "two_loop", (1,), {"P": 3, "Q": 5})
    assert cost.loop_trip_counts == [3, 5]
    assert cost.tile_flops_per_program == (3 + 5) * dot
    assert not cost.notes


def test_dot_with_attributes_is_counted():
    """验证带属性的 tt.dot 计入浮点运算量。"""
    line = ("      %S = tt.dot %Q, %K, %cst, inputPrecision = tf32 : "
            "tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>")
    ttir = "module {\n  tt.func public @k(%A: !tt.ptr<f16>) {\n" + line + "\n    tt.return\n  }\n}\n"
    cost = analyze_ir(ttir, "k", (1,), {})
    assert cost.flops == 2 * 128 * 64 * 128


def test_unresolved_loop_is_flagged_not_silently_zero():
    """循环次数折不出来时必须留 note，绝不静默当 0 flops。"""
    cost = analyze_ir(BMM_TTIR, "bmm_kernel", (1, 1, 1), {})   # 不给 K
    assert cost.flops > 0
    assert any("循环次数" in n for n in cost.notes)


# BMM TTIR 对应的 PIM IR 样例。
_LAYOUT = "#pim.tasklet_tiled<{sizePerTasklet = [1, 1], taskletsPerDpu = [1, 16], order = [1, 0]}>"
BMM_PIMIR = """
module attributes {"pim.num-dpus" = 1 : i32, "pim.num-tasklets" = 16 : i32, pim.target = "pim:v1", "pim.wram-bytes" = 65536 : i32, "pim.wram-bytes-used" = 6144 : i32} {
  tt.func public @bmm_kernel(%A: !tt.ptr<f16>, %B: !tt.ptr<f16>, %O: !tt.ptr<f16>, %K: i32) {
    %c31_i32 = arith.constant 31 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %n0 = arith.addi %K, %c31_i32 : i32
    %n1 = arith.divsi %n0, %c32_i32 : i32
    %abuf = pim.wram_alloc : !pim.memdesc<32x32xf16, #pim.wram>
    %bbuf = pim.wram_alloc : !pim.memdesc<32x32xf16, #pim.wram>
    %o:1 = scf.for %i = %c0_i32 to %n1 step %c1_i32 iter_args(%acc = %cst) -> (tensor<32x32xf32, LAYOUT>) : i32 {
      pim.dma_load %ap -> %abuf {base_arg = 0 : i64, contiguous_dim = 1 : i64, elem_stride = 1 : i64} : tensor<32x32x!tt.ptr<f16>, LAYOUT> -> !pim.memdesc<32x32xf16, #pim.wram>
      pim.barrier
      %a = pim.wram_load %abuf : !pim.memdesc<32x32xf16, #pim.wram> -> tensor<32x32xf16, LAYOUT>
      pim.dma_load %bp -> %bbuf : tensor<32x32x!tt.ptr<f16>, LAYOUT> -> !pim.memdesc<32x32xf16, #pim.wram>
      pim.barrier
      %b = pim.wram_load %bbuf : !pim.memdesc<32x32xf16, #pim.wram> -> tensor<32x32xf16, LAYOUT>
      %d = tt.dot %a, %b, %acc : tensor<32x32xf16, LAYOUT> * tensor<32x32xf16, LAYOUT> -> tensor<32x32xf32, LAYOUT>
      scf.yield %d : tensor<32x32xf32, LAYOUT>
    }
    %obuf = pim.wram_alloc : !pim.memdesc<32x32xf32, #pim.wram>
    pim.wram_store %o#0, %obuf : tensor<32x32xf32, LAYOUT> -> !pim.memdesc<32x32xf32, #pim.wram>
    pim.barrier
    pim.dma_store %obuf -> %op : !pim.memdesc<32x32xf32, #pim.wram> -> tensor<32x32x!tt.ptr<f32>, LAYOUT>
    tt.return
  }
}
""".replace("LAYOUT", _LAYOUT)


def test_pimir_flops_identical_to_ttir():
    """验证 TTIR 和 PIM IR 的浮点运算量一致。"""
    ttir = analyze_ir(BMM_TTIR, "bmm_kernel", (4, 4, 1), {"K": 64})
    pimir = analyze_ir(BMM_PIMIR, "bmm_kernel", (4, 4, 1), {"K": 64}, ir_level="pimir")

    assert pimir.flops == ttir.flops == 2 * 128 * 128 * 64
    assert pimir.tile_flops_per_program == ttir.tile_flops_per_program
    assert pimir.loop_trip_counts == ttir.loop_trip_counts == [2]
    assert not pimir.notes


def test_pimir_mram_traffic_counts_tile_level_repeats():
    """验证 MRAM 搬运字节数包含循环和网格重复。"""
    cost = analyze_ir(BMM_PIMIR, "bmm_kernel", (4, 4, 1), {"K": 64}, ir_level="pimir")
    assert cost.mram_traffic_bytes == 16 * (2 * 2 * 2048 + 4096)
    assert cost.dma_ops == 3
    assert cost.dma_ops_with_layout == 1        # 只有第一个 dma_load 标了 elem_stride
    assert [b.bytes for b in cost.wram_buffers] == [2048, 2048, 4096]
    assert cost.wram_bytes_used == 6144
    assert cost.wram_bytes_budget == 65536


def test_ttir_level_leaves_pim_fields_unset():
    """验证 TTIR 不填充 PIM 专用成本字段。"""
    cost = analyze_ir(BMM_TTIR, "bmm_kernel", (4, 4, 1), {"K": 64})
    assert cost.mram_traffic_bytes is None
    assert cost.wram_bytes_used is None
    assert cost.wram_buffers == []
    assert cost.dma_ops == 0


def test_wram_over_budget_is_flagged():
    """验证 WRAM 超预算记录说明。"""
    over = BMM_PIMIR.replace('"pim.wram-bytes" = 65536', '"pim.wram-bytes" = 4096')
    cost = analyze_ir(over, "bmm_kernel", (1, 1, 1), {"K": 64}, ir_level="pimir")
    assert any("WRAM 超预算" in n for n in cost.notes)


def test_pimir_without_dma_is_rejected():
    """验证缺少 DMA 指令的 PIM IR 会被拒绝。"""
    with pytest.raises(AssertionError, match="pass 可能没生效"):
        analyze_ir(BMM_TTIR, "bmm_kernel", (1, 1, 1), {"K": 64}, ir_level="pimir")


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
    """验证 GEMM 成本包含输入、权重和输出的字节数。"""
    op = _gemm_op()
    decode = ShapePoint(tq=1, tp=512)
    got = _net_data_bytes(op, decode, element_bytes=2)
    # 按 fp16 统计激活和权重字节数。
    assert got == float((512 + 2048 + 512 * 2048) * 2)
    # 权重字节数占主要部分。
    assert got > 100 * float((512 + 2048) * 2)


def test_gemm_features_from_symbolic_shapes():
    assert gemm_features(_gemm_op()) == (512, 2048)


def test_gemm_net_bytes_uses_local_shard_widths():
    """给了本地分片宽度时，激活和权重都按分片算，而不是全局形状。"""
    op = _gemm_op()                      # 全局 512 -> 2048
    decode = ShapePoint(tq=1, tp=512)
    local = (512, 256)                   # tp8：输出维切成 1/8
    got = _net_data_bytes(op, decode, element_bytes=2, local_features=local)
    assert got == float((1 * 512 + 1 * 256 + 512 * 256) * 2)
    # 必须显著小于全局口径，否则说明分片没生效。
    assert got < _net_data_bytes(op, decode, element_bytes=2)


def test_local_shapes_absent_falls_back_to_global():
    """不传本地形状时结果与改造前完全一致。"""
    op = _gemm_op()
    decode = ShapePoint(tq=1, tp=512)
    assert (_net_data_bytes(op, decode, element_bytes=2, local_features=None)
            == _net_data_bytes(op, decode, element_bytes=2))


def test_load_local_shapes_reads_placement_sidecar(tmp_path):
    """从放置 sidecar 读出本地宽度，跳过缺字段的条目。"""
    path = tmp_path / "placement.json"
    path.write_text(json.dumps({
        "version": 2,
        "operators": {
            "0": {"dpu_id": 0, "local_in_features": 4096,
                  "local_out_features": 1536},
            "97": {"dpu_id": 0, "local_in_features": 512,
                   "local_out_features": 4096},
            # 旧版条目没有本地宽度，必须被跳过而不是报错。
            "98": {"dpu_id": 0},
        },
    }))
    assert load_local_shapes(path) == {0: (4096, 1536), 97: (512, 4096)}


def test_load_local_shapes_on_version1_sidecar_is_empty(tmp_path):
    """version 1 的 sidecar 完全没有这两个字段，返回空字典让调用方退回全局口径。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "version": 1,
        "operators": {"0": {"device_hint": "pim", "dpu_id": 0}},
    }))
    assert load_local_shapes(path) == {}


def _ir_with(operators):
    return {"operators": operators}


def test_validate_local_shapes_accepts_matching_gemm_ids():
    """op_id 都存在且都是 GEMM 时通过。"""
    ir = _ir_with([
        {"op_id": 0, "op_type": "GEMM"},
        {"op_id": 1, "op_type": "SOFTMAX"},
    ])
    validate_local_shapes_against_ir({0: (4096, 12288)}, ir)


def test_validate_local_shapes_rejects_unknown_op_id():
    """引用了当前 IR 里不存在的 op_id —— IR 重新生成后编号变了。"""
    ir = _ir_with([{"op_id": 0, "op_type": "GEMM"}])
    with pytest.raises(ValueError, match="不存在的 op_id"):
        validate_local_shapes_against_ir({0: (16, 16), 99: (16, 16)}, ir)


def test_validate_local_shapes_rejects_shifted_ids_landing_on_non_gemm():
    """算子数相同但编号错位，本地宽度落到了非 GEMM 上。

    这是最危险的一类：不校验就会把某个 GEMM 的本地宽度套到 SOFTMAX 上，
    成本悄悄算错而不报错。
    """
    ir = _ir_with([
        {"op_id": 0, "op_type": "GEMM"},
        {"op_id": 1, "op_type": "SOFTMAX"},
    ])
    with pytest.raises(ValueError, match="不是 GEMM"):
        validate_local_shapes_against_ir({1: (4096, 12288)}, ir)


def test_export_costs_validates_local_shapes_before_measuring(tmp_path):
    """`export_costs_to_genesim` 必须在开始编译前就把错位的 op_id 拦下。

    否则会先花时间编译一批 kernel，再把结果套到错误的算子上。
    """
    ir = {
        "hidden_size": 64, "head_dim": 16, "num_heads": 4, "num_layers": 1,
        "operators": [{"op_id": 0, "op_type": "SOFTMAX",
                       "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]]}],
        "dependencies": [], "subgraphs": [[0]],
    }
    ir_path = tmp_path / "in.ir"
    ir_path.write_text(json.dumps(ir))
    with pytest.raises(ValueError, match="不是 GEMM"):
        export_costs_to_genesim(
            ir_path=ir_path,
            out_ir_path=tmp_path / "out.ir",
            sidecar_path=tmp_path / "sc.json",
            seq_len=128,
            cross_validate=False,
            local_shapes={0: (64, 64)},
        )
    # 拦在编译之前：不应产出任何文件。
    assert not (tmp_path / "out.ir").exists()


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
    """验证负斜率拟合改用过原点解。"""
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

    # 预测成本为非负值。
    for tq, tp in ((512, 0), (1024, 0), (1, 512), (1, 2048)):
        term = tq * (tp + tq)
        value = f["Tq(Tp+Tq)"] * term + f.get("constant", 0.0)
        assert value >= 0, f"Tq={tq},Tp={tp} 算出负 flops: {value}"

    # 过原点拟合恢复理论斜率。
    assert f["Tq(Tp+Tq)"] == pytest.approx(128.0, rel=0.02)


@pytest.mark.parametrize("refined_name", ["llama2_7b_flagtree.ir", "llama2_7b_pimir.ir"])
def test_refined_ir_preserves_structure(refined_name):
    """验证成本回填不改变 IR 结构。"""
    ir_dir = genesim_models_dir(required=False)
    if ir_dir is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    base_path, ref_path = ir_dir / "llama2_7b.ir", ir_dir / refined_name
    if not (base_path.is_file() and ref_path.is_file()):
        pytest.skip("需要先跑 scripts/refine_ir_with_flagtree.py 生成产物")

    base, ref = json.loads(base_path.read_text()), json.loads(ref_path.read_text())

    assert ref["dependencies"] == base["dependencies"]
    assert ref["subgraphs"] == base["subgraphs"]
    # 仅比较两侧均存在的原始 JSON 字段。
    if "max_seq" in base:
        assert ref["max_seq"] == base["max_seq"]
    if "vocab_size" in base:
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


def test_pimir_sidecar_agrees_with_ttir_on_flops():
    """验证 PIM IR 与 TTIR 的运算量和附加成本字段。"""
    ir_dir = genesim_models_dir(required=False)
    if ir_dir is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    ttir_path = ir_dir / "llama2_7b_flagtree_extensions.json"
    pimir_path = ir_dir / "llama2_7b_pimir_extensions.json"
    if not (ttir_path.is_file() and pimir_path.is_file()):
        pytest.skip("需要先跑 scripts/refine_ir_with_flagtree.py 生成两条路的产物")

    ttir = json.loads(ttir_path.read_text())
    pimir = json.loads(pimir_path.read_text())
    assert ttir["ir_level"] == "ttir" and pimir["ir_level"] == "pimir"
    assert pimir["coverage"] == ttir["coverage"]
    assert pimir["pim_options"]["pim_target"]

    for op_id, entry in pimir["operators"].items():
        for point in ("prefill", "decode"):
            got = entry["measurements"][point]
            want = ttir["operators"][op_id]["measurements"][point]
            assert got["flops"] == want["flops"], f"op {op_id} {point} flops 两层不一致"
            assert got["data_bytes"] == want["data_bytes"]
            # PIM IR 的搬运字节数为正且不小于净读写字节数。
            assert got["mram_traffic_bytes"] > 0
            assert got["mram_amplification"] >= 1.0
            assert got["pim_kernels"], f"op {op_id} {point} 缺 pim_kernels"
            for kernel in got["pim_kernels"]:
                assert kernel["dma_ops"] > 0
                assert kernel["wram_bytes_used"] is not None

        # 仅 PIM IR 填充该字段。
        assert ttir["operators"][op_id]["mram_traffic_bytes"] is None
        assert set(entry["mram_traffic_bytes"]) == {"prefill", "decode"}
