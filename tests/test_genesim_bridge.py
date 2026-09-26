"""验证 GeneSim IR 成本分析、拟合和结果回填。"""

from __future__ import annotations

import json
import re
import sys
import tempfile
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
from genesim_bridge.op_classify import (
    MNEMONIC_OF,
    MNEMONICS,
    UNCOVERED_OP_TYPES,
    ShapePoint,
    build_recipes,
    gemm_features,
    oplevel_ir,
)
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


def test_every_op_type_is_either_measurable_or_explicitly_uncovered():
    """IR 里的每个算子类型都必须有归属：能测量，或明确列入模板成本。

    漏掉一种就会让 export_costs_to_genesim 抛 KeyError 挂在半路。GeneSim 侧
    更新 model_parser 后确实发生过：IR 从 3232 个算子扩到 3491 个，新增了
    RMSNORM/SILU/VECTOR_ADD/VECTOR_MUL 和 MODEL_INPUT/MODEL_OUTPUT，其中
    MODEL_INPUT 第一个撞上 `KeyError: 'MODEL_INPUT'`。

    这个测试对着真实 IR 检查，所以下次 model_parser 再引入新类型时会立刻失败，
    而不是等到跑精化脚本才发现。
    """
    ir_dir = genesim_models_dir(required=False)
    if ir_dir is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    ir_path = ir_dir / "llama2_7b.ir"
    if not ir_path.is_file():
        pytest.skip("需要先在 GeneSim 侧生成 llama2_7b.ir")

    ir = json.loads(ir_path.read_text())
    dims = {
        "hidden_size": ir["hidden_size"],
        "head_dim": ir["head_dim"],
        "num_heads": ir["num_heads"],
        "ffn_dim": 4 * ir["hidden_size"],
    }
    measurable = set(build_recipes(dims))
    present = {op["op_type"] for op in ir["operators"]}
    orphans = sorted(present - measurable - set(UNCOVERED_OP_TYPES))
    assert not orphans, (
        f"IR 里的算子类型 {orphans} 既没有 FlagGems 配方、也不在 "
        "UNCOVERED_OP_TYPES 里。export_costs_to_genesim 会对它们抛 KeyError。"
        "要么在 op_classify.build_recipes 里加配方，要么把它们列入 "
        "UNCOVERED_OP_TYPES 以保留 model_parser 的模板成本。"
    )


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


def _expand_oplevel(mlir_text: str) -> str:
    """把一段整算子级 PIM IR 展开成相位 SSA（走算子编译器）。"""
    from genesim_bridge.flagtree_driver import lower_oplevel_to_pimir

    return lower_oplevel_to_pimir(mlir_text)


_SOFTMAX_MLIR = '''module attributes {pim.target = "pim:v1"} {
  tt.func @sm(%s: tensor<1x1024xf16>) {
    %p = pim.softmax %s {axis = 1 : i64} : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.return
  }
}
'''

_DQ_MLIR = '''module attributes {pim.target = "pim:v1"} {
  tt.func @dq(%x: tensor<1x4096xf16>, %s: tensor<32xf16>) {
    %q = pim.quantize %x, %s {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128, spg = true, spgAxis = 3, spgGroupSize = 128>} : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    tt.return
  }
}
'''


def _pim_passes_available() -> bool:
    try:
        from genesim_bridge.env import assert_pim_passes_available

        assert_pim_passes_available()
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_oplevel_softmax_costs_more_than_zero():
    """展开后的 Softmax 链要对上非零成本。

    在此之前 `ir_cost` 只认 `tt.dot` 与 `arith.*`，整算子级 mnemonic 一个都不
    认识——成本恒为 0。仿真照常跑完、给出一个看起来合理的总耗时，只是那个数字
    是错的，而且不会报错。

    预期值可以手推：相 0/1/2/4 各扫 1024 个元素，相 3 扫 1 个（标量倒数），
    外加折进 exp 相的稳定化 `x - max` 也是 1024 个元素。每元素计一次运算。
    """
    cost = analyze_ir(_expand_oplevel(_SOFTMAX_MLIR), "sm", (1,), {},
                      ir_level="pimir")
    assert cost.flops == 5 * 1024 + 1, cost.flops


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_oplevel_dynamic_quant_costs_more_than_zero():
    """动态量化的四相：相 0 扫全部 4096 个元素，相 1/2 各扫 32 组，相 3 再扫
    4096 个。"""
    cost = analyze_ir(_expand_oplevel(_DQ_MLIR), "dq", (1,), {},
                      ir_level="pimir")
    assert cost.flops == 2 * 4096 + 2 * 32, cost.flops


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_oplevel_kernel_has_no_dma_and_that_is_fine():
    """B 路没有 DMA 可显式化，不该被 A 路那条「没有 dma 就是 pass 没生效」的
    断言判成坏产物。

    **「没有显式 DMA」不等于「不搬字节」**：展开后的计算类算子把每一相的搬运量
    写在 `#pim.phase_spec` 的 `bytes` 上，这里量到的就是那五个相位之和
    （max 2 + exp 2048 + sum 4 + reciprocal 2 + mul 2048）。把两者混为一谈
    会让整条计算类的搬运列恒为 0。
    """
    cost = analyze_ir(_expand_oplevel(_SOFTMAX_MLIR), "sm", (1,), {},
                      ir_level="pimir")
    assert cost.dma_ops == 0
    assert cost.mram_traffic_bytes == 2 + 2048 + 4 + 2 + 2048


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_gather_costs_movement_but_no_flops():
    """查表**不计 flops、计搬运**。

    记成 0 会让仿真以为这个节点是免费的——它不做算术，但确实在动字节，而且动的
    是嵌入表那 256 MiB 里的行。
    """
    text = '''module attributes {pim.target = "pim:v1"} {
  tt.func @gather(%t: tensor<100x8xf16>, %i: tensor<1x5xi32>) {
    %y = pim.gather %t, %i : tensor<100x8xf16>, tensor<1x5xi32> -> tensor<1x5x8xf16>
    tt.return
  }
}
'''
    cost = analyze_ir(text, "gather", (1,), {}, ir_level="pimir")
    assert cost.flops == 0.0, "查表不做算术"
    # 5 行 × 8 个元素 × 2 字节。
    assert cost.mram_traffic_bytes == 80.0


def _matmul_ir(stationarity: str, extra: str = "") -> str:
    return f'''module attributes {{pim.target = "pim:v1"}} {{
  tt.func @mm(%a: tensor<128x512xi8>, %b: tensor<512x256xi8>) {{
    %m = pim.matmul %a, %b {{datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, stationarity = #pim.stationarity<{stationarity}>{extra}}} : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    tt.return
  }}
}}
'''


def test_stationarity_changes_weight_side_traffic():
    """驻留形态必须改变搬运量——这是 prefill 与 decode 的分野。

    驻留侧装载一次，流式侧每趟重读。少了这个区分，两种形态在仿真里搬运量相同，
    而实际差一个数量级；仿真照常跑完、给出看起来合理的总耗时，只是数字是错的。

    `activation`（prefill 两侧都是激活）记 0 不是"忽略"：那条链本来就不存在权值
    驻留，硬记一笔会让 prefill 的搬运量凭空多出一个张量。
    """
    weight = analyze_ir(
        _matmul_ir("weight",
                   ", weightBinding = #pim.weight_binding<format = weight, "
                   "role = model_weight, elemBits = 4>"),
        "mm", (1,), {}, ir_level="pimir")
    kv = analyze_ir(
        _matmul_ir("kv",
                   ", weightBinding = #pim.weight_binding<format = weight, "
                   "role = activation_as_weight, elemBits = 8>, bIsActivation"),
        "mm", (1,), {}, ir_level="pimir")
    activation = analyze_ir(
        _matmul_ir("activation", ", bIsActivation"),
        "mm", (1,), {}, ir_level="pimir")

    # 512 × 256 个 int8 = 131072 字节，装载一次。
    assert weight.mram_traffic_bytes == 131072.0
    assert kv.mram_traffic_bytes == 131072.0
    assert activation.mram_traffic_bytes == 0.0

    # 运算量与驻留无关：同一个乘法，搬的次数不同而已。
    assert weight.flops == kv.flops == activation.flops == 2 * 128 * 512 * 256


# 组反量化累加：`K = 512`，每组 128 个元素，整数累加器停 4 次把这一组折进
# 浮点总数。乘加次数没变（`flops` 仍是 `2MNK`），变的只是累加顺序。
GROUP_DEQUANT_PIMIR = """
module attributes {"pim.mram-bytes" = 8589934592 : i32} {
  tt.func @w4a8_proj(%a: tensor<128x512xi8>, %w: tensor<512x256xi8>) {
    %0 = pim.matmul %a, %w {datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point, groupDequantAccum = true, groupSize = 128>, weightBinding = #pim.weight_binding<format = weight, role = model_weight, elemBits = 4>, stationarity = #pim.stationarity<weight>} : tensor<128x512xi8>, tensor<512x256xi8> -> tensor<128x256xi8>
    tt.return
  }
}
"""


def test_group_dequant_steps_counts_one_stop_per_group():
    """验证组反量化次数按 `K / groupSize` 计，且不改 flops。"""
    cost = analyze_ir(GROUP_DEQUANT_PIMIR, "w4a8_proj", (1, 1, 1), {},
                      ir_level="pimir")
    assert cost.group_dequant_steps == 512 // 128 == 4
    assert cost.flops == 2 * 128 * 512 * 256


def test_group_dequant_steps_is_zero_without_the_flag():
    """验证整段累加的矩阵乘不记组反量化次数。"""
    line = GROUP_DEQUANT_PIMIR.replace("groupDequantAccum = true, groupSize = 128",
                                       "groupDequantAccum = false")
    cost = analyze_ir(line, "w4a8_proj", (1, 1, 1), {}, ir_level="pimir")
    assert cost.group_dequant_steps == 0


def test_group_dequant_steps_reaches_the_sidecar():
    """验证组反量化次数写进 sidecar。"""
    from genesim_bridge.cost_extractor import _pim_kernel_dict

    cost = analyze_ir(GROUP_DEQUANT_PIMIR, "w4a8_proj", (1, 1, 1), {},
                      ir_level="pimir")
    assert _pim_kernel_dict(cost)["group_dequant_steps"] == 4


# 表 1.2.4 的 14 个 mnemonic 分两拨计费。前 8 个真的做算术，成本记在 flops 上；
# kv_cache 与 gather 在方案里归在"计算类"，但硬件上只是寻址加搬运，给它们记
# flops 会凭空多出一笔运算，所以与四个视图类一起只记搬运字节。
_ARITHMETIC_MNEMONICS = (
    "pim.normalize", "pim.matmul", "pim.softmax", "pim.mask", "pim.rope",
    "pim.lut", "pim.eltwise", "pim.dynamic_quant",
)
_MOVEMENT_MNEMONICS = (
    "pim.kv_cache", "pim.gather",
    "pim.transpose", "pim.reshape", "pim.split_heads", "pim.concat",
)

_OPLEVEL_DIMS = {"hidden_size": 512, "head_dim": 64, "num_heads": 8,
                 "ffn_dim": 2048}


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_every_mnemonic_has_a_representative_ir_and_a_nonzero_cost():
    """14 个 mnemonic 逐个量一遍，一个都不许是 0。

    少一个名字就是成本 0，而 0 在仿真里看起来完全正常——只是总耗时算小了，不会
    报错。所以每个 mnemonic 都要在 `_OPLEVEL_IR` 里有代表实现、展开后计得到成本，
    且两个形状点都不许出现"循环次数没折叠"这类低估提示。
    """
    assert len(MNEMONICS) == 14
    # 两拨清单合起来必须正好是 MNEMONICS：漏一个就没测，多一个就有名字对不上。
    assert set(_ARITHMETIC_MNEMONICS) | set(_MOVEMENT_MNEMONICS) == set(MNEMONICS)

    for point in (ShapePoint(tq=1, tp=128), ShapePoint(tq=128, tp=0)):
        for mnemonic in MNEMONICS:
            cost = analyze_ir(
                _expand_oplevel(oplevel_ir(_OPLEVEL_DIMS, mnemonic, point)),
                mnemonic.replace(".", "_"), (1,), {}, ir_level="pimir")
            if mnemonic in _ARITHMETIC_MNEMONICS:
                assert cost.flops > 0, f"{mnemonic} @{point.label} 没算到 flops"
            else:
                assert cost.mram_traffic_bytes > 0, (
                    f"{mnemonic} @{point.label} 既不算 flops 也不搬字节")
            assert cost.notes == [], f"{mnemonic} @{point.label}: {cost.notes}"


def test_unknown_mnemonic_is_rejected():
    """表里没有的 mnemonic 直接抛，不许静默返回。

    这是 B 路的入口。静默返回空串等于那个算子成本 0，而 0 不会报错。
    """
    with pytest.raises(KeyError, match="pim.not_a_mnemonic"):
        oplevel_ir(_OPLEVEL_DIMS, "pim.not_a_mnemonic", ShapePoint(tq=1, tp=128))


def test_unrecognized_operator_mnemonic_is_noted() -> None:
    """成本抽取遇到不认识的 `pim.*` 算子时要留一条 note，不能静默记 0。

    表里认得的 12 个已经接上，但方言侧改名或加算子不会让这里报错——成本会直接
    回到 0，而 0 不会报错。这正是这一轮要消掉的那类失真，只是范围从「整类不认」
    缩到了「除已知的以外不认」。
    """
    unknown = ("module { tt.func @k() { "
               "%0 = pim.zoneout %1 : tensor<4x8xf16> -> tensor<4x8xf16> } }")
    cost = analyze_ir(unknown, "k", (1,), {}, ir_level="oplevel")
    assert cost.flops == 0.0, "不认识的算子本来就量不到 flops"
    assert any("pim.zoneout" in note for note in cost.notes), (
        f"未识别的算子级 mnemonic 必须出现在 notes 里，实际 notes={cost.notes}")


def test_known_but_unbilled_operator_is_noted() -> None:
    """登记了却没有计费规则的算子也要留 note，不能静默记 0。

    评审九轮问题 3：`_KNOWN_PIM_OPS` 把 `pim.split` 这类名字登记成「已知」，
    于是既不计 flops、也不计搬运、也不进 notes——成本静默为 0。
    """
    text = ("module { tt.func @k() { "
            "%0 = pim.split %1 : tensor<4x8xf16> -> tensor<4x4xf16> } }")
    cost = analyze_ir(text, "k", (1,), {}, ir_level="oplevel")
    assert any("pim.split" in note for note in cost.notes), (
        f"没有计费规则的已知算子必须出现在 notes 里，实际 notes={cost.notes}")


def test_known_mnemonic_produces_no_such_note() -> None:
    """对照：认得的名字不该被当成「未识别」。"""
    known = ("module { tt.func @k() { "
             "%0 = pim.normalize %1 : tensor<4x8xf16> -> tensor<4x8xf16> } }")
    cost = analyze_ir(known, "k", (1,), {}, ir_level="oplevel")
    assert cost.flops > 0
    assert not any("未识别" in note for note in cost.notes), cost.notes


def test_convert_layout_is_known_zero_cost() -> None:
    """`pim.convert_layout` 是类型转换器插入的布局修正 op，零成本是它的语义。

    NoMemoryEffect、降级成零代码（LowerPIMToEmitC 只转发 buffer 视图），
    本来就不该计 flops 或搬运。它必须登记在 `_ZERO_COST_OPS` 里，否则
    完整性守卫会给每个带它的 kernel 刷一条「未识别」note（实测 pimir 全量
    4544 条），把真正的新算子淹在噪声里。
    """
    text = ("module { tt.func @k() { "
            "%0 = pim.convert_layout %1 : "
            "tensor<4x8xf16> -> tensor<4x8xf16> } }")
    cost = analyze_ir(text, "k", (1,), {}, ir_level="pimir")
    assert cost.flops == 0.0
    assert cost.mram_traffic_bytes == 0.0
    assert not any("未识别" in note for note in cost.notes), cost.notes


def test_a_path_without_a_dot_still_requires_dma() -> None:
    """A 路产物缺 `pim.dma_*` 必须报错——**不看它有没有 `tt.dot`**。

    逐元素 / 归约类的 FlagGems 内核本来就没有 `tt.dot`，用 `tt.dot` 当
    「这是 A 路」的代理，会让它们带着一个没跑起来的 `-pim-explicit-dma`
    静默通过（`dma_ops == 0`、搬运字节 0）。
    """
    no_dot = ("module { tt.func @k(%a: tensor<4x8x!tt.ptr<f16>>, "
              "%o: tensor<4x8x!tt.ptr<f16>>) { "
              "%0 = tt.load %a : tensor<4x8x!tt.ptr<f16>> "
              "tt.store %o, %0 : tensor<4x8x!tt.ptr<f16>> "
              "tt.return } }")
    with pytest.raises(AssertionError, match="pass 可能没生效"):
        analyze_ir(no_dot, "k", (1,), {}, ir_level="pimir")


def test_oplevel_product_needs_no_dma() -> None:
    """对照：B 路（整算子级）没有访存可显式化，不该被同一条判据卡住。"""
    oplevel = ("module { tt.func @k() { "
               "%0 = pim.normalize %1 : tensor<4x8xf16> -> tensor<4x8xf16> } }")
    cost = analyze_ir(oplevel, "k", (1,), {}, ir_level="oplevel")
    assert cost.flops > 0


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_oplevel_sidecar_records_mnemonics(tmp_path):
    """B 路的 `source_name` 记 mnemonic，且每个桥接到的算子成本非零。

    两路的 sidecar 会并排比较，`source_name` 是分辨一笔成本从哪条路量出来的唯一
    字段——B 路记成 `flag_gems.ops.softmax` 就会被当成 FlagGems 的实测值。桥接上
    但成本为 0 比不桥接更糟：coverage 看着是全的，数字却是空的。
    """
    ir_dir = genesim_models_dir(required=False)
    if ir_dir is None:
        pytest.skip("需要在 paths.json 配置 genesim_root")
    ir_path = ir_dir / "llama2_7b.ir"
    if not ir_path.is_file():
        pytest.skip("需要先在 GeneSim 侧生成 llama2_7b.ir")

    sidecar = export_costs_to_genesim(
        ir_path=ir_path,
        out_ir_path=tmp_path / "out.ir",
        sidecar_path=tmp_path / "sidecar.json",
        seq_len=128,
        cross_validate=False,
        ir_level="oplevel",
    )

    assert sidecar["ir_level"] == "oplevel"
    # B 路不跑 FlagGems，融合注意力基准无从测起，干脆不留这个字段。
    assert "cross_validation" not in sidecar
    assert sidecar["coverage"]["bridged"], "B 路一个算子都没桥接上"

    for op_id, entry in sidecar["operators"].items():
        assert entry["source_name"] in MNEMONICS, (
            f"op {op_id} 的 source_name={entry['source_name']} 不是 mnemonic")
        for point in ("prefill", "decode"):
            measured = entry["measurements"][point]
            assert measured["flops"] > 0 or measured["mram_traffic_bytes"] > 0, (
                f"op {op_id} @{point} 成本为 0")


# --- 两份 libtriton 同源自检（评审 20260923 的 P1-2）------------------------


def test_inprocess_libtriton_is_not_behind_triton_opt() -> None:
    """进程内 `libtriton` 与 `triton-opt` 必须认同一组 pim 属性。

    A 路成本抽取走进程内 `libtriton`，B 路走 `triton-opt` 可执行文件，两者是
    **分别构建**的。实测某一轮里 `triton-opt` 是 20:05 的产物而两份
    `libtriton.so` 都停在 16:01，后者对 `#pim.stationarity` 报
    `unknown attribute`。A 路当时不报错，只因为它用的 pass 撞不到新属性 ——
    那是巧合。哪天 A 路开始读新属性，进程内那份会解析失败或静默忽略，
    量出来的成本与 B 路对不上，而没有任何地方会说这件事。
    """
    from genesim_bridge.flagtree_driver import (
        _check_inprocess_matches_triton_opt)

    # 不抛就是同源。抛出来的信息里写了怎么重装。
    _check_inprocess_matches_triton_opt()


def test_the_probe_attribute_is_one_triton_opt_accepts() -> None:
    """探针 IR 必须是 `triton-opt` 认的**语义合法**的 IR。

    探的是「这份二进制认不认这个属性」。探针本身被 verifier 拒掉的话，
    「不认识这个属性」与「认识但这段 IR 写错了」两种情况分不开 —— 自检就会
    对一份其实没问题的绑定报警，而那种假警报很快会被人加个 try 绕掉。
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from genesim_bridge import flagtree_driver as driver
    from opcompiler_bridge.driver import _triton_opt

    binary = _triton_opt()
    if not binary.is_file():
        pytest.skip("没有 triton-opt，无法验证探针本身合法")

    # 与 `_check_inprocess_matches_triton_opt` 里那段完全一致的构造方式。
    assert driver._PROBE_ATTR == "stationarity"
    probe = (
        'module {\n'
        '  tt.func @probe(%a: tensor<4x8xf16>, %b: tensor<8x4xf16>) {\n'
        '    %0 = pim.matmul %a, %b {datapath = #pim.datapath<'
        'nmuMode = floating_point, scaleMode = floating_point>, '
        'weightBinding = #pim.weight_binding<format = weight, '
        'role = model_weight, elemBits = 4>, '
        f'{driver._PROBE_ATTR} = #pim.{driver._PROBE_ATTR}<weight>}}'
        ' : tensor<4x8xf16>, tensor<8x4xf16> -> tensor<4x4xf16>\n'
        '    tt.return\n  }\n}\n')
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(probe)
        path = handle.name
    try:
        proc = subprocess.run([str(binary), path], capture_output=True, text=True)
    finally:
        Path(path).unlink()
    assert proc.returncode == 0, (
        f"triton-opt 拒绝了探针 IR，说明探针自己写错了:\n{proc.stderr[:800]}")


def test_real_model_bridge_coverage_is_recorded() -> None:
    """整模型上跑 B 路，把**实际覆盖到的 mnemonic** 钉住。

    `test_every_mnemonic_has_a_representative_ir_and_a_nonzero_cost` 量的是
    合成代表形状——14 个 mnemonic 全过。但真实模型上出现几个是另一回事：
    genesim 的 `.ir` 骨架过去把 attention 抽象成 GEMV_SCORE / GEMV_CONTEXT，
    骨架里没有 ROPE / MASK / DQ / KV / 视图这些 op_type，那 9 个在整模型上
    一次都不会出现；现在骨架补齐了这些节点，14 个全都能在整模型上量到成本。

    这条把「实际覆盖到哪些」写成断言：哪天上游改了骨架或配方，数字会变，这里
    必须跟着更新——而不是让「14 个都能回归」这句话一直听起来像是整模型的结论。
    """
    model_ir = Path(genesim_models_dir()) / "llama2_7b.ir"
    if not model_ir.is_file():
        pytest.skip("没有 genesim 的 llama2_7b.ir，跳过整模型覆盖")

    sidecar = export_costs_to_genesim(
        ir_path=model_ir,
        out_ir_path=Path(tempfile.mkdtemp()) / "out.ir",
        sidecar_path=Path(tempfile.mkdtemp()) / "out.json",
        seq_len=128,
        cross_validate=False,
        ir_level="oplevel",
    )
    covered = sorted({e["source_name"] for e in sidecar["operators"].values()})
    # llama2.ir 的 op_type 覆盖表 1.2.4 的全部 14 个 mnemonic：计算类来自
    # GEMM/GEMV/ SOFTMAX/RMSNORM/SILU/VECTOR_*，视图与缓存类来自
    # ROPE/MASK/QUANT/MEM_COPY/GATHER 和 RESHAPE/TRANSPOSE/SPLIT/CONCAT。
    assert covered == sorted(MNEMONICS), (
        f"整模型覆盖的 mnemonic 变了：{covered}。是模型 IR 少了 op_type，"
        "还是桥接漏了？"
    )


def test_non_gemm_ops_get_a_pimir_path(tmp_path):
    """算子级节点也要写 `pimir_path`，否则相位链那一支在生产里没人喂。

    判据：给一份带 GEMM 与 ROPE 的小 IR，导出后 ROPE 那条也有 `pimir_path`，
    且指向的文件是 B 路产物（含 `pim.rope`、不含 `tt.dot`）。
    """
    from genesim_bridge.placement_export import _bpath_pimir

    path = _bpath_pimir("rope", [("Tq", 128)], [("Tq", 128)])
    text = Path(path).read_text()
    # rope 展开后是相位链（pim.eltwise 三相），原 op 名不再出现。
    # 判据是「有相位、且不是 A 路的 tt.dot」。
    assert "#pim.phase_spec" in text, "rope 展开后没有相位链"
    assert "tt.dot" not in text, "rope 的 pimir 混进了 A 路"

    # 视图类同样要有产物，而且不是相位链（它们展开后没有 phase_spec）。
    reshape = Path(_bpath_pimir("reshape", [("Tq", 4096)],
                                [("Tq", 32, 128)])).read_text()
    assert "pim.reshape" in reshape


def test_placement_uses_the_same_mnemonic_map_as_cost():
    """放置导出与成本抽取必须用同一张算子到 mnemonic 的映射。

    两张表各写一份时已经漂过：`MEM_COPY` 一边是 `kv_cache`、一边是 `reshape`，
    注意力的 `GEMV_SCORE` 与 `GEMV_CONTEXT` 只在成本侧有、放置侧没有，于是
    这两千个节点拿不到 B 路 pimir。映射只该有一份。
    """
    from genesim_bridge.placement_export import _BPATH_MNEMONIC

    # 注意力矩阵乘是 pim.matmul，KV 写入是 pim.kv_cache。
    assert MNEMONIC_OF["GEMV_SCORE"] == "matmul"
    assert MNEMONIC_OF["GEMV_CONTEXT"] == "matmul"
    assert MNEMONIC_OF["MEM_COPY"] == "kv_cache"

    # 放置侧的表必须是这份映射的子集，不能另起一份。
    for op_type, mnemonic in _BPATH_MNEMONIC.items():
        assert MNEMONIC_OF[op_type] == mnemonic, op_type


# --- ODS 未声明的裸属性 -------------------------------------------------
#
# MLIR 的 `attr-dict` 会把任何名字都静默收下：ODS 里没有这个字段，属性照样挂在
# op 上，打印出来一模一样，verifier 也不查——方言的 `verifyOperationAttribute`
# 只管 `pim.` 前缀的模块级属性名，裸名字（`datapath`）根本走不到它。
#
# 后果不是报错而是静默：读回侧按正则抠文本时，「ODS 里没这个字段」和「有字段但
# 这次没写」两种情况长得一样；下游没人读它，它就是一份装饰，改它不改变任何产物。
# 实测抓到过两处：DQ 相 3 的 `pim.kantor` 与旧缓存里的 `pim.quantize` 都挂过
# `datapath`，而两个 op 的 ODS 都没有这个字段。

def _declared_attrs_by_mnemonic() -> dict:
    """从 ODS 读出每个 `pim.*` op 声明了哪些字段名。

    算子级 op 的 `phases` / `isPhased` / `fpsu` / `kantor` 四项声明在
    `TTPIM_CommonOperatorAttrs` 里，由子类用 `!con(...)` 引入，所以要把基类那份
    并进来。只认 `let arguments` 那段，不能全文件捞 `$name`——builder 的参数表里
    也有同样的写法，会把不存在的字段当成已声明。
    """
    from genesim_bridge.paths import flagtree_source

    source = flagtree_source()
    td = (source / "include/triton/Dialect/TritonPIM/IR/PIMOps.td").read_text()

    def balanced(text: str, start: int, opening="(", closing=")") -> str:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == opening:
                depth += 1
            elif text[index] == closing:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        raise AssertionError("ODS 里的括号不配对")

    head = re.search(r"class TTPIM_CommonOperatorAttrs\s*\{\s*dag value = ", td)
    assert head, "ODS 里找不到 TTPIM_CommonOperatorAttrs"
    common = set(re.findall(
        r"\$(\w+)", balanced(td, td.index("(", head.end()))))

    declared = {}
    for match in re.finditer(
            r'def TTPIM_(\w+)Op\s*:\s*TTPIM_(Operator)?Op<"([a-z_0-9]+)"', td):
        mnemonic, body = match.group(3), match.end()
        args = re.search(r"let arguments = (!con\(|\()", td[body:body + 8000])
        if args is None:
            declared[mnemonic] = set(common) if match.group(2) else set()
            continue
        block = balanced(td, body + args.end() - 1)
        names = set(re.findall(r"\$(\w+)", block))
        if "TTPIM_OperatorAttrs.value" in block:
            names |= common
        declared[mnemonic] = names
    return declared


_PIM_OP_LINE_RE = re.compile(
    r"^\s*(?:%[\w#]+(?::\d+)?\s*(?:,\s*%[\w#]+(?::\d+)?\s*)*=\s*)?"
    r"pim\.(\w+)\s+(.*)$", re.MULTILINE)


def _toplevel_attr_names(rest: str) -> list:
    """取一行 IR 里最外层 `{...}` 的键名。

    只认最外层：属性值里还有 `<...>`（`#pim.datapath<nmuMode = ...>`），按里层的
    等号切会把参数名当成属性名。
    """
    start = rest.find("{")
    if start < 0:
        return []
    depth = 0
    for char in rest[:start]:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
    if depth != 0:            # 花括号在属性值里面，不是 attr-dict
        return []
    body, depth, current, parts = rest[start:], 0, "", []
    end = 0
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    body, depth, current = body[1:end], 0, ""
    for char in body:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [p.strip().split("=")[0].strip() for p in parts if p.strip()]


@pytest.mark.skipif(not _pim_passes_available(),
                    reason="当前 triton 没有 PIM pass")
def test_no_operator_carries_an_attribute_its_ods_does_not_declare():
    """14 个 mnemonic 展开后的 IR 里，不许出现 ODS 未声明的属性名。

    这条是「裸属性」这一类问题的通用闸门，不是针对某一个名字：新加一处
    `setAttr("foo", ...)` 而忘了在 ODS 里声明 `foo`，这里立刻变红。
    """
    declared = _declared_attrs_by_mnemonic()
    assert "kantor" in declared and "datapath" in declared["matmul"], (
        "ODS 解析失效了：连已知字段都没读出来，这条测试会变成永真")

    residue = {}
    for point in (ShapePoint(tq=1, tp=128), ShapePoint(tq=128, tp=0)):
        for mnemonic in MNEMONICS:
            text = _expand_oplevel(oplevel_ir(_OPLEVEL_DIMS, mnemonic, point))
            for match in _PIM_OP_LINE_RE.finditer(text):
                op = match.group(1)
                if op not in declared:
                    continue
                for name in _toplevel_attr_names(match.group(2)):
                    if name not in declared[op]:
                        residue.setdefault(f"pim.{op}", set()).add(name)
    assert not residue, (
        f"这些 op 挂了 ODS 没声明的属性：{ {k: sorted(v) for k, v in residue.items()} }。"
        "裸属性没人读、verifier 也不查，改它不会改变任何产物。")
