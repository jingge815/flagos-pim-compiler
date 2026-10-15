"""融合开关要反映到搬运字节上。

带激活的矩阵乘把 silu 折进主算子，未融合时 silu 单独占一次遍历。
省掉的字节等于那次遍历的输入加输出。
"""
from __future__ import annotations

import json

import pytest

from genesim_bridge.op_classify import ShapePoint, build_recipes


def _gpu_available() -> bool:
    """pimir 级要真实执行 FlagGems 算子，纯 CPU 上捕获不到 kernel。"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _passes_available() -> bool:
    try:
        from genesim_bridge.env import assert_pim_passes_available
        assert_pim_passes_available()
    except Exception:
        return False
    return True


def _use_genesim():
    """让本仓与 GeneSim 同时可导入，本仓优先。"""
    import sys
    from pathlib import Path
    from genesim_bridge.paths import genesim_root
    root = str(genesim_root() / "src")
    here = str(Path(__file__).resolve().parent.parent)
    for entry in (root, here):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


_DIMS = {"hidden_size": 64, "head_dim": 64, "num_heads": 1, "ffn_dim": 128}



def _module_text(body: str) -> str:
    return f'module attributes {{pim.target = "pim:v1"}} {{\n{body}\n}}\n'


def _memory_cycles(body: str) -> float:
    """走 GeneSim 的相位展开与计时，取访存周期。

    先按导出路径那样把整算子 IR 展开成相位链：融合的激活只有展开后才进账。
    """
    _use_genesim()
    from genesim_bridge.flagtree_driver import lower_oplevel_to_pimir
    from pim.pimir_trace import compile_phase_chain, parse_phase_chain
    from pim.pim_trace_loader import PIMTrace

    text = lower_oplevel_to_pimir(_module_text(body))
    instrs = compile_phase_chain(parse_phase_chain(text))
    return PIMTrace(instrs, {}).estimate_cycle_breakdown(None)["memory_cycles"]


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_fusion_saves_the_silu_traversal():
    """带激活的矩阵乘比「无激活矩阵乘加一次 silu」更省。

    省掉的是 silu 单独那次读入，但主算子要多写回一次激活结果，两者抵消后
    仍是净省。两边都用 fp16，比的是同一块中间张量。
    """
    from opcompiler_bridge.oplevel_kernel import lut_kernel, matmul_kernel

    point = ShapePoint(tq=4, tp=0)
    # 两边用同一份有权值的配方，差别只在激活折没折进去。
    fused = _memory_cycles(matmul_kernel(
        "linear", point.tq, 64, 64, activation="silu"))
    plain = _memory_cycles(matmul_kernel("linear", point.tq, 64, 64))
    # silu 作用在矩阵乘的输出上，形状必须与之相同，否则省掉的字节对不上。
    silu = _memory_cycles(lut_kernel("lut", (point.tq, 64), "silu"))
    saved = plain + silu - fused
    assert saved > 0
    # 折进主算子后激活直接消费结果，省掉 silu 单独的读写（fp16 的一次 LOADN
    # 加一次 STOREN），主算子只多写回一次激活结果（i8 的一次 STOREN）。
    silu_bytes = point.tq * 64 * 2
    saved_read = 12 + silu_bytes * 0.125
    saved_write = 10 + silu_bytes * 0.1
    extra_write = 10 + point.tq * 64 * 0.1
    assert saved == pytest.approx(saved_read + saved_write - extra_write)


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_fused_activation_costs_one_lookup():
    """折进主算子的激活只发一条查表指令，与单独的 silu 同口径。

    按搬运上限把查表指令切块会让融合侧的计算周期随规模上涨，而未融合侧
    始终一条，两边就不再可比。
    """
    from opcompiler_bridge.oplevel_kernel import lut_kernel, matmul_kernel

    _use_genesim()
    from genesim_bridge.flagtree_driver import lower_oplevel_to_pimir
    from pim.pim_isa import OPCODE_LUT, decode_instruction
    from pim.pimir_trace import compile_phase_chain, parse_phase_chain

    def lookups(body: str) -> int:
        text = lower_oplevel_to_pimir(_module_text(body))
        instrs = compile_phase_chain(parse_phase_chain(text))
        return sum(
            1 for raw in instrs
            if decode_instruction(raw)["opcode"] == OPCODE_LUT
        )

    point = ShapePoint(tq=64, tp=0)
    fused = lookups(matmul_kernel("linear", point.tq, 512, 512, activation="silu"))
    separate = lookups(lut_kernel("lut", (point.tq, 512), "silu"))
    assert fused == separate == 1


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_fused_matmul_trace_bills_the_activation_writeback():
    """带激活的矩阵乘编出的 trace 要比不带的多一次写回。

    折进去的激活直接消费主算子的结果，省掉单独的读入，但仍要写回一次。
    仿真时延取自这条 trace，写回漏了，融合就变成纯收益。
    """
    from opcompiler_bridge.oplevel_kernel import matmul_kernel

    _use_genesim()
    from genesim_bridge.flagtree_driver import lower_oplevel_to_pimir
    from pim.pim_isa import OPCODE_STOREN, decode_instruction
    from pim.pimir_trace import compile_phase_chain, parse_phase_chain

    def stores(body: str) -> int:
        text = lower_oplevel_to_pimir(_module_text(body))
        instrs = compile_phase_chain(parse_phase_chain(text))
        return sum(
            decode_instruction(raw)["imm"]
            for raw in instrs
            if decode_instruction(raw)["opcode"] == OPCODE_STOREN
        )

    point = ShapePoint(tq=4, tp=0)
    plain = stores(matmul_kernel("linear", point.tq, 64, 64))
    fused = stores(matmul_kernel("linear", point.tq, 64, 64, activation="silu"))
    # 结果是 i8，多出来的那一次写回就是结果张量本身。
    assert fused - plain == point.tq * 64



def _tiny_ir(path, *, fused: bool) -> None:
    """两层小模型：gate 投影后接 silu。

    `fused` 为真时 silu 已被图编译器吸收，模型 IR 里不再有这个节点。
    """
    import json

    operators = []
    dependencies = []
    op_id = 0
    for _layer in range(2):
        gate = op_id
        operators.append({
            "op_id": op_id, "op_type": "GEMM", "semantic_role": "gate_proj",
            "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
            "flops_coeffs": {}, "data_bytes_coeffs": {},
        })
        op_id += 1
        if not fused:
            operators.append({
                "op_id": op_id, "op_type": "SILU", "semantic_role": "",
                "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
                "flops_coeffs": {}, "data_bytes_coeffs": {},
            })
            dependencies.append({"src_op_id": gate, "dst_op_id": op_id})
            op_id += 1
    path.write_text(json.dumps({
        "model_id": "tiny", "num_layers": 2, "num_heads": 1, "head_dim": 64,
        "hidden_size": 64, "operators": operators, "dependencies": dependencies,
        # 每层两个算子：融合关时是投影加 silu，开时只有投影。
        "subgraphs": [list(range(i, i + (1 if fused else 2)))
                      for i in range(0, op_id, 1 if fused else 2)],
    }))


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_absorbed_activation_is_not_billed_twice(tmp_path):
    """gate 投影后接 silu 时，silu 的搬运不再单独计入汇总。

    主算子的配方已经带上激活，被吸收的节点再算一遍就是双计。融合与否
    由放置 sidecar 给出，不从 IR 形状推断。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=placement,
    )
    billed = [
        entry for entry in sidecar["operators"].values()
        if not entry["absorbed"]
    ]
    absorbed = [
        entry for entry in sidecar["operators"].values() if entry["absorbed"]
    ]
    assert absorbed, "被吸收的 silu 应该还在，只是不再单独计费"
    assert all(entry["op_type"] == "SILU" for entry in absorbed)
    assert all(entry["op_type"] == "GEMM" for entry in billed)



def test_four_dimensions_come_from_distinct_inputs():
    """四项代价各有来源，且互不串扰。

    计算与访存来自 trace，通信来自补边，并行来自切分宽度。这里只核对本仓
    能独立断言的两项：切分宽度改变单分片字节，且不改变算子的运算量。
    通信与墙钟在 genesim 的测试里断言，不跨仓导入。
    """
    from genesim_bridge.cost_extractor import _net_data_bytes
    from genesim_bridge.ir_cost import analyze_ir

    op = {
        "op_type": "GEMM",
        "input_shapes": [["Tq", 512]], "output_shapes": [["Tq", 2048]],
    }
    point = ShapePoint(tq=1, tp=0)
    full = _net_data_bytes(op, point, 2)
    half = _net_data_bytes(op, point, 2, local_features=(512, 1024))
    # tp 从 1 到 2，单分片字节下降且比例落在 (1, 2]。
    ratio = full / half
    assert 1.0 < ratio <= 2.0, ratio

    # 运算量只看形状，与切分宽度无关。
    text = (
        'module { tt.func @k(%a: tensor<4x64xf16>, %b: tensor<64x64xf16>) {'
        ' %0 = pim.matmul %a, %b : tensor<4x64xf16>, tensor<64x64xf16>'
        ' -> tensor<4x64xf16> } }')
    cost = analyze_ir(text, "k", (1,), {}, ir_level="oplevel")
    assert cost.flops == 2 * 4 * 64 * 64


def test_removing_one_input_zeroes_only_its_dimension():
    """去掉任何一项的输入，只有该项归零，其余三项不变。

    四项各有来源：运算数来自算子形状、搬运字节来自张量、单分片字节数来自
    tp_width，通信时延来自补边的字节数（在 genesim 的补边测试里断言，本仓
    不跨仓导入）。这里逐项去掉输入，看是否只动了自己那一项。
    """
    from genesim_bridge.cost_extractor import _net_data_bytes
    from genesim_bridge.ir_cost import analyze_ir

    compute_only = (
        'module { tt.func @k(%a: tensor<4x64xf16>, %b: tensor<64x64xf16>) {'
        ' %0 = pim.matmul %a, %b : tensor<4x64xf16>, tensor<64x64xf16>'
        ' -> tensor<4x64xf16> } }')
    move_only = (
        'module { tt.func @k(%a: !tt.ptr<f16>) {'
        ' pim.dma_load %a -> %b : tensor<4x64x!tt.ptr<f16>>'
        ' -> !pim.memdesc<4x64xf16, #pim.wram> } }')

    # 去掉张量输入：搬运归零，运算数不动。
    moved = analyze_ir(move_only, "k", (1,), {}, ir_level="oplevel")
    computed = analyze_ir(compute_only, "k", (1,), {}, ir_level="oplevel")
    assert moved.mram_traffic_bytes > 0 and moved.flops == 0
    assert computed.flops == 2 * 4 * 64 * 64 and computed.mram_traffic_bytes == 0

    # 去掉 tp_width（只剩一台 DPU）：单分片字节等于全局。
    op = {"op_type": "GEMM",
          "input_shapes": [["Tq", 512]], "output_shapes": [["Tq", 2048]]}
    point = ShapePoint(tq=1, tp=0)
    unsplit = _net_data_bytes(op, point, 2)
    split = _net_data_bytes(op, point, 2, local_features=(512, 1024))
    assert unsplit > split > 0
    # 切分宽度只喂给字节这一项：算子形状算出的运算数是 2MNK，与切了几刀无关。
    # 这一条在本仓只能锚成精确值——`analyze_ir` 没有切分参数，两侧无从相减。
    assert analyze_ir(compute_only, "k", (1,), {},
                      ir_level="oplevel").tile_flops_per_program == 2 * 4 * 64 * 64


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_more_operators_with_less_traffic_costs_less(tmp_path):
    """算子数更多但因融合搬运更少时，代价随搬运下降。"""
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)

    off = tmp_path / "off.json"
    _placement_sidecar(off, [])
    off_sc = export_costs_to_genesim(
        ir_path, tmp_path / "off.ir", tmp_path / "off_cost.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=off,
    )
    on = tmp_path / "on.json"
    _placement_sidecar(on, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])
    on_sc = export_costs_to_genesim(
        ir_path, tmp_path / "on.ir", tmp_path / "on_cost.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=on,
    )

    off_ir = json.loads((tmp_path / "off.ir").read_text())
    on_ir = json.loads((tmp_path / "on.ir").read_text())
    off_silu = [op for op in off_ir["operators"] if op["op_type"] == "SILU"]
    on_silu = [op for op in on_ir["operators"] if op["op_type"] == "SILU"]
    assert len(on_sc["operators"]) == len(off_sc["operators"])
    assert off_silu and all(op["data_bytes_coeffs"] for op in off_silu)
    assert on_silu and all(not op["data_bytes_coeffs"] for op in on_silu)
    off_coeff = sum(sum(op["data_bytes_coeffs"].values()) for op in off_silu)
    on_coeff = sum(sum(op["data_bytes_coeffs"].values()) for op in on_silu)
    assert on_coeff < off_coeff


def test_more_operators_but_less_traffic_costs_less(tmp_path):
    """算子数相同的两份 IR，融合的那份搬运更少，整图代价随搬运下降。

    两侧用同一份 IR（投影加 silu），差别只在放置清单标没标融合。融合侧
    的 silu 不再计费，主算子只补一次结果写回，整图字节必须更低。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)
    plain = tmp_path / "plain.json"
    _placement_sidecar(plain, [])
    fused = tmp_path / "fused.json"
    _placement_sidecar(fused, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])

    def billed(place, tag):
        sidecar = export_costs_to_genesim(
            ir_path, tmp_path / f"{tag}.out.ir", tmp_path / f"{tag}_cost.json",
            seq_len=4, cross_validate=False, ir_level="oplevel",
            placement_sidecar=place,
        )
        total = sum(
            entry["measurements"]["decode"]["data_bytes"]
            for entry in sidecar["operators"].values()
            if not entry["absorbed"]
        )
        return total

    plain_bytes = billed(plain, "plain")
    fused_bytes = billed(fused, "fused")
    assert fused_bytes < plain_bytes, (plain_bytes, fused_bytes)


def test_fusion_saves_exactly_the_absorbed_tensor_bytes(tmp_path):
    """融合开与关各导出一次，省掉的字节精确等于被吸收中间张量的字节数。

    关的一侧 silu 按读加写两份计，开的一侧这笔清零、主算子只多一次结果
    写回。decode 点上两项相抵，每个被吸收节点正好省下那份中间张量的字节数。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)
    point = ShapePoint(tq=1, tp=4)

    def billed_bytes(sidecar_path, out_ir):
        export_costs_to_genesim(
            ir_path, out_ir, tmp_path / (out_ir.stem + "_cost.json"),
            seq_len=4, cross_validate=False, ir_level="oplevel",
            placement_sidecar=sidecar_path,
        )
        total = 0.0
        for op in json.loads(out_ir.read_text())["operators"]:
            for key, coeff in op["data_bytes_coeffs"].items():
                term = {"Tq": point.tq, "constant": 1.0,
                        "Tq(Tp+Tq)": point.tq * point.lkv}[key]
                total += coeff * term
        return total

    off = tmp_path / "off.json"
    _placement_sidecar(off, [])
    on = tmp_path / "on.json"
    _placement_sidecar(on, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])
    saved = billed_bytes(off, tmp_path / "off.ir") - billed_bytes(on, tmp_path / "on.ir")
    # decode 点上 Tq(Tp+Tq) 是 5：silu 的系数 64 是一份读加一份写的中间张量，
    # 清零省下 64 x 5；主算子只多一次结果写回（128 字节）。两个节点都算上。
    absorbed_tensor_bytes = 2 * (64 * point.tq * point.lkv - 64 * 2)
    assert saved == absorbed_tensor_bytes, (saved, absorbed_tensor_bytes)


def test_four_ir_kinds_change_distinct_fields():
    """未融合、融合后、加宽 TP、加重通信四次评估，变化落在各自的项上。"""
    from genesim_bridge.cost_extractor import _net_data_bytes

    point = ShapePoint(tq=1, tp=0)
    plain = {"op_type": "GEMM",
             "input_shapes": [["Tq", 512]], "output_shapes": [["Tq", 2048]]}
    fused = dict(plain, fused_activation="silu")
    plain_bytes = _net_data_bytes(plain, point, 2)
    fused_bytes = _net_data_bytes(fused, point, 2)
    tp_bytes = _net_data_bytes(plain, point, 2, local_features=(512, 1024))
    # 融合只改搬运：折进去的是一次结果写回。
    assert fused_bytes == plain_bytes + 1 * 2048 * 2
    # 加宽 TP 只改单分片字节，比例落在 (1, 2]。
    assert 1.0 < plain_bytes / tp_bytes <= 2.0
    # 加重通信不改算子搬运。
    assert _net_data_bytes(plain, point, 2) == plain_bytes
    # 生产精化带放置 sidecar，GEMM 走本地分片的早退路径。融合多出的那次
    # 结果写回也得算在这条路径上，否则分片后的融合与未融合字节完全相同。
    local = (512, 1024)
    assert (_net_data_bytes(fused, point, 2, local_features=local)
            == _net_data_bytes(plain, point, 2, local_features=local)
            + point.tq * local[1] * 2)


def _fused_ir(path) -> None:
    """一笔带模板系数的 SILU：pimir 级不桥接它，得靠清零才不计费。"""
    import json

    path.write_text(json.dumps({
        "model_id": "tiny", "num_layers": 1, "num_heads": 1, "head_dim": 64,
        "hidden_size": 64,
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
             "flops_coeffs": {}, "data_bytes_coeffs": {}},
            {"op_id": 1, "op_type": "SILU", "semantic_role": "",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
             # 模板系数：不主动清掉就会原样留在 IR 里被 GeneSim 照计。
             "flops_coeffs": {"Tq": 44032}, "data_bytes_coeffs": {"Tq": 44032}},
        ],
        "dependencies": [{"src_op_id": 0, "dst_op_id": 1}],
        "subgraphs": [[0, 1]],
    }))


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_absorbed_operator_is_not_billed(tmp_path):
    """被吸收的 silu 系数必须是空的，否则 GeneSim 会再计一次。"""
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _fused_ir(ir_path)
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=placement,
    )
    refined = json.loads((tmp_path / "out.ir").read_text())
    absorbed = [op for op in refined["operators"] if op["op_type"] == "SILU"]
    assert absorbed
    for op in absorbed:
        assert op["data_bytes_coeffs"] == {}, op
        assert op["flops_coeffs"] == {}, op
    # sidecar 里仍留着测量，供对照。
    marked = [e for e in sidecar["operators"].values() if e["absorbed"]]
    assert marked and all(e["op_type"] == "SILU" for e in marked)


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
@pytest.mark.skipif(not _gpu_available(), reason="pimir 级测量需要 GPU")
def test_absorbed_operator_is_not_billed_at_pimir_level(tmp_path):
    """pimir 级同样要把被吸收的 silu 系数清零。

    这一级 SILU 不在桥接表里，走的是模板分支，清零不能只放在桥接分支上。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _fused_ir(ir_path)
    source = json.loads(ir_path.read_text())
    silu = next(op for op in source["operators"] if op["op_type"] == "SILU")
    silu["data_bytes"] = 4096
    ir_path.write_text(json.dumps(source))
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="pimir",
        placement_sidecar=placement,
    )
    refined = json.loads((tmp_path / "out.ir").read_text())
    absorbed = [op for op in refined["operators"] if op["op_type"] == "SILU"]
    assert absorbed
    for op in absorbed:
        assert op["data_bytes_coeffs"] == {}, op["data_bytes_coeffs"]
        assert op["flops_coeffs"] == {}, op["flops_coeffs"]
        assert op["data_bytes"] == 0, op["data_bytes"]

@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
@pytest.mark.skipif(not _gpu_available(), reason="pimir 级测量需要 GPU")
def test_fused_gemm_exports_at_pimir_level(tmp_path):
    """带 fused_activation 的 GEMM 在 pimir 级导出不得抛错。

    pimir 是 llama-2.md 第四步的默认口径，这条路必须能跑完，且权值搬运
    不能因为折了激活就丢掉。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _fused_ir(ir_path)
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="pimir",
        placement_sidecar=placement,
    )
    gemm = sidecar["operators"]["0"]
    assert gemm["op_type"] == "GEMM"
    assert gemm["mram_traffic_bytes"]["prefill"] > 0


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
@pytest.mark.skipif(not _gpu_available(), reason="pimir 级测量需要 GPU")
def test_fused_projection_bills_one_result_traversal_at_pimir_level(tmp_path):
    """pimir 级折激活的主算子搬运 = 未融合主算子 + 一次结果遍历。

    生产链路走 --ir-level pimir。A 路 kernel 本身不带激活属性，不注解就
    等于主算子一分不多计，融合收益被记成整个 SILU。
    """
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _same_shape_gate_up_ir(ir_path, gate_first=True)
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="pimir",
        placement_sidecar=placement,
    )
    gate = sidecar["operators"]["0"]["mram_traffic_bytes"]["prefill"]
    up = sidecar["operators"]["2"]["mram_traffic_bytes"]["prefill"]
    extra = gate - up
    kernel = sidecar["operators"]["0"]["measurements"]["prefill"]["pim_kernels"][0]
    # 写回在 K 循环外。本夹具输出宽等于 tile_n，只有一个输出 tile。
    # 乘上内层趟数仍能被 128 整除，所以必须钉成精确值。
    one_tile = kernel["tile_m"] * kernel["tile_n"] * 2
    grid = 1
    for g in kernel["grid"] or [1]:
        grid *= g
    assert extra == one_tile * grid, (extra, one_tile, grid, kernel["loop_trip_counts"])


def _same_shape_gate_up_ir(path, *, gate_first: bool) -> None:
    """同形状的 gate 投影（后接 silu）与 up 投影。

    `gate_first` 决定测量顺序：缓存键若漏掉融合状态，先测到的那条配方
    会被另一个算子复用，两个值会随顺序一起翻转。
    """
    gate = {"op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj",
            "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
            "flops_coeffs": {}, "data_bytes_coeffs": {}}
    silu = {"op_id": 1, "op_type": "SILU", "semantic_role": "",
            "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
            "flops_coeffs": {}, "data_bytes_coeffs": {}}
    up = {"op_id": 2, "op_type": "GEMM", "semantic_role": "up_proj",
          "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
          "flops_coeffs": {}, "data_bytes_coeffs": {}}
    operators = [gate, silu, up] if gate_first else [up, gate, silu]
    path.write_text(json.dumps({
        "model_id": "tiny", "num_layers": 1, "num_heads": 1, "head_dim": 64,
        "hidden_size": 64, "operators": operators,
        "dependencies": [{"src_op_id": 0, "dst_op_id": 1}],
        "subgraphs": [[0, 1, 2]],
    }))


def _export_gate_up(tmp_path, *, gate_first: bool, tag: str) -> tuple[float, float]:
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / f"{tag}.ir"
    _same_shape_gate_up_ir(ir_path, gate_first=gate_first)
    placement = tmp_path / f"{tag}.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / f"{tag}_out.ir", tmp_path / f"{tag}_out.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=placement,
    )
    return (
        sidecar["operators"]["0"]["mram_traffic_bytes"]["prefill"],
        sidecar["operators"]["2"]["mram_traffic_bytes"]["prefill"],
    )


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_fused_projection_keeps_its_weight_traffic(tmp_path):
    """折激活的投影搬运必须多于不折激活的同形状投影。

    折进去的是一次结果遍历，权值侧不能丢。同形状的 up 投影不折，两者
    不得被测量缓存合成同一个数。
    """
    gate_mram, up_mram = _export_gate_up(tmp_path, gate_first=True, tag="fwd")
    assert gate_mram > up_mram, (gate_mram, up_mram)


@pytest.mark.skipif(not _passes_available(), reason="当前 triton 没有 PIM pass")
def test_fusion_measurement_does_not_leak_across_same_shape_ops(tmp_path):
    """同形状算子的融合测量不得随出现顺序互相污染。"""
    fused_first, plain_first = _export_gate_up(
        tmp_path, gate_first=True, tag="gate_first")
    fused_second, plain_second = _export_gate_up(
        tmp_path, gate_first=False, tag="up_first")
    assert fused_first == fused_second, (fused_first, fused_second)
    assert plain_first == plain_second, (plain_first, plain_second)
    assert fused_first > plain_first, (fused_first, plain_first)


def test_refined_ir_loads_back_in_genesim(tmp_path):
    """导出的精化 IR 必须能被 GeneSim 的 ModelIR.load 读回。

    融合标记是导出侧的内部记号，写进跨仓产物会让仿真在加载阶段抛
    TypeError，第五步 `./run.sh` 在建模之前就崩。
    """
    import json
    import sys

    from genesim_bridge.cost_extractor import export_costs_to_genesim
    from genesim_bridge.paths import genesim_root

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)
    out_ir = tmp_path / "out.ir"
    export_costs_to_genesim(
        ir_path, out_ir, tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
    )

    # 写出的算子只能带 GeneSim Operator 认识的字段。
    refined = json.loads(out_ir.read_text())
    leaked = [
        (op["op_id"], key)
        for op in refined["operators"]
        for key in ("fused_activation", "absorbed")
        if key in op
    ]
    assert not leaked, f"内部标记写进了精化 IR：{leaked}"

    root = str(genesim_root() / "src")
    if root not in sys.path:
        sys.path.insert(0, root)
    from ir.model_ir import ModelIR

    loaded = ModelIR.load(out_ir)
    assert len(loaded.operators) == len(refined["operators"])


def _placement_sidecar(path, fused_activations) -> None:
    """一份最小的放置 sidecar，只带融合清单。"""
    import json

    path.write_text(json.dumps({
        "version": 3, "fused_activations": fused_activations,
    }))


def test_fusion_comes_from_the_placement_sidecar(tmp_path):
    """融合开关由放置 sidecar 决定，不看模型 IR 里有没有 silu 节点。

    同一份 IR：sidecar 标了融合才折进主算子，没标就两个算子都照常计费。
    """
    import json

    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)

    off = tmp_path / "off.json"
    _placement_sidecar(off, [])
    export_costs_to_genesim(
        ir_path, tmp_path / "off.ir", tmp_path / "off_cost.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=off,
    )
    off_ir = json.loads((tmp_path / "off.ir").read_text())
    assert all(op["data_bytes_coeffs"] for op in off_ir["operators"]), \
        "没标融合时 silu 不该被清零"

    on = tmp_path / "on.json"
    _placement_sidecar(on, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])
    export_costs_to_genesim(
        ir_path, tmp_path / "on.ir", tmp_path / "on_cost.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=on,
    )
    on_ir = json.loads((tmp_path / "on.ir").read_text())
    cleared = [op for op in on_ir["operators"] if not op["data_bytes_coeffs"]]
    assert len(cleared) == 2 and all(op["op_type"] == "SILU" for op in cleared)


def test_absorbed_activation_clears_concrete_data_bytes_too(tmp_path):
    """融合节点的零系数不能回退到源 IR 的静态搬运量。"""
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _tiny_ir(ir_path, fused=False)
    source = json.loads(ir_path.read_text())
    silu = next(op for op in source["operators"] if op["op_type"] == "SILU")
    silu["data_bytes"] = 4096
    silu["data_bytes_coeffs"] = {"Tq": 44032}
    ir_path.write_text(json.dumps(source))

    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"},
        {"layer": 1, "role": "gate_proj", "activation": "silu"},
    ])
    sidecar = export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level="oplevel",
        placement_sidecar=placement,
    )

    refined = json.loads((tmp_path / "out.ir").read_text())
    silu_ops = [op for op in refined["operators"] if op["op_type"] == "SILU"]
    assert all(op["data_bytes"] == 0 for op in silu_ops)
    assert all(not op["data_bytes_coeffs"] for op in silu_ops)



def test_fusion_rejects_activation_name_mismatch(tmp_path):
    """IR 后继是 SILU 时，清单不能把它误标成 relu。"""
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    ir_path.write_text(json.dumps({
        "model_id": "tiny", "num_layers": 1, "num_heads": 1, "head_dim": 64,
        "hidden_size": 64,
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
             "flops_coeffs": {}, "data_bytes_coeffs": {}},
            {"op_id": 1, "op_type": "SILU", "semantic_role": "",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]],
             "flops_coeffs": {"Tq": 1}, "data_bytes_coeffs": {"Tq": 1}},
        ],
        "dependencies": [{"src_op_id": 0, "dst_op_id": 1}],
        "subgraphs": [[0, 1]],
    }))
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "relu"}])
    with pytest.raises(ValueError, match="relu"):
        export_costs_to_genesim(
            ir_path, tmp_path / "out.ir", tmp_path / "out.json",
            seq_len=4, cross_validate=False, ir_level="oplevel",
            placement_sidecar=placement,
        )


def test_fusion_skips_an_unrelated_consumer_before_the_named_activation():
    """融合清单命中的激活必须由依赖边唯一确定，不能取第一条边。"""
    from genesim_bridge.cost_extractor import _mark_fused_activations

    ir = {
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj"},
            {"op_id": 1, "op_type": "GEMM", "semantic_role": "up_proj"},
            {"op_id": 2, "op_type": "SILU", "semantic_role": ""},
        ],
        # 故意把无关消费者放在 SILU 前，复现按依赖边顺序取 dst 的问题。
        "dependencies": [
            {"src_op_id": 0, "dst_op_id": 1},
            {"src_op_id": 0, "dst_op_id": 2},
        ],
        "subgraphs": [[0, 1, 2]],
    }

    _mark_fused_activations(ir, {(0, "gate_proj", "silu")})

    assert ir["operators"][0]["fused_activation"] == "silu"
    assert ir["operators"][2]["absorbed"] is True
    assert "absorbed" not in ir["operators"][1]


def test_fusion_rejects_missing_named_activation_consumer():
    """清单指定的激活不存在时必须报错，不能静默吸收错误节点。"""
    from genesim_bridge.cost_extractor import _mark_fused_activations

    ir = {
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "semantic_role": "gate_proj"},
            {"op_id": 1, "op_type": "SILU", "semantic_role": ""},
        ],
        "dependencies": [{"src_op_id": 0, "dst_op_id": 1}],
        "subgraphs": [[0, 1]],
    }

    with pytest.raises(ValueError, match="relu"):
        _mark_fused_activations(ir, {(0, "gate_proj", "relu")})


@pytest.mark.parametrize("ir_level", ["oplevel", "pimir"])
def test_absorbed_ops_are_listed_for_the_simulator(tmp_path, ir_level):
    """被吸收的算子编号要写进放置 sidecar，仿真侧才知道跳过它的 trace。

    系数清零只落在导出量上：被吸收节点照样带着 pimir_path 进算子编译，
    PIM 时延取自 trace 周期，融合开与关的仿真报告逐位相同。两个级别都要
    覆盖——pimir 级的被吸收算子走的是「不桥接」分支，不进成本 sidecar，
    只按成本 sidecar 收集会漏掉真实链路上的那一批。
    """
    if ir_level == "pimir" and not _gpu_available():
        pytest.skip("pimir 级测量需要 GPU")
    from genesim_bridge.cost_extractor import export_costs_to_genesim

    ir_path = tmp_path / "model.ir"
    _fused_ir(ir_path)
    placement = tmp_path / "placement.json"
    _placement_sidecar(placement, [
        {"layer": 0, "role": "gate_proj", "activation": "silu"}])
    export_costs_to_genesim(
        ir_path, tmp_path / "out.ir", tmp_path / "out.json",
        seq_len=4, cross_validate=False, ir_level=ir_level,
        placement_sidecar=placement,
    )
    updated = json.loads(placement.read_text())
    absorbed = next(
        op for op in json.loads(ir_path.read_text())["operators"]
        if op["op_type"] == "SILU"
    )
    assert updated["absorbed_ops"] == [absorbed["op_id"]]


def test_refine_step_passes_fusion_from(tmp_path, monkeypatch):
    """精化命令必须带上放置 sidecar 的融合清单。"""
    from scripts.run_full_pipeline import step_a2_refine_pimir_ir, _PIMIR_IR_NAME

    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    ir = genesim / "models" / "llama2_7b.ir"
    ir.write_text(json.dumps({
        "operators": [{"op_id": 0, "op_type": "GEMM"}],
    }))
    placement = tmp_path / "placement.json"
    placement.write_text("{}")
    captured = {}

    def fake_run(cmd, cwd, log_path):
        captured["cmd"] = cmd
        out = genesim / "models" / _PIMIR_IR_NAME
        out.write_text(ir.read_text())

    monkeypatch.setattr("scripts.run_full_pipeline._run", fake_run)
    step_a2_refine_pimir_ir(genesim, ir, tmp_path, placement=placement)
    assert "--fusion-from" in captured["cmd"]
    assert str(placement) in captured["cmd"]


def test_refine_does_not_reuse_ir_when_fusion_sidecar_is_nonempty(tmp_path, monkeypatch):
    """同编号的旧精化 IR 也不能掩盖新的融合清单。"""
    from scripts.run_full_pipeline import step_a2_refine_pimir_ir, _PIMIR_IR_NAME

    genesim = tmp_path / "genesim"
    (genesim / "models").mkdir(parents=True)
    ir = genesim / "models" / "llama2_7b.ir"
    payload = {"operators": [{"op_id": 0, "op_type": "GEMM"}]}
    ir.write_text(json.dumps(payload))
    (genesim / "models" / _PIMIR_IR_NAME).write_text(json.dumps(payload))
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"fused_activations": [{"layer": 0}]}))
    captured = {}

    def fake_run(cmd, cwd, log_path):
        captured["cmd"] = cmd

    monkeypatch.setattr("scripts.run_full_pipeline._run", fake_run)
    step_a2_refine_pimir_ir(genesim, ir, tmp_path, placement=placement)
    assert "--fusion-from" in captured["cmd"]


def test_fused_activation_flops_agree_across_ir_levels():
    """同一份带 act_spec 的分块 GEMM，ttir 与 pimir 的运算数必须一致。

    折激活按最后一次写回计。只在 pimir 分支加这一笔，两路对照在融合算子
    上永远对不上，而需求 2.4 正是用这条抓漏算或重算。
    """
    from genesim_bridge.ir_cost import analyze_ir, fused_act_spec_attr

    attr = fused_act_spec_attr("silu")
    text = f"""module {{
  tt.func @k(%a: !tt.ptr<f16>) {{
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c32 = arith.constant 32 : i32
    scf.for %i = %c0 to %c64 step %c32 : i32 {{
      %0 = tt.dot %a, %b {{{attr}}} : tensor<1x32xf16> * tensor<32x32xf16> -> tensor<1x32xf32>
      pim.dma_store %s -> %p {{base_arg = 2 : i64}} : !pim.memdesc<1x32xf16, #pim.wram> -> tensor<1x32x!tt.ptr<f16>>
    }}
    tt.return
  }}
}}"""
    ttir = analyze_ir(text, "k", (1,), {}, ir_level="ttir")
    pimir = analyze_ir(text, "k", (1,), {}, ir_level="pimir")
    assert ttir.flops == pimir.flops, (ttir.flops, pimir.flops)
    # 2MNK × 2 趟，再加最后一次写回 1×32 × 2 趟。
    assert ttir.flops == 2 * 1 * 32 * 32 * 2 + 32 * 2


def test_tail_rounding_comment_matches_the_code():
    """尾块注释的方向要和向上取整一致。

    上界不是步长整数倍时，按整趟计是把不满的尾块漏掉，不是多算出去；
    向上取整才把尾块算进去，为的是和 trace 侧的整块累加对齐。
    """
    from pathlib import Path

    text = Path("genesim_bridge/ir_cost.py").read_text()
    assert "多算出去" not in text


def test_fused_act_spec_is_written_and_matched_the_same_way():
    """写出与识别必须共用同一份属性字面量。"""
    from genesim_bridge.ir_cost import fused_act_spec_attr, has_fused_act_spec
    from opcompiler_bridge.oplevel_kernel import matmul_kernel

    attr = fused_act_spec_attr("silu")
    assert has_fused_act_spec(f"%0 = pim.matmul %a, %b {{{attr}}} : tensor<1x8xi8>")
    body = matmul_kernel("linear", 4, 64, 64, activation="silu")
    assert attr in body
    assert has_fused_act_spec(body)


def test_stale_fused_kernels_are_rejected(tmp_path):
    """融合清单非空、kernel 却没有 act_spec 时必须拒绝这份 sidecar。"""
    from scripts.run_full_pipeline import fused_kernels_missing_act_spec

    kernel = tmp_path / "plain.pimir.mlir"
    kernel.write_text("tt.dot %a, %b : tensor<1x32xf16>")
    sidecar = {
        "fused_activations": [
            {"layer": 0, "role": "gate_proj", "activation": "silu"},
        ],
        "operators": {
            "0": {"semantic_role": "gate_proj", "pimir_path": str(kernel)},
        },
    }
    assert fused_kernels_missing_act_spec(sidecar) == [str(kernel)]
    kernel.write_text(
        "tt.dot %a, %b {activation = #pim.act_spec<kind = silu>} : tensor<1x32xf16>"
    )
    assert fused_kernels_missing_act_spec(sidecar) == []
