"""算子层五类优化在 A、B 两路上的结构断言。

A 路吃 TTIR，B 路吃整算子级 pim mlir。每个 Pass 一条落在适用范围内、
断言 IR 真的被改了，一条落在不支持范围内、断言不动。数值对拍沿用
`tests/test_opcompiler_ops.py` 的整算子编译用例，这里只查结构。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from opcompiler_bridge.driver import _run_passes, _triton_opt


def _has_pim_passes() -> bool:
    binary = _triton_opt()
    if not binary.is_file():
        return False
    import subprocess

    proc = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
    return "--pim-constant-fold" in proc.stdout


pytestmark = pytest.mark.skipif(
    not _has_pim_passes(),
    reason="当前 triton-opt 没有 PIM 优化 pass，需重编 FlagTree",
)


def _opt(text: str, *passes: str) -> str:
    return _run_passes(_triton_opt(), text, list(passes), "优化")


# ---------------------------------------------------------------- A 路

_A_FOLD = """
module {
  tt.func public @fold(%x: tensor<4xi32>) -> tensor<4xi32> {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %sum = arith.addi %c2, %c3 : i32
    %v = tt.splat %sum : i32 -> tensor<4xi32>
    %y = arith.addi %v, %x : tensor<4xi32>
    tt.return %y : tensor<4xi32>
  }
}
"""


def test_a_path_folds_arith_constant() -> None:
    """循环边界里的 arith 常量运算被 -pim-constant-fold 折成一个常量。

    这条覆盖新 Pass 自己的标量路径，不依赖上游 -canonicalize。
    """
    out = _opt(_A_FOLD, "-pim-constant-fold")
    assert "arith.constant 5" in out, out
    assert out.count("arith.addi") == 1, out


def test_a_path_folds_dense_constant() -> None:
    """两个 dense 常量张量相加被折成一个常量。

    真实 linear kernel 转 PIM 后，常量全是 dense 张量而不是标量：
    地址偏移是 `arith.constant dense<64>` 再加另一个常量。只折标量的话，
    这条通路上常量折叠一次都不会触发。
    """
    text = """
module {
  tt.func public @dense(%o: tensor<4x!tt.ptr<i32>>) {
    %a = arith.constant dense<64> : tensor<4xi32>
    %b = arith.constant dense<1> : tensor<4xi32>
    %s = arith.addi %a, %b : tensor<4xi32>
    tt.store %o, %s : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-constant-fold")
    assert "arith.addi" not in out, out
    assert "arith.constant dense<65>" in out, out


def test_a_path_fold_keeps_dense_with_dynamic_operand() -> None:
    """dense 常量的另一个操作数来自输入时不折。"""
    text = """
module {
  tt.func public @keep(%x: tensor<4xi32>, %o: tensor<4x!tt.ptr<i32>>) {
    %a = arith.constant dense<64> : tensor<4xi32>
    %s = arith.addi %a, %x : tensor<4xi32>
    tt.store %o, %s : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-constant-fold")
    assert "arith.addi" in out, out


def test_a_path_fold_keeps_dynamic() -> None:
    """操作数来自输入时不折。"""
    text = """
module {
  tt.func public @keep(%a: i32, %b: i32) -> i32 {
    %s = arith.addi %a, %b : i32
    tt.return %s : i32
  }
}
"""
    out = _opt(text, "-canonicalize")
    assert "arith.addi" in out, out


_A_CSE = """
module {
  tt.func public @cse(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi32> {
    %m1 = arith.addi %x, %y : tensor<4xi32>
    %m2 = arith.addi %x, %y : tensor<4xi32>
    %s = arith.addi %m1, %m2 : tensor<4xi32>
    tt.return %s : tensor<4xi32>
  }
}
"""


def test_a_path_cse_merges_arith() -> None:
    """循环内结构相同的 arith 运算由 -pim-cse 合并成一处。"""
    out = _opt(_A_CSE, "-pim-cse")
    assert out.count("arith.addi") == 2, out


def test_a_path_cse_keeps_distinct() -> None:
    """操作数不同的同型运算不合并。"""
    text = """
module {
  tt.func public @keep(%x: tensor<4xi32>, %y: tensor<4xi32>,
                       %z: tensor<4xi32>) -> tensor<4xi32> {
    %m1 = arith.addi %x, %y : tensor<4xi32>
    %m2 = arith.addi %x, %z : tensor<4xi32>
    %s = arith.addi %m1, %m2 : tensor<4xi32>
    tt.return %s : tensor<4xi32>
  }
}
"""
    out = _opt(text, "-pim-cse")
    assert out.count("arith.addi") == 3, out


_A_DCE = """
module {
  tt.func public @dce(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi32> {
    %dead = arith.muli %x, %y : tensor<4xi32>
    %s = arith.addi %x, %y : tensor<4xi32>
    tt.return %s : tensor<4xi32>
  }
}
"""


def test_a_path_dce_drops_unused_arith() -> None:
    """A 路的输出走 tt.store 或返回值，没人用的张量 arith 由 -pim-dce 删除。"""
    out = _opt(_A_DCE, "-pim-dce")
    assert "arith.muli" not in out, out
    assert "arith.addi" in out, out


def test_a_path_dce_drops_unused_tensor_chain() -> None:
    """无用户的张量地址链多轮删除，被存储的结果保留。"""
    text = """
module {
  tt.func public @chain(%x: tensor<4xi32>, %p: tensor<4x!tt.ptr<i32>>) {
    %a = arith.muli %x, %x : tensor<4xi32>
    %b = arith.addi %a, %x : tensor<4xi32>
    %q = tt.addptr %p, %b : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %s = arith.subi %x, %x : tensor<4xi32>
    tt.store %p, %s : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-dce")
    for dead in ("arith.muli", "arith.addi", "tt.addptr"):
        assert dead not in out, out
    assert "arith.subi" in out and "tt.store" in out, out


def test_a_path_dce_triggers_on_real_gated_linear() -> None:
    """真实门控 linear 默认 A 路上，silu 链折进 tt.dot 后留下的常量 1 由 -pim-dce 删除。

    不带激活的请求没有这条残留，`-pim-dce` 前后逐字节相同。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    passes = a_path_passes(DEFAULT_HARDWARE_CONFIG, target="pim:v1")
    before_dce = passes[:passes.index("-pim-dce")]
    one = "dense<1.000000e+00>"
    for activation in ("silu", None):
        req = OpCompileRequest(
            op="linear", arg_shapes=[(64, 64), (64, 64)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
            activation=activation)
        pre = _opt(_make_ttir(req), *before_dce)
        post = _opt(pre, "-pim-dce")
        if activation == "silu":
            assert one in pre, pre
            assert one not in post, post
            assert "activation = " in post, post
        else:
            assert post == pre


def test_a_path_dce_drops_dead_pim_layout_op() -> None:
    """真实 A 路 IR 里没人用的 `pim.convert_layout` 由 -pim-dce 删除。

    「无用户张量即内核输出」只在整算子函数里成立；A 路输出走 tt.store，
    pim 布局算子没人用就是死的。
    """
    import re

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    passes = a_path_passes(DEFAULT_HARDWARE_CONFIG, target="pim:v1")
    req = OpCompileRequest(
        op="linear", arg_shapes=[(64, 64), (64, 64)],
        hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16", activation=None)
    pre = _opt(_make_ttir(req), *passes[:passes.index("-pim-dce")])
    live = pre.count("pim.convert_layout")
    assert live > 0, pre
    # 复制一条已有的布局转换，结果不给任何人用。
    m = re.search(r"^(\s*)%\S+ = (pim\.convert_layout .*)$", pre, re.M)
    text = pre[:m.end() + 1] + f"{m.group(1)}%dead_cl = {m.group(2)}\n" + pre[m.end() + 1:]

    out = _opt(text, "-pim-dce")

    assert "%dead_cl" in text
    assert out.count("pim.convert_layout") == live, out


def test_a_path_dce_keeps_used() -> None:
    """结果被返回的同型运算保留。"""
    text = """
module {
  tt.func public @keep(%x: tensor<4xi32>, %y: tensor<4xi32>) -> tensor<4xi32> {
    %s = arith.muli %x, %y : tensor<4xi32>
    tt.return %s : tensor<4xi32>
  }
}
"""
    out = _opt(text, "-pim-dce")
    assert "arith.muli" in out, out


def test_real_linear_kernel_has_nothing_to_fold() -> None:
    """真实 linear kernel 上没有两个操作数都是常量的运算，折叠改动数为 0。

    16、64、128、256 四个尺寸都查。这是默认通路的口径，不是漏触发：
    折叠在合成 IR 上验证，见需求文档第五节第 3 条。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    convert = a_path_passes(DEFAULT_HARDWARE_CONFIG, target="pim:v1")[0]
    for size in (16, 64, 128, 256):
        req = OpCompileRequest(
            op="linear", arg_shapes=[(size, size), (size, size)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16")
        ttir = _make_ttir(req)
        converted = _opt(ttir, convert)
        folded = _opt(ttir, convert, "-pim-constant-fold")
        assert folded == converted, f"{size} 的真实 kernel 被折叠改动了"


def test_a_path_pipeline_runs_all_five() -> None:
    """A 路管线带着五类优化；分块与融合在真实 kernel 上确实改动 IR。

    折叠、CSE、DCE 在这份 16×16 kernel 上没有作用对象（设计 7.3 已注明），
    由各自在合成 IR 上的用例验证；这里钉住管线接线与两个真实触发的类别。
    """
    import dataclasses

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import OPT_PASSES, a_path_passes

    hw = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1,
        mram_bytes_per_dpu=1 << 20)
    req = OpCompileRequest(
        op="linear", arg_shapes=[(16, 16), (16, 16)], hardware=hw,
        dtype="float16")
    passes = a_path_passes(hw, target="pim:v1")
    missing = [p for p in OPT_PASSES if p not in passes]
    assert not missing, f"A 路管线缺 {missing}"
    out = _opt(_make_ttir(req), *passes)
    for attr in ("pim.tile-m", "pim.tile-n", "pim.tile-k"):
        assert attr in out, f"分块没有写 {attr}"

    # 融合这一类：请求带 activation=silu 时，dot 后的 silu 五连必须被折进
    # tt.dot（链算子消失、属性落位）。16x16 没有 K 循环，链根是 dot 结果。
    fused = _opt(_make_ttir(dataclasses.replace(req, activation="silu")),
                 *a_path_passes(hw, target="pim:v1"))
    for gone in ("math.exp", "arith.subf", "arith.divf", "arith.mulf"):
        assert gone not in fused, f"silu 链没有折掉：{gone}"
    assert "activation = #pim.act_spec<kind = silu>" in fused


# ---------------------------------------------------------------- B 路

_DATAPATH = ("datapath = #pim.datapath<nmuMode = fixed_point, "
             "scaleMode = fixed_point>")


def _eltwise(result, lhs, rhs, kind):
    return (f"    {result} = pim.eltwise {lhs}, {rhs} {{{_DATAPATH}, "
            f"kind = #pim.eltwise<{kind}>}} "
            f": tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>")


def _op_count(text, op):
    """只数算子出现次数，属性里的同名片段不算。"""
    return text.count(f" = {op} ") + text.count(f"\n    {op} ")


def _b_func(name, args, body, results=""):
    ret = f"    tt.return {results}: tensor<4xf32>\n" if results else \
        "    tt.return\n"
    return (f"module {{\n  tt.func @{name}({args}) {{\n"
            + "\n".join(body) + "\n" + ret + "  }\n}\n")


def test_b_path_folds_constant_eltwise() -> None:
    """展开后操作数全是常量的 pim.eltwise 被折成常量。"""
    text = _b_func("fold", "%o: tensor<4x!tt.ptr<f32>>", [
        "    %a = arith.constant dense<2.000000e+00> : tensor<4xf32>",
        "    %b = arith.constant dense<3.000000e+00> : tensor<4xf32>",
        _eltwise("%c", "%a", "%b", "add"),
        "    tt.store %o, %c : tensor<4x!tt.ptr<f32>>",
    ])
    out = _opt(text, "-pim-expand-phases", "-pim-constant-fold")
    assert "pim.eltwise" not in out, out
    assert "arith.constant dense<5.000000e+00>" in out, out


def test_b_path_fold_skips_oversized_result() -> None:
    """结果元素数超过上限的常量 pim.eltwise 不折。"""
    n = 8192
    sig = "tensor<%dxf32>" % n
    ptr = "tensor<%dx!tt.ptr<f32>>" % n
    body = [
        "    %%a = arith.constant dense<2.000000e+00> : %s" % sig,
        "    %%b = arith.constant dense<3.000000e+00> : %s" % sig,
        "    %%c = pim.eltwise %%a, %%b {%s, kind = #pim.eltwise<add>} "
        ": %s, %s -> %s" % (_DATAPATH, sig, sig, sig),
        "    tt.store %%o, %%c : %s" % ptr,
    ]
    text = ("module {{\n  tt.func @big(%o: {ptr}) {{\n".format(ptr=ptr)
            + "\n".join(body) + "\n    tt.return\n  }\n}\n")
    out = _opt(text, "-pim-constant-fold")
    assert "pim.eltwise" in out, out


def test_real_b_kernels_have_nothing_to_fold() -> None:
    """真实 B 路 kernel 展开后没有常量，折叠输出与输入逐字节相同。

    softmax、rope、eltwise、lut 的操作数全是函数参数或上游相位的结果。
    这是默认通路的口径，见需求文档第五节第 3 条。
    """
    from opcompiler_bridge.oplevel_kernel import (
        eltwise_kernel, lut_kernel, rope_kernel, softmax_kernel)

    cases = {
        "softmax": softmax_kernel("k", 2, 16),
        "rope": rope_kernel("k", 4, 16, 16),
        "eltwise": eltwise_kernel("k", (2, 16), "add"),
        "lut": lut_kernel("k", (2, 16), "silu"),
    }
    for name, body in cases.items():
        text = f'module attributes {{pim.target = "pim:v1"}} {{\n{body}\n}}\n'
        expanded = _opt(text, "-pim-expand-phases")
        folded = _opt(text, "-pim-expand-phases", "-pim-constant-fold")
        assert "arith.constant" not in expanded, name
        assert folded == expanded, f"{name} 的真实展开被折叠改动了"


def test_b_path_fold_keeps_dynamic() -> None:
    """操作数来自输入时不折。"""
    text = _b_func("keep", "%x: tensor<4xf32>, %y: tensor<4xf32>, "
                   "%o: tensor<4x!tt.ptr<f32>>", [
        _eltwise("%c", "%x", "%y", "add"),
        "    tt.store %o, %c : tensor<4x!tt.ptr<f32>>",
    ])
    out = _opt(text, "-pim-expand-phases", "-pim-constant-fold")
    assert "pim.eltwise" in out, out


def test_b_path_cse_merges_duplicate_phase() -> None:
    """展开后结构相同的两个 pim.eltwise 合并成一个。"""
    text = _b_func(
        "merge",
        "%x: tensor<4xf32>, %y: tensor<4xf32>, "
        "%o1: tensor<4x!tt.ptr<f32>>, %o2: tensor<4x!tt.ptr<f32>>",
        [_eltwise("%a", "%x", "%y", "add"),
         _eltwise("%b", "%x", "%y", "add"),
         "    tt.store %o1, %a : tensor<4x!tt.ptr<f32>>",
         "    tt.store %o2, %b : tensor<4x!tt.ptr<f32>>"])
    out = _opt(text, "-pim-expand-phases", "-pim-cse")
    assert _op_count(out, "pim.eltwise") == 1, out


def _b_kernel(name, body):
    """整算子函数：两个 f32 输入，输出是最后一批没人用的张量结果。"""
    return ("module {\n"
            f"  tt.func @{name}(%x: tensor<4xf32>, %y: tensor<4xf32>) {{\n"
            + "\n".join(body) + "\n    tt.return\n  }\n}\n")


def _run_b_arrays(text, passes, inputs, out):
    """跑给定的 B 路 pass，降到 C 编成共享库，按输入在前、输出在后调用一次。"""
    import ctypes
    import subprocess
    import tempfile

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
    from opcompiler_bridge.driver import _translate_to_c
    from opcompiler_bridge.pipeline import with_budget

    pimir = _opt(with_budget(text, DEFAULT_HARDWARE_CONFIG), *passes)
    emitc = _opt(pimir, "-pim-lower-to-emitc", "-convert-func-to-emitc")
    with tempfile.TemporaryDirectory() as tmp:
        c_path, so_path = Path(tmp) / "k.c", Path(tmp) / "k.so"
        c_path.write_text("#include <stdint.h>\n#include <stdlib.h>\n"
                          "#include <math.h>\n#include <stdbool.h>\n"
                          + _translate_to_c(emitc))
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_path),
                        str(c_path), "-lm"], check=True)
        fn = getattr(ctypes.CDLL(str(so_path)), text.split("@")[1].split("(")[0])
        fn(*(a.ctypes.data_as(ctypes.c_void_p) for a in (*inputs, out)))
    return pimir, out


def _run_b_kernel(text, passes, x, y):
    """两个 4 元素 f32 输入、一个 4 元素 f32 输出的整算子函数。"""
    import numpy as np

    return _run_b_arrays(text, passes, (x, y), np.zeros(4, dtype=np.float32))


def _b_inputs():
    import numpy as np

    rng = np.random.default_rng(3)
    return (rng.standard_normal(4).astype(np.float32),
            rng.standard_normal(4).astype(np.float32))


def _b_passes(with_cse: bool):
    from opcompiler_bridge.pipeline import b_path_passes

    return [p for p in b_path_passes() if with_cse or p != "-pim-cse"]


def test_b_path_cse_keeps_operator_output() -> None:
    """整算子函数里与中间值结构相同的输出不能被合并掉，否则 C 写出的是别的值。"""
    import numpy as np

    text = _b_kernel("out_dup", [
        _eltwise("%a", "%x", "%y", "add"),
        _eltwise("%w", "%a", "%x", "mul"),
        _eltwise("%z", "%x", "%y", "add")])
    x, y = _b_inputs()
    _, ref = _run_b_kernel(text, _b_passes(False), x, y)
    _, got = _run_b_kernel(text, _b_passes(True), x, y)
    np.testing.assert_allclose(ref, x + y, rtol=1e-6)
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_b_path_cse_merges_and_matches_numerically() -> None:
    """整算子函数里重复的中间值被合并，降到 C 后与不合并的结果逐位一致。"""
    import numpy as np

    text = _b_kernel("sq", [
        _eltwise("%a", "%x", "%y", "add"),
        _eltwise("%b", "%x", "%y", "add"),
        _eltwise("%z", "%a", "%b", "mul")])
    x, y = _b_inputs()
    plain, ref = _run_b_kernel(text, _b_passes(False), x, y)
    merged, got = _run_b_kernel(text, _b_passes(True), x, y)
    assert _op_count(plain, "pim.eltwise") == 3, plain
    assert _op_count(merged, "pim.eltwise") == 2, merged
    np.testing.assert_allclose(ref, (x + y) * (x + y), rtol=1e-6)
    np.testing.assert_array_equal(got, ref)


def test_b_path_cse_merges_chain_in_one_run() -> None:
    """两级链式重复跑一遍 -pim-cse 就要合并完，只剩一组。"""
    text = _b_func(
        "chain",
        "%x: tensor<4xf32>, %y: tensor<4xf32>, "
        "%o1: tensor<4x!tt.ptr<f32>>, %o2: tensor<4x!tt.ptr<f32>>",
        [_eltwise("%a1", "%x", "%y", "add"),
         _eltwise("%b1", "%a1", "%x", "mul"),
         _eltwise("%a2", "%x", "%y", "add"),
         _eltwise("%b2", "%a2", "%x", "mul"),
         "    tt.store %o1, %b1 : tensor<4x!tt.ptr<f32>>",
         "    tt.store %o2, %b2 : tensor<4x!tt.ptr<f32>>"])
    out = _opt(text, "-pim-cse")
    assert _op_count(out, "pim.eltwise") == 2, out


def test_b_path_cse_keeps_distinct() -> None:
    """属性不同的两个不合并。"""
    text = _b_func(
        "keep",
        "%x: tensor<4xf32>, %y: tensor<4xf32>, "
        "%o1: tensor<4x!tt.ptr<f32>>, %o2: tensor<4x!tt.ptr<f32>>",
        [_eltwise("%a", "%x", "%y", "add"),
         _eltwise("%b", "%x", "%y", "mul"),
         "    tt.store %o1, %a : tensor<4x!tt.ptr<f32>>",
         "    tt.store %o2, %b : tensor<4x!tt.ptr<f32>>"])
    out = _opt(text, "-pim-expand-phases", "-pim-cse")
    assert _op_count(out, "pim.eltwise") == 2, out


def test_b_path_cse_keeps_rope_chains_with_shared_prefix() -> None:
    """同一函数里 Q、K 两个 rope 输入相同、末相不同，默认 B 路跑完契约校验仍通过。

    两条链的前两相相同。只合并前缀会把两条链连成一条，校验报相位重号。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
    from opcompiler_bridge import oplevel_kernel
    from opcompiler_bridge.pipeline import b_path_passes, with_budget

    q_body = oplevel_kernel.rope_kernel("qk", 2, 4, 8)
    k_body = oplevel_kernel.rope_kernel("qk", 2, 4, 8, tail_card_value=3)
    k_rope = k_body.split("\n")[1:3]
    k_rope[0] = k_rope[0].replace("%y =", "%z =")
    lines = q_body.split("\n")
    body = "\n".join(lines[:3] + k_rope + lines[3:])
    text = with_budget("module {\n" + body + "\n}", DEFAULT_HARDWARE_CONFIG)

    without_cse = _opt(text, *_b_passes(with_cse=False))
    out = _opt(text, *b_path_passes())

    assert _op_count(without_cse, "pim.eltwise") == 6, without_cse
    assert _op_count(out, "pim.eltwise") == 6, out


def test_b_path_cse_merges_identical_rope_chains() -> None:
    """两个完全相同的 rope 展开后整条链相同，-pim-cse 合并成一条，契约校验通过。"""
    from opcompiler_bridge import oplevel_kernel

    body = oplevel_kernel.rope_kernel("qq", 2, 4, 8)
    lines = body.split("\n")
    second = [lines[1].replace("%y =", "%z ="), lines[2]]
    store_ty = "tensor<1x2x4x8x!tt.ptr<f16>>"
    head = lines[0].replace(") {", f", %o: {store_ty}, %p: {store_ty}) {{")
    text = ("module {\n" + "\n".join([
        head, lines[1], lines[2], *second,
        f"    tt.store %o, %y : {store_ty}",
        f"    tt.store %p, %z : {store_ty}",
        "    tt.return", "  }"]) + "\n}\n")

    out = _opt(text, "-pim-expand-phases", "-pim-cse",
               "-pim-verify-gml-contract")
    assert _op_count(out, "pim.eltwise") == 3, out


_SM = "tensor<4x16xf32>"
_FP_DP = "datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>"


def test_b_path_cse_merges_softmax_chains_and_matches_numerically() -> None:
    """两条相同的 softmax 相位链整条合并，降到 C 后与不合并的结果逐位一致。"""
    import numpy as np

    softmax = (f"pim.softmax %s {{axis = 1 : i64, unit = #pim.unit<cstl>}}"
               f" : {_SM} -> {_SM}")
    text = ("module {\n"
            f"  tt.func @sm2(%s: {_SM}) {{\n"
            f"    %p = {softmax}\n    %r = {softmax}\n"
            f"    %o = pim.eltwise %p, %r {{kind = #pim.eltwise<add>, {_FP_DP}}}"
            f" : {_SM}, {_SM} -> {_SM}\n"
            "    tt.return\n  }\n}\n")
    x = np.random.default_rng(5).standard_normal((4, 16)).astype(np.float32)
    plain, ref = _run_b_arrays(text, _b_passes(False), (x,),
                               np.zeros((4, 16), np.float32))
    merged, got = _run_b_arrays(text, _b_passes(True), (x,),
                                np.zeros((4, 16), np.float32))

    assert merged.count("pim.phase_spec") * 2 == plain.count("pim.phase_spec"), merged
    e = np.exp(x - x.max(axis=1, keepdims=True))
    # 相位间按 fp16 取整，与闭式解只到 fp16 精度；合并与否逐位比较。
    np.testing.assert_allclose(ref, 2 * e / e.sum(axis=1, keepdims=True), rtol=2e-3)
    np.testing.assert_array_equal(got, ref)


def test_b_path_cse_keeps_rope_prefix_and_matches_numerically() -> None:
    """Q、K 两条 rope 链只有前缀相同，一相都不合并，降到 C 后与不跑 CSE 逐位一致。"""
    import numpy as np

    src, tab = "tensor<1x2x4x8xf16>", "tensor<1x1x4x8xf16>"
    rope = (f"pim.rope %x, %c, %s {{{{{{tail}}numHeads = 2 : i64, unit = #pim.unit<cstl>}}}}"
            f" : {src}, {tab}, {tab} -> {src}")
    text = ("module {\n"
            f"  tt.func @qk(%x: {src}, %c: {tab}, %s: {tab}) {{\n"
            f"    %q = {rope.format(tail='')}\n"
            f"    %k = {rope.format(tail='tailCardValue = 3 : i64, ')}\n"
            f"    %o = pim.eltwise %q, %k {{kind = #pim.eltwise<add>, {_FP_DP}}}"
            f" : {src}, {src} -> {src}\n"
            "    tt.return\n  }\n}\n")
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 2, 4, 8)).astype(np.float16)
    cos = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)
    sin = rng.standard_normal((1, 1, 4, 8)).astype(np.float16)
    plain, ref = _run_b_arrays(text, _b_passes(False), (x, cos, sin),
                               np.zeros((1, 2, 4, 8), np.float16))
    merged, got = _run_b_arrays(text, _b_passes(True), (x, cos, sin),
                                np.zeros((1, 2, 4, 8), np.float16))

    assert merged.count("pim.phase_spec") == plain.count("pim.phase_spec") == 6, merged
    np.testing.assert_array_equal(got, ref)


def test_b_path_dce_drops_dead_scalar_after_real_expansion() -> None:
    """真实 softmax 展开成相位链后，夹在里面的无用户标量被删，相位一条不少。

    无用户标量由 `pim.tasklet_id` 派生，走完整的默认 B 路。相位都是张量
    结果，按设计四·附 B 保留，所以删除前后相位数相同。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
    from opcompiler_bridge import oplevel_kernel
    from opcompiler_bridge.pipeline import b_path_passes, with_budget

    body = oplevel_kernel.softmax_kernel("sm", 4, 16)
    dead = ("    %tid = pim.tasklet_id : i32\n"
            "    %dead = arith.addi %tid, %tid : i32\n")
    body = body.replace("    tt.return", dead + "    tt.return")
    text = with_budget("module {\n" + body + "\n}", DEFAULT_HARDWARE_CONFIG)

    expanded = _opt(text, "-pim-fuse-activation", "-pim-expand-phases")
    out = _opt(text, *b_path_passes())

    assert "arith.addi" in expanded and "pim.tasklet_id" in expanded, expanded
    assert "arith.addi" not in out, out
    assert "pim.tasklet_id" not in out, out
    phases = expanded.count("pim.phase_spec")
    assert phases > 1, expanded
    assert out.count("pim.phase_spec") == phases, out


def test_b_path_dce_keeps_operator_output() -> None:
    """整算子级 IR 里没人用的张量结果是内核输出，不删。"""
    text = _b_func("keep", "%x: tensor<4xf32>, %y: tensor<4xf32>", [
        _eltwise("%c", "%x", "%y", "add"),
    ])
    out = _opt(text, "-pim-expand-phases", "-pim-dce")
    assert "pim.eltwise" in out, out


# ---------------------------------------------------------------- B 路融合与分块

_CONV = """
module {
  tt.func @conv(%x: tensor<1x4x8x8xf16>, %w: tensor<4x4x1x1xf16>,
                %o: tensor<1x4x8x8x!tt.ptr<f16>>) {
    %y = pim.conv %x, %w
       {window = #pim.window<kernel = [1, 1], strides = [1, 1],
                             pads = [0, 0, 0, 0], dilations = [1, 1], group = 1>,
        datapath = #pim.datapath<nmuMode = floating_point,
                                 scaleMode = floating_point>}
       : tensor<1x4x8x8xf16>, tensor<4x4x1x1xf16> -> tensor<1x4x8x8xf16>
    %BODY%
    tt.return
  }
}
"""


def test_b_path_fuses_unlisted_main_op() -> None:
    """带 activation 属性位的主算子后接唯一用户 pim.lut 时，激活折进主算子。

    判定改成按属性位之后，当前带这个属性位的仍是 matmul、conv、eltwise
    三个，与改造前的名单相同。conv 在改造前就在名单里，这里用它验证的是
    按属性位判定的路径，不是一个名单之外的新算子。
    """
    from opcompiler_bridge.pipeline import b_path_passes

    body = ("    %a = pim.lut %y {kind = #pim.activation<relu>} "
            ": tensor<1x4x8x8xf16> -> tensor<1x4x8x8xf16>\n"
            "    tt.store %o, %a : tensor<1x4x8x8x!tt.ptr<f16>>")
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
    from opcompiler_bridge.pipeline import with_budget

    text = with_budget(_CONV.replace("%BODY%", body), DEFAULT_HARDWARE_CONFIG)
    out = _opt(text, *b_path_passes())
    assert "pim.lut" not in out, out
    assert "activation = " in out, out


def test_b_path_fuses_eltwise_with_standalone_lut() -> None:
    """逐元素主算子后接唯一用户 pim.lut 时，激活折进主算子。

    这是真实 B 路在激活没有被图编译器提前折掉时发出的形态：主算子不带
    activation，激活是独立的 pim.lut。融合之后 lut 消失，属性落到主算子上。
    """
    text = """
module {
  tt.func @elt(%a: tensor<1x16xf16>, %b: tensor<1x16xf16>) {
    %y = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point,
                                 scaleMode = floating_point>}
       : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    %z = pim.lut %y {kind = #pim.activation<relu>}
       : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-fuse-activation")
    assert "pim.lut" not in out, out
    assert "activation =" in out, out


def test_b_path_keeps_lut_when_eltwise_has_second_user() -> None:
    """逐元素主算子另有用户时，激活留在原处，不折。"""
    text = """
module {
  tt.func @elt(%a: tensor<1x16xf16>, %b: tensor<1x16xf16>,
               %o: tensor<1x16x!tt.ptr<f16>>) {
    %y = pim.eltwise %a, %b
       {kind = #pim.eltwise<add>,
        datapath = #pim.datapath<nmuMode = floating_point,
                                 scaleMode = floating_point>}
       : tensor<1x16xf16>, tensor<1x16xf16> -> tensor<1x16xf16>
    %z = pim.lut %y {kind = #pim.activation<relu>}
       : tensor<1x16xf16> -> tensor<1x16xf16>
    tt.store %o, %y : tensor<1x16x!tt.ptr<f16>>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-fuse-activation")
    assert "pim.lut" in out, out


def test_b_path_fuse_blocked_by_second_user() -> None:
    """pim.lut 的输入另有用户时不折。"""
    body = ("    %a = pim.lut %y {kind = #pim.activation<relu>} "
            ": tensor<1x4x8x8xf16> -> tensor<1x4x8x8xf16>\n"
            "    tt.store %o, %y : tensor<1x4x8x8x!tt.ptr<f16>>")
    out = _opt(_CONV.replace("%BODY%", body), "-pim-fuse-activation")
    assert "pim.lut" in out, out


def _bpath_matmul_text(wram: int, shape=((16, 128), (128, 8)), group=64):
    """真实 B 路矩阵乘的整算子 IR，按给定 WRAM 预算带上模块属性。"""
    import dataclasses

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_oplevel_mlir
    from opcompiler_bridge.pipeline import with_budget

    hw = dataclasses.replace(DEFAULT_HARDWARE_CONFIG, num_dpus=1,
                             wram_bytes_per_dpu=wram)
    req = OpCompileRequest(op="matmul", arg_shapes=list(shape), hardware=hw,
                           dtype="int8", group_size=group)
    return with_budget(_make_oplevel_mlir(req), hw)


def test_b_path_default_pipeline_tiles_oversized_matmul() -> None:
    """B 路默认管线含分块：超预算的矩阵乘按形状选出 tile，写在算子上。

    16x128 乘 128x8、组宽 64、预算 1024：操作数 1 字节，两个 f32 累加器
    各 4 字节。按评估模型手算，能整除各维且组宽可被整除的候选里元素最多
    的是 M=4、N=8、K=64，工作集 4*64 + 8*64 + 4*8*8 = 1024 字节。
    """
    from opcompiler_bridge.pipeline import b_path_passes

    passes = b_path_passes()
    assert "-pim-tile-to-budget" in passes, passes
    out = _opt(_bpath_matmul_text(1024), *passes)
    assert "tile = array<i64: 4, 8, 64>" in out, out
    assert '"pim.tile-m" = 4' in out and '"pim.tile-k" = 64' in out, out


_DP = "{datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}"


def _two_matmul_module(wram: int, first: tuple, second: tuple) -> str:
    """同一模块两个函数各一个矩阵乘，参数是 (m, k, n, dtype)。"""
    funcs = []
    for name, (m, k, n, dt) in (("m1", first), ("m2", second)):
        a, w, y = f"tensor<{m}x{k}x{dt}>", f"tensor<{k}x{n}x{dt}>", f"tensor<{m}x{n}x{dt}>"
        funcs.append(f"  tt.func @{name}(%a: {a}, %w: {w}) {{\n"
                     f"    %y = pim.matmul %a, %w {_DP} : {a}, {w} -> {y}\n"
                     "    tt.return\n  }\n")
    return ('module attributes {"pim.wram-bytes" = ' f"{wram}"
            ' : i64, "pim.mram-bytes" = 1048576 : i64, "pim.dma-align" = 8 : i64} {\n'
            + "".join(funcs) + "}\n")


@pytest.mark.parametrize("wram, first, second, tiles", [
    # tile 不同：2x2x8 与 1x2x16。
    (64, (4, 8, 4, "i8"), (4, 16, 4, "i8"), [("2", "2", "8"), ("1", "2", "16")]),
    # tile 相同、字节数不同：i8 352 字节，f16 448 字节。
    (512, (8, 8, 8, "i8"), (8, 8, 8, "f16"), [("4", "8", "8"), ("4", "8", "8")]),
    # 前一个在预算内不切，后一个切成 2x2x8。
    (64, (1, 8, 1, "i8"), (4, 8, 4, "i8"), [("2", "2", "8")]),
])
def test_b_path_module_head_never_describes_a_different_matmul(
        wram, first, second, tiles) -> None:
    """模块头一组 tile 属性描述不了各不相同的矩阵乘时不写。

    各矩阵乘照常写自己的 tile；布局回传与成本模型按模块头读，读不到就是
    没结论，不会把第一个算子的 tile 或字节数套到第二个上。
    """
    import re

    from contracts.ir_payloads import layout_feedback_of_module
    from genesim_bridge.ir_cost import analyze_ir

    out = _opt(_two_matmul_module(wram, first, second), "-pim-tile-to-budget")
    assert re.findall(r"tile = array<i64: (\d+), (\d+), (\d+)>", out) == tiles, out
    assert "pim.tile-" not in out.split("\n", 1)[0], out
    assert layout_feedback_of_module(out).tile_shape is None
    cost = analyze_ir(out, "m1", (1, 1, 1), {}, ir_level="oplevel")
    assert cost.tile_m is None and cost.tile_wram_bytes is None, cost


def test_b_path_layout_feedback_reports_tile_wram_bytes() -> None:
    """B 路切开的矩阵乘只写 `pim.tile-wram-bytes`，布局回传要读到它。

    16x128 乘 128x8、组宽 64、预算 1024 选 4x8x64，工作集 1024 字节。
    """
    from contracts.ir_payloads import layout_feedback_of_module
    from opcompiler_bridge.pipeline import b_path_passes

    fb = layout_feedback_of_module(_opt(_bpath_matmul_text(1024), *b_path_passes()))
    assert fb.tile_shape == (4, 8), fb
    assert fb.wram_bytes_used == 1024 and fb.wram_bytes_budget == 1024, fb
    assert not fb.over_wram_budget


def test_b_path_keeps_matmul_within_budget() -> None:
    """预算内的矩阵乘不写 tile，产物与不分块一致。"""
    from opcompiler_bridge.pipeline import b_path_passes

    out = _opt(_bpath_matmul_text(65536), *b_path_passes())
    assert "tile = array" not in out and "pim.tile-" not in out, out


def test_gml_phase_pipeline_has_no_tiling() -> None:
    """GML 相位模板按整块算子编码，与图层 GML 路同一个例外，不分块。"""
    from opcompiler_bridge.phase_source import PASS_PIPELINE

    assert "-pim-tile-to-budget" not in PASS_PIPELINE, PASS_PIPELINE


def test_generic_tile_rewrites_loop() -> None:
    """只验证 pass 本身：超预算的循环按预算改写步长，并写出 tile 属性。

    B 路默认管线的分块对象是整算子矩阵乘，见
    `test_b_path_default_pipeline_tiles_oversized_matmul`。
    """
    text = """
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @loop(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}
"""
    out = _opt(text, "-pim-expand-phases", "-pim-tile-to-budget")
    assert "step %c8" in out, out
    assert '"pim.tile-m" = 8' in out, out



def test_generic_tile_keeps_loop_that_uses_induction_var() -> None:
    """只验证 pass 本身：循环体用了归纳变量时不改步长，否则取值的位置会错。"""
    text = """
module attributes {"pim.wram-bytes" = 32 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @uses_iv(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      %idx = arith.index_cast %i : index to i32
      %v = tt.splat %idx : i32 -> tensor<16xi32>
      %f = arith.sitofp %v : tensor<16xi32> to tensor<16xf32>
      %s = arith.addf %acc, %f : tensor<16xf32>
      scf.yield %s : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}
"""
    out = _opt(text, "-pim-expand-phases", "-pim-tile-to-budget")
    assert "step %c1" in out, out


def test_generic_tile_keeps_within_budget() -> None:
    """只验证 pass 本身：循环跨度已在预算内时不改。"""
    text = """
module attributes {"pim.wram-bytes" = 65536 : i64, "pim.mram-bytes" = 1048576 : i64,
                   "pim.dma-align" = 8 : i64} {
  tt.func @loop(%x: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %x) -> tensor<16xf32> {
      scf.yield %acc : tensor<16xf32>
    }
    tt.return %r : tensor<16xf32>
  }
}
"""
    out = _opt(text, "-pim-expand-phases", "-pim-tile-to-budget")
    assert "step %c1" in out, out


def test_a_path_cse_respects_result_type() -> None:
    """结果类型不同的同结构运算不合并，否则用户拿到类型对不上的值。"""
    text = """
module {
  tt.func public @typed(%x: tensor<4xi32>, %y: tensor<4xi32>,
                        %o1: tensor<4x!tt.ptr<i32>>,
                        %o2: tensor<4x!tt.ptr<i32>>) {
    %a = arith.addi %x, %y : tensor<4xi32>
    %b = arith.addi %x, %y : tensor<4xi32>
    %e = tt.expand_dims %a {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %z = arith.addi %e, %e : tensor<1x4xi32>
    tt.store %o1, %b : tensor<4x!tt.ptr<i32>>
    %flat = tt.reshape %z : tensor<1x4xi32> -> tensor<4xi32>
    tt.store %o2, %flat : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
"""
    out = _opt(text, "-pim-cse")
    # 两个 tensor<4xi32> 的 addi 合并，tensor<1x4xi32> 的那个类型不同必须留下。
    assert out.count("arith.addi") == 2, out



def test_a_path_tiling_rewrites_oversized_kernel() -> None:
    """超预算的 linear kernel 被改写出循环，预算内的只记 tile 不改写。"""
    import dataclasses
    import re

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    hw = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1,
        mram_bytes_per_dpu=1 << 20)

    def run(shape):
        req = OpCompileRequest(
            op="linear", arg_shapes=[shape, shape], hardware=hw,
            dtype="float16")
        return _opt(_make_ttir(req), *a_path_passes(hw, target="pim:v1"))

    small = run((16, 16))
    assert "scf.for" not in small, "预算内的 kernel 不该改写循环"
    assert re.search(r'"pim\.tile-m" = \d+', small), small

    big = run((128, 128))
    assert big.count("scf.for") == 2, "超预算的 kernel 应改写出两层循环"
    assert re.search(r'"pim\.tile-k" = \d+', big), big


def test_a_path_hoists_loop_invariants() -> None:
    """真实 linear kernel 的地址计算不依赖内层归纳变量，应被提到循环外。

    转换后的 kernel 在最内层循环里重复算同一批布局转换与偏移，
    每轮结果都一样。外提之后循环内只留下随迭代变化的加载与点积。
    """
    import dataclasses

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    hw = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1,
        mram_bytes_per_dpu=1 << 20)
    req = OpCompileRequest(
        op="linear", arg_shapes=[(64, 64), (64, 64)], hardware=hw,
        dtype="float16")
    out = _opt(_make_ttir(req), *a_path_passes(hw, target="pim:v1"))

    # 最内层循环体：最后一个 scf.for 到它配对的 scf.yield 之间。
    start = out.rfind("scf.for")
    body = out[start:out.find("scf.yield", start)]
    # 依赖本轮加载结果的那一次布局转换必须留在循环里，其余的地址与
    # 掩码计算都不随迭代变化，应当全部外提。
    assert body.count("pim.convert_layout") == 1, body
    assert body.count("arith.muli") == 0, "偏移的乘法还留在循环里"


def test_a_path_pipeline_on_real_kernel() -> None:
    """真实 linear kernel 走完整 A 路：分块改写出循环且 IR 仍然合法。"""
    import dataclasses

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import _make_ttir
    from opcompiler_bridge.pipeline import a_path_passes

    hw = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1,
        mram_bytes_per_dpu=1 << 20)
    req = OpCompileRequest(
        op="linear", arg_shapes=[(64, 64), (64, 64)], hardware=hw,
        dtype="float16")
    out = _opt(_make_ttir(req), *a_path_passes(hw, target="pim:v1"))
    assert "scf.for" in out, "超预算的 kernel 没有被改写出循环"
    assert '"pim.tile-m"' in out, out


def test_genesim_a_path_matches_pipeline() -> None:
    """genesim 的 A 路与 pipeline.a_path_passes 同一份顺序。

    那边走进程内绑定，不调 pipeline.py，顺序是手写的，这里钉住不漂移。
    """
    import inspect

    import genesim_bridge.flagtree_driver as drv
    from opcompiler_bridge.pipeline import a_path_passes

    body = inspect.getsource(drv.lower_ttir_to_pimir)
    names = {
        "-canonicalize": "add_canonicalizer",
        "-cse": "add_cse",
        "-loop-invariant-code-motion": "add_licm",
        "-pim-constant-fold": "add_constant_fold",
        "-pim-cse": "add_pim_cse",
        "-pim-fuse-activation": "add_fuse_activation",
        "-pim-dce": "add_pim_dce",
        "-pim-tile-to-budget": "add_tile_to_budget",
        "-pim-explicit-dma": "add_explicit_dma",
    }
    positions = [body.index(f"{names[p]}(") for p in a_path_passes(
        type("H", (), {"num_dpus": 1, "num_tasklets": 1,
                       "wram_bytes_per_dpu": 1, "mram_bytes_per_dpu": 1,
                       "dma_align": 1})(), target="pim:v1")
                 if p in names]
    assert positions == sorted(positions), "genesim 的 A 路顺序与 pipeline 不一致"


def test_sidecar_matches_pipeline() -> None:
    """FlagTree 的 pim_sidecar 与 a_path_passes 同一份顺序。

    那个文件不引本仓，顺序是内联的，这里钉住它不漂移。
    """
    from genesim_bridge.paths import flagtree_source
    from opcompiler_bridge.pipeline import a_path_passes

    root = flagtree_source(required=False)
    if root is None:
        import pytest
        pytest.skip("FlagTree 源码树不在位")
    body = (root / "python/triton/backends/pim_sidecar.py").read_text()
    body = body[body.index("def make_pimir"):body.index("def emit_pim_ir")]
    names = {
        "-canonicalize": "add_canonicalizer",
        "-cse": "add_cse",
        "-loop-invariant-code-motion": "add_licm",
        "-pim-constant-fold": "add_constant_fold",
        "-pim-cse": "add_pim_cse",
        "-pim-fuse-activation": "add_fuse_activation",
        "-pim-dce": "add_pim_dce",
        "-pim-tile-to-budget": "add_tile_to_budget",
        "-pim-explicit-dma": "add_explicit_dma",
    }
    positions = [body.index(f"{names[p]}(") for p in a_path_passes(
        type("H", (), {"num_dpus": 1, "num_tasklets": 1,
                       "wram_bytes_per_dpu": 1, "mram_bytes_per_dpu": 1,
                       "dma_align": 1})(), target="pim:v1")
                 if p in names]
    assert positions == sorted(positions), "pim_sidecar 的顺序与 pipeline 不一致"


def test_generic_tiling_runs_without_a_dot() -> None:
    """没有 tt.dot 的 kernel 在 genesim 与 sidecar 两个入口也要跑分块。

    通用循环分块存在的意义就是处理没有点积的循环。两个入口用文本里有没有
    `tt.dot` 决定是否加入分块，纯循环 kernel 因此被跳过，而且注释里出现
    这几个字也会被算进去。入口必须无条件加入，交给 pass 自己判断有没有
    可切的循环。
    """
    import inspect

    import genesim_bridge.flagtree_driver as drv
    from genesim_bridge.paths import flagtree_source

    def guarded(body: str) -> bool:
        # 分块调用落在一个以 "tt.dot" 为条件的分支里，就是被文本匹配挡住了。
        idx = body.index("add_tile_to_budget(")
        window = body[max(0, idx - 400):idx]
        return "tt.dot" in window

    assert not guarded(inspect.getsource(drv.lower_ttir_to_pimir)),         "genesim 的 A 路仍按文本里有没有 tt.dot 决定是否分块"

    root = flagtree_source(required=False)
    if root is None:
        import pytest
        pytest.skip("FlagTree 源码树不在位")
    side = (root / "python/triton/backends/pim_sidecar.py").read_text()
    side = side[side.index("def make_pimir"):side.index("def emit_pim_ir")]
    assert not guarded(side), "pim_sidecar 仍按文本里有没有 tt.dot 决定是否分块"


@pytest.mark.parametrize("kernel", ["mul_generic_nd_kernel", "isin_by_comparation_kernel"])
def test_real_dotless_kernel_passes_a_path_untiled(kernel) -> None:
    """真实捕获的无点积内核走完 A 路不出错，也不写 tile 属性。

    2026-10-23 实测 FlagGems 缓存里 179 个无点积内核，177 个没有 scf.for，
    剩下 2 个的循环边界是运行期值。通用循环分块在真实内核上没有作用对象，
    只在合成 IR 上验证；这里钉住它在真实内核上保持原样、不崩。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
    from opcompiler_bridge.pipeline import a_path_passes

    ttir = (Path(__file__).parent / "data/real_kernels" / f"{kernel}.ttir").read_text()
    # 预算取默认值：压小后 DMA 缓冲检查先报错，那不是分块的口径。
    out = _opt(ttir, *a_path_passes(DEFAULT_HARDWARE_CONFIG, target="pim:v1"))
    assert "pim.tile-m" not in out, out
    assert out.count("scf.for") == ttir.count("scf.for"), "循环个数变了"


def test_flagtree_pim_lit_suite_passes() -> None:
    """用构建目录里的 lit 跑 `test/Dialect/TritonPIM`，全部用例必须通过。

    逐条把 RUN 行交给 shell 执行没有 pipefail，管道左侧 triton-opt 报错会被吞掉；
    lit 默认开 pipefail，这里以它的结果为准。
    """
    import os
    import subprocess

    from genesim_bridge.paths import flagtree_prefix

    prefix = flagtree_prefix()
    build = prefix / "build" / "flagtree-cmake"
    python = prefix / "python-3.10.20" / "bin" / "python3"
    suite = build / "test" / "Dialect" / "TritonPIM"
    if not (python.is_file() and suite.is_dir()):
        pytest.skip("FlagTree 构建目录或 lit 不在位")
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(prefix / "llvm-7d5de303" / "bin"), str(build / "bin"), env["PATH"]])
    proc = subprocess.run(
        [str(python), "-c",
         "import sys; from lit.main import main; sys.argv[0] = 'lit'; main()",
         "-q", "-j1", str(suite)],
        capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
