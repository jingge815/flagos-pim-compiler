"""运行时内核表里，哪些算子真的走了算子编译器编出来的 `.so`。

这不是"表里有名字"的自证：每个用例都给 `compile_op` 打桩，再跑一次那个
内核函数，只有真的调了编译器的才算数。方案 4.8 要求"每个名字一对
numpy 镜像 + 可选编译内核"，所以这里钉住的是**当前实际**:哪些有、哪些没有。
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime import kernels


def _run(kernel, args, shape, calls, *, dtype="float16"):
    """跑一次内核，`compile_op` 被调用就记进 `calls`。"""
    out = np.zeros(shape, dtype=np.float16)
    written = {}

    def fake_compile(request, **kw):
        calls.append(request.op)
        fn = mock.Mock()
        # 返回一个能当 .so 载入的结果对象。
        result = mock.Mock(so_path=__file__, argtypes=[], by_value=[])
        return result

    with mock.patch("runtime.kernels._read_tensor_args", return_value=args), \
         mock.patch("runtime.kernels._write_result",
                    side_effect=lambda hal, d, cmd, r: written.setdefault("out", r)), \
         mock.patch("opcompiler_bridge.driver.compile_op", side_effect=fake_compile), \
         mock.patch("opcompiler_bridge.driver.load_kernel",
                    return_value=lambda *a: None):
        kernel(None, 0, mock.Mock(payload={
            "dtype": dtype, "arg_kinds": ["tensor"] * len(args),
            "arg_shapes": [tuple(a.shape) if hasattr(a, "shape") else ()
                           for a in args], "arg_dtypes": None,
            "out_shape": shape}))
    return written.get("out")


def test_softmax_reaches_the_compiler() -> None:
    calls: list[str] = []
    x = np.zeros((2, 8), dtype=np.float16)
    _run(kernels.softmax_kernel, [x], (2, 8), calls)
    assert "softmax" in calls


def test_mask_reaches_the_compiler() -> None:
    calls: list[str] = []
    scores = np.zeros((4, 8), dtype=np.float16)
    mask = np.zeros((4, 8), dtype=np.bool_)
    _run(kernels.masked_fill_kernel, [scores, mask, -65504.0], (4, 8), calls)
    assert "mask" in calls


def test_alias_is_an_identity() -> None:
    """`aten.alias` 是视图，原样传回，不编译动态量化。

    以前这里走 `pim.dynamic_quant`，把 fp16 量化成 int8 再按 fp16 的长度写回，
    读出来是 fp16 最大值。量化是权重侧的事。
    """
    calls: list[str] = []
    x = np.arange(64, dtype=np.float16).reshape(1, 64)
    out = _run(kernels.dynamic_quant_kernel, [x], (1, 64), calls)
    assert calls == []
    assert np.array_equal(out, x)


def test_gather_reaches_the_compiler() -> None:
    calls: list[str] = []
    table = np.zeros((8, 4), dtype=np.float16)
    ids = np.zeros((2,), dtype=np.int64)
    _run(kernels.gather_kernel, [table, ids], (2, 4), calls)
    assert "gather" in calls


def test_eltwise_reaches_the_compiler() -> None:
    """同形状的加/乘/减走编译内核——标量与广播没有对应的设备循环。"""
    calls: list[str] = []
    x = np.zeros((2, 8), dtype=np.float16)
    y = np.zeros((2, 8), dtype=np.float16)
    _run(kernels.add_kernel, [x, y], (2, 8), calls)
    assert "eltwise" in calls


def test_attention_matmul_reaches_the_compiler() -> None:
    """注意力的矩阵乘真的编出 `.so`：fp16×fp16，与 `x @ w` 逐元素一致。

    这条上一轮正相反——那时 `pim.matmul` 只有 i8×i8 一种形态，而这条 aten
    喂的是 fp16 激活，于是它被写成「钉住现状」的断言。矩阵单元本身不区分
    元素类型，EmitC 按缓冲区的类型取值，所以缺的只是发射侧那一种形态。
    """
    calls: list[str] = []
    x = np.zeros((4, 8), dtype=np.float16)
    w = np.zeros((8, 4), dtype=np.float16)
    _run(kernels.matmul_kernel, [x, w], (4, 4), calls)
    assert "matmul" in calls

    # 不只是"调了编译器"：编出来的东西必须和镜像逐元素一致。
    from tests.test_opcompiler_ops import _call
    from opcompiler_bridge.driver import compile_op, load_kernel
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest

    rng = np.random.default_rng(11)
    a = rng.standard_normal((4, 8)).astype(np.float16)
    b = rng.standard_normal((8, 4)).astype(np.float16)
    fn = load_kernel(compile_op(OpCompileRequest(
        op="matmul", arg_shapes=[(4, 8), (8, 4)],
        hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16"), force=True))
    got = _call(fn, a, b, out_shape=(4, 4), out_dtype=np.float16)
    ref = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)
    assert np.array_equal(got, ref), (
        f"{int((got != ref).sum())} 个元素不一致，编译内核与镜像分道")


def test_convert_reaches_the_compiler() -> None:
    """`aten.to.dtype` 走 `pim.convert`，目标类型按图的实参走。"""
    calls: list[str] = []
    x = np.zeros((2, 8), dtype=np.float16)
    _run(kernels.convert_kernel, [x, "torch.float32"], (2, 8), calls,
         dtype="float32")

    # 数值也要对：f16→i8 与 f16→f32 是两条不同的编译产物。
    got = kernels.convert(np.array([1.5, -2.5], dtype=np.float16), np.int8)
    assert np.array_equal(got, np.array([2, -2], dtype=np.int8)), got
    got = kernels.convert(np.array([1.5, -2.5], dtype=np.float16), np.float32)
    assert np.array_equal(got, np.array([1.5, -2.5], dtype=np.float32)), got


def test_attention_mask_reaches_the_compiler() -> None:
    """注意力那笔加性掩码走 `pim.mask`，不是在这一处现加。"""
    scores = np.zeros((8,), dtype=np.float16)
    offset = np.zeros((8,), dtype=np.float16)
    kernels._COMPILED_KERNEL_CACHE.clear()
    try:
        with mock.patch("opcompiler_bridge.driver.compile_op") as spy:
            spy.return_value = mock.Mock(so_path=__file__, argtypes=[],
                                         by_value=[])
            with mock.patch("opcompiler_bridge.driver.load_kernel",
                            return_value=lambda *a: None):
                kernels.add_mask(scores, offset)
        assert spy.call_args[0][0].op == "mask"
    finally:
        kernels._COMPILED_KERNEL_CACHE.clear()


def test_the_compiled_set_is_what_the_docs_claim() -> None:
    """运行时真正调 `compile_op` 的那一组，与文档写的一致。

    少一个 = 那个算子只在 numpy 里跑，读者会以为它进了设备编译器的通路。
    """
    import re
    from pathlib import Path as P

    src = P(__file__).parent.parent / "runtime" / "kernels.py"
    text = src.read_text(encoding="utf-8")
    called = set(re.findall(r'op="([a-z_]+)"', text))
    called |= set(re.findall(r'_compiled_view\("([a-z_]+)"', text))
    assert called == {
        "linear", "softmax", "mask", "gather", "eltwise",
        "matmul", "convert", "transpose", "reshape", "concat",
        "normalize", "rope", "lut", "kv_cache", "split_heads",
    }, f"实际调用编译器的算子集合变了：{sorted(called)}"


def test_normalize_reaches_the_compiler() -> None:
    calls: list[str] = []
    x = np.zeros((4, 16), dtype=np.float16)
    gamma = np.ones((16,), dtype=np.float16)
    _run(kernels.rmsnorm_kernel, [x, gamma], (4, 16), calls)
    assert "normalize" in calls


def test_rope_reaches_the_compiler() -> None:
    calls: list[str] = []
    x = np.zeros((1, 2, 4, 8), dtype=np.float16)
    cos = np.zeros((1, 1, 4, 8), dtype=np.float16)
    sin = np.zeros((1, 1, 4, 8), dtype=np.float16)
    _run(kernels.rope_kernel, [x, cos, sin], (1, 2, 4, 8), calls)
    assert "rope" in calls


def test_silu_reaches_the_compiler() -> None:
    """silu 走 `pim.lut`，求值用闭式。"""
    calls: list[str] = []
    x = np.zeros((4, 16), dtype=np.float16)
    _run(kernels.silu_kernel, [x], (4, 16), calls)
    assert "lut" in calls


def test_split_heads_reaches_the_compiler() -> None:
    calls: list[str] = []
    x = np.zeros((4, 32), dtype=np.float16)
    _run(kernels.split_heads_kernel, [x, 4, 1], (4, 4, 8), calls)
    assert "split_heads" in calls
