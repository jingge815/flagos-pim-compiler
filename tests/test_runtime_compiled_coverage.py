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
        "reduce",
    }, f"实际调用编译器的算子集合变了：{sorted(called)}"


def test_neg_and_slice_fallbacks_are_recorded() -> None:
    """退回计数要覆盖到每个算子。

    neg 与步长为 1 的 slice 都有编译形态，记命中。不记的话，
    「兜底次数为 0」分不清它们是走了编译还是没被看见。
    """
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.exec_plan import Access, Command

    kernels.reset_route_counts()
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
    x = np.ones((2, 4), dtype=np.float16)
    backend.write_local(0, 0, x)

    def cmd(name, shapes, kinds, out_shape, nbytes):
        return Command(
            id=0, op="launch", dpu_id=0,
            payload={"kernel": name, "node": "n", "arg_kinds": kinds,
                     "arg_shapes": shapes, "dtype": "float16",
                     "out_shape": out_shape},
            reads=[Access(("dpu", 0), 0, x.nbytes)],
            writes=[Access(("dpu", 0), 64, nbytes)], waits=[], num_tasklets=1)

    kernels.neg_kernel(backend, 0, cmd("neg", [x.shape], ["tensor"], x.shape, x.nbytes))
    kernels.slice_kernel(backend, 0, cmd(
        "slice", [x.shape, None, None, None], ["tensor", 1, 0, 2], (2, 2), 8))
    counts = kernels.route_counts()
    assert counts.get(("eltwise", "hit"), 0) == 1, counts
    assert counts.get(("reshape", "hit"), 0) == 1, counts
    assert counts.get(("neg", "fallback"), 0) == 0, counts
    assert counts.get(("slice", "fallback"), 0) == 0, counts

    # 半区切分是 RoPE 的形态，数值要与 numpy 一致。
    src = np.arange(8, dtype=np.float16).reshape(2, 4)
    backend.write_local(0, 0, src)
    kernels.slice_kernel(backend, 0, cmd(
        "slice", [src.shape, None, None, None], ["tensor", 1, 0, 2], (2, 2), 8))
    got = backend.read_local(0, 64, (2, 2), np.float16)
    assert np.array_equal(got, src[:, :2]), got


def test_toolchain_miss_is_recorded(monkeypatch) -> None:
    """工具链缺失时退回镜像，必须计入 fallback。

    这些内核编译失败后直接走镜像，不调用 record_route。端到端的
    「退回次数为 0」因此看不见它们。
    """
    from opcompiler_bridge.driver import ToolchainUnavailable

    kernels.reset_route_counts()
    kernels._COMPILED_KERNEL_CACHE.clear()

    def miss(*args, **kwargs):
        raise ToolchainUnavailable("测试：工具链不在位")

    monkeypatch.setattr("opcompiler_bridge.driver.compile_op", miss)

    x = np.ones((2, 4), dtype=np.float16)
    gamma = np.ones((4,), dtype=np.float16)
    cos = np.ones((1, 1, 2, 4), dtype=np.float16)
    scores = np.ones((2, 4), dtype=np.float16)
    offset = np.zeros((2, 4), dtype=np.float16)
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.exec_plan import Access, Command

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
    backend.write_local(0, 0, x)
    backend.write_local(0, 64, x)

    def cmd(kinds, shapes, out_shape, nread):
        return Command(
            id=0, op="launch", dpu_id=0,
            payload={"kernel": "n", "node": "n", "arg_kinds": kinds,
                     "arg_shapes": shapes, "dtype": "float16",
                     "out_shape": out_shape},
            reads=[Access(("dpu", 0), 0, x.nbytes) for _ in range(nread)],
            writes=[Access(("dpu", 0), 256, x.nbytes)], waits=[], num_tasklets=1)

    kernels._silu(x)
    kernels._normalize(x, gamma)
    kernels._rope(x.reshape(1, 1, 2, 4), cos, cos)
    kernels.add_mask(scores, offset)
    kernels.convert(x, np.float32)
    kernels.transpose_kernel(backend, 0, cmd(
        ["tensor", (1, 0)], [x.shape, None], (4, 2), 1))
    kernels.reshape_kernel(backend, 0, cmd(
        ["tensor", (8,)], [x.shape, None], (8,), 1))
    kernels.concat_kernel(backend, 0, cmd(
        ["tensor", "tensor", 0], [x.shape, x.shape, None], (4, 4), 2))
    kernels.split_heads_kernel(backend, 0, cmd(
        ["tensor", 2, 1], [x.shape, None, None], (2, 2, 2), 1))
    counts = kernels.route_counts()
    expected = {
        "lut", "normalize", "rope", "mask", "convert",
        "transpose", "reshape", "concat", "split_heads",
    }
    missing = sorted(
        name for name in expected if counts.get((name, "fallback"), 0) < 1)
    kernels._COMPILED_KERNEL_CACHE.clear()
    assert not missing, f"这些退回没有记账：{missing}，实际 {counts}"


def _cmd_of(args, out_shape):
    """造一条只给内核读 payload 的命令。"""
    return mock.Mock(payload={
        "dtype": "float16",
        "arg_kinds": ["tensor" if hasattr(a, "shape") else a for a in args],
        "arg_shapes": [tuple(a.shape) if hasattr(a, "shape") else None
                       for a in args],
        "arg_dtypes": None,
        "out_shape": out_shape,
    })


def test_strided_slice_falls_back() -> None:
    """步长不是 1 的切片没有编译形态，退回镜像并计入退回。"""
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.exec_plan import Access, Command

    kernels.reset_route_counts()
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
    x = np.arange(8, dtype=np.float16).reshape(2, 4)
    backend.write_local(0, 0, x)
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": "slice", "node": "n",
                 "arg_kinds": ["tensor", 1, 0, 4, 2],
                 "arg_shapes": [x.shape, None, None, None, None],
                 "dtype": "float16", "out_shape": (2, 2)},
        reads=[Access(("dpu", 0), 0, x.nbytes)],
        writes=[Access(("dpu", 0), 64, 8)], waits=[], num_tasklets=1)
    kernels.slice_kernel(backend, 0, cmd)
    got = backend.read_local(0, 64, (2, 2), np.float16)
    assert np.array_equal(got, x[:, ::2]), got
    assert kernels.route_counts().get(("slice", "fallback"), 0) == 1


def test_unsqueeze_reaches_the_compiler() -> None:
    """`aten.unsqueeze` 要走 `pim.reshape`，不能只在 numpy 里插轴。"""
    calls: list[str] = []
    x = np.ones((4, 16), dtype=np.float16)
    _run(kernels.unsqueeze_kernel, [x, 1], (4, 1, 16), calls)
    assert "reshape" in calls


def test_mean_reaches_the_compiler() -> None:
    """`aten.mean.dim` 要走编译内核，不能只在 numpy 里求均值。"""
    calls: list[str] = []
    x = np.ones((4, 16), dtype=np.float16)
    _run(kernels.mean_dim_kernel, [x, [-1], True], (4, 1), calls)
    assert "reduce" in calls


def test_pow_reaches_the_compiler() -> None:
    """`aten.pow` 指数为 2 时要走 `pim.eltwise` 的乘，不能只在 numpy 里算。"""
    calls: list[str] = []
    x = np.ones((4, 16), dtype=np.float16)
    _run(kernels.pow_kernel, [x, 2.0], (4, 16), calls)
    assert "eltwise" in calls


def test_rsqrt_reaches_the_compiler() -> None:
    """`aten.rsqrt` 要走 `pim.lut` 的 rsqrt，不能只在 numpy 里算。"""
    calls: list[str] = []
    x = np.ones((4, 16), dtype=np.float16)
    _run(kernels.rsqrt_kernel, [x], (4, 16), calls)
    assert "lut" in calls


def test_fp32_rmsnorm_chain_reaches_the_compiler() -> None:
    """RMSNorm 在 fp32 上做平方、求均值、求倒数平方根，三条都要进编译内核。

    图里是 `to.fp32 → pow → mean → rsqrt`，输入不是 fp16。只覆盖 fp16 时，
    这条链整段退回 numpy，端到端的退回计数不为 0。
    """
    calls: list[str] = []
    x = np.ones((4, 16), dtype=np.float32)
    kernels._COMPILED_KERNEL_CACHE.clear()
    _run(kernels.pow_kernel, [x, 2.0], (4, 16), calls, dtype="float32")
    _run(kernels.mean_dim_kernel, [x, [-1], True], (4, 1), calls, dtype="float32")
    _run(kernels.rsqrt_kernel, [x], (4, 16), calls, dtype="float32")
    assert calls.count("eltwise") == 1, calls
    assert calls.count("reduce") == 1, calls
    assert calls.count("lut") == 1, calls


def test_transpose_int_reaches_the_compiler() -> None:
    """`aten.transpose.int` 要走 `pim.transpose`，不能只换两轴就退回镜像。

    轴序由两个轴号拼出来，与 `permute` 走同一条编译路径。
    """
    calls: list[str] = []
    x = np.zeros((2, 4, 8), dtype=np.float16)
    _run(kernels.transpose_int_kernel, [x, 1, 2], (2, 8, 4), calls)
    assert "transpose" in calls


def test_normalize_reaches_the_compiler() -> None:
    calls: list[str] = []
    x = np.zeros((4, 16), dtype=np.float16)
    gamma = np.ones((16,), dtype=np.float16)
    _run(kernels.rmsnorm_kernel, [x, gamma], (4, 16), calls)
    assert "normalize" in calls


def test_normalize_matches_numpy() -> None:
    """运行时入口编出来的 normalize 要与镜像逐元素一致。

    `_compiled_normalize` 传的形状与算子级契约不一致时，这里会直接抛
    `ValueError`，mock 掉 `compile_op` 的用例看不见。
    """
    from runtime.kernels import _normalize
    from runtime import kernels_pim

    x = np.zeros((4, 16), dtype=np.float16)
    gamma = np.ones((16,), dtype=np.float16)
    got = _normalize(x, gamma)
    assert np.array_equal(got, kernels_pim.normalize(x, gamma))


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



def _kernels_reaching_compile() -> set[str]:
    """函数体里记了命中的 aten 内核名。

    `record_route(op, "hit")` 只出现在调用编译内核之前，所以记了命中就
    说明这个内核有编译路径。按函数调用关系扫会把「提到但走不到」的算子
    也算进去，豁免集合因此失效。
    """
    import inspect

    import runtime.kernels as km

    # 闭包包出来的内核（_eltwise、_unary）源码在工厂函数里，不在闭包上。
    factories = {}
    for fname, f in vars(km).items():
        if callable(f) and getattr(f, "__module__", "") == km.__name__:
            try:
                factories[fname] = inspect.getsource(f)
            except (OSError, TypeError):
                pass

    def source_of(fn):
        try:
            return inspect.getsource(fn)
        except (OSError, TypeError):
            return ""

    # 记了命中，或把活交给一个记了命中的函数，都算有编译路径。
    # 跟两层：linear 交给 compiled_linear_kernel，embedding 交给 gather。
    hit = {fname for fname, fsrc in factories.items()
           if 'record_route(' in fsrc and '"hit"' in fsrc}
    reaching = set(hit)
    for _ in range(2):
        for fname, fsrc in factories.items():
            if fname not in reaching and any(h + "(" in fsrc for h in reaching):
                reaching.add(fname)

    out = set()
    for name, entry in km._KERNELS.items():
        fns = [entry.mirror]
        if isinstance(entry.compiled, str):
            fns.append(getattr(km, entry.compiled, None))
        elif entry.compiled is not None:
            fns.append(entry.compiled)
        if any(getattr(fn, "__name__", "") in reaching for fn in fns if fn):
            out.add(name)
    return out


def test_every_kernel_in_the_plan_has_a_compiled_path() -> None:
    """计划里的每个 launch 内核都要在编译算子集合里。

    退回计数只覆盖有编译路径的算子，「兜底次数为 0」看不见纯 numpy 实现。
    这条判据钉住计划侧：缺一个就报出来。
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2
    from runtime.kernels import _KERNELS

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32000, hidden_size=64, intermediate_size=176,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
        pad_token_id=0,
    )).float().eval()
    strategy = llama_strategy(
        1, num_stages=1, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=32000, num_layers=1)
    hw = HwBudget(mram_bytes=4 * 2**30, align=64, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=1, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        model, strategy, prefill_seq_len=4, max_seq=16, hw=hw,
        hardware=hardware, kv_dtype_bytes=2, dtype=torch.float32)

    # alias 是视图，不改数值，按设计不编。neg 走减法、slice 走转置加拷贝，
    # 都进编译集合，不在这里豁免。
    numpy_only = {"aten.alias.default"}
    compiled_ops = set(_kernels_reaching_compile())
    kernels = {
        str(cmd.payload["kernel"])
        for plan in (compiled.prefill.plan, compiled.decode.plan)
        for cmd in plan.commands
        if cmd.payload and cmd.payload.get("kernel")
    }
    missing = sorted(kernels - compiled_ops - numpy_only)
    assert not missing, f"这些计划里的内核没有编译路径：{missing}"
