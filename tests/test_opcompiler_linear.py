"""验证已编译线性内核与 NumPy 内核的结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command
from contracts.op_contract import PIMHardwareConfig
from runtime.kernels import compiled_linear_kernel, linear_kernel, register_all

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="opcompiler_bridge.driver 编译时需要在 GPU 上跑一次目标 kernel",
)


def _backend(mram_bytes: int = 1 << 20) -> NumpyBackend:
    return NumpyBackend(
        NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=mram_bytes)
    )


_ALIGN = 4096


def _run(backend: NumpyBackend, kernel_name: str, arg_shapes,
         reads_data: list[np.ndarray], out_shape, dtype="float32",
         hardware: PIMHardwareConfig | None = None):
    """写入张量、执行启动命令并返回结果。"""
    off = 0
    reads = []
    for data in reads_data:
        blob = data.astype(np.dtype(dtype))
        backend.write_local(0, off, blob)
        reads.append(Access(("dpu", 0), off, blob.nbytes))
        off += -(-blob.nbytes // _ALIGN) * _ALIGN
    write_off = off
    write_nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
    if hardware is None:
        hardware = PIMHardwareConfig(
            num_dpus=1,
            num_tasklets=4,
            mram_bytes_per_dpu=backend.config.mram_bytes_per_dpu,
            wram_bytes_per_dpu=64 * 1024,
            # DMA 分块采用 8 字节对齐。
            dma_align=8,
        )
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": kernel_name, "node": "n",
                  "arg_kinds": ["tensor", "tensor"], "arg_shapes": arg_shapes,
                  "dtype": dtype, "out_shape": out_shape,
                  "hardware": hardware.to_payload()},
        reads=reads, writes=[Access(("dpu", 0), write_off, write_nbytes)], waits=[],
        num_tasklets=hardware.num_tasklets,
    )
    event = backend.submit(cmd)
    backend.wait(event)
    return backend.read_local(0, write_off, out_shape, np.dtype(dtype))


def test_compile_request_uses_explicit_hardware_budget(monkeypatch) -> None:
    from contracts.op_contract import OpCompileRequest
    import opcompiler_bridge.driver as driver

    seen = {}

    def fake_make_ttir(request):
        return "module {}"

    def fake_run(ttir, hardware):
        seen["hardware"] = hardware
        # 现在返回 (pim mlir, EmitC) 两段：pim mlir 要留给 GeneSim 的成本模型。
        return (
            "module { func.func @k() { return } }",
            "module { func.func @k(%a: !emitc.ptr<f32>, %b: !emitc.ptr<f32>, "
            "%c: !emitc.ptr<f32>) { return } }",
        )

    monkeypatch.setattr(driver, "_make_ttir", fake_make_ttir)
    monkeypatch.setattr(driver, "_run_triton_opt", fake_run)
    monkeypatch.setattr(driver, "_translate_to_c", lambda _: "void k(float *a, float *b, float *c) {}")

    def fake_subprocess_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "gcc":
            Path(cmd[5]).write_bytes(b"")
        return type("P", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(driver.subprocess, "run", fake_subprocess_run)

    hw = PIMHardwareConfig(1, 7, 1 << 20, 4096, 256)
    result = driver.compile_op(
        OpCompileRequest("linear", [(1, 16), (4, 16)], hw, "float32"),
        force=True,
    )

    assert result.symbol == "k"
    assert seen["hardware"] == hw


def test_compiled_linear_matches_handwritten_numpy() -> None:
    """编译产物与手写 `linear_kernel` 在相同随机输入下逐元素一致。"""
    rng = np.random.default_rng(0)
    M, K, N = 2, 16, 4
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    hand_backend = _backend()
    register_all(hand_backend)
    hand_result = _run(
        hand_backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )

    compiled_backend = _backend()
    register_all(compiled_backend, use_compiled_linear=True)
    compiled_result = _run(
        compiled_backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )

    np.testing.assert_allclose(compiled_result, hand_result, atol=1e-4)


@pytest.mark.parametrize("num_tasklets", [1, 2, 4, 8])
def test_compiled_linear_matches_torch_across_tasklet_counts(num_tasklets) -> None:
    """验证不同 tasklet 数下的已编译线性内核结果。"""
    rng = np.random.default_rng(1)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
        hardware=PIMHardwareConfig(1, num_tasklets, 1 << 20, 64 * 1024, 64),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


def test_compiled_linear_matches_torch() -> None:
    rng = np.random.default_rng(1)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 4096, 512),   # decode 的 q/k/v_proj（8 卡切分后的本地分片）
        (1, 512, 4096),   # decode 的 o_proj（行切）
        (2, 16, 4),       # 小 shape，跑得快，覆盖同一条路径
    ],
)
def test_compiled_linear_float16_matches_handwritten_numpy(M, K, N) -> None:
    """验证 float16 存储下的已编译线性内核结果。"""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    # MRAM 容量覆盖真实权重和对齐余量。
    mram = max(1 << 20, (x.nbytes + w.nbytes + M * N * 2) * 2 + 3 * _ALIGN)

    hand = _backend(mram)
    register_all(hand)
    hand_result = _run(
        hand, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
    )

    compiled = _backend(mram)
    register_all(compiled, use_compiled_linear=True)
    compiled_result = _run(
        compiled, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
    )

    np.testing.assert_allclose(
        compiled_result.astype(np.float32),
        hand_result.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_compiled_linear_real_llama_shape_with_tight_wram_triggers_tile_rewrite() -> None:
    """验证真实 Llama 形状在紧 WRAM 预算下的分块结果。"""
    rng = np.random.default_rng(7)
    M, K, N = 1, 512, 4096
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    mram = max(1 << 20, (x.nbytes + w.nbytes + M * N * 2) * 2 + 3 * _ALIGN)

    hand = _backend(mram)
    register_all(hand)
    hand_result = _run(
        hand, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
        hardware=PIMHardwareConfig(1, 4, mram, 65536, 64),
    )

    compiled = _backend(mram)
    register_all(compiled, use_compiled_linear=True)
    compiled_result = _run(
        compiled, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N), dtype="float16",
        hardware=PIMHardwareConfig(1, 4, mram, 16384, 64),
    )

    np.testing.assert_allclose(
        compiled_result.astype(np.float32),
        hand_result.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


def test_compiled_linear_falls_back_for_non_power_of_two_shapes() -> None:
    """验证非二次幂形状使用 NumPy 线性内核。"""
    rng = np.random.default_rng(2)
    M, K, N = 2, 16, 11  # N=11 不是 2 的幂
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


def test_pick_blocks_returns_power_of_two_below_default() -> None:
    """分块必须自己是 2 的幂，而不只是能整除。

    `_pick` 原先用 `min(want, full)` 起步：`full` 比 `want` 小且不是 2 的幂时
    （如 176），`full % full == 0` 让折半循环一次都不走，返回的 176 直接触发
    Triton 的 "arange's range must be a power of 2"。
    """
    from opcompiler_bridge.kernel_src import DEFAULT_BLOCK_N, _pick, pick_blocks

    for full in (176, 688, 1376, 2752, 5504, 11008, 2048, 4096, 64):
        block = _pick(full, DEFAULT_BLOCK_N, 16)
        assert block & (block - 1) == 0, (full, block)
        assert full % block == 0, (full, block)
        assert block <= DEFAULT_BLOCK_N

    # 最大 2 的幂因子小于下界时应明确报错，而不是返回一个非法分块。
    # 344 = 2^3 × 43，最大 2 的幂因子是 8 < 16。
    with pytest.raises(ValueError, match="2 的幂"):
        _pick(344, DEFAULT_BLOCK_N, 16)

    # llama2 的 MLP 本地形状：intermediate_size = 11008 = 2^8 × 43。
    block_n, block_k = pick_blocks(4096, 5504)
    assert 5504 % block_n == 0 and block_n & (block_n - 1) == 0
    assert 4096 % block_k == 0 and block_k & (block_k - 1) == 0


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 64, 176),     # N 不是 2 的幂，且小于默认 BLOCK_N
        (1, 176, 64),     # K 不是 2 的幂
        (2, 64, 688),     # 688 = 2^4 × 43，与 llama2 的 MLP 同构
    ],
)
def test_compiled_linear_handles_non_power_of_two_k_and_n(M, K, N) -> None:
    """K/N 不是 2 的幂时也要走真实编译产物，并与 NumPy 一致。

    只有 M 需要自己是 2 的幂（`tl.arange(0, M)` 铺满 M）；K/N 是按分块遍历的，
    约束是「存在既是 2 的幂、又能整除它的分块」。原先的守卫对三个维度一律要求
    2 的幂，把 llama2-7b 的 MLP 整个挡在门外——intermediate_size = 11008
    = 2^8 × 43，任何切分下都不是 2 的幂。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel
    import ctypes
    import dataclasses

    hardware = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1
    )
    request = OpCompileRequest(
        op="linear", arg_shapes=[(M, K), (N, K)], hardware=hardware, dtype="float32"
    )
    result = compile_op(request)
    fn = load_kernel(result)

    rng = np.random.default_rng(11)
    x = (rng.standard_normal((M, K)) * 0.05).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    np.testing.assert_allclose(out, x @ w.T, atol=1e-4)


def test_compile_result_carries_pim_mlir() -> None:
    """算子编译要把 pim mlir 一起交回来，供 GeneSim 的代价模型解析真实分块。

    没有它，GeneSim 只能用 conf/sim.yaml 里拍下的 tile_size 常量——llama2 的
    4096 宽投影上真实分块是 512，差 16 倍。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op
    import dataclasses

    hardware = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1
    )
    request = OpCompileRequest(
        op="linear", arg_shapes=[(1, 4096), (2048, 4096)],
        hardware=hardware, dtype="float16",
    )
    result = compile_op(request, force=True)
    assert result.pimir, "编译产物没带上 pim mlir"
    assert result.pimir_path and Path(result.pimir_path).is_file()
    # pim mlir 该有的标志：显式 DMA 和 WRAM 暖存。
    assert "pim.dma_load" in result.pimir
    assert "pim.memdesc" in result.pimir

    cost = analyze_ir(
        result.pimir, kernel_name="linear_kernel", grid=(1,),
        arg_values={}, ir_level="pimir",
    )
    # 真实分块必须读得出来，且不等于 GeneSim 的默认常量 32。
    assert cost.tile_n and cost.tile_n > 0
    assert cost.tile_n != 32, "分块与默认常量相同，测不出这条链路的价值"
    assert cost.mram_traffic_bytes and cost.mram_traffic_bytes > 0
    assert cost.wram_bytes_used and cost.wram_bytes_used <= hardware.wram_bytes_per_dpu

    # 命中缓存时 pim mlir 也要能拿到（从 .pimir.mlir 读回），不必重编。
    cached = compile_op(request)
    assert cached.pimir == result.pimir


def test_compiled_linear_with_tight_wram_budget_triggers_tile_rewrite() -> None:
    """验证紧 WRAM 预算下的已编译线性内核结果。"""
    rng = np.random.default_rng(4)
    M, K, N = 4, 32, 8
    x = rng.standard_normal((M, K)).astype(np.float32)
    w = rng.standard_normal((N, K)).astype(np.float32)

    backend = _backend()
    register_all(backend, use_compiled_linear=True)
    result = _run(
        backend, str(torch.ops.aten.linear.default), [(M, K), (N, K)],
        [x, w], (M, N),
        hardware=PIMHardwareConfig(1, 4, 1 << 20, 512, 64),
    )
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    np.testing.assert_allclose(result, ref, atol=1e-4)


def _run_with_offsets(backend, arg_shapes, reads_data, out_shape, out_off,
                      dtype, hardware):
    """按调用方指定的 offset 布置输入和输出，用于构造读写别名。"""
    npdt = np.dtype(dtype)
    reads = []
    for data, off in reads_data:
        blob = data.astype(npdt)
        backend.write_local(0, off, blob)
        reads.append(Access(("dpu", 0), off, blob.nbytes))
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": str(torch.ops.aten.linear.default), "node": "n",
                  "arg_kinds": ["tensor", "tensor"], "arg_shapes": arg_shapes,
                  "dtype": dtype, "out_shape": out_shape,
                  "hardware": hardware.to_payload()},
        reads=reads,
        writes=[Access(("dpu", 0), out_off, int(np.prod(out_shape)) * npdt.itemsize)],
        waits=[], num_tasklets=hardware.num_tasklets,
    )
    backend.wait(backend.submit(cmd))
    return backend.read_local(0, out_off, out_shape, npdt)


def test_compiled_linear_requires_non_aliasing_output_buffer() -> None:
    """记录已编译内核的读写别名契约：out 与 x 不得重叠。

    已编译内核把裸指针交给 C 函数，逐块读输入、逐块写输出，所以 out 与 x 同基址
    且 out 更大时会覆盖尚未读取的 x 行，算出错误结果。NumPy 内核先整块读入再写回，
    对别名安全，两者因此会不一致。

    这个前提由 `memory/mem_planner.py` 的 `greedy_reuse` 保证：它的两个生命周期
    判据都用严格不等号，不会把某个节点的输出复用到它自己输入的地址上。本测试固定
    这条契约的方向——如果哪天内核改成先把输入读进暂存区（对别名安全），这里会失败，
    提示可以把规划器的判据放宽回去，把激活区省回来。
    """
    M, K, N = 4, 64, 256          # out 2048B > x 512B，M>1：最容易踩的形状
    dtype = "float16"
    rng = np.random.default_rng(11)
    x = rng.standard_normal((M, K)).astype(np.float16)
    w = rng.standard_normal((N, K)).astype(np.float16)
    mram = 1 << 22
    hardware = PIMHardwareConfig(1, 4, mram, 65536, 64)
    x_off, w_off = _ALIGN, _ALIGN + 65536
    disjoint_off = _ALIGN + 2 * 65536

    def run(use_compiled: bool, out_off: int):
        backend = _backend(mram)
        register_all(backend, use_compiled_linear=use_compiled)
        return _run_with_offsets(
            backend, [(M, K), (N, K)], [(x, x_off), (w, w_off)], (M, N),
            out_off, dtype, hardware,
        )

    # out 与输入不重叠：两条内核必须一致。
    np.testing.assert_allclose(
        run(True, disjoint_off).astype(np.float32),
        run(False, disjoint_off).astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )

    # out 与 x 同基址：已编译内核会算错，规划器必须避免造出这种布局。
    aliased = run(True, x_off).astype(np.float32)
    reference = run(False, disjoint_off).astype(np.float32)
    scale = max(np.abs(reference).max(), 1e-6)
    assert np.abs(aliased - reference).max() / scale > 1.0, (
        "已编译内核在读写别名下竟与参考值接近——若内核已改为对别名安全，"
        "可以放宽 greedy_reuse 的判据并删除本断言"
    )
