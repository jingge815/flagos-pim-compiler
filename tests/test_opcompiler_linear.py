"""验证已编译线性内核与 NumPy 内核的结果。"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command
from contracts.op_contract import PIMHardwareConfig
from runtime.kernels import compiled_linear_kernel, linear_kernel, register_all


def _pim_passes_available() -> bool:
    """这份 triton 带 PIM pass 吗（不问有没有 GPU 硬件）。"""
    try:
        from genesim_bridge.env import assert_pim_passes_available

        assert_pim_passes_available()
    except Exception:
        return False
    return True

# 算子编译不需要 GPU 硬件：TTIR 是纯前端产物，无卡时走 cpu_host 的前端路径，
# 产出的 pim mlir 与有卡路径 sha256 相同（见 opcompiler_bridge/cpu_host.py）。
# 真正的前提是这份 triton 里带 PIM pass——缺了它才没有可测的东西。
pytestmark = pytest.mark.skipif(
    not _pim_passes_available(),
    reason="当前 triton 没有 PIM pass，需重跑 0-install-flagtree.sh",
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

    def fake_run(ttir, hardware, shard=None, elem_strides=(),
                 mram_offset=0, align_bytes=0):
        seen["hardware"] = hardware
        seen["shard"] = shard
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


def test_compiled_linear_handles_non_power_of_two_n() -> None:
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


def test_clamp_block_takes_the_largest_power_of_two_below_full() -> None:
    """分块取不超过维度和上限的最大 2 的幂，不再要求整除维度。

    维度本身不超过上限且是 2 的幂时直接取整维。不是 2 的幂时向下取到 2 的幂：
    176 取 128、344 取 256。超过上限（默认 512）的一律取 512。余下的部分由
    尾块掩码处理，所以不需要整除，也不需要为「最大 2 的幂因子小于下界」报错。
    """
    from opcompiler_bridge.kernel_src import (
        DEFAULT_BLOCK_N, _clamp_block, pick_blocks)

    for full, expected in (
        (64, 64), (176, 128), (344, 256), (512, 512),
        (688, 512), (1376, 512), (2752, 512), (5504, 512),
        (11008, 512), (2048, 512), (4096, 512),
    ):
        block = _clamp_block(full, DEFAULT_BLOCK_N)
        assert block & (block - 1) == 0, (full, block)
        assert block <= min(full, DEFAULT_BLOCK_N), (full, block)
        assert block == expected, (full, block, expected)

    # 维度 1 也要给出合法分块，不能退化成 0。
    assert _clamp_block(1, DEFAULT_BLOCK_N) == 1

    # llama2 的 MLP 本地形状：intermediate_size = 11008 = 2^8 × 43。
    # 三个维都只要求分块是 2 的幂且不超维度。
    block_m, block_n, block_k = pick_blocks(4096, 5504, 1)
    assert block_m == 1
    assert block_n == 512
    assert block_k == 32


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 64, 176),     # N 不是 2 的幂，且小于默认 BLOCK_N
        (1, 176, 64),     # K 不是 2 的幂
        (2, 64, 688),     # 688 = 2^4 × 43，与 llama2 的 MLP 同构
        (1, 20, 16),      # K=20 落在 [16, 31]，BLOCK_K 取 16，K 维出尾块
    ],
)
def test_compiled_linear_handles_non_power_of_two_k_and_n(M, K, N) -> None:
    """K/N 不是 2 的幂时也要走真实编译产物，并与 NumPy 一致。

    M、K、N 三个维度都按分块遍历，约束只剩「分块自己是 2 的幂」，维度不必被
    分块整除、也不必是 2 的幂，余数由尾块掩码处理。原先的守卫对三个维度一律
    要求 2 的幂，把 llama2-7b 的 MLP 整个挡在门外——intermediate_size
    = 11008 = 2^8 × 43，任何切分下都不是 2 的幂。
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


def test_mlp_shapes_reach_the_compiled_kernel() -> None:
    """1376 和 32000 这类维度必须走编译内核，不能在进编译器前被退回。

    这是 llama2 的 MLP 与 lm_head 在 8 DPU 下的真实维度，原先被
    `_compiled_linear_supports` 的 2 的幂判断挡掉。
    """
    from runtime.kernels import _compiled_linear_supports

    cases = [
        ([(1, 8, 4096), (1376, 4096)], "float16"),
        ([(1, 8, 1376), (4096, 1376)], "float16"),
        ([(1, 1, 4096), (32000, 4096)], "float16"),
        ([(3, 32), (16, 32)], "float32"),
    ]
    for shapes, dtype in cases:
        assert _compiled_linear_supports(shapes, dtype), (shapes, dtype)


def test_arbitrary_shape_compiles() -> None:
    """llama2 的真实投影形状都要真的编出内核，并与 numpy 对拍。

    MLP 的 `intermediate_size = 1376`（本地宽度）、`down_proj` 的 K=1376、
    `lm_head` 的 N=32000 原先被 `_compiled_linear_supports` 的 2 的幂判断
    整个挡在编译器外面。硬件用 `DEFAULT_HARDWARE_CONFIG`（8 DPU / 16
    tasklet）：分块装不装得进 WRAM 由 DPU 数与 tasklet 数决定，1 DPU 编得过
    不等于 8 DPU 编得过。
    """
    import ctypes

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel
    from runtime.kernels import _compiled_linear_supports

    hardware = DEFAULT_HARDWARE_CONFIG
    rng = np.random.default_rng(13)
    # N=1376 是 gate/up 的本地输出宽度，K=1376 是 down_proj 的本地输入宽度，
    # N=32000 是 lm_head 的词表宽度。
    cases = (
        ("gate_proj", 4096, 1376),
        ("down_proj", 1376, 4096),
        ("lm_head", 4096, 32000),
    )
    for M in (1, 128):
        for name, k, n in cases:
            shapes = [(M, k), (n, k)]
            assert _compiled_linear_supports(shapes, "float16"), (name, M)
            result = compile_op(OpCompileRequest(
                op="linear", arg_shapes=shapes, hardware=hardware,
                dtype="float16",
            ))
            assert Path(result.so_path).is_file(), f"{name} M={M} 没有产出内核"
            fn = load_kernel(result)

            x = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
            w = (rng.standard_normal((n, k)) * 0.05).astype(np.float16)
            out = np.zeros((M, n), dtype=np.float16)
            fn(
                x.ctypes.data_as(ctypes.c_void_p),
                w.ctypes.data_as(ctypes.c_void_p),
                out.ctypes.data_as(ctypes.c_void_p),
            )
            ref = x.astype(np.float32) @ w.astype(np.float32).T
            rel = np.abs(out.astype(np.float32) - ref).max() / max(
                np.abs(ref).max(), 1e-6)
            assert rel < 0.05, f"{name} M={M} 相对误差 {rel:.4e}"


def test_odd_m_compiles() -> None:
    """M 不是 2 的幂时也要编译通过，并与 numpy 逐元素一致。

    内核原先用 `tl.arange(0, M)` 一次铺满 M，Triton 要求这个范围是 2 的幂，
    于是 prefill 的序列长度被限制死。M 维也要能分块。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel
    import ctypes
    import dataclasses

    M, K, N = 3, 32, 16
    hardware = dataclasses.replace(DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1)
    result = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(M, K), (N, K)], hardware=hardware, dtype="float32",
    ))
    fn = load_kernel(result)

    rng = np.random.default_rng(5)
    x = (rng.standard_normal((M, K)) * 0.05).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    np.testing.assert_allclose(out, x @ w.T, atol=1e-4)


def test_cross_check_wrapper_records_the_fallback_route() -> None:
    """对拍包装退回镜像时也要记 fallback。

    端到端测试用 `_wrap_with_numpy_cross_check` 包住 `compiled_linear_kernel`，
    而不支持的形状下包装函数直接调 `km.linear_kernel`——记录点在
    `compiled_linear_kernel` 里，被这一绕就没了。于是「每个算子的兜底次数为 0」
    这条判据看不见 linear 的退回。
    """
    import runtime.kernels as km
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.exec_plan import Access, Command
    from tests.test_opcompiler_e2e_llama2_7b import _wrap_with_numpy_cross_check

    km.reset_route_counts()
    wrapped, _stats = _wrap_with_numpy_cross_check(km.compiled_linear_kernel)

    backend = NumpyBackend(
        NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=1 << 20))
    m, k, n = 1, 8, 4  # K=8 小于 tl.dot 的下限 16，只能退回镜像
    x = np.ones((m, k), dtype=np.float16)
    w = np.ones((n, k), dtype=np.float16)
    backend.write_local(0, 0, x)
    backend.write_local(0, 64, w)
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": "linear", "node": "n",
                 "arg_kinds": ["tensor", "tensor"],
                 "arg_shapes": [x.shape, w.shape], "dtype": "float16",
                 "out_shape": (m, n)},
        reads=[Access(("dpu", 0), 0, x.nbytes), Access(("dpu", 0), 64, w.nbytes)],
        writes=[Access(("dpu", 0), 128, m * n * 2)],
        waits=[], num_tasklets=1,
    )
    wrapped(backend, 0, cmd)

    counts = km.route_counts()
    assert counts.get(("linear", "fallback"), 0) == 1, counts


def test_cache_key_follows_the_compiler_fingerprint(monkeypatch) -> None:
    """缓存键要随编译器一起失效，否则改内核或改 pass 后仍复用旧产物。

    缓存原先只比对请求参数（形状、dtype、硬件……）。改了 `kernel_src.py` 的
    内核结构或 FlagTree 的降级 pass 之后，磁盘上的 `.pimir.mlir` 与 `.so`
    不会被判失效，仿真拿到的就不是当前代码的产物。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge import driver

    request = OpCompileRequest(
        op="linear", arg_shapes=[(2, 32), (16, 32)],
        hardware=DEFAULT_HARDWARE_CONFIG, dtype="float32",
    )
    before = driver._cache_key(request)
    monkeypatch.setattr(driver, "_compiler_fingerprint", lambda: "另一个编译器")
    assert driver._cache_key(request) != before, "缓存键没有跟着编译器指纹变"


def test_compiler_fingerprint_reflects_the_kernel_source() -> None:
    """指纹要由真实的内核源码与工具链文件算出来，且可重复。"""
    from opcompiler_bridge.driver import _compiler_fingerprint

    first = _compiler_fingerprint()
    assert first and first == _compiler_fingerprint()


def test_compiler_fingerprint_covers_ir_generators(monkeypatch) -> None:
    """指纹要覆盖生成 IR 的源文件，不能只盯 kernel_src.py。

    算子级（B 路）的 IR 由 `oplevel_kernel.py` 与 `oplevel_emitter.py` 生成。
    改了它们而不改 kernel_src 时，旧产物会被静默复用，形状、类型全对、
    数值全错。把这一项从摘要里拿掉，指纹必须因此不同。
    """
    from pathlib import Path

    from opcompiler_bridge import driver

    original = Path.read_bytes

    def stripped(self):
        if self.name in ("oplevel_kernel.py", "oplevel_emitter.py"):
            return b""
        return original(self)

    before = driver._compiler_fingerprint()
    monkeypatch.setattr(Path, "read_bytes", stripped)
    assert driver._compiler_fingerprint() != before, (
        "指纹没有覆盖 oplevel_kernel.py / oplevel_emitter.py")


def test_compiler_fingerprint_covers_the_loaded_libtriton() -> None:
    """指纹要覆盖进程实际加载的 `libtriton.so`，不能只盯 `triton-opt`。

    降级 pass 编进两份产物：离线的 `triton-opt` 与进程内的 `libtriton.so`。
    只盯前者时，单独重编后者不会让缓存失效，运行时会拿旧内核的产物去对拍。
    """
    import hashlib
    from pathlib import Path

    import triton

    from opcompiler_bridge import driver

    loaded = driver._loaded_libtriton()
    assert loaded is not None and loaded.is_file(), loaded
    assert loaded == Path(triton.__file__).parent / "_C" / "libtriton.so"

    # 用同一套算法去掉 libtriton 这一项，指纹必须因此不同。
    digest = hashlib.sha256()
    for name in ("kernel_src.py", "oplevel_kernel.py", "oplevel_emitter.py"):
        digest.update(Path(driver.__file__).with_name(name).read_bytes())
    tool = driver._triton_opt()
    if tool.is_file():
        stat = tool.stat()
        digest.update(f"{tool}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    assert driver._compiler_fingerprint() != digest.hexdigest()[:16], (
        "指纹没有覆盖 libtriton.so")


def test_cache_key_lock_serializes_processes(tmp_path, monkeypatch) -> None:
    """同一缓存 key 的两个进程不能同时进入编译区。"""
    from opcompiler_bridge import driver

    monkeypatch.setattr(driver, "_CACHE_DIR", tmp_path)
    context = mp.get_context("fork")
    events = context.Queue()

    def worker(queue):
        with driver._cache_key_lock("same-key"):
            queue.put(("start", time.monotonic()))
            time.sleep(0.12)
            queue.put(("end", time.monotonic()))

    processes = [context.Process(target=worker, args=(events,)) for _ in range(2)]
    for process in processes:
        process.start()
    records = [events.get(timeout=5) for _ in range(4)]
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    starts = sorted(t for kind, t in records if kind == "start")
    ends = sorted(t for kind, t in records if kind == "end")
    assert len(starts) == len(ends) == 2
    assert starts[1] >= ends[0] - 0.02


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


def test_tail_block_matches_numpy() -> None:
    """维度不被分块整除时，最后一块也要与 numpy 逐元素一致。

    三个维度都取不能被分块整除的值，尾块掩码算多或算少都会在这一块上露出来。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel
    import ctypes
    import dataclasses

    M, K, N = 3, 40, 20
    hardware = dataclasses.replace(DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1)
    result = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(M, K), (N, K)], hardware=hardware, dtype="float32",
    ))
    fn = load_kernel(result)

    rng = np.random.default_rng(9)
    x = (rng.standard_normal((M, K)) * 0.05).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    out = np.zeros((M, N), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    np.testing.assert_allclose(out, x @ w.T, atol=1e-4)


def test_triton_and_emitc_agree_on_the_tail_block() -> None:
    """同一个尾块形状下，Triton 内核与 EmitC 产物必须逐元素一致。

    Triton 侧用掩码（`other=0`）丢掉越界元素，EmitC 侧没有掩码机制，改成把
    越界下标夹到最后一个合法位置。两种口径只有在越界元素**不参与累加**时
    才等价，这条测试就是钉住这个前提。

    数据里埋了陷阱：K 的尾块越界位置会夹到最后一个合法下标，于是那个元素
    被反复读到。把它设成 100，一旦它进了累加，结果会比参考值大 8 万倍量级
    （8 个越界位置各多算一次），任何一处口径出错都藏不住。
    """
    import ctypes
    import dataclasses

    import torch

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel
    from opcompiler_bridge.kernel_src import NUM_STAGES, pick_blocks

    if not torch.cuda.is_available():
        pytest.skip("Triton 原生内核需要 GPU 才能对照")

    M, K, N = 3, 40, 20
    block_m, block_n, block_k = pick_blocks(K, N, M)
    # 三个维都要有尾块，否则这条对照退化成整除情形。
    assert M % block_m and K % block_k and N % block_n, (block_m, block_n, block_k)

    rng = np.random.default_rng(3)
    x = (rng.standard_normal((M, K)) * 0.05).astype(np.float32)
    w = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    x[:, K - 1] = 100.0
    w[:, K - 1] = 100.0

    # 一、Triton 原生内核（掩码口径）。
    from opcompiler_bridge.kernel_src import linear_kernel
    tx = torch.from_numpy(x).cuda()
    tw = torch.from_numpy(w).cuda()
    t_out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    linear_kernel[(1,)](
        tx, tw, t_out, M=M, K=K, N=N,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_stages=NUM_STAGES,
    )
    triton_out = t_out.cpu().numpy()

    # 二、EmitC 产物（下标夹取口径）。
    hardware = dataclasses.replace(
        DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1)
    result = compile_op(OpCompileRequest(
        op="linear", arg_shapes=[(M, K), (N, K)],
        hardware=hardware, dtype="float32",
    ))
    fn = load_kernel(result)
    emitc_out = np.zeros((M, N), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        emitc_out.ctypes.data_as(ctypes.c_void_p),
    )

    ref = x @ w.T
    np.testing.assert_allclose(triton_out, ref, atol=1e-2)
    np.testing.assert_allclose(emitc_out, ref, atol=1e-2)
    # 互比：两套口径的产物之间也要一致。
    np.testing.assert_allclose(emitc_out, triton_out, atol=1e-3)


def test_k_below_16_stays_on_the_numpy_mirror() -> None:
    """K 小于 16 时必须退回 numpy 镜像，不能进编译。

    Triton 的 tl.dot 要求 K>=16，编译器对这种形状直接抛 ValueError。
    如果谓词仍判它"支持"，运行时就会把一次正常的小矩阵乘变成崩溃，
    而且这次崩溃不进退回计数。
    """
    from runtime.kernels import _compiled_linear_supports

    assert not _compiled_linear_supports([(1, 8), (4, 8)], "float16")
    assert not _compiled_linear_supports([(2, 4), (4, 4)], "float32")
    # 恰好 16 仍走编译。
    assert _compiled_linear_supports([(1, 16), (4, 16)], "float16")
