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
        return (
            "module { func.func @k(%a: !emitc.ptr<f32>, %b: !emitc.ptr<f32>, "
            "%c: !emitc.ptr<f32>) { return } }"
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
