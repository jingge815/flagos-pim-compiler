"""问题 6 阶段 B 单测：DPU 白名单算子的 NumPy 镜像 kernel 逐算子对拍。

判据：每个 kernel 直接构造一条 `Command`（不经 exec_plan_gen，隔离验证 kernel
本身的 payload 解析 + 计算），在 NumpyBackend 上写输入、submit、读回，与
torch 参考逐元素对齐。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Access, Command
from runtime.kernels import register_all


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=4096))


def _run(backend: NumpyBackend, kernel_name: str, arg_kinds, arg_shapes,
         reads_data: list[np.ndarray], out_shape, dtype="float32"):
    """写入全部张量参数、构造 launch 命令、submit/wait，返回结果与写地址。"""
    off = 0
    reads = []
    for data in reads_data:
        backend.write_local(0, off, data.astype(np.dtype(dtype)))
        reads.append(Access(("dpu", 0), off, data.astype(np.dtype(dtype)).nbytes))
        off += 512  # 对齐留足空间，避免相邻张量重叠
    write_off = off
    write_nbytes = int(np.prod(out_shape)) * np.dtype(dtype).itemsize
    cmd = Command(
        id=0, op="launch", dpu_id=0,
        payload={"kernel": kernel_name, "node": "n", "arg_kinds": arg_kinds,
                  "arg_shapes": arg_shapes, "dtype": dtype, "out_shape": out_shape},
        reads=reads, writes=[Access(("dpu", 0), write_off, write_nbytes)], waits=[],
    )
    event = backend.submit(cmd)
    backend.wait(event)
    return backend.read_local(0, write_off, out_shape, np.dtype(dtype))


def test_linear_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 4)).astype(np.float32)
    w = rng.standard_normal((3, 4)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.linear.default), ["tensor", "tensor"],
                  [(2, 4), (3, 4)], [x, w], (2, 3))
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(result, ref, atol=1e-5)


def test_add_kernel_tensor_tensor_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3,)).astype(np.float32)
    y = rng.standard_normal((3,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.add.Tensor), ["tensor", "tensor"],
                  [(3,), (3,)], [x, y], (3,))
    assert np.allclose(result, x + y, atol=1e-6)


def test_add_kernel_tensor_scalar_matches_torch() -> None:
    """RMSNorm 的 `add(x, eps)` 形态：第二参数是字面量，不占 reads。"""
    backend = _backend()
    register_all(backend)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _run(backend, str(torch.ops.aten.add.Tensor), ["tensor", 1e-5],
                  [(3,), None], [x], (3,))
    assert np.allclose(result, x + 1e-5, atol=1e-8)


def test_mul_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4,)).astype(np.float32)
    y = rng.standard_normal((4,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.mul.Tensor), ["tensor", "tensor"],
                  [(4,), (4,)], [x, y], (4,))
    assert np.allclose(result, x * y, atol=1e-6)


def test_tanh_kernel_matches_torch() -> None:
    backend = _backend()
    register_all(backend)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5,)).astype(np.float32)
    result = _run(backend, str(torch.ops.aten.tanh.default), ["tensor"], [(5,)], [x], (5,))
    assert np.allclose(result, np.tanh(x), atol=1e-6)
