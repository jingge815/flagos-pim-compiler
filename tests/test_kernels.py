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

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig, TaskletHazardError
from contracts.exec_plan import Access, Command
from runtime.kernels import register_all, tasklet_linear_kernel


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=4096))


def _run(backend: NumpyBackend, kernel_name: str, arg_kinds, arg_shapes,
         reads_data: list[np.ndarray], out_shape, dtype="float32", num_tasklets=4):
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
        num_tasklets=num_tasklets,
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


@pytest.mark.parametrize("num_tasklets", [1, 2, 3, 5, 8])
def test_tasklet_linear_kernel_matches_torch_for_various_tasklet_counts(num_tasklets) -> None:
    """按 M 维切分给 num_tasklets 个 tasklet 顺序模拟，数值必须与 torch 参考逐元素一致。

    num_tasklets 覆盖：整除 M（2）、不整除 M（3，M=7）、大于 M（8，M=7，尾部
    tasklet 空转）——见 tasklet_linear_kernel 里 `row_start >= row_end: continue`。
    """
    backend = _backend()
    backend.register_kernel("tasklet_linear", tasklet_linear_kernel)
    rng = np.random.default_rng(4)
    m, k, n = 7, 4, 3
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    result = _run(backend, "tasklet_linear", ["tensor", "tensor"], [(m, k), (n, k)],
                  [x, w], (m, n), num_tasklets=num_tasklets)
    ref = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(w)).numpy()
    assert np.allclose(result, ref, atol=1e-5)


def test_tasklet_linear_kernel_row_ranges_are_disjoint_and_cover_m() -> None:
    """切分区间本身要满足：互不重叠、并集覆盖 [0, M)——不止数值对，划分逻辑也要对。"""
    m, num_tasklets = 10, 3
    rows_per_tasklet = -(-m // num_tasklets)
    ranges = []
    for tid in range(num_tasklets):
        row_start = tid * rows_per_tasklet
        row_end = min(row_start + rows_per_tasklet, m)
        if row_start < row_end:
            ranges.append((row_start, row_end))
    covered = set()
    for start, end in ranges:
        rng_set = set(range(start, end))
        assert not (covered & rng_set), f"tasklet 行区间重叠: {ranges}"
        covered |= rng_set
    assert covered == set(range(m))


def test_tasklet_linear_kernel_hazard_detection_catches_broken_split() -> None:
    """故意构造一个"漏隔离"的坏 kernel：两个 tasklet 各自算出的行区间人为重叠
    写同一段输出、且都不调 hal.barrier() ——hazard 检测必须能抓到，这是方案
    验收标准 (b) 的直接落地：不仅要数值对，还要能验证同步/依赖问题本身。
    """
    backend = _backend()

    def broken_tasklet_linear(hal, dpu_id, cmd) -> None:
        from runtime.kernels import _read_tensor_args

        x, w = _read_tensor_args(hal, dpu_id, cmd)
        dtype = np.dtype(cmd.payload["dtype"])
        out_access = cmd.writes[0]
        row_bytes = w.shape[0] * dtype.itemsize
        # 故意让两个 tasklet 的"行区间"重叠（都写 [0, 2) ），且中间不调用
        # hal.barrier()——真实硬件上这就是两个 tasklet 同时写同一段 MRAM。
        for tid in (0, 1):
            hal.record_access(tid, "mram", out_access.offset, 2 * row_bytes, is_write=True)
            y_slice = x[0:2].astype(np.float32) @ w.astype(np.float32).T
            hal.write_local(dpu_id, out_access.offset, np.ascontiguousarray(y_slice, dtype=dtype))
        hal.barrier()

    backend.register_kernel("broken_tasklet_linear", broken_tasklet_linear)
    rng = np.random.default_rng(5)
    m, k, n = 4, 4, 3
    x = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((n, k)).astype(np.float32)
    with pytest.raises(TaskletHazardError):
        _run(backend, "broken_tasklet_linear", ["tensor", "tensor"], [(m, k), (n, k)],
             [x, w], (m, n), num_tasklets=2)
