"""提供 DPU 白名单算子的 NumPy 镜像内核和已编译内核调用。"""

from __future__ import annotations

import ctypes
import glob
import os
import threading

import numpy as np
import torch


def _limit_blas_threads(n: int = 1) -> bool:
    """将 OpenBLAS 线程数设为 ``n``；未找到库时返回 ``False``。"""
    try:
        numpy_libs_dir = os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs")
        candidates = glob.glob(os.path.join(numpy_libs_dir, "*openblas*"))
        if not candidates:
            return False
        lib = ctypes.CDLL(candidates[0])
        for name in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_",
                     "openblas_set_num_threads"):
            fn = getattr(lib, name, None)
            if fn is not None:
                fn(n)
                return True
        return False
    except OSError:
        return False


_limit_blas_threads(1)

# 配置 PyTorch 的 MKL 线程数。
torch.set_num_threads(1)


def _read_tensor_args(hal, dpu_id: int, cmd) -> list:
    """按 `arg_kinds`/`arg_shapes` 把 `cmd.reads` 还原成完整调用参数列表。"""
    dtype = np.dtype(cmd.payload["dtype"])
    args = []
    read_i = 0
    for kind, shape in zip(cmd.payload["arg_kinds"], cmd.payload["arg_shapes"]):
        if kind == "tensor":
            access = cmd.reads[read_i]
            args.append(hal.read_local(dpu_id, access.offset, tuple(shape), dtype))
            read_i += 1
        else:
            args.append(kind)
    return args


def _write_result(hal, dpu_id: int, cmd, result: np.ndarray) -> None:
    dtype = np.dtype(cmd.payload["dtype"])
    hal.write_local(dpu_id, cmd.writes[0].offset, np.ascontiguousarray(result, dtype=dtype))


def linear_kernel(hal, dpu_id: int, cmd) -> None:
    """计算 ``aten.linear(x, w)``。"""
    x, w = _read_tensor_args(hal, dpu_id, cmd)
    y = x.astype(np.float32) @ w.astype(np.float32).T
    _write_result(hal, dpu_id, cmd, y)


def tasklet_linear_kernel(hal, dpu_id: int, cmd) -> None:
    """按行将线性计算分给各 tasklet，并记录读写区间。"""
    x, w = _read_tensor_args(hal, dpu_id, cmd)
    num_tasklets = cmd.num_tasklets
    m = x.shape[0]
    rows_per_tasklet = -(-m // num_tasklets)  # 向上取整

    dtype = np.dtype(cmd.payload["dtype"])
    out_access = cmd.writes[0]
    x_access = cmd.reads[0]
    k = x.shape[1]
    row_bytes_x = k * dtype.itemsize
    row_bytes_out = w.shape[0] * dtype.itemsize

    for tid in range(num_tasklets):
        row_start = tid * rows_per_tasklet
        row_end = min(row_start + rows_per_tasklet, m)
        if row_start >= row_end:
            continue
        hal.record_access(tid, "mram", x_access.offset + row_start * row_bytes_x,
                           (row_end - row_start) * row_bytes_x, is_write=False)
        y_slice = x[row_start:row_end].astype(np.float32) @ w.astype(np.float32).T
        hal.record_access(tid, "mram", out_access.offset + row_start * row_bytes_out,
                           (row_end - row_start) * row_bytes_out, is_write=True)
        hal.write_local(dpu_id, out_access.offset + row_start * row_bytes_out,
                         np.ascontiguousarray(y_slice, dtype=dtype))
    hal.barrier()


# 已加载编译内核的缓存。
_COMPILED_KERNEL_CACHE: dict[tuple, object] = {}

# 保护编译缓存的互斥锁。
_COMPILE_LOCK = threading.Lock()


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _compiled_linear_supports(arg_shapes, dtype: str = "float32") -> bool:
    """判断线性算子形状和数据类型是否可由已编译内核处理。"""
    from contracts.op_contract import flatten_leading_dims

    if dtype not in ("float16", "float32"):
        return False
    m, k = flatten_leading_dims(arg_shapes[0])
    n = arg_shapes[1][0]
    return k >= 16 and _is_pow2(m) and _is_pow2(k) and _is_pow2(n)


def compiled_linear_kernel(hal, dpu_id: int, cmd) -> None:
    """运行已编译的线性内核；不支持的形状改用 NumPy 内核。"""
    arg_shapes = tuple(tuple(s) for s in cmd.payload["arg_shapes"])
    dtype = str(cmd.payload["dtype"])
    if not _compiled_linear_supports(arg_shapes, dtype):
        linear_kernel(hal, dpu_id, cmd)
        return

    from contracts.op_contract import OpCompileRequest, PIMHardwareConfig
    from opcompiler_bridge.driver import compile_op, load_kernel

    hardware = PIMHardwareConfig.from_payload(cmd.payload["hardware"])
    if cmd.num_tasklets != hardware.num_tasklets:
        raise ValueError(
            f"cmd.num_tasklets ({cmd.num_tasklets}) must match hardware.num_tasklets ({hardware.num_tasklets})"
        )
    num_tasklets = hardware.num_tasklets
    # 键包含数据类型、tasklet 数和硬件配置。
    key = ("linear", arg_shapes, dtype, tuple(hardware.to_payload().items()))
    fn = _COMPILED_KERNEL_CACHE.get(key)
    if fn is None:
        # 锁内检查编译缓存。
        with _COMPILE_LOCK:
            fn = _COMPILED_KERNEL_CACHE.get(key)
            if fn is None:
                result = compile_op(
                    OpCompileRequest(
                        op="linear", arg_shapes=list(arg_shapes), hardware=hardware,
                        dtype=dtype, num_tasklets=num_tasklets,
                    )
                )
                fn = load_kernel(result)
                _COMPILED_KERNEL_CACHE[key] = fn

    x_access, w_access = cmd.reads
    (out_access,) = cmd.writes
    base = hal.raw_mram_ptr(dpu_id)
    fn(
        ctypes.c_void_p(base + x_access.offset),
        ctypes.c_void_p(base + w_access.offset),
        ctypes.c_void_p(base + out_access.offset),
    )


def add_kernel(hal, dpu_id: int, cmd) -> None:
    """计算逐元素加法。"""
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0].astype(np.float32)
    y = args[1].astype(np.float32) if isinstance(args[1], np.ndarray) else args[1]
    _write_result(hal, dpu_id, cmd, x + y)


def mul_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.mul.Tensor(x, y_or_scalar)`：逐元素乘。"""
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0].astype(np.float32)
    y = args[1].astype(np.float32) if isinstance(args[1], np.ndarray) else args[1]
    _write_result(hal, dpu_id, cmd, x * y)


def tanh_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.tanh(x)`：逐元素 tanh。"""
    (x,) = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, np.tanh(x.astype(np.float32)))


_KERNELS = {
    str(torch.ops.aten.linear.default): linear_kernel,
    str(torch.ops.aten.add.Tensor): add_kernel,
    str(torch.ops.aten.mul.Tensor): mul_kernel,
    str(torch.ops.aten.tanh.default): tanh_kernel,
}


def register_all(hal, *, use_compiled_linear: bool = False) -> None:
    """向后端注册白名单内核，可选择注册已编译的线性内核。"""
    kernels = dict(_KERNELS)
    if use_compiled_linear:
        kernels[str(torch.ops.aten.linear.default)] = compiled_linear_kernel
    for name, fn in kernels.items():
        hal.register_kernel(name, fn)
