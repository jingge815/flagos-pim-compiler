"""提供 DPU 白名单算子的 NumPy 镜像内核和已编译内核调用。"""

from __future__ import annotations

import ctypes
import glob
import os
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from memory.kv_layout import PIMStaticKVCache, decode_mask, prefill_mask
from runtime import kernels_pim


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
    """按 `arg_kinds`/`arg_shapes`/`arg_dtypes` 还原成完整调用参数列表。

    输入 dtype 逐参取 `arg_dtypes`，**不能**用 payload 的 `dtype`（那是输出
    的）：`to.dtype` 的 f16→f32 会按 f32 去读 f16 缓冲，整块读成垃圾。
    老 payload 没有这个键时退回输出 dtype，行为不变。
    """
    dtypes = cmd.payload.get("arg_dtypes")
    default_dtype = np.dtype(cmd.payload["dtype"])
    args = []
    read_i = 0
    for index, (kind, shape) in enumerate(
            zip(cmd.payload["arg_kinds"], cmd.payload["arg_shapes"])):
        if kind == "tensor":
            access = cmd.reads[read_i]
            name = dtypes[index] if dtypes else None
            dtype = np.dtype(name) if name else default_dtype
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


def _compiled_eltwise(kind: str, shape: tuple[int, ...]):
    """按形状编一个逐元素算子（带缓存）；工具链不在位时返回 None。

    只回退工具链缺失。编译器自己报的错往上抛——静默回退镜像会让
    「编译内核与镜像一致」这条判据在编译失败时也通过。
    """
    key = ("eltwise", kind, shape)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="eltwise", arg_shapes=[shape, shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16", kind=kind,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def _eltwise(kind: str, numpy_op):
    """逐元素二元算子：同形状且两边都是张量时走编译内核，其余走 numpy。

    编译内核只覆盖两槽同形（`driver.py` 的契约），标量与广播没有对应的
    设备循环，那些形态仍由 numpy 算。
    """
    def kernel(hal, dpu_id: int, cmd) -> None:
        args = _read_tensor_args(hal, dpu_id, cmd)
        x, y = args[0], args[1]
        fn = None
        if (isinstance(y, np.ndarray) and x.shape == y.shape
                and x.dtype == np.float16):
            fn = _compiled_eltwise(kind, tuple(int(d) for d in x.shape))
        if fn is None:
            y_value = y.astype(np.float32) if isinstance(y, np.ndarray) else y
            _write_result(hal, dpu_id, cmd,
                          numpy_op(x.astype(np.float32), y_value))
            return
        out = np.zeros(x.shape, dtype=np.float16)
        fn(np.ascontiguousarray(x, dtype=np.float16).ctypes.data_as(ctypes.c_void_p),
           np.ascontiguousarray(y, dtype=np.float16).ctypes.data_as(ctypes.c_void_p),
           out.ctypes.data_as(ctypes.c_void_p))
        _write_result(hal, dpu_id, cmd, out)

    return kernel


def add_kernel(hal, dpu_id: int, cmd) -> None:
    """计算逐元素加法。"""
    _eltwise("add", lambda x, y: x + y)(hal, dpu_id, cmd)


def mul_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.mul.Tensor(x, y_or_scalar)`：逐元素乘。"""
    _eltwise("mul", lambda x, y: x * y)(hal, dpu_id, cmd)


def tanh_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.tanh(x)`：逐元素 tanh。"""
    (x,) = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, np.tanh(x.astype(np.float32)))


def sub_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.sub.Tensor(x, y_or_scalar)`：逐元素减。"""
    _eltwise("sub", lambda x, y: x - y)(hal, dpu_id, cmd)


def div_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.div.Tensor(x, y_or_scalar)`：逐元素除。"""
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0].astype(np.float32)
    y = args[1].astype(np.float32) if isinstance(args[1], np.ndarray) else args[1]
    _write_result(hal, dpu_id, cmd, x / y)


def _unary(fn):
    """把一个 numpy 一元函数包成内核。

    运算一律在 fp32 里做，存储 dtype 由 `_write_result` 按 payload 决定——与
    `linear_kernel` 同口径，也是硬件的口径（fp16 存、fp32 算）。
    """

    def kernel(hal, dpu_id: int, cmd) -> None:
        (x,) = _read_tensor_args(hal, dpu_id, cmd)
        _write_result(hal, dpu_id, cmd, fn(x.astype(np.float32)))

    return kernel


def pow_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.pow.Tensor_Scalar(x, exponent)`：逐元素幂。

    指数是标量（`_read_tensor_args` 会把非张量实参原样带回来），RMSNorm 里恒为 2。
    """
    x, exponent = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, np.power(x.astype(np.float32), exponent))


def mean_dim_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.mean.dim(x, dim, keepdim)`：沿给定维求均值。

    切分规则（`spec_prop._rule_reduce_last`）只放行沿最后一维、keepdim=True 的
    形态，所以这里拿到的 `dim` 必然是最后一维；仍按实参算而不是写死 -1，免得
    规则将来放宽了这里悄悄算错。
    """
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0].astype(np.float32)
    dim = args[1] if len(args) > 1 else -1
    keepdim = bool(args[2]) if len(args) > 2 else False
    axis = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    _write_result(hal, dpu_id, cmd, np.mean(x, axis=axis, keepdims=keepdim))


def _silu(x: np.ndarray) -> np.ndarray:
    """silu 走 `pim.lut` 查表，与矩阵乘融合路径读同一张 288 B 表。

    闭式 `x/(1+exp(-x))` 和查表不是同一个数，两边各算各的会让对拍绿在
    一个参考产物并不执行的公式上，所以这里不再保留闭式。
    """
    array = np.ascontiguousarray(x, dtype=np.float16)
    fn = _compiled_lut(tuple(array.shape), "silu")
    if fn is None:
        return kernels_pim._apply_activation(array, "silu")
    out = np.zeros(array.shape, dtype=np.float16)
    fn(array.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
    return out


def _compiled_lut(shape: tuple[int, ...], kind: str):
    """按形状编一个 `pim.lut` 查表激活；工具链不在位时返回 None。"""
    key = ("lut", shape, kind)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="lut", arg_shapes=[shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
            activation=kind,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def silu_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.silu` 的 DPU 内核：走 `pim.lut` 查表，不走闭式公式。"""
    (x,) = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, _silu(x))


def rmsnorm_kernel(hal, dpu_id: int, cmd) -> None:
    """RMSNorm：走 `pim.normalize`，不再拆成 pow/mean/rsqrt/mul 四步。"""
    x, gamma = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, _normalize(x, gamma))


def _normalize(x: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(x, dtype=np.float16)
    g = np.ascontiguousarray(gamma, dtype=np.float16)
    fn = _compiled_normalize(tuple(array.shape))
    if fn is None:
        return kernels_pim.normalize(array, g)
    out = np.zeros(array.shape, dtype=np.float16)
    fn(array.ctypes.data_as(ctypes.c_void_p),
       g.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p))
    return out


def _compiled_normalize(shape: tuple[int, ...]):
    """按形状编一个 `pim.normalize`；工具链不在位时返回 None。"""
    key = ("normalize", shape)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="normalize", arg_shapes=[shape, (shape[-1],)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def rope_kernel(hal, dpu_id: int, cmd) -> None:
    """RoPE：走 `pim.rope` 三相链，不再拆成 neg/mul/add。"""
    x, cos, sin = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, _rope(x, cos, sin))


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(x, dtype=np.float16)
    c = np.ascontiguousarray(cos, dtype=np.float16)
    s = np.ascontiguousarray(sin, dtype=np.float16)
    fn = _compiled_rope(tuple(array.shape))
    if fn is None:
        return kernels_pim.rope(array, c, s)
    out = np.zeros(array.shape, dtype=np.float16)
    fn(array.ctypes.data_as(ctypes.c_void_p),
       c.ctypes.data_as(ctypes.c_void_p),
       s.ctypes.data_as(ctypes.c_void_p),
       out.ctypes.data_as(ctypes.c_void_p))
    return out


def _compiled_rope(shape: tuple[int, ...]):
    """按形状编一个 `pim.rope`；工具链不在位时返回 None。"""
    key = ("rope", shape)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="rope", arg_shapes=[shape, shape, shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def _mirror(fn: Callable) -> Callable:
    """把一个 numpy 镜像（吃张量实参、吐张量）包成 HAL 内核。

    与 `_unary` 同口径：实参由 `_read_tensor_args` 按 `arg_kinds`/`arg_shapes`
    还原，结果按 payload 的 dtype 写回。非张量实参（轴号、形状、dim）原样传进去。
    """
    def kernel(hal, dpu_id: int, cmd) -> None:
        _write_result(hal, dpu_id, cmd, fn(*_read_tensor_args(hal, dpu_id, cmd)))

    return kernel


def softmax_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten._softmax` 的 DPU 内核：走算子编译器的 `pim.softmax` 五相链。

    **不是 `np.exp`**。设备侧的 softmax 是"求最大值 → 查表取指数 → 求和 →
    查表取倒数 → 逐元素乘"五相，指数是 288 B 表上插值出来的，本来就不是
    `expf`。在主机上现算等于把这个误差凭空抹掉，而对拍器比的就是这个误差。
    """
    (x,) = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, softmax(x))


def softmax(x: np.ndarray) -> np.ndarray:
    """对一个张量沿最后一维做 softmax：编译内核优先，镜像兜底。

    `pim.softmax` 只沿最后一维归约（相位链把轴写死在展开里），所以这里也只
    认最后一维；调用方给错轴会拿到一个形状对、语义错的结果，所以把轴写进函数
    名而不是参数。

    回退的是 `runtime.kernels_pim.softmax`——同一个算子的 numpy 镜像，仍然不是
    主机上现算。
    """
    array = np.ascontiguousarray(x, dtype=_SOFTMAX_DTYPE)
    cols = int(array.shape[-1])
    rows = int(array.size // cols)
    fn = _compiled_softmax(rows, cols)
    if fn is None:
        return kernels_pim.softmax(x)
    source = array.reshape(rows, cols)
    out = np.zeros((rows, cols), dtype=_SOFTMAX_DTYPE)
    fn(source.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p))
    return out.reshape(array.shape)


# softmax 的存储类型。相位链按 fp16 落盘（`phase_data.py` 的每个相位都截回
# fp16），编译内核也只认这一种。
_SOFTMAX_DTYPE = np.float16


def _compiled_softmax(rows: int, cols: int):
    """按 `[rows, cols]` 编一个 softmax（带缓存）；工具链不在位时返回 None。

    **只回退工具链缺失这一种**（评审 20260923 的 P1-5）。原来这里捕获
    `(ValueError, NotImplementedError, RuntimeError)` 三类一起回退镜像，于是
    「编译失败」与「编译成功且与镜像一致」给出同一个结果——`test_opcompiler_ops`
    那条「编译内核 vs numpy 镜像逐元素一致」的判据在编译静默失败时照样通过，
    因为两边比的是同一份镜像。实测 softmax 对 `1x16` / `4x7` / `3x1024` /
    `1x1` / `2x13` 全都编得出来，所以那两个宽泛的异常类型捕的不是形状边界，
    只是把真正的编译错误盖住了。

    环境缺失是真实边界（CI 上可能没重建 FlagTree），所以它仍回退镜像；
    编译器本身报的错一律往上抛。
    """
    key = (rows, cols)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="softmax", arg_shapes=[(rows, cols)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def matmul_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.matmul` / `aten.bmm`：交给 `matmul`。"""
    x, w = _read_tensor_args(hal, dpu_id, cmd)
    _write_result(hal, dpu_id, cmd, matmul(x, w))


def _compiled_matmul(a: np.ndarray, b: np.ndarray):
    """按两个操作数的形状编一个 fp16 `pim.matmul`；条件不满足时返回 None。

    只走二维：矩阵单元乘的就是那两个维度，`pim.matmul` 是整算子级 op，
    三维的 `bmm` 要先把头展开成逐头二维乘，那是展开 pass 的事。

    回退口径与 `_compiled_softmax` 一致：只回退 `ToolchainUnavailable`，
    编译器自己报的错往上抛。
    """
    if a.ndim != 2 or b.ndim != 2 or a.dtype != np.float16 or b.dtype != np.float16:
        return None
    key = ("matmul", a.shape, b.shape)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="matmul", arg_shapes=[tuple(a.shape), tuple(b.shape)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """二维 fp16 矩阵乘：编译内核优先，`x @ w` 兜底。

    设备侧的矩阵乘按 fp16 存、f32 累加（与 EmitC 的 `pim_f16_to_f32` /
    `pim_f32_to_f16` 一致），所以对拍取的就是这条路径。其它形状与 dtype
    （三维 bmm、fp32 图）没有对应的编译内核，退回同口径的 numpy 乘。
    """
    a16 = np.ascontiguousarray(a, dtype=np.float16) if a.dtype == np.float16 else a
    b16 = np.ascontiguousarray(b, dtype=np.float16) if b.dtype == np.float16 else b
    fn = _compiled_matmul(a16, b16)
    if fn is None:
        return a.astype(np.float32) @ b.astype(np.float32)
    out = np.zeros((a16.shape[0], b16.shape[1]), dtype=np.float16)
    fn(ctypes.c_void_p(a16.ctypes.data), ctypes.c_void_p(b16.ctypes.data),
       ctypes.c_void_p(out.ctypes.data))
    return out


def _compiled_mask(scores_shape: tuple[int, ...], mask_shape: tuple[int, ...]):
    """按分数与掩码的形状编一个 `pim.mask`；工具链不在位时返回 None。

    回退口径与 `_compiled_softmax` 一致：只回退 `ToolchainUnavailable`，
    编译器自己报的错往上抛。
    """
    key = ("mask", scores_shape, mask_shape)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="mask", arg_shapes=[scores_shape, mask_shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def masked_fill_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.masked_fill(x, mask, value)`：走编出来的 `pim.mask`。

    设备算子做的是 `scores + mask`——掩码在浮点域里是偏置，布尔掩码在这里折算
    成同一个加法（被掩的位置加 `value - scores`）。公式只有这一份，编译内核与
    镜像对的是同一笔加法。
    """
    scores, mask, value = _read_tensor_args(hal, dpu_id, cmd)
    scores = scores.astype(np.float32)
    offset = np.where(np.asarray(mask, dtype=bool),
                      np.float32(value) - scores, np.float32(0.0))
    _write_result(hal, dpu_id, cmd, add_mask(scores, offset))


def add_mask(scores: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """`scores + offset` 的加性掩码：编译内核优先，镜像兜底。

    `pim.mask` 做的就是这一笔加法（掩码在浮点域里是偏置），所以注意力把它
    遮在分数上、`aten.where` 把它加在别处，走的是同一个算子。
    """
    scores16 = np.ascontiguousarray(scores, dtype=np.float16)
    offset16 = np.ascontiguousarray(offset, dtype=np.float16)
    fn = _compiled_mask(scores16.shape, offset16.shape)
    if fn is None:
        return kernels_pim.mask(np.asarray(scores, dtype=np.float32),
                                np.asarray(offset, dtype=np.float32))
    out = np.zeros(scores16.shape, dtype=np.float16)
    fn(ctypes.c_void_p(scores16.ctypes.data),
       ctypes.c_void_p(offset16.ctypes.data),
       ctypes.c_void_p(out.ctypes.data))
    return out


def gather_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.embedding` 的 DPU 内核：编出来的 `pim.gather`，查表不走主机。

    查表是纯访存，在设备上做和在主机上做的区别是那 256 MiB 的嵌入表要不要搬
    过去——这正是编排器要规划的东西，所以不能回退主机 embedding。编不出来
    （没有 PIM pass 之类）才回退 `kernels_pim.gather`，那还是设备算子。
    """
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import compile_op, load_kernel

    # `aten.embedding` 的实参不止两个：`F.embedding` 会把 `padding_idx` 与
    # 后面三个开关一并带上（config 里 `pad_token_id` 非空时就在）。查表只用
    # 前两个，其余按定义不影响前向结果，多读一遍只会拿到用不上的缓冲。
    table, indices = _read_tensor_args(hal, dpu_id, cmd)[:2]
    ids = np.ascontiguousarray(indices, dtype=np.int32)
    expected = kernels_pim.gather(table, ids)
    key = ("gather", table.shape, ids.shape)
    fn = _COMPILED_KERNEL_CACHE.get(key)
    if fn is None:
        result = compile_op(OpCompileRequest(
            op="gather", arg_shapes=[table.shape, ids.shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
        ))
        fn = load_kernel(result)
        _COMPILED_KERNEL_CACHE[key] = fn
    out = np.zeros(expected.shape, dtype=np.float16)
    fn(ctypes.c_void_p(np.ascontiguousarray(table, np.float16).ctypes.data),
       ctypes.c_void_p(ids.ctypes.data), ctypes.c_void_p(out.ctypes.data))
    _write_result(hal, dpu_id, cmd, out)


def _compiled_dynamic_quant(shape: tuple[int, ...], group_size: int):
    """按形状与组宽编一个 `pim.dynamic_quant`；工具链不在位时返回 None。"""
    key = ("dynamic_quant", shape, group_size)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="dynamic_quant", arg_shapes=[shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
            group_size=group_size,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def dynamic_quant_kernel(hal, dpu_id: int, cmd) -> None:
    """动态量化的载体内核：走编出来的 `pim.dynamic_quant` 四相链。

    图上 `aten.alias` 只是承载量化字段的节点，数值上它必须真的把 fp16 量化成
    int8——恒等镜像会让这一步在设备上什么都不做。组宽按隐藏维能整除的最大
    2 的幂取，上限 128（Llama2 的组宽）。
    """
    (x,) = _read_tensor_args(hal, dpu_id, cmd)
    source = np.ascontiguousarray(x, dtype=np.float16)
    group_size = 128
    while source.size % group_size:
        group_size //= 2
    fn = _compiled_dynamic_quant(source.shape, group_size)
    if fn is None:
        _write_result(hal, dpu_id, cmd,
                      kernels_pim.dynamic_quant(source, group_size))
        return
    out = np.zeros(source.shape, dtype=np.int8)
    fn(ctypes.c_void_p(source.ctypes.data), ctypes.c_void_p(out.ctypes.data))
    _write_result(hal, dpu_id, cmd, out)


def where_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.where(cond, a, b)`：折算成同一个加性偏置。

    与 `masked_fill_kernel` 同理：设备算子 `pim.mask` 只有加法这一条路径，
    条件选择化成"被选中的位置加上 `a - b`"，公式不动。
    """
    def mirror(cond, a, b):
        base = b.astype(np.float32)
        offset = np.where(np.asarray(cond, dtype=bool),
                          a.astype(np.float32) - base, np.float32(0.0))
        return kernels_pim.mask(base, offset)

    _mirror(mirror)(hal, dpu_id, cmd)


def _compiled_view(op: str, arg_shapes: list, *, group_size: int | None = None):  # op 取 "transpose" / "reshape" / "concat"
    """按形状编一个视图类算子；工具链不在位时返回 None。"""
    key = (op, tuple(tuple(s) if isinstance(s, (list, tuple)) else s
                     for s in arg_shapes), group_size)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)
    try:
        result = compile_op(OpCompileRequest(
            op=op, arg_shapes=arg_shapes, group_size=group_size,
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16"))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def transpose_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.permute`：编出来的 `pim.transpose`，镜像兜底。"""
    args = _read_tensor_args(hal, dpu_id, cmd)
    x, order = args[0], tuple(args[1])
    fn = _compiled_view("transpose", [tuple(x.shape), order])
    if fn is None:
        _write_result(hal, dpu_id, cmd, kernels_pim.transpose(x, order))
        return
    out = np.zeros(tuple(x.shape[i] for i in order), dtype=np.float16)
    fn(ctypes.c_void_p(np.ascontiguousarray(x, np.float16).ctypes.data),
       ctypes.c_void_p(out.ctypes.data))
    _write_result(hal, dpu_id, cmd, out)


def transpose_int_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.transpose.int(x, dim0, dim1)`：换两轴，再交给同一个镜像。

    `pim.transpose` 收的是**全轴序**，所以这里先把两个轴对换成轴序。
    """
    def mirror(x, dim0, dim1):
        order = list(range(x.ndim))
        order[int(dim0)], order[int(dim1)] = order[int(dim1)], order[int(dim0)]
        return kernels_pim.transpose(x, tuple(order))

    _mirror(mirror)(hal, dpu_id, cmd)


def reshape_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.view` / `aten.reshape`：编出来的 `pim.reshape`，镜像兜底。"""
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0]
    raw = [int(d) for d in args[1]]
    # `-1` 是推断维：图上的 view 常写成 (-1, n)，契约要两侧元素数相同，
    # 所以在这里算出来再传给编译器。
    if raw.count(-1) == 1:
        known = 1
        for d in raw:
            if d != -1:
                known *= d
        raw[raw.index(-1)] = int(np.prod(x.shape)) // known
    shape = tuple(raw)
    fn = _compiled_view("reshape", [tuple(x.shape), shape])
    if fn is None:
        _write_result(hal, dpu_id, cmd, kernels_pim.reshape(x, shape))
        return
    out = np.zeros(shape, dtype=np.float16)
    fn(ctypes.c_void_p(np.ascontiguousarray(x, np.float16).ctypes.data),
       ctypes.c_void_p(out.ctypes.data))
    _write_result(hal, dpu_id, cmd, out)


def slice_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.slice.Tensor(x, dim, start, end, step)`：按维切一段。

    没有编译内核：切片改的是**视图的形状**，`pim.reshape` 只改形状不丢元素，
    两者不是同一个算子；按 reshape 编会把整段数据原样发出去而不报错。
    """
    def mirror(x, dim, start, end, step=1):
        axis = x.shape[int(dim)]
        lo = 0 if start is None else max(int(start), 0)
        hi = axis if end is None else min(int(end), axis)
        index = [slice(None)] * x.ndim
        index[int(dim)] = slice(lo, hi, 1 if step is None else int(step))
        return x[tuple(index)]

    _mirror(mirror)(hal, dpu_id, cmd)


def concat_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.cat` 的镜像：沿一根轴把若干张量接起来。

    实参形态与图上的 `aten.cat(tensors, dim)` 不同——命令按位置编码 `Node`
    实参，列表实参收不了（`partition.HOST_ONLY` 里写的正是这条理由）。所以这里
    按 `pim.concat(张量..., axis)` 的形态收实参。
    """
    args = _read_tensor_args(hal, dpu_id, cmd)
    tensors, axis = list(args[:-1]), int(args[-1])
    # `-1` 是末轴：图上的 cat 常写 dim=-1，契约要的是非负轴。
    if axis < 0:
        axis += tensors[0].ndim
    fn = _compiled_view("concat", [tuple(t.shape) for t in tensors], group_size=axis)
    if fn is None:
        _write_result(hal, dpu_id, cmd, kernels_pim.concat(tensors, axis))
        return
    out_shape = list(tensors[0].shape)
    out_shape[axis] = sum(t.shape[axis] for t in tensors)
    out = np.zeros(tuple(out_shape), dtype=np.float16)
    ptrs = [ctypes.c_void_p(np.ascontiguousarray(t, np.float16).ctypes.data)
            for t in tensors]
    fn(*ptrs, ctypes.c_void_p(out.ctypes.data))
    _write_result(hal, dpu_id, cmd, out)


def split_heads_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.split` 的 DPU 内核：走 `pim.split_heads`，不再是纯 numpy 切分。

    `pim.split_heads` 是"按头数均分"，所以这里要求能被份数整除——切不出整份
    就报错，不悄悄多切一份出来。
    """
    args = _read_tensor_args(hal, dpu_id, cmd)
    x, pieces, dim = args[0], int(args[1]), int(args[2]) if len(args) > 2 else 1
    axis = dim if dim >= 0 else dim + x.ndim
    if x.shape[axis] % pieces:
        raise ValueError(f"轴 {axis} 长 {x.shape[axis]} 分不出 {pieces} 个整份")
    fn = _compiled_split_heads(tuple(x.shape), axis, pieces)
    if fn is None:
        _write_result(hal, dpu_id, cmd, np.split(x, pieces, axis=axis))
        return
    head = x.shape[axis] // pieces
    piece_shape = list(x.shape)
    piece_shape[axis] = head
    outs = [np.zeros(tuple(piece_shape), dtype=np.float16) for _ in range(pieces)]
    ptrs = [ctypes.c_void_p(np.ascontiguousarray(x, np.float16).ctypes.data)]
    ptrs += [ctypes.c_void_p(o.ctypes.data) for o in outs]
    fn(*ptrs)
    _write_result(hal, dpu_id, cmd, outs)


def _compiled_split_heads(shape: tuple[int, ...], axis: int, num_heads: int):
    """按形状编一个 `pim.split_heads`；工具链不在位时返回 None。"""
    key = ("split_heads", shape, axis, num_heads)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="split_heads", arg_shapes=[shape],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype="float16",
            group_size=num_heads,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


# SDPA 的设备内核要 KV 缓存与层号，两者都是编译期才知道的，由
# `configure_sdpa_kv` 注入。
_SDPA_KV = None


def sdpa_kv_info(node, kv_specs, layer_of_node, np_dtype) -> dict | None:
    """把一个注意力节点要用的设备侧 KV 区域整理成可进命令 payload 的形式。

    编译期算好、烘进命令，而不是让内核去查全局：三种切分策略连着编译时，
    模块级的那份会被后一次覆盖，前面那些 plan 再执行就用了**别人**的 KV
    规格——实测解码结果因此错成另一个 token。
    """
    layer = layer_of_node.get(node.name)
    if layer is None:
        return None
    heads = []
    for dpu_id, spec in sorted(kv_specs.items()):
        if layer not in spec.layers:
            continue
        for head in spec.kv_heads:
            row = spec.head_dim * spec.dtype_bytes
            heads.append({
                "head": int(head),
                "dpu": int(dpu_id),
                "k_off": int(spec.kv_off[(layer, head, "k")]),
                "v_off": int(spec.kv_off[(layer, head, "v")]),
                "row": int(row),
            })
    if not heads:
        return None
    return {
        "layer": int(layer),
        "max_seq": int(next(iter(kv_specs.values())).max_seq),
        "head_dim": int(next(iter(kv_specs.values())).head_dim),
        "np_dtype": str(np_dtype),
        "heads": heads,
    }


def _kv_row_write(hal, owner: int, base: int, slot_index: int,
                  row: np.ndarray, max_seq: int) -> None:
    """把一行 K 或 V 写进缓存：走 `pim.kv_cache` 的散写相，不再直连 HAL。

    散写的目的行由**索引**给出（校验器要求 scatter 必须有 indices，`pos` 不是
    目的地），所以缓存基址传区域头、行号传 `slot_index`，落点与直写
    `base + slot*head_dim` 一致。

    编译不出来（工具链缺失）才回退 `hal.write_local`——回退口径与其它
    `_compiled_*` 一致，只挡工具链缺失这一种。
    """
    row = np.ascontiguousarray(row)
    fn = _compiled_kv_cache(tuple(row.shape), max_seq * row.shape[-1], row.dtype)
    if fn is None:
        hal.write_local(owner, base + slot_index * row.nbytes, row)
        return
    slot = np.zeros((1,), dtype=np.int16)
    slot[0] = slot_index
    fn(ctypes.c_void_p(row.ctypes.data),
       ctypes.c_void_p(hal.raw_mram_ptr(owner) + base),
       ctypes.c_int32(0),
       ctypes.c_void_p(slot.ctypes.data))


def _compiled_kv_cache(value_shape: tuple[int, ...], cache_elems: int,
                       dtype: np.dtype):
    """按一行新值的形状编一个 `pim.kv_cache` 散写；工具链不在位时返回 None。

    元素类型跟着 KV 区域的 dtype 走（运行时存 fp16），不是量化后的 i8。
    """
    key = ("kv_cache", value_shape, cache_elems, str(dtype))
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="kv_cache", arg_shapes=[value_shape, (cache_elems,)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype=str(dtype),
            group_size=1,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


def sdpa_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.scaled_dot_product_attention` 的 DPU 内核。

    拆头之后这条不再是主路径：QKᵀ、掩码、softmax、PV 各自是设备命令。
    留着是因为没拆头的图（直接 `build_execution_plan`、不经 `compile_llama2`）
    仍会遇到整体 SDPA 节点，那时 KV 缓存的读写还在这里。
    """
    info = cmd.payload.get("sdpa")
    if info is None:
        raise RuntimeError(
            "注意力命令里没有 KV 区域信息；build_execution_plan 要传 sdpa_info_of")
    np_dtype = np.dtype(info["np_dtype"])
    layer = int(info["layer"])
    max_seq = int(info["max_seq"])
    head_dim = int(info["head_dim"])

    args = _read_tensor_args(hal, dpu_id, cmd)
    q, k, v = args[0], args[1], args[2]
    scale = 1.0 / np.sqrt(q.shape[-1])
    num_heads, tq = q.shape[1], q.shape[2]

    pos = hal.bound_pos if hal.bound_pos is not None else 0
    for entry in info["heads"]:
        head, owner = entry["head"], entry["dpu"]
        if head >= num_heads:
            continue
        for t in range(tq):
            row = pos + t
            _kv_row_write(hal, owner, entry["k_off"], row,
                          np.ascontiguousarray(k[0, head, t], np_dtype), max_seq)
            _kv_row_write(hal, owner, entry["v_off"], row,
                          np.ascontiguousarray(v[0, head, t], np_dtype), max_seq)

    valid_len = pos + tq
    out = np.zeros((1, num_heads, tq, head_dim), dtype=np.float32)
    for entry in info["heads"]:
        head, owner = entry["head"], entry["dpu"]
        if head >= num_heads:
            continue
        k_hist = hal.read_local(owner, entry["k_off"], (max_seq, head_dim), np_dtype)
        v_hist = hal.read_local(owner, entry["v_off"], (max_seq, head_dim), np_dtype)
        # 掩码优先用图上传入的第 4 个实参。按位置重算只是没有这个实参时的兜底：
        # 图上换成滑动窗口或自定义 bias 时，重算会静默给出错误结果。
        graph_mask = args[3] if len(args) > 3 else None
        for t in range(tq):
            # 三段各自走设备算子，不在这一处现算：`pim.matmul` 两次、`pim.mask`
            # 一次、`pim.softmax` 一次。它们的数值口径与单测对拍的那条完全一样，
            # 只是调用点在注意力里。
            scores = matmul(k_hist, q[0, head, t][:, None]).reshape(-1)
            scores = scores.astype(np.float32) * scale
            # 图上的掩码只有在长度与分数一致时才用得上：分数含 KV 历史
            # （长 max_seq），而图上的掩码只覆盖当前 query，短一截时补零
            # 会把历史位置当成可见，结果就错了。长度对不上就退回按位置重算。
            if graph_mask is not None and graph_mask.shape[-1] == scores.shape[0]:
                row = graph_mask.reshape(-1, graph_mask.shape[-1])
                mask_row = row[min(t, row.shape[0] - 1)].astype(scores.dtype)
            else:
                mask = (prefill_mask(t + 1, max_seq) if tq > 1
                        else decode_mask(valid_len - 1, max_seq))
                mask_row = mask[t] if tq > 1 else mask
            weights = softmax(add_mask(scores, mask_row).astype(np.float32))
            out[0, head, t] = matmul(weights[None, :], v_hist).reshape(-1)
    _write_result(hal, dpu_id, cmd, out.astype(np_dtype))


def unsqueeze_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.unsqueeze.default(x, dim)`：在 dim 处插一根长度为 1 的轴。"""
    def mirror(x, dim):
        return np.expand_dims(x, int(dim))
    _mirror(mirror)(hal, dpu_id, cmd)


def convert_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.to.dtype` / `to.dtype_layout`：只换元素类型。

    `to.dtype` 的目标类型是第二位置实参；`to.dtype_layout` 走 kwargs，
    命令编码只收位置实参，那种形态按输出缓冲的 dtype 转。
    """
    args = _read_tensor_args(hal, dpu_id, cmd)
    x = args[0]
    if len(args) >= 2 and args[1] is not None and not isinstance(args[1], np.ndarray):
        dtype = _NP_DTYPE[str(args[1])]
    else:
        out = cmd.payload.get("dtype", "float32")
        key = f"torch.{out}" if not str(out).startswith("torch.") else str(out)
        dtype = _NP_DTYPE.get(key, np.dtype(out))
    _write_result(hal, dpu_id, cmd, convert(x, dtype))


def convert(source: np.ndarray, dtype) -> np.ndarray:
    """换元素类型：编译内核优先，`astype` 兜底。

    `pim.convert` 的源侧按 fp16 读（相位缓冲按 fp16 落盘），所以只有 fp16
    输入才走编译内核；别的 dtype 没有对应的内核形态，退回同一个语义的
    `astype`。

    两侧类型相同就是恒等，**不编**：图上 `to.dtype` 有一半是这种空转（同一个
    类型再转一次），而 `pim.convert` 的 verifier 明确拒绝同类型——那不是一次
    转换，硬发出去只会得到一个语义上不存在的算子。
    """
    if np.dtype(dtype) == source.dtype:
        return source
    fn = _compiled_convert(source.shape, source.dtype.name, np.dtype(dtype).name)
    if fn is None:
        return kernels_pim.convert(source, dtype)
    src = np.ascontiguousarray(source)
    out = np.zeros(src.shape, dtype=dtype)
    fn(ctypes.c_void_p(src.ctypes.data), ctypes.c_void_p(out.ctypes.data))
    return out


def _compiled_convert(shape: tuple[int, ...], source: str, target: str):
    """按形状与两侧类型编一个 `pim.convert`；没有对应形态时返回 None。"""
    supported = ("float16", "float32", "int8")
    if source not in supported or target not in supported or source == target:
        return None
    key = ("convert", tuple(shape), source, target)
    if key in _COMPILED_KERNEL_CACHE:
        return _COMPILED_KERNEL_CACHE[key]

    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from opcompiler_bridge.driver import (
        ToolchainUnavailable, compile_op, load_kernel)

    try:
        result = compile_op(OpCompileRequest(
            op="convert", arg_shapes=[tuple(shape)],
            hardware=DEFAULT_HARDWARE_CONFIG, dtype=source,
            out_dtype=target,
        ))
    except ToolchainUnavailable:
        _COMPILED_KERNEL_CACHE[key] = None
        return None
    fn = load_kernel(result)
    _COMPILED_KERNEL_CACHE[key] = fn
    return fn


_NP_DTYPE = {
    "torch.int8": np.int8,
    "torch.float16": np.float16,
    "torch.float32": np.float32,
}


@dataclass(frozen=True)
class KernelEntry:
    """一个 aten 目标的实现：numpy 镜像 + 可选的编译内核。

    `mirror` 任何形状都能跑，是"缺名字就报错"这条判据的落点——设备侧算子少了
    它，后端 `submit` 时就会找不到内核。`compiled` 是本模块里编译包装的函数名，
    有值表示这个名字有算子编译器编出来的 `.so` 可用。
    """

    mirror: Callable
    compiled: str | None = None


# 设备侧算子的持有表：aten 目标 -> 一对实现。缺名字时后端 `submit` 找不到
# 内核，直接失败，所以设备侧算子必须都在这张表里。
#
# `compiled=None` 表示这个名字目前走 numpy 镜像。视图、切片、cat 这些没有
# 对应的编译内核；加、乘、减、softmax、掩码、动态量化、gather 在各自的内核
# 函数里调 `compile_op`，不经过这个字段。
_LINEAR = str(torch.ops.aten.linear.default)

_KERNELS = {
    _LINEAR: KernelEntry(linear_kernel, "compiled_linear_kernel"),
    str(torch.ops.aten.add.Tensor): KernelEntry(add_kernel),
    str(torch.ops.aten.mul.Tensor): KernelEntry(mul_kernel),
    str(torch.ops.aten.sub.Tensor): KernelEntry(sub_kernel),
    str(torch.ops.aten.div.Tensor): KernelEntry(div_kernel),
    str(torch.ops.aten.tanh.default): KernelEntry(tanh_kernel),
    # RMSNorm 拆开后的三步、RoPE 的取负、MLP 的 SiLU——原来这些全落在主机上。
    str(torch.ops.aten.pow.Tensor_Scalar): KernelEntry(pow_kernel),
    str(torch.ops.aten.mean.dim): KernelEntry(mean_dim_kernel),
    str(torch.ops.aten.rsqrt.default): KernelEntry(_unary(lambda x: 1.0 / np.sqrt(x))),
    str(torch.ops.aten.neg.default): KernelEntry(_unary(np.negative)),
    str(torch.ops.aten.silu.default): KernelEntry(silu_kernel),
    str(torch.ops.aten.relu.default): KernelEntry(_unary(lambda x: np.maximum(x, 0.0))),
    str(torch.ops.aten.sigmoid.default): KernelEntry(
        _unary(lambda x: 1.0 / (1.0 + np.exp(-x)))),
    str(torch.ops.aten.exp.default): KernelEntry(_unary(np.exp)),
    str(torch.ops.aten.sqrt.default): KernelEntry(_unary(np.sqrt)),
    str(torch.ops.aten.reciprocal.default): KernelEntry(_unary(np.reciprocal)),
    # 算子级 mnemonic 对应的 aten（方案 4.8 的映射表）。
    str(torch.ops.aten.matmul.default): KernelEntry(matmul_kernel),
    str(torch.ops.aten.bmm.default): KernelEntry(matmul_kernel),
    str(torch.ops.aten._softmax.default): KernelEntry(softmax_kernel),
    str(torch.ops.aten.masked_fill.Tensor): KernelEntry(masked_fill_kernel),
    str(torch.ops.aten.where.self): KernelEntry(where_kernel),
    str(torch.ops.aten.split.Tensor): KernelEntry(split_heads_kernel),
    str(torch.ops.aten.slice.Tensor): KernelEntry(slice_kernel),
    str(torch.ops.aten.cat.default): KernelEntry(concat_kernel),
    str(torch.ops.aten.embedding.default): KernelEntry(gather_kernel),
    str(torch.ops.aten.scaled_dot_product_attention.default): KernelEntry(sdpa_kernel),
    # `alias` 是动态量化的载体：图上真正的量化发生在 `pim.dynamic_quant`
    # 的四相流水线里，载体本身不再是恒等。
    str(torch.ops.aten.alias.default): KernelEntry(dynamic_quant_kernel),
    str(torch.ops.aten.view.default): KernelEntry(reshape_kernel),
    str(torch.ops.aten.reshape.default): KernelEntry(reshape_kernel),
    str(torch.ops.aten.transpose.int): KernelEntry(transpose_int_kernel),
    str(torch.ops.aten.permute.default): KernelEntry(transpose_kernel),
    str(torch.ops.aten.unsqueeze.default): KernelEntry(unsqueeze_kernel),
    str(torch.ops.aten.to.dtype): KernelEntry(convert_kernel),
    str(torch.ops.aten.to.dtype_layout): KernelEntry(convert_kernel),
}
# 下面这些 aten 目标 **故意不在表里**：它们仍在主机侧口径里，放进这张表
# 就是按构造永不执行的死代码。反向闭合测试
# `test_no_kernel_is_registered_for_a_host_only_op` 守着这条。
#
#
#   view / reshape / transpose / permute / unsqueeze / to.dtype
#              已下设备（见 `_KERNELS`）。
# `concat_kernel` / `slice_kernel` / `gather_kernel` 仍留给算子编译器按
# mnemonic 调，不经过这张 aten 表。


def register_all(hal, *, use_compiled_linear: bool = False) -> None:
    """向后端注册白名单内核。

    注册的是镜像；`use_compiled_linear` 只管线性的位置，开着就换成 A 路编出来的
    `compiled_linear_kernel`——`test_opcompiler_linear.py` 要同时跑这两种对拍。
    其余名字的编译内核不受它控制：那条路没有"A 路对照"这回事。
    """
    for name, entry in _KERNELS.items():
        compiled = entry.compiled
        if name == _LINEAR and not use_compiled_linear:
            compiled = None
        hal.register_kernel(name, globals()[compiled] if compiled else entry.mirror)
