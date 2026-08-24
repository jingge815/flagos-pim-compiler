"""问题 6（数值代填半）：DPU 白名单算子的 NumPy 镜像 kernel（方案问题 4/5 的
真实边界——FlagTree 的 pim mlir 目前只到 MLIR 文本，无 LLVM 下降、无二进制、
无 ABI，genesim 是纯离线成本模拟器，两者都不做真实数值执行；本仓数值闭环只能
靠这层 NumPy 镜像 kernel，在 `backend/dpu_sdk.py` 的假硬件上跑）。

每个 kernel 是 `fn(hal, dpu_id, cmd) -> None`，签名对齐 `backend/hal_numpy.py`
的 `KernelStubRegistry`（`register_kernel(name, fn)` + launch 时
`kernel(hal, dpu_id, cmd)`）。四个白名单算子（`graph/partition.py`
`DPU_LOWERABLE`）各一个，覆盖 `linear`/`add.Tensor`/`mul.Tensor`/`tanh`。

payload 契约（`runtime/exec_plan_gen.py` 生成 launch 命令时写入，本模块只读）：
`arg_kinds`（按 `node.args` 顺序，`"tensor"` 或字面量本身）、`arg_shapes`
（张量参数对应位置的本地分片 shape，非张量位置为 `None`）、`dtype`（输出
dtype 名）、`out_shape`（本地输出 shape）。张量参数按 `arg_kinds` 里
`"tensor"` 出现的顺序对应 `cmd.reads` 的顺序（`exec_plan_gen` 保证两者同序）。

`register_all(hal)` 一次性注册全部四个 kernel，供测试/编排器调用。

**为什么导入时要限制 BLAS 线程数**。8 个 DPU 各自的 `launch` 命令由
`ThreadPoolExecutor` 并发调度（问题 6 验证 3b 要求的真实并行），而
`linear_kernel` 的 `x @ w.T` 走 NumPy 的 OpenBLAS 后端，OpenBLAS 自己又开
一个内部线程池（`DYNAMIC_ARCH MAX_THREADS=64`）——8 条 DPU 线程同时各自
触发一次多线程矩乘会互相抢占 CPU，导致某些矩乘内部的归约顺序在不同次运行
间不一致，浮点结果因此不可复现。host 节点（RMSNorm 的 `pow`/`mean`/`rsqrt`
等）经 `exec_plan_gen` 的通用路径调用真实 torch 算子，torch 走的是 MKL
（与 numpy 的 OpenBLAS 是两套独立后端），所以两侧都要限制：
`_limit_blas_threads(1)` 管 OpenBLAS，`torch.set_num_threads(1)` 管 MKL。
让"并行"只发生在问题 6 编排器自己的 DPU 级线程池这一层，BLAS 内部退化为
单线程——不影响验证 3b 要验证的"8 台 DPU 真的并发执行"这件事本身。

需要说明的是：这一项**不是** decode 循环那个间歇性 NaN 的根因。那个 bug 的
真正原因是 `runtime/exec_plan_gen.py` 里 WAR 依赖被静默丢弃（两张表键方向
相反），限制线程数之后仍会失败，详见 `docs/executor-20260824.md`。本模块这两行只
负责让数值可复现，不负责依赖正确性。
"""

from __future__ import annotations

import ctypes
import glob
import os

import numpy as np
import torch


def _limit_blas_threads(n: int = 1) -> bool:
    """把 NumPy 底层 OpenBLAS 的内部线程数钳到 `n`（默认 1，见模块 docstring）。

    多个 DPU 线程各自触发的矩乘不需要 BLAS 自己再开线程池——DPU 级并行已经
    是问题 6 编排器自己的线程池给的。找不到 OpenBLAS 库（换了别的 BLAS 后端）
    时安静跳过，不影响正确性，只是退回默认线程数（可能变慢或数值上不够
    确定，但不是本模块能控制的环境差异）。
    """
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

# torch 走 MKL（不是 OpenBLAS，上面那段钳的是 numpy 的 OpenBLAS，两套后端
# 各自独立）。host 节点（RMSNorm 的 pow/mean/rsqrt、SDPA 兜底等）经
# `runtime/exec_plan_gen.py` 的通用路径直接调 `node.target`，落在真实 torch
# 算子上；这些 host_op 与 8 条 DPU 线程共享同一个 `ThreadPoolExecutor`，若
# MKL 自己也开多线程，会出现与 OpenBLAS 那一侧相同的问题（真实复现：只
# 限制 OpenBLAS、不限制 MKL 时，decode 循环仍偶发数值不确定）。
# `torch.set_num_threads` 是官方运行时 API，不需要 ctypes。
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
    """`aten.linear(x, w)`：Y = X @ W.T（方案附录 A 的两层 Linear 即本算子）。"""
    x, w = _read_tensor_args(hal, dpu_id, cmd)
    y = x.astype(np.float32) @ w.astype(np.float32).T
    _write_result(hal, dpu_id, cmd, y)


def add_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.add.Tensor(x, y_or_scalar)`：逐元素加（方案二.(8) 逐元素行）。"""
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


def register_all(hal) -> None:
    """把全部白名单 kernel 注册进一个 `NumpyBackend`（`launch` 命令按
    `payload["kernel"]` 字符串查表，字符串即 `str(node.target)`）。"""
    for name, fn in _KERNELS.items():
        hal.register_kernel(name, fn)
