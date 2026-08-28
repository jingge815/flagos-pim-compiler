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
import threading

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


def tasklet_linear_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.linear(x, w)` 的多 tasklet 版本：按 M 维（batch*seq 展平后的行数）
    切分给 `cmd.num_tasklets` 个 tasklet，仿 downmem `GEMV.c` 的
    `my_rows = num_rows / NR_TASKLETS` 模式。

    每个 tasklet 顺序模拟跑完自己的行区间（`backend/hal_numpy.py` 的
    `HazardTracker`：不用真 pthread，按固定顺序依次执行，保持数值确定可
    复现），每次读/写前调 `hal.record_access` 记录地址区间——两个 tasklet
    若因为切分算错、行区间重叠，`record_access` 会立刻抛
    `TaskletHazardError`。全部 tasklet 跑完后调一次 `hal.barrier()`，对应
    真实硬件上"落盘前的一次全 tasklet 同步"。

    `w`（权重）不按 tasklet 切分——全部 tasklet 共享同一份只读权重，只有
    `x`/输出按 M 行切分，这与 FlagTree 泛化后 `pim-lower-to-emitc` pass 生成
    的 C 代码的切分方式一一对应（方案 2.2 节）。
    """
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


# (op, arg_shapes 元组化) -> ctypes 函数对象，进程内缓存，避免重复
# compile_op()/dlopen（compile_op 自己在磁盘按同一个 key 缓存 .so，这里只是不
# 想每次 launch 都重新构造 ctypes 函数签名）。
_COMPILED_KERNEL_CACHE: dict[tuple, object] = {}

# `NumpyBackend` 每台 DPU 一个线程并发 launch（问题 6 验证 3b 要求的真实并行）。
# 8 台 DPU 第一次同时遇到同一个新 shape 时，若不加锁，会有多个线程同时判定
# _COMPILED_KERNEL_CACHE 里没有、都去调 compile_op——多个 gcc 进程并发写同一个
# .so 路径，非原子写导致文件损坏，ctypes 加载报 `undefined symbol`（实测：
# 8 卡真实 decode 跑到第一次遇到新 shape 时稳定复现）。这把锁只序列化"缓存未
# 命中时的编译"这一段，命中之后（绝大多数调用）直接读字典不用等锁。
_COMPILE_LOCK = threading.Lock()


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _compiled_linear_supports(arg_shapes, dtype: str = "float32") -> bool:
    """`opcompiler_bridge/kernel_src.py` 的 kernel 用 `tl.arange` 直接生成整块
    索引（M/N 方向不切分，K 方向按 BLOCK_K 分块），要求 M/K/N 都是 2 的幂
    （`tl.arange` 的范围必须是 2 的幂），且 K>=16（`tl.dot` 的硬约束）。

    `arg_shapes[0]`（x）真实图上带 batch 维（如 `(1, 6, 4096)`），先展平成
    `(M, K)` 再判断——见 `contracts/op_contract.py` 的 `flatten_leading_dims`。
    这是从真实端到端测试插桩抓出来的：最早一版误以为 x 恒为二维，实测直接在
    这里 `ValueError: too many values to unpack` 崩掉，`linear` 的每一次调用
    都会经过这里，不修就是 100% 崩，不是极端情况。

    真实 llama2-7b 里 hidden_size=4096（q/k/v/o_proj 的展平后 M/K/N，decode
    时 M=1 也是 2 的幂）满足这两条，但 intermediate_size=11008（MLP 的
    gate/up/down_proj）和 vocab_size=32000（lm_head）都不是 2 的幂——这两类
    算子第一期垂直切片不覆盖，`compiled_linear_kernel` 遇到时退回
    `linear_kernel`（纯 NumPy），不是缺陷，是本期明确的范围边界（见
    docs/opcompiler_bridge-20260825.md）。

    注意 prefill 的 M 是 prompt 长度（本仓测试里是 6），一般不是 2 的幂，所以
    prefill 的 linear 基本都走 fallback；decode 的 M=1 是 2 的幂，才是编译产物
    真正跑到的地方。
    """
    from contracts.op_contract import flatten_leading_dims

    if dtype not in ("float16", "float32"):
        return False
    m, k = flatten_leading_dims(arg_shapes[0])
    n = arg_shapes[1][0]
    return k >= 16 and _is_pow2(m) and _is_pow2(k) and _is_pow2(n)


def compiled_linear_kernel(hal, dpu_id: int, cmd) -> None:
    """`aten.linear(x, w)` 的算子编译器产物版本——contracts/op_contract.py
    定义的第一期垂直切片：把 `linear_kernel` 换成 opcompiler_bridge 编译出的
    `.so`（TTIR -> pim mlir -> EmitC -> C -> gcc），在真实 MRAM 内存上原地
    计算，而不是拷贝出来用 NumPy 算。

    shape 超出 `_compiled_linear_supports` 覆盖范围（见其 docstring）时静默
    退回 `linear_kernel`——上层（图编译器/exec_plan_gen）不需要关心某个
    `linear` 节点最终是算子编译器产物执行还是 NumPy 执行，数值结果一致。

    K>=16 且 M/K/N 都是 2 的幂时：数值上与 `linear_kernel` 一致（两者都固定
    用 float32 计算，MRAM 存储按 `dtype` 走，见 `contracts/op_contract.py`
    顶部关于 dtype 字段的说明）。

    不需要预先切 triton 环境——这台机器上的 triton 现在只有一份，默认自带
    PIM pass 支持（`opcompiler_bridge/driver.py` 模块 docstring 有说明为什么
    不能再维护两份：真实端到端场景下 transformers/torch 会在任何人切环境
    之前就 import 到普通版，这是实测到的真实故障，不是理论风险）。
    """
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
    # dtype、num_tasklets 和硬件字典都进 key：同 shape 的 f16/f32 产物元素宽度
    # 不同，混用会静默算错；num_tasklets 不同时生成的 C 代码结构不同；硬件
    # 配置不同时，编译产物需要满足的约束不同。
    key = ("linear", arg_shapes, dtype, tuple(hardware.to_payload().items()))
    fn = _COMPILED_KERNEL_CACHE.get(key)
    if fn is None:
        # 缓存未命中才需要排队：多台 DPU 线程第一次同时遇到同一个新 shape 时，
        # 不加锁会有多个 gcc 进程并发写同一个 .so 路径导致文件损坏（见
        # _COMPILE_LOCK 的注释）。拿到锁后要重新查一次缓存——等锁的这段时间
        # 里，先拿到锁的那个线程可能已经编译完并把结果放进去了。
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


def register_all(hal, *, use_compiled_linear: bool = False) -> None:
    """把全部白名单 kernel 注册进一个 `NumpyBackend`（`launch` 命令按
    `payload["kernel"]` 字符串查表，字符串即 `str(node.target)`）。

    `use_compiled_linear=True` 时把 `linear` 换成 `compiled_linear_kernel`
    （算子编译器产物）——默认关闭，因为它第一次调用会触发 GPU 编译
    （`opcompiler_bridge.driver.compile_op`），跟现有测试/编排器默认的纯
    numpy 路径要求不同的环境（GPU + `prepare_triton_env(pim=True)`）。
    """
    kernels = dict(_KERNELS)
    if use_compiled_linear:
        kernels[str(torch.ops.aten.linear.default)] = compiled_linear_kernel
    for name, fn in kernels.items():
        hal.register_kernel(name, fn)
