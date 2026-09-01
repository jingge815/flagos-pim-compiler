"""定义用于生成 PIM 算子代码的分块 Triton `linear` 内核。"""

from __future__ import annotations

import triton
import triton.language as tl

# N 和 K 方向使用固定大小的分块。
DEFAULT_BLOCK_N = 512
DEFAULT_BLOCK_K = 32

# 使用单个软件流水阶段。
NUM_STAGES = 1


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """计算 `y = x @ w.T`，其中输入形状为 `(M, K)` 和 `(N, K)`。"""
    offs_m = tl.arange(0, M)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x_off = offs_m[:, None] * K + offs_k[None, :]
            w_off = offs_n[:, None] * K + offs_k[None, :]
            x_blk = tl.load(x_ptr + x_off)
            w_blk = tl.load(w_ptr + w_off)
            acc = tl.dot(x_blk, tl.trans(w_blk), acc, allow_tf32=False)
        o_off = offs_m[:, None] * N + offs_n[None, :]
        tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty))


def _pick(full: int, want: int, floor: int) -> int:
    """选一个能整除 `full`、不超过 `want`、不小于 `floor` 的 2 的幂分块大小。"""
    block = min(want, full)
    while block > floor and full % block != 0:
        block //= 2
    if full % block != 0:
        raise ValueError(
            f"找不到能整除 {full} 的分块大小（下界 {floor}）；这条链路要求 "
            f"M/K/N 都是 2 的幂，driver.py 应该已经拦住了其它 shape"
        )
    return block


def pick_blocks(K: int, N: int) -> tuple[int, int]:
    """返回 `(BLOCK_N, BLOCK_K)`。`tl.dot` 要求参与的 K 维 >= 16，故 K 的下界
    是 16；N 方向没有这个约束，但小于 16 也没意义，取同一个下界。"""
    return _pick(N, DEFAULT_BLOCK_N, 16), _pick(K, DEFAULT_BLOCK_K, 16)


def make_kernel_launcher(M: int, K: int, N: int):
    """返回固定 M、K、N 的 Triton 内核启动函数。"""
    block_n, block_k = pick_blocks(K, N)

    def launch(x, w, out):
        return linear_kernel[(1,)](
            x, w, out, M=M, K=K, N=N,
            BLOCK_N=block_n, BLOCK_K=block_k, num_stages=NUM_STAGES,
        )

    return launch
