"""定义用于生成 PIM 算子代码的分块 Triton `linear` 内核。"""

from __future__ import annotations

import triton
import triton.language as tl

# 三个方向都用固定大小的分块。M 也分块，才能接受任意序列长度。
DEFAULT_BLOCK_M = 16
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """计算 `y = x @ w.T`，其中输入形状为 `(M, K)` 和 `(N, K)`。

    三个维度都按分块遍历。最后一块可能不满，用掩码丢掉越界的元素，
    这样维度不必被分块整除，也不必是 2 的幂。
    """
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        for n0 in range(0, N, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                x_off = offs_m[:, None] * K + offs_k[None, :]
                w_off = offs_n[:, None] * K + offs_k[None, :]
                x_blk = tl.load(x_ptr + x_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                w_blk = tl.load(w_ptr + w_off, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
                acc = tl.dot(x_blk, tl.trans(w_blk), acc, allow_tf32=False)
            o_off = offs_m[:, None] * N + offs_n[None, :]
            tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty),
                     mask=mask_m[:, None] & mask_n[None, :])


def _clamp_block(full: int, want: int) -> int:
    """取不超过 `want` 和 `full` 的最大 2 的幂，作为分块大小。

    不再要求分块整除维度：最后一块由内核里的掩码处理。维度本身是 2 的幂
    且不超过 `want` 时直接取整维，否则取不超过两者的最大 2 的幂。
    """
    if full <= want and (full & (full - 1)) == 0:
        return full
    block = 1 << (min(want, full).bit_length() - 1)
    return max(block, 1)


def pick_blocks(K: int, N: int, M: int = 1) -> tuple[int, int, int]:
    """返回 `(BLOCK_M, BLOCK_N, BLOCK_K)`。"""
    return (
        _clamp_block(M, DEFAULT_BLOCK_M),
        _clamp_block(N, DEFAULT_BLOCK_N),
        _clamp_block(K, DEFAULT_BLOCK_K),
    )


def make_kernel_launcher(M: int, K: int, N: int):
    """返回固定 M、K、N 的 Triton 内核启动函数。"""
    block_m, block_n, block_k = pick_blocks(K, N, M)

    def launch(x, w, out):
        return linear_kernel[(1,)](
            x, w, out, M=M, K=K, N=N,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_stages=NUM_STAGES,
        )

    return launch
