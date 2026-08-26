"""驱动编译用的最小 `linear` Triton kernel（K 方向分块，无 mask）。

跟 FlagGems `ops/linear.py` 的 `linear_kernel` 比，两处刻意的不同：

**不做 autotune、不做 mask**。`LowerPIMSingleTasklet`（FlagTree 新 pass）要求
每次 DMA 的地址模式能被 `pim-explicit-dma` 的指针分析证明（`base_arg` 属性
存在），FlagGems 版本的 `mask=input_mask` 一旦 M/N/K 不是 BLOCK 整数倍就引入
掩码分支，这条分析证不出来。本仓契约（`contracts/op_contract.py`）本来就是
单算子、单 DPU 本地视图、静态 shape，grid 恒为 `(1,)`，不需要跨线程块切分。

**但 K 方向必须分块**。整块一次算完的写法在真实 llama2-7b 尺度上过不去：
`w` 的本地分片是 `(512, 4096)`，`offs_n[:, None] * K + offs_k[None, :]` 会
产生 512×4096≈210 万元素的 offset 张量，超过 Triton 自身的单张量元素数上限
`2^20`（实测报错 `numel (2097152) exceeds triton maximum tensor numel
(1048576)`）。所以按 `BLOCK_K` 切 K，逐块累加——这也正是真实 PIM 设备上
WRAM 装不下整个 K 时必须做的事，`pim.wram_alloc` 的尺寸因此也跟着降下来。

分块引入的 `scf.for`（累加器是循环携带的 tensor、load 地址依赖归纳变量）由
新 pass 识别并**折叠回一个完整 K 的平坦归约**——因为 numpy 后端上 MRAM 直接
可寻址、WRAM 搬运被省略，分块就只剩"地址怎么算"的意义，数学上等价于不分块。
详见 `LowerPIMSingleTasklet.cpp` 文件头的 "K-tiling" 一节。

offset 计算必须先把完整 offset 张量算完再一次性 `ptr + off`（而不是对行/列
offset 各做一次 `tt.addptr`）：`pim-explicit-dma` 里 `traceBaseArg`
（`ExplicitDMA.cpp`）只认「`tt.addptr` → `tt.splat` → 函数参数」一跳到底，
多一层 `tt.addptr` 就追不到底，产出的 DMA 缺 `base_arg`，新 pass 直接报错。

只读参考 FlagGems 的算法（`y = x @ w.T`），不导入 FlagGems、不引入依赖。
"""

from __future__ import annotations

import triton
import triton.language as tl

# N 和 K 两个方向都要分块，两个上限各自卡在不同地方（都是实测撞出来的）：
#
# - K 方向：不分块时 `w` 的 offset 张量是 N×K 个元素，llama2-7b 的 512×4096
#   超过 Triton 单张量 `2^20` 元素上限（报 `numel (2097152) exceeds triton
#   maximum tensor numel (1048576)`）。
# - N 方向：`tl.dot` 的操作数 tile 走 GPU shared memory，o_proj 的本地分片
#   N=4096 即使 BLOCK_K=16 也要 262208 B，超过 A100 每 CTA 的 166912 B
#   （报 `OutOfResources: shared memory`）。所以 N 也必须切。
#
# BLOCK_N=512 / BLOCK_K=32 时两类 shape（q/k/v 的 N=512、o_proj 的 N=4096）
# 的 tile 都是 67712 B，留了一倍余量。这条链路的目标是数值正确与可编译，不是
# GPU 性能（真正执行发生在编译出的 C 里，见 docs/opcompiler_bridge-20260825.md），所以
# 不为性能去调这两个值。
DEFAULT_BLOCK_N = 512
DEFAULT_BLOCK_K = 32

# 关掉 Triton 的软件流水（默认为每个 stage 各留一份 shared memory 缓冲，直接
# 把上面的预算乘几倍）。同理：不追求 GPU 性能。
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
    """`y = x @ w.T`。`x`: (M, K)，`w`: (N, K)，`out`: (M, N)，行主序、无 padding。

    N/K 两个方向分块（`N % BLOCK_N == 0`、`K % BLOCK_K == 0`，否则最后一块要
    mask，`pim-explicit-dma` 证不出 `base_arg`——`pick_blocks` 负责保证整除）。
    M 方向不分块：真实场景 decode 时 M=1。

    存储精度由传进来的指针 dtype 决定（llama2-7b 是 float16），累加器固定
    float32——跟 `runtime/kernels.py` 的 `linear_kernel` 一致（按 dtype 读写
    MRAM、用 float32 算）。`acc.to(x_ptr.dtype.element_ty)` 把结果窄回存储
    精度，产出的 TTIR 里是一个 `arith.truncf`，新 pass 认得。
    """
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
    """返回 `(x, w, out) -> CompiledKernel` 的 launcher，固定 M/K/N。

    每个 shape 一份编译产物，跟图编译器每个不同 shape 单独出 `ExecutionPlan`
    的架构一致（见 `docs/opcompiler_bridge-20260825.md`）。返回值是 Triton 的
    `CompiledKernel`（带 `.asm["ttir"]`），供 `driver.py` 取 TTIR 用。
    """
    block_n, block_k = pick_blocks(K, N)

    def launch(x, w, out):
        return linear_kernel[(1,)](
            x, w, out, M=M, K=K, N=N,
            BLOCK_N=block_n, BLOCK_K=block_k, num_stages=NUM_STAGES,
        )

    return launch
