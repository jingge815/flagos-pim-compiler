"""图层编译器 → 算子编译器的唯一契约（方案 spec.md:89,1120 的"单算子本地视图"约定）。

字段是四个：`op`、`arg_shapes`、`dtype`、`num_tasklets`。不需要更多——`linear`
白名单实现（`graph/spec_prop.py` 拒绝 tensor bias、`runtime/kernels.py` 无 bias
add）不需要 bias；每个 DPU 收到的已经是切分传播后的本地 shape，算子编译器不需要
知道全局切分方式；MRAM 内的字节 offset 是运行期值（KV cache 增长会变），本就不该
进编译期契约，运行期经 `cmd.reads`/`cmd.writes` 传给 kernel。

`num_tasklets` 是必需输入,不是可选传递：FlagTree 的 `pim-lower-to-emitc` pass
按这个数字生成外层 tid 循环、把 M 维切成 `ceil(M/num_tasklets)` 行一组交给每个
tasklet（仿 downmem `GEMV.c` 的 `my_rows = num_rows/NR_TASKLETS`），也按单个
tasklet 的 tile 大小（不是整个 M）计算 WRAM 预算——`driver.py::_wram_budget`。
`backend/dpu_sdk.py` 现在有真实的 WRAM 字节数组（`_Dpu.wram`），多 tasklet 场景
下 WRAM 相关字段是有代码消费的。

**`arg_shapes[0]`（x）可以是任意 rank，不只是 2-D。** 真实图上 `aten.linear`
的 `x` 带 batch 维——`graph/spec_prop.py` 的 `_rule_linear` 用
`x_node.meta["val"].ndim - 1` 取"contraction 维之前的所有维"，llama2-7b 的
prefill/decode 里实测是 `(batch, seq, hidden)` 三维（如 `(1, 6, 4096)`），不
是 `(M, K)` 二维——这是从真实端到端测试插桩抓到的，不是理论推断，之前一版
`opcompiler_bridge` 假设 `x.shape` 恒为二维是错的，已经在 `driver.py`/
`runtime/kernels.py` 里改成"取最后一维为 K，其余维乘起来展平成 M"（内存里
本来就是行主序连续存储，展平不改变字节布局，等价于 reshape 到 2-D）。
`arg_shapes[1]`（weight）恒为 2-D `(N, K)`，`aten.linear` 的权重定义决定的，
不需要展平。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


def flatten_leading_dims(shape: tuple[int, ...]) -> tuple[int, int]:
    """把一个 rank>=2 的 shape 按最后一维为 K、其余维展平成 M，返回 `(M, K)`。

    行主序连续内存下，展平前几维就是把 `(d0, d1, ..., dn-1, K)` 的元素按顺序
    重排读成 `(d0*d1*...*dn-1, K)`——字节布局完全不变，这也是 NumPy
    `ndarray.reshape` 在连续数组上零拷贝的原理。`linear_kernel`（现有 NumPy
    实现）靠 `@` 的批量矩乘广播语义等价地做了这件事，这里把它显式化，因为
    编译出的 C kernel 需要一个具体的循环边界数字。
    """
    if len(shape) < 2:
        raise ValueError(f"expected rank >= 2, got shape={shape!r}")
    *leading, k = shape
    return prod(leading), k


@dataclass(frozen=True)
class OpCompileRequest:
    op: str
    arg_shapes: list[tuple[int, ...]]
    # MRAM 里的存储 dtype（NumPy dtype 名，如 "float16"/"float32"），来自
    # `cmd.payload["dtype"]`。见模块 docstring 里为什么这个字段必须存在。
    dtype: str = "float32"
    # 这个 DPU 内部按几个 tasklet 顺序模拟切分 M 维（来自 `cmd.num_tasklets`）。
    # 默认 4，与 `contracts/exec_plan.py::Command.num_tasklets` 的默认值一致。
    num_tasklets: int = 4


@dataclass(frozen=True)
class OpCompileResult:
    so_path: str
    symbol: str
    argtypes: list[str]
