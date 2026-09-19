"""切分怎么进 GML：通过**形状**，不通过字段。

实测依据：把参考产物 `relay2gml_graph.gml` 的全部 616 个键名提出来，搜
`device|dpu|shard|tp_|pp_|stage|rank|cluster|placement` —— **零命中**。
GML 里没有任何切分/放置字段，参考产物本身就是单卡 decode block。

那切分怎么体现？**张量形状**。`q_proj` 在 tp=4 下本地权重是 `[4096, 1024]`
而不是 `[4096, 4096]`，于是那份 GML 的 `edge.dims`、`weight_buffer` 字节数、
DQ 组数全都跟着变。所以：

    ShardStrategy(tp_width=k) → 每个 DPU 一份子图 → 每份子图导一个 GML

本模块只做「把策略换算成每个 DPU 的本地形状」这一步，不改 GML 序列化器。

**默认 tp=1**（单卡），此时本地形状恒等于全局形状，GML 产出与不传策略时
逐字节相同。这是当前的默认路径，也是「接入算子编译器前后 GML 相同」那条
判据能成立的前提。多卡切分的接口已备好，但要等每 DPU 一份 GML 的产出规则
与甲方对齐后再启用。
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.strategy import ShardStrategy, llama_strategy


@dataclass(frozen=True)
class LocalShapes:
    """一个 DPU 上的本地权重宽度。

    字段名对应 llama 的七个投影。值是**本地**宽度：tp=1 时等于全局宽度。
    """

    dpu_id: int
    tp_width: int
    hidden_size: int
    # q/k/v 列切：输出宽度按 tp_width 分。
    q_out: int
    k_out: int
    v_out: int
    # o 行切：输入宽度按 tp_width 分，输出保持 hidden。
    o_in: int
    # gate/up 列切，down 行切。
    gate_out: int
    up_out: int
    down_in: int

    @property
    def is_trivial(self) -> bool:
        """tp=1，即本地形状等于全局形状。"""
        return self.tp_width == 1


def single_device_strategy(
    *,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
) -> ShardStrategy:
    """默认策略：单卡，不切分。

    这是当前 GML 导出走的路径。`num_dpus=1, num_stages=1` 使 `tp_width=1`，
    所有本地形状等于全局形状。
    """
    return llama_strategy(
        num_dpus=1,
        num_stages=1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
    )


def local_shapes(
    strategy: ShardStrategy,
    *,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    dpu_id: int = 0,
) -> LocalShapes:
    """按策略算出一个 DPU 上的本地权重宽度。

    切法沿用 `graph/strategy.py` 的 `LLAMA_WEIGHT_RULES`（已验证）：
    q/k/v/gate/up 列切，o/down 行切。列切分输出维，行切分输入维。

    GQA 下 k/v 按 `num_kv_heads` 而不是 `num_heads` 算——两者不等时
    直接用 hidden_size 会把 k/v 切错。
    """
    tp_width = strategy.tp_width
    head_dim = hidden_size // num_heads
    kv_width = num_kv_heads * head_dim

    for label, length in (
        ("hidden_size", hidden_size),
        ("kv_width", kv_width),
        ("intermediate_size", intermediate_size),
    ):
        if length % tp_width:
            raise ValueError(
                f"{label}={length} 不能被 tp_width={tp_width} 整除")

    return LocalShapes(
        dpu_id=dpu_id,
        tp_width=tp_width,
        hidden_size=hidden_size,
        q_out=hidden_size // tp_width,
        k_out=kv_width // tp_width,
        v_out=kv_width // tp_width,
        o_in=hidden_size // tp_width,
        gate_out=intermediate_size // tp_width,
        up_out=intermediate_size // tp_width,
        down_in=intermediate_size // tp_width,
    )


def describe(shapes: LocalShapes) -> str:
    """一行摘要，给 CLI 打印。"""
    if shapes.is_trivial:
        return f"切分: 单卡（tp=1），本地形状 = 全局形状"
    return (f"切分: tp={shapes.tp_width} dpu={shapes.dpu_id}，"
            f"q/k/v 出宽 {shapes.q_out}/{shapes.k_out}/{shapes.v_out}，"
            f"gate/up 出宽 {shapes.gate_out}/{shapes.up_out}，"
            f"o/down 入宽 {shapes.o_in}/{shapes.down_in}")
