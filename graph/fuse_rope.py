"""把 RoPE 的六算子链折成一个 `Llama2Activation` 节点。

这一步补齐参考产物里最后两个未产出的算子（`Llama2Activation` 与
`Llama2ActivationDQ`），同时**消掉**我方多出的 8 个逐元素节点 ——
它们本来就是这条链拆开的样子。

aten 图里 RoPE 长这样（Q 路，K 路同构）：

    transpose_1 ─┬─────────────────────────────┐
                 │                             │
                 ├─→ slice_1 ──────────┐       ├─→ mul_4 = x * cos
                 └─→ slice_2 → neg ────┴─→ cat_1 → mul_5 = rot * sin
                                                       │
                                        add_1 = mul_4 + mul_5

`cat(neg(后半), 前半)` 就是 `rotate_half`，所以整条链等价于

    q * cos + rotate_half(q) * sin

在 GML 里它是**一个**节点，cos/sin 走 `Llama2Activation_Cos` /
`Llama2Activation_Sin` 两个子块（各带自己的定标与 Kantor 配置）。

折叠口径与 `fuse_pim` / `split_heads` 一致：**只打标记、不删节点**，
图仍可执行、数值不变（那次返工的教训见 docs 的 §11.5.1）。

锚点取链尾的 `add`：它是整条链真正的输出节点，下游天然接在它后面。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.fx import GraphModule, Node

from graph.fuse_pim import ABSORBED_META_KEY

# 折出来的 RoPE 节点带这个键，GML 侧见到它就发 Llama2Activation。
ROPE_META_KEY = "pim_rope"


@dataclass
class RopeMatch:
    """一条匹配上的 RoPE 链。

    `cos` / `sin` 是两个广播张量的产出节点 —— GML 侧要按它们生成
    `Llama2Activation_Cos` / `_Sin` 两个子块的缓冲与定标。
    """

    source: Node          # 被旋转的张量（Q 或 K）
    cos: Node
    sin: Node
    absorbed: list[Node] = field(default_factory=list)


@dataclass
class RopeReport:
    fused: int = 0
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        detail = f"折叠 {self.fused} 条 RoPE 链"
        if self.skipped:
            detail += f"；跳过 {len(self.skipped)} 处（{self.skipped[0]}）"
        return detail


def _is(node: object, target) -> bool:
    return isinstance(node, Node) and node.target is target


def _match_rotate_half(node: Node) -> tuple[Node, list[Node]] | None:
    """认出 `cat([neg(x[..., d/2:]), x[..., :d/2]], -1)`，返回被旋转的 x。

    判据必须**两半都查**：只查 `neg` 会把任意 `cat(neg(a), b)` 当成
    rotate_half，而它要求两半来自**同一个**张量、且切分点是最后一维的一半。
    """
    if not _is(node, torch.ops.aten.cat.default):
        return None
    parts = node.args[0]
    if not isinstance(parts, (list, tuple)) or len(parts) != 2:
        return None
    negated, plain = parts
    if not _is(negated, torch.ops.aten.neg.default):
        return None
    upper = negated.args[0]
    if not (_is(upper, torch.ops.aten.slice.Tensor)
            and _is(plain, torch.ops.aten.slice.Tensor)):
        return None
    # 两个 slice 必须切同一个张量的最后一维。
    if upper.args[0] is not plain.args[0]:
        return None
    source = upper.args[0]
    value = source.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    half = int(shape[-1]) // 2
    # 前半 [0, half)，后半 [half, ...)。
    if int(plain.args[2]) != 0 or int(plain.args[3]) != half:
        return None
    if int(upper.args[2]) != half:
        return None
    return source, [node, negated, upper, plain]


def _match(anchor: Node) -> RopeMatch | None:
    """从链尾的 `add` 往上认整条 RoPE。"""
    if not _is(anchor, torch.ops.aten.add.Tensor):
        return None
    left, right = (anchor.args + (None, None))[:2]
    if not (_is(left, torch.ops.aten.mul.Tensor)
            and _is(right, torch.ops.aten.mul.Tensor)):
        return None

    # 两个 mul 里，吃 rotate_half 的那个配 sin，另一个配 cos。
    # 不能假定顺序 —— 按能否匹配 rotate_half 来判。
    for direct, rotated in ((left, right), (right, left)):
        rot = _match_rotate_half(rotated.args[0])
        if rot is None:
            continue
        source, rot_nodes = rot
        if direct.args[0] is not source:
            continue
        cos, sin = direct.args[1], rotated.args[1]
        if not (isinstance(cos, Node) and isinstance(sin, Node)):
            continue
        return RopeMatch(
            source=source, cos=cos, sin=sin,
            absorbed=[direct, rotated] + rot_nodes)
    return None


def fuse_rope(gm: GraphModule) -> RopeReport:
    """把图里每条 RoPE 链标注成一个 `Llama2Activation` 节点。

    必须在 `split_attention_heads` **之前**跑：逐头展开会把 Q/K 切成每头一份，
    切完之后这条链的形状判据（最后一维的一半）仍成立但节点翻 32 倍，
    白白多做 31 次匹配。
    """
    report = RopeReport()
    for node in list(gm.graph.nodes):
        if node.meta.get(ABSORBED_META_KEY) or ROPE_META_KEY in node.meta:
            continue
        match = _match(node)
        if match is None:
            continue
        node.meta[ROPE_META_KEY] = match
        for absorbed in match.absorbed:
            absorbed.meta[ABSORBED_META_KEY] = True
        report.fused += 1

    gm.graph.lint()
    return report
