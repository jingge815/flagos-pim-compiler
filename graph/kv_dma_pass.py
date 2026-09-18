"""插入 KV_Cache_DMA 与 Split 节点。

这两类节点**都不来自 aten 图**：

- `KV_Cache_DMA`：图是用 `use_cache=False` 导出的（见 `runtime.compile`），
  所以没有 KV 写回算子。改成 `use_cache=True` 会把 cache 变成图的输入输出，
  牵动整条已通过回归的运行时路径 —— 收益不值这个代价。
  KV 写回本身是**部署形态**的事（decode 时把新 K/V 写进 cache），
  编译期知道它必然发生在 RoPE 之后、attention 之前，位置是确定的。

- `Split`：逐头展开用 `slice` 隐式拆头（每头一个 slice），
  实物是**一个** 32 输出的 `Split`。语义相同、粒度不同。

与前面几个 pass 同口径：**只加节点、不删原算子**，图仍可执行、数值不变。

实测参考产物的位置与形状：

| 节点 | 个数 | 入边 | 出边 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `KV_Cache_DMA` | 2 | 3 | 2 | K 一个、V 一个 |
| `Split` | 3 | 1 | 32 / 1 | Q/K/V 拆头 |

`KV_Cache_DMA` 的三个输入是 (cache, 索引, 新值)，两个输出是
(给 attention 的那一路, 回写 cache 的那一路)。索引那一路带
`use_input_buffer_1 "L2A_ignore"` —— 它是 int16 的位置下标，
不参与数值计算，所以对方的 L2Analyzer 忽略它。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.fx import GraphModule, Node

from graph.fuse_rope import ROPE_META_KEY
from graph.split_heads import HEAD_INDEX_META_KEY, ROLE_SPLIT

# 插出来的节点带这些键，GML 侧按它们发射。
KV_DMA_META_KEY = "pim_kv_cache_dma"
SPLIT_META_KEY = "pim_split"


@dataclass
class KvDmaSpec:
    """一个 KV_Cache_DMA 的编译期规格。

    `is_key` 区分 K / V 两路 —— 它们配置相同，但落盘的 cache 不同。
    `numel` 定各缓冲的元素数。
    """

    is_key: bool
    numel: int


@dataclass
class SplitSpec:
    """一个 Split 的编译期规格。`heads` 是输出个数。"""

    heads: int
    numel: int


@dataclass
class KvDmaReport:
    kv_dma: int = 0
    splits: int = 0
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        detail = f"插入 {self.kv_dma} 个 KV_Cache_DMA、{self.splits} 个 Split"
        if self.skipped:
            detail += f"；跳过 {len(self.skipped)} 处（{self.skipped[0]}）"
        return detail


def _numel_of(node: Node) -> int:
    value = node.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return 0
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _insert_after(gm: GraphModule, anchor: Node, name: str) -> Node:
    """在 `anchor` 之后插一个恒等载体节点。

    用 `alias`（恒等操作）当载体，所以图的数值完全不变 —— 真正的 DMA 与拆分
    发生在硬件上，编译期只需要一个占位节点承载那些字段。
    这与 `quant_pass` 里 DQ 的做法一致。
    """
    with gm.graph.inserting_after(anchor):
        node = gm.graph.call_function(torch.ops.aten.alias.default, (anchor,))
    node.meta["val"] = anchor.meta.get("val")
    node.name = gm.graph._graph_namespace.create_name(name, None)
    return node


def insert_kv_dma_and_split(gm: GraphModule) -> KvDmaReport:
    """在每条 RoPE 之后插 KV_Cache_DMA，在每组逐头 slice 前插 Split。

    必须在 `fuse_rope` 与 `split_attention_heads` **之后**跑：
    位置判据依赖它们留下的标记。
    """
    report = KvDmaReport()

    # ---- KV_Cache_DMA：每条 RoPE 链之后一个 ----
    # 实测 2 个（K 一个、V 一个）。RoPE 折出来的锚点正好是 Q/K 各一个，
    # 取 K 那一路 —— Q 不进 cache。两个变体里带 DQ 的那个是 K。
    ropes = [n for n in gm.graph.nodes if ROPE_META_KEY in n.meta]
    # 实测 2 个 KV_Cache_DMA：K 一个、V 一个。Q 不进 cache。
    # K 是第二条 RoPE；V 是没有 RoPE 的那一路（v_proj 直接进拆头）。
    anchors: list[tuple[Node, bool]] = []
    if len(ropes) >= 2:
        anchors.append((ropes[1], True))   # K
    # V：被逐头 slice 消费、自身不是 RoPE 的那个源。
    sliced_sources = []
    for n in gm.graph.nodes:
        if n.meta.get("pim_head_role") != ROLE_SPLIT:
            continue
        src = n.args[0] if n.args else None
        if isinstance(src, Node) and src not in sliced_sources:
            sliced_sources.append(src)
    for src in sliced_sources:
        if ROPE_META_KEY not in src.meta and src not in [a for a, _ in anchors]:
            anchors.append((src, False))  # V
            break
    for index, (anchor, is_key) in enumerate(anchors):
        numel = _numel_of(anchor)
        if not numel:
            report.skipped.append(f"{anchor.name}: 取不到形状")
            continue
        node = _insert_after(gm, anchor, f"kv_cache_dma_{index}")
        node.meta[KV_DMA_META_KEY] = KvDmaSpec(is_key=is_key, numel=numel)
        for user in list(anchor.users):
            if user is node:
                continue
            if user.meta.get(HEAD_INDEX_META_KEY) is None:
                continue
            user.replace_input_with(anchor, node)
        report.kv_dma += 1

    # ---- Split：每组逐头 slice 合成一个 ----
    # 逐头展开给每个 slice 打了 ROLE_SPLIT + 头下标。按**被切的张量**分组，
    # 每组一个 Split（实测 3 个：Q/K/V 各一个）。
    groups: dict[Node, list[Node]] = {}
    for node in gm.graph.nodes:
        if node.meta.get("pim_head_role") != ROLE_SPLIT:
            continue
        source = node.args[0] if node.args else None
        if isinstance(source, Node):
            groups.setdefault(source, []).append(node)

    for source, slices in groups.items():
        numel = _numel_of(source)
        if not numel:
            report.skipped.append(f"{source.name}: 取不到形状")
            continue
        node = _insert_after(gm, source, "split")
        node.meta[SPLIT_META_KEY] = SplitSpec(heads=len(slices), numel=numel)
        for sliced in slices:
            sliced.replace_input_with(source, node)
        report.splits += 1

    gm.graph.lint()
    gm.recompile()
    return report
