"""L2 地址分配：liveness + 贪心复用。

这是编排器存在的核心理由。实测参考产物的 L2 物理段折叠率 **96.4%**，而
`L2 input buffer offset` 全图出现 **0 次**（只分配输出段）——这不是「每层各要
一块」，是跨整张图做生命周期分析后大量复用同一地址。单个算子的 pass 看不到
全局，图编译器也从没做过地址分配，所以只能在这里做。

算法与 `memory/mem_planner.py::greedy_reuse` 同类（那份服务 WRAM/MRAM），
换成 L1/L2 地址空间：

    按尺寸降序放；能塞进已有槽（尺寸够 + 生命周期不重叠）就复用，否则开新槽

生命周期判据用**严格不等号**，与 `mem_planner` 保持一致：取等意味着某层在同一
趟里既读旧缓冲又写新缓冲，两者拿到同一基址会让写覆盖未读完的输入。

**不做的部分**：`docs/prepare_out-域确认表-20260918.md` B7 那批
`L2 fpsu buffer size`（512 的倍数，倍数 1/2/14/56/112/151/152/302 对不上任何
单一几何公式，文档明确写「换形状要重测」）按层类型查表，不在这里硬凑公式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.layer_id import LayerIdentity

# L2 窗口常量。来自 docs/prepare_out-域确认表-20260918.md 步骤 C：
#   QMAN offset = 0x1FFF0000，size = 65536，本样例全层相同
# 注意文档已说明它与手册 Table 7-12 的 L2 内部地址（0x05000000 起）不是同一套，
# 更像 4 GB 虚拟地址（DACU 把 32 位 VA 翻成 64 位 PA）。照样例走。
QMAN_OFFSET = 0x1FFF0000
QMAN_SIZE = 65536
# 数据区顶端 = QMAN 起点，向下不与队列区重叠。
L2_DATA_TOP = QMAN_OFFSET
# 对齐粒度。文档 B7：L2 output size 按 align16(Width) + 16 算。
L2_ALIGN = 16
# 输出段的尾部填充，同上。
L2_OUTPUT_PAD = 16


def align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def l2_output_bytes(width: int, elem_bytes: int) -> int:
    """输出段字节数（闭合公式，文档步骤 C）。

        L2 output size = (align16(Width) + 16) * elem_bytes
    """
    return (align_up(width, L2_ALIGN) + L2_OUTPUT_PAD) * elem_bytes


@dataclass
class L2Buffer:
    """一块待分配的 L2 缓冲。"""

    name: str
    size: int
    # 生命周期用层序号表示：produced_at 写入，last_read_at 最后一次被读。
    produced_at: int
    last_read_at: int


@dataclass
class L2Plan:
    """分配结果。"""

    offsets: dict[str, int] = field(default_factory=dict)
    top: int = 0
    slots: int = 0
    buffers: int = 0

    @property
    def reuse_ratio(self) -> float:
        """复用率 = 1 - 物理槽数 / 缓冲数。

        **别拿这个数直接跟参考产物的 96.4% 对**：两者的缓冲**尺寸**来源不同。
        参考产物的 `L2 fpsu buffer size` 是按层类型查表的（文档 B7 明确写
        「换形状要重测」，倍数 1/2/14/56/112/151/152/302 对不上任何单一几何
        公式），我方用的是闭合公式 `(align16(Width)+16) * elem_bytes`。
        尺寸分布不同，能塞进同一槽的组合就不同，槽数自然不同。

        这个属性有用的地方是**回归**：同一份图改了 liveness 或分配算法后，
        复用率大幅变化说明动到了实质逻辑。要与参考产物逐格对齐，得先拿到
        B7 那张表的完整取值（见文档 Q17）。
        """
        if not self.buffers:
            return 0.0
        return 1.0 - self.slots / self.buffers

    def __str__(self) -> str:
        # `top` 是相对 base 的偏移，就是数据区实际占用的字节数。
        return (f"L2 分配 {self.buffers} 块 → {self.slots} 个物理槽"
                f"（复用率 {self.reuse_ratio:.1%}），"
                f"数据区占用 {self.top} 字节")


def allocate(buffers: list[L2Buffer], *, base: int = 0,
             align: int = L2_ALIGN) -> L2Plan:
    """按生命周期贪心复用 L2 地址。

    尺寸降序是关键：先放大的，小的才能塞进大槽的空档。反过来会让大缓冲永远
    开新槽，复用率掉到接近 0。
    """
    slots: list[dict] = []
    offsets: dict[str, int] = {}
    top = base

    for buffer in sorted(buffers, key=lambda b: (-b.size, b.produced_at, b.name)):
        placed = False
        for slot in slots:
            if buffer.size > slot["size"]:
                continue
            # 严格不等号：同一趟里读旧写新会互相踩，见模块 docstring。
            if all(start > buffer.last_read_at or buffer.produced_at > end
                   for start, end in slot["timeline"]):
                slot["timeline"].append((buffer.produced_at, buffer.last_read_at))
                offsets[buffer.name] = slot["offset"]
                placed = True
                break
        if not placed:
            offset = align_up(top, align)
            slots.append({
                "offset": offset, "size": buffer.size,
                "timeline": [(buffer.produced_at, buffer.last_read_at)],
            })
            offsets[buffer.name] = offset
            top = offset + buffer.size

    return L2Plan(offsets=offsets, top=align_up(top, align),
                  slots=len(slots), buffers=len(buffers))


def output_width_by_node(edges) -> dict[int, int]:
    """每个 GML 节点的输出宽度，取自边上的 `dims`（末维）。

    单层算子（Gemm / MatMul / Mask / Eltwise…）拿不到 `pim.phase-bytes`——
    算子编译器只给多相算子产相位字节数。但它们的输出宽度在 GML 的边上有，
    配合文档步骤 C 的闭合公式就能算出 L2 输出段字节数。

    这批层恰恰是**寿命长**的那些（Gemm 的输出要跨多层被读），漏掉它们会让
    复用率虚高——实测漏掉时算出 99.1%，而参考产物是 96.4%。
    """
    width: dict[int, int] = {}
    for edge in edges:
        dims = str(getattr(edge, "dims", "") or "")
        if not dims:
            continue
        tail = dims.split("x")[-1]
        if not tail.isdigit():
            continue
        # 同一源节点的多条出边宽度相同，取一次即可。
        width.setdefault(int(edge.source), int(tail))
    return width


def consumers_by_node(edges) -> dict[int, set[int]]:
    """每个 GML 节点的下游消费者节点集合。

    真实的消费者关系只有边知道。早先我用「层列表里紧跟的下一个算子」近似，
    但层列表本身就是拓扑序，那个近似退化成了常量 1，让所有缓冲寿命都是 1、
    复用率虚高到 99%+。用边才是对的。
    """
    consumers: dict[int, set[int]] = {}
    for edge in edges:
        consumers.setdefault(int(edge.source), set()).add(int(edge.target))
    return consumers


def buffers_from_layers(identities: list[LayerIdentity],
                        *, output_width: dict[int, int] | None = None,
                        consumers: dict[int, set[int]] | None = None,
                        elem_bytes: int = 2) -> list[L2Buffer]:
    """从层列表推出待分配的输出缓冲。

    只分配**输出**段：实测参考产物里 `L2 input buffer offset` 出现 0 次，
    每层的输入就是上游的输出，不另占地址。

    生命周期：本层写入，被下一个引用它的层读完为止。同一算子的相位链内部相邻
    相位互为生产/消费，所以链内缓冲寿命很短——这正是 96.4% 折叠率的来源。
    """
    buffers: list[L2Buffer] = []
    # 层序号 -> 该层所属算子（label）。
    index_of_label: dict[str, list[int]] = {}
    for index, identity in enumerate(identities):
        index_of_label.setdefault(identity.layer.label, []).append(index)

    # 每个算子的末层序号：跨算子的消费者最早也只能从这之后读。
    last_layer_of: dict[str, int] = {
        label: indices[-1] for label, indices in index_of_label.items()
    }

    # 消费者关系按 GML 的边算：一个算子的输出要活到**最后一个**下游算子读完。
    #
    # 不能用「层列表里紧跟的下一个算子」近似——层列表本身就是拓扑序，那个近似
    # 退化成常量 1，所有缓冲寿命都是 1，复用率虚高到 99%+（实测）。
    first_layer_of_node: dict[int, int] = {}
    last_layer_of_node: dict[int, int] = {}
    for index, identity in enumerate(identities):
        node_id = identity.layer.gml_node_id
        first_layer_of_node.setdefault(node_id, index)
        last_layer_of_node[node_id] = index

    edge_consumers = consumers or {}
    # 每个节点的输出被读完的层序号 = 所有下游节点首层里最晚的那个。
    read_until: dict[int, int] = {}
    for node_id, targets in edge_consumers.items():
        starts = [first_layer_of_node[t] for t in targets
                  if t in first_layer_of_node]
        if starts:
            read_until[node_id] = max(starts)

    widths = output_width or {}

    for index, identity in enumerate(identities):
        layer = identity.layer
        size = layer.phase_bytes or 0
        if size <= 0:
            # 单层算子没有相位字节数，按闭合公式从输出宽度算
            # （文档步骤 C：L2 output size = (align16(Width)+16) * elem_bytes）。
            width = widths.get(layer.gml_node_id)
            if width:
                size = l2_output_bytes(width, elem_bytes)
        if size <= 0:
            continue

        if layer.phase is not None and index < last_layer_of[layer.label]:
            # 相位链内部：下一相立刻读，寿命一步。这是高折叠率的来源。
            end = index + 1
        else:
            # 算子的末层（含单层算子）：按边活到最后一个下游算子读完为止。
            # 查不到下游（图的出口）就只活本层。
            end = read_until.get(layer.gml_node_id, index)

        buffers.append(L2Buffer(
            name=identity.stem, size=size,
            produced_at=index, last_read_at=max(end, index),
        ))
    return buffers
