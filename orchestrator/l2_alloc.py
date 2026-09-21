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
from contracts.compile_slots import DEFAULT_SLOTS

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


def _is_dual_input(op_type: str, phase: int | None, node=None) -> bool:
    """这一层是不是双输入层（两个真实数据槎，不是权重/表节点那种旁路）。

    判据抄自 `orchestrator/layer_fields.py::build_layer_fields` 的 `dual`
    变量（`kind in ("residual", "mask", "mlp_mul") or
    kind.startswith("rope_")`），但只用 `Layer.op_type`/`phase` 这两个字段
    重新表达一遍——不能直接 import `layer_fields.classify`，那边已经
    import 了本模块（`from orchestrator import l2_alloc`），双向 import
    会循环。两处判据必须保持同步：改一处忘改另一处，`L2 input buffer
    offset 0/1` 就会退回同一个地址（这正是之前的 bug）。

    Mask 是例外——它是否真的双输入取决于 GML 图里有没有接上 causal mask
    边界节点，不是纯靠 op_type 能判断的通用规则（复核 20260921 §2.6）：
    `gml_bridge/from_fx.py` 只在真的找到 causal mask placeholder 时才把
    Mask 的 `input_count` 记成 2，seq_len==1 这类退化场景仍是单槎。传入
    `node`（GML 节点，读它的 `input_count` 字段）时按边判；不传时保持旧的
    纯 op_type 判（向后兼容，同 `layer_fields.py` 里同名判据没读到
    `node.fields` 时的退回路径一致）。
    """
    if op_type == "Mask":
        if node is not None:
            return int(getattr(node, "fields", {}).get("input_count") or 1) >= 2
        return True
    if op_type in ("EltwiseAdd", "EltwiseMul"):
        return True
    if op_type in ("Llama2Activation", "Llama2ActivationDQ"):
        # phase 0/1 是 mul_cos/mul_sin，phase 2 是 rope_add_k/rope_add_q——
        # 三者在 layer_fields 里的 kind 都以 "rope_" 开头，全部算双输入。
        # phase >= 3（Llama2ActivationDQ 尾部借用的 DQ 四相）不是。
        return phase is not None and phase < 3
    return False


# RoPE 表宽（元素数）。mul_cos/mul_sin 的两个输入槎宽度不同（数据槎 H、
# 表槎 HD）。常量从编译期槽位取，不能 import layer_fields（循环 import）。
_ROPE_H = DEFAULT_SLOTS.hidden
_ROPE_HD = DEFAULT_SLOTS.head_dim


def _dual_slot1_size(op_type: str, phase: int | None, slot0_size: int,
                     *, bcast: int | None = None, slots=None) -> int:
    """槎 1 该分配多大。

    多数双输入层两个槎同宽。RoPE mul 是例外：表槎 HD×2、数据槎 H×2。
    `bcast` 是 `Eltwise broadcast input index`：1 表示表在槎 1（K 路），
    0 表示表在槎 0、槎 1 是数据（Q 路 mul_cos）。不传时按旧约定（表在槎 1）。
    """
    slots = slots or DEFAULT_SLOTS
    if (op_type in ("Llama2Activation", "Llama2ActivationDQ")
            and phase is not None and phase < 2):
        if bcast == 0:
            return slots.hidden * 2
        return slots.head_dim * 2
    return slot0_size


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
    slot_sizes: dict[int, int] = field(default_factory=dict)
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

    return L2Plan(
        offsets=offsets,
        slot_sizes={slot["offset"]: slot["size"] for slot in slots},
        top=align_up(top, align),
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
                        elem_bytes: int = 2,
                        nodes_by_id: dict[int, object] | None = None,
                        slots=None) -> list[L2Buffer]:
    """从层列表推出待分配的输出缓冲。

    只分配**输出**段：实测参考产物里 `L2 input buffer offset` 出现 0 次，
    每层的输入就是上游的输出，不另占地址。

    生命周期：本层写入，被下一个引用它的层读完为止。同一算子的相位链内部相邻
    相位互为生产/消费，所以链内缓冲寿命很短——这正是 96.4% 折叠率的来源。

    `nodes_by_id` 供 Mask 的双输入判定用（见 `_is_dual_input` 的调用点）：
    Mask 是否真的有 causal mask 边界节点这条边，只有 GML 节点自己的
    `input_count` 知道，不给这个参数时退回纯按 op_type 判（复核 20260921
    §2.6，seq_len==1 这类没有 causal mask 的退化场景需要它才能不多分配
    出一个没有对应边的 `#1` 槎）。
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
    slots = slots or DEFAULT_SLOTS

    for index, identity in enumerate(identities):
        layer = identity.layer
        node = (nodes_by_id.get(layer.gml_node_id)
               if nodes_by_id is not None else None)
        # 尺寸与 txt 声明同源：按编译期槽位，不按导出图边宽 / phase_bytes。
        kind = _kind_hint(layer, node)
        declared = _declared_l2_size(kind, layer, widths, slots)
        size = declared or layer.phase_bytes or 0
        if size <= 0:
            width = widths.get(layer.gml_node_id)
            if kind == "mask":
                width = slots.seq
            if width:
                size = l2_output_bytes(width, elem_bytes)
        if size <= 0:
            continue

        if layer.phase is not None and index < last_layer_of[layer.label]:
            end = index + 1
        else:
            end = read_until.get(layer.gml_node_id, index)

        buffers.append(L2Buffer(
            name=identity.stem, size=size,
            produced_at=index, last_read_at=max(end, index),
        ))

        if _is_dual_input(layer.op_type, layer.phase, node):
            bcast = None
            if (layer.op_type in ("Llama2Activation", "Llama2ActivationDQ")
                    and layer.phase is not None and layer.phase < 2):
                # Q 路 mul_cos：表在槎 0，槎 1 是数据。
                if layer.op_type == "Llama2ActivationDQ" and layer.phase == 0:
                    bcast = 0
                else:
                    bcast = 1
            slot1_size = _dual_slot1_size(
                layer.op_type, layer.phase, size, bcast=bcast, slots=slots)
            buffers.append(L2Buffer(
                name=f"{identity.stem}#1", size=slot1_size,
                produced_at=index, last_read_at=max(end, index),
            ))
    return buffers


def _kind_hint(layer, node) -> str:
    op = layer.op_type
    phase = layer.phase
    if op == "Mask":
        return "mask"
    if op == "EltwiseAdd":
        return "residual"
    if op == "EltwiseMul":
        return "mlp_mul"
    if op == "Llama2Activation" and phase is not None:
        return ("rope_mul_cos", "rope_mul_sin", "rope_add_k")[phase]
    if op == "Llama2ActivationDQ" and phase is not None and phase < 3:
        return ("rope_mul_cos", "rope_mul_sin", "rope_add_q")[phase]
    if op == "MatMul":
        fields = getattr(node, "fields", {}) or {}
        if fields.get("weight_format") == "weights_transpose":
            return "bmm1"
        return "bmm2"
    if op == "Softmax" and phase is not None:
        return f"sm_p{phase + 1}"
    if op == "DynamicScaling" and phase is not None:
        return f"dq_p{phase + 1}"
    return ""


def _declared_l2_size(kind: str, layer, widths: dict[int, int],
                      slots) -> int:
    """与 txt 的 L2 声明同源的字节数（输出段，闭合公式）。"""
    if kind == "mask":
        return l2_output_bytes(slots.seq, 2)
    if kind == "mlp_mul":
        return l2_output_bytes(slots.intermediate, 2)
    if kind.startswith("rope_") or kind == "residual":
        return l2_output_bytes(slots.hidden, 2)
    if kind == "bmm1":
        return l2_output_bytes(slots.seq, 2)
    if kind == "bmm2":
        return l2_output_bytes(slots.hidden, 2)
    if kind.startswith("sm_"):
        return l2_output_bytes(slots.seq, 2)
    return 0
