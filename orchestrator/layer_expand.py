"""GML 节点 × 相位 → 层列表（一层 = 一次引擎遍历）。

硬件上不存在「算子」这个可执行实体，最小可编程单位是**一个引擎在存储层级之间
的一次流式遍历**。所以 GML 的 200 个节点要展开成 422 层，多出来的 222 层几乎
全是 DynamicScaling 的 4 相与 Softmax 的 5 相。

**必须按「是否逐头」分开算，不能按算子类型统一乘系数。** 这是本模块最容易写错
的一处（我自己先算错过一次，得到 418）：

    非逐头  RMSNorm 2×1 + DQ 5×4 + Gemm 7×1 + RoPE 2×3 + Add 2×1 + mul 1×1 = 38
    逐头    32 × (bmm1 + mask + sm×5 + DQ×4 + bmm2 = 12)                    = 384
    合计                                                                    = 422

错法是把 36 个 DQ 全部 ×4——其中 32 个逐头 score DQ 已经计在「每头 12 层」里，
会重复计一次。判据用 `HEAD_INDEX_META_KEY`：带头下标的归入逐头那 384 层。

相位数的真源是**算子编译器**（`opcompiler_bridge/phase_source.py`），静态表
`contracts/gml_quant.PHASE_COUNTS` 作为没装 FlagTree 时的回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.gml_quant import PHASE_COUNTS

# 每类 GML 算子拆成几层。依据 docs/prepare_out-域确认表-20260918.md 步骤 A
# 的「拆成几层」栏，逐条与参考产物核对过。
#
# 没列出的算子（Transpose / Reshape / Concat / Split / KV_Cache_DMA）在参考
# 产物里不单独占层——它们是布局或搬运，折进相邻层的 Datain/Dataout。
LAYERS_PER_OP: dict[str, int] = {
    "RMSNorm_vpu": 1,
    "DynamicScaling": PHASE_COUNTS["DynamicScaling"],      # 4
    "Llama2ActivationDQ": 3 + PHASE_COUNTS["DynamicScaling"],  # 3 连 + 4 相
    "Llama2Activation": 3,                                  # RoPE 三连
    "Softmax": PHASE_COUNTS["Softmax"],                     # 5
    "Gemm": 1,
    "MatMul": 1,
    "Mask": 1,
    "EltwiseAdd": 1,
    "EltwiseMul": 1,
    "Silu": 1,
    "Lut": 1,
}

# 不单独占层的算子：布局/搬运类，折进相邻层。
FOLDED_OPS = frozenset({
    "Transpose", "Reshape", "Concat", "Split", "KV_Cache_DMA",
})


@dataclass(frozen=True)
class Layer:
    """一层 = 一次引擎遍历。

    `phase` 为 None 表示单层算子；否则是 0-based 相位号。
    `head_index` 非 None 表示它属于逐头那一批。
    """

    gml_node_id: int
    label: str                 # GML 的 label，即 FX 节点名
    op_type: str
    phase: int | None = None
    head_index: int | None = None
    # 逻辑缓冲字节数，来自算子编译器的 `pim.phase-bytes`；取不到则 0。
    phase_bytes: int = 0
    # 执行引擎（vpu / cstl / nmu），同样来自算子编译器。
    unit: str = ""
    # RoPE 三连不能被打散。
    force_consecutive: bool = False
    # 算子编译器给的硬件域；没有则 layer_fields 回退查表。
    flp: tuple[int, int, int] | None = None
    kantor_mode: int | None = None
    fpsu_mode: int | None = None
    transpose_type: int | None = None
    activation_mode: int | None = None

    @property
    def is_per_head(self) -> bool:
        return self.head_index is not None


@dataclass
class ExpandReport:
    """一次展开的统计。"""

    layers: list[Layer] = field(default_factory=list)
    # 被折进相邻层、不单独占层的节点。
    folded: list[str] = field(default_factory=list)
    # 不认识的 op_type——宁可报出来也不要静默丢层。
    unknown: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.layers)

    @property
    def per_head(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.is_per_head]

    @property
    def non_per_head(self) -> list[Layer]:
        return [layer for layer in self.layers if not layer.is_per_head]

    def __str__(self) -> str:
        detail = (f"{self.total} 层"
                  f"（非逐头 {len(self.non_per_head)}、"
                  f"逐头 {len(self.per_head)}）")
        if self.unknown:
            detail += f"；未识别 op_type {len(self.unknown)} 个（{self.unknown[0]}）"
        return detail


def _head_index_of(fields: dict) -> int | None:
    """这个节点属于逐头那一批吗，是则返回头下标。

    **判据是 label 里的 `headN`**，不是某个字段。实测 GML 的 node.fields 里
    没有头下标字段：只有 QK 那一路的 `split_channel_number`（32 个）和
    Split 的 `num_heads`（5 个），逐头的 Mask / Softmax / DQ 都不带。

    实测一层导出：label 含 `head` 的 160 个节点 = MatMul 64 + Mask 32 +
    Softmax 32 + DynamicScaling 32，正是逐头的四类；剩下 38 个非逐头。
    这与文档步骤 A 的「循环内每头 12 层 × 32」对得上。
    """
    # 用 FX 名（内部键）：`label` 已改成参考风格的语义名，逐头那批的
    # `headN` 只在 FX 名里。
    label = str(fields.get("pim_fx_name") or fields.get("label", ""))
    marker = "head"
    at = label.rfind(marker)
    if at < 0:
        return None
    digits = ""
    for ch in label[at + len(marker):]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        return int(digits)
    # label 里有 head 但没跟数字（如 `mha_concat_heads`）：算非逐头，
    # 它是把 32 头拼回去的那一步，只有一个。
    return None


def _phase_count_of(
    op_type: str, label: str, phase_lookup, fallback: int
) -> int:
    """这个节点有几层。优先问算子编译器，问不到用静态表。"""
    if phase_lookup is None:
        return fallback
    # Llama2ActivationDQ 是「RoPE 3 连 + DQ 4 相」，两份计划加起来。
    if op_type == "Llama2ActivationDQ":
        rope = phase_lookup(label, "rope")
        dq = phase_lookup(label, "dq")
        if rope is not None and dq is not None:
            return rope.count + dq.count
        return fallback
    kind = {
        "DynamicScaling": "dq",
        "Softmax": "softmax",
        "Llama2Activation": "rope",
    }.get(op_type)
    if kind is None:
        return fallback
    plan = phase_lookup(label, kind)
    return fallback if plan is None else plan.count


def expand_layers(nodes, *, phase_lookup=None) -> ExpandReport:
    """把 GML 节点展开成层。

    `nodes` 是 `gml_bridge.writer.Node` 列表（`GmlArtifact.nodes`）。
    `phase_lookup(label, kind) -> PhasePlan | None` 由
    `opcompiler_bridge.phase_source.PhaseSource.plan` 提供；传 None 则全部
    走静态表。
    """
    report = ExpandReport()

    for node in nodes:
        fields = node.fields
        # 边界缓冲节点不是算子，不占层。
        if fields.get("is_buffer"):
            continue

        op_type = str(fields.get("op_type", ""))
        # 相位模板按 **FX 名** 查（算子编译器是按 FX 名索引的）。
        # `label` 已改成参考风格的语义名，内部键 `pim_fx_name` 才是 FX 名。
        label = str(fields.get("pim_fx_name") or fields.get("label", ""))
        head_index = _head_index_of(fields)

        if op_type in FOLDED_OPS:
            report.folded.append(label)
            continue

        fallback = LAYERS_PER_OP.get(op_type)
        if fallback is None:
            report.unknown.append(f"{label}:{op_type}")
            continue

        count = _phase_count_of(op_type, label, phase_lookup, fallback)
        # 单层算子不带相位号，与多相算子区分开——prepare_out 的文件名
        # 靠这个决定要不要 `_phase_N` 后缀。
        multi = count > 1
        for index in range(count):
            report.layers.append(Layer(
                gml_node_id=node.node_id,
                label=label,
                op_type=op_type,
                phase=index if multi else None,
                head_index=head_index,
            ))

    return report
