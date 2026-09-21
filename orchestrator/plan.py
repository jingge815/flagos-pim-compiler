"""编排器入口：GML 产物 → 层参数结构 + net.ini。

把四步串起来：

    expand_layers   200 节点 → 422 层（一层 = 一次引擎遍历）
    assign_ids      Layer ID 发号、Task 链
    allocate        L2 地址 liveness + 贪心复用
    net_ini.render  执行序

相位数与逐相字节数的真源是**算子编译器**；不传 `phase_source` 时退回
`contracts.gml_quant.PHASE_COUNTS` 静态表，此时 `phase_bytes` 取不到，
L2 分配会跳过那些层（不会瞎猜尺寸）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from contracts.compile_slots import DEFAULT_SLOTS
from orchestrator import l2_alloc, net_ini
from orchestrator.layer_expand import ExpandReport, expand_layers
from orchestrator.layer_id import IdentityReport, assign_ids


@dataclass
class OrchestrationPlan:
    """编排结果。"""

    expand: ExpandReport
    identity: IdentityReport
    l2: l2_alloc.L2Plan
    net_ini_text: str
    layer_texts: dict[str, str] = field(default_factory=dict)

    @property
    def total_layers(self) -> int:
        return self.expand.total

    def __str__(self) -> str:
        return (f"{self.expand}\n{self.identity}\n{self.l2}")


def _attach_phase_data(expand: ExpandReport, phase_source) -> ExpandReport:
    """把算子编译器给的逐相字节数与引擎填进层。

    `phase_source` 是 `opcompiler_bridge.phase_source.PhaseSource`。
    取不到就留 0/空——L2 分配会跳过尺寸为 0 的层，不会拿假尺寸去算地址。
    """
    if phase_source is None:
        return expand

    kind_of = {
        "DynamicScaling": "dq",
        "Softmax": "softmax",
        "Llama2Activation": "rope",
    }

    filled: list = []
    for layer in expand.layers:
        kind = kind_of.get(layer.op_type)
        # Llama2ActivationDQ 是 RoPE 3 连 + DQ 4 相：前 3 相查 rope，后 4 相查 dq。
        if layer.op_type == "Llama2ActivationDQ" and layer.phase is not None:
            kind = "rope" if layer.phase < 3 else "dq"
        if kind is None or layer.phase is None:
            # 单相算子（带尾部激活的矩阵乘）也有算子编译器盖的硬件域，只是
            # 不在 `phases` 里而在 `op_attrs`（见 PhasePlan 的说明）。
            filled.append(_attach_op_attrs(layer, phase_source))
            continue

        plan = phase_source.plan(layer.label, kind)
        if plan is None:
            filled.append(layer)
            continue
        # Llama2ActivationDQ 的后 4 相在 dq 计划里是 0..3。
        index = layer.phase
        if layer.op_type == "Llama2ActivationDQ" and layer.phase >= 3:
            index = layer.phase - 3
        if not 0 <= index < len(plan.phases):
            filled.append(layer)
            continue

        phase = plan.phases[index]
        filled.append(replace(
            layer,
            phase_bytes=phase.bytes,
            unit=phase.unit,
            force_consecutive=phase.force_consecutive,
            flp=((phase.flp_min, phase.flp_max, phase.flp_mantisa)
                 if phase.flp_min is not None and phase.flp_max is not None
                 else None),
            kantor_mode=phase.kantor_mode,
            fpsu_mode=phase.fpsu_mode,
            transpose_type=phase.transpose_type,
            activation_mode=phase.activation_mode,
        ))

    return ExpandReport(layers=filled, folded=expand.folded,
                        unknown=expand.unknown)


def _attach_op_attrs(layer, phase_source):
    """把单相算子的硬件域填进层。

    矩阵乘不展开相位，但算子编译器仍在 `pim.matmul` 上盖了 FPSU / FLP /
    激活模式（见 FlagTree `stampMatmul`）。那些域在 `PhasePlan.op_attrs`
    里，不在 `phases` 里——放进 phases 会让相位数把单相算子算进去。

    取不到就原样返回：`layer_fields` 会回退查表，不会写出空值。
    """
    plan = phase_source.plan(layer.label, "fused_matmul")
    if plan is None or not plan.op_attrs:
        return layer
    return replace(
        layer,
        flp=plan.flp,
        fpsu_mode=plan.op_attrs.get("fpsu-mode"),
        activation_mode=plan.op_attrs.get("activation-mode"),
        kantor_mode=plan.op_attrs.get("kantor-mode"),
    )


def orchestrate(artifact, *, phase_source=None,
                gml_version: str = "26.2.1",
                decode_block_only: bool = False) -> OrchestrationPlan:
    """跑完编排：展开、发号、L2、逐层 txt、net.ini。

    `artifact` 是 `gml_bridge.export.GmlArtifact`。
    `decode_block_only` 丢掉模型末尾 RMSNorm + lm_head + 它们的 DQ，
    层数与参考纯 decode block 的 422 对齐。
    """
    from orchestrator.layer_fields import build_layer_fields, classify, semantic_stem
    from orchestrator.layer_render import render_layer_txt

    # 几何真源跟着产物走：GML 侧已按模型 config 定好编译期槽位。
    slots = getattr(artifact, "slots", None) or DEFAULT_SLOTS
    lookup = None if phase_source is None else phase_source.plan
    expand = expand_layers(artifact.nodes, phase_lookup=lookup)
    expand = _attach_phase_data(expand, phase_source)

    identity = assign_ids(expand.layers)
    widths = l2_alloc.output_width_by_node(artifact.edges)
    consumers = l2_alloc.consumers_by_node(artifact.edges)
    # `nodes_by_id` 要在分配 L2 之前建好：Mask 是否真的双输入（有没有接上
    # causal mask 边界节点）只有 GML 节点自己的 `input_count` 知道，`Layer`
    # 上没有这个信息（复核 20260921 §2.6）。
    nodes_by_id = {node.node_id: node for node in artifact.nodes}
    buffers = l2_alloc.buffers_from_layers(
        identity.identities, output_width=widths, consumers=consumers,
        nodes_by_id=nodes_by_id, slots=slots)
    l2 = l2_alloc.allocate(buffers)

    identities = identity.identities
    # decode-block 已在 GML `_trim_decode_block` 裁过，这里不再用另一套判据。
    identities = _order_like_reference(identities, nodes_by_id, widths)

    layer_texts: dict[str, str] = {}
    stems: list[str] = []
    residual_index = 0
    for item in identities:
        node = nodes_by_id.get(item.layer.gml_node_id)
        if node is None:
            continue
        kind = classify(item, node, widths, nodes_by_id=nodes_by_id,
                        weight_params=getattr(artifact, "weight_params", None),
                        slots=slots)
        extra_residual = 0
        if kind == "residual":
            residual_index += 1
            extra_residual = residual_index
        stem = semantic_stem(item, node, kind, residual_index=extra_residual)
        fields = build_layer_fields(
            item, node, widths=widths, l2_offsets=l2.offsets,
            nodes_by_id=nodes_by_id, stem=stem,
            terminal=item is identities[-1], slots=slots)
        layer_texts[stem + ".txt"] = render_layer_txt(fields)
        stems.append(stem)

    text = net_ini.render(identities, gml_version=gml_version, stems=stems)
    return OrchestrationPlan(expand=expand, identity=identity, l2=l2,
                             net_ini_text=text, layer_texts=layer_texts)


def _order_like_reference(identities, nodes_by_id, widths):
    """按参考骨架排：RMSNorm → DQ → v → k → RoPE_K → q → RoPE_Q/DQ_Q → 32 头。

    FX 拓扑是 q 再 k 再 v，参考 L2A 把写 cache 的 v/k 提前。RoPE 三连必须连续。
    """
    from orchestrator.layer_fields import classify

    tagged = []
    for item in identities:
        node = nodes_by_id.get(item.layer.gml_node_id)
        kind = "other"
        if node is not None:
            kind = classify(item, node, widths, nodes_by_id=nodes_by_id)
        tagged.append((kind, item))

    # gemm_qko 里 v 已分走。k 在 q 前：用权重名区分。
    def gemm_sub(node):
        param = str((node.fields if node else {}).get("pim_weight_param", ""))
        if "k_proj" in param:
            return 0
        if "q_proj" in param:
            return 2
        if "o_proj" in param:
            return 9
        return 1

    def rope_sub(kind, item):
        # K 三连（Llama2Activation）在 Q 三连+DQ（Llama2ActivationDQ）之前。
        if item.layer.op_type == "Llama2Activation":
            return {"rope_mul_cos": 0, "rope_mul_sin": 1, "rope_add_k": 2}.get(kind, 3)
        # Q
        if kind.startswith("rope_"):
            return {"rope_mul_cos": 3, "rope_mul_sin": 4, "rope_add_q": 5}.get(kind, 6)
        if kind.startswith("dq_"):
            return 6 + int(kind[-1])
        return 8

    prefix, heads, tail = [], [], []
    seen_head = False
    for kind, item in tagged:
        is_head = item.layer.head_index is not None or kind.startswith("bmm") or kind == "mask"
        if is_head:
            seen_head = True
            heads.append((kind, item))
            continue
        if seen_head:
            tail.append((kind, item))
            continue
        prefix.append((kind, item))

    def prefix_key(pair):
        kind, item = pair
        node = nodes_by_id.get(item.layer.gml_node_id)
        # 参考：RMSNorm → DQ → v → k → RoPE_K → q → RoPE_Q+DQ_Q
        if kind == "rmsnorm":
            return (0, 0, item.layer.gml_node_id)
        if kind.startswith("dq_") and item.layer.op_type != "Llama2ActivationDQ":
            return (1, item.layer.phase or 0, item.layer.gml_node_id)
        if kind == "gemm_v":
            return (2, 0, item.layer.gml_node_id)
        if kind == "gemm_qko":
            sub = gemm_sub(node)
            if sub == 0:  # k
                return (3, 0, item.layer.gml_node_id)
            if sub == 2:  # q
                return (5, 0, item.layer.gml_node_id)
            return (8, sub, item.layer.gml_node_id)
        if item.layer.op_type == "Llama2Activation":
            return (4, rope_sub(kind, item), item.layer.gml_node_id)
        if kind.startswith("rope_") or (kind.startswith("dq_") and item.layer.op_type == "Llama2ActivationDQ"):
            return (6, rope_sub(kind, item), item.layer.gml_node_id)
        return (7, 0, item.layer.gml_node_id)

    prefix.sort(key=prefix_key)
    # 32 头内部保持原序（bmm1, mask, sm, dq, bmm2）。
    ordered = [item for _, item in prefix + heads + tail]
    return ordered
