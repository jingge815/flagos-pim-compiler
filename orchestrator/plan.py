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
            filled.append(layer)
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
        ))

    return ExpandReport(layers=filled, folded=expand.folded,
                        unknown=expand.unknown)


def orchestrate(artifact, *, phase_source=None,
                gml_version: str = "26.2.1") -> OrchestrationPlan:
    """跑完编排四步。

    `artifact` 是 `gml_bridge.export.GmlArtifact`。
    """
    lookup = None if phase_source is None else phase_source.plan
    expand = expand_layers(artifact.nodes, phase_lookup=lookup)
    expand = _attach_phase_data(expand, phase_source)

    identity = assign_ids(expand.layers)
    # 单层算子的输出宽度从 GML 边上取——它们拿不到 `pim.phase-bytes`，
    # 漏掉会让复用率虚高（实测 99.1% vs 参考 96.4%）。
    widths = l2_alloc.output_width_by_node(artifact.edges)
    # 消费者关系按边算，不用拓扑序近似（近似会让寿命恒为 1）。
    consumers = l2_alloc.consumers_by_node(artifact.edges)
    buffers = l2_alloc.buffers_from_layers(
        identity.identities, output_width=widths, consumers=consumers)
    l2 = l2_alloc.allocate(buffers)
    text = net_ini.render(identity.identities, gml_version=gml_version)

    return OrchestrationPlan(expand=expand, identity=identity, l2=l2,
                             net_ini_text=text)
