"""Layer ID 发号与 Task 图。

规则来自 `docs/prepare_out-域确认表-20260918.md` 步骤 C，逐条与参考产物核对：

    Layer ID
        单层算子、多相的最后一相 = GML node_id
        多相的前几相、RoPE 三连  = 从 201 起按出现顺序 +1
        文件名末尾的 params_N 就是该文件的 Layer ID

    Task ID / Prev / Next
        相位链：Task ID = 相位号 - 1，Prev/Next 只连本链内部
        单层算子：Task ID = 0，两个 count = 0
        跨算子依赖**不写** Prev/Next，写进 Datain/Dataout 文件名

「末相沿用 node_id」这条不是美学选择：下游节点的 `Datain file` 引用的是上游的
`Dataout`，而那个名字按 node_id 生成。若末相另发新号，整张图的 buffer 引用
全部错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.layer_expand import Layer

# 多相算子前几相的 Layer ID 起始值。参考产物实测从 201 开始，
# 与 GML 的 200 个节点号不重叠。
EXTRA_LAYER_ID_BASE = 201


@dataclass(frozen=True)
class LayerIdentity:
    """一层的编号与任务链位置。"""

    layer: Layer
    layer_id: int
    task_id: int
    prev_tasks: tuple[int, ...] = ()
    next_tasks: tuple[int, ...] = ()

    @property
    def stem(self) -> str:
        """prepare_out 文件名主干（不含 `_params_N.txt`）。

        多相层带 `_phase_N` 后缀，单层不带——参考产物就是这个约定。
        """
        if self.layer.phase is None:
            return self.layer.label
        return f"{self.layer.label}_phase_{self.layer.phase}"

    @property
    def filename(self) -> str:
        return f"{self.stem}_params_{self.layer_id}.txt"


@dataclass
class IdentityReport:
    identities: list[LayerIdentity] = field(default_factory=list)
    # 发出去的额外 Layer ID 个数（多相算子的前几相）。
    extra_ids: int = 0

    @property
    def total(self) -> int:
        return len(self.identities)

    def by_layer_id(self) -> dict[int, LayerIdentity]:
        return {identity.layer_id: identity for identity in self.identities}

    def __str__(self) -> str:
        return (f"{self.total} 层发号完成"
                f"（沿用 node_id {self.total - self.extra_ids} 个，"
                f"另发 {self.extra_ids} 个）")


def assign_ids(layers: list[Layer]) -> IdentityReport:
    """给每层发 Layer ID 与 Task ID。

    输入是 `layer_expand.expand_layers` 的结果，顺序即执行骨架顺序。
    """
    report = IdentityReport()
    next_extra = EXTRA_LAYER_ID_BASE

    # 先按 (node_id, label) 分组，好知道每个算子有几相、哪一相是末相。
    groups: dict[tuple[int, str], list[Layer]] = {}
    order: list[tuple[int, str]] = []
    for layer in layers:
        key = (layer.gml_node_id, layer.label)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(layer)

    for key in order:
        group = groups[key]
        node_id = key[0]
        last_index = len(group) - 1

        # 本链内部的 task 号，用来连 Prev/Next。
        task_ids = [max(0, (layer.phase or 0)) for layer in group]

        for index, layer in enumerate(group):
            if index == last_index:
                # 末相（或单层算子）沿用 node_id，下游的 buffer 引用才对得上。
                layer_id = node_id
            else:
                layer_id = next_extra
                next_extra += 1
                report.extra_ids += 1

            if len(group) == 1:
                # 单层算子：Task 0，不连任何 Prev/Next。
                task_id, prev, next_ = 0, (), ()
            else:
                task_id = task_ids[index]
                prev = (task_ids[index - 1],) if index > 0 else ()
                next_ = (task_ids[index + 1],) if index < last_index else ()

            report.identities.append(LayerIdentity(
                layer=layer, layer_id=layer_id, task_id=task_id,
                prev_tasks=prev, next_tasks=next_,
            ))

    return report
