"""从算子编译器产出的 PIM IR 里读回相位模板。

这是「相位由谁决定」的交接点。在此之前，相位结构（DQ 4 相、Softmax 5 相、
RoPE 3 连）写死在 `contracts/gml_hw_table.py` 的静态表里——那张表本质是
把算子编译器该算的东西抄了一份。现在真源移到 FlagTree 的
`-pim-expand-phases`，本模块只负责读回来。

读的是 pass 打在每个相位 op 上的三个属性：

    pim.phase        0-based 硬件相位号。**只有真正占一次引擎遍历的 op 才带**
                     ——`pim.reshape` 是布局、Softmax 的 `sub` 是折进 exp 相的
                     FPSU 仿射，两者都不带，所以数相位就是数这个属性的去重值，
                     不是数 op 个数。
    pim.phase-bytes  该相的逻辑缓冲字节数（numel × elem_bytes）。不是 L2 物理
                     段大小——物理段有对齐和跨节点折叠，归编排器。
    unit             引擎（vpu / cstl / nmu）。

与静态表的关系：本模块产出的 `PhasePlan` 用来**交叉校验**静态表，而不是立刻
替换它。理由是静态表里还有一批 pass 目前不产出的字段（`flp_min_exp` 这类
FPSU 浮点域配置、LUT 文件名、Kantor 缓冲名），贸然切换会丢字段。校验先行、
逐族迁移，是这一步能做到「不退化」的唯一顺序。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 一个带相位号的 op 行。`pim.phase = 3 : i64` 里取 3。
_PHASE_RE = re.compile(r"\bpim\.phase\s*=\s*(\d+)\s*:\s*i64")
# `"pim.phase-bytes" = 4096 : i64`。注意键名带引号（MLIR 对带连字符的键加引号）。
_PHASE_BYTES_RE = re.compile(r'"pim\.phase-bytes"\s*=\s*(\d+)\s*:\s*i64')
# `unit = #pim.unit<cstl>`
_UNIT_RE = re.compile(r"\bunit\s*=\s*#pim\.unit<(\w+)>")
# 行首的算子名：`%3 = pim.lut %2 {...}` 里取 `pim.lut`。
_OP_RE = re.compile(r"=\s*(pim\.[a-z_]+)")
# `kind = #pim.activation<exp>` / `kind = #pim.eltwise<absmax>`
_KIND_RE = re.compile(r"\bkind\s*=\s*#pim\.(?:activation|eltwise)<(\w+)>")
# 被展开的函数名，用来把相位归到哪个算子上。
_FUNC_RE = re.compile(r"tt\.func\s+@(\w+)")
# `pim.force-consecutive` 是 unit attr，无值。
_FORCE_CONSECUTIVE = "pim.force-consecutive"
_ROTATE_HALF = "pim.rotate-half"
_I64_ATTR = re.compile(
    r'"pim\.(flp-min-exp|flp-max-exp|flp-mantisa|kantor-mode|'
    r'fpsu-mode|transpose-type|activation-mode)"\s*=\s*(-?\d+)\s*:\s*i64')


@dataclass(frozen=True)
class Phase:
    """一个硬件相位：一次引擎遍历。"""

    index: int
    op: str                 # `pim.lut` / `pim.reduce_axis` / `pim.quantize` / `pim.eltwise`
    unit: str               # `vpu` / `cstl` / `nmu`
    bytes: int              # 逻辑缓冲字节数
    kind: str | None = None  # `absmax` / `exp` / `reciprocal` / `mul` ...
    force_consecutive: bool = False
    rotate_half: bool = False
    flp_min: int | None = None
    flp_max: int | None = None
    flp_mantisa: int | None = None
    kantor_mode: int | None = None
    fpsu_mode: int | None = None
    transpose_type: int | None = None
    activation_mode: int | None = None


@dataclass
class PhasePlan:
    """一个被展开算子的相位序列。

    `op_attrs` 收**单相算子**（矩阵乘这类不展开的）身上的硬件域。它们没有
    `pim.phase`，所以不进 `phases`——否则 `count` 会把它们算成相位，
    `cross_check` 的相位数就对不上。但那些域仍然是算子编译器的产出，
    txt 要用（gemm_gate 的 Flp / Fpsu mode 就在这里）。
    """

    func: str
    phases: list[Phase] = field(default_factory=list)
    op_attrs: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.phases)

    @property
    def flp(self) -> tuple[int, int, int] | None:
        """单相算子的 FLP 窗口，取不到返回 None。"""
        if "flp-min-exp" not in self.op_attrs:
            return None
        return (self.op_attrs["flp-min-exp"],
                self.op_attrs.get("flp-max-exp", 0),
                self.op_attrs.get("flp-mantisa", 0))

    def units(self) -> list[str]:
        return [p.unit for p in self.phases]

    def kinds(self) -> list[str | None]:
        return [p.kind for p in self.phases]


def parse_phase_plans(pimir_text: str) -> dict[str, PhasePlan]:
    """从展开后的 PIM IR 文本里读出每个函数的相位序列。

    返回 `{函数名: PhasePlan}`。同一相位号出现多次时保留第一条——
    pass 保证每个相位号只发一次，重复说明 IR 被改过。
    """
    plans: dict[str, PhasePlan] = {}
    current: PhasePlan | None = None

    for raw in pimir_text.splitlines():
        line = raw.split(" loc(")[0]

        func = _FUNC_RE.search(line)
        if func:
            current = PhasePlan(func=func.group(1))
            plans[current.func] = current
            continue

        if current is None:
            continue

        phase_match = _PHASE_RE.search(line)
        if not phase_match:
            # 单相算子（`pim.matmul` 这类不展开的）没有 `pim.phase`，但 pass
            # 仍在它身上盖了硬件域。收进 `op_attrs` 而**不是** `phases`：
            # 进 phases 会让 `count` 把它当成一相，`cross_check` 的相位数
            # 立刻对不上（矩阵乘是单相，期望值 0）。
            if _OP_RE.search(line):
                current.op_attrs.update(
                    {m.group(1): int(m.group(2))
                     for m in _I64_ATTR.finditer(line)})
            continue

        index = int(phase_match.group(1))
        if any(p.index == index for p in current.phases):
            continue

        op_match = _OP_RE.search(line)
        unit_match = _UNIT_RE.search(line)
        bytes_match = _PHASE_BYTES_RE.search(line)
        kind_match = _KIND_RE.search(line)

        attrs = {m.group(1): int(m.group(2)) for m in _I64_ATTR.finditer(line)}
        flp = None
        if "flp-min-exp" in attrs and "flp-max-exp" in attrs:
            flp = (attrs["flp-min-exp"], attrs["flp-max-exp"],
                   attrs.get("flp-mantisa", 0))
        current.phases.append(Phase(
            index=index,
            op=op_match.group(1) if op_match else "",
            unit=unit_match.group(1) if unit_match else "",
            bytes=int(bytes_match.group(1)) if bytes_match else 0,
            kind=kind_match.group(1) if kind_match else None,
            force_consecutive=_FORCE_CONSECUTIVE in line,
            rotate_half=_ROTATE_HALF in line,
            flp_min=attrs.get("flp-min-exp"),
            flp_max=attrs.get("flp-max-exp"),
            flp_mantisa=attrs.get("flp-mantisa"),
            kantor_mode=attrs.get("kantor-mode"),
            fpsu_mode=attrs.get("fpsu-mode"),
            transpose_type=attrs.get("transpose-type"),
            activation_mode=attrs.get("activation-mode"),
        ))

    for plan in plans.values():
        plan.phases.sort(key=lambda p: p.index)
    return plans
