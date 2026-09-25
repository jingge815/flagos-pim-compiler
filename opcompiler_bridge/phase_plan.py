"""从算子编译器产出的 PIM IR 里读回相位模板。

这是「相位由谁决定」的交接点。在此之前，相位结构（DQ 4 相、Softmax 5 相、
RoPE 3 连）写死在 `contracts/gml_hw_table.py` 的静态表里——那张表本质是
把算子编译器该算的东西抄了一份。现在真源移到 FlagTree 的
`-pim-expand-phases`，本模块只负责读回来。

读的是 pass 打在每个相位 op 上的结构化属性：

    phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu, reads = [0]>]

`index` 是 0-based 硬件相位号，**只有真正占一次引擎遍历的 op 才带**——
`pim.reshape` 是布局、Softmax 的 `sub` 是折进 exp 相的 FPSU 仿射，两者都不带，
所以数相位就是数这个属性的去重值，不是数 op 个数。

`bytes` 是该相的逻辑缓冲字节数（numel × elem_bytes）。不是 L2 物理段
大小——物理段有对齐和跨节点折叠，归编排器。

`unit` 是引擎（vpu / cstl / nmu）；`forceConsecutive` 标记不得重排的相
（RoPE 三连共享中间缓冲）；`reads` 是本相读哪些前序相位。

与静态表的关系：本模块产出的 `PhasePlan` 用来**交叉校验**静态表，而不是立刻
替换它。理由是静态表里还有一批 pass 目前不产出的字段（`flp_min_exp` 这类
FPSU 浮点域配置、LUT 文件名、Kantor 缓冲名），贸然切换会丢字段。校验先行、
逐族迁移，是这一步能做到「不退化」的唯一顺序。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# `phases = [#pim.phase_spec<index = 0, bytes = 64, unit = vpu, reads = [0]>]`。
# `[^>]*` 够用：phase_spec 的参数里没有 `>`（`reads` 是 `[0]` / `[1, 3]` 这种）。
_PHASE_SPEC_RE = re.compile(r"#pim\.phase_spec<([^>]*)>")
# `reads = [1, 3]`。值里带逗号，所以先从 body 里摘出去再按逗号切其余参数。
_READS_RE = re.compile(r"reads\s*=\s*\[([^\]]*)\]")
_PARAM_RE = re.compile(r"(\w+)\s*=\s*([^,]+)")
# 行首的算子名：`%3 = pim.lut %2 {...}` 里取 `pim.lut`。
_OP_RE = re.compile(r"=\s*(pim\.[a-z_]+)")
# `kind = #pim.activation<exp>` / `kind = #pim.eltwise<absmax>` /
# `kind = #pim.pool_kind<absmax>`
_KIND_RE = re.compile(
    r"\bkind\s*=\s*#pim\.(?:activation|eltwise|pool_kind)<(\w+)>")
# 被展开的函数名，用来把相位归到哪个算子上。
_FUNC_RE = re.compile(r"tt\.func\s+@(\w+)")
# `rotateHalf` 是 UnitAttr，无值，出现即为真。已从裸字符串
# `pim.rotate-half` 收编成 `pim.eltwise` 自己的 ODS 属性——改名会在这里
# 立刻对不上，而不是静默读成 False。
_ROTATE_HALF = "rotateHalf"
# LUT 的寻址窗口与激活模式。这些是 LutOp 的 ODS 属性，键名是驼峰、不带引号。
_I64_ATTR = re.compile(
    r"\b(flpMinExp|flpMaxExp|flpMantisa|activationMode)\s*=\s*(-?\d+)\s*:\s*i64")
# 相位走的是哪张寻址卡：`#pim.transpose_purpose<purpose = layout_reorder,
# cardValue = 1>`。原来是裸属性 `"pim.transpose-type"`，已收编进 ODS；取的是
# `cardValue` 那一项——`purpose` 是枚举，编排器按卡值分流。
_TRANSPOSE_TYPE_RE = re.compile(
    r"#pim\.transpose_purpose<([^>]*)>")
# `kantor = #pim.kantor_spec<mode = fp2int_converter, cardValue = 3>`。卡值单
# 独存：几个不同卡值共用一个方言模式，从模式反推不出来。
_KANTOR_RE = re.compile(r"#pim\.kantor_spec<([^>]*)>")
_FPSU_RE = re.compile(r"#pim\.fpsu_spec<([^>]*)>")
# `vpuParams = #pim.vpu_params<axis = -1, useScaling = false>`。RMSNorm 的
# 向量单元参数块，单相算子身上没有相位可挂，走 `op_attrs` 读回。
_VPU_PARAMS_RE = re.compile(r"#pim\.vpu_params<([^>]*)>")

# 定点单元的卡值。参考实测只出现 1 与 2：1 是 16 位相位通路，2 是 32 位累加
# 通路。GML 文本写模式名，编排器的 txt 写卡值，所以要映射回去。
# `fixed_point` 没有实测卡值（唯一的用处是 KV 写入，那条链目前不展开），
# 取不到就返回 None，让下游退回静态表。
_FPSU_CARD = {"floating_point": 1, "floating_point_32": 2}

# ODS 属性名 -> 编排器惯用的键名。
_ATTR_ALIAS = {
    "flpMinExp": "flp-min-exp",
    "flpMaxExp": "flp-max-exp",
    "flpMantisa": "flp-mantisa",
    "activationMode": "activation-mode",
}


def _attr_params(body: str) -> dict[str, str]:
    """属性体里的 `键 = 值` 对。`blocks` 这类含逗号的数组会截断，但调用方只取
    它前面的标量参数，所以够用。"""
    return {key: value.strip() for key, value in _PARAM_RE.findall(body)}


def _parse_hw_fields(line: str) -> dict[str, int]:
    """一行里的硬件域。

    键名沿用编排器一直在用的**旧契约**（带连字符）：`orchestrator/plan.py`
    与 `layer_fields` 按这些名字取值，属性换成 ODS 形态不该顺带改这层名字。
    """
    fields = {_ATTR_ALIAS[m.group(1)]: int(m.group(2))
              for m in _I64_ATTR.finditer(line)}

    if match := _TRANSPOSE_TYPE_RE.search(line):
        card = _attr_params(match.group(1)).get("cardValue")
        if card is not None:
            fields["transpose-type"] = int(card)

    if match := _KANTOR_RE.search(line):
        card = _attr_params(match.group(1)).get("cardValue")
        if card is not None:
            fields["kantor-mode"] = int(card)

    if match := _FPSU_RE.search(line):
        mode = _attr_params(match.group(1)).get("mode")
        if mode in _FPSU_CARD:
            fields["fpsu-mode"] = _FPSU_CARD[mode]

    if match := _VPU_PARAMS_RE.search(line):
        # 空体 `#pim.vpu_params<>` 是默认值：MLIR 打印时省略等于默认的参数，
        # 而默认值（axis=-1、useScaling=false）恰好是参考产物的实测值。
        params = _attr_params(match.group(1))
        fields["vpu-axis"] = int(params.get("axis", "-1"))
        fields["use-scaling"] = 1 if params.get("useScaling") == "true" else 0

    return fields


def _parse_phase_specs(line: str) -> list[tuple[int, int, str, bool, list[int]]]:
    """一行里出现的相位属性，按出现顺序返回 (index, bytes, unit, 连续, reads)。"""
    specs = []
    for body in _PHASE_SPEC_RE.findall(line):
        reads: list[int] = []
        match = _READS_RE.search(body)
        if match:
            reads = [int(x) for x in match.group(1).split(",") if x.strip()]
            body = body[:match.start()] + body[match.end():]
        params = {key: value.strip()
                  for key, value in _PARAM_RE.findall(body)}
        specs.append((
            int(params["index"]),
            int(params["bytes"]),
            params.get("unit", ""),
            params.get("forceConsecutive", "false") == "true",
            reads,
        ))
    return specs


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
    # 属性可能换行写在 op 的下一行（`pim.normalize` 的 `vpuParams` 就是这样）。
    # 记住上一行是不是 op，续行上的硬件域才有地方收。
    prev_was_op = False

    for raw in pimir_text.splitlines():
        line = raw.split(" loc(")[0]

        func = _FUNC_RE.search(line)
        if func:
            current = PhasePlan(func=func.group(1))
            plans[current.func] = current
            prev_was_op = False
            continue

        if current is None:
            continue

        specs = _parse_phase_specs(line)
        if not specs:
            # 单相算子（`pim.matmul` 这类不展开的）没有相位属性，但 pass
            # 仍在它身上盖了硬件域。收进 `op_attrs` 而**不是** `phases`：
            # 进 phases 会让 `count` 把它当成一相，`cross_check` 的相位数
            # 立刻对不上（矩阵乘是单相，期望值 0）。
            is_op = _OP_RE.search(line) is not None
            if is_op or prev_was_op:
                current.op_attrs.update(_parse_hw_fields(line))
            prev_was_op = is_op
            continue

        prev_was_op = False

        op_match = _OP_RE.search(line)
        kind_match = _KIND_RE.search(line)
        attrs = _parse_hw_fields(line)

        for index, phase_bytes, unit, consecutive, _reads in specs:
            if any(p.index == index for p in current.phases):
                continue
            current.phases.append(Phase(
                index=index,
                op=op_match.group(1) if op_match else "",
                unit=unit,
                bytes=phase_bytes,
                kind=kind_match.group(1) if kind_match else None,
                force_consecutive=consecutive,
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
