"""运行期窗口物化：按窗口号从窗口 0 模板现算其余窗口的计划。

静态路径仍在编译期物化全部窗口（`runtime/binding.py::bind_plan`）；动态路径
编译期只留窗口 0 模板和符号化访问表（`RegionRef`），运行期由 `materialize`
按 `{"W": 窗口号}` 解析出具体地址——窗口数不再锁死在编译产物里。平移语义与
`bind_plan` 逐字节一致，`tests/test_runtime_materialize.py` 有等价锚点。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from contracts.exec_plan import Access, ExecutionPlan
from contracts.runtime_graph import RegionRef, ShapeExpr
from runtime.binding import _bind_payload

_WINDOW_DIM = "W"


@dataclass(frozen=True)
class WindowSpec:
    """窗口池几何：编译期定死，物化只选槽号。"""

    act_base_by_dpu: dict[int, int]  # 各 DPU 激活区基址（窗口 0）
    act_stride: int                  # 相邻激活窗口的统一步长
    bt_stride: int                   # 块表窗口步长（= block_table_bytes）
    pool: int                        # 池深：编译期 fit 出的窗口槽总数


@dataclass(frozen=True)
class AccessRef:
    """一条激活区访问的符号化表达：offset 随 W 平移，物化时解析回 Access。"""

    target: str  # "reads" 或 "writes"
    index: int   # 在命令访问列表中的位置
    ref: RegionRef


@dataclass
class WindowTemplate:
    """窗口 0 计划 + 激活区访问的符号表；物化结果按窗口号缓存。"""

    plan: ExecutionPlan
    spec: WindowSpec
    access_refs: dict[int, list[AccessRef]] = field(default_factory=dict)
    _plans: dict[int, ExecutionPlan] = field(default_factory=dict, repr=False)


def build_window_template(
    plan: ExecutionPlan, *, act_base_by_dpu: dict[int, int],
    act_stride: int, bt_stride: int, pool: int,
) -> WindowTemplate:
    """从窗口 0 计划提取激活区访问的符号表；不改任何地址。"""
    refs: dict[int, list[AccessRef]] = {}
    for cmd in plan.commands:
        items: list[AccessRef] = []
        for target in ("reads", "writes"):
            for index, access in enumerate(getattr(cmd, target)):
                if access.loc[0] != "dpu":
                    continue
                base = act_base_by_dpu.get(access.loc[1])
                if base is None or access.offset < base:
                    continue
                # offset = 原 offset + W × stride；窗口 0 时解析回原值。
                offset = ShapeExpr.add(
                    ShapeExpr.const(access.offset),
                    ShapeExpr.mul(ShapeExpr.dim(_WINDOW_DIM),
                                  ShapeExpr.const(act_stride)))
                items.append(AccessRef(
                    target=target, index=index,
                    ref=RegionRef(loc=access.loc, offset=offset,
                                  length=ShapeExpr.const(access.length))))
        if items:
            refs[cmd.id] = items
    spec = WindowSpec(act_base_by_dpu=dict(act_base_by_dpu),
                      act_stride=act_stride, bt_stride=bt_stride, pool=pool)
    return WindowTemplate(plan=plan, spec=spec, access_refs=refs)


def materialize(template: WindowTemplate, window: int) -> ExecutionPlan:
    """把模板解析到第 `window` 号窗口；0 号零拷贝，其余按窗口号缓存。

    访问地址走 `RegionRef.resolve`，payload 平移复用 binding 的实现，两条
    机制各管一半；窗口号越界或区间越池直接抛错，不回退包络。
    """
    if window == 0:
        return template.plan
    cached = template._plans.get(window)
    if cached is not None:
        return cached
    spec = template.spec
    if not 0 <= window < spec.pool:
        raise ValueError(
            f"窗口号 {window} 超出激活窗口池（pool={spec.pool}）")
    env = {_WINDOW_DIM: window}
    act_shift = window * spec.act_stride
    commands = []
    for cmd in template.plan.commands:
        reads, writes = list(cmd.reads), list(cmd.writes)
        for item in template.access_refs.get(cmd.id, ()):
            loc, offset, length = item.ref.resolve(env)
            base = spec.act_base_by_dpu[loc[1]]
            limit = base + spec.pool * spec.act_stride
            if offset + length > limit:
                raise ValueError(
                    f"窗口 {window} 命令 {cmd.id} 的 {item.target}[{item.index}]"
                    f" 越出激活窗口池：[{offset}, {offset + length}) 上限 {limit}")
            access = Access(loc=loc, offset=offset, length=length)
            if item.target == "reads":
                reads[item.index] = access
            else:
                writes[item.index] = access
        commands.append(replace(
            cmd, payload=_bind_payload(cmd.payload, spec.act_base_by_dpu,
                                       act_shift, window, spec.bt_stride),
            reads=reads, writes=writes))
    plan = ExecutionPlan(commands=commands)
    template._plans[window] = plan
    return plan
