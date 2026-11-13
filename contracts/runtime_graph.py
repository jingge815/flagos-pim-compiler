"""参数化运行时图契约：符号形状、区域引用与 host 驱动控制流。

编译期产出 RuntimeGraph，运行期由 shape/addr 绑定成可执行实例；本模块只定义
数据结构与可打印文本，不依赖 torch，也不携带绝对地址。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ShapeExpr:
    """形状表达式；`const` 是常量，`dim` 是运行期绑定的符号维。"""

    kind: str
    args: tuple[Any, ...] = ()

    @classmethod
    def const(cls, value: int) -> "ShapeExpr":
        return cls("const", (int(value),))

    @classmethod
    def dim(cls, name: str) -> "ShapeExpr":
        return cls("dim", (str(name),))

    @classmethod
    def add(cls, *items: "ShapeExpr") -> "ShapeExpr":
        return cls("add", tuple(items))

    @classmethod
    def sub(cls, left: "ShapeExpr", right: "ShapeExpr") -> "ShapeExpr":
        return cls("sub", (left, right))

    @classmethod
    def mul(cls, *items: "ShapeExpr") -> "ShapeExpr":
        return cls("mul", tuple(items))

    @classmethod
    def floordiv(cls, left: "ShapeExpr", right: "ShapeExpr") -> "ShapeExpr":
        return cls("floordiv", (left, right))

    @classmethod
    def ceildiv(cls, left: "ShapeExpr", right: "ShapeExpr") -> "ShapeExpr":
        return cls("ceildiv", (left, right))

    @classmethod
    def prod(cls, *items: "ShapeExpr") -> "ShapeExpr":
        return cls("prod", tuple(items))

    @classmethod
    def slice(cls, item: "ShapeExpr", start: int, stop: int) -> "ShapeExpr":
        return cls("slice", (item, int(start), int(stop)))

    @classmethod
    def cat(cls, *items: "ShapeExpr") -> "ShapeExpr":
        return cls("cat", tuple(items))

    def eval(self, env: Mapping[str, int]) -> int:
        """按 shape env 求值；缺维直接报错，不猜默认值。"""
        if self.kind == "const":
            return int(self.args[0])
        if self.kind == "dim":
            return int(env[str(self.args[0])])
        if self.kind in ("add", "mul", "prod", "cat"):
            values = [arg.eval(env) for arg in self.args]
            if self.kind == "add":
                return sum(values)
            if self.kind == "cat":
                return sum(values)
            result = 1
            for value in values:
                result *= value
            return result
        if self.kind == "sub":
            return self.args[0].eval(env) - self.args[1].eval(env)
        if self.kind == "floordiv":
            return self.args[0].eval(env) // self.args[1].eval(env)
        if self.kind == "ceildiv":
            left = self.args[0].eval(env)
            right = self.args[1].eval(env)
            return (left + right - 1) // right
        if self.kind == "slice":
            return int(self.args[2]) - int(self.args[1])
        raise ValueError(f"未知形状表达式：{self.kind}")

    def free_dims(self) -> set[str]:
        """返回表达式依赖的符号维，供绑定前校验。"""
        if self.kind == "dim":
            return {str(self.args[0])}
        result: set[str] = set()
        for arg in self.args:
            if isinstance(arg, ShapeExpr):
                result |= arg.free_dims()
        return result

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, ShapeExpr):
                return value.to_dict()
            return value

        return {"kind": self.kind, "args": [encode(arg) for arg in self.args]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShapeExpr":
        def decode(value: Any) -> Any:
            if isinstance(value, dict):
                return cls.from_dict(value)
            return value

        return cls(str(data["kind"]), tuple(decode(arg) for arg in data["args"]))


@dataclass(frozen=True)
class RegionRef:
    """区域引用：loc 是 (dpu|host, dpu_id)，offset/length 由 ShapeExpr 求值。"""

    loc: tuple[str, int | None]
    offset: ShapeExpr
    length: ShapeExpr

    def resolve(self, env: Mapping[str, int]) -> tuple[tuple[str, int | None], int, int]:
        return self.loc, self.offset.eval(env), self.length.eval(env)


@dataclass
class RTOp:
    """运行时图中的一个操作；payload 与 Command 同构，地址在绑定后填充。"""

    id: int
    op: str
    dpu_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    reads: list[RegionRef] = field(default_factory=list)
    writes: list[RegionRef] = field(default_factory=list)
    deps_hint: list[int] = field(default_factory=list)
    shape_bindings: dict[str, ShapeExpr] = field(default_factory=dict)


@dataclass
class RTControl:
    """host 驱动的控制流节点：if/while/dispatch；分支体由 RTOp 编号表示。"""

    id: int
    kind: str
    cond: str
    branches: dict[str, list[int]] = field(default_factory=dict)
    max_iters: int = 1


@dataclass
class RuntimeGraph:
    """编译期产出的参数化图；运行期由 PlanInstance 绑定成可执行实例。"""

    ops: list[RTOp] = field(default_factory=list)
    controls: list[RTControl] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    symbolic_dims: list[str] = field(default_factory=list)

    def op_by_id(self, op_id: int) -> RTOp:
        for op in self.ops:
            if op.id == op_id:
                return op
        raise KeyError(f"RuntimeGraph 没有 op {op_id}")

    def format(self) -> str:
        return format_runtime_graph(self)


def _format_reads(items: Iterable[RegionRef]) -> str:
    parts = []
    for ref in items:
        loc, suffix = ref.loc, ""
        if loc[1] is not None:
            suffix = f":{loc[1]}"
        parts.append(
            f"{loc[0]}{suffix}{ref.offset.to_dict()}+{ref.length.to_dict()}")
    return "[" + ", ".join(parts) + "]"


def format_runtime_graph(graph: RuntimeGraph) -> str:
    """生成稳定的可读文本，便于断点调试与产物核对。"""
    lines = [
        f"runtime_graph dims={graph.symbolic_dims} inputs={graph.inputs} "
        f"outputs={graph.outputs} ops={len(graph.ops)} "
        f"controls={len(graph.controls)}"
    ]
    for op in graph.ops:
        bindings = {name: expr.to_dict() for name, expr in op.shape_bindings.items()}
        lines.append(
            f"  op {op.id}: {op.op} dpu={op.dpu_id} deps={op.deps_hint} "
            f"reads={_format_reads(op.reads)} writes={_format_reads(op.writes)} "
            f"shape_bindings={bindings}"
        )
    for control in graph.controls:
        lines.append(
            f"  control {control.id}: {control.kind} cond={control.cond} "
            f"branches={control.branches} max_iters={control.max_iters}"
        )
    return "\n".join(lines)
