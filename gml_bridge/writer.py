"""把整算子图写成 GML 文本。

格式是从参考产物逐条验证出来的，见 docs/gml-lowering-20260914.md 第 3 节。几处
容易写错的地方：

- 数组字段展开成重复键（`kernel_shape 3` 两行），不是 `[3, 3]`；PDF 明确要求。
- 形状只在边上（`edge.dims`），节点内不带形状字段。
- 缓冲区按消费者编号，命名规则在 contracts/gml_names.py。
- 不产出硬件放置字段，执行顺序由对方的 L2Analyzer 自行推导。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# GML 缩进：graph 下一层 2 空格，node/edge 内 4 空格。
_INDENT = "  "


@dataclass
class Node:
    """一个 GML 节点，即一个整算子（已折入激活与池化）。

    `fields` 按插入顺序写出。值是 str 时加引号，int 直接写，list 展开成重复键。
    `contraction` 是折进本节点的子算子，每项一个 (名字, 字段字典)。
    """

    node_id: int
    fields: dict[str, object] = field(default_factory=dict)
    contraction: list[tuple[str, dict[str, object]]] = field(default_factory=list)


@dataclass
class Edge:
    """一条数据流边。`dims` 是形状字符串，如 "1x64x56x56"。"""

    source: int
    target: int
    dims: str


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("GML 没有布尔类型，用 0/1 表示")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return f'"{value}"'


def _emit_field(name: str, value: object, depth: int) -> list[str]:
    """写一个字段。列表展开成重复键——PDF 要求数组拆成单值。"""
    pad = _INDENT * depth
    if isinstance(value, (list, tuple)):
        return [f"{pad}{name} {_format_value(item)}" for item in value]
    return [f"{pad}{name} {_format_value(value)}"]


def _emit_node(node: Node) -> list[str]:
    lines = [f"{_INDENT}node ["]
    # id 与 node_id 是同一个值，两者都要写（前者给 networkx，后者给 L2）。
    lines += _emit_field("id", node.node_id, 2)
    lines += _emit_field("node_id", node.node_id, 2)
    for name, value in node.fields.items():
        lines += _emit_field(name, value, 2)

    if node.contraction:
        lines.append(f"{_INDENT * 2}contraction [")
        for child_name, child_fields in node.contraction:
            lines.append(f"{_INDENT * 3}{child_name} [")
            for name, value in child_fields.items():
                lines += _emit_field(name, value, 4)
            lines.append(f"{_INDENT * 3}]")
        lines.append(f"{_INDENT * 2}]")

    lines.append(f"{_INDENT}]")
    return lines


def _emit_edge(edge: Edge) -> list[str]:
    lines = [f"{_INDENT}edge ["]
    lines += _emit_field("source", edge.source, 2)
    lines += _emit_field("target", edge.target, 2)
    # label 与 dims 同值，参考产物两者都写。
    lines += _emit_field("label", edge.dims, 2)
    lines += _emit_field("dims", edge.dims, 2)
    lines.append(f"{_INDENT}]")
    return lines


def write_gml(nodes: list[Node], edges: list[Edge], *, version: str) -> str:
    """产出完整的 GML 文本。"""
    lines = ["graph ["]
    lines += _emit_field("directed", 1, 1)
    lines += _emit_field("relay2gml_version", version, 1)
    for node in nodes:
        lines += _emit_node(node)
    for edge in edges:
        lines += _emit_edge(edge)
    lines.append("]")
    return "\n".join(lines) + "\n"
