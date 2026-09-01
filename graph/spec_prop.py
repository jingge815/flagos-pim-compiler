"""为 FX 图传播张量布局，并标记需要重分布的边。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch.fx import GraphModule, Node

from contracts.graph_meta import (
    DEVICE_DPU,
    DEVICE_HOST,
    DEVICE_META_KEY,
    REDISTRIBUTE_META_KEY,
    SPEC_META_KEY,
)
from contracts.pim_tensor_spec import (
    PIMTensorSpec,
    Placement,
    RedistributeEdge,
    TensorShardDetail,
)
from graph.strategy import ShardStrategy, llama_strategy

REPLICATE = Placement("Replicate")


# 从模块路径中提取层号，忽略 `slice(...)` 的层数参数。
_SLICE_RE = re.compile(r"slice\([^)]*\)")
_STACK_LAYER_RE = re.compile(r"layers\D*?(\d+)")
# 从权重名称中提取层号。
_WEIGHT_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _layer_of_node(node: Node) -> int | None:
    """从权重名称或模块路径返回节点层号；非层节点返回 `None`。"""
    if node.op == "get_attr":
        match = _WEIGHT_LAYER_RE.search(str(node.target))
        return int(match.group(1)) if match else None
    stack = node.meta.get("nn_module_stack") or {}
    for path, _type in reversed(list(stack.values())):
        match = _STACK_LAYER_RE.search(_SLICE_RE.sub("", str(path)))
        if match:
            return int(match.group(1))
    return None


def _num_layers_of(gm: GraphModule) -> int:
    """从权重名称推导层数；没有层权重时返回 1。"""
    layers = [
        layer
        for node in gm.graph.nodes
        if node.op == "get_attr" and (layer := _layer_of_node(node)) is not None
    ]
    return max(layers) + 1 if layers else 1


def _dpus_of_node(node: Node, strategy: ShardStrategy, num_layers: int) -> tuple[int, ...]:
    """返回节点张量所在的 DPU；非层节点归入最后一个流水段。"""
    if strategy.num_stages == 1:
        return strategy.dpu_ids
    layer = _layer_of_node(node)
    if layer is None:
        return strategy.dpus_of_stage(strategy.num_stages - 1)
    return strategy.dpus_of_layer(layer, num_layers)


def llama_shard_config(
    num_dpus: int,
    *,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    dpu_ids: tuple[int, ...] | None = None,
) -> ShardStrategy:
    """构造使用全部 DPU 的 Llama 张量并行策略。"""
    return llama_strategy(
        num_dpus,
        num_stages=1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_layers=1,
        dpu_ids=dpu_ids,
    )


def _host_spec() -> PIMTensorSpec:
    """构造无 DPU 分片的 host 张量规格。"""
    spec = PIMTensorSpec(DEVICE_HOST, REPLICATE, "transient", None, {}, None)
    spec.validate()
    return spec


def _dpu_spec(
    placement: Placement,
    shape: tuple[int, ...],
    dpu_ids: tuple[int, ...],
    residency: Literal["transient", "pinned"] = "transient",
) -> PIMTensorSpec:
    spec = PIMTensorSpec(
        DEVICE_DPU,
        placement,
        residency,
        None,
        _shard_map(shape, placement, dpu_ids),
        placement.reduce_type,
    )
    spec.validate()
    return spec


def _shard_map(
    shape: tuple[int, ...], placement: Placement, dpu_ids: tuple[int, ...]
) -> dict[int, TensorShardDetail]:
    """按布局将全局形状展开为每台 DPU 的连续分片。"""
    if placement.kind == "Shard":
        dim = placement.dim
        placement.validate()
        if dim >= len(shape):
            raise ValueError(f"shape={shape} does not have shard dim {dim}")
        length = shape[dim]
        if length % len(dpu_ids):
            raise ValueError(f"shape={shape} 的 dim {dim} 长 {length} 不能被 num_dpus={len(dpu_ids)} 整除")
        width = length // len(dpu_ids)
        return {
            dpu_id: TensorShardDetail(
                dpu_id=dpu_id,
                shard_dim=dim,
                start_idx=i * width,
                end_idx=(i + 1) * width,
                local_shape=shape[:dim] + (width,) + shape[dim + 1 :],
            )
            for i, dpu_id in enumerate(dpu_ids)
        }
    numel = prod(shape)
    full = tuple(shape)
    return {
        dpu_id: TensorShardDetail(dpu_id, -1, 0, numel, full)
        for dpu_id in dpu_ids
    }


def _weight_spec(node: Node, strategy: ShardStrategy, num_layers: int) -> PIMTensorSpec:
    """按策略为权重构造常驻的 host 或 DPU 张量规格。"""
    shape = tuple(node.meta["val"].shape)
    dpus = _dpus_of_node(node, strategy, num_layers)
    dpu_users = [u for u in node.users if u.meta.get(DEVICE_META_KEY) == DEVICE_DPU]
    linear_users = [
        u for u in dpu_users if u.target in (torch.ops.aten.linear.default, torch.ops.aten.addmm.default)
    ]
    if linear_users:
        mode = strategy.match(node.target)
        if mode is None:
            raise ValueError(f"权重 {node.target} 被 linear 消费，但策略未指定其切法")
        return _dpu_spec(
            Placement("Shard", 0 if mode == "col" else 1), shape, dpus, residency="pinned"
        )
    if dpu_users:
        return _dpu_spec(REPLICATE, shape, dpus, residency="pinned")
    return _host_spec()


@dataclass(frozen=True)
class _Req:
    """规则表要求的输入设备和布局。"""

    device: Literal["host", "dpu"]
    placement: Placement


def _dpu_req(placement: Placement) -> _Req:
    return _Req(DEVICE_DPU, placement)


def _unsupported_layouts(node: Node, *specs: PIMTensorSpec) -> NotImplementedError:
    layouts = ", ".join(f"{spec.placement}@{spec.device}" for spec in specs)
    return NotImplementedError(f"{node.name} 不支持的布局: {layouts}")


def _rule_linear(
    node: Node, x: tuple[Node, PIMTensorSpec], weight: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """返回 `aten.linear` 输入要求和输出布局。"""
    x_node, x_spec = x
    weight_node, w_spec = weight
    out_dim = x_node.meta["val"].ndim - 1  # 输出沿最后一维切分。
    if w_spec.placement == Placement("Shard", 0):
        return [(x_node, _dpu_req(REPLICATE)), (weight_node, _dpu_req(w_spec.placement))], Placement("Shard", out_dim)
    if w_spec.placement == Placement("Shard", 1):
        return [
            (x_node, _dpu_req(Placement("Shard", out_dim))),
            (weight_node, _dpu_req(w_spec.placement)),
        ], Placement("Partial", reduce_type="sum")
    raise _unsupported_layouts(node, x_spec, w_spec)


def _rule_addmm(node: Node) -> tuple[list[tuple[Node, _Req]], Placement]:
    """拒绝当前切分契约未支持的带偏置线性算子。"""
    raise NotImplementedError("aten.addmm（带 bias 线性）不在第 1 阶段切分契约内")


def _rule_elementwise_binary(
    node: Node,
    x: tuple[Node, PIMTensorSpec],
    y: tuple[Node, PIMTensorSpec] | None,
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """返回二元逐元素算子的输入要求和输出布局。"""
    x_node, x_spec = x
    if y is None:  # 标量操作数继承非部分和输入的布局。
        if x_spec.placement.kind == "Partial":
            raise _unsupported_layouts(node, x_spec)
        return [(x_node, _dpu_req(x_spec.placement))], x_spec.placement
    y_node, y_spec = y
    x_kind, y_kind = x_spec.placement.kind, y_spec.placement.kind
    if x_kind == y_kind == "Shard" and x_spec.placement.dim != y_spec.placement.dim:
        raise _unsupported_layouts(node, x_spec, y_spec)
    if x_kind == y_kind == "Shard":
        if x_spec.placement.dim != y_spec.placement.dim:
            raise NotImplementedError("Shard 维不一致的逐元素运算需 all_to_all，不在第 1 阶段契约内")
        out = x_spec.placement
        return [(x_node, _dpu_req(out)), (y_node, _dpu_req(out))], out
    if {x_kind, y_kind} == {"Shard", "Replicate"}:
        out = x_spec.placement if x_kind == "Shard" else y_spec.placement
        return [(x_node, _dpu_req(out)), (y_node, _dpu_req(out))], out
    if x_kind == y_kind == "Replicate":
        return [(x_node, _dpu_req(REPLICATE)), (y_node, _dpu_req(REPLICATE))], REPLICATE
    if {x_kind, y_kind} == {"Partial", "Replicate"}:
        return [(x_node, _dpu_req(REPLICATE)), (y_node, _dpu_req(REPLICATE))], REPLICATE
    if x_kind == y_kind == "Partial":
        if x_spec.placement.reduce_type != y_spec.placement.reduce_type:
            raise ValueError(f"{node.name} Partial 规约类型不一致: {x_spec.placement}, {y_spec.placement}")
        out = x_spec.placement
        return [(x_node, _dpu_req(out)), (y_node, _dpu_req(out))], out
    raise _unsupported_layouts(node, x_spec, y_spec)


def _rule_unary(node: Node, x: tuple[Node, PIMTensorSpec]) -> tuple[list[tuple[Node, _Req]], Placement]:
    """返回一元逐元素算子的输入要求和输出布局。"""
    x_node, x_spec = x
    if x_spec.placement.kind == "Partial":
        raise _unsupported_layouts(node, x_spec)
    return [(x_node, _dpu_req(x_spec.placement))], x_spec.placement


RULE_TABLE = {
    torch.ops.aten.linear.default: _rule_linear,
    torch.ops.aten.addmm.default: _rule_addmm,
    torch.ops.aten.add.Tensor: _rule_elementwise_binary,
    torch.ops.aten.mul.Tensor: _rule_elementwise_binary,
    torch.ops.aten.tanh.default: _rule_unary,
}


def _loc_of_spec(spec: PIMTensorSpec) -> dict:
    if spec.device == DEVICE_HOST:
        return {"device": DEVICE_HOST}
    return {"device": DEVICE_DPU, "dpus": sorted(spec.shard_map)}


def _required_spec(
    src: Node, req: _Req, dst_dpus: tuple[int, ...]
) -> PIMTensorSpec:
    """按消费节点的 DPU 集合构造所需的输入规格。"""
    if req.device == DEVICE_HOST:
        return _host_spec()
    return _dpu_spec(req.placement, tuple(src.meta["val"].shape), dst_dpus)


def _edge_type(node: Node, actual: PIMTensorSpec, req: _Req) -> str:
    """根据源布局和目标布局返回重分布类型。"""
    a, r = actual.placement, req.placement
    if not (
        a == r
        or (a.kind == "Partial" and r.kind == "Replicate")
        or (a.kind == "Shard" and r.kind == "Replicate")
        or (a.kind == "Replicate" and r.kind == "Shard")
        or (a.kind == r.kind == "Shard" and a.dim != r.dim and actual.device == req.device == DEVICE_DPU)
    ):
        raise ValueError(f"{node.name} 不支持的布局转换: {a}@{actual.device} → {r}@{req.device}")
    if a == r:  # 相同布局仅在 host 和 DPU 间移动。
        return "scatter" if actual.device == DEVICE_HOST else "all_gather"
    if a.kind == r.kind == "Shard":
        return "all_to_all"
    if a.kind == "Partial" and r.kind == "Replicate":
        return "all_reduce"
    if a.kind == "Shard" and r.kind == "Replicate":
        return "all_gather"
    if a.kind == "Replicate" and r.kind == "Shard":
        return "scatter" if actual.device == DEVICE_HOST else "local_slice"
    raise ValueError(f"第 1 阶段切分契约外的布局转换: {a} @{actual.device} → {r} @{req.device}")


def _diff_edge(
    edge_id: int, src: Node, dst: Node, actual: PIMTensorSpec, req: _Req,
    dst_dpus: tuple[int, ...]
) -> RedistributeEdge | None:
    """比较边两端规格；相同返回 `None`，否则返回重分布标记。"""
    dst_spec = _required_spec(src, req, dst_dpus)
    if (
        actual.device == req.device
        and actual.placement == req.placement
        and set(actual.shard_map) == set(dst_spec.shard_map)
    ):
        return None
    if actual.residency == "pinned":
        # 常驻张量不支持重分布。
        raise ValueError(f"pinned 张量 {src.name} 不允许重分布（{actual.placement} → {req.placement}）")
    val = src.meta["val"]
    return RedistributeEdge(
        edge_id=edge_id,
        src=src.name,
        dst=dst.name,
        from_placement=actual.placement,
        to_placement=req.placement,
        src_spec=actual,
        dst_spec=dst_spec,
        type=_edge_type(dst, actual, req),
        src_loc=_loc_of_spec(actual),
        dst_loc=_loc_of_spec(dst_spec),
        nbytes=val.numel() * val.element_size(),
        reduce_type=actual.reduce_type,
        shape=tuple(val.shape),
        dtype=str(val.dtype).removeprefix("torch."),
    )


def propagate_specs(gm: GraphModule, strategy: ShardStrategy) -> list[RedistributeEdge]:
    """原地写入节点规格和重分布标记，并返回全部重分布边。"""
    num_layers = _num_layers_of(gm)
    nodes = list(gm.graph.nodes)
    for node in nodes:
        if DEVICE_META_KEY not in node.meta:
            raise ValueError(f"图节点 {node.name} 缺少 device 标注，请先运行 graph.partition.partition_graph")
        node.meta.pop(SPEC_META_KEY, None)
        node.meta[REDISTRIBUTE_META_KEY] = []

    # 1. 初始化输入和权重规格。
    for node in nodes:
        if node.op == "placeholder":
            node.meta[SPEC_META_KEY] = _host_spec()
        elif node.op == "get_attr":
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):  # 跳过非张量属性。
                node.meta[SPEC_META_KEY] = _weight_spec(node, strategy, num_layers)

    # 2. 按拓扑顺序传播算子布局。
    edges: list[RedistributeEdge] = []
    for node in nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":  # 图输出使用 host 布局。
            _require_host_inputs(node, edges, strategy, num_layers)
        elif node.meta[DEVICE_META_KEY] == DEVICE_HOST:
            # 跳过没有张量输出的守卫节点。
            if node.meta.get("val") is not None:
                _require_host_inputs(node, edges, strategy, num_layers)
            node.meta[SPEC_META_KEY] = _host_spec()
        else:
            _propagate_dpu_node(node, edges, strategy, num_layers)
    return edges


def _require_host_inputs(
    node: Node, edges: list[RedistributeEdge], strategy: ShardStrategy, num_layers: int
) -> None:
    """将 host 节点的张量输入转换为 host 副本。"""
    req = _Req(DEVICE_HOST, REPLICATE)
    for input_node in node.all_input_nodes:
        if SPEC_META_KEY not in input_node.meta:  # 跳过非张量属性。
            continue
        _append_edge(node, input_node, req, edges, strategy, num_layers)


def _propagate_dpu_node(
    node: Node, edges: list[RedistributeEdge], strategy: ShardStrategy, num_layers: int
) -> None:
    """按规则表为 DPU 节点生成输入重分布和输出规格。"""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"DPU 节点 {node.name} 缺少张量 meta['val']，无法推导布局")
    rule = RULE_TABLE.get(node.target)
    if rule is None:
        raise ValueError(f"白名单算子 {node.target} 缺少切分规则（与 partition.DPU_LOWERABLE 不一致）")
    if node.target == torch.ops.aten.linear.default:
        x = _tensor_operand(node, 0, "input")
        weight = _tensor_operand(node, 1, "weight")
        _reject_tensor_scalar_operand(node, 2, "bias")
        reqs, out_placement = rule(node, x, weight)
    elif node.target in (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor):
        x = _tensor_operand(node, 0, "self")
        y = _tensor_or_scalar_operand(node, 1, "other")
        _reject_tensor_scalar_operand(node, 2, "alpha")
        reqs, out_placement = rule(node, x, y)
    elif node.target == torch.ops.aten.addmm.default:
        reqs, out_placement = rule(node)
    else:
        reqs, out_placement = rule(node, _tensor_operand(node, 0, "self"))
    for src, req in reqs:
        _append_edge(node, src, req, edges, strategy, num_layers)
    node.meta[SPEC_META_KEY] = _dpu_spec(
        out_placement, tuple(val.shape), _dpus_of_node(node, strategy, num_layers)
    )


_MISSING = object()


def _operand_value(node: Node, position: int, keyword: str) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(keyword, _MISSING)


def _reject_nested_tensor_container(node: Node, operand: object, keyword: str) -> None:
    if isinstance(operand, (tuple, list, dict)):
        raise NotImplementedError(f"{node.name} 的 {keyword} 不支持嵌套 tensor 容器")


def _tensor_operand(node: Node, position: int, keyword: str) -> tuple[Node, PIMTensorSpec]:
    operand = _operand_value(node, position, keyword)
    if operand is _MISSING:
        raise ValueError(f"{node.name} 缺少必需 tensor 操作数 {keyword}")
    _reject_nested_tensor_container(node, operand, keyword)
    if not isinstance(operand, Node):
        raise TypeError(f"{node.name} 的 {keyword} 必须是 tensor 节点")
    return operand, operand.meta[SPEC_META_KEY]


def _tensor_or_scalar_operand(
    node: Node, position: int, keyword: str
) -> tuple[Node, PIMTensorSpec] | None:
    operand = _operand_value(node, position, keyword)
    if operand is _MISSING:
        raise ValueError(f"{node.name} 缺少必需操作数 {keyword}")
    _reject_nested_tensor_container(node, operand, keyword)
    if isinstance(operand, Node):
        return operand, operand.meta[SPEC_META_KEY]
    if isinstance(operand, (int, float, bool)):
        return None
    raise TypeError(f"{node.name} 的 {keyword} 必须是 tensor 节点或标量")


def _reject_tensor_scalar_operand(node: Node, position: int, keyword: str) -> None:
    operand = _operand_value(node, position, keyword)
    if operand is _MISSING or operand is None:
        return
    _reject_nested_tensor_container(node, operand, keyword)
    if isinstance(operand, Node):
        raise NotImplementedError(f"{node.name} 不支持 tensor-valued {keyword}")
    if not isinstance(operand, (int, float, bool)):
        raise TypeError(f"{node.name} 的 {keyword} 必须是标量")


def _append_edge(
    dst: Node, src: Node, req: _Req, edges: list[RedistributeEdge],
    strategy: ShardStrategy, num_layers: int
) -> None:
    """根据消费节点的 DPU 集合，为一条输入边追加重分布标记。"""
    dst_dpus = _dpus_of_node(dst, strategy, num_layers)
    edge = _diff_edge(len(edges), src, dst, src.meta[SPEC_META_KEY], req, dst_dpus)
    if edge is not None:
        edges.append(edge)
        dst.meta[REDISTRIBUTE_META_KEY].append(edge)


def _fmt_p(p: Placement) -> str:
    if p.kind == "Shard":
        return f"Shard({p.dim})"
    if p.kind == "Partial":
        return f"Partial({p.reduce_type})"
    return "Replicate"


def _fmt_placement(spec: PIMTensorSpec) -> str:
    return f"{_fmt_p(spec.placement)}@{spec.device}"


def _fmt_spec(spec: PIMTensorSpec) -> str:
    text = _fmt_placement(spec)
    if spec.device == DEVICE_DPU:
        details = ", ".join(
            f"DPU{detail.dpu_id}:[{detail.start_idx},{detail.end_idx}) shape={detail.local_shape}"
            for _, detail in sorted(spec.shard_map.items())
        )
        text += f"  {details}"
    return text


def format_spec_report(
    gm: GraphModule, edges: list[RedistributeEdge], *, max_nodes: int | None = None
) -> str:
    """返回节点规格、重分布边和分类统计的文本。"""
    nodes = list(gm.graph.nodes)
    lines = [f"== 节点布局（共 {len(nodes)} 个）=="]
    shown = nodes if max_nodes is None else nodes[:max_nodes]
    for node in shown:
        spec = node.meta.get(SPEC_META_KEY)
        text = f"  {node.name}  [{node.meta.get(DEVICE_META_KEY)}]"
        text += f"  -> {_fmt_spec(spec)}" if spec else ""
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            text += f"\n    入边 e{edge.edge_id}: {edge.src} 的 {_fmt_p(edge.from_placement)} 需 {_fmt_p(edge.to_placement)} -> {edge.type}"
        lines.append(text)
    if len(shown) < len(nodes):
        lines.append(f"  ... 省略其余 {len(nodes) - len(shown)} 个节点")

    lines.append(f"== redistribute 边（共 {len(edges)} 条）==")
    for edge in edges:
        lines.append(
            f"  e{edge.edge_id}: {edge.src} -> {edge.dst}  {_fmt_p(edge.from_placement)} -> {_fmt_p(edge.to_placement)}"
            f"  {edge.type}  {edge.nbytes / 2**20:.2f}MiB  {edge.src_loc} -> {edge.dst_loc}"
        )
    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.type] = counts.get(edge.type, 0) + 1
    lines.append(f"== 分类统计 == {dict(sorted(counts.items()))}")
    return "\n".join(lines)
