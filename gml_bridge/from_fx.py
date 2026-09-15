"""把融合后的 FX 图转成 GML 节点与边。

只做结构映射——算子类型、拓扑、缓冲区命名、形状。量化参数留给第 4 轮，那时
`node.meta` 里会带上 scale/zp，本模块按同一套字段名填进去即可。

三条容易写错的约定，都在 docs/gml-lowering-20260914.md 第 3 节有验证依据：

- 节点 id 逆拓扑编号：id 越小越靠输出，最终输出是 id 2。
- 缓冲区按**消费者**编号，命名规则见 contracts/gml_names.py。
- 形状只在 edge.dims 上，节点内不带。
"""

from __future__ import annotations

import torch
from torch.fx import GraphModule, Node as FxNode

from contracts import gml_names as names
from contracts.graph_meta import FUSED_TAIL_META_KEY
from gml_bridge.writer import Edge, Node

# FX 算子到 GML `op_type` 的映射。GML 的算子集比 aten 小得多，因为定点流水线里
# 很多 aten 算子（类型转换、断言）没有对应的硬件节点。
OP_TYPES = {
    torch.ops.aten.linear.default: "Gemm",
    torch.ops.aten.addmm.default: "Gemm",
    torch.ops.aten.mm.default: "MatMul",
    torch.ops.aten.bmm.default: "MatMul",
    torch.ops.aten.add.Tensor: "EltwiseAdd",
    torch.ops.aten.sub.Tensor: "EltwiseSub",
    torch.ops.aten.mul.Tensor: "EltwiseMul",
    torch.ops.aten.div.Tensor: "EltwiseDiv",
    torch.ops.aten._softmax.default: "Softmax",
    # 以下按 llama2 W4A8 实物校正：它们在 GML 里是**独立节点**，
    # 不是折进主算子 contraction 的项（见文档 18.6）。
    # `RMSNorm_vpu` 的 `_vpu` 后缀说明它显式绑定在向量单元上，
    # 还带 vpu_params / Vpu_Axis / Use_Scaling / RMSNorm_Add_Const 等专属字段。
    torch.ops.aten.rsqrt.default: "RMSNorm_vpu",
    # `Silu` 自带 nmu_mode / fpsu_* / kantor_mode，以及一个放 fused_Silu_act
    # 的 contraction——它是主算子而非激活。
    torch.ops.aten.silu.default: "Silu",
    # causal mask 在硬件上是一步操作，实物里 32 个（每 head 一个）。
    torch.ops.aten.masked_fill.Scalar: "Mask",
    torch.ops.aten.where.self: "Mask",
    # 按 head 切分：实物用一个 Split 节点带 num_heads 表达。
    torch.ops.aten.split.Tensor: "Split",
    torch.ops.aten.split_with_sizes.default: "Split",
    torch.ops.aten.transpose.int: "Transpose",
    torch.ops.aten.permute.default: "Transpose",
    torch.ops.aten.view.default: "Reshape",
    torch.ops.aten.reshape.default: "Reshape",
    torch.ops.aten.cat.default: "Concat",
    torch.ops.aten.max_pool2d.default: "MaxPool",
    torch.ops.aten.avg_pool2d.default: "AveragePool",
    torch.ops.aten.convolution.default: "Conv",
    # attention 在 GML 里是真实节点（scores/context 两个 MatMul 加一个 Softmax）。
    # 本仓的 NumPy 路径把它放在主机侧执行，但那是执行策略，不是图结构——
    # 如果这里跨过它，它的 q/k/v 三个上游会被下游节点误当成自己的输入。
    torch.ops.aten.scaled_dot_product_attention.default: "MatMul",
}

# 折进 contraction 的激活，GML 用 `Lut` 加 `activation_op_type` 表示。
_ACTIVATION_NAMES = {
    "relu": "Relu",
    "silu": "Silu",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "gelu": "Gelu",
    "exp": "Exp",
    "sqrt": "Sqrt",
    "rsqrt": "Rsqrt",
    "reciprocal": "Reciprocal",
}

_POOL_NAMES = {
    "max": "MaxPool",
    "average": "AveragePool",
    "global_average": "GlobalAveragePool",
}


def _shape_of(node: FxNode) -> str | None:
    """节点输出的形状，写成 GML 的 `1x64x56x56` 形式。"""
    value = node.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return "x".join(str(int(dim)) for dim in shape)


def _is_emittable(node: FxNode) -> bool:
    """这个 FX 节点是否对应一个 GML 硬件节点。"""
    return node.op == "call_function" and node.target in OP_TYPES


def _tensor_inputs(node: FxNode, emittable: set[FxNode]) -> list[FxNode]:
    """节点上游最近的那些 GML 节点。

    不可映射的算子（`to`、`slice`、`pow`、`embedding` 之类）要**跨过**而不是丢弃：
    它们夹在可映射节点之间，直接断开会把图切成互不相连的碎片，入口/出口就不再唯一。
    所以顺着它们继续往上找，直到碰到可映射节点为止。

    权重、常量、标量不在此列——它们在 GML 里是节点属性或编译期数据，不是边。
    """
    found: list[FxNode] = []
    seen: set[FxNode] = set()

    def walk(current: FxNode) -> None:
        for arg in current.all_input_nodes:
            if arg in seen:
                continue
            seen.add(arg)
            if arg in emittable:
                if arg not in found:
                    found.append(arg)
            elif arg.op == "call_function":
                # 这一层在 GML 里不存在，继续往它的上游找。
                walk(arg)

    walk(node)
    return found


def _contraction_of(node: FxNode) -> list[tuple[str, dict[str, object]]]:
    """折进本节点的激活与池化，写成 contraction 块的内容。"""
    tail = node.meta.get(FUSED_TAIL_META_KEY)
    if tail is None:
        return []

    entries: list[tuple[str, dict[str, object]]] = []
    activation = _ACTIVATION_NAMES.get(tail.activation, tail.activation)
    entries.append((
        f"fused_{node.name}_activation",
        {"name": f"{node.name}_activation", "op_type": "Lut",
         "activation_op_type": activation},
    ))
    if tail.pool:
        entries.append((
            f"fused_{node.name}_pool",
            {"name": f"{node.name}_pool",
             "op_type": _POOL_NAMES.get(tail.pool, tail.pool)},
        ))
    return entries


def _weight_param_of(node: FxNode) -> str | None:
    """节点的权重参数名。

    `get_attr` 节点指向模型参数，取第一个二维以上的——一维的是 norm 的缩放或
    偏置，不走 `weight_buffer`。
    """
    for arg in node.all_input_nodes:
        if arg.op != "get_attr":
            continue
        value = arg.meta.get("val")
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            return str(arg.target)
    return None


def convert(gm: GraphModule, *, version: str = "26.10.1") -> tuple[list[Node], list[Edge], dict[int, str]]:
    """把融合后的图转成 GML 节点与边。

    要求 `gm` 已经跑过 `graph.fuse.fuse_graph`：图里若还有独立激活节点，
    GML 无法表达，这里会直接抛。
    """
    emittable = {node for node in gm.graph.nodes if _is_emittable(node)}
    if not emittable:
        raise ValueError("图里没有可映射到 GML 的算子")

    # 逆拓扑编号：id 越小越靠输出。参考产物就是这个约定，最终输出是 id 2。
    ordered = [node for node in gm.graph.nodes if node in emittable]
    node_ids = {node: len(ordered) - index + 2
                for index, node in enumerate(ordered)}

    consumers: dict[FxNode, list[FxNode]] = {node: [] for node in ordered}
    for node in ordered:
        for source in _tensor_inputs(node, emittable):
            consumers[source].append(node)

    gml_nodes: list[Node] = []
    edges: list[Edge] = []
    # node_id -> FX 参数名。写盘阶段按它从图里取 f32 权重去量化。
    weight_params: dict[int, str] = {}

    # 图的入口与出口要有缓冲区节点：两份参考产物都是这样（ResNet50 两个，
    # llama2 的 decode block 十个），而且算子节点悬空会被结构校验判为违规。
    # 入口是没有可映射上游的算子，出口是没有可映射下游的算子。
    entry_targets = [node for node in ordered
                     if not _tensor_inputs(node, emittable)]
    exit_sources = [node for node in ordered if not consumers[node]]
    boundary_base = len(ordered) + 3
    entry_ids = {node: boundary_base + index
                 for index, node in enumerate(entry_targets)}
    exit_ids = {node: boundary_base + len(entry_targets) + index
                for index, node in enumerate(exit_sources)}

    for node, buffer_id in entry_ids.items():
        gml_nodes.append(Node(buffer_id, {
            "label": f"in_{node.name}", "name": f"in_{node.name}",
            "is_buffer": 1,
            "output_buffer": names.data_buffer(node_ids[node]),
            "residual_output_buffer": node_ids[node],
            "output0_node_id": node_ids[node],
        }))
        edges.append(Edge(buffer_id, node_ids[node],
                          _shape_of(node) or "unknown"))

    for node, buffer_id in exit_ids.items():
        gml_nodes.append(Node(buffer_id, {
            "label": f"out_{node.name}", "name": f"out_{node.name}",
            "is_buffer": 1,
            "input_buffer": names.data_buffer(buffer_id),
            "residual_input_buffer": node_ids[node],
            "input0_node_id": node_ids[node],
            "input_count": 1,
        }))

    for node in ordered:
        node_id = node_ids[node]
        inputs = _tensor_inputs(node, emittable)

        fields: dict[str, object] = {
            "label": node.name,
            "name": node.name,
            "op_type": OP_TYPES[node.target],
        }

        # 输入缓冲区按本节点编号——本节点是这些数据的消费者。多输入算子带槽位号。
        # `residual_input_buffer` 在 GML 里是重复键（每个输入一条），所以用列表：
        # writer 会把列表展开成多行同名字段。
        multi = len(inputs) > 1
        upstream_ids: list[int] = []
        for slot, source in enumerate(inputs):
            slot_index = slot if multi else None
            fields[f"input_buffer_{slot}" if multi else "input_buffer"] = (
                names.data_buffer(node_id, slot_index))
            fields[f"input_{slot}_sf" if multi else "input_sf"] = (
                names.scale(node_id, slot_index))
            fields[f"input{slot}_node_id"] = node_ids[source]
            upstream_ids.append(node_ids[source])
        if inputs:
            fields["residual_input_buffer"] = upstream_ids
            fields["input_count"] = len(inputs)
        elif node in entry_ids:
            # 入口算子的上游是入口缓冲节点，字段照常写——否则那个缓冲节点的
            # output_buffer 找不到读者，规则 2 会判为悬空引用。
            fields["input_buffer"] = names.data_buffer(node_id)
            fields["input_sf"] = names.scale(node_id)
            fields["residual_input_buffer"] = [entry_ids[node]]
            fields["input0_node_id"] = entry_ids[node]
            fields["input_count"] = 1

        # 输出缓冲区按**消费者**编号。多个消费者时记第一个，其余靠
        # residual_output_buffer 列出——参考产物就是这样。
        downstream = consumers[node]
        if not downstream and node in exit_ids:
            # 末端算子的输出流向出口缓冲节点，按消费者（即该缓冲节点）编号。
            fields["output_buffer"] = names.data_buffer(exit_ids[node])
            fields["output0_node_id"] = exit_ids[node]
            edges.append(Edge(node_id, exit_ids[node],
                              _shape_of(node) or "unknown"))
        if downstream:
            first = downstream[0]
            first_inputs = _tensor_inputs(first, emittable)
            slot = first_inputs.index(node) if len(first_inputs) > 1 else None
            fields["output_buffer"] = names.data_buffer(node_ids[first], slot)
            for index, consumer in enumerate(downstream):
                fields[f"output{index}_node_id"] = node_ids[consumer]

        # 权重按**本节点**编号——它属于节点自己，不属于某条边（规则 2）。
        # 名字记在 `weight_param` 里（不是 GML 字段，只给写盘用），让写盘阶段
        # 能从 FX 图取到对应的 f32 张量。
        weight_param = _weight_param_of(node)
        if weight_param:
            fields["weight_buffer"] = names.weight_buffer(node_id)
            fields["weight_sf"] = names.weight_scale(node_id)

        contraction = _contraction_of(node)
        gml_node = Node(node_id, fields, contraction)
        if weight_param:
            weight_params[node_id] = weight_param
        gml_nodes.append(gml_node)

        # 边带形状，节点不带。
        shape = _shape_of(node)
        for consumer in downstream:
            edges.append(Edge(node_id, node_ids[consumer], shape or "unknown"))

    return gml_nodes, edges, weight_params
