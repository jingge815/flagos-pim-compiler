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

from contracts import gml_hw_table as hw_table
from contracts import gml_names as names
from contracts.gml_quant import GML_VERSION, PHASE_COUNTS
from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse_pim import (
    ABSORBED_META_KEY,
    ATTENTION_SCALE_META_KEY,
    RMS_NORM_META_KEY,
)
from graph.quant_pass import DQ_META_KEY
from graph.fuse_rope import ROPE_META_KEY
from graph.kv_dma_pass import KV_DMA_META_KEY, SPLIT_META_KEY
from graph.split_heads import (
    HEAD_INDEX_META_KEY,
    HEAD_ROLE_META_KEY,
    ROLE_MASK,
    ROLE_MATMUL_PV,
    ROLE_MATMUL_QK,
    ROLE_SOFTMAX,
)
from gml_bridge.writer import Edge, Node

# FX 算子到 GML `op_type` 的映射。GML 的算子集比 aten 小得多，因为定点流水线里
# 很多 aten 算子（类型转换、断言）没有对应的硬件节点。
OP_TYPES = {
    torch.ops.aten.linear.default: "Gemm",
    torch.ops.aten.addmm.default: "Gemm",
    torch.ops.aten.mm.default: "MatMul",
    torch.ops.aten.bmm.default: "MatMul",
    # 逐头展开产出的是 `matmul`（保留 4 维），不是 `bmm`（要先压成 3 维）。
    torch.ops.aten.matmul.default: "MatMul",
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

# 带 output_sf / output_zp 的算子。实测规则：**做计算的带，纯布局的不带**。
#
#   带（个数/总数）：MatMul 64/64、DynamicScaling 36/36、Softmax 32/32、
#                    Mask 32/32、Gemm 7/7、EltwiseAdd 2/2、EltwiseMul 1/1、
#                    RMSNorm_vpu 2/2、KV_Cache_DMA 2/2
#   不带：Transpose 0/4、Reshape 0/2、Concat 0/1
#
# 判据是「这个算子会不会改变数值的动态范围」—— 换轴、改形状、拼接都不会，
# 所以下游沿用上游的 scale，不需要 requant。
_OUTPUT_SCALE_OPS = frozenset({
    "Gemm", "MatMul", "Softmax", "Mask", "DynamicScaling",
    "EltwiseAdd", "EltwiseMul", "RMSNorm_vpu", "KV_Cache_DMA",
    "Llama2Activation", "Llama2ActivationDQ", "Silu",
})


def _op_type_of(node):
    """节点的 GML `op_type`。

    优先级固定：DQ > RMSNorm 折叠 > 逐头角色 > aten 目标查表。
    抽成函数是因为三处都要用它：发射本节点的字段、生产者给输出缓冲定槽号、
    以及 dtype 沿边传播时定键名 —— 三处必须得到同一个答案。
    """
    if DQ_META_KEY in node.meta:
        return "DynamicScaling"
    if RMS_NORM_META_KEY in node.meta:
        return "RMSNorm_vpu"
    if KV_DMA_META_KEY in node.meta:
        return "KV_Cache_DMA"
    if SPLIT_META_KEY in node.meta:
        return "Split"
    if ROPE_META_KEY in node.meta:
        # 折叠出来的 RoPE。实测两个变体各出现一次：
        #   第一条（Q）-> Llama2Activation（transpose=0，下游 Transpose）
        #   第二条（K）-> Llama2ActivationDQ（transpose=1，进 cache）
        # llama 图里 q_proj 在 k_proj 之前，所以拓扑序第一条就是 Q。
        # 不能靠「下游有没有 Split」判 —— 逐头展开之后 Q/K 都有 slice。
        return ("Llama2ActivationDQ"
                if _is_second_rope(node) else "Llama2Activation")
    role = node.meta.get(HEAD_ROLE_META_KEY)
    if role in _ROLE_OP_TYPES:
        return _ROLE_OP_TYPES[role]
    return OP_TYPES.get(node.target)


def _data_slot_count(op_type, input_count):
    """这个算子有几个**数据**输入槽。

    MatMul 的第二个 operand 走权重通路（`MatMul_input_as_weight 1`），
    不占数据槽 —— 所以两条入边只对应一个数据槽，键名不带槽号
    （实测 64/64 个 MatMul 都是 `input_buffer`，且 `input_count` 记 1）。

    生产者给输出缓冲命名时要按**消费者**的槽号，所以两边必须用同一判据：
    只在一处判、另一处沿用，否则生产者写 `input_buffer_1_176.bin`
    而消费者只声明 `input_buffer`，一边悬空一边多余。
    """
    if op_type == "MatMul":
        return max(1, input_count - 1)
    return input_count


def _shape_tuple_of(node) -> tuple[int, ...] | None:
    """节点输出的形状，取成 int 元组。

    **不要叫 `_shape_of`** —— 本模块已有一个同名函数（见下方 144 行附近），
    它返回的是 GML 边上的 dims 字符串（`"1x64x56x56"`）。两者重名的话
    后定义的会覆盖先定义的，调用处拿到字符串再去 `int()` 就炸在
    `invalid literal for int(): 'x'`。
    """
    value = getattr(node, "meta", {}).get("val") if node is not None else None
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(x) for x in shape)


def _is_second_rope(node) -> bool:
    """是不是图里第二条 RoPE 链（K 路）。

    llama 的 q_proj 排在 k_proj 前面，所以按拓扑序，第二条就是 K。
    K 进 cache 要定点化，走 Llama2ActivationDQ。
    """
    ropes = [n for n in node.graph.nodes if ROPE_META_KEY in n.meta]
    return len(ropes) >= 2 and node is ropes[1]


def _numel_of_fx(node) -> int:
    """FX 节点输出的元素数。取不到形状时返回 0。"""
    shape = _shape_tuple_of(node)
    if not shape:
        return 0
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _head_count_of(node) -> int:
    """RoPE 节点的头数。cos/sin 要广播到每个头，所以这个数进 `num_heads`。

    形状是 `[batch, heads, seq, head_dim]`，取第 1 维。
    """
    shape = _shape_tuple_of(node)
    return int(shape[1]) if shape and len(shape) == 4 else 1


def _shape_literal(shape: tuple[int, ...]) -> str:
    """形状写成 `[1, 1, 1, 4096]` 这种字面量，照实物的格式（逗号后带空格）。"""
    return "[" + ", ".join(str(int(x)) for x in shape) + "]"


# 带 FPSU 定标三族的算子，以及各自的槽数（实测）。
# 逐元素算子按槽配置（每个输入一组系数），其余单组。
_FPSU_OPS = frozenset({
    "Gemm", "MatMul", "KV_Cache_DMA", "EltwiseAdd", "EltwiseMul"})
_FPSU_SLOTS = {"EltwiseAdd": 2, "EltwiseMul": 2}


# 逐头展开产出的节点，按角色定 op_type。
_ROLE_OP_TYPES = {
    ROLE_MATMUL_QK: "MatMul",
    ROLE_MATMUL_PV: "MatMul",
    ROLE_MASK: "Mask",
    ROLE_SOFTMAX: "Softmax",
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
    """这个 FX 节点是否对应一个 GML 硬件节点。

    折过的 RMSNorm 以 `pow` 为锚点（`graph.fuse_pim` 把整条六算子链折到那里），
    而 `pow` 不在 `OP_TYPES` 里——所以要按 meta 标记额外认它。
    """
    if node.op != "call_function":
        return False
    # 被折进别的节点的算子不单独发射（RMSNorm 链的中间项、折进 contraction
    # 的激活、被吸收的 attention 定标）。它们仍留在图里以保持可执行。
    if node.meta.get(ABSORBED_META_KEY):
        return False
    if RMS_NORM_META_KEY in node.meta:
        return True
    # 逐头展开打了角色标记的节点一律发射：角色决定 op_type，
    # 不要求它的 aten 目标在 OP_TYPES 里。
    if node.meta.get(HEAD_ROLE_META_KEY) in _ROLE_OP_TYPES:
        return True
    # DQ 节点的载体是 `alias`（fx 里的恒等操作），不在 OP_TYPES 里，
    # 靠这个标记认。它在 GML 里是完整的 4 相 DynamicScaling。
    if DQ_META_KEY in node.meta:
        return True
    if ROPE_META_KEY in node.meta:
        return True
    if KV_DMA_META_KEY in node.meta or SPLIT_META_KEY in node.meta:
        return True
    return node.target in OP_TYPES


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


def convert(gm: GraphModule, *, version: str = GML_VERSION) -> tuple[list[Node], list[Edge], dict[int, str]]:
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
            # 用列表而不是裸 int：`residual_*_buffer` 是重复键，
            # 统一成列表让下游（校验器、测试）不必兼容两种类型。
            names.residual_buffer_key("output", 0): [node_ids[node]],
            names.port_node_id_key("output", 0): node_ids[node],
        }))
        edges.append(Edge(buffer_id, node_ids[node],
                          _shape_of(node) or "unknown"))

    for node, buffer_id in exit_ids.items():
        gml_nodes.append(Node(buffer_id, {
            "label": f"out_{node.name}", "name": f"out_{node.name}",
            "is_buffer": 1,
            "input_buffer": names.data_buffer(buffer_id),
            names.residual_buffer_key("input", 0): [node_ids[node]],
            names.port_node_id_key("input", 0): node_ids[node],
            "input_count": 1,
        }))

    for node in ordered:
        node_id = node_ids[node]
        inputs = _tensor_inputs(node, emittable)

        rms_norm = node.meta.get(RMS_NORM_META_KEY)
        role = node.meta.get(HEAD_ROLE_META_KEY)
        dq_spec = node.meta.get(DQ_META_KEY)

        # 吃 DQ 输出的节点，输入 dtype 跟着上游变成 int8。
        # 实测 DQ 12 -> Gemm 11：DQ 声明 `output_buffer_dtype int8`，
        # 消费者也声明 `input_buffer_dtype int8`。dtype 是**沿边传播**的，
        # 不是每个节点独立猜 —— 写死 fp16 会让缓冲宽度差一倍。
        #
        # 这里只**收集**，到节点收尾时再统一写。曾经直接写在
        # `if downstream:` 块内的 output_buffer 之后，结果把后面写
        # residual_output_buffer / outputN 的代码一起吞进了循环体，
        # 三份连接信息不再同步，结构自检报「仅在 edge」。
        upstream_int8 = {}
        for slot_index, source in enumerate(inputs):
            if DQ_META_KEY not in getattr(source, 'meta', {}):
                continue
            key = ('input_buffer_%d_dtype' % slot_index
                   if _data_slot_count(_op_type_of(node), len(inputs)) > 1
                   else 'input_buffer_dtype')
            upstream_int8[key] = 'int8'

        # 逐头展开产出的节点按**角色**定 op_type，不按 aten 目标：
        # 逐头的 `add` 是 Mask 而不是 EltwiseAdd，`matmul` 是 MatMul 的两种角色。
        op_type = _op_type_of(node)

        fields: dict[str, object] = {
            "label": node.name,
            "name": node.name,
            "op_type": op_type,
        }

        # 两个 attention matmul 的硬件配置不同，实测 32/32 各自一致：
        #   matmul1 (QKᵀ) 吃 K 的转置 -> weights_transpose，定标 1/√head_dim
        #   matmul2 (PV)  直接吃 V    -> weight，定标 1.0
        # 这是**数学决定的**（哪个是 QKᵀ 图编译器完全知道），不是排布优化。
        if role in (ROLE_MATMUL_QK, ROLE_MATMUL_PV):
            fields.update(hw_table.top_level_fields(
                "MatMul", transposed=role == ROLE_MATMUL_QK))
            head = node.meta.get(HEAD_INDEX_META_KEY)
            if role == ROLE_MATMUL_QK and head is not None:
                fields["split_channel_number"] = head
        elif role == ROLE_MASK:
            fields.update(hw_table.top_level_fields("Mask"))
        elif role == ROLE_SOFTMAX:
            fields.update(hw_table.top_level_fields("Softmax"))

        # 输入缓冲区按本节点编号——本节点是这些数据的消费者。多输入算子带槽位号。
        # `residual_input_buffer` 在 GML 里是重复键（每个输入一条），所以用列表：
        # writer 会把列表展开成多行同名字段。
        # MatMul 的第二个 operand 走**权重通路**，不占数据输入槽——所以它只有
        # 一个数据槽，键名不带槽号（实测 64/64 个 MatMul 都是 `input_buffer`）。
        # 端口 `inputN_node_id` 仍然两个都要，因为边确实是两条。
        data_slots = _data_slot_count(fields.get("op_type"), len(inputs))
        multi = data_slots > 1
        upstream_ids: list[int] = []
        for slot, source in enumerate(inputs):
            if slot < data_slots:
                slot_index = slot if multi else None
                # 上游是 phase 型节点（DQ）时，缓冲与 scale 都**引用生产者的
                # 文件**，不由消费者另起名字：DQ 自命名 output_buffer_<self>，
                # 而它 phase1 的输出就是这一路的 scale（p1 = p0/256）。
                # 实测 MatMul 16 -> input_buffer "output_buffer_17.bin"、
                # input_sf "output_buffer_phase_1_17.bin"（17 是上游 DQ）。
                # zp 例外，始终由消费者命名（input_zp_16.bin）。
                if DQ_META_KEY in getattr(source, "meta", {}):
                    buffer_name = names.phase_output_buffer_self(
                        node_ids[source])
                    scale_name = names.phase_output_buffer(node_ids[source], 1)
                else:
                    buffer_name = names.data_buffer(node_id, slot_index)
                    scale_name = names.scale(node_id, slot_index)
                fields[f"input_buffer_{slot}" if multi else "input_buffer"] = (
                    buffer_name)
                fields[f"input_{slot}_sf" if multi else "input_sf"] = scale_name
                # 每个 *_sf 都配一个 *_zp。对称量化下 zp 恒为 0，但**文件必须
                # 存在** —— 实测 381 个 zp 文件全是 4 字节 int32 的 0，
                # 缺一个就是悬空引用。
                fields[f"input_{slot}_zp" if multi else "input_zp"] = (
                    names.zero_point(node_id, slot_index))
            fields[names.port_node_id_key("input", slot)] = node_ids[source]
            upstream_ids.append(node_ids[source])
        if inputs:
            # 同样分两组写：端口 >= 10 的键名多一个下划线。
            if upstream_ids[:10]:
                fields[names.residual_buffer_key("input", 0)] = upstream_ids[:10]
            if upstream_ids[10:]:
                fields[names.residual_buffer_key("input", 10)] = upstream_ids[10:]

            # MatMul 的第二个 operand 走**权重通路**、不占输入槽，所以
            # `input_count` 比实际入边少记 1（实测参考产物 64 个 MatMul 全如此，
            # 且 Σ input_count + MatMul 数 == 边数 这条恒等式依赖它）。
            if fields["op_type"] == "MatMul":
                fields["MatMul_input_as_weight"] = 1
                fields["input_count"] = max(1, len(inputs) - 1)
            else:
                fields["input_count"] = len(inputs)
        elif node in entry_ids:
            # 入口算子的上游是入口缓冲节点，字段照常写——否则那个缓冲节点的
            # output_buffer 找不到读者，规则 2 会判为悬空引用。
            fields["input_buffer"] = names.data_buffer(node_id)
            fields["input_sf"] = names.scale(node_id)
            fields[names.residual_buffer_key("input", 0)] = [entry_ids[node]]
            fields[names.port_node_id_key("input", 0)] = entry_ids[node]
            fields["input_count"] = 1

        # 输出缓冲区按**消费者**编号。多个消费者时记第一个，其余靠
        # residual_output_buffer 列出——参考产物就是这样。
        downstream = consumers[node]
        if not downstream and node in exit_ids:
            # 末端算子的输出流向出口缓冲节点，按消费者（即该缓冲节点）编号。
            fields["output_buffer"] = names.data_buffer(exit_ids[node])
            fields[names.residual_buffer_key("output", 0)] = [exit_ids[node]]
            fields[names.port_node_id_key("output", 0)] = exit_ids[node]
            edges.append(Edge(node_id, exit_ids[node],
                              _shape_of(node) or "unknown"))
        if downstream:
            first = downstream[0]
            first_inputs = _tensor_inputs(first, emittable)
            # 槽号按**消费者**的数据槽规则算，不是它的入边数 ——
            # 见 _data_slot_count 的说明。
            first_slots = _data_slot_count(
                _op_type_of(first), len(first_inputs))
            index = first_inputs.index(node)
            slot = index if first_slots > 1 else None
            # phase 型节点（带 rtl_version 的 DQ）**自命名** output_buffer，
            # 其余按消费者编号。实测 37 个自命名节点与 37 个 rtl_version 完全重合：
            #   DynamicScaling 36 + Llama2ActivationDQ 1 -> output_buffer_<self>
            #   其余 152 个算子节点 -> input_buffer_<消费者>
            # 判据就是「有没有 rtl_version」，不需要额外规则。
            if (dq_spec is not None
                    or op_type in ("DynamicScaling", "Llama2ActivationDQ")):
                # phase 型节点自命名（见 phase_output_buffer_self）。
                # 不能只看 dq_spec：Llama2ActivationDQ 的标记是在字段
                # 发射时才打上的，那时 output_buffer 已经写过了。
                fields["output_buffer"] = (
                    names.phase_output_buffer_self(node_id))
            elif index >= first_slots:
                # 这一路流进消费者的**权重通路**（MatMul 的第二个
                # operand），不占数据槽，所以按 `weight_buffer_<消费者>`
                # 命名而不是 `input_buffer_<消费者>`。
                # 实测 Split 32 -> MatMul 16：Split 的 output_buffer
                # 是 weight_buffer_187.bin。写成 input_buffer 会悬空 ——
                # 消费者那侧根本没声明这个槽。
                fields["output_buffer"] = names.weight_buffer(
                    node_ids[first])
            else:
                fields["output_buffer"] = names.data_buffer(
                    node_ids[first], slot)

            # `residual_output_buffer` 与端口字段必须成对出现——三份连接信息
            # （edge / outputN / residual_*）要同步。原实现只写了 outputN，
            # 漏了 residual 这一份，对方按它推依赖时会缺边。
            #
            # 端口号 >= 10 时键名多一个下划线，所以分两组写（见 gml_names）。
            plain = [node_ids[c] for c in downstream[:10]]
            suffixed = [node_ids[c] for c in downstream[10:]]
            if plain:
                fields[names.residual_buffer_key("output", 0)] = plain
            if suffixed:
                fields[names.residual_buffer_key("output", 10)] = suffixed
            for index, consumer in enumerate(downstream):
                fields[names.port_node_id_key("output", index)] = \
                    node_ids[consumer]

        # 权重按**本节点**编号——它属于节点自己，不属于某条边（规则 2）。
        # 名字记在 `weight_param` 里（不是 GML 字段，只给写盘用），让写盘阶段
        # 能从 FX 图取到对应的 f32 张量。
        # output_sf / output_zp：做计算的算子带，纯布局的不带（见 _OUTPUT_SCALE_OPS）。
        # DQ 与 RMSNorm 在各自分支里已经写过，这里跳过避免覆盖它们的 dtype。
        if (op_type in _OUTPUT_SCALE_OPS
                and dq_spec is None and rms_norm is None):
            fields["output_sf"] = names.output_scale(node_id)
            fields["output_sf_dtype"] = "float16"
            fields["output_zp"] = names.output_zero_point(node_id)

        # FPSU 定标三族。实测哪些算子带它：
        #   Gemm 7、MatMul 64、KV_Cache_DMA 2（各 1 组）
        #   EltwiseAdd 2、EltwiseMul 1（各 2 组，逐槽）
        # 三个文件的宽度各不相同（fp16 / u8 / fp32），对应硬件 FPSU 的三个操作数。
        if op_type in _FPSU_OPS:
            # attention 定标（1/√head_dim）由 fuse_pim / split_heads 记在 meta 上。
            # 用一个**非 GML 字段**把它传给写盘阶段：名字以 pim_ 开头，
            # `_referenced_buffers` 与序列化器都只认 .bin 结尾的值，所以
            # 它既不会被当成文件名、也不会写进 GML 文本。
            scale = node.meta.get(ATTENTION_SCALE_META_KEY)
            if scale is not None:
                fields["pim_attention_scale"] = float(scale)

            slots = _FPSU_SLOTS.get(op_type, 1)
            for index in range(slots):
                suffix = index if slots > 1 else None
                fields[f"Scaling_buffer_file_{index}" if suffix is not None
                       else "Scaling_buffer_file"] = names.fpsu_scale(
                           node_id, suffix)
                fields[f"Bias_buffer_file_{index}" if suffix is not None
                       else "Bias_buffer_file"] = names.fpsu_bias(
                           node_id, suffix)
                fields[f"Scaling_PS_buffer_file_{index}" if suffix is not None
                       else "Scaling_PS_buffer_file"] = names.fpsu_post_shift(
                           node_id, suffix)

        weight_param = _weight_param_of(node)
        if weight_param:
            fields["weight_buffer"] = names.weight_buffer(node_id)
            fields["weight_sf"] = names.weight_scale(node_id)
            fields["weight_zp"] = names.weight_zero_point(node_id)

        # DynamicScaling：4 相字段族 + 每相的缓冲与定标系数。
        # phase 在 GML 里**不是独立节点**，而是同一节点内的 *_phase_<k> 字段。
        if dq_spec is not None:
            fields.update(hw_table.top_level_fields("DynamicScaling"))
            fields["use_dynamic_quantization"] = 1
            # 节点级字段（不带 _phase_ 后缀）。dtype 说的是**整个节点**的
            # 输入输出：吃 fp16、吐 int8，与 p3 那一相一致。
            fields["input_buffer_dtype"] = "float16"
            fields["input_data_extensions"] = hw_table.data_extension("float16")
            fields["output_buffer_dtype"] = "int8"
            fields["output_data_extension"] = hw_table.data_extension("int8")
            fields["rtl_version"] = hw_table.RTL_VERSION
            fields["transpose"] = 1
            # 两个形状字段。`original_shape` 是被量化张量的形状，
            # `output_shape_by_group` 把最后一维按 group_size 拆成
            # `[..., groups, group_size]` —— 硬件按这个分组求 absmax。
            # 必须按真实张量算，不能写死（实测 4096 与 11008 两种）。
            shape = _shape_tuple_of(node.args[0]) if node.args else None
            if shape is not None:
                fields["original_shape"] = _shape_literal(shape)
                fields["output_shape_by_group"] = _shape_literal(
                    tuple(shape[:-1]) + (dq_spec.groups, dq_spec.group_size))
            for phase in range(PHASE_COUNTS["DynamicScaling"]):
                fields.update(hw_table.phase_fields(
                    "DynamicScaling", phase, group_size=dq_spec.group_size))
                fields[f"input_buffer_phase_{phase}"] = (
                    names.phase_input_buffer(node_id, phase))
                fields[f"output_buffer_phase_{phase}"] = (
                    names.phase_output_buffer(node_id, phase))
                fields[f"Scaling_buffer_phase_{phase}"] = (
                    names.phase_fpsu_scale(node_id, phase))
                fields[f"Scaling_PS_buffer_phase_{phase}"] = (
                    names.phase_fpsu_post_shift(node_id, phase))
                fields[f"Bias_buffer_phase_{phase}"] = (
                    names.phase_fpsu_bias(node_id, phase))
            # p1 走恒等表、p2 走倒数表（两张都自行合成，见 contracts/gml_lut）。
            fields["LUT_phase_1"] = names.phase_lut(node_id, 1)
            fields["LUT_phase_2"] = names.phase_lut(node_id, 2)
            # p3 由 Kantor 做浮点转定点，这是量化真正落定的一相。
            fields["kantor_A_scale_buffer_file_phase_3"] = (
                names.phase_kantor_scale(node_id, 3))
            fields["kantor_A_bias_buffer_file_phase_3"] = (
                names.phase_kantor_bias(node_id, 3))
            fields["kantor_A_Shift_buffer_file_phase_3"] = (
                names.phase_kantor_shift(node_id, 3))
            # 输出的 requant scale 逐字节等于 phase1 的输出。
            fields["output_sf"] = names.output_scale(node_id)
            fields["output_sf_dtype"] = "float16"
            fields["output_zp"] = names.output_zero_point(node_id)

        # KV_Cache_DMA：定点通路 + updates 那一路的量化参数。
        kv_dma = node.meta.get(KV_DMA_META_KEY)
        if kv_dma is not None:
            fields.update(hw_table.top_level_fields("KV_Cache_DMA"))
            fields["updates_sf"] = names.kv_updates_scale(node_id)
            fields["updates_sf_dtype"] = "float16"
            fields["updates_zp"] = names.kv_updates_zero_point(node_id)
            # 索引那一路是 int16 的位置下标，不参与数值计算 ——
            # 对方的 L2Analyzer 靠这个标记忽略它。
            fields["use_input_buffer_1"] = "L2A_ignore"

        # Split：一进多出，`num_heads` 记输出个数。
        split = node.meta.get(SPLIT_META_KEY)
        if split is not None:
            fields["num_heads"] = str(split.heads)
            fields["axis"] = 1
            fields["kantor_mode"] = "off"
            fields["use_dynamic_quantization"] = 1

        # RoPE 折叠节点：一整套子块配置 + 每个子块的定标/零点/Kantor 文件。
        rope = node.meta.get(ROPE_META_KEY)
        if rope is not None:
            is_dq = op_type == "Llama2ActivationDQ"
            fields.update(hw_table.rope_fields(dq=is_dq))
            fields["num_heads"] = str(_head_count_of(node))
            if is_dq:
                fields["rtl_version"] = hw_table.RTL_VERSION
                fields["dq_contraction"] = 1
                fields["transpose"] = 1
                for phase in range(PHASE_COUNTS["DynamicScaling"]):
                    fields.update(hw_table.phase_fields(
                        "Llama2ActivationDQ", phase, group_size=128))
                    fields[f"input_buffer_phase_{phase}"] = (
                        names.phase_input_buffer(node_id, phase))
                    fields[f"output_buffer_phase_{phase}"] = (
                        names.phase_output_buffer(node_id, phase))
                    fields[f"Scaling_buffer_phase_{phase}"] = (
                        names.phase_fpsu_scale(node_id, phase))
                    fields[f"Scaling_PS_buffer_phase_{phase}"] = (
                        names.phase_fpsu_post_shift(node_id, phase))
                    fields[f"Bias_buffer_phase_{phase}"] = (
                        names.phase_fpsu_bias(node_id, phase))
                fields["LUT_phase_1"] = names.phase_lut(node_id, 1)
                fields["LUT_phase_2"] = names.phase_lut(node_id, 2)
                fields["kantor_A_scale_buffer_file_phase_3"] = (
                    names.phase_kantor_scale(node_id, 3))
                fields["kantor_A_bias_buffer_file_phase_3"] = (
                    names.phase_kantor_bias(node_id, 3))
                fields["kantor_A_Shift_buffer_file_phase_3"] = (
                    names.phase_kantor_shift(node_id, 3))
                # 复用已有的 DQ 写盘路径：给这个 FX 节点打上同样的标记，
                # `_dq_specs` 会按 label 反查 node_id 然后 write_dq_phases。
                from graph.quant_pass import DynamicScalingSpec
                numel = _numel_of_fx(node)
                node.meta[DQ_META_KEY] = DynamicScalingSpec(
                    group_size=128, numel=numel or 128,
                    is_attention_scores=False)
            for unit, block in hw_table.ROPE_UNITS:
                fields[f"Scaling_buffer_file_{unit}_{block}"] = (
                    names.rope_scale(node_id, unit, block))
                fields[f"Scaling_PS_buffer_file_{unit}_{block}"] = (
                    names.rope_post_shift(node_id, unit, block))
            # 每个子块的输出量化参数（含广播那一路）。
            for block in hw_table.ROPE_SCALE_BLOCKS:
                fields[f"{block}_sf"] = names.rope_quant_scale(node_id, block)
                fields[f"{block}_zp"] = (
                    names.rope_quant_zero_point(node_id, block))
            # Kantor：cos/sin 各有 A/B 两侧，末段加法只有 A 侧。
            for block in hw_table.ROPE_KANTOR_BLOCKS:
                for side in ("A", "B"):
                    fields[f"Kantor_{side}_{block}_bias_buffer_file"] = (
                        names.rope_kantor_bias(node_id, block, side))
                    fields[f"Kantor_{side}_Shift_{block}"] = (
                        names.rope_kantor_shift(node_id, block, side))
                # A 侧的 scale 由 B 侧那份承担，实测只有 B 有 scale_buffer。
                fields[f"Kantor_B_{block}_scale_buffer_file"] = (
                    names.rope_kantor_scale(node_id, block, "B"))
            for key, fn in (("bias_buffer_file", names.rope_kantor_bias),
                            ("scale_buffer_file", names.rope_kantor_scale)):
                fields[f"Kantor_A_Llama2Activation_add_{key}"] = fn(
                    node_id, "Llama2Activation_add", "A")
            fields["Kantor_A_Shift_Llama2Activation_add"] = (
                names.rope_kantor_shift(node_id, "Llama2Activation_add", "A"))
            # 两个中间态：x*cos 与 rotate_half(x)*sin。
            fields["cos_mul_output"] = names.rope_intermediate(node_id, "cos")
            fields["sin_mul_output"] = names.rope_intermediate(node_id, "sin")

        # RMSNorm 绑在向量单元上，配置走 vpu_params 子块，还带 eps 常量。
        # 三处 sf 都是 **fp32**（实测 2 个 RMSNorm 节点全是 float32，
        # 其余算子是 float16）——这是唯一的 dtype 例外。
        nested: dict[str, dict[str, object]] = {}
        if rms_norm is not None:
            # 用 `target`（带点的参数路径）而不是 `name`（下划线化的节点名）——
            # 写盘时要靠它逐段 getattr 取出真实张量。
            weight_name = (
                str(rms_norm.weight_node.target) if rms_norm.weight_node else None)
            if weight_name:
                # 一维缩放张量走 weight_buffer，per-tensor 量化。
                fields["weight_buffer_dtype"] = "int8"
                fields["weight_buffer"] = names.weight_buffer(node_id)
                fields["weight_sf"] = names.weight_scale(node_id)
                fields["weight_zp"] = names.weight_zero_point(node_id)
                weight_params[node_id] = weight_name

            fields["input_sf_dtype"] = "float32"
            fields["weight_sf_dtype"] = "float32"
            fields["output_sf_dtype"] = "float32"
            fields["output_zp"] = names.output_zero_point(node_id)
            fields["RMSNorm_Add_Const"] = names.rms_norm_epsilon(node_id)
            fields["Use_Scaling"] = 0
            nested["vpu_params"] = {
                # -1 表示沿最后一维归约，实测如此。
                "Vpu_Axis": -1,
                "input_scale_factor_buffer": names.scale(node_id),
                "output_scale_factor_buffer": names.output_scale(node_id),
                "Weights_buffer_file": names.weight_buffer(node_id),
                "weights_scaling_buffer_file": names.weight_scale(node_id),
                "bias_buffer_file": names.rms_norm_epsilon(node_id),
            }

        contraction = _contraction_of(node)
        gml_node = Node(node_id, fields, contraction, nested)
        if weight_param:
            weight_params[node_id] = weight_param
        # 上游是 DQ 的输入槽，dtype 覆盖成 int8（放最后，压过默认的 fp16）。
        if upstream_int8:
            gml_node.fields.update(upstream_int8)
            gml_node.fields['input_data_extensions'] = (
                hw_table.data_extension('int8'))

        gml_nodes.append(gml_node)

        # 边带形状，节点不带。
        shape = _shape_of(node)
        for consumer in downstream:
            edges.append(Edge(node_id, node_ids[consumer], shape or "unknown"))

    return gml_nodes, edges, weight_params
