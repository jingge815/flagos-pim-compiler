"""为 FX 图传播张量布局，并标记需要重分布的边。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch.fx import GraphModule, Node

from contracts.gml_quant import WEIGHT_LAYOUT, QuantLayout
from contracts.mem_layout import row_major_strides
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
from contracts.unified_ir import (
    STAGE_SPECS,
    mark_stage,
    validate_graph_dimensions,
)
from graph.strategy import ShardStrategy, llama_strategy

REPLICATE = Placement("Replicate")

# 取两个操作数（第二个可以是标量）的逐元素算子。`_propagate_dpu_node` 按这张表
# 决定怎么把实参喂给规则函数。
_BINARY_TARGETS = (
    torch.ops.aten.add.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.div.Tensor,
)


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


def _earliest_consumer_layer(node: Node, seen: set[str] | None = None) -> int | None:
    """非层节点最早被哪一层的算子消费。

    RoPE 的 cos/sin 表不算某一层的权重，但第 0 层的 RoPE 就要用它。一律归
    最后一段会让表在最后一段算完再倒着送回第 0 段。
    """
    seen = seen or set()
    if node.name in seen:
        return None
    seen.add(node.name)
    layers = []
    for user in node.users:
        inner = _layer_of_node(user)
        if inner is None:
            inner = _earliest_consumer_layer(user, seen)
        if inner is not None:
            layers.append(inner)
    return min(layers) if layers else None


def _dpus_of_node(node: Node, strategy: ShardStrategy, num_layers: int) -> tuple[int, ...]:
    """返回节点张量所在的 DPU。

    非层节点落在**最早消费它的那一层**所在段；没有层消费者（图出口那一类）
    才归最后一段。
    """
    if strategy.num_stages == 1:
        return strategy.dpu_ids
    layer = _layer_of_node(node)
    if layer is None:
        layer = _earliest_consumer_layer(node)
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


def _dtype_of(node: Node) -> str:
    """节点输出的元素类型名。

    这是 dtype 从 PyTorch 进入统一 IR 的**唯一入口**：之后全链路查
    `spec.dtype`，不再问 `meta["val"]`。跨维校验会守住两者不漂移。

    `val` 可能是元组（`split_with_sizes` 这类节点带的是形状列表），
    那不是张量、没有类型，返回空串。
    """
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return ""
    return str(val.dtype).removeprefix("torch.")


def _host_spec(dtype: str = "") -> PIMTensorSpec:
    """构造无 DPU 分片的 host 张量规格。"""
    spec = PIMTensorSpec(DEVICE_HOST, REPLICATE, "transient", None, {}, None,
                         dtype=dtype)
    spec.validate()
    return spec


def _dpu_spec(
    placement: Placement,
    shape: tuple[int, ...],
    dpu_ids: tuple[int, ...],
    residency: Literal["transient", "pinned"] = "transient",
    *,
    dtype: str = "",
    quant: QuantLayout | None = None,
) -> PIMTensorSpec:
    spec = PIMTensorSpec(
        DEVICE_DPU,
        placement,
        residency,
        None,
        _shard_map(shape, placement, dpu_ids),
        placement.reduce_type,
        dtype=dtype,
        quant=quant,
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
        local = shape[:dim] + (width,) + shape[dim + 1 :]
        return {
            dpu_id: TensorShardDetail(
                dpu_id=dpu_id,
                shard_dim=dim,
                start_idx=i * width,
                end_idx=(i + 1) * width,
                local_shape=local,
                # 排布层：分片一出生就把「怎么摆」写成数据。原先这是
                # `bytes_of()` 里的隐含假设，字段恒为空 —— 等于没人声明。
                elem_strides=row_major_strides(local),
            )
            for i, dpu_id in enumerate(dpu_ids)
        }
    numel = prod(shape)
    full = tuple(shape)
    # replicate / partial 每台持有完整形状，排布同样是行主序紧密。
    return {
        dpu_id: TensorShardDetail(dpu_id, -1, 0, numel, full,
                                  elem_strides=row_major_strides(full))
        for dpu_id in dpu_ids
    }



# 插在权重和 linear 之间、但不改变切分的算子。切片取子区间，类型转换
# 只改 dtype，两者都要穿透才能找到真正消费这块权重的 linear。
_LAYOUT_NEUTRAL = frozenset((
    torch.ops.aten.slice.Tensor,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
))


def _consumers_through_slices(node: Node) -> list[Node]:
    """权重的真实消费者。

    分块会在权重和 linear 之间插入切片和类型转换，只看直接用户会把
    这块权重误判成没有 linear 消费，从而丢掉按列切分。
    """
    seen = []
    pending = list(node.users)
    visited = set()
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        # 两维同时切会产生切片套切片，要顺着链一路往下找到真实消费者。
        if user.target in _LAYOUT_NEUTRAL:
            pending.extend(user.users)
        else:
            seen.append(user)
    return seen


def _weight_spec(node: Node, strategy: ShardStrategy, num_layers: int) -> PIMTensorSpec:
    """按策略为权重构造常驻的 host 或 DPU 张量规格。"""
    shape = tuple(node.meta["val"].shape)
    dt = _dtype_of(node)
    # 权重量化布局：定点权重按 WEIGHT_LAYOUT，浮点权重不带量化。
    qz = WEIGHT_LAYOUT if dt in ("int4", "int8") else None
    dpus = _dpus_of_node(node, strategy, num_layers)
    # propagate_specs 的入口断言已保证全图每个节点都有 device，这里不必再容忍缺失
    dpu_users = [u for u in _consumers_through_slices(node)
                 if u.meta[DEVICE_META_KEY] == DEVICE_DPU]
    linear_users = [
        u for u in dpu_users if u.target in (torch.ops.aten.linear.default, torch.ops.aten.addmm.default)
    ]
    if linear_users:
        mode = strategy.match(node.target)
        if mode is None:
            raise ValueError(f"权重 {node.target} 被 linear 消费，但策略未指定其切法")
        return _dpu_spec(
            Placement("Shard", 0 if mode == "col" else 1), shape, dpus,
            residency="pinned", dtype=dt, quant=qz,
        )
    if dpu_users:
        return _dpu_spec(REPLICATE, shape, dpus, residency="pinned",
                         dtype=dt, quant=qz)
    return _host_spec(dt)


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


def _last_dim_complete(
    node: Node, x: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """要求张量沿末轴完整：末轴是归约轴，切了就只能拿到部分结果。

    每台 DPU 只拿到一段时，各自归约出来的是局部值，拼起来不等于全局值。
    所以末轴上是 Shard 的输入要先聚合成 Replicate（`_append_edge` 据此发
    all_gather），按别的维切分可以——那些维在归约里是独立的行。输出布局
    与输入一致（`mean.dim` 带 keepdim，`softmax` 逐元素归一化不改形状）。
    """
    x_node, x_spec = x
    placement = x_spec.placement
    if placement.kind == "Partial":
        raise _unsupported_layouts(node, x_spec)

    if placement.kind == "Shard" and placement.dim == x_node.meta["val"].ndim - 1:
        return [(x_node, _dpu_req(REPLICATE))], REPLICATE
    return [(x_node, _dpu_req(placement))], placement


def _rule_reduce_last(
    node: Node, x: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """沿最后一维归约（`aten.mean.dim`，RMSNorm 的均方步）。见 `_last_dim_complete`。"""
    return _last_dim_complete(node, x)


def _rule_softmax(
    node: Node, x: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """`aten._softmax(self, dim, half_to_float)`：沿 dim 归一化，输出形状不变。

    只认 dim=-1（注意力分数那条，拆头后每头一个节点）。归约轴不能切分，理由与
    `_rule_reduce_last` 相同：各自归一化不等于全局归一化。别的 dim 会改变
    「哪根轴要完整」的判断，所以显式拒绝，而不是按末轴算完悄悄给错布局。
    """
    dim = _operand_value(node, 1, "dim")
    if dim not in (-1, node.meta["val"].ndim - 1):
        raise NotImplementedError(
            f"{node.name}: softmax 只支持 dim=-1（注意力分数那条），收到 dim={dim!r}"
        )
    return _last_dim_complete(node, x)


def _rule_matmul(
    node: Node,
    a: tuple[Node, PIMTensorSpec],
    b: tuple[Node, PIMTensorSpec],
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """`aten.matmul(a, b)`：沿 a 的末轴与 b 的倒数第二轴收缩。

    两个操作数都来自图节点（拆头后是逐头的激活 x 激活），没有权重那一路。

    收缩维上切了就不对：每台 DPU 只算出一段的部分和，拼起来才是全和，所以那
    一路要求聚合成 Replicate。别的地方切分要按「某一维映射到输出的哪一维」逐
    操作数算，拆头后的注意力全是整副本（每个头一个节点），没有可验证的形态，
    所以显式拒绝而不是猜一个布局——猜错是数值错且不报错。

    实测拆头后的注意力就是这条：每头 `[1, 1, Tq, hd] x [1, 1, hd, Tp+Tq]`。
    """
    a_node, a_spec = a
    b_node, b_spec = b
    for x_node, x_spec, contract_dim in ((a_node, a_spec, -1), (b_node, b_spec, -2)):
        if x_spec.placement.kind == "Partial":
            raise _unsupported_layouts(node, x_spec)
        if x_spec.placement.kind == "Shard" and (
            x_spec.placement.dim != contract_dim % x_node.meta["val"].ndim
        ):
            raise NotImplementedError(
                f"{node.name}: {x_node.name} 在维 {x_spec.placement.dim} 上是 "
                f"Shard，非收缩维的切分要按维映射另算，不在第 1 阶段契约内"
            )
    # 收缩维上是 Shard 时要求聚合成完整副本（`_append_edge` 据此发 all_gather）；
    # 本来就不是 Shard 的也写 REPLICATE，两者同一条路径。
    return ([(a_node, _dpu_req(REPLICATE)), (b_node, _dpu_req(REPLICATE))],
            REPLICATE)


def _rule_convert_dtype(
    node: Node, x: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """`aten.to.dtype`：逐元素类型转换，下标不动，所以布局原样传。

    与视图族分开写：那一族要聚合是因为元素会换位置，这个不换。
    部分和也原样传：各台先转再归约，与不分块时 linear 直接产出低精度部分和相同。
    """
    x_node, x_spec = x
    return [(x_node, _dpu_req(x_spec.placement))], x_spec.placement


def _rule_attention(
    node: Node,
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """`scaled_dot_product_attention(q, k, v[, mask])`：逐头算，输出给所有 DPU。

    q/k/v 都要完整副本——注意力的 K/V 来自**缓存**而不是当前 token，切分后
    每台 DPU 只拿得到自己那些头，读不回完整历史。输出 Replicate：每一台都能
    独立算出同样的结果（各自读自己那份缓存），下游再按需 local_slice。
    """
    reqs = []
    for position, keyword in ((0, "query"), (1, "key"), (2, "value")):
        source, spec = _tensor_operand(node, position, keyword)
        if spec.placement.kind == "Partial":
            raise _unsupported_layouts(node, spec)
        reqs.append((source, _dpu_req(REPLICATE)))
    mask = _operand_value(node, 3, "attn_mask")
    if isinstance(mask, Node):
        reqs.append((mask, _dpu_req(REPLICATE)))
    return reqs, REPLICATE


def _rule_embedding(
    node: Node,
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """`aten.embedding(table, indices)`：表整份驻留，输出跟着索引的布局。

    两个操作数都要发 req——只处理第一个的话，索引那一路没有重分布边，
    命令编码取 `shard_map[dpu_id]` 会直接 KeyError。
    """
    table = _tensor_operand(node, 0, "weight")
    indices = _tensor_operand(node, 1, "indices")
    if table[1].placement.kind == "Partial":
        raise _unsupported_layouts(node, table[1])
    return [(table[0], _dpu_req(REPLICATE)),
            (indices[0], _dpu_req(REPLICATE))], REPLICATE


def _rule_view(
    node: Node, x: tuple[Node, PIMTensorSpec]
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """视图族：元素会换位置，切分维编号跟着重排动，所以输入必须是完整副本。

    下游要 Shard 时，由 `_append_edge` 发一条 `local_slice`，把副本里属于
    本 DPU 的那段拷到分片缓冲。
    """
    x_node, x_spec = x
    if x_spec.placement.kind == "Partial":
        raise _unsupported_layouts(node, x_spec)
    return [(x_node, _dpu_req(REPLICATE))], REPLICATE


RULE_TABLE = {
    torch.ops.aten.linear.default: _rule_linear,
    torch.ops.aten.addmm.default: _rule_addmm,
    torch.ops.aten.add.Tensor: _rule_elementwise_binary,
    torch.ops.aten.mul.Tensor: _rule_elementwise_binary,
    torch.ops.aten.sub.Tensor: _rule_elementwise_binary,
    torch.ops.aten.div.Tensor: _rule_elementwise_binary,
    torch.ops.aten.tanh.default: _rule_unary,
    # 逐元素一元：布局原样传下去。RMSNorm 拆开后的 `pow`/`rsqrt`、
    # 以及 RoPE 的 `neg`、MLP 的 `silu` 都是这一类。
    torch.ops.aten.pow.Tensor_Scalar: _rule_unary,
    torch.ops.aten.rsqrt.default: _rule_unary,
    torch.ops.aten.neg.default: _rule_unary,
    torch.ops.aten.silu.default: _rule_unary,
    torch.ops.aten.relu.default: _rule_unary,
    torch.ops.aten.sigmoid.default: _rule_unary,
    torch.ops.aten.exp.default: _rule_unary,
    torch.ops.aten.sqrt.default: _rule_unary,
    torch.ops.aten.reciprocal.default: _rule_unary,
    # 归约：只支持沿最后一维（RMSNorm 的形态）。
    torch.ops.aten.mean.dim: _rule_reduce_last,
    # 拆头之后的注意力：每头一次 softmax（dim=-1）与两次 batched matmul。
    # 这两族原本不在表里，`split_attention_heads` 一跑就整条编译失败。
    torch.ops.aten._softmax.default: _rule_softmax,
    torch.ops.aten.matmul.default: _rule_matmul,
    # 类型转换：逐元素，下标不动，布局原样传。
    torch.ops.aten.to.dtype: _rule_convert_dtype,
    torch.ops.aten.to.dtype_layout: _rule_convert_dtype,
    # 分块把 float16 提升到 float32 时用的就是这个。与 to.dtype 一样逐元素，
    # 下标不动，所以布局原样传。
    torch.ops.aten._to_copy.default: _rule_convert_dtype,
    # 切片与拼接也是视图：元素的**位置**变了（切片取子区间、拼接把两段接起来），
    # 所以与视图族同一条规则——输入必须完整副本，下游要 Shard 时发 local_slice。
    # `cat` 的实参里第一个是张量列表，`_tensor_inputs` 已按列表展开。
    # `alias` 是恒等，布局原样传。
    torch.ops.aten.alias.default: _rule_unary,
    torch.ops.aten.embedding.default: _rule_embedding,
    torch.ops.aten.scaled_dot_product_attention.default: _rule_attention,
    torch.ops.aten.slice.Tensor: _rule_view,
    torch.ops.aten.cat.default: _rule_view,
    torch.ops.aten.view.default: _rule_view,
    torch.ops.aten.reshape.default: _rule_view,
    torch.ops.aten.transpose.int: _rule_view,
    torch.ops.aten.permute.default: _rule_view,
    torch.ops.aten.unsqueeze.default: _rule_view,
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
        return _host_spec(_dtype_of(src))
    return _dpu_spec(req.placement, tuple(src.meta["val"].shape), dst_dpus,
                     dtype=_dtype_of(src))


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
    dst_dpus: tuple[int, ...], dst_spec: PIMTensorSpec | None = None,
) -> RedistributeEdge | None:
    """比较边两端规格；相同返回 `None`，否则返回重分布标记。"""
    if dst_spec is None:
        dst_spec = _required_spec(src, req, dst_dpus)
    if (
        actual.device == req.device
        and actual.placement == req.placement
        and set(actual.shard_map) == set(dst_spec.shard_map)
    ):
        return None
    # 权重切片只被与自己有交集的 DPU 持有，持有者比整块权重少。
    # 这是切片本身的结果，不是一次重分布；别的同布局、不同 DPU 集合仍要搬。
    # 类型转换插在切片和 linear 之间时也一样：它的持有者继承自切片。
    if (actual.device == req.device and actual.placement == req.placement
            and (_is_weight_slice(src)
                 or src.target is torch.ops.aten._to_copy.default)):
        return None
    # 部分和只落在部分台上时，其余台的贡献是 0，不必搬：执行计划在那些台上
    # 把这个操作数当作 0。
    if (actual.device == req.device == DEVICE_DPU
            and actual.placement.kind == req.placement.kind == "Partial"
            and set(actual.shard_map) <= set(dst_spec.shard_map)):
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
        dtype=_dtype_of(src),
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
            node.meta[SPEC_META_KEY] = _host_spec(_dtype_of(node))
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
            node.meta[SPEC_META_KEY] = _host_spec(_dtype_of(node))
        else:
            _propagate_dpu_node(node, edges, strategy, num_layers)
    # 出口校验：此刻全图应当满足 STAGE_SPECS 的跨维不变式
    # （spec.dtype 与 val.dtype 不漂移、切分维落在本地形状的秩内）。
    validate_graph_dimensions(gm, stage=STAGE_SPECS)
    mark_stage(gm, STAGE_SPECS)
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



def _weight_root(node: Node) -> Node | None:
    """沿切片链找到最里面的权重。不是在切权重时返回 None。

    两维同时切时，归约维那一刀的源是输出特征那一刀，不是 `get_attr`。
    只认直接源会把第二刀当成普通视图，切分随之丢掉。
    """
    seen = set()
    while node.target is torch.ops.aten.slice.Tensor:
        if node in seen:
            return None
        seen.add(node)
        src = node.args[0] if node.args else None
        if not isinstance(src, Node):
            return None
        node = src
    return node if node.op == "get_attr" else None


def _is_weight_slice(node: Node) -> bool:
    """这个切片是不是在切一块权重。

    权重是 `get_attr`，激活是别的算子的输出。只对权重切片保持切分布局，
    激活上的切片仍走视图规则。切片套切片时顺着链往里找。
    """
    return _weight_root(node) is not None


def _slice_span(node: Node, src_len: int) -> tuple[int, int, int]:
    """切片的轴与真实区间。上界超过源长度时截到末尾。"""
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    start = int(node.args[2]) if len(node.args) > 2 else 0
    end = int(node.args[3]) if len(node.args) > 3 else src_len
    return dim, max(start, 0), min(end, src_len)


def _clipped_detail(det: TensorShardDetail, dim: int,
                     start: int, end: int) -> TensorShardDetail | None:
    """把一条分片夹进切片区间，没有交集就返回空。"""
    lo = max(det.start_idx, start)
    hi = min(det.end_idx, end)
    if lo >= hi:
        return None
    local = list(det.local_shape)
    local[dim] = hi - lo
    local = tuple(local)
    return TensorShardDetail(det.dpu_id, det.shard_dim, lo, hi, local,
                             elem_strides=row_major_strides(local))


def _weight_slice_spec(node: Node, src_spec: PIMTensorSpec) -> PIMTensorSpec:
    """按切片落在权重的哪一段重算输出规格。

    切片轴与切分轴不同时，被切的那一维在每台上都是完整的，切完只是变短，
    切分方式照搬。轴相同时，每台只保留切片区间与自己持有区间的交集：权重
    按连续的行段存在各台上，一台拿不到别人那一段的行，按切片长度重新对半
    会把行分给没有这些行的台。没有交集的台不持有这个切片。
    """
    # 切片套切片时，区间要相对整块权重算，而不是相对外面那一刀。
    src_shape = tuple(_weight_root(node).meta["val"].shape)
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    dim, start, end = _slice_span(node, src_shape[dim])
    out_shape = tuple(node.meta["val"].shape)
    placement = src_spec.placement
    if placement.kind != "Shard" or dim != placement.dim:
        # 切的不是切分轴。源是整块权重时各台持有完整的这一维，按输出形状
        # 重新平分即可。源本身是切片时，各台持有的区间已经夹过，重新平分
        # 会把全局下标重置成从 0 开始，所以沿用源的区间，只把被切的那一维
        # 缩短到这段的长度。
        # 只对权重切片沿用源的区间。激活切片的源区间是另一套坐标系，
        # 沿用它会把行号抄到归约维上。
        # 切片是内核写出的新缓冲，不能继承源的常驻属性：标成常驻时内存规划
        # 不给它分地址，切片写到偏移 0，盖掉同一位置上的整块权重。
        if node.args[0].op == "get_attr" or not _is_weight_slice(node):
            return _dpu_spec(placement, out_shape, tuple(src_spec.shard_map),
                             residency="transient", dtype=src_spec.dtype,
                             quant=src_spec.quant)
        length = out_shape[placement.dim]
        holders = {}
        for dpu_id, det in src_spec.shard_map.items():
            local = list(out_shape)
            local[placement.dim] = det.end_idx - det.start_idx
            local = tuple(local)
            holders[dpu_id] = TensorShardDetail(
                dpu_id, det.shard_dim, det.start_idx, det.end_idx, local,
                elem_strides=row_major_strides(local))
        spec = PIMTensorSpec(
            DEVICE_DPU, placement, "transient", None, holders,
            placement.reduce_type, dtype=src_spec.dtype, quant=src_spec.quant)
        spec.validate()
        return spec
    holders = {
        dpu_id: clipped
        for dpu_id, det in src_spec.shard_map.items()
        if (clipped := _clipped_detail(det, dim, start, end))
    }
    spec = PIMTensorSpec(
        DEVICE_DPU, placement, "transient", None, holders,
        placement.reduce_type, dtype=src_spec.dtype, quant=src_spec.quant)
    spec.validate()
    return spec


def _row_slice_linear(node: Node, x, weight, edges, strategy, num_layers) -> bool:
    """行切权重被切片后的 linear：激活跟着权重切片走，部分和只落在持有者上。

    K 段只与部分台的分片相交，激活按整段对半会把这段 K 分给没有对应权重的台。
    激活的要求按权重切片在每台的 K 区间给出，输出部分和只标在这些台上，其余台
    相当于贡献 0，后面相加时按 0 处理。不是权重切片返回 False，调用方照旧处理。
    """
    w_node = node.args[1]
    while (isinstance(w_node, Node) and w_node.target in _LAYOUT_NEUTRAL
           and w_node.target is not torch.ops.aten.slice.Tensor):
        w_node = w_node.args[0]
    if not isinstance(w_node, Node) or not _is_weight_slice(w_node):
        return False
    w_spec = w_node.meta[SPEC_META_KEY]
    if w_spec.placement != Placement("Shard", 1):
        return False
    # 这段 K 在整块权重里的起点：沿切片链找切 K 的那一刀。没切 K 时每台
    # 仍持有自己整段的 K，照旧按整段对半，不走这里。
    k_start = None
    cur = w_node
    while cur.target is torch.ops.aten.slice.Tensor:
        dim = int(cur.args[1]) if len(cur.args) > 1 else 0
        if dim in (1, -1):
            k_start = _slice_span(cur, _weight_root(cur).meta["val"].shape[1])[1]
            break
        cur = cur.args[0]
    if k_start is None:
        return False
    x_node, _ = x
    x_shape = tuple(x_node.meta["val"].shape)
    out_dim = len(x_shape) - 1
    holders = {}
    for dpu_id, det in w_spec.shard_map.items():
        local = x_shape[:-1] + (det.end_idx - det.start_idx,)
        holders[dpu_id] = TensorShardDetail(
            dpu_id, out_dim, det.start_idx - k_start, det.end_idx - k_start, local,
            elem_strides=row_major_strides(local))
    x_spec = PIMTensorSpec(DEVICE_DPU, Placement("Shard", out_dim), "transient",
                           None, holders, None, dtype=_dtype_of(x_node))
    x_spec.validate()
    dpus = tuple(sorted(holders))
    _append_edge(node, x_node, _dpu_req(x_spec.placement), edges, strategy,
                 num_layers, dst_dpus=dpus, dst_spec=x_spec)
    _append_edge(node, weight[0], _dpu_req(weight[1].placement), edges, strategy,
                 num_layers, dst_dpus=dpus)
    node.meta[SPEC_META_KEY] = _dpu_spec(
        Placement("Partial", reduce_type="sum"), tuple(node.meta["val"].shape),
        dpus, dtype=_dtype_of(node))
    return True


def _linear_output_following_weight(node: Node, out_placement: Placement,
                                     out_shape: tuple[int, ...]) -> PIMTensorSpec | None:
    """按列切的权重被切开时，输出特征跟权重行走，不按整段重新对半。

    一段切片只落在持有这些行的台上，输出的对应列也只能由这些台算。按整段
    输出重新对半会让没有这些权重行的台去算它的列。返回 None 表示不属这种情况，
    调用方照旧按整段切分。
    """
    if node.target is not torch.ops.aten.linear.default or out_placement.kind != "Shard":
        return None
    weight = node.args[1] if len(node.args) > 1 else None
    # 权重和 linear 之间可能隔着类型转换，穿透后才是切片。
    while isinstance(weight, Node) and weight.target in _LAYOUT_NEUTRAL             and weight.target is not torch.ops.aten.slice.Tensor:
        weight = weight.args[0]
    if not isinstance(weight, Node) or not _is_weight_slice(weight):
        return None
    w_spec = weight.meta[SPEC_META_KEY]
    if w_spec.placement != Placement("Shard", 0):
        return None
    # 下标沿用权重规格的全局下标：每段 linear 的输出列与它用的权重行
    # 在整块里的位置相同，拼接据此把各段排到输出的对应位置。减掉切片
    # 起点会让每段都从 0 开始，拼接时各段重叠，输出的后半列全是 0。
    holders = {}
    for dpu_id, det in w_spec.shard_map.items():
        length = det.end_idx - det.start_idx
        local = out_shape[:-1] + (length,)
        holders[dpu_id] = TensorShardDetail(
            dpu_id, out_placement.dim, det.start_idx, det.end_idx, local,
            elem_strides=row_major_strides(local))
    spec = PIMTensorSpec(
        DEVICE_DPU, out_placement, "transient", None, holders,
        out_placement.reduce_type, dtype=_dtype_of(node))
    spec.validate()
    return spec


def _stacked_cat_spec(node: Node, tensors, dim: int):
    """几段沿另一维切分、沿这一维拼接时，返回按段序拼好的输出规格。

    每段的切分维与拼接维不同，每台持有的区间在各段上一致。拼完后切分还在
    原来那一维，每台持有的长度不变。对不齐返回 None，调用方照旧聚合。
    """
    out_shape = tuple(node.meta["val"].shape)
    if dim < 0:
        dim += len(out_shape)
    first = tensors[0].meta.get(SPEC_META_KEY)
    shard_dim = first.placement.dim if first is not None else None
    if (first is None or first.placement.kind != "Shard"
            or shard_dim == dim or shard_dim is None):
        return None
    spans: dict[int, tuple[int, int]] = {}
    base = 0
    for source in tensors:
        spec = source.meta.get(SPEC_META_KEY)
        if spec is None or spec.placement != first.placement:
            return None
        if set(spec.shard_map) != set(first.shard_map):
            return None
        for dpu_id, det in spec.shard_map.items():
            seen = spans.get(dpu_id)
            if seen is None:
                spans[dpu_id] = (det.start_idx, det.end_idx)
            elif seen != (det.start_idx, det.end_idx):
                return None
        base += source.meta["val"].shape[dim]
    if base != out_shape[dim]:
        return None
    holders = {}
    for dpu_id, (lo, hi) in spans.items():
        local = list(out_shape)
        # 切分维的长度是这一台持有的，不是各段之和：各段在这一维上的
        # 区间相同，拼完之后每台还是持有这么长。
        local[shard_dim] = hi - lo
        local = tuple(local)
        holders[dpu_id] = TensorShardDetail(
            dpu_id, shard_dim, lo, hi, local,
            elem_strides=row_major_strides(local))
    spec = PIMTensorSpec(
        DEVICE_DPU, first.placement, "transient", None, holders,
        first.reduce_type, dtype=_dtype_of(node))
    spec.validate()
    return spec


def _aligned_cat_spec(node: Node, tensors, dim: int):
    """几段沿拼接维的切分对齐时，返回拼接后仍保持切分的输出规格。

    每段都沿拼接维切分，把每段的区间平移到它在输出里的起点后，同一台的
    区间要首尾相接成一段，各台的段再无重叠地铺满输出。对得齐就不用先把
    每段聚成副本，每台只拼自己那几列。对不齐返回 None，调用方照旧聚合。
    """
    out_shape = tuple(node.meta["val"].shape)
    if dim < 0:
        dim += len(out_shape)
    spans: dict[int, list[tuple[int, int]]] = {}
    base = 0
    for source in tensors:
        operand = _tensor_operand_of(source)
        spec = operand[1]
        if spec.placement.kind != "Shard" or spec.placement.dim != dim:
            return None
        # 起点落在本段范围内的是相对下标，平移到它在输出里的位置；
        # 落在本段之后的是全局下标，直接用。两行各算一次时，第二行的
        # 区间也从 0 开始，按「不小于前面总长」会把它当成全局下标，
        # 排到第一行的位置上。
        extent = source.meta["val"].shape[dim]
        for dpu_id, det in spec.shard_map.items():
            if det.start_idx < extent:
                spans.setdefault(dpu_id, []).append(
                    (base + det.start_idx, base + det.end_idx))
            else:
                spans.setdefault(dpu_id, []).append((det.start_idx, det.end_idx))
        base += extent
    if base != out_shape[dim]:
        return None
    merged = []
    holders = {}
    for dpu_id, pieces in spans.items():
        pieces.sort()
        cursor = pieces[0][0]
        for start, end in pieces:
            if start != cursor:
                return None
            cursor = end
        lo, hi = pieces[0][0], pieces[-1][1]
        merged.append((lo, hi, dpu_id))
    merged.sort()
    cursor = 0
    for lo, hi, dpu_id in merged:
        if lo != cursor:
            return None
        local = out_shape[:dim] + (hi - lo,) + out_shape[dim + 1:]
        holders[dpu_id] = TensorShardDetail(
            dpu_id, dim, lo, hi, local, elem_strides=row_major_strides(local))
        cursor = hi
    if cursor != out_shape[dim]:
        return None
    spec = PIMTensorSpec(
        DEVICE_DPU, Placement("Shard", dim), "transient", None, holders,
        None, dtype=_dtype_of(node))
    spec.validate()
    return spec


def _propagate_dpu_node(
    node: Node, edges: list[RedistributeEdge], strategy: ShardStrategy, num_layers: int
) -> None:
    """按规则表为 DPU 节点生成输入重分布和输出规格。"""
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise TypeError(f"DPU 节点 {node.name} 缺少张量 meta['val']，无法推导布局")
    rule = RULE_TABLE.get(node.target)
    if rule is None:
        # 设备侧算子没有切分规则是**编译失败**，不是「退回主机」。退回主机会让这个
        # 算子静默不在设备上跑，而图照样产出正确数值——正是要消掉的那种失真。
        raise ValueError(
            f"设备侧算子 {node.target} 缺少切分规则。"
            f"它不在 partition.HOST_ONLY 里，所以被判为设备节点；"
            f"要么给它补一条 RULE_TABLE 规则，要么把它列进 HOST_ONLY 并说明理由"
        )
    if node.target == torch.ops.aten.linear.default:
        x = _tensor_operand(node, 0, "input")
        weight = _tensor_operand(node, 1, "weight")
        _reject_tensor_scalar_operand(node, 2, "bias")
        reqs, out_placement = rule(node, x, weight)
        if out_placement.kind == "Partial" and _row_slice_linear(
                node, x, weight, edges, strategy, num_layers):
            return
    elif node.target in _BINARY_TARGETS:
        x = _tensor_operand(node, 0, "self")
        y = _tensor_or_scalar_operand(node, 1, "other")
        _reject_tensor_scalar_operand(node, 2, "alpha")
        reqs, out_placement = rule(node, x, y)
        # 两侧沿同一维切分且区间完全相同时，输出沿用这个区间。按输出形状
        # 重新平分会把一段切成两台各一半，K 维各段相加的结果因此对不齐。
        if (y is not None and x[1].placement == y[1].placement == out_placement
                and out_placement.kind == "Shard"
                and x[1].shard_map.keys() == y[1].shard_map.keys()
                and all(x[1].shard_map[d].start_idx == y[1].shard_map[d].start_idx
                        and x[1].shard_map[d].end_idx == y[1].shard_map[d].end_idx
                        for d in x[1].shard_map)):
            shape = tuple(val.shape)
            holders = {}
            for dpu_id, det in x[1].shard_map.items():
                local = list(shape)
                local[out_placement.dim] = det.end_idx - det.start_idx
                local = tuple(local)
                holders[dpu_id] = TensorShardDetail(
                    dpu_id, det.shard_dim, det.start_idx, det.end_idx, local,
                    elem_strides=row_major_strides(local))
            node.meta[SPEC_META_KEY] = PIMTensorSpec(
                DEVICE_DPU, out_placement, "transient", None, holders,
                out_placement.reduce_type, dtype=_dtype_of(node))
            node.meta[SPEC_META_KEY].validate()
            # 目标按收窄后的持有者算：K 段只落在部分台上，按全部台算会给
            # 同一区间插一条多余的收集，收集的各段区间也拼不成整块。
            for src, req in reqs:
                _append_edge(node, src, req, edges, strategy, num_layers,
                             dst_dpus=tuple(sorted(holders)))
            return
    elif node.target == torch.ops.aten.addmm.default:
        reqs, out_placement = rule(node)
    elif node.target == torch.ops.aten.matmul.default:
        # `matmul(a, b)`：两个都是图节点（拆头后是逐头的激活 x 激活），
        # 没有权重那一路，所以不走 `_rule_linear`。
        reqs, out_placement = rule(
            node,
            _tensor_operand(node, 0, "input"),
            _tensor_operand(node, 1, "other"),
        )
    elif node.target == torch.ops.aten.mean.dim:
        # `mean.dim(self, dim, keepdim)`：只支持 `dim=[-1]` 且 keepdim，
        # 也就是 RMSNorm 的形态。别的形态会改变秩，输出布局要另算。
        dims = _operand_value(node, 1, "dim")
        keepdim = _operand_value(node, 2, "keepdim")
        rank = val.ndim
        normalized = [d % rank for d in dims] if isinstance(dims, (list, tuple)) else None
        if normalized != [rank - 1] or keepdim is not True:
            raise NotImplementedError(
                f"{node.name}: 只支持沿最后一维、keepdim=True 的归约"
                f"（收到 dim={dims!r}, keepdim={keepdim!r}）"
            )
        reqs, out_placement = rule(node, _tensor_operand(node, 0, "self"))
    elif node.target == torch.ops.aten.pow.Tensor_Scalar:
        _reject_tensor_scalar_operand(node, 1, "exponent")
        reqs, out_placement = rule(node, _tensor_operand(node, 0, "self"))
    elif node.target in (torch.ops.aten.embedding.default,
                         torch.ops.aten.scaled_dot_product_attention.default):
        reqs, out_placement = rule(node)
    elif _is_weight_slice(node):
        # 分块沿特征维切权重。普通视图规则把输出判成复制，而 linear 只接受
        # 按列或按行切分的权重，于是切过的图无法传播。输出规格按切片区间
        # 重算，不能照搬整块权重的切分。源可能隔着一次类型转换，穿透它。
        src = node.args[0]
        while src.target in _LAYOUT_NEUTRAL and src.target is not torch.ops.aten.slice.Tensor:
            src = src.args[0]
        src_spec = src.meta[SPEC_META_KEY]
        node.meta[SPEC_META_KEY] = _weight_slice_spec(node, src_spec)
        reqs = [(src, _dpu_req(src_spec.placement))]
        for src_node, req in reqs:
            _append_edge(node, src_node, req, edges, strategy, num_layers)
        return
    elif node.target == torch.ops.aten.cat.default:
        # `cat([a, b, ...], dim)`：第一个实参是**张量列表**，不是单个张量。
        # 搬运的是每一段，所以逐段要完整副本。
        tensors = _operand_value(node, 0, "tensors")
        cat_dim = _operand_value(node, 1, "dim")
        if not isinstance(tensors, (list, tuple)) or not tensors:
            raise ValueError(f"{node.name}: cat 的第一个实参必须是非空张量列表")
        reqs = []
        for source in tensors:
            if not isinstance(source, Node):
                raise ValueError(f"{node.name}: cat 的表里混进了非张量实参")
            operand = _tensor_operand_of(source)
            # 部分和是各卡持有的一段累加，拼之前要先归约成复制，
            # 否则每段只是局部值，拼起来不等于整块。
            if operand[1].placement.kind == "Partial":
                reqs.append((operand[0], _dpu_req(REPLICATE)))
            else:
                sub_reqs, _ = rule(node, operand)
                reqs.extend(sub_reqs)
        # 各段沿拼接维的切分对齐时，每台只拼自己的列，输出保持切分，
        # 不再要求每段先聚成完整副本。
        aligned = _aligned_cat_spec(node, tensors, int(cat_dim))
        if aligned is not None:
            # 每段的切分已经是拼接要的样子，持有者也覆盖了输出的每一列，
            # 不需要再搬。每台直接拼自己持有的那些段。
            node.meta[SPEC_META_KEY] = aligned
            return
        stacked = _stacked_cat_spec(node, tensors, int(cat_dim))
        if stacked is not None:
            # 各段沿另一维切分，沿这一维拼。每台持有的列数不变，
            # 不再把每段聚成整块后互相覆盖。
            node.meta[SPEC_META_KEY] = stacked
            return
        out_placement = REPLICATE
    else:
        reqs, out_placement = rule(node, _tensor_operand(node, 0, "self"))
        # 类型转换不改变谁持有这块数据。输入是一段只落在部分台上的切片时，
        # 输出沿用这些台和它们的区间，而不是按输出形状重新平分到全部台。
        if node.target in _LAYOUT_NEUTRAL and out_placement.kind == "Shard":
            src_node, src_spec = _tensor_operand(node, 0, "self")
            # 持有者沿用输入的。按阶段重新分配会把只在一台的切片
            # 平分到全部台，收集校验按全局形状对不上。
            if src_spec.placement == out_placement:
                shape = tuple(val.shape)
                holders = {}
                for dpu_id, det in src_spec.shard_map.items():
                    local = list(shape)
                    length = det.end_idx - det.start_idx
                    local[out_placement.dim] = length
                    local = tuple(local)
                    # 源是切片时输出是一段新张量，区间从 0 计。源本身已按全局
                    # 下标切分时，输出沿用这个下标，否则拼接会对不齐。
                    lo = 0 if src_node.target is torch.ops.aten.slice.Tensor else det.start_idx
                    holders[dpu_id] = TensorShardDetail(
                        dpu_id, det.shard_dim, lo, lo + length, local,
                        elem_strides=row_major_strides(local))
                node.meta[SPEC_META_KEY] = PIMTensorSpec(
                    DEVICE_DPU, out_placement, "transient", None, holders,
                    out_placement.reduce_type, dtype=_dtype_of(node))
                node.meta[SPEC_META_KEY].validate()
                for src, req in reqs:
                    _append_edge(node, src, req, edges, strategy, num_layers)
                return
    for src, req in reqs:
        _append_edge(node, src, req, edges, strategy, num_layers)
    followed = _linear_output_following_weight(node, out_placement, tuple(val.shape))
    node.meta[SPEC_META_KEY] = followed or _dpu_spec(
        out_placement, tuple(val.shape), _dpus_of_node(node, strategy, num_layers),
        dtype=_dtype_of(node),
    )


_MISSING = object()


def _operand_value(node: Node, position: int, keyword: str) -> object:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(keyword, _MISSING)


def _reject_nested_tensor_container(node: Node, operand: object, keyword: str) -> None:
    if isinstance(operand, (tuple, list, dict)):
        raise NotImplementedError(f"{node.name} 的 {keyword} 不支持嵌套 tensor 容器")


def _tensor_operand_of(node: Node) -> tuple[Node, PIMTensorSpec]:
    """已确定是张量的实参，包成 `_tensor_operand` 的返回形态。"""
    spec = node.meta.get(SPEC_META_KEY)
    if spec is None:
        raise ValueError(f"{node.name} 还没有布局规格，无法作为卷积/拼接的输入")
    return node, spec


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
    strategy: ShardStrategy, num_layers: int,
    dst_dpus: tuple[int, ...] | None = None,
    dst_spec: PIMTensorSpec | None = None,
) -> None:
    """根据消费节点的 DPU 集合，为一条输入边追加重分布标记。

    类型转换的持有者继承自输入，可能只是阶段里的一部分台。按阶段的
    全部台重建目标规格会多出没有数据的台，收集校验按全局形状对不上。
    """
    # 只收窄分块插入的 _to_copy。图里原有的 to.dtype 是跨阶段的类型
    # 转换，它的下游在另一个阶段，按输入的持有者收窄会让下游读不到数据。
    if dst_dpus is not None:
        pass
    elif (dst.target is torch.ops.aten._to_copy.default
            and SPEC_META_KEY in dst.meta):
        dst_dpus = tuple(sorted(dst.meta[SPEC_META_KEY].shard_map))
    else:
        dst_dpus = _dpus_of_node(dst, strategy, num_layers)
    edge = _diff_edge(len(edges), src, dst, src.meta[SPEC_META_KEY], req, dst_dpus,
                      dst_spec)
    if edge is not None:
        # 同一份源张量搬到同一组目标 DPU 只发一条边。残差 add 的输出同时
        # 喂下一层 layernorm 的 to.dtype 与下一次残差 add 时，两个消费者
        # 都在设备上就会发两条完全相同的跨段边（实测 tp2_pp2 1→2）。
        key = (edge.src, tuple(sorted(edge.dst_loc.get("dpus", ()))),
               edge.type, edge.to_placement)
        for existing in edges:
            exist_key = (existing.src,
                         tuple(sorted(existing.dst_loc.get("dpus", ()))),
                         existing.type, existing.to_placement)
            if exist_key == key:
                dst.meta[REDISTRIBUTE_META_KEY].append(existing)
                return
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
