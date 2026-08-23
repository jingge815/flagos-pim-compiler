"""问题 2：设备映射与切分传播（方案 docs/spec.md:275-603，验收基准为附录 A）。

数据流：问题 1 的标注全图（每节点带 device/part_id）
  → ① 初始切分：ShardConfig 给每份权重钉起始布局（pinned，不再被传播改写）
  → ② 切分传播：拓扑序逐算子查规则表推出每个中间张量的布局
  → ③ 上下游布局不一致的边上打 redistribute 标注
  → ④ 收口为 node.meta["spec"]（PIMTensorSpec）与消费方 node.meta["redistribute"]

全程只算元数据，不执行算子、不搬数据（附录 A.4）。host 作为一种 location
一并参与：host 节点要求其张量输入为 Replicate@host，输出恒为 Replicate@host
（方案二.(10)），跨 host↔dpu 位置的搬运借此自动落成 redistribute 边。

第 1 阶段切分契约（方案二.(11)）：单维切、单段连续、全体参与、Partial 仅
sum/mean、num_dpus 为 2 的幂且整除 head 数；契约外情形（Shard(i)→Shard(j)
all_to_all、Partial 上的一元逐元素等）命中即抛错，不做静默兜底。
"""

from __future__ import annotations

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

REPLICATE = Placement("Replicate")


# ---------------------------------------------------------------------------
# ① 初始切分配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardConfig:
    """第 1 阶段手工固定切分配置（方案二.(2)：不做自动求优，人工钉死）。

    weight_rules 按 get_attr 节点名（如 ``model.layers.0.self_attn.q_proj.weight``）
    做子串匹配，首个命中生效；"col" = 列切（切输出维，HF 权重 [out, in] 的
    Shard(0)），"row" = 行切（切输入维，Shard(1)）。
    """

    num_dpus: int
    weight_rules: tuple[tuple[str, Literal["col", "row"]], ...]
    dpu_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_dpus <= 0:
            raise ValueError("num_dpus must be positive")
        dpu_ids = tuple(range(self.num_dpus)) if self.dpu_ids is None else tuple(self.dpu_ids)
        if len(dpu_ids) != self.num_dpus or len(set(dpu_ids)) != len(dpu_ids) or any(dpu_id < 0 for dpu_id in dpu_ids):
            raise ValueError("dpu_ids must contain unique non-negative IDs for every DPU")
        object.__setattr__(self, "dpu_ids", dpu_ids)

    def match(self, weight_name: str) -> Literal["col", "row"] | None:
        for pattern, mode in self.weight_rules:
            if pattern in weight_name:
                return mode
        return None


def llama_shard_config(
    num_dpus: int,
    *,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    dpu_ids: tuple[int, ...] | None = None,
) -> ShardConfig:
    """Llama 系（GQA 通用、MHA 为特例）默认切分：列切/行切交替的 Megatron 配对。

    q/k/v/gate/up/lm_head 列切，o/down 行切（方案二.(2) 初始切分表）。校验第 1
    阶段切分契约：num_dpus 为 2 的幂、整除 Q/KV head 数（保证注意力权重切点
    对齐 head 边界，为问题 7 的 KV 本地驻留留好前提）、MLP 与 vocab 维均匀可分。

    入: num_dpus —— DPU 台数；其余为模型结构参数（LlamaConfig 同名字段）。
    出: ShardConfig。契约不满足时抛 ValueError。
    """
    if num_dpus <= 0:
        raise ValueError("num_dpus must be positive")
    if num_dpus & (num_dpus - 1):
        raise ValueError(f"num_dpus={num_dpus} 不是 2 的整数次幂（切分契约 5）")
    for name, length in (
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("intermediate_size", intermediate_size),
        ("vocab_size", vocab_size),
    ):
        if length % num_dpus:
            raise ValueError(f"{name}={length} 不能被 num_dpus={num_dpus} 整除（切分契约 5）")
    return ShardConfig(
        num_dpus=num_dpus,
        weight_rules=(
            ("q_proj.weight", "col"),
            ("k_proj.weight", "col"),
            ("v_proj.weight", "col"),
            ("o_proj.weight", "row"),
            ("gate_proj.weight", "col"),
            ("up_proj.weight", "col"),
            ("down_proj.weight", "row"),
            ("lm_head.weight", "col"),
        ),
        dpu_ids=dpu_ids,
    )


# ---------------------------------------------------------------------------
# PIMTensorSpec 构造
# ---------------------------------------------------------------------------


def _host_spec() -> PIMTensorSpec:
    """host 张量恒为 Replicate@host（方案二.(10)），无 DPU 分片，shard_map 为空。"""
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
    """按布局把全局 shape 展开成逐 DPU 的分片明细（契约 1/2：单维切、单段连续均匀切）。

    Shard(d)：每台持 [i*w, (i+1)*w) 一段，w = L/num_dpus，L 不整除即抛（契约 5）。
    Replicate/Partial：每台持全量，start/end 记摊平后的 [0, numel)，与附录 B
    通信段的 global_range 语义一致。
    """
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


def _weight_spec(node: Node, config: ShardConfig) -> PIMTensorSpec:
    """get_attr 权重的初始布局（方案二.(2)：只有权重需人工指定，即本函数的依据）。

    被 linear 消费的权重必须命中 shard_config（否则配置不全，抛错）；命中后按
    col/row 给 Shard(0)/Shard(1)，residency=pinned（权重常驻 MRAM，不参与传播
    改写）。仅被逐元素 DPU 算子消费的权重（如 RMSNorm 权重）给 Replicate@dpu；
    其余（embedding、纯 host 消费）给 Replicate@host。
    """
    shape = tuple(node.meta["val"].shape)
    dpu_users = [u for u in node.users if u.meta.get(DEVICE_META_KEY) == DEVICE_DPU]
    linear_users = [
        u for u in dpu_users if u.target in (torch.ops.aten.linear.default, torch.ops.aten.addmm.default)
    ]
    if linear_users:
        mode = config.match(node.target)
        if mode is None:
            raise ValueError(f"权重 {node.target} 被 linear 消费，但 shard_config 未指定其切法")
        return _dpu_spec(
            Placement("Shard", 0 if mode == "col" else 1), shape, config.dpu_ids, residency="pinned"
        )
    if dpu_users:
        return _dpu_spec(REPLICATE, shape, config.dpu_ids, residency="pinned")
    return _host_spec()


# ---------------------------------------------------------------------------
# ② 算子布局规则表（方案二.(8)）：输入布局 + 算子 → 要求输入布局 + 输出布局
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Req:
    """规则表对某个输入的要求：落在哪个 device、什么 placement（shard_map 由全局 shape 推）。"""

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
    """aten.linear(x, w)：Y = X @ W.T，W 形状 [N, K]（附录 A 的两层 Linear 即本规则）。

    列切 w Shard(0)（切输出维 N）：要求 X 为 Replicate，输出 Shard(最后一维)。
    行切 w Shard(1)（切 contraction 维 K）：要求 X 沿最后一维切（与 K 同维、均匀
    切分下各 DPU 区间天然对齐），输出 Partial(sum)（附录 A.2：contraction 维被
    切则输出为部分和）。带 bias 不在第 1 阶段契约内，命中即抛。
    """
    x_node, x_spec = x
    weight_node, w_spec = weight
    out_dim = x_node.meta["val"].ndim - 1  # linear 保秩，输出最后一维 = N
    if w_spec.placement == Placement("Shard", 0):
        return [(x_node, _dpu_req(REPLICATE)), (weight_node, _dpu_req(w_spec.placement))], Placement("Shard", out_dim)
    if w_spec.placement == Placement("Shard", 1):
        return [
            (x_node, _dpu_req(Placement("Shard", out_dim))),
            (weight_node, _dpu_req(w_spec.placement)),
        ], Placement("Partial", reduce_type="sum")
    raise _unsupported_layouts(node, x_spec, w_spec)


def _rule_addmm(node: Node) -> tuple[list[tuple[Node, _Req]], Placement]:
    """aten.addmm 必带 bias（融合加），第 1 阶段无对应切分契约；白名单保留它属
    GPT-2 遗留，命中即抛，提示收窄白名单或扩充契约。"""
    raise NotImplementedError("aten.addmm（带 bias 线性）不在第 1 阶段切分契约内")


def _rule_elementwise_binary(
    node: Node,
    x: tuple[Node, PIMTensorSpec],
    y: tuple[Node, PIMTensorSpec] | None,
) -> tuple[list[tuple[Node, _Req]], Placement]:
    """aten.add.Tensor / aten.mul.Tensor（方案二.(8) 逐元素行）：切分方式完全继承输入。

    两输入同为 Shard(d) → Shard(d)；Shard + Replicate → Shard（Replicate 端
    按目标布局 scatter / local_slice 对齐）；同为 Replicate → Replicate；
    Partial + Replicate → 先对 Partial 端插 all_reduce 还原，再按 Replicate 算
    （残差连接的固定路径）；同为 Partial 且规约类型一致 → Partial。
    标量右操作数 → 布局透传。契约外组合（Shard 维不一致、Partial+Shard、Partial
    上加/乘标量等）抛错。
    """
    x_node, x_spec = x
    if y is None:  # 标量操作数：一元透传；Partial 上加/乘标量数学上不成立
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
    """aten.tanh 等一元逐元素：布局透传；Partial 上的一元非线性变换数学上不成立，抛错。"""
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


# ---------------------------------------------------------------------------
# ③ redistribute 边：比较上游实际产出与本算子要求（方案二.(9)）
# ---------------------------------------------------------------------------


def _loc_of_spec(spec: PIMTensorSpec) -> dict:
    if spec.device == DEVICE_HOST:
        return {"device": DEVICE_HOST}
    return {"device": DEVICE_DPU, "dpus": sorted(spec.shard_map)}


def _required_spec(src: Node, req: _Req, config: ShardConfig) -> PIMTensorSpec:
    """Materialize a consumer requirement using the producer's global tensor shape."""
    if req.device == DEVICE_HOST:
        return _host_spec()
    return _dpu_spec(req.placement, tuple(src.meta["val"].shape), config.dpu_ids)


def _edge_type(node: Node, actual: PIMTensorSpec, req: _Req) -> str:
    """由 (实际布局, 要求布局, 位置) 定重分布类型（方案问题 3 的四种类型 + 退化规则）。

    Partial→Replicate=all_reduce；Shard→Replicate=all_gather；Replicate→Shard
    跨位置=scatter、同位置=local_slice（本地切片，零通信，方案二.(8) 逐元素行）；
    同 placement 纯跨位置：host→dpu 记 scatter（目标全量即 broadcast 退化）、
    dpu→host 记 all_gather（源为全量副本，问题 3 只收一份）。不同切分维的
    DPU Shard(i)→Shard(j) 记 all_to_all；其余契约外组合（如 Partial→Shard）抛错。
    """
    a, r = actual.placement, req.placement
    if not (
        a == r
        or (a.kind == "Partial" and r.kind == "Replicate")
        or (a.kind == "Shard" and r.kind == "Replicate")
        or (a.kind == "Replicate" and r.kind == "Shard")
        or (a.kind == r.kind == "Shard" and a.dim != r.dim and actual.device == req.device == DEVICE_DPU)
    ):
        raise ValueError(f"{node.name} 不支持的布局转换: {a}@{actual.device} → {r}@{req.device}")
    if a == r:  # 布局相同、仅跨 host↔dpu 位置
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
    edge_id: int, src: Node, dst: Node, actual: PIMTensorSpec, req: _Req, config: ShardConfig
) -> RedistributeEdge | None:
    """比较一条边的上游实际布局与下游要求布局；一致返回 None，否则生成标注。

    入: edge_id —— 全图递增编号；src/dst —— 边两端节点；actual —— 上游的
    PIMTensorSpec；req —— 下游规则表给出的要求；config —— 全体参与契约。
    出: RedistributeEdge 或 None。nbytes 取逻辑张量总字节数（from val）。
    """
    if actual.device == req.device and actual.placement == req.placement:
        return None
    if actual.residency == "pinned":
        # 方案二.(7)：pinned 张量禁止搬运、不允许插 redistribute；第 1 阶段 pinned
        # 只用于权重，其布局由 shard_config 给定且消费方规则必然匹配——不一致即 bug。
        raise ValueError(f"pinned 张量 {src.name} 不允许重分布（{actual.placement} → {req.placement}）")
    val = src.meta["val"]
    dst_spec = _required_spec(src, req, config)
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


# ---------------------------------------------------------------------------
# 传播驱动
# ---------------------------------------------------------------------------


def propagate_specs(gm: GraphModule, config: ShardConfig) -> list[RedistributeEdge]:
    """问题 2 主 pass：初始切分 → 拓扑序传播 → redistribute 标注（方案二.(5)/(7)）。

    入: gm —— 问题 1 已标注 device/part_id 的 GraphModule（每节点 meta["val"]
        携带 shape/dtype）；config —— 初始切分配置。
    出: 全图 redistribute 边列表（按推导顺序编号）；同时写回
        node.meta["spec"]（每个张量节点）与 node.meta["redistribute"]
        （消费方节点的入边列表，无入边分歧时为空列表）。
    """
    nodes = list(gm.graph.nodes)
    for node in nodes:
        if DEVICE_META_KEY not in node.meta:
            raise ValueError(f"图节点 {node.name} 缺少 device 标注，请先运行 graph.partition.partition_graph")
        node.meta.pop(SPEC_META_KEY, None)
        node.meta[REDISTRIBUTE_META_KEY] = []

    # 1. 初始切分：placeholder 恒为 Replicate@host；get_attr 权重按 config 钉死
    for node in nodes:
        if node.op == "placeholder":
            node.meta[SPEC_META_KEY] = _host_spec()
        elif node.op == "get_attr":
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor):  # 非张量 get_attr（子图模块等）不是数据边，跳过
                node.meta[SPEC_META_KEY] = _weight_spec(node, config)

    # 2. 拓扑序传播（export 图本身即拓扑序，上游先算）；已钉死的权重跳过
    edges: list[RedistributeEdge] = []
    for node in nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op == "output":  # 图出口视为 host 消费方
            _require_host_inputs(node, edges, config)
        elif node.meta[DEVICE_META_KEY] == DEVICE_HOST:
            # val 为 None 的是 _assert_tensor_metadata 之类的死端 guard：无张量输出、
            # 无消费者，不参与数据流，不为它的入边打 redistribute。
            if node.meta.get("val") is not None:
                _require_host_inputs(node, edges, config)
            node.meta[SPEC_META_KEY] = _host_spec()
        else:
            _propagate_dpu_node(node, edges, config)
    return edges


def _require_host_inputs(node: Node, edges: list[RedistributeEdge], config: ShardConfig) -> None:
    """host 节点（含 output）要求其全部张量输入为 Replicate@host（方案二.(10)）。

    上游是 Shard/Partial 或位于 dpu 时，在该入边上自动触发 gather 或 reduce。
    """
    req = _Req(DEVICE_HOST, REPLICATE)
    for input_node in node.all_input_nodes:
        if SPEC_META_KEY not in input_node.meta:  # 非张量 get_attr（子图模块等），不是数据边
            continue
        _append_edge(node, input_node, req, edges, config)


def _propagate_dpu_node(node: Node, edges: list[RedistributeEdge], config: ShardConfig) -> None:
    """DPU 节点：查规则表得要求输入布局与输出布局，逐入边比对打标注，写输出 spec。"""
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
        _append_edge(node, src, req, edges, config)
    node.meta[SPEC_META_KEY] = _dpu_spec(out_placement, tuple(val.shape), config.dpu_ids)


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
    dst: Node, src: Node, req: _Req, edges: list[RedistributeEdge], config: ShardConfig
) -> None:
    edge = _diff_edge(len(edges), src, dst, src.meta[SPEC_META_KEY], req, config)
    if edge is not None:
        edges.append(edge)
        dst.meta[REDISTRIBUTE_META_KEY].append(edge)


# ---------------------------------------------------------------------------
# ④ 可读报告（定位问题用）
# ---------------------------------------------------------------------------


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
    """把标注结果格式化为可读文本：逐节点布局 + 逐 redistribute 边 + 分类统计。

    入: gm —— propagate_specs 跑过的图；edges —— 其返回值；max_nodes ——
        只详列前 N 个节点（大模型用），None 为全列。
    出: 多行字符串。
    """
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
