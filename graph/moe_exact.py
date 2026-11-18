"""exact_dispatch 的图拓扑分析：专家链、待退役尾链、acc 出口。

纯图分析（不碰计划与内存），编译器与内存规划共用：exec_plan_gen 按它
抑制 dense 尾链并发射行级命令，mem_planner 按它定路由缓冲的生命周期。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.fx import Node

from contracts.graph_meta import (
    DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, SPEC_META_KEY,
)


@dataclass(frozen=True)
class MoELayerTopology:
    """一层 MoE 在某张导出图里的 exact 发射拓扑。"""

    layer: int
    xs: str                       # MoE 公共输入节点名（专家链要改读紧凑缓冲）
    mask_scatter: str             # 布尔选中掩码 scatter（命中来源）
    routing_scatter: str          # 浮点路由权重 scatter（dispatch v2 读它）
    down_outs: dict[int, str]     # 专家号 → down_proj 输出节点名
    chains: set[str] = field(default_factory=set)   # 专家链成员（含 down_proj）
    xs_readers: dict[int, tuple[str, ...]] = field(default_factory=dict)
    # 每专家链上直接读 xs 的节点（exact 改读该专家的紧凑缓冲）
    tail: set[str] = field(default_factory=set)     # 待抑制的尾链（含融合累加）
    exits: set[str] = field(default_factory=set)    # 出口投影节点（读 acc 落地）
    exit_consumer: str = ""       # 第一个带外部实参的消费者（acc 生命期终点）


def _device_node(node: Node) -> bool:
    """节点是否落在设备侧；主机脚手架（getitem 等）没有 device meta。"""
    return node.meta.get(DEVICE_META_KEY) == DEVICE_DPU


# 出口投影允许下行的值透传节点：它们只搬值不改值，发射后改读 acc 落地即可。
# 算术节点（term mul、求和 add）不在其列——那是 MoE 表达式本体。
_FORWARDER_PREFIXES = ("aten.view", "aten.reshape", "aten.alias", "aten.detach",
                       "aten.slice", "aten.to", "aten._to_copy")


def _node_args(node: Node) -> list[Node]:
    return [arg for arg in node.args if isinstance(arg, Node)]


def _gate_linear_of(nodes: list[Node], layer: int) -> Node | None:
    """本层的 router linear：实参里挂着 `mlp.gate` 权重的那个。"""
    for node in nodes:
        if str(node.target) != "aten.linear.default":
            continue
        for arg in _node_args(node):
            if arg.op == "get_attr" and ".mlp.gate" in str(arg.target):
                part = str(arg.target).split(".")
                if int(part[part.index("layers") + 1]) == layer:
                    return node
    return None


def _scatter_layers(nodes: list[Node]) -> dict[str, tuple[int, str]]:
    """MoE scatter 节点名 → (层号, 类别)：mask 是布尔命中来源，routing 是权重。"""
    out: dict[str, tuple[int, str]] = {}
    for node in nodes:
        if node.target not in (torch.ops.aten.scatter.src,
                               torch.ops.aten.scatter.value):
            continue
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or node.meta.get(DEVICE_META_KEY) != DEVICE_HOST:
            continue
        layer = _layer_of_weight_walk(node)
        if layer is None:
            continue
        kind = "mask" if spec.dtype == "bool" else "routing"
        out[node.name] = (layer, kind)
    return out


def _layer_of_weight_walk(node: Node) -> int | None:
    """沿上游找第一个 `mlp.gate` 权重，返回层号（与 exec_plan_gen 同口径）。"""
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "get_attr" and ".mlp.gate" in str(cur.target):
            part = str(cur.target).split(".")
            return int(part[part.index("layers") + 1])
        stack.extend(cur.all_input_nodes)
    return None


def _chain_of(down_out: Node, xs: Node) -> set[str]:
    """专家链成员：down_proj 输出的实参反向闭包，停在 xs/get_attr/主机。"""
    chain: set[str] = set()
    stack = [down_out]
    while stack:
        cur = stack.pop()
        if cur.name in chain:
            continue
        chain.add(cur.name)
        for arg in _node_args(cur):
            if arg is xs or arg.op == "get_attr" or not _device_node(arg):
                continue
            stack.append(arg)
    return chain


def moe_exact_topology(nodes: list[Node]) -> dict[int, MoELayerTopology]:
    """分析导出图里每层 MoE 的 exact 发射拓扑；非 MoE 图返回空表。

    尾链 = 从专家输出与两个 scatter 出发、实参全在闭包内的前向闭包，减去
    专家链；遇到第一个带外部实参的消费者（残差加）为止。出口投影 = 该消费
    者的反向闭包里落在尾链边界上的节点（如 acc 的 view），它们要改读 acc 落地。
    """
    nodes = list(nodes)
    scatters = _scatter_layers(nodes)
    by_name = {n.name: n for n in nodes}
    layers = sorted({layer for layer, _ in scatters.values()})
    topo: dict[int, MoELayerTopology] = {}
    for layer in layers:
        layer_mask = {name for name, (l, kind) in scatters.items()
                      if l == layer and kind == "mask"}
        layer_routing = {name for name, (l, kind) in scatters.items()
                         if l == layer and kind == "routing"}
        if len(layer_mask) != 1 or len(layer_routing) != 1:
            raise ValueError(
                f"层 {layer} 应各有一个布尔掩码与一个路由权重 scatter，"
                f"实际 mask={sorted(layer_mask)} routing={sorted(layer_routing)}")
        gate = _gate_linear_of(nodes, layer)
        if gate is None:
            raise ValueError(f"层 {layer} 找不到 router linear（mlp.gate 权重）")
        xs = next(arg for arg in _node_args(gate) if arg.op != "get_attr")
        down_outs: dict[int, Node] = {}
        for node in nodes:
            if str(node.target) != "aten.linear.default":
                continue
            for arg in _node_args(node):
                if arg.op != "get_attr" or ".mlp.experts." not in str(arg.target):
                    continue
                part = str(arg.target).split(".")
                if int(part[part.index("layers") + 1]) != layer:
                    continue
                if "down_proj" not in str(arg.target):
                    continue
                expert = int(part[part.index("experts") + 1])
                if expert in down_outs:
                    raise ValueError(
                        f"层 {layer} 专家 {expert} 出现两个 down_proj linear")
                down_outs[expert] = node
        if not down_outs:
            raise ValueError(f"层 {layer} 找不到任何专家 down_proj linear")

        chains: set[str] = set()
        xs_readers: dict[int, tuple[str, ...]] = {}
        xs_args = {xs.name}
        for expert, node in down_outs.items():
            chain = _chain_of(node, xs)
            chains |= chain
            xs_readers[expert] = tuple(sorted(
                name for name in chain
                if any(arg.name in xs_args
                       for arg in _node_args(by_name[name]))))

        # 前向闭包：种子 = 专家输出 + 两个 scatter；实参全在闭包内才收录，
        # 带外部实参的节点就是 MoE 出口（残差加），闭包在它之前止步。
        seeds = {n.name for n in down_outs.values()} | layer_mask | layer_routing
        closure: set[str] = set(seeds)
        exit_consumer = None
        for node in nodes:
            if node.name in closure:
                continue
            args = _node_args(node)
            foreign = [arg for arg in args
                       if arg.op != "get_attr" and arg.name not in closure]
            if foreign and any(arg.name in closure for arg in args):
                exit_consumer = node.name
                break
            if foreign or not args:
                continue
            closure.add(node.name)

        if exit_consumer is None:
            raise ValueError(
                f"层 {layer} 的 MoE 输出没有汇入外部消费者（残差加），"
                "无法确定 acc 生命期终点")

        # 出口投影：出口消费者的反向闭包，只沿值透传节点下行（view/reshape
        # 一类，发射后改读 acc 落地即可）；碰到算术成员（term mul、求和 add）
        # 就停步——那是 MoE 表达式本体，exact 里整体被行级 scatter+归约替代，
        # 留在尾链退役。不能一路下行到主机成员为止：E=2 的两项求和是普通
        # 图内 add 而非融合主机闭包，一路下行会把 term mul 误划进出口，
        # 使 acc 落地被再乘一遍路由权重。
        exits: set[str] = set()
        stack = [by_name[exit_consumer]]
        while stack:
            cur = stack.pop()
            if cur.name in exits:
                continue
            if cur.name != exit_consumer:
                if cur.name not in closure or not _device_node(cur):
                    continue
                if not str(cur.target).startswith(_FORWARDER_PREFIXES):
                    continue
            exits.add(cur.name)
            stack.extend(arg for arg in _node_args(cur) if arg.name in closure)

        tail = closure - chains - seeds - exits
        topo[layer] = MoELayerTopology(
            layer=layer,
            xs=xs.name,
            mask_scatter=next(iter(layer_mask)),
            routing_scatter=next(iter(layer_routing)),
            down_outs={e: n.name for e, n in sorted(down_outs.items())},
            chains=chains,
            xs_readers=xs_readers,
            tail=tail,
            exits=exits,
            exit_consumer=exit_consumer,
        )
    return topo
