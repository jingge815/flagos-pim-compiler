"""标记 FX 图中的 DPU 节点并划分连通子图。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, PART_ID_META_KEY


# 留在主机的算子。这是一张**黑名单**：默认下设备，列进来的才留主机。
#
# 反过来做（白名单）会让每个没想到的算子静默落到主机上——图照样跑完、数值也对，
# 只是那些算子根本没在设备上执行，而这正是本项目要消掉的东西。黑名单则相反：
# 漏掉一个新算子会在 `spec_prop` 的规则表那儿直接报错，吵闹但看得见。
#
# 每类都有它必须留在主机的理由。
#
# 只留编译脚手架与图入口的位置序列；所有真正动张量的算子都走 PIM。
# 注意力曾经靠 `runtime/executor.py::make_sdpa_handler` 挂在主机上，现在那段
# 计算搬进了 `runtime/kernels.sdpa_kernel`——是设备命令，读写的 KV 缓存本来
# 也就是 DPU 本地 MRAM。
HOST_ONLY = frozenset(
    {
        # 1. 编译脚手架：不动张量，或者根本没有张量输出。
        torch.ops.aten._assert_tensor_metadata.default,
        # 2. 图入口的位置序列：`arange` 只产位置下标。
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.default,
        # 3. 广播视图：`expand` / `repeat` 是 GQA 把 kv_heads 广播到 heads 用的
        #    （kv_heads < heads 时出现）。它们的输出元素数大于输入，而命令按
        #    `out_shape` 分配输出、按 `arg_shapes` 读输入，广播倍数没有地方放，
        #    下设备只会按输入形状读回一块小的。`contiguous` 是一次布局归一，
        #    在 numpy 镜像里恒等，发一个什么都不做的内核没有意义。
        torch.ops.aten.expand.default,
        torch.ops.aten.repeat.default,
        torch.ops.aten.contiguous.default,
        # `squeeze.dim` 目标模型里不出现（`unsqueeze` 出现，见下），
        # 没有可验证的形态就不配规则与内核。
        torch.ops.aten.squeeze.dim,
        # 5. 目标模型里不出现的算子。Llama2 用的是 RMSNorm（在 aten 里是
        #    `pow -> mean -> add -> rsqrt -> mul` 一串，都已下设备），不是
        #    LayerNorm。`layer_norm` 只在合成测试里作主机脚手架出现。
        #    归一化本身确实是设备侧的活（GML 里 `RMSNorm_vpu` 绑在向量单元上），
        #    但给一个目标模型里根本不出现的算子编内核，编出来也无从验证。
        #    等真有模型要它，再补规则与内核。
        torch.ops.aten.layer_norm.default,
    }
)


# 按**算子名**留主机的族，不分重载。dtype 转换有 `to.dtype` /
# `to.dtype_layout` / `to.device` 等一串重载，断言有 `_assert_tensor_metadata`
# 等若干个——逐个重载列进去，等于每遇到一个新重载就被规则表拦一次，
# 而它们的归属理由完全相同。
HOST_ONLY_PACKETS = frozenset(
    {
        # `to.dtype` 已下设备（逐参 dtype 读入修好后），不再整族留主机。
        "_assert_tensor_metadata",
        "_assert_scalar",
        "sym_size",
        "sym_numel",
        "sym_constrain_range",
        "sym_constrain_range_for_size",
        "_local_scalar_dense",
        "lift_fresh_copy",
        "detach",
    }
)


def _is_host_only(target) -> bool:
    """这个目标是否必须留在主机。

    三种情况：显式列名的、整族列名的，以及**不是 `OpOverload`** 的——
    `operator.getitem`、`wrap_with_set_grad_enabled` 这类高阶算子与内置函数
    没有张量语义，也没有 `meta['val']` 可推布局。
    """
    if not isinstance(target, torch._ops.OpOverload):
        return True
    if target in HOST_ONLY:
        return True
    return target.name().split("::")[-1].split(".")[0] in HOST_ONLY_PACKETS


@dataclass
class Partition:
    """按拓扑顺序保存一个只含 DPU 节点的连通子图。"""

    part_id: int
    nodes: list[Node]


def _is_dpu_node(node: Node) -> bool:
    """除了显式留主机的，`call_function` 一律下设备。

    这里**不**顺手检查 `meta['val']` 在不在。少了它推不出布局，但那是
    `propagate_specs` 的事，它已经会为此抛错。在这儿悄悄把节点降成主机，
    等于用"看起来还能跑"换掉一条本该炸出来的错——黑名单要的就是相反的行为。
    """
    return node.op == "call_function" and not _is_host_only(node.target)


def partition_graph(gm: GraphModule) -> list[Partition]:
    """原地标记 `gm`，并返回 DPU 直连子图。"""

    nodes = list(gm.graph.nodes)
    node_order = {node: index for index, node in enumerate(nodes)}
    parent: dict[Node, Node] = {}

    for node in nodes:
        node.meta[DEVICE_META_KEY] = DEVICE_DPU if _is_dpu_node(node) else DEVICE_HOST
        node.meta.pop(PART_ID_META_KEY, None)
        if node.meta[DEVICE_META_KEY] == DEVICE_DPU:
            parent[node] = node

    def find(node: Node) -> Node:
        while parent[node] is not node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: Node, right: Node) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root is not right_root:
            parent[right_root] = left_root

    for node in parent:
        # 仅连接直接相邻的 DPU 节点。
        for input_node in node.all_input_nodes:
            if input_node in parent:
                union(node, input_node)

    components: dict[Node, list[Node]] = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)

    sorted_components = sorted(
        (sorted(component, key=node_order.__getitem__) for component in components.values()),
        key=lambda component: node_order[component[0]],
    )

    partitions: list[Partition] = []
    for part_id, component in enumerate(sorted_components):
        for node in component:
            node.meta[PART_ID_META_KEY] = part_id
        partitions.append(Partition(part_id=part_id, nodes=component))
    return partitions
