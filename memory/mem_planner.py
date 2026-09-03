"""规划每台 DPU 的权重、KV 和激活内存区域。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

import numpy as np
from torch.fx import Node

from contracts.graph_meta import DEVICE_DPU, REDISTRIBUTE_META_KEY, SPEC_META_KEY
from contracts.pim_tensor_spec import RedistributeEdge
from memory.kv_layout import KVRegionSpec, align_up, build_kv_layout


@dataclass(frozen=True)
class HwBudget:
    """定义单台 DPU 的 MRAM 容量、对齐和系统预留。"""

    mram_bytes: int          # 单 DPU MRAM 总量（≤8GB）
    align: int                # DMA 对齐边界
    sys_reserve_bytes: int    # 系统预留：kernel 二进制 + runtime 栈 + workspace 等非三区占用


@dataclass
class DPUPlan:
    """保存单台 DPU 的静态内存偏移和读者依赖。"""

    weight: dict[str, int]              # 权重名 -> offset，两图共用
    kv_base: int                         # KV 区基址，区内 offset 见对应 KVRegionSpec.kv_off
    act_base: int                        # 激活区基址，两图共享
    act_prefill: dict[str, int]         # prefill 图节点名 -> 激活 offset
    act_decode: dict[str, int]          # decode 图节点名 -> 激活 offset
    total: int                           # 持久区 + max(两图激活峰值)
    pending_readers_prefill: dict[tuple, list[str]]
    pending_readers_decode: dict[tuple, list[str]]


@dataclass
class TransientTensor:
    """保存临时张量的大小、生命周期和读者。"""

    name: str
    size: int
    produced_at: int
    last_read_at: int
    readers: list[str] = field(default_factory=list)


def bytes_of(local_shape: tuple[int, ...], itemsize: int) -> int:
    """返回本地分片的字节数。"""
    return prod(local_shape) * itemsize


def weights_of(dpu_nodes: list[Node], dpu_id: int) -> dict[str, Node]:
    """返回归属指定 DPU 的权重节点，以权重名称为键。"""
    out: dict[str, Node] = {}
    for node in dpu_nodes:
        if node.op != "get_attr":
            continue
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or spec.device != DEVICE_DPU or dpu_id not in spec.shard_map:
            continue
        out.setdefault(node.target, node)
    return out


def _pack_weights(
    dpu_id: int, prefill_nodes: list[Node], decode_nodes: list[Node], align: int
) -> tuple[dict[str, int], int]:
    """打包两张图共享的权重，并回填每个权重的 MRAM 偏移。"""
    prefill_weights = weights_of(prefill_nodes, dpu_id)
    decode_weights = weights_of(decode_nodes, dpu_id)
    names = sorted(set(prefill_weights) | set(decode_weights))
    offsets: dict[str, int] = {}
    off = 0
    for name in names:
        nodes = [n for n in (prefill_weights.get(name), decode_weights.get(name)) if n is not None]
        details = [n.meta[SPEC_META_KEY].shard_map[dpu_id] for n in nodes]
        first = details[0]
        for detail in details[1:]:
            if detail.local_shape != first.local_shape or detail.shard_dim != first.shard_dim:
                raise ValueError(
                    f"权重 {name} 在 prefill/decode 两图的切分不一致: {detail} vs {first}"
                )
        offsets[name] = off
        for node in nodes:
            spec = node.meta[SPEC_META_KEY]
            spec.shard_map[dpu_id] = replace(spec.shard_map[dpu_id], mram_offset=off)
        itemsize = nodes[0].meta["val"].element_size()
        off += align_up(bytes_of(first.local_shape, itemsize), align)
    return offsets, off


def _build_step_index(dpu_nodes: list[Node]) -> tuple[dict[Node, int], dict[int, int]]:
    """为节点计算和输入重分布命令分配执行顺序编号。"""
    node_step: dict[Node, int] = {}
    edge_step: dict[int, int] = {}
    step = 0
    for node in dpu_nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.edge_id not in edge_step:
                edge_step[edge.edge_id] = step
                step += 1
        node_step[node] = step
        step += 1
    return node_step, edge_step


def _redistribute_edges_landing_on(dpu_nodes: list[Node], dpu_id: int) -> dict[int, RedistributeEdge]:
    """返回需要在指定 DPU 分配落地缓冲的重分布边。"""
    out: dict[int, RedistributeEdge] = {}
    for node in dpu_nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.type == "local_slice":
                continue
            if edge.dst_loc.get("device") == DEVICE_DPU and dpu_id in edge.dst_loc.get("dpus", []):
                out[edge.edge_id] = edge
    return out


def redistribute_landing_tensors(
    dpu_nodes: list[Node], dpu_id: int, node_step: dict[Node, int], edge_step: dict[int, int]
) -> list[TransientTensor]:
    """返回落在指定 DPU 的重分布缓冲及其生命周期。"""
    tensors: list[TransientTensor] = []
    for node in dpu_nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.type == "local_slice":
                continue  # local_slice 不产生 DMA，也不需要落地缓冲。
            if edge.dst_loc.get("device") != DEVICE_DPU or dpu_id not in edge.dst_loc.get("dpus", []):
                continue
            detail = edge.dst_spec.shard_map[dpu_id]
            itemsize = np.dtype(edge.dtype).itemsize
            tensors.append(TransientTensor(
                name=f"redist_dst:e{edge.edge_id}",
                size=bytes_of(detail.local_shape, itemsize),
                produced_at=edge_step[edge.edge_id],
                last_read_at=node_step[node],
                readers=[node.name],
            ))
    return tensors


def transient_tensors(dpu_nodes: list[Node], dpu_id: int) -> list[TransientTensor]:
    """返回指定 DPU 的临时激活和重分布缓冲生命周期。"""
    node_step, edge_step = _build_step_index(dpu_nodes)
    redist_by_src: dict[str, list[tuple[Node, RedistributeEdge]]] = {}
    for other in dpu_nodes:
        for edge in other.meta.get(REDISTRIBUTE_META_KEY, []):
            redist_by_src.setdefault(edge.src, []).append((other, edge))

    tensors: list[TransientTensor] = list(
        redistribute_landing_tensors(dpu_nodes, dpu_id, node_step, edge_step)
    )
    for node in dpu_nodes:
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or spec.device != DEVICE_DPU or spec.residency != "transient":
            continue
        if dpu_id not in spec.shard_map:
            continue
        i = node_step[node]
        detail = spec.shard_map[dpu_id]
        itemsize = node.meta["val"].element_size()
        readers: list[str] = []
        last_read_at = i

        redistributed_users = {other for other, _ in redist_by_src.get(node.name, [])}
        for user in node.users:
            if user in redistributed_users:
                continue  # 重分布边通过 DMA 读取，不计入本地读取。
            readers.append(user.name)
            last_read_at = max(last_read_at, node_step[user])

        for other, edge in redist_by_src.get(node.name, []):
            if edge.src_loc.get("device") != DEVICE_DPU or dpu_id not in edge.src_loc.get("dpus", []):
                continue
            readers.append(f"redist:e{edge.edge_id}")
            # 重分布读取发生在该边的 DMA 步骤。
            last_read_at = max(last_read_at, edge_step[edge.edge_id])

        tensors.append(TransientTensor(
            name=node.name,
            size=bytes_of(detail.local_shape, itemsize),
            produced_at=i,
            last_read_at=last_read_at,
            readers=readers,
        ))
    return tensors


def greedy_reuse(tensors: list[TransientTensor], base: int, align: int) -> tuple[dict[str, int], int]:
    """按生命周期和读者关系为临时张量复用激活区地址。

    两个生命周期判据都用严格不等号，故意不取等：取等意味着某个节点在同一个
    step 里既读旧张量又写新张量，两者拿到同一个基址。已编译内核按裸指针逐块
    读写，写输出的前几行会覆盖尚未读取的输入行，算出错误结果（NumPy 内核先
    整块读入再写回，不受影响，所以这个坑只在编译产物路径上出现）。
    `node_step` 给每个节点唯一编号，所以两个判据取等只可能是同节点的读写别名，
    严格化不会误伤真正无重叠的复用。代价是激活区增大百分之二三十，都是千字节
    量级，相对 MRAM 预算可以忽略。
    """
    slots: list[dict] = []  # [{"offset", "size", "timeline": [(produced_at, last_read_at, readers_set)]}]
    offsets: dict[str, int] = {}
    top = base
    for t in sorted(tensors, key=lambda t: (-t.size, t.produced_at, t.name)):
        t_readers = set(t.readers)
        placed = False
        for slot in slots:
            if t.size <= slot["size"] and all(
                (start > t.last_read_at or t.produced_at > end) and not (t_readers & readers)
                for start, end, readers in slot["timeline"]
            ):
                slot["timeline"].append((t.produced_at, t.last_read_at, t_readers))
                offsets[t.name] = slot["offset"]
                placed = True
                break
        if not placed:
            offset = align_up(top, align)
            slots.append({"offset": offset, "size": t.size, "timeline": [(t.produced_at, t.last_read_at, t_readers)]})
            offsets[t.name] = offset
            top = offset + t.size
    return offsets, align_up(top, align)


def pending_readers_of_reused_addresses(
    tensors: list[TransientTensor], offsets: dict[str, int], dpu_id: int
) -> dict[tuple, list[str]]:
    """返回复用地址写入前必须等待的读者。"""
    by_offset: dict[int, list[TransientTensor]] = {}
    by_name = {t.name: t for t in tensors}
    for name, off in offsets.items():
        by_offset.setdefault(off, []).append(by_name[name])

    pending: dict[tuple, list[str]] = {}
    for off, occupants in by_offset.items():
        if len(occupants) < 2:
            continue
        occupants.sort(key=lambda t: t.produced_at)
        readers: list[str] = []
        for occupant in occupants[:-1]:
            readers.extend(occupant.readers)
        pending[(("dpu", dpu_id), off)] = readers
    return pending


def plan_dpu(
    dpu_id: int,
    prefill_nodes: list[Node],
    decode_nodes: list[Node],
    kv_specs: dict[int, KVRegionSpec],
    hw: HwBudget,
) -> DPUPlan:
    """规划三区偏移，回填规格，并返回单台 DPU 的内存蓝图。"""
    weight, off = _pack_weights(dpu_id, prefill_nodes, decode_nodes, hw.align)

    kv_spec = kv_specs[dpu_id]
    kv_spec.kv_base = off
    build_kv_layout(kv_spec, hw.align)
    kv_base = off
    off += kv_spec.kv_allocated_bytes

    act_base = off
    prefill_tensors = transient_tensors(prefill_nodes, dpu_id)
    decode_tensors = transient_tensors(decode_nodes, dpu_id)
    act_prefill, end_prefill = greedy_reuse(prefill_tensors, act_base, hw.align)
    act_decode, end_decode = greedy_reuse(decode_tensors, act_base, hw.align)
    total = max(end_prefill, end_decode)

    # 写回激活节点的 MRAM 偏移。
    for nodes, offsets in ((prefill_nodes, act_prefill), (decode_nodes, act_decode)):
        by_name = {n.name: n for n in nodes}
        for name, node_off in offsets.items():
            if name.startswith("redist_dst:"):
                continue  # 落地缓冲在后续循环写回。
            node = by_name[name]
            spec = node.meta[SPEC_META_KEY]
            spec.shard_map[dpu_id] = replace(spec.shard_map[dpu_id], mram_offset=node_off)

    # 写回重分布落地缓冲的 MRAM 偏移。
    for edges, offsets in ((_redistribute_edges_landing_on(prefill_nodes, dpu_id), act_prefill),
                           (_redistribute_edges_landing_on(decode_nodes, dpu_id), act_decode)):
        for edge_id, edge in edges.items():
            name = f"redist_dst:e{edge_id}"
            edge.dst_spec.shard_map[dpu_id] = replace(edge.dst_spec.shard_map[dpu_id], mram_offset=offsets[name])

    budget = hw.mram_bytes - hw.sys_reserve_bytes
    if total > budget:
        raise ValueError(f"dpu{dpu_id} 内存超限: total={total} > 可用预算={budget}")

    return DPUPlan(
        weight=weight,
        kv_base=kv_base,
        act_base=act_base,
        act_prefill=act_prefill,
        act_decode=act_decode,
        total=total,
        pending_readers_prefill=pending_readers_of_reused_addresses(prefill_tensors, act_prefill, dpu_id),
        pending_readers_decode=pending_readers_of_reused_addresses(decode_tensors, act_decode, dpu_id),
    )


def format_mem_plan(plans: dict[int, DPUPlan], hw: HwBudget) -> str:
    """把各 DPU 的内存蓝图格式化为可读文本（对齐 format_kv_layout/format_comm_plan 惯例）。"""
    budget = hw.mram_bytes - hw.sys_reserve_bytes
    lines = [
        f"== 内存规划（共 {len(plans)} 台 DPU，预算 {hw.mram_bytes / 2**20:.1f}MiB "
        f"- 预留 {hw.sys_reserve_bytes / 2**20:.1f}MiB = {budget / 2**20:.1f}MiB）=="
    ]
    for dpu_id, plan in sorted(plans.items()):
        weight_bytes = plan.kv_base
        kv_bytes_ = plan.act_base - plan.kv_base
        act_bytes = plan.total - plan.act_base
        margin = budget - plan.total
        lines.append(
            f"dpu{dpu_id}: weight={weight_bytes / 2**20:.2f}MiB kv={kv_bytes_ / 2**20:.2f}MiB "
            f"act={act_bytes / 2**20:.2f}MiB total={plan.total / 2**20:.2f}MiB "
            f"margin={margin / 2**20:.2f}MiB"
        )
    return "\n".join(lines)
