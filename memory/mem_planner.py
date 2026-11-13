"""规划每台 DPU 的权重、KV 和激活内存区域。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

from torch.fx import Node

from contracts.dtypes import dtype_bytes
from contracts.graph_meta import DEVICE_DPU, REDISTRIBUTE_META_KEY, SPEC_META_KEY
from contracts.pim_tensor_spec import RedistributeEdge
from memory.kv_layout import KVRegionSpec, align_up, build_kv_layout


@dataclass(frozen=True)
class HwBudget:
    """定义单台 DPU 的 MRAM 容量、对齐和系统预留。"""

    mram_bytes: int          # 单 DPU MRAM 总量（≤8GB）
    # DMA 对齐边界。必须与 `PIMHardwareConfig.dma_align` 取同一个值：这个数会被
    # 写进分片的 `align_bytes`，随命令下发给算子编译器，成为 `-pim-tile-to-budget`
    # 选分块时的**每块缓冲**对齐。给大了（如 1024）就凑不出整数倍的缓冲——
    # decode 的 M=1/K=32 分片一行只有 64 字节，分块搜索会直接失败。
    align: int
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
    # 单块最大常驻/瞬时分配的字节数。多 TensorPU 的簇可以整簇共享容量，
    # 但任何一块张量必须能放进单个 TensorPU。
    peak_allocation_bytes: int = 0
    # 逐常驻分配台账：权重分片、KV 与内核 tile。同一 TensorPU 上按
    # tensor_key 去重模型常驻，请求峰值再相加，与 GeneSim 的容量语义一致。
    resident_allocations: tuple["ResidentAllocation", ...] = ()


@dataclass(frozen=True)
class ResidentAllocation:
    """一笔常驻分配；owner 是初始放置单元，tensor_key 用于去重。"""

    owner: str
    tensor_key: str
    model_bytes: int = 0
    request_bytes_by_request: tuple[int, ...] = ()


@dataclass
class TransientTensor:
    """保存临时张量的大小、生命周期和读者。"""

    name: str
    size: int
    produced_at: int
    last_read_at: int
    readers: list[str] = field(default_factory=list)


def bytes_of(local_shape: tuple[int, ...], itemsize: int,
             elem_strides: tuple[int, ...] = ()) -> int:
    """本地分片占的字节数。

    `elem_strides` 为空时按行主序紧密算 —— 与本轮改动前**逐字节等价**，
    这是三个调用点（权重区 / 激活区 / KV 区）的现状。
    非空时按最外维步幅算，覆盖带填充的排布：此时字节数不等于
    `prod(shape) * itemsize`，因为每行末尾有填充。
    """
    if not elem_strides:
        return prod(local_shape) * itemsize       # 原路径，一字未改
    if len(elem_strides) != len(local_shape):
        raise ValueError(
            f"elem_strides 秩 {len(elem_strides)} 与 local_shape 秩 "
            f"{len(local_shape)} 不符")
    return local_shape[0] * elem_strides[0] * itemsize


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
            # 排布层：把这次规划实际用的对齐落到分片上，而不是只留在
            # 全局的 `HwBudget.align` 里（设计 §4.4.2 的回填写法）。
            spec.shard_map[dpu_id] = replace(
                spec.shard_map[dpu_id], mram_offset=off, align_bytes=align)
        itemsize = dtype_bytes(nodes[0].meta[SPEC_META_KEY].dtype)
        off += align_up(bytes_of(first.local_shape, itemsize, first.elem_strides), align)
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
            if edge.dst_loc.get("device") == DEVICE_DPU and dpu_id in edge.dst_loc.get("dpus", []):
                out[edge.edge_id] = edge
    return out


def redistribute_landing_tensors(
    dpu_nodes: list[Node], dpu_id: int, node_step: dict[Node, int], edge_step: dict[int, int]
) -> list[TransientTensor]:
    """返回落在指定 DPU 的重分布缓冲及其生命周期。

    同一份源张量搬到同一组 DPU 只发一条边（spec_prop 的 `_append_edge`
    去重），这条边挂在每个消费者上。落地缓冲只有一块，生命周期必须并到
    最后一个读者、读者集合取全部；按消费者各记一份会让缓冲被提前复用，
    后面的消费者读到覆盖后的数据。
    """
    merged: dict[int, TransientTensor] = {}
    for node in dpu_nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.dst_loc.get("device") != DEVICE_DPU or dpu_id not in edge.dst_loc.get("dpus", []):
                continue
            entry = merged.get(edge.edge_id)
            if entry is not None:
                entry.last_read_at = max(entry.last_read_at, node_step[node])
                entry.readers.append(node.name)
                continue
            detail = edge.dst_spec.shard_map[dpu_id]
            itemsize = dtype_bytes(edge.dtype)
            merged[edge.edge_id] = TransientTensor(
                name=f"redist_dst:e{edge.edge_id}",
                size=bytes_of(detail.local_shape, itemsize, detail.elem_strides),
                produced_at=edge_step[edge.edge_id],
                last_read_at=node_step[node],
                readers=[node.name],
            )
    return list(merged.values())


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
        itemsize = dtype_bytes(node.meta[SPEC_META_KEY].dtype)
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
            size=bytes_of(detail.local_shape, itemsize, detail.elem_strides),
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
    *,
    kernel_mram_bytes: int = 0,
    request_kv_tokens: tuple[int, ...] | None = None,
) -> DPUPlan:
    """规划三区偏移，回填规格，并返回单台 DPU 的内存蓝图。

    `kernel_mram_bytes` 是算子编译器回传的单台内核 tile 占用
    （`pim.placed-mram-bytes`，含 `partial` 档的归约暂存）。它与三区占的是同一
    块 MRAM，所以算进容量判据。默认 0 = 没有回传可用，判据退回只看三区，
    与改动前逐字节等价。`request_kv_tokens` 是本轮轨迹逐请求的上下文上界，
    缺省时 KV 台账按最大序列长度登记一个请求峰值。
    """
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
            spec.shard_map[dpu_id] = replace(spec.shard_map[dpu_id],
                                              mram_offset=node_off,
                                              align_bytes=hw.align)

    # 写回重分布落地缓冲的 MRAM 偏移。
    for edges, offsets in ((_redistribute_edges_landing_on(prefill_nodes, dpu_id), act_prefill),
                           (_redistribute_edges_landing_on(decode_nodes, dpu_id), act_decode)):
        for edge_id, edge in edges.items():
            name = f"redist_dst:e{edge_id}"
            edge.dst_spec.shard_map[dpu_id] = replace(
                edge.dst_spec.shard_map[dpu_id], mram_offset=offsets[name],
                align_bytes=hw.align)

    budget = hw.mram_bytes - hw.sys_reserve_bytes
    # `total` 只是**三区**（权重 + KV + 激活）的末端。内核运行时还要为
    # x / w / out 三块 tile 占一份 MRAM，而那份占用由算子编译器定分块后才知道，
    # 只能由它回传（`pim.placed-mram-bytes`，含 `partial` 档的归约暂存）。
    #
    # 两个口径以前从未相加：这里只校验三区、GeneSim 那侧只核对 tile 占用，
    # 于是「三区占满 + tile 再要一份」这种真实超限两边都看不见。相加才是单台
    # DPU 的真实峰值。取不到回传时按 0 计 —— 与改动前逐字节等价。
    kernel_bytes = kernel_mram_bytes or 0
    peak = total + kernel_bytes
    if peak > budget:
        detail = (f"（三区 {total} + 内核 tile {kernel_bytes}）"
                  if kernel_bytes else "")
        raise ValueError(
            f"dpu{dpu_id} 内存超限: peak={peak}{detail} > 可用预算={budget}")

    peak_allocation = _peak_single_allocation(
        dpu_id, prefill_nodes, decode_nodes, prefill_tensors, decode_tensors,
        kv_spec, hw, kernel_bytes)
    allocations = _resident_allocations(
        dpu_id, prefill_nodes, decode_nodes, kv_spec, kernel_bytes, hw.align,
        request_kv_tokens=request_kv_tokens)
    return DPUPlan(
        weight=weight,
        kv_base=kv_base,
        act_base=act_base,
        act_prefill=act_prefill,
        act_decode=act_decode,
        total=total,
        pending_readers_prefill=pending_readers_of_reused_addresses(prefill_tensors, act_prefill, dpu_id),
        pending_readers_decode=pending_readers_of_reused_addresses(decode_tensors, act_decode, dpu_id),
        peak_allocation_bytes=peak_allocation,
        resident_allocations=allocations,
    )


def _peak_single_allocation(
    dpu_id: int,
    prefill_nodes: list[Node],
    decode_nodes: list[Node],
    prefill_tensors: list[TransientTensor],
    decode_tensors: list[TransientTensor],
    kv_spec: KVRegionSpec,
    hw: HwBudget,
    kernel_bytes: int,
) -> int:
    """单块最大分配：任一权重分片、KV 张量、激活张量或内核 tile。

    整簇容量可以被多个 TensorPU 共享，但单块张量只能落在一个 TensorPU；
    用整簇预算去核验「单 PU 常驻」会让仿真前才暴露非法候选。
    """
    sizes = [max(0, int(kernel_bytes))]
    sizes.append(kv_spec.max_seq * kv_spec.head_dim * kv_spec.dtype_bytes)
    for nodes in (prefill_nodes, decode_nodes):
        for node in weights_of(nodes, dpu_id).values():
            spec = node.meta[SPEC_META_KEY]
            detail = spec.shard_map[dpu_id]
            size = bytes_of(
                detail.local_shape, dtype_bytes(spec.dtype), detail.elem_strides)
            sizes.append(align_up(size, hw.align))
    for tensor in (*prefill_tensors, *decode_tensors):
        sizes.append(align_up(tensor.size, hw.align))
    return max(sizes)


def _weight_owner_map(
    dpu_id: int,
    prefill_nodes: list[Node],
    decode_nodes: list[Node],
) -> dict[str, tuple[tuple[int, int], Node, str]]:
    """权重名 → (执行序, 消费算子名, 节点)。

    权重按**消费它的执行算子**归组，而不是按权重名排序：同一次执行单元
    的权重必须落在同一个 TensorPU，权重名的字典序与算子执行顺序无关。
    两张图共用的权重取较早出现的那个消费者。
    """
    positions: dict[Node, tuple[int, int]] = {}
    for rank, nodes in enumerate((prefill_nodes, decode_nodes)):
        for index, node in enumerate(nodes):
            positions.setdefault(node, (rank, index))
    owners: dict[str, tuple[tuple[int, int], Node, str]] = {}
    for rank, nodes in enumerate((prefill_nodes, decode_nodes)):
        for name, node in weights_of(nodes, dpu_id).items():
            consumers = [user for user in node.users if user in positions]
            if consumers:
                first = min(consumers, key=lambda user: positions[user])
                order = positions[first]
                owner = first.name
            else:
                order = (rank, len(nodes))
                owner = f"weight:{name}"
            current = owners.get(name)
            if current is None or order < current[0]:
                owners[name] = (order, node, owner)
    return owners


def _attention_unit_map(nodes: list[Node]) -> dict[int, str]:
    """层号 → 该层 attention 执行算子名；没有 attention 的图返回空。"""
    from runtime.compile import _q_proj_layer_of

    units: dict[int, str] = {}
    for node in nodes:
        if "scaled_dot_product_attention" not in str(
                getattr(node, "target", "")):
            continue
        try:
            layer = _q_proj_layer_of(node)
        except ValueError:
            continue
        units.setdefault(layer, node.name)
    return units


def _kv_request_bytes(
    kv_spec: KVRegionSpec,
    max_seq: int,
    request_kv_tokens: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """KV 张量的逐请求峰值：优先用本轮轨迹，缺省退回最大序列长度。"""
    per_token = kv_spec.head_dim * kv_spec.dtype_bytes
    if request_kv_tokens is None:
        return (max_seq * per_token,)
    if not request_kv_tokens or any(
            isinstance(tokens, bool) or not isinstance(tokens, int)
            or tokens <= 0 for tokens in request_kv_tokens):
        raise ValueError(
            f"逐请求 KV 上下文长度必须是正整数：{request_kv_tokens!r}")
    return tuple(tokens * per_token for tokens in request_kv_tokens)


def _resident_allocations(
    dpu_id: int,
    prefill_nodes: list[Node],
    decode_nodes: list[Node],
    kv_spec: KVRegionSpec,
    kernel_bytes: int,
    align: int,
    request_kv_tokens: tuple[int, ...] | None = None,
) -> tuple[ResidentAllocation, ...]:
    """逐笔常驻分配台账：权重分片、KV 请求与内核 tile。

    权重记录所属执行算子并按执行顺序排列；KV 按每层、每 KV head 的
    K/V 两份逐块登记，同一 KV head 的两块共用所属 attention 执行单元，
    请求峰值按本轮轨迹逐请求给出。激活是两图复用的瞬时区，已计入三区
    总量，不作为跨请求常驻重复计。
    """
    owners = _weight_owner_map(dpu_id, prefill_nodes, decode_nodes)
    grouped: dict[str, list[tuple[str, int]]] = {}
    order_keys: dict[str, tuple[int, int]] = {}
    for name, (order, node, owner) in owners.items():
        spec = node.meta[SPEC_META_KEY]
        detail = spec.shard_map[dpu_id]
        size = bytes_of(
            detail.local_shape, dtype_bytes(spec.dtype), detail.elem_strides)
        grouped.setdefault(owner, []).append((name, align_up(size, align)))
        order_keys.setdefault(owner, order)
    allocations = []
    for owner in sorted(grouped, key=lambda name: order_keys[name]):
        for tensor_name, size in sorted(grouped[owner]):
            allocations.append(ResidentAllocation(
                owner=owner, tensor_key=tensor_name, model_bytes=size))
    attention = (_attention_unit_map(prefill_nodes)
                 or _attention_unit_map(decode_nodes))
    request_bytes = _kv_request_bytes(
        kv_spec, kv_spec.max_seq, request_kv_tokens)
    for layer in sorted(kv_spec.layers):
        # 同一层同一 KV head 的 K/V 属于同一个执行单元，必须落在同一 PU。
        unit = f"{attention[layer]}#kv" if layer in attention else f"kv:{layer}"
        for head in sorted(kv_spec.kv_heads):
            for which in ("k", "v"):
                allocations.append(ResidentAllocation(
                    owner=f"{unit}:{head}",
                    tensor_key=f"kv:{layer}:{head}:{which}",
                    request_bytes_by_request=request_bytes,
                ))
    if kernel_bytes:
        allocations.append(ResidentAllocation(
            owner="kernel", tensor_key="kernel",
            model_bytes=int(kernel_bytes)))
    return tuple(allocations)


def tensor_pu_usage(
    allocations,
    tensors_per_cluster: int,
    *,
    concurrency: int | None = None,
) -> dict[int, dict[str, int]]:
    """按执行单元轮转的初始放置聚合常驻占用。

    与 GeneSim `tensor_pu_capacity_metrics` 相同的容量语义：同一 TensorPU
    上的同一 tensor_key 只算一次模型常驻；逐请求峰值按请求下标相加后，
    再取最大的 concurrency 个请求求和。轮转单位是执行单元（同一算子的
    全部权重与 KV 块落在同一 PU），而不是逐笔分配的下标。
    """
    if isinstance(tensors_per_cluster, bool) or not isinstance(
            tensors_per_cluster, int) or tensors_per_cluster <= 0:
        raise ValueError(
            f"tensors_per_cluster 必须是正整数，收到 {tensors_per_cluster!r}")
    unit_order: list[str] = []
    by_unit: dict[str, list] = {}
    for allocation in allocations:
        if allocation.owner not in by_unit:
            unit_order.append(allocation.owner)
            by_unit[allocation.owner] = []
        by_unit[allocation.owner].append(allocation)
    per_pu: dict[int, dict[str, dict]] = {}
    for unit_index, owner in enumerate(unit_order):
        slot = unit_index % tensors_per_cluster
        tensors = per_pu.setdefault(slot, {})
        for allocation in by_unit[owner]:
            entry = tensors.setdefault(
                allocation.tensor_key, {"model_bytes": 0, "requests": ()})
            entry["model_bytes"] = max(
                entry["model_bytes"], int(allocation.model_bytes))
            requests = tuple(int(value) for value in
                             allocation.request_bytes_by_request)
            if requests:
                if entry["requests"]:
                    width = max(len(entry["requests"]), len(requests))
                    padded = tuple(entry["requests"]) + (0,) * (
                        width - len(entry["requests"]))
                    incoming = requests + (0,) * (width - len(requests))
                    entry["requests"] = tuple(
                        max(left, right)
                        for left, right in zip(padded, incoming))
                else:
                    entry["requests"] = requests
    usage: dict[int, dict[str, int]] = {}
    for slot, tensors in per_pu.items():
        model_bytes = sum(
            entry["model_bytes"] for entry in tensors.values())
        # 与 GeneSim `tensor_pu_capacity_metrics` 相同：先把各张量的逐请求
        # 峰值按请求下标相加，再取最大的 concurrency 个请求求和。
        totals: list[int] = []
        for entry in tensors.values():
            for index, value in enumerate(entry["requests"]):
                if index == len(totals):
                    totals.append(0)
                totals[index] += value
        request_bytes = _top_requests(tuple(totals), concurrency)
        usage[slot] = {
            "model_bytes": model_bytes,
            "request_bytes": request_bytes,
            "total_bytes": model_bytes + request_bytes,
        }
    return usage


def _top_requests(values: tuple[int, ...], concurrency: int | None) -> int:
    """请求峰值上界：取最大的若干个请求占用相加。"""
    if not values:
        return 0
    if concurrency is None or concurrency >= len(values):
        return sum(values)
    if concurrency <= 0:
        return 0
    return sum(sorted(values, reverse=True)[:concurrency])


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
