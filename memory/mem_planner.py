"""问题 8：内存管理——权重/KV/激活三区 offset 规划 + 容量校验（方案问题 8 三）。

`plan_dpu` 是编译期主入口，每个 DPU 各调一次，把该 DPU 在 prefill/decode 两张
导出图上的全部节点联合规划成一份 `DPUPlan`：

- 权重区：两图共用同一套 offset（两图权重相同，按名字去重后顺序打包），
  算出的 offset 直接回填进两张图里所有同名 get_attr 节点的
  `PIMTensorSpec.shard_map[dpu_id].mram_offset`（问题 3 的通信计划表已在读
  这个字段，一直是默认值 0）。
- KV 区：只定 `kv_base`，区内偏移与真实占用交给问题 7 的 `build_kv_layout`
  （用它返回的 `kv_allocated_bytes` 推进激活区起点，不重算 `kv_bytes` 公式，
  避免对齐 padding 累积侵入激活区）。
- 激活区：prefill/decode 互斥 overlay，共享同一 `act_base`，各自跑一遍
  liveness + `greedy_reuse` 贪心装箱，大小取两图峰值。

不实现方案里提到的 `[阶段2]` 自动重切反馈——装不下直接抛错，交给人工调整
切分配置（CLAUDE.md：第 2/3 阶段特性当前不实现）。
"""

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
    """硬件输入（方案问题 8 二"蓝图同时依赖模型/切分/硬件三组输入"里的硬件部分）。"""

    mram_bytes: int          # 单 DPU MRAM 总量（≤8GB）
    align: int                # DMA 对齐边界
    sys_reserve_bytes: int    # 系统预留：kernel 二进制 + runtime 栈 + workspace 等非三区占用


@dataclass
class DPUPlan:
    """单台 DPU 的静态内存蓝图（方案问题 8 三"规划器骨架"的输出契约）。"""

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
    """一个临时激活张量的 liveness 区间（方案"激活区"一节的 `transient_tensors`）。

    readers 除本地消费者节点名外，还含 redistribute 边隐式产生的读者，标记为
    ``f"redist:e{edge_id}"``——`node.users` 覆盖不到这类读取（方案问题 8 四）。
    """

    name: str
    size: int
    produced_at: int
    last_read_at: int
    readers: list[str] = field(default_factory=list)


def bytes_of(local_shape: tuple[int, ...], itemsize: int) -> int:
    """本地分片 shape × dtype 字节数（方案骨架同名 helper）。"""
    return prod(local_shape) * itemsize


def weights_of(dpu_nodes: list[Node], dpu_id: int) -> dict[str, Node]:
    """挑出归属 dpu_id 的权重常量（get_attr 且 spec.device=="dpu"）。

    键为权重名（`node.target`），供跨 prefill/decode 两图按名字去重合并。
    """
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
    """权重区打包：两图权重按名字去重取并集，顺序打包 + 对齐，回填两图的 mram_offset。

    两图的同一份权重是两次独立 export 出的不同 Node 对象，必须按名字（而非
    Node 对象）去重，否则会被当成两份各占一次空间。回填前校验两图同名权重的
    本地分片 shape 一致——两图共享同一份切分，不一致即 bug，不静默择一。
    """
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
    """按 `exec_plan_gen` 真实的命令生成顺序算细粒度步骤号（问题 8 缺口修复）。

    此前 liveness 用 `enumerate(dpu_nodes)` 的粗粒度节点位置当时间戳，把"一个
    消费节点自己的计算"和"它入边上全部 redistribute 边的搬运命令"都记成同一
    个位置——但 `exec_plan_gen.build_execution_plan` 对同一节点的多条
    redistribute 边是按列表顺序**逐条生成命令、严格先后执行**的（先搬完第
    一条边、再搬第二条边、最后才轮到节点自己），粗粒度时间戳把这些先后关系
    全部抹平成"同时刻"，让本该顺序执行、后一条会读前一个地址的两个动作被
    liveness 误判为"互不冲突可共用地址"。真实复现：`add_3` 同时有
    `embedding` 的 scatter 边（先搬）与 `linear_3` 的 all_reduce 边（后搬，
    要读 `linear_3` 自己刚写的输出），若两者的落地缓冲/隐式读时刻被记成同一
    个点，scatter 可能被判定与 `linear_3` 输出复用同一 offset，先把它覆盖，
    all_reduce 再读到的就是 scatter 的数据而非 `linear_3` 的输出。

    出: (node_step, edge_step)——`node_step[node]` 是该节点自己命令的步骤号，
    `edge_step[edge_id]` 是该边搬运命令的步骤号，严格早于消费节点自己的
    步骤号；同一消费节点上的多条边按列表顺序各占独立的步骤号。
    """
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
    """本 DPU 落地的 redistribute 边，按 edge_id 索引（供 `plan_dpu` 回填用）。

    排除 `local_slice`：它是同 DPU 上 Replicate→Shard 的本地视角切换，零 DMA
    （`comm/plan.py` 对它直接返回空 segments，从不算 `dst_addr`），没有真实的
    落地缓冲——给它分配激活区地址是浪费空间且没人会用（写不进 `dst_addr`）。
    """
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
    """挑出落在本 DPU 的 redistribute 回写落地缓冲，算 liveness（方案问题 8 四）。

    `comm/plan.py` 的 `_writeback_segments` 从 `edge.dst_spec.shard_map[dpu_id]
    .mram_offset` 取回写地址——但 `dst_spec` 是问题 2 `_diff_edge` 里当场
    `_required_spec()` 新建的一份 `PIMTensorSpec`，不是消费节点自己的
    `node.meta["spec"]`，从未被回填过（一直是构造默认值 0）。这里把每条落在
    本 DPU 的回写边算成一个临时张量，参与 `greedy_reuse` 统一装箱，装箱结果
    在 `plan_dpu` 里回填回 `edge.dst_spec.shard_map[dpu_id].mram_offset`。

    生命周期用 `_build_step_index` 算出的细粒度步骤号：`produced_at` = 该边
    自己的搬运命令步骤号（`edge_step`，回写发生的时刻），`last_read_at` =
    消费节点自己命令的步骤号（`node_step`，消费节点读它的时刻）——不能再用
    粗粒度的"消费节点位置"同时当 produced_at 和 last_read_at（问题 8 缺口
    修复：那样会把"同一消费节点上先搬的这条边"和"后搬、要读别的张量的那
    条边/张量"错判成同一时刻，允许后者复用前者的地址，实际执行时后者的
    搬运命令会先把前者覆盖掉）。键名 ``f"redist_dst:e{edge_id}"`` 与
    `transient_tensors` 用的 `f"redist:e{edge_id}"` 读者标记是同一命名体系
    里的一对（一个记落地空间、一个记隐式读取）。
    """
    tensors: list[TransientTensor] = []
    for node in dpu_nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.type == "local_slice":
                continue  # 零 DMA，无落地缓冲（同上 _redistribute_edges_landing_on）
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
    """挑出归属 dpu_id 的临时激活并算 liveness 区间（方案"激活区"一节）。

    本地读者取 `node.users` 中未触发 redistribute 的同图消费者；redistribute
    边的隐式读者（`copy_from`）另计，取自消费方 `node.meta["redistribute"]`
    里 src 指向本张量、且本 dpu 在 `src_loc["dpus"]` 内的边。第 1 阶段白名单
    （linear/add/mul/tanh）不产出 view，故不处理别名共享存储的情形。同时并入
    `redistribute_landing_tensors` 的回写落地缓冲，一次性交给 `greedy_reuse`
    统一装箱（不单开一个区，复用判定与其余激活张量共享同一份逻辑）。
    """
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
                continue  # 该边需要重分布，真实读取是下面的 copy_from，不是本地直读
            readers.append(user.name)
            last_read_at = max(last_read_at, node_step[user])

        for other, edge in redist_by_src.get(node.name, []):
            if edge.src_loc.get("device") != DEVICE_DPU or dpu_id not in edge.src_loc.get("dpus", []):
                continue
            readers.append(f"redist:e{edge.edge_id}")
            # 真实读取时刻是该边自己搬运命令的步骤（edge_step），不是消费节点
            # 自己的步骤（node_step[other]）——同一消费节点上排在它前面的别的
            # 边会先执行，若用 node_step[other] 当读取时刻，会让"这条边执行
            # 前"的窗口被误判成"已经读完"，允许更早的、不相关的写把这块地址
            # 提前覆盖（问题 8 缺口修复，见 `_build_step_index`）。
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
    """激活区 liveness + 贪心装箱（借 ExecuTorch `greedy`，方案"激活区"一节）。

    按大小降序处理：每个张量尝试复用已开的、大小够且时间区间不冲突的最低
    offset 的 slot；找不到就在当前顶端新开一个 slot。两个张量的区间
    `[produced_at, last_read_at]` 只有在其一的写入不早于另一的最后读取时才
    视为不冲突（即复用者的写发生在被复用者最后一个读者已执行之后，允许在
    同一条命令内先读后写、时刻相等）。

    额外的强制约束：**共享读者的两个张量永不复用同一 offset**，不管区间
    数学怎么算。这条规则堵住一个真实出现过的漏洞——同一个消费节点若同时
    读两个不同的 redistribute 落地缓冲（如 `mul(to_7, rsqrt)` 两个入边分别
    landing 到 `mul_2`），两个落地缓冲的 `produced_at`/`last_read_at` 都退化
    成同一个点（消费节点自己的图位置，redistribute 命令没有独立的图节点
    表示自己的执行时刻）。纯区间判定会把这两个"同一时刻互不相关"的退化点
    错判成不冲突，两者共用一个地址，导致消费节点读到的第二个操作数其实是
    第一个操作数的数据——这是一次真实的数值 bug（真实 Llama-2-7B 图上
    RMSNorm 的 `mul(weight, x*rsqrt)` 复现过），不是理论假设。区间判定继续
    保留（它覆盖"跨越多个图位置的正常生命周期"场景），共享读者判定专门
    补上区间判定在退化点上失效的这个缺口。
    出: (张量名 -> offset, 该图激活区末尾地址，对齐后)。
    """
    slots: list[dict] = []  # [{"offset", "size", "timeline": [(produced_at, last_read_at, readers_set)]}]
    offsets: dict[str, int] = {}
    top = base
    for t in sorted(tensors, key=lambda t: (-t.size, t.produced_at, t.name)):
        t_readers = set(t.readers)
        placed = False
        for slot in slots:
            if t.size <= slot["size"] and all(
                (start >= t.last_read_at or t.produced_at >= end) and not (t_readers & readers)
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
    """对每个被 `greedy_reuse` 判定复用的地址，产出复用前必须等待的读者列表。

    一个地址被 N(>=2) 个张量占用时，除时间上最后一个占用者外，其余占用者的
    读者列表原样并入该地址的 `pending_readers`——供问题 6 生成 `ExecutionPlan`
    时把这份节点级读者翻译为具体命令的 `waits`（方案问题 8 三"原地写回的
    安全性"）。等待多于严格必需的读者不影响正确性，只是保守。
    """
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
    """问题 8 主入口：算三区 offset、回填 mram_offset、容量校验（方案问题 8 三骨架）。

    入: dpu_id；prefill_nodes/decode_nodes —— 该 DPU 参与的两张导出图各自的
    【全部】节点（`list(gm.graph.nodes)`，内部按 dpu_id 过滤）；kv_specs ——
    问题 7 `kv_specs_from_placement` 产出的、`kv_base` 待填的规格表；hw ——
    硬件预算。
    出: DPUPlan。副作用：回填两图内该 dpu 全部权重节点的
    `spec.shard_map[dpu_id].mram_offset`，以及 `kv_specs[dpu_id]` 的
    `kv_base`/`kv_off`/`kv_allocated_bytes`。容量超限抛 ValueError。
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

    # 回填激活张量自己的 mram_offset：greedy_reuse 只算出 offset 表，从未写回
    # 节点自己的 spec.shard_map[dpu_id]（此前只有权重区、KV 区回填，这里补上
    # 激活区这一类——没有它，问题 6 的 exec_plan_gen 无法从 spec 拿到正确地址）。
    for nodes, offsets in ((prefill_nodes, act_prefill), (decode_nodes, act_decode)):
        by_name = {n.name: n for n in nodes}
        for name, node_off in offsets.items():
            if name.startswith("redist_dst:"):
                continue  # 落地缓冲不对应图节点自己的 spec，下面单独回填
            node = by_name[name]
            spec = node.meta[SPEC_META_KEY]
            spec.shard_map[dpu_id] = replace(spec.shard_map[dpu_id], mram_offset=node_off)

    # 回填 redistribute 落地缓冲的 mram_offset（问题 8 缺口修复，见模块 docstring）：
    # dst_spec 是问题 2 当场新建的 PIMTensorSpec，不是消费节点自己的 spec，需单独回填。
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
