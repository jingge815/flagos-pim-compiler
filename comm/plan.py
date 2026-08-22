"""问题 3（编译期半）：通信计划表——把问题 2 的 RedistributeEdge 按数据段展开。

每条 redistribute 边展开为一个 CommPlanEntry（方案问题 3 二.(4) 的表）：
边级 ``wait_for``（读前等待的生产者 DPU 集合）+ 逐 segment 行。segment 字段与
方案表格一一对应；``dst_ready_after`` 的数据源是问题 8 的 pending_readers，问题 8
未就位前恒为空。

区间换算统一在"全局摊平（行主序）元素坐标"下进行：任何 Shard(d) 的连续分片
= 摊平后 prod(shape[:d]) 段连续 run（外维循环 × 切维连续段），每段在全局缓冲与
本 DPU 本地缓冲各自连续。这修正了方案 nbytes 公式只在一维/最内维切分下成立的
漏洞，使任意维切分（如 logits 的 Shard(2)，S>1 时全局缓冲中交错分布）的段描述
与真实字节布局一致。

segment 上除方案表格的元素区间外，另烘烤两个绝对字节地址字段（src_addr/dst_addr
= TensorShardDetail.mram_offset + 本地元素偏移 × itemsize），使本表自包含：
编排器/通信库照表执行即可，无需回查 shard_map。问题 8 就位后只改 mram_offset
取值，生成规则不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Iterator, Literal

import numpy as np

from contracts.pim_tensor_spec import PIMTensorSpec, RedistributeEdge, TensorShardDetail

CommType = Literal["all_reduce", "all_gather", "all_to_all", "scatter", "local_slice"]


@dataclass(frozen=True)
class CommSegment:
    """通信计划表的一行（方案问题 3 二.(4) 表，逐段字段）。

    元素区间均为半开 [start, end)。src_dpu/dst_dpu 为 None 表示该端是 host
    归约/合并缓冲区；host 端的 src_local_range/dst_local_offset 解释为 host
    缓冲区内的元素位置（host 缓冲即全局摊平张量布局）。all_to_all 的段
    src_dpu 与 dst_dpu 均非 None（两跳经 host，源段与目标段合一描述）。
    """

    edge_id: int
    type: CommType
    src_dpu: int | None
    src_local_range: tuple[int, int]
    global_range: tuple[int, int]
    dst_dpu: int | None
    dst_local_offset: int
    nbytes: int
    reduce: str | None = None
    src_addr: int = 0              # 源端绝对字节地址（DPU 侧=MRAM，host 侧=缓冲内偏移）
    dst_addr: int = 0              # 目标端绝对字节地址，同上
    dst_ready_after: tuple = ()    # 写前等待的读者；问题 8 pending_readers 填入


@dataclass
class CommPlanEntry:
    """一条 redistribute 边的完整搬运计划：wait_for + 逐 segment 行。"""

    edge_id: int
    type: CommType
    reduce: str | None             # 归约方式，仅 all_reduce 使用
    shape: tuple[int, ...]         # 全局逻辑张量形状
    dtype: np.dtype
    dst_loc: dict                  # {"device": "host"} 或 {"device": "dpu", "dpus": [...]}
    wait_for: tuple[int, ...]      # 边级读前等待：全部源 DPU（host 源为空）
    segments: list[CommSegment] = field(default_factory=list)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def collect_segments(self) -> list[CommSegment]:
        """DPU → host 的收集段（all_to_all 段两端皆 DPU，同时属于收集与回写）。"""
        return [s for s in self.segments if s.src_dpu is not None]

    @property
    def writeback_segments(self) -> list[CommSegment]:
        """host → DPU 的回写段；dst_loc 为 host 时为空（方案二.(4)：结果只落 host）。"""
        return [s for s in self.segments if s.dst_dpu is not None]


# ---------------------------------------------------------------------------
# 分片 → 全局摊平坐标下的连续 run 列表
# ---------------------------------------------------------------------------


def _runs(
    shape: tuple[int, ...], placement_kind: str, shard_dim: int, detail: TensorShardDetail
) -> Iterator[tuple[int, int, int]]:
    """一个分片展开为 [(global_start, local_start, length)]（元素单位，摊平坐标）。

    Shard(d) 的分片在摊平坐标下是 prod(shape[:d]) 段连续 run（外维每行一段，
    段长 = 切宽 × 内维积）；Replicate/Partial 恒为单段 [0, numel)。
    """
    if placement_kind != "Shard":
        yield 0, 0, prod(shape)
        return
    inner = prod(shape[shard_dim + 1 :])
    width = detail.end_idx - detail.start_idx
    for p in range(prod(shape[:shard_dim])):
        yield (
            p * shape[shard_dim] * inner + detail.start_idx * inner,
            p * width * inner,
            width * inner,
        )


def _global_shape(spec: PIMTensorSpec) -> tuple[int, ...]:
    """从 shard_map 反推全局形状：Shard(d) 的切维长度 = 各分片宽度之和。"""
    detail = next(iter(spec.shard_map.values()))
    shape = detail.local_shape
    if spec.placement.kind != "Shard":
        return shape
    dim = spec.placement.dim
    total = sum(d.end_idx - d.start_idx for d in spec.shard_map.values())
    return shape[:dim] + (total,) + shape[dim + 1 :]


def _spec_runs(spec: PIMTensorSpec) -> Iterator[tuple[int, TensorShardDetail, int, int, int]]:
    """spec 的全体 (dpu_id, detail, global_start, local_start, length)。host spec 为空。"""
    if not spec.shard_map:
        return
    shape = _global_shape(spec)
    for dpu_id, detail in spec.shard_map.items():
        for global_start, local_start, length in _runs(
            shape, spec.placement.kind, spec.placement.dim or 0, detail
        ):
            yield dpu_id, detail, global_start, local_start, length


# ---------------------------------------------------------------------------
# RedistributeEdge → CommPlanEntry（方案二.(4) 各类型生成规则）
# ---------------------------------------------------------------------------


def _segment(
    edge: RedistributeEdge,
    itemsize: int,
    *,
    src_dpu: int | None,
    src_range: tuple[int, int],
    src_base: int,
    global_range: tuple[int, int],
    dst_dpu: int | None,
    dst_offset: int,
    dst_base: int,
) -> CommSegment:
    length = global_range[1] - global_range[0]
    return CommSegment(
        edge_id=edge.edge_id,
        type=edge.type,
        src_dpu=src_dpu,
        src_local_range=src_range,
        global_range=global_range,
        dst_dpu=dst_dpu,
        dst_local_offset=dst_offset,
        nbytes=length * itemsize,
        reduce=edge.reduce_type if edge.type == "all_reduce" else None,
        src_addr=src_base + src_range[0] * itemsize,
        dst_addr=dst_base + dst_offset * itemsize,
    )


def _collect_segments(edge: RedistributeEdge, itemsize: int) -> list[CommSegment]:
    """收集段：源 shard_map 每个分片按其全局 run 展开，dst_dpu=None（去向 host）。

    源为 Replicate（同布局 dpu→host 退化，方案二.(9)：只收一份）时只取一台。
    """
    segs = []
    for dpu_id, detail, g, local, length in _spec_runs(edge.src_spec):
        segs.append(
            _segment(
                edge,
                itemsize,
                src_dpu=dpu_id,
                src_range=(local, local + length),
                src_base=detail.mram_offset,
                global_range=(g, g + length),
                dst_dpu=None,
                dst_offset=g,  # 分片在 host 合并缓冲区中的位置 = 全局摊平偏移
                dst_base=0,
            )
        )
        if edge.src_spec.placement.kind == "Replicate":
            break  # 全量副本只收一台（方案：源为全量副本时只收一份）
    return segs


def _writeback_segments(edge: RedistributeEdge, itemsize: int) -> list[CommSegment]:
    """回写段：dst_loc 为 host 时不生成；为 dpu 时按 dst_spec 分片逐 run 展开。

    目标 DPU 集合取自 dst_spec 的 shard_map（与 dst_loc.dpus 一致），与源集合
    相互独立（方案二.(4)：不隐含源集合 = 目标集合）。src_dpu=None（来源 host）。
    """
    if edge.dst_loc["device"] == "host":
        return []
    return [
        _segment(
            edge,
            itemsize,
            src_dpu=None,
            src_range=(g, g + length),  # host 源：缓冲内区间即全局摊平区间
            src_base=0,
            global_range=(g, g + length),
            dst_dpu=dpu_id,
            dst_offset=local,
            dst_base=detail.mram_offset,
        )
        for dpu_id, detail, g, local, length in _spec_runs(edge.dst_spec)
    ]


def _all_to_all_segments(edge: RedistributeEdge, itemsize: int) -> list[CommSegment]:
    """Shard(i)→Shard(j)：源、目标两份 shard_map 的 run 两两求交，每个非空交集一行。"""
    segs = []
    for s_dpu, s_detail, s_g, s_local, s_len in _spec_runs(edge.src_spec):
        for d_dpu, d_detail, d_g, d_local, d_len in _spec_runs(edge.dst_spec):
            start, end = max(s_g, d_g), min(s_g + s_len, d_g + d_len)
            if start >= end:
                continue
            segs.append(
                _segment(
                    edge,
                    itemsize,
                    src_dpu=s_dpu,
                    src_range=(s_local + start - s_g, s_local + end - s_g),
                    src_base=s_detail.mram_offset,
                    global_range=(start, end),
                    dst_dpu=d_dpu,
                    dst_offset=d_local + start - d_g,
                    dst_base=d_detail.mram_offset,
                )
            )
    return segs


def _check_coverage(segs: list[CommSegment], expect_full: bool, expected: int, label: str) -> None:
    """段的 global_range 校验：Shard 端必须恰好平铺 [0, expected) 一次；
    Replicate/Partial 端每段都必须是全量 [0, expected)。"""
    if expect_full:
        for seg in segs:
            if seg.global_range != (0, expected):
                raise ValueError(f"{label} 段 {seg.global_range} 不是全量 [0, {expected})")
        return
    cursor = 0
    for start, end in sorted(seg.global_range for seg in segs):
        if start != cursor:
            raise ValueError(f"{label} 段全局区间在 [{cursor}, {start}) 处断裂/重叠")
        cursor = end
    if cursor != expected:
        raise ValueError(f"{label} 段覆盖 [0, {cursor}) != [0, {expected})")


def _entry_of_edge(edge: RedistributeEdge) -> CommPlanEntry:
    try:
        dtype = np.dtype(edge.dtype)
    except TypeError:
        raise ValueError(f"edge {edge.edge_id} 的 dtype {edge.dtype!r} 无法用 numpy 表示（如 bfloat16）") from None
    numel = prod(edge.shape)
    if numel * dtype.itemsize != edge.nbytes:
        raise ValueError(f"edge {edge.edge_id} 的 shape/dtype 与 nbytes={edge.nbytes} 不符")
    wait_for = tuple(edge.src_loc["dpus"]) if edge.src_loc["device"] == "dpu" else ()
    entry = CommPlanEntry(
        edge_id=edge.edge_id,
        type=edge.type,
        reduce=edge.reduce_type if edge.type == "all_reduce" else None,
        shape=edge.shape,
        dtype=dtype,
        dst_loc=edge.dst_loc,
        wait_for=wait_for,
    )
    if edge.type == "local_slice":
        return entry  # 本地视角切换，零 DMA（方案二.(8)）
    if edge.type == "all_to_all":
        entry.segments = _all_to_all_segments(edge, dtype.itemsize)
        _check_coverage(entry.segments, False, numel, f"edge {edge.edge_id} all_to_all")
        return entry
    collect, writeback = _collect_segments(edge, dtype.itemsize), _writeback_segments(edge, dtype.itemsize)
    entry.segments = collect + writeback
    if collect:
        _check_coverage(
            collect, edge.src_spec.placement.kind != "Shard", numel, f"edge {edge.edge_id} 收集"
        )
    if writeback:
        _check_coverage(
            writeback, edge.dst_spec.placement.kind != "Shard", numel, f"edge {edge.edge_id} 回写"
        )
    return entry


def build_comm_plan(edges: list[RedistributeEdge]) -> list[CommPlanEntry]:
    """问题 3 编译期主入口：redistribute 边列表 → 通信计划表（每边一个条目）。

    入: edges —— propagate_specs 的产出（问题 2）。
    出: CommPlanEntry 列表（与 edges 同序）；local_slice 边保留空 segments 的
        占位条目（零 DMA，供问题 6 统一按 edge_id 查表）。
    """
    return [_entry_of_edge(edge) for edge in edges]


# ---------------------------------------------------------------------------
# DMA 序列展开（附录 B.3）：编译期可算，成本模型与 comm/lowering.py 执行共用
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DmaOp:
    """一条展开的厂商 SDK 传输调用。

    kind: copy_from/copy_to（点对点单段）；push_from/push_to（同地址同长度的
    逐 DPU 传输合批为一次 dpu_push_xfer，方案二.(2) 的批量优化）；broadcast_to
    （同一份 host 数据写全部目标 DPU，一次 dpu_broadcast_to）。
    """

    kind: Literal["copy_from", "copy_to", "push_from", "push_to", "broadcast_to"]
    segments: tuple[CommSegment, ...]


def _batch(segs: list[CommSegment], key) -> Iterator[list[CommSegment]]:
    """按 key（字节地址 + 长度）把段合批：同 key 的逐 DPU 段 = 一次 push_xfer。"""
    groups: dict[tuple[int, int], list[CommSegment]] = {}
    for seg in segs:
        groups.setdefault(key(seg), []).append(seg)
    yield from groups.values()


def dma_sequence(entry: CommPlanEntry) -> list[DmaOp]:
    """把条目展开为有序的 SDK 传输调用序列（收集方向在前，回写方向在后）。

    host 端的归约/拼接/重排发生在这两个方向之间，由 comm/lowering.py 的各原语
    完成，不在本序列内。回写段全体共享同一 host 区间与同一 dst_addr 时（同一份
    结果写全部目标），合并为一次 broadcast_to；否则按 (dst_addr, nbytes) 合批为
    push_to；收集段按 (src_addr, nbytes) 合批为 push_from；单段退化为 copy_*。
    """
    ops: list[DmaOp] = []
    for group in _batch(entry.collect_segments, lambda s: (s.src_addr, s.nbytes)):
        ops.append(DmaOp("push_from" if len(group) > 1 else "copy_from", tuple(group)))
    writeback = entry.writeback_segments
    if writeback and len({(s.global_range, s.dst_addr) for s in writeback}) == 1:
        ops.append(DmaOp("broadcast_to", tuple(writeback)))
    else:
        for group in _batch(writeback, lambda s: (s.dst_addr, s.nbytes)):
            ops.append(DmaOp("push_to" if len(group) > 1 else "copy_to", tuple(group)))
    return ops


# ---------------------------------------------------------------------------
# 接口成本模型（host-star 拓扑）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostStarCostModel:
    """host-star 拓扑的接口成本：每次 SDK 传输一次固定建立开销 + 主机带宽线性项。

    DPU 间无直连，一切跨 DPU 交换经 host 两跳，成本只按 host↔DPU 字节计；
    push_xfer/broadcast_to 合批只付一次建立开销。
    """

    dma_setup_s: float = 1e-5
    host_bytes_per_s: float = 25e9
    topology: str = "host_star"


@dataclass(frozen=True)
class CommCost:
    transfers: int  # 实际 SDK 传输调用次数（合批后）
    nbytes: int     # host↔DPU 传输总字节
    seconds: float


def plan_cost(entry: CommPlanEntry, model: HostStarCostModel | None = None) -> CommCost:
    """估算一个条目的接口成本：按 dma_sequence 的实际传输计次、按段字节计量。"""
    model = model or HostStarCostModel()
    ops = dma_sequence(entry)
    nbytes = sum(seg.nbytes for op in ops for seg in op.segments)
    return CommCost(
        transfers=len(ops),
        nbytes=nbytes,
        seconds=len(ops) * model.dma_setup_s + nbytes / model.host_bytes_per_s,
    )


# ---------------------------------------------------------------------------
# 可读报告（定位问题用）
# ---------------------------------------------------------------------------


def format_comm_plan(entries: list[CommPlanEntry], *, max_segments: int | None = None) -> str:
    """把通信计划表格式化为可读文本：逐条目的边信息 + segment 行 + 分类统计。"""
    lines = [f"== 通信计划表（共 {len(entries)} 条）=="]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.type] = counts.get(entry.type, 0) + 1
        lines.append(
            f"e{entry.edge_id}: {entry.type} shape={entry.shape} dtype={entry.dtype}"
            f" wait_for={list(entry.wait_for)} dst_loc={entry.dst_loc} segments={len(entry.segments)}"
        )
        shown = entry.segments if max_segments is None else entry.segments[:max_segments]
        for seg in shown:
            lines.append(
                f"    src_dpu={seg.src_dpu} src_local={seg.src_local_range}"
                f" global={seg.global_range} dst_dpu={seg.dst_dpu}"
                f" dst_off={seg.dst_local_offset} nbytes={seg.nbytes}"
                f" src_addr={seg.src_addr} dst_addr={seg.dst_addr}"
                + (f" reduce={seg.reduce}" if seg.reduce else "")
            )
        if len(shown) < len(entry.segments):
            lines.append(f"    ... 省略其余 {len(entry.segments) - len(shown)} 段")
    lines.append(f"== 分类统计 == {dict(sorted(counts.items()))}")
    return "\n".join(lines)
