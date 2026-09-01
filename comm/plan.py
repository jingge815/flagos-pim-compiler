"""根据张量分片信息生成通信计划和 DMA 序列。"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Iterator, Literal

import numpy as np

from contracts.pim_tensor_spec import PIMTensorSpec, RedistributeEdge, TensorShardDetail

CommType = Literal["all_reduce", "all_gather", "all_to_all", "scatter", "local_slice"]


@dataclass(frozen=True)
class CommSegment:
    """通信计划中的数据段，区间采用半开形式 ``[start, end)``。"""

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
    dst_ready_after: tuple = ()    # 写入前等待的读者。


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
        """返回主机到 DPU 的回写段。"""
        return [s for s in self.segments if s.dst_dpu is not None]


# 分片到全局摊平坐标的连续区间。


def _runs(
    shape: tuple[int, ...], placement_kind: str, shard_dim: int, detail: TensorShardDetail
) -> Iterator[tuple[int, int, int]]:
    """将一个分片展开为全局和本地坐标下的连续区间。"""
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


# 重分布边到通信计划条目。


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
    """生成 DPU 到主机的收集段；复制布局只收集一份。"""
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
                dst_offset=g,  # 主机缓冲区中的全局偏移。
                dst_base=0,
            )
        )
        if edge.src_spec.placement.kind == "Replicate":
            break  # 复制布局只收集一份。
    return segs


def _writeback_segments(edge: RedistributeEdge, itemsize: int) -> list[CommSegment]:
    """生成主机到目标 DPU 的回写段；主机目标不生成回写。"""
    if edge.dst_loc["device"] == "host":
        return []
    return [
        _segment(
            edge,
            itemsize,
            src_dpu=None,
            src_range=(g, g + length),  # 主机缓冲区的全局区间。
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


def _check_endpoint_location(name: str, location: dict, spec: PIMTensorSpec) -> None:
    """校验端点位置和张量规格使用相同设备集合。"""
    expected = {"device": "host"} if spec.device == "host" else {
        "device": "dpu", "dpus": sorted(spec.shard_map)
    }
    if location != expected:
        raise ValueError(f"edge endpoint {name}={location} 与 spec={expected} 不一致")


def _entry_of_edge(edge: RedistributeEdge) -> CommPlanEntry:
    try:
        dtype = np.dtype(edge.dtype)
    except TypeError:
        raise ValueError(f"edge {edge.edge_id} 的 dtype {edge.dtype!r} 无法用 numpy 表示（如 bfloat16）") from None
    numel = prod(edge.shape)
    if numel * dtype.itemsize != edge.nbytes:
        raise ValueError(f"edge {edge.edge_id} 的 shape/dtype 与 nbytes={edge.nbytes} 不符")
    _check_endpoint_location("src_loc", edge.src_loc, edge.src_spec)
    _check_endpoint_location("dst_loc", edge.dst_loc, edge.dst_spec)
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
        return entry  # 本地视图切换不需要 DMA。
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
    """将重分布边列表转换为同序的通信计划条目。"""
    return [_entry_of_edge(edge) for edge in edges]


# DMA 序列展开。


@dataclass(frozen=True)
class DmaOp:
    """一条展开的 DMA 调用及其数据段。"""

    kind: Literal["copy_from", "copy_to", "push_from", "push_to", "broadcast_to"]
    segments: tuple[CommSegment, ...]


def _batch(segs: list[CommSegment], key) -> Iterator[list[CommSegment]]:
    """按 key（字节地址 + 长度）把段合批：同 key 的逐 DPU 段 = 一次 push_xfer。"""
    groups: dict[tuple[int, int], list[CommSegment]] = {}
    for seg in segs:
        groups.setdefault(key(seg), []).append(seg)
    yield from groups.values()


def dma_sequence(entry: CommPlanEntry) -> list[DmaOp]:
    """按收集、回写顺序生成 DMA 调用，并合并可批量传输的段。"""
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


# 主机星型拓扑成本模型。


@dataclass(frozen=True)
class HostStarCostModel:
    """主机星型拓扑下的传输建立开销和带宽参数。"""

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


# 通信计划报告。


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
