"""将通信计划转换为经主机转发的 DMA 操作。"""

from __future__ import annotations

import numpy as np

from dataclasses import replace
from backend.dpu_sdk import (
    DPU_XFER_FROM_DPU,
    DPU_XFER_TO_DPU,
    DpuSet,
    dpu_broadcast_to,
    dpu_copy_from,
    dpu_copy_to,
    dpu_prepare_xfer,
    dpu_push_xfer,
)
from comm.plan import CommPlanEntry, CommSegment, DmaOp, dma_sequence


class DmaEngine:
    """封装点对点和批量 DMA 调用，可按请求窗口平移激活区地址。"""

    def __init__(self, dpu_set: DpuSet, act_base_by_dpu: dict[int, int] | None = None,
                 act_shift: int = 0) -> None:
        self._set = dpu_set
        self._act_base_by_dpu = act_base_by_dpu
        self._act_shift = act_shift

    def _addr(self, dpu_id: int, addr: int) -> int:
        """落在激活区的 DPU 地址平移；host 缓冲与权重/KV 区不动。"""
        if self._act_base_by_dpu is None:
            return addr
        base = self._act_base_by_dpu.get(dpu_id)
        if base is None or addr < base:
            return addr
        return addr + self._act_shift

    def copy_from_dpu(self, dpu_id: int, addr: int, length: int, dtype: np.dtype) -> np.ndarray:
        """从一 DPU 的 MRAM[addr, addr+length*itemsize) 读回一段摊平数组（点对点一跳）。"""
        buf = np.empty(length, dtype=dtype)
        dpu_copy_from(self._set.dpu(dpu_id), self._addr(dpu_id, addr), buf, buf.nbytes)
        return buf

    def copy_to_dpu(self, dpu_id: int, addr: int, data: np.ndarray) -> None:
        """把一段摊平数组写入一 DPU 的 MRAM[addr, ...)（点对点一跳）。"""
        flat = np.ascontiguousarray(data).reshape(-1)
        dpu_copy_to(self._set.dpu(dpu_id), self._addr(dpu_id, addr), flat, flat.nbytes)

    def run_collect(self, op: DmaOp, dtype: np.dtype) -> list[np.ndarray]:
        """执行一条收集方向的传输调用，返回与 op.segments 对齐的逐段 host 缓冲。"""
        if op.kind == "copy_from":
            seg = op.segments[0]
            return [self.copy_from_dpu(seg.src_dpu, seg.src_addr, seg.nbytes // dtype.itemsize, dtype)]
        if op.kind != "push_from":
            raise ValueError(f"非收集方向的 DmaOp: {op.kind}")
        bufs = [np.empty(s.nbytes // dtype.itemsize, dtype=dtype) for s in op.segments]
        for seg, buf in zip(op.segments, bufs):
            dpu_prepare_xfer(self._set.dpu(seg.src_dpu), buf)
        first = op.segments[0]
        # 批量扇出共用 first 的基址，按它的 DPU 判一次激活区即可。
        dpu_push_xfer(
            self._set.subset(tuple(s.src_dpu for s in op.segments)),
            DPU_XFER_FROM_DPU,
            self._addr(first.src_dpu, first.src_addr),
            first.nbytes,
        )
        return bufs

    def run_writeback(self, op: DmaOp, host_buf: np.ndarray) -> None:
        """执行一条回写方向的传输调用：从 host_buf 按各段 global_range 取数写 DPU。"""
        if op.kind == "copy_to":
            seg = op.segments[0]
            self.copy_to_dpu(seg.dst_dpu, seg.dst_addr, host_buf[seg.global_range[0] : seg.global_range[1]])
            return
        if op.kind == "broadcast_to":  # 全段共享同一 host 区间与 dst_addr，取第一段
            seg = op.segments[0]
            targets = self._set.subset(tuple(s.dst_dpu for s in op.segments))
            payload = np.ascontiguousarray(host_buf[seg.global_range[0] : seg.global_range[1]])
            dpu_broadcast_to(targets, self._addr(seg.dst_dpu, seg.dst_addr),
                             payload, payload.nbytes)
            return
        if op.kind != "push_to":
            raise ValueError(f"非回写方向的 DmaOp: {op.kind}")
        first = op.segments[0]
        for seg in op.segments:
            view = np.ascontiguousarray(host_buf[seg.global_range[0] : seg.global_range[1]])
            dpu_prepare_xfer(self._set.dpu(seg.dst_dpu), view)
        dpu_push_xfer(
            self._set.subset(tuple(s.dst_dpu for s in op.segments)),
            DPU_XFER_TO_DPU,
            self._addr(first.dst_dpu, first.dst_addr),
            first.nbytes,
        )


def clamped_entry(entry: CommPlanEntry, rows: int | None) -> CommPlanEntry:
    """按实际行数裁剪通信段（D2-full）：截断到前 rows 行。

    段的 global_range 落在摊平行主序数组上：列切片段每段恰好一行内连续，
    整行段则可能跨多行，必须把段尾截到 rows*R 并同步 nbytes，否则会把
    DPU 尾行的旧值读进 host。rows 为空时原样返回。
    """
    if rows is None or not entry.shape:
        return entry
    if not entry.t_axes:
        raise ValueError(
            f"通信条目 {entry.edge_id} 缺少 t_axes：行收缩只允许显式声明 T 维的"
            "条目（禁止静默按启发式裁剪）")
    if len(entry.t_axes) > 1:
        raise NotImplementedError(
            f"通信条目 {entry.edge_id} 有多个 T 轴 {entry.t_axes}："
            "多 T 轴布局重映射尚未实现")
    row_elems = entry.numel // entry.shape[entry.t_axes[0]]
    limit = rows * row_elems
    itemsize = entry.dtype.itemsize
    segments = []
    for seg in entry.segments:
        start, end = seg.global_range
        if start >= limit:
            continue
        end = min(end, limit)
        if end != seg.global_range[1]:
            seg = replace(seg, global_range=(start, end),
                          nbytes=(end - start) * itemsize)
        segments.append(seg)
    return replace(entry, segments=segments)


def _check_entry(entry: CommPlanEntry, expect: str) -> None:
    if entry.type != expect:
        raise ValueError(f"{expect} 收到 type={entry.type} 的条目（edge {entry.edge_id}）")


def _collect(entry: CommPlanEntry, engine: DmaEngine,
             rows: int | None = None) -> list[tuple[CommSegment, np.ndarray]]:
    """收集阶段：执行全部收集传输，返回 (段, 该段 host 缓冲) 对。"""
    entry = clamped_entry(entry, rows)
    out: list[tuple[CommSegment, np.ndarray]] = []
    for op in dma_sequence(entry):
        if op.kind in ("copy_from", "push_from"):
            out.extend(zip(op.segments, engine.run_collect(op, entry.dtype)))
    return out


def _writeback(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray,
               rows: int | None = None) -> None:
    """将主机缓冲区写回目标 DPU。"""
    entry = clamped_entry(entry, rows)
    for op in dma_sequence(entry):
        if op.kind in ("copy_to", "push_to", "broadcast_to"):
            engine.run_writeback(op, host_buf)


def _row_elems(entry: CommPlanEntry) -> int:
    """T 轴之外一行的元素数（行收缩的搬运粒度）。"""
    return entry.numel // entry.shape[entry.t_axes[0]]


def _actual_shape(entry: CommPlanEntry, rows: int) -> tuple[int, ...]:
    """行收缩后的实际形状：T 维换成实际行数。"""
    axis = entry.t_axes[0]
    return (*entry.shape[:axis], rows, *entry.shape[axis + 1:])


def _out_shape(entry: CommPlanEntry, rows: int | None) -> tuple[int, ...]:
    """收缩时返回实际形状，否则包络形状。"""
    return _actual_shape(entry, rows) if rows is not None else entry.shape


def _merged(entry: CommPlanEntry, collected: list[tuple[CommSegment, np.ndarray]],
            rows: int | None = None) -> np.ndarray:
    """按全局区间将各段缓冲合并为摊平数组。"""
    total = rows * _row_elems(entry) if rows is not None else entry.numel
    host_buf = np.zeros(total, dtype=entry.dtype)
    for seg, buf in sorted(collected, key=lambda item: item[0].global_range[0]):
        host_buf[seg.global_range[0] : seg.global_range[1]] = buf
    return host_buf


def _check_host_size(entry: CommPlanEntry, flat: np.ndarray, rows: int | None) -> None:
    """主机缓冲必须盖住要写的范围；动态收缩时下界是实际行数，不是包络。"""
    if rows is None:
        if flat.size != entry.numel:
            raise ValueError(
                f"scatter/broadcast 的 host_buf 元素数 {flat.size} != 条目 numel {entry.numel}")
        return
    if not entry.t_axes:
        raise ValueError(
            f"通信条目 {entry.edge_id} 缺少 t_axes：行收缩只允许显式声明 T 维的条目")
    row_elems = entry.numel // entry.shape[entry.t_axes[0]]
    if flat.size < rows * row_elems:
        raise ValueError(
            f"scatter/broadcast 的 host_buf 元素数 {flat.size} 少于实际 "
            f"{rows} 行所需的 {rows * row_elems}")


# 通信原语：收集、主机处理和回写。


def all_reduce(entry: CommPlanEntry, engine: DmaEngine,
          rows: int | None = None) -> np.ndarray:
    """收集各 DPU 分片并执行求和或求均值，返回全局形状数组。"""
    _check_entry(entry, "all_reduce")
    total = rows * _row_elems(entry) if rows is not None else entry.numel
    acc = np.zeros(total, dtype=entry.dtype)
    for seg, buf in _collect(entry, engine, rows):
        acc[seg.global_range[0] : seg.global_range[1]] += buf
    if entry.reduce == "mean":
        # 除数取原始条目的收集段数：行收缩只减少搬运量，不改变参与者个数。
        acc /= len(entry.collect_segments)
    _writeback(entry, engine, acc, rows)
    return acc.reshape(_out_shape(entry, rows))


def all_gather(entry: CommPlanEntry, engine: DmaEngine,
          rows: int | None = None) -> np.ndarray:
    """收集各 DPU 分片并按全局偏移拼接，返回全局形状数组。"""
    _check_entry(entry, "all_gather")
    merged = _merged(entry, _collect(entry, engine, rows), rows)
    _writeback(entry, engine, merged, rows)
    return merged.reshape(_out_shape(entry, rows))


def all_to_all(entry: CommPlanEntry, engine: DmaEngine,
          rows: int | None = None) -> np.ndarray:
    """按全局区间重排分片并写回目标 DPU，返回摊平后的全局数组。"""
    _check_entry(entry, "all_to_all")
    staging = _merged(entry, _collect(entry, engine, rows), rows)
    _writeback(entry, engine, staging, rows)
    return staging.reshape(_out_shape(entry, rows))


def scatter(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray,
           rows: int | None = None) -> None:
    """将与条目形状一致的主机数组按目标分片写入各 DPU。"""
    _check_entry(entry, "scatter")
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    _check_host_size(entry, flat, rows)
    _writeback(entry, engine, flat, rows)


def broadcast(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray,
             rows: int | None = None) -> None:
    """将主机数组完整复制到条目列出的各 DPU。"""
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    _check_host_size(entry, flat, rows)
    _writeback(entry, engine, flat, rows)
