"""将通信计划转换为经主机转发的 DMA 操作。"""

from __future__ import annotations

import numpy as np

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
    """封装点对点和批量 DMA 调用。"""

    def __init__(self, dpu_set: DpuSet) -> None:
        self._set = dpu_set

    def copy_from_dpu(self, dpu_id: int, addr: int, length: int, dtype: np.dtype) -> np.ndarray:
        """从一 DPU 的 MRAM[addr, addr+length*itemsize) 读回一段摊平数组（点对点一跳）。"""
        buf = np.empty(length, dtype=dtype)
        dpu_copy_from(self._set.dpu(dpu_id), addr, buf, buf.nbytes)
        return buf

    def copy_to_dpu(self, dpu_id: int, addr: int, data: np.ndarray) -> None:
        """把一段摊平数组写入一 DPU 的 MRAM[addr, ...)（点对点一跳）。"""
        flat = np.ascontiguousarray(data).reshape(-1)
        dpu_copy_to(self._set.dpu(dpu_id), addr, flat, flat.nbytes)

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
        dpu_push_xfer(
            self._set.subset(tuple(s.src_dpu for s in op.segments)),
            DPU_XFER_FROM_DPU,
            first.src_addr,
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
            dpu_broadcast_to(targets, seg.dst_addr, payload, payload.nbytes)
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
            first.dst_addr,
            first.nbytes,
        )


def _check_entry(entry: CommPlanEntry, expect: str) -> None:
    if entry.type != expect:
        raise ValueError(f"{expect} 收到 type={entry.type} 的条目（edge {entry.edge_id}）")


def _collect(entry: CommPlanEntry, engine: DmaEngine) -> list[tuple[CommSegment, np.ndarray]]:
    """收集阶段：执行全部收集传输，返回 (段, 该段 host 缓冲) 对。"""
    out: list[tuple[CommSegment, np.ndarray]] = []
    for op in dma_sequence(entry):
        if op.kind in ("copy_from", "push_from"):
            out.extend(zip(op.segments, engine.run_collect(op, entry.dtype)))
    return out


def _writeback(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray) -> None:
    """将主机缓冲区写回目标 DPU。"""
    for op in dma_sequence(entry):
        if op.kind in ("copy_to", "push_to", "broadcast_to"):
            engine.run_writeback(op, host_buf)


def _merged(entry: CommPlanEntry, collected: list[tuple[CommSegment, np.ndarray]]) -> np.ndarray:
    """按全局区间将各段缓冲合并为摊平数组。"""
    host_buf = np.zeros(entry.numel, dtype=entry.dtype)
    for seg, buf in sorted(collected, key=lambda item: item[0].global_range[0]):
        host_buf[seg.global_range[0] : seg.global_range[1]] = buf
    return host_buf


# 通信原语：收集、主机处理和回写。


def all_reduce(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """收集各 DPU 分片并执行求和或求均值，返回全局形状数组。"""
    _check_entry(entry, "all_reduce")
    acc = np.zeros(entry.numel, dtype=entry.dtype)
    for seg, buf in _collect(entry, engine):
        acc[seg.global_range[0] : seg.global_range[1]] += buf
    if entry.reduce == "mean":
        acc /= len(entry.collect_segments)
    _writeback(entry, engine, acc)
    return acc.reshape(entry.shape)


def all_gather(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """收集各 DPU 分片并按全局偏移拼接，返回全局形状数组。"""
    _check_entry(entry, "all_gather")
    merged = _merged(entry, _collect(entry, engine))
    _writeback(entry, engine, merged)
    return merged.reshape(entry.shape)


def all_to_all(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """按全局区间重排分片并写回目标 DPU，返回摊平后的全局数组。"""
    _check_entry(entry, "all_to_all")
    staging = _merged(entry, _collect(entry, engine))
    _writeback(entry, engine, staging)
    return staging.reshape(entry.shape)


def scatter(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray) -> None:
    """将与条目形状一致的主机数组按目标分片写入各 DPU。"""
    _check_entry(entry, "scatter")
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    if flat.size != entry.numel:
        raise ValueError(f"scatter 的 host_buf 元素数 {flat.size} != 条目 numel {entry.numel}")
    _writeback(entry, engine, flat)


def broadcast(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray) -> None:
    """将主机数组完整复制到条目列出的各 DPU。"""
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    if flat.size != entry.numel:
        raise ValueError(f"broadcast 的 host_buf 元素数 {flat.size} != 条目 numel {entry.numel}")
    _writeback(entry, engine, flat)
