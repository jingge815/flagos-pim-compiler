"""问题 3（运行时半）：通信库——把通信计划表条目展开为 host 中转的 DMA 序列。

两层结构（方案问题 3 二.(3)）：

- 第二组（底层）：``DmaEngine``，架在 backend/dpu_sdk（厂商 SDK 镜像）之上的
  DMA 封装，只用 dpu_copy_to / dpu_copy_from / dpu_push_xfer / dpu_broadcast_to
  四件套，host 端归约/拼接为纯 numpy 操作；
- 第一组（对外原语）：``all_reduce`` / ``all_gather`` / ``all_to_all`` /
  ``scatter`` / ``broadcast``，一一对应 redistribute 类型，统一接收
  comm/plan.py 的 CommPlanEntry，按 dma_sequence 展开执行。

通信库内部不做任何等待：wait_for / dst_ready_after 已由问题 6 展开进
Command.waits，拿到本层时每段的前驱均已完成（方案三.(1)）。拓扑为 host-star：
DPU 间无直连，一切跨 DPU 交换经 host 两跳，SDK 不提供 dpu→dpu 直达原语。

收集方向每个段读回一块独立 host 缓冲（各 DPU 部分和的区间互相重叠，不能落同一
缓冲，方案一. 的 DMA 序列示例）；归约/拼接/重排由原语在这些缓冲与全局摊平缓冲
之间完成；回写方向从全局摊平缓冲按段的 global_range 取出写向目标 DPU。
"""

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
    """底层 DMA 封装（方案二.(3) 第二组）：点对点 copy_* + 批量 push_xfer/broadcast。"""

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
    """回写阶段：dst_loc 为 host 时无回写传输，自然空转（方案二.(4)）。"""
    for op in dma_sequence(entry):
        if op.kind in ("copy_to", "push_to", "broadcast_to"):
            engine.run_writeback(op, host_buf)


def _merged(entry: CommPlanEntry, collected: list[tuple[CommSegment, np.ndarray]]) -> np.ndarray:
    """把逐段收集缓冲按 global_range 落位为全局摊平缓冲（即拼接/重排的实现）。

    落位依据段自带的全局偏移，与 DPU 编号顺序无关（附录 B.4）。
    """
    host_buf = np.zeros(entry.numel, dtype=entry.dtype)
    for seg, buf in sorted(collected, key=lambda item: item[0].global_range[0]):
        host_buf[seg.global_range[0] : seg.global_range[1]] = buf
    return host_buf


# ---------------------------------------------------------------------------
# 第一组：通信原语（方案二.(3) 表；三段循环"收集 → host 归约 → 回写"）
# ---------------------------------------------------------------------------


def all_reduce(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """Partial → Replicate：各 DPU 部分和收 host → 累加 → 回写 dst_loc 列出的 DPU。

    出: 全局形状的归约结果（host 缓冲）。dst_loc 为 host 时无回写段，结果只落
    host，由调用方（下游 host 节点）直接读取（方案二.(4)）。
    """
    _check_entry(entry, "all_reduce")
    acc = np.zeros(entry.numel, dtype=entry.dtype)
    for seg, buf in _collect(entry, engine):
        acc[seg.global_range[0] : seg.global_range[1]] += buf
    if entry.reduce == "mean":
        acc /= len(entry.collect_segments)
    _writeback(entry, engine, acc)
    return acc.reshape(entry.shape)


def all_gather(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """Shard → Replicate：各 DPU 分片收 host → 按全局偏移拼接 → 广播回各目标 DPU。

    dst_loc 为 host 时无广播段，合并结果只落 host（方案二.(4)）。
    """
    _check_entry(entry, "all_gather")
    merged = _merged(entry, _collect(entry, engine))
    _writeback(entry, engine, merged)
    return merged.reshape(entry.shape)


def all_to_all(entry: CommPlanEntry, engine: DmaEngine) -> np.ndarray:
    """Shard(i) → Shard(j)：全部收 host → 按段交集原位重排 → 重新分发。

    段的 src_dpu/dst_dpu 均非 None；host 中转缓冲即全局摊平布局，收集按
    global_range 落位后，回写按同一 global_range 取出即完成重排。
    """
    _check_entry(entry, "all_to_all")
    staging = _merged(entry, _collect(entry, engine))
    _writeback(entry, engine, staging)
    return staging.reshape(entry.shape)


def scatter(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray) -> None:
    """Replicate → Shard：host 完整张量按目标 shard_map 切片，逐片下发各 DPU。

    host_buf 为全局逻辑张量（形状须与条目一致）；目标为 Replicate 时退化为
    broadcast（每 DPU 收全量，方案二.(9)）。
    """
    _check_entry(entry, "scatter")
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    if flat.size != entry.numel:
        raise ValueError(f"scatter 的 host_buf 元素数 {flat.size} != 条目 numel {entry.numel}")
    _writeback(entry, engine, flat)


def broadcast(entry: CommPlanEntry, engine: DmaEngine, host_buf: np.ndarray) -> None:
    """host 一份数据复制到条目回写段列出的每个 DPU（scatter 退化 / gather 回写段）。"""
    flat = np.ascontiguousarray(host_buf, dtype=entry.dtype).reshape(-1)
    if flat.size != entry.numel:
        raise ValueError(f"broadcast 的 host_buf 元素数 {flat.size} != 条目 numel {entry.numel}")
    _writeback(entry, engine, flat)
