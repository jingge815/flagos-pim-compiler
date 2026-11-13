"""验证通信段按实际行数裁剪（D2-full 第一步）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from comm.lowering import DmaEngine, broadcast, clamped_entry, scatter
from comm.plan import CommPlanEntry, CommSegment


class _RecordingEngine(DmaEngine):
    """记录 copy_to_dpu 调用的假引擎。"""

    def __init__(self):
        super().__init__(dpu_set=None)
        self.calls: list[tuple[int, int, bytes]] = []

    def copy_to_dpu(self, dpu_id, addr, data):
        self.calls.append((dpu_id, addr, np.asarray(data).tobytes()))

    def run_writeback(self, op, host_buf):
        for seg in op.segments:
            self.copy_to_dpu(
                seg.dst_dpu, seg.dst_addr,
                host_buf[seg.global_range[0]:seg.global_range[1]])


def _col_shard_entry() -> CommPlanEntry:
    """(1,4,8) 按末维切两半：每 DPU 每行一段 4 元素的列切片。"""
    segs = []
    for row in range(4):
        base = row * 8
        segs.append(CommSegment(
            edge_id=0, type="scatter", src_dpu=None,
            src_local_range=(0, 0), global_range=(base, base + 4),
            dst_dpu=0, dst_local_offset=row * 4, nbytes=8,
            src_addr=0, dst_addr=0x1000))
        segs.append(CommSegment(
            edge_id=0, type="scatter", src_dpu=None,
            src_local_range=(0, 0), global_range=(base + 4, base + 8),
            dst_dpu=1, dst_local_offset=row * 4, nbytes=8,
            src_addr=0, dst_addr=0x2000))
    return CommPlanEntry(
        edge_id=0, type="scatter", reduce=None, shape=(1, 4, 8),
        dtype=np.dtype(np.float16), dst_loc={"device": "dpu", "dpus": [0, 1]},
        wait_for=(), segments=segs)


def test_clamped_entry_drops_segments_beyond_actual_rows() -> None:
    entry = _col_shard_entry()
    assert len(entry.segments) == 8
    clamped = clamped_entry(entry, rows=2)
    assert len(clamped.segments) == 4
    assert all(s.global_range[0] // 8 < 2 for s in clamped.segments)
    # rows=None 或超出包络行数时原样返回。
    assert clamped_entry(entry, None) is entry
    assert len(clamped_entry(entry, rows=9).segments) == 8


def test_scatter_rows_only_writes_actual_rows() -> None:
    entry = _col_shard_entry()
    host = np.arange(32, dtype=np.float16)
    engine = _RecordingEngine()
    scatter(entry, engine, host, rows=2)
    # 只应有前两行：每行两段（dpu0/dpu1 各 4 元素）。
    assert len(engine.calls) == 4
    values = sorted(np.frombuffer(data, dtype=np.float16).tolist()
                    for _d, _a, data in engine.calls)
    assert values[0] == [0.0, 1.0, 2.0, 3.0]      # row0 左半
    assert values[-1] == [12.0, 13.0, 14.0, 15.0]  # row1 右半
    # 缺省行为不变：全部 8 段。
    engine2 = _RecordingEngine()
    scatter(entry, engine2, host)
    assert len(engine2.calls) == 8


def test_broadcast_rows_passes_through() -> None:
    entry = _col_shard_entry()
    host = np.ones(32, dtype=np.float16)
    engine = _RecordingEngine()
    broadcast(entry, engine, host, rows=1)
    assert len(engine.calls) == 2  # 每行两段，rows=1 只留第 0 行


def test_clamped_entry_truncates_segments_spanning_multiple_rows() -> None:
    """跨多行的整段必须截到 rows*R，尾行旧值不能进 host 缓冲。"""
    seg = CommSegment(
        edge_id=0, type="all_gather", src_dpu=0,
        src_local_range=(0, 8), global_range=(0, 8),
        dst_dpu=None, dst_local_offset=0, nbytes=16, src_addr=0)
    entry = CommPlanEntry(
        edge_id=0, type="all_gather", reduce=None, shape=(1, 4, 2),
        dtype=np.dtype(np.float16), dst_loc={"device": "host"},
        wait_for=(), segments=[seg])
    clamped = clamped_entry(entry, rows=2)
    assert len(clamped.segments) == 1
    assert clamped.segments[0].global_range == (0, 4)
    assert clamped.segments[0].nbytes == 8
    assert clamped_entry(entry, None) is entry
