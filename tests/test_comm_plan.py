"""验证通信计划条目、DMA 序列和传输成本。"""

from __future__ import annotations

import sys
from math import prod
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from comm.plan import (
    CommPlanEntry,
    HostStarCostModel,
    build_comm_plan,
    dma_sequence,
    format_comm_plan,
    plan_cost,
)
from contracts.pim_tensor_spec import (
    PIMTensorSpec,
    Placement,
    RedistributeEdge,
    TensorShardDetail,
)

REPLICATE = Placement("Replicate")
PARTIAL_SUM = Placement("Partial", reduce_type="sum")


def _dpu_spec(
    placement: Placement,
    shape: tuple[int, ...],
    dpu_ids: tuple[int, ...],
    *,
    permuted: bool = False,
    mram_offset: int = 0,
) -> PIMTensorSpec:
    """构造均匀单段分片规格，可选反转 DPU 到分片的映射。"""
    if placement.kind == "Shard":
        dim = placement.dim
        width = shape[dim] // len(dpu_ids)
        shard_map = {}
        for i, dpu_id in enumerate(dpu_ids):
            chunk = len(dpu_ids) - 1 - i if permuted else i
            shard_map[dpu_id] = TensorShardDetail(
                dpu_id, dim, chunk * width, (chunk + 1) * width,
                shape[:dim] + (width,) + shape[dim + 1 :], mram_offset,
            )
    else:
        numel = prod(shape)
        shard_map = {
            dpu_id: TensorShardDetail(dpu_id, -1, 0, numel, shape, mram_offset)
            for dpu_id in dpu_ids
        }
    spec = PIMTensorSpec("dpu", placement, "transient", None, shard_map, placement.reduce_type)
    spec.validate()
    return spec


def _host_spec() -> PIMTensorSpec:
    spec = PIMTensorSpec("host", REPLICATE, "transient", None, {}, None)
    spec.validate()
    return spec


def _loc_of(spec: PIMTensorSpec) -> dict:
    if spec.device == "host":
        return {"device": "host"}
    return {"device": "dpu", "dpus": sorted(spec.shard_map)}


def _edge(
    edge_id: int,
    type_: str,
    src_spec: PIMTensorSpec,
    dst_spec: PIMTensorSpec,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
    reduce_type: str | None = None,
) -> RedistributeEdge:
    return RedistributeEdge(
        edge_id=edge_id,
        src="producer",
        dst="consumer",
        from_placement=src_spec.placement,
        to_placement=dst_spec.placement,
        src_spec=src_spec,
        dst_spec=dst_spec,
        type=type_,
        src_loc=_loc_of(src_spec),
        dst_loc=_loc_of(dst_spec),
        nbytes=prod(shape) * np.dtype(dtype).itemsize,
        reduce_type=reduce_type,
        shape=shape,
        dtype=dtype,
    )


# Partial 到主机 Replicate。


def test_all_reduce_segments_match_appendix_b1() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1))
    edge = _edge(42, "all_reduce", src, _host_spec(), shape, reduce_type="sum")

    (entry,) = build_comm_plan([edge])

    assert entry.type == "all_reduce" and entry.reduce == "sum"
    assert entry.wait_for == (0, 1)  # 边级读前等待 = 全体源 DPU
    assert entry.dst_loc == {"device": "host"}
    assert len(entry.segments) == 2  # dst_loc 为 host：无回写段
    for seg, dpu_id in zip(entry.segments, (0, 1)):
        assert seg.edge_id == 42 and seg.type == "all_reduce" and seg.reduce == "sum"
        assert (seg.src_dpu, seg.src_local_range) == (dpu_id, (0, 8))
        assert seg.global_range == (0, 8)
        assert seg.dst_dpu is None and seg.dst_local_offset == 0
        assert seg.nbytes == 32
        assert seg.dst_ready_after == ()


def test_all_reduce_writeback_only_to_dst_loc_dpus() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1))
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(1, "all_reduce", src, dst, shape, reduce_type="sum")])

    writeback = entry.writeback_segments
    assert len(writeback) == 2
    for seg, dpu_id in zip(writeback, (0, 1)):
        assert seg.src_dpu is None and seg.src_local_range == (0, 8)  # host 归约缓冲
        assert (seg.dst_dpu, seg.dst_local_offset) == (dpu_id, 0)
        assert seg.global_range == (0, 8) and seg.nbytes == 32


# Shard 到 Replicate。


def test_all_gather_collect_segments_carry_global_position() -> None:
    shape = (6,)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1), permuted=True)  # DPU0 持后半段
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(43, "all_gather", src, dst, shape)])

    collect = sorted(entry.collect_segments, key=lambda s: s.global_range[0])
    assert [(s.src_dpu, s.global_range, s.src_local_range, s.dst_local_offset) for s in collect] == [
        (1, (0, 3), (0, 3), 0),  # DPU1 持全局前半段
        (0, (3, 6), (0, 3), 3),  # DPU0 持全局后半段
    ]
    broadcast = entry.writeback_segments
    assert [(s.src_dpu, s.global_range, s.dst_dpu, s.dst_local_offset) for s in broadcast] == [
        (None, (0, 6), 0, 0),
        (None, (0, 6), 1, 0),
    ]


def test_all_gather_multidim_shard_unfolds_to_per_row_runs() -> None:
    """[1,4,8] 沿 dim2 切 2 台：每台 4 段连续 run，全局缓冲中交错分布。"""
    shape = (1, 4, 8)
    src = _dpu_spec(Placement("Shard", 2), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(2, "all_gather", src, _host_spec(), shape)])

    collect = entry.collect_segments
    assert len(collect) == 8  # 每台 prod(shape[:2])=4 段
    by_dpu = {0: [], 1: []}
    for seg in collect:
        by_dpu[seg.src_dpu].append((seg.global_range, seg.src_local_range, seg.dst_local_offset))
    assert by_dpu[0] == [
        ((0, 4), (0, 4), 0),
        ((8, 12), (4, 8), 8),
        ((16, 20), (8, 12), 16),
        ((24, 28), (12, 16), 24),
    ]
    assert by_dpu[1][0] == ((4, 8), (0, 4), 4)  # DPU1 从全局偏移 4 起交错
    assert all(seg.nbytes == 16 for seg in collect)
    assert entry.writeback_segments == []  # dst host：无广播段


def test_all_gather_from_replicate_source_collects_only_one_copy() -> None:
    """复制布局到主机时只收集一份数据。"""
    shape = (4,)
    src = _dpu_spec(REPLICATE, shape, (0, 1, 2, 3))
    (entry,) = build_comm_plan([_edge(3, "all_gather", src, _host_spec(), shape)])

    assert len(entry.segments) == 1
    seg = entry.segments[0]
    assert seg.src_dpu == 0 and seg.global_range == (0, 4) and seg.dst_dpu is None


# Shard 到 Shard 的区间交集。


def test_all_to_all_segments_are_run_intersections() -> None:
    shape = (4, 6)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1))  # 每台持 2 行 = 12 元素一段
    dst = _dpu_spec(Placement("Shard", 1), shape, (0, 1))  # 每台持 3 列 × 4 行 = 4 段
    (entry,) = build_comm_plan([_edge(4, "all_to_all", src, dst, shape)])

    assert len(entry.segments) == 8  # 2×4 个非空交集
    seg00 = next(s for s in entry.segments if s.src_dpu == 0 and s.dst_dpu == 1)
    assert seg00.global_range == (3, 6)  # DPU0 前 2 行的后 3 列 → 目标 DPU1
    assert seg00.src_local_range == (3, 6) and seg00.dst_local_offset == 0
    assert seg00.nbytes == 12
    # 一个源分片可对应多个目标分片。
    assert sum(1 for s in entry.segments if s.src_dpu == 0) == 4
    targets_of_src0 = {s.dst_dpu for s in entry.segments if s.src_dpu == 0}
    assert targets_of_src0 == {0, 1}


# scatter 与 local_slice。


def test_scatter_segments_slice_host_tensor_per_dst_shard() -> None:
    shape = (4, 6)
    dst = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(5, "scatter", _host_spec(), dst, shape)])

    assert entry.wait_for == ()  # 源在 host，无 DPU 生产者
    assert [(s.src_dpu, s.global_range, s.dst_dpu, s.dst_local_offset) for s in entry.segments] == [
        (None, (0, 12), 0, 0),
        (None, (12, 24), 1, 0),
    ]


def test_scatter_to_replicate_degenerates_to_broadcast() -> None:
    """主机到复制布局使用一次广播写入全部 DPU。"""
    shape = (4,)
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(6, "scatter", _host_spec(), dst, shape)])

    assert [(s.global_range, s.dst_dpu) for s in entry.segments] == [((0, 4), 0), ((0, 4), 1)]


def test_local_slice_keeps_empty_placeholder_entry() -> None:
    shape = (4,)
    src = _dpu_spec(REPLICATE, shape, (0, 1))
    dst = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(7, "local_slice", src, dst, shape)])

    assert entry.type == "local_slice" and entry.segments == []
    assert dma_sequence(entry) == []


# DMA 序列和接口成本。


def test_dma_sequence_batches_uniform_transfers() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1, 2, 3))
    dst = _dpu_spec(REPLICATE, shape, (0, 1, 2, 3))
    (entry,) = build_comm_plan([_edge(8, "all_reduce", src, dst, shape, reduce_type="sum")])

    ops = dma_sequence(entry)
    assert [(op.kind, len(op.segments)) for op in ops] == [("push_from", 4), ("broadcast_to", 4)]


def test_dma_sequence_multidim_gather_batches_per_row() -> None:
    shape = (1, 4, 8)
    src = _dpu_spec(Placement("Shard", 2), shape, (0, 1))
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(9, "all_gather", src, dst, shape)])

    ops = dma_sequence(entry)
    # 相同地址的段合并为批量传输，复制结果使用广播。
    assert [(op.kind, len(op.segments)) for op in ops] == [
        ("push_from", 2), ("push_from", 2), ("push_from", 2), ("push_from", 2),
        ("broadcast_to", 2),
    ]


def test_plan_cost_counts_batched_transfers_and_bytes() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1))
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(10, "all_reduce", src, dst, shape, reduce_type="sum")])

    model = HostStarCostModel(dma_setup_s=1e-5, host_bytes_per_s=25e9)
    cost = plan_cost(entry, model)
    assert model.topology == "host_star"
    assert cost.transfers == 2  # 1 次 push_from + 1 次 broadcast_to
    assert cost.nbytes == 4 * 32  # 收集 2×32 + 回写 2×32
    assert cost.seconds == pytest.approx(2 * 1e-5 + 128 / 25e9)


def test_coverage_check_catches_gaps_and_dtype_mismatch() -> None:
    shape = (8,)
    broken = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    broken.shard_map[1] = TensorShardDetail(1, 0, 5, 8, (3,))  # 全局区间存在缺口。
    with pytest.raises(ValueError, match="断裂/重叠"):
        build_comm_plan([_edge(11, "all_gather", broken, _host_spec(), shape)])
    good = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    edge = RedistributeEdge(
        edge_id=12, src="p", dst="c", from_placement=good.placement, to_placement=REPLICATE,
        src_spec=good, dst_spec=_host_spec(), type="all_gather",
        src_loc=_loc_of(good), dst_loc={"device": "host"},
        nbytes=999, shape=shape, dtype="float32",
    )
    with pytest.raises(ValueError, match="nbytes"):
        build_comm_plan([edge])


def test_plan_rejects_endpoint_locations_inconsistent_with_specs() -> None:
    """验证端点位置必须与张量规格一致。"""
    shape = (8,)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    edge = _edge(14, "all_gather", src, _host_spec(), shape)
    edge = RedistributeEdge(
        **{**edge.__dict__, "src_loc": {"device": "dpu", "dpus": [1, 2]}}
    )

    with pytest.raises(ValueError, match="src_loc"):
        build_comm_plan([edge])


def test_format_comm_plan_printable() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1))
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    entries = build_comm_plan([_edge(13, "all_reduce", src, dst, shape, reduce_type="sum")])
    report = format_comm_plan(entries)
    assert "e13: all_reduce" in report and "src_dpu=0" in report and "分类统计" in report
