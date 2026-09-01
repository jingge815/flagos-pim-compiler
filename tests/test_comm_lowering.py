"""验证通信原语在 SDK NumPy 镜像上的数值结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.dpu_sdk import dpu_alloc
from comm.lowering import DmaEngine, all_gather, all_reduce, all_to_all, broadcast, scatter
from comm.plan import build_comm_plan
from contracts.pim_tensor_spec import Placement
from tests.test_comm_plan import PARTIAL_SUM, REPLICATE, _dpu_spec, _edge, _host_spec


def _engine(num_dpus: int = 2) -> DmaEngine:
    return DmaEngine(dpu_alloc(num_dpus, mram_bytes=2**20))


def _read(engine: DmaEngine, dpu_id: int, length: int, dtype=np.float32, addr: int = 0) -> np.ndarray:
    return engine.copy_from_dpu(dpu_id, addr, length, dtype)


def test_all_reduce_sums_partials_and_writes_back() -> None:
    shape = (8,)
    src = _dpu_spec(PARTIAL_SUM, shape, (0, 1))
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "all_reduce", src, dst, shape, reduce_type="sum")])
    engine = _engine()
    partials = [np.arange(8, dtype=np.float32) + i * 10 for i in (0, 1)]
    for dpu_id, part in enumerate(partials):
        engine.copy_to_dpu(dpu_id, 0, part)

    acc = all_reduce(entry, engine)

    assert np.array_equal(acc, sum(partials).reshape(shape))
    for dpu_id in (0, 1):  # 向每个目标 DPU 回写结果。
        assert np.array_equal(_read(engine, dpu_id, 8), sum(partials))


def test_all_reduce_to_host_leaves_dpus_untouched_and_supports_mean() -> None:
    shape = (4,)
    src = _dpu_spec(Placement("Partial", reduce_type="mean"), shape, (0, 1))
    (entry,) = build_comm_plan(
        [_edge(0, "all_reduce", src, _host_spec(), shape, reduce_type="mean")]
    )
    engine = _engine()
    partials = [np.ones(4, dtype=np.float32) * (i + 1) for i in (0, 1)]
    for dpu_id, part in enumerate(partials):
        engine.copy_to_dpu(dpu_id, 0, part)

    acc = all_reduce(entry, engine)

    assert np.array_equal(acc, (partials[0] + partials[1]) / 2)
    for dpu_id, part in enumerate(partials):  # 主机目标不改变 DPU 内容。
        assert np.array_equal(_read(engine, dpu_id, 4), part)


def test_all_gather_orders_shards_by_global_range_not_dpu_id() -> None:
    """验证分片按全局位置而非 DPU 编号拼接。"""
    shape = (6,)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1), permuted=True)
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "all_gather", src, dst, shape)])
    engine = _engine()
    global_tensor = np.arange(6, dtype=np.float32)
    engine.copy_to_dpu(0, 0, global_tensor[3:6])
    engine.copy_to_dpu(1, 0, global_tensor[0:3])

    merged = all_gather(entry, engine)

    assert np.array_equal(merged, global_tensor)
    for dpu_id in (0, 1):  # 每个 DPU 接收完整张量。
        assert np.array_equal(_read(engine, dpu_id, 6), global_tensor)


def test_all_gather_multidim_shard_interleaves_rows() -> None:
    """验证多维分片按外维连续区间合并。"""
    shape = (1, 4, 8)
    src = _dpu_spec(Placement("Shard", 2), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "all_gather", src, _host_spec(), shape)])
    engine = _engine()
    global_tensor = np.arange(32, dtype=np.float32).reshape(shape)
    engine.copy_to_dpu(0, 0, np.ascontiguousarray(global_tensor[:, :, 0:4]).reshape(-1))
    engine.copy_to_dpu(1, 0, np.ascontiguousarray(global_tensor[:, :, 4:8]).reshape(-1))

    merged = all_gather(entry, engine)

    assert np.array_equal(merged, global_tensor)
    assert np.array_equal(_read(engine, 0, 16), global_tensor[:, :, 0:4].reshape(-1))


def test_all_to_all_reshards_via_host_staging() -> None:
    shape = (4, 6)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    dst = _dpu_spec(Placement("Shard", 1), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "all_to_all", src, dst, shape)])
    engine = _engine()
    global_tensor = np.arange(24, dtype=np.float32).reshape(shape)
    engine.copy_to_dpu(0, 0, global_tensor[0:2, :].reshape(-1))
    engine.copy_to_dpu(1, 0, global_tensor[2:4, :].reshape(-1))

    all_to_all(entry, engine)

    for dpu_id, cols in ((0, slice(0, 3)), (1, slice(3, 6))):  # Shard(1) 分片
        assert np.array_equal(
            _read(engine, dpu_id, 12), np.ascontiguousarray(global_tensor[:, cols]).reshape(-1)
        )


def test_scatter_delivers_per_dst_slices() -> None:
    shape = (4, 6)
    dst = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "scatter", _host_spec(), dst, shape)])
    engine = _engine()
    global_tensor = np.arange(24, dtype=np.float32).reshape(shape)

    scatter(entry, engine, global_tensor)

    assert np.array_equal(_read(engine, 0, 12), global_tensor[0:2, :].reshape(-1))
    assert np.array_equal(_read(engine, 1, 12), global_tensor[2:4, :].reshape(-1))


def test_broadcast_copies_one_buffer_to_every_dst() -> None:
    shape = (4,)
    dst = _dpu_spec(REPLICATE, shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "scatter", _host_spec(), dst, shape)])
    engine = _engine()
    payload = np.arange(4, dtype=np.float32)

    broadcast(entry, engine, payload)

    for dpu_id in (0, 1):
        assert np.array_equal(_read(engine, dpu_id, 4), payload)


def test_primitive_rejects_mismatched_entry_type() -> None:
    shape = (4,)
    src = _dpu_spec(Placement("Shard", 0), shape, (0, 1))
    (entry,) = build_comm_plan([_edge(0, "all_gather", src, _host_spec(), shape)])
    with pytest.raises(ValueError, match="all_reduce 收到"):
        all_reduce(entry, _engine())


def test_megatron_pair_end_to_end_matches_torch() -> None:
    """验证列切、行切和 all_reduce 的端到端结果。"""
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    w1 = torch.randn(6, 4)  # HF [out, in]，列切 Shard(0)
    w2 = torch.randn(4, 6)  # 行切 Shard(1)
    ref = torch.nn.functional.linear(torch.nn.functional.linear(x, w1), w2)

    engine = _engine()
    x_spec = _dpu_spec(REPLICATE, (1, 4), (0, 1))
    (scatter_entry,) = build_comm_plan([_edge(0, "scatter", _host_spec(), x_spec, (1, 4))])
    scatter(scatter_entry, engine, x.numpy())

    for dpu_id in (0, 1):
        local_x = _read(engine, dpu_id, 4).reshape(1, 4)
        w1_d = w1[dpu_id * 3 : (dpu_id + 1) * 3, :].numpy()  # 列切
        w2_d = w2[:, dpu_id * 3 : (dpu_id + 1) * 3].numpy()  # 行切
        partial = (local_x @ w1_d.T) @ w2_d.T
        engine.copy_to_dpu(dpu_id, 64, partial.reshape(-1))

    src = _dpu_spec(PARTIAL_SUM, (1, 4), (0, 1), mram_offset=64)  # 部分和地址非 0：addr 随段走
    (reduce_entry,) = build_comm_plan(
        [_edge(1, "all_reduce", src, _host_spec(), (1, 4), reduce_type="sum")]
    )
    acc = all_reduce(reduce_entry, engine)

    torch.testing.assert_close(torch.from_numpy(acc), ref, rtol=1e-5, atol=1e-6)
