"""厂商 SDK numpy 镜像（backend/dpu_sdk.py）的逐函数验证。

判据：每个 SDK 函数在 numpy 伪硬件上的行为与 pre-g-driver-api（api/include/dpu.h）
的语义一致——独立地址空间、越界报错、push_xfer 的 prepare/push 两段式、集合语义。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.dpu_sdk import (
    DEFAULT_MRAM_BYTES,
    DPU_ASYNCHRONOUS,
    DPU_SYNCHRONOUS,
    DPU_XFER_FROM_DPU,
    DPU_XFER_TO_DPU,
    DpuError,
    dpu_alloc,
    dpu_broadcast_to,
    dpu_copy_from,
    dpu_copy_to,
    dpu_free,
    dpu_get_nr_dpus,
    dpu_launch,
    dpu_load,
    dpu_log_read,
    dpu_prepare_xfer,
    dpu_push_xfer,
    dpu_status,
    dpu_sync,
)


def test_alloc_enumerates_dpus_and_free_retires_the_set() -> None:
    dpu_set = dpu_alloc(4)
    assert dpu_get_nr_dpus(dpu_set) == 4
    assert dpu_set.dpu_ids == (0, 1, 2, 3)
    assert [subset.dpu_ids for subset in dpu_set] == [(0,), (1,), (2,), (3,)]  # DPU_FOREACH
    dpu_free(dpu_set)
    with pytest.raises(DpuError, match="dpu_free"):
        dpu_get_nr_dpus(dpu_set)
    with pytest.raises(DpuError):
        dpu_alloc(0)


def test_alloc_default_capacity_matches_upmem_mram() -> None:
    assert DEFAULT_MRAM_BYTES == 64 * 2**20
    dpu_set = dpu_alloc(1)
    with pytest.raises(DpuError, match="MRAM 越界"):
        dpu_copy_to(dpu_set, DEFAULT_MRAM_BYTES - 4, np.zeros(8, np.uint8), 8)


def test_copy_to_from_round_trip_and_per_dpu_isolation() -> None:
    dpu_set = dpu_alloc(2, mram_bytes=1024)
    payload = np.arange(16, dtype=np.float32)
    dpu_copy_to(dpu_set.dpu(0), 64, payload, payload.nbytes)
    out = np.empty_like(payload)
    dpu_copy_from(dpu_set.dpu(0), 64, out, out.nbytes)
    assert np.array_equal(out, payload)
    other = np.empty_like(payload)  # 独立地址空间：写 DPU0 不影响 DPU1
    dpu_copy_from(dpu_set.dpu(1), 64, other, other.nbytes)
    assert np.array_equal(other, np.zeros_like(payload))
    with pytest.raises(DpuError, match="MRAM 越界"):
        dpu_copy_from(dpu_set.dpu(0), 1020, np.empty(2, np.float32), 8)
    with pytest.raises(DpuError, match="不符"):
        dpu_copy_to(dpu_set.dpu(0), 0, payload, payload.nbytes - 4)


def test_copy_to_multi_dpu_set_broadcasts_and_copy_from_requires_single() -> None:
    dpu_set = dpu_alloc(3, mram_bytes=256)
    payload = np.arange(4, dtype=np.int32)
    dpu_copy_to(dpu_set, 0, payload, payload.nbytes)  # 多成员集合 = 广播语义
    for dpu_id in range(3):
        out = np.empty_like(payload)
        dpu_copy_from(dpu_set.dpu(dpu_id), 0, out, out.nbytes)
        assert np.array_equal(out, payload)
    with pytest.raises(DpuError, match="单 DPU 集合"):
        dpu_copy_from(dpu_set, 0, np.empty_like(payload), payload.nbytes)


def test_push_xfer_fans_out_prepared_buffers_in_both_directions() -> None:
    dpu_set = dpu_alloc(4, mram_bytes=256)
    per_dpu = [np.full(8, fill_value=i, dtype=np.int32) for i in range(4)]
    for subset, buf in zip(dpu_set, per_dpu):  # DPU_FOREACH 内逐台 prepare
        dpu_prepare_xfer(subset, buf)
    dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 32, per_dpu[0].nbytes)

    backs = [np.empty(8, dtype=np.int32) for _ in range(4)]
    for subset, buf in zip(dpu_set, backs):
        dpu_prepare_xfer(subset, buf)
    dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, 32, backs[0].nbytes)
    for dpu_id, buf in enumerate(backs):
        assert np.array_equal(buf, per_dpu[dpu_id])


def test_push_xfer_without_prepare_or_with_short_buffer_raises() -> None:
    dpu_set = dpu_alloc(2, mram_bytes=256)
    with pytest.raises(DpuError, match="未 dpu_prepare_xfer"):
        dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 0, 8)
    dpu_prepare_xfer(dpu_set.dpu(0), np.empty(2, np.int32))
    dpu_prepare_xfer(dpu_set.dpu(1), np.empty(1, np.int32))
    with pytest.raises(DpuError, match="小于 length"):
        dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, 0, 8)


def test_broadcast_to_writes_one_buffer_to_every_dpu() -> None:
    dpu_set = dpu_alloc(2, mram_bytes=128)
    payload = np.arange(8, dtype=np.float16)
    dpu_broadcast_to(dpu_set, 16, payload, payload.nbytes)
    for subset in dpu_set:
        out = np.empty_like(payload)
        dpu_copy_from(subset, 16, out, out.nbytes)
        assert np.array_equal(out, payload)


def test_load_launch_sync_status_with_python_kernel() -> None:
    dpu_set = dpu_alloc(2, mram_bytes=128)

    def kernel(dpu_id: int, mram: np.ndarray) -> None:
        mram[0:4] = np.asarray([dpu_id + 1], dtype=np.int32).view(np.uint8)

    dpu_load(dpu_set, kernel)
    dpu_launch(dpu_set, DPU_SYNCHRONOUS)
    dpu_sync(dpu_set)
    assert dpu_status(dpu_set) == (True, False)
    for dpu_id in range(2):
        out = np.empty(1, dtype=np.int32)
        dpu_copy_from(dpu_set.dpu(dpu_id), 0, out, 4)
        assert out[0] == dpu_id + 1

    dpu_load(dpu_set.dpu(0), b"\x00\x01")  # 二进制仅登记，镜像不可执行
    with pytest.raises(DpuError, match="无法执行 kernel 二进制"):
        dpu_launch(dpu_set.dpu(0), DPU_ASYNCHRONOUS)
    with pytest.raises(DpuError, match="未 dpu_load"):
        dpu_launch(dpu_alloc(1, mram_bytes=64), DPU_SYNCHRONOUS)


def test_log_read_and_no_direct_dpu_to_dpu_primitive() -> None:
    dpu_set = dpu_alloc(1, mram_bytes=64)
    stream = io.StringIO()
    dpu_log_read(dpu_set, stream)
    assert "DPU0" in stream.getvalue() or "DpuSet" in stream.getvalue()
    # host-star 拓扑：SDK 不提供任何 dpu→dpu 直达传输原语
    import backend.dpu_sdk as sdk

    assert not any(
        name.startswith("dpu_copy_dpu") or "dpu_to_dpu" in name for name in dir(sdk)
    )
