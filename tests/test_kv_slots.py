"""验证固定大小槽的几何、全局块表间接寻址和槽池容量。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from memory.kv_layout import KVRegionSpec, build_kv_layout, write_block_table
from runtime.executor import KVBlockTable, KVSlotPool, RequestSlot
from tests.kv_helpers import read_rows, write_row

HEAD_DIM = 4
DTYPE_BYTES = 4
ALIGN = 64
MAX_SEQ = 6
SLOT_TOKENS = 2


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=1, mram_bytes_per_dpu=2**16))


def _spec(kv_base: int = 1024, pool_slots: int | None = None) -> KVRegionSpec:
    spec = KVRegionSpec(
        dpu_id=0,
        layers=[0],
        kv_heads=[0],
        q_heads_by_kv={0: [0]},
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype_bytes=DTYPE_BYTES,
        kv_base=kv_base,
        slot_tokens=SLOT_TOKENS,
    )
    return build_kv_layout(spec, ALIGN, pool_slots=pool_slots)


def test_slot_geometry_exact() -> None:
    """槽、块表缓冲和槽池的几何按公式精确计算。"""
    spec = _spec(kv_base=1024, pool_slots=6)
    assert spec.slot_tokens == 2
    assert spec.slot_bytes == 2 * 1 * 1 * 64 == 128
    assert spec.blocks_per_request == 3
    assert spec.pool_slots == 6
    assert spec.block_table_base == 1024
    assert spec.block_table_bytes == 64
    assert spec.slots_base == 1024 + 64 == 1088
    assert spec.kv_allocated_bytes == 64 + 6 * 128 == 832
    assert spec.kv_off == {(0, 0, "k"): 1088, (0, 0, "v"): 1152}


def test_default_pool_fits_one_full_length_request() -> None:
    """不传 pool_slots 时池大小取满长单请求的块数。"""
    spec = _spec()
    assert spec.blocks_per_request == 3
    assert spec.pool_slots == 3
    assert spec.kv_allocated_bytes == 64 + 3 * 128 == 448


def test_indirect_addressing_follows_block_table() -> None:
    """块表把逻辑块映射到任意物理槽，按逻辑顺序 gather 与物理布局无关。"""
    hal = _backend()
    spec = _spec()
    block_table = [2, 0, 1]  # 逻辑块 0/1/2 -> 物理槽 2/0/1
    write_block_table(spec, hal, block_table)

    expected = []
    for position in range(MAX_SEQ):
        row = np.arange(HEAD_DIM, dtype=np.float32) + position * 10
        expected.append(row)
        physical_slot = block_table[position // SLOT_TOKENS]
        write_row(hal, spec, physical_slot, layer=0, head=0, which="k",
                  row_index=position % SLOT_TOKENS, row=row)

    got = read_rows(hal, spec, layer=0, head=0, which="k", n_tokens=MAX_SEQ)
    assert np.array_equal(got, np.stack(expected))

    # 物理槽 2 收到的是逻辑块 0（前两个 token），不是物理顺序的第一块。
    physical_two = hal.read_local(
        0, spec.kv_off[(0, 0, "k")] + 2 * spec.slot_bytes,
        (SLOT_TOKENS, HEAD_DIM), np.float32)
    assert np.array_equal(physical_two, np.stack(expected[:SLOT_TOKENS]))


def test_short_request_clears_tail_of_long_block_table() -> None:
    """长请求后写入短请求的块表，表尾必须整段补 -1，不能继承旧物理槽。"""
    hal = _backend()
    spec = _spec(pool_slots=6)
    write_block_table(spec, hal, [2, 0, 1])
    write_block_table(spec, hal, [4])

    entries = hal.read_local(0, spec.block_table_base,
                             (spec.blocks_per_request,), np.int32)
    assert entries.tolist() == [4, -1, -1]

    write_row(hal, spec, 4, layer=0, head=0, which="k", row_index=0,
              row=np.full(HEAD_DIM, 9.0))
    assert np.array_equal(read_rows(hal, spec, layer=0, head=0, which="k",
                                    n_tokens=1)[0], np.full(HEAD_DIM, 9.0))


def test_block_table_rejects_out_of_pool_and_too_long() -> None:
    """块号必须落在池内；块表不能超过缓冲表项数。"""
    hal = _backend()
    spec = _spec(pool_slots=6)
    with pytest.raises(ValueError, match="越界"):
        write_block_table(spec, hal, [spec.pool_slots])
    with pytest.raises(ValueError, match="超过"):
        write_block_table(spec, hal, [0] * (spec.blocks_per_request + 1))


def test_pool_sized_for_one_request_exhausts_on_second_long_request() -> None:
    """池只够一个满长请求时，第二个请求申请同样多块会耗尽。"""
    spec = _spec()
    assert spec.pool_slots == spec.blocks_per_request
    pool = KVSlotPool(spec.pool_slots)
    first = KVBlockTable(pool)
    first.ensure(spec.blocks_per_request)
    assert first.slots == [0, 1, 2]

    second = KVBlockTable(pool)
    with pytest.raises(RuntimeError, match="耗尽"):
        second.ensure(spec.blocks_per_request)


def test_short_prompt_allocates_only_needed_blocks() -> None:
    """短 prompt 只申请 ceil(n/slot_tokens) 个槽，槽池仍有余额。"""
    pool = KVSlotPool(4)
    blocks = KVBlockTable(pool)
    blocks.ensure(-(-3 // SLOT_TOKENS))
    assert blocks.slots == [0, 1]
    blocks.ensure(2)  # 已经够用，不重复申请
    assert blocks.slots == [0, 1]

    other = KVBlockTable(pool)
    other.ensure(2)  # 池里还剩 2 个槽，第二个请求能申请到
    assert other.slots == [2, 3]


def test_pool_exhaustion_and_reset_reuse() -> None:
    """池耗尽抛错；释放后槽可复用。"""
    pool = KVSlotPool(1)
    first = KVBlockTable(pool)
    first.ensure(1)
    second = KVBlockTable(pool)
    with pytest.raises(RuntimeError, match="耗尽"):
        second.ensure(1)

    first.release()
    second.ensure(1)
    assert second.slots == [0]

    slot = RequestSlot(request_id="r", request_slot=0, blocks=KVBlockTable(KVSlotPool(2)))
    slot.blocks.ensure(2)
    slot.reset([1, 2, 3])
    assert slot.blocks.slots == []
    slot.blocks.ensure(1)  # reset 归还的槽能再次申请
    assert slot.blocks.slots


def test_rewriting_block_table_switches_target_request() -> None:
    """设备侧只有一份块表缓冲：重写块表即切换读到的请求，物理槽互不相交。"""
    hal = _backend()
    spec = _spec(pool_slots=6)
    pool = KVSlotPool(spec.pool_slots)
    first = KVBlockTable(pool)
    first.ensure(1)
    second = KVBlockTable(pool)
    second.ensure(1)
    assert set(first.slots).isdisjoint(second.slots)

    write_row(hal, spec, first.slots[0], layer=0, head=0, which="k",
              row_index=0, row=np.full(HEAD_DIM, 7.0))
    write_row(hal, spec, second.slots[0], layer=0, head=0, which="k",
              row_index=0, row=np.full(HEAD_DIM, 3.0))

    write_block_table(spec, hal, first.slots)
    assert np.array_equal(read_rows(hal, spec, 0, 0, "k", 1)[0],
                          np.full(HEAD_DIM, 7.0))

    write_block_table(spec, hal, second.slots)
    assert np.array_equal(read_rows(hal, spec, 0, 0, "k", 1)[0],
                          np.full(HEAD_DIM, 3.0))


def test_unallocated_block_table_entry_raises() -> None:
    """块表项为 -1 时读历史直接抛错，不读垃圾数据。"""
    hal = _backend()
    spec = _spec()
    write_block_table(spec, hal, [0])
    with pytest.raises(RuntimeError, match="未分配"):
        read_rows(hal, spec, layer=0, head=0, which="k", n_tokens=3)
