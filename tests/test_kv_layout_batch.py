"""多 batch 槽位 KV 布局判据：容量公式、槽间隔离与逐槽寻址。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.kv_layout import (
    KVRegionSpec,
    build_kv_layout,
    decode_mask,
    kv_access,
    kv_bytes,
)

NUM_LAYERS, HEAD_DIM, MAX_SEQ = 2, 8, 16
DTYPE_BYTES = 4
BLOCK = MAX_SEQ * HEAD_DIM * DTYPE_BYTES  # 单个 (slot, layer, head, k|v) 子块


def _spec(num_slots: int = 1, kv_base: int = 0) -> KVRegionSpec:
    return KVRegionSpec(
        dpu_id=0,
        layers=list(range(NUM_LAYERS)),
        kv_heads=[0, 1],
        q_heads_by_kv={h: [h] for h in (0, 1)},
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype_bytes=DTYPE_BYTES,
        kv_base=kv_base,
        num_slots=num_slots,
    )


def test_slots_bytes() -> None:
    """num_slots=4：容量 = 4×单槽公式；槽间块互不重叠；slot0 与单槽布局一致。"""
    single = build_kv_layout(_spec(1), align=64)
    quad = build_kv_layout(_spec(4), align=64)
    assert kv_bytes(quad) == 4 * kv_bytes(single)
    assert quad.kv_allocated_bytes == 4 * single.kv_allocated_bytes
    # 逐块区间两两不相交（BLOCK 已按 64 对齐，无填充）。
    blocks = sorted((off, off + BLOCK) for off in quad.kv_off.values())
    for (_, end), (start, _) in zip(blocks, blocks[1:]):
        assert end <= start
    # 槽位主序：slot0 的偏移与单槽布局逐项相同（两边键都含 slot 轴）。
    for key, off in single.kv_off.items():
        assert quad.kv_off[key] == off
    # 相邻槽同块偏移差 = 单槽全量字节数（每槽整段 KV 连续）。
    stride = quad.kv_off[(1, 0, 0, "k")] - quad.kv_off[(0, 0, 0, "k")]
    assert stride == single.kv_allocated_bytes


def test_num_slots_default_is_one() -> None:
    """缺省 num_slots=1：容量与键集合和改动前等价（键含 slot=0）。"""
    spec = build_kv_layout(_spec(), align=64)
    assert spec.num_slots == 1
    assert kv_bytes(spec) == 2 * 2 * 16 * 2 * 8 * 4
    assert set(spec.kv_off) == {
        (0, layer, head, which)
        for layer in range(NUM_LAYERS) for head in (0, 1) for which in ("k", "v")
    }


def test_kv_access_per_slot() -> None:
    """按槽寻址：slot1 与 slot0 同区间偏移差一个槽步长；槽越界直接抛。"""
    spec = build_kv_layout(_spec(2), align=64)
    acc0 = kv_access(spec, 0, 1, "k", 3, 5, slot=0)
    acc1 = kv_access(spec, 0, 1, "k", 3, 5, slot=1)
    assert acc0.offset == spec.kv_off[(0, 0, 1, "k")] + 3 * HEAD_DIM * DTYPE_BYTES
    assert acc1.offset - acc0.offset == spec.kv_allocated_bytes // 2
    assert acc1.length == acc0.length
    with pytest.raises(ValueError):
        kv_access(spec, 0, 1, "k", 0, 4, slot=2)


def test_decode_mask_per_slot() -> None:
    """逐槽有效长度：返回 [B, max_seq]，每行与标量版一致。"""
    per_slot = decode_mask((2, 4), 6)
    assert per_slot.shape == (2, 6)
    assert np.array_equal(per_slot[0], decode_mask(2, 6))
    assert np.array_equal(per_slot[1], decode_mask(4, 6))
    with pytest.raises(ValueError, match="valid_len"):
        decode_mask((2, 6), 6)  # 某槽越界，整组拒绝
