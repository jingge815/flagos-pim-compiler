"""测试用的 KV 槽读写辅助：按块表映射直接读写某一行。"""

from __future__ import annotations

import numpy as np

from memory.kv_layout import KVRegionSpec

_DTYPE = {2: np.float16, 4: np.float32}


def write_row(hal, spec: KVRegionSpec, physical_slot: int, layer: int, head: int,
              which: str, row_index: int, row: np.ndarray) -> None:
    """把一行 K/V 写进指定物理槽的块内行。"""
    offset = (spec.kv_off[(layer, head, which)] + physical_slot * spec.slot_bytes
              + row_index * spec.head_dim * spec.dtype_bytes)
    hal.write_local(spec.dpu_id, offset,
                    np.ascontiguousarray(row, dtype=_DTYPE[spec.dtype_bytes]))


def read_rows(hal, spec: KVRegionSpec, layer: int, head: int,
              which: str, n_tokens: int) -> np.ndarray:
    """按全局块表把前 n_tokens 行按逻辑顺序读回来。"""
    dtype = _DTYPE[spec.dtype_bytes]
    block_table = hal.read_local(
        spec.dpu_id, spec.block_table_base,
        (spec.blocks_per_request,), np.int32)
    parts = []
    for block in range(len(block_table)):
        start = block * spec.slot_tokens
        if start >= n_tokens:
            break
        rows = min(spec.slot_tokens, n_tokens - start)
        slot = int(block_table[block])
        if slot < 0:
            raise RuntimeError(f"dpu{spec.dpu_id} 块表项 block={block} 未分配")
        offset = spec.kv_off[(layer, head, which)] + slot * spec.slot_bytes
        parts.append(hal.read_local(spec.dpu_id, offset,
                                    (rows, spec.head_dim), dtype))
    return np.concatenate(parts, axis=0)
