"""验证 KV 槽布局、块表读写和掩码。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.pim_tensor_spec import PIMTensorSpec, Placement, TensorShardDetail
from memory.kv_layout import (
    KVRegionSpec,
    align_up,
    build_kv_layout,
    decode_mask,
    format_kv_layout,
    kv_bytes,
    kv_specs_from_placement,
    prefill_mask,
    write_block_table,
)
from tests.kv_helpers import read_rows, write_row

# 小型 KV 布局配置。
NUM_LAYERS, HEAD_DIM, MAX_SEQ = 2, 8, 16
DTYPE_BYTES = 4
SLOT_TOKENS = 4
PLANE_STRIDE = SLOT_TOKENS * HEAD_DIM * DTYPE_BYTES  # 128 B，一个 (layer, head, k|v) 平面
SLOT_BYTES = 2 * NUM_LAYERS * 2 * PLANE_STRIDE  # 1024 B
BLOCKS = MAX_SEQ // SLOT_TOKENS  # 4


def _spec(dpu_id: int, kv_base: int = 0, kv_heads=(0, 1), layers=(0, 1)) -> KVRegionSpec:
    return KVRegionSpec(
        dpu_id=dpu_id,
        layers=list(layers),
        kv_heads=list(kv_heads),
        q_heads_by_kv={h: [h] for h in kv_heads},
        max_seq=MAX_SEQ,
        head_dim=HEAD_DIM,
        dtype_bytes=DTYPE_BYTES,
        kv_base=kv_base,
        slot_tokens=SLOT_TOKENS,
    )


def _specs(kv_base: int = 0) -> dict[int, KVRegionSpec]:
    return {
        0: build_kv_layout(_spec(0, kv_base, kv_heads=(0, 1)), align=64),
        1: build_kv_layout(_spec(1, kv_base, kv_heads=(2, 3)), align=64),
    }


def _backend() -> NumpyBackend:
    return NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=2**16))


def _random_kv(rng) -> tuple[np.ndarray, np.ndarray]:
    return (rng.standard_normal(HEAD_DIM).astype(np.float32),
            rng.standard_normal(HEAD_DIM).astype(np.float32))


def _k_proj_spec(num_dpus: int, num_kv_heads: int, head_dim: int, hidden: int) -> PIMTensorSpec:
    """构造沿输出维分片的 k_proj 权重规格。"""
    rows = num_kv_heads * head_dim
    width = rows // num_dpus
    return PIMTensorSpec(
        device="dpu",
        placement=Placement("Shard", 0),
        residency="pinned",
        pinned_dpu_id=None,
        shard_map={
            d: TensorShardDetail(d, 0, d * width, (d + 1) * width, (width, hidden))
            for d in range(num_dpus)
        },
        reduce_type=None,
    )


# 编译期 KV 布局。


def test_align_up() -> None:
    assert align_up(512, 64) == 512
    assert align_up(513, 64) == 576
    with pytest.raises(ValueError):
        align_up(1, 0)


def test_kv_bytes_lower_bound() -> None:
    # 一个请求的逻辑 KV 字节数。
    assert kv_bytes(_spec(0)) == 2 * 2 * 16 * 2 * 8 * 4 == 4096
    # Llama-2-7B 单 DPU 的 KV 字节数。
    llama = KVRegionSpec(0, list(range(32)), [0, 1, 2, 3], {h: [h] for h in range(4)},
                         max_seq=256, head_dim=128, dtype_bytes=2, kv_base=0,
                         slot_tokens=16)
    assert kv_bytes(llama) == 2 * 32 * 256 * 4 * 128 * 2 == 16 * 2**20


def test_build_kv_layout_offsets_exact() -> None:
    """槽 0 内的绝对地址和块表/槽池几何。"""
    spec = build_kv_layout(_spec(0, kv_base=1024), align=64)
    assert spec.slot_tokens == SLOT_TOKENS
    assert spec.slot_bytes == SLOT_BYTES == 1024
    assert spec.blocks_per_request == BLOCKS == 4
    assert spec.pool_slots == 4
    # 4 个块表项 × 4B 对齐到 64。
    assert spec.block_table_bytes == 64
    assert spec.slots_base == 1024 + 64 == 1088
    assert spec.kv_allocated_bytes == 64 + 4 * 1024 == 4160
    assert spec.kv_off == {
        (0, 0, "k"): 1088, (0, 0, "v"): 1216,
        (0, 1, "k"): 1344, (0, 1, "v"): 1472,
        (1, 0, "k"): 1600, (1, 0, "v"): 1728,
        (1, 1, "k"): 1856, (1, 1, "v"): 1984,
    }
    assert max(spec.kv_off.values()) + PLANE_STRIDE == spec.slots_base + spec.slot_bytes

    # 对齐放大后，每个平面和块表缓冲都是 1024。
    padded = build_kv_layout(_spec(0, kv_base=1024), align=1024)
    assert padded.kv_off[(0, 0, "k")] == 1024 + 1024 == 2048
    assert padded.slot_bytes == 2 * 2 * 2 * 1024 == 8192
    assert padded.kv_allocated_bytes == 1024 + 4 * 8192 == 33792
    assert all(off % 1024 == 0 for off in padded.kv_off.values())


def test_kv_region_spec_crud_and_rebuild() -> None:
    """KVRegionSpec 可初始化/读/写/增/删/改：dataclass 字段可变，改完重建布局即可。"""
    spec = build_kv_layout(_spec(0, kv_base=0), align=64)
    assert spec.kv_off[(0, 0, "k")] == 64  # 读：块表缓冲之后是槽池
    spec.kv_base = 2048                   # 改：基址平移
    build_kv_layout(spec, align=64)
    assert spec.kv_off[(0, 0, "k")] == 2112 and spec.kv_allocated_bytes == 4160
    spec.layers.append(2)                 # 增：多一层，每槽多 2 头 × 2(K/V) × 128B
    build_kv_layout(spec, align=64)
    assert spec.slot_bytes == 1536
    assert spec.kv_allocated_bytes == 64 + 4 * 1536
    spec.layers.remove(0)                 # 删：少一层
    spec.kv_heads.remove(1)               # 删：少一个 head
    build_kv_layout(spec, align=64)
    assert set(spec.kv_off) == {(L, 0, w) for L in spec.layers for w in ("k", "v")}
    broken = _spec(0)
    del broken.q_heads_by_kv[1]           # 破坏契约：kv head 1 无 q 映射
    with pytest.raises(ValueError):
        build_kv_layout(broken, align=64)
    with pytest.raises(ValueError):       # 池装不下一个满长请求
        build_kv_layout(_spec(0, kv_base=0, kv_heads=(0,), layers=(0,)), align=64,
                        pool_slots=0)


def test_kv_specs_from_placement_mha() -> None:
    """验证列切 k_proj 的 KV head 按 DPU 分配。"""
    spec = _k_proj_spec(num_dpus=4, num_kv_heads=32, head_dim=128, hidden=4096)
    specs = kv_specs_from_placement(spec, layers=list(range(32)), num_kv_heads=32, num_q_heads=32,
                                    head_dim=128, max_seq=256, dtype_bytes=2, kv_base=0,
                                    slot_tokens=8)
    assert set(specs) == {0, 1, 2, 3}
    for d in range(4):
        assert specs[d].kv_heads == list(range(8 * d, 8 * d + 8))
        assert specs[d].q_heads_by_kv == {h: [h] for h in specs[d].kv_heads}
        assert specs[d].layers == list(range(32))
        assert specs[d].slot_tokens == 8


def test_kv_specs_from_placement_gqa() -> None:
    """GQA 通用写法：8 KV head / 32 Q head，一个 KV head 服务连续 4 个 Q head。"""
    spec = _k_proj_spec(num_dpus=4, num_kv_heads=8, head_dim=128, hidden=4096)
    specs = kv_specs_from_placement(spec, layers=list(range(2)), num_kv_heads=8, num_q_heads=32,
                                    head_dim=128, max_seq=16, dtype_bytes=2, kv_base=0)
    assert specs[0].kv_heads == [0, 1]
    assert specs[0].q_heads_by_kv == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    with pytest.raises(ValueError):  # 切点未对齐 head 边界 → KV 本地驻留前提被破坏
        bad = _k_proj_spec(4, 8, 128, 4096)
        bad.shard_map[0] = TensorShardDetail(0, 0, 64, 320, (256, 4096))
        kv_specs_from_placement(bad, layers=list(range(2)), num_kv_heads=8, num_q_heads=32,
                                head_dim=128, max_seq=16, dtype_bytes=2, kv_base=0)
    with pytest.raises(ValueError):  # 非列切不满足"KV 按 head 切"
        rep = PIMTensorSpec("dpu", Placement("Replicate"), "pinned", None, {}, None)
        kv_specs_from_placement(rep, layers=list(range(2)), num_kv_heads=8, num_q_heads=32,
                                head_dim=128, max_seq=16, dtype_bytes=2, kv_base=0)


def test_kv_specs_from_placement_rejects_incomplete_or_overlapping_heads() -> None:
    """验证 k_proj 分片恰好覆盖每个 KV head。"""
    incomplete = _k_proj_spec(2, 4, 2, 8)
    incomplete.shard_map[1] = TensorShardDetail(1, 0, 2, 6, (4, 8))
    with pytest.raises(ValueError, match="覆盖"):
        kv_specs_from_placement(incomplete, layers=list(range(2)), num_kv_heads=4, num_q_heads=4,
                                head_dim=2, max_seq=16, dtype_bytes=2, kv_base=0)

    overlapping = _k_proj_spec(2, 4, 2, 8)
    overlapping.shard_map[1] = TensorShardDetail(1, 0, 2, 8, (6, 8))
    with pytest.raises(ValueError, match="覆盖"):
        kv_specs_from_placement(overlapping, layers=list(range(2)), num_kv_heads=4, num_q_heads=4,
                                head_dim=2, max_seq=16, dtype_bytes=2, kv_base=0)


def test_write_block_table_pads_and_validates() -> None:
    """块表按映射填值、不足补 -1；块号越界和表太长直接抛错。"""
    hal = _backend()
    spec = build_kv_layout(_spec(0, kv_base=0), align=64, pool_slots=8)
    write_block_table(spec, hal, [5, 2])
    entries = hal.read_local(0, spec.block_table_base,
                             (spec.blocks_per_request,), np.int32)
    assert entries.tolist() == [5, 2, -1, -1]
    with pytest.raises(ValueError):
        write_block_table(spec, hal, [spec.pool_slots])
    with pytest.raises(ValueError):
        write_block_table(spec, hal, [0] * 5)


def test_format_kv_layout_printable() -> None:
    text = format_kv_layout(_specs())
    assert "dpu0" in text and "slot_tokens=4" in text and "pool_slots=4" in text
    assert "block_table_bytes=64" in text and "max_full_len_requests=1" in text
    assert "kv_heads=[0..1]" in text


# 掩码。


def test_masks() -> None:
    # 预填充掩码屏蔽未来位置和预留位置。
    expected = np.array([
        [0, -np.inf, -np.inf, -np.inf, -np.inf],
        [0, 0, -np.inf, -np.inf, -np.inf],
        [0, 0, 0, -np.inf, -np.inf],
        [0, 0, 0, -np.inf, -np.inf],
        [0, 0, 0, -np.inf, -np.inf],
    ], dtype=np.float32)
    assert np.array_equal(prefill_mask(3, 5), expected)
    assert np.array_equal(decode_mask(2, 5),
                          np.array([0, 0, 0, -np.inf, -np.inf], dtype=np.float32))
    assert prefill_mask(3, 5).dtype == np.float32
    with pytest.raises(ValueError, match="prompt_len"):
        prefill_mask(6, 5)
    with pytest.raises(ValueError, match="prompt_len"):
        prefill_mask(1.5, 5)
    with pytest.raises(ValueError, match="valid_len"):
        decode_mask(5, 5)
    with pytest.raises(ValueError, match="valid_len"):
        decode_mask(1.5, 5)


# 按页表 gather 的注意力数值验证。

IDENTITY_TABLE = list(range(BLOCKS))  # 逻辑块 b -> 物理槽 b


def _write_step(hal, specs, rng, layer: int, pos: int,
                ref: dict[int, list]) -> None:
    """按 identity 块表写入一个位置的所有 head。"""
    for dpu_id, spec in specs.items():
        for head in spec.kv_heads:
            k, v = _random_kv(rng)
            write_row(hal, spec, pos // SLOT_TOKENS, layer, head, "k",
                      pos % SLOT_TOKENS, k)
            write_row(hal, spec, pos // SLOT_TOKENS, layer, head, "v",
                      pos % SLOT_TOKENS, v)
            ref.setdefault(head, []).append((k, v))


def _host_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """计算主机端 softmax。"""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def test_attention_decode_matches_torch() -> None:
    """按块表逐块 gather 的 KV 注意力与 Torch 参考一致。"""
    hal = _backend()
    specs = _specs()
    for spec in specs.values():
        write_block_table(spec, hal, IDENTITY_TABLE)
    rng = np.random.default_rng(2)
    layer = 1
    kv_ref: dict[int, list] = {}
    for pos in range(5):
        _write_step(hal, specs, rng, layer, pos, kv_ref)
    for valid_len in range(5, 8):
        _write_step(hal, specs, rng, layer, valid_len, kv_ref)
        q_by_head = {h: rng.standard_normal(HEAD_DIM).astype(np.float32) for h in range(4)}
        mask = decode_mask(valid_len, valid_len + 1)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        for dpu_id, spec in specs.items():
            for head in spec.kv_heads:
                K = read_rows(hal, spec, layer, head, "k", valid_len + 1)
                V = read_rows(hal, spec, layer, head, "v", valid_len + 1)
                out = _host_softmax(K @ q_by_head[head] * scale + mask) @ V
                K_ref = torch.tensor(np.stack([k for k, _ in kv_ref[head]]))
                V_ref = torch.tensor(np.stack([v for _, v in kv_ref[head]]))
                ref = torch.softmax(K_ref @ torch.tensor(q_by_head[head]) / np.sqrt(HEAD_DIM),
                                    dim=0) @ V_ref
                assert np.allclose(out, ref.numpy(), atol=1e-6), f"head {head}"


def test_attention_prefill_matches_torch() -> None:
    """prefill：按块表 gather 的满历史 + prefill_mask 与 torch 因果参考一致。"""
    hal = _backend()
    specs = _specs()
    for spec in specs.values():
        write_block_table(spec, hal, IDENTITY_TABLE)
    rng = np.random.default_rng(3)
    layer, prompt_len = 0, 5
    kv_ref: dict[int, list] = {}
    for pos in range(prompt_len):
        _write_step(hal, specs, rng, layer, pos, kv_ref)
    mask = prefill_mask(prompt_len, prompt_len)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for dpu_id, spec in specs.items():
        for head in spec.kv_heads:
            Q = rng.standard_normal((prompt_len, HEAD_DIM)).astype(np.float32)
            K = read_rows(hal, spec, layer, head, "k", prompt_len)
            V = read_rows(hal, spec, layer, head, "v", prompt_len)
            out = _host_softmax(Q @ K.T * scale + mask) @ V
            K_ref = torch.tensor(np.stack([k for k, _ in kv_ref[head]]))
            V_ref = torch.tensor(np.stack([v for _, v in kv_ref[head]]))
            causal = torch.triu(torch.ones(prompt_len, prompt_len, dtype=torch.bool), diagonal=1)
            ref_scores = torch.tensor(Q) @ K_ref.T * scale
            ref_scores.masked_fill_(causal, float("-inf"))
            ref = torch.softmax(ref_scores, dim=-1) @ V_ref
            assert np.allclose(out, ref.numpy(), atol=1e-6), f"head {head}"
