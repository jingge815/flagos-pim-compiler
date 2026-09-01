"""验证 KV 缓存布局、读写、掩码和注意力计算。"""

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
    PIMStaticKVCache,
    align_up,
    build_kv_layout,
    decode_mask,
    format_kv_layout,
    kv_access,
    kv_bytes,
    kv_specs_from_placement,
    prefill_mask,
)

# 小型 KV 布局配置。
NUM_LAYERS, HEAD_DIM, MAX_SEQ = 2, 8, 16
DTYPE_BYTES = 4
BLOCK = MAX_SEQ * HEAD_DIM * DTYPE_BYTES  # 512 B，一个 (layer, head, k|v) 子块
WRAM_BUDGET = 4 * BLOCK


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


def _write_one_step(cache: PIMStaticKVCache, rng, layer: int, pos: int,
                    ref: dict[int, list] | None = None) -> None:
    """在指定位置写入一轮 KV 数据。"""
    k_by_dpu, v_by_dpu = {}, {}
    for dpu_id, spec in cache.specs.items():
        k_by_dpu[dpu_id], v_by_dpu[dpu_id] = {}, {}
        for head in spec.kv_heads:
            k, v = _random_kv(rng)
            k_by_dpu[dpu_id][head], v_by_dpu[dpu_id][head] = k, v
            if ref is not None:
                ref.setdefault(head, []).append((k, v))
    cache.update(layer, pos, k_by_dpu, v_by_dpu)


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
    # 手工计算的 KV 总字节数。
    assert kv_bytes(_spec(0)) == 2 * 2 * 16 * 2 * 8 * 4 == 4096
    # Llama-2-7B 单 DPU 的 KV 字节数。
    llama = KVRegionSpec(0, list(range(32)), [0, 1, 2, 3], {h: [h] for h in range(4)},
                         max_seq=256, head_dim=128, dtype_bytes=2, kv_base=0)
    assert kv_bytes(llama) == 2 * 32 * 256 * 4 * 128 * 2 == 16 * 2**20


def test_build_kv_layout_offsets_exact() -> None:
    spec = build_kv_layout(_spec(0, kv_base=1024), align=64)  # BLOCK=512 已对齐，无 padding
    assert spec.kv_off == {
        (0, 0, "k"): 1024, (0, 0, "v"): 1536,
        (0, 1, "k"): 2048, (0, 1, "v"): 2560,
        (1, 0, "k"): 3072, (1, 0, "v"): 3584,
        (1, 1, "k"): 4096, (1, 1, "v"): 4608,
    }
    assert spec.kv_allocated_bytes == 4096 == kv_bytes(spec)  # 无 padding 时二者相等
    # 对齐填充后的已分配字节数。
    padded = build_kv_layout(_spec(0, kv_base=1024), align=1024)
    assert padded.kv_allocated_bytes == 8192 == 2 * kv_bytes(padded)
    assert all(off % 1024 == 0 for off in padded.kv_off.values())


def test_kv_region_spec_crud_and_rebuild() -> None:
    """KVRegionSpec 可初始化/读/写/增/删/改：dataclass 字段可变，改完重建布局即可。"""
    spec = build_kv_layout(_spec(0, kv_base=0), align=64)
    assert spec.kv_off[(0, 0, "k")] == 0  # 读
    spec.kv_base = 2048                   # 改：基址平移
    build_kv_layout(spec, align=64)
    assert spec.kv_off[(0, 0, "k")] == 2048 and spec.kv_allocated_bytes == 4096
    spec.layers.append(2)                 # 增：多一层，占用多 2头×2(K/V)×512B
    build_kv_layout(spec, align=64)
    assert spec.kv_allocated_bytes == 4096 + 4 * BLOCK
    spec.layers.remove(0)                 # 删：少一层
    spec.kv_heads.remove(1)               # 删：少一个 head
    build_kv_layout(spec, align=64)
    assert spec.kv_allocated_bytes == len(spec.layers) * len(spec.kv_heads) * 2 * BLOCK
    assert set(spec.kv_off) == {(L, 0, w) for L in spec.layers for w in ("k", "v")}
    broken = _spec(0)
    del broken.q_heads_by_kv[1]           # 破坏契约：kv head 1 无 q 映射
    with pytest.raises(ValueError):
        build_kv_layout(broken, align=64)


def test_kv_specs_from_placement_mha() -> None:
    """验证列切 k_proj 的 KV head 按 DPU 分配。"""
    spec = _k_proj_spec(num_dpus=4, num_kv_heads=32, head_dim=128, hidden=4096)
    specs = kv_specs_from_placement(spec, layers=list(range(32)), num_kv_heads=32, num_q_heads=32,
                                    head_dim=128, max_seq=256, dtype_bytes=2, kv_base=0)
    assert set(specs) == {0, 1, 2, 3}
    for d in range(4):
        assert specs[d].kv_heads == list(range(8 * d, 8 * d + 8))
        assert specs[d].q_heads_by_kv == {h: [h] for h in specs[d].kv_heads}
        assert specs[d].layers == list(range(32))


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


def test_kv_access_matches_layout() -> None:
    """验证命令访问区间与 KV 实际地址一致。"""
    spec = _specs()[0]
    acc = kv_access(spec, layer=1, head=0, which="v", pos_start=3, pos_end=7)
    assert acc.loc == ("dpu", 0)
    assert acc.offset == spec.kv_off[(1, 0, "v")] + 3 * HEAD_DIM * DTYPE_BYTES
    assert acc.length == 4 * HEAD_DIM * DTYPE_BYTES
    with pytest.raises(ValueError):
        kv_access(spec, 0, 0, "k", 0, MAX_SEQ + 1)


def test_format_kv_layout_printable() -> None:
    text = format_kv_layout(_specs())
    assert "dpu0" in text and "kv_allocated_bytes=4096" in text and "kv_heads=[0..1]" in text


# 运行时 KV 读写和掩码。


def test_update_then_read_tile_roundtrip() -> None:
    """按 valid_len 逐步追加（pos 由调用方传入），再按 tile 读回，逐元素相等。"""
    cache = PIMStaticKVCache(_backend(), _specs(), wram_budget_bytes=WRAM_BUDGET)
    rng = np.random.default_rng(0)
    ref: dict[tuple[int, int, int], list] = {}  # (layer, dpu_id, head) -> [(k, v)]
    for pos in range(6):
        for layer in range(NUM_LAYERS):
            k_by_dpu, v_by_dpu = {}, {}
            for dpu_id, spec in cache.specs.items():
                k_by_dpu[dpu_id], v_by_dpu[dpu_id] = {}, {}
                for head in spec.kv_heads:
                    k, v = _random_kv(rng)
                    k_by_dpu[dpu_id][head], v_by_dpu[dpu_id][head] = k, v
                    ref.setdefault((layer, dpu_id, head), []).append((k, v))
            cache.update(layer, pos, k_by_dpu, v_by_dpu)
    for (layer, dpu_id, head), history in ref.items():
        K_tile, V_tile = cache.read_tile(layer, dpu_id, head, 0, 6)
        assert np.array_equal(K_tile, np.stack([k for k, _ in history]))
        assert np.array_equal(V_tile, np.stack([v for _, v in history]))
        K_part, V_part = cache.read_tile(layer, dpu_id, head, 2, 5)  # 不满块读取
        assert np.array_equal(K_part, K_tile[2:5]) and np.array_equal(V_part, V_tile[2:5])


def test_cache_object_holds_no_state() -> None:
    """"删除"= 丢掉 cache 对象：数据驻留各 DPU MRAM，同一 backend 上重建对象读回不变。"""
    backend = _backend()
    specs = _specs()
    rng = np.random.default_rng(1)
    cache = PIMStaticKVCache(backend, specs, wram_budget_bytes=WRAM_BUDGET)
    ref: dict[int, list] = {}
    _write_one_step(cache, rng, layer=0, pos=0, ref=ref)
    k0, v0 = ref[0][0]  # head 0 写在 dpu0
    del cache  # cache 对象不自持数据、无内部指针，删掉不影响 MRAM 内容
    fresh = PIMStaticKVCache(backend, specs, wram_budget_bytes=WRAM_BUDGET)
    K, V = fresh.read_tile(0, 0, 0, 0, 1)
    assert np.array_equal(K[0], k0) and np.array_equal(V[0], v0)
    # 未布局的规格不能创建缓存。
    with pytest.raises(ValueError):
        PIMStaticKVCache(backend, {0: _spec(0)}, WRAM_BUDGET)


def test_update_and_read_tile_contract_violations() -> None:
    cache = PIMStaticKVCache(_backend(), _specs(), wram_budget_bytes=WRAM_BUDGET)
    vec = np.zeros(HEAD_DIM, dtype=np.float32)
    full = {d: {h: vec for h in s.kv_heads} for d, s in cache.specs.items()}
    with pytest.raises(ValueError, match="pos=.*越界"):  # pos 越界（写进预留区外）
        cache.update(0, MAX_SEQ, full, full)
    with pytest.raises(ValueError):  # dtype 与规格不符
        bad = {d: {h: vec.astype(np.float64) for h in s.kv_heads} for d, s in cache.specs.items()}
        cache.update(0, 0, bad, bad)
    with pytest.raises(ValueError, match="dtype"):  # 字节数相同但语义不同的 dtype 也不能写入
        wrong_dtype = {d: {h: np.zeros(HEAD_DIM, dtype=np.uint32) for h in s.kv_heads}
                       for d, s in cache.specs.items()}
        cache.update(0, 0, wrong_dtype, wrong_dtype)
    with pytest.raises(ValueError):  # tile 越界
        cache.read_tile(0, 0, 0, 0, MAX_SEQ + 1)
    with pytest.raises(KeyError):  # head 不在本 DPU
        cache.read_tile(0, 0, 3, 0, 1)
    tight = PIMStaticKVCache(cache.backend, cache.specs,
                             wram_budget_bytes=2 * HEAD_DIM * DTYPE_BYTES)
    with pytest.raises(ValueError):  # 单 tile 超 WRAM 预算
        tight.read_tile(0, 0, 0, 0, 3)
    tight.read_tile(0, 0, 0, 0, 2)  # 恰好预算内可读


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


# 按分块读取 KV 的注意力数值验证。


def _host_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """计算主机端 softmax。"""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _pim_decode_attention(kv: PIMStaticKVCache, layer: int, q_by_head: dict[int, np.ndarray],
                          valid_len: int, tile: int) -> dict[int, np.ndarray]:
    """按分块读取 KV 并计算单步解码注意力。"""
    mask = decode_mask(valid_len, MAX_SEQ)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    out = {}
    for dpu_id, spec in kv.specs.items():
        for head in spec.kv_heads:
            q = q_by_head[head]
            scores = np.concatenate([
                kv.read_tile(layer, dpu_id, head, s, min(s + tile, MAX_SEQ))[0] @ q * scale
                for s in range(0, MAX_SEQ, tile)
            ])
            weights = _host_softmax(scores + mask)  # host 胶水
            acc = np.zeros(HEAD_DIM, dtype=np.float32)
            for s in range(0, MAX_SEQ, tile):
                _, V_tile = kv.read_tile(layer, dpu_id, head, s, min(s + tile, MAX_SEQ))
                acc += weights[s:s + V_tile.shape[0]] @ V_tile
            out[head] = acc
    return out


def test_attention_decode_matches_torch() -> None:
    """验证分块 KV 注意力与 Torch 参考一致。"""
    cache = PIMStaticKVCache(_backend(), _specs(), wram_budget_bytes=WRAM_BUDGET)
    rng = np.random.default_rng(2)
    layer = 1
    kv_ref: dict[int, list] = {h: [] for h in range(4)}  # head -> [(k, v)]
    for pos in range(5):  # prefill：逐 pos 写入（真机由投影 kernel 完成）
        _write_one_step(cache, rng, layer, pos, kv_ref)
    for valid_len in range(5, 8):  # decode 3 步：先写 pos=valid_len，再按 [0..valid_len] 读
        _write_one_step(cache, rng, layer, valid_len, kv_ref)
        q_by_head = {h: rng.standard_normal(HEAD_DIM).astype(np.float32) for h in range(4)}
        out = _pim_decode_attention(cache, layer, q_by_head, valid_len, tile=6)
        for h in range(4):  # 与单卡 PyTorch 逐元素对齐
            K = torch.tensor(np.stack([k for k, _ in kv_ref[h]]))
            V = torch.tensor(np.stack([v for _, v in kv_ref[h]]))
            q = torch.tensor(q_by_head[h])
            ref = torch.softmax(K @ q / np.sqrt(HEAD_DIM), dim=0) @ V
            assert np.allclose(out[h], ref.numpy(), atol=1e-6), f"head {h} valid_len {valid_len}"


def test_attention_prefill_matches_torch() -> None:
    """prefill：scores 方阵 + prefill_mask，前 P 行与 torch 因果参考一致。"""
    cache = PIMStaticKVCache(_backend(), _specs(), wram_budget_bytes=WRAM_BUDGET)
    rng = np.random.default_rng(3)
    layer, prompt_len = 0, 5
    kv_ref: dict[int, list] = {h: [] for h in range(4)}
    for pos in range(prompt_len):
        _write_one_step(cache, rng, layer, pos, kv_ref)
    mask = prefill_mask(prompt_len, MAX_SEQ)
    scale = 1.0 / np.sqrt(HEAD_DIM)
    for h in range(4):
        dpu_id = 0 if h < 2 else 1
        Q = rng.standard_normal((prompt_len, HEAD_DIM)).astype(np.float32)
        K_full, V_full = cache.read_tile(layer, dpu_id, h, 0, MAX_SEQ)  # 满算 + mask
        scores = Q @ K_full.T * scale + mask[:prompt_len]
        out = _host_softmax(scores) @ V_full
        K = torch.tensor(np.stack([k for k, _ in kv_ref[h]]))
        V = torch.tensor(np.stack([v for _, v in kv_ref[h]]))
        causal = torch.triu(torch.ones(prompt_len, prompt_len, dtype=torch.bool), diagonal=1)
        ref_scores = torch.tensor(Q) @ K.T * scale
        ref_scores.masked_fill_(causal, float("-inf"))
        ref = torch.softmax(ref_scores, dim=-1) @ V
        assert np.allclose(out, ref.numpy(), atol=1e-6), f"head {h}"
