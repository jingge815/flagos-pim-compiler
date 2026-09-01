"""验证 Llama-2-7B 的 KV 布局和运行时读写。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.graph_meta import SPEC_META_KEY
from genesim_bridge.paths import llama2_7b_model_dir
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import (
    PIMStaticKVCache,
    build_kv_layout,
    format_kv_layout,
    kv_access,
    kv_bytes,
    kv_specs_from_placement,
)
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8  # 与 test_spec_prop_llama2_7b 一致：8 整除 32 heads，KV 按 head 均分
SEQ_LEN = 16
MAX_SEQ = 256
DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 llama2_7b_model_dir",
)


@pytest.fixture(scope="module")
def annotated_llama2():
    """构建一次带分片规格的 Llama-2-7B 图。"""
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config
    input_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()
    partition_graph(gm)
    edges = propagate_specs(gm, llama_shard_config(
        NUM_DPUS,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
    ))
    return gm, cfg, edges


def _weight_spec(gm, pattern: str):
    (node,) = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    return node.meta[SPEC_META_KEY]


@pytest.fixture(scope="module")
def kv_specs(annotated_llama2):
    gm, cfg, _ = annotated_llama2
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    specs = kv_specs_from_placement(
        _weight_spec(gm, "layers.0.self_attn.k_proj.weight"),
        layers=list(range(cfg.num_hidden_layers)),
        num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        max_seq=MAX_SEQ,
        dtype_bytes=DTYPE_BYTES,
        kv_base=0,
    )
    return {d: build_kv_layout(s, align=1024) for d, s in specs.items()}


def test_kv_sharding_follows_problem2_placement(annotated_llama2, kv_specs) -> None:
    """验证 KV 分片跟随 k_proj 和 v_proj 的放置。"""
    gm, cfg, _ = annotated_llama2
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    ref = {d: (det.start_idx, det.end_idx)
           for d, det in _weight_spec(gm, "layers.0.self_attn.k_proj.weight").shard_map.items()}
    for layer in range(cfg.num_hidden_layers):
        for proj in ("k_proj", "v_proj"):
            sm = _weight_spec(gm, f"layers.{layer}.self_attn.{proj}.weight").shard_map
            assert {d: (det.start_idx, det.end_idx) for d, det in sm.items()} == ref
    assert set(kv_specs) == set(range(NUM_DPUS))
    heads_per_dpu = cfg.num_key_value_heads // NUM_DPUS  # 32/8 = 4
    for d in range(NUM_DPUS):
        spec = kv_specs[d]
        assert spec.kv_heads == list(range(heads_per_dpu * d, heads_per_dpu * (d + 1)))
        assert spec.kv_heads[0] == ref[d][0] // head_dim
        assert spec.layers == list(range(cfg.num_hidden_layers))
        assert spec.q_heads_by_kv == {h: [h] for h in spec.kv_heads}


def test_kv_bytes_and_layout_llama2(kv_specs) -> None:
    """验证 KV 字节数和对齐布局。"""
    for spec in kv_specs.values():
        assert kv_bytes(spec) == 2 * 32 * MAX_SEQ * 4 * 128 * DTYPE_BYTES == 16 * 2**20
        # 每个 KV 块已满足 1024 字节对齐。
        assert spec.kv_allocated_bytes == kv_bytes(spec)
        assert len(spec.kv_off) == 32 * 4 * 2
        assert all(off % 1024 == 0 for off in spec.kv_off.values())
        last = max(spec.kv_off.values())
        assert last + MAX_SEQ * 128 * DTYPE_BYTES == spec.kv_base + spec.kv_allocated_bytes
    print("\n" + "\n".join(format_kv_layout(kv_specs).splitlines()[:3]))


def test_kv_runtime_smoke_on_numpy_backend(kv_specs) -> None:
    """验证 KV 追加、分块读取和访问地址。"""
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=2**26))
    cache = PIMStaticKVCache(backend, kv_specs, wram_budget_bytes=2**20)
    rng = np.random.default_rng(0)
    ref = {}
    for pos in range(3):
        for layer in (0, 31):
            k_by_dpu, v_by_dpu = {}, {}
            for dpu_id, spec in cache.specs.items():
                k_by_dpu[dpu_id], v_by_dpu[dpu_id] = {}, {}
                for head in spec.kv_heads:
                    k = rng.standard_normal(128).astype(np.float16)
                    v = rng.standard_normal(128).astype(np.float16)
                    k_by_dpu[dpu_id][head], v_by_dpu[dpu_id][head] = k, v
                    ref[(layer, dpu_id, head, pos)] = (k, v)
            cache.update(layer, pos, k_by_dpu, v_by_dpu)
    for (layer, dpu_id, head, pos), (k, v) in ref.items():
        K_tile, V_tile = cache.read_tile(layer, dpu_id, head, pos, pos + 1)
        assert np.array_equal(K_tile[0], k) and np.array_equal(V_tile[0], v)
        acc = kv_access(cache.specs[dpu_id], layer, head, "k", pos, pos + 1)
        raw = backend.read_local(dpu_id, acc.offset, (1, 128), np.float16)
        assert np.array_equal(raw[0], k)
