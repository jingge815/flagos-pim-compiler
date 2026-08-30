"""真实 Llama-2-7B 的问题 1 → 2 → 7 端到端验证：KV cache 布局与运行时读写。

切法来自问题 2 的真实产物（k_proj.weight 的 PIMTensorSpec）：8 台 DPU 各驻留
32/8=4 个 KV head、全部 32 层（MHA，q_heads_by_kv 退化为恒等）。结构性判据对照
手推值（kv_bytes = 2×32×max_seq×4×128×2B）；运行时判据是在 NumpyBackend 上
按 valid_len 追加、按 tile 读回，逐元素相等。本图同时带着问题 3 的 redistribute
边（test_spec_prop_llama2_7b / test_comm_llama2_7b 已覆盖），KV 区与其共用
同一份标注图，互不依赖。
"""

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

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8  # 与 test_spec_prop_llama2_7b 一致：8 整除 32 heads，KV 按 head 均分
SEQ_LEN = 16
MAX_SEQ = 256   # 第 1 阶段取小值控制 KV 区总量（方案问题 7 四）
DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


@pytest.fixture(scope="module")
def annotated_llama2():
    """真实 7B：export → partition → propagate（问题 1/2 产物），模块内只跑一次。"""
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
        kv_base=0,  # plan.kv_base 是问题 8 的产物；容量/布局验证用 0 基址即可
    )
    return {d: build_kv_layout(s, align=1024) for d, s in specs.items()}


def test_kv_sharding_follows_problem2_placement(annotated_llama2, kv_specs) -> None:
    """怎么切：KV 驻留由问题 2 的 k_proj 列切推出；各层 k/v_proj 切法一致是 layers=全层的前提。"""
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
        assert spec.kv_heads[0] == ref[d][0] // head_dim  # 与问题 2 切点逐 DPU 对齐
        assert spec.layers == list(range(cfg.num_hidden_layers))
        assert spec.q_heads_by_kv == {h: [h] for h in spec.kv_heads}  # MHA 恒等映射


def test_kv_bytes_and_layout_llama2(kv_specs) -> None:
    """kv_bytes 手推值：2(K/V) × 32层 × 256 × 4头 × 128 × 2B = 16 MiB / DPU。"""
    for spec in kv_specs.values():
        assert kv_bytes(spec) == 2 * 32 * MAX_SEQ * 4 * 128 * DTYPE_BYTES == 16 * 2**20
        # block = 256×128×2B = 64 KiB 已对齐 1024 → 无 padding，allocated == kv_bytes
        assert spec.kv_allocated_bytes == kv_bytes(spec)
        assert len(spec.kv_off) == 32 * 4 * 2
        assert all(off % 1024 == 0 for off in spec.kv_off.values())
        last = max(spec.kv_off.values())
        assert last + MAX_SEQ * 128 * DTYPE_BYTES == spec.kv_base + spec.kv_allocated_bytes
    print("\n" + "\n".join(format_kv_layout(kv_specs).splitlines()[:3]))


def test_kv_runtime_smoke_on_numpy_backend(kv_specs) -> None:
    """运行时：按 valid_len 追加 → read_tile 读回 → kv_access 区间与实际访问地址一致。"""
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=2**26))
    cache = PIMStaticKVCache(backend, kv_specs, wram_budget_bytes=2**20)
    rng = np.random.default_rng(0)
    ref = {}
    for pos in range(3):  # 模拟 DecodeState.valid_len = 0,1,2 三步
        for layer in (0, 31):  # 首尾两层抽查
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
        raw = backend.read_local(dpu_id, acc.offset, (1, 128), np.float16)  # Access 区间直读
        assert np.array_equal(raw[0], k)  # 问题 6 填 reads/writes 的区间 = 实际写地址
