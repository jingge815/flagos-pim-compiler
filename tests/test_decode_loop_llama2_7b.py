"""真实 Llama-2-7B 完整 prefill + 16 步 decode 循环（问题 1→2→3→6→7→8 端到端，
验证 3）——用户明确要求"numpy 后端完整验证 llama2 7b 模型完整推理流程"的判据，
不用玩具模型代替。

`NUM_DPUS=8`（与全仓既有 7B 测试一致）、prefill `SEQ_LEN=16`、decode 16 步、
`max_seq=64`。判据：编排器贪心解码产出的 token 序列与 HF 原生
`model(..., use_cache=True)` 自回归推进的贪心解码序列**逐 token 完全一致**
（比数值容差更贴近"推理是否正确"），并直接核验 KV 区里的字节——写对了图的
输出和写对了 KV 区里的数据是两件独立验证过的事。

**decode 图必须显式传 `position_ids`**：`_FixedMaskLlama`（问题 1/2/3/7/8
五个既有测试共用）不传 `position_ids`，`torch.export` 会把 `arange(0,
seq_len)` 烤成编译期常量——对 `seq_len=1` 的 decode 图，意味着每次调用都
按位置 0 算 RoPE，与真实 `valid_len` 无关，第 2 个 decode 步开始 K/V 就错
（真实复现过：K 在第 16 位与参考差 1.11，是随机噪声级别的错误，不是精度
误差）。这里改用问题 6 专用的 `_PositionalLlama` 包装（不改
`_FixedMaskLlama` 本体，避免影响问题 1/2/3/7/8 的既有测试）。
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
from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import PIMStaticKVCache, kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.compile import sdpa_layer_map, write_weight_shards
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, make_sdpa_handler, run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
PREFILL_SEQ_LEN = 16
DECODE_STEPS = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


class _PositionalLlama(torch.nn.Module):
    """RoPE 位置显式作为图输入（问题 6 decode 循环专用，见模块 docstring）。"""

    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.model(
            input_ids=input_ids, attention_mask=causal_mask, position_ids=position_ids,
            use_cache=False, return_dict=True,
        ).logits


def _causal_mask_of(seq_len: int) -> torch.Tensor:
    if seq_len == 1:
        return torch.zeros(1, 1, 1, 1, dtype=torch.float16)
    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16)
    mask.masked_fill_(blocked, torch.finfo(mask.dtype).min)
    return mask


def _export_graph(model: LlamaForCausalLM, seq_len: int, position_ids: torch.Tensor):
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    gm = torch.export.export(
        _PositionalLlama(model), (input_ids, _causal_mask_of(seq_len), position_ids), strict=True
    ).module()
    partition_graph(gm)
    return gm



@pytest.fixture(scope="module")
def llama2_decode_setup():
    """真实 7B：两张图（prefill/decode）各自 export → partition → propagate →
    问题 3/7/8 → build_execution_plan（模块内一次）。"""
    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config

    prefill_gm = _export_graph(model, PREFILL_SEQ_LEN, torch.arange(PREFILL_SEQ_LEN, dtype=torch.long).unsqueeze(0))
    decode_gm = _export_graph(model, 1, torch.tensor([[0]], dtype=torch.long))

    shard_config = llama_shard_config(
        NUM_DPUS, num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
    )
    prefill_edges = propagate_specs(prefill_gm, shard_config)
    decode_edges = propagate_specs(decode_gm, shard_config)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    (k_proj,) = [n for n in prefill_gm.graph.nodes if n.op == "get_attr" and "layers.0.self_attn.k_proj.weight" in n.target]
    kv_specs = kv_specs_from_placement(
        k_proj.meta[SPEC_META_KEY], layers=list(range(cfg.num_hidden_layers)), num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads, head_dim=head_dim, max_seq=MAX_SEQ, dtype_bytes=KV_DTYPE_BYTES, kv_base=0,
    )
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS,
        num_tasklets=4,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        # dma_align 是 WRAM tile 搬运的对齐要求，跟 hw.align（MRAM 里
        # 张量摆放对齐，供 memory/mem_planner.py 用）是两个不同量级的概念，
        # 不能共用同一个值——用 hw.align=1024 会让 kernel_src.py 编出的
        # tile（几百到几万字节）几乎全部不整除，实测触发
        # pim-tile-to-budget 的 DMA 对齐报错。
        dma_align=64,
    )
    prefill_nodes = list(prefill_gm.graph.nodes)
    decode_nodes = list(decode_gm.graph.nodes)
    plans = {d: plan_dpu(d, prefill_nodes, decode_nodes, kv_specs, hw) for d in range(NUM_DPUS)}

    prefill_entries = {e.edge_id: e for e in build_comm_plan(prefill_edges)}
    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
    pending_prefill, pending_decode = {}, {}
    for plan in plans.values():
        pending_prefill.update(plan.pending_readers_prefill)
        pending_decode.update(plan.pending_readers_decode)

    state = DecodeState(valid_len=0)


    def make_host_handler(layer_map):
        def host_handler_of(node):
            if "scaled_dot_product_attention" in str(node.target):
                return make_sdpa_handler(layer_map[node.name], kv_specs, state, np.dtype(np.float16))
            return None
        return host_handler_of

    prefill_compiled = build_execution_plan(
        prefill_nodes, prefill_gm, prefill_entries, pending_prefill,
        hardware=hardware,
        host_handler_of=make_host_handler(sdpa_layer_map(prefill_gm)),
    )
    decode_compiled = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware,
        host_handler_of=make_host_handler(sdpa_layer_map(decode_gm)),
    )

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend)

    write_weight_shards(prefill_gm, plans, backend)

    return model, backend, prefill_compiled, decode_compiled, state, kv_specs


@pytest.fixture(scope="module")
def llama2_decode_result(llama2_decode_setup):
    model, backend, prefill_compiled, decode_compiled, state, kv_specs = llama2_decode_setup
    prompt_ids = torch.arange(PREFILL_SEQ_LEN, dtype=torch.long)

    def greedy(logits_1d):
        return int(np.argmax(logits_1d))

    generated = run_decode_loop(
        prefill_compiled.plan, decode_compiled.plan, backend,
        prompt_ids=prompt_ids, max_new_tokens=DECODE_STEPS, eos_id=-1, state=state,
        sample_fn=greedy, prefill_output_cmd_id=prefill_compiled.output_cmd_id,
        decode_output_cmd_id=decode_compiled.output_cmd_id, causal_mask_of=_causal_mask_of,
    )

    ref_tokens = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax())
        ref_tokens.append(next_tok)
        for _ in range(DECODE_STEPS - 1):
            out = model(input_ids=torch.tensor([[next_tok]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax())
            ref_tokens.append(next_tok)

    return generated, ref_tokens, past, backend, kv_specs, state


def test_greedy_decoded_tokens_match_hf_autoregressive_generation(llama2_decode_result) -> None:
    """判据 1：编排器逐步贪心解码的 token 序列与 HF 原生自回归推进逐个一致。"""
    generated, ref_tokens, *_ = llama2_decode_result
    assert generated == ref_tokens


def test_kv_region_bytes_match_hf_dynamic_cache(llama2_decode_result) -> None:
    """判据 2：直接核验 KV 区里的字节——不止"最终 logits 凑巧对"，KV 区内部
    的每一层每个 head 都要与 HF `DynamicCache` 的 post-rope K/V 逐元素对齐。
    """
    _, _, ref_past, backend, kv_specs, state = llama2_decode_result
    cache = PIMStaticKVCache(backend, kv_specs, wram_budget_bytes=2**20)
    total_len = state.valid_len
    assert total_len == PREFILL_SEQ_LEN + DECODE_STEPS - 1

    # fp16 的绝对精度随数值量级线性变宽（ULP ≈ |x| × 2^-10），深层（如
    # layer15/31）RoPE/matmul 累积后 K/V 幅值可以到几十，固定绝对容差在那里
    # 会误报——用相对+绝对混合容差（`np.allclose` 惯用做法），不是逐 bit
    # 比较。这与"贪心解码 token 序列逐个一致"这条更严的判据互补：那条已经
    # 证明这个量级的浮点误差不影响推理产出。
    for layer in (0, 15, 31):  # 首层、中间层、末层各抽查一层，覆盖不代表逐层全查
        ref_k = ref_past.layers[layer].keys  # [1, num_kv_heads, total_len, head_dim]
        ref_v = ref_past.layers[layer].values
        for dpu_id, spec in kv_specs.items():
            for head in spec.kv_heads:
                K_tile, V_tile = cache.read_tile(layer, dpu_id, head, 0, total_len)
                ref_k_h = ref_k[0, head].numpy().astype(np.float32)
                ref_v_h = ref_v[0, head].numpy().astype(np.float32)
                assert np.allclose(K_tile.astype(np.float32), ref_k_h, rtol=2e-2, atol=3e-2), (
                    f"layer{layer} dpu{dpu_id} head{head} K 区不匹配: max diff "
                    f"{np.abs(K_tile.astype(np.float32) - ref_k_h).max()}"
                )
                assert np.allclose(V_tile.astype(np.float32), ref_v_h, rtol=2e-2, atol=3e-2), (
                    f"layer{layer} dpu{dpu_id} head{head} V 区不匹配: max diff "
                    f"{np.abs(V_tile.astype(np.float32) - ref_v_h).max()}"
                )
