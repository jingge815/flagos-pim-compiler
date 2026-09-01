"""验证编译线性内核在 Llama-2-7B 解码中的数值和生成结果。"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig
from genesim_bridge.paths import llama2_7b_model_dir
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.compile import sdpa_layer_map, write_weight_shards
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, make_sdpa_handler, run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
PROMPT = "The capital of France is"
DECODE_STEPS = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = [
    pytest.mark.skipif(
        MODEL_DIR is None or not MODEL_DIR.is_dir(),
        reason="需要在 paths.json 配置 llama2_7b_model_dir",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="opcompiler_bridge 编译期需要 GPU"),
]


class _PositionalLlama(torch.nn.Module):
    """将 RoPE 位置作为图输入的 Llama 包装。"""

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



def _wrap_with_numpy_cross_check(orig_compiled_linear_kernel):
    """包装已编译内核并记录其与 NumPy 内核的相对误差。"""
    import runtime.kernels as km

    stats: list[tuple[tuple, str, float]] = []

    def wrapped(hal, dpu_id, cmd):
        shapes = tuple(tuple(s) for s in cmd.payload["arg_shapes"])
        dtype = str(cmd.payload["dtype"])
        if not km._compiled_linear_supports(shapes, dtype):
            return km.linear_kernel(hal, dpu_id, cmd)

        npdt = np.dtype(dtype)
        x_a, w_a = cmd.reads
        (out_a,) = cmd.writes

        # 两次计算使用相同的原始输入。
        x_backup = hal.read_local(dpu_id, x_a.offset, (x_a.length // npdt.itemsize,), npdt).copy()
        w_backup = hal.read_local(dpu_id, w_a.offset, (w_a.length // npdt.itemsize,), npdt).copy()

        km.linear_kernel(hal, dpu_id, cmd)
        ref = hal.read_local(dpu_id, out_a.offset, tuple(cmd.payload["out_shape"]), npdt).copy()

        hal.write_local(dpu_id, x_a.offset, x_backup)
        hal.write_local(dpu_id, w_a.offset, w_backup)
        orig_compiled_linear_kernel(hal, dpu_id, cmd)
        got = hal.read_local(dpu_id, out_a.offset, tuple(cmd.payload["out_shape"]), npdt)

        r, g = ref.astype(np.float32), got.astype(np.float32)
        rel = float(np.abs(g - r).max() / max(np.abs(r).max(), 1e-6))
        stats.append((shapes, dtype, rel))

        # 使用 NumPy 结果继续生成后续 token。
        hal.write_local(dpu_id, out_a.offset, np.ascontiguousarray(ref, dtype=npdt))

    return wrapped, stats


@pytest.mark.parametrize("num_tasklets", [4])
def test_compiled_linear_end_to_end_matches_hf_generate(monkeypatch, num_tasklets) -> None:
    """验证已编译线性内核的数值和完整生成文本。"""
    import runtime.kernels as km

    wrapped, stats = _wrap_with_numpy_cross_check(km.compiled_linear_kernel)
    monkeypatch.setattr(km, "compiled_linear_kernel", wrapped)

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config

    prompt_ids_list = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    prefill_seq_len = len(prompt_ids_list)

    prefill_gm = _export_graph(model, prefill_seq_len, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0))
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
        num_tasklets=num_tasklets,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        # DMA 分块使用独立于 MRAM 布局的字节对齐。
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
        num_tasklets=num_tasklets,
    )
    decode_compiled = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware,
        host_handler_of=make_host_handler(sdpa_layer_map(decode_gm)),
        num_tasklets=num_tasklets,
    )

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))
    register_all(backend, use_compiled_linear=True)

    write_weight_shards(prefill_gm, plans, backend)

    def greedy(logits_1d):
        return int(np.argmax(logits_1d))

    prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)
    generated = run_decode_loop(
        prefill_compiled.plan, decode_compiled.plan, backend,
        prompt_ids=prompt_ids, max_new_tokens=DECODE_STEPS, eos_id=tokenizer.eos_token_id, state=state,
        sample_fn=greedy, prefill_output_cmd_id=prefill_compiled.output_cmd_id,
        decode_output_cmd_id=decode_compiled.output_cmd_id, causal_mask_of=_causal_mask_of,
    )

    our_ids = prompt_ids_list + generated
    our_text = tokenizer.decode(our_ids, skip_special_tokens=True)

    ref_ids = model.generate(
        input_ids=torch.tensor([prompt_ids_list], dtype=torch.long),
        max_new_tokens=DECODE_STEPS, do_sample=False,
    )[0].tolist()
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

    per = defaultdict(list)
    for shapes, dtype, rel in stats:
        per[(shapes, dtype)].append(rel)
    print(f"\n=== 编译产物逐次对拍统计：共 {len(stats)} 次调用 ===")
    for (shapes, dtype), rels in per.items():
        a = np.array(rels)
        print(
            f"  shapes={shapes} dtype={dtype}: n={len(a)} "
            f"max_rel={a.max():.4e} mean_rel={a.mean():.4e} "
            f"超 5% 容差次数={int((a > 0.05).sum())}"
        )

    print(f"\nprompt: {PROMPT!r}")
    print(f"generated (our orchestrator, 含编译产物): {our_text!r}")
    print(f"generated (HF model.generate): {ref_text!r}")

    assert len(stats) > 0, "没有任何调用走到编译产物路径——检查 shape 是否满足 2 的幂约束"
    for (shapes, dtype), rels in per.items():
        assert max(rels) < 0.05, f"{shapes} ({dtype}) 编译产物与 NumPy 参考值偏差过大: max_rel={max(rels):.4e}"

    assert our_ids == ref_ids
    assert our_text == ref_text
    assert our_text.startswith(PROMPT)
