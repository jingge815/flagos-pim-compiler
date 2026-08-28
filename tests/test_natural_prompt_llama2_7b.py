"""真实自然语言 prompt 的端到端验证：输入可读文本、走完整编排器、输出可读文本。

`tests/test_decode_loop_llama2_7b.py`（验证 3）用的 prompt 是
`torch.arange(16)` 的合成 token 序列，只在 token id 层面比对，从未验证过
"喂真实提示词、通过真实 tokenizer、解码出可读文本"这条链路——用户明确要求
这一点必须单独确认，本文件补上。

判据：`tokenizer.decode(编排器产出的 token 序列)` 与 HF 官方
`model.generate()`（相同贪心策略、相同真实提示词）解码出的文本逐字符一致；
同时打印生成文本供人工阅读确认"输出正常的推理内容"。
"""

from __future__ import annotations

import sys
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
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, make_sdpa_handler, run_decode_loop
from runtime.kernels import register_all

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
PROMPT = "The capital of France is"
DECODE_STEPS = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


class _PositionalLlama(torch.nn.Module):
    """RoPE 位置显式作为图输入（问题 6 decode 循环专用，见
    `tests/test_decode_loop_llama2_7b.py` 模块 docstring 的说明）。"""

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


def _q_proj_layer_of(node) -> int:
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "get_attr" and "q_proj.weight" in str(cur.target):
            return int(str(cur.target).split(".")[3])
        stack.extend(a for a in cur.args if hasattr(a, "name"))
    raise ValueError(f"未能从 {node.name} 回溯到 q_proj 权重")


def test_real_prompt_produces_readable_text_matching_hf_generate() -> None:
    """输入真实英文提示词、走完整问题 1→2→3→6→7→8 编排器，输出可读文本，
    与 HF `model.generate()`（同样贪心策略）逐 token、逐字符一致。
    """
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
        k_proj.meta[SPEC_META_KEY], num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
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

    def sdpa_layer_map(gm):
        return {n.name: _q_proj_layer_of(n.args[0]) for n in gm.graph.nodes if "scaled_dot_product_attention" in str(n.target)}

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

    by_target = {n.target: n for n in prefill_gm.graph.nodes if n.op == "get_attr"}
    for dpu_id in range(NUM_DPUS):
        for name, off in plans[dpu_id].weight.items():
            node = by_target[name]
            detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
            obj = prefill_gm
            for part in name.split("."):
                obj = getattr(obj, part)
            obj = obj.detach()
            if detail.shard_dim == 0:
                local = obj[detail.start_idx:detail.end_idx].numpy()
            elif detail.shard_dim == 1:
                local = obj[:, detail.start_idx:detail.end_idx].numpy()
            else:
                local = obj.numpy()
            backend.write_local(dpu_id, off, np.ascontiguousarray(local))

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

    print(f"\nprompt: {PROMPT!r}")
    print(f"generated (our orchestrator): {our_text!r}")
    print(f"generated (HF model.generate): {ref_text!r}")

    assert our_ids == ref_ids
    assert our_text == ref_text
    assert our_text.startswith(PROMPT)
