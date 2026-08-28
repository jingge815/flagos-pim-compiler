"""真实 Llama-2-7B 整段 prefill 数值闭环（问题 1→2→3→6→7→8 端到端，验证 2）。

复用已各自验证过的问题 1→2（`test_spec_prop_llama2_7b.py`）、问题 3
（`test_comm_llama2_7b.py`）、问题 7/8（`test_mem_planner_llama2_7b.py`）
链路，第一次让**编排器**（问题 6）真正执行整张标注图产出的 `ExecutionPlan`
——不是像以前的测试那样手动摆数据验证单个算子/单个 block，而是
`build_execution_plan` 生成、`NumpyBackend.submit/wait` 真正跑一遍全部
7000+ 条命令，覆盖全部 32 层的 linear/add/mul/tanh DPU kernel、全部
redistribute 边（Megatron all_reduce、scatter、logits all_gather）、SDPA 的
KV 感知 handler（prefill 场景下 `valid_len` 从 0 起，重算注意力应与图内原生
SDPA 完全一致）。

判据：编排器算出的 logits 与 HF 单卡原生前向逐元素对齐（fp16 累积误差范围
内），且每个位置的贪心 argmax 完全一致——后者是"模型推理产出的下一个 token
预测"这一实际可观察行为层面的判据，比纯数值容差更贴近"推理是否正确"。
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
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, execute_plan, make_sdpa_handler
from runtime.kernels import register_all
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
SEQ_LEN = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2  # fp16

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


def _q_proj_layer_of(node) -> int:
    """从 SDPA 节点的 Q 输入回溯到对应层的 q_proj 权重，解出层号（静态信息，
    在 `build_execution_plan` 前算好，供 `host_handler_of` 闭包捕获）。"""
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


@pytest.fixture(scope="module")
def llama2_compiled_plan():
    """真实 7B：export → partition → propagate → 问题 3/7/8 → build_execution_plan（模块内一次）。"""
    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config

    input_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(_FixedMaskLlama(model), (input_ids, causal_mask), strict=True).module()

    partition_graph(gm)
    shard_config = llama_shard_config(
        NUM_DPUS, num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
    )
    edges = propagate_specs(gm, shard_config)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    (k_proj,) = [n for n in gm.graph.nodes if n.op == "get_attr" and "layers.0.self_attn.k_proj.weight" in n.target]
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
    nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, nodes, nodes, kv_specs, hw) for d in range(NUM_DPUS)}

    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)

    state = DecodeState(valid_len=0)
    sdpa_layer = {n.name: _q_proj_layer_of(n.args[0]) for n in gm.graph.nodes if "scaled_dot_product_attention" in str(n.target)}

    def host_handler_of(node):
        if "scaled_dot_product_attention" in str(node.target):
            return make_sdpa_handler(sdpa_layer[node.name], kv_specs, state, np.dtype(np.float16))
        return None

    compiled = build_execution_plan(
        nodes, gm, entries_by_id, pending, hardware=hardware,
        host_handler_of=host_handler_of,
    )
    return model, cfg, gm, input_ids, causal_mask, compiled, plans, state


@pytest.fixture(scope="module")
def llama2_executed(llama2_compiled_plan):
    """权重真实搬入各 DPU 的 MRAM，`execute_plan` 真正跑一遍整份 ExecutionPlan（模块内一次）。"""
    model, cfg, gm, input_ids, causal_mask, compiled, plans, state = llama2_compiled_plan
    hw_mram_bytes = 4 * 2**30

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw_mram_bytes))
    register_all(backend)

    by_target = {n.target: n for n in gm.graph.nodes if n.op == "get_attr"}
    for dpu_id in range(NUM_DPUS):
        for name, off in plans[dpu_id].weight.items():
            node = by_target[name]
            detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
            obj = gm
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

    events = execute_plan(compiled.plan, backend, values={"input_ids": input_ids, "causal_mask": causal_mask}, pos=state.valid_len)
    result = np.asarray(backend.wait(events[compiled.output_cmd_id]))

    ref = model(input_ids=input_ids, attention_mask=causal_mask, use_cache=False).logits
    return result, ref


def test_logits_match_torch_reference(llama2_executed) -> None:
    result, ref = llama2_executed
    diff = np.abs(result.astype(np.float32) - ref.numpy().astype(np.float32))
    assert diff.max() < 0.2  # fp16 在 32 层累积误差范围内
    assert diff.mean() < 0.02


def test_greedy_argmax_matches_at_every_position(llama2_executed) -> None:
    """比数值容差更贴近"推理是否正确"的判据：每个位置贪心解码选出的 token 一致。"""
    result, ref = llama2_executed
    ours = np.asarray(result).argmax(-1)
    theirs = ref.numpy().argmax(-1)
    assert np.array_equal(ours, theirs)
