"""MoE 专家并行（P1-b，PP×EP）：放置、host 聚合与端到端对拍。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from backend.architectures import ARCHITECTURES, make_static_moe_forward

NUM_LAYERS = 4
NUM_EXPERTS = 4
MODEL_KWARGS = dict(
    max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
    num_hidden_layers=NUM_LAYERS, num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, vocab_size=128, num_experts=NUM_EXPERTS, num_experts_per_tok=2,
    norm_topk_prob=True,
)


def _tiny_moe(seed: int = 6) -> torch.nn.Module:
    """静态化的小 MoE 模型。"""
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**MODEL_KWARGS)).eval().to(torch.float16)
    for layer in model.model.layers:
        layer.mlp.forward = make_static_moe_forward(layer.mlp)
    return model


@pytest.fixture(scope="module")
def ep_model_dir(tmp_path_factory):
    """存盘后按注册表路径加载（口径与其余架构用例一致）。"""
    model = _tiny_moe()
    path = tmp_path_factory.mktemp("qwen3_moe_ep")
    model.save_pretrained(path)
    return model, path


def test_ep_weight_specs_and_host_aggregation(ep_model_dir) -> None:
    """PP×EP：专家权重单 DPU 整块轮转；attention REPLICATE 段内；累加链改标 host。"""
    from contracts.graph_meta import DEVICE_HOST, DEVICE_META_KEY, SPEC_META_KEY
    from graph.spec_prop import propagate_specs, remark_moe_aggregation_for_ep
    from graph.strategy import ShardStrategy
    from runtime.compile import export_annotated_graph

    model, _ = ep_model_dir
    strategy = ShardStrategy(
        name="tp1_pp2_ep2", num_dpus=4, num_stages=2, ep_width=2,
        weight_rules=ARCHITECTURES["qwen3_moe"].strategy(4, model.config).weight_rules,
    )
    gm = export_annotated_graph(
        model, 4, torch.arange(4, dtype=torch.long).unsqueeze(0))
    remark_moe_aggregation_for_ep(gm)
    propagate_specs(gm, strategy)

    def spec_of(marker: str):
        for node in gm.graph.nodes:
            if node.op == "get_attr" and marker in str(node.target):
                return node.meta[SPEC_META_KEY]
        raise AssertionError(f"找不到权重 {marker}")

    # 层 0 在段 0（dpu 0,1）：专家 e → dpu[e % 2]，单 DPU 整块。
    for expert, dpu in ((0, 0), (1, 1), (2, 0), (3, 1)):
        spec = spec_of(f"layers.0.mlp.experts.{expert}.gate_proj.weight")
        assert spec.placement.kind == "Replicate", expert
        assert set(spec.shard_map) == {dpu}, f"expert{expert}: {set(spec.shard_map)}"
    # 层 2 在段 1（dpu 2,3）：4 层 ÷ 2 段，层 0-1 在段 0、层 2-3 在段 1。
    spec = spec_of("layers.2.mlp.experts.0.gate_proj.weight")
    assert set(spec.shard_map) == {2}
    # attention 与 router：段内 REPLICATE。
    assert set(spec_of("layers.0.self_attn.q_proj.weight").shard_map) == {0, 1}
    assert set(spec_of("layers.0.mlp.gate.weight").shard_map) == {0, 1}
    # MoE 累加链（带 experts 祖先的 mul/add）改标 host。
    host_agg = [
        n for n in gm.graph.nodes
        if n.op == "call_function"
        and str(n.target) in ("aten.mul.Tensor", "aten.add.Tensor")
        and n.meta[DEVICE_META_KEY] == DEVICE_HOST
    ]
    assert host_agg, "累加链应改标 host"


def test_ep_16_dpu_generate_matches_hf(ep_model_dir) -> None:
    """16 DPU pp4×ep4：贪心生成与 HF 逐 token 一致，专家映射均衡。"""
    from collections import Counter

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import ShardStrategy
    from memory.mem_planner import HwBudget
    from runtime.compile import causal_mask_of, compile_llama2, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    model, path = ep_model_dir
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ShardStrategy(
        name="tp1_pp4_ep4", num_dpus=16, num_stages=4, ep_width=4,
        weight_rules=ARCHITECTURES["qwen3_moe"].strategy(4, loaded.config).weight_rules,
    )
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=PIMHardwareConfig(
            num_dpus=16, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
            wram_bytes_per_dpu=65536, dma_align=64),
        num_request_slots=1, num_parallel_slots=1)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=16, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)

    got = run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
        max_new_tokens=3, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)

    torch.manual_seed(6)
    ref_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**MODEL_KWARGS)).eval().to(torch.float16)
    ref = ref_model.generate(torch.tensor([[1, 2, 3, 4]]),
                             max_new_tokens=3, do_sample=False)
    assert got == ref[0][4:].tolist()

    # 专家映射均衡：每层 4 专家轮转到段内 4 台 DPU，各 1 个，差 ≤ 1。
    for layer in range(NUM_LAYERS):
        counts = Counter(
            strategy.ep_dpu_of_expert(layer, e, NUM_LAYERS)
            for e in range(NUM_EXPERTS))
        assert max(counts.values()) - min(counts.values()) <= 1


def test_dpu_to_host_reads_are_war_protected(ep_model_dir) -> None:
    """dpu→host 重分布的读区必须登记在（消费者名, 读取 DPU）上：激活区复用时，
    后写者要 WAR 等它——否则 gather 可能读到被覆盖的旧值（K3 竞态）。"""
    from collections.abc import Iterable

    from contracts.graph_meta import DEVICE_HOST
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import ShardStrategy
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2

    model, path = ep_model_dir
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ShardStrategy(
        name="pp2ep4", num_dpus=8, num_stages=2, ep_width=4,
        weight_rules=ARCHITECTURES["qwen3_moe"].strategy(4, loaded.config).weight_rules,
    )
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=PIMHardwareConfig(
            num_dpus=8, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
            wram_bytes_per_dpu=65536, dma_align=64),
        num_request_slots=1, num_parallel_slots=1)

    for plan_obj in compiled.prefill_plans[4]:
        cmds = plan_obj if isinstance(plan_obj, Iterable) and hasattr(plan_obj, "__iter__") and not hasattr(plan_obj, "commands") else plan_obj.commands
        edges = {e.edge_id: e for e in compiled.prefill_edges}
        by_id = {c.id: c for c in cmds}
        gathers = [c for c in cmds
                   if c.op in ("host_concat", "host_reduce")
                   and c.payload.get("edge_id") in edges
                   and edges[c.payload["edge_id"]].dst_loc.get("device") == DEVICE_HOST]
        assert gathers, "应存在 dpu→host 收集命令"
        violations = []
        for g in gathers:
            edge = edges[g.payload["edge_id"]]
            for r in g.reads:
                for w in cmds:
                    if w.id <= g.id or w.op == "host_op":
                        continue
                    for wa in w.writes:
                        if wa.loc != r.loc or (wa.offset + wa.length <= r.offset
                                               or r.offset + r.length <= wa.offset):
                            continue
                        # 同 DPU 后写者必须显式等 gather（gather 是 host 命令，不在 DPU 泳道链上）。
                        if g.id not in w.waits:
                            violations.append(
                                f"cmd{w.id}({w.payload.get('node')}) 重写 {r.loc}@{r.offset} "
                                f"未等 gather cmd{g.id}({edge.src}->{edge.dst})")
        assert not violations, "\n".join(violations[:5])
