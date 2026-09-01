"""遍历小型 Llama 的切分策略并验证推理和内存布局。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import run_decode_loop
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 4
DECODE_STEPS = 3
MAX_SEQ = 16

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128, num_layers=NUM_LAYERS,
)


@pytest.fixture(scope="module")
def small_llama():
    """创建固定随机种子的小型 Llama 模型。"""
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**MODEL_KWARGS)).eval().to(torch.float16)


@pytest.fixture(scope="module")
def hf_reference(small_llama):
    """生成所有策略共享的 HF 贪心解码参考。"""
    prompt = torch.arange(PREFILL_SEQ_LEN, dtype=torch.long)
    tokens: list[int] = []
    with torch.no_grad():
        out = small_llama(input_ids=prompt.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        tokens.append(nxt)
        for _ in range(DECODE_STEPS - 1):
            out = small_llama(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())
            tokens.append(nxt)
    return tokens


def _strategies():
    return llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)


def _hw_and_hardware():
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    return hw, hardware


@pytest.fixture(scope="module")
def compiled_by_strategy(small_llama):
    """按策略编译并缓存运行计划。"""
    hw, hardware = _hw_and_hardware()
    return {
        strategy.name: compile_llama2(
            small_llama, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
            max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        )
        for strategy in _strategies()
    }


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_every_strategy_decodes_the_same_tokens_as_single_card_pytorch(
    strategy, compiled_by_strategy, hf_reference
) -> None:
    """验证每种切分策略的解码 token 与 HF 参考一致。"""
    compiled = compiled_by_strategy[strategy.name]
    _, hardware = _hw_and_hardware()
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    compiled.state.valid_len = 0

    generated = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=torch.arange(PREFILL_SEQ_LEN, dtype=torch.long),
        max_new_tokens=DECODE_STEPS, eos_id=-1, state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )

    assert generated == hf_reference


def test_cross_stage_edge_count_grows_with_stage_count(compiled_by_strategy) -> None:
    """验证跨 stage 的通信边数量。"""
    counts = {}
    for name, compiled in compiled_by_strategy.items():
        cross = [
            e for e in compiled.prefill_edges
            if e.src_loc.get("device") == "dpu" and e.dst_loc.get("device") == "dpu"
            and set(e.src_loc["dpus"]) != set(e.dst_loc["dpus"])
        ]
        counts[name] = len(cross)
        assert len(cross) == compiled.strategy.num_stages - 1, (
            f"{name}: 跨 stage 边 {len(cross)} 条，期望 {compiled.strategy.num_stages - 1} 条"
        )
    assert counts["tp4_pp1"] == 0  # 张量并行下不该有任何跨 stage 搬运


def test_kv_region_holds_only_this_stage_layers(compiled_by_strategy) -> None:
    """结构判据：每台 DPU 的 KV 区只含本 stage 的层，且各 stage 的层不重不漏。"""
    for name, compiled in compiled_by_strategy.items():
        strategy = compiled.strategy
        per_stage = NUM_LAYERS // strategy.num_stages
        for dpu_id, spec in compiled.kv_specs.items():
            assert spec.layers == strategy.layers_of_dpu(dpu_id, NUM_LAYERS), name
            assert len(spec.layers) == per_stage, name
        # 所有流水段覆盖全部层。
        covered = {
            layer
            for dpu_id in strategy.dpu_ids
            for layer in compiled.kv_specs[dpu_id].layers
        }
        assert covered == set(range(NUM_LAYERS)), name


def test_kv_bytes_per_dpu_are_invariant_across_strategies(compiled_by_strategy) -> None:
    """验证各策略的单 DPU KV 占用相同。"""
    expected_cells = NUM_LAYERS * MODEL_KWARGS["num_key_value_heads"] // NUM_DPUS
    per_dpu_bytes = set()
    for name, compiled in compiled_by_strategy.items():
        for dpu_id, spec in compiled.kv_specs.items():
            assert len(spec.layers) * len(spec.kv_heads) == expected_cells, (
                f"{name} dpu{dpu_id}: {len(spec.layers)} 层 × {len(spec.kv_heads)} head "
                f"!= {expected_cells}"
            )
            per_dpu_bytes.add(spec.kv_allocated_bytes)

    assert len(per_dpu_bytes) == 1, f"单台 DPU 的 KV 占用随策略变化了: {per_dpu_bytes}"


def _weight_bytes_by_kind(compiled) -> dict[str, int]:
    """全机 DPU 侧权重字节数，按 placement 类别分开统计（fp16，2 字节/元素）。"""
    totals = {"Shard": 0, "Replicate": 0}
    for node in compiled.prefill_gm.graph.nodes:
        if node.op != "get_attr":
            continue
        spec = node.meta.get("spec")
        if spec is None or spec.device != "dpu":
            continue
        for detail in spec.shard_map.values():
            totals[spec.placement.kind] += int(np.prod(detail.local_shape)) * 2
    return totals


def test_sharded_weight_bytes_are_invariant_while_replication_shrinks(
    compiled_by_strategy,
) -> None:
    """验证分片权重总量不变且复制权重随并行宽度变化。"""
    shard_totals = {}
    for name, compiled in compiled_by_strategy.items():
        kinds = _weight_bytes_by_kind(compiled)
        shard_totals[name] = kinds["Shard"]
        # 复制字节数等于单份大小乘张量并行宽度。
        per_copy = kinds["Replicate"] // compiled.strategy.tp_width
        assert kinds["Replicate"] == per_copy * compiled.strategy.tp_width, name

    assert len(set(shard_totals.values())) == 1, f"被切权重的总量随策略变化了: {shard_totals}"


def test_replicated_weight_bytes_scale_with_tp_width(compiled_by_strategy) -> None:
    """同一份 RMSNorm 权重的复制开销严格正比于 tp_width（不是近似，是整除关系）。"""
    per_copy_bytes = set()
    for compiled in compiled_by_strategy.values():
        kinds = _weight_bytes_by_kind(compiled)
        per_copy_bytes.add(kinds["Replicate"] // compiled.strategy.tp_width)

    # 各策略的单份复制权重大小相同。
    assert len(per_copy_bytes) == 1, f"单份 Replicate 权重量不一致: {per_copy_bytes}"


def test_each_dpu_carries_a_balanced_share_of_weights(compiled_by_strategy) -> None:
    """验证各 DPU 的权重负载基本均衡。"""
    for name, compiled in compiled_by_strategy.items():
        per_dpu = {
            dpu_id: sum(
                int(np.prod(node.meta["spec"].shard_map[dpu_id].local_shape)) * 2
                for node in compiled.prefill_gm.graph.nodes
                if node.op == "get_attr"
                and node.meta.get("spec") is not None
                and node.meta["spec"].device == "dpu"
                and dpu_id in node.meta["spec"].shard_map
            )
            for dpu_id in compiled.strategy.dpu_ids
        }
        assert min(per_dpu.values()) > 0, f"{name}: 有 DPU 一份权重都没分到 {per_dpu}"
        assert max(per_dpu.values()) <= 2 * min(per_dpu.values()), f"{name}: {per_dpu}"
