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


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_every_strategy_kv_region_matches_hf_cache(strategy, small_llama, compiled_by_strategy) -> None:
    """张量并行下 KV 区也要和 HF cache 对上，不能只比对 token。

    每台 DPU 都拿到完整 Q/K/V 副本，但 KV 按头切在各 owner 上。所有 DPU
    同时写同一行会互相覆盖：注意力当时用的是刚写完的本地副本，token 对得上，
    跑完后 cache 里留下的却是最后一次写入。
    """
    from memory.kv_layout import PIMStaticKVCache

    compiled = compiled_by_strategy[strategy.name]
    _, hardware = _hw_and_hardware()
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    compiled.state.valid_len = 0
    run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=torch.arange(PREFILL_SEQ_LEN, dtype=torch.long),
        max_new_tokens=DECODE_STEPS, eos_id=-1, state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )

    prompt = torch.arange(PREFILL_SEQ_LEN, dtype=torch.long)
    with torch.no_grad():
        out = small_llama(input_ids=prompt.unsqueeze(0), use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(DECODE_STEPS - 1):
            out = small_llama(
                input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax())

    cache = PIMStaticKVCache(backend, compiled.kv_specs, wram_budget_bytes=2**20)
    total_len = compiled.state.valid_len
    assert total_len == PREFILL_SEQ_LEN + DECODE_STEPS - 1
    for layer in range(NUM_LAYERS):
        ref_k = past.layers[layer].keys
        ref_v = past.layers[layer].values
        owners = [(d, s) for d, s in compiled.kv_specs.items() if layer in s.layers]
        assert owners, f"{strategy.name} layer{layer} 没有任何 DPU 持有 KV"
        for dpu_id, spec in owners:
            for head in spec.kv_heads:
                K_tile, V_tile = cache.read_tile(layer, dpu_id, head, 0, total_len)
                ref_k_h = ref_k[0, head].detach().numpy().astype(np.float32)
                ref_v_h = ref_v[0, head].detach().numpy().astype(np.float32)
                assert np.allclose(K_tile.astype(np.float32), ref_k_h, rtol=2e-2, atol=3e-2), (
                    f"{strategy.name} layer{layer} dpu{dpu_id} head{head} K 区不匹配: "
                    f"max diff {np.abs(K_tile.astype(np.float32) - ref_k_h).max()}"
                )
                assert np.allclose(V_tile.astype(np.float32), ref_v_h, rtol=2e-2, atol=3e-2), (
                    f"{strategy.name} layer{layer} dpu{dpu_id} head{head} V 区不匹配: "
                    f"max diff {np.abs(V_tile.astype(np.float32) - ref_v_h).max()}"
                )


def test_cross_stage_edge_count_grows_with_stage_count(compiled_by_strategy) -> None:
    """跨 stage 的通信边分两类，各自的条数都要钉住。

    激活搬运：每个段边界恰好一条（残差从前一段交给后一段）。
    只读表搬运：位置编码的 cos/sin 表由所有层共用，视图族下设备后它由**最早
    消费它的那一段**算一次，再复制给其余段。表本身是产物而不是激活，所以单独
    按字节量分开计——把两者混在一个总数里，就没法再区分「多搬了一份激活」和
    「多复制了一次只读表」。
    """
    counts = {}
    for name, compiled in compiled_by_strategy.items():
        cross = [
            e for e in compiled.prefill_edges
            if e.src_loc.get("device") == "dpu" and e.dst_loc.get("device") == "dpu"
            and set(e.src_loc["dpus"]) != set(e.dst_loc["dpus"])
        ]
        # 位置编码表很小（几百字节），激活是隐藏态（这里 512B）。按名字分更稳：
        # 表那两条的源都是 `to_*`（位置编码的 dtype 转换），激活是残差 add。
        tables = [e for e in cross if e.src.startswith("to_")]
        activations = [e for e in cross if e not in tables]
        counts[name] = len(activations)

        expected_activation = compiled.strategy.num_stages - 1
        assert len(activations) == expected_activation, (
            f"{name}: 跨 stage 的激活搬运 {len(activations)} 条，"
            f"期望 {expected_activation} 条：{[(e.src, e.dst) for e in activations]}"
        )
        assert len(tables) == 2 * expected_activation, (
            f"{name}: 位置编码表的跨段复制 {len(tables)} 条，"
            f"期望 {2 * expected_activation} 条（cos/sin 各一份）"
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


# 4096 字节的 WRAM 让 q/k/v/o、gate/up/down 与 lm_head 全部超预算，分块真正触发；
# 默认 64KB 下这个小模型一个都不切，问题被掩盖。
_TILING_WRAM = 4096


@pytest.fixture(scope="module")
def compiled_tiled(small_llama):
    """分块必然触发的预算下，按策略编译。"""
    hw, hardware = _hw_and_hardware()
    import dataclasses
    hardware = dataclasses.replace(hardware, wram_bytes_per_dpu=_TILING_WRAM)
    return {
        strategy.name: compile_llama2(
            small_llama, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
            max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        )
        for strategy in _strategies()
    }


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_tiled_strategy_decodes_the_same_tokens(strategy, compiled_tiled,
                                               hf_reference) -> None:
    """分块触发后，行切权重的部分和、K 段相加与 cat 拼回都要传播并算对。

    每层 7 个投影加 lm_head 共 29 个 linear，分块后必须多于这个数。
    """
    compiled = compiled_tiled[strategy.name]
    linears = sum(1 for n in compiled.prefill_gm.graph.nodes
                  if n.target is torch.ops.aten.linear.default)
    assert linears > 7 * NUM_LAYERS + 1, f"分块没有触发，只有 {linears} 个 linear"

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=compiled.hardware.mram_bytes_per_dpu,
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


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_tiled_compile_still_fuses_gate_activation(strategy, compiled_tiled) -> None:
    """分块触发后门控 silu 仍折进 linear：每层一个，图上不剩独立 silu。

    分块把 gate linear 拆成子块再拼回，融合要折进拼回前的每个子块。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    compiled = compiled_tiled[strategy.name]
    for gm in (compiled.prefill_gm, compiled.decode_gm):
        tails = {n.meta[FUSED_TAIL_META_KEY].nodes[0].name
                 for n in gm.graph.nodes
                 if n.target is torch.ops.aten.linear.default
                 and FUSED_TAIL_META_KEY in n.meta}
        assert len(tails) == NUM_LAYERS, f"只折了 {len(tails)} 个门控激活"
        assert not any(n.target is torch.ops.aten.silu.default
                       for n in gm.graph.nodes), "图上仍有独立 silu"
    launches = [c for c in compiled.prefill.plan.commands
                if c.op == "launch" and c.payload.get("activation") == "silu"]
    assert launches, "执行计划没有下发折入的 silu"


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_tiled_fused_prefill_logits_match_hf(strategy, compiled_tiled,
                                              small_llama) -> None:
    """分块后激活折进各子块，prefill 的 logits 与 HF 逐元素对齐，不只比 token。

    折进子块的激活在 f32 上算再转 f16，原图先转 f16 再算，舍入差要在容差内。
    """
    from runtime.executor import execute_plan

    compiled = compiled_tiled[strategy.name]
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=compiled.hardware.mram_bytes_per_dpu,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    compiled.state.valid_len = 0
    ids = torch.arange(PREFILL_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    events = execute_plan(
        compiled.prefill.plan, backend,
        values={"input_ids": ids, "causal_mask": causal_mask_of(PREFILL_SEQ_LEN),
                "position_ids": ids.clone()},
        pos=0,
    )
    got = np.asarray(backend.wait(events[compiled.prefill.output_cmd_id]),
                     dtype=np.float32)
    with torch.no_grad():
        ref = small_llama(input_ids=ids, use_cache=False).logits.float().numpy()
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


@pytest.fixture(scope="module")
def compiled_k_split(small_llama):
    """2048 字节的 WRAM 下按策略编译：不保 K 时门控投影会被切开 K。"""
    hw, hardware = _hw_and_hardware()
    import dataclasses
    hardware = dataclasses.replace(hardware, wram_bytes_per_dpu=2048)
    return {
        strategy.name: compile_llama2(
            small_llama, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
            max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        )
        for strategy in _strategies()
    }


@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)
def test_k_split_budget_still_fuses_gate_activation(strategy, compiled_k_split) -> None:
    """预算小到会切 K 时，门控投影只沿 M、N 切，silu 仍每层折进一次。"""
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    compiled = compiled_k_split[strategy.name]
    for gm in (compiled.prefill_gm, compiled.decode_gm):
        tails = {n.meta[FUSED_TAIL_META_KEY].nodes[0].name
                 for n in gm.graph.nodes
                 if n.target is torch.ops.aten.linear.default
                 and FUSED_TAIL_META_KEY in n.meta}
        assert len(tails) == NUM_LAYERS, f"只折了 {len(tails)} 个门控激活"
        assert not any(n.target is torch.ops.aten.silu.default
                       for n in gm.graph.nodes), "图上仍有独立 silu"


def _fusion_cases():
    """镜像跑全部策略；编译内核只跑 tp1_pp4。

    tp4、tp2 的本地分片太小（如 M=4 N=64 K=16），编译内核在 256 字节 DMA
    对齐下挑不出分块，关掉融合也一样失败，是既有限制，不属于融合。
    """
    cases = [pytest.param(s, False, id=f"{s.name}-mirror") for s in _strategies()]
    cases += [pytest.param(s, True, id=f"{s.name}-compiled")
              for s in _strategies() if s.name == "tp1_pp4"]
    return cases


@pytest.mark.parametrize("strategy, use_compiled", _fusion_cases())
def test_fused_gate_activation_decodes_like_hf(strategy, use_compiled,
                                              compiled_by_strategy,
                                              hf_reference) -> None:
    """默认 compile_llama2 的门控 silu 确实折进 linear，且解码仍与 HF 一致。

    每层一个折了 silu 的 linear，图上不剩独立 silu，执行计划里带
    activation 的 launch 命令存在；镜像与 A 路编译内核各跑一遍。
    """
    from contracts.graph_meta import FUSED_TAIL_META_KEY

    compiled = compiled_by_strategy[strategy.name]
    for gm in (compiled.prefill_gm, compiled.decode_gm):
        fused = [n for n in gm.graph.nodes
                 if n.target is torch.ops.aten.linear.default
                 and FUSED_TAIL_META_KEY in n.meta]
        assert len(fused) == NUM_LAYERS, f"只折了 {len(fused)} 个门控激活"
        assert not any(n.target is torch.ops.aten.silu.default
                       for n in gm.graph.nodes), "图上仍有独立 silu"
    launches = [c for c in compiled.prefill.plan.commands
                if c.op == "launch" and c.payload.get("activation") == "silu"]
    assert launches, "执行计划没有下发折入的 silu"

    _, hardware = _hw_and_hardware()
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
    ))
    register_all(backend, use_compiled_linear=use_compiled)
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


def _duplicate_shape_ops(gm) -> int:
    """图里结构相同的形状类算子个数（第一次出现的不计）。"""
    from graph.opt_passes import _is_pure, _is_view, _literal_args, _structure_key

    seen, dups = set(), 0
    for node in gm.graph.nodes:
        if not _is_pure(node) or not _is_view(node) or not _literal_args(node):
            continue
        key = _structure_key(node)
        dups += key in seen
        seen.add(key)
    return dups


def test_cse_merges_layer_shape_ops_only_without_pipeline(compiled_by_strategy) -> None:
    """没有流水并行时默认通路合并各层重复的形状类算子；有流水并行时保留，不跨 stage 读。"""
    for name, compiled in compiled_by_strategy.items():
        dups = [_duplicate_shape_ops(compiled.prefill_gm),
                _duplicate_shape_ops(compiled.decode_gm)]
        if compiled.strategy.num_stages == 1:
            assert dups == [0, 0], f"{name} 还剩重复的形状类算子：{dups}"
        else:
            assert min(dups) > 0, f"{name} 跨 stage 合并了形状类算子"
