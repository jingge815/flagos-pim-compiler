"""验证运行期窗口物化：与 bind_plan 逐字节等价、fail-fast 防线、4 请求 wave 对拍。"""

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
from runtime.binding import bind_plan
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import RequestTable, run_decode_loop, run_one_step
from runtime.kernels import register_all
from runtime.materialize import build_window_template, materialize

NUM_DPUS = 4
NUM_LAYERS = 4
PREFILL_SEQ_LEN = 8
MAX_SEQ = 16
NUM_SLOTS = 4
POOL = 8
MAX_NEW_TOKENS = 3

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128, num_layers=NUM_LAYERS,
)

PROMPTS = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16]]


def _greedy(logits):
    return int(np.argmax(logits))


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**MODEL_KWARGS)).eval().to(torch.float16)


@pytest.fixture(scope="module")
def hw_and_hardware():
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    return hw, hardware


def _compile(model, hw, hardware, **kwargs):
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    return compile_llama2(
        model, strategy, prefill_seq_len=PREFILL_SEQ_LEN,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        num_request_slots=NUM_SLOTS, kv_slot_tokens=MAX_SEQ, **kwargs)


@pytest.fixture(scope="module")
def dynamic_compiled(small_model, hw_and_hardware):
    """动态编译：窗口池 8、并发宽度默认 min(4, 8, 4)。"""
    hw, hardware = hw_and_hardware
    return _compile(small_model, hw, hardware, dynamic=True,
                    act_pool_slots=POOL)


@pytest.fixture(scope="module")
def static_compiled(small_model, hw_and_hardware):
    hw, hardware = hw_and_hardware
    return _compile(small_model, hw, hardware)


def _plan_signature(plan):
    """命令序列的可比较签名：payload 与访问三元组逐项相等。"""
    return [
        (cmd.id, cmd.op, cmd.dpu_id, cmd.payload,
         tuple((a.loc, a.offset, a.length) for a in cmd.reads),
         tuple((a.loc, a.offset, a.length) for a in cmd.writes),
         tuple(cmd.waits), cmd.num_tasklets)
        for cmd in plan.commands
    ]


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("window", list(range(POOL)))
def test_materialize_matches_bind_plan(dynamic_compiled, phase, window) -> None:
    """物化结果与编译期 bind_plan 平移逐字节一致——迁移正确性锚点。"""
    template = getattr(dynamic_compiled, f"{phase}_window_template")
    spec = template.spec
    bound = bind_plan(template.plan, act_base_by_dpu=spec.act_base_by_dpu,
                      act_shift=window * spec.act_stride, block_slot=window,
                      bt_stride=spec.bt_stride)
    got = materialize(template, window)
    assert _plan_signature(got) == _plan_signature(bound)


def test_window_zero_is_template_plan(dynamic_compiled) -> None:
    template = dynamic_compiled.decode_window_template
    assert materialize(template, 0) is template.plan


def test_materialize_cache_returns_same_plan(dynamic_compiled) -> None:
    template = dynamic_compiled.decode_window_template
    assert materialize(template, 3) is materialize(template, 3)


def test_materialize_rejects_out_of_pool(dynamic_compiled) -> None:
    template = dynamic_compiled.decode_window_template
    with pytest.raises(ValueError, match="超出激活窗口池"):
        materialize(template, template.spec.pool)
    with pytest.raises(ValueError, match="超出激活窗口池"):
        materialize(template, -1)


def test_materialize_rejects_overflowing_geometry(dynamic_compiled) -> None:
    """步长被破坏（act_stride=0）时区间越池必须抛错，不许静默错址。"""
    template = dynamic_compiled.decode_window_template
    shrunk = build_window_template(
        template.plan, act_base_by_dpu=template.spec.act_base_by_dpu,
        act_stride=0, bt_stride=template.spec.bt_stride, pool=2)
    with pytest.raises(ValueError, match="越出激活窗口池"):
        materialize(shrunk, 1)


def test_dynamic_compile_records_pool_and_width(dynamic_compiled) -> None:
    compiled = dynamic_compiled
    assert compiled.compile_mode == "dynamic"
    assert compiled.num_parallel_slots == 4
    assert compiled.decode_window_template.spec.pool == POOL
    assert compiled.prefill_window_template is not None
    # 编译产物只留窗口 0：动态路径不再展开多窗口。
    assert len(compiled.decode_plans) == 1
    assert len(compiled.prefill_plans[compiled.prefill_seq_len]) == 1


def test_static_compile_keeps_window_expansion(static_compiled) -> None:
    """静态路径行为不变：模板为空、窗口表按并行数展开、默认宽度 2。"""
    compiled = static_compiled
    assert compiled.compile_mode == "static"
    assert compiled.decode_window_template is None
    assert compiled.prefill_window_template is None
    assert compiled.num_parallel_slots == 2
    assert len(compiled.decode_plans) == compiled.num_parallel_slots
    assert len(compiled.prefill_plans[compiled.prefill_seq_len]) \
        == compiled.num_parallel_slots


def test_static_rejects_act_pool_slots(static_compiled, small_model,
                                       hw_and_hardware) -> None:
    hw, hardware = hw_and_hardware
    with pytest.raises(ValueError, match="act_pool_slots"):
        _compile(small_model, hw, hardware, act_pool_slots=POOL)


def test_env_overrides_pool(small_model, hw_and_hardware, monkeypatch) -> None:
    monkeypatch.setenv("PIM_ACT_POOL_SLOTS", "5")
    hw, hardware = hw_and_hardware
    compiled = _compile(small_model, hw, hardware, dynamic=True)
    assert compiled.decode_window_template.spec.pool == 5
    assert compiled.num_parallel_slots == 4


def _backend_with_weights(compiled, hardware) -> NumpyBackend:
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=hardware.mram_bytes_per_dpu,
        num_parallel_slots=compiled.num_parallel_slots,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    return backend


def _reference(compiled, backend, prompt, max_new_tokens=MAX_NEW_TOKENS):
    return run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor(prompt),
        max_new_tokens=max_new_tokens, eos_id=-1,
        sample_fn=_greedy, causal_mask_of=causal_mask_of,
    )


def test_four_requests_dynamic_wave_matches_single_runs(
        dynamic_compiled, hw_and_hardware) -> None:
    """4 个变长 prompt 同一 wave 并发，生成结果分别等于单独运行。"""
    hw, hardware = hw_and_hardware
    compiled = dynamic_compiled
    backend = _backend_with_weights(compiled, hardware)
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    for index, prompt in enumerate(PROMPTS):
        table.add_request(f"r{index}", prompt, eos_id=-1)

    for _ in range(MAX_NEW_TOKENS):
        run_one_step(compiled, backend, table, causal_mask_of, sample_fn=_greedy)

    for index, prompt in enumerate(PROMPTS):
        got = list(table.get(f"r{index}").generated)
        assert got == _reference(compiled, backend, prompt), (
            f"r{index} 并发生成 {got} 与单独运行不一致")
