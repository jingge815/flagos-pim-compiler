"""验证 D2-lite：单张符号 T 的 prefill 图与静态基线逐 token 一致。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from comm.plan import t_axes_from_exprs
from contracts.runtime_graph import ShapeExpr
from contracts.unified_ir import SHAPE_EXPR_META_KEY, VAL_META_KEY
from contracts.op_contract import PIMHardwareConfig
from graph.strategy import llama_strategies
from memory.mem_planner import HwBudget
from runtime.compile import (
    causal_mask_of,
    compile_llama2,
    load_weights,
)
from runtime.executor import RequestTable, step_request
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 4
MAX_SEQ = 16
NUM_SLOTS = 2
MAX_NEW_TOKENS = 3

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
)
STRATEGY_KWARGS = dict(
    num_heads=4, num_kv_heads=4, intermediate_size=128, vocab_size=128,
    num_layers=NUM_LAYERS,
)


def _compile(model, dynamic: bool):
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64,
    )
    strategy = llama_strategies(NUM_DPUS, **STRATEGY_KWARGS)[0]
    return compile_llama2(
        model, strategy, prefill_seq_len=MAX_SEQ,
        max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        num_request_slots=NUM_SLOTS, kv_slot_tokens=MAX_SEQ,
        dynamic=dynamic,
    )


def _backend(compiled) -> NumpyBackend:
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=compiled.hardware.mram_bytes_per_dpu,
        num_parallel_slots=compiled.num_parallel_slots,
    ))
    register_all(backend)
    load_weights(compiled, backend)
    return backend


def _run(compiled, backend, prompt: list[int]) -> list[int]:
    table = RequestTable(NUM_SLOTS, compiled.kv_pool_slots)
    slot = table.add_request("r", prompt)
    return [step_request(compiled, backend, slot, causal_mask_of)
            for _ in range(MAX_NEW_TOKENS)]


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**MODEL_KWARGS)).eval().to(torch.float16)


@pytest.fixture(scope="module")
def baseline(small_model):
    compiled = _compile(small_model, dynamic=False)
    return compiled, _backend(compiled)


@pytest.fixture(scope="module")
def dynamic(small_model):
    compiled = _compile(small_model, dynamic=True)
    return compiled, _backend(compiled)


def test_dynamic_compile_records_mode_and_single_bucket(dynamic) -> None:
    compiled, _ = dynamic
    assert compiled.compile_mode == "dynamic"
    assert compiled.dynamic_axes == {"T": (1, MAX_SEQ)}
    assert compiled.prefill_seq_lens == (MAX_SEQ,)


def test_dynamic_export_concretizes_meta_and_keeps_exprs(dynamic) -> None:
    """规划用的 val 形状全为整数；符号维留的是真正的 T 表达式。"""
    compiled, _ = dynamic
    symbolic = 0
    dim_exprs = 0
    for node in compiled.prefill_gm.graph.nodes:
        val = node.meta.get(VAL_META_KEY)
        if not isinstance(val, torch.Tensor):
            continue
        assert all(isinstance(d, int) for d in val.shape), node.name
        exprs = node.meta.get(SHAPE_EXPR_META_KEY)
        if exprs is None:
            continue
        symbolic += 1
        assert len(exprs) == len(val.shape)
        for expr, dim in zip(exprs, val.shape):
            if expr is not None:
                assert expr.eval({"T": MAX_SEQ}) == dim
                dim_exprs += expr.kind == "dim"
    assert symbolic > 0, "动态导出应至少在一个节点上记录符号维"
    assert dim_exprs > 0, "符号维必须记成 dim(T)，不能退化成包络常量"


def test_dynamic_plan_carries_shape_exprs(dynamic) -> None:
    compiled, _ = dynamic
    carriers = [
        cmd for plan in compiled.prefill_plans[MAX_SEQ]
        for cmd in plan.commands if "out_shape_expr" in cmd.payload
    ]
    assert carriers, "动态计划的 launch 命令应携带形状表达式"
    dims = [entry for entries in
            (cmd.payload["out_shape_expr"] for cmd in carriers)
            for entry in entries if entry is not None and entry["kind"] == "dim"]
    assert dims, "至少一个输出维是 dim(T) 表达式"
    expr = ShapeExpr.from_dict(dims[0])
    assert expr.eval({"T": 5}) == 5


def test_dynamic_export_annotates_shape_list_args(dynamic) -> None:
    """view/reshape 的形状列表记下符号元素，并能在运行期改成实际 T。"""
    from contracts.unified_ir import ARG_EXPR_META_KEY

    compiled, _ = dynamic
    annotated = []
    for node in compiled.prefill_gm.graph.nodes:
        ann = node.meta.get(ARG_EXPR_META_KEY)
        if ann is None:
            continue
        for arg, entry in zip(node.args, ann):
            if isinstance(arg, list) and isinstance(entry, list):
                for literal, expr in zip(arg, entry):
                    if expr is not None:
                        assert expr.kind == "dim"
                        annotated.append((literal, expr.eval({"T": 5})))
    assert annotated, "动态导出的形状列表实参应带符号元素"
    assert any(literal != value for literal, value in annotated)


def test_static_plan_has_no_shape_expr_payload(baseline) -> None:
    compiled, _ = baseline
    for plan in compiled.prefill_plans[MAX_SEQ]:
        for cmd in plan.commands:
            assert "out_shape_expr" not in cmd.payload
            assert "arg_shape_exprs" not in cmd.payload


@pytest.mark.parametrize("prompt_len", [1, 2, 3, 5, 8, 12, MAX_SEQ - MAX_NEW_TOKENS])
def test_dynamic_tokens_match_static_baseline(prompt_len, baseline, dynamic) -> None:
    """单张符号 T 图在任意 prompt 长度下与静态基线逐 token 一致。"""
    base_compiled, base_backend = baseline
    dyn_compiled, dyn_backend = dynamic
    prompt = [1 + i for i in range(prompt_len)]
    assert _run(dyn_compiled, dyn_backend, prompt) == _run(
        base_compiled, base_backend, prompt)


def test_dynamic_comm_entries_carry_t_axes(dynamic, baseline) -> None:
    """动态 prefill 的通信条目按源节点表达式标出 T 轴；静态条目保持空。"""
    from comm.plan import build_comm_plan
    from runtime.compile import _shape_exprs_of

    dyn_compiled, _ = dynamic
    dyn_entries = build_comm_plan(
        dyn_compiled.prefill_edges, _shape_exprs_of(dyn_compiled.prefill_gm))
    assert any(entry.t_axes for entry in dyn_entries), \
        "动态 prefill 应至少有一条通信条目依赖 T"
    for entry in dyn_entries:
        if entry.shape_exprs:
            assert len(entry.shape_exprs) == len(entry.shape)
            assert entry.t_axes == t_axes_from_exprs(
                entry.shape, entry.shape_exprs)

    base_compiled, _ = baseline
    base_entries = build_comm_plan(
        base_compiled.prefill_edges, _shape_exprs_of(base_compiled.prefill_gm))
    assert all(not entry.t_axes for entry in base_entries)
    assert all(not entry.shape_exprs for entry in base_entries)


def test_dynamic_prefill_uses_actual_length(dynamic, baseline) -> None:
    """动态 prefill 不再补 pad；静态路径仍按桶长补齐。"""
    from runtime.executor import prefill_bucket

    dyn_compiled, _ = dynamic
    base_compiled, _ = baseline
    assert prefill_bucket(dyn_compiled, 5) == 5
    assert prefill_bucket(base_compiled, 5) == MAX_SEQ
