"""验证 Llama-2-7B 计划在多 DPU 上的并发和依赖等待。"""

from __future__ import annotations

import sys
import threading
import time
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
from genesim_bridge.paths import llama2_7b_model_dir
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime import kernels as kernels_mod
from runtime.compile import sdpa_layer_map, write_weight_shards
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, execute_plan, make_sdpa_handler
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = llama2_7b_model_dir(required=False)
NUM_DPUS = 8
SEQ_LEN = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2

pytestmark = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir(),
    reason="需要在 paths.json 配置 llama2_7b_model_dir",
)



@pytest.fixture(scope="module")
def llama2_instrumented_run():
    """执行一次预填充并记录每条启动命令的时间区间。"""
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
        k_proj.meta[SPEC_META_KEY], layers=list(range(cfg.num_hidden_layers)), num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads, head_dim=head_dim, max_seq=MAX_SEQ, dtype_bytes=KV_DTYPE_BYTES, kv_base=0,
    )
    hw = HwBudget(mram_bytes=4 * 2**30, align=1024, sys_reserve_bytes=64 * 2**20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS,
        num_tasklets=4,
        mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536,
        # DMA 分块使用独立于 MRAM 布局的字节对齐。
        dma_align=64,
    )
    nodes = list(gm.graph.nodes)
    plans = {d: plan_dpu(d, nodes, nodes, kv_specs, hw) for d in range(NUM_DPUS)}

    entries_by_id = {e.edge_id: e for e in build_comm_plan(edges)}
    pending = {}
    for plan in plans.values():
        pending.update(plan.pending_readers_prefill)

    state = DecodeState(valid_len=0)
    sdpa_layer = sdpa_layer_map(gm)

    def host_handler_of(node):
        if "scaled_dot_product_attention" in str(node.target):
            return make_sdpa_handler(sdpa_layer[node.name], kv_specs, state, np.dtype(np.float16))
        return None

    compiled = build_execution_plan(
        nodes, gm, entries_by_id, pending, hardware=hardware,
        host_handler_of=host_handler_of,
    )

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))

    # 在 launch 前后记录时间并扩大并发窗口。
    intervals: list[tuple[int, int, float, float]] = []
    lock = threading.Lock()

    def wrap(fn):
        def wrapped(hal, dpu_id, cmd):
            t0 = time.perf_counter()
            time.sleep(0.01)
            fn(hal, dpu_id, cmd)
            t1 = time.perf_counter()
            with lock:
                intervals.append((cmd.id, dpu_id, t0, t1))
        return wrapped

    for name, fn in kernels_mod._KERNELS.items():
        backend.register_kernel(name, wrap(fn))

    write_weight_shards(gm, plans, backend)

    events = execute_plan(compiled.plan, backend, values={"input_ids": input_ids, "causal_mask": causal_mask}, pos=state.valid_len)
    backend.wait(events[compiled.output_cmd_id])

    return compiled, intervals


def test_independent_dpu_launches_overlap_in_time(llama2_instrumented_run) -> None:
    """验证无依赖 DPU 启动命令的执行区间重叠。"""
    compiled, intervals = llama2_instrumented_run
    q_proj_cmds = [c for c in compiled.plan.commands if c.op == "launch" and c.payload.get("node") == "linear"]
    assert len(q_proj_cmds) == 8

    ids = {c.id for c in q_proj_cmds}
    assert all(not (set(c.waits) & ids) for c in q_proj_cmds), "这 8 条 launch 之间不该互相等待"

    by_id = {cid: (dpu, t0, t1) for cid, dpu, t0, t1 in intervals}
    windows = [by_id[c.id] for c in q_proj_cmds if c.id in by_id]
    assert len(windows) == 8

    overlap_found = any(
        a0 < b1 and b0 < a1
        for i, (_, a0, a1) in enumerate(windows)
        for _, b0, b1 in windows[i + 1:]
    )
    assert overlap_found


def test_dependent_launch_waits_for_true_completion(llama2_instrumented_run) -> None:
    """验证依赖启动命令在前驱完成后执行。"""
    compiled, intervals = llama2_instrumented_run
    by_id = {cid: (dpu, t0, t1) for cid, dpu, t0, t1 in intervals}

    checked = 0
    for cmd in compiled.plan.commands:
        if cmd.op != "launch" or cmd.id not in by_id:
            continue
        _, dep_start, _ = by_id[cmd.id]
        for w in cmd.waits:
            if w not in by_id:
                continue
            _, _, waited_end = by_id[w]
            assert dep_start >= waited_end, (
                f"命令 {cmd.id} 等待命令 {w}，但 {cmd.id} 的开始时刻 {dep_start} "
                f"早于 {w} 的完成时刻 {waited_end}"
            )
            checked += 1
    assert checked > 0, "至少要找到一对有真实 RAW 依赖的 launch 命令用于验证"
