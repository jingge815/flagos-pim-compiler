"""真实 Llama-2-7B 上 8 DPU 并发调度的显式验证（验证 3b）。

用户明确指出：只用 8 台 DPU 跑通不足以证明"可并行性/交互性"，需要专门验证
`execute_plan` 确实把无依赖命令一次性提交给 8 条独立 DPU 流并发执行，而不是
隐式串行；同时验证 `waits` 精确等待生效——依赖命令不会在前驱完成前抢跑。

两条断言都基于 `NumpyBackend` 已有的线程池执行模型（每个 DPU 一条独立线程
流，`ThreadPoolExecutor`），不新增编排器代码，只在 kernel 外面包一层记录
时间戳的探针（测试专用，不改 `runtime/kernels.py` 本体）。
"""

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
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from memory.kv_layout import kv_specs_from_placement
from memory.mem_planner import HwBudget, plan_dpu
from runtime import kernels as kernels_mod
from runtime.exec_plan_gen import build_execution_plan
from runtime.executor import DecodeState, execute_plan, make_sdpa_handler
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8
SEQ_LEN = 16
MAX_SEQ = 64
KV_DTYPE_BYTES = 2

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


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


@pytest.fixture(scope="module")
def llama2_instrumented_run():
    """真实 7B 单次 prefill，kernel 包一层时间戳探针，记录每条 launch 命令
    的 (cmd_id, dpu_id, start, end)（模块内一次）。"""
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

    compiled = build_execution_plan(nodes, gm, entries_by_id, pending, host_handler_of=host_handler_of)

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=NUM_DPUS, mram_bytes_per_dpu=hw.mram_bytes))

    # 逐条 launch 命令包一层时间戳探针 + 人为延时（放大并发窗口，让"确实
    # 重叠"这件事在真实硬件级并发下也能被稳定观察到，不依赖运气）。
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
    backend.wait(events[compiled.output_cmd_id])

    return compiled, intervals


def test_independent_dpu_launches_overlap_in_time(llama2_instrumented_run) -> None:
    """无依赖的 8 台 DPU 的同一层 q_proj launch，执行区间确有真实重叠——
    证明 `execute_plan` 把它们一次性交给 8 条独立线程流，不是隐式串行。
    """
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
    """同 DPU 上有 RAW 依赖的两条 `launch`（如 q_proj 输出被同层内下一个
    DPU 算子读取）：依赖方的开始时刻不早于被依赖方的完成时刻——证明
    `waits` 精确等待生效，不是碰巧先后执行。两者都在探针覆盖范围内（都是
    `launch`），不依赖 host_op 的计时。
    """
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
