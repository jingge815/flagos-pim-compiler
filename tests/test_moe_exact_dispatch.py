"""exact_dispatch 第三模式的红测：dispatch v2、行级命令与稠密 where 逐位对拍。

先拆红测（计划 M2③）：本文件在 exact_dispatch 落地前应整文件红——
模式不被 resolve 接受、计划没有行级命令、尾链还是 where 主机闭包。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _moe_model(num_layers=2, num_experts=4, norm_topk_prob=True):
    """tiny Qwen3MoE：默认 2 层 × 4 专家，top_k=2，norm_topk_prob=True。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(5)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=num_layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, vocab_size=128,
        num_experts=num_experts, num_experts_per_tok=2,
        norm_topk_prob=norm_topk_prob)).eval().to(torch.float16)


def _hf_reference(model, prompt=(1, 2, 3, 4), steps=3):
    """同一份权重跑 HF 贪心解码，产出参考 token。"""
    import torch

    ref = model.generate(torch.tensor([list(prompt)]),
                         max_new_tokens=steps, do_sample=False)
    return ref[0][len(prompt):].tolist()


def _dispatch_cmds_of(compiled):
    """decode 计划里的 dispatch v2（exact 每层一条 host_op）。"""
    return [c for c in compiled.decode_plans[0].commands
            if c.op == "host_op"
            and str(c.payload.get("node", "")).startswith("moe_dispatch_")]


def _compile(model, *, moe_execution):
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2

    import tempfile

    path = tempfile.mkdtemp() + "/mdl"
    model.save_pretrained(path)
    from backend.architectures import ARCHITECTURES
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = llama_strategy(2, num_stages=1, num_heads=4, num_kv_heads=2,
                              intermediate_size=32, vocab_size=128,
                              num_layers=model.config.num_hidden_layers)
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=2, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    return compile_llama2(
        loaded, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=hardware,
        num_request_slots=1, num_parallel_slots=1,
        moe_execution=moe_execution)


@pytest.fixture(scope="module")
def exact_dense_pair():
    """同一份 tiny 模型：exact 与 dense_sim 各编译一份，供逐位对拍。"""
    from runtime.exec_plan_gen import resolve_moe_execution, moe_execution_default

    cfg = type("C", (), {"model_type": "qwen3_moe",
                         "num_experts": 4, "num_experts_per_tok": 2})()

    # 默认语义不变：不带环境时仍是 dense_sim / dynamic_dispatch 系。
    assert moe_execution_default() != "exact_dispatch", (
        "exact_dispatch 不能是默认模式")
    assert resolve_moe_execution(cfg, "dense_sim") == "dense_sim"

    model = _moe_model()
    return _compile(model, moe_execution="exact_dispatch"), _compile(
        _moe_model(), moe_execution="dense_sim")


# ---------- 模式解析 ----------


def test_exact_dispatch_mode_accepted(monkeypatch) -> None:
    """exact_dispatch 进模式表；PIM_MOE_DISPATCH=2|exact 启用，默认不变。"""
    from runtime.exec_plan_gen import (
        MOE_EXECUTION_MODES, moe_execution_default, resolve_moe_execution)

    # 最小 qwen3_moe 配置桩：resolve 只看 model_type 与专家数。
    cfg = type("C", (), {"model_type": "qwen3_moe",
                         "num_experts": 4, "num_experts_per_tok": 2})()

    assert "exact_dispatch" in MOE_EXECUTION_MODES
    assert resolve_moe_execution(cfg, "exact_dispatch") == "exact_dispatch"
    monkeypatch.setenv("PIM_MOE_DISPATCH", "2")
    assert moe_execution_default() == "exact_dispatch"
    monkeypatch.setenv("PIM_MOE_DISPATCH", "exact")
    assert moe_execution_default() == "exact_dispatch"
    monkeypatch.setenv("PIM_MOE_DISPATCH", "0")
    assert moe_execution_default() != "exact_dispatch"
    monkeypatch.delenv("PIM_MOE_DISPATCH")
    assert moe_execution_default() != "exact_dispatch"


# ---------- 计划结构 ----------


def test_exact_plan_replaces_where_tail(exact_dense_pair) -> None:
    """exact 计划：where 闭包与融合累加退役，行级命令三明治专家链。"""
    import torch

    from contracts.op_semantics import GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL

    compiled = exact_dense_pair[0]
    nodes = {n.name: n for n in compiled.decode_gm.graph.nodes}
    cmds = compiled.decode_plans[0].commands

    # 尾链退役：没有 where 主机闭包，也没有融合加法闭包。
    where_nodes = {n.name for n in nodes.values()
                   if n.target == torch.ops.aten.where.ScalarOther}
    emitted = {c.payload.get("node") for c in cmds if c.op == "host_op"}
    assert not (where_nodes & emitted), "exact 模式不应发射 where 主机闭包"
    fused = [name for name in emitted
             if isinstance(name, str) and name.startswith("fused_")]
    assert not fused, f"exact 模式不应发射融合累加闭包：{fused}"

    # 行级命令：每层每专家每台 DPU 各一条 gather 与 scatter。
    for name in (GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL):
        row_cmds = [c for c in cmds
                    if c.op == "launch" and c.payload.get("kernel") == name]
        assert row_cmds, f"exact 计划缺少 {name} 命令"
        # 2 层 × 4 专家 × 2 DPU（被门控跳过是运行期行为，计划里都在）。
        assert len(row_cmds) == 2 * 4 * 2, (
            f"{name} 应为 2层×4专家×2DPU=16 条，实际 {len(row_cmds)}")
        assert all("gate" in c.payload and "expert" in c.payload
                   for c in row_cmds)


def test_exact_expert_chain_reads_compact(exact_dense_pair) -> None:
    """专家链首层 linear 不再读 xs 原缓冲，改读 gather 写的紧凑缓冲。"""
    compiled = exact_dense_pair[0]
    nodes = {n.name: n for n in compiled.decode_gm.graph.nodes}
    cmds = compiled.decode_plans[0].commands

    # xs = gate linear 的输入（MoE 公共输入）；dense 计划里专家 gate/up_proj
    # 直接读它。exact 里同形状不同偏移——每台 DPU 一条输入读访问，且不等于
    # xs 自己的规划偏移。
    gate_linears = [
        n for n in nodes.values()
        if str(n.target) == "aten.linear.default"
        and any(".mlp.experts." in str(a.target) and "gate_proj" in str(a.target)
                for a in n.args if hasattr(a, "target"))]
    assert gate_linears
    by_node = {}
    for c in cmds:
        by_node.setdefault(c.payload.get("node"), []).append(c)
    all_input_offsets = set()
    for layer_gate in gate_linears:
        xs = layer_gate.args[0]
        xs_spec = xs.meta.get("spec")
        # reads[0] 是第一个张量实参（输入）；第二条是权重，不参比对拍。
        offsets = {(c.reads[0].loc, c.reads[0].offset)
                   for c in by_node.get(layer_gate.name, [])}
        assert len(offsets) == 2, f"每台 DPU 一条输入读访问，实际 {offsets}"
        if xs_spec is not None:
            xs_off = {(("dpu", dpu_id), detail.mram_offset)
                      for dpu_id, detail in xs_spec.shard_map.items()}
            assert not (offsets & xs_off), (
                f"{layer_gate.name} 仍直接读 xs 原缓冲")
        all_input_offsets |= offsets
    # 各专家的紧凑缓冲互不重叠：4 个 gate_proj（2 层 × 2 DPU 段）偏移两两不同。
    assert len(all_input_offsets) == 2 * len(gate_linears) // 2


def test_exact_dispatch_writes_rows_and_tables(exact_dense_pair) -> None:
    """运行一步：moe_rows 全专家登记行数，零命中进 gate_skips，行数守恒。"""
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    compiled = exact_dense_pair[0]
    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    run_decode_loop(compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
                    max_new_tokens=1, eos_id=-1,
                    sample_fn=lambda logits: int(np.argmax(logits)),
                    causal_mask_of=causal_mask_of)
    assert backend.moe_rows, "dispatch v2 应写 moe_rows"
    for (lane, dispatch_id, expert), rows in backend.moe_rows.items():
        # 4 token 的 prompt 先走 T=4 的 prefill（每专家至多 4 行），再是
        # T=1 的 decode（至多 1 行）；上限取两者中大的。
        assert 0 <= rows <= 4
    # 每层每泳道的行数守恒：Σ_e k_e == top_k × T。prompt prefill T=4 → 8，
    # decode T=1 → 2，一步一个 dispatch。
    by_lane: dict[tuple, int] = {}
    for (lane, dispatch_id, _expert), rows in backend.moe_rows.items():
        by_lane[(lane, dispatch_id)] = by_lane.get((lane, dispatch_id), 0) + rows
    assert by_lane and all(total in (2, 8) for total in by_lane.values()), (
        f"行数守恒失败：{by_lane}")


# ---------- 数值对拍 ----------


def _decode_tokens(compiled, prompt, steps):
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    got = run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor(prompt),
        max_new_tokens=steps, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)
    return list(got), backend


def test_exact_matches_dense_bitwise(exact_dense_pair) -> None:
    """exact 与 dense_sim 同权重解码逐位一致（贪心 token 序列）。"""
    exact, dense = exact_dense_pair
    got_exact, _ = _decode_tokens(exact, [1, 2, 3, 4], 3)
    got_dense, _ = _decode_tokens(dense, [1, 2, 3, 4], 3)
    assert got_exact == got_dense, f"exact={got_exact} dense={got_dense}"


def test_exact_matches_hf_sparse(exact_dense_pair) -> None:
    """exact 解码与 HF 稀疏前向参考一致——精确行数不丢 token。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(5)
    ref_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True)).eval().to(torch.float16)
    ref = ref_model.generate(torch.tensor([[1, 2, 3, 4]]),
                             max_new_tokens=3, do_sample=False)

    exact = exact_dense_pair[0]
    got, backend = _decode_tokens(exact, [1, 2, 3, 4], 3)
    assert got == ref[0][4:].tolist(), f"exact={got} hf={ref[0][4:].tolist()}"
    # 零命中专家真的零提交：跳过集非空且每层跳过 E-topk 个。
    assert backend.gate_skips


# ---------- 路由矩阵逐位（M2⑥） ----------


def _decode_dispatch_rows(backend, compiled):
    """取各层 dispatch 的（泳道，每专家行数，跳过集）——读终态，供全命中态用。"""
    out = {}
    for cmd in _dispatch_cmds_of(compiled):
        lanes = {k[0] for k in backend.moe_rows if k[1] == cmd.id}
        assert len(lanes) == 1, f"dispatch {cmd.id} 泳道不唯一：{lanes}"
        lane = lanes.pop()
        rows = {e: backend.moe_rows[(lane, cmd.id, e)]
                for (_, _d, e) in backend.moe_rows if _d == cmd.id}
        out[cmd.payload["node"]] = (
            lane, cmd.id, rows, backend.gate_skips[(lane, cmd.id)])
    return out


def _step_snapshots(compiled, steps):
    """每步执行完抓一份（行数、跳过集、门控统计）快照，供逐步断言。"""
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    snaps = []

    def sample(logits):
        snaps.append({"rows": dict(backend.moe_rows),
                      "skips": {k: set(v) for k, v in backend.gate_skips.items()},
                      "stats": {k: dict(v) for k, v in backend.gate_stats.items()}})
        return int(np.argmax(logits))

    run_decode_loop(compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
                    max_new_tokens=steps, eos_id=-1, sample_fn=sample,
                    causal_mask_of=causal_mask_of)
    return backend, snaps


def test_exact_skip_set_matches_moe_rows(exact_dense_pair) -> None:
    """零命中一致性：T=1 decode 必有 E-topk=2 个零命中专家，且真的零提交。

    第 0 步是 T=4 的 prompt（可能全命中），第 1 步起是 T=1 decode；快照取
    第 1 步。moe_rows/gate_skips 每步覆写，事后读只剩末步终态，必须在执行
    中抓。零提交用内核执行计数证：被跳过专家的 gather/scatter 一条都没跑。
    """
    import torch

    from contracts.op_semantics import GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL
    from runtime.kernels import register_all

    compiled = exact_dense_pair[0]
    backend, snaps = _step_snapshots(compiled, steps=2)
    assert len(snaps) == 2, "每步应各有一份快照"

    dispatch_ids = {c.id for c in _dispatch_cmds_of(compiled)}
    cur = snaps[1]
    zero_hit = {}
    for (lane, gate_id), skips in sorted(cur["skips"].items()):
        if gate_id not in dispatch_ids:
            continue  # 只看 exact 的 dispatch v2，dense 门控另测
        rows = {e: cur["rows"].get((lane, gate_id, e)) for e in range(4)}
        assert None not in rows.values(), f"dispatch {gate_id} 行数登记不全"
        assert skips == {e for e, k in rows.items() if k == 0}, (
            f"gate_skips 与 moe_rows 不一致：{skips} vs {rows}")
        assert len(skips) == 2, f"T=1 应恰好跳过 2 个专家：{skips}"
        assert all(k in (0, 1) for k in rows.values()), f"行数应为 0/1：{rows}"
        assert sum(rows.values()) == 2, f"行数守恒（topk=2×T=1）失败：{rows}"
        zero_hit.setdefault(len(zero_hit), skips)
    assert zero_hit, "T=1 decode 应有零命中专家"

    # 零提交：重跑一步，按步计数每专家 gather/scatter 的实际执行次数；
    # 末步（T=1 decode）被跳过专家应为 0 次，命中专家为 4 次
    # （gather+scatter × 2 DPU）。
    import numpy as np
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop

    backend2 = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend2)
    load_weights(compiled, backend2)
    state = {"step": 0}
    executed: dict[tuple, int] = {}
    for kernel in (GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL):
        orig = backend2._kernels.lookup(kernel)

        def wrapper(hal, dpu_id, cmd, _orig=orig):
            key = (state["step"], cmd.payload["gate"], cmd.payload["expert"])
            executed[key] = executed.get(key, 0) + 1
            _orig(hal, dpu_id, cmd)
        backend2._kernels.register(kernel, wrapper)

    last_skips = {}
    dispatch_by_id = {c.id: c for c in _dispatch_cmds_of(compiled)}

    def sample2(logits):
        last_skips.clear()
        for (lane, gate_id), skips in backend2.gate_skips.items():
            if gate_id in dispatch_by_id:
                last_skips[gate_id] = skips
        state["step"] += 1
        return int(np.argmax(logits))

    run_decode_loop(compiled, backend2, prompt_ids=torch.tensor([1, 2, 3, 4]),
                    max_new_tokens=2, eos_id=-1, sample_fn=sample2,
                    causal_mask_of=causal_mask_of)
    assert last_skips, "T=1 decode 应有零命中专家"
    last = state["step"] - 1
    for gate_id, skips in sorted(last_skips.items()):
        for expert in range(4):
            n = sum(v for (s, g, _e), v in executed.items()
                    if s == last and g == gate_id and _e == expert)
            if expert in skips:
                assert n == 0, f"被跳过专家 {expert} 仍有 {n} 次内核执行"
            else:
                assert n == 4, (
                    f"命中专家 {expert} 应执行 gather+scatter×2DPU=4 次，实得 {n}")


def test_exact_routing_tables_bitwise(exact_dense_pair) -> None:
    """路由表逐位：idx/w 表段内容与 dispatch 实读的 mask 重算按字节一致。

    dispatch 闭包前两个 int 是（掩码、路由权重）两条 scatter 的命令号；
    waits 经 fold_waits 排序折叠，不能按位置取。mask/routing 是每步执行的
    中间结果，事件表跨步清理，只能包装 result_of 在执行中抓快照。
    """
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from contracts.dtypes import dtype_bytes
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    compiled = exact_dense_pair[0]
    dispatch = _dispatch_cmds_of(compiled)
    assert dispatch
    closure_ids = {}
    for cmd in dispatch:
        ints = []
        for cell in cmd.payload["fn"].__closure__ or ():
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if isinstance(v, int) and not isinstance(v, bool):
                ints.append(v)
        closure_ids[cmd.id] = ints[:2]

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    seen = {}
    orig_result = backend.result_of

    def spy(cmd_id):
        v = orig_result(cmd_id)
        seen[cmd_id] = v.clone() if hasattr(v, "clone") else v
        return v

    backend.result_of = spy
    # 跑 2 步：第 0 步是 prefill 计划（T=4，地址偏移与 decode 计划不同），
    # 末步是 decode 计划——正是 _dispatch_cmds_of 读的这份，末步终态可核。
    run_decode_loop(compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
                    max_new_tokens=2, eos_id=-1,
                    sample_fn=lambda logits: int(np.argmax(logits)),
                    causal_mask_of=causal_mask_of)

    num_dpus, num_experts = 2, 4
    idx_b = dtype_bytes("int64")
    w_b = dtype_bytes("float16")
    for cmd in dispatch:
        mask_id, routing_id = closure_ids[cmd.id]
        mask, routing = seen.get(mask_id), seen.get(routing_id)
        assert mask is not None and routing is not None, (
            f"{cmd.payload['node']} 的 mask/routing 未在执行中产生")
        assert mask.dtype == torch.bool, (
            f"闭包首 int 应是布尔掩码命令号，实得 dtype={mask.dtype}")
        t_rows = cmd.writes[0].length // (num_experts * idx_b)
        seg_idx = t_rows * idx_b   # 每专家段步长（dispatch 写表用同一算术）
        seg_w = t_rows * w_b
        lane = {k[0] for k in backend.moe_rows if k[1] == cmd.id}.pop()
        for expert in range(num_experts):
            rows = mask[:, expert].nonzero(as_tuple=True)[0]
            k_e = backend.moe_rows[(lane, cmd.id, expert)]
            assert int(rows.numel()) == k_e, "moe_rows 与掩码重算不一致"
            if k_e == 0:
                continue
            expect_w = routing[rows, expert].to(torch.float16).numpy()
            for dpu in range(num_dpus):
                got_idx = backend.copy_from_dpu(
                    dpu, cmd.writes[2 * dpu].offset + expert * seg_idx,
                    (t_rows,), np.int64)
                got_w = backend.copy_from_dpu(
                    dpu, cmd.writes[2 * dpu + 1].offset + expert * seg_w,
                    (t_rows,), np.float16)
                assert np.array_equal(got_idx[:k_e], rows.numpy()), (
                    f"{cmd.payload['node']} 专家 {expert} DPU{dpu} idx 表不符")
                assert np.all(np.diff(got_idx[:k_e]) > 0), "行号应按 token 升序"
                assert np.array_equal(got_w[:k_e], expect_w), (
                    f"{cmd.payload['node']} 专家 {expert} DPU{dpu} w 表不符")


def test_exact_scatter_waw_chain(exact_dense_pair) -> None:
    """scatter 的 WAW 链：发射序专家升序，RAW 地址依赖找到前一条写者。

    静态 waits 经 fold_waits 折叠——同 DPU 前驱只留末位（同泳道按计划序
    FIFO 执行），链不体现在 waits 里；用计划序重放 writers 表，在依赖层
    断言每条 scatter 读的 acc 都能找到前一条 scatter 的写。
    """
    from collections import defaultdict

    from contracts.op_semantics import SCATTER_ADD_KERNEL
    from runtime.dependency import deps_of

    plan = exact_dense_pair[0].decode_plans[0].commands
    cmds = [c for c in plan
            if c.op == "launch" and c.payload.get("kernel") == SCATTER_ADD_KERNEL]
    by_chain = defaultdict(list)
    for c in cmds:
        by_chain[(c.payload["node"].rsplit(":", 1)[0], c.dpu_id)].append(c)
    for key, chain in sorted(by_chain.items()):
        chain.sort(key=lambda c: c.payload["expert"])
        experts = [c.payload["expert"] for c in chain]
        assert experts == sorted(experts), f"{key} 发射序非专家升序：{experts}"
        for prev, cur in zip(chain, chain[1:]):
            writers = defaultdict(list)
            for c in plan:
                if c.id == cur.id:
                    break
                for w in c.writes:
                    writers[w.loc].append((w, c.id))
            acc_read = next(r for r in cur.reads
                            if r.loc == cur.writes[0].loc
                            and r.offset == cur.writes[0].offset)
            assert prev.id in deps_of([acc_read], writers), (
                f"{key} 专家 {cur.payload['expert']} 的 acc 读在依赖层找不到"
                f"前序写者（专家 {prev.payload['expert']}，id={prev.id}）")


def test_exact_kernel_cache_bounded(exact_dense_pair) -> None:
    """行级内核按包络形状编一次：多跑几步缓存条目数不涨，形状签名全同。"""
    import torch

    from runtime.kernels import _COMPILED_KERNEL_CACHE, register_all
    from contracts.op_semantics import GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop

    compiled = exact_dense_pair[0]
    # 结构：同一内核的所有命令形状键一致——缓存键只含包络，行数不进键。
    for kernel in (GATHER_ROWS_KERNEL, SCATTER_ADD_KERNEL):
        sigs = {(tuple(c.payload["arg_shapes"]), c.payload["out_shape"])
                for c in compiled.decode_plans[0].commands
                if c.op == "launch" and c.payload.get("kernel") == kernel}
        assert len(sigs) == 1, f"{kernel} 形状签名不唯一：{sigs}"

    def one_run(steps):
        backend = NumpyBackend(NumpyBackendConfig(
            num_dpus=2, mram_bytes_per_dpu=1 << 28))
        register_all(backend)
        load_weights(compiled, backend)
        run_decode_loop(compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
                        max_new_tokens=steps, eos_id=-1,
                        sample_fn=lambda logits: int(np.argmax(logits)),
                        causal_mask_of=causal_mask_of)

    # 预热要覆盖 T=4（prompt 步）与 T=1（decode 步）两种形状绑定，再断言
    # 同形状重跑零增长——增长只能来自新形状，不能来自步数。
    one_run(4)
    warm = len(_COMPILED_KERNEL_CACHE)
    one_run(4)
    assert len(_COMPILED_KERNEL_CACHE) == warm, (
        f"内核缓存随步数增长：{warm} -> {len(_COMPILED_KERNEL_CACHE)}")


def test_exact_topk_equals_experts() -> None:
    """top_k == E 极端态（E=2）：单 token 全命中，无跳过，token 对 HF 一致。"""
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    model = _moe_model(num_experts=2)
    expect = _hf_reference(model)
    compiled = _compile(model, moe_execution="exact_dispatch")

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    got = run_decode_loop(compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
                          max_new_tokens=3, eos_id=-1,
                          sample_fn=lambda logits: int(np.argmax(logits)),
                          causal_mask_of=causal_mask_of)
    assert list(got) == expect, f"exact={list(got)} hf={expect}"

    # 计划里行级命令减半：2 层 × 2 专家 × 2 DPU = 8 条 gather。
    gathers = [c for c in compiled.decode_plans[0].commands
               if c.op == "launch"
               and c.payload.get("kernel") == "pim.gather_rows"]
    assert len(gathers) == 8
    # 全命中：decode 每专家 k_e == 1，跳过集全空。
    for _name, (_lane, _id, rows, skips) in _decode_dispatch_rows(
            backend, compiled).items():
        assert skips == set(), f"全命中配置不应有跳过：{skips}"
        assert all(k == 1 for k in rows.values()), f"行数应为 T=1：{rows}"


def test_exact_norm_topk_prob_false() -> None:
    """norm_topk_prob=False 语义：不重归一的原始 softmax 权重照样逐位对 HF。"""
    model = _moe_model(norm_topk_prob=False)
    expect = _hf_reference(model)
    compiled = _compile(model, moe_execution="exact_dispatch")
    got, _backend = _decode_tokens(compiled, [1, 2, 3, 4], 3)
    assert got == expect, f"exact={got} hf={expect}"
