"""多 batch 编排器判据：B=2/4 与单卡 PyTorch batched 贪心解码逐 token 对齐。"""

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
from graph.strategy import llama_strategy
from memory.mem_planner import HwBudget
from runtime.compile import causal_mask_of, compile_llama2, load_weights
from runtime.executor import execute_plan, run_decode_loop


def _batch_of(prompt_ids) -> int:
    """按 prompt 入参的实际槽数取 batch（tensor/ndarray 与 list 同口径）。"""
    from runtime.executor import _as_prompt_rows

    return len(_as_prompt_rows(prompt_ids))
from runtime.kernels import register_all

NUM_DPUS = 4
NUM_LAYERS = 1
PROMPT_LEN = 8
DECODE_STEPS = 4
MAX_SEQ = 16
VOCAB = 128

MODEL_KWARGS = dict(
    hidden_size=64, intermediate_size=128, num_hidden_layers=NUM_LAYERS,
    num_attention_heads=8, num_key_value_heads=8, vocab_size=VOCAB,
)


def _prompts(batch: int) -> list[list[int]]:
    """每个槽一条互不相同的固定 prompt。"""
    return [[(i * 7 + b * 13) % (VOCAB - 1) + 1 for i in range(PROMPT_LEN)]
            for b in range(batch)]


def _greedy_reference(model, prompts: list[list[int]], steps: int) -> list[list[int]]:
    """单卡 PyTorch batched 贪心解码参考（每步全序列重算，无 cache）。"""
    seqs = [list(p) for p in prompts]
    batch = len(seqs)
    for _ in range(steps):
        cur = len(seqs[0])
        input_ids = torch.tensor(seqs, dtype=torch.long)
        position_ids = (torch.arange(cur, dtype=torch.long)
                        .unsqueeze(0).repeat(batch, 1))
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=causal_mask_of(cur),
                position_ids=position_ids, use_cache=False).logits
        for b in range(batch):
            seqs[b].append(int(logits[b, -1].argmax()))
    return [seq[PROMPT_LEN:] for seq in seqs]


@pytest.fixture(scope="module")
def small_llama():
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(**MODEL_KWARGS)).eval().to(torch.float16)


def _compile(model, batch: int, *, per_slot_prefill: bool = False):
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    return compile_llama2(
        model, llama_strategy(
            NUM_DPUS, num_stages=1, num_heads=8, num_kv_heads=8,
            intermediate_size=128, vocab_size=VOCAB, num_layers=NUM_LAYERS),
        prefill_seq_len=PROMPT_LEN, max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        batch_size=batch, per_slot_prefill=per_slot_prefill)


class _RecordingHal:
    """转发 HAL 并记录每次 execute_plan 绑定的窗口（执行顺序对拍用）。

    `execute_plan` 每次调用恰好一次 `bind_inputs`，窗口序列即计划执行
    序列——逐槽/副本路由的执行顺序用公开可观察信号对拍，不侵入实现。
    """

    def __init__(self, backend):
        self._backend = backend
        self.windows: list[int] = []

    def bind_inputs(self, values, *, pos=None, dpu_window=0):
        if values:  # 计划执行的真实绑定；执行后的窗口复位不计数。
            self.windows.append(dpu_window)
        return self._backend.bind_inputs(values, pos=pos, dpu_window=dpu_window)

    def __getattr__(self, name):
        return getattr(self._backend, name)


@pytest.fixture(scope="module")
def backends(small_llama):
    """按 B 档编译并装载权重，共享同一 backend 复用 KV 覆写。"""
    out = {}
    for batch in (1, 2, 4):
        compiled = _compile(small_llama, batch)
        # 编译期契约：state 的初始值就是 B 槽零长（消费前断言，顺序无关）。
        assert compiled.state.valid_lens == (0,) * batch
        backend = NumpyBackend(NumpyBackendConfig(
            num_dpus=NUM_DPUS, mram_bytes_per_dpu=1 << 28))
        register_all(backend)
        load_weights(compiled, backend)
        out[batch] = (compiled, backend)
    return out


def _run(backends_entry, prompt_ids, *, eos_id=-1, max_new_tokens=DECODE_STEPS):
    compiled, backend = backends_entry
    # 公开契约：直接消费编译结果的 state（上一轮跑完停在终长，跑前归零）。
    compiled.state.valid_len = 0
    return run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, backend,
        prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_id=eos_id,
        state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
    )


@pytest.mark.parametrize("batch", [2, 4])
def test_compiled_state_drives_public_decode_loop(
    batch, small_llama, backends
) -> None:
    """公开编排契约：编译结果的 state 原样驱动完整 prefill+decode（round14 严重1）。

    `compile_llama2(batch_size=B)` 的 state 必须就是 B 槽初始状态，调用方
    不得手工另建 DecodeState 补槽数——那正是把该断链掩盖三轮的路径。
    """
    compiled, backend = backends[batch]
    prompts = _prompts(batch)
    # 槽数是编译契约（模块级 fixture 可能已被前序用例跑过，长度停在终长，
    # 归零复用即公开 API 的正常用法）；单槽状态在此直接失败。
    assert len(compiled.state.valid_lens) == batch
    generated = _run(backends[batch], prompts)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    assert generated == reference


@pytest.mark.parametrize("batch", [2, 4])
def test_b2_b4_match_torch(batch, small_llama, backends) -> None:
    """B=2/4：prefill+decode 的逐槽 token 与 batched PyTorch 贪心参考一致。"""
    prompts = _prompts(batch)
    generated = _run(backends[batch], prompts)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    assert generated == reference


def _assert_logits_close(result: np.ndarray, ref: torch.Tensor) -> None:
    """7B 对拍容差口径：max<0.2 / mean<0.02（沿用 test_executor_llama2_7b）。"""
    diff = np.abs(result.astype(np.float32) - ref.numpy().astype(np.float32))
    assert diff.max() < 0.2, diff.max()
    assert diff.mean() < 0.02, diff.mean()


@pytest.mark.parametrize("batch", [2, 4])
def test_prefill_logits_match_torch(batch, small_llama, backends) -> None:
    """B=2/4 的 prefill logits 与单卡 PyTorch batched 前向对齐（容差口径）。"""
    compiled, backend = backends[batch]
    prompts = _prompts(batch)
    input_ids = torch.tensor(prompts, dtype=torch.long)
    position_ids = (torch.arange(PROMPT_LEN, dtype=torch.long)
                    .unsqueeze(0).repeat(batch, 1))
    events = execute_plan(
        compiled.prefill.plan, backend,
        values={"input_ids": input_ids, "causal_mask": causal_mask_of(PROMPT_LEN),
                "position_ids": position_ids},
        pos=(0,) * batch)
    result = np.asarray(backend.wait(events[compiled.prefill.output_cmd_id]))
    with torch.no_grad():
        ref = small_llama(input_ids=input_ids,
                          attention_mask=causal_mask_of(PROMPT_LEN),
                          position_ids=position_ids, use_cache=False).logits
    _assert_logits_close(result, ref)


@pytest.mark.parametrize("batch", [2, 4])
def test_decode_logits_match_torch(batch, small_llama, backends) -> None:
    """B=2/4 的 decode 每步 logits 与全量重算参考按同一容差对齐。

    token 相等是较强但不等价的锚点——logits 中等漂移在 argmax 不翻转时
    不可见；decode 步与 B=4 的容差对拍补齐验收判据（需求 R4）。
    """
    compiled, backend = backends[batch]
    prompts = _prompts(batch)
    seqs = [list(p) for p in prompts]
    # 消费公开字段：与 run_decode_loop 同一份初始状态，不另建。
    state = compiled.state
    state.valid_len = 0
    # prefill：拿各槽首 token（贪心），随后逐步 decode 对拍每步 logits。
    events = execute_plan(
        compiled.prefill.plan, backend,
        values={"input_ids": torch.tensor(seqs, dtype=torch.long),
                "causal_mask": causal_mask_of(PROMPT_LEN),
                "position_ids": (torch.arange(PROMPT_LEN, dtype=torch.long)
                                 .unsqueeze(0).repeat(batch, 1))},
        pos=tuple(state.valid_lens))
    logits = np.asarray(backend.wait(events[compiled.prefill.output_cmd_id]))
    state.valid_lens = (PROMPT_LEN,) * batch
    last = [int(np.argmax(logits[b, -1])) for b in range(batch)]
    for seq, token in zip(seqs, last):
        seq.append(token)

    for _ in range(DECODE_STEPS - 1):
        events = execute_plan(
            compiled.decode.plan, backend,
            values={"input_ids": torch.tensor([[t] for t in last], dtype=torch.long),
                    "causal_mask": causal_mask_of(1),
                    "position_ids": torch.tensor(state.valid_lens).reshape(-1, 1)},
            pos=tuple(state.valid_lens))
        logits = np.asarray(backend.wait(events[compiled.decode.output_cmd_id]))
        state.commit_one_position()
        cur = len(seqs[0])
        with torch.no_grad():
            ref = small_llama(
                input_ids=torch.tensor(seqs, dtype=torch.long),
                attention_mask=causal_mask_of(cur),
                position_ids=(torch.arange(cur, dtype=torch.long)
                              .unsqueeze(0).repeat(batch, 1)),
                use_cache=False).logits[:, -1]
        # 逐槽比较本步 [vocab] logits（KV cache 路径 vs 全量重算）。
        _assert_logits_close(logits[:, -1], ref)
        last = [int(np.argmax(logits[b, -1])) for b in range(batch)]
        for seq, token in zip(seqs, last):
            seq.append(token)


def test_b1_returns_flat_list(backends) -> None:
    """B=1：list[int] 入参仍返回 list[int]，行为与改动前一致。"""
    prompt = _prompts(1)[0]
    generated = _run(backends[1], prompt)
    assert generated and isinstance(generated[0], int)


def test_finished_slot_feeds_pad_and_group_continues(small_llama, backends) -> None:
    """已结束槽喂 pad 占位、输出丢弃；未结束槽继续生成且结果不变。"""
    prompts = _prompts(2)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    # 把槽 0 的首个生成 token 当作 eos：槽 0 立即结束，槽 1 不受影响。
    eos_id = reference[0][0]
    assert reference[1][0] != eos_id, "参考序列退化，换一组 prompt"
    generated = _run(backends[2], prompts, eos_id=eos_id)
    assert generated[0] == [eos_id]  # 结束槽不再产出
    assert generated[1] == reference[1]  # 组继续锁步推进，结果与无抢占参考一致


def test_unequal_prompts_raise(backends) -> None:
    """不等长 prompt 列表违反 closed batch 前提，直接抛 ValueError。"""
    with pytest.raises(ValueError, match="等长"):
        _run(backends[2], [[1, 2, 3], [1, 2]])


def test_2d_tensor_prompt_keeps_row_semantics(small_llama, backends) -> None:
    """[B,S] tensor/ndarray 按行解释为一槽一条，不得静默压平成单序列。"""
    prompts = _prompts(2)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    via_tensor = _run(backends[2], torch.tensor(prompts, dtype=torch.long))
    via_array = _run(backends[2], np.array(prompts, dtype=np.int64))
    assert via_tensor == reference
    assert via_array == reference


def test_1d_tensor_prompt_stays_single_row(backends) -> None:
    """一维 tensor 仍是 B=1 单行语义（改动前行为）。"""
    prompt = _prompts(1)[0]
    generated = _run(backends[1], torch.tensor(prompt, dtype=torch.long))
    assert generated and isinstance(generated[0], int)


def test_3d_prompt_raises(backends) -> None:
    """三维输入违反 prompt 契约，直接抛 ValueError 说明维度。"""
    with pytest.raises(ValueError, match="一维或二维"):
        _run(backends[2], torch.zeros(2, 2, 4, dtype=torch.long))


@pytest.mark.parametrize("batch", [2, 4])
def test_per_slot_prefill_matches_torch(small_llama, batch) -> None:
    """R21-02a 逐槽 prefill 计划：B 个 M=S 步，数值与合批/PyTorch 一致。

    `per_slot_prefill=True` 编译出 [1,S] 逐槽计划（每步 M=S、KV 落本槽
    段），执行序为 B 次 prefill 计划执行（合批为 1 次）；各槽行独立，
    最终 KV 与逐槽首 token 必须与合批路径逐位一致。
    """
    compiled = _compile(small_llama, batch, per_slot_prefill=True)
    assert compiled.per_slot_prefill is not None
    assert len(compiled.per_slot_prefill) == batch
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    prompts = _prompts(batch)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    recorder = _RecordingHal(backend)
    compiled.state.valid_len = 0
    generated = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, recorder,
        prompt_ids=prompts, max_new_tokens=DECODE_STEPS, eos_id=-1,
        state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
        per_slot_prefill=compiled.per_slot_prefill)
    assert generated == reference
    # 执行顺序：B 次逐槽 prefill + (DECODE_STEPS-1) 次合并 decode。
    assert len(recorder.windows) == batch + DECODE_STEPS - 1
    # 合批路径同规模对照：1 次 prefill + 同数 decode。
    merged = _run(_merged_entry(small_llama, batch), prompts)
    assert merged == reference


def _merged_entry(model, batch):
    """按需编译合批档并装载权重（per-slot 用例的合批对照）。"""
    compiled = _compile(model, batch)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    return compiled, backend


def test_per_slot_prefill_default_off_unchanged(small_llama) -> None:
    """缺省关闭：不产逐槽计划，B=1 编译结果形状与现状一致。"""
    compiled = _compile(small_llama, 2)
    assert compiled.per_slot_prefill is None
    single = _compile(small_llama, 1, per_slot_prefill=True)
    # B=1 无逐槽对象：两态行为一致，不另建计划。
    assert single.per_slot_prefill is None


@pytest.mark.parametrize("batch", [2, 4])
def test_dp_replica_routing_matches_torch(small_llama, batch) -> None:
    """R21-02b DP 副本：组 i 路由副本 i mod G，物理块各持一份权重复制。

    单一 backend 承载 2×NUM_DPUS 台物理 DPU；`load_weights` 为每副本各写
    一份权重，两组 closed batch 各自路由到副本的物理块执行（窗口平移），
    数值与单卡 PyTorch 一致；组 1 执行后副本 0 的 KV 区保持未写。
    """
    from runtime.executor import replica_dpu_window, replica_of_group
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=NUM_DPUS * 2, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        small_llama, llama_strategy(
            NUM_DPUS, num_stages=1, num_heads=8, num_kv_heads=8,
            intermediate_size=128, vocab_size=VOCAB, num_layers=NUM_LAYERS,
            replicas=2),
        prefill_seq_len=PROMPT_LEN, max_seq=MAX_SEQ, hw=hw, hardware=hardware,
        batch_size=batch)
    assert compiled.replicas == 2
    # 副本集合：每副本 mem_plans 覆盖自己的物理块（g×N 偏移，不重叠）。
    assert set(compiled.mem_plans_of_replica(0)) == set(range(NUM_DPUS))
    assert set(compiled.mem_plans_of_replica(1)) == set(
        range(NUM_DPUS, NUM_DPUS * 2))
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS * 2, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)  # 两个副本的物理块各写一份权重

    group_prompts = [
        _prompts(batch),
        [[(t + 5) % (VOCAB - 1) + 1 for t in row] for row in _prompts(batch)],
    ]
    references = [
        _greedy_reference(small_llama, prompts, DECODE_STEPS)
        for prompts in group_prompts
    ]

    def run_group(group_index, prompts):
        replica = replica_of_group(group_index, compiled.replicas)
        window = replica_dpu_window(replica, NUM_DPUS)
        assert replica == group_index % 2
        recorder = _RecordingHal(backend)
        compiled.state.valid_len = 0
        generated = run_decode_loop(
            compiled.prefill.plan, compiled.decode.plan, recorder,
            prompt_ids=prompts, max_new_tokens=DECODE_STEPS, eos_id=-1,
            state=compiled.state,
            sample_fn=lambda logits: int(np.argmax(logits)),
            prefill_output_cmd_id=compiled.prefill.output_cmd_id,
            decode_output_cmd_id=compiled.decode.output_cmd_id,
            causal_mask_of=causal_mask_of,
            dpu_window=window)
        # 每次计划执行都绑定该副本的窗口（执行序与路由的可观察信号）。
        assert recorder.windows and set(recorder.windows) == {window}
        return generated

    # 先跑组 1（副本 1）：副本 0 的 KV 区应保持全零——隔离即路由证据。
    assert run_group(1, group_prompts[1]) == references[1]
    for dpu_id, spec in compiled.kv_specs.items():
        kv_base = compiled.mem_plans[dpu_id].kv_base
        stride = spec.kv_allocated_bytes // spec.num_slots
        replica0 = backend.read_local(
            dpu_id, kv_base, (stride,), np.dtype(np.uint8))
        replica1 = backend.read_local(
            dpu_id + NUM_DPUS, kv_base, (stride,), np.dtype(np.uint8))
        assert not replica0.any(), f"dpu{dpu_id} 副本 0 的 KV 被组 1 写入"
        assert replica1.any(), f"dpu{dpu_id} 副本 1 的 KV 应已写入"
    # 再跑组 0（副本 0）：数值同样对拍。
    assert run_group(0, group_prompts[0]) == references[0]


def test_replica_routing_contract() -> None:
    """副本路由与窗口的公开契约：轮转口径与越界拒绝。"""
    from runtime.executor import replica_dpu_window, replica_of_group
    assert [replica_of_group(i, 2) for i in range(4)] == [0, 1, 0, 1]
    assert replica_dpu_window(1, 4) == 4
    with pytest.raises(ValueError):
        replica_of_group(0, 0)
    with pytest.raises(ValueError):
        replica_dpu_window(-1, 4)


def test_per_slot_prefill_kv_lands_in_own_slot(small_llama) -> None:
    """逐槽步只写本槽 KV：槽 0 步执行后，槽 1 的 KV 区保持全零。"""
    compiled = _compile(small_llama, 2, per_slot_prefill=True)
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    prompts = _prompts(2)
    plan = compiled.per_slot_prefill[0]
    execute_plan(
        plan.plan, backend,
        values={
            "input_ids": torch.tensor([prompts[0]], dtype=torch.long),
            "causal_mask": causal_mask_of(PROMPT_LEN),
            "position_ids": torch.arange(PROMPT_LEN, dtype=torch.long)
            .unsqueeze(0),
        },
        pos=(0,))
    # 槽位主序：槽 s 的整段 KV = [kv_base+s*stride, kv_base+(s+1)*stride)，
    # 逐 DPU 读原始字节核对写入落位。
    for dpu_id, spec in compiled.kv_specs.items():
        kv_base = compiled.mem_plans[dpu_id].kv_base
        stride = spec.kv_allocated_bytes // spec.num_slots
        slot0 = backend.read_local(
            dpu_id, kv_base, (stride,), np.dtype(np.uint8))
        assert slot0.any(), f"dpu{dpu_id} 槽 0 应已写入 K/V"
        slot1 = backend.read_local(
            dpu_id, kv_base + stride, (stride,), np.dtype(np.uint8))
        assert not slot1.any(), "dpu{0} 槽 1 被槽 0 的逐槽步意外写入".format(
            dpu_id)


def test_production_export_per_slot_plans_drive_decode_loop(
        small_llama, tmp_path) -> None:
    """生产导出链交付的逐槽主机计划驱动执行器（R25-03）。

    不直调 `compile_llama2`：走 `export_placement_for_strategy`
    （run_full_pipeline 导出步的同一入口）拿到 `result.compiled`，其
    B 份逐槽计划直接进 `run_decode_loop`——数值与 PyTorch 贪心一致、
    执行序为 B 个 M=S 逐槽步。
    """
    from genesim_bridge.placement_export import export_placement_for_strategy
    from tests.test_placement_export import _write_fixture_ir

    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget

    batch = 2
    result = export_placement_for_strategy(
        small_llama, ir_path, tmp_path / "out",
        strategy=llama_strategy(
            NUM_DPUS, num_stages=1, num_heads=8, num_kv_heads=8,
            intermediate_size=128, vocab_size=VOCAB, num_layers=NUM_LAYERS),
        hw=HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20),
        hardware=PIMHardwareConfig(
            num_dpus=NUM_DPUS, num_tasklets=4, mram_bytes_per_dpu=1 << 28,
            wram_bytes_per_dpu=65536, dma_align=64),
        prefill_seq_len=PROMPT_LEN, max_seq=MAX_SEQ,
        measure_kernel_tiles=False,
        batch_sizes=(batch,), per_slot_prefill=True)
    # 注：measure_kernel_tiles=False——本用例断言主机计划交付与执行器
    # 消费，kernel tile 产物链由 test_cost_batch 的导出用例覆盖。
    compiled = result.compiled
    assert compiled.per_slot_prefill is not None
    assert len(compiled.per_slot_prefill) == batch

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=NUM_DPUS, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    prompts = _prompts(batch)
    reference = _greedy_reference(small_llama, prompts, DECODE_STEPS)
    recorder = _RecordingHal(backend)
    compiled.state.valid_len = 0
    generated = run_decode_loop(
        compiled.prefill.plan, compiled.decode.plan, recorder,
        prompt_ids=prompts, max_new_tokens=DECODE_STEPS, eos_id=-1,
        state=compiled.state,
        sample_fn=lambda logits: int(np.argmax(logits)),
        prefill_output_cmd_id=compiled.prefill.output_cmd_id,
        decode_output_cmd_id=compiled.decode.output_cmd_id,
        causal_mask_of=causal_mask_of,
        per_slot_prefill=compiled.per_slot_prefill)
    assert generated == reference
    # 执行序：B 个逐槽 prefill 步 + (DECODE_STEPS-1) 个合并 decode 步。
    assert len(recorder.windows) == batch + DECODE_STEPS - 1
