"""MoE 动态分派（P1）：gate 跳过机制与 dispatch 命令的验证。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Command


def _backend_with_counter(num_dpus: int = 1):
    """建一个带计数内核的后端，返回 (hal, counter)。"""
    hal = NumpyBackend(NumpyBackendConfig(
        num_dpus=num_dpus, mram_bytes_per_dpu=1 << 20))
    counter = {"runs": 0}

    def counting_kernel(hal, dpu_id, cmd):
        counter["runs"] += 1

    hal.register_kernel("count", counting_kernel)
    return hal, counter


def _launch_cmd(cmd_id: int, *, expert: int, gate: int) -> Command:
    return Command(
        id=cmd_id, op="launch", dpu_id=0,
        payload={"kernel": "count", "gate": gate, "expert": expert},
    )


def test_gate_skipped_launch_runs_no_kernel() -> None:
    """命中跳过集的专家 launch 不执行内核，事件照常完成。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    event = hal.submit(_launch_cmd(1, expert=3, gate=7))
    hal.wait(event)
    assert counter["runs"] == 0
    assert event.future.done()


def test_gate_not_skipped_launch_runs_kernel() -> None:
    """不在跳过集里的专家 launch 正常执行。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    event = hal.submit(_launch_cmd(1, expert=5, gate=7))
    hal.wait(event)
    assert counter["runs"] == 1


def test_gate_key_is_lane_aware() -> None:
    """跳过集按 (lane, dispatch_id) 键控：别的泳道的跳过集不影响本泳道。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(1, 7)] = {3}  # 泳道 1 的跳过集
    event = hal.submit(_launch_cmd(1, expert=3, gate=7))  # 无 ctx = 泳道 0
    hal.wait(event)
    assert counter["runs"] == 1


def test_no_gate_payload_runs_kernel() -> None:
    """不带 gate 字段的命令（全部既有命令）行为不变。"""
    hal, counter = _backend_with_counter()
    cmd = Command(id=1, op="launch", dpu_id=0, payload={"kernel": "count"})
    hal.wait(hal.submit(cmd))
    assert counter["runs"] == 1


@pytest.fixture(scope="module")
def moe_compiled(tmp_path_factory):
    """小 MoE（4 层 × 8 专家）编译出带 dispatch 的 decode 计划。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from backend.architectures import ARCHITECTURES
    from contracts.op_contract import PIMHardwareConfig
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2

    torch.manual_seed(5)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True)).eval().to(torch.float16)
    path = tmp_path_factory.mktemp("qwen3_moe_plan")
    model.save_pretrained(path)
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = ARCHITECTURES["qwen3_moe"].strategy(4, loaded.config)
    hw = HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=4, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=hardware, num_request_slots=1, num_parallel_slots=1)
    return compiled, loaded


def test_decode_plan_carries_dispatch_and_expert_tags(moe_compiled) -> None:
    """decode 计划：每层一个 dispatch host_op；专家 launch 带 gate/expert 且 waits 含 gate。"""
    cmds = moe_compiled[0].decode_plans[0].commands
    dispatch = [c for c in cmds if c.op == "host_op"
                and str(c.payload.get("node", "")).startswith("moe_dispatch")]
    assert len(dispatch) == 4  # 4 层各一个
    dispatch_ids = {c.id for c in dispatch}
    tagged = [c for c in cmds if c.op == "launch" and "expert" in c.payload]
    assert tagged, "专家 launch 应带 expert 标签"
    assert {c.payload["gate"] for c in tagged} <= dispatch_ids
    assert all(c.payload["gate"] in c.waits for c in tagged)
    assert {c.payload["expert"] for c in tagged} <= set(range(8))
    # 只有专家的三个 linear 被打标：每层 8 专家 × 3 linear × tp 分片。
    untagged_linears = [c for c in cmds if c.op == "launch"
                        and "linear" in str(c.payload.get("kernel", ""))
                        and "expert" not in c.payload]
    assert untagged_linears  # 注意力/lm_head 的 linear 不带标签


def _decode_three_tokens(compiled):
    """装权重跑 3 步贪心 decode，返回 (token 列表, backend)。"""
    import numpy as np
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=4, mram_bytes_per_dpu=1 << 28))
    register_all(backend)
    load_weights(compiled, backend)
    got = run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor([1, 2, 3, 4]),
        max_new_tokens=3, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)
    return list(got), backend


def test_dispatch_skips_and_matches_hf(moe_compiled) -> None:
    """开启动态分派：每层跳过 E-top_k 个专家，贪心生成与 HF 一致。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    # fixture 的模型在 ARCHITECTURES 加载时已被静态化；参考值用未打补丁的原始权重。
    torch.manual_seed(5)
    ref_model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
        norm_topk_prob=True)).eval().to(torch.float16)
    ref = ref_model.generate(torch.tensor([[1, 2, 3, 4]]),
                             max_new_tokens=3, do_sample=False)

    got, backend = _decode_three_tokens(moe_compiled[0])
    assert got == ref[0][4:].tolist()
    # decode（T=1）每层命中 top_k=2 → 跳过 8-2=6 个专家。
    assert backend.gate_skips, "dispatch 应写入跳过集"
    for (lane, dispatch_id), skips in backend.gate_skips.items():
        assert len(skips) == 6, f"lane={lane} dispatch={dispatch_id}: {sorted(skips)}"


def test_dispatch_disabled_matches_enabled(moe_compiled, monkeypatch) -> None:
    """关闭动态分派（退回稠密模拟）与开启时生成一致——跳过是纯运行期行为。"""
    import runtime.exec_plan_gen as epg
    from contracts.op_contract import PIMHardwareConfig
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2

    enabled, _ = _decode_three_tokens(moe_compiled[0])

    monkeypatch.setattr(epg, "_MOE_DISPATCH", False)
    compiled, loaded = moe_compiled
    disabled_compiled = compile_llama2(
        loaded, compiled.strategy, prefill_seq_len=4, max_seq=16,
        hw=HwBudget(mram_bytes=1 << 28, align=256, sys_reserve_bytes=1 << 20),
        hardware=PIMHardwareConfig(
            num_dpus=4, num_tasklets=4, mram_bytes_per_dpu=1 << 28,
            wram_bytes_per_dpu=65536, dma_align=64),
        num_request_slots=1, num_parallel_slots=1)
    disabled, _ = _decode_three_tokens(disabled_compiled)
    assert enabled == disabled


def test_dispatch_two_requests_match_single_runs(moe_compiled) -> None:
    """两请求并发 decode：各泳道的跳过集互不覆盖，token 与单独运行一致。"""
    import torch

    from runtime.executor import RequestTable, run_decode_loop, run_one_step

    compiled, _ = moe_compiled
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def alone(prompt):
        import numpy as np

        from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
        from runtime.compile import causal_mask_of, load_weights
        from runtime.kernels import register_all
        backend = NumpyBackend(NumpyBackendConfig(
            num_dpus=4, mram_bytes_per_dpu=1 << 28,
            num_parallel_slots=compiled.num_parallel_slots))
        register_all(backend)
        load_weights(compiled, backend)
        return list(run_decode_loop(
            compiled, backend, prompt_ids=torch.tensor(prompt),
            max_new_tokens=3, eos_id=-1,
            sample_fn=lambda l: int(np.argmax(l)), causal_mask_of=causal_mask_of))

    singles = [alone(p) for p in prompts]

    import numpy as np

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.kernels import register_all
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=4, mram_bytes_per_dpu=1 << 28,
        num_parallel_slots=compiled.num_parallel_slots))
    register_all(backend)
    load_weights(compiled, backend)
    table = RequestTable(2, compiled.kv_pool_slots)
    for i, p in enumerate(prompts):
        table.add_request(f"r{i}", p, eos_id=-1)
    for _ in range(3):
        run_one_step(compiled, backend, table, causal_mask_of,
                     sample_fn=lambda l: int(np.argmax(l)))
    got = [list(table.get(f"r{i}").generated) for i in range(2)]
    assert got == singles


def test_gate_skipped_launch_zeroes_its_output() -> None:
    """被跳过的专家 launch 把输出区写零：旧缓冲里的 inf 不会以 inf×0=nan 毒化下游。"""
    import numpy as np

    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    # 预先把输出区填成 inf（模拟上一步溢出留下的脏值）。
    from contracts.exec_plan import Access
    offset = 0
    dirty = np.full(16, np.inf, dtype=np.float16)
    hal.write_local(0, offset, dirty)
    cmd = Command(
        id=1, op="launch", dpu_id=0,
        payload={"kernel": "count", "gate": 7, "expert": 3},
        writes=[Access(("dpu", 0), offset, dirty.nbytes)])
    hal.wait(hal.submit(cmd))
    assert counter["runs"] == 0
    got = hal.read_local(0, offset, (16,), np.float16)
    assert np.all(got == 0), f"输出区应为零，得 {got[:4]}"
