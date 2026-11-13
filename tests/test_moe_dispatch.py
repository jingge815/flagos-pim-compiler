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


def _launch_cmd(cmd_id: int, *, expert: int, gate: int, **payload) -> Command:
    return Command(
        id=cmd_id, op="launch", dpu_id=0,
        payload={"kernel": "count", "gate": gate, "expert": expert, **payload},
    )


def _submit_dispatch(hal, dispatch_id: int = 7) -> None:
    """先提交一条 dispatch host_op，把 gate 命令登记进事件表。

    跳过发生在提交前：hal 要能等到 dispatch 的结果才决定这条命令是否入队，
    所以真实计划里 dispatch 一定排在专家命令前面。
    """
    cmd = Command(id=dispatch_id, op="host_op", dpu_id=None,
                  payload={"fn": lambda backend, command: None})
    hal.wait(hal.submit(cmd))


def test_gate_skipped_launch_runs_no_kernel() -> None:
    """命中跳过集的专家 launch 不执行内核，事件照常完成。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    event = hal.submit(_launch_cmd(1, expert=3, gate=7))
    hal.wait(event)
    assert counter["runs"] == 0
    assert event.future.done()


def test_gate_not_skipped_launch_runs_kernel() -> None:
    """不在跳过集里的专家 launch 正常执行。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    event = hal.submit(_launch_cmd(1, expert=5, gate=7))
    hal.wait(event)
    assert counter["runs"] == 1


def test_gate_key_is_lane_aware() -> None:
    """跳过集按 (lane, dispatch_id) 键控：别的泳道的跳过集不影响本泳道。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(1, 7)] = {3}  # 泳道 1 的跳过集
    _submit_dispatch(hal)
    event = hal.submit(_launch_cmd(1, expert=3, gate=7))  # 无 ctx = 泳道 0
    hal.wait(event)
    assert counter["runs"] == 1


def test_no_gate_payload_runs_kernel() -> None:
    """不带 gate 字段的命令（全部既有命令）行为不变。"""
    hal, counter = _backend_with_counter()
    cmd = Command(id=1, op="launch", dpu_id=0, payload={"kernel": "count"})
    hal.wait(hal.submit(cmd))
    assert counter["runs"] == 1


def test_skipped_launch_is_never_submitted() -> None:
    """未命中专家在提交前就被拦下：不占线程池任务，也不再逐条查跳过集。"""
    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    submitted = []
    original = hal._submit_on_dpu

    def spy(*args, **kwargs):
        submitted.append(args[0])
        return original(*args, **kwargs)

    hal._submit_on_dpu = spy
    hal.wait(hal.submit(_launch_cmd(1, expert=3, gate=7)))
    assert counter["runs"] == 0
    assert submitted == [], f"被跳过的命令仍提交到 DPU 队列：{submitted}"

    hal.wait(hal.submit(_launch_cmd(2, expert=5, gate=7)))
    assert counter["runs"] == 1
    assert submitted == [0]


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
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
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


def _assert_dispatch_reads_selected_mask(gm, plan) -> None:
    """每层的 dispatch 必须读 top-k 选中掩码，而不是路由权重张量。"""
    import torch

    nodes = {n.name: n for n in gm.graph.nodes}
    bool_scatters = {
        n.name for n in nodes.values()
        if n.target in (torch.ops.aten.scatter.src,
                        torch.ops.aten.scatter.value)
        and n.meta.get("val") is not None
        and n.meta["val"].dtype == torch.bool
    }
    assert len(bool_scatters) == 4, (
        f"每层应有一个布尔选中掩码 scatter，实际 {sorted(bool_scatters)}")
    by_id = {c.id: c for c in plan.commands}
    dispatch = [
        c for c in plan.commands
        if c.op == "host_op"
        and str(c.payload.get("node", "")).startswith("moe_dispatch")
    ]
    assert len(dispatch) == 4
    for cmd in dispatch:
        sources = [
            by_id[waited].payload.get("node")
            for waited in cmd.waits if waited in by_id
        ]
        assert any(source in bool_scatters for source in sources), (
            f"{cmd.payload['node']} 没有读选中掩码，来源是 {sources}")


def test_decode_and_prefill_dispatch_read_the_selected_mask(moe_compiled) -> None:
    """decode 与 prefill 图的 dispatch 都绑定到 top-k 选中掩码。"""
    compiled, _ = moe_compiled
    _assert_dispatch_reads_selected_mask(
        compiled.decode_gm, compiled.decode_plans[0])
    _assert_dispatch_reads_selected_mask(
        compiled.prefill_gm,
        compiled.prefill_plans[compiled.prefill_seq_len][0])


def _decode_three_tokens(compiled, *, moe_dispatch=None):
    """装权重跑 3 步贪心 decode，返回 (token 列表, backend)。"""
    import numpy as np
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=4, mram_bytes_per_dpu=1 << 28,
        moe_dispatch=moe_dispatch))
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


def test_dispatch_disabled_matches_enabled(moe_compiled) -> None:
    """关闭动态分派（退回稠密模拟）与开启时生成一致——跳过是纯运行期行为。

    两种语义跑的是同一份已编译计划：计划始终携带 gate 标签，后端配置决定
    是否执行跳过，A/B 才能只差一个变量。
    """
    enabled, _ = _decode_three_tokens(moe_compiled[0])
    disabled, _ = _decode_three_tokens(moe_compiled[0], moe_dispatch=False)
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
    """被跳过的专家 launch 把输出区写零：旧缓冲里的 inf 不会以 inf×0=nan 毒化下游。

    只有计划里标了 `skip_zero`（输出会被不同的门控/无门控读者读到）的命令
    才付这份清零成本；同专家子图内的读者会一起被跳过。
    """
    import numpy as np

    from contracts.exec_plan import Access

    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    # 预先把输出区填成 inf（模拟上一步溢出留下的脏值）。
    offset = 0
    dirty = np.full(16, np.inf, dtype=np.float16)
    hal.write_local(0, offset, dirty)
    cmd = Command(
        id=1, op="launch", dpu_id=0,
        payload={"kernel": "count", "gate": 7, "expert": 3,
                 "skip_zero": True},
        writes=[Access(("dpu", 0), offset, dirty.nbytes)])
    hal.wait(hal.submit(cmd))
    assert counter["runs"] == 0
    got = hal.read_local(0, offset, (16,), np.float16)
    assert np.all(got == 0), f"输出区应为零，得 {got[:4]}"


def test_gate_skipped_launch_without_skip_zero_leaves_the_buffer() -> None:
    """同专家子图的读者会一起跳过，这类命令不必付清零成本。"""
    import numpy as np

    from contracts.exec_plan import Access

    hal, counter = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
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
    assert np.all(np.isinf(got)), "未标 skip_zero 时不应多写一遍零"


# ---------------------------------------------------------------- 分块与子图门控

def _tiny_tiled_moe(tmp_path_factory):
    """2 层 × 2 专家的小 MoE；hidden 16 使很小 WRAM 就能触发图级分块。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(5)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=8, hidden_size=16, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, vocab_size=32, num_experts=2, num_experts_per_tok=1,
        norm_topk_prob=True)).eval().to(torch.float16)
    path = tmp_path_factory.mktemp("qwen3_moe_tiled")
    model.save_pretrained(path)
    return model, path


def _compile_tiled(loaded, wram=128):
    from contracts.op_contract import PIMHardwareConfig
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2
    strategy = ARCHITECTURES_loaded_strategy(loaded)
    return compile_llama2(
        loaded, strategy, prefill_seq_len=2, max_seq=8,
        hw=HwBudget(mram_bytes=1 << 24, align=64, sys_reserve_bytes=1 << 20),
        hardware=PIMHardwareConfig(
            num_dpus=2, num_tasklets=4, mram_bytes_per_dpu=1 << 24,
            wram_bytes_per_dpu=wram, dma_align=64),
        num_request_slots=1, num_parallel_slots=1, graph_tiling=True)


def ARCHITECTURES_loaded_strategy(loaded):
    from backend.architectures import ARCHITECTURES
    return ARCHITECTURES["qwen3_moe"].strategy(2, loaded.config)


def _loaded_tiny_tiled(tmp_path_factory):
    from backend.architectures import ARCHITECTURES
    model, path = _tiny_tiled_moe(tmp_path_factory)
    return model, ARCHITECTURES["qwen3_moe"].load(str(path))


@pytest.fixture(scope="module")
def tiled_moe(tmp_path_factory):
    """分块后的 MoE 编译结果与未静态化的 HF 参考模型。"""
    model, loaded = _loaded_tiny_tiled(tmp_path_factory)
    return model, loaded, _compile_tiled(loaded)


def test_tiled_expert_subgraph_is_recognized_and_gated(tiled_moe) -> None:
    """分块插入 slice/to_copy/cat 后，整条专家子图仍被识别并打上 gate。"""
    import torch

    from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of

    _, _, compiled = tiled_moe
    nodes = list(compiled.decode_gm.graph.nodes)
    provenance = _expert_provenance(nodes)
    expert_linears = {
        node.name for node in nodes
        if node.target is torch.ops.aten.linear.default
        and _expert_tag_of(provenance, node) is not None
    }
    # 2 层 × 2 专家 × 3 个 linear 被分块拆成更多子块。
    assert len(expert_linears) > 2 * 2 * 3, (
        f"分块后只识别出 {len(expert_linears)} 个专家 linear；"
        "分块插入的 slice/to_copy 没有被穿透")

    commands = compiled.decode_plans[0].commands
    tagged_linears = {
        cmd.payload["node"] for cmd in commands
        if cmd.op == "launch" and "linear" in str(cmd.payload.get("kernel"))
        and "expert" in cmd.payload
    }
    assert expert_linears <= tagged_linears, (
        f"这些专家 linear 没有 gate：{sorted(expert_linears - tagged_linears)[:5]}")

    # 分支内的非 linear 命令（视图/逐元素/通信）也在标签覆盖范围内。
    tagged_nonlinear = {
        str(cmd.payload.get("kernel", cmd.op)) for cmd in commands
        if cmd.op == "launch" and "expert" in cmd.payload
        and "linear" not in str(cmd.payload.get("kernel"))
    }
    assert tagged_nonlinear, "只有 linear 被打标，专家子图没有整条门控"
    gated_comm = [cmd for cmd in commands
                  if cmd.op != "launch" and "gate" in cmd.payload]
    assert any(cmd.op in ("host_reduce", "host_concat", "dpu_slice")
               for cmd in gated_comm), "专家通信命令没有 gate"


def test_tiled_dispatch_tokens_match_hf(tiled_moe) -> None:
    """强制分块下动态分派仍与 HF 原始稀疏前向逐 token 一致。"""
    import numpy as np
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    model, _, compiled = tiled_moe
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 24))
    register_all(backend)
    load_weights(compiled, backend)
    prompt = torch.tensor([1, 2])
    got = run_decode_loop(
        compiled, backend, prompt_ids=prompt, max_new_tokens=2, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)
    ref = model.generate(prompt.unsqueeze(0), max_new_tokens=2, do_sample=False)
    assert got == ref[0][len(prompt):].tolist()
    assert backend.gate_skips
    # prefill 两个 token 可能命中全部专家；decode 每层应恰好跳过 1 个。
    assert any(len(skips) == 1 for skips in backend.gate_skips.values()), (
        f"没有任何一层在 decode 跳过 1 个专家："
        f"{ {k: sorted(v) for k, v in backend.gate_skips.items()} }")
    assert all(len(skips) <= 1 for skips in backend.gate_skips.values())
    # 强制分块下同样要逐层生效：每层的 dispatch 都得挂上本层命令。
    dispatch, gated = _gate_assignments(compiled.decode_plans[0].commands)
    assert {layer for _, layer in gated} == set(range(2)), (
        f"分块后只有部分层被门控：{sorted({layer for _, layer in gated})}")


def test_tiled_dispatch_skips_expert_communication(
        tmp_path_factory, monkeypatch) -> None:
    """未命中专家的归约/收集命令真的不执行：通信次数随分派下降。"""
    import numpy as np
    import torch

    import runtime.exec_plan_gen as epg
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    model, loaded = _loaded_tiny_tiled(tmp_path_factory)
    prompt = torch.tensor([1, 2])

    counts = {"all_reduce": 0}
    original = epg._REDISTRIBUTE_FN["all_reduce"]

    def counted(entry, engine):
        counts["all_reduce"] += 1
        return original(entry, engine)

    # 计数函数要在编译前装上：计划构建时就把归约原语闭包进命令。
    with monkeypatch.context() as patch:
        patch.setitem(epg._REDISTRIBUTE_FN, "all_reduce", counted)
        compiled = _compile_tiled(loaded)

    def run(dispatch: bool) -> int:
        counts["all_reduce"] = 0
        backend = NumpyBackend(NumpyBackendConfig(
            num_dpus=2, mram_bytes_per_dpu=1 << 24, moe_dispatch=dispatch))
        register_all(backend)
        load_weights(compiled, backend)
        run_decode_loop(
            compiled, backend, prompt_ids=prompt, max_new_tokens=2, eos_id=-1,
            sample_fn=lambda logits: int(np.argmax(logits)),
            causal_mask_of=causal_mask_of)
        return counts["all_reduce"]

    on = run(True)
    off = run(False)
    assert off > 0, "没有可比较的专家归约，用例失去意义"
    assert on < off, f"开启动态分派后通信次数没有下降：on={on}, off={off}"


def test_gate_skipped_host_command_returns_zero() -> None:
    """被跳过的通信命令不执行，但主机消费者拿到零张量而不是 None。"""
    import numpy as np

    from contracts.exec_plan import Command

    hal, counter = _backend_with_counter()
    hal.register_kernel("host_reduce", lambda *args: counter.__setitem__(
        "runs", counter["runs"] + 1))
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    cmd = Command(
        id=1, op="host_reduce", dpu_id=None,
        payload={"gate": 7, "expert": 3,
                 "skip_value": [[2, 4], "float16"]})
    event = hal.submit(cmd)
    hal.wait(event)
    assert counter["runs"] == 0
    result = event.future.result()
    assert result.shape == (2, 4) and result.dtype == np.float16
    assert np.all(result == 0)


def test_gate_skipped_host_gather_does_not_transfer() -> None:
    """未命中的 host 收集命令直接返回零张量，不产生跨设备搬运。"""
    import numpy as np

    from contracts.exec_plan import Command

    hal, _ = _backend_with_counter()
    hal.gate_skips[(0, 7)] = {3}
    _submit_dispatch(hal)
    transfers = []

    def gather(backend, command):
        transfers.append(command.id)
        return np.ones((2, 4), dtype=np.float16)

    def submit(expert: int) -> object:
        command = Command(
            id=1, op="host_concat", dpu_id=None,
            payload={"gate": 7, "expert": expert, "fn": gather,
                     "skip_value": [[2, 4], "float16"]})
        event = hal.submit(command)
        hal.wait(event)
        del hal._events_by_id[command.id]
        return event.future.result()

    skipped = submit(3)
    assert transfers == [], "未命中专家不应走收集搬运"
    assert skipped.shape == (2, 4) and np.all(skipped == 0)
    # 命中专家照常走收集，证明上面的空列表不是因为路径失效。
    hit = submit(5)
    assert transfers == [1]
    assert np.all(hit == 1)


def test_benchmark_harness_flips_modes_on_one_backend(moe_compiled) -> None:
    """基准入口的计时口径：同一份计划与后端，只切换运行期分派开关。

    `scripts/bench_moe_dispatch.py` 先预填计时再取中位数；这里验证它的
    执行循环真的能在同一个后端上跑两种语义，且 token 一致、动态模式确实
    写入了跳过集。
    """
    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import load_weights
    from runtime.kernels import register_all
    from scripts.bench_moe_dispatch import _hit_expert_counts, _run_steps

    compiled, _ = moe_compiled
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=4, mram_bytes_per_dpu=1 << 28, moe_dispatch=False))
    register_all(backend)
    load_weights(compiled, backend)

    dense_tokens, _, _ = _run_steps(compiled, backend, [1, 2, 3, 4], 2)
    backend.config.moe_dispatch = True
    backend.moe_dispatch = True
    dynamic_tokens, _, _ = _run_steps(compiled, backend, [1, 2, 3, 4], 2)

    assert dense_tokens == dynamic_tokens
    assert backend.gate_skips, "动态模式下应写入跳过集"
    assert _hit_expert_counts(backend), "命中专家数应可报告"
    from scripts.bench_moe_dispatch import _dispatch_coverage
    coverage = _dispatch_coverage(backend)
    assert coverage, "逐层命令覆盖应可报告"
    assert all(v["submitted"] > 0 and v["skipped"] > 0
               for v in coverage.values()), (
        f"有 dispatch 没拦下或没放行命令，逐层覆盖不成立：{coverage}")


def test_dispatch_skips_expert_subgraph_commands_ratio(tiled_moe) -> None:
    """剔除装载/编译后，未命中专家的整条子图命令按比例不执行。

    这是执行期加速的确定性口径：命令总数与跳过集合都来自同一份计划，
    不受计时噪声影响；decode 每层跳掉一部分专家，gated 命令的跳过比例
    应与 top-k/E 同量级。
    """
    import numpy as np
    import torch

    from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
    from runtime.compile import causal_mask_of, load_weights
    from runtime.executor import run_decode_loop
    from runtime.kernels import register_all

    _, _, compiled = tiled_moe
    backend = NumpyBackend(NumpyBackendConfig(
        num_dpus=2, mram_bytes_per_dpu=1 << 24))
    register_all(backend)
    load_weights(compiled, backend)
    run_decode_loop(
        compiled, backend, prompt_ids=torch.tensor([1, 2]),
        max_new_tokens=2, eos_id=-1,
        sample_fn=lambda logits: int(np.argmax(logits)),
        causal_mask_of=causal_mask_of)

    commands = compiled.decode_plans[0].commands
    gated = [cmd for cmd in commands if "expert" in cmd.payload]
    assert gated, "计划里没有专家子图命令"
    skip_by_gate: dict = {}
    for (_lane, dispatch), skipped in backend.gate_skips.items():
        skip_by_gate.setdefault(dispatch, set()).update(skipped)
    executed = sum(
        1 for cmd in gated
        if cmd.payload["expert"] not in skip_by_gate.get(cmd.payload["gate"], ()))
    skipped = len(gated) - executed
    # 2 专家 top-1、2 层：decode 每层恰好跳 1 个专家，整片子图命令
    # 应有一半左右不执行；prefill 可能命中全部专家，所以不用精确值。
    assert 0 < skipped < len(gated), (
        f"gated 命令 {len(gated)} 条，跳过 {skipped} 条")
    assert 0.2 < skipped / len(gated) < 0.8, (
        f"跳过比例 {skipped / len(gated):.2f} 不在预期量级")


# ---------------------------------------------------------------- 跨层传播不变式

def _dispatch_layer(dispatch_name: str) -> int:
    """`moe_dispatch_L` → L。"""
    return int(str(dispatch_name).rsplit("_", 1)[1])


def _gate_assignments(cmds):
    """返回 {dispatch 命令号: dispatch 名} 与 [(命令, 层号)] 的门控归属。"""
    dispatch = {
        cmd.id: str(cmd.payload.get("node", "")) for cmd in cmds
        if cmd.op == "host_op"
        and str(cmd.payload.get("node", "")).startswith("moe_dispatch")
    }
    gated = [
        (cmd, _dispatch_layer(dispatch[cmd.payload["gate"]]))
        for cmd in cmds
        if "gate" in cmd.payload and "expert" in cmd.payload
    ]
    return dispatch, gated


def test_every_layer_gates_its_own_experts(moe_compiled) -> None:
    """每一层的 dispatch 都必须拦下本层专家子图的命令，且标签层号与 dispatch 一致。

    旧实现让上一层的多专家累加状态穿过层边界：第 0 层累加结果变成多专家标签，
    第 1 层专家权重与它相遇后整层被判成多专家，全部失去 gate。只数 dispatch
    个数（每层都有）或总跳过比例（首层单独生效也能凑够比例）都发现不了。
    """
    from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of

    compiled = moe_compiled[0]
    for gm, plan in (
            (compiled.decode_gm, compiled.decode_plans[0]),
            (compiled.prefill_gm, compiled.prefill_plans[compiled.prefill_seq_len][0])):
        nodes = {node.name: node for node in gm.graph.nodes}
        provenance = _expert_provenance(list(nodes.values()))
        dispatch, gated = _gate_assignments(plan.commands)
        layers = sorted({_dispatch_layer(name) for name in dispatch.values()})
        assert layers == list(range(len(dispatch))), (
            f"dispatch 层号不连续：{sorted(dispatch.values())}")
        per_layer: dict[int, int] = {}
        for cmd, layer in gated:
            if cmd.op == "launch":
                tag = _expert_tag_of(provenance, nodes[cmd.payload["node"]])
                assert tag is not None, (
                    f"命令 {cmd.id}（{cmd.payload.get('node')}）带 gate 却没有"
                    "专家归属：计划与源图不一致")
                assert tag == (layer, cmd.payload["expert"]), (
                    f"命令 {cmd.id}（{cmd.payload.get('node')}）挂到第 {layer} 层 "
                    f"dispatch，源图归属却是 {tag}")
            per_layer[layer] = per_layer.get(layer, 0) + 1
        assert set(per_layer) == set(layers), (
            f"这些层没有任何命令挂到本层 dispatch："
            f"{sorted(set(layers) - set(per_layer))}；实际 {per_layer}")
        # 8 专家 ×（gate/up/down + silu + 专家内乘 + where + 路由加权乘）
        # 至少 7 条，分块/切分只会更多。
        for layer, count in per_layer.items():
            assert count >= 8 * 7, (
                f"第 {layer} 层只有 {count} 条命令被门控，专家子图没有整条覆盖")


def test_moe_tail_is_host_shared_work(moe_compiled) -> None:
    """选择/加权累加是主机共享工作，设备侧只剩可门控的专家子图。

    PIM 原语集没有逐元素 select（FlagTree 只有 eltwise 的加减乘除与加性
    mask），把 `where` 留在设备上等于借一个语义不同的原语，还把这段设备工作
    排除在 pimir 覆盖之外。计划里选择与随后融合的加权累加必须一起落在主机、
    不挂 gate；设备侧的专家计算与视图则整条带 gate。
    """
    import torch

    from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of

    compiled = moe_compiled[0]
    nodes = {node.name: node for node in compiled.decode_gm.graph.nodes}
    commands = compiled.decode_plans[0].commands
    by_node = {cmd.payload.get("node"): cmd for cmd in commands}

    selects = [node for node in nodes.values()
               if node.target in (torch.ops.aten.ne.Scalar,
                                  torch.ops.aten.where.ScalarOther)]
    assert selects, "静态 MoE 前向应有路由掩码与选择节点"
    for node in selects:
        cmd = by_node.get(node.name)
        assert cmd is not None and cmd.op == "host_op", (
            f"选择节点 {node.name} 落在了设备上：没有对应原语")
        assert "gate" not in cmd.payload, (
            f"选择节点 {node.name} 被门控：它和加权累加在同一段主机闭包里")

    provenance = _expert_provenance(list(nodes.values()))
    ungated = [
        cmd for cmd in commands
        if cmd.op == "launch"
        and _expert_tag_of(provenance, nodes[cmd.payload["node"]]) is not None
        and "gate" not in cmd.payload
    ]
    assert not ungated, (
        f"这些专家子图命令没有 gate：{[c.payload['node'] for c in ungated[:5]]}")


@pytest.fixture(scope="module")
def moe_pp_compiled(tmp_path_factory):
    """4 层 × 4 专家小 MoE，2 DPU × 2 段：层跨流水段。"""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    from backend.architectures import ARCHITECTURES
    from contracts.op_contract import PIMHardwareConfig
    from graph.strategy import llama_strategy
    from memory.mem_planner import HwBudget
    from runtime.compile import compile_llama2

    torch.manual_seed(5)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(
        max_position_embeddings=16, hidden_size=64, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True)).eval().to(torch.float16)
    path = tmp_path_factory.mktemp("qwen3_moe_pp")
    model.save_pretrained(path)
    loaded = ARCHITECTURES["qwen3_moe"].load(str(path))
    strategy = llama_strategy(
        2, num_stages=2, num_heads=4, num_kv_heads=2,
        intermediate_size=32, vocab_size=128, num_layers=4)
    hw = HwBudget(mram_bytes=1 << 28, align=64, sys_reserve_bytes=1 << 20)
    hardware = PIMHardwareConfig(
        num_dpus=2, num_tasklets=4, mram_bytes_per_dpu=hw.mram_bytes,
        wram_bytes_per_dpu=65536, dma_align=64)
    compiled = compile_llama2(
        loaded, strategy, prefill_seq_len=4, max_seq=16,
        hw=hw, hardware=hardware, num_request_slots=1, num_parallel_slots=1)
    return compiled, strategy


def test_pp_stages_each_gate_their_own_layers(moe_pp_compiled) -> None:
    """跨流水段：每个 stage 的层都必须挂到本层 dispatch，不能只在首段生效。"""
    from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of

    compiled, strategy = moe_pp_compiled
    assert strategy.num_stages == 2
    nodes = {node.name: node for node in compiled.decode_gm.graph.nodes}
    provenance = _expert_provenance(list(nodes.values()))
    commands = compiled.decode_plans[0].commands
    dispatch, gated = _gate_assignments(commands)
    gated_layers = {layer for _, layer in gated}
    # 层 0/1 在 stage0（dpu0），层 2/3 在 stage1（dpu1）；四层都要有命令。
    assert gated_layers == {0, 1, 2, 3}, f"缺门控的层：{gated_layers}"
    for cmd, layer in gated:
        if cmd.op == "launch":
            tag = _expert_tag_of(provenance, nodes[cmd.payload["node"]])
            assert tag is not None and tag == (layer, cmd.payload["expert"])
        expected_dpu = 0 if layer < 2 else 1
        if cmd.dpu_id is not None:
            assert cmd.dpu_id == expected_dpu, (
                f"层 {layer} 的命令 {cmd.id} 落在 dpu{cmd.dpu_id}，"
                f"应为 dpu{expected_dpu}")
    # 两段各自至少拦下一条专家计算命令。
    dispatch_by_layer = {_dispatch_layer(name): cid for cid, name in dispatch.items()}
    for stage_layers in ((0, 1), (2, 3)):
        assert any(cmd.payload["gate"] in {dispatch_by_layer[l] for l in stage_layers}
                   for cmd, layer in gated if layer in stage_layers)


def test_backend_records_per_dispatch_execution(moe_compiled) -> None:
    """后端按 dispatch 记录实际提交/未提交的命令数，基准据此报告逐层覆盖。

    只统计 dispatch 写出的跳过集合看不出「这一层的命令根本没挂上来」；
    计数必须落在提交侧，才能证明每个 dispatch 都真的拦下了命令。
    """
    _, backend = _decode_three_tokens(moe_compiled[0])
    stats = backend.gate_stats
    dispatches = {dispatch_id for (_lane, dispatch_id) in stats}
    assert len(dispatches) == 4, f"只有 {len(dispatches)} 个 dispatch 有记录"
    for dispatch_id in dispatches:
        entries = [v for (lane, d), v in stats.items() if d == dispatch_id]
        assert any(v["submitted"] > 0 for v in entries), (
            f"dispatch {dispatch_id} 没有放行任何命令：这一层没有专家计算")
        assert any(v["skipped"] > 0 for v in entries), (
            f"dispatch {dispatch_id} 没有拦下任何命令：gate 没有生效")


def test_plan_and_genesim_ir_agree_on_moe_roles(moe_compiled, tmp_path) -> None:
    """跨仓对拍：FX 计划的落点/门控与 GeneSim IR 的成本角色逐项一致。

    IR 声明为 `expert_work` 的 (层, 专家) 组合必须整条带 gate；IR 声明为
    host `shared_reduce` 的选择/加权累加必须落在主机命令上。只对总时延或
    「trace 全是 pimir」下结论看不出这类落点错配。
    """
    import sys

    import torch

    from genesim_bridge.paths import genesim_root
    from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of

    src = genesim_root() / "src"
    # 追加而不是插到最前：两仓都有 `backend` 包，遮住本仓的会让 vLLM
    # 子进程导入不到 `backend.vllm_executor`。
    if str(src) not in sys.path:
        sys.path.append(str(src))
    from ir.model_ir import build_from_hf_config

    compiled, loaded = moe_compiled
    config_path = tmp_path / "config.json"
    loaded.config.to_json_file(str(config_path))
    ir = build_from_hf_config(
        str(config_path), execution_dtype="float16",
        moe_execution="dynamic_dispatch")

    ir_work = {
        (layer, int(op.expert_id))
        for layer, group in enumerate(ir.subgraphs)
        for op in (ir.get_operator(op_id) for op_id in group)
        if op.moe_cost_role == "expert_work"
    }
    ir_tail = [
        op for op in ir.operators if op.op_type == "MOE_ACCUM"
    ]
    assert len(ir_tail) == int(ir.num_layers) * int(ir.moe_num_experts)
    for op in ir_tail:
        assert op.device_hint == "cpu" and op.moe_cost_role == "shared_reduce"

    nodes = {node.name: node for node in compiled.decode_gm.graph.nodes}
    provenance = _expert_provenance(list(nodes.values()))
    commands = compiled.decode_plans[0].commands
    plan_gated = {
        _expert_tag_of(provenance, nodes[cmd.payload["node"]])
        for cmd in commands
        if cmd.op == "launch" and "gate" in cmd.payload
    }
    plan_gated.discard(None)
    assert plan_gated == ir_work, (
        "计划门控的 (层, 专家) 与 IR 的 expert_work 不一致："
        f"计划多 {sorted(plan_gated - ir_work)[:5]}，"
        f"IR 多 {sorted(ir_work - plan_gated)[:5]}")

    selects = [node for node in nodes.values()
               if node.target in (torch.ops.aten.ne.Scalar,
                                  torch.ops.aten.where.ScalarOther)]
    host_nodes = {cmd.payload.get("node") for cmd in commands
                  if cmd.op == "host_op"}
    assert selects and {node.name for node in selects} <= host_nodes, (
        "IR 把选择/加权累加记为 host，计划里这些节点却不是主机命令")
