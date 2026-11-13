"""验证 D1 事件驱动调度器：shadow 依赖、动态执行与 gate 解析。"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import Event, NumpyBackend, NumpyBackendConfig
from contracts.exec_plan import Command, ExecutionPlan
from graph.spec_prop import propagate_specs
from runtime.dependency import address_wait_gaps, address_waits
from runtime.executor import execute_plan
from runtime.kernels import register_all
from tests.test_exec_plan_gen import (
    _appendix_a_config,
    _built_appendix_a,
    _plan_and_compile,
)


def _done_future(value=None) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


class _FakeHAL:
    """最小的调度器 HAL：记录提交/跳过，gate 命令的完成由测试控制。"""

    def __init__(self, gate_future: Future) -> None:
        self.events: dict[int, Event] = {}
        self.submitted: list[int] = []
        self.skipped: list[int] = []
        self.gate_future = gate_future
        self.moe_dispatch = True
        self.gate_skips: dict[tuple, set[int]] = {}
        self.gate_stats: dict[tuple, dict[str, int]] = {}
        self.submitted_event = threading.Event()

    def events_table(self, ctx=None):
        return self.events

    def submit(self, cmd: Command, ctx=None) -> Event:
        self.submitted.append(cmd.id)
        self.submitted_event.set()
        future = self.gate_future if cmd.id == 0 else _done_future()
        event = Event("host", future)
        self.events[cmd.id] = event
        return event

    def make_skipped_event(self, cmd, lane_key, events, ctx=None) -> Event:
        self.skipped.append(cmd.id)
        event = Event("host", _done_future())
        events[cmd.id] = event
        return event


def _gate_plan() -> ExecutionPlan:
    return ExecutionPlan([
        Command(0, "host_op", None, {"node": "gate"}),
        Command(1, "host_op", None,
                {"node": "expert", "gate": 0, "expert": 3}, waits=[0]),
        Command(2, "host_op", None, {"node": "downstream"}, waits=[1]),
    ])


def test_address_waits_are_a_subset_of_static_waits() -> None:
    gm, nodes_by_key, _ = _built_appendix_a()
    del nodes_by_key
    edges = propagate_specs(gm, _appendix_a_config())
    compiled, _ = _plan_and_compile(gm, edges)
    runtime = address_waits(compiled.plan.commands)
    assert address_wait_gaps(compiled.plan.commands, runtime) == []
    assert any(runtime.values()), "附录 A 计划应至少有一条地址依赖"


def test_dynamic_execution_matches_static_on_appendix_a() -> None:
    gm, nodes_by_key, _ = _built_appendix_a()
    for node in gm.graph.nodes:
        if "layer_norm" in str(node.target):
            node.target = lambda x, *a, orig=node.target, **k: orig(
                torch.from_numpy(np.asarray(x)), *a, **k).numpy()
    edges = propagate_specs(gm, _appendix_a_config())
    compiled, plans = _plan_and_compile(gm, edges)

    torch.manual_seed(0)
    gm.w1.copy_(torch.randn(6, 4))
    gm.w2.copy_(torch.randn(4, 6))
    w1n, w2n = gm.w1.detach().numpy(), gm.w2.detach().numpy()
    x_val = torch.randn(1, 4)

    def run(mode: str):
        backend = NumpyBackend(
            NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
        register_all(backend)
        for dpu_id in (0, 1):
            w1_off = plans[dpu_id].weight[nodes_by_key["w1"].target]
            w2_off = plans[dpu_id].weight[nodes_by_key["w2"].target]
            backend.write_local(
                dpu_id, w1_off,
                w1n[dpu_id * 3:(dpu_id + 1) * 3].astype(np.float32))
            backend.write_local(
                dpu_id, w2_off,
                w2n[:, dpu_id * 3:(dpu_id + 1) * 3].astype(np.float32))
        execute_plan(compiled.plan, backend, values={"x": x_val.numpy()},
                     mode=mode)
        return backend.result_of(compiled.output_cmd_id)

    static_result = run("static")
    dynamic_result = run("dynamic")
    ref = torch.nn.functional.layer_norm(
        torch.nn.functional.linear(
            torch.nn.functional.linear(x_val, gm.w1), gm.w2),
        [4], None, None, 1e-5,
    ).detach().numpy()
    assert np.allclose(static_result, ref, atol=1e-4)
    assert np.allclose(dynamic_result, static_result, atol=1e-6)


def test_scheduler_submits_all_commands_in_plan_order() -> None:
    """计划序全量提交：gate 由 HAL 在 submit 内同步解析，调度器不拦提交。"""
    gate_future: Future = Future()
    hal = _FakeHAL(gate_future)
    from runtime.scheduler import EventScheduler

    scheduler = EventScheduler(_gate_plan(), hal)
    scheduler.run(drain=False)
    assert hal.submitted_event.wait(1.0)
    assert hal.submitted == [0, 1, 2]
    assert hal.skipped == []
    gate_future.set_result(None)
    scheduler.wait()
    assert hal.gate_stats == {}  # 跳过判定在 HAL 的 submit 里，不在调度器。
