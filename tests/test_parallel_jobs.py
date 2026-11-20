"""并行编译工具与唯一键预测量的行为验证。

判据分两层：`run_jobs` 本身的顺序与并行度语义；`export_costs_to_genesim`
在并行档与串行档产出完全一致、且每个唯一键只测一次、pimir 档的非 GEMM
键不进线程池（GPU 捕获不可并发）。
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge import cost_extractor as ce
from genesim_bridge.cost_extractor import Measurement, export_costs_to_genesim
from opcompiler_bridge.parallel import job_workers, run_jobs


def test_job_workers_reads_env(monkeypatch):
    monkeypatch.delenv("OPCOMPILER_JOBS", raising=False)
    assert job_workers() == 32
    monkeypatch.setenv("OPCOMPILER_JOBS", "4")
    assert job_workers() == 4
    # 非法值退回默认，0 夹到 1：并行度是资源参数，不该让坏值炸掉编译。
    monkeypatch.setenv("OPCOMPILER_JOBS", "not-a-number")
    assert job_workers() == 32
    monkeypatch.setenv("OPCOMPILER_JOBS", "0")
    assert job_workers() == 1


def test_run_jobs_keeps_submission_order(monkeypatch):
    """结果按提交顺序返回：聚合逻辑（如 widths.setdefault）才与串行版一致。"""
    monkeypatch.setenv("OPCOMPILER_JOBS", "8")
    rng = random.Random(0)

    def work(i):
        # 各任务耗时乱序，验证不是「谁先完成谁在前」。
        time.sleep(rng.uniform(0, 0.01))
        return i * i

    assert run_jobs(work, range(40)) == [i * i for i in range(40)]


def test_run_jobs_small_inputs(monkeypatch):
    monkeypatch.setenv("OPCOMPILER_JOBS", "1")
    assert run_jobs(lambda x: x + 1, []) == []
    assert run_jobs(lambda x: x + 1, [7]) == [8]
    assert run_jobs(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]


def _tiny_ir() -> dict:
    """两同键 + 一异键的小 IR：SOFTMAX 两个共享测量，RMSNORM 各自一份。"""
    return {
        "hidden_size": 64, "head_dim": 16, "num_heads": 4, "num_layers": 1,
        "operators": [
            {"op_id": 0, "op_type": "SOFTMAX",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]]},
            {"op_id": 1, "op_type": "SOFTMAX",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]]},
            {"op_id": 2, "op_type": "RMSNORM",
             "input_shapes": [["Tq", 64]], "output_shapes": [["Tq", 64]]},
        ],
        "dependencies": [], "subgraphs": [[0, 1, 2]],
    }


def _fake_measure(calls):
    """替身测量：不碰工具链，记下每次调用的键身份，回一个可拟合的数。"""
    def measure(op_dict, point, dims, ir_level, local_features=None,
                hardware=None, force_compile=False):
        calls.append((op_dict["op_type"], point.tq, point.tp))
        return Measurement(
            point=point,
            flops=float(10 * point.tq + point.tp),
            data_bytes=float(4 * point.tq + point.tp),
            dtype="f16", element_bytes=2,
        )
    return measure


def _write_ir(tmp_path: Path) -> Path:
    ir_path = tmp_path / "in.ir"
    ir_path.write_text(json.dumps(_tiny_ir()))
    return ir_path


def test_export_costs_parallel_matches_serial(tmp_path, monkeypatch):
    """并行档与串行档的精化 IR、sidecar 必须逐字节等价，且每键只测一次。"""
    ir_path = _write_ir(tmp_path)
    calls = []
    monkeypatch.setattr(ce, "_measure", _fake_measure(calls))

    outputs = {}
    per_run_calls = {}
    for jobs in ("1", "8"):
        monkeypatch.setenv("OPCOMPILER_JOBS", jobs)
        calls.clear()
        out = tmp_path / f"out{jobs}.ir"
        side = tmp_path / f"side{jobs}.json"
        export_costs_to_genesim(
            ir_path=ir_path, out_ir_path=out, sidecar_path=side,
            seq_len=128, cross_validate=False, ir_level="oplevel",
        )
        outputs[jobs] = (out.read_text(), side.read_text())
        per_run_calls[jobs] = sorted(calls)
    assert outputs["1"] == outputs["8"]
    # 两个唯一键 × 两个形状点 = 4 次测量；op1 复用 op0 的结果。
    expected = [("RMSNORM", 1, 128), ("RMSNORM", 128, 0),
                ("SOFTMAX", 1, 128), ("SOFTMAX", 128, 0)]
    assert per_run_calls["1"] == expected
    assert per_run_calls["8"] == expected


def test_pimir_non_gemm_keys_stay_serial(tmp_path, monkeypatch):
    """pimir 档的非 GEMM 键必须留在串行侧：FlagGems 捕获是全局补丁，不可并发。"""
    import opcompiler_bridge.parallel as parallel

    dispatched = []
    original = parallel.run_jobs

    def spy(fn, items):
        items = list(items)
        dispatched.extend(items)
        return original(fn, items)

    monkeypatch.setattr(parallel, "run_jobs", spy)
    monkeypatch.setattr(ce, "_measure", _fake_measure([]))
    monkeypatch.setenv("OPCOMPILER_JOBS", "8")
    export_costs_to_genesim(
        ir_path=_write_ir(tmp_path),
        out_ir_path=tmp_path / "out.ir", sidecar_path=tmp_path / "sc.json",
        seq_len=128, cross_validate=False, ir_level="pimir",
    )
    assert dispatched == [], "非 GEMM 键不该进线程池"
