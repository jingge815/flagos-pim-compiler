"""外层循环：预算门控、确定性续跑与失败缓存。"""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from contracts.hardware_profile import load_profile
from tune.candidate import CandidateSpace, OpKnobs, StrategyChoice, TuneState, model_dims
from tune.config import load_config
from tune.loop import DeadlineReached, read_responses, run_tune
from tune.report import ArtifactManifest, RoundRecord, read_rounds

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"


def _config(tmp_path, **search):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 32, "vocab_size": 64,
        "num_hidden_layers": 4, "hidden_size": 16,
    }))
    ir = tmp_path / "model.ir"
    ir.write_text('{"operators": []}')
    raw = {
        "version": 1, "run_name": "loop",
        "model": {"model_dir": str(model_dir), "ir": str(ir),
                  "prefill_seq_len": 16, "max_seq": 32,
                  "trace": {"path": str(tmp_path / "trace.json"),
                            "num_requests": 1, "synthetic": None}},
        "profile": {"path": str(PROFILE)},
        "space": {"strategy": {"enabled": True, "num_dpus": [4, 8],
                               "num_stages": [2, 4]},
                  "op": {"enabled": True, "num_tasklets": [4, 8]},
                  "hardware": {"enabled": False, "fields": {}}},
        "placement": {"enabled": False, "algorithm": "simulated_annealing",
                      "evaluation_backend": "pipeline",
                      "validation_backend": "pipeline",
                      "objective_metric": "total_time_s",
                      "objective_weights": {"compute": 0.5,
                                            "communication": 0.25,
                                            "memory": 0.25},
                      "max_evaluations": 1, "max_time_s": 0,
                      "max_proposals": None, "seed": 0,
                      "moves": {"max_segment_ops": 8,
                                "max_layer_block_layers": 4, "weights": {}},
                      "algorithm_params": {}},
        "search": {"algorithm": "enumerate", "seed": 0,
                   "max_evaluations": 2, "max_time_s": 0,
                   "max_proposals": None,
                   "layer_weights": {"strategy": 0.5, "op": 0.5, "hardware": 0},
                   "algorithm_params": {}},
        "cache": {"mode": "fast"},
        "simulation": {"use_cost_sidecar": False},
        "calibration": {"validation_report": None},
        "sweep": {"fields": {}, "core_policy": "fixed_dpus",
                  "knee_improvement_threshold": 0.05,
                  "pareto": {"resource": "total_physical_memory_bytes",
                             "direction": "minimize"}},
        "output": {"dir": str(tmp_path / "out")},
    }
    (tmp_path / "trace.json").write_text(json.dumps({"requests": [{
        "request_id": "r0", "input_tokens": list(range(8)),
        "output_length": 2, "arrival_time_ns": 0, "is_prefill": True,
        "seq_len": 8, "prompt": None, "padded_query_length": None,
        "initial_kv_length": 0}]}))
    raw["search"].update(search)
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def _space(cfg):
    return CandidateSpace(cfg, model_dims(cfg.model.model_dir))


def _fake_evaluator(calls, *, fail_at=None):
    def evaluate(state):
        calls.append((state.strategy.num_dpus, state.strategy.num_stages))
        if fail_at is not None and len(calls) == fail_at:
            return RoundRecord(
                sequence=0, candidate_id="fail", compile_key="", sim_key="",
                cache="fast", status="error", result_maturity="model_prediction",
                state=state, score=None, failure="injected")
        score = float(10 - state.strategy.num_dpus)
        return RoundRecord(
            sequence=0, candidate_id=f"c{state.strategy.num_dpus}_"
                                    f"{state.strategy.num_stages}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=score,
            metrics={"total_time_s": score})

    return evaluate


def test_enumerate_resume_replays_committed_calls(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    calls = []
    first = run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert first.actual_calls == 2
    assert first.logical_calls == 2
    assert calls

    calls_again = []
    resumed = run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls_again))
    assert calls_again == [], "已提交的调用不应重复执行"
    assert resumed.resumed_calls == 2
    assert resumed.actual_calls == 0
    assert resumed.best.candidate_id == first.best.candidate_id

    wider = replace(cfg, search=replace(cfg.search, max_evaluations=4))
    more_calls = []
    continued = run_tune(wider, None, space,
                         evaluate_fn=_fake_evaluator(more_calls))
    assert len(more_calls) == 2, "只执行日志尾部的新项"
    assert continued.resumed_calls == 2
    assert continued.logical_calls == 4
    assert read_responses(cfg.output.dir / "responses.jsonl")[1].state_key


def test_search_coverage_separates_new_pipeline_from_cache_hits(tmp_path):
    """缓存命中与回放不能当作本算法的新真实评估。"""
    cfg = _config(tmp_path)
    space = _space(cfg)
    seen = set()

    def evaluate(state):
        key = (state.strategy.num_dpus, state.strategy.num_stages)
        cached = key in seen
        seen.add(key)
        score = float(10 - state.strategy.num_dpus)
        return RoundRecord(
            sequence=0, candidate_id=f"c{key[0]}_{key[1]}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=score,
            compile_cache_hit=cached, sim_cache_hit=cached,
            inner_evaluations=0 if cached else 2,
            validation_evaluations=1,
            paths={} if cached else {"effective_placement": "placement.json"},
            metrics={"total_time_s": score})

    result = run_tune(cfg, None, space, evaluate_fn=evaluate)
    coverage = result.coverage
    assert coverage.algorithm == "enumerate"
    assert coverage.actual_calls == 2
    assert coverage.new_pipeline_evaluations == 1
    assert coverage.compile_cache_hits == 1
    assert coverage.sim_cache_hits == 1
    assert coverage.new_backend_executions == 3
    assert coverage.final_mapping_replays == 1
    assert coverage.real_search is True


def test_search_coverage_is_false_when_every_candidate_is_cached(tmp_path):
    cfg = _config(tmp_path)

    def evaluate(state):
        return RoundRecord(
            sequence=0, candidate_id=f"c{state.strategy.num_dpus}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=1.0,
            compile_cache_hit=True, sim_cache_hit=True)

    result = run_tune(cfg, None, _space(cfg), evaluate_fn=evaluate)
    coverage = result.coverage
    assert coverage.new_pipeline_evaluations == 0
    assert coverage.new_backend_executions == 0
    assert coverage.final_mapping_replays == 0
    assert coverage.real_search is False


def test_search_coverage_is_zero_on_pure_resume(tmp_path):
    cfg = _config(tmp_path)

    def evaluate(state):
        return RoundRecord(
            sequence=0, candidate_id=f"c{state.strategy.num_dpus}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=1.0,
            validation_evaluations=1,
            paths={"effective_placement": "placement.json"})

    first = run_tune(cfg, None, _space(cfg), evaluate_fn=evaluate)
    assert first.coverage.real_search is True
    resumed = run_tune(cfg, None, _space(cfg), evaluate_fn=evaluate)
    assert resumed.resumed_calls == 2
    assert resumed.coverage.actual_calls == 0
    assert resumed.coverage.new_pipeline_evaluations == 0
    assert resumed.coverage.real_search is False


@pytest.mark.parametrize("algorithm", ["simulated_annealing",
                                       "genetic_algorithm", "tabu_search"])
def test_each_algorithm_records_a_new_pipeline_and_final_replay(
        tmp_path, algorithm):
    """三种算法各自发起新评估并重放最终映射，不能只靠缓存回放。"""
    cfg = _config(tmp_path, algorithm=algorithm, max_evaluations=2,
                  algorithm_params={"min_improvement": 0.0})
    space = _space(cfg)
    seen = set()

    def evaluate(state):
        key = space.key(state)
        cached = key in seen
        seen.add(key)
        score = float(10 - state.strategy.num_dpus)
        return RoundRecord(
            sequence=0, candidate_id=f"c{state.strategy.num_dpus}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=score,
            compile_cache_hit=cached, sim_cache_hit=cached,
            inner_evaluations=0 if cached else 2,
            validation_evaluations=1,
            paths={} if cached else {"effective_placement": "placement.json"},
            metrics={"total_time_s": score})

    result = run_tune(cfg, None, space, evaluate_fn=evaluate)
    coverage = result.coverage
    assert coverage.algorithm == algorithm
    assert coverage.actual_calls >= 1
    assert coverage.new_pipeline_evaluations >= 1
    assert coverage.final_mapping_replays >= 1
    assert coverage.real_search is True


def test_resume_order_mismatch_is_rejected(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))
    path = cfg.output.dir / "responses.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["state_key"] = "tampered"
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="恢复顺序"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))


@pytest.mark.parametrize("algorithm", ["simulated_annealing",
                                       "genetic_algorithm", "tabu_search"])
def test_algorithms_resume_with_same_prefix(tmp_path, algorithm):
    cfg = _config(tmp_path, algorithm=algorithm, max_evaluations=4,
                  algorithm_params={"min_improvement": 0.0})
    space = _space(cfg)
    calls = []
    first = run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert first.actual_calls == 4
    prefix = [record.state.strategy.num_stages for record in first.records]

    wider = replace(cfg, search=replace(cfg.search, max_evaluations=6))
    new_calls = []
    second = run_tune(wider, None, space,
                      evaluate_fn=_fake_evaluator(new_calls))
    all_records = read_rounds(cfg.output.dir / "rounds.jsonl")
    assert [record.state.strategy.num_stages for record in all_records][:4] == prefix
    assert second.resumed_calls == 4
    assert len(new_calls) == 2
    assert second.logical_calls == 6


def test_failed_candidate_is_cached_and_not_retried(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    calls = []
    first = run_tune(cfg, None, space,
                     evaluate_fn=_fake_evaluator(calls, fail_at=1))
    assert first.actual_calls == 2
    assert first.successful and len(first.successful) == 1

    calls_again = []
    resumed = run_tune(cfg, None, space,
                       evaluate_fn=_fake_evaluator(calls_again, fail_at=1))
    assert calls_again == []
    assert resumed.resumed_calls == 2


def test_deadline_stops_before_starting_new_evaluation(tmp_path):
    cfg = _config(tmp_path, algorithm="simulated_annealing",
                  max_evaluations=20, max_time_s=3.0,
                  algorithm_params={"min_improvement": 0.0})

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    calls = []

    def evaluate(state):
        clock.now += 1.0
        calls.append(state)
        return _fake_evaluator([])(state)

    result = run_tune(cfg, None, _space(cfg), evaluate_fn=evaluate, clock=clock)
    assert len(calls) == 3, "基线完成后，预算耗尽前不再发起新评估"
    assert result.stop_reason == "max_time_s"

    # 历史活动耗时计入总预算：同预算续跑直接汇总，不再发起新任务。
    again = []
    resumed = run_tune(cfg, None, _space(cfg),
                       evaluate_fn=_fake_evaluator(again), clock=Clock())
    assert again == []
    assert resumed.actual_calls == 0


def test_resume_replay_and_offline_time_are_not_charged_to_budget(tmp_path):
    """扩大预算后，回放与离线等待耗时不能吃掉新增的时间预算。"""
    cfg = _config(tmp_path, algorithm="simulated_annealing",
                  max_evaluations=20, max_time_s=3.0,
                  algorithm_params={"min_improvement": 0.0})

    class EvalClock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = EvalClock()

    def evaluate(state):
        clock.now += 1.0
        return _fake_evaluator([])(state)

    first = run_tune(cfg, None, _space(cfg), evaluate_fn=evaluate, clock=clock)
    assert first.actual_calls == 3
    assert first.stop_reason == "max_time_s"

    wider = replace(cfg, search=replace(cfg.search, max_time_s=4.0))

    class ReplayClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            # 首次调用是运行起点；随后回放/离线阶段时钟跳到 2 秒。
            return 0.0 if self.calls == 1 else 2.0

    new_calls = []
    resumed = run_tune(wider, None, _space(wider),
                       evaluate_fn=_fake_evaluator(new_calls),
                       clock=ReplayClock())
    assert len(new_calls) > 0, "历史活动 3 秒 < 新预算 4 秒，回放 2 秒不应阻止新任务"
    assert resumed.stop_reason != "max_time_s"


def test_responses_log_truncates_half_line_and_rejects_bad_line(tmp_path):
    cfg = _config(tmp_path)
    run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    path = cfg.output.dir / "responses.jsonl"
    with path.open("a") as handle:
        handle.write('{"call_index": 2')
    assert len(read_responses(path)) == 2
    path.write_text(path.read_text() + '{"call_index": 9\n')
    with pytest.raises(ValueError, match="坏行"):
        read_responses(path)


# ---- 评审 round01：续跑必须保持同一生效配置 ----


def test_resume_rejects_changed_simulation_config(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))

    changed = replace(cfg, simulation=replace(
        cfg.simulation, use_cost_sidecar=True))
    calls = []
    with pytest.raises(ValueError, match="续跑"):
        run_tune(changed, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == [], "异配置续跑不能复用旧响应"


def test_strict_cache_does_not_resume_fast_results(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))

    strict = replace(cfg, cache=replace(cfg.cache, mode="strict"))
    calls = []
    with pytest.raises(ValueError, match="续跑"):
        run_tune(strict, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == []


def test_response_records_run_identity_and_sim_key(tmp_path):
    cfg = _config(tmp_path)
    run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    responses = read_responses(cfg.output.dir / "responses.jsonl")
    assert responses
    assert all(response.run_id for response in responses)
    assert all(response.cache_mode == "fast" for response in responses)


def test_pipeline_writes_search_coverage_and_config_digest(
        tmp_path, monkeypatch):
    """汇总与 best 必须留下可独立复算的覆盖计数与配置内容身份。"""
    import sys

    from scripts import tune_pipeline
    from tune.inputs import file_sha256, prepare_inputs
    from tune.loop import LoopResult, SearchCoverage
    from tune.report import candidate_id

    cfg = _config(tmp_path)
    inputs = prepare_inputs(cfg)
    state = _space(cfg).initial_state
    record = RoundRecord(
        sequence=0, candidate_id=candidate_id(state),
        compile_key="c", sim_key="s", cache="fast", status="success",
        result_maturity="model_prediction", state=state, score=1.0,
        validation_evaluations=1,
        paths={"effective_placement": "placement.json"},
        metrics={"total_time_s": 1.0})
    coverage = SearchCoverage(
        algorithm="enumerate", logical_calls=1, actual_calls=1,
        resumed_calls=0, new_pipeline_evaluations=1, compile_cache_hits=0,
        sim_cache_hits=0, new_backend_executions=1, final_mapping_replays=1)
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(tune_pipeline, "prepare_inputs", lambda value: inputs)
    monkeypatch.setattr(
        tune_pipeline, "publish_inputs", lambda config, value: value)
    monkeypatch.setattr(
        tune_pipeline, "run_tune",
        lambda *args, **kwargs: LoopResult(
            records=(record,), best=record, stop_reason="max_evaluations",
            logical_calls=1, actual_calls=1, resumed_calls=0,
            elapsed_s=0.0, coverage=coverage))
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml",
        "--output-dir", str(cfg.output.dir)])
    assert tune_pipeline.main() == 0
    best = json.loads((cfg.output.dir / "best.json").read_text())
    assert best["effective_config_sha256"] == file_sha256(
        cfg.output.dir / "effective_config.yaml")
    assert best["search_coverage"]["new_pipeline_evaluations"] == 1
    assert best["search_coverage"]["real_search"] is True
    summary = (cfg.output.dir / "summary.md").read_text()
    assert "真实搜索覆盖" in summary
    assert "新 pipeline 评估=1" in summary


def test_pipeline_resume_check_does_not_rewrite_effective_config(
        tmp_path, monkeypatch):
    import sys

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    cfg.output.dir.mkdir(parents=True, exist_ok=True)
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    (cfg.output.dir / "responses.jsonl").write_text("")
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "prepare_inputs",
        lambda cfg: pytest.fail("已有日志且未 --resume 时不应继续"))
    monkeypatch.setattr(sys, "argv",
                        ["tune_pipeline.py", "--config", "tune.yaml"])
    assert tune_pipeline.main() == 2
    assert effective.read_text() == "original: evidence\n"


def test_pipeline_dry_run_does_not_overwrite_existing_state(
        tmp_path, monkeypatch):
    """dry-run 只做校验，不能改写已有响应日志旁的生效配置证据。"""
    import sys

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    cfg.output.dir.mkdir(parents=True, exist_ok=True)
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    journal = cfg.output.dir / "responses.jsonl"
    journal.write_text("")
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--dry-run",
        "--output-dir", str(cfg.output.dir)])
    assert tune_pipeline.main() == 0
    assert effective.read_text() == "original: evidence\n"


def test_pipeline_resume_rejects_changed_config_before_writing(
        tmp_path, monkeypatch):
    import sys

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    changed = replace(cfg, simulation=replace(
        cfg.simulation, use_cost_sidecar=True))
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: changed)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "prepare_inputs",
        lambda cfg: pytest.fail("异配置续跑应在输入快照前拒绝"))
    monkeypatch.setattr(sys, "argv",
                        ["tune_pipeline.py", "--config", "tune.yaml", "--resume"])
    assert tune_pipeline.main() == 3
    assert effective.read_text() == "original: evidence\n"


def test_pipeline_resume_rejects_changed_inputs_before_writing(
        tmp_path, monkeypatch):
    import sys

    from scripts import tune_pipeline
    from tune.inputs import snapshot_inputs

    cfg = _config(tmp_path)
    inputs = snapshot_inputs(cfg)
    run_tune(cfg, inputs, _space(cfg), evaluate_fn=_fake_evaluator([]))
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    snapshot_trace = cfg.output.dir / "inputs" / "trace.json"
    from tune.inputs import file_sha256

    original_trace = file_sha256(snapshot_trace)

    trace = tmp_path / "trace.json"
    trace.write_text(trace.read_text().replace(
        '"output_length": 2', '"output_length": 3'))
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--resume"])
    assert tune_pipeline.main() == 3
    assert effective.read_text() == "original: evidence\n"
    assert file_sha256(snapshot_trace) == original_trace


# ---- 评审 round02：续跑语义身份、预算单调扩展与仿真键重算 ----


def _real_key_evaluator(cfg, inputs, calls):
    from tune.steps import compile_key, sim_key

    def evaluate(state):
        calls.append(state)
        key = compile_key(cfg, inputs, state)
        return RoundRecord(
            sequence=0, candidate_id=f"c{state.strategy.num_dpus}",
            compile_key=key, sim_key=sim_key(cfg, inputs, state, key),
            cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=10.0,
            metrics={"total_time_s": 10.0})
    return evaluate


def test_run_identity_covers_source_tools_base_config_and_environment(
        tmp_path, monkeypatch):
    from tune import steps
    from tune.config import run_identity

    cfg = _config(tmp_path)
    baseline = run_identity(cfg, None)

    monkeypatch.setenv("GENESIM_PIM_CLOCK_PERIOD_NS", "999")
    env_changed = run_identity(cfg, None)
    assert env_changed != baseline, "环境覆盖必须让运行身份失效"

    monkeypatch.setattr(
        steps, "source_fingerprint", lambda role="compile": "source-a")
    source_a = run_identity(cfg, None)
    assert source_a != env_changed, "源码内容必须进入运行身份"
    monkeypatch.setattr(
        steps, "source_fingerprint", lambda role="compile": "source-b")
    assert run_identity(cfg, None) != source_a

    monkeypatch.setattr(steps, "genesim_source_fingerprint",
                        lambda: "genesim-a")
    genesim_a = run_identity(cfg, None)
    assert genesim_a != source_a
    monkeypatch.setattr(steps, "genesim_source_fingerprint",
                        lambda: "genesim-b")
    assert run_identity(cfg, None) != genesim_a

    monkeypatch.setattr(steps, "refine_level", lambda: "pimir")
    refine_a = run_identity(cfg, None)
    monkeypatch.setattr(steps, "refine_level", lambda: "oplevel")
    assert run_identity(cfg, None) != refine_a, "编译档位必须进入运行身份"


def test_resume_rejects_shrunk_evaluation_budget(tmp_path):
    cfg = _config(tmp_path)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))

    shrunk = replace(cfg, search=replace(cfg.search, max_evaluations=1))
    calls = []
    with pytest.raises(ValueError, match="预算"):
        run_tune(shrunk, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == [], "预算收缩不能继续回放或发起新评估"


def test_resume_rejects_changed_environment(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))

    monkeypatch.setenv("GENESIM_PIM_CLOCK_PERIOD_NS", "999")
    calls = []
    with pytest.raises(ValueError, match="续跑"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == []


def test_resume_recomputes_compile_and_sim_keys(tmp_path):
    from tune.inputs import snapshot_inputs

    cfg = _config(tmp_path)
    inputs = snapshot_inputs(cfg)
    space = _space(cfg)
    calls = []
    first = run_tune(cfg, inputs, space,
                     evaluate_fn=_real_key_evaluator(cfg, inputs, calls))
    assert first.actual_calls == 2

    again = []
    resumed = run_tune(cfg, inputs, space,
                       evaluate_fn=_real_key_evaluator(cfg, inputs, again))
    assert again == []
    assert resumed.resumed_calls == 2

    path = cfg.output.dir / "responses.jsonl"
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    record["sim_key"] = "0" * 64
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="仿真键"):
        run_tune(cfg, inputs, space,
                 evaluate_fn=_real_key_evaluator(cfg, inputs, []))


# ---- 评审 round08：外部语义文件内容必须进入运行身份 ----


def _profile_copy(tmp_path):
    path = tmp_path / "profile.json"
    path.write_bytes(PROFILE.read_bytes())
    return path


def test_resume_rejects_profile_changed_in_place_before_writing(
        tmp_path, monkeypatch):
    """同一路径的画像内容变化必须在写输入与生效配置之前拒绝。"""
    import sys

    from scripts import tune_pipeline
    from tune.inputs import file_sha256, snapshot_inputs

    cfg = _config(tmp_path)
    profile_path = _profile_copy(tmp_path)
    cfg = replace(cfg, profile_path=profile_path,
                  profile=load_profile(profile_path))
    inputs = snapshot_inputs(cfg)
    run_tune(cfg, inputs, _space(cfg), evaluate_fn=_fake_evaluator([]))
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    snapshot = cfg.output.dir / "inputs" / "trace.json"
    original_snapshot = file_sha256(snapshot)

    payload = json.loads(profile_path.read_text())
    payload["wram_bytes_per_dpu"] *= 2
    profile_path.write_text(json.dumps(payload))
    changed = replace(cfg, profile=load_profile(profile_path))
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: changed)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "prepare_inputs",
        lambda cfg: pytest.fail("画像内容变化必须在输入快照前拒绝"))
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--resume"])
    assert tune_pipeline.main() == 3
    assert effective.read_text() == "original: evidence\n"
    assert file_sha256(snapshot) == original_snapshot


def test_resume_rejects_report_changed_in_place_before_writing(
        tmp_path, monkeypatch):
    """同一路径的校准报告内容变化也必须在写证据之前拒绝。"""
    import sys

    from scripts import tune_pipeline
    from tune.inputs import file_sha256, snapshot_inputs

    report_path = tmp_path / "report.json"
    report_path.write_text('{"status": "passed"}')
    cfg = _config(tmp_path)
    cfg = replace(cfg, calibration=replace(
        cfg.calibration, validation_report=report_path))
    inputs = snapshot_inputs(cfg)
    run_tune(cfg, inputs, _space(cfg), evaluate_fn=_fake_evaluator([]))
    effective = cfg.output.dir / "effective_config.yaml"
    effective.write_text("original: evidence\n")
    snapshot = cfg.output.dir / "inputs" / "trace.json"
    original_snapshot = file_sha256(snapshot)

    report_path.write_text('{"status": "report_only"}')
    monkeypatch.setattr(
        tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "prepare_inputs",
        lambda cfg: pytest.fail("报告内容变化必须在输入快照前拒绝"))
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--resume"])
    assert tune_pipeline.main() == 3
    assert effective.read_text() == "original: evidence\n"
    assert file_sha256(snapshot) == original_snapshot


# ---- 评审 round01：--replay-rounds 路由与最终放置 ----


def test_replay_rounds_accepts_config_and_uses_saved_placement(
        tmp_path, monkeypatch):
    from scripts import tune_pipeline
    from tune.config import write_effective_config

    cfg = _config(tmp_path)
    write_effective_config(cfg)
    record = _replay_round_files(tmp_path, cfg)
    _patch_replay_keys(monkeypatch, tune_pipeline)

    replay_cfg = tune_pipeline._replay_config(
        type("Args", (), {"config": None,
                          "output_dir": str(cfg.output.dir),
                          "cache_mode": None, "profile": None,
                          "max_evaluations": None, "max_time_s": None,
                          "algorithm": None})())
    assert replay_cfg.output.dir == cfg.output.dir

    _replay_env(monkeypatch, tune_pipeline)
    seen = {}

    def fake_evaluate(cfg, inputs, state, *, force_compile=False,
                      replay_placement=None, **kwargs):
        seen["placement"] = replay_placement
        return RoundRecord(
            sequence=0, candidate_id="cand", compile_key="compile-key",
            sim_key="sim-key", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=10.0,
            metrics={"total_time_s": 10.0})

    monkeypatch.setattr(tune_pipeline, "evaluate_candidate", fake_evaluate)
    assert tune_pipeline._replay_rounds(replay_cfg) == 0
    assert seen["placement"] == Path(record.paths["effective_placement"])


def test_replay_rounds_rejects_missing_saved_placement(tmp_path, monkeypatch):
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, placement=False)
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *args, **kwargs: pytest.fail("缺最终放置时不应重新搜索"))
    assert tune_pipeline._replay_rounds(cfg) == 1


def test_parse_args_allows_replay_rounds_with_config(monkeypatch):
    import sys

    from scripts import tune_pipeline

    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--replay-rounds"])
    args = tune_pipeline.parse_args()
    assert args.replay_rounds is True
    assert args.config == "tune.yaml"


def test_parse_args_rejects_replay_without_config_or_output(monkeypatch):
    import sys

    from scripts import tune_pipeline

    monkeypatch.setattr(sys, "argv",
                        ["tune_pipeline.py", "--replay-rounds"])
    with pytest.raises(SystemExit):
        tune_pipeline.parse_args()


def test_pipeline_dry_run_writes_preview_in_empty_output_dir(
        tmp_path, monkeypatch):
    """空输出目录里 dry-run 仍落一份生效配置供预览。"""
    import sys

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    monkeypatch.setattr(tune_pipeline, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(sys, "argv", [
        "tune_pipeline.py", "--config", "tune.yaml", "--dry-run",
        "--output-dir", str(cfg.output.dir)])
    assert tune_pipeline.main() == 0
    assert (cfg.output.dir / "effective_config.yaml").is_file()


# ---- 评审 round04：严格复放的输入与内容键身份 ----

def _complete_compile_dir(root: Path, compile_key: str, inputs):
    """写一份满足完成判据的编译目录：设计必需角色 + sidecar 覆盖的 PIM MLIR。"""
    import hashlib

    from tune.report import manifest_from_dir
    from tune.steps import manifest_inputs

    root.mkdir(parents=True, exist_ok=True)
    (root / "partition_plan.json").write_text("{}")
    (root / "placed.ir").write_text("{}")
    # 精化 IR 记录派生它的模型与原始 IR 身份，单独迁移产物时无法自证。
    (root / "refined.ir").write_text(json.dumps({
        "operators": [],
        "source_model_sha256": inputs.model_sha256,
        "source_ir_sha256": inputs.ir_sha256,
    }))
    (root / "cost.json").write_text("{}")
    body = b"module { }"
    digest = hashlib.sha256(body).hexdigest()
    pimir = root / "pimir" / f"{digest}.mlir"
    pimir.parent.mkdir(exist_ok=True)
    pimir.write_bytes(body)
    (root / "placement.json").write_text(json.dumps({
        "operators": {"0": {
            "op_type": "GEMM", "pimir_path": str(pimir),
            "pimir_sha256": digest,
        }},
    }))
    return manifest_from_dir(root, compile_key, manifest_inputs(inputs))


def _best_payload(cfg, tmp_path, *, overrides=None):
    from tune.inputs import snapshot_inputs
    from tune.report import manifest_from_dir, state_to_json
    from tune.steps import compile_key, sim_key

    inputs = snapshot_inputs(cfg)
    state = _space(cfg).initial_state
    key = compile_key(cfg, inputs, state)
    compile_dir = tmp_path / "compile"
    manifest = _complete_compile_dir(compile_dir, key, inputs)
    placement = tmp_path / "effective_placement.json"
    placement.write_text(json.dumps({
        "schema_version": 1, "targets_include_relay": False,
        "inputs_sha256": "0" * 64,
        "execution_graph_sha256": "1" * 64,
        "final_graph_sha256": "2" * 64,
        "targets": [{"op_id": 0, "node_id": 0, "vpu_id": 1,
                     "attached_pu_id": 0}]}))
    payload = {
        "sequence": 0,
        "candidate_id": "cand",
        "status": "success",
        "effective_config": str(tmp_path / "tune.yaml"),
        "state": state_to_json(state),
        "score": 10.0,
        "compile_key": key,
        "sim_key": sim_key(cfg, inputs, state, key),
        "inputs": inputs.to_json(),
        "artifacts": manifest.to_json(),
        "paths": {
            "compile_dir": str(compile_dir),
            "effective_placement": str(placement),
        },
    }
    payload.update(overrides or {})
    best_path = tmp_path / "best.json"
    best_path.write_text(json.dumps(payload))
    return best_path, payload


def test_strict_replay_rejects_mismatched_best_inputs_before_any_write(
        tmp_path, monkeypatch):
    """best.json 的输入身份不符时必须在编译前拒绝且不改写证据。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline
    from tune.inputs import file_sha256

    cfg = _config(tmp_path)
    best_path, payload = _best_payload(cfg, tmp_path)
    snapshot_trace = cfg.output.dir / "inputs" / "trace.json"
    original = file_sha256(snapshot_trace)
    payload["inputs"] = dict(payload["inputs"], trace_sha256="0" * 64)
    best_path.write_text(json.dumps(payload))
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *args, **kwargs: pytest.fail("身份不符时不能开始编译"))
    assert tune_pipeline._evaluate_candidate(
        best_path, SimpleNamespace(cache_mode="strict")) == 3
    assert file_sha256(snapshot_trace) == original


def test_strict_replay_rejects_a_rewritten_published_trace(tmp_path, monkeypatch):
    """已发布轨迹在 prepare 后被原地改写时，严格复跑在编译前拒绝。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    best_path, _ = _best_payload(cfg, tmp_path)
    snapshot_trace = cfg.output.dir / "inputs" / "trace.json"
    payload = json.loads(snapshot_trace.read_text())
    payload["requests"][0]["output_length"] = 5
    snapshot_trace.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *args, **kwargs: pytest.fail("输入快照已变时不能开始编译"))
    assert tune_pipeline._evaluate_candidate(
        best_path, SimpleNamespace(cache_mode="strict")) == 3


def test_strict_replay_rejects_mismatched_compile_and_sim_keys(
        tmp_path, monkeypatch):
    """best.json 的编译键或仿真键与重新生成的不一致时拒绝。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *args, **kwargs: pytest.fail("内容键不符时不能开始编译"))
    for field in ("compile_key", "sim_key"):
        best_path, payload = _best_payload(cfg, tmp_path)
        payload[field] = "0" * 64
        best_path.write_text(json.dumps(payload))
        assert tune_pipeline._evaluate_candidate(
            best_path, SimpleNamespace(cache_mode="strict")) == 3


def test_strict_replay_accepts_matching_identity_and_publishes_inputs(
        tmp_path, monkeypatch):
    """身份完全一致时才进入严格复跑，且发布的快照就是原输入。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline
    from tune.inputs import file_sha256

    cfg = _config(tmp_path)
    best_path, payload = _best_payload(cfg, tmp_path)
    snapshot_trace = cfg.output.dir / "inputs" / "trace.json"
    original = file_sha256(snapshot_trace)
    seen = {}

    def fake_evaluate(config, inputs, state, **kwargs):
        seen["inputs"] = inputs
        seen["force"] = kwargs.get("force_compile")
        return RoundRecord(
            sequence=0, candidate_id="replay", compile_key=payload["compile_key"],
            sim_key=payload["sim_key"], cache="strict", status="success",
            result_maturity="model_prediction", state=state, score=10.0,
            metrics={"total_time_s": 10.0})

    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(tune_pipeline, "evaluate_candidate", fake_evaluate)
    assert tune_pipeline._evaluate_candidate(
        best_path, SimpleNamespace(cache_mode="strict")) == 0
    assert seen["force"] is True
    assert seen["inputs"].ir_sha256 == payload["inputs"]["ir_sha256"]
    assert file_sha256(snapshot_trace) == original


# ---- 评审 round04：响应提交点与轮次对账 ----

def _crash_between_round_and_response(tmp_path, monkeypatch):
    from tune import loop

    cfg = _config(tmp_path, max_evaluations=1)
    original = loop.append_response

    def crash_before_response(*args, **kwargs):
        raise RuntimeError("injected crash before response commit")

    monkeypatch.setattr(loop, "append_response", crash_before_response)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    finally:
        monkeypatch.setattr(loop, "append_response", original)
    return cfg


def test_crash_before_response_isolates_orphan_round(tmp_path, monkeypatch):
    """完整但未提交的轮次不能参与排序，恢复后同一候选只出现一次。"""
    from tune import loop

    cfg = _crash_between_round_and_response(tmp_path, monkeypatch)
    assert len(loop.read_rounds(cfg.output.dir / "rounds.jsonl")) == 1
    assert len(loop.read_responses(cfg.output.dir / "responses.jsonl")) == 0
    resumed = run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    assert resumed.actual_calls == 1
    assert len(resumed.records) == 1
    assert [record.sequence for record in resumed.records] == [0]
    assert len(loop.read_rounds(cfg.output.dir / "rounds.jsonl")) == 1
    orphans = (cfg.output.dir / "orphan_rounds.jsonl").read_text().splitlines()
    assert len(orphans) == 1
    assert "响应未提交" in orphans[0]


def test_crash_before_round_write_reruns_without_orphan(tmp_path, monkeypatch):
    """评估后、轮次写入前崩溃：没有残留记录，恢复只重跑一次。"""
    from tune import loop

    cfg = _config(tmp_path, max_evaluations=1)
    original = loop.append_round

    def crash_before_round(*args, **kwargs):
        raise RuntimeError("injected crash before round commit")

    monkeypatch.setattr(loop, "append_round", crash_before_round)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    finally:
        monkeypatch.setattr(loop, "append_round", original)
    assert not (cfg.output.dir / "rounds.jsonl").exists()
    resumed = run_tune(cfg, None, _space(cfg), evaluate_fn=_fake_evaluator([]))
    assert resumed.actual_calls == 1
    assert len(resumed.records) == 1
    assert not (cfg.output.dir / "orphan_rounds.jsonl").exists()


def test_failed_candidate_orphan_is_replayed_without_duplicate(tmp_path,
                                                               monkeypatch):
    """失败候选的轮次同样以响应为提交点，恢复后不重复计数。"""
    from tune import loop

    cfg = _config(tmp_path, max_evaluations=1)
    original = loop.append_response

    def crash_before_response(*args, **kwargs):
        raise RuntimeError("injected crash before response commit")

    monkeypatch.setattr(loop, "append_response", crash_before_response)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            run_tune(cfg, None, _space(cfg),
                     evaluate_fn=_fake_evaluator([], fail_at=1))
    finally:
        monkeypatch.setattr(loop, "append_response", original)
    resumed = run_tune(cfg, None, _space(cfg),
                       evaluate_fn=_fake_evaluator([], fail_at=1))
    assert resumed.actual_calls == 1
    assert len(resumed.records) == 1
    assert resumed.records[0].status == "error"
    assert resumed.best is None


def test_fast_cache_hit_orphan_is_reconciled_once(tmp_path, monkeypatch):
    """快速缓存命中的记录同样以响应为提交点，崩溃后只留一条轮次。"""
    from tune import loop

    cfg = _config(tmp_path, max_evaluations=1)
    original = loop.append_response

    def crash_before_response(*args, **kwargs):
        raise RuntimeError("injected crash before response commit")

    def cache_hit(state):
        return RoundRecord(
            sequence=0, candidate_id="cached", compile_key="c", sim_key="s",
            cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=10.0,
            sim_cache_hit=True, metrics={"total_time_s": 10.0})

    monkeypatch.setattr(loop, "append_response", crash_before_response)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            run_tune(cfg, None, _space(cfg), evaluate_fn=cache_hit)
    finally:
        monkeypatch.setattr(loop, "append_response", original)
    resumed = run_tune(cfg, None, _space(cfg), evaluate_fn=cache_hit)
    assert resumed.actual_calls == 1
    assert len(resumed.records) == 1
    assert resumed.records[0].sim_cache_hit is True


# ---- 评审 round05：轮次重放必须逐轮核对内容键与产物证据 ----

def _replay_round_files(tmp_path, cfg, *, candidate_id="cand",
                        compile_key="compile-key", sim_key="sim-key",
                        with_artifacts=True, placement=True,
                        tamper_compile_dir=False):
    from tune.report import append_round, manifest_from_dir
    from tune.report import state_to_json
    from tune.report import RoundRecord

    state = _space(cfg).initial_state
    paths = {}
    from tune.inputs import snapshot_inputs
    inputs = snapshot_inputs(cfg)
    if with_artifacts:
        compile_dir = tmp_path / f"compile-{candidate_id}"
        artifacts = _complete_compile_dir(compile_dir, compile_key, inputs)
        if tamper_compile_dir:
            (compile_dir / "refined.ir").write_text("tampered")
        paths["compile_dir"] = str(compile_dir)
    else:
        artifacts = None
    if placement:
        path = tmp_path / "effective_placement.json"
        path.write_text(json.dumps({
            "schema_version": 1, "targets_include_relay": False,
            "inputs_sha256": "0" * 64,
            "execution_graph_sha256": "1" * 64,
            "final_graph_sha256": "2" * 64,
            "targets": [{"op_id": 0, "node_id": 0, "vpu_id": 1,
                         "attached_pu_id": 0}]}))
        paths["effective_placement"] = str(path)
    record = RoundRecord(
        sequence=0, candidate_id=candidate_id, compile_key=compile_key,
        sim_key=sim_key, cache="fast", status="success",
        result_maturity="model_prediction", state=state, score=10.0,
        artifacts=artifacts, paths=paths)
    append_round(cfg.output.dir / "rounds.jsonl", record)
    return record


def _patch_replay_keys(monkeypatch, tune_pipeline, *,
                       compile_key="compile-key", sim_key="sim-key"):
    monkeypatch.setattr(tune_pipeline, "compile_key", lambda *a: compile_key)
    monkeypatch.setattr(
        tune_pipeline, "sim_key",
        lambda cfg, inputs, state, compile_key_value, placement=None: sim_key)


def _replay_env(monkeypatch, tune_pipeline):
    from tune.inputs import snapshot_inputs

    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    # 重放前置校验与正式候选共用同一份真实输入身份，不能用占位对象绕过。
    monkeypatch.setattr(tune_pipeline, "prepare_inputs", snapshot_inputs)
    monkeypatch.setattr(
        tune_pipeline, "publish_inputs", lambda cfg, inputs: inputs)


def test_replay_rounds_rejects_stale_compile_key_before_writes(
        tmp_path, monkeypatch):
    """compile_key 不符时在任何发布或评估前拒绝。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, compile_key="stale-compile")
    _patch_replay_keys(monkeypatch, tune_pipeline, compile_key="fresh-compile")
    _replay_env(monkeypatch, tune_pipeline)
    published = []
    monkeypatch.setattr(
        tune_pipeline, "publish_inputs",
        lambda cfg, inputs: published.append(True) or inputs)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *a, **k: pytest.fail("内容键不符时不能开始编译"))
    assert tune_pipeline._replay_rounds(cfg) == 3
    assert published == []


def test_replay_rounds_rejects_stale_sim_key_with_same_score(
        tmp_path, monkeypatch):
    """分数相同但仿真键是旧配置时不能报告为复现一致。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, sim_key="stale-sim")
    _patch_replay_keys(monkeypatch, tune_pipeline, sim_key="fresh-sim")
    _replay_env(monkeypatch, tune_pipeline)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *a, **k: pytest.fail("仿真键不符时不能开始编译"))
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_replay_rounds_rejects_one_stale_key_in_a_multi_round_log(
        tmp_path, monkeypatch):
    """多轮日志里只要有一轮内容键不符，整次重放在发布前拒绝。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, candidate_id="c0")
    _replay_round_files(tmp_path, cfg, candidate_id="c1", sim_key="stale-sim")
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    published = []
    monkeypatch.setattr(
        tune_pipeline, "publish_inputs",
        lambda cfg, inputs: published.append(True) or inputs)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *a, **k: pytest.fail("存在 stale 键时不能开始重放"))
    assert tune_pipeline._replay_rounds(cfg) == 3
    assert published == []


def test_replay_rounds_rejects_missing_artifact_manifest(tmp_path, monkeypatch):
    """成功轮次缺少产物清单时不能重放。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, with_artifacts=False)
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_replay_rounds_rejects_foreign_manifest_input_identity(
        tmp_path, monkeypatch):
    """轮次重放与正式候选共用产物清单输入身份判据，错误迁移不能重放。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    record = _replay_round_files(tmp_path, cfg)
    manifest = record.artifacts.to_json()
    for entry in manifest["inputs"]:
        if entry["role"] == "ir":
            entry["sha256"] = "f" * 64
    record = replace(record, artifacts=ArtifactManifest.from_json(manifest))
    (cfg.output.dir / "rounds.jsonl").write_text(
        json.dumps(record.to_json()) + "\n")
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_replay_rounds_rejects_a_rewritten_published_trace(
        tmp_path, monkeypatch):
    """已发布轨迹被原地改写时，轮次重放在发布与评估前拒绝。"""
    from scripts import tune_pipeline
    from tune.inputs import prepare_inputs as read_only_prepare

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg)
    snapshot_trace = cfg.output.dir / "inputs" / "trace.json"
    payload = json.loads(snapshot_trace.read_text())
    payload["requests"][0]["output_length"] = 5
    snapshot_trace.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    _patch_replay_keys(monkeypatch, tune_pipeline)
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    monkeypatch.setattr(tune_pipeline, "prepare_inputs", read_only_prepare)
    monkeypatch.setattr(
        tune_pipeline, "publish_inputs", lambda cfg, inputs: inputs)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *args, **kwargs: pytest.fail("输入快照已变时不能开始重放"))
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_strict_replay_rejects_foreign_manifest_input_identity(
        tmp_path, monkeypatch):
    """严格复跑同样拒绝输入身份被错误迁移的编译清单。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    best_path, payload = _best_payload(cfg, tmp_path)
    for entry in payload["artifacts"]["inputs"]:
        if entry["role"] == "model":
            entry["sha256"] = "f" * 64
    best_path.write_text(json.dumps(payload))
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    assert tune_pipeline._evaluate_candidate(
        best_path, SimpleNamespace(cache_mode="strict")) == 3


def test_replay_rounds_rejects_tampered_compile_products(tmp_path, monkeypatch):
    """产物清单与磁盘内容不一致时不能重放。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    _replay_round_files(tmp_path, cfg, tamper_compile_dir=True)
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    assert tune_pipeline._replay_rounds(cfg) == 3


def _drop_manifest_role(manifest, role):
    """从产物清单移除一个角色并删除对应文件。"""
    kept = [entry for entry in manifest["files"] if entry["role"] != role]
    dropped = next(
        entry for entry in manifest["files"] if entry["role"] == role)
    manifest["files"] = kept
    return dropped["path"]


def test_replay_rounds_rejects_missing_placed_ir(tmp_path, monkeypatch):
    """编译清单与目录同时缺 placed.ir 时轮次重放必须在评估前拒绝。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    record = _replay_round_files(tmp_path, cfg)
    manifest = record.artifacts.to_json()
    relative = _drop_manifest_role(manifest, "placed.ir")
    (Path(record.paths["compile_dir"]) / relative).unlink()
    record = replace(record, artifacts=ArtifactManifest.from_json(manifest))
    rounds_path = cfg.output.dir / "rounds.jsonl"
    rounds_path.write_text(json.dumps(record.to_json()) + "\n")
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_strict_replay_rejects_missing_placed_ir(tmp_path, monkeypatch):
    """单候选严格复跑同样拒绝缺 placed.ir 的编译清单。"""
    from types import SimpleNamespace

    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    best_path, payload = _best_payload(cfg, tmp_path)
    relative = _drop_manifest_role(payload["artifacts"], "placed.ir")
    (tmp_path / "compile" / relative).unlink()
    best_path.write_text(json.dumps(payload))
    monkeypatch.setattr(tune_pipeline, "_prepare_environment", lambda: None)
    assert tune_pipeline._evaluate_candidate(
        best_path, SimpleNamespace(cache_mode="strict")) == 3


def test_replay_rounds_rejects_invalid_final_placement(tmp_path, monkeypatch):
    """最终放置缺版本或含 relay 目标时拒绝重放。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    record = _replay_round_files(tmp_path, cfg)
    placement = json.loads(Path(record.paths["effective_placement"]).read_text())
    placement["targets_include_relay"] = True
    Path(record.paths["effective_placement"]).write_text(json.dumps(placement))
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    assert tune_pipeline._replay_rounds(cfg) == 3


def test_replay_rounds_accepts_matching_keys_and_passes_saved_placement(
        tmp_path, monkeypatch):
    """合法同键时重放，并把保存的最终映射交给评估器。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    record = _replay_round_files(tmp_path, cfg)
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    seen = {}

    def fake_evaluate(cfg, inputs, state, *, force_compile=False,
                      replay_placement=None, **kwargs):
        seen["placement"] = replay_placement
        return RoundRecord(
            sequence=0, candidate_id="cand", compile_key="compile-key",
            sim_key="sim-key", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=10.0,
            metrics={"total_time_s": 10.0})

    monkeypatch.setattr(tune_pipeline, "evaluate_candidate", fake_evaluate)
    assert tune_pipeline._replay_rounds(cfg) == 0
    assert seen["placement"] == Path(record.paths["effective_placement"])


# ---- 评审 round06：响应是唯一提交点，轮次必须按内容对账 ----

def _tamper_response(path: Path, index: int, mutate) -> None:
    lines = path.read_text().splitlines()
    record = json.loads(lines[index])
    mutate(record)
    lines[index] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")


def test_resume_rejects_response_error_round_success_conflict(tmp_path):
    """响应记为失败、轮次记为成功时必须拒绝恢复，不能选失败候选为最优。"""
    cfg = _config(tmp_path, max_evaluations=2)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))
    _tamper_response(
        cfg.output.dir / "responses.jsonl", 0,
        lambda record: record.update(
            {"status": "error", "score": None, "score_special": None,
             "failure": "injected response failure"}))
    calls = []
    with pytest.raises(ValueError, match="对账|状态|分数"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == [], "冲突必须在任何评估之前拒绝"


def test_resume_rejects_response_round_score_conflict(tmp_path):
    """同序号响应与轮次分数不同时拒绝恢复。"""
    cfg = _config(tmp_path, max_evaluations=2)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))
    _tamper_response(
        cfg.output.dir / "responses.jsonl", 0,
        lambda record: record.update({"score": 0.001}))
    calls = []
    with pytest.raises(ValueError, match="对账|分数"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == []


def test_resume_rejects_response_round_key_conflict(tmp_path):
    """同序号响应的编译/仿真键与轮次不同时拒绝恢复。"""
    cfg = _config(tmp_path, max_evaluations=2)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))
    _tamper_response(
        cfg.output.dir / "responses.jsonl", 1,
        lambda record: record.update(
            {"compile_key": "stale-compile", "sim_key": "stale-sim"}))
    calls = []
    with pytest.raises(ValueError, match="对账|键"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == []


def test_resume_rejects_response_without_matching_round(tmp_path):
    """响应提交了但轮次缺失或序号不连续时，日志已损坏，必须拒绝恢复。"""
    cfg = _config(tmp_path, max_evaluations=2)
    space = _space(cfg)
    run_tune(cfg, None, space, evaluate_fn=_fake_evaluator([]))
    responses = cfg.output.dir / "responses.jsonl"
    responses.write_text("\n".join(
        responses.read_text().splitlines()[1:]) + "\n")
    calls = []
    with pytest.raises(ValueError, match="对账|连续|缺少|恢复顺序"):
        run_tune(cfg, None, space, evaluate_fn=_fake_evaluator(calls))
    assert calls == []


# ---- 评审 round06：缺失编译目录不能通过重放前置 ----

def test_replay_rounds_rejects_missing_compile_dir(tmp_path, monkeypatch):
    """成功轮次的编译目录被删除后，重放必须前置失败。"""
    from scripts import tune_pipeline

    cfg = _config(tmp_path)
    record = _replay_round_files(tmp_path, cfg)
    shutil.rmtree(record.paths["compile_dir"])
    _patch_replay_keys(monkeypatch, tune_pipeline)
    _replay_env(monkeypatch, tune_pipeline)
    monkeypatch.setattr(
        tune_pipeline, "evaluate_candidate",
        lambda *a, **k: pytest.fail("缺编译目录时不能开始重放"))
    assert tune_pipeline._replay_rounds(cfg) == 3
