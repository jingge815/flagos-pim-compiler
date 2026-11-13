"""硬件扫描：核策略、重编判断、拐点与 Pareto 前沿。"""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from contracts.hardware_profile import load_profile
from tune.candidate import OpKnobs, StrategyChoice, TuneState, model_dims
from tune.config import load_config
from tune.report import RoundRecord
from tune.sweep import run_sweep, write_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"
GENERIC = REPO_ROOT / "contracts" / "profiles" / "generic.json"
GIB = 2**30


def _config(tmp_path, *, fields, core_policy="fixed_dpus", profile=PROFILE):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "num_attention_heads": 64, "num_key_value_heads": 64,
        "intermediate_size": 128, "vocab_size": 64,
        "num_hidden_layers": 4, "hidden_size": 128,
    }))
    (tmp_path / "trace.json").write_text(json.dumps({"requests": [{
        "request_id": "r0", "input_tokens": [1, 2, 3, 4],
        "output_length": 2, "arrival_time_ns": 0, "is_prefill": True,
        "seq_len": 4, "prompt": None, "padded_query_length": None,
        "initial_kv_length": 0}]}))
    raw = {
        "version": 1, "run_name": "sweep",
        "model": {"model_dir": str(model_dir), "ir": str(tmp_path / "m.ir"),
                  "prefill_seq_len": 4, "max_seq": 8,
                  "trace": {"path": str(tmp_path / "trace.json"),
                            "num_requests": 1, "synthetic": None}},
        "profile": {"path": str(profile)},
        "space": {"strategy": {"enabled": True, "num_dpus": [8],
                               "num_stages": [4]},
                  "op": {"enabled": True, "num_tasklets": [4]},
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
        "search": {"algorithm": "enumerate", "seed": 0, "max_evaluations": 1,
                   "max_time_s": 0, "max_proposals": None,
                   "layer_weights": {"strategy": 1, "op": 0, "hardware": 0},
                   "algorithm_params": {}},
        "cache": {"mode": "fast"},
        "simulation": {"use_cost_sidecar": False},
        "calibration": {"validation_report": None},
        "sweep": {"fields": fields, "core_policy": core_policy,
                  "knee_improvement_threshold": 0.05,
                  "pareto": {"resource": "total_physical_memory_bytes",
                             "direction": "minimize"}},
        "output": {"dir": str(tmp_path / "out")},
    }
    (tmp_path / "m.ir").write_text('{"operators": []}')
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def _best(cfg):
    return TuneState(StrategyChoice(8, 4), OpKnobs(4), cfg.profile)


def _evaluator(clock_scores=None, capacity_scores=None):
    def evaluate(state):
        if clock_scores and state.profile.clock_period_ns in clock_scores:
            score = clock_scores[state.profile.clock_period_ns]
        elif capacity_scores:
            score = capacity_scores[state.profile.cluster_memory_bytes]
        else:
            score = 1.0
        return RoundRecord(
            sequence=0,
            candidate_id=f"c{state.profile.clock_period_ns}",
            compile_key="c", sim_key="s", cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=score,
            metrics={"total_time_s": score})

    return evaluate


def test_clock_only_changes_simulation_fields(tmp_path):
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63, 0.5]})
    result = run_sweep(
        cfg, None, _best(cfg),
        evaluate_fn=_evaluator(clock_scores={0.63: 1.0, 0.5: 0.9}))
    assert len(result.points) == 2
    assert all(not point.compile_fields_changed for point in result.points)
    assert {point.state.profile.clock_period_ns for point in result.points} == {0.63, 0.5}
    # 扫描新值没有适用证据，来源降为 assumption。
    assert all(point.state.profile.parameter_sources["pim.clock_period_ns"]["origin"]
               == "assumption" for point in result.points)
    continuous = result.sensitivity["clock_period_ns"]
    assert continuous["kind"] == "continuous"
    # 值按升序排列：0.5 → 0.63 的时延变化为 +0.1。
    assert continuous["deltas"][0]["delta_time"] == pytest.approx(0.1)
    baseline_score = 1.0  # 0.63 是基准画像
    elasticity = continuous["elasticity"][0]["elasticity"]
    expected = ((0.1) / baseline_score) / ((0.63 - 0.5) / 0.5)
    assert elasticity == pytest.approx(expected)


def test_capacity_change_requires_recompile_and_finds_knee(tmp_path):
    values = [GIB, 2 * GIB, 4 * GIB, 8 * GIB]
    scores = {GIB: 10.0, 2 * GIB: 9.5, 4 * GIB: 9.0, 8 * GIB: 8.9}
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": values})
    result = run_sweep(cfg, None, _best(cfg),
                       evaluate_fn=_evaluator(capacity_scores=scores))
    assert all(point.compile_fields_changed for point in result.points)
    assert result.knees["cluster_memory_bytes"]["value"] == 8 * GIB
    assert result.knees["cluster_memory_bytes"]["improvement"] == pytest.approx(
        (9.0 - 8.9) / 9.0)


def test_knee_compares_like_for_like_across_cartesian_points(tmp_path):
    """不同时钟点的分数不能当成容量相邻点，否则会出现负改善。"""
    values = [GIB, 2 * GIB]
    cfg = _config(tmp_path, fields={
        "cluster_memory_bytes": values,
        "clock_period_ns": [0.63, 0.5],
    })
    scores = {
        (GIB, 0.63): 10.0, (GIB, 0.5): 8.2,
        (2 * GIB, 0.63): 10.0, (2 * GIB, 0.5): 8.2,
    }

    def evaluator(state):
        score = scores[(state.profile.cluster_memory_bytes,
                        state.profile.clock_period_ns)]
        return RoundRecord(
            sequence=0, candidate_id="c", compile_key="c", sim_key="s",
            cache="fast", status="success", result_maturity="model_prediction",
            state=state, score=score, metrics={"total_time_s": score})

    result = run_sweep(cfg, None, _best(cfg), evaluate_fn=evaluator)
    # 基准画像不在扫描点里时用第一个成功点作参考，弹性仍可计算。
    continuous = result.sensitivity["clock_period_ns"]
    assert continuous["baseline_source"] == "first_point"
    assert continuous["deltas"][0]["delta_over_base"] is not None
    knee = result.knees["cluster_memory_bytes"]
    assert knee["value"] == 2 * GIB
    assert knee["improvement"] == pytest.approx(0.0)


def test_use_all_clusters_sets_dpu_count_and_validates_tp(tmp_path):
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]},
                  core_policy="use_all_clusters")
    result = run_sweep(cfg, None, _best(cfg), evaluate_fn=_evaluator())
    assert result.points[0].state.strategy.num_dpus == 64


def test_invalid_point_keeps_reason(tmp_path):
    cfg = _config(tmp_path, fields={"num_pim_devices": [1, 2]})
    result = run_sweep(cfg, None, _best(cfg), evaluate_fn=_evaluator())
    invalid = [point for point in result.points if point.status == "invalid"]
    assert len(invalid) == 1
    assert invalid[0].values == {"num_pim_devices": 2}
    assert "CEVA" in invalid[0].reason


def test_pareto_front_matches_hand_computation(tmp_path):
    values = [GIB, 2 * GIB, 4 * GIB]
    scores = {GIB: 1.0, 2 * GIB: 0.9, 4 * GIB: 0.95}
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": values})
    result = run_sweep(cfg, None, _best(cfg),
                       evaluate_fn=_evaluator(capacity_scores=scores))
    front = {point.state.profile.cluster_memory_bytes for point in result.pareto}
    # 1 GiB 时间更差但资源更少，仍在前沿；4 GiB 时间与资源都不占优，被 2 GiB 支配。
    assert front == {GIB, 2 * GIB}


def test_outputs_are_recomputable(tmp_path):
    values = [GIB, 2 * GIB, 4 * GIB, 8 * GIB]
    scores = {GIB: 10.0, 2 * GIB: 9.5, 4 * GIB: 9.0, 8 * GIB: 8.9}
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": values})
    result = run_sweep(cfg, None, _best(cfg),
                       evaluate_fn=_evaluator(capacity_scores=scores))
    paths = write_sweep(result, cfg, cfg.output.dir / "sweep")
    # 原始 JSONL 独立复算：拐点仍是 (Tprev-Tnext)/Tprev 最小改善处。
    rows = [json.loads(line) for line in
            Path(paths["sweep"]).read_text().splitlines()]
    ordered = sorted(rows, key=lambda row: row["values"]["cluster_memory_bytes"])
    improvements = [
        (previous["score"] - current["score"]) / previous["score"]
        for previous, current in zip(ordered, ordered[1:])
    ]
    assert min(improvements) == pytest.approx((9.0 - 8.9) / 9.0)
    assert Path(paths["pareto_svg"]).read_text().startswith("<svg")
    assert "相关性不等于因果" in Path(paths["report"]).read_text()


# ---- 评审 round01：Pareto 方向、分组与来源/成熟度证据 ----


def test_pareto_respects_resource_maximize_direction(tmp_path):
    values = [GIB, 2 * GIB]
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": values})
    cfg = replace(cfg, sweep=replace(
        cfg.sweep,
        pareto=replace(cfg.sweep.pareto, direction="maximize")))
    result = run_sweep(
        cfg, None, _best(cfg),
        evaluate_fn=_evaluator(capacity_scores={GIB: 2.0, 2 * GIB: 1.0}))
    front = [point.state.profile.cluster_memory_bytes for point in result.pareto]
    # 资源最大化：2 GiB 时间更短且资源更大，支配 1 GiB 点。
    assert front == [2 * GIB]


def test_pareto_compares_within_source_and_maturity_groups(tmp_path):
    values = [GIB, 2 * GIB]
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": values})

    def evaluator(state):
        small = state.profile.cluster_memory_bytes == GIB
        return RoundRecord(
            sequence=0, candidate_id="c", compile_key="c", sim_key="s",
            cache="fast", status="success",
            result_maturity=("validated_in_scope" if small
                             else "model_prediction"),
            state=state, score=(2.0 if small else 1.0),
            metrics={"total_time_s": 2.0 if small else 1.0})

    result = run_sweep(cfg, None, _best(cfg), evaluate_fn=evaluator)
    # 两个点成熟度不同，不能互相支配，前沿保留两个。
    assert len(result.pareto) == 2
    for point in result.points:
        payload = point.to_json()
        assert payload["result_maturity"] in {
            "validated_in_scope", "model_prediction"}
        assert "parameter_sources" in payload
        assert payload["objective_direction"] == "minimize"
        assert payload["resource_direction"] == "minimize"


def test_scanned_link_and_cycle_values_are_downgraded_to_assumption(tmp_path):
    base_links = [link.to_json() for link in load_profile(PROFILE).links]
    cfg = _config(tmp_path, fields={"links": [base_links]})
    base = _best(cfg)
    link = dict(base.profile.links[0].to_json())
    link["bandwidth_gbps"] = link["bandwidth_gbps"] * 2
    new_links = [link] + [item.to_json() for item in base.profile.links[1:]]
    state = TuneState(base.strategy, base.op,
                      base.profile.with_fields(links=new_links))
    from tune.sweep import _mark_scanned

    marked = _mark_scanned(state.profile, {"links": new_links}, base.profile)
    assert marked.parameter_sources["comm_config.links"]["origin"] == "assumption"
    cycle_costs = dict(base.profile.cycle_costs)
    cycle_costs["MAC"] = cycle_costs["MAC"] * 2
    state = TuneState(base.strategy, base.op,
                      base.profile.with_fields(cycle_costs=cycle_costs))
    marked = _mark_scanned(state.profile, {"cycle_costs": cycle_costs},
                           base.profile)
    # CEVA 台账按原语登记：只降级实际变化的 MAC，其他原语保留原来源。
    assert marked.parameter_sources[
        "pim.pim_trace_cycle_costs.MAC"]["origin"] == "assumption"
    assert marked.parameter_sources[
        "pim.pim_trace_cycle_costs.LOADN"]["origin"] == (
        base.profile.parameter_sources[
            "pim.pim_trace_cycle_costs.LOADN"]["origin"])


def test_same_value_scan_keeps_sources_and_profile_identity(tmp_path):
    """与基准语义相同的 YAML 链路值不能降级来源或改写画像摘要。"""
    from tune.sweep import _mark_scanned

    base_links = [link.to_json() for link in load_profile(PROFILE).links]
    cfg = _config(tmp_path, fields={"links": [base_links]})
    base = _best(cfg)
    marked = _mark_scanned(base.profile, {"links": base_links}, base.profile)
    assert marked is base.profile
    assert marked.parameter_sources["comm_config.links"]["origin"] == (
        base.profile.parameter_sources["comm_config.links"]["origin"])
    assert marked.profile_sha256() == base.profile.profile_sha256()


def test_report_lists_sources_maturity_and_raw_log(tmp_path):
    cfg = _config(tmp_path, fields={"cluster_memory_bytes": [GIB, 2 * GIB]})
    result = run_sweep(
        cfg, None, _best(cfg),
        evaluate_fn=_evaluator(capacity_scores={GIB: 2.0, 2 * GIB: 1.0}))
    paths = write_sweep(result, cfg, cfg.output.dir / "sweep")
    report = Path(paths["report"]).read_text()
    assert "参数来源" in report
    assert "结果成熟度" in report
    assert "sweep.jsonl" in report


# ---- 评审 round10：扫描前核对 best 的工作负载与基线画像 ----

def _best_payload(cfg, tmp_path, *, status="success", score=2.0,
                  inputs=None, effective_config=None, state=None,
                  candidate=None, baseline_profile_sha256=None,
                  content_keys=True):
    from tune.config import write_effective_config
    from tune.inputs import file_sha256, prepare_inputs
    from tune.report import candidate_id, state_to_json
    from tune.steps import compile_key, sim_key

    state = state or _best(cfg)
    path = effective_config or write_effective_config(cfg)
    real_inputs = prepare_inputs(cfg)
    payload = {
        "status": status,
        "score": score,
        "candidate_id": candidate or candidate_id(state),
        "state": state_to_json(state),
        "baseline_profile_sha256": (
            baseline_profile_sha256 or cfg.profile.profile_sha256()),
        "inputs": (real_inputs.to_json() if inputs is None else inputs),
        "effective_config": str(path),
    }
    if content_keys:
        compile_value = compile_key(cfg, real_inputs, state)
        payload["compile_key"] = compile_value
        payload["sim_key"] = sim_key(
            cfg, real_inputs, state, compile_value)
        payload["effective_config_sha256"] = file_sha256(Path(path))
    target = tmp_path / "best.json"
    target.write_text(json.dumps(payload))
    return target, state


def _run_sweep_cli(tmp_path, monkeypatch, best_path, cfg, out_dir, *,
                   prepared=None, published=None):
    import sys
    from types import SimpleNamespace

    from scripts import tune_sweep
    from tune.inputs import prepare_inputs

    inputs = prepare_inputs(cfg) if prepared is None else prepared
    publish_value = inputs if published is None else published
    seen = {"evaluations": 0}

    def fake_load(path, overrides=None):
        if not overrides:
            return cfg
        output = Path(overrides["output"]["dir"])
        return replace(cfg, output=replace(cfg.output, dir=output))

    monkeypatch.setattr(tune_sweep, "load_config", fake_load)
    monkeypatch.setattr(tune_sweep, "prepare_inputs", lambda config: inputs)
    monkeypatch.setattr(
        tune_sweep, "publish_inputs", lambda config, value: publish_value)

    point = SimpleNamespace(
        score=1.0, values={}, status="success", reason=None)

    def fake_run_sweep(config, value, state, **kwargs):
        seen["evaluations"] += 1
        seen["best"] = kwargs.get("best")
        return SimpleNamespace(points=(point,), pareto=(), knees={})

    monkeypatch.setattr(tune_sweep, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(tune_sweep, "write_sweep", lambda *args: {})
    monkeypatch.setattr(sys, "argv", [
        "tune_sweep.py", "--config", str(tmp_path / "sweep.yaml"),
        "--best", str(best_path), "--output-dir", str(out_dir)])
    code = tune_sweep.main()
    return code, seen


def test_sweep_cli_rejects_best_with_another_trace_before_any_write(
        tmp_path, monkeypatch):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    record = prepare_inputs(cfg).to_json()
    record["trace_sha256"] = "0" * 64
    best, _ = _best_payload(cfg, tmp_path, inputs=record)
    out_dir = tmp_path / "sweep-out"
    code, seen = _run_sweep_cli(tmp_path, monkeypatch, best, cfg, out_dir)
    assert code == 3
    assert seen["evaluations"] == 0
    assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_cli_rejects_best_with_changed_model_ir_or_requests(
        tmp_path, monkeypatch):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    base = prepare_inputs(cfg).to_json()
    mutations = {
        "model_sha256": "1" * 64,
        "ir_sha256": "2" * 64,
        "max_seq": base["max_seq"] + 1,
        "num_requests": base["num_requests"] + 1,
    }
    for index, (key, value) in enumerate(mutations.items()):
        record = dict(base)
        record[key] = value
        case = tmp_path / f"case-{index}"
        case.mkdir()
        best, _ = _best_payload(cfg, case, inputs=record)
        out_dir = tmp_path / f"out-{key}"
        code, seen = _run_sweep_cli(tmp_path, monkeypatch, best, cfg, out_dir)
        assert code == 3, key
        assert seen["evaluations"] == 0, key
        assert not (out_dir / "effective_config.yaml").exists(), key


def test_sweep_cli_rejects_failed_or_scoreless_best(tmp_path, monkeypatch):
    for index, (status, score) in enumerate(
            (("error", None), ("success", None))):
        case = tmp_path / f"case-{index}"
        case.mkdir()
        cfg = _config(case, fields={"clock_period_ns": [0.63]})
        best, _ = _best_payload(cfg, case, status=status, score=score)
        out_dir = tmp_path / f"out-{index}"
        code, seen = _run_sweep_cli(case, monkeypatch, best, cfg, out_dir)
        assert code == 3
        assert seen["evaluations"] == 0
        assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_cli_rejects_best_with_another_baseline_profile(
        tmp_path, monkeypatch):
    generic = load_profile(GENERIC)
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    other = TuneState(StrategyChoice(8, 4), OpKnobs(4), generic)
    best, _ = _best_payload(
        cfg, tmp_path, state=other,
        baseline_profile_sha256=generic.profile_sha256())
    out_dir = tmp_path / "out-other-profile"
    code, seen = _run_sweep_cli(tmp_path, monkeypatch, best, cfg, out_dir)
    assert code == 3
    assert seen["evaluations"] == 0
    assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_cli_rejects_best_from_another_workload_config(
        tmp_path, monkeypatch):
    import yaml as yaml_module

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    foreign = tmp_path / "foreign" / "effective_config.yaml"
    foreign.parent.mkdir()
    other = replace(cfg, model=replace(cfg.model, prefill_seq_len=8))
    foreign.write_text(yaml_module.safe_dump(other.to_json(), sort_keys=False))
    best, _ = _best_payload(cfg, tmp_path, effective_config=foreign)
    out_dir = tmp_path / "out-foreign-config"
    code, seen = _run_sweep_cli(tmp_path, monkeypatch, best, cfg, out_dir)
    assert code == 3
    assert seen["evaluations"] == 0
    assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_cli_accepts_bound_best_and_records_binding(
        tmp_path, monkeypatch):
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    best, _ = _best_payload(cfg, tmp_path)
    out_dir = tmp_path / "out-bound"
    code, seen = _run_sweep_cli(tmp_path, monkeypatch, best, cfg, out_dir)
    assert code == 0
    assert seen["evaluations"] == 1
    assert seen["best"]["status"] == "success"
    assert seen["best"]["inputs"]["trace_sha256"]
    assert (out_dir / "effective_config.yaml").is_file()


def test_best_reference_allows_pending_ir_before_and_checks_it_after(tmp_path):
    """扫描目录还没有 IR 时先核对模型/轨迹；IR 生成后再核对内容。"""
    from tune.inputs import prepare_inputs
    from tune.sweep import verify_best_reference

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best, state = _best_payload(cfg, tmp_path)
    payload = json.loads(best.read_text())
    pending = replace(inputs, ir_ready=False, ir_sha256="")
    verify_best_reference(cfg, pending, payload, state)
    published = replace(
        pending, ir_ready=True, ir_sha256=payload["inputs"]["ir_sha256"])
    verify_best_reference(cfg, published, payload, state)
    mismatched = replace(published, ir_sha256="f" * 64)
    with pytest.raises(ValueError, match="ir_sha256"):
        verify_best_reference(cfg, mismatched, payload, state)


def test_sweep_cli_rechecks_generated_ir_after_publish(tmp_path, monkeypatch):
    """发布后生成的 IR 与 best 记录不同时，仍要在写生效配置前拒绝。"""
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    record = prepare_inputs(cfg).to_json()
    record["ir_sha256"] = "f" * 64
    best, _ = _best_payload(cfg, tmp_path, inputs=record)
    pending = replace(prepare_inputs(cfg), ir_ready=False, ir_sha256="")
    published = replace(pending, ir_ready=True, ir_sha256="e" * 64)
    out_dir = tmp_path / "out-ir-mismatch"
    code, seen = _run_sweep_cli(
        tmp_path, monkeypatch, best, cfg, out_dir,
        prepared=pending, published=published)
    assert code == 3
    assert seen["evaluations"] == 0
    assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_cli_accepts_generated_ir_matching_the_best(tmp_path, monkeypatch):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    record = prepare_inputs(cfg).to_json()
    best, _ = _best_payload(cfg, tmp_path, inputs=record)
    pending = replace(prepare_inputs(cfg), ir_ready=False, ir_sha256="")
    published = replace(
        pending, ir_ready=True, ir_sha256=record["ir_sha256"])
    out_dir = tmp_path / "out-ir-match"
    code, seen = _run_sweep_cli(
        tmp_path, monkeypatch, best, cfg, out_dir,
        prepared=pending, published=published)
    assert code == 0
    assert seen["evaluations"] == 1


# ---- 评审 round11：best 的历史生效配置必须按内容绑定 ----

def test_sweep_rejects_in_place_modified_effective_config(tmp_path):
    from tune.inputs import prepare_inputs
    from tune.sweep import verify_best_reference

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best_path, state = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    verify_best_reference(cfg, inputs, payload, state)

    effective = Path(payload["effective_config"])
    raw = yaml.safe_load(effective.read_text())
    raw["placement"]["seed"] = 1
    effective.write_text(yaml.safe_dump(raw, sort_keys=False))
    changed = replace(
        cfg, placement=replace(cfg.placement, seed=1))
    with pytest.raises(ValueError, match="best.json"):
        verify_best_reference(changed, inputs, payload, state)


def test_sweep_cli_rejects_modified_effective_config_before_any_write(
        tmp_path, monkeypatch):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    best_path, _ = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    effective = Path(payload["effective_config"])
    raw = yaml.safe_load(effective.read_text())
    raw["placement"]["seed"] = 1
    effective.write_text(yaml.safe_dump(raw, sort_keys=False))

    changed = replace(cfg, placement=replace(cfg.placement, seed=1))
    out_dir = tmp_path / "out-modified-config"
    code, seen = _run_sweep_cli(
        tmp_path, monkeypatch, best_path, changed, out_dir,
        prepared=prepare_inputs(changed), published=prepare_inputs(changed))
    assert code == 3
    assert seen["evaluations"] == 0
    assert not (out_dir / "effective_config.yaml").exists()


def test_sweep_accepts_moved_best_with_unchanged_content(tmp_path, monkeypatch):
    import shutil

    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    best_path, _ = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    moved = tmp_path / "moved"
    moved.mkdir()
    effective = moved / "effective_config.yaml"
    shutil.copyfile(payload["effective_config"], effective)
    payload["effective_config"] = str(effective)
    moved_best = moved / "best.json"
    moved_best.write_text(json.dumps(payload))

    inputs = prepare_inputs(cfg)
    out_dir = tmp_path / "out-moved-best"
    code, seen = _run_sweep_cli(
        tmp_path, monkeypatch, moved_best, cfg, out_dir,
        prepared=inputs, published=inputs)
    assert code == 0
    assert seen["evaluations"] == 1


def test_sweep_rejects_best_without_content_keys(tmp_path):
    from tune.inputs import prepare_inputs
    from tune.sweep import verify_best_reference

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best_path, state = _best_payload(cfg, tmp_path, content_keys=False)
    payload = json.loads(best_path.read_text())
    with pytest.raises(ValueError, match="编译键|仿真键"):
        verify_best_reference(cfg, inputs, payload, state)


def test_sweep_allows_scan_output_cache_and_budget_changes(tmp_path):
    from tune.inputs import prepare_inputs
    from tune.sweep import verify_best_reference

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best_path, state = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    changed = replace(
        cfg,
        output=replace(cfg.output, dir=tmp_path / "another-out"),
        cache=replace(cfg.cache, mode="strict"),
        search=replace(cfg.search, max_evaluations=4),
        sweep=replace(
            cfg.sweep, fields={"clock_period_ns": [0.63, 0.5]}),
    )
    verify_best_reference(changed, inputs, payload, state)


def test_sweep_rejects_best_with_another_placement_simulation_or_calibration(
        tmp_path):
    from tune.inputs import prepare_inputs
    from tune.sweep import verify_best_reference

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best_path, state = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    report = tmp_path / "calibration_report.json"
    report.write_text("{}")
    cases = {
        "placement": replace(
            cfg, placement=replace(cfg.placement, seed=1)),
        "simulation": replace(
            cfg, simulation=replace(
                cfg.simulation, use_cost_sidecar=True)),
        "calibration": replace(
            cfg, calibration=replace(
                cfg.calibration, validation_report=report)),
    }
    for label, changed in cases.items():
        with pytest.raises(ValueError, match="best.json"):
            verify_best_reference(changed, inputs, payload, state)


def test_binding_records_generation_content_identity(tmp_path):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)
    best_path, state = _best_payload(cfg, tmp_path)
    payload = json.loads(best_path.read_text())
    result = run_sweep(
        cfg, inputs, state, best=payload,
        evaluate_fn=_evaluator(clock_scores={0.63: 1.0}))
    binding = result.binding
    assert binding["effective_config_sha256"] == (
        payload["effective_config_sha256"])
    assert binding["compile_key"] == payload["compile_key"]
    assert binding["sim_key"] == payload["sim_key"]


def test_load_key_covers_inputs_and_baseline_profile(tmp_path):
    from tune.inputs import prepare_inputs

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    inputs = prepare_inputs(cfg)

    def evaluate(state):
        return RoundRecord(
            sequence=0, candidate_id="c", compile_key="c", sim_key="s",
            cache="fast", status="success",
            result_maturity="model_prediction", state=state, score=1.0)

    result = run_sweep(cfg, inputs, _best(cfg), evaluate_fn=evaluate)
    point = result.points[0]
    assert inputs.model_sha256[:12] in point.load_key
    assert inputs.ir_sha256[:12] in point.load_key
    assert inputs.trace_sha256[:12] in point.load_key
    assert cfg.profile.profile_sha256()[:12] in point.load_key
    assert f"seq{inputs.max_seq}" in point.load_key
    assert f"req{len(inputs.trace.requests)}" in point.load_key
    # 换成另一份输入内容必须换负载分组键。
    other = replace(inputs, trace_sha256="f" * 64)
    other_result = run_sweep(cfg, other, _best(cfg), evaluate_fn=evaluate)
    assert other_result.points[0].load_key != point.load_key


# ---- 评审 round03：扫描入口在快照前合并输出目录 ----

def test_sweep_cli_merges_output_dir_before_snapshot(tmp_path, monkeypatch):
    import sys
    from dataclasses import replace
    from types import SimpleNamespace

    from scripts import tune_sweep

    from tune.inputs import prepare_inputs as real_prepare_inputs

    cfg = _config(tmp_path, fields={"cluster_memory_bytes": [GIB]})
    best, best_state = _best_payload(cfg, tmp_path)
    inputs = real_prepare_inputs(cfg)
    seen = {}

    def fake_load(path, overrides=None):
        seen["overrides"] = overrides
        output = (Path(overrides["output"]["dir"]) if overrides
                  else cfg.output.dir)
        return replace(cfg, output=replace(cfg.output, dir=output))

    monkeypatch.setattr(tune_sweep, "load_config", fake_load)
    monkeypatch.setattr(
        tune_sweep, "state_from_json", lambda raw: best_state)

    def fake_prepare(config):
        seen["snapshot_dir"] = config.output.dir
        return inputs

    def fake_publish(config, value):
        seen["publish_dir"] = config.output.dir
        return value

    monkeypatch.setattr(tune_sweep, "prepare_inputs", fake_prepare)
    monkeypatch.setattr(tune_sweep, "publish_inputs", fake_publish)
    point = SimpleNamespace(
        score=1.0, values={}, status="success", reason=None)
    monkeypatch.setattr(
        tune_sweep, "run_sweep",
        lambda config, inputs, state, **kwargs: SimpleNamespace(
            points=(point,), pareto=(), knees={}))

    def fake_write(result, config, out_dir):
        seen["report_dir"] = Path(out_dir)
        return {}

    monkeypatch.setattr(tune_sweep, "write_sweep", fake_write)
    out_dir = tmp_path / "cli-only-output"
    monkeypatch.setattr(sys, "argv", [
        "tune_sweep.py", "--config", str(tmp_path / "sweep.yaml"),
        "--best", str(best), "--output-dir", str(out_dir)])
    assert tune_sweep.main() == 0
    assert seen["overrides"] == {"output": {"dir": str(out_dir.resolve())}}
    assert seen["snapshot_dir"] == out_dir.resolve()
    assert seen["publish_dir"] == out_dir.resolve()
    assert seen["report_dir"] == out_dir.resolve()
    assert (out_dir / "effective_config.yaml").is_file()


def test_scanned_source_entry_without_base_metadata_is_complete(tmp_path):
    """画像未登记来源的扫描字段也要写出非空单位与作用域。"""
    from tune.sweep import _mark_scanned

    base = load_profile(PROFILE)
    assert "pim.clusters_per_device" not in base.parameter_sources
    marked = _mark_scanned(
        base, {"clusters_per_device": 128}, base)
    entry = marked.parameter_sources["pim.clusters_per_device"]
    assert entry["origin"] == "assumption"
    assert entry["unit"] == "core"
    assert entry["scope"] == "per device"
    # 已有台账的字段沿用原单位与作用域。
    clock = _mark_scanned(
        base, {"clock_period_ns": 0.5}, base).parameter_sources[
            "pim.clock_period_ns"]
    assert clock["unit"] == "ns"
    assert clock["scope"] == "npm core"


# ---- 评审 round04：扫描失败原因与前置资源判定 ----

def test_downstream_failure_reason_is_kept_in_jsonl_and_report(tmp_path):
    """编译/仿真失败必须把 record.failure 写进点、原始 JSONL 与报告。"""
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})

    def failing(state):
        return RoundRecord(
            sequence=0, candidate_id="c", compile_key="c", sim_key="s",
            cache="fast", status="error",
            result_maturity="model_prediction", state=state, score=None,
            failure="compile: num_tasklets=99 注入失败")

    result = run_sweep(cfg, None, _best(cfg), evaluate_fn=failing)
    point = result.points[0]
    assert point.status == "error"
    assert point.reason == "compile: num_tasklets=99 注入失败"
    assert point.to_json()["reason"] == point.reason
    out = tmp_path / "sweep-out"
    write_sweep(result, cfg, out)
    raw = json.loads((out / "sweep.jsonl").read_text().splitlines()[0])
    assert "注入失败" in raw["reason"]
    assert "注入失败" in (out / "report.md").read_text()


def test_failure_without_failure_text_keeps_a_stage_reason(tmp_path):
    """失败记录没有原文时也不能留空白原因。"""
    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    result = run_sweep(
        cfg, None, _best(cfg),
        evaluate_fn=lambda state: RoundRecord(
            sequence=0, candidate_id="c", compile_key="c", sim_key="s",
            cache="fast", status="error",
            result_maturity="model_prediction", state=state, score=None))
    assert result.points[0].reason


def test_physical_resource_combination_is_invalid_before_compile(tmp_path):
    """4 个逻辑 DPU 配 1 个物理簇必须前置判定，不能进入昂贵编译。"""
    generic = load_profile(
        REPO_ROOT / "contracts" / "profiles" / "generic.json").with_fields(
            num_pim_devices=1, clusters_per_device=1)
    cfg = _config(tmp_path, fields={"clusters_per_device": [1]})
    cfg = replace(cfg, profile=generic, sweep=replace(
        cfg.sweep, fields={"clusters_per_device": (1,)}))
    state = TuneState(StrategyChoice(8, 4), OpKnobs(4), generic)

    def must_not_run(candidate):
        pytest.fail("物理资源非法的扫描点不应进入编译/仿真")

    result = run_sweep(cfg, None, state, evaluate_fn=must_not_run)
    point = result.points[0]
    assert point.status == "invalid"
    assert "超过物理簇数" in point.reason
    assert point.to_json()["invalid"] is True
    assert point.to_json()["reason"] == point.reason


# ---- 评审 round06：扫描来源必须指向后端的真实消费键 ----

def _with_sources(profile, updates):
    from dataclasses import replace

    sources = {key: dict(value) for key, value in
               profile.parameter_sources.items()}
    for key, origin in updates.items():
        sources[key] = {
            "origin": origin, "evidence": "既有实测证据",
            "unit": sources.get(key, {}).get("unit", "unit"),
            "scope": sources.get(key, {}).get("scope", "scope"),
        }
    return replace(profile, parameter_sources=sources)


def test_generic_capacity_and_reserve_sources_follow_pim_keys():
    """通用画像的容量/预留扫描必须更新 pim.* 真实消费键。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(GENERIC),
        {"pim.cluster_memory_capacity_gb": "customer",
         "pim.sys_reserve_bytes": "customer"})
    values = {"cluster_memory_bytes": 7 * GIB, "sys_reserve_bytes": 1024}
    profile = base.with_fields(**values)
    marked = _mark_scanned(profile, values, base)
    assert marked.parameter_sources[
        "pim.cluster_memory_capacity_gb"]["origin"] == "assumption"
    assert marked.parameter_sources[
        "pim.sys_reserve_bytes"]["origin"] == "assumption"
    # CEVA 专用键不能被写进通用画像的台账。
    assert "ceva.local_memory_bytes" not in marked.parameter_sources
    assert "ceva.sys_reserve_bytes" not in marked.parameter_sources
    # 报告来源必须与 genesim_overrides 实际下发的值一致。
    overrides = marked.genesim_overrides()
    assert overrides["pim"]["cluster_memory_capacity_gb"] == pytest.approx(
        7 * GIB / 2**30)
    assert overrides["pim"]["sys_reserve_bytes"] == 1024


def test_generic_source_only_downgrades_the_changed_field():
    """只改容量时预留的既有来源保留。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(GENERIC),
        {"pim.cluster_memory_capacity_gb": "customer",
         "pim.sys_reserve_bytes": "fitted"})
    values = {"cluster_memory_bytes": 7 * GIB}
    profile = base.with_fields(**values)
    marked = _mark_scanned(profile, values, base)
    assert marked.parameter_sources[
        "pim.cluster_memory_capacity_gb"]["origin"] == "assumption"
    assert marked.parameter_sources[
        "pim.sys_reserve_bytes"]["origin"] == "fitted"


def test_ceva_capacity_and_reserve_sources_follow_ceva_keys():
    """CEVA 画像继续使用 ceva.* 容量/预留键，不混入通用键。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(PROFILE),
        {"ceva.local_memory_bytes": "ceva_measured",
         "ceva.sys_reserve_bytes": "customer"})
    values = {"cluster_memory_bytes": 7_000_000_000,
              "sys_reserve_bytes": 4096}
    profile = base.with_fields(**values)
    marked = _mark_scanned(profile, values, base)
    assert marked.parameter_sources[
        "ceva.local_memory_bytes"]["origin"] == "assumption"
    assert marked.parameter_sources[
        "ceva.sys_reserve_bytes"]["origin"] == "assumption"
    assert "pim.cluster_memory_capacity_gb" not in marked.parameter_sources
    assert "pim.sys_reserve_bytes" not in marked.parameter_sources
    overrides = marked.genesim_overrides()
    assert overrides["ceva"]["local_memory_bytes"] == 7_000_000_000
    assert overrides["ceva"]["sys_reserve_bytes"] == 4096


def test_ceva_cycle_scan_downgrades_only_the_measured_primitive():
    """周期表扫描只降级实际变化的原语叶子来源。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(PROFILE),
        {"pim.pim_trace_cycle_costs.MAC": "ceva_measured"})
    cycle_costs = dict(base.cycle_costs)
    cycle_costs["MAC"] = cycle_costs["MAC"] * 2
    values = {"cycle_costs": cycle_costs}
    profile = base.with_fields(**values)
    marked = _mark_scanned(profile, values, base)
    assert marked.parameter_sources[
        "pim.pim_trace_cycle_costs.MAC"]["origin"] == "assumption"
    # 未变化的原语保留原来源，不新增整表以外的脏条目。
    for key in ("LOADN", "STOREN", "SYNC"):
        source_key = f"pim.pim_trace_cycle_costs.{key}"
        assert marked.parameter_sources[source_key]["origin"] == (
            base.parameter_sources[source_key]["origin"])


def test_generic_link_scan_keeps_the_shared_link_source_key():
    """链路扫描仍是 comm_config.links 整组来源，不拆错叶子。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(GENERIC), {"comm_config.links": "customer"})
    links = [link.to_json() for link in base.links]
    links[0]["bandwidth_gbps"] = links[0]["bandwidth_gbps"] * 2
    values = {"links": links}
    profile = base.with_fields(**values)
    marked = _mark_scanned(profile, values, base)
    assert marked.parameter_sources[
        "comm_config.links"]["origin"] == "assumption"
    assert marked.genesim_overrides()["comm_config"]["links"][0][
        "bandwidth_gbps"] == pytest.approx(
            base.links[0].bandwidth_gbps * 2)


# ---- 评审 round07：通用周期父键不能整组覆盖未扫描原语 ----

def _generic_cycle_scan(changes):
    """在通用画像的整表周期来源下扫描指定原语。"""
    from tune.sweep import _mark_scanned

    base = _with_sources(
        load_profile(GENERIC), {"pim.pim_trace_cycle_costs": "customer"})
    cycle_costs = dict(base.cycle_costs)
    for primitive, factor in changes.items():
        cycle_costs[primitive] = cycle_costs[primitive] * factor
    values = {"cycle_costs": cycle_costs}
    marked = _mark_scanned(base.with_fields(**values), values, base)
    return base, marked


def test_generic_cycle_scan_keeps_parent_source_for_unscanned_primitives():
    """只改 MAC 时，父键的 customer 证据仍覆盖未变化的 LOADN。"""
    base, marked = _generic_cycle_scan({"MAC": 2.0})
    leaf = marked.parameter_sources[
        "pim.pim_trace_cycle_costs.MAC"]
    assert leaf["origin"] == "assumption"
    assert leaf["unit"] == "cycle"
    assert leaf["scope"]
    # 父键保持原证据，未变化的原语不因本次扫描失去 customer 来源。
    parent = marked.parameter_sources["pim.pim_trace_cycle_costs"]
    assert parent["origin"] == "customer"
    assert parent["evidence"] == base.parameter_sources[
        "pim.pim_trace_cycle_costs"]["evidence"]
    assert marked.cycle_costs["LOADN"] == base.cycle_costs["LOADN"]
    # 报告来源必须与最终下发给 GeneSim 的生效值一致。
    overrides = marked.genesim_overrides()["pim"]["pim_trace_cycle_costs"]
    assert overrides["MAC"] == pytest.approx(base.cycle_costs["MAC"] * 2.0)
    assert overrides["LOADN"] == pytest.approx(base.cycle_costs["LOADN"])


def test_generic_cycle_scan_only_changed_primitives_become_assumption():
    """多个原语选择性变化时，只有变化叶子降级。"""
    base, marked = _generic_cycle_scan({"MAC": 2.0, "LOADN": 3.0})
    for primitive in ("MAC", "LOADN"):
        source = marked.parameter_sources[
            f"pim.pim_trace_cycle_costs.{primitive}"]
        assert source["origin"] == "assumption"
    assert marked.parameter_sources[
        "pim.pim_trace_cycle_costs"]["origin"] == "customer"
    assert "pim.pim_trace_cycle_costs.STOREN" not in marked.parameter_sources
    overrides = marked.genesim_overrides()["pim"]["pim_trace_cycle_costs"]
    assert overrides["MAC"] == pytest.approx(base.cycle_costs["MAC"] * 2.0)
    assert overrides["LOADN"] == pytest.approx(base.cycle_costs["LOADN"] * 3.0)
    assert overrides["STOREN"] == pytest.approx(base.cycle_costs["STOREN"])


# ---- 评审 round09：非法资源轴必须在任何候选求值前拒绝 ----


def test_invalid_pareto_resource_is_rejected_before_any_evaluation(
        tmp_path, monkeypatch):
    """配置非法时扫描入口在 evaluator 调用前失败，不承担编译/仿真成本。"""
    from tune import sweep

    cfg = _config(tmp_path, fields={"clock_period_ns": [0.63]})
    path = tmp_path / "sweep.yaml"
    path.write_text(path.read_text().replace(
        "resource: total_physical_memory_bytes",
        "resource: not_a_resource"))
    calls = []
    monkeypatch.setattr(
        sweep, "evaluate_candidate", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match="sweep.pareto.resource"):
        run_sweep(load_config(path), None, _best(cfg))
    assert calls == []


def test_resource_axis_helper_reuses_the_configured_enum():
    """扫描层只消费配置层同一组资源轴，未知取值仍显式失败。"""
    from tune.config import PARETO_RESOURCES
    from tune.sweep import _resource_of

    profile = load_profile(PROFILE)
    assert set(PARETO_RESOURCES) == {
        "total_physical_memory_bytes", "num_physical_cores"}
    for resource in PARETO_RESOURCES:
        assert _resource_of(profile, resource) > 0
    with pytest.raises(ValueError, match="未知的 Pareto 资源轴"):
        _resource_of(profile, "not_a_resource")
