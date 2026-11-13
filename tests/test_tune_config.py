"""统一调优配置：未知字段与非法取值在任何编译前报错。"""

import json
from pathlib import Path

import pytest
import yaml

from contracts.op_contract import PimOptimizations
from tune.config import (
    RequestBound,
    TraceSnapshot,
    load_config,
    run_identity,
    validate_workload_bounds,
    write_effective_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"

BASE = {
    "version": 1,
    "run_name": "unit",
    "model": {
        "model_dir": "model",
        "ir": None,
        "prefill_seq_len": 128,
        "max_seq": None,
        "trace": {
            "path": None,
            "num_requests": 1,
            "synthetic": {"seed": 0, "num_requests": 10},
        },
    },
    "profile": {"path": str(PROFILE)},
    "space": {
        "strategy": {"enabled": True, "num_dpus": [8], "num_stages": [2, 4]},
        "op": {"enabled": True, "num_tasklets": [4]},
        "hardware": {"enabled": False, "fields": {}},
    },
    "placement": {
        "enabled": False,
        "algorithm": "simulated_annealing",
        "evaluation_backend": "pipeline",
        "validation_backend": "pipeline",
        "objective_metric": "total_time_s",
        "objective_weights": {"compute": 0.5, "communication": 0.25, "memory": 0.25},
        "max_evaluations": 2,
        "max_time_s": 0,
        "max_proposals": None,
        "seed": 0,
        "moves": {"max_segment_ops": 8, "max_layer_block_layers": 4, "weights": {}},
        "algorithm_params": {},
    },
    "search": {
        "algorithm": "enumerate",
        "seed": 0,
        "max_evaluations": 2,
        "max_time_s": 0,
        "max_proposals": None,
        "layer_weights": {"strategy": 0.4, "op": 0.4, "hardware": 0.2},
        "algorithm_params": {},
    },
    "cache": {"mode": "fast"},
    "simulation": {"use_cost_sidecar": False},
    "calibration": {"validation_report": None},
    "sweep": {
        "fields": {},
        "core_policy": "fixed_dpus",
        "knee_improvement_threshold": 0.05,
        "pareto": {"resource": "total_physical_memory_bytes", "direction": "minimize"},
    },
    "output": {"dir": "out"},
}


def write_config(tmp_path: Path, mutate=None, *, raw=None) -> Path:
    data = json.loads(json.dumps(raw if raw is not None else BASE))
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "model" / "config.json").write_text("{}")
    if mutate is not None:
        mutate(data)
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_valid_config_resolves_paths_against_yaml_dir(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.run_name == "unit"
    assert cfg.model.model_dir == tmp_path / "model"
    assert cfg.output.dir == tmp_path / "out"
    assert cfg.profile_path == PROFILE
    assert cfg.space.strategy.num_stages == (2, 4)
    assert cfg.cache.mode == "fast"


def test_unknown_top_level_field_is_rejected(tmp_path):
    def mutate(data):
        data["mystery"] = 1
    with pytest.raises(ValueError, match="未知字段"):
        load_config(write_config(tmp_path, mutate))


def test_unknown_nested_field_is_rejected(tmp_path):
    def mutate(data):
        data["space"]["op"]["super_tasklets"] = 8
    with pytest.raises(ValueError, match="super_tasklets"):
        load_config(write_config(tmp_path, mutate))


def test_bool_is_not_accepted_as_integer(tmp_path):
    def mutate(data):
        data["search"]["max_evaluations"] = True
    with pytest.raises(ValueError, match="max_evaluations"):
        load_config(write_config(tmp_path, mutate))


def test_negative_budget_is_rejected(tmp_path):
    def mutate(data):
        data["search"]["max_time_s"] = -1
    with pytest.raises(ValueError, match="max_time_s"):
        load_config(write_config(tmp_path, mutate))


def test_missing_profile_is_rejected(tmp_path):
    def mutate(data):
        data["profile"]["path"] = "no_such_profile.json"
    with pytest.raises(FileNotFoundError, match="硬件画像"):
        load_config(write_config(tmp_path, mutate))


def test_trace_has_exactly_one_source(tmp_path):
    def mutate(data):
        data["model"]["trace"]["path"] = "req.json"
    with pytest.raises(ValueError, match="二选一"):
        load_config(write_config(tmp_path, mutate))


def test_unknown_move_weight_is_rejected(tmp_path):
    def mutate(data):
        data["placement"]["moves"]["weights"] = {"teleport": 1.0}
    with pytest.raises(ValueError, match="teleport"):
        load_config(write_config(tmp_path, mutate))


def test_illegal_algorithm_is_rejected(tmp_path):
    def mutate(data):
        data["search"]["algorithm"] = "anneal"
    with pytest.raises(ValueError, match="anneal"):
        load_config(write_config(tmp_path, mutate))


def test_illegal_algorithm_params_are_rejected(tmp_path):
    def mutate(data):
        data["search"]["algorithm_params"] = {"no_such_knob": 1}
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, mutate))


def test_closed_layer_must_have_exactly_one_value(tmp_path):
    def mutate(data):
        data["space"]["strategy"]["enabled"] = False
    with pytest.raises(ValueError, match="冻结"):
        load_config(write_config(tmp_path, mutate))


def test_empty_range_is_rejected(tmp_path):
    def mutate(data):
        data["space"]["strategy"]["num_stages"] = []
    with pytest.raises(ValueError, match="num_stages"):
        load_config(write_config(tmp_path, mutate))


def test_efficiency_objective_requires_pipeline_validation(tmp_path):
    def mutate(data):
        data["placement"]["objective_metric"] = "system_efficiency_utility"
        data["placement"]["validation_backend"] = "predictor"
    with pytest.raises(ValueError, match="pipeline"):
        load_config(write_config(tmp_path, mutate))


def test_validation_backend_must_be_pipeline(tmp_path):
    def mutate(data):
        data["placement"]["evaluation_backend"] = "predictor"
        data["placement"]["validation_backend"] = "predictor"
    with pytest.raises(ValueError, match="validation_backend"):
        load_config(write_config(tmp_path, mutate))
    # predictor 只做筛选，最终验证仍是 pipeline 时合法（检查点见专项用例）。
    profile_path = _predictor_config(tmp_path)

    def accept(data):
        data["profile"]["path"] = str(profile_path)
        data["placement"].update(
            {"evaluation_backend": "predictor", "validation_backend": "pipeline"})
    valid = load_config(write_config(tmp_path, accept))
    assert valid.placement.evaluation_backend == "predictor"


def test_ceva_profile_rejects_native_placement_search(tmp_path):
    def mutate(data):
        data["placement"]["enabled"] = True
    with pytest.raises(ValueError, match="CEVA"):
        load_config(write_config(tmp_path, mutate))


def test_cli_overrides_merge_and_persist(tmp_path):
    path = write_config(tmp_path)
    cfg = load_config(path, {
        "search": {"max_evaluations": 5, "algorithm": "simulated_annealing"},
        "cache": {"mode": "strict"},
        "output": {"dir": str(tmp_path / "other")},
    })
    assert cfg.search.max_evaluations == 5
    assert cfg.search.algorithm == "simulated_annealing"
    assert cfg.cache.mode == "strict"
    assert cfg.output.dir == tmp_path / "other"
    written = write_effective_config(cfg)
    reloaded = yaml.safe_load(written.read_text())
    assert reloaded["search"]["max_evaluations"] == 5
    assert reloaded["cache"]["mode"] == "strict"
    # 高频覆盖没有形成第二份隐式状态：生效文件与内存对象一致。
    assert load_config(written).search.max_evaluations == 5


def test_cli_paths_resolve_against_cwd(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tune.config import cli_overrides

    monkeypatch.chdir(tmp_path)
    overrides = cli_overrides(SimpleNamespace(
        max_evaluations=None, max_time_s=None, algorithm=None,
        profile="p.json", cache_mode=None, output_dir="rel-out"))
    assert Path(overrides["profile"]["path"]) == tmp_path / "p.json"
    assert Path(overrides["output"]["dir"]) == tmp_path / "rel-out"


def test_validate_workload_bounds_derives_max_seq_and_checks_prefill():
    trace = TraceSnapshot(
        source="unit",
        requests=(RequestBound(prompt_tokens=128, output_tokens=32),),
    )
    model_dir = REPO_ROOT
    from tune.config import ModelConfig, TraceConfig

    model = ModelConfig(
        model_dir=model_dir, ir=None, prefill_seq_len=128, max_seq=None,
        trace=TraceConfig(path=None, num_requests=1, synthetic=None),
    )
    assert validate_workload_bounds(model, trace) == 160
    too_short = ModelConfig(
        model_dir=model_dir, ir=None, prefill_seq_len=64, max_seq=None,
        trace=model.trace,
    )
    with pytest.raises(ValueError, match="prefill"):
        validate_workload_bounds(too_short, trace)
    insufficient = ModelConfig(
        model_dir=model_dir, ir=None, prefill_seq_len=128, max_seq=100,
        trace=model.trace,
    )
    with pytest.raises(ValueError, match="max_seq"):
        validate_workload_bounds(insufficient, trace)


# ---- 评审 round01：配置字段与跨字段约束 ----


def test_closed_hardware_layer_must_have_exactly_one_value(tmp_path):
    """关闭的硬件层是冻结：两个时钟值会展开成多个候选。"""
    def mutate(data):
        data["space"]["hardware"] = {
            "enabled": False,
            "fields": {"clock_period_ns": [0.63, 0.5]},
        }
    with pytest.raises(ValueError, match="冻结"):
        load_config(write_config(tmp_path, mutate))


def test_zero_clock_is_rejected(tmp_path):
    def mutate(data):
        data["space"]["hardware"] = {
            "enabled": True, "fields": {"clock_period_ns": [0.0]}}
    with pytest.raises(ValueError, match="clock_period_ns"):
        load_config(write_config(tmp_path, mutate))


def test_all_zero_move_weights_are_rejected(tmp_path):
    def mutate(data):
        data["placement"]["moves"]["weights"] = {
            name: 0.0 for name in (
                "op_relocate", "op_swap", "connected_segment_relocate",
                "attention_branch_relocate", "kv_group_pack_or_spread",
                "layer_block_relocate", "layer_boundary_shift",
                "pim_rebalance_or_swap")}
        data["placement"]["enabled"] = True
    with pytest.raises(ValueError, match="移动权重"):
        load_config(write_config(tmp_path, mutate))


def test_non_independent_profile_fields_cannot_be_scanned(tmp_path):
    for field in ("parameter_sources", "bytes_per_element", "name",
                  "base_genesim_config"):
        def mutate(data):
            data["space"]["hardware"] = {
                "enabled": True, "fields": {field: [1]}}
        with pytest.raises(ValueError, match=field):
            load_config(write_config(tmp_path, mutate))


def test_zone_factor_scan_reuses_genesim_encoding_rules(tmp_path):
    """非法区间倍率必须在编译前被拒，而不是等 GeneSim 启动才报错。"""
    def mutate(data):
        data["space"]["hardware"] = {
            "enabled": True,
            "fields": {"zone_cycle_factors": [{
                "weight": 1.0, "kv": 1.0, "activation": 0.2}]},
        }

    with pytest.raises(ValueError, match="区间倍率"):
        load_config(write_config(tmp_path, mutate))


def test_link_scan_is_normalized_from_yaml_dicts(tmp_path):
    from contracts.hardware_profile import load_profile

    profile = load_profile(PROFILE)
    link = profile.links[0].to_json()
    link["bandwidth_gbps"] = 1024.0
    links = [link] + [
        item.to_json() for item in profile.links[1:]]
    cfg = load_config(write_config(
        tmp_path, lambda data: data["space"]["hardware"].update(
            {"enabled": True, "fields": {"links": [links]}})))
    assert cfg.space.hardware.fields["links"][0][0]["bandwidth_gbps"] == 1024.0


# ---- 评审 round03：硬件候选按完整变体复核、predictor 输入契约前置 ----

def test_legal_128core_candidate_survives_base_profile_precheck(tmp_path):
    """基线 64 核画像不能提前拒绝硬件层展开出的 128 核变体。"""
    from tune.candidate import CandidateSpace, model_dims

    def mutate(data):
        data["space"]["strategy"] = {
            "enabled": True, "num_dpus": [128], "num_stages": [16]}
        data["space"]["hardware"] = {
            "enabled": True, "fields": {"clusters_per_device": [64, 128]}}
    path = write_config(tmp_path, mutate)
    (tmp_path / "model" / "config.json").write_text(json.dumps({
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "intermediate_size": 128, "vocab_size": 64,
        "num_hidden_layers": 32, "hidden_size": 64,
    }))
    cfg = load_config(path)
    space = CandidateSpace(cfg, model_dims(tmp_path / "model"))
    pairs = {(state.strategy.num_dpus, state.strategy.num_stages)
             for state in space.states}
    assert (128, 16) in pairs
    # 64 核变体仍然保留拒绝原因，不能静默丢掉非法点。
    assert any("64" in reason and "超过物理簇数" in reason
               for _, reason in space.plan()["rejected"])


def _predictor_config(tmp_path, *, checkpoint=True,
                      checkpoint_bytes=None):
    """复制基配置并指向一个临时检查点，供 predictor 前置校验使用。"""
    import torch

    genesim = REPO_ROOT.parent / "genesim"
    base = yaml.safe_load((genesim / "conf" / "sim.yaml").read_text())
    checkpoint_path = tmp_path / "predictor.pt"
    if checkpoint_bytes is not None:
        checkpoint_path.write_bytes(checkpoint_bytes)
    elif checkpoint:
        torch.save({"checkpoint_schema_version": "genesim_gnn_checkpoint_v6"},
                   checkpoint_path)
    base["predictor"] = dict(base.get("predictor") or {},
                             checkpoint_path=str(checkpoint_path))
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    profile = json.loads(Path(PROFILE).read_text())
    profile["base_genesim_config"] = str(base_path)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    return profile_path


def test_predictor_requires_the_fixed_100ns_host_relay(tmp_path):
    profile_path = _predictor_config(tmp_path)

    def mutate(data):
        data["profile"]["path"] = str(profile_path)
        data["placement"]["evaluation_backend"] = "predictor"
        data["space"]["hardware"] = {
            "enabled": True,
            "fields": {"host_relay_overhead_ns": [100.0, 200.0]}}
    with pytest.raises(ValueError, match="100 ns"):
        load_config(write_config(tmp_path, mutate))

    def mutate_sweep(data):
        data["profile"]["path"] = str(profile_path)
        data["placement"]["evaluation_backend"] = "predictor"
        data["sweep"]["fields"] = {"host_relay_overhead_ns": [200.0]}
    with pytest.raises(ValueError, match="100 ns"):
        load_config(write_config(tmp_path, mutate_sweep))


def test_predictor_requires_existing_parseable_checkpoint(tmp_path):
    missing = _predictor_config(tmp_path, checkpoint=False)

    def mutate(data, path):
        data["profile"]["path"] = str(path)
        data["placement"]["evaluation_backend"] = "predictor"
    with pytest.raises(ValueError, match="检查点不存在"):
        load_config(write_config(tmp_path, lambda data: mutate(data, missing)))

    corrupt = _predictor_config(tmp_path, checkpoint_bytes=b"not-a-torch-file")
    with pytest.raises(ValueError, match="无法解析"):
        load_config(write_config(
            tmp_path, lambda data: mutate(data, corrupt)))

    bad_version = tmp_path / "bad-version.pt"
    import torch

    torch.save({"checkpoint_schema_version": "v0"}, bad_version)
    wrong = _predictor_config(tmp_path)
    base = yaml.safe_load((tmp_path / "base.yaml").read_text())
    base["predictor"] = dict(base.get("predictor") or {},
                             checkpoint_path=str(bad_version))
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    with pytest.raises(ValueError, match="schema"):
        load_config(write_config(tmp_path, lambda data: mutate(data, wrong)))


def test_predictor_with_100ns_and_valid_checkpoint_is_accepted(tmp_path):
    profile_path = _predictor_config(tmp_path)

    def mutate(data):
        data["profile"]["path"] = str(profile_path)
        data["placement"]["evaluation_backend"] = "predictor"
    cfg = load_config(write_config(tmp_path, mutate))
    assert cfg.placement.evaluation_backend == "predictor"
    assert cfg.profile.host_relay_overhead_ns == pytest.approx(100.0)


# ---- 评审 round09：Pareto 资源轴必须在配置层拒绝 ----


def test_pareto_resource_axis_must_be_known(tmp_path):
    """非法资源轴在 load_config 给出字段级错误，不能延迟到报告阶段。"""
    def mutate(data):
        data["sweep"]["pareto"]["resource"] = "not_a_resource"

    with pytest.raises(ValueError, match="sweep.pareto.resource"):
        load_config(write_config(tmp_path, mutate))


@pytest.mark.parametrize("resource", [
    "total_physical_memory_bytes", "num_physical_cores"])
def test_legal_pareto_resource_axes_survive_config_load(tmp_path, resource):
    """两种合法资源轴不能被收紧校验误伤。"""
    def mutate(data):
        data["sweep"]["pareto"]["resource"] = resource

    cfg = load_config(write_config(tmp_path, mutate))
    assert cfg.sweep.pareto.resource == resource


# ---- 编译优化开关：缺省全开，显式关闭后进入生效配置与运行身份 ----


def test_optimization_switches_default_to_all_enabled(tmp_path):
    """旧配置没有 optimization 段时保持原有五类优化行为。"""
    cfg = load_config(write_config(tmp_path))
    assert cfg.optimization == PimOptimizations()


def test_optimization_switches_parse_and_reach_effective_config(tmp_path):
    def mutate(data):
        data["optimization"] = {
            "constant_fold": False,
            "cse": False,
            "dce": False,
            "loop_tiling": False,
        }

    cfg = load_config(write_config(tmp_path, mutate))
    assert cfg.optimization == PimOptimizations(
        constant_fold=False, cse=False, dce=False, loop_tiling=False)
    assert cfg.to_json()["optimization"] == {
        "constant_fold": False,
        "cse": False,
        "dce": False,
        "loop_tiling": False,
    }
    effective = write_effective_config(cfg).read_text()
    assert "optimization:" in effective
    assert "loop_tiling: false" in effective


def test_unknown_optimization_switch_is_rejected(tmp_path):
    def mutate(data):
        data["optimization"] = {"super_fusion": True}

    with pytest.raises(ValueError, match="super_fusion"):
        load_config(write_config(tmp_path, mutate))


def test_optimization_switch_changes_run_identity(tmp_path):
    """同一配置换优化开关必须让续跑/缓存身份失效。"""
    def mutate(data):
        data["optimization"] = {"cse": False}

    (tmp_path / "plain").mkdir()
    (tmp_path / "changed").mkdir()
    plain = load_config(write_config(tmp_path / "plain"))
    changed = load_config(write_config(tmp_path / "changed", mutate))
    assert run_identity(plain) != run_identity(changed)


# ---- 优化开关作为第四搜索维度：只扫描一个开关，其余取顶层配置 ----


def test_optimization_space_parses_one_switch(tmp_path):
    def mutate(data):
        data["space"]["optimization"] = {
            "enabled": True, "field": "loop_tiling", "values": [True, False]}

    cfg = load_config(write_config(tmp_path, mutate))
    assert cfg.space.optimization.enabled is True
    assert cfg.space.optimization.field == "loop_tiling"
    assert cfg.space.optimization.values == (True, False)
    assert cfg.to_json()["space"]["optimization"] == {
        "enabled": True, "field": "loop_tiling", "values": [True, False]}
    # 生效配置可以原样再解析一次，续跑不会因为新增段而拒绝。
    written = write_effective_config(cfg)
    assert load_config(written).space.optimization == cfg.space.optimization


def test_optimization_space_is_omitted_when_not_configured(tmp_path):
    """旧配置没有该层时生效配置也不写，保持既有文本形态。"""
    cfg = load_config(write_config(tmp_path))
    assert cfg.space.optimization.field is None
    assert "optimization" not in cfg.to_json()["space"]


def test_optimization_space_rejects_unknown_switch(tmp_path):
    def mutate(data):
        data["space"]["optimization"] = {
            "enabled": True, "field": "fast_math", "values": [True, False]}

    with pytest.raises(ValueError, match="fast_math"):
        load_config(write_config(tmp_path, mutate))


def test_optimization_space_rejects_non_bool_values(tmp_path):
    def mutate(data):
        data["space"]["optimization"] = {
            "enabled": True, "field": "cse", "values": [True, 1]}

    with pytest.raises(ValueError, match="space.optimization.values"):
        load_config(write_config(tmp_path, mutate))


def test_frozen_optimization_layer_takes_exactly_one_value(tmp_path):
    def mutate(data):
        data["space"]["optimization"] = {
            "enabled": False, "field": "cse", "values": [True, False]}

    with pytest.raises(ValueError, match="冻结"):
        load_config(write_config(tmp_path, mutate))
