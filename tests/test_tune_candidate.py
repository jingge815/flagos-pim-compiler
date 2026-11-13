"""分层候选空间：合法性、确定性邻居与算法复用。"""

import json
import random
from dataclasses import replace

import pytest
import yaml

from contracts.hardware_profile import load_profile
from tune.candidate import (
    CandidateSpace,
    OpKnobs,
    StrategyChoice,
    TuneState,
    build_strategy,
    model_dims,
    run_enumerate,
)
from tune.config import load_config

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"


def _model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "vocab_size": 64,
        "num_hidden_layers": 4,
        "hidden_size": 16,
    }))
    return model_dir


def _config(tmp_path, *, num_dpus=(4, 8), num_stages=(1, 2, 4),
            tasklets=(4, 8), hardware_fields=None, layers=None,
            profile=PROFILE, model_dir=None, optimization_space=None):
    space = {
        "strategy": {"enabled": True, "num_dpus": list(num_dpus),
                     "num_stages": list(num_stages)},
        "op": {"enabled": True, "num_tasklets": list(tasklets)},
        "hardware": {"enabled": bool(hardware_fields),
                     "fields": hardware_fields or {}},
    }
    if optimization_space is not None:
        space["optimization"] = optimization_space
    raw = {
        "version": 1,
        "run_name": "candidate",
        "model": {
            "model_dir": str(model_dir or _model_dir(tmp_path)), "ir": None,
            "prefill_seq_len": 128, "max_seq": None,
            "trace": {"path": None, "num_requests": 1,
                      "synthetic": {"seed": 0, "num_requests": 1}},
        },
        "profile": {"path": str(profile)},
        "space": space,
        "placement": {
            "enabled": False, "algorithm": "simulated_annealing",
            "evaluation_backend": "pipeline", "validation_backend": "pipeline",
            "objective_metric": "total_time_s",
            "objective_weights": {"compute": 0.5, "communication": 0.25,
                                  "memory": 0.25},
            "max_evaluations": 2, "max_time_s": 0, "max_proposals": None,
            "seed": 0,
            "moves": {"max_segment_ops": 8, "max_layer_block_layers": 4,
                      "weights": {}},
            "algorithm_params": {},
        },
        "search": {
            "algorithm": "enumerate", "seed": 0, "max_evaluations": 2,
            "max_time_s": 0, "max_proposals": None,
            "layer_weights": layers or {"strategy": 0.4, "op": 0.4,
                                        "hardware": 0.2},
            "algorithm_params": {},
        },
        "cache": {"mode": "fast"},
        "simulation": {"use_cost_sidecar": False},
        "calibration": {"validation_report": None},
        "sweep": {
            "fields": {}, "core_policy": "fixed_dpus",
            "knee_improvement_threshold": 0.05,
            "pareto": {"resource": "total_physical_memory_bytes",
                       "direction": "minimize"},
        },
        "output": {"dir": str(tmp_path / "out")},
    }
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def test_model_dims_read_config_json_without_weights(tmp_path):
    dims = model_dims(_model_dir(tmp_path))
    assert dims.num_heads == 4
    assert dims.num_kv_heads == 2
    assert dims.num_layers == 4


def test_enumeration_only_keeps_legal_strategies(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    states = space.states
    pairs = {(state.strategy.num_dpus, state.strategy.num_stages)
             for state in states}
    # kv_heads=2 限制 tp 宽度 ≤2：(4,2)、(4,4)、(8,4) 合法。
    assert pairs == {(4, 2), (4, 4), (8, 4)}
    assert len(states) == 3 * 2  # 3 策略 × 2 tasklet


def test_enumeration_is_sorted_by_numeric_tuple(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    keys = [space.key(state) for state in space.states]
    assert keys == sorted(keys)


def test_initial_state_is_the_first_legal_candidate(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    assert space.initial_state == space.states[0]


def test_build_strategy_uses_llama_rules(tmp_path):
    from graph.strategy import ShardStrategy

    dims = model_dims(_model_dir(tmp_path))
    state = TuneState(
        strategy=StrategyChoice(num_dpus=8, num_stages=4),
        op=OpKnobs(num_tasklets=4),
        profile=load_profile(PROFILE),
    )
    strategy = build_strategy(state, dims)
    assert isinstance(strategy, ShardStrategy)
    assert strategy.tp_width == 2
    assert strategy.num_stages == 4


def _wide_model_dir(tmp_path):
    """kv_heads=8 让 DPU 数与段数两个字段都有合法邻居。"""
    model_dir = tmp_path / "model_wide"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 32,
        "vocab_size": 64,
        "num_hidden_layers": 8,
        "hidden_size": 16,
    }))
    return model_dir


def test_strategy_neighbor_changes_exactly_one_declared_field(tmp_path):
    """策略邻居一次只改 num_dpus 或 num_stages，不能整对替换。"""
    cfg = _config(
        tmp_path, num_dpus=(8, 16), num_stages=(1, 2, 4), tasklets=(4,),
        layers={"strategy": 1, "op": 0, "hardware": 0},
        model_dir=_wide_model_dir(tmp_path))
    space = CandidateSpace(cfg, model_dims(cfg.model.model_dir))
    rng = random.Random(2)
    current = space.initial_state
    changed_fields = set()
    seen = 0
    for _ in range(200):
        proposal = space.propose(current, rng)
        if proposal is None:
            continue
        differences = [
            name for name in ("num_dpus", "num_stages")
            if getattr(current.strategy, name)
            != getattr(proposal.state.strategy, name)
        ]
        assert len(differences) == 1, (current.strategy, proposal.state.strategy)
        changed_fields.update(differences)
        current = proposal.state
        seen += 1
    assert seen > 0
    assert changed_fields == {"num_dpus", "num_stages"}


def test_zero_weight_layer_never_produces_a_proposal(tmp_path):
    """没有正权重可用层时返回 None，不能落到默认分支硬提一个。"""
    cfg = _config(
        tmp_path, num_dpus=(8,), num_stages=(4,), tasklets=(4, 8),
        layers={"strategy": 1, "op": 0, "hardware": 0})
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    assert space.propose(space.initial_state, random.Random(0)) is None


def test_proposals_are_deterministic_and_one_field_at_a_time(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    state = space.initial_state
    first = _proposals(space, state, seed=7, count=30)
    second = _proposals(space, state, seed=7, count=30)
    assert [space.key(p.state) for p in first] == [space.key(p.state) for p in second]
    previous = state
    for proposal in first:
        changed = sum((
            proposal.state.strategy != previous.strategy,
            proposal.state.op != previous.op,
            proposal.state.profile != previous.profile,
        ))
        assert changed == 1
        assert proposal.kind.endswith("_change")
        previous = proposal.state


def test_layer_weights_steer_which_layer_moves(tmp_path):
    cfg = _config(tmp_path, layers={"strategy": 0, "op": 1,
                                    "hardware": 0})
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    proposals = _proposals(space, space.initial_state, seed=3, count=20)
    assert {proposal.kind for proposal in proposals} == {"op_change"}


def test_recombination_rejects_illegal_mixes(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    first = space.states[0]
    second = space.states[-1]
    outcomes = []
    for seed in range(80):
        proposal = space.recombine(first, second, random.Random(seed))
        outcomes.append(proposal)
        if proposal is not None:
            assert _is_legal(space, proposal.state)
    assert any(proposal is None for proposal in outcomes)


def test_enumerate_returns_states_in_order_and_counts_baseline(tmp_path):
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))
    calls = []

    def objective(state):
        calls.append(space.key(state))
        return type("Evaluation", (), {
            "successful": True, "score": 1.0, "status": "success",
            "evaluator_invoked": True,
        })()

    result = run_enumerate(space, objective)
    assert calls == [space.key(state) for state in space.states]
    assert result.best_state == space.states[0]
    assert result.evaluations == len(space.states)


def test_same_seed_algorithms_produce_identical_sequences(tmp_path):
    from tune._genesim import build_algorithm, search_algorithms

    _, Budget, Evaluation, EvaluationStatus = search_algorithms()
    space = CandidateSpace(_config(tmp_path), model_dims(_model_dir(tmp_path)))

    class Objective:
        def evaluate(self, state):
            score = float(abs(hash(space.key(state))) % 1000)
            return Evaluation(score=score, status=EvaluationStatus.SUCCESS,
                              evaluator_invoked=True)

    for name in ("simulated_annealing", "genetic_algorithm", "tabu_search"):
        runs = []
        for _ in range(2):
            seen = []

            class Recorder:
                def on_event(self, event):
                    if getattr(event, "state", None) is not None:
                        seen.append(space.key(event.state))

            algorithm = build_algorithm(name, {"min_improvement": 0.0}, seed=5)
            algorithm.run(space, Objective(), Budget(max_evaluations=6), Recorder())
            runs.append(seen)
        assert runs[0] == runs[1], name
        assert runs[0], name


def test_hardware_layer_changes_profile_and_is_validated(tmp_path):
    cfg = _config(tmp_path, hardware_fields={"clock_period_ns": [0.63, 0.5]})
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    states = space.states
    clocks = {state.profile.clock_period_ns for state in states}
    assert clocks == {0.63, 0.5}
    assert len(states) == 3 * 2 * 2


def test_dry_run_counts_and_rejections(tmp_path):
    cfg = _config(tmp_path, num_dpus=(4, 8, 16), num_stages=(1, 2, 4))
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    report = space.plan()
    assert report["legal"] == len(space.states)
    assert report["rejected"]
    assert any("num_dpus" in reason or "tp_width" in reason or "num_heads" in reason
               for _, reason in report["rejected"])


def _proposals(space, state, *, seed, count):
    rng = random.Random(seed)
    proposals = []
    current = state
    for _ in range(count):
        proposal = space.propose(current, rng)
        assert proposal is not None
        proposals.append(proposal)
        current = proposal.state
    return proposals


def _is_legal(space, state):
    from graph.strategy import llama_strategy

    try:
        llama_strategy(
            state.strategy.num_dpus,
            num_stages=state.strategy.num_stages,
            num_heads=space.dims.num_heads,
            num_kv_heads=space.dims.num_kv_heads,
            intermediate_size=space.dims.intermediate_size,
            vocab_size=space.dims.vocab_size,
            num_layers=space.dims.num_layers,
        )
    except ValueError:
        return False
    return True


# ---- 评审 round02：懒生成、单字段邻居与规范化硬件值 ----


def test_initial_state_and_proposals_do_not_materialize_full_space(
        tmp_path, monkeypatch):
    """大空间不能为初始状态或邻居展开三层笛卡尔积。"""
    clock_values = [0.63 - 0.001 * index for index in range(100)]
    fields = {
        "clock_period_ns": clock_values,
        "wram_bytes_per_dpu": [65536, 131072],
        "dma_align": [64, 128],
    }
    cfg = _config(tmp_path, num_dpus=(4, 8), num_stages=(1, 2, 4),
                  tasklets=(4, 8), hardware_fields=fields)
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))

    def forbidden():
        raise AssertionError("初始状态与邻居不能物化完整候选空间")

    monkeypatch.setattr(space, "_enumerate_states", forbidden)
    state = space.initial_state
    assert state is not None
    rng = random.Random(5)
    current = state
    for _ in range(20):
        proposal = space.propose(current, rng)
        assert proposal is not None
        current = proposal.state
    assert space._hardware_combos_cache is None


def test_hardware_neighbor_changes_exactly_one_declared_field(tmp_path):
    cfg = _config(
        tmp_path, num_dpus=(4,), num_stages=(2,), tasklets=(4,),
        hardware_fields={
            "clock_period_ns": [0.63, 0.5],
            "wram_bytes_per_dpu": [65536, 131072],
            "dma_align": [64, 128],
        },
        layers={"strategy": 0, "op": 0, "hardware": 1})
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    rng = random.Random(11)
    current = space.initial_state
    changed_fields = set()
    for _ in range(40):
        proposal = space.propose(current, rng)
        assert proposal is not None
        differences = [
            key for key in cfg.space.hardware.fields
            if getattr(proposal.state.profile, key)
            != getattr(current.profile, key)
        ]
        assert len(differences) == 1, differences
        changed_fields.update(differences)
        current = proposal.state
    assert changed_fields == set(cfg.space.hardware.fields)


def test_link_scan_compares_normalized_values(tmp_path):
    base = load_profile(REPO_ROOT / "contracts" / "profiles" / "generic.json")
    links = [link.to_json() for link in base.links]
    changed = json.loads(json.dumps(links))
    changed[0]["bandwidth_gbps"] += 1.0
    cfg = _config(tmp_path, num_dpus=(8,), num_stages=(4,), tasklets=(4,),
                  hardware_fields={"links": [links, changed]},
                  layers={"strategy": 0, "op": 0, "hardware": 1},
                  profile=REPO_ROOT / "contracts" / "profiles" / "generic.json")
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    state = space.initial_state
    # 原始 YAML 字典形态与画像里的 LinkSpec 规范化后相等。
    assert space._profile_matches_space(state.profile)
    rng = random.Random(3)
    proposal = None
    for _ in range(20):
        candidate = space.propose(state, rng)
        if candidate is not None:
            proposal = candidate
            break
    assert proposal is not None, "链路备选画像不能在成员检查处被误拒"
    assert proposal.state.profile.links != state.profile.links
    assert proposal.state.profile.clock_period_ns == (
        state.profile.clock_period_ns)


def test_illegal_hardware_neighbor_records_reason(tmp_path):
    cfg = _config(tmp_path, num_dpus=(8,), num_stages=(4,), tasklets=(4,),
                  hardware_fields={"num_pim_devices": [8, 1],
                                   "clusters_per_device": [4]},
                  layers={"strategy": 0, "op": 0, "hardware": 1},
                  profile=REPO_ROOT / "contracts" / "profiles" / "generic.json")
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    state = space.initial_state
    rng = random.Random(1)
    proposals = [space.propose(state, rng) for _ in range(20)]
    # 1 台设备 × 8 簇装不下 16 个 DPU；非法邻居被拒绝并记录原因。
    assert any(proposal is None for proposal in proposals)
    rejected = space.plan()["rejected"]
    assert any("物理簇数" in reason for _, reason in rejected)


def test_hardware_combination_must_fit_physical_dpus(tmp_path):
    """8 个 DPU 配 1 个物理簇不是合法候选，dry-run 不能报成 1 个。"""
    generic = load_profile(REPO_ROOT / "contracts" / "profiles" / "generic.json")
    cfg = _config(tmp_path, num_dpus=(4,), num_stages=(2,),
                  hardware_fields={"num_pim_devices": [1],
                                   "clusters_per_device": [1]},
                  profile=REPO_ROOT / "contracts" / "profiles" / "generic.json")
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    assert space.plan()["legal"] == 0
    assert any("物理簇" in reason for _, reason in space.plan()["rejected"])
    with pytest.raises(ValueError, match="合法"):
        _ = space.states


def test_three_algorithms_propose_legal_hardware_values(tmp_path):
    """链路、周期表与主机模型的硬件邻居在三种算法下都合法可提案。"""
    from tune._genesim import build_algorithm, search_algorithms

    generic_path = REPO_ROOT / "contracts" / "profiles" / "generic.json"
    base = load_profile(generic_path)
    links = [link.to_json() for link in base.links]
    changed_links = json.loads(json.dumps(links))
    changed_links[0]["bandwidth_gbps"] += 1.0
    cycles = dict(base.cycle_costs)
    changed_cycles = {**cycles, "MAC": cycles["MAC"] * 2.0}
    cpu = dict(base.cpu_model)
    changed_cpu = {**cpu, "base_latency_us": cpu["base_latency_us"] * 2.0}
    cfg = _config(
        tmp_path, num_dpus=(8,), num_stages=(4,), tasklets=(4,),
        hardware_fields={
            "links": [links, changed_links],
            "cycle_costs": [cycles, changed_cycles],
            "cpu_model": [cpu, changed_cpu],
        },
        layers={"strategy": 0, "op": 0, "hardware": 1},
        profile=generic_path)
    space = CandidateSpace(cfg, model_dims(_model_dir(tmp_path)))
    _, Budget, Evaluation, EvaluationStatus = search_algorithms()
    for name in ("simulated_annealing", "genetic_algorithm", "tabu_search"):
        seen = []

        class Objective:
            def evaluate(self, state):
                assert space._state_rejection(state) is None
                assert space._profile_matches_space(state.profile)
                seen.append(state)
                return Evaluation(
                    score=1.0, status=EvaluationStatus.SUCCESS,
                    evaluator_invoked=True)

        algorithm = build_algorithm(name, {"min_improvement": 0.0}, seed=3)
        algorithm.run(space, Objective(), Budget(max_evaluations=5))
        assert seen, name
        assert any(
            state.profile != space.initial_state.profile for state in seen), name


# ---- 第四个搜索维度：编译优化开关 ----


def _four_dim_config(tmp_path, *, layers=None):
    return _config(
        tmp_path, num_dpus=(4, 8), num_stages=(2, 4), tasklets=(4, 8),
        optimization_space={
            "enabled": True, "field": "loop_tiling", "values": [True, False]},
        layers=layers,
        model_dir=_wide_model_dir(tmp_path))


def test_four_layer_enumeration_covers_two_to_the_fourth(tmp_path):
    """2 优化开关 × 2 tasklet × 2 DPU 数 × 2 PP 段数 = 16。"""
    cfg = _four_dim_config(tmp_path)
    space = CandidateSpace(cfg, model_dims(cfg.model.model_dir))
    states = space.states

    assert len(states) == 16
    assert len({space.key(state) for state in states}) == 16
    assert {state.optimization.loop_tiling for state in states} == {True, False}
    assert {state.strategy.tp_width for state in states} == {1, 2, 4}
    # 未被扫描的开关沿用顶层配置，不能跟着维度乱动。
    assert all(state.optimization.cse for state in states)


def test_optimization_neighbor_changes_only_the_switch(tmp_path):
    cfg = _four_dim_config(
        tmp_path, layers={"strategy": 0, "op": 0, "hardware": 0,
                          "optimization": 1})
    space = CandidateSpace(cfg, model_dims(cfg.model.model_dir))
    state = space.initial_state
    proposal = space.propose(state, random.Random(0))

    assert proposal is not None
    assert proposal.metadata["layer"] == "optimization"
    other = proposal.state
    assert other.strategy == state.strategy
    assert other.op == state.op
    assert other.profile == state.profile
    assert other.optimization != state.optimization
