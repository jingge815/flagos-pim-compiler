"""候选执行：原子发布、缓存失效、一致性校验与失败隔离。"""

import json
import shutil
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from contracts.hardware_profile import load_profile
from contracts.op_contract import PimOptimizations
from tune.candidate import OpKnobs, StrategyChoice, TuneState
from tune.config import RequestBound, TraceSnapshot, load_config
from tune.inputs import TuneInputs, file_sha256
from tune.report import RoundRecord
from tune.steps import (
    SimOutput,
    build_candidate,
    build_sim,
    compile_key,
    evaluate_candidate,
    sim_key,
    verify_compiled,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"
GENERIC_PROFILE_PATH = REPO_ROOT / "contracts" / "profiles" / "generic.json"


def _config(tmp_path, profile=PROFILE_PATH):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text('{"a": 1}')
    ir = tmp_path / "model.ir"
    ir.write_text('{"num_layers": 1, "subgraphs": [[0]], "operators": '
                  '[{"op_id": 0, "op_type": "GEMM"}]}')
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"requests": [{
        "request_id": "r0", "input_tokens": [1, 2, 3, 4],
        "output_length": 2, "arrival_time_ns": 0, "is_prefill": True,
        "seq_len": 4, "prompt": None, "padded_query_length": None,
        "initial_kv_length": 0}]}))
    raw = {
        "version": 1, "run_name": "steps",
        "model": {"model_dir": str(model_dir), "ir": str(ir),
                  "prefill_seq_len": 4, "max_seq": None,
                  "trace": {"path": str(trace), "num_requests": 1,
                            "synthetic": None}},
        "profile": {"path": str(profile)},
        "space": {"strategy": {"enabled": True, "num_dpus": [4],
                               "num_stages": [2]},
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
        "sweep": {"fields": {}, "core_policy": "fixed_dpus",
                  "knee_improvement_threshold": 0.05,
                  "pareto": {"resource": "total_physical_memory_bytes",
                             "direction": "minimize"}},
        "output": {"dir": str(tmp_path / "out")},
    }
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_config(path)


def _inputs(cfg):
    from tune.inputs import snapshot_inputs

    return snapshot_inputs(cfg)


def _state(cfg):
    return TuneState(StrategyChoice(4, 2), OpKnobs(4), cfg.profile)


@lru_cache(maxsize=1)
def _compiler_toolchain():
    """当前进程实际编译工具的显式身份，供 sidecar 夹具复用。"""
    from opcompiler_bridge.driver import compiler_identity

    return compiler_identity()


def _fake_compile(cfg, inputs, state, out_dir, force_compile):
    hardware = state.profile.to_pim_hardware_config(
        state.strategy.num_dpus, state.op.num_tasklets,
        optimizations=state.resolved_optimization(cfg))
    from tune.steps import _fixed_partition_plan

    plan = _fixed_partition_plan(state)
    plan.write(out_dir / "partition_plan.json")
    pimir = out_dir / "raw.pimir.mlir"
    pimir.write_text(
        "module attributes {"
        f'"pim.dma-align" = {hardware.dma_align} : i64, '
        f'"pim.mram-bytes" = {hardware.mram_bytes_per_dpu} : i64, '
        f'"pim.num-dpus" = {hardware.num_dpus} : i32, '
        f'"pim.num-tasklets" = {hardware.num_tasklets} : i32, '
        f'pim.target = "{hardware.compiler_target}", '
        f'"pim.wram-bytes" = {hardware.wram_bytes_per_dpu} : i64'
        "} {\n}\n")
    sidecar = {
        "version": 5,
        "requires_pimir": True,
        "operators": {"0": {
            "op_type": "GEMM", "shards": [], "pimir_path": str(pimir),
            "pimir_sha256": file_sha256(pimir), "pimir_route": "a",
        }},
        "dpu_to_cluster": [list(entry) for entry in plan.dpu_to_cluster],
        "hardware": {
            "mram_bytes_per_dpu": state.profile.cluster_memory_bytes,
            "sys_reserve_bytes": state.profile.sys_reserve_bytes,
            "usable_mram_bytes": (state.profile.cluster_memory_bytes
                                  - state.profile.sys_reserve_bytes),
        },
        "compiler": {
            "strategy": {
                "name": (f"tp{state.strategy.tp_width}"
                         f"_pp{state.strategy.num_stages}"),
                "num_dpus": plan.num_dpus,
                "num_stages": plan.num_stages,
                "tp_width": state.strategy.tp_width,
                "dpu_ids": list(plan.dpu_ids or ()),
            },
            "compile_profile_sha256": state.profile.compile_profile_sha256(),
            "hardware": {
                "num_dpus": hardware.num_dpus,
                "num_tasklets": hardware.num_tasklets,
                "mram_bytes_per_dpu": hardware.mram_bytes_per_dpu,
                "wram_bytes_per_dpu": hardware.wram_bytes_per_dpu,
                "dma_align": hardware.dma_align,
                "compiler_target": hardware.compiler_target,
                "optimizations": hardware.optimizations.to_payload(),
            },
            "physical_mram_bytes": state.profile.cluster_memory_bytes,
            "reserved_mram_bytes": state.profile.sys_reserve_bytes,
            "usable_mram_bytes": (state.profile.cluster_memory_bytes
                                  - state.profile.sys_reserve_bytes),
            "toolchain": _compiler_toolchain(),
        },
    }
    path = out_dir / "placement.json"
    path.write_text(json.dumps(sidecar))
    # 图编译的固定产物之一；编译完成判据要求它必须随目录一起发布。
    (out_dir / "placed.ir").write_text('{"operators": []}')
    return SimpleNamespace(sidecar=path, out_dir=out_dir)


def _fake_refine(cfg, inputs, state, placement, out_ir, cost_path,
                 force_compile):
    # 发布时由 build_candidate 回写放置 sidecar 内容身份，夹具只产精化量。
    out_ir.write_text(json.dumps({"operators": [
        {"op_id": 0, "op_type": "GEMM"}]}))
    cost_path.write_text("{}")


# CEVA 固定映射的仿真没有原生库：摘要记录的计时后端就是内置轨迹。
NATIVE_TRACE_FALLBACK = {
    "pim_type": "CEVA-NPM", "core": "pim_trace",
    "native": False, "fallback": None,
}
SUMMARY = {"total_time_s": 1.5, "throughput_tokens_per_s": 100.0,
           "result_producer": "pipeline_simulator",
           "pim_native_backend": NATIVE_TRACE_FALLBACK,
           "processed_tokens": 6, "completed_requests": 1}


def _write_effective_placement(sim_dir, targets=None):
    """写一份最小但完整的最终放置：op0 挂在一个 PIM PU 上。"""
    path = sim_dir / "effective_placement.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "targets_include_relay": False,
        "inputs_sha256": "0" * 64,
        "execution_graph_sha256": "1" * 64,
        "final_graph_sha256": "2" * 64,
        "targets": targets if targets is not None else [
            {"op_id": 0, "node_id": 0, "vpu_id": 1, "attached_pu_id": 0}],
    }))
    return path


def _fixed_autotune():
    """固定映射的仿真指标：零搜索、一次最终执行。"""
    return {
        "enabled": False, "replayed": False,
        "algorithm": "simulated_annealing", "stop_reason": "disabled",
        "evaluations": 0, "evaluation_backend": "pipeline",
        "final_backend_evaluations": 1, "total_backend_evaluations": 1,
        "validation_backend": "pipeline",
        "search_and_validation_backend_match": True,
        "final_backend_matches_search": True,
    }


def _search_autotune(evaluations=1):
    """原生搜索的仿真指标：搜索评估数 + 一次最终执行。"""
    return {
        "enabled": True, "algorithm": "simulated_annealing",
        "stop_reason": "max_evaluations",
        "evaluations": evaluations, "evaluation_backend": "pipeline",
        "final_backend_evaluations": 1,
        "total_backend_evaluations": evaluations + 1,
        "validation_backend": "pipeline",
        "search_and_validation_backend_match": True,
        "final_backend_matches_search": True,
    }


def _fake_sim(cfg, sim_dir, config_path,
              targets=None, traces=True, autotune=None, write_autotune=True,
              native_backend=None):
    summary = dict(SUMMARY)
    if native_backend is not None:
        summary["pim_native_backend"] = native_backend
    sim_dir.mkdir(parents=True, exist_ok=True)
    results = sim_dir / "results"
    results.mkdir(exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary))
    if autotune is None:
        autotune = (_search_autotune() if cfg.placement.enabled
                    else _fixed_autotune())
    if write_autotune:
        (results / "autotune_metrics.json").write_text(json.dumps(autotune))
    trace_dir = sim_dir / "pim_traces"
    trace_dir.mkdir(exist_ok=True)
    if traces:
        (trace_dir / "op_0_GEMM.pim_trace").write_text(json.dumps(
            {"op_id": 0, "compile_signature": {"trace_source": "pimir"}}))
    placement = _write_effective_placement(sim_dir, targets)
    return SimOutput(
        sim_dir=sim_dir, config_path=config_path, results_dir=results,
        traces_dir=trace_dir, summary=summary, autotune=autotune,
        effective_placement=placement,
    )


def test_build_candidate_publishes_atomically_and_reuses_cache(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                            refine_fn=_fake_refine)
    assert not first.cache_hit
    assert (first.compile_dir / "artifacts.json").is_file()
    assert (first.compile_dir / "placement.json").is_file()
    # sidecar 的 pimir 路径已改写进发布目录，哈希仍对得上。
    sidecar = json.loads(first.placement_sidecar.read_text())
    entry = sidecar["operators"]["0"]
    assert Path(entry["pimir_path"]).is_file()
    assert file_sha256(Path(entry["pimir_path"])) == entry["pimir_sha256"]
    second = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                             refine_fn=_fake_refine)
    assert second.cache_hit
    assert second.compile_key == first.compile_key


def test_corrupt_cache_is_rebuilt(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                            refine_fn=_fake_refine)
    (first.compile_dir / "refined.ir").write_text("{}")
    second = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                             refine_fn=_fake_refine)
    assert not second.cache_hit
    # 坏 attempt 改名隔离，不再参与后续快速命中。
    assert not first.compile_dir.exists()
    quarantined = [path for path in first.compile_dir.parent.iterdir()
                   if path.name.startswith("invalid-attempt-")]
    assert quarantined


def test_failed_candidate_leaves_no_complete_cache(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)

    def boom(*args, **kwargs):
        raise RuntimeError("compile broke")

    with pytest.raises(RuntimeError):
        build_candidate(cfg, inputs, state, compile_fn=boom,
                        refine_fn=_fake_refine)
    key = compile_key(cfg, inputs, state)
    assert not (cfg.output.dir / "cache" / "compile" / key).exists()


def test_compile_and_sim_keys_split_by_field(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    faster = replace(
        state, profile=state.profile.with_fields(clock_period_ns=0.5))
    base_key = compile_key(cfg, inputs, state)
    assert compile_key(cfg, inputs, faster) == base_key, "时钟不该触发重编"
    assert sim_key(cfg, inputs, state, base_key) != sim_key(
        cfg, inputs, faster, base_key)
    smaller = replace(
        state, profile=state.profile.with_fields(cluster_memory_bytes=7_000_000_000))
    assert compile_key(cfg, inputs, smaller) != base_key, "容量变化必须重编"


def test_compile_key_changes_with_optimization_switches(tmp_path):
    """同一候选换优化开关必须重编，不能命中旧编译产物。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    base = compile_key(cfg, inputs, state)
    disabled = replace(cfg, optimization=PimOptimizations(
        constant_fold=False, cse=False, dce=False, loop_tiling=False))
    assert compile_key(disabled, inputs, state) != base


def test_compile_key_changes_with_candidate_optimization(tmp_path):
    """候选自带的开关取值也必须进入编译身份。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    base = compile_key(cfg, inputs, state)
    changed = replace(state, optimization=PimOptimizations(loop_tiling=False))
    assert compile_key(cfg, inputs, changed) != base


def test_verify_compiled_uses_the_candidate_optimization(tmp_path):
    """侧车必须按候选开关核对，而不是只看顶层配置。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    changed = replace(state, optimization=PimOptimizations(loop_tiling=False))
    artifacts = build_candidate(cfg, inputs, changed, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, changed, artifacts)
    with pytest.raises(ValueError, match="优化"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_another_optimization_set(tmp_path):
    """侧车记录的优化开关必须与本轮候选一致。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["compiler"]["hardware"]["optimizations"]["cse"] = False
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="优化"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_five_tampering_modes(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)

    sidecar = json.loads(artifacts.placement_sidecar.read_text())

    # 1. 画像摘要
    bad = json.loads(json.dumps(sidecar))
    bad["compiler"]["compile_profile_sha256"] = "0" * 64
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="画像"):
        verify_compiled(cfg, inputs, state, artifacts)

    # 2. 容量差 1 B
    bad = json.loads(json.dumps(sidecar))
    bad["compiler"]["physical_mram_bytes"] += 1
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="物理容量"):
        verify_compiled(cfg, inputs, state, artifacts)

    # 3. DPU 映射
    bad = json.loads(json.dumps(sidecar))
    bad["dpu_to_cluster"] = [[0, 0], [0, 1], [0, 2], [0, 5]]
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="映射"):
        verify_compiled(cfg, inputs, state, artifacts)

    # 4. PIM MLIR 哈希
    bad = json.loads(json.dumps(sidecar))
    Path(bad["operators"]["0"]["pimir_path"]).write_text("module { changed }")
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="pimir"):
        verify_compiled(cfg, inputs, state, artifacts)

    # 5. 精化 IR 的算子编号与 sidecar 不匹配
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    artifacts.refined_ir.write_text(json.dumps({"operators": [
        {"op_id": 0, "op_type": "GEMM"}, {"op_id": 7, "op_type": "SOFTMAX"}]}))
    with pytest.raises(ValueError, match="pimir_path"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_sidecar_with_another_compiler_target(tmp_path):
    """仿真前必须确认产物来自与画像同一编译目标。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["compiler"]["hardware"]["compiler_target"] = "pim:other"
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="编译目标"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_capacity_copies_from_another_budget(tmp_path):
    """可用容量与顶层 hardware 副本也必须与候选画像同源。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())

    bad = json.loads(json.dumps(sidecar))
    bad["compiler"]["usable_mram_bytes"] = 7_000_000_000
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="可用容量"):
        verify_compiled(cfg, inputs, state, artifacts)

    bad = json.loads(json.dumps(sidecar))
    bad["hardware"]["mram_bytes_per_dpu"] = 7_000_000_000
    bad["hardware"]["usable_mram_bytes"] = 7_000_000_000
    artifacts.placement_sidecar.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="顶层"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_pimir_attributes_from_another_request(tmp_path):
    """PIM MLIR 的 DPU/tasklet 预算必须与候选请求一致，不能只看哈希。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    pimir = Path(sidecar["operators"]["0"]["pimir_path"])
    text = pimir.read_text().replace(
        '"pim.num-tasklets" = 4 : i32', '"pim.num-tasklets" = 16 : i32')
    assert '"pim.num-tasklets" = 16' in text
    pimir.write_text(text)
    sidecar["operators"]["0"]["pimir_sha256"] = file_sha256(pimir)
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="num-tasklets|tasklet"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_sidecar_from_an_older_schema(tmp_path):
    """调优必须消费当前 schema 的产物，旧 schema 需重新导出。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["version"] = 4
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="schema|版本"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_an_a_route_without_tasklet_encoding(tmp_path):
    """A 路 PIM MLIR 缺 tasklet 编码时不能按 B 路豁免进入仿真。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    pimir = Path(sidecar["operators"]["0"]["pimir_path"])
    text = pimir.read_text().replace(
        '"pim.num-tasklets" = 4 : i32, ', "")
    assert '"pim.num-tasklets"' not in text
    pimir.write_text(text)
    sidecar["operators"]["0"]["pimir_sha256"] = file_sha256(pimir)
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="num-tasklets|预算属性"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_missing_pimir_route(tmp_path):
    """当前 schema 的每条 PIM MLIR 引用都必须声明编译路由。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["operators"]["0"].pop("pimir_route")
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="route|路由"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_shared_pimir_reference_without_route(tmp_path):
    """共享 PIM MLIR 的后续引用不能因为首条合法就跳过路由核对。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["operators"]["1"] = dict(sidecar["operators"]["0"])
    sidecar["operators"]["1"].pop("pimir_route")
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="route|路由"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_later_shared_reference_with_a_wrong_route(
        tmp_path):
    """同一文件的第二条引用必须按自己的算子类型重新核对路由。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["operators"]["1"] = dict(sidecar["operators"]["0"])
    sidecar["operators"]["1"]["pimir_route"] = "b"
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="路由|route"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_shared_reference_missing_a_budget(
        tmp_path):
    """共享文件的 A 路预算属性要对每条引用核对，不能由首条引用抵扣。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    pimir = Path(sidecar["operators"]["0"]["pimir_path"])
    text = pimir.read_text().replace('"pim.num-tasklets" = 4 : i32, ', "")
    assert '"pim.num-tasklets"' not in text
    pimir.write_text(text)
    digest = file_sha256(pimir)
    sidecar["operators"]["0"]["pimir_sha256"] = digest
    sidecar["operators"]["1"] = dict(sidecar["operators"]["0"])
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="num-tasklets|预算属性"):
        verify_compiled(cfg, inputs, state, artifacts)


def _republish_manifest(artifacts, inputs):
    """模拟真实编译：产物内容变化后按当前目录重写完成清单。"""
    from tune.report import manifest_from_dir
    from tune.steps import manifest_inputs

    manifest = manifest_from_dir(
        artifacts.compile_dir, artifacts.compile_key, manifest_inputs(inputs))
    (artifacts.compile_dir / "artifacts.json").write_text(
        json.dumps(manifest.to_json()))
    return replace(artifacts, manifest=manifest)


def test_verify_compiled_accepts_legal_shared_pimir_references(tmp_path):
    """同一 A 路文件被多个算子共享且属性齐全时正常通过。"""
    from tune.steps import _bind_refined_sidecar

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["operators"]["1"] = dict(sidecar["operators"]["0"])
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    # 这里模拟一份合法共享引用的编译产物，精化身份与完成清单随产物一起重绑。
    _bind_refined_sidecar(artifacts.refined_ir, artifacts.placement_sidecar,
                          inputs)
    artifacts = _republish_manifest(artifacts, inputs)
    verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_a_deleted_compiler_block(tmp_path):
    """删除整个 compiler 块不能只靠缓存或画像摘要通过调优校验。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar.pop("compiler")
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="compiler|工具链|硬件契约"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_empty_dpu_ids(tmp_path):
    """声明了 dpu_ids 就必须覆盖每台 DPU，空列表不能当未声明。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["compiler"]["strategy"]["dpu_ids"] = []
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="dpu_ids"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_accepts_generic_shared_pu_mapping(tmp_path):
    """通用 PIM 的共享 ClusterPU 映射是合法生产路径，正式校验不能误拒。"""
    from tune.steps import _bind_refined_sidecar

    cfg = _config(tmp_path, profile=GENERIC_PROFILE_PATH)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    shared = [[0, 0], [0, 0], [0, 1], [0, 1]]
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["dpu_to_cluster"] = shared
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    plan = json.loads(artifacts.partition_plan.read_text())
    plan["dpu_to_cluster"] = shared
    artifacts.partition_plan.write_text(json.dumps(plan))
    # 合法共享映射的产物在真实编译里会一起生成，这里同步重绑精化身份与清单。
    _bind_refined_sidecar(artifacts.refined_ir, artifacts.placement_sidecar,
                          inputs)
    artifacts = _republish_manifest(artifacts, inputs)
    verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_ceva_shared_pu_mapping(tmp_path):
    """CEVA 一对一约束不变：重复物理核仍要被正式校验拒绝。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    shared = [[0, 0], [0, 0], [0, 1], [0, 1]]
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["dpu_to_cluster"] = shared
    artifacts.placement_sidecar.write_text(json.dumps(sidecar))
    plan = json.loads(artifacts.partition_plan.read_text())
    plan["dpu_to_cluster"] = shared
    artifacts.partition_plan.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="一对一|映射"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_requires_a_verified_toolchain(tmp_path):
    """sidecar 的工具链身份必须与当前编译器和实际工具内容一致。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())

    def relabel(data):
        data["compiler"]["toolchain"]["graph_compiler"]["root"] = "/other/root"
        data["compiler"]["toolchain"]["tools"]["triton_opt"]["path"] = (
            "/other/triton-opt")
        return data

    def missing_block(data):
        data["compiler"].pop("toolchain")
        return data

    def bad_version(data):
        data["compiler"]["toolchain"]["version"] = 2
        return data

    def missing_graph_source(data):
        data["compiler"]["toolchain"]["graph_compiler"].pop("source_sha256")
        return data

    def wrong_graph_source(data):
        data["compiler"]["toolchain"]["graph_compiler"]["source_sha256"] = (
            "0" * 64)
        return data

    def missing_fingerprint(data):
        data["compiler"]["toolchain"].pop("compiler_fingerprint")
        return data

    def zero_fingerprint(data):
        data["compiler"]["toolchain"]["compiler_fingerprint"] = "0" * 16
        return data

    def null_tool_digest(data):
        data["compiler"]["toolchain"]["tools"]["triton_opt"]["sha256"] = None
        return data

    def zero_tool_digest(data):
        data["compiler"]["toolchain"]["tools"]["triton_opt"]["sha256"] = (
            "0" * 64)
        return data

    def wrong_tool_digest(data):
        data["compiler"]["toolchain"]["tools"]["triton_opt"]["sha256"] = (
            "1" * 64)
        return data

    def missing_tool_entry(data):
        data["compiler"]["toolchain"]["tools"].pop("triton_opt")
        return data

    # 路径可以换（产物可搬运），内容身份不能任意改写。
    artifacts.placement_sidecar.write_text(
        json.dumps(relabel(json.loads(json.dumps(sidecar)))))
    artifacts = _republish_manifest(artifacts, inputs)
    verify_compiled(cfg, inputs, state, artifacts)

    for name, mutate in (
        ("missing_block", missing_block),
        ("bad_version", bad_version),
        ("missing_graph_source", missing_graph_source),
        ("wrong_graph_source", wrong_graph_source),
        ("missing_fingerprint", missing_fingerprint),
        ("zero_fingerprint", zero_fingerprint),
        ("null_tool_digest", null_tool_digest),
        ("zero_tool_digest", zero_tool_digest),
        ("wrong_tool_digest", wrong_tool_digest),
        ("missing_tool_entry", missing_tool_entry),
    ):
        bad = mutate(json.loads(json.dumps(sidecar)))
        artifacts.placement_sidecar.write_text(json.dumps(bad))
        try:
            verify_compiled(cfg, inputs, state,
                            _republish_manifest(artifacts, inputs))
        except ValueError as exc:
            assert "工具链" in str(exc), f"{name}: {exc}"
        else:
            raise AssertionError(f"{name}: 缺失或伪造的工具链身份未被拒绝")


def test_evaluate_candidate_isolates_failures(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)

    def boom(*args, **kwargs):
        raise RuntimeError("compile broke")

    failed = evaluate_candidate(
        cfg, inputs, state, sequence=0, compile_fn=boom,
        refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert isinstance(failed, RoundRecord)
    assert failed.status == "error" and failed.score is None
    assert "compile broke" in failed.failure

    ok = evaluate_candidate(
        cfg, inputs, state, sequence=1, compile_fn=_fake_compile,
        refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert ok.status == "success"
    assert ok.score == 1.5
    assert ok.result_maturity == "model_prediction"


def test_fast_mode_reuses_successful_sim(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = evaluate_candidate(cfg, inputs, state, sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    assert first.status == "success"

    calls = []

    def counting_sim(*args, **kwargs):
        calls.append(1)
        return _fake_sim(*args, **kwargs)

    second = evaluate_candidate(cfg, inputs, state, sequence=1,
                                compile_fn=_fake_compile, refine_fn=_fake_refine,
                                sim_fn=counting_sim)
    assert calls == [], "快速档应复用同一 sim_key 的成功结果"
    assert second.score == first.score

    strict = evaluate_candidate(cfg, inputs, state, sequence=2,
                                force_compile=True,
                                compile_fn=_fake_compile, refine_fn=_fake_refine,
                                sim_fn=counting_sim)
    assert calls, "strict 必须新开尝试目录并重跑"
    assert strict.score == first.score


def test_autotune_objective_direction_follows_metric(tmp_path):
    from dataclasses import replace

    from tune.steps import _autotune_overrides

    cfg = _config(tmp_path)
    placement = replace(cfg.placement, enabled=True,
                        objective_metric="total_time_s")
    payload = _autotune_overrides(replace(cfg, placement=placement), tmp_path)
    assert payload["objective_direction"] == "minimize"
    efficiency = replace(placement, objective_metric="system_efficiency_utility")
    payload = _autotune_overrides(replace(cfg, placement=efficiency), tmp_path)
    assert payload["objective_direction"] == "maximize"


def test_explicit_move_weights_disable_the_other_families(tmp_path):
    """只配一类权重时必须把其余七类显式置 0，不能沿用基配置默认值。"""
    from tune.config import MOVES
    from tune.steps import _autotune_overrides

    cfg = _config(tmp_path)
    placement = replace(cfg.placement, moves=replace(
        cfg.placement.moves, weights={"op_relocate": 1.0}))
    payload = _autotune_overrides(replace(cfg, placement=placement), tmp_path)
    weights = payload["moves"]["weights"]
    assert set(weights) == set(MOVES)
    assert weights["op_relocate"] == 1.0
    assert all(value == 0.0 for name, value in weights.items()
               if name != "op_relocate")


def test_build_sim_resolves_explicit_move_weights_exactly(tmp_path):
    """生效 GeneSim 配置不能把基配置里未指定的移动权重合并回来。"""
    from tune._genesim import genesim_config_class
    from tune.steps import _autotune_overrides, build_sim

    cfg = _config(tmp_path)
    placement = replace(
        cfg.placement, enabled=True,
        moves=replace(cfg.placement.moves, weights={"op_relocate": 1.0}))
    cfg = replace(cfg, placement=placement)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    config_path, _ = build_sim(cfg, inputs, state, artifacts,
                               cfg.output.dir / "sim" / "move-weights")
    effective = genesim_config_class()(str(config_path))
    weights = effective.get("autotune.moves.weights")
    assert weights["op_relocate"] == 1.0
    assert all(value == 0.0 for name, value in weights.items()
               if name != "op_relocate")


def test_strict_compiles_keep_independent_attempts(tmp_path):
    """严格模式每次重编写入独立 attempt，旧产物与引用必须保留。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    fast = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                           refine_fn=_fake_refine)
    assert not fast.cache_hit
    marker = fast.compile_dir / "marker.txt"
    marker.write_text("first")

    first_strict = build_candidate(
        cfg, inputs, state, force_compile=True,
        compile_fn=_fake_compile, refine_fn=_fake_refine)
    second_strict = build_candidate(
        cfg, inputs, state, force_compile=True,
        compile_fn=_fake_compile, refine_fn=_fake_refine)
    assert first_strict.compile_key == second_strict.compile_key
    assert first_strict.compile_dir != second_strict.compile_dir
    assert first_strict.compile_dir != fast.compile_dir
    # 旧产物不被覆盖或删除，且仍可完整校验。
    assert marker.read_text() == "first"
    for attempt in (fast, first_strict, second_strict):
        assert (attempt.compile_dir / "artifacts.json").is_file()
        assert (attempt.compile_dir / "placement.json").is_file()
    # 快速档仍可命中任一有效 attempt。
    reused = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                             refine_fn=_fake_refine)
    assert reused.cache_hit


def test_sweep_strict_mode_forces_recompile(tmp_path, monkeypatch):
    from tune import sweep

    cfg = _config(tmp_path)
    strict = replace(cfg, cache=replace(cfg.cache, mode="strict"))
    seen = []

    def fake_evaluate(cfg_arg, inputs, state, **kwargs):
        seen.append(kwargs.get("force_compile"))
        return RoundRecord(
            sequence=0, candidate_id="c", compile_key="k", sim_key="s",
            cache=cfg_arg.cache.mode, status="success",
            result_maturity="model_prediction", state=state, score=1.0,
            metrics={"total_time_s": 1.0})

    monkeypatch.setattr(sweep, "evaluate_candidate", fake_evaluate)
    monkeypatch.setattr(sweep, "model_dims", lambda path: SimpleNamespace(
        num_heads=4, num_kv_heads=2, intermediate_size=32, vocab_size=64,
        num_layers=4, hidden_size=16))
    inputs = _inputs(strict)
    state = _state(strict)
    sweep_cfg = replace(strict, sweep=replace(
        strict.sweep, fields={"clock_period_ns": (0.63, 0.5)},
        pareto=replace(strict.sweep.pareto,
                       resource="total_physical_memory_bytes")))
    sweep.run_sweep(sweep_cfg, inputs, state)
    assert seen and all(flag is True for flag in seen)

    seen.clear()
    fast_cfg = replace(sweep_cfg, cache=replace(strict.cache, mode="fast"))
    sweep.run_sweep(fast_cfg, inputs, state)
    assert seen and all(flag is False for flag in seen)


def test_autotune_overrides_pass_proposal_budget(tmp_path):
    from tune.steps import _autotune_overrides

    cfg = _config(tmp_path)
    placement = replace(cfg.placement, enabled=True, max_proposals=3)
    payload = _autotune_overrides(replace(cfg, placement=placement), tmp_path)
    assert payload["max_proposals"] == 3


def test_build_sim_writes_strict_config_and_contract(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    sim_dir = cfg.output.dir / "sim" / "unit"
    config_path, contract_path = build_sim(cfg, inputs, state, artifacts, sim_dir)
    config = yaml.safe_load(config_path.read_text())
    assert config["scheduler"]["require_compiler_pimir"] is True
    assert config["scheduler"]["use_compiler_local_shapes"] is True
    assert Path(config["scheduler"]["compiler_placement_file"]).is_file()
    assert Path(config["model"]["ir_path"]) == artifacts.refined_ir
    assert Path(config["output"]["results_dir"]) == sim_dir / "results"
    assert Path(config["output"]["pim_traces_dir"]) == sim_dir / "pim_traces"
    assert config["autotune"]["enabled"] is False
    assert config["autotune"]["save_effective_placement"] is True
    assert Path(config["scheduler"]["effective_placement_file"]) == (
        sim_dir / "effective_placement.json")
    assert config["pim"]["clock_period_ns"] == state.profile.clock_period_ns
    contract = json.loads(contract_path.read_text())
    assert contract["profile_sha256"] == state.profile.profile_sha256()
    assert contract["expected"]["system.num_pim_devices"] == 1
    assert contract["expected"]["ceva.local_memory_bytes"] == 8_000_000_000


# ---- 评审 round01：仿真完成协议、结果有效性与完整画像契约 ----


def _custom_sim(summary, *, traces=True):
    # 默认轨迹是一个 4 token 提示加 2 token 输出的单请求。
    summary = {"processed_tokens": 6, "completed_requests": 1, **summary}

    def simulate(cfg, sim_dir, config_path):
        sim_dir.mkdir(parents=True, exist_ok=True)
        results = sim_dir / "results"
        results.mkdir(exist_ok=True)
        (results / "summary.json").write_text(json.dumps(summary))
        (results / "autotune_metrics.json").write_text(
            json.dumps(_fixed_autotune()))
        trace_dir = sim_dir / "pim_traces"
        trace_dir.mkdir(exist_ok=True)
        if traces:
            (trace_dir / "op_0_GEMM.pim_trace").write_text(json.dumps(
                {"op_id": 0, "compile_signature": {"trace_source": "pimir"}}))
        placement = _write_effective_placement(sim_dir)
        return SimOutput(
            sim_dir=sim_dir, config_path=config_path, results_dir=results,
            traces_dir=trace_dir, summary=summary,
            autotune=_fixed_autotune(), effective_placement=placement)
    return simulate


def test_contract_covers_every_profile_override_leaf(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    _, contract_path = build_sim(cfg, inputs, state, artifacts,
                                 cfg.output.dir / "sim" / "contract")
    expected = json.loads(contract_path.read_text())["expected"]
    profile = state.profile
    assert expected["pim.pim_trace_cycle_costs.MAC"] == profile.cycle_costs["MAC"]
    for key, value in profile.zone_cycle_factors.items():
        assert expected[f"pim.zone_cycle_factors.{key}"] == value
    assert expected["pim.tensor_pu_config.multipliers"] == profile.multipliers
    assert expected["comm_config.links"] == [
        link.to_json() for link in profile.links]
    assert expected["comm_config.host_relay_overhead_ns"] == (
        profile.host_relay_overhead_ns)
    for key, value in profile.cpu_model.items():
        assert expected[f"cpu_model.{key}"] == value
    assert expected["ceva.parameter_sources"] == dict(profile.parameter_sources)
    assert expected["ceva.local_memory_bytes"] == profile.ceva.local_memory_bytes


def test_incomplete_sim_cache_is_not_reused(tmp_path):
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    attempt = cfg.output.dir / "sim" / "key" / "attempt-a"
    (attempt / "results").mkdir(parents=True)
    (attempt / "results" / "summary.json").write_text(json.dumps({
        "total_time_s": 1.0, "result_producer": "pipeline_simulator"}))
    assert _find_cached_sim(cfg, _inputs(cfg), "key") is None


def test_tampered_cached_result_is_not_reused(tmp_path):
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    first = evaluate_candidate(cfg, _inputs(cfg), _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    assert first.status == "success"
    completion = Path(first.paths["sim_dir"]) / "completion.json"
    assert completion.is_file()
    assert json.loads(completion.read_text())["sim_key"] == first.sim_key

    summary = Path(first.paths["sim_dir"]) / "results" / "summary.json"
    summary.write_text(json.dumps({
        "total_time_s": 99.0, "throughput_tokens_per_s": 100.0,
        "result_producer": "pipeline_simulator"}))
    assert _find_cached_sim(cfg, _inputs(cfg), first.sim_key) is None


def test_missing_result_producer_is_rejected(tmp_path):
    cfg = _config(tmp_path)
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=_custom_sim({"total_time_s": 1.0}))
    assert record.status == "error"
    assert "result_producer" in record.failure


def test_non_finite_or_non_positive_delay_is_rejected(tmp_path):
    for value in (float("inf"), float("nan"), 0.0, -1.0):
        case_dir = tmp_path / f"case-{str(value).replace('/', '_')}"
        case_dir.mkdir()
        cfg = _config(case_dir)
        record = evaluate_candidate(
            cfg, _inputs(cfg), _state(cfg), sequence=0,
            compile_fn=_fake_compile, refine_fn=_fake_refine,
            sim_fn=_custom_sim({"total_time_s": value,
                                "result_producer": "pipeline_simulator"}))
        assert record.status == "error", value
        assert record.score is None


def test_failed_sim_keeps_compile_identity_and_artifact_paths(tmp_path):
    cfg = _config(tmp_path)
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=_custom_sim({"total_time_s": float("inf"),
                            "result_producer": "pipeline_simulator"}))
    assert record.status == "error"
    assert record.compile_key
    assert record.paths.get("compile_dir")


def test_round_record_carries_full_efficiency_and_capacity_metrics(tmp_path):
    cfg = _config(tmp_path)
    summary = {
        "total_time_s": 1.5,
        "throughput_tokens_per_s": 100.0,
        "result_producer": "pipeline_simulator",
        "pim_native_backend": NATIVE_TRACE_FALLBACK,
        "compute_efficiency": 0.5,
        "communication_efficiency": 0.25,
        "memory_efficiency": 0.125,
        "communication_time_s": 0.2,
        "memory_service_time_s": 0.3,
        "pim_capacity_max_utilization": 0.75,
        "pim_capacity_fits": True,
        "pim_capacity_max_resident_bytes": 1234,
        "efficiency_metrics": {
            "metrics_schema_version": 1,
            "compute": {"system_mfu": 0.5},
            "communication": {"efficiency": 0.25},
            "memory": {"efficiency": 0.125},
        },
    }
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=_custom_sim(summary))
    assert record.status == "success"
    for key in ("total_time_s", "throughput_tokens_per_s",
                "compute_efficiency", "communication_efficiency",
                "memory_efficiency", "communication_time_s",
                "memory_service_time_s", "pim_capacity_max_utilization",
                "pim_capacity_fits", "pim_capacity_max_resident_bytes"):
        assert key in record.metrics, key
    assert record.metrics["compute_efficiency"] == 0.5
    assert record.metrics["communication_efficiency"] == 0.25
    assert record.metrics["memory_efficiency"] == 0.125
    assert record.metrics["efficiency_metrics"]["metrics_schema_version"] == 1
    assert record.compile_cache_hit is False
    assert record.sim_cache_hit is False


def test_sim_key_includes_genesim_environment(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    base = compile_key(cfg, inputs, state)
    first = sim_key(cfg, inputs, state, base)
    monkeypatch.setenv("GENESIM_PIM_CLOCK_PERIOD_NS", "0.5")
    second = sim_key(cfg, inputs, state, base)
    assert first != second, "环境覆盖会改变生效配置，必须使仿真键失效"


def test_sim_key_includes_predictor_checkpoint_content(tmp_path, monkeypatch):
    from dataclasses import replace

    cfg = _config(tmp_path)
    placement = replace(cfg.placement, evaluation_backend="predictor")
    cfg = replace(cfg, placement=placement)
    inputs = _inputs(cfg)
    state = _state(cfg)
    base = compile_key(cfg, inputs, state)
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint-one")
    monkeypatch.setenv("GENESIM_PREDICTOR_CHECKPOINT_PATH", str(checkpoint))
    first = sim_key(cfg, inputs, state, base)
    checkpoint.write_bytes(b"checkpoint-two")
    second = sim_key(cfg, inputs, state, base)
    assert first != second, "换检查点内容必须使仿真键失效"


def test_build_sim_separates_replay_input_from_output(tmp_path):
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    replay = tmp_path / "saved_effective_placement.json"
    replay.write_text("{}")
    sim_dir = cfg.output.dir / "sim" / "replay"
    config_path, _ = build_sim(cfg, inputs, state, artifacts, sim_dir,
                               replay_placement=replay)
    config = yaml.safe_load(config_path.read_text())
    assert config["scheduler"]["replay_placement_file"] == str(replay)
    assert Path(config["scheduler"]["effective_placement_file"]) == (
        sim_dir / "effective_placement.json")
    assert config["autotune"]["enabled"] is False
    assert config["autotune"]["save_effective_placement"] is True


def test_generated_contract_passes_genesim_validation_and_detects_env(
        tmp_path, monkeypatch):
    """交叉仓库：同一份契约在 GeneSim 侧逐项核对，环境覆盖也逃不掉。"""
    from tune._genesim import _ensure_genesim_path, genesim_config_class

    _ensure_genesim_path()
    import hardware_profile as genesim_hardware_profile

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    config_path, _ = build_sim(cfg, inputs, state, artifacts,
                               cfg.output.dir / "sim" / "contract-cross")
    config = genesim_config_class()(str(config_path))
    assert genesim_hardware_profile.validate_hardware_contract(config) == []
    monkeypatch.setenv("GENESIM_PIM_PIM_TRACE_CYCLE_COSTS_MAC", "999")
    errors = genesim_hardware_profile.validate_hardware_contract(
        genesim_config_class()(str(config_path)))
    assert any("pim_trace_cycle_costs" in error for error in errors), errors


def test_generic_profile_contract_also_passes_genesim(tmp_path):
    from dataclasses import replace

    from contracts.hardware_profile import load_profile
    from tune._genesim import _ensure_genesim_path, genesim_config_class

    _ensure_genesim_path()
    import hardware_profile as genesim_hardware_profile

    generic_path = REPO_ROOT / "contracts" / "profiles" / "generic.json"
    cfg = replace(_config(tmp_path), profile_path=generic_path,
                  profile=load_profile(generic_path))
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    config_path, _ = build_sim(cfg, inputs, state, artifacts,
                               cfg.output.dir / "sim" / "generic")
    config = genesim_config_class()(str(config_path))
    assert genesim_hardware_profile.validate_hardware_contract(config) == []


def test_build_sim_contract_binds_the_compile_request(tmp_path):
    """画像契约必须携带同一候选派生的编译请求绑定，不能只写期望值。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    config_path, contract_path = build_sim(
        cfg, inputs, state, artifacts, cfg.output.dir / "sim" / "binding")
    contract = json.loads(contract_path.read_text())
    hardware = state.profile.to_pim_hardware_config(
        state.strategy.num_dpus, state.op.num_tasklets,
        optimizations=state.resolved_optimization(cfg))
    assert contract["schema_version"] == 3
    binding = contract["compile"]
    assert binding["compile_profile_sha256"] == (
        state.profile.compile_profile_sha256())
    assert binding["hardware"] == {
        "num_dpus": hardware.num_dpus,
        "num_tasklets": hardware.num_tasklets,
        "mram_bytes_per_dpu": hardware.mram_bytes_per_dpu,
        "wram_bytes_per_dpu": hardware.wram_bytes_per_dpu,
        "dma_align": hardware.dma_align,
        "compiler_target": hardware.compiler_target,
        "optimizations": hardware.optimizations.to_payload(),
    }
    assert binding["physical_mram_bytes"] == state.profile.cluster_memory_bytes
    assert binding["reserved_mram_bytes"] == state.profile.sys_reserve_bytes
    assert binding["usable_mram_bytes"] == (
        state.profile.cluster_memory_bytes - state.profile.sys_reserve_bytes)
    # schema 3 起绑定完整候选编译身份：编译键、图策略、计划映射与产物摘要。
    assert binding["compile_key"] == artifacts.compile_key
    assert binding["strategy"] == {
        "name": "tp2_pp2",
        "num_dpus": 4,
        "num_stages": 2,
        "tp_width": 2,
        "dpu_ids": [0, 1, 2, 3],
    }
    plan = json.loads(artifacts.partition_plan.read_text())
    assert binding["dpu_to_cluster"] == plan["dpu_to_cluster"]
    assert binding["artifacts"] == {
        "refined_ir_sha256": file_sha256(artifacts.refined_ir),
        "placement_sidecar_sha256": file_sha256(artifacts.placement_sidecar),
        "trace_sha256": file_sha256(inputs.trace_path),
        "trace_num_requests": len(inputs.trace.requests),
    }
    # 契约里的绑定必须与同一次构建的 sidecar 一致，否则新产物启动即失败。
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    assert sidecar["compiler"]["hardware"] == binding["hardware"]
    assert sidecar["compiler"]["compile_profile_sha256"] == (
        binding["compile_profile_sha256"])
    assert sidecar["compiler"]["strategy"] == binding["strategy"]
    assert sidecar["dpu_to_cluster"] == binding["dpu_to_cluster"]


def test_verify_compiled_rejects_a_sidecar_from_another_graph_strategy(tmp_path):
    """同硬件、同 DPU 集合但 TP/PP 不同的产物不能冒充当前候选。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state_a = _state(cfg)
    state_b = TuneState(StrategyChoice(4, 4), OpKnobs(4), cfg.profile)
    artifacts_a = build_candidate(cfg, inputs, state_a,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    artifacts_b = build_candidate(cfg, inputs, state_b,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state_a, artifacts_a)
    mixed = replace(artifacts_a, placement_sidecar=artifacts_b.placement_sidecar)
    with pytest.raises(ValueError, match="策略"):
        verify_compiled(cfg, inputs, state_a, mixed)
    # 策略的 DPU 集合同样属于候选身份，长度相同但编号不同也不能放行。
    sidecar = json.loads(artifacts_a.placement_sidecar.read_text())
    sidecar["compiler"]["strategy"]["dpu_ids"] = [1, 2, 3, 4]
    artifacts_a.placement_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="策略"):
        verify_compiled(cfg, inputs, state_a, artifacts_a)


def test_verify_compiled_rejects_a_plan_from_another_graph_strategy(tmp_path):
    """计划文件的段数必须与当前候选状态一致。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state_a = _state(cfg)
    state_b = TuneState(StrategyChoice(4, 4), OpKnobs(4), cfg.profile)
    artifacts_a = build_candidate(cfg, inputs, state_a,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    artifacts_b = build_candidate(cfg, inputs, state_b,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    mixed = replace(artifacts_a, partition_plan=artifacts_b.partition_plan)
    with pytest.raises(ValueError, match="partition_plan"):
        verify_compiled(cfg, inputs, state_a, mixed)


def test_refine_contract_rejects_an_inconsistent_strategy_shape(tmp_path):
    """独立精化入口也要核对策略段数/TP 宽度与 DPU 预算自洽。"""
    from genesim_bridge.refine_profile import verify_placement_contract

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    sidecar["compiler"]["strategy"]["name"] = "tp2_pp4"
    sidecar["compiler"]["strategy"]["num_stages"] = 4
    sidecar["compiler"]["strategy"]["tp_width"] = 2
    with pytest.raises(ValueError, match="tp_width"):
        verify_placement_contract(
            cfg.profile, sidecar, placement_path=artifacts.placement_sidecar,
            search_dirs=(artifacts.compile_dir / "pimir",))


def test_verify_compiled_rejects_a_refined_ir_from_another_candidate(tmp_path):
    """只交换精化 IR：它记录的放置 sidecar 身份与当前 sidecar 不符。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state_a = _state(cfg)
    state_b = TuneState(StrategyChoice(4, 4), OpKnobs(4), cfg.profile)
    artifacts_a = build_candidate(cfg, inputs, state_a,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    artifacts_b = build_candidate(cfg, inputs, state_b,
                                  compile_fn=_fake_compile,
                                  refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state_a, artifacts_a)
    mixed = replace(artifacts_a, refined_ir=artifacts_b.refined_ir)
    with pytest.raises(ValueError, match="精化 IR"):
        verify_compiled(cfg, inputs, state_a, mixed)


# ---- 评审 round21：候选输入身份必须绑定编译产物与跨仓契约 ----

def test_compile_manifest_records_the_full_input_identity(tmp_path):
    """产物清单保存模型完整指纹、原始 IR 与轨迹窗口，不能只存单文件哈希。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    recorded = {entry.role: entry.sha256 for entry in artifacts.manifest.inputs}
    assert recorded == {
        "model": inputs.model_sha256,
        "ir": inputs.ir_sha256,
        "trace": inputs.trace_sha256,
    }
    # 模型角色记录的是 config 加全部权重的指纹，不是单个 config.json 的哈希。
    assert recorded["model"] != file_sha256(inputs.model_dir / "config.json")


def test_verify_compiled_rejects_artifacts_from_another_model(tmp_path):
    """同策略、同硬件但模型内容不同的输入不能复用旧编译产物。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    other = replace(inputs, model_sha256="a" * 64)
    with pytest.raises(ValueError, match="编译键"):
        verify_compiled(cfg, other, state, artifacts)
    with pytest.raises(ValueError, match="编译键"):
        build_sim(cfg, other, state, artifacts,
                  cfg.output.dir / "sim" / "foreign-model")


def test_verify_compiled_rejects_artifacts_from_another_source_ir(tmp_path):
    """原始 IR 内容变化后，旧产物不能在未重编的情况下进入仿真。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    other = replace(inputs, ir_sha256="b" * 64)
    with pytest.raises(ValueError, match="编译键"):
        verify_compiled(cfg, other, state, artifacts)


def test_verify_compiled_rejects_artifacts_from_another_trace_window(tmp_path):
    """请求输出长度变化会改变容量与仿真键，旧产物必须重编。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    other = replace(
        inputs,
        trace=TraceSnapshot(source="memory", requests=(
            RequestBound(prompt_tokens=4, output_tokens=3, initial_kv=0),)),
        trace_sha256="c" * 64, max_seq=inputs.max_seq + 1)
    with pytest.raises(ValueError, match="编译键"):
        verify_compiled(cfg, other, state, artifacts)
    with pytest.raises(ValueError, match="编译键"):
        build_sim(cfg, other, state, artifacts,
                  cfg.output.dir / "sim" / "foreign-trace")


def test_build_sim_rejects_a_foreign_compile_key(tmp_path):
    """候选对象里保存的旧编译键不能自证；重算不符时不得生成仿真契约。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    mixed = replace(artifacts, compile_key="d" * 64)
    with pytest.raises(ValueError, match="编译键"):
        build_sim(cfg, inputs, state, mixed, cfg.output.dir / "sim" / "foreign")


def test_verify_compiled_rejects_a_foreign_manifest_compile_key(tmp_path):
    """清单里的编译键也必须与当前输入重算结果一致。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    mixed = replace(
        artifacts,
        manifest=replace(artifacts.manifest, compile_key="e" * 64))
    with pytest.raises(ValueError, match="编译键"):
        verify_compiled(cfg, inputs, state, mixed)


def test_verify_compiled_rejects_a_manifest_with_foreign_input_identity(tmp_path):
    """清单角色被错误迁移后，即使目录内容完整也不能作为当前候选。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    foreign = tuple(
        replace(entry, sha256="f" * 64) if entry.role == "model" else entry
        for entry in artifacts.manifest.inputs)
    mixed = replace(
        artifacts, manifest=replace(artifacts.manifest, inputs=foreign))
    with pytest.raises(ValueError, match="产物清单"):
        verify_compiled(cfg, inputs, state, mixed)


def test_fast_cache_ignores_an_attempt_with_foreign_input_identity(tmp_path):
    """快速缓存只认与当前输入逐项一致的清单，错误迁移目录必须重建。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                            refine_fn=_fake_refine)
    manifest_path = first.compile_dir / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["inputs"]:
        if entry["role"] == "trace":
            entry["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    second = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                             refine_fn=_fake_refine)
    assert second.cache_hit is False
    assert second.compile_dir != first.compile_dir
    verify_compiled(cfg, inputs, state, second)


def test_fast_cache_rejects_a_manifest_without_input_identity(tmp_path):
    """升级前的清单没有输入角色，同等条件下不能继续命中快速缓存。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                            refine_fn=_fake_refine)
    manifest_path = first.compile_dir / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = []
    manifest_path.write_text(json.dumps(manifest))
    second = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                             refine_fn=_fake_refine)
    assert second.cache_hit is False
    assert second.compile_dir != first.compile_dir


def test_generated_contract_binds_the_request_trace(tmp_path):
    """schema 3 契约写入实际消费轨迹的内容摘要与请求窗口。"""
    from tune._genesim import _ensure_genesim_path, genesim_config_class

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    config_path, contract_path = build_sim(
        cfg, inputs, state, artifacts, cfg.output.dir / "sim" / "trace-binding")
    artifacts_block = json.loads(contract_path.read_text())[
        "compile"]["artifacts"]
    assert artifacts_block["trace_sha256"] == file_sha256(inputs.trace_path)
    assert artifacts_block["trace_num_requests"] == len(inputs.trace.requests)

    _ensure_genesim_path()
    import hardware_profile as genesim_hardware_profile

    config = genesim_config_class()(str(config_path))
    hashes = genesim_hardware_profile.configured_artifact_hashes(config)
    assert hashes["trace_sha256"] == file_sha256(inputs.trace_path)
    assert hashes["trace_num_requests"] == len(inputs.trace.requests)


# ---- 评审 round22：输入快照必须在编译、仿真、发布与缓存全生命周期不变 ----

def _rewrite_first_output(inputs, output_length=5):
    """原地改写已发布轨迹副本的第一个请求输出长度。"""
    payload = json.loads(inputs.trace_path.read_text())
    payload["requests"][0]["output_length"] = output_length
    inputs.trace_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))


def test_verify_compiled_rejects_a_trace_rewritten_after_prepare(tmp_path):
    """同一输入对象在 prepare 后被改写轨迹时，旧摘要不能继续放行。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    _rewrite_first_output(inputs)
    with pytest.raises(ValueError, match="输入快照"):
        verify_compiled(cfg, inputs, state, artifacts)
    with pytest.raises(ValueError, match="输入快照"):
        build_sim(cfg, inputs, state, artifacts,
                  cfg.output.dir / "sim" / "rewritten-trace")


def test_verify_compiled_rejects_a_model_rewritten_after_prepare(tmp_path):
    """模型权重在 prepare 后被改写时，编译产物不能再匹配旧指纹。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    (inputs.model_dir / "config.json").write_text('{"a": 2}')
    with pytest.raises(ValueError, match="模型"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_verify_compiled_rejects_an_ir_rewritten_after_prepare(tmp_path):
    """已发布原始 IR 被原地改写时，精化 IR 的派生身份必须失败。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    verify_compiled(cfg, inputs, state, artifacts)
    inputs.ir_path.write_text('{"operators": [{"op_id": 0}]}')
    with pytest.raises(ValueError, match="IR"):
        verify_compiled(cfg, inputs, state, artifacts)


def test_build_candidate_rejects_a_trace_changed_during_compile(tmp_path):
    """编译窗口内轨迹被改写时不得发布为该输入的编译产物。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)

    def mutating_compile(*args, **kwargs):
        result = _fake_compile(*args, **kwargs)
        _rewrite_first_output(inputs)
        return result

    with pytest.raises(ValueError, match="输入快照"):
        build_candidate(cfg, inputs, state, compile_fn=mutating_compile,
                        refine_fn=_fake_refine)
    key = compile_key(cfg, inputs, state)
    cache_root = cfg.output.dir / "cache" / "compile" / key
    assert not list(cache_root.glob("attempt-*"))


def test_build_sim_binds_contract_manifest_and_key_to_one_trace(tmp_path):
    """画像契约、manifest 与 sim_key 必须消费同一份已复核轨迹。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    key = sim_key(cfg, inputs, state, artifacts.compile_key)
    sim_dir = cfg.output.dir / "sim" / key / "attempt-a"
    _, contract_path = build_sim(cfg, inputs, state, artifacts, sim_dir)
    contract = json.loads(contract_path.read_text())
    manifest = json.loads((sim_dir / "manifest.json").read_text())
    assert contract["compile"]["artifacts"]["trace_sha256"] == inputs.trace_sha256
    assert contract["compile"]["artifacts"]["trace_num_requests"] == 1
    assert manifest["input_sha256"]["trace"] == inputs.trace_sha256
    assert manifest["sim_key"] == key


def test_publish_completion_rejects_a_trace_rewritten_after_build_sim(tmp_path):
    """契约写出后轨迹被原地改写时，完成清单不得发布。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    _rewrite_first_output(inputs)
    with pytest.raises(ValueError, match="输入快照"):
        _publish_sim_complete(
            cfg, inputs, sim_dir, key, compile_key, output, "fixed")
    assert not (sim_dir / "completion.json").exists()


def test_publish_completion_rejects_a_processed_token_mismatch(tmp_path):
    """仿真摘要实际处理的 token 数必须与请求窗口一致。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    summary = dict(output.summary)
    summary["processed_tokens"] = int(summary["processed_tokens"]) + 1
    (sim_dir / "results" / "summary.json").write_text(json.dumps(summary))
    output = replace(output, summary=summary)
    with pytest.raises(RuntimeError, match="token"):
        _publish_sim_complete(
            cfg, inputs, sim_dir, key, compile_key, output, "fixed")
    assert not (sim_dir / "completion.json").exists()


def test_publish_completion_rejects_a_contract_bound_to_another_trace(tmp_path):
    """契约登记另一条轨迹时，即使 manifest 摘要同步也不能发布。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    contract_path = sim_dir / "hardware_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["compile"]["artifacts"]["trace_sha256"] = "a" * 64
    contract_path.write_text(json.dumps(contract))
    manifest_path = sim_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hardware_contract_sha256"] = file_sha256(contract_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="轨迹"):
        _publish_sim_complete(
            cfg, inputs, sim_dir, key, compile_key, output, "fixed")


@pytest.mark.parametrize("role", ["model", "ir", "trace"])
def test_publish_completion_rejects_a_manifest_from_another_input(
        tmp_path, role):
    """manifest 的三个输入角色都必须与当前已复核输入逐项一致。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    manifest_path = sim_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"][role] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="输入"):
        _publish_sim_complete(
            cfg, inputs, sim_dir, key, compile_key, output, "fixed")


def test_fast_cache_rejects_a_self_consistent_directory_for_another_trace(
        tmp_path):
    """目录内部完全自洽但绑定了另一条轨迹时，快速缓存不得命中。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    first = evaluate_candidate(cfg, inputs, _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    assert first.status == "success", first.failure
    attempt = Path(first.paths["sim_dir"])
    contract_path = attempt / "hardware_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["compile"]["artifacts"]["trace_sha256"] = "a" * 64
    contract_path.write_text(json.dumps(contract))
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"]["trace"] = "a" * 64
    manifest["hardware_contract_sha256"] = file_sha256(contract_path)
    manifest_path.write_text(json.dumps(manifest))
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["manifest.json"] = file_sha256(manifest_path)
    completion["files"]["hardware_contract.json"] = file_sha256(contract_path)
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, inputs, first.sim_key) is None


def test_fast_cache_rejects_a_trace_rewritten_after_completion(tmp_path):
    """完成清单存在但当前轨迹已改写时，快速缓存不得返回旧结果。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    first = evaluate_candidate(cfg, inputs, _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    assert first.status == "success", first.failure
    _rewrite_first_output(inputs)
    with pytest.raises(ValueError, match="输入快照"):
        _find_cached_sim(cfg, inputs, first.sim_key)


def test_input_snapshot_accepts_a_same_content_trace_copy(tmp_path):
    """同内容搬迁不是输入变化，合法路径不能被误拒。"""
    from tune.inputs import verify_input_snapshot

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    copy = tmp_path / "trace-copy.json"
    copy.write_bytes(inputs.trace_path.read_bytes())
    view = verify_input_snapshot(cfg, replace(inputs, trace_path=copy))
    assert view.trace_sha256 == inputs.trace_sha256


def test_input_snapshot_rewindows_a_legal_dynamic_decode_request(tmp_path):
    """合法动态形状按 GeneSim 口径重算处理 token 数，不误拒 decode 请求。"""
    from tune.inputs import verify_input_snapshot

    cfg = _config(tmp_path)
    trace_path = cfg.model.trace.path
    trace_path.write_text(json.dumps({"requests": [
        {"request_id": "a", "input_tokens": [1, 2, 3, 4],
         "output_length": 2, "arrival_time_ns": 0, "is_prefill": True,
         "seq_len": 4, "prompt": None, "padded_query_length": None,
         "initial_kv_length": 0},
        {"request_id": "b", "input_tokens": [],
         "output_length": 3, "arrival_time_ns": 0, "is_prefill": False,
         "seq_len": 4, "prompt": None, "padded_query_length": None,
         "initial_kv_length": 4},
    ]}))
    cfg = replace(cfg, model=replace(cfg.model, trace=replace(
        cfg.model.trace, num_requests=2)))
    inputs = _inputs(cfg)
    view = verify_input_snapshot(cfg, inputs)
    assert view.num_requests == 2
    assert view.max_seq == 7
    assert view.processed_tokens == 4 + 2 + 3


# ---- 评审 round03：原生后端身份按最终配置与实际加载路径登记 ----

def _patch_dramsim3_load(pim_core, monkeypatch):
    """测试替身：让真实库路径的加载与入口符号检查通过。

    DRAMsim3 的原生初始化遇到坏配置会直接退出进程，调优探测只做加载与
    符号检查，因此测试替身也只模拟加载。
    """
    from types import SimpleNamespace

    def fake_load(path):
        return SimpleNamespace(
            dramsim3_init=lambda *args: 1,
            dramsim3_load_trace=lambda *args: 0,
            dramsim3_pim_execute=lambda *args: 4.0,
        )

    monkeypatch.setattr(pim_core.ctypes, "CDLL", fake_load)
    monkeypatch.setattr(pim_core, "DRAMSIM3_LIBRARY_CANDIDATES", ())


def _patch_attacc_native_init(pim_core, monkeypatch):
    """测试替身：模块可导入时 AttAcc 指针非空。"""

    def fake_init(self):
        self.attacc_api = object()

    monkeypatch.setattr(pim_core.Ramulator2PIMCore, "_init_attacc", fake_init)


def test_sim_runtime_identity_tracks_custom_dramsim_library(
        tmp_path, monkeypatch):
    from tune import steps
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import pim.pim_core as pim_core

    _patch_dramsim3_load(pim_core, monkeypatch)
    cfg = _config(tmp_path)
    genesim = REPO_ROOT.parent / "genesim"
    base_raw = yaml.safe_load(
        (genesim / "conf" / "sim.yaml").read_text())
    library = tmp_path / "custom-dramsim.so"
    library.write_bytes(b"native-v1")
    base_raw["dramsim3_lib_path"] = str(library)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    profile = cfg.profile.with_fields(
        pim_type="DIMM-PIM", base_genesim_config=str(base))
    cfg = replace(cfg, profile=profile)

    first = steps._sim_runtime_identity(cfg, base)
    backend = first["native_backend"]
    assert backend["native"] is True
    assert backend["fallback"] is None
    assert backend["library_sha256"]

    # 同内容换路径不是语义变化。
    same_content = tmp_path / "same-content.so"
    same_content.write_bytes(b"native-v1")
    base_raw["dramsim3_lib_path"] = str(same_content)
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    same = steps._sim_runtime_identity(cfg, base)
    assert same["native_backend"] == backend

    # 同一路径替换库内容必须让身份失效。
    library.write_bytes(b"native-v2-different")
    base_raw["dramsim3_lib_path"] = str(library)
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    changed = steps._sim_runtime_identity(cfg, base)
    assert changed["native_backend"] != backend

    # 库缺失时记录实际回退，不能伪装成原生后端。
    base_raw["dramsim3_lib_path"] = str(tmp_path / "missing.so")
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    missing = steps._sim_runtime_identity(cfg, base)
    assert missing["native_backend"]["native"] is False
    assert missing["native_backend"]["fallback"] == "pim_trace"


def test_sim_runtime_identity_records_unloadable_library_as_fallback(tmp_path):
    """文件存在但加载失败时，键必须声明实际使用的轨迹回退。"""
    from tune import steps

    cfg = _config(tmp_path)
    base_raw = yaml.safe_load(
        (REPO_ROOT.parent / "genesim" / "conf" / "sim.yaml").read_text())
    library = tmp_path / "broken-dramsim.so"
    library.write_bytes(b"not-a-shared-object")
    base_raw["dramsim3_lib_path"] = str(library)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    profile = cfg.profile.with_fields(
        pim_type="DIMM-PIM", base_genesim_config=str(base))
    cfg = replace(cfg, profile=profile)

    identity = steps._sim_runtime_identity(cfg, base)["native_backend"]
    assert identity["native"] is False
    assert identity["fallback"] == "pim_trace"
    assert identity["fallback_stage"] == "load_failed"


def test_sim_runtime_identity_tracks_attacc_native_dependency(
        tmp_path, monkeypatch):
    from tune import steps
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import pim.pim_core as pim_core

    _patch_attacc_native_init(pim_core, monkeypatch)
    attacc_dir = tmp_path / "attacc"
    attacc_dir.mkdir()
    module = attacc_dir / "attacc_sim.so"
    module.write_bytes(b"attacc-v1")
    monkeypatch.setattr(pim_core, "attacc_module_paths", lambda: [module])
    monkeypatch.setattr(pim_core, "attacc_module_root", lambda: attacc_dir)

    cfg = _config(tmp_path)
    base_raw = yaml.safe_load(
        (REPO_ROOT.parent / "genesim" / "conf" / "sim.yaml").read_text())
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    profile = cfg.profile.with_fields(
        pim_type="HBM-PIM", base_genesim_config=str(base))
    cfg = replace(cfg, profile=profile)

    first = steps._sim_runtime_identity(cfg, base)
    assert first["native_backend"]["native"] is True
    assert first["native_backend"]["attacc_sha256"]
    module.write_bytes(b"attacc-v2-different")
    second = steps._sim_runtime_identity(cfg, base)
    assert second["native_backend"] != first["native_backend"]


# ---- 评审 round08：原生依赖按稳定角色名与内容建立身份 ----

def test_sim_runtime_identity_tracks_attacc_module_roles(
        tmp_path, monkeypatch):
    """模块内容互换必须让身份失效；同内容换目录不失效；单依赖升级失效。"""
    from tune import steps
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import pim.pim_core as pim_core

    _patch_attacc_native_init(pim_core, monkeypatch)
    first_root = tmp_path / "attacc-first"
    first_root.mkdir()
    (first_root / "attacc_sim.so").write_bytes(b"module-a")
    (first_root / "libattacc.so").write_bytes(b"module-b")

    second_root = tmp_path / "attacc-second"
    second_root.mkdir()
    (second_root / "attacc_sim.so").write_bytes(b"module-a")
    (second_root / "libattacc.so").write_bytes(b"module-b")

    cfg = _config(tmp_path)
    base_raw = yaml.safe_load(
        (REPO_ROOT.parent / "genesim" / "conf" / "sim.yaml").read_text())
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_raw, sort_keys=False))
    profile = cfg.profile.with_fields(
        pim_type="HBM-PIM", base_genesim_config=str(base))
    cfg = replace(cfg, profile=profile)

    def use(root):
        files = sorted(
            path for path in root.rglob("*") if path.is_file())
        monkeypatch.setattr(pim_core, "attacc_module_paths", lambda: files)
        monkeypatch.setattr(pim_core, "attacc_module_root", lambda: root)

    def sim_identity():
        return (steps._sim_runtime_identity(cfg, base)["native_backend"],
                steps.sim_key(cfg, _inputs(cfg), _state(cfg), "compile-key"))

    use(first_root)
    baseline_backend, baseline_sim = sim_identity()

    # 同内容整体换目录：绝对安装路径不参与身份。
    use(second_root)
    assert sim_identity() == (baseline_backend, baseline_sim)

    # 两个模块互换内容：角色到内容的绑定变化，sim_key 必须失效。
    (second_root / "attacc_sim.so").write_bytes(b"module-b")
    (second_root / "libattacc.so").write_bytes(b"module-a")
    swapped_backend, swapped_sim = sim_identity()
    assert swapped_backend != baseline_backend
    assert swapped_sim != baseline_sim

    # 单个依赖真实升级同样必须失效。
    (second_root / "libattacc.so").write_bytes(b"module-c")
    upgraded_backend, upgraded_sim = sim_identity()
    assert upgraded_backend != swapped_backend
    assert upgraded_sim != swapped_sim


# ---- 评审 round10：键声明的后端必须与运行台账一致 ----

def _native_identity(**overrides):
    identity = {
        "pim_type": "DIMM-PIM", "core": "dramsim3",
        "native": True, "fallback": None,
        "library_sha256": "a" * 64,
    }
    identity.update(overrides)
    return identity


def test_native_identity_rejects_unprepared_trace_or_binding():
    """准备阶段失败的身份不能被当成原生结果发布。"""
    from tune import steps

    expected = _native_identity()
    load_failure = _native_identity(
        native=False, fallback="pim_trace", library_sha256=None,
        prepare_failures=1, prepare_stages=["load"])
    bind_failure = _native_identity(
        native=False, fallback="pim_trace", library_sha256=None,
        prepare_failures=1, prepare_stages=["bind"])
    assert steps._native_identity_error(expected, load_failure)
    assert steps._native_identity_error(expected, bind_failure)


def test_publish_rejects_native_backend_that_fell_back(tmp_path, monkeypatch):
    """键声明原生可用、运行实际回退时不得发布完成清单。"""
    from tune import steps

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    monkeypatch.setattr(
        steps, "_native_backend_identity", lambda config: _native_identity())
    fallback = _native_identity(
        native=False, fallback="pim_trace", library_sha256=None)
    record = evaluate_candidate(
        cfg, inputs, state, compile_fn=_fake_compile,
        refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, native_backend=fallback))
    assert record.status == "error"
    assert "原生后端" in (record.failure or "")
    assert not list((cfg.output.dir / "sim").rglob("completion.json"))


def test_declared_trace_fallback_is_published_and_reused(tmp_path):
    """键与运行台账都声明内置轨迹回退时是合法结果，可复用。"""
    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = evaluate_candidate(
        cfg, inputs, state, compile_fn=_fake_compile,
        refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert first.status == "success", first.failure
    completion = json.loads(
        (Path(first.paths["sim_dir"]) / "completion.json").read_text())
    assert completion["pim_native_backend"] == NATIVE_TRACE_FALLBACK
    expected = completion["expected_native_backend"]
    for key, value in NATIVE_TRACE_FALLBACK.items():
        assert expected[key] == value
    second = evaluate_candidate(
        cfg, inputs, state, compile_fn=_fake_compile,
        refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert second.status == "success", second.failure
    assert second.sim_cache_hit is True


def test_cached_completion_with_mismatched_backend_identity_is_not_reused(
        tmp_path):
    """完成清单自报另一计时后端时，快速缓存不得命中。"""
    from tune import steps

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    first = evaluate_candidate(
        cfg, inputs, state, compile_fn=_fake_compile,
        refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert first.status == "success", first.failure
    attempt = Path(first.paths["sim_dir"])
    summary_path = attempt / "results" / "summary.json"
    completion_path = attempt / "completion.json"
    summary = json.loads(summary_path.read_text())
    completion = json.loads(completion_path.read_text())
    forged = _native_identity()
    summary["pim_native_backend"] = forged
    summary_path.write_text(json.dumps(summary))
    completion["files"]["results/summary.json"] = file_sha256(summary_path)
    completion["pim_native_backend"] = forged
    completion["expected_native_backend"] = forged
    completion_path.write_text(json.dumps(completion))

    expected = dict(NATIVE_TRACE_FALLBACK)
    assert steps._find_cached_sim(
        cfg, inputs, first.sim_key, expected_native=expected) is None


def test_native_backend_identity_is_part_of_sim_key(tmp_path, monkeypatch):
    """同一配置下实际后端从原生变成回退，仿真键必须失效。"""
    from tune import steps

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    monkeypatch.setattr(
        steps, "_native_backend_identity", lambda config: _native_identity())
    native_key = sim_key(cfg, inputs, state, "compile-key")
    monkeypatch.setattr(
        steps, "_native_backend_identity",
        lambda config: _native_identity(
            native=False, fallback="pim_trace", library_sha256=None))
    fallback_key = sim_key(cfg, inputs, state, "compile-key")
    assert native_key != fallback_key


def test_run_semantic_identity_changes_when_backend_probe_changes(
        tmp_path, monkeypatch):
    """续跑身份包含实际后端；环境变化必须在回放前失效。"""
    from tune import steps

    cfg = _config(tmp_path)
    monkeypatch.setattr(
        steps, "_native_backend_identity", lambda config: _native_identity())
    first = steps.run_semantic_identity(cfg)
    monkeypatch.setattr(
        steps, "_native_backend_identity",
        lambda config: _native_identity(
            native=False, fallback="pim_trace", library_sha256=None))
    second = steps.run_semantic_identity(cfg)
    assert first["runtime"] != second["runtime"]


# ---- 评审 round03：源码身份按生产者角色覆盖 ----

def _source_tree(root):
    for name in ("contracts", "graph", "comm", "memory", "runtime",
                 "opcompiler_bridge", "genesim_bridge", "tune", "scripts"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "paths.json").write_text("{}")
    files = {
        "tune/steps.py": "STEPS = 1\n",
        "tune/candidate.py": "CANDIDATE = 1\n",
        "tune/config.py": "CONFIG = 1\n",
        "tune/loop.py": "LOOP = 1\n",
        "tune/_genesim.py": "GENESIM = 1\n",
        "scripts/run_full_pipeline.py": "PIPELINE = 1\n",
        "scripts/tune_pipeline.py": "ENTRY = 1\n",
        "scripts/tune_sweep.py": "SWEEP = 1\n",
        "scripts/calibrate_profile.py": "CALIBRATE = 1\n",
    }
    for name, content in files.items():
        (root / name).write_text(content)


def test_semantic_source_identity_is_role_based(tmp_path, monkeypatch):
    """编译、仿真与运行身份各覆盖实际生产者；注释变化不触发重编。"""
    from tune import steps

    root = tmp_path / "repo"
    root.mkdir()
    _source_tree(root)
    monkeypatch.setattr(steps, "REPO_ROOT", root)

    compile_before = steps.source_fingerprint("compile")
    sim_before = steps.source_fingerprint("sim")
    run_before = steps.source_fingerprint("run")

    # 纯注释变化不是语义变化。
    (root / "tune" / "steps.py").write_text("STEPS = 1  # 只改注释\n")
    assert steps.source_fingerprint("compile") == compile_before

    # tune/steps.py 是图/算子编译生产者，必须让编译键失效。
    (root / "tune" / "steps.py").write_text("STEPS = 2\n")
    compile_changed = steps.source_fingerprint("compile")
    assert compile_changed != compile_before

    # 仿真命令生产者只影响仿真键与运行身份，不改变编译键。
    (root / "scripts" / "run_full_pipeline.py").write_text("PIPELINE = 2\n")
    assert steps.source_fingerprint("compile") == compile_changed
    assert steps.source_fingerprint("sim") != sim_before
    assert steps.source_fingerprint("run") != run_before

    # 续跑算法属于运行身份，不改变已编好的图/算子。
    (root / "tune" / "loop.py").write_text("LOOP = 2\n")
    after_loop = steps.source_fingerprint("run")
    assert after_loop != run_before
    assert steps.source_fingerprint("compile") == compile_changed


def test_run_semantic_identity_declares_search_source(tmp_path):
    from tune import steps

    cfg = _config(tmp_path)
    identity = steps.run_semantic_identity(cfg)
    assert identity["search_source"] == steps.source_fingerprint("run")
    assert identity["compiler_source"] == steps.source_fingerprint("compile")


def test_sim_key_declares_sim_source(tmp_path):
    from tune import steps

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    base = compile_key(cfg, inputs, state)
    first = sim_key(cfg, inputs, state, base)
    original = steps.sim_source_fingerprint
    try:
        steps.sim_source_fingerprint = lambda: "sim-source-a"
        second = sim_key(cfg, inputs, state, base)
        steps.sim_source_fingerprint = lambda: "sim-source-b"
        third = sim_key(cfg, inputs, state, base)
    finally:
        steps.sim_source_fingerprint = original
    assert len({first, second, third}) == 3


# ---- 评审 round03：校准报告经 GeneSim 成熟度进入轮次记录 ----

def test_round_maturity_consumes_the_real_calibration_contract(tmp_path):
    from tests.test_tune_calibration import _passed_report
    from tune.steps import _result_maturity

    report_path = tmp_path / "calibration_report.json"
    report = _passed_report(tmp_path)
    report_path.write_text(json.dumps(report))
    cfg = _config(tmp_path)
    cfg = replace(cfg, calibration=replace(
        cfg.calibration, validation_report=report_path))
    calibrated_profile = cfg.profile.with_fields(parameter_sources={
        "pim.clock_period_ns": {
            "origin": "ceva_measured", "evidence": "probe",
            "unit": "ns", "scope": "ceva"}})
    state = TuneState(StrategyChoice(4, 2), OpKnobs(4), calibrated_profile)
    output = SimOutput(
        sim_dir=tmp_path, config_path=tmp_path / "config.yaml",
        results_dir=tmp_path, traces_dir=tmp_path,
        summary={"simulation_backend": "pipeline"},
        autotune={}, effective_placement=None)
    assert _result_maturity(output, cfg, state) == "validated_in_scope"

    # 报告生成后篡改逐点证据：同一份摘要不能继续升级轮次记录。
    broken = json.loads(json.dumps(report))
    broken["scope"]["validation"]["samples"][0]["observed"] += 1.0
    report_path.write_text(json.dumps(broken))
    assert _result_maturity(output, cfg, state) == "model_prediction"


# ---- 评审 round05：仿真完成协议必须覆盖全部关键产物 ----

_REQUIRED_SIM_ROLES = (
    "results/summary.json", "results/autotune_metrics.json",
    "config.yaml", "manifest.json",
    "hardware_contract.json", "effective_placement.json")


def _built_sim(tmp_path, *, targets=None, traces=True):
    """按真实 build_sim 生成配置/契约/manifest，再补仿真结果与放置。"""
    from tune.steps import _publish_sim_complete  # noqa: F401

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    artifacts = build_candidate(cfg, inputs, state, compile_fn=_fake_compile,
                                refine_fn=_fake_refine)
    key = sim_key(cfg, inputs, state, artifacts.compile_key)
    sim_dir = cfg.output.dir / "sim" / key / "attempt-a"
    build_sim(cfg, inputs, state, artifacts, sim_dir)
    output = _fake_sim(cfg, sim_dir, sim_dir / "config.yaml",
                       targets=targets, traces=traces)
    return cfg, inputs, sim_dir, output, key, artifacts.compile_key


@pytest.mark.parametrize("role", _REQUIRED_SIM_ROLES)
def test_sim_completion_refuses_missing_required_role(tmp_path, role):
    """写完成清单时缺任一必需角色都必须拒绝，不能只登记 summary。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    (sim_dir / role).unlink()
    with pytest.raises(RuntimeError, match="完成清单|仿真完成"):
        _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, "fixed")
    assert not (sim_dir / "completion.json").exists()


def test_sim_completion_rejects_missing_trace_for_attached_target(tmp_path):
    """最终放置里挂在 PU 上的算子必须有对应 trace，覆盖不全不得发布。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(
        tmp_path,
        targets=[
            {"op_id": 0, "node_id": 0, "vpu_id": 1, "attached_pu_id": 0},
            {"op_id": 1, "node_id": 0, "vpu_id": 1, "attached_pu_id": 0},
        ])
    with pytest.raises(RuntimeError, match="trace"):
        _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, "fixed")


def test_sim_completion_rejects_trace_outside_the_final_mapping(tmp_path):
    """trace 集合必须与最终执行图一致，多出的 trace 同样拒绝。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(tmp_path)
    (sim_dir / "pim_traces" / "op_9_GEMM.pim_trace").write_text(
        json.dumps({"op_id": 9,
                    "compile_signature": {"trace_source": "pimir"}}))
    with pytest.raises(RuntimeError, match="trace"):
        _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, "fixed")


# ---- 评审 round09：trace 必须精确对应最终 PIM 执行目标 ----


def _mixed_targets():
    """op0 留在 PIM，op1 在搜索后迁离 PIM。"""
    return [
        {"op_id": 0, "node_id": 0, "vpu_id": 1, "attached_pu_id": 0},
        {"op_id": 1, "node_id": 0, "vpu_id": 2, "attached_pu_id": -1},
    ]


def test_sim_completion_rejects_trace_for_target_migrated_off_pim(tmp_path):
    """迁离 PIM 后残留的 trace 不能被当成完整覆盖发布。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(
        tmp_path, targets=_mixed_targets())
    (sim_dir / "pim_traces" / "op_1_GEMM.pim_trace").write_text(json.dumps(
        {"op_id": 1, "compile_signature": {"trace_source": "pimir"}}))
    with pytest.raises(RuntimeError, match="trace"):
        _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, "fixed")
    assert not (sim_dir / "completion.json").exists()


def test_mixed_placement_with_only_pim_trace_is_accepted(tmp_path):
    """合法对照：非 PIM 目标没有 trace 时，覆盖集合仍精确相等。"""
    from tune.steps import _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(
        tmp_path, targets=_mixed_targets())
    _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, "fixed")
    assert (sim_dir / "completion.json").is_file()


def test_cached_completion_with_residual_non_pim_trace_is_not_reused(tmp_path):
    """已登记哈希的非 PIM 残留 trace 仍不能让完成目录命中缓存。"""
    from tune.steps import _execution_mode, _find_cached_sim, _publish_sim_complete

    cfg, inputs, sim_dir, output, key, compile_key = _built_sim(
        tmp_path, targets=_mixed_targets())
    mode = _execution_mode(cfg, None)
    _publish_sim_complete(cfg, inputs, sim_dir, key, compile_key, output, mode)
    assert _find_cached_sim(cfg, _inputs(cfg), key) is not None
    residual = sim_dir / "pim_traces" / "op_1_GEMM.pim_trace"
    residual.write_text(json.dumps(
        {"op_id": 1, "compile_signature": {"trace_source": "pimir"}}))
    completion_path = sim_dir / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["pim_traces/op_1_GEMM.pim_trace"] = file_sha256(
        residual)
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, _inputs(cfg), key) is None


def test_incomplete_completion_manifest_is_not_reused(tmp_path):
    """只列 summary 与 trace 的完成清单不能命中快速缓存。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    attempt = cfg.output.dir / "sim" / "probe-key" / "attempt-incomplete"
    (attempt / "results").mkdir(parents=True)
    (attempt / "pim_traces").mkdir()
    summary = attempt / "results" / "summary.json"
    summary.write_text(json.dumps(SUMMARY))
    trace = attempt / "pim_traces" / "op_0_GEMM.pim_trace"
    trace.write_text(json.dumps(
        {"op_id": 0, "compile_signature": {"trace_source": "pimir"}}))
    files = {
        "results/summary.json": file_sha256(summary),
        "pim_traces/op_0_GEMM.pim_trace": file_sha256(trace),
    }
    (attempt / "completion.json").write_text(json.dumps({
        "schema_version": 1, "complete": True, "sim_key": "probe-key",
        "compile_key": "compile-key", "result_producer": "pipeline_simulator",
        "total_time_s": 1.5, "files": files,
    }))
    assert _find_cached_sim(cfg, _inputs(cfg), "probe-key") is None


def test_completion_key_must_match_the_run_manifest(tmp_path):
    """完成清单的 compile/sim 键必须与本次仿真的 manifest 同源。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    first = evaluate_candidate(cfg, _inputs(cfg), _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    attempt = Path(first.paths["sim_dir"])
    manifest_path = attempt / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["compile_key"] = "another-compile"
    manifest_path.write_text(json.dumps(manifest))
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["manifest.json"] = file_sha256(manifest_path)
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, _inputs(cfg), first.sim_key) is None


def test_complete_sim_artifact_set_is_reused(tmp_path):
    """完整完成清单（summary/配置/manifest/契约/放置/全覆盖 trace）可命中。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    first = evaluate_candidate(cfg, _inputs(cfg), _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    cached = _find_cached_sim(cfg, _inputs(cfg), first.sim_key)
    assert cached is not None
    assert cached.effective_placement is not None
    assert Path(cached.effective_placement).is_file()


# ---- 评审 round05：逐请求 KV 峰值与并发上限贯通到图编译 ----

def test_request_kv_tokens_follow_genesim_peak_offsets():
    """逐请求 KV 峰值与 GeneSim 的上下文上界逐项一致。"""
    from tune.config import RequestBound
    from tune.steps import _request_kv_tokens

    assert _request_kv_tokens((
        RequestBound(prompt_tokens=16, output_tokens=8, initial_kv=0),
    )) == (23,)
    assert _request_kv_tokens((
        RequestBound(prompt_tokens=0, output_tokens=3, initial_kv=10),
    )) == (13,)
    assert _request_kv_tokens((
        RequestBound(prompt_tokens=4, output_tokens=1, initial_kv=2),
    )) == (6,)


def test_real_compile_passes_request_peaks_and_concurrency(
        tmp_path, monkeypatch):
    """图编译入口必须收到逐请求峰值与仿真同一并发上限。"""
    from types import SimpleNamespace

    from tune import steps

    cfg = _config(tmp_path)
    inputs = _inputs(cfg)
    state = _state(cfg)
    seen = {}

    def fake_export(model, ir_path, out_dir, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(sidecar=out_dir / "placement.json")

    monkeypatch.setattr(
        "genesim_bridge.placement_export.export_placement_for_strategy",
        fake_export)
    monkeypatch.setattr(steps, "_load_model", lambda model_dir: object())
    monkeypatch.setattr(steps, "model_dims", lambda model_dir: object())
    monkeypatch.setattr(
        steps, "build_strategy",
        lambda state, dims: SimpleNamespace(
            name="stub", num_dpus=4, num_stages=2, tp_width=2,
            weight_rules=(), dpu_ids=tuple(range(4))))
    steps._real_compile(cfg, inputs, state, tmp_path / "out", False)
    # 轨迹：prefill 4 token、输出 2、无初始 KV → 峰值 4 + (2 - 1) = 5。
    assert seen["request_kv_tokens"] == (5,)
    # CEVA 基配置 max_inflight_requests=1，与仿真侧同一并发上限。
    assert seen["max_concurrent_requests"] == 1


def test_completion_must_register_every_trace_hash(tmp_path):
    """完成清单的哈希表漏登 trace 时不能命中缓存。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    first = evaluate_candidate(cfg, _inputs(cfg), _state(cfg), sequence=0,
                               compile_fn=_fake_compile, refine_fn=_fake_refine,
                               sim_fn=_fake_sim)
    attempt = Path(first.paths["sim_dir"])
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"] = {
        role: digest for role, digest in completion["files"].items()
        if not role.startswith("pim_traces/")}
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, _inputs(cfg), first.sim_key) is None


# ---- 评审 round06：编译键必须覆盖请求分布与并发准入 ----

def _write_trace_requests(path: Path, requests: list) -> None:
    path.write_text(json.dumps({"requests": requests}))


def _read_inputs(cfg):
    """只读解析输入身份；密钥测试不发布快照，避免改写输出目录。"""
    from tune.inputs import prepare_inputs

    return prepare_inputs(cfg)


def _request(request_id="r0", *, tokens=(1, 2, 3, 4), output=2,
             prefill=True, initial=0, seq_len=4):
    return {
        "request_id": request_id, "input_tokens": list(tokens),
        "output_length": output, "arrival_time_ns": 0,
        "is_prefill": prefill, "seq_len": seq_len, "prompt": None,
        "padded_query_length": None, "initial_kv_length": initial,
    }


def test_compile_key_covers_request_peaks_and_concurrency(tmp_path):
    """相同 max_seq 下，逐请求 KV 峰值不同必须得到不同编译键。"""
    from dataclasses import replace

    cfg = _config(tmp_path)
    cfg = replace(
        cfg, model=replace(cfg.model, max_seq=24,
                           trace=replace(cfg.model.trace, num_requests=1)))
    state = _state(cfg)
    trace_path = cfg.model.trace.path

    _write_trace_requests(trace_path, [_request("a")])
    inputs_one = _inputs(cfg)
    _write_trace_requests(
        trace_path,
        [_request("a"),
         _request("b", tokens=(), output=2, prefill=False,
                  initial=4, seq_len=4)])
    cfg_two = replace(cfg, model=replace(cfg.model, trace=replace(
        cfg.model.trace, num_requests=2)))
    inputs_two = _read_inputs(cfg_two)

    assert inputs_one.max_seq == inputs_two.max_seq == 24
    assert compile_key(cfg, inputs_one, state) != compile_key(
        cfg_two, inputs_two, state), "请求分布变化必须让编译缓存失效"
    # 完全相同的请求仍复用同一编译键（合法对照）。
    assert compile_key(cfg, inputs_one, state) == compile_key(
        cfg, _read_inputs(cfg), state)


def test_compile_key_covers_effective_request_concurrency(tmp_path):
    """基配置的并发准入上限变化同样必须改变编译身份。"""
    from dataclasses import replace

    cfg = _config(tmp_path)
    cfg = replace(
        cfg, model=replace(cfg.model, trace=replace(
            cfg.model.trace, num_requests=2)))
    _write_trace_requests(cfg.model.trace.path, [
        _request("a"),
        _request("b", tokens=(), output=2, prefill=False,
                 initial=4, seq_len=4),
    ])
    inputs = _read_inputs(cfg)
    state = _state(cfg)
    from tune.steps import resolve_base_genesim_config

    base = resolve_base_genesim_config(state.profile, cfg.profile_path)
    if not base.is_absolute():
        base = (cfg.profile_path.parent / base).resolve()
    custom = tmp_path / "sim_concurrency.yaml"
    payload = yaml.safe_load(base.read_text())
    payload.setdefault("scheduler", {})["max_inflight_requests"] = 2
    custom.write_text(yaml.safe_dump(payload, sort_keys=False))
    profile = replace(state.profile, base_genesim_config=str(custom))
    slower_admission = replace(state, profile=profile)
    assert compile_key(cfg, inputs, state) != compile_key(
        cfg, inputs, slower_admission)


def test_build_candidate_recompiles_for_new_request_distribution(tmp_path):
    """请求峰值变化时 build_candidate 不能复用旧请求的容量复核结果。"""
    from dataclasses import replace

    cfg = _config(tmp_path)
    cfg = replace(
        cfg, model=replace(cfg.model, max_seq=24,
                           trace=replace(cfg.model.trace, num_requests=1)))
    state = _state(cfg)
    trace_path = cfg.model.trace.path
    calls = []

    def counting_compile(*args, **kwargs):
        calls.append(1)
        return _fake_compile(*args, **kwargs)

    _write_trace_requests(trace_path, [_request("a")])
    first = build_candidate(cfg, _inputs(cfg), state,
                            compile_fn=counting_compile,
                            refine_fn=_fake_refine)
    assert first.cache_hit is False
    _write_trace_requests(
        trace_path,
        [_request("a"),
         _request("b", tokens=(), output=2, prefill=False,
                  initial=4, seq_len=4)])
    cfg_two = replace(cfg, model=replace(cfg.model, trace=replace(
        cfg.model.trace, num_requests=2)))
    # 已发布快照不允许原地覆盖；这里把新窗口按同一规范字节单独发布，
    # 模拟新请求分布进入同一输出目录后的编译输入。
    from tune.inputs import _trace_window_bytes

    data, _, _ = _trace_window_bytes(cfg_two, cfg.output.dir / "inputs")
    window_path = tmp_path / "trace-window.json"
    window_path.write_bytes(data)
    second = build_candidate(
        cfg_two, replace(_read_inputs(cfg_two), trace_path=window_path),
        state, compile_fn=counting_compile, refine_fn=_fake_refine)
    assert second.cache_hit is False
    assert len(calls) == 2, "新请求分布必须重新编译，不能复用旧目录"


# ---- 评审 round06：编译产物清单必须有必需角色与 PIM MLIR 覆盖 ----

def test_empty_compile_manifest_is_not_reused(tmp_path):
    """空产物清单不能当作完整编译缓存命中。"""
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    empty = ArtifactManifest(
        version=1, compile_key=first.compile_key, inputs=(), files=())
    (first.compile_dir / "artifacts.json").write_text(
        json.dumps(empty.to_json()))
    with pytest.raises(ValueError, match="角色|为空"):
        verify_compile_products(first.compile_dir, empty)
    second = build_candidate(cfg, _inputs(cfg), _state(cfg),
                             compile_fn=_fake_compile, refine_fn=_fake_refine)
    assert second.cache_hit is False


@pytest.mark.parametrize("role", [
    "partition_plan.json", "placement.json", "refined.ir", "cost.json",
    "placed.ir",
])
def test_missing_compile_role_is_not_reused(tmp_path, role):
    """设计产物的每个必需角色缺失时都必须在读取端拒绝。"""
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    manifest_path = first.compile_dir / "artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    removed = next(entry for entry in manifest["files"]
                   if entry["role"] == role)
    manifest["files"] = [entry for entry in manifest["files"]
                         if entry["role"] != role]
    manifest_path.write_text(json.dumps(manifest))
    (first.compile_dir / removed["role"]).unlink()
    parsed = ArtifactManifest.from_json(manifest)
    with pytest.raises(ValueError, match="角色|缺少"):
        verify_compile_products(first.compile_dir, parsed)
    second = build_candidate(cfg, _inputs(cfg), _state(cfg),
                             compile_fn=_fake_compile, refine_fn=_fake_refine)
    assert second.cache_hit is False
    # 坏 attempt 被改名隔离，后续读取不再重复尝试。
    assert any(
        path.name.startswith("invalid-")
        for path in first.compile_dir.parent.iterdir())


def test_build_candidate_refuses_to_publish_without_placed_ir(tmp_path):
    """编译入口漏写 placed.ir 时不能发布完整 attempt。"""
    cfg = _config(tmp_path)

    def compile_without_placed(*args, **kwargs):
        result = _fake_compile(*args, **kwargs)
        (result.out_dir / "placed.ir").unlink()
        return result

    with pytest.raises(ValueError, match="placed.ir|必需角色"):
        build_candidate(
            cfg, _inputs(cfg), _state(cfg),
            compile_fn=compile_without_placed, refine_fn=_fake_refine)
    cache_root = cfg.output.dir / "cache" / "compile"
    assert not cache_root.exists() or not list(
        cache_root.glob("*/attempt-*"))


def test_compile_manifest_must_cover_every_sidecar_pimir(tmp_path):
    """sidecar 引用的 PIM MLIR 必须全部登记在清单里。"""
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    manifest = ArtifactManifest.from_json(
        json.loads((first.compile_dir / "artifacts.json").read_text()))
    kept = tuple(entry for entry in manifest.files
                 if not entry.role.startswith("pimir/"))
    reduced = ArtifactManifest(
        version=manifest.version, compile_key=manifest.compile_key,
        inputs=manifest.inputs, files=kept)
    with pytest.raises(ValueError, match="pimir|PIM"):
        verify_compile_products(first.compile_dir, reduced)


# ---- 评审 round08：清单角色必须绑定规范相对路径 ----

def test_compile_manifest_role_cannot_alias_another_file(tmp_path):
    """缺 placed.ir 时不能把角色别名到 refined.ir 冒充完整产物。"""
    from tune.inputs import file_sha256
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    manifest = json.loads((first.compile_dir / "artifacts.json").read_text())
    (first.compile_dir / "placed.ir").unlink()
    refined = first.compile_dir / "refined.ir"
    for entry in manifest["files"]:
        if entry["role"] == "placed.ir":
            entry["path"] = "refined.ir"
            entry["sha256"] = file_sha256(refined)
    with pytest.raises(ValueError, match="角色|路径"):
        verify_compile_products(
            first.compile_dir, ArtifactManifest.from_json(manifest))


def test_compile_manifest_rejects_duplicate_role_or_path(tmp_path):
    """重复角色/重复路径的别名清单不能被当成完整编译产物。"""
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    manifest = json.loads((first.compile_dir / "artifacts.json").read_text())
    duplicate = dict(next(entry for entry in manifest["files"]
                          if entry["role"] == "cost.json"))
    manifest["files"].append(duplicate)
    with pytest.raises(ValueError, match="重复"):
        verify_compile_products(
            first.compile_dir, ArtifactManifest.from_json(manifest))


def test_compile_manifest_rejects_symlink_escape(tmp_path):
    """清单路径解析到编译目录之外的符号链接必须拒绝。"""
    from tune.inputs import file_sha256
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    outside = tmp_path / "outside.ir"
    outside.write_text("outside")
    link = first.compile_dir / "extra.link"
    link.symlink_to(outside)
    manifest = json.loads((first.compile_dir / "artifacts.json").read_text())
    manifest["files"].append({
        "role": "extra.link", "path": "extra.link",
        "sha256": file_sha256(outside)})
    with pytest.raises(ValueError, match="越界|符号链接|之外"):
        verify_compile_products(
            first.compile_dir, ArtifactManifest.from_json(manifest))


def test_compile_manifest_rejects_symlink_alias_inside_root(tmp_path):
    """目录内符号链接也是路径别名，不能冒充另一个设计产物。"""
    from tune.inputs import file_sha256
    from tune.report import ArtifactManifest, verify_compile_products

    cfg = _config(tmp_path)
    first = build_candidate(cfg, _inputs(cfg), _state(cfg),
                            compile_fn=_fake_compile, refine_fn=_fake_refine)
    refined = first.compile_dir / "refined.ir"
    link = first.compile_dir / "alias.ir"
    link.symlink_to(refined)
    manifest = json.loads((first.compile_dir / "artifacts.json").read_text())
    manifest["files"].append({
        "role": "alias.ir", "path": "alias.ir",
        "sha256": file_sha256(refined)})
    with pytest.raises(ValueError, match="符号链接"):
        verify_compile_products(
            first.compile_dir, ArtifactManifest.from_json(manifest))


# ---- 评审 round06：内层搜索指标缺失或矛盾不得发布/命中 ----

def _search_cfg(tmp_path):
    cfg = _config(tmp_path)
    return replace(cfg, placement=replace(cfg.placement, enabled=True))


def test_native_search_requires_autotune_metrics(tmp_path):
    """启用原生放置但缺 autotune 指标时，候选不能算成功。"""
    cfg = _search_cfg(tmp_path)
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(
            c, d, p, autotune={}, write_autotune=False))
    assert record.status == "error"
    assert "autotune" in (record.failure or "")


def test_native_search_rejects_inconsistent_metric_counts(tmp_path):
    """搜索次数与总后端评估数矛盾时拒绝发布。"""
    cfg = _search_cfg(tmp_path)
    metrics = _search_autotune(evaluations=2)
    metrics["total_backend_evaluations"] = 2
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=metrics))
    assert record.status == "error"
    assert "autotune" in (record.failure or "")


def test_fixed_mapping_records_zero_search_and_one_final_execution(tmp_path):
    """固定映射显式记录零搜索与一次最终执行。"""
    cfg = _config(tmp_path)
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine, sim_fn=_fake_sim)
    assert record.status == "success"
    assert record.inner_evaluations == 0
    assert record.validation_evaluations == 1
    assert record.metrics["autotune"]["execution_mode"] == "fixed"
    assert record.metrics["autotune"]["evaluations"] == 0
    assert record.metrics["autotune"]["final_backend_evaluations"] == 1
    completion = json.loads(
        (Path(record.paths["sim_dir"]) / "completion.json").read_text())
    assert completion["execution_mode"] == "fixed"
    assert "results/autotune_metrics.json" in completion["files"]


def test_replay_records_zero_search_and_one_final_execution(tmp_path):
    """严格重放同样显式记录零搜索与一次最终执行。"""
    cfg = _config(tmp_path)
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    placement = _write_effective_placement(saved_dir)
    replayed = dict(_fixed_autotune(), replayed=True, stop_reason="replayed")
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=replayed),
        replay_placement=placement)
    assert record.status == "success"
    assert record.metrics["autotune"]["execution_mode"] == "replay"
    assert record.metrics["autotune"]["evaluations"] == 0


def test_native_search_records_search_and_final_counts(tmp_path):
    """原生搜索把算法、后端、逻辑/实际评估数与最终重放一致性写进轮次。"""
    cfg = _search_cfg(tmp_path)
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(
            c, d, p, autotune=_search_autotune(evaluations=3)))
    assert record.status == "success"
    summary = record.metrics["autotune"]
    assert summary["execution_mode"] == "search"
    assert summary["algorithm"] == "simulated_annealing"
    assert summary["evaluation_backend"] == "pipeline"
    assert summary["evaluations"] == 3
    assert summary["final_backend_evaluations"] == 1
    assert summary["total_backend_evaluations"] == 4
    assert summary["search_and_validation_backend_match"] is True
    assert record.inner_evaluations == 3


def test_tampered_autotune_metrics_are_not_reused(tmp_path):
    """完成清单 hash 同步后，模式矛盾的内层指标仍不能命中缓存。"""
    from tune.steps import _find_cached_sim

    cfg = _config(tmp_path)
    first = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine, sim_fn=_fake_sim)
    attempt = Path(first.paths["sim_dir"])
    autotune_path = attempt / "results" / "autotune_metrics.json"
    tampered = dict(_fixed_autotune(), enabled=True)
    autotune_path.write_text(json.dumps(tampered))
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["results/autotune_metrics.json"] = file_sha256(
        autotune_path)
    completion["autotune_metrics_sha256"] = file_sha256(autotune_path)
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, _inputs(cfg), first.sim_key) is None


# ---- 评审 round07：内层指标必须绑定本次生效放置策略 ----

def test_search_metrics_must_match_the_effective_algorithm(tmp_path):
    """记录的内层算法与生效配置不一致时，候选不能发布成功。"""
    cfg = _search_cfg(tmp_path)
    metrics = _search_autotune(evaluations=2)
    metrics["algorithm"] = "tabu_search"
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=metrics))
    assert record.status == "error"
    assert "算法" in (record.failure or "")


def test_search_metrics_must_match_the_effective_backends(tmp_path):
    """搜索后端或验证后端与配置不一致时都不能发布。"""
    cfg = _search_cfg(tmp_path)
    wrong_search = _search_autotune(evaluations=2)
    wrong_search["evaluation_backend"] = "predictor"
    wrong_search["search_and_validation_backend_match"] = False
    wrong_search["final_backend_matches_search"] = False
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=wrong_search))
    assert record.status == "error"
    assert "后端" in (record.failure or "")

    wrong_validation = _search_autotune(evaluations=2)
    wrong_validation["validation_backend"] = "predictor"
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=wrong_validation))
    assert record.status == "error"
    assert "后端" in (record.failure or "")


def test_search_with_configured_predictor_filter_is_accepted(tmp_path):
    """配置明确允许 predictor 筛选时合法，最终验证仍须是 pipeline。"""
    cfg = _search_cfg(tmp_path)
    cfg = replace(cfg, placement=replace(
        cfg.placement, evaluation_backend="predictor"))
    metrics = _search_autotune(evaluations=2)
    metrics["evaluation_backend"] = "predictor"
    record = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(c, d, p, autotune=metrics))
    assert record.status == "success"
    assert record.metrics["autotune"]["evaluation_backend"] == "predictor"
    assert record.metrics["autotune"]["validation_backend"] == "pipeline"


def test_cached_search_metrics_must_match_the_effective_config(tmp_path):
    """缓存读取同样要按生效配置核对算法，不能只接受枚举值。"""
    from tune.steps import _find_cached_sim

    cfg = _search_cfg(tmp_path)
    first = evaluate_candidate(
        cfg, _inputs(cfg), _state(cfg), sequence=0,
        compile_fn=_fake_compile, refine_fn=_fake_refine,
        sim_fn=lambda c, d, p: _fake_sim(
            c, d, p, autotune=_search_autotune(evaluations=2)))
    assert first.status == "success"
    attempt = Path(first.paths["sim_dir"])
    autotune_path = attempt / "results" / "autotune_metrics.json"
    autotune_path.write_text(json.dumps(
        dict(_search_autotune(evaluations=2), algorithm="tabu_search")))
    completion_path = attempt / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["files"]["results/autotune_metrics.json"] = file_sha256(
        autotune_path)
    completion["autotune_metrics_sha256"] = file_sha256(autotune_path)
    completion_path.write_text(json.dumps(completion))
    assert _find_cached_sim(cfg, _inputs(cfg), first.sim_key) is None
