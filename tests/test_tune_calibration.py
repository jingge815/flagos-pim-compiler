"""校准：逐点复算、留出去重与结果分级。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tune.calibration import (
    CalibrationSample,
    build_report,
    group_statistics,
    load_samples,
    profile_with_fit,
    result_maturity,
    split_datasets,
)


def _target(scope=None):
    """测点必须声明的规范化目标配置；默认取自报告适用范围。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    return ceva_validation.target_config(scope or _scope())


def _other_target():
    target = json.loads(json.dumps(_target()))
    target["target_name"] = "unrelated-target"
    return target


def _sample(sample_id, *, dataset="train", source="ceva_simulator",
            phase="prefill", kind="compute", shape=(2, 16),
            predicted=100.0, observed=110.0, target=None, features=None,
            unit="ns", operator="", artifact_phase=""):
    return CalibrationSample(
        sample_id=sample_id, dataset_id=dataset, source=source, phase=phase,
        kind=kind, shape=shape, dtype="float16", predicted=predicted,
        observed=observed, target=target or _target(),
        unit=unit, features=dict(features or {}),
        operator=operator, artifact_phase=artifact_phase,
    )


def test_statistics_are_recomputable_from_points():
    samples = [
        _sample("s0", predicted=100, observed=110),   # rel 0.0909
        _sample("s1", predicted=100, observed=120),   # rel 0.1667
        _sample("s2", predicted=100, observed=130),   # rel 0.2308
        _sample("s3", predicted=100, observed=140),   # rel 0.2857
        _sample("s4", predicted=100, observed=150),   # rel 0.3333
    ]
    stats = group_statistics(samples)["prefill/compute"]
    relatives = sorted(abs(s.observed - s.predicted) / s.observed for s in samples)
    assert stats["count"] == 5
    assert stats["median"] == pytest.approx(relatives[2])
    assert stats["p90"] == pytest.approx(relatives[4])  # ceil(4.5)-1 = 4
    assert stats["max"] == pytest.approx(relatives[-1])
    errors = sorted(s.error for s in samples)
    assert stats["median_error"] == pytest.approx(errors[2])


def test_training_and_heldout_duplicate_identity_is_rejected():
    train = _sample("a", dataset="train")
    held = _sample("b", dataset="heldout")  # 同一测点换个名字
    with pytest.raises(ValueError, match="改名两用"):
        split_datasets([train, held], training_dataset="train",
                       validation_dataset="heldout")


def test_microbenchmark_cannot_become_passed(tmp_path):
    samples = [_sample("t", dataset="train",
                       operator="GEMM", artifact_phase="compute"),
               _sample("v1", dataset="heldout", shape=(4, 16),
                       operator="GEMM", artifact_phase="compute"),
               _sample("v2", dataset="heldout", shape=(8, 16),
                       predicted=200, observed=200,
                       operator="GEMM", artifact_phase="compute")]
    report = build_report(samples, training_dataset="train",
                          validation_dataset="heldout")
    assert report["status"] == "report_only"
    assert "产物" in report["reason"]
    # 即使证据齐全，结果来源不是 pipeline 也不能升级。
    report = _passed_report(
        tmp_path,
        simulation_summary=_summary(tmp_path, producer="predictor_surrogate"))
    assert report["status"] == "report_only"
    assert "pipeline" in report["reason"]


def test_assumption_sources_keep_model_prediction(tmp_path):
    # 只有 status=passed 的报告没有逐点证据、产物哈希和适用范围，不能升级。
    bare = {"status": "passed"}
    assert result_maturity(
        bare, profile=_profile(), profile_sources={},
        backend="pipeline") == "model_prediction"
    report = _passed_report(tmp_path)
    assert report["status"] == "passed"
    maturity = result_maturity(
        report, profile=_profile(),
        profile_sources={"pim.clock_period_ns": {"origin": "assumption"}},
        backend="pipeline")
    assert maturity == "model_prediction"
    maturity = result_maturity(
        report, profile=_profile(),
        profile_sources={"pim.clock_period_ns": {"origin": "ceva_measured"}},
        backend="pipeline")
    assert maturity == "validated_in_scope"
    assert result_maturity(
        report, profile=_profile(), profile_sources={
            "pim.clock_period_ns": {"origin": "ceva_measured"}},
        backend="pipeline", used_predictor=True) == "model_prediction"
    assert result_maturity(
        report, profile=_profile(), profile_sources={
            "pim.clock_period_ns": {"origin": "ceva_measured"}},
        backend="pipeline", out_of_scope=True) == "model_prediction"


def test_inconsistent_error_fields_are_rejected(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({
        "sample_id": "s", "dataset_id": "train", "source": "real_hardware",
        "phase": "decode", "kind": "dma", "shape": [1, 16],
        "dtype": "float16", "predicted": 100.0, "observed": 110.0,
        "target": {"profile": "x"}, "error": 5.0,
    }) + "\n")
    with pytest.raises(ValueError, match="复算"):
        load_samples(path)


def test_load_samples_requires_positive_observed(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({
        "sample_id": "s", "dataset_id": "train", "source": "real_hardware",
        "phase": "decode", "kind": "dma", "shape": [1, 16],
        "dtype": "float16", "predicted": 100.0, "observed": 0.0,
        "target": {"profile": "x"},
    }) + "\n")
    with pytest.raises(ValueError, match="observed"):
        load_samples(path)


# ---- 评审 round01：训练拟合、冻结预测与证据门禁 ----

import math


def _artifacts(tmp_path):
    """四个真实产物文件；报告只接受能按字节复核的哈希。"""
    from tune.inputs import file_sha256

    contents = {
        "model_ir": b"model-ir-bytes",
        "placement_sidecar": b"placement-sidecar-bytes",
        "request_trace": b"request-trace-bytes",
        "cost_sidecar": b"cost-sidecar-bytes",
    }
    artifacts = {}
    for role, content in contents.items():
        path = tmp_path / f"{role}.bin"
        path.write_bytes(content)
        artifacts[role] = {"path": str(path), "sha256": file_sha256(path)}
    return artifacts


PROFILE_PATH = Path(__file__).resolve().parent.parent / "contracts" / "profiles" / "ceva_64core_4e.json"


def _profile():
    from contracts.hardware_profile import load_profile

    return load_profile(PROFILE_PATH)


def _package(artifacts):
    return {
        "ir_sha256": artifacts["model_ir"]["sha256"],
        "placement_sha256": artifacts["placement_sidecar"]["sha256"],
        "trace_sha256": artifacts["request_trace"]["sha256"],
        "cost_sidecar": {"used": False},
    }


def _summary(tmp_path, *, producer="pipeline_simulator", total=1.0,
             package=None):
    """真实 pipeline 摘要附带的运行清单；清单绑定本轮产物身份。"""
    package = package or _package(_artifacts(tmp_path))
    manifest = {
        "inputs": {
            "model_ir": {"sha256": package["ir_sha256"]},
            "placement_sidecar": {"sha256": package["placement_sha256"]},
            "request_trace": {"sha256": package["trace_sha256"]},
        },
        "cost_sidecar_consumption": {
            "configured": package["cost_sidecar"]["used"],
            "sha256": package["cost_sidecar"].get("sha256"),
        },
    }
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    path = tmp_path / f"simulation_summary_{producer}.json"
    path.write_text(json.dumps({
        "result_producer": producer,
        "total_time_s": total,
        "simulation_backend": "pipeline",
        "run_manifest": "run_manifest.json",
    }))
    return path


def _scope(profile=None):
    profile = profile or _profile()
    return {
        "target_name": profile.ceva.target_name,
        "engines_per_core": profile.ceva.engines_per_core,
        "core_count": profile.clusters_per_device,
        "used_core_count": profile.clusters_per_device,
        "execution_scope": "npm_core",
        "clock_period_ns": profile.clock_period_ns,
        "dtype": "float16",
        "element_bytes": profile.bytes_per_element,
        "dtype_layout": {"GEMM": "f16"},
        "cost_model": {
            "mac_width": profile.mac_width,
            "tile_size": profile.tile_size,
            "chunk_size": profile.chunk_size,
            "max_bulk_transfer_bytes": profile.max_bulk_transfer_bytes,
            "multipliers": profile.multipliers,
            "cpu_model": dict(profile.cpu_model),
            "loop_encoding": {
                "use_long_loop_counts": profile.use_long_loop_counts,
                "dynamic_loop_primitives": ["LOOP_BEGIN_INDIRECT", "LOOP_END"],
                "fixed_loop_primitives": ["LOOP_BEGIN", "LOOP_END"],
            },
            "cycle_costs": dict(profile.cycle_costs),
            "zone_cycle_factors": dict(profile.zone_cycle_factors),
        },
        "timing_boundary": {"execution_scope": "npm_core"},
        "request_boundary": {"max_context": 256},
        "communication_model": {
            "inter_cluster_communication": profile.inter_cluster_communication,
            "host_relay_overhead_ns": profile.host_relay_overhead_ns,
            "links": [link.to_json() for link in profile.links],
        },
        "shape_range": {"M": [1, 128], "N": [16, 4096], "K": [16, 4096]},
    }


def _thresholds(groups):
    return {f"{group}/{field}": 1.0
            for group in groups for field in ("median", "p90", "max")}


GROUPS = (
    "prefill/compute", "decode/compute", "prefill/dma", "decode/dma",
    "prefill/host", "decode/host", "prefill/communication",
    "decode/communication",
)


def _valid_samples():
    """覆盖计算、DMA、主机与通信两阶段；预测与观测固定为 1:2。"""
    samples = []
    for phase in ("prefill", "decode"):
        for kind, operator, artifact_phase, unit in (
            ("compute", "GEMM", "compute", "ns"),
            ("dma", "GEMM", "load", "cycle"),
            ("host", "ALL_REDUCE", "host", "ns"),
            ("communication", "PIM_Inter_Cluster", "transfer", "ns"),
        ):
            for dataset, prefix, shape in (
                    ("train", "t", (2, 16)), ("heldout", "v", (8, 16))):
                samples.append(_sample(
                    f"{prefix}-{phase}-{kind}", dataset=dataset,
                    phase=phase, kind=kind, shape=shape,
                    predicted=100.0, observed=200.0,
                    unit=unit, operator=operator,
                    artifact_phase=artifact_phase))
    return samples


def _passed_report(tmp_path, **overrides):
    artifacts = _artifacts(tmp_path)
    kwargs = dict(
        training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=_scope(),
        package=_package(artifacts),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path))
    kwargs.update(overrides)
    return build_report(_valid_samples(), **kwargs)


def test_empty_artifacts_and_scope_cannot_pass(tmp_path):
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout",
        thresholds={"median": 1.0, "p90": 1.0, "max": 1.0},
        artifacts={}, scope={}, package={},
        simulation_summary=_summary(tmp_path))
    assert report["status"] == "report_only"
    maturity = result_maturity(
        report, profile=_profile(),
        profile_sources={"pim.clock_period_ns": {"origin": "customer"}},
        backend="pipeline")
    assert maturity == "model_prediction"


def test_fit_is_frozen_from_training_and_applied_to_heldout(tmp_path):
    report = _passed_report(tmp_path)
    # 训练集观测是预测的 2 倍；留出集用冻结系数预测后误差应为 0。
    assert report["fit"]["prefill/compute"]["scale"] == pytest.approx(2.0)
    assert report["statistics"]["prefill/compute"]["median"] == pytest.approx(
        0.0)
    assert len(report["points"]) == 8
    for point in report["points"]:
        assert point["frozen_predicted"] == pytest.approx(200.0)
        assert point["observed"] == pytest.approx(200.0)
        assert point["relative_error"] == pytest.approx(0.0)
    assert report["status"] == "passed"


def test_threshold_must_cover_every_group_and_field(tmp_path):
    report = _passed_report(
        tmp_path, thresholds=_thresholds(["prefill/compute"]))
    assert report["status"] == "report_only"
    assert "阈值" in report["reason"]


def _sources():
    return {
        "pim.clock_period_ns": {"origin": "customer"},
        "ceva.local_memory_bytes": {"origin": "ceva_measured"},
    }


def test_full_evidence_passes_and_returns_validated(tmp_path):
    report = _passed_report(tmp_path)
    assert report["schema_version"] == 11
    # 生成的三类证据必须通过 GeneSim 自己的校验入口。
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    assert ceva_validation.validation_evidence_error(
        report["scope"]["validation"]) is None
    assert ceva_validation.host_evidence_error(
        report["scope"]["host_orchestration"]) is None
    assert ceva_validation.communication_evidence_error(
        report["scope"]["communication_validation"]) is None
    assert result_maturity(report, profile=_profile(),
                           profile_sources=_sources(),
                           backend="pipeline") == "validated_in_scope"
    # 空来源没有升级依据，不能只凭报告文字升级。
    assert result_maturity(
        report, profile=_profile(), profile_sources={},
        backend="pipeline") == "model_prediction"


def test_missing_communication_or_host_evidence_stays_prediction(tmp_path):
    """缺通信或主机测点时报告只能保持模型预测。"""
    samples = [sample for sample in _valid_samples()
               if sample.kind not in ("host", "communication")]
    artifacts = _artifacts(tmp_path)
    report = build_report(
        samples, training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=_scope(), package=_package(artifacts),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path))
    assert report["status"] == "report_only"
    assert any(token in report["reason"]
               for token in ("主机", "通信", "host_orchestration",
                             "communication_validation"))
    assert result_maturity(
        report, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_fake_artifact_hash_or_unbound_backend_cannot_pass(tmp_path):
    artifacts = _artifacts(tmp_path)
    # 哈希格式合法但文件字节被改：必须失败。
    artifacts["model_ir"]["sha256"] = "0" * 64
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=artifacts,
        scope=_scope(), package=_package(_artifacts(tmp_path)),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path))
    assert report["status"] == "report_only"
    assert "哈希" in report["reason"]
    # 没有真实 pipeline 摘要：后端证据不能自报。
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=_artifacts(tmp_path),
        scope=_scope(), package=_package(_artifacts(tmp_path)),
        thresholds=_thresholds(GROUPS))
    assert report["status"] == "report_only"
    assert "摘要" in report["reason"]
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=_artifacts(tmp_path),
        scope=_scope(), package=_package(_artifacts(tmp_path)),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path, producer="predictor_surrogate"))
    assert report["status"] == "report_only"
    assert "pipeline" in report["reason"]


def test_maturity_rejects_missing_points_or_wrong_artifacts(tmp_path):
    report = _passed_report(tmp_path)
    broken = dict(report)
    broken.pop("points")
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    broken = json.loads(json.dumps(report))
    broken["artifacts"]["model_ir"]["sha256"] = "not-a-sha"
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    broken = json.loads(json.dumps(report))
    broken["scope"] = {}
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    broken = json.loads(json.dumps(report))
    broken["backends"]["validation"] = "predictor"
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    # 报告生成后篡改验证样本：成熟度必须复算失败。
    broken = json.loads(json.dumps(report))
    broken["scope"]["validation"]["samples"][0]["observed"] += 1.0
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_non_ceva_profile_never_gets_validated_grade(tmp_path):
    from contracts.hardware_profile import load_profile

    report = _passed_report(tmp_path)
    generic = load_profile(
        Path(__file__).resolve().parent.parent
        / "contracts" / "profiles" / "generic.json")
    assert result_maturity(
        report, profile=generic, profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    assert result_maturity(
        report, profile=None, profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_scope_must_match_the_profile(tmp_path):
    report = _passed_report(tmp_path)
    mismatched = json.loads(json.dumps(report))
    mismatched["scope"]["clock_period_ns"] += 1.0
    assert result_maturity(
        mismatched, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    mismatched = json.loads(json.dumps(report))
    mismatched["scope"]["target_name"] = "another-target"
    assert result_maturity(
        mismatched, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    mismatched = json.loads(json.dumps(report))
    mismatched["scope"]["communication_model"]["links"][0][
        "bandwidth_gbps"] += 1.0
    assert result_maturity(
        mismatched, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_report_schema_matches_genesim_contract():
    """调优侧报告版本必须与 GeneSim 的唯一契约版本一致。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from config_loader import (
        CEVA_VALIDATION_REQUIRED_PACKAGE_KEYS,
        CEVA_VALIDATION_REQUIRED_SCOPE_KEYS,
        CEVA_VALIDATION_SCHEMA_VERSION,
    )

    from tune.calibration import REPORT_SCHEMA_VERSION, _genesim_report_keys

    assert REPORT_SCHEMA_VERSION == CEVA_VALIDATION_SCHEMA_VERSION
    assert _genesim_report_keys() == (
        CEVA_VALIDATION_REQUIRED_PACKAGE_KEYS,
        CEVA_VALIDATION_REQUIRED_SCOPE_KEYS,
    )


def _scale_target(identity, *, scale, count, fields, operator="",
                  artifact_phase=""):
    """构造一条带身份与可写字段的冻结缩放条目。"""
    return {
        "method": "least_squares_through_origin",
        "count": count,
        "scale": scale,
        "fields": list(fields),
        "operator": operator,
        "artifact_phase": artifact_phase,
    }


def test_profile_with_fit_updates_actual_timing_fields():
    profile = _profile()
    intermediate = next(
        link for link in profile.links if link.name == "PIM_Inter_Cluster")
    fit = {
        "prefill/dma": {
            "method": "least_squares_through_origin", "count": 3,
            "targets": {"dma:GEMM": _scale_target(
                "dma:GEMM", scale=2.0, count=3,
                fields=["LOADN", "LOADN_PER_BYTE", "STOREN_PER_BYTE"],
                operator="GEMM", artifact_phase="load")}},
        "prefill/compute": {
            "method": "least_squares_through_origin", "count": 4,
            "targets": {"compute:GEMM|MAC": _scale_target(
                "compute:GEMM|MAC", scale=0.5, count=4, fields=["MAC"],
                operator="GEMM", artifact_phase="MAC")}},
        "decode/host": {
            "method": "least_squares_through_origin", "count": 2,
            "targets": {"host:ALL_REDUCE": _scale_target(
                "host:ALL_REDUCE", scale=3.0, count=2,
                fields=["cpu_model.base_latency_us"],
                operator="ALL_REDUCE", artifact_phase="host")}},
        "decode/communication": {
            "method": "least_squares_through_origin", "count": 2,
            "targets": {"communication:PIM_Inter_Cluster": _scale_target(
                "communication:PIM_Inter_Cluster", scale=1.5, count=2,
                fields=[
                    "comm_config.links.PIM_Inter_Cluster.latency_base_ns"],
                operator="PIM_Inter_Cluster", artifact_phase="transfer")}},
    }
    updated = profile_with_fit(profile.to_json(), fit)
    from contracts.hardware_profile import build_profile

    written = build_profile(updated)
    assert written.cycle_costs["LOADN"] == pytest.approx(
        profile.cycle_costs["LOADN"] * 2.0)
    assert written.cycle_costs["LOADN_PER_BYTE"] == pytest.approx(
        profile.cycle_costs["LOADN_PER_BYTE"] * 2.0)
    assert written.cycle_costs["MAC"] == pytest.approx(
        profile.cycle_costs["MAC"] * 0.5)
    assert written.cycle_costs["STOREN_PER_BYTE"] == pytest.approx(
        profile.cycle_costs["STOREN_PER_BYTE"] * 2.0)
    assert written.cpu_model["base_latency_us"] == pytest.approx(
        profile.cpu_model["base_latency_us"] * 3.0)
    written_intermediate = next(
        link for link in written.links if link.name == "PIM_Inter_Cluster")
    assert written_intermediate.latency_base_ns == pytest.approx(
        intermediate.latency_base_ns * 1.5)
    # 未参与拟合的字段保持原值：单点 compute 不能改 SILU，单链路不能改
    # 其他链路，未测的主机参数也不能被顺带缩放。
    assert written.clock_period_ns == profile.clock_period_ns
    assert written.cycle_costs["SILU"] == profile.cycle_costs["SILU"]
    assert written.cycle_costs["REDUCE"] == profile.cycle_costs["REDUCE"]
    assert written.cpu_model["peak_flops"] == profile.cpu_model["peak_flops"]
    other_links = {
        link.name: link for link in profile.links
        if link.name != "PIM_Inter_Cluster"}
    for link in written.links:
        if link.name in other_links:
            assert link.latency_base_ns == pytest.approx(
                other_links[link.name].latency_base_ns)
    # 证据写回到实际受影响的字段，而不是只写一个组标签。
    assert written.parameter_sources[
        "pim.pim_trace_cycle_costs.LOADN"]["origin"] == "fitted"
    assert written.parameter_sources[
        "cpu_model.base_latency_us"]["origin"] == "fitted"
    assert written.parameter_sources[
        "comm_config.links.PIM_Inter_Cluster.latency_base_ns"][
            "origin"] == "fitted"
    assert "pim.pim_trace_cycle_costs.SILU" not in written.parameter_sources
    assert "pim.pim_trace_cycle_costs.SILU" not in profile.parameter_sources


def test_dma_features_reuse_genesim_relative_error_fit(tmp_path):
    """带搬运特征的样本复用 GeneSim 既有拟合，并写回真实周期字段。"""

    def read(sample_id, dataset, count, nbytes, cycles):
        return _sample(
            sample_id, dataset=dataset, kind="dma", predicted=1.0,
            observed=cycles, unit="cycle", operator="GEMM",
            artifact_phase="load",
            features={"operator": "MRAM_READ", "transfer_count": count,
                      "total_bytes": nbytes})

    def write(sample_id, dataset, count, nbytes, cycles):
        return _sample(
            sample_id, dataset=dataset, kind="dma", predicted=1.0,
            observed=cycles, unit="cycle", operator="GEMM",
            artifact_phase="store",
            features={"operator": "MRAM_WRITE", "transfer_count": count,
                      "total_bytes": nbytes})

    # 读：cycles = 10*count + 2*bytes；写：cycles = 5*count + 1*bytes。
    samples = [
        read("r0", "train", 2, 100, 220),
        read("r1", "train", 4, 50, 140),
        read("r2", "train", 1, 200, 410),
        write("w0", "train", 2, 100, 110),
        write("w1", "train", 4, 50, 70),
        write("w2", "train", 1, 200, 205),
        read("rv", "heldout", 3, 150, 330),
        write("wv", "heldout", 3, 150, 165),
    ]
    artifacts = _artifacts(tmp_path)
    report = build_report(
        samples, training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=_scope(), package=_package(artifacts),
        thresholds=_thresholds(["prefill/dma"]),
        simulation_summary=_summary(tmp_path))
    fit = report["fit"]["prefill/dma"]
    assert fit["method"] == "genesim_relative_error_wls"
    assert fit["dma_fit"]["MRAM_READ"]["base_cycles_per_transfer"] == (
        pytest.approx(10.0, rel=1e-9))
    assert fit["dma_fit"]["MRAM_READ"]["cycles_per_byte"] == (
        pytest.approx(2.0, rel=1e-9))
    # 留出集预测来自冻结的搬运模型，而不是样本自带的 predicted。
    assert all(point["relative_error"] == pytest.approx(0.0)
               for point in report["points"])
    updated = profile_with_fit(_profile().to_json(), {"prefill/dma": fit})
    from contracts.hardware_profile import build_profile

    written = build_profile(updated)
    assert written.cycle_costs["LOADN"] == pytest.approx(10.0)
    assert written.cycle_costs["LOADN_PER_BYTE"] == pytest.approx(2.0)
    assert written.cycle_costs["STOREN"] == pytest.approx(5.0)
    assert written.cycle_costs["STOREN_PER_BYTE"] == pytest.approx(1.0)


# ---- 评审 round03：共享计时字段的联合拟合与单位契约 ----

def test_shared_timing_fields_use_one_joint_phase_fit():
    from tune.calibration import fit_parameters, predict_with_frozen_parameters

    train = [
        _sample("t0", phase="prefill", predicted=100.0, observed=200.0,
                operator="GEMM", artifact_phase="MAC"),
        _sample("t1", phase="decode", predicted=100.0, observed=300.0,
                operator="GEMM", artifact_phase="MAC"),
    ]
    heldout = [
        _sample("v0", phase="prefill", dataset="heldout",
                predicted=100.0, observed=250.0,
                operator="GEMM", artifact_phase="MAC"),
        _sample("v1", phase="decode", dataset="heldout",
                predicted=100.0, observed=250.0,
                operator="GEMM", artifact_phase="MAC"),
    ]
    fit = fit_parameters(train)
    joint = sum(s.predicted * s.observed for s in train) / sum(
        s.predicted ** 2 for s in train)
    assert fit["prefill/compute"]["scale"] == pytest.approx(joint)
    assert fit["decode/compute"]["scale"] == pytest.approx(joint)
    frozen = predict_with_frozen_parameters(heldout, fit)
    assert [point.predicted for point in frozen] == pytest.approx([250.0, 250.0])

    profile = _profile()
    updated = profile_with_fit(profile.to_json(), fit)
    observed = updated["cycle_costs"]["MAC"] / profile.cycle_costs["MAC"]
    assert observed == pytest.approx(joint)
    # 阶段专属拟合不能连乘：2 × 3 = 6 的伪校准必须消失。
    assert observed != pytest.approx(6.0)
    # 未测的 SILU 保留原值，不能被同一缩放顺带改掉。
    assert updated["cycle_costs"]["SILU"] == profile.cycle_costs["SILU"]


def test_conflicting_phase_fits_are_rejected_instead_of_multiplied():
    fit = {
        "prefill/compute": {"scale": 2.0, "count": 1,
                            "method": "least_squares_through_origin"},
        "decode/compute": {"scale": 3.0, "count": 1,
                           "method": "least_squares_through_origin"},
    }
    with pytest.raises(ValueError, match="联合样本"):
        profile_with_fit(_profile().to_json(), fit)


def test_dma_nanosecond_samples_are_rejected_before_fitting():
    from tune.calibration import fit_parameters

    sample = _sample(
        "d0", kind="dma", observed=220.0, unit="ns",
        features={"operator": "MRAM_READ", "transfer_count": 2,
                  "total_bytes": 100})
    with pytest.raises(ValueError, match="cycle"):
        fit_parameters([sample])


def test_generated_scope_passes_scheduler_artifact_binding(tmp_path):
    """同一份报告在 GeneSim 的真实产物校验入口上也应通过（独立进程）。"""
    import os
    import subprocess
    import sys

    from genesim_bridge.paths import genesim_root

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_passed_report(tmp_path)))
    script = tmp_path / "check_scope.py"
    script.write_text(
        "import json, sys\n"
        "from scheduler.gene_sim_scheduler import GeneSimScheduler\n"
        "report = json.load(open(sys.argv[1]))\n"
        "phases = set(sys.argv[2].split(','))\n"
        "scheduler = object.__new__(GeneSimScheduler)\n"
        "scheduler._ceva_run_compute_dtypes = lambda: ("
        "{'GEMM': 'f16'}, {'f16': 4})\n"
        "scheduler._ceva_artifact_phases = lambda: {'GEMM': phases}\n"
        "scheduler._ceva_sample_shape_error = lambda *a, **k: None\n"
        "error = scheduler._ceva_validation_evidence_error("
        "report['scope'], {}, {})\n"
        "print('OK' if error is None else 'ERR:' + error)\n")
    root = genesim_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get(
        "PYTHONPATH", "")
    passed = subprocess.run(
        [sys.executable, str(script), str(report_path), "compute,load,store"],
        cwd=str(root), env=env, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert passed.stdout.strip() == "OK", passed.stdout
    # 产物里没有该相位时，同一个入口必须拒绝。
    rejected = subprocess.run(
        [sys.executable, str(script), str(report_path), "compute"],
        cwd=str(root), env=env, capture_output=True, text=True)
    assert rejected.returncode == 0, rejected.stderr
    assert "ERR:独立验证样本" in rejected.stdout and "相位" in rejected.stdout


# ---- 评审 round04：报告外层证据必须可复盘 ----

def test_report_saves_thresholds_training_inputs_and_refits(tmp_path):
    """报告落盘阈值与训练集拟合输入，成熟度复核时重算拟合。"""
    report = _passed_report(tmp_path)
    assert report["status"] == "passed"
    assert report["thresholds"] == _thresholds(GROUPS)
    assert len(report["training_points"]) == 8
    assert {point["dataset_id"] for point in report["training_points"]} == {
        "train"}
    # 篡改冻结参数：无法由训练集重算，只能保持模型预测。
    broken = json.loads(json.dumps(report))
    broken["fit"]["prefill/compute"]["scale"] = 1.5
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    # 篡改统计量同样必须被复算拒绝。
    broken = json.loads(json.dumps(report))
    broken["statistics"]["prefill/compute"]["median"] = 0.5
    assert result_maturity(
        broken, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_status_flip_without_thresholds_stays_prediction(tmp_path):
    """没有用户阈值的报告即使把状态改成 passed 也不能升级。"""
    report = _passed_report(tmp_path, thresholds={})
    assert report["status"] == "report_only"
    report["status"] = "passed"
    assert result_maturity(
        report, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    # 阈值覆盖不全同样不能只靠状态升级。
    incomplete = _passed_report(
        tmp_path, thresholds=_thresholds(["prefill/compute"]))
    incomplete["status"] = "passed"
    assert result_maturity(
        incomplete, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_fabricated_sample_source_cannot_be_validated(tmp_path):
    """逐点来源必须来自独立观测；伪造来源不能升级。"""
    report = _passed_report(tmp_path)
    changed = json.loads(json.dumps(report))
    changed["points"][0]["source"] = "fabricated"
    assert result_maturity(
        changed, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    changed = json.loads(json.dumps(report))
    changed["training_points"][0]["source"] = "fabricated"
    assert result_maturity(
        changed, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    # 留出样本混入训练数据集身份也不能升级。
    changed = json.loads(json.dumps(report))
    changed["points"][0]["dataset_id"] = "train"
    assert result_maturity(
        changed, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_out_of_range_sample_shape_cannot_be_validated(tmp_path):
    """声明的形状范围必须覆盖实际验证样本，而不是只检查格式。"""
    report = _passed_report(tmp_path)
    changed = json.loads(json.dumps(report))
    changed["scope"]["shape_range"]["M"] = [1, 1]
    assert result_maturity(
        changed, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"
    # 合法范围内的小写轴名继续可用。
    lower = _passed_report(
        tmp_path, scope=dict(_scope(), shape_range={
            "m": [1, 128], "n": [16, 4096], "k": [16, 4096]}))
    assert lower["status"] == "passed"
    assert result_maturity(
        lower, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "validated_in_scope"


def test_unrelated_two_field_summary_cannot_prove_backend(tmp_path):
    """只有 result_producer/total_time_s 的无关 JSON 不是运行归属。"""
    unrelated = tmp_path / "unrelated-summary.json"
    unrelated.write_text(json.dumps({
        "result_producer": "pipeline_simulator", "total_time_s": 999.0}))
    report = _passed_report(tmp_path, simulation_summary=unrelated)
    assert report["status"] == "report_only"
    assert "运行清单" in report["reason"]


def test_run_manifest_must_bind_the_package_hashes(tmp_path):
    """运行清单里的 IR/放置/trace/成本身份必须与 package 一致。"""
    artifacts = _artifacts(tmp_path)
    package = _package(artifacts)
    summary = _summary(tmp_path, package=package)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["request_trace"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=artifacts, scope=_scope(),
        package=package, thresholds=_thresholds(GROUPS),
        simulation_summary=summary)
    assert report["status"] == "report_only"
    assert "请求轨迹" in report["reason"]
    # 成本 sidecar 使用登记也必须与运行清单一致。
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"]["request_trace"]["sha256"] = package["trace_sha256"]
    manifest["cost_sidecar_consumption"]["configured"] = True
    manifest_path.write_text(json.dumps(manifest))
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=artifacts, scope=_scope(),
        package=package, thresholds=_thresholds(GROUPS),
        simulation_summary=summary)
    assert report["status"] == "report_only"
    assert "成本" in report["reason"]


# ---- 评审 round05：非 GEMM 形状也必须落在声明的适用范围 ----

def _operator_report(tmp_path, operator, declared_shapes=None):
    """把计算/DMA 样本换成目标算子，声明逐算子执行形状。"""
    artifacts = _artifacts(tmp_path)
    scope = _scope()
    scope["dtype_layout"] = {operator: "f16"}
    shape_range = dict(scope["shape_range"])
    if declared_shapes is not None:
        shape_range["operator_shapes"] = {operator: declared_shapes}
    scope["shape_range"] = shape_range
    samples = []
    target = _target(scope)
    for sample in _valid_samples():
        if sample.kind in ("compute", "dma"):
            samples.append(replace(sample, operator=operator, target=target))
        else:
            samples.append(replace(sample, target=target))
    return build_report(
        samples,
        training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=scope, package=_package(artifacts),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path, package=_package(artifacts)))


@pytest.mark.parametrize("operator", ["SILU", "RMSNORM"])
@pytest.mark.parametrize("kind", ["compute", "dma"])
def test_non_gemm_sample_without_declared_shape_stays_prediction(
        tmp_path, operator, kind):
    """缺少逐算子形状声明时不能跳过检查，报告只能保持模型预测。"""
    case = tmp_path / f"{operator}-{kind}"
    case.mkdir()
    artifacts = _artifacts(case)
    scope = _scope()
    scope["dtype_layout"] = {"GEMM": "f16", operator: "f16"}
    samples = []
    target = _target(scope)
    for sample in _valid_samples():
        if sample.kind == kind:
            samples.append(replace(sample, operator=operator, target=target))
        else:
            samples.append(replace(sample, target=target))
    report = build_report(
        samples, training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=scope, package=_package(artifacts),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(case, package=_package(artifacts)))
    assert report["status"] == "report_only"
    assert operator in report["reason"] or "形状" in report["reason"]


@pytest.mark.parametrize("operator", ["SILU", "RMSNORM"])
def test_non_gemm_sample_inside_declared_shapes_can_pass(tmp_path, operator):
    """声明了逐算子执行形状且样本在其中时，其余证据齐全可以升级。"""
    report = _operator_report(
        tmp_path, operator, declared_shapes=[[2, 16], [8, 16]])
    assert report["status"] == "passed", report.get("reason")
    maturity = result_maturity(
        report, profile=_profile(), profile_sources=_sources(),
        backend="pipeline")
    assert maturity == "validated_in_scope"


@pytest.mark.parametrize("operator", ["SILU", "RMSNORM"])
def test_non_gemm_sample_outside_declared_shapes_stays_prediction(
        tmp_path, operator):
    """逐算子形状声明不含训练/留出形状时，报告不得升级。"""
    report = _operator_report(
        tmp_path, operator, declared_shapes=[[2, 16]])
    assert report["status"] == "report_only"
    assert "形状" in report["reason"]


def test_gemm_shape_range_contract_is_unchanged(tmp_path):
    """GEMM 仍按 M/N/K 轴核对；越界样本保持模型预测（合法对照）。"""
    scope = _scope()
    scope["shape_range"] = {"M": [1, 1], "N": [1, 1], "K": [1, 1]}
    artifacts = _artifacts(tmp_path)
    report = build_report(
        _valid_samples(), training_dataset="train",
        validation_dataset="heldout", artifacts=artifacts, scope=scope,
        package=_package(artifacts), thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path, package=_package(artifacts)))
    assert report["status"] == "report_only"
    assert "M" in report["reason"] or "形状" in report["reason"]


# ---- 评审 round06：GeneSim 共享契约必须覆盖产物与后端证据 ----

def test_genesim_contract_requires_artifacts_and_pipeline_backend(tmp_path):
    """缺产物或 predictor 后端的报告在 GeneSim 唯一入口上不能复核通过。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    report = _passed_report(tmp_path)
    assert report["status"] == "passed"
    assert ceva_validation.report_evidence_error(
        report, require_host_and_communication=True) is None

    without_artifacts = json.loads(json.dumps(report))
    without_artifacts.pop("artifacts")
    assert ceva_validation.report_evidence_error(
        without_artifacts, require_host_and_communication=True) is not None

    predictor = json.loads(json.dumps(report))
    predictor["backends"] = {
        "validation": "predictor",
        "source": {"path": "x", "sha256": "0" * 64},
    }
    assert ceva_validation.report_evidence_error(
        predictor, require_host_and_communication=True) is not None

    forged = json.loads(json.dumps(report))
    forged["backends"]["source"]["sha256"] = "0" * 64
    assert ceva_validation.report_evidence_error(
        forged, require_host_and_communication=True) is not None


# ---- 评审 round06：写回范围必须由训练集实际测点决定 ----

def test_mixed_compute_identities_are_written_separately():
    """两个原语各自测点分别写回，未测字段保持原值。"""
    profile = _profile()
    fit = {
        "prefill/compute": {
            "method": "least_squares_through_origin", "count": 2,
            "targets": {
                "compute:GEMM|MAC": _scale_target(
                    "compute:GEMM|MAC", scale=2.0, count=1,
                    fields=["MAC"], operator="GEMM",
                    artifact_phase="MAC"),
                "compute:SILU|SILU": _scale_target(
                    "compute:SILU|SILU", scale=3.0, count=1,
                    fields=["SILU"], operator="SILU",
                    artifact_phase="SILU"),
            }},
    }
    written = profile_with_fit(profile.to_json(), fit)
    assert written["cycle_costs"]["MAC"] == pytest.approx(
        profile.cycle_costs["MAC"] * 2.0)
    assert written["cycle_costs"]["SILU"] == pytest.approx(
        profile.cycle_costs["SILU"] * 3.0)
    assert written["cycle_costs"]["MUL"] == profile.cycle_costs["MUL"]
    assert written["parameter_sources"][
        "pim.pim_trace_cycle_costs.MAC"]["origin"] == "fitted"
    assert written["parameter_sources"].get(
        "pim.pim_trace_cycle_costs.MUL") is None


def test_single_dma_direction_does_not_touch_the_other():
    """只测 MRAM_READ 时不得改写 STOREN 等写方向字段。"""
    profile = _profile()
    fit = {
        "prefill/dma": {
            "method": "genesim_relative_error_wls", "count": 2,
            "targets": {"dma:MRAM_READ": {
                "method": "genesim_relative_error_wls", "count": 2,
                "fields": ["LOADN", "LOADN_PER_BYTE"],
                "operator": "GEMM", "artifact_phase": "load",
                "dma_fit": {"MRAM_READ": {
                    "base_cycles_per_transfer": 7.0,
                    "cycles_per_byte": 0.5,
                    "sample_count": 2}},
            }},
        },
    }
    written = profile_with_fit(profile.to_json(), fit)
    assert written["cycle_costs"]["LOADN"] == pytest.approx(7.0)
    assert written["cycle_costs"]["LOADN_PER_BYTE"] == pytest.approx(0.5)
    assert written["cycle_costs"]["STOREN"] == profile.cycle_costs["STOREN"]
    assert written["cycle_costs"]["STOREN_PER_BYTE"] == (
        profile.cycle_costs["STOREN_PER_BYTE"])


def test_uncovered_heldout_identity_keeps_report_only(tmp_path):
    """留出测点身份不在训练集时，报告不得生成 passed。"""
    samples = [
        _sample("t", operator="GEMM", artifact_phase="MAC"),
        _sample("v", dataset="heldout", operator="SILU",
                artifact_phase="SILU"),
    ]
    artifacts = _artifacts(tmp_path)
    report = build_report(
        samples, training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=_scope(), package=_package(artifacts),
        thresholds=_thresholds(["prefill/compute"]),
        simulation_summary=_summary(tmp_path))
    assert report["status"] == "report_only"
    assert "覆盖" in report["reason"] or "身份" in report["reason"]


def test_identity_without_writable_field_refuses_write_back():
    """测点没有可归属字段时拒绝写回，不能顺带缩放整个类别。"""
    profile = _profile()
    fit = {
        "prefill/compute": {
            "method": "least_squares_through_origin", "count": 1,
            "targets": {"compute:GEMM|": _scale_target(
                "compute:GEMM|", scale=2.0, count=1, fields=[],
                operator="GEMM")},
        },
    }
    with pytest.raises(ValueError, match="没有写回"):
        profile_with_fit(profile.to_json(), fit)


def test_calibrate_script_refuses_profile_write_after_failed_report(tmp_path):
    """报告未通过时校准入口不得把拟合结果写回画像。"""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    samples = tmp_path / "samples.jsonl"
    rows = []
    for dataset, predicted, observed, shape in (
            ("train", 100.0, 200.0, [2, 16]),
            ("heldout", 100.0, 200.0, [4, 16])):
        rows.append({
            "sample_id": f"{dataset}-0", "dataset_id": dataset,
            "source": "ceva_simulator", "phase": "prefill",
            "kind": "compute", "shape": shape, "dtype": "float16",
            "predicted": predicted, "observed": observed,
            "target": _target(), "unit": "ns",
            "operator": "GEMM", "artifact_phase": "MAC",
        })
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = tmp_path / "report.json"
    profile_out = tmp_path / "profile.json"
    proc = subprocess.run(
        [sys.executable, "scripts/calibrate_profile.py",
         "--samples", str(samples), "--training-dataset", "train",
         "--validation-dataset", "heldout", "--profile", str(PROFILE_PATH),
         "--profile-out", str(profile_out), "--out", str(report)],
        cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    assert not profile_out.exists()
    assert json.loads(report.read_text())["status"] == "report_only"


def test_calibrate_script_refuses_write_back_for_cross_target_samples(
        tmp_path):
    """训练目标与留出集不一致时，校准入口不得写回画像。"""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    samples = tmp_path / "samples.jsonl"
    rows = []
    for dataset, shape in (("train", [2, 16]), ("heldout", [4, 16])):
        rows.append({
            "sample_id": f"{dataset}-0", "dataset_id": dataset,
            "source": "ceva_simulator", "phase": "prefill",
            "kind": "compute", "shape": shape, "dtype": "float16",
            "predicted": 100.0, "observed": 200.0,
            "target": (
                _other_target() if dataset == "train" else _target()),
            "unit": "ns", "operator": "GEMM", "artifact_phase": "MAC",
        })
    samples.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    scope = tmp_path / "scope.json"
    scope.write_text(json.dumps(_scope()))
    report = tmp_path / "report.json"
    profile_out = tmp_path / "profile.json"
    proc = subprocess.run(
        [sys.executable, "scripts/calibrate_profile.py",
         "--samples", str(samples), "--training-dataset", "train",
         "--validation-dataset", "heldout", "--profile", str(PROFILE_PATH),
         "--scope", str(scope),
         "--profile-out", str(profile_out), "--out", str(report)],
        cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    assert not profile_out.exists()
    payload = json.loads(report.read_text())
    assert payload["status"] == "report_only"
    assert "目标" in payload["reason"]


def test_unknown_primitive_field_is_skipped_without_extrapolation():
    """原语身份无法归属时跳过该身份，其他已覆盖字段仍正常写回。"""
    profile = _profile()
    fit = {
        "prefill/compute": {
            "method": "least_squares_through_origin", "count": 1,
            "targets": {"compute:GEMM|tt.dot": _scale_target(
                "compute:GEMM|tt.dot", scale=2.0, count=1,
                fields=["TT.DOT"], operator="GEMM",
                artifact_phase="tt.dot")},
        },
        "prefill/host": {
            "method": "least_squares_through_origin", "count": 1,
            "targets": {"host:MODEL_INPUT": _scale_target(
                "host:MODEL_INPUT", scale=2.0, count=1,
                fields=["cpu_model.base_latency_us"],
                operator="MODEL_INPUT", artifact_phase="host")},
        },
    }
    written = profile_with_fit(profile.to_json(), fit)
    assert written["cpu_model"]["base_latency_us"] == pytest.approx(
        profile.cpu_model["base_latency_us"] * 2.0)
    # 无法归属的 compute 身份不改变任何周期字段。
    assert written["cycle_costs"] == profile.to_json()["cycle_costs"]


def test_multi_identity_phase_entries_are_accepted_as_shared_fit():
    """多身份条目没有顶层 scale 时，prefill/decode 仍应视为同一组参数。"""
    from tune.calibration import fit_parameters

    train = [
        _sample("t0", phase="prefill", operator="MODEL_INPUT",
                artifact_phase="host", kind="host", observed=200.0),
        _sample("t1", phase="decode", operator="MODEL_INPUT",
                artifact_phase="host", kind="host", observed=200.0),
        _sample("t2", phase="prefill", operator="MODEL_OUTPUT",
                artifact_phase="host", kind="host", observed=200.0),
        _sample("t3", phase="decode", operator="MODEL_OUTPUT",
                artifact_phase="host", kind="host", observed=200.0),
    ]
    fit = fit_parameters(train)
    profile = _profile()
    updated = profile_with_fit(profile.to_json(), fit)
    # 两个主机类型的缩放系数一致，全局字段只写回一次。
    assert updated["cpu_model"]["base_latency_us"] == pytest.approx(
        profile.cpu_model["base_latency_us"] * 2.0)


# ---- 评审 round07：共享契约按完整测点身份排除训练/留出改名两用 ----

def _forged_renamed_holdout_report(report):
    """把一个留出点替换成训练点的改名副本，并同步嵌套证据。"""
    import copy

    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    training = next(
        record for record in report["training_points"]
        if record["phase"] == "prefill" and record["kind"] == "compute")
    duplicate = copy.deepcopy(training)
    duplicate["sample_id"] = "renamed-heldout-copy"
    duplicate["dataset_id"] = report["validation_dataset"]
    duplicate["frozen_predicted"] = ceva_validation.predict_point(
        duplicate, report["fit"]["prefill/compute"])
    duplicate["error"] = duplicate["observed"] - duplicate["frozen_predicted"]
    duplicate["relative_error"] = (
        abs(duplicate["error"]) / duplicate["observed"])
    index = next(
        position for position, record in enumerate(report["points"])
        if record["phase"] == "prefill" and record["kind"] == "compute")
    report["points"][index] = duplicate
    nested = next(
        sample for sample in report["scope"]["validation"]["samples"]
        if sample["stage"] == "prefill" and sample["kind"] == "compute")
    nested.update({
        "shape": list(duplicate["shape"]),
        "predicted": duplicate["frozen_predicted"],
        "observed": duplicate["observed"],
        "error": duplicate["error"],
        "relative_error": duplicate["relative_error"],
    })
    return report


def test_shared_contract_rejects_renamed_training_point(tmp_path):
    """改名副本混入留出集时，GeneSim 共享契约与成熟度都必须拒绝。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    report = _passed_report(tmp_path)
    assert ceva_validation.report_evidence_error(
        report, require_host_and_communication=True) is None
    forged = _forged_renamed_holdout_report(json.loads(json.dumps(report)))
    error = ceva_validation.report_evidence_error(
        forged, require_host_and_communication=True)
    assert error is not None
    assert "改名两用" in error
    assert result_maturity(
        forged, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def test_shared_contract_accepts_different_shapes_for_the_same_identity(
        tmp_path):
    """同算子/阶段的合法不同形状不能误判成改名两用。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    report = _passed_report(tmp_path)
    shapes = {(point["phase"], point["kind"]): point["shape"]
              for point in report["training_points"]}
    heldout_shapes = {(point["phase"], point["kind"]): point["shape"]
                      for point in report["points"]}
    # 训练与留出覆盖同一算子/阶段，但形状不同；这是合法独立观测。
    assert shapes[("prefill", "compute")] != heldout_shapes[
        ("prefill", "compute")]
    assert ceva_validation.report_evidence_error(
        report, require_host_and_communication=True) is None
    assert result_maturity(
        report, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "validated_in_scope"


# ---- 评审 round08：训练测点必须与留出集和报告适用范围兼容 ----

def _require_rejected(report, *, token):
    """共享契约与成熟度都必须拒绝跨目标/跨精度/跨单位的报告。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    error = ceva_validation.report_evidence_error(
        report, require_host_and_communication=True)
    assert error is not None and token in error
    assert result_maturity(
        report, profile=_profile(), profile_sources=_sources(),
        backend="pipeline") == "model_prediction"


def _generated_report(tmp_path, samples):
    artifacts = _artifacts(tmp_path)
    return build_report(
        samples, training_dataset="train", validation_dataset="heldout",
        artifacts=artifacts, scope=_scope(), package=_package(artifacts),
        thresholds=_thresholds(GROUPS),
        simulation_summary=_summary(tmp_path))


def test_training_target_must_match_the_heldout_target(tmp_path):
    """训练点改成另一目标配置时不能生成 passed 或升级成熟度。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    for record in forged["training_points"]:
        record["target"] = _other_target()
    _require_rejected(forged, token="目标")

    samples = [
        replace(sample, target=_other_target())
        if sample.dataset_id == "train" else sample
        for sample in _valid_samples()]
    built = _generated_report(tmp_path, samples)
    assert built["status"] == "report_only"
    assert "目标" in built["reason"]


def test_all_points_moved_to_another_target_cannot_pass(tmp_path):
    """训练与留出同步换成另一目标配置时，整份报告仍不能升级。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    for record in [*forged["training_points"], *forged["points"]]:
        record["target"] = _other_target()
    _require_rejected(forged, token="target_name")

    samples = [
        replace(sample, target=_other_target())
        for sample in _valid_samples()]
    built = _generated_report(tmp_path, samples)
    assert built["status"] == "report_only"
    assert "target_name" in built["reason"]


def test_target_label_without_config_cannot_prove_applicability(tmp_path):
    """任意标签（如 profile 名）不能充当目标配置身份。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    for record in [*forged["training_points"], *forged["points"]]:
        record["target"] = {"profile": "ceva"}
    _require_rejected(forged, token="目标配置")


def test_matching_label_but_different_cost_model_cannot_pass(tmp_path):
    """目标标签相同、成本模型字段不同时仍不是同一目标配置。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    for record in [*forged["training_points"], *forged["points"]]:
        record["target"]["cost_model"]["mac_width"] = 16
    _require_rejected(forged, token="cost_model")


def test_mixed_training_targets_cannot_pass(tmp_path):
    """训练集内部混入另一目标配置时也不能联合拟合并升级。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    forged["training_points"][0]["target"] = _other_target()
    _require_rejected(forged, token="目标")


def test_training_dtype_must_match_the_scope_layout(tmp_path):
    """训练精度与报告适用范围的逐算子布局不一致时不能升级。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    for record in forged["training_points"]:
        record["dtype"] = "float32"
    _require_rejected(forged, token="数据类型")

    samples = [
        replace(sample, dtype="float32")
        if sample.dataset_id == "train" else sample
        for sample in _valid_samples()]
    built = _generated_report(tmp_path, samples)
    assert built["status"] == "report_only"
    assert "数据类型" in built["reason"]


def test_sample_unit_must_match_the_measurement_kind(tmp_path):
    """DMA 测点必须用周期单位，主机与计算测点必须用纳秒。"""
    report = _passed_report(tmp_path)
    forged = json.loads(json.dumps(report))
    point = next(record for record in forged["points"]
                 if record["kind"] == "dma")
    point["unit"] = "ns"
    _require_rejected(forged, token="单位")


# ---- 评审 round07：共享计时字段必须按最终画像字段一次性写回 ----

def _shared_mac_fit(scales):
    """构造多个计算身份共同写回 MAC 的冻结缩放条目。"""
    targets = {}
    for index, (operator, scale) in enumerate(scales.items()):
        targets[f"compute:{operator}|MAC"] = {
            "method": "least_squares_through_origin", "count": 1,
            "scale": scale, "fields": ["MAC"],
            "operator": operator, "artifact_phase": "MAC",
        }
    return {"prefill/compute": {
        "method": "least_squares_through_origin",
        "count": len(targets), "targets": targets}}


def test_shared_mac_field_is_written_once_for_the_same_scale():
    """GEMM 与 GEMV 同为 2 倍时，MAC 只能按原值写回一次。"""
    profile = _profile()
    written = profile_with_fit(
        profile.to_json(), _shared_mac_fit({"GEMM": 2.0, "GEMV": 2.0}))
    assert written["cycle_costs"]["MAC"] == pytest.approx(
        profile.cycle_costs["MAC"] * 2.0)
    source = written["parameter_sources"][
        "pim.pim_trace_cycle_costs.MAC"]
    assert source["origin"] == "fitted"
    assert "GEMM" in source["evidence"]
    assert "GEMV" in source["evidence"]
    # 未测字段保持原值，GeneSim 读到的生效值也只缩放一次。
    assert written["cycle_costs"]["SILU"] == profile.cycle_costs["SILU"]
    from contracts.hardware_profile import build_profile

    overrides = build_profile(written).genesim_overrides()
    assert overrides["pim"]["pim_trace_cycle_costs"]["MAC"] == pytest.approx(
        profile.cycle_costs["MAC"] * 2.0)


def test_conflicting_scales_for_a_shared_field_are_rejected():
    """同一共享字段出现不同系数时必须拒绝，不能连乘或后写覆盖。"""
    with pytest.raises(ValueError, match="共享"):
        profile_with_fit(
            _profile().to_json(), _shared_mac_fit({"GEMM": 2.0, "GEMV": 3.0}))


def test_multi_phase_shared_field_is_applied_once():
    """prefill/decode 的联合参数只能写回一次共享字段。"""
    fit = _shared_mac_fit({"GEMM": 2.0})
    joint = json.loads(json.dumps(fit["prefill/compute"]))
    fit["decode/compute"] = joint
    profile = _profile()
    written = profile_with_fit(profile.to_json(), fit)
    assert written["cycle_costs"]["MAC"] == pytest.approx(
        profile.cycle_costs["MAC"] * 2.0)


def _communication_fit(scales):
    """两个通信身份测同一条链路；身份键不同但写回字段相同。"""
    targets = {}
    for index, (key, (operator, scale)) in enumerate(scales.items()):
        targets[f"communication:{key}"] = {
            "method": "least_squares_through_origin", "count": 1,
            "scale": scale,
            "fields": [f"comm_config.links.{operator}.latency_base_ns"],
            "operator": operator, "artifact_phase": "transfer",
        }
    return {"prefill/communication": {
        "method": "least_squares_through_origin",
        "count": len(targets), "targets": targets}}


def test_shared_link_field_uses_one_frozen_scale():
    """同一链路的多个测点同值只写一次，冲突值必须拒绝。"""
    profile = _profile()
    written = profile_with_fit(profile.to_json(), _communication_fit(
        {"a": ("HOST_to_PIM", 2.0), "b": ("HOST_to_PIM", 2.0)}))
    links = {link["name"]: link for link in written["links"]}
    originals = {link.name: link for link in profile.links}
    assert links["HOST_to_PIM"]["latency_base_ns"] == pytest.approx(
        originals["HOST_to_PIM"].latency_base_ns * 2.0)
    source = written["parameter_sources"][
        "comm_config.links.HOST_to_PIM.latency_base_ns"]
    assert source["origin"] == "fitted"
    assert "communication:a" in source["evidence"]
    assert "communication:b" in source["evidence"]
    with pytest.raises(ValueError, match="共享"):
        profile_with_fit(profile.to_json(), _communication_fit(
            {"a": ("HOST_to_PIM", 2.0), "b": ("HOST_to_PIM", 3.0)}))


def _dma_fit(coefficients):
    targets = {}
    for index, (phase, coefficient) in enumerate(coefficients.items()):
        targets[f"dma:GEMM|{phase}"] = {
            "method": "genesim_relative_error_wls", "count": 1,
            "fields": ["LOADN", "LOADN_PER_BYTE"],
            "operator": "GEMM", "artifact_phase": phase,
            "dma_fit": {"MRAM_READ": {
                "base_cycles_per_transfer": coefficient,
                "cycles_per_byte": coefficient / 10.0,
                "sample_count": 1}},
        }
    return {"prefill/dma": {
        "method": "genesim_relative_error_wls", "count": len(targets),
        "targets": targets}}


def test_dma_identities_sharing_a_field_use_one_frozen_value():
    """两个 DMA 身份写同一字段时同值只写一次，冲突必须拒绝。"""
    profile = _profile()
    written = profile_with_fit(profile.to_json(), _dma_fit(
        {"load": 7.0, "store": 7.0}))
    assert written["cycle_costs"]["LOADN"] == pytest.approx(7.0)
    assert written["cycle_costs"]["LOADN_PER_BYTE"] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="共享"):
        profile_with_fit(profile.to_json(), _dma_fit(
            {"load": 7.0, "store": 9.0}))


# ---- 评审 round09：样本与 scope 同步换目标但画像不符时也不能写回 ----

def _write_sample_rows(path, samples):
    rows = []
    for sample in samples:
        rows.append({
            "sample_id": sample.sample_id, "dataset_id": sample.dataset_id,
            "source": sample.source, "phase": sample.phase,
            "kind": sample.kind, "shape": list(sample.shape),
            "dtype": sample.dtype, "predicted": sample.predicted,
            "observed": sample.observed, "target": dict(sample.target),
            "unit": sample.unit, "features": dict(sample.features),
            "operator": sample.operator,
            "artifact_phase": sample.artifact_phase,
        })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_calibrate_script_refuses_write_back_for_another_profile_scope(
        tmp_path):
    """样本与 scope 同步换成另一目标、却与本画像不符时不能写回画像。"""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    scope = _scope()
    scope["target_name"] = "unrelated-target"
    target = _target(scope)
    samples = [replace(sample, target=target) for sample in _valid_samples()]
    samples_path = tmp_path / "samples.jsonl"
    _write_sample_rows(samples_path, samples)
    artifacts = _artifacts(tmp_path)
    for name, payload in (
        ("artifacts.json", artifacts),
        ("package.json", _package(artifacts)),
        ("thresholds.json", _thresholds(GROUPS)),
        ("scope.json", scope),
    ):
        (tmp_path / name).write_text(json.dumps(payload))
    report = tmp_path / "report.json"
    profile_out = tmp_path / "profile.json"
    proc = subprocess.run(
        [sys.executable, "scripts/calibrate_profile.py",
         "--samples", str(samples_path), "--training-dataset", "train",
         "--validation-dataset", "heldout", "--profile", str(PROFILE_PATH),
         "--scope", str(tmp_path / "scope.json"),
         "--thresholds", str(tmp_path / "thresholds.json"),
         "--artifacts", str(tmp_path / "artifacts.json"),
         "--package", str(tmp_path / "package.json"),
         "--simulation-summary", str(_summary(tmp_path)),
         "--profile-out", str(profile_out), "--out", str(report)],
        cwd=str(repo), capture_output=True, text=True)
    payload = json.loads(report.read_text())
    assert payload["status"] == "passed", payload.get("reason")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not profile_out.exists()
    assert "画像" in (proc.stdout + proc.stderr)
