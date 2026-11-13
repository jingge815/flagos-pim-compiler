"""输入快照：内容指纹、轨迹窗口与上下文上界。"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from tune.config import load_config
from tune.inputs import (file_sha256, model_fingerprint, snapshot_inputs,
                         validate_dry_inputs)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"


def _trace(path, requests):
    path.write_text(json.dumps({"description": "unit", "requests": requests}))


def _request(index, tokens, output=8, prefill=True):
    return {
        "request_id": f"r{index}",
        "input_tokens": list(range(1, tokens + 1)),
        "output_length": output,
        "arrival_time_ns": 0,
        "is_prefill": prefill,
        "seq_len": tokens if prefill else 16,
        "prompt": None,
        "padded_query_length": None,
        "initial_kv_length": 0 if prefill else 16,
    }


def _config(tmp_path, *, trace_path, num_requests=1, prefill=16, max_seq=None,
            ir=None):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text('{"num_hidden_layers": 1}')
    raw = {
        "version": 1, "run_name": "inputs",
        "model": {
            "model_dir": str(model_dir), "ir": ir,
            "prefill_seq_len": prefill, "max_seq": max_seq,
            "trace": {"path": str(trace_path), "num_requests": num_requests,
                      "synthetic": None},
        },
        "profile": {"path": str(PROFILE)},
        "space": {
            "strategy": {"enabled": True, "num_dpus": [4], "num_stages": [2]},
            "op": {"enabled": True, "num_tasklets": [4]},
            "hardware": {"enabled": False, "fields": {}},
        },
        "placement": {
            "enabled": False, "algorithm": "simulated_annealing",
            "evaluation_backend": "pipeline", "validation_backend": "pipeline",
            "objective_metric": "total_time_s",
            "objective_weights": {"compute": 0.5, "communication": 0.25,
                                  "memory": 0.25},
            "max_evaluations": 1, "max_time_s": 0, "max_proposals": None,
            "seed": 0, "moves": {"max_segment_ops": 8,
                                 "max_layer_block_layers": 4, "weights": {}},
            "algorithm_params": {},
        },
        "search": {
            "algorithm": "enumerate", "seed": 0, "max_evaluations": 1,
            "max_time_s": 0, "max_proposals": None,
            "layer_weights": {"strategy": 1, "op": 0, "hardware": 0},
            "algorithm_params": {},
        },
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


def test_snapshot_limits_requests_and_writes_window(tmp_path):
    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8), _request(1, 12), _request(2, 16)])
    ir = tmp_path / "model.ir"
    ir.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, num_requests=2, ir=str(ir))
    inputs = snapshot_inputs(cfg)
    assert len(inputs.trace.requests) == 2
    assert inputs.max_seq == 12 + 8
    saved = json.loads(inputs.trace_path.read_text())
    assert len(saved["requests"]) == 2
    assert (inputs.input_dir / "inputs.json").is_file()


def test_snapshot_rejects_prefill_shorter_than_prompt(tmp_path):
    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 32)])
    cfg = _config(tmp_path, trace_path=trace_path, prefill=16)
    with pytest.raises(ValueError, match="prefill"):
        snapshot_inputs(cfg)


def test_dry_run_validation_does_not_generate_ir(tmp_path):
    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 32)])
    cfg = _config(tmp_path, trace_path=trace_path, prefill=16, ir=None)
    with pytest.raises(ValueError, match="prefill"):
        validate_dry_inputs(cfg)
    assert not (cfg.output.dir / "inputs" / "model.ir").exists()


def test_model_fingerprint_tracks_weight_content(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"a": 1}')
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights-v1")
    first = model_fingerprint(model_dir)
    weights.write_bytes(b"weights-v2")
    second = model_fingerprint(model_dir)
    assert first != second


def test_file_sha256_is_content_addressed(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    first = file_sha256(path)
    assert first == file_sha256(path)
    path.write_bytes(b"abd")
    assert file_sha256(path) != first


# ---- 评审 round04：输入身份前置、外置 IR 快照与请求数契约 ----

def test_prepare_inputs_is_read_only_and_external_ir_is_snapshotted(tmp_path):
    """只读解析不写输出目录；外置 IR 与轨迹一样进入不可变快照。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    ir = tmp_path / "external.ir"
    ir.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(ir))
    prepared = prepare_inputs(cfg)
    assert not cfg.output.dir.exists()
    assert prepared.external_ir == ir
    assert prepared.ir_sha256 == file_sha256(ir)
    published = publish_inputs(cfg, prepared)
    assert published.ir_path == cfg.output.dir / "inputs" / "model.ir"
    assert published.ir_path.read_text() == ir.read_text()
    # 快照发布后改外置 IR 不影响已发布内容。
    ir.write_text('{"operators": [1]}')
    assert published.ir_path.read_text() == '{"operators": []}'
    with pytest.raises(ValueError, match="输入身份"):
        from tune.inputs import verify_input_identity
        verify_input_identity(
            prepare_inputs(cfg),
            json.loads((cfg.output.dir / "inputs" / "inputs.json").read_text()))


def test_short_trace_is_rejected_by_dry_run_and_snapshot(tmp_path):
    """轨迹请求数不足必须在输入前置契约里拒绝。"""
    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    cfg = _config(tmp_path, trace_path=trace_path, num_requests=2)
    with pytest.raises(ValueError, match="num_requests"):
        validate_dry_inputs(cfg)
    with pytest.raises(ValueError, match="num_requests"):
        snapshot_inputs(cfg)
    assert not (cfg.output.dir / "inputs" / "trace.json").exists()


def test_publish_refuses_trace_changed_after_identity_check(tmp_path):
    """身份核对后轨迹再变化时，发布必须失败且不改写旧快照。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    ir = tmp_path / "external.ir"
    ir.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(ir))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    original = first.trace_path.read_bytes()
    prepared = prepare_inputs(cfg)
    _trace(trace_path, [_request(0, 12)])
    with pytest.raises(RuntimeError, match="轨迹"):
        publish_inputs(cfg, prepared)
    assert first.trace_path.read_bytes() == original


# ---- 评审 round05：外置 IR 竞态不得改写已发布快照 ----

def _published_hashes(inputs):
    return {
        "ir": file_sha256(inputs.ir_path),
        "trace": file_sha256(inputs.trace_path),
        "identity": file_sha256(inputs.input_dir / "inputs.json"),
    }


def _staging_dirs(input_dir):
    return [path for path in input_dir.parent.glob(input_dir.name + ".staging*")
            if path.is_dir()]


def test_publish_external_ir_changed_after_prepare_preserves_old_snapshot(
        tmp_path, monkeypatch):
    """外置 IR 在 prepare 后变化时，失败只清理暂存区，旧快照字节不变。"""
    import tune.inputs as module
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text("prepared-content")
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)

    prepared = prepare_inputs(cfg)
    original = module._atomic_copy

    def changing_copy(source, destination):
        Path(source).write_text("changed-after-prepare")
        return original(source, destination)

    monkeypatch.setattr(module, "_atomic_copy", changing_copy)
    with pytest.raises(RuntimeError, match="外置模型 IR"):
        publish_inputs(cfg, prepared)
    assert _published_hashes(first) == before
    assert _staging_dirs(first.input_dir) == []


def test_publish_trace_change_preserves_every_published_file(tmp_path):
    """轨迹在身份核对后变化时，IR、轨迹与身份清单都保持原字节。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)
    prepared = prepare_inputs(cfg)
    _trace(trace_path, [_request(0, 12)])
    with pytest.raises(RuntimeError, match="轨迹"):
        publish_inputs(cfg, prepared)
    assert _published_hashes(first) == before
    assert _staging_dirs(first.input_dir) == []
    assert json.loads((first.input_dir / "inputs.json").read_text())[
        "trace_sha256"] == before["trace"]


def test_publish_generated_ir_failure_keeps_old_snapshot(tmp_path, monkeypatch):
    """生成 IR 失败时，已发布的轨迹与身份清单不被改写。"""
    import tune.inputs as module
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)

    # 删掉已发布 IR，让下一次 prepare 走“现场生成”路径并失败。
    (first.input_dir / "model.ir").unlink()
    generated = prepare_inputs(cfg)
    assert generated.external_ir is not None
    generated = replace(generated, external_ir=None, ir_sha256="", ir_ready=False)

    def boom(model_dir, out_path):
        raise RuntimeError("model_parser broke")

    monkeypatch.setattr(module, "_generate_model_ir", boom)
    with pytest.raises(RuntimeError, match="model_parser"):
        publish_inputs(cfg, generated)
    assert file_sha256(first.trace_path) == before["trace"]
    assert file_sha256(first.input_dir / "inputs.json") == before["identity"]
    assert not (first.input_dir / "model.ir").exists()
    assert _staging_dirs(first.input_dir) == []


def test_snapshot_publish_is_atomic_from_a_staging_directory(tmp_path):
    """正常发布后暂存目录不残留，发布目录包含三件必需证据。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    inputs = publish_inputs(cfg, prepare_inputs(cfg))
    assert (inputs.input_dir / "model.ir").is_file()
    assert (inputs.input_dir / "trace.json").is_file()
    assert (inputs.input_dir / "inputs.json").is_file()
    assert _staging_dirs(inputs.input_dir) == []


def test_publish_copy_failure_preserves_published_snapshot(
        tmp_path, monkeypatch):
    """外置 IR 复制失败时旧 IR/轨迹/身份清单都不变。"""
    import tune.inputs as module
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)

    def broken_copy(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(module, "_atomic_copy", broken_copy)
    with pytest.raises(OSError, match="disk full"):
        publish_inputs(cfg, prepare_inputs(cfg))
    assert _published_hashes(first) == before
    assert _staging_dirs(first.input_dir) == []


def test_publish_refuses_to_overwrite_a_different_published_snapshot(tmp_path):
    """已发布快照与本次身份不同时整组拒绝，不做逐文件覆盖。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)

    _trace(trace_path, [_request(0, 12)])
    prepared = prepare_inputs(cfg)
    with pytest.raises(RuntimeError, match="不同的输入快照"):
        publish_inputs(cfg, prepared)
    assert _published_hashes(first) == before
    assert _staging_dirs(first.input_dir) == []


def test_publish_identical_snapshot_is_a_noop(tmp_path):
    """同内容再次发布不写任何文件，旧快照保持原字节。"""
    from tune.inputs import prepare_inputs, publish_inputs

    trace_path = tmp_path / "requests.json"
    _trace(trace_path, [_request(0, 8)])
    external = tmp_path / "external.ir"
    external.write_text('{"operators": []}')
    cfg = _config(tmp_path, trace_path=trace_path, ir=str(external))
    first = publish_inputs(cfg, prepare_inputs(cfg))
    before = _published_hashes(first)
    published = publish_inputs(cfg, prepare_inputs(cfg))
    assert _published_hashes(published) == before
    assert _staging_dirs(first.input_dir) == []
