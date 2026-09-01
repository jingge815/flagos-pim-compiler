"""站点路径配置单测。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import genesim_bridge.paths as paths


def _use_config(monkeypatch, tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "paths.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(paths, "_CONFIG_FILE", config_path)
    for env_var in paths._ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    return config_path


def test_paths_are_read_from_paths_json(monkeypatch, tmp_path) -> None:
    _use_config(
        monkeypatch,
        tmp_path,
        {
            "pytorch_env_script": "/customer/pytorch/env-pytorch.sh",
            "llama2_7b_model_dir": "/customer/models/Llama-2-7b-hf",
            "flagtree_prefix": "/customer/flagTree",
            "genesim_root": "/customer/genesim",
        },
    )

    assert paths.pytorch_env_script() == Path("/customer/pytorch/env-pytorch.sh")
    assert paths.llama2_7b_model_dir() == Path("/customer/models/Llama-2-7b-hf")
    assert paths.flagtree_prefix() == Path("/customer/flagTree")
    assert paths.genesim_root() == Path("/customer/genesim")


def test_environment_variable_overrides_paths_json(monkeypatch, tmp_path) -> None:
    _use_config(monkeypatch, tmp_path, {"flagtree_prefix": "/customer/flagTree"})
    monkeypatch.setenv("FLAGTREE_PREFIX", "/override/flagTree")

    assert paths.flagtree_prefix() == Path("/override/flagTree")


def test_missing_path_reports_configuration_instructions(monkeypatch, tmp_path) -> None:
    config_path = _use_config(monkeypatch, tmp_path, {})

    with pytest.raises(RuntimeError, match="FLAGTREE_PREFIX") as exc_info:
        paths.flagtree_prefix()

    assert "flagtree_prefix" in str(exc_info.value)
    assert str(config_path) in str(exc_info.value)
    assert paths.llama2_7b_model_dir(required=False) is None


def test_describe_marks_missing_paths_as_unconfigured(monkeypatch, tmp_path) -> None:
    _use_config(monkeypatch, tmp_path, {})

    assert "flagtree_prefix = <未配置>" in paths.describe()
    assert "pim_target = pim:v1  (内置默认值 _PIM_DEFAULTS)" in paths.describe()
