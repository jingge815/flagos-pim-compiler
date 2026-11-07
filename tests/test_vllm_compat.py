"""验证 vLLM 版本锁定：支持范围内放行，范围外点名当前版本与支持范围。"""

from __future__ import annotations

import pytest
import vllm

from backend.vllm_compat import SUPPORTED_VLLM, check_vllm_version


def test_supported_version_passes(monkeypatch) -> None:
    """0.14.x 放行。"""
    monkeypatch.setattr(vllm, "__version__", "0.14.0")

    check_vllm_version()


def test_unsupported_version_names_both_ranges(monkeypatch) -> None:
    """0.31 报错，消息里同时有当前版本与支持范围。"""
    monkeypatch.setattr(vllm, "__version__", "0.31.0")

    with pytest.raises(RuntimeError, match=SUPPORTED_VLLM) as excinfo:
        check_vllm_version()
    assert "0.31.0" in str(excinfo.value)
