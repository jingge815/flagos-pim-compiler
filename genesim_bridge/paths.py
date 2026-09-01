"""读取仓库路径和 PIM 硬件参数。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from contracts.op_contract import DEFAULT_HARDWARE_CONFIG

# 本仓库根目录。
REPO_ROOT = Path(__file__).resolve().parent.parent

_CONFIG_FILE = REPO_ROOT / "paths.json"

# 配置键对应的环境变量。
_ENV_VARS = {
    "pytorch_env_script": "PYTORCH_ENV_SCRIPT",
    "llama2_7b_model_dir": "LLAMA2_7B_MODEL_DIR",
    "flagtree_prefix": "FLAGTREE_PREFIX",
    "genesim_root": "GENESIM_ROOT",
}

# Triton NVIDIA 后端目录。
_NVIDIA_BACKEND_SUBPATH = "python/lib/python3.10/site-packages/triton/backends/nvidia"

# PIM 编译 pass 使用的默认硬件参数。
_PIM_DEFAULTS = {
    "pim_target": "pim:v1",
    "pim_num_dpus": 1,
    "pim_num_tasklets": DEFAULT_HARDWARE_CONFIG.num_tasklets,
    "pim_wram_bytes": DEFAULT_HARDWARE_CONFIG.wram_bytes_per_dpu,
    "pim_mram_bytes": DEFAULT_HARDWARE_CONFIG.mram_bytes_per_dpu,
    "pim_dma_align": DEFAULT_HARDWARE_CONFIG.dma_align,
}

_PIM_ENV_VARS = {
    "pim_target": "FLAGTREE_PIM_TARGET",
    "pim_num_dpus": "FLAGTREE_PIM_NUM_DPUS",
    "pim_num_tasklets": "FLAGTREE_PIM_NUM_TASKLETS",
    "pim_wram_bytes": "FLAGTREE_PIM_WRAM_BYTES",
    "pim_mram_bytes": "FLAGTREE_PIM_MRAM_BYTES",
    "pim_dma_align": "FLAGTREE_PIM_DMA_ALIGN",
}


def _load_file_config() -> dict:
    if not _CONFIG_FILE.is_file():
        return {}
    try:
        data = json.loads(_CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{_CONFIG_FILE} 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{_CONFIG_FILE} 顶层必须是 JSON object")
    return data


def _configured_path(key: str) -> Path | None:
    """按环境变量优先读取一个可选的站点路径。"""
    env_value = os.environ.get(_ENV_VARS[key])
    if env_value:
        return Path(env_value)
    file_value = _load_file_config().get(key)
    if file_value is None:
        return None
    if not isinstance(file_value, str) or not file_value.strip():
        raise RuntimeError(f"{_CONFIG_FILE} 的 {key} 必须是非空字符串")
    return Path(file_value)


def _resolve(key: str) -> Path:
    """读取一个必填站点路径；未配置时给出明确的配置指引。"""
    path = _configured_path(key)
    if path is not None:
        return path
    raise RuntimeError(
        f"未配置站点路径 {key}。请在 {_CONFIG_FILE} 设置 {key}，"
        f"或设置环境变量 {_ENV_VARS[key]}。"
    )


def pytorch_env_script() -> Path:
    """PyTorch 环境初始化脚本。"""
    return _resolve("pytorch_env_script")


def llama2_7b_model_dir(*, required: bool = True) -> Path | None:
    """真实 Llama-2-7B 权重目录；外部依赖测试可传 ``required=False``。"""
    path = _configured_path("llama2_7b_model_dir")
    if path is not None or not required:
        return path
    return _resolve("llama2_7b_model_dir")


def flagtree_prefix() -> Path:
    """flagTree 安装根目录（唯一安装，带 PIM pass 支持）。"""
    return _resolve("flagtree_prefix")


def flagtree_nvidia_backend() -> Path:
    """flagTree 里 triton 的 nvidia backend 目录（含 include/cuda.h 与 bin/ptxas）。"""
    return flagtree_prefix() / _NVIDIA_BACKEND_SUBPATH


def pim_options() -> dict:
    """PIM pass 的硬件参数，按环境变量 > paths.json > 内置默认值取值。"""
    file_config = _load_file_config()
    options = {}
    for key, default in _PIM_DEFAULTS.items():
        raw = os.environ.get(_PIM_ENV_VARS[key]) or file_config.get(key) or default
        options[key] = raw if isinstance(default, str) else int(raw)
    return options


def genesim_root(*, required: bool = True) -> Path | None:
    """GeneSim 仓库根目录；外部产物检查可传 ``required=False``。"""
    path = _configured_path("genesim_root")
    if path is not None or not required:
        return path
    return _resolve("genesim_root")


def genesim_models_dir(*, required: bool = True) -> Path | None:
    """GeneSim 的 models/ 目录（.ir 产物所在）。"""
    root = genesim_root(required=required)
    return root / "models" if root is not None else None


def describe() -> str:
    """返回当前生效的路径与 PIM 参数来源，供报错信息与调试使用。"""
    file_config = _load_file_config()

    def source_of(key: str, env_var: str, default_name: str | None = None) -> str:
        if os.environ.get(env_var):
            return f"环境变量 {env_var}"
        if file_config.get(key):
            return f"配置文件 {_CONFIG_FILE.name}"
        if default_name is not None:
            return f"内置默认值 {default_name}"
        return "未配置"

    lines = []
    for key, env_var in _ENV_VARS.items():
        path = _configured_path(key)
        value = path if path is not None else "<未配置>"
        lines.append(f"  {key} = {value}  ({source_of(key, env_var)})")
    options = pim_options()
    lines += [
        f"  {key} = {options[key]}  ({source_of(key, env_var, '_PIM_DEFAULTS')})"
        for key, env_var in _PIM_ENV_VARS.items()
    ]
    return "\n".join(lines)
