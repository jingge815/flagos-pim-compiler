"""站点相关路径集中配置。

本文件是 genesim_bridge 里**唯一**允许出现机器绝对路径的地方。其余模块
一律从这里取值，不再各自硬编码。

三层优先级，从高到低：

1. 环境变量（推荐，换机器不用改代码）：

       export FLAGTREE_PREFIX=/path/to/flagOS-installed/flagTree
       export GENESIM_ROOT=/path/to/genesim

2. 配置文件 `paths.local.json`（放在本仓库根，已被 .gitignore 忽略）：

       {
         "flagtree_prefix": "/path/to/flagOS-installed/flagTree",
         "genesim_root": "/path/to/genesim"
       }

3. 下面的 `_DEFAULTS`——当前开发机的实测路径，仅作兜底。

换机器时首选方式 1 或 2，不要改 `_DEFAULTS`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 本仓库根（flagos-pim-compiler/），由文件位置推出，不需要配置
REPO_ROOT = Path(__file__).resolve().parent.parent

_CONFIG_FILE = REPO_ROOT / "paths.local.json"

# 兜底默认值：当前开发机实测路径。优先用环境变量或 paths.local.json 覆盖。
_DEFAULTS = {
    "flagtree_prefix": "/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree",
    "genesim_root": "/media/disk/fengjingge/src/genesim",
}

# 环境变量名 ← 配置键
_ENV_VARS = {
    "flagtree_prefix": "FLAGTREE_PREFIX",
    "genesim_root": "GENESIM_ROOT",
}

# flagTree 安装里 triton nvidia backend 的相对位置（cuda.h 与 ptxas 所在）
_NVIDIA_BACKEND_SUBPATH = "python/lib/python3.10/site-packages/triton/backends/nvidia"


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


def _resolve(key: str) -> Path:
    """按 环境变量 > paths.local.json > _DEFAULTS 的优先级取一个路径。"""
    env_value = os.environ.get(_ENV_VARS[key])
    if env_value:
        return Path(env_value)
    file_value = _load_file_config().get(key)
    if file_value:
        return Path(file_value)
    return Path(_DEFAULTS[key])


def flagtree_prefix() -> Path:
    """flagTree 安装根目录。"""
    return _resolve("flagtree_prefix")


def flagtree_nvidia_backend() -> Path:
    """flagTree 里 triton 的 nvidia backend 目录（含 include/cuda.h 与 bin/ptxas）。"""
    return flagtree_prefix() / _NVIDIA_BACKEND_SUBPATH


def genesim_root() -> Path:
    """GeneSim 仓库根目录。"""
    return _resolve("genesim_root")


def genesim_models_dir() -> Path:
    """GeneSim 的 models/ 目录（.ir 产物所在）。"""
    return genesim_root() / "models"


def describe() -> str:
    """返回当前生效的路径来源，供报错信息与调试使用。"""
    lines = []
    for key, env_var in _ENV_VARS.items():
        if os.environ.get(env_var):
            source = f"环境变量 {env_var}"
        elif _load_file_config().get(key):
            source = f"配置文件 {_CONFIG_FILE.name}"
        else:
            source = "内置默认值 _DEFAULTS"
        lines.append(f"  {key} = {_resolve(key)}  ({source})")
    return "\n".join(lines)
