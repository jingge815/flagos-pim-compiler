"""站点相关路径集中配置。

本文件是 genesim_bridge 里**唯一**允许出现机器绝对路径的地方。其余模块
一律从这里取值，不再各自硬编码。

三层优先级，从高到低：

1. 环境变量（推荐，换机器不用改代码）：

       export FLAGTREE_PREFIX=/path/to/flagOS-installed/flagTree
       export FLAGTREE_PIM_PREFIX=/path/to/flagOS-installed/flagTree-pim
       export GENESIM_ROOT=/path/to/genesim

2. 配置文件 `paths.local.json`（放在本仓库根，已被 .gitignore 忽略）：

       {
         "flagtree_prefix": "/path/to/flagOS-installed/flagTree",
         "flagtree_pim_prefix": "/path/to/flagOS-installed/flagTree-pim",
         "genesim_root": "/path/to/genesim"
       }

3. 下面的 `_DEFAULTS`——当前开发机的实测路径，仅作兜底。

换机器时首选方式 1 或 2，不要改 `_DEFAULTS`。

PIM 硬件参数（`pim_*`）走同一套三层优先级，见 `pim_options()`。
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
    "flagtree_pim_prefix": "/media/disk/fengjingge/src/flagOS/flagOS-installed/flagTree-pim",
    "genesim_root": "/media/disk/fengjingge/src/genesim",
}

# 环境变量名 ← 配置键
_ENV_VARS = {
    "flagtree_prefix": "FLAGTREE_PREFIX",
    "flagtree_pim_prefix": "FLAGTREE_PIM_PREFIX",
    "genesim_root": "GENESIM_ROOT",
}

# triton nvidia backend 的相对位置（cuda.h 与 ptxas 所在）。两份 flagTree 安装
# 的 python 环境布局不同：非 PIM 那份装在 `python/`，PIM 那份装在 `venv/`。
_NVIDIA_BACKEND_SUBPATH = "python/lib/python3.10/site-packages/triton/backends/nvidia"
_PIM_SITE_PACKAGES_SUBPATH = "venv/lib/python3.10/site-packages"

# PIM 硬件参数默认值。与 FlagTree 的 convert-triton-to-pim pass 选项、
# pim_sidecar.py 的 DEFAULT_* 保持一致，改这里等于换 pass 的输入参数。
# 实测这几个值当前不影响产物（FlagGems 的 tile 由 GPU autotune 定，PIM pass
# 不重切），但仍要记进 sidecar：换硬件配置时它们是唯一的溯源依据。
_PIM_DEFAULTS = {
    "pim_target": "pim:v1",
    "pim_num_dpus": 1,
    "pim_num_tasklets": 16,
    "pim_wram_bytes": 65536,
}

_PIM_ENV_VARS = {
    "pim_target": "FLAGTREE_PIM_TARGET",
    "pim_num_dpus": "FLAGTREE_PIM_NUM_DPUS",
    "pim_num_tasklets": "FLAGTREE_PIM_NUM_TASKLETS",
    "pim_wram_bytes": "FLAGTREE_PIM_WRAM_BYTES",
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


def flagtree_pim_prefix() -> Path:
    """带 PIM 支持的 flagTree 安装根目录。

    与 `flagtree_prefix()` 是两份独立安装：只有这一份的 `libtriton.so` 里有
    `convert-triton-to-pim` / `pim-explicit-dma`，pytorch env 里那份没有。
    """
    return _resolve("flagtree_pim_prefix")


def flagtree_pim_site_packages() -> Path:
    """PIM 安装的 site-packages（前插 sys.path 即可让 `import triton` 拿到 PIM 版）。"""
    return flagtree_pim_prefix() / _PIM_SITE_PACKAGES_SUBPATH


def flagtree_pim_nvidia_backend() -> Path:
    """PIM 安装里 triton 的 nvidia backend 目录。

    切到 PIM triton 时 cuda.h / ptxas 必须取**同一份安装**的，混用两份安装的
    头文件与 triton 二进制会在 driver 初始化编 cuda_utils.c 时炸。
    """
    return flagtree_pim_site_packages() / "triton" / "backends" / "nvidia"


def pim_options() -> dict:
    """PIM pass 的硬件参数，按 环境变量 > paths.local.json > _PIM_DEFAULTS 取值。"""
    file_config = _load_file_config()
    options = {}
    for key, default in _PIM_DEFAULTS.items():
        raw = os.environ.get(_PIM_ENV_VARS[key]) or file_config.get(key) or default
        options[key] = raw if isinstance(default, str) else int(raw)
    return options


def genesim_root() -> Path:
    """GeneSim 仓库根目录。"""
    return _resolve("genesim_root")


def genesim_models_dir() -> Path:
    """GeneSim 的 models/ 目录（.ir 产物所在）。"""
    return genesim_root() / "models"


def describe() -> str:
    """返回当前生效的路径与 PIM 参数来源，供报错信息与调试使用。"""
    file_config = _load_file_config()

    def source_of(key: str, env_var: str, default_name: str) -> str:
        if os.environ.get(env_var):
            return f"环境变量 {env_var}"
        if file_config.get(key):
            return f"配置文件 {_CONFIG_FILE.name}"
        return f"内置默认值 {default_name}"

    lines = [
        f"  {key} = {_resolve(key)}  ({source_of(key, env_var, '_DEFAULTS')})"
        for key, env_var in _ENV_VARS.items()
    ]
    options = pim_options()
    lines += [
        f"  {key} = {options[key]}  ({source_of(key, env_var, '_PIM_DEFAULTS')})"
        for key, env_var in _PIM_ENV_VARS.items()
    ]
    return "\n".join(lines)
