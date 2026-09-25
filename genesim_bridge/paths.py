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
    "gml_reference_dir": "GML_REFERENCE_DIR",
    "gml_llama2_reference_dir": "GML_LLAMA2_REFERENCE_DIR",
}

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


def gml_reference_dir(*, required: bool = True) -> Path | None:
    """GML 格式参考产物目录；结构与字段覆盖率测试要用，缺失时可传 ``required=False``。"""
    path = _configured_path("gml_reference_dir")
    if path is not None or not required:
        return path
    return _resolve("gml_reference_dir")


def gml_llama2_reference_dir(*, required: bool = True) -> Path | None:
    """llama2 W4A8 的 GML 参考产物目录。

    比 ResNet50 那份更贴近目标模型：它是 decode block，带动态量化与 KV cache，
    所以结构规则要在两份上同时成立才算可靠。

    指向 **v2**（`model_layers_0_decode_v2`，图头 `relay2gml_version "19.2.0"`）。
    v1（`llama2_w4a8_decode_block_0`，`"26.2.1"`）留着作历史对照，把
    `paths.json` 指回去它仍然可用——`tests/test_gml_hw_table.py` 的例外集合就是
    为它准备的。

    **版本字符串不是判据**：切换参考后按字段集合对拍，不拿版本号当失败条件。
    我方 `GML_VERSION` 也不随参考改。
    """
    path = _configured_path("gml_llama2_reference_dir")
    if path is not None or not required:
        return path
    return _resolve("gml_llama2_reference_dir")


def flagtree_prefix() -> Path:
    """flagTree 安装根目录（唯一安装，带 PIM pass 支持）。"""
    return _resolve("flagtree_prefix")


def flagtree_source(required: bool = True) -> Path | None:
    """FlagTree **源码树**（不是安装目录）。

    从安装里的 `env-flagtree.sh` 读 `FLAGTREE_SOURCE`，而不是在 `paths.json` 里
    再写一条：那个脚本是安装时生成的，它指向的就是这次安装用的源码。两处各写
    一份的话，换个源码树重装之后 `paths.json` 那条会静默指向旧的。

    `contracts/fusion_contract.py` 与 FlagTree 的 `FuseActivation.cpp` 必须描述
    同一张融合表，而 C++ 读不了 Python。唯一能自动对起来的办法是读那份 C++ 源码
    的文本，所以这里要能找到它。
    """
    env_file = flagtree_prefix() / "env-flagtree.sh"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("FLAGTREE_SOURCE="):
                path = Path(line.split("=", 1)[1].strip().strip('"'))
                if path.is_dir():
                    return path
    if required:
        raise RuntimeError(
            f"没能从 {env_file} 读出可用的 FLAGTREE_SOURCE。"
            f"重装 FlagTree 会重新生成这个脚本。")
    return None


def _flagtree_site_packages() -> Path:
    """flagTree 安装里 Python 的 site-packages 目录（python3.X 版本号动态探测）。"""
    lib_dir = flagtree_prefix() / "python" / "lib"
    matches = sorted(lib_dir.glob("python3.*"))
    if len(matches) != 1:
        raise RuntimeError(
            f"{lib_dir} 下应有且只有一个 python3.* 目录，实际找到 {len(matches)} 个：{matches}\n"
            f"当前生效路径：\n{describe()}"
        )
    return matches[0] / "site-packages"


def flagtree_nvidia_backend() -> Path:
    """flagTree 里 triton 的 nvidia backend 目录（含 include/cuda.h 与 bin/ptxas）。"""
    return _flagtree_site_packages() / "triton" / "backends" / "nvidia"


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
