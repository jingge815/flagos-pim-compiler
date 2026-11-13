"""复用 GeneSim 的搜索算法，不复制一份。"""

from __future__ import annotations

import sys
from functools import lru_cache

from genesim_bridge.paths import genesim_root


@lru_cache(maxsize=1)
def _ensure_genesim_path() -> None:
    src = genesim_root() / "src"
    if not src.is_dir():
        raise RuntimeError(f"找不到 GeneSim 源码目录：{src}")
    # 追加到末尾：GeneSim 的 `src/backend` 与本仓的 `backend` 同名，插到前面会
    # 让后面对 `backend.dpu_sdk` 的导入解析到另一个包。
    if str(src) not in sys.path:
        sys.path.append(str(src))


def search_algorithms():
    """导入 GeneSim 的 `create_search_algorithm` 与预算类型。"""
    _ensure_genesim_path()
    from autotune.api import Budget, Evaluation, EvaluationStatus
    from autotune.search import create_search_algorithm

    return create_search_algorithm, Budget, Evaluation, EvaluationStatus


def build_algorithm(name: str, parameters: dict | None, seed: int):
    """按名字构造算法；非法参数在这里直接报错。"""
    create_search_algorithm, _, _, _ = search_algorithms()
    if name == "enumerate":
        if parameters:
            raise ValueError("enumerate 不接受算法参数")
        return None
    try:
        return create_search_algorithm(name, parameters or {}, seed=seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"算法 {name} 的参数非法：{exc}") from exc


def trace_api():
    """导入 GeneSim 的轨迹类型与合成发生器。"""
    _ensure_genesim_path()
    from ir.trace_ir import Trace, generate_synthetic_trace

    return Trace, generate_synthetic_trace


def genesim_config_class():
    """导入 GeneSim 的配置类，用它的 override/resolved_config 生成候选配置。"""
    _ensure_genesim_path()
    from config_loader import GeneSimConfig

    return GeneSimConfig
