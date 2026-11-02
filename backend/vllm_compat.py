"""锁定的 vLLM 版本范围与启动期校验。"""

from __future__ import annotations

import vllm

SUPPORTED_VLLM = "0.14.x"


def check_vllm_version() -> None:
    """版本不在支持范围内就立刻失败，免得在接口不兼容的版本上跑出难查的错。"""
    version = vllm.__version__
    if not version.startswith("0.14."):
        raise RuntimeError(
            f"PIM 后端只支持 vLLM {SUPPORTED_VLLM}，当前 {version}；"
            f"装回 0.14.x，或按 docs/vllm-0.31-gap-20261102.md 完成适配")
