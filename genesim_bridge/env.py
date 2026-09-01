"""配置 Triton 所需的 CUDA 头文件和汇编器路径。"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import describe, flagtree_nvidia_backend


def prepare_triton_env(pim: bool = False) -> None:
    """设置 Triton 使用的 CUDA 头文件和 ``ptxas`` 路径。"""
    _add_cuda_toolchain(flagtree_nvidia_backend())


def _add_cuda_toolchain(backend: Path) -> None:
    """把 triton nvidia backend 的 cuda.h 与 ptxas 挂进环境变量。"""
    include_dir = backend / "include"
    ptxas = backend / "bin" / "ptxas"

    if not (include_dir / "cuda.h").is_file():
        raise RuntimeError(
            f"找不到 cuda.h：{include_dir}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.json 指向正确的 "
            "flagTree 安装。"
        )
    if not ptxas.is_file():
        raise RuntimeError(
            f"找不到 ptxas：{ptxas}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.json 指向正确的 "
            "flagTree 安装。"
        )

    cpath = os.environ.get("CPATH", "")
    if str(include_dir) not in cpath.split(":"):
        os.environ["CPATH"] = f"{include_dir}:{cpath}" if cpath else str(include_dir)
    os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas))


def assert_pim_passes_available() -> None:
    """确认当前 triton 里有 PIM pass。缺了直接抛，不静默降级到 TTIR。"""
    from triton._C.libtriton import passes

    if not hasattr(passes, "pim"):
        import triton

        raise RuntimeError(
            f"当前 triton 没有 PIM 支持：{triton.__file__}\n"
            f"当前生效路径：\n{describe()}\n"
            "需要重新跑一遍 0-install-flagtree.sh（不加 --skip-pytorch-sync）"
            "把带 PIM pass 的 triton 同步进 pytorch 环境。"
        )
