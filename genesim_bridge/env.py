"""环境修复：让 triton 能初始化 CUDA driver，并可切到带 PIM 支持的那份安装。

两件事：

1. **补 cuda.h / ptxas。** env-pytorch.sh 装的那份 triton 缺
   `backends/nvidia/include`（cuda.h）和 `backends/nvidia/bin`（ptxas），
   driver 初始化时编 cuda_utils.c 直接失败，`import flag_gems` 就抛。
   flagTree 那份安装两者都有，指过去即可。

2. **`pim=True` 时切到 PIM triton。** pytorch env 里那份 triton 的
   `libtriton.so` **没有** PIM 支持（实测 0 个 `convert-triton-to-pim` 符号），
   而带 PIM 的 `flagTree-pim` 安装是个裸 venv、没有 torch / flag_gems。
   把 PIM 安装的 site-packages 前插 `sys.path`，`import triton` 就拿到 PIM 版，
   torch 与 flag_gems 仍从 pytorch env 解析——实测三者共存正常。

不改 FlagTree / FlagGems / triton 源码，只调 `sys.path` 与几个环境变量。
必须在 `import triton` / `import flag_gems` 之前调用。

路径不在本文件里硬编码，取自 paths.py（可用 FLAGTREE_PREFIX /
FLAGTREE_PIM_PREFIX 环境变量或 paths.local.json 覆盖）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .paths import (
    describe,
    flagtree_nvidia_backend,
    flagtree_pim_nvidia_backend,
    flagtree_pim_prefix,
    flagtree_pim_site_packages,
)


def prepare_triton_env(pim: bool = False) -> None:
    """准备 triton 运行环境。

    pim=False（默认，TTIR 路）：沿用 pytorch env 的 triton，只补 cuda.h / ptxas。
    pim=True（pim mlir 路）：改用带 PIM 支持的那份 triton 安装。

    pim=True 时 cuda.h / ptxas 必须取**同一份安装**的：混用两份安装的头文件与
    triton 二进制会在编 cuda_utils.c 时炸。缓存目录也单独隔离——两份 triton
    的 kernel 二进制不可互换，共用缓存会拿到另一份编出来的产物。
    """
    if pim:
        site_packages = flagtree_pim_site_packages()
        if not (site_packages / "triton").is_dir():
            raise RuntimeError(
                f"找不到 PIM triton 安装：{site_packages}\n"
                f"当前生效路径：\n{describe()}\n"
                "请用 FLAGTREE_PIM_PREFIX 环境变量或 paths.local.json 指向"
                "带 PIM 支持的 flagTree 安装。"
            )
        if "triton" in sys.modules:
            raise RuntimeError(
                "triton 已被 import，无法再切到 PIM 安装。"
                "prepare_triton_env(pim=True) 必须在 import triton 之前调用。"
            )
        sys.path.insert(0, str(site_packages))
        backend = flagtree_pim_nvidia_backend()
        # 两份 triton 的 kernel 二进制不可互换，缓存必须分开
        os.environ["TRITON_CACHE_DIR"] = str(flagtree_pim_prefix() / "cache-genesim-bridge")
    else:
        backend = flagtree_nvidia_backend()

    _add_cuda_toolchain(backend)


def _add_cuda_toolchain(backend: Path) -> None:
    """把一份 triton nvidia backend 的 cuda.h 与 ptxas 挂进环境变量。"""
    include_dir = backend / "include"
    ptxas = backend / "bin" / "ptxas"

    if not (include_dir / "cuda.h").is_file():
        raise RuntimeError(
            f"找不到 cuda.h：{include_dir}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX / FLAGTREE_PIM_PREFIX 环境变量或 "
            "paths.local.json 指向正确的 flagTree 安装。"
        )
    if not ptxas.is_file():
        raise RuntimeError(
            f"找不到 ptxas：{ptxas}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX / FLAGTREE_PIM_PREFIX 环境变量或 "
            "paths.local.json 指向正确的 flagTree 安装。"
        )

    cpath = os.environ.get("CPATH", "")
    if str(include_dir) not in cpath.split(":"):
        os.environ["CPATH"] = f"{include_dir}:{cpath}" if cpath else str(include_dir)
    # PIM 路要覆盖而非 setdefault：env-flagtree.sh 可能已把 ptxas 指到另一份安装
    os.environ["TRITON_PTXAS_PATH"] = str(ptxas)


def assert_pim_passes_available() -> None:
    """确认当前 triton 里有 PIM pass。缺了直接抛，不静默降级到 TTIR。"""
    from triton._C.libtriton import passes

    if not hasattr(passes, "pim"):
        import triton

        raise RuntimeError(
            f"当前 triton 没有 PIM 支持：{triton.__file__}\n"
            f"当前生效路径：\n{describe()}\n"
            "说明 sys.path 上先出现了不带 PIM 的那份 triton。"
            "请确认 prepare_triton_env(pim=True) 在任何 import triton 之前调用。"
        )
