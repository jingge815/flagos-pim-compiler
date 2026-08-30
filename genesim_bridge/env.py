"""环境修复：让 triton 能初始化 CUDA driver。

**补 cuda.h / ptxas。** env-pytorch.sh 装的那份 triton 原本缺
`backends/nvidia/include`（cuda.h）和 `backends/nvidia/bin`（ptxas），
driver 初始化时编 cuda_utils.c 直接失败，`import flag_gems` 就抛。
flagTree 那份安装两者都有，指过去即可。

不改 FlagTree / FlagGems / triton 源码，只调几个环境变量。
必须在 `import triton` / `import flag_gems` 之前调用。

路径不在本文件里硬编码，取自 paths.py（可用 FLAGTREE_PREFIX 环境变量或
paths.local.json 覆盖）。

历史注记（2026-08-29 起不再适用，保留供追溯）：这里曾经维护两份独立的
triton 安装——pytorch 环境自带的普通版和 `flagTree-pim` 裸 venv 里带 PIM
pass 的版本，`prepare_triton_env(pim=True)` 负责在 `import triton` 之前把
`sys.path` 切到后者。当时 pytorch env 那份 `libtriton.so` 确实没有 PIM
pass。这个前提现在不成立：`flagTree` 重新编译后已经把带 PIM pass 的
`libtriton.so`/`backends/pim_sidecar.py`/nvidia backend 同步进了 pytorch
环境（`0-install-flagtree.sh::sync_triton_to_pytorch`），任何时候
`import triton` 都自带 PIM 支持，不再需要、也不应该再切换 sys.path——这与
`opcompiler_bridge/driver.py` 在 2026-08-25 完成的同一次迁移一致。
`flagtree_pim_prefix()` 等函数已从 paths.py 删除，`prepare_triton_env`
的 `pim` 参数保留仅为向后兼容调用方签名，不再做任何实际切换。
"""

from __future__ import annotations

import os
from pathlib import Path

from .paths import describe, flagtree_nvidia_backend


def prepare_triton_env(pim: bool = False) -> None:
    """准备 triton 运行环境：补 cuda.h / ptxas。

    `pim` 参数保留仅为向后兼容旧调用方（如
    `scripts/refine_ir_with_flagtree.py`）的签名——现在只有一份 triton
    安装，任何时候 `import triton` 都自带 PIM pass 支持，`pim=True` 不再
    触发任何 sys.path 切换。
    """
    _add_cuda_toolchain(flagtree_nvidia_backend())


def _add_cuda_toolchain(backend: Path) -> None:
    """把 triton nvidia backend 的 cuda.h 与 ptxas 挂进环境变量。"""
    include_dir = backend / "include"
    ptxas = backend / "bin" / "ptxas"

    if not (include_dir / "cuda.h").is_file():
        raise RuntimeError(
            f"找不到 cuda.h：{include_dir}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.local.json 指向正确的 "
            "flagTree 安装。"
        )
    if not ptxas.is_file():
        raise RuntimeError(
            f"找不到 ptxas：{ptxas}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.local.json 指向正确的 "
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
