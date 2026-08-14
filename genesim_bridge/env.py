"""环境修复：让 env-pytorch.sh 里的 triton 能初始化 CUDA driver。

env-pytorch.sh 装的那份 triton 缺 `backends/nvidia/include`（cuda.h）和
`backends/nvidia/bin`（ptxas），driver 初始化时编 cuda_utils.c 直接失败，
`import flag_gems` 就抛。flagTree 那份安装两者都有，指过去即可。

不改 FlagTree / FlagGems / triton 源码，只补两个环境变量。
必须在 `import triton` / `import flag_gems` 之前调用。

flagTree 安装路径不在本文件里硬编码，取自 paths.py（可用 FLAGTREE_PREFIX
环境变量或 paths.local.json 覆盖）。
"""

from __future__ import annotations

import os

from .paths import describe, flagtree_nvidia_backend


def prepare_triton_env() -> None:
    backend = flagtree_nvidia_backend()
    include_dir = backend / "include"
    ptxas = backend / "bin" / "ptxas"

    if not (include_dir / "cuda.h").is_file():
        raise RuntimeError(
            f"找不到 cuda.h：{include_dir}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.local.json 指向正确的 flagTree 安装。"
        )
    if not ptxas.is_file():
        raise RuntimeError(
            f"找不到 ptxas：{ptxas}\n"
            f"当前生效路径：\n{describe()}\n"
            "请用 FLAGTREE_PREFIX 环境变量或 paths.local.json 指向正确的 flagTree 安装。"
        )

    cpath = os.environ.get("CPATH", "")
    if str(include_dir) not in cpath.split(":"):
        os.environ["CPATH"] = f"{include_dir}:{cpath}" if cpath else str(include_dir)
    os.environ.setdefault("TRITON_PTXAS_PATH", str(ptxas))
