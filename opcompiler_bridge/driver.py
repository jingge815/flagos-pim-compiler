"""将算子编译请求转换为可由 `ctypes` 加载的共享库。"""

from __future__ import annotations

import ctypes
import hashlib
import os
import threading
import re
import subprocess
import tempfile
from pathlib import Path

from contracts.op_contract import (
    DEFAULT_HARDWARE_CONFIG,
    OpCompileRequest,
    OpCompileResult,
    PIMHardwareConfig,
    flatten_leading_dims,
)
from genesim_bridge.paths import flagtree_prefix, pim_options

# 保存按编译请求区分的共享库缓存。
_CACHE_DIR = Path(
    os.environ.get(
        "OPCOMPILER_CACHE_DIR",
        str(Path(__file__).resolve().parent.parent / ".opcompiler_cache"),
    )
)

# 匹配生成的 C 函数名和裸指针参数。
_SIG_RE = re.compile(r"void\s+(\w+)\s*\(([^)]*)\)")
_PARAM_RE = re.compile(r"^\s*(\w+)\s*\*\s*\w+\s*$")

_CTYPE_BY_C_ELEM = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int16_t": ctypes.c_int16,   # f16 存储：C 里没有可移植的 half，见新 pass
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
}

# 将存储数据类型映射为 Triton 实参的 PyTorch 数据类型。
_TORCH_DTYPES: dict[str, object] = {}


def _torch_dtypes() -> dict:
    global _TORCH_DTYPES
    if not _TORCH_DTYPES:
        import torch

        _TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}
    return _TORCH_DTYPES


def _triton_opt() -> Path:
    """返回带有 PIM 降级 pass 的 ``triton-opt`` 路径。"""
    override = os.environ.get("OPCOMPILER_TRITON_OPT")
    if override:
        return Path(override)
    return flagtree_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"


def _mlir_translate() -> Path:
    return flagtree_prefix() / "llvm-7d5de303" / "bin" / "mlir-translate"


def _cache_key(request: OpCompileRequest) -> str:
    # 缓存键包含数据类型、tasklet 数和硬件配置。
    payload = (
        f"{request.op}:{request.arg_shapes}:{request.dtype}:"
        f"{request.num_tasklets}:{request.hardware.to_payload()}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _kernel_launcher(request: OpCompileRequest):
    if request.op != "linear":
        raise NotImplementedError(
            f"opcompiler_bridge 第一期只覆盖 linear，收到: {request.op!r}"
        )
    if len(request.arg_shapes) != 2:
        raise ValueError(
            f"linear 契约要求 arg_shapes=[x.shape, weight.shape]，收到: "
            f"{request.arg_shapes!r}"
        )
    # 将输入前导维合并为 M，权重保持二维。
    m, k = flatten_leading_dims(request.arg_shapes[0])
    n, k2 = request.arg_shapes[1]
    if k != k2:
        raise ValueError(
            f"x 和 weight 的 K 维不一致: x.shape={request.arg_shapes[0]!r} "
            f"weight.shape={request.arg_shapes[1]!r}"
        )
    if k < 16:
        raise ValueError(
            f"tl.dot 要求 K>=16（Triton 自身对 tensor-core 输入的硬约束），"
            f"收到 K={k}。llama2-7b 的真实 K 远超这个下限，仅在自验证等极小 "
            f"shape 测试时可能触发。"
        )
    # 只有 M 需要自己是 2 的幂：kernel_src.py 里 `tl.arange(0, M)` 直接铺满 M，
    # 而 K/N 是按 BLOCK_K/BLOCK_N 分块遍历的，arange 作用在分块上。所以 K/N 的
    # 约束是「存在一个既是 2 的幂、又能整除它的分块」，由 pick_blocks 判定。
    #
    # 这条约束以前对 M/K/N 一律要求 2 的幂，把 llama2-7b 的 MLP 直接挡在门外：
    # intermediate_size = 11008 = 2^8 × 43，任何切分下都不是 2 的幂（tp2 是
    # 5504、tp4 是 2752）。而 5504 = 128 × 43，分块 128 完全合法。
    if m & (m - 1) != 0:
        raise ValueError(
            f"kernel_src.py 用 tl.arange(0, M) 直接生成整块索引，Triton 要求 "
            f"arange 的范围是 2 的幂，收到 M={m}。decode 口径下 M=1，prefill 用 "
            f"2 的幂序列长度即可。"
        )

    from .kernel_src import pick_blocks

    # 分块不合法时在这里报错，而不是等 Triton 抛出含义模糊的 arange 报错。
    pick_blocks(k, n)

    from .kernel_src import make_kernel_launcher

    return make_kernel_launcher(m, k, n)


def _make_ttir(request: OpCompileRequest) -> str:
    """在 GPU 上编译目标形状并返回 TTIR 文本。"""
    import torch

    launch = _kernel_launcher(request)
    m, k = flatten_leading_dims(request.arg_shapes[0])
    n, _ = request.arg_shapes[1]
    # 使用请求指定的数据类型构造 Triton 输入张量。
    dtypes = _torch_dtypes()
    if request.dtype not in dtypes:
        raise ValueError(
            f"opcompiler_bridge 只支持 float16/float32 存储（新 pass 的 "
            f"checkElementType 同样），收到 dtype={request.dtype!r}"
        )
    torch_dtype = dtypes[request.dtype]
    x = torch.empty((m, k), dtype=torch_dtype, device="cuda")
    w = torch.empty((n, k), dtype=torch_dtype, device="cuda")
    out = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    compiled = launch(x, w, out)
    return compiled.asm["ttir"]


def _run_triton_opt(ttir: str, hardware: PIMHardwareConfig) -> tuple[str, str]:
    """将 TTIR 转换为 PIM IR 和 EmitC 文本，两者都返回。

    分两次调用 `triton-opt`，而不是把五个 pass 串成一条命令：pim mlir 是
    `-pim-lower-to-emitc` 的输入，串起来跑就只剩最后的 EmitC，中间态拿不到。
    拆开后多一次进程启动，换来的是 pim mlir 文本——GeneSim 的代价模型要靠它
    拿到真实分块和 DMA 结构（见 genesim_bridge/ir_cost.py 的 analyze_ir）。

    两段的产物与合并跑完全一致：pass 顺序没变，只是在中间落了一次盘。
    """
    opts = pim_options()
    triton_opt = _triton_opt()
    if not triton_opt.is_file():
        raise RuntimeError(
            f"找不到带 pim-lower-to-emitc 的 triton-opt: {triton_opt}\n"
            "需要先在 FlagTree 里重新编译（该 pass 是本次新增，若安装未重建会"
            "缺这个 pass）。"
        )

    def _run(input_text: str, passes: list[str], stage: str) -> str:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".mlir", delete=False
        ) as handle:
            handle.write(input_text)
            path = handle.name
        try:
            proc = subprocess.run(
                [str(triton_opt), path, *passes],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"triton-opt {stage} 阶段失败 (exit {proc.returncode}):\n"
                    f"{proc.stderr}"
                )
            return proc.stdout
        finally:
            os.unlink(path)

    # 第一段到 pim mlir：这一层带 !pim.memdesc、pim.dma_load/store 和真实分块。
    pimir = _run(
        ttir,
        [
            f"-convert-triton-to-pim=target={opts['pim_target']} "
            f"num-dpus={hardware.num_dpus} "
            f"num-tasklets={hardware.num_tasklets} "
            f"wram-bytes={hardware.wram_bytes_per_dpu} "
            f"mram-bytes={hardware.mram_bytes_per_dpu} "
            f"dma-align={hardware.dma_align}",
            "-pim-tile-to-budget",
            "-pim-explicit-dma",
        ],
        "pim mlir",
    )
    # 第二段到 EmitC：后续 mlir-translate 生成 C 的输入。
    emitc = _run(
        pimir,
        ["-pim-lower-to-emitc", "-convert-func-to-emitc"],
        "emitc",
    )
    return pimir, emitc


def _translate_to_c(emitc_text: str) -> str:
    mlir_translate = _mlir_translate()
    if not mlir_translate.is_file():
        raise RuntimeError(f"找不到 mlir-translate: {mlir_translate}")
    proc = subprocess.run(
        [str(mlir_translate), "--mlir-to-cpp"],
        input=emitc_text,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mlir-translate 失败:\n{proc.stderr}")
    return proc.stdout


def _parse_signature(c_source: str) -> tuple[str, list[str]]:
    """从生成的 C 源码解析函数名和参数类型。"""
    match = _SIG_RE.search(c_source)
    if not match:
        raise RuntimeError(
            f"无法从生成的 C 源码中解析出函数签名:\n{c_source}"
        )
    symbol, params_str = match.group(1), match.group(2)
    argtypes: list[str] = []
    for param in (p.strip() for p in params_str.split(",") if p.strip()):
        m = _PARAM_RE.match(param)
        if not m:
            raise RuntimeError(
                f"生成的 C 函数签名带有非裸指针参数，超出本 pass 的 ABI 设计"
                f"（不应该出现偏移量或 memref descriptor）: {param!r}\n"
                f"完整签名: {c_source[match.start():match.end()]}"
            )
        elem = m.group(1)
        if elem not in _CTYPE_BY_C_ELEM:
            raise RuntimeError(f"未知的 C 元素类型: {elem!r}（参数 {param!r}）")
        argtypes.append(elem)
    return symbol, argtypes


def compile_op(request: OpCompileRequest, *, force: bool = False) -> OpCompileResult:
    """编译算子请求并返回共享库描述，结果按请求参数缓存。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(request)
    so_path = _CACHE_DIR / f"{key}.so"
    meta_path = _CACHE_DIR / f"{key}.meta"
    pimir_path = _CACHE_DIR / f"{key}.pimir.mlir"

    if not force and so_path.is_file() and meta_path.is_file():
        symbol, argtypes_str = meta_path.read_text().splitlines()
        # pim mlir 与 `.so` 一起缓存，命中时直接读回来，成本模型不必重编。
        cached_pimir = (
            pimir_path.read_text() if pimir_path.is_file() else None
        )
        return OpCompileResult(
            so_path=str(so_path),
            symbol=symbol,
            argtypes=argtypes_str.split(","),
            pimir=cached_pimir,
            pimir_path=str(pimir_path) if cached_pimir is not None else None,
        )

    ttir = _make_ttir(request)
    pimir_text, emitc_text = _run_triton_opt(ttir, request.hardware)
    c_source = _translate_to_c(emitc_text)
    symbol, argtypes = _parse_signature(c_source)

    # 使用进程和线程唯一的临时路径编译共享库。
    tmp_tag = f"{os.getpid()}.{threading.get_ident()}"
    c_path = _CACHE_DIR / f"{key}.{tmp_tag}.c"
    so_tmp_path = _CACHE_DIR / f"{key}.{tmp_tag}.so"
    # 生成的 C 代码需要 `malloc` 和 `free`。
    try:
        c_path.write_text("#include <stdint.h>\n#include <stdlib.h>\n" + c_source)
        proc = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_tmp_path), str(c_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gcc 编译生成的 C 失败:\n{proc.stderr}")
        os.replace(so_tmp_path, so_path)
    finally:
        c_path.unlink(missing_ok=True)
        so_tmp_path.unlink(missing_ok=True)  # no-op if os.replace already moved it

    meta_path.write_text(f"{symbol}\n{','.join(argtypes)}")
    # pim mlir 与 `.so` 一起落盘：下次命中缓存时成本模型直接读它，不必重编。
    pimir_path.write_text(pimir_text)
    return OpCompileResult(
        so_path=str(so_path),
        symbol=symbol,
        argtypes=argtypes,
        pimir=pimir_text,
        pimir_path=str(pimir_path),
    )


def load_kernel(result: OpCompileResult):
    """加载共享库并返回已设置参数类型的 ``ctypes`` 函数。"""
    lib = ctypes.CDLL(result.so_path)
    fn = getattr(lib, result.symbol)
    fn.argtypes = [ctypes.c_void_p for _ in result.argtypes]
    fn.restype = None
    return fn


def _selftest() -> None:
    """编译固定形状的线性算子并与 NumPy 结果比较。"""
    import dataclasses
    import numpy as np

    os.environ.setdefault("FLAGTREE_PIM_NUM_DPUS", "1")
    os.environ.setdefault("FLAGTREE_PIM_NUM_TASKLETS", "1")

    request = OpCompileRequest(
        op="linear",
        arg_shapes=[(2, 16), (4, 16)],
        hardware=dataclasses.replace(
            DEFAULT_HARDWARE_CONFIG, num_dpus=1, num_tasklets=1, mram_bytes_per_dpu=1 << 20
        ),
    )
    result = compile_op(request, force=True)
    print(f"compiled: {result}")

    fn = load_kernel(result)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 16)).astype(np.float32)
    w = rng.standard_normal((4, 16)).astype(np.float32)
    out = np.zeros((2, 4), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.c_void_p),
        w.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    ref = x @ w.T
    ok = np.allclose(out, ref, atol=1e-4)
    print(f"out={out.tolist()}")
    print(f"ref={ref.tolist()}")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
