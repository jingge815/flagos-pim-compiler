"""算子编译驱动：`OpCompileRequest` → 一个可 `ctypes.CDLL` 加载的 `.so`。

完整链路（每一步都已经手工跑通过一次，细节见 `docs/opcompiler_bridge-20260825.md`）：

    triton.compile（FLAGTREE_EMIT_PIM=1，num_dpus=1 num_tasklets=1）
        -> pim_sidecar 产出 `.pimir`（其实是重新对 TTIR 跑
           convert-triton-to-pim + pim-explicit-dma，见下）
        -> triton-opt -pim-lower-single-tasklet -convert-func-to-emitc
        -> mlir-translate --mlir-to-cpp
        -> gcc -shared -fPIC
        -> .so，签名 `void <symbol>(float*, float*, float*)`

不直接读 `pim_sidecar` 的 dump（同 `genesim_bridge/flagtree_driver.py` 的理由：
不依赖 `FLAGTREE_EMIT_PIM`/`TRITON_DUMP_DIR`、不依赖编译缓存 miss），而是拿到
`CompiledKernel.asm['ttir']` 后重新 parse 成 module，自己跑 pass。

用法：

    from opcompiler_bridge.driver import compile_op
    from contracts.op_contract import OpCompileRequest
    result = compile_op(OpCompileRequest(op="linear", arg_shapes=[(2, 16), (4, 16)]))
    # result.so_path / result.symbol / result.argtypes

**环境要求（2026-08-25 之后不再需要 `prepare_triton_env(pim=True)`）**：
这台机器上原来有两份 triton 安装——pytorch 环境自带的普通版和
`flagTree-pim` venv 里带 PIM pass 支持的版本——`import triton` 只认进程内第
一次加载的那份，且不能中途切换（`genesim_bridge.env.prepare_triton_env`
的限制）。真实端到端场景下 `transformers`/`torch` 会在任何人想切环境之前就
先 `import triton`，导致 `compiled_linear_kernel` 实际编译时永远只能拿到没
有 PIM pass 的普通版——这是从真实 llama2-7b 端到端测试插桩里实测到的、不是
猜的。修法是不再维护两份 triton：已经把 pytorch 环境这份 triton 的
`_C/libtriton.so`、`backends/pim_sidecar.py`、
`backends/nvidia/{compiler.py,bin,include,lib/cupti}` 换成 `flagTree-pim`
venv 里的对应文件（两者用同一个 LLVM commit 构建，ABI 兼容，已验证换过去后
普通 Triton kernel 和 PIM pass 都能跑）。现在只有一份 triton，任何时候
`import triton` 都自带 PIM pass 支持，不需要也不应该再调
`prepare_triton_env`。
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import threading
import re
import subprocess
import tempfile
from pathlib import Path

from contracts.op_contract import OpCompileRequest, OpCompileResult, flatten_leading_dims
from genesim_bridge.paths import flagtree_pim_prefix, pim_options

# 编译产物缓存目录：按 (op, arg_shapes) 的哈希分文件，同一 shape 只编译一次。
# 与 pim_sidecar/genesim 的缓存目录分开，理由同它们互相分开——不同 pass 链的
# 产物不可混用。
_CACHE_DIR = Path(
    os.environ.get(
        "OPCOMPILER_CACHE_DIR",
        str(Path(__file__).resolve().parent.parent / ".opcompiler_cache"),
    )
)

# 生成的 C 函数签名固定形如 `void <name>(float* v1, float* v2, ...)`——
# LowerPIMSingleTasklet 把每个原始 !tt.ptr<f32> 参数改写成一个 !emitc.ptr<f32>，
# 不带偏移量参数（见 kernel_src.py 顶部注释、docs/opcompiler_bridge-20260825.md）。
# 这里解析实际产出的签名，而不是硬编码假设，一旦形状不对就直接报错。
_SIG_RE = re.compile(r"void\s+(\w+)\s*\(([^)]*)\)")
_PARAM_RE = re.compile(r"^\s*(\w+)\s*\*\s*\w+\s*$")

_CTYPE_BY_C_ELEM = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int16_t": ctypes.c_int16,   # f16 存储：C 里没有可移植的 half，见新 pass
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
}

# 契约里的存储 dtype（NumPy 名）-> 建 Triton 实参用的 torch dtype。只列这条
# 链路支持的两种：新 pass 的 checkElementType 也只接受 f16/f32。
_TORCH_DTYPES: dict[str, object] = {}


def _torch_dtypes() -> dict:
    global _TORCH_DTYPES
    if not _TORCH_DTYPES:
        import torch

        _TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}
    return _TORCH_DTYPES


def _triton_opt() -> Path:
    """带 `pim-lower-single-tasklet` 的 `triton-opt`。

    可用 `OPCOMPILER_TRITON_OPT` 覆盖。默认指向 `flagTree-pim` 安装里的构建
    目录——注意这台机器上还有一份 `FlagTree-back/build-pim` 构建树，改了 pass
    源码后**两份都要重新 `ninja bin/triton-opt`**，否则这里用到的是旧二进制、
    症状是报一个源码里已经不存在的错误（踩过一次）。
    """
    override = os.environ.get("OPCOMPILER_TRITON_OPT")
    if override:
        return Path(override)
    return flagtree_pim_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"


def _mlir_translate() -> Path:
    return flagtree_pim_prefix() / "llvm-7d5de303" / "bin" / "mlir-translate"


def _cache_key(request: OpCompileRequest) -> str:
    # dtype 必须进 key：同一 shape 的 f16 / f32 产物读写的元素宽度不同，
    # 互换会静默算错（见 contracts/op_contract.py 里记的那个 bug）。
    payload = f"{request.op}:{request.arg_shapes}:{request.dtype}".encode()
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
    # x 可能带 batch 维（真实图上是 (batch, seq, hidden) 三维，见
    # contracts/op_contract.py 顶部注释），展平成 (M, K)；weight 恒二维。
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
    for name, dim in (("M", m), ("K", k), ("N", n)):
        if dim & (dim - 1) != 0:
            raise ValueError(
                f"kernel_src.py 用 tl.arange(0, {name}) 直接生成整块索引"
                f"（第一期不做 tile 切分/padding，见该文件顶部注释），"
                f"Triton 要求 arange 的范围是 2 的幂，收到 {name}={dim}。"
                f"真实 llama2-7b 的 M/K/N 都满足这个约束（hidden_size/"
                f"num_heads 等惯例上是 2 的幂），只有手工测试用到非 2 的幂"
                f"形状时会触发；若未来需要支持任意 shape，需要在这一层加"
                f"padding，不在本次范围内。"
            )

    from .kernel_src import make_kernel_launcher

    return make_kernel_launcher(m, k, n)


def _make_ttir(request: OpCompileRequest) -> str:
    """在 GPU 上真的调一次目标 shape 的 kernel，拿到 Triton 编译出的 TTIR。

    不手搓 ASTSource——同 `genesim_bridge/flagtree_driver.py` 的理由：手写会
    绕开 Triton 自己的类型/layout 推导，编出的 IR 与真实 launch 不符。
    """
    import torch

    launch = _kernel_launcher(request)
    m, k = flatten_leading_dims(request.arg_shapes[0])
    n, _ = request.arg_shapes[1]
    # Triton 从实参的 dtype 推出 `!tt.ptr<f16>` / `!tt.ptr<f32>`，新 pass 再
    # 据此决定生成的 C 里每个元素怎么读写（f16 走位转换 helper）。所以这里
    # 必须用契约里的存储 dtype 建张量，不能一律 float32。
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


def _wram_budget(request: OpCompileRequest) -> int:
    """本次 shape 下 `pim-explicit-dma` 需要的 WRAM 预算（字节）。

    三个 tile：`x` 是 `M x BLOCK_K`、`w` 是 `BLOCK_N x BLOCK_K`、输出是
    `M x BLOCK_N`，都是 float32。给一倍余量后向上取到 2 的幂，避免贴着边界。
    """
    from .kernel_src import pick_blocks

    m, k = flatten_leading_dims(request.arg_shapes[0])
    n = request.arg_shapes[1][0]
    block_n, block_k = pick_blocks(k, n)
    itemsize = 2 if request.dtype == "float16" else 4
    needed = (m * block_k + block_n * block_k + m * block_n) * itemsize
    budget = 1 << max(16, (needed * 2 - 1).bit_length())
    return budget


def _run_triton_opt(ttir: str, wram_bytes: int) -> str:
    """ttir 文本 -> 跑完 convert-triton-to-pim/pim-explicit-dma/
    pim-lower-single-tasklet/convert-func-to-emitc 之后的 emitc 文本。

    `pim-lower-single-tasklet` 要求 num-tasklets=1（见该 pass 的
    runOnOperation 检查），单 DPU 单 tasklet 是本期契约的前提。

    `wram_bytes` 由调用方按本次 shape 的实际 tile 大小算出来传进来，而不是取
    `pim_options()` 里的默认值：`pim-explicit-dma` 会对每个 `pim.wram_alloc`
    检查是否超预算并报错，而 K 分块后 `w` 的 tile 是 `N x BLOCK_K x 4` 字节
    （llama2-7b 的 N=512、BLOCK_K=64 就是 128KiB，超过默认的 64KiB）。这里
    的预算只影响 `pim-explicit-dma` 的这条检查——numpy 后端上没有真实 WRAM，
    见 docs/opcompiler_bridge-20260825.md。
    """
    opts = pim_options()
    triton_opt = _triton_opt()
    if not triton_opt.is_file():
        raise RuntimeError(
            f"找不到带 pim-lower-single-tasklet 的 triton-opt: {triton_opt}\n"
            "需要先在 FlagTree 里重新编译（该 pass 是本次新增，若安装未重建会"
            "缺这个 pass）。"
        )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".ttir", delete=False
    ) as handle:
        handle.write(ttir)
        ttir_path = handle.name
    try:
        cmd = [
            str(triton_opt),
            ttir_path,
            f"-convert-triton-to-pim=target={opts['pim_target']} "
            f"num-tasklets=1 wram-bytes={wram_bytes}",
            "-pim-explicit-dma",
            "-pim-lower-single-tasklet",
            "-convert-func-to-emitc",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"triton-opt pass 链失败 (exit {proc.returncode}):\n{proc.stderr}"
            )
        return proc.stdout
    finally:
        os.unlink(ttir_path)


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
    """从生成的 C 源码里解析出函数名和 ctypes.argtypes 列表。

    不硬编码假设 ABI——按 LowerPIMSingleTasklet 的设计，参数应该全是
    `<elem>*`（没有偏移量参数），但这里仍然去读实际文本，形状不对就直接报错，
    而不是静默假设。
    """
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
    """编译 `request` 对应的算子，返回可加载的 `.so` 描述。

    按 `(op, arg_shapes)` 缓存（`_CACHE_DIR`），同一 shape 不会重复编译——
    对应图编译器每个不同 M（prefill 长度 vs decode M=1）本来就是独立
    `ExecutionPlan` 这件事，见 `contracts/op_contract.py` 顶部注释。
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(request)
    so_path = _CACHE_DIR / f"{key}.so"
    meta_path = _CACHE_DIR / f"{key}.meta"

    if not force and so_path.is_file() and meta_path.is_file():
        symbol, argtypes_str = meta_path.read_text().splitlines()
        return OpCompileResult(
            so_path=str(so_path),
            symbol=symbol,
            argtypes=argtypes_str.split(","),
        )

    ttir = _make_ttir(request)
    emitc_text = _run_triton_opt(ttir, _wram_budget(request))
    c_source = _translate_to_c(emitc_text)
    symbol, argtypes = _parse_signature(c_source)

    # 编译到 <key>.<pid>.<tid>.so 这个每次调用独有的临时名，成功后再原子
    # rename 到最终的 <key>.so——两个调用者并发编译同一个 shape 时（runtime/
    # kernels.py 用 _COMPILE_LOCK 序列化了这条路径上唯一会真的并发的调用者，
    # 但这里不假设所有调用方都会拿那把锁），各自写各自的临时文件，不会互相
    # 踩到对方正在写的目标路径。同名旧文件直接被 rename 覆盖，也是原子的。
    # 之前踩过的坑：两个 gcc 进程并发 `-o <key>.so` 写同一个路径，产出的
    # .so 被截断，ctypes 加载时报 `undefined symbol`（8 卡 decode 第一次遇到
    # 新 shape 时稳定复现）。
    tmp_tag = f"{os.getpid()}.{threading.get_ident()}"
    c_path = _CACHE_DIR / f"{key}.{tmp_tag}.c"
    so_tmp_path = _CACHE_DIR / f"{key}.{tmp_tag}.so"
    # stdlib.h: LowerPIMSingleTasklet emits malloc/free to snapshot MRAM
    # operands into a private heap buffer before computing (mem_planner can
    # alias an op's output onto a dead input's address, and the compiled
    # kernel runs in a ThreadPoolExecutor worker whose pthread stack is too
    # small for a multi-MiB local array -- see LowerPIMSingleTasklet.cpp's
    # snapshotToLocal for both).
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
    return OpCompileResult(so_path=str(so_path), symbol=symbol, argtypes=argtypes)


def load_kernel(result: OpCompileResult):
    """加载 `compile_op` 的产物，返回一个 `ctypes` 函数对象，argtypes 已设置好
    （全部是 `c_void_p`——裸指针，调用方自己用 `ndarray.ctypes.data_as` 转换，
    偏移量在 Python 侧用指针地址算术提前加好，见 `runtime/kernels.py`）。
    """
    lib = ctypes.CDLL(result.so_path)
    fn = getattr(lib, result.symbol)
    fn.argtypes = [ctypes.c_void_p for _ in result.argtypes]
    fn.restype = None
    return fn


def _selftest() -> None:
    """`python -m opcompiler_bridge.driver --selftest`：固定小 shape
    （M=2,K=16,N=4，K=16 是 tl.dot 的硬下限）跑完整链路，和 numpy 对拍。
    在接入 runtime/kernels.py 之前先跑这个，隔离"编译产物本身对不对"和
    "跟执行器接线对不对"两类问题。
    """
    import numpy as np

    os.environ.setdefault("FLAGTREE_PIM_NUM_DPUS", "1")
    os.environ.setdefault("FLAGTREE_PIM_NUM_TASKLETS", "1")

    request = OpCompileRequest(op="linear", arg_shapes=[(2, 16), (4, 16)])
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
