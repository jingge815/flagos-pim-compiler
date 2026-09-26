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
from opcompiler_bridge.oplevel_emitter import PIM_TARGET
from opcompiler_bridge.oplevel_kernel import (
    concat_kernel,
    convert_kernel,
    dynamic_quant_kernel,
    eltwise_kernel,
    gather_kernel,
    kv_cache_kernel,
    lut_kernel,
    mask_kernel,
    matmul_kernel,
    normalize_kernel,
    reshape_kernel,
    rope_kernel,
    softmax_kernel,
    split_heads_kernel,
    transpose_kernel,
)

# 保存按编译请求区分的共享库缓存。
_CACHE_DIR = Path(
    os.environ.get(
        "OPCOMPILER_CACHE_DIR",
        str(Path(__file__).resolve().parent.parent / ".opcompiler_cache"),
    )
)

# 匹配生成的 C 函数名和裸指针参数。
_SIG_RE = re.compile(r"void\s+(\w+)\s*\(([^)]*)\)")
# 按值标量参数，带不带形参名都算：`int32_t` 或 `int32_t v3`。
_SCALAR_PARAM_RE = re.compile(
    r"^(int(?:8|16|32|64)_t|float|double)(?:\s+\w+)?$")
_PARAM_RE = re.compile(r"^\s*(\w+)\s*\*\s*\w+\s*$")

_CTYPE_BY_C_ELEM = {
    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "int16_t": ctypes.c_int16,   # f16 存储：C 里没有可移植的 half，见新 pass
    "int8_t": ctypes.c_int8,     # 量化输出
    "int32_t": ctypes.c_int32,
    "int64_t": ctypes.c_int64,
}

class ToolchainUnavailable(RuntimeError):
    """算子编译器的工具链不在位（`triton-opt` / `mlir-translate` 缺失或没有 PIM pass）。

    单独一个类型，是为了让调用方能把「这台机器上编不了」与「这个算子编错了」
    分开处理（评审 20260923 的 P1-5）。`runtime/kernels.py` 原来对 softmax
    捕获 `(ValueError, NotImplementedError, RuntimeError)` 三类一起回退镜像，
    于是「编译失败」和「编译成功且与镜像一致」给出同一个结果——那条对拍判据
    在编译静默失败时照样通过，因为两边比的是同一份镜像。

    环境缺失是真实边界（CI 上可能没重建 FlagTree），所以它该回退；
    其余一律往上抛。
    """


# 算子级（B 路）能编的 mnemonic。与 A 路是两条不同的入口：那条从 Triton
# kernel 出发经 TTIR 与显式 DMA，这条从整算子级 PIM IR 出发，没有 DMA。
# 图上 14 类设备侧算子各自的 mnemonic。缺一个名字 = 那个算子执行失败，
# 而不是悄悄跑回主机 numpy：静默回退会让「两端对齐」这个判据失去意义。
_OPLEVEL_OPS = frozenset({
    "softmax", "dynamic_quant", "gather", "rope", "matmul", "normalize",
    "mask", "transpose", "reshape", "concat", "convert", "lut", "eltwise",
    "kv_cache", "split_heads",
})

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
    # 缓存键包含数据类型、目标类型、tasklet 数、组大小和硬件配置。
    # `out_dtype` 必须在内：`convert` 的 f16→i8 与 f16→f32 只有这一位不同，
    # 漏掉它两条请求会落到同一个 `.so` 上——而 `ctypes.CDLL` 按路径缓存句柄，
    # 第二次 `load_kernel` 拿到的是第一份的代码，数值全错而形状全对。
    payload = (
        f"{request.op}:{request.arg_shapes}:{request.dtype}:"
        f"{request.num_tasklets}:{request.group_size}:{request.activation}:"
        f"{request.kind}:{request.tail_card_value}:{request.sf_multiplier}:"
        f"{request.out_dtype}:"
        f"{request.hardware.to_payload()}"
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


_TRITON_DTYPE_BY_NAME = {"float16": "fp16", "float32": "fp32"}


def _make_ttir(request: OpCompileRequest) -> str:
    """编译目标形状并返回 TTIR 文本。

    有 GPU 时走原生路径（真实 launch 取 `asm["ttir"]`）；没有 GPU 时走
    `cpu_host.make_ttir` 的纯前端路径。TTIR 是 AST → IR 的产物，本来就不需要设备，
    两条路径产出的 pim mlir 一致（同形状同硬件参数下 sha256 相同，见
    `opcompiler_bridge/cpu_host.py` 的说明）。

    保留原生路径而不是一律走前端：有卡时 `asm["ttir"]` 是 Triton 自己维护的口径，
    跟着上游演进最稳妥；前端路径要自己拼 signature 和特化，是无卡机器的补偿实现。
    """
    from .cpu_host import gpu_hardware_present

    m, k = flatten_leading_dims(request.arg_shapes[0])
    n, _ = request.arg_shapes[1]
    dtypes = _torch_dtypes()
    if request.dtype not in dtypes:
        raise ValueError(
            f"opcompiler_bridge 只支持 float16/float32 存储（新 pass 的 "
            f"checkElementType 同样），收到 dtype={request.dtype!r}"
        )

    if not gpu_hardware_present():
        return _make_ttir_without_gpu(request, m, k, n)

    import torch

    launch = _kernel_launcher(request)
    torch_dtype = dtypes[request.dtype]
    x = torch.empty((m, k), dtype=torch_dtype, device="cuda")
    w = torch.empty((n, k), dtype=torch_dtype, device="cuda")
    out = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    compiled = launch(x, w, out)
    return compiled.asm["ttir"]


def _make_ttir_without_gpu(
    request: OpCompileRequest, m: int, k: int, n: int
) -> str:
    """无 GPU 机器上的 TTIR：不构造设备张量，直接前端编译。"""
    from .cpu_host import make_ttir
    from .kernel_src import NUM_STAGES, linear_kernel, pick_blocks

    # 形状约束与有卡路径共用同一套判定（M 是 2 的幂、K/N 有合法分块）。
    _kernel_launcher(request)
    block_n, block_k = pick_blocks(k, n)
    elem = _TRITON_DTYPE_BY_NAME[request.dtype]
    return make_ttir(
        linear_kernel,
        signature={
            "x_ptr": f"*{elem}",
            "w_ptr": f"*{elem}",
            "out_ptr": f"*{elem}",
            "M": "constexpr",
            "K": "constexpr",
            "N": "constexpr",
            "BLOCK_N": "constexpr",
            "BLOCK_K": "constexpr",
        },
        constexprs={
            "M": m, "K": k, "N": n,
            "BLOCK_N": block_n, "BLOCK_K": block_k,
        },
        num_stages=NUM_STAGES,
    )


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
        raise ToolchainUnavailable(
            f"找不到带 pim-lower-to-emitc 的 triton-opt: {triton_opt}\n"
            "需要先在 FlagTree 里重新编译（该 pass 是本次新增，若安装未重建会"
            "缺这个 pass）。"
        )

    # 第一段到 pim mlir：这一层带 !pim.memdesc、pim.dma_load/store 和真实分块。
    pimir = _run_passes(
        triton_opt,
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
    emitc = _run_passes(
        triton_opt,
        pimir,
        ["-pim-lower-to-emitc", "-convert-func-to-emitc"],
        "emitc",
    )
    return pimir, emitc


def _run_passes(triton_opt: Path, input_text: str, passes: list[str],
                stage: str) -> str:
    """跑一遍 `triton-opt`，返回它的输出文本。"""
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
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


def _make_oplevel_mlir(request: OpCompileRequest) -> str:
    """按请求拼一小段**算子级** PIM IR。

    与 `_make_ttir` 对称：那条路是「一个算子 = 一段 Triton 源码」，这条路是
    「一个算子 = 一段算子级 MLIR」。文本由 `oplevel_kernel` 出，与图编译器
    `oplevel_emitter` 共用同一份——两条路径编出的 IR 是同一份，GML 与 C 才
    不会各说各话。

    形状约定（压平/不压平）的来由见 `oplevel_kernel` 的模块说明。
    """
    if request.op == "matmul":
        # 两个操作数都是 i8，所以不走下面那条 fp16 检查：int4 按 1 字节/元素
        # 存、符号扩展到 [-8, 7]，相位缓冲那一套与它无关。
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"matmul 契约要求 arg_shapes=[(M, K), (K, N)]，收到 "
                f"{request.arg_shapes!r}"
            )
        (m, k), (k2, n) = request.arg_shapes
        if k != k2:
            raise ValueError(
                f"matmul 两个操作数的收缩维对不上：{k} != {k2}")
        if request.group_size and k % request.group_size:
            raise ValueError(
                f"收缩维 {k} 不是 group_size {request.group_size} 的整数倍，"
                f"最后一组会短一截"
            )
        # 元素类型由 `dtype` 定：`int8` 是 w4a8 投影（两个操作数都是 i8，
        # int4 按 1 字节/元素存、符号扩展到 [-8, 7]），`float16` 是注意力
        # 那两个矩阵乘（两个操作数都是 fp16 激活，没有权值侧）。
        fp16 = request.dtype in ("float16", "bfloat16")
        if fp16 and request.group_size:
            raise ValueError("fp16 的矩阵乘没有权值侧，不能带 group_size")
        body = matmul_kernel("kernel", m, k, n, request.group_size,
                             activation=request.activation,
                             sf_multiplier=request.sf_multiplier,
                             fp16=fp16)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "lut":
        if len(request.arg_shapes) != 1:
            raise ValueError(
                f"lut 契约要求 arg_shapes=[输入形状]，收到 {request.arg_shapes!r}")
        body = lut_kernel("kernel", tuple(request.arg_shapes[0]), "silu")
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "eltwise":
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"eltwise 契约要求 arg_shapes=[左, 右]，收到 {request.arg_shapes!r}")
        if tuple(request.arg_shapes[0]) != tuple(request.arg_shapes[1]):
            raise ValueError(
                f"eltwise 两槽同形；收到 {request.arg_shapes!r}。广播形态没有"
                f"编译内核——广播索引要按两个形状算，不是一层循环能覆盖的")
        # 运算种类由请求给。缺省是 `add`（残差加那条路径的既有行为），
        # 但 `mul` 必须能表达——门控乘与 RoPE 乘都在图上。
        kind = request.kind or "add"
        if kind not in ("add", "mul", "sub"):
            raise ValueError(
                f"eltwise 只覆盖 add / mul / sub，收到 kind={kind!r}")
        body = eltwise_kernel("kernel", tuple(request.arg_shapes[0]), kind)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "kv_cache":
        # arg_shapes=[新值形状, (缓存元素数,)]；group_size 非 0 表示散写。
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"kv_cache 契约要求 arg_shapes=[新值形状, (缓存元素数,)]，收到 "
                f"{request.arg_shapes!r}")
        value_shape = tuple(request.arg_shapes[0])
        cache_shape = request.arg_shapes[1]
        if len(cache_shape) != 1:
            raise ValueError(
                f"kv_cache 第二个形状只放缓存元素数，收到 {cache_shape!r}")
        cache_elems = int(cache_shape[0])
        if cache_elems < _prod(value_shape):
            raise ValueError(
                f"缓存 {cache_elems} 个元素装不下一次写的 {_prod(value_shape)} 个")
        # 元素类型跟着请求的 dtype 走：量化后的 KV 是 i8，运行时那条
        # 浮点 KV 路径是 f16。写死 i8 会让后者的元素数差一倍。
        elem = {"float16": "f16", "float32": "f32"}.get(request.dtype, "i8")
        body = kv_cache_kernel("kernel", value_shape, cache_elems,
                               indexed=bool(request.group_size), elem=elem)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "split_heads":
        if len(request.arg_shapes) != 1:
            raise ValueError(
                f"split_heads 契约要求 arg_shapes=[输入形状]，收到 "
                f"{request.arg_shapes!r}")
        shape = tuple(request.arg_shapes[0])
        heads = request.group_size or 0
        if heads <= 0:
            raise ValueError(f"split_heads 需要 group_size 给头数，收到 {heads!r}")
        axis = 1 if len(shape) > 1 else 0
        if shape[axis] % heads:
            raise ValueError(
                f"轴 {axis} 长 {shape[axis]} 分不出 {heads} 个整头")
        body = split_heads_kernel("kernel", shape, axis, heads)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "transpose":
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"transpose 契约要求 arg_shapes=[形状, 轴序]，收到 "
                f"{request.arg_shapes!r}"
            )
        shape, axes = request.arg_shapes
        if sorted(axes) != list(range(len(shape))):
            raise ValueError(f"轴序 {axes} 不是 {len(shape)} 维的排列")
        body = transpose_kernel("kernel", tuple(shape), tuple(axes))
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "reshape":
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"reshape 契约要求 arg_shapes=[原形状, 新形状]，收到 "
                f"{request.arg_shapes!r}"
            )
        shape, out_shape = request.arg_shapes
        if _prod(shape) != _prod(out_shape):
            raise ValueError(
                f"reshape 两侧元素数必须相同：{_prod(shape)} != {_prod(out_shape)}")
        body = reshape_kernel("kernel", tuple(shape), tuple(out_shape))
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "concat":
        # 最后一维是轴号；前面若干维是各输入的形状。
        if len(request.arg_shapes) < 2:
            raise ValueError(
                f"concat 契约要求 arg_shapes=[形状0, 形状1, ...] 且 axis 由 "
                f"group_size 之外的字段给；收到 {request.arg_shapes!r}"
            )
        shapes = [tuple(s) for s in request.arg_shapes]
        if request.group_size is None:
            raise ValueError(
                "concat 需要 group_size 给拼接轴，省略会静默沿轴 0 拼")
        axis = request.group_size
        if len({s[:axis] + s[axis + 1:] for s in shapes}) != 1:
            raise ValueError(f"concat 除轴 {axis} 外的维度必须一致：{shapes}")
        body = concat_kernel("kernel", shapes, axis)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "convert":
        if len(request.arg_shapes) != 1:
            raise ValueError(
                f"convert 契约要求 arg_shapes=[输入形状]，收到 "
                f"{request.arg_shapes!r}")
        if not request.out_dtype:
            raise ValueError(
                "convert 契约要求 out_dtype 给出目标元素类型；缺了它降级侧"
                "只能猜一个，而猜错的后果是整块按另一种类型重解释")
        body = convert_kernel("kernel", tuple(request.arg_shapes[0]),
                              request.out_dtype, src_dtype=request.dtype)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "mask":
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"mask 契约要求 arg_shapes=[分数形状, 掩码形状]，收到 "
                f"{request.arg_shapes!r}")
        body = mask_kernel("kernel", tuple(request.arg_shapes[0]),
                           tuple(request.arg_shapes[1]))
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.dtype != "float16":
        raise ValueError(
            f"算子级路径只支持 float16 存储（相位缓冲按 fp16 落盘），"
            f"收到 dtype={request.dtype!r}"
        )
    if request.op == "normalize":
        if len(request.arg_shapes) != 1 or len(request.arg_shapes[0]) != 2:
            raise ValueError(
                f"normalize 契约要求 arg_shapes=[(rows, cols)]，收到 "
                f"{request.arg_shapes!r}")
        rows, cols = request.arg_shapes[0]
        body = normalize_kernel("kernel", rows, cols, cols)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "gather":
        # 两个输入：表与索引。其余算子只有一个，所以形状校验按算子分。
        if len(request.arg_shapes) != 2:
            raise ValueError(
                f"gather 契约要求 arg_shapes=[表形状, 索引形状]，收到 "
                f"{request.arg_shapes!r}"
            )
        (vocab, hidden), ids_shape = request.arg_shapes
        # 索引宽度跟着请求走：图上的 token id 是 int64，按 int32 读会把
        # 相邻两个 id 的字节拼成一个行号，整张表查错。
        index_dtype = {"int32": "i32", "int64": "i64"}.get(request.out_dtype, "i32")
        body = gather_kernel("kernel", vocab, hidden, _prod(ids_shape),
                             index_dtype=index_dtype)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if request.op == "rope":
        # 保留 rank-4：广播沿 head 轴发生，压平会把那个轴抹掉。
        if len(request.arg_shapes) != 1 or len(request.arg_shapes[0]) != 4:
            raise ValueError(
                f"rope 契约要求 arg_shapes=[(1, heads, seq, head_dim)]，收到 "
                f"{request.arg_shapes!r}"
            )
        _, heads, seq, head_dim = request.arg_shapes[0]
        body = rope_kernel("kernel", heads, seq, head_dim,
                           tail_card_value=request.tail_card_value)
        return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
                f"{body}\n}}\n")

    if len(request.arg_shapes) != 1:
        raise ValueError(
            f"{request.op} 契约要求 arg_shapes=[输入形状]，收到 "
            f"{request.arg_shapes!r}"
        )

    shape = request.arg_shapes[0]
    if request.op == "softmax":
        *leading, cols = shape
        body = softmax_kernel("kernel", _prod(leading), cols)
    elif request.op == "dynamic_quant":
        numel = _prod(shape)
        group_size = request.group_size
        if not group_size or group_size <= 0 or numel % group_size:
            raise ValueError(
                f"dynamic_quant 需要能整除 {numel} 个元素的 group_size，"
                f"收到 {group_size!r}"
            )
        body = dynamic_quant_kernel("kernel", numel, numel // group_size,
                                    group_size)
    else:
        raise NotImplementedError(
            f"算子级路径还不覆盖 {request.op!r}；已覆盖："
            f"{sorted(_OPLEVEL_OPS)}"
        )

    # `pim.target` 必须在：手写的 MLIR 不经过 `convert-triton-to-pim`，
    # 没有它 Triton 的 tensor 元素数 2 的幂限制会挡下 11008 这类维度。
    return (f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
            f"{body}\n}}\n")


def _run_oplevel_triton_opt(mlir_text: str) -> tuple[str, str]:
    """算子级 IR 走一遍：融合 → 展开 → 校验 → 降到 EmitC。

    与 A 路的两次调用同构，只是 pass 列表不同：这条没有 DMA，也就没有
    `convert-triton-to-pim` / `tile-to-budget` / `explicit-dma` 三段。
    """
    triton_opt = _triton_opt()
    if not triton_opt.is_file():
        raise ToolchainUnavailable(
            f"找不到带 pim-lower-to-emitc 的 triton-opt: {triton_opt}\n"
            "需要先在 FlagTree 里重新编译。"
        )
    pimir = _run_passes(
        triton_opt, mlir_text,
        ["-pim-fuse-activation", "-pim-expand-phases",
         "-pim-verify-gml-contract"],
        "pim mlir",
    )
    emitc = _run_passes(
        triton_opt, pimir,
        ["-pim-lower-to-emitc", "-convert-func-to-emitc"],
        "emitc",
    )
    return pimir, emitc


def _prod(shape) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _translate_to_c(emitc_text: str) -> str:
    mlir_translate = _mlir_translate()
    if not mlir_translate.is_file():
        raise ToolchainUnavailable(f"找不到 mlir-translate: {mlir_translate}")
    proc = subprocess.run(
        [str(mlir_translate), "--mlir-to-cpp"],
        input=emitc_text,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mlir-translate 失败:\n{proc.stderr}")
    return proc.stdout


def _parse_signature(
    c_source: str,
) -> tuple[str, list[str], list[bool]]:
    """从生成的 C 源码解析函数名、参数元素类型、以及哪些是**按值**标量。

    第三项必须带上：`argtypes` 只记元素类型，`int32_t` 既可能是
    `int32_t*`（缓冲）也可能是 `int32_t`（按值）。丢了这一位，调用方只能
    把标量也按指针塞进去，C 侧读到的就是那个地址的低 32 位。
    """
    match = _SIG_RE.search(c_source)
    if not match:
        raise RuntimeError(
            f"无法从生成的 C 源码中解析出函数签名:\n{c_source}"
        )
    symbol, params_str = match.group(1), match.group(2)
    argtypes: list[str] = []
    by_value: list[bool] = []
    for param in (p.strip() for p in params_str.split(",") if p.strip()):
        # 一条按值标量：步计数器这类，它是个数不是缓冲，传指针反而要调用方
        # 先找地方把它存起来。守卫要挡的是偏移量与 memref descriptor —— 多字段
        # 的聚合类型，那种签名说明降级把地址算错了。
        if m := _SCALAR_PARAM_RE.match(param):
            argtypes.append(m.group(1))
            by_value.append(True)
            continue
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
        by_value.append(False)
    return symbol, argtypes, by_value


def compile_op(request: OpCompileRequest, *, force: bool = False) -> OpCompileResult:
    """编译算子请求并返回共享库描述，结果按请求参数缓存。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(request)
    so_path = _CACHE_DIR / f"{key}.so"
    meta_path = _CACHE_DIR / f"{key}.meta"
    pimir_path = _CACHE_DIR / f"{key}.pimir.mlir"

    if not force and so_path.is_file() and meta_path.is_file():
        lines = meta_path.read_text().splitlines()
        symbol, argtypes_str = lines[0], lines[1]
        flags = lines[2] if len(lines) > 2 else ""
        # pim mlir 与 `.so` 一起缓存，命中时直接读回来，成本模型不必重编。
        cached_pimir = (
            pimir_path.read_text() if pimir_path.is_file() else None
        )
        return OpCompileResult(
            so_path=str(so_path),
            symbol=symbol,
            argtypes=argtypes_str.split(","),
            by_value=[c == "1" for c in flags.split(",")] if flags else [],
            pimir=cached_pimir,
            pimir_path=str(pimir_path) if cached_pimir is not None else None,
        )

    if request.op in _OPLEVEL_OPS:
        # B 路：整算子级 IR → 融合/展开/校验 → EmitC。没有 DMA 可显式化，
        # 所以不走 A 路那三段；降级 pass 是同一个。
        mlir_text = _make_oplevel_mlir(request)
        pimir_text, emitc_text = _run_oplevel_triton_opt(mlir_text)
    elif request.op == "linear":
        ttir = _make_ttir(request)
        pimir_text, emitc_text = _run_triton_opt(ttir, request.hardware)
    else:
        # 名字不认识就在这里断掉。放到 A 路里面判会让 `arg_shapes[1]` 先解包，
        # 未知名拿到的是 IndexError，看现场只看到"列表越界"，看不出是名字错了。
        raise NotImplementedError(
            f"没有 {request.op!r} 的编译路径：既不在整算子级表"
            f"（{sorted(_OPLEVEL_OPS)}），也不是 A 路的 linear"
        )
    c_source = _translate_to_c(emitc_text)
    symbol, argtypes, by_value = _parse_signature(c_source)

    # 使用进程和线程唯一的临时路径编译共享库。
    tmp_tag = f"{os.getpid()}.{threading.get_ident()}"
    c_path = _CACHE_DIR / f"{key}.{tmp_tag}.c"
    so_tmp_path = _CACHE_DIR / f"{key}.{tmp_tag}.so"
    # 生成的 C 代码需要 `malloc` 和 `free`。
    try:
        # `stdbool.h`：比较运算降出来的是 C 的 `bool`（RoPE 半旋转要按半区选边）。
        c_path.write_text(
            "#include <stdint.h>\n#include <stdlib.h>\n#include <math.h>\n"
            "#include <stdbool.h>\n"
            + c_source)
        proc = subprocess.run(
            # `-lm`：算子级的 LUT 相用 `expf`。glibc 新版已把 libm 并进 libc，
            # 显式带上，老环境也不会漏。
            ["gcc", "-shared", "-fPIC", "-O2", "-o", str(so_tmp_path),
             str(c_path), "-lm"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gcc 编译生成的 C 失败:\n{proc.stderr}")
        os.replace(so_tmp_path, so_path)
    finally:
        c_path.unlink(missing_ok=True)
        so_tmp_path.unlink(missing_ok=True)  # no-op if os.replace already moved it

    meta_path.write_text(
        f"{symbol}\n{','.join(argtypes)}\n"
        f"{','.join('1' if v else '0' for v in by_value)}")
    # pim mlir 与 `.so` 一起落盘：下次命中缓存时成本模型直接读它，不必重编。
    pimir_path.write_text(pimir_text)
    return OpCompileResult(
        so_path=str(so_path),
        symbol=symbol,
        argtypes=argtypes,
        by_value=by_value,
        pimir=pimir_text,
        pimir_path=str(pimir_path),
    )


def load_kernel(result: OpCompileResult):
    """加载共享库并返回已设置参数类型的 ``ctypes`` 函数。"""
    lib = ctypes.CDLL(result.so_path)
    fn = getattr(lib, result.symbol)
    # 按值标量要用它自己的 ctypes 类型：设成 `c_void_p` 会把一个数当指针
    # 传，C 侧读到的就是那个地址的低位。`by_value` 为空（旧产物）时按
    # 全指针处理，与原先一致。
    fn.argtypes = [
        _CTYPE_BY_C_ELEM[t] if by_value else ctypes.c_void_p
        for t, by_value in zip(result.argtypes,
                               result.by_value or [False] * len(result.argtypes))
    ]
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
