"""定义 GeneSim 算子类型与代表实现及其形状。

配方分两路，同一个 `OpRecipe` 装：

- **A 路**（`--ir-level ttir/pimir`）：`build` 跑一遍 FlagGems 算子，抓下发的
  Triton kernel，从 TTIR / pim mlir 里抽成本。需要 GPU。
- **B 路**（`--ir-level oplevel`）：`pimir` 发一段整算子级 PIM IR，经
  `-pim-fuse-activation` + `-pim-expand-phases` 展开成相位链再抽成本。
  不发 kernel、不碰 GPU。

两路给的是同一个算子的成本，口径不同：A 路量的是 FlagGems 那条 trace 编译
出来的 kernel，B 路量的是本仓自己发的整算子 IR。`source_name` 因此也要分开
记（`flag_gems.ops.softmax` vs `pim.softmax`），否则 sidecar 里分不清一笔成本
是从哪条路来的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from opcompiler_bridge.oplevel_kernel import (
    concat_kernel,
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

# 目标串。手写的 MLIR 不经过 convert-triton-to-pim，得自己带上它放开张量尺寸
# 的 2 的幂限制（理由见 opcompiler_bridge/oplevel_kernel.py）。
_PIM_TARGET = 'pim:v1'

# 量化组宽。与 `opcompiler_bridge/oplevel_emitter.py` 的 `ROPE_DQ_GROUP_SIZE`
# 一致，两边不能各挑一个。
_DQ_GROUP_SIZE = 128


def _probe_device() -> str:
    """探针张量该建在哪个设备上。

    这些张量只是用来触发一次 FlagGems 调用、好让 `capture_kernels` 抓到内核并从中
    取出 pim mlir——算的是什么值无关紧要，跑在哪个设备上也无关紧要。有卡时仍走
    cuda（与历史行为一致），无卡时落到 cpu。
    """
    from opcompiler_bridge.cpu_host import gpu_hardware_present

    return "cuda" if gpu_hardware_present() else "cpu"

# 使用 GeneSim 模板成本的算子类型：本桥接不为它们编译 FlagGems 代表实现，
# 直接保留 model_parser 写进 IR 的模板系数，并记入 sidecar 的 coverage.template。
#
# 逐元素/归约类（RMSNORM、SILU、VECTOR_ADD、VECTOR_MUL）：GeneSim 侧的
# model_parser 已经给出 flops_coeffs / data_bytes_coeffs，而且它们在
# pim_compiler 里有各自的 trace 编译器，成本由那条路径负责。这里不硬编一套
# FlagGems 配方去覆盖，否则等于引入一组未经校准的数值。
#
# 注意力图上的算子（ROPE、MASK、QUANT、MEM_COPY、GATHER 与四个视图类）：
# 本仓没有编它们的 FlagGems 代表实现，A 路照旧用 GeneSim 写的模板系数。
#
# **只挡 A 路**。B 路（`--ir-level oplevel`）量的是本仓自己发的整算子 IR，
# 不掺 FlagGems、也无所谓校准，所以这些在 B 路上照常桥接（`build=None`，
# `pimir` 有值），这样 7.3 要的"rms 非零"才成立。见 `export_costs_to_genesim`
# 里对 `UNCOVERED_OP_TYPES` 的那个 `ir_level` 判断。
#
# 图的边界节点（MODEL_INPUT、MODEL_OUTPUT）：零成本占位，flops 和 data_bytes
# 都是 0、也没有系数，本来就没有什么可测量的。两条路都保留模板。
UNCOVERED_OP_TYPES = frozenset({
    "GELU",
    "RMSNORM",
    "SILU",
    "VECTOR_ADD",
    "VECTOR_MUL",
    "MODEL_INPUT",
    "MODEL_OUTPUT",
    "GATHER",
    "QUANT",
    "ROPE",
    "MASK",
    "MEM_COPY",
    "TRANSPOSE",
    "RESHAPE",
    "SPLIT",
    "CONCAT",
})


@dataclass(frozen=True)
class ShapePoint:
    """一个编译代表点：prefill 用 (Tq=seq_len, Tp=0)，decode 用 (Tq=1, Tp=seq_len)。"""
    tq: int
    tp: int

    @property
    def lkv(self) -> int:
        """Tp+Tq，即 attention 的 KV 长度。"""
        return self.tp + self.tq

    @property
    def label(self) -> str:
        return f"Tq={self.tq},Tp={self.tp}"


@dataclass
class OpRecipe:
    """一个 GeneSim op_type 的编译配方。"""
    source_name: str                       # FlagGems 实现名，写入 sidecar
    # A 路的执行体：无参可调用，跑一遍算子让 capture_kernels 抓到内核。
    # 只有 B 路的算子留 None（它们仍算未覆盖，保留模板成本）。
    build: Optional[Callable] = None
    expected_kernels: int = 1
    # B 路（`--ir-level oplevel`）的编译代表实现：一段整算子级 PIM IR。
    # B 路不经 FlagGems、也不需要 GPU——发 IR 再展开成相位链就能量成本。
    pimir: Optional[Callable[[], str]] = None
    # B 路写进 sidecar 的 `source_name`，即方言里的 mnemonic。
    mnemonic: str = ""


def _module(body: str) -> str:
    """把一段 `tt.func` 包成可被 triton-opt 解析的 module。"""
    return f'module attributes {{pim.target = "{_PIM_TARGET}"}} {{\n{body}\n}}\n'


def _normalize_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """RMSNorm 锚点：[Tq, hidden] 的激活配 [hidden] 的缩放。"""
    cols = dims["hidden_size"]
    return _module(normalize_kernel("rmsnorm", point.tq, cols, cols))


def _attention_matmul_body(func: str, m: int, k: int, n: int) -> str:
    """注意力那两次矩阵乘的 body：驻留的是缓存里的 K/V 片，不是模型权值。

    `matmul_kernel` 只发 `stationarity = weight`（投影那条），注意力照抄会把
    KV 缓存说成模型权值。搬运量恰好一样，所以错了也看不出来——校验器要求
    三者一致（`kv` 配 `activation_as_weight` 的绑定、并给上 `bIsActivation`），
    这里就按它给。
    """
    a_ty = f"tensor<{m}x{k}xi8>"
    b_ty = f"tensor<{k}x{n}xi8>"
    out_ty = f"tensor<{m}x{n}xi8>"
    return f"""  tt.func @{func}(%a: {a_ty}, %b: {b_ty}) {{
    %y = pim.matmul %a, %b {{datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>, weightBinding = #pim.weight_binding<format = weight, role = activation_as_weight, elemBits = 8>, stationarity = #pim.stationarity<kv>, bIsActivation}} : {a_ty}, {b_ty} -> {out_ty}
    tt.return
  }}"""


def _matmul_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """矩阵乘的代表形状取注意力得分那一次：[Tq, head_dim] x [head_dim, Tp+Tq]。"""
    return _module(_attention_matmul_body("matmul", point.tq, dims["head_dim"],
                                          point.lkv))


def _softmax_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """逐头 softmax：真实图里每个头单独一个节点，形状是 [Tq, Tp+Tq]，
    不把 heads 折进行数——那会让一个节点的搬运量放大 num_heads 倍。"""
    return _module(softmax_kernel("softmax", point.tq, point.lkv))


def _mask_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """加性因果掩码：分数 `[heads, Tq, Tp+Tq]` 配按末轴广播的掩码。

    两种几何由 `layout` 分开：decode 是"一行对整段缓存"（`vector`，除末轴外
    全是 1），prefill 是"每个位置只看见它前面"的三角（`causal_tril`，末两轴
    相等）。形状相近时单看形状分不出来，所以必须把 layout 显式传给
    `mask_kernel`。
    """
    heads, tq, lkv = dims["num_heads"], point.tq, point.lkv
    if tq == 1:
        return _module(mask_kernel("mask", (heads, tq, lkv), (1, 1, lkv),
                                   layout="vector"))
    return _module(mask_kernel("mask", (heads, tq, lkv), (tq, tq),
                               layout="causal_tril"))


def _rope_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """RoPE 保留 rank-4：cos/sin 沿 head 轴广播，压平会把那个轴抹掉。"""
    return _module(rope_kernel("rope", dims["num_heads"], point.tq,
                               dims["head_dim"]))


def _lut_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """查表激活。Silu 作用在 FFN 中间宽度上，不是 hidden_size。"""
    return _module(lut_kernel("lut", (point.tq, dims["ffn_dim"]), "silu"))


def _eltwise_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """逐元素二元运算，按残差加的宽度取形状。"""
    return _module(eltwise_kernel("eltwise", (point.tq, dims["hidden_size"]),
                                  "add"))


def _dynamic_quant_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """动态量化压成 `[1, numel]`：逐组归约，压平后行主序下分组等价。"""
    numel = point.tq * dims["hidden_size"]
    return _module(dynamic_quant_kernel("dq", numel, numel // _DQ_GROUP_SIZE,
                                        _DQ_GROUP_SIZE))


def _kv_cache_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """KV 写入：一个 token 一个头的 K 或 V 那一段，散写到缓存行上。"""
    value = (1, 1, 1, dims["head_dim"])
    return _module(kv_cache_kernel("kv_cache", value,
                                   point.lkv * dims["head_dim"], indexed=True))


def _gather_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """词嵌入查表：取 Tq 行，行宽是 hidden。"""
    return _module(gather_kernel("gather", 32000, dims["hidden_size"], point.tq))


def _transpose_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """转置：头轴换到 KV 轴前面。"""
    shape = (dims["num_heads"], point.tq, dims["head_dim"])
    return _module(transpose_kernel("transpose", shape, (1, 0, 2)))


def _reshape_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """改形状：`[Tq, heads*head_dim]` 与 `[Tq, heads, head_dim]` 元素顺序不变。"""
    heads, head_dim = dims["num_heads"], dims["head_dim"]
    return _module(reshape_kernel("reshape", (point.tq, heads * head_dim),
                                  (point.tq, heads, head_dim)))


def _split_heads_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """按头拆开打包的投影：轴 1 是 heads*head_dim，拆成 heads 份。"""
    heads, head_dim = dims["num_heads"], dims["head_dim"]
    return _module(split_heads_kernel("split_heads", (point.tq, heads * head_dim),
                                      1, heads))


def _concat_ir(dims: Dict[str, int], point: ShapePoint) -> str:
    """沿末轴把 num_heads 个头拼回 hidden_size。"""
    head_dim = dims["head_dim"]
    pieces = [(point.tq, head_dim)] * dims["num_heads"]
    return _module(concat_kernel("concat", pieces, 1))


# 表 1.2.4 的 14 个设备侧 mnemonic，计算 10 + 视图 4。B 路每个都得能发 IR、
# 展开、抽出成本：计算类 flops > 0，视图类搬运 > 0。
MNEMONICS = (
    "pim.normalize", "pim.matmul", "pim.softmax", "pim.mask", "pim.rope",
    "pim.lut", "pim.eltwise", "pim.dynamic_quant", "pim.kv_cache", "pim.gather",
    "pim.transpose", "pim.reshape", "pim.split_heads", "pim.concat",
)

# 每个 mnemonic 的代表形状。视图类四个不在此列之外另计（见 MNEMONICS 后半段）。
_OPLEVEL_IR: Dict[str, Callable[[Dict[str, int], ShapePoint], str]] = {
    "pim.normalize": _normalize_ir,
    "pim.matmul": _matmul_ir,
    "pim.softmax": _softmax_ir,
    "pim.mask": _mask_ir,
    "pim.rope": _rope_ir,
    "pim.lut": _lut_ir,
    "pim.eltwise": _eltwise_ir,
    "pim.dynamic_quant": _dynamic_quant_ir,
    "pim.kv_cache": _kv_cache_ir,
    "pim.gather": _gather_ir,
    "pim.transpose": _transpose_ir,
    "pim.reshape": _reshape_ir,
    "pim.split_heads": _split_heads_ir,
    "pim.concat": _concat_ir,
}


def oplevel_ir(dims: Dict[str, int], mnemonic: str, point: ShapePoint) -> str:
    """按 mnemonic 发一段代表形状的整算子级 PIM IR。

    这是 B 路的"编译代表实现"：与 A 路的 FlagGems 配方一一对应，量的是同一个
    算子的成本，只是从本仓自己发的 IR 里量。表里没有的 mnemonic 直接抛——
    缺名字要立刻暴露，不能悄悄跳过（跳过等于成本 0，而 0 看起来很正常）。
    """
    if mnemonic not in _OPLEVEL_IR:
        raise KeyError(f"没有 {mnemonic} 的整算子级代表实现；"
                       f"表 1.2.4 的 14 个是 {list(MNEMONICS)}")
    return _OPLEVEL_IR[mnemonic](dims, point)


def _linear(dims: Dict[str, int], point: ShapePoint, in_features: int, out_features: int):
    """FlagGems linear：GeneSim 的 GEMM 都是 [Tq, in] x [in, out]。"""
    import torch

    x = torch.randn(point.tq, in_features, device=_probe_device(), dtype=torch.float16)
    w = torch.randn(out_features, in_features, device=_probe_device(), dtype=torch.float16)
    b = torch.randn(out_features, device=_probe_device(), dtype=torch.float16)
    return lambda: torch.nn.functional.linear(x, w, b)


def _bmm(dims: Dict[str, int], m: int, k: int, n: int):
    """FlagGems bmm，单 head（batch=1）。"""
    import torch

    a = torch.randn(1, m, k, device=_probe_device(), dtype=torch.float16)
    b = torch.randn(1, k, n, device=_probe_device(), dtype=torch.float16)
    return lambda: torch.bmm(a, b)


def _softmax(dims: Dict[str, int], rows: int, cols: int):
    import torch

    x = torch.randn(rows, cols, device=_probe_device(), dtype=torch.float16)
    return lambda: torch.softmax(x, dim=-1)


def build_recipes(dims: Dict[str, int]) -> Dict[str, Callable[[ShapePoint], OpRecipe]]:
    """按模型维度返回各算子类型的配方工厂。"""
    hidden = dims["hidden_size"]
    head_dim = dims["head_dim"]

    def gemm(point: ShapePoint, in_features: int, out_features: int) -> OpRecipe:
        return OpRecipe(
            source_name="flag_gems.ops.linear",
            build=_linear(dims, point, in_features, out_features),
            mnemonic="pim.matmul",
            pimir=lambda: _module(matmul_kernel("linear", point.tq, in_features,
                                                out_features)),
        )

    def gemv_score(point: ShapePoint) -> OpRecipe:
        # 注意力得分的矩阵乘形状。
        return OpRecipe(
            source_name="flag_gems.ops.bmm (score 代表实现)",
            build=_bmm(dims, point.tq, head_dim, point.lkv),
            mnemonic="pim.matmul",
            pimir=lambda: _matmul_ir(dims, point),
        )

    def softmax(point: ShapePoint) -> OpRecipe:
        # 注意力得分的 Softmax 形状。
        return OpRecipe(
            source_name="flag_gems.ops.softmax",
            build=_softmax(dims, point.tq, point.lkv),
            mnemonic="pim.softmax",
            pimir=lambda: _softmax_ir(dims, point),
        )

    def gemv_context(point: ShapePoint) -> OpRecipe:
        # 注意力上下文的矩阵乘形状。
        return OpRecipe(
            source_name="flag_gems.ops.bmm (context 代表实现)",
            build=_bmm(dims, point.tq, point.lkv, head_dim),
            mnemonic="pim.matmul",
            pimir=lambda: _module(_attention_matmul_body(
                "context", point.tq, point.lkv, head_dim)),
        )

    def bpath(mnemonic: str, builder) -> Callable[[ShapePoint], OpRecipe]:
        """只有 B 路的配方：A 路保留模板成本，理由见 UNCOVERED_OP_TYPES。

        成本公式不在这里重写一份：B 路直接发 `opcompiler_bridge/oplevel_kernel`
        里该 mnemonic 的整算子 IR，形状取它自己的代表点。
        """
        def make(point: ShapePoint) -> OpRecipe:
            return OpRecipe(source_name=mnemonic, mnemonic=mnemonic,
                            pimir=lambda: builder(dims, point))
        return make

    def _binary(source_name: str, kind: str):
        def make(point: ShapePoint) -> OpRecipe:
            return OpRecipe(
                source_name=source_name,
                mnemonic="pim.eltwise",
                pimir=lambda: _module(eltwise_kernel(
                    "eltwise", (point.tq, dims["hidden_size"]), kind)),
            )
        return make

    return {
        "GEMM": gemm,
        "GEMV_SCORE": gemv_score,
        "SOFTMAX": softmax,
        "GEMV_CONTEXT": gemv_context,
        "RMSNORM": bpath("pim.normalize", _normalize_ir),
        "SILU": bpath("pim.lut", _lut_ir),
        "VECTOR_ADD": _binary("pim.eltwise add", "add"),
        "VECTOR_MUL": _binary("pim.eltwise mul", "mul"),
        # 注意力与视图类：模型骨架里真有这些节点，A 路留模板、B 路逐个桥接。
        "GATHER": bpath("pim.gather", _gather_ir),
        "QUANT": bpath("pim.dynamic_quant", _dynamic_quant_ir),
        "ROPE": bpath("pim.rope", _rope_ir),
        "MASK": bpath("pim.mask", _mask_ir),
        "MEM_COPY": bpath("pim.kv_cache", _kv_cache_ir),
        "TRANSPOSE": bpath("pim.transpose", _transpose_ir),
        "RESHAPE": bpath("pim.reshape", _reshape_ir),
        "SPLIT": bpath("pim.split_heads", _split_heads_ir),
        "CONCAT": bpath("pim.concat", _concat_ir),
    }


def mnemonic_of(op_type: str) -> str | None:
    """算子类型对应的方言 mnemonic，不带 `pim.` 前缀。

    从 `build_recipes` 取，放置导出与成本抽取共用这一份，不再各写一张表。
    维度只用来把配方建出来，mnemonic 与维度无关。
    """
    recipes = build_recipes({"hidden_size": 64, "head_dim": 64, "num_heads": 1,
                             "ffn_dim": 64})
    factory = recipes.get(op_type)
    if factory is None:
        return None
    try:
        recipe = factory(ShapePoint(tq=1, tp=0))
    except TypeError:
        # GEMM 的配方还要 in/out 两个宽度，mnemonic 与宽度无关，给个占位即可。
        recipe = factory(ShapePoint(tq=1, tp=0), 64, 64)
    return recipe.mnemonic.removeprefix("pim.") or None


MNEMONIC_OF = {
    op_type: mnemonic
    for op_type in (
        "GEMM", "GEMV_SCORE", "GEMV_CONTEXT", "SOFTMAX", "RMSNORM", "SILU",
        "VECTOR_ADD", "VECTOR_MUL", "GATHER", "QUANT", "ROPE", "MASK",
        "MEM_COPY", "TRANSPOSE", "RESHAPE", "SPLIT", "CONCAT",
    )
    if (mnemonic := mnemonic_of(op_type))
}


def gemm_features(op_dict: dict) -> tuple:
    """从算子形状读取 GEMM 的输入和输出特征数。"""
    in_shape = op_dict["input_shapes"][0]
    out_shape = op_dict["output_shapes"][0]
    assert in_shape[0] == "Tq" and out_shape[0] == "Tq", \
        f"GEMM shape 不是 [Tq, N] 形式: {in_shape} -> {out_shape}"
    return int(in_shape[1]), int(out_shape[1])


def flash_attention_probe(dims: Dict[str, int], point: ShapePoint):
    """构造一次融合 attention 调用，用于测量总成本。"""
    import torch

    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    q = torch.randn(1, num_heads, point.tq, head_dim, device=_probe_device(), dtype=torch.float16)
    k = torch.randn(1, num_heads, point.lkv, head_dim, device=_probe_device(), dtype=torch.float16)
    v = torch.randn(1, num_heads, point.lkv, head_dim, device=_probe_device(), dtype=torch.float16)
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
