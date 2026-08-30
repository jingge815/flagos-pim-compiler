"""IR 成本分析器：从 FlagTree 编出的 TTIR 或 pim mlir 抽单算子成本。

方案依据：spec.md 问题 4 三.(3)。TTIR 是 tile 级表示，不是算子级：
kernel 体内只有一块 tile 的运算（如 `tt.dot` on `tensor<32x32>`），
全局规模由 grid 和 `scf.for` 的循环次数决定，而这两者都不在 TTIR 里
（循环边界形如 `ceil(%K/32)`，%K 是运行时 i32 参数）。

因此算子级 flops 按下式还原：

    flops = grid_size x Σ(每个 scf.for 帧: 循环体 tile 运算 x 该帧循环次数)
                      + 循环外的 tile 运算

按帧独立记次数：一个 kernel 可以有多个顺序排列的顶层循环（flash_fwd
就有两个，分别处理 masking 段与非 masking 段），共用一个 trip 会把后
一个循环的次数错用到前一个身上。

grid 与 kernel 实参由 flagtree_driver 在 launch 处捕获后传进来，
所以循环次数是**代入实参求值**得到的，不是猜的。

data_bytes 不从这里出——按方案三.(3)，它取「算子对外净读写字节数」
（Σ输入 shape + Σ输出 shape，按 IR 里的真实 dtype），不统计 tile 内部
的重复搬运，由调用方按全局 shape 算。本模块只负责 flops、dtype，以及
pim mlir 上额外的 mram_traffic_bytes。

## 两层 IR 的差别

`convert-triton-to-pim` + `pim-explicit-dma` 只做两件事：给张量类型加
`#pim.tasklet_tiled` 布局，把 `tt.load`/`tt.store` 换成
`pim.wram_alloc` + `pim.dma_load/store` + `pim.barrier` + `pim.wram_load/store`。
`tt.dot`、`arith.*`、`scf.for` **原样保留**（实测 linear_kernel 的
`scf.for` 与 `tt.dot` 数量在两层完全一致）。

所以 **flops 在两层上必然逐位相等**，这不是巧合而是 pass 语义的直接后果，
测试里以此作断言。pim mlir 真正新增的是 `mram_traffic_bytes`——按方案
三.(3)，它统计 MRAM↔WRAM 的**显式搬运指令**字节数：WRAM 容量远小于算子
输入输出，同一份 MRAM 数据要按 tile 分批搬入多次，这个累计量系统性大于
「净读写字节数」。它只落 sidecar，**不进 `data_bytes`**（否则会把跨 VPU
传输量估算污染成虚高值）。

注意正则必须吃下 `#pim.tasklet_tiled<...>` 尾缀：pim mlir 的 tensor 类型
形如 `tensor<32x32xf16, #pim.tasklet_tiled<{...}>>`，而不是 TTIR 的
`tensor<32x32xf16>`。早期只锚定 TTIR 形态的正则在 pim mlir 上 `tt.dot`
命中 0 次，会把矩阵乘静默算成 0 flops。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 每元素浮点运算权重。超越函数按硬件上多条指令计，与 GeneSim 模板
# 对 GELU 用 10 flops/元素 的口径同量级。
_ELEMENTWISE_FLOPS = {
    "arith.addf": 1.0,
    "arith.subf": 1.0,
    "arith.mulf": 1.0,
    "arith.divf": 1.0,
    "arith.maxnumf": 1.0,
    "arith.minnumf": 1.0,
    "arith.negf": 1.0,
    "math.exp": 4.0,
    "math.exp2": 4.0,
    "math.log": 4.0,
    "math.sqrt": 4.0,
    "math.rsqrt": 4.0,
    "math.tanh": 8.0,
    "math.erf": 8.0,
}

_DTYPE_BYTES = {
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
}

# tensor<32x64xf16> / tensor<128xf32> / tensor<32x64xf16, #pim.tasklet_tiled<{..}>>
#   -> ("32x64", "f16")
# 尾部允许 `,`（pim mlir 带 layout encoding）或 `>`（TTIR 裸类型）。指针张量
# （`tensor<32x32x!tt.ptr<f16>, ...>`）不匹配——`!` 不在 `[a-z0-9]` 里——正合
# 需要：指针张量不参与浮点运算计数。
_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x([a-z0-9]+)[,>]")
# `!pim.memdesc<32x32xf16, #pim.wram>` -> ("32x32", "f16", "wram")
_MEMDESC_RE = re.compile(r"!pim\.memdesc<([0-9x]+)x([a-z0-9]+),\s*#pim\.(wram|mram)>")
_DMA_RE = re.compile(r"(?<![\w.])pim\.(dma_load|dma_store)(?![\w.])")
_WRAM_ALLOC_RE = re.compile(
    r"(?<![\w.])pim\.wram_alloc(?![\w.]).*?!pim\.memdesc<([0-9x]+)x([a-z0-9]+),"
)
_DOT_RE = re.compile(r"(?<![\w.])tt\.dot(?![\w.])")
_SCF_FOR_RE = re.compile(r"scf\.for\s+%\S+\s*=\s*(\S+)\s+to\s+(\S+)\s+step\s+(\S+)")
_SSA_RE = re.compile(r"^\s*(%[\w#]+)\s*=\s*(\S+)(.*)$")


@dataclass
class WramBuffer:
    """一个 `pim.wram_alloc` 分配出的 staging buffer。"""
    shape: str            # 形如 "32x64"
    dtype: str
    bytes: int


@dataclass
class KernelCost:
    """单个 kernel 的成本分析结果。"""
    kernel_name: str
    flops: float
    dtype: str
    element_bytes: int
    grid: tuple
    tile_flops_per_program: float = 0.0        # 单个 program 的浮点运算数
    loop_trip_counts: List[Optional[float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # 以下仅 pim mlir 有值（TTIR 上没有 pim.* 算子，恒为 None / 空）
    mram_traffic_bytes: Optional[float] = None   # MRAM↔WRAM 显式搬运字节，只进 sidecar
    wram_bytes_used: Optional[int] = None        # pass 统计的 WRAM 分配总量
    wram_bytes_budget: Optional[int] = None      # module 上声明的 WRAM 预算
    wram_buffers: List[WramBuffer] = field(default_factory=list)
    dma_ops: int = 0
    dma_ops_with_layout: int = 0                 # 指针分析证明了 stride 的那些
    # add_tile_to_budget 接入后新增：真实硬件预算 + 按预算选出的 tile 形状。
    # mram_bytes_budget/dma_align 是传给 convert-triton-to-pim 的硬件契约本身
    # （module 属性 pim.mram-bytes/pim.dma-align）；tile_m/n/k/wram_bytes 是
    # pim-tile-to-budget 按该预算算出的选择（module 属性 pim.tile-*）。
    mram_bytes_budget: Optional[int] = None
    dma_align: Optional[int] = None
    tile_m: Optional[int] = None
    tile_n: Optional[int] = None
    tile_k: Optional[int] = None
    tile_wram_bytes: Optional[int] = None


def _tensor_numel(dims: str) -> int:
    n = 1
    for d in dims.split("x"):
        n *= int(d)
    return n


class _ConstFolder:
    """对 TTIR 顶层做常量折叠，用于求 scf.for 的循环次数。

    只处理循环边界会用到的整数运算。kernel 实参（%K、%stride_* 等）由
    driver 捕获的实参表提供；折不出来的值留 None，由调用方降级处理。
    """

    def __init__(self, arg_values: Dict[str, float]):
        self.vals: Dict[str, Optional[float]] = {}
        for name, v in arg_values.items():
            self.vals["%" + name] = float(v)

    def feed(self, line: str) -> None:
        m = _SSA_RE.match(line)
        if not m:
            return
        dst, opname, rest = m.group(1), m.group(2), m.group(3)

        if opname == "arith.constant":
            cm = re.search(r"arith\.constant\s+(-?\d+)\s*:", line)
            # dense<...> 是张量常量，不是标量，跳过
            if cm and "dense<" not in line:
                self.vals[dst] = float(cm.group(1))
            return

        binops = {
            "arith.addi": lambda a, b: a + b,
            "arith.subi": lambda a, b: a - b,
            "arith.muli": lambda a, b: a * b,
            "arith.divsi": lambda a, b: a // b if b else None,
            "arith.divui": lambda a, b: a // b if b else None,
            "arith.maxsi": max,
            "arith.minsi": min,
        }
        if opname in binops:
            ops = re.findall(r"%[\w#]+", rest)
            if len(ops) >= 2:
                a, b = self.vals.get(ops[0]), self.vals.get(ops[1])
                if a is not None and b is not None:
                    self.vals[dst] = binops[opname](a, b)
            return

        # extsi/trunci/index_cast 只改位宽，值不变
        if opname in ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast"):
            ops = re.findall(r"%[\w#]+", rest)
            if ops:
                self.vals[dst] = self.vals.get(ops[0])

    def resolve(self, token: str) -> Optional[float]:
        if token.startswith("%"):
            return self.vals.get(token)
        m = re.match(r"^c(-?\d+)_i\d+$", token)
        if m:
            return float(m.group(1))
        try:
            return float(token)
        except ValueError:
            return None


def _line_tile_flops(line: str) -> float:
    """一行 IR 贡献的 tile 级浮点运算数。TTIR 与 pim mlir 共用。"""
    if _DOT_RE.search(line):
        # 类型签名在冒号之后，形如 `tensor<MxKxf16[, layout]> * tensor<KxNxf16[, layout]>
        # -> tensor<MxNxf32[, layout]>`。冒号之前只有 SSA 名与可选属性（如
        # flash_fwd 的 `inputPrecision = tf32`），不含 tensor 类型，故直接取
        # 全行的前两个 tensor 类型作两个操作数。
        operands = _TENSOR_RE.findall(line)
        assert len(operands) >= 2, f"tt.dot 找不到两个操作数类型: {line.strip()}"
        lhs = [int(d) for d in operands[0][0].split("x")]
        rhs = [int(d) for d in operands[1][0].split("x")]
        assert len(lhs) == len(rhs) == 2, f"tt.dot 操作数不是二维: {lhs} x {rhs}"
        assert lhs[1] == rhs[0], f"tt.dot 内维不一致: {lhs[1]} vs {rhs[0]}"
        return 2.0 * lhs[0] * lhs[1] * rhs[1]

    for opname, weight in _ELEMENTWISE_FLOPS.items():
        # 用词边界避免 arith.addf 匹配到 arith.addi 之类
        if re.search(r"(?<![\w.])" + re.escape(opname) + r"(?![\w.])", line):
            tm = _TENSOR_RE.search(line)
            numel = _tensor_numel(tm.group(1)) if tm else 1
            return weight * numel
    return 0.0


def _line_dma_bytes(line: str) -> float:
    """一行 pim mlir 贡献的 MRAM↔WRAM 搬运字节数。

    `pim.dma_load` / `pim.dma_store` 的搬运量等于其 `!pim.memdesc` 目标（源）
    buffer 的字节数——pass 保证指针张量与 buffer 的 shape、元素类型一致
    （`PIMOps.td` 的 verifier）。TTIR 上没有这些算子，恒返回 0。
    """
    if not _DMA_RE.search(line):
        return 0.0
    match = _MEMDESC_RE.search(line)
    assert match, f"pim.dma_* 找不到 memdesc 类型: {line.strip()}"
    dims, dtype, _space = match.groups()
    return float(_tensor_numel(dims) * _DTYPE_BYTES[dtype])


def _infer_dtype(ttir: str) -> str:
    """取 kernel 签名里指针的元素类型作为算子 dtype。"""
    ptrs = re.findall(r"!tt\.ptr<([a-z0-9]+)>", ttir)
    floats = [p for p in ptrs if p.startswith(("f", "bf"))]
    if floats:
        # 取出现最多的浮点类型（f32 累加器不出现在签名里，签名多为 IO dtype）
        return max(set(floats), key=floats.count)
    return ptrs[0] if ptrs else "f16"


def analyze_ir(
    text: str,
    kernel_name: str,
    grid: tuple,
    arg_values: Dict[str, float],
    ir_level: str = "ttir",
) -> KernelCost:
    """分析一份 TTIR 或 pim mlir，返回该 kernel 的算子级成本。

    text       -- IR 文本（`CompiledKernel.asm['ttir']` 或 driver 降出的 pim mlir）
    grid       -- launch 时的真实 grid（driver 捕获）
    arg_values -- kernel 标量实参名 -> 值（driver 捕获），用于求循环次数
    ir_level   -- "ttir" 或 "pimir"。pimir 时额外统计 mram_traffic_bytes
                  与 WRAM 用量；flops 的算法两层完全相同。

    flops 与 mram_traffic 都按同一套循环帧栈还原到算子级：帧内的量乘该帧
    trip 计入父帧，最后乘 grid_size。WRAM 分配不参与——pass 已把
    `pim.wram_alloc` 提到最外层循环之前（doc 9.4），每个 buffer 一个 program
    只分配一次。
    """
    assert ir_level in ("ttir", "pimir"), f"未知 ir_level: {ir_level}"
    dtype = _infer_dtype(text)
    folder = _ConstFolder(arg_values)

    notes: List[str] = []
    # 循环帧栈。栈底是 kernel 体本身（trip=1），每个 scf.for 压一帧。
    # 帧出栈时把 body 的量 * trip 计入父帧——这样顺序排列的多个顶层循环
    # 各自带自己的 trip（flash_fwd 就有两个顺序顶层循环，共用一个 trip
    # 会把后一个的次数错用到前一个身上），嵌套循环则自然逐层相乘。
    stack: List[Dict[str, Any]] = [
        {"flops": 0.0, "dma_bytes": 0.0, "trip": 1.0, "bounds": None}
    ]
    top_level_trips: List[Optional[float]] = []
    wram_buffers: List[WramBuffer] = []
    dma_ops = 0
    dma_ops_with_layout = 0

    for raw in text.splitlines():
        line = raw.split(" loc(")[0]  # 去掉 loc 注解，避免误匹配
        if len(stack) == 1:
            folder.feed(line)

        fm = _SCF_FOR_RE.search(line)
        if fm:
            lo = folder.resolve(fm.group(1))
            hi = folder.resolve(fm.group(2))
            step = folder.resolve(fm.group(3))
            bounds = f"[{fm.group(1)},{fm.group(2)}) step {fm.group(3)}"
            if None not in (lo, hi, step) and step:
                trip: Optional[float] = max(0.0, (hi - lo) / step)
            else:
                trip = None
                notes.append(f"循环次数未折叠: {bounds}")
            if len(stack) == 1:
                top_level_trips.append(trip)
            stack.append(
                {"flops": 0.0, "dma_bytes": 0.0, "trip": trip, "bounds": bounds}
            )
            continue

        if len(stack) > 1 and re.match(r"^\s*\}", line):
            frame = stack.pop()
            trip = frame["trip"]
            if (frame["flops"] or frame["dma_bytes"]) and trip is None:
                # 循环体有计算但次数不明：记 1 次并标注，绝不静默当成 0
                trip = 1.0
                notes.append(
                    f"循环次数缺失，按 1 次计（该 kernel 成本被低估）: {frame['bounds']}"
                )
            stack[-1]["flops"] += frame["flops"] * (trip or 0.0)
            stack[-1]["dma_bytes"] += frame["dma_bytes"] * (trip or 0.0)
            continue

        f = _line_tile_flops(line)
        if f:
            stack[-1]["flops"] += f

        if ir_level == "pimir":
            dma_bytes = _line_dma_bytes(line)
            if dma_bytes:
                stack[-1]["dma_bytes"] += dma_bytes
                dma_ops += 1
                # 指针分析证明了 stride 才会写这三个属性；缺失不代表连续，
                # 只代表「当前没有证明」（doc 9.5）
                if "elem_stride" in line:
                    dma_ops_with_layout += 1
            alloc = _WRAM_ALLOC_RE.search(line)
            if alloc:
                dims, alloc_dtype = alloc.group(1), alloc.group(2)
                wram_buffers.append(WramBuffer(
                    shape=dims,
                    dtype=alloc_dtype,
                    bytes=_tensor_numel(dims) * _DTYPE_BYTES[alloc_dtype],
                ))

    assert len(stack) == 1, f"IR 括号不配平，剩余 {len(stack) - 1} 帧未闭合"

    grid_size = 1
    for g in grid:
        grid_size *= int(g)

    per_program = stack[0]["flops"]
    cost = KernelCost(
        kernel_name=kernel_name,
        flops=grid_size * per_program,
        dtype=dtype,
        element_bytes=_DTYPE_BYTES.get(dtype, 2),
        grid=tuple(int(g) for g in grid),
        tile_flops_per_program=per_program,
        loop_trip_counts=top_level_trips,
        notes=notes,
    )
    if ir_level == "ttir":
        return cost

    cost.mram_traffic_bytes = grid_size * stack[0]["dma_bytes"]
    cost.wram_buffers = wram_buffers
    cost.dma_ops = dma_ops
    cost.dma_ops_with_layout = dma_ops_with_layout
    cost.wram_bytes_used = _module_int_attr(text, "pim.wram-bytes-used")
    cost.wram_bytes_budget = _module_int_attr(text, "pim.wram-bytes")
    cost.mram_bytes_budget = _module_int_attr(text, "pim.mram-bytes")
    cost.dma_align = _module_int_attr(text, "pim.dma-align")
    cost.tile_m = _module_int_attr(text, "pim.tile-m")
    cost.tile_n = _module_int_attr(text, "pim.tile-n")
    cost.tile_k = _module_int_attr(text, "pim.tile-k")
    cost.tile_wram_bytes = _module_int_attr(text, "pim.tile-wram-bytes")

    assert dma_ops, f"{kernel_name} 的 pim mlir 里没有 pim.dma_*，pass 可能没生效"
    # 超预算时 pim-explicit-dma 只发 warning、不重切 tile（doc 15.3），
    # 于是 IR 仍可能不可执行。落 note 让它在 sidecar 里可见，不静默放过。
    if (
        cost.wram_bytes_used is not None
        and cost.wram_bytes_budget is not None
        and cost.wram_bytes_used > cost.wram_bytes_budget
    ):
        notes.append(
            f"WRAM 超预算: 用了 {cost.wram_bytes_used} B / 预算 "
            f"{cost.wram_bytes_budget} B（FlagTree 当前只 warning、不重切 tile）"
        )
    return cost


def _module_int_attr(text: str, name: str) -> Optional[int]:
    """读 module 上的整数属性，如 `"pim.wram-bytes-used" = 28672 : i32`。"""
    match = re.search(re.escape(f'"{name}"') + r"\s*=\s*(\d+)\s*:", text)
    return int(match.group(1)) if match else None
