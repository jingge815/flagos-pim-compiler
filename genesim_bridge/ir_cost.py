"""从 TTIR 或 PIM IR 提取算子浮点运算量、数据类型和搬运量。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 定义逐元素算子的每元素浮点运算量。
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

# 匹配 IR 中的普通张量形状和元素类型。
_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x([a-z0-9]+)[,>]")
# 匹配 PIM 内存描述符的形状、类型和存储位置。
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
    # 以下字段仅由 PIM IR 填充。
    mram_traffic_bytes: Optional[float] = None   # MRAM↔WRAM 显式搬运字节，只进 sidecar
    wram_bytes_used: Optional[int] = None        # pass 统计的 WRAM 分配总量
    wram_bytes_budget: Optional[int] = None      # module 上声明的 WRAM 预算
    wram_buffers: List[WramBuffer] = field(default_factory=list)
    dma_ops: int = 0
    dma_ops_with_layout: int = 0                 # 指针分析证明了 stride 的那些
    # PIM 硬件预算和自动选择的分块大小。
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
    """折叠循环边界中的整数表达式。"""

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
            # 记录标量常量。
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

        # 位宽转换不改变标量值。
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
        # 前两个张量类型表示矩阵乘的输入。
        operands = _TENSOR_RE.findall(line)
        assert len(operands) >= 2, f"tt.dot 找不到两个操作数类型: {line.strip()}"
        lhs = [int(d) for d in operands[0][0].split("x")]
        rhs = [int(d) for d in operands[1][0].split("x")]
        assert len(lhs) == len(rhs) == 2, f"tt.dot 操作数不是二维: {lhs} x {rhs}"
        assert lhs[1] == rhs[0], f"tt.dot 内维不一致: {lhs[1]} vs {rhs[0]}"
        return 2.0 * lhs[0] * lhs[1] * rhs[1]

    for opname, weight in _ELEMENTWISE_FLOPS.items():
        # 使用词边界区分相近的算子名称。
        if re.search(r"(?<![\w.])" + re.escape(opname) + r"(?![\w.])", line):
            tm = _TENSOR_RE.search(line)
            numel = _tensor_numel(tm.group(1)) if tm else 1
            return weight * numel
    return 0.0


def _line_dma_bytes(line: str) -> float:
    """返回一行 PIM DMA 指令的搬运字节数。"""
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
        # 使用签名中最常见的浮点类型。
        return max(set(floats), key=floats.count)
    return ptrs[0] if ptrs else "f16"


def analyze_ir(
    text: str,
    kernel_name: str,
    grid: tuple,
    arg_values: Dict[str, float],
    ir_level: str = "ttir",
) -> KernelCost:
    """分析 IR 文本并返回算子级成本。"""
    assert ir_level in ("ttir", "pimir"), f"未知 ir_level: {ir_level}"
    dtype = _infer_dtype(text)
    folder = _ConstFolder(arg_values)

    notes: List[str] = []
    # 循环帧栈用于累计嵌套循环的成本。
    stack: List[Dict[str, Any]] = [
        {"flops": 0.0, "dma_bytes": 0.0, "trip": 1.0, "bounds": None}
    ]
    top_level_trips: List[Optional[float]] = []
    wram_buffers: List[WramBuffer] = []
    dma_ops = 0
    dma_ops_with_layout = 0

    for raw in text.splitlines():
        line = raw.split(" loc(")[0]  # 移除 loc 注解后匹配指令。
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
                # 未知循环次数按一次计算并记录说明。
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
                # 指针分析确认布局时才记录步长属性。
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
    # 在结果中记录超过 WRAM 预算的分块。
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
