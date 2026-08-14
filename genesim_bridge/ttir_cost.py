"""TTIR 成本分析器：从 FlagTree 编出的 TTIR 抽单算子成本。

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
的重复搬运，由调用方按全局 shape 算。本模块只负责 flops 与 dtype。
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

# tensor<32x64xf16> / tensor<128xf32> -> ("32x64", "f16")
_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x([a-z0-9]+)>")
# tt.dot 的两个操作数类型。只锚定 `tt.dot` 与其后的 `tensor<> * tensor<>`
# 类型签名，中间的操作数列表与可选属性（如 `inputPrecision = tf32`，
# flash_fwd 的 dot 就带这个）一概跳过——早期版本要求「恰好三个操作数后
# 紧跟冒号」，遇到带属性的 dot 直接匹配失败，把 flash kernel 的矩阵乘
# 静默算成 0 flops。
_DOT_RE = re.compile(
    r"tt\.dot[^:]*:\s*"
    r"tensor<(\d+)x(\d+)x[a-z0-9]+>\s*\*\s*tensor<(\d+)x(\d+)x[a-z0-9]+>"
)
_SCF_FOR_RE = re.compile(r"scf\.for\s+%\S+\s*=\s*(\S+)\s+to\s+(\S+)\s+step\s+(\S+)")
_SSA_RE = re.compile(r"^\s*(%[\w#]+)\s*=\s*(\S+)(.*)$")


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
    """一行 TTIR 贡献的 tile 级浮点运算数。"""
    dm = _DOT_RE.search(line)
    if dm:
        m, k, k2, n = (int(dm.group(i)) for i in (1, 2, 3, 4))
        assert k == k2, f"tt.dot 内维不一致: {k} vs {k2}"
        return 2.0 * m * k * n

    for opname, weight in _ELEMENTWISE_FLOPS.items():
        # 用词边界避免 arith.addf 匹配到 arith.addi 之类
        if re.search(r"(?<![\w.])" + re.escape(opname) + r"(?![\w.])", line):
            tm = _TENSOR_RE.search(line)
            numel = _tensor_numel(tm.group(1)) if tm else 1
            return weight * numel
    return 0.0


def _infer_dtype(ttir: str) -> str:
    """取 kernel 签名里指针的元素类型作为算子 dtype。"""
    ptrs = re.findall(r"!tt\.ptr<([a-z0-9]+)>", ttir)
    floats = [p for p in ptrs if p.startswith(("f", "bf"))]
    if floats:
        # 取出现最多的浮点类型（f32 累加器不出现在签名里，签名多为 IO dtype）
        return max(set(floats), key=floats.count)
    return ptrs[0] if ptrs else "f16"


def analyze_ttir(
    ttir: str,
    kernel_name: str,
    grid: tuple,
    arg_values: Dict[str, float],
) -> KernelCost:
    """分析一份 TTIR，返回该 kernel 的算子级成本。

    ttir       -- CompiledKernel.asm['ttir'] 文本
    grid       -- launch 时的真实 grid（driver 捕获）
    arg_values -- kernel 标量实参名 -> 值（driver 捕获），用于求循环次数
    """
    dtype = _infer_dtype(ttir)
    folder = _ConstFolder(arg_values)

    notes: List[str] = []
    # 循环帧栈。栈底是 kernel 体本身（trip=1），每个 scf.for 压一帧。
    # 帧出栈时把 body_flops * trip 计入父帧——这样顺序排列的多个顶层循环
    # 各自带自己的 trip（flash_fwd 就有两个顺序顶层循环，共用一个 trip
    # 会把后一个的次数错用到前一个身上），嵌套循环则自然逐层相乘。
    stack: List[Dict[str, Any]] = [{"flops": 0.0, "trip": 1.0, "bounds": None}]
    top_level_trips: List[Optional[float]] = []

    for raw in ttir.splitlines():
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
            stack.append({"flops": 0.0, "trip": trip, "bounds": bounds})
            continue

        if len(stack) > 1 and re.match(r"^\s*\}", line):
            frame = stack.pop()
            trip = frame["trip"]
            if frame["flops"] and trip is None:
                # 循环体有计算但次数不明：记 1 次并标注，绝不静默当成 0
                trip = 1.0
                notes.append(
                    f"循环次数缺失，按 1 次计（该 kernel flops 被低估）: {frame['bounds']}"
                )
            stack[-1]["flops"] += frame["flops"] * (trip or 0.0)
            continue

        f = _line_tile_flops(line)
        if f:
            stack[-1]["flops"] += f

    assert len(stack) == 1, f"TTIR 括号不配平，剩余 {len(stack) - 1} 帧未闭合"

    grid_size = 1
    for g in grid:
        grid_size *= int(g)

    per_program = stack[0]["flops"]
    return KernelCost(
        kernel_name=kernel_name,
        flops=grid_size * per_program,
        dtype=dtype,
        element_bytes=_DTYPE_BYTES.get(dtype, 2),
        grid=tuple(int(g) for g in grid),
        tile_flops_per_program=per_program,
        loop_trip_counts=top_level_trips,
        notes=notes,
    )
