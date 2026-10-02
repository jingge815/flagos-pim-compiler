"""从 TTIR 或 PIM IR 提取算子浮点运算量、数据类型和搬运量。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from contracts.ir_payloads import (
    LayoutFeedback,
    PlacementBack,
    layout_feedback_of_module,
    module_int_attrs,
    placement_of_module,
)

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
# A 路的访存还留在 `tt.load` / `tt.store` 上，说明 `-pim-explicit-dma` 没跑。
_TT_LOAD_RE = re.compile(r"(?<![\w.])tt\.(?:load|store)(?![\w.])")
# 算子级（B 路）mnemonic -> 计费用的规范名。A 路只认 tt.dot 与 arith/math，
# 不认识这些，所以少了这张表，B 路每个算子的成本都恒为 0 —— 仿真照常跑完、
# 给出一个看起来合理的总耗时，只是那个数字是错的，而且不会报错。
#
# 同一个算子有两种拼法的归一到这里：`pim.quantize`（展开前的 ODS 拼法）与
# `pim.dynamic_quant`（方案 8.2-12 要改成的名字）是同一个算子，各记一份就会
# 按拼法算出两个成本。
_OPLEVEL_CANONICAL = {
    "pim.matmul": "pim.matmul",
    "pim.softmax": "pim.softmax",
    "pim.lut": "pim.lut",
    "pim.eltwise": "pim.eltwise",
    "pim.reduce_axis": "pim.reduce_axis",
    "pim.global_pool": "pim.global_pool",
    "pim.normalize": "pim.normalize",
    "pim.rope": "pim.rope",
    "pim.mask": "pim.mask",
    "pim.quantize": "pim.dynamic_quant",
    "pim.dynamic_quant": "pim.dynamic_quant",
    # DQ 相 3 现在是独立 `pim.kantor`（卡值在 cardValue 上）。不认它，那一相
    # 的成本会静默变成 0——展开后每相各计一遍，少一相就是少 4096 个元素。
    "pim.kantor": "pim.kantor",
    "pim.fpsu_scale": "pim.fpsu_scale",
}
# 每个**元素**的浮点运算量。整算子（未展开）按相数计：展开成相位链之后每一相
# 各自计一遍，两边同量级，sidecar 不会因为"还没展开"就少一大截。
_OPLEVEL_FLOPS_PER_ELEMENT = {
    "pim.lut": 1.0,             # 每个元素一次查表
    "pim.eltwise": 1.0,         # 每个元素一次运算
    "pim.reduce_axis": 1.0,     # 每个被归约的元素一次比较/累加
    "pim.global_pool": 1.0,     # 同上，按组内的元素计
    "pim.dynamic_quant": 4.0,   # 四相：统计、倒数、定标、定点化
    "pim.softmax": 5.0,         # 五相：相 0/2 归约 + 相 1/3 查表 + 相 4 逐元素
    "pim.normalize": 5.0,       # 平方、求均、加 ε、rsqrt、乘 γ
    "pim.rope": 3.0,            # 每元素 2 乘 1 加，三相各落一次 fp16
    "pim.mask": 1.0,            # 加性掩码，每元素一次加法
    "pim.kantor": 1.0,          # 定点化：每元素一次乘加
    "pim.fpsu_scale": 1.0,      # 定点重定标：每元素一次乘加
}
# 视图与查表类：**不计 flops，计搬运**。它们不做算术，但确实在动字节，
# 记成 0 会让仿真以为这些节点是免费的。
_MOVEMENT_MNEMONICS = (
    "pim.gather", "pim.transpose", "pim.reshape", "pim.split_heads",
    "pim.concat", "pim.convert",
)
# KV 缓存写入单独一条：它动的字节不是**结果**张量（那是整个缓存），而是写进去
# 的那一段。
_KV_WRITE_MNEMONICS = ("pim.kv_cache",)
# 方言里**全部**的算子名，用于「看到一个不认识的 `pim.*` 就记一条 note」。
# `_OPLEVEL_RE` 不命中时 `_line_operator_flops` 直接返回 0，而 0 不会报错——
# 方言侧改名或加算子，成本会静默回到 0。tile 级也列进来，否则 A 路的
# `pim.dma_load` 之类会被当成「未识别」刷屏。
# 确认零成本的名字：tile 级脚手架与纯地址/缓冲操作，本来就不该计 flops 或搬运。
# 出现在这里的名字不进 notes。
_ZERO_COST_OPS = frozenset({
    "pim.tasklet_id", "pim.dpu_id", "pim.wram_alloc", "pim.dma_load",
    "pim.dma_store", "pim.wram_load", "pim.wram_store", "pim.barrier",
    "pim.buffer_alloc", "pim.buffer_copy", "pim.decompress_weight", "pim.param",
    # 类型转换器插入的布局修正 op：NoMemoryEffect，降级成零代码，只改 tasklet
    # 归属标记。不进这张表会被当成「未识别」刷 note。
    "pim.convert_layout",
})
_KNOWN_PIM_OPS = frozenset(
    set(_OPLEVEL_CANONICAL)
    | set(_MOVEMENT_MNEMONICS)
    | set(_KV_WRITE_MNEMONICS)
    | _ZERO_COST_OPS
)
# 算子名只出现在**定义位置**：`%3 = pim.lut %2 {...}`。不能在整行里捞
# `pim.` 前缀——`#pim.phase_spec` / `!pim.memdesc` / `#pim.wram` 是属性与类型
# 的命名空间，把它们当成算子名会刷屏。
_PIM_OP_NAME_RE = re.compile(r"=\s*pim\.([a-z_][a-z_0-9]*)")
# 已经展开的相位 op 会带 `phases = [#pim.phase_spec<…>]`；整算子级的 opaque
# 算子上没有这个属性。计费要分开：整算子按相数乘，展开后的每一行只是一相。
_PHASES_ATTR_RE = re.compile(r"phases\s*=\s*\[")
# 展开后每一相的搬运字节数写在 `bytes` 字段里，计算类算子（softmax/lut/...）
# 没有自己的访存 mnemonic，搬运量全靠这个字段，不解析就整列是 0。
_PHASE_SPEC_BYTES_RE = re.compile(r"#pim\.phase_spec<[^>]*bytes\s*=\s*(\d+)")


def _mnemonic_re(names) -> re.Pattern:
    """算子级 mnemonic 的词匹配。

    `-` 也要挡：`"pim.transpose-type"` 是相位上的属性名，不是 transpose 算子，
    放它进来会给每个带该属性的行凭空记一笔搬运（实测 softmax 链上多出 2050 B）。
    """
    body = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(r"(?<![\w.])(?:%s)(?![\w.\-])" % body)


_OPLEVEL_RE = _mnemonic_re(_OPLEVEL_CANONICAL)
_MOVEMENT_RE = _mnemonic_re(_MOVEMENT_MNEMONICS)
_KV_WRITE_RE = _mnemonic_re(_KV_WRITE_MNEMONICS)
# `stationarity = #pim.stationarity<kv>`。驻留侧只装载一次，流式侧每趟都重读——
# 这是 PIM 矩阵乘内部最大的一笔成本差（prefill 与 decode 的分野就在这里）。
# 少了它，两种形态在仿真里搬运量相同，而实际差一个数量级。
_STATIONARITY_RE = re.compile(r"#pim\.stationarity<(\w+)>")
_MATMUL_RE = re.compile(r"(?<![\w.])pim\.matmul(?![\w.])")
# 组反量化累加：`#pim.datapath<… groupDequantAccum = true, groupSize = 128 …>`。
# 两个域分开抓——打印顺序由 MLIR 按字母排，不能假设谁在前。
_DATPATH_RE = re.compile(r"#pim\.datapath<([^>]*)>")
_GROUP_ACCUM_RE = re.compile(r"groupDequantAccum\s*=\s*(true|false)")
_GROUP_SIZE_RE = re.compile(r"groupSize\s*=\s*(\d+)")
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
    # 组反量化累加的反量化次数（`K / groupSize` 之和）。不改 flops，只作对照。
    group_dequant_steps: int = 0
    # PIM 硬件预算和自动选择的分块大小。
    mram_bytes_budget: Optional[int] = None
    dma_align: Optional[int] = None
    tile_m: Optional[int] = None
    tile_n: Optional[int] = None
    tile_k: Optional[int] = None
    tile_wram_bytes: Optional[int] = None
    # 图编译器下发的跨 DPU 切分决策（统一 IR 的 Placement 维）。A 路把它写成
    # 模块属性（张量编码由 FlagTree 生成、`dpusPerDevice` 恒全 1，写不进去），
    # B 路写进 `#pim.tasklet_tiled`。读回来是为了让成本模型按**本地**规模算
    # 代价：切了 N 台 DPU，单台只算 1/N。
    shard_dim: Optional[int] = None
    shard_dpus: Optional[int] = None
    # 回传载体原件。上面两个字段是它的摊平副本（sidecar 既有字段名不动）。
    placement: PlacementBack = field(default_factory=PlacementBack)
    # 算子编译器回传的单台 DPU MRAM 占用（切分之后）。
    mram_bytes_per_dpu: Optional[int] = None
    # 算子编译器的布局回传原件。上面几个字段是它的摊平副本（sidecar 的既有
    # 字段名不动），判「超不超 WRAM 预算」这类结论走它自己的属性。
    layout_feedback: LayoutFeedback = field(default_factory=LayoutFeedback)


def _tensor_numel(dims: str) -> int:
    n = 1
    for d in dims.split("x"):
        n *= int(d)
    return n


# 函数签名里的一对「参数名 : 类型」。
_SIG_PARAM_RE = re.compile(r"(%arg\d+)\s*:\s*([^,)]+)")


class _ConstFolder:
    """折叠循环边界中的整数表达式。"""

    def __init__(self, arg_values: Dict[str, float]):
        self.vals: Dict[str, Optional[float]] = {}
        for name, v in arg_values.items():
            self.vals["%" + name] = float(v)
        # 按参数顺序排列的标量实参，供 `_bind_signature` 给位置参数取值。
        self.ordered_args = [float(v) for v in arg_values.values()]

    def _bind_signature(self, line: str) -> None:
        """把签名里的位置参数 `%argN` 与 `arg_values` 里的标量实参对上。

        真实捕获的 TTIR 里 Triton 不保留形参名，参数打印成 `%arg0`/`%arg3`，
        而 `arg_values` 是按形参名采的（`K`、`stride_wn`……）。只按名字匹配
        （`%K`）永远对不上，由它算出的循环次数就折不出来、按 1 次计，成本被
        低估一个数量级。这里数一遍签名里哪些参数是指针、哪些是标量，标量的
        按顺序取 `arg_values` 的前若干个——指针实参是张量，采集时已被过滤掉，
        所以剩下这些就是署名里那些标量参数，顺序与签名一致。
        """
        scalar = 0
        for name, ty in _SIG_PARAM_RE.findall(line):
            if "!tt.ptr" in ty:
                continue
            if scalar < len(self.ordered_args):
                self.vals.setdefault(name, self.ordered_args[scalar])
            scalar += 1

    def feed(self, line: str) -> None:
        if "tt.func" in line:
            self._bind_signature(line)
            return
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


def _line_operator_flops(line: str) -> float:
    """一行算子级 IR 贡献的浮点运算数。

    归约按**输入**的元素数计——要扫完才有结果，输出只有一个数；查表与逐元素
    按输出计。两者在除归约以外都相等，所以取输入那一侧对两者都对。

    整算子（未展开）按相数乘：`pim.softmax` 五相、`pim.quantize{dynamic}` 四相，
    一项都没少。展开之后每一行只是一相，乘 1，否则同一份工作会被乘两遍——
    展开前按 4 相、展开后每相再按 4 相，成本凭空涨 4 倍。
    """
    match = _OPLEVEL_RE.search(line)
    if not match:
        return 0.0
    mnemonic = _OPLEVEL_CANONICAL[match.group(0)]

    tensors = _TENSOR_RE.findall(line)
    if not tensors:
        return 0.0

    if mnemonic == "pim.matmul":
        assert _MATMUL_RE.search(line)
        # 投影与注意力同为二维 `[M, K] x [K, N]`：`pim-expand-phases` 不展开
        # 矩阵乘，图编译器发过来的就是二维。
        shapes = [[int(d) for d in dims.split("x")] for dims, _ in tensors]
        assert len(shapes) >= 2 and all(len(s) == 2 for s in shapes[:2]), (
            f"pim.matmul 的操作数不是二维: {line.strip()}")
        m, k = shapes[0]
        n = shapes[1][1]
        return 2.0 * m * k * n

    phases = 1.0 if _PHASES_ATTR_RE.search(line) else _OPLEVEL_FLOPS_PER_ELEMENT[mnemonic]
    return phases * _tensor_numel(tensors[0][0])


def _line_weight_residency_bytes(line: str) -> float:
    """一行 `pim.matmul` 里**权值侧**的搬运字节数。

    三种驻留形态的搬运量不同：

        weight / kv  驻留侧装载一次 -> 计一遍权值张量的字节
        activation   两侧都流式     -> 权值侧没有"一次装载"，计 0
                                       （它的搬运已经算在激活那一侧）

    `activation` 记 0 不是"忽略"：prefill 的两个操作数都是激活，本来就不存在
    权值驻留这件事，硬记一笔会让 prefill 的搬运量凭空多出一个张量。
    """
    match = _STATIONARITY_RE.search(line)
    if not match or not _MATMUL_RE.search(line):
        return 0.0
    if match.group(1) == "activation":
        return 0.0

    tensors = _TENSOR_RE.findall(line)
    if len(tensors) < 2:
        return 0.0
    # 第二个张量是权值侧操作数（`pim.matmul %a, %b` 的 b）。
    dims, dtype = tensors[1]
    return float(_tensor_numel(dims) * _DTYPE_BYTES.get(dtype, 1))


def _line_group_dequant_steps(line: str) -> int:
    """一行 `pim.matmul` 里按组反量化做了多少次。

    开组反量化累加时，整数累加器每 `groupSize` 个 K 就得停下来把这一组反量化
    再折进浮点总数，所以次数是 `K / groupSize`。**不改 flops**——乘加次数一个
    没变，变的只是累加顺序。但它决定反量化单元被调用几趟，是算子编译器这侧
    唯一能拿到的对照量，所以单独记一笔。
    """
    match = _DATPATH_RE.search(line)
    if not match or not _MATMUL_RE.search(line):
        return 0
    body = match.group(1)
    flag = _GROUP_ACCUM_RE.search(body)
    size = _GROUP_SIZE_RE.search(body)
    if not flag or flag.group(1) != "true" or not size:
        return 0

    tensors = _TENSOR_RE.findall(line)
    if not tensors:
        return 0
    # `[M, K] x [K, N]`：K 是第一个操作数的第二维，也是累加的深度。
    dims = [int(d) for d in tensors[0][0].split("x")]
    k = dims[-1]
    group = int(size.group(1))
    return k // group if group else 0


def _line_movement_bytes(line: str) -> float:
    """一行访存类算子搬运的字节数。

    `pim.gather` 读走 `indices.numel()` 行、每行是表的行宽；四个视图算子整体搬
    一遍。两者的搬运量都等于**结果**的元素数乘元素宽，所以取最后一个张量
    （`-> tensor<...>`）。不计 flops：这些算子不做算术。

    `pim.kv_cache` 是例外：它的结果张量是那整个跨步存活的缓存，按它计会凭空放大
    几个数量级。写入是"往缓存的第 pos 行塞一段新值"，所以按**第一个**张量
    （新值）计。散写与区间写在这条口径下天然不同：后者一次写多行。

    三种 purpose 计费**完全相同**，这是刻意的，不是漏算：`absorbed` /
    `onthefly` 说的是图层面「这一层不单独发射节点」，而**内核仍然要把值排到正确
    顺序上**——输出是一个扁平缓冲（见 FlagTree `LowerPIMToEmitC.cpp` 里那段
    逐元素置换）。按 absorbed 少计就是把真实搬运漏掉。

    图层面的那个区分有它自己的消费方，走的是 `node.meta`：
    `gml_bridge/from_fx.py:233,925` 的 `ABSORBED_META_KEY` 决定发不发节点。
    两个载体（图的 meta 与 IR 的属性）回答的是不同层次的问题。
    """
    tensors = _TENSOR_RE.findall(line)
    if not tensors:
        return 0.0
    if _MOVEMENT_RE.search(line):
        dims, dtype = tensors[-1]
    elif _KV_WRITE_RE.search(line):
        dims, dtype = tensors[0]
    else:
        return 0.0
    return float(_tensor_numel(dims) * _DTYPE_BYTES.get(dtype, 2))


def _line_phase_movement_bytes(line: str) -> float:
    """展开后的相位链里，每一相的搬运字节数写在 `#pim.phase_spec` 的 `bytes` 上。

    计算类算子（softmax/lut/pool/eltwise/normalize/rope/mask/dq）没有自己的
    访存 mnemonic，`_line_movement_bytes` 对它们恒返回 0；真实搬运量全在
    这个字段里，一行里可能有多个 `phase_spec`，逐个累加。
    """
    return float(sum(int(b) for b in _PHASE_SPEC_BYTES_RE.findall(line)))


def _line_dma_bytes(line: str) -> float:
    """返回一行 PIM DMA 指令的搬运字节数。

    带 `elem_stride` 时按步幅计：那是统一 IR 的排布字段落到 pimir 上的那一位，
    步幅大于 1 说明行间有填充，真实搬运大于 `元素数 × 元素宽`。不带时按紧密计，
    与改动前逐字节相同。
    """
    if not _DMA_RE.search(line):
        return 0.0
    match = _MEMDESC_RE.search(line)
    assert match, f"pim.dma_* 找不到 memdesc 类型: {line.strip()}"
    dims, dtype, _space = match.groups()
    nbytes = _tensor_numel(dims) * _DTYPE_BYTES[dtype]
    stride = re.search(r"elem_stride\s*=\s*(\d+)", line)
    if stride:
        # 步幅是「沿 contiguous_dim 每走一步跨几个元素」，紧密时为 1。
        nbytes *= int(stride.group(1))
    return float(nbytes)


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
    # `oplevel` 是 B 路自己的口径：IR 由 `lower_oplevel_to_pimir` 展开而来，
    # 没有访存可显式化，所以 A 路那条「必须有 dma」的判据不适用；计费方式与
    # `pimir` 相同。两者分开，是因为**调用方知道自己在哪条路上**，而 IR 本身
    # 没有可靠的标记——曾经用「有没有 `tt.dot`」当代理，于是无 dot 的
    # 逐元素内核带着坏掉的 explicit-dma 静默通过。
    assert ir_level in ("ttir", "pimir", "oplevel"), \
        f"未知 ir_level: {ir_level}"
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
    group_dequant_steps = 0

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

        f = _line_tile_flops(line) + _line_operator_flops(line)
        if f:
            stack[-1]["flops"] += f

        if ir_level in ("pimir", "oplevel"):
            moved = (_line_movement_bytes(line)
                     + _line_weight_residency_bytes(line)
                     + _line_phase_movement_bytes(line))
            if moved:
                stack[-1]["dma_bytes"] += moved
            dma_bytes = _line_dma_bytes(line)
            if dma_bytes:
                stack[-1]["dma_bytes"] += dma_bytes
                dma_ops += 1
                # 指针分析确认布局时才记录步长属性。
                if "elem_stride" in line:
                    dma_ops_with_layout += 1
            group_dequant_steps += _line_group_dequant_steps(line)
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

    seen = {f"pim.{name}" for name in _PIM_OP_NAME_RE.findall(text)}
    unknown = sorted(name for name in seen if name not in _KNOWN_PIM_OPS)
    if unknown:
        notes.append(
            "未识别的 pim 算子（成本按 0 计，表里没有这一项）: "
            + ", ".join(unknown))
    billed = (set(_OPLEVEL_CANONICAL) | set(_MOVEMENT_MNEMONICS)
              | set(_KV_WRITE_MNEMONICS))
    unbilled = sorted(seen - billed - _ZERO_COST_OPS - set(unknown))
    if unbilled:
        notes.append(
            "没有计费规则的 pim 算子（成本按 0 计）: " + ", ".join(unbilled))

    cost.mram_traffic_bytes = grid_size * stack[0]["dma_bytes"]
    cost.wram_buffers = wram_buffers
    cost.dma_ops = dma_ops
    cost.dma_ops_with_layout = dma_ops_with_layout
    cost.group_dequant_steps = group_dequant_steps
    # 布局回传走统一 IR 的载体（`LayoutFeedback`），不再各自写一遍正则：
    # tile 形状与 WRAM 用量是算子编译器的决策，成本模型是它的消费方。
    feedback = layout_feedback_of_module(text)
    cost.layout_feedback = feedback
    cost.wram_bytes_used = feedback.wram_bytes_used
    cost.wram_bytes_budget = feedback.wram_bytes_budget
    cost.mram_bytes_budget = feedback.mram_bytes
    tile = feedback.tile_shape
    cost.tile_m = None if tile is None else tile[0]
    cost.tile_n = None if tile is None else tile[1]
    # `tile-k` / `tile-wram-bytes` / `dma-align` 不在回传载体里（载体只收
    # 四维相关的那几项），仍按模块属性直取 —— 用同一份解析结果，不另起正则。
    attrs = module_int_attrs(text)
    cost.dma_align = attrs.get("pim.dma-align")
    cost.tile_k = attrs.get("pim.tile-k")
    cost.tile_wram_bytes = attrs.get("pim.tile-wram-bytes")
    # 跨 DPU 切分决策：图编译器下发 `#pim.placement`，这里读回来参与代价计算。
    placement = placement_of_module(text)
    cost.placement = placement
    cost.shard_dim = placement.dim
    cost.shard_dpus = placement.num_dpus if placement.kind else None
    if placement.is_sharded:
        # 这里**不除** `shard_dpus`：搬运量是从这份 pimir 自己的 `pim.dma_*`
        # 累出来的，而这份 pimir 由图编译器按执行计划的 `local_shape` 生成 ——
        # 文本里的字节数本来就是单台口径（tp2 的权重已经是 32x64，不是 64x64）。
        # 再除一次就把单台流量算成了实际的 1/N：实测同一份本地形状，带 placement
        # 得 3584B、不带得 7168B，而真实单台搬运量是后者。
        #
        # 切分决策仍然是被消费的，只是落点不是这个除法：`shard_dim` /
        # `shard_dpus` 读回来进 sidecar 并在 GeneSim 侧参与容量核对，
        # `placed_mram_bytes` 覆盖单台占用，意图与效果不符时出 note。
        cost.notes.append(
            f"跨 DPU 切分：第 {cost.shard_dim} 维分给 {cost.shard_dpus} 台 DPU；"
            f"搬运量取自本地形状，不再按 1/{cost.shard_dpus} 分摊")
    # 回传的单台 MRAM 占用优先于我们自己按「全局/N」估：那个 pass 定了分块，
    # 只有它知道切完实际摆了多少字节。取不到才退回静态估算（与 phase_value()
    # 的「取不到才退回常量表」同口径）。
    if placement.placed_mram_bytes is not None:
        cost.mram_bytes_per_dpu = placement.placed_mram_bytes
    # dtype 维的回传：元素宽度优先用算子编译器实际用的那个，而不是按类型名猜。
    # `_DTYPE_BYTES.get(dtype, 2)` 的默认 2 是个猜测 —— 认不出的类型（int8 的
    # w4a8 投影是 1 字节）会让由它推出的每个字节数一起偏掉。取不到回传才退回
    # 猜测，与 `phase_value()` 的「取不到才退回常量表」同口径。
    if placement.placed_elem_bytes:
        if placement.placed_elem_bytes != cost.element_bytes:
            cost.notes.append(
                f"元素宽度按回传取 {placement.placed_elem_bytes} 字节"
                f"（按类型名 {cost.dtype!r} 猜的是 {cost.element_bytes}）")
        cost.element_bytes = placement.placed_elem_bytes
    # 同一份文本里的两个 MRAM 数自相矛盾：`pim.mram-bytes` 是下发的每台预算的
    # 回显，`pim.placed-mram-bytes` 是那个 pass 定完分块后算出的单台占用。
    # 占用超过预算说明两者不是同一套配置下算出来的 —— 静默下去会让容量结论
    # 基于一个自相矛盾的输入。能同时出现这两个数的地方只有手写或旧产物。
    budget, footprint = feedback.mram_bytes, placement.placed_mram_bytes
    if budget is not None and footprint is not None and footprint > budget:
        cost.notes.append(
            f"MRAM 超预算：算子编译器回传的单台占用 {footprint} B 超过它自己"
            f"声明的每台预算 {budget} B（同一份 IR 内两个数矛盾）")
    # 意图与效果不符说明两侧对切分的理解漂了，静默下去会让成本按错的规模算。
    if not placement.intent_matches_effect:
        cost.notes.append(
            f"切分意图与算子编译器实际除数不符：下发 kind={placement.kind} "
            f"numDpus={placement.num_dpus}，回传 placed_shards="
            f"{placement.placed_shards}")

    # 显式 DMA 是 **A 路**的判据：那条链从 tt.load/store 出发，必然经过它。
    # B 路（整算子级）没有访存可显式化，拿这条去卡会把正常的产物判成
    # "pass 没生效"。
    # 判据是「还有没有没被显式化的访存」，不是「有没有 tt.dot」：
    # `-pim-explicit-dma` 没跑起来时留下的是 `tt.load` / `tt.store`，逐元素与
    # 归约类的内核本来就没有 `tt.dot`，拿它当代理会让那些内核带着坏掉的 DMA
    # 静默通过。B 路（整算子级）压根没有访存，两个都不命中，不会被误卡。
    if ir_level != "oplevel" and (
            _TT_LOAD_RE.search(text) or _DOT_RE.search(text)):
        assert dma_ops, (
            f"{kernel_name} 的 pim mlir 里还有没显式化的 tt.load/tt.store 或 "
            f"tt.dot，却没有 pim.dma_*：pass 可能没生效")
    # 在结果中记录超过 WRAM 预算的分块。判据走回传载体自己的属性，
    # 免得「怎么算超限」在消费方各写一遍。
    if feedback.over_wram_budget:
        notes.append(
            f"WRAM 超预算: 用了 {cost.wram_bytes_used} B / 预算 "
            f"{cost.wram_bytes_budget} B（FlagTree 当前只 warning、不重切 tile）"
        )
    return cost


def count_mnemonics(text: str) -> Dict[str, float]:
    """数一份**整算子级** IR 里各 mnemonic 出现了几次。

    必须在 `lower_oplevel_to_pimir` **之前**数：展开之后 `pim.softmax` 已经变成
    `pim.reduce_axis` / `pim.eltwise` / `pim.lut` 的相位链，`pim.rope` 同理，
    再数就一个都数不到，而"数到 0"看起来和"这个算子没发出来"一模一样。

    一行的算子名有别名（`pim.quantize` 与 `pim.dynamic_quant`），计数也走
    `_OPLEVEL_CANONICAL`，与计费同一份口径。视图类另有 `pim.view` 一栏汇总。
    """
    ops_re = _mnemonic_re(_OPLEVEL_CANONICAL)
    counts: Dict[str, float] = {}
    for raw in text.splitlines():
        for name in ops_re.findall(raw):
            key = _OPLEVEL_CANONICAL[name]
            counts[key] = counts.get(key, 0.0) + 1.0
        if _MOVEMENT_RE.search(raw) or _KV_WRITE_RE.search(raw):
            counts["pim.view"] = counts.get("pim.view", 0.0) + 1.0
    return counts


