"""把融合后的 FX 图发射成整算子级 PIM MLIR，喂给算子编译器展开相位。

这是「图编译器 → 算子编译器」链路上原先缺的一环。在此之前相位结构写死在
`contracts/gml_hw_table.py` 的静态表里；现在真实模型的算子能走到 FlagTree 的
`-pim-expand-phases` 面前，由它给出相位模板。

发射三类有内部相位结构的算子，其余跳过（单相算子不需要展开）：

    DQ（DynamicScaling）   -> pim.quantize {dynamic, per_group}   4 相
    Softmax                -> pim.softmax                          5 相
    RoPE（Llama2Activation）-> pim.rope                             3 相

**每个算子一个 `tt.func`**，函数名是 `<FX 节点名>__<kind>`。后缀不是装饰：
K 路 RoPE 的锚点同时带 `ROPE_META_KEY` 和 `DQ_META_KEY`（对应 GML 的
`Llama2ActivationDQ`，实测参考产物里那个节点**同时**有 4 个相位号和
`Llama2Activation_*` 子块），一个节点要发两个算子，不加后缀第二个会同名。

对回 FX 节点用 `EmittedOp.fx_name`（保存的是不带后缀的原名），再顺着 GML 的
`label` 字段对回 node_id（`gml_bridge/export.py::_dq_specs` 已经在用同一条
反查）。一算子一函数还有个好处：单个算子发射有问题不会连带其余，整图一函数
会一损俱损。

**module 必须带 `pim.target`**：手写的 MLIR 不经过 `convert-triton-to-pim`，
而 Triton 的 `verifyTensorSize` 靠这个属性放开 2 的幂限制
（`lib/Dialect/Triton/IR/Traits.cpp`）。不写它，llama2 的 11008 宽张量会被
`tt.store` 的校验拒掉。

**形状口径按算子分别定**，逐条实测过（见 docs/gml-expand-phases-20260918.md）：

    DQ       压成 [1, numel]    让校验器算出的组数与 DynamicScalingSpec 一致
    Softmax  压成 [rows, S]     保住归约轴；压成 [1, rows*S] 会让归约跨行
    RoPE     保留原 rank-4      广播沿 head 轴，压平会把那个轴抹掉

RoPE 那条是反直觉的：src 有 65536 个元素而 cos/sin 只有 2048，压平后
`[1,65536]` 与 `[1,2048]` 不可广播，校验器直接拒。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch.fx import GraphModule, Node as FxNode

from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse_rope import ROPE_META_KEY
from graph.quant_pass import DQ_META_KEY
from graph.split_heads import HEAD_INDEX_META_KEY, HEAD_ROLE_META_KEY, ROLE_SOFTMAX

# K 路 RoPE 的判据。复用 GML 侧同一个函数，两边对「哪条是 K」必须一致 ——
# 各自判会在 Q/K 顺序变化时静默分叉。
from gml_bridge.from_fx import _is_second_rope

# 目标串。`verifyTensorSize` 认这个属性放开 2 的幂限制。
PIM_TARGET = "pim:v1"

# `FusedTail.activation` → FlagTree `ActivationKind` 的助记符。
#
# 两个融合 pass 的大小写不统一（`fuse_pim.py:48` 写 `"Silu"`，
# `fuse.py:39` 写 `"relu"`），所以这里统一按小写查。
_ACTIVATION_KINDS = frozenset({
    "relu", "relu_x", "leaky_relu", "sigmoid", "silu",
    "tanh", "gelu", "exp", "reciprocal", "sqrt", "rsqrt",
})

# K 路 RoPE 尾部量化的分组宽度。GML 侧 `from_fx.py` 发 `Llama2ActivationDQ`
# 时固定用 128，这里跟它一致（那个锚点上没有 DynamicScalingSpec 可查）。
ROPE_DQ_GROUP_SIZE = 128

# fp16 是参考产物里激活与中间态的存储类型（W4A8 的 A8 指量化后的 int8，
# 量化前的激活是 fp16）。
_F16 = "f16"
_I8 = "i8"


@dataclass
class EmittedOp:
    """发射出的一个算子。

    `func` 是 MLIR 函数名（= FX 节点名）；`kind` 是三类之一，
    供调用方核对相位数（DQ 4 / Softmax 5 / RoPE 3）。
    """

    func: str
    kind: str                # "dq" / "softmax" / "rope"
    fx_name: str
    expected_phases: int


@dataclass
class EmitReport:
    """一次发射的统计。"""

    text: str = ""
    ops: list[EmittedOp] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[EmittedOp]:
        return [op for op in self.ops if op.kind == kind]

    def __str__(self) -> str:
        counts = {k: len(self.by_kind(k)) for k in ("dq", "softmax", "rope")}
        return (f"发射 {len(self.ops)} 个算子"
                f"（DQ {counts['dq']}、Softmax {counts['softmax']}、"
                f"RoPE {counts['rope']}）")


def _shape_of(node: object) -> tuple[int, ...] | None:
    """节点输出的静态形状。取不到就不发射——宁可少发也不要瞎猜。"""
    if not isinstance(node, FxNode):
        return None
    value = node.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(extent) for extent in shape)
    except (TypeError, ValueError):
        # 动态维（SymInt）取不到 int，MLIR 需要静态类型，只能跳过。
        return None


def _numel(shape: tuple[int, ...]) -> int:
    count = 1
    for extent in shape:
        count *= extent
    return count


def _tensor(shape: tuple[int, ...], dtype: str) -> str:
    """MLIR 张量类型字面量，如 `tensor<1x4096xf16>`。"""
    return f"tensor<{'x'.join(str(d) for d in shape)}x{dtype}>"


def _sanitize(name: str) -> str:
    """FX 节点名转成合法的 MLIR 符号名。

    FX 的名字本来就是 `[A-Za-z0-9_]`，但保险起见把其余字符换成下划线 ——
    名字要能原样对回 FX 节点，所以只做替换、不做截断。
    """
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def _emit_dq(node: FxNode) -> tuple[str, EmittedOp] | None:
    """DQ -> `pim.quantize {dynamic, per_group}`，压成 `[1, numel]`。

    压平的理由：校验器（`Ops.cpp::verifyQuantOperand`）按**单轴**算组数，
    对 `(1,16,4096)` axis=2 得 4096/128 = 32，而 spec 要 512 组（16 行各
    32 组）。压成 `[1,65536]` 后校验器算 65536/128 = 512，与 spec 一致。
    行主序下「第 r 行第 g 组」就是扁平的第 `r*32+g` 组，语义等价。
    """
    spec = node.meta[DQ_META_KEY]
    shape = _shape_of(node)
    if shape is None:
        return None

    numel = _numel(shape)
    # spec.numel 来自图编译器的量化 pass；与形状不符说明有一侧算错了，
    # 这种情况宁可跳过也不要发一份错的 IR 出去。
    if numel != spec.numel:
        return None
    group_size = spec.group_size
    if group_size <= 0 or numel % group_size:
        return None
    groups = numel // group_size

    # 后缀 `__dq`：K 路 RoPE 的锚点同时要发 RoPE 和 DQ 两个算子，
    # 不加后缀两者同名，第二个会被去重吃掉。
    func = f"{_sanitize(node.name)}__dq"
    src_ty = _tensor((1, numel), _F16)
    scale_ty = _tensor((groups,), _F16)
    out_ty = _tensor((1, numel), _I8)
    body = f"""  tt.func @{func}(%x: {src_ty}, %s: {scale_ty}) {{
    %q = pim.quantize %x, %s
       {{dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = {group_size}>}}
       : {src_ty}, {scale_ty} -> {out_ty}
    tt.return
  }}"""
    return body, EmittedOp(func=func, kind="dq", fx_name=node.name,
                           expected_phases=4)


def _emit_dq_for_rope(node: FxNode) -> tuple[str, EmittedOp] | None:
    """K 路 RoPE 尾部那一段量化 -> `pim.quantize {dynamic, per_group}`。

    与 `_emit_dq` 的区别只在 **group_size 的来源**：RoPE 锚点上没有
    `DynamicScalingSpec`（那是量化 pass 给 DQ 载体节点打的），而 GML 侧
    `from_fx.py` 对这一段固定用 128（见它发 `Llama2ActivationDQ` 时
    `phase_fields(..., group_size=128)` 那处）。这里跟它保持一致。
    """
    shape = _shape_of(node)
    if shape is None:
        return None
    numel = _numel(shape)
    group_size = ROPE_DQ_GROUP_SIZE
    if numel % group_size:
        return None
    groups = numel // group_size

    func = f"{_sanitize(node.name)}__dq"
    src_ty = _tensor((1, numel), _F16)
    scale_ty = _tensor((groups,), _F16)
    out_ty = _tensor((1, numel), _I8)
    body = f"""  tt.func @{func}(%x: {src_ty}, %s: {scale_ty}) {{
    %q = pim.quantize %x, %s
       {{dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = {group_size}>}}
       : {src_ty}, {scale_ty} -> {out_ty}
    tt.return
  }}"""
    return body, EmittedOp(func=func, kind="dq", fx_name=node.name,
                           expected_phases=4)


def _emit_softmax(node: FxNode) -> tuple[str, EmittedOp] | None:
    """Softmax -> `pim.softmax`，压成 `[rows, S]`。

    **不能压成 `[1, rows*S]`**：softmax 沿最后一维归约，压平会让归约跨行，
    32 个头的分数混在一起。`(1,1,16,16)` -> `[16,16]` axis=1。
    """
    shape = _shape_of(node)
    if shape is None or len(shape) < 2:
        return None
    last = shape[-1]
    rows = _numel(shape) // last

    func = f"{_sanitize(node.name)}__softmax"
    ty = _tensor((rows, last), _F16)
    body = f"""  tt.func @{func}(%s: {ty}) {{
    %p = pim.softmax %s {{axis = 1 : i64, unit = #pim.unit<cstl>}} : {ty} -> {ty}
    tt.return
  }}"""
    return body, EmittedOp(func=func, kind="softmax", fx_name=node.name,
                           expected_phases=5)


def _emit_rope(node: FxNode) -> tuple[str, EmittedOp] | None:
    """RoPE -> `pim.rope`，**保留原 rank-4 形状**。

    这一条与另外两类相反，压平会坏掉：src `(1,32,16,128)` 有 65536 个元素，
    cos/sin `(1,1,16,128)` 只有 2048 个，压成 rank-2 后 `[1,65536]` 与
    `[1,2048]` 不可广播，`pim.eltwise` 的校验器直接拒。广播必须沿 head 轴
    发生（32 个头共用一份 cos/sin），压平把那个轴抹掉了。
    """
    match = node.meta[ROPE_META_KEY]
    src_shape = _shape_of(match.source)
    cos_shape = _shape_of(match.cos)
    sin_shape = _shape_of(match.sin)
    if src_shape is None or cos_shape is None or sin_shape is None:
        return None
    if cos_shape != sin_shape:
        return None
    # 广播要求 cos/sin 的每一维要么等于 src，要么是 1。
    if len(cos_shape) != len(src_shape):
        return None
    if any(c not in (1, s) for c, s in zip(cos_shape, src_shape)):
        return None

    # head 数取 src 里被 cos/sin 广播掉的那一维；取不到就按整条一头算。
    num_heads = next((s for c, s in zip(cos_shape, src_shape) if c == 1 and s > 1), 1)

    func = f"{_sanitize(node.name)}__rope"
    src_ty = _tensor(src_shape, _F16)
    tab_ty = _tensor(cos_shape, _F16)
    body = f"""  tt.func @{func}(%x: {src_ty}, %c: {tab_ty}, %s: {tab_ty}) {{
    %y = pim.rope %x, %c, %s
       {{numHeads = {num_heads} : i64, unit = #pim.unit<cstl>}}
       : {src_ty}, {tab_ty}, {tab_ty} -> {src_ty}
    tt.return
  }}"""
    return body, EmittedOp(func=func, kind="rope", fx_name=node.name,
                           expected_phases=3)


def _emit_fused_matmul(node: FxNode) -> tuple[str, EmittedOp] | None:
    """带尾部激活的矩阵乘 -> `pim.matmul {activation}`。

    这一类**不产出相位**（矩阵乘是单相），发它是为了让
    `-pim-fuse-activation` 有东西可校验：图编译器已经把激活折进主算子了，
    所以我方直接发带 `activation` 属性的 `pim.matmul`，pass 跑过应该
    「无事可做」——它是幂等的，已带 `activation` 的算子会跳过。

    验收就是跑 pass 前后 `activation` 属性个数不变：相等说明两边的融合口径
    一致；变多说明我方漏折了某处（pass 替我们补上），变少则是 pass 不认我方
    的表达。
    """
    tail = node.meta[FUSED_TAIL_META_KEY]
    kind = str(tail.activation).lower()
    if kind not in _ACTIVATION_KINDS:
        return None

    out_shape = _shape_of(node)
    if out_shape is None or len(out_shape) < 2:
        return None
    # 压成 [M, N]：矩阵乘的前导维在目标格式里同样是压平的。
    n = out_shape[-1]
    m = _numel(out_shape) // n
    # 权重宽度取激活输入的末维；取不到就不发（宁可少发也不猜）。
    src = node.args[0] if node.args else None
    src_shape = _shape_of(src)
    if src_shape is None:
        return None
    k = src_shape[-1]

    func = f"{_sanitize(node.name)}__matmul"
    a_ty = _tensor((m, k), _I8)
    b_ty = _tensor((k, n), _I8)
    out_ty = _tensor((m, n), _F16)
    body = f"""  tt.func @{func}(%a: {a_ty}, %b: {b_ty}) {{
    %o = pim.matmul %a, %b
       {{activation = #pim.act_spec<kind = {kind}>,
        datapath = #pim.datapath<nmuMode = fixed_point, scaleMode = fixed_point>}}
       : {a_ty}, {b_ty} -> {out_ty}
    tt.return
  }}"""
    return body, EmittedOp(func=func, kind="fused_matmul", fx_name=node.name,
                           expected_phases=0)


def _is_softmax(node: FxNode) -> bool:
    """逐头展开后的 softmax。

    两条判据都要：`ROLE_SOFTMAX` 是 `split_attention_heads` 打的标记，
    而未展开的图里 softmax 还是裸的 aten 目标。
    """
    if node.meta.get(HEAD_ROLE_META_KEY) == ROLE_SOFTMAX:
        return True
    return "softmax" in str(node.target)


def emit_oplevel_mlir(gm: GraphModule) -> EmitReport:
    """把融合后的图发射成整算子级 PIM MLIR。

    只发 DQ / Softmax / RoPE 三类。**要求 `gm` 已经跑过
    `gml_bridge.export.export_graph` 的那串 pass**，否则 RoPE 还没折成单节点、
    DQ 还没插进去。

    注意那串 pass **从不删节点**：被折叠/吸收的算子仍留在图里（打
    `ABSORBED_META_KEY`），DQ / KV-DMA / Split 由新插的 `alias` 节点承载。
    所以这里按 meta 标记认算子，不按 aten 目标扫——否则同一条 RoPE 会既按
    3 相发一次、又按散落的 mul/add 发若干次。
    """
    report = EmitReport()
    bodies: list[str] = []
    seen: set[str] = set()

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        # **一个节点可能要发两个算子**：K 路的 RoPE 尾部带量化，对应 GML 的
        # `Llama2ActivationDQ` —— 实测参考产物里那个节点**同时**有
        # `_phase_0..3` 四个相位号和 `Llama2Activation_*` 子块，所以它是
        # 「3 连 + 4 相」而不是二选一。
        #
        # K 路身份用 `_is_second_rope` 自己判，**不看 `DQ_META_KEY`**：
        # 那个标记是 `gml_bridge.from_fx.convert()` 的副作用（它为复用 DQ 的
        # 写盘路径补上去的），依赖它就意味着 emitter 的产出取决于序列化跑过
        # 没有 —— 实测老流程先调 export_graph 时 DQ 数是 41，改成只融合后
        # 变 40，而 40 才是对的。
        candidates: list[tuple[str, EmittedOp] | None] = []
        if ROPE_META_KEY in node.meta:
            candidates.append(_emit_rope(node))
            if _is_second_rope(node):
                candidates.append(_emit_dq_for_rope(node))
        elif DQ_META_KEY in node.meta:
            candidates.append(_emit_dq(node))
        if not candidates and _is_softmax(node):
            candidates.append(_emit_softmax(node))
        # 带尾部激活的矩阵乘：不产相位，发它是给 -pim-fuse-activation 校验用。
        if not candidates and FUSED_TAIL_META_KEY in node.meta:
            candidates.append(_emit_fused_matmul(node))
        if not candidates:
            continue

        for emitted in candidates:
            if emitted is None:
                report.skipped.append(node.name)
                continue
            body, op = emitted
            # 同名函数会让 MLIR 解析失败。同一节点发两个算子时，函数名
            # 已在各 _emit_* 里按 kind 加了后缀区分。
            if op.func in seen:
                report.skipped.append(node.name)
                continue
            seen.add(op.func)
            bodies.append(body)
            report.ops.append(op)

    report.text = (
        f'module attributes {{pim.target = "{PIM_TARGET}"}} {{\n'
        + "\n\n".join(bodies)
        + "\n}\n"
    )
    return report
