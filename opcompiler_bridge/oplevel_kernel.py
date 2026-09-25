"""算子级 kernel 的 MLIR 文本：一段 `tt.func`，形状由调用方给定。

两条路径共用这一份：

- 图编译器侧 `oplevel_emitter` 按 FX 节点发（一次发整张图的所有算子）；
- 算子编译器 driver 侧按 `OpCompileRequest` 发（一次发一个算子）。

两边必须是同一份文本。同一个算子在两条路径上编出不同的 IR，就会让 GML 与 C
各说各话——这正是本仓一直在防的那种失真：不报错，只是两边算的不是一回事。

形状约定与 `oplevel_emitter` 一致，理由也在那边：

- 动态量化压成 `[1, numel]`。算子是逐组归约，压平后「第 r 行第 g 组」就是扁平
  的第 `r*组数+g` 组，行主序下语义等价；不压平则校验器按单轴算组数对不上。
- Softmax 压成 `[rows, S]` 而**不能**压成 `[1, rows*S]`：它沿最后一维归约，
  压平会让归约跨行，各头的分数混在一起。
"""

from __future__ import annotations

_F16 = "f16"
_I8 = "i8"


def _tensor(shape: tuple[int, ...], dtype: str) -> str:
    """MLIR 张量类型字面量，如 `tensor<1x4096xf16>`。"""
    return f"tensor<{'x'.join(str(d) for d in shape)}x{dtype}>"


def dynamic_quant_kernel(func: str, numel: int, groups: int,
                         group_size: int) -> str:
    """动态量化：一个算子展开成四相（分组 absmax / 恒等表 / 倒数表 / 定点化）。"""
    src_ty = _tensor((1, numel), _F16)
    scale_ty = _tensor((groups,), _F16)
    out_ty = _tensor((1, numel), _I8)
    return f"""  tt.func @{func}(%x: {src_ty}) {{
    %q, %s = pim.dynamic_quant %x
       {{groupSize = {group_size} : i64, axis = 1 : i64,
        spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = {group_size}, spg = true, spgAxis = 3, spgGroupSize = {group_size}>}}
       : {src_ty} -> {out_ty}, {scale_ty}
    tt.return
  }}"""


def softmax_kernel(func: str, rows: int, cols: int) -> str:
    """Softmax：沿最后一维，展开成五相。"""
    ty = _tensor((rows, cols), _F16)
    return f"""  tt.func @{func}(%s: {ty}) {{
    %p = pim.softmax %s {{axis = 1 : i64, unit = #pim.unit<cstl>}} : {ty} -> {ty}
    tt.return
  }}"""


def gather_kernel(func: str, vocab: int, hidden: int, count: int,
                  index_dtype: str = "i32") -> str:
    """词嵌入查表：按索引取表的行。纯访存，不做算术。

    索引压成 `[1, count]`：行号与行内偏移无关，压平不改变"取哪几行"。
    表保持 `[vocab, hidden]` 不压——行宽就是它的第二维，压掉就没有行的概念了。
    """
    table_ty = _tensor((vocab, hidden), _F16)
    ids_ty = _tensor((1, count), index_dtype)
    out_ty = _tensor((1, count, hidden), _F16)
    return f"""  tt.func @{func}(%t: {table_ty}, %i: {ids_ty}) {{
    %y = pim.gather %t, %i : {table_ty}, {ids_ty} -> {out_ty}
    tt.return
  }}"""


def rope_kernel(func: str, heads: int, seq: int, head_dim: int,
                tail_card_value: int = 0) -> str:
    """RoPE：`x*cos + rotate_half(x)*sin`，展开成三相。

    **保留 rank-4 形状**，与另外两类相反。压平会坏掉：src 有 heads×seq×head_dim
    个元素，cos/sin 只有 seq×head_dim 个，压成 rank-2 后两者不可广播——广播必须
    沿 head 轴发生（所有头共用一份 cos/sin），压平把那个轴抹掉了。

    `tail_card_value` 是末相的寻址卡值：K 路写 cache 前重定标是 3，Q 路留给
    紧随的 DQ 是 0。缺了它展开 pass 会把末相当成卡值 0，K 路的 `dq_contraction`
    标记一起消失，而 IR 仍合法。
    """
    src_ty = _tensor((1, heads, seq, head_dim), _F16)
    tab_ty = _tensor((1, 1, seq, head_dim), _F16)
    kantor = f"tailCardValue = {tail_card_value} : i64, " if tail_card_value else ""
    return f"""  tt.func @{func}(%x: {src_ty}, %c: {tab_ty}, %s: {tab_ty}) {{
    %y = pim.rope %x, %c, %s {{{kantor}numHeads = {heads} : i64, unit = #pim.unit<cstl>}}
       : {src_ty}, {tab_ty}, {tab_ty} -> {src_ty}
    tt.return
  }}"""


def matmul_kernel(func: str, m: int, k: int, n: int,
                  group_size: int | None = None,
                  elem_bits: int = 4,
                  activation: str | None = None,
                  out_dtype: str | None = None,
                  nmu_mode: str = "floating_point",
                  scale_mode: str = "floating_point",
                  sf_multiplier: int = 1,
                  fp16: bool = False) -> str:
    """整算子矩阵乘，可带按组反量化累加。

    **不压平**：`[M, K] x [K, N]` 的三维就是矩阵单元乘的那两维，压掉任一维
    就不再是矩阵乘。`fp16=False`（默认）时两个操作数都是 i8——int4 权值按
    1 字节/元素存、符号扩展到 `[-8, 7]`，没有半字节打包。

    `fp16=True` 是注意力那条路：两个操作数都是 fp16 激活，没有权值，所以
    **不发 `weightBinding`**（那是描述权值侧字段的，激活之间相乘没有权值侧）。
    矩阵单元本身不区分这两者，EmitC 侧按缓冲区的元素类型取值，同一套三重循环。

    `group_size` 给了就是 w4a8 投影：整数累加器每 `group_size` 个 K 停一次，
    把这一组反量化再折进 f32 总数。不给则整段 K 一个组——两个累加顺序在浮点
    域里不是同一个数，所以它必须显式写进 IR 而不是留默认。

    分组时**必须带 `[K/group_size, N]` 的逐组定标操作数**（评审 20260923 的
    P1-4）。原来这里不发它，C 侧在组边界乘的是 `weight_binding.sfMultiplier`
    一个标量——而一个标量乘在组边界与乘在整行末尾在精确算术里相等，于是
    「按组反量化」与「整段累加完再反量化」给出同一个数，方案 §5.15 那条判据
    无论实现对错都会通过。
    """
    if fp16 and group_size is not None:
        raise ValueError("fp16 的矩阵乘没有权值侧，不支持按组反量化")
    operand_dtype = _F16 if fp16 else _I8
    a_ty = _tensor((m, k), operand_dtype)
    w_ty = _tensor((k, n), operand_dtype)
    out_ty = _tensor((m, n), out_dtype or operand_dtype)
    if group_size is not None and k % group_size:
        raise ValueError(
            f"matmul 的 K={k} 不是 {group_size} 的整数倍，最后一组会短一截")
    extra = f", sfMultiplier = {sf_multiplier} : i64" if sf_multiplier != 1 else ""
    binding = (f"#pim.weight_binding<format = weight, role = model_weight, "
               f"elemBits = {elem_bits}{extra}>")
    if group_size is None:
        datapath = (f"#pim.datapath<nmuMode = {nmu_mode}, "
                    f"scaleMode = {scale_mode}>")
    else:
        datapath = (f"#pim.datapath<nmuMode = {nmu_mode}, "
                    f"scaleMode = {scale_mode}, groupDequantAccum = true, "
                    f"groupSize = {group_size}>")
        binding = (f"#pim.weight_binding<format = weight, "
                   f"role = model_weight, elemBits = {elem_bits}, "
                   f"groupSize = {group_size}{extra}>")
    act = ""
    if activation:
        kind = str(activation).lower()
        act = f", activation = #pim.act_spec<kind = {kind}>"
    if fp16:
        # 没有权值侧，所以 `weightBinding` 与 `stationarity` 都不发：后者一旦
        # 写成 `kv`，verifier 就要求 `weightBinding` 声明
        # `role = activation_as_weight`，而那是 int8 权值路的说法。这两个缓冲
        # 这里是 fp16 激活，硬套一份权值绑定等于给读回侧一个不成立的事实。
        return f"""  tt.func @{func}(%a: {a_ty}, %w: {w_ty}) {{
    %y = pim.matmul %a, %w
       {{datapath = {datapath}{act}}}
       : {a_ty}, {w_ty} -> {out_ty}
    tt.return
  }}"""
    if group_size is None:
        return f"""  tt.func @{func}(%a: {a_ty}, %w: {w_ty}) {{
    %y = pim.matmul %a, %w
       {{datapath = {datapath}, weightBinding = {binding},
        stationarity = #pim.stationarity<weight>{act}}}
       : {a_ty}, {w_ty} -> {out_ty}
    tt.return
  }}"""
    scale_ty = _tensor((k // group_size, n), _F16)
    return f"""  tt.func @{func}(%a: {a_ty}, %w: {w_ty}, %s: {scale_ty}) {{
    %y = pim.matmul %a, %w scales %s
       {{datapath = {datapath}, weightBinding = {binding},
        stationarity = #pim.stationarity<weight>{act}}}
       : {a_ty}, {w_ty} scales {scale_ty} -> {out_ty}
    tt.return
  }}"""


def normalize_kernel(func: str, rows: int, cols: int, gamma: int) -> str:
    """RMS 归一化：`[rows, cols]` 的激活配 `[gamma]` 的缩放和标量 ε。

    压成二维而不是一维：归约沿末轴发生，压平会让每一行都跨过别的行。
    ε 是操作数而不是属性：图侧从 `RMSNorm_Add_Const` 读它，缺了就让
    lowering 报错，不在 C 里发明 1e-5。
    """
    src_ty = _tensor((rows, cols), _F16)
    gamma_ty = _tensor((gamma,), _F16)
    eps_ty = _tensor((1,), "f32")
    return f"""  tt.func @{func}(%x: {src_ty}, %g: {gamma_ty}, %e: {eps_ty}) {{
    %y = pim.normalize %x, %g eps %e {{axis = 1 : i64, rmsNorm,
        vpuParams = #pim.vpu_params<axis = -1, useScaling = false>}}
       : {src_ty}, {gamma_ty} eps {eps_ty} -> {src_ty}
    tt.return
  }}"""


def mask_kernel(func: str, scores: tuple[int, ...],
                mask_shape: tuple[int, ...], *,
                layout: str | None = None) -> str:
    """加性掩码：分数 + 掩码，掩码按末轴广播。

    `layout` 不给时按形状选：除末轴外全是 1 走 `vector`（decode 一行对整段
    缓存），末两轴相等走 `causal_tril`（prefill 三角）。形状相近时单看形状
    分不出来，调用方必须显式给。
    """
    scores_ty = _tensor(scores, _F16)
    mask_ty = _tensor(mask_shape, _F16)
    if layout is None:
        if all(d == 1 for d in mask_shape[:-1]):
            layout = "vector"
        elif len(mask_shape) >= 2 and mask_shape[-1] == mask_shape[-2]:
            layout = "causal_tril"
        else:
            raise ValueError(
                f"mask 形状 {mask_shape} 分不出 layout，调用方必须显式给")
    return f"""  tt.func @{func}(%s: {scores_ty}, %m: {mask_ty}) {{
    %y = pim.mask %s, %m {{layout = #pim.mask_layout<{layout}>}} : {scores_ty}, {mask_ty} -> {scores_ty}
    tt.return
  }}"""


def transpose_kernel(func: str, shape: tuple[int, ...],
                     axes: tuple[int, ...]) -> str:
    """按轴序重排。`onthefly` 与 `absorbed` 一起说「这一层不产生独立节点」——
    但值仍然要落到正确顺序上，内核的输出就是一个扁平缓冲。"""
    out_shape = tuple(shape[a] for a in axes)
    axes_attr = ", ".join(str(a) for a in axes)
    src_ty = _tensor(shape, _F16)
    out_ty = _tensor(out_shape, _F16)
    return f"""  tt.func @{func}(%x: {src_ty}) {{
    %y = pim.transpose %x {{axes = array<i64: {axes_attr}>, onthefly, purpose = #pim.transpose_purpose<purpose = absorbed>}}
       : {src_ty} -> {out_ty}
    tt.return
  }}"""


def reshape_kernel(func: str, shape: tuple[int, ...],
                   out_shape: tuple[int, ...]) -> str:
    """改形状，元素顺序不变。"""
    src_ty = _tensor(shape, _F16)
    out_ty = _tensor(out_shape, _F16)
    return f"""  tt.func @{func}(%x: {src_ty}) {{
    %y = pim.reshape %x : {src_ty} -> {out_ty}
    tt.return
  }}"""


def concat_kernel(func: str, shapes: list[tuple[int, ...]], axis: int) -> str:
    """沿一根轴接起来。"""
    tys = [_tensor(s, _F16) for s in shapes]
    out_shape = list(shapes[0])
    out_shape[axis] = sum(s[axis] for s in shapes)
    out_ty = _tensor(tuple(out_shape), _F16)
    args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(tys))
    operands = ", ".join(f"%a{i}" for i in range(len(tys)))
    return f"""  tt.func @{func}({args}) {{
    %y = pim.concat {operands} {{axis = {axis} : i64}} : {", ".join(tys)} -> {out_ty}
    tt.return
  }}"""


# 契约里的元素类型名（torch 风格）到 MLIR 类型名的映射。`pim.convert` 的目标
# 类型来自契约，不能把 `float32` 直接拼进 IR——那不是合法的 MLIR 类型。
_CONTRACT_DTYPES = {"float16": _F16, "float32": "f32", "int8": _I8}


def convert_kernel(func: str, shape: tuple[int, ...], dtype: str,
                   src_dtype: str = "float16") -> str:
    """只换元素类型。

    **两侧都要给**：图上 `to.dtype` 两个方向都有（RMSNorm 链里是 f32→f16，
    量化落盘是 f16→i8）。源侧写死成 fp16 的话，f32→f16 会被发成 f16→f16——
    而那不是一次转换，verifier 直接拒。
    """
    for label, name in (("目标", dtype), ("源", src_dtype)):
        if name not in _CONTRACT_DTYPES:
            raise ValueError(
                f"convert 的{label}类型 {name!r} 没有对应的 MLIR 类型，"
                f"可选：{sorted(_CONTRACT_DTYPES)}")
    if src_dtype == dtype:
        raise ValueError(
            f"convert 两侧都是 {dtype!r}，这不是一次转换；同类型的 `to.dtype` "
            f"是恒等，调用方应当在发算子之前就把它略过")
    src_ty = _tensor(shape, _CONTRACT_DTYPES[src_dtype])
    out_ty = _tensor(shape, _CONTRACT_DTYPES[dtype])
    return f"""  tt.func @{func}(%x: {src_ty}) {{
    %y = pim.convert %x : {src_ty} -> {out_ty}
    tt.return
  }}"""


def lut_kernel(func: str, shape: tuple[int, ...], kind: str) -> str:
    """查表激活。Silu 是 Llama2 里唯一未融合时的兜底路径。

    表体不在这里发：C 侧按 kind 直接算那个函数，`phase_data.py` 是数值真源，
    再带一份 288 字节的表就是第三份公式。
    """
    ty = _tensor(shape, _F16)
    return f"""  tt.func @{func}(%x: {ty}) {{
    %y = pim.lut %x {{kind = #pim.activation<{kind}>}} : {ty} -> {ty}
    tt.return
  }}"""


def eltwise_kernel(func: str, shape: tuple[int, ...], kind: str) -> str:
    """逐元素二元运算。两槽形态打印与旧糖一致，成本抽取按 mnemonic 计数。"""
    ty = _tensor(shape, _F16)
    return f"""  tt.func @{func}(%a: {ty}, %b: {ty}) {{
    %y = pim.eltwise %a, %b
       {{kind = #pim.eltwise<{kind}>,
         datapath = #pim.datapath<nmuMode = floating_point, scaleMode = floating_point>}}
       : {ty}, {ty} -> {ty}
    tt.return
  }}"""


def split_heads_kernel(func: str, shape: tuple[int, ...], axis: int,
                       num_heads: int) -> str:
    """把一个打包投影按头数拆开，每份大小相同。

    多个结果 → 多个输出指针，按结果顺序排在输入之后。分不出头的那份数据
    （轴长不能被头数整除）在这里就报错，不留给内核。
    """
    src_ty = _tensor(shape, _F16)
    extent = shape[axis]
    head = extent // num_heads
    piece = list(shape)
    piece[axis] = head
    piece_ty = _tensor(tuple(piece), _F16)
    results = ", ".join([piece_ty] * num_heads)
    names = ", ".join(f"%h{i}" for i in range(num_heads))
    return f"""  tt.func @{func}(%x: {src_ty}) {{
    {names} = pim.split_heads %x {{axis = {axis} : i64, numHeads = {num_heads} : i64}}
       : {src_ty} -> {results}
    tt.return
  }}"""


def kv_cache_kernel(func: str, value_shape: tuple[int, ...], cache_elems: int,
                    indexed: bool, elem: str = _I8) -> str:
    """KV cache 写入。给了索引是散写一行，不给是区间写。

    缓存是 memdesc 不是张量：它跨步存活，调用方持有那块缓冲，内核只拿一个
    指针进去。也没有输出——写进去就是全部结果。

    `elem` 是缓存与写入值共用的元素类型，默认 i8（量化后的 KV）。浮点 KV
    那条路（运行时 `sdpa_kernel`）传 `f16`——校验器只要求两者一致。
    """
    value_ty = _tensor(value_shape, elem)
    cache_ty = f"!pim.memdesc<{cache_elems}x{elem}, #pim.mram>"
    if indexed:
        idx_ty = _tensor((1,), "i16")
        return f"""  tt.func @{func}(%v: {value_ty}, %c: {cache_ty}, %pos: i32, %slot: {idx_ty}) {{
    pim.kv_cache %v, %c, %pos[%slot]
        {{layer = 0 : i64, isKey, inputBufferPolicy = "L2A_ignore"}}
       : {value_ty}, {cache_ty}, i32 [{idx_ty}]
    tt.return
  }}"""
    return f"""  tt.func @{func}(%v: {value_ty}, %c: {cache_ty}, %pos: i32) {{
    pim.kv_cache %v, %c, %pos
        {{layer = 0 : i64, isKey, mode = #pim.kv_mode<range_write>,
         inputBufferPolicy = "L2A_ignore"}}
       : {value_ty}, {cache_ty}, i32
    tt.return
  }}"""
