"""把融合后的 FX 图转成 GML 节点与边。

只做结构映射——算子类型、拓扑、缓冲区命名、形状。量化参数留给第 4 轮，那时
`node.meta` 里会带上 scale/zp，本模块按同一套字段名填进去即可。

三条容易写错的约定，都在 docs/gml-lowering-20260914.md 第 3 节有验证依据：

- 节点 id 逆拓扑编号：id 越小越靠输出，最终输出是 id 2。
- 缓冲区按**消费者**编号，命名规则见 contracts/gml_names.py。
- 形状只在 edge.dims 上，节点内不带。
"""

from __future__ import annotations

import torch
from torch.fx import GraphModule, Node as FxNode

from contracts import gml_hw_table as hw_table
from contracts import gml_names as names
from contracts.compile_slots import DEFAULT_SLOTS, CompileSlots
from contracts.gml_quant import GML_VERSION, PHASE_COUNTS
from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse_pim import (
    ABSORBED_META_KEY,
    ATTENTION_SCALE_META_KEY,
    RMS_NORM_META_KEY,
)
from graph.quant_pass import DQ_META_KEY
from graph.fuse_rope import ROPE_META_KEY
from graph.kv_dma_pass import KV_DMA_META_KEY, SPLIT_META_KEY
from graph.split_heads import (
    HEAD_INDEX_META_KEY,
    HEAD_ROLE_META_KEY,
    ROLE_MASK,
    ROLE_MATMUL_PV,
    ROLE_MATMUL_QK,
    ROLE_SOFTMAX,
)
from gml_bridge.writer import Edge, Node

# FX 算子到 GML `op_type` 的映射。GML 的算子集比 aten 小得多，因为定点流水线里
# 很多 aten 算子（类型转换、断言）没有对应的硬件节点。
OP_TYPES = {
    torch.ops.aten.linear.default: "Gemm",
    torch.ops.aten.addmm.default: "Gemm",
    torch.ops.aten.mm.default: "MatMul",
    torch.ops.aten.bmm.default: "MatMul",
    # 逐头展开产出的是 `matmul`（保留 4 维），不是 `bmm`（要先压成 3 维）。
    torch.ops.aten.matmul.default: "MatMul",
    torch.ops.aten.add.Tensor: "EltwiseAdd",
    torch.ops.aten.sub.Tensor: "EltwiseSub",
    torch.ops.aten.mul.Tensor: "EltwiseMul",
    torch.ops.aten.div.Tensor: "EltwiseDiv",
    torch.ops.aten._softmax.default: "Softmax",
    # 以下按 llama2 W4A8 实物校正：它们在 GML 里是**独立节点**，
    # 不是折进主算子 contraction 的项（见文档 18.6）。
    # `RMSNorm_vpu` 的 `_vpu` 后缀说明它显式绑定在向量单元上，
    # 还带 vpu_params / Vpu_Axis / Use_Scaling / RMSNorm_Add_Const 等专属字段。
    torch.ops.aten.rsqrt.default: "RMSNorm_vpu",
    # `Silu` 自带 nmu_mode / fpsu_* / kantor_mode，以及一个放 fused_Silu_act
    # 的 contraction——它是主算子而非激活。
    torch.ops.aten.silu.default: "Silu",
    # causal mask 在硬件上是一步操作，实物里 32 个（每 head 一个）。
    torch.ops.aten.masked_fill.Scalar: "Mask",
    torch.ops.aten.where.self: "Mask",
    # 按 head 切分：实物用一个 Split 节点带 num_heads 表达。
    torch.ops.aten.split.Tensor: "Split",
    torch.ops.aten.split_with_sizes.default: "Split",
    torch.ops.aten.transpose.int: "Transpose",
    torch.ops.aten.permute.default: "Transpose",
    torch.ops.aten.view.default: "Reshape",
    torch.ops.aten.reshape.default: "Reshape",
    torch.ops.aten.cat.default: "Concat",
    torch.ops.aten.max_pool2d.default: "MaxPool",
    torch.ops.aten.avg_pool2d.default: "AveragePool",
    torch.ops.aten.convolution.default: "Conv",
    # attention 在 GML 里是真实节点（scores/context 两个 MatMul 加一个 Softmax）。
    # 本仓的 NumPy 路径把它放在主机侧执行，但那是执行策略，不是图结构——
    # 如果这里跨过它，它的 q/k/v 三个上游会被下游节点误当成自己的输入。
    torch.ops.aten.scaled_dot_product_attention.default: "MatMul",
}

# 带 output_sf / output_zp 的算子。实测规则：**做计算的带，纯布局的不带**。
#
#   带（个数/总数）：MatMul 64/64、DynamicScaling 36/36、Softmax 32/32、
#                    Mask 32/32、Gemm 7/7、EltwiseAdd 2/2、EltwiseMul 1/1、
#                    RMSNorm_vpu 2/2、KV_Cache_DMA 2/2
#   不带：Transpose 0/4、Reshape 0/2、Concat 0/1
#
# 判据是「这个算子会不会改变数值的动态范围」—— 换轴、改形状、拼接都不会，
# 所以下游沿用上游的 scale，不需要 requant。
_OUTPUT_SCALE_OPS = frozenset({
    "Gemm", "MatMul", "Softmax", "Mask", "DynamicScaling",
    "EltwiseAdd", "EltwiseMul", "RMSNorm_vpu", "KV_Cache_DMA",
    "Llama2Activation", "Llama2ActivationDQ", "Silu",
})


def _op_type_of(node):
    """节点的 GML `op_type`。

    优先级固定：DQ > RMSNorm 折叠 > 逐头角色 > aten 目标查表。
    抽成函数是因为三处都要用它：发射本节点的字段、生产者给输出缓冲定槽号、
    以及 dtype 沿边传播时定键名 —— 三处必须得到同一个答案。
    """
    # RoPE 要在 DQ **之前**判。K 路的 RoPE 节点两个标记都有：
    # `fuse_rope` 打 ROPE，而本模块发射 `Llama2ActivationDQ` 时会补一个
    # DQ 标记（为了复用 DQ 的写盘路径，见 `convert` 里那处 `node.meta[...]`）。
    # 顺序反了的话，同一张图第二次 convert 就会把它降级成 DynamicScaling
    # —— 实测会让 GML 从 200 节点变 206、op_type 也变掉。
    if ROPE_META_KEY in node.meta:
        # 折叠出来的 RoPE。参考产物：
        #   Q -> Llama2ActivationDQ（RoPE 三连 + 后面 4 相量化）
        #   K -> Llama2Activation（三连，add 写 cache）
        # llama 图里 q_proj 在 k_proj 之前，拓扑序第一条是 Q。
        return ("Llama2ActivationDQ"
                if not _is_second_rope(node) else "Llama2Activation")
    if DQ_META_KEY in node.meta:
        return "DynamicScaling"
    if RMS_NORM_META_KEY in node.meta:
        return "RMSNorm_vpu"
    if KV_DMA_META_KEY in node.meta:
        return "KV_Cache_DMA"
    if SPLIT_META_KEY in node.meta:
        return "Split"
    role = node.meta.get(HEAD_ROLE_META_KEY)
    if role in _ROLE_OP_TYPES:
        return _ROLE_OP_TYPES[role]
    return OP_TYPES.get(node.target)


def _data_slot_count(op_type, input_count):
    """这个算子有几个**数据**输入槽。

    MatMul 的第二个 operand 走权重通路（`MatMul_input_as_weight 1`），
    不占数据槽 —— 所以两条入边只对应一个数据槽，键名不带槽号
    （实测 64/64 个 MatMul 都是 `input_buffer`，且 `input_count` 记 1）。

    RoPE 除了源张量还吃 cos / sin 两张表（它们是 `is_buffer` 边界节点，
    见 convert 里建表那段），所以是 **3 个**数据槽，`input_buffer_0..2`。
    `input_count` 只数算子入边，表节点不在其中，这里要加回来。

    Mask 不在这里处理：它是否有第二个数据槎（causal mask 边界节点）取决于
    具体节点是否真的接上了那条边（`_mask_placeholder_of` 找不到时就没有，
    比如 seq_len==1），不是纯靠 op_type 能判断的通用规则，所以调用点各自
    按 `node in mask_boundary_of` 处理，不进这个函数。

    生产者给输出缓冲命名时要按**消费者**的槎号，所以两边必须用同一判据：
    只在一处判、另一处沿用，否则生产者写 `input_buffer_1_176.bin`
    而消费者只声明 `input_buffer`，一边悬空一边多余。
    """
    if op_type == "MatMul":
        return max(1, input_count - 1)
    if op_type in ("Llama2Activation", "Llama2ActivationDQ"):
        return input_count + 2
    return input_count


def _shape_tuple_of(node) -> tuple[int, ...] | None:
    """节点输出的形状，取成 int 元组。

    **不要叫 `_shape_of`** —— 本模块已有一个同名函数（见下方 144 行附近），
    它返回的是 GML 边上的 dims 字符串（`"1x64x56x56"`）。两者重名的话
    后定义的会覆盖先定义的，调用处拿到字符串再去 `int()` 就炸在
    `invalid literal for int(): 'x'`。
    """
    value = getattr(node, "meta", {}).get("val") if node is not None else None
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(x) for x in shape)


def _is_second_rope(node) -> bool:
    """是不是图里第二条 RoPE 链（K 路）。

    llama 的 q_proj 排在 k_proj 前面，拓扑序第一条是 Q。
    Q 后面接 DQ 四相，走 Llama2ActivationDQ；K 进 cache 走 Llama2Activation。
    """
    ropes = [n for n in node.graph.nodes if ROPE_META_KEY in n.meta]
    return len(ropes) >= 2 and node is ropes[1]


def _numel_of_fx(node) -> int:
    """FX 节点输出的元素数。取不到形状时返回 0。"""
    shape = _shape_tuple_of(node)
    if not shape:
        return 0
    count = 1
    for extent in shape:
        count *= int(extent)
    return count


def _head_count_of(node) -> int:
    """RoPE 节点的头数。cos/sin 要广播到每个头，所以这个数进 `num_heads`。

    形状是 `[batch, heads, seq, head_dim]`，取第 1 维。
    """
    shape = _shape_tuple_of(node)
    return int(shape[1]) if shape and len(shape) == 4 else 1


def _shape_literal(shape: tuple[int, ...]) -> str:
    """形状写成 `[1, 1, 1, 4096]` 这种字面量，照实物的格式（逗号后带空格）。"""
    return "[" + ", ".join(str(int(x)) for x in shape) + "]"


# 带 FPSU 定标三族的算子，以及各自的槽数（实测）。
# 逐元素算子按槽配置（每个输入一组系数），其余单组。
_FPSU_OPS = frozenset({
    "Gemm", "MatMul", "KV_Cache_DMA", "EltwiseAdd", "EltwiseMul"})
_FPSU_SLOTS = {"EltwiseAdd": 2, "EltwiseMul": 2}


# 逐头展开产出的节点，按角色定 op_type。
_ROLE_OP_TYPES = {
    ROLE_MATMUL_QK: "MatMul",
    ROLE_MATMUL_PV: "MatMul",
    ROLE_MASK: "Mask",
    ROLE_SOFTMAX: "Softmax",
}


# 折进 contraction 的激活，GML 用 `Lut` 加 `activation_op_type` 表示。
_ACTIVATION_NAMES = {
    "relu": "Relu",
    "silu": "Silu",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "gelu": "Gelu",
    "exp": "Exp",
    "sqrt": "Sqrt",
    "rsqrt": "Rsqrt",
    "reciprocal": "Reciprocal",
}

_POOL_NAMES = {
    "max": "MaxPool",
    "average": "AveragePool",
    "global_average": "GlobalAveragePool",
}


def _shape_of(node: FxNode) -> str | None:
    """节点输出的形状，写成 GML 的 `1x64x56x56` 形式。"""
    value = node.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return "x".join(str(int(dim)) for dim in shape)


def _is_emittable(node: FxNode) -> bool:
    """这个 FX 节点是否对应一个 GML 硬件节点。

    折过的 RMSNorm 以 `pow` 为锚点（`graph.fuse_pim` 把整条六算子链折到那里），
    而 `pow` 不在 `OP_TYPES` 里——所以要按 meta 标记额外认它。
    """
    if node.op != "call_function":
        return False
    # 被折进别的节点的算子不单独发射（RMSNorm 链的中间项、折进 contraction
    # 的激活、被吸收的 attention 定标）。它们仍留在图里以保持可执行。
    if node.meta.get(ABSORBED_META_KEY):
        return False
    if RMS_NORM_META_KEY in node.meta:
        return True
    # 逐头展开打了角色标记的节点一律发射：角色决定 op_type，
    # 不要求它的 aten 目标在 OP_TYPES 里。
    if node.meta.get(HEAD_ROLE_META_KEY) in _ROLE_OP_TYPES:
        return True
    # DQ 节点的载体是 `alias`（fx 里的恒等操作），不在 OP_TYPES 里，
    # 靠这个标记认。它在 GML 里是完整的 4 相 DynamicScaling。
    if DQ_META_KEY in node.meta:
        return True
    if ROPE_META_KEY in node.meta:
        return True
    if KV_DMA_META_KEY in node.meta or SPLIT_META_KEY in node.meta:
        return True
    # 无计算的 view/reshape：参考只留两处（拆头前后的语义 reshape）。
    # 喂 RMSNorm / 残差 / 出口的那些跨过去，不单独成节点（评审 4 §3.4）。
    if node.target in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
        return not _reshape_is_layout_only(node)
    return node.target in OP_TYPES


def _tensor_inputs(node: FxNode, emittable: set[FxNode]) -> list[FxNode]:
    """节点上游最近的那些 GML 节点。

    不可映射的算子（`to`、`slice`、`pow`、`embedding` 之类）要**跨过**而不是丢弃：
    它们夹在可映射节点之间，直接断开会把图切成互不相连的碎片，入口/出口就不再唯一。
    所以顺着它们继续往上找，直到碰到可映射节点为止。

    权重、常量、标量不在此列——它们在 GML 里是节点属性或编译期数据，不是边。
    """
    found: list[FxNode] = []
    seen: set[FxNode] = set()

    def walk(current: FxNode) -> None:
        for arg in current.all_input_nodes:
            if arg in seen:
                continue
            seen.add(arg)
            if arg in emittable:
                if arg not in found:
                    found.append(arg)
            elif arg.op == "call_function":
                # 这一层在 GML 里不存在，继续往它的上游找。
                walk(arg)

    walk(node)
    return found


def _reshape_is_layout_only(node: FxNode) -> bool:
    """这个 view/reshape 没有自己的计算，只改形状。

    参考只留两处语义 Reshape：v 拆头前、concat 后。q/k 喂 RoPE 的 view 跨过去。
    """
    if _feeds_meta(node, ROPE_META_KEY):
        return True
    users = [u for u in node.users if getattr(u, "op", None) == "call_function"]
    if not users:
        return True
    for user in users:
        if RMS_NORM_META_KEY in user.meta:
            return True
        if user.target in (torch.ops.aten.add.Tensor,):
            return True
    return False


def _feeds_meta(node: FxNode, key: str, depth: int = 0) -> bool:
    if depth > 8:
        return False
    for user in node.users:
        if not hasattr(user, "meta"):
            continue
        if key in user.meta:
            return True
        if getattr(user, "op", None) == "call_function":
            if _feeds_meta(user, key, depth + 1):
                return True
    return False


def _mask_placeholder_of(mask_node: FxNode) -> FxNode | None:
    """Mask 节点（`add.Tensor(current, mask)`）的第二个操作数最终指向的
    graph `placeholder`。

    实测该操作数是一个 `alias`（fx 的恒等操作），直接包一层 causal mask
    这个 `placeholder`（`runtime/compile.py::PositionalLlama.forward` 的
    第二个入参）。这里只沿着 `call_function` 链跨过恒等/reshape 类算子找
    到那个 `placeholder`，找不到就返回 `None`（例如没有 causal mask 的
    seq_len==1 场景，参考同样没有这条边）。
    """
    args = mask_node.all_input_nodes
    if len(args) < 2:
        return None
    current = args[1]
    seen: set[FxNode] = set()
    while current.op == "call_function" and current not in seen:
        seen.add(current)
        inputs = current.all_input_nodes
        if not inputs:
            return None
        current = inputs[0]
    return current if current.op == "placeholder" else None


def _slot_dims(op_type: str | None, role, slots: CompileSlots,
               fallback: str | None, *, rewrite: bool) -> str:
    """边的 dims 按编译期槽位写，不按导出图 seq_len。

    只在 llama2-7B 图上改写（图里能看到 hidden=4096）。小图测试沿用导出形状。
    """
    H, I, HD, S, nh = (slots.hidden, slots.intermediate, slots.head_dim,
                       slots.seq, slots.heads)
    if rewrite:
        if role == ROLE_MATMUL_QK:
            return f"1x1x1x{S}"
        if role in (ROLE_MASK, ROLE_SOFTMAX):
            return f"1x1x1x{S}"
        if role == ROLE_MATMUL_PV:
            return f"1x1x1x{HD}"
        if op_type == "KV_Cache_DMA":
            return f"1x{nh}x{S}x{HD}"
        if fallback and fallback != "unknown":
            parts = fallback.split("x")
            ints = [int(p) for p in parts if p.isdigit()]
            if ints and ints[-1] in (1, 16) and S not in ints:
                parts[-1] = str(S)
                return "x".join(parts)
    return fallback or "unknown"


# 输出恒为 int8 的算子：它们把数值落成定点（量化 / 写 cache）。
_OUT_INT8_OPS = frozenset({
    "DynamicScaling", "Llama2Activation", "Llama2ActivationDQ",
    "KV_Cache_DMA", "Split",
})
# 输出恒为 fp16 的算子：累加在浮点 FPSU 上，结果不落定点。
_OUT_FP16_OPS = frozenset({
    "MatMul", "Softmax", "Mask", "EltwiseAdd", "EltwiseMul", "RMSNorm_vpu",
})
# 吃定点激活的算子：参考 7 个 Gemm、64 个 MatMul 的 input 全是 int8。
_IN_INT8_OPS = frozenset({"Gemm", "MatMul"})
# 纯布局算子：不改数值，所以 **没有** input_sf / input_zp，dtype 沿边传播。
_NO_SCALE_OPS = frozenset({"Split", "Transpose", "Reshape", "Concat"})
# 顶层不写 input dtype 的算子（只有槽字段，或根本没有输入侧声明）。
_NO_TOP_IN_DTYPE = frozenset({
    "Mask", "EltwiseAdd", "EltwiseMul", "Concat", "KV_Cache_DMA",
})


def _stamp_dtypes(nodes: list[Node]) -> None:
    """补齐节点级 dtype / extension，并删不该有的 sf/zp。

    **布局算子的 dtype 是沿边传播的，不是按 op_type 固定**。参考里同一个
    `Transpose` 既有 fp16 的（node 14，concat 之后那条）也有 int8 的
    （node 27/29/34，KV 与 QK 那条）；`Reshape` 同样两种。按 op_type 写死
    会让一半节点位宽错一倍，而这在我们这侧不会报错。

    其余按参考逐 op 实测（评审 4 §2.5）：

      Gemm        in int8 / out fp16（v_proj 落 cache 走 int8）/ weight int4
      MatMul      in int8（32 个 bmm1 的 Q 来自 RoPE-DQ，也是定点）
      Softmax     in fp16 —— 我方之前只写了 out
      KV_DMA      out int8 + **顶层** input_sf/input_zp（槽字段之外）
      Split       out 恒 int8（3/3），in 传播（fp16 1 个、int8 2 个）
      DQ          自命名输出，参考顶层 input_sf 为 0 个，我方曾多写 36 个

    `nodes` 必须是**拓扑序**（`convert` 按 FX 序 append，生产者在前），
    否则传播拿不到上游已定的 dtype。
    """
    out_dtype: dict[int, str] = {}
    op_of: dict[int, str] = {}

    for gml_node in nodes:
        fields = gml_node.fields
        op_type = fields.get("op_type")
        node_id = gml_node.node_id
        op_of[node_id] = str(op_type or "")

        # 边界缓冲节点自己声明过 dtype，只记下来供下游传播。
        if not op_type:
            declared = fields.get("output_buffer_dtype")
            if isinstance(declared, str):
                out_dtype[node_id] = declared
            continue

        producers = _gml_int_list(gml_node, names.residual_buffer_key("input", 0))
        producers += _gml_int_list(gml_node, names.residual_buffer_key("input", 10))
        # 传播用第一个生产者：布局算子只有一路数据。
        flowing = next((out_dtype[p] for p in producers if p in out_dtype),
                       "float16")

        if op_type in _OUT_INT8_OPS:
            out_dt = "int8"
        elif op_type in _OUT_FP16_OPS:
            out_dt = "float16"
        elif op_type == "Gemm":
            # v_proj 把 value 落成 int8 存进 KV cache，`resolve()` 已按
            # 输出 dtype 选了 kantor_mode，这里复用同一个判据。
            out_dt = ("int8" if fields.get("kantor_mode") == "fp2int_converter"
                      else "float16")
        else:
            out_dt = flowing  # Transpose / Reshape / Concat：原样传播
        out_dtype[node_id] = out_dt

        if "output_buffer_dtype" not in fields:
            fields["output_buffer_dtype"] = out_dt
            fields["output_data_extension"] = hw_table.data_extension(out_dt)

        if op_type in _IN_INT8_OPS:
            in_dt = "int8"
        elif op_type in _NO_TOP_IN_DTYPE:
            in_dt = None
        elif op_type in _NO_SCALE_OPS:
            in_dt = flowing
            # 例外：吃 `Llama2ActivationDQ` 的那个 Split，参考声明 **fp16**
            # （node 21），而该生产者自己的 `output_buffer_dtype` 是 int8。
            # 不是传播能推出来的：这个生产者是「RoPE 3 连 + DQ 4 相」的融合
            # 节点，RoPE 那侧是 fp16、DQ 那侧才是 int8，两种声明各自都讲得通。
            # 这里照抄参考，不自己推导。
            if op_type == "Split" and any(
                    op_of.get(p) == "Llama2ActivationDQ" for p in producers):
                in_dt = "float16"
        else:
            in_dt = "float16"
        # 已有带槽号的 dtype（多输入算子）时不再写顶层那份。
        slotted = any(key.startswith("input_buffer_") and key.endswith("_dtype")
                      for key in fields)
        if in_dt and "input_buffer_dtype" not in fields and not slotted:
            fields["input_buffer_dtype"] = in_dt
            fields["input_data_extensions"] = hw_table.data_extension(in_dt)

        if op_type in _NO_SCALE_OPS or op_type == "DynamicScaling":
            for key in ("input_sf", "input_zp"):
                fields.pop(key, None)

        if op_type == "KV_Cache_DMA":
            # 参考 node 28 顶层同时有 input_sf / input_zp，槽字段之外再来一份。
            fields.setdefault("input_sf", names.scale(node_id))
            fields.setdefault("input_zp", names.zero_point(node_id))
            fields.pop("input_buffer_dtype", None)

        # 权重 dtype 按**操作数来源**盖：get_attr 二维权重是 int4（W4A8），
        # MatMul 的 KV cache 权重通路是 int8（发射时已写死，不要覆盖）。
        if (op_type == "Gemm" and fields.get("weight_buffer")
                and fields.get("pim_weight_param")
                and "weight_buffer_dtype" not in fields):
            fields["weight_buffer_dtype"] = "int4"
            fields.setdefault("weight_sf_dtype", "float16")


def _contraction_of(node: FxNode) -> list[tuple[str, dict[str, object]]]:
    """折进本节点的激活与池化，写成 contraction 块的内容。"""
    tail = node.meta.get(FUSED_TAIL_META_KEY)
    if tail is None:
        return []

    entries: list[tuple[str, dict[str, object]]] = []
    activation = _ACTIVATION_NAMES.get(tail.activation, tail.activation)
    entries.append((
        f"fused_{node.name}_activation",
        {"name": f"{node.name}_activation", "op_type": "Lut",
         "activation_op_type": activation},
    ))
    if tail.pool:
        entries.append((
            f"fused_{node.name}_pool",
            {"name": f"{node.name}_pool",
             "op_type": _POOL_NAMES.get(tail.pool, tail.pool)},
        ))
    return entries


def _weight_param_of(node: FxNode) -> str | None:
    """节点的权重参数名。

    `get_attr` 节点指向模型参数，取第一个二维以上的——一维的是 norm 的缩放或
    偏置，不走 `weight_buffer`。
    """
    for arg in node.all_input_nodes:
        if arg.op != "get_attr":
            continue
        value = arg.meta.get("val")
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2:
            return str(arg.target)
    return None


def _phase_count_for(phase_source, op_type: str, label: str) -> int:
    """一个多相算子发几套 `*_phase_N` 字段。

    **真源是算子编译器**：`phase_source` 非空时按它给的相位数发字段，所以
    FlagTree 的 `-pim-expand-phases` 改了相位结构，GML 的字段套数就跟着变。
    没接上算子编译器时退回 `contracts.gml_quant.PHASE_COUNTS` 静态表。

    `Llama2ActivationDQ` 的 DQ 部分查 `dq`：它是「RoPE 3 连 + DQ 4 相」，
    `*_phase_N` 字段只对应后面那 4 相。
    """
    kind = {
        "DynamicScaling": "dq",
        "Llama2ActivationDQ": "dq",
        "Softmax": "softmax",
        "Llama2Activation": "rope",
    }.get(op_type)
    fallback = PHASE_COUNTS.get(op_type, 0)

    if phase_source is None or kind is None:
        return fallback
    plan = phase_source.plan(label, kind)
    return fallback if plan is None else plan.count


def convert(
    gm: GraphModule, *, version: str = GML_VERSION, phase_source=None,
    decode_block_only: bool = False,
    slots: CompileSlots | None = None,
) -> tuple[list[Node], list[Edge], dict[int, str], dict[int, object]]:
    """把融合后的图转成 GML 节点与边。

    要求 `gm` 已经跑过 `graph.fuse.fuse_graph`：图里若还有独立激活节点，
    GML 无法表达，这里会直接抛。

    `phase_source` 是 `opcompiler_bridge.phase_source.PhaseSource`：**相位数的
    真源**。给了它，每个多相算子发几套 `*_phase_N` 字段就由算子编译器
    （FlagTree 的 `-pim-expand-phases`）决定；没给则退回
    `contracts.gml_quant.PHASE_COUNTS` 静态表。

    两条路径当前产出**相同**的 GML，因为算子编译器算出的相位数与静态表一致
    ——这正是 `phase_source.cross_check()` 在保证的。但依赖是真的：pass 若改了
    相位结构，GML 的字段套数会跟着变。

    `slots` 是编译期槽位。边的 dims、KV cache 边界节点按它写，不按导出图的
    seq_len（prepare_out 的尺寸真源，见 contracts.compile_slots）。
    """
    slots = slots or DEFAULT_SLOTS
    rewrite_dims = any(
        4096 in tuple(int(x) for x in getattr(n.meta.get("val"), "shape", ()) or ())
        for n in gm.graph.nodes)
    emittable = {node for node in gm.graph.nodes if _is_emittable(node)}
    if not emittable:
        raise ValueError("图里没有可映射到 GML 的算子")

    def phase_count(op_type: str, label: str) -> int:
        """这个算子发几套 `*_phase_N` 字段。

        真源是算子编译器；没接上时退回静态表。两者不一致会在
        `phase_source.cross_check()` 里被拦下，所以这里直接用。
        """
        return _phase_count_for(phase_source, op_type, label)

    # 逆拓扑编号：id 越小越靠输出。参考产物就是这个约定，最终输出是 id 2。
    ordered = [node for node in gm.graph.nodes if node in emittable]
    node_ids = {node: len(ordered) - index + 2
                for index, node in enumerate(ordered)}

    consumers: dict[FxNode, list[FxNode]] = {node: [] for node in ordered}
    for node in ordered:
        for source in _tensor_inputs(node, emittable):
            consumers[source].append(node)

    gml_nodes: list[Node] = []
    edges: list[Edge] = []
    # K 路 RoPE 那个节点也要走 DQ 的写盘路径。收在这里而不是写回
    # `node.meta`，序列化才是纯函数（见下面赋值处的说明）。
    extra_dq_specs: dict[int, object] = {}
    # node_id -> FX 参数名。写盘阶段按它从图里取 f32 权重去量化。
    weight_params: dict[int, str] = {}

    # 图的入口与出口要有缓冲区节点：两份参考产物都是这样（ResNet50 两个，
    # llama2 的 decode block 十个），而且算子节点悬空会被结构校验判为违规。
    # 入口是没有可映射上游的算子，出口是没有可映射下游的算子。
    entry_targets = [node for node in ordered
                     if not _tensor_inputs(node, emittable)]
    exit_sources = [node for node in ordered if not consumers[node]]

    # 残差旁路：第一条残差 add 的一路往上追到 embedding 就断了。参考把
    # 这一路接到图入口缓冲（node 1 同时喂 RMSNorm 与第一条残差）。
    # 入口缓冲的 output_buffer 仍按第一个消费者（RMSNorm）命名；残差自己
    # 声明 input_buffer_0，规则 2 只要求生产者的名字被某个读者读到。
    entry_bypass: dict[FxNode, int] = {}
    for node in ordered:
        if _op_type_of(node) != "EltwiseAdd":
            continue
        if _tensor_inputs(node, emittable):
            # 只有一路可映射上游 → 另一路是 embedding 旁路。
            if len(_tensor_inputs(node, emittable)) == 1:
                entry_bypass[node] = 0
            break
    boundary_base = len(ordered) + 3
    entry_ids = {node: boundary_base + index
                 for index, node in enumerate(entry_targets)}
    exit_ids = {node: boundary_base + len(entry_targets) + index
                for index, node in enumerate(exit_sources)}

    # 旁路挂到**第一个**入口缓冲上（参考的 node 1 就是那个）。
    first_entry_id = min(entry_ids.values()) if entry_ids else None
    for node in list(entry_bypass):
        if first_entry_id is None:
            entry_bypass.pop(node)
        else:
            entry_bypass[node] = first_entry_id

    for node, buffer_id in entry_ids.items():
        readers = [node_ids[node]]
        # 这个入口同时喂残差旁路时，多记一个消费者。
        if buffer_id == first_entry_id:
            readers += [node_ids[n] for n in entry_bypass]
        # 缓冲按**第一个**消费者编号（规则 2）。喂残差旁路的入口有两个读者，
        # `output_buffer` 仍指第一个（那个入口算子），旁路那条边由残差侧
        # 自己声明 `input_buffer_0`。
        out_name = names.data_buffer(node_ids[node])
        fields_buf: dict[str, object] = {
            "label": f"in_{node.name}", "name": f"in_{node.name}",
            "is_buffer": 1,
            "output_buffer": out_name,
            # 用列表而不是裸 int：`residual_*_buffer` 是重复键，
            # 统一成列表让下游（校验器、测试）不必兼容两种类型。
            names.residual_buffer_key("output", 0): readers,
        }
        for index, reader_id in enumerate(readers):
            fields_buf[names.port_node_id_key("output", index)] = reader_id
        gml_nodes.append(Node(buffer_id, fields_buf))
        edges.append(Edge(buffer_id, node_ids[node],
                          _shape_of(node) or "unknown"))
        if buffer_id == first_entry_id:
            for bypass_node in entry_bypass:
                edges.append(Edge(buffer_id, node_ids[bypass_node],
                                  _shape_of(bypass_node) or "unknown"))

    for node, buffer_id in exit_ids.items():
        gml_nodes.append(Node(buffer_id, {
            "label": f"out_{node.name}", "name": f"out_{node.name}",
            "is_buffer": 1,
            "input_buffer": names.data_buffer(buffer_id),
            names.residual_buffer_key("input", 0): [node_ids[node]],
            names.port_node_id_key("input", 0): node_ids[node],
            "input_count": 1,
        }))

    # RoPE 的 cos / sin 表：参考把它们建成 `is_buffer` 边界节点，并连边进
    # **两条** RoPE（Q 与 K 共用同一份表）。我方原先把表折进节点字段，于是
    # RoPE 的 `input_count` 记 1 而参考记 3，下游 DQ 的
    # `Residual input buffer 0/1/2` 也就无从生成。
    #
    # 表本身是编译期常量（`get_attr` 或广播产出），不在 `emittable` 里，所以
    # 这里单独建节点，不影响算子节点的逆拓扑编号。
    rope_nodes = [n for n in ordered if ROPE_META_KEY in n.meta]
    rope_tables: dict[FxNode, int] = {}
    # tensor -> 表节点最终声明的共享文件名（K 路锚点）。下面每个 RoPE
    # 消费者自己声明 `input_buffer_<slot>` 时要读这个名字，不能各自按
    # `names.data_buffer(node_id, slot)` 重新拼——否则 Q 消费者拼出的名字
    # 和边界节点实际声明的 `output_buffer` 不是同一个字符串，读者与生产者
    # 就此失去同步（复核 20260921 发现：K 锚点落地后，Q 侧仍各自重新拼，
    # `test_output_buffer_points_at_the_consumer` 因此挂了）。
    rope_table_names: dict[FxNode, str] = {}
    if rope_nodes:
        table_base = boundary_base + len(entry_targets) + len(exit_sources)
        # 按 (cos, sin) 去重：两条 RoPE 指向同一对表节点。
        for rope in rope_nodes:
            match = rope.meta[ROPE_META_KEY]
            for tensor in (match.sin, match.cos):
                if tensor is not None and tensor not in rope_tables:
                    rope_tables[tensor] = table_base + len(rope_tables)
        for tensor, buffer_id in rope_tables.items():
            readers = [n for n in rope_nodes
                       if n.meta[ROPE_META_KEY].cos is tensor
                       or n.meta[ROPE_META_KEY].sin is tensor]
            # 不写 `original_name` / `from_tvm`：那两个是参考走 TVM 路径留下的
            # 溯源字段，`contracts/gml_coverage.py` 明确声明我方不适用。
            fields: dict[str, object] = {
                "label": f"in_{tensor.name}", "name": f"in_{tensor.name}",
                "is_buffer": 1,
            }
            reader_ids = [node_ids[r] for r in readers]
            fields[names.residual_buffer_key("output", 0)] = reader_ids
            for index, reader_id in enumerate(reader_ids):
                fields[names.port_node_id_key("output", index)] = reader_id
            # 缓冲按**K 路**消费者编号，不是"第一个"（复核 20260921
            # 纠正——上一版这里写"按第一个消费者编号"是错的，没有验证过
            # 两条 RoPE 共享表节点时到底按哪个consumer 编号）。实测参考
            # 产物两张表节点（cos=node2、sin=node3）都被 Q（node22）与
            # K（node30）共用，`output0_node_id` 记的是 Q（先出现），但
            # `output_buffer` 两张表都写成 `input_buffer_*_30`——按 K 编号，
            # 不是按"第一个"。K 路固定走 `Llama2ActivationDQ`
            # （`_is_second_rope` 的判据），用它选锚点消费者，Q 也读这个
            # 共享名。找不到 K 路（比如某条 RoPE 没有配对的第二条）时退回
            # 第一个消费者，不留 None。
            # 表占 RoPE 的槎 1（sin）或 2（cos）。取最后一个会让
            # test_output_buffer_points_at_the_consumer 判为悬空。
            slot = 2 if any(r.meta[ROPE_META_KEY].cos is tensor
                            for r in readers) else 1
            anchor = next((r for r in readers if _is_second_rope(r)),
                         readers[0])
            shared_name = names.data_buffer(node_ids[anchor], slot)
            fields["output_buffer"] = shared_name
            rope_table_names[tensor] = shared_name
            gml_nodes.append(Node(buffer_id, fields))
            for reader in readers:
                edges.append(Edge(buffer_id, node_ids[reader],
                                  _shape_of(tensor) or "unknown"))

    # Mask 的第二路输入（causal mask）：图里是一个 `placeholder`
    # （`runtime/compile.py::PositionalLlama.forward` 的第二个入参），经过一个
    # `alias`（fx 的恒等操作）喂给全部 32 个 Mask 节点，共享同一份 placeholder。
    # `_is_emittable`/`_tensor_inputs` 只认 `call_function`，两者都不满足，
    # 这条边原来直接被丢弃——Mask 的 `input_count` 因此少算 1，被误判成单槎
    # 算子，图里也再没有节点声明这个 buffer，写盘阶段自然不会落它（实测参考
    # 产物 32 个头的 `Datain file 1` 都真实存在于 parser_output，不是虚引用）。
    #
    # 处理方式与 RoPE cos/sin 表同构：单独建一个 `is_buffer` 边界节点，不进
    # `_tensor_inputs` 的返回值，不影响算子节点的逆拓扑编号。实测参考产物
    # 32 个头的 `Datain file 1` 全部指向同一个文件名，所以只建**一个**边界
    # 节点，全部 Mask 节点共享（同 RoPE 表「一份表喂两条 RoPE」的处理）。
    #
    # `mask_boundary_of` 只登记**真的**接上这条边的 Mask 节点（找不到
    # placeholder 的退化场景——比如 seq_len==1 没有 causal mask——不登记，
    # 那些节点仍走原来的单槎路径），下面两处都要查它而不是笼统按 op_type 判：
    # `_data_slot_count` 收不到"这个具体节点是否有边界边"这个信息，只能在
    # 这里、在消费者具体节点上做判断。
    mask_nodes = [n for n in ordered
                 if n.meta.get(HEAD_ROLE_META_KEY) == ROLE_MASK]
    mask_boundary_of: dict[FxNode, int] = {}
    mask_buffer_name: str | None = None
    if mask_nodes:
        connected = [n for n in mask_nodes
                    if _mask_placeholder_of(n) is not None]
        if connected:
            mask_placeholder = _mask_placeholder_of(connected[0])
            mask_buffer_id = (boundary_base + len(entry_targets)
                              + len(exit_sources) + len(rope_tables))
            reader_ids = [node_ids[n] for n in connected]
            # 参考产物全部头的 `Datain file 1` 是**同一个文件名**（按第一个
            # 消费者编号，同 RoPE 表的规则 2），不是各自按自己的 node_id 命名。
            mask_buffer_name = names.data_buffer(reader_ids[0], 1)
            fields = {
                "label": f"in_{mask_placeholder.name}",
                "name": f"in_{mask_placeholder.name}",
                "is_buffer": 1,
            }
            fields[names.residual_buffer_key("output", 0)] = reader_ids[:10]
            if reader_ids[10:]:
                fields[names.residual_buffer_key("output", 10)] = (
                    reader_ids[10:])
            for index, reader_id in enumerate(reader_ids):
                fields[names.port_node_id_key("output", index)] = reader_id
            # Mask 的槎 0 是 bmm1 的输出（算子上游），槎 1 留给这块边界缓冲——
            # 与参考产物 `Datain file 1` 的槎位一致。
            fields["output_buffer"] = mask_buffer_name
            gml_nodes.append(Node(mask_buffer_id, fields))
            for reader in connected:
                edges.append(Edge(mask_buffer_id, node_ids[reader],
                                  _shape_of(mask_placeholder) or "unknown"))
                mask_boundary_of[reader] = mask_buffer_id

    # KV cache 初始平面 + 位置下标：参考 3 个 is_buffer（K 4MB、V 4MB、
    # 位置 int16 共享一份）。图是 use_cache=False 导出的，这两个输入不在
    # FX 里，必须按编译期槽位单独建（评审 4 §2.3）。
    kv_cache_of: dict[FxNode, tuple[int, int]] = {}
    kv_nodes = [n for n in ordered if KV_DMA_META_KEY in n.meta]
    if kv_nodes:
        kv_base = (boundary_base + len(entry_targets) + len(exit_sources)
                   + len(rope_tables) + (1 if mask_buffer_name else 0))
        pos_id = kv_base
        pos_readers = [node_ids[n] for n in kv_nodes]
        pos_name = names.data_buffer(pos_readers[0], 1)
        pos_fields: dict[str, object] = {
            "label": "in_kv_position", "name": "in_kv_position",
            "is_buffer": 1,
            "output_buffer": pos_name,
            "output_buffer_dtype": "int16",
            names.residual_buffer_key("output", 0): pos_readers,
        }
        for index, reader_id in enumerate(pos_readers):
            pos_fields[names.port_node_id_key("output", index)] = reader_id
        gml_nodes.append(Node(pos_id, pos_fields))
        pos_dims = (f"1x{slots.heads}x1x3" if rewrite_dims else "1x1x1x3")
        for reader in kv_nodes:
            edges.append(Edge(pos_id, node_ids[reader], pos_dims))
        for index, dma in enumerate(kv_nodes):
            spec = dma.meta[KV_DMA_META_KEY]
            cache_id = kv_base + 1 + index
            reader_id = node_ids[dma]
            cache_name = names.data_buffer(reader_id, 0)
            cache_fields: dict[str, object] = {
                "label": "in_key_cache" if spec.is_key else "in_value_cache",
                "name": "in_key_cache" if spec.is_key else "in_value_cache",
                "is_buffer": 1,
                "output_buffer": cache_name,
                "output_buffer_dtype": "int8",
                names.residual_buffer_key("output", 0): [reader_id],
                names.port_node_id_key("output", 0): reader_id,
            }
            gml_nodes.append(Node(cache_id, cache_fields))
            cache_dims = (f"1x{slots.heads}x{slots.seq}x{slots.head_dim}"
                          if rewrite_dims else "1x1x1x1")
            edges.append(Edge(cache_id, reader_id, cache_dims))
            kv_cache_of[dma] = (cache_id, pos_id)

    # KV 写回出口：参考 key_cache_out / value_cache_out 两个 is_buffer。
    kv_out_of: dict[FxNode, int] = {}
    if kv_nodes:
        out_base = (boundary_base + len(entry_targets) + len(exit_sources)
                    + len(rope_tables) + (1 if mask_buffer_name else 0)
                    + 1 + len(kv_nodes))
        for index, dma in enumerate(kv_nodes):
            spec = dma.meta[KV_DMA_META_KEY]
            out_id = out_base + index
            kv_out_of[dma] = out_id
            dma_id = node_ids[dma]
            label = "key_cache_out" if spec.is_key else "value_cache_out"
            gml_nodes.append(Node(out_id, {
                "label": label, "name": label, "is_buffer": 1,
                "input_buffer": names.data_buffer(out_id),
                "input_buffer_dtype": "int8",
                names.residual_buffer_key("input", 0): [dma_id],
                names.port_node_id_key("input", 0): dma_id,
                "input_count": 1,
            }))
            cache_dims = (f"1x{slots.heads}x{slots.seq}x{slots.head_dim}"
                          if rewrite_dims else "1x1x1x1")
            edges.append(Edge(dma_id, out_id, cache_dims))

    for node in ordered:
        node_id = node_ids[node]
        inputs = _tensor_inputs(node, emittable)

        rms_norm = node.meta.get(RMS_NORM_META_KEY)
        role = node.meta.get(HEAD_ROLE_META_KEY)
        dq_spec = node.meta.get(DQ_META_KEY)

        # 吃 DQ 输出的节点，输入 dtype 跟着上游变成 int8。
        # 实测 DQ 12 -> Gemm 11：DQ 声明 `output_buffer_dtype int8`，
        # 消费者也声明 `input_buffer_dtype int8`。dtype 是**沿边传播**的，
        # 不是每个节点独立猜 —— 写死 fp16 会让缓冲宽度差一倍。
        #
        # 这里只**收集**，到节点收尾时再统一写。曾经直接写在
        # `if downstream:` 块内的 output_buffer 之后，结果把后面写
        # residual_output_buffer / outputN 的代码一起吞进了循环体，
        # 三份连接信息不再同步，结构自检报「仅在 edge」。
        upstream_int8 = {}
        for slot_index, source in enumerate(inputs):
            if DQ_META_KEY not in getattr(source, 'meta', {}):
                continue
            key = ('input_buffer_%d_dtype' % slot_index
                   if _data_slot_count(_op_type_of(node), len(inputs)) > 1
                   else 'input_buffer_dtype')
            upstream_int8[key] = 'int8'

        # 逐头展开产出的节点按**角色**定 op_type，不按 aten 目标：
        # 逐头的 `add` 是 Mask 而不是 EltwiseAdd，`matmul` 是 MatMul 的两种角色。
        op_type = _op_type_of(node)

        fields: dict[str, object] = {
            "label": node.name,
            "name": node.name,
            "op_type": op_type,
            # 内部键（`pim_` 前缀不进 GML 文本）：label 会被改成参考风格的
            # 语义名，写盘与编排器仍需按 FX 名反查，所以原名留在这里。
            "pim_fx_name": node.name,
        }

        # 两个 attention matmul 的硬件配置不同，实测 32/32 各自一致：
        #   matmul1 (QKᵀ) 吃 K 的转置 -> weights_transpose，定标 1/√head_dim
        #   matmul2 (PV)  直接吃 V    -> weight，定标 1.0
        # 这是**数学决定的**（哪个是 QKᵀ 图编译器完全知道），不是排布优化。
        if role in (ROLE_MATMUL_QK, ROLE_MATMUL_PV):
            fields.update(hw_table.top_level_fields(
                "MatMul", transposed=role == ROLE_MATMUL_QK))
            head = node.meta.get(HEAD_INDEX_META_KEY)
            if role == ROLE_MATMUL_QK and head is not None:
                fields["split_channel_number"] = head
        elif role == ROLE_MASK:
            fields.update(hw_table.top_level_fields("Mask"))
        elif role == ROLE_SOFTMAX:
            fields.update(hw_table.top_level_fields("Softmax"))
            # Softmax 五相：字段族同 DynamicScaling 的处理（见下面 dq_spec
            # 分支），只是这里没有 dq_spec 那样的图侧标记——role 本身就是
            # 判据。写盘侧 `write_softmax_phases`（runtime_files.py）已经
            # 按同一套 `names.phase_*` 函数把 bin 落盘，这里补上 GML 文本里
            # 该有的引用字段，否则那些 bin 从 GML 的角度看永远是悬空的
            # （之前就是这样：write_softmax_phases 写了但 GML 没声明任何
            # `_phase_` 字段去引用它们，交叉校验判成「写了盘但没引用」）。
            #
            # **5 入 + 5 出对称结构**（复核 20260921 纠正：上一版这里写
            # "phase1 没有 output_buffer_phase_1"，那是只看
            # `prepare_out/txt_files` 层卡字段推出来的，没有去对参考 GML
            # 文本本身——实测参考 `relay2gml_graph.gml` 的 node 18 逐相都
            # 声明了 `input_buffer_phase_N` 与 `output_buffer_phase_N`
            # （N=0..4），`parser_output` 也确实有全部 10 个文件。字段集合
            # 必须跟 `write_softmax_phases` 逐一对上，那边现在是每相都写
            # input + output，这里也每相都声明。LUT 只在 phase1（exp 表
            # 占位）、phase3（倒数表）写。
            for phase in range(phase_count("Softmax", node.name)):
                fields.update(hw_table.phase_fields("Softmax", phase))
                fields[f"input_buffer_phase_{phase}"] = (
                    names.phase_input_buffer(node_id, phase))
                fields[f"output_buffer_phase_{phase}"] = (
                    names.phase_output_buffer(node_id, phase))
                fields[f"Scaling_buffer_phase_{phase}"] = (
                    names.phase_fpsu_scale(node_id, phase))
                fields[f"Scaling_PS_buffer_phase_{phase}"] = (
                    names.phase_fpsu_post_shift(node_id, phase))
                fields[f"Bias_buffer_phase_{phase}"] = (
                    names.phase_fpsu_bias(node_id, phase))
            fields["LUT_phase_1"] = names.phase_lut(node_id, 1)
            fields["LUT_phase_3"] = names.phase_lut(node_id, 3)


        # 输入缓冲区按本节点编号——本节点是这些数据的消费者。多输入算子带槽位号。
        # `residual_input_buffer` 在 GML 里是重复键（每个输入一条），所以用列表：
        # writer 会把列表展开成多行同名字段。
        # MatMul 的第二个 operand 走**权重通路**，不占数据输入槽——所以它只有
        # 一个数据槽，键名不带槽号（实测 64/64 个 MatMul 都是 `input_buffer`）。
        # 端口 `inputN_node_id` 仍然两个都要，因为边确实是两条。
        data_slots = _data_slot_count(fields.get("op_type"), len(inputs))
        multi = data_slots > 1
        upstream_ids: list[int] = []
        for slot, source in enumerate(inputs):
            if slot < data_slots:
                slot_index = slot if multi else None
                # 上游是 phase 型节点（DQ）时，缓冲与 scale 都**引用生产者的
                # 文件**，不由消费者另起名字：DQ 自命名 output_buffer_<self>，
                # 而它 phase1 的输出就是这一路的 scale（p1 = p0/256）。
                # 实测 MatMul 16 -> input_buffer "output_buffer_17.bin"、
                # input_sf "output_buffer_phase_1_17.bin"（17 是上游 DQ）。
                # zp 例外，始终由消费者命名（input_zp_16.bin）。
                if DQ_META_KEY in getattr(source, "meta", {}):
                    buffer_name = names.phase_output_buffer_self(
                        node_ids[source])
                    scale_name = names.phase_output_buffer(node_ids[source], 1)
                else:
                    buffer_name = names.data_buffer(node_id, slot_index)
                    scale_name = names.scale(node_id, slot_index)
                fields[f"input_buffer_{slot}" if multi else "input_buffer"] = (
                    buffer_name)
                layout = op_type in ("Split", "Transpose", "Reshape", "Concat")
                if not layout:
                    fields[f"input_{slot}_sf" if multi else "input_sf"] = scale_name
                    fields[f"input_{slot}_zp" if multi else "input_zp"] = (
                        names.zero_point(node_id, slot_index))
            fields[names.port_node_id_key("input", slot)] = node_ids[source]
            upstream_ids.append(node_ids[source])
        if inputs:
            # 同样分两组写：端口 >= 10 的键名多一个下划线。
            if upstream_ids[:10]:
                fields[names.residual_buffer_key("input", 0)] = upstream_ids[:10]
            if upstream_ids[10:]:
                fields[names.residual_buffer_key("input", 10)] = upstream_ids[10:]

            # MatMul 的第二个 operand 走**权重通路**、不占输入槽，所以
            # `input_count` 比实际入边少记 1（实测参考产物 64 个 MatMul 全如此，
            # 且 Σ input_count + MatMul 数 == 边数 这条恒等式依赖它）。
            if fields["op_type"] == "MatMul":
                fields["MatMul_input_as_weight"] = 1
                fields["input_count"] = max(1, len(inputs) - 1)
                # 第二个 operand 走权重通路：参考 64 个 MatMul 都声明
                # weight_buffer / weight_sf / weight_zp（int8，尺寸 S×hd）。
                # 这些不是 get_attr 二维权重，`_weight_param_of` 抓不到。
                fields["weight_buffer"] = names.weight_buffer(node_id)
                fields["weight_buffer_dtype"] = "int8"
                fields["weight_sf"] = names.weight_scale(node_id)
                fields["weight_sf_dtype"] = "float16"
                fields["weight_zp"] = names.weight_zero_point(node_id)
            else:
                fields["input_count"] = len(inputs)

            # 残差旁路：那一路上游是 embedding（不可发射），边接在图入口缓冲上。
            # 参考的第一条残差 `input_count` 是 2，两个槽都声明 `input_buffer_N`。
            if node in entry_bypass:
                bypass_id = entry_bypass[node]
                fields.pop("input_buffer", None)
                fields.pop("input_sf", None)
                fields.pop("input_zp", None)
                # 旁路占槽 0（参考 input0_node_id 指入口缓冲），算子上游占槽 1。
                fields["input_buffer_0"] = names.data_buffer(node_id, 0)
                fields["input_buffer_0_dtype"] = "float16"
                fields["input_0_sf"] = names.scale(node_id, 0)
                fields["input_0_zp"] = names.zero_point(node_id, 0)
                fields["input_buffer_1"] = names.data_buffer(node_id, 1)
                fields["input_buffer_1_dtype"] = "float16"
                fields["input_1_sf"] = names.scale(node_id, 1)
                fields["input_1_zp"] = names.zero_point(node_id, 1)
                fields[names.port_node_id_key("input", 0)] = bypass_id
                fields[names.port_node_id_key("input", 1)] = upstream_ids[0]
                fields[names.residual_buffer_key("input", 0)] = (
                    [bypass_id] + upstream_ids)
                fields["input_count"] = len(inputs) + 1

            # RoPE 还要把 cos / sin 两张表的边界节点记成输入边：参考的
            # `input_count` 是 3（源 + sin + cos），下游 DQ 的
            # `Residual input buffer 0/1/2` 就是这三条。表节点在上面已建好。
            if ROPE_META_KEY in node.meta and rope_tables:
                match = node.meta[ROPE_META_KEY]
                table_tensors = [t for t in (match.sin, match.cos)
                                 if t in rope_tables]
                table_ids = [rope_tables[t] for t in table_tensors]
                if table_ids:
                    # 变成多槎：源占槎 0，sin/cos 占槎 1/2。三个槎都要声明
                    # `input_buffer_N`（参考如此），写盘侧按这些字段落 bin ——
                    # 只在表节点写 output_buffer 会让那两个 bin 没人写。
                    fields.pop("input_buffer", None)
                    fields.pop("input_sf", None)
                    fields.pop("input_zp", None)
                    fields["input_buffer_0"] = names.data_buffer(node_id, 0)
                    fields["input_buffer_0_dtype"] = "float16"
                    fields["input_0_sf"] = names.scale(node_id, 0)
                    fields["input_0_zp"] = names.zero_point(node_id, 0)
                    slot = len(inputs)
                    for tensor, table_id in zip(table_tensors, table_ids):
                        fields[names.port_node_id_key("input", slot)] = table_id
                        # 读表节点已经声明的共享名（`rope_table_names`），
                        # 不能各自按 `names.data_buffer(node_id, slot)`
                        # 重新拼——K 锚点落地后，Q 消费者自己拼出来的名字
                        # 和边界节点实际声明的 `output_buffer` 不再是同一个
                        # 字符串，读者与生产者就此失去同步（复核 20260921
                        # 发现的回归，`test_output_buffer_points_at_
                        # the_consumer` 会挂）。
                        fields[f"input_buffer_{slot}"] = rope_table_names.get(
                            tensor, names.data_buffer(node_id, slot))
                        fields[f"input_buffer_{slot}_dtype"] = "float16"
                        fields[f"input_{slot}_sf"] = names.scale(node_id, slot)
                        fields[f"input_{slot}_zp"] = names.zero_point(
                            node_id, slot)
                        slot += 1
                    fields[names.residual_buffer_key("input", 0)] = (
                        upstream_ids + table_ids)
                    fields["input_count"] = len(inputs) + len(table_ids)


            # Mask 还要把 causal mask 边界节点记成第二路输入：参考的
            # `input_count` 是 2（bmm1 输出 + causal mask），`Datain file 1`/
            # `Residual input buffer 1` 就是这条边。边界节点在上面已建好。
            # 槎 1 的缓冲名**不**按本节点编号——全部头共享同一个文件名
            # （`mask_buffer_name`，按第一个消费者编号，规则同 RoPE 表）。
            if node in mask_boundary_of and mask_buffer_name is not None:
                table_id = mask_boundary_of[node]
                fields.pop("input_buffer", None)
                fields.pop("input_sf", None)
                fields.pop("input_zp", None)
                fields["input_buffer_0"] = names.data_buffer(node_id, 0)
                fields["input_buffer_0_dtype"] = "float16"
                fields["input_0_sf"] = names.scale(node_id, 0)
                fields["input_0_zp"] = names.zero_point(node_id, 0)
                fields["input_buffer_1"] = mask_buffer_name
                fields["input_buffer_1_dtype"] = "float16"
                fields["input_1_sf"] = names.scale(node_id, 1)
                fields["input_1_zp"] = names.zero_point(node_id, 1)
                fields[names.port_node_id_key("input", 1)] = table_id
                fields[names.residual_buffer_key("input", 0)] = (
                    upstream_ids + [table_id])
                fields["input_count"] = len(inputs) + 1

        elif node in entry_ids:
            # 入口算子的上游是入口缓冲节点，字段照常写——否则那个缓冲节点的
            # output_buffer 找不到读者，规则 2 会判为悬空引用。
            fields["input_buffer"] = names.data_buffer(node_id)
            fields["input_sf"] = names.scale(node_id)
            fields[names.residual_buffer_key("input", 0)] = [entry_ids[node]]
            fields[names.port_node_id_key("input", 0)] = entry_ids[node]
            fields["input_count"] = 1

        # 输出缓冲区按**消费者**编号。多个消费者时记第一个，其余靠
        # residual_output_buffer 列出——参考产物就是这样。
        downstream = consumers[node]
        if not downstream and node in exit_ids:
            # 末端算子的输出流向出口缓冲节点，按消费者（即该缓冲节点）编号。
            fields["output_buffer"] = names.data_buffer(exit_ids[node])
            fields[names.residual_buffer_key("output", 0)] = [exit_ids[node]]
            fields[names.port_node_id_key("output", 0)] = exit_ids[node]
            edges.append(Edge(node_id, exit_ids[node],
                              _shape_of(node) or "unknown"))
        if downstream:
            first = downstream[0]
            first_inputs = _tensor_inputs(first, emittable)
            # 槽号按**消费者**的数据槽规则算，不是它的入边数 ——
            # 见 _data_slot_count 的说明。
            first_slots = _data_slot_count(
                _op_type_of(first), len(first_inputs))
            # 消费者是接了 causal mask 边界节点的 Mask：它是双槎算子，但
            # `_data_slot_count` 判不出来（只有具体节点知道自己是否接了那条
            # 边，见 `mask_boundary_of` 的说明），这里按消费者节点本身查。
            if first in mask_boundary_of:
                first_slots = 2
            index = first_inputs.index(node)
            slot = index if first_slots > 1 else None
            # 消费者是 KV_Cache_DMA：三个槽是 (cache, 索引, 新值)，
            # 真正的数据生产者只占槽 2。不顺移的话生产者写无槽名 /
            # 槽 0，而消费者声明 `input_buffer_2`，两边悬空。
            if _op_type_of(first) == "KV_Cache_DMA":
                slot = 2
                first_slots = 3
            # 消费者是带残差旁路的 eltwise：它有两个数据槽，算子上游占槽 1
            # （槽 0 留给来自图入口缓冲的旁路）。不顺移的话生产者写无槽名、
            # 消费者只声明 `input_buffer_1`，两边悬空。
            if first in entry_bypass:
                slot = index + 1
                first_slots = max(first_slots, 2)
            # phase 型节点（带 rtl_version 的 DQ）**自命名** output_buffer，
            # 其余按消费者编号。实测 37 个自命名节点与 37 个 rtl_version 完全重合：
            #   DynamicScaling 36 + Llama2ActivationDQ 1 -> output_buffer_<self>
            #   其余 152 个算子节点 -> input_buffer_<消费者>
            # 判据就是「有没有 rtl_version」，不需要额外规则。
            if (dq_spec is not None
                    or op_type in ("DynamicScaling", "Llama2ActivationDQ")):
                # phase 型节点自命名（见 phase_output_buffer_self）。
                # 不能只看 dq_spec：Llama2ActivationDQ 的标记是在字段
                # 发射时才打上的，那时 output_buffer 已经写过了。
                fields["output_buffer"] = (
                    names.phase_output_buffer_self(node_id))
            elif index >= first_slots:
                # 这一路流进消费者的**权重通路**（MatMul 的第二个
                # operand），不占数据槽，所以按 `weight_buffer_<消费者>`
                # 命名而不是 `input_buffer_<消费者>`。
                # 实测 Split 32 -> MatMul 16：Split 的 output_buffer
                # 是 weight_buffer_187.bin。写成 input_buffer 会悬空 ——
                # 消费者那侧根本没声明这个槽。
                fields["output_buffer"] = names.weight_buffer(
                    node_ids[first])
            else:
                fields["output_buffer"] = names.data_buffer(
                    node_ids[first], slot)

            # `residual_output_buffer` 与端口字段必须成对出现——三份连接信息
            # （edge / outputN / residual_*）要同步。原实现只写了 outputN，
            # 漏了 residual 这一份，对方按它推依赖时会缺边。
            #
            # 端口号 >= 10 时键名多一个下划线，所以分两组写（见 gml_names）。
            plain = [node_ids[c] for c in downstream[:10]]
            suffixed = [node_ids[c] for c in downstream[10:]]
            if plain:
                fields[names.residual_buffer_key("output", 0)] = plain
            if suffixed:
                fields[names.residual_buffer_key("output", 10)] = suffixed
            for index, consumer in enumerate(downstream):
                fields[names.port_node_id_key("output", index)] = \
                    node_ids[consumer]

        if node in kv_out_of:
            out_id = kv_out_of[node]
            outs = fields.get(names.residual_buffer_key("output", 0)) or []
            if not isinstance(outs, list):
                outs = [outs]
            if out_id not in outs:
                outs = list(outs) + [out_id]
                fields[names.residual_buffer_key("output", 0)] = outs
                fields[names.port_node_id_key("output", len(outs) - 1)] = out_id

        # 权重按**本节点**编号——它属于节点自己，不属于某条边（规则 2）。
        # 名字记在 `weight_param` 里（不是 GML 字段，只给写盘用），让写盘阶段
        # 能从 FX 图取到对应的 f32 张量。
        # output_sf / output_zp：做计算的算子带，纯布局的不带（见 _OUTPUT_SCALE_OPS）。
        # DQ 与 RMSNorm 在各自分支里已经写过，这里跳过避免覆盖它们的 dtype。
        if (op_type in _OUTPUT_SCALE_OPS
                and dq_spec is None and rms_norm is None):
            fields["output_sf"] = names.output_scale(node_id)
            fields["output_sf_dtype"] = "float16"
            fields["output_zp"] = names.output_zero_point(node_id)

        # FPSU 定标三族。实测哪些算子带它：
        #   Gemm 7、MatMul 64、KV_Cache_DMA 2（各 1 组）
        #   EltwiseAdd 2、EltwiseMul 1（各 2 组，逐槽）
        # 三个文件的宽度各不相同（fp16 / u8 / fp32），对应硬件 FPSU 的三个操作数。
        if op_type in _FPSU_OPS:
            # attention 定标（1/√head_dim）由 fuse_pim / split_heads 记在 meta 上。
            # 用一个**非 GML 字段**把它传给写盘阶段：名字以 pim_ 开头，
            # `_referenced_buffers` 与序列化器都只认 .bin 结尾的值，所以
            # 它既不会被当成文件名、也不会写进 GML 文本。
            scale = node.meta.get(ATTENTION_SCALE_META_KEY)
            if scale is not None:
                fields["pim_attention_scale"] = float(scale)

            n_fpsu = _FPSU_SLOTS.get(op_type, 1)
            for index in range(n_fpsu):
                suffix = index if n_fpsu > 1 else None
                fields[f"Scaling_buffer_file_{index}" if suffix is not None
                       else "Scaling_buffer_file"] = names.fpsu_scale(
                           node_id, suffix)
                fields[f"Bias_buffer_file_{index}" if suffix is not None
                       else "Bias_buffer_file"] = names.fpsu_bias(
                           node_id, suffix)
                fields[f"Scaling_PS_buffer_file_{index}" if suffix is not None
                       else "Scaling_PS_buffer_file"] = names.fpsu_post_shift(
                           node_id, suffix)

        weight_param = _weight_param_of(node)
        if weight_param:
            fields["weight_buffer"] = names.weight_buffer(node_id)
            fields["weight_sf"] = names.weight_scale(node_id)
            fields["weight_zp"] = names.weight_zero_point(node_id)
            # 内部键，不进 GML 文本。编排器用它区分 q/k/v/o/gate/up/down。
            fields["pim_weight_param"] = weight_param

        # DynamicScaling：4 相字段族 + 每相的缓冲与定标系数。
        # phase 在 GML 里**不是独立节点**，而是同一节点内的 *_phase_<k> 字段。
        if dq_spec is not None:
            fields.update(hw_table.top_level_fields("DynamicScaling"))
            fields["use_dynamic_quantization"] = 1
            # 节点级字段（不带 _phase_ 后缀）。dtype 说的是**整个节点**的
            # 输入输出：吃 fp16、吐 int8，与 p3 那一相一致。
            fields["input_buffer_dtype"] = "float16"
            fields["input_data_extensions"] = hw_table.data_extension("float16")
            fields["output_buffer_dtype"] = "int8"
            fields["output_data_extension"] = hw_table.data_extension("int8")
            fields["rtl_version"] = hw_table.RTL_VERSION
            fields["transpose"] = 1
            # 两个形状字段。`original_shape` 是被量化张量的形状，
            # `output_shape_by_group` 把最后一维按 group_size 拆成
            # `[..., groups, group_size]` —— 硬件按这个分组求 absmax。
            # 必须按真实张量算，不能写死（实测 4096 与 11008 两种）。
            shape = _shape_tuple_of(node.args[0]) if node.args else None
            if shape is not None:
                fields["original_shape"] = _shape_literal(shape)
                fields["output_shape_by_group"] = _shape_literal(
                    tuple(shape[:-1]) + (dq_spec.groups, dq_spec.group_size))
            # 相位数由算子编译器给出（没接上时退回静态表）。
            for phase in range(phase_count("DynamicScaling", node.name)):
                fields.update(hw_table.phase_fields(
                    "DynamicScaling", phase, group_size=dq_spec.group_size))
                fields[f"input_buffer_phase_{phase}"] = (
                    names.phase_input_buffer(node_id, phase))
                fields[f"output_buffer_phase_{phase}"] = (
                    names.phase_output_buffer(node_id, phase))
                fields[f"Scaling_buffer_phase_{phase}"] = (
                    names.phase_fpsu_scale(node_id, phase))
                fields[f"Scaling_PS_buffer_phase_{phase}"] = (
                    names.phase_fpsu_post_shift(node_id, phase))
                fields[f"Bias_buffer_phase_{phase}"] = (
                    names.phase_fpsu_bias(node_id, phase))
            # p1 走恒等表、p2 走倒数表（两张都自行合成，见 contracts/gml_lut）。
            fields["LUT_phase_1"] = names.phase_lut(node_id, 1)
            fields["LUT_phase_2"] = names.phase_lut(node_id, 2)
            # p3 由 Kantor 做浮点转定点，这是量化真正落定的一相。
            fields["kantor_A_scale_buffer_file_phase_3"] = (
                names.phase_kantor_scale(node_id, 3))
            fields["kantor_A_bias_buffer_file_phase_3"] = (
                names.phase_kantor_bias(node_id, 3))
            fields["kantor_A_Shift_buffer_file_phase_3"] = (
                names.phase_kantor_shift(node_id, 3))
            # 输出的 requant scale 逐字节等于 phase1 的输出。
            fields["output_sf"] = names.output_scale(node_id)
            fields["output_sf_dtype"] = "float16"
            fields["output_zp"] = names.output_zero_point(node_id)

        # KV_Cache_DMA：定点通路 + updates 那一路的量化参数。
        kv_dma = node.meta.get(KV_DMA_META_KEY)
        if kv_dma is not None:
            fields.update(hw_table.top_level_fields("KV_Cache_DMA"))
            fields["updates_sf"] = names.kv_updates_scale(node_id)
            fields["updates_sf_dtype"] = "float16"
            fields["updates_zp"] = names.kv_updates_zero_point(node_id)
            # 三个输入槽：0=cache 平面、1=位置下标（int16，L2A 忽略）、2=新值。
            # 参考 node 28：`input_buffer_0/1/2` 全声明，txt 的
            # `Original cache file` 取槽 0。单数 `input_buffer` 是错的。
            fields.pop("input_buffer", None)
            fields.pop("input_sf", None)
            fields.pop("input_zp", None)
            fields["input_buffer_0"] = names.data_buffer(node_id, 0)
            fields["input_buffer_0_dtype"] = "int8"
            fields["input_0_sf"] = names.scale(node_id, 0)
            fields["input_0_zp"] = names.zero_point(node_id, 0)
            fields["input_buffer_1"] = names.data_buffer(node_id, 1)
            fields["input_buffer_1_dtype"] = "int16"
            fields["use_input_buffer_1"] = "L2A_ignore"
            fields["input_buffer_2"] = names.data_buffer(node_id, 2)
            fields["input_buffer_2_dtype"] = "int8"
            fields["input_2_sf"] = names.scale(node_id, 2)
            fields["input_2_zp"] = names.zero_point(node_id, 2)
            fields["pim_kv_is_key"] = 1 if kv_dma.is_key else 0
            # 三槽真实入边：cache 平面、位置下标、新值（评审 4 §2.3）。
            if node in kv_cache_of:
                cache_id, pos_id = kv_cache_of[node]
                new_id = upstream_ids[0] if upstream_ids else node_id
                fields[names.port_node_id_key("input", 0)] = cache_id
                fields[names.port_node_id_key("input", 1)] = pos_id
                fields[names.port_node_id_key("input", 2)] = new_id
                fields[names.residual_buffer_key("input", 0)] = [
                    cache_id, pos_id, new_id]
                fields["input_count"] = 3

        # Split：一进多出，`num_heads` 记输出个数。
        split = node.meta.get(SPLIT_META_KEY)
        if split is not None:
            fields["num_heads"] = str(split.heads)
            fields["axis"] = 1
            fields["kantor_mode"] = "off"
            fields["use_dynamic_quantization"] = 1

        # RoPE 折叠节点：一整套子块配置 + 每个子块的定标/零点/Kantor 文件。
        rope = node.meta.get(ROPE_META_KEY)
        if rope is not None:
            is_dq = op_type == "Llama2ActivationDQ"
            fields.update(hw_table.rope_fields(dq=is_dq))
            fields["num_heads"] = str(_head_count_of(node))
            if is_dq:
                fields["rtl_version"] = hw_table.RTL_VERSION
                fields["dq_contraction"] = 1
                fields["transpose"] = 1
                # 同样由算子编译器给相位数（查 dq：`*_phase_N` 只对应后 4 相）。
                for phase in range(phase_count("Llama2ActivationDQ", node.name)):
                    fields.update(hw_table.phase_fields(
                        "Llama2ActivationDQ", phase, group_size=128))
                    fields[f"input_buffer_phase_{phase}"] = (
                        names.phase_input_buffer(node_id, phase))
                    fields[f"output_buffer_phase_{phase}"] = (
                        names.phase_output_buffer(node_id, phase))
                    fields[f"Scaling_buffer_phase_{phase}"] = (
                        names.phase_fpsu_scale(node_id, phase))
                    fields[f"Scaling_PS_buffer_phase_{phase}"] = (
                        names.phase_fpsu_post_shift(node_id, phase))
                    fields[f"Bias_buffer_phase_{phase}"] = (
                        names.phase_fpsu_bias(node_id, phase))
                fields["LUT_phase_1"] = names.phase_lut(node_id, 1)
                fields["LUT_phase_2"] = names.phase_lut(node_id, 2)
                fields["kantor_A_scale_buffer_file_phase_3"] = (
                    names.phase_kantor_scale(node_id, 3))
                fields["kantor_A_bias_buffer_file_phase_3"] = (
                    names.phase_kantor_bias(node_id, 3))
                fields["kantor_A_Shift_buffer_file_phase_3"] = (
                    names.phase_kantor_shift(node_id, 3))
                # 复用已有的 DQ 写盘路径：这个节点也要 write_dq_phases。
                #
                # **不写 `node.meta`**：那会让 `convert()` 改图，同一份图序列化
                # 两次结果就不同（`_dq_specs` 第二次会多认出这个节点，RoPE 的
                # `output_sf` 等字段跟着变）。收集到局部 dict 里返回，序列化就是
                # 纯函数。
                from graph.quant_pass import DynamicScalingSpec
                numel = slots.hidden if rewrite_dims else (_numel_of_fx(node) or 128)
                extra_dq_specs[node_id] = DynamicScalingSpec(
                    group_size=128, numel=numel,
                    is_attention_scores=False)
            for unit, block in hw_table.ROPE_UNITS:
                fields[f"Scaling_buffer_file_{unit}_{block}"] = (
                    names.rope_scale(node_id, unit, block))
                fields[f"Scaling_PS_buffer_file_{unit}_{block}"] = (
                    names.rope_post_shift(node_id, unit, block))
            # 每个子块的输出量化参数（含广播那一路）。
            for block in hw_table.ROPE_SCALE_BLOCKS:
                fields[f"{block}_sf"] = names.rope_quant_scale(node_id, block)
                fields[f"{block}_zp"] = (
                    names.rope_quant_zero_point(node_id, block))
            # Kantor：cos/sin 各有 A/B 两侧，末段加法只有 A 侧。
            for block in hw_table.ROPE_KANTOR_BLOCKS:
                for side in ("A", "B"):
                    fields[f"Kantor_{side}_{block}_bias_buffer_file"] = (
                        names.rope_kantor_bias(node_id, block, side))
                    fields[f"Kantor_{side}_Shift_{block}"] = (
                        names.rope_kantor_shift(node_id, block, side))
                # A 侧的 scale 由 B 侧那份承担，实测只有 B 有 scale_buffer。
                fields[f"Kantor_B_{block}_scale_buffer_file"] = (
                    names.rope_kantor_scale(node_id, block, "B"))
            for key, fn in (("bias_buffer_file", names.rope_kantor_bias),
                            ("scale_buffer_file", names.rope_kantor_scale)):
                fields[f"Kantor_A_Llama2Activation_add_{key}"] = fn(
                    node_id, "Llama2Activation_add", "A")
            fields["Kantor_A_Shift_Llama2Activation_add"] = (
                names.rope_kantor_shift(node_id, "Llama2Activation_add", "A"))
            # 两个中间态：x*cos 与 rotate_half(x)*sin。
            fields["cos_mul_output"] = names.rope_intermediate(node_id, "cos")
            fields["sin_mul_output"] = names.rope_intermediate(node_id, "sin")

        # RMSNorm 绑在向量单元上，配置走 vpu_params 子块，还带 eps 常量。
        # 三处 sf 都是 **fp32**（实测 2 个 RMSNorm 节点全是 float32，
        # 其余算子是 float16）——这是唯一的 dtype 例外。
        nested: dict[str, dict[str, object]] = {}
        if rms_norm is not None:
            # 用 `target`（带点的参数路径）而不是 `name`（下划线化的节点名）——
            # 写盘时要靠它逐段 getattr 取出真实张量。
            weight_name = (
                str(rms_norm.weight_node.target) if rms_norm.weight_node else None)
            if weight_name:
                # 一维缩放张量走 weight_buffer，per-tensor 量化。
                fields["weight_buffer_dtype"] = "int8"
                fields["weight_buffer"] = names.weight_buffer(node_id)
                fields["weight_sf"] = names.weight_scale(node_id)
                fields["weight_zp"] = names.weight_zero_point(node_id)
                weight_params[node_id] = weight_name

            fields["input_sf_dtype"] = "float32"
            fields["weight_sf_dtype"] = "float32"
            fields["output_sf_dtype"] = "float32"
            fields["output_zp"] = names.output_zero_point(node_id)
            fields["RMSNorm_Add_Const"] = names.rms_norm_epsilon(node_id)
            fields["Use_Scaling"] = 0
            nested["vpu_params"] = {
                # -1 表示沿最后一维归约，实测如此。
                "Vpu_Axis": -1,
                "input_scale_factor_buffer": names.scale(node_id),
                "output_scale_factor_buffer": names.output_scale(node_id),
                "Weights_buffer_file": names.weight_buffer(node_id),
                "weights_scaling_buffer_file": names.weight_scale(node_id),
                "bias_buffer_file": names.rms_norm_epsilon(node_id),
            }

        if op_type == "Gemm":
            is_v = (weight_param is not None and "v_proj" in weight_param) or any(
                (consumers[node] and KV_DMA_META_KEY in c.meta
                 and not c.meta[KV_DMA_META_KEY].is_key)
                for c in consumers[node])
            fields.update(hw_table.top_level_fields(
                "Gemm", output_dtype="int8" if is_v else "float16"))
            if is_v:
                fields["kantor_A_spc"] = 1
                fields["kantor_A_scale_axis"] = 1
                fields["kantor_A_spg"] = 0
                fields["kantor_A_scale_buffer_file"] = names.kantor_scale(node_id)
                fields["kantor_A_bias_buffer_file"] = names.kantor_bias(node_id)
                fields["kantor_A_Shift"] = names.kantor_shift(node_id)
            if any(c[-1].get("activation_op_type") == "Silu"
                   for c in _contraction_of(node) if isinstance(c[-1], dict)):
                fields["activation_lut_file"] = names.activation_lut(node_id)
                fields["flp_min_exp"] = 10
                fields["flp_max_exp"] = 17
                fields["flp_mantisa"] = 3
        if op_type == "EltwiseMul":
            fields.update(hw_table.top_level_fields("EltwiseMul"))
            fields["kantor_A_bias_buffer_file"] = names.kantor_bias(node_id, "A")
            fields["kantor_A_Shift"] = names.kantor_shift(node_id, "A")
            fields["kantor_B_scale_buffer_file"] = names.kantor_scale(node_id, "B")
            fields["kantor_B_bias_buffer_file"] = names.kantor_bias(node_id, "B")
            fields["kantor_B_Shift"] = names.kantor_shift(node_id, "B")
        if op_type == "EltwiseAdd":
            fields.update(hw_table.top_level_fields("EltwiseAdd"))

        contraction = _contraction_of(node)
        semantic = _semantic_gml_name(op_type, node, node_id, weight_param, role)
        if semantic:
            fields["label"] = semantic
            fields["name"] = semantic
        gml_node = Node(node_id, fields, contraction, nested)
        if weight_param:
            weight_params[node_id] = weight_param
        # 上游是 DQ 的输入槽，dtype 覆盖成 int8（放最后，压过默认的 fp16）。
        if upstream_int8:
            gml_node.fields.update(upstream_int8)
            gml_node.fields['input_data_extensions'] = (
                hw_table.data_extension('int8'))
        gml_nodes.append(gml_node)

        # 边带形状，节点不带。dims 按编译期槽位，不按导出图 seq_len。
        shape = _slot_dims(op_type, role, slots, _shape_of(node),
                           rewrite=rewrite_dims)
        for consumer in downstream:
            edges.append(Edge(node_id, node_ids[consumer], shape or "unknown"))

    # dtype 收尾：布局算子要沿边传播，所以必须等全部节点发射完再统一盖章。
    _stamp_dtypes(gml_nodes)

    if decode_block_only:
        gml_nodes, edges = _trim_decode_block(gml_nodes, edges)

    return gml_nodes, edges, weight_params, extra_dq_specs


def _trim_decode_block(nodes: list[Node], edges: list[Edge]):
    """丢掉模型末尾 final RMSNorm + lm_head + 它前面的 DQ，对齐参考 decode block。

    判据用结构，不靠 FX 名字符串（评审 3 §2.2）：
    - 末尾 Gemm 权重不是 q/k/v/o/gate/up/down（即 lm_head）
    - 喂它的 DynamicScaling
    - 没有任何 Gemm/MatMul 消费者的 RMSNorm（final norm）
    以及只连向这些节点的边界缓冲。
    """
    by_id = {n.node_id: n for n in nodes}
    drop: set[int] = set()
    gemms = [n for n in nodes if n.fields.get("op_type") == "Gemm"]
    for n in gemms:
        param = str(n.fields.get("pim_weight_param") or "")
        if any(role in param for role in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")):
            continue
        # 没有权重参数的小图 Gemm（测试夹具）不是 lm_head，不要裁。
        if not param:
            continue
        drop.add(n.node_id)
        for src in _gml_int_list(n, "residual_input_buffer"):
            parent = by_id.get(src)
            if parent is not None and parent.fields.get("op_type") == "DynamicScaling":
                drop.add(parent.node_id)
    rms = [n for n in nodes if n.fields.get("op_type") == "RMSNorm_vpu"]
    for n in rms:
        outs = _gml_int_list(n, "residual_output_buffer")
        # final norm 只连向被丢掉的 lm_head 链（或出口缓冲），没有
        # Gemm/MatMul/DQ/残差这些块内消费者。
        keepers = {"Gemm", "MatMul", "DynamicScaling", "EltwiseAdd",
                   "EltwiseMul", "Llama2Activation", "Llama2ActivationDQ"}
        if not any((by_id.get(c) is not None
                    and c not in drop
                    and by_id[c].fields.get("op_type") in keepers)
                   for c in outs):
            drop.add(n.node_id)
    # 喂被丢掉节点、自己没有其它读者的边界缓冲 / Reshape。
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.node_id in drop:
                continue
            outs = _gml_int_list(n, "residual_output_buffer")
            if not outs:
                continue
            if n.fields.get("is_buffer") or n.fields.get("op_type") in (
                    "Reshape", "Transpose"):
                if all(c in drop for c in outs):
                    drop.add(n.node_id)
                    changed = True
    kept = [n for n in nodes if n.node_id not in drop]
    kept_ids = {n.node_id for n in kept}
    kept_edges = [e for e in edges if e.source in kept_ids and e.target in kept_ids]
    # 被丢掉的消费者还在残留节点的 output_buffer / residual_output 里。
    # 参考 decode block 把块出口接到一个 is_buffer 节点上，这里同样补一个，
    # 否则残差的 `output_buffer "input_buffer_<已删>.bin"` 变成悬空引用。
    next_id = max((n.node_id for n in kept), default=2) + 1
    extra_nodes: list[Node] = []
    extra_edges: list[Edge] = []
    for n in kept:
        outs = _gml_int_list(n, "residual_output_buffer")
        alive = [c for c in outs if c in kept_ids]
        if alive == outs:
            continue
        if not alive:
            exit_id = next_id
            next_id += 1
            extra_nodes.append(Node(exit_id, {
                "label": f"out_{n.fields.get('label', n.node_id)}",
                "name": f"out_{n.fields.get('label', n.node_id)}",
                "is_buffer": 1,
                "input_buffer": names.data_buffer(exit_id),
                "input_sf": names.scale(exit_id),
                "input_zp": names.zero_point(exit_id),
                names.residual_buffer_key("input", 0): [n.node_id],
                names.port_node_id_key("input", 0): n.node_id,
                "input_count": 1,
            }))
            n.fields["output_buffer"] = names.data_buffer(exit_id)
            n.fields[names.residual_buffer_key("output", 0)] = [exit_id]
            n.fields[names.port_node_id_key("output", 0)] = exit_id
            extra_edges.append(Edge(n.node_id, exit_id, "unknown"))
            kept_ids.add(exit_id)
        else:
            n.fields[names.residual_buffer_key("output", 0)] = alive
            for index, cid in enumerate(alive):
                n.fields[names.port_node_id_key("output", index)] = cid
    result = kept + extra_nodes
    result_edges = kept_edges + extra_edges
    targets = {e.target for e in result_edges}
    # lm_head 裁掉后，它的出口缓冲没有入边，丢掉。
    result = [n for n in result
              if not (n.fields.get("is_buffer") and n.node_id not in targets
                      and str(n.fields.get("label", "")).startswith("out_"))]
    kept_ids = {n.node_id for n in result}
    result_edges = [e for e in result_edges
                    if e.source in kept_ids and e.target in kept_ids]
    return result, result_edges


def _gml_int_list(node: Node, key: str) -> list[int]:
    value = node.fields.get(key, [])
    if not isinstance(value, list):
        value = [] if value in (None, "") else [value]
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _semantic_gml_name(op_type, fx_node, node_id, weight_param, role) -> str | None:
    """参考风格的 GML label：`<角色>_qidx{fx拓扑号}_params_{node_id}`。

    参考产物的 `params_N` 等于 GML `node_id`；`qidx` 在 TVM 路径是 Relay 号。
    我们没有 Relay，qidx 用 FX 出现序（同一张图稳定），params 用 node_id。
    逐头节点已经叫 `mha_*_headN`，保持不动（layer_expand 靠这个认头）。
    """
    if role in _ROLE_OP_TYPES:
        return None
    qidx = _fx_qidx(fx_node)
    if op_type == "RMSNorm_vpu":
        return f"RMSNorm_params_{node_id}"
    if op_type == "Gemm" and weight_param:
        for role_name, front in (
            ("q_proj", "self_attn_q_proj_MatMul"),
            ("k_proj", "self_attn_k_proj_MatMul"),
            ("v_proj", "self_attn_v_proj_MatMul"),
            ("o_proj", "self_attn_o_proj_MatMul"),
            ("gate_proj", "mlp_gate_proj_MatMul"),
            ("up_proj", "mlp_up_proj_MatMul"),
            ("down_proj", "mlp_down_proj_MatMul"),
        ):
            if role_name in weight_param:
                return f"{front}_qidx{qidx}_params_{node_id}"
    if op_type == "Llama2ActivationDQ":
        return f"self_attn_Reshape_qidx{qidx}_params_{node_id}"
    if op_type == "Llama2Activation":
        return f"self_attn_Reshape_1_qidx{qidx}_params_{node_id}"
    if op_type == "EltwiseAdd":
        return f"{fx_node.name}_Add_qidx{qidx}_params_{node_id}"
    if op_type == "EltwiseMul":
        return f"mlp_mul_Mul_qidx{qidx}_params_{node_id}"
    if op_type == "DynamicScaling":
        return f"dynamic_quantization_params_{node_id}"
    return None


def _fx_qidx(fx_node) -> int:
    """FX 图里可发射节点的出现序，同一张图稳定。"""
    graph = fx_node.graph
    index = 0
    for n in graph.nodes:
        if n.op != "call_function":
            continue
        index += 1
        if n is fx_node:
            return index
    return 0
