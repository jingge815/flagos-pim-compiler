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
from contracts.gml_quant import ACTIVATION_LAYOUT, GML_VERSION, PHASE_COUNTS, QuantLayout
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
    # 类型转换是一次真实的数据运动：输入输出位宽不同，必须有节点承载
    # `input_data_extensions` / `output_data_extension`，不能被跨过。
    torch.ops.aten.to.dtype: "Convert",
    torch.ops.aten.to.dtype_layout: "Convert",
    # 词嵌入查表。全模型导出的图入口是 input_ids，嵌入在块内，必须发这个节点；
    # decode 块以隐藏态为入口、嵌入在块外，那条导出在 `convert` 里按
    # `decode_block_only` 裁掉（见那里的说明），所以同一个算子两种导出各发各的。
    torch.ops.aten.embedding.default: "Gather",
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


def _cast_dtypes(node: FxNode) -> tuple[torch.dtype, torch.dtype] | None:
    """`to.dtype` 两侧的元素类型（源, 目标）；任何一侧读不到就返回 None。"""
    src = node.args[0].meta.get("val") if node.args else None
    dst = node.meta.get("val")
    if src is None or dst is None:
        return None
    return src.dtype, dst.dtype


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
    # 类型转换只在位宽真的变了时才成节点；同 dtype 的 `to` 是恒等，跨过。
    if node.target in (torch.ops.aten.to.dtype,
                       torch.ops.aten.to.dtype_layout):
        pair = _cast_dtypes(node)
        if pair is None or pair[0] == pair[1]:
            return False
    return node.target in OP_TYPES


# 明确不发 GML 节点、但有据可查的算子。这不是守卫，是记账：谁被跨过、
# 为什么，下一个人能查到，不必从 `OP_TYPES` 的缺失里反推。
# 词嵌入**不在**这里：它在全模型导出里是要发的节点，只有 decode 块那条导出
# 不发（图入口是隐藏态，嵌入在块外），那个条件写在 `convert` 里。
_WALK_THROUGH = frozenset({
    # 同 dtype 的类型转换是恒等，`_is_emittable` 判为不发射；位宽真变了才成
    # `Convert` 节点。两个重载是同一件事。
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
})


def _tensor_inputs(node: FxNode, emittable: set[FxNode]) -> list[FxNode]:
    """节点上游最近的那些 GML 节点。

    `_WALK_THROUGH` 里的算子要**跨过**而不是丢弃：它们夹在可映射节点之间，
    直接断开会把图切成互不相连的碎片。顺着它们继续往上找，直到可映射节点。
    名单之外的不可映射算子直接报错。

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


def _boundary_dims(node: FxNode, slots: CompileSlots, rewrite: bool,
                   export_seq: int | None) -> str:
    """边界缓冲节点出边的形状，与算子边同一套槽位口径。

    这些边喂的是同一批消费者，所以入口侧写导出图的 seq_len、算子侧写编译期槽位
    就成了同一条数据通路上两个不一致的声明——下游按哪一个分配缓冲都是错的。
    """
    dims = _require_shape(node)
    return _decode_dims(dims, export_seq, slots) if rewrite else dims


def _require_shape(node: FxNode) -> str:
    """节点输出形状，取不到就抛。

    边的形状是可知的（FX 节点带 `meta["val"]`）。写 `"unknown"` 兜底会让写盘侧
    按 1 个元素分配缓冲，而声明与文件仍然自洽——校验器比的就是这两者，
    于是这个错永远不会被发现。
    """
    shape = _shape_of(node)
    if not shape:
        raise ValueError(
            f"FX 节点 {node.name} 没有 meta['val'].shape，算不出边的 dims。"
            f"上游没标注形状就是上游的错，这里不替它编一个")
    return shape


def _export_seq_len(gm: GraphModule) -> int | None:
    """导出图那次用的 seq_len，取自 `input_ids` 占位符的末维。

    它是判断「哪一维是序列轴」的真源。拿不到就返回 None，`_slot_dims`
    随之不改写——宁可留着导出形状让参考对比发现，也不按位置猜一维。
    """
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        shape = tuple(getattr(node.meta.get("val"), "shape", ()) or ())
        if len(shape) == 2:
            return int(shape[-1])
    return None



def _split_out_dims(producer: FxNode, consumer: FxNode, emittable: set,
                    slots: CompileSlots) -> str:
    """Split 一条出边的形状：看它喂的是 Q、Kᵀ 还是 V。

    三个 Split（Q/K/V）在 GML 里都叫 Split，只能靠消费者的角色和槽位区分。
    Q 是 QKᵀ 的左操作数，K 是右操作数（走权重通路，形状是转置后的），
    V 是 PV 的右操作数。
    """
    HD, S = slots.head_dim, slots.seq
    role = consumer.meta.get(HEAD_ROLE_META_KEY)
    inputs = _tensor_inputs(consumer, emittable)
    left = bool(inputs) and inputs[0] is producer
    if role == ROLE_MATMUL_QK:
        return f"1x1x1x{HD}" if left else f"1x1x{HD}x{S}"
    if role == ROLE_MATMUL_PV:
        return f"1x1x1x{HD}" if left else f"1x1x{S}x{HD}"
    return f"1x1x1x{HD}"


def _slot_dims(op_type: str | None, role, slots: CompileSlots,
               fallback: str | None, *, rewrite: bool,
               export_seq: int | None = None) -> str:
    """边的 dims 按 decode 参考口径写，不按导出图 seq_len。

    只在 llama2-7B 图上改写（图里能看到 hidden=4096）。小图测试沿用导出形状。

    decode 参考里 **token 轴恒为 1**，1024 只属于 KV 长度轴：hidden 是
    `1x1x1x4096`，单头 Q 是 `1x1x1x128`，K/V cache 才是 `1x32x1024x128`。
    把导出 seq_len 换成 `slots.seq` 会让激活边按 1024 个 token 分配缓冲
    （实测 `input_buffer_*` 一族 270MB vs 参考 21.6MB）。
    """
    HD, S, nh = slots.head_dim, slots.seq, slots.heads
    if rewrite:
        if role == ROLE_MATMUL_QK:
            return f"1x1x1x{S}"
        if role in (ROLE_MASK, ROLE_SOFTMAX):
            return f"1x1x1x{S}"
        if role == ROLE_MATMUL_PV:
            return f"1x1x1x{HD}"
        if op_type == "KV_Cache_DMA":
            return f"1x{nh}x{S}x{HD}"
        # K 路 RoPE 参考是 `1x1x32x128`（头轴在倒数第二），不是 `1x32x1x128`。
        if op_type == "Llama2Activation":
            return f"1x1x{nh}x{HD}"
        if fallback:
            return _decode_dims(fallback, export_seq, slots)
    if not fallback:
        raise ValueError(
            f"边的 dims 算不出来（op_type={op_type!r} role={role!r}）。"
            f"形状是可知的，写 unknown 只会让下游按 1 个元素分配缓冲")
    return fallback


def _decode_dims(dims: str, export_seq: int | None,
                 slots: CompileSlots) -> str:
    """把导出图的 token 轴换成 1，再左补 1 到四维。

    候选是「值等于导出 seq_len」的那些维。光靠这个值有时分不出来：
    `1x32x128x128` 在 seq_len=128 时末维也是 128，而末维是 head_dim。
    所以先按**已知轴**排除——末维若等于 head_dim / hidden / intermediate 就不是
    token 轴，下标 1 若等于 heads 也不是。排除后仍剩多于一处就只做补维：
    换错一维得到的是形状对、语义错的声明，不如留着让参考对比发现。

    补四维是参考口径：`1x4096` / `1x1x4096` 都写成 `1x1x1x4096`。
    """
    parts = [int(p) for p in dims.split("x")]
    if export_seq is not None:
        known_last = {slots.head_dim, slots.hidden, slots.intermediate}
        hits = [
            i for i, value in enumerate(parts)
            if value == export_seq
            and not (i == len(parts) - 1 and value in known_last)
            and not (i == 1 and len(parts) == 4 and value == slots.heads)
        ]
        if len(hits) == 1:
            parts[hits[0]] = 1
    while len(parts) < 4:
        parts.insert(0, 1)
    return "x".join(str(p) for p in parts)


def _rewrite_seq_axis(dims: str, export_seq: int | None,
                      slots: CompileSlots) -> str:
    """兼容旧名字：decode 参考把 token 轴写成 1，不是 `slots.seq`。"""
    return _decode_dims(dims, export_seq, slots)


# 动态量化算子：输入 fp16、输出定点，四种量化参数由它们自己算。
#   DynamicScaling 36 个（attn 分数 32 + hidden/MLP）+ Q 路 RoPE 那一个。
_DQ_OPS = frozenset({"DynamicScaling", "Llama2ActivationDQ"})
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
# 顶层不带输入定标的算子——只有纯布局的两个。参考实测：Transpose 4/4、
# Reshape 2/2 都不带 `input_sf`；而 Split 吃 DQ 的那一路带、
# Concat 的 32 槽逐槽带，所以它们不能整族豁免。
_NO_TOP_IN_SCALE = frozenset({"Transpose", "Reshape"})
# 顶层不写 input dtype 的算子（只有槽字段，或根本没有输入侧声明）。
_NO_TOP_IN_DTYPE = frozenset({
    "Mask", "EltwiseAdd", "EltwiseMul", "Concat", "KV_Cache_DMA",
})

# 产物能落盘的缓冲元素类型。`int64` 这类索引/主机侧类型不在其中——声明了
# 没有宽度可核文件大小（`verify_gml_artifact` 的宽度表只认这四种）。
_BUFFER_DTYPES = frozenset({"int8", "int16", "float16", "float32"})

# 不逐槽声明输入定标的算子。参考实测：Mask / KV_Cache_DMA / 两个 RoPE 锚点
# 都**没有** `input_<槽>_sf` / `_zp` / `_sf_dtype`——Mask 读的是浮点分数与
# 掩码、自己产出量化结果，KV 与 RoPE 的入槽是 fp16 直通。**多输入不等于
# 逐槽量化**：槽位字段的判据是「这一路的输入是不是量化张量」，不是输入个数。
_NO_INPUT_SLOT_SCALE = frozenset({
    "Mask", "KV_Cache_DMA", "Llama2Activation", "Llama2ActivationDQ",
})

# 参考只写定标文件名、不带 dtype 兄弟字段的算子。实测：
#   Split 3/3、Mask 32/32、EltwiseAdd 2/2 的 input_sf / output_sf 都没有
#   `*_sf_dtype`；DynamicScaling 反过来——`output_sf_dtype` 有、
#   `input_sf_dtype` 没有。多写一个 dtype 不是「更完整」，是两份产物对不上。
_NO_INPUT_SF_DTYPE = frozenset({
    "DynamicScaling", "EltwiseAdd", "Mask", "Split",
})
_NO_OUTPUT_SF_DTYPE = frozenset({"EltwiseAdd", "Mask", "Split"})

# 带 `kantor_mode` 的算子。参考实测：Split / Concat / Reshape / Transpose
# 这四个布局类算子也带，值恒为 "off"——它们落到 Kantor 单元上做直通，
# 字段是给下游看「这一路不做定点转换」。
_LAYOUT_KANTOR_OPS = frozenset({"Split", "Concat", "Reshape", "Transpose"})


def _transpose_axes_of(node: FxNode) -> tuple[int, ...] | None:
    """从 FX 节点读出 Transpose / permute 的轴序。

    `aten.transpose.int` 是两维对换，展开成完整排列；`aten.permute` 直接给轴序。
    取不到就不发，宁缺勿猜。
    """
    target = node.target
    args = node.args
    if target == torch.ops.aten.permute.default and len(args) >= 2:
        perm = args[1]
        if isinstance(perm, (list, tuple)) and all(isinstance(i, int) for i in perm):
            return tuple(perm)
        return None
    if target == torch.ops.aten.transpose.int and len(args) >= 3:
        dim0, dim1 = args[1], args[2]
        rank_src = _shape_tuple_of(args[0]) if args else None
        rank = len(rank_src) if rank_src is not None else 4
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            return None
        axes = list(range(rank))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return tuple(axes)
    return None


def _stamp_idx(nodes: list[Node]) -> None:
    """给每个有输出的节点补 `idx`：它的输出挂在消费者的第几个输入端口。

    参考产物里 197 个节点各带且仅带一个 `idx`，规则是
    `consumer.input<idx>_node_id == self.node_id`（197/197 成立）。它不是
    「第几个输入」——每个节点只有一个，取值 0..31 是 head 序号，
    `input{i}_node_id` 不编码任何 head 信息。
    """
    port_of: dict[int, int] = {}
    for node in nodes:
        for index in range(64):
            producer = node.fields.get(f"input{index}_node_id")
            if isinstance(producer, int):
                port_of.setdefault(producer, index)

    for node in nodes:
        if node.node_id in port_of:
            node.fields["idx"] = port_of[node.node_id]


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

        # 转换节点的两侧位宽由图侧元素类型定，发射时已经写好（见 `convert` 里
        # `op_type == "Convert"` 那段）。这里只把目标位宽记进传播表供下游用，
        # 其余推导全部跳过——它是**唯一**按自己改位宽的算子，传播会抹平它。
        if op_type == "Convert":
            out_dtype[node_id] = str(fields.get("output_buffer_dtype") or flowing)
            continue

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
        # 数据扩展位与 input_buffer_dtype 独立：Mask / Concat / Eltwise /
        # KV_Cache_DMA 顶层不写 input dtype，但仍声明扩展位。
        ext_dt = in_dt
        if op_type in _NO_TOP_IN_DTYPE:
            ext_dt = "int8" if op_type == "KV_Cache_DMA" else (flowing or "float16")
        if ext_dt and "input_data_extensions" not in fields:
            fields["input_data_extensions"] = hw_table.data_extension(ext_dt)

        # 顶层不写 `input_sf` / `input_zp` 的算子。**不含 Split / Concat**：
        # 参考给 Split 写了那一路的输入定标、给 Concat 逐槽写了 32 组，
        # 整族豁免会把它们一起抹掉（`_NO_SCALE_OPS` 管的是 dtype 沿边传播，
        # 是另一件事，不要混用）。
        if op_type in _NO_TOP_IN_SCALE or op_type == "DynamicScaling":
            for key in ("input_sf", "input_zp", "input_sf_dtype"):
                fields.pop(key, None)

        if op_type == "KV_Cache_DMA":
            # 参考 node 28 顶层同时有 input_sf / input_zp，槽字段之外再来一份。
            fields.setdefault("input_sf", names.scale(node_id))
            fields.setdefault("input_zp", names.zero_point(node_id))
            fields.setdefault("input_sf_dtype", "float16")
            fields.pop("input_buffer_dtype", None)

        # 权重 dtype 按**操作数来源**盖：get_attr 二维权重是 int4（W4A8），
        # MatMul 的 KV cache 权重通路是 int8（发射时已写死，不要覆盖）。
        if (op_type == "Gemm" and fields.get("weight_buffer")
                and fields.get("pim_weight_param")
                and "weight_buffer_dtype" not in fields):
            fields["weight_buffer_dtype"] = "int4"
            fields.setdefault("weight_sf_dtype", "float16")


def _contraction_of(
    node: FxNode, node_id: int | None = None
) -> list[tuple[str, dict[str, object]]]:
    """折进本节点的激活与池化，写成 contraction 块的内容。

    `node_id` 用来填 `residual_input_buffer`——**本节点自己的编号**（实测参考
    产物节点 195 的块里就是 195）。折进来的激活读的是主算子的累加结果，那块缓冲
    属于主算子，所以指回自己。取不到编号时不发这一项，宁缺勿错。
    """
    tail = node.meta.get(FUSED_TAIL_META_KEY)
    if tail is None:
        return []

    entries: list[tuple[str, dict[str, object]]] = []
    activation = _ACTIVATION_NAMES.get(tail.activation, tail.activation)
    # 块名与 `name` 都按**激活**命名，不按 FX 节点名。实测参考产物是
    # `fused_Silu_act` / `name "Silu_act"`（节点 195），而按节点名会写成
    # `fused_linear_4_activation`——同一个融合，对方解析器按块名找不到。
    # FlagTree 侧 `#pim.contraction<form = named, blockName = ...>` 是同一份口径。
    block: dict[str, object] = {
        "name": f"{activation}_act", "op_type": "Lut",
        "activation_op_type": activation,
    }
    if node_id is not None:
        block["residual_input_buffer"] = node_id
    entries.append((f"fused_{activation}_act", block))
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


# 相位上的硬件域：ODS 属性名 -> GML 字段的词干。
# 值优先取算子编译器给的（`#pim.phase_spec` 之外的那些 spec 属性），取不到才
# 用常量表——表的定位是缺省值，不是真源。
_PHASE_ODS_FIELDS = (
    ("flp_min", "flp_min_exp"),
    ("flp_max", "flp_max_exp"),
    ("flp_mantisa", "flp_mantisa"),
    ("activation_mode", "activation_mode"),
    ("kantor_mode", "kantor_mode"),
    ("fpsu_mode", "fpsu_mode"),
    ("transpose_type", "transpose_type"),
)

# 卡值 -> GML 文本里的模式名。`phase_plan` 存的是**卡值**（编排器的 txt 用
# 卡值），而 GML 文本写模式名，所以这里映射回去。值与
# `phase_plan._FPSU_CARD`、FlagTree 的 `TTPIM_FpsuMode` / `TTPIM_KantorMode`
# 枚举逐项对应，改一处必须改三处。
_FPSU_MODE_NAME = {1: "floating_point", 2: "floating_point_32"}
_KANTOR_MODE_NAME = {
    0: "off",
    1: "elementwise_mul_fp16",
    2: "float_elt_wise_and_scale",
    3: "fp2int_converter",
    4: "elementwise_mul_fixed_point",
    5: "scalar",
}
# 存卡值、要映回名字的两个域；其余域两边都是整数，直接用。
_CARD_FIELDS = {
    "fpsu_mode": _FPSU_MODE_NAME,
    "kantor_mode": _KANTOR_MODE_NAME,
}

# GML op_type -> 算子编译器那边的 kind。
_PHASE_KIND = {
    "DynamicScaling": "dq",
    "Llama2ActivationDQ": "dq",
    "Softmax": "softmax",
    "Llama2Activation": "rope",
}


def _resolve_rtl_version(declared: str | None) -> str:
    """定版 `rtl_version`：接了算子编译器就以 IR 的模块属性为真源。

    两处各写一份版本号、谁都不核对，改一边另一边不会报错——所以这里要么
    用 IR 声明的值，要么在两者不一致时直接抛。
    """
    if declared is None:
        return hw_table.RTL_VERSION
    if declared != hw_table.RTL_VERSION:
        raise ValueError(
            f"rtl_version 两侧不一致：IR 模块属性是 {declared!r}，"
            f"本仓常量是 {hw_table.RTL_VERSION!r}")
    return declared


def _overlay_phase_source(fields: dict, phase_source, op_type: str,
                          label: str, phase: int) -> None:
    """把算子编译器给的相位域值盖到常量表算出来的字段上。

    `phase_source` 为空、或那一相没给某个域时保持表值不变——所以没接算子
    编译器时产物逐字节不变。
    """
    kind = _PHASE_KIND.get(op_type)
    if phase_source is None or kind is None:
        return
    for ods_name, stem in _PHASE_ODS_FIELDS:
        value = phase_source.phase_value(label, kind, phase, ods_name)
        if value is None:
            continue
        names = _CARD_FIELDS.get(ods_name)
        if names is not None:
            value = names.get(value)
            if value is None:
                continue
        key = f"{stem}_phase_{phase}"
        if key in fields:
            fields[key] = value


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
    export_seq = _export_seq_len(gm)
    emittable = {node for node in gm.graph.nodes if _is_emittable(node)}
    if decode_block_only:
        # decode 块以隐藏态为图入口、词嵌入在块外，那条导出不发 `Gather`
        # ——参考产物里没有这个节点。**必须在这里裁**，不能留到收尾：
        # 节点编号按 `len(ordered)` 逆拓扑算，少一个节点全图编号都会平移。
        emittable = {
            node for node in emittable
            if node.target is not torch.ops.aten.embedding.default}
    if not emittable:
        raise ValueError("图里没有可映射到 GML 的算子")

    # 版本号定版一次：接了算子编译器就以 IR 的模块属性为真源。
    rtl_version = _resolve_rtl_version(
        getattr(phase_source, "rtl_version", None))

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

    # RoPE 与它前面那个转置**可交换**：旋转只作用在末维（head_dim），把
    # heads 轴与 seq 轴对调的转置移到 RoPE 前面还是后面都是同一个数。
    # 参考产物把转置放在 RoPE **之后**（RoPE 直接吃主算子的输出），所以这里
    # 把两者的入边、出边对调，节点本身与编号都不动。
    rope_transpose_of: dict[FxNode, FxNode] = {}
    transpose_rope_of: dict[FxNode, FxNode] = {}
    for node in ordered:
        if ROPE_META_KEY not in node.meta:
            continue
        source = node.meta[ROPE_META_KEY].source
        if not (source in emittable and source.target in (
                torch.ops.aten.transpose.int, torch.ops.aten.permute.default)):
            continue
        # 只对**写回 KV cache 的那条**（K 路）交换。参考产物里 Q 路的 RoPE
        # 直接喂 Split、中间没有转置，K 路才是「RoPE → 转置 → 写 cache」。
        if not any(KV_DMA_META_KEY in user.meta for user in node.users):
            continue
        rope_transpose_of[node] = source
        transpose_rope_of[source] = node

    # 发射口径的入边与出边表：先把 RoPE 与转置对调，再逐处引用。
    # 入边：RoPE 吃转置的输入，转置吃 RoPE。
    # 出边：转置原来的上游改由 RoPE 承接，RoPE 原来的消费者改由转置承接。
    emitted_in: dict[FxNode, list[FxNode]] = {node: None for node in ordered}
    for node in ordered:
        emitted_in[node] = _tensor_inputs(node, emittable)
    emitted_out: dict[FxNode, list[FxNode]] = {
        node: list(consumers[node]) for node in ordered
    }
    for rope_node, trans_node in rope_transpose_of.items():
        emitted_in[trans_node] = [rope_node]
        emitted_in[rope_node] = _tensor_inputs(trans_node, emittable)
        for producer in _tensor_inputs(trans_node, emittable):
            emitted_out[producer] = [
                rope_node if consumer is trans_node else consumer
                for consumer in emitted_out[producer]
            ]
        # RoPE 原来的消费者改吃转置——它们的 FX 上游仍是 RoPE 那个锚点，
        # 出边对调后必须同步改，否则消费者按 FX 入边找槽号会找不到。
        for consumer in consumers[rope_node]:
            emitted_in[consumer] = [
                trans_node if item is rope_node else item
                for item in emitted_in[consumer]
            ]
        emitted_out[rope_node] = [trans_node]
        emitted_out[trans_node] = list(consumers[rope_node])

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
                          _boundary_dims(node, slots, rewrite_dims, export_seq)))
        if buffer_id == first_entry_id:
            for bypass_node in entry_bypass:
                edges.append(Edge(buffer_id, node_ids[bypass_node],
                                  _boundary_dims(bypass_node, slots,
                                                 rewrite_dims, export_seq)))

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
                                  _boundary_dims(tensor, slots,
                                                 rewrite_dims, export_seq)))

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
                # `is_mask` 挂在**缓冲节点**上，不挂在 32 个 Mask 算子上：
                # 实测参考产物里该标志只有 1 处，就在这个共享边界节点
                # （与 `is_buffer 1` 同节点），扇出给全部 32 个头。它描述的
                # 是「这块缓冲是掩码张量」这个来源事实，不是某个算子的配置。
                "is_mask": 1,
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
            # 掩码按编译期槽位：落盘的是 1 个 token 的 seq 个 fp16
            # （参考的掩码边也是 `1x1x1x1024`），导出图的 16x16 不能拿去写。
            mask_dims = _slot_dims("Mask", ROLE_MASK, slots,
                                   _shape_of(mask_placeholder),
                                   rewrite=rewrite_dims, export_seq=export_seq)
            for reader in connected:
                edges.append(Edge(mask_buffer_id, node_ids[reader], mask_dims))
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
        pos_dims = (f"3x1x{slots.heads}x1" if rewrite_dims else "1x1x1x3")
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
        inputs = emitted_in[node]

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

        # `use_dynamic_quantization 1` 挂在**吃定点激活的矩阵乘**上（`_IN_INT8_OPS`，
        # 正是「输入走动态量化」这层意思）。参考产物实测 72 处 = Gemm 7 +
        # MatMul 64 + 被 DQ 喂的那个 Split 1；DQ 节点自己**没有**这个字段
        # ——它的输入是 fp16、量化是它算的，不是它读的。
        if op_type in _IN_INT8_OPS:
            fields["use_dynamic_quantization"] = 1

        # 两个 attention matmul 的硬件配置不同，实测 32/32 各自一致：
        #   matmul1 (QKᵀ) 吃 K 的转置 -> weights_transpose，定标 1/√head_dim
        #   matmul2 (PV)  直接吃 V    -> weight，定标 1.0
        # 这是**数学决定的**（哪个是 QKᵀ 图编译器完全知道），不是排布优化。
        if role in (ROLE_MATMUL_QK, ROLE_MATMUL_PV):
            fields.update(hw_table.top_level_fields(
                "MatMul", transposed=role == ROLE_MATMUL_QK,
                spec=ACTIVATION_LAYOUT))
            head = node.meta.get(HEAD_INDEX_META_KEY)
            if role == ROLE_MATMUL_QK and head is not None:
                fields["split_channel_number"] = head
        elif role == ROLE_MASK:
            fields.update(hw_table.top_level_fields("Mask"))
        elif role == ROLE_SOFTMAX:
            fields.update(hw_table.top_level_fields("Softmax"))
            # 参考把 Softmax 写成 4 维 `[1,1,1,S]`，归约轴是最后一维。
            fields["axis"] = 3
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
                fields.update(hw_table.phase_fields(
                    "Softmax", phase, spec=ACTIVATION_LAYOUT))
                _overlay_phase_source(
                    fields, phase_source, "Softmax", node.name, phase)
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
                if multi and op_type in _NO_TOP_IN_DTYPE:
                    # 这些算子顶层不写 input dtype，改按槽声明。吃 DQ 输出的
                    # 槽稍后会被 `upstream_int8` 的收集结果覆盖成 int8。
                    fields.setdefault(f"input_buffer_{slot}_dtype", "float16")
                # 纯布局算子（换轴、改形状）不改数值的动态范围，既不带输入
                # 定标也不带输出定标。Split / Concat 不在其列：参考给它们
                # 声明了逐槽定标（Split 是那唯一一路），见各自分支。
                layout = op_type in ("Transpose", "Reshape")
                if not layout and op_type not in _NO_INPUT_SLOT_SCALE:
                    sf_key = f"input_{slot}_sf" if multi else "input_sf"
                    fields[sf_key] = scale_name
                    fields[f"input_{slot}_zp" if multi else "input_zp"] = (
                        names.zero_point(node_id, slot_index))
                    # 参考的 EltwiseAdd 逐槽带 `_sf_dtype`（2/2）。decode 块那条
                    # 产物是与参考对齐的冻结基线、本次不改它（带入口旁路的那个
                    # 由旁路分支手写这两份）；全模型这条路的 add 走通用分支，
                    # 按参考补齐。
                    if (op_type not in _NO_INPUT_SF_DTYPE
                            or (op_type == "EltwiseAdd"
                                and not decode_block_only)):
                        fields.setdefault(
                            f"input_{slot}_sf_dtype" if multi
                            else "input_sf_dtype", "float16")
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
                fields.setdefault("input_0_sf_dtype", "float16")
                fields["input_0_zp"] = names.zero_point(node_id, 0)
                fields["input_buffer_1"] = names.data_buffer(node_id, 1)
                fields["input_buffer_1_dtype"] = "float16"
                fields["input_1_sf"] = names.scale(node_id, 1)
                fields.setdefault("input_1_sf_dtype", "float16")
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
                fields["input_buffer_1"] = mask_buffer_name
                fields["input_buffer_1_dtype"] = "float16"
                fields[names.port_node_id_key("input", 1)] = table_id
                fields[names.residual_buffer_key("input", 0)] = (
                    upstream_ids + [table_id])
                fields["input_count"] = len(inputs) + 1

        elif node in entry_ids:
            # 入口算子的上游是入口缓冲节点，字段照常写——否则那个缓冲节点的
            # output_buffer 找不到读者，规则 2 会判为悬空引用。
            fields["input_buffer"] = names.data_buffer(node_id)
            fields["input_sf"] = names.scale(node_id)
            fields.setdefault("input_sf_dtype", "float16")
            fields[names.residual_buffer_key("input", 0)] = [entry_ids[node]]
            fields[names.port_node_id_key("input", 0)] = entry_ids[node]
            fields["input_count"] = 1

        # 词嵌入查表。**参考产物无此节点，字段名待求证**——这里按方言
        # （`pim.gather`）的两个操作数名发，缓冲名沿用参考的命名规则：表是
        # 编译期常量、走权重通路（与上面通用权重分支写的 `weight_buffer`
        # 是同一个文件），索引就是图入口那一路（token id）。
        if op_type == "Gather":
            fields["table"] = names.weight_buffer(node_id)
            fields["indices"] = fields["input_buffer"]

        # 类型转换：两侧位宽由图侧的元素类型定，**不能按 `_stamp_dtypes` 的
        # 沿边传播写**——它自己就是改位宽的那一步，传播会把两侧写成同一个值。
        # 名字取 FX 的类型原名（`torch.float32` 去掉前缀）。只声明产物认识的
        # 缓冲元素类型（见 `_BUFFER_DTYPES`）；扩展位只覆盖 fp16 与 int8
        # （实测编码表），其余没有编码可写，那一份就不写：宁缺勿猜。
        if op_type == "Convert":
            src, dst = _cast_dtypes(node)
            for key, ext_key, dtype in (
                    ("input_buffer_dtype", "input_data_extensions", src),
                    ("output_buffer_dtype", "output_data_extension", dst)):
                dt_name = str(dtype).removeprefix("torch.")
                if dt_name not in _BUFFER_DTYPES:
                    continue
                fields[key] = dt_name
                if dt_name in hw_table.DATA_EXTENSION:
                    fields[ext_key] = hw_table.data_extension(dt_name)

        # 输出缓冲区按**消费者**编号。多个消费者时记第一个，其余靠
        # residual_output_buffer 列出——参考产物就是这样。
        downstream = emitted_out[node]
        if not downstream and node in exit_ids:
            # 末端算子的输出流向出口缓冲节点，按消费者（即该缓冲节点）编号。
            fields["output_buffer"] = names.data_buffer(exit_ids[node])
            fields[names.residual_buffer_key("output", 0)] = [exit_ids[node]]
            fields[names.port_node_id_key("output", 0)] = exit_ids[node]
            # 出口边与算子边同一套槽位口径：它就是这个末端算子的输出，
            # 写导出图的 seq_len 会让出口缓冲按 16 个 token 分配。
            edges.append(Edge(node_id, exit_ids[node],
                              _boundary_dims(node, slots, rewrite_dims,
                                             export_seq)))
        if downstream:
            first = downstream[0]
            first_inputs = emitted_in[first]
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
            if dq_spec is not None or op_type in _DQ_OPS:
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
            if op_type not in _NO_OUTPUT_SF_DTYPE:
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
            # 次正规保护：参考只在 q_proj（×2）与 v_proj（×4）两处发。
            # 这是参考产物的约定，不是从权重算出来的——实测这两个权重的逐组
            # 缩放都在正规区（最小约 5.7e-4），按数值判定会一个都不发。
            if "q_proj" in weight_param:
                fields["weight_sf_multiplier"] = 2
            elif "v_proj" in weight_param:
                fields["weight_sf_multiplier"] = 4
        # DynamicScaling：4 相字段族 + 每相的缓冲与定标系数。
        # phase 在 GML 里**不是独立节点**，而是同一节点内的 *_phase_<k> 字段。
        # DynamicScaling 节点的输出宽 = 落盘的那一行，按这个口径给出边形状。
        dq_out_shape: str | None = None
        if dq_spec is not None:
            fields.update(hw_table.top_level_fields("DynamicScaling"))
            # 节点级字段（不带 _phase_ 后缀）。dtype 说的是**整个节点**的
            # 输入输出：吃 fp16、吐 int8，与 p3 那一相一致。
            fields["input_buffer_dtype"] = "float16"
            fields["input_data_extensions"] = hw_table.data_extension("float16")
            fields["output_buffer_dtype"] = "int8"
            fields["output_data_extension"] = hw_table.data_extension("int8")
            fields["rtl_version"] = rtl_version
            fields["transpose"] = 1
            # 两个形状字段。`original_shape` 是被量化张量的形状，
            # `output_shape_by_group` 把最后一维按 group_size 拆成
            # `[..., groups, group_size]` —— 硬件按这个分组求 absmax。
            #
            # 两者按**落盘口径**写：DQ 落盘的是**一行**（一个 token），
            # `output_buffer_<self>.bin` 装 numel 个 int8、`output_sf_<self>.bin`
            # 装 numel/group_size 个 fp16。拿导出图的 [1,16,4096] 去写会声明出
            # 512 组，而盘上只有 32 组（16 倍错位）。参考 node 12 就是
            # `[1, 1, 1, 4096]` + `[1, 1, 1, 32, 128]`、output_sf 64 字节；
            # 注意力分数的组宽也要换成整行的 1024（参考 node 17 是
            # `[1, 1, 1, 1, 1024]`，不是导出图那一行的 256）。
            # 小图（hidden != 4096）不改写，两个字段与落盘都用导出形状。
            dq_group = dq_spec.group_size
            if rewrite_dims:
                shape = _shape_tuple_of(node.args[0]) if node.args else None
                if not shape:
                    raise ValueError(
                        f"DQ 节点 {node.name} 取不到被量化张量的形状，"
                        f"dq_layout 不能拿 0 去猜 hidden")
                last = int(shape[-1])
                dq_numel, dq_group = slots.dq_layout(
                    last, is_attention_scores=dq_spec.is_attention_scores,
                    group_size=dq_spec.group_size)
                fields["original_shape"] = _shape_literal((1, 1, 1, dq_numel))
                fields["output_shape_by_group"] = _shape_literal(
                    (1, 1, 1, dq_numel // dq_group, dq_group))
                dq_out_shape = f"1x1x1x{dq_numel}"
            else:
                shape = _shape_tuple_of(node.args[0]) if node.args else None
                if shape is not None:
                    fields["original_shape"] = _shape_literal(shape)
                    fields["output_shape_by_group"] = _shape_literal(
                        tuple(shape[:-1]) + (dq_spec.groups, dq_spec.group_size))
            # 这一路的量化规格：沿最后一维分组，组宽随张量变（attention scores
            # 整条一组）。三族 spc/spg 由它派生，与组宽同源。
            dq_layout = QuantLayout("per_group", group_size=dq_group, axis=-1)
            # 相位数由算子编译器给出（没接上时退回静态表）。
            for phase in range(phase_count("DynamicScaling", node.name)):
                fields.update(hw_table.phase_fields(
                    "DynamicScaling", phase, spec=dq_layout))
                _overlay_phase_source(
                    fields, phase_source, "DynamicScaling", node.name, phase)
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
            fields.update(hw_table.top_level_fields(
                "KV_Cache_DMA", spec=ACTIVATION_LAYOUT))
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
            fields["input_buffer_1"] = names.data_buffer(node_id, 1)
            fields["input_buffer_1_dtype"] = "int16"
            fields["use_input_buffer_1"] = "L2A_ignore"
            fields["input_buffer_2"] = names.data_buffer(node_id, 2)
            fields["input_buffer_2_dtype"] = "int8"
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

        if op_type == "Concat":
            # 32 头沿 heads 维拼接，参考写 axis 1。
            fields["axis"] = 1
        if op_type == "Transpose":
            # llama2 的 K/V 排布交换是 [0, 2, 1, 3]。
            perm = _transpose_axes_of(node)
            if perm is not None:
                # 参考写成带引号的字面量，不是数组字段；序列化器会加引号。
                fields["axes"] = "[" + ", ".join(str(i) for i in perm) + "]"

        # 布局类算子也带 `kantor_mode`（参考恒为 "off"）：它们落到 Kantor
        # 单元上直通，字段说的是「这一路不做定点转换」。Split 在下面自己有
        # 一份，`setdefault` 不覆盖它。
        if op_type in _LAYOUT_KANTOR_OPS:
            fields.setdefault("kantor_mode", "off")

        # Split：一进多出，`num_heads` 记输出个数。
        split = node.meta.get(SPLIT_META_KEY)
        if split is not None:
            fields["num_heads"] = str(split.heads)
            fields["axis"] = 1
            fields["kantor_mode"] = "off"
            # 三个 Split 里只有**输入被 DQ 量化过**的那个带这个字段
            # （参考实测 1 处：吃 Q 路 RoPE-DQ 输出的那个）。判据看直接上游
            # 的 op_type，与 phase 型节点自命名的判据同源。
            if any(_op_type_of(source) in _DQ_OPS for source in inputs):
                fields["use_dynamic_quantization"] = 1
                # 参考只留 `input_sf` 这一份，`input_zp` 不带。
                fields.pop("input_zp", None)
            else:
                # 参考实测：只有吃 DQ 输出的那一个 Split 带 `input_sf`
                # （沿用上游的 scale，槽字段之外的一份）；另外两个不带输入
                # 定标，改为自己发 `output_sf` / `output_zp`。
                fields.pop("input_sf", None)
                fields.pop("input_sf_dtype", None)
                fields.pop("input_zp", None)
                fields["output_sf"] = names.output_scale(node_id)
                fields["output_zp"] = names.output_zero_point(node_id)

        # RoPE 折叠节点：一整套子块配置 + 每个子块的定标/零点/Kantor 文件。
        rope = node.meta.get(ROPE_META_KEY)
        if rope is not None:
            is_dq = op_type == "Llama2ActivationDQ"
            fields.update(hw_table.rope_fields(dq=is_dq))
            fields["num_heads"] = str(_head_count_of(node))
            if is_dq:
                fields["rtl_version"] = rtl_version
                fields["dq_contraction"] = 1
                fields["transpose"] = 1
                # 同样由算子编译器给相位数（查 dq：`*_phase_N` 只对应后 4 相）。
                for phase in range(phase_count("Llama2ActivationDQ", node.name)):
                    fields.update(hw_table.phase_fields(
                        "Llama2ActivationDQ", phase, spec=ACTIVATION_LAYOUT))
                    _overlay_phase_source(
                        fields, phase_source, "Llama2ActivationDQ", node.name, phase)
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
            # 末段加法的 Kantor 配置只挂在 K 路那个 `Llama2Activation` 上。
            # 参考实测：node 30（K 路）带这三项，node 22（Q 路的 DQ）不带，
            # 只留下 `kantor_mode_Llama2Activation_add`。照抄参考。
            if not is_dq:
                for key, fn in (("bias_buffer_file", names.rope_kantor_bias),
                                ("scale_buffer_file", names.rope_kantor_scale)):
                    fields[f"Kantor_A_Llama2Activation_add_{key}"] = fn(
                        node_id, "Llama2Activation_add", "A")
                fields["Kantor_A_Shift_Llama2Activation_add"] = (
                    names.rope_kantor_shift(node_id, "Llama2Activation_add",
                                            "A"))
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
            # `output_sf` 与下面的 `output_scale_factor_buffer` 是同一个文件：
            # 参考两个 RMSNorm_vpu 节点都写了顶层这一份，漏了会让下游按
            # 自己的命名习惯去找一个没人声明的名字。
            fields["output_sf"] = names.output_scale(node_id)
            fields["output_sf_dtype"] = "float32"
            fields["output_zp"] = names.output_zero_point(node_id)
            fields["RMSNorm_Add_Const"] = names.rms_norm_epsilon(node_id)
            # 向量单元参数以算子编译器读回的为准，没接上时退回实测常量。
            vpu_axis = -1
            use_scaling = 0
            if phase_source is not None:
                if (got := phase_source.op_value(
                        node.name, "normalize", "vpu-axis")) is not None:
                    vpu_axis = got
                if (got := phase_source.op_value(
                        node.name, "normalize", "use-scaling")) is not None:
                    use_scaling = got
            fields["Use_Scaling"] = use_scaling
            nested["vpu_params"] = {
                # -1 表示沿最后一维归约，实测如此。
                "Vpu_Axis": vpu_axis,
                "input_scale_factor_buffer": names.scale(node_id),
                "output_scale_factor_buffer": names.output_scale(node_id),
                "Weights_buffer_file": names.weight_buffer(node_id),
                "weights_scaling_buffer_file": names.weight_scale(node_id),
                "bias_buffer_file": names.rms_norm_epsilon(node_id),
            }

        # 参考给矩阵单元的两类算子多写一个 `A`，值逐节点等于 `input0_node_id`
        # （实测 71/71 相等）：同一个操作数，矩阵单元那边按 A 侧称呼。
        # 按 `A` 取值的读者只认这个名字，所以两个名字都写。
        if op_type in ("MatMul", "Gemm") and "input0_node_id" in fields:
            fields["A"] = fields["input0_node_id"]

        if op_type == "Gemm":
            is_v = (weight_param is not None and "v_proj" in weight_param) or any(
                (consumers[node] and KV_DMA_META_KEY in c.meta
                 and not c.meta[KV_DMA_META_KEY].is_key)
                for c in consumers[node])
            fields.update(hw_table.top_level_fields(
                "Gemm", output_dtype="int8" if is_v else "float16",
                spec=ACTIVATION_LAYOUT))
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
                fields["activation_mode"] = 0
                fields["activation_special_operators"] = 0
                fields["flp_min_exp"] = 10
                fields["flp_max_exp"] = 17
                fields["flp_mantisa"] = 3
        if op_type == "EltwiseMul":
            fields.update(hw_table.top_level_fields(
                "EltwiseMul", spec=ACTIVATION_LAYOUT))
            fields["kantor_A_bias_buffer_file"] = names.kantor_bias(node_id, "A")
            fields["kantor_A_Shift"] = names.kantor_shift(node_id, "A")
            fields["kantor_B_scale_buffer_file"] = names.kantor_scale(node_id, "B")
            fields["kantor_B_bias_buffer_file"] = names.kantor_bias(node_id, "B")
            fields["kantor_B_Shift"] = names.kantor_shift(node_id, "B")
        if op_type == "EltwiseAdd":
            fields.update(hw_table.top_level_fields(
                "EltwiseAdd", spec=ACTIVATION_LAYOUT))

        contraction = _contraction_of(node, node_id)
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
        # DQ 的输出是它自己那一行（与 output_buffer_<self>.bin 同源），
        # 不按导出图的 seq_len 摊开。
        shape = dq_out_shape or _slot_dims(
            op_type, role, slots, _shape_of(node), rewrite=rewrite_dims,
            export_seq=export_seq)
        for consumer in downstream:
            dims = shape
            # KV 写回的**新值槽**（槽 2）落盘的是 1 个 token 的 nh×hd
            # （参考 `input_buffer_2` 边是 `1x32x1x128`、文件 4096 字节）。
            if rewrite_dims and KV_DMA_META_KEY in consumer.meta:
                dims = f"1x{slots.heads}x1x{slots.head_dim}"
            # K 路：参考在 KV DMA 与 Split 之间有一个 Transpose，把
            # `1x32x1024x128` 转成 `1x32x128x1024` 再切。我方把转置折进了
            # Split，入边仍是未转置的缓存平面。元素数相同，尺寸检查看不见。
            # 把这条边改成参考的 Kᵀ 平面，让 Split 的输入与出边同一套布局。
            elif (rewrite_dims and op_type == "KV_Cache_DMA"
                  and consumer.meta.get(SPLIT_META_KEY) is not None
                  and node.meta.get(KV_DMA_META_KEY) is not None
                  and node.meta[KV_DMA_META_KEY].is_key):
                dims = f"1x{slots.heads}x{slots.head_dim}x{slots.seq}"
            # Split 的 32 路出边按消费者角色分三份：Q / Kᵀ / V。
            # 三个 Split 共用一个 op_type，按生产者形状改写会把 96 条边
            # 全写成 KV cache 全量（`1x32x1024x128`），参考是
            # `1x1x1x128`×32 + `1x1x128x1024`×32 + `1x1x1024x128`×32。
            elif rewrite_dims and op_type == "Split":
                dims = _split_out_dims(node, consumer, emittable, slots)
            if not dims:
                raise ValueError(
                    f"{node.name} -> {consumer.name} 这条边的 dims 算不出来。"
                    f"形状可知，写 unknown 只会让缓冲按 1 个元素分配")
            edges.append(Edge(node_id, node_ids[consumer], dims))

    # dtype 收尾：布局算子要沿边传播，所以必须等全部节点发射完再统一盖章。
    _stamp_dtypes(gml_nodes)
    # idx 要等所有 inputN_node_id 都写完才能反查，与 dtype 同一处收尾。
    _stamp_idx(gml_nodes)

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
    # 裁剪**之前**每个节点出边的形状。出口边要用它，所以必须在这里取：
    # 裁完之后那条边已经不在 `kept_edges` 里了。
    out_dims = {e.source: e.dims for e in edges}
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
                "input_sf_dtype": "float16",
                "input_zp": names.zero_point(exit_id),
                names.residual_buffer_key("input", 0): [n.node_id],
                names.port_node_id_key("input", 0): n.node_id,
                "input_count": 1,
            }))
            n.fields["output_buffer"] = names.data_buffer(exit_id)
            n.fields[names.residual_buffer_key("output", 0)] = [exit_id]
            n.fields[names.port_node_id_key("output", 0)] = exit_id
            # 出口边的形状 = 这个残留节点自己的输出形状。它是可知的：
            # 该节点原来那条出边（被裁掉的消费者那条）就带着它。写 unknown
            # 会让写盘侧按 1 个元素分配缓冲，而声明与文件仍然自洽，谁都发现不了。
            extra_edges.append(Edge(n.node_id, exit_id, out_dims[n.node_id]))
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
