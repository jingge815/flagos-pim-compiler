"""图层五类通用优化。

执行路径按 `node.target` 逐节点发命令，只写 meta 不改变执行，所以这五个
Pass 都直接改图：常量折叠与 CSE 把重复计算收成一个节点，融合把相邻的
逐元素算子换成一个合并算子，DCE 删掉没人用的节点，分块把超预算的算子
拆成多个预算内的子节点。

顺序固定为 常量折叠 → CSE → 融合 → DCE → 分块：折叠产出的常量节点是
CSE 的合并对象，融合替换出的旧节点由 DCE 清理，分块要按前面化简后的
最终形状比较预算。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace

import torch
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg

from contracts.dtypes import dtype_bytes
from contracts.graph_meta import (
    DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, FUSED_TAIL_META_KEY,
    REDISTRIBUTE_META_KEY, SPEC_META_KEY)
from contracts.ir_payloads import FusedTail
from contracts.fusion_contract import (
    GATE_ACTIVATIONS, GATE_TARGETS, LINEAR_TAIL_ACTIVATIONS)
from contracts.op_semantics import ELEMENTWISE, FOLDABLE
from contracts.pim_tensor_spec import RedistributeEdge
from contracts.unified_ir import VAL_META_KEY
from graph.fuse_pim import _match_rms_norm, _real_users, is_float_cast

# torch 的 dtype 到契约里的类型名。分块只为算字节数，类型不在这张表里就
# 交给 dtype_bytes 抛错，不在这里兜底。
_DTYPE_NAMES = {
    torch.float32: "float32", torch.float16: "float16",
    torch.int32: "int32", torch.int64: "int64", torch.int8: "int8",
}


def _aten(name: str):
    return getattr(torch.ops.aten, name.split(".")[0])


def _target_name(target) -> str:
    """`aten.add.Tensor` 取 `add.Tensor` 这段，用来对性质表。"""
    text = str(target)
    marker = "aten."
    pos = text.find(marker)
    return text[pos + len(marker):] if pos >= 0 else ""


def _is_pure(node: Node) -> bool:
    """按 aten schema 判纯：不写输入、不随机、不在副作用表里。与 DCE 同一判据。"""
    return (node.op == "call_function"
            and getattr(node.target, "_schema", None) is not None
            and not node.is_impure())


def _is_view(node: Node) -> bool:
    """返回值是输入的别名，即形状类算子（unsqueeze、expand、select 等）。"""
    return any(r.alias_info is not None for r in node.target._schema.returns)


def _is_foldable(node: Node) -> bool:
    return node.op == "call_function" and _target_name(node.target) in FOLDABLE


def _is_elementwise(node: Node) -> bool:
    return node.op == "call_function" and _target_name(node.target) in ELEMENTWISE


def _literal_args(node: Node) -> bool:
    """参数里除节点引用外只有可哈希的字面量，结构键才有意义。"""
    for arg in (*node.args, *node.kwargs.values()):
        if isinstance(arg, Node):
            continue
        try:
            hash(arg)
        except TypeError:
            return False
    return True


def _const_value(node: Node):
    """追溯到常量的值；追溯不到返回 None。

    `get_attr` 的目标按点号逐层取：`lin.weight` 是子模块的参数，
    折叠出的常量是本模块的 buffer，两者都是编译期就确定的张量。
    取到的不是张量（比如子模块本身）就不是常量。
    """
    if node.op != "get_attr":
        return None
    value = node.graph.owning_module
    for part in node.target.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value if isinstance(value, torch.Tensor) else None


def _constant_of(arg):
    """单个参数的常量值。节点追溯不到常量，或不是张量，返回 None。"""
    if isinstance(arg, Node):
        value = _const_value(arg)
        return value if isinstance(value, torch.Tensor) else None
    return arg


def _all_constant(node: Node):
    """所有输入都能追溯到常量时返回求值用的位置参数与关键字参数。

    关键字参数也是输入：`add(a, b, alpha=2)` 丢掉 `alpha` 会算成 `a + b`。
    任一输入追溯不到就返回 None，位置参数与关键字参数成对返回。
    """
    values = []
    for arg in node.args:
        value = _constant_of(arg)
        if value is None and isinstance(arg, Node):
            return None
        values.append(arg if value is None else value)
    kwargs = {}
    for name, arg in node.kwargs.items():
        value = _constant_of(arg)
        if value is None and isinstance(arg, Node):
            return None
        kwargs[name] = arg if value is None else value
    return values, kwargs


def _eval_target(target, args, kwargs=None):
    """按 aten 目标求值。目标本身就是可调用的重载，直接用它。"""
    return target(*args, **(kwargs or {}))


# ---------------------------------------------------------------- 常量折叠

# 结果元素数超过这个上限就不折：大常量链折进编译期会放大编译内存和时间。
FOLD_MAX_ELEMENTS = 4096

def _install_constant(gm: GraphModule, node: Node, value: torch.Tensor) -> Node:
    """把求值结果做成一个 `get_attr` 常量节点。

    只有 `get_attr` 能被后续折叠继续追溯，也能被 FX 直接求值，
    所以常量不能做成普通的 `call_function`。
    """
    name = f"_folded_{node.name}"
    # buffer 才能被 get_attr 取到；普通属性会被 FX 当成子模块去找。
    gm.register_buffer(name, value)
    with gm.graph.inserting_before(node):
        const = gm.graph.get_attr(name)
    const.meta["val"] = value
    # 分区在优化之前跑。新节点不补标注，切分传播会因缺标注报错。
    # 折叠出的常量不是 OpOverload 的结果，与 partition 一样判成主机。
    const.meta[DEVICE_META_KEY] = DEVICE_HOST
    return const


def fold_constants(gm: GraphModule) -> int:
    """输入全部可追溯到常量的纯算子换成一个常量节点。

    只折有实际计算的逐元素算子，形状类不折：reshape、transpose、squeeze
    不产生新数值，折掉会改变切分传播看到的节点。`linear` 的权重虽是常量，
    输入是激活，不在此列；随机与写内存算子不在纯算子表里，天然排除。
    原节点已无用户，当场删除。
    """
    folded = 0
    for node in list(gm.graph.nodes):
        if not _is_foldable(node):
            continue
        resolved = _all_constant(node)
        if resolved is None:
            continue
        args, kwargs = resolved
        value = _eval_target(node.target, args, kwargs)
        # 结果太大就留到运行期算，不把整块计算搬进编译期。
        if getattr(value, "numel", lambda: 0)() > FOLD_MAX_ELEMENTS:
            continue
        const = _install_constant(gm, node, value)
        node.replace_all_uses_with(const)
        gm.graph.erase_node(node)
        folded += 1
    gm.graph.lint()
    return folded


# ---------------------------------------------------------------- CSE

def _arg_key(arg):
    """参数里的节点取名字，字面量取类型和值，容器逐项展开。

    类型必须入键：`1 == 1.0 == True` 且哈希相同，只取值会把结果类型不同的算子合并。
    """
    if isinstance(arg, Node):
        return ("node", arg.name)
    if isinstance(arg, (tuple, list)):
        return (type(arg), tuple(_arg_key(a) for a in arg))
    return ("lit", type(arg), arg)


def _structure_key(node: Node):
    """结构键：目标、位置参数、关键字参数。取值不参与，参数不同就不合并。

    关键字参数必须入键：`add(x, y, alpha=1)` 与 `add(x, y, alpha=2)`
    位置参数相同但结果不同，漏掉就会静默算错。
    """
    args = tuple(_arg_key(arg) for arg in node.args)
    kwargs = tuple(sorted(
        (name, _arg_key(arg)) for name, arg in node.kwargs.items()))
    return (node.target, args, kwargs)


def _alias_root(node: Node, roots: dict[Node, Node]) -> Node:
    """形状类算子与它的输入共用一块缓冲，沿第一个参数找到这块缓冲的源头。"""
    if node in roots:
        return roots[node]
    root = node
    if (node.op == "call_function" and getattr(node.target, "_schema", None)
            and _is_view(node) and node.args and isinstance(node.args[0], Node)):
        root = _alias_root(node.args[0], roots)
    elif node.target is operator.getitem and isinstance(node.args[0], Node):
        # 多输出算子经 getitem 取元素，元素就是生产者输出的一部分，写它等于写生产者。
        root = _alias_root(node.args[0], roots)
    roots[node] = root
    return root


def _written_nodes(node: Node) -> list[Node]:
    """schema 标了写别名的参数，即这个算子原地改写的输入。"""
    schema = getattr(node.target, "_schema", None)
    if node.op != "call_function" or schema is None:
        return []
    written = []
    for pos, param in enumerate(schema.arguments):
        if param.alias_info is None or not param.alias_info.is_write:
            continue
        arg = node.args[pos] if pos < len(node.args) else node.kwargs.get(param.name)
        # `Tensor(a!)[]` 这类写参数是张量列表，容器里的每个节点都被写。
        map_arg(arg, written.append)
    return written


def _mutated_roots(gm: GraphModule, roots: dict[Node, Node]) -> set[Node]:
    """图里被原地写过的缓冲源头。"""
    return {_alias_root(arg, roots)
            for node in gm.graph.nodes for arg in _written_nodes(node)}


def _folded_key(gm: GraphModule, node: Node):
    """折叠产生的常量按类型、形状和值入键；其余节点返回 None。

    模型参数不按值合并，它们的身份与放置由切分决定。
    """
    if node.op != "get_attr" or not node.target.startswith("_folded_"):
        return None
    value = getattr(gm, node.target)
    # 按字节比值，bfloat16 这类 numpy 不认的类型也能入键。
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return ("folded", value.dtype, tuple(value.shape), raw.numpy().tobytes())


def eliminate_common(gm: GraphModule, *, merge_shape_ops: bool = False) -> int:
    """同一图内目标与参数都相同的纯算子只留第一次出现的。

    键取结构不取值，所以参数不同的同型算子不会被合并。纯性按 schema 判定，
    写内存与随机算子不参与。重复节点已无用户，当场删除。折叠产生的常量
    例外，按值合并，模型参数不按值合并。

    结果会被原地写的节点不参与，否则两个用户共用一块缓冲；读被原地写过的
    缓冲的节点也不参与，同样的键在写之前和写之后取到的值不同。

    默认不合并形状类（返回别名的算子）。它们在每层各出现一次，流水并行时
    合并会让后面的层跨 stage 去读第一层的结果；没有流水并行时所有层在同一组
    DPU 上，`merge_shape_ops` 打开后连形状类一起合并。
    """
    seen: dict[tuple, Node] = {}
    roots: dict[Node, Node] = {}
    mutated = _mutated_roots(gm, roots)
    merged = 0
    for node in list(gm.graph.nodes):
        key = _folded_key(gm, node)
        if key is None:
            if not _is_pure(node) or not _literal_args(node):
                continue
            if _is_view(node) and not merge_shape_ops:
                continue
            key = _structure_key(node)
        touched = [node, *node.all_input_nodes]
        if any(_alias_root(n, roots) in mutated for n in touched):
            continue
        first = seen.get(key)
        if first is None:
            seen[key] = node
            continue
        node.replace_all_uses_with(first)
        gm.graph.erase_node(node)
        if node.op == "get_attr":
            # 折叠常量的 buffer 每个节点独一份，节点删了 buffer 也一起删。
            delattr(gm, node.target)
        merged += 1
    gm.graph.lint()
    return merged


# ---------------------------------------------------------------- 融合

def _fused_target(producer: Node, consumer: Node):
    """两个逐元素算子合成的合并算子，一个节点承载两者的计算。

    用普通函数而不是新的 aten 目标：FX 对普通函数照样求值。
    但普通函数不是 `OpOverload`，`graph/partition.py` 会把它判成主机节点，
    所以合并算子在主机上求值，不走 `runtime/kernels.py` 的设备内核。

    闭包按调用时展开的参数求值：生产者的位置参数在前，关键字参数随后，
    再是消费者除生产者结果之外的位置参数，最后是消费者的关键字参数。
    这样 `pow(x, 3)` 的 `3`、`pow(x, exponent=3)` 的 `exponent`、
    `add(x, y, alpha=2)` 的 `alpha` 都不会在合并时丢掉。
    """
    producer_target = producer.target
    consumer_target = consumer.target
    producer_kw = tuple(producer.kwargs)
    consumer_kw = tuple(consumer.kwargs)
    # 消费者每个参数的来源：生产者结果用一个标记位，其余原样接在后面。
    # 标记位按位置还原，这样生产者结果在第二个参数上时顺序也不会反。
    consumed = tuple(arg is producer for arg in consumer.args)
    consumer_rest = tuple(
        arg for arg in consumer.args if arg is not producer)

    def fused(*args):
        pos = len(producer.args)
        producer_args = args[:pos]
        producer_kwargs = dict(zip(producer_kw, args[pos:pos + len(producer_kw)]))
        pos += len(producer_kw)
        rest = list(args[pos:pos + len(consumer_rest)])
        pos += len(consumer_rest)
        consumer_kwargs = dict(zip(consumer_kw, args[pos:]))
        produced = _eval_target(producer_target, producer_args, producer_kwargs)
        # 按原位置把生产者结果填回去，其余参数按出现顺序接上。
        consumer_args = tuple(
            produced if was_producer else rest.pop(0)
            for was_producer in consumed)
        return _eval_target(consumer_target, consumer_args, consumer_kwargs)

    fused.__name__ = (
        f"fused_{_target_name(producer_target).split('.')[0]}_"
        f"{_target_name(consumer_target).split('.')[0]}"
    )
    return fused, consumer_rest




def fuse_elementwise(gm: GraphModule) -> int:
    """单消费者的纯逐元素算子对换成一个合并算子节点。

    判定按算子性质不按名单：生产者只有一个用户、双方都是逐元素纯算子、
    消费者不改变形状。消费者的其它输入原样保留，所以二元逐元素算子也能
    合并。多用户不合并，否则另一个用户拿到的值会变；含归约或形状变化的
    不在逐元素表里，不合并。专用融合认的链（silu、rope、rsqrt）不在这张
    表里；GML 路整个不跑这个 pass（见 optimize_graph）。
    """
    # RMSNorm 的 rsqrt 后接 mul，两者都是逐元素算子，但整条链归 GML 的
    # 专用融合。匹配从 pow 开始，要求 rsqrt 仍是独立节点，所以链上所有
    # 节点都不能提前合并，包括被跳过的 to.dtype。
    claimed = set()
    for node in gm.graph.nodes:
        match = _match_rms_norm(node)
        if match is not None:
            claimed.update(match.eaten)
            claimed.update(match.passthrough)

    fused = 0
    for node in list(gm.graph.nodes):
        # 用户数要在循环里现查：前面的融合会删掉节点，改变后面节点的用户数。
        # 导出器插入的元数据断言不是值消费者，不算用户；残留的断言交给 DCE。
        users = _real_users(node)
        if node in claimed or not _is_elementwise(node) or len(users) != 1:
            continue
        consumer = users[0]
        if consumer in claimed or not _is_elementwise(consumer):
            continue
        # 主算子后接门控激活归 GML 的专用融合：它把激活折进主算子的
        # contraction 块。这里提前合并，那个块就发不出来。
        if (node.target in GATE_TARGETS
                and consumer.target in GATE_ACTIVATIONS):
            continue
        # RoPE 从链尾的 add 往上认，要求两个输入都是独立的 mul。
        # 只有这一种形态不能提前合并；单个 mul 再加一个偏置不是 RoPE。
        if (consumer.target is torch.ops.aten.add.Tensor
                and node.target is torch.ops.aten.mul.Tensor
                and sum(isinstance(arg, Node)
                        and arg.target is torch.ops.aten.mul.Tensor
                        for arg in consumer.args) == 2):
            continue
        # linear 后的尾部激活（含 SwiGLU 的 silu）留成独立节点：执行路径由
        # fuse_linear_activation 折进 linear 的编译内核，通用融合不碰它。
        # 只给真能折进 linear 的激活让路，判定与 fuse_linear_activation 同一个函数。
        if (node.target in LINEAR_TAIL_ACTIVATIONS
                and isinstance(node.args[0], Node)
                and _linear_blocks(node.args[0]) is not None):
            continue
        # 消费者的参数里必须真的用到生产者，否则合并没有意义。
        # 其它输入由合并节点原样接上，哪怕它另有用户：那个用户拿到的值
        # 不变，不要求它只有一个用户。
        if node not in consumer.args:
            continue
        target, consumer_rest = _fused_target(node, consumer)
        merged_args = (*node.args, *node.kwargs.values(),
                       *consumer_rest, *consumer.kwargs.values())
        with gm.graph.inserting_before(consumer):
            merged = gm.graph.call_function(target, merged_args)
        if "val" in consumer.meta:
            merged.meta["val"] = consumer.meta["val"]
        # 合并算子是普通函数，不是 OpOverload，partition 会把它判成主机。
        merged.meta[DEVICE_META_KEY] = DEVICE_HOST
        # 跳过生产者：它的参数里可能指着消费者（消费者的结果又被生产者
        # 使用）。那种位置要留着指向原节点，不能改接到合并节点，否则
        # 生产者的另一个用户拿到的就是合并结果。
        consumer.replace_all_uses_with(
            merged, delete_user_cb=lambda user: user is not node)
        gm.graph.erase_node(consumer)
        if not node.users:
            gm.graph.erase_node(node)
        fused += 1
    gm.graph.lint()
    return fused


def _retarget_edges(gm: GraphModule, edges, old: Node, new: Node) -> None:
    """把源是 `old` 的重分布边换成源是 `new` 的新边。

    边的 `src_spec` 与源节点的规格是同一个对象，内存规划在它上面回填偏移，
    所以要连规格一起换。同一条边既在 `edges` 里也挂在消费者 meta 上，两处换成同一个新对象。
    """
    def moved(edge):
        return replace(edge, src=new.name, src_spec=new.meta[SPEC_META_KEY])

    renamed = {}
    for node in gm.graph.nodes:
        attached = node.meta.get(REDISTRIBUTE_META_KEY)
        if not attached:
            continue
        for i, edge in enumerate(attached):
            if edge.src == old.name:
                if id(edge) not in renamed:
                    renamed[id(edge)] = moved(edge)
                attached[i] = renamed[id(edge)]
    for i, edge in enumerate(edges or []):
        if edge.src == old.name:
            edges[i] = renamed.get(id(edge)) or moved(edge)


def _linear_blocks(node: Node) -> list[Node] | None:
    """`node` 由哪些 linear 子块拼成；不是只沿 M、N 拼回的就返回 None。

    分块后 linear 换成 `cat` 拼回各子块，中间可能夹任意层浮点转换。每个子块都算完整的 K，
    激活可以逐块算；K 切开时各段用 `add` 相加，激活必须算在最终和上，所以不认。
    带 bias 的 linear 不认：A 路内核与镜像只有 x、w 两个实参，与 B 路发射器同一口径。
    """
    if len(_real_users(node)) != 1:
        return None
    if node.target is torch.ops.aten.linear.default:
        bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        return None if bias is not None else [node]
    # 只认浮点之间的转换：激活挪到整数转换之前，结果就不是原来的值了。
    if is_float_cast(node):
        return _linear_blocks(node.args[0])
    if node.target is torch.ops.aten.cat.default:
        blocks = []
        for part in node.args[0]:
            found = _linear_blocks(part) if isinstance(part, Node) else None
            if found is None:
                return None
            blocks += found
        return blocks
    return None


def _feeds_tail_activation(node: Node) -> bool:
    """linear `node` 经唯一用户链是否落到可折激活上，可夹任意层浮点转换。

    与 `_linear_blocks` 认同一种形态，分块据此决定不切 K。
    """
    if _linear_blocks(node) is None:
        return False
    (user,) = _real_users(node)
    while is_float_cast(user):
        users = _real_users(user)
        if len(users) != 1:
            return False
        (user,) = users
    return user.target in LINEAR_TAIL_ACTIVATIONS


def fuse_linear_activation(gm: GraphModule,
                           edges: list[RedistributeEdge] | None = None) -> int:
    """把 linear 后的单用户尾部激活折进 linear 节点，交给 A 路编译内核算。

    激活按 `LINEAR_TAIL_ACTIVATIONS` 判定，不按名字写死。放在切分传播之后跑：
    激活不能作用在部分和上，而部分和的投影经激活过传播就会报错，所以能到
    这里的 linear 输出都可以折。折掉后 linear 的 target 不变，执行计划按
    `FUSED_TAIL_META_KEY` 把激活下发进编译请求。

    `edges` 是切分传播产出的重分布边。激活节点被删掉，以它为源的边改指
    linear：一元激活不改布局，两者的切分规格相同。
    """
    fused = 0
    for node in list(gm.graph.nodes):
        kind = LINEAR_TAIL_ACTIVATIONS.get(node.target)
        if kind is None:
            continue
        (producer,) = node.args
        if not isinstance(producer, Node):
            continue
        # 分块拼回的 linear 也认：激活折进每个子块，拼回的结果就是激活后的值。
        # 激活必须是唯一用户，否则另一个用户拿不到线性结果。
        blocks = _linear_blocks(producer)
        if blocks is None:
            continue
        # 已有尾部（GML 专用融合的标记）不再折。
        if any(FUSED_TAIL_META_KEY in block.meta for block in blocks):
            continue
        # 激活自己要收重分布时，布局在两者之间变了，折进去就丢了这一步。
        if node.meta.get(REDISTRIBUTE_META_KEY):
            continue
        for block in blocks:
            block.meta[FUSED_TAIL_META_KEY] = FusedTail(
                activation=kind, pool=None, nodes=[node])
        _retarget_edges(gm, edges, node, producer)
        node.replace_all_uses_with(producer)
        gm.graph.erase_node(node)
        fused += 1
    if fused:
        gm.graph.lint()
        gm.recompile()
    return fused


# ---------------------------------------------------------------- DCE

def _impure(node: Node) -> bool:
    """沿用 FX 的副作用判定：写内存、随机、运行期断言都是可达性的根，不删。

    唯一例外是导出器插入的张量元数据断言：它只核对编译期已知的形状与类型，
    没有运行期效果，结果又没人用，留着只是死代码。
    """
    if node.target is torch.ops.aten._assert_tensor_metadata.default:
        return False
    return node.is_impure()


def eliminate_dead(gm: GraphModule) -> int:
    """删除没有用户的纯算子。

    复用 `Graph.eliminate_dead_code` 的可达性：从输出和有副作用的节点
    反向标可达，走不到的删除。返回值是删除的节点数。
    """
    before = len(list(gm.graph.nodes))
    gm.graph.eliminate_dead_code(is_impure_node=_impure)
    gm.graph.lint()
    return before - len(list(gm.graph.nodes))


# ---------------------------------------------------------------- 分块

# 只有这两个算子的输出能靠切分变小：逐元素算子的工作集等于输出本身，
# 拆开不减少单块字节。
_TILEABLE = frozenset({
    str(torch.ops.aten.linear.default),
    str(torch.ops.aten.matmul.default),
})


def _tensor_of(node: Node):
    """节点的输出张量。只认已登记的 `val`，没有就返回 None，调用方跳过。"""
    val = node.meta.get(VAL_META_KEY)
    if val is None or not hasattr(val, "shape"):
        return None
    return val


def _elem_bytes(val) -> int:
    """张量一个元素的字节数。dtype 不在表里直接报，不猜。"""
    name = _DTYPE_NAMES.get(val.dtype)
    if name is None:
        raise ValueError(f"分块不认识的 dtype: {val.dtype}")
    return dtype_bytes(name)


def _sliced_shape(val, dim: int, start: int, length: int):
    """切片后的形状。上界超出原维度时截断，对应最后一块不满的情形。"""
    if val is None or not hasattr(val, "shape"):
        return None
    shape = list(val.shape)
    shape[dim] = min(length, shape[dim] - start)
    return torch.empty(shape, dtype=val.dtype)


def _slice_arg(graph, arg: Node, dim: int, start: int, length: int) -> Node:
    """沿 `dim` 切出 `[start, start+length)` 这一段。"""
    sliced = graph.call_function(
        torch.ops.aten.slice.Tensor, (arg, dim, start, start + length))
    # 切片是设备算子。权重节点此时还没有设备标注（它的设备要等
    # 切分传播按用户反推），所以不能继承它，一律标成设备。
    sliced.meta[DEVICE_META_KEY] = DEVICE_DPU
    shaped = _sliced_shape(arg.meta.get(VAL_META_KEY), dim, start, length)
    if shaped is not None:
        sliced.meta[VAL_META_KEY] = shaped
    return sliced


def _power_lengths(extent: int, floor: int) -> list[int]:
    """从 `floor` 起、不超过 `extent` 的 2 的幂块长，整维本身也算一个候选。

    整维不是 2 的幂时仍保留：不切这一维时块长就是整维。
    """
    if extent < floor:
        return []
    lengths = []
    length = floor
    while length < extent:
        lengths.append(length)
        nxt = length * 2
        if nxt <= length:
            break
        length = nxt
    lengths.append(extent)
    return lengths


def _gemm_layout(node):
    """返回 `(激活, 权重, 输出行数, 归约长度, 输出特征数)`。对不上返回 None。

    `linear` 的权重是 `(N, K)`，`matmul` 的第二个操作数是 `(K, N)`；右操作数
    带 batch 的 matmul 不切，返回 None。输出行数是输出去掉最后一维后的元素数。
    """
    out = _tensor_of(node)
    args = [a for a in node.args
            if isinstance(a, Node) and _tensor_of(a) is not None]
    if out is None or len(args) < 2:
        return None
    left, right = args[0], args[1]
    lv, rv = _tensor_of(left), _tensor_of(right)
    # 先读元素宽度：dtype 不在表里要在这里报，而不是静默跳过。
    _elem_bytes(lv)
    is_matmul = node.target is torch.ops.aten.matmul.default
    # 设计边界：右操作数带 batch 的 matmul 不切。各批的权重不同，行窗口按左
    # 操作数切会把批次对错；Llama 里这类 matmul 只有注意力，工作集远小于预算。
    if is_matmul and rv.ndim > 2:
        return None
    k_dim = rv.ndim - 2 if is_matmul else rv.ndim - 1
    n_dim = rv.ndim - 1 if is_matmul else 0
    if (k_dim < 0 or lv.shape[-1] != rv.shape[k_dim]
            or out.shape[-1] != rv.shape[n_dim]):
        return None
    rows = out.numel() // out.shape[-1]
    return left, right, rows, rv.shape[k_dim], rv.shape[n_dim]


def _resident_bytes(val) -> int:
    """这一块实际驻留的字节。

    float16 在计算前被提升到 float32，预算按提升后的宽度算，
    否则选出的子块驻留量是预算的两倍。
    """
    eb = _elem_bytes(val)
    if val.dtype.is_floating_point and val.dtype != torch.float32:
        eb = dtype_bytes("float32")
    return val.numel() * eb


# 图层分块一个节点最多拆出的块数。超过时图层不切，留给算子层 A 路的
# tt.dot 分块压进预算，否则真实 7B 投影会拆出数千个节点。
MAX_TILE_BLOCKS = 512


def _tile_plan(node, budget: int, tp_width: int):
    """块数最少且单块工作集不超预算的三维切分。切不开返回 None。

    工作集按子块实际驻留的张量算：输入 `M*K`、权重 `N*K`、输出 `M*N`，
    与算子层 `fitsBudget` 同一口径。浮点输入先提升到 float32，字节按
    提升后的宽度计。M、N、K 三块长度各自取 2 的幂，整维也是候选。
    K 切成多段时各段输出要相加，所以输出按整段计。块数相同优先保留
    更长的 K，其次 N，与算子层的取舍一致。块数超过 `MAX_TILE_BLOCKS` 时不切。
    """
    layout = _gemm_layout(node)
    if layout is None:
        return None
    left, right, rows, k_extent, n_extent = layout
    out = _tensor_of(node)
    # 浮点先提升到 float32 再算，输入、权重、输出三块都按提升后的宽度计。
    eb = dtype_bytes("float32") if out.dtype.is_floating_point else _elem_bytes(out)
    # 每块的输出特征数必须能被张量并行宽度整除，否则按卡再切时除不尽。
    n_floor = tp_width if n_extent % tp_width == 0 else 1
    # bias 与输出特征对齐，跟着 N 一起缩小；对不上的输入整块保留。
    fixed = 0
    bias = None
    for arg in node.args:
        if not isinstance(arg, Node) or arg is left or arg is right:
            continue
        val = _tensor_of(arg)
        if val is None:
            continue
        if val.ndim == 1 and val.shape[-1] == n_extent:
            bias = arg
        else:
            fixed += _resident_bytes(val)
    # bias 与输出特征对齐，跟着 N 缩小，宽度同样按提升后算。
    bias_eb = 0
    if bias is not None:
        bias_eb = (dtype_bytes("float32")
                   if _tensor_of(bias).dtype.is_floating_point
                   else _elem_bytes(_tensor_of(bias)))

    def cost(m, n, k):
        k_blocks = (k_extent + k - 1) // k
        out_bytes = m * n * eb * (k_blocks if k_blocks > 1 else 1)
        return (m * k + n * k) * eb + out_bytes + fixed + n * bias_eb

    # 整块是否超预算与候选用同一个成本，bias 与其余常驻输入都算进去。
    if cost(rows, n_extent, k_extent) <= budget:
        return None
    # 后接可折激活的 linear 不切 K：K 段相加后激活要算在和上，逐段折不进去。
    # 只沿 M、N 切仍超预算时整块不切，K 留给 A 路的 tt.dot 分块。
    tail_act = (node.target is torch.ops.aten.linear.default
                and _feeds_tail_activation(node))
    k_lengths = [k_extent] if tail_act else _power_lengths(k_extent, 1)
    best = None
    for m in _power_lengths(rows, 1):
        for n in _power_lengths(n_extent, n_floor):
            if n < n_extent and n % n_floor:
                continue
            for k in k_lengths:
                k_blocks = (k_extent + k - 1) // k
                total = cost(m, n, k)
                blocks = (((rows + m - 1) // m) * ((n_extent + n - 1) // n)
                          * k_blocks)
                if total > budget or blocks <= 1:
                    continue
                rank = (blocks, -k, -n, -m)
                if best is None or rank < best[0]:
                    best = (rank, m, n, k)
    if best is None or best[0][0] > MAX_TILE_BLOCKS:
        return None
    _, m, n, k = best
    return (left, right, bias, m, n, k, rows, k_extent, n_extent)


def _cut(graph, arg, dim: int, start: int, length: int, extent: int):
    """沿 `dim` 切一段。块长覆盖整维时不切，原样返回。"""
    if length >= extent:
        return arg
    return _slice_arg(graph, arg, dim, start, length)


def _annotate(node, part, shape, dtype=None) -> None:
    """子块继承原节点的设备，输出形状按切出来的大小重算。

    `dtype` 缺省取原节点的。float16 先提升到 float32 再算时，结果也是
    float32，调用方要显式传入，否则累加会按 float16 把它降回去。
    """
    part.meta[DEVICE_META_KEY] = node.meta.get(DEVICE_META_KEY)
    out = node.meta.get(VAL_META_KEY)
    dtype = dtype if dtype is not None else out.dtype if out is not None else None
    if dtype is not None:
        part.meta[VAL_META_KEY] = torch.empty(tuple(shape), dtype=dtype)


def _widen(graph, node, value, shape):
    """浮点张量提升到 float32。整数与 float32 原样返回。

    dtype 看这一段自己的标注，不看被切分的原节点：子块的 linear 已经
    在 float32 上算完，再提升一次会把累加和无谓地转来转去。
    """
    own = value.meta.get(VAL_META_KEY)
    out = own if own is not None else node.meta.get(VAL_META_KEY)
    if out is None or not out.dtype.is_floating_point or out.dtype == torch.float32:
        return value
    cast = graph.call_function(
        torch.ops.aten._to_copy.default, (value,),
        {"dtype": torch.float32})
    cast.meta[DEVICE_META_KEY] = node.meta.get(DEVICE_META_KEY)
    # 形状取这一段自己的，不取整块输出：子块可能只是一行、一列。
    cast.meta[VAL_META_KEY] = torch.empty(tuple(shape), dtype=torch.float32)
    return cast


def _add_parts(graph, node, parts, shape):
    """多段结果相加。只有一段时原样返回。

    低精度上逐段相加的舍入比一次矩阵乘大，所以先提升到 float32
    累加，写回时再落到原 dtype。
    """
    total = _widen(graph, node, parts[0], shape)
    for part in parts[1:]:
        total = graph.call_function(
            torch.ops.aten.add.Tensor, (total, _widen(graph, node, part, shape)))
        _annotate(node, total, shape, torch.float32)
    out = node.meta.get(VAL_META_KEY)
    if out is not None and out.dtype.is_floating_point and out.dtype != torch.float32:
        total = graph.call_function(
            torch.ops.aten._to_copy.default, (total,), {"dtype": out.dtype})
        _annotate(node, total, shape)
    return total


def _cat_parts(graph, node, parts, dim: int, shape):
    """多段沿 `dim` 拼回。只有一段时原样返回。"""
    if len(parts) == 1:
        return parts[0]
    joined = graph.call_function(torch.ops.aten.cat.default, (parts, dim))
    _annotate(node, joined, shape)
    return joined


def _row_origin(index: int, leading: tuple[int, ...]) -> list[int]:
    """展平行号在各前导维上的坐标。行是行主序，最内层在最后一维。"""
    coords = []
    for size in reversed(leading):
        coords.append(index % size)
        index //= size
    coords.reverse()
    return coords


def _row_window(start: int, count: int, leading: tuple[int, ...]):
    """`[start, start+count)` 这段行拆成若干行主序连续的矩形切片。

    每个切片是 `(各维起点, 各维长度)`：某一维取若干层，比它内的维取满，
    比它外的维长度为 1。起点不在内层边界上时先取到边界，剩余行下一轮再取。
    """
    windows = []
    pos = start
    left = count
    while left:
        origin = _row_origin(pos, leading)
        last = len(leading) - 1
        dim, take, block = last, min(left, leading[last] - origin[last]), 1
        # 起点对齐到内层边界、剩余行够一整层时，改在更外的维上整层取。
        inner = 1
        for d in range(last - 1, -1, -1):
            inner *= leading[d + 1]
            if origin[d + 1] != 0 or left < inner:
                break
            dim, take, block = d, min(left // inner, leading[d] - origin[d]), inner
        length_at = [1] * dim + [take] + list(leading[dim + 1:])
        windows.append((list(origin), length_at))
        pos += take * block
        left -= take * block
    return windows


def _apply_plan(gm, node, plan) -> None:
    """按三维切分改图：K 维各段相加，M 与 N 两维用 `cat` 拼回。

    M 是输出去掉最后一维后的元素数。行数不被块长整除时最后一组按剩余
    行数截短；行区间跨过前导维边界时拆成多个窗口，各自切片再沿被切的
    那一维拼回。
    """
    left, right, bias, m, n, k, rows, k_extent, n_extent = plan
    out = _tensor_of(node)
    rv = _tensor_of(right)
    lv = _tensor_of(left)
    is_matmul = node.target is torch.ops.aten.matmul.default
    k_dim = rv.ndim - 2 if is_matmul else rv.ndim - 1
    n_dim = rv.ndim - 1 if is_matmul else 0
    leading = tuple(lv.shape[:-1])
    # 行组数向上取整，最后一组不足一块时按剩余行数取。
    n_row = max(1, (rows + m - 1) // m)
    with gm.graph.inserting_before(node):
        # 每段记录 (结果, 各前导维起点, 各前导维长度)，拼接按位置分组。
        # 只看形状区分不了「第 0 批的第 4 行」和「第 1 批的第 0 行」。
        row_groups = []
        for ri in range(n_row):
            row_count = min(m, rows - ri * m)
            windows = (_row_window(ri * m, row_count, leading)
                       if row_count < rows else [None])
            for window in windows:
                col_groups = []
                for nstart in range(0, n_extent, n):
                    parts = []
                    piece = None
                    for kstart in range(0, k_extent, k):
                        w = _cut(gm.graph, right, n_dim, nstart, n, n_extent)
                        w = _cut(gm.graph, w, k_dim, kstart, k, k_extent)
                        # 行切在 K 切之前：先切最后一维会让前面的维号错位。
                        x = left
                        if window is not None:
                            origin, length_at = window
                            for dim in range(len(leading)):
                                x = _cut(gm.graph, x, dim, origin[dim],
                                         length_at[dim], leading[dim])
                        x = _cut(gm.graph, x, x.meta["val"].ndim - 1,
                                 kstart, k, k_extent)
                        # bias 不进子块：K 维多段时每段都加一次会把偏置算多遍。
                        # 两侧都提升到 float32：只提一侧会让 linear 的两个输入
                        # dtype 不同。权重侧的提升插在切片之后，权重到 linear
                        # 的链上仍只经过切片，切分传播认得出按列切分。
                        args = tuple(
                            _widen(gm.graph, node, w, tuple(_tensor_of(w).shape))
                            if item is right else
                            _widen(gm.graph, node, x, tuple(_tensor_of(x).shape))
                            if item is left else
                            None if item is bias else item
                            for item in node.args)
                        part = gm.graph.call_function(node.target, args)
                        piece = (_tensor_of(x).shape[:-1]
                                 + (_tensor_of(w).shape[n_dim],))
                        # 两个输入都已提升到 float32，结果也是 float32。
                        _annotate(node, part, piece, torch.float32)
                        parts.append(part)
                    reduced = _add_parts(gm.graph, node, parts, piece)
                    if bias is not None:
                        b = _cut(gm.graph, bias, 0, nstart, n, n_extent)
                        reduced = gm.graph.call_function(
                            torch.ops.aten.add.Tensor, (reduced, b))
                        _annotate(node, reduced, piece)
                    col_groups.append(reduced)
                # 各列沿输出的最后一维拼回。窗口形状取切片后的前导维，
                # 照搬整块行数会让后面的拼接把一行当成整块。
                # 标注用这个窗口自己的长度，同一行组的窗口长度可能不同。
                if window is None:
                    window_shape = tuple(out.shape)
                else:
                    window_shape = tuple(window[1]) + (out.shape[-1],)
                # 每个窗口单独成段，不在这里拼。一个行组的窗口可能跨过
                # 批次边界，先拼起来会让这一段在批次维上覆盖两行。
                piece = _cat_parts(
                    gm.graph, node, col_groups, out.ndim - 1, window_shape)
                if window is None:
                    origin = [0] * len(leading)
                    span = list(leading)
                else:
                    origin = list(window[0])
                    span = list(window[1])
                row_groups.append((piece, origin, span))
        # 行组可能在多个前导维上都比输出短，一个拼接维拼不回。
        # 从最内层往外逐层拼，按起点分组：同一组在其他维上起点和长度
        # 都相同，只在当前维上前后相接。
        def join(batch, dim):
            nodes = [b[0] for b in batch]
            # 结果形状取各段标注之和。位置跨度不能当形状用：一段的标注
            # 长度可能比它的位置跨度短，混用会让下一次拼接两侧对不齐。
            shape = list(nodes[0].meta["val"].shape)
            shape[dim] = sum(g.meta["val"].shape[dim] for g in nodes)
            joined = _cat_parts(gm.graph, node, nodes, dim, tuple(shape))
            origin = list(batch[0][1])
            span = list(batch[0][2])
            # 跨度按位置记录累加，不取拼接结果的标注形状：标注是输出的
            # 整维长度，比这一段实际覆盖的行数大。
            span[dim] = sum(b[2][dim] for b in batch)
            return (joined, origin, span)

        pending = list(row_groups)
        # 从最内层往外拼。段按行主序排列，最内层上前后相邻的段就是
        # 同一行的前后段；先拼完内层，外层上原本隔开的段才变成相邻。
        for dim in range(len(leading) - 1, -1, -1):
            merged = []
            batch = []
            for group in pending:
                # 是否同一段看终点是否接上，不能比起点：跨边界的第二段
                # 起点是下一行的开头，和前面的段起点不同。
                if batch and any(
                        group[1][d] != batch[-1][1][d] + batch[-1][2][d]
                        if d == dim else
                        group[1][d] != batch[-1][1][d] or group[2][d] != batch[-1][2][d]
                        for d in range(len(leading))):
                    merged.append(join(batch, dim))
                    batch = []
                batch.append(group)
            if batch:
                merged.append(join(batch, dim))
            pending = merged
        if len(pending) != 1:
            raise RuntimeError(f"{node.name} 的行窗口没有拼回整块: {[p[1:] for p in pending]}")
        total = pending[0][0]

    node.replace_all_uses_with(total)
    gm.graph.erase_node(node)



def tile_graph(gm: GraphModule, wram_bytes: int, tp_width: int = 1) -> int:
    """工作集超过 `wram_bytes` 的 linear 与 matmul 拆成预算内的子块。

    工作集是输入、权重与输出三块字节之和，与算子层 `fitsBudget` 同一口径。
    M、N、K 三个维一起取块长，选块数最少的那个。K 维切成多段时各段输出
    相加，M 与 N 两维用 `cat` 拼回。原样已在预算内，或没有任何切法能压进
    预算，则不改。
    """
    split = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) not in _TILEABLE:
            continue
        # dtype 不认识时直接报，不因形状对不上而静默跳过。
        for arg in (node, *node.args):
            if isinstance(arg, Node):
                val = _tensor_of(arg)
                if val is not None:
                    _elem_bytes(val)
        layout = _gemm_layout(node)
        if layout is None:
            continue
        plan = _tile_plan(node, wram_bytes, tp_width)
        if plan is None:
            continue
        _apply_plan(gm, node, plan)
        split += 1
    gm.graph.lint()
    return split


# ---------------------------------------------------------------- 入口

@dataclass
class OptReport:
    """一次 `optimize_graph` 的改动数，五类各记一笔。"""

    folded: int
    merged: int
    fused: int
    removed: int
    tiled: int


def optimize_graph(gm: GraphModule, *, wram_bytes: int = 65536,
                    tp_width: int = 1, tile: bool = True,
                    merge_shape_ops: bool = False) -> OptReport:
    """按固定顺序跑完五类优化，返回每步的改动数。

    `wram_bytes` 只有分块用，默认取硬件配置的单台 WRAM 预算。
    `merge_shape_ops` 只在没有流水并行时打开，见 `eliminate_common`。
    `tile` 只给 GML 路关：GML 路连融合一起跳过——切块会改掉缓冲名与字段，
    融合换出的闭包不可发射，两者都只服务执行路径。
    """
    report = OptReport(
        folded=fold_constants(gm),
        merged=eliminate_common(gm, merge_shape_ops=merge_shape_ops),
        # GML 路不跑通用融合：换出的合并节点是闭包，`from_fx` 不认它，
        # 被吃掉的计算会从产物里静默消失。融合由专用 Pass 负责。
        fused=fuse_elementwise(gm) if tile else 0,
        removed=eliminate_dead(gm),
        tiled=tile_graph(gm, wram_bytes, tp_width) if tile else 0,
    )
    gm.recompile()
    return report
