"""将标注图、通信计划和内存蓝图展开为 `ExecutionPlan`。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.fx import Node
from torch.fx.node import map_aggregate, map_arg

from comm.lowering import DmaEngine, all_gather, all_reduce, all_to_all, scatter
from comm.plan import CommPlanEntry
from contracts.dtypes import dtype_bytes
from contracts.exec_plan import Access, Command, ExecutionPlan
from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, REDISTRIBUTE_META_KEY, SPEC_META_KEY
from contracts.op_contract import DpuShard, PIMHardwareConfig

_REDISTRIBUTE_OP = {
    "all_reduce": "host_reduce",
    "all_gather": "host_concat",
    "all_to_all": "host_permute",
    "scatter": "host_slice",
    "local_slice": "dpu_slice",
}
_REDISTRIBUTE_FN = {"all_reduce": all_reduce, "all_gather": all_gather, "all_to_all": all_to_all}

KvAccessFn = Callable[[Node], "tuple[list[Access], list[Access]] | None"]
HostHandlerFn = Callable[[Node], "Callable | None"]


@dataclass
class CompiledPlan:
    """保存命令计划和图输出对应的命令编号。"""

    plan: ExecutionPlan
    output_cmd_id: int  # 图 output 节点对应命令的 id（取最终结果，如 logits）


def _as_torch(x: object) -> object:
    """将 NumPy 数组转为 PyTorch 张量，其余值保持不变。"""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def overlap(a: Access, b: Access) -> bool:
    """判断两个访问区间是否相交。"""
    return a.loc == b.loc and a.offset < b.offset + b.length and b.offset < a.offset + a.length


def _deps_of(reads: list[Access], writers: dict[tuple, list[tuple[Access, int]]]) -> list[int]:
    """返回与读区间相交的历史写命令。"""
    return [cid for a in reads for (wa, cid) in writers.get(a.loc, []) if overlap(a, wa)]


def _war_waits(
    loc: tuple, offset: int, pending_readers: dict[tuple, list[str]], reader_cmds: dict[tuple, list[int]]
) -> list[int]:
    """返回目标地址旧值的读取命令。"""
    return [cid for rn in pending_readers.get((loc, offset), []) for cid in reader_cmds.get((rn, loc[1]), [])]


def _shard_decision_of(spec, detail) -> list | None:
    """命令所处理张量的跨 DPU **放置**决策，交给算子编译器写进 pimir。

    三档都带出来，不只带切分那一档：`spec.placement.kind` 是统一 IR 里本来就有
    的信息（`Shard` / `Replicate` / `Partial`），早先这里只看 `detail.shard_dim`，
    于是 Replicate 与 Partial 被压成同一个 None —— PIMMLIR 侧因此分不清「复制」
    与「单 DPU」，而这两者的归约与容量口径并不相同（partial 每台持有一份全形状的
    局部和、要跨 DPU 归约才完整）。

    单 DPU（分片数 <= 1）仍返回 None：一个字不下发，文本与改动前逐字节相同。
    """
    num_dpus = len(spec.shard_map)
    if num_dpus <= 1:
        return None
    kind = str(getattr(spec.placement, "kind", "Shard")).lower()
    if kind == "shard":
        if detail.shard_dim < 0:
            # placement 说按维切，分片却没记维号 —— 两者必须自洽，静默按
            # 「没切」处理会把决策悄悄丢掉。
            raise ValueError(
                f"placement 是 Shard 但分片没有 shard_dim：{detail}")
        return DpuShard(dim=detail.shard_dim, num_dpus=num_dpus,
                        kind="shard").to_payload()
    if kind == "partial":
        # 归约方式取统一 IR 的 `reduce_type`；它是 partial 的全部内容。
        reduce = (spec.placement.reduce_type or spec.reduce_type or "sum")
        return DpuShard(dim=-1, num_dpus=num_dpus, kind="partial",
                        reduce=str(reduce).lower()).to_payload()
    return DpuShard(dim=-1, num_dpus=num_dpus,
                    kind="replicate").to_payload()


def _oplevel_op_of(node):
    """这个节点走 B 路（整算子级）的哪个内核入口；A 路或不认识的返回 None。

    判据与发射侧同源：`opcompiler_bridge/driver` 按 `request.op in _OPLEVEL_OPS`
    分路，而入口集合由 `contracts.op_semantics.oplevel_ops()` 派生。GML 名与内核
    入口不是一一对应（四种逐元素共用 `eltwise`，`silu` 走 `lut`），所以按
    `kernel_entry_of` 解析，不另立一份名单。
    """
    from contracts.op_semantics import kernel_entry_of

    return kernel_entry_of(str(node.target))


def _assert_same_shape_args_share_the_decision(
    node, dpu_id: int, arg_decisions: list[tuple], spec, out_detail
) -> None:
    """与输出同形的实参必须与输出同切分，否则抛错。

    发射侧（`opcompiler_bridge/driver._attach_layout`）把 `#pim.tasklet_tiled`
    贴到结果类型**以及与结果同形的那些类型**上 —— 不贴的话 `-pim-expand-phases`
    重建结果类型时编码会整个丢掉。代价是它默认「同形状 ⇒ 同切分」，而切分决策
    只取自输出的 `shard_map`，实参各自的分片从不参与判断。

    这条外推在这里钉住：不成立就抛错，而不是静默给那个实参写一个与事实不符的
    `dpusPerDevice`。实参与输出同形但切分方式不同（例如 redistribute 落地后的
    复制张量）时命中。

    只管 B 路（`_attach_layout` 走的那条）。A 路豁免的理由不是「实参不会被贴上
    编码」—— 恰恰相反：FlagTree 的 `TritonPIMTypeConverter` 把模块级 placement
    贴到该 kernel **所有**没有编码的张量上，实测 tp2 `linear` 的 pimir 里连索引
    张量（`tensor<4x32xi32>`，35 处）与指针张量都带 `dpusPerDevice = [1, 2]`。

    豁免成立的真实理由是：A 路那些编码当前**没有按轴取值的消费者**。读它的只有
    `verifyAgreesWith`（只比切分宽度，不比轴）与 `TritonSplitOpPattern`；资源算术
    走的是模块属性 `#pim.placement`，不是编码。所以多贴出来的编码不产生数值后果。
    将来若有 pass 真按轴分摊，这条豁免要连同那个 pass 一起重新评估。
    """
    if _oplevel_op_of(node) is None:
        return                              # A 路：不贴张量编码，无从写错
    out_decision = _shard_decision_of(spec, out_detail)
    if out_decision is None:                # 没有切分就不贴编码，无从写错
        return
    for name, shape, decision in arg_decisions:
        if shape == out_detail.local_shape and decision != out_decision:
            raise ValueError(
                f"{node.name} 在 DPU{dpu_id} 上：实参 {name} 与输出同形 "
                f"{shape}，但切分决策不同（实参 {decision}、输出 "
                f"{out_decision}）。发射侧会按同形状把输出的编码一起贴到它身上，"
                f"那就写出了与事实不符的 dpusPerDevice。")


def _placed_width(node: Node, placed_elem_bytes) -> int | None:
    """这个算子回传的元素宽度；没有回传就返回 None。

    键是 GML 的 op_type（与 GML 侧同一份），linear 这种没有内核入口的也查得到。
    """
    if not placed_elem_bytes:
        return None
    from gml_bridge.from_fx import _op_type_of
    back = placed_elem_bytes.get(_op_type_of(node))
    return back.placed_elem_bytes if back else None


def _node_access(node: Node, dpu_id: int, placed_elem_bytes=None) -> Access:
    """一个 DPU 张量节点在本 dpu 的地址区间（自身输出）。

    `placed_elem_bytes` 是 PIMMLIR 回传的元素宽度：给了就按它算字节数，
    不给才按图上的 dtype。回传与图上不一致时，区间长度必须跟着变。
    """
    spec = node.meta[SPEC_META_KEY]
    detail = spec.shard_map[dpu_id]
    itemsize = placed_elem_bytes or dtype_bytes(spec.dtype)
    nbytes = itemsize
    for dim in detail.local_shape:
        nbytes *= dim
    return Access(("dpu", dpu_id), detail.mram_offset, nbytes)


def _flatten_node_args(args) -> list:
    """把**张量列表**摊平，其余原样。

    `aten.cat([a, b], dim)` 的第一个实参是张量列表，按位置逐个编码看不出列表
    边界，所以摊平成 `a, b, dim`——内核按同一个顺序收。

    只摊平元素全是 `Node` 的列表。`aten.view(x, [1, 4, 32, 128])` 的第二个实参
    也是列表，但那是**形状字面量**：摊平会把 5 个参数喂给只收 2 个的内核。
    """
    flat: list = []
    for arg in args:
        if (isinstance(arg, (list, tuple)) and arg
                and all(isinstance(item, Node) for item in arg)):
            flat.extend(arg)
        else:
            flat.append(arg)
    return flat


def _edge_accesses(entry: CommPlanEntry) -> tuple[list[Access], list[Access]]:
    """一条 redistribute 边全部收集段（reads）与回写段（writes）的地址区间。"""
    reads = [Access(("dpu", s.src_dpu), s.src_addr, s.nbytes) for s in entry.collect_segments]
    writes = [Access(("dpu", s.dst_dpu), s.dst_addr, s.nbytes) for s in entry.writeback_segments]
    return reads, writes


class _PlanBuilder:
    """保存构建命令计划时的命令、写者和读者索引。"""

    def __init__(self, gm_root, pending_readers: dict[tuple, list[str]]) -> None:
        self.gm_root = gm_root
        self.commands: list[Command] = []
        self.writers: dict[tuple, list[tuple[Access, int]]] = {}
        self.reader_cmds: dict[tuple, list[int]] = {}
        self.pending_readers = pending_readers
        self.host_value_of: dict[str, int] = {}  # 节点名称到 host 结果命令的映射。
        self._next_id = 0

    def append(self, op: str, dpu_id: int | None, payload: dict,
               reads: list[Access], writes: list[Access], waits: list[int],
               num_tasklets: int = 1) -> Command:
        cmd = Command(id=self._next_id, op=op, dpu_id=dpu_id, payload=payload,
                       reads=reads, writes=writes, waits=sorted(set(waits)),
                       num_tasklets=num_tasklets)
        self._next_id += 1
        self.commands.append(cmd)
        for w in writes:
            self.writers.setdefault(w.loc, []).append((w, cmd.id))
        return cmd

    def register_reader(self, reader_name: str, dpu_id: int | None, cmd_id: int) -> None:
        """登记读者节点在指定 DPU 上对应的命令编号。"""
        self.reader_cmds.setdefault((reader_name, dpu_id), []).append(cmd_id)
        if dpu_id is not None:
            self.reader_cmds.setdefault((reader_name, None), []).append(cmd_id)

    def resolve_node(self, node: Node) -> tuple[Callable[[object], object], int | None]:
        """返回节点的运行时取值函数和生产命令编号。"""
        if node.op == "get_attr":
            obj = self.gm_root
            for part in str(node.target).split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                obj = obj.detach()  # 使用不参与梯度计算的常量张量。
            return (lambda hal: obj), None
        if node.op == "placeholder":
            name = node.name
            return (lambda hal: hal.bound_value(name)), None
        cmd_id = self.host_value_of[node.name]
        return (lambda hal: _as_torch(hal.result_of(cmd_id))), cmd_id


def _emit_redistribute(builder: _PlanBuilder, edge, entry: CommPlanEntry, src_node: Node) -> None:
    """将一条重分布边生成包含全部段访问的命令。"""
    reads, writes = _edge_accesses(entry)
    waits = _deps_of(reads, builder.writers)
    for w in writes:
        waits += _war_waits(w.loc, w.offset, builder.pending_readers, builder.reader_cmds)

    if edge.type == "scatter":
        get_host_buf, dep_id = builder.resolve_node(src_node)
        if dep_id is not None:
            waits.append(dep_id)

        def fn(hal, cmd):
            engine = DmaEngine(hal.dpu_set)
            scatter(entry, engine, get_host_buf(hal))
            return None
    elif edge.type == "local_slice":
        def fn(hal, cmd):
            engine = DmaEngine(hal.dpu_set)
            itemsize = entry.dtype.itemsize
            for seg in entry.segments:
                n = seg.nbytes // itemsize
                data = engine.copy_from_dpu(
                    seg.src_dpu, seg.src_addr, n, entry.dtype)
                engine.copy_to_dpu(seg.dst_dpu, seg.dst_addr, data)
            return None
    else:
        primitive = _REDISTRIBUTE_FN[edge.type]

        def fn(hal, cmd):
            engine = DmaEngine(hal.dpu_set)
            return primitive(entry, engine)

    op = _REDISTRIBUTE_OP[edge.type]
    cmd = builder.append(op, None, {"edge_id": edge.edge_id, "fn": fn}, reads, writes, waits)
    if edge.dst_loc.get("device") == DEVICE_HOST:
        builder.host_value_of[edge.src] = cmd.id
    # 将重分布命令登记为源张量读者。
    builder.register_reader(f"redist:e{edge.edge_id}", None, cmd.id)
    for w in writes:
        builder.register_reader(f"redist:e{edge.edge_id}", w.loc[1], cmd.id)


def _emit_host_op(builder: _PlanBuilder, node: Node, host_handler_of: HostHandlerFn | None) -> Command:
    """生成一个解析节点实参后执行的 host 命令。"""
    waits: list[int] = []

    def collect(n: Node):
        getter, dep_id = builder.resolve_node(n)
        if dep_id is not None:
            waits.append(dep_id)
        return getter

    arg_getters = map_arg(node.args, collect)
    kwarg_getters = map_arg(node.kwargs, collect)
    resolve = lambda x, hal: x(hal) if callable(x) else x  # noqa: E731

    handler = host_handler_of(node) if host_handler_of is not None else None
    if handler is not None:
        def fn(hal, cmd, _h=handler, _a=arg_getters, _k=kwarg_getters):
            args = map_aggregate(_a, lambda x: resolve(x, hal))
            kwargs = map_aggregate(_k, lambda x: resolve(x, hal))
            return _h(hal, cmd, args, kwargs)
    else:
        def fn(hal, cmd, _target=node.target, _a=arg_getters, _k=kwarg_getters):
            args = map_aggregate(_a, lambda x: resolve(x, hal))
            kwargs = map_aggregate(_k, lambda x: resolve(x, hal))
            return _target(*args, **kwargs)

    cmd = builder.append("host_op", None, {"node": node.name, "fn": fn}, [], [], waits)
    builder.host_value_of[node.name] = cmd.id
    return cmd


def build_execution_plan(
    nodes: list[Node],
    gm_root,
    comm_entries: dict[int, CommPlanEntry],
    pending_readers: dict[tuple, list[str]],
    *,
    hardware: PIMHardwareConfig,
    kv_access_of: KvAccessFn | None = None,
    host_handler_of: HostHandlerFn | None = None,
    sdpa_info_of: "Callable[[Node], dict | None] | None" = None,
    num_tasklets: int = 4,
    placed_elem_bytes=None,
) -> CompiledPlan:
    """从图、通信计划和内存依赖构建按拓扑顺序执行的命令计划。"""
    if hardware.num_tasklets != num_tasklets:
        raise ValueError(
            f"hardware.num_tasklets ({hardware.num_tasklets}) must match num_tasklets ({num_tasklets})"
        )
    builder = _PlanBuilder(gm_root, pending_readers)
    by_name = {n.name: n for n in nodes}
    output_cmd_id = -1
    for node in nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op != "output" and node.meta.get("val") is None:
            # 无张量输出的节点不生成命令。
            continue

        # 处理节点输入的重分布。
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            _emit_redistribute(builder, edge, comm_entries[edge.edge_id], by_name[edge.src])

        if node.op == "output":
            # 图输出使用最后一次写入 host 的命令结果。
            (out_arg,) = node.args
            src = out_arg[0] if isinstance(out_arg, (list, tuple)) else out_arg
            output_cmd_id = builder.host_value_of.get(src.name, output_cmd_id)
            continue

        if node.meta.get(DEVICE_META_KEY) == DEVICE_DPU:
            spec = node.meta[SPEC_META_KEY]
            # 校验张量分片对应有效的 DPU 地址。
            if not spec.shard_map or len(spec.shard_map) > hardware.num_dpus:
                raise ValueError(
                    f"{node.name} shard count ({len(spec.shard_map)}) must be in "
                    f"[1, hardware.num_dpus={hardware.num_dpus}]"
                )
            if any(not 0 <= dpu_id < hardware.num_dpus for dpu_id in spec.shard_map):
                raise ValueError(
                    f"{node.name} shard_map 含越界 dpu_id: {sorted(spec.shard_map)}，"
                    f"hardware.num_dpus={hardware.num_dpus}"
                )
            landing_by_src = {
                e.src: e for e in node.meta.get(REDISTRIBUTE_META_KEY, [])
                if e.dst_loc.get("device") == DEVICE_DPU
            }
            for dpu_id in spec.shard_map:
                # 按节点参数顺序收集输入地址。`cat` 的第一个实参是**张量
                # 列表**，要摊平——不摊平它整段被跳过，内核一个输入都读不到。
                flat_args = _flatten_node_args(node.args)
                reads: list[Access] = []
                for arg in flat_args:
                    if not isinstance(arg, Node):
                        continue
                    if arg.name in landing_by_src and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", []):
                        detail = landing_by_src[arg.name].dst_spec.shard_map[dpu_id]
                        nbytes = dtype_bytes(landing_by_src[arg.name].dtype)
                        for dim in detail.local_shape:
                            nbytes *= dim
                        reads.append(Access(("dpu", dpu_id), detail.mram_offset, nbytes))
                        continue
                    arg_spec = arg.meta.get(SPEC_META_KEY)
                    if arg_spec is not None and arg_spec.device == DEVICE_DPU and dpu_id in arg_spec.shard_map:
                        reads.append(_node_access(arg, dpu_id, _placed_width(
                            arg, placed_elem_bytes)))
                if kv_access_of is not None:
                    hook = kv_access_of(node)
                    kv_reads, kv_writes = hook if hook is not None else ([], [])
                else:
                    kv_reads, kv_writes = [], []
                reads = reads + kv_reads
                write = _node_access(node, dpu_id, _placed_width(
                    node, placed_elem_bytes))
                waits = _deps_of(reads, builder.writers)
                waits += _war_waits(write.loc, write.offset, pending_readers, builder.reader_cmds)
                # 按节点参数顺序记录形状、dtype 和字面量参数。
                # `arg_dtypes` 逐参记输入 dtype，不能只有输出 dtype：
                # `to.dtype` 的 f16→f32 会按 f32 去读 f16 的缓冲，整块读错。
                arg_kinds = []
                arg_shapes = []
                arg_dtypes = []
                arg_decisions: list[tuple] = []
                for arg in flat_args:
                    if not isinstance(arg, Node):
                        arg_kinds.append(arg)
                        arg_shapes.append(None)
                        arg_dtypes.append(None)
                        continue
                    arg_kinds.append("tensor")
                    if arg.name in landing_by_src and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", []):
                        arg_spec_here = landing_by_src[arg.name].dst_spec
                    else:
                        arg_spec_here = arg.meta[SPEC_META_KEY]
                    arg_detail = arg_spec_here.shard_map[dpu_id]
                    arg_shapes.append(arg_detail.local_shape)
                    arg_dtypes.append(arg.meta[SPEC_META_KEY].dtype)
                    # 形状与切分决策取同一个 spec：发射侧按「同形状 ⇒ 同切分」
                    # 把编码一起贴上去，这里留下逐实参的决策供出口断言核对。
                    arg_decisions.append(
                        (arg.name, arg_detail.local_shape,
                         _shard_decision_of(arg_spec_here, arg_detail)))
                out_detail = spec.shard_map[dpu_id]
                _assert_same_shape_args_share_the_decision(
                    node, dpu_id, arg_decisions, spec, out_detail)
                payload = {
                    "kernel": str(node.target), "node": node.name,
                    "arg_kinds": arg_kinds, "arg_shapes": arg_shapes,
                    "arg_dtypes": arg_dtypes,
                    "dtype": spec.dtype,
                    "out_shape": out_detail.local_shape,
                    "hardware": hardware.to_payload(),
                    # 跨 DPU 放置决策（三档），下发时落成 pimir 的
                    # `#pim.placement` 与 `dpusPerDevice`。单 DPU 不给，
                    # 文本与改动前逐字节一致。
                    "shard": _shard_decision_of(spec, out_detail),
                    # 排布（Memory Layout 第 3 层）。与切分决策取自同一个
                    # `out_detail`：两者描述同一块结果张量，分头取值会算出
                    # 描述另一块张量的 `order`。
                    "elem_strides": list(out_detail.elem_strides),
                    # 地址与对齐（Memory Layout 第 2、4 层），与排布取自同一个分片。
                    "mram_offset": out_detail.mram_offset,
                    "align_bytes": out_detail.align_bytes,
                }
                # 注意力要读的设备侧 KV 区域，编译期才知道，在这里烘进命令——
                # 内核自己拿不到 `kv_specs`，而用模块级全局会在多个策略之间
                # 互相覆盖（实测三种策略连着编译时后面的把前面的盖掉）。
                if sdpa_info_of is not None:
                    info = sdpa_info_of(node)
                    if info is not None:
                        payload["sdpa"] = info
                cmd = builder.append(
                    "launch", dpu_id, payload,
                    reads, [write] + kv_writes, waits,
                    num_tasklets=num_tasklets,
                )
                # 登记本节点为输入地址读者。
                builder.register_reader(node.name, dpu_id, cmd.id)
            continue

        # 生成主机计算命令。
        cmd = _emit_host_op(builder, node, host_handler_of)
        builder.register_reader(node.name, None, cmd.id)

    return CompiledPlan(plan=ExecutionPlan(commands=builder.commands), output_cmd_id=output_cmd_id)
