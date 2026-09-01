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
from contracts.exec_plan import Access, Command, ExecutionPlan
from contracts.graph_meta import DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, REDISTRIBUTE_META_KEY, SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig

_REDISTRIBUTE_OP = {
    "all_reduce": "host_reduce",
    "all_gather": "host_concat",
    "all_to_all": "host_permute",
    "scatter": "host_slice",
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


def _node_access(node: Node, dpu_id: int) -> Access:
    """一个 DPU 张量节点在本 dpu 的地址区间（自身输出）。"""
    detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
    itemsize = node.meta["val"].element_size()
    nbytes = itemsize
    for dim in detail.local_shape:
        nbytes *= dim
    return Access(("dpu", dpu_id), detail.mram_offset, nbytes)


def _itemsize(dtype_name: str) -> int:
    return np.dtype(dtype_name).itemsize


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
    num_tasklets: int = 4,
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
                # 按节点参数顺序收集输入地址。
                reads: list[Access] = []
                for arg in node.args:
                    if not isinstance(arg, Node):
                        continue
                    if arg.name in landing_by_src and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", []):
                        detail = landing_by_src[arg.name].dst_spec.shard_map[dpu_id]
                        nbytes = _itemsize(landing_by_src[arg.name].dtype)
                        for dim in detail.local_shape:
                            nbytes *= dim
                        reads.append(Access(("dpu", dpu_id), detail.mram_offset, nbytes))
                        continue
                    arg_spec = arg.meta.get(SPEC_META_KEY)
                    if arg_spec is not None and arg_spec.device == DEVICE_DPU and dpu_id in arg_spec.shard_map:
                        reads.append(_node_access(arg, dpu_id))
                if kv_access_of is not None:
                    hook = kv_access_of(node)
                    kv_reads, kv_writes = hook if hook is not None else ([], [])
                else:
                    kv_reads, kv_writes = [], []
                reads = reads + kv_reads
                write = _node_access(node, dpu_id)
                waits = _deps_of(reads, builder.writers)
                waits += _war_waits(write.loc, write.offset, pending_readers, builder.reader_cmds)
                # 按节点参数顺序记录形状和字面量参数。
                arg_kinds = []
                arg_shapes = []
                for arg in node.args:
                    if not isinstance(arg, Node):
                        arg_kinds.append(arg)
                        arg_shapes.append(None)
                        continue
                    arg_kinds.append("tensor")
                    if arg.name in landing_by_src and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", []):
                        arg_shapes.append(landing_by_src[arg.name].dst_spec.shard_map[dpu_id].local_shape)
                    else:
                        arg_shapes.append(arg.meta[SPEC_META_KEY].shard_map[dpu_id].local_shape)
                out_detail = spec.shard_map[dpu_id]
                cmd = builder.append(
                    "launch", dpu_id,
                    {"kernel": str(node.target), "node": node.name, "arg_kinds": arg_kinds,
                     "arg_shapes": arg_shapes, "dtype": str(node.meta["val"].dtype).removeprefix("torch."),
                     "out_shape": out_detail.local_shape, "hardware": hardware.to_payload()},
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
