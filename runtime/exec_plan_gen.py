"""问题 6（编译期半）：`ExecutionPlan` 生成器（方案问题 6 三.(2)）。

`build_execution_plan` 遍历一张标注图（问题 1/2 产物）一次，把标注图 + 通信
计划表（问题 3）+ 内存蓝图（问题 8）三份编译期产物展开为一份线性命令 DAG：
每条命令带精确到地址区间的 `waits`，运行时（`runtime/executor.py`）只解释、
不再做任何切分或依赖推断。

核心依赖算法（`overlap`/`deps_of`/`war_waits`）照方案 1428-1524 行伪代码
实现。落地时对方案没写全的三处做了具体决定（都在函数 docstring 标注依据）：

1. **DPU 节点与 redistribute 消费方不是互斥分支**：伪代码 `if device=="dpu"
   / elif redistribute is not None` 隐含二者互斥，但真实图里一个节点可以
   同时是 DPU 计算节点、又因为某个输入要从别处重分布进来而带
   `meta["redistribute"]`（如 `linear(x, w1)` 里 `x` 从 host scatter 进来）。
   本实现对每个节点先处理它自己的 `redistribute` 列表（把输入搬到位），再按
   `device` 生成节点自身的命令。
2. **不重新拆解 redistribute 边的段级 DMA 调度**：方案伪代码把一条边展开成
   "多 dma_in → 一条 host 归约/拼接/重排/切片 → 多 dma_out"的逐段命令。这套
   收集/归约/回写的正确实现已经在 `comm/lowering.py`（问题 3，有完整数值
   测试）里，不重复实现。改为每条边生成**一条**粗粒度命令（`op` 取
   `host_reduce`/`host_concat`/`host_permute`/`host_slice`，对应
   `contracts/exec_plan.py` 的 `Command.op`），`payload["fn"]` 是运行时调用
   `comm.lowering.all_reduce`/`all_gather`/`all_to_all`/`scatter` 的闭包；
   `reads`/`writes` 仍按该边全部 segment 的地址区间精确计算，依赖精度不降级
   （问题 3 已经决定"怎么搬"，问题 6 只决定"这条搬运该等谁、该被谁等"）。
3. **host 节点的实参解析统一用 `torch.fx.node.map_arg`**：不按 op 类型手写
   分支（`cat` 的参数是 `list[Node]`、RoPE 的 `wrap_with_set_grad_enabled`
   一个参数是指向子图模块的 `get_attr`、embedding 权重是被 host 节点直接
   使用的常量 `get_attr`——手写分支会漏掉这些真实存在的形态）。统一用
   `_resolve_node(node)` 处理每一种输入 `Node`，返回 `(getter, dep_cmd_id)`：
   `get_attr` 编译期取真实属性闭包捕获，`dep_cmd_id=None`（常量不等待任何
   命令）；`placeholder` 运行时读 `hal.bound_value(name)`，`dep_cmd_id=None`
   （图输入不是某条命令的产出，由 `execute_plan` 在提交前绑定好）；其余节点
   查 `host_value_of` 换算出生产它的命令 id，`getter` 运行时调
   `hal.result_of(id)`。`map_arg` 保持参数原始嵌套结构（列表/字典），解析
   完直接 `node.target(*args, **kwargs)`——与 `torch.fx.Interpreter
   .call_function` 同一思路，不重新发明结构解析；scatter 边的 host 源同样
   经 `_resolve_node` 取值，不另写一套。

KV 读写、SDPA 专用 handler 通过两个可选回调钩子接入（方案要求"KV 读写必须
作为显式地址区间纳入 writers"，但没给出具体接口形态，这是本实现补的形态）：

- `kv_access_of(node) -> tuple[list[Access], list[Access]] | None`：非 None
  时，该节点对应 launch 命令的 `reads`/`writes` 分别并入返回值——K/V 投影
  kernel 把写入的 `(layer, head, k/v)` 区间放进 writes，attention 读 kernel
  把要读的历史 KV 区间放进 reads。
- `host_handler_of(node) -> Callable[[hal, cmd, args, kwargs], object] | None`：
  非 None 时该 host 节点的 `payload["fn"]` 用这个专用 handler（如 SDPA 的 KV
  感知重算），`args`/`kwargs` 已经是解析完的具体值（与通用路径共用同一套
  `resolve_node`，handler 不用重新处理"常量/占位/上游命令结果"这层区分）；
  否则退化为直接调用 `node.target(*args, **kwargs)`。
"""

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
    """`build_execution_plan` 的完整产物：命令 DAG + 运行时按需的取值元数据。"""

    plan: ExecutionPlan
    output_cmd_id: int  # 图 output 节点对应命令的 id（取最终结果，如 logits）


def _as_torch(x: object) -> object:
    """把上游命令的返回值转成 torch 张量（host 计算节点调 `node.target` 需要）。

    redistribute 命令（`comm.lowering` 的 all_reduce/all_gather/…）与阶段 B
    的 numpy kernel 都返回 `np.ndarray`；host 节点的 `node.target` 是真实
    torch 算子，要求输入是 `torch.Tensor`。非数组值（如另一个 host_op 直接
    返回 torch 结果）原样透传。
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def overlap(a: Access, b: Access) -> bool:
    """两个访问区间是否相交（方案 1429 行）。"""
    return a.loc == b.loc and a.offset < b.offset + b.length and b.offset < a.offset + a.length


def _deps_of(reads: list[Access], writers: dict[tuple, list[tuple[Access, int]]]) -> list[int]:
    """RAW：与任一读区间相交的历史写命令（方案 1432 行 `deps_of`）。"""
    return [cid for a in reads for (wa, cid) in writers.get(a.loc, []) if overlap(a, wa)]


def _war_waits(
    loc: tuple, offset: int, pending_readers: dict[tuple, list[str]], reader_cmds: dict[tuple, list[int]]
) -> list[int]:
    """WAR：目标地址旧值读者，节点名经 `reader_cmds` 翻译成命令 id（方案 1483 行）。"""
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
    """一次 `build_execution_plan` 调用的可变状态：命令表 + writers + reader_cmds。"""

    def __init__(self, gm_root, pending_readers: dict[tuple, list[str]]) -> None:
        self.gm_root = gm_root
        self.commands: list[Command] = []
        self.writers: dict[tuple, list[tuple[Access, int]]] = {}
        self.reader_cmds: dict[tuple, list[int]] = {}
        self.pending_readers = pending_readers
        self.host_value_of: dict[str, int] = {}  # 节点名 -> 提供其 host 可见值的命令 id
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
        """登记"读者 `reader_name` 在 dpu_id 上展开成了命令 cmd_id"。

        键必须是**读者自己的名字**，不是它读的那个张量的名字——问题 8 的
        `pending_readers` 给出的正是"这个地址的旧值还有哪些**读者节点**没执行
        完"（`transient_tensors` 里 `readers` 收的是 `node.users` 的名字与
        `redist:e{edge_id}` 标记），`_war_waits` 拿这些读者名字来查命令编号。
        早先这里错按"被读张量名"建键，两张表方向相反、永远 join 不上，导致
        **全部 WAR 依赖被静默丢弃**（实测 14 万次查询 89% 命中不到），并发下
        就会出现"新命令覆盖了旧值、而旧值的读者还没读"的真实数据损坏。

        同一读者在张量并行下会展开成多条命令（每台 DPU 一条），因此还要按
        `dpu_id=None` 再登记一份：跨 DPU 的 redistribute 命令与 host 命令
        写某台 DPU 的地址时，要等的是该读者在**那台 DPU 上**的命令；而
        `pending_readers` 的键里 `loc[1]` 就是那台 DPU 的编号，所以按
        (读者名, dpu_id) 与 (读者名, None) 两种键都登记，查得到即可。
        """
        self.reader_cmds.setdefault((reader_name, dpu_id), []).append(cmd_id)
        if dpu_id is not None:
            self.reader_cmds.setdefault((reader_name, None), []).append(cmd_id)

    def resolve_node(self, node: Node) -> tuple[Callable[[object], object], int | None]:
        """把一个输入 `Node` 解析成 (运行时取值闭包, 依赖的命令 id 或 None)（设计决策 3）。"""
        if node.op == "get_attr":
            obj = self.gm_root
            for part in str(node.target).split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.Tensor):
                obj = obj.detach()  # 常量权重/buffer：脱离 autograd 图，host 算子拿到能直转 numpy 的张量
            return (lambda hal: obj), None
        if node.op == "placeholder":
            name = node.name
            return (lambda hal: hal.bound_value(name)), None
        cmd_id = self.host_value_of[node.name]
        return (lambda hal: _as_torch(hal.result_of(cmd_id))), cmd_id


def _emit_redistribute(builder: _PlanBuilder, edge, entry: CommPlanEntry, src_node: Node) -> None:
    """一条 redistribute 边 → 一条粗粒度命令（设计决策 2，见模块 docstring）。

    `payload["fn"]` 在运行时被 `NumpyBackend._run_host_or_dma` 以
    `fn(hal, cmd)` 调用，内部用 `comm.lowering` 的已验证原语跑完整段级收集/
    归约/回写；`reads`/`writes` 独立按段地址算好，供 RAW/WAR 依赖分析使用。
    回写目标为 host 时（`dst_loc.device=="host"`）结果只落 host，供后续
    消费节点经 `host_value_of` 查询取值。
    """
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
    # 本条搬运命令读取源张量，问题 8 的 `transient_tensors` 把这次读取记作
    # 读者 `redist:e{edge_id}`（`node.users` 覆盖不到 copy_from）。按同一个
    # 标记登记命令编号，后续命令覆盖该地址时才等得到这条搬运（否则 WAR
    # 依赖丢失，实测这类占被丢弃依赖的三分之一）。
    builder.register_reader(f"redist:e{edge.edge_id}", None, cmd.id)
    for w in writes:
        builder.register_reader(f"redist:e{edge.edge_id}", w.loc[1], cmd.id)


def _emit_host_op(builder: _PlanBuilder, node: Node, host_handler_of: HostHandlerFn | None) -> Command:
    """host 节点 → 一条 `host_op`（设计决策 3：`map_arg`/`map_aggregate` 统一解析实参）。

    专用 handler（`host_handler_of` 命中时）签名为
    `handler(hal, cmd, args, kwargs) -> object`——`args`/`kwargs` 已经是解析
    完的具体值（与通用路径共用同一套 `resolve_node`），handler 只管算，不用
    重新处理"这个参数是常量/占位/上游命令结果"这层。
    """
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
    """问题 6 编译期主入口：标注图 + 通信计划表 + pending_readers → ExecutionPlan。

    入: nodes —— 一张标注图（prefill 或 decode）的 `list(gm.graph.nodes)`；
    gm_root —— 该图的 `GraphModule`（取 `get_attr` 常量真实值用）；
    comm_entries —— `{edge_id: CommPlanEntry}`（问题 3 产物按 edge_id 建的
    索引）；pending_readers —— 该图对应的那份（问题 8 `DPUPlan
    .pending_readers_prefill`/`_decode`）；hardware —— 共享硬件契约，
    每条 `launch` 命令都会写入 `payload["hardware"]`；num_tasklets —— 每个
    DPU launch 命令内部按几个 tasklet 顺序模拟切分（默认 4，全图统一一个
    数字，见 `contracts/exec_plan.py::Command.num_tasklets` 的说明；不做
    per-op 不同 tasklet 数，那需要代价模型才有意义，属于第 2 阶段范畴）。
    出: CompiledPlan（`plan.commands` 按生成顺序即拓扑序排列）。
    """
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
            # 死端 guard（`_assert_tensor_metadata` 是 call_function 形态，
            # `_guards_fn` 是 call_module 形态——两者都无张量输出、无消费者，
            # 不参与数据流）；`propagate_specs` 本就不为它打 SPEC_META_KEY/
            # redistribute（问题 2 docstring 同一处说明："val 为 None 的是
            # ... 死端 guard"，未限定 op 类型），这里同样跳过，不生成任何
            # 命令——它检查的是 torch.export 阶段就已保证的算子/守卫元数据，
            # 与 PIM 侧的数值执行无关。
            continue

        # 1. 先把本节点入边里的 redistribute 边搬完（设计决策 1）。
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            _emit_redistribute(builder, edge, comm_entries[edge.edge_id], by_name[edge.src])

        if node.op == "output":
            # 图出口视为 host 消费方：直接查最后一次落 host 的值对应命令。
            (out_arg,) = node.args
            src = out_arg[0] if isinstance(out_arg, (list, tuple)) else out_arg
            output_cmd_id = builder.host_value_of.get(src.name, output_cmd_id)
            continue

        if node.meta.get(DEVICE_META_KEY) == DEVICE_DPU:
            spec = node.meta[SPEC_META_KEY]
            # 分片数可以少于 num_dpus：流水切分下一个张量只落在本 stage 的
            # tp_width 台上，不是全部 DPU 上都有一份（张量并行下两者相等）。
            # 仍然校验「不超过总台数」与「每个编号合法」——越界的 dpu_id 会让
            # 下游按不存在的地址空间生成命令。
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
                # reads 按 node.args 原始顺序排列（DPU 白名单算子 linear/add/
                # mul/tanh 的参数天然扁平，不含嵌套容器），供 kernel 按位置
                # 索引取值——第 1 个位置参数对应 reads[0]，以此类推。走
                # redistribute 落地的输入读落地缓冲地址，不读原节点地址。
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
                # arg_kinds 按 node.args 顺序标出每个位置是 "tensor"（对应
                # reads 里下一个区间，形状/dtype 见 arg_shapes/arg_dtype）还是
                # 字面量本身（如 RMSNorm 的 eps）——kernel（阶段 B）靠这个把
                # reads 与非张量参数重新拼成完整调用签名，不必对每个算子硬编码
                # 参数个数。张量参数的 shape 取自其 spec 的本地分片 shape
                # （问题 2 产物，与 reads 里对应区间的字节数互相印证）。
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
                # 登记"本节点（作为读者）在这台 DPU 上就是这条命令"，供后续
                # 命令的 WAR 查询（`pending_readers` 给的是读者节点名）。
                builder.register_reader(node.name, dpu_id, cmd.id)
            continue

        # host 计算节点。
        cmd = _emit_host_op(builder, node, host_handler_of)
        builder.register_reader(node.name, None, cmd.id)

    return CompiledPlan(plan=ExecutionPlan(commands=builder.commands), output_cmd_id=output_cmd_id)
