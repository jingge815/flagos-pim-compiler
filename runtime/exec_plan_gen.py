"""将标注图、通信计划和内存蓝图展开为 `ExecutionPlan`。"""

from __future__ import annotations

import os
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
from contracts.graph_meta import (
    DEVICE_DPU, DEVICE_HOST, DEVICE_META_KEY, FUSED_TAIL_META_KEY,
    REDISTRIBUTE_META_KEY, SPEC_META_KEY)
from contracts.op_contract import DpuShard, PIMHardwareConfig
from contracts.op_semantics import RMS_NORM_KERNEL
from contracts.unified_ir import ABSORBED_META_KEY, RMS_NORM_META_KEY, SHAPE_EXPR_META_KEY
from graph.spec_prop import _layer_of_node
from runtime.dependency import deps_of as _deps_of, overlap, war_waits as _war_waits
from runtime.shape_bind import dynamic_shape_enabled


def _comm_rows(hal) -> int | None:
    """通信搬运的行数钳制：仅形状绑定开启时按真实 token 数收缩。

    非绑定模式下内核按包络形状读 DPU，通信先裁剪会让尾行读到旧值；
    绑定模式内核只读实际 t 行，裁剪掉的尾行不会被消费（4 层真实权重
    gather/scatter/组合三方向均已实测逐 token 一致）。
    """
    if not dynamic_shape_enabled():
        return None
    rows = hal.bound_num_tokens
    return int(rows) if rows else None

# MoE 执行语义。图编译器与 GeneSim 共用这两个取值；默认跟随 PIM_MOE_DISPATCH。
MOE_EXECUTION_MODES = ("dense_sim", "dynamic_dispatch")


def moe_execution_default() -> str:
    """编译配置的默认执行语义：PIM_MOE_DISPATCH=0 退回稠密模拟。"""
    return "dense_sim" if os.environ.get("PIM_MOE_DISPATCH", "1") == "0" \
        else "dynamic_dispatch"


def is_moe_config(cfg) -> bool:
    """模型是否为 MoE：由 model_type 与有效 MoE 字段共同判定。

    不能用「专家数 > 1」当代替：1 专家/top-1 是合法的 MoE 配置，按阈值
    判定会让它丢失执行语义、MoE 元数据与默认分块策略。
    """
    if str(getattr(cfg, "model_type", "") or "") == "qwen3_moe":
        return True
    return (
        int(getattr(cfg, "num_experts", 0) or 0) > 0
        or int(getattr(cfg, "moe_intermediate_size", 0) or 0) > 0)


def resolve_moe_execution(cfg, requested: str | None = None) -> str:
    """按模型配置确定执行语义；非 MoE 模型不得带语义，未知取值直接报错。

    这一份值由 `compile_llama2` 同时写入执行计划、放置 sidecar 与仿真 IR，
    各入口不再各自读环境变量或硬编。
    """
    if not is_moe_config(cfg):
        if requested not in (None, ""):
            raise ValueError(
                f"非 MoE 模型不应指定 moe_execution={requested!r}")
        return ""
    model_type = str(getattr(cfg, "model_type", "") or "")
    if model_type != "qwen3_moe":
        raise ValueError(
            f"model_type={model_type!r} 带 MoE 配置，但只支持 qwen3_moe 的"
            "执行语义；不支持的模型类型不能静默按 dense 处理。")
    num_experts = int(getattr(cfg, "num_experts", 0) or 0)
    top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    if num_experts < 1:
        raise ValueError(f"qwen3_moe 的 num_experts={num_experts} 必须为正整数")
    if not 1 <= top_k <= num_experts:
        raise ValueError(
            f"num_experts_per_tok={top_k} 越界 [1, {num_experts}]")
    mode = moe_execution_default() if requested is None else requested
    if mode not in MOE_EXECUTION_MODES:
        raise ValueError(
            f"未知 MoE 执行语义 {mode!r}：只认 {MOE_EXECUTION_MODES}")
    return mode


def _moe_layer_of(node: Node, marker: str) -> int | None:
    """沿上游找第一个含 marker 的 get_attr 权重，返回所属层号。"""
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "get_attr" and marker in str(cur.target):
            part = str(cur.target).split(".")
            return int(part[part.index("layers") + 1])
        # 用 all_input_nodes 而不是只扫 args：分块后的 `cat([a, b], dim)` 把
        # 上游包在列表里，扫位置实参会整段跳过，层号就找不到了。
        stack.extend(cur.all_input_nodes)
    return None


def _expert_of_weight(node: Node) -> tuple[int, int] | None:
    """专家权重 get_attr 的 (layer, expert)；形如 layers.L.mlp.experts.E.*。"""
    if node.op != "get_attr" or ".mlp.experts." not in str(node.target):
        return None
    part = str(node.target).split(".")
    return int(part[part.index("layers") + 1]), int(part[part.index("experts") + 1])


def _node_layer_of(node: Node, provenance: dict[str, tuple | None]) -> int | None:
    """节点的层号：优先取权重名/模块路径，其次取输入标签的唯一层号。

    分块会在专家子图里插入切片与类型转换，它们没有模块路径；这些节点只能
    从输入继承层号。输入来自不同层时说明是层边界（残差、MoE 出口），返回
    None，跨层激活按中性输入处理。
    """
    layer = _layer_of_node(node)
    if layer is not None:
        return layer
    layers = {
        value[0]
        for arg in node.all_input_nodes
        if isinstance(value := provenance.get(arg.name), tuple)
    }
    return layers.pop() if len(layers) == 1 else None


def _expert_provenance(nodes: list[Node]) -> dict[str, tuple | None]:
    """每个节点在本层内的专家归属：(layer, expert) 或 (layer, None)。

    (layer, None) 表示「本层多个专家的汇合点」（累加链一类），它没有单一
    专家身份，执行计划不给它 gate。分块会在权重与 linear 之间插入切片/类型
    转换，激活链上还有 silu/mul/cat 等无权重节点，所以按拓扑序把输入的专家
    归属并上来，整条只由一个专家喂出来的子图都能识别。

    传播域限定在「本层」：上一层的输出、甚至上一层的多专家累加状态，都是
    下一层的中性激活输入，不得与下一层专家权重相遇后污染候选。层号取模块
    路径（torch.export 的 `nn_module_stack`）或权重名，两者都没有时从输入
    继承；输入跨层即层边界，归属到此终止。
    """
    provenance: dict[str, tuple | None] = {}
    for node in nodes:
        own = _expert_of_weight(node)
        if own is not None:
            provenance[node.name] = own
            continue
        layer = _node_layer_of(node, provenance)
        values = [
            value
            for arg in node.all_input_nodes
            if isinstance(value := provenance.get(arg.name), tuple)
            and (layer is None or value[0] == layer)
        ]
        if not values:
            provenance[node.name] = None
            continue
        first = values[0]
        if all(value == first for value in values):
            provenance[node.name] = first
        elif layer is not None:
            # 同一层里多个专家汇合：是共享累加这类出口，没有单一专家身份。
            provenance[node.name] = (layer, None)
        else:
            # 层号未知又出现不同来源：跨层汇合，按中性输入处理。
            provenance[node.name] = None
    return provenance


def _expert_tag_of(provenance: dict[str, tuple | None], node: Node) -> tuple[int, int] | None:
    """节点的唯一专家归属；多专家或无归属返回 None。"""
    value = provenance.get(node.name)
    return value if isinstance(value, tuple) and value[1] is not None else None


def _emit_moe_dispatch(builder, layer: int, route_cmd_id: int) -> int:
    """发一个 dispatch host_op：读 top-k 选中掩码，把本步无 token 命中的专家
    写进跳过集。

    命中条件只能是选中索引生成的布尔掩码：路由权重在 softmax/重归一化后
    可能下溢为 0，用它判命中会把已选中的专家错误跳过。跳过集键
    (lane_key, 本命令号)：并发窗口各写各的，互不覆盖；被跳过的专家 launch
    已 waits 本命令，读到的一定是本步的结果。
    """
    def fn(hal, cmd):
        selected_mask = hal.result_of(route_cmd_id)
        lane = hal.current_lane_key()
        hit = selected_mask.any(dim=0)
        hal.gate_skips[(lane, cmd.id)] = {
            int(i) for i in (~hit).nonzero().flatten().tolist()}

    return builder.append(
        "host_op", None, {"node": f"moe_dispatch_{layer}", "fn": fn},
        [], [], [route_cmd_id]).id

def _skip_zero_nodes(nodes: list[Node], provenance: dict[str, object]) -> set[str]:
    """哪些专家节点的输出会被专家子图之外、不会一起跳过的命令读到。

    跳过发生在提交前，未命中的命令不再入队。子图内部的读者（同一 expert）
    会一起被跳过，不需要清零；一旦有跨出子图的读者（例如未门控的累加链、
    残差），旧值就会被读走，这条命令必须自己把输出写零。
    """
    needs_zero: set[str] = set()
    for node in nodes:
        tag = _expert_tag_of(provenance, node)
        if tag is None:
            continue
        for user in node.users:
            if _expert_tag_of(provenance, user) != tag:
                needs_zero.add(node.name)
                break
    return needs_zero


_REDISTRIBUTE_OP = {
    "all_reduce": "host_reduce",
    "all_gather": "host_concat",
    "all_to_all": "host_permute",
    "scatter": "host_slice",
    "local_slice": "dpu_slice",
}
_REDISTRIBUTE_FN = {"all_reduce": all_reduce, "all_gather": all_gather, "all_to_all": all_to_all}

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


def _expr_dicts(exprs) -> list | None:
    """把节点上的形状表达式元组转成可序列化的字典列表；无则 None。"""
    if not exprs:
        return None
    return [expr.to_dict() if expr is not None else None for expr in exprs]


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
    if node.target is torch.ops.aten.cat.default:
        # 拼接的每个实参只是一段，不是输出本身；与本地输出同形只是本台恰好
        # 持有一段与它等长。MLIR 要求同值同型，编码只能跟着结果类型一起贴，
        # B 路没有按轴读这段编码的消费者，不产生数值后果。
        return
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


def _weight_slice_local_bounds(node: Node, dpu_id: int) -> tuple[int, int] | None:
    """权重切片在这台上的本地下标区间，越界的部分夹到本地权重内。

    只认切的是权重、且切的就是权重分片那一维的切片。切的是激活、或者切
    别的轴时返回 None，下标保持原样——另一根轴上每台都持有全段，平移会
    切到不存在的行。平移后还要夹到这台持有的长度，否则命令的起止会落到
    本地权重之外，靠内核越界夹取掩盖规格与命令的不一致。
    """
    if node.target is not torch.ops.aten.slice.Tensor:
        return None
    src = node.args[0] if node.args else None
    # 两维都切时 K 那一刀挂在 N 那一刀上；N 不是分片轴，本地仍是整段 K，
    # 所以按根权重的分片换算。
    while (isinstance(src, Node) and src.target is torch.ops.aten.slice.Tensor
           and src.args):
        src = src.args[0]
    if not isinstance(src, Node) or src.op != "get_attr":
        return None
    spec = src.meta.get(SPEC_META_KEY)
    if spec is None or dpu_id not in spec.shard_map:
        return None
    dim = int(node.args[1]) if len(node.args) > 1 else 0
    if dim < 0:
        dim += src.meta["val"].ndim
    if spec.placement.kind != "Shard" or spec.placement.dim != dim:
        return None
    detail = spec.shard_map[dpu_id]
    start = node.args[2] if len(node.args) > 2 and node.args[2] is not None else 0
    end = node.args[3] if len(node.args) > 3 else None
    end = detail.end_idx if end is None else int(end)
    return (max(int(start) - detail.start_idx, 0),
            min(end - detail.start_idx, detail.local_shape[dim]))


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
        self.lane_of: dict[int, int | None] = {}  # 命令所属执行泳道：DPU 号或 None。
        self._next_id = 0

    def append(self, op: str, dpu_id: int | None, payload: dict,
               reads: list[Access], writes: list[Access], waits: list[int],
               num_tasklets: int = 1) -> Command:
        # 写后写：等所有写过同一区间的旧命令，否则旧写者晚跑会覆盖新值。
        waits = waits + _deps_of(writes, self.writers)
        # 同一 DPU 的命令由执行后端按提交序串行，等该 DPU 上最后一条前驱即可；
        # host 命令之间没有这种串行关系，逐条保留。
        last: dict[tuple, int] = {}
        for cid in waits:
            lane = self.lane_of[cid]
            key = ("dpu", lane) if lane is not None else ("host", cid)
            if cid > last.get(key, -1):
                last[key] = cid
        cmd = Command(id=self._next_id, op=op, dpu_id=dpu_id, payload=payload,
                       reads=reads, writes=writes, waits=sorted(last.values()),
                       num_tasklets=num_tasklets)
        self._next_id += 1
        self.commands.append(cmd)
        self.lane_of[cmd.id] = dpu_id
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
            # 绑定值是 numpy 时转成 torch：主机节点（如融合闭包）里的
            # aten 算子只认张量，其余读取路径都已过 `_as_torch`。
            return (lambda hal: _as_torch(hal.bound_value(name))), None
        cmd_id = self.host_value_of[node.name]
        return (lambda hal: _as_torch(hal.result_of(cmd_id))), cmd_id


def _dma_engine(hal, cmd) -> DmaEngine:
    """按命令上的窗口信息构造 DMA 引擎；原计划没有这些字段时零平移。"""
    return DmaEngine(
        hal.dpu_set,
        act_base_by_dpu=cmd.payload.get("act_base_by_dpu"),
        act_shift=cmd.payload.get("act_shift", 0),
    )


def _emit_redistribute(
    builder: _PlanBuilder, edge, entry: CommPlanEntry, src_node: Node,
    dispatch_by_layer: dict[int, int] | None = None,
    provenance: dict[str, object] | None = None,
) -> None:
    """将一条重分布边生成包含全部段访问的命令。

    专家子图专属的通信带上 gate/expert：未命中专家时连归约/收集一起跳过，
    否则动态分派只省计算、不省通信。
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
            engine = _dma_engine(hal, cmd)
            scatter(entry, engine, get_host_buf(hal), rows=_comm_rows(hal))
            return None
    elif edge.type == "local_slice":
        def fn(hal, cmd):
            engine = _dma_engine(hal, cmd)
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
            engine = _dma_engine(hal, cmd)
            return primitive(entry, engine, rows=_comm_rows(hal))

    op = _REDISTRIBUTE_OP[edge.type]
    payload: dict = {"edge_id": edge.edge_id, "fn": fn}
    if dispatch_by_layer is not None and provenance is not None:
        tag = _expert_tag_of(provenance, src_node)
        if tag is not None and tag[0] in dispatch_by_layer:
            payload["gate"] = dispatch_by_layer[tag[0]]
            payload["expert"] = tag[1]
            # 跳过时若这条命令的返回值被主机闭包消费，要给零张量而不是 None：
            # 未命中专家的贡献就是零，主机侧融合的乘加照常算出一个正确的累加。
            if edge.dst_loc.get("device") == DEVICE_HOST:
                payload["skip_value"] = [
                    list(entry.shape), np.dtype(entry.dtype).name]
            waits = waits + [payload["gate"]]
    cmd = builder.append(op, None, payload, reads, writes, waits)
    if edge.dst_loc.get("device") == DEVICE_HOST:
        builder.host_value_of[edge.src] = cmd.id
    # 将重分布命令登记为源张量读者。
    builder.register_reader(f"redist:e{edge.edge_id}", None, cmd.id)
    for w in writes:
        builder.register_reader(f"redist:e{edge.edge_id}", w.loc[1], cmd.id)
    # 读区也要登记：内存规划的 WAR 链把重分布边记作 `redist:eNN` 读者名，
    # 按（读者名, 读取 DPU）查命令号；原先只登记 (名, None)，键形对不上，
    # 激活区复用的后写者查不到——dpu→host 收集是 host 命令，不在 DPU 泳道
    # 链上，没有这条 WAR 就裸奔（K3 竞态，≥8 worker 下必现）。
    for r in reads:
        builder.register_reader(f"redist:e{edge.edge_id}", r.loc[1], cmd.id)


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


def _fused_activation_of(node: Node) -> str | None:
    """linear 节点折进主算子的尾部激活种类；没折返回 None。"""
    tail = node.meta.get(FUSED_TAIL_META_KEY)
    if tail is None or node.target is not torch.ops.aten.linear.default:
        return None
    return str(tail.activation)


def build_execution_plan(
    nodes: list[Node],
    gm_root,
    comm_entries: dict[int, CommPlanEntry],
    pending_readers: dict[tuple, list[str]],
    *,
    hardware: PIMHardwareConfig,
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
    emitted_edges: set[int] = set()
    # MoE 动态分派：层号 → 本层 dispatch host_op 的命令号。
    dispatch_by_layer: dict[int, int] = {}
    # 每个节点的唯一专家归属，供整条专家子图（含分块/通信）统一打标。
    provenance = _expert_provenance(nodes)
    # 只有输出会跨出专家子图的命令才需要在跳过时写零；子图内部的读者会
    # 一起被跳过，给每条命令都清零只是把省下的调度成本又花了回去。
    skip_zero_nodes = _skip_zero_nodes(nodes, provenance)
    for node in nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op != "output" and node.meta.get("val") is None:
            # 无张量输出的节点不生成命令。
            continue

        # 处理节点输入的重分布。同一条边挂在每个消费者上（spec_prop 去重），
        # 命令只发一次：重复发会在后面的消费者处重读源分片，而源缓冲彼时
        # 已被复用覆盖，读到的全是垃圾。
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.edge_id in emitted_edges:
                continue
            emitted_edges.add(edge.edge_id)
            _emit_redistribute(
                builder, edge, comm_entries[edge.edge_id], by_name[edge.src],
                dispatch_by_layer, provenance)

        if node.op == "output":
            # 图输出使用最后一次写入 host 的命令结果。
            (out_arg,) = node.args
            src = out_arg[0] if isinstance(out_arg, (list, tuple)) else out_arg
            output_cmd_id = builder.host_value_of.get(src.name, output_cmd_id)
            continue

        # 已折进锚点的算子不再发命令，只留锚点那一条。
        if node.meta.get(ABSORBED_META_KEY):
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
            # 融合锚点的实参是链首输入和缩放权重，不是它自己那两个操作数。
            fusion = node.meta.get(RMS_NORM_META_KEY)
            for dpu_id in spec.shard_map:
                # 按节点参数顺序收集输入地址。`cat` 的第一个实参是**张量
                # 列表**，要摊平——不摊平它整段被跳过，内核一个输入都读不到。
                flat_args = ([fusion.eaten[0].args[0], fusion.weight_node]
                             if fusion is not None else _flatten_node_args(node.args))
                reads: list[Access] = []
                for arg in flat_args:
                    if not isinstance(arg, Node):
                        continue
                    if arg.name in landing_by_src and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", []):
                        # 有落地边时读落地规格，`cat` 也一样：部分和归约后的值在
                        # 落地缓冲里，源缓冲是归约前的。只有对齐或堆叠的切分 cat
                        # 没有落地边，那时走下面按源规格读、不持有的段跳过。
                        detail = landing_by_src[arg.name].dst_spec.shard_map[dpu_id]
                        nbytes = dtype_bytes(landing_by_src[arg.name].dtype)
                        for dim in detail.local_shape:
                            nbytes *= dim
                        reads.append(Access(("dpu", dpu_id), detail.mram_offset, nbytes))
                        continue
                    arg_spec = arg.meta.get(SPEC_META_KEY)
                    if (arg_spec is not None and arg_spec.device == DEVICE_DPU
                            and dpu_id in arg_spec.shard_map):
                        reads.append(_node_access(arg, dpu_id, _placed_width(
                            arg, placed_elem_bytes)))
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
                # 权重切片的起止是全局下标，每台的本地权重只是全局的一段。
                # 下发前换算成本地区间并夹到本地长度，否则每台都用全局下标
                # 切自己的本地权重，越界的被夹成空、没越界的读到别人的行。
                slice_bounds = _weight_slice_local_bounds(node, dpu_id)
                for index, arg in enumerate(flat_args):
                    if not isinstance(arg, Node):
                        if slice_bounds is not None and index in (2, 3):
                            arg = slice_bounds[index - 2]
                        arg_kinds.append(arg)
                        arg_shapes.append(None)
                        arg_dtypes.append(None)
                        continue
                    held = arg.meta.get(SPEC_META_KEY)
                    if (held is not None and held.placement.kind == "Partial"
                            and dpu_id not in held.shard_map):
                        # 部分和只落在部分台上，这台的贡献是 0。
                        arg_kinds.append(0.0)
                        arg_shapes.append(None)
                        arg_dtypes.append(None)
                        continue
                    arg_kinds.append("tensor")
                    if (arg.name in landing_by_src
                            and dpu_id in landing_by_src[arg.name].dst_loc.get("dpus", [])):
                        arg_spec_here = landing_by_src[arg.name].dst_spec
                    else:
                        arg_spec_here = arg.meta[SPEC_META_KEY]
                    # 拼接的某段这台并不持有：那段的列在别的台上，这里跳过，
                    # 只拼这台自己的列。reads 那边本来就按持有者过滤，两边一致。
                    if (node.target is torch.ops.aten.cat.default
                            and dpu_id not in arg_spec_here.shard_map):
                        arg_kinds.pop()
                        continue
                    arg_detail = arg_spec_here.shard_map[dpu_id]
                    arg_shapes.append(arg_detail.local_shape)
                    arg_dtypes.append(arg.meta[SPEC_META_KEY].dtype)
                    # 形状与切分决策取同一个 spec：发射侧按「同形状 ⇒ 同切分」
                    # 把编码一起贴上去，这里留下逐实参的决策供出口断言核对。
                    decision = _shard_decision_of(arg_spec_here, arg_detail)
                    # 只落在部分台上的部分和，缺的台贡献 0，归约跨的仍是输出那些台。
                    if (arg_spec_here.placement.kind == spec.placement.kind == "Partial"
                            and set(arg_spec_here.shard_map) < set(spec.shard_map)):
                        decision = _shard_decision_of(spec, spec.shard_map[dpu_id])
                    arg_decisions.append(
                        (arg.name, arg_detail.local_shape, decision))
                out_detail = spec.shard_map[dpu_id]
                _assert_same_shape_args_share_the_decision(
                    node, dpu_id, arg_decisions, spec, out_detail)
                payload = {
                    "kernel": RMS_NORM_KERNEL if fusion is not None else str(node.target),
                    "node": node.name,
                    # 折进主算子的尾部激活：A 路编译内核按它带 ACTIVATION
                    # 常量，numpy 镜像按它补上激活再写回。
                    "activation": _fused_activation_of(node),
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
                if fusion is not None:
                    payload["epsilon"] = fusion.epsilon
                # 动态形状（D2）：节点带符号表达式时随命令下发，供运行期按
                # 实际 T 绑定形状；静态导出没有这个 meta，payload 保持原样。
                out_expr = node.meta.get(SHAPE_EXPR_META_KEY)
                if out_expr is not None or any(
                        arg.meta.get(SHAPE_EXPR_META_KEY) is not None
                        for arg in flat_args if isinstance(arg, Node)):
                    arg_exprs = [
                        _expr_dicts(arg.meta.get(SHAPE_EXPR_META_KEY))
                        for arg in flat_args if isinstance(arg, Node)]
                    payload["arg_shape_exprs"] = arg_exprs
                    payload["out_shape_expr"] = _expr_dicts(out_expr)
                # 注意力要读的设备侧 KV 区域，编译期才知道，在这里烘进命令——
                # 内核自己拿不到 `kv_specs`，而用模块级全局会在多个策略之间
                # 互相覆盖（实测三种策略连着编译时后面的把前面的盖掉）。
                if sdpa_info_of is not None:
                    info = sdpa_info_of(node)
                    if info is not None:
                        payload["sdpa"] = info
                # MoE 动态分派：专家子图里的每条命令都打上 (gate, expert) 标签，
                # 并把 dispatch 加进 waits——运行期先看跳过集再决定发不发内核。
                expert = _expert_tag_of(provenance, node)
                if expert is not None and expert[0] in dispatch_by_layer:
                    payload["gate"] = dispatch_by_layer[expert[0]]
                    payload["expert"] = expert[1]
                    waits = waits + [payload["gate"]]
                    if node.name in skip_zero_nodes:
                        payload["skip_zero"] = True
                cmd = builder.append(
                    "launch", dpu_id, payload,
                    reads, [write], waits,
                    num_tasklets=num_tasklets,
                )
                # 登记本节点为输入地址读者。
                builder.register_reader(node.name, dpu_id, cmd.id)
            continue

        # 生成主机计算命令。
        cmd = _emit_host_op(builder, node, host_handler_of)
        builder.register_reader(node.name, None, cmd.id)
        # MoE 门控的 scatter 落地后紧跟一个 dispatch：只有 top-k 选中掩码
        # （布尔输出）才是命中来源；权重 scatter 的输出只用于加权，不能
        # 再发一个 dispatch 覆盖同一层的门控。
        # dtype 从统一 IR 的 spec 读（`meta["val"]` 的 dtype 只允许在
        # spec_prop 里进入 IR）；掩码 scatter 是唯一的布尔 scatter。
        spec = node.meta.get(SPEC_META_KEY)
        if (node.target in (torch.ops.aten.scatter.src,
                            torch.ops.aten.scatter.value)
                and spec is not None and spec.dtype == "bool"):
            layer = _moe_layer_of(node, "mlp.gate.weight")
            if layer is not None:
                dispatch_by_layer[layer] = _emit_moe_dispatch(builder, layer, cmd.id)

    return CompiledPlan(plan=ExecutionPlan(commands=builder.commands), output_cmd_id=output_cmd_id)
