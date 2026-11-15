"""将模型、切分策略和硬件配置编译为可执行蓝图。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.fx import GraphModule

from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import DpuShard, PIMHardwareConfig
from contracts.pim_tensor_spec import RedistributeEdge
from contracts.runtime_graph import ShapeExpr
from contracts.unified_ir import (
    ARG_EXPR_META_KEY, SHAPE_EXPR_META_KEY, STAGE_PLANNED, VAL_META_KEY,
    mark_stage)
from graph.fuse_pim import fuse_rms_norms
from graph.opt_passes import fuse_linear_activation, optimize_graph
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from graph.strategy import ShardStrategy
from memory.kv_layout import KVRegionSpec, kv_specs_from_strategy
from memory.mem_planner import (
    DPUPlan, HwBudget, fit_activation_windows, fit_pool_slots, plan_dpu)
from runtime.binding import bind_plan
from runtime.exec_plan_gen import (
    CompiledPlan, build_execution_plan, is_moe_config,
    resolve_moe_execution)


@dataclass
class CompiledModel:
    """保存标注图、内存和通信蓝图，以及 prefill 和 decode 命令计划。"""

    strategy: ShardStrategy
    prefill_gm: GraphModule
    decode_gm: GraphModule
    prefill_edges: list[RedistributeEdge]
    decode_edges: list[RedistributeEdge]
    kv_specs: dict[int, KVRegionSpec]
    mem_plans: dict[int, DPUPlan]
    prefill: CompiledPlan
    decode: CompiledPlan
    hw: HwBudget
    hardware: PIMHardwareConfig
    num_request_slots: int = 1
    kv_slot_tokens: int = 16
    kv_pool_slots: int = 0
    prefill_seq_len: int = 0
    num_parallel_slots: int = 1            # 同一步并发执行的请求窗口数。
    act_stride: int = 0                    # 相邻激活窗口的统一步长。
    block_table_bytes: int = 0             # 单份块表窗口的字节数。
    prefill_seq_lens: tuple[int, ...] = ()        # 可用的 prefill 桶（升序）。
    prefill_plans: dict[int, list] | None = None  # 桶长 → 各窗口的 prefill 计划。
    prefill_cmd_ids: dict[int, int] | None = None  # 桶长 → 输出命令编号。
    decode_plans: list | None = None
    # MoE 执行语义：dense_sim / dynamic_dispatch；非 MoE 模型为空串。
    # 执行后端未显式配置时按它决定是否执行动态分派。
    moe_execution: str = ""
    # 动态图路径：static 走现有静态蓝图；dynamic 由 RuntimeGraph 绑定执行。
    compile_mode: str = "static"
    runtime_graph: object | None = None
    dynamic_axes: dict[str, tuple[int, int]] = field(default_factory=dict)
    kernel_classes: tuple[tuple[int, ...], ...] = ()


class PositionalLlama(torch.nn.Module):
    """将 `position_ids` 作为图输入传给 Llama 模型。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, causal_mask: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).logits


def causal_mask_of(seq_len: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """生成指定序列长度的因果掩码。"""
    if seq_len == 1:
        return torch.zeros(1, 1, 1, 1, dtype=dtype)
    blocked = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
    mask.masked_fill_(blocked, torch.finfo(dtype).min)
    return mask


def _sympy_to_shape_expr(expr, hint: int, dyn_symbol: str = "T"):
    """sympy 表达式转 ShapeExpr；不支持的形状退回包络常量。

    `dyn_symbol` 是导出器为 Dim("T") 生成的符号名（如 `s30`），只有它映射到
    运行期绑定维 T；形状环境派生的辅助符号没有绑定来源，按包络常量处理。
    """
    import sympy

    if isinstance(expr, sympy.Integer):
        return ShapeExpr.const(int(expr))
    if isinstance(expr, sympy.Symbol):
        if str(expr.name) == dyn_symbol:
            return ShapeExpr.dim("T")
        return ShapeExpr.const(hint)
    if not all(str(s) == dyn_symbol for s in expr.free_symbols):
        return ShapeExpr.const(hint)
    if isinstance(expr, sympy.Add):
        parts = [_sympy_to_shape_expr(a, hint, dyn_symbol) for a in expr.args]
        return ShapeExpr.add(*parts)
    if isinstance(expr, sympy.Mul):
        parts = [_sympy_to_shape_expr(a, hint, dyn_symbol) for a in expr.args]
        return ShapeExpr.mul(*parts)
    return ShapeExpr.const(hint)


def _shape_exprs_of(gm: GraphModule):
    """返回按节点名查逐维形状表达式的函数；静态图的节点没有该元数据。"""
    exprs = {node.name: node.meta.get(SHAPE_EXPR_META_KEY) for node in gm.graph.nodes}
    return exprs.get


def capture_shape_exprs(gm: GraphModule, dyn_symbol: str = "T") -> None:
    """记录各节点的符号形状，并把 `val` 取整为 T_max 包络。

    导出样例取在包络上（`seq_len == max`），SymInt 的 hint 就是包络值，
    `int(dim)` 即规划用的整数；表达式留给运行期绑定实际 T。
    `dyn_symbol` 是导出器给动态维生成的符号名。

    必须先读完全部 sympy 表达式再取 `int`：对一个 SymInt 取整会让图中其余
    SymInt 的 `.node.expr` 一并退化成常量，之后读到的就只剩包络值。
    """
    carriers = []
    scalar_raw: dict[str, object] = {}
    for node in gm.graph.nodes:
        val = node.meta.get(VAL_META_KEY)
        if isinstance(val, torch.SymInt):
            scalar_raw[node.name] = val.node.expr
            continue
        if not isinstance(val, torch.Tensor):
            continue
        dims = tuple(val.shape)
        if all(isinstance(d, int) for d in dims):
            continue
        raws = tuple(None if isinstance(d, int) else d.node.expr for d in dims)
        carriers.append((node, val, dims, raws))
    for node, val, dims, raws in carriers:
        exprs = []
        for dim, raw in zip(dims, raws):
            if raw is None:
                exprs.append(None)
                continue
            try:
                exprs.append(_sympy_to_shape_expr(raw, int(dim), dyn_symbol))
            except Exception:
                exprs.append(ShapeExpr.const(int(dim)))
        node.meta[SHAPE_EXPR_META_KEY] = tuple(exprs)
        node.meta[VAL_META_KEY] = val.new_empty(tuple(int(d) for d in dims))
    nodes = {node.name: node for node in gm.graph.nodes}
    scalar_exprs = {
        name: _sympy_to_shape_expr(
            raw, int(nodes[name].meta[VAL_META_KEY]), dyn_symbol)
        for name, raw in scalar_raw.items()}
    _fold_symbolic_scalars(gm, scalar_exprs)


def _annotate_arg_exprs(user, node, expr) -> None:
    """在用户节点的实参位置上记下符号元素，供运行期绑定字面量形状列表。"""
    if user.op != "call_function":
        return
    ann = list(user.meta.get(ARG_EXPR_META_KEY) or [None] * len(user.args))
    if len(ann) < len(user.args):
        ann.extend([None] * (len(user.args) - len(ann)))
    for index, arg in enumerate(user.args):
        if arg is node:
            ann[index] = expr
        elif isinstance(arg, (list, tuple)) and any(item is node for item in arg):
            elems = ann[index] if isinstance(ann[index], list) else [None] * len(arg)
            for pos, item in enumerate(arg):
                if item is node:
                    elems[pos] = expr
            ann[index] = elems
    user.meta[ARG_EXPR_META_KEY] = ann


def _fold_symbolic_scalars(gm: GraphModule, scalar_exprs=None) -> None:
    """把产出标量 SymInt 的节点（sym_size/getitem 链）折叠成包络常量。

    动态导出会把 reshape/view 的形状列表换成符号节点；计划与内核只认整数，
    这里按 hint（=包络值）常量折叠，图结构回到与静态包络一致。
    折叠前把符号表达式记到用户节点上，运行期再按实际 T 重算这些字面量。
    """
    from torch.fx.node import map_aggregate

    scalar_exprs = scalar_exprs or {}
    folded = True
    while folded:
        folded = False
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            val = node.meta.get(VAL_META_KEY)
            if not isinstance(val, torch.SymInt):
                continue
            const = int(val)
            expr = scalar_exprs.get(node.name)
            for user in list(node.users):
                if expr is not None and expr.kind != "const":
                    _annotate_arg_exprs(user, node, expr)
                def substitute(arg):
                    if isinstance(arg, torch.fx.Node) and arg is node:
                        return const
                    return arg
                user.args = map_aggregate(user.args, substitute)
                user.kwargs = map_aggregate(user.kwargs, substitute)
            gm.graph.erase_node(node)
            folded = True


def export_annotated_graph(
    model: torch.nn.Module, seq_len: int, position_ids: torch.Tensor, *,
    dtype: torch.dtype = torch.float16,
    dynamic_shapes: dict | None = None,
) -> GraphModule:
    """导出图并标记节点设备和分区编号。

    `dynamic_shapes` 为空时保持原来的具体形状导出；动态路径只把序列维符号化，
    导出后立刻把 `val` 取整为包络，下游 pass 与内存规划只见整数。
    """
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    program = torch.export.export(
        PositionalLlama(model),
        (input_ids, causal_mask_of(seq_len, dtype), position_ids),
        strict=True,
        dynamic_shapes=dynamic_shapes,
    )
    gm = program.module()
    if dynamic_shapes is not None:
        symbols = {str(sym) for sym in program.range_constraints}
        if len(symbols) != 1:
            raise ValueError(
                f"动态导出应只有一个符号维 T，实际得到 {sorted(symbols)}")
        capture_shape_exprs(gm, symbols.pop())
    partition_graph(gm)
    return gm


def sdpa_layer_map(gm: GraphModule) -> dict[str, int]:
    """返回每个 SDPA 节点到模型层号的映射。"""
    return {
        node.name: _q_proj_layer_of(node.args[0])
        for node in gm.graph.nodes
        if "scaled_dot_product_attention" in str(node.target)
    }


def _q_proj_layer_of(node) -> int:
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur.op == "get_attr" and "q_proj.weight" in str(cur.target):
            return int(str(cur.target).split(".")[3])
        # 分块后投影输出是 cat 的列表参数，要连列表里的节点一起回溯。
        stack.extend(cur.all_input_nodes)
    raise ValueError(f"未能从 {node.name} 回溯到 q_proj 权重")


def peak_kernel_mram_bytes(nodes, *, hardware: PIMHardwareConfig,
                           widths: dict | None = None) -> int:
    """计划里各内核回传的单台 MRAM 占用取最大值，取不到回传时返回 0。

    `widths` 非空时，顺手把回传的元素宽度按 GML op_type 收进去，
    供执行计划按它算访问字节数。探测只编译一次，两处共用这份结果。

    取**最大**而不是求和：内核的 tile 缓冲是一个算子执行期间的临时占用，
    算子之间复用同一块地方，所以峰值由最大的那个决定。三区是常驻、内核 tile 是
    瞬时，两者并存，相加才是单台 DPU 的真实峰值 —— 这是 `plan_dpu` 容量判据要
    的那个数。

    这份占用含 `partial` 档的归约暂存（`pim.placed-mram-bytes` 已经把它算进去）。

    工具链不在位、或某个形状走不了编译内核（`_compiled_linear_supports` 判否、
    或分块挑不出来）时跳过那个算子：拿不到就不猜，返回 0 让判据退回只看三区。
    """
    from contracts.dtypes import dtype_bytes
    from contracts.graph_meta import SPEC_META_KEY
    from contracts.ir_payloads import placement_of_module
    from contracts.op_contract import OpCompileRequest
    from opcompiler_bridge.driver import compile_op
    from runtime.exec_plan_gen import _fused_activation_of, _shard_decision_of
    from runtime.kernels import _compiled_linear_supports

    peak = 0
    seen: set[tuple] = set()
    for node in nodes:
        if "linear" not in str(getattr(node, "target", "")):
            continue
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or not getattr(spec, "shard_map", None):
            continue
        # 形状要取**实参**的，不是输出的：`linear` 的契约是
        # `arg_shapes=[x.shape, weight.shape]`，而 `spec.shard_map` 挂的是这个
        # 节点**输出**的分片。拿输出当 x 会让 K 维对不上（实测全部 8 个 linear
        # 都抛「K 维不一致」，于是峰值恒为 0 —— 判据静默退化成只看三区）。
        if len(node.args) < 2:
            continue
        x_spec = getattr(node.args[0], "meta", {}).get(SPEC_META_KEY)
        w_spec = getattr(node.args[1], "meta", {}).get(SPEC_META_KEY)
        if x_spec is None or w_spec is None:
            continue
        for dpu_id in spec.shard_map:
            if (dpu_id not in getattr(x_spec, "shard_map", {})
                    or dpu_id not in getattr(w_spec, "shard_map", {})):
                continue
            x_shape = tuple(x_spec.shard_map[dpu_id].local_shape)
            w_shape = tuple(w_spec.shard_map[dpu_id].local_shape)
            dtype = spec.dtype or "float16"
            # 折进 linear 的尾部激活一起下发，A 路融合才有对象可折。
            activation = _fused_activation_of(node)
            key = (x_shape, w_shape, dtype, activation)
            if key in seen:
                continue
            seen.add(key)
            if not _compiled_linear_supports([x_shape, w_shape], dtype):
                continue
            raw = _shard_decision_of(spec, spec.shard_map[dpu_id])
            try:
                result = compile_op(OpCompileRequest(
                    op="linear", arg_shapes=[x_shape, w_shape],
                    hardware=hardware, dtype=dtype,
                    num_tasklets=hardware.num_tasklets,
                    shard=None if raw is None else DpuShard.from_payload(raw),
                    activation=activation))
            except (RuntimeError, ValueError):
                # 这是**探测**，不是编译：拿不到就跳过这个形状，绝不让它打断
                # 整个 compile_llama2。三种拿不到的情形都走这里：
                #   - `ToolchainUnavailable`（RuntimeError 子类）：没装 PIM pass
                #   - `ValueError`：形状不满足 A 路契约（K 维、2 的幂等）
                #   - `RuntimeError`：`triton-opt` 自己拒了。实测 tp4 的
                #     `M=1 N=16 K=64` 没有合法的 2 的幂分块，而运行时对这种
                #     形状本来就退回 numpy 内核 —— 它不占内核 tile。
                continue
            if not result.pimir:
                continue
            back = placement_of_module(result.pimir)
            if back.placed_mram_bytes:
                peak = max(peak, back.placed_mram_bytes)
            # 宽度与图上 dtype 相同就是回声，不是新信息：探测按 spec.dtype 编译，
            # 回传必然等于 dtype_bytes(spec.dtype)。只留真正不同的那个。
            if (widths is not None and back.placed_elem_bytes
                    and back.placed_elem_bytes != dtype_bytes(dtype)):
                widths.setdefault("Gemm", back)
    return peak


DEFAULT_PREFILL_BUCKETS = (128, 512, 2048)


def default_prefill_buckets(max_len: int) -> tuple[int, ...]:
    """默认按 128/512/2048 分桶，末项恒为 `max_len`；超过它的桶丢弃。"""
    return tuple(sorted(
        {bucket for bucket in DEFAULT_PREFILL_BUCKETS if bucket < max_len}
        | {max_len}))


def graph_tiling_enabled(cfg, tile: bool | None) -> bool:
    """图级分块开关：显式参数 > `PIM_DISABLE_TILE` > 按架构的默认值。

    MoE 的专家 linear 多而窄，图级分块按 fp32 工作集拆块会把每个专家拆成
    上百个子节点（4 层试算 >130 万节点），而这些形状本来就由 A 路算子的
    tt.dot 内部分块负责，所以默认对 MoE 关闭。`PIM_DISABLE_TILE=1` 是仓库内
    受支持的关闭入口，供外部脚本与排查复现使用。
    """
    if tile is not None:
        return tile
    if os.environ.get("PIM_DISABLE_TILE", "") not in ("", "0"):
        return False
    # 1 专家也是 MoE：按模型身份判定，不用专家数阈值。
    return not is_moe_config(cfg)


def compile_llama2(
    model: torch.nn.Module,
    strategy: ShardStrategy,
    *,
    prefill_seq_len: int,
    max_seq: int,
    hw: HwBudget,
    hardware: PIMHardwareConfig,
    kv_dtype_bytes: int = 2,
    dtype: torch.dtype = torch.float16,
    num_request_slots: int = 1,
    kv_slot_tokens: int = 16,
    kv_pool_slots: int | None = None,
    num_parallel_slots: int | None = None,
    block_table_slots: int | None = None,
    prefill_seq_lens: tuple[int, ...] = (),
    graph_tiling: bool | None = None,
    moe_execution: str | None = None,
    dynamic: bool = False,
) -> CompiledModel:
    """编译 Llama 的 decode 图与一组按长度分桶的 prefill 图。

    `prefill_seq_len` 是最长桶，始终参与编译；`prefill_seq_lens` 是额外桶。
    各桶共享权重区与 KV 区（`kv_base` 一致），激活区按最长桶规划出统一步长
    `act_stride`，短桶只用到窗口的前一段，不额外占内存。
    `num_request_slots` 只做 host 侧并发上限记账；设备侧有
    `num_parallel_slots` 份激活窗口与块表窗口，供同一步的多个请求并行使用，
    物理槽池仍是全局一份（`kv_pool_slots`）。
    `num_parallel_slots` 为空时取 `min(2, num_request_slots)`，放不下时自动减到 1；
    `block_table_slots` 为空时等于它。`kv_pool_slots` 为空时按 MRAM 剩余空间
    自动定容（上限是 `num_request_slots` 个满长请求的块数再加一块）。
    `graph_tiling` 控制图级分块：缺省按架构定（MoE 关闭，其余打开），
    显式 True/False 可覆盖；`PIM_DISABLE_TILE=1` 也走关闭。

    `moe_execution` 是 MoE 的执行语义（`dense_sim` / `dynamic_dispatch`），
    缺省跟随 `PIM_MOE_DISPATCH`。这一份值随 CompiledModel 交给放置导出与
    仿真配置，主链各入口不再各自读环境变量或写死。

    `dynamic=True` 预留给动态图路径；D0 只提供接口，真正的调度/形状绑定在
    D1/D2 实现，未实现前显式报错，避免静默退回静态行为。
    """
    if dynamic and prefill_seq_lens:
        raise ValueError(
            "dynamic=True 只编译单张符号 T 的 prefill 图；额外的桶请留空 "
            "prefill_seq_lens")
    cfg = model.config
    # 容量与对齐是同一份契约：HwBudget 决定三区偏移，PIMHardwareConfig 决定
    # 算子编译的 DMA 对齐，两者不一致时 sidecar 自报与真实 MLIR 会各说一套。
    if hw.align != hardware.dma_align:
        raise ValueError(
            f"HwBudget.align={hw.align} 与 PIMHardwareConfig.dma_align="
            f"{hardware.dma_align} 不一致：容量契约的对齐只有一个值，"
            "请用 memory.mem_planner.default_site_capacity() 派生。")
    moe_execution = resolve_moe_execution(cfg, moe_execution)
    seq_lens = ((prefill_seq_len,) if dynamic
                else tuple(sorted({*prefill_seq_lens, prefill_seq_len})))
    if seq_lens[-1] != prefill_seq_len:
        raise ValueError(
            f"最长的 prefill 桶 {seq_lens[-1]} 与 prefill_seq_len="
            f"{prefill_seq_len} 不一致；把最长长度传给 prefill_seq_len")

    def _export_prefill(seq_len: int) -> GraphModule:
        shapes = None
        if dynamic:
            from torch.export import Dim
            t_dim = Dim("T", min=1, max=seq_len)
            shapes = {"input_ids": {1: t_dim}, "causal_mask": {2: t_dim, 3: t_dim},
                      "position_ids": {1: t_dim}}
        return export_annotated_graph(
            model, seq_len,
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0), dtype=dtype,
            dynamic_shapes=shapes)

    prefill_gms = {seq_len: _export_prefill(seq_len) for seq_len in seq_lens}
    decode_gm = export_annotated_graph(
        model, 1, torch.tensor([[0]], dtype=torch.long), dtype=dtype
    )
    # 切分传播看到的必须是优化后的图，否则分块拆出的子节点没有布局。
    # 没有流水并行时各层在同一组 DPU 上，各层重复的形状类算子可以合并。
    single_stage = strategy.num_stages == 1
    do_tiling = graph_tiling_enabled(cfg, graph_tiling)
    for gm in [*prefill_gms.values(), decode_gm]:
        optimize_graph(gm, wram_bytes=hardware.wram_bytes_per_dpu,
                       tp_width=strategy.tp_width, merge_shape_ops=single_stage,
                       graph_tiling=do_tiling)

    # PP×EP：专家输出落在互不相交的 DPU，先把 MoE 累加链改标 host 再传播规格。
    if getattr(strategy, "ep_width", 1) > 1:
        from graph.spec_prop import remark_moe_aggregation_for_ep
        for gm in [*prefill_gms.values(), decode_gm]:
            remark_moe_aggregation_for_ep(gm)
    prefill_edges_of = {
        seq_len: propagate_specs(gm, strategy) for seq_len, gm in prefill_gms.items()}
    decode_edges = propagate_specs(decode_gm, strategy)
    # 融合标注要看传播出来的重分布边，所以放在 `propagate_specs` 之后。
    for gm in [*prefill_gms.values(), decode_gm]:
        fuse_rms_norms(gm)

    # 门控 silu 折进 linear，A 路编译内核按 activation 参数算。必须放在
    # 切分传播之后：部分和的 gate 投影（行切）经 silu 会直接报错，传播
    # 通过说明这里的 linear 输出都不是部分和。
    for seq_len, gm in prefill_gms.items():
        fuse_linear_activation(gm, prefill_edges_of[seq_len])
    fuse_linear_activation(decode_gm, decode_edges)

    # Qwen3 显式声明 head_dim 且可以不等于 hidden/heads；未声明的架构（Llama、
    # Qwen2）才按老口径推导。
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # KV 几何与序列长度无关，用最长桶的图推一次即可。
    kv_specs = kv_specs_from_strategy(
        prefill_gms[prefill_seq_len],
        strategy,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        max_seq=max_seq,
        dtype_bytes=kv_dtype_bytes,
        slot_tokens=kv_slot_tokens,
    )

    prefill_nodes_of = {
        seq_len: list(gm.graph.nodes) for seq_len, gm in prefill_gms.items()}
    decode_nodes = list(decode_gm.graph.nodes)
    # 内核 tile 占的是与三区同一块 MRAM，所以容量判据要把两者相加。那份占用只有
    # 算子编译器知道（它定分块），这里问它回传；工具链不在位时退回 0，判据只看
    # 三区，与改动前逐字节等价。多桶取峰值，不是求和。
    kernel_mram = peak_kernel_mram_bytes(
        [node for nodes in prefill_nodes_of.values() for node in nodes] + decode_nodes,
        hardware=hardware)
    blocks_per_request = -(-max_seq // kv_slot_tokens)
    # 同一步最多并飞几个请求：host 请求槽仍是上限，激活窗口与它对齐。
    parallel_slots = min(num_parallel_slots or 2, num_request_slots)
    auto_parallel = num_parallel_slots is None
    auto_block_table = block_table_slots is None
    if auto_block_table:
        block_table_slots = parallel_slots
    if block_table_slots < parallel_slots:
        raise ValueError(
            f"block_table_slots={block_table_slots} 小于并行窗口数 {parallel_slots}")

    def _plan(nodes, pool_slots, bt_slots):
        return {
            dpu_id: plan_dpu(dpu_id, nodes, decode_nodes, kv_specs, hw,
                             kernel_mram_bytes=kernel_mram,
                             kv_pool_slots=pool_slots,
                             act_slots=1,
                             block_table_slots=bt_slots)
            for dpu_id in strategy.dpu_ids
        }

    def _max_slots(plans):
        return fit_pool_slots(
            plans, kv_specs, hw, kernel_mram_bytes=kernel_mram,
            act_slots=parallel_slots, block_table_slots=block_table_slots)

    # 自动定容：先用最小槽池探一次几何，再按剩余 MRAM 收窗口数、定槽数。
    # 槽池与窗口都不改变权重占用，所以探测得到的蓝图就能反推容量。
    if kv_pool_slots is None or auto_parallel:
        probe = _plan(prefill_nodes_of[prefill_seq_len],
                      blocks_per_request, block_table_slots)
        if auto_parallel:
            while parallel_slots > 1 and _max_slots(probe) < blocks_per_request:
                parallel_slots -= 1
                if auto_block_table:
                    block_table_slots = parallel_slots
        if kv_pool_slots is None:
            # vLLM 的块池永久留一块当 null 块，槽池比满并发容量多留一块。
            request_capacity = num_request_slots * blocks_per_request + 1
            kv_pool_slots = min(request_capacity, _max_slots(probe))
            if kv_pool_slots <= blocks_per_request:
                raise ValueError(
                    f"MRAM 装不下一个满长请求：需要 {blocks_per_request} 个 KV 槽"
                    f"再加一块空位，按当前权重、块表 ×{block_table_slots}、激活窗口 "
                    f"×{parallel_slots} 只能放 {kv_pool_slots} 个；"
                    f"调小 max_model_len、num_request_slots 或 PIM_MRAM_BYTES 不够"
                    f"时调大 PIM_MRAM_BYTES")

    mem_plans_of = {
        seq_len: _plan(nodes, kv_pool_slots, block_table_slots)
        for seq_len, nodes in prefill_nodes_of.items()}
    mem_plans = mem_plans_of[prefill_seq_len]
    # 权重区和 KV 区只按一遍算，短桶必须落在同一组地址上，否则窗口平移失效。
    for seq_len, plans in mem_plans_of.items():
        for dpu_id, plan in plans.items():
            base = mem_plans[dpu_id]
            if plan.weight != base.weight or plan.kv_base != base.kv_base:
                raise ValueError(
                    f"prefill 桶 {seq_len} 的权重/KV 布局与最长桶不一致（dpu{dpu_id}）")
    act_stride = fit_activation_windows(mem_plans, parallel_slots, hw, kernel_mram)

    bt_bytes = {spec.block_table_bytes for spec in kv_specs.values()}
    if len(bt_bytes) != 1:
        raise ValueError(f"各 DPU 的块表字节数必须一致，got {sorted(bt_bytes)}")
    block_table_bytes = bt_bytes.pop()

    pool_slots = {spec.pool_slots for spec in kv_specs.values()}
    if pool_slots != {kv_pool_slots}:
        raise ValueError(
            f"各 DPU 的物理槽数 {pool_slots} 与 kv_pool_slots={kv_pool_slots} 不一致")
    # `plan_dpu` 收的是节点列表、手里没有 gm，所以阶段标记落在这里 ——
    # 所有图的 mram_offset 都已回填。
    for gm in prefill_gms.values():
        mark_stage(gm, STAGE_PLANNED)
    mark_stage(decode_gm, STAGE_PLANNED)

    np_dtype = np.dtype(np.float16 if dtype == torch.float16 else np.float32)

    # 注意力不再是主机回调：那段计算搬进了设备内核
    # （`runtime.kernels.sdpa_kernel`）。它要读的设备侧 KV 区域编译期才知道，
    # 所以在这里算好、烘进命令 payload——三种策略连着编译时用模块级全局
    # 会互相覆盖，前面那些 plan 再执行就用错了别人的 KV 规格。
    from runtime.kernels import sdpa_kv_info

    def sdpa_info_for(gm: GraphModule):
        layer_of_node = sdpa_layer_map(gm)

        def sdpa_info_of(node):
            if "scaled_dot_product_attention" not in str(node.target):
                return None
            return sdpa_kv_info(node, kv_specs, layer_of_node, np_dtype)

        return sdpa_info_of

    # 每个窗口一份计划：0 号就是原计划（零拷贝），其余把激活地址和块表基址平移到
    # 自己的窗口。所有图共用同一份激活布局，所以窗口步长完全一致。
    act_base_by_dpu = {dpu_id: plan.act_base for dpu_id, plan in mem_plans.items()}

    def _windows(plan):
        return [
            bind_plan(plan, act_base_by_dpu=act_base_by_dpu,
                      act_shift=k * act_stride, block_slot=k,
                      bt_stride=block_table_bytes)
            for k in range(parallel_slots)
        ]

    prefill_plans: dict[int, list] = {}
    prefill_cmd_ids: dict[int, int] = {}
    prefill_plan_max = None
    for seq_len in seq_lens:
        entries = {e.edge_id: e for e in build_comm_plan(
            prefill_edges_of[seq_len], _shape_exprs_of(prefill_gms[seq_len]))}
        pending: dict = {}
        for plan in mem_plans_of[seq_len].values():
            pending.update(plan.pending_readers_prefill)
        compiled_plan = build_execution_plan(
            prefill_nodes_of[seq_len], prefill_gms[seq_len], entries, pending,
            hardware=hardware, sdpa_info_of=sdpa_info_for(prefill_gms[seq_len]),
        )
        prefill_plans[seq_len] = _windows(compiled_plan.plan)
        prefill_cmd_ids[seq_len] = compiled_plan.output_cmd_id
        if seq_len == prefill_seq_len:
            prefill_plan_max = compiled_plan

    decode_entries = {e.edge_id: e for e in build_comm_plan(
        decode_edges, _shape_exprs_of(decode_gm))}
    pending_decode: dict = {}
    for plan in mem_plans.values():
        pending_decode.update(plan.pending_readers_decode)
    decode = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware, sdpa_info_of=sdpa_info_for(decode_gm),
    )

    return CompiledModel(
        strategy=strategy,
        prefill_gm=prefill_gms[prefill_seq_len],
        decode_gm=decode_gm,
        prefill_edges=prefill_edges_of[prefill_seq_len],
        decode_edges=decode_edges,
        kv_specs=kv_specs,
        mem_plans=mem_plans,
        prefill=prefill_plan_max,
        decode=decode,
        hw=hw,
        hardware=hardware,
        num_request_slots=num_request_slots,
        kv_slot_tokens=kv_slot_tokens,
        kv_pool_slots=kv_pool_slots,
        prefill_seq_len=prefill_seq_len,
        num_parallel_slots=parallel_slots,
        act_stride=act_stride,
        block_table_bytes=block_table_bytes,
        prefill_seq_lens=seq_lens,
        prefill_plans=prefill_plans,
        prefill_cmd_ids=prefill_cmd_ids,
        decode_plans=_windows(decode.plan),
        moe_execution=moe_execution,
        compile_mode="dynamic" if dynamic else "static",
        dynamic_axes=( {"T": (1, prefill_seq_len)} if dynamic else {}),
    )


def write_weight_shards(gm: GraphModule, mem_plans: dict[int, DPUPlan], backend) -> None:
    """按内存蓝图将权重的本地分片写入各 DPU 的 MRAM。"""
    by_target = {n.target: n for n in gm.graph.nodes if n.op == "get_attr"}
    for dpu_id, plan in mem_plans.items():
        for name, off in plan.weight.items():
            node = by_target[name]
            detail = node.meta[SPEC_META_KEY].shard_map[dpu_id]
            obj = gm
            for part in name.split("."):
                obj = getattr(obj, part)
            obj = obj.detach()
            if detail.shard_dim == 0:
                local = obj[detail.start_idx : detail.end_idx].numpy()
            elif detail.shard_dim == 1:
                local = obj[:, detail.start_idx : detail.end_idx].numpy()
            else:
                local = obj.numpy()
            backend.write_local(dpu_id, off, np.ascontiguousarray(local))


def load_weights(compiled: CompiledModel, backend) -> None:
    """`write_weight_shards` 的 `CompiledModel` 入口（权重区 offset 两图共用）。"""
    write_weight_shards(compiled.prefill_gm, compiled.mem_plans, backend)
