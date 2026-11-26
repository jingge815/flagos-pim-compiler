"""将模型、切分策略和硬件配置编译为可执行蓝图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.fx import GraphModule

from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import DpuShard, PIMHardwareConfig
from contracts.pim_tensor_spec import RedistributeEdge
from contracts.unified_ir import STAGE_PLANNED, mark_stage
from graph.opt_passes import fuse_linear_activation, optimize_graph
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from graph.strategy import ShardStrategy
from memory.kv_layout import KVRegionSpec, kv_specs_from_strategy
from memory.mem_planner import DPUPlan, HwBudget, plan_dpu
from runtime.exec_plan_gen import CompiledPlan, build_execution_plan
from runtime.executor import DecodeState


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
    state: DecodeState
    hw: HwBudget
    hardware: PIMHardwareConfig
    # 逐槽 prefill 计划（R21-02a）：B 份 [1,S] 计划，第 b 份把 KV 读写
    # 平移到槽 b 的段——服务 pipeline_overlap / weight_reuse 关的逐槽
    # 执行策略；None = 未编译（缺省，B=1 无逐槽对象）。
    per_slot_prefill: tuple[CompiledPlan, ...] | None = None

    @property
    def replicas(self) -> int:
        """DP 副本数（strategy 副本维度；1 即单副本，行为不变）。"""
        return self.strategy.replicas

    def mem_plans_of_replica(self, replica: int) -> dict[int, DPUPlan]:
        """副本 g 的物理 DPU -> 内存计划（物理块 = base + g×num_dpus）。

        布局与副本 0 同源（同策略同形状），权重装载由 `load_weights` 按
        副本各写一份（权重复制 ×G）。
        """
        if not 0 <= replica < self.strategy.replicas:
            raise ValueError(
                f"replica={replica} 越界 [0,{self.strategy.replicas})")
        offset = replica * self.strategy.num_dpus
        return {dpu_id + offset: plan
                for dpu_id, plan in self.mem_plans.items()}


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


def export_annotated_graph(
    model: torch.nn.Module, seq_len: int, position_ids: torch.Tensor, *,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
) -> GraphModule:
    """按 batch 维导出静态图并标记节点设备和分区编号。

    `input_ids` 内部按 `[batch_size, seq_len]` 构造；`position_ids` 由调用方
    给出，行数必须等于 batch_size。掩码保持 `[1,1,S,S]` 广播：各槽是独立
    等长因果序列，共享掩码语义正确。
    """
    if tuple(position_ids.shape) != (batch_size, seq_len):
        raise ValueError(
            f"position_ids 形状 {tuple(position_ids.shape)} 与 "
            f"(batch_size={batch_size}, seq_len={seq_len}) 不符")
    input_ids = (torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                 .repeat(batch_size, 1))
    gm = torch.export.export(
        PositionalLlama(model),
        (input_ids, causal_mask_of(seq_len, dtype), position_ids),
        strict=True,
    ).module()
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
                           widths: dict | None = None,
                           force_compile: bool = False) -> int:
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
    from opcompiler_bridge.parallel import run_jobs
    from runtime.exec_plan_gen import _fused_activation_of, _shard_decision_of
    from runtime.kernels import _compiled_linear_supports

    # 1. 串行收集唯一编译任务：探测的跳过与去重逻辑保持串行语义不变。
    seen: set[tuple] = set()
    jobs: list[tuple] = []
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
            jobs.append((x_shape, w_shape, dtype, activation, raw))

    def probe(job: tuple):
        # 这是**探测**，不是编译：拿不到就返回 None，绝不让它打断整个
        # compile_llama2。三种拿不到的情形都走这里：
        #   - `ToolchainUnavailable`（RuntimeError 子类）：没装 PIM pass
        #   - `ValueError`：形状不满足 A 路契约（K 维、2 的幂等）
        #   - `RuntimeError`：`triton-opt` 自己拒了。实测 tp4 的
        #     `M=1 N=16 K=64` 没有合法的 2 的幂分块，而运行时对这种
        #     形状本来就退回 numpy 内核 —— 它不占内核 tile。
        x_shape, w_shape, dtype, activation, raw = job
        try:
            result = compile_op(OpCompileRequest(
                op="linear", arg_shapes=[x_shape, w_shape],
                hardware=hardware, dtype=dtype,
                num_tasklets=hardware.num_tasklets,
                shard=None if raw is None else DpuShard.from_payload(raw),
                activation=activation), force=force_compile)
        except (RuntimeError, ValueError):
            return None
        if not result.pimir:
            return None
        return placement_of_module(result.pimir)

    # 2. 并行探测：各任务走 compile_op 的独立缓存 key 与文件锁，互不影响；
    #    结果按提交顺序聚合，与串行版逐项一致。
    peak = 0
    for job, back in zip(jobs, run_jobs(probe, jobs)):
        if back is None:
            continue
        if back.placed_mram_bytes:
            peak = max(peak, back.placed_mram_bytes)
        # 宽度与图上 dtype 相同就是回声，不是新信息：探测按 spec.dtype 编译，
        # 回传必然等于 dtype_bytes(spec.dtype)。只留真正不同的那个。
        dtype = job[2]
        if (widths is not None and back.placed_elem_bytes
                and back.placed_elem_bytes != dtype_bytes(dtype)):
            widths.setdefault("Gemm", back)
    return peak


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
    force_compile: bool = False,
    request_kv_tokens: tuple[int, ...] | None = None,
    batch_size: int = 1,
    per_slot_prefill: bool = False,
) -> CompiledModel:
    """编译 Llama 的两张图，返回共享内存蓝图的命令计划。

    `force_compile=True` 表示本轮要真实重编：图测量阶段的内核探测不走缓存，
    与放置/成本侧的 strict 档共用同一个开关。`batch_size` 是 closed batch
    槽数：prefill 导出 `[B,S]`、decode 导出 `[B,1]`，KV 布局按 B 槽规划。
    `per_slot_prefill=True` 且 B>1 时额外导出一张 `[1,S]` 逐槽图并构建
    B 份槽位平移的 prefill 计划（每步 M=S、KV 落本槽段，R21-02a）。
    """
    cfg = model.config
    prefill_gm = export_annotated_graph(
        model, prefill_seq_len,
        torch.arange(prefill_seq_len, dtype=torch.long)
        .unsqueeze(0).repeat(batch_size, 1),
        dtype=dtype, batch_size=batch_size,
    )
    decode_gm = export_annotated_graph(
        model, 1, torch.zeros(batch_size, 1, dtype=torch.long),
        dtype=dtype, batch_size=batch_size,
    )
    per_slot_gm = None
    if per_slot_prefill and batch_size > 1:
        # 逐槽 prefill 图：[1,S] 一次导出，B 份计划共用（只差 KV 槽平移）。
        per_slot_gm = export_annotated_graph(
            model, prefill_seq_len,
            torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0),
            dtype=dtype, batch_size=1,
        )

    # 切分传播看到的必须是优化后的图，否则分块拆出的子节点没有布局。
    # 没有流水并行时各层在同一组 DPU 上，各层重复的形状类算子可以合并。
    single_stage = strategy.num_stages == 1
    optimize_graph(prefill_gm, wram_bytes=hardware.wram_bytes_per_dpu,
                   tp_width=strategy.tp_width, merge_shape_ops=single_stage)
    optimize_graph(decode_gm, wram_bytes=hardware.wram_bytes_per_dpu,
                   tp_width=strategy.tp_width, merge_shape_ops=single_stage)
    if per_slot_gm is not None:
        optimize_graph(per_slot_gm, wram_bytes=hardware.wram_bytes_per_dpu,
                       tp_width=strategy.tp_width, merge_shape_ops=single_stage)

    prefill_edges = propagate_specs(prefill_gm, strategy)
    decode_edges = propagate_specs(decode_gm, strategy)
    per_slot_edges = (propagate_specs(per_slot_gm, strategy)
                      if per_slot_gm is not None else [])

    # 门控 silu 折进 linear，A 路编译内核按 activation 参数算。必须放在
    # 切分传播之后：部分和的 gate 投影（行切）经 silu 会直接报错，传播
    # 通过说明这里的 linear 输出都不是部分和。
    fuse_linear_activation(prefill_gm, prefill_edges)
    fuse_linear_activation(decode_gm, decode_edges)
    if per_slot_gm is not None:
        fuse_linear_activation(per_slot_gm, per_slot_edges)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_specs = kv_specs_from_strategy(
        prefill_gm,
        strategy,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        num_q_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        max_seq=max_seq,
        dtype_bytes=kv_dtype_bytes,
        num_slots=batch_size,
    )

    prefill_nodes = list(prefill_gm.graph.nodes)
    decode_nodes = list(decode_gm.graph.nodes)
    per_slot_nodes = (list(per_slot_gm.graph.nodes)
                      if per_slot_gm is not None else None)
    # 内核 tile 占的是与三区同一块 MRAM，所以容量判据要把两者相加。那份占用只有
    # 算子编译器知道（它定分块），这里问它回传；工具链不在位时退回 0，判据只看
    # 三区，与改动前逐字节等价。
    kernel_mram = peak_kernel_mram_bytes(prefill_nodes + decode_nodes,
                                         hardware=hardware,
                                         force_compile=force_compile)
    mem_plans = {
        dpu_id: plan_dpu(dpu_id, prefill_nodes, decode_nodes, kv_specs, hw,
                         kernel_mram_bytes=kernel_mram,
                         request_kv_tokens=request_kv_tokens,
                         per_slot_nodes=per_slot_nodes)
        for dpu_id in strategy.dpu_ids
    }
    # `plan_dpu` 收的是节点列表、手里没有 gm，所以阶段标记落在这里 ——
    # 各图的 mram_offset 都已回填。
    mark_stage(prefill_gm, STAGE_PLANNED)
    mark_stage(decode_gm, STAGE_PLANNED)
    if per_slot_gm is not None:
        mark_stage(per_slot_gm, STAGE_PLANNED)

    prefill_entries = {e.edge_id: e for e in build_comm_plan(prefill_edges)}
    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
    pending_prefill: dict = {}
    pending_decode: dict = {}
    pending_per_slot: dict = {}
    for plan in mem_plans.values():
        pending_prefill.update(plan.pending_readers_prefill)
        pending_decode.update(plan.pending_readers_decode)
        pending_per_slot.update(plan.pending_readers_per_slot)

    # 运行时状态是编译结果的公开契约字段：槽数必须与 batch 语义同步，
    # 否则 B>1 时调用方被迫手工另建状态（round12–14 严重问题 1）。
    state = DecodeState(valid_lens=(0,) * batch_size)
    np_dtype = np.dtype(np.float16 if dtype == torch.float16 else np.float32)

    # 注意力不再是主机回调：那段计算搬进了设备内核
    # （`runtime/kernels.sdpa_kernel`）。它要读的设备侧 KV 区域编译期才知道，
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

    def _slot_shifted(sdpa_info_of, slot: int):
        # 槽位主序：槽 s 的 KV 基址 = 槽 0 基址 + s×槽步长；[1,S] 逐槽图
        # 的内核行号恒为 0，槽选择由计划 payload 的基址平移表达。
        def info_of(node):
            info = sdpa_info_of(node)
            if info is None or slot == 0:
                return info
            return {
                **info,
                "heads": [
                    {**entry,
                     "k_off": entry["k_off"] + slot * entry["slot_stride"],
                     "v_off": entry["v_off"] + slot * entry["slot_stride"]}
                    for entry in info["heads"]],
            }
        return info_of

    prefill = build_execution_plan(
        prefill_nodes, prefill_gm, prefill_entries, pending_prefill,
        hardware=hardware, sdpa_info_of=sdpa_info_for(prefill_gm),
        num_tasklets=hardware.num_tasklets,
    )
    decode = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware, sdpa_info_of=sdpa_info_for(decode_gm),
        num_tasklets=hardware.num_tasklets,
    )
    per_slot_plans = None
    if per_slot_gm is not None:
        per_slot_entries = {
            e.edge_id: e for e in build_comm_plan(per_slot_edges)}
        base_info_of = sdpa_info_for(per_slot_gm)
        per_slot_plans = tuple(
            build_execution_plan(
                per_slot_nodes, per_slot_gm, per_slot_entries,
                pending_per_slot,
                hardware=hardware,
                sdpa_info_of=_slot_shifted(base_info_of, slot),
                num_tasklets=hardware.num_tasklets,
            )
            for slot in range(batch_size))

    return CompiledModel(
        strategy=strategy,
        prefill_gm=prefill_gm,
        decode_gm=decode_gm,
        prefill_edges=prefill_edges,
        decode_edges=decode_edges,
        kv_specs=kv_specs,
        mem_plans=mem_plans,
        prefill=prefill,
        decode=decode,
        state=state,
        hw=hw,
        hardware=hardware,
        per_slot_prefill=per_slot_plans,
    )


def write_weight_shards(gm: GraphModule, mem_plans: dict[int, DPUPlan], backend,
                        *, dpu_id_offset: int = 0) -> None:
    """按内存蓝图将权重的本地分片写入各 DPU 的 MRAM。

    `dpu_id_offset` 把写入目标平移到副本 g 的物理块（g×num_dpus）——
    mem_plans 键仍是副本 0 的基数（spec 查询按基数），缺省 0 即原行为。
    """
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
            backend.write_local(dpu_id + dpu_id_offset, off,
                                np.ascontiguousarray(local))


def load_weights(compiled: CompiledModel, backend) -> None:
    """`write_weight_shards` 的 `CompiledModel` 入口（权重区 offset 两图共用）。

    DP 副本（R21-02b）：每个副本的物理块各写一份权重（权重复制 ×G）；
    replicas=1 时单副本即原行为。"""
    num_dpus = compiled.strategy.num_dpus
    for replica in range(compiled.strategy.replicas):
        write_weight_shards(compiled.prefill_gm, compiled.mem_plans, backend,
                            dpu_id_offset=replica * num_dpus)
