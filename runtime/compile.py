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
from graph.fuse_pim import fuse_rms_norms
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from graph.strategy import ShardStrategy
from memory.kv_layout import KVRegionSpec, kv_specs_from_strategy
from memory.mem_planner import (
    DPUPlan, HwBudget, fit_activation_windows, fit_pool_slots, plan_dpu)
from runtime.binding import bind_plan
from runtime.exec_plan_gen import CompiledPlan, build_execution_plan


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
) -> GraphModule:
    """导出图并标记节点设备和分区编号。"""
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
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
        stack.extend(a for a in cur.args if hasattr(a, "name"))
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
    from runtime.exec_plan_gen import _shard_decision_of
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
            key = (x_shape, w_shape, dtype)
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
                    shard=None if raw is None else DpuShard.from_payload(raw)))
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
    """
    cfg = model.config
    seq_lens = tuple(sorted({*prefill_seq_lens, prefill_seq_len}))
    if seq_lens[-1] != prefill_seq_len:
        raise ValueError(
            f"最长的 prefill 桶 {seq_lens[-1]} 与 prefill_seq_len="
            f"{prefill_seq_len} 不一致；把最长长度传给 prefill_seq_len")

    def _export_prefill(seq_len: int) -> GraphModule:
        return export_annotated_graph(
            model, seq_len,
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0), dtype=dtype)

    prefill_gms = {seq_len: _export_prefill(seq_len) for seq_len in seq_lens}
    decode_gm = export_annotated_graph(
        model, 1, torch.tensor([[0]], dtype=torch.long), dtype=dtype
    )
    prefill_edges_of = {
        seq_len: propagate_specs(gm, strategy) for seq_len, gm in prefill_gms.items()}
    decode_edges = propagate_specs(decode_gm, strategy)
    # 融合标注要看传播出来的重分布边，所以放在 `propagate_specs` 之后。
    for gm in [*prefill_gms.values(), decode_gm]:
        fuse_rms_norms(gm)

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
        entries = {e.edge_id: e for e in build_comm_plan(prefill_edges_of[seq_len])}
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

    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
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
