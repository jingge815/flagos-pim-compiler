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
) -> CompiledModel:
    """编译 Llama 的两张图，返回共享内存蓝图的命令计划。"""
    cfg = model.config
    prefill_gm = export_annotated_graph(
        model, prefill_seq_len,
        torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0), dtype=dtype,
    )
    decode_gm = export_annotated_graph(
        model, 1, torch.tensor([[0]], dtype=torch.long), dtype=dtype
    )

    prefill_edges = propagate_specs(prefill_gm, strategy)
    decode_edges = propagate_specs(decode_gm, strategy)

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
    )

    prefill_nodes = list(prefill_gm.graph.nodes)
    decode_nodes = list(decode_gm.graph.nodes)
    # 内核 tile 占的是与三区同一块 MRAM，所以容量判据要把两者相加。那份占用只有
    # 算子编译器知道（它定分块），这里问它回传；工具链不在位时退回 0，判据只看
    # 三区，与改动前逐字节等价。
    kernel_mram = peak_kernel_mram_bytes(prefill_nodes + decode_nodes,
                                         hardware=hardware)
    mem_plans = {
        dpu_id: plan_dpu(dpu_id, prefill_nodes, decode_nodes, kv_specs, hw,
                         kernel_mram_bytes=kernel_mram)
        for dpu_id in strategy.dpu_ids
    }
    # `plan_dpu` 收的是节点列表、手里没有 gm，所以阶段标记落在这里 ——
    # 两张图的 mram_offset 都已回填。
    mark_stage(prefill_gm, STAGE_PLANNED)
    mark_stage(decode_gm, STAGE_PLANNED)

    prefill_entries = {e.edge_id: e for e in build_comm_plan(prefill_edges)}
    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
    pending_prefill: dict = {}
    pending_decode: dict = {}
    for plan in mem_plans.values():
        pending_prefill.update(plan.pending_readers_prefill)
        pending_decode.update(plan.pending_readers_decode)

    state = DecodeState(valid_len=0)
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

    prefill = build_execution_plan(
        prefill_nodes, prefill_gm, prefill_entries, pending_prefill,
        hardware=hardware, sdpa_info_of=sdpa_info_for(prefill_gm),
    )
    decode = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware, sdpa_info_of=sdpa_info_for(decode_gm),
    )

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
