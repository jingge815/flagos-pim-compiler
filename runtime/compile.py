"""将模型、切分策略和硬件配置编译为可执行蓝图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.fx import GraphModule

from comm.plan import build_comm_plan
from contracts.graph_meta import SPEC_META_KEY
from contracts.op_contract import PIMHardwareConfig
from contracts.pim_tensor_spec import RedistributeEdge
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from graph.strategy import ShardStrategy
from memory.kv_layout import KVRegionSpec, kv_specs_from_strategy
from memory.mem_planner import DPUPlan, HwBudget, plan_dpu
from runtime.exec_plan_gen import CompiledPlan, build_execution_plan
from runtime.executor import DecodeState, make_sdpa_handler


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
    mem_plans = {
        dpu_id: plan_dpu(dpu_id, prefill_nodes, decode_nodes, kv_specs, hw)
        for dpu_id in strategy.dpu_ids
    }

    prefill_entries = {e.edge_id: e for e in build_comm_plan(prefill_edges)}
    decode_entries = {e.edge_id: e for e in build_comm_plan(decode_edges)}
    pending_prefill: dict = {}
    pending_decode: dict = {}
    for plan in mem_plans.values():
        pending_prefill.update(plan.pending_readers_prefill)
        pending_decode.update(plan.pending_readers_decode)

    state = DecodeState(valid_len=0)
    np_dtype = np.dtype(np.float16 if dtype == torch.float16 else np.float32)

    def host_handler_for(gm: GraphModule):
        layer_map = sdpa_layer_map(gm)

        def host_handler_of(node):
            if "scaled_dot_product_attention" in str(node.target):
                return make_sdpa_handler(layer_map[node.name], kv_specs, state, np_dtype)
            return None

        return host_handler_of

    prefill = build_execution_plan(
        prefill_nodes, prefill_gm, prefill_entries, pending_prefill,
        hardware=hardware, host_handler_of=host_handler_for(prefill_gm),
    )
    decode = build_execution_plan(
        decode_nodes, decode_gm, decode_entries, pending_decode,
        hardware=hardware, host_handler_of=host_handler_for(decode_gm),
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
