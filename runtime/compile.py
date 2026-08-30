"""「模型 + 策略 + 硬件 → 可执行蓝图」的唯一编译入口。

把原先在 8 个测试文件里各自复制一遍的编译流水收成一处：

    export（prefill/decode 两张图）
      → 问题 1 partition_graph
      → 问题 2 propagate_specs（按策略切分）
      → 问题 7 kv_specs_from_strategy
      → 问题 8 plan_dpu（回填全部 mram_offset）
      → 问题 3 build_comm_plan
      → 问题 6 build_execution_plan（含 KV 感知 SDPA handler）

这条流水本身与切分策略无关——策略只作为一个参数传给 `propagate_specs` 与
`kv_specs_from_strategy`，下游模块全部按 `spec.shard_map` 迭代，自然吃下不同的
DPU 子集。所以「换一种切分」在调用方看来就是换一个 `strategy` 实参，这也是本
模块存在的意义：让策略遍历不需要复制流水。

`compile_llama2` 只做编译期的事（不碰后端、不搬数据）；把权重真正搬进各 DPU 的
MRAM 是 `load_weights` 的事，两者分开是因为编译期产物可以在多次执行间复用。
"""

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
    """一次 `compile_llama2` 的全部产物：标注图 + 三份编译期蓝图 + 两份命令 DAG。

    编译期测试（问题 1/2/3/7/8）用前半段字段；端到端测试（问题 6）用
    `prefill`/`decode`。`state` 跨两图共享，是 `valid_len` 的唯一真值源。
    """

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
    """RoPE 位置显式作为图输入的导出包装。

    `position_ids` 必须是显式图输入：不传的话 `torch.export` 会把
    `arange(0, seq_len)` 烤成编译期常量，对 `seq_len=1` 的 decode 图意味着每步都
    按位置 0 算 RoPE，第 2 个 decode 步开始 K/V 全错（真实复现过，见
    `docs/executor-20260824.md`）。
    """

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
    """因果 mask；decode 图（seq_len=1）只需形状占位，历史可见性由 KV handler 决定。"""
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
    """export 一张图并跑完问题 1 的 device/part_id 打标。"""
    input_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    gm = torch.export.export(
        PositionalLlama(model),
        (input_ids, causal_mask_of(seq_len, dtype), position_ids),
        strict=True,
    ).module()
    partition_graph(gm)
    return gm


def sdpa_layer_map(gm: GraphModule) -> dict[str, int]:
    """SDPA 节点名 → 它所属的层号（编译期静态信息，供 handler 闭包捕获）。

    从 SDPA 的 Q 输入沿 args 回溯到 q_proj 权重、由权重名解出层号。这是静态的图
    结构信息，在 `build_execution_plan` 之前算好。
    """
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
    """编译期全链路：两张图 → 标注 → 切分传播 → KV/内存/通信蓝图 → 两份命令 DAG。

    入: model —— HF `LlamaForCausalLM`（已 eval、已转 dtype）；strategy ——
        切分策略；prefill_seq_len —— prefill 图的固定序列长度；max_seq —— KV 区
        按此长度预留；hw/hardware —— 内存预算与硬件契约。
    出: `CompiledModel`。纯编译期，不碰后端、不搬数据。

    两张图共用同一份 `plan_dpu` 蓝图（权重区 offset 两图相同、激活区 overlay），
    因此必须一次 `plan_dpu` 同时喂两图的节点，不能各调一次。
    """
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
    """按内存蓝图把每份权重的本地分片真正搬进对应 DPU 的 MRAM。

    切片方式由 `TensorShardDetail.shard_dim` 决定：0 = 切输出维（取行区间）、
    1 = 切 contraction 维（取列区间）、-1 = 全量副本。流水下某台 DPU 的蓝图里
    只有本 stage 那些层的权重，循环自然只搬这些——不需要额外的过滤。

    入: gm —— 带 spec 标注的图（权重实体从它身上按名字取）；mem_plans ——
        `plan_dpu` 的产物（dpu_id -> DPUPlan）；backend —— 目标后端。

    与 `load_weights` 的关系：本函数是核心实现，接三个裸参数，供尚未改用
    `compile_llama2` 的既有测试直接调用；`load_weights` 只是从 `CompiledModel`
    取出这三样再转调，两者不是两套实现。
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
            backend.write_local(dpu_id, off, np.ascontiguousarray(local))


def load_weights(compiled: CompiledModel, backend) -> None:
    """`write_weight_shards` 的 `CompiledModel` 入口（权重区 offset 两图共用）。"""
    write_weight_shards(compiled.prefill_gm, compiled.mem_plans, backend)
