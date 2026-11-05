"""按模型架构选择加载与切分方式；白名单之外的架构在启动期被拒绝。

新增一类架构只要往 `ARCHITECTURES` 里加一条：它返回的模型必须能按
`compile_llama2` 的同一套算子模式导出（rmsnorm + 标准 RoPE + 全注意力）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import (
    LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM)

from graph.strategy import ShardStrategy, llama_strategies


def make_static_moe_forward(blk: torch.nn.Module) -> Callable:
    """把 Qwen3MoE 的稀疏前向换成形状静态的稠密模拟版。

    HF 原版按「命中专家」动态循环并按 token 子集 gather，形状依赖数据，
    torch.export 导不出。这里改成 topk/scatter 生成稀疏门控权重后全专家
    加权累加：未选中专家权重为 0，乘加恒等；按专家号升序累加与 HF 的
    index_add 顺序一致，数值逐元素相等（norm_topk_prob 两种语义均验证）。
    """
    def forward(hidden_states: torch.Tensor, **kwargs):
        b, s, h = hidden_states.shape
        xs = hidden_states.view(-1, h)
        logits = blk.gate(xs)
        probs = F.softmax(logits, dim=1, dtype=torch.float)
        top_weights, selected = torch.topk(probs, blk.top_k, dim=-1)
        if blk.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        routing = torch.zeros_like(probs).scatter(1, selected, top_weights.to(probs.dtype))
        acc = blk.experts[0](xs) * routing[:, 0:1].to(xs.dtype)
        for e in range(1, blk.num_experts):
            acc = acc + blk.experts[e](xs) * routing[:, e:e + 1].to(xs.dtype)
        return acc.view(b, s, h), logits

    return forward


@dataclass(frozen=True)
class Architecture:
    """一类架构的两个接入点：怎么装权重、怎么按 DPU 切分。"""

    load: Callable[[str], torch.nn.Module]
    strategy: Callable[[int, object], ShardStrategy]


def _load_llama(path: str) -> torch.nn.Module:
    """读 fp16 权重；不改全局梯度开关，否则导出的图会变。"""
    return LlamaForCausalLM.from_pretrained(path, dtype=torch.float16).eval()


def _load_qwen2(path: str) -> torch.nn.Module:
    """Qwen2 与 Llama 的差别只有 q/k/v 偏置，加载口径相同。"""
    return Qwen2ForCausalLM.from_pretrained(path, dtype=torch.float16).eval()


def _load_qwen3(path: str) -> torch.nn.Module:
    """Qwen3 相对 Qwen2：q/k 各多一个 head 维 RMSNorm，qkv 无偏置，head_dim 显式。"""
    return Qwen3ForCausalLM.from_pretrained(path, dtype=torch.float16).eval()


def _load_qwen3_moe(path: str) -> torch.nn.Module:
    """Qwen3 MoE：加载后把稀疏 MoE 前向换成静态稠密模拟版，权重引用不变。"""
    model = Qwen3MoeForCausalLM.from_pretrained(path, dtype=torch.float16).eval()
    for layer in model.model.layers:
        layer.mlp.forward = make_static_moe_forward(layer.mlp)
    return model


def _dense_strategy(num_dpus: int, cfg) -> ShardStrategy:
    """按配置里的头数/层数取第一个可行的切分，Llama 与 Qwen2/Qwen3 共用。"""
    return llama_strategies(
        num_dpus,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )[0]


def _moe_strategy(num_dpus: int, cfg) -> ShardStrategy:
    """MoE 策略：按专家 FFN 宽（moe_intermediate_size）切，dense 的 intermediate_size 是配置残留。"""
    return llama_strategies(
        num_dpus,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.moe_intermediate_size,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )[0]


ARCHITECTURES: dict[str, Architecture] = {
    "llama": Architecture(_load_llama, _dense_strategy),
    "qwen2": Architecture(_load_qwen2, _dense_strategy),
    "qwen3": Architecture(_load_qwen3, _dense_strategy),
    "qwen3_moe": Architecture(_load_qwen3_moe, _moe_strategy),
}
