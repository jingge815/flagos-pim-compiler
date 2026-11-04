"""按模型架构选择加载与切分方式；白名单之外的架构在启动期被拒绝。

新增一类架构只要往 `ARCHITECTURES` 里加一条：它返回的模型必须能按
`compile_llama2` 的同一套算子模式导出（rmsnorm + 标准 RoPE + 全注意力）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM

from graph.strategy import ShardStrategy, llama_strategies


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


ARCHITECTURES: dict[str, Architecture] = {
    "llama": Architecture(_load_llama, _dense_strategy),
    "qwen2": Architecture(_load_qwen2, _dense_strategy),
    "qwen3": Architecture(_load_qwen3, _dense_strategy),
}
