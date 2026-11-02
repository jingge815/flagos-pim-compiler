"""按模型架构选择加载与切分方式；白名单之外的架构在启动期被拒绝。

新增一类架构只要往 `ARCHITECTURES` 里加一条：它返回的模型必须能按
`compile_llama2` 的同一套算子模式导出（rmsnorm + 标准 RoPE + 全注意力）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from transformers import LlamaForCausalLM

from graph.strategy import ShardStrategy, llama_strategies


@dataclass(frozen=True)
class Architecture:
    """一类架构的两个接入点：怎么装权重、怎么按 DPU 切分。"""

    load: Callable[[str], torch.nn.Module]
    strategy: Callable[[int, object], ShardStrategy]


def _load_llama(path: str) -> torch.nn.Module:
    """读 fp16 权重；不改全局梯度开关，否则导出的图会变。"""
    return LlamaForCausalLM.from_pretrained(path, dtype=torch.float16).eval()


def _llama_strategy(num_dpus: int, cfg) -> ShardStrategy:
    """按配置里的头数/层数取第一个可行的 Llama 切分。"""
    return llama_strategies(
        num_dpus,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )[0]


ARCHITECTURES: dict[str, Architecture] = {
    "llama": Architecture(_load_llama, _llama_strategy),
}
