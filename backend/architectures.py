"""按模型架构选择加载与切分方式；白名单之外的架构在启动期被拒绝。

新增一类架构只要往 `ARCHITECTURES` 里加一条：它返回的模型必须能按
`compile_llama2` 的同一套算子模式导出（rmsnorm + 标准 RoPE + 全注意力）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
        # 命中与否只由 top-k 选中索引决定；路由权重只参与加权。权重下溢为 0
        # 的已选中专家仍要执行（HF 会算 0×输出），不能用非零权重当命中条件。
        selected_mask = torch.zeros_like(probs, dtype=torch.bool).scatter( 1, selected, True)
        routing = torch.zeros_like(probs).scatter(1, selected, top_weights.to(probs.dtype))

        def expert_out(e: int) -> torch.Tensor:
            # 未选中的专家不参与 HF 计算；先绕开它的输出再乘路由权重。
            # 直接 0×inf/nan 会得到 nan，把它抬成「选中则取输出、否则取零」。
            out = blk.experts[e](xs)
            return torch.where(selected_mask[:, e:e + 1], out, 0.0)

        acc = expert_out(0) * routing[:, 0:1].to(xs.dtype)
        for e in range(1, blk.num_experts):
            acc = acc + expert_out(e) * routing[:, e:e + 1].to(xs.dtype)
        return acc.view(b, s, h), logits

    # 标记静态改写：参考路径必须能断言自己没走这条实现。
    forward._static_moe = True
    return forward


# 稠密模型的七种投影，加上整图一个的输出头；MoE 另有 router 与每专家三件套。
# lm_head 不是层内权重，但同样是编译器导出图的一部分，主链必须核对它存在，
# 否则词表维计算与 622 MB 输出头权重会整段漏计。
_DENSE_ROLES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
})
# MoE 除了专家的 gate/up/down，还有注意力四投影与 router；sidecar 校验要接受
# 全部合法身份，主链自检则要求它们都出现。
_MOE_ROLES = _DENSE_ROLES | {"router"}


@dataclass(frozen=True)
class Architecture:
    """一类架构的接入点与主链描述。

    除了「怎么装权重、怎么按 DPU 切分」，主链还需要产物命名、必需投影身份
    与仿真配置名；把这些放进注册表，`run_full_pipeline` 与放置导出就不用
    各自再硬编一套模型名字。
    """

    load: Callable[[str], torch.nn.Module]
    strategy: Callable[[int, object], ShardStrategy]
    artifact: str                            # 产物名前缀，如 llama2_7b
    required_roles: frozenset[str]           # 图骨架必须出现的投影身份
    widths: Callable[[object], dict]         # 切分枚举用的宽度参数
    sim_config: str | None = None            # GeneSim 仿真配置模板
    sim_config_ab: str | None = None         # A/B 对照配置模板
    # 架构特有的切分约束；自动选择、--num-stages 与 PartitionPlan 共用。
    strategy_check: Callable[[object, ShardStrategy], None] | None = None

    def check_strategy(self, cfg, strategy: ShardStrategy) -> None:
        """按架构约束校验策略；不合法直接报错，不静默换一个能跑的策略。"""
        if self.strategy_check is not None:
            self.strategy_check(cfg, strategy)


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
        block = layer.mlp
        # 静态改写只用于导出；保存 HF 原始稀疏前向，供测试生成独立参考。
        block._hf_sparse_forward = block.forward
        block.forward = make_static_moe_forward(block)
    return model


def restore_hf_moe_forward(model: torch.nn.Module) -> int:
    """把 MoE 层恢复成 HF 原始稀疏前向，返回恢复的层数。

    编译导出之后调用：若参考也跑静态改写版，两边共享同一套假设，改写自身的
    路由/溢出错误就发现不了。只认 `_hf_sparse_forward`，普通模型返回 0。
    """
    restored = 0
    for layer in getattr(model, "model", model).layers:
        block = layer.mlp
        original = getattr(block, "_hf_sparse_forward", None)
        if original is None:
            continue
        block.forward = original
        restored += 1
    return restored


def _dense_widths(cfg) -> dict:
    """稠密切分枚举的宽度参数；Llama 与 Qwen2/Qwen3 共用。"""
    return dict(
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_hidden_layers,
    )


def _moe_widths(cfg) -> dict:
    """MoE 按专家 FFN 宽切；dense 的 intermediate_size 是配置残留。"""
    widths = _dense_widths(cfg)
    widths["intermediate_size"] = cfg.moe_intermediate_size
    return widths


def _dense_strategy(num_dpus: int, cfg) -> ShardStrategy:
    """按配置里的头数/层数取第一个可行的切分。"""
    return llama_strategies(num_dpus, **_dense_widths(cfg))[0]


def _moe_strategy_check(cfg, strategy: ShardStrategy) -> None:
    """MoE 的专家整除约束：router 按专家维列切，TP 宽度必须整除专家数。"""
    num_experts = int(getattr(cfg, "num_experts", 0) or 0)
    if num_experts > 0 and num_experts % strategy.tp_width:
        raise ValueError(
            f"tp_width={strategy.tp_width} 不能整除 num_experts={num_experts}："
            "router 的专家维不能跨 DPU 劈开；请改用能整除专家数的 TP 宽度"
            "（如 tp1）。")


def _moe_strategy(num_dpus: int, cfg) -> ShardStrategy:
    """MoE 策略：按专家 FFN 宽切，并要求 TP 宽度能整除专家数。

    router 权重按输出维（专家维）列切，专家数不能被 TP 宽度整除时
    1 专家模型会连合法策略都选不出来。
    """
    strategies = llama_strategies(num_dpus, **_moe_widths(cfg))
    legal = []
    for strategy in strategies:
        try:
            _moe_strategy_check(cfg, strategy)
        except ValueError:
            continue
        legal.append(strategy)
    if not legal:
        num_experts = int(getattr(cfg, "num_experts", 0) or 0)
        raise ValueError(
            f"num_dpus={num_dpus} 下没有能整除 num_experts={num_experts} "
            "的切分策略：router 的专家维不能跨 DPU 劈开。")
    return legal[0]


ARCHITECTURES: dict[str, Architecture] = {
    "llama": Architecture(
        _load_llama, _dense_strategy,
        artifact="llama2_7b", required_roles=_DENSE_ROLES, widths=_dense_widths,
        sim_config="sim_llama2_7b_pp_tp{tp_width}pp{num_stages}_globalcost.yaml",
        sim_config_ab="sim_llama2_7b_pp_tp{tp_width}pp{num_stages}_ab_global.yaml",
    ),
    "qwen2": Architecture(
        _load_qwen2, _dense_strategy,
        artifact="qwen2_7b", required_roles=_DENSE_ROLES, widths=_dense_widths,
    ),
    "qwen3": Architecture(
        _load_qwen3, _dense_strategy,
        artifact="qwen3_8b", required_roles=_DENSE_ROLES, widths=_dense_widths,
    ),
    "qwen3_moe": Architecture(
        _load_qwen3_moe, _moe_strategy,
        artifact="qwen3_moe", required_roles=_MOE_ROLES, widths=_moe_widths,
        sim_config="sim_qwen3_moe_pimir.yaml",
        strategy_check=_moe_strategy_check,
    ),
}


def architecture_of(model_dir: str | Path) -> tuple[str, Architecture]:
    """按模型目录里的 config.json 找出注册表条目，返回 (model_type, 架构)。

    主链的角色契约、产物命名与仿真配置都从注册表派生，不允许在脚本里再硬编
    模型名字；未知 model_type 直接报错并列出受支持项。
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{model_dir} 下没有 config.json")
    model_type = str(json.loads(config_path.read_text()).get("model_type", ""))
    if model_type not in ARCHITECTURES:
        raise ValueError(
            f"model_type={model_type!r} 不受支持；已知的有 "
            f"{'、'.join(ARCHITECTURES)}")
    return model_type, ARCHITECTURES[model_type]
