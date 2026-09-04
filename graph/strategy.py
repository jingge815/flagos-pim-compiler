"""定义张量并行、流水并行和混合切分策略。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # 只为类型标注，避免 contracts 与 graph 之间的运行时循环导入。
    from contracts.partition_plan import PartitionPlan

# Llama 权重的列切和行切规则。
LLAMA_WEIGHT_RULES: tuple[tuple[str, Literal["col", "row"]], ...] = (
    ("q_proj.weight", "col"),
    ("k_proj.weight", "col"),
    ("v_proj.weight", "col"),
    ("o_proj.weight", "row"),
    ("gate_proj.weight", "col"),
    ("up_proj.weight", "col"),
    ("down_proj.weight", "row"),
    ("lm_head.weight", "col"),
)


@dataclass(frozen=True)
class ShardStrategy:
    """定义流水段、张量并行宽度和权重切分规则。"""

    name: str
    num_dpus: int
    num_stages: int = 1
    weight_rules: tuple[tuple[str, Literal["col", "row"]], ...] = ()
    dpu_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_dpus <= 0:
            raise ValueError("num_dpus must be positive")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be positive")
        if self.num_dpus % self.num_stages:
            raise ValueError(
                f"num_stages={self.num_stages} 不能整除 num_dpus={self.num_dpus}"
                "（每个 stage 必须持有同样多的 DPU）"
            )
        dpu_ids = tuple(range(self.num_dpus)) if self.dpu_ids is None else tuple(self.dpu_ids)
        if (
            len(dpu_ids) != self.num_dpus
            or len(set(dpu_ids)) != len(dpu_ids)
            or any(dpu_id < 0 for dpu_id in dpu_ids)
        ):
            raise ValueError("dpu_ids must contain unique non-negative IDs for every DPU")
        object.__setattr__(self, "dpu_ids", dpu_ids)

    @property
    def tp_width(self) -> int:
        """每个 stage 内的张量并行宽度（1 = 段内不切，纯流水）。"""
        return self.num_dpus // self.num_stages

    @property
    def kind(self) -> Literal["tensor", "pipeline", "hybrid"]:
        """策略类别，用于报告与测试 id；不参与任何地址/布局计算。"""
        if self.num_stages == 1:
            return "tensor"
        return "pipeline" if self.tp_width == 1 else "hybrid"

    def match(self, weight_name: str) -> Literal["col", "row"] | None:
        for pattern, mode in self.weight_rules:
            if pattern in weight_name:
                return mode
        return None

    def dpus_of_stage(self, stage: int) -> tuple[int, ...]:
        """stage 持有的物理 DPU 编号（按 dpu_ids 顺序切段）。"""
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"stage={stage} 越界 [0,{self.num_stages})")
        width = self.tp_width
        return self.dpu_ids[stage * width : (stage + 1) * width]

    def stage_of_layer(self, layer: int, num_layers: int) -> int:
        """返回层号所属的流水段，要求各段层数相同。"""
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_layers % self.num_stages:
            raise ValueError(
                f"num_layers={num_layers} 不能被 num_stages={self.num_stages} 整除"
            )
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer={layer} 越界 [0,{num_layers})")
        return layer // (num_layers // self.num_stages)

    def dpus_of_layer(self, layer: int, num_layers: int) -> tuple[int, ...]:
        """返回存放指定层权重和激活的 DPU。"""
        return self.dpus_of_stage(self.stage_of_layer(layer, num_layers))

    def layers_of_stage(self, stage: int, num_layers: int) -> list[int]:
        """返回指定流水段持有的层号。"""
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"stage={stage} 越界 [0,{self.num_stages})")
        per_stage = num_layers // self.num_stages
        if num_layers % self.num_stages:
            raise ValueError(
                f"num_layers={num_layers} 不能被 num_stages={self.num_stages} 整除"
            )
        return list(range(stage * per_stage, (stage + 1) * per_stage))

    def stage_of_dpu(self, dpu_id: int) -> int:
        """返回 DPU 所属的流水段。"""
        index = self.dpu_ids.index(dpu_id)
        return index // self.tp_width

    def layers_of_dpu(self, dpu_id: int, num_layers: int) -> list[int]:
        """返回 DPU 持有的层号。"""
        return self.layers_of_stage(self.stage_of_dpu(dpu_id), num_layers)


def llama_strategy(
    num_dpus: int,
    *,
    num_stages: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
    dpu_ids: tuple[int, ...] | None = None,
) -> ShardStrategy:
    """构造 Llama 切分策略，并校验各维可被张量并行宽度整除。"""
    strategy = ShardStrategy(
        name=f"tp{num_dpus // num_stages}_pp{num_stages}",
        num_dpus=num_dpus,
        num_stages=num_stages,
        weight_rules=LLAMA_WEIGHT_RULES,
        dpu_ids=dpu_ids,
    )
    tp_width = strategy.tp_width
    if tp_width & (tp_width - 1):
        raise ValueError(f"tp_width={tp_width} 不是 2 的整数次幂（切分契约 5）")
    for label, length in (
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("intermediate_size", intermediate_size),
        ("vocab_size", vocab_size),
    ):
        if length % tp_width:
            raise ValueError(f"{label}={length} 不能被 tp_width={tp_width} 整除（切分契约 5）")
    strategy.stage_of_layer(0, num_layers)  # 校验流水段可均分层。
    return strategy


def llama_strategies(
    num_dpus: int,
    *,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
) -> list[ShardStrategy]:
    """枚举满足切分约束的 Llama 策略，按流水段数升序返回。"""
    strategies: list[ShardStrategy] = []
    num_stages = 1
    while num_stages <= num_dpus:
        try:
            strategies.append(
                llama_strategy(
                    num_dpus,
                    num_stages=num_stages,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    intermediate_size=intermediate_size,
                    vocab_size=vocab_size,
                    num_layers=num_layers,
                )
            )
        except ValueError:
            pass  # 跳过不满足约束的流水段数。
        num_stages *= 2
    if not strategies:
        raise ValueError(
            f"num_dpus={num_dpus} 下没有任何契约内的切分策略"
            f"（num_layers={num_layers} num_kv_heads={num_kv_heads}）"
        )
    return strategies


def strategy_from_partition_plan(
    plan: "PartitionPlan",
    *,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    vocab_size: int,
    num_layers: int,
) -> ShardStrategy:
    """把 GeneSim 给定的切分方案转成图编译器的 `ShardStrategy`。

    这是四段接口里方向相反的那一段：其余三段是编译器算完告诉 GeneSim，这一段是
    GeneSim 定下 PU 映射来约束编译器（契约见 contracts/partition_plan.py）。

    方案里的段数、DPU 数、权重切分规则直接对应 `ShardStrategy` 的同名概念，所以
    这里只做转换加校验，不做推断。校验走 `llama_strategy` 同一条路径——GeneSim
    给的方案同样要满足切分契约（tp_width 是 2 的幂、各维可整除、层数可均分），
    不满足就直接报错，而不是悄悄换一个能跑的策略。
    """
    strategy = llama_strategy(
        plan.num_dpus,
        num_stages=plan.num_stages,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        dpu_ids=plan.dpu_ids,
    )
    if not plan.weight_rules:
        return strategy
    # 方案显式给了权重切分规则就用它的，覆盖 LLAMA_WEIGHT_RULES 的默认值。
    return replace(strategy, weight_rules=plan.weight_rules)


def format_strategy(strategy: ShardStrategy, num_layers: int) -> str:
    """返回策略的流水段、DPU 和层号分配文本。"""
    lines = [
        f"== 策略 {strategy.name}（{strategy.kind}）: {strategy.num_dpus} DPU = "
        f"{strategy.num_stages} stage × tp{strategy.tp_width}，{num_layers} 层 =="
    ]
    for stage in range(strategy.num_stages):
        layers = strategy.layers_of_stage(stage, num_layers)
        lines.append(
            f"stage{stage}: dpus={list(strategy.dpus_of_stage(stage))} "
            f"layers=[{layers[0]}..{layers[-1]}]（{len(layers)} 层）"
        )
    return "\n".join(lines)
