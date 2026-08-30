"""切分策略：把「哪些 DPU 参与这个张量」从全局常量变成按层查的函数。

原先 `ShardConfig` 隐含「每个张量都摊在全部 DPU 上」（张量并行），策略这一层
把它推广成两级划分：DPU 先按 `num_stages` 分成若干 **stage**（流水段），层按
stage 均分；每个 stage 内部再按 `tp_width` 做张量并行。三种切分由同一组参数
表达：

- 张量并行：`num_stages=1`，全部 DPU 一个 stage，层不参与决策；
- 流水并行：`tp_width=1`，每个 stage 一台 DPU，段内不切；
- 混合：两者都 >1。

`num_stages=1` 时本模块的全部方法退化为「恒返回全体 DPU」，与推广之前逐字
等价——这是既有张量并行路径不受影响的依据。

层 → stage 的归属只依赖层号；层号由 `graph/spec_prop.py` 从
`node.meta["nn_module_stack"]` 解出（解不出层号的节点如 `model.norm`/`lm_head`
归最后一个 stage，见该模块 `_dpus_of_node`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Llama 系默认的 Megatron 列切/行切配对（方案二.(2) 初始切分表）：q/k/v/gate/
# up/lm_head 切输出维、o/down 切 contraction 维，两者配对使段内只需一次
# all_reduce。策略只改变「哪些 DPU 参与」，不改变这张表。
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
    """一种切分策略：DPU 如何分成流水段 + 每段内怎么做张量并行。

    `dpu_ids` 是物理 DPU 编号的有序列表（默认 `range(num_dpus)`）；stage s 持有
    其中第 `[s*tp_width, (s+1)*tp_width)` 段。`tp_width` 不是独立字段而是
    `num_dpus // num_stages` 的推导值——两者都存字段就会有「互相矛盾」这种
    非法状态，这里只留一个真值源。

    `weight_rules` 按 get_attr 节点名做子串匹配，首个命中生效；"col" = 切输出维
    （HF 权重 [out, in] 的 Shard(0)），"row" = 切 contraction 维（Shard(1)）。
    """

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
        """层号 → stage 编号（层按 stage 均分，连续分配）。

        要求 `num_stages` 整除 `num_layers`——不整除则各 stage 层数不等，权重与
        KV 的分布随之不均，属契约外情形，直接抛错而不做静默取整。
        """
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
        """层号 → 该层的权重/激活落在哪些 DPU 上（本模块对外主接口）。"""
        return self.dpus_of_stage(self.stage_of_layer(layer, num_layers))

    def layers_of_stage(self, stage: int, num_layers: int) -> list[int]:
        """stage 持有的层号列表（问题 7 按 stage 留 KV 区用）。"""
        if not 0 <= stage < self.num_stages:
            raise ValueError(f"stage={stage} 越界 [0,{self.num_stages})")
        per_stage = num_layers // self.num_stages
        if num_layers % self.num_stages:
            raise ValueError(
                f"num_layers={num_layers} 不能被 num_stages={self.num_stages} 整除"
            )
        return list(range(stage * per_stage, (stage + 1) * per_stage))

    def stage_of_dpu(self, dpu_id: int) -> int:
        """物理 DPU 编号 → 它所属的 stage（问题 7/8 按 DPU 反查本 stage 的层）。"""
        index = self.dpu_ids.index(dpu_id)
        return index // self.tp_width

    def layers_of_dpu(self, dpu_id: int, num_layers: int) -> list[int]:
        """物理 DPU → 它持有哪些层（张量并行下为全部层）。"""
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
    """构造一个 Llama 系策略并校验第 1 阶段切分契约（方案二.(11)）。

    契约校验的除数从推广前的 `num_dpus` 换成 `tp_width`——被切的是 stage 内部
    的那一维，与 stage 数无关：`tp_width` 为 2 的幂、整除 Q/KV head 数（保证切点
    对齐 head 边界，是问题 7 KV 本地驻留的前提）、整除 intermediate_size 与
    vocab_size；此外 `num_stages` 必须整除层数。契约不满足抛 ValueError。
    """
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
    strategy.stage_of_layer(0, num_layers)  # 校验 num_stages 整除层数
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
    """搜索空间：枚举 `num_stages` 取 num_dpus 的全部 2 的幂因子，保留契约内的。

    第 1 阶段不做代价评估——本函数只负责「枚举出哪些切法是合法的」，调用方按
    顺序逐个编译验证，不打分、不排序（自动求优属 [阶段2]）。契约外的组合
    （如 tp_width 切不到 head 数、num_stages 切不到层数）在枚举时跳过，因为
    「这个点不在搜索空间内」正是本函数要回答的问题，不是错误。

    出: 按 num_stages 升序（张量并行在前、纯流水在后）的策略列表。
    """
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
            pass  # 契约外的 num_stages，不在搜索空间内
        num_stages *= 2
    if not strategies:
        raise ValueError(
            f"num_dpus={num_dpus} 下没有任何契约内的切分策略"
            f"（num_layers={num_layers} num_kv_heads={num_kv_heads}）"
        )
    return strategies


def format_strategy(strategy: ShardStrategy, num_layers: int) -> str:
    """把策略的 stage → (DPU, 层) 划分打印成可读文本（对齐 format_* 惯例）。"""
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
