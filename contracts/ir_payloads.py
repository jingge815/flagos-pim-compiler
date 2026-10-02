"""统一 IR 挂在 node.meta 上的载荷类型。

原先这 6 个 dataclass 各自住在写入它的那个 pass 文件里，契约只能靠注释描述
「这个键挂什么类型」。集中到这里之后，键登记表（`contracts/unified_ir.py`）
与载荷类型同处一层，可以由测试静态校验两者对得上。

`Node` 用 `TYPE_CHECKING` 引用：只在类型检查期生效，运行时零导入，
所以 `contracts/` 不会因此依赖 torch.fx。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.fx import Node


@dataclass
class FusedTail:
    """折进主算子的尾部算子。

    `activation` 是激活的 GML 名；`pool` 是紧随其后的池化，没有则为 None。
    `nodes` 保留被折掉的 FX 节点，供序列化器生成 contraction 块时取参数。
    """

    activation: str
    pool: str | None
    nodes: list[Node]


@dataclass
class RopeMatch:
    """一条匹配上的 RoPE 链。

    `cos` / `sin` 是两个广播张量的产出节点 —— GML 侧要按它们生成
    `Llama2Activation_Cos` / `_Sin` 两个子块的缓冲与定标。
    """

    source: Node          # 被旋转的张量（Q 或 K）
    cos: Node
    sin: Node
    absorbed: list[Node] = field(default_factory=list)


@dataclass
class RmsNormFusion:
    """一个折出来的 RMSNorm_vpu 节点要用到的东西。

    `epsilon` 落进 `RMSNorm_Add_Const_<id>.bin`（fp32），取自 `config.json` 的
    `rms_norm_eps`——但这里是从图里读出来的，避免与 config 不一致。

    `weight_node` 是那个一维缩放张量（`input_layernorm.weight`），
    走 `weight_buffer` 但量化粒度是 per-tensor、sf 是 **fp32**（实测 2 处）。
    """

    epsilon: float
    weight_node: Node | None
    eaten: list[Node] = field(default_factory=list)
    # 链中间跨过的透明节点（`to` / `alias` / 断言）。必须一起删，
    # 否则它们仍引用链中间的节点，`rsqrt` 就删不掉。
    passthrough: list[Node] = field(default_factory=list)


@dataclass
class KvDmaSpec:
    """一个 KV_Cache_DMA 的编译期规格。

    `is_key` 区分 K / V 两路 —— 它们配置相同，但落盘的 cache 不同。
    `numel` 定各缓冲的元素数。
    """

    is_key: bool
    numel: int


@dataclass
class SplitSpec:
    """一个 Split 的编译期规格。`heads` 是输出个数。"""

    heads: int
    numel: int


@dataclass
class DynamicScalingSpec:
    """一个 DQ 节点的编译期规格。

    `group_size` 决定 phase0 的 `global_pooling_group_size_phase_0`
    与 phase3 的 `kantor_A_spg_group_size_phase_3`（实测两者在同节点上必相等）。

    `numel` / `groups` 决定各相 bin 的元素数，写盘时按它分配。
    `is_attention_scores` 记下它是不是 attention scores 那一路 ——
    那 32 个节点整条当一组。
    """

    group_size: int
    numel: int
    is_attention_scores: bool
    head_index: int | None = None

    @property
    def groups(self) -> int:
        return self.numel // self.group_size


@dataclass(frozen=True)
class LayoutFeedback:
    """算子编译器对一个算子的布局与资源决策。

    与 `PhasePlan` 并列：PhasePlan 回答「这个算子分几相、每相多少字节」，
    本结构回答「它实际把 tile 切成多大、占多少 WRAM」。

    字段都可能是 None —— 算子编译器没算出来（或这条路径不适用）时不造假值，
    消费方据此退回静态规则。这与 `phase_value()` 的「取不到才退回常量表」
    是同一口径。

    `pim.tile-*` 与 `pim.wram-bytes-used` 是 `-pim-tile-to-budget` 的产出
    （FlagTree `TileToBudget.cpp`），只有走过那条 pass 的 A 路模块头才带；
    B 路（图编译器手写整算子级 IR）没有 tile 可切，全字段为 None。
    """

    # 实际切出的 tile 形状 `(m, n)`，来自 `pim.tile-m` / `pim.tile-n`。
    tile_shape: tuple[int, ...] | None = None
    # 这个 tile 实际占的 WRAM 字节数（`pim.wram-bytes-used`）。
    wram_bytes_used: int | None = None
    # 单台 DPU 的 WRAM 预算（`pim.wram-bytes`）。与上一个成对才能判是否超限。
    wram_bytes_budget: int | None = None
    # 单台 DPU 的 MRAM 预算（`pim.mram-bytes`）。**是我们下发的预算回显，
    # 不是实测占用** —— 实测值 8589934592 正是 `mram_bytes_per_dpu`。
    # 消费方是成本模型：它与同一份 IR 里回传的单台占用对照，占用超过预算时
    # 记一条 note（两个数矛盾说明它们不是同一套配置下算出来的）。
    mram_bytes: int | None = None

    @property
    def over_wram_budget(self) -> bool:
        """这个 tile 是否超了 WRAM 预算。两个字段缺一个就判 False（没结论）。"""
        if self.wram_bytes_used is None or self.wram_bytes_budget is None:
            return False
        return self.wram_bytes_used > self.wram_bytes_budget


# 回传载体的属性名。FlagTree 侧写入点见 `TileToBudget.cpp`；
# 名字在两个仓之间是契约，本仓只负责读回来，登记在此避免各处再写一遍字面量。
LAYOUT_FEEDBACK_ATTRS = (
    "pim.tile-m", "pim.tile-n", "pim.wram-bytes-used",
    "pim.wram-bytes", "pim.mram-bytes",
)


@dataclass(frozen=True)
class PlacementBack:
    """从 pimir 模块头读回来的 Placement 决策。

    与 `contracts/mlir_layout.placement_attribute` 写出去的是同一个载体
    （`#pim.placement`），两仓之间的契约。字段为 None 表示模块头没带 ——
    单 DPU 口径，不是错误。
    """

    kind: str | None = None            # shard / replicate / partial
    dim: int | None = None             # 切分维；非 shard 时为 None
    num_dpus: int | None = None

    # ---- 以下是 PIMMLIR 的**回传**，不是我们下发的 ----
    # `pim-tile-to-budget` 实际记到单台 DPU 头上的 MRAM 字节数。图编译器知道
    # 怎么切，但切完一台 DPU 到底占多少字节取决于分块 —— 而分块是那个 pass
    # 定的，所以这个数只能由它回传。
    placed_mram_bytes: int | None = None
    # 那个 pass 实际用的除数。与 `num_dpus` 分开记：前者是**意图**，这个是
    # **效果**，两者能对比才谈得上校验，否则只能假定一致。
    placed_shards: int | None = None
    # `partial` 档下，那个 pass 在每台 DPU 上为「接收对端那一份局部和」留的字节数；
    # `shard` / `replicate` 不欠归约，回传 0。同样只有它知道 —— 留多少取决于分块。
    placed_reduce_bytes: int | None = None
    # 那个 pass 定分块与 footprint 时实际按多少字节/元素算的（取自操作数类型）。
    # 这是 **dtype 维的回传**：文本层的消费方只看得见元素**类型名**，宽度要靠
    # 名字表猜，猜错会把由它推出的每个字节数一起带偏。
    placed_elem_bytes: int | None = None

    @property
    def is_sharded(self) -> bool:
        """这块张量是否真的被切开（而非复制或局部）。"""
        return self.kind == "shard" and (self.num_dpus or 1) > 1

    @property
    def intent_matches_effect(self) -> bool:
        """下发的切分意图与算子编译器实际用的除数是否一致。

        两者缺一个就判 True（没结论）—— 与 `LayoutFeedback.over_wram_budget`
        同口径：取不到就不下结论，不造假值。
        """
        if self.placed_shards is None or self.kind is None:
            return True
        expected = self.num_dpus if self.is_sharded else 1
        return self.placed_shards == expected


# `#pim.placement<kind = shard, dim = 1, numDpus = 2>` 里的字段。
# 名字与 FlagTree 的 ODS 参数名逐字对应，改名两边一起改。
# 属性体里可能嵌套方括号（order = [0, 1]），所以按括号配平取，不用 [^>]*。
_PLACEMENT_RE = re.compile(
    r"pim\.placement\s*=\s*#pim\.placement<((?:[^<>]|<[^<>]*>)*)>")

# PIMMLIR 侧 `pim-tile-to-budget` 的回传属性名（FlagTree `Dialect.h` 里的
# `AttrPlacedMramBytesName` / `AttrPlacedShardsName`），跨仓契约。
PLACED_MRAM_BYTES_ATTR = "pim.placed-mram-bytes"
PLACED_SHARDS_ATTR = "pim.placed-shards"
PLACED_REDUCE_BYTES_ATTR = "pim.placed-reduce-bytes"
PLACED_ELEM_BYTES_ATTR = "pim.placed-elem-bytes"

# 全部回传载体属性名，供跨仓校验用（`tests/test_flagtree_ods_hygiene.py` 的
# 「改名就失败」那条扫的就是这个集合）。与 `LAYOUT_FEEDBACK_ATTRS` 分开：
# 那一组是布局/分块的调试回传，这一组是四维的正式回程，两边写入点都在
# FlagTree 的 `TileToBudget.cpp`，名字登记在它的 `Dialect.h`。
PLACEMENT_FEEDBACK_ATTRS = (
    PLACED_MRAM_BYTES_ATTR, PLACED_SHARDS_ATTR,
    PLACED_REDUCE_BYTES_ATTR, PLACED_ELEM_BYTES_ATTR,
)


def placement_of_module(mlir_text: str) -> PlacementBack:
    """取模块头的 `#pim.placement`。没带就返回全 None。

    只看模块属性字典那一处，与 `module_int_attrs` 同口径：正文里同名的字符串
    （注释、算子属性）不算数。
    """
    head = re.search(r"module\s+attributes\s*\{", mlir_text)
    if head is None:
        return PlacementBack()
    m = _PLACEMENT_RE.search(mlir_text, head.start())
    if m is None:
        # 没有下发 placement，但回传字段可能在（单 DPU 也会回传除数 1）。
        attrs = module_int_attrs(mlir_text)
        return PlacementBack(
            placed_mram_bytes=attrs.get(PLACED_MRAM_BYTES_ATTR),
            placed_shards=attrs.get(PLACED_SHARDS_ATTR),
            placed_reduce_bytes=attrs.get(PLACED_REDUCE_BYTES_ATTR),
            placed_elem_bytes=attrs.get(PLACED_ELEM_BYTES_ATTR))
    body = m.group(1)

    def field(name: str) -> str | None:
        f = re.search(rf"\b{name}\s*=\s*([A-Za-z0-9_-]+)", body)
        return f.group(1) if f else None

    dim = field("dim")
    dpus = field("numDpus")
    attrs = module_int_attrs(mlir_text)
    return PlacementBack(
        kind=field("kind"),
        dim=None if dim is None else int(dim),
        # numDpus 省略时 ODS 的默认值是 1。
        num_dpus=1 if dpus is None else int(dpus),
        placed_mram_bytes=attrs.get(PLACED_MRAM_BYTES_ATTR),
        placed_shards=attrs.get(PLACED_SHARDS_ATTR),
        placed_reduce_bytes=attrs.get(PLACED_REDUCE_BYTES_ATTR),
        placed_elem_bytes=attrs.get(PLACED_ELEM_BYTES_ATTR),
    )


def module_int_attrs(mlir_text: str) -> dict[str, int]:
    """取 `module attributes {...}` 里的整数属性，名字去引号。

    只看模块属性字典那一处：正文里同名的字符串（注释、算子属性）不算数。
    只认整数属性（`= 64 : i32` 这种），字符串属性由调用方各自处理。

    这是「模块属性 → 数值」的唯一实现。原先 `opcompiler_bridge/phase_source.py`
    与 `genesim_bridge/ir_cost.py` 各有一份正则，两处会各自漂移。
    """
    head = re.search(r"module\s+attributes\s*\{([^}]*)\}", mlir_text)
    if head is None:
        return {}
    return {name: int(value) for name, value in re.findall(
        r'"?([A-Za-z][\w.\-]*)"?\s*=\s*(\d+)\s*:', head.group(1))}


def layout_feedback_of_module(mlir_text: str) -> LayoutFeedback:
    """从展开后 IR 的模块头解析出布局回传。

    模块级属性对全 kernel 生效，所以同一次编译里各 op 共享同一份。
    模块头不带这些属性时全字段为 None —— 合法的「算子编译器没意见」。
    """
    attrs = module_int_attrs(mlir_text)
    m, n = attrs.get("pim.tile-m"), attrs.get("pim.tile-n")
    return LayoutFeedback(
        tile_shape=None if m is None or n is None else (m, n),
        wram_bytes_used=attrs.get("pim.wram-bytes-used"),
        wram_bytes_budget=attrs.get("pim.wram-bytes"),
        mram_bytes=attrs.get("pim.mram-bytes"),
    )
