"""把 aten 算子级的图折成硬件算子级：RMSNorm 六合一、Gemm+SiLU、Mask 吸收缩放。

`graph/fuse.py` 做的是「主算子 + 尾部激活」这一种通用折叠。本模块做的是
**llama2 特有的固定模式**，它们在 aten 里是一串算子，在 GML 里是一个硬件节点。

三种模式，都以实物为准（见 docs/gml-parser-output-plan-20260917.md §4 步骤 ①）：

| aten 形态 | GML 节点 | 依据 |
| --- | --- | --- |
| `pow -> mean -> add -> rsqrt -> mul -> (to) -> mul` | 1 个 `RMSNorm_vpu` | 实物 2 个，带 `vpu_params` 子块 |
| `linear -> silu` | 1 个 `Gemm` + `contraction[fused_Silu_act]` | 实物节点 195 |
| `div(√d) -> add(mask)` | 1 个 `Mask`，`1/√d` 折进上游 matmul 的定标 | 实物 32 个（逐头） |

**为什么不复用 `fuse.py`**：那里的 `ACTIVATIONS` 有意排除了 `silu` 与 `rsqrt`
（当时判断它们在实物里是独立节点）。实测 **SiLU 折在 Gemm 195 的
`contraction[fused_Silu_act]` 里**，`op_type "Silu"` 与 `"Lut"` 各 1 次都出现在
那个嵌套块内、不是顶层节点。所以这里改判，但不动 `fuse.py` 的既有语义
——它还服务 ResNet 那条路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.fx import GraphModule, Node

from contracts.graph_meta import FUSED_TAIL_META_KEY

# RMSNorm 在 aten 里的算子链。顺序固定，中间可能夹 `to` 与 `_assert_tensor_metadata`
# （dtype 提升与断言），跳过它们不影响语义。
RMS_NORM_CHAIN = (
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.mean.dim,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.mul.Tensor,
)

# 遍历算子链时可以跳过的「透明」节点：只做 dtype 转换或断言，不改数值。
TRANSPARENT = (
    torch.ops.aten.to.dtype,
    torch.ops.aten._assert_tensor_metadata.default,
    torch.ops.aten.alias.default,
)

# 可以折进主算子的激活。与 fuse.py 的 ACTIVATIONS 不同，这里**包含 silu**。
FUSABLE_ACTIVATIONS = {
    torch.ops.aten.silu.default: "Silu",
    torch.ops.aten.relu.default: "Relu",
    torch.ops.aten.gelu.default: "Gelu",
}

# 能吃激活的主算子。
ACTIVATION_HOSTS = (
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
)

# 本模块自己的 meta 键。
RMS_NORM_META_KEY = "pim_rms_norm"
ATTENTION_SCALE_META_KEY = "pim_attention_scale"

# 被折进别的节点、因此**不该单独发射成 GML 节点**的算子。
#
# 折叠用「打标记」而不是「删节点」：删掉链中间的算子会让 fx 图不再可执行
# （锚点算的是 x²，不是整条 RMSNorm），语义等价性测试会失败。打标记则两者兼得
# —— 图照旧能跑出正确数值，GML 侧靠 `_is_emittable` 跨过这些节点。
#
# `gml_bridge.from_fx._tensor_inputs` 本来就会「跨过」不可映射的算子去找最近的
# 上游 GML 节点，所以这条路是现成的。
ABSORBED_META_KEY = "pim_absorbed"


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
class FusionReport:
    """一次折叠的统计，供调用方打印与测试断言。"""

    rms_norms: int = 0
    activations: int = 0
    attention_scales: int = 0

    @property
    def total(self) -> int:
        return self.rms_norms + self.activations + self.attention_scales

    def __str__(self) -> str:
        return (f"RMSNorm {self.rms_norms} 个、"
                f"融合激活 {self.activations} 个、"
                f"attention 定标 {self.attention_scales} 处")


def _skip_transparent(node: Node) -> Node:
    """顺着唯一消费者往下走，跳过只做 dtype 转换/断言的节点。"""
    current = node
    while True:
        users = [u for u in current.users
                 if u.target is not torch.ops.aten._assert_tensor_metadata.default]
        if len(users) != 1:
            return current
        nxt = users[0]
        if nxt.op != "call_function" or nxt.target not in TRANSPARENT:
            return current
        current = nxt


def _next_op(node: Node, target, skipped: list[Node] | None = None) -> Node | None:
    """紧跟 `node`（可跨透明节点）且唯一消费它的、目标为 `target` 的节点。

    跨过的透明节点记进 `skipped` —— 折叠时必须把它们一起删掉，否则它们仍然
    引用链中间的节点，导致 `rsqrt` 之类删不掉、在 GML 里冒出多余的
    `RMSNorm_vpu`（`OP_TYPES` 把 `rsqrt` 也映射到 RMSNorm_vpu）。
    """
    anchor = node
    while True:
        users = [u for u in anchor.users
                 if u.target is not torch.ops.aten._assert_tensor_metadata.default]
        if len(users) != 1:
            return None
        nxt = users[0]
        if nxt.op == "call_function" and nxt.target is target:
            return nxt
        if nxt.op != "call_function" or nxt.target not in TRANSPARENT:
            return None
        if skipped is not None:
            skipped.append(nxt)
        anchor = nxt


def _match_rms_norm(start: Node) -> RmsNormFusion | None:
    """从 `pow` 开始匹配 RMSNorm 的六算子链。

    形态（实测，节点号取自真实 llama2 单层图）：

        24 to     ──┬─────────────────────────────┐
        25 pow  ◄───┘                             │
        26 mean                                   │
        27 add   (+eps)                           │
        28 rsqrt                                  │
        29 mul  ◄─────────────────────────────────┘   （原值 × rsqrt）
        31 to
        32 mul   (× layernorm.weight)

    最后那个 `mul` 的另一个操作数是权重，所以要区分：`mul_2` 吃的是
    `to_7 × rsqrt`（都是激活），`mul_3` 吃的是 `weight × to_8`（一个是 get_attr）。
    """
    if start.op != "call_function" or start.target is not RMS_NORM_CHAIN[0]:
        return None

    chain = [start]
    passthrough: list[Node] = []
    for target in RMS_NORM_CHAIN[1:]:
        nxt = _next_op(chain[-1], target, passthrough)
        if nxt is None:
            return None
        chain.append(nxt)

    add_node = chain[2]
    # eps 是 add 的第二个标量参数。
    epsilon = add_node.args[1] if len(add_node.args) > 1 else None
    if not isinstance(epsilon, (int, float)):
        return None

    # 缩放那一步：紧跟第一个 mul 的第二个 mul，其操作数里有 get_attr。
    tail_skipped: list[Node] = []
    scale_mul = _next_op(chain[-1], torch.ops.aten.mul.Tensor, tail_skipped)
    weight_node = None
    eaten = list(chain)
    if scale_mul is not None:
        weight = next(
            (a for a in scale_mul.all_input_nodes if a.op == "get_attr"), None)
        if weight is not None:
            weight_node = weight
            passthrough.extend(tail_skipped)
            eaten.append(scale_mul)

    return RmsNormFusion(
        epsilon=float(epsilon), weight_node=weight_node,
        eaten=eaten, passthrough=passthrough)


def _fuse_rms_norms(gm: GraphModule) -> int:
    """把每条 RMSNorm 链标注成一个 `RMSNorm_vpu` 节点。

    **锚点取链尾那个乘权重的 `mul`**，不是链首的 `pow`。两个理由：

    1. 语义：锚点是整条链真正的输出节点，图照旧可执行、数值不变。
       取 `pow` 当锚点则要删掉链中间的算子，图就只算 x² 了。
    2. 拓扑：GML 节点的输出应当接到 RMSNorm 的下游，链尾天然满足。

    链上其余算子打 `ABSORBED_META_KEY`，GML 侧跨过它们。**一个节点都不删** ——
    `pow`/`mean`/`rsqrt` 本来就不在 `OP_TYPES` 里（不会误发射），
    而 `add`/`mul` 在，所以必须靠这个标记挡住。
    """
    fused = 0
    starts = [n for n in gm.graph.nodes
              if n.op == "call_function" and n.target is RMS_NORM_CHAIN[0]]

    for start in starts:
        if ABSORBED_META_KEY in start.meta:
            continue
        match = _match_rms_norm(start)
        if match is None:
            continue

        anchor = match.eaten[-1]
        anchor.meta[RMS_NORM_META_KEY] = match
        for node in match.eaten[:-1] + match.passthrough:
            node.meta[ABSORBED_META_KEY] = True
        fused += 1

    return fused


def _fuse_activations(gm: GraphModule) -> int:
    """把 `linear -> silu` 折成一个带 contraction 的 Gemm。

    复用 `FUSED_TAIL_META_KEY`，这样 `gml_bridge.from_fx._contraction_of`
    不必改就能发射 `contraction[fused_..._activation]` 块。
    """
    from graph.fuse import FusedTail

    fused = 0
    hosts = [n for n in gm.graph.nodes
             if n.op == "call_function" and n.target in ACTIVATION_HOSTS]

    for host in hosts:
        if FUSED_TAIL_META_KEY in host.meta:
            continue
        users = [u for u in host.users
                 if u.target is not torch.ops.aten._assert_tensor_metadata.default]
        if len(users) != 1:
            continue
        activation = users[0]
        if (activation.op != "call_function"
                or activation.target not in FUSABLE_ACTIVATIONS):
            continue

        host.meta[FUSED_TAIL_META_KEY] = FusedTail(
            activation=FUSABLE_ACTIVATIONS[activation.target],
            pool=None,
            nodes=[activation],
        )
        # 同样只打标记不删：激活留在图里，图仍可执行；GML 侧跨过它，
        # 由主算子的 contraction 块表达。
        activation.meta[ABSORBED_META_KEY] = True
        fused += 1

    return fused


def _absorb_attention_scale(gm: GraphModule) -> int:
    """把 QK^T 后的 `1/√head_dim` 折进上游 matmul 的定标系数。

    实物里这个缩放**没有单独成节点**，而是写进 matmul1 的
    `Scaling_buffer_file`（实测 32 个 matmul1 全是 1/√128 = 0.088388）。

    漏掉它等于丢掉 attention scale —— 数值全错，而结构校验查不出来，
    所以这一步必须显式做。

    注：真实 llama2 的 SDPA 把缩放藏在 `scaled_dot_product_attention` 内部，
    图里看不到独立的 div。这里处理的是**已经拆开 SDPA 之后**的形态
    （逐头展开 pass 会产出它），以及手写 attention 的形态。
    """
    import math

    absorbed = 0
    divisions = [
        n for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target in (torch.ops.aten.div.Tensor, torch.ops.aten.mul.Tensor)
    ]

    for node in divisions:
        if len(node.args) < 2 or not isinstance(node.args[1], (int, float)):
            continue
        producer = node.args[0]
        if not isinstance(producer, Node) or producer.op != "call_function":
            continue
        if producer.target not in (torch.ops.aten.bmm.default,
                                   torch.ops.aten.matmul.default,
                                   torch.ops.aten.mm.default):
            continue

        scalar = float(node.args[1])
        # div 除以 √d，mul 乘以 1/√d —— 统一成「乘上的系数」。
        factor = 1.0 / scalar if node.target is torch.ops.aten.div.Tensor else scalar
        if not 0.0 < factor < 1.0:
            continue

        # 只吸收真的是 1/√head_dim 的系数。
        #
        # 判据必须**限定 head_dim 是 2 的幂**，不能只查「factor 是某个整数的
        # 平方根倒数」—— 后者太松：随手写的 `/3.0` 会被反解成 head_dim=9
        # 且 √9=3 精确成立，于是把无关的缩放也吃掉。
        # 真实 head_dim = hidden_size / num_heads，两者都是 2 的幂。
        head_dim = round(1.0 / (factor * factor))
        if head_dim < 8 or head_dim & (head_dim - 1):
            continue
        if abs(factor - 1.0 / math.sqrt(head_dim)) > 1e-3:
            continue

        producer.meta[ATTENTION_SCALE_META_KEY] = factor
        node.meta[ABSORBED_META_KEY] = True
        absorbed += 1

    return absorbed


def fuse_for_pim(gm: GraphModule) -> FusionReport:
    """按硬件算子粒度折叠整张图，返回统计。

    顺序有讲究：先折 RMSNorm（它的链最长、最容易被别的 pass 打断），
    再折激活，最后吸收 attention 定标。
    """
    report = FusionReport()
    report.rms_norms = _fuse_rms_norms(gm)
    report.activations = _fuse_activations(gm)
    report.attention_scales = _absorb_attention_scale(gm)

    # 不删节点，所以不需要 DCE 也不需要 recompile —— 图的可执行形态没变，
    # 变的只是 meta 标注。lint 仍跑一遍确认图结构没被弄坏。
    gm.graph.lint()
    return report
