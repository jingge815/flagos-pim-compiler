"""把批量 attention 拆成逐头的硬件算子链。

这一步贡献参考产物 200 个节点里的 **128 个（64%）**，是粒度对齐的主体。

`scaled_dot_product_attention` 在 aten 里是**一个**算子，在 GML 里展开成：

    Split(Q) ──┬─→ 头 h: MatMul(QKᵀ) → Mask → Softmax → DQ → MatMul(PV) ─┬─→ Concat
               │                                                          │
            （32 头）                                                （32 头）

实测参考产物（Llama2-7B，32 头）的节点数：

| op_type | 个数 | 来自 |
| --- | ---: | --- |
| `MatMul` | 64 | 每头两个（QKᵀ 与 PV） |
| `Softmax` | 32 | 每头一个 |
| `Mask` | 32 | 每头一个 |
| `DynamicScaling` | 32（另有 4 个在 attention 外） | 每头一个，量化 attention scores |
| `Split` / `Concat` | 3 / 1 | 拆头与合头 |

两个 MatMul 的差异（实测 32/32 各自一致）：

| | matmul1 (QKᵀ) | matmul2 (PV) |
| --- | --- | --- |
| `weight_format` | `weights_transpose` | `weight` |
| `Scaling_buffer_file` | **1/√head_dim** | 1.0 |
| `split_channel_number` | 有（头下标） | 无 |

**与 `fuse_pim` 一样：只加节点、不删 SDPA**，靠 `ABSORBED_META_KEY` 让 GML 侧
跨过原算子。这样 fx 图仍能执行、数值不变（见 `fuse_pim` 那次返工的教训）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch.fx import Graph, GraphModule, Node

from graph.fuse_pim import ABSORBED_META_KEY, ATTENTION_SCALE_META_KEY

# 逐头展开产出的节点用这个键标注它在 attention 里的角色，
# GML 侧按角色填 weight_format / Scaling_buffer_file / split_channel_number。
HEAD_ROLE_META_KEY = "pim_head_role"
HEAD_INDEX_META_KEY = "pim_head_index"

# 角色取值。
ROLE_MATMUL_QK = "matmul1"
ROLE_MASK = "mask"
ROLE_SOFTMAX = "softmax"
ROLE_DQ = "dq"
ROLE_MATMUL_PV = "matmul2"
ROLE_SPLIT = "split"
ROLE_CONCAT = "concat"


@dataclass
class HeadExpansion:
    """一次逐头展开的统计。"""

    attentions: int = 0
    heads: int = 0
    nodes_added: int = 0
    dropped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        detail = (f"{self.attentions} 个 attention 展开成 {self.heads} 头，"
                  f"新增 {self.nodes_added} 个节点")
        if self.dropped:
            detail += f"；跳过 {len(self.dropped)} 个（{self.dropped[0]}）"
        return detail


def _head_count(query: Node) -> int | None:
    """从 Q 的形状取头数。SDPA 的输入是 `[batch, heads, seq, head_dim]`。"""
    value = query.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        return None
    return int(shape[1])


def _head_dim(query: Node) -> int | None:
    value = query.meta.get("val")
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        return None
    return int(shape[3])


def _set_shape(node: Node, example: "torch.Tensor | None") -> None:
    """给新建节点写上 `meta["val"]`。

    fx 的 `call_function` 不会自动推形状，而后续 pass（量化插 DQ、GML 发射边的
    dims）都要读它。缺了就只能跳过那个节点 —— 实测表现为
    「取不到 mha_softmax_head0 的形状」而静默少插 DQ。
    """
    if example is not None:
        node.meta["val"] = example


def _rename(graph: Graph, node: Node, wanted: str) -> None:
    """给节点起名，重名时由 fx 自动加后缀。

    **不能直接赋 `node.name`** —— 多个 attention 块（多层模型、或一层里有多个
    SDPA）会产出同名节点，fx 立刻抛 `Node redefined name`。
    `graph._graph_namespace` 负责去重，第二个 `split_q_head0` 变成
    `split_q_head0_1`，标签因此仍然可读、也仍然唯一。

    `create_name` 的第二个参数必须传 `None`：传 `node` 会把名字**登记到该节点
    名下**，随后 `node.name = ...` 的 setter 认为这个名字已被占用，于是
    静默保留原名（`matmul_default_8` 之类），改名等于没做。
    """
    node.name = graph._graph_namespace.create_name(wanted, None)


def _expand_one(
    graph: Graph, sdpa: Node, expansion: HeadExpansion
) -> bool:
    """把一个 SDPA 展开成逐头链。返回是否真的展开了。

    新节点插在 SDPA **之前**，最后用一个 `cat` 汇合、让 SDPA 的下游改读它。
    SDPA 本身留在图里但标记为已吸收 —— 这样图仍可执行（下游读的是等价结果），
    GML 侧则只看到逐头的那些节点。
    """
    if len(sdpa.args) < 3:
        expansion.dropped.append(f"{sdpa.name}: 参数不足 3 个")
        return False

    query, key, value = sdpa.args[0], sdpa.args[1], sdpa.args[2]
    mask = sdpa.args[3] if len(sdpa.args) > 3 else sdpa.kwargs.get("attn_mask")
    if not all(isinstance(n, Node) for n in (query, key, value)):
        expansion.dropped.append(f"{sdpa.name}: q/k/v 不都是张量节点")
        return False

    heads = _head_count(query)
    head_dim = _head_dim(query)
    if not heads or not head_dim:
        expansion.dropped.append(f"{sdpa.name}: 取不到头数/head_dim")
        return False

    # scale 优先用 SDPA 自己声明的，没有则按 1/√head_dim。
    scale = sdpa.kwargs.get("scale")
    if not isinstance(scale, (int, float)):
        scale = 1.0 / math.sqrt(head_dim)

    before = len(graph.nodes)
    per_head_outputs: list[Node] = []

    # 用 meta 里的样例张量推形状。取 fake tensor 做一次符号化运算即可 ——
    # 只要形状与 dtype，不关心数值。
    q_val = query.meta.get("val")
    k_val = key.meta.get("val")
    v_val = value.meta.get("val")
    mask_val = mask.meta.get("val") if isinstance(mask, Node) else None

    def head_slice(sample, index: int):
        return sample[:, index:index + 1] if sample is not None else None

    with graph.inserting_before(sdpa):
        for head in range(heads):
            def slice_head(source: Node, name: str) -> Node:
                """取第 `head` 个头：`source[:, head:head+1]`。

                用 slice 而不是 select，保留 4 维形状 —— GML 的边形状是
                `1x1x seq x head_dim`，少一维会让 dims 对不上。
                """
                node = graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (source, 1, head, head + 1),
                )
                _set_shape(node, head_slice(source.meta.get("val"), head))
                _rename(graph, node, f"{name}_head{head}")
                node.meta[HEAD_INDEX_META_KEY] = head
                node.meta[ROLE_SPLIT] = True
                node.meta[HEAD_ROLE_META_KEY] = ROLE_SPLIT
                return node

            q_head = slice_head(query, "split_q")
            k_head = slice_head(key, "split_k")
            v_head = slice_head(value, "split_v")

            # matmul1：QKᵀ。转置 K 的最后两维。
            k_t = graph.call_function(
                torch.ops.aten.transpose.int, (k_head, -2, -1))
            k_head_val = head_slice(k_val, head)
            _set_shape(
                k_t, k_head_val.transpose(-2, -1) if k_head_val is not None else None)
            k_t.meta[ABSORBED_META_KEY] = True

            scores = graph.call_function(
                torch.ops.aten.matmul.default, (q_head, k_t))
            q_head_val = head_slice(q_val, head)
            scores_val = (
                q_head_val @ k_head_val.transpose(-2, -1)
                if q_head_val is not None and k_head_val is not None else None)
            _set_shape(scores, scores_val)
            _rename(graph, scores, f"mha_batch_matmul1_head{head}")
            scores.meta[HEAD_ROLE_META_KEY] = ROLE_MATMUL_QK
            scores.meta[HEAD_INDEX_META_KEY] = head
            # attention scale 折进本节点的定标系数（不单独成节点）。
            scores.meta[ATTENTION_SCALE_META_KEY] = float(scale)

            scaled = graph.call_function(
                torch.ops.aten.mul.Tensor, (scores, float(scale)))
            _set_shape(scaled, scores_val)
            scaled.meta[ABSORBED_META_KEY] = True

            # Mask：加上 causal mask。没有 mask 时跳过这一步。
            current = scaled
            if isinstance(mask, Node):
                masked = graph.call_function(
                    torch.ops.aten.add.Tensor, (current, mask))
                _set_shape(
                    masked,
                    scores_val + mask_val
                    if scores_val is not None and mask_val is not None
                    else scores_val)
                _rename(graph, masked, f"mha_mask_head{head}")
                masked.meta[HEAD_ROLE_META_KEY] = ROLE_MASK
                masked.meta[HEAD_INDEX_META_KEY] = head
                current = masked

            # Softmax。
            probs = graph.call_function(
                torch.ops.aten._softmax.default, (current, -1, False))
            _set_shape(probs, current.meta.get("val"))
            _rename(graph, probs, f"mha_softmax_head{head}")
            probs.meta[HEAD_ROLE_META_KEY] = ROLE_SOFTMAX
            probs.meta[HEAD_INDEX_META_KEY] = head

            # matmul2：PV。
            context = graph.call_function(
                torch.ops.aten.matmul.default, (probs, v_head))
            probs_val = probs.meta.get("val")
            v_head_val = head_slice(v_val, head)
            _set_shape(
                context,
                probs_val @ v_head_val
                if probs_val is not None and v_head_val is not None else None)
            _rename(graph, context, f"mha_batch_matmul2_head{head}")
            context.meta[HEAD_ROLE_META_KEY] = ROLE_MATMUL_PV
            context.meta[HEAD_INDEX_META_KEY] = head

            per_head_outputs.append(context)

        # Concat：把 32 头拼回 `[batch, heads, seq, head_dim]`。
        merged = graph.call_function(
            torch.ops.aten.cat.default, (per_head_outputs, 1))
        _set_shape(merged, sdpa.meta.get("val"))
        _rename(graph, merged, "mha_concat")
        merged.meta[HEAD_ROLE_META_KEY] = ROLE_CONCAT

    # 下游改读拼回来的结果；SDPA 自己被标记为已吸收。
    sdpa.replace_all_uses_with(merged)
    sdpa.meta[ABSORBED_META_KEY] = True

    expansion.attentions += 1
    expansion.heads += heads
    expansion.nodes_added += len(graph.nodes) - before
    return True


def split_attention_heads(gm: GraphModule) -> HeadExpansion:
    """把图里每个 `scaled_dot_product_attention` 展开成逐头链。

    `replace_all_uses_with` 之后 SDPA 仍在图里（标记为已吸收），
    所以**图的可执行语义不变**：下游读的是逐头链算出的等价结果。
    """
    expansion = HeadExpansion()
    targets = [
        node for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.aten.scaled_dot_product_attention.default
        and not node.meta.get(ABSORBED_META_KEY)
    ]

    for sdpa in targets:
        _expand_one(gm.graph, sdpa, expansion)

    gm.graph.lint()
    gm.recompile()
    return expansion
