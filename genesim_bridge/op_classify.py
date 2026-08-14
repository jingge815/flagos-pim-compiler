"""算子归类表：GeneSim op_type ↔ FlagGems 代表实现 + 目标 shape 构造。

方案依据：spec.md 问题 4 二.算子边界假设，按实测修正后的 a+c 口径。

实测事实（gpt2 推理的 994 个 IR dump）：FlagGems 的 attention 走融合
flash 路线，整条推理里带 `tt.dot` 的 kernel 只有 linear/addmm/flash_fwd，
没有独立的 score / softmax / context kernel。而 GeneSim 图骨架把 attention
拆成 GEMV_SCORE + SOFTMAX + GEMV_CONTEXT（每层每 head 各一个，占 116 个
算子里的 96 个，且全在 PIM 侧）。方案原文假设的「GEMV 走 FlagGems 分离
实现、1:1 对齐」不成立。

因此采用 a+c：
  - GEMM（16 个）    -> FlagGems linear，边界天然 1:1，抽真实成本
  - GEMV_SCORE (32)  -> 用 bmm 按单 head shape 编，作代表实现（c）
  - SOFTMAX    (32)  -> 用 softmax 按单 head shape 编，作代表实现（c）
  - GEMV_CONTEXT(32) -> 用 bmm 按单 head shape 编，作代表实现（c）
  - GELU       (4)   -> 不覆盖，沿用 GeneSim 模板成本（a）

代表实现编出的 kernel 不是推理实跑的那个（实跑是融合的 flash_fwd），
但算子边界与 GeneSim 对齐，成本可逐个回填。flash_fwd 的总成本另记
sidecar 作交叉验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

# 不覆盖的算子：沿用 GeneSim 模板成本。GELU 属方案里的 B/C 类。
UNCOVERED_OP_TYPES = frozenset({"GELU"})


@dataclass(frozen=True)
class ShapePoint:
    """一个编译代表点：prefill 用 (Tq=seq_len, Tp=0)，decode 用 (Tq=1, Tp=seq_len)。"""
    tq: int
    tp: int

    @property
    def lkv(self) -> int:
        """Tp+Tq，即 attention 的 KV 长度。"""
        return self.tp + self.tq

    @property
    def label(self) -> str:
        return f"Tq={self.tq},Tp={self.tp}"


@dataclass
class OpRecipe:
    """一个 GeneSim op_type 的编译配方。"""
    source_name: str                       # FlagGems 实现名，写入 sidecar
    build: Callable                        # (dims, point) -> 无参可调用，执行该算子
    expected_kernels: int = 1


def _linear(dims: Dict[str, int], point: ShapePoint, in_features: int, out_features: int):
    """FlagGems linear：GeneSim 的 GEMM 都是 [Tq, in] x [in, out]。"""
    import torch

    x = torch.randn(point.tq, in_features, device="cuda", dtype=torch.float16)
    w = torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
    b = torch.randn(out_features, device="cuda", dtype=torch.float16)
    return lambda: torch.nn.functional.linear(x, w, b)


def _bmm(dims: Dict[str, int], m: int, k: int, n: int):
    """FlagGems bmm，单 head（batch=1）。"""
    import torch

    a = torch.randn(1, m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(1, k, n, device="cuda", dtype=torch.float16)
    return lambda: torch.bmm(a, b)


def _softmax(dims: Dict[str, int], rows: int, cols: int):
    import torch

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
    return lambda: torch.softmax(x, dim=-1)


def build_recipes(dims: Dict[str, int]) -> Dict[str, Callable[[ShapePoint], OpRecipe]]:
    """按模型维度返回每个 op_type 的配方工厂。

    dims 需含 hidden_size / head_dim / num_heads / ffn_dim。
    GEMM 有 4 种角色（QKV / proj / fc1 / fc2），shape 不同，故 GEMM 的
    配方按 (in_features, out_features) 参数化，由调用方从 Operator 的
    符号 input/output_shapes 里读出。
    """
    hidden = dims["hidden_size"]
    head_dim = dims["head_dim"]

    def gemm(point: ShapePoint, in_features: int, out_features: int) -> OpRecipe:
        return OpRecipe(
            source_name="flag_gems.ops.linear",
            build=_linear(dims, point, in_features, out_features),
        )

    def gemv_score(point: ShapePoint) -> OpRecipe:
        # score = Q[Tq, D] x K^T[D, Tp+Tq]
        return OpRecipe(
            source_name="flag_gems.ops.bmm (score 代表实现)",
            build=_bmm(dims, point.tq, head_dim, point.lkv),
        )

    def softmax(point: ShapePoint) -> OpRecipe:
        # softmax over [Tq, Tp+Tq]
        return OpRecipe(
            source_name="flag_gems.ops.softmax",
            build=_softmax(dims, point.tq, point.lkv),
        )

    def gemv_context(point: ShapePoint) -> OpRecipe:
        # context = P[Tq, Tp+Tq] x V[Tp+Tq, D]
        return OpRecipe(
            source_name="flag_gems.ops.bmm (context 代表实现)",
            build=_bmm(dims, point.tq, point.lkv, head_dim),
        )

    return {
        "GEMM": gemm,
        "GEMV_SCORE": gemv_score,
        "SOFTMAX": softmax,
        "GEMV_CONTEXT": gemv_context,
    }


def gemm_features(op_dict: dict) -> tuple:
    """从 GeneSim Operator 的符号 shape 读出 GEMM 的 (in_features, out_features)。

    GeneSim 里 GEMM 的 shape 形如 input [["Tq", 512]] / output [["Tq", 1536]]，
    第 0 维是符号 Tq，第 1 维是具体特征数。
    """
    in_shape = op_dict["input_shapes"][0]
    out_shape = op_dict["output_shapes"][0]
    assert in_shape[0] == "Tq" and out_shape[0] == "Tq", \
        f"GEMM shape 不是 [Tq, N] 形式: {in_shape} -> {out_shape}"
    return int(in_shape[1]), int(out_shape[1])


def flash_attention_probe(dims: Dict[str, int], point: ShapePoint):
    """交叉验证用：跑一次融合 flash attention，取其总成本。

    这是 FlagGems 实跑 attention 的真实路径。96 个 attention 算子的
    代表实现成本之和应与它同量级——量级差太远说明代表实现选错了。
    """
    import torch

    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    q = torch.randn(1, num_heads, point.tq, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, num_heads, point.lkv, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, num_heads, point.lkv, head_dim, device="cuda", dtype=torch.float16)
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
