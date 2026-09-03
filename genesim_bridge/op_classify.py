"""定义 GeneSim 算子类型与 FlagGems 代表实现及其形状。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

# 使用 GeneSim 模板成本的算子类型：本桥接不为它们编译 FlagGems 代表实现，
# 直接保留 model_parser 写进 IR 的模板系数，并记入 sidecar 的 coverage.template。
#
# 逐元素/归约类（RMSNORM、SILU、VECTOR_ADD、VECTOR_MUL）：GeneSim 侧的
# model_parser 已经给出 flops_coeffs / data_bytes_coeffs，而且它们在
# pim_compiler 里有各自的 trace 编译器，成本由那条路径负责。这里不硬编一套
# FlagGems 配方去覆盖，否则等于引入一组未经校准的数值。
#
# 图的边界节点（MODEL_INPUT、MODEL_OUTPUT）：零成本占位，flops 和 data_bytes
# 都是 0、也没有系数，本来就没有什么可测量的。
UNCOVERED_OP_TYPES = frozenset({
    "GELU",
    "RMSNORM",
    "SILU",
    "VECTOR_ADD",
    "VECTOR_MUL",
    "MODEL_INPUT",
    "MODEL_OUTPUT",
})


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
    """按模型维度返回各算子类型的配方工厂。"""
    hidden = dims["hidden_size"]
    head_dim = dims["head_dim"]

    def gemm(point: ShapePoint, in_features: int, out_features: int) -> OpRecipe:
        return OpRecipe(
            source_name="flag_gems.ops.linear",
            build=_linear(dims, point, in_features, out_features),
        )

    def gemv_score(point: ShapePoint) -> OpRecipe:
        # 注意力得分的矩阵乘形状。
        return OpRecipe(
            source_name="flag_gems.ops.bmm (score 代表实现)",
            build=_bmm(dims, point.tq, head_dim, point.lkv),
        )

    def softmax(point: ShapePoint) -> OpRecipe:
        # 注意力得分的 Softmax 形状。
        return OpRecipe(
            source_name="flag_gems.ops.softmax",
            build=_softmax(dims, point.tq, point.lkv),
        )

    def gemv_context(point: ShapePoint) -> OpRecipe:
        # 注意力上下文的矩阵乘形状。
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
    """从算子形状读取 GEMM 的输入和输出特征数。"""
    in_shape = op_dict["input_shapes"][0]
    out_shape = op_dict["output_shapes"][0]
    assert in_shape[0] == "Tq" and out_shape[0] == "Tq", \
        f"GEMM shape 不是 [Tq, N] 形式: {in_shape} -> {out_shape}"
    return int(in_shape[1]), int(out_shape[1])


def flash_attention_probe(dims: Dict[str, int], point: ShapePoint):
    """构造一次融合 attention 调用，用于测量总成本。"""
    import torch

    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    q = torch.randn(1, num_heads, point.tq, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, num_heads, point.lkv, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, num_heads, point.lkv, head_dim, device="cuda", dtype=torch.float16)
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
