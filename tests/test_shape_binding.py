"""验证运行期形状绑定：按实际 T 求值 payload 表达式。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.exec_plan import Command
from contracts.runtime_graph import ShapeExpr
from runtime.shape_bind import bind_command_shapes


def _t_expr(n: int = 1):
    expr = ShapeExpr.dim("T")
    for _ in range(n - 1):
        expr = ShapeExpr.mul(expr, ShapeExpr.dim("T"))
    return expr.to_dict()


def test_bind_rewrites_shapes_for_symbolic_dims() -> None:
    cmd = Command(
        0, "launch", 1,
        {"arg_shapes": [[1, 16, 32], [32, 32]],
         "arg_shape_exprs": [[None, _t_expr(), None], None],
         "out_shape_expr": [None, _t_expr(), None],
         "out_shape": [1, 16, 32]},
    )
    bound = bind_command_shapes(cmd, {"T": 5})
    assert bound.payload["arg_shapes"] == [[1, 5, 32], [32, 32]]
    assert bound.payload["out_shape"] == [1, 5, 32]
    # 原命令不被改动（调度器每次绑定都基于原始计划）。
    assert cmd.payload["out_shape"] == [1, 16, 32]


def test_bind_leaves_commands_without_exprs() -> None:
    cmd = Command(1, "launch", 0, {"arg_shapes": [[4, 4]], "out_shape": [4, 4]})
    assert bind_command_shapes(cmd, {"T": 5}) is cmd


def test_bind_binds_symbolic_dims_and_keeps_static_dims() -> None:
    """混合维度逐维处理：符号维求值，None 维保留包络值。"""
    cmd = Command(
        2, "launch", 0,
        {"arg_shapes": [[16, 8]],
         "arg_shape_exprs": [[_t_expr(), None]]})
    bound = bind_command_shapes(cmd, {"T": 3})
    assert bound.payload["arg_shapes"] == [[3, 8]]


def test_bind_evaluates_composite_expressions() -> None:
    expr = ShapeExpr.add(ShapeExpr.dim("T"), ShapeExpr.const(2)).to_dict()
    cmd = Command(
        3, "launch", 0,
        {"out_shape": [18], "out_shape_expr": [expr]})
    bound = bind_command_shapes(cmd, {"T": 7})
    assert bound.payload["out_shape"] == [9]
