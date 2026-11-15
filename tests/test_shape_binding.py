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


def test_bind_rewrites_literal_shape_lists() -> None:
    """view/reshape 的字面量形状列表按元素绑定，None 元素保留原值。"""
    cmd = Command(
        4, "launch", 0,
        {"arg_kinds": ["tensor", [1, 16, -1, 16]],
         "arg_elem_exprs": [None, [None, _t_expr(), None, None]]})
    bound = bind_command_shapes(cmd, {"T": 5})
    assert bound.payload["arg_kinds"] == ["tensor", [1, 5, -1, 16]]
    assert cmd.payload["arg_kinds"] == ["tensor", [1, 16, -1, 16]]


def test_bind_rewrites_scalar_shape_args() -> None:
    """标量形状实参（如 arange 的终点）也按表达式求值。"""
    cmd = Command(
        5, "launch", 0,
        {"arg_kinds": [0, 16], "arg_elem_exprs": [None, _t_expr()]})
    assert bind_command_shapes(cmd, {"T": 7}).payload["arg_kinds"] == [0, 7]


def test_bind_skips_layouts_that_do_not_shrink_as_a_prefix() -> None:
    """T 不在最外层非单位维时不绑定：内层步长会变，包络缓冲无法前缀读。"""
    # (1, 4, T, 16)：T 在 heads 之后，实际布局与包络不同。
    middle = Command(
        6, "launch", 0,
        {"out_shape": [1, 4, 16, 16],
         "out_shape_expr": [None, None, _t_expr(), None]})
    assert bind_command_shapes(middle, {"T": 5}) is middle
    # 多 T 轴 (1, 1, T, T)：行宽也随 T 变。
    multi = Command(
        7, "launch", 0,
        {"out_shape": [1, 1, 16, 16],
         "out_shape_expr": [None, None, _t_expr(), _t_expr()]})
    assert bind_command_shapes(multi, {"T": 5}) is multi
    # 输入实参不满足前缀布局时，整条命令都不绑定。
    arg_bad = Command(
        8, "launch", 0,
        {"arg_shapes": [[1, 4, 16, 16], [16, 16]],
         "arg_shape_exprs": [[None, None, _t_expr(), None], None],
         "out_shape": [1, 16, 16],
         "out_shape_expr": [None, _t_expr(), None]})
    assert bind_command_shapes(arg_bad, {"T": 5}) is arg_bad
    # 前导维全是 1 时可以前缀绑定。
    ok = Command(
        9, "launch", 0,
        {"out_shape": [1, 1, 16, 16],
         "out_shape_expr": [None, None, _t_expr(), None]})
    assert bind_command_shapes(ok, {"T": 5}).payload["out_shape"] == [1, 1, 5, 16]
