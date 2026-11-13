"""验证 RuntimeGraph 契约：形状表达式、区域引用与可打印文本。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.runtime_graph import (
    RegionRef,
    RTOp,
    RTControl,
    RuntimeGraph,
    ShapeExpr,
    format_runtime_graph,
)


def test_shape_expr_evaluates_all_operators() -> None:
    T = ShapeExpr.dim("T")
    env = {"T": 7}
    assert ShapeExpr.const(3).eval(env) == 3
    assert ShapeExpr.add(T, ShapeExpr.const(2)).eval(env) == 9
    assert ShapeExpr.sub(T, ShapeExpr.const(1)).eval(env) == 6
    assert ShapeExpr.mul(T, ShapeExpr.const(4)).eval(env) == 28
    assert ShapeExpr.floordiv(T, ShapeExpr.const(2)).eval(env) == 3
    assert ShapeExpr.ceildiv(T, ShapeExpr.const(2)).eval(env) == 4
    assert ShapeExpr.prod(T, ShapeExpr.const(4)).eval(env) == 28
    assert ShapeExpr.slice(T, 1, 4).eval(env) == 3
    assert ShapeExpr.cat(T, ShapeExpr.const(1)).eval(env) == 8


def test_shape_expr_requires_bound_dim() -> None:
    with pytest.raises(KeyError):
        ShapeExpr.dim("T").eval({})


def test_shape_expr_round_trip_and_free_dims() -> None:
    expr = ShapeExpr.mul(ShapeExpr.dim("T"), ShapeExpr.const(4))
    assert expr.free_dims() == {"T"}
    assert ShapeExpr.from_dict(expr.to_dict()) == expr


def test_region_ref_and_runtime_graph_format() -> None:
    T = ShapeExpr.dim("T")
    ref = RegionRef(
        ("dpu", 0), ShapeExpr.const(16), ShapeExpr.mul(T, ShapeExpr.const(2)))
    assert ref.resolve({"T": 5}) == (("dpu", 0), 16, 10)
    graph = RuntimeGraph(
        ops=[RTOp(id=0, op="launch", dpu_id=0, reads=[ref],
                  shape_bindings={"out": ShapeExpr.mul(T, ShapeExpr.const(2))})],
        controls=[RTControl(id=1, kind="dispatch", cond="mask",
                            branches={"0": [0]})],
        inputs=["input_ids"],
        outputs=[0],
        symbolic_dims=["T"],
    )
    text = format_runtime_graph(graph)
    assert "runtime_graph dims=['T']" in text
    assert "op 0: launch dpu=0" in text
    assert "control 1: dispatch cond=mask" in text
    assert graph.op_by_id(0).op == "launch"
