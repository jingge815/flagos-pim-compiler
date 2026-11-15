"""验证通信条目从节点形状表达式推导 T 轴（D2-full 前置）。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from comm.plan import build_comm_plan, t_axes_from_exprs
from contracts.runtime_graph import ShapeExpr
from tests.test_comm_plan import _dpu_spec, _edge, _host_spec, REPLICATE


def _t() -> ShapeExpr:
    return ShapeExpr.dim("T")


def test_t_axes_from_exprs_marks_symbolic_dims() -> None:
    exprs = (None, ShapeExpr.mul(_t(), ShapeExpr.const(2)), None)
    assert t_axes_from_exprs((1, 16, 8), exprs) == (1,)
    # 4-D 掩码 (1,1,T,T)：两个 T 轴都要标出来，供后续显式拒绝行收缩。
    assert t_axes_from_exprs((1, 1, 16, 16), (None, None, _t(), _t())) == (2, 3)
    # 常量表达式不算符号维。
    assert t_axes_from_exprs((1, 16, 8), (None, ShapeExpr.const(16), None)) == ()


def test_t_axes_from_exprs_requires_matching_shape() -> None:
    assert t_axes_from_exprs((1, 16), None) == ()
    assert t_axes_from_exprs((1, 16), ()) == ()
    assert t_axes_from_exprs((1, 16), (None, _t(), None)) == ()


def test_build_comm_plan_attaches_exprs_by_source_node() -> None:
    shape = (1, 8, 4)
    src = _dpu_spec(REPLICATE, shape, (0, 1))
    edge = _edge(0, "scatter", _host_spec(), src, shape)
    exprs = {edge.src: (None, _t(), None)}
    entry = build_comm_plan([edge], exprs.get)[0]
    assert entry.t_axes == (1,)
    assert entry.shape_exprs == (None, _t(), None)


def test_build_comm_plan_without_lookup_keeps_static_metadata() -> None:
    shape = (1, 8, 4)
    src = _dpu_spec(REPLICATE, shape, (0, 1))
    edge = _edge(0, "scatter", _host_spec(), src, shape)
    entry = build_comm_plan([edge])[0]
    assert entry.t_axes == ()
    assert entry.shape_exprs == ()
