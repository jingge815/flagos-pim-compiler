"""运行期形状绑定：按实际 T 求值命令携带的形状表达式。"""

from __future__ import annotations

import os
from dataclasses import replace

from contracts.exec_plan import Command
from contracts.runtime_graph import ShapeExpr


def dynamic_shape_enabled() -> bool:
    """PIM_DYN_SHAPE=1 时，动态执行按实际 T 绑定内核形状。"""
    return os.environ.get("PIM_DYN_SHAPE", "") == "1"


def _bind_dims(entries, envelope, env) -> list | None:
    """逐维绑定：None 维保留包络值，表达式维按 env 求值。"""
    if entries is None or envelope is None or len(envelope) != len(entries):
        return None
    dims = []
    for entry, hint in zip(entries, envelope):
        if entry is None:
            dims.append(hint)
        else:
            dims.append(ShapeExpr.from_dict(entry).eval(env))
    return dims


def bind_command_shapes(cmd: Command, env: dict[str, int]) -> Command:
    """把命令 payload 的形状表达式求值成具体形状。

    只改写 `arg_shapes` / `out_shape`；地址（`mram_offset`/`reads`/`writes`）
    仍是包络规划的区间，缓冲按包络分配，绑定只收缩计算行数。
    """
    payload = cmd.payload
    arg_exprs = payload.get("arg_shape_exprs")
    out_exprs = payload.get("out_shape_expr")
    if arg_exprs is None and out_exprs is None:
        return cmd
    updates: dict[str, object] = {}
    if arg_exprs is not None:
        # 逐实参绑定：某实参没有表达式或维度数不齐时保留包络形状。
        shapes = list(payload["arg_shapes"])
        for index, entries in enumerate(arg_exprs):
            if index >= len(shapes):
                break
            bound = _bind_dims(entries, shapes[index], env)
            if bound is not None:
                shapes[index] = bound
        updates["arg_shapes"] = shapes
    if out_exprs is not None:
        bound = _bind_dims(out_exprs, payload.get("out_shape"), env)
        if bound is not None:
            updates["out_shape"] = bound
    if not updates:
        return cmd
    return replace(cmd, payload={**payload, **updates})
