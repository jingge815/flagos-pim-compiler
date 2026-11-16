"""运行期形状绑定：按实际 T 求值命令携带的形状表达式。"""

from __future__ import annotations

import os
from dataclasses import replace

from contracts.exec_plan import Command
from contracts.runtime_graph import ShapeExpr


def runtime_static_mode() -> bool:
    """PIM_RUNTIME_MODE=static：动态编译产物运行期整体回退静态包络执行。"""
    return os.environ.get("PIM_RUNTIME_MODE", "") == "static"


def dynamic_shape_enabled() -> bool:
    """PIM_DYN_SHAPE=1 且运行期未回退静态时，动态执行按实际 T 绑定内核形状。"""
    return os.environ.get("PIM_DYN_SHAPE", "") == "1" and not runtime_static_mode()


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


def _t_axes(entries) -> list[int]:
    """形状表达式里依赖 T 的维。"""
    axes = []
    for axis, entry in enumerate(entries or ()):
        if entry is None:
            continue
        if "T" in ShapeExpr.from_dict(entry).free_dims():
            axes.append(axis)
    return axes


def _bindable(shape, entries) -> bool:
    """T 行收缩是否保持行主序前缀布局。

    只有 T 是唯一符号维、且排在所有非单位维之前时，实际形状的前几行才和
    包络缓冲逐一对应；T 在中间维（如 (1,heads,T,head_dim)）会改变内层步长，
    多 T 轴（如 (1,1,T,T)）连行宽都会变，都不能按前缀绑定。
    """
    if not entries:
        return True
    axes = _t_axes(entries)
    if not axes:
        return True
    if len(axes) != 1:
        return False
    axis = axes[0]
    return axis < len(shape) and all(dim == 1 for dim in shape[:axis])


def _command_bindable(payload) -> bool:
    """命令的输入/输出形状都能安全收缩时才绑定。"""
    out_exprs = payload.get("out_shape_expr")
    if out_exprs is not None and not _bindable(
            payload.get("out_shape") or (), out_exprs):
        return False
    arg_exprs = payload.get("arg_shape_exprs") or ()
    arg_shapes = payload.get("arg_shapes") or ()
    for entries, shape in zip(arg_exprs, arg_shapes):
        if not _bindable(shape or (), entries):
            return False
    return True


def _bind_literal(expr, hint, env):
    """字面量形状元素：表达式求值，None 保留折叠时的包络值。"""
    if expr is None:
        return hint
    return ShapeExpr.from_dict(expr).eval(env)


def bind_command_shapes(cmd: Command, env: dict[str, int]) -> Command:
    """把命令 payload 的形状表达式求值成具体形状。

    改写 `arg_shapes` / `out_shape` 与形状列表实参；地址
    （`mram_offset`/`reads`/`writes`）仍是包络规划的区间，缓冲按包络分配，
    绑定只收缩计算行数。
    """
    payload = cmd.payload
    arg_exprs = payload.get("arg_shape_exprs")
    out_exprs = payload.get("out_shape_expr")
    elem_exprs = payload.get("arg_elem_exprs")
    if arg_exprs is None and out_exprs is None and elem_exprs is None:
        return cmd
    if not _command_bindable(payload):
        return cmd
    updates: dict[str, object] = {}
    if elem_exprs is not None:
        # 字面量形状列表逐元素绑定；标量实参直接换成实际值。
        kinds = list(payload["arg_kinds"])
        for index, entry in enumerate(elem_exprs):
            if entry is None or index >= len(kinds):
                continue
            if isinstance(entry, list):
                literal = kinds[index]
                if not isinstance(literal, (list, tuple)) \
                        or len(literal) != len(entry):
                    continue
                kinds[index] = [
                    _bind_literal(expr, hint, env)
                    for expr, hint in zip(entry, literal)]
            else:
                kinds[index] = _bind_literal(entry, kinds[index], env)
        updates["arg_kinds"] = kinds
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
