"""验证 D0 改造：静态路径默认不变，动态入口显式未实现。"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.exec_plan import Access, ExecutionPlan
from runtime import dependency, exec_plan_gen
from runtime.compile import (
    CompiledModel,
    compile_llama2,
    export_annotated_graph,
)
from runtime.executor import execute_plan


def test_dynamic_entry_points_default_to_static() -> None:
    assert inspect.signature(compile_llama2).parameters["dynamic"].default is False
    assert (
        inspect.signature(export_annotated_graph)
        .parameters["dynamic_shapes"].default is None
    )
    assert inspect.signature(execute_plan).parameters["mode"].default == "static"

    fields = {item.name: item for item in dataclasses.fields(CompiledModel)}
    assert fields["compile_mode"].default == "static"
    assert fields["runtime_graph"].default is None
    assert fields["dynamic_axes"].default_factory is dict
    assert fields["kernel_classes"].default == ()


def test_unknown_execute_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        execute_plan(ExecutionPlan([]), None, mode="bogus")


def test_dependency_helpers_are_reexported() -> None:
    assert exec_plan_gen._deps_of is dependency.deps_of
    assert exec_plan_gen._war_waits is dependency.war_waits
    assert exec_plan_gen.overlap is dependency.overlap


def test_dependency_semantics() -> None:
    first = Access(("dpu", 0), 0, 16)
    second = Access(("dpu", 0), 8, 16)
    other_dpu = Access(("dpu", 1), 0, 16)
    assert dependency.overlap(first, second)
    assert not dependency.overlap(first, other_dpu)

    same_range = {("dpu", 0): [(first, 0), (first, 3)]}
    assert dependency.deps_of([Access(("dpu", 0), 10, 4)], same_range) == [3]
    overlapping = {("dpu", 0): [(first, 0), (second, 3)]}
    assert sorted(dependency.deps_of(
        [Access(("dpu", 0), 10, 4)], overlapping)) == [0, 3]

    pending = {(("dpu", 0), 0): ["reader"]}
    reader_cmds = {("reader", 0): [7]}
    assert dependency.war_waits(
        ("dpu", 0), 0, pending, reader_cmds) == [7]
