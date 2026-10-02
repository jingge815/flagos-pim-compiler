"""Memory Layout 的排布层（第 3 层）。

切分与地址原先已有，缺的是「这段字节内部怎么摆」。本轮把
`bytes_of()` 里隐含的「行主序紧密排列」变成显式字段，并让对齐只有一份实现。
"""

from __future__ import annotations

import re
import sys
from math import prod
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.mem_layout import align_up, row_major_strides
from contracts.pim_tensor_spec import TensorShardDetail
from graph.partition import partition_graph
from graph.spec_prop import propagate_specs
from memory.mem_planner import HwBudget, bytes_of


def test_default_strides_are_byte_identical_to_the_old_formula() -> None:
    """空步幅必须与改动前的 prod(shape)*itemsize 完全相同。

    这是 P0-4「只加字段不改行为」的直接证据：三个调用点走的还是原路径。
    """
    for shape in [(3, 4), (1, 4096), (32, 128), (11008, 4096), (1,)]:
        for itemsize in (1, 2, 4):
            assert bytes_of(shape, itemsize) == bytes_of(shape, itemsize, ())
            assert bytes_of(shape, itemsize) == prod(shape) * itemsize


def test_padded_strides_account_for_the_padding() -> None:
    """带填充时字节数大于紧密排列 —— 这是新增能力的证明。"""
    # 4 行、每行 200 个元素、行间跨 256（每行末尾 56 个填充）
    assert bytes_of((4, 200), 2, (256, 1)) == 4 * 256 * 2
    assert bytes_of((4, 200), 2, (256, 1)) > bytes_of((4, 200), 2)


def test_stride_rank_must_match_the_shape_rank() -> None:
    """反例：步幅按错的维度解释，字节数会静默算错。"""
    with pytest.raises(ValueError, match="不符"):
        bytes_of((4, 200), 2, (256,))


def test_a_stride_smaller_than_the_inner_span_is_rejected() -> None:
    """反例：第 0 维步幅 100 装不下内层 200 个元素，相邻行会互相覆盖。"""
    detail = TensorShardDetail(0, 0, 0, 4, (4, 200), 0, (100, 1), 0)
    with pytest.raises(ValueError, match="相邻元素会重叠"):
        detail.validate()


def test_strides_of_the_wrong_rank_are_rejected_by_validate() -> None:
    detail = TensorShardDetail(0, 0, 0, 4, (4, 200), 0, (256,), 0)
    with pytest.raises(ValueError, match="elem_strides 秩"):
        detail.validate()


def test_strides_must_be_positive() -> None:
    detail = TensorShardDetail(0, 0, 0, 4, (4, 200), 0, (256, 0), 0)
    with pytest.raises(ValueError, match="必须全为正"):
        detail.validate()


def test_align_bytes_must_be_a_power_of_two() -> None:
    """反例：24 不是 2 的幂，对齐计算会出错。"""
    detail = TensorShardDetail(0, 0, 0, 4, (4, 200), 0, (), 24)
    with pytest.raises(ValueError, match="2 的幂"):
        detail.validate()


def test_mram_offset_must_satisfy_align_bytes() -> None:
    """反例：DMA 会做未对齐访问。"""
    detail = TensorShardDetail(0, 0, 0, 4, (4, 200), 64, (), 128)
    with pytest.raises(ValueError, match="不满足"):
        detail.validate()


def test_empty_strides_and_zero_align_are_the_confirmed_defaults() -> None:
    """空值的语义是「已确认」，不是「未知」。

    与本项目的 PIMMLIR 侧口径相反且是有意的：图编译器**知道**张量怎么摆
    （它是做内存规划的那一方），所以 () 表示确认紧密。
    """
    TensorShardDetail(0, 0, 0, 4, (4, 200), 0, (), 0).validate()


def test_align_up_rejects_non_positive_align() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="align 必须为正"):
            align_up(10, bad)


def test_align_up_rounds_up() -> None:
    assert align_up(0, 16) == 0
    assert align_up(1, 16) == 16
    assert align_up(16, 16) == 16
    assert align_up(17, 16) == 32


def test_align_up_has_exactly_one_implementation() -> None:
    """源码扫描：全仓只能有一处 `def align_up` 定义。

    原先两份实现算法相同、校验不同（`kv_layout` 那份带校验、`l2_alloc` 那份没有），
    合并时取严不取宽。
    """
    root = Path(__file__).parent.parent
    hits = [f"{p.relative_to(root)}:{i}"
            for p in root.rglob("*.py") if "test" not in p.name
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if re.match(r"\s*def align_up\b", line)]
    assert len(hits) == 1, f"align_up 应只有一处实现，实际 {hits}"


def test_both_old_import_sites_now_resolve_to_the_single_source() -> None:
    """两处旧调用方仍然可用，但拿到的是同一份实现。"""
    from memory.kv_layout import align_up as kv_align_up
    from orchestrator.l2_alloc import align_up as l2_align_up

    assert kv_align_up is align_up
    assert l2_align_up is align_up


# ---- Memory Layout 的两个生产方：排布与逐张量对齐 ----
#
# 本轮之前这两个字段全仓零生产方（恒为 `()` 与 0）：「行主序紧密」只是
# `bytes_of` 里的隐含假设，`mram_offset` 用的对齐从不落到分片上。下面几条钉住
# 生产方真的存在，且写出来的值与改动前的隐含值逐字节等价。

def test_row_major_strides_are_the_explicit_form_of_the_implicit_assumption() -> None:
    """显式行主序步幅与 `prod(shape) * itemsize` 完全等价。

    等价是「只把假设显式化、不改行为」的判据：`bytes_of` 的非空分支走
    `local_shape[0] * elem_strides[0] * itemsize`，必须与紧密排列同值。
    """
    for shape in [(3, 4), (1, 4096), (32, 128), (11008, 4096), (1,), (2, 3, 4)]:
        strides = row_major_strides(shape)
        assert len(strides) == len(shape)
        for itemsize in (1, 2, 4):
            assert bytes_of(shape, itemsize, strides) == bytes_of(shape, itemsize)
            assert bytes_of(shape, itemsize, strides) == prod(shape) * itemsize


def test_row_major_strides_reject_a_rankless_shape() -> None:
    """反例：标量没有「怎么摆」可言，秩 0 直接抛而不是返回空步幅。"""
    with pytest.raises(ValueError, match="秩"):
        row_major_strides(())


def test_the_shard_is_born_with_its_layout_and_alignment() -> None:
    """分片出生时带显式步幅，内存规划后带实际对齐 —— 两个载体都要有生产方。

    判据是**真实计划**：partition + propagate 之后每个 DPU 分片都有行主序步幅，
    `plan_dpu` 之后 `align_bytes` 等于这次规划真正用的对齐。
    """
    from contracts.graph_meta import SPEC_META_KEY
    from memory.mem_planner import plan_dpu
    from tests.test_mem_planner import _kv_specs, _two_appendix_a_graphs
    from tests.test_spec_prop import _appendix_a_config, _appendix_a_graph

    gm, _ = _appendix_a_graph()
    partition_graph(gm)
    propagate_specs(gm, _appendix_a_config())

    seen = 0
    for node in gm.graph.nodes:
        spec = node.meta.get(SPEC_META_KEY)
        for detail in (getattr(spec, "shard_map", None) or {}).values():
            seen += 1
            assert detail.elem_strides == row_major_strides(detail.local_shape), \
                f"{node.name} 的分片没有显式步幅：{detail}"
            assert detail.align_bytes == 0, "内存规划之前不该凭空有对齐"
    assert seen, "这张图上没有 DPU 分片，用例失去意义"

    hw = HwBudget(mram_bytes=1 << 30, align=256, sys_reserve_bytes=0)
    (gm1, _), (gm2, _) = _two_appendix_a_graphs()
    nodes1, nodes2 = list(gm1.graph.nodes), list(gm2.graph.nodes)
    plan_dpu(0, nodes1, nodes2, _kv_specs(), hw)

    aligned = 0
    for node in nodes1 + nodes2:
        spec = node.meta.get(SPEC_META_KEY)
        for detail in (getattr(spec, "shard_map", None) or {}).values():
            if detail.dpu_id != 0:
                continue        # 只规划了 dpu0，其余分片此后由它们自己的 DPU 规划
            assert detail.align_bytes == hw.align, \
                f"{node.name} 的分片没有记下实际对齐：{detail}"
            assert detail.mram_offset % hw.align == 0
            aligned += 1
    assert aligned, "规划之后没有任何分片被回填对齐"
