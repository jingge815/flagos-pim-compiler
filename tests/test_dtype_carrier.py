"""数据类型维度的载体与真源。

P0-2 的判据是不访问 `node.meta["val"]` 就能查出 dtype 与量化布局；
六处类型定义收敛为「一处真源 + 一处改引用 + 两处加交叉校验 + 三处性质不同不动」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.dtypes import (
    ELEMENT_DTYPES,
    INDEX_DTYPES,
    _DTYPE_BYTES,
    dtype_bytes,
    validate_dtype,
)
from contracts.gml_quant import DTYPES, QuantLayout, WEIGHT_GROUP_SIZE
from contracts.gml_hw_table import DATA_EXTENSION
from contracts.graph_meta import SPEC_META_KEY
from contracts.pim_tensor_spec import PIMTensorSpec, Placement
from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from tests.test_partition import _export_random_llama
from tests.test_spec_prop import REPLICATE

# 与 tests/test_partition._export_random_llama 的 LlamaConfig 保持一致。
_HIDDEN_SIZE = 64
_NUM_HEADS = 4
_INTERMEDIATE_SIZE = 176
_VOCAB_SIZE = 32000


def _partitioned() -> tuple:
    """导出一个小 llama、切分并传播规格 —— dtype 要在这一步之后可查。"""
    gm = _export_random_llama()
    partition_graph(gm)
    strategy = llama_shard_config(
        2,
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_HEADS,
        intermediate_size=_INTERMEDIATE_SIZE,
        vocab_size=_VOCAB_SIZE,
    )
    edges = propagate_specs(gm, strategy)
    return gm, strategy, edges


def test_dtype_bytes_matches_pytorch_element_size() -> None:
    """dtype_bytes() 对本项目全部类型与 element_size() 等值。

    这是把 5 处 element_size() 改成 dtype_bytes() 的行为等价依据。
    int4 走 int8 路径（PyTorch 无 int4 类型），两者都是 1 字节。
    """
    for name, torch_dt in (("int8", torch.int8), ("int16", torch.int16),
                           ("int32", torch.int32), ("float16", torch.float16),
                           ("float32", torch.float32)):
        assert dtype_bytes(name) == torch.empty(0, dtype=torch_dt).element_size()
    assert dtype_bytes("int4") == 1          # 不打包，一字节一值（实测）


def test_unknown_dtype_raises_and_lists_the_allowed_set() -> None:
    """模块文档 §4.9.5 的反例：bf16 不在集合内，必须抛错。

    今天无载体也无校验，写进去不会报错 —— 这条就是补上那个缺口。
    """
    with pytest.raises(ValueError, match="未知元素类型"):
        validate_dtype("bf16")
    with pytest.raises(ValueError) as excinfo:
        dtype_bytes("float64")
    assert all(name in str(excinfo.value) for name in ELEMENT_DTYPES | INDEX_DTYPES)


def test_index_dtypes_are_separate_from_compute_dtypes() -> None:
    """索引类型单列，不是计算类型。

    实测导出的 llama 图里有 3 个 int64 索引张量（input_ids / arange / unsqueeze），
    其中一个还是 DPU 节点 —— 内存规划要对它算字节数，所以真源必须认识它；
    但它不参与计算，不该混进计算类型集合一起当成「本轮新增的数值格式」。
    """
    assert INDEX_DTYPES == {"int64"}
    assert not (INDEX_DTYPES & ELEMENT_DTYPES)
    assert dtype_bytes("int64") == 8


def test_dtype_is_queryable_without_touching_pytorch() -> None:
    """P0-2 的核心判据：不访问 meta["val"] 即可查出 dtype。

    与 PyTorch 真值一致 —— 这是「派生而非第二真源」的证明。
    """
    gm, _, _ = _partitioned()

    checked = 0
    for node in gm.graph.nodes:
        spec = node.meta.get(SPEC_META_KEY)
        if spec is None or spec.device != "dpu":
            continue
        assert spec.dtype, f"{node.name} 的 spec 没有 dtype"
        validate_dtype(spec.dtype)
        assert spec.dtype == str(node.meta["val"].dtype).removeprefix("torch.")
        checked += 1
    assert checked, "这张图里一个 DPU 节点都没有，用例失去意义"


def test_weight_specs_carry_their_quant_layout() -> None:
    """定点权重必须带 WEIGHT_LAYOUT，浮点权重必须不带。"""
    gm, _, _ = _partitioned()

    weights = [n for n in gm.graph.nodes
               if n.meta.get(SPEC_META_KEY) is not None
               and n.meta[SPEC_META_KEY].residency == "pinned"
               and n.meta[SPEC_META_KEY].device == "dpu"]
    assert weights, "没有找到常驻 DPU 权重"

    for node in weights:
        spec = node.meta[SPEC_META_KEY]
        if spec.dtype in ("int4", "int8"):
            assert spec.quant == QuantLayout("per_group", WEIGHT_GROUP_SIZE, -1), \
                f"{node.name} 是定点权重却没有量化布局"
        else:
            assert spec.quant is None, f"{node.name} 是浮点权重却带了量化布局"


def test_float_tensor_with_quant_layout_raises() -> None:
    """浮点张量带量化布局必须抛错。

    否则 fp16 权重会被当定点处理，scale 文件多出无意义字节。
    """
    spec = _spec(dtype="float16", quant=QuantLayout("per_group", 128, -1))
    with pytest.raises(ValueError, match="不应带量化布局"):
        spec.validate()


def test_quant_without_dtype_raises() -> None:
    """有量化布局却没 dtype：两者必须同时填。"""
    spec = _spec(dtype="", quant=QuantLayout("per_tensor"))
    with pytest.raises(ValueError, match="没有 dtype"):
        spec.validate()


def test_per_group_quant_needs_a_positive_group_size() -> None:
    spec = _spec(dtype="int8", quant=QuantLayout("per_group", 0, -1))
    with pytest.raises(ValueError, match="group_size 必须为正"):
        spec.validate()


def test_empty_dtype_is_legal_but_not_validated_as_a_type() -> None:
    """空 dtype 表示「尚未填充」，是合法状态（host 张量与早期阶段都这样）。"""
    _spec(dtype="", quant=None).validate()


def test_buffer_dtypes_derive_from_the_single_source() -> None:
    """GML 能落盘的缓冲类型由真源派生，不再各写一份字面量。

    判据是**恰好等于**派生式，不是「是子集」—— 子集断言在两边各改一处时仍会通过。
    """
    from gml_bridge.from_fx import _BUFFER_DTYPES

    assert _BUFFER_DTYPES == ELEMENT_DTYPES - {"int4"}, \
        "int4 不单独落盘，它按 int8 存；其余计算类型都能落盘"


def test_encoding_tables_only_key_on_known_dtypes() -> None:
    """两张编码表的键必须是真源子集。

    它们的**值**是两套不同的编号（DT_FP16=1 vs DATA_EXTENSION["int8"]=1），
    刻意不归一；但**键**必须来自同一个类型集合，否则会出现
    「某处认识 bf16 而另一处不认识」的分裂。
    """
    from orchestrator.layer_fields import DT_FP16, DT_FP32, DT_INT8

    assert set(DATA_EXTENSION) <= ELEMENT_DTYPES
    assert DT_INT8 == 0 and DT_FP16 == 1 and DT_FP32 == 3
    assert {"int8", "float16", "float32"} <= ELEMENT_DTYPES


def test_the_other_three_dtype_definitions_keep_their_own_meaning() -> None:
    """三处性质不同的定义保持独立，不被合并进真源。

    - `DTYPES` 回答「weight 缓冲允许哪些类型」，不是「某张量是什么类型」；
    - 位宽与定标常量是量化算法参数；
    - `node.meta["val"].dtype` 是派生来源本身。
    """
    assert set(DTYPES) == {"activation", "weight", "bias", "scale", "output"}
    assert set(DTYPES["weight"]) == {"int4", "int8"}
    from contracts.gml_quant import INT4_BYTES_PER_VALUE

    assert INT4_BYTES_PER_VALUE == 1
    assert _DTYPE_BYTES["int4"] == INT4_BYTES_PER_VALUE


def _spec(*, dtype: str, quant: QuantLayout | None) -> PIMTensorSpec:
    """构造一个只关心类型维度的最小规格。

    单维校验不覆盖 host / DPU 的 shard 约束，这里用 host 形态避免干扰。
    """
    return PIMTensorSpec("host", REPLICATE, "transient", None, {}, None,
                         dtype=dtype, quant=quant)


# ---- 跨维一致性（设计 §4.9.2（4））----
#
# 单维校验管「一个结构自己自洽」，这两条管「维度之间不矛盾」。
# 跨维一守住「spec.dtype 是 val.dtype 的派生物、不是第二个真源」——
# P0-5 要防的旁路正是这种：有 pass 改了 val 却不更新 spec.dtype。

def test_a_dtype_that_drifts_from_the_tensor_is_rejected() -> None:
    """跨维一的反例：spec.dtype 与 meta["val"].dtype 不符必须抛错。"""
    from contracts.unified_ir import (
        STAGE_SPECS,
        validate_node_dimensions,
    )
    import dataclasses

    gm, _, _ = _partitioned()
    node = next(n for n in gm.graph.nodes
                if n.meta.get(SPEC_META_KEY) is not None
                and n.meta[SPEC_META_KEY].device == "dpu"
                and isinstance(n.meta.get("val"), torch.Tensor))

    spec = node.meta[SPEC_META_KEY]
    # 改动前必须先通过：否则这条用例证明不了是漂移被抓到。
    validate_node_dimensions(node, stage=STAGE_SPECS)

    drifted = "int32" if spec.dtype != "int32" else "int8"
    node.meta[SPEC_META_KEY] = dataclasses.replace(spec, dtype=drifted, quant=None)
    with pytest.raises(ValueError, match="不符"):
        validate_node_dimensions(node, stage=STAGE_SPECS)


def test_a_shard_dim_beyond_the_local_rank_is_rejected() -> None:
    """跨维二的反例：切分维超出 local_shape 的秩必须抛错。

    构造期只校验「维号 < 全局形状的秩」，切完之后的 local_shape 才是算子
    编译器实际看到的那块 —— 秩对不上就是标到了不存在的轴上。
    """
    from contracts.pim_tensor_spec import TensorShardDetail
    from contracts.unified_ir import STAGE_SPECS, validate_node_dimensions

    class _Node:                            # 只要 name 与 meta 两样
        name = "fake"

        def __init__(self, meta):
            self.meta = meta

    detail = TensorShardDetail(dpu_id=0, shard_dim=2, start_idx=0, end_idx=4,
                               local_shape=(4, 8))   # 秩 2，切分维 2 越界
    spec = PIMTensorSpec(device="dpu", placement=Placement("Shard", dim=2),
                         residency="transient", pinned_dpu_id=None,
                         shard_map={0: detail}, reduce_type=None,
                         dtype="float16")
    node = _Node({SPEC_META_KEY: spec})
    with pytest.raises(ValueError, match="超出 local_shape 秩"):
        validate_node_dimensions(node, stage=STAGE_SPECS)


def test_the_exit_check_runs_inside_propagate_specs(monkeypatch) -> None:
    """校验必须真的在 pass 出口跑，不是一个没人调的公开 API。

    判据是「跑一遍 `propagate_specs`，那个校验函数真的被调到了」，不是源码里
    出现过它的名字：改成注释、挪进一条走不到的分支、或者调的是同名的别处函数，
    源码扫法全都发现不了。
    """
    import torch

    from graph import spec_prop
    from graph.partition import partition_graph
    from graph.strategy import llama_strategy

    calls: list[str] = []
    real = spec_prop.validate_graph_dimensions
    monkeypatch.setattr(
        spec_prop, "validate_graph_dimensions",
        lambda gm, *, stage: (calls.append(stage), real(gm, stage=stage))[1])

    class _Tiny(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm = torch.export.export(_Tiny(), (torch.zeros(2, 4),)).module()
    partition_graph(gm)
    spec_prop.propagate_specs(gm, llama_strategy(
        1, num_stages=1, num_heads=8, num_kv_heads=8,
        intermediate_size=176, vocab_size=320, num_layers=1))

    assert calls, "propagate_specs 出口没有调跨维校验，这条不变式没有执行点"
