"""GML 的相位字段真的由算子编译器决定。

这组测试守的是一个容易退化成假的性质：**pimmlir 必须真的影响 GML**，
同时又要与静态表路径产出相同的结果。两条缺一不可：

- 只有「相同」→ 依赖可能是假的（算了但不用，纯冗余）
- 只有「影响」→ 破坏了「接入前后 GML 不变」的验收

所以每条性质各有测试：`test_*_identical` 守相同，`test_*_changes_gml` 守依赖。

判据用**字段套数**而不是文本 sha：sha 只能告诉你「变了」，字段套数能告诉你
「变在相位上」。
"""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest
import torch
from torch.fx import Graph, GraphModule

sys.path.insert(0, str(Path(__file__).parent.parent))

from gml_bridge.from_fx import _phase_count_for
from opcompiler_bridge.phase_plan import Phase, PhasePlan
from opcompiler_bridge.phase_source import PhaseSource


def _plan(func: str, count: int) -> PhasePlan:
    return PhasePlan(func=func, phases=[
        Phase(index=i, op="", unit="", bytes=0) for i in range(count)
    ])


# --- 单元：相位数从哪来 -----------------------------------------------------


def test_falls_back_to_static_table_without_opcompiler() -> None:
    """没接算子编译器时退回静态表，行为与改造前一致。"""
    assert _phase_count_for(None, "DynamicScaling", "dq0") == 4
    assert _phase_count_for(None, "Softmax", "sm0") == 5


def test_opcompiler_overrides_static_table() -> None:
    """接上算子编译器后，相位数由它说了算。

    这里故意给 2 相（静态表是 4），返回 2 才说明真源换了。
    """
    source = PhaseSource(by_node={"dq0": {"dq": _plan("dq0__dq", 2)}})
    assert _phase_count_for(source, "DynamicScaling", "dq0") == 2


def test_missing_plan_falls_back_rather_than_crashing() -> None:
    """算子编译器没覆盖到的节点退回静态表，不是报错也不是发 0 相。"""
    source = PhaseSource(by_node={"other": {"dq": _plan("other__dq", 2)}})
    assert _phase_count_for(source, "DynamicScaling", "dq0") == 4


def test_rope_dq_phase_fields_come_from_dq_plan() -> None:
    """`Llama2ActivationDQ` 的 `*_phase_N` 只对应它的 DQ 那 4 相。

    它是「RoPE 3 连 + DQ 4 相」，相位字段查 `dq` 不查 `rope`。
    """
    source = PhaseSource(by_node={"rope_k": {
        "rope": _plan("rope_k__rope", 3),
        "dq": _plan("rope_k__dq", 4),
    }})
    assert _phase_count_for(source, "Llama2ActivationDQ", "rope_k") == 4


def test_single_phase_ops_are_unaffected() -> None:
    """单相算子不发 `*_phase_N`，两条路径都返回 0。"""
    source = PhaseSource(by_node={"g0": {"dq": _plan("g0__dq", 4)}})
    assert _phase_count_for(source, "Gemm", "g0") == 0
    assert _phase_count_for(None, "Gemm", "g0") == 0


# --- 端到端：真实图上两条路径的关系 ----------------------------------------

from genesim_bridge.paths import llama2_7b_model_dir  # noqa: E402
from opcompiler_bridge.phase_source import opcompiler_available  # noqa: E402

MODEL_DIR = llama2_7b_model_dir(required=False)

requires_live = pytest.mark.skipif(
    MODEL_DIR is None or not MODEL_DIR.is_dir() or not opcompiler_available(),
    reason="需要 llama2_7b_model_dir 和带 PIM pass 的 triton-opt",
)


def _fresh_graph():
    """一张全新的融合图。

    `serialize_gml` 现在是纯函数（`convert()` 不再写 `node.meta`），所以同一张
    图序列化多次结果相同。但这里仍每次给新图，因为 `fuse_for_gml` 会改图。
    """
    from gml_bridge.export import fuse_for_gml
    from runtime.compile import export_annotated_graph
    from scripts.export_gml import _load_model

    model = _load_model(1)
    position_ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, 16, position_ids, dtype=torch.float32)
    return gm, fuse_for_gml(gm)


def _phase_field_counts(text: str) -> dict[int, int]:
    return {n: len(re.findall(rf"_phase_{n}\b", text)) for n in range(5)}


@requires_live
def test_both_paths_produce_identical_gml() -> None:
    """性质一：接入算子编译器后 GML 与静态表路径**逐字节相同**。

    这是验收要求。相同的原因是算子编译器算出的相位数与静态表一致
    （`cross_check` 在保证），不是因为没用上。
    """
    from gml_bridge.export import serialize_gml
    from opcompiler_bridge.phase_source import phase_source_from_graph

    gm_a, fusion_a = _fresh_graph()
    static = serialize_gml(gm_a, fusion_a, phase_source=None)

    gm_b, fusion_b = _fresh_graph()
    source = phase_source_from_graph(gm_b)
    from_pim = serialize_gml(gm_b, fusion_b, phase_source=source)

    assert static.text == from_pim.text
    # 197 算子 + cos/sin 表 2 + mask 1 + KV cache 2 + 位置下标 1 = 203。
    # 参考 decode block 是 200（编号体系不同，边界节点集合对齐）。
    assert len(static.nodes) == len(from_pim.nodes)
    assert len(static.nodes) >= 200
    n_buf = sum(1 for n in static.nodes if n.fields.get("is_buffer"))
    # 参考 decode block 10 个；未裁尾时入口/出口/表/mask/KV 至少 6。
    assert n_buf >= 6


@requires_live
def test_changing_opcompiler_output_changes_gml() -> None:
    """性质二：**依赖是真的** —— 改 pimmlir 的相位数，GML 跟着变。

    这条是「相同」不退化成「无关」的保险。把 DQ 从 4 相砍到 2 相，
    `_phase_2` / `_phase_3` 的字段应该大幅减少。
    """
    from gml_bridge.export import serialize_gml
    from opcompiler_bridge.phase_source import phase_source_from_graph

    gm_a, fusion_a = _fresh_graph()
    source = phase_source_from_graph(gm_a)
    normal = serialize_gml(gm_a, fusion_a, phase_source=source)

    gm_b, fusion_b = _fresh_graph()
    tampered = phase_source_from_graph(gm_b)
    for kinds in tampered.by_node.values():
        if "dq" in kinds:
            kinds["dq"] = copy.deepcopy(kinds["dq"])
            kinds["dq"].phases = kinds["dq"].phases[:2]
    changed = serialize_gml(gm_b, fusion_b, phase_source=tampered)

    assert changed.text != normal.text

    before = _phase_field_counts(normal.text)
    after = _phase_field_counts(changed.text)
    # 前两相不受影响（砍的是第 3、4 相）。
    assert after[0] == before[0]
    assert after[1] == before[1]
    # 后两相要明显减少，不是只差几个字段。
    assert after[2] < before[2] / 2
    assert after[3] < before[3] / 2
    assert len(changed.text) < len(normal.text)


@requires_live
def test_serialize_is_a_pure_function() -> None:
    """`serialize_gml` 是纯函数：同一张图连调三次结果相同。

    这条守的是 `convert()` 不再写 `node.meta`。早先它给 K 路 RoPE 节点补
    `DQ_META_KEY`，于是第二次序列化时 `_dq_specs` 多认出那个节点，下游
    `KV_Cache_DMA` 的 `input_sf` 跟着变错（指向上游的相位缓冲而不是自己的
    `input_sf_N.bin`，与参考产物不符）。
    """
    from gml_bridge.export import serialize_gml

    gm, fusion = _fresh_graph()
    runs = [serialize_gml(gm, fusion) for _ in range(3)]
    assert runs[0].text == runs[1].text == runs[2].text
    assert len({len(r.dq_specs) for r in runs}) == 1


@requires_live
def test_kv_cache_dma_input_sf_uses_own_node_id() -> None:
    """`KV_Cache_DMA` 声明三个槽：0=cache、1=索引、2=新值。

    参考产物：`input_buffer_0 "input_buffer_0_28.bin"`，txt 的
    `Original cache file` 取它。单数 `input_buffer` / `input_sf` 是错的。
    """
    from gml_bridge.export import serialize_gml

    gm, fusion = _fresh_graph()
    artifact = serialize_gml(gm, fusion)
    dma = [n for n in artifact.nodes
           if n.fields.get("op_type") == "KV_Cache_DMA"]
    assert dma, "图里应有 KV_Cache_DMA 节点"
    for node in dma:
        assert node.fields["input_buffer_0"] == (
            f"input_buffer_0_{node.node_id}.bin")
        assert node.fields.get("input_buffer_1_dtype") == "int16"
        assert node.fields.get("use_input_buffer_1") == "L2A_ignore"
        assert "input_buffer_dtype" not in node.fields
        assert node.fields.get("input_count") == 3
        residual = node.fields.get("residual_input_buffer") or []
        assert len(residual) == 3
    n_buf = sum(1 for n in artifact.nodes if n.fields.get("is_buffer"))
    labels = {n.fields.get("label") for n in artifact.nodes
              if n.fields.get("is_buffer")}
    assert "in_kv_position" in labels
    assert "in_key_cache" in labels or "in_value_cache" in labels
    assert n_buf >= 6
