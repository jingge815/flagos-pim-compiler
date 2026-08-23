"""Tests for problem-2 device mapping and sharding propagation.

判据（CLAUDE.md 测试约定）：编译期模块的判据是方案手推结果——附录 A 的
placement 推演（元数据对拍），另加 NumpyBackend 上的数值对拍证明
shard_map 的 start/end/local_shape 在真实数据搬运下正确。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.fx import Graph, GraphModule, Node

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.hal_numpy import NumpyBackend, NumpyBackendConfig
from contracts.graph_meta import (
    DEVICE_DPU,
    DEVICE_HOST,
    DEVICE_META_KEY,
    REDISTRIBUTE_META_KEY,
    SPEC_META_KEY,
)
from contracts.pim_tensor_spec import PIMTensorSpec, Placement, TensorShardDetail
from graph.partition import partition_graph
from graph.spec_prop import (
    _Req,
    _edge_type,
    ShardConfig,
    format_spec_report,
    llama_shard_config,
    propagate_specs,
)
from tests.test_partition import _export_random_llama

REPLICATE = Placement("Replicate")
PARTIAL_SUM = Placement("Partial", reduce_type="sum")


def test_shard_dimension_change_between_dpus_is_all_to_all() -> None:
    """问题 3 已支持的 Shard(i) → Shard(j) 必须能由问题 2 产生。"""
    graph = Graph()
    producer = graph.placeholder("producer")
    actual = PIMTensorSpec(
        DEVICE_DPU,
        Placement("Shard", 0),
        "transient",
        None,
        {
            0: TensorShardDetail(0, 0, 0, 2, (2, 4)),
            1: TensorShardDetail(1, 0, 2, 4, (2, 4)),
        },
        None,
    )

    assert _edge_type(producer, actual, _Req(DEVICE_DPU, Placement("Shard", 1))) == "all_to_all"


# ---------------------------------------------------------------------------
# 附录 A 最小示例：hidden=4, ffn=6, 2 DPU（torch.export 的图自带 meta["val"]，
# 手搭的图需要手填）
# ---------------------------------------------------------------------------


def _appendix_a_graph() -> tuple[GraphModule, dict[str, Node]]:
    root = torch.nn.Module()
    root.register_buffer("w1", torch.empty(6, 4))  # HF 布局 [out, in]：列切 Shard(0)
    root.register_buffer("w2", torch.empty(4, 6))  # 行切 Shard(1)
    graph = Graph()
    x = graph.placeholder("x")
    w1 = graph.get_attr("w1")
    w2 = graph.get_attr("w2")
    y1 = graph.call_function(torch.ops.aten.linear.default, (x, w1))
    y2 = graph.call_function(torch.ops.aten.linear.default, (y1, w2))
    norm = graph.call_function(torch.ops.aten.layer_norm.default, (y2, [4], None, None, 1e-5))
    graph.output(norm)
    gm = GraphModule(root, graph)
    x.meta["val"] = torch.empty(1, 4)
    w1.meta["val"] = root.w1
    w2.meta["val"] = root.w2
    y1.meta["val"] = torch.empty(1, 6)
    y2.meta["val"] = torch.empty(1, 4)
    norm.meta["val"] = torch.empty(1, 4)
    return gm, {"x": x, "w1": w1, "w2": w2, "y1": y1, "y2": y2, "norm": norm}


def _appendix_a_config() -> ShardConfig:
    return ShardConfig(num_dpus=2, weight_rules=(("w1", "col"), ("w2", "row")))


def test_physical_dpu_ids_key_shard_maps_and_report_endpoint_ranges() -> None:
    """物理 DPU ID 直接成为所有 DPU shard_map 的键与 detail.dpu_id。"""
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)

    edges = propagate_specs(
        gm,
        ShardConfig(
            num_dpus=2,
            dpu_ids=(2, 5),
            weight_rules=(("w1", "col"), ("w2", "row")),
        ),
    )

    for name in ("w1", "w2", "y1", "y2"):
        shard_map = nodes[name].meta[SPEC_META_KEY].shard_map
        assert tuple(shard_map) == (2, 5)
        assert {detail.dpu_id for detail in shard_map.values()} == {2, 5}
    for edge in edges:
        for location in (edge.src_loc, edge.dst_loc):
            if location["device"] == DEVICE_DPU:
                assert location["dpus"] == [2, 5]
    report = format_spec_report(gm, edges)
    assert "DPU2:[0,3) shape=(3, 4)" in report
    assert "DPU5:[3,6) shape=(3, 4)" in report


def _keyword_add_graph() -> tuple[GraphModule, dict[str, Node]]:
    """A DPU add with its right operand supplied through ``kwargs[\"other\"]``."""
    graph = Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    add = graph.call_function(torch.ops.aten.add.Tensor, (left,), {"other": right})
    graph.output(add)
    gm = GraphModule(torch.nn.Module(), graph)
    for node in (left, right, add):
        node.meta["val"] = torch.empty(2, 4)
    return gm, {"left": left, "right": right, "add": add}


def test_keyword_operand_edges_materialize_endpoint_specs() -> None:
    """Both positional and keyword add operands receive concrete DPU endpoint specs."""
    gm, nodes = _keyword_add_graph()
    partition_graph(gm)

    edges = propagate_specs(gm, ShardConfig(num_dpus=2, dpu_ids=(2, 5), weight_rules=()))

    assert {edge.src for edge in edges if edge.dst == nodes["add"].name} == {"left", "right"}
    for edge in nodes["add"].meta[REDISTRIBUTE_META_KEY]:
        assert edge.src_spec == nodes[edge.src].meta[SPEC_META_KEY]
        assert edge.dst_spec.placement == REPLICATE
        assert edge.dst_spec.device == DEVICE_DPU
        assert tuple(edge.dst_spec.shard_map) == (2, 5)


def test_replicate_dpu_to_host_is_degenerate_all_gather_from_smallest_dpu() -> None:
    """A replicated DPU result reaches host as one canonical full-copy transfer."""
    graph = Graph()
    x = graph.placeholder("x")
    dpu_tanh = graph.call_function(torch.ops.aten.tanh.default, (x,))
    host_neg = graph.call_function(torch.ops.aten.neg.default, (dpu_tanh,))
    graph.output(host_neg)
    gm = GraphModule(torch.nn.Module(), graph)
    for node in (x, dpu_tanh, host_neg):
        node.meta["val"] = torch.empty(2, 4)
    partition_graph(gm)

    edges = propagate_specs(gm, ShardConfig(num_dpus=2, dpu_ids=(5, 2), weight_rules=()))

    (edge,) = [edge for edge in edges if edge.src == dpu_tanh.name and edge.dst == host_neg.name]
    assert edge.type == "all_gather"
    assert edge.src_spec.placement == REPLICATE
    assert edge.src_spec.device == DEVICE_DPU
    assert edge.dst_spec.placement == REPLICATE
    assert edge.dst_spec.device == DEVICE_HOST
    assert edge.src_loc == {"device": DEVICE_DPU, "dpus": [2, 5]}
    assert edge.dst_loc == {"device": DEVICE_HOST}
    assert min(edge.src_spec.shard_map) == 2  # Problem 3's canonical one-copy DMA source.


def test_propagation_clears_stale_metadata_and_reuses_stable_edge_ids() -> None:
    """Rerunning propagation replaces stale metadata instead of accumulating old edges/specs."""
    gm, nodes = _keyword_add_graph()
    partition_graph(gm)
    output = next(node for node in gm.graph.nodes if node.op == "output")
    nodes["left"].meta[SPEC_META_KEY] = object()
    nodes["left"].meta[REDISTRIBUTE_META_KEY] = ["stale"]
    output.meta[SPEC_META_KEY] = object()
    output.meta[REDISTRIBUTE_META_KEY] = ["stale"]
    config = ShardConfig(num_dpus=2, dpu_ids=(2, 5), weight_rules=())

    first = propagate_specs(gm, config)
    second = propagate_specs(gm, config)

    assert first == second
    assert [edge.edge_id for edge in second] == list(range(len(second)))
    assert nodes["left"].meta[REDISTRIBUTE_META_KEY] == []
    assert SPEC_META_KEY not in output.meta
    assert all("stale" not in node.meta[REDISTRIBUTE_META_KEY] for node in gm.graph.nodes)


@pytest.mark.parametrize("dpu_ids", [(), (2,), (2, 2), (2, -1)])
def test_invalid_physical_dpu_ids_are_rejected(dpu_ids: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="dpu_ids"):
        ShardConfig(num_dpus=2, dpu_ids=dpu_ids, weight_rules=())


@pytest.mark.parametrize(
    "placement",
    [Placement("Shard"), Placement("Shard", -1), Placement("Partial"), Placement("Replicate", 0)],
)
def test_invalid_dtensor_placements_are_rejected(placement: Placement) -> None:
    with pytest.raises(ValueError):
        placement.validate()


def test_invalid_dtensor_placement_kind_is_rejected() -> None:
    with pytest.raises(ValueError):
        Placement("Unknown").validate()  # type: ignore[arg-type]


def test_appendix_a_placements() -> None:
    """附录 A 元数据对拍：Y1=Shard(1) 零通信、Y2=Partial(sum)、LN 前插 all_reduce。"""
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)
    edges = propagate_specs(gm, _appendix_a_config())

    w1_spec = nodes["w1"].meta[SPEC_META_KEY]
    assert w1_spec.placement == Placement("Shard", 0)
    assert w1_spec.residency == "pinned"
    assert w1_spec.shard_map[0] == TensorShardDetail(0, 0, 0, 3, (3, 4))
    assert w1_spec.shard_map[1] == TensorShardDetail(1, 0, 3, 6, (3, 4))

    # A.1：列切产出 Shard(1)，per-DPU 持 [0,3) / [3,6) 列分片
    y1_spec = nodes["y1"].meta[SPEC_META_KEY]
    assert y1_spec.device == DEVICE_DPU
    assert y1_spec.placement == Placement("Shard", 1)
    assert y1_spec.shard_map[0].local_shape == (1, 3)
    assert (y1_spec.shard_map[0].start_idx, y1_spec.shard_map[0].end_idx) == (0, 3)
    assert (y1_spec.shard_map[1].start_idx, y1_spec.shard_map[1].end_idx) == (3, 6)

    # X 从 host 进 DPU：同布局纯跨位置，scatter（broadcast 退化）；此外计算阶段零通信
    (x_edge,) = nodes["y1"].meta[REDISTRIBUTE_META_KEY]
    assert x_edge.type == "scatter"
    assert x_edge.src_loc == {"device": DEVICE_HOST}
    assert x_edge.dst_loc == {"device": DEVICE_DPU, "dpus": [0, 1]}

    # A.2：行切权重与 Shard(1) 输入天然对齐（Megatron 列→行配对），产出 Partial(sum)
    y2_spec = nodes["y2"].meta[SPEC_META_KEY]
    assert y2_spec.placement == PARTIAL_SUM
    assert y2_spec.reduce_type == "sum"
    assert y2_spec.shard_map[0].local_shape == (1, 4)  # Partial 每台持完整形状
    assert nodes["y2"].meta[REDISTRIBUTE_META_KEY] == []

    # A.4：host LayerNorm 要求 Replicate@host → Partial→Replicate 的 all_reduce 边
    (edge,) = nodes["norm"].meta[REDISTRIBUTE_META_KEY]
    assert edge.from_placement == PARTIAL_SUM
    assert edge.to_placement == REPLICATE
    assert edge.type == "all_reduce"
    assert edge.reduce_type == "sum"
    assert edge.src_loc == {"device": DEVICE_DPU, "dpus": [0, 1]}
    assert edge.dst_loc == {"device": DEVICE_HOST}
    assert edges == [x_edge, edge]


def test_appendix_a_numeric_on_numpy_backend() -> None:
    """附录 A 数值对拍：按 shard_map 把分片写进独立 DPU buffer，逐 DPU 算、
    host 拼接/累加后与单卡 torch 参考逐元素对齐。"""
    torch.manual_seed(0)
    x_ref = torch.randn(1, 4)
    w1_ref = torch.randn(6, 4)
    w2_ref = torch.randn(4, 6)

    gm, nodes = _appendix_a_graph()
    with torch.no_grad():
        nodes["w1"].meta["val"].copy_(w1_ref)
        nodes["w2"].meta["val"].copy_(w2_ref)
    partition_graph(gm)
    propagate_specs(gm, _appendix_a_config())

    backend = NumpyBackend(NumpyBackendConfig(num_dpus=2, mram_bytes_per_dpu=1 << 16))
    offsets = {"x": 0, "w1": 256, "y1": 512, "w2": 1024, "y2": 2048}

    def shard_of(t: torch.Tensor, det: TensorShardDetail) -> np.ndarray:
        sl = [slice(None)] * t.ndim
        sl[det.shard_dim] = slice(det.start_idx, det.end_idx)
        return t[tuple(sl)].numpy()

    # 1. 按 shard_map 装载：X 广播（Replicate 全量），W1/W2 各持分片
    specs = {k: nodes[k].meta[SPEC_META_KEY] for k in ("x", "w1", "w2", "y1", "y2")}
    for dpu_id in range(2):
        backend.copy_to_dpu(dpu_id, offsets["x"], x_ref.numpy())
        backend.copy_to_dpu(dpu_id, offsets["w1"], shard_of(w1_ref, specs["w1"].shard_map[dpu_id]))
        backend.copy_to_dpu(dpu_id, offsets["w2"], shard_of(w2_ref, specs["w2"].shard_map[dpu_id]))

    # 2. A.1 列切：每 DPU 读本地 X、W1 分片，算 Y1 分片并写回本地 MRAM
    for dpu_id, det in specs["y1"].shard_map.items():
        x_local = backend.copy_from_dpu(dpu_id, offsets["x"], (1, 4), np.float32)
        w1_local = backend.copy_from_dpu(dpu_id, offsets["w1"], (3, 4), np.float32)
        backend.copy_to_dpu(dpu_id, offsets["y1"], x_local @ w1_local.T)

    # 3. host 按 global_range（start_idx）排序拼接 = Shard→Replicate 的 all_gather
    y1_parts = [
        backend.copy_from_dpu(dpu_id, offsets["y1"], det.local_shape, np.float32)
        for dpu_id, det in sorted(specs["y1"].shard_map.items(), key=lambda kv: kv[1].start_idx)
    ]
    y1 = np.concatenate(y1_parts, axis=1)
    np.testing.assert_allclose(y1, (x_ref @ w1_ref.T).numpy(), rtol=1e-6)

    # 4. A.2 行切：每 DPU 用本地 Y1 分片 × W2 列分片 = 部分和；host 累加 = all_reduce
    y2 = np.zeros((1, 4), dtype=np.float32)
    for dpu_id, det in specs["y2"].shard_map.items():
        y1_local = backend.copy_from_dpu(dpu_id, offsets["y1"], (1, 3), np.float32)
        w2_local = backend.copy_from_dpu(dpu_id, offsets["w2"], (4, 3), np.float32)
        backend.copy_to_dpu(dpu_id, offsets["y2"], y1_local @ w2_local.T)
        y2 += backend.copy_from_dpu(dpu_id, offsets["y2"], det.local_shape, np.float32)
    np.testing.assert_allclose(y2, ((x_ref @ w1_ref.T) @ w2_ref.T).numpy(), rtol=1e-6)


# ---------------------------------------------------------------------------
# tiny Llama 全图：问题 1 + 问题 2 衔接的结构断言
# ---------------------------------------------------------------------------


def _propagate_tiny_llama():
    gm = _export_random_llama()
    partition_graph(gm)
    config = llama_shard_config(4, num_heads=4, num_kv_heads=4, intermediate_size=176, vocab_size=32000)
    edges = propagate_specs(gm, config)
    return gm, {n.name: n for n in gm.graph.nodes}, edges


def test_tiny_llama_weight_initial_sharding() -> None:
    gm, by_name, _ = _propagate_tiny_llama()
    for name, node in by_name.items():
        if node.op != "get_attr" or not isinstance(node.meta.get("val"), torch.Tensor):
            continue
        spec = node.meta[SPEC_META_KEY]
        target = node.target
        if "q_proj" in target or "gate_proj" in target or "lm_head" in target:
            assert spec.placement == Placement("Shard", 0), target
            assert spec.residency == "pinned"
        if "o_proj" in target or "down_proj" in target:
            assert spec.placement == Placement("Shard", 1), target
        if "layernorm" in target:
            assert spec.placement == REPLICATE and spec.device == DEVICE_DPU, target
        if "embed_tokens" in target:
            assert spec.device == DEVICE_HOST, target
    # hidden=64 / 4 DPU = 16 = head_dim，即每台恰好 1 个 head（切点对齐 head 边界）
    q_weight = by_name["model_model_layers_0_self_attn_q_proj_weight"].meta[SPEC_META_KEY]
    assert q_weight.shard_map[0].local_shape == (16, 64)
    assert q_weight.shard_map[3].start_idx == 48


def test_tiny_llama_propagation_structure() -> None:
    gm, by_name, edges = _propagate_tiny_llama()

    # 每个张量节点都有 spec；非张量 get_attr（子图）除外
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "get_attr" and not isinstance(node.meta.get("val"), torch.Tensor):
            continue
        assert SPEC_META_KEY in node.meta, node.name

    # o_proj / down_proj 行切产出 Partial；其后的残差 add 上各有一条 all_reduce 边
    for proj in ("o_proj", "down_proj"):
        weight = next(n for n in gm.graph.nodes if n.op == "get_attr" and f"{proj}.weight" in n.target)
        (linear,) = weight.users
        assert linear.meta[SPEC_META_KEY].placement == PARTIAL_SUM, proj
        (add,) = [u for u in linear.users if u.target == torch.ops.aten.add.Tensor]
        (edge,) = [e for e in add.meta[REDISTRIBUTE_META_KEY] if e.src == linear.name]
        assert edge.type == "all_reduce"
        assert edge.dst_loc == {"device": DEVICE_DPU, "dpus": [0, 1, 2, 3]}

    # 行切 down_proj 的输入（silu 之后的 mul）已是 Shard(2)，零通信进入（无 scatter 边）
    down_weight = next(n for n in gm.graph.nodes if n.op == "get_attr" and "down_proj.weight" in n.target)
    (down_linear,) = down_weight.users
    act = down_linear.args[0]
    assert act.meta[SPEC_META_KEY].placement == Placement("Shard", 2)
    assert all(e.src != act.name for e in down_linear.meta[REDISTRIBUTE_META_KEY])

    # 每层两处 all_reduce（o_proj、down_proj 各一），与问题 6 的通信点表一致
    assert sum(e.type == "all_reduce" for e in edges) == 2
    assert {e.type for e in edges} <= {"all_reduce", "all_gather", "scatter"}
    # 权重是 pinned，永不出现在 redistribute 边的任何一端
    for edge in edges:
        assert by_name[edge.src].op != "get_attr"
        for loc in (edge.src_loc, edge.dst_loc):
            assert loc["device"] in ("host", "dpu")
            assert loc["device"] == "host" or sorted(loc["dpus"]) == [0, 1, 2, 3]

    # 图出口：logits 列切 Shard(2) 经 all_gather 回 host
    output = next(n for n in gm.graph.nodes if n.op == "output")
    (final_edge,) = output.meta[REDISTRIBUTE_META_KEY]
    assert final_edge.type == "all_gather"
    assert final_edge.from_placement == Placement("Shard", 2)

    # 报告可打印且重跑幂等
    assert "all_reduce" in format_spec_report(gm, edges, max_nodes=10)
    assert len(propagate_specs(gm, llama_shard_config(
        4, num_heads=4, num_kv_heads=4, intermediate_size=176, vocab_size=32000))) == len(edges)


def test_llama_shard_config_rejects_contract_violations() -> None:
    kwargs = dict(num_heads=4, num_kv_heads=4, intermediate_size=176, vocab_size=32000)
    with pytest.raises(ValueError, match="num_dpus must be positive"):
        llama_shard_config(0, **kwargs)
    with pytest.raises(ValueError, match="2 的整数次幂"):
        llama_shard_config(3, **kwargs)
    with pytest.raises(ValueError, match="num_heads"):
        llama_shard_config(8, **kwargs)  # 4 个 head 切不到 8 台
    with pytest.raises(ValueError, match="intermediate_size"):
        llama_shard_config(2, **{**kwargs, "intermediate_size": 175})


def test_propagate_requires_partition_first() -> None:
    gm, _ = _appendix_a_graph()
    with pytest.raises(ValueError, match="partition_graph"):
        propagate_specs(gm, _appendix_a_config())


def test_linear_weight_must_be_configured() -> None:
    gm, _ = _appendix_a_graph()
    partition_graph(gm)
    with pytest.raises(ValueError, match="w2"):
        propagate_specs(gm, ShardConfig(num_dpus=2, weight_rules=(("w1", "col"),)))
