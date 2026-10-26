"""验证设备映射和张量分片传播。"""

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
    _diff_edge,
    _dpu_req,
    _dpu_spec,
    _edge_type,
    _layer_of_node,
    format_spec_report,
    llama_shard_config,
    propagate_specs,
)
from graph.strategy import ShardStrategy, llama_strategy
from tests.test_partition import _export_random_llama

REPLICATE = Placement("Replicate")
PARTIAL_SUM = Placement("Partial", reduce_type="sum")


def test_shard_dimension_change_between_dpus_is_all_to_all() -> None:
    """验证不同切分维度之间生成 all_to_all 通信边。"""
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


def test_same_placement_on_different_dpus_still_needs_a_redistribute_edge() -> None:
    """验证相同布局在不同 DPU 集合间仍生成重分布边。"""
    graph = Graph()
    producer = graph.placeholder("producer")
    producer.meta["val"] = torch.zeros(2, 4)
    consumer = graph.placeholder("consumer")

    on_dpu0 = _dpu_spec(REPLICATE, (2, 4), (0,))
    edge = _diff_edge(0, producer, consumer, on_dpu0, _dpu_req(REPLICATE), (1,))

    assert edge is not None, "同 placement、不同 DPU 集合必须产生一条搬运边"
    assert edge.type == "all_gather"
    assert edge.src_loc == {"device": DEVICE_DPU, "dpus": [0]}
    assert edge.dst_loc == {"device": DEVICE_DPU, "dpus": [1]}

    # 相同 DPU 集合无需数据搬运。
    assert _diff_edge(0, producer, consumer, on_dpu0, _dpu_req(REPLICATE), (0,)) is None


@pytest.mark.parametrize(
    "path, expected",
    [
        # 两种模块栈路径格式。
        ("L['s'].model.model.layers[slice(None, 32, None)]._modules['0'].self_attn", 0),
        ("L['s'].model.model.layers[slice(None, 32, None)]._modules['31'].mlp", 31),
        ("model.model.layers.slice(None, 4, None).0.input_layernorm", 0),
        ("model.model.layers.slice(None, 4, None).13.mlp", 13),
        # 层外节点没有层号。
        ("model.model.norm", None),
        ("L['s'].model.model.embed_tokens", None),
        ("model.lm_head", None),
    ],
)
def test_layer_number_parsed_from_both_module_stack_formats(path: str, expected) -> None:
    """验证从两种模块栈格式中提取层号。"""
    graph = Graph()
    node = graph.placeholder("n")
    node.meta["nn_module_stack"] = {"k": (path, type(None))}

    assert _layer_of_node(node) == expected


def test_layer_number_of_weight_comes_from_its_name() -> None:
    """验证从权重名称中提取层号。"""
    gm, _ = _appendix_a_graph()
    (weight,) = [
        node for node in gm.graph.nodes if node.op == "get_attr" and node.target == "w1"
    ]
    weight.target = "model.model.layers.7.self_attn.q_proj.weight"

    assert _layer_of_node(weight) == 7


# 最小的两 DPU 分片示例。


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
    return ShardStrategy(name="a", num_dpus=2, weight_rules=(("w1", "col"), ("w2", "row")))


def test_physical_dpu_ids_key_shard_maps_and_report_endpoint_ranges() -> None:
    """物理 DPU ID 直接成为所有 DPU shard_map 的键与 detail.dpu_id。"""
    gm, nodes = _appendix_a_graph()
    partition_graph(gm)

    edges = propagate_specs(
        gm,
        ShardStrategy(
            name="a",
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
    """构造右操作数位于 ``kwargs[\"other\"]`` 的 DPU 加法图。"""
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
    """验证位置参数和关键字参数都生成具体端点规格。"""
    gm, nodes = _keyword_add_graph()
    partition_graph(gm)

    edges = propagate_specs(gm, ShardStrategy(name="a", num_dpus=2, dpu_ids=(2, 5), weight_rules=()))

    assert {edge.src for edge in edges if edge.dst == nodes["add"].name} == {"left", "right"}
    for edge in nodes["add"].meta[REDISTRIBUTE_META_KEY]:
        assert edge.src_spec == nodes[edge.src].meta[SPEC_META_KEY]
        assert edge.dst_spec.placement == REPLICATE
        assert edge.dst_spec.device == DEVICE_DPU
        assert tuple(edge.dst_spec.shard_map) == (2, 5)


def test_replicate_dpu_to_host_is_degenerate_all_gather_from_smallest_dpu() -> None:
    """验证复制布局从编号最小的 DPU 向主机生成单次全量传输。

    主机侧消费者要用**真正**留主机的算子。`neg` 不行：RoPE 的旋转半区要它，
    它是设备算子。`slice` 可以：它的实参是 (dim, start, end) 描述的子区间，
    命令编码不了，所以它留主机（`view` 同样留主机，但换成 `slice` 更能说明
    「留主机」的理由是实参形态，不是「布局算子都留主机」）。
    """
    graph = Graph()
    x = graph.placeholder("x")
    dpu_tanh = graph.call_function(torch.ops.aten.tanh.default, (x,))
    # 主机断点要用**真正**留主机的算子：`slice` 已下设备，改用脚手架。
    host_neg = graph.call_function(
        torch.ops.aten._assert_tensor_metadata.default, (dpu_tanh,))
    graph.output(host_neg)
    gm = GraphModule(torch.nn.Module(), graph)
    for node in (x, dpu_tanh, host_neg):
        node.meta["val"] = torch.empty(2, 4)
    partition_graph(gm)

    edges = propagate_specs(gm, ShardStrategy(name="a", num_dpus=2, dpu_ids=(5, 2), weight_rules=()))

    (edge,) = [edge for edge in edges if edge.src == dpu_tanh.name and edge.dst == host_neg.name]
    assert edge.type == "all_gather"
    assert edge.src_spec.placement == REPLICATE
    assert edge.src_spec.device == DEVICE_DPU
    assert edge.dst_spec.placement == REPLICATE
    assert edge.dst_spec.device == DEVICE_HOST
    assert edge.src_loc == {"device": DEVICE_DPU, "dpus": [2, 5]}
    assert edge.dst_loc == {"device": DEVICE_HOST}
    assert min(edge.src_spec.shard_map) == 2  # 复制布局使用编号最小的 DPU 作为传输源。


def test_propagation_clears_stale_metadata_and_reuses_stable_edge_ids() -> None:
    """验证重复传播会重建节点规格和通信边。"""
    gm, nodes = _keyword_add_graph()
    partition_graph(gm)
    output = next(node for node in gm.graph.nodes if node.op == "output")
    nodes["left"].meta[SPEC_META_KEY] = object()
    nodes["left"].meta[REDISTRIBUTE_META_KEY] = ["stale"]
    output.meta[SPEC_META_KEY] = object()
    output.meta[REDISTRIBUTE_META_KEY] = ["stale"]
    config = ShardStrategy(name="a", num_dpus=2, dpu_ids=(2, 5), weight_rules=())

    first = propagate_specs(gm, config)
    second = propagate_specs(gm, config)

    assert first == second
    assert [edge.edge_id for edge in second] == list(range(len(second)))
    assert nodes["left"].meta[REDISTRIBUTE_META_KEY] == []
    assert SPEC_META_KEY not in output.meta
    assert all("stale" not in node.meta[REDISTRIBUTE_META_KEY] for node in gm.graph.nodes)


# DPU 编号范围测试。


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
    # 分片一出生就带显式排布：行主序紧密是**数据**，不再是 `bytes_of` 的假设。
    assert w1_spec.shard_map[0] == TensorShardDetail(0, 0, 0, 3, (3, 4),
                                                     elem_strides=(4, 1))
    assert w1_spec.shard_map[1] == TensorShardDetail(1, 0, 3, 6, (3, 4),
                                                     elem_strides=(4, 1))

    # 列切输出沿第 1 维分片。
    y1_spec = nodes["y1"].meta[SPEC_META_KEY]
    assert y1_spec.device == DEVICE_DPU
    assert y1_spec.placement == Placement("Shard", 1)
    assert y1_spec.shard_map[0].local_shape == (1, 3)
    assert (y1_spec.shard_map[0].start_idx, y1_spec.shard_map[0].end_idx) == (0, 3)
    assert (y1_spec.shard_map[1].start_idx, y1_spec.shard_map[1].end_idx) == (3, 6)

    # 输入从主机复制到各 DPU。
    (x_edge,) = nodes["y1"].meta[REDISTRIBUTE_META_KEY]
    assert x_edge.type == "scatter"
    assert x_edge.src_loc == {"device": DEVICE_HOST}
    assert x_edge.dst_loc == {"device": DEVICE_DPU, "dpus": [0, 1]}

    # 行切层消费对齐分片并输出部分和。
    y2_spec = nodes["y2"].meta[SPEC_META_KEY]
    assert y2_spec.placement == PARTIAL_SUM
    assert y2_spec.reduce_type == "sum"
    assert y2_spec.shard_map[0].local_shape == (1, 4)  # Partial 每台持完整形状
    assert nodes["y2"].meta[REDISTRIBUTE_META_KEY] == []

    # 主机归一化层需要全量归约结果。
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

    # 将输入和权重写入对应 DPU。
    specs = {k: nodes[k].meta[SPEC_META_KEY] for k in ("x", "w1", "w2", "y1", "y2")}
    for dpu_id in range(2):
        backend.copy_to_dpu(dpu_id, offsets["x"], x_ref.numpy())
        backend.copy_to_dpu(dpu_id, offsets["w1"], shard_of(w1_ref, specs["w1"].shard_map[dpu_id]))
        backend.copy_to_dpu(dpu_id, offsets["w2"], shard_of(w2_ref, specs["w2"].shard_map[dpu_id]))

    # 计算列切线性层的本地输出。
    for dpu_id, det in specs["y1"].shard_map.items():
        x_local = backend.copy_from_dpu(dpu_id, offsets["x"], (1, 4), np.float32)
        w1_local = backend.copy_from_dpu(dpu_id, offsets["w1"], (3, 4), np.float32)
        backend.copy_to_dpu(dpu_id, offsets["y1"], x_local @ w1_local.T)

    # 按全局区间拼接分片。
    y1_parts = [
        backend.copy_from_dpu(dpu_id, offsets["y1"], det.local_shape, np.float32)
        for dpu_id, det in sorted(specs["y1"].shard_map.items(), key=lambda kv: kv[1].start_idx)
    ]
    y1 = np.concatenate(y1_parts, axis=1)
    np.testing.assert_allclose(y1, (x_ref @ w1_ref.T).numpy(), rtol=1e-6)

    # 累加行切层的局部结果。
    y2 = np.zeros((1, 4), dtype=np.float32)
    for dpu_id, det in specs["y2"].shard_map.items():
        y1_local = backend.copy_from_dpu(dpu_id, offsets["y1"], (1, 3), np.float32)
        w2_local = backend.copy_from_dpu(dpu_id, offsets["w2"], (4, 3), np.float32)
        backend.copy_to_dpu(dpu_id, offsets["y2"], y1_local @ w2_local.T)
        y2 += backend.copy_from_dpu(dpu_id, offsets["y2"], det.local_shape, np.float32)
    np.testing.assert_allclose(y2, ((x_ref @ w1_ref.T) @ w2_ref.T).numpy(), rtol=1e-6)


# 小型 Llama 图的分片断言。


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
            # 建表权重随 embedding 一起下了设备：查表是设备算子，表就得在设备上。
            assert spec.device == DEVICE_DPU, target
            assert spec.placement == REPLICATE, target
    # 每个 DPU 对应一个注意力头。
    q_weight = by_name["model_model_layers_0_self_attn_q_proj_weight"].meta[SPEC_META_KEY]
    assert q_weight.shard_map[0].local_shape == (16, 64)
    assert q_weight.shard_map[3].start_idx == 48


def test_tiny_llama_propagation_structure() -> None:
    gm, by_name, edges = _propagate_tiny_llama()

    # 每个张量节点都具有规格信息。
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "get_attr" and not isinstance(node.meta.get("val"), torch.Tensor):
            continue
        assert SPEC_META_KEY in node.meta, node.name

    # 行切输出在残差相加前归约。
    for proj in ("o_proj", "down_proj"):
        weight = next(n for n in gm.graph.nodes if n.op == "get_attr" and f"{proj}.weight" in n.target)
        (linear,) = weight.users
        assert linear.meta[SPEC_META_KEY].placement == PARTIAL_SUM, proj
        (add,) = [u for u in linear.users if u.target == torch.ops.aten.add.Tensor]
        (edge,) = [e for e in add.meta[REDISTRIBUTE_META_KEY] if e.src == linear.name]
        assert edge.type == "all_reduce"
        assert edge.dst_loc == {"device": DEVICE_DPU, "dpus": [0, 1, 2, 3]}

    # 行切输入已位于目标 DPU，无需 scatter。
    down_weight = next(n for n in gm.graph.nodes if n.op == "get_attr" and "down_proj.weight" in n.target)
    (down_linear,) = down_weight.users
    act = down_linear.args[0]
    assert act.meta[SPEC_META_KEY].placement == Placement("Shard", 2)
    assert all(e.src != act.name for e in down_linear.meta[REDISTRIBUTE_META_KEY])

    # 每层包含两次 all_reduce。
    assert sum(e.type == "all_reduce" for e in edges) == 2
    # 视图族下设备后允许 `local_slice`（Replicate → Shard 的 DPU 内切分）。
    assert {e.type for e in edges} <= {
        "all_reduce", "all_gather", "scatter", "local_slice"}
    # 常驻权重不参与重分布。
    for edge in edges:
        assert by_name[edge.src].op != "get_attr"
        for loc in (edge.src_loc, edge.dst_loc):
            assert loc["device"] in ("host", "dpu")
            assert loc["device"] == "host" or sorted(loc["dpus"]) == [0, 1, 2, 3]

    # 输出 logits 经 all_gather 返回主机。
    output = next(n for n in gm.graph.nodes if n.op == "output")
    (final_edge,) = output.meta[REDISTRIBUTE_META_KEY]
    assert final_edge.type == "all_gather"
    assert final_edge.from_placement == Placement("Shard", 2)

    # 报告可打印，重复传播结果一致。
    assert "all_reduce" in format_spec_report(gm, edges, max_nodes=10)
    assert len(propagate_specs(gm, llama_shard_config(
        4, num_heads=4, num_kv_heads=4, intermediate_size=176, vocab_size=32000))) == len(edges)


def test_llama_shard_config_rejects_contract_violations() -> None:
    """纯张量并行下 tp_width == num_dpus，契约校验的报错落在 tp_width 上。"""
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
        propagate_specs(gm, ShardStrategy(name="a", num_dpus=2, weight_rules=(("w1", "col"),)))


def _split_head_llama() -> tuple[GraphModule, ShardStrategy]:
    """拆头之后的 tiny llama 图 —— 注意力被展成逐头的 matmul + softmax。

    这是 `runtime.compile` 真实走的形态：先 `split_attention_heads`，
    再标注、切分。用夹具默认那套维度，与 `test_placement_export` 一致。
    """
    from graph.split_heads import split_attention_heads
    from tests.test_placement_export import annotated_tiny_llama

    gm = annotated_tiny_llama.__wrapped__()
    split_attention_heads(gm)
    partition_graph(gm)
    strategy = llama_strategy(
        2, num_stages=1, num_heads=4, num_kv_heads=4,
        intermediate_size=176, vocab_size=32000, num_layers=1,
    )
    return gm, strategy


def test_split_head_graph_has_rules_for_every_device_op() -> None:
    """拆头之后每个设备侧算子都要有切分规则，不能卡在「缺少切分规则」。

    拆头把注意力展成 `aten.matmul` 与 `aten._softmax`（每头一次），这两个
    原本不在 RULE_TABLE 里。缺规则的后果不是退回主机，而是整条编译**直接失败**
    ——规则表里没有、`HOST_ONLY` 里也没有，`propagate_specs` 当场抛。
    """
    gm, strategy = _split_head_llama()
    edges = propagate_specs(gm, strategy)  # 缺规则会在这里抛

    # 注意力那两个矩阵乘与 softmax 必须真的被判成设备节点并拿到布局。
    from graph.spec_prop import SPEC_META_KEY

    matmul_nodes = [n for n in gm.graph.nodes if str(n.target).endswith("matmul.default")]
    softmax_nodes = [n for n in gm.graph.nodes if str(n.target).endswith("_softmax.default")]
    assert matmul_nodes and softmax_nodes, "拆头图里没有 matmul/softmax，夹具变了"
    for node in matmul_nodes + softmax_nodes:
        spec = node.meta.get(SPEC_META_KEY)
        assert spec is not None, f"{node.name} 没有拿到布局规格"
        assert spec.device == "dpu", f"{node.name} 落到了 {spec.device}"
    assert edges is not None


def test_weight_slice_keeps_the_weight_shard() -> None:
    """切权重得到的片段保持权重的切分布局，而不是变成复制。

    分块沿输出特征维切开 linear 的权重。视图规则会把切片输出判成复制，
    下游 linear 只接受按列或按行切分的权重，于是整条编译失败。
    """
    from graph.strategy import ShardStrategy

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("up_proj_weight")
    part = graph.call_function(torch.ops.aten.slice.Tensor, (w, 0, 0, 32))
    out = graph.call_function(torch.ops.aten.linear.default, (x, part))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("up_proj_weight", torch.empty(64, 64))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 4, 64)
    w.meta["val"] = torch.empty(64, 64)
    part.meta["val"] = torch.empty(32, 64)
    out.meta["val"] = torch.empty(1, 4, 32)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("up_proj_weight", "col"),))

    propagate_specs(gm, strategy)

    spec = part.meta[SPEC_META_KEY]
    assert spec.placement.kind == "Shard", spec.placement
    assert spec.placement.dim == 0, spec.placement


def _slice_ranges(node):
    """每台 DPU 在这个切片上持有的全局区间。"""
    spec = node.meta[SPEC_META_KEY]
    return {dpu: (det.start_idx, det.end_idx)
            for dpu, det in spec.shard_map.items()}


def test_weight_slice_ranges_follow_the_sliced_span() -> None:
    """按列切的权重被沿行切开后，每台只保留与自己持有区间的交集。

    权重按连续的行段分给各台，一台拿不到别人的行。分块再沿行切开时，
    每段只落在持有这些行的台上，而不是把每段的行重新对半。352 行两台各
    持 176 行，176 之前的段全在第一台，之后的全在第二台。
    """
    from graph.opt_passes import optimize_graph

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", torch.empty(352, 64, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 8, 64, dtype=torch.float16)
    w.meta["val"] = torch.empty(352, 64, dtype=torch.float16)
    out.meta["val"] = torch.empty(1, 8, 352, dtype=torch.float16)
    partition_graph(gm)
    optimize_graph(gm, wram_bytes=4096, tp_width=2)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))

    propagate_specs(gm, strategy)

    # 每台持有的全局行段，取自权重本身的切分。
    held = {dpu: (det.start_idx, det.end_idx)
            for dpu, det in w.meta[SPEC_META_KEY].shard_map.items()}
    assert held == {0: (0, 176), 1: (176, 352)}, held
    # 只看直接切权重行的切片。归约维切片和嵌套切片的起点会和它重复，
    # 用起点当键会把行切片的区间覆盖掉。
    slices = [n for n in gm.graph.nodes
              if n.target is torch.ops.aten.slice.Tensor and int(n.args[1]) == 0
              and n.args[0].op == "get_attr"]
    ranges = {}
    for node in slices:
        start = int(node.args[2])
        ranges[start] = _slice_ranges(node)
    # 行被切成 16 一段。每段只落在持有这些行的台上：176 之前全在第一台，
    # 之后全在第二台，没有一段被重新对半。
    assert len(ranges) == 352 // 16
    assert ranges[0] == {0: (0, 16)}, ranges[0]
    assert ranges[176 - 16] == {0: (160, 176)}, ranges[176 - 16]
    assert ranges[176] == {1: (176, 192)}, ranges[176]
    assert ranges[352 - 16] == {1: (336, 352)}, ranges[352 - 16]


def test_weight_slice_rows_split_evenly_inside_each_span() -> None:
    """按列切的权重被沿行切开后，每台只持有落在自己行段里的那些行。

    352 行两台各持 176 行。分块按 16 行一段切开，176 之前的段全在第一台，
    之后的全在第二台。把每段对半会分给没有这些行的台。
    """
    from graph.opt_passes import optimize_graph

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", torch.empty(352, 64, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 8, 64, dtype=torch.float16)
    w.meta["val"] = torch.empty(352, 64, dtype=torch.float16)
    out.meta["val"] = torch.empty(1, 8, 352, dtype=torch.float16)
    partition_graph(gm)
    optimize_graph(gm, wram_bytes=4096, tp_width=2)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))

    propagate_specs(gm, strategy)

    # 只看沿切分轴（行）的切片，归约维切片的起点会和行切片重复。
    slices = [n for n in gm.graph.nodes
              if n.target is torch.ops.aten.slice.Tensor and int(n.args[1]) == 0
              and n.meta[SPEC_META_KEY].placement.kind == "Shard"]
    rows = {}
    for node in slices:
        start = int(node.args[2])
        spec = node.meta[SPEC_META_KEY]
        rows[start] = {dpu: det.local_shape[0] for dpu, det in spec.shard_map.items()}
    # 每段 16 行，且整段落在持有它的那一台，不与另一台对半。
    assert rows[0] == {0: 16}, rows[0]
    assert rows[160] == {0: 16}, rows[160]
    assert rows[176] == {1: 16}, rows[176]
    assert rows[336] == {1: 16}, rows[336]


def test_nested_weight_slice_keeps_the_shard() -> None:
    """权重先按输出特征切、再按归约维切时，第二刀仍然保持切分。

    两维同时切会产生切片套切片。第二刀的源是第一刀而不是 `get_attr`，
    按「源必须是 get_attr」判定会把它当成普通视图判成复制，linear 随后
    因为拿到复制的权重而报不支持的布局。归约维不是切分轴，切完每台
    持有的行不变，只是变短。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    outer = graph.call_function(torch.ops.aten.slice.Tensor, (w, 0, 0, 8))
    inner = graph.call_function(torch.ops.aten.slice.Tensor, (outer, 1, 0, 4))
    out = graph.call_function(torch.ops.aten.linear.default, (x, inner))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", torch.empty(16, 8))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 2, 4)
    w.meta["val"] = torch.empty(16, 8)
    outer.meta["val"] = torch.empty(8, 8)
    inner.meta["val"] = torch.empty(8, 4)
    out.meta["val"] = torch.empty(1, 2, 8)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))

    propagate_specs(gm, strategy)

    spec = inner.meta[SPEC_META_KEY]
    assert spec.placement == Placement("Shard", 0), spec.placement
    # 第一刀取了前 8 行，整段落在持有 [0,8) 的第一台上。
    assert _slice_ranges(inner) == {0: (0, 8)}, _slice_ranges(inner)
    for det in spec.shard_map.values():
        # 归约维被切短到 4，持有的行数不变。
        assert det.local_shape == (8, 4), det.local_shape


def test_weight_slice_on_the_shard_axis_clips_to_its_span() -> None:
    """切片切的就是切分轴时，每台只保留与切片区间的交集。

    按行切沿权重第 1 维分片，两台各持有 2048 列。切片取第 1024 到 3072 列，
    横跨两台的分界：每台只持有其中落在自己分片里的 1024 列。照搬整块权重的
    切分会让每台各拿 1024 列且都从第 0 列算起，两台拿到同一批列。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("down_proj_weight")
    part = graph.call_function(torch.ops.aten.slice.Tensor, (w, 1, 1024, 3072))
    out = graph.call_function(torch.ops.aten.linear.default, (x, part))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("down_proj_weight", torch.empty(2048, 4096))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 4, 2048)
    w.meta["val"] = torch.empty(2048, 4096)
    part.meta["val"] = torch.empty(2048, 2048)
    out.meta["val"] = torch.empty(1, 4, 2048)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("down_proj_weight", "row"),))

    propagate_specs(gm, strategy)

    spec = part.meta[SPEC_META_KEY]
    assert spec.placement == Placement("Shard", 1), spec.placement
    assert _slice_ranges(part) == {0: (1024, 2048), 1: (2048, 3072)}
    for det in spec.shard_map.values():
        assert det.local_shape == (2048, 1024), det.local_shape


def test_tiled_row_sharded_weight_propagates() -> None:
    """按行切的权重被分块切开后，拼接结果仍能传播。

    行切的 linear 产出部分和，视图规则拒绝部分和输入。分块沿特征维
    切开再拼回时，每段要先归约成复制，拼接才成立。
    """
    from graph.strategy import ShardStrategy

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("down_proj_weight")
    part = graph.call_function(torch.ops.aten.slice.Tensor, (w, 0, 0, 32))
    tile = graph.call_function(torch.ops.aten.linear.default, (x, part))
    out = graph.call_function(torch.ops.aten.cat.default, ([tile], -1))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("down_proj_weight", torch.empty(64, 64))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 4, 64)
    w.meta["val"] = torch.empty(64, 64)
    part.meta["val"] = torch.empty(32, 64)
    tile.meta["val"] = torch.empty(1, 4, 32)
    out.meta["val"] = torch.empty(1, 4, 32)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("down_proj_weight", "row"),))

    propagate_specs(gm, strategy)

    assert out.meta[SPEC_META_KEY].placement.kind == "Replicate"


def test_cat_of_aligned_column_shards_stays_sharded() -> None:
    """沿拼接维切分且切分对齐的几段拼接后，每台只拼自己的列，不先聚成副本。

    权重按列切成两台各 6 行，沿行切成三段 [0:4)、[4:8)、[8:12)：第一段整段
    在第一台、中段跨分界、第三段整段在第二台。三段拼接后每台持有的列正好
    是完整输出的连续一半，不需要先各聚成完整副本再拼。
    """
    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("w")
    parts = []
    for start, end in ((0, 4), (4, 8), (8, 12)):
        sliced = graph.call_function(torch.ops.aten.slice.Tensor, (w, 0, start, end))
        parts.append(graph.call_function(torch.ops.aten.linear.default, (x, sliced)))
    cat = graph.call_function(torch.ops.aten.cat.default, (parts, 2))
    graph.output(cat)
    mod = torch.nn.Module()
    mod.register_buffer("w", torch.empty(12, 4))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 2, 4)
    w.meta["val"] = torch.empty(12, 4)
    for node in gm.graph.nodes:
        if node.target is torch.ops.aten.slice.Tensor:
            node.meta["val"] = torch.empty(4, 4)
        elif node.target is torch.ops.aten.linear.default:
            node.meta["val"] = torch.empty(1, 2, 4)
    cat.meta["val"] = torch.empty(1, 2, 12)
    partition_graph(gm)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("w", "col"),))

    edges = propagate_specs(gm, strategy)

    spec = cat.meta[SPEC_META_KEY]
    assert spec.placement == Placement("Shard", 2), spec.placement
    assert _slice_ranges(cat) == {0: (0, 6), 1: (6, 12)}, _slice_ranges(cat)
    gathered = [e for e in edges if e.dst == cat.name and e.type == "all_gather"]
    assert not gathered, f"切分对齐的拼接仍聚合了 {[e.src for e in gathered]}"


def test_two_axis_tile_propagates_without_all_gather() -> None:
    """两维同时切时，拼接结果保持切分，不再退回全聚合。

    预算小到必须同时切输出特征维和归约维。权重按列切成两台，每段 linear
    的输出应跟着权重行走，沿特征维拼回后每台只持有自己的列。用归约维的
    切片起点去减行号会得到负区间，传播直接报错；退回全聚合则会给拼接
    插入 all_gather，把分块省下的通信加倍搬回来。
    """
    from graph.opt_passes import optimize_graph

    graph = Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("gate_proj_weight")
    out = graph.call_function(torch.ops.aten.linear.default, (x, w))
    graph.output(out)
    mod = torch.nn.Module()
    mod.register_buffer("gate_proj_weight", torch.empty(64, 64, dtype=torch.float16))
    gm = GraphModule(mod, graph)
    x.meta["val"] = torch.empty(1, 4, 64, dtype=torch.float16)
    w.meta["val"] = torch.empty(64, 64, dtype=torch.float16)
    out.meta["val"] = torch.empty(1, 4, 64, dtype=torch.float16)
    partition_graph(gm)
    # 预算 256 字节时 M、N、K 都要切，必然出现切片套切片。
    # 浮点按提升到 float32 后的宽度计，128 字节已经没有任何切法压得进。
    optimize_graph(gm, wram_bytes=256, tp_width=2)
    strategy = ShardStrategy(
        name="tp2", num_dpus=2, dpu_ids=(0, 1),
        weight_rules=(("gate_proj_weight", "col"),))

    edges = propagate_specs(gm, strategy)

    cats = [n for n in gm.graph.nodes if n.target is torch.ops.aten.cat.default]
    assert cats, "两维切分应产生拼接"
    # 沿输出特征维拼回的那一层，结果应保持切分。
    feature_cats = [n for n in cats
                    if int(n.args[1]) == n.meta["val"].ndim - 1]
    assert feature_cats, "没有沿特征维的拼接"
    for cat in feature_cats:
        spec = cat.meta[SPEC_META_KEY]
        assert spec.placement.kind == "Shard", spec.placement
        for det in spec.shard_map.values():
            assert det.start_idx >= 0, det
    gathered = [e for e in edges
                if e.dst in {c.name for c in feature_cats} and e.type == "all_gather"]
    assert not gathered, f"两维切分的拼接退回了全聚合 {[e.src for e in gathered]}"
