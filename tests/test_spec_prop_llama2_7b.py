"""真实 Llama-2-7B 的问题 1 → 问题 2 端到端结构验证（编译期判据，方案二.(11) 契约）。

加载官方权重 torch.export 出静态图（2139 节点），过 partition_graph 后做切分传播。
第 1 阶段窄白名单下，注意力/RMSNorm 的大量分解算子（view、rope、softmax、pow/mean
等）留在 host，其 host↔dpu 往返产生 scatter / all_gather 边——属预期行为；结构性判据
是 Megatron 配对保持完好：每层恰有 o_proj、down_proj 两条 all_reduce（32 层共 64 条），
权重 pinned 且永不重分布，logits 经一次 all_gather 回 host。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.graph_meta import (
    DEVICE_DPU,
    DEVICE_HOST,
    DEVICE_META_KEY,
    REDISTRIBUTE_META_KEY,
    SPEC_META_KEY,
)
from contracts.pim_tensor_spec import Placement
from graph.partition import partition_graph
from graph.spec_prop import format_spec_report, llama_shard_config, propagate_specs
from tests.test_partition import _FixedMaskLlama

MODEL_DIR = Path(
    "/media/disk/fengjingge/src/flagOS/flagOS-installed/model-inference/models/Llama-2-7b-hf"
)
NUM_DPUS = 8  # 7B fp16 ≈ 13.5GiB，8 台 8GB DPU 容量充裕；8 整除 32 heads / 11008 / 32000
SEQ_LEN = 16

REPLICATE = Placement("Replicate")
PARTIAL_SUM = Placement("Partial", reduce_type="sum")

pytestmark = pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="需要本地 Llama-2-7b-hf 权重")


@pytest.fixture(scope="module")
def annotated_llama2():
    """真实 7B：export → partition → propagate，模块内只跑一次。"""
    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16).eval()
    cfg = model.config
    input_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.float16)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()
    partition_graph(gm)
    shard_config = llama_shard_config(
        NUM_DPUS,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
    )
    edges = propagate_specs(gm, shard_config)
    return gm, cfg, shard_config, edges


def _get_attr_node(gm, pattern: str):
    (node,) = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    return node


def test_every_tensor_node_has_spec(annotated_llama2) -> None:
    gm, _, _, _ = annotated_llama2
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if node.op == "get_attr" and not isinstance(node.meta.get("val"), torch.Tensor):
            continue  # 非张量 get_attr（rotary 子图模块等），不是数据边
        assert SPEC_META_KEY in node.meta, node.name


def test_weight_initial_sharding_and_dpu_mapping(annotated_llama2) -> None:
    """每份权重的起始布局与逐 DPU 分片映射（shard_map 连续全覆盖、切点对齐 head 边界）。"""
    gm, cfg, shard_config, _ = annotated_llama2
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    for node in gm.graph.nodes:
        if node.op != "get_attr" or not isinstance(node.meta.get("val"), torch.Tensor):
            continue
        target, spec = node.target, node.meta[SPEC_META_KEY]
        if any(k in target for k in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "lm_head")):
            assert spec.placement == Placement("Shard", 0), target
        elif any(k in target for k in ("o_proj", "down_proj")):
            assert spec.placement == Placement("Shard", 1), target
        elif "layernorm" in target or target.endswith("norm.weight"):
            assert spec.placement == REPLICATE and spec.device == DEVICE_DPU, target
        elif "embed_tokens" in target or "inv_freq" in target:
            assert spec.device == DEVICE_HOST, target
            continue
        if spec.placement.kind != "Shard":
            continue
        assert spec.residency == "pinned", target
        shape = tuple(node.meta["val"].shape)
        dim = spec.placement.dim
        assert set(spec.shard_map) == set(range(NUM_DPUS)), target
        width = shape[dim] // NUM_DPUS
        for dpu_id, det in sorted(spec.shard_map.items()):  # 单段连续、均匀、全覆盖（契约 1/2/5）
            assert (det.dpu_id, det.shard_dim) == (dpu_id, dim)
            assert (det.start_idx, det.end_idx) == (dpu_id * width, (dpu_id + 1) * width)
            assert det.local_shape == shape[:dim] + (width,) + shape[dim + 1 :]
        if any(k in target for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            assert width % head_dim == 0, f"{target} 切点未对齐 head 边界"
    q_w = _get_attr_node(gm, "layers.0.self_attn.q_proj.weight").meta[SPEC_META_KEY]
    assert q_w.shard_map[0].local_shape == (cfg.hidden_size // NUM_DPUS, cfg.hidden_size)
    assert q_w.shard_map[0].end_idx // head_dim == cfg.num_attention_heads // NUM_DPUS  # 每台 4 个 head


def test_megatron_pattern_holds_for_all_32_layers(annotated_llama2) -> None:
    """列→行配对：o/down 行切产出 Partial，残差 add 上各一条 all_reduce，down 输入零通信。"""
    gm, cfg, _, edges = annotated_llama2
    for layer in range(cfg.num_hidden_layers):
        for proj, pattern in (
            ("o_proj", f"layers.{layer}.self_attn.o_proj.weight"),
            ("down_proj", f"layers.{layer}.mlp.down_proj.weight"),
        ):
            weight = _get_attr_node(gm, pattern)
            (linear,) = weight.users
            assert linear.meta[SPEC_META_KEY].placement == PARTIAL_SUM, f"layers.{layer}.{proj}"
            (add,) = [u for u in linear.users if u.target == torch.ops.aten.add.Tensor]
            (edge,) = [e for e in add.meta[REDISTRIBUTE_META_KEY] if e.src == linear.name]
            assert edge.type == "all_reduce"
            assert edge.dst_loc == {"device": DEVICE_DPU, "dpus": list(range(NUM_DPUS))}
        (down_linear,) = _get_attr_node(gm, f"layers.{layer}.mlp.down_proj.weight").users
        act = down_linear.args[0]
        assert act.meta[SPEC_META_KEY].placement == Placement("Shard", 2)
        assert all(e.src != act.name for e in down_linear.meta[REDISTRIBUTE_META_KEY])
    assert sum(e.type == "all_reduce" for e in edges) == 2 * cfg.num_hidden_layers
    assert {e.type for e in edges} <= {"all_reduce", "all_gather", "scatter"}


def test_redistribute_edges_well_formed_and_weights_never_moved(annotated_llama2) -> None:
    gm, _, _, edges = annotated_llama2
    by_name = {n.name: n for n in gm.graph.nodes}
    assert [e.edge_id for e in edges] == list(range(len(edges)))
    for edge in edges:
        assert by_name[edge.src].op != "get_attr"  # 权重 pinned，永不出现在边的任何一端
        assert edge.src_spec == by_name[edge.src].meta[SPEC_META_KEY]
        assert edge.dst_spec.placement == edge.to_placement
        for loc in (edge.src_loc, edge.dst_loc):
            assert loc["device"] in (DEVICE_HOST, DEVICE_DPU)
            assert loc["device"] == DEVICE_HOST or sorted(loc["dpus"]) == list(range(NUM_DPUS))
        assert edge.nbytes > 0


def test_logits_exit_via_all_gather(annotated_llama2) -> None:
    gm, _, _, _ = annotated_llama2
    lm_head = _get_attr_node(gm, "lm_head.weight")
    assert lm_head.meta[SPEC_META_KEY].placement == Placement("Shard", 0)
    output = next(n for n in gm.graph.nodes if n.op == "output")
    (final_edge,) = output.meta[REDISTRIBUTE_META_KEY]
    assert final_edge.type == "all_gather"
    assert final_edge.from_placement == Placement("Shard", 2)
    assert final_edge.to_placement == REPLICATE
    assert final_edge.dst_loc == {"device": DEVICE_HOST}


def test_report_printable_and_propagation_idempotent(annotated_llama2) -> None:
    gm, cfg, shard_config, edges = annotated_llama2
    report = format_spec_report(gm, edges, max_nodes=12)
    assert "all_reduce" in report and "redistribute" in report
    # 只打印首尾摘要（-s 可见）：节点布局示例 + 分类统计 + 代表性权重的逐 DPU 映射
    lines = report.splitlines()
    print("\n" + "\n".join(lines[:14]) + "\n  ...\n" + lines[-1])
    q_w = _get_attr_node(gm, "layers.0.self_attn.q_proj.weight").meta[SPEC_META_KEY]
    print(f"layers.0.q_proj.weight Shard({q_w.placement.dim}) pinned: "
          f"{[(d.dpu_id, d.start_idx, d.end_idx, d.local_shape) for d in q_w.shard_map.values()]}")
    assert len(propagate_specs(gm, shard_config)) == len(edges)
