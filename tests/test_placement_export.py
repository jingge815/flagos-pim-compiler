"""验证图编译器放置结果写入 GeneSim IR 的逻辑。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.fx import GraphModule
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.partition import partition_graph
from graph.spec_prop import llama_shard_config, propagate_specs
from graph.strategy import llama_strategy
from genesim_bridge.placement_export import export_placement_to_genesim
from tests.test_partition import _FixedMaskLlama

NUM_DPUS = 4
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 176
NUM_HEADS = 4


@pytest.fixture(scope="module")
def annotated_tiny_llama() -> GraphModule:
    sequence_length = 16
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_hidden_layers=1,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=NUM_HEADS,
            max_position_embeddings=sequence_length,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1)
    causal_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.float32)
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()
    partition_graph(gm)
    shard_config = llama_shard_config(
        NUM_DPUS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=32000,
    )
    propagate_specs(gm, shard_config)
    return gm


GQA_NUM_HEADS = 8
GQA_NUM_KV_HEADS = 4
GQA_TP_WIDTH = 2
GQA_HIDDEN_SIZE = 64
GQA_INTERMEDIATE_SIZE = 176


@pytest.fixture(scope="module")
def annotated_gqa_llama() -> GraphModule:
    """分组查询注意力的小模型：k/v 头数少于 q，三者本地宽度并不相同。"""
    sequence_length = 16
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000,
            hidden_size=GQA_HIDDEN_SIZE,
            intermediate_size=GQA_INTERMEDIATE_SIZE,
            num_hidden_layers=1,
            num_attention_heads=GQA_NUM_HEADS,
            num_key_value_heads=GQA_NUM_KV_HEADS,
            max_position_embeddings=sequence_length,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    input_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    blocked = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
    )
    causal_mask = torch.zeros(
        (1, 1, sequence_length, sequence_length), dtype=torch.float32
    )
    causal_mask.masked_fill_(blocked, torch.finfo(causal_mask.dtype).min)
    gm = torch.export.export(
        _FixedMaskLlama(model), (input_ids, causal_mask), strict=True
    ).module()
    partition_graph(gm)
    # tp_width=2：GQA 下 num_kv_heads=4 也能被整除。
    strategy = llama_strategy(
        GQA_TP_WIDTH,
        num_stages=1,
        num_heads=GQA_NUM_HEADS,
        num_kv_heads=GQA_NUM_KV_HEADS,
        intermediate_size=GQA_INTERMEDIATE_SIZE,
        vocab_size=32000,
        num_layers=1,
    )
    propagate_specs(gm, strategy)
    return gm


def _write_fixture_ir(path: Path) -> None:
    """写入含四个 GEMM 和注意力算子的最小 GeneSim IR。"""
    operators = [
        {"op_id": 0, "op_type": "GEMM", "device_hint": "gpu",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 1, "op_type": "GEMV_SCORE", "device_hint": "pim",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 2, "op_type": "SOFTMAX", "device_hint": "pim",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 3, "op_type": "GEMV_CONTEXT", "device_hint": "pim",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 4, "op_type": "GEMM", "device_hint": "gpu",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 5, "op_type": "GEMM", "device_hint": "gpu",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 6, "op_type": "GELU", "device_hint": "cpu",
         "input_shapes": [], "output_shapes": []},
        {"op_id": 7, "op_type": "GEMM", "device_hint": "gpu",
         "input_shapes": [], "output_shapes": []},
    ]
    ir = {
        "model_id": "tiny-llama-fixture",
        "num_layers": 1,
        "num_heads": NUM_HEADS,
        "head_dim": HIDDEN_SIZE // NUM_HEADS,
        "hidden_size": HIDDEN_SIZE,
        "operators": operators,
        "dependencies": [],
        "subgraphs": [[op["op_id"] for op in operators]],
    }
    path.write_text(json.dumps(ir))


def test_gemm_device_hint_overwritten_to_pim(annotated_tiny_llama, tmp_path) -> None:
    ir_path = tmp_path / "base.ir"
    out_ir_path = tmp_path / "placed.ir"
    sidecar_path = tmp_path / "placed_sidecar.json"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(annotated_tiny_llama, ir_path, out_ir_path, sidecar_path)

    placed = json.loads(out_ir_path.read_text())
    ops_by_id = {op["op_id"]: op for op in placed["operators"]}

    gemm_ids = [0, 4, 5, 7]
    for op_id in gemm_ids:
        assert ops_by_id[op_id]["device_hint"] == "pim", op_id
        assert str(op_id) in sidecar["operators"], op_id
        dpu_id = sidecar["operators"][str(op_id)]["dpu_id"]
        assert 0 <= dpu_id < NUM_DPUS, dpu_id

    for op_id in (1, 2, 3, 6):
        assert ops_by_id[op_id]["device_hint"] == {
            1: "pim", 2: "pim", 3: "pim", 6: "cpu",
        }[op_id], op_id
        assert str(op_id) not in sidecar["operators"], op_id


def test_local_shard_widths_written_with_qkv_summed(
    annotated_tiny_llama, tmp_path
) -> None:
    """sidecar 要带上本地分片宽度，qkv 那个 GEMM 的输出宽度是 q/k/v 三者之和。

    GeneSim 的 IR 把 q/k/v 合并成一个 GEMM（真实 7B 上是 4096 -> 12288），
    而代表权重只是 q_proj。只算 q 一份的话这个 GEMM 的成本会被低估到三分之一。
    这个 fixture 是多头注意力（kv 头数等于 q 头数），所以三者宽度相同、
    累加结果恰好等于 q 的三倍；GQA 下不成立，见下一个测试。
    """
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(
        annotated_tiny_llama, ir_path,
        tmp_path / "placed.ir", tmp_path / "sc.json",
    )

    tp_width = NUM_DPUS          # 该 fixture 是纯张量并行，段内宽度即 DPU 数
    ops = sidecar["operators"]

    # op0 = qkv：q/k/v 都按输出维切，各 HIDDEN/tp，累加三份。
    qkv = ops["0"]
    assert qkv["local_in_features"] == HIDDEN_SIZE
    assert qkv["local_out_features"] == 3 * (HIDDEN_SIZE // tp_width)

    # op4 = o_proj：按规约维切，所以是入口宽度变窄、出口保持全宽。
    o_proj = ops["4"]
    assert o_proj["local_in_features"] == HIDDEN_SIZE // tp_width
    assert o_proj["local_out_features"] == HIDDEN_SIZE

    # op5 = fc1（gate_proj）：按输出维切，倍数为一。
    fc1 = ops["5"]
    assert fc1["local_in_features"] == HIDDEN_SIZE
    assert fc1["local_out_features"] == INTERMEDIATE_SIZE // tp_width

    # op7 = fc2（down_proj）：按规约维切。
    fc2 = ops["7"]
    assert fc2["local_in_features"] == INTERMEDIATE_SIZE // tp_width
    assert fc2["local_out_features"] == HIDDEN_SIZE

    assert sidecar["version"] == 2
    assert sidecar["ir_num_operators"] == 8


def test_qkv_local_width_sums_actual_kv_shards_under_gqa(
    annotated_gqa_llama, tmp_path
) -> None:
    """分组查询注意力下，qkv 的本地宽度必须按 q/k/v 实际分片累加。

    早先的实现是拿 q_proj 的本地宽度乘三，这在 k/v 头数等于 q 头数时才成立。
    GQA 下 k/v 更窄，乘三会高估——这个 fixture 是 8 个 q 头、4 个 kv 头，
    乘三算出 48 而真实只有 32，高估一点五倍。llama2-7b 恰好是 32/32，
    掩盖了这个缺陷，但 `llama_strategy` 是接受 num_kv_heads != num_heads 的。
    """
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(
        annotated_gqa_llama, ir_path,
        tmp_path / "placed.ir", tmp_path / "sc.json",
    )

    head_dim = GQA_HIDDEN_SIZE // GQA_NUM_HEADS
    q_out = GQA_HIDDEN_SIZE // GQA_TP_WIDTH                       # 32
    kv_out = (GQA_NUM_KV_HEADS * head_dim) // GQA_TP_WIDTH        # 16

    qkv = sidecar["operators"]["0"]
    assert qkv["local_in_features"] == GQA_HIDDEN_SIZE
    assert qkv["local_out_features"] == q_out + 2 * kv_out        # 32 + 16 + 16
    # 明确记下：按 q 乘三会得到 48，不是这里的 64 // 2 == 32 之和。
    assert qkv["local_out_features"] != 3 * q_out


def test_layer_gemm_count_mismatch_raises(annotated_tiny_llama, tmp_path) -> None:
    ir_path = tmp_path / "bad.ir"
    ir = {
        "model_id": "bad-fixture",
        "num_layers": 1,
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "device_hint": "gpu",
             "input_shapes": [], "output_shapes": []},
        ],
        "dependencies": [],
        "subgraphs": [[0]],
    }
    ir_path.write_text(json.dumps(ir))

    with pytest.raises(ValueError, match="期望 4 个 GEMM"):
        export_placement_to_genesim(
            annotated_tiny_llama, ir_path, tmp_path / "out.ir", tmp_path / "out_sidecar.json"
        )
