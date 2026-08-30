"""genesim_bridge/placement_export.py 单测（本轮只对接 GEMM 类算子）。

用一个小随机 Llama（同 tests/test_partition.py 的 `_export_random_llama` 规格）
过 partition_graph + propagate_specs，构造一份手写的 GeneSim .ir 骨架（1 层、
4 个 GEMM，op_type/顺序对应 qkv/o_proj/gate_proj/down_proj），验证：

- 每层恰好 4 个 GEMM 的 device_hint 从 "gpu" 改写为 "pim"；
- 写回的 dpu_id 落在 [0, num_dpus)；
- 非 GEMM 算子（如 attention 三类）不被改写。
"""

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


def _write_fixture_ir(path: Path) -> None:
    """手写一份最小 GeneSim .ir：1 层，4 个 GEMM（顺序对应 qkv/proj/fc1/fc2）
    加 3 个 attention 算子（不应被改写），device_hint 全部先设成 GeneSim 的
    默认假设（qkv/proj/fc1/fc2 -> gpu，attention -> pim）。
    """
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
