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


# 每层的七个投影各自一个 GEMM，与当前 model_parser 的产出一致：先前的图骨架把
# q/k/v 合并成一个 GEMM、并用 gate_proj 顶替整个 fc1，现在逐个独立表示。
#
# op_id 的升序**故意不等于** _GEMM_WEIGHT_SUFFIXES 的声明顺序：导出侧必须按权重名
# 匹配，如果退回"按 subgraph 顺序 zip"，这里就会把 down_proj 的分片宽度套到
# q_proj 上，测试随即失败。顺序一致的 fixture 测不出这个区别。
_FIXTURE_GEMM_WEIGHTS = (
    (0, "down_proj.weight"),
    (4, "up_proj.weight"),
    (5, "gate_proj.weight"),
    (7, "o_proj.weight"),
    (8, "v_proj.weight"),
    (9, "k_proj.weight"),
    (10, "q_proj.weight"),
)


def _write_fixture_ir(path: Path) -> None:
    """写入每层七个 GEMM 加注意力算子的最小 GeneSim IR。

    GEMM 的身份由 `dependencies` 里的权重 `tensor_id` 给出，导出侧按名字匹配，
    不依赖 subgraph 里的出现顺序——所以这里故意把 op_id 打散排列。
    """
    gemm_ids = {op_id for op_id, _ in _FIXTURE_GEMM_WEIGHTS}
    operators = []
    for op_id in range(11):
        if op_id in gemm_ids:
            op_type, hint = "GEMM", "gpu"
        elif op_id == 1:
            op_type, hint = "GEMV_SCORE", "pim"
        elif op_id == 2:
            op_type, hint = "SOFTMAX", "pim"
        elif op_id == 3:
            op_type, hint = "GEMV_CONTEXT", "pim"
        else:
            op_type, hint = "GELU", "cpu"
        operators.append({
            "op_id": op_id, "op_type": op_type, "device_hint": hint,
            "input_shapes": [], "output_shapes": [],
        })

    dependencies = [
        {"src_op_id": -1, "dst_op_id": op_id, "tensor_id": f"layer.0.{suffix}"}
        for op_id, suffix in _FIXTURE_GEMM_WEIGHTS
    ]

    ir = {
        "model_id": "tiny-llama-fixture",
        "num_layers": 1,
        "num_heads": NUM_HEADS,
        "head_dim": HIDDEN_SIZE // NUM_HEADS,
        "hidden_size": HIDDEN_SIZE,
        "operators": operators,
        "dependencies": dependencies,
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

    for op_id, suffix in _FIXTURE_GEMM_WEIGHTS:
        assert ops_by_id[op_id]["device_hint"] == "pim", op_id
        entry = sidecar["operators"].get(str(op_id))
        assert entry is not None, op_id
        assert 0 <= entry["dpu_id"] < NUM_DPUS, entry
        # 身份由权重名确定，与 subgraph 里的出现顺序无关。
        assert entry["weight"] == f"layer.0.{suffix}"

    for op_id in (1, 2, 3, 6):
        assert ops_by_id[op_id]["device_hint"] == {
            1: "pim", 2: "pim", 3: "pim", 6: "cpu",
        }[op_id], op_id
        assert str(op_id) not in sidecar["operators"], op_id


def test_local_shard_widths_follow_each_projection(
    annotated_tiny_llama, tmp_path
) -> None:
    """每个投影的本地分片宽度都要如实写进 sidecar。

    IR 里七个投影各自是一个 GEMM，所以每个 GEMM 的宽度直接取自它自己的权重，
    不需要累加或折算。切分方向决定哪一侧变窄：按输出维切（q/k/v/gate/up）是
    出口变窄，按规约维切（o_proj/down）是入口变窄。
    """
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(
        annotated_tiny_llama, ir_path,
        tmp_path / "placed.ir", tmp_path / "sc.json",
    )

    tp = NUM_DPUS            # 该 fixture 是纯张量并行，段内宽度即 DPU 数
    ops = sidecar["operators"]
    H, I = HIDDEN_SIZE, INTERMEDIATE_SIZE

    # (op_id, 期望 in_features, 期望 out_features)
    # op_id 与投影的对应见 _FIXTURE_GEMM_WEIGHTS —— 那里的顺序是打散的。
    expected = [
        (0, I // tp, H),          # down_proj：按规约维切
        (4, H, I // tp),          # up_proj：按输出维切，不再被漏掉
        (5, H, I // tp),          # gate_proj：按输出维切
        (7, H // tp, H),          # o_proj：按规约维切
        (8, H, H // tp),          # v_proj：按输出维切
        (9, H, H // tp),          # k_proj：同上
        (10, H, H // tp),         # q_proj：同上
    ]
    for op_id, want_in, want_out in expected:
        entry = ops[str(op_id)]
        assert entry["local_in_features"] == want_in, (op_id, entry)
        assert entry["local_out_features"] == want_out, (op_id, entry)

    assert sidecar["version"] == 2
    assert sidecar["ir_num_operators"] == 11


def test_kv_projections_keep_their_own_width_under_gqa(
    annotated_gqa_llama, tmp_path
) -> None:
    """分组查询注意力下，k/v 的本地宽度比 q 窄，各自如实写出。

    这里曾经踩过一个坑：那时 IR 把 q/k/v 合并成一个 GEMM，导出侧只好拿 q_proj
    的宽度乘三来凑。GQA 下 k/v 的头数少于 q，乘三会高估——8 个 q 头、4 个 kv 头
    的模型，真实宽度之和是 32+16+16=64，乘三却算出 96。llama2-7b 恰好是 32/32
    掩盖了这个缺陷，但 `llama_strategy` 是接受 num_kv_heads != num_heads 的。

    现在每个投影独立成 GEMM，各取自己的 local_shape，这类折算不再存在。
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
    assert kv_out < q_out, "fixture 必须真的是 GQA，否则测不到这件事"

    ops = sidecar["operators"]
    # op_id 与投影的对应见 _FIXTURE_GEMM_WEIGHTS：q 是 op10，k 是 op9，v 是 op8。
    assert ops["10"]["local_out_features"] == q_out      # q_proj
    assert ops["9"]["local_out_features"] == kv_out      # k_proj，更窄
    assert ops["8"]["local_out_features"] == kv_out      # v_proj，更窄


def test_gemm_without_weight_dependency_raises(annotated_tiny_llama, tmp_path) -> None:
    """GEMM 在 dependencies 里找不到权重时必须报错，而不是静默跳过。

    身份靠权重 tensor_id 确定；缺了它就无法判断这个 GEMM 是哪个投影，继续下去
    只会把某个投影的分片宽度套到别的算子上。
    """
    ir_path = tmp_path / "bad.ir"
    ir = {
        "model_id": "bad-fixture",
        "num_layers": 1,
        "num_heads": NUM_HEADS,
        "head_dim": HIDDEN_SIZE // NUM_HEADS,
        "hidden_size": HIDDEN_SIZE,
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "device_hint": "gpu",
             "input_shapes": [], "output_shapes": []},
        ],
        "dependencies": [],          # 故意不给权重依赖
        "subgraphs": [[0]],
    }
    ir_path.write_text(json.dumps(ir))

    with pytest.raises(ValueError, match="找不到"):
        export_placement_to_genesim(
            annotated_tiny_llama, ir_path, tmp_path / "out.ir", tmp_path / "out_sidecar.json"
        )


def test_unknown_projection_weight_raises(annotated_tiny_llama, tmp_path) -> None:
    """权重名不在已知投影表里时必须报错，提示同步那张表。"""
    ir_path = tmp_path / "unknown.ir"
    ir = {
        "model_id": "unknown-fixture",
        "num_layers": 1,
        "num_heads": NUM_HEADS,
        "head_dim": HIDDEN_SIZE // NUM_HEADS,
        "hidden_size": HIDDEN_SIZE,
        "operators": [
            {"op_id": 0, "op_type": "GEMM", "device_hint": "gpu",
             "input_shapes": [], "output_shapes": []},
        ],
        "dependencies": [
            {"src_op_id": -1, "dst_op_id": 0,
             "tensor_id": "layer.0.some_new_proj.weight"},
        ],
        "subgraphs": [[0]],
    }
    ir_path.write_text(json.dumps(ir))

    with pytest.raises(ValueError, match="不在已知投影列表里"):
        export_placement_to_genesim(
            annotated_tiny_llama, ir_path, tmp_path / "out.ir", tmp_path / "out_sidecar.json"
        )
