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


# 每层的七个投影各自一个 GEMM，与当前 model_parser 的产出一致。每个 GEMM 的身份
# 由 IR 自带的 `semantic_role` 给出。
#
# op_id 的升序**故意不等于** _ROLE_TO_WEIGHT_PATTERN 的声明顺序：导出侧必须按
# 语义标签匹配，如果退回"按 subgraph 顺序 zip"，这里就会把 down_proj 的分片宽度
# 套到 q_proj 上，测试随即失败。顺序一致的 fixture 测不出这个区别。
_FIXTURE_GEMM_ROLES = (
    (0, "down_proj"),
    (4, "up_proj"),
    (5, "gate_proj"),
    (7, "o_proj"),
    (8, "v_proj"),
    (9, "k_proj"),
    (10, "q_proj"),
)


def _write_fixture_ir(path: Path, *, roles: dict[int, str] | None = None) -> None:
    """写入每层七个 GEMM 加注意力算子的最小 GeneSim IR。

    GEMM 的身份由 `semantic_role` 给出，导出侧按它匹配，不依赖 subgraph 里的出现
    顺序——所以这里故意把 op_id 打散排列。`roles` 可覆盖默认标签，用于构造标签
    缺失或未知的负例。
    """
    role_by_id = dict(_FIXTURE_GEMM_ROLES) if roles is None else roles
    gemm_ids = set(role_by_id)
    # 每个投影的全局形状 (in_features, out_features)，与真实 IR 的口径一致：
    # 导出侧要读它的前导符号维 Tq 来拼本地分片形状。
    global_features = {
        "q_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "k_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "v_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "o_proj": (HIDDEN_SIZE, HIDDEN_SIZE),
        "gate_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "up_proj": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        "down_proj": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
    }
    operators = []
    for op_id in range(11):
        role = ""
        in_shapes: list = []
        out_shapes: list = []
        if op_id in gemm_ids:
            op_type, hint = "GEMM", "gpu"
            role = role_by_id[op_id]
            # 负例里 role 可能是空串或未知名字，此时退回 hidden_size 的方阵——
            # 那些用例在读形状之前就该报错，形状取值不影响结论。
            in_features, out_features = global_features.get(
                role, (HIDDEN_SIZE, HIDDEN_SIZE)
            )
            in_shapes = [["Tq", in_features]]
            out_shapes = [["Tq", out_features]]
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
            "input_shapes": in_shapes, "output_shapes": out_shapes,
            "semantic_role": role,
        })

    # 权重 tensor_id 故意与 semantic_role **不一致**（循环错位一位）。
    #
    # 这一点是必要的：导出侧不再解析 tensor_id，只认 semantic_role。如果让两者
    # 一致，退回按 tensor_id 匹配也能得到同样结果，测试就分辨不出机制换没换——
    # 实测过，那样五个用例全过。错位之后按 tensor_id 匹配会把 down_proj 的分片
    # 宽度套到别的投影上，local_shard_widths 那条断言立刻失败。
    #
    # 真实 IR 里两者本来是一致的，这里的错位纯粹是为了锚定「只读 semantic_role」。
    ordered = sorted(role_by_id.items())
    dependencies = [
        {
            "src_op_id": -1,
            "dst_op_id": op_id,
            "tensor_id": f"layer.0.{ordered[(index + 1) % len(ordered)][1]}.weight",
        }
        for index, (op_id, _role) in enumerate(ordered)
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

    for op_id, role in _FIXTURE_GEMM_ROLES:
        assert ops_by_id[op_id]["device_hint"] == "pim", op_id
        entry = sidecar["operators"].get(str(op_id))
        assert entry is not None, op_id
        assert entry["shards"], entry
        for shard in entry["shards"]:
            assert 0 <= shard["dpu_id"] < NUM_DPUS, entry
        # 身份由 semantic_role 确定，与 subgraph 里的出现顺序无关。
        assert entry["semantic_role"] == role
        # 记录的是实际匹配到的 fx 节点，据此可核对 role 落在了正确的权重上。
        assert entry["weight"].endswith(f"{role}.weight")
        assert f"layers.0." in entry["weight"]

    for op_id in (1, 2, 3, 6):
        assert ops_by_id[op_id]["device_hint"] == {
            1: "pim", 2: "pim", 3: "pim", 6: "cpu",
        }[op_id], op_id
        assert str(op_id) not in sidecar["operators"], op_id


def test_bpath_entries_keep_the_sidecar_shape() -> None:
    """B 路条目并进 sidecar 后，每条都必须带 `shards`。

    `export_pp_placement.py` 逐条遍历 `entry["shards"]` 统计每台 DPU 的分片数。
    B 路回填（`_attach_bpath_pimir`）给非 GEMM 节点新建的条目若缺 `shards`，
    那次遍历直接抛 KeyError，全链路导出中断。
    """
    from genesim_bridge.placement_export import _attach_bpath_pimir

    operators = {
        0: {"op_id": 0, "op_type": "GEMM", "input_shapes": [["Tq", 64]],
            "output_shapes": [["Tq", 64]]},
        1: {"op_id": 1, "op_type": "SOFTMAX", "input_shapes": [["Tq", 64]],
            "output_shapes": [["Tq", 64]]},
        2: {"op_id": 2, "op_type": "ROPE", "input_shapes": [["Tq", 64]],
            "output_shapes": [["Tq", 64]]},
    }
    sidecar = {"operators": {
        "0": {"op_type": "GEMM", "shards": [{"dpu_id": 0}]},
    }}

    _attach_bpath_pimir(operators, sidecar)

    # 非 GEMM 的算子级节点也要进 sidecar，且带上 B 路产物。
    for op_id, op_type in (("1", "SOFTMAX"), ("2", "ROPE")):
        entry = sidecar["operators"].get(op_id)
        assert entry is not None, f"{op_type} 没进 sidecar"
        assert entry["op_type"] == op_type
        assert "pimir_path" in entry, f"{op_type} 没有 pimir_path"
        # 关键：缺了它，消费方 `entry["shards"]` 会抛 KeyError。
        assert "shards" in entry, f"{op_type} 的条目没有 shards"

    # 消费方的遍历口径：每条都要能取到 shards，GEMM 的分片不能被冲掉。
    for entry in sidecar["operators"].values():
        assert isinstance(entry["shards"], list), entry["op_type"]
    assert sidecar["operators"]["0"]["shards"] == [{"dpu_id": 0}]


def test_bpath_kv_cache_declares_a_cache_big_enough_for_one_write() -> None:
    """KV 缓存的容量要盖得住一次写入，否则整条 B 路导出被 verifier 拦下。

    `pim.kv_cache` 的 memdesc 必须装得下一次写的元素数。写成末维（`[Tq, 4096]`
    取 4096）看着像「一行」，但一次搬的是整块 128x4096，verifier 直接报
    `cache holds 4096 elements, fewer than the 524288 being moved`，而这条路径
    是直接发 IR 文本给 `triton-opt`，绕开了 `driver` 里同名的守卫。

    形状取自真实 decode 块的 MEM_COPY 节点（`[Tq, 4096]`，Tq 代入 128）。
    """
    import re

    from genesim_bridge.placement_export import _bpath_pimir

    text = Path(_bpath_pimir("kv_cache", [("Tq", 4096)], [("Tq", 4096)])).read_text()
    assert "pim.kv_cache" in text, "B 路产物里没有 kv_cache"
    match = re.search(r"!pim\.memdesc<(\d+)xi8", text)
    assert match, f"没找到缓存 memdesc：\n{text}"
    assert int(match.group(1)) >= 128 * 4096, (
        f"缓存只声明了 {match.group(1)} 个元素，装不下一次写的 {128 * 4096} 个"
    )


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
    # op_id 与投影的对应见 _FIXTURE_GEMM_ROLES —— 那里的顺序是打散的。
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
        # 该 fixture 是纯张量并行（num_stages=1），每个 shard 的本地宽度一致，
        # 取哪个都行。
        for shard in entry["shards"]:
            assert shard["local_in_features"] == want_in, (op_id, entry)
            assert shard["local_out_features"] == want_out, (op_id, entry)

    assert sidecar["version"] == 3
    assert sidecar["ir_num_operators"] == 11


def test_requires_pimir_declares_whether_opcompiler_products_are_carried(
    annotated_tiny_llama, tmp_path
) -> None:
    """sidecar 要自己声明带没带算子编译产物。

    GeneSim 据此自动进入严格模式（pim mlir 读不到就报错），不再依赖配置文件写对
    `require_compiler_pimir`——tp4pp2 的配置就漏写过那一行，后果是实测分块不进
    代价链、仿真照样跑完但数字悄悄偏掉。

    这里只验纯放置导出（不带产物）声明为 false；带产物那条路径要真编 FlagTree，
    由 scripts/run_full_pipeline.py 覆盖。
    """
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(
        annotated_tiny_llama, ir_path,
        tmp_path / "placed.ir", tmp_path / "sc.json",
    )

    assert sidecar["requires_pimir"] is False
    # 没带产物时也不该出现这两个字段，否则消费侧会去找不存在的文件。
    for entry in sidecar["operators"].values():
        assert "pimir_path" not in entry
        assert "kernel_tile_n" not in entry


def test_local_shapes_written_into_ir_without_touching_global(
    annotated_tiny_llama, tmp_path
) -> None:
    """本地分片形状写进 IR 的新字段，原全局形状一个都不许改。

    这是切分影响仿真时间的入口：GeneSim 的 compile_gemm 和 _execute_runtime 优先
    读 local_*_shapes。同时原字段必须保持模型级全局语义，这样同一份 IR 既读得出
    模型结构、也读得出实际执行规模。
    """
    ir_path = tmp_path / "base.ir"
    out_ir_path = tmp_path / "placed.ir"
    _write_fixture_ir(ir_path)

    before = {op["op_id"]: op for op in json.loads(ir_path.read_text())["operators"]}
    export_placement_to_genesim(
        annotated_tiny_llama, ir_path, out_ir_path, tmp_path / "sc.json"
    )
    after = {op["op_id"]: op for op in json.loads(out_ir_path.read_text())["operators"]}

    tp = NUM_DPUS
    H, I = HIDDEN_SIZE, INTERMEDIATE_SIZE
    expected = [
        (0, I // tp, H),          # down_proj
        (4, H, I // tp),          # up_proj
        (5, H, I // tp),          # gate_proj
        (7, H // tp, H),          # o_proj
        (8, H, H // tp),          # v_proj
        (9, H, H // tp),          # k_proj
        (10, H, H // tp),         # q_proj
    ]
    for op_id, want_in, want_out in expected:
        op = after[op_id]
        # 前导维保持 IR 原本的符号维，不被解析成数字。
        assert op["local_input_shapes"] == [["Tq", want_in]], (op_id, op)
        assert op["local_output_shapes"] == [["Tq", want_out]], (op_id, op)
        # 全局形状原样保留。
        assert op["input_shapes"] == before[op_id]["input_shapes"], op_id
        assert op["output_shapes"] == before[op_id]["output_shapes"], op_id

    # 非 GEMM 算子不该被写上本地形状。
    for op_id in (1, 2, 3, 6):
        assert not after[op_id].get("local_input_shapes"), op_id
        assert not after[op_id].get("local_output_shapes"), op_id


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
    # op_id 与投影的对应见 _FIXTURE_GEMM_ROLES：q 是 op10，k 是 op9，v 是 op8。
    assert ops["10"]["shards"][0]["local_out_features"] == q_out      # q_proj
    assert ops["9"]["shards"][0]["local_out_features"] == kv_out      # k_proj，更窄
    assert ops["8"]["shards"][0]["local_out_features"] == kv_out      # v_proj，更窄


def test_tp_shards_cover_every_participating_dpu(annotated_gqa_llama, tmp_path) -> None:
    """TP 宽度 > 1 时，sidecar 必须记下组内每一台 DPU，不能只留一个代表。

    之前这里只取 `min(spec.shard_map)`，GQA_TP_WIDTH=2 的组里另一台 DPU 会被
    直接丢弃。这里用同一个 fixture 核对 `shards` 列表覆盖了两台不同的 DPU，
    并且列切（q/k/v/gate/up）标 `output_channel`、行切（o_proj/down_proj）标
    `input_channel`——这两个标签之后决定要不要插入 all_reduce。
    """
    ir_path = tmp_path / "base.ir"
    _write_fixture_ir(ir_path)

    sidecar = export_placement_to_genesim(
        annotated_gqa_llama, ir_path,
        tmp_path / "placed.ir", tmp_path / "sc.json",
    )

    ops = sidecar["operators"]
    for op_id, role in _FIXTURE_GEMM_ROLES:
        entry = ops[str(op_id)]
        assert len(entry["shards"]) == GQA_TP_WIDTH, (op_id, entry)
        dpu_ids = {shard["dpu_id"] for shard in entry["shards"]}
        assert len(dpu_ids) == GQA_TP_WIDTH, (op_id, entry)

        want_axis = "input_channel" if role in ("o_proj", "down_proj") else "output_channel"
        assert entry["shard_axis"] == want_axis, (op_id, role, entry)


def test_gemm_without_semantic_role_raises(annotated_tiny_llama, tmp_path) -> None:
    """GEMM 缺 semantic_role 时必须报错，而不是静默跳过。

    身份靠语义标签确定；缺了它就无法判断这个 GEMM 是哪个投影，继续下去只会把某个
    投影的分片宽度套到别的算子上。用旧版 model_parser 生成的 IR 会命中这一条。
    """
    ir_path = tmp_path / "bad.ir"
    # 七个投影里去掉 down_proj 的标签，其余保持正常——只有一个算子缺标签也要拦下。
    roles = {op_id: role for op_id, role in _FIXTURE_GEMM_ROLES}
    roles[0] = ""
    _write_fixture_ir(ir_path, roles=roles)

    with pytest.raises(ValueError, match="没有 semantic_role"):
        export_placement_to_genesim(
            annotated_tiny_llama, ir_path, tmp_path / "out.ir", tmp_path / "out_sidecar.json"
        )


def test_unknown_semantic_role_raises(annotated_tiny_llama, tmp_path) -> None:
    """semantic_role 不在已知投影表里时必须报错，提示同步那张表。

    上游给 model_parser 新增一种投影时会命中这一条，而不是等到取 fx 节点时才撞上
    「应恰好匹配 1 个」——那个报错指向的位置是错的。
    """
    ir_path = tmp_path / "unknown.ir"
    roles = {op_id: role for op_id, role in _FIXTURE_GEMM_ROLES}
    roles[0] = "some_new_proj"
    _write_fixture_ir(ir_path, roles=roles)

    with pytest.raises(ValueError, match="不在已知投影列表里"):
        export_placement_to_genesim(
            annotated_tiny_llama, ir_path, tmp_path / "out.ir", tmp_path / "out_sidecar.json"
        )


# --- 全链路 sidecar 校验的分流口径 -------------------------------------
#
# sidecar 里有两类条目，契约不同：
#
#   GEMM（A 路）  按图切分归属到各台 DPU，有 shards / semantic_role /
#                 kernel_tile_n，pim mlir 是分块 GEMM
#   算子级（B 路） 不做张量并行切分，shards 是空列表，没有投影身份，pim mlir
#                 是整算子级的相位链或单相 op
#
# `run_full_pipeline.py` 的校验若把 GEMM 那套字段要求套到全部条目，B 路的
# 6626 条会全部判失败，全链路验收命令跑不通——而它们本身是完好的。

def test_sidecar_checks_split_by_entry_class() -> None:
    """GEMM 条目查五项，B 路条目查自己那套，都要通过。"""
    from scripts.run_full_pipeline import check_sidecar_entries

    ops = {
        "0": {
            "op_type": "GEMM", "semantic_role": "q_proj",
            "shards": [{"dpu_id": 0, "local_in_features": 4096,
                        "local_out_features": 2048}],
            "kernel_tile_n": 512, "pimir_path": "/tmp/a.mlir",
        },
        "1": {
            "op_type": "SOFTMAX", "device_hint": "pim", "shards": [],
            "pimir_path": "/tmp/b.mlir",
        },
    }
    # 不抛错即通过，并报出两类各有几条。
    gemm_count, bpath_count = check_sidecar_entries(ops)
    assert (gemm_count, bpath_count) == (1, 1)


def test_gemm_entry_still_needs_all_of_its_fields() -> None:
    """分流不是放松：GEMM 条目缺 kernel_tile_n 仍要失败。"""
    from scripts.run_full_pipeline import StepFailed, check_sidecar_entries

    ops = {
        "0": {
            "op_type": "GEMM", "semantic_role": "q_proj",
            "shards": [{"dpu_id": 0, "local_in_features": 4096,
                        "local_out_features": 2048}],
            "pimir_path": "/tmp/a.mlir",
        },
    }
    with pytest.raises(StepFailed, match="kernel_tile_n"):
        check_sidecar_entries(ops)


def test_bpath_entry_without_pimir_path_fails() -> None:
    """B 路条目的判据是 pim mlir 落盘路径，缺了要失败。

    缺路径意味着这个算子的原语没进 GeneSim，仿真会静默退回手写模板——正是
    全链路脚本要防的那种静默退化。
    """
    from scripts.run_full_pipeline import StepFailed, check_sidecar_entries

    ops = {"1": {"op_type": "SOFTMAX", "device_hint": "pim", "shards": []}}
    with pytest.raises(StepFailed, match="pimir_path"):
        check_sidecar_entries(ops)


def test_bpath_entry_with_unknown_op_type_fails() -> None:
    """B 路条目的 op_type 必须是 mnemonic 单表认得的名字。

    认不出来就说明上游新增了算子而映射表没跟上，此时它在 GeneSim 侧没有原语
    落点，必须报错而不是当成一条普通条目放过。
    """
    from scripts.run_full_pipeline import StepFailed, check_sidecar_entries

    ops = {
        "1": {"op_type": "BRAND_NEW_OP", "device_hint": "pim", "shards": [],
              "pimir_path": "/tmp/b.mlir"},
    }
    with pytest.raises(StepFailed, match="op_type"):
        check_sidecar_entries(ops)


def test_sidecar_counts_report_the_two_classes_apart() -> None:
    """统计口径要把 GEMM 与算子级条目分开数。

    `export_pp_placement.py` 打印「放置的 GEMM 算子数」时若取 sidecar 条目总数，
    B 路的算子级节点会被算成 GEMM——llama2-7B 的 tp2_pp4 下是 224 个 GEMM 被报成
    6850 个，数字看着对、含义已经错了。
    """
    from genesim_bridge.placement_export import count_sidecar_classes

    sidecar = {"operators": {
        "0": {"op_type": "GEMM", "shards": [{"dpu_id": 0}, {"dpu_id": 1}]},
        "1": {"op_type": "GEMM", "shards": [{"dpu_id": 0}]},
        "2": {"op_type": "SOFTMAX", "shards": []},
        "3": {"op_type": "ROPE", "shards": []},
    }}

    gemm_count, bpath_count, by_dpu = count_sidecar_classes(sidecar)

    assert (gemm_count, bpath_count) == (2, 2)
    # 分片数按每台参与的 DPU 各记一次：dpu0 拿到 op0 与 op1，dpu1 只有 op0。
    assert by_dpu == {0: 2, 1: 1}


def test_bpath_compile_failure_is_reported_not_swallowed() -> None:
    """编不出 pim mlir 的算子级节点必须报错，不能静默跳过。

    静默跳过的后果是这条条目没有 `pimir_path`，仿真侧退回手写模板，而现场看不到
    是谁、为什么编不出——全链路校验只会报一句「缺 pimir_path」，信息量不足以定位。
    契约不满足就直接抛，让原因立刻暴露。
    """
    import genesim_bridge.placement_export as pe

    operators = {
        7: {"op_id": 7, "op_type": "SOFTMAX", "input_shapes": [[1, 64]],
            "output_shapes": [[1, 64]]},
    }
    sidecar = {"operators": {}}

    def boom(mnemonic, input_shapes, output_shapes):
        raise RuntimeError("triton-opt 挂了")

    original = pe._bpath_pimir
    pe._bpath_pimir = boom
    try:
        with pytest.raises(RuntimeError, match="op7"):
            pe._attach_bpath_pimir(operators, sidecar)
    finally:
        pe._bpath_pimir = original


def test_bpath_matmul_carries_stationarity_and_non_degenerate_shape(tmp_path) -> None:
    """GEMV_SCORE/GEMV_CONTEXT 走 B 路 matmul 时，IR 必须带 stationarity 标记，
    符号维也不能退化成字面 1。

    这两类 op_type 映射到 matmul mnemonic，但不属于 `_BPATH_MNEMONIC` 排除的
    GEMM，所以真实会走到 `_bpath_pimir` 的 matmul builder。那条 builder 若直接
    调 `matmul_kernel` 而不走 `_attention_matmul_body`，产出的 IR 就缺
    `stationarity = #pim.stationarity<kv>`——搬运量算得一样，但 KV 缓存被说成
    模型权值，校验器要求三者一致时对不上。符号维折成 1 同理：trace 外层循环
    次数按真实序列长度绑定，IR 里写死 1 会让搬运量比真实 prefill 小几个数量级。
    """
    from genesim_bridge.placement_export import _bpath_pimir

    path = _bpath_pimir(
        "matmul",
        [["Tq", 64], [64, "Tp+Tq"]],
        [["Tq", "Tp+Tq"]],
    )
    text = Path(path).read_text()
    assert "stationarity = #pim.stationarity<kv>" in text, text
    # 符号维不能退化成字面 1（`<1x` 是 tensor 形状里"第一维是 1"的写法）。
    assert "<1x" not in text, text
