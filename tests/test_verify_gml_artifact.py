"""验证产物自洽性检查器本身是对的。

产物无法在硬件上执行，所以这个检查器是数值层面唯一的守卫——它必须真的会失败。
每条检查都配一个「构造违规产物，断言被抓到」的用例，否则它只是装饰。

最有价值的一条是反量化比对：它能抓到「张量拿错」「转置」「错位」这类结构校验
完全看不见的错。判据的分辨力实测约 19 倍（正确 0.071 vs 拿错 1.37）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import gml_names as names
from contracts.gml_quant import WEIGHT_GROUP_SIZE
from quant.weights import quantize_weight
from scripts.gml_structure_check import field, parse_blocks
from scripts.verify_gml_artifact import (
    Report,
    check_attributes_survive_parsing,
    check_data_buffer_sizes,
    check_parses_with_networkx,
    check_luts,
    check_no_orphan_files,
    check_referenced_files_exist,
    check_scales_are_finite,
    check_weight_layout,
)

_SEQ_LEN = 16


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory) -> Path:
    """产出一份正常的 GML + .bin，供各项检查作为基线。"""
    from gml_bridge.export import export_graph, write_artifact, write_runtime_files
    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32000, hidden_size=64, intermediate_size=176,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=_SEQ_LEN, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        model, _SEQ_LEN, position_ids, dtype=torch.float32)

    out_dir = tmp_path_factory.mktemp("artifact")
    artifact = export_graph(graph)
    write_artifact(artifact, out_dir)
    write_runtime_files(artifact, out_dir, gm=graph)
    return out_dir


@pytest.fixture(scope="module")
def blocks(artifact_dir: Path) -> tuple[list[str], list[str]]:
    text = (artifact_dir / "relay2gml_graph.gml").read_text()
    return parse_blocks(text, "node"), parse_blocks(text, "edge")


def test_normal_artifact_passes_every_check(artifact_dir, blocks) -> None:
    """正常产物要全过——否则检查器过严，会误报。"""
    nodes, edges = blocks
    report = Report()

    referenced = check_referenced_files_exist(artifact_dir, nodes, report)
    check_no_orphan_files(artifact_dir, referenced, report)
    check_data_buffer_sizes(artifact_dir, nodes, edges, report)
    weights = check_weight_layout(artifact_dir, nodes, report)
    check_scales_are_finite(weights, report)
    check_luts(artifact_dir, nodes, report)

    assert report.failed == [], f"正常产物被误报: {report.failed}"
    assert referenced, "应当引用了缓冲区"


def test_missing_file_is_caught(artifact_dir, blocks, tmp_path) -> None:
    """GML 引用了但文件不存在——最危险的一种，对方解析时才炸。"""
    nodes, _ = blocks
    empty = tmp_path / "empty"
    empty.mkdir()

    report = Report()
    check_referenced_files_exist(empty, nodes, report)
    assert report.failed, "空目录应当被抓到"
    assert "缺" in report.failed[0]


def test_orphan_file_is_caught(artifact_dir, tmp_path) -> None:
    """写了盘但 GML 没引用——说明两侧命名已发散。"""
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "input_buffer_999.bin").write_bytes(b"\x00")

    report = Report()
    check_no_orphan_files(staged, set(), report)
    assert report.failed
    assert "多" in report.failed[0]


def test_wrong_buffer_size_is_caught(artifact_dir, blocks, tmp_path) -> None:
    """尺寸写错要被抓到——结构校验只看图，看不出文件大小不对。"""
    nodes, edges = blocks
    staged = tmp_path / "truncated"
    staged.mkdir()
    for path in artifact_dir.glob("*.bin"):
        # 每个文件都截成 1 字节，尺寸检查应当报一大片不符。
        (staged / path.name).write_bytes(b"\x00")

    report = Report()
    check_data_buffer_sizes(staged, nodes, edges, report)
    assert report.failed, "截断的缓冲区应当被抓到"
    assert "不符" in report.failed[0]


def test_out_of_range_int4_is_caught(tmp_path) -> None:
    """int4 越界要被抓到：值域必须在 [-8, 7]。"""
    from gml_bridge.writer import Node, write_gml

    staged = tmp_path / "bad_range"
    staged.mkdir()
    # 故意写出 int8 全量程的值。
    np.arange(-128, 128, dtype=np.int8).tofile(
        staged / names.weight_buffer(3))
    np.ones(256 // WEIGHT_GROUP_SIZE, dtype=np.float16).tofile(
        staged / names.weight_scale(3))

    text = write_gml(
        [Node(3, {"label": "w", "name": "w", "op_type": "Gemm",
                  "weight_buffer": names.weight_buffer(3),
                  "weight_sf": names.weight_scale(3)})],
        [], version="26.10.1")
    nodes = parse_blocks(text, "node")

    report = Report()
    check_weight_layout(staged, nodes, report)
    assert any("越界" in line for line in report.failed)


def test_wrong_group_size_is_caught(tmp_path) -> None:
    """scale 数量与 group_size 不符要被抓到——这会让 scale 与权重错位。"""
    from gml_bridge.writer import Node, write_gml

    staged = tmp_path / "bad_group"
    staged.mkdir()
    np.zeros(256, dtype=np.int8).tofile(staged / names.weight_buffer(3))
    # 应当是 256/128 = 2 个，故意写 4 个。
    np.ones(4, dtype=np.float16).tofile(staged / names.weight_scale(3))

    text = write_gml(
        [Node(3, {"label": "w", "name": "w", "op_type": "Gemm",
                  "weight_buffer": names.weight_buffer(3),
                  "weight_sf": names.weight_scale(3)})],
        [], version="26.10.1")

    report = Report()
    check_weight_layout(staged, parse_blocks(text, "node"), report)
    assert any("不符" in line for line in report.failed)


def test_nan_scale_is_caught(tmp_path) -> None:
    """nan 或 0 的 scale 要被抓到——前者是除零留下的，后者让反量化恒为零。"""
    staged = tmp_path / "nan_scale"
    staged.mkdir()
    weight_path = staged / names.weight_buffer(3)
    scale_path = staged / names.weight_scale(3)
    np.zeros(256, dtype=np.int8).tofile(weight_path)
    np.array([np.nan, 1.0], dtype=np.float16).tofile(scale_path)

    report = Report()
    check_scales_are_finite([("3", weight_path, scale_path)], report)
    assert report.failed
    assert "nan" in report.failed[0]


def test_dequantization_error_discriminates_wrong_tensors() -> None:
    """反量化误差判据必须能区分「对的张量」与「拿错/转置/错位」。

    这是整个 C 层验证的立足点：若判据对错误不敏感，那条检查就没有意义。
    实测分辨力约 19 倍。
    """
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((512, 128)) * 0.02).astype(np.float32)
    recovered = quantize_weight(weight).dequantize()

    flat = weight.ravel()
    peak = np.abs(flat).max()

    correct = float(np.abs(flat - recovered).max() / peak)
    other = (rng.standard_normal((512, 128)) * 0.02).astype(np.float32).ravel()
    wrong = float(np.abs(other - recovered).max() / peak)
    transposed = float(
        np.abs(np.ascontiguousarray(weight.T).ravel() - recovered).max() / peak)
    shifted = float(np.abs(np.roll(flat, 1) - recovered).max() / peak)

    # 正确匹配落在 int4 量级。
    assert correct < 0.15
    # 三种错误都要显著更大——至少 5 倍，实测约 19 倍。
    for label, error in (("拿错", wrong), ("转置", transposed),
                         ("错位", shifted)):
        assert error > correct * 5, f"{label}的误差 {error} 与正确 {correct} 区分不足"

def test_networkx_can_parse_our_output(artifact_dir) -> None:
    """用独立第三方解析器读一遍。

    GML 规范本身就是给 networkx 用的（节点的 `label` 字段注明「needed for
    Networkx」），所以读得通是格式合法的独立证据——解析器不是我们写的，
    不会因为我们理解错格式而一起错。

    实测三份 GML 都能读：ResNet50 实物 74/89、llama2 实物 200/331、我方 43/50。
    """
    report = Report()
    check_parses_with_networkx(artifact_dir / "relay2gml_graph.gml", report)
    assert report.failed == [], f"networkx 读不通: {report.failed}"


def test_networkx_check_catches_malformed_gml(tmp_path) -> None:
    """畸形 GML 必须被 networkx 抓到，否则这项检查是装饰。"""
    broken = tmp_path / "relay2gml_graph.gml"
    broken.write_text("graph [\n  directed 1\n  node [\n    id\n")

    report = Report()
    check_parses_with_networkx(broken, report)
    assert report.failed, "畸形 GML 应当被抓到"

def test_attributes_are_readable_by_networkx(artifact_dir) -> None:
    """独立解析器不只要读通语法，还要能取出正确的属性值。

    比「语法合法」强一层：括号配对不代表字段能被下游用上。这里验证两点——
    `op_type` 可读出，以及重复键 `residual_input_buffer` 被聚合成列表
    （数组展开写法是否合规）。
    """
    report = Report()
    check_attributes_survive_parsing(
        artifact_dir / "relay2gml_graph.gml", report)
    assert report.failed == [], f"属性读不出: {report.failed}"


def test_output_buffer_points_at_the_consumer(artifact_dir) -> None:
    """用独立解析器确认规则 2 那条反直觉约定真的落在产物里。

    一个节点的 `output_buffer` 应当是它下游节点的 `input_buffer`——缓冲区代表边，
    编号取读它的那个节点。这条最容易写反，所以用第三方解析器交叉确认一次。
    """
    networkx = pytest.importorskip("networkx")

    graph = networkx.read_gml(str(artifact_dir / "relay2gml_graph.gml"))
    by_node_id = {
        attrs["node_id"]: attrs for _, attrs in graph.nodes(data=True)
        if "node_id" in attrs
    }

    checked = 0
    for _, attrs in graph.nodes(data=True):
        produced = attrs.get("output_buffer")
        downstream = attrs.get("output0_node_id")
        if not isinstance(produced, str) or downstream is None:
            continue
        # phase 型节点（带 rtl_version 的 DQ）自命名 output_buffer_<self>，
        # 不走「按消费者编号」那条规则。实测 37 个自命名节点与 37 个
        # rtl_version 完全重合，所以这里跳过它们。
        if produced.startswith("output_buffer_"):
            continue
        # 流进消费者**权重通路**的那一路命名为 `weight_buffer_<消费者>`
        # （MatMul 的第二个 operand 不占数据槽），所以不会出现在下游的
        # input_buffer 里 —— 见计划 §11.5.6 那张归属表。
        if produced.startswith("weight_buffer_"):
            continue
        consumer = by_node_id.get(downstream)
        if consumer is None:
            continue
        # 扫到 32 槽为止，不能只查 0..2：逐头展开后 Concat 有 32 个输入
        # （每头一个），GML 的槽号上限就是 31。
        consumed = [consumer.get("input_buffer")]
        consumed += [
            consumer.get(f"input_buffer_{slot}") for slot in range(32)
        ]
        assert produced in [name for name in consumed if name], (
            f"节点输出 {produced} 不在下游 {downstream} 的输入里")
        checked += 1

    assert checked > 0, "应当有可核对的生产者-消费者对"


@pytest.fixture(scope="module")
def slot_layout_dir(tmp_path_factory) -> Path:
    """`hidden=4096` 的小模型产出的产物：走**编译期槽位**口径。

    槽位口径的开关是 `_looks_like_llama7b`（判据只认「图里出现 4096」），
    所以 4096 宽、一层、词表 256 的小模型就能把这条路径跑出来——秒级完成，
    不必加载 7B 权重。落盘尺寸取 `CompileSlots`（seq=1024、nh=32、hd=128），
    不是导出图的 seq_len=16。
    """
    from gml_bridge.export import (
        export_graph,
        write_artifact,
        write_runtime_files,
    )
    from runtime.compile import export_annotated_graph

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256, hidden_size=4096, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=32,
            max_position_embeddings=_SEQ_LEN, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        model, _SEQ_LEN, position_ids, dtype=torch.float32)

    out_dir = tmp_path_factory.mktemp("slot-layout")
    artifact = export_graph(graph)
    write_artifact(artifact, out_dir)
    write_runtime_files(artifact, out_dir, gm=graph)
    return out_dir


@pytest.fixture(scope="module")
def slot_blocks(slot_layout_dir: Path) -> tuple[list[str], list[str]]:
    text = (slot_layout_dir / "relay2gml_graph.gml").read_text()
    return parse_blocks(text, "node"), parse_blocks(text, "edge")


def test_slot_layout_artifact_keeps_declarations_and_files_in_step(
    slot_layout_dir: Path, slot_blocks,
) -> None:
    """槽位口径下，声明与落盘必须同源——这一档曾经有 73 处不符。

    DQ 的输出、掩码、KV 新值这三族落盘的是**一个 token** 的量（4096 / 2048 /
    4096 字节），而它们的边 dims 和 DQ 的 `original_shape` 一度还是导出图的
    16 个 token，于是校验器报「文件 4096B、声明 65536B（65536 × int8）」。
    三族的尺寸规则在写盘侧（`export.write_runtime_files` 的槽位分支与
    `_dq_specs`），这里钉住「写盘那一份口径 == GML 里声明的那一份」。
    """
    nodes, edges = slot_blocks
    report = Report()
    check_data_buffer_sizes(slot_layout_dir, nodes, edges, report)
    assert report.failed == [], report.failed

    # DQ 自己的两个形状字段也要与它落盘的两份一致：output_buffer 装 numel 个
    # int8，output_sf 装 numel/group_size 个 fp16（参考 node 12 的 4096B/64B）。
    checked = 0
    for block in nodes:
        original = field(block, "original_shape")
        # Q 路的 RoPE-DQ 不带形状字段（与参考一致），跳过。
        if field(block, "op_type") != "DynamicScaling" or original is None:
            continue
        numel = 1
        for part in original.strip("[]").split(","):
            numel *= int(part)
        group_size = int(
            field(block, "output_shape_by_group").strip("[]").rsplit(",", 1)[-1])
        checked += 1
        output = slot_layout_dir / field(block, "output_buffer")
        scale = slot_layout_dir / field(block, "output_sf")
        assert output.stat().st_size == numel, f"节点 {block} 的 output_buffer"
        assert scale.stat().st_size == numel // group_size * 2, (
            f"节点 {field(block, 'node_id')} 的 output_sf")
    assert checked
