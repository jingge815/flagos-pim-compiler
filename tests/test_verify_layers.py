"""新增校验层的判据：每条检查都要**真的能抓到**它针对的那类错误。

只测「正确的产物能通过」是不够的——那样一条永远返回 True 的检查也能过。
所以每条检查都配一个反例：故意写坏一处，确认它失败。
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.verify_gml_artifact as verify
from contracts.gml_names import port_node_id_key, residual_buffer_key
from genesim_bridge.paths import gml_llama2_reference_dir
from gml_bridge.phase_data import dynamic_scaling, pack_fp16_in_high_half, softmax
from gml_bridge.runtime_files import (
    WrittenFiles,
    write_dq_phases,
    write_softmax_phases,
)
from gml_bridge.writer import Edge, Node, write_gml
from scripts.gml_structure_check import parse_blocks


def _artifact(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    """造一个最小但结构完整的产物：一个 DQ 节点接一个 Softmax 节点。"""
    reference = gml_llama2_reference_dir()
    files = WrittenFiles(tmp_path)

    source = np.fromfile(
        reference / "input_buffer_phase_0_12.bin", dtype=np.float16)
    write_dq_phases(files, 12, dynamic_scaling(source, group_size=128))

    scores = np.fromfile(
        reference / "input_buffer_phase_0_18.bin", dtype=np.float16)
    write_softmax_phases(files, 18, softmax(scores))

    nodes = [
        Node(12, {
            "label": "dq_12", "name": "dq_12", "op_type": "DynamicScaling",
            "input_count": 1,
            residual_buffer_key("output", 0): 18,
            port_node_id_key("output", 0): 18,
        }),
        Node(18, {
            "label": "sm_18", "name": "sm_18", "op_type": "Softmax",
            "input_count": 1,
            residual_buffer_key("input", 0): 12,
            port_node_id_key("input", 0): 12,
        }),
    ]
    text = write_gml(nodes, [Edge(12, 18, "1x1x1x1024")], version="26.2.1")
    (tmp_path / "relay2gml_graph.gml").write_text(text)
    return tmp_path, parse_blocks(text, "node"), parse_blocks(text, "edge")


def _failed(report: verify.Report) -> list[str]:
    return report.failed


# ---------------------------------------------------------------------------
# 正例：我方生成的产物应当通过全部新增检查
# ---------------------------------------------------------------------------


def test_our_artifact_passes_every_new_check(tmp_path: Path) -> None:
    """我方按公式生成的产物要过全部新增层。

    这条同时是个集成测试：writer + runtime_files + phase_data 三者产出的东西
    必须彼此一致，任一处不匹配都会在这里暴露。
    """
    out_dir, nodes, edges = _artifact(tmp_path)
    report = verify.Report()

    verify.check_identity_fields(nodes, report)
    verify.check_connection_records_agree(nodes, edges, report)
    verify.check_residual_matches_port_fields(nodes, report)
    verify.check_dq_phase_math(out_dir, nodes, report)
    verify.check_dq_phase_constants(out_dir, nodes, report)
    verify.check_softmax_reduction_encodings(out_dir, nodes, report)
    verify.check_softmax_is_normalised(out_dir, nodes, report)

    assert not _failed(report), _failed(report)


def test_reference_passes_the_structural_layer() -> None:
    """参考产物应当过第一层——它是这层检查的基线。"""
    reference = gml_llama2_reference_dir()
    text = (reference / "relay2gml_graph.gml").read_text()
    nodes, edges = parse_blocks(text, "node"), parse_blocks(text, "edge")

    report = verify.Report()
    verify.check_identity_fields(nodes, report)
    verify.check_connection_records_agree(nodes, edges, report)
    verify.check_input_count_identity(nodes, edges, report)
    verify.check_residual_matches_port_fields(nodes, report)

    assert not _failed(report), _failed(report)


def test_reference_passes_the_phase_math_layer() -> None:
    """参考产物的 phase 数值应当过第四层——公式就是从它反推的。"""
    reference = gml_llama2_reference_dir()
    nodes = parse_blocks(
        (reference / "relay2gml_graph.gml").read_text(), "node")

    report = verify.Report()
    verify.check_dq_phase_math(reference, nodes, report)
    verify.check_dq_phase_constants(reference, nodes, report)
    verify.check_softmax_reduction_encodings(reference, nodes, report)
    verify.check_softmax_is_normalised(reference, nodes, report)
    verify.check_zero_points_are_zero(reference, report)

    assert not _failed(report), _failed(report)


# ---------------------------------------------------------------------------
# 反例：每条检查都要能抓到它针对的错误
# ---------------------------------------------------------------------------


def test_identity_check_catches_mismatched_id() -> None:
    """`id != node_id` 必须被抓到。"""
    text = write_gml(
        [Node(7, {"label": "a", "name": "a"})], [], version="26.2.1")
    broken = text.replace("    id 7", "    id 8")

    report = verify.Report()
    verify.check_identity_fields(parse_blocks(broken, "node"), report)
    assert any("id ≡ node_id" in line for line in _failed(report))


def test_identity_check_catches_mismatched_label() -> None:
    text = write_gml(
        [Node(7, {"label": "a", "name": "b"})], [], version="26.2.1")

    report = verify.Report()
    verify.check_identity_fields(parse_blocks(text, "node"), report)
    assert any("name ≡ label" in line for line in _failed(report))


def test_connection_check_catches_a_missing_port_field() -> None:
    """边存在但 `outputN_node_id` 漏写——三份连接信息不同步。"""
    nodes = [Node(1, {"label": "a", "name": "a"}), Node(2, {"label": "b", "name": "b"})]
    text = write_gml(nodes, [Edge(1, 2, "1x1x1x4")], version="26.2.1")

    report = verify.Report()
    verify.check_connection_records_agree(
        parse_blocks(text, "node"), parse_blocks(text, "edge"), report)
    assert any("outputN" in line for line in _failed(report))


def test_input_count_check_catches_a_wrong_count() -> None:
    """`input_count` 与边数不符要被抓到。"""
    nodes = [Node(1, {"label": "a", "name": "a", "input_count": 5})]
    text = write_gml(nodes, [Edge(1, 2, "1x1x1x4")], version="26.2.1")

    report = verify.Report()
    verify.check_input_count_identity(
        parse_blocks(text, "node"), parse_blocks(text, "edge"), report)
    assert any("input_count" in line for line in _failed(report))


def test_residual_check_catches_the_missing_underscore(tmp_path: Path) -> None:
    """端口 >= 10 时漏掉那个下划线要被抓到——这正是键名 bug 的复现点。

    注意 fixture 的构造方式：`residual_*_buffer` 要重复出现多次，而 dict 的键
    唯一，所以**必须把同名键的值聚成 list**（序列化器会展开成重复键）。
    直接对每个端口赋一次值会让 10 个端口塌成 1 个——这是生成器侧同样要注意的坑。
    """
    fields: dict[str, object] = {"label": "a", "name": "a"}
    plain, suffixed = [], []
    for port in range(12):
        (plain if port < 10 else suffixed).append(100 + port)
        fields[port_node_id_key("output", port)] = 100 + port
    fields[residual_buffer_key("output", 0)] = plain
    fields[residual_buffer_key("output", 10)] = suffixed
    text = write_gml([Node(1, fields)], [], version="26.2.1")

    report = verify.Report()
    verify.check_residual_matches_port_fields(parse_blocks(text, "node"), report)
    assert not _failed(report), "正确的键名应当通过"

    # 把带下划线的键改回不带下划线——模拟没复现那个 bug。
    broken = text.replace("residual_output_buffer_ ", "residual_output_buffer ")
    report = verify.Report()
    verify.check_residual_matches_port_fields(parse_blocks(broken, "node"), report)
    assert _failed(report), "漏掉键名 bug 应当被抓到"


def test_dq_math_check_catches_a_wrong_phase1_scale(tmp_path: Path) -> None:
    """把 p1 写成 p0/128（而非 /256）要被抓到。"""
    out_dir, nodes, _ = _artifact(tmp_path)

    phase0 = np.fromfile(
        out_dir / "output_buffer_phase_0_12.bin", dtype=np.float16)
    (phase0.astype(np.float32) / 128).astype(np.float16).tofile(
        out_dir / "output_buffer_phase_1_12.bin")

    report = verify.Report()
    verify.check_dq_phase_math(out_dir, nodes, report)
    assert any("p1 != p0/256" in line for line in _failed(report))


def test_dq_math_check_catches_a_wrong_absmax(tmp_path: Path) -> None:
    """把 p0 写成 absmax（漏了 ×2）要被抓到。"""
    out_dir, nodes, _ = _artifact(tmp_path)

    source = np.fromfile(
        out_dir / "input_buffer_phase_0_12.bin", dtype=np.float16)
    absmax = np.abs(source.astype(np.float32).reshape(-1, 128)).max(axis=1)
    absmax.astype(np.float16).tofile(out_dir / "output_buffer_phase_0_12.bin")

    report = verify.Report()
    verify.check_dq_phase_math(out_dir, nodes, report)
    assert any("p0 != 2*absmax" in line for line in _failed(report))


def test_dq_math_check_catches_a_zeroed_phase0_bias(tmp_path: Path) -> None:
    """把 phase0 的 bias 写成 0（而非 2^-63）要被抓到。"""
    out_dir, nodes, _ = _artifact(tmp_path)
    np.zeros(1, dtype=np.float32).tofile(out_dir / "Bias_buffer_phase_0_12.bin")

    report = verify.Report()
    verify.check_dq_phase_math(out_dir, nodes, report)
    assert any("2^-63" in line for line in _failed(report))


def test_dq_math_check_catches_a_wrong_shift(tmp_path: Path) -> None:
    """Kantor 右移量写成 0（而非 -8）要被抓到。"""
    out_dir, nodes, _ = _artifact(tmp_path)
    np.zeros(32, dtype=np.int8).tofile(
        out_dir / "kantor_A_Shift_buffer_file_phase_3_12.bin")

    report = verify.Report()
    verify.check_dq_phase_math(out_dir, nodes, report)
    assert any("shift" in line for line in _failed(report))


def test_softmax_check_catches_a_true_fp32_phase0(tmp_path: Path) -> None:
    """把 phase0 按**真 fp32** 落盘要被抓到——这是最容易犯的那个错。

    两种编码的字节数相同，所以只有这条低 2 字节的检查能区分。
    """
    out_dir, nodes, _ = _artifact(tmp_path)

    source = np.fromfile(
        out_dir / "input_buffer_phase_0_18.bin", dtype=np.float16)
    (out_dir / "output_buffer_phase_0_18.bin").write_bytes(
        struct.pack("<f", -float(source.max())))

    report = verify.Report()
    verify.check_softmax_reduction_encodings(out_dir, nodes, report)
    assert any("低 2 字节非零" in line for line in _failed(report))


def test_hardcoding_minus_thirty_point_seven_five_is_a_coincidence() -> None:
    """`fp32(-30.75)` 与 `fp16(-2.98046875) 放高半` **字节完全相同**。

    这解释了参考产物里为什么会解出 -30.75：它就是那份合成输入的
    `max = 2.98046875` 按「fp16 放高半」编码后被误当作 fp32 读出来的值。

    直接后果：对那一份特定输入，硬编码 -30.75 与正确编码无法区分。
    所以「不要硬编码」这条要靠**换一份输入**来验（见下一个测试），
    不能拿参考产物自己的节点验。
    """
    assert pack_fp16_in_high_half(-2.98046875) == struct.pack("<f", -30.75)


def test_softmax_check_catches_a_hardcoded_bias(tmp_path: Path) -> None:
    """把 `Bias_buffer_phase_1` 硬编码成 -30.75 要被抓到。

    用一份 max 不等于 2.98046875 的输入，绕开上一个测试记录的巧合。
    """
    files = WrittenFiles(tmp_path)
    source = np.linspace(-5, 4, 512).astype(np.float16)
    write_softmax_phases(files, 18, softmax(source))

    nodes = parse_blocks(write_gml(
        [Node(18, {"label": "sm", "name": "sm", "op_type": "Softmax"})],
        [], version="26.2.1"), "node")

    report = verify.Report()
    verify.check_softmax_reduction_encodings(tmp_path, nodes, report)
    assert not _failed(report), "正确的落点应当通过"

    (tmp_path / "Bias_buffer_phase_1_18.bin").write_bytes(
        struct.pack("<f", -30.75))
    report = verify.Report()
    verify.check_softmax_reduction_encodings(tmp_path, nodes, report)
    assert any("Bias_phase_1 != phase0" in line for line in _failed(report))


def test_softmax_check_catches_a_hardcoded_phase4_scale(tmp_path: Path) -> None:
    """把 `Scaling_buffer_phase_4` 写成 1.0 要被抓到。"""
    out_dir, nodes, _ = _artifact(tmp_path)
    np.ones(1, dtype=np.float16).tofile(
        out_dir / "Scaling_buffer_phase_4_18.bin")

    report = verify.Report()
    verify.check_softmax_reduction_encodings(out_dir, nodes, report)
    assert any("Scaling_phase_4 != phase3" in line for line in _failed(report))


def test_normalisation_check_catches_an_unnormalised_output(tmp_path: Path) -> None:
    """phase4 之和偏离 1 要被抓到（比如漏了乘 phase3）。"""
    out_dir, nodes, _ = _artifact(tmp_path)

    # phase1 的逐元素输出（exp 数组）按下一相的输入命名——
    # `input_buffer_phase_2`，不是 `output_buffer_phase_1`（参考产物没有
    # 这个名字，phase1 没有独立的标量输出，见 write_softmax_phases 的说明）。
    phase1 = np.fromfile(
        out_dir / "input_buffer_phase_2_18.bin", dtype=np.float16)
    phase1.tofile(out_dir / "output_buffer_phase_4_18.bin")

    report = verify.Report()
    verify.check_softmax_is_normalised(out_dir, nodes, report)
    assert any("Σ" in line for line in _failed(report))


def test_zero_point_check_catches_a_nonzero_value(tmp_path: Path) -> None:
    np.ones(1, dtype=np.int32).tofile(tmp_path / "output_zp_1.bin")

    report = verify.Report()
    verify.check_zero_points_are_zero(tmp_path, report)
    assert any("非零" in line for line in _failed(report))


def test_zero_point_check_catches_a_wrong_width(tmp_path: Path) -> None:
    """zp 写成 2 字节（而非 4 字节 int32）要被抓到。"""
    np.zeros(1, dtype=np.int16).tofile(tmp_path / "output_zp_1.bin")

    report = verify.Report()
    verify.check_zero_points_are_zero(tmp_path, report)
    assert any("宽度不对" in line for line in _failed(report))


def test_saturation_check_distinguishes_a_wrong_grouping_axis(
    tmp_path: Path,
) -> None:
    """分组轴取错时饱和度显著下降——这条是该探针的存在理由。

    实测：正确分组 100%、sf 取自错误的轴 68.6%。阈值 0.95 能区分。
    """
    from quant.weights import quantize_weight

    rng = np.random.default_rng(0)
    weight = rng.normal(0, 0.02, (256, 512)).astype(np.float32)

    correct = quantize_weight(weight)
    correct.values.tofile(tmp_path / "weight_buffer_1.bin")
    correct.scales.tofile(tmp_path / "weight_sf_1.bin")

    nodes = parse_blocks(write_gml(
        [Node(1, {"label": "w", "name": "w", "weight_buffer_dtype": "int4"})],
        [], version="26.2.1"), "node")

    report = verify.Report()
    verify.check_group_saturation(tmp_path, nodes, report)
    assert not _failed(report), "正确分组应当通过"

    # sf 取自错误的轴：per-column 峰值，再按连续 128 分组套用。
    groups = weight.size // 128
    wrong_scales = (np.abs(weight).max(axis=0) / 8).astype(np.float16)
    tiled = np.resize(wrong_scales, groups).astype(np.float32)
    wrong = np.clip(
        np.rint(weight.reshape(groups, 128) / tiled[:, None]), -8, 7
    ).astype(np.int8)
    wrong.ravel().tofile(tmp_path / "weight_buffer_1.bin")
    tiled.astype(np.float16).tofile(tmp_path / "weight_sf_1.bin")

    report = verify.Report()
    verify.check_group_saturation(tmp_path, nodes, report)
    assert _failed(report), "分组轴错误应当被抓到"


def test_checks_stay_silent_when_there_is_nothing_to_check(tmp_path: Path) -> None:
    """产物里没有对应文件时，检查不应虚报通过也不应报错。

    否则一个空目录会「全部通过」，给出虚假的安全感。
    """
    report = verify.Report()
    verify.check_dq_phase_math(tmp_path, [], report)
    verify.check_dq_phase_constants(tmp_path, [], report)
    verify.check_softmax_reduction_encodings(tmp_path, [], report)
    verify.check_group_saturation(tmp_path, [], report)
    verify.check_zero_points_are_zero(tmp_path, report)

    assert not report.passed
    assert not report.failed
