"""实施文档数值证据与 summary 产物的一致性检查。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_implement_evidence import (  # noqa: E402
    check_document,
    summary_metrics,
)

BLOCK_START = "<!-- evidence-block:start -->"
BLOCK_END = "<!-- evidence-block:end -->"


def _payload(
    total_time_s: float,
    throughput_tokens_per_s: float,
    total_useful_flops: float | None,
    *,
    layers: int = 4,
    model_type: str = "qwen3_moe",
    execution: str = "dynamic_dispatch",
) -> dict:
    payload = {
        "total_time_s": total_time_s,
        "throughput_tokens_per_s": throughput_tokens_per_s,
        "efficiency_metrics": {"compute": {}},
        "model": {
            "model_type": model_type,
            "num_layers": layers,
            "moe": {"execution": execution},
        },
    }
    if total_useful_flops is not None:
        payload["efficiency_metrics"]["compute"][
            "total_useful_flops"] = total_useful_flops
    return payload


DYNAMIC = _payload(
    88.48313625886303, 8.984762900742771, 5415241254987.654)
DENSE = _payload(
    91.01234662163839, 8.735078585601341, 5826198177867.654,
    execution="dense_sim")
ALT = _payload(138.589, 5.736, None)
FULL48 = _payload(
    566.8630033645501, 1.402455258645156, 28734818737588.309,
    layers=48)
ROUND03 = _payload(73.593, 10.809, 5430713177457.654)

DYNAMIC_ENTRY = {
    "label": "dynamic",
    "stratum": "current",
    "artifact": "qwen3_moe",
    "layers": 4,
    "semantics": "dynamic_dispatch",
    "command": "run_full_pipeline --num-stages 2",
    "summary": "/tmp/round11/dynamic/summary.json",
    "metrics": {
        "total_time_s": DYNAMIC["total_time_s"],
        "throughput_tokens_per_s": DYNAMIC["throughput_tokens_per_s"],
        "total_useful_flops": 5415241254987.654,
    },
}
DENSE_ENTRY = {
    "label": "dense",
    "stratum": "current",
    "artifact": "qwen3_moe",
    "layers": 4,
    "semantics": "dense_sim",
    "command": "run_full_pipeline --moe-execution dense_sim",
    "summary": "/tmp/round11/dense/summary.json",
    "metrics": {
        "total_time_s": DENSE["total_time_s"],
        "throughput_tokens_per_s": DENSE["throughput_tokens_per_s"],
        "total_useful_flops": 5826198177867.654,
    },
}
ALT_ENTRY = {
    "label": "alt",
    "stratum": "current",
    "artifact": "qwen3_moe",
    "layers": 4,
    "semantics": "dynamic_dispatch",
    "command": "run_full_pipeline --num-dpus 4 --num-stages 4",
    "summary": "/tmp/round11/alt/summary.json",
    "metrics": {
        "total_time_s": ALT["total_time_s"],
        "throughput_tokens_per_s": ALT["throughput_tokens_per_s"],
    },
}
FULL48_ENTRY = {
    "label": "48l",
    "stratum": "current",
    "artifact": "qwen3_moe",
    "layers": 48,
    "semantics": "dynamic_dispatch",
    "command": "run_full_pipeline --model-dir <48 层>",
    "summary": "/tmp/round11/48l/summary.json",
    "metrics": {
        "total_time_s": FULL48["total_time_s"],
        "throughput_tokens_per_s": FULL48["throughput_tokens_per_s"],
    },
}
ROUND03_ENTRY = {
    "label": "round03",
    "stratum": "history",
    "artifact": "qwen3_moe",
    "layers": 4,
    "metrics": {"total_useful_flops": 5430713177457.654},
}
DERIVED = [
    {
        "id": "dynamic_vs_round03_flops_delta",
        "op": "subtract",
        "left": "round03.total_useful_flops",
        "right": "dynamic.total_useful_flops",
        "value": 1.55e10,
    },
    {
        "id": "dense_over_dynamic_time",
        "op": "ratio",
        "left": "dense.total_time_s",
        "right": "dynamic.total_time_s",
        "value": 1.029,
    },
    {
        "id": "dense_flops_above_dynamic",
        "op": "greater_than",
        "left": "dense.total_useful_flops",
        "right": "dynamic.total_useful_flops",
        "value": True,
    },
]


def _base_block() -> dict:
    block = copy.deepcopy({
        "runs": [
            DYNAMIC_ENTRY, DENSE_ENTRY, ALT_ENTRY, FULL48_ENTRY,
            ROUND03_ENTRY,
        ],
        "derived": [dict(item) for item in DERIVED],
        "facts": [
            {"id": "dynamic_comm_bytes", "run": "dynamic",
             "source": "E 阶段依赖边累计", "value": 14692832968},
            {"id": "dense_comm_bytes", "run": "dense",
             "source": "E 阶段依赖边累计", "value": 15464703688},
            {"id": "alt_comm_bytes", "run": "alt",
             "source": "E 阶段依赖边累计", "value": 8218497128},
        ],
        "measurements": [
            {"id": "bench_prefill_speedup", "stratum": "history",
             "value": 1.148},
            {"id": "bench_decode_speedup", "stratum": "history",
             "value": 1.202},
            {"id": "old_ratio", "stratum": "history", "value": 1.034},
            {"id": "reviewer_old_ab", "stratum": "history", "value": 0.994},
        ],
    })
    aliases = {
        "dynamic": ["dynamic_dispatch", "dynamic"],
        "dense": ["dense_sim", "dense"],
        "alt": ["tp1_pp4", "Alt"],
        "48l": ["48 层", "48L"],
        "round03": ["round03"],
    }
    for entry in block["runs"]:
        entry["prose_aliases"] = aliases[entry["label"]]
    return block


def _doc(block: dict, prose: str = "") -> str:
    body = yaml.safe_dump(block, sort_keys=False)
    return (
        prose
        + f"\n\n{BLOCK_START}\n```evidence\n{body}```\n{BLOCK_END}\n"
    )


def _prose() -> str:
    return (
        "dynamic **88.483 s / 8.985 t/s**，有效 FLOPs "
        "`5,415,241,254,987.654`，通信字节 14,692,832,968；"
        "dense **91.012 s / 8.735 t/s**，"
        "`5,826,198,177,867.654`，通信字节 15,464,703,688。"
        "Alt 138.589 s / 5.736 t/s，通信字节 8,218,497,128。"
        "48 层 566.863 s / 1.402 t/s。round03 为 "
        "`5,430,713,177,457.654`，差值 1.55e10；dense/dynamic = 1.029×。\n"
    )


def _summary(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _paths(tmp_path: Path) -> dict:
    return {
        "dynamic": _summary(tmp_path / "dynamic.json", DYNAMIC),
        "dense": _summary(tmp_path / "dense.json", DENSE),
        "alt": _summary(tmp_path / "alt.json", ALT),
        "48l": _summary(tmp_path / "48l.json", FULL48),
    }


def _check(tmp_path: Path, block: dict, prose: str = None):
    return check_document(
        _doc(block, _prose() if prose is None else prose),
        _paths(tmp_path),
        time_only={"alt", "48l"},
    )


def test_summary_metrics_are_extracted(tmp_path: Path) -> None:
    metrics = summary_metrics(_summary(tmp_path / "s.json", DYNAMIC))
    assert metrics["total_useful_flops"] == 5415241254987.654
    assert metrics["total_time_s"] == 88.48313625886303


def test_document_with_current_values_passes(tmp_path: Path) -> None:
    problems, unknown = _check(tmp_path, _base_block())
    assert problems == []
    assert unknown == []


def test_swapped_dynamic_and_dense_values_are_rejected(tmp_path: Path) -> None:
    """dynamic/dense 对调后，标签与 summary 的归属不符。"""
    block = _base_block()
    block["runs"][0]["metrics"], block["runs"][1]["metrics"] = (
        block["runs"][1]["metrics"], block["runs"][0]["metrics"])
    problems, _ = _check(tmp_path, block)
    assert any("dynamic" in item for item in problems), problems


def test_four_layer_summary_cannot_pose_as_forty_eight(tmp_path: Path) -> None:
    """4 层 summary 填进 48 层标签时，层数与模型身份对不上。"""
    block = _base_block()
    block["runs"][3]["layers"] = 48
    paths = _paths(tmp_path)
    paths["48l"] = _summary(tmp_path / "fake48.json", DYNAMIC)
    problems, _ = check_document(_doc(block, _prose()), paths,
                                 time_only={"alt", "48l"})
    assert any("48l" in item and "层" in item for item in problems), problems


def test_history_value_cannot_be_declared_as_current(tmp_path: Path) -> None:
    """历史数值必须有独立命名空间，不能冒充本轮 current 证据。"""
    block = _base_block()
    block["runs"][4]["stratum"] = "current"
    problems, _ = _check(tmp_path, block)
    assert any("round03" in item for item in problems), problems


def test_current_label_without_a_summary_is_rejected(tmp_path: Path) -> None:
    """文档声明为 current、检查时却没有对应 summary 的条目必须失败。"""
    block = _base_block()
    block["runs"].append({
        "label": "imaginary",
        "stratum": "current",
        "artifact": "qwen3_moe",
        "layers": 4,
        "metrics": {"total_time_s": 1.0},
    })
    problems, _ = _check(tmp_path, block)
    assert any("imaginary" in item for item in problems), problems


def test_wrong_derived_difference_is_rejected(tmp_path: Path) -> None:
    """派生差值必须由同一组指标重算，科学计数法错误不能绕过。"""
    block = _base_block()
    block["derived"][0]["value"] = 9.99e99
    problems, _ = _check(tmp_path, block)
    assert any("dynamic_vs_round03_flops_delta" in item for item in problems)


def test_wrong_derived_ratio_is_rejected(tmp_path: Path) -> None:
    block = _base_block()
    block["derived"][1]["value"] = 1.030
    problems, _ = _check(tmp_path, block)
    assert any("dense_over_dynamic_time" in item for item in problems)


def test_wrong_derived_conclusion_is_rejected(tmp_path: Path) -> None:
    block = _base_block()
    block["derived"][2]["value"] = False
    problems, _ = _check(tmp_path, block)
    assert any("dense_flops_above_dynamic" in item for item in problems)


def test_declared_history_values_are_allowed(tmp_path: Path) -> None:
    problems, unknown = _check(tmp_path, _base_block())
    assert problems == []
    assert unknown == []


def test_scientific_notation_outside_the_evidence_block_is_rejected(
    tmp_path: Path,
) -> None:
    """正文里的科学计数法错误派生值不能因为格式特殊而被忽略。"""
    prose = _prose() + "旧实现差值 9.99e99。\n"
    problems, unknown = _check(tmp_path, _base_block(), prose=prose)
    assert "9.99e99" in unknown


def test_time_only_metrics_do_not_require_flops(tmp_path: Path) -> None:
    metrics = summary_metrics(_summary(tmp_path / "d.json", DYNAMIC))
    metrics.pop("total_useful_flops")
    block = _base_block()
    problems, _ = _check(tmp_path, block)
    assert problems == []


def test_missing_evidence_block_is_rejected(tmp_path: Path) -> None:
    problems, _ = check_document(
        "dynamic 88.483 s / 8.985 t/s\n", _paths(tmp_path))
    assert any("证据块" in item for item in problems), problems


# ── round12 问题2：正文数值必须绑定到所属运行标签 ────────────────────


def test_prose_only_swap_of_dynamic_and_dense_is_rejected(
        tmp_path: Path) -> None:
    """证据块保持正确、只把正文的 dynamic/dense 数值对调也必须失败。"""
    prose = (
        "dynamic **91.012 s / 8.735 t/s**，有效 FLOPs "
        "`5,826,198,177,867.654`；dense **88.483 s / 8.985 t/s**，"
        "`5,415,241,254,987.654`。Alt 138.589 s / 5.736 t/s。"
        "48 层 566.863 s / 1.402 t/s。round03 为 "
        "`5,430,713,177,457.654`，差值 1.55e10；dense/dynamic = 1.029×。\n")
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert problems, "只交换正文也必须被拒绝"
    assert any("dynamic" in item or "dense" in item for item in problems)


def test_prose_cannot_borrow_a_forty_eight_layer_value(
        tmp_path: Path) -> None:
    """48 层语境里出现 4 层运行的值必须失败，即使该值在别处合法。"""
    prose = _prose() + "\n48 层复跑 88.483 s。\n"
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("48l" in item or "88.483" in item for item in problems), problems


def test_prose_history_value_cannot_pose_as_current(tmp_path: Path) -> None:
    """历史运行的指标不能写进本轮运行的语境。"""
    prose = _prose() + "\ndynamic 复跑 73.593 s。\n"
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("dynamic" in item or "73.593" in item for item in problems)


def test_prose_comparison_conclusion_is_checked(tmp_path: Path) -> None:
    """正文里两个证据数值的大小结论必须与实际值一致。"""
    right = _prose() + "dense 的 91.012 s 高于 dynamic 的 88.483 s。\n"
    problems, _ = _check(tmp_path, _base_block(), prose=right)
    assert problems == []

    wrong = _prose() + "\ndynamic 的 88.483 s 高于 dense 的 91.012 s。\n"
    problems, _ = _check(tmp_path, _base_block(), prose=wrong)
    assert any("高于" in item for item in problems), problems


def test_required_values_must_appear_in_the_prose(tmp_path: Path) -> None:
    """必需值只出现在证据块里不算正文引用。"""
    prose = _prose().replace(
        "dynamic **88.483 s / 8.985 t/s**", "dynamic 已重跑")
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("88.483" in item for item in problems), problems


def test_measurements_are_allowed_in_a_run_context(tmp_path: Path) -> None:
    """独立 measurements 是历史/外部命名空间，可以出现在任何运行语境。"""
    prose = _prose() + "\ndynamic 复跑加速比为 1.148×。\n"
    problems, unknown = _check(tmp_path, _base_block(), prose=prose)
    assert problems == []
    assert unknown == []


# ── round13/14 问题5：逐值归属、必需值语境与比较身份 ─────────────────


def test_same_block_former_latter_swap_is_rejected(tmp_path: Path) -> None:
    """同块两个运行都在值前面时，前者/后者必须按出现顺序绑定运行身份。"""
    prose = _prose() + (
        "dense_sim 与 dynamic_dispatch 中，前者的 88.483 s 低于后者的 "
        "91.012 s。\n")
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("前者" in item for item in problems), problems


def test_former_latter_identity_must_match_the_declared_runs(
        tmp_path: Path) -> None:
    """正确方向的「前者/后者」仍要核对两侧数值确实属于对应运行。"""
    prose = _prose() + (
        "dense_sim 与 dynamic_dispatch 中，前者的 91.012 s 高于后者的 "
        "88.483 s。\n")
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert problems == [], problems


def test_required_values_in_an_unlabeled_preamble_do_not_count(
        tmp_path: Path) -> None:
    """必需指标只写在无标签前言、标签后不再给值时不能算正文引用。"""
    preamble = (
        "全部指标：88.483 s、8.985 t/s、91.012 s、8.735 t/s、138.589 s、"
        "566.863 s、1.402 t/s。\n")
    prose = preamble + (
        _prose()
        .replace("dynamic **88.483 s / 8.985 t/s**", "dynamic 已重跑")
        .replace("dense **91.012 s / 8.735 t/s**", "dense 已重跑")
        .replace("Alt 138.589 s / 5.736 t/s", "Alt 已重跑")
        .replace("48 层 566.863 s / 1.402 t/s", "48 层已重跑"))
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("没有出现在" in item for item in problems), problems


def test_fact_values_are_bound_to_their_run(tmp_path: Path) -> None:
    """容量/通信/trace 等非 summary 证据值也必须归属到运行语境。"""
    prose = _prose().replace(
        "通信字节 14,692,832,968", "通信字节 15,464,703,688")
    problems, _ = _check(tmp_path, _base_block(), prose=prose)
    assert any("15,464,703,688" in item for item in problems), problems


def test_facts_with_an_unknown_run_are_rejected(tmp_path: Path) -> None:
    """事实项引用不存在的运行标签时不能静默跳过。"""
    block = _base_block()
    block["facts"].append(
        {"id": "ghost", "run": "imaginary", "value": 12345678901})
    problems, _ = _check(tmp_path, block)
    assert any("imaginary" in item for item in problems), problems


def test_facts_with_a_non_numeric_value_are_rejected(tmp_path: Path) -> None:
    """事实项的值必须是数值，否则绑定与溯源都无从谈起。"""
    block = _base_block()
    block["facts"].append(
        {"id": "bad", "run": "dynamic", "value": "很大"})
    problems, _ = _check(tmp_path, block)
    assert any("不是数值" in item for item in problems), problems


def test_documented_checker_command_only_uses_current_summaries() -> None:
    """文档声明的 checker 命令必须只核 current summary，并实际退出码 0。

    history 条目（alt/48l）只能作为正文引用，不能靠 `--summary`/`--time-only`
    核验；把「四份 summary 通过」写成结论会让历史值冒充本轮机器核验。
    """
    import subprocess

    from scripts.check_implement_evidence import parse_evidence_block

    doc_path = ROOT / "docs" / "implement-qwen3-moe-20261105.md"
    doc = doc_path.read_text()
    for stale in ("checker 对四份 summary", "当前文档 + 四份 summary",
                  "当前文档与四份 summary"):
        assert stale not in doc, (
            f"文档仍声称「{stale}」通过：上轮命令只核 dynamic/dense 两份")
    block = parse_evidence_block(doc)
    current = [entry for entry in block["runs"]
               if str(entry.get("stratum", "history")) == "current"]
    assert {str(entry["label"]) for entry in current} == {"dynamic", "dense"}
    command = [sys.executable, "scripts/check_implement_evidence.py",
               "--doc", str(doc_path)]
    for entry in current:
        summary = Path(str(entry["summary"]))
        if not summary.is_file():
            pytest.skip(f"本轮 summary 产物不存在：{summary}")
        command += ["--summary", f"{entry['label']}={summary}"]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
