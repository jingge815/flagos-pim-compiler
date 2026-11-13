"""结果记录：轮次 JSONL、稳定排序、best 与摘要。"""

import json
import math
from dataclasses import replace

import pytest

from contracts.hardware_profile import load_profile
from contracts.op_contract import PimOptimizations
from tune.candidate import OpKnobs, StrategyChoice, TuneState
from tune.report import (
    RoundRecord,
    append_round,
    candidate_id,
    compare_scores,
    rank_records,
    read_rounds,
    state_from_json,
    state_to_json,
    write_best,
    write_summary,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PROFILE = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"


def _state(num_dpus=8, num_stages=4, tasklets=4):
    return TuneState(StrategyChoice(num_dpus, num_stages), OpKnobs(tasklets),
                     load_profile(PROFILE))


def _record(sequence, score, candidate_id=None, status="success"):
    return RoundRecord(
        sequence=sequence,
        candidate_id=candidate_id or f"cand-{sequence}",
        compile_key=f"c{sequence}",
        sim_key=f"s{sequence}",
        cache="fast",
        status=status,
        result_maturity="model_prediction",
        state=_state(),
        score=score,
        metrics={"total_time_s": score} if score is not None else {},
        stop_reasons={},
        elapsed_s=0.0,
        search_elapsed_s=0.0,
        inner_evaluations=0,
        validation_evaluations=1,
        artifacts=None,
        failure=None if status == "success" else "boom",
        paths={},
    )


def test_ties_sort_by_candidate_id():
    records = [_record(0, 1.0, "b"), _record(1, 2.0, "a"),
               _record(2, 1.0, "a"), _record(3, 3.0, "c")]
    ranked = rank_records(records)
    # 先按分数，同分再按候选编号。
    assert [r.candidate_id for r in ranked] == ["a", "b", "a", "c"]
    assert [r.score for r in ranked] == [1.0, 1.0, 2.0, 3.0]
    # 输入顺序不影响排序结果。
    assert rank_records(list(reversed(records))) == ranked


def test_failed_records_rank_last_and_never_become_best(tmp_path):
    records = [_record(0, None, "bad", status="error"),
               _record(1, 2.0, "ok")]
    assert [r.candidate_id for r in rank_records(records)] == ["ok", "bad"]
    best_path = write_best(tmp_path, records)
    best = json.loads(best_path.read_text())
    assert best["candidate_id"] == "ok"


def test_all_failed_produces_no_best(tmp_path):
    records = [_record(0, None, "x", status="error")]
    assert write_best(tmp_path, records) is None
    write_summary(tmp_path, "unit", records)
    assert not (tmp_path / "best.json").exists()
    summary = (tmp_path / "summary.md").read_text()
    assert "没有成功候选" in summary


def test_rounds_jsonl_round_trips_and_rejects_half_lines(tmp_path):
    path = tmp_path / "rounds.jsonl"
    append_round(path, _record(0, 1.0))
    append_round(path, _record(1, 0.5))
    loaded = read_rounds(path)
    assert [record.sequence for record in loaded] == [0, 1]
    # 末尾半行是崩溃留下的，截断到最后换行；完整坏行必须拒绝。
    with path.open("a") as handle:
        handle.write('{"sequence": 2')
    assert len(read_rounds(path)) == 2
    path.write_text(path.read_text() + '{"sequence": 9\n')
    with pytest.raises(ValueError, match="坏行"):
        read_rounds(path)


def test_strict_replay_uses_relative_tolerance():
    compare_scores(1.0, 1.0 + 5e-10)
    with pytest.raises(AssertionError):
        compare_scores(1.0, 1.0 + 5e-9)


# ---- 候选状态里的优化开关：记录、续跑与候选编号都要带上 ----


def test_state_round_trip_keeps_the_optimization_switch():
    state = TuneState(
        StrategyChoice(8, 4), OpKnobs(4), load_profile(PROFILE),
        PimOptimizations(loop_tiling=False))
    raw = json.loads(json.dumps(state_to_json(state)))
    restored = state_from_json(raw)

    assert restored.optimization == state.optimization
    assert candidate_id(restored) != candidate_id(_state())
    assert "_opt1110_" in candidate_id(restored)


def test_old_state_without_optimization_keeps_none():
    raw = json.loads(json.dumps(state_to_json(_state())))
    raw.pop("optimization")
    assert state_from_json(raw).optimization is None
