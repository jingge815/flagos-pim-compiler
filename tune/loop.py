"""外层调优循环：预算、响应日志、确定性续跑与失败隔离。

算法按原种子、原总评估与提案预算从头重跑；响应按调用序号日志回放，
已提交的候选不实际重复执行。墙钟预算由这里的单调时钟门控，只统计历史
活动耗时，回放与离线等待不计。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from tune._genesim import build_algorithm, search_algorithms
from tune.candidate import CandidateSpace, TuneState
from tune.config import SearchConfig, TuneConfig, run_identity
from tune.report import RoundRecord, append_round, read_rounds, rank_records
from tune.steps import compile_key, evaluate_candidate, sim_key


class DeadlineReached(RuntimeError):
    """本轮预算耗尽：停止发起新的昂贵评估。"""


@dataclass(frozen=True)
class ResponseRecord:
    """一次 Objective 调用的日志，按调用序号保存。"""

    call_index: int
    state_key: str
    status: str
    score: float | None
    score_special: str | None
    evaluator_invoked: bool
    elapsed_s: float
    sequence: int
    run_id: str = ""
    config_id: str = ""
    compile_key: str = ""
    sim_key: str = ""
    cache_mode: str = ""
    # 记录该次评估生效的外层预算，续跑据此拒绝收缩。
    budget: Mapping[str, Any] | None = None
    failure: str | None = None

    def to_json(self) -> dict:
        return {
            "call_index": self.call_index,
            "state_key": self.state_key,
            "status": self.status,
            "score": self.score,
            "score_special": self.score_special,
            "evaluator_invoked": self.evaluator_invoked,
            "elapsed_s": self.elapsed_s,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "config_id": self.config_id,
            "compile_key": self.compile_key,
            "sim_key": self.sim_key,
            "cache_mode": self.cache_mode,
            "budget": None if self.budget is None else dict(self.budget),
            "failure": self.failure,
        }

    @classmethod
    def from_json(cls, raw: Mapping) -> "ResponseRecord":
        budget = raw.get("budget")
        return cls(
            call_index=int(raw["call_index"]),
            state_key=str(raw["state_key"]),
            status=str(raw["status"]),
            score=None if raw.get("score") is None else float(raw["score"]),
            score_special=raw.get("score_special"),
            evaluator_invoked=bool(raw["evaluator_invoked"]),
            elapsed_s=float(raw.get("elapsed_s", 0.0)),
            sequence=int(raw["sequence"]),
            run_id=str(raw.get("run_id", "")),
            config_id=str(raw.get("config_id", "")),
            compile_key=str(raw.get("compile_key", "")),
            sim_key=str(raw.get("sim_key", "")),
            cache_mode=str(raw.get("cache_mode", "")),
            budget=dict(budget) if isinstance(budget, Mapping) else None,
            failure=raw.get("failure"),
        )

    @classmethod
    def from_round(cls, call_index: int, state_key: str,
                   record: RoundRecord, *,
                   elapsed_s: float | None = None,
                   config_id: str = "",
                   budget: Mapping[str, Any] | None = None) -> "ResponseRecord":
        score = None
        special = None
        if record.score is not None:
            if math.isinf(record.score):
                special = "inf" if record.score > 0 else "-inf"
            else:
                score = record.score
        status = "success" if record.status == "success" else "error"
        return cls(
            call_index=call_index, state_key=state_key, status=status,
            score=score, score_special=special, evaluator_invoked=True,
            elapsed_s=record.elapsed_s if elapsed_s is None else elapsed_s,
            sequence=record.sequence,
            run_id=record.run_id, config_id=config_id,
            compile_key=record.compile_key,
            sim_key=record.sim_key,
            cache_mode=record.cache, budget=budget,
            failure=record.failure,
        )
    def to_evaluation(self):
        _, _, Evaluation, EvaluationStatus = search_algorithms()
        score = self.score
        if self.score_special == "inf":
            score = math.inf
        elif self.score_special == "-inf":
            score = -math.inf
        if score is None:
            score = math.inf
        status = {
            "success": EvaluationStatus.SUCCESS,
            "invalid": EvaluationStatus.INVALID,
            "error": EvaluationStatus.ERROR,
        }[self.status]
        return Evaluation(score=score, status=status,
                          evaluator_invoked=self.evaluator_invoked)


@dataclass(frozen=True)
class SearchCoverage:
    """一次外层运行的真实搜索覆盖；缓存命中不能冒充新评估。"""

    algorithm: str
    logical_calls: int
    actual_calls: int
    resumed_calls: int
    new_pipeline_evaluations: int
    compile_cache_hits: int
    sim_cache_hits: int
    new_backend_executions: int
    final_mapping_replays: int

    @property
    def real_search(self) -> bool:
        """真实算法调用至少发起一次新 pipeline 并重放最终映射。"""
        return bool(
            self.actual_calls > 0
            and self.new_pipeline_evaluations > 0
            and self.final_mapping_replays > 0)

    def to_json(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "logical_calls": self.logical_calls,
            "actual_calls": self.actual_calls,
            "resumed_calls": self.resumed_calls,
            "new_pipeline_evaluations": self.new_pipeline_evaluations,
            "compile_cache_hits": self.compile_cache_hits,
            "sim_cache_hits": self.sim_cache_hits,
            "new_backend_executions": self.new_backend_executions,
            "final_mapping_replays": self.final_mapping_replays,
            "real_search": self.real_search,
        }


@dataclass(frozen=True)
class LoopResult:
    """一轮调优的汇总。"""

    records: tuple[RoundRecord, ...]
    best: RoundRecord | None
    stop_reason: str
    logical_calls: int
    actual_calls: int
    resumed_calls: int
    elapsed_s: float
    coverage: SearchCoverage

    @property
    def successful(self) -> list[RoundRecord]:
        return [record for record in self.records if record.successful]


def run_tune(
    cfg: TuneConfig,
    inputs,
    space: CandidateSpace,
    *,
    evaluate_fn: Callable[[TuneState], RoundRecord] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> LoopResult:
    """跑完外层搜索；已有日志前缀按原顺序回放，尾部才执行新项。"""
    output = cfg.output.dir
    output.mkdir(parents=True, exist_ok=True)
    responses_path = output / "responses.jsonl"
    rounds_path = output / "rounds.jsonl"
    identity = run_identity(cfg, inputs)
    config_identity = run_identity(cfg, None)
    saved = read_responses(responses_path)
    _verify_resume_identity(saved, identity, responses_path)
    _verify_resume_budget(saved, cfg.search, responses_path)
    records = _reconcile_rounds(rounds_path, saved)
    next_sequence = max((record.sequence for record in records), default=-1) + 1
    historical = sum(response.elapsed_s for response in saved)
    run_start = clock()
    # 只有新评估本身消耗墙钟预算；回放、离线等待和算法自身计算不计。
    new_activity = 0.0
    evaluator = evaluate_fn or (
        lambda state: evaluate_candidate(
            cfg, inputs, state,
            force_compile=cfg.cache.mode == "strict",
            # 一次运行只认一个身份：运行中即使环境或源码被改动，记录也不能
            # 从本轮排序里掉队，续跑仍按同一身份核对。
            run_id=identity))
    state = {"call_index": 0, "actual": 0, "resumed": 0}
    stop = {"reason": "search_space_exhausted"}

    def deadline_exceeded() -> bool:
        if cfg.search.max_time_s <= 0:
            return False
        return historical + new_activity >= cfg.search.max_time_s

    def objective(candidate: TuneState):
        nonlocal next_sequence, new_activity
        call_index = state["call_index"]
        key = _state_key(space.key(candidate))
        # 评估数预算先于回放：收缩后的预算不能靠旧日志继续推进。
        if call_index >= cfg.search.max_evaluations:
            raise DeadlineReached
        if call_index < len(saved):
            response = saved[call_index]
            if response.state_key != key:
                raise ValueError(
                    "恢复顺序与响应日志不一致："
                    f"第 {call_index} 次调用记录的是 {response.state_key}，"
                    f"本次重放得到 {key}")
            if inputs is not None:
                _verify_replayed_keys(
                    cfg, inputs, candidate, response, call_index)
            state["call_index"] += 1
            state["resumed"] += 1
            return response.to_evaluation()
        # 基线（第 0 次调用）必须完成；之后每次新评估前检查墙钟。
        if call_index > 0 and deadline_exceeded():
            raise DeadlineReached
        started = clock()
        record = evaluator(candidate)
        activity = max(0.0, clock() - started)
        new_activity += activity
        record = replace(record, sequence=next_sequence, run_id=identity,
                         call_index=call_index)
        next_sequence += 1
        append_round(rounds_path, record)
        records.append(record)
        response = ResponseRecord.from_round(
            call_index, key, record, elapsed_s=activity,
            config_id=config_identity, budget=_budget_payload(cfg.search))
        append_response(responses_path, response)
        state["call_index"] += 1
        state["actual"] += 1
        return response.to_evaluation()

    try:
        if cfg.search.algorithm == "enumerate":
            stop["reason"] = _run_enumerate(space, objective)
        else:
            algorithm = build_algorithm(
                cfg.search.algorithm, dict(cfg.search.algorithm_params),
                cfg.search.seed)
            _, Budget, _, _ = search_algorithms()
            budget = Budget(
                max_evaluations=cfg.search.max_evaluations,
                max_time_s=0.0,
                max_proposals=cfg.search.max_proposals,
            )
            result = algorithm.run(space, _ObjectiveAdapter(objective), budget)
            stop["reason"] = result.stop_reason
    except DeadlineReached:
        stop["reason"] = (
            "max_time_s" if deadline_exceeded() else "max_evaluations")

    # 排序只认本次运行的记录：同一输出目录里的其他配置记录不能混入结论。
    current = [record for record in records if record.run_id == identity]
    # 真实覆盖只看本次调用新产生的记录；回放与缓存命中不计入搜索证据。
    new_records = [
        record for record in current
        if record.call_index is not None
        and record.call_index >= len(saved)]
    ranked = [record for record in rank_records(current) if record.successful]
    return LoopResult(
        records=tuple(current),
        best=ranked[0] if ranked else None,
        stop_reason=stop["reason"],
        logical_calls=state["call_index"],
        actual_calls=state["actual"],
        resumed_calls=state["resumed"],
        elapsed_s=clock() - run_start,
        coverage=_search_coverage(
            cfg.search.algorithm, new_records,
            logical_calls=state["call_index"],
            actual_calls=state["actual"],
            resumed_calls=state["resumed"]),
    )


class _ObjectiveAdapter:
    """把普通函数适配成 GeneSim 的 Objective 协议。"""

    def __init__(self, fn: Callable[[TuneState], Any]) -> None:
        self._fn = fn

    def evaluate(self, state: TuneState):
        return self._fn(state)


def _search_coverage(
    algorithm: str,
    records: list[RoundRecord],
    *,
    logical_calls: int,
    actual_calls: int,
    resumed_calls: int,
) -> SearchCoverage:
    """只统计本次调用新产生的记录：回放与缓存命中不算真实搜索。"""
    new_records = [record for record in records if record.successful
                   and not record.sim_cache_hit]
    return SearchCoverage(
        algorithm=algorithm,
        logical_calls=logical_calls,
        actual_calls=actual_calls,
        resumed_calls=resumed_calls,
        new_pipeline_evaluations=len(new_records),
        compile_cache_hits=sum(
            1 for record in records if record.compile_cache_hit),
        sim_cache_hits=sum(
            1 for record in records if record.sim_cache_hit),
        new_backend_executions=sum(
            int(record.inner_evaluations)
            + int(record.validation_evaluations)
            for record in new_records),
        final_mapping_replays=sum(
            1 for record in new_records
            if int(record.validation_evaluations) >= 1
            and (record.paths or {}).get("effective_placement")),
    )


def _run_enumerate(space: CandidateSpace, objective) -> str:
    """确定性枚举；预算与墙钟由 objective 内部统一门控。"""
    for state in space.states:
        objective(state)
    return "search_space_exhausted"


def _reconcile_rounds(
    rounds_path: Path, saved: list[ResponseRecord],
) -> list[RoundRecord]:
    """响应记录是唯一提交点；轮次必须与响应按内容逐项对账。

    轮次先于响应落盘，崩溃可能留下完整但未提交的记录。恢复时只把
    有同序号响应且状态、分数、键与运行身份都一致的轮次送入排序与审计；
    孤儿写入单独文件并明确重跑。两份日志语义冲突时拒绝恢复，不能一边
    回放失败响应、一边从轮次日志选出成功候选。
    """
    records = read_rounds(rounds_path)
    indices = [response.call_index for response in saved]
    if indices != list(range(len(indices))):
        raise ValueError(
            "响应日志的调用序号不连续，无法安全续跑；"
            "请换一个输出目录或清理旧日志后重跑")
    saved_by_index = {response.call_index: response for response in saved}
    if not records:
        if saved:
            raise ValueError(
                f"响应日志有 {len(saved)} 条已提交调用，但 {rounds_path} "
                "没有对应轮次；日志已损坏")
        return []
    kept: list[RoundRecord] = []
    orphans: list[RoundRecord] = []
    seen: set[int] = set()
    for record in records:
        if record.call_index is None:
            raise ValueError(
                f"{rounds_path} 里的轮次记录缺少调用序号，无法与响应日志对账；"
                "请换一个输出目录或清理旧日志后重跑")
        if record.call_index in seen:
            raise ValueError(
                f"{rounds_path} 里的调用序号 {record.call_index} 出现多次，"
                "轮次日志已损坏")
        seen.add(record.call_index)
        response = saved_by_index.get(record.call_index)
        if response is None:
            orphans.append(record)
            continue
        _verify_round_response(record, response, rounds_path)
        kept.append(record)
    missing = sorted(set(saved_by_index) - {record.call_index for record in kept})
    if missing:
        raise ValueError(
            f"响应日志的调用序号 {missing} 没有对应的轮次记录，日志已损坏")
    if not orphans:
        return kept
    orphan_path = rounds_path.with_name("orphan_rounds.jsonl")
    with orphan_path.open("a") as handle:
        for record in orphans:
            handle.write(json.dumps({
                "reason": "响应未提交（崩溃窗口），已隔离并重跑",
                "record": record.to_json(),
            }, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _rewrite_rounds(rounds_path, kept)
    return kept


def _response_score(response: ResponseRecord) -> float | None:
    """还原响应里保存的分数；特殊值按原始编码还原。"""
    if response.score_special == "inf":
        return math.inf
    if response.score_special == "-inf":
        return -math.inf
    return response.score


def _verify_round_response(
    record: RoundRecord, response: ResponseRecord, path: Path,
) -> None:
    """轮次与同序号响应必须是同一次评估的同一结果。"""
    where = f"{path} 的调用序号 {record.call_index}"
    expected_status = "success" if record.status == "success" else "error"
    if response.status != expected_status:
        raise ValueError(
            f"{where} 响应状态 {response.status!r} 与轮次状态 "
            f"{record.status!r} 不一致，拒绝从冲突日志恢复")
    if not response.evaluator_invoked:
        raise ValueError(f"{where} 响应没有记录实际评估，无法作为提交点")
    if response.sequence != record.sequence:
        raise ValueError(f"{where} 响应的轮次编号与轮次日志不一致")
    if response.run_id != record.run_id:
        raise ValueError(f"{where} 响应的运行身份与轮次日志不一致")
    if response.compile_key != record.compile_key:
        raise ValueError(f"{where} 响应的编译键与轮次日志不一致")
    if response.sim_key != record.sim_key:
        raise ValueError(f"{where} 响应的仿真键与轮次日志不一致")
    expected_score = _response_score(response)
    if (expected_score is None) != (record.score is None) or (
            expected_score is not None and expected_score != record.score):
        raise ValueError(
            f"{where} 响应分数 {expected_score!r} 与轮次分数 "
            f"{record.score!r} 不一致")
    if response.failure != record.failure:
        raise ValueError(f"{where} 响应的失败原因与轮次日志不一致")


def _rewrite_rounds(path: Path, records: list[RoundRecord]) -> None:
    """用已对账记录原子重建轮次日志；快照发布不留半份文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record.to_json(), ensure_ascii=False) + "\n"
        for record in records)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def verify_resume_config(cfg: TuneConfig, path: Path) -> None:
    """入口处按配置身份预检续跑；输入内容身份在 run_tune 里再核对。"""
    responses = read_responses(path)
    expected = run_identity(cfg, None)
    mismatched = [response.call_index for response in responses
                  if response.config_id != expected]
    if mismatched:
        raise ValueError(
            f"{path} 里的响应来自另一份生效配置（首个不一致的调用序号 "
            f"{mismatched[0]}）。请换一个输出目录，或恢复原配置后再续跑。")
    _verify_resume_budget(responses, cfg.search, path)


def verify_resume_identity(cfg: TuneConfig, inputs, path: Path) -> None:
    """续跑在写生效配置前核对语义输入身份；异输入必须显式拒绝。"""
    responses = read_responses(path)
    identity = run_identity(cfg, inputs)
    mismatched = [response.call_index for response in responses
                  if response.run_id != identity]
    if mismatched:
        raise ValueError(
            f"{path} 里的响应来自另一份源码、工具、环境或输入（首个不一致的"
            f"调用序号 {mismatched[0]}）。续跑必须保持同一语义输入；请换一个"
            "输出目录，或恢复原输入后再试。")


def _budget_payload(search: SearchConfig) -> dict[str, Any]:
    return {
        "max_evaluations": int(search.max_evaluations),
        "max_time_s": float(search.max_time_s),
        "max_proposals": search.max_proposals,
    }


def _verify_resume_budget(saved: list[ResponseRecord], search: SearchConfig,
                          path: Path) -> None:
    """续跑只允许保持或扩大外层预算；收缩必须显式拒绝。

    每条历史响应都记录它执行时的预算。0 表示墙钟不限，因此从“不限”改成
    有限值也是收缩；提案预算未显式给出时按 GeneSim 的派生上限比较。
    """
    if not saved:
        return
    current = _budget_payload(search)

    def proposal_limit(budget: Mapping[str, Any]) -> int:
        explicit = budget.get("max_proposals")
        if explicit is not None:
            return int(explicit)
        return max(100, int(budget.get("max_evaluations", 1)) * 50)

    for response in saved:
        budget = response.budget
        if not budget:
            continue
        previous_evaluations = int(budget.get("max_evaluations", 0))
        previous_time = float(budget.get("max_time_s", 0.0))
        reasons = []
        if previous_evaluations > current["max_evaluations"]:
            reasons.append(
                f"max_evaluations {previous_evaluations} → "
                f"{current['max_evaluations']}")
        if previous_time == 0.0:
            if current["max_time_s"] > 0:
                reasons.append(
                    f"max_time_s 从不限改为 {current['max_time_s']}")
        elif (current["max_time_s"] != 0
              and current["max_time_s"] < previous_time):
            reasons.append(
                f"max_time_s {previous_time} → {current['max_time_s']}")
        if proposal_limit(budget) > proposal_limit(current):
            reasons.append(
                f"max_proposals {proposal_limit(budget)} → "
                f"{proposal_limit(current)}")
        if reasons:
            raise ValueError(
                f"{path} 里的响应使用了更大的外层预算（第 "
                f"{response.call_index} 次调用：{'；'.join(reasons)}）。"
                "续跑只允许保持或扩大预算，请恢复原预算或换一个输出目录。")


def _verify_replayed_keys(cfg: TuneConfig, inputs, candidate: TuneState,
                          response: ResponseRecord, call_index: int) -> None:
    """回放前按当前源码、工具、画像、基配置、环境和输入重算内容键。"""
    expected_compile = compile_key(cfg, inputs, candidate)
    if response.compile_key and response.compile_key != expected_compile:
        raise ValueError(
            f"响应日志第 {call_index} 次调用的编译键与当前源码、工具或画像"
            "不一致，拒绝续跑")
    expected_sim = sim_key(cfg, inputs, candidate, expected_compile)
    if response.sim_key and response.sim_key != expected_sim:
        raise ValueError(
            f"响应日志第 {call_index} 次调用的仿真键与当前源码、工具、画像、"
            "基础配置、环境、轨迹或放置输入不一致，拒绝续跑")


def _verify_resume_identity(saved: list[ResponseRecord], identity: str,
                            path: Path) -> None:
    """续跑只允许同一生效配置与输入；异配置必须拒绝而不是混合排序。"""
    mismatched = [response.call_index for response in saved
                  if response.run_id != identity]
    if mismatched:
        raise ValueError(
            f"{path} 里的响应来自另一份生效配置或输入（首个不一致的调用序号 "
            f"{mismatched[0]}）。续跑必须保持同一配置；请换一个输出目录，"
            "或恢复原配置后再试。")


def append_response(path: Path, response: ResponseRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(response.to_json(), ensure_ascii=False)
    with path.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())


def read_responses(path: Path) -> list[ResponseRecord]:
    """读响应日志；末尾半行截断，完整坏行直接报错。"""
    if not Path(path).is_file():
        return []
    text = Path(path).read_text()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    responses = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            responses.append(ResponseRecord.from_json(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"响应日志第 {number} 行是坏行：{exc}") from exc
    return responses


def _state_key(key: Any) -> str:
    return json.dumps(key, default=str, sort_keys=True)
