"""硬件敏感性扫描：逐字段求值、拐点与 Pareto 前沿。

扫描沿用与调优同一个 Objective；只改仿真字段时复用编译产物，改变核数、
容量或编译预算时按新画像重编。非法扫描点保留原因，不静默删除。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from tune.candidate import TuneState, _values_equal, build_strategy, model_dims
from tune.config import PARETO_RESOURCES, TuneConfig, load_config
from tune.inputs import file_sha256, verify_input_identity
from tune.report import RoundRecord, candidate_id
from tune.steps import compile_key, evaluate_candidate, sim_key

RESOURCE_FIELDS = frozenset({
    "num_pim_devices", "clusters_per_device", "cluster_memory_bytes",
    "wram_bytes_per_dpu", "num_dpus",
})


@dataclass(frozen=True)
class SweepPoint:
    values: Mapping[str, Any]
    state: TuneState | None
    record: RoundRecord | None
    compile_fields_changed: bool
    reason: str | None = None
    objective_direction: str = "minimize"
    resource_direction: str = "minimize"
    result_maturity: str = "model_prediction"
    source_origins: tuple[str, ...] = ()
    load_key: str = ""
    # 构造/前置资源失败在执行前判定；执行期失败仍保留候选的原始状态。
    invalid: bool = False

    @property
    def score(self) -> float | None:
        return None if self.record is None else self.record.score

    @property
    def status(self) -> str:
        if self.invalid:
            return "invalid"
        return "error" if self.record is None else self.record.status

    def to_json(self) -> dict:
        return {
            "values": dict(self.values),
            "compile_fields_changed": self.compile_fields_changed,
            "status": self.status,
            "invalid": self.invalid,
            "score": self.score,
            "reason": self.reason,
            "candidate_id": None if self.record is None else self.record.candidate_id,
            "compile_key": None if self.record is None else self.record.compile_key,
            "sim_key": None if self.record is None else self.record.sim_key,
            "metrics": {} if self.record is None else dict(self.record.metrics),
            "objective_direction": self.objective_direction,
            "resource_direction": self.resource_direction,
            "result_maturity": self.result_maturity,
            "source_origins": list(self.source_origins),
            "load_key": self.load_key,
            "parameter_sources": (
                {} if self.state is None
                else {key: dict(value) for key, value in
                      self.state.profile.parameter_sources.items()}),
        }


@dataclass(frozen=True)
class SweepResult:
    points: tuple[SweepPoint, ...]
    sensitivity: Mapping[str, Any]
    pareto: tuple[SweepPoint, ...]
    knees: Mapping[str, Any]
    # best 与本次扫描负载/配置/基线画像的绑定证据。
    binding: Mapping[str, Any] | None = None


# 扫描只允许改变扫描字段、预算、缓存档位与输出目录；这些配置段必须与
# best 生成时一致，否则等于借另一 best 隐式换工作负载。
_SWEEP_SHARED_PLACEMENT = (
    "enabled", "algorithm", "evaluation_backend", "validation_backend",
    "objective_metric", "objective_weights", "seed", "moves",
    "algorithm_params",
)


def verify_best_reference(cfg: TuneConfig, inputs, best: Mapping[str, Any],
                          best_state: TuneState) -> None:
    """扫描前核对 best 的成功状态、输入内容、配置与基线画像。

    任何不符都在发布输入快照、写生效配置与开始求值之前抛出 ValueError。
    现场还没生成 IR 时先跳过 IR 内容，发布生成后由调用方用同一入口再核对。
    """
    if not isinstance(best, Mapping):
        raise ValueError("best.json 必须是 JSON 对象")
    if best.get("status") != "success":
        raise ValueError(f"best.json 状态不是 success：{best.get('status')!r}")
    score = best.get("score")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score))):
        raise ValueError(f"best.json 没有有限分数：{score!r}")
    if inputs is None:
        raise ValueError("扫描缺少输入快照，无法核对 best 的工作负载")
    try:
        # 扫描目录尚未生成 IR 时先核对模型/轨迹与请求窗口；IR 生成后，
        # CLI 会在写生效配置前用同一个入口再核对一次内容。
        verify_input_identity(
            inputs, best.get("inputs") or {}, require_ir=inputs.ir_ready)
    except ValueError as exc:
        raise ValueError(f"best.json 的输入与本次扫描不一致：{exc}") from exc
    # best 状态可带硬件层搜索后的画像，因此基线身份按生成时记录的
    # 配置画像摘要核对，而不是要求状态画像等于配置画像。
    if best.get("baseline_profile_sha256") != cfg.profile.profile_sha256():
        raise ValueError("best.json 的基线画像与本次扫描画像不一致")
    if best.get("candidate_id") != candidate_id(best_state):
        raise ValueError("best.json 的候选编号与其状态不一致")
    recorded = best.get("effective_config")
    if not isinstance(recorded, str) or not Path(recorded).is_file():
        raise ValueError(f"best.json 没有可读取的生效配置：{recorded!r}")
    try:
        best_cfg = load_config(Path(recorded))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"best.json 的生效配置无法解析：{exc}") from exc
    # 生效配置是生成时的不可变内容身份：同路径原地改写必须先于任何签名核对。
    recorded_sha = best.get("effective_config_sha256")
    if (recorded_sha is not None
            and recorded_sha != file_sha256(Path(recorded))):
        raise ValueError(
            "best.json 的生效配置内容与生成时摘要不一致，"
            "不能把另一配置的分数当作本次扫描基准")
    if _shared_config_identity(cfg) != _shared_config_identity(best_cfg):
        raise ValueError(
            "best.json 的生效配置与本次扫描的模型/空间/仿真配置不一致")
    # IR 尚未生成时无法重算内容键；发布生成后同一入口会再核对一次。
    if not inputs.ir_ready:
        return
    best_compile = best.get("compile_key")
    if not isinstance(best_compile, str) or not best_compile:
        raise ValueError("best.json 缺少编译键，无法证明生成时的配置内容")
    expected_compile = compile_key(best_cfg, inputs, best_state)
    if best_compile != expected_compile:
        raise ValueError(
            "best.json 的历史编译键与当前生效配置/输入不一致："
            f"{best_compile!r} vs {expected_compile!r}")
    best_sim = best.get("sim_key")
    if not isinstance(best_sim, str) or not best_sim:
        raise ValueError("best.json 缺少仿真键，无法证明生成时的配置内容")
    expected_sim = sim_key(best_cfg, inputs, best_state, expected_compile)
    if best_sim != expected_sim:
        raise ValueError(
            "best.json 的历史仿真键与当前生效配置/输入不一致："
            f"{best_sim!r} vs {expected_sim!r}")


def _shared_config_identity(cfg: TuneConfig) -> dict:
    """同一负载的配置段；候选空间、搜索预算、扫描字段、缓存与输出目录除外。

    扫描固定使用 best 状态，不重新搜索候选，所以候选空间与搜索预算不
    改变被扫描的负载；模型、放置语义、仿真与校准配置必须一致。
    """
    data = cfg.to_json()
    model = dict(data["model"])
    # 路径不入身份：输入内容由 verify_input_identity 逐项核对。
    model.pop("model_dir", None)
    model.pop("ir", None)
    trace = dict(model.get("trace") or {})
    trace.pop("path", None)
    model["trace"] = trace
    placement = dict(data["placement"])
    return {
        "model": model,
        "placement": {
            key: placement.get(key) for key in _SWEEP_SHARED_PLACEMENT},
        "simulation": data["simulation"],
        "calibration": _calibration_identity(data),
        "profile_sha256": cfg.profile.profile_sha256(),
    }


def _calibration_identity(data: Mapping[str, Any]) -> Any:
    raw = (data.get("calibration") or {}).get("validation_report")
    if not raw:
        return None
    path = Path(raw)
    return file_sha256(path) if path.is_file() else str(raw)


def run_sweep(
    cfg: TuneConfig,
    inputs,
    best_state: TuneState,
    *,
    best: Mapping[str, Any] | None = None,
    evaluate_fn: Callable[[TuneState], RoundRecord] | None = None,
) -> SweepResult:
    """逐字段扫描并输出可独立复算的结论。"""
    if not cfg.sweep.fields:
        raise ValueError("sweep.fields 为空，没有可扫描的字段")
    if best is not None:
        # 库调用方也走同一前置契约，不能绕过 CLI 直接扫描另一 best。
        verify_best_reference(cfg, inputs, best, best_state)
    dims = model_dims(cfg.model.model_dir)
    # 扫描入口与主循环共用同一执行计划：strict 必须强制重编，不能复用缓存。
    evaluator = evaluate_fn or (lambda state: evaluate_candidate(
        cfg, inputs, state, force_compile=cfg.cache.mode == "strict"))
    base_profile = best_state.profile
    points: list[SweepPoint] = []
    keys = list(cfg.sweep.fields)
    for combo in itertools.product(*(cfg.sweep.fields[key] for key in keys)):
        values = dict(zip(keys, combo))
        try:
            profile = _mark_scanned(
                base_profile.with_fields(**values), values, base_profile)
            state = _state_for(cfg, best_state, profile, dims)
        except (TypeError, ValueError) as exc:
            points.append(SweepPoint(
                values, None, None, True, str(exc), invalid=True))
            continue
        changed = profile.compile_profile_sha256() != base_profile.compile_profile_sha256()
        record = evaluator(state)
        points.append(SweepPoint(
            values, state, record, changed,
            reason=(
                None if record is None or record.successful
                else (record.failure or f"候选执行失败（{record.status}）")),
            objective_direction="minimize",
            resource_direction=cfg.sweep.pareto.direction,
            result_maturity=(
                "model_prediction" if record is None else record.result_maturity),
            source_origins=_source_origins(profile),
            load_key=_load_key(
                state, baseline_profile=base_profile, inputs=inputs)))
    sensitivity = _sensitivity(points, base_profile)
    knees = _knees(points, cfg.sweep.knee_improvement_threshold)
    pareto = _pareto(points, cfg.sweep.pareto.resource,
                     cfg.sweep.pareto.direction)
    return SweepResult(tuple(points), sensitivity, pareto, knees,
                       binding=_binding(cfg, inputs, best))


def write_sweep(result: SweepResult, cfg: TuneConfig, out_dir: Path) -> dict:
    """落盘原始数据、敏感性、Pareto 图与报告。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "sweep.jsonl"
    jsonl.write_text("\n".join(
        json.dumps(point.to_json(), ensure_ascii=False) for point in result.points) + "\n")
    (out_dir / "sensitivity.json").write_text(
        json.dumps({"fields": result.sensitivity, "knees": result.knees,
                    "pareto": [point.to_json() for point in result.pareto],
                    "binding": result.binding},
                   indent=2, ensure_ascii=False))
    _write_pareto_csv(result.pareto, out_dir / "pareto.csv",
                      cfg.sweep.pareto.resource)
    _write_pareto_svg(result.pareto, out_dir / "pareto.svg", cfg)
    (out_dir / "report.md").write_text(_report(result, cfg, out_dir))
    return {
        "sweep": str(jsonl),
        "sensitivity": str(out_dir / "sensitivity.json"),
        "pareto_csv": str(out_dir / "pareto.csv"),
        "pareto_svg": str(out_dir / "pareto.svg"),
        "report": str(out_dir / "report.md"),
    }


# 扫描新值必须自带完整来源台账；画像里没有登记过的字段用这里的物理单位
# 与作用域，避免写入空串后被 GeneSim 的契约校验拒绝。
_SOURCE_UNITS = {
    "num_pim_devices": "device",
    "clusters_per_device": "core",
    "tensors_per_cluster": "tensor pu",
    "cluster_memory_bytes": "byte",
    "sys_reserve_bytes": "byte",
    "wram_bytes_per_dpu": "byte",
    "dma_align": "byte",
    "clock_period_ns": "ns",
    "multipliers": "count",
    "cycle_costs": "cycle",
    "zone_cycle_factors": "ratio",
    "tile_size": "rows",
    "mac_width": "lanes",
    "chunk_size": "elements",
    "max_bulk_transfer_bytes": "byte",
    "use_long_loop_counts": "mode",
    "inter_cluster_communication": "mode",
    "host_relay_overhead_ns": "ns",
    "links": "gbps",
    "cpu_model": "mixed",
}
_SOURCE_SCOPES = {
    "num_pim_devices": "per system",
    "clusters_per_device": "per device",
    "tensors_per_cluster": "per cluster",
    "cluster_memory_bytes": "per cluster",
    "sys_reserve_bytes": "per cluster",
    "wram_bytes_per_dpu": "per dpu",
    "dma_align": "per dma",
    "clock_period_ns": "per cycle",
    "multipliers": "per tensor pu",
    "cycle_costs": "per work unit",
    "zone_cycle_factors": "region",
    "tile_size": "per tile",
    "mac_width": "per instruction",
    "chunk_size": "per chunk",
    "max_bulk_transfer_bytes": "per dma",
    "use_long_loop_counts": "per dynamic loop",
    "inter_cluster_communication": "inter-cluster",
    "host_relay_overhead_ns": "per relayed transfer",
    "links": "link",
    "cpu_model": "host cpu",
}

_SOURCE_KEY_OF = {
    "num_pim_devices": "system.num_pim_devices",
    "clusters_per_device": "pim.clusters_per_device",
    "tensors_per_cluster": "pim.tensors_per_cluster",
    "cluster_memory_bytes": "ceva.local_memory_bytes",
    "sys_reserve_bytes": "ceva.sys_reserve_bytes",
    "wram_bytes_per_dpu": "pim.compile.wram_bytes_per_dpu",
    "dma_align": "pim.compile.dma_align",
    "clock_period_ns": "pim.clock_period_ns",
    "multipliers": "pim.tensor_pu_config.multipliers",
    "cycle_costs": "pim.pim_trace_cycle_costs",
    "zone_cycle_factors": "pim.zone_cycle_factors",
    "tile_size": "pim.compile.tile_size",
    "mac_width": "pim.compile.mac_width",
    "chunk_size": "pim.compile.chunk_size",
    "max_bulk_transfer_bytes": "pim.compile.max_bulk_transfer_bytes",
    "use_long_loop_counts": "pim.compile.use_long_loop_counts",
    "host_relay_overhead_ns": "comm_config.host_relay_overhead_ns",
    "links": "comm_config.links",
    "cpu_model": "cpu_model",
    "inter_cluster_communication": "pim.inter_cluster_communication",
}


def _source_origins(profile) -> tuple[str, ...]:
    """扫描点的证据来源集合；来源不同的点不放在同一个 Pareto 组里比较。"""
    return tuple(sorted({
        str(entry.get("origin", ""))
        for entry in profile.parameter_sources.values()}))


def _load_key(state: TuneState, *, baseline_profile=None,
              inputs=None) -> str:
    """同负载分组：段数与算子参数之外，还要绑定输入内容与基线画像。

    只用 PP/tasklet 分组会把另一模型、IR、轨迹或画像的 best 与当前扫描
    误认为同一负载。
    """
    parts = [f"pp{state.strategy.num_stages}_tl{state.op.num_tasklets}"]
    if baseline_profile is not None:
        parts.append(f"sig{baseline_profile.profile_sha256()[:12]}")
    if inputs is not None:
        parts.extend((
            f"model{inputs.model_sha256[:12]}",
            f"ir{inputs.ir_sha256[:12]}",
            f"trace{inputs.trace_sha256[:12]}",
            f"seq{inputs.max_seq}",
            f"req{len(inputs.trace.requests)}",
        ))
    return "_".join(parts)


def _binding(cfg: TuneConfig, inputs, best: Mapping[str, Any] | None) -> dict | None:
    """报告里保留 best 与本次扫描配置的绑定证据。"""
    if best is None:
        return None
    recorded = Path(str(best.get("effective_config", "")))
    return {
        "best_candidate_id": best.get("candidate_id"),
        "best_score": best.get("score"),
        "best_status": best.get("status"),
        "effective_config": str(recorded),
        # 生成时摘要与当前文件摘要分开记录，扫描不能改写历史语义。
        "effective_config_sha256": best.get("effective_config_sha256"),
        "effective_config_current_sha256": (
            file_sha256(recorded) if recorded.is_file() else None),
        "compile_key": best.get("compile_key"),
        "sim_key": best.get("sim_key"),
        "inputs": None if inputs is None else inputs.to_json(),
        "baseline_profile_sha256": cfg.profile.profile_sha256(),
    }


# 通用与 CEVA 描述同一物理量的键；按后端只登记实际消费的那一份。
_GENERIC_TO_CEVA_SOURCE = {
    "pim.cluster_memory_capacity_gb": "ceva.local_memory_bytes",
    "pim.sys_reserve_bytes": "ceva.sys_reserve_bytes",
}


def _flatten_override_leaves(mapping: Mapping[str, Any],
                             prefix: str = "") -> dict[str, Any]:
    """把 genesim_overrides 展平成叶子路径；列表整体作为一个叶子。"""
    leaves: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            leaves.update(_flatten_override_leaves(value, path))
        else:
            leaves[path] = value
    return leaves


def _field_override_leaves(base, field: str, value: Any) -> set[str]:
    """一个画像字段实际改变的下发叶子；与 genesim_overrides 同源。"""
    try:
        candidate = base.with_fields(**{field: value})
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None:
        before = _flatten_override_leaves(base.genesim_overrides())
        after = _flatten_override_leaves(candidate.genesim_overrides())
        changed = {
            key for key in set(before) | set(after)
            if not _values_equal(before.get(key), after.get(key))}
        if changed:
            return changed
    # 编译预算字段（WRAM/DMA 对齐）不下发 GeneSim，保留声明台账键。
    fallback = _SOURCE_KEY_OF.get(field)
    return {fallback} if fallback else set()


def _mark_scanned(profile, values: Mapping[str, Any], base):
    """只有真正偏离基准的生效字段才把来源降为 assumption。

    扫描字段到来源键的映射从 `genesim_overrides()` 的叶子差异派生：通用
    容量/预留落在 `pim.*`，CEVA 落在 `ceva.*`，周期表按原语叶子更新；
    未变化的字段保留原证据。来源键必须精确到这个生效叶子：父键只能表示
    整组语义字段本身，不能用它承接变化子叶子而抹掉同组未扫描字段的证据。
    """
    sources = {
        key: dict(entry) for key, entry in profile.parameter_sources.items()}
    changed_fields = {
        field: value for field, value in values.items()
        if not _values_equal(value, getattr(base, field, None))}
    if not changed_fields:
        return profile
    planned: dict[str, list[str]] = {}
    for field, value in changed_fields.items():
        leaves = _field_override_leaves(base, field, value)
        for leaf in sorted(leaves):
            if profile.ceva is not None and leaf in _GENERIC_TO_CEVA_SOURCE:
                counterpart = _GENERIC_TO_CEVA_SOURCE[leaf]
                if counterpart in leaves or counterpart in sources:
                    continue
            planned.setdefault(leaf, []).append(field)
    for key, fields in sorted(planned.items()):
        previous = (
            sources.get(key) if isinstance(sources.get(key), Mapping) else {})
        field = fields[0]
        sources[key] = {
            "origin": "assumption",
            "evidence": (
                f"硬件扫描新值 {'、'.join(sorted(set(fields)))}，"
                "无适用实测证据"),
            "unit": previous.get("unit") or _SOURCE_UNITS.get(field, "value"),
            "scope": previous.get("scope") or _SOURCE_SCOPES.get(
                field, "profile"),
        }
    return replace(profile, parameter_sources=sources)


def _state_for(cfg: TuneConfig, best: TuneState, profile, dims) -> TuneState:
    """按核策略调整 DPU 数，其余层固定为最优候选。"""
    num_dpus = best.strategy.num_dpus
    if cfg.sweep.core_policy == "use_all_clusters":
        num_dpus = profile.total_clusters
    strategy = replace(best.strategy, num_dpus=num_dpus)
    state = TuneState(strategy, best.op, profile, best.optimization)
    # 物理资源组合能在这里判定时先标 invalid，别把昂贵编译当探针。
    profile.validate_dpus(num_dpus)
    build_strategy(state, dims)  # 非法 TP 在这里报错
    return state


def _sensitivity(points: list[SweepPoint], base_profile) -> dict:
    fields = {}
    successful = [point for point in points if point.score is not None]
    baseline = next(
        (point for point in successful
         if point.state is not None
         and point.state.profile.profile_sha256()
         == base_profile.profile_sha256()),
        None)
    baseline_source = "exact_profile"
    if baseline is None and successful:
        # 基准画像的取值不在扫描列表里时，用第一个成功点作参考并在报告里注明。
        baseline = successful[0]
        baseline_source = "first_point"
    baseline_score = baseline.score if baseline else None
    for key in (points[0].values if points else {}):
        by_value: dict[tuple, list[tuple[Any, float]]] = {}
        for point in successful:
            if key not in point.values:
                continue
            value = point.values[key]
            by_value.setdefault(_sort_key(value), []).append((value, point.score))
        entries = []
        for sort_key in sorted(by_value):
            pairs = by_value[sort_key]
            scores = [score for _, score in pairs]
            entries.append({"value": pairs[0][0],
                            "mean_score": sum(scores) / len(scores)})
        if len(entries) >= 2 and _numeric_values(entries):
            deltas = []
            elasticities = []
            for previous, current in zip(entries, entries[1:]):
                dx = current["value"] - previous["value"]
                delta = current["mean_score"] - previous["mean_score"]
                deltas.append({
                    "from": previous["value"], "to": current["value"],
                    "delta_time": delta,
                    "delta_over_base": (None if baseline_score in (None, 0)
                                        else delta / baseline_score),
                    "per_unit": None if dx == 0 else delta / dx,
                })
                if previous["value"] != 0 and baseline_score not in (None, 0):
                    elasticities.append({
                        "from": previous["value"], "to": current["value"],
                        "elasticity": (delta / baseline_score)
                        / (dx / previous["value"]),
                    })
            fields[key] = {"kind": "continuous", "values": entries,
                           "deltas": deltas, "elasticity": elasticities,
                           "baseline_score": baseline_score,
                           "baseline_source": baseline_source}
        else:
            diffs = [
                {"value": entry["value"],
                 "mean_score": entry["mean_score"],
                 "difference_from_first": entry["mean_score"]
                 - entries[0]["mean_score"]}
                for entry in entries
            ]
            fields[key] = {"kind": "discrete", "values": diffs}
    return fields


def _sort_key(value: Any) -> tuple:
    if _is_number(value):
        return (0, float(value), "")
    return (1, 0.0, str(value))


def _knees(points: list[SweepPoint], threshold: float) -> dict:
    """资源方向明确且相邻翻倍时，改善不足阈值处记为拐点。

    笛卡尔积里必须先把其他字段固定成同一组再比较，否则会把不同时钟或链路
    配置的分数当成相邻点，算出负改善。
    """
    knees = {}
    for field in RESOURCE_FIELDS:
        groups: dict[tuple, list[tuple[float, float]]] = {}
        for point in points:
            value = point.values.get(field)
            if (point.score is None or value is None
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))):
                continue
            others = tuple(sorted(
                (key, str(item)) for key, item in point.values.items()
                if key != field))
            groups.setdefault(others, []).append((float(value), point.score))
        for rows in groups.values():
            rows.sort(key=lambda item: item[0])
            for (previous_x, previous), (current_x, current) in zip(rows, rows[1:]):
                if previous_x <= 0 or current_x != previous_x * 2:
                    continue
                improvement = (previous - current) / previous
                if improvement < threshold:
                    knees[field] = {"value": current_x,
                                    "improvement": improvement}
                    break
            if field in knees:
                break
    return knees


def _pareto(points: list[SweepPoint], resource: str,
            resource_direction: str = "minimize") -> tuple[SweepPoint, ...]:
    """按配置的资源方向计算支配关系；不同负载/来源/成熟度组间不比较。"""
    if resource_direction not in ("minimize", "maximize"):
        raise ValueError(f"未知的 Pareto 资源方向：{resource_direction!r}")
    viable = [
        point for point in points
        if point.score is not None and point.state is not None
    ]

    def axes(point):
        return (point.score, _resource_of(point.state.profile, resource))

    def dominates(first: SweepPoint, second: SweepPoint) -> bool:
        px, py = axes(first)
        ox, oy = axes(second)
        if resource_direction == "minimize":
            resource_no_worse = py <= oy
            resource_better = py < oy
        else:
            resource_no_worse = py >= oy
            resource_better = py > oy
        return (px <= ox and resource_no_worse
                and (px < ox or resource_better))

    groups: dict[tuple, list[SweepPoint]] = {}
    for point in viable:
        groups.setdefault(
            (point.load_key, point.source_origins, point.result_maturity),
            []).append(point)
    front = []
    for group in groups.values():
        for point in group:
            if not any(dominates(other, point)
                       for other in group if other is not point):
                front.append(point)
    return tuple(sorted(front, key=lambda point: (
        point.score, _resource_of(point.state.profile, resource))))


def _resource_of(profile, resource: str) -> float:
    if resource not in PARETO_RESOURCES:
        raise ValueError(f"未知的 Pareto 资源轴：{resource!r}")
    if resource == "total_physical_memory_bytes":
        return float(profile.num_pim_devices * profile.clusters_per_device
                     * profile.cluster_memory_bytes)
    return float(profile.num_pim_devices * profile.clusters_per_device)


def _write_pareto_csv(front, path: Path, resource: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "total_time_s", "resource", "values"])
        for point in front:
            writer.writerow([
                point.record.candidate_id, f"{point.score:.12g}",
                _resource_of(point.state.profile, resource),
                json.dumps(dict(point.values), sort_keys=True)])


def _write_pareto_svg(front, path: Path, cfg: TuneConfig) -> None:
    width, height, margin = 640, 420, 60
    xs = [_resource_of(point.state.profile, cfg.sweep.pareto.resource)
          for point in front] or [0.0, 1.0]
    ys = [point.score for point in front] or [0.0, 1.0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = (xmax - xmin) or 1.0
    span_y = (ymax - ymin) or 1.0
    circles = []
    for point in front:
        x = _resource_of(point.state.profile, cfg.sweep.pareto.resource)
        y = point.score
        cx = margin + (x - xmin) / span_x * (width - 2 * margin)
        cy = height - margin - (y - ymin) / span_y * (height - 2 * margin)
        circles.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#2b6cb0"/>'
            f'<text x="{cx + 8:.1f}" y="{cy - 8:.1f}" font-size="11">'
            f'{point.record.candidate_id}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="black"/>'
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" '
        f'stroke="black"/>'
        f'<text x="{width / 2}" y="{height - 12}" font-size="12" '
        f'text-anchor="middle">{cfg.sweep.pareto.resource}</text>'
        f'<text x="14" y="{height / 2}" font-size="12" '
        f'transform="rotate(-90 14 {height / 2})">total_time_s</text>'
        + "".join(circles) + "</svg>")


def _report(result: SweepResult, cfg: TuneConfig, out_dir: Path) -> str:
    lines = [
        f"# 硬件敏感性报告：{cfg.run_name}",
        "",
        f"- 扫描字段：{', '.join(sorted(cfg.sweep.fields))}",
        f"- 核策略：{cfg.sweep.core_policy}",
        f"- Pareto 资源轴：{cfg.sweep.pareto.resource}",
        f"- 拐点阈值：{cfg.sweep.knee_improvement_threshold}",
    ]
    binding = result.binding or {}
    if binding:
        lines.extend([
            f"- best 候选：{binding.get('best_candidate_id')}",
            f"- best 分数：{binding.get('best_score')}",
            f"- best 生效配置：{binding.get('effective_config')}"
            f"（sha256 {binding.get('effective_config_sha256')}）",
        ])
    lines.extend([
        "",
        "## 扫描点",
        "",
        "| 取值 | 状态 | total_time_s | 重编 | 参数来源 | 结果成熟度 | 原因 |",
        "|---|---|---:|---|---|---|---|",
    ])
    for point in result.points:
        score = "" if point.score is None else f"{point.score:.9f}"
        sources = ",".join(point.source_origins) or "-"
        lines.append(
            f"| {json.dumps(dict(point.values), sort_keys=True)} | {point.status} "
            f"| {score} | {'是' if point.compile_fields_changed else '否'} "
            f"| {sources} | {point.result_maturity} | {point.reason or ''} |")
    lines.extend(["", "## Pareto 前沿", ""])
    for point in result.pareto:
        lines.append(f"- {point.record.candidate_id}: total_time_s={point.score:.9f}，"
                     f"资源={_resource_of(point.state.profile, cfg.sweep.pareto.resource)}，"
                     f"来源={','.join(point.source_origins) or '-'}，"
                     f"成熟度={point.result_maturity}")
    lines.extend([
        "",
        "## 边界",
        "",
        "- 固定项：模型、请求窗口、图策略与算子参数沿用最优候选。",
        "- 重编项：改变核数、容量或编译预算的扫描点必须重编，不能复用旧 PIM MLIR。",
        f"- 原始扫描数据：{Path(out_dir) / 'sweep.jsonl'}。",
        f"- Pareto 资源方向：{cfg.sweep.pareto.direction}；"
        "不同负载、参数来源与结果成熟度分组内比较。",
        "- 参数来源与结果成熟度逐点记录；相关性不等于因果。",
    ])
    return "\n".join(lines) + "\n"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_values(entries: list[dict]) -> bool:
    return all(_is_number(entry["value"]) for entry in entries)
