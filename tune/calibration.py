"""代价模型校准：样本契约、分组统计与结果分级。

不更换计时公式，只把独立观测与预测值对齐报告。缺少留出观测、用户阈值、
产物绑定或仍在适用范围外时，结论只能是模型预测，不能生成 passed。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from tune.inputs import file_sha256

KINDS = frozenset({"compute", "dma", "host", "communication"})
PHASES = frozenset({"prefill", "decode"})
SOURCES = frozenset({"ceva_simulator", "real_hardware"})
# 与 GeneSim `CEVA_VALIDATION_SCHEMA_VERSION` 同一契约版本；报告只有满足
# schema 11 的产物绑定与适用范围，GeneSim 的成熟度入口才会消费它。
REPORT_SCHEMA_VERSION = 11
CEVA_EXECUTION_SCOPE = "npm_core"
# 允许升级「范围内已验证」的参数来源等级，与 GeneSim 同一条契约。
CALIBRATED_ORIGINS = frozenset({"customer", "ceva_measured", "fitted"})
REQUIRED_COST_MODEL_KEYS = (
    "mac_width", "tile_size", "chunk_size", "max_bulk_transfer_bytes",
    "multipliers", "cpu_model")
STATISTIC_FIELDS = ("median", "p90", "max")
# 把拟合系数写回真正的计时字段。分组没有可写字段时显式失败，避免又退回
# 「只加一个来源标签」的假校准。
_DMA_COST_KEYS = frozenset({
    "LOAD", "STORE", "LOADN", "LOADN_PER_BYTE", "STOREN",
    "STOREN_PER_BYTE", "SEND",
})
_COMPUTE_COST_KEYS = frozenset({
    "MAC", "CONTEXT_MAC", "SCORE_MAC", "MUL", "EXP", "DIV", "MAX", "GELU",
    "SILU", "RSQRT", "LUT", "ROPE", "QUANT", "GATHER", "TRANSPOSE",
    "RESHAPE", "CONCAT", "SPLIT", "MASK", "SOFTMAX_CHUNK",
})
_FIT_KIND_OF_GROUP = {
    "dma": "dma",
    "compute": "compute",
    "host": "host",
    "communication": "communication",
}


@dataclass(frozen=True)
class CalibrationSample:
    """一条逐点观测：目标配置、形状、预测值与实测值。"""

    sample_id: str
    dataset_id: str
    source: str
    phase: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    predicted: float
    observed: float
    target: Mapping[str, Any]
    unit: str = "ns"
    # 可选测量特征：DMA 样本带 operator/transfer_count/total_bytes 时，
    # 直接复用 GeneSim 的 `fit_dma_operator` 拟合真实搬运模型。
    features: Mapping[str, Any] = field(default_factory=dict)
    # 产物绑定：算子类型与 PIM MLIR 里的相位原语。生成 schema 11 的
    # scope.validation 时，它们与运行侧产物逐项核对。
    operator: str = ""
    artifact_phase: str = ""

    @property
    def error(self) -> float:
        return self.observed - self.predicted

    @property
    def relative_error(self) -> float:
        return abs(self.error) / self.observed

    @property
    def identity(self) -> tuple:
        """同一条测点的完整身份；与 GeneSim 共享契约同一实现。"""
        return _genesim_validation().measurement_identity({
            "target": dict(self.target),
            "features": dict(self.features),
            "phase": self.phase,
            "kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "unit": self.unit,
            "source": self.source,
            "operator": self.operator,
            "artifact_phase": self.artifact_phase,
        })


def target_config(scope: Mapping[str, Any]) -> dict | None:
    """样本必须声明的目标配置身份；与 GeneSim 共享契约同一实现。"""
    return _genesim_validation().target_config(scope)


def load_samples(path: Path) -> list[CalibrationSample]:
    """读取 JSONL 样本并逐条校验字段。"""
    samples = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"样本第 {number} 行不是合法 JSON：{exc}") from exc
        samples.append(_sample_from_json(raw, number))
    if not samples:
        raise ValueError(f"{path} 里没有样本")
    return samples


def split_datasets(samples: Iterable[CalibrationSample], *,
                   training_dataset: str,
                   validation_dataset: str) -> tuple[list, list]:
    """按数据集名分离训练与留出，并按测点身份拒绝两用。"""
    training = [s for s in samples if s.dataset_id == training_dataset]
    heldout = [s for s in samples if s.dataset_id == validation_dataset]
    if not training:
        raise ValueError(f"没有训练集 {training_dataset!r} 的样本")
    if not heldout:
        raise ValueError(f"没有留出集 {validation_dataset!r} 的样本")
    seen = {sample.identity for sample in training}
    duplicates = [sample.sample_id for sample in heldout
                  if sample.identity in seen]
    if duplicates:
        raise ValueError(
            "训练集与留出集存在同一测点（改名两用）："
            f"{sorted(set(duplicates))[:5]}")
    return training, heldout


def group_statistics(samples: Iterable[CalibrationSample]) -> dict:
    """按阶段与类别分组，复算带符号误差与绝对相对误差的统计量。"""
    groups: dict[tuple[str, str], list[CalibrationSample]] = {}
    for sample in samples:
        groups.setdefault((sample.phase, sample.kind), []).append(sample)
    return {
        f"{phase}/{kind}": _statistics(items)
        for (phase, kind), items in sorted(groups.items())
    }


def fit_parameters(training: Iterable[CalibrationSample]) -> dict:
    """只用训练集拟合共享计时参数；系数由 GeneSim 唯一契约复算。

    画像里的周期、主机与链路字段是全局的，没有 prefill/decode 分阶段接口。
    因此同一类别的两个阶段必须用联合样本得到一个参数组；报告仍按阶段
    复算统计量，但写回只发生一次，禁止把两组拟合连乘到同一字段。
    """
    evidence = _genesim_validation()
    return evidence.fit_training_points(
        [_training_record(sample) for sample in training])


def _training_record(sample: CalibrationSample) -> dict:
    """把一条训练观测转换成报告里保存的拟合输入。"""
    return {
        "sample_id": sample.sample_id,
        "dataset_id": sample.dataset_id,
        "source": sample.source,
        "phase": sample.phase,
        "kind": sample.kind,
        "shape": list(sample.shape),
        "dtype": sample.dtype,
        "unit": sample.unit,
        "predicted": sample.predicted,
        "observed": sample.observed,
        "target": json.loads(json.dumps(dict(sample.target), default=str)),
        "features": json.loads(json.dumps(dict(sample.features), default=str)),
        "operator": sample.operator,
        "artifact_phase": sample.artifact_phase,
    }


def _require_cycle_units(samples: Iterable[CalibrationSample],
                         kind: str) -> None:
    """DMA 拟合直接写周期字段，纳秒样本必须先换算，不能默认当周期。"""
    offenders = [sample.sample_id for sample in samples
                 if sample.unit != "cycle"]
    if offenders:
        raise ValueError(
            f"{kind} 样本的单位必须是 cycle（纳秒需先按画像时钟换算）："
            f"{sorted(offenders)[:5]}")


def predict_with_frozen_parameters(
    samples: Iterable[CalibrationSample], fit: Mapping[str, Mapping[str, Any]],
) -> list[CalibrationSample]:
    """用冻结系数为留出集生成预测；缺对应分组直接拒绝。"""
    evidence = _genesim_validation()
    frozen = []
    for sample in samples:
        key = f"{sample.phase}/{sample.kind}"
        if key not in fit:
            raise ValueError(
                f"留出集分组 {key} 没有训练集拟合参数，不能生成独立验证")
        frozen.append(replace(
            sample,
            predicted=evidence.predict_point(_training_record(sample), fit[key])))
    return frozen


def build_report(
    samples: Iterable[CalibrationSample],
    *,
    training_dataset: str,
    validation_dataset: str,
    artifacts: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, float] | None = None,
    scope: Mapping[str, Any] | None = None,
    package: Mapping[str, Any] | None = None,
    simulation_summary: Path | str | None = None,
) -> dict:
    """生成 schema 11 校准报告；缺任一证据时状态只能是 report_only。

    报告结构、逐点证据与统计公式全部复用 GeneSim 的 `ceva_validation`
    契约。`artifacts` 必须登记真实文件路径与内容哈希，`package` 的哈希
    必须与这些文件一致，`simulation_summary` 必须是本轮真实 pipeline
    仿真的摘要；任何一项缺失或伪造都只能保持模型预测。
    """
    training, heldout = split_datasets(
        samples, training_dataset=training_dataset,
        validation_dataset=validation_dataset)
    # 拟合前先核对训练与留出、当前范围是否同一目标/精度/单位：跨域样本
    # 不能先混合拟合，再靠误差统计升级为独立验证。
    compatibility_error = _genesim_validation().report_applicability_error(
        [_training_record(sample) for sample in training],
        _point_records(heldout, heldout), scope)
    if compatibility_error is not None:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "report_only",
            "training_dataset": training_dataset,
            "validation_dataset": validation_dataset,
            "training_samples": len(training),
            "validation_samples": len(heldout),
            "fit": {},
            "points": [],
            "reason": compatibility_error,
        }
    try:
        fit = fit_parameters(training)
        frozen = predict_with_frozen_parameters(heldout, fit)
    except ValueError as exc:
        # 训练集没有覆盖留出测点的身份时，冻结参数不适用于该点；报告只能
        # 保持 report_only，不能用未覆盖的模型给出通过结论。
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "report_only",
            "training_dataset": training_dataset,
            "validation_dataset": validation_dataset,
            "training_samples": len(training),
            "validation_samples": len(heldout),
            "fit": {},
            "points": [],
            "reason": f"拟合覆盖不足：{exc}",
        }
    statistics = group_statistics(frozen)
    evidence = _genesim_validation()
    verified_artifacts, artifact_error = evidence.verify_artifacts(artifacts)
    verified_package, package_error = evidence.verify_package(
        package, verified_artifacts)
    generated_scope, scope_error = _build_scope_evidence(
        scope, frozen, validation_dataset)
    backend_evidence, backend_error = _simulation_backend_evidence(
        simulation_summary, verified_package if package_error is None else None)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "report_only",
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "training_samples": len(training),
        "validation_samples": len(heldout),
        # 冻结参数、训练集拟合输入与用户阈值都随报告落盘：复核时重算拟合、
        # 统计和阈值判定，不依赖生成时的进程内检查。
        "fit": fit,
        "training_points": [_training_record(sample) for sample in training],
        "points": _point_records(heldout, frozen),
        "statistics": statistics,
        "thresholds": dict(thresholds or {}),
        "artifacts": verified_artifacts,
        "package": verified_package,
        "scope": generated_scope,
        "backends": backend_evidence,
    }
    if artifact_error is not None:
        report["reason"] = artifact_error
        return report
    if package_error is not None:
        report["reason"] = package_error
        return report
    if scope_error is not None:
        report["reason"] = scope_error
        return report
    scope_error = _scope_evidence_error(report["scope"])
    if scope_error is not None:
        report["reason"] = scope_error
        return report
    evidence_error = _generated_evidence_error(report["scope"])
    if evidence_error is not None:
        report["reason"] = evidence_error
        return report
    if backend_error is not None:
        report["reason"] = backend_error
        return report
    threshold_error = _genesim_validation().threshold_table_error(
        statistics, thresholds)
    if threshold_error is not None:
        report["reason"] = threshold_error
        return report
    report["status"] = "passed"
    report["reason"] = "逐点证据、产物绑定、真实仿真后端与全部分组阈值均满足"
    # 生成后立即用与成熟度复核相同的入口重算一遍；任何一项不能复算
    # 都回到 report_only，避免「生成时检查过」变成不可复核的口头证据。
    evidence_error = _genesim_validation().report_evidence_error(
        report, require_host_and_communication=True)
    if evidence_error is not None:
        report["status"] = "report_only"
        report["reason"] = evidence_error
    return report


def _build_scope_evidence(
    scope: Mapping[str, Any] | None,
    frozen: Iterable[CalibrationSample],
    sample_set: str,
) -> tuple[dict, str | None]:
    """把留出观测转换成 GeneSim 契约里的三类验证证据。"""
    if not isinstance(scope, Mapping) or not scope:
        return {}, "缺少适用范围，不能升级为独立验证"
    evidence = _genesim_validation()
    data = json.loads(json.dumps(dict(scope)))
    validation_points = []
    host_records = []
    communication_records = []
    for sample in frozen:
        if sample.kind in ("compute", "dma"):
            if not sample.operator or not sample.artifact_phase:
                return {}, (
                    f"样本 {sample.sample_id} 缺少 operator/artifact_phase，"
                    "无法与算子编译产物逐点绑定")
            validation_points.append(evidence.build_point(
                operator=sample.operator,
                phase=sample.artifact_phase,
                kind=sample.kind,
                stage=sample.phase,
                shape=sample.shape,
                dtype=sample.dtype,
                predicted=sample.predicted,
                observed=sample.observed,
            ))
        elif sample.kind == "host":
            if not sample.operator:
                return {}, f"主机样本 {sample.sample_id} 缺少主机算子类型"
            point = evidence.build_point(
                operator=sample.operator, phase="host", kind="host",
                stage=sample.phase, shape=(1,), dtype=sample.dtype,
                predicted=sample.predicted, observed=sample.observed)
            host_records.append({
                key: point[key] for key in (
                    "operator", "stage", "predicted", "observed",
                    "error", "relative_error")})
        else:
            if not sample.operator:
                return {}, f"通信样本 {sample.sample_id} 缺少链路标识"
            point = evidence.build_point(
                operator=sample.operator, phase="communication",
                kind="communication", stage=sample.phase, shape=(1,),
                dtype=sample.dtype, predicted=sample.predicted,
                observed=sample.observed)
            communication_records.append({
                "link": point["operator"],
                "stage": point["stage"],
                "predicted": point["predicted"],
                "observed": point["observed"],
                "error": point["error"],
                "relative_error": point["relative_error"],
            })
    if validation_points:
        data["validation"] = evidence.build_evidence(
            f"validation:{sample_set}", validation_points)
    if host_records:
        host_evidence = evidence.build_evidence(
            f"host:{sample_set}", host_records)
        host_evidence["op_types"] = sorted(
            {record["operator"] for record in host_records})
        data["host_orchestration"] = host_evidence
    if communication_records:
        data["communication_validation"] = evidence.build_evidence(
            f"communication:{sample_set}", communication_records)
    return data, None


def _generated_evidence_error(scope: Mapping[str, Any]) -> str | None:
    """用 GeneSim 的唯一契约核对三类逐点证据。"""
    evidence = _genesim_validation()
    return (
        evidence.validation_evidence_error(scope.get("validation"))
        or evidence.host_evidence_error(scope.get("host_orchestration"))
        or evidence.communication_evidence_error(
            scope.get("communication_validation"))
    )


@lru_cache(maxsize=1)
def _genesim_validation():
    """加载 GeneSim 的验证报告构造与校验模块（唯一真源）。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    import ceva_validation

    return ceva_validation


def _simulation_backend_evidence(
    path: Path | str | None,
    package: Mapping[str, Any] | None = None,
) -> tuple[dict, str | None]:
    """验证后端必须来自本轮真实 pipeline 摘要，且经运行清单绑定产物。

    单个带 `result_producer`/`total_time_s` 的 JSON 不构成运行归属：
    摘要必须指向 GeneSim 写出的运行清单，清单里本轮消费的模型 IR、
    放置 sidecar、请求轨迹与成本 sidecar 哈希必须与报告 package 一致。
    """
    if path is None:
        return {}, "缺少真实 pipeline 仿真摘要，无法证明验证后端"
    summary_path = Path(path)
    if not summary_path.is_file():
        return {}, f"仿真摘要不存在：{summary_path}"
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"仿真摘要无法解析：{exc}"
    if not isinstance(summary, Mapping):
        return {}, "仿真摘要必须是 JSON 对象"
    producer = summary.get("result_producer")
    if producer != "pipeline_simulator":
        return {}, f"仿真结果的来源不是 pipeline：{producer!r}"
    if summary.get("simulation_backend") not in (None, "pipeline"):
        return {}, "仿真摘要的实际后端不是 pipeline"
    total = _finite(summary.get("total_time_s"))
    if total is None or total <= 0:
        return {}, "仿真摘要缺少有限正的总时延"
    evidence = _genesim_validation()
    manifest_error, manifest = evidence.run_manifest_evidence(
        summary_path, summary)
    if manifest_error is not None:
        return {}, manifest_error
    if package is None:
        return {}, "缺少产物绑定 package，无法证明仿真消费了本轮产物"
    binding_error = evidence.run_manifest_binding_error(manifest, package)
    if binding_error is not None:
        return {}, binding_error
    digest = file_sha256(summary_path)
    manifest_path = Path(manifest["path"])
    return {
        "validation": "pipeline",
        "source": {
            "path": str(summary_path),
            "sha256": digest,
            "result_producer": producer,
            "total_time_s": total,
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": file_sha256(manifest_path),
        },
    }, None


def profile_with_fit(profile_json: Mapping[str, Any],
                     fit: Mapping[str, Mapping[str, Any]]) -> dict:
    """只把训练集实际覆盖的字段写回画像，并登记逐字段证据。

    拟合结果按测点身份（算子/原语、主机类型、链路名）保存；一个身份的
    缩放系数只能写回该身份声明的字段。单点实测不会扩展到未测的 SILU、
    其他链路或其他主机类型；没有可归属字段的身份直接跳过，全部身份都
    没有字段时整次写回拒绝。host 计时字段是全局的，不同主机类型必须给出
    同一冻结系数。
    """
    data = json.loads(json.dumps(dict(profile_json)))
    sources = {
        key: dict(entry)
        for key, entry in (data.get("parameter_sources") or {}).items()}
    cycle_costs = dict(data.get("cycle_costs") or {})
    cpu_model = dict(data.get("cpu_model") or {})
    links = [dict(link) for link in data.get("links") or []]
    link_index = {str(link.get("name")): link for link in links}
    # 先按最终画像字段汇总所有测点，再以原画像值为基数一次性写回；
    # 同一字段出现不同冻结值必须在修改任何画像内容前拒绝，不能连乘或覆盖。
    planned: dict[tuple[str, str], dict[str, Any]] = {}
    for kind, parameters, groups in _shared_fit_groups(fit):
        targets = parameters.get("targets")
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError(
                f"拟合类别 {kind} 缺少测点身份，无法确定写回范围")
        phase_text = "、".join(groups)
        if kind == "host":
            _plan_host_target(planned, targets, cpu_model, kind, phase_text)
            continue
        for identity, target in sorted(targets.items()):
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"拟合类别 {kind} 的测点 {identity} 不是对象")
            fields = target.get("fields")
            if not isinstance(fields, list):
                raise ValueError(
                    f"拟合类别 {kind} 的测点 {identity} 缺少可写字段声明")
            if not fields:
                # 测点没有可归属的画像字段：不写回，也不扩散到其他字段。
                continue
            method = target.get("method")
            if method == _genesim_validation().FIT_METHOD_DMA:
                _plan_dma_target(
                    planned, kind, identity, target, cycle_costs, phase_text)
            elif method == _genesim_validation().FIT_METHOD_SCALE:
                scale = float(target.get("scale", math.nan))
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        f"拟合类别 {kind} 的测点 {identity} 缩放系数不是正数："
                        f"{scale!r}")
                evidence = (
                    f"训练集测点 {identity} 最小二乘缩放 scale={scale:.9g}"
                    f"（{int(target.get('count', 0))} 个样本；覆盖 {phase_text}）")
                for field in fields:
                    if kind in ("dma", "compute"):
                        if field not in cycle_costs:
                            # 测点没有点名画像里的原语字段：不能把缩放外推到
                            # 同类其他字段，跳过该身份。
                            continue
                        _plan_write(
                            planned, kind, identity, "cycle_costs", field,
                            float(cycle_costs[field]) * scale, evidence,
                            unit="cycle",
                            source_key=f"pim.pim_trace_cycle_costs.{field}")
                    elif kind == "communication":
                        link_name = str(target.get("operator") or "")
                        link = link_index.get(link_name)
                        if link is None:
                            # 链路身份不在本画像里：不把系数套到其他链路上。
                            continue
                        _plan_write(
                            planned, kind, identity, "links", link_name,
                            float(link["latency_base_ns"]) * scale, evidence,
                            unit="ns",
                            source_key=(
                                f"comm_config.links.{link_name}."
                                "latency_base_ns"))
                    else:
                        raise ValueError(
                            f"拟合类别 {kind} 的测点 {identity} 没有已知写回路径")
            else:
                raise ValueError(
                    f"拟合类别 {kind} 的测点 {identity} 拟合方法 "
                    f"{method!r} 无法写回")
    written = _apply_planned_writes(
        planned, cycle_costs, cpu_model, link_index, sources)
    if fit and written == 0:
        raise ValueError("拟合结果没有写回任何实际计时字段")
    data["cycle_costs"] = cycle_costs
    data["cpu_model"] = cpu_model
    data["links"] = links
    data["parameter_sources"] = sources
    # 写回后仍必须是一份合法画像。
    from contracts.hardware_profile import build_profile

    build_profile(data)
    return data


def _plan_write(planned, kind, identity, storage, key, value, evidence,
                *, unit, source_key) -> None:
    """登记一个画像字段的写回值；同一字段的冻结值必须一致。"""
    slot = planned.setdefault((storage, key), {
        "value": float(value), "evidence": [], "kinds": [],
        "unit": unit, "source_key": source_key})
    if not math.isclose(slot["value"], float(value), rel_tol=1e-12,
                        abs_tol=1e-15):
        raise ValueError(
            f"多个测点对共享计时字段 {key} 给出了不同系数；"
            "共享字段必须用联合样本拟合，不能连乘或覆盖")
    slot["evidence"].append(evidence)
    if kind not in slot["kinds"]:
        slot["kinds"].append(kind)


def _apply_planned_writes(planned, cycle_costs, cpu_model, link_index,
                          sources) -> int:
    """所有冲突核对通过后才按原画像基数写回一次，并合并逐字段证据。"""
    written = 0
    for (storage, key), slot in sorted(planned.items()):
        if storage == "cycle_costs":
            cycle_costs[key] = slot["value"]
        elif storage == "cpu_model":
            cpu_model[key] = slot["value"]
        elif storage == "links":
            link_index[key]["latency_base_ns"] = slot["value"]
        else:
            raise ValueError(f"未知的画像写回目标：{storage}")
        sources[slot["source_key"]] = {
            "origin": "fitted",
            "evidence": "；".join(dict.fromkeys(slot["evidence"])),
            "unit": slot["unit"],
            "scope": "、".join(sorted(slot["kinds"])),
        }
        written += 1
    return written


def _plan_dma_target(planned, kind, identity, target, cycle_costs,
                     phase_text) -> None:
    """DMA WLS 只写回训练集实际出现的搬运方向。"""
    dma_fit = target.get("dma_fit")
    if not isinstance(dma_fit, Mapping) or not dma_fit:
        raise ValueError(f"拟合类别 {kind} 的测点 {identity} 缺少 DMA 系数")
    field_of_operator = {
        "MRAM_READ": ("LOADN", "LOADN_PER_BYTE"),
        "MRAM_WRITE": ("STOREN", "STOREN_PER_BYTE"),
    }
    for operator, coefficients in sorted(dma_fit.items()):
        keys = field_of_operator.get(operator)
        if keys is None:
            raise ValueError(
                f"拟合类别 {kind} 的测点 {identity} 引用了未知 DMA 方向 "
                f"{operator!r}")
        evidence = (
            f"GeneSim 相对误差 WLS 拟合 {kind} 的 {operator} 参数"
            f"（{int(coefficients['sample_count'])} 个样本；"
            f"覆盖 {phase_text}）")
        values = (
            coefficients["base_cycles_per_transfer"],
            coefficients["cycles_per_byte"])
        for key, value in zip(keys, values):
            if key not in cycle_costs:
                raise ValueError(
                    f"拟合类别 {kind} 的测点 {identity} 缺少可写回的周期"
                    f"成本字段 {key}")
            _plan_write(
                planned, kind, identity, "cycle_costs", key, float(value),
                evidence, unit="cycle",
                source_key=f"pim.pim_trace_cycle_costs.{key}")


def _plan_host_target(planned, targets, cpu_model, kind, phase_text) -> None:
    """host 计时字段是全局的；不同主机类型的系数必须一致。"""
    scales = []
    identities = []
    total = 0
    for identity, target in sorted(targets.items()):
        if not isinstance(target, Mapping):
            raise ValueError(f"拟合类别 {kind} 的测点 {identity} 不是对象")
        fields = target.get("fields")
        if not isinstance(fields, list):
            raise ValueError(
                f"拟合类别 {kind} 的测点 {identity} 缺少可写字段声明")
        if not fields:
            continue
        method = target.get("method")
        if method != _genesim_validation().FIT_METHOD_SCALE:
            raise ValueError(
                f"拟合类别 {kind} 的测点 {identity} 拟合方法 {method!r} "
                "无法写回主机计时字段")
        scales.append(float(target.get("scale", math.nan)))
        identities.append(identity)
        total += int(target.get("count", 0))
    if not scales:
        return
    first = scales[0]
    if not math.isfinite(first) or first <= 0:
        raise ValueError(f"拟合类别 {kind} 的缩放系数不是正数：{first!r}")
    if any(not math.isclose(scale, first, rel_tol=1e-12, abs_tol=1e-15)
           for scale in scales[1:]):
        raise ValueError(
            f"拟合类别 {kind} 的不同主机类型给出了不同缩放系数；"
            "主机计时字段是全局的，不能分别写回")
    key = "base_latency_us"
    if key not in cpu_model:
        return
    _plan_write(
        planned, kind, "、".join(identities), "cpu_model", key,
        float(cpu_model[key]) * first,
        f"训练集测点 {'、'.join(identities)} 最小二乘缩放 "
        f"scale={first:.9g}（{total} 个样本；覆盖 {phase_text}）",
        unit="us", source_key=f"cpu_model.{key}")


def _shared_fit_groups(fit: Mapping[str, Mapping[str, Any]]):
    """把阶段拟合收敛成每类一份参数；冲突的阶段值直接拒绝。

    同一类别的计时字段是全局的，prefill 与 decode 必须给出同一组冻结
    参数；否则写回时会连乘，把 2× 与 3× 变成 6× 的伪校准。
    """
    by_kind: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for group, parameters in fit.items():
        match = re.fullmatch(r"(prefill|decode)/([a-z]+)", str(group))
        if match is None or match.group(2) not in _FIT_KIND_OF_GROUP:
            raise ValueError(f"拟合分组 {group!r} 无法映射到实际计时字段")
        by_kind.setdefault(_FIT_KIND_OF_GROUP[match.group(2)], []).append(
            (str(group), parameters))
    for kind in sorted(by_kind):
        entries = by_kind[kind]
        first_group, first = entries[0]
        for group, other in entries[1:]:
            if not _same_fit_parameters(first, other):
                raise ValueError(
                    f"计时类别 {kind} 在 {first_group} 与 {group} 给出了不同"
                    "的拟合参数；共享全局字段必须用 prefill/decode 联合样本"
                    "拟合，不能按阶段连乘或覆盖")
        yield kind, first, [group for group, _ in entries]


def _same_fit_parameters(left: Mapping[str, Any],
                         right: Mapping[str, Any]) -> bool:
    if left.get("method") != right.get("method"):
        return False
    if left.get("targets") != right.get("targets"):
        return False
    left_scale = left.get("scale")
    right_scale = right.get("scale")
    if (left_scale is None) != (right_scale is None):
        return False
    if left_scale is not None and not math.isclose(
            float(left_scale), float(right_scale),
            rel_tol=1e-12, abs_tol=1e-15):
        return False
    return left.get("dma_fit") == right.get("dma_fit")


def write_report(report: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def profile_applicability_error(report: Mapping[str, Any] | None,
                                profile) -> str | None:
    """报告适用范围必须与本次生效画像逐项一致；不一致不能写回或升级。"""
    if not isinstance(report, Mapping):
        return "报告不是对象"
    return _profile_scope_error(profile, report.get("scope"))


def result_maturity(
    report: Mapping[str, Any] | None,
    *,
    profile,
    profile_sources: Mapping[str, Mapping[str, str]],
    backend: str,
    out_of_scope: bool = False,
    used_predictor: bool = False,
) -> str:
    """只有独立报告通过、画像为 CEVA 且来源/范围全部匹配才算已验证。

    通用 PIM 画像没有对应实测数据，不能套用 CEVA 的已验证等级；空来源
    同样没有升级依据。范围必须与本画像逐项一致，版本或产物不符一律保持
    模型预测。
    """
    if used_predictor or backend != "pipeline" or out_of_scope:
        return "model_prediction"
    if not _report_evidence_valid(report):
        return "model_prediction"
    if profile is None or getattr(profile, "ceva", None) is None:
        return "model_prediction"
    if not isinstance(profile_sources, Mapping) or not profile_sources:
        return "model_prediction"
    for entry in profile_sources.values():
        if not isinstance(entry, Mapping):
            return "model_prediction"
        if entry.get("origin") not in CALIBRATED_ORIGINS:
            return "model_prediction"
    if profile_applicability_error(report, profile) is not None:
        return "model_prediction"
    return "validated_in_scope"


def _point_records(observations: Iterable[CalibrationSample],
                   frozen: Iterable[CalibrationSample]) -> list[dict]:
    records = []
    for observed, predicted in zip(observations, frozen):
        records.append({
            "sample_id": observed.sample_id,
            "dataset_id": observed.dataset_id,
            "source": observed.source,
            "phase": observed.phase,
            "kind": observed.kind,
            "shape": list(observed.shape),
            "dtype": observed.dtype,
            "unit": observed.unit,
            "target": json.loads(json.dumps(
                dict(observed.target), default=str)),
            "features": json.loads(json.dumps(
                dict(observed.features), default=str)),
            "operator": observed.operator,
            "artifact_phase": observed.artifact_phase,
            "predicted": observed.predicted,
            "frozen_predicted": predicted.predicted,
            "observed": observed.observed,
            "error": predicted.error,
            "relative_error": predicted.relative_error,
        })
    return records


@lru_cache(maxsize=1)
def _genesim_report_keys():
    """从 GeneSim 读取 schema 11 的产物/范围键清单，避免两仓各维护一份。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from config_loader import (
        CEVA_VALIDATION_REQUIRED_PACKAGE_KEYS,
        CEVA_VALIDATION_REQUIRED_SCOPE_KEYS,
    )

    return (CEVA_VALIDATION_REQUIRED_PACKAGE_KEYS,
            CEVA_VALIDATION_REQUIRED_SCOPE_KEYS)


def _scope_evidence_error(scope: Mapping[str, Any]) -> str | None:
    if not isinstance(scope, Mapping) or not scope:
        return "缺少适用范围，不能升级为独立验证"
    _, required = _genesim_report_keys()
    missing = [key for key in required if key not in scope]
    if missing:
        return f"适用范围缺少 {missing}"
    for key in ("cost_model", "host_orchestration", "timing_boundary",
                "request_boundary", "communication_model", "shape_range"):
        if not isinstance(scope.get(key), Mapping):
            return f"适用范围 {key} 必须是映射"
    if not isinstance(scope["communication_model"].get("links"), (list, tuple)):
        return "适用范围的通信链路必须是列表"
    shapes_error = _operator_shapes_error(scope["shape_range"])
    if shapes_error is not None:
        return shapes_error
    dtype = scope.get("dtype")
    if not isinstance(dtype, str) or not dtype.strip():
        return "适用范围缺少数据类型"
    layout = scope.get("dtype_layout")
    if not isinstance(layout, Mapping) or not layout:
        return "适用范围缺少逐算子数据类型布局"
    normalize = _genesim_validation().normalize_dtype
    for op_type, value in layout.items():
        if not isinstance(op_type, str) or not op_type.strip():
            return "适用范围的数据类型布局含空算子名"
        if normalize(value) is None:
            return f"适用范围的数据类型布局 {op_type} 无法识别"
    validation = scope.get("validation")
    samples = (
        validation.get("samples") if isinstance(validation, Mapping) else None)
    for index, sample in enumerate(samples or []):
        if not isinstance(sample, Mapping):
            continue
        operator = sample.get("operator")
        expected = layout.get(operator)
        if expected is None:
            return f"数据类型布局缺少验证样本算子 {operator!r}"
        if normalize(expected) != normalize(sample.get("dtype")):
            return (
                f"数据类型布局 {operator} 与验证样本精度不一致")
    return None


def _operator_shapes_error(shape_range: Mapping[str, Any]) -> str | None:
    """逐算子形状声明必须是“算子名 -> 非空正整数形状列表”的映射。"""
    for axis, bounds in shape_range.items():
        if str(axis) != "operator_shapes":
            continue
        if not isinstance(bounds, Mapping) or not bounds:
            return "报告形状范围的逐算子执行形状必须是非空映射"
        for operator, rows in bounds.items():
            if not isinstance(operator, str) or not operator.strip():
                return "报告形状范围的逐算子执行形状含空算子名"
            if not isinstance(rows, (list, tuple)) or not rows:
                return f"报告形状范围缺少 {operator} 的执行形状"
            for row in rows:
                if (
                    not isinstance(row, (list, tuple)) or not row
                    or any(
                        isinstance(dim, bool)
                        or not isinstance(dim, int)
                        or dim <= 0
                        for dim in row)
                ):
                    return f"报告形状范围 {operator} 的执行形状不合法"
    return None


def _profile_scope_error(profile, scope: Any) -> str | None:
    """报告适用范围必须与本次画像逐项一致，而不是只要求键存在。"""
    if not isinstance(scope, Mapping):
        return "报告缺少适用范围"
    ceva = getattr(profile, "ceva", None)
    if ceva is None:
        return "画像不是 CEVA，不能套用已验证等级"
    if scope.get("target_name") != ceva.target_name:
        return "报告目标型号与画像不一致"
    if scope.get("engines_per_core") != ceva.engines_per_core:
        return "报告 engine 数与画像不一致"
    if scope.get("core_count") != profile.clusters_per_device:
        return "报告配置核数与画像不一致"
    used = scope.get("used_core_count")
    if (isinstance(used, bool) or not isinstance(used, int)
            or used <= 0 or used > profile.clusters_per_device):
        return "报告已用核数不合法"
    if scope.get("execution_scope") != CEVA_EXECUTION_SCOPE:
        return "报告执行边界与画像不一致"
    clock = _finite(scope.get("clock_period_ns"))
    if clock is None or not math.isclose(
            clock, profile.clock_period_ns, rel_tol=1e-9, abs_tol=0.0):
        return "报告时钟与画像不一致"
    if scope.get("element_bytes") != profile.bytes_per_element:
        return "报告数据类型宽度与画像不一致"
    cost = scope.get("cost_model")
    if not isinstance(cost, Mapping):
        return "报告缺少成本模型"
    expected = {
        "mac_width": profile.mac_width,
        "tile_size": profile.tile_size,
        "chunk_size": profile.chunk_size,
        "max_bulk_transfer_bytes": profile.max_bulk_transfer_bytes,
        "multipliers": profile.multipliers,
    }
    for key, value in expected.items():
        if cost.get(key) != value:
            return f"报告成本模型 {key} 与画像不一致"
    reported_cpu = cost.get("cpu_model")
    if not isinstance(reported_cpu, Mapping):
        return "报告成本模型缺少 cpu_model"
    for key, value in profile.cpu_model.items():
        other = _finite(reported_cpu.get(key))
        if other is None or not math.isclose(
                other, float(value), rel_tol=1e-12, abs_tol=0.0):
            return f"报告成本模型 cpu_model.{key} 与画像不一致"
    loop = cost.get("loop_encoding")
    if not isinstance(loop, Mapping):
        return "报告成本模型缺少 loop_encoding"
    if loop.get("use_long_loop_counts") != profile.use_long_loop_counts:
        return "报告成本模型的循环编码开关与画像不一致"
    if "dynamic_loop_primitives" not in loop or "fixed_loop_primitives" not in loop:
        return "报告成本模型的循环原语清单不完整"
    reported_cycles = cost.get("cycle_costs")
    if not isinstance(reported_cycles, Mapping):
        return "报告成本模型缺少 cycle_costs"
    if set(reported_cycles) != set(profile.cycle_costs):
        return "报告成本模型的周期成本键集合与画像不一致"
    for key, value in profile.cycle_costs.items():
        other = _finite(reported_cycles.get(key))
        if other is None or not math.isclose(
                other, float(value), rel_tol=1e-12, abs_tol=1e-12):
            return f"报告成本模型的 cycle_costs.{key} 与画像不一致"
    reported_zones = cost.get("zone_cycle_factors")
    if not isinstance(reported_zones, Mapping):
        return "报告成本模型缺少 zone_cycle_factors"
    if set(reported_zones) != set(profile.zone_cycle_factors):
        return "报告成本模型的区间倍率键集合与画像不一致"
    for key, value in profile.zone_cycle_factors.items():
        other = _finite(reported_zones.get(key))
        if other is None or not math.isclose(
                other, float(value), rel_tol=1e-12, abs_tol=1e-12):
            return f"报告成本模型的 zone_cycle_factors.{key} 与画像不一致"
    communication = scope.get("communication_model")
    if not isinstance(communication, Mapping):
        return "报告缺少通信模型"
    if communication.get("inter_cluster_communication") != (
            profile.inter_cluster_communication):
        return "报告跨核路由模式与画像不一致"
    relay = _finite(communication.get("host_relay_overhead_ns"))
    if relay is None or not math.isclose(
            relay, profile.host_relay_overhead_ns, rel_tol=1e-9,
            abs_tol=1e-9):
        return "报告 HOST 中转开销与画像不一致"
    if not _links_equal(communication.get("links"), profile.links):
        return "报告通信链路与画像不一致"
    shape_range = scope.get("shape_range")
    if not isinstance(shape_range, Mapping):
        return "报告缺少形状范围"
    for axis, bounds in shape_range.items():
        if str(axis) == "operator_shapes":
            if _operator_shapes_error({"operator_shapes": bounds}) is not None:
                return "报告形状范围的逐算子执行形状不合法"
            continue
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(isinstance(value, bool)
                       or not isinstance(value, int) for value in bounds)
                or bounds[0] > bounds[1]):
            return f"报告形状范围 {axis} 不合法"
    return None


def _links_equal(reported: Any, expected) -> bool:
    if not isinstance(reported, (list, tuple)):
        return False
    if len(reported) != len(expected):
        return False

    def normalized(rows) -> list[dict] | None:
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                return None
            result.append({
                "name": row.get("name"),
                "src_domain": row.get("src_domain"),
                "dst_domain": row.get("dst_domain"),
                "bandwidth_gbps": _finite(row.get("bandwidth_gbps")),
                "latency_base_ns": _finite(row.get("latency_base_ns")),
                "is_full_duplex": row.get("is_full_duplex"),
                "intra_cluster_only": row.get("intra_cluster_only"),
                "inter_cluster_only": row.get("inter_cluster_only"),
            })
        return sorted(result, key=lambda entry: str(entry["name"]))

    left = normalized(reported)
    right = normalized([link.to_json() for link in expected])
    if left is None or right is None:
        return False
    for first, second in zip(left, right):
        for key in ("name", "src_domain", "dst_domain", "is_full_duplex",
                    "intra_cluster_only", "inter_cluster_only"):
            if first[key] != second[key]:
                return False
        for key in ("bandwidth_gbps", "latency_base_ns"):
            if first[key] is None or second[key] is None:
                return False
            if not math.isclose(first[key], second[key],
                                rel_tol=1e-9, abs_tol=1e-12):
                return False
    return True


def _report_evidence_valid(report: Mapping[str, Any] | None) -> bool:
    """逐点来源、拟合、统计、阈值、产物与后端都必须能从报告本身复核。

    报告内部证据、产物文件与真实 pipeline 后端全部由 GeneSim
    `ceva_validation.report_evidence_error` 唯一复核；调优侧只补充画像
    适用范围与生成证据的额外结构检查，不再另写一套外层证据判据。
    """
    if not isinstance(report, Mapping):
        return False
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        return False
    if _scope_evidence_error(report.get("scope") or {}) is not None:
        return False
    if _generated_evidence_error(report.get("scope") or {}) is not None:
        return False
    return _genesim_validation().report_evidence_error(
        report, require_host_and_communication=True) is None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _statistics(samples: list[CalibrationSample]) -> dict:
    errors = sorted(sample.error for sample in samples)
    relatives = sorted(sample.relative_error for sample in samples)
    count = len(samples)
    p90_index = math.ceil(0.9 * count) - 1
    return {
        "count": count,
        "median_error": _median(errors),
        "max_abs_error": max(abs(value) for value in errors),
        "median": _median(relatives),
        "p90": relatives[p90_index],
        "max": relatives[-1],
        "samples": [sample.sample_id for sample in samples],
    }


def _median(values: list[float]) -> float:
    count = len(values)
    middle = count // 2
    if count % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _sample_from_json(raw: Mapping, line: int) -> CalibrationSample:
    where = f"样本第 {line} 行"
    try:
        sample = CalibrationSample(
            sample_id=_text(raw, "sample_id", where),
            dataset_id=_text(raw, "dataset_id", where),
            source=_text(raw, "source", where),
            phase=_text(raw, "phase", where),
            kind=_text(raw, "kind", where),
            shape=tuple(int(value) for value in raw["shape"]),
            dtype=_text(raw, "dtype", where),
            predicted=_number(raw, "predicted", where),
            observed=_number(raw, "observed", where),
            target=dict(raw.get("target") or {}),
            unit=str(raw.get("unit", "ns")),
            features=dict(raw.get("features") or {}),
            operator=_optional_text(raw.get("operator")),
            artifact_phase=_optional_text(raw.get("artifact_phase")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where} 字段不合法：{exc}") from exc
    if sample.source not in SOURCES:
        raise ValueError(f"{where} 的来源 {sample.source!r} 不在 {sorted(SOURCES)}")
    if sample.phase not in PHASES:
        raise ValueError(f"{where} 的 phase {sample.phase!r} 不合法")
    if sample.kind not in KINDS:
        raise ValueError(f"{where} 的 kind {sample.kind!r} 不合法")
    if sample.observed <= 0:
        raise ValueError(f"{where} 的 observed 必须是正数")
    if not sample.target:
        raise ValueError(f"{where} 缺少目标配置身份 target")
    # 逐点误差必须能由预测值与观测值复算，与样本自带值不符即拒绝。
    for field in ("error", "relative_error"):
        if field in raw:
            expected = sample.error if field == "error" else sample.relative_error
            if not math.isclose(float(raw[field]), expected, rel_tol=1e-9):
                raise ValueError(f"{where} 的 {field} 无法由 predicted/observed 复算")
    return sample


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text(raw: Mapping, key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 缺少非空字符串 {key}")
    return value


def _number(raw: Mapping, key: str, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} 的 {key} 必须是数值")
    if not math.isfinite(value):
        raise ValueError(f"{where} 的 {key} 必须是有限数")
    return float(value)
