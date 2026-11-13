"""结果记录：轮次 JSONL、稳定排序、best 与人类可读摘要。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from contracts.hardware_profile import build_profile
from contracts.op_contract import PimOptimizations
from tune.candidate import OpKnobs, StrategyChoice, TuneState

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ArtifactEntry:
    role: str
    path: str
    sha256: str

    def to_json(self) -> dict:
        return {"role": self.role, "path": self.path, "sha256": self.sha256}

    @classmethod
    def from_json(cls, raw: Mapping) -> "ArtifactEntry":
        return cls(role=str(raw["role"]), path=str(raw["path"]),
                   sha256=str(raw["sha256"]))


@dataclass(frozen=True)
class ArtifactManifest:
    """一个候选的产物清单；完成后才创建，失败目录不会命中缓存。"""

    version: int
    compile_key: str
    inputs: tuple[ArtifactEntry, ...]
    files: tuple[ArtifactEntry, ...]

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "compile_key": self.compile_key,
            "inputs": [entry.to_json() for entry in self.inputs],
            "files": [entry.to_json() for entry in self.files],
        }

    @classmethod
    def from_json(cls, raw: Mapping) -> "ArtifactManifest":
        return cls(
            version=int(raw["version"]),
            compile_key=str(raw["compile_key"]),
            inputs=tuple(ArtifactEntry.from_json(e) for e in raw.get("inputs", ())),
            files=tuple(ArtifactEntry.from_json(e) for e in raw.get("files", ())),
        )


@dataclass(frozen=True)
class RoundRecord:
    """一轮候选的完整记录；失败候选的 score 为 None。"""

    sequence: int
    candidate_id: str
    compile_key: str
    sim_key: str
    cache: str
    status: str
    result_maturity: str
    state: TuneState | None
    score: float | None
    run_id: str = ""
    compile_cache_hit: bool = False
    sim_cache_hit: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    stop_reasons: Mapping[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    search_elapsed_s: float = 0.0
    inner_evaluations: int = 0
    validation_evaluations: int = 1
    artifacts: ArtifactManifest | None = None
    failure: str | None = None
    paths: Mapping[str, str] = field(default_factory=dict)
    # 与响应日志同源的调用序号；响应记录才是提交点，恢复时据此对账。
    call_index: int | None = None

    @property
    def successful(self) -> bool:
        return self.status == "success" and self.score is not None

    def to_json(self) -> dict:
        return {
            "sequence": self.sequence,
            "candidate_id": self.candidate_id,
            "compile_key": self.compile_key,
            "sim_key": self.sim_key,
            "cache": self.cache,
            "status": self.status,
            "result_maturity": self.result_maturity,
            "run_id": self.run_id,
            "compile_cache_hit": self.compile_cache_hit,
            "sim_cache_hit": self.sim_cache_hit,
            "state": None if self.state is None else state_to_json(self.state),
            "score": self.score,
            "metrics": dict(self.metrics),
            "stop_reasons": dict(self.stop_reasons),
            "elapsed_s": self.elapsed_s,
            "search_elapsed_s": self.search_elapsed_s,
            "inner_evaluations": self.inner_evaluations,
            "validation_evaluations": self.validation_evaluations,
            "artifacts": None if self.artifacts is None else self.artifacts.to_json(),
            "failure": self.failure,
            "paths": dict(self.paths),
            "call_index": self.call_index,
        }

    @classmethod
    def from_json(cls, raw: Mapping) -> "RoundRecord":
        score = raw.get("score")
        return cls(
            sequence=int(raw["sequence"]),
            candidate_id=str(raw["candidate_id"]),
            compile_key=str(raw.get("compile_key", "")),
            sim_key=str(raw.get("sim_key", "")),
            cache=str(raw.get("cache", "fast")),
            status=str(raw["status"]),
            result_maturity=str(raw.get("result_maturity", "model_prediction")),
            run_id=str(raw.get("run_id", "")),
            compile_cache_hit=bool(raw.get("compile_cache_hit", False)),
            sim_cache_hit=bool(raw.get("sim_cache_hit", False)),
            state=None if raw.get("state") is None
            else state_from_json(raw["state"]),
            score=None if score is None else float(score),
            metrics=dict(raw.get("metrics", {})),
            stop_reasons=dict(raw.get("stop_reasons", {})),
            elapsed_s=float(raw.get("elapsed_s", 0.0)),
            search_elapsed_s=float(raw.get("search_elapsed_s", 0.0)),
            inner_evaluations=int(raw.get("inner_evaluations", 0)),
            validation_evaluations=int(raw.get("validation_evaluations", 1)),
            artifacts=None if raw.get("artifacts") is None
            else ArtifactManifest.from_json(raw["artifacts"]),
            failure=raw.get("failure"),
            paths=dict(raw.get("paths", {})),
            call_index=(
                None if raw.get("call_index") is None
                else int(raw["call_index"])),
        )


def state_to_json(state: TuneState) -> dict:
    return {
        "strategy": {
            "num_dpus": state.strategy.num_dpus,
            "num_stages": state.strategy.num_stages,
        },
        "op": {"num_tasklets": state.op.num_tasklets},
        "optimization": (
            None if state.optimization is None
            else state.optimization.to_payload()),
        "profile": state.profile.to_json(),
    }


def state_from_json(raw: Mapping) -> TuneState:
    optimization = raw.get("optimization")
    return TuneState(
        strategy=StrategyChoice(
            int(raw["strategy"]["num_dpus"]), int(raw["strategy"]["num_stages"])),
        op=OpKnobs(int(raw["op"]["num_tasklets"])),
        profile=build_profile(raw["profile"]),
        optimization=(
            None if optimization is None
            else PimOptimizations.from_payload(optimization)),
    )


def candidate_id(state: TuneState) -> str:
    """候选编号由各层取值构成；硬件层用完整画像摘要，时钟不同也能区分。"""
    profile = state.profile
    optimization = ""
    if state.optimization is not None:
        flags = "".join(
            "1" if value else "0"
            for value in state.optimization.to_payload().values())
        optimization = f"_opt{flags}"
    return (
        f"tp{state.strategy.tp_width}_pp{state.strategy.num_stages}"
        f"_dpu{state.strategy.num_dpus}"
        f"_tl{state.op.num_tasklets}{optimization}"
        f"_{profile.name}_p{profile.profile_sha256()[:8]}"
    )


def append_round(path: Path, record: RoundRecord) -> None:
    """按行追加并落盘；完成记录能被后续进程读到。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_json(), ensure_ascii=False)
    with path.open("a") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_rounds(path: Path) -> list[RoundRecord]:
    """读轮次日志；末尾半行截断，完整坏行直接报错。"""
    if not Path(path).is_file():
        return []
    text = Path(path).read_text()
    if text and not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    records = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(RoundRecord.from_json(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"轮次日志第 {number} 行是坏行：{exc}") from exc
    return records


def rank_records(records: list[RoundRecord]) -> list[RoundRecord]:
    """成功候选按分数、候选编号稳定排序；失败候选排在最后。"""
    return sorted(
        records,
        key=lambda record: (
            0 if record.successful else 1,
            record.score if record.successful else math.inf,
            record.candidate_id,
            record.sequence,
        ),
    )


def write_best(output_dir: Path, records: list[RoundRecord],
               *, extra: Mapping[str, Any] | None = None) -> Path | None:
    """写出最优成功候选；全失败时不创建文件。"""
    ranked = [record for record in rank_records(records) if record.successful]
    if not ranked:
        return None
    best = ranked[0]
    payload = best.to_json()
    payload.update(dict(extra or {}))
    path = Path(output_dir) / "best.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def write_summary(output_dir: Path, run_name: str, records: list[RoundRecord],
                  *, extra_lines: list[str] | None = None) -> Path:
    """人类可读摘要，内容全部来自完整日志。"""
    path = Path(output_dir) / "summary.md"
    ranked = rank_records(records)
    lines = [f"# 调优摘要：{run_name}", ""]
    if not any(record.successful for record in records):
        lines.append("没有成功候选。失败原因见 rounds.jsonl。")
    else:
        lines.append("| 序号 | 候选 | 状态 | total_time_s | 缓存 | 产物 |")
        lines.append("|---:|---|---|---:|---|---|")
        for record in ranked:
            score = "" if record.score is None else f"{record.score:.9f}"
            lines.append(
                f"| {record.sequence} | {record.candidate_id} | {record.status} "
                f"| {score} | {record.cache} | {record.compile_key} |")
        best = next(record for record in ranked if record.successful)
        lines.append("")
        lines.append(f"最优候选：{best.candidate_id}（total_time_s={best.score:.9f}）")
    if extra_lines:
        lines.extend(["", *extra_lines])
    path.write_text("\n".join(lines) + "\n")
    return path


def compare_scores(expected: float, actual: float, *, rtol: float = 1e-9) -> None:
    """严格重放判据：模型输出只在确定性容差内比较。"""
    if not math.isclose(expected, actual, rel_tol=rtol, abs_tol=0.0):
        raise AssertionError(
            f"重放分数不一致：{actual!r} vs {expected!r}（相对容差 {rtol}）")


def manifest_from_dir(root: Path, compile_key: str,
                      inputs: tuple[ArtifactEntry, ...] | None = None
                      ) -> ArtifactManifest:
    """按目录内容生成产物清单；`artifacts.json` 自身不计入。

    `inputs` 是调用方按当前输入派生的身份条目，模型等多项输入不能用
    单文件哈希表达，必须由调用方给出。
    """
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifacts.json":
            continue
        files.append(ArtifactEntry(
            role=path.relative_to(root).as_posix(),
            path=path.relative_to(root).as_posix(),
            sha256=_sha256(path),
        ))
    return ArtifactManifest(version=MANIFEST_VERSION, compile_key=compile_key,
                            inputs=tuple(inputs or ()), files=tuple(files))


def verify_manifest(root: Path, manifest: ArtifactManifest) -> None:
    """逐文件核对内容哈希；任何缺失、改写或越界链接都视为缓存不完整。"""
    root_resolved = Path(root).resolve()
    for entry in manifest.files:
        relative = Path(entry.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"产物清单含越界路径：{entry.path}")
        path = Path(root) / relative
        if not path.is_file():
            raise ValueError(f"产物缺失：{path}")
        # 符号链接可以保持角色/路径文本一致却指向另一个文件，等同路径别名。
        if path.is_symlink():
            raise ValueError(f"产物不能是符号链接：{path}")
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"产物路径解析到编译目录之外：{path}") from exc
        if _sha256(path) != entry.sha256:
            raise ValueError(f"产物内容已变：{path}")


def _verify_manifest_roles(manifest: ArtifactManifest) -> None:
    """角色必须与规范相对路径一一对应；禁止重复角色与重复路径别名。"""
    roles = [entry.role for entry in manifest.files]
    paths = [Path(entry.path).as_posix() for entry in manifest.files]
    duplicates = sorted(
        role for role, count in Counter(roles).items() if count > 1)
    if duplicates:
        raise ValueError(f"编译产物清单含重复角色：{duplicates}")
    duplicate_paths = sorted(
        path for path, count in Counter(paths).items() if count > 1)
    if duplicate_paths:
        raise ValueError(f"编译产物清单含重复路径：{duplicate_paths}")
    for entry in manifest.files:
        if Path(entry.path).as_posix() != entry.role:
            raise ValueError(
                f"编译产物清单的角色 {entry.role!r} 与路径 "
                f"{entry.path!r} 不一致")


# 一次完整编译必须产出的设计角色；写入、快速读取与两种重放共用这一判据。
COMPILE_REQUIRED_ROLES = (
    "partition_plan.json", "placed.ir", "placement.json", "refined.ir",
    "cost.json")


def verify_compile_products(root: Path, manifest: ArtifactManifest) -> None:
    """编译完成判据：必需角色、PIM MLIR 覆盖与逐文件哈希。

    只检查清单里实际列出的文件是不够的：空清单、缺 role、角色别名到
    另一个文件、sidecar 指向清单外的 MLIR 都必须视为缓存不完整，
    不能命中也不能重放。
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"编译产物目录不存在：{root}")
    if not manifest.files:
        raise ValueError("编译产物清单为空，不能作为完整编译命中")
    _verify_manifest_roles(manifest)
    roles = {entry.role for entry in manifest.files}
    missing = [role for role in COMPILE_REQUIRED_ROLES if role not in roles]
    if missing:
        raise ValueError(f"编译产物清单缺少必需角色：{missing}")
    verify_manifest(root, manifest)
    sidecar_path = root / "placement.json"
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"放置 sidecar 无法读取：{exc}") from exc
    operators = sidecar.get("operators")
    if not isinstance(operators, Mapping) or not operators:
        raise ValueError("放置 sidecar 没有 PIM 算子产物，编译不完整")
    referenced: set[str] = set()
    root_resolved = root.resolve()
    for op_id, entry in operators.items():
        raw = entry.get("pimir_path") if isinstance(entry, Mapping) else None
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"算子 {op_id} 没有 PIM MLIR 产物")
        try:
            relative = Path(raw).resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(
                f"算子 {op_id} 的 PIM MLIR 不在编译目录内：{raw}") from exc
        role = relative.as_posix()
        if role not in roles:
            raise ValueError(
                f"算子 {op_id} 的 PIM MLIR 未登记在产物清单：{role}")
        digest = _sha256(Path(raw))
        if digest != entry.get("pimir_sha256"):
            raise ValueError(f"算子 {op_id} 的 PIM MLIR 内容已变")
        referenced.add(role)
    unlisted = sorted(
        role for role in roles
        if role.startswith("pimir/") and role not in referenced)
    if unlisted:
        raise ValueError(f"产物清单含 sidecar 未引用的 PIM MLIR：{unlisted[:8]}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
