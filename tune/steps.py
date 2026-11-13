"""候选执行：图/算子编译、成本精化、一致性校验与仿真。

每一步的产物都写进候选专属目录；编译缓存按内容键原子发布，仿真按
`sim_key` 与尝试目录隔离。只在这里调用生产编译入口，不另写调优专用路径。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from contracts.hardware_profile import HardwareProfile
from contracts.partition_plan import PartitionPlan
from genesim_bridge.paths import genesim_root
from tune._genesim import genesim_config_class
from tune.candidate import TuneState, build_strategy, model_dims
from tune.config import MOVES, TuneConfig, run_identity
from tune.inputs import (
    InputSnapshotView,
    TuneInputs,
    file_sha256,
    verify_input_snapshot,
)
from tune.report import (
    ArtifactEntry,
    ArtifactManifest,
    RoundRecord,
    candidate_id,
    manifest_from_dir,
    verify_compile_products,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# 按角色登记语义源码：编译键只收真正产出图/算子产物的代码，仿真键再收
# 配置生成、启动命令与产物校验，运行身份再收候选生成与续跑算法。注释和
# 排版变化不改变摘要，语义代码变化必须让对应键失效。
_COMPILE_SOURCE_DIRS = (
    "contracts", "graph", "comm", "memory", "runtime",
    "opcompiler_bridge", "genesim_bridge",
)
_COMPILE_SOURCE_FILES = (
    "tune/steps.py", "tune/candidate.py", "tune/_genesim.py", "paths.json",
)
_SIM_SOURCE_FILES = (
    "tune/steps.py", "tune/config.py", "tune/_genesim.py",
    "scripts/run_full_pipeline.py", "paths.json",
)
_RUN_SOURCE_DIRS = ("tune",)
_RUN_SOURCE_FILES = (
    "scripts/tune_pipeline.py", "scripts/tune_sweep.py",
    "scripts/calibrate_profile.py", "scripts/run_full_pipeline.py",
    "paths.json",
)


def _semantic_bytes(path: Path) -> bytes:
    """Python 源码按规范化语法树取摘要；注释与排版不参与身份。"""
    import ast

    if path.suffix != ".py":
        return path.read_bytes()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return path.read_bytes()
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode()


def _source_digest(dirs: tuple[str, ...], files: tuple[str, ...]) -> str:
    paths = set()
    for name in dirs:
        root = REPO_ROOT / name
        paths.update(
            path for path in root.rglob("*.py")
            if "__pycache__" not in path.parts)
    paths.update(REPO_ROOT / name for name in files)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(_semantic_bytes(path))
    return digest.hexdigest()


@dataclass(frozen=True)
class CandidateArtifacts:
    """一个候选的编译产物目录与缓存状态。"""

    compile_key: str
    compile_dir: Path
    placement_sidecar: Path
    partition_plan: Path
    refined_ir: Path
    cost_sidecar: Path
    manifest: ArtifactManifest
    cache_hit: bool


@dataclass(frozen=True)
class SimOutput:
    """一次仿真的结果目录与指标。"""

    sim_dir: Path
    config_path: Path
    results_dir: Path
    traces_dir: Path
    summary: Mapping[str, Any]
    autotune: Mapping[str, Any]
    effective_placement: Path | None


def source_fingerprint(role: str = "compile") -> str:
    """按角色取 G 仓语义源码摘要；同名常量保留编译角色的默认行为。"""
    roles = {
        "compile": (_COMPILE_SOURCE_DIRS, _COMPILE_SOURCE_FILES),
        "sim": ((), _SIM_SOURCE_FILES),
        "run": (_RUN_SOURCE_DIRS, _RUN_SOURCE_FILES),
    }
    try:
        dirs, files = roles[role]
    except KeyError as exc:
        raise ValueError(f"未知的源码角色：{role!r}") from exc
    return _source_digest(dirs, files)


def sim_source_fingerprint() -> str:
    """仿真配置生成、启动命令与产物校验生产者的内容摘要。"""
    return source_fingerprint("sim")


def run_source_fingerprint() -> str:
    """候选生成、外层搜索与续跑算法的内容摘要。"""
    return source_fingerprint("run")


def genesim_source_fingerprint() -> str:
    """S 仓源码内容摘要，进入仿真键。"""
    digest = hashlib.sha256()
    root = genesim_root() / "src"
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def refine_level() -> str:
    """有 GPU 走 pimir 成本；纯 CPU 机器退回 oplevel，档位进内容键。"""
    import torch

    return "pimir" if torch.cuda.is_available() else "oplevel"


def run_semantic_identity(cfg: TuneConfig) -> dict:
    """一次运行依赖的源码、工具、基配置与环境身份。

    这些输入不在配置文本里，但会改变编译或仿真结果；续跑必须逐项核对，
    否则换了环境或工具仍会复用旧响应。路径与时间戳不入身份。
    """
    from opcompiler_bridge.driver import _compiler_fingerprint

    base = resolve_base_genesim_config(cfg.profile, cfg.profile_path)
    return {
        "compiler_source": source_fingerprint("compile"),
        "compiler_tools": _compiler_fingerprint(),
        "genesim_source": genesim_source_fingerprint(),
        "search_source": source_fingerprint("run"),
        "base_config_sha256": file_sha256(base),
        "runtime": _sim_runtime_identity(cfg, base),
        # 画像与校准报告按内容入身份：同路径原地修改也必须让续跑失效。
        "profile_sha256": cfg.profile.profile_sha256(),
        "validation_report_sha256": (
            None if cfg.calibration.validation_report is None
            else file_sha256(cfg.calibration.validation_report)),
        "refine_level": refine_level(),
    }


def compile_key(cfg: TuneConfig, inputs: TuneInputs, state: TuneState) -> str:
    """编译内容键：模型、形状、策略、算子、编译画像与源码/工具链。

    图编译前的容量门禁消费逐请求 KV 峰值与有效并发上限；这两项必须
    进入编译身份，否则换了请求分布仍会命中旧请求的容量复核结果。
    """
    from opcompiler_bridge.driver import _compiler_fingerprint

    request_peaks = _request_kv_tokens(inputs.trace.requests)
    payload = {
        "version": 2,
        "model_sha256": inputs.model_sha256,
        "ir_sha256": inputs.ir_sha256,
        "dtype": "float16",
        "prefill_seq_len": cfg.model.prefill_seq_len,
        "max_seq": inputs.max_seq,
        # 容量门禁与并发准入的实际输入；仅 max_seq 不能区分请求分布。
        "request_kv_tokens": list(request_peaks),
        "max_inflight_requests": _max_inflight_requests(
            cfg, state, len(request_peaks)),
        "strategy": {
            "num_dpus": state.strategy.num_dpus,
            "num_stages": state.strategy.num_stages,
            "tp_width": state.strategy.tp_width,
        },
        "op": {"num_tasklets": state.op.num_tasklets},
        "optimization": state.resolved_optimization(cfg).to_payload(),
        "compile_profile_sha256": state.profile.compile_profile_sha256(),
        "refine_level": refine_level(),
        "source": source_fingerprint(),
        "flagtree": _compiler_fingerprint(),
    }
    return _sha256_json(payload)


def sim_key(cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
            compile_key_value: str, replay_placement: Path | None = None) -> str:
    """仿真键：编译键 + 完整画像、S 源码、基配置与轨迹窗口。"""
    base = resolve_base_genesim_config(state.profile, cfg.profile_path)
    payload = {
        "version": 1,
        "compile_key": compile_key_value,
        "replay_placement_sha256": (
            None if replay_placement is None
            else file_sha256(Path(replay_placement))),
        "profile_sha256": state.profile.profile_sha256(),
        "genesim_source": genesim_source_fingerprint(),
        "sim_source": sim_source_fingerprint(),
        "base_config_sha256": file_sha256(base),
        "genesim_overrides": _canonical(_genesim_profile_overrides(state.profile)),
        "runtime_identity": _sim_runtime_identity(cfg, base, state.profile),
        "trace_sha256": inputs.trace_sha256,
        "placement": _canonical(_placement_payload(cfg)),
        "cost_sidecar": cfg.simulation.use_cost_sidecar,
        "validation_report_sha256": (
            None if cfg.calibration.validation_report is None
            else file_sha256(cfg.calibration.validation_report)),
        "num_requests": len(inputs.trace.requests),
    }
    return _sha256_json(payload)


# 一次编译必须绑定的输入角色：完整模型指纹、原始 IR 与请求轨迹内容。
_MANIFEST_INPUT_LABELS = {
    "model": "模型",
    "ir": "原始 IR",
    "trace": "请求轨迹",
}
_ARTIFACT_ROLE_OF_FIELD = (
    ("placement_sidecar", "placement.json"),
    ("partition_plan", "partition_plan.json"),
    ("refined_ir", "refined.ir"),
    ("cost_sidecar", "cost.json"),
)


def manifest_inputs(inputs: TuneInputs) -> tuple[ArtifactEntry, ...]:
    """编译产物的输入身份：模型完整指纹、原始 IR 与轨迹窗口内容。"""
    return (
        ArtifactEntry(role="model", path=str(inputs.model_dir),
                      sha256=inputs.model_sha256),
        ArtifactEntry(role="ir", path=str(inputs.ir_path),
                      sha256=inputs.ir_sha256),
        ArtifactEntry(role="trace", path=str(inputs.trace_path),
                      sha256=inputs.trace_sha256),
    )


def _manifest_input_errors(manifest: ArtifactManifest,
                           inputs: TuneInputs) -> list[str]:
    """产物清单的输入角色必须与当前模型、原始 IR 和轨迹逐项一致。"""
    recorded: dict[str, ArtifactEntry] = {}
    errors: list[str] = []
    for entry in manifest.inputs:
        if entry.role in recorded:
            errors.append(f"产物清单含重复输入角色：{entry.role}")
        recorded[entry.role] = entry
    for expected in manifest_inputs(inputs):
        entry = recorded.get(expected.role)
        label = _MANIFEST_INPUT_LABELS[expected.role]
        if entry is None:
            errors.append(f"产物清单缺少{label}输入身份")
        elif entry.sha256 != expected.sha256:
            errors.append(
                f"产物清单的{label}身份 {entry.sha256!r} 与当前输入 "
                f"{expected.sha256!r} 不一致")
    return errors


def verify_manifest_inputs(manifest: ArtifactManifest,
                           inputs: TuneInputs) -> None:
    """严格重放与快速缓存共用的清单输入身份判据。"""
    errors = _manifest_input_errors(manifest, inputs)
    if errors:
        raise ValueError("候选输入身份校验失败：" + "；".join(errors))


def verify_candidate_identity(
    cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
    artifacts: CandidateArtifacts, *, expected_key: str | None = None,
) -> str:
    """由当前配置、输入、候选状态与产物目录共同核对候选身份。

    编译键必须在调用时重算，产物对象里保存的旧键不能自证；清单输入角色、
    五个产物路径与内容、精化 IR 的模型/原始 IR 派生身份逐项对账。
    """
    key = (compile_key(cfg, inputs, state)
           if expected_key is None else expected_key)
    errors: list[str] = []
    # 内存身份不能自证：先按磁盘当前字节复核模型、IR 与轨迹仍未变化。
    try:
        verify_input_snapshot(cfg, inputs)
    except ValueError as exc:
        errors.append(str(exc))
    if artifacts.compile_key != key:
        errors.append(
            f"候选对象记录的编译键 {artifacts.compile_key!r} 与当前输入/候选"
            f"重算结果 {key!r} 不一致")
    if artifacts.manifest.compile_key != key:
        errors.append(
            f"产物清单的编译键 {artifacts.manifest.compile_key!r} 与当前输入/"
            f"候选重算结果 {key!r} 不一致")
    errors.extend(_manifest_input_errors(artifacts.manifest, inputs))
    try:
        verify_compile_products(artifacts.compile_dir, artifacts.manifest)
    except (OSError, ValueError) as exc:
        errors.append(f"编译产物不完整：{exc}")
    root = Path(artifacts.compile_dir).resolve()
    declared = {entry.role: entry for entry in artifacts.manifest.files}
    for field, role in _ARTIFACT_ROLE_OF_FIELD:
        path = Path(getattr(artifacts, field))
        entry = declared.get(role)
        if entry is None:
            errors.append(f"产物清单缺少角色 {role}")
            continue
        if not path.is_file():
            errors.append(f"编译产物 {role} 不存在：{path}")
            continue
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            errors.append(f"编译产物 {role} 不在编译目录内：{path}")
            continue
        if relative != role:
            errors.append(f"编译产物 {role} 的路径 {relative!r} 与清单角色不一致")
        elif file_sha256(path) != entry.sha256:
            errors.append(f"编译产物 {role} 的内容与清单不一致")
    try:
        refined = json.loads(Path(artifacts.refined_ir).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"精化 IR 无法读取：{exc}")
    else:
        if refined.get("source_model_sha256") != inputs.model_sha256:
            errors.append("精化 IR 记录的模型身份与当前输入不一致")
        if refined.get("source_ir_sha256") != inputs.ir_sha256:
            errors.append("精化 IR 记录的原始 IR 身份与当前输入不一致")
    if errors:
        raise ValueError("候选身份校验失败：" + "；".join(errors))
    return key


def _expected_native_identity(cfg: TuneConfig,
                              profile: HardwareProfile) -> dict:
    """本次候选键声明的实际计时后端；发布与缓存命中都用它复核。"""
    base = resolve_base_genesim_config(profile, cfg.profile_path)
    effective = genesim_config_class()(str(base)).override(
        **_genesim_profile_overrides(profile))
    return _native_backend_identity(effective)


def _sim_runtime_identity(cfg: TuneConfig, base_path: Path,
                          profile: HardwareProfile | None = None) -> dict:
    """仿真环境、预测器检查点与原生后端的实际内容身份。

    后端类型来自画像覆盖后的最终生效配置；原生库按运行期同一套解析与
    加载规则探测内容哈希，加载失败就声明轨迹回退，不能只猜文件是否存在。
    这些输入不由画像描述，但会改变 GeneSim 的计时结果；漏进 sim_key 会让
    合法缓存命中返回另一套环境的结果。
    """
    GeneSimConfig = genesim_config_class()
    base = GeneSimConfig(str(base_path))
    effective = base.override(
        **_genesim_profile_overrides(cfg.profile if profile is None else profile))
    backends = {cfg.placement.evaluation_backend,
                cfg.placement.validation_backend}
    identity: dict[str, Any] = {
        "genesim_env": {
            key: value for key, value in sorted(os.environ.items())
            if key.startswith("GENESIM_")},
        "predictor_checkpoint": None,
        "native_backend": _native_backend_identity(effective),
        "dramsim3_config": None,
    }
    if "predictor" in backends:
        raw = effective.get("predictor.checkpoint_path", "")
        identity["predictor_checkpoint"] = _content_or_path(raw)
    dramsim = effective.get("system.dramsim3_config", "")
    if isinstance(dramsim, str) and dramsim.strip():
        identity["dramsim3_config"] = _content_or_path(dramsim)
    return identity


def _native_backend_identity(config) -> dict:
    """按最终生效配置探测实际可用的原生后端与库内容。

    加载能力与角色化内容身份都复用 GeneSim 运行期实现；路径本身不入
    身份，只保留内容摘要与是否回退。
    """
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from pim.pim_core import probe_native_backend

    return probe_native_backend(config)


# 决定计时口径的身份字段；执行计数等审计字段不参与跨进程比较。
_NATIVE_IDENTITY_FIELDS = (
    "pim_type", "core", "native", "fallback",
    "library_sha256", "attacc_sha256")


def _native_identity_error(expected: Mapping, actual: Any) -> str | None:
    """键声明的后端与运行台账不一致时返回原因；一致返回 None。"""
    if not isinstance(actual, Mapping):
        return f"仿真摘要缺少原生后端身份：{actual!r}"
    for key in _NATIVE_IDENTITY_FIELDS:
        if expected.get(key) != actual.get(key):
            return (
                f"{key} 在仿真键里为 {expected.get(key)!r}，"
                f"实际运行台账为 {actual.get(key)!r}")
    return None


def _content_or_path(raw: Any) -> dict:
    """按解析到的文件内容建立身份；找不到时保留声明路径供审计。"""
    if not isinstance(raw, str) or not raw.strip():
        return {"path": None, "sha256": None}
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path, genesim_root() / path]
    for candidate in candidates:
        if candidate.is_file():
            return {"path": raw, "sha256": file_sha256(candidate)}
    return {"path": raw, "sha256": None}


def build_candidate(
    cfg: TuneConfig,
    inputs: TuneInputs,
    state: TuneState,
    *,
    force_compile: bool = False,
    compile_fn: Callable | None = None,
    refine_fn: Callable | None = None,
) -> CandidateArtifacts:
    """构建或复用编译缓存；每次编译写入独立 attempt，失败不留完成标记。

    快速档命中任一已完整发布的 attempt；严格档总是新建 attempt，旧目录、
    旧清单与 PIM MLIR 一律保留，不再先删后发。
    """
    # 编译前按磁盘字节复核输入；发布前再复核一次，编译窗口内的改写不得入库。
    verify_input_snapshot(cfg, inputs)
    key = compile_key(cfg, inputs, state)
    cache_root = cfg.output.dir / "cache" / "compile" / key
    if not force_compile:
        cached = _read_compiled(cache_root, key, inputs)
        if cached is not None:
            return _replace_cache_hit(cached, True)
    cache_root.mkdir(parents=True, exist_ok=True)
    attempt_dir = cache_root / f"attempt-{uuid.uuid4().hex[:8]}"
    tmp = cache_root / f".tmp-{key}-{uuid.uuid4().hex[:8]}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        placement = (compile_fn or _real_compile)(cfg, inputs, state, tmp,
                                                  force_compile)
        refined_ir = tmp / "refined.ir"
        cost_sidecar = tmp / "cost.json"
        (refine_fn or _real_refine)(cfg, inputs, state, placement, refined_ir,
                                    cost_sidecar, force_compile)
        _archive_pimir(tmp, placement.sidecar, attempt_dir)
        # 侧车把 PIM 路径改写到发布位置后，再把最终内容身份写进精化 IR；
        # 两者必须成对入库，单独交换任一份都不能通过一致性校验。
        _bind_refined_sidecar(refined_ir, placement.sidecar, inputs)
        manifest = manifest_from_dir(tmp, key, manifest_inputs(inputs))
        (tmp / "artifacts.json").write_text(
            json.dumps(manifest.to_json(), indent=2, ensure_ascii=False))
        # sidecar 已把 PIM MLIR 路径改写到发布后的 attempt 目录，完整性
        # 判据必须在换名之后按最终位置核对。
        verify_input_snapshot(cfg, inputs)
        os.replace(tmp, attempt_dir)
        verify_compile_products(attempt_dir, manifest)
    except Exception:
        _remove_dir(tmp)
        _remove_dir(attempt_dir)
        _remove_empty_dir(cache_root)
        raise
    result = CandidateArtifacts(
        compile_key=key,
        compile_dir=attempt_dir,
        placement_sidecar=attempt_dir / "placement.json",
        partition_plan=attempt_dir / "partition_plan.json",
        refined_ir=attempt_dir / "refined.ir",
        cost_sidecar=attempt_dir / "cost.json",
        manifest=manifest,
        cache_hit=False,
    )
    return result


def _valid_sha256(value) -> bool:
    """内容摘要必须是 64 位小写十六进制且不是全零占位。"""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return value != "0" * 64 and all(c in "0123456789abcdef" for c in value)


def _toolchain_errors(compiler: dict) -> list[str]:
    """sidecar 声明的工具链身份必须与当前编译器及实际工具内容一致。"""
    from opcompiler_bridge.driver import compiler_identity

    toolchain = compiler.get("toolchain")
    if not isinstance(toolchain, dict):
        return ["sidecar 缺少工具链身份，无法证明仿真工具就是本轮编译工具"]
    errors: list[str] = []
    if toolchain.get("version") != 1:
        errors.append(
            f"sidecar 的工具链身份版本 {toolchain.get('version')!r} 不受支持")
    expected = compiler_identity()
    graph = toolchain.get("graph_compiler")
    source = graph.get("source_sha256") if isinstance(graph, dict) else None
    if source != expected["graph_compiler"]["source_sha256"]:
        errors.append("sidecar 的工具链图编译源码摘要与当前编译器不一致")
    fingerprint = toolchain.get("compiler_fingerprint")
    if (not isinstance(fingerprint, str) or not fingerprint
            or set(fingerprint) == {"0"}):
        errors.append("sidecar 的工具链身份缺少合法编译指纹")
    elif fingerprint != expected["compiler_fingerprint"]:
        errors.append("sidecar 的工具链编译指纹与当前编译器不一致")
    tools = toolchain.get("tools")
    if not isinstance(tools, dict) or not tools:
        errors.append("sidecar 的工具链身份缺少工具内容摘要")
        return errors
    if "triton_opt" not in tools:
        errors.append("sidecar 的工具链身份缺少 tools.triton_opt")
    for name, entry in tools.items():
        if not isinstance(entry, dict):
            errors.append(f"sidecar 的工具链工具 {name} 必须是对象")
            continue
        digest = entry.get("sha256")
        # 有实际路径的工具必须声明内容摘要；未解析到的工具允许为空。
        if entry.get("path") is not None or name == "triton_opt":
            if not _valid_sha256(digest):
                errors.append(
                    f"sidecar 的工具链工具 {name} 缺少合法的内容摘要")
    for name, actual in expected["tools"].items():
        digest = actual.get("sha256")
        if digest is None:
            continue
        entry = tools.get(name)
        declared = entry.get("sha256") if isinstance(entry, dict) else None
        if declared != digest:
            errors.append(f"sidecar 的工具链工具 {name} 内容摘要与当前工具不一致")
    return errors


def verify_compiled(cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
                    artifacts: CandidateArtifacts) -> None:
    """仿真前核对画像、容量、映射、PIM 哈希、工具链与算子编号。"""
    from scripts.run_full_pipeline import stale_pimir_entries

    errors: list[str] = []
    # 产物对象里的键不能自证：按当前 cfg/inputs/state 重算并核对完整身份。
    try:
        verify_candidate_identity(cfg, inputs, state, artifacts)
    except ValueError as exc:
        errors.append(str(exc))
    sidecar = json.loads(artifacts.placement_sidecar.read_text())
    compiler = sidecar.get("compiler") or {}
    if not sidecar.get("requires_pimir"):
        errors.append("sidecar 没有声明 requires_pimir，仿真可能静默退回手写模板")
    if compiler.get("compile_profile_sha256") != state.profile.compile_profile_sha256():
        errors.append("sidecar 的编译画像摘要与候选不一致")
    hardware = state.profile.to_pim_hardware_config(
        state.strategy.num_dpus, state.op.num_tasklets,
        optimizations=state.resolved_optimization(cfg))
    reported_hw = compiler.get("hardware")
    if not isinstance(reported_hw, dict):
        errors.append("sidecar 缺少编译硬件契约")
        reported_hw = {}
    # 目标与五元组同属请求级契约：仿真不能消费另一套编译目标的产物。
    if reported_hw.get("compiler_target") != hardware.compiler_target:
        errors.append("sidecar 的编译目标与候选契约不一致")
    expected_hw = {
        "num_dpus": hardware.num_dpus,
        "num_tasklets": hardware.num_tasklets,
        "mram_bytes_per_dpu": hardware.mram_bytes_per_dpu,
        "wram_bytes_per_dpu": hardware.wram_bytes_per_dpu,
        "dma_align": hardware.dma_align,
    }
    if {key: reported_hw.get(key) for key in expected_hw} != expected_hw:
        errors.append(f"sidecar 的硬件五元组与候选不一致：{reported_hw}")
    expected_optimizations = hardware.optimizations.to_payload()
    if reported_hw.get("optimizations") != expected_optimizations:
        errors.append(
            "sidecar 的优化开关与候选不一致："
            f"{reported_hw.get('optimizations')!r} vs {expected_optimizations!r}")
    # 同硬件不同 TP/PP 也必须区分：图策略、DPU 集合与计划映射同属候选身份。
    expected_strategy = {
        "name": f"tp{state.strategy.tp_width}_pp{state.strategy.num_stages}",
        "num_dpus": state.strategy.num_dpus,
        "num_stages": state.strategy.num_stages,
        "tp_width": state.strategy.tp_width,
        "dpu_ids": list(range(state.strategy.num_dpus)),
    }
    reported_strategy = compiler.get("strategy")
    if not isinstance(reported_strategy, dict) or {
            key: reported_strategy.get(key)
            for key in expected_strategy} != expected_strategy:
        errors.append(
            f"sidecar 的编译策略与候选不一致：{reported_strategy}")
    # 独立精化与正式调优共用同一编译契约：策略、PIM MLIR 属性与物理映射
    # 必须属于候选请求的同一套预算。
    from genesim_bridge.refine_profile import (
        SIDECAR_SCHEMA_VERSION, verify_placement_contract)

    if sidecar.get("version") != SIDECAR_SCHEMA_VERSION:
        errors.append(
            f"sidecar 的 schema 版本 {sidecar.get('version')!r} 不是当前导出器"
            f"的 {SIDECAR_SCHEMA_VERSION}；旧产物需重新生成")
    try:
        request_dpus, request_tasklets = verify_placement_contract(
            state.profile, sidecar,
            placement_path=artifacts.placement_sidecar,
            search_dirs=(artifacts.compile_dir / "pimir",))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if (request_dpus, request_tasklets) != (
                state.strategy.num_dpus, state.op.num_tasklets):
            errors.append(
                "放置 sidecar 的编译请求与候选的 DPU/tasklet 不一致")
    physical = compiler.get("physical_mram_bytes")
    if physical != state.profile.cluster_memory_bytes:
        errors.append("sidecar 的物理容量与画像不一致")
    if compiler.get("reserved_mram_bytes") != state.profile.sys_reserve_bytes:
        errors.append("sidecar 的预留容量与画像不一致")
    usable = state.profile.cluster_memory_bytes - state.profile.sys_reserve_bytes
    if compiler.get("usable_mram_bytes") != usable:
        errors.append("sidecar 的可用容量与画像不一致")
    # 顶层 hardware 是编译预算的可回读副本，必须与画像逐项对账。
    budget = sidecar.get("hardware")
    expected_budget = {
        "mram_bytes_per_dpu": state.profile.cluster_memory_bytes,
        "sys_reserve_bytes": state.profile.sys_reserve_bytes,
        "usable_mram_bytes": usable,
    }
    if not isinstance(budget, dict) or {
            key: budget.get(key) for key in expected_budget} != expected_budget:
        errors.append(f"sidecar 的顶层硬件预算与画像不一致：{budget}")
    plan = json.loads(artifacts.partition_plan.read_text())
    if plan.get("num_dpus") != state.strategy.num_dpus or \
            plan.get("num_stages") != state.strategy.num_stages:
        errors.append("partition_plan 与实际策略不一致")
    if plan.get("dpu_ids") != list(range(state.strategy.num_dpus)):
        errors.append("partition_plan 的 DPU 集合与实际策略不一致")
    if sidecar.get("dpu_to_cluster") != plan.get("dpu_to_cluster"):
        errors.append("sidecar 的 DPU 映射与方案不一致")
    stale = stale_pimir_entries(sidecar["operators"])
    if stale:
        errors.append(f"{len(stale)} 个算子的 pimir 哈希与磁盘不一致")
    refined = json.loads(artifacts.refined_ir.read_text())
    # 精化 IR 与放置 sidecar 必须由同一次编译产出，单独交换任一份都不能入库。
    if refined.get("placement_sidecar_sha256") != _semantic_sidecar_sha256(
            artifacts.placement_sidecar):
        errors.append("精化 IR 记录的放置 sidecar 身份与当前产物不一致")
    missing = []
    wrong_type = []
    for op in refined["operators"]:
        if op["op_type"] in ("MODEL_INPUT", "MODEL_OUTPUT"):
            continue
        entry = sidecar["operators"].get(str(op["op_id"]))
        if not entry or not entry.get("pimir_path"):
            missing.append(int(op["op_id"]))
        elif entry.get("op_type") != op["op_type"]:
            wrong_type.append(int(op["op_id"]))
    if missing:
        errors.append(
            f"{len(missing)} 个算子在 sidecar 里没有 pimir_path（如 op{missing[:5]}）")
    if wrong_type:
        errors.append(
            f"{len(wrong_type)} 个算子的类型与 sidecar 不一致（如 op{wrong_type[:5]}）")
    if len(inputs.trace.requests) != cfg.model.trace.num_requests:
        errors.append("轨迹窗口的请求数与配置不一致")
    errors.extend(_toolchain_errors(compiler))
    if errors:
        raise ValueError("候选一致性校验失败：" + "；".join(errors))


def build_sim(cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
              artifacts: CandidateArtifacts, sim_dir: Path,
              *, replay_placement: Path | None = None) -> tuple[Path, Path]:
    """生成候选专属 GeneSim 配置与画像契约。

    `replay_placement` 给了就关闭内层搜索，按保存的最终映射严格重放。
    """
    # 配置生成不能直接信任产物对象里的旧键：先按当前输入重算完整身份。
    compile_key_value = verify_candidate_identity(cfg, inputs, state, artifacts)
    GeneSimConfig = genesim_config_class()
    base_path = resolve_base_genesim_config(state.profile, cfg.profile_path)
    base = GeneSimConfig(str(base_path))
    results_dir = sim_dir / "results"
    traces_dir = sim_dir / "pim_traces"
    overrides = _genesim_profile_overrides(state.profile)
    overrides["model"] = {"ir_path": str(artifacts.refined_ir)}
    overrides["trace"] = {
        "path": str(inputs.trace_path),
        "num_requests": len(inputs.trace.requests),
    }
    overrides["output"] = {
        "results_dir": str(results_dir),
        "pim_traces_dir": str(traces_dir),
    }
    scheduler = {
        "compiler_placement_file": str(artifacts.placement_sidecar),
        "cost_sidecar_file": (
            str(artifacts.cost_sidecar) if cfg.simulation.use_cost_sidecar else ""),
        "require_compiler_pimir": True,
        "use_compiler_local_shapes": True,
        # 输出位置固定为本次 attempt；严格重放另用输入字段，两者不再混用。
        "effective_placement_file": str(sim_dir / "effective_placement.json"),
    }
    overrides["scheduler"] = scheduler
    if replay_placement is not None:
        overrides["scheduler"]["replay_placement_file"] = str(replay_placement)
        overrides["autotune"] = {
            "enabled": False, "save_effective_placement": True}
    elif cfg.placement.enabled:
        overrides["autotune"] = _autotune_overrides(cfg, sim_dir)
    else:
        # 内层搜索关闭时也要留最终映射：CEVA 由编译器一对一钉死。
        overrides["autotune"] = {
            "enabled": False, "save_effective_placement": True}
    if cfg.calibration.validation_report is not None:
        # 复用 GeneSim 的成熟度机制：报告路径进生效配置，结果由仿真侧给出。
        overrides.setdefault("ceva", {})["validation_report"] = str(
            cfg.calibration.validation_report)
    sim_dir.mkdir(parents=True, exist_ok=True)
    # 先算出生效值写契约，再把契约文件摘要写回配置；两者互不引用自身。
    # schema 3 起契约绑定完整候选编译身份：编译键、图策略、计划映射与
    # 实际产物内容，GeneSim 启动、清单与重放据此拒绝同硬件异策略产物。
    binding_hardware = state.profile.to_pim_hardware_config(
        state.strategy.num_dpus, state.op.num_tasklets,
        optimizations=state.resolved_optimization(cfg))
    plan = json.loads(artifacts.partition_plan.read_text())
    binding_artifacts = {
        "refined_ir_sha256": file_sha256(artifacts.refined_ir),
        "placement_sidecar_sha256": file_sha256(artifacts.placement_sidecar),
        # 仿真实际消费的请求轨迹与其窗口也是候选身份；这里只消费刚复核
        # 过的同一份快照，不能各自重读磁盘把新内容绑到旧键上。
        "trace_sha256": inputs.trace_sha256,
        "trace_num_requests": len(inputs.trace.requests),
    }
    if cfg.simulation.use_cost_sidecar:
        binding_artifacts["cost_sidecar_sha256"] = file_sha256(
            artifacts.cost_sidecar)
    contract = {
        "schema_version": 3,
        "profile_sha256": state.profile.profile_sha256(),
        "compile": {
            "compile_profile_sha256": state.profile.compile_profile_sha256(),
            "hardware": {
                "num_dpus": binding_hardware.num_dpus,
                "num_tasklets": binding_hardware.num_tasklets,
                "mram_bytes_per_dpu": binding_hardware.mram_bytes_per_dpu,
                "wram_bytes_per_dpu": binding_hardware.wram_bytes_per_dpu,
                "dma_align": binding_hardware.dma_align,
                "compiler_target": binding_hardware.compiler_target,
                "optimizations": binding_hardware.optimizations.to_payload(),
            },
            "physical_mram_bytes": state.profile.cluster_memory_bytes,
            "reserved_mram_bytes": state.profile.sys_reserve_bytes,
            "usable_mram_bytes": (
                state.profile.cluster_memory_bytes
                - state.profile.sys_reserve_bytes),
            "compile_key": compile_key_value,
            "strategy": {
                "name": (f"tp{state.strategy.tp_width}"
                         f"_pp{state.strategy.num_stages}"),
                "num_dpus": state.strategy.num_dpus,
                "num_stages": state.strategy.num_stages,
                "tp_width": state.strategy.tp_width,
                "dpu_ids": list(range(state.strategy.num_dpus)),
            },
            "dpu_to_cluster": plan.get("dpu_to_cluster"),
            "artifacts": binding_artifacts,
        },
        "expected": _flat_contract(state.profile),
    }
    contract_path = sim_dir / "hardware_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False))
    overrides["scheduler"]["hardware_profile_file"] = str(contract_path)
    overrides["scheduler"]["hardware_profile_sha256"] = file_sha256(contract_path)
    resolved = base.override(**overrides).resolved_config()
    config_path = sim_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    manifest = {
        "compile_key": compile_key_value,
        "sim_key": sim_key(cfg, inputs, state, compile_key_value,
                           replay_placement),
        "input_sha256": {
            "model": inputs.model_sha256,
            "ir": inputs.ir_sha256,
            "trace": inputs.trace_sha256,
        },
        "artifact_manifest": artifacts.manifest.to_json(),
        "config_sha256": file_sha256(config_path),
        "hardware_contract_sha256": file_sha256(contract_path),
    }
    (sim_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    # 配置与契约写出后再复核一次：仿真启动前输入已变时不能继续消费。
    verify_input_snapshot(cfg, inputs)
    return config_path, contract_path


def run_simulation(cfg: TuneConfig, sim_dir: Path, config_path: Path) -> SimOutput:
    """跑一次 GeneSim，核对结果来源与轨迹来源。"""
    from scripts.run_full_pipeline import _genesim_sim_command, _trace_metadata

    root = genesim_root()
    results_dir = sim_dir / "results"
    traces_dir = sim_dir / "pim_traces"
    log_path = sim_dir / "sim.log"
    cmd = _genesim_sim_command(root, config_path)
    with log_path.open("w") as handle:
        proc = subprocess.run(
            cmd, cwd=str(root), stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"GeneSim 仿真失败（exit {proc.returncode}）：\n{tail}")
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"仿真没有产出 {summary_path}")
    summary = json.loads(summary_path.read_text())
    autotune_path = results_dir / "autotune_metrics.json"
    if not autotune_path.is_file():
        raise RuntimeError(
            "仿真没有产出 results/autotune_metrics.json；"
            "内层搜索次数与最终重放无法审计")
    autotune = json.loads(autotune_path.read_text())
    effective = sim_dir / "effective_placement.json"
    if not effective.is_file():
        raise RuntimeError(
            "仿真没有产出最终放置 effective_placement.json；"
            "严格复跑与审计都依赖这份映射")
    output = SimOutput(
        sim_dir=sim_dir,
        config_path=config_path,
        results_dir=results_dir,
        traces_dir=traces_dir,
        summary=summary,
        autotune=autotune,
        effective_placement=effective,
    )
    _validate_sim_output(cfg, output, _trace_metadata)
    return output


def evaluate_candidate(
    cfg: TuneConfig,
    inputs: TuneInputs,
    state: TuneState,
    *,
    sequence: int = 0,
    force_compile: bool = False,
    compile_fn: Callable | None = None,
    refine_fn: Callable | None = None,
    sim_fn: Callable | None = None,
    replay_placement: Path | None = None,
    run_id: str | None = None,
) -> RoundRecord:
    """一个候选的完整执行；失败只淘汰自己并保留诊断。"""
    start = time.monotonic()
    cid = candidate_id(state)
    if run_id is None:
        run_id = run_identity(cfg, inputs)
    artifacts: CandidateArtifacts | None = None
    compile_key_value = ""
    sim_key_value = ""
    sim_dir: Path | None = None
    stage = "compile"
    try:
        artifacts = build_candidate(
            cfg, inputs, state, force_compile=force_compile,
            compile_fn=compile_fn, refine_fn=refine_fn)
        compile_key_value = artifacts.compile_key
        stage = "verify"
        verify_compiled(cfg, inputs, state, artifacts)
        expected_native = _expected_native_identity(cfg, state.profile)
        key = sim_key(cfg, inputs, state, artifacts.compile_key,
                      replay_placement)
        sim_key_value = key
        cached = None
        if (not force_compile and cfg.cache.mode == "fast"
                and replay_placement is None):
            cached = _find_cached_sim(
                cfg, inputs, key, expected_native=expected_native)
        if cached is not None:
            output = cached
        else:
            stage = "simulate"
            sim_dir = (cfg.output.dir / "sim" / key
                       / f"attempt-{uuid.uuid4().hex[:8]}")
            config_path, _ = build_sim(cfg, inputs, state, artifacts, sim_dir,
                                       replay_placement=replay_placement)
            output = (sim_fn or run_simulation)(cfg, sim_dir, config_path)
        sim_dir = output.sim_dir
        stage = "simulate"
        _validate_sim_output(
            cfg, output, expected_native=expected_native)
        mode = _execution_mode(cfg, replay_placement)
        expected_autotune = _autotune_identity(cfg)
        _validate_autotune_metrics(
            mode, output.autotune, expected=expected_autotune)
        if cached is None:
            _publish_sim_complete(
                cfg, inputs, sim_dir, key, artifacts.compile_key, output, mode,
                expected_autotune=expected_autotune,
                expected_native=expected_native)
        metrics = _sim_metrics(output.summary)
        metrics["autotune"] = _autotune_summary(mode, output.autotune)
        score = metrics["total_time_s"]
        maturity = _result_maturity(output, cfg, state)
        return RoundRecord(
            sequence=sequence,
            candidate_id=cid,
            compile_key=artifacts.compile_key,
            sim_key=key,
            cache=cfg.cache.mode,
            status="success",
            result_maturity=maturity,
            state=state,
            score=score,
            run_id=run_id,
            compile_cache_hit=artifacts.cache_hit,
            sim_cache_hit=cached is not None,
            metrics=metrics,
            stop_reasons={"search": str(output.autotune.get("stop_reason", "")),
                          "inner_evaluations": output.autotune.get("evaluations", 0)},
            elapsed_s=time.monotonic() - start,
            search_elapsed_s=float(output.autotune.get("elapsed_s", 0.0)),
            inner_evaluations=int(output.autotune.get("evaluations", 0)),
            validation_evaluations=int(
                output.autotune.get("final_backend_evaluations", 1)),
            artifacts=artifacts.manifest,
            failure=None,
            paths={
                "compile_dir": str(artifacts.compile_dir),
                "sim_dir": str(sim_dir),
                "config": str(output.config_path),
                "effective_placement": (
                    "" if output.effective_placement is None
                    else str(output.effective_placement)),
            },
        )
    except Exception as exc:  # 单候选失败不终止搜索
        paths = {}
        if artifacts is not None:
            paths["compile_dir"] = str(artifacts.compile_dir)
        if sim_dir is not None:
            paths["sim_dir"] = str(sim_dir)
        return RoundRecord(
            sequence=sequence, candidate_id=cid,
            compile_key=compile_key_value, sim_key=sim_key_value,
            cache=cfg.cache.mode, status="error",
            result_maturity="model_prediction", state=state, score=None,
            run_id=run_id,
            compile_cache_hit=bool(artifacts and artifacts.cache_hit),
            failure=f"{stage}: {type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - start,
            paths=paths,
        )


def resolve_base_genesim_config(profile: HardwareProfile,
                                profile_path: Path) -> Path:
    """画像内基配置路径先相对画像文件，再相对 GeneSim 根。"""
    raw = Path(profile.base_genesim_config)
    if raw.is_absolute():
        return raw
    candidate = Path(profile_path).resolve().parent / raw
    if candidate.is_file():
        return candidate
    candidate = genesim_root() / raw
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"找不到画像的基配置：{profile.base_genesim_config}")


# ---- 内部实现 ----


def _request_kv_tokens(requests) -> tuple[int, ...]:
    """逐请求 KV 上下文峰值；与 GeneSim 的 prefill/decode 上界同一口径。"""
    tokens = []
    for bound in requests:
        if bound.prompt_tokens > 0:
            # prefill 的峰值出现在生成阶段：initial + prompt + output - 1。
            value = (bound.initial_kv + bound.prompt_tokens
                     + max(0, bound.output_tokens - 1))
        else:
            value = bound.initial_kv + bound.output_tokens
        tokens.append(int(value))
    return tuple(tokens)


def _max_inflight_requests(cfg: TuneConfig, state: TuneState,
                           request_count: int) -> int:
    """与 GeneSim `_effective_max_inflight_requests` 相同的并发上限。"""
    GeneSimConfig = genesim_config_class()
    base = resolve_base_genesim_config(state.profile, cfg.profile_path)
    effective = GeneSimConfig(str(base)).override(
        **_genesim_profile_overrides(state.profile))
    configured = effective.get(
        "scheduler.max_inflight_requests", request_count)
    if (isinstance(configured, bool) or not isinstance(configured, int)
            or configured <= 0):
        raise ValueError(
            "scheduler.max_inflight_requests 必须是正整数，"
            f"收到 {configured!r}")
    return min(configured, request_count)


def _real_compile(cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
                  out_dir: Path, force_compile: bool):
    from genesim_bridge.placement_export import export_placement_for_strategy

    model = _load_model(inputs.model_dir)
    strategy = build_strategy(state, model_dims(inputs.model_dir))
    plan = _fixed_partition_plan(state)
    request_kv_tokens = _request_kv_tokens(inputs.trace.requests)
    return export_placement_for_strategy(
        model, inputs.ir_path, out_dir,
        strategy=strategy, plan=plan,
        hw=state.profile.to_hw_budget(),
        hardware=state.profile.to_pim_hardware_config(
            state.strategy.num_dpus, state.op.num_tasklets,
            optimizations=state.resolved_optimization(cfg)),
        prefill_seq_len=cfg.model.prefill_seq_len,
        max_seq=inputs.max_seq,
        profile=state.profile,
        force_compile=force_compile,
        request_kv_tokens=request_kv_tokens,
        max_concurrent_requests=_max_inflight_requests(
            cfg, state, len(request_kv_tokens)),
    )


def _real_refine(cfg: TuneConfig, inputs: TuneInputs, state: TuneState,
                 placement, out_ir: Path, cost_path: Path,
                 force_compile: bool) -> None:
    from genesim_bridge import export_costs_to_genesim, load_local_shapes

    level = refine_level()
    local = load_local_shapes(placement.sidecar)
    export_costs_to_genesim(
        ir_path=inputs.ir_path,
        out_ir_path=out_ir,
        sidecar_path=cost_path,
        seq_len=cfg.model.prefill_seq_len,
        cross_validate=level != "oplevel",
        ir_level=level,
        local_shapes=local,
        # 同时给本地形状与融合清单：形状按本地产物测，融合按图编译决策折。
        placement_sidecar=placement.sidecar,
        hardware=state.profile.to_pim_hardware_config(
            state.strategy.num_dpus, state.op.num_tasklets,
            optimizations=state.resolved_optimization(cfg)),
        force_compile=force_compile,
    )


def _fixed_partition_plan(state: TuneState) -> PartitionPlan:
    """固定 PU 映射：逻辑 DPU 按顺序落到物理 ClusterPU。"""
    profile = state.profile
    mapping = []
    for dpu_id in range(state.strategy.num_dpus):
        device = dpu_id // profile.clusters_per_device
        cluster = dpu_id % profile.clusters_per_device
        mapping.append((device, cluster))
    return PartitionPlan(
        num_dpus=state.strategy.num_dpus,
        num_stages=state.strategy.num_stages,
        weight_rules=(),
        dpu_ids=tuple(range(state.strategy.num_dpus)),
        dpu_to_cluster=tuple(mapping),
        source="tune_fixed",
    )


_MODEL_CACHE: dict[str, Any] = {}


def _load_model(model_dir: Path):
    """同一进程只加载一次权重，候选之间复用。"""
    key = str(Path(model_dir).resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import torch
    from transformers import LlamaForCausalLM

    torch.set_grad_enabled(False)
    model = LlamaForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16).eval()
    _MODEL_CACHE[key] = model
    return model


def _find_cached_sim(
    cfg: TuneConfig, inputs: TuneInputs, key: str, *,
    expected_native: Mapping | None = None,
) -> SimOutput | None:
    """快速档复用同一 sim_key 的完整成功结果；strict 永远新开尝试目录。

    只有完成清单存在且其中每个文件哈希仍与磁盘一致时才命中：只写了一份
    summary、缺轨迹/配置、结果被改写或实际计时后端与键声明不一致的目录
    一律跳过重建。读取前先按磁盘字节复核输入快照，输入已变时直接失败，
    不能靠目录内部自洽放行另一条轨迹的旧结果。
    """
    from scripts.run_full_pipeline import _trace_metadata

    view = verify_input_snapshot(cfg, inputs)
    root = cfg.output.dir / "sim" / key
    if not root.is_dir():
        return None
    for attempt in sorted(root.glob("attempt-*")):
        completion = _read_completion(attempt)
        if completion is None or completion.get("sim_key") != key:
            continue
        try:
            _verify_completion_files(attempt, completion)
            _validate_completion_payload(
                completion, attempt, _execution_mode(cfg, None),
                expected_autotune=_autotune_identity(cfg),
                expected_native=expected_native,
                inputs=inputs, expected_view=view)
        except (OSError, ValueError, RuntimeError):
            continue
        summary_path = attempt / "results" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        autotune_path = attempt / "results" / "autotune_metrics.json"
        if not autotune_path.is_file():
            continue
        try:
            autotune = json.loads(autotune_path.read_text())
        except json.JSONDecodeError:
            continue
        effective = attempt / "effective_placement.json"
        output = SimOutput(
            sim_dir=attempt,
            config_path=attempt / "config.yaml",
            results_dir=attempt / "results",
            traces_dir=attempt / "pim_traces",
            summary=summary,
            autotune=autotune,
            effective_placement=effective if effective.is_file() else None,
        )
        try:
            _validate_sim_output(
                cfg, output, _trace_metadata,
                expected_native=expected_native)
        except RuntimeError:
            continue
        return output
    return None


def _validate_sim_output(cfg: TuneConfig, output: SimOutput,
                         metadata_reader: Callable | None = None,
                         *, expected_native: Mapping | None = None) -> None:
    """统一的成功判据：显式 pipeline 来源、有限正时延与有效轨迹覆盖。

    首次执行和缓存命中共用这一个入口，避免两处各写一份宽严不同的检查。
    `expected_native` 是仿真键声明的计时后端；摘要里的实际台账必须一致，
    否则原生与回退的计时语义会被缓存身份掩盖。
    """
    summary = output.summary
    if "result_producer" not in summary:
        raise RuntimeError("仿真摘要缺少 result_producer，无法确认结果来源")
    producer = summary["result_producer"]
    if producer != "pipeline_simulator":
        raise RuntimeError(f"结果来源不是 pipeline：{producer!r}")
    if expected_native is not None:
        error = _native_identity_error(
            expected_native, summary.get("pim_native_backend"))
        if error is not None:
            raise RuntimeError(f"仿真实际原生后端与键声明不一致：{error}")
    value = summary.get("total_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"仿真总时延不是数值：{value!r}")
    score = float(value)
    if not math.isfinite(score) or score <= 0:
        raise RuntimeError(f"仿真总时延必须是有限正数：{value!r}")
    throughput = summary.get("throughput_tokens_per_s")
    if throughput is not None:
        if isinstance(throughput, bool) or not isinstance(
                throughput, (int, float)) or not math.isfinite(float(throughput)):
            raise RuntimeError(f"吞吐量不是有限数值：{throughput!r}")
    if metadata_reader is not None:
        _verify_trace_sources(output.traces_dir, metadata_reader)


_SIM_METRIC_KEYS = (
    "throughput_tokens_per_s",
    "compute_efficiency", "communication_efficiency", "memory_efficiency",
    "communication_time_s", "communication_ratio",
    "memory_service_time_s", "memory_service_ratio",
    "pipeline_utilization", "pim_capacity_fits",
    "pim_capacity_max_utilization", "pim_capacity_max_resident_bytes",
    "pim_capacity_limit_bytes_per_cluster",
    "pim_capacity_limit_bytes_per_tensor_pu", "pim_tensors_per_cluster",
    # 实际原生后端还是轨迹回退，决定计时口径的可解释性。
    "pim_native_backend",
)


def _sim_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    """逐轮保留排序标量之外的效率、通信与容量指标。"""
    metrics: dict[str, Any] = {"total_time_s": float(summary["total_time_s"])}
    for key in _SIM_METRIC_KEYS:
        if key in summary:
            metrics[key] = summary[key]
    efficiency = summary.get("efficiency_metrics")
    if isinstance(efficiency, Mapping):
        # 保留完整 schema 与来源，并把三轴效率展开成可直接排序/复核的标量。
        metrics["efficiency_metrics"] = dict(efficiency)
        derived = {
            "compute_efficiency": _nested_number(
                efficiency, "compute", "system_mfu"),
            "communication_efficiency": _nested_number(
                efficiency, "communication", "efficiency"),
            "memory_efficiency": _nested_number(
                efficiency, "memory", "efficiency"),
        }
        for key, value in derived.items():
            if value is not None:
                metrics[key] = value
    return metrics


def _nested_number(mapping: Mapping[str, Any], *path: str) -> float | None:
    value: Any = mapping
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# 仿真完成清单的必需角色：缺任一项都不能发布，也不能命中快速缓存。
_SIM_COMPLETION_REQUIRED = (
    "results/summary.json",
    # 内层搜索与最终重放次数是闭环审计的一部分，缺它与缺 summary 同级。
    "results/autotune_metrics.json",
    "config.yaml",
    "manifest.json",
    "hardware_contract.json",
    "effective_placement.json",
)
_SIM_TRACE_NAME = re.compile(r"^op_(\d+)(?:_[^.]+)?\.pim_trace$")


def _execution_mode(cfg: TuneConfig, replay_placement: Path | None) -> str:
    """本次仿真属于内层搜索、固定映射还是严格重放。"""
    if replay_placement is not None:
        return "replay"
    return "search" if cfg.placement.enabled else "fixed"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _autotune_identity(cfg: TuneConfig) -> tuple[str, str, str]:
    """本次生效放置策略的算法、搜索后端与验证后端。"""
    placement = cfg.placement
    return (
        placement.algorithm,
        placement.evaluation_backend,
        placement.validation_backend,
    )


def _validate_autotune_metrics(
    mode: str, metrics: Mapping[str, Any],
    *, expected: tuple[str, str, str] | None = None,
) -> None:
    """按执行模式复核内层搜索与最终重放次数，缺证据不得发布或命中。

    原生搜索必须记录算法、搜索后端、逻辑/实际评估数和一次最终流水线
    执行；固定映射与严格重放显式记录零搜索加一次最终执行。计数必须与
    总后端评估数自洽，缺失时不能靠调用方默认值补成 0。`expected` 是本次
    生效配置派生的算法/搜索后端/验证后端，记录与之不符即拒绝。
    """
    if not isinstance(metrics, Mapping) or not metrics:
        raise RuntimeError(
            "仿真缺少 autotune_metrics.json，内层搜索与最终重放次数无法审计")
    evaluations = metrics.get("evaluations")
    if not _is_int(evaluations) or evaluations < 0:
        raise RuntimeError(f"autotune 评估次数不合法：{evaluations!r}")
    final = metrics.get("final_backend_evaluations")
    if final != 1:
        raise RuntimeError(
            f"autotune 的最终后端评估次数必须是 1，收到 {final!r}")
    total = metrics.get("total_backend_evaluations")
    if total != evaluations + 1:
        raise RuntimeError(
            "autotune 的总后端评估数与搜索/最终执行次数不一致："
            f"total={total!r}, evaluations={evaluations!r}")
    validation_backend = metrics.get("validation_backend")
    expected_validation = "pipeline" if expected is None else expected[2]
    if validation_backend != expected_validation:
        raise RuntimeError(
            "autotune 的最终验证后端与生效配置不一致："
            f"{validation_backend!r} vs {expected_validation!r}")
    if mode == "search":
        if metrics.get("enabled") is not True:
            raise RuntimeError("内层搜索已启用，但 autotune 指标标记为未启用")
        algorithm = metrics.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise RuntimeError("autotune 指标缺少搜索算法")
        backend = metrics.get("evaluation_backend")
        if backend not in ("pipeline", "predictor"):
            raise RuntimeError(f"autotune 的搜索后端不合法：{backend!r}")
        if expected is not None:
            # 记录必须绑定本次生效配置，不能只落在允许枚举里。
            if algorithm != expected[0]:
                raise RuntimeError(
                    "autotune 的搜索算法与生效配置不一致："
                    f"{algorithm!r} vs {expected[0]!r}")
            if backend != expected[1]:
                raise RuntimeError(
                    "autotune 的搜索后端与生效配置不一致："
                    f"{backend!r} vs {expected[1]!r}")
        if evaluations < 1:
            raise RuntimeError("内层搜索没有任何评估记录")
        same_backend = metrics.get("search_and_validation_backend_match")
        matches = metrics.get("final_backend_matches_search")
        if not isinstance(same_backend, bool) or not isinstance(matches, bool):
            raise RuntimeError("autotune 指标缺少搜索/重放一致性标记")
        if backend == "pipeline" and not (same_backend and matches):
            raise RuntimeError("pipeline 搜索的最终重放与搜索结果不一致")
        return
    expected_reason = "replayed" if mode == "replay" else "disabled"
    if metrics.get("enabled") is not False:
        raise RuntimeError(f"{mode} 模式不得记录为启用了内层搜索")
    if evaluations != 0:
        raise RuntimeError(f"{mode} 模式的搜索评估次数必须为 0")
    if metrics.get("stop_reason") != expected_reason:
        raise RuntimeError(
            f"{mode} 模式的停止原因应为 {expected_reason!r}，"
            f"收到 {metrics.get('stop_reason')!r}")
    if mode == "replay" and metrics.get("replayed") is not True:
        raise RuntimeError("严格重放的 autotune 指标没有标记 replayed")
    if mode == "fixed" and metrics.get("replayed") is True:
        raise RuntimeError("固定映射的 autotune 指标不能标记为重放")


def _autotune_summary(mode: str, metrics: Mapping[str, Any]) -> dict:
    """轮次记录里的内层预算摘要，便于事后审计而不读大文件。"""
    return {
        "execution_mode": mode,
        "algorithm": metrics.get("algorithm"),
        "evaluation_backend": metrics.get("evaluation_backend"),
        "validation_backend": metrics.get("validation_backend"),
        "evaluations": metrics.get("evaluations"),
        "final_backend_evaluations": metrics.get("final_backend_evaluations"),
        "total_backend_evaluations": metrics.get("total_backend_evaluations"),
        "search_and_validation_backend_match": metrics.get(
            "search_and_validation_backend_match"),
        "final_backend_matches_search": metrics.get(
            "final_backend_matches_search"),
        "stop_reason": metrics.get("stop_reason"),
    }


def _publish_sim_complete(
    cfg: TuneConfig, inputs: TuneInputs, sim_dir: Path, key: str,
    compile_key: str, output: SimOutput, mode: str,
    *, expected_autotune: tuple[str, str, str] | None = None,
    expected_native: Mapping | None = None,
) -> None:
    """全部后置校验通过后原子发布完成清单；缺任一角色都不得发布。"""
    # 发布前按磁盘字节复核输入；完成清单登记的身份必须与当前输入一致。
    view = verify_input_snapshot(cfg, inputs)
    root = Path(output.sim_dir)
    actual_native = output.summary.get("pim_native_backend")
    if not isinstance(actual_native, Mapping):
        raise RuntimeError(
            "仿真摘要缺少 pim_native_backend，无法证明实际计时后端")
    if expected_native is not None:
        error = _native_identity_error(expected_native, actual_native)
        if error is not None:
            raise RuntimeError(f"仿真实际原生后端与键声明不一致：{error}")
    files: dict[str, str] = {}
    for role in _SIM_COMPLETION_REQUIRED:
        path = root / role
        if not path.is_file():
            raise RuntimeError(f"仿真完成清单缺少必需产物 {role}：{path}")
        files[role] = file_sha256(path)
    # trace 必须与最终执行图一一对应：挂 PU 的算子有 trace，其余没有。
    for op_id, path in _completion_traces(root).items():
        files[path.relative_to(root).as_posix()] = file_sha256(path)
    payload = {
        "schema_version": 1,
        "complete": True,
        "sim_key": key,
        "compile_key": compile_key,
        "execution_mode": mode,
        "autotune_metrics_sha256": files["results/autotune_metrics.json"],
        "result_producer": output.summary["result_producer"],
        "total_time_s": float(output.summary["total_time_s"]),
        "pim_native_backend": dict(actual_native),
        "expected_native_backend": (
            None if expected_native is None else dict(expected_native)),
        "files": files,
    }
    _validate_completion_payload(
        payload, root, mode, expected_autotune=expected_autotune,
        expected_native=expected_native, inputs=inputs, expected_view=view)
    target = root / "completion.json"
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, target)


def _completion_traces(attempt: Path) -> dict[int, Path]:
    """按算子编号索引完成清单里的 pim trace；重复或非法命名直接拒绝。"""
    traces: dict[int, Path] = {}
    for path in sorted(Path(attempt).glob("pim_traces/op_*.pim_trace")):
        match = _SIM_TRACE_NAME.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"pim trace 文件名无法解析算子编号：{path.name}")
        op_id = int(match.group(1))
        if op_id in traces:
            raise RuntimeError(f"算子 {op_id} 有多份 pim trace，无法确认执行图覆盖")
        traces[op_id] = path
    return traces


def _validate_completion_payload(
    payload: Mapping[str, Any], attempt: Path,
    expected_mode: str | None = None,
    *, expected_autotune: tuple[str, str, str] | None = None,
    expected_native: Mapping | None = None,
    inputs: TuneInputs,
    expected_view: InputSnapshotView,
) -> None:
    """写入与读取共用的完成判据：角色齐全、键同源、映射与 trace 对得上。

    除目录内部自洽外，还要按当前已复核输入交叉核对 manifest 的输入身份、
    画像契约登记的轨迹与窗口、配置实际消费的轨迹以及摘要的实际处理规模，
    防止另一条轨迹的结果套用旧 sim_key 发布或被缓存复用。
    """
    attempt = Path(attempt)
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("仿真完成清单缺少文件登记")
    missing = [role for role in _SIM_COMPLETION_REQUIRED if role not in files]
    if missing:
        raise RuntimeError(f"仿真完成清单缺少必需角色：{missing}")
    manifest_path = attempt / "manifest.json"
    summary_path = attempt / "results" / "summary.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"仿真完成清单引用的 manifest/summary 无法读取：{exc}")
    # 完成清单的键必须与本次仿真写下的 manifest 同源，不能自报一套。
    for field, expected in (("sim_key", payload.get("sim_key")),
                            ("compile_key", payload.get("compile_key"))):
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"完成清单的 {field} 与仿真 manifest 不一致："
                f"{expected!r} vs {manifest.get(field)!r}")
    expected_inputs = {
        "model": inputs.model_sha256,
        "ir": inputs.ir_sha256,
        "trace": inputs.trace_sha256,
    }
    recorded_inputs = manifest.get("input_sha256")
    if not isinstance(recorded_inputs, Mapping):
        raise RuntimeError("仿真 manifest 缺少输入身份，无法核对轨迹归属")
    for role, value in expected_inputs.items():
        if recorded_inputs.get(role) != value:
            raise RuntimeError(
                f"仿真 manifest 的{role}输入身份与当前输入不一致："
                f"{recorded_inputs.get(role)!r} vs {value!r}")
    config_path = attempt / "config.yaml"
    contract_path = attempt / "hardware_contract.json"
    if manifest.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("仿真 manifest 的配置摘要与磁盘不一致")
    if manifest.get("hardware_contract_sha256") != file_sha256(contract_path):
        raise RuntimeError("仿真 manifest 的画像契约摘要与磁盘不一致")
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"仿真画像契约无法读取：{exc}")
    if not isinstance(contract, Mapping):
        raise RuntimeError("仿真画像契约必须是 JSON 对象")
    binding = contract.get("compile") or {}
    if not isinstance(binding, Mapping):
        raise RuntimeError("仿真画像契约的 compile 绑定必须是对象")
    if binding.get("compile_key") != manifest.get("compile_key"):
        raise RuntimeError("画像契约的编译键与仿真 manifest 不一致")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("画像契约缺少实际消费轨迹的绑定")
    if artifacts.get("trace_sha256") != expected_inputs["trace"]:
        raise RuntimeError("画像契约绑定的请求轨迹与当前输入不一致")
    if artifacts.get("trace_num_requests") != expected_view.num_requests:
        raise RuntimeError("画像契约的请求窗口与当前输入不一致")
    if summary.get("result_producer") != payload.get("result_producer"):
        raise RuntimeError("完成清单的结果来源与仿真摘要不一致")
    actual_native = summary.get("pim_native_backend")
    if not isinstance(actual_native, Mapping):
        raise RuntimeError("仿真摘要缺少 pim_native_backend，无法证明计时后端")
    if payload.get("pim_native_backend") != actual_native:
        raise RuntimeError("完成清单的原生后端身份与仿真摘要不一致")
    if expected_native is not None:
        error = _native_identity_error(expected_native, actual_native)
        if error is not None:
            raise RuntimeError(f"仿真实际原生后端与键声明不一致：{error}")
    mode = payload.get("execution_mode")
    if mode not in ("search", "fixed", "replay"):
        raise RuntimeError(f"完成清单的执行模式 {mode!r} 不合法")
    if expected_mode is not None and mode != expected_mode:
        raise RuntimeError(
            f"完成清单的执行模式 {mode!r} 与本次运行 {expected_mode!r} 不一致")
    autotune_path = attempt / "results" / "autotune_metrics.json"
    try:
        autotune = json.loads(autotune_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"完成清单缺少可读取的 autotune 指标：{exc}")
    digest = file_sha256(autotune_path)
    if files.get("results/autotune_metrics.json") != digest:
        raise RuntimeError("完成清单登记的 autotune 指标哈希与磁盘不一致")
    if payload.get("autotune_metrics_sha256") != digest:
        raise RuntimeError("完成清单的 autotune 指标哈希字段与文件不一致")
    _validate_autotune_metrics(mode, autotune, expected=expected_autotune)
    try:
        summary_total = float(summary["total_time_s"])
        recorded_total = float(payload["total_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"完成清单的仿真时延无法核对：{exc}")
    if not math.isclose(recorded_total, summary_total, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("完成清单的仿真时延与仿真摘要不一致")
    placement_path = attempt / "effective_placement.json"
    try:
        placement = json.loads(placement_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"最终放置无法读取：{exc}")
    if placement.get("schema_version") != 1:
        raise RuntimeError("最终放置版本不受支持")
    if placement.get("targets_include_relay") is not False:
        raise RuntimeError("最终放置包含合成 relay 目标")
    targets = placement.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("最终放置没有执行图目标")
    target_ids: set[int] = set()
    attached_ids: set[int] = set()
    for entry in targets:
        if not isinstance(entry, Mapping):
            raise RuntimeError("最终放置目标不是对象")
        try:
            op_id = int(entry["op_id"])
            attached = int(entry["attached_pu_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"最终放置目标字段不完整：{exc}")
        target_ids.add(op_id)
        if attached >= 0:
            attached_ids.add(op_id)
    try:
        config = yaml.safe_load((attempt / "config.yaml").read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"仿真配置无法读取：{exc}")
    # 配置实际消费的轨迹必须就是本次复核过的那条，窗口也必须一致。
    trace_config = ((config or {}).get("trace") or {})
    trace_path = trace_config.get("path")
    if not isinstance(trace_path, str) or not Path(trace_path).is_file():
        raise RuntimeError("仿真配置没有登记可读取的请求轨迹路径")
    if file_sha256(Path(trace_path)) != expected_inputs["trace"]:
        raise RuntimeError("仿真配置消费的轨迹内容与已复核输入不一致")
    if trace_config.get("num_requests") != expected_view.num_requests:
        raise RuntimeError("仿真配置的请求窗口与已复核输入不一致")
    # 摘要的实际处理规模是仿真真正消费窗口的直接证据。
    if summary.get("processed_tokens") != expected_view.processed_tokens:
        raise RuntimeError(
            "仿真摘要实际处理的 token 数与请求窗口不一致："
            f"{summary.get('processed_tokens')!r} vs "
            f"{expected_view.processed_tokens}")
    if summary.get("completed_requests") != expected_view.num_requests:
        raise RuntimeError(
            "仿真摘要完成的请求数与请求窗口不一致："
            f"{summary.get('completed_requests')!r} vs "
            f"{expected_view.num_requests}")
    configured = ((config or {}).get("scheduler") or {}).get(
        "effective_placement_file")
    if not isinstance(configured, str) or (
            Path(configured).resolve() != placement_path.resolve()):
        raise RuntimeError("仿真配置的最终放置输出与完成清单登记的文件不一致")
    traces = _completion_traces(attempt)
    if not traces:
        raise RuntimeError("仿真完成清单没有任何 pim trace")
    # 完成清单的哈希表必须恰好覆盖用于核对覆盖率的全部 trace。
    registered = {
        role for role in files if role.startswith("pim_traces/")}
    actual = {
        path.relative_to(attempt).as_posix() for path in traces.values()}
    if registered != actual:
        raise RuntimeError(
            "仿真完成清单登记的 trace 与磁盘上的 trace 集合不一致")
    extra = sorted(set(traces) - target_ids)
    if extra:
        raise RuntimeError(f"pim trace 不在最终执行图里：{extra[:8]}")
    # 最终由 PIM 执行的目标必须与 trace 集合精确相等：搜索把算子迁离
    # PIM 后残留的轨迹属于另一次放置，不能当成本轮覆盖放行。
    residual = sorted(set(traces) - attached_ids)
    if residual:
        raise RuntimeError(
            "pim trace 对应的算子没有挂在 PU 上（非 PIM 残留）："
            f"{residual[:8]}")
    uncovered = sorted(attached_ids - set(traces))
    if uncovered:
        raise RuntimeError(
            f"最终执行图上 {len(uncovered)} 个算子的 trace 未登记"
            f"（如 {uncovered[:8]}）")


def _read_completion(attempt: Path) -> Mapping[str, Any] | None:
    path = Path(attempt) / "completion.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("schema_version") != 1 or payload.get("complete") is not True:
        return None
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        return None
    if any(role not in files for role in _SIM_COMPLETION_REQUIRED):
        return None
    return payload


def _verify_completion_files(attempt: Path, payload: Mapping[str, Any]) -> None:
    """完成清单里每个文件都必须存在且内容未变。"""
    for role, digest in payload["files"].items():
        path = Path(attempt) / role
        if not path.is_file():
            raise ValueError(f"仿真完成清单的 {role} 已缺失")
        if file_sha256(path) != digest:
            raise ValueError(f"仿真完成清单的 {role} 内容已变")


def _read_compiled(cache_root: Path, key: str,
                   inputs: TuneInputs) -> CandidateArtifacts | None:
    """快速档命中任一完整 attempt；最新发布的优先，坏目录隔离后跳过。"""
    for candidate_dir in _compiled_attempt_dirs(cache_root):
        artifacts = _read_compiled_attempt(candidate_dir, key, inputs)
        if artifacts is not None:
            return artifacts
        if candidate_dir != cache_root:
            _quarantine_attempt(candidate_dir)
    return None


def _quarantine_attempt(attempt: Path) -> None:
    """把不完整的 attempt 改名移出候选集合，后续读取不再重复尝试。"""
    target = attempt.with_name(
        f"invalid-{attempt.name}-{uuid.uuid4().hex[:8]}")
    try:
        attempt.rename(target)
    except OSError:
        # 改名失败不影响本次重编；下次仍会按不完整目录跳过。
        return


def _compiled_attempt_dirs(cache_root: Path) -> list[Path]:
    if not cache_root.is_dir():
        return []
    attempts = [
        path for path in cache_root.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")]
    attempts.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    # 旧版本的扁平发布目录仍可读取，只用于兼容既有产物，不再写入。
    if (cache_root / "artifacts.json").is_file():
        attempts.append(cache_root)
    return attempts


def _read_compiled_attempt(cache_dir: Path, key: str,
                           inputs: TuneInputs) -> CandidateArtifacts | None:
    manifest_path = cache_dir / "artifacts.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = ArtifactManifest.from_json(json.loads(manifest_path.read_text()))
        if manifest.compile_key != key:
            return None
        verify_compile_products(cache_dir, manifest)
        # 目录与元数据被错误迁移时，清单输入角色同样不属于当前输入。
        verify_manifest_inputs(manifest, inputs)
    except (ValueError, json.JSONDecodeError, KeyError):
        return None
    return CandidateArtifacts(
        compile_key=key,
        compile_dir=cache_dir,
        placement_sidecar=cache_dir / "placement.json",
        partition_plan=cache_dir / "partition_plan.json",
        refined_ir=cache_dir / "refined.ir",
        cost_sidecar=cache_dir / "cost.json",
        manifest=manifest,
        cache_hit=True,
    )


def _replace_cache_hit(artifacts: CandidateArtifacts, hit: bool) -> CandidateArtifacts:
    return CandidateArtifacts(
        compile_key=artifacts.compile_key,
        compile_dir=artifacts.compile_dir,
        placement_sidecar=artifacts.placement_sidecar,
        partition_plan=artifacts.partition_plan,
        refined_ir=artifacts.refined_ir,
        cost_sidecar=artifacts.cost_sidecar,
        manifest=artifacts.manifest,
        cache_hit=hit,
    )


def _semantic_sidecar_sha256(sidecar_path: Path) -> str:
    """放置 sidecar 的语义内容摘要：忽略可搬运的物理路径字段。

    `source_ir`、PIM MLIR 路径与工具链安装路径随导出机器或发布 attempt
    变化，不构成候选身份；策略、映射、预算与逐算子产物哈希全量参与。
    """
    sidecar = json.loads(Path(sidecar_path).read_text())
    sidecar.pop("source_ir", None)
    for entry in (sidecar.get("operators") or {}).values():
        if isinstance(entry, dict):
            entry.pop("pimir_path", None)
    toolchain = (sidecar.get("compiler") or {}).get("toolchain")
    if isinstance(toolchain, dict):
        graph = toolchain.get("graph_compiler")
        if isinstance(graph, dict):
            graph.pop("root", None)
        flagtree = toolchain.get("flagtree")
        if isinstance(flagtree, dict):
            flagtree.pop("prefix", None)
            flagtree.pop("source_root", None)
        for entry in (toolchain.get("tools") or {}).values():
            if isinstance(entry, dict):
                entry.pop("path", None)
    return _sha256_json(sidecar)


def _bind_refined_sidecar(refined_ir: Path, sidecar_path: Path,
                          inputs: TuneInputs) -> None:
    """把放置 sidecar 与源模型/原始 IR 身份写进精化 IR，构成成对自证。"""
    refined = json.loads(Path(refined_ir).read_text())
    refined["placement_sidecar_sha256"] = _semantic_sidecar_sha256(sidecar_path)
    # 精化 IR 由该模型与该原始 IR 派生；旧键或错误迁移的目录都不能自证。
    refined["source_model_sha256"] = inputs.model_sha256
    refined["source_ir_sha256"] = inputs.ir_sha256
    Path(refined_ir).write_text(json.dumps(refined, indent=2))


def _archive_pimir(root: Path, sidecar_path: Path, final_dir: Path) -> None:
    """把 pim mlir 收进候选目录，并把 sidecar 路径改写到发布后的目录。"""
    sidecar = json.loads(Path(sidecar_path).read_text())
    archive = root / "pimir"
    archive.mkdir(exist_ok=True)
    published = final_dir / "pimir"
    for entry in sidecar.get("operators", {}).values():
        source = entry.get("pimir_path")
        if not source:
            continue
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"sidecar 指向的 pim mlir 不存在：{path}")
        digest = file_sha256(path)
        target = archive / f"{digest}.mlir"
        if not target.is_file():
            shutil.copy2(path, target)
        # 发布后的绝对路径：GeneSim 直接打开，缓存目录整体搬迁时仍可按内容找回。
        entry["pimir_path"] = str((published / f"{digest}.mlir").resolve())
        entry["pimir_sha256"] = digest
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))


def _verify_trace_sources(traces_dir: Path, metadata_reader) -> None:
    sources: dict[str, int] = {}
    for path in sorted(Path(traces_dir).glob("op_*.pim_trace")):
        meta = metadata_reader(path)
        source = (meta.get("compile_signature") or {}).get(
            "trace_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    if not sources:
        raise RuntimeError("没有找到任何 pim trace，无法判断来源")
    if sources.get("template") or sources.get("unknown"):
        raise RuntimeError(f"pim trace 来源不合格：{sources}（要求零模板、零未知）")


def _genesim_profile_overrides(profile: HardwareProfile) -> dict:
    return profile.genesim_overrides()


def _move_weights(placement) -> dict[str, float]:
    """把用户显式权重按八类全量展开，未指定项显式置 0。

    GeneSimConfig.override 对映射做深合并；只下发子集会保留基配置里的
    其他移动权重，使“只启用一类”实际启用全部移动。
    """
    configured = dict(placement.moves.weights)
    if not configured:
        return {name: 1.0 for name in sorted(MOVES)}
    return {
        name: float(configured.get(name, 0.0))
        for name in sorted(MOVES)
    }


def _autotune_overrides(cfg: TuneConfig, sim_dir: Path) -> dict:
    placement = cfg.placement
    algorithm = placement.algorithm
    payload = {
        "enabled": True,
        "algorithm": algorithm,
        "evaluation_backend": placement.evaluation_backend,
        "validation_backend": placement.validation_backend,
        "objective_metric": placement.objective_metric,
        "objective_direction": (
            "maximize" if placement.objective_metric
            == "system_efficiency_utility" else "minimize"),
        "objective_weights": dict(placement.objective_weights),
        "max_evaluations": placement.max_evaluations,
        "max_time_s": placement.max_time_s,
        "seed": placement.seed,
        "moves": {
            "max_segment_ops": placement.moves.max_segment_ops,
            "max_layer_block_layers": placement.moves.max_layer_block_layers,
            "weights": _move_weights(placement),
        },
        "algorithms": {algorithm: dict(placement.algorithm_params)},
        "output_ir_path": str(sim_dir / "effective.ir.json"),
        # 搜索成功后由 GeneSim 把最终映射写到 scheduler.effective_placement_file。
        "save_effective_placement": True,
    }
    if placement.max_proposals is not None:
        payload["max_proposals"] = placement.max_proposals
    return payload


def _flat_contract(profile: HardwareProfile) -> dict:
    """画像契约里的扁平期望值，供 GeneSim 在生效配置上逐项核对。

    期望值由 `HardwareProfile.genesim_overrides()` 递归生成，覆盖全部周期项、
    区间倍率、链路列表、主机模型和 CEVA 来源台账；不再靠手工白名单，新增
    画像字段会自动进入契约。键含点号时整张映射作为叶子，避免路径歧义。
    """
    contract: dict[str, Any] = {}
    for path, value in _contract_leaves(profile.genesim_overrides()):
        contract[path] = value
    contract["scheduler.require_compiler_pimir"] = True
    contract["scheduler.use_compiler_local_shapes"] = True
    return contract


def _contract_leaves(node: Any, prefix: str = ""):
    if isinstance(node, Mapping) and not any(
            "." in str(key) for key in node):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _contract_leaves(value, path)
        return
    yield prefix, node


def _placement_payload(cfg: TuneConfig) -> dict:
    placement = cfg.placement
    return {
        "enabled": placement.enabled,
        "algorithm": placement.algorithm,
        "evaluation_backend": placement.evaluation_backend,
        "validation_backend": placement.validation_backend,
        "objective_metric": placement.objective_metric,
        "objective_direction": (
            "maximize" if placement.objective_metric
            == "system_efficiency_utility" else "minimize"),
        "objective_weights": dict(placement.objective_weights),
        "max_evaluations": placement.max_evaluations,
        "max_time_s": placement.max_time_s,
        "max_proposals": placement.max_proposals,
        "seed": placement.seed,
        "moves": {
            "max_segment_ops": placement.moves.max_segment_ops,
            "max_layer_block_layers": placement.moves.max_layer_block_layers,
            "weights": _move_weights(placement),
        },
        "algorithm_params": dict(placement.algorithm_params),
    }


def _result_maturity(output: SimOutput, cfg: TuneConfig,
                     state: TuneState) -> str:
    """结果级别优先采用 GeneSim 的成熟度机制，不在调优侧另造一套。"""
    from tune.calibration import result_maturity

    reported = output.summary.get("result_maturity")
    if isinstance(reported, str) and reported.strip():
        return reported
    report = None
    if cfg.calibration.validation_report is not None:
        report = json.loads(Path(cfg.calibration.validation_report).read_text())
    backend = output.summary.get("simulation_backend", "pipeline")
    used_predictor = (
        cfg.placement.enabled
        and cfg.placement.evaluation_backend == "predictor")
    return result_maturity(
        report, profile=state.profile,
        profile_sources=state.profile.parameter_sources,
        backend=backend, used_predictor=used_predictor)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical(payload: Mapping[str, Any]) -> Any:
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _remove_empty_dir(path: Path) -> None:
    """失败后清掉空的内容键目录，避免留下看似完整的缓存位置。"""
    try:
        path.rmdir()
    except OSError:
        pass
