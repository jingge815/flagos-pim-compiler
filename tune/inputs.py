"""输入快照：模型、GeneSim 图骨架与请求轨迹。

内容指纹只由文件内容决定，路径和时间戳不进语义键。轨迹先保存成快照，
再按 `num_requests` 取观测窗口，续跑和复现读的是同一份。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from genesim_bridge.paths import genesim_root, llama2_7b_model_dir
from tune._genesim import trace_api
from tune.config import RequestBound, TuneConfig, TraceSnapshot, validate_workload_bounds


@dataclass(frozen=True)
class InputSnapshotView:
    """按磁盘当前内容复核后的输入视图。

    完成清单用它核对仿真实际处理的请求窗口，避免各阶段各自读盘。
    """

    trace_sha256: str
    num_requests: int
    processed_tokens: int
    max_seq: int


@dataclass(frozen=True)
class TuneInputs:
    """本轮调优的固定输入与内容身份。"""

    model_dir: Path
    ir_path: Path
    trace_path: Path
    trace: TraceSnapshot
    max_seq: int
    model_sha256: str
    ir_sha256: str
    trace_sha256: str
    input_dir: Path
    # 外置 IR 的原路径；发布时内容必须与 prepare 时读取的哈希一致。
    external_ir: Path | None = None
    ir_ready: bool = True

    def to_json(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "model_sha256": self.model_sha256,
            "ir": str(self.ir_path),
            "ir_sha256": self.ir_sha256,
            "trace": str(self.trace_path),
            "trace_sha256": self.trace_sha256,
            "num_requests": len(self.trace.requests),
            "max_seq": self.max_seq,
        }


def prepare_inputs(cfg: TuneConfig) -> TuneInputs:
    """只读解析输入并计算内容身份；不写输出目录。

    续跑与严格重放必须先比对这里的身份，确认不符时直接拒绝；发布动作
    只能放在所有身份核对之后，失败不能改写旧证据。
    """
    input_dir = cfg.output.dir / "inputs"
    model_dir = cfg.model.model_dir or llama2_7b_model_dir()
    trace_bytes, trace, trace_path = _trace_window_bytes(cfg, input_dir)
    max_seq = validate_workload_bounds(cfg.model, trace)
    external_ir = None
    ir_path = input_dir / "model.ir"
    ir_ready = True
    if cfg.model.ir is not None:
        external_ir = Path(cfg.model.ir)
        if not external_ir.is_file():
            raise FileNotFoundError(f"外置模型 IR 不存在：{external_ir}")
        ir_sha = file_sha256(external_ir)
    elif ir_path.is_file():
        ir_sha = file_sha256(ir_path)
    else:
        ir_sha = ""
        ir_ready = False
    return TuneInputs(
        model_dir=model_dir,
        ir_path=ir_path,
        trace_path=trace_path,
        trace=trace,
        max_seq=max_seq,
        model_sha256=model_fingerprint(model_dir),
        ir_sha256=ir_sha,
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        input_dir=input_dir,
        external_ir=external_ir,
        ir_ready=ir_ready,
    )


def publish_inputs(cfg: TuneConfig, inputs: TuneInputs) -> TuneInputs:
    """原子发布输入快照：先写暂存区复核全部身份，再提交整个输入集合。

    轨迹、外置 IR、生成 IR 与身份清单共用一个提交边界；任何失败都只
    清理暂存区，已发布快照保持原字节，不会被半份新内容覆盖。
    """
    input_dir = inputs.input_dir
    staging = input_dir.with_name(
        f"{input_dir.name}.staging-{uuid.uuid4().hex[:8]}")
    try:
        staging.mkdir(parents=True, exist_ok=True)
        trace_bytes = _trace_window_bytes(cfg, input_dir)[0]
        staged_trace = staging / "trace.json"
        staged_trace.write_bytes(trace_bytes)
        staged_ir = staging / "model.ir"
        if inputs.external_ir is not None:
            _atomic_copy(inputs.external_ir, staged_ir)
        elif not inputs.ir_ready:
            _generate_model_ir(inputs.model_dir, staged_ir)
            if not staged_ir.is_file():
                raise RuntimeError(f"model_parser 没有产出 {staged_ir}")
            # 新生成的 IR 没有预置身份：以实际内容为准重新登记。
            inputs = replace(inputs, ir_sha256=file_sha256(staged_ir),
                             ir_ready=True)
        else:
            _atomic_copy(inputs.ir_path, staged_ir)
        # 重新读取暂存字节，逐项核对 prepare 阶段身份后才允许提交。
        if file_sha256(staged_trace) != inputs.trace_sha256:
            raise RuntimeError("轨迹内容在身份核对后发生变化，拒绝发布输入快照")
        if not inputs.ir_sha256 or file_sha256(staged_ir) != inputs.ir_sha256:
            raise RuntimeError(
                "外置模型 IR 内容在身份核对后发生变化，拒绝发布输入快照"
                if inputs.external_ir is not None
                else "模型 IR 内容与 prepare 阶段身份不一致，拒绝发布输入快照")
        staged_identity = staging / "inputs.json"
        staged_identity.write_text(
            json.dumps(inputs.to_json(), indent=2, ensure_ascii=False))
        _commit_inputs(input_dir, staging)
        return inputs
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _commit_inputs(input_dir: Path, staging: Path) -> None:
    """提交整个输入集合：目录不存在时一次换名，已存在时只允许无差异。

    已有发布目录意味着旧证据；逐文件覆盖会在崩溃窗口里留下半新半旧
    的集合。因此内容完全一致时什么都不写，不一致时拒绝并保留旧快照。
    """
    if not input_dir.exists():
        os.replace(staging, input_dir)
        return
    for name in ("model.ir", "trace.json", "inputs.json"):
        target = input_dir / name
        source = staging / name
        if not target.is_file() or not source.is_file():
            raise RuntimeError(
                f"输出目录已有不完整的输入快照（缺 {name}），"
                "拒绝覆盖旧证据；请换一个 --output-dir")
        if file_sha256(target) != file_sha256(source):
            raise RuntimeError(
                f"输出目录已有不同的输入快照（{name}），"
                "拒绝覆盖旧证据；请换一个 --output-dir 或恢复原输入")


def verify_input_identity(inputs: TuneInputs, recorded: Mapping,
                          *, require_ir: bool = True) -> None:
    """重新生成的输入身份必须与记录逐项一致。

    `require_ir=False` 只用于“现场还没生成 IR”的扫描前置核对：此时模型
    内容已核对，IR 由同一 model_parser 从该模型派生；生成后必须再用
    `require_ir=True` 核对一次。
    """
    if not isinstance(recorded, Mapping):
        raise ValueError("记录里缺少输入身份，无法严格重放")
    expected = {
        "model_sha256": inputs.model_sha256,
        "ir_sha256": inputs.ir_sha256,
        "trace_sha256": inputs.trace_sha256,
        "max_seq": inputs.max_seq,
        "num_requests": len(inputs.trace.requests),
    }
    if not require_ir:
        expected.pop("ir_sha256")
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(
                f"输入身份与记录不一致：{key} 记录 {recorded.get(key)!r}，"
                f"本次 {value!r}")


def snapshot_inputs(cfg: TuneConfig) -> TuneInputs:
    """生成/读取模型、IR 与轨迹，写进 `<output>/inputs/` 并返回身份。"""
    return publish_inputs(cfg, prepare_inputs(cfg))


def verify_input_snapshot(cfg: TuneConfig,
                          inputs: TuneInputs) -> InputSnapshotView:
    """按磁盘当前字节复核输入快照，任一变化都失败。

    模型重算 config 与全部权重的指纹，IR 重读字节，轨迹重新解析并重算
    请求窗口、上下文上界与处理 token 数；编译发布、仿真配置生成、完成
    清单发布、快速缓存读取与两种重放共用这一入口，不能用新文件内容重绑
    旧键。返回的视图与本次复核出自同一批字节，供后续台账交叉核对。
    """
    errors, view = _snapshot_check(cfg, inputs)
    if errors:
        raise ValueError("输入快照在读取后发生变化：" + "；".join(errors))
    return view


def _snapshot_check(cfg: TuneConfig,
                    inputs: TuneInputs) -> tuple[list[str], InputSnapshotView]:
    errors: list[str] = []
    digest = ""
    try:
        current_model = model_fingerprint(inputs.model_dir)
    except (OSError, FileNotFoundError) as exc:
        errors.append(f"模型输入无法重新读取：{exc}")
    else:
        if current_model != inputs.model_sha256:
            errors.append(
                f"模型的当前内容指纹 {current_model!r} 与 prepare 阶段 "
                f"{inputs.model_sha256!r} 不一致")
    if inputs.ir_ready:
        ir_path = _current_ir_path(inputs)
        if ir_path is None:
            errors.append("找不到当前输入 IR：发布快照与外置来源都不存在")
        else:
            try:
                current_ir = file_sha256(ir_path)
            except OSError as exc:
                errors.append(f"输入 IR 无法重新读取：{exc}")
            else:
                if current_ir != inputs.ir_sha256:
                    label = ("输入 IR 发布快照" if Path(ir_path) == inputs.ir_path
                             else "外置输入 IR")
                    errors.append(
                        f"{label}的当前内容摘要 {current_ir!r} 与 prepare "
                        f"阶段 {inputs.ir_sha256!r} 不一致")
    try:
        data, requests, bounds, processed = _current_trace_window(cfg, inputs)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"请求轨迹无法重新读取：{exc}")
    else:
        digest = hashlib.sha256(data).hexdigest()
        if digest != inputs.trace_sha256:
            errors.append(
                f"请求轨迹的当前内容摘要 {digest!r} 与 prepare 阶段 "
                f"{inputs.trace_sha256!r} 不一致")
        if bounds != inputs.trace.requests:
            errors.append("请求轨迹的当前请求窗口与 prepare 阶段不一致")
        snapshot = TraceSnapshot(source=str(inputs.trace_path), requests=bounds)
        try:
            max_seq = validate_workload_bounds(cfg.model, snapshot)
        except ValueError as exc:
            errors.append(f"请求轨迹的当前窗口不满足配置边界：{exc}")
        else:
            if max_seq != inputs.max_seq:
                errors.append(
                    f"请求轨迹的当前 max_seq {max_seq} 与 prepare 阶段 "
                    f"{inputs.max_seq} 不一致")
    if errors:
        return errors, InputSnapshotView(
            trace_sha256=digest,
            num_requests=len(inputs.trace.requests),
            processed_tokens=0,
            max_seq=inputs.max_seq,
        )
    return [], InputSnapshotView(
        trace_sha256=digest,
        num_requests=len(requests),
        processed_tokens=processed,
        max_seq=inputs.max_seq,
    )


def _current_ir_path(inputs: TuneInputs) -> Path | None:
    """已发布快照优先；尚未发布时退回外置 IR 原文件。"""
    if inputs.ir_path.is_file():
        return inputs.ir_path
    if inputs.external_ir is not None and Path(inputs.external_ir).is_file():
        return Path(inputs.external_ir)
    return None


def _current_trace_window(cfg: TuneConfig, inputs: TuneInputs):
    """按当前磁盘内容重算轨迹窗口的规范字节、请求边界与处理 token 数。

    已发布快照是仿真实际消费的文件；尚未发布时按 prepare 的同一来源重新
    取窗口。处理 token 数与 GeneSim 的 prompt/generated 累计口径一致：
    prefill 计 seq_len 与输出，decode 只计输出。
    """
    Trace, _ = trace_api()
    if inputs.trace_path.is_file():
        trace = Trace.load_json(inputs.trace_path)
        requests = list(trace.requests)[: cfg.model.trace.num_requests]
        if len(requests) < cfg.model.trace.num_requests:
            raise ValueError(
                f"轨迹只有 {len(requests)} 个请求，少于配置要求的 "
                f"num_requests={cfg.model.trace.num_requests}")
        window = Trace(requests=requests, description=trace.description)
        data = json.dumps(
            window.to_dict(), indent=2, ensure_ascii=False).encode()
    else:
        # 尚未发布：按 prepare 的同一来源与同一序列化口径重取窗口。
        data, _, _ = _trace_window_bytes(cfg, inputs.input_dir)
        requests = list(Trace.from_dict(json.loads(data)).requests)
    bounds = tuple(_request_bound(request) for request in requests)
    return data, requests, bounds, _processed_tokens(requests)


def _processed_tokens(requests) -> int:
    """与 GeneSim 调度器相同的 token 累计：prefill 含 seq_len，decode 只含输出。"""
    total = 0
    for request in requests:
        total += int(request.output_length)
        if request.is_prefill:
            total += int(request.seq_len)
    return total


def _trace_window_bytes(
    cfg: TuneConfig, input_dir: Path,
) -> tuple[bytes, TraceSnapshot, Path]:
    """只读生成轨迹窗口的规范字节与边界快照。"""
    Trace, _ = trace_api()
    trace, requests, _ = _read_trace_window(cfg)
    window = Trace(requests=requests, description=trace.description)
    # 与 GeneSim `Trace.save_json` 相同的序列化口径：内存字节即发布文件。
    data = json.dumps(
        window.to_dict(), indent=2, ensure_ascii=False).encode()
    trace_path = input_dir / "trace.json"
    bounds = tuple(_request_bound(request) for request in requests)
    snapshot = TraceSnapshot(source=str(trace_path), requests=bounds)
    return data, snapshot, trace_path


def _atomic_write(path: Path, data: bytes) -> None:
    """同目录临时文件加原子替换，避免半份证据留在输出目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    _atomic_write(destination, Path(source).read_bytes())


# 模型内容摘要缓存：键为每个文件的设备、inode、大小、修改与变更时间。
# 任何写入都会更新 ctime，命中即代表自上次摘要以来文件未被改写。
_MODEL_FINGERPRINT_CACHE: dict[str, tuple[tuple, str]] = {}


def model_fingerprint(model_dir: Path) -> str:
    """模型内容摘要：config 与全部权重文件按字节计算。

    不加载权重，也不看路径/时间戳：同一份权重换个目录不应触发重编。
    未改写的模型复用上次内容摘要；写入会改变 ctime，从而强制重算字节。
    """
    model_dir = Path(model_dir)
    files = [model_dir / "config.json"]
    files.extend(sorted(model_dir.glob("*.safetensors")))
    files.extend(sorted(model_dir.glob("pytorch_model*.bin")))
    existing = [path for path in files if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"模型目录里没有 config.json 或权重文件：{model_dir}")
    signature = []
    for path in existing:
        stat = path.stat()
        signature.append((path.name, stat.st_dev, stat.st_ino, stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns))
    key = str(model_dir.resolve())
    cached = _MODEL_FINGERPRINT_CACHE.get(key)
    if cached is not None and cached[0] == tuple(signature):
        return cached[1]
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.name.encode())
        # 分块读，避免把单个权重分片整份载入内存。
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    value = digest.hexdigest()
    _MODEL_FINGERPRINT_CACHE[key] = (tuple(signature), value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_model_ir(model_dir: Path, out_path: Path) -> None:
    """调用 GeneSim 的模型解析器生成图骨架，编号与仿真加载的那份同源。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "scripts/model_parser.py",
         "--model_name", str(model_dir), "--output", str(out_path)],
        cwd=str(genesim_root()), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"生成模型 IR 失败（exit {proc.returncode}）：\n{proc.stderr[-2000:]}")
    if not out_path.is_file():
        raise RuntimeError(f"model_parser 没有产出 {out_path}")


def validate_dry_inputs(cfg: TuneConfig) -> TraceSnapshot:
    """dry-run 的输入检查：只读轨迹与请求边界，不生成 IR、不加载权重。"""
    _, _, snapshot = _read_trace_window(cfg)
    validate_workload_bounds(cfg.model, snapshot)
    return snapshot


def _read_trace_window(cfg: TuneConfig) -> tuple:
    Trace, generate_synthetic_trace = trace_api()
    trace_cfg = cfg.model.trace
    if trace_cfg.path is not None:
        trace = Trace.load_json(trace_cfg.path)
    else:
        synthetic = trace_cfg.synthetic
        trace = generate_synthetic_trace(
            synthetic.num_requests, seed=synthetic.seed)
    requests = list(trace.requests)[: trace_cfg.num_requests]
    if not requests:
        raise ValueError("轨迹里没有可观测的请求")
    # 请求数在输入前置契约里核对：dry-run 与普通入口同一判据，不能等
    # 候选编译完才在产物校验阶段失败。
    if len(requests) < trace_cfg.num_requests:
        raise ValueError(
            f"轨迹只有 {len(requests)} 个请求，少于配置要求的 "
            f"num_requests={trace_cfg.num_requests}")
    bounds = tuple(_request_bound(request) for request in requests)
    return trace, requests, TraceSnapshot(source="memory", requests=bounds)


def _request_bound(request) -> RequestBound:
    """把一个 GeneSim 请求折算成上下文上界。"""
    tokens = len(request.input_tokens)
    output = int(request.output_length)
    initial = request.initial_kv_length
    initial = 0 if initial is None else int(initial)
    if request.is_prefill:
        return RequestBound(prompt_tokens=tokens, output_tokens=output,
                            initial_kv=initial)
    # decode 请求的 prefill 已经发生，本次前向只处理 1 个 token。
    return RequestBound(prompt_tokens=0, output_tokens=output,
                        initial_kv=max(initial, request.seq_len, tokens))
