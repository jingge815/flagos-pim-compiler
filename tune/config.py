"""统一调优配置：一份版本化 YAML 描述整轮调优。

所有路径相对 YAML 所在目录解析；CLI 高频覆盖先合并进同一份生效配置再落盘，
不形成第二份隐式状态。未知字段、非法取值、非法算法参数在任何编译前报错。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from contracts.hardware_profile import PROFILE_FIELDS, HardwareProfile, load_profile
from contracts.op_contract import OPTIMIZATION_FIELDS, PimOptimizations
from tune._genesim import build_algorithm

# 八类放置移动，与 GeneSim `src/autotune/placement.py` 的提案类型逐字对应。
MOVES = frozenset({
    "op_relocate",
    "op_swap",
    "connected_segment_relocate",
    "attention_branch_relocate",
    "kv_group_pack_or_spread",
    "layer_block_relocate",
    "layer_boundary_shift",
    "pim_rebalance_or_swap",
})
SEARCH_ALGORITHMS = frozenset({"enumerate", "simulated_annealing",
                               "genetic_algorithm", "tabu_search"})
PLACEMENT_ALGORITHMS = SEARCH_ALGORITHMS - {"enumerate"}
BACKENDS = frozenset({"pipeline", "predictor"})
OBJECTIVE_METRICS = frozenset({"total_time_s", "system_efficiency_utility"})
CACHE_MODES = frozenset({"fast", "strict"})
CORE_POLICIES = frozenset({"fixed_dpus", "use_all_clusters"})
# Pareto 资源轴的唯一枚举：配置解析、扫描与报告共用，非法值在 load_config 拒绝。
PARETO_RESOURCES = frozenset({
    "total_physical_memory_bytes", "num_physical_cores"})
LAYER_NAMES = ("strategy", "op", "hardware")
# 硬件层可扫描字段：需求 R2/R8 明确允许的物理量。画像名称、版本、基配置路径
# 和参数来源台账不是独立语义字段，扫描它们会伪造证据或改坏身份，一律不开放；
# 精度 bytes_per_element 必须与模型 dtype 一致，也不能单独扫描。
HARDWARE_FIELDS = frozenset({
    "num_pim_devices", "clusters_per_device", "tensors_per_cluster",
    "cluster_memory_bytes", "sys_reserve_bytes",
    "wram_bytes_per_dpu", "dma_align",
    "clock_period_ns", "multipliers", "cycle_costs", "zone_cycle_factors",
    "tile_size", "mac_width", "chunk_size", "max_bulk_transfer_bytes",
    "use_long_loop_counts", "inter_cluster_communication",
    "host_relay_overhead_ns", "links", "cpu_model",
})
_TOP_KEYS = frozenset({
    "version", "run_name", "model", "profile", "optimization", "space",
    "placement", "search", "cache", "simulation", "calibration", "sweep",
    "output",
})


@dataclass(frozen=True)
class _TopLevel:
    """顶层键集合的校验载体；不参与运行状态。"""

    version: int
    run_name: str
    model: Any
    profile: Any
    optimization: Any
    space: Any
    placement: Any
    search: Any
    cache: Any
    simulation: Any
    calibration: Any
    sweep: Any
    output: Any


@dataclass(frozen=True)
class SyntheticTraceConfig:
    seed: int
    num_requests: int


@dataclass(frozen=True)
class TraceConfig:
    path: Path | None
    num_requests: int
    synthetic: SyntheticTraceConfig | None


@dataclass(frozen=True)
class ModelConfig:
    model_dir: Path | None
    ir: Path | None
    prefill_seq_len: int
    max_seq: int | None
    trace: TraceConfig


@dataclass(frozen=True)
class StrategySpace:
    enabled: bool
    num_dpus: tuple[int, ...]
    num_stages: tuple[int, ...]


@dataclass(frozen=True)
class OpSpace:
    enabled: bool
    num_tasklets: tuple[int, ...]


@dataclass(frozen=True)
class OptimizationSpace:
    """优化开关层：只扫描一个开关，其余开关取顶层 optimization。"""

    enabled: bool
    field: str | None
    values: tuple[bool, ...]


@dataclass(frozen=True)
class HardwareSpace:
    enabled: bool
    fields: Mapping[str, tuple[Any, ...]]


@dataclass(frozen=True)
class SpaceConfig:
    strategy: StrategySpace
    op: OpSpace
    hardware: HardwareSpace
    optimization: OptimizationSpace


@dataclass(frozen=True)
class PlacementMoves:
    max_segment_ops: int
    max_layer_block_layers: int
    weights: Mapping[str, float]


@dataclass(frozen=True)
class PlacementConfig:
    enabled: bool
    algorithm: str
    evaluation_backend: str
    validation_backend: str
    objective_metric: str
    objective_weights: Mapping[str, float]
    max_evaluations: int
    max_time_s: float
    max_proposals: int | None
    seed: int
    moves: PlacementMoves
    algorithm_params: Mapping[str, Any]


@dataclass(frozen=True)
class SearchConfig:
    algorithm: str
    seed: int
    max_evaluations: int
    max_time_s: float
    max_proposals: int | None
    layer_weights: Mapping[str, float]
    algorithm_params: Mapping[str, Any]


@dataclass(frozen=True)
class CacheConfig:
    mode: str


@dataclass(frozen=True)
class SimulationConfig:
    use_cost_sidecar: bool


@dataclass(frozen=True)
class CalibrationConfig:
    validation_report: Path | None


@dataclass(frozen=True)
class ParetoConfig:
    resource: str
    direction: str


@dataclass(frozen=True)
class SweepConfig:
    fields: Mapping[str, tuple[Any, ...]]
    core_policy: str
    knee_improvement_threshold: float
    pareto: ParetoConfig


@dataclass(frozen=True)
class OutputConfig:
    dir: Path


@dataclass(frozen=True)
class TuneConfig:
    version: int
    run_name: str
    model: ModelConfig
    profile_path: Path
    profile: HardwareProfile
    optimization: PimOptimizations
    space: SpaceConfig
    placement: PlacementConfig
    search: SearchConfig
    cache: CacheConfig
    simulation: SimulationConfig
    calibration: CalibrationConfig
    sweep: SweepConfig
    output: OutputConfig
    config_dir: Path

    def to_json(self) -> dict[str, Any]:
        """生效配置的 JSON 形态；路径一律为绝对路径。"""
        return {
            "version": self.version,
            "run_name": self.run_name,
            "model": {
                "model_dir": _path_text(self.model.model_dir),
                "ir": _path_text(self.model.ir),
                "prefill_seq_len": self.model.prefill_seq_len,
                "max_seq": self.model.max_seq,
                "trace": {
                    "path": _path_text(self.model.trace.path),
                    "num_requests": self.model.trace.num_requests,
                    "synthetic": (
                        None if self.model.trace.synthetic is None else {
                            "seed": self.model.trace.synthetic.seed,
                            "num_requests": self.model.trace.synthetic.num_requests,
                        }
                    ),
                },
            },
            "profile": {"path": str(self.profile_path)},
            "optimization": self.optimization.to_payload(),
            "space": {
                "strategy": {
                    "enabled": self.space.strategy.enabled,
                    "num_dpus": list(self.space.strategy.num_dpus),
                    "num_stages": list(self.space.strategy.num_stages),
                },
                "op": {
                    "enabled": self.space.op.enabled,
                    "num_tasklets": list(self.space.op.num_tasklets),
                },
                "hardware": {
                    "enabled": self.space.hardware.enabled,
                    "fields": {key: list(value)
                               for key, value in self.space.hardware.fields.items()},
                },
                **({} if self.space.optimization.field is None else {
                    "optimization": {
                        "enabled": self.space.optimization.enabled,
                        "field": self.space.optimization.field,
                        "values": list(self.space.optimization.values),
                    },
                }),
            },
            "placement": {
                "enabled": self.placement.enabled,
                "algorithm": self.placement.algorithm,
                "evaluation_backend": self.placement.evaluation_backend,
                "validation_backend": self.placement.validation_backend,
                "objective_metric": self.placement.objective_metric,
                "objective_weights": dict(self.placement.objective_weights),
                "max_evaluations": self.placement.max_evaluations,
                "max_time_s": self.placement.max_time_s,
                "max_proposals": self.placement.max_proposals,
                "seed": self.placement.seed,
                "moves": {
                    "max_segment_ops": self.placement.moves.max_segment_ops,
                    "max_layer_block_layers": self.placement.moves.max_layer_block_layers,
                    "weights": dict(self.placement.moves.weights),
                },
                "algorithm_params": dict(self.placement.algorithm_params),
            },
            "search": {
                "algorithm": self.search.algorithm,
                "seed": self.search.seed,
                "max_evaluations": self.search.max_evaluations,
                "max_time_s": self.search.max_time_s,
                "max_proposals": self.search.max_proposals,
                "layer_weights": dict(self.search.layer_weights),
                "algorithm_params": dict(self.search.algorithm_params),
            },
            "cache": {"mode": self.cache.mode},
            "simulation": {"use_cost_sidecar": self.simulation.use_cost_sidecar},
            "calibration": {
                "validation_report": _path_text(self.calibration.validation_report),
            },
            "sweep": {
                "fields": {key: list(value) for key, value in self.sweep.fields.items()},
                "core_policy": self.sweep.core_policy,
                "knee_improvement_threshold": self.sweep.knee_improvement_threshold,
                "pareto": {
                    "resource": self.sweep.pareto.resource,
                    "direction": self.sweep.pareto.direction,
                },
            },
            "output": {"dir": str(self.output.dir)},
        }


@dataclass(frozen=True)
class RequestBound:
    """一个请求的上下文上界：已有 KV、本次提示与要生成的 token 数。"""

    prompt_tokens: int
    output_tokens: int
    initial_kv: int = 0

    @property
    def context(self) -> int:
        return self.initial_kv + self.prompt_tokens + self.output_tokens


@dataclass(frozen=True)
class TraceSnapshot:
    source: str
    requests: tuple[RequestBound, ...]

    @property
    def max_prompt(self) -> int:
        return max((request.prompt_tokens for request in self.requests), default=0)

    @property
    def max_context(self) -> int:
        return max((request.context for request in self.requests), default=0)


def load_config(path: Path | str, overrides: Mapping[str, Any] | None = None) -> TuneConfig:
    """读取并校验一份调优配置；CLI 覆盖在解析前合并。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到调优配置：{path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"调优配置顶层必须是 YAML 映射：{path}")
    raw = _deep_merge(dict(raw), dict(overrides or {}))
    base = path.parent

    _check_keys(raw, _TopLevel, "顶层")
    # optimization 是可选的：旧配置没有该段时保持原有全开行为。
    missing = _TOP_KEYS - set(raw) - {"optimization"}
    if missing:
        raise ValueError(f"调优配置缺少字段：{sorted(missing)}")
    version = _int(raw["version"], "version")
    if version != 1:
        raise ValueError(f"不支持的配置版本 {version}")
    run_name = _str(raw["run_name"], "run_name")

    model = _parse_model(raw["model"], base)
    profile_path = _parse_profile_path(raw.get("profile"), base)
    profile = load_profile(profile_path)
    optimization = _parse_optimization(raw.get("optimization"))
    space = _parse_space(raw["space"], profile)
    placement = _parse_placement(raw["placement"])
    if profile.ceva is not None and placement.enabled:
        raise ValueError(
            "CEVA 画像不启用内层放置搜索：编译器一对一映射，"
            "autotune.enabled 必须为 false")
    search = _parse_search(raw["search"])
    cache = _parse_cache(raw["cache"])
    simulation = _parse_simulation(raw["simulation"])
    calibration = _parse_calibration(raw["calibration"], base)
    sweep = _parse_sweep(raw["sweep"])
    output = _parse_output(raw["output"], base)
    _validate_predictor_contract(profile, profile_path, placement, space, sweep)

    return TuneConfig(
        version=version, run_name=run_name, model=model,
        profile_path=profile_path, profile=profile,
        optimization=optimization, space=space,
        placement=placement, search=search, cache=cache,
        simulation=simulation, calibration=calibration, sweep=sweep,
        output=output, config_dir=base,
    )


def write_effective_config(cfg: TuneConfig) -> Path:
    """把合并后的生效配置落盘，作为本轮唯一状态。"""
    cfg.output.dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output.dir / "effective_config.yaml"
    path.write_text(yaml.safe_dump(cfg.to_json(), sort_keys=False, allow_unicode=True))
    return path


def run_identity(cfg: TuneConfig, inputs: Any | None = None) -> str:
    """一次运行的语义身份：生效配置、输入内容、源码/工具/基配置与环境。

    外层预算（候选数、墙钟、提案数）不进入身份，允许只加预算的合法续跑；
    输出目录、缓存档位、仿真开关、语义输入和运行环境都在身份内，改变它们
    不能复用旧响应。
    """
    from tune.steps import run_semantic_identity

    data = cfg.to_json()
    data.pop("output", None)
    for key in ("max_evaluations", "max_time_s", "max_proposals"):
        data["search"].pop(key, None)
    payload = {
        "config": data,
        "semantics": run_semantic_identity(cfg),
        "inputs": None if inputs is None else {
            "model_sha256": inputs.model_sha256,
            "ir_sha256": inputs.ir_sha256,
            "trace_sha256": inputs.trace_sha256,
            "max_seq": inputs.max_seq,
            "num_requests": len(inputs.trace.requests),
        },
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_workload_bounds(model: ModelConfig, trace: TraceSnapshot) -> int:
    """校验 prefill 覆盖提示长度，返回解析后的 max_seq。

    `max_seq=null` 由所选请求的完整上下文上界推导；显式值不足即拒绝，
    不能让仿真的 KV 区比请求实际需要的短。
    """
    if not trace.requests:
        raise ValueError("轨迹里没有请求")
    if model.prefill_seq_len < trace.max_prompt:
        raise ValueError(
            f"prefill_seq_len={model.prefill_seq_len} 覆盖不了所选请求的"
            f"最长提示 {trace.max_prompt}")
    resolved = trace.max_context if model.max_seq is None else model.max_seq
    if resolved < trace.max_context:
        raise ValueError(
            f"max_seq={resolved} 小于所选请求的上下文上界 {trace.max_context}")
    return resolved


def cli_overrides(args: Any) -> dict[str, Any]:
    """把命令行高频覆盖项翻译成配置覆盖；未给的项不出现。"""
    overrides: dict[str, Any] = {}
    search: dict[str, Any] = {}
    if getattr(args, "max_evaluations", None) is not None:
        search["max_evaluations"] = args.max_evaluations
    if getattr(args, "max_time_s", None) is not None:
        search["max_time_s"] = args.max_time_s
    if getattr(args, "algorithm", None) is not None:
        search["algorithm"] = args.algorithm
    if search:
        overrides["search"] = search
    if getattr(args, "profile", None) is not None:
        # 命令行给的路径按当前目录解析；YAML 内的相对路径才按 YAML 所在目录。
        overrides["profile"] = {"path": str(Path(args.profile).resolve())}
    if getattr(args, "cache_mode", None) is not None:
        overrides["cache"] = {"mode": args.cache_mode}
    if getattr(args, "output_dir", None) is not None:
        overrides["output"] = {"dir": str(Path(args.output_dir).resolve())}
    return overrides


def _parse_model(raw: Any, base: Path) -> ModelConfig:
    _require_mapping(raw, "model")
    _check_keys(raw, ModelConfig, "model")
    model_dir = _optional_path(raw.get("model_dir"), base, "model.model_dir")
    ir = _optional_path(raw.get("ir"), base, "model.ir")
    if model_dir is None and ir is None:
        # 两个都留空表示走 paths.json 的默认模型；显式写 null 也一样。
        from genesim_bridge.paths import llama2_7b_model_dir

        model_dir = llama2_7b_model_dir()
    if model_dir is not None and not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    if ir is not None and not ir.is_file():
        raise FileNotFoundError(f"模型 IR 不存在：{ir}")
    prefill = _int(raw.get("prefill_seq_len"), "model.prefill_seq_len")
    max_seq = raw.get("max_seq")
    if max_seq is not None:
        max_seq = _int(max_seq, "model.max_seq")
    trace_raw = raw.get("trace")
    _require_mapping(trace_raw, "model.trace")
    _check_keys(trace_raw, TraceConfig, "model.trace")
    trace_path = _optional_path(trace_raw.get("path"), base, "model.trace.path")
    synthetic_raw = trace_raw.get("synthetic")
    synthetic = None
    if synthetic_raw is not None:
        _require_mapping(synthetic_raw, "model.trace.synthetic")
        _check_keys(synthetic_raw, SyntheticTraceConfig, "model.trace.synthetic")
        synthetic = SyntheticTraceConfig(
            seed=_int(synthetic_raw.get("seed"), "model.trace.synthetic.seed",
                      minimum=0),
            num_requests=_int(synthetic_raw.get("num_requests"),
                              "model.trace.synthetic.num_requests"),
        )
    if (trace_path is None) == (synthetic is None):
        raise ValueError("model.trace.path 与 model.trace.synthetic 必须二选一")
    if trace_path is not None and not trace_path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在：{trace_path}")
    num_requests = _int(trace_raw.get("num_requests"), "model.trace.num_requests")
    return ModelConfig(
        model_dir=model_dir, ir=ir, prefill_seq_len=prefill, max_seq=max_seq,
        trace=TraceConfig(path=trace_path, num_requests=num_requests,
                          synthetic=synthetic),
    )


def _parse_profile_path(raw: Any, base: Path) -> Path:
    if raw is None:
        raise ValueError("缺少 profile.path")
    if isinstance(raw, str):
        raw = {"path": raw}
    _require_mapping(raw, "profile")
    _check_keys(raw, _ProfileSection, "profile")
    return _required_path(raw.get("path"), base, "profile.path")


@dataclass(frozen=True)
class _ProfileSection:
    path: str


def _parse_optimization(raw: Any) -> PimOptimizations:
    """解析编译优化开关；缺省全开，未知开关在编译前拒绝。"""
    if raw is None:
        return PimOptimizations()
    _require_mapping(raw, "optimization")
    _check_keys(raw, PimOptimizations, "optimization")
    return PimOptimizations(
        constant_fold=_bool(raw.get("constant_fold", True),
                            "optimization.constant_fold"),
        cse=_bool(raw.get("cse", True), "optimization.cse"),
        dce=_bool(raw.get("dce", True), "optimization.dce"),
        loop_tiling=_bool(raw.get("loop_tiling", True),
                          "optimization.loop_tiling"),
    )


def _parse_space(raw: Any, profile: HardwareProfile) -> SpaceConfig:
    _require_mapping(raw, "space")
    _check_keys(raw, SpaceConfig, "space")
    strategy_raw = raw.get("strategy")
    _require_mapping(strategy_raw, "space.strategy")
    _check_keys(strategy_raw, StrategySpace, "space.strategy")
    strategy = StrategySpace(
        enabled=_bool(strategy_raw.get("enabled"), "space.strategy.enabled"),
        num_dpus=_int_tuple(strategy_raw.get("num_dpus"), "space.strategy.num_dpus"),
        num_stages=_int_tuple(strategy_raw.get("num_stages"),
                              "space.strategy.num_stages"),
    )
    _closed_layer(strategy.enabled, (strategy.num_dpus, strategy.num_stages),
                  "strategy")
    # 物理核数要按完整候选（含硬件层展开）复核，不能用基线画像提前误拒
    # 更大的合法变体；候选空间与 dry-run 会按各变体画像记录拒绝原因。
    for num_dpus in strategy.num_dpus:
        for num_stages in strategy.num_stages:
            if num_dpus % num_stages:
                raise ValueError(
                    f"num_stages={num_stages} 不能整除 num_dpus={num_dpus}")

    op_raw = raw.get("op")
    _require_mapping(op_raw, "space.op")
    _check_keys(op_raw, OpSpace, "space.op")
    op = OpSpace(
        enabled=_bool(op_raw.get("enabled"), "space.op.enabled"),
        num_tasklets=_int_tuple(op_raw.get("num_tasklets"), "space.op.num_tasklets"),
    )
    _closed_layer(op.enabled, (op.num_tasklets,), "op")

    hardware_raw = raw.get("hardware")
    _require_mapping(hardware_raw, "space.hardware")
    _check_keys(hardware_raw, HardwareSpace, "space.hardware")
    fields_raw = hardware_raw.get("fields")
    _require_mapping(fields_raw, "space.hardware.fields")
    unknown = set(fields_raw) - HARDWARE_FIELDS
    if unknown:
        raise ValueError(f"space.hardware.fields 有未知字段：{sorted(unknown)}")
    hardware_fields = {}
    for key, values in fields_raw.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"space.hardware.fields.{key} 必须是非空取值列表")
        hardware_fields[key] = tuple(values)
    hardware = HardwareSpace(
        enabled=_bool(hardware_raw.get("enabled"), "space.hardware.enabled"),
        fields=hardware_fields,
    )
    if hardware.enabled and not hardware.fields:
        raise ValueError("space.hardware.enabled=true 时 fields 不能为空")
    # 关闭的硬件层同样表示冻结：每个字段只能有一个取值，否则候选会静默展开。
    _closed_layer(hardware.enabled, tuple(hardware.fields.values()), "hardware")
    _validate_hardware_fields(hardware.fields, profile)

    optimization = _parse_optimization_space(raw.get("optimization"))
    return SpaceConfig(strategy=strategy, op=op, hardware=hardware,
                       optimization=optimization)


def _parse_optimization_space(raw: Any) -> OptimizationSpace:
    """优化开关层可省略；省略时冻结在顶层 optimization 上。"""
    if raw is None:
        return OptimizationSpace(False, None, ())
    _require_mapping(raw, "space.optimization")
    _check_keys(raw, OptimizationSpace, "space.optimization")
    field = _str(raw.get("field"), "space.optimization.field")
    if field not in OPTIMIZATION_FIELDS:
        raise ValueError(
            f"space.optimization.field={field!r} 不在 {list(OPTIMIZATION_FIELDS)}")
    values_raw = raw.get("values")
    if not isinstance(values_raw, (list, tuple)) or not values_raw:
        raise ValueError(
            f"space.optimization.values 必须是非空布尔列表，收到 {values_raw!r}")
    values = tuple(
        _bool(value, "space.optimization.values") for value in values_raw)
    enabled = _bool(raw.get("enabled"), "space.optimization.enabled")
    _closed_layer(enabled, (values,), "optimization")
    return OptimizationSpace(enabled, field, values)


def _validate_hardware_fields(fields: Mapping[str, tuple], profile: HardwareProfile) -> None:
    """硬件层字段值必须能派生出一份合法画像。"""
    if not fields:
        return
    for key, values in fields.items():
        for value in values:
            try:
                profile.with_fields(**{key: value})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"space.hardware.fields.{key}={value!r} 派生不出合法画像：{exc}"
                ) from exc


def _parse_placement(raw: Any) -> PlacementConfig:
    _require_mapping(raw, "placement")
    _check_keys(raw, PlacementConfig, "placement")
    algorithm = _str(raw.get("algorithm"), "placement.algorithm")
    if algorithm not in PLACEMENT_ALGORITHMS:
        raise ValueError(
            f"placement.algorithm={algorithm!r} 不在 {sorted(PLACEMENT_ALGORITHMS)}")
    evaluation = _str(raw.get("evaluation_backend"), "placement.evaluation_backend")
    validation = _str(raw.get("validation_backend"), "placement.validation_backend")
    for label, value in (("evaluation_backend", evaluation),
                         ("validation_backend", validation)):
        if value not in BACKENDS:
            raise ValueError(f"placement.{label}={value!r} 不在 {sorted(BACKENDS)}")
    metric = _str(raw.get("objective_metric"), "placement.objective_metric")
    if metric not in OBJECTIVE_METRICS:
        raise ValueError(
            f"placement.objective_metric={metric!r} 不在 {sorted(OBJECTIVE_METRICS)}")
    if validation != "pipeline":
        raise ValueError("placement.validation_backend 必须是 pipeline：最终排序"
                         "和结论只能用 pipeline 重放")
    if metric == "system_efficiency_utility" and evaluation != "pipeline":
        raise ValueError(
            "system_efficiency_utility 只允许 pipeline/pipeline，predictor 筛选"
            "使用 total_time_s")
    weights = _weights(raw.get("objective_weights"), "placement.objective_weights")
    if set(weights) != {"compute", "communication", "memory"}:
        raise ValueError(
            "placement.objective_weights 必须定义 compute / communication / memory")
    seed = _int(raw.get("seed"), "placement.seed", minimum=0)
    params = _mapping(raw.get("algorithm_params"), "placement.algorithm_params")
    enabled = _bool(raw.get("enabled"), "placement.enabled")
    build_algorithm(algorithm, dict(params), seed)
    moves_raw = raw.get("moves")
    _require_mapping(moves_raw, "placement.moves")
    _check_keys(moves_raw, PlacementMoves, "placement.moves")
    move_weights = _mapping(moves_raw.get("weights"), "placement.moves.weights")
    unknown = set(move_weights) - MOVES
    if unknown:
        raise ValueError(f"未知的放置移动：{sorted(unknown)}")
    for name, value in move_weights.items():
        _number(value, f"placement.moves.weights.{name}", minimum=0.0)
    if move_weights and sum(move_weights.values()) <= 0:
        # GeneSim 要到搜索空间构造时才拒绝全零权重；统一入口必须编译前失败。
        raise ValueError("placement.moves.weights 至少要有一类移动权重为正")
    return PlacementConfig(
        enabled=enabled,
        algorithm=algorithm,
        evaluation_backend=evaluation,
        validation_backend=validation,
        objective_metric=metric,
        objective_weights=weights,
        max_evaluations=_int(raw.get("max_evaluations"),
                             "placement.max_evaluations"),
        max_time_s=_number(raw.get("max_time_s"), "placement.max_time_s"),
        max_proposals=_optional_positive_int(raw.get("max_proposals"),
                                             "placement.max_proposals"),
        seed=seed,
        moves=PlacementMoves(
            max_segment_ops=_int(moves_raw.get("max_segment_ops"),
                                 "placement.moves.max_segment_ops"),
            max_layer_block_layers=_int(moves_raw.get("max_layer_block_layers"),
                                        "placement.moves.max_layer_block_layers"),
            weights=move_weights,
        ),
        algorithm_params=params,
    )


def _parse_search(raw: Any) -> SearchConfig:
    _require_mapping(raw, "search")
    _check_keys(raw, SearchConfig, "search")
    algorithm = _str(raw.get("algorithm"), "search.algorithm")
    if algorithm not in SEARCH_ALGORITHMS:
        raise ValueError(
            f"search.algorithm={algorithm!r} 不在 {sorted(SEARCH_ALGORITHMS)}")
    seed = _int(raw.get("seed"), "search.seed", minimum=0)
    params = _mapping(raw.get("algorithm_params"), "search.algorithm_params")
    build_algorithm(algorithm, dict(params), seed)
    weights = _weights(raw.get("layer_weights"), "search.layer_weights")
    if not set(LAYER_NAMES) <= set(weights):
        raise ValueError(f"search.layer_weights 必须定义 {list(LAYER_NAMES)}")
    unknown = set(weights) - set(LAYER_NAMES) - {"optimization"}
    if unknown:
        raise ValueError(f"search.layer_weights 有未知层：{sorted(unknown)}")
    if sum(weights.values()) <= 0:
        raise ValueError("search.layer_weights 至少要有一层权重为正")
    return SearchConfig(
        algorithm=algorithm,
        seed=seed,
        max_evaluations=_int(raw.get("max_evaluations"), "search.max_evaluations"),
        max_time_s=_number(raw.get("max_time_s"), "search.max_time_s"),
        max_proposals=_optional_positive_int(raw.get("max_proposals"),
                                             "search.max_proposals"),
        layer_weights=weights,
        algorithm_params=params,
    )


def _parse_cache(raw: Any) -> CacheConfig:
    _require_mapping(raw, "cache")
    _check_keys(raw, CacheConfig, "cache")
    mode = _str(raw.get("mode"), "cache.mode")
    if mode not in CACHE_MODES:
        raise ValueError(f"cache.mode={mode!r} 不在 {sorted(CACHE_MODES)}")
    return CacheConfig(mode=mode)


def _parse_simulation(raw: Any) -> SimulationConfig:
    _require_mapping(raw, "simulation")
    _check_keys(raw, SimulationConfig, "simulation")
    return SimulationConfig(
        use_cost_sidecar=_bool(raw.get("use_cost_sidecar"),
                               "simulation.use_cost_sidecar"))


def _parse_calibration(raw: Any, base: Path) -> CalibrationConfig:
    _require_mapping(raw, "calibration")
    _check_keys(raw, CalibrationConfig, "calibration")
    report = _optional_path(raw.get("validation_report"), base,
                            "calibration.validation_report")
    if report is not None and not report.is_file():
        raise FileNotFoundError(f"验证报告不存在：{report}")
    return CalibrationConfig(validation_report=report)


def _parse_sweep(raw: Any) -> SweepConfig:
    _require_mapping(raw, "sweep")
    _check_keys(raw, SweepConfig, "sweep")
    fields_raw = _mapping(raw.get("fields"), "sweep.fields")
    unknown = set(fields_raw) - HARDWARE_FIELDS
    if unknown:
        raise ValueError(f"sweep.fields 有未知字段：{sorted(unknown)}")
    sweep_fields = {}
    for key, values in fields_raw.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"sweep.fields.{key} 必须是非空取值列表")
        sweep_fields[key] = tuple(values)
    core_policy = _str(raw.get("core_policy"), "sweep.core_policy")
    if core_policy not in CORE_POLICIES:
        raise ValueError(
            f"sweep.core_policy={core_policy!r} 不在 {sorted(CORE_POLICIES)}")
    threshold = _number(raw.get("knee_improvement_threshold"),
                        "sweep.knee_improvement_threshold")
    if threshold >= 1.0:
        raise ValueError("sweep.knee_improvement_threshold 必须小于 1")
    pareto_raw = raw.get("pareto")
    _require_mapping(pareto_raw, "sweep.pareto")
    _check_keys(pareto_raw, ParetoConfig, "sweep.pareto")
    direction = _str(pareto_raw.get("direction"), "sweep.pareto.direction")
    if direction not in ("minimize", "maximize"):
        raise ValueError("sweep.pareto.direction 必须是 minimize 或 maximize")
    resource = _str(pareto_raw.get("resource"), "sweep.pareto.resource")
    if resource not in PARETO_RESOURCES:
        raise ValueError(
            f"sweep.pareto.resource={resource!r} 不在 "
            f"{sorted(PARETO_RESOURCES)}")
    return SweepConfig(
        fields=sweep_fields,
        core_policy=core_policy,
        knee_improvement_threshold=threshold,
        pareto=ParetoConfig(resource=resource, direction=direction),
    )


_PREDICTOR_HOST_RELAY_NS = 100.0


def _validate_predictor_contract(
    profile: HardwareProfile,
    profile_path: Path,
    placement: PlacementConfig,
    space: SpaceConfig,
    sweep: SweepConfig,
) -> None:
    """predictor 只接受它训练特征所固定的输入，编译前拒绝其余组合。

    GeneSim 的主机特征固定加 100 ns 中转；画像或候选扫描成其他值时，
    内层搜索会按另一套画像引导候选，与最终 pipeline 重放不一致。检查点
    也必须存在且元数据版本可解析，不能等到跑完编译才发现。
    """
    if "predictor" not in {placement.evaluation_backend,
                           placement.validation_backend}:
        return
    values = {float(profile.host_relay_overhead_ns)}
    values.update(float(value) for value in
                  space.hardware.fields.get("host_relay_overhead_ns", ()))
    values.update(float(value) for value in
                  sweep.fields.get("host_relay_overhead_ns", ()))
    unsupported = sorted(
        value for value in values
        if not math.isclose(value, _PREDICTOR_HOST_RELAY_NS,
                            rel_tol=0.0, abs_tol=1e-9))
    if unsupported:
        raise ValueError(
            "predictor 特征固定 HOST 中转 100 ns，当前画像或扫描值为 "
            f"{unsupported}；请改用 pipeline，或把 host_relay_overhead_ns "
            "限制为 100 ns")
    checkpoint = _predictor_checkpoint_error(profile, profile_path)
    if checkpoint is not None:
        raise ValueError(checkpoint)


def _predictor_checkpoint_error(profile: HardwareProfile,
                                profile_path: Path) -> str | None:
    """预测器检查点必须存在且是带 schema 版本的 checkpoint。"""
    from tune._genesim import _ensure_genesim_path, genesim_config_class
    from tune.steps import resolve_base_genesim_config
    from genesim_bridge.paths import genesim_root

    base_path = resolve_base_genesim_config(profile, profile_path)
    raw = genesim_config_class()(str(base_path)).get(
        "predictor.checkpoint_path", "")
    if not isinstance(raw, str) or not raw.strip():
        return "predictor 已启用，但没有配置 predictor.checkpoint_path"
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path, genesim_root() / path]
    resolved = next((item for item in candidates if item.is_file()), None)
    if resolved is None:
        return f"predictor 检查点不存在：{raw}"
    try:
        import torch

        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as exc:  # 损坏的检查点不能等到仿真阶段才暴露
        return f"predictor 检查点无法解析：{raw}（{type(exc).__name__}: {exc}）"
    if not isinstance(payload, Mapping):
        return f"predictor 检查点根节点必须是映射：{raw}"
    _ensure_genesim_path()
    from predictor.model import CHECKPOINT_SCHEMA_VERSION

    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return (
            f"predictor 检查点 schema 版本不匹配："
            f"{payload.get('checkpoint_schema_version')!r}，"
            f"期望 {CHECKPOINT_SCHEMA_VERSION!r}")
    return None


def _parse_output(raw: Any, base: Path) -> OutputConfig:
    _require_mapping(raw, "output")
    _check_keys(raw, OutputConfig, "output")
    return OutputConfig(dir=_required_path(raw.get("dir"), base, "output.dir"))


# ---- 基础校验 ----


def _check_keys(raw: Mapping, cls: type, where: str) -> None:
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{where} 有未知字段：{sorted(unknown)}")


def _require_mapping(value: Any, where: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} 必须是映射，收到 {type(value).__name__}")


def _mapping(value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} 必须是映射，收到 {type(value).__name__}")
    return dict(value)


def _int(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} 必须是 ≥{minimum} 的整数，收到 {value!r}")
    return value


def _optional_positive_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    return _int(value, where)


def _number(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} 必须是有限数值，收到 {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{where} 必须是 ≥{minimum} 的有限数值，收到 {value!r}")
    return float(value)


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where} 必须是布尔值，收到 {value!r}")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 必须是非空字符串，收到 {value!r}")
    return value


def _int_tuple(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} 必须是非空整数列表，收到 {value!r}")
    return tuple(_int(item, where) for item in value)


def _weights(value: Any, where: str) -> dict[str, float]:
    mapping = _mapping(value, where)
    if not mapping:
        raise ValueError(f"{where} 不能为空")
    return {key: _number(item, f"{where}.{key}", minimum=0.0)
            for key, item in mapping.items()}


def _closed_layer(enabled: bool, ranges: tuple[tuple, ...], label: str) -> None:
    """关闭的层表示冻结，只能有一个取值。"""
    if enabled:
        return
    for values in ranges:
        if len(values) != 1:
            raise ValueError(
                f"space.{label}.enabled=false 表示冻结，取值必须恰好一个，"
                f"收到 {list(values)}")


def _optional_path(value: Any, base: Path, where: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 必须是路径字符串或 null，收到 {value!r}")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _required_path(value: Any, base: Path, where: str) -> Path:
    path = _optional_path(value, base, where)
    if path is None:
        raise ValueError(f"{where} 不能为空")
    return path


def _path_text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _deep_merge(base: dict, overlay: Mapping) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged
