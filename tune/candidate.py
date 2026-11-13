"""分层候选空间：图策略、算子参数与硬件画像。

外层状态实现 GeneSim `SearchSpace` 协议（`initial_state` / `key` / `propose`
/ `recombine`），枚举与三种搜索算法共用同一套合法性判据。`full_m/full_n/
full_k` 由真实形状派生，不作为搜索变量；WRAM 只在硬件层定义。
"""

from __future__ import annotations

import json
import itertools
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from contracts.hardware_profile import HardwareProfile, LinkSpec
from contracts.op_contract import PimOptimizations
from graph.strategy import ShardStrategy, llama_strategy
from tune.config import TuneConfig


@dataclass(frozen=True)
class ModelDims:
    """从 `config.json` 读出的切分约束，不加载权重。"""

    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    num_layers: int
    hidden_size: int


@dataclass(frozen=True)
class StrategyChoice:
    """图策略层的搜索变量。"""

    num_dpus: int
    num_stages: int

    @property
    def tp_width(self) -> int:
        return self.num_dpus // self.num_stages


@dataclass(frozen=True)
class OpKnobs:
    """算子层的搜索变量；num_tasklets 继承旧导出的默认值 4。"""

    num_tasklets: int = 4


@dataclass(frozen=True)
class TuneState:
    """一个完整候选：图策略 + 算子参数 + 硬件画像 + 编译优化开关。"""

    strategy: StrategyChoice
    op: OpKnobs
    profile: HardwareProfile
    # None 表示沿用顶层 optimization；开启优化开关层时由候选空间填入具体值。
    optimization: PimOptimizations | None = None

    def resolved_optimization(self, cfg: TuneConfig) -> PimOptimizations:
        """取本候选实际生效的优化开关。"""
        return cfg.optimization if self.optimization is None else self.optimization


def _optimization_tuple(optimization: PimOptimizations) -> tuple[bool, ...]:
    """优化开关键：按字段声明顺序取布尔值，用于排序与身份比较。"""
    return tuple(optimization.to_payload().values())


def model_dims(model_dir: Path) -> ModelDims:
    """从 HuggingFace `config.json` 读取切分约束，不加载 7B 权重。"""
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"模型目录里没有 config.json：{path}")
    raw = json.loads(path.read_text())
    try:
        return ModelDims(
            num_heads=int(raw["num_attention_heads"]),
            num_kv_heads=int(raw.get("num_key_value_heads",
                                     raw["num_attention_heads"])),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            num_layers=int(raw["num_hidden_layers"]),
            hidden_size=int(raw["hidden_size"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"config.json 缺少 Llama 切分字段：{exc}") from exc


def build_strategy(state: TuneState, dims: ModelDims) -> ShardStrategy:
    """把候选转成 `ShardStrategy`，走与生产同一条合法性校验。"""
    return llama_strategy(
        state.strategy.num_dpus,
        num_stages=state.strategy.num_stages,
        num_heads=dims.num_heads,
        num_kv_heads=dims.num_kv_heads,
        intermediate_size=dims.intermediate_size,
        vocab_size=dims.vocab_size,
        num_layers=dims.num_layers,
    )


class CandidateSpace:
    """三层外层候选空间；小空间可完整枚举，大空间按邻居生成。

    关闭的层表示冻结，配置校验阶段已保证只有一个取值。枚举按数值元组
    排序；邻居一次只改一层；交叉不合法即拒绝。
    """

    def __init__(self, cfg: TuneConfig, dims: ModelDims) -> None:
        self.cfg = cfg
        self.dims = dims
        self._rejected: list[tuple[str, str]] = []
        self._strategy_choices = self._legal_strategies()
        self._op_values = tuple(cfg.space.op.num_tasklets)
        self._optimization_values = self._legal_optimizations()
        # 完整硬件组合只在 enumerate/dry-run 的完整枚举里物化；初始状态和
        # 邻居按字段懒生成，大空间不必先摊开三层笛卡尔积。
        self._hardware_combos_cache: tuple[HardwareProfile, ...] | None = None
        self._states: tuple[TuneState, ...] | None = None

    # ---- SearchSpace 协议 ----

    @property
    def initial_state(self) -> TuneState:
        """第一个合法候选；按排序主序懒扫描，不物化完整空间。"""
        for strategy in sorted(
                self._strategy_choices,
                key=lambda choice: (choice.num_dpus, choice.num_stages)):
            for tasklets in sorted(self._op_values):
                for optimization in self._optimization_values:
                    for profile in self._lazy_hardware_profiles():
                        state = TuneState(
                            strategy, OpKnobs(tasklets), profile, optimization)
                        reason = self._state_rejection(state)
                        if reason is None:
                            return state
                        label = (
                            f"strategy(num_dpus={strategy.num_dpus},"
                            f"num_stages={strategy.num_stages})+"
                            f"hardware({profile.name})")
                        self._rejected.append((label, reason))
        raise ValueError("候选空间里没有合法组合，拒绝原因见 dry-run 报告")

    def key(self, state: TuneState) -> tuple:
        """稳定身份：各层取值 + 完整画像摘要。"""
        return (
            state.strategy.num_dpus,
            state.strategy.num_stages,
            state.op.num_tasklets,
            (None if state.optimization is None
             else _optimization_tuple(state.optimization)),
            state.profile.profile_sha256(),
        )

    def propose(self, state: TuneState, rng: random.Random):
        """按层权重挑一层，生成一个只改一个字段的合法邻居。"""
        proposal_cls = _proposal_type()
        layers = self._layer_alternatives(state)
        weighted = [
            (layer, self.cfg.search.layer_weights.get(layer, 0.0), options)
            for layer, options in layers.items()
            if options and self.cfg.search.layer_weights.get(layer, 0.0) > 0.0
        ]
        total = sum(weight for _, weight, _ in weighted)
        if not weighted or total <= 0.0:
            return None
        pick = rng.random() * total
        chosen = weighted[-1]
        accum = 0.0
        for item in weighted:
            accum += item[1]
            if pick <= accum:
                chosen = item
                break
        layer, _, options = chosen
        if layer == "strategy":
            field = rng.choice(sorted(options))
            value = rng.choice(options[field])
            strategy = StrategyChoice(
                num_dpus=(value if field == "num_dpus"
                          else state.strategy.num_dpus),
                num_stages=(value if field == "num_stages"
                            else state.strategy.num_stages),
            )
            candidate = TuneState(
                strategy, state.op, state.profile, state.optimization)
        elif layer == "op":
            candidate = TuneState(
                state.strategy, OpKnobs(rng.choice(options)), state.profile,
                state.optimization)
        elif layer == "optimization":
            candidate = TuneState(state.strategy, state.op, state.profile,
                                  rng.choice(options))
        else:
            profile = rng.choice(options)
            candidate = TuneState(state.strategy, state.op, profile,
                                  state.optimization)
        reason = self._state_rejection(candidate)
        if reason is not None:
            self._rejected.append((f"{layer}_change", reason))
            return None
        return proposal_cls(state=candidate, kind=f"{layer}_change",
                            metadata={"layer": layer})

    def recombine(self, first: TuneState, second: TuneState, rng: random.Random):
        """逐字段从两个父代取一，混合后不合法即拒绝。

        策略的 DPU 数和段数分开取，混合可能得到不满足切分约束的组合；
        这类子代直接返回 None，交给算法换一个父代。
        """
        proposal_cls = _proposal_type()
        strategy = StrategyChoice(
            num_dpus=first.strategy.num_dpus if rng.random() < 0.5
            else second.strategy.num_dpus,
            num_stages=first.strategy.num_stages if rng.random() < 0.5
            else second.strategy.num_stages,
        )
        op = first.op if rng.random() < 0.5 else second.op
        optimization = (first.optimization if rng.random() < 0.5
                        else second.optimization)
        if not self.cfg.space.hardware.fields:
            profile = first.profile
        else:
            overrides = {
                key: (getattr(first.profile, key) if rng.random() < 0.5
                      else getattr(second.profile, key))
                for key in self.cfg.space.hardware.fields
            }
            try:
                profile = self.cfg.profile.with_fields(**overrides)
            except (TypeError, ValueError):
                return None
        candidate = TuneState(strategy, op, profile, optimization)
        if not self._is_known(candidate):
            return None
        return proposal_cls(state=candidate, kind="recombine",
                            metadata={"layer": "mixed"})

    # ---- 枚举与计划 ----

    @property
    def states(self) -> tuple[TuneState, ...]:
        """完整枚举（仅小空间使用），按数值元组排序。"""
        states = self._enumerate_states()
        if not states:
            raise ValueError(
                "候选空间里没有合法组合，拒绝原因见 dry-run 报告")
        return states

    def _enumerate_states(self) -> tuple[TuneState, ...]:
        """各层笛卡尔积逐项校验；非法组合保留原因但不进入候选。"""
        if self._states is None:
            states = []
            for strategy, tasklets, optimization, profile in itertools.product(
                    self._strategy_choices, self._op_values,
                    self._optimization_values, self._hardware_combos()):
                label = (
                    f"strategy(num_dpus={strategy.num_dpus},"
                    f"num_stages={strategy.num_stages})+"
                    f"hardware({profile.name})")
                state = TuneState(
                    strategy, OpKnobs(tasklets), profile, optimization)
                try:
                    # 每个完整候选都要按自己的画像核对物理资源：
                    # 8 个 DPU 配 1 个物理簇不是合法组合。
                    profile.validate_dpus(strategy.num_dpus)
                    build_strategy(state, self.dims)
                except ValueError as exc:
                    self._rejected.append((label, str(exc)))
                    continue
                states.append(state)
            self._states = tuple(sorted(states, key=self._sort_key))
        return self._states

    def plan(self) -> dict[str, Any]:
        """dry-run 用的组合计数与拒绝原因，不编译也不加载权重。"""
        return {
            "legal": len(self._enumerate_states()),
            "rejected": list(self._rejected),
            "strategy_choices": [
                (choice.num_dpus, choice.num_stages)
                for choice in self._strategy_choices
            ],
            "op_values": list(self._op_values),
            "optimization_values": [
                item.to_payload() for item in self._optimization_values],
            "hardware_combos": len(self._hardware_combos()),
        }

    # ---- 内部 ----

    def _legal_optimizations(self) -> tuple[PimOptimizations, ...]:
        """优化开关层：只改声明的那一个开关，其余取顶层配置。"""
        space = self.cfg.space.optimization
        if space.field is None:
            return (self.cfg.optimization,)
        unique = {}
        for value in space.values:
            item = replace(
                self.cfg.optimization, **{space.field: value})
            unique[_optimization_tuple(item)] = item
        return tuple(unique[key] for key in sorted(unique))

    def _legal_strategies(self) -> tuple[StrategyChoice, ...]:
        choices: list[StrategyChoice] = []
        seen = set()
        for num_dpus in self.cfg.space.strategy.num_dpus:
            for num_stages in self.cfg.space.strategy.num_stages:
                label = f"strategy(num_dpus={num_dpus},num_stages={num_stages})"
                if num_dpus % num_stages:
                    self._rejected.append(
                        (label, f"num_stages={num_stages} 不能整除 num_dpus={num_dpus}"))
                    continue
                choice = StrategyChoice(num_dpus, num_stages)
                try:
                    build_strategy(
                        TuneState(choice, OpKnobs(), self.cfg.profile), self.dims)
                except ValueError as exc:
                    self._rejected.append((label, str(exc)))
                    continue
                if (num_dpus, num_stages) in seen:
                    continue
                seen.add((num_dpus, num_stages))
                choices.append(choice)
        if not choices:
            raise ValueError(
                "图策略层没有任何合法组合，拒绝原因见 dry-run 报告")
        return tuple(choices)

    def _hardware_combos(self) -> tuple[HardwareProfile, ...]:
        """完整硬件组合只在 enumerate/dry-run 时物化一次。"""
        if self._hardware_combos_cache is None:
            self._hardware_combos_cache = self._legal_hardware()
        return self._hardware_combos_cache

    def _legal_hardware(self) -> tuple[HardwareProfile, ...]:
        combos = list(self._lazy_hardware_profiles())
        if not combos:
            raise ValueError("硬件层没有任何合法组合")
        return tuple(combos)

    def _lazy_hardware_profiles(self):
        """按配置声明顺序逐个派生硬件画像，不预先物化笛卡尔积。"""
        fields = self.cfg.space.hardware.fields
        if not fields:
            yield self.cfg.profile
            return
        keys = sorted(fields)
        for values in itertools.product(*(fields[key] for key in keys)):
            overrides = dict(zip(keys, values))
            label = "hardware(" + ",".join(
                f"{key}={value!r}" for key, value in overrides.items()) + ")"
            try:
                yield self.cfg.profile.with_fields(**overrides)
            except (TypeError, ValueError) as exc:
                self._rejected.append((label, str(exc)))

    def _sort_key(self, state: TuneState) -> tuple:
        hardware = tuple(sorted(
            (key, repr(getattr(state.profile, key)))
            for key in self.cfg.space.hardware.fields
            if key not in ("cycle_costs", "zone_cycle_factors")
        ))
        return (
            state.strategy.num_dpus,
            state.strategy.num_stages,
            state.op.num_tasklets,
            (None if state.optimization is None
             else _optimization_tuple(state.optimization)),
            hardware,
            state.profile.profile_sha256(),
        )

    def _state_rejection(self, state: TuneState) -> str | None:
        """完整候选的拒绝原因；合法时返回 None。"""
        if (state.strategy.num_dpus, state.strategy.num_stages) not in {
                (choice.num_dpus, choice.num_stages)
                for choice in self._strategy_choices}:
            return "策略组合不在声明的合法集合里"
        if state.op.num_tasklets not in self._op_values:
            return "tasklet 取值不在声明的合法集合里"
        if state.optimization is not None:
            allowed = {_optimization_tuple(item)
                       for item in self._optimization_values}
            if _optimization_tuple(state.optimization) not in allowed:
                return "优化开关取值不在声明的合法集合里"
        try:
            state.profile.validate_dpus(state.strategy.num_dpus)
        except ValueError as exc:
            return str(exc)
        try:
            build_strategy(state, self.dims)
        except ValueError as exc:
            return str(exc)
        if not self._profile_matches_space(state.profile):
            return "画像字段不在配置声明的取值集合里"
        return None

    def _is_known(self, state: TuneState) -> bool:
        return self._state_rejection(state) is None

    def _profile_matches_space(self, profile: HardwareProfile) -> bool:
        """画像必须来自配置声明的取值组合，不能是交叉出的新值。

        YAML 里的链路/复合值是字典列表，画像里已规范化成 dataclass；
        比较前统一规范化，避免把合法链路邻居误判为空间外取值。
        """
        for key, values in self.cfg.space.hardware.fields.items():
            current = getattr(profile, key)
            if not any(_values_equal(current, value) for value in values):
                return False
        return True

    def _layer_alternatives(self, state: TuneState) -> dict[str, Any]:
        """按字段给出备选值；策略层拆成 num_dpus 与 num_stages 两个字段。

        每个备选都保持另一个字段不变并与合法策略集合核对，保证一次只改
        一个字段；没有合法备选的层不参与抽样。
        """
        strategy: dict[str, tuple[int, ...]] = {}
        num_dpus = tuple(sorted({
            choice.num_dpus for choice in self._strategy_choices
            if choice.num_stages == state.strategy.num_stages
            and choice.num_dpus != state.strategy.num_dpus}))
        num_stages = tuple(sorted({
            choice.num_stages for choice in self._strategy_choices
            if choice.num_dpus == state.strategy.num_dpus
            and choice.num_stages != state.strategy.num_stages}))
        if num_dpus:
            strategy["num_dpus"] = num_dpus
        if num_stages:
            strategy["num_stages"] = num_stages
        tasklets = [
            value for value in self._op_values
            if value != state.op.num_tasklets
        ]
        optimizations = [
            item for item in self._optimization_values
            if item != state.optimization
        ]
        hardware = self._hardware_alternatives(state.profile)
        return {"strategy": strategy, "op": tasklets,
                "optimization": optimizations, "hardware": hardware}

    def _hardware_alternatives(self, current: HardwareProfile) -> list[HardwareProfile]:
        """逐字段生成硬件邻居：每个备选只改一个声明字段。"""
        diffs: list[HardwareProfile] = []
        fields = self.cfg.space.hardware.fields
        for key in sorted(fields):
            current_value = getattr(current, key)
            for value in fields[key]:
                if _values_equal(current_value, value):
                    continue
                label = f"hardware({key}={value!r})"
                try:
                    profile = current.with_fields(**{key: value})
                except (TypeError, ValueError) as exc:
                    self._rejected.append((label, str(exc)))
                    continue
                diffs.append(profile)
        return diffs


def _normalized_value(value: Any) -> Any:
    """把配置里的 YAML 形态与画像里的 dataclass 规范成同一比较形态。"""
    if isinstance(value, LinkSpec):
        return value.to_json()
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _values_equal(left: Any, right: Any) -> bool:
    return _normalized_value(left) == _normalized_value(right)


def run_enumerate(space: CandidateSpace, objective):
    """确定性枚举：按排序后的候选逐个评估，基线计入评估次数。"""
    from autotune.api import SearchResult

    start = time.monotonic()
    evaluations = []
    best_state = None
    best_evaluation = None
    for state in space.states:
        evaluation = objective(state)
        evaluations.append(evaluation)
        if best_evaluation is None or _rank(evaluation) < _rank(best_evaluation):
            best_state, best_evaluation = state, evaluation
    if best_state is None:
        raise ValueError("候选空间为空")
    return SearchResult(
        algorithm="enumerate",
        best_state=best_state,
        best_evaluation=best_evaluation,
        initial_evaluation=evaluations[0],
        evaluations=len(evaluations),
        objective_calls=len(evaluations),
        proposals=0,
        accepted_moves=0,
        iterations=len(evaluations),
        stop_reason="search_space_exhausted",
        elapsed_s=time.monotonic() - start,
    )


def _rank(evaluation) -> float:
    return evaluation.score if evaluation.successful else math.inf


def _proposal_type():
    from autotune.api import Proposal

    return Proposal
