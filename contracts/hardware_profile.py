"""硬件画像：编译与仿真共用的唯一参数源。

一份画像同时派生 `PIMHardwareConfig`、`HwBudget`、固定 PU 映射输入和
GeneSim 的生效配置。容量在接口上一律用整数字节，GiB 只在 GeneSim 字段边界
按 `2**30` 换算；时间用纳秒。时钟、周期和链路只改仿真，其余改动参与编译键。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from contracts.op_contract import (
    PIM_COMPILER_TARGET,
    PIMHardwareConfig,
    PimOptimizations,
)

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
# 编译器目标固定为 pim:v1；`pim_type` 是器件身份，两者不是一回事。
COMPILER_TARGET = PIM_COMPILER_TARGET
# 与 GeneSim 通信库同一组路由模式与资源域；CEVA 不允许 GPU 数据通路。
ROUTE_MODES = ("direct", "host_relay")
CEVA_LINK_DOMAINS = ("HOST", "PIM", "CPU")

# 参与编译键的字段。时钟、周期、链路和基配置只影响仿真。
_COMPILE_FIELDS = (
    "version", "compiler_target", "pim_type",
    "num_pim_devices", "clusters_per_device", "tensors_per_cluster",
    "cluster_memory_bytes", "sys_reserve_bytes",
    "wram_bytes_per_dpu", "dma_align",
)


@dataclass(frozen=True)
class LinkSpec:
    """一条链路的带宽与基础延迟，单位 Gbit/s、ns。"""

    name: str
    src_domain: str
    dst_domain: str
    bandwidth_gbps: float
    latency_base_ns: float
    is_full_duplex: bool = True
    intra_cluster_only: bool = False
    inter_cluster_only: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "src_domain": self.src_domain,
            "dst_domain": self.dst_domain,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_base_ns": self.latency_base_ns,
            "is_full_duplex": self.is_full_duplex,
            "intra_cluster_only": self.intra_cluster_only,
            "inter_cluster_only": self.inter_cluster_only,
        }


@dataclass(frozen=True)
class CevaProfile:
    """CEVA 专属字段；非 CEVA 画像为 None。"""

    engines_per_core: int
    target_name: str
    local_memory_bytes: int

    def to_json(self) -> dict[str, Any]:
        return {
            "engines_per_core": self.engines_per_core,
            "target_name": self.target_name,
            "local_memory_bytes": self.local_memory_bytes,
        }


@dataclass(frozen=True)
class HardwareProfile:
    """一套完整的软硬件参数。所有字段必填，三份基线画像完整给出。"""

    version: int
    name: str
    compiler_target: str
    pim_type: str
    num_pim_devices: int
    clusters_per_device: int
    tensors_per_cluster: int
    cluster_memory_bytes: int
    sys_reserve_bytes: int
    wram_bytes_per_dpu: int
    dma_align: int
    clock_period_ns: float
    multipliers: int
    cycle_costs: Mapping[str, float]
    zone_cycle_factors: Mapping[str, float]
    tile_size: int
    mac_width: int
    chunk_size: int
    max_bulk_transfer_bytes: int
    bytes_per_element: int
    use_long_loop_counts: bool
    inter_cluster_communication: str
    host_relay_overhead_ns: float
    links: tuple[LinkSpec, ...]
    cpu_model: Mapping[str, float]
    parameter_sources: Mapping[str, Mapping[str, str]]
    base_genesim_config: str
    ceva: CevaProfile | None = None

    def __post_init__(self) -> None:
        # 加载时复制映射，候选通过 replace 修改，不共享可变状态。
        object.__setattr__(self, "cycle_costs", dict(self.cycle_costs))
        object.__setattr__(self, "zone_cycle_factors", dict(self.zone_cycle_factors))
        object.__setattr__(self, "cpu_model", dict(self.cpu_model))
        object.__setattr__(
            self, "parameter_sources",
            {key: dict(value) for key, value in self.parameter_sources.items()},
        )
        object.__setattr__(self, "links", tuple(self.links))
        _validate_profile(self)

    @property
    def total_clusters(self) -> int:
        """物理 ClusterPU 总数，即逻辑 DPU 上限。"""
        return self.num_pim_devices * self.clusters_per_device

    @property
    def cluster_capacity_gib(self) -> float:
        """整簇容量按 GiB 表示，只用于 GeneSim 的 GiB 字段。"""
        return self.cluster_memory_bytes / 2**30

    @property
    def tensor_pu_memory_bytes(self) -> int:
        """单个 TensorPU 的可用容量：扣除系统预留后按簇内 PU 数均分。

        与 GeneSim `resolve_pim_capacity_limits` 同一公式，不能再按物理
        容量均分，否则编译前置会比仿真多出预留份额。
        """
        usable = self.cluster_memory_bytes - self.sys_reserve_bytes
        return usable // self.tensors_per_cluster

    def validate_dpus(self, num_dpus: int) -> int:
        """校验逻辑 DPU 数是 2 的幂且不超过物理簇数。"""
        if isinstance(num_dpus, bool) or not isinstance(num_dpus, int) or num_dpus <= 0:
            raise ValueError(f"num_dpus 必须是正整数，收到 {num_dpus!r}")
        if num_dpus > self.total_clusters:
            raise ValueError(
                f"num_dpus={num_dpus} 超过物理簇数 {self.total_clusters}"
                "（设备数 × 每设备簇数）")
        if num_dpus & (num_dpus - 1):
            raise ValueError(f"num_dpus 必须是 2 的幂，收到 {num_dpus}")
        return num_dpus

    def validate_tensor_pu_bytes(self, nbytes: int) -> None:
        """单 PU 容量判据：多 TensorPU 时不能复用整簇预算。"""
        if nbytes > self.tensor_pu_memory_bytes:
            raise ValueError(
                f"单台 DPU 需要 {nbytes} B，超过单个 TensorPU 容量 "
                f"{self.tensor_pu_memory_bytes} B")

    def to_pim_hardware_config(
        self, num_dpus: int, num_tasklets: int, *,
        optimizations: PimOptimizations | None = None,
    ) -> PIMHardwareConfig:
        """派生图/算子编译用的硬件契约；编译目标与画像同源。

        `optimizations` 是调优配置的编译优化开关，缺省全开，旧调用不变。
        """
        self.validate_dpus(num_dpus)
        return PIMHardwareConfig(
            num_dpus=num_dpus,
            num_tasklets=num_tasklets,
            mram_bytes_per_dpu=self.cluster_memory_bytes,
            wram_bytes_per_dpu=self.wram_bytes_per_dpu,
            dma_align=self.dma_align,
            compiler_target=self.compiler_target,
            optimizations=optimizations or PimOptimizations(),
        )

    def to_hw_budget(self):
        """派生单台 DPU 的 MRAM 预算，与编译五元组同源。"""
        from memory.mem_planner import HwBudget

        return HwBudget(
            mram_bytes=self.cluster_memory_bytes,
            align=self.dma_align,
            sys_reserve_bytes=self.sys_reserve_bytes,
        )

    def compile_profile_sha256(self) -> str:
        """只覆盖编译相关字段的摘要；时钟变化不触发重编。"""
        payload = {field: getattr(self, field) for field in _COMPILE_FIELDS}
        return _sha256_json(payload)

    def profile_sha256(self) -> str:
        """完整画像摘要。基配置路径不入语义键，内容在外层另哈希。"""
        payload = self.to_json()
        payload.pop("base_genesim_config", None)
        return _sha256_json(payload)

    def genesim_overrides(self) -> dict[str, Any]:
        """把画像翻译成 GeneSim 配置覆盖项，GiB 只在边界换算。"""
        overrides: dict[str, Any] = {
            "system": {
                "num_pim_devices": self.num_pim_devices,
                "pim_type": self.pim_type,
            },
            "pim": {
                "clusters_per_device": self.clusters_per_device,
                "tensors_per_cluster": self.tensors_per_cluster,
                "inter_cluster_communication": self.inter_cluster_communication,
                # 物理容量与系统预留必须同源进入仿真；通用后端同样消费。
                "sys_reserve_bytes": self.sys_reserve_bytes,
                "tensor_pu_config": {"multipliers": self.multipliers},
                "clock_period_ns": self.clock_period_ns,
                "cluster_memory_capacity_gb": self.cluster_capacity_gib,
                "zone_cycle_factors": dict(self.zone_cycle_factors),
                "compile": {
                    "tile_size": self.tile_size,
                    "mac_width": self.mac_width,
                    "chunk_size": self.chunk_size,
                    "max_bulk_transfer_bytes": self.max_bulk_transfer_bytes,
                    "bytes_per_element": self.bytes_per_element,
                    "use_long_loop_counts": self.use_long_loop_counts,
                },
                "pim_trace_cycle_costs": dict(self.cycle_costs),
            },
            "comm_config": {
                "host_relay_overhead_ns": self.host_relay_overhead_ns,
                "links": [link.to_json() for link in self.links],
            },
            "cpu_model": dict(self.cpu_model),
        }
        if self.ceva is not None:
            overrides["ceva"] = {
                "execution_scope": "npm_core",
                **self.ceva.to_json(),
                "sys_reserve_bytes": self.sys_reserve_bytes,
                "parameter_sources": {
                    key: dict(value) for key, value in self.parameter_sources.items()
                },
            }
        return overrides

    def to_json(self) -> dict[str, Any]:
        """完整画像的 JSON 形态，供落盘和内容键使用。"""
        return {
            "version": self.version,
            "name": self.name,
            "compiler_target": self.compiler_target,
            "pim_type": self.pim_type,
            "num_pim_devices": self.num_pim_devices,
            "clusters_per_device": self.clusters_per_device,
            "tensors_per_cluster": self.tensors_per_cluster,
            "cluster_memory_bytes": self.cluster_memory_bytes,
            "sys_reserve_bytes": self.sys_reserve_bytes,
            "wram_bytes_per_dpu": self.wram_bytes_per_dpu,
            "dma_align": self.dma_align,
            "clock_period_ns": self.clock_period_ns,
            "multipliers": self.multipliers,
            "cycle_costs": dict(self.cycle_costs),
            "zone_cycle_factors": dict(self.zone_cycle_factors),
            "tile_size": self.tile_size,
            "mac_width": self.mac_width,
            "chunk_size": self.chunk_size,
            "max_bulk_transfer_bytes": self.max_bulk_transfer_bytes,
            "bytes_per_element": self.bytes_per_element,
            "use_long_loop_counts": self.use_long_loop_counts,
            "inter_cluster_communication": self.inter_cluster_communication,
            "host_relay_overhead_ns": self.host_relay_overhead_ns,
            "links": [link.to_json() for link in self.links],
            "cpu_model": dict(self.cpu_model),
            "parameter_sources": {
                key: dict(value) for key, value in self.parameter_sources.items()
            },
            "base_genesim_config": self.base_genesim_config,
            "ceva": None if self.ceva is None else self.ceva.to_json(),
        }

    def with_fields(self, **changes: Any) -> "HardwareProfile":
        """按字段派生新画像，用于硬件层候选和扫描。

        与 `build_profile()` 共用同一条构造路径：YAML 形态的 links/ceva 字典
        会被规范化成 dataclass，非法值在候选展开前失败。
        """
        data = self.to_json()
        ceva_raw = data.get("ceva")
        if "cluster_memory_bytes" in changes and "ceva" not in changes and ceva_raw:
            changes["ceva"] = {
                **ceva_raw, "local_memory_bytes": int(changes["cluster_memory_bytes"])}
        return build_profile(data, changes)


PROFILE_FIELDS = frozenset(HardwareProfile.__dataclass_fields__)


def build_profile(raw: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> HardwareProfile:
    """从 JSON 形态构造画像；未知字段和非法取值直接报错。"""
    if not isinstance(raw, Mapping):
        raise ValueError(f"画像必须是 JSON object，收到 {type(raw).__name__}")
    data = dict(raw)
    unknown = set(data) - PROFILE_FIELDS
    if unknown:
        raise ValueError(f"画像里有未知字段：{sorted(unknown)}")
    missing = PROFILE_FIELDS - set(data) - {"ceva"}
    if missing:
        raise ValueError(f"画像缺少必填字段：{sorted(missing)}")
    if overrides:
        unknown = set(overrides) - PROFILE_FIELDS - {"ceva"}
        if unknown:
            raise ValueError(f"画像覆盖里有未知字段：{sorted(unknown)}")
        data.update(overrides)
        # CEVA 容量与 local_memory_bytes 必须同源，覆盖容量时一起改。
        if "cluster_memory_bytes" in overrides and "ceva" not in overrides:
            ceva_raw = data.get("ceva")
            if isinstance(ceva_raw, Mapping):
                data["ceva"] = {**ceva_raw,
                                "local_memory_bytes": int(overrides["cluster_memory_bytes"])}
    data["links"] = tuple(
        LinkSpec(**link) if isinstance(link, Mapping) else link
        for link in data["links"]
    )
    ceva = data.get("ceva")
    if isinstance(ceva, Mapping):
        unknown = set(ceva) - set(CevaProfile.__dataclass_fields__)
        if unknown:
            raise ValueError(f"CEVA 画像里有未知字段：{sorted(unknown)}")
        data["ceva"] = CevaProfile(**ceva)
    return HardwareProfile(**data)


def load_profile(path: Path | str) -> HardwareProfile:
    """读取一份 JSON 画像；文件不存在时明确报错，不退回默认值。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到硬件画像：{path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"硬件画像 {path} 不是合法 JSON：{exc}") from exc
    return build_profile(raw)


def profile_files() -> list[Path]:
    """列出仓库内置的全部画像文件。"""
    return sorted(PROFILE_DIR.glob("*.json"))


def _sha256_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _require_int(value: Any, name: str, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必须是 ≥{minimum} 的整数，收到 {value!r}")


def _require_float(value: Any, name: str, *, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值，收到 {value!r}")
    if value != value or value in (float("inf"), float("-inf")) or value < minimum:
        raise ValueError(f"{name} 必须是 ≥{minimum} 的有限数值，收到 {value!r}")


def _require_positive_float(value: Any, name: str) -> None:
    """时钟等物理量必须严格为正；零会让所有周期成本消失。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是正数，收到 {value!r}")
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        raise ValueError(f"{name} 必须是正的有限数值，收到 {value!r}")


def _validate_profile(profile: HardwareProfile) -> None:
    """字段级校验：类型、范围、CEVA 硬约束和三份画像的完整性。"""
    _require_int(profile.version, "version")
    if profile.version != 1:
        raise ValueError(f"不支持的画像版本 {profile.version}")
    for name in ("name", "pim_type", "base_genesim_config"):
        value = getattr(profile, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串，收到 {value!r}")
    # 跨核路由只有两种实现；未登记的取值在运行时会按另一种模式处理。
    if profile.inter_cluster_communication not in ROUTE_MODES:
        raise ValueError(
            "inter_cluster_communication 只能是 "
            f"{'/'.join(ROUTE_MODES)}，收到 "
            f"{profile.inter_cluster_communication!r}")
    if profile.compiler_target != COMPILER_TARGET:
        raise ValueError(
            f"compiler_target 固定为 {COMPILER_TARGET!r}，收到 {profile.compiler_target!r}")
    for name in (
        "num_pim_devices", "clusters_per_device", "tensors_per_cluster",
        "cluster_memory_bytes", "wram_bytes_per_dpu", "multipliers",
        "tile_size", "mac_width", "chunk_size", "max_bulk_transfer_bytes",
        "bytes_per_element",
    ):
        _require_int(getattr(profile, name), name)
    _require_int(profile.dma_align, "dma_align")
    if profile.dma_align & (profile.dma_align - 1):
        raise ValueError(f"dma_align 必须是 2 的幂，收到 {profile.dma_align}")
    if isinstance(profile.sys_reserve_bytes, bool) or not isinstance(profile.sys_reserve_bytes, int):
        raise ValueError("sys_reserve_bytes 必须是整数")
    if not 0 <= profile.sys_reserve_bytes < profile.cluster_memory_bytes:
        raise ValueError("sys_reserve_bytes 必须非负且小于 cluster_memory_bytes")
    # 与 GeneSim 的容量校验同口径：扣预留后单 PU 至少要有一字节。
    if ((profile.cluster_memory_bytes - profile.sys_reserve_bytes)
            // profile.tensors_per_cluster) <= 0:
        raise ValueError(
            "扣除系统预留后单 TensorPU 容量必须为正："
            f"{profile.cluster_memory_bytes} - {profile.sys_reserve_bytes} "
            f"不能按 {profile.tensors_per_cluster} 个 PU 均分")
    _require_positive_float(profile.clock_period_ns, "clock_period_ns")
    _require_float(profile.host_relay_overhead_ns, "host_relay_overhead_ns")
    if not isinstance(profile.use_long_loop_counts, bool):
        raise ValueError("use_long_loop_counts 必须是布尔值")
    for name, mapping in (
        ("cycle_costs", profile.cycle_costs),
        ("zone_cycle_factors", profile.zone_cycle_factors),
        ("cpu_model", profile.cpu_model),
    ):
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"{name} 必须是非空映射")
        for key, value in mapping.items():
            _require_float(value, f"{name}.{key}")
    if not profile.links:
        raise ValueError("links 不能为空")
    allowed_domains = (
        CEVA_LINK_DOMAINS if profile.ceva is not None else None)
    seen_names: set[str] = set()
    for link in profile.links:
        if not isinstance(link.name, str) or not link.name.strip():
            raise ValueError(f"链路名称必须是非空字符串，收到 {link.name!r}")
        if link.name in seen_names:
            raise ValueError(f"链路名称重复：{link.name!r}")
        seen_names.add(link.name)
        for field in ("src_domain", "dst_domain"):
            value = getattr(link, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"链路 {link.name!r} 的 {field} 必须是非空字符串，"
                    f"收到 {value!r}")
            if allowed_domains is not None and value not in allowed_domains:
                raise ValueError(
                    f"链路 {link.name!r} 的 {field} 端点 {value!r} 不属于 "
                    f"可用资源域 {list(allowed_domains)}")
        # 语义标记只认布尔值；字符串/整数的真值转换会改变计费和路由。
        for field in (
                "is_full_duplex", "intra_cluster_only", "inter_cluster_only"):
            value = getattr(link, field)
            if not isinstance(value, bool):
                raise ValueError(
                    f"链路 {link.name!r} 的 {field} 必须是布尔值，"
                    f"收到 {value!r}")
        _require_float(link.bandwidth_gbps, f"links.{link.name}.bandwidth_gbps")
        _require_float(link.latency_base_ns, f"links.{link.name}.latency_base_ns")
    _validate_zone_cycle_factors(profile.zone_cycle_factors)
    if not isinstance(profile.parameter_sources, Mapping) or not profile.parameter_sources:
        raise ValueError("parameter_sources 必须是非空映射")
    for key, entry in profile.parameter_sources.items():
        if not isinstance(entry, Mapping) or not entry:
            raise ValueError(f"parameter_sources.{key} 必须是非空映射")
        for field, value in entry.items():
            if not isinstance(value, str):
                raise ValueError(f"parameter_sources.{key}.{field} 必须是字符串")
    if profile.ceva is not None:
        _validate_ceva(profile)


@lru_cache(maxsize=1)
def _genesim_zone_validator():
    """加载 GeneSim 的区间倍率校验函数；跨仓调用方可能撤掉了本仓根路径。"""
    repo = Path(__file__).resolve().parent.parent
    added = str(repo) not in sys.path
    if added:
        sys.path.append(str(repo))
    try:
        from genesim_bridge.paths import genesim_root

        root = genesim_root()
    finally:
        if added:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.append(str(src))
    from pim.pim_isa import validate_zone_cycle_factors

    return validate_zone_cycle_factors


def _validate_zone_cycle_factors(factors: Mapping[str, float]) -> None:
    """复用 GeneSim 的区间倍率编码约束，避免两份宽严不同的校验。

    倍率会被编码进 12 位指令字段（src2=2×factor），低于 1 或非 0.5 整数倍
    都会被硬件静默舍入；调优侧必须在编译前拒绝同一批非法值。
    """
    _genesim_zone_validator()(dict(factors), require_all=True)


def _validate_ceva(profile: HardwareProfile) -> None:
    """CEVA 的既有约束：单设备、64/128 核、每簇一个 TensorPU。"""
    if profile.num_pim_devices != 1:
        raise ValueError("CEVA 画像只允许单台 PIM 设备")
    if profile.clusters_per_device not in (64, 128):
        raise ValueError(
            f"CEVA 画像每设备只支持 64 或 128 核，收到 {profile.clusters_per_device}")
    if profile.tensors_per_cluster != 1:
        raise ValueError("CEVA 画像每簇只允许一个 TensorPU")
    ceva = profile.ceva
    _require_int(ceva.engines_per_core, "ceva.engines_per_core")
    _require_int(ceva.local_memory_bytes, "ceva.local_memory_bytes")
    if not isinstance(ceva.target_name, str) or not ceva.target_name.strip():
        raise ValueError("ceva.target_name 必须是非空字符串")
    if ceva.local_memory_bytes != profile.cluster_memory_bytes:
        raise ValueError(
            "CEVA local_memory_bytes 必须与 cluster_memory_bytes 同源："
            f"{ceva.local_memory_bytes} vs {profile.cluster_memory_bytes}")
    # GeneSim 的 CEVA 链路契约要求带宽严格为正；零带宽会在通信库构造时才失败。
    for link in profile.links:
        if link.bandwidth_gbps <= 0:
            raise ValueError(
                f"CEVA 链路 {link.name!r} 的 bandwidth_gbps 必须为正数，"
                f"收到 {link.bandwidth_gbps!r}")
    # 主机编排成本进入总时延；CEVA 正式评估要求显式且严格为正。
    for key in ("base_latency_us", "peak_flops", "peak_bandwidth_gbs"):
        if key not in profile.cpu_model:
            raise ValueError(f"CEVA 画像必须显式配置 cpu_model.{key}")
        _require_positive_float(
            profile.cpu_model[key], f"cpu_model.{key}")
    pairs = {
        (link.src_domain, link.dst_domain) for link in profile.links}
    for pair in (("HOST", "PIM"), ("PIM", "HOST"),
                 ("CPU", "HOST"), ("HOST", "CPU")):
        if pair not in pairs:
            raise ValueError(
                f"CEVA 画像缺少链路 {pair[0]}→{pair[1]}")
    if (profile.inter_cluster_communication == "direct"
            and ("PIM", "PIM") not in pairs):
        raise ValueError(
            "inter_cluster_communication=direct 时缺少 PIM→PIM 链路")
