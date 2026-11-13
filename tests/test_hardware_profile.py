"""硬件画像的单一真源：单位换算、DPU 上限与摘要口径。"""

import json
from pathlib import Path

import pytest

from contracts.hardware_profile import (
    HardwareProfile,
    build_profile,
    load_profile,
    profile_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "contracts" / "profiles"


def test_ceva_decimal_bytes_convert_to_gib_at_the_boundary():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    assert profile.cluster_memory_bytes == 8_000_000_000
    assert profile.cluster_capacity_gib == pytest.approx(7.450580596923828)
    # 十进制 8 GB 不是 8 GiB，两者不能混写。
    assert profile.cluster_capacity_gib != 8.0


def test_generic_profile_uses_binary_gib_capacity():
    profile = load_profile(PROFILE_DIR / "generic.json")
    assert profile.cluster_memory_bytes == 8 * 2**30
    assert profile.cluster_capacity_gib == 8.0
    # 每簇 16 个 TensorPU 时单 PU 容量按整簇预算均分。
    assert profile.tensor_pu_memory_bytes == (8 * 2**30) // 16


def test_ceva_64core_accepts_eight_dpus_and_rejects_sixty_five():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    assert profile.total_clusters == 64
    assert profile.validate_dpus(8) == 8
    with pytest.raises(ValueError, match="超过物理簇数"):
        profile.validate_dpus(65)


def test_generic_single_tensor_pu_capacity_rejects_oversized_need():
    profile = load_profile(PROFILE_DIR / "generic.json")
    assert profile.tensor_pu_memory_bytes == (8 * 2**30) // 16
    profile.validate_tensor_pu_bytes(profile.tensor_pu_memory_bytes)
    with pytest.raises(ValueError, match="超过单个 TensorPU 容量"):
        profile.validate_tensor_pu_bytes(profile.tensor_pu_memory_bytes + 1)


def test_zone_cycle_factor_reuses_genesim_encoding_rules():
    """区间倍率必须满足 GeneSim 的 12 位编码约束，不能只查非负有限。"""
    profile = load_profile(PROFILE_DIR / "generic.json")
    accepted = build_profile(
        profile.to_json(),
        overrides={"zone_cycle_factors": {
            "weight": 1.5, "kv": 2.0, "activation": 1.0}})
    assert accepted.zone_cycle_factors["weight"] == 1.5
    with pytest.raises(ValueError, match="0.5 的整数倍"):
        build_profile(
            profile.to_json(),
            overrides={"zone_cycle_factors": {
                "weight": 0.2, "kv": 1.0, "activation": 1.0}})
    with pytest.raises(ValueError, match="区间倍率"):
        build_profile(
            profile.to_json(),
            overrides={"zone_cycle_factors": {"weight": 1.0}})


def test_budget_and_compile_config_share_one_source():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    hw = profile.to_pim_hardware_config(num_dpus=8, num_tasklets=4)
    budget = profile.to_hw_budget()
    assert hw.mram_bytes_per_dpu == budget.mram_bytes
    assert hw.dma_align == budget.align
    assert hw.wram_bytes_per_dpu == profile.wram_bytes_per_dpu
    assert hw.num_dpus == 8
    assert hw.num_tasklets == 4
    assert budget.sys_reserve_bytes == profile.sys_reserve_bytes


def test_compile_digest_ignores_clock_but_full_digest_tracks_it():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    faster = build_profile(
        profile.to_json(),
        overrides={"clock_period_ns": 0.5, "cycle_costs": {**profile.cycle_costs, "MAC": 2}},
    )
    assert faster.compile_profile_sha256() == profile.compile_profile_sha256()
    assert faster.profile_sha256() != profile.profile_sha256()
    assert faster.to_pim_hardware_config(8, 4) == profile.to_pim_hardware_config(8, 4)


def test_capacity_change_updates_compile_digest():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    smaller = build_profile(
        profile.to_json(),
        overrides={"cluster_memory_bytes": 7_000_000_000},
    )
    assert smaller.compile_profile_sha256() != profile.compile_profile_sha256()


def test_ceva_constraints_reject_wrong_shape(tmp_path):
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    raw = profile.to_json()
    raw["clusters_per_device"] = 32
    path = tmp_path / "bad_ceva.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="CEVA"):
        load_profile(path)


def test_unknown_field_is_rejected(tmp_path):
    raw = json.loads((PROFILE_DIR / "generic.json").read_text())
    raw["unexpected_knob"] = 1
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="未知字段"):
        load_profile(path)


def test_missing_profile_file_is_rejected():
    with pytest.raises(FileNotFoundError):
        load_profile(PROFILE_DIR / "does_not_exist.json")


def test_bool_is_not_an_int(tmp_path):
    raw = json.loads((PROFILE_DIR / "generic.json").read_text())
    raw["tile_size"] = True
    path = tmp_path / "bool.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="tile_size"):
        load_profile(path)


def test_profiles_exist_for_all_three_baselines():
    names = {path.stem for path in profile_files()}
    assert {"generic", "ceva_64core_4e", "ceva_128core_4e"} <= names


def test_profile_round_trips_and_copies_mappings():
    profile = load_profile(PROFILE_DIR / "generic.json")
    rebuilt = build_profile(profile.to_json())
    assert rebuilt == profile
    assert isinstance(rebuilt.cycle_costs, dict)
    # 加载出的映射是副本，改它不会污染原画像。
    rebuilt.cycle_costs["MAC"] = 999
    assert profile.cycle_costs["MAC"] != 999


def test_genesim_overrides_use_boundary_conversion():
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    overrides = profile.genesim_overrides()
    assert overrides["pim"]["cluster_memory_capacity_gb"] == pytest.approx(
        profile.cluster_capacity_gib)
    assert overrides["pim"]["tensor_pu_config"]["multipliers"] == profile.multipliers
    assert overrides["comm_config"]["links"][0]["name"] == profile.links[0].name
    assert overrides["ceva"]["local_memory_bytes"] == profile.cluster_memory_bytes


def test_with_fields_normalizes_link_and_ceva_mappings():
    """扫描/交叉传 YAML 形态的字典时也走同一条构造校验。"""
    from contracts.hardware_profile import LinkSpec, load_profile

    path = REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"
    profile = load_profile(path)
    raw_links = [link.to_json() for link in profile.links]
    raw_links[0]["bandwidth_gbps"] = profile.links[0].bandwidth_gbps * 2
    changed = profile.with_fields(links=raw_links)
    assert isinstance(changed.links[0], LinkSpec)
    assert changed.links[0].bandwidth_gbps == pytest.approx(
        profile.links[0].bandwidth_gbps * 2)
    # 改容量仍同步 CEVA 的同源视图。
    smaller = profile.with_fields(cluster_memory_bytes=7_000_000_000)
    assert smaller.ceva.local_memory_bytes == 7_000_000_000


# ---- 评审 round03：CEVA 链路正数契约 ----

def test_ceva_zero_bandwidth_link_is_rejected_before_compile():
    """GeneSim 的 CEVA 通信库拒绝零带宽；画像必须在同一契约上提前失败。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    links = [link.to_json() for link in profile.links]
    links[0]["bandwidth_gbps"] = 0.0
    with pytest.raises(ValueError, match="bandwidth_gbps"):
        profile.with_fields(links=links)


def test_generic_zero_bandwidth_link_keeps_legacy_behavior():
    """通用画像的零带宽表示不可用路由，不套用 CEVA 的正数限制。"""
    profile = load_profile(PROFILE_DIR / "generic.json")
    links = [link.to_json() for link in profile.links]
    links[0]["bandwidth_gbps"] = 0.0
    changed = profile.with_fields(links=links)
    assert changed.links[0].bandwidth_gbps == 0.0


# ---- 评审 round04：路由、链路语义标记与主机参数的前置契约 ----

def test_unknown_route_mode_is_rejected_before_compile():
    """未登记的跨核路由必须前置拒绝，不能留给运行期按默认模式处理。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    with pytest.raises(ValueError, match="inter_cluster_communication"):
        profile.with_fields(inter_cluster_communication="bogus")
    # 合法的 direct/host_relay 不能被误拒。
    assert profile.with_fields(
        inter_cluster_communication="host_relay"
    ).inter_cluster_communication == "host_relay"


def test_link_semantic_flags_must_be_real_booleans():
    """链路三个语义标记只认布尔值，字符串 'false' 不能被真值转换接受。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    for flag in ("is_full_duplex", "intra_cluster_only",
                 "inter_cluster_only"):
        links = [link.to_json() for link in profile.links]
        links[0][flag] = "false"
        with pytest.raises(ValueError, match=flag):
            profile.with_fields(links=links)


def test_link_endpoint_domains_follow_the_backend_capability():
    """通用画像允许 GPU 端点；CEVA 画像不允许 GPU 数据通路。"""
    generic = load_profile(PROFILE_DIR / "generic.json")
    assert any("GPU" in (link.src_domain, link.dst_domain)
               for link in generic.links)
    ceva = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    links = [link.to_json() for link in ceva.links]
    links[0]["src_domain"] = "GPU"
    with pytest.raises(ValueError, match="资源域"):
        ceva.with_fields(links=links)
    links = [link.to_json() for link in ceva.links]
    links[0]["src_domain"] = "OTHER"
    with pytest.raises(ValueError, match="资源域"):
        ceva.with_fields(links=links)


def test_ceva_host_parameters_must_be_strictly_positive():
    """CEVA 主机参数零值会让 GeneSim 拒绝，画像必须提前报同一错误。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    for key in ("base_latency_us", "peak_flops", "peak_bandwidth_gbs"):
        with pytest.raises(ValueError, match=key):
            profile.with_fields(cpu_model=dict(profile.cpu_model, **{key: 0.0}))
    # 通用画像的零值保持既有兼容行为。
    generic = load_profile(PROFILE_DIR / "generic.json")
    assert generic.with_fields(
        cpu_model=dict(generic.cpu_model, base_latency_us=0.0)
    ).cpu_model["base_latency_us"] == 0.0


def test_ceva_direct_route_requires_the_inter_cluster_link():
    """direct 路由必须有 PIM→PIM 链路，否则运行期会缺通道。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    links = [
        link.to_json() for link in profile.links
        if (link.src_domain, link.dst_domain) != ("PIM", "PIM")]
    with pytest.raises(ValueError, match="PIM→PIM"):
        profile.with_fields(links=links)


# ---- 评审 round04：通用系统预留与仿真容量同源 ----

def test_generic_reserve_flows_into_effective_simulation_capacity():
    """通用画像的预留必须进入 GeneSim 生效配置并产生同一可用容量。"""
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from config_loader import GeneSimConfig
    from pim.capacity import resolve_pim_capacity_limits

    from tune.steps import resolve_base_genesim_config

    profile = load_profile(PROFILE_DIR / "generic.json").with_fields(
        sys_reserve_bytes=1024)
    overrides = profile.genesim_overrides()
    assert overrides["pim"]["sys_reserve_bytes"] == 1024
    base = resolve_base_genesim_config(
        profile, PROFILE_DIR / "generic.json")
    effective = GeneSimConfig(str(base)).override(**overrides)
    limits = resolve_pim_capacity_limits(effective)
    budget = profile.to_hw_budget()
    assert limits["usable_bytes"] == budget.mram_bytes - budget.sys_reserve_bytes
    assert limits["tensor_pu_capacity_bytes"] == (
        budget.mram_bytes - budget.sys_reserve_bytes
    ) // profile.tensors_per_cluster
    # 编译态与仿真态读的是同一物理量与同一预留。
    assert budget.mram_bytes == limits["total_capacity_bytes"]


def test_ceva_reserve_keeps_both_views_consistent():
    """CEVA 的预留同时出现在 pim 与 ceva 视图，且数值一致。"""
    profile = load_profile(PROFILE_DIR / "ceva_64core_4e.json")
    overrides = profile.genesim_overrides()
    assert overrides["pim"]["sys_reserve_bytes"] == profile.sys_reserve_bytes
    assert overrides["ceva"]["sys_reserve_bytes"] == profile.sys_reserve_bytes


# ---- 评审 round05：单 TensorPU 限额必须扣除系统预留 ----

def _simulation_tensor_pu_limit(profile, profile_path):
    from tune._genesim import _ensure_genesim_path

    _ensure_genesim_path()
    from config_loader import GeneSimConfig
    from pim.capacity import resolve_pim_capacity_limits

    from tune.steps import resolve_base_genesim_config

    base = resolve_base_genesim_config(profile, profile_path)
    effective = GeneSimConfig(str(base)).override(**profile.genesim_overrides())
    return resolve_pim_capacity_limits(effective)


@pytest.mark.parametrize(
    "name", ["generic.json", "ceva_64core_4e.json", "ceva_128core_4e.json"])
def test_tensor_pu_memory_bytes_is_derived_from_usable_capacity(name):
    """零/非零预留下，画像导出的单 PU 限额与 GeneSim 逐字节一致。"""
    for reserve in (0, 1024):
        profile = load_profile(PROFILE_DIR / name).with_fields(
            sys_reserve_bytes=reserve)
        expected = (profile.cluster_memory_bytes
                    - profile.sys_reserve_bytes) // profile.tensors_per_cluster
        assert profile.tensor_pu_memory_bytes == expected
        limits = _simulation_tensor_pu_limit(profile, PROFILE_DIR / name)
        assert profile.tensor_pu_memory_bytes == (
            limits["tensor_pu_capacity_bytes"])
        assert limits["usable_bytes"] == (
            profile.cluster_memory_bytes - reserve)


@pytest.mark.parametrize(
    "name", ["generic.json", "ceva_64core_4e.json"])
def test_tensor_pu_limit_boundary_rejects_legacy_physical_share(name):
    """单 PU 判据恰好到限额可过，超 1 B 与旧物理均分口径都必须拒绝。"""
    profile = load_profile(PROFILE_DIR / name).with_fields(
        sys_reserve_bytes=1024)
    limit = profile.tensor_pu_memory_bytes
    profile.validate_tensor_pu_bytes(limit)
    with pytest.raises(ValueError, match="TensorPU"):
        profile.validate_tensor_pu_bytes(limit + 1)
    legacy = profile.cluster_memory_bytes // profile.tensors_per_cluster
    assert legacy == limit + 1024 // profile.tensors_per_cluster
    if legacy > limit:
        with pytest.raises(ValueError, match="TensorPU"):
            profile.validate_tensor_pu_bytes(legacy)


def test_reserve_must_leave_a_positive_single_pu_capacity():
    """扣预留后单 PU 容量为零时必须在画像加载阶段拒绝。"""
    generic = load_profile(PROFILE_DIR / "generic.json")
    with pytest.raises(ValueError, match="单 TensorPU"):
        generic.with_fields(
            sys_reserve_bytes=generic.cluster_memory_bytes - 1)
    # 每簇 PU 数大于 1 时，剩余不足一个 PU 的份额同样拒绝。
    with pytest.raises(ValueError, match="单 TensorPU"):
        generic.with_fields(
            sys_reserve_bytes=generic.cluster_memory_bytes
            - generic.tensors_per_cluster + 1)


def test_to_pim_hardware_config_carries_the_optimization_switches():
    """编译请求的优化开关由画像派生入口显式传入，旧调用保持全开。"""
    from contracts.op_contract import PimOptimizations

    profile = load_profile(PROFILE_DIR / "generic.json")
    default = profile.to_pim_hardware_config(8, 4)
    assert default.optimizations == PimOptimizations()

    changed = profile.to_pim_hardware_config(
        8, 4, optimizations=PimOptimizations(cse=False, loop_tiling=False))
    assert changed.optimizations == PimOptimizations(
        cse=False, loop_tiling=False)
