"""成本精化的硬件画像入口：测量预算必须与放置 sidecar 同源。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from contracts.hardware_profile import load_profile
from contracts.op_contract import PimOptimizations


_PIM_ATTR_RE = re.compile(r'"?(pim\.[A-Za-z0-9_.-]+)"?\s*=\s*([^,}]+)')
_HEX_DIGITS = frozenset("0123456789abcdef")
# 放置 sidecar 的 schema 版本；5 起必须带完整编译契约与 A/B 编译路由。
SIDECAR_SCHEMA_VERSION = 5
# 请求级预算与 PIM MLIR 模块属性的对应。
_PIM_BUDGET_ATTRS = (
    ("pim.num-dpus", "num_dpus"),
    ("pim.num-tasklets", "num_tasklets"),
    ("pim.mram-bytes", "mram_bytes_per_dpu"),
    ("pim.wram-bytes", "wram_bytes_per_dpu"),
    ("pim.dma-align", "dma_align"),
    ("pim.target", "compiler_target"),
)
_PIM_FIELD_OF_ATTR = dict(_PIM_BUDGET_ATTRS)
# A 路必须写全六项；B 路只豁免 DPU/tasklet 编码，其余四项仍须核对。
_PIM_ROUTE_REQUIRED_ATTRS = {
    "a": tuple(name for name, _ in _PIM_BUDGET_ATTRS),
    "b": ("pim.mram-bytes", "pim.wram-bytes", "pim.dma-align", "pim.target"),
}
_PIM_ROUTE_OPTIONAL_ATTRS = {
    "a": (),
    "b": ("pim.num-dpus", "pim.num-tasklets"),
}
# 算子类型决定编译路由：GEMM 走 A 路，其余算子级节点走 B 路。
_PIM_ROUTE_OF_OP_TYPE = {"GEMM": "a"}


def sidecar_schema_version(placement: dict) -> int:
    """读取并校验放置 sidecar 的顶层 schema 版本。

    严格与旧产物的分流只看这个显式版本；缺版本或删契约块都不能自动降级。
    """
    version = placement.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError(
            f"放置 sidecar 的 schema 版本非法：{version!r}；"
            "无法判断是否必须提供完整编译契约。")
    if version > SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            f"放置 sidecar 的 schema 版本 {version} 高于当前导出器支持的 "
            f"{SIDECAR_SCHEMA_VERSION}；请重新导出产物。")
    return version


def _require_digest(value: Any, label: str) -> str:
    """内容摘要必须是 64 位小写十六进制且不是全零占位。"""
    if (not isinstance(value, str) or len(value) != 64
            or value == "0" * 64
            or any(ch not in _HEX_DIGITS for ch in value)):
        raise ValueError(
            f"{label} 必须是非零的 64 位十六进制内容摘要，收到 {value!r}。")
    return value


def _pim_module_attributes(text: str) -> Dict[str, Any]:
    """读 PIM MLIR 模块头的 pim.* 属性；没有属性头时返回空字典。"""
    header = re.search(r"module attributes \{([^}]*)\}", text)
    if header is None:
        return {}
    attributes: Dict[str, Any] = {}
    for name, raw in _PIM_ATTR_RE.findall(header.group(1)):
        value = raw.strip()
        if value.startswith('"') and value.endswith('"'):
            attributes[name] = value[1:-1]
            continue
        token = value.split(":", 1)[0].strip()
        try:
            attributes[name] = int(token)
        except ValueError:
            attributes[name] = token
    return attributes


def _verify_strategy_contract(
    placement: dict, compiler: dict, num_dpus: int,
) -> None:
    """策略、DPU 映射与硬件请求必须描述同一套编译预算。"""
    strict = sidecar_schema_version(placement) >= SIDECAR_SCHEMA_VERSION
    strategy = compiler.get("strategy")
    if not isinstance(strategy, dict) or not strategy:
        raise ValueError(
            "放置 sidecar 缺少 compiler.strategy 契约，无法核对 DPU 预算；"
            "请用当前导出器重新生成。")
    strategy_dpus = strategy.get("num_dpus")
    if (not isinstance(strategy_dpus, int) or isinstance(strategy_dpus, bool)
            or strategy_dpus <= 0):
        raise ValueError(
            f"放置 sidecar 的策略 num_dpus={strategy_dpus!r} 必须是正整数。")
    if strategy_dpus != num_dpus:
        raise ValueError(
            f"放置 sidecar 的策略 num_dpus={strategy_dpus} 与硬件 "
            f"num_dpus={num_dpus} 不一致，产物不是这套预算编出来的。")
    # 段数与 TP 宽度必须与 num_dpus 自洽；当前 schema 必须写全，旧格式
    # 只在字段存在时核对，保住既有兼容边界。
    stages = strategy.get("num_stages")
    tp_width = strategy.get("tp_width")
    if strict or stages is not None or tp_width is not None:
        for label, value in (("num_stages", stages), ("tp_width", tp_width)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ValueError(
                    f"放置 sidecar 的策略 {label}={value!r} 必须是正整数。")
        if num_dpus % stages:
            raise ValueError(
                f"放置 sidecar 的策略 num_stages={stages} 不能整除 "
                f"num_dpus={num_dpus}。")
        if tp_width != num_dpus // stages:
            raise ValueError(
                f"放置 sidecar 的策略 tp_width={tp_width} 与 "
                f"num_dpus={num_dpus}/num_stages={stages} 不一致；"
                "策略不是同一套图切分。")
    if "dpu_ids" not in strategy:
        if strict:
            raise ValueError(
                "当前 schema 的放置 sidecar 缺少策略 dpu_ids，无法证明预算"
                "覆盖全部 DPU；请用当前导出器重新生成。")
        return
    dpu_ids = strategy["dpu_ids"]
    ids = list(dpu_ids) if isinstance(dpu_ids, (list, tuple)) else None
    if (ids is None or len(ids) != num_dpus
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or value < 0 for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError(
            f"放置 sidecar 的策略 dpu_ids={dpu_ids!r} 与硬件 "
            f"num_dpus={num_dpus} 不一致。")


def _verify_profile_contract(
    profile, compiler: dict, hardware_block: dict,
) -> None:
    """侧车请求的容量、对齐、编译画像与目标必须来自当前画像。"""
    expected = {
        "mram_bytes_per_dpu": profile.cluster_memory_bytes,
        "wram_bytes_per_dpu": profile.wram_bytes_per_dpu,
        "dma_align": profile.dma_align,
    }
    for key, value in expected.items():
        if int(hardware_block.get(key, -1)) != value:
            raise ValueError(
                f"放置 sidecar 的 {key}={hardware_block.get(key)!r} "
                f"与画像 {value!r} 不一致，不能按同一套预算测量。")
    digest = compiler.get("compile_profile_sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError(
            "放置 sidecar 缺少非空编译画像摘要，产物不是这套预算编出来的。")
    if digest != profile.compile_profile_sha256():
        raise ValueError(
            "放置 sidecar 的编译画像摘要与 --profile 不一致："
            "产物不是这套预算编出来的。")
    if hardware_block.get("compiler_target") != profile.compiler_target:
        raise ValueError(
            "放置 sidecar 的编译目标 "
            f"{hardware_block.get('compiler_target')!r} 与画像 "
            f"{profile.compiler_target!r} 不一致；"
            "缺少该字段的旧产物需用新导出器重新生成。")


def _verify_dpu_mapping(profile, placement: dict, num_dpus: int) -> None:
    """所有后端核对数量/类型/范围；CEVA 另要求显式一对一物理核映射。"""
    mapping = placement.get("dpu_to_cluster")
    if mapping is None:
        if profile.pim_type == "CEVA-NPM":
            raise ValueError(
                "放置 sidecar 缺少 dpu_to_cluster：CEVA 精化要求每个逻辑 "
                "DPU 显式映射到一个物理核，禁止缺省换算测量；"
                "请用当前导出器重新生成。")
        return
    if not isinstance(mapping, (list, tuple)) or len(mapping) != num_dpus:
        raise ValueError(
            f"放置 sidecar 的 dpu_to_cluster 映射数量与硬件 "
            f"num_dpus={num_dpus} 不一致：{mapping!r}。")
    seen = set()
    for entry in mapping:
        if (not isinstance(entry, (list, tuple)) or len(entry) != 2
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 0 for value in entry)):
            raise ValueError(
                "放置 sidecar 的 dpu_to_cluster 每项必须是 "
                f"(device_id, cluster_id)：{entry!r}。")
        device_id, cluster_id = entry
        if (device_id >= profile.num_pim_devices
                or cluster_id >= profile.clusters_per_device):
            raise ValueError(
                f"放置 sidecar 的 dpu_to_cluster 映射 {entry!r} 越界：画像只有 "
                f"{profile.num_pim_devices} 台设备 × "
                f"{profile.clusters_per_device} 个 ClusterPU。")
        pair = (device_id, cluster_id)
        if profile.pim_type == "CEVA-NPM" and pair in seen:
            raise ValueError(
                "放置 sidecar 的 dpu_to_cluster 把多个逻辑 DPU 映射到同一个"
                f"物理核 {pair!r}；CEVA 编译契约要求一对一映射。")
        seen.add(pair)


def _resolve_pimir_path(
    raw_path: Any, placement_path: Optional[Path],
    search_dirs: Iterable[Any],
) -> Optional[Path]:
    """按原路径、sidecar 同级 pimir/、显式搜索目录依次找回 PIM MLIR。"""
    path = Path(str(raw_path))
    name = path.name
    candidates = [path]
    if name and placement_path is not None:
        candidates.append(Path(placement_path).resolve().parent / "pimir" / name)
    for directory in search_dirs:
        candidates.append(Path(str(directory)) / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _verify_pimir_contract(
    hardware_block: dict, placement: dict,
    placement_path: Optional[Path], search_dirs: Iterable[Any],
) -> None:
    """引用的 PIM MLIR 必须内容一致，且实际属性与编译请求同源。

    当前 schema 按每条引用的编译路由核对：A 路必须写全六项预算属性，
    B 路只豁免 DPU/tasklet 编码；显式旧 schema 保留“写了才核对”的原有
    兼容行为。
    """
    if not placement.get("requires_pimir"):
        return
    strict = sidecar_schema_version(placement) >= SIDECAR_SCHEMA_VERSION
    operators = placement.get("operators")
    if not isinstance(operators, dict):
        raise ValueError(
            "放置 sidecar 声明 requires_pimir，但 operators 不是对象，"
            "无法核对算子编译产物。")
    # 缓存只复用文件字节、摘要与模块属性；路由与预算属性逐引用核对，
    # 否则先出现的合法引用会为后续引用抵扣必需属性。
    resolved: Dict[str, tuple[str, Dict[str, Any]]] = {}
    referenced = 0
    for raw_op_id, entry in operators.items():
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("pimir_path")
        if not raw_path:
            continue
        referenced += 1
        path = _resolve_pimir_path(raw_path, placement_path, search_dirs)
        if path is None:
            raise ValueError(
                f"op{raw_op_id} 的 PIM MLIR 读不到：{raw_path}；"
                "缺少算子编译产物时不能按另一套成本测量。")
        declared = _require_digest(
            entry.get("pimir_sha256"), f"op{raw_op_id} 的 pimir_sha256")
        key = str(path.resolve())
        cached = resolved.get(key)
        if cached is not None:
            if cached[0] != declared:
                raise ValueError(
                    f"op{raw_op_id} 的 PIM MLIR 声明了与同文件其他算子不同的"
                    " pimir_sha256。")
            attributes = cached[1]
        else:
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise ValueError(
                    f"op{raw_op_id} 的 PIM MLIR 无法读取：{path}（{exc}）"
                ) from exc
            if actual != declared:
                raise ValueError(
                    f"op{raw_op_id} 的 PIM MLIR 哈希与 sidecar 声明不一致："
                    f"声明 {declared}，实际 {actual}")
            attributes = _pim_module_attributes(path.read_text())
            resolved[key] = (declared, attributes)
        if strict:
            route = entry.get("pimir_route")
            if route not in _PIM_ROUTE_REQUIRED_ATTRS:
                raise ValueError(
                    f"op{raw_op_id} 的 PIM MLIR 缺少合法编译路由 "
                    f"pimir_route={route!r}；当前 schema 必须声明 A/B 路由。")
            expected_route = _PIM_ROUTE_OF_OP_TYPE.get(
                entry.get("op_type"), "b")
            if route != expected_route:
                raise ValueError(
                    f"op{raw_op_id}（{entry.get('op_type')}）声明路由 "
                    f"{route!r} 与算子类型应走的 {expected_route!r} 路不一致。")
            required = _PIM_ROUTE_REQUIRED_ATTRS[route]
            missing = [name for name in required if name not in attributes]
            if missing:
                raise ValueError(
                    f"op{raw_op_id} 的 {route} 路 PIM MLIR 缺少预算属性 "
                    f"{missing[0]}；产物不属于同一套预算。")
            names = (*required, *(
                name for name in _PIM_ROUTE_OPTIONAL_ATTRS[route]
                if name in attributes))
        else:
            names = tuple(
                name for name, _ in _PIM_BUDGET_ATTRS
                if name in attributes)
        for name in names:
            expected = hardware_block.get(_PIM_FIELD_OF_ATTR[name])
            if attributes[name] != expected:
                raise ValueError(
                    f"op{raw_op_id} 的 PIM MLIR {name}={attributes[name]!r} "
                    f"与编译请求 {_PIM_FIELD_OF_ATTR[name]}={expected!r} "
                    "不一致；产物不属于同一套预算。")
    if referenced == 0:
        raise ValueError(
            "放置 sidecar 声明 requires_pimir，但没有任何算子引用 PIM MLIR；"
            "请用当前导出器重新生成。")


def verify_placement_contract(
    profile, placement: dict, *,
    placement_path: Optional[str | Path] = None,
    search_dirs: Iterable[Any] = (),
) -> tuple[int, int]:
    """核对放置 sidecar 的请求、PIM MLIR 与物理映射同属一套编译预算。

    返回请求的 `(num_dpus, num_tasklets)`。任何缺失、伪造或互相矛盾的
    字段都在读取本地形状与成本编译前拒绝。
    """
    if not isinstance(placement, dict):
        raise ValueError("放置 sidecar 顶层必须是 JSON 对象。")
    sidecar_schema_version(placement)
    compiler = placement.get("compiler")
    if not isinstance(compiler, dict) or not compiler:
        raise ValueError(
            "放置 sidecar 缺少 compiler 契约，不能按画像预算测量；"
            "请用当前导出器重新生成。")
    hardware_block = compiler.get("hardware")
    if not isinstance(hardware_block, dict) or not hardware_block:
        raise ValueError(
            "放置 sidecar 缺少 compiler.hardware 契约，不能按画像预算测量；"
            "请用当前导出器重新生成。")
    counts: Dict[str, int] = {}
    for key in ("num_dpus", "num_tasklets"):
        value = hardware_block.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"放置 sidecar 的 {key}={value!r} 必须是正整数，"
                "不能按默认预算测量。")
        counts[key] = value
    _verify_strategy_contract(placement, compiler, counts["num_dpus"])
    _verify_profile_contract(profile, compiler, hardware_block)
    _verify_dpu_mapping(profile, placement, counts["num_dpus"])
    _verify_pimir_contract(
        hardware_block, placement, placement_path, search_dirs)
    return counts["num_dpus"], counts["num_tasklets"]


def resolve_hardware(profile_path: str | Path, placement_path: str | Path | None):
    """返回 `(PIMHardwareConfig, HardwareProfile)`。

    没有画像时退回 `(None, None)`，成本提取沿用全局 `pim_options()`，旧命令
    行为不变。只有画像没有放置 sidecar 时按文档明确的单 DPU 默认测量；
    一旦给出放置 sidecar，就必须有完整的编译契约，任何缺失都在测量前拒绝。
    """
    if not profile_path:
        return None, None
    profile = load_profile(Path(profile_path))
    if not placement_path:
        return profile.to_pim_hardware_config(1, 4), profile

    placement = json.loads(Path(placement_path).read_text())
    num_dpus, num_tasklets = verify_placement_contract(
        profile, placement, placement_path=placement_path)
    # 优化开关必须与图/算子编译的侧车同源；旧侧车没有该块时按全开测量。
    raw_optimizations = (
        (placement.get("compiler") or {}).get("hardware") or {}
    ).get("optimizations")
    optimizations = (
        PimOptimizations() if raw_optimizations is None
        else PimOptimizations.from_payload(raw_optimizations))
    return profile.to_pim_hardware_config(
        num_dpus, num_tasklets, optimizations=optimizations), profile
