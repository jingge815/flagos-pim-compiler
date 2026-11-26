#!/usr/bin/env python3
"""跨三个仓库跑通 Llama2-7B 的完整链路，并核对每一段真的生效了。

链路与每段的载体：

    HuggingFace 权重 + config
      │
      ├─(A)─> GeneSim model_parser        → 图骨架 IR（semantic_role 标好投影身份）
      │
      ├─(A2)> refine_ir_with_flagtree     → 精化 pimir 成本 IR（仿真加载的那份，
      │                                      与图骨架同一次运行、同一套算子编号）
      │
      ├─(B)─> 图编译器 compile_llama2      → PIMTensorSpec（每个权重的切分与归属）
      │         策略由 --num-stages 决定
      │
      ├─(C)─> 算子编译器 FlagTree          → .so + pim mlir（真实分块由 WRAM 预算定）
      │         每种本地分片形状编一次
      │
      ├─(D)─> placement sidecar           → dpu_id / local_*_features /
      │                                      kernel_tile_n / pimir_path
      │
      └─(E)─> GeneSim 仿真                 → 照 pim mlir 生成 PIM trace，出代价

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/run_full_pipeline.py --num-stages 4

    # 只想快速验证接线、不跑完整仿真：
    python scripts/run_full_pipeline.py --num-stages 4 --skip-simulation

    # A/B 对照：同一份 sidecar，关掉本地分片形状再跑一次
    python scripts/run_full_pipeline.py --num-stages 4 --ab-compare

CEVA 后端（我方 GML/bin 走厂商 GC 与 CEVA 仿真）：

    bash scripts/run_ceva_pipeline.sh --run-name ours_e2e --decode-steps 10

    启动脚本自动加载 paths.json 中的编译环境。主流程导出 65 张子图并强制走
    FlagTree → 来源清单校验 → 工作区 run_all.sh --gml-bundle（GC、DDR、仿真、判定）。
    我方 bin 默认用 --token-check warn；需要逐 token 一致时显式传 strict。
    整轮约 10 小时，建议 nohup 后台运行。

每一步跑完都会核对产物，不满足就直接失败——这个脚本的用途是回答「链路通没通」，
所以任何一段静默退化都必须变成非零退出码。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesim_bridge.paths import (ceva_reference_dir, ceva_workspace, genesim_models_dir,
                                  genesim_root, llama2_7b_model_dir)

_REPO_ROOT = Path(__file__).resolve().parent.parent
# 默认走 GeneSim 的 CEVA 64 核 4e 画像：编译预算必须与画像一致，否则严格门禁会拒绝产物。
CEVA_MRAM_BYTES = 8_000_000_000
CEVA_SYS_RESERVE_BYTES = 0
CEVA_SIM_CONFIG = "ceva_64core_4e.yaml"
# DP 副本按物理核数选 CEVA 场景：G=2 × 64 DPU = 128 核配置（R8 多 batch 并行）。
CEVA_SIM_CONFIGS_BY_CORES = {64: CEVA_SIM_CONFIG, 128: "ceva_128core_4e.yaml"}
# DPU 数缺省也从这份画像推导（64）；与 export_pp_placement.py 同源。
DEFAULT_PROFILE = _REPO_ROOT / "contracts" / "profiles" / "ceva_64core_4e.json"
# 七个投影，缺一个都说明 IR 或匹配退化了。
_EXPECTED_ROLES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}

# 仿真配置 `model.ir_path` 指向的精化 IR 文件名（pimir 成本口径），由 (A2) 步产出。
_PIMIR_IR_NAME = "llama2_7b_pimir.ir"
# 精化的序列长度口径：与 docs/llama-2.md 第六步、现有产物、仿真 trace 一致。
_REFINE_SEQ_LEN = "128"


def _refine_ir_level() -> str:
    """没有 GPU 时精化走 B 路。

    pimir 级要真实执行 FlagGems 算子来测量成本，纯 CPU 机器上做不到，会在
    捕获 kernel 时直接报错。oplevel 是整算子 IR，只编译不执行，纯 CPU 可跑。
    """
    import torch

    if torch.cuda.is_available():
        return "pimir"
    print("    未检测到 GPU，精化改用 oplevel（只编译不执行）")
    return "oplevel"


class StepFailed(RuntimeError):
    """某一段没达到判据。"""


def parse_batch_list(text: str) -> tuple[int, ...]:
    """把逗号分隔的 batch/groups 档位解析成正整数元组（去重升序）。"""
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part.isdigit() or int(part) <= 0:
            raise ValueError(f"档位必须是正整数，收到 {text!r}")
        value = int(part)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"档位列表不能为空：{text!r}")
    return tuple(sorted(values))


def write_closed_batch_trace(source: dict, *, num_requests: int,
                             out_path: Path) -> Path:
    """把基准 trace 的首请求复制成等长 closed batch trace（JSON）。

    closed batch 要求全部请求等长：取首请求的形状复制 `num_requests` 份、
    到达时刻对齐。各 batch 档共用同一条，保证档间同指纹可比。
    """
    if num_requests <= 0:
        raise ValueError(f"closed batch 请求数必须为正：{num_requests}")
    template = source["requests"][0]
    requests = [dict(template, request_id=f"syn_closed_{i:06d}")
                for i in range(num_requests)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "description": (f"closed batch: 首请求 {template['request_id']} "
                        f"复制 {num_requests} 份"),
        "requests": requests,
    }, ensure_ascii=False))
    return out_path


def closed_trace_max_seq(source: dict) -> int:
    """closed trace 的容量口径序列长度：首请求 seq_len + output_length。

    KV 布局与容量表的 max_seq 必须与仿真 trace 同源，否则容量墙位对不上
    真实序列长度（round05 问题 5 的根因）。
    """
    template = source["requests"][0]
    seq_len = int(template["seq_len"])
    output_length = int(template.get("output_length", 1))
    if seq_len <= 0 or output_length <= 0:
        raise StepFailed(
            f"closed trace 首请求长度不合法：seq_len={seq_len}、"
            f"output_length={output_length}，无法推导容量口径 max_seq。")
    return seq_len + output_length


def decode_tile_m_by_batch(placement: Path) -> dict[int, int]:
    """逐 B 档取 decode 相位产物的 tile_m（同档多算子取最大）；旧格式返回空。"""
    if not Path(placement).is_file():
        return {}
    ops = json.loads(Path(placement).read_text()).get("operators", {})
    by_batch: dict[int, int] = {}
    for entry in ops.values():
        for key, phase in (entry.get("pimir_by_phase") or {}).items():
            if not (key.startswith("decode_b") and phase.get("tile_m")):
                continue
            batch = int(key[len("decode_b"):])
            by_batch[batch] = max(by_batch.get(batch, 0), int(phase["tile_m"]))
    return by_batch


@dataclass
class SweepRun:
    """一个 (batch, groups) 档的仿真产物。"""

    batch_size: int
    groups: int
    summary: Dict[str, Any]                    # summary.json
    batch_metrics: Optional[Dict[str, Any]]    # batch_metrics.json；B=1 老口径可为 None
    capacity_note: str = ""                    # 容量表摘录（可读文本，可选）
    pipeline_overlap: bool = False             # 该档是否开启 R8 流水重叠调度
    replicas: int = 1                          # 该档的 DP 副本数（真实副本产物 >1）
    fingerprint: str = ""                      # 产物内容指纹（batch_metrics+summary 摘要前 8 位）
    link_wait_total_s: float | None = None     # 全局链路等待（communication_metrics 的 total_queue_wait_time_s）
    kv_head_pin: bool = False                  # 该档是否开启执行侧 KV head 钉扎
    cores: int = 0                             # 该档物理核数（num_dpus × replicas）
    weight_reuse: bool = True                  # 该档是否保持 M 合并（R21-04 对照开关；False=逐槽产物）
    forward_steps: int = 0                     # 首请求 forward 步数（口径门禁：合批应 == output_length）
    output_length: int = 0                     # 首请求输出长度（与 forward_steps 同源）


def tier_capacity_status(results_dir: Path) -> tuple[bool, int, int]:
    """读单档 pim_capacity_metrics.json 的 fits 与溢出计数（R17-08）。

    用 `.get(…, 0)` 链读不存在的字段会把「数据缺失」折叠成「零溢出」，
    验证 silently 通过——字段缺失直接抛。返回
    (fits, overflow_tensor_pu_count, overflow_cluster_count)。
    """
    path = Path(results_dir) / "pim_capacity_metrics.json"
    if not path.is_file():
        raise StepFailed(f"容量核验缺档：{path}")
    data = json.loads(path.read_text())
    missing = [key for key in (
        "fits", "overflow_tensor_pu_count", "overflow_cluster_count")
        if key not in data]
    if missing:
        raise StepFailed(
            f"{path} 缺容量字段 {missing}：零溢出结论无效，拒绝默认成功")
    return (bool(data["fits"]), int(data["overflow_tensor_pu_count"]),
            int(data["overflow_cluster_count"]))


def assert_tier_capacity_no_overflow(results_dir: Path) -> None:
    """容量零溢出核验：非零溢出或 fits=false 即报警（硬失败）。"""
    fits, tensor_pu, cluster = tier_capacity_status(results_dir)
    if not fits or tensor_pu or cluster:
        raise StepFailed(
            f"{Path(results_dir).name} 容量溢出：fits={fits}、"
            f"tensor_pu 溢出 {tensor_pu} 台、cluster 溢出 {cluster} 个")


def _artifact_fingerprint(results_dir: Path) -> str:
    """档级产物的内容指纹：batch_metrics 与 summary 两个文件字节的 sha256 前 8 位。

    报告行据此自证消费的是哪一代盘上产物；产物目录被后续运行覆盖后指纹
    随之变化，「报告引用已消亡世代」才能被过期核验发现。缺文件返回空。
    """
    digest = hashlib.sha256()
    for name in ("batch_metrics.json", "summary.json"):
        path = Path(results_dir) / name
        if not path.is_file():
            return ""
        digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


def _sweep_thr(run: SweepRun) -> float:
    return float(run.summary.get("throughput_output_tokens_per_s", 0.0))


def _sweep_wbpt(metrics: dict) -> float:
    """逐组权重字节按生成 token 加总折算成档级 weight_bytes/token。"""
    weight = sum(float(g.get("weight_bytes", 0.0)) for g in metrics.get("groups", []))
    tokens = sum(int(g.get("generated_tokens", 0)) for g in metrics.get("groups", []))
    return weight / tokens if tokens else 0.0


def _sweep_decode_wbpt(metrics: dict) -> float | None:
    """档级 decode 相位 weight_bytes/token；老口径产物无相位桶时为 None。"""
    weight = sum(float(g.get("weight_bytes_decode", 0.0))
                 for g in metrics.get("groups", []))
    tokens = sum(int(g.get("decode_tokens", 0))
                 for g in metrics.get("groups", []))
    return weight / tokens if tokens else None


BATCH_METRICS_RESULT_SCHEMA = 3


def _caliber_findings(runs: list[SweepRun]) -> list[str]:
    """口径门禁（R21-03）：按产物内容判执行语义，不符即降级标注旧口径。

    三类判据（用户裁决）：`result_schema`（batch_metrics 的口径版本）、
    `forward_steps` 的 prefill 步数语义（合批一步 → B>1 时
    forward_steps == output_length；B=1 两口径同值，不构成判据）、归因
    字段集合（链路争用三字段）。判据不符不拒绝生成、显式降级：报告头
    警示 + 推荐未定（重扫前推荐身份无效）。
    """
    findings: list[str] = []
    for run in runs:
        metrics = run.batch_metrics
        if metrics is None:
            findings.append(
                f"B={run.batch_size},G={run.groups}: 无 batch_metrics（老口径产物）")
            continue
        schema = metrics.get("result_schema")
        if schema is not None and int(schema) != BATCH_METRICS_RESULT_SCHEMA:
            findings.append(
                f"B={run.batch_size},G={run.groups}: result_schema={schema}"
                f"（当前为 {BATCH_METRICS_RESULT_SCHEMA}）")
        groups = metrics.get("groups") or []
        if groups and not all(
                all(key in group for key in (
                    "link_wait_s", "intra_link_contention_s",
                    "inter_link_contention_s"))
                for group in groups):
            findings.append(
                f"B={run.batch_size},G={run.groups}: 归因字段集合缺链路争用列")
        # 逐槽执行策略（流水开/复用关）是当期合法语义：forward_steps 本就
        # 是 slots+output-1，不构成旧口径证据（round25 修正：否则 wr/ov 档
        # 被误降级）。
        deliberate_per_slot = (
            run.pipeline_overlap or not run.weight_reuse)
        if (run.batch_size > 1 and not deliberate_per_slot
                and run.forward_steps and run.output_length
                and run.forward_steps != run.output_length):
            findings.append(
                f"B={run.batch_size},G={run.groups}: forward_steps="
                f"{run.forward_steps}≠output_length={run.output_length}"
                "（prefill 逐槽旧口径）")
    return findings


def aggregate_batch_report(
    runs: list[SweepRun],
    *,
    tile_m_by_batch: dict[int, int] | None = None,
    capacity_wall_at: int | None = None,
    capacity_wall_source: str = "",
    capacity_wall_hits: list[tuple[int, int]] | None = None,
    coverage: str = "",
    sweep_root: str = "",
) -> str:
    """把逐档仿真产物聚成扫参报告：逐档表、时间构成守恒、瓶颈与推荐。

    产出由调用方落盘：专项扫参写 `report-<scope>.md`，canonical 的
    `report.md` 固定为全量扫参索引（round09 问题 4）。瓶颈与推荐是报告侧的
    解读：吞吐最优档取实测值，容量墙的触墙组合由扫参循环直接传入
    （`capacity_wall_hits`），B 墙位（`capacity_wall_at`）只收编译期折算或
    全 G 档触墙的运行时归纳，DP 估算只按 B 墙；界列与 wbpt 比值同口径
    （相对基准档），tile_m 逐档取 decode_b{B} 产物。`coverage`/`sweep_root`
    是机器可读的扫参范围与产物根（round09 问题 2），报告头嵌出供索引与
    过期核验消费。
    """
    tile_m_by_batch = tile_m_by_batch or {}

    # 基准档：B=1 优先；不含 B=1 的扫参（如 G 曲线）取最小 B 并注明。
    base_run = next((r for r in runs if r.batch_size == 1), None)
    base_note = ""
    if base_run is None:
        with_metrics = [r for r in runs if r.batch_metrics is not None]
        base_run = min(with_metrics, key=lambda r: r.batch_size) \
            if with_metrics else None
        if base_run is not None:
            base_note = f"（基准档取最小 B={base_run.batch_size}，扫参未含 B=1）"
    base_wbpt = None
    base_decode_wbpt = None
    if base_run is not None and base_run.batch_metrics is not None:
        base_wbpt = _sweep_wbpt(base_run.batch_metrics) or None
        base_decode_wbpt = _sweep_decode_wbpt(base_run.batch_metrics)

    lines = ["# batch 扫参汇总报告", ""]
    lines.append("同指纹口径：同 trace 种子、同编译产物、同硬件画像，档间差异只来自 batch/组数。")
    # 口径门禁（R21-03）：判据不符的产物显式降级——索引的「口径」列读
    # 头部「旧口径警示」行，旧口径产物不再被标成新口径；推荐段同步未定。
    caliber_findings = _caliber_findings(runs)
    if caliber_findings:
        lines.append(
            "> ⚠ 旧口径警示：下列档产物为旧执行语义（合批 prefill/链路归因/"
            "result_schema 之前），推荐未定（待重扫，登记条目见实施文档"
            "第八节 R9/R10）。")
        for finding in caliber_findings[:8]:
            lines.append(f">   - {finding}")
        if len(caliber_findings) > 8:
            lines.append(f">   - ……共 {len(caliber_findings)} 项")
    if coverage:
        lines.append(f"> 扫参范围：{coverage}")
    if sweep_root:
        lines.append(f"> 产物根：{sweep_root}")
    lines.append("")

    # 逐档主表：绝对字节列与相对比值列并列；界 1/min(B,tile_m(B)) 是比值，
    # 只与「相对基准」列同口径对照，不与绝对字节数直比。气泡占比直读
    # summary 的 pipeline 指标；DMA 占比与 zone 构成来自 batch_metrics。
    # TTFT 给两列：P95 含排队（全部请求的分布）与首组不含排队（首组无
    # 排队，跨 B 可比）；时间拆分按相位分桶（与权重字节分桶同构）。
    # 「复用」列（R21-04）：关 = 逐槽产物对照档（wr 后缀产物目录）。
    lines.append("## 逐档结果")
    lines.append("| B | G | 重叠 | 复用 | total_time_s | 输出 tokens/s | TTFT P95 含排队 | TTFT 首组 | TPOT P50/P95 | prefill 时间 s | decode 时间 s | decode 占比 | weight_bytes/token | decode wbpt | 相对基准 | 界 1/min(B,tile_m) | 组间重叠 | 气泡占比 | DMA 占比 | zone 搬运 W/A/K | 容量占用 | 产物指纹 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        s = run.summary
        wbpt = "缺失"
        decode_wbpt = "—"
        ratio = "—"
        overlap = "—"
        bubble = "—"
        dma_ratio = "—"
        zone_cell = "—"
        prefill_time = "—"
        decode_time = "—"
        decode_share = "—"
        first_ttft = "—"
        if run.batch_metrics is not None:
            value = _sweep_wbpt(run.batch_metrics)
            wbpt = f"{value:.0f}"
            decode_value = _sweep_decode_wbpt(run.batch_metrics)
            if decode_value is not None:
                decode_wbpt = f"{decode_value:.0f}"
            # 比值口径优先 decode 相位（验收判据的口径），老产物退回混合。
            ratio_base = (base_decode_wbpt if decode_value is not None
                          and base_decode_wbpt else base_wbpt)
            ratio_value = decode_value if decode_value is not None else value
            if ratio_base and ratio_value:
                ratio = f"{ratio_value / ratio_base:.3f}"
            groups = run.batch_metrics.get("groups", [])
            overlaps = [float(g.get("inter_group_overlap_s", 0.0))
                        for g in groups]
            if overlaps:
                overlap = f"{sum(overlaps) / len(overlaps):.2f}"
            # 计算/DMA 时间拆分与 zone 字节构成（R7 归因），缺字段标「—」。
            # DMA 占比分母是「计算+搬运」服务预算（逐算子累加，与逐组
            # 守恒断言同口径）；busy 并集是时间轴测度，多 PU 并行时远小于
            # 服务预算，作分母会出 200%+ 的占比。
            memory_time = sum(float(g.get("memory_time_s", 0.0))
                              for g in groups)
            service_time = sum(
                float(g.get("memory_time_s", 0.0))
                + float(g.get("compute_time_s", 0.0))
                for g in groups)
            if service_time > 0.0:
                dma_ratio = f"{100.0 * memory_time / service_time:.1f}%"
            # 相位时间拆分：prefill/decode 服务时延分桶 + decode 占比；
            # TTFT 不含排队取首组（首组无排队延迟）。
            prefill_sum = sum(float(g.get("prefill_time_s", 0.0))
                              for g in groups)
            decode_sum = sum(float(g.get("decode_time_s", 0.0))
                             for g in groups)
            if prefill_sum or decode_sum:
                prefill_time = f"{prefill_sum:.2f}"
                decode_time = f"{decode_sum:.2f}"
                if prefill_sum + decode_sum > 0.0:
                    decode_share = (
                        f"{100.0 * decode_sum / (prefill_sum + decode_sum):.1f}%")
            first_group = groups[0] if groups else {}
            if first_group.get("ttft_s") is not None:
                first_ttft = f"{float(first_group['ttft_s']):.3f}"
            zone_totals: dict[str, float] = {}
            has_zone = False
            for group in groups:
                zones = group.get("zone_bytes")
                if isinstance(zones, dict) and zones:
                    has_zone = True
                    for zone, size in zones.items():
                        zone_totals[zone] = (
                            zone_totals.get(zone, 0.0) + float(size))
            if has_zone:
                zone_sum = sum(zone_totals.values())
                zone_cell = "/".join(
                    (f"{100.0 * zone_totals.get(z, 0.0) / zone_sum:.1f}%"
                     if zone_sum > 0 else "—")
                    for z in ("weight", "activation", "kv"))
        bubble_value = s.get("pipeline_bubble_ratio")
        if bubble_value is not None:
            bubble = f"{100.0 * float(bubble_value):.1f}%"
        tile_m = tile_m_by_batch.get(run.batch_size)
        bound = "—" if tile_m is None else f"{1.0 / min(run.batch_size, tile_m):.3f}"
        lines.append(
            f"| {run.batch_size} | {run.groups} | {'开' if run.pipeline_overlap else '关'} "
            f"| {'开' if run.weight_reuse else '关'} "
            f"| {float(s.get('total_time_s', 0.0)):.2f} "
            f"| {_sweep_thr(run):.2f} "
            f"| {float(s.get('p95_ttft_s', 0.0)):.3f} "
            f"| {first_ttft} "
            f"| {float(s.get('p50_mean_tpot_s', 0.0)):.3f}/{float(s.get('p95_mean_tpot_s', 0.0)):.3f} "
            f"| {prefill_time} | {decode_time} | {decode_share} "
            f"| {wbpt} | {decode_wbpt} | {ratio} | {bound} | {overlap} "
            f"| {bubble} | {dma_ratio} | {zone_cell} "
            f"| {run.capacity_note or '—'} "
            f"| {run.fingerprint or '—'} |")
    if not runs:
        lines.append("（无成功档）")
    if base_note:
        lines.append(f"{base_note}；decode wbpt 为 decode 相位权重字节/decode token"
                     "（混合口径含逐槽 prefill 权重）")
    lines.append("")

    # 时间构成与守恒：wall = 纯忙时 + 组内争用 + 组间争用 + sync 余项。
    # 组内/组间争用按目标资源维度累计（PU 与链路同口径；等待期间目标
    # 资源被同组占用即组内、被其它组占用即组间——R21-01 目标资源口径）；
    # 四项在时间轴上互斥。链路份额单列（R17-02）：链路等待不再沉入
    # sync，组内/组间链路争用分列呈现，PU 份额 = 总额减链路份额。
    lines.append("## 时间构成归因（逐组，秒）")
    lines.append(
        "| B | 组 | wall | PU busy | 纯忙时 | 组内争用(PU) | 组内链路争用 | "
        "组间争用(PU) | 组间链路争用 | 同步间隔 | 组间重叠 | 守恒 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        if run.batch_metrics is None:
            lines.append(
                f"| {run.batch_size} | — | 缺失 batch_metrics（老口径档） "
                "| | | | | | | | | — |")
            continue
        for group in run.batch_metrics.get("groups", []):
            wall = float(group.get("wall_time_s", 0.0))
            if not any(k in group for k in (
                    "busy_pure_s", "intra_group_contention_s")):
                lines.append(
                    f"| {run.batch_size} | {group.get('request_index')} "
                    f"| {wall:.3f} | 旧口径产物（无四项守恒字段） "
                    "| | | | | | | | | — |")
                continue
            parts = [float(group.get(k, 0.0)) for k in
                     ("busy_pure_s", "intra_group_contention_s",
                      "inter_group_contention_s", "sync_gap_s")]
            residual = wall - sum(parts)
            ok = abs(residual) <= max(1e-6, 1e-3 * wall)
            group_overlap = group.get("inter_group_overlap_s")
            overlap_cell = (f"{float(group_overlap):.3f}"
                            if group_overlap is not None else "—")
            if "link_wait_s" in group:
                intra_link = f"{float(group.get('intra_link_contention_s', 0.0)):.3f}"
                inter_link = f"{float(group.get('inter_link_contention_s', 0.0)):.3f}"
                intra_pu = f"{max(0.0, parts[1] - float(group.get('intra_link_contention_s', 0.0))):.3f}"
                inter_pu = f"{max(0.0, parts[2] - float(group.get('inter_link_contention_s', 0.0))):.3f}"
            else:
                intra_pu, intra_link = f"{parts[1]:.3f}", "—"
                inter_pu, inter_link = f"{parts[2]:.3f}", "—"
            lines.append(
                f"| {run.batch_size} | {group.get('request_index')} | {wall:.3f} "
                f"| {float(group.get('pu_busy_union_s', 0.0)):.3f} "
                f"| {parts[0]:.3f} | {intra_pu} | {intra_link} | {inter_pu} "
                f"| {inter_link} | {parts[3]:.3f} "
                f"| {overlap_cell} "
                f"| {'通过' if ok else f'不守恒（残差 {residual:.3f}）'} |")
    lines.append("")

    # 机制性饱和判定（R17-07）：tile_m 达产物封顶值且不再随 B 增长，即
    # 「由内核上限证明的机制饱和」。末档没有更高档可对照：tile_m==封顶
    # 且 <B 即封顶证明（封顶值由更低档达到同一值给出）；压线档
    # （tile_m==B==封顶）需更高档同值确认摊薄界已停在本档。
    ordered = sorted({b for b in tile_m_by_batch if b in
                      {r.batch_size for r in runs}})
    valid_tiles = {
        b: t for b, t in tile_m_by_batch.items()
        if 0 < t <= b and b in {r.batch_size for r in runs}}
    tile_cap = max(valid_tiles.values(), default=0)
    saturated: list[int] = []
    for index, batch in enumerate(ordered):
        tile_m = tile_m_by_batch.get(batch) or 0
        if tile_m != tile_cap:
            continue
        next_tiles = [
            tile_m_by_batch.get(later) or 0
            for later in ordered[index + 1:]]
        if tile_m < batch or (next_tiles and max(next_tiles) <= tile_m):
            saturated.append(batch)

    # 瓶颈标注：触墙组合直接列出（扫参循环捕获的失败档），B 墙位只收
    # 编译期折算或全 G 档触墙的归纳——成功档不再被反推成命中档。
    lines.append("## 瓶颈归因")
    tile_m_max = max(tile_m_by_batch.values(), default=0)
    if tile_m_max > 0:
        lines.append(
            f"- 权重摊薄上限：逐档产物 tile_m 最大 {tile_m_max}"
            f"（内核 BLOCK_M 与 WRAM 预算共同决定），1/tile_m_max = "
            f"{1.0 / tile_m_max:.4f}；B 超过 {tile_m_max} 后每多一份 "
            f"ceil(B/{tile_m_max}) 重放，decode wbpt 不再下降")
    if capacity_wall_hits:
        combos = "、".join(f"(B={b}, G={g})" for b, g in capacity_wall_hits)
        lines.append(f"- 容量墙（运行时捕获）：触墙组合 {combos}；"
                     "KV 容量按组内槽数与并发组数计")
    if capacity_wall_at is not None:
        source = f"（{capacity_wall_source}）" if capacity_wall_source else ""
        lines.append(f"- 容量墙：B≥{capacity_wall_at} 触发{source}；"
                     "DP 估算只按 B 墙口径")
    # 链路争用（R17-02）：全局链路等待与逐组链路争用进瓶颈归因——修复前
    # 该等待被记成 sync 余项，瓶颈段对链路维度不可见。
    link_runs = [r for r in runs
                 if (r.link_wait_total_s or 0.0) > 0.0
                 or any((g.get("link_wait_s") or 0.0) > 0.0
                        for g in ((r.batch_metrics or {}).get("groups") or []))]
    if link_runs:
        total_link_wait = sum(r.link_wait_total_s or 0.0 for r in link_runs)
        max_inter = max(
            (float(g.get("inter_link_contention_s") or 0.0)
             for r in link_runs
             for g in ((r.batch_metrics or {}).get("groups") or [])),
            default=0.0)
        max_intra = max(
            (float(g.get("intra_link_contention_s") or 0.0)
             for r in link_runs
             for g in ((r.batch_metrics or {}).get("groups") or [])),
            default=0.0)
        lines.append(
            f"- 链路争用：全局链路等待合计 {total_link_wait:.1f}s"
            f"（逐组台账 Σlink_wait_s 同口径）；组间链路争用最大 "
            f"{max_inter:.1f}s、组内最大 {max_intra:.1f}s"
            "（传输就绪→派发的排队等待，与 PU 争用分开归因）")
    if saturated:
        tiles = {b: tile_m_by_batch.get(b) for b in saturated}
        lines.append(
            f"- 权重字节机制饱和（内核上限证明）：B∈{saturated} 的 tile_m "
            f"已达产物封顶 {tile_cap} 且不再随 B 增长（逐档 tile_m={tiles}），"
            "继续加 B 只线性放大重放份数，不再摊薄")
    # 观察窗内吞吐持平 ≠ 机制饱和：没有内核上限证明时如实区分措辞。
    if runs and not saturated:
        by_batch = {}
        for run in runs:
            by_batch.setdefault(run.batch_size, []).append(run)
        flat: list[int] = []
        ordered_thr = sorted(
            (b, max(_sweep_thr(r) for r in rs)) for b, rs in by_batch.items())
        for index, (batch, thr) in enumerate(ordered_thr):
            later = ordered_thr[index + 1:]
            if later and all(
                    abs(later_thr / thr - 1.0) <= 0.01
                    for _, later_thr in later):
                flat.append(batch)
        if flat:
            lines.append(
                f"- 观察窗内吞吐持平（观察口径）：B∈{flat} 更高档未扫或 "
                "tile_m 仍在增长，没有内核上限证明，持平只是观察窗内现象、"
                "不构成摊薄封顶结论")
    if capacity_wall_at is None and not saturated:
        lines.append("- 未见明确瓶颈（无容量墙输入、权重字节未机制饱和）")
    lines.append("")

    # 推荐：吞吐最优档与时延最优档（无外部 SLO，按 Pareto 口径给出）。
    # 时延最优按「首组 TTFT（不含排队）」比较：closed trace 全部请求同刻
    # 到达时，P95 含排队混入的是后续组的串行排队时间，跨 B 不可比。
    # 全墙终态（runs 为空）是合法输入：没有成功档可比，只保留墙位行。
    lines.append("## 推荐")
    if not runs:
        lines.append("- 无成功档：跳过吞吐/时延推荐")
    elif caliber_findings:
        # 口径门禁（R21-03）：旧执行语义产物的推荐身份无效——吞吐/时延
        # 最优值产生于旧口径计价，重扫前不构成验收依据，只保留墙位事实。
        lines.append("- 推荐未定（旧口径）：逐档产物未过口径门禁"
                     "（见报告头旧口径警示），重扫前最优配置不作为验收依据；"
                     "容量墙位置行不受影响")
    else:
        best = max(runs, key=_sweep_thr)

        def _first_ttft(run: SweepRun) -> float | None:
            if run.batch_metrics is None:
                return None
            groups = run.batch_metrics.get("groups") or []
            if not groups or groups[0].get("ttft_s") is None:
                return None
            return float(groups[0]["ttft_s"])

        with_first = [(r, _first_ttft(r)) for r in runs]
        comparable = [(r, t) for r, t in with_first if t is not None]
        if comparable:
            fastest_latency, latency_value = min(
                comparable, key=lambda item: item[1])
            latency_note = f"首组 TTFT（不含排队）{latency_value:.3f}s"
        else:
            fastest_latency = min(
                runs, key=lambda r: float(r.summary.get("p95_ttft_s", 0.0)))
            latency_note = (f"P95 TTFT {float(fastest_latency.summary.get('p95_ttft_s', 0.0)):.3f}s"
                            "（含排队，老口径产物无首组 TTFT）")
        lines.append(f"- 吞吐最优档：B={best.batch_size}, G={best.groups}"
                     f"（{_sweep_thr(best):.2f} 输出 tokens/s，"
                     f"重叠={'开' if best.pipeline_overlap else '关'}，"
                     f"复用={'开' if best.weight_reuse else '关'}，"
                     f"副本={best.replicas}，"
                     f"KV head 钉扎={'开' if best.kv_head_pin else '关'}，"
                     f"核数={best.cores if best.cores else '—'}，"
                     f"产物指纹={best.fingerprint or '未核验'}，"
                     "配置与产物见报告头「产物根」的同名目录，可直接执行）")
        lines.append(f"- 时延最优档：B={fastest_latency.batch_size}, G={fastest_latency.groups}"
                     f"（{latency_note}，"
                     f"重叠={'开' if fastest_latency.pipeline_overlap else '关'}，"
                     f"复用={'开' if fastest_latency.weight_reuse else '关'}，"
                     f"副本={fastest_latency.replicas}，"
                     f"产物指纹={fastest_latency.fingerprint or '未核验'}）；"
                     "甲方 SLO 未定，按吞吐-时延 Pareto 取点")
    if capacity_wall_at is not None:
        source = f"（{capacity_wall_source}）" if capacity_wall_source else ""
        lines.append(f"- 容量墙位置：B={capacity_wall_at}{source}；"
                     "超出扫参档位时为编译期容量折算的外推值")
    # G 驱动的墙只约束并发推荐：B 还没到 B 墙就触墙的组合是并发组数压满
    # 容量，不升格成 B 墙、不进 DP 折算。
    g_walls = [
        (b, g) for b, g in (capacity_wall_hits or [])
        if capacity_wall_at is None or b < capacity_wall_at]
    if g_walls:
        combos = "、".join(f"B={b} 在 G={g}" for b, g in g_walls)
        lines.append(f"- 并发组数受容量约束：{combos} 触墙；"
                     "G 档上限以触墙组合为界，不折算 B 墙")

    # DP 空间并行两态呈现：真实副本产物（--replicas 开启，逐档带 dp_replicas）
    # 按实测归因；仅解析产物时维持估算口径并如实标注开关未开启。
    dp_runs = [run for run in runs if run.replicas > 1]
    lines.append("")
    if dp_runs:
        lines.append("## DP 空间并行（实测：真实副本放置 + 按副本路由）")
        for run in sorted(dp_runs, key=lambda r: (r.batch_size, r.groups)):
            total_time = float(run.summary.get("total_time_s", 0.0))
            groups = (run.batch_metrics or {}).get("groups", [])
            by_replica: dict[int, dict[str, float]] = {}
            for group in groups:
                slot = by_replica.setdefault(int(group.get("replica", 0)), {
                    "tokens": 0.0, "weight": 0.0, "kv": 0.0})
                slot["tokens"] += float(group.get("generated_tokens", 0.0))
                slot["weight"] += float(group.get("weight_bytes", 0.0))
                slot["kv"] += float(
                    (group.get("zone_bytes") or {}).get("kv", 0.0))
            per_replica = " / ".join(
                f"{(stats['tokens'] / total_time if total_time else 0.0):.2f}"
                for _, stats in sorted(by_replica.items()))
            kv_note = " / ".join(
                f"{stats['kv']:.3e}B" for _, stats in sorted(by_replica.items()))
            # 静态副本量与动态搬运分开（R17-04）：静态 = sidecar 声明的每副
            # 本纯权重/内核 tile（两份物理 DPU 各持一份权重）；动态 =
            # group.weight_bytes 的逐相位累计搬运，两者相差数量级、不得混写。
            static = (run.batch_metrics or {}).get("dp_replicas_static") or {}
            static_note = "；".join(
                f"静态{label} {float(value):.3e}B"
                for label, value in (
                    ("每副本纯权重复制", static.get("weight_bytes_per_replica")),
                    ("每副本内核 tile", static.get("kernel_tile_bytes_per_replica")))
                if value is not None)
            if static_note:
                static_note += f"（{run.replicas} 份物理 DPU 各持一份）"
            else:
                static_note = "静态副本字节：产物缺 dp_replicas_static"
            dynamic = sum(stats["weight"] for stats in by_replica.values())
            lines.append(
                f"- B={run.batch_size}, G={run.groups}, 副本数 "
                f"{run.replicas}：逐副本吞吐 {per_replica} tokens/s，"
                f"{static_note}；动态权重搬运（全生命周期逐相位累计，"
                f"非权重复制量）合计 {dynamic:.3e}B，"
                f"KV 按副本划分（逐副本 KV 搬运 {kv_note}）")
    else:
        lines.append("## DP 空间并行（解析估算，未校准；--replicas 未开启）")
        if not runs:
            lines.append("- 无成功档，无吞吐锚点，DP 估算略")
        elif capacity_wall_at is not None:
            for g in (2, 4):
                feasible = g * best.batch_size <= capacity_wall_at
                estimate = f"{g * _sweep_thr(best):.2f} tokens/s" if feasible \
                    else f"触容量墙（需 B≤{capacity_wall_at // g}）"
                lines.append(f"- {g} 副本：权重复制 ×{g}，吞吐上限 {estimate}")
        else:
            lines.append("- 缺容量墙输入，DP 估算略")

    # 抢占解析项：来自 batch_metrics 的 analytical 估算（带宽未配置时值为空）。
    # 分子是一次换出的整组 KV（槽数 × 单槽峰值），不随并发组数放大。
    lines.append("")
    lines.append("## 抢占代价（解析估算，不进事件引擎）")
    preempts = [r for r in runs
                if r.batch_metrics and r.batch_metrics.get("preemption_estimate")]
    if preempts:
        # 按 (B, 副本) 去重：换出分子是一次换出的整组 KV，与 G 档和重叠态
        # 都无关（解析估算不进事件引擎），去重键必须与行文案渲染的维度集
        # 一致——键含重叠态而成对报告文案不含时，两态渲染出逐字相同的两行
        # （round09 问题 4 的成对残留，round11 问题 1）。
        seen_preempt: set[tuple[int, int]] = set()
        for run in preempts:
            key = (run.batch_size, run.replicas)
            if key in seen_preempt:
                continue
            seen_preempt.add(key)
            est = run.batch_metrics["preemption_estimate"]
            # 一次抢占 = 换出 + 换入 + 切换常数，分项与总值分别给出
            # （R17-05：换入按 HOST→PIM 带独立带宽，缺失项不并入总值）。
            parts = []
            for name, key in (("换出", "swap_out_s"), ("换入", "swap_in_s"),
                              ("切换", "switch_cost_s")):
                item = est.get(key)
                parts.append(
                    f"{name} {float(item):.3f}s" if item is not None
                    else f"{name} 缺带宽")
            estimate = est.get("estimate_s")
            value = (f"总 {float(estimate):.3f}s（{' + '.join(parts)}）"
                     if estimate is not None
                     else f"{' + '.join(parts)}，总代价缺方向带宽未给出")
            slots = est.get("swap_slots")
            swap_note = (f"一次换出/换入各 {slots} 槽 KV "
                         f"{float(est.get('kv_migration_bytes_per_swap', 0.0)):.3e}B"
                         if slots else "口径：一次换出+换入整组 KV")
            replica_note = (f"（副本 ×{run.replicas}）" if run.replicas > 1 else "")
            lines.append(f"- B={run.batch_size}{replica_note}: {value}"
                         f"（{swap_note}，mode={est.get('mode', 'analytical')}，"
                         "calibrated=false）")
    else:
        lines.append("- 本轮产物未含抢占解析项")

    # R8 流水重叠两态对比：--pipeline-overlap 的成对扫参产出同一 (B,G) 的
    # 开/关两行（开=逐槽 micro-batch 流水策略，关=合批 prefill 策略），这里
    # 并列吞吐与气泡占比；收益只表述为吞吐改善/步间重叠生效（表述纪律）。
    lines.append("")
    lines.append("## 流水重叠对比（开/关）")
    pairs: dict[tuple[int, int], dict[bool, SweepRun]] = {}
    for run in runs:
        pairs.setdefault((run.batch_size, run.groups), {})[run.pipeline_overlap] = run
    paired = {key: both for key, both in pairs.items() if len(both) == 2}
    if paired:
        lines.append("| B | G | 吞吐（关 → 开） | 吞吐提升 | 气泡占比（关 → 开） |")
        lines.append("|---|---|---|---|---|")
        for (batch, groups), both in sorted(paired.items()):
            off_run, on_run = both[False], both[True]
            off_bubble = off_run.summary.get("pipeline_bubble_ratio")
            on_bubble = on_run.summary.get("pipeline_bubble_ratio")
            bubble_cell = (
                f"{100.0 * float(off_bubble):.1f}% → {100.0 * float(on_bubble):.1f}%"
                if off_bubble is not None and on_bubble is not None else "—")
            gain = _sweep_thr(on_run) / _sweep_thr(off_run) if _sweep_thr(off_run) else float("inf")
            lines.append(
                f"| {batch} | {groups} "
                f"| {_sweep_thr(off_run):.2f} → {_sweep_thr(on_run):.2f} "
                f"| {100.0 * (gain - 1.0):+.1f}% "
                f"| {bubble_cell} |")
    else:
        lines.append("- 仅单态（未开启 --pipeline-overlap 成对扫参），无开/关对比")

    # R21-04 权重复用固定 B 对照：--weight-reuse 的成对扫参产出同一
    # (B,G) 的复用开（合批产物）/关（逐槽产物）两行，这里并列 decode 相位
    # 权重字节/token——同一 B/trace/资源下差值即权重复用收益本身，与
    # batch 维度的组大小收益严格分离。
    lines.append("")
    lines.append("## 权重复用对比（开/关，固定 B）")
    reuse_pairs: dict[tuple[int, int], dict[bool, SweepRun]] = {}
    for run in runs:
        if run.pipeline_overlap:
            continue  # 流水开档不是复用对照对象，两开关各自成对。
        reuse_pairs.setdefault(
            (run.batch_size, run.groups), {})[run.weight_reuse] = run
    reuse_paired = {key: both for key, both in reuse_pairs.items()
                    if len(both) == 2}
    if reuse_paired:
        lines.append("| B | G | weight_bytes/token（关 → 开） | decode wbpt（关 → 开） | 吞吐（关 → 开） |")
        lines.append("|---|---|---|---|---|")
        for (batch, groups), both in sorted(reuse_paired.items()):
            off_run, on_run = both[False], both[True]

            def _wbpt_cell(run: SweepRun) -> str:
                value = _sweep_wbpt(run.batch_metrics or {})
                return f"{value:.0f}" if value else "缺失"

            def _decode_wbpt_cell(run: SweepRun) -> str:
                value = _sweep_decode_wbpt(run.batch_metrics or {})
                return f"{value:.0f}" if value else "—"

            lines.append(
                f"| {batch} | {groups} "
                f"| {_wbpt_cell(off_run)} → {_wbpt_cell(on_run)} "
                f"| {_decode_wbpt_cell(off_run)} → {_decode_wbpt_cell(on_run)} "
                f"| {_sweep_thr(off_run):.2f} → {_sweep_thr(on_run):.2f} |")
    else:
        lines.append("- 仅单态（未开启 --weight-reuse 成对对照），无开/关对比")
    return "\n".join(lines)


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: Optional[dict] = None) -> None:
    """跑一条命令，输出落到 log_path；失败时把尾部贴出来再抛。"""
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    print(f"      日志: {log_path}")
    with log_path.open("w") as handle:
        proc = subprocess.run(
            cmd, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise StepFailed(
            f"命令失败（exit {proc.returncode}）：{' '.join(str(c) for c in cmd)}\n"
            f"日志尾部：\n{tail}"
        )


def step_a_model_ir(genesim: Path, model_dir: Path, log_dir: Path,
                    *, out_path: Path | None = None) -> Path:
    """(A) 从 HF config 生成 GeneSim 图骨架，核对投影身份标全了。

    `out_path` 给了就写到指定位置（调优候选目录），缺省仍是共享的 models/。
    """
    print("\n[A] HuggingFace config → GeneSim 图骨架 IR")
    ir_path = Path(out_path) if out_path else genesim / "models" / "llama2_7b.ir"
    _run(
        ["python", "scripts/model_parser.py",
         "--model_name", str(model_dir), "--output", str(ir_path)],
        cwd=genesim, log_path=log_dir / "a_model_parser.log",
    )

    ir = json.loads(ir_path.read_text())
    gemms = [op for op in ir["operators"] if op["op_type"] == "GEMM"]
    roles = {op.get("semantic_role", "") for op in gemms}
    missing = _EXPECTED_ROLES - roles
    if missing:
        raise StepFailed(
            f"图骨架里缺这些投影的 semantic_role: {sorted(missing)}。"
            "放置导出按语义标签匹配，缺了就无法确定 GEMM 身份。"
        )
    unlabeled = [op["op_id"] for op in gemms if not op.get("semantic_role")]
    if unlabeled:
        raise StepFailed(f"{len(unlabeled)} 个 GEMM 没有 semantic_role，如 {unlabeled[:5]}")
    print(f"    算子 {len(ir['operators'])} 个，GEMM {len(gemms)} 个，七种投影身份齐全")
    return ir_path


def _ir_op_ids(ir_path: Path) -> list[int]:
    """读出 IR 里按出现顺序的算子编号序列。"""
    ir = json.loads(ir_path.read_text())
    return [int(op["op_id"]) for op in ir["operators"]]


def step_a2_refine_pimir_ir(
    genesim: Path, ir_path: Path, log_dir: Path, placement: Path | None = None,
) -> Path:
    """(A2) 从刚生成的图骨架 IR 精化出仿真要加载的 pimir 成本 IR。

    仿真配置加载的 `models/llama2_7b_pimir.ir` 历来是手动跑
    `refine_ir_with_flagtree.py` 的产物：图骨架改了结构（"支持全部算子"给
    注意力补了算子级节点）而没重跑精化时，sidecar 按新编号、仿真 IR 还是旧
    编号，调度器按 op_id 查不到 pimir_path，注意力算子整批退回手写模板
    （2026-10-07 的 6144 个模板 trace 就是这么来的）。放进主流程后，图骨架
    与精化 IR 永远是同一次运行里的同一套编号。

    已存在且编号序列与图骨架一致时跳过重新精化：refine 只回填成本、不改
    结构，结构一致就是同一份。注意成本口径不在比对范围——工具链重编后
    想强制重算成本，删掉这份文件再跑即可。
    """
    print("\n[A2] 图骨架 IR → 精化 pimir 成本 IR（仿真加载的那份）")
    pimir_ir = genesim / "models" / _PIMIR_IR_NAME
    fusion_requested = False
    if placement is not None and Path(placement).is_file():
        placement_data = json.loads(Path(placement).read_text())
        fusion_requested = bool(placement_data.get("fused_activations"))
    # sidecar 按实际档位命名：无 GPU 时走 oplevel，不能覆盖 pimir 那份。
    ir_level = _refine_ir_level()
    sidecar = genesim / "models" / f"llama2_7b_{ir_level}_extensions.json"
    # 跳过还要求本档位的 sidecar 在且自报档位相符：换了档位（有无 GPU）就重新精化。
    sidecar_ok = (sidecar.is_file()
                  and json.loads(sidecar.read_text()).get("ir_level") == ir_level)
    if (pimir_ir.is_file() and _ir_op_ids(pimir_ir) == _ir_op_ids(ir_path)
            and not fusion_requested and sidecar_ok):
        print(f"    精化 IR 与图骨架同套编号（{pimir_ir.name}），{ir_level} sidecar 在，跳过重新精化")
        return pimir_ir
    cmd = [
        "python", "scripts/refine_ir_with_flagtree.py",
        "--ir", str(ir_path), "--out-ir", str(pimir_ir),
        "--sidecar", str(sidecar),
        "--seq-len", _REFINE_SEQ_LEN, "--ir-level", ir_level,
    ]
    # 放置 sidecar 里带图编译器的融合清单。没有它就按未融合算，会把已被吸收的
    # 激活再计一遍——所以必须在放置导出之后跑，并且把这份 sidecar 传进来。
    if placement is not None and Path(placement).is_file():
        # 只取融合清单，不动形状口径：这份精化 IR 按模型级全局形状测，
        # 而融合与否是图编译器的决策，两种口径都要带上。
        cmd += ["--fusion-from", str(placement)]
    _run(cmd, cwd=genesim, log_path=log_dir / "a2_refine_pimir.log")
    if _ir_op_ids(pimir_ir) != _ir_op_ids(ir_path):
        raise StepFailed(
            f"精化 IR 的算子编号与图骨架不一致：{pimir_ir} vs {ir_path}。"
            "refine 应该只回填成本、不改结构，请检查 refine_ir_with_flagtree.py。")
    print(f"    精化完成：{pimir_ir.name}"
          f"（{len(_ir_op_ids(pimir_ir))} 个算子，编号与图骨架一致）")
    return pimir_ir


def sim_config_for_cores(physical_dpus: int) -> str:
    """按物理核数选 CEVA 场景配置；没有对应配置时直接抛。"""
    config = CEVA_SIM_CONFIGS_BY_CORES.get(physical_dpus)
    if config is None:
        raise StepFailed(
            f"物理 DPU 数 {physical_dpus} 没有对应的 CEVA 场景配置"
            f"（可用核数：{sorted(CEVA_SIM_CONFIGS_BY_CORES)}）；DP 副本形态"
            "当前只覆盖 64/128 核。")
    return config


def step_zero_pu_mapping(
    genesim: Path, ir_path: Path, num_stages: int, num_dpus: int, log_dir: Path,
    *, replicas: int = 1, sim_config: str = CEVA_SIM_CONFIG,
) -> Path:
    """(0) GeneSim 给出固定 PU 映射，核对逐段的 Cluster 归属合理。

    这一段方向与其余三段相反：不是"编译器算完告诉 GeneSim"，而是"GeneSim 定下
    PU 映射、约束编译器"。产物是 PartitionPlan，由编译器侧转成 ShardStrategy。
    `num_dpus` 是每副本 DPU 数；DP 副本 >1 时方案的 dpu_to_cluster 覆盖全部
    物理 DPU（num_dpus × replicas）。
    """
    print("\n[0] GeneSim 固定 PU 映射 → PartitionPlan")
    tp_width = num_dpus // num_stages
    physical = num_dpus * replicas
    stem = f"llama2_7b_tp{tp_width}_pp{num_stages}_plan.json"
    if replicas > 1:
        stem = f"llama2_7b_tp{tp_width}_pp{num_stages}_dp{replicas}_plan.json"
    plan_path = genesim / "models" / stem
    mapping_args = [] if replicas == 1 else ["--replicas", str(replicas)]
    _run(
        ["python", "scripts/export_fixed_pu_mapping.py",
         "--num-stages", str(num_stages), "--num-dpus", str(num_dpus),
         "--config", f"conf/{sim_config}",
         "--ir", str(ir_path), "--out", str(plan_path), *mapping_args],
        cwd=genesim, log_path=log_dir / "0_pu_mapping.log",
    )

    plan = json.loads(plan_path.read_text())
    mapping = plan.get("dpu_to_cluster")
    if not mapping or len(mapping) != physical:
        raise StepFailed(
            f"方案里的 dpu_to_cluster 应有 {physical} 项（{num_dpus} DPU × "
            f"{replicas} 副本），实际 {len(mapping) if mapping else 0} 项"
        )
    if plan.get("num_stages") != num_stages or plan.get("num_dpus") != num_dpus:
        raise StepFailed(
            f"方案的段数/DPU 数与请求不一致：{plan.get('num_stages')}/{plan.get('num_dpus')}"
        )
    # 报告段内是否走快链路——同 Cluster 是 512 GB/s，跨 Cluster 是 128 GB/s。
    for stage in range(num_stages):
        members = [tuple(e) for e in mapping[stage * tp_width : (stage + 1) * tp_width]]
        shared = len(set(members)) == 1
        print(f"    stage{stage}: {members} "
              f"({'同 Cluster 快链路' if shared else '跨 Cluster'})")
    if replicas > 1:
        print(f"    DP 副本 ×{replicas}：副本 g 的物理 DPU = "
              f"[{num_dpus}*g, {num_dpus}*(g+1))")
    print(f"    方案: {plan_path.name}（source={plan.get('source', '未标注')}）")
    return plan_path


def check_sidecar_entries(ops: Dict[str, Dict[str, Any]]) -> tuple[int, int]:
    """按条目类别分流核对 sidecar，返回 (GEMM 条数, 算子级条数)。

    sidecar 里两类条目的契约不同，用一套谓词查会把好的判成坏的：

      GEMM（A 路）   按图切分归属到各台 DPU，所以有 `shards` / `semantic_role`
                     / `kernel_tile_n`，pim mlir 是分块 GEMM
      算子级（B 路）  不做张量并行切分，`shards` 是空列表、没有投影身份，
                     pim mlir 是整算子级的相位链或单相 op

    两类共有的判据只有一条：`pimir_path` 必须在。缺了就说明这个算子的原语没进
    GeneSim，仿真会静默退回手写模板——正是本脚本要防的退化。
    """
    from genesim_bridge.op_classify import MNEMONIC_OF

    # GEMM 条目：图切分那一段的四项证据，逐项都要在。
    gemm_checks = {
        "shards（图切分归属，每台参与 DPU 各一项）": lambda e: bool(e.get("shards")),
        "shards[*].local_in/out_features（本地分片形状）": lambda e: all(
            s.get("local_in_features") and s.get("local_out_features")
            for s in e.get("shards", [])
        ),
        "semantic_role（投影身份）": lambda e: e.get("semantic_role") in _EXPECTED_ROLES,
        "kernel_tile_n（算子编译器实测分块）": lambda e: (e.get("kernel_tile_n") or 0) > 0,
    }
    # 算子级条目：op_type 要是 mnemonic 单表认得的名字，否则它在 GeneSim 侧
    # 没有原语落点，得报错而不是当普通条目放过。
    bpath_checks = {
        "op_type（mnemonic 单表认得的算子名）":
            lambda e: e.get("op_type") in MNEMONIC_OF,
    }
    common_checks = {
        "pimir_path（pim mlir 落盘路径）": lambda e: bool(e.get("pimir_path")),
    }

    # 有 shards 列表且非空 = 走过图切分的 GEMM；空列表 = 算子级节点。
    gemm_ops = {k: e for k, e in ops.items() if e.get("shards")}
    bpath_ops = {k: e for k, e in ops.items() if not e.get("shards")}

    for label, entries in (("GEMM", gemm_ops), ("算子级", bpath_ops)):
        checks = dict(common_checks)
        checks.update(gemm_checks if label == "GEMM" else bpath_checks)
        for field_label, predicate in checks.items():
            bad = [op_id for op_id, entry in entries.items()
                   if not predicate(entry)]
            if bad:
                raise StepFailed(
                    f"{len(bad)} 个{label}算子缺 {field_label}，如 op{bad[:5]}"
                )
    return len(gemm_ops), len(bpath_ops)


def fused_kernels_missing_act_spec(sidecar: Dict[str, Any]) -> list[str]:
    """融合清单非空时，折激活的投影 kernel 必须带 act_spec。

    旧产物把同形状的 gate 与 up 编成同一份不带激活的 kernel，仿真时延取自
    那份 trace，激活一分不付。
    """
    from genesim_bridge.ir_cost import has_fused_act_spec

    fused_roles = {entry["role"] for entry in sidecar.get("fused_activations") or []}
    if not fused_roles:
        return []
    missing = []
    seen = set()
    for entry in sidecar.get("operators", {}).values():
        if entry.get("semantic_role") not in fused_roles:
            continue
        path = entry.get("pimir_path")
        if not path or path in seen or not Path(path).is_file():
            continue
        seen.add(path)
        if not has_fused_act_spec(Path(path).read_text()):
            missing.append(path)
    return missing


def step_bcd_export(
    num_stages: int, num_dpus: int, log_dir: Path,
    *, plan_path: Optional[Path] = None, batch_sizes: tuple[int, ...] = (1,),
    max_seq: Optional[int] = None, replicas: int = 1, kv_head_pin: bool = False,
    per_slot_prefill: bool = False,
) -> Path:
    """(B)(C)(D) 图编译 + 算子编译 + 导出 sidecar，核对各类字段都写了。

    `plan_path` 给了就按 GeneSim 的方案编译（段数由方案决定），否则用 --num-stages。
    `batch_sizes` 非单档 1 时把档列表透传给导出脚本（逐相位逐档产物）。
    `max_seq` 给了就透传给导出脚本作 KV 布局与容量表的序列长度口径
    （扫参必须与仿真 trace 同源；经典单档路径不传、沿用导出脚本缺省）。
    `replicas`/`kv_head_pin` 透传 DP 副本与执行侧 KV head 分片开关（缺省
    关闭，经典路径产物不变）。`per_slot_prefill` 把逐槽执行策略传给图编译
    （R25-03：额外导出 [1,S] 逐槽图与 B 份槽位平移计划，缺省合批）。
    """
    print("\n[B+C+D] 图编译 → 算子编译（真实分块）→ placement sidecar")
    strategy_args = (
        ["--partition-plan", str(plan_path)] if plan_path is not None
        else ["--num-stages", str(num_stages), "--num-dpus", str(num_dpus)]
    )
    batch_args = ([] if batch_sizes == (1,)
                  else ["--batch", ",".join(str(b) for b in batch_sizes)])
    max_seq_args = ([] if max_seq is None else ["--max-seq", str(max_seq)])
    replica_args = ([] if replicas == 1 else ["--replicas", str(replicas)])
    pin_args = ["--kv-head-pin"] if kv_head_pin else []
    slot_args = ["--per-slot-prefill"] if per_slot_prefill else []
    _run(
        ["python", "scripts/export_pp_placement.py",
         *strategy_args, "--measure-kernel-tiles",
         "--mram-bytes", str(CEVA_MRAM_BYTES),
         "--sys-reserve-bytes", str(CEVA_SYS_RESERVE_BYTES),
         *batch_args, *max_seq_args, *replica_args, *pin_args, *slot_args],
        cwd=_REPO_ROOT, log_path=log_dir / "bcd_export.log",
    )

    models_dir = genesim_models_dir()
    tp_width = num_dpus // num_stages
    name = f"tp{tp_width}_pp{num_stages}"
    if replicas > 1:
        name += f"_dp{replicas}"
    sidecar_path = models_dir / f"llama2_7b_{name}_placement.json"
    if not sidecar_path.is_file():
        raise StepFailed(f"没找到 sidecar：{sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    ops = sidecar["operators"]
    if not ops:
        raise StepFailed("sidecar 里没有任何放置结果")

    # 本步骤用 --measure-kernel-tiles 编了算子，所以 sidecar 必须自报带产物。
    # GeneSim 靠这个字段自动进严格模式；缺了就退回"看配置文件写没写对"，也就是
    # 本脚本要防的那种静默退化。
    if not sidecar.get("requires_pimir"):
        raise StepFailed(
            "sidecar 没有 requires_pimir=true。带算子编译产物的导出必须自报，"
            "否则 GeneSim 不会自动进严格模式，pim mlir 读不到会静默退回手写模板。"
        )

    gemm_count, bpath_count = check_sidecar_entries(ops)
    print(f"    条目分两类核对：GEMM {gemm_count} 条、算子级 {bpath_count} 条")

    # 扫参导出必须带上逐相位逐档产物：缺档的 B 在仿真侧查不到 decode_b{B}
    # 或合批 prefill 的 prefill_b{B}，会按契约直接抛错——在这里先拦下，
    # 省掉一轮无效编译后的失败仿真。
    if batch_sizes != (1,):
        phase_keys: set[str] = set()
        for entry in ops.values():
            phase_keys |= set((entry.get("pimir_by_phase") or {}).keys())
        missing = [
            key for b in batch_sizes
            for key in (f"decode_b{b}", f"prefill_b{b}")
            if key not in phase_keys]
        if missing or "prefill" not in phase_keys:
            raise StepFailed(
                f"sidecar 缺逐相位产物：{missing or ['prefill']}；"
                "算子编译没有按 --batch 档列表出产物，重跑本步骤。"
            )

    pimir_files = {entry["pimir_path"] for entry in ops.values()}
    absent = [p for p in pimir_files if not Path(p).is_file()]
    if absent:
        raise StepFailed(f"pimir_path 指向的文件不存在：{absent[:3]}")

    # 内容哈希要和磁盘上的文件对得上：sidecar 记的是绝对路径，如果缓存被重建过、
    # 而 sidecar 还是旧的，路径可能存在但内容已经变了。哈希不符说明这份 sidecar
    # 与当前的算子编译产物不是一套。
    #
    # pimir_sha256 是后加的字段（见 docs/genesimsupporTp-20260905.md 第五之二节）。
    # 本步骤自己产出 sidecar，所以正常情况下一定有；缺了说明是早于该字段导出的旧
    # 产物，此时提示一句并跳过这一项核对——报"缺 pimir_sha256"信息量太低，看起来
    # 像 bug 而不是"重新导出即可"。注意只跳过哈希核对，后面的分块打印和
    # dpu_to_cluster 核对照常执行。
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        print(
            f"    注意: {len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 "
            "pimir_sha256（早于该字段导出的旧产物），跳过内容一致性核对；"
            "重新跑一次本脚本即可补上。"
        )
    else:
        stale = stale_pimir_entries(ops)
        if stale:
            raise StepFailed(_stale_sidecar_message(stale))

    # 分块是 GEMM 才有的字段（B 路算子级节点不做分块搜索），所以只在 GEMM
    # 条目上取；对全部条目取会在第一条 B 路条目上抛 KeyError。
    tiles = sorted({int(e["kernel_tile_n"]) for e in ops.values()
                    if e.get("kernel_tile_n")})
    print(f"    放置 {gemm_count} 个 GEMM 与 {bpath_count} 个算子级节点，"
          f"本地形状 {len(pimir_files)} 种 pim mlir")
    print(f"    算子编译器选出的分块: {tiles}（GeneSim 默认常量是 32）")

    # 编译期容量表回显到主流程日志（产物本体在 sidecar 的 capacity 键里）。
    capacity = sidecar.get("capacity") or {}
    if capacity.get("table"):
        print(f"\n{capacity['table']}")

    # 走方案路径时，Cluster 映射必须随 sidecar 回传，否则 GeneSim 会退回取模换算。
    if plan_path is not None:
        carried = sidecar.get("dpu_to_cluster")
        if not carried:
            raise StepFailed(
                "按 GeneSim 方案编译，但 sidecar 里没有 dpu_to_cluster——"
                "映射没有回传，GeneSim 会退回取模换算。"
            )
        declared = json.loads(Path(plan_path).read_text())["dpu_to_cluster"]
        if carried != declared:
            raise StepFailed(
                f"sidecar 带回的 dpu_to_cluster 与方案声明的不一致：\n"
                f"  方案: {declared}\n  sidecar: {carried}"
            )
        print(f"    Cluster 映射已随 sidecar 回传，与方案一致（{len(carried)} 项）")
    stale_fused = fused_kernels_missing_act_spec(sidecar)
    if stale_fused:
        raise StepFailed(
            "融合清单非空，但折激活的投影 kernel 没有 act_spec："
            f"{stale_fused[:3]}。按当前代码重跑放置导出。"
        )
    return sidecar_path


def _config_text(genesim: Path, config: str | Path) -> Optional[str]:
    """读仿真配置的文本。没有这份配置就返回 None。

    传文件名时从 genesim 的 conf 目录找；传路径时直接读，
    临时收过请求数的配置不在那个目录里。
    """
    path = config if isinstance(config, Path) else genesim / "conf" / config
    if not path.is_file():
        return None
    return path.read_text()


def _sidecar_of(genesim: Path, config_text: str) -> Optional[Path]:
    """从配置里取出 `compiler_placement_file`，没有就返回 None。"""
    import yaml

    config = yaml.safe_load(config_text) or {}
    relative = (config.get("scheduler") or {}).get("compiler_placement_file")
    if not relative:
        return None
    path = Path(relative)
    return path if path.is_absolute() else genesim / path


def limited_config(source: Path, conf_dir: Path, max_requests: int) -> Path:
    """复制一份配置，把请求数收成 max_requests，原文件不动。

    仿真时间跟着 token 数走。10 个请求共 8084 个 token 要二十多分钟，
    而第 1 个请求已经同时有 prefill 和 decode。原配置留着，
    需要完整数字时显式把请求数加回去。
    """
    import yaml

    loaded = yaml.safe_load(source.read_text())
    loaded.setdefault("trace", {})["num_requests"] = max_requests
    conf_dir.mkdir(parents=True, exist_ok=True)
    out = conf_dir / source.name
    out.write_text(yaml.safe_dump(loaded, sort_keys=False))
    return out


def ceva_sim_config(source: Path, conf_dir: Path, placement: Path, *,
                    batch_size: int = 1, groups: int | None = None,
                    num_requests: int | None = None,
                    trace_path: str | None = None,
                    pipeline_overlap: bool | None = None,
                    dp_replicas: int | None = None,
                    weight_reuse: bool | None = None) -> Path:
    """复制 CEVA 场景，把放置 sidecar 指向本流程刚生成的产物，原文件不动。

    扫参参数（组大小/组数/请求数/trace 路径/流水重叠/DP 副本/权重复用）
    只在显式给出时写入：经典单档路径产出的配置与改动前逐字节一致。
    """
    import yaml

    loaded = yaml.safe_load(source.read_text())
    scheduler = loaded.setdefault("scheduler", {})
    scheduler["compiler_placement_file"] = str(placement)
    if batch_size != 1:
        scheduler["execution_microbatch_size"] = batch_size
    if groups is not None:
        scheduler["max_inflight_requests"] = groups
    if pipeline_overlap is not None:
        scheduler["pipeline_overlap"] = pipeline_overlap
    if dp_replicas is not None:
        scheduler["dp_replicas"] = dp_replicas
    if weight_reuse is not None:
        scheduler["weight_reuse"] = weight_reuse
    trace = loaded.setdefault("trace", {})
    if num_requests is not None:
        trace["num_requests"] = num_requests
    if trace_path is not None:
        trace["path"] = trace_path
    conf_dir.mkdir(parents=True, exist_ok=True)
    out = conf_dir / source.name
    out.write_text(yaml.safe_dump(loaded, sort_keys=False))
    return out


def _genesim_sim_command(genesim: Path, config_path: Path) -> list[str]:
    """返回跑 GeneSim 仿真的命令。

    首选 `./run.sh`——那是 GeneSim 上游设计的入口。但它每个子命令都先 `check_uv` +
    `check_venv`，机器上没装 uv、或者没跑过 `install.sh` 建 `.venv` 时会直接退出。

    这种情况下退回"用当前 python 直接跑 src/main.py"：仿真读的是已经生成好的 pim
    mlir，不自己编译算子，所以不需要 GeneSim 那套独立 venv。实测两条路径的
    `total_time_s` 小数位完全一致（1418.1092459087413），uv 只是包管理工具。

    这条退路的用途是让纯 CPU 容器里的验证能跑完；交付给用户仍应走 `install.sh` +
    `run.sh`，因为 `run.sh` 的其余子命令（predictor、upmem_checker）没有这条退路。
    """
    uv_available = shutil.which("uv") is not None
    venv_ready = (genesim / ".venv" / "bin" / "python").is_file()
    if uv_available and venv_ready:
        return ["./run.sh", "--config", str(config_path)]

    missing = "uv" if not uv_available else ".venv"
    print(f"    （缺 {missing}，改用当前 python 直接跑 src/main.py；"
          "结果与 run.sh 一致，见 _genesim_sim_command 的说明）")
    return [sys.executable, "src/main.py", "--config", str(config_path)]


def _sha256_file(path: Path) -> str:
    """文件的 sha256（冻结落位的复验口径，与 genesim 侧一致）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relocate_manifest_to_tier(results_dir: Path) -> None:
    """把冻结清单的路径重定位到本档目录并复验哈希（R21-06）。

    genesim 在自己的 `results/` 下冻结输入，copytree 到逐档目录后 manifest
    仍指向共享路径、`pim_mlir` 只记 `.opcompiler_cache` 原路径——搬移结果
    目录或清理共享 `results/` 后不可独立复现。重定位：inputs 改指本档
    `inputs/` 副本（copytree 已带）；pim_mlir 逐份冻结成本档 `pim_mlir/`
    内容寻址副本；全部复验哈希，缺档或失配直接失败。
    """
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return  # 无清单的产物（桩仿真）无从重定位。
    manifest = json.loads(manifest_path.read_text())
    for label, entry in sorted((manifest.get("inputs") or {}).items()):
        frozen = results_dir / "inputs" / Path(str(entry["path"])).name
        if not frozen.is_file():
            raise StepFailed(
                f"冻结落位失败：inputs.{label} 的本档副本缺失"
                f"（{frozen}），copytree 与冻结清单不同步")
        if _sha256_file(frozen) != entry.get("sha256"):
            raise StepFailed(
                f"冻结落位失败：inputs.{label} 的本档副本哈希与清单失配"
                f"（{frozen}）")
        entry["path"] = str(frozen)
    config_entry = manifest.get("config_file")
    if isinstance(config_entry, dict) and config_entry.get("path"):
        frozen = results_dir / "inputs" / Path(
            str(config_entry["path"])).name
        if not frozen.is_file() or _sha256_file(frozen) != config_entry.get(
                "sha256"):
            raise StepFailed(
                f"冻结落位失败：config_file 的本档副本缺失或哈希失配"
                f"（{frozen}）")
        config_entry["path"] = str(frozen)
    mlir_entries = manifest.get("pim_mlir") or []
    if mlir_entries:
        mlir_dir = results_dir / "pim_mlir"
        mlir_dir.mkdir(parents=True, exist_ok=True)
        for entry in mlir_entries:
            source = Path(str(entry["path"]))
            digest = str(entry.get("sha256"))
            if not source.is_file():
                raise StepFailed(
                    f"冻结落位失败：pim_mlir 的 {source} 已不在盘上，"
                    "无法冻结进本档")
            frozen = mlir_dir / f"{source.stem}-{digest[:16]}{source.suffix}"
            if not frozen.is_file():
                shutil.copy2(source, frozen)
            if _sha256_file(frozen) != digest:
                raise StepFailed(
                    f"冻结落位失败：pim_mlir 副本哈希与清单失配（{frozen}）")
            entry["path"] = str(frozen)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False,
                                        indent=2))


def step_e_simulate(
    genesim: Path, config: str | Path, log_dir: Path, *, label: str,
    results_dir: Optional[Path] = None, max_requests: Optional[int] = None,
) -> Dict[str, Any]:
    """(E) 跑仿真，核对 GEMM 的 trace 真的来自 pim mlir。"""
    print(f"\n[E] GeneSim 仿真（{label}）")
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = genesim / "conf" / config_path
    if max_requests is not None:
        # 临时配置放日志目录，不写进 genesim 仓，免得留下一份请求数被改过的副本。
        config_path = limited_config(config_path, log_dir, max_requests)
        print(f"    请求数收成 {max_requests}：{config_path.name}")
    # sidecar 与磁盘产物不是一套时，仿真跑出来的数字没有意义，先拦下来，
    # 省掉十几分钟的无效仿真。
    verify_sidecar_freshness(genesim, config_path)
    # 仿真 IR 与 sidecar 不是同一套算子编号时，注意力算子会整批退回手写
    # 模板，同样在启动前拦下。
    verify_sim_ir_matches_sidecar(genesim, config_path)
    # trace 缓存按签名判新旧，但这里要的是"这一轮确实重编过"，所以先清掉。
    traces = genesim / "pim_traces"
    if traces.is_dir():
        shutil.rmtree(traces)
    _run(_genesim_sim_command(genesim, config_path),
         cwd=genesim, log_path=log_dir / f"e_sim_{label}.log")

    # 来源校验挂在仿真入口上，而不是只挂在主流程里。慢速仿真测试也走这里，
    # 整批 trace 退回手写模板时必须在这里失败，不能只在主流程里才看得见。
    verify_trace_provenance(genesim)

    summary_path = genesim / "results" / "summary.json"
    if not summary_path.is_file():
        raise StepFailed(f"仿真没产出 {summary_path}")
    summary = json.loads(summary_path.read_text())
    if results_dir is not None:
        if results_dir.exists():
            shutil.rmtree(results_dir)
        shutil.copytree(genesim / "results", results_dir)
        # 冻结落位（R21-06）：copytree 只带来字节副本，manifest 的 inputs
        # 仍指向 genesim/results/inputs 共享目录、pim_mlir 只记
        # .opcompiler_cache 原路径——重定位到本档并冻结 MLIR 副本后，
        # 结果目录可脱离共享目录独立复现。
        relocate_manifest_to_tier(results_dir)

    print(f"    total_time_s = {summary['total_time_s']:.3f}")
    print(f"    tokens/s     = {summary['throughput_tokens_per_s']:.3f}")
    return summary


def _trace_metadata(path: Path) -> Dict[str, Any]:
    """读一条 pim trace 的元数据。

    头部是 `PIMT` + version(4) + 指令条数(8) + 元数据长度(4)，与
    genesim `pim_isa.save_trace_file` 同一布局。头部是 `PIMT` 时按长度字段
    切 JSON，长度超出文件就是头部损坏，直接报错。头部不是 `PIMT` 的
    （测试里的简化头）退回按字节扫描。
    """
    raw = path.read_bytes()
    if len(raw) >= 20 and raw[:4] == b"PIMT":
        meta_len = struct.unpack_from("<I", raw, 16)[0]
        # 头部是 PIMT 就按长度字段切。长度超出文件说明头部损坏，
        # 不能退回扫描——文件里碰巧有 `{` 会让损坏被静默放过。
        if 20 + meta_len > len(raw):
            raise StepFailed(
                f"trace 头部损坏：{path.name} 声明元数据 {meta_len} 字节，"
                f"文件只有 {len(raw)} 字节")
        try:
            return json.loads(raw[20:20 + meta_len].decode("utf-8"))
        except json.JSONDecodeError:
            raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    # 头部不是 PIMT（测试里的简化头）时退回扫描。
    start = raw.find(b"{")
    if start < 0:
        raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    try:
        meta, _ = json.JSONDecoder().raw_decode(
            raw[start:].decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        raise StepFailed(f"trace 文件损坏，读不出元数据：{path.name}")
    return meta


def stale_pimir_entries(ops: Dict[str, Dict[str, Any]]) -> list[str]:
    """返回 `pimir_sha256` 与磁盘文件对不上的算子 id。

    sidecar 记的是绝对路径：缓存重建后路径还在、内容已经变了，哈希是唯一
    能发现"这份 sidecar 与当前算子编译产物不是一套"的办法。
    """
    stale = []
    for op_id, entry in ops.items():
        digest = hashlib.sha256(
            Path(entry["pimir_path"]).read_bytes()
        ).hexdigest()
        if digest != entry["pimir_sha256"]:
            stale.append(op_id)
    return stale


def _stale_sidecar_message(stale: list[str]) -> str:
    return (
        f"{len(stale)} 个算子的 pimir_sha256 与磁盘上的 pim mlir 不一致"
        f"（如 op{stale[:5]}）：sidecar 与当前算子编译产物不是一套，"
        "请重新导出（scripts/run_full_pipeline.py 不带 --skip-simulation，"
        "或单独跑 scripts/export_pp_placement.py）。"
    )


def verify_sidecar_freshness(genesim: Path, config: str | Path) -> None:
    """仿真用的 sidecar 必须与磁盘上的 pim mlir 是一套。

    主流程的哈希核对只在导出步骤里，仿真入口绕得开它：缓存被重建过、
    sidecar 还是旧的时，仿真照样跑完，代价链吃的却不是当前的产物。
    配置里没有 `compiler_placement_file` 时没有 sidecar 可核对，直接返回。
    """
    config_text = _config_text(genesim, config)
    if config_text is None:
        return
    sidecar_path = _sidecar_of(genesim, config_text)
    if sidecar_path is None:
        return
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    ops = json.loads(sidecar_path.read_text())["operators"]
    without_hash = [op_id for op_id, e in ops.items() if not e.get("pimir_sha256")]
    if without_hash:
        raise StepFailed(
            f"{len(without_hash)}/{len(ops)} 个算子的 sidecar 没有 pimir_sha256，"
            "无法确认它与当前算子编译产物是一套，请重新导出。"
        )
    stale = stale_pimir_entries(ops)
    if stale:
        raise StepFailed(_stale_sidecar_message(stale))


def verify_sim_ir_matches_sidecar(genesim: Path, config: str | Path) -> None:
    """仿真加载的 IR 与 sidecar 必须是同一套算子编号。

    调度器按 op_id 查 sidecar 里的 pimir_path：精化 IR 是旧结构、sidecar 是
    新导出时，查不到的算子静默退回手写模板，仿真照样跑完（2026-10-07 的
    6144 个模板 trace 就是这么来的）。模型入口/出口没有设备算子，本来就不
    在 sidecar 里，其余算子缺一个都拦在仿真启动前，省十几分钟无效仿真。
    配置里没有 ir_path 或 sidecar 时没有可对拍的对象，直接返回。
    """
    config_text = _config_text(genesim, config)
    if config_text is None:
        return
    import yaml
    loaded = yaml.safe_load(config_text) or {}
    ir_rel = (loaded.get("model") or {}).get("ir_path")
    sidecar_path = _sidecar_of(genesim, config_text)
    if not ir_rel or sidecar_path is None:
        return
    ir_path = genesim / ir_rel
    if not ir_path.is_file():
        raise StepFailed(
            f"配置指向的仿真 IR 不存在：{ir_path}。先跑 (A2) 精化步骤生成"
            "（scripts/run_full_pipeline.py 主流程已包含这一步）。")
    if not sidecar_path.is_file():
        raise StepFailed(f"配置指向的 sidecar 不存在：{sidecar_path}")
    sidecar_ops = json.loads(sidecar_path.read_text())["operators"]
    ir = json.loads(ir_path.read_text())
    missing = [
        (int(op["op_id"]), str(op["op_type"])) for op in ir["operators"]
        if op["op_type"] not in ("MODEL_INPUT", "MODEL_OUTPUT")
        and not (sidecar_ops.get(str(op["op_id"])) or {}).get("pimir_path")
    ]
    if missing:
        kinds = sorted({t for _, t in missing})
        raise StepFailed(
            f"{len(missing)} 个算子在 sidecar 里没有 pimir_path（类型 {kinds}，"
            f"如 op{[i for i, _ in missing[:5]]}）：仿真 IR（{ir_path.name}）与 "
            f"sidecar（{sidecar_path.name}）不是同一套算子编号。"
            "精化 IR 多半是图骨架改结构之前生成的旧产物——重跑 "
            "scripts/run_full_pipeline.py（(A2) 会重新精化），或手动跑 "
            "genesim/scripts/refine_ir_with_flagtree.py。")


def verify_trace_provenance(genesim: Path, *, trace_dir: Path | None = None,
                            allowed_sources: set[str] | None = None) -> None:
    """核对每个算子的 trace 来源，避免"跑通了但其实走的是手写模板"。

    原先只扫 GEMM。注意力那一组（ROPE、MASK、SOFTMAX、GEMV）退回模板时
    这里看不见，仿真照样算通过，而那些算子的分块根本没进代价链。

    `trace_dir` 缺省仍是 genesim/pim_traces；`allowed_sources` 给了就额外要求
    每条 trace 的来源都在白名单里（调优正式路径要求零模板、零未知）。
    """
    sources: Dict[str, int] = {}
    by_op: Dict[str, Dict[str, int]] = {}
    tiles: Dict[Any, int] = {}
    root = Path(trace_dir) if trace_dir is not None else genesim / "pim_traces"
    for path in sorted(root.glob("op_*.pim_trace")):
        meta = _trace_metadata(path)
        source = (meta.get("compile_signature") or {}).get("trace_source", "unknown")
        op_type = str(meta.get("op_type", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        counts = by_op.setdefault(op_type, {})
        counts[source] = counts.get(source, 0) + 1
        pimir_meta = meta.get("pimir") or {}
        if pimir_meta:
            key = (pimir_meta.get("tile_n"), pimir_meta.get("k_iterations"))
            tiles[key] = tiles.get(key, 0) + 1

    if not sources:
        raise StepFailed("没有找到任何 trace，无法判断来源")
    print(f"    trace 来源: {sources}")
    if tiles:
        print(f"    (tile_n, k_iterations) 分布: {tiles}")
    if sources.get("template"):
        templated = sorted(op for op, counts in by_op.items() if counts.get("template"))
        raise StepFailed(
            f"{sources['template']} 个算子退回了手写模板（{', '.join(templated)}）；"
            "算子编译器的分块没有进入代价链。"
        )
    if allowed_sources is not None:
        extra = sorted(set(sources) - set(allowed_sources))
        if extra:
            raise StepFailed(f"trace 来源 {extra} 不在允许集合 {sorted(allowed_sources)} 里")



def ceva_verdict(verdict_path: Path) -> list[str]:
    """读 run_all 的判定报告，返回 token 行；结论不是 PASS 就失败。"""
    if not verdict_path.is_file():
        raise StepFailed(f"没有判定报告 {verdict_path}")
    text = verdict_path.read_text()
    if "结论: PASS" not in text or "[FAIL]" in text or "[SKIP]" in text:
        raise StepFailed(f"CEVA 判定未通过：{verdict_path}")
    return [line for line in text.splitlines() if "token 与 baseline" in line]


def run_ceva(args: argparse.Namespace, log_dir: Path) -> Path:
    """CEVA 后端：我方导出 65 图，交给工作区 run_all.sh 编译、仿真并判定。

    `--num-ceva >= 2` 时改走工作区 run_cascade.sh（设计 4.8）：导出带策略、
    按核 GC 与 DDR 布局、级联仿真与判定，全在工作区脚本里编排。
    """
    workspace = ceva_workspace()
    if getattr(args, "skip_simulation", False):
        # 该开关只在 genesim 分支生效；CEVA 分支曾静默忽略，容易让人误以为没有仿真。
        raise StepFailed("CEVA 后端不支持 --skip-simulation；要只到蓝图请用 --dry-run")
    bundle = workspace / "bundles" / args.run_name
    verdict_path = workspace / "runs" / args.run_name / "results" / "verdict.txt"
    num_ceva = getattr(args, "num_ceva", 1)
    strategy = getattr(args, "strategy", "single")
    dry_run = getattr(args, "dry_run", False)
    if num_ceva >= 2:
        print(f"\n[级联 1/3] 图编译器按策略导出 {strategy} bundle（{num_ceva} 核）")
        _run([sys.executable, "scripts/export_gml.py",
              "--ceva-reference", str(ceva_reference_dir()), "--use-opcompiler",
              "--out-dir", str(bundle), "--num-ceva", str(num_ceva),
              "--strategy", strategy],
             cwd=_REPO_ROOT, log_path=log_dir / "ceva_cascade_export.log")
        print("\n[级联 2/3] 清单校验")
        manifest = bundle / "core0" / "bundle.json" if strategy == "tp2" \
            else bundle / "bundle.json"
        _run([sys.executable, "scripts/verify_gml_artifact.py", str(manifest.parent),
              "--ceva-manifest", str(manifest)],
             cwd=_REPO_ROOT, log_path=log_dir / "ceva_cascade_manifest.log")
        print(f"\n[级联 3/3] {num_ceva} CEVA、策略 {strategy}：run_cascade.sh 编排全流程")
        cmd = ["bash", "run_cascade.sh", "-n", args.run_name,
               "--cores", str(num_ceva), "--strategy", strategy,
               "--steps", str(args.decode_steps),
               "--token-check", args.token_check]
        if dry_run:
            cmd.append("--dry-run")
        if getattr(args, "fresh", False):
            cmd.append("--fresh")
        _run(cmd, cwd=workspace, log_path=log_dir / "ceva_cascade.log")
        if not dry_run:                      # 干跑到蓝图为止，没有判定报告
            for line in ceva_verdict(verdict_path):
                print(f"    {line.strip()}")
        return verdict_path
    print("\n[1] 图编译器导出 65 张 CEVA 子图（厂商反量化权重作 checkpoint，bin 全部我方生成）")
    _run([sys.executable, "scripts/export_gml.py", "--ceva-reference", str(ceva_reference_dir()),
          "--use-opcompiler", "--out-dir", str(bundle)],
         cwd=_REPO_ROOT, log_path=log_dir / "ceva_export.log")
    print("\n[2] 清单校验（全部文件由我方生成，尺寸与摘要一致）")
    _run([sys.executable, "scripts/verify_gml_artifact.py", str(bundle),
          "--ceva-manifest", str(bundle / "bundle.json")],
         cwd=_REPO_ROOT, log_path=log_dir / "ceva_manifest.log")
    print("\n[3] 厂商 GC → DDR 布局 → CEVA 仿真 → token 判定")
    _run(["bash", "run_all.sh", "-n", args.run_name, "--gml-bundle", str(bundle),
          "-s", str(args.decode_steps), "--token-check", args.token_check],
         cwd=workspace, log_path=log_dir / "ceva_run_all.log")
    for line in ceva_verdict(verdict_path):
        print(f"    {line.strip()}")
    return verdict_path


def sweep_capacity_note(capacity: dict | None, batch_size: int) -> str:
    """从编译期容量折算出该档占用摘要：最紧 DPU 的可用预算占比。

    图按最大档编译一次（capacity.slots 记编译档）：非编译档是
    `fixed + var × B/slots` 的线性折算，不是该档真实的 plan_dpu 结果，
    显式标「折算」防止被读成逐档实测。
    """
    if not capacity:
        return "—"
    budget = float(capacity["budget_bytes"])
    compiled_slots = int(capacity.get("slots") or 1)
    folded = compiled_slots != batch_size
    note, worst = "—", -1.0
    for dpu_id, stats in capacity.get("dpus", {}).items():
        scaled = (stats["fixed_bytes"]
                  + stats["var_bytes_at_slots"] * batch_size / stats["slots"])
        ratio = scaled / budget
        if ratio > worst:
            note, worst = f"dpu{dpu_id} {ratio:.0%}", ratio
    tag = "编译期折算" if folded else "编译期"
    return f"{note}（{tag}）"


def synthesize_closed_trace(
    genesim: Path, log_dir: Path, batches: tuple[int, ...],
    max_requests: int | None,
    groups_list: tuple[int, ...] = (1,),
) -> tuple[str | None, int | None, int]:
    """合成扫参共用的 closed trace，返回 (trace 路径, 容量口径 max_seq, 请求数)。

    放在导出之前跑：容量表的序列长度必须与 trace 同源（round05 问题 5），
    导出子进程要拿同一个 max_seq 建 KV 布局。全部档 B=1 时不合成 trace；
    但 B=1 且扫多组（R21-05）时容量口径仍从仿真实际消费的原始 trace 首请求
    推导（seq_len+output_length），不让导出侧回落 MAX_SEQ=64——经典
    B=1、G=1 单档路径不进扫参分支，维持不推导。
    """
    from math import lcm

    span = lcm(*batches)
    num_requests = (max(span, (max_requests // span) * span)
                    if max_requests else 4 * span)
    if max(batches) <= 1:
        if max(groups_list) <= 1:
            return None, None, num_requests
        from tune._genesim import trace_api

        trace_cls, _ = trace_api()
        source = trace_cls.load_binary(genesim / "traces" / "llama2_7b.trace")
        derived = closed_trace_max_seq(source.to_dict())
        print(f"\n[E] B=1 多组扫参：容量口径 max_seq={derived}"
              "（原始 trace 首请求 seq_len+output_length，不合成 closed trace）")
        return None, derived, num_requests
    from tune._genesim import trace_api

    trace_cls, _ = trace_api()
    source = trace_cls.load_binary(genesim / "traces" / "llama2_7b.trace")
    source_dict = source.to_dict()
    closed = write_closed_batch_trace(
        source_dict, num_requests=num_requests,
        out_path=log_dir / "llama2_7b_closed.trace.json")
    print(f"\n[E] closed batch trace：首请求复制 {num_requests} 份（{closed.name}）")
    return str(closed), closed_trace_max_seq(source_dict), num_requests


def sweep_scope_dimensions(batches: tuple[int, ...],
                           groups_list: tuple[int, ...], *, overlap: bool,
                           replicas: int, requests: int,
                           kv_head_pin: bool = False,
                           max_seq: int | None = None,
                           weight_reuse_off: bool = False) -> dict:
    """一次扫参的口径身份（单一真源）：全部影响 sidecar/放置/容量口径的输入。

    coverage 行与专项报告自动名都从这份维度集生成（round10 问题 3）：新的
    放置类开关上线时只改这里，报告身份自动认识它。max_seq 只在显式覆盖时
    进身份（缺省从 trace 推导，口径由 sidecar capacity.max_seq 承载）；
    weight_reuse 只在关闭（逐槽产物对照，R21-04）时进身份——缺省开与
    现行 sweep 的 coverage 行逐字节一致。
    """
    dims: dict = {
        "batches": tuple(batches),
        "groups": tuple(groups_list),
        "overlap": bool(overlap),
        "replicas": int(replicas),
        "kv_head_pin": bool(kv_head_pin),
        "requests": int(requests),
    }
    if max_seq is not None:
        dims["max_seq"] = int(max_seq)
    if weight_reuse_off:
        dims["weight_reuse"] = False
    return dims


def coverage_line(dims: dict) -> str:
    """口径维度集的机器可读行（报告头「扫参范围」，索引与核验据此消费）。"""
    parts = [
        f"batches={','.join(str(b) for b in dims['batches'])}",
        f"groups={','.join(str(g) for g in dims['groups'])}",
        f"overlap={'on' if dims['overlap'] else 'off'}",
        f"replicas={dims['replicas']}",
        f"kv_head_pin={'on' if dims['kv_head_pin'] else 'off'}",
        f"requests={dims['requests']}",
    ]
    if "max_seq" in dims:
        parts.append(f"max_seq={dims['max_seq']}")
    if dims.get("weight_reuse") is False:
        parts.append("weight_reuse=off")
    return " ".join(parts)


def sweep_scope_name(dims: dict) -> str:
    """专项报告自动名：与口径行同一维度集——档位拼名，非缺省开关带后缀。"""
    name = (f"b{'-'.join(str(b) for b in dims['batches'])}"
            f"g{'-'.join(str(g) for g in dims['groups'])}")
    if dims["overlap"]:
        name += "ov"
    if dims["replicas"] > 1:
        name += f"dp{dims['replicas']}"
    if dims["kv_head_pin"]:
        name += "khp"
    if "max_seq" in dims:
        name += f"ms{dims['max_seq']}"
    if dims.get("weight_reuse") is False:
        name += "wr"
    return name


def _report_metadata(text: str) -> tuple[str, str]:
    """读报告头嵌的（扫参范围，产物根）；缺行返回空串。"""
    def label_value(label: str) -> str:
        return next((line.split("：", 1)[1].strip() for line in text.splitlines()
                     if line.startswith(f"> {label}：")), "")
    return label_value("扫参范围"), label_value("产物根")


def _covers_full_grid(coverage: str) -> bool:
    """扫参范围是否覆盖 --sweep 默认档（B 1..32 × G 1,2,4）。"""
    def tiers(key: str) -> set[int]:
        found = re.search(rf"\b{key}=([\d,]*)", coverage)
        return ({int(x) for x in found.group(1).split(",") if x}
                if found else set())
    return (tiers("batches") >= set(DEFAULT_SWEEP_BATCHES)
            and tiers("groups") >= set(DEFAULT_SWEEP_GROUPS))


def _write_report_index(out_dir: Path) -> Path:
    """report.md 固定为全量扫参语义：指向全量报告或列出各专项指针。

    专项聚合只写专项文件名，canonical 名不再被末次专项覆盖成「专项对」
    （round09 问题 4）；无全量产物时如实写明未产出。口径世代从各报告头部
    自带的「旧口径警示」行读取（round16 问题 2）：手工加在索引头部的警示
    会随重建静默丢弃，改由生成器逐行标注，重建后 R2/R3 重出即自然退役。
    """
    entries = []
    for path in sorted(out_dir.glob("report-*.md")):
        text = path.read_text()
        coverage, root = _report_metadata(text)
        entries.append((path.name, coverage, root, "旧口径警示" in text))
    lines = ["# batch 扫参报告索引", ""]
    full = [name for name, coverage, _, _ in entries if _covers_full_grid(coverage)]
    if full:
        labeled = [f"{n}（旧口径/推荐未定，待重扫）" if any(
            e[0] == n and e[3] for e in entries) else n for n in full]
        lines.append(f"全量扫参（--sweep 默认档 B {','.join(str(b) for b in DEFAULT_SWEEP_BATCHES)}"
                     f" × G {','.join(str(g) for g in DEFAULT_SWEEP_GROUPS)}）报告："
                     f"{'、'.join(labeled)}")
    else:
        lines.append("全量扫参（--sweep 默认档）报告：未产出；以下为专项报告及其覆盖范围。")
    if any(stale for _, _, _, stale in entries):
        lines += ["", "> ⚠ 口径混合：标「旧口径」的报告数字产生于旧执行语义"
                  "（集合通信旧口径/逐槽 prefill/无链路归因，重扫前推荐视为"
                  "未定），重跑登记见实施文档第八节 R9/R10。"]
    lines += ["", "| 报告 | 扫参范围 | 产物根 | 口径 |", "|---|---|---|---|"]
    for name, coverage, root, stale in entries:
        caliber = ("旧口径/推荐未定（待重扫 R9/R10）" if stale
                   else "新口径")
        lines.append(f"| {name} | {coverage or '—'} | {root or '—'} | {caliber} |")
    index = out_dir / "report.md"
    index.write_text("\n".join(lines) + "\n")
    return index


def mark_stale_report_rows(out_dir: Path) -> None:
    """核对各专项报告逐档行的产物指纹与盘上目录，不一致的行标注过期。

    产物目录被后续运行覆盖后旧报告既不失效也不标注，是报告引用已消亡
    世代的根因（round09 问题 2）；每轮扫参落盘后让盘上全部报告自检一遍。
    """
    for report_path in sorted(out_dir.glob("report-*.md")):
        lines = report_path.read_text().splitlines()
        _, root_text = _report_metadata("\n".join(lines))
        if not root_text:
            continue
        root_path = Path(root_text)
        sweep_root = root_path if root_path.is_absolute() else _REPO_ROOT / root_path
        changed = False
        in_table = False
        for i, line in enumerate(lines):
            if line.startswith("## "):
                in_table = line.startswith("## 逐档结果")
                continue
            if not in_table or not line.startswith("|"):
                continue
            if "本行过期" in line or "未核验" in line:
                continue
            cells = line.split("|")
            if len(cells) < 6:
                continue
            batch, groups, overlap, reuse = (
                cells[1].strip(), cells[2].strip(), cells[3].strip(),
                cells[4].strip() if len(cells) > 4 else "开")
            # 只认数据行（B/G 为数字），表头与分隔线天然跳过。
            if not (batch.isdigit() and groups.isdigit()):
                continue
            fingerprint = cells[-2].strip()
            if not re.fullmatch(r"[0-9a-f]{8}", fingerprint):
                lines[i] = line[:-1] + " ⚠ 无产物指纹，未核验 |"
                changed = True
                continue
            # 复用列是后加的：旧报告（无该列）按重叠列后即时间列识别，
            # reuse 缺省「开」不影响 stem——wr 后缀只属于显式关档。
            suffix = ("ov" if overlap == "开" else "") + (
                "wr" if reuse == "关" else "")
            stem = f"b{batch}g{groups}{suffix}"
            # 候选目录按 stem 前缀匹配（dp/khp/ms 口径后缀都在其后），
            # 指纹相等最终判别——报告身份与产物目录后缀同维度集。
            candidates = [d for d in sweep_root.glob(stem + "*") if d.is_dir()]
            if not any(_artifact_fingerprint(d) == fingerprint
                       for d in candidates):
                lines[i] = line[:-1] + "（产物已被覆盖，本行过期） |"
                changed = True
        if changed:
            report_path.write_text("\n".join(lines) + "\n")


def run_batch_sweep(
    genesim: Path, log_dir: Path, placement: Path,
    batches: tuple[int, ...], groups_list: tuple[int, ...],
    args: argparse.Namespace,
    *, trace_path: str | None, num_requests: int, max_seq: int | None,
    sim_config: str = CEVA_SIM_CONFIG,
    report_name: str | None = None,
) -> Path:
    """按 (B, G) 笛卡尔积逐档仿真，聚合报告写 test-results/batch-sweep/。

    closed trace 与容量口径 max_seq 在导出前已由 `synthesize_closed_trace`
    备好（同一份 trace 喂导出与仿真，容量墙与序列长度同源）；组数 G 映射
    `max_inflight_requests` 的组数口径。触容量墙的档记墙位后继续其余档，
    其余失败直接抛。容量墙位由运行时捕获与编译期容量折算互为印证。
    专项聚合写 `report-<name>.md`（`--report-name` 或按档位自动命名），
    `report.md` 固定为索引；落盘后自检盘上报告的产物指纹。返回专项报告路径。
    """
    base_config = genesim / "conf" / sim_config
    if not base_config.is_file():
        raise StepFailed(f"没有 CEVA 场景配置 conf/{sim_config}")
    # DP 副本开关写进每档仿真配置：genesim 按副本放置把不同组路由到不同
    # DPU 组（sidecar 的 dp_replicas 载荷与它互检）。
    dp_replicas_arg = (args.replicas if args.replicas > 1 else None)

    # 编译期容量表：sidecar 产物带逐 DPU 折算与墙位（问题 2 的接线落点）。
    placement_data = (json.loads(Path(placement).read_text())
                      if Path(placement).is_file() else {})
    capacity = placement_data.get("capacity") or None
    # 容量口径核对：sidecar 记录的 max_seq 必须与本轮 trace 推导值一致，
    # 不一致说明容量表对着另一条序列长度折算，墙位不可信，直接抛。
    if capacity is not None and max_seq is not None:
        recorded = capacity.get("max_seq")
        if recorded is None:
            raise StepFailed(
                "sidecar capacity 载荷没有记录 max_seq，容量墙位无法与"
                "仿真 trace 的序列长度核对；请用当前导出脚本重新导出。")
        if int(recorded) != int(max_seq):
            raise StepFailed(
                f"容量表 max_seq={recorded} 与仿真 trace 推导值 {max_seq} "
                "不一致：KV 布局与 trace 不是同一序列长度口径，墙位不可信。"
                "重跑本轮扫参让导出与 trace 同源。")

    runs: list[SweepRun] = []
    capacity_walls: list[tuple[int, int]] = []
    # --pipeline-overlap 成对扫参：同一 (B,G) 档关/开各跑一次（标签带 ov 后缀），
    # 报告的「流水重叠对比」段消费两态产物（R8 曲线对比交付面）。
    overlap_states = ((False, ""), (True, "ov")) if args.pipeline_overlap \
        else ((False, ""),)
    # --weight-reuse 成对对照（R21-04）：同一 B/trace/资源下复用开（缺省，
    # 合批产物）与关（逐槽产物）各跑一次（标签带 wr 后缀），报告的
    # 「权重复用对比」段消费两态产物。
    reuse_states = ((True, ""), (False, "wr")) if getattr(
        args, "weight_reuse", False) else ((True, ""),)
    # 放置/容量口径后缀（dp/khp/ms）与报告身份同一维度集：同 (B,G) 档不同
    # 口径的产物目录互不覆盖（round09 问题 2 的 dp 同类路径，round10 问题 3）。
    tier_suffix = (f"dp{dp_replicas_arg}" if dp_replicas_arg is not None else "")
    if getattr(args, "kv_head_pin", False):
        tier_suffix += "khp"
    if getattr(args, "max_seq", None) is not None:
        tier_suffix += f"ms{args.max_seq}"
    for batch_size in batches:
        for groups in groups_list:
            for overlap_on, suffix in overlap_states:
                for reuse_on, reuse_suffix in reuse_states:
                    label = (f"b{batch_size}g{groups}{suffix}"
                             f"{reuse_suffix}{tier_suffix}")
                    config = ceva_sim_config(
                        base_config, log_dir / "ceva-config", placement,
                        batch_size=batch_size, groups=groups,
                        num_requests=num_requests, trace_path=trace_path,
                        pipeline_overlap=(True if overlap_on else None),
                        dp_replicas=dp_replicas_arg,
                        weight_reuse=(False if not reuse_on else None))
                    try:
                        results_dir = log_dir / "sweep" / label
                        summary = step_e_simulate(
                            genesim, config, log_dir, label=label,
                            results_dir=results_dir)
                    except StepFailed as exc:
                        # 按 genesim 容量溢出的稳定标记判类，不做消息语言嗅探：
                        # 真实报错是英文，中文「容量」字样反而来自产物契约失败。
                        if "PIM_CAPACITY_EXCEEDED" in str(exc):
                            print(f"    B={batch_size}, G={groups} 触容量墙：{exc}")
                            capacity_walls.append((batch_size, groups))
                            continue
                        raise
                    metrics_path = results_dir / "batch_metrics.json"
                    metrics = (json.loads(metrics_path.read_text())
                               if metrics_path.is_file() else None)
                    # 副本身份以产物为准（genesim 只在路由开启时写 dp_replicas），
                    # 产物缺键（如桩仿真）时回落本轮 --replicas。
                    tier_replicas = (
                        int(metrics["dp_replicas"])
                        if metrics and metrics.get("dp_replicas")
                        else int(getattr(args, "replicas", 1)))
                    comm_path = results_dir / "communication_metrics.json"
                    link_wait_total = None
                    if comm_path.is_file():
                        link_wait_total = float(json.loads(
                            comm_path.read_text()).get(
                                "total_queue_wait_time_s", 0.0) or 0.0)
                    # 口径门禁输入（R21-03）：首请求的 forward 步数与输出长度
                    # ——合批语义下 forward_steps == output_length，逐槽旧口径
                    # 会多出 B-1 步，门禁据此降级标注。
                    forward_steps = output_length = 0
                    requests_path = results_dir / "request_metrics.json"
                    if requests_path.is_file():
                        first = json.loads(requests_path.read_text())[0]
                        forward_steps = int(first.get("forward_steps", 0))
                        output_length = int(first.get("output_length", 0))
                    runs.append(SweepRun(
                        batch_size, groups, summary, metrics,
                        sweep_capacity_note(capacity, batch_size),
                        pipeline_overlap=overlap_on,
                        replicas=tier_replicas,
                        fingerprint=_artifact_fingerprint(results_dir),
                        link_wait_total_s=link_wait_total,
                        kv_head_pin=bool(getattr(args, "kv_head_pin", False)),
                        cores=args.num_dpus * tier_replicas,
                        weight_reuse=reuse_on,
                        forward_steps=forward_steps,
                        output_length=output_length))

    # 运行时 B 墙归纳：只有该 B 的全部 G 档都触墙（组并发不再区分）才升格
    # 为 B 墙；部分 G 触墙是并发组数压满容量，保留 (B,G) 组合不进 B 墙。
    # 编译期折算只看 B，两者取小、以运行时实测优先标注。
    b_only_walls = sorted({
        batch for batch, _ in capacity_walls
        if all((batch, g) in capacity_walls for g in groups_list)
    })
    wall_at, wall_source = None, ""
    if b_only_walls:
        wall_at, wall_source = min(b_only_walls), "运行时捕获"
    compile_wall = capacity.get("wall_batch") if capacity else None
    if compile_wall is not None and (wall_at is None or compile_wall < wall_at):
        wall_at, wall_source = compile_wall, "编译期容量表判定"

    # 报告头嵌机器可读的扫参范围与产物根：索引据此归档、过期核验据此定位
    # 产物目录（round09 问题 2/4）。口径维度集是单一真源（round10 问题 3），
    # coverage 行与自动命名都从它生成。
    dims = sweep_scope_dimensions(
        batches, groups_list, overlap=args.pipeline_overlap,
        replicas=int(getattr(args, "replicas", 1)), requests=num_requests,
        kv_head_pin=bool(getattr(args, "kv_head_pin", False)),
        max_seq=getattr(args, "max_seq", None),
        weight_reuse_off=bool(getattr(args, "weight_reuse", False)))
    coverage = coverage_line(dims)
    sweep_root_dir = log_dir / "sweep"
    root_ref = (str(sweep_root_dir.relative_to(_REPO_ROOT))
                if sweep_root_dir.is_relative_to(_REPO_ROOT)
                else str(sweep_root_dir))
    report = aggregate_batch_report(
        runs, tile_m_by_batch=decode_tile_m_by_batch(placement),
        capacity_wall_at=wall_at, capacity_wall_source=wall_source,
        capacity_wall_hits=sorted(set(capacity_walls)),
        coverage=coverage, sweep_root=root_ref)
    out_dir = _REPO_ROOT / "test-results" / "batch-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = report_name or sweep_scope_name(dims)
    scope_path = out_dir / f"report-{name}.md"
    scope_path.write_text(report + "\n")
    _write_report_index(out_dir)
    mark_stale_report_rows(out_dir)
    print(f"\n扫参报告: {scope_path}")
    print(f"报告索引: {out_dir / 'report.md'}")
    return scope_path


DEFAULT_SWEEP_BATCHES = (1, 2, 4, 8, 16, 32)
DEFAULT_SWEEP_GROUPS = (1, 2, 4)


def expand_sweep_defaults(batches: tuple[int, ...],
                          groups_list: tuple[int, ...]) -> tuple[
        tuple[int, ...], tuple[int, ...]]:
    """--sweep 的默认档位：B 档延伸到 tile_m 上限（内核 BLOCK_M=16）与
    容量墙以上（1,2,4,8,16,32），G 档 1,2,4——饱和点与墙位都落在扫过的
    档位内（round05 问题 6）。与显式 --batch/--groups 取并集。"""
    return (tuple(sorted(set(batches) | set(DEFAULT_SWEEP_BATCHES))),
            tuple(sorted(set(groups_list) | set(DEFAULT_SWEEP_GROUPS))))


def resolve_sim_batches(sim_batches_text: str | None,
                        batches: tuple[int, ...]) -> tuple[int, ...]:
    """--sim-batches 子集解析：缺省全集；给出时必须 ⊆ --batch 档。

    全档命令按 G 拆分仍超长时（B=1 单档在 128 请求下达小时级），用子集把
    命令拆短；导出仍按全 --batch 档，sidecar 世代不变，只有仿真范围收窄。
    """
    if sim_batches_text is None:
        return batches
    sim_batches = parse_batch_list(sim_batches_text)
    unknown = sorted(set(sim_batches) - set(batches))
    if unknown:
        raise ValueError(f"--sim-batches {unknown} 不在 --batch 档内")
    return sim_batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("genesim", "ceva"), default="genesim",
                        help="genesim=多 DPU 代价仿真；ceva=我方 GML 走厂商 GC 与 CEVA 仿真")
    parser.add_argument("--run-name", default=None,
                        help="CEVA 后端的实验名，包与运行目录都按它命名")
    parser.add_argument("--token-check", choices=("strict", "warn"), default="warn",
                        help="CEVA：strict 要求 token 与厂商基线一致；warn 不一致只告警（默认）")
    parser.add_argument("--decode-steps", type=int, default=10,
                        help="CEVA 后端的 decode 步数，至少 1（厂商基线为 10）")
    parser.add_argument("--num-ceva", type=int, default=1,
                        help="CEVA 级联核数，默认 1（单核路径不变）")
    parser.add_argument("--strategy", choices=("single", "pp2", "tp2"),
                        default="single", help="CEVA 级联切分策略")
    parser.add_argument("--fresh", action="store_true",
                        help="级联仿真忽略断点状态重跑（保留已编译 GML）")
    parser.add_argument("--dry-run", action="store_true",
                        help="CEVA 级联干跑：到蓝图与布局为止，不启动仿真")
    parser.add_argument("--num-stages", type=int, default=4,
                        help="流水段数：1=纯张量并行，num_dpus=纯流水")
    parser.add_argument("--batch", default="1",
                        help=("batch 档，逗号列表（如 1,2,4,8）进扫参模式："
                              "一次导出逐档产物，逐档仿真并聚合报告；缺省单档 1 走原流程"))
    parser.add_argument("--groups", default="1",
                        help=("并发组数 G 档（逗号列表，映射 max_inflight_requests "
                              "的组数口径）；与 --batch 笛卡尔积扫参"))
    parser.add_argument("--sweep", action="store_true",
                        help=("按默认档位全量扫参：B 档 1,2,4,8,16,32（覆盖 "
                              "tile_m 饱和点与容量墙）、G 档 1,2,4。经典单档"
                              "路径不受影响（--batch 缺省仍为 1）"))
    parser.add_argument("--sim-batches", default=None,
                        help=("只仿真 --batch 的这个子集（导出仍按全 --batch "
                              "档，sidecar 世代不变）；把超长扫参命令拆成多次"
                              "短命令用，缺省仿真全部档"))
    parser.add_argument("--max-seq", type=int, default=None,
                        help=("容量表的最大序列长度。缺省从仿真 trace 的 "
                              "seq_len+output_length 推导，保证容量墙与真实"
                              "序列同源；显式给出时以它为准"))
    parser.add_argument("--replicas", type=int, default=1,
                        help=("DP 副本数（R8 多 batch 并行）：每副本持有全部"
                              "权重、KV/激活按副本划分，genesim 把不同组路由到"
                              "不同副本的 DPU 组；物理核数 = --num-dpus × 副本数"
                              "（当前覆盖 64/128 核）。缺省 1 等价现状"))
    parser.add_argument("--pipeline-overlap", action="store_true",
                        help=("R8 流水优化的扫参开关：开启后逐档在关/开两态"
                              "各跑一次仿真，报告输出「流水重叠对比」段"
                              "（开关两态曲线对比）；缺省关闭走经典单态"))
    parser.add_argument("--weight-reuse", action="store_true",
                        help=("R21-04 权重复用固定 B 对照开关：同一 B/trace/"
                              "资源下复用开（缺省，合批 M 合并产物）与关"
                              "（逐槽产物）各跑一次，报告输出「权重复用对比」"
                              "段；负载形态不变，只切换装载份数"))
    parser.add_argument("--kv-head-pin", action="store_true",
                        help=("执行侧 KV head 分片：KV 读取算子按 head 归属"
                              "钉到持有该 head 的 DPU；缺省整段首 DPU 钉扎、"
                              "B=1 执行放置不变"))
    parser.add_argument("--report-name", default=None,
                        help=("扫参报告的专项文件名（写 report-<name>.md），"
                              "缺省按档位自动命名；report.md 固定为全量扫参"
                              "索引，不再被专项聚合覆盖"))
    parser.add_argument("--num-dpus", type=int, default=None,
                        help="逻辑 DPU 数；缺省取 CEVA 画像的核数（64）")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="仿真的请求数上限。默认跟随配置；"
                             "慢速测试传 1，只要 prefill 与 decode 各一次")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="只验证到 sidecar，不跑仿真（几分钟 vs 十几分钟）")
    parser.add_argument("--ab-compare", action="store_true",
                        help="额外跑一次关掉本地分片形状的对照，输出比值")
    parser.add_argument("--log-dir", default=None,
                        help="日志目录，默认 <repo>/test-results/full-pipeline")
    parser.add_argument(
        "--no-pu-mapping", action="store_true",
        help=(
            "跳过第 [0] 步，不让 GeneSim 给 PU 映射，改由 --num-stages 直接指定切分。"
            "用于对照：验证走方案与直接给段数产出的 sidecar 语义相同。"
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=32,
        help=(
            "算子编译子任务的并行线程数（默认 32）。只并行子进程型编译，"
            "各任务写各自的内容寻址缓存文件，互不影响；经 OPCOMPILER_JOBS "
            "传给导出与精化子进程。"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_dpus is None:
        from contracts.hardware_profile import load_profile
        args.num_dpus = load_profile(DEFAULT_PROFILE).total_clusters
    if args.jobs < 1:
        print("错误: --jobs 至少为 1", file=sys.stderr)
        return 2
    # 并行度经环境变量下发给导出/精化子进程（opcompiler_bridge.parallel 读它）。
    os.environ["OPCOMPILER_JOBS"] = str(args.jobs)
    if args.backend == "ceva":
        if not args.run_name:
            print("错误: --backend ceva 需要 --run-name", file=sys.stderr)
            return 2
        if args.decode_steps < 1:
            print("错误: CEVA 全流程至少需要 1 步 decode", file=sys.stderr)
            return 2
        log_dir = Path(args.log_dir) if args.log_dir else _REPO_ROOT / "test-results" / f"ceva-{args.run_name}"
        # 入口处归一为绝对路径：下游 ceva_sim_config/trace 写入按各自基准拼
        # 相对路径（genesim 仓根、genesim/conf），相对 log_dir 会让配置与
        # trace 双双找不到。
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            verdict_path = run_ceva(args, log_dir)
            token_lines = ceva_verdict(verdict_path)
        except StepFailed as exc:
            print(f"\n失败: {exc}", file=sys.stderr)
            return 1
        print("\nCEVA 全流程通过：图编译 → 算子编译 → GML/bin 生成 → 厂商 GC → CEVA 仿真")
        # 告警档的 PASS 只说明仿真正常，token 是否与基线一致要单独报。
        matched = bool(token_lines) and all(line.lstrip().startswith("[PASS]")
                                            for line in token_lines)
        print("token 对拍：通过（与厂商基线逐个一致）" if matched
              else "token 对拍：告警（与厂商基线不一致，见判定报告）")
        print(f"判定报告: {verdict_path}")
        return 0
    if args.num_dpus % args.num_stages:
        print(f"错误: num_stages={args.num_stages} 不能整除 num_dpus={args.num_dpus}",
              file=sys.stderr)
        return 2
    if args.replicas <= 0:
        print("错误: --replicas 必须为正", file=sys.stderr)
        return 2
    try:
        batches = parse_batch_list(args.batch)
        groups_list = parse_batch_list(args.groups)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if args.sweep:
        batches, groups_list = expand_sweep_defaults(batches, groups_list)
    try:
        # 仿真子集在默认档扩展之后解析：子集针对最终的导出档列表。
        sim_batches = resolve_sim_batches(
            getattr(args, "sim_batches", None), batches)
    except ValueError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    sweep_mode = batches != (1,) or groups_list != (1,)
    if args.max_seq is not None and args.max_seq <= 0:
        print("错误: --max-seq 必须为正", file=sys.stderr)
        return 2

    log_dir = Path(args.log_dir) if args.log_dir else _REPO_ROOT / "test-results" / "full-pipeline"
    log_dir = log_dir.resolve()  # 相对路径在下游各拼各的基准，入口统一归一
    log_dir.mkdir(parents=True, exist_ok=True)
    genesim = genesim_root()
    model_dir = llama2_7b_model_dir()
    tp_width = args.num_dpus // args.num_stages

    print("=" * 78)
    print(f"Llama2-7B 全流程：tp{tp_width}_pp{args.num_stages}（{args.num_dpus} 台 DPU）")
    print(f"  图编译: {_REPO_ROOT}")
    print(f"  GeneSim: {genesim}")
    print("=" * 78)

    try:
        ir_path = step_a_model_ir(genesim, model_dir, log_dir)
        # DP 副本按物理核数选场景配置：G=2 × 64 DPU 走 128 核配置。
        sim_config = sim_config_for_cores(args.num_dpus * args.replicas)
        if sim_config != CEVA_SIM_CONFIG:
            print(f"DP 副本 ×{args.replicas}：CEVA 场景切换为 conf/{sim_config}")
        plan_path = None
        if not args.no_pu_mapping:
            plan_path = step_zero_pu_mapping(
                genesim, ir_path, args.num_stages, args.num_dpus, log_dir,
                replicas=args.replicas, sim_config=sim_config,
            )
        # closed trace 与容量口径 max_seq 在导出之前备好：容量表必须按仿真
        # 同一条 trace 的序列长度建 KV 布局，容量墙才与真实序列同源。
        trace_path: str | None = None
        max_seq: int | None = args.max_seq
        num_requests = 0
        if sweep_mode:
            trace_path, derived_max_seq, num_requests = (
                synthesize_closed_trace(
                    genesim, log_dir, batches, args.max_requests,
                    groups_list=groups_list))
            if derived_max_seq is not None and max_seq is None:
                max_seq = derived_max_seq
            if max_seq is not None:
                print(f"    容量口径 max_seq={max_seq}（trace seq_len+output_length）")
        # 执行策略作为统一编译输入（R25-03）：本轮流水的执行策略含逐槽态
        # （--pipeline-overlap 的开态 / --weight-reuse 的关态）时，导出
        # 带逐槽编译，与各档仿真配置选择同一策略。
        placement = step_bcd_export(
            args.num_stages, args.num_dpus, log_dir, plan_path=plan_path,
            batch_sizes=batches, max_seq=max_seq, replicas=args.replicas,
            kv_head_pin=args.kv_head_pin,
            per_slot_prefill=bool(
                args.pipeline_overlap
                or getattr(args, "weight_reuse", False)),
        )
        # 精化放在放置导出之后：融合清单随放置 sidecar 一起去，先跑就没有它。
        step_a2_refine_pimir_ir(genesim, ir_path, log_dir, placement=placement)

        if args.skip_simulation:
            print("\n跳过仿真（--skip-simulation）。到 sidecar 为止的链路已验证。")
            print("\n全流程验证通过（未含仿真）。")
            return 0

        if sweep_mode:
            run_batch_sweep(genesim, log_dir, placement, sim_batches,
                            groups_list,
                            args, trace_path=trace_path,
                            num_requests=num_requests, max_seq=max_seq,
                            sim_config=sim_config,
                            report_name=getattr(args, "report_name", None))
            print("\n" + "=" * 78)
            print("batch 扫参完成：图编译 → 算子编译（逐相位逐档）→ 逐档仿真 → 汇总报告")
            print("=" * 78)
            return 0

        config_path = genesim / "conf" / sim_config
        if not config_path.is_file():
            raise StepFailed(f"没有 CEVA 场景配置 conf/{sim_config}")
        # CEVA 场景按物理核数选定；放置 sidecar 指向本流程刚生成的产物。
        config_path = ceva_sim_config(
            config_path, log_dir / "ceva-config", placement,
            pipeline_overlap=(True if args.pipeline_overlap else None),
            dp_replicas=(args.replicas if args.replicas > 1 else None))
        main_summary = step_e_simulate(
            genesim, config_path, log_dir, label="ceva",
            results_dir=Path("/tmp/full_pipeline_pimir"),
            max_requests=args.max_requests,
        )
        # 来源校验在 step_e_simulate 里已经做过，这里不再重扫一遍 trace。

        if args.ab_compare:
            ab_config = f"sim_llama2_7b_pp_tp{tp_width}pp{args.num_stages}_ab_global.yaml"
            if not (genesim / "conf" / ab_config).is_file():
                print(f"\n跳过 A/B：没有 conf/{ab_config}")
            else:
                ab_summary = step_e_simulate(
                    genesim, ab_config, log_dir, label="global",
                    results_dir=Path("/tmp/full_pipeline_global"),
                )
                ratio = main_summary["total_time_s"] / ab_summary["total_time_s"]
                print("\n[A/B] 本地分片形状 vs 模型级全局形状")
                print(f"    全局形状 total_time_s = {ab_summary['total_time_s']:.3f}")
                print(f"    本地分片 total_time_s = {main_summary['total_time_s']:.3f}")
                print(f"    比值 = {ratio:.4f}（tp_width={tp_width}，理论上 ≈ 1/{tp_width}）")
                if abs(ratio - 1.0) < 1e-6:
                    raise StepFailed(
                        "两次仿真结果完全相同——切分没有进入代价链。"
                    )
    except StepFailed as exc:
        print(f"\n失败: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("全流程验证通过：模型加载 → 图编译切分 → 算子编译 → GeneSim 代价")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
