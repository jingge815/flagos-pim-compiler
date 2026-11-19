"""多 CEVA 级联组合评估模型（设计 4.9）。

在每核实测分段 cycles 之上叠加核间依赖与通信代价，按策略结构组合端到端：
PP2 是串行链（全部段与搬运相加），TP2 是层内同步（同层两核取 max 再加归约）。
所有量统一换算到 cycles 域，最后按频率换算秒；参数来源写进溯源节。
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

from cascade.plan import CascadeBlueprint, authoritative_timing, canonical_seg_id

# 频率来源：/data/fjg/ceva 厂商硬件画像 cnnha_upd_b.cfg:15 frequencyOfNPM=1250。
FREQ_MHZ = 1250
FREQ_SOURCE = "cnnha_upd_b.cfg:15 frequencyOfNPM=1250"


@dataclasses.dataclass(frozen=True)
class LinkParams:
    """级联链路参数（host 中转链路的解析模型）。初始为 assumption，实测后标定。"""

    latency_s: float = 1e-6
    bandwidth_bytes_per_s: float = 25e9
    source: str = "assumption"


def parse_performance_counters(path: Path) -> dict[str, int]:
    """读 driver 落盘的 `stage: N cycles` 计数文件。"""
    counters = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^(.+?): (\d+) cycles$", line.strip())
        if match:
            counters[match.group(1)] = int(match.group(2))
    return counters


def parse_state_cycles(run_dir: Path) -> tuple[dict[int, dict[str, int]], dict]:
    """取每段的设备执行时间（优先 app TIMER 纯执行 cycles）。

    取值优先级：state.exec_cycles（TIMER，不含装载/快照 I/O）>
    state.device_cycles / results/device_cycles.json（EndOfExecution，含装载）>
    CycleEstimation 尾值（偏小）。后两者需要减去装载基线（`init_cycles.json`
    标定值或最小段），TIMER 不需要。`results/cycles_correction.json` 可对个别
    旧实现段做已知开销扣减，但每条必须是 `{value, derived_from, measured_at}`
    ——无溯源的手工扣减会让性能数字不可审计，直接拒绝。
    """
    run_dir = Path(run_dir)
    results = run_dir / "results"
    state = json.loads((results / "state.json").read_text())
    rows = authoritative_timing(state.get("timing", []))
    if not rows:
        raise KeyError(f"{run_dir}/results/state.json 没有分段记录")
    timer, device, raw, all_clocks = {}, {}, {}, {}
    for row in rows:
        seg = row["seg"]
        raw[seg] = int(row["cycles"])
        exec_cycles = int(row.get("exec_cycles", 0))
        device_cycles = int(row.get("device_cycles", 0))
        if device_cycles > 0:
            all_clocks[seg] = device_cycles
        if exec_cycles > 0:
            timer[seg] = exec_cycles
        elif device_cycles > 0:
            device[seg] = device_cycles
    if not device and not timer:
        legacy = results / "device_cycles.json"          # 旧版 state 的守护文件
        if legacy.is_file():
            for seg, entry in json.loads(legacy.read_text()).items():
                all_clocks[seg] = int(entry["clock"])
                if int(entry.get("exec_cycles", 0)) > 0:
                    timer[seg] = int(entry["exec_cycles"])
                else:
                    device[seg] = int(entry["clock"])
    non_timer = [seg for seg in raw if seg not in timer]
    init, init_source = 0, "timer:no_init"
    if non_timer:
        # EndOfExecution / CycleEstimation 含装载开销；TIMER 不含。
        # 基线取所有段（含 TIMER 段的 EndOfExecution）的最小值，避免只剩一段时
        # 把它自身减成 0。
        calib = results / "init_cycles.json"
        if calib.is_file():
            init = int(json.loads(calib.read_text())["cycles"])
            init_source = "calibrated:cmd0"
        else:
            pool = list(all_clocks.values()) or list(device.values())
            if not pool:
                pool = [raw[s] for s in non_timer]
            init = min(pool)
            init_source = "estimated:min_segment"
    correction = _load_correction(results / "cycles_correction.json")
    cycles: dict[int, dict[str, int]] = {}
    sources = set()
    for row in rows:
        seg = row["seg"]
        if seg in timer:
            value, src = timer[seg], "TIMER"
        elif seg in device:
            value, src = device[seg] - init, "EndOfExecution"
        else:
            value, src = raw[seg] - init, "CycleEstimation"
        sources.add(src)
        value -= correction.get(canonical_seg_id(seg), 0)
        cycles.setdefault(int(row["core"]), {})[canonical_seg_id(seg)] = max(value, 0)
    return cycles, {"init_cycles": init, "init_source": init_source,
                    "cycles_source": "+".join(sorted(sources))}


def _load_correction(corr_path: Path) -> dict[str, int]:
    """读 cycles_correction.json；每条扣减必须带推导来源与测量时间。"""
    if not corr_path.is_file():
        return {}
    raw = json.loads(corr_path.read_text())
    correction = {}
    for seg, entry in raw.items():
        missing = [k for k in ("value", "derived_from", "measured_at")
                   if not isinstance(entry, dict) or k not in entry]
        if missing:
            raise ValueError(
                f"{corr_path} 的 {seg} 缺 {missing}：手工扣减必须可复现，"
                "格式为 {value, derived_from, measured_at}")
        correction[canonical_seg_id(seg)] = int(entry["value"])
    return correction


def load_transfers(run_dir: Path) -> list[dict]:
    """读跨核搬运日志（字节 + host 耗时）。"""
    path = Path(run_dir) / "run" / "transfer_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def calibrate_link(transfers: list[dict]) -> LinkParams:
    """用实测搬运记录标定链路 `t = λ + bytes/W`（最小二乘拟合）。

    需要至少两种字节数；点数不足或拟合出非正带宽时保留 assumption 并注明。
    """
    points = [(t["bytes"], t["t_host"]) for t in transfers
              if t.get("bytes", 0) > 0 and t.get("t_host", 0) > 0]
    sizes = {size for size, _ in points}
    if len(points) < 4 or len(sizes) < 2:
        return LinkParams(source="assumption:insufficient_points")
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    if var == 0:
        return LinkParams(source="assumption:insufficient_points")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var
    intercept = mean_y - slope * mean_x
    if slope <= 0 or intercept < 0:
        return LinkParams(source="assumption:negative_fit")
    # 多进程仿真里"链路"就是 host 文件通道，标定值可复现但不等于 PIM 互联硬件参数。
    return LinkParams(latency_s=intercept,
                      bandwidth_bytes_per_s=1.0 / slope,
                      source=f"calibrated:host_file_channel:{n}points")


def _xfer_cycles(nbytes: int, link: LinkParams, freq_hz: float) -> float:
    return (link.latency_s + nbytes / link.bandwidth_bytes_per_s) * freq_hz


def compose(plan: CascadeBlueprint, cycles: dict[int, dict[str, int]],
            link: LinkParams, *, freq_mhz: float = FREQ_MHZ) -> dict:
    """按策略结构组合端到端指标。cycles 是 {core: {seg_id: busy cycles}}。"""
    freq_hz = freq_mhz * 1e6
    missing = [s.seg_id for s in plan.segments
               if s.seg_id not in cycles.get(s.core_id, {})]
    if missing:
        raise KeyError(f"缺少分段 cycles：{missing[:3]}")

    if plan.strategy == "pp2":
        seg_total = sum(cycles[s.core_id][s.seg_id] for s in plan.segments)
    else:
        # TP2：同 (相, 层, 段类型) 的两核取 max，head 同相取 max。
        groups: dict[tuple, int] = {}
        for s in plan.segments:
            if s.kind in ("attn_shard", "mlp_shard"):
                phase = ":".join(s.seg_id.split(":")[:2])       # prompt 或 decode:k
                key = (phase, s.layer_lo, s.kind)
            else:
                key = ("head", s.seg_id.rsplit(":c", 1)[0])     # head:{i}
            groups[key] = max(groups.get(key, 0), cycles[s.core_id][s.seg_id])
        seg_total = sum(groups.values())

    xfer_cycles = sum(_xfer_cycles(t.nbytes, link, freq_hz)
                      for t in plan.transfers)
    total_cycles = seg_total + xfer_cycles
    busy = {core: sum(cycles.get(core, {}).values())
            for core in range(plan.num_cores)}
    tokens = plan.decode_steps + 1          # prompt 首 token + 每步 decode 一个
    total_time_s = total_cycles / freq_hz
    return {"total_cycles": total_cycles, "total_time_s": total_time_s,
            "tokens": tokens,
            "tokens_per_s": tokens / total_time_s if total_time_s else 0.0,
            "per_core_busy_cycles": busy,
            "comm_cycles": xfer_cycles,
            "comm_ratio": xfer_cycles / total_cycles if total_cycles else 0.0}


def _ordered_stage(single: dict[str, int], prefix: str) -> list[int]:
    """按编号排序取 stage 序列；无编号的裸名（如 lm_head）排第一。"""
    keyed = []
    for name, value in single.items():
        if name == prefix:
            keyed.append((-1, value))
        elif name.startswith(prefix + "_") and name[len(prefix) + 1:].isdigit():
            keyed.append((int(name[len(prefix) + 1:]), value))
    return [value for _, value in sorted(keyed)]


def _decode_stage(single: dict[str, int], step: int) -> int:
    """单核计数的 decode 编号从 prompt_len+1 起；按顺序映射到第 step 步。"""
    values = _ordered_stage(single, "decode")
    if not values:
        return 0
    return values[min(step, len(values) - 1)]


def _head_stage(single: dict[str, int], index: int) -> int:
    """lm_head+argmax 的第 index 次（0=prompt）：裸名段在最前，其后按编号。"""
    heads = _ordered_stage(single, "lm_head")
    argmaxes = _ordered_stage(single, "argmax")
    if not heads:
        return 0
    head = heads[min(index, len(heads) - 1)]
    argmax = argmaxes[min(index, len(argmaxes) - 1)] if argmaxes else 0
    return head + argmax


def _baseline_total(single: dict[str, int], plan: CascadeBlueprint) -> int:
    """按蓝图范围折算单核基线总周期：prompt 一次 + 每步 decode + 每个 head。

    必须与实测范围等宽才能相除。步数少于基线时直接取 full_execution，会把完整
    基线除以部分实测，加速比与偏差都失真。步数取蓝图的 decode_steps（段名
    前缀口径，PP2/TP2 通用）。
    """
    steps = plan.decode_steps
    total = single.get("prompt", 0)
    for step in range(steps):
        total += _decode_stage(single, step)
    for n in range(steps + 1):
        total += _head_stage(single, n)
    return total


def _layer_scaling(run_dir: Path, override: Path | None = None) -> dict:
    """读「每层开销随本核驻留层数变化」的实测标定系数。

    实测：32 层单核 28.3M/层 → 16 层分核 19.3M/层（prompt，1.47×），decode 无此效应。
    `source_runs` 登记系数来自哪些运行——标定与被评估运行相交即 in_sample。
    外部文件（override）不声明 source_runs 直接拒绝，与 _load_correction 缺
    provenance 即拒绝同口径：静默空表会让自拟合系数伪装成 out-of-sample。
    文件缺失时退化为 1.0（即原始的「每核一半」假设），并在 provenance 里标 assumption。
    """
    if override is not None:
        data = json.loads(Path(override).read_text())
        runs = data.get("source_runs")
        if not str(data.get("source", "")).startswith("assumption"):
            if not isinstance(runs, list) or not runs:
                raise ValueError(
                    f"{override} 缺 source_runs：外部标定必须声明系数来自哪些运行，"
                    "或以 source: assumption 显式声明为假设口径")
            data["source"] = f"measured:{override}（外部标定）"
        data["source_runs"] = runs if isinstance(runs, list) else []
        return data
    path = Path(run_dir) / "results" / "layer_scaling.json"
    if path.is_file():
        data = json.loads(path.read_text())
        data.setdefault("source", "measured:layer_scaling")
        # 现场标定流程就是拿本次运行拟合的：没登记来源时按同源处理
        data.setdefault("source_runs", [Path(run_dir).name])
        return data
    # 本次运行没测过就退到仓库级默认（llama2-7b / PP2 / 2 核，一次性实测标定）
    default = Path(__file__).resolve().parent / "layer_scaling_llama2_pp2.json"
    if default.is_file():
        data = json.loads(default.read_text())
        data["source"] = f"{data.get('source', 'measured')}（仓库默认）"
        data.setdefault("source_runs", [])
        return data
    return {"prompt_per_layer_ratio": 1.0, "decode_per_layer_ratio": 1.0,
            "source": "assumption:half_per_core", "source_runs": []}


def predict_pp2(single: dict[str, int], plan: CascadeBlueprint,
                scaling: dict | None = None) -> dict:
    """预测模式（PP2）：单核 stage 级 cycles 按层均分外推两核分段。

    纯「每核一半」在实测中偏差约 47%：每层开销随本核驻留层数下降（prompt 1.47×）。
    这里乘上实测标定系数 `layer_scaling.json`；缺省（1.0）即原始 assumption 口径。
    """
    scaling = scaling or {}
    ratio_p = float(scaling.get("prompt_per_layer_ratio", 1.0))
    ratio_d = float(scaling.get("decode_per_layer_ratio", 1.0))
    cycles = {c: {} for c in range(plan.num_cores)}
    for seg in plan.segments:
        if seg.kind == "prompt_layers":
            value = int(single.get("prompt", 0) / 2 * ratio_p)
        elif seg.kind == "decode_layers":
            step = int(seg.seg_id.split(":")[1])
            value = int(_decode_stage(single, step) / 2 * ratio_d)
        else:
            value = _head_stage(single, int(seg.seg_id.split(":")[1]))
        cycles[seg.core_id][seg.seg_id] = value
    return cycles


def evaluate(run_dir: Path, plan: CascadeBlueprint, link: LinkParams | None = None, *,
             single_counters: dict[str, int] | None = None,
             single_total_cycles: int | None = None,
             freq_mhz: float = FREQ_MHZ,
             scaling_json: Path | None = None) -> dict:
    """归集一次级联运行的评估：实测组合 + 可选预测对比，落盘 cascade_eval.json。

    实测数据优先取 `results/state.json`（逐段全局模拟时钟，减装载基线）；
    缺失时回退到 `run/performance_counters_core{N}.txt`。链路参数缺省时用本次
    运行的搬运记录现场标定，标定失败则保留 assumption 并标注来源。
    标定与验证分离是机制不是约定：每个标定参数登记来源 run，与被评估 run 相交
    时偏差标注 in_sample、不能作为验收依据（需求 §5.4/§5.5）。
    """
    run_dir = Path(run_dir)
    state = run_dir / "results" / "state.json"
    if state.is_file():
        measured, baseline = parse_state_cycles(run_dir)
        source = "state.json"
    else:
        measured = {core: parse_performance_counters(
            run_dir / "run" / f"performance_counters_core{core}.txt")
            for core in range(plan.num_cores)}
        baseline = {"init_cycles": 0, "init_source": "counters_file"}
        source = "performance_counters"
    transfers = load_transfers(run_dir)
    link_from_this_run = link is None       # 现场标定的链路与被评估 run 同源
    if link is None:
        link = calibrate_link(transfers)
    result = compose(plan, measured, link, freq_mhz=freq_mhz)
    result["measured_source"] = source
    # 依赖结构：PP2 是层段串行链（单请求无并行收益），TP2 是层内两核同步。
    result["structure"] = {
        "dependency": "serial_chain" if plan.strategy == "pp2" else "intra_layer_sync",
        "parallel_cores": 1 if plan.strategy == "pp2" else plan.num_cores,
        "segments": len(plan.segments),
        "handoffs": len(plan.transfers),
    }
    result["params"] = {"frequency_mhz": freq_mhz, "frequency_source": FREQ_SOURCE,
                        "link": dataclasses.asdict(link),
                        "baseline": baseline,
                        # 芯方舟口径：PIM 颗粒即设备，每颗粒单运算核；三层抽象 (设备,核,引擎)。
                        "three_layer_abstraction": [plan.num_cores, 1, 1],
                        "three_layer_source": "厂商口径：每 PIM 单运算核（2 核实测即 2,1,1）"}
    result["mode"] = "measured"
    timing = json.loads(state.read_text()).get("timing", []) if state.is_file() else []
    if timing:
        # host 汇总取权威投影：重跑段的同物异名记录不重复计入
        auth = authoritative_timing(timing)
        result["host"] = {
            "wall_s": sum(t["wall_s"] for t in auth),
            "segments": len(auth),
            "transfer_log_s": sum(t["t_host"] for t in transfers),
            "transfer_points": len(transfers),
        }
    if single_counters is not None:
        if plan.strategy == "pp2":
            scaling = _layer_scaling(run_dir, scaling_json)
            pred_cycles = predict_pp2(single_counters, plan, scaling)
            result["params"]["layer_scaling"] = scaling
        else:
            pred_cycles = predict_tp2(single_counters, plan)
        predicted = compose(plan, pred_cycles, link, freq_mhz=freq_mhz)
        deviation = (abs(predicted["total_time_s"] - result["total_time_s"])
                     / result["total_time_s"]) if result["total_time_s"] else 0.0
        result["predicted_time_s"] = predicted["total_time_s"]
        result["deviation"] = deviation
        result["mode"] = "measured+predicted"
        # 标定来源 run 与被评估 run 相交 ⇒ 偏差属自证（in_sample），不作为验收
        calib_runs = {Path(s).name for s in
                      result["params"].get("layer_scaling", {}).get("source_runs", [])}
        if link_from_this_run and link.source.startswith("calibrated"):
            calib_runs.add(run_dir.name)
        result["in_sample"] = run_dir.name in calib_runs
        result["deviation_qualification"] = (
            "in_sample:标定与验证同源，偏差不能作为验收依据"
            if result["in_sample"] else "out_of_sample")
    if single_total_cycles:
        single_time_s = single_total_cycles / (freq_mhz * 1e6)
        result["speedup_vs_single"] = (
            single_time_s / result["total_time_s"] if result["total_time_s"] else 0.0)
        result["single_total_cycles_source"] = "input"
    out = run_dir / "results" / "cascade_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (out.parent / "cascade_eval.txt").write_text(report_text(result) + "\n")
    return result


def report_text(result: dict) -> str:
    """评估结果的可读文本（关键中间产物要能直接看）。"""
    lines = [f"策略 {result['mode']}：设备总时间 {result['total_time_s']:.6f}s"
             f"（{result['total_cycles']} cycles，{result['params']['frequency_mhz']}MHz）",
             f"tokens {result['tokens']}，吞吐 {result['tokens_per_s']:.3f} tokens/s",
             f"通信代价 {result['comm_cycles']:.0f} cycles，占比 {result['comm_ratio']:.3%}",
             f"每核 busy cycles: {result['per_core_busy_cycles']}"]
    if "deviation" in result:
        lines.append(f"预测 {result['predicted_time_s']:.6f}s，"
                     f"偏差 {result['deviation']:.1%}"
                     f"（{result['deviation_qualification']}）")
    if "speedup_vs_single" in result:
        lines.append(f"相对单核加速 {result['speedup_vs_single']:.3f}×")
    if "host" in result:
        lines.append(f"仿真 host 墙钟 {result['host']['wall_s']:.0f}s，"
                     f"{result['host']['segments']} 段，"
                     f"搬运记录 {result['host']['transfer_points']} 条")
    lines.append(f"数据来源 {result['measured_source']}；"
                 f"装载基线 {result['params']['baseline']['init_cycles']} cycles"
                 f"（{result['params']['baseline']['init_source']}）")
    lines.append(f"链路参数 {result['params']['link']}")
    lines.append(f"依赖结构 {result['structure']}")
    return "\n".join(lines)


def predict_tp2(single: dict[str, int], plan: CascadeBlueprint) -> dict:
    """预测模式（TP2）：单核 stage 级按层均摊后按半宽外推（assumption 口径）。

    每层拆 attn/mlp 两段、两核各持半宽：每段每核 ≈ 1/4 层 cycles
    （attn/mlp 均分假设）；head 分片每核 ≈ 半个 lm_head+argmax。
    成熟度标 assumption，实测后用 M2 运行复核系数。
    """
    cycles = {c: {} for c in range(plan.num_cores)}
    layers = max((s.layer_hi for s in plan.segments if s.kind == "attn_shard"), default=1)
    for seg in plan.segments:
        if seg.kind in ("attn_shard", "mlp_shard"):
            if seg.seg_id.startswith("prompt"):
                stage = single.get("prompt", 0)
            else:
                step = int(seg.seg_id.split(":")[1])
                stage = _decode_stage(single, step)
            value = stage // layers // 4
        else:
            value = _head_stage(single, int(seg.seg_id.split(":")[1])) // 2
        cycles[seg.core_id][seg.seg_id] = value
    return cycles


def main() -> int:
    """命令行入口：run_cascade.sh 第 6 步调用。

    单核基线取 runs/ours_e2e/run/performance_counters.txt（stage 级 cycles，
    与需求验收口径一致：基线数字取自已归档运行、不重测）。
    """
    import argparse

    from cascade.plan import from_bundle

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--strategy", choices=("pp2", "tp2"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--baseline-run", default="/data/fjg/ceva/runs/ours_e2e")
    parser.add_argument("--scaling-json", default=None,
                        help="外部层标定文件；out-of-sample 验证时指向别的 run 的系数")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    plan = from_bundle(args.bundle, run_dir, steps=args.steps)
    single, total = None, None
    baseline = Path(args.baseline_run) / "run" / "performance_counters.txt"
    if baseline.is_file():
        single = parse_performance_counters(baseline)
        # 按本次蓝图范围折算，避免「完整基线 ÷ 部分实测」的量纲失真。
        total = _baseline_total(single, plan)
    result = evaluate(run_dir, plan, None, single_counters=single,
                      single_total_cycles=total,
                      scaling_json=Path(args.scaling_json) if args.scaling_json else None)
    print(report_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
