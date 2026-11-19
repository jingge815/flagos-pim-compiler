"""cascade/eval.py 组合评估模型测试（设计 4.9、7.1：手推算例）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade.eval import (LinkParams, compose, parse_performance_counters,
                          predict_pp2)
from cascade.plan import from_bundle

PROMPT_HIDDEN = 4 * 4 * 2
DECODE_HIDDEN = 4 * 2
FREQ_MHZ = 1250


def _write_ini(path: Path, sections: list[dict]) -> None:
    lines = []
    for i, kv in enumerate(sections):
        lines.append(f"[{i}]")
        lines += [f"{k}={v}" for k, v in kv.items()]
    path.write_text("\n".join(lines) + "\n")


def _maps(run_dir: Path, core: int, *, logits: int = 4) -> None:
    d = run_dir / f"ddr_core{core}"
    d.mkdir(parents=True)
    base = 0x100000000 + core * (1 << 33)
    _write_ini(d / "inputs_map.ini", [
        {"input_file": f"/x/h{core}.bin", "input_name": "hidden_states",
         "address": hex(base + 0x2000), "size": PROMPT_HIDDEN},
        {"input_file": f"/x/row{core}.bin", "input_name": "embedding_row",
         "address": hex(base + 0x5000), "size": DECODE_HIDDEN}])
    _write_ini(d / "outputs_map.ini", [
        {"output_name": "output", "address": hex(base + 0x10000),
         "size": PROMPT_HIDDEN},
        {"output_name": "lm_output", "address": hex(base + 0x30000),
         "size": logits}])
    (d / "control_map.ini").write_text(
        f"[control]\ncore_id={core}\naddress={hex(base + 0x1000)}\n")


def _pp2_plan(tmp_path: Path, steps: int = 1):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "pp2", "num_cores": 2,
                                  "core_layers": [[0, 1], [1, 2]],
                                  "lm_head_core": 1}}))
    _maps(tmp_path, 0)
    _maps(tmp_path, 1)
    return from_bundle(bundle, tmp_path, steps=steps)


def _tp2_plan(tmp_path: Path, steps: int = 1):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "graphs": [], "cascade": {"strategy": "tp2", "num_cores": 2, "tp_width": 2,
                                  "core_layers": [[0, 1], [0, 1]],
                                  "lm_head_core": None}}))
    _maps(tmp_path, 0)
    _maps(tmp_path, 1)
    return from_bundle(bundle, tmp_path, steps=steps)


# 带宽 2.5e9 B/s 在 1.25GHz 下等于 2 字节/周期，搬运 cycles = bytes/2，便于手推。
LINK = LinkParams(latency_s=0.0, bandwidth_bytes_per_s=2.5e9, source="test")


def test_compose_pp2_hand_computed(tmp_path) -> None:
    """PP2 串行链：全部段 cycles 与全部搬运代价相加（手推 460 cycles）。"""
    plan = _pp2_plan(tmp_path, steps=1)
    cycles = {0: {"prompt:c0": 100, "decode:0:c0": 50},
              1: {"prompt:c1": 200, "decode:0:c1": 60, "head:0": 10, "head:1": 10}}
    result = compose(plan, cycles, LINK, freq_mhz=FREQ_MHZ)
    # prompt 步：100 + 32/2 + (200+10) + (2+8)/2 = 331；decode 步：50+4+70+5 = 129
    assert result["total_cycles"] == 331 + 129
    assert result["total_time_s"] == (331 + 129) / (FREQ_MHZ * 1e6)
    assert result["tokens"] == 2
    assert abs(result["tokens_per_s"] - 2 / result["total_time_s"]) < 1e-9
    assert result["per_core_busy_cycles"] == {0: 150, 1: 280}
    # 通信占比：搬运时间 / 总时间
    xfer = 16 + 5 + 4 + 5
    assert abs(result["comm_ratio"] - xfer / 460) < 1e-9


def test_compose_tp2_hand_computed(tmp_path) -> None:
    """TP2 层同步：max(attn)+归约+max(mlp)+归约，head 取 max 加 gather。"""
    plan = _tp2_plan(tmp_path, steps=1)
    cycles = {0: {"prompt:L0:attn:c0": 100, "prompt:L0:mlp:c0": 40,
                  "decode:0:L0:attn:c0": 50, "decode:0:L0:mlp:c0": 30,
                  "head:0:c0": 10, "head:1:c0": 10},
              1: {"prompt:L0:attn:c1": 80, "prompt:L0:mlp:c1": 60,
                  "decode:0:L0:attn:c1": 70, "decode:0:L0:mlp:c1": 30,
                  "head:0:c1": 12, "head:1:c1": 12}}
    result = compose(plan, cycles, LINK, freq_mhz=FREQ_MHZ)
    # prompt 层：(100 + 4*32/2) + (60 + 4*32/2) = 164+124 = 288
    # decode 层：(70 + 4*8/2) + (30 + 4*8/2) = 86+46 = 132
    # head 每相：max 12 + gather 2×2 + token 1 + embedding 2×4 = 25，两相共 50
    assert result["total_cycles"] == 288 + 132 + 50
    assert result["per_core_busy_cycles"] == {0: 240, 1: 264}


def test_parse_performance_counters(tmp_path) -> None:
    path = tmp_path / "performance_counters_core0.txt"
    path.write_text("prompt:c0: 100 cycles\ndecode:0:c0: 50 cycles\n")
    assert parse_performance_counters(path) == {"prompt:c0": 100, "decode:0:c0": 50}


def test_predict_pp2_from_single(tmp_path) -> None:
    """预测模式：单核 stage 级 cycles 按层均分外推两核分段。"""
    plan = _pp2_plan(tmp_path, steps=1)
    single = {"prompt": 600, "decode_0": 200, "lm_head": 20, "argmax_0": 4}
    cycles = predict_pp2(single, plan)
    assert cycles[0]["prompt:c0"] == 300 and cycles[1]["prompt:c1"] == 300
    assert cycles[0]["decode:0:c0"] == 100 and cycles[1]["decode:0:c1"] == 100
    assert cycles[1]["head:0"] == 20 + 4      # lm_head + argmax
    measured = {0: {"prompt:c0": 310, "decode:0:c0": 98},
                1: {"prompt:c1": 295, "decode:0:c1": 103, "head:0": 25, "head:1": 24}}
    result = compose(plan, measured, LINK, freq_mhz=FREQ_MHZ)
    pred = compose(plan, cycles, LINK, freq_mhz=FREQ_MHZ)
    deviation = abs(pred["total_time_s"] - result["total_time_s"]) / result["total_time_s"]
    assert deviation < 0.10                    # 均摊外推对合成数据的偏差


def test_evaluate_writes_json_with_provenance(tmp_path) -> None:
    """evaluate 落盘 cascade_eval.json：实测组合 + 预测偏差 + 参数溯源。"""
    from cascade.eval import evaluate

    plan = _pp2_plan(tmp_path, steps=1)
    # 本用例校验落盘与溯源，不校验标定效果：显式给 1.0，保持「每核一半」原始口径
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "layer_scaling.json").write_text(
        json.dumps({"prompt_per_layer_ratio": 1.0, "decode_per_layer_ratio": 1.0}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "performance_counters_core0.txt").write_text(
        "prompt:c0: 310 cycles\ndecode:0:c0: 98 cycles\n")
    (run / "performance_counters_core1.txt").write_text(
        "prompt:c1: 295 cycles\ndecode:0:c1: 103 cycles\n"
        "head:0: 25 cycles\nhead:1: 24 cycles\n")
    single = {"prompt": 600, "decode_0": 200, "lm_head": 20, "argmax": 4}
    result = evaluate(tmp_path, plan, LINK, single_counters=single,
                      single_total_cycles=600 + 200 + 24)
    out = json.loads((tmp_path / "results" / "cascade_eval.json").read_text())
    assert out["mode"] == "measured+predicted"
    assert out["params"]["frequency_source"].startswith("cnnha_upd_b.cfg")
    assert out["params"]["link"]["source"] == "test"
    assert out["params"]["three_layer_abstraction"] == [2, 1, 1]
    assert 0 < out["deviation"] < 0.10
    # PP2 是串行链，加速比接近 1（此处合成数据略慢，断言精确公式）
    single_s = (600 + 200 + 24) / (FREQ_MHZ * 1e6)
    assert abs(out["speedup_vs_single"] - single_s / out["total_time_s"]) < 1e-9
    assert 0.9 < out["speedup_vs_single"] < 1.1
    out["per_core_busy_cycles"] = {int(k): v for k, v in
                                   out["per_core_busy_cycles"].items()}
    assert result == out


def test_predict_pp2_maps_numbered_stages_in_order(tmp_path) -> None:
    """单核计数从 decode_6 起（prompt_len+1）：按顺序取，不退回第一个值。"""
    from cascade.eval import predict_pp2

    plan = _pp2_plan(tmp_path, steps=2)
    single = {"prompt": 1000, "lm_head": 10, "argmax": 1,
              "decode_6": 100, "lm_head_6": 11, "argmax_6": 2,
              "decode_7": 300, "lm_head_7": 12, "argmax_7": 3}
    cycles = predict_pp2(single, plan)
    assert cycles[0]["decode:0:c0"] == 50 and cycles[0]["decode:1:c0"] == 150
    # head:0 = prompt 首次 lm_head；head:1/2 依次对应 lm_head_6/7
    assert cycles[1]["head:0"] == 11
    assert cycles[1]["head:1"] == 13 and cycles[1]["head:2"] == 15


def test_parse_state_prefers_device_cycles(tmp_path) -> None:
    """state.json 自带 device_cycles 时直接用它（EndOfExecution 口径）。"""
    from cascade.eval import parse_state_cycles

    results = tmp_path / "results"
    results.mkdir()
    timing = [
        {"seg": "prompt:c0", "core": 0, "wall_s": 1.0, "cycles": 100,
         "device_cycles": 500},
        {"seg": "prompt:c1", "core": 1, "wall_s": 1.0, "cycles": 100,
         "device_cycles": 700},
    ]
    (results / "state.json").write_text(json.dumps({"timing": timing}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["cycles_source"] == "EndOfExecution"
    assert baseline["init_cycles"] == 500
    assert cycles == {0: {"prompt:c0": 0}, 1: {"prompt:c1": 200}}


def test_parse_state_cycles_subtracts_init(tmp_path) -> None:
    """逐段全局时钟减装载基线：标定文件优先，缺失时取最小值并标来源。"""
    from cascade.eval import parse_state_cycles

    results = tmp_path / "results"
    results.mkdir()
    timing = [
        {"seg": "prompt:c0", "core": 0, "wall_s": 10.0, "cycles": 500},
        {"seg": "prompt:c1", "core": 1, "wall_s": 11.0, "cycles": 520},
        {"seg": "head:0", "core": 1, "wall_s": 2.0, "cycles": 100},
    ]
    (results / "state.json").write_text(json.dumps({"timing": timing}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["init_cycles"] == 100
    assert baseline["init_source"] == "estimated:min_segment"
    assert baseline["cycles_source"] == "CycleEstimation"
    assert cycles == {0: {"prompt:c0": 400}, 1: {"prompt:c1": 420, "head:0": 0}}

    (results / "init_cycles.json").write_text(json.dumps({"cycles": 80}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["init_source"] == "calibrated:cmd0"
    assert cycles[0]["prompt:c0"] == 420

    # device_cycles（EndOfExecution 时钟）优先于 CycleEstimation 尾值
    (results / "device_cycles.json").write_text(json.dumps({
        "prompt:c0": {"clock": 900}, "prompt:c1": {"clock": 800},
        "head:0": {"clock": 700}}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["cycles_source"] == "EndOfExecution"
    assert cycles[0]["prompt:c0"] == 820          # 900 - 标定基线 80
    assert cycles[1]["head:0"] == 620

    # 守护文件里带 TIMER 执行 cycles 的段直接可用（不减装载基线）
    (results / "device_cycles.json").write_text(json.dumps({
        "prompt:c0": {"clock": 900, "exec_cycles": 850},
        "prompt:c1": {"clock": 800}, "head:0": {"clock": 700}}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert cycles[0]["prompt:c0"] == 850
    assert baseline["cycles_source"] == "EndOfExecution+TIMER"


def test_parse_state_mixed_timer_and_device(tmp_path) -> None:
    """混合口径：旧段只有 EndOfExecution、新段有 TIMER；基线取所有段最小值。"""
    from cascade.eval import parse_state_cycles

    results = tmp_path / "results"
    results.mkdir()
    timing = [
        {"seg": "prompt:c0", "core": 0, "wall_s": 1.0, "cycles": 1,
         "device_cycles": 500},
        {"seg": "prompt:c1", "core": 1, "wall_s": 1.0, "cycles": 1,
         "device_cycles": 300, "exec_cycles": 280},
    ]
    (results / "state.json").write_text(json.dumps({"timing": timing}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["init_cycles"] == 300          # 不得把 prompt:c0 自身当基线
    assert cycles[0]["prompt:c0"] == 200
    assert cycles[1]["prompt:c1"] == 280


def test_parse_legacy_mixed_init_from_all_clocks(tmp_path) -> None:
    """守护文件混合口径：只有一段缺 TIMER 时，基线仍取所有段时钟的最小值。"""
    from cascade.eval import parse_state_cycles

    results = tmp_path / "results"
    results.mkdir()
    timing = [
        {"seg": "prompt:c0", "core": 0, "wall_s": 1.0, "cycles": 1},
        {"seg": "prompt:c1", "core": 1, "wall_s": 1.0, "cycles": 1},
        {"seg": "head:0", "core": 1, "wall_s": 1.0, "cycles": 1},
    ]
    (results / "state.json").write_text(json.dumps({"timing": timing}))
    (results / "device_cycles.json").write_text(json.dumps({
        "prompt:c0": {"clock": 5000},
        "prompt:c1": {"clock": 3000, "exec_cycles": 2800},
        "head:0": {"clock": 2000, "exec_cycles": 1800}}))
    cycles, baseline = parse_state_cycles(tmp_path)
    assert baseline["init_cycles"] == 2000          # 含 TIMER 段时钟的最小值
    assert cycles[0]["prompt:c0"] == 3000
    assert cycles[1]["prompt:c1"] == 2800


def test_calibrate_link_recovers_transfer_params(tmp_path) -> None:
    """两次不同字节数的实测搬运应能拟合回 λ 与带宽。"""
    from cascade.eval import calibrate_link

    lam, width = 2e-6, 1.5e9
    transfers = [{"kind": "hidden_handoff", "bytes": n, "t_host": lam + n / width}
                 for n in (8192, 8388608, 8192, 8388608)]
    link = calibrate_link(transfers)
    assert link.source == "calibrated:host_file_channel:4points"
    assert abs(link.latency_s - lam) < 1e-9
    assert abs(link.bandwidth_bytes_per_s - width) / width < 1e-6
    # 单一字节数无法定标，保留 assumption
    weak = calibrate_link([{"kind": "x", "bytes": 8192, "t_host": 1e-4}] * 4)
    assert weak.source == "assumption:insufficient_points"


def test_evaluate_from_state_with_host_and_calibration(tmp_path) -> None:
    """state.json + 搬运日志：实测组合、host 墙钟与标定链路一起落盘。"""
    from cascade.eval import evaluate

    plan = _pp2_plan(tmp_path, steps=1)
    (tmp_path / "results").mkdir()
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
               "cycles": 1000 + i * 10} for i, s in enumerate(plan.segments)]
    (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
    (tmp_path / "run").mkdir()
    rows = [{"kind": "hidden_handoff", "bytes": 8192, "t_host": 2e-6 + 8192 / 1.5e9},
            {"kind": "hidden_handoff", "bytes": 8388608, "t_host": 2e-6 + 8388608 / 1.5e9}] * 2
    (tmp_path / "run" / "transfer_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    result = evaluate(tmp_path, plan, None, single_total_cycles=5000)
    assert result["measured_source"] == "state.json"
    assert result["host"]["segments"] == len(plan.segments)
    assert result["host"]["wall_s"] == 5.0 * len(plan.segments)
    assert result["params"]["link"]["source"] == "calibrated:host_file_channel:4points"
    assert (tmp_path / "results" / "cascade_eval.txt").read_text().count("设备总时间") == 1
    # 加速比 = 单核时间 / 级联时间，频率按 Hz 换算（此前误用 MHz，放大 1000 倍）
    expected = (5000 / (FREQ_MHZ * 1e6)) / result["total_time_s"]
    assert abs(result["speedup_vs_single"] - expected) < 1e-9


def test_predict_tp2_halfwidth(tmp_path) -> None:
    """TP2 预测：每段每核 1/4 层 cycles，head 半宽——层内 max 相加约减半。"""
    from cascade.eval import compose, predict_tp2

    plan = _tp2_plan(tmp_path, steps=1)
    single = {"prompt": 400, "decode_0": 200, "lm_head": 20, "argmax": 4}
    cycles = predict_tp2(single, plan)
    assert cycles[0]["prompt:L0:attn:c0"] == 100     # 400 / 1 层 / 4
    assert cycles[0]["decode:0:L0:mlp:c0"] == 50
    assert cycles[1]["head:0:c1"] == 12             # (20+4)/2
    result = compose(plan, cycles, LINK, freq_mhz=FREQ_MHZ)
    # prompt 层：max(100)+64+max(100)+64 = 328；decode 层 = 50+16+50+16 = 132
    # head 两相：(12+4+1+8)×2 = 50；总 510，约为「单核同口径」的一半。
    assert result["total_cycles"] == 328 + 132 + 50


def test_baseline_total_scoped_to_plan(tmp_path) -> None:
    """基线总周期必须按蓝图范围折算，否则「加速比」会拿完整基线除以部分实测。"""
    from cascade.eval import _baseline_total

    single = {"prompt": 1000, "full_execution": 9999,
              "lm_head": 10, "argmax": 1,
              "decode_6": 100, "lm_head_6": 10, "argmax_6": 1,
              "decode_7": 100, "lm_head_7": 10, "argmax_7": 1}
    (tmp_path / "s0").mkdir()
    (tmp_path / "s1").mkdir()
    assert _baseline_total(single, _pp2_plan(tmp_path / "s0", steps=0)) == 1000 + 11
    assert _baseline_total(single, _pp2_plan(tmp_path / "s1", steps=1)) == 1000 + 100 + 11 + 11


def test_baseline_total_tp2_scoped_to_plan(tmp_path) -> None:
    """TP2 蓝图基线同口径折算：段 kind 不是 decode_layers，步数按段名前缀取。

    旧实现只认 decode_layers 段，TP2 蓝图恒得 0 步、基线被截成 prompt+1 head
    （round02 问题4），speedup_vs_single 的分母随之失真。"""
    from cascade.eval import _baseline_total

    single = {"prompt": 1000, "lm_head": 10, "argmax": 1}
    for k in range(10):
        single[f"decode_{6 + k}"] = 100
        single[f"lm_head_{6 + k}"] = 10
        single[f"argmax_{6 + k}"] = 1
    for steps, name in ((0, "t0"), (1, "t1"), (10, "t10")):
        (tmp_path / name).mkdir()
    assert _baseline_total(single, _tp2_plan(tmp_path / "t0", steps=0)) == 1000 + 11
    assert _baseline_total(single, _tp2_plan(tmp_path / "t1", steps=1)) == 1000 + 100 + 22
    # 10 步：prompt + 10 decode + 11 个 head 的手推值
    assert _baseline_total(single, _tp2_plan(tmp_path / "t10", steps=10)) \
        == 1000 + 10 * 100 + 11 * 11


def test_predict_pp2_layer_scaling(tmp_path) -> None:
    """层数标定系数：1.0 是原始假设口径，0.68 复现实测的「16 层每层更便宜」。"""
    from cascade.eval import predict_pp2

    plan = _pp2_plan(tmp_path, steps=1)
    single = {"prompt": 1000, "decode_6": 200, "lm_head": 20, "argmax": 4}
    base = predict_pp2(single, plan)
    assert base[0]["prompt:c0"] == 500 and base[1]["decode:0:c1"] == 100
    scaled = predict_pp2(single, plan, {"prompt_per_layer_ratio": 0.5,
                                        "decode_per_layer_ratio": 1.0})
    assert scaled[0]["prompt:c0"] == 250      # 500 × 0.5
    assert scaled[1]["decode:0:c1"] == 100    # decode 不受影响


def test_layer_scaling_repo_default(tmp_path) -> None:
    """本次运行没有标定文件时，退到仓库级实测默认，而不是退化成 1.0 的粗糙假设。"""
    from cascade.eval import _layer_scaling

    data = _layer_scaling(tmp_path)
    assert 0.5 < data["prompt_per_layer_ratio"] < 0.9
    # 仓库默认系数的来源 run 必须可追溯（含 source_runs 清单）
    assert "source_runs" in data and data["source_runs"]


def test_correction_requires_provenance(tmp_path) -> None:
    """cycles_correction 必须带 value/derived_from/measured_at，缺溯源直接报错。"""
    from cascade.eval import parse_state_cycles

    plan = _pp2_plan(tmp_path)          # 布局只建一次，多次调用会撞已存在目录

    def _state_with_correction(corr: dict) -> Path:
        (tmp_path / "results").mkdir(exist_ok=True)
        timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
                   "cycles": 1000, "exec_cycles": 1000} for s in plan.segments]
        (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
        if corr is not None:
            (tmp_path / "results" / "cycles_correction.json").write_text(json.dumps(corr))
        return tmp_path

    # 旧格式 {seg: int}：无推导来源，拒绝采用
    with pytest.raises(ValueError, match="derived_from"):
        parse_state_cycles(_state_with_correction({"prompt:c0": 176964315}))
    # 新格式缺 measured_at 同样拒绝
    with pytest.raises(ValueError, match="measured_at"):
        parse_state_cycles(_state_with_correction(
            {"prompt:c0": {"value": 100, "derived_from": "state.json:xxx"}}))
    # 合法格式：扣减生效且可追溯
    cycles, _ = parse_state_cycles(_state_with_correction(
        {"prompt:c0": {"value": 400, "derived_from": "state.json 两核对照",
                       "measured_at": "2026-11-17"}}))
    assert cycles[0]["prompt:c0"] == 600


def test_evaluate_marks_in_sample_calibration(tmp_path) -> None:
    """标定来源 run 与被评估 run 相交时：偏差标注 in_sample，不作为验收依据。"""
    from cascade.eval import evaluate

    plan = _pp2_plan(tmp_path, steps=1)
    (tmp_path / "results").mkdir()
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
               "cycles": 1000, "exec_cycles": 1000} for s in plan.segments]
    (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
    scaling = {"prompt_per_layer_ratio": 1.0, "decode_per_layer_ratio": 1.0,
               "source_runs": [f"{tmp_path.name}"]}       # 标定自本次运行自身
    (tmp_path / "results" / "layer_scaling.json").write_text(json.dumps(scaling))
    result = evaluate(tmp_path, plan, LINK, single_counters={"prompt": 2000})
    assert result["in_sample"] is True
    assert "in_sample" in result["deviation_qualification"]
    assert "in_sample" in (tmp_path / "results" / "cascade_eval.txt").read_text()
    # 标定来自别的 run：out-of-sample，偏差可作为验收
    scaling["source_runs"] = ["runs/other_run"]
    (tmp_path / "results" / "layer_scaling.json").write_text(json.dumps(scaling))
    result = evaluate(tmp_path, plan, LINK, single_counters={"prompt": 2000})
    assert result["in_sample"] is False
    assert "out_of_sample" in result["deviation_qualification"]


def test_scaling_override_requires_source_runs(tmp_path) -> None:
    """外部标定文件必须声明来源（round03 问题3）。

    无 source_runs 的外部文件静默按空表处理会把自拟合系数伪装成
    out-of-sample 验收——与 _load_correction 缺 provenance 即拒绝同口径。"""
    import pytest

    from cascade.eval import _layer_scaling

    bad = tmp_path / "ext.json"
    bad.write_text(json.dumps({"prompt_per_layer_ratio": 0.68,
                               "decode_per_layer_ratio": 0.94}))
    with pytest.raises(ValueError, match="source_runs"):
        _layer_scaling(tmp_path, bad)
    # 声明来源的 measured 文件照常生效
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"prompt_per_layer_ratio": 0.68,
                                "decode_per_layer_ratio": 0.94,
                                "source_runs": ["runs/single_ours"]}))
    assert _layer_scaling(tmp_path, good)["source_runs"] == ["runs/single_ours"]
    # 显式声明 assumption 口径可接受（诚实声明非实测）
    assume = tmp_path / "assume.json"
    assume.write_text(json.dumps({"prompt_per_layer_ratio": 1.0,
                                  "decode_per_layer_ratio": 1.0,
                                  "source": "assumption:half_per_core"}))
    assert _layer_scaling(tmp_path, assume)["source_runs"] == []


def test_evaluate_override_declared_source_marks_in_sample(tmp_path) -> None:
    """外部标定声明来源含被评估 run 时：in_sample 判定对 override 同样生效。"""
    from cascade.eval import evaluate

    plan = _pp2_plan(tmp_path, steps=1)
    (tmp_path / "results").mkdir()
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
               "cycles": 1000, "exec_cycles": 1000} for s in plan.segments]
    (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
    ext = tmp_path / "ext.json"
    ext.write_text(json.dumps({"prompt_per_layer_ratio": 0.68,
                               "decode_per_layer_ratio": 0.94,
                               # 对被评估 run 自拟合的系数被如实标了来源
                               "source_runs": [f"runs/{tmp_path.name}"]}))
    result = evaluate(tmp_path, plan, LINK, single_counters={"prompt": 2000},
                      scaling_json=ext)
    assert result["in_sample"] is True
    assert "in_sample" in result["deviation_qualification"]


def test_evaluate_host_sum_excludes_duplicate_segments(tmp_path) -> None:
    """timing 里同物异名的重复段（decode:0:lm 与 head:1）：host 汇总只取权威投影。"""
    from cascade.eval import evaluate

    plan = _pp2_plan(tmp_path, steps=1)
    (tmp_path / "results").mkdir()
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
               "cycles": 1000, "exec_cycles": 1000} for s in plan.segments]
    # 追加旧脚本口径的重复记录：decode:0:lm 与 head:1 同物异名，重跑发生在后
    timing.append({"seg": "decode:0:lm", "core": 1, "wall_s": 50.0,
                   "cycles": 900, "exec_cycles": 900})
    timing.append({"seg": "head:1", "core": 1, "wall_s": 7.0,
                   "cycles": 700, "exec_cycles": 700})
    (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
    result = evaluate(tmp_path, plan, LINK)
    # 权威段数 = 蓝图段数（8 条记录里 2 条是重复），墙钟只算每段最后一次
    assert result["host"]["segments"] == len(plan.segments)
    assert result["host"]["wall_s"] == 5.0 * (len(plan.segments) - 1) + 7.0


def test_eval_cli_tp2_smoke(tmp_path) -> None:
    """`python -m cascade.eval --strategy tp2` 能自举（predict_tp2 曾定义在
    __main__ 之后，模块方式执行到 main() 时 NameError）。"""
    import subprocess
    import sys

    plan = _tp2_plan(tmp_path, steps=1)
    (tmp_path / "results").mkdir()
    timing = [{"seg": s.seg_id, "core": s.core_id, "wall_s": 5.0,
               "cycles": 1000, "exec_cycles": 1000} for s in plan.segments]
    (tmp_path / "results" / "state.json").write_text(json.dumps({"timing": timing}))
    baseline = tmp_path / "baseline"
    (baseline / "run").mkdir(parents=True)
    (baseline / "run" / "performance_counters.txt").write_text(
        "prompt: 400 cycles\ndecode_6: 200 cycles\nlm_head: 20 cycles\nargmax: 4 cycles\n")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "cascade.eval",
         "--run-dir", str(tmp_path), "--bundle", str(tmp_path / "bundle.json"),
         "--strategy", "tp2", "--steps", "1", "--baseline-run", str(baseline)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stderr
    assert "NameError" not in proc.stderr
