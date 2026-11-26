"""batch 扫参汇总报告的聚合判据：档位解析、逐档表、守恒校验、瓶颈与推荐。"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_full_pipeline import SweepRun, aggregate_batch_report, parse_batch_list


def test_parse_batch_list() -> None:
    assert parse_batch_list("1,2,4,8") == (1, 2, 4, 8)
    assert parse_batch_list("4") == (4,)
    assert parse_batch_list("1, 4, 4") == (1, 4)  # 去重升序
    assert parse_batch_list("8,2") == (2, 8)
    with pytest.raises(ValueError):
        parse_batch_list("1,0")   # batch 必须为正
    with pytest.raises(ValueError):
        parse_batch_list("a,b")


def _summary(total_time: float, output_tps: float, ttft: float, tpot: float) -> dict:
    return {
        "total_time_s": total_time,
        "throughput_output_tokens_per_s": output_tps,
        "p50_ttft_s": ttft,
        "p95_ttft_s": ttft * 1.2,
        "p50_mean_tpot_s": tpot,
        "p95_mean_tpot_s": tpot * 1.1,
    }


def _metrics(wall: float, busy: float, wait: float, sync: float,
             wbpt: float, tokens: int = 100, overlap: float = 0.0) -> dict:
    """守恒四项 busy_pure+组内+组间+sync=wall 的 batch_metrics 样例。

    wait 参数按「组内争用」解释、sync 为余项，与 genesim 侧四项口径一致；
    链路三字段与 result_schema 按当前产物形状携带（口径门禁的判据集）。
    """
    from scripts.run_full_pipeline import BATCH_METRICS_RESULT_SCHEMA

    return {
        "group_size": 0,
        "result_schema": BATCH_METRICS_RESULT_SCHEMA,
        "groups": [{
            "request_index": 0,
            "wall_time_s": wall,
            "pu_busy_union_s": busy + wait,
            "busy_pure_s": busy,
            "intra_group_contention_s": wait,
            "inter_group_contention_s": 0.0,
            "sync_gap_s": sync,
            "inter_group_overlap_s": overlap,
            "ttft_s": 0.4,
            "weight_bytes": wbpt * tokens,
            "generated_tokens": tokens,
            "link_wait_s": 0.0,
            "intra_link_contention_s": 0.0,
            "inter_link_contention_s": 0.0,
        }],
        "preemption_estimate": {
            "mode": "analytical",
            "estimate_s": 0.5,
        },
    }


def _runs() -> list[SweepRun]:
    """B=1..8：吞吐升到 B=4 后触容量墙走平，权重字节/token 单调下降。"""
    return [
        SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05),
                 _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)),
        SweepRun(2, 1, _summary(10.5, 19.0, 0.6, 0.06),
                 _metrics(10.5, 9.0, 0.8, 0.7, 500.0)),
        SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08),
                 _metrics(11.0, 10.0, 0.5, 0.5, 250.0)),
        SweepRun(8, 1, _summary(12.0, 30.5, 1.5, 0.12),
                 _metrics(12.0, 11.0, 0.4, 0.6, 130.0)),
    ]


def test_batch_parser_single_source(monkeypatch) -> None:
    """档位解析单点定义：export CLI 与 run_full_pipeline 共用同一实现。"""
    import importlib.util

    script = Path(__file__).resolve().parent.parent / "scripts" / "export_pp_placement.py"
    spec = importlib.util.spec_from_file_location("export_pp_placement_single_src", script)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)

    from scripts import run_full_pipeline as fp

    assert cli.parse_batch_list is fp.parse_batch_list
    # 两入口校验同口径：argparse 侧非法档位（非纯数字如 +4）按同一实现拒绝。
    monkeypatch.setattr(sys, "argv",
                        ["export_pp_placement.py", "--num-stages", "4",
                         "--batch", "+4,1"])
    with pytest.raises(SystemExit):
        cli.parse_args()


def test_report_rows_and_best_batch() -> None:
    """逐档表包含 B/G/吞吐/时延/权重字节，推荐吞吐最优档 B=4。"""
    text = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=4)
    for needle in ("B=1", "B=2", "B=4", "B=8", "10.00", "30.00", "1000", "250"):
        assert needle in text, needle
    assert "B=4" in text.split("推荐")[1]  # 推荐段落指向 B=4


def test_report_conservation_check() -> None:
    """守恒四项残差在容差内标通过；人为破坏后标注不守恒。"""
    text = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=4)
    assert "守恒" in text
    runs = _runs()
    broken = _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)
    broken["groups"][0]["busy_pure_s"] = 5.0  # 四项之和 != wall
    runs[0] = SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), broken)
    text = aggregate_batch_report(runs, tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=4)
    assert "不守恒" in text


def test_report_marks_capacity_wall_and_weight_saturation() -> None:
    """容量墙档位与权重字节机制饱和（tile_m 封顶）都要标注出来。

    机制口径：该档 tile_m(B) < B 且下一档 tile_m 不再增长才判饱和；只扫到
    B=8、更高档未扫时不能凭「本档压线」下结论。
    """
    runs = _runs() + [SweepRun(16, 1, _summary(13.0, 31.0, 1.6, 0.13),
                                _metrics(13.0, 11.0, 0.4, 0.6, 130.0))]
    text = aggregate_batch_report(
        runs, tile_m_by_batch={1: 8, 2: 8, 4: 4, 8: 4, 16: 4}, capacity_wall_at=4)
    assert "容量墙" in text          # B=4 处触墙
    assert "1/tile_m_max" in text    # 摊薄上限随 tile_m 最大值打印
    assert "机制饱和" in text and "B=8" in text  # B=8 起 tile_m 封顶在 4
    # 本档压线（tile_m==B）不判饱和：B=4 不在饱和集合里。
    assert "B∈[4]" not in text


def test_report_recommendation_carries_switch_identity() -> None:
    """R17-06 反例沉淀：推荐行携带开关身份（重叠/副本/钉扎/核数/指纹）。

    同 B/G 一开一关两档（关 1.09、开 1.11 tokens/s）输入聚合，推荐必须
    带出胜出档的重叠=开，不能再只写 B=4, G=1。
    """
    runs = [
        SweepRun(4, 1, _summary(10.0, 1.09, 0.5, 0.05),
                 _metrics(10.0, 8.0, 1.0, 1.0, 250.0),
                 pipeline_overlap=False, cores=64, fingerprint="aaaa1111"),
        SweepRun(4, 1, _summary(9.8, 1.11, 0.5, 0.05),
                 _metrics(9.8, 8.0, 1.0, 0.8, 250.0),
                 pipeline_overlap=True, cores=64, fingerprint="bbbb2222"),
    ]
    text = aggregate_batch_report(runs, tile_m_by_batch={4: 4})
    best_line = text.split("吞吐最优档")[1].splitlines()[0]
    assert "重叠=开" in best_line
    assert "副本=1" in best_line and "KV head 钉扎=关" in best_line
    assert "核数=64" in best_line
    # 胜出档的产物指纹进推荐行；落选档（关态）的指纹不进吞吐推荐行。
    assert "产物指纹=bbbb2222" in best_line
    assert "aaaa1111" not in best_line
    # 时延最优档同样携带自身身份（本夹具为关态档）。
    latency_line = text.split("时延最优档")[1].splitlines()[0]
    assert "重叠=关" in latency_line and "aaaa1111" in latency_line


def test_report_marks_last_tier_mechanism_saturation() -> None:
    """R17-07：末档 tile_m 封顶（无更高档可对照）也判机制饱和。

    {16:16, 32:16}（round17 探针）：B=16 压线且 B=32 同值、B=32 是末档且
    tile_m<B——两档都应进饱和集合，报告不再输出「未见机制饱和」。
    """
    runs = [
        SweepRun(b, 1, _summary(13.0, 1.0, 0.5, 0.05),
                 _metrics(13.0, 11.0, 0.4, 0.6, 130.0))
        for b in (16, 32)
    ]
    text = aggregate_batch_report(runs, tile_m_by_batch={16: 16, 32: 16})
    assert "机制饱和" in text
    assert "B=16" in text.split("机制饱和")[1]
    assert "B=32" in text.split("机制饱和")[1]
    assert "未见明确瓶颈" not in text


def test_report_saturation_distinguishes_window_flatness() -> None:
    """R17-07：观察窗内吞吐持平与内核上限证明分开措辞。

    {2:2, 4:4, 8:8}（tile_m 仍随 B 增长、末档压线）+ 末两档吞吐持平：
    不得判机制饱和，只给「观察窗内吞吐持平」的观察性结论。
    """
    runs = [
        SweepRun(2, 1, _summary(10.0, 1.0, 0.5, 0.05),
                 _metrics(10.0, 8.0, 1.0, 1.0, 500.0)),
        SweepRun(4, 1, _summary(10.0, 1.2, 0.5, 0.05),
                 _metrics(10.0, 8.0, 1.0, 1.0, 250.0)),
        SweepRun(8, 1, _summary(10.0, 1.21, 0.5, 0.05),
                 _metrics(10.0, 8.0, 1.0, 1.0, 130.0)),
    ]
    text = aggregate_batch_report(runs, tile_m_by_batch={2: 2, 4: 4, 8: 8})
    # 不判机制饱和（无内核上限证明），只给观察口径的持平结论。
    assert "内核上限证明）：" not in text
    assert "观察窗内吞吐持平（观察口径）" in text
    assert "不构成摊薄封顶结论" in text


def test_tier_capacity_status_hard_fails_and_alarms(tmp_path) -> None:
    """R17-08 反例沉淀：容量核验读 pim_capacity_metrics 的 fits/overflow_*。

    旧口径 `.get("capacity", {}).get("overflow_bytes", 0)` 读 batch_metrics
    里不存在的键，缺字段构造性得 0、零溢出恒「通过」。修复后：缺档/缺字段
    硬失败；非零溢出样例触发报警（硬失败），零溢出样例才通过。
    """
    from scripts.run_full_pipeline import (
        StepFailed, assert_tier_capacity_no_overflow, tier_capacity_status)

    # 真实形态样例（final2 b4g1 同构）：fits=true、零溢出 → 通过。
    ok_dir = tmp_path / "b4g1"
    ok_dir.mkdir()
    (ok_dir / "pim_capacity_metrics.json").write_text(json.dumps({
        "fits": True, "overflow_tensor_pu_count": 0,
        "overflow_cluster_count": 0}))
    assert tier_capacity_status(ok_dir) == (True, 0, 0)
    assert_tier_capacity_no_overflow(ok_dir)

    # 构造的非零溢出样例：检查必须报警（硬失败），不是打印后放行。
    bad_dir = tmp_path / "b4g2"
    bad_dir.mkdir()
    (bad_dir / "pim_capacity_metrics.json").write_text(json.dumps({
        "fits": False, "overflow_tensor_pu_count": 2,
        "overflow_cluster_count": 1}))
    assert tier_capacity_status(bad_dir) == (False, 2, 1)
    with pytest.raises(StepFailed, match="容量溢出"):
        assert_tier_capacity_no_overflow(bad_dir)

    # 缺字段（旧 batch_metrics 的 capacity 键根本不存在）→ 硬失败。
    miss_dir = tmp_path / "b8g1"
    miss_dir.mkdir()
    (miss_dir / "pim_capacity_metrics.json").write_text(json.dumps({
        "fits": True, "overflow_tensor_pu_count": 0}))
    with pytest.raises(StepFailed, match="缺容量字段"):
        tier_capacity_status(miss_dir)
    # 缺整档 → 硬失败。
    with pytest.raises(StepFailed, match="缺档"):
        tier_capacity_status(tmp_path / "b16g1")


def test_report_bottleneck_consumes_link_wait() -> None:
    """R17-02 报告侧：瓶颈段消费链路等待，时间构成表带链路争用列。"""
    metrics = _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)
    metrics["groups"][0].update({
        "inter_group_contention_s": 9.0,
        "sync_gap_s": 1.0,
        "link_wait_s": 9.0,
        "intra_link_contention_s": 0.0,
        "inter_link_contention_s": 9.0,
    })
    runs = [SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), metrics,
                     link_wait_total_s=9.0)]
    text = aggregate_batch_report(runs, tile_m_by_batch={1: 8})
    assert "链路争用" in text
    assert "全局链路等待合计 9.0s" in text
    assert "组间链路争用最大 9.0s" in text
    # 时间构成表：链路列分列，PU 份额 = 总额减链路份额。
    assert "组内链路争用" in text and "组间链路争用" in text
    comp = text.split("时间构成归因")[1]
    assert "9.000" in comp


def test_report_without_metrics_marks_missing() -> None:
    """缺 batch_metrics 的档（B=1 老口径）标注缺失，不崩。"""
    runs = [SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), None)]
    text = aggregate_batch_report(runs)
    assert "B=1" in text
    assert "缺失" in text


def test_report_monotonic_decreasing_weight_bytes_highlighted() -> None:
    """weight_bytes_per_token 随 B 单调下降时报告给出界对照列。"""
    text = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=4)
    assert "weight_bytes/token" in text


def test_report_capacity_column_and_wall_source() -> None:
    """逐档表含「容量占用」列（编译期容量表摘录）；墙位标注来源。"""
    runs = _runs()
    for i, run in enumerate(runs):
        run.capacity_note = f"{(i + 1) * 20}%（编译期）"
    text = aggregate_batch_report(
        runs, tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=4,
        capacity_wall_source="编译期容量表判定")
    assert "容量占用" in text
    assert "20%（编译期）" in text and "80%（编译期）" in text
    assert "编译期容量表判定" in text


def test_report_all_wall_no_successful_runs() -> None:
    """全墙终态契约：空 runs 只出墙位与瓶颈段，不崩溃、不产生推荐。"""
    text = aggregate_batch_report([], capacity_wall_at=16,
                                  capacity_wall_source="运行时捕获")
    assert "容量墙" in text and "B=16" in text   # 推荐段的墙位行保留
    assert "无成功档" in text
    assert "吞吐最优档" not in text and "时延最优档" not in text
    assert "2 副本" not in text   # DP 段无吞吐锚点不出数


def test_report_dp_estimate_uses_capacity_wall() -> None:
    """有容量墙输入时 DP 解析估算出数，不再整段跳过。"""
    text = aggregate_batch_report(
        _runs(), tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8}, capacity_wall_at=8,
        capacity_wall_source="编译期容量表判定")
    assert "2 副本" in text and "4 副本" in text
    assert "缺容量墙输入" not in text


def test_report_dma_ratio_and_zone_columns() -> None:
    """逐档表输出 DMA 占比与按 zone 的搬运字节构成（R7 计算/DMA 分解）。

    DMA 占比分母是「计算+搬运」服务预算（逐算子累加，与逐组守恒断言同
    口径）；busy 并集是时间轴测度，多 PU 并行时远小于服务预算，不能作
    分母（e2e 实测会出 200%+ 的占比）。
    """
    metrics = _metrics(10.0, 6.0, 2.0, 2.0, 1000.0)
    metrics["groups"][0].update({
        "compute_time_s": 6.0,
        "memory_time_s": 3.0,
        "dma_bytes": 900.0,
        "zone_bytes": {"weight": 300.0, "activation": 600.0, "kv": 0.0},
    })
    text = aggregate_batch_report(
        [SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), metrics)])
    header = next(line for line in text.splitlines() if line.startswith("| B |"))
    assert "DMA 占比" in header and "zone 搬运" in header
    assert "33.3%" in text       # memory/(compute+memory) = 3/9，非 3/6
    assert "33.3%/66.7%/0.0%" in text   # 权重/激活/KV 构成（一位小数）
    # 老口径产物（无 DMA 字段）列标「—」不崩。
    text_old = aggregate_batch_report(_runs())
    assert "—" in text_old


def test_report_bubble_ratio_column() -> None:
    """逐档表直读 summary 的流水气泡占比（R8 气泡收缩判据）。"""
    runs = _runs()
    for run in runs:
        run.summary["pipeline_bubble_ratio"] = 0.25
    text = aggregate_batch_report(runs, tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8})
    header = next(line for line in text.splitlines() if line.startswith("| B |"))
    assert "气泡占比" in header
    assert "25.0%" in text
    # 缺字段的档标「—」，与组间重叠两列共同判定气泡收缩。
    runs[0].summary.pop("pipeline_bubble_ratio")
    text = aggregate_batch_report(runs, tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8})
    assert text.count("—") >= 1


def test_report_preemption_uses_migration_wording() -> None:
    """抢占段按迁移口径出数：分子是一次换出/换入的整组 KV，记录槽数。"""
    metrics = _metrics(10.0, 6.0, 2.0, 2.0, 1000.0)
    metrics["preemption_estimate"] = {
        "mode": "analytical",
        "estimate_s": 0.5,
        "swap_out_s": 0.3,
        "swap_in_s": 0.2,
        "switch_cost_s": 0.0,
        "swap_slots": 4,
        "kv_migration_bytes_per_swap": 4128.0,
    }
    text = aggregate_batch_report(
        [SweepRun(4, 1, _summary(10.0, 10.0, 0.5, 0.05), metrics)])
    assert "一次换出/换入各 4 槽 KV" in text
    assert "0.500s" in text
    # 分项与总值分别可见（R17-05）。
    assert "换出 0.300s" in text and "换入 0.200s" in text
    assert "calibrated=false" in text


def test_report_wall_hits_list_failed_combos() -> None:
    """容量墙命中直接列触墙 (B,G) 组合，不再从成功档反推。"""
    runs = [SweepRun(4, 1, _summary(10.0, 10.0, 0.5, 0.05), None)]
    text = aggregate_batch_report(
        runs, capacity_wall_hits=[(4, 4), (8, 1)])
    assert "(B=4, G=4)" in text and "(B=8, G=1)" in text
    # 成功执行的 B=4,G=1 不是命中档；旧口径「命中档 [4, 8]」类反推不再出现。
    assert "(B=4, G=1)" not in text
    assert "命中档" not in text


def test_report_wall_hits_constrain_groups_not_batch_wall() -> None:
    """部分 G 触墙只约束并发推荐：不升格成 B 墙、DP 不按它折容量。"""
    runs = [SweepRun(4, 1, _summary(10.0, 10.0, 0.5, 0.05), None)]
    text = aggregate_batch_report(
        runs, capacity_wall_hits=[(4, 4)], tile_m_by_batch={4: 8})
    assert "并发" in text            # G 档受容量约束的提示行
    assert "容量墙：B≥" not in text  # 不再把 (4,4) 说成 B 墙
    assert "DP" in text and "缺容量墙输入" in text  # 无 B 墙输入，DP 略


def test_report_inter_group_overlap_column() -> None:
    """组间流水重叠进逐档表与逐组归因表（G 曲线的重叠收益可直读）。"""
    runs = _runs()
    for run in runs:
        for group in run.batch_metrics["groups"]:
            group["inter_group_overlap_s"] = 137.0
    text = aggregate_batch_report(runs, tile_m_by_batch={1: 8, 2: 8, 4: 8, 8: 8})
    assert "组间重叠" in text
    assert "137.00" in text   # 逐档表（组均）
    assert "137.000" in text  # 逐组归因表


def test_report_bound_columns_same_dimension() -> None:
    """界列与 wbpt 同口径：decode wbpt 比值 vs 1/min(B,tile_m(B)) 逐档对照。"""
    runs = _runs()
    # decode 相位桶：B=1..8 的 decode wbpt = 500/250/125/62.5（比值 1/0.5/0.25/0.125）。
    decode_wbpts = {1: 500.0, 2: 250.0, 4: 125.0, 8: 62.5}
    for run in runs:
        for group in run.batch_metrics["groups"]:
            value = decode_wbpts[run.batch_size]
            group["weight_bytes_decode"] = value * 50
            group["decode_tokens"] = 50
    text = aggregate_batch_report(
        runs, tile_m_by_batch={1: 8, 2: 8, 4: 4, 8: 4})
    header = next(line for line in text.splitlines()
                  if line.startswith("| B |"))
    assert "decode wbpt" in header and "相对基准" in header
    # 比值列与界列逐档同口径：B=2 是 0.500 vs 界 0.500；B=8 是 0.125 vs 界 0.250。
    assert "| 0.500 | 0.500 |" in text
    assert "| 0.125 | 0.250 |" in text
    # 缺相位桶的老口径档不再把绝对字节数与比值并排。
    legacy = [SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), None),
              SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08), _metrics(
                  11.0, 10.0, 0.5, 0.5, 250.0))]
    legacy_text = aggregate_batch_report(legacy, tile_m_by_batch={1: 8, 4: 4})
    assert "| — | — |" in legacy_text  # 缺 metrics 的档比值与界按缺省呈现


# —— 扫参接线的可单测部分：closed batch trace 合成与仿真配置写入 ——


def test_closed_batch_trace_synthesis(tmp_path) -> None:
    """closed batch trace：首请求复制 n 份，等长、id 唯一、落 JSON。"""
    import json

    from scripts.run_full_pipeline import write_closed_batch_trace

    source = {
        "description": "src",
        "requests": [
            {"request_id": "a", "input_tokens": [1, 2, 3], "output_length": 4,
             "arrival_time_ns": 500, "is_prefill": True, "seq_len": 3},
            {"request_id": "b", "input_tokens": [1], "output_length": 9,
             "arrival_time_ns": 700, "is_prefill": True, "seq_len": 1},
        ],
    }
    out = write_closed_batch_trace(source, num_requests=4, out_path=tmp_path / "closed.trace.json")
    data = json.loads(Path(out).read_text())
    assert len(data["requests"]) == 4
    assert all(r["input_tokens"] == [1, 2, 3] and r["output_length"] == 4
               and r["arrival_time_ns"] == 500 for r in data["requests"])
    assert len({r["request_id"] for r in data["requests"]}) == 4


def test_ceva_sim_config_batch_groups(tmp_path) -> None:
    """扫参配置写入组大小/组数/请求数/trace 路径；缺省不写多余键。"""
    import yaml

    from scripts.run_full_pipeline import ceva_sim_config

    src = tmp_path / "src.yaml"
    src.write_text("scheduler:\n  max_inflight_requests: 1\ntrace:\n  num_requests: 10\n")
    conf_dir = tmp_path / "conf"

    out = ceva_sim_config(src, conf_dir, tmp_path / "placement.json",
                          batch_size=4, groups=2, num_requests=8,
                          trace_path="traces/closed.trace.json")
    loaded = yaml.safe_load(out.read_text())
    assert loaded["scheduler"]["execution_microbatch_size"] == 4
    assert loaded["scheduler"]["max_inflight_requests"] == 2
    assert loaded["scheduler"]["compiler_placement_file"].endswith("placement.json")
    assert loaded["trace"]["num_requests"] == 8
    assert loaded["trace"]["path"] == "traces/closed.trace.json"

    # 缺省调用（经典单档路径）只写 placement，不引入其余键。
    plain = ceva_sim_config(src, conf_dir, tmp_path / "placement.json")
    plain_loaded = yaml.safe_load(plain.read_text())
    assert plain_loaded["scheduler"] == {
        "max_inflight_requests": 1,
        "compiler_placement_file": str(tmp_path / "placement.json"),
    }
    assert plain_loaded["trace"] == {"num_requests": 10}


def test_relative_log_dir_smoke(monkeypatch, tmp_path) -> None:
    """相对 --log-dir 在入口归一为绝对路径，配置/trace 落点不再错拼。

    step_e_simulate 对相对配置统一拼 `genesim/conf/`，genesim 子进程又以
    仓根解析相对 trace——相对 log_dir 会让两者双双找不到。入口 resolve
    后，ceva_sim_config 产出的配置路径即为绝对路径，不再触发错拼分支。
    """
    import yaml

    monkeypatch.chdir(tmp_path)
    from scripts import run_full_pipeline as fp

    # 模拟 main 的 genesim 分支：相对入口参数 → 归一 → 配置路径绝对。
    log_dir = Path("relsweep").resolve()
    assert log_dir.is_absolute()

    genesim = tmp_path / "genesim"
    (genesim / "conf").mkdir(parents=True)
    src = genesim / "conf" / fp.CEVA_SIM_CONFIG
    src.write_text("scheduler: {}\n")
    config = fp.ceva_sim_config(
        src, log_dir / "ceva-config", tmp_path / "placement.json",
        batch_size=2, num_requests=4, trace_path=str(log_dir / "closed.json"))
    config_path = Path(config)
    assert config_path.is_absolute()
    # 相对配置才会被拼到 genesim/conf 下：断言落点在预期的日志目录里。
    assert (log_dir / "ceva-config" / fp.CEVA_SIM_CONFIG).is_file()
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["trace"]["path"] == str(log_dir / "closed.json")


def test_sidecar_tile_m_reader(tmp_path) -> None:
    """逐 B 档读 decode 相位产物的 tile_m（界 1/min(B,tile_m(B)) 逐档取值）。"""
    from scripts.run_full_pipeline import decode_tile_m_by_batch

    sidecar = tmp_path / "placement.json"
    sidecar.write_text(json.dumps({"operators": {
        "1": {"pimir_by_phase": {"prefill": {"tile_m": 16}, "decode_b1": {"tile_m": 1},
                                 "decode_b4": {"tile_m": 4}}},
        "2": {"pimir_by_phase": {"decode_b4": {"tile_m": 2},
                                 "decode_b8": {"tile_m": 8}}},
        "3": {"kernel_tile_n": 512},  # 旧格式条目：跳过
    }}))
    assert decode_tile_m_by_batch(sidecar) == {1: 1, 4: 4, 8: 8}
    assert decode_tile_m_by_batch(tmp_path / "absent.json") == {}


# —— 容量墙捕获链路：稳定标记判类，触墙记墙位继续、其余失败照常抛 ——


class _FakeTraceSource:
    """trace_api 的替身：跳过真实 trace 二进制装载。"""

    @staticmethod
    def load_binary(path):
        class _Src:
            @staticmethod
            def to_dict():
                return {"requests": [{
                    "request_id": "t0", "input_tokens": [1, 2],
                    "output_length": 2, "arrival_time_ns": 0,
                    "is_prefill": True, "seq_len": 2}]}

        return _Src()


def _fake_simulate_factory(fail_labels: dict[str, str]):
    """按档注入失败：label -> StepFailed 消息；其余档写产物并返回 summary。"""

    def simulate(genesim, config, log_dir, *, label, results_dir=None,
                 max_requests=None):
        from scripts.run_full_pipeline import StepFailed

        if label in fail_labels:
            raise StepFailed(fail_labels[label])
        assert results_dir is not None
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "batch_metrics.json").write_text(json.dumps(_metrics(
            10.0, 8.0, 1.0, 1.0, 1000.0)))
        return _summary(10.0, 10.0, 0.5, 0.05)

    return simulate


def _sweep_env(monkeypatch, tmp_path):
    """搭一个最小扫参环境：临时 genesim conf、假 trace 源、报告落 tmp。"""
    import yaml

    from scripts import run_full_pipeline as fp

    genesim = tmp_path / "genesim"
    (genesim / "conf").mkdir(parents=True)
    (genesim / "conf" / fp.CEVA_SIM_CONFIG).write_text(yaml.safe_dump(
        {"scheduler": {}, "trace": {"num_requests": 4}}))
    monkeypatch.setattr(fp, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr("tune._genesim.trace_api",
                        lambda: (_FakeTraceSource, None))
    return fp, genesim


def test_sweep_capacity_wall_continues(monkeypatch, tmp_path) -> None:
    """某档触容量墙：记墙位后继续其余档，报告含容量墙行（不再中断整轮扫参）。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory(
        {"b4g1": "命令失败（exit 1）：…\nValueError: PIM_CAPACITY_EXCEEDED: "
                 "PIM resident-memory capacity exceeded (limit=8 B)"}))

    args = Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False)
    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2, 4), (1,), args,
                               trace_path=None, num_requests=4, max_seq=None)
    report = scope.read_text()
    assert "容量墙" in report and "B=4" in report
    # 未触墙的档照常进逐档表。
    assert "| 1 | 1 |" in report and "| 2 | 1 |" in report


def test_sweep_all_batches_hit_wall(monkeypatch, tmp_path) -> None:
    """单档扫参触墙（全墙终态）：命令正常退出且报告落盘含墙位行。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory(
        {"b4g1": "命令失败（exit 1）：…\nValueError: PIM_CAPACITY_EXCEEDED: "
                 "PIM resident-memory capacity exceeded (limit=8 B)"}))

    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (4,), (1,),
                               Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                               trace_path=None, num_requests=4, max_seq=None)
    report = scope.read_text()
    assert "容量墙" in report and "B=4" in report
    assert "无成功档" in report
    assert "吞吐最优档" not in report


def test_sweep_non_capacity_failure_raises(monkeypatch, tmp_path) -> None:
    """含「容量」字样但非容量墙的失败（如产物契约不一致）不得被误判成墙。"""
    from argparse import Namespace

    import pytest

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory(
        {"b2g1": "放置 sidecar 的物理/预留/可用容量 8 与画像契约不一致"}))

    with pytest.raises(fp.StepFailed):
        fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,),
                           Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                           trace_path=None, num_requests=4, max_seq=None)


def test_sweep_wall_keeps_g_dimension(monkeypatch, tmp_path) -> None:
    """同 B 不同 G 触墙：报告列 (B,G) 组合，成功档不进命中、不升格 B 墙。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory(
        {"b4g4": "命令失败（exit 1）：…\nValueError: PIM_CAPACITY_EXCEEDED: "
                 "PIM resident-memory capacity exceeded (limit=8 B)"}))

    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (4,), (1, 4),
                               Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                               trace_path=None, num_requests=4, max_seq=None)
    report = scope.read_text()
    # 失败组合 (4,4) 直接列出；成功的 (4,1) 不被反推成命中档。
    assert "(B=4, G=4)" in report
    assert "(B=4, G=1)" not in report
    assert "命中档" not in report
    # 部分 G 触墙不升格 B 墙：并发约束提示在，DP 不把 B=4 当墙折容量。
    assert "并发组数受容量约束" in report
    assert "B≥4" not in report


def test_sweep_capacity_note_and_dp_estimate(monkeypatch, tmp_path) -> None:
    """编译期容量表接进扫参：逐档「容量占用」列出数、DP 估算拿到墙位输入。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}, "capacity": {
        "budget_bytes": 1000, "wall_batch": 5, "slots": 2, "max_seq": 795,
        "dpus": {"3": {"fixed_bytes": 100, "var_bytes_at_slots": 800,
                       "slots": 2, "max_batch": 4}}}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory({}))

    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,),
                               Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                               trace_path=None, num_requests=4, max_seq=795)
    report = scope.read_text()
    # B=1：100+400=500/1000=50%；B=2：900/1000=90%，最紧 DPU 是 dpu3。
    # 编译档 slots=2：B=2 是编译档（编译期），B=1 是线性折算档（标「折算」）。
    assert "容量占用" in report
    assert "dpu3 50%（编译期折算）" in report and "dpu3 90%（编译期）" in report
    # 编译期墙位 B=5 进入瓶颈与推荐，DP 估算不再跳过。
    assert "B=5" in report and "编译期容量表判定" in report
    assert "2 副本" in report and "缺容量墙输入" not in report


def test_sweep_max_seq_mismatch_raises(monkeypatch, tmp_path) -> None:
    """容量表 max_seq 与 trace 推导值不一致：直接抛，不用错误墙位出报告。"""
    from argparse import Namespace

    import pytest

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}, "capacity": {
        "budget_bytes": 1000, "wall_batch": 4370, "slots": 2, "max_seq": 64,
        "dpus": {"3": {"fixed_bytes": 100, "var_bytes_at_slots": 800,
                       "slots": 2, "max_batch": 4}}}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory({}))
    with pytest.raises(fp.StepFailed, match="max_seq"):
        fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,),
                           Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                           trace_path=None, num_requests=4, max_seq=795)


def test_sweep_capacity_without_max_seq_raises(monkeypatch, tmp_path) -> None:
    """capacity 载荷缺 max_seq 键（旧导出产物）：无法核对序列口径，直接抛。"""
    from argparse import Namespace

    import pytest

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}, "capacity": {
        "budget_bytes": 1000, "wall_batch": 5, "slots": 2,
        "dpus": {"3": {"fixed_bytes": 100, "var_bytes_at_slots": 800,
                       "slots": 2, "max_batch": 4}}}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory({}))
    with pytest.raises(fp.StepFailed, match="max_seq"):
        fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,),
                           Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
                           trace_path=None, num_requests=4, max_seq=795)


def test_synthesize_closed_trace_feeds_max_seq(monkeypatch, tmp_path) -> None:
    """closed trace 合成前置到导出之前：同一份 trace 喂导出与仿真，容量口径
    max_seq = seq_len+output_length 与请求数一起返回。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    trace_path, max_seq, num_requests = fp.synthesize_closed_trace(
        genesim, tmp_path / "logs", (1, 2, 4), None)
    assert trace_path is not None and max_seq == 4  # 首请求 seq 2 + output 2
    assert num_requests == 16                       # 4 × lcm(1,2,4)
    assert Path(trace_path).is_file()
    # 全部档 B=1：不合成 trace，max_seq 也不推导。
    none_path, none_seq, num = fp.synthesize_closed_trace(
        genesim, tmp_path / "logs", (1,), 8)
    assert none_path is None and none_seq is None and num == 8


def test_closed_trace_max_seq_derivation() -> None:
    """容量口径 max_seq = 首请求 seq_len + output_length；非法长度抛。"""
    from scripts.run_full_pipeline import closed_trace_max_seq

    source = {"requests": [{
        "request_id": "t0", "seq_len": 726, "output_length": 69}]}
    assert closed_trace_max_seq(source) == 795
    with pytest.raises(Exception, match="max_seq"):
        closed_trace_max_seq({"requests": [{"seq_len": 0,
                                            "output_length": 69}]})


def test_sweep_default_tiers_cover_tile_m_cap() -> None:
    """--sweep 默认档位 1..32：覆盖内核 tile_m 上限（16）与外推容量墙。"""
    from scripts.run_full_pipeline import expand_sweep_defaults

    batches, groups = expand_sweep_defaults((4,), (1,))
    assert batches == (1, 2, 4, 8, 16, 32)
    assert groups == (1, 2, 4)
    # 显式给过的档保留（并集），不去删用户输入。
    batches, _ = expand_sweep_defaults((64,), (5,))
    assert batches == (1, 2, 4, 8, 16, 32, 64)


def test_report_ttft_and_phase_time_columns() -> None:
    """逐档表给 TTFT 两列（含排队/首组）与 prefill/decode 时间拆分列。"""
    metrics = _metrics(10.0, 6.0, 2.0, 2.0, 1000.0)
    metrics["groups"][0].update({
        "ttft_s": 3.5, "prefill_time_s": 7.0, "decode_time_s": 3.0})
    text = aggregate_batch_report(
        [SweepRun(4, 1, _summary(10.0, 10.0, 0.5, 0.05), metrics)])
    header = next(line for line in text.splitlines()
                  if line.startswith("| B |"))
    assert "TTFT P95 含排队" in header and "TTFT 首组" in header
    assert "prefill 时间" in header and "decode 时间" in header
    assert "decode 占比" in header
    assert "3.500" in text          # 首组 TTFT
    assert "30.0%" in text          # decode 占比 3/10
    # 推荐段的时延最优用首组 TTFT（不含排队）口径。
    assert "首组 TTFT（不含排队）" in text


# --- R8 流水开关接入扫参入口：--pipeline-overlap 与两态成对呈现 ---------------------


def test_ceva_sim_config_writes_overlap_only_when_given(tmp_path) -> None:
    """显式给出时写 scheduler.pipeline_overlap；缺省不写（经典配置逐字节不变）。"""
    import yaml

    from scripts.run_full_pipeline import ceva_sim_config

    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump({
        "scheduler": {"compiler_placement_file": "/tmp/x.json"},
    }))

    plain = ceva_sim_config(source, tmp_path / "c1", tmp_path / "p.json")
    loaded = yaml.safe_load(plain.read_text())
    assert "pipeline_overlap" not in loaded["scheduler"]

    switched = ceva_sim_config(source, tmp_path / "c2", tmp_path / "p.json",
                               pipeline_overlap=True)
    loaded = yaml.safe_load(switched.read_text())
    assert loaded["scheduler"]["pipeline_overlap"] is True


def test_report_overlap_paired_presentation() -> None:
    """两态产物成对呈现：同一 (B,G) 的开/关两行进「流水重叠对比」段。"""
    off = SweepRun(4, 1, _summary(100.0, 10.0, 0.8, 0.08),
                   _metrics(100.0, 90.0, 5.0, 5.0, 250.0))
    on = SweepRun(4, 1, _summary(70.0, 14.3, 0.6, 0.06),
                  _metrics(70.0, 65.0, 2.0, 3.0, 250.0),
                  pipeline_overlap=True)
    text = aggregate_batch_report([off, on],
                                  tile_m_by_batch={4: 8})
    section = text.split("## 流水重叠对比")[1]
    assert "| 4 | 1 |" in section
    assert "关" in section and "开" in section
    # 吞吐与气泡占比两态并列：关 10.00 → 开 14.30（表述纪律：
    # 收益只写吞吐改善/步间重叠生效，不写气泡下降）。
    assert "10.00" in section and "14.30" in section


def test_report_overlap_single_state_notes_absence() -> None:
    """单态产物没有可对比的开/关对：如实标注，不出对比行。"""
    text = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8})
    assert "仅单态（未开启 --pipeline-overlap 成对扫参）" in text


# --- R8 DP 副本产物进报告与扫参接线（SweepRun.replicas / 两态 DP 段） ---------------


def test_ceva_sim_config_writes_dp_replicas_only_when_given(tmp_path) -> None:
    """显式给出时写 scheduler.dp_replicas；缺省不写（经典配置逐字节不变）。"""
    import yaml

    from scripts.run_full_pipeline import ceva_sim_config

    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump({
        "scheduler": {"compiler_placement_file": "/tmp/x.json"},
    }))

    plain = ceva_sim_config(source, tmp_path / "c1", tmp_path / "p.json")
    loaded = yaml.safe_load(plain.read_text())
    assert "dp_replicas" not in loaded["scheduler"]

    switched = ceva_sim_config(source, tmp_path / "c2", tmp_path / "p.json",
                               dp_replicas=2)
    loaded = yaml.safe_load(switched.read_text())
    assert loaded["scheduler"]["dp_replicas"] == 2


def test_sweep_passes_replicas_to_tier_configs(monkeypatch, tmp_path) -> None:
    """--replicas > 1 时逐档仿真配置写 scheduler.dp_replicas，档产物带副本身份。"""
    import yaml
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))

    written: list[Path] = []
    real_simulate = fp.step_e_simulate

    def capture_simulate(genesim, config, log_dir, *, label, results_dir=None,
                         max_requests=None):
        written.append(Path(config))
        return _fake_simulate_factory({})(genesim, config, log_dir,
                                          label=label,
                                          results_dir=results_dir)

    monkeypatch.setattr(fp, "step_e_simulate", capture_simulate)
    args = Namespace(max_requests=4, replicas=2, pipeline_overlap=False,
                             num_dpus=64, kv_head_pin=False)
    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,), args,
                               trace_path=None, num_requests=4, max_seq=None)
    assert len(written) == 2
    for config in written:
        loaded = yaml.safe_load(config.read_text())
        assert loaded["scheduler"]["dp_replicas"] == 2
    report = scope.read_text()
    # 档产物（fake metrics 无 dp_replicas 时回落 args.replicas）进实测 DP 段。
    assert "真实副本" in report and "R8 独立切片" not in report


def test_report_dp_measured_segment_for_replica_runs() -> None:
    """真实副本产物：DP 段按逐组 replica 归因（逐副本吞吐/权重复制），不再
    输出「R8 独立切片」过时声明。"""
    metrics = _metrics(20.0, 18.0, 1.0, 1.0, 250.0, tokens=50)
    metrics["dp_replicas"] = 2
    metrics["groups"] = [
        {**metrics["groups"][0], "replica": 0,
         "weight_bytes": 500.0, "generated_tokens": 50,
         "zone_bytes": {"weight": 400.0, "kv": 60.0, "activation": 40.0}},
        {**metrics["groups"][0], "request_index": 1, "replica": 1,
         "weight_bytes": 500.0, "generated_tokens": 50,
         "zone_bytes": {"weight": 400.0, "kv": 60.0, "activation": 40.0}},
    ]
    # R17-04：静态副本字节（sidecar 声明的每副本纯权重/tile）与动态权重
    # 搬运（group.weight_bytes 逐相位累计）分开列出，单位不得混写。
    metrics["dp_replicas_static"] = {
        "weight_bytes_per_replica": 18531880960.0,
        "kernel_tile_bytes_per_replica": 0.0,
    }
    metrics["groups"][0]["weight_bytes"] = 5.015448059904e12 / 2
    metrics["groups"][1]["weight_bytes"] = 5.015448059904e12 / 2
    run = SweepRun(4, 2, _summary(20.0, 5.0, 0.8, 0.08), metrics, replicas=2)
    text = aggregate_batch_report([run])
    assert "R8 独立切片" not in text
    section = text.split("## DP 空间并行")[1].split("\n## ")[0]
    assert "实测" in section
    # 逐副本吞吐：每副本 50 token / 20s = 2.50 tokens/s。
    assert "2.50" in section
    # 静态 18.53GB 与动态 5.015e12B 并列出现（相差约 270 倍，各标语义）。
    assert "静态每副本纯权重复制 1.853e+10B" in section
    assert "（2 份物理 DPU 各持一份）" in section
    assert "动态权重搬运（全生命周期逐相位累计，非权重复制量）合计 5.015e+12B" in section
    # 旧口径「每副本全量 5.015e+12 B」的混写不再出现。
    assert "每副本全量" not in section
    assert "KV" in section


def test_report_dp_analytical_when_no_replica_runs() -> None:
    """仅非副本产物：DP 段维持解析估算并如实标注 --replicas 未开启。"""
    text = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8},
                                  capacity_wall_at=8)
    section = text.split("## DP 空间并行")[1].split("\n## ")[0]
    assert "--replicas 未开启" in section
    assert "解析" in section
    assert "R8 独立切片" not in text


def test_bcd_export_passes_kv_head_pin_flag(monkeypatch, tmp_path) -> None:
    """--kv-head-pin 入口透传：显式给出时进导出命令，缺省不出现。"""
    from scripts import run_full_pipeline as fp

    captured: list[list[str]] = []

    def fake_run(cmd, *, cwd, log_path, env=None):
        captured.append(list(cmd))

    monkeypatch.setattr(fp, "_run", fake_run)
    monkeypatch.setattr(fp, "genesim_models_dir", lambda: tmp_path)
    sidecar = tmp_path / "llama2_7b_tp16_pp4_placement.json"
    sidecar.write_text(json.dumps({
        "operators": {"op0": {"pimir_path": "/tmp/x.pimir"}},
        "requires_pimir": True,
    }))
    monkeypatch.setattr(fp, "check_sidecar_entries", lambda ops: (1, 0))
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)

    fp.step_bcd_export(4, 64, tmp_path / "logs", kv_head_pin=True)
    assert "--kv-head-pin" in captured[0]

    fp.step_bcd_export(4, 64, tmp_path / "logs2", kv_head_pin=False)
    assert "--kv-head-pin" not in captured[1]


# ---- 评审 round09 问题 2/4：报告与产物的世代一致性 ----

def test_report_rows_carry_artifact_fingerprint() -> None:
    """逐档表带「产物指纹」列：报告行自证消费的是哪一代盘上产物。"""
    runs = _runs()
    for i, run in enumerate(runs):
        run.fingerprint = f"{i:08x}"
    text = aggregate_batch_report(runs)
    header = next(line for line in text.splitlines() if line.startswith("| B |"))
    assert "产物指纹" in header
    assert "| 00000000 |" in text and "| 00000003 |" in text
    # 缺指纹的档（桩仿真）标「—」不崩。
    text_missing = aggregate_batch_report(_runs()[:1])
    assert "产物指纹" in text_missing


def test_report_embeds_coverage_and_sweep_root() -> None:
    """报告头嵌扫参范围与产物根：索引与过期核验据此定位产物目录。"""
    text = aggregate_batch_report(
        _runs()[:1], coverage="batches=1 groups=1 overlap=off replicas=1 requests=8",
        sweep_root="test-results/full-pipeline-r09-b/sweep")
    assert "> 扫参范围：batches=1 groups=1 overlap=off replicas=1 requests=8" in text
    assert "> 产物根：test-results/full-pipeline-r09-b/sweep" in text


def test_preemption_segment_dedups_same_batch_runs() -> None:
    """抢占代价段按 (B, 副本) 去重：b4g1/b4g2 同为 B=4 只出一行。"""
    runs = [
        SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08), _metrics(11.0, 10.0, 0.5, 0.5, 250.0)),
        SweepRun(4, 2, _summary(12.0, 31.0, 0.9, 0.09), _metrics(12.0, 10.0, 0.5, 0.5, 250.0)),
    ]
    text = aggregate_batch_report(runs)
    segment = text.split("## 抢占代价")[1].split("\n## ")[0]
    assert segment.count("B=4:") == 1
    # 副本档与同 B 的非副本档是两个键：各留一行。
    dp_run = SweepRun(4, 2, _summary(13.0, 32.0, 1.0, 0.1),
                      _metrics(13.0, 10.0, 0.5, 0.5, 250.0), replicas=2)
    text_dp = aggregate_batch_report(runs + [dp_run])
    segment_dp = text_dp.split("## 抢占代价")[1].split("\n## ")[0]
    assert segment_dp.count("B=4:") == 1 and "B=4（副本 ×2）" in segment_dp


def test_preemption_segment_dedups_overlap_paired_runs() -> None:
    """成对扫参（--pipeline-overlap）的关/开两态同 B 只出一行（round11 问题 1）。

    解析估算的分子是一次换出的整组 KV，不随重叠态变化；行文案也不渲染
    重叠态，去重键必须与渲染维度一致，否则两态渲染出两行逐字相同文案。
    """
    runs = [
        SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08), _metrics(11.0, 10.0, 0.5, 0.5, 250.0),
                 pipeline_overlap=False),
        SweepRun(4, 1, _summary(11.5, 31.0, 0.9, 0.09), _metrics(11.0, 10.0, 0.5, 0.5, 250.0),
                 pipeline_overlap=True),
    ]
    text = aggregate_batch_report(runs)
    segment = text.split("## 抢占代价")[1].split("\n## ")[0]
    assert segment.count("B=4:") == 1


def test_sweep_report_writes_named_scope_file_and_index(
        monkeypatch, tmp_path) -> None:
    """专项聚合只写专项文件名；report.md 固定为索引，不再被末次专项覆盖。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory({}))

    args = Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                     num_dpus=64, kv_head_pin=False, report_name="b-curve")
    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (1, 2), (1,),
                               args, trace_path=None, num_requests=4, max_seq=None,
                               report_name=args.report_name)
    out_dir = tmp_path / "test-results" / "batch-sweep"
    assert scope == out_dir / "report-b-curve.md" and scope.is_file()
    scope_text = scope.read_text()
    assert "> 扫参范围：batches=1,2 groups=1" in scope_text
    assert "> 产物根：" in scope_text and "产物指纹" in scope_text
    # canonical 名固定为索引：指向专项报告，本身不是专项内容。
    index = (out_dir / "report.md").read_text()
    assert "report-b-curve.md" in index and "未产出" in index
    assert "## 逐档结果" not in index
    # 缺省 --report-name 时按档位自动命名。
    scope_auto = fp.run_batch_sweep(
        genesim, tmp_path / "logs", placement, (1, 2), (1,),
        Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                               num_dpus=64, kv_head_pin=False),
        trace_path=None, num_requests=4, max_seq=None)
    assert scope_auto == out_dir / "report-b1-2g1.md" and scope_auto.is_file()


def test_report_index_marks_old_caliber_rows_from_report_headers(tmp_path) -> None:
    """索引按报告头部「旧口径警示」行标注口径：重建不再丢警示（round16 问题2）。

    手工加在 report.md 头部的警示行会被下一次索引重建静默丢弃；口径标注
    改由生成器从各报告自带的警示行读取，旧口径条目行内标「待重扫」
    （round22 起登记条目为 R9/R10）。
    """
    from scripts.run_full_pipeline import _write_report_index

    (tmp_path / "report-b-curve.md").write_text(
        "# batch 扫参汇总报告\n\n"
        "> ⚠ 旧口径警示：本报告数字产生于集合通信字节携带编译档 batch 维的旧口径"
        "（Tq 型通信虚高），重扫登记见第八节 R9/R10，重建前推荐视为未定。\n\n"
        "> 扫参范围：batches=1,2,4,8 groups=1 overlap=off replicas=1 requests=8\n"
        "> 产物根：test-results/full-pipeline-r09-b/sweep\n")
    (tmp_path / "report-b1g1.md").write_text(
        "# batch 扫参汇总报告\n\n"
        "> 扫参范围：batches=1 groups=1 overlap=off replicas=1 kv_head_pin=off "
        "requests=32\n"
        "> 产物根：test-results/full-pipeline-final2/sweep\n")

    index = _write_report_index(tmp_path)
    text = index.read_text()
    # 混合口径窗口头部警示随重建再生，不靠手工补。
    assert "口径混合" in text
    old_row = next(l for l in text.splitlines() if "report-b-curve.md" in l)
    assert "旧口径/推荐未定（待重扫 R9/R10）" in old_row
    new_row = next(l for l in text.splitlines() if "report-b1g1.md" in l)
    assert "待重扫" not in new_row and "新口径" in new_row
    # 再次重建保持稳定：警示不因重生成丢失。
    _write_report_index(tmp_path)
    assert "口径混合" in index.read_text()


def test_sweep_labels_replica_tiers_separately(monkeypatch, tmp_path) -> None:
    """DP 档的产物目录带 dp 后缀：b4g2dp2 与非副本 b4g2 并存，不互相覆盖。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    labels: list[str] = []
    real_simulate = fp.step_e_simulate

    def capture(genesim, config, log_dir, *, label, results_dir=None,
                max_requests=None):
        labels.append(label)
        return _fake_simulate_factory({})(genesim, config, log_dir, label=label,
                                         results_dir=results_dir)

    monkeypatch.setattr(fp, "step_e_simulate", capture)
    fp.run_batch_sweep(genesim, tmp_path / "logs", placement, (4,), (2,),
                       Namespace(max_requests=4, replicas=2, pipeline_overlap=False,
                             num_dpus=64, kv_head_pin=False),
                       trace_path=None, num_requests=4, max_seq=None)
    assert labels == ["b4g2dp2"]
    assert (tmp_path / "logs" / "sweep" / "b4g2dp2" / "batch_metrics.json").is_file()


def test_mark_stale_report_rows_flags_overwritten_generation(tmp_path) -> None:
    """产物目录被覆盖后，旧报告对应行显式标注「本行过期」；一致行不动。"""
    from scripts import run_full_pipeline as fp

    out_dir = tmp_path / "batch-sweep"
    out_dir.mkdir()
    sweep_root = tmp_path / "sweep"
    tier = sweep_root / "b4g1"
    tier.mkdir(parents=True)
    (tier / "batch_metrics.json").write_text(json.dumps(_metrics(11.0, 10.0, 0.5, 0.5, 250.0)))
    (tier / "summary.json").write_text(json.dumps(_summary(11.0, 30.0, 0.8, 0.08)))
    fresh_fp = fp._artifact_fingerprint(tier)

    report = out_dir / "report-b-curve.md"
    stale_row = (f"| 1 | 1 | 关 | 723.14 | 0.76 | … | dpu0 6%（编译期折算）"
                 f" | 0000dead |")
    fresh_row = (f"| 4 | 1 | 关 | 649.89 | 0.85 | … | dpu0 6%（编译期折算）"
                 f" | {fresh_fp} |")
    report.write_text(
        "# batch 扫参汇总报告\n> 产物根：" + str(sweep_root) +
        "\n\n## 逐档结果\n| B | G | 重叠 | … | 容量占用 | 产物指纹 |\n|---|"
        + "---|" * 5 + "\n" + stale_row + "\n" + fresh_row + "\n")

    fp.mark_stale_report_rows(out_dir)
    text = report.read_text()
    assert "产物已被覆盖，本行过期" in text.splitlines()[6]
    assert "本行过期" not in text.splitlines()[7]
    # 幂等：再核一遍不重复标注。
    fp.mark_stale_report_rows(out_dir)
    assert text == report.read_text()


# ---- 评审 round10 问题 3：口径维度集单一真源（kv_head_pin/max_seq 覆盖进报告身份） ----

def test_sweep_scope_dimensions_single_source() -> None:
    """口径维度集是报告身份的单一真源：coverage 行与自动名都从它生成。

    放置类开关（kv_head_pin）与容量口径覆盖（max_seq 显式给出）都进身份；
    同 (B,G) 点只差这些输入时，口径行与自动名互异——同口径行必同放置语义。
    """
    from scripts import run_full_pipeline as fp

    base = dict(batches=(4,), groups_list=(1,), overlap=False, replicas=1,
                requests=8)
    off = fp.sweep_scope_dimensions(kv_head_pin=False, **base)
    on = fp.sweep_scope_dimensions(kv_head_pin=True, **base)
    assert fp.coverage_line(off) != fp.coverage_line(on)
    assert fp.sweep_scope_name(off) != fp.sweep_scope_name(on)
    assert fp.sweep_scope_name(off) == "b4g1"
    assert fp.sweep_scope_name(on) == "b4g1khp"
    assert "kv_head_pin=off" in fp.coverage_line(off)
    assert "kv_head_pin=on" in fp.coverage_line(on)
    # 既有身份维度在重构后保持：重叠与副本后缀。
    ov = fp.sweep_scope_dimensions(batches=(4,), groups_list=(1,), overlap=True,
                                   replicas=1, requests=8)
    dp = fp.sweep_scope_dimensions(batches=(4,), groups_list=(2,), overlap=False,
                                   replicas=2, requests=8)
    assert fp.sweep_scope_name(ov) == "b4g1ov"
    assert fp.sweep_scope_name(dp) == "b4g2dp2"
    # 显式 max_seq 覆盖改变容量口径：进口径行与自动名；缺省（trace 推导）不占维度。
    ms = fp.sweep_scope_dimensions(max_seq=795, **base)
    assert "max_seq=795" in fp.coverage_line(ms)
    assert "max_seq" not in fp.coverage_line(off)
    assert fp.sweep_scope_name(ms) == "b4g1ms795"


def test_sweep_report_identity_separates_kv_head_pin_states(
        monkeypatch, tmp_path) -> None:
    """钉扎两态/max_seq 覆盖的口径行与自动名互异：报告与产物目录都不互相覆盖。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    labels: list[str] = []

    def simulate(genesim, config, log_dir, *, label, results_dir=None,
                 max_requests=None):
        labels.append(label)
        assert results_dir is not None
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "batch_metrics.json").write_text(json.dumps(
            _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)))
        summary = _summary(10.0, 10.0, 0.5, 0.05)
        (results_dir / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(fp, "step_e_simulate", simulate)
    common = dict(max_requests=4, replicas=1, pipeline_overlap=False,
                  num_dpus=64)
    for logs, extra in ((tmp_path / "logs", {}),
                        (tmp_path / "logs-khp", {"kv_head_pin": True}),
                        (tmp_path / "logs-ms", {"max_seq": 795})):
        fp.run_batch_sweep(genesim, logs, placement, (4,), (1,),
                           Namespace(**common, **extra),
                           trace_path=None, num_requests=4, max_seq=None)
    out_dir = tmp_path / "test-results" / "batch-sweep"
    plain = (out_dir / "report-b4g1.md").read_text()
    pinned = (out_dir / "report-b4g1khp.md").read_text()
    ms_over = (out_dir / "report-b4g1ms795.md").read_text()
    # 三份报告并存不覆盖，口径行互异且各自声明钉扎/覆盖维度。
    assert ("> 扫参范围：batches=4 groups=1 overlap=off replicas=1 "
            "kv_head_pin=off requests=4") in plain
    assert "kv_head_pin=on" in pinned and "kv_head_pin=off" not in pinned
    assert "max_seq=795" in ms_over and "kv_head_pin=on" not in ms_over
    # 产物目录同样带口径后缀，同 (B,G) 档互不覆盖。
    assert labels == ["b4g1", "b4g1khp", "b4g1ms795"]
    # 指纹核验按带后缀目录找到产物：三份报告都不被标「本行过期/未核验」。
    for text in (plain, pinned, ms_over):
        assert "本行过期" not in text and "未核验" not in text


# ---- round10 全档收口的批拆分机制：--sim-batches 子集 ----

def test_resolve_sim_batches_subset() -> None:
    """--sim-batches 子集解析：缺省全集（经典路径不变）；非子集直接拒绝。

    B=1 档单档仿真在 128 请求下可达小时级（round10 实测 b1g1@8 请求 15 分钟），
    全档命令按 G 拆分仍超长时，用子集把命令拆短——导出仍按全 --batch 档，
    sidecar 世代不变（K6 口径），只有仿真范围收窄。
    """
    from scripts import run_full_pipeline as fp

    batches = (1, 2, 4, 8, 16, 32)
    assert fp.resolve_sim_batches(None, batches) == batches
    assert fp.resolve_sim_batches("2,4,8,16,32", batches) == (2, 4, 8, 16, 32)
    assert fp.resolve_sim_batches("1", batches) == (1,)
    with pytest.raises(ValueError):
        fp.resolve_sim_batches("1,64", batches)   # 64 不在 --batch 档内


# ---- round22：口径门禁（R21-03）与权重复用固定 B 对照（R21-04）----

def test_report_caliber_gate_degrades_old_products() -> None:
    """口径门禁：旧执行语义产物触发头部警示与推荐未定，新口径不受影响。"""
    old = SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08), _metrics(
        11.0, 10.0, 0.5, 0.5, 250.0))
    # 旧口径产物：归因字段缺链路三键 + forward_steps 逐槽旧语义。
    for key in ("link_wait_s", "intra_link_contention_s",
                "inter_link_contention_s"):
        del old.batch_metrics["groups"][0][key]
    old.forward_steps, old.output_length = 72, 69
    text = aggregate_batch_report([old], tile_m_by_batch={4: 8})
    assert "旧口径警示" in text
    assert "forward_steps=72" in text
    assert "缺链路争用列" in text
    # 推荐段降级：旧口径产物不产出吞吐/时延最优档。
    assert "推荐未定（旧口径）" in text
    assert "吞吐最优档" not in text

    fresh = SweepRun(4, 1, _summary(11.0, 30.0, 0.8, 0.08), _metrics(
        11.0, 10.0, 0.5, 0.5, 250.0),
        forward_steps=69, output_length=69)
    text = aggregate_batch_report([fresh], tile_m_by_batch={4: 8})
    assert "旧口径警示" not in text
    assert "吞吐最优档" in text


def test_report_caliber_gate_schema_mismatch() -> None:
    """result_schema 不等于当前版本即判旧口径（含无 metrics 的老档）。"""
    wrong = _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)
    wrong["result_schema"] = 1
    text = aggregate_batch_report(
        [SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), wrong)])
    assert "result_schema=1" in text and "旧口径警示" in text
    legacy = SweepRun(1, 1, _summary(10.0, 10.0, 0.5, 0.05), None)
    text = aggregate_batch_report([legacy])
    assert "无 batch_metrics（老口径产物）" in text


def test_report_weight_reuse_paired_presentation() -> None:
    """权重复用对比段：同一 (B,G) 的复用开/关两行并列 wbpt 与吞吐。"""
    reuse_on = SweepRun(4, 1, _summary(100.0, 10.0, 0.8, 0.08),
                        _metrics(100.0, 90.0, 5.0, 5.0, 250.0))
    reuse_off = SweepRun(4, 1, _summary(120.0, 8.3, 0.8, 0.08),
                         _metrics(120.0, 90.0, 5.0, 5.0, 1000.0),
                         weight_reuse=False)
    text = aggregate_batch_report([reuse_on, reuse_off],
                                  tile_m_by_batch={4: 8})
    section = text.split("## 权重复用对比")[1]
    assert "| 4 | 1 |" in section
    # wbpt（关 → 开）= 1000 → 250：固定 B 下装载份数 4 倍差值。
    assert "1000" in section and "250" in section
    # 逐档表带「复用」列且两态成行（重叠列同态「关」）。
    table = text.split("## 逐档结果")[1].split("## ")[0]
    assert "| 4 | 1 | 关 | 开 |" in table
    assert "| 4 | 1 | 关 | 关 |" in table
    # 单态时如实标注无对比。
    single = aggregate_batch_report(_runs(), tile_m_by_batch={1: 8})
    assert "仅单态（未开启 --weight-reuse 成对对照）" in single


def test_ceva_sim_config_writes_weight_reuse_only_when_off(tmp_path) -> None:
    """显式给出时写 scheduler.weight_reuse；缺省不写（经典配置逐字节不变）。"""
    import yaml

    from scripts.run_full_pipeline import ceva_sim_config

    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump({
        "scheduler": {"compiler_placement_file": "/tmp/x.json"},
    }))
    plain = ceva_sim_config(source, tmp_path / "c1", tmp_path / "p.json")
    assert "weight_reuse" not in yaml.safe_load(plain.read_text())["scheduler"]
    switched = ceva_sim_config(source, tmp_path / "c2", tmp_path / "p.json",
                               weight_reuse=False)
    assert yaml.safe_load(switched.read_text())["scheduler"][
        "weight_reuse"] is False


def test_sweep_scope_dimensions_weight_reuse_identity() -> None:
    """weight_reuse 只在关闭时进口径身份：缺省 coverage 行与现行逐字节一致。"""
    from scripts.run_full_pipeline import (
        coverage_line, sweep_scope_dimensions, sweep_scope_name)

    default = sweep_scope_dimensions((4,), (1,), overlap=False, replicas=1,
                                     requests=8)
    assert coverage_line(default) == (
        "batches=4 groups=1 overlap=off replicas=1 kv_head_pin=off requests=8")
    assert "weight_reuse" not in default
    assert "wr" not in sweep_scope_name(default)

    off = sweep_scope_dimensions((4,), (1,), overlap=False, replicas=1,
                                 requests=8, weight_reuse_off=True)
    assert coverage_line(off).endswith("weight_reuse=off")
    assert sweep_scope_name(off).endswith("wr")


def test_sweep_weight_reuse_paired_runs(monkeypatch, tmp_path) -> None:
    """--weight-reuse 成对扫参：同 (B,G) 档复用开/关各跑一次（wr 后缀目录）。"""
    from argparse import Namespace

    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    placement = tmp_path / "placement.json"
    placement.write_text(json.dumps({"operators": {}}))
    monkeypatch.setattr(fp, "step_e_simulate", _fake_simulate_factory({}))

    args = Namespace(max_requests=4, replicas=1, pipeline_overlap=False,
                     num_dpus=64, kv_head_pin=False, weight_reuse=True,
                     max_seq=None)
    scope = fp.run_batch_sweep(genesim, tmp_path / "logs", placement,
                               (2,), (1,), args, trace_path=None,
                               num_requests=4, max_seq=None)
    report = scope.read_text()
    # 两态产物目录互不覆盖（wr 后缀），报告带成对对比段。
    labels = sorted(d.name for d in
                    (tmp_path / "logs" / "sweep").iterdir())
    assert labels == ["b2g1", "b2g1wr"]
    assert "## 权重复用对比" in report


def test_synthesize_closed_trace_b1_multi_group_derives_max_seq(
        monkeypatch, tmp_path) -> None:
    """R21-05：B=1 多组扫参不合成 trace，但容量口径从原始 trace 首请求推导。"""
    fp, genesim = _sweep_env(monkeypatch, tmp_path)
    # B=1 × G 1,2,4：返回 (None, 首请求 seq+output, 请求数)。
    trace_path, derived, num = fp.synthesize_closed_trace(
        genesim, tmp_path / "logs", (1,), 8, groups_list=(1, 2, 4))
    assert trace_path is None and derived == 4 and num == 8
    # 经典 B=1、G=1：不推导（导出侧沿用缺省）。
    none_path, none_seq, _ = fp.synthesize_closed_trace(
        genesim, tmp_path / "logs", (1,), 8, groups_list=(1,))
    assert none_path is None and none_seq is None


# ---- round22：冻结落位（R21-06）与自包含重建回归 ----

def _tier_with_manifest(tmp_path: Path, name: str) -> Path:
    """构造一个带冻结清单的逐档结果目录（inputs 副本 + pim_mlir 原路径）。"""
    shared = tmp_path / "shared" / "inputs"
    shared.mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "opcompiler_cache"
    cache.mkdir(parents=True, exist_ok=True)

    tier = tmp_path / "sweep" / name
    (tier / "inputs").mkdir(parents=True)
    sidecar_text = json.dumps({"operators": {}})
    sidecar_sha = hashlib.sha256(sidecar_text.encode()).hexdigest()
    shared_copy = shared / f"placement-{sidecar_sha[:16]}.json"
    shared_copy.write_text(sidecar_text)
    (tier / "inputs" / shared_copy.name).write_text(sidecar_text)

    mlir_text = "module { tt.func @linear }"
    mlir_sha = hashlib.sha256(mlir_text.encode()).hexdigest()
    cache_mlir = cache / "abc123.pimir.mlir"
    cache_mlir.write_text(mlir_text)

    (tier / "batch_metrics.json").write_text(json.dumps(_metrics(
        10.0, 8.0, 1.0, 1.0, 1000.0)))
    (tier / "summary.json").write_text(json.dumps(_summary(
        10.0, 10.0, 0.5, 0.05)))
    (tier / "run_manifest.json").write_text(json.dumps({
        "inputs": {"placement_sidecar": {
            "path": str(shared_copy), "sha256": sidecar_sha,
            "origin_path": str(tmp_path / "models" / "placement.json")}},
        "config_file": None,
        "pim_mlir": [{"path": str(cache_mlir), "sha256": mlir_sha}],
    }))
    return tier


def test_relocate_manifest_rewrites_paths_and_freezes_mlir(tmp_path) -> None:
    """冻结落位：inputs 指本档、pim_mlir 冻成本档内容寻址副本并复验。"""
    from scripts.run_full_pipeline import relocate_manifest_to_tier

    tier = _tier_with_manifest(tmp_path, "b4g1")
    shared = tier / "inputs" / Path(
        json.loads((tier / "run_manifest.json").read_text())["inputs"][
            "placement_sidecar"]["path"]).name
    relocate_manifest_to_tier(tier)
    manifest = json.loads((tier / "run_manifest.json").read_text())
    entry = manifest["inputs"]["placement_sidecar"]
    assert entry["path"].startswith(str(tier))
    assert Path(entry["path"]) == shared
    # pim_mlir 冻结进本档 pim_mlir/ 且哈希核验通过。
    mlir_entry = manifest["pim_mlir"][0]
    assert mlir_entry["path"].startswith(str(tier / "pim_mlir"))
    frozen_mlir = Path(mlir_entry["path"])
    assert frozen_mlir.is_file() and "abc123" in frozen_mlir.name
    # 缺档与哈希失配直接失败：删本档副本 → 失败；篡改缓存 → 失败。
    broken = _tier_with_manifest(tmp_path, "b4g1bad")
    for stale in (broken / "inputs").iterdir():
        stale.unlink()
    with pytest.raises(Exception, match="冻结落位失败"):
        relocate_manifest_to_tier(broken)
    tampered = _tier_with_manifest(tmp_path, "b4g1tamper")
    cache_file = json.loads(
        (tampered / "run_manifest.json").read_text())["pim_mlir"][0]["path"]
    Path(cache_file).write_text("module { tampered }")
    with pytest.raises(Exception, match="冻结落位失败"):
        relocate_manifest_to_tier(tampered)


def test_full_grid_report_rebuild_byte_identical(tmp_path) -> None:
    """自包含重建回归：共享目录删除后重建全档报告逐字节一致（R21-06）。

    全档→单档→重建全档的静态链路：逐档目录经冻结落位自包含（manifest
    指本档、MLIR 为本档副本），共享 genesim/results 与算子缓存删除后，
    从各档产物重建的汇总报告与删除前逐字节一致。
    """
    from scripts.run_full_pipeline import (
        _artifact_fingerprint, relocate_manifest_to_tier)

    tiers = {name: _tier_with_manifest(tmp_path, name)
             for name in ("b1g1", "b2g1")}

    def rebuild() -> str:
        runs = []
        for batch, tier in sorted(tiers.items(), key=lambda kv: int(
                kv[0][1:kv[0].index("g")])):
            metrics = json.loads(
                (tier / "batch_metrics.json").read_text())
            summary = json.loads((tier / "summary.json").read_text())
            runs.append(SweepRun(
                int(batch[1:batch.index("g")]),
                int(batch[batch.index("g") + 1:]), summary, metrics,
                forward_steps=69, output_length=69,
                fingerprint=_artifact_fingerprint(tier)))
        return aggregate_batch_report(runs)

    for tier in tiers.values():
        relocate_manifest_to_tier(tier)
    fingerprints = {name: _artifact_fingerprint(tier)
                    for name, tier in tiers.items()}
    first = rebuild()
    # 清掉共享冻结目录与算子缓存：结果目录必须已自包含。
    shutil.rmtree(tmp_path / "shared")
    shutil.rmtree(tmp_path / "opcompiler_cache")
    assert {name: _artifact_fingerprint(tier)
            for name, tier in tiers.items()} == fingerprints
    second = rebuild()
    assert first == second
    assert "旧口径警示" not in first


def test_run_full_pipeline_has_no_unreachable_code() -> None:
    # round23 问题 2 回归：round22 插入 copytree/冻结调用时把 step_e_simulate
    # 尾部复制了一份未删原块——同层 return 之后的语句永不可达，AST 全文件
    # 扫描钉住同类编辑事故。
    source = Path(__file__).parent.parent / "scripts" / "run_full_pipeline.py"
    tree = ast.parse(source.read_text(), filename=str(source))

    def unreachable_after_return(body):
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Return) and body[i + 1:]:
                return (f"line {stmt.lineno} 的 return 之后有不可达语句"
                        f"（{body[i + 1].lineno}-{body[-1].end_lineno}）")
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list) and sub:
                    hit = unreachable_after_return(sub)
                    if hit:
                        return hit
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            hit = unreachable_after_return(node.body)
            assert hit is None, f"{node.name}: {hit}"


def test_bcd_export_passes_per_slot_prefill_flag(monkeypatch, tmp_path) -> None:
    """--per-slot-prefill 入口透传：显式给出时进导出命令，缺省不出现。"""
    from scripts import run_full_pipeline as fp

    captured: list[list[str]] = []

    def fake_run(cmd, *, cwd, log_path, env=None):
        captured.append(list(cmd))

    monkeypatch.setattr(fp, "_run", fake_run)
    monkeypatch.setattr(fp, "genesim_models_dir", lambda: tmp_path)
    sidecar = tmp_path / "llama2_7b_tp16_pp4_placement.json"
    sidecar.write_text(json.dumps({
        "operators": {"op0": {"pimir_path": "/tmp/x.pimir"}},
        "requires_pimir": True,
    }))
    monkeypatch.setattr(fp, "check_sidecar_entries", lambda ops: (1, 0))
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)

    fp.step_bcd_export(4, 64, tmp_path / "logs", per_slot_prefill=True)
    assert "--per-slot-prefill" in captured[0]

    fp.step_bcd_export(4, 64, tmp_path / "logs2", per_slot_prefill=False)
    assert "--per-slot-prefill" not in captured[1]


def test_main_export_includes_per_slot_for_overlap_and_reuse(
        monkeypatch, tmp_path) -> None:
    """执行策略贯穿生产链（R25-03）：--pipeline-overlap / --weight-reuse
    的导出带逐槽编译，缺省不带。"""
    from scripts import run_full_pipeline as fp

    captured: list[bool] = []

    def fake_export(num_stages, num_dpus, log_dir, **kwargs):
        captured.append(bool(kwargs.get("per_slot_prefill")))
        placement = tmp_path / "placement.json"
        placement.write_text(json.dumps({"operators": {}}))
        return placement

    monkeypatch.setattr(fp, "genesim_root", lambda: tmp_path / "genesim")
    monkeypatch.setattr(fp, "llama2_7b_model_dir", lambda: tmp_path / "model")
    monkeypatch.setattr(
        fp, "step_a_model_ir",
        lambda genesim, model_dir, log_dir: tmp_path / "m.ir")
    monkeypatch.setattr(fp, "step_a2_refine_pimir_ir", lambda *a, **k: None)
    monkeypatch.setattr(fp, "step_bcd_export", fake_export)

    def run_main(extra: list[str]) -> None:
        argv = ["run_full_pipeline.py", "--num-stages", "4",
                "--no-pu-mapping", "--skip-simulation",
                "--log-dir", str(tmp_path / "logs"), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        assert fp.main() == 0

    run_main([])
    run_main(["--pipeline-overlap"])
    run_main(["--weight-reuse"])
    # 缺省合批；两个逐槽态开关（流水开/复用关）都让导出带逐槽编译。
    assert captured == [False, True, True]


def test_caliber_gate_exempts_deliberate_per_slot_tiers() -> None:
    """口径门禁不误伤当期逐槽策略档（round25 修正）。

    wr 关档（weight_reuse=False）/流水开档（pipeline_overlap=True）的
    forward_steps 本就是 slots+output-1，是当期合法执行语义而非旧口径；
    合批档步数异常仍照常降级。
    """
    from scripts import run_full_pipeline as fp

    def _per_slot_run(**flags):
        metrics = _metrics(10.0, 8.0, 1.0, 1.0, 1000.0)
        return SweepRun(4, 1, _summary(10.0, 10.0, 0.5, 0.05), metrics,
                        forward_steps=72, output_length=69, **flags)

    # 逐槽策略档：不因步数多出 B-1 步被降级。
    for run in (_per_slot_run(weight_reuse=False),
                _per_slot_run(pipeline_overlap=True)):
        assert fp._caliber_findings([run]) == []
    # 合批档步数异常（真正的旧口径形态）仍被点名。
    merged = _per_slot_run()
    findings = fp._caliber_findings([merged])
    assert any("forward_steps=72≠output_length=69" in f for f in findings)
