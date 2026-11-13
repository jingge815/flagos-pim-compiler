#!/usr/bin/env python3
"""代价模型校准适配器：样本统计、留出验证与结果分级。

    python scripts/calibrate_profile.py \
        --samples observations.jsonl \
        --training-dataset calib --validation-dataset heldout \
        --profile contracts/profiles/ceva_64core_4e.json \
        --out test-results/calibration/report.json

缺用户阈值、真实 pipeline 仿真摘要或独立留出观测时报告状态为
report_only，结论仍为模型预测。阈值格式为 JSON，如
{"median": 0.05, "p90": 0.1} 或 {"prefill/compute/median": 0.05}。
`--artifacts` 里每个角色必须登记 path 与 sha256，报告会按文件字节复核；
`--simulation-summary` 必须是本轮真实 pipeline 仿真的 summary.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.hardware_profile import load_profile
from tune.calibration import (
    build_report,
    load_samples,
    profile_applicability_error,
    profile_with_fit,
    result_maturity,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="观测样本 JSONL")
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--profile", default=None, help="硬件画像 JSON")
    parser.add_argument("--profile-out", default=None,
                        help="把冻结拟合系数与来源写回的新画像 JSON")
    parser.add_argument("--thresholds", default=None, help="用户确认的阈值 JSON")
    parser.add_argument("--artifacts", default=None,
                        help="产物哈希 JSON（IR/放置/trace/成本）")
    parser.add_argument("--package", default=None,
                        help="schema 11 产物绑定 JSON（IR/放置/trace/成本使用登记）")
    parser.add_argument("--scope", default=None, help="适用范围 JSON")
    parser.add_argument("--simulation-summary", default=None,
                        help="本轮真实 pipeline 仿真的 summary.json；"
                             "缺少它时报告只能保持 report_only")
    parser.add_argument("--out", required=True, help="报告输出路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = load_samples(Path(args.samples))
    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text())
    artifacts = (json.loads(Path(args.artifacts).read_text())
                 if args.artifacts else None)
    package = (json.loads(Path(args.package).read_text())
               if args.package else None)
    scope = json.loads(Path(args.scope).read_text()) if args.scope else None
    report = build_report(
        samples, training_dataset=args.training_dataset,
        validation_dataset=args.validation_dataset,
        artifacts=artifacts, thresholds=thresholds, scope=scope,
        package=package, simulation_summary=args.simulation_summary)
    path = write_report(report, Path(args.out))
    print(f"报告: {path}")
    print(f"状态: {report['status']}（{report.get('reason', '')}）")
    for group, parameters in report.get("fit", {}).items():
        if "scale" in parameters:
            print(f"  拟合 {group}: scale={parameters['scale']:.9g} "
                  f"n={parameters['count']}")
            continue
        for identity, target in (parameters.get("targets") or {}).items():
            print(f"  拟合 {group} [{identity}]: "
                  f"fields={target.get('fields')} "
                  f"n={target.get('count')}")
    for group, stats in (report.get("statistics") or {}).items():
        print(f"  {group}: n={stats['count']} median={stats['median']:.6f} "
              f"p90={stats['p90']:.6f} max={stats['max']:.6f}")
    if report["status"] != "passed":
        # 报告未通过时拟合尚未得到独立证据确认，不能把来源写成 fitted。
        if args.profile_out:
            print("报告未通过，拒绝写回画像；修正证据后重跑")
        return 1
    profile = load_profile(args.profile) if args.profile else None
    if profile is not None and getattr(profile, "ceva", None) is not None:
        # 样本与 scope 同步换目标时共享契约仍能生成 passed；CEVA 写回与
        # 升级还必须核对报告适用范围就是本次生效画像，而不是另一套硬件。
        scope_error = profile_applicability_error(report, profile)
        if scope_error is not None:
            print(f"报告适用范围与画像不一致，拒绝写回：{scope_error}")
            return 1
    if profile is not None and args.profile_out:
        written = profile_with_fit(
            profile.to_json(), report.get("fit") or {})
        out_path = Path(args.profile_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(written, indent=2, ensure_ascii=False))
        print(f"画像写回: {out_path}")
    if profile is not None:
        backend = (report.get("backends") or {}).get("validation", "unknown")
        maturity = result_maturity(
            report, profile=profile,
            profile_sources=profile.parameter_sources,
            backend=backend)
        print(f"结果级别: {maturity}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
