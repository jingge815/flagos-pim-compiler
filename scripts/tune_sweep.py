#!/usr/bin/env python3
"""硬件敏感性扫描入口。

    python scripts/tune_sweep.py --config examples/tune/hardware_sweep.yaml \
        --best test-results/tune-quick/best.json

扫描沿用调优的候选执行入口；非法点保留原因，改变核数/容量/编译预算的点
必须按新画像重编。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tune.config import load_config, write_effective_config
from tune.inputs import prepare_inputs, publish_inputs, verify_input_identity
from tune.report import state_from_json
from tune.sweep import run_sweep, verify_best_reference, write_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="扫描配置 YAML")
    parser.add_argument("--best", required=True, help="上一轮 best.json")
    parser.add_argument("--output-dir", default=None, help="覆盖扫描输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides = None
    if args.output_dir:
        # 在输入快照、编译与仿真之前合并输出目录，三者才落在同一目录。
        overrides = {"output": {"dir": str(Path(args.output_dir).resolve())}}
    cfg = load_config(args.config, overrides)
    best = json.loads(Path(args.best).read_text())
    try:
        best_state = state_from_json(best["state"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"硬件扫描被拒绝：best.json 缺少合法候选状态：{exc}",
              file=sys.stderr)
        return 3
    # 输入身份先只读核对；异输入或异 best 不能先改写已有快照与生效配置。
    inputs = prepare_inputs(cfg)
    try:
        verify_best_reference(cfg, inputs, best, best_state)
    except ValueError as exc:
        print(f"硬件扫描被拒绝：{exc}", file=sys.stderr)
        return 3
    recorded = cfg.output.dir / "inputs" / "inputs.json"
    if recorded.is_file():
        try:
            verify_input_identity(inputs, json.loads(recorded.read_text()))
        except (ValueError, OSError) as exc:
            print(f"硬件扫描被拒绝：{exc}", file=sys.stderr)
            return 3
    inputs = publish_inputs(cfg, inputs)
    # IR 在发布时生成；写生效配置与求值前再核对一次生成内容。
    try:
        verify_best_reference(cfg, inputs, best, best_state)
    except ValueError as exc:
        print(f"硬件扫描被拒绝：{exc}", file=sys.stderr)
        return 3
    write_effective_config(cfg)
    result = run_sweep(cfg, inputs, best_state, best=best)
    paths = write_sweep(result, cfg, cfg.output.dir)
    print(f"扫描点 {len(result.points)} 个：")
    for point in result.points:
        score = "" if point.score is None else f"{point.score:.6f}"
        print(f"  {dict(point.values)} -> {point.status} {score} {point.reason or ''}")
    print(f"Pareto 前沿 {len(result.pareto)} 个；拐点 {dict(result.knees)}")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0 if any(point.score is not None for point in result.points) else 1


if __name__ == "__main__":
    sys.exit(main())
