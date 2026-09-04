#!/usr/bin/env python3
"""对比两次 GeneSim 仿真的关键指标，判断本地分片形状是否真的影响了代价。

用法：先跑两次仿真，各自把 results/ 复制到不同目录，再

    python scripts/ab_compare_local_shapes.py <A 目录> <B 目录>

判据（第二步）：切分生效后 total_time_s 必须变化，且 PIM 上 GEMM 的服务时间
应约按 1/tp_width 缩短。数值完全相同意味着切分没有进入代价链。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 从 summary.json 里取这些标量做对比。
_SCALARS = (
    "total_time_s",
    "average_request_time_s",
    "average_ttft_s",
    "average_mean_tpot_s",
    "throughput_tokens_per_s",
    "processed_tokens",
    "completed_requests",
)


def _load(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path, help="基线 results 目录（全局形状）")
    parser.add_argument("run_b", type=Path, help="对照 results 目录（本地分片形状）")
    args = parser.parse_args()

    a, b = _load(args.run_a), _load(args.run_b)

    print(f"{'指标':32s} {'A（全局）':>18s} {'B（本地分片）':>18s} {'B/A':>10s}")
    print("-" * 82)
    changed = []
    for key in _SCALARS:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        ratio = (vb / va) if isinstance(va, (int, float)) and va else float("nan")
        mark = "" if va == vb else "  <-- 变了"
        if va != vb:
            changed.append(key)
        print(f"{key:32s} {va:18.6f} {vb:18.6f} {ratio:10.4f}{mark}")

    print()
    if not changed:
        print("结论：两次运行逐项相同 —— 本地分片形状没有进入代价链，第二步未生效。")
    else:
        print(f"结论：{len(changed)} 项指标发生变化，本地分片形状已进入代价链。")
        print("变化的指标：" + ", ".join(changed))


if __name__ == "__main__":
    main()
