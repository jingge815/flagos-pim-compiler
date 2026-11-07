"""跑固定子集基准并与 `tests/benchmarks.json` 的基线比对，越界即非零退出。

用法（先 source 环境）：
    python scripts/check_benchmarks.py            # 比对，越界报错
    python scripts/check_benchmarks.py --update   # 用本次结果刷新基线

基线文件里每条记录带容差（相对偏差），墙钟这类受机器影响的指标容差宽，
计数类指标容差为 0。改动基线必须说明「为什么预期变化」，写在提交信息里。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "benchmarks.json"


def _run_benchmark(script: str, command: list[str], out: Path) -> list[dict]:
    """跑一个基准脚本，返回它写出的 JSON 结果。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *command, "--json", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"{script} 失败，退出码 {proc.returncode}")
    return json.loads(out.read_text())


def _group_key(suite: str, item: dict) -> str:
    """两个基准的「同一条结果」标识字段不同：窗口数 vs 配置名。"""
    if suite == "bench_parallel":
        return str(item["parallel_slots"])
    return str(item["config"]["name"])


def _compare(name: str, expected: float, actual: float, tolerance: float) -> bool:
    """相对偏差在容差内算通过；容差为 0 时要求完全相等。"""
    if tolerance == 0:
        ok = actual == expected
        shown = f"{actual - expected:+g}"
    else:
        deviation = (actual - expected) / expected if expected else actual
        ok = abs(deviation) <= tolerance
        shown = f"{deviation:+.3%}"
    print(f"| {name} | {expected:g} | {actual:g} | {shown}"
          f" | {tolerance:.0%} | {'通过' if ok else '**超出**'} |")
    return ok


def _check(spec: dict) -> tuple[bool, list[dict]]:
    """跑一个基准套件并逐指标比对；返回（是否全过，本次结果）。"""
    suite = spec["suite"]
    with tempfile.TemporaryDirectory() as tmp:
        results = _run_benchmark(spec["script"], spec.get("command", []),
                                 Path(tmp) / "result.json")
    print(f"\n==> {suite}")
    print("| 分组.指标 | 基线 | 本次 | 偏差 | 容差 | 结论 |")
    print("|---|---|---|---|---|---|")
    by_key = {_group_key(suite, item): item for item in results}
    passed = True
    for group, metrics in spec["metrics"].items():
        item = by_key.get(group)
        if item is None:
            print(f"| {group} | — | 缺失 | — | — | **超出** |")
            passed = False
            continue
        for metric, expected in metrics.items():
            ok = _compare(f"{group}.{metric}", expected, item[metric],
                          spec["tolerance"][metric])
            passed = passed and ok
    return passed, results


def _update(spec: dict, results: list[dict]) -> dict:
    """用本次结果刷新一个套件的指标值，容差与说明保持不变。"""
    by_key = {_group_key(spec["suite"], item): item for item in results}
    spec["metrics"] = {
        group: {metric: by_key[group][metric] for metric in metrics}
        for group, metrics in spec["metrics"].items()
    }
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description="基准回归门限")
    parser.add_argument("--update", action="store_true", help="用本次结果刷新基线")
    args = parser.parse_args()

    baseline = json.loads(BASELINE.read_text())
    passed = True
    for spec in baseline["suites"]:
        ok, results = _check(spec)
        passed = passed and ok
        if args.update:
            _update(spec, results)

    if args.update:
        BASELINE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
        print(f"\n基线已写入 {BASELINE.relative_to(ROOT)}")
        return 0

    print(f"\n基准回归：{'通过' if passed else '失败'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
