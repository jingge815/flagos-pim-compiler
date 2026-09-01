from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time


_REPORT_STATE = None


@dataclass
class TestResult:
    nodeid: str
    status: str = "NOT RUN"
    duration: float | None = None
    detail: str = ""


def pytest_sessionstart(session):
    global _REPORT_STATE
    root = Path(session.config.rootpath)
    out_dir = root / "test-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pytest-latest.md").unlink(missing_ok=True)
    _REPORT_STATE = {
        "root": root,
        "started": datetime.now(),
        "perf": time.perf_counter(),
        "results": {},
        "ordered": [],
    }


def pytest_collection_modifyitems(session, config, items):
    state = _REPORT_STATE
    for item in sorted(items, key=lambda value: value.nodeid):
        state["ordered"].append(item.nodeid)
        state["results"][item.nodeid] = TestResult(item.nodeid)


def pytest_runtest_logreport(report):
    if _REPORT_STATE is None or report.nodeid not in _REPORT_STATE["results"]:
        return
    state = _REPORT_STATE
    result = state["results"][report.nodeid]
    if report.when == "call":
        result.duration = report.duration
        result.status = (
            "PASS" if report.passed else "FAIL" if report.failed else "SKIPPED"
        )
    elif report.failed:
        result.status = "ERROR"
    elif report.skipped and result.status == "NOT RUN":
        result.status = "SKIPPED"
    if result.status in {"FAIL", "ERROR"}:
        detail = getattr(report, "longreprtext", "")
        result.detail = next((line.strip() for line in detail.splitlines() if line.strip()), "")


def pytest_sessionfinish(session, exitstatus):
    state = _REPORT_STATE
    ended = datetime.now()
    elapsed = time.perf_counter() - state["perf"]
    results = [state["results"][nodeid] for nodeid in state["ordered"]]
    counts = {status: sum(item.status == status for item in results) for status in
              ("PASS", "FAIL", "ERROR", "SKIPPED", "NOT RUN")}
    lines = [
        "# Pytest 测试结果",
        "",
        f"- 开始时间：{state['started']:%Y-%m-%d %H:%M:%S}",
        f"- 结束时间：{ended:%Y-%m-%d %H:%M:%S}",
        f"- pytest exit code：{int(exitstatus)}",
        f"- 总耗时：{elapsed:.2f} 秒",
        "- 汇总：" + "，".join(f"{status} {counts[status]}" for status in counts),
        "",
        "| 测试文件 | 测试项 | 状态 | 耗时（秒） |",
        "| --- | --- | --- | ---: |",
    ]
    for result in results:
        filename, sep, testname = result.nodeid.partition("::")
        rel = Path(filename)
        try:
            rel = rel.relative_to(state["root"])
        except ValueError:
            pass
        duration = "—" if result.duration is None else f"{result.duration:.2f}"
        lines.append(f"| `{rel}` | `{testname if sep else ''}` | {result.status} | {duration} |")
    details = [item for item in results if item.detail]
    if details:
        lines.extend(["", "## 失败详情", ""])
        for item in details:
            lines.extend([f"### `{item.nodeid}`", "", "```", item.detail, "```", ""])
    path = state["root"] / "test-results" / "pytest-latest.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"测试结果日志: {path.resolve()}")
