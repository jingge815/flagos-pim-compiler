from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import sys
import time

import pytest


_REPORT_STATE = None


def _is_worker(config) -> bool:
    """是否运行在 xdist worker 进程里（主进程汇总报告，worker 不写）。"""
    return hasattr(config, "workerinput")


def pytest_configure(config):
    # 32 个 worker 各自默认开满核 BLAS/torch 线程会互相争抢，限到 4线程/worker。
    # 须在 worker 导入 numpy/torch 之前设置（configure 先于收集导入测试模块）。
    if _is_worker(config):
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "4")


@pytest.fixture(scope="session", autouse=True)
def _forbid_test_writes_to_real_ceva_workspace():
    """测试不得写真实归档树：CEVA 工作区与本仓 runs/（round02 问题5、round03 问题1）。

    runs/ 归档树是验收证据，单测往里落文件会制造不可溯源的伪证据
    （如伪造的 PASS verdict）。要写工作区的测试必须 monkeypatch
    ceva_workspace 到 tmp_path；本守卫经系统审计钩子拦截一切对两处
    归档树的写/建/改名/删除/替换并让该测试立刻失败。
    工作区取本仓库 paths.json 的 ceva_workspace；hermetic 子环境
    （test_pytest_report 把 conftest 拷进临时目录）没有该配置，
    无真实工作区可保护，不装钩子。"""
    config = Path(__file__).resolve().parent.parent / "paths.json"
    if not config.is_file():
        yield
        return
    # (前缀, 报错文案)：工作区 + 本仓 runs/，两处都是真实归档树
    prefixes = [(json.loads(config.read_text())["ceva_workspace"], "测试不得写真实工作区"),
                (str(config.parent / "runs"), "测试不得写仓内真实归档树")]
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC

    def _hook(event, args):
        candidates = [str(a) for a in args[:2]
                      if isinstance(a, (str, bytes, os.PathLike))]
        hits = [(p, msg) for p in candidates for prefix, msg in prefixes
                if p == prefix or p.startswith(prefix + os.sep)]
        if not hits:
            return
        if event == "open":
            if args[2] & write_flags:
                raise AssertionError(f"{hits[0][1]}: {hits[0][0]}")
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename",
                       "os.replace", "os.symlink", "os.link", "os.truncate"):
            raise AssertionError(f"{hits[0][1]}: {hits[0][0]}")

    sys.addaudithook(_hook)
    yield


@pytest.fixture(scope="module", autouse=True)
def _restore_grad_mode():
    """每个测试文件结束后恢复梯度开关。7B 夹具会全局关闭梯度，漏到后续文件会改变 torch.export 的图。"""
    import torch

    enabled = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(enabled)


@dataclass
class TestResult:
    nodeid: str
    status: str = "NOT RUN"
    duration: float | None = None
    detail: str = ""


def pytest_sessionstart(session):
    global _REPORT_STATE
    # xdist worker 进程不写报告，避免多进程踩踏同一文件，由主进程汇总
    if _is_worker(session.config):
        return
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
    if state is None:
        return
    for item in sorted(items, key=lambda value: value.nodeid):
        state["ordered"].append(item.nodeid)
        state["results"][item.nodeid] = TestResult(item.nodeid)


def pytest_runtest_logreport(report):
    if _REPORT_STATE is None:
        return
    state = _REPORT_STATE
    # xdist 的主进程不做收集（modifyitems 不触发），报告到达时懒注册条目
    result = state["results"].setdefault(
        report.nodeid, TestResult(report.nodeid))
    if report.nodeid not in state["ordered"]:
        state["ordered"].append(report.nodeid)
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
    if _REPORT_STATE is None:
        return
    state = _REPORT_STATE
    ended = datetime.now()
    elapsed = time.perf_counter() - state["perf"]
    results = [state["results"][nodeid] for nodeid in sorted(state["ordered"])]
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
