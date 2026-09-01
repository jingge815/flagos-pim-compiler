from pathlib import Path
import re

pytest_plugins = ("pytester",)


def test_report_replaces_old_log_and_marks_xfail_tail(pytester) -> None:
    repo_conftest = Path(__file__).with_name("conftest.py")
    pytester.makeconftest(repo_conftest.read_text())
    pytester.makepyfile(
        """
        def test_pass():
            assert True

        def test_fail():
            assert False

        def test_not_run():
            assert True
        """
    )
    report_path = pytester.path / "test-results" / "pytest-latest.md"
    report_path.parent.mkdir()
    report_path.write_text("obsolete report")

    result = pytester.runpytest_subprocess("-x", "-q")

    assert result.ret != 0
    report = report_path.read_text()
    assert "obsolete report" not in report
    assert "test_pass" in report and "PASS" in report
    assert "test_fail" in report and "FAIL" in report
    assert "test_not_run" in report and "NOT RUN" in report


def test_report_records_call_phase_duration(pytester) -> None:
    repo_conftest = Path(__file__).with_name("conftest.py")
    pytester.makeconftest(repo_conftest.read_text())
    pytester.makepyfile(
        """
        import time

        def test_slow_call():
            time.sleep(0.02)
        """
    )

    result = pytester.runpytest_subprocess("-q")

    assert result.ret == 0
    report = (pytester.path / "test-results" / "pytest-latest.md").read_text()
    row = next(line for line in report.splitlines() if "test_slow_call" in line)
    duration = float(re.search(r"\| ([0-9]+(?:\.[0-9]+)?) \|$", row).group(1))
    assert duration >= 0.01
