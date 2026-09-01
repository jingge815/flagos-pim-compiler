# 测试集精简与结果报告实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 缩减默认测试中重复的真实 LLaMA-2-7B 策略推理，并在每次 pytest 会话后生成可读的测试结果表格。

**Architecture:** `tests/conftest.py` 保存 pytest 收集项和各阶段报告，在 session 结束时生成唯一的 `test-results/pytest-latest.md`。真实 7B 策略文件仍保留原有三类断言，但三者仅参数化为同时包含 TP/PP 的 `tp4_pp2`；小模型策略遍历继续覆盖全部合法 TP/PP 组合。

**Tech Stack:** Python 3.10、pytest 8.3、pytester、Markdown。

## Global Constraints

- 默认执行命令保持 `source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh && python -m pytest tests/ -x -q`。
- 不修改用户已有的未提交业务文件；仅触及本计划列出的测试、配置和文档文件。
- 默认集只保留真实 7B 的 `tp4_pp2` 完整端到端策略；小模型 `test_strategy_sweep.py` 继续保留 `tp4_pp1`、`tp2_pp2`、`tp1_pp4` 的全策略检查。
- 每次会话先移除旧日志，结束后无论 PASS、FAIL、ERROR、SKIPPED 或 `-x` 早停都生成 `test-results/pytest-latest.md`。
- 日志表格必须包含测试文件、测试项、状态和耗时；未运行的已收集项目为 `NOT RUN`。
- `test-results/` 是运行产物，必须被 git 忽略。
- 未获用户明确要求，不创建 git commit。

---

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `tests/conftest.py` | pytest 会话钩子：收集、状态归类、日志替换和 Markdown 写入 |
| `tests/test_pytest_report.py` | 用真实子 pytest 进程验证日志替换和 `-x` 早停状态 |
| `.gitignore` | 忽略运行产物目录 `test-results/` |
| `tests/test_strategy_llama2_7b.py` | 将真实 7B 策略参数化收敛为 `tp4_pp2`，保留原始断言 |
| `docs/测试集-20260901.md` | 用中文记录本次精简内容、覆盖保留、测试命令及已知缺口 |

### Task 1: pytest 结果日志

**Files:**

- Create: `tests/conftest.py`
- Create: `tests/test_pytest_report.py`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: pytest hooks `pytest_sessionstart(session)`, `pytest_collection_modifyitems(session, config, items)`, `pytest_runtest_logreport(report)`, `pytest_sessionfinish(session, exitstatus)`。
- Produces: 每次 pytest 会话的 `<rootpath>/test-results/pytest-latest.md`，包含汇总和项目表格。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_pytest_report.py`，启用 `pytester`，把仓库的 `tests/conftest.py` 文本复制到临时 pytest 根目录。子测试文件的三个项目按源码顺序为 `test_pass`、`test_fail`、`test_not_run`。在子测试根目录预写入旧日志文本 `obsolete report`，以 `-x -q` 运行子 pytest，并断言运行失败。

```python
from pathlib import Path

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
```

- [ ] **Step 2: 确认测试因功能缺失而失败**

运行：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/test_pytest_report.py -q
```

预期：失败，原因是临时目录中尚不存在 `test-results/pytest-latest.md`，而非导入或测试拼写错误。

- [ ] **Step 3: 实现最小会话钩子**

在 `tests/conftest.py` 定义 `TestResult` 数据类和会话级状态。`pytest_sessionstart` 记录 `datetime.now()` 和 `time.perf_counter()`，创建 `<rootpath>/test-results` 并用 `unlink(missing_ok=True)` 删除旧日志。`pytest_collection_modifyitems` 以 nodeid 顺序保存项目。`pytest_runtest_logreport` 将 call 阶段的 passed/failed/skipped 归类为 `PASS`/`FAIL`/`SKIPPED`；setup/teardown 阶段失败归类为 `ERROR`，setup skipped 归类为 `SKIPPED`。`pytest_sessionfinish` 让所有未登记项目成为 `NOT RUN`，使用 `Path.write_text(..., encoding="utf-8")` 原子写入 Markdown，并打印 `测试结果日志: <absolute path>`。

日志正文格式：

```markdown
# Pytest 测试结果

- 开始时间：2026-09-01 12:00:00
- 结束时间：2026-09-01 12:00:05
- pytest exit code：1
- 总耗时：5.00 秒
- 汇总：PASS 1，FAIL 1，ERROR 0，SKIPPED 0，NOT RUN 1

| 测试文件 | 测试项 | 状态 | 耗时（秒） |
| --- | --- | --- | ---: |
| `test_sample.py` | `test_pass` | PASS | 0.01 |
| `test_sample.py` | `test_fail` | FAIL | 0.01 |
| `test_sample.py` | `test_not_run` | NOT RUN | — |
```

解析 nodeid 时用 `nodeid.partition("::")`，文件列保留测试根目录相对路径，测试项列保留 `::` 之后完整参数化名称。对失败/错误 `report.longreprtext` 只保存首个非空行作为 `TestResult.detail`，在日志末尾的“失败详情”中使用代码块输出。

在 `.gitignore` 追加：

```gitignore

# pytest 每次执行生成的最新结果表格
test-results/
```

- [ ] **Step 4: 运行报告测试并检查产物**

运行：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/test_pytest_report.py -q
```

预期：1 passed；测试所启动的子 pytest 返回失败但其日志中包含 `PASS`、`FAIL`、`NOT RUN`，主测试通过。再运行：

```bash
python -m pytest tests/test_op_contract.py -q
sed -n '1,80p' test-results/pytest-latest.md
```

预期：8 passed，并且日志仅包含本次 8 项 `test_op_contract.py` 测试及 PASS 汇总。

### Task 2: 真实 7B 代表策略和研发记录

**Files:**

- Modify: `tests/test_strategy_llama2_7b.py:50-62,135-194`
- Create: `docs/测试集-20260901.md`

**Interfaces:**

- Consumes: `llama_strategies(NUM_DPUS, ...)` 返回的 `Strategy` 列表及其稳定 `name`。
- Produces: `_representative_strategies() -> list[Strategy]`，唯一返回 `tp4_pp2`；三个真实 7B 参数化断言使用该列表。

- [ ] **Step 1: 写失败策略选择测试**

在 `tests/test_strategy_llama2_7b.py` 的策略辅助函数之后添加：

```python
def test_real_7b_suite_keeps_tp4_pp2_as_representative_strategy() -> None:
    assert [strategy.name for strategy in _representative_strategies()] == ["tp4_pp2"]
```

不要先定义 `_representative_strategies`。

- [ ] **Step 2: 确认测试因选择器缺失而失败**

运行：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/test_strategy_llama2_7b.py::test_real_7b_suite_keeps_tp4_pp2_as_representative_strategy -q
```

预期：失败，报 `NameError: name '_representative_strategies' is not defined`；若本地模型目录不存在则记录环境 skip，不实施替代断言。

- [ ] **Step 3: 实现选择器并收敛三个参数化装饰器**

在 `_strategies()` 之后实现：

```python
def _representative_strategies():
    return [strategy for strategy in _strategies() if strategy.name == "tp4_pp2"]
```

将三个 `@pytest.mark.parametrize("strategy", _strategies(), ids=lambda s: s.name)` 都替换为：

```python
@pytest.mark.parametrize("strategy", _representative_strategies(), ids=lambda s: s.name)
```

同步简化模块 docstring 和 `run_cache` 文档，使其说明默认集只跑一条 `tp4_pp2` 真实 7B 代表链路，而所有 TP/PP 组合仍在 `tests/test_strategy_sweep.py` 的小模型测试中覆盖。不要改动三条断言、7B 硬件配置、模型路径或轻量策略测试。

- [ ] **Step 4: 运行选择器和收集验证**

运行：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/test_strategy_llama2_7b.py::test_real_7b_suite_keeps_tp4_pp2_as_representative_strategy -q
python -m pytest tests/test_strategy_llama2_7b.py --collect-only -q
python -m pytest tests/test_strategy_sweep.py -q
```

预期：选择器测试通过；真实 7B 文件收集 4 项（1 项选择器 + 三项 `[tp4_pp2]`）；小模型策略测试全部通过，且仍出现 `tp4_pp1`、`tp2_pp2`、`tp1_pp4`。

- [ ] **Step 5: 写研发文档**

创建 `docs/测试集-20260901.md`，用中文说明：删除的真实 7B 参数组合、保留的代表策略和理由、结果日志路径与列、默认命令、轻量 TP/PP 覆盖文件、未覆盖项目（IR dump/serialize/replay、实际 GeneSim 二进制端到端、SDK 语义/PU 映射专门端到端测试）。文档中不声称未运行的真实 7B 推理已经通过。

### Task 3: 最终验证与变更审计

**Files:**

- Verify only: `tests/conftest.py`, `tests/test_pytest_report.py`, `tests/test_strategy_llama2_7b.py`, `.gitignore`, `docs/测试集-20260901.md`

- [ ] **Step 1: 运行快速相关测试**

运行：

```bash
source /media/disk/fengjingge/src/flagOS/flagOS-installed/pytorch/env-pytorch.sh
python -m pytest tests/test_pytest_report.py tests/test_strategy.py tests/test_strategy_sweep.py tests/test_op_contract.py -x -q
```

预期：通过；`test-results/pytest-latest.md` 的汇总全部为 PASS。

- [ ] **Step 2: 收集完整默认集并检查真实 7B 条目数**

运行：

```bash
python -m pytest tests/ --collect-only -q > /tmp/flagos-pim-compiler-tests.txt
rg -c 'tests/test_strategy_llama2_7b.py::' /tmp/flagos-pim-compiler-tests.txt
rg 'tests/test_strategy_llama2_7b.py::' /tmp/flagos-pim-compiler-tests.txt
tail -n 1 /tmp/flagos-pim-compiler-tests.txt
```

预期：真实 7B 策略文件为 4 项，三条慢用例均只带 `[tp4_pp2]`；总数从旧的 268 减少 8 项至 260（新增报告测试与选择器测试会抵消其中 2 项，最终以收集输出为准）。

- [ ] **Step 3: 运行默认测试集一次**

运行：

```bash
python -m pytest tests/ -x -q
```

预期：结束后一定存在 `test-results/pytest-latest.md`；无论 exit code 是否为零，都读取日志并如实记录失败、跳过或未运行项目。若通过，记录总耗时以与约 40 分钟旧基线对比。

- [ ] **Step 4: 审计差异与产物**

运行：

```bash
git diff --check
git status --short
git diff --stat -- .gitignore tests/conftest.py tests/test_pytest_report.py tests/test_strategy_llama2_7b.py docs/测试集-20260901.md
```

预期：无空白错误；`test-results/pytest-latest.md` 不出现在 git status；不修改用户已有业务文件。报告最终净增删行数。
