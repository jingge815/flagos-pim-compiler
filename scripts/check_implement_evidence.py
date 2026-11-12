"""核对实施文档记录的验证数字与 summary 产物一致。

用法（先 source 环境）：
    python scripts/check_implement_evidence.py --doc docs/implement-xxx.md \
        --summary dynamic=/path/summary.json \
        --summary dense=/path/summary.json \
        --time-only alt=/path/summary.json

实施文档必须嵌入机器可读证据块：

    <!-- evidence-block:start -->
    ```evidence
    runs:
      - label: dynamic
        stratum: current
        artifact: qwen3_moe
        layers: 4
        semantics: dynamic_dispatch
        prose_aliases: [dynamic_dispatch, dynamic]
        command: "..."
        summary: /path/summary.json
        metrics: {total_time_s: ..., throughput_tokens_per_s: ...,
                  total_useful_flops: ...}
      - label: round03
        stratum: history
        prose_aliases: [round03]
        metrics: {total_useful_flops: ...}
    derived:
      - id: flops_delta
        op: subtract
        left: dynamic.total_useful_flops
        right: round03.total_useful_flops
        value: 1.55e10
    measurements:
      - id: bench_prefill_speedup
        source: "scripts/bench_moe_dispatch.py"
        value: 1.148
    facts:
      - id: dynamic_comm_bytes
        run: dynamic          # 省略 run 表示独立命名空间
        source: "E 阶段依赖边累计"
        value: 14692832968
    ```
    <!-- evidence-block:end -->

`current` 条目必须由命令行 `--summary`/`--time-only` 的同名 summary 逐值
核验，并交叉核对 summary 的模型层数/类型/执行语义；`history` 与
`measurements` 是独立命名空间，只允许正文引用，不冒充本轮产物。派生差值、
比值与结论由同一组指标重算，正文里的高精度数字必须能溯源到上述三类值。

正文里的数值还必须与所在语境的运行标签一致：标签或 `prose_aliases` 只在
数值前面找，表格按行、正文按段落/列表项取所在块；数值只归属**最近**的
前置运行标签，防止同一块里把 dynamic/dense 的值对调。必需指标必须在它
自己的运行语境里出现，写在无标签前言里不算引用。带数值的大小结论（高于/
低于等）由两侧数值与运行身份直接判定；「前者/后者」按块内最近的两个不同
标签解析。

证据块还可以用 `facts` 声明不属于 summary 三项指标的运行证据值（容量字节、
通信字节、trace 计数等）：`run` 指向运行标签时，该值只允许出现在对应运行
的语境里；省略 `run` 表示独立命名空间。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

BLOCK_START = "<!-- evidence-block:start -->"
BLOCK_END = "<!-- evidence-block:end -->"
METRIC_KEYS = (
    "total_time_s", "throughput_tokens_per_s", "total_useful_flops",
)
TIME_ONLY_KEYS = ("total_time_s", "throughput_tokens_per_s")

# 正文里需要溯源的数值格式：千分位高精度、科学计数法、长小数与比值。
NUMBER_PATTERNS = (
    re.compile(r"\d{1,3}(?:,\d{3}){3,}\.\d{3}"),
    re.compile(r"\d+\.\d+[eE][+-]?\d+"),
    re.compile(r"\d+\.\d{6,}"),
    re.compile(r"(\d+\.\d{3})\s*(?:t/s|s)\b"),
    re.compile(r"=\s*(\d+\.\d{3})\b"),
    re.compile(r"(\d+\.\d{3})×"),
)

# 正文里带数值的大小结论；用于核对「谁高于谁」与证据值一致。
_NUMBER = r"[\d][\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?"
COMPARISON_PATTERN = re.compile(
    rf"({_NUMBER})\s*[`*_]*\s*(?:[a-zA-Z/]+)?\s*[`*_]*\s*"
    rf"(?<!不)(高于|低于|大于|小于|超过)\s*[^0-9\n]{{0,40}}?"
    rf"({_NUMBER})")
# 列表项起始标记：每个条目是独立的语义块。
_LIST_MARKER = re.compile(r"(?:[-*+]|\d+\.)\s")


class EvidenceError(ValueError):
    """证据块缺失或结构非法。"""


def summary_metrics(path: Path) -> dict:
    """取出实施文档会引用的三项指标；缺少的指标不补默认值。"""
    summary = json.loads(Path(path).read_text())
    compute = (summary.get("efficiency_metrics") or {}).get("compute") or {}
    metrics = {
        "total_time_s": float(summary["total_time_s"]),
        "throughput_tokens_per_s": float(
            summary["throughput_tokens_per_s"]),
    }
    if "total_useful_flops" in compute:
        metrics["total_useful_flops"] = float(
            compute["total_useful_flops"])
    return metrics


def summary_identity(path: Path) -> dict:
    """取出 summary 自带的模型身份，用于核对层数/类型/执行语义。"""
    summary = json.loads(Path(path).read_text())
    model = summary.get("model") or {}
    moe = model.get("moe") or {}
    return {
        "artifact": str(model.get("model_type", "")),
        "layers": int(model.get("num_layers", 0)),
        "semantics": str(moe.get("execution", "")),
    }


def format_number(value: float) -> str:
    """按实施文档的千分位与 3 位小数格式渲染。"""
    return f"{value:,.3f}"


def parse_evidence_block(doc_text: str) -> dict:
    """解析实施文档里的机器可读证据块。"""
    start = doc_text.find(BLOCK_START)
    end = doc_text.find(BLOCK_END)
    if start < 0 or end <= start:
        raise EvidenceError("实施文档缺少机器可读证据块")
    body = doc_text[start + len(BLOCK_START):end]
    fenced = re.search(r"```(?:evidence|yaml)?\s*\n(.*?)```", body, re.S)
    if fenced is None:
        raise EvidenceError("证据块里缺少 evidence 代码围栏")
    loaded = yaml.safe_load(fenced.group(1)) or {}
    if not isinstance(loaded, dict) or not isinstance(
            loaded.get("runs"), list):
        raise EvidenceError("证据块必须包含 runs 列表")
    return loaded


def _prose_without_block(doc_text: str) -> str:
    """去掉证据块，只检查正文里的引用值。"""
    start = doc_text.find(BLOCK_START)
    end = doc_text.find(BLOCK_END)
    if start < 0 or end <= start:
        return doc_text
    return doc_text[:start] + doc_text[end + len(BLOCK_END):]


def _normalise_number(token: str) -> str:
    """去掉千分位与科学计数法里的冗余写法，便于逐值比较。"""
    token = token.replace(",", "").replace(" ", "")
    token = re.sub(r"[eE]\+?", "e", token)
    return re.sub(r"e(-?)0*(\d+)", r"e\1\2", token)


def _scientific_tokens(value: float) -> Tuple[str, str]:
    rendered = f"{value:.2e}"
    return rendered, rendered.replace("e+", "e")


def _runs_by_label(block: dict, problems: List[str]) -> Dict[str, dict]:
    runs: Dict[str, dict] = {}
    for entry in block.get("runs") or []:
        label = str(entry.get("label", ""))
        if not label:
            problems.append("证据块里有运行条目缺少 label")
            continue
        if label in runs:
            problems.append(f"证据块里运行标签 {label} 重复")
            continue
        runs[label] = entry
    return runs


def _resolve_operand(runs: Dict[str, dict], reference: str) -> float:
    label, separator, metric = str(reference).partition(".")
    if not separator or not label or not metric:
        raise EvidenceError(f"派生值引用格式应为 标签.指标：{reference!r}")
    entry = runs.get(label)
    if entry is None:
        raise EvidenceError(f"派生值引用了不存在的运行标签：{label}")
    metrics = entry.get("metrics") or {}
    if metric not in metrics:
        raise EvidenceError(f"{label} 没有声明指标 {metric}")
    return float(metrics[metric])


def _derived_result(entry: dict, runs: Dict[str, dict]) -> tuple:
    """按声明的算式重算派生值，返回（计算值，渲染文本）。"""
    operation = str(entry.get("op", ""))
    left = _resolve_operand(runs, entry.get("left"))
    right = _resolve_operand(runs, entry.get("right"))
    if operation == "subtract":
        value = left - right
        return value, f"{value:.2e}"
    if operation == "ratio":
        if right == 0:
            raise EvidenceError("比值派生值的分母为 0")
        value = left / right
        return value, f"{value:.3f}"
    if operation == "greater_than":
        return left > right, ""
    raise EvidenceError(f"未知的派生运算：{operation!r}")


def _compare_metric(
    label: str, key: str, declared: Any, actual: float,
    problems: List[str],
) -> None:
    try:
        declared_value = float(declared)
    except (TypeError, ValueError):
        problems.append(f"{label}.{key}={declared!r} 不是数值")
        return
    if abs(declared_value - actual) > 1e-9 * max(1.0, abs(actual)):
        problems.append(
            f"{label}.{key} 声明 {declared_value!r}，对应 summary 是 "
            f"{actual!r}")


def _check_derived(
    block: dict, runs: Dict[str, dict], problems: List[str],
) -> Tuple[List[str], Dict[str, set]]:
    """重算全部派生值，返回（可溯源的派生文本，按参与标签归类的文本）。"""
    rendered_values: List[str] = []
    by_label: Dict[str, set] = {}
    for entry in block.get("derived") or []:
        identity = str(entry.get("id", ""))
        try:
            computed, rendered = _derived_result(entry, runs)
        except EvidenceError as exc:
            problems.append(f"派生值 {identity}: {exc}")
            continue
        if rendered:
            rendered_values.append(rendered)
            rendered_tokens = {
                _normalise_number(rendered),
                _normalise_number(rendered.replace("e+", "e")),
            }
            for label in _derived_labels(entry):
                by_label.setdefault(label, set()).update(rendered_tokens)
        declared = entry.get("value")
        if computed is True or computed is False:
            if bool(declared) != computed:
                problems.append(
                    f"派生值 {identity} 声明 {declared!r}，重算为 {computed}")
            continue
        try:
            declared_value = float(declared)
        except (TypeError, ValueError):
            problems.append(f"派生值 {identity} 的 value 不是数值")
            continue
        if str(entry.get("op")) == "subtract":
            if f"{declared_value:.2e}" != rendered:
                problems.append(
                    f"派生值 {identity} 声明 {declared_value!r}，重算为 "
                    f"{rendered}")
        elif f"{declared_value:.3f}" != rendered:
            problems.append(
                f"派生值 {identity} 声明 {declared_value!r}，重算为 "
                f"{rendered}")
    return rendered_values, by_label


def _derived_labels(entry: dict) -> set:
    """派生值引用的运行标签；引用格式非法时不在这里报错。"""
    labels = set()
    for reference in (entry.get("left"), entry.get("right")):
        label, separator, _metric = str(reference).partition(".")
        if separator and label:
            labels.add(label)
    return labels


def _run_aliases(entry: dict) -> List[str]:
    """运行条目的正文别名；未显式声明时用标签本身。"""
    declared = entry.get("prose_aliases")
    aliases = [str(item) for item in declared] if isinstance(
        declared, list) else []
    label = str(entry.get("label", ""))
    if label and label not in aliases:
        aliases.append(label)
    return [alias for alias in aliases if alias]


def _label_spans(prose: str, runs: Dict[str, dict]) -> List[Tuple[int, int, str]]:
    """正文里运行标签/别名的出现区间，按起始位置排序。"""
    lowered = prose.lower()
    spans: List[Tuple[int, int, str]] = []
    for label, entry in runs.items():
        for alias in _run_aliases(entry):
            needle = alias.lower()
            start = lowered.find(needle)
            while start >= 0:
                spans.append((start, start + len(needle), label))
                start = lowered.find(needle, start + 1)
    return sorted(spans)


def _token_context(
    prose: str, position: int, spans: List[Tuple[int, int, str]],
) -> Tuple[str | None, set]:
    """数值所在块里前置的运行标签：返回（最近标签，全部前置标签）。

    列表项与表格行即使只用单个换行分隔也是独立的块，语义上各自成句；
    按空行切段会让相邻条目互相串味。
    """
    line_start = prose.rfind("\n", 0, position) + 1
    line_end = prose.find("\n", position)
    if line_end < 0:
        line_end = len(prose)
    window_start = _block_start(prose, line_start, line_end, position)
    preceding = [
        (start, label) for start, _end, label in spans
        if window_start <= start < position
    ]
    if not preceding:
        return None, set()
    nearest = max(preceding, key=lambda item: item[0])[1]
    return nearest, {label for _start, label in preceding}


def _block_start(
    prose: str, line_start: int, line_end: int, position: int,
) -> int:
    """当前数值所属块的起点：表格行/列表项各自成块，其余按空行分段。"""
    current = prose[line_start:line_end].lstrip()
    if current.startswith("|") or _LIST_MARKER.match(current):
        return line_start
    paragraph = prose.rfind("\n\n", 0, position)
    start = paragraph + 2 if paragraph >= 0 else 0
    last = start
    offset = start
    while offset < line_start:
        end = prose.find("\n", offset)
        if end < 0 or end >= line_start:
            break
        line = prose[offset:end]
        if line.lstrip().startswith("|") or _LIST_MARKER.match(line.lstrip()):
            last = offset
        offset = end + 1
    return last


def _prose_tokens(prose: str) -> List[Tuple[str, int]]:
    """正文里需要溯源的高精度数值及其起始位置。"""
    tokens: List[Tuple[str, int]] = []
    for pattern in NUMBER_PATTERNS:
        for match in pattern.finditer(prose):
            if match.groups():
                tokens.append((match.group(1), match.start(1)))
            else:
                tokens.append((match.group(0), match.start(0)))
    return tokens


def _measurement_values(measurements: Iterable[dict]) -> set:
    """独立测量项允许的数值写法。"""
    values = set()
    for entry in measurements:
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        values.update(_normalise_number(text) for text in (
            format_number(value), f"{value:.2e}", f"{value:.3f}",
            f"{value:.15e}", f"{value:.17g}", repr(value)))
    return values


def _metric_values(entry: dict) -> set:
    """某个运行条目声明的三项指标允许的正文写法。"""
    values = set()
    for value in (entry.get("metrics") or {}).values():
        try:
            values.add(_normalise_number(format_number(float(value))))
        except (TypeError, ValueError):
            continue
    return values


def _fact_renderings(value: Any) -> set:
    """事实值在正文里的常见写法：千分位、纯整数、3 位小数与科学计数法。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return set()
    renderings = {format_number(number), f"{number:.2e}", f"{number:.3f}"}
    if number == int(number):
        renderings.add(f"{int(number):,}")
        renderings.add(str(int(number)))
    return {text for text in renderings if text}


def _fact_bindings(
    block: dict,
) -> Tuple[Dict[str, set], List[Tuple[str, str, set]]]:
    """解析 `facts`：返回（按运行允许的归一化值，正文事实字面 → 允许运行）。

    同一事实值可被多个运行声明；正文出现时最近标签必须属于这些运行之一。
    """
    allowed_by_label: Dict[str, set] = {}
    texts: Dict[str, set] = {}
    runs_by_token: Dict[str, set] = {}
    for entry in block.get("facts") or []:
        run = str(entry.get("run") or "")
        if not run:
            continue
        for text in _fact_renderings(entry.get("value")):
            token = _normalise_number(text)
            allowed_by_label.setdefault(run, set()).add(token)
            texts.setdefault(token, set()).add(text)
            runs_by_token.setdefault(token, set()).add(run)
    return allowed_by_label, [
        (text, token, runs_by_token[token])
        for token, renderings in texts.items()
        for text in renderings
    ]


def _prose_context_problems(
    prose: str, runs: Dict[str, dict],
    derived_by_label: Dict[str, set], measurements: Iterable[dict],
    fact_allowed: Dict[str, set], fact_occurrences: List[Tuple[str, str, set]],
) -> List[str]:
    """正文数值必须属于它所在语境的运行标签，不能借用别的运行的值。

    运行标签只在数值前面找：正文里「dynamic 88.483 s、dense 91.012 s」
    这类写法把归属写在了值的前面。表格按行、正文按段落与列表项取所在块，
    数值只归属**最近**的前置标签；同块多运行不能互相借值。measurements
    是独立命名空间，任何语境都允许；带 `run` 的 facts 只能出现在对应运行
    的语境里。
    """
    spans = _label_spans(prose, runs)
    if not spans:
        return []
    measurement_values = _measurement_values(measurements)
    allowed: Dict[str, set] = {}
    for label, entry in runs.items():
        allowed[label] = (
            _metric_values(entry)
            | derived_by_label.get(label, set())
            | measurement_values
            | fact_allowed.get(label, set()))
    bad = set()
    for token, position in _prose_tokens(prose):
        labels = _effective_labels(prose, position, spans)
        if not labels:
            continue
        if any(_normalise_number(token) in allowed[label] for label in labels):
            continue
        bad.add((sorted(labels)[0], token))
    # facts 可能是不带小数的大整数，不在 NUMBER_PATTERNS 里；按字面逐次
    # 定位，确保容量、通信字节与 trace 计数也受运行语境约束。
    for text, _token, labels in fact_occurrences:
        start = prose.find(text)
        while start >= 0:
            effective = _effective_labels(prose, start, spans)
            if not effective or not (effective & labels):
                bad.add((sorted(effective)[0] if effective else "(无标签)",
                         text))
            start = prose.find(text, start + 1)
    return [
        f"正文数值 {token} 出现在 {label} 的语境里，但 {label} 没有这个值："
        "运行标签与数值归属不一致"
        for label, token in sorted(bad)
    ]


def _effective_labels(
    prose: str, position: int, spans: List[Tuple[int, int, str]],
) -> set:
    """数值归属的运行标签；「前者/后者」按块内最近两个运行解析。"""
    prefix = prose[max(0, position - 12):position]
    if "前者" in prefix or "后者" in prefix:
        labels = _nearest_labels(position, spans)
        if len(labels) >= 2:
            return {labels[-2] if "前者" in prefix else labels[-1]}
    nearest, _preceding = _token_context(prose, position, spans)
    return {nearest} if nearest else set()


def _nearest_labels(
    position: int, spans: List[Tuple[int, int, str]],
) -> List[str]:
    """该位置之前最近的两个不同运行标签，按出现顺序返回。"""
    chosen: List[str] = []
    for start, _end, label in reversed(spans):
        if start >= position:
            continue
        if label not in chosen:
            chosen.append(label)
        if len(chosen) == 2:
            break
    return list(reversed(chosen))


def _check_comparisons(
    prose: str, spans: List[Tuple[int, int, str]],
    allowed_by_label: Dict[str, set], allowed_values: set,
) -> List[str]:
    """正文里两个证据数值的大小结论必须与数值及运行身份一致。

    「A 与 B 中，前者的 X 低于后者的 Y」按块内最近的两个不同运行标签解析：
    前者取较早的标签，后者取较晚的标签；数值必须分别属于这两个运行。
    """
    problems = []
    for match in COMPARISON_PATTERN.finditer(prose):
        left_token = match.group(1)
        right_token = match.group(3)
        left = _normalise_number(left_token)
        right = _normalise_number(right_token)
        if left not in allowed_values or right not in allowed_values:
            continue
        try:
            left_value, right_value = float(left), float(right)
        except ValueError:
            continue
        operator = match.group(2)
        holds = (left_value > right_value if operator in ("高于", "大于", "超过")
                 else left_value < right_value)
        if not holds:
            problems.append(
                f"正文结论「{match.group(0).strip()}」与实际数值相反")
        between = prose[match.end(1):match.start(3)]
        prefix = prose[max(0, match.start() - 80):match.start(1)]
        if "前者" in prefix and "后者" in between:
            labels = _nearest_labels(match.start(1), spans)
            if len(labels) >= 2:
                former, latter = labels[-2], labels[-1]
                if (left not in allowed_by_label.get(former, set())
                        or right not in allowed_by_label.get(latter, set())):
                    problems.append(
                        f"正文结论「{match.group(0).strip()}」的运行身份不符："
                        f"前者应为 {former}，后者应为 {latter}")
    return problems


def _allowed_tokens(
    runs: Dict[str, dict], derived_rendered: Iterable[str],
    measurements: Iterable[dict], facts: Iterable[dict] = (),
) -> set:
    """正文允许出现的高精度数值：运行指标、派生值、独立测量与事实值。"""
    allowed = set()
    for entry in runs.values():
        allowed.update(_metric_values(entry))
    for rendered in derived_rendered:
        allowed.add(_normalise_number(rendered))
        allowed.add(_normalise_number(rendered.replace("e+", "e")))
    allowed.update(_measurement_values(measurements))
    for entry in facts:
        for text in _fact_renderings(entry.get("value")):
            allowed.add(_normalise_number(text))
    return allowed


def _unknown_numbers(doc_text: str, allowed: set) -> List[str]:
    prose = _prose_without_block(doc_text)
    unknown = set()
    for pattern in NUMBER_PATTERNS:
        for token in pattern.findall(prose):
            if _normalise_number(token) not in allowed:
                unknown.add(token)
    return sorted(unknown)


def _value_in_run_context(
    prose: str, text: str, label: str,
    spans: List[Tuple[int, int, str]],
) -> bool:
    """必需值必须在它自己的运行标签语境里出现，无标签前言不算引用。"""
    wanted = _normalise_number(text)
    for token, position in _prose_tokens(prose):
        if _normalise_number(token) != wanted:
            continue
        if label in _effective_labels(prose, position, spans):
            return True
    return False


def check_document(
    doc_text: str,
    summaries: Dict[str, Path],
    time_only: Iterable[str] = (),
) -> tuple:
    """返回（证据问题，正文里无法溯源的高精度数字）。"""
    problems: List[str] = []
    try:
        block = parse_evidence_block(doc_text)
    except EvidenceError as exc:
        return [str(exc)], []
    prose = _prose_without_block(doc_text)
    runs = _runs_by_label(block, problems)
    strata = {
        label: str(entry.get("stratum", "")) for label, entry in runs.items()
    }
    for label, stratum in strata.items():
        if stratum not in ("current", "history"):
            problems.append(
                f"{label} 的 stratum={stratum!r}，只允许 current/history")
    time_only = set(time_only)
    for label, path in summaries.items():
        entry = runs.get(label)
        if entry is None:
            problems.append(f"证据块缺少本次核验的 current 运行 {label}")
            continue
        if strata[label] != "current":
            problems.append(
                f"{label} 在证据块里标为 {strata[label]}，不能用本轮 "
                "summary 核验，也不能冒充 current 证据")
            continue
        metrics = summary_metrics(path)
        identity = summary_identity(path)
        if int(entry.get("layers", -1)) != identity["layers"]:
            problems.append(
                f"{label} 声明 {entry.get('layers')} 层，summary 是 "
                f"{identity['layers']} 层")
        if entry.get("artifact") and str(
                entry["artifact"]) != identity["artifact"]:
            problems.append(
                f"{label} 声明模型 {entry['artifact']}，summary 是 "
                f"{identity['artifact']}")
        if entry.get("semantics") and str(
                entry["semantics"]) != identity["semantics"]:
            problems.append(
                f"{label} 声明执行语义 {entry['semantics']}，summary 是 "
                f"{identity['semantics']}")
        declared = entry.get("metrics") or {}
        required = TIME_ONLY_KEYS if label in time_only else METRIC_KEYS
        for key in required:
            if key not in declared:
                problems.append(f"{label}.{key} 没有在证据块里声明")
            elif key not in metrics:
                problems.append(f"{label}.{key} 的 summary 里没有该指标")
            else:
                _compare_metric(
                    label, key, declared[key], metrics[key], problems)
        for key, value in declared.items():
            if key in metrics:
                _compare_metric(label, key, value, metrics[key], problems)
            elif key not in required:
                problems.append(
                    f"{label}.{key} 在 summary 里不存在，不能引用")
        for key in required:
            if key in declared:
                try:
                    text = format_number(float(declared[key]))
                except (TypeError, ValueError):
                    continue
                if not _value_in_run_context(
                        prose, text, label, _label_spans(prose, runs)):
                    problems.append(
                        f"{label}.{key}={text} 没有出现在 {label} 的运行"
                        "语境里（无标签前言不算正文引用）")
    # current 条目必须有本轮 summary 背书；history 用独立命名空间。
    for label, stratum in strata.items():
        if stratum == "current" and label not in summaries:
            problems.append(
                f"current 运行 {label} 没有对应的 --summary/--time-only "
                "产物，不能只凭文档自报")
    derived_rendered, derived_by_label = _check_derived(
        block, runs, problems)
    measurements = block.get("measurements") or []
    if not isinstance(measurements, list):
        problems.append("证据块的 measurements 必须是列表")
        measurements = []
    for entry in measurements:
        if str(entry.get("stratum", "history")) != "history":
            problems.append(
                f"测量项 {entry.get('id', '')} 必须标为 history 命名空间")
    facts = block.get("facts") or []
    if not isinstance(facts, list):
        problems.append("证据块的 facts 必须是列表")
        facts = []
    for entry in facts:
        run = str(entry.get("run") or "")
        if run and run not in runs:
            problems.append(
                f"事实项 {entry.get('id', '')} 引用了不存在的运行标签 {run}")
        if not _fact_renderings(entry.get("value")):
            problems.append(
                f"事实项 {entry.get('id', '')} 的 value 不是数值")
    fact_allowed, fact_occurrences = _fact_bindings(block)
    problems.extend(_prose_context_problems(
        prose, runs, derived_by_label, measurements,
        fact_allowed, fact_occurrences))
    allowed = _allowed_tokens(
        runs, derived_rendered, measurements, block.get("facts") or [])
    allowed_by_label = {
        label: (_metric_values(entry)
                | derived_by_label.get(label, set())
                | fact_allowed.get(label, set()))
        for label, entry in runs.items()
    }
    problems.extend(_check_comparisons(
        prose, _label_spans(prose, runs), allowed_by_label, allowed))
    unknown = _unknown_numbers(doc_text, allowed)
    return problems, unknown


def _parse_spec(spec: str) -> tuple:
    """解析命令行里的 `标签=路径`。"""
    label, separator, raw_path = spec.partition("=")
    if not separator or not label or not raw_path:
        raise SystemExit(f"参数应为 标签=summary 路径：{spec!r}")
    return label, Path(raw_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="核对实施文档数值证据与 summary 产物一致")
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--summary", action="append", default=[],
                        metavar="LABEL=PATH")
    parser.add_argument("--time-only", action="append", default=[],
                        metavar="LABEL=PATH")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.summary and not args.time_only:
        raise SystemExit("至少给一个 --summary 或 --time-only")
    summaries = {}
    time_only = set()
    for spec in args.summary:
        label, path = _parse_spec(spec)
        summaries[label] = path
    for spec in args.time_only:
        label, path = _parse_spec(spec)
        summaries[label] = path
        time_only.add(label)
    problems, unknown = check_document(
        args.doc.read_text(), summaries, time_only)
    for label, path in summaries.items():
        metrics = summary_metrics(path)
        shown = "，".join(
            f"{metric}={format_number(value)}"
            for metric, value in metrics.items())
        print(f"{label}: {shown}")
    if problems:
        print("证据问题：")
        for item in problems:
            print(f"- {item}")
    if unknown:
        print("文档存在无法溯源的高精度数字：", ", ".join(unknown))
    if problems or unknown:
        return 1
    print("实施文档数值与产物一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
