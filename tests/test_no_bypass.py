"""四维取数不得绕过统一 IR。

P0-5 的判据：三个 bridge 与内存/运行时不再自行从 PyTorch 或 numpy 推导
dtype 与字节宽度，字节宽度只从 `contracts.dtypes.dtype_bytes` 来。

**不是**「零 `.meta["val"]`」—— shape / ndim 与「写入 val 示例张量」两类是
FX 图自身的生态，必须保留（`val` 是 torch.export 的产物，dtype 字段正是从它
派生而来）。断言只针对 dtype 与字节宽度这两种取数。
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SCAN_DIRS = ("graph", "gml_bridge", "genesim_bridge", "opcompiler_bridge",
             "memory", "runtime", "comm", "contracts", "orchestrator")

# dtype 从 PyTorch 进入统一 IR 的唯一入口。多一处白名单 = 又开了一条旁路，
# 两个真源会漂移（跨维校验正是为此）。
ALLOWED = {"graph/spec_prop.py"}

# 三种旁路写法：
#   `.element_size()` / `.meta["val"].dtype` —— 直接问 PyTorch；
#   `np.dtype(<类型名>).itemsize` —— 拿 numpy 当第二份「类型名 → 字节宽度」表。
# 第三种必须一起扫：它与 `dtype_bytes` 是同一功能的两个实现，且**定义域不同**
# （`np.dtype("int4")` 抛 TypeError，而 `dtype_bytes("int4")` 是 1），
# 只扫前两种的话「单一真源」这条判据实际没被守住。
# 注意只禁「从类型名推」这一种：`entry.dtype.itemsize`（`dtype` 本身已是
# np.dtype 对象，见 `comm/plan.py`）是正常用法，不在其列。
_PATTERN = re.compile(
    r'\.element_size\(\)'
    r'|\.meta\[.val.\]\.dtype'
    r'|np\.dtype\([^)]*\)\.itemsize')


def _code_lines(text: str) -> list[str]:
    """去掉注释与文档字符串后的源码行，行号保持不变。

    直接对原文正则会把文档里举的例子也算成调用点 —— 注释里提一句
    「取代 `.meta["val"].element_size()`」就会误判。
    **只去注释与文档字符串**，代码里的 `meta["val"]` 是正常实参、要保留。
    """
    lines = text.splitlines()
    chars = [list(line) for line in lines]

    def blank(r1: int, c1: int, r2: int, c2: int) -> None:
        for row in range(r1, r2 + 1):
            if row - 1 >= len(chars):
                continue
            line_chars = chars[row - 1]
            start = c1 if row == r1 else 0
            end = c2 if row == r2 else len(line_chars)
            for col in range(start, min(end, len(line_chars))):
                line_chars[col] = " "

    # 1. 注释：tokenize 的列号即真实列号。
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(*tok.start, *tok.end)
    except tokenize.TokenError:
        pass

    # 2. 文档字符串：只有模块/类/函数的第一条字符串才是文档，其余字符串是代码。
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ["".join(line) for line in chars]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            blank(first.value.lineno, first.value.col_offset,
                  first.value.end_lineno, first.value.end_col_offset)
    return ["".join(line) for line in chars]


# 间接旁路：先把 val 取到变量、再从变量取 `.dtype`。逐行正则看不见中间那一步。

# 函数级例外。GML 路的 dtype 沿边传播（`_stamp_dtypes`），而那条路按设计就没有
# spec（SPECS 与 FUSED 是两条分叉），所以只能从 val 派生 —— 这是结构性例外，
# 不是漏网；登记成函数级而不是整个文件，免得 `from_fx.py` 其余部分跟着被豁免。
ALLOWED_VAL_DERIVED_FUNCS = {
    # 四维从 PyTorch 进入统一 IR 的唯一入口（与 ALLOWED 里那处同一函数）。
    "graph/spec_prop.py::_dtype_of",
    # 见上：GML 路没有 spec。
    "gml_bridge/from_fx.py::_cast_dtypes",
}


def _is_meta(node: object) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "meta"


def _reads_val(node: ast.AST) -> bool:
    """这棵子树里有没有 `meta["val"]` / `meta.get("val")` 形态。

    走语法树而不是 `get_source_segment`：后者按文件长度复制字符串，逐赋值点调用
    会让整个扫描退化成平方级（实测两个用例合计 31 秒）。
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and _is_meta(sub.value):
            if isinstance(sub.slice, ast.Constant) and sub.slice.value == "val":
                return True
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get" and _is_meta(sub.func.value)
                and sub.args and isinstance(sub.args[0], ast.Constant)
                and sub.args[0].value == "val"):
            return True
    return False


def _val_derived_dtype_uses(text: str) -> list[tuple[str, int]]:
    """把 val 取到变量、随后从那个变量取 `.dtype` 的位置（函数名, 行号）。

    判据放宽到「同一函数内既绑定了 val 又用了 `.<变量>.dtype`」，不追绑定与
    使用的前后顺序：宁可多报一处让人核对，也不要因为控制流而漏掉真实旁路。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    uses: list[tuple[str, int]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound: set[str] = set()
        for sub in ast.walk(func):
            if isinstance(sub, ast.Assign) and _reads_val(sub.value):
                bound.update(t.id for t in sub.targets if isinstance(t, ast.Name))
        if not bound:
            continue
        for sub in ast.walk(func):
            if (isinstance(sub, ast.Attribute) and sub.attr == "dtype"
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in bound):
                uses.append((func.name, sub.lineno))
    return sorted(set(uses))


def test_dtype_enters_the_ir_through_exactly_one_door() -> None:
    """除派生入口外，生产代码不得从 val 取 dtype 或 element_size。"""
    root = Path(__file__).parent.parent
    offenders: list[str] = []
    for d in SCAN_DIRS:
        for py in (root / d).rglob("*.py"):
            rel = f"{d}/{py.name}"
            if rel in ALLOWED:
                continue
            for i, line in enumerate(_code_lines(py.read_text(encoding="utf-8")), 1):
                if _PATTERN.search(line):
                    offenders.append(f"{rel}:{i}")
    assert offenders == [], \
        f"这些位置绕过统一 IR 直接问 PyTorch 取 dtype：{offenders}"


def test_docstring_mentions_are_not_counted_as_call_sites() -> None:
    """扫描器本身要能区分「代码里调用」与「注释里提到」。

    `contracts/dtypes.py` 的文档里就写着这个方法取代了哪个调用 ——
    它不该被当成一处旁路。
    """
    sample = '"""取代 `.meta["val"].element_size()`。"""\nx = node.meta["val"].dtype\n'
    lines = _code_lines(sample)
    assert not _PATTERN.search(lines[0]), "文档里的例子被当成了调用点"
    assert _PATTERN.search(lines[1]), "真正的调用点没被扫出来"


def test_the_whitelist_is_still_needed() -> None:
    """白名单里的那处派生入口仍然存在，且真的在做派生。

    白名单会随时间变成死配置（那个文件改完就没用了），这条守着它。
    """
    root = Path(__file__).parent.parent
    text = (root / "graph" / "spec_prop.py").read_text(encoding="utf-8")
    assert "def _dtype_of(" in text
    assert 'removeprefix("torch.")' in text

def test_the_scanner_catches_a_second_width_table() -> None:
    """扫描面要与判据对齐：拿 numpy 当第二份宽度表也算旁路。

    原先只扫 `.element_size()` 与 `.meta["val"].dtype` 两种写法，
    `np.dtype(name).itemsize` 扫不到 —— 于是测试通过并不代表
    「单一入口」成立（`runtime/exec_plan_gen.py` 与 `memory/mem_planner.py`
    当时各有一处）。这条钉住扫描面。
    """
    assert _PATTERN.search("    itemsize = np.dtype(edge.dtype).itemsize")
    # `dtype` 本身就是 np.dtype 对象时不算旁路：没有「类型名 → 宽度」这一步。
    assert not _PATTERN.search("    itemsize = entry.dtype.itemsize")


def test_byte_width_comes_only_from_the_single_source() -> None:
    """`bytes_of` 的 itemsize 实参只能来自 `dtype_bytes`。

    内存规划的宽度算错会静默错到 offset 与区间大小上，对拍器未必立刻报错，
    所以这里单独钉住取数来源（改动前有一处走的是 `np.dtype(...).itemsize`）。
    """
    root = Path(__file__).parent.parent
    text = (root / "memory" / "mem_planner.py").read_text(encoding="utf-8")
    for i, line in enumerate(_code_lines(text), 1):
        if re.search(r"\bitemsize\s*=", line) and "def " not in line:
            assert "dtype_bytes(" in line, \
                f"mem_planner.py:{i} 的字节宽度不是从真源来的：{line.strip()}"


def test_the_scanner_catches_a_two_step_val_dtype_read() -> None:
    """先把 val 取到变量、再从变量取 `.dtype` 也要算旁路。

    逐行正则看不见中间那一步：实测把它注入副本后，本文件与
    `test_unified_ir_contract.py` 一起 17 passed —— 判据在真实的生产写法上失效。
    """
    sample = ('def f(node):\n'
              '    src = node.meta.get("val")\n'
              '    return src.dtype\n')
    assert _val_derived_dtype_uses(sample) == [("f", 3)]


def test_no_production_code_reads_dtype_through_a_val_variable() -> None:
    """全仓非测试代码不得走两步形式取 dtype，除登记的函数级例外。"""
    root = Path(__file__).parent.parent
    offenders: list[str] = []
    for d in SCAN_DIRS:
        for py in (root / d).rglob("*.py"):
            rel = f"{d}/{py.name}"
            for func, line in _val_derived_dtype_uses(py.read_text(encoding="utf-8")):
                if f"{rel}::{func}" in ALLOWED_VAL_DERIVED_FUNCS:
                    continue
                offenders.append(f"{rel}:{line} ({func})")
    assert offenders == [], (
        f"这些位置通过中间变量从 val 取 dtype，绕过统一 IR：{offenders}")


def test_the_val_derived_exception_is_still_needed() -> None:
    """登记的函数级例外仍然存在且仍然在做那件事。

    例外会随时间变成死配置 —— 函数改名或不再派生之后，这条要红。
    """
    root = Path(__file__).parent.parent
    for entry in ALLOWED_VAL_DERIVED_FUNCS:
        rel, func = entry.split("::")
        text = (root / rel).read_text(encoding="utf-8")
        assert f"def {func}(" in text, f"{entry} 指的函数不在了"
        assert _val_derived_dtype_uses(text), f"{entry} 已经不派生 dtype 了"
