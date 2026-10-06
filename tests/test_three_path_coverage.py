"""三条路径按算子汇总：缺一条就失败。

numpy 对拍、genesim 助记符、gml 字段族原先各用各的命名，缺产物看不出来。
这里按算子编译器的内核入口逐个核对三份归属，另加 A 路的 linear。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.op_semantics import OP_SEMANTICS, oplevel_ops

# genesim 的助记符不含这些算子，汇总时按归属注明，不算缺失。
# 值为 None 的没有 genesim 名字（convert 不是助记符，reduce 折进 RMSNorm）；
# 有值的归到那个算子名下，那个算子在 genesim 侧有名字才算覆盖到。
_GENESIM_COVERED_BY = {"linear": "matmul", "convert": None, "reduce": None}

# 没有独立 GML 名的内核入口：归约只出现在 RMSNorm 链里，GML 侧折进
# RMSNorm_vpu，不单发节点。
_NO_GML = {"reduce"}


def _covered() -> dict[str, set[str]]:
    """返回每个编译算子实际覆盖到的路径名。"""
    names = set(oplevel_ops()) | {"linear"}
    out = {name: set() for name in names}
    for spec in OP_SEMANTICS:
        entry = spec.kernel or (spec.name if spec.has_kernel else None)
        # gemm 是 linear 落在 GML 里的名字，两者是同一个算子。
        if spec.name == "gemm":
            entry = "linear"
        if entry not in out:
            continue
        if spec.is_mnemonic:
            out[entry].add("genesim")
        if spec.gml_op_type is not None:
            out[entry].add("gml")
    # numpy 对拍按算子名写在测试文件里，按真实调用形式核对，不另列一份清单。
    numpy_text = "\n".join(
        (Path(__file__).parent / f).read_text()
        for f in ("test_opcompiler_ops.py", "test_opcompiler_linear.py"))
    for name in _names_compiled_in(numpy_text):
        if name in out:
            out[name].add("numpy")
    return out


# 算子名只有出现在这两种调用形式里才算真的编过一次。直接搜子串会把注释、
# 参数列表、错误信息里的名字也算成覆盖：删掉真实用例后名字还在，汇总测试
# 照样是绿的。
_COMPILE_CALL_RE = re.compile(
    r"""op\s*=\s*["']([a-z_]+)["']"""
    r"""|_compile(?:_shapes)?\(\s*["']([a-z_]+)["']"""
)


def _names_compiled_in(text: str) -> set[str]:
    """从测试文件正文里取出真正被编译过的算子名。"""
    # 先按 # 截断每行，注释掉的调用形式不算覆盖。
    live = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    names = set()
    for match in _COMPILE_CALL_RE.finditer(live):
        names.add(match.group(1) or match.group(2))
    return names


def test_linear_is_covered_by_matmul_on_the_genesim_side() -> None:
    """genesim 没有 linear 这个名字，它并入 matmul，汇总时不算缺失。

    评审 r2 问题 4：归属表的取值原先没有任何用处，只查了键。这里把
    「linear 归到 matmul」这个取值本身钉住——matmul 不在 genesim 侧时，
    linear 要报缺口。
    """
    from tests import test_three_path_coverage as mod

    covered = mod._covered()
    covered["matmul"].discard("genesim")
    missing = mod._gaps(covered)
    assert "linear" in missing and "genesim" in missing["linear"], missing


def test_names_only_in_comments_are_not_coverage() -> None:
    """名字出现在注释里不算覆盖，只有真实调用形式才算。

    原先直接搜算子名子串，注释、参数列表、错误信息里的名字都会被当成
    「有对拍用例」。删掉真实用例后，只要名字还在文件里，汇总测试照样是绿的。
    """
    from tests.test_three_path_coverage import _names_compiled_in

    assert _names_compiled_in('# softmax 的说明文字\n') == set()
    assert _names_compiled_in('"""rope 的 docstring"""\n') == set()
    assert _names_compiled_in('raise ValueError("matmul 形状不对")\n') == set()

    assert _names_compiled_in('OpCompileRequest(op="softmax", arg_shapes=[(4, 16)])\n') == {"softmax"}
    assert _names_compiled_in("OpCompileRequest(op='rope', arg_shapes=[(4, 8)])\n") == {"rope"}
    assert _names_compiled_in('_compile("dynamic_quant", (1, 4096))\n') == {"dynamic_quant"}
    assert _names_compiled_in('_compile_shapes("normalize", [(4, 8)])\n') == {"normalize"}

    # 整行注释、行尾注释里的调用形式也不算覆盖：否则把对拍用例注释掉，
    # 汇总测试照样是绿的。
    assert _names_compiled_in('# _compile_shapes("gather", [(2, 64)])\n') == set()
    assert _names_compiled_in('x = 1  # 旧写法 _compile("rope", (4, 8)) 不再使用\n') == set()


def _gaps(covered: dict[str, set[str]]) -> dict[str, list[str]]:
    """返回每个算子缺的路径。

    genesim 侧按归属表折算：没有名字的不算缺，归到别的算子名下的要那个
    算子在 genesim 侧确实有名字才算覆盖到。
    """
    missing = {}
    for op, paths in sorted(covered.items()):
        need = {"numpy", "genesim", "gml"}
        covered_by = _GENESIM_COVERED_BY.get(op, ...)
        if covered_by is None:
            need.discard("genesim")
        elif covered_by is not ... and "genesim" in covered.get(covered_by, set()):
            need.discard("genesim")
        if op in _NO_GML:
            need.discard("gml")
        gap = need - paths
        if gap:
            missing[op] = sorted(gap)
    return missing


def test_three_paths_cover_every_compiled_op() -> None:
    """每个编译算子都要有三份产物的归属，缺一即失败。"""
    missing = _gaps(_covered())
    assert not missing, missing
