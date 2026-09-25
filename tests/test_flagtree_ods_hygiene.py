"""FlagTree 侧 ODS 定义的消费方守卫。

CLAUDE.md：不预造抽象、删优于加。一个属性若既没有 op 挂它、也没有 pass /
verifier 读它，那它在三条消费链上都不存在——留着只会让下一个人以为语义已落地。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import flagtree_prefix

def _source_root() -> Path:
    """FlagTree 源码树。安装树只有 build 产物，源码路径记在 CMakeCache 里。"""
    cache = (flagtree_prefix() / "build" / "flagtree-cmake" / "CMakeCache.txt")
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:"):
                return Path(line.split("=", 1)[1].strip())
    return flagtree_prefix()


_ROOT = _source_root()
_PIM = _ROOT / "include" / "triton" / "Dialect" / "TritonPIM" / "IR"
_ATTRS = _PIM / "PIMAttrDefs.td"
_OPS = _PIM / "PIMOps.td"
_LIB = _ROOT / "lib" / "Dialect" / "TritonPIM"

pytestmark = pytest.mark.skipif(
    not _ATTRS.is_file(), reason="FlagTree 源码树不在位")


def _defined_attrs() -> list[str]:
    """`PIMAttrDefs.td` 里定义的 `*Attr` 名字。"""
    text = _ATTRS.read_text()
    return re.findall(r"^def (TTPIM_\w+Attr)\b", text, re.M)


def _all_consumers() -> str:
    """所有可能引用属性的文本：ODS 两份 + C++ 实现。"""
    parts = [_OPS.read_text(), _ATTRS.read_text()]
    for path in sorted(_LIB.rglob("*.cpp")):
        parts.append(path.read_text())
    return "\n".join(parts)


def test_every_defined_attribute_has_a_consumer() -> None:
    """每个属性定义至少要被一个 op 挂上或被 C++ 读到。"""
    text = _all_consumers()
    orphans = []
    for name in _defined_attrs():
        # C++ 侧用的是 TableGen 生成的名字，没有 `TTPIM_` 前缀；ODS 侧用全名。
        # 两种拼法都要数，否则会把 `PhaseSpecAttr` 这类误判成没人用。
        cpp = name.removeprefix("TTPIM_")
        uses = len(re.findall(rf"\b{name}\b|\b{cpp}\b", text))
        definition = len(re.findall(rf"^def {name}\b", text, re.M))
        if uses - definition == 0:
            orphans.append(name)
    assert orphans == [], (
        f"这些属性定义了但没有任何消费方，按 CLAUDE.md 应删除或接上: {orphans}")


# 本仓 emitter 必须真的发这些 op 属性：ODS 上定义了、GML 侧有对应字段，
# 却没有任何一侧写它，等于那段语义只存在于方言自测里。
_MUST_BE_EMITTED = {
    # `pim.normalize` 的向量单元参数块 -> GML 的 `vpu_params` 子块（设计 §5.11）。
    "vpuParams": "normalize",
    # KV cache 操作数的暂存方式 -> GML 的 `use_input_buffer_1`（实测 "L2A_ignore"）。
    "inputBufferPolicy": "kv_cache",
}


def test_op_attributes_with_gml_fields_are_actually_emitted() -> None:
    """有 GML 落点的 op 属性必须由本仓 emitter 真的发出来。"""
    from opcompiler_bridge import oplevel_kernel

    text = Path(oplevel_kernel.__file__).read_text()
    missing = [attr for attr in _MUST_BE_EMITTED if attr not in text]
    assert missing == [], (
        f"这些 op 属性 ODS 上有、GML 侧有对应字段，但 emitter 从不发: {missing}")
