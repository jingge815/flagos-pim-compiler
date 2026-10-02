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
# 模块属性的真正写入方在这里（`mod->setAttr(AttrNumDpusName, ...)`），
# 不在 `lib/Dialect` 下。只扫后者会让属性名的断言由 ODS 的散文满足。
_CONV = _ROOT / "lib" / "Conversion" / "TritonToTritonPIM"

pytestmark = pytest.mark.skipif(
    not _ATTRS.is_file(), reason="FlagTree 源码树不在位")


def _defined_attrs() -> list[str]:
    """`PIMAttrDefs.td` 里定义的 `*Attr` 名字。"""
    text = _ATTRS.read_text()
    return re.findall(r"^def (TTPIM_\w+Attr)\b", text, re.M)


def _all_consumers() -> str:
    """所有可能引用属性的文本：ODS 两份 + C++ 实现 + 方言头。

    头文件要一起读：跨仓的属性名以 `constexpr char AttrTileMName[]` 这种
    常量定义在 `Dialect.h` 里，只扫 `.td` 与 `.cpp` 会漏掉它们。
    """
    parts = [_OPS.read_text(), _ATTRS.read_text()]
    for path in sorted(_PIM.glob("*.h")):
        parts.append(path.read_text())
    for root in (_LIB, _CONV):
        for path in sorted(root.rglob("*.cpp")):
            parts.append(path.read_text())
    return "\n".join(parts)


def _flagtree_attr_sites() -> dict:
    """属性名 → 真正读写它的 FlagTree 源文件（相对路径，保序去重）。

    两步，而不是在整块文本里找子串：

    1. 从 `Dialect.h` 取「名字 → 常量标识符」（`AttrTileMName[] = "pim.tile-m"`）。
       带引号整段匹配，不做子串 —— `pim.wram-bytes` 是 `pim.wram-bytes-used`
       的前缀，`name in text` 会让前者被后者满足，「改名就失败」因此不成立
       （实测：只改前者的名字，旧断言照旧通过）。
    2. 再按**词边界**找哪些 `.cpp` 用了那个常量。名字只在头文件里定义、没有任何
       读写点时，这里得到空列表 —— 旧写法把「文档里提过」也算作读者，
       `pim.num-dpus` / `pim.num-tasklets` 就在 ODS 的 `description` 散文里
       出现过，于是那两条断言可以不靠代码满足。

    扫描面含 `lib/Conversion/`：三个模块属性的写入方在那里，不在 `lib/Dialect`。
    """
    names = {}
    for path in sorted(_PIM.glob("*.h")):
        for ident, name in re.findall(
                r'constexpr\s+static\s+char\s+(\w+)\[\]\s*=\s*"(pim\.[\w.\-]+)"',
                path.read_text()):
            names[name] = ident

    sources = []
    for root in (_LIB, _CONV):
        for path in sorted(root.rglob("*.cpp")):
            sources.append((path.relative_to(_ROOT).as_posix(), path.read_text()))

    sites = {}
    for name, ident in names.items():
        sites[name] = [rel for rel, text in sources
                       if re.search(rf"\b{ident}\b", text)]
    return sites


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


# ---- P1-1 的第二半判据：两侧四维字段逐项对应 ----
#
# 需求 §5.3 的 P1-1 要求「解析 pimir 断言 + **交叉校验**」。前者在
# `tests/test_pimir_layout.py`；这里是后者：本仓写出去的字段名与 FlagTree
# ODS 定义的参数名逐项对上。两边任一侧改名，这条就失败 ——
# 而不是等到 `triton-opt` 报一个「unexpected key」再回头查。

def _ods_encoding_parameters() -> list[str]:
    """`TaskletTiledEncodingAttr` 在 ODS 里声明的参数名，保序。"""
    text = _ATTRS.read_text()
    start = text.index("def TTPIM_TaskletTiledEncodingAttr")
    body = text[start:start + 4000]
    return re.findall(r'ArrayRefParameter<"unsigned">:\$(\w+)', body)


def test_the_layout_encoding_fields_match_the_ods_parameters() -> None:
    """本仓写出的四个字段名 = ODS 声明的四个参数名，顺序也一致。

    MLIR 的 assembly format 按声明顺序打印，所以顺序不是风格问题：
    错序的文本解析不出来。
    """
    from contracts.mlir_layout import tasklet_tiled
    from contracts.op_contract import DpuShard

    text = tasklet_tiled((16, 128), shard=DpuShard(dim=0, num_dpus=2),
                         num_tasklets=16)
    written = re.findall(r"(\w+) = \[", text)
    assert written == _ods_encoding_parameters(), (
        f"本仓写出 {written}，ODS 声明 {_ods_encoding_parameters()}")


def test_the_module_attribute_names_exist_on_the_flagtree_side() -> None:
    """模块级硬件属性名必须是 FlagTree 认的那几个。

    这些名字是跨仓契约：本仓写、FlagTree 读。拼错不会报错 ——
    MLIR 接受任意模块属性，只是那条 pass 读不到、静默退回默认值。
    """
    from contracts.mlir_layout import module_attributes
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG

    # 钉在「代码里的字符串字面量」上，不是任意文本的子串：ODS 的
    # `let description` 散文里提到名字不算读者，而 `pim.num-dpus` /
    # `pim.num-tasklets` 恰好都在散文里出现过 —— 按子串扫会让这条恒真。
    sites = _flagtree_attr_sites()
    for attr in module_attributes(DEFAULT_HARDWARE_CONFIG):
        name = attr.split("=")[0].strip().strip('"')
        assert sites.get(name), (
            f"{name} 在 FlagTree 的 C++ 里找不到读写点（当前 {sites.get(name)}）")


def test_the_feedback_attribute_names_exist_on_the_flagtree_side() -> None:
    """回传载体读的属性名同样要在 FlagTree 侧找得到写入方。

    这是 P1-2 的跨仓半边：本仓读、FlagTree 写。FlagTree 改名而本仓不知道时，
    回传会静默变成全 None（「算子编译器没意见」），看不出是断了。
    """
    from contracts.ir_payloads import (
        LAYOUT_FEEDBACK_ATTRS,
        PLACEMENT_FEEDBACK_ATTRS,
    )

    sites = _flagtree_attr_sites()
    # 两组都是跨仓契约：布局/分块的调试回传，以及 Placement / dtype 维的正式
    # 回程。`PLACEMENT_FEEDBACK_ATTRS` 是本轮补进来的 —— 那四个 `pim.placed-*`
    # 此前不在扫描面内，改名只会让本仓静默读回 None。
    for name in LAYOUT_FEEDBACK_ATTRS + PLACEMENT_FEEDBACK_ATTRS:
        assert sites.get(name), (
            f"{name} 在 FlagTree 的 C++ 里找不到写入方（当前 {sites.get(name)}）")


def test_the_cross_check_is_actually_failable() -> None:
    """交叉校验必须「改一个名字就红」，否则它只是个恒真断言。

    `pim.wram-bytes` 是 `pim.wram-bytes-used` 的前缀。旧写法用
    `name in 整块文本` 判有没有读者，于是只改前者的名字，断言会被后者满足而
    照旧通过（实测）。这条钉住两点：名字按整段带引号取、读写点按词边界找。
    """
    from contracts.ir_payloads import LAYOUT_FEEDBACK_ATTRS

    assert "pim.wram-bytes" in LAYOUT_FEEDBACK_ATTRS
    assert "pim.wram-bytes-used" in LAYOUT_FEEDBACK_ATTRS

    # 子串写法在这里必然误判：短名字是长名字的前缀。
    blob = "\n".join(f'"{n}"' for n in _flagtree_attr_sites()
                      if n != "pim.wram-bytes")
    assert "pim.wram-bytes" in blob, "用例前提：子串扫法确实会被前缀吞并"

    # 带边界的取法必须判「没了」。
    names = {n for n in re.findall(r'"(pim\.[\w.\-]+)"', blob)}
    assert "pim.wram-bytes-used" in names, "用例前提：长名字仍在"
    assert "pim.wram-bytes" not in names, (
        "前缀被长名字吞并，这条交叉校验不可失败")


def test_the_attribute_names_are_anchored_on_real_code_sites() -> None:
    """断言要钉在「C++ 里真的读写它」上，不是「文本里出现过」。

    `pim.num-dpus` / `pim.num-tasklets` 在 ODS 的 `let description` 散文里
    出现过，所以按任意文本扫时，这两条可以不靠任何代码满足。这里要求每个
    名字都能指出至少一个 `.cpp` 读写点，并且扫描面必须含真正的写入方目录
    `lib/Conversion/`（三个模块属性在那里写）。
    """
    assert _CONV.is_dir(), f"写入方目录不在位：{_CONV}"
    sites = _flagtree_attr_sites()

    # 探针：这三个的写入方只在 lib/Conversion 下，漏掉那个目录就指不出写入点。
    for name in ("pim.num-dpus", "pim.num-tasklets", "pim.dma-align"):
        writers = [s for s in sites.get(name, []) if "lib/Conversion" in s]
        assert writers, (
            f"{name} 指不出 lib/Conversion 下的写入点，扫描面没盖住写入方")

    # ODS 散文不该构成读者：名字若只在 .td 里被提到，这里必须是空列表。
    ods = _OPS.read_text() + _ATTRS.read_text()
    assert "pim.num-dpus" in ods, "用例前提：ODS 散文里确实提到了这个名字"
