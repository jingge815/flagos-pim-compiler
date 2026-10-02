"""统一 IR 的键登记契约。

四维信息挂在 node.meta 上，键名集中登记在 contracts/unified_ir.py。
本文件的三条断言是 P0-1「契约收口」的执行点：键集合恰好相等、四维键不得
用裸字符串读、登记表声明的载荷类型必须真实存在。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from contracts.unified_ir import (
    DIMENSIONS,
    DIM_DTYPE,
    DIM_INFRA,
    DIM_MEM_LAYOUT,
    DIM_OP_SEMANTICS,
    DIM_PLACEMENT,
    GRAPH_STAGE_KEY,
    META_KEYS,
    STAGE_EXPORTED,
    STAGE_FUSED,
    STAGE_PARTITIONED,
    STAGE_PLANNED,
    STAGE_SPECS,
    dimensions_of,
    mark_stage,
    meta_keys_of,
    require_stage,
    spec_of,
    stages_of,
)

# 登记表以外的读者目录：源码扫描的范围。
SCAN_DIRS = ("graph", "gml_bridge", "genesim_bridge", "opcompiler_bridge",
             "memory", "runtime", "comm", "contracts", "orchestrator")

# 阶段标记挂在 GraphModule 上而非 node.meta，扫描时按键名区分不了接收者，
# 显式排除。它描述整张图的状态，不进键登记表。
_GRAPH_LEVEL_KEYS = {GRAPH_STAGE_KEY}

# 载荷类型名不属本仓定义的：torch 的 FakeTensor 与 Python 内建 dict。
# 它们不需要在 contracts/ir_payloads.py 里有定义，其余都必须有。
_EXTERNAL_PAYLOADS = {"FakeTensor", "dict"}

_CONST_DEF = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"([^"]+)"', re.M)
# meta 的读写写法有五种。`setdefault` / `pop` / `update` 原先没在里面 ——
# 新建键只要走那三种就能与登记表静默分叉，而「键集合恰好相等」这条判据的
# 全部价值就在于没有第二种写法能溜过去。
_META_ACCESS = r'\.meta(?:\[|\.get\(|\.setdefault\(|\.pop\(|\.update\(\{?)'
_META_READ = re.compile(
    _META_ACCESS + r'\s*(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))')


def _bare_string_meta_read() -> "re.Pattern[str]":
    """裸字符串读取的扫描面，与 `_META_READ` 同一个 `_META_ACCESS`。

    两半用两份正则就会各漂各的：一种写法在集合相等那条判据里算了、在裸字符串
    那条判据里没算，后者照样能漏。
    """
    return re.compile(_META_ACCESS + r'\s*"([^"]+)"')


def _scan_meta_keys() -> dict[str, set[str]]:
    """全仓非测试代码读写的 meta 键 → 出现位置。

    两趟扫描：先收齐所有 `XXX = "yyy"` 的常量定义，再解析 `.meta[...]` 的实参。
    一趟扫描会因文件遍历顺序而把尚未定义过的常量当成字面量键名。
    """
    root = Path(__file__).parent.parent
    files = [py for d in SCAN_DIRS for py in (root / d).rglob("*.py")]

    consts: dict[str, str] = {}
    texts = {py: py.read_text(encoding="utf-8") for py in files}
    for text in texts.values():
        consts.update(_CONST_DEF.findall(text))

    found: dict[str, set[str]] = {}
    for py, text in texts.items():
        for lit, name in _META_READ.findall(text):
            key = lit or consts.get(name, name)
            found.setdefault(key, set()).add(str(py.relative_to(root)))
    return found


def test_registered_keys_exactly_match_keys_in_use() -> None:
    """契约登记的键集合 == 全仓实际在用的键集合。

    多一个 = 契约里有没人用的键（该删或该标注）。
    少一个 = 有 pass 私建了键，契约漏登记 —— 这是 P0-1 要防的主要腐化。
    """
    in_use = set(_scan_meta_keys()) - _GRAPH_LEVEL_KEYS
    registered = {s.key for s in META_KEYS}
    assert in_use == registered, (
        f"契约与实际不符。\n仅在用未登记：{sorted(in_use - registered)}"
        f"\n仅登记未在用：{sorted(registered - in_use)}")


def test_no_four_dimension_key_is_read_by_bare_string() -> None:
    """四维键不得用裸字符串读取（val 等基础设施键豁免）。

    反例就是 graph/kv_dma_pass.py:122,151 —— 跨 pass 传递语义却不走常量，
    改名时不会有任何报错。
    """
    root = Path(__file__).parent.parent
    literal = _bare_string_meta_read()
    offenders: list[str] = []
    for d in SCAN_DIRS:
        for py in (root / d).rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                for key in literal.findall(line):
                    if dimensions_of(key) != (DIM_INFRA,):
                        offenders.append(f"{py.relative_to(root)}:{i} 读了 {key!r}")
    assert offenders == [], f"四维键被裸字符串读取：{offenders}"


def test_every_payload_type_actually_exists() -> None:
    """登记表声明的载荷类型名必须在契约层有定义。

    这把「键」与「类型」两半绑起来 —— 键与类型分处两地就没法静态校验。
    """
    import contracts.ir_payloads as payloads
    import contracts.pim_tensor_spec as tensor_spec

    missing: list[str] = []
    for spec in META_KEYS:
        base = spec.payload.removeprefix("list[").removesuffix("]")
        if base in ("str", "int", "bool", "float") or base in _EXTERNAL_PAYLOADS:
            continue
        if not (hasattr(payloads, base) or hasattr(tensor_spec, base)):
            missing.append(f"{spec.key} → {base}")
    assert missing == [], f"登记表声明的载荷类型不存在：{missing}"


def test_each_key_declares_a_producer() -> None:
    """每个键都要写清谁生产它。空 producer 说明契约没写完。"""
    for spec in META_KEYS:
        assert spec.producer, f"{spec.key} 没写生产方"


def test_dimensions_are_queryable() -> None:
    """四维都能查出自己的键，且至少有一个。"""
    for dim in DIMENSIONS:
        keys = meta_keys_of(dim)
        assert keys, f"维度 {dim} 没有任何键"
        assert all(dim in dimensions_of(k) for k in keys)


def test_unknown_key_or_dimension_raises() -> None:
    """未登记的键、未知维度都直接抛错 —— 契约不满足不静默。"""
    with pytest.raises(ValueError, match="未登记的 meta 键"):
        dimensions_of("not_a_key")
    with pytest.raises(ValueError, match="未知维度"):
        meta_keys_of("not_a_dimension")


def test_spec_of_raises_when_absent() -> None:
    """spec_of 缺 spec 直接抛错，不返回 None（否则调用方到处要判空）。"""
    from contracts.graph_meta import SPEC_META_KEY

    class _Node:
        name = "fake"
        meta: dict = {}

    with pytest.raises(ValueError, match="没有 spec"):
        spec_of(_Node())

    node = _Node()
    node.meta[SPEC_META_KEY] = "sentinel"
    assert spec_of(node) == "sentinel"


def test_dtype_and_mem_layout_live_inside_spec_not_at_top_level() -> None:
    """数据类型与 Memory Layout 都挂在 spec 内部，不另开顶层键。

    它们与 Placement 同源：shard_map 既是切分也是本地形状与地址，
    拆成两个顶层键会让同一个 TensorShardDetail 被两处引用而分裂。
    """
    top_level = {s.key for s in META_KEYS if s.dimensions != (DIM_INFRA,)}
    assert not any("dtype" in k for k in top_level)
    assert not any("stride" in k or "layout" in k for k in top_level)
    dims_of_spec = dimensions_of("spec")
    assert set(dims_of_spec) == {DIM_PLACEMENT, DIM_MEM_LAYOUT, DIM_DTYPE}


# ---- 阶段协议：登记的每个阶段都要可达，入口断言要真的挡住 ----

def test_every_registered_stage_has_a_marking_site() -> None:
    """登记了但没人标记的阶段是不可达的，会让人照登记表读出错误结论。

    `STAGE_PLANNED` 原先就是这种状态：登记在表里、全仓无 `mark_stage`
    调用点，而登记的生产方名字（`plan_memory`）还是个不存在的函数。
    `STAGE_EXPORTED` 是例外 —— 它是 `stages_of` 的隐含起点，不需要标记。
    """
    root = Path(__file__).parent.parent
    src = "\n".join(
        p.read_text(encoding="utf-8")
        for d in ("graph", "gml_bridge", "memory", "runtime", "opcompiler_bridge")
        for p in (root / d).rglob("*.py"))
    # 取 `mark_stage(<任意实参>, <阶段常量名>)` 里的第二个实参。
    marked = set(re.findall(r"mark_stage\([^,()]+,\s*(STAGE_[A-Z]+)\s*\)", src))
    for name in ("STAGE_PARTITIONED", "STAGE_SPECS", "STAGE_FUSED", "STAGE_PLANNED"):
        assert name in marked, f"{name} 没有标记点，是不可达阶段（已标记：{sorted(marked)}）"


def test_every_registered_producer_actually_exists() -> None:
    """登记的生产方要指向真实存在的函数。

    指向不存在的函数时，`require_stage` 的错误信息会把人引到一个找不到的
    地方去（原先登记的是 `mem_planner.plan_memory`，真实入口是 `plan_dpu`）。
    """
    import importlib

    from contracts.unified_ir import _STAGE_PRODUCER

    for stage, producer in _STAGE_PRODUCER.items():
        # 登记值可能带中文说明（「经 runtime.compile.compile_model」），取前半段。
        dotted = producer.split("（")[0].strip()
        module_path, _, func = dotted.rpartition(".")
        module = importlib.import_module(module_path)
        assert hasattr(module, func), \
            f"{stage} 登记的生产方 {dotted} 不存在"


def test_require_stage_blocks_a_graph_that_skipped_the_pass() -> None:
    """入口断言要真的抛错，不能是只写不读的登记簿。

    出口标记本身不解决顺序依赖 —— 前置断言才是执行点（设计 §3.3.3）。
    """
    import torch
    from torch.fx import symbolic_trace

    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm = symbolic_trace(M())
    assert stages_of(gm) == {STAGE_EXPORTED}
    with pytest.raises(ValueError, match="要求图已达到阶段"):
        require_stage(gm, STAGE_FUSED, who="test")
    # 标记后放行，且错误信息里带得出生产方名字。
    mark_stage(gm, STAGE_PARTITIONED)
    mark_stage(gm, STAGE_FUSED)
    require_stage(gm, STAGE_FUSED, who="test")


def test_require_stage_is_wired_at_a_real_entry() -> None:
    """`require_stage` 必须有生产代码调用者，否则是无调用者的公开 API。"""
    root = Path(__file__).parent.parent
    callers = [
        f"{d}/{p.name}"
        for d in ("graph", "gml_bridge", "memory", "runtime", "opcompiler_bridge")
        for p in (root / d).rglob("*.py")
        if "require_stage(" in p.read_text(encoding="utf-8")
    ]
    assert callers, "require_stage 没有任何生产代码调用者"


def test_the_key_scanner_covers_every_way_meta_is_read() -> None:
    """`.meta.setdefault / .pop / .update` 也是读写 meta 的写法。

    漏一类写法，新建键只要走它就能与登记表静默分叉 —— 而「键集合恰好相等」
    这条判据的全部价值就在于没有第二种写法能溜过去。
    """
    for sample in ('.meta.setdefault("pim_x", 1)',
                   '.meta.pop("pim_x")',
                   '.meta.update({"pim_x": 1})'):
        assert _META_READ.search(sample), f"扫描器看不见这种写法：{sample}"


def test_the_bare_string_check_covers_the_same_ways() -> None:
    """裸字符串那条判据用同一份扫描面，否则两半的覆盖面不一样。"""
    literal = _bare_string_meta_read()
    for sample in ('.meta["pim_x"]', '.meta.get("pim_x")',
                   '.meta.setdefault("pim_x", 1)', '.meta.pop("pim_x")',
                   '.meta.update({"pim_x": 1})'):
        assert literal.search(sample), f"扫描器看不见这种写法：{sample}"


def test_every_registered_consumer_really_touches_the_key() -> None:
    """`consumers` 一栏里出现的模块必须真的碰过那个键。

    这一栏是排查「谁在取四维信息」时唯一的索引。登记了一条不存在的读者，
    排查时就会以为那条路在用统一 IR，而它其实没有 —— 或者模块改名后这份索引
    静默失效。

    只查「登记的读者真的碰过它」这一个方向：反方向（有读者没登记）要求把
    `.meta[...]`、`.meta.get(...)`、`KEY in node.meta` 三种写法都认全，
    漏一种就会误报，那属于另一个问题。
    """
    from contracts.unified_ir import META_KEYS

    root = Path(__file__).parent.parent
    consts = {name: literal for name, literal in
              _CONST_DEF.findall((root / "contracts" / "unified_ir.py")
                                 .read_text(encoding="utf-8"))}
    offenders: list[str] = []
    for spec in META_KEYS:
        for consumer in spec.consumers:
            py = root / (consumer.replace(".", "/") + ".py")
            if not py.is_file():
                offenders.append(f"{spec.key} 登记了不存在的读者 {consumer}")
                continue
            text = py.read_text(encoding="utf-8")
            # 常量名（走常量的写法）或字面量（裸字符串写法）出现即可。
            names = [n for n, lit in consts.items() if lit == spec.key]
            if not any(n in text for n in names) and f'"{spec.key}"' not in text:
                offenders.append(f"{spec.key} 的读者 {consumer} 从没碰过这个键")
    assert offenders == [], "\n".join(offenders)
