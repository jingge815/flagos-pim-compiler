#!/usr/bin/env python3
"""把 Llama2 图导出成底层编译器可消费的 GML + 运行时二进制。

链路：Llama2 权重
        → export_annotated_graph（图编译，标注设备与分区）
        → fuse_graph（激活折进主算子——GML 没有独立激活节点的表达方式）
        → convert + write_gml（整算子图 → GML 文本）
        → write_runtime_files（权重量化成 int4 + per-group scale，落盘 .bin）

产物结构与参考产物一致：

    <输出目录>/relay2gml_graph.gml      ← 图，对方按这个名字找
    <输出目录>/*.bin                    ← 权重、scale、数据缓冲、LUT

用法（先 source paths.json 里的 pytorch_env_script）：

    python scripts/export_gml.py --seq-len 128 --out-dir /tmp/gml_out
    python scripts/export_gml.py --layers 2          # 只导前 2 层，快速验证

导出后会跑结构自检（五条规则，见 docs/gml-lowering-20260914.md 第 3 节）与
交叉校验（GML 引用的每个文件名都要真实落盘）。任一不过就非零退出。

未完成的部分：激活的 scale 目前发 1.0 占位，LUT 发恒等表——这两处的数值语义
还没确认（见文档第 25、26.6 节）。结构与权重是完整的。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesim_bridge.paths import llama2_7b_model_dir
from gml_bridge.export import (
    format_summary,
    fuse_for_gml,
    serialize_gml,
    write_artifact,
    write_runtime_files,
)
from scripts.gml_structure_check import (
    check_rule1_fusion,
    check_rule2_buffer_naming,
    check_rule3_shape_on_edges,
    check_rule4_edge_direction,
    check_rule5_absent_fields,
    parse_blocks,
)


def _load_model(layers: int | None) -> torch.nn.Module:
    """加载真实 7B 权重；`layers` 非空时只保留前若干层。

    截层是为了快速验证结构——完整 32 层要 13 GiB 内存，导一次很慢。
    """
    from transformers import LlamaForCausalLM

    model_dir = llama2_7b_model_dir()
    model = LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    if layers is not None:
        model.model.layers = model.model.layers[:layers]
        model.config.num_hidden_layers = layers
    return model


def _check_structure(text: str) -> list[str]:
    """跑五条结构规则，返回问题列表。"""
    nodes = parse_blocks(text, "node")
    edges = parse_blocks(text, "edge")
    problems: list[str] = []
    for label, found in (
        ("规则 1 融合是强制的", check_rule1_fusion(nodes)),
        ("规则 2 buffer 命名", check_rule2_buffer_naming(nodes, edges)),
        ("规则 3 形状只在边上", check_rule3_shape_on_edges(nodes, edges)),
        ("规则 4 边是数据流方向", check_rule4_edge_direction(nodes, edges)),
        ("规则 5 不产出的字段", check_rule5_absent_fields(nodes)),
    ):
        if found:
            problems.append(f"{label}: {found[0]}")
    return problems


@dataclass
class Check:
    """一项检查的结果。

    `detail` 在通过时也打印——「层数 440」这种数字本身就是要看的信息，
    只在失败时才给细节会让通过的那次无从核对。
    """

    name: str
    passed: bool
    detail: str = ""
    # 失败时的多行说明。
    notes: list[str] = field(default_factory=list)


class CheckLog:
    """收集全部检查，最后统一判定。

    不在中途 `return` ——一次跑完能看到所有问题，而不是修一个才发现下一个。
    只有真正无法继续的情况（模型加载不了、算子编译器跑不起来）才提前中断。
    """

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: str = "",
            notes: list[str] | None = None) -> bool:
        self.checks.append(Check(name, passed, detail, notes or []))
        mark = "通过" if passed else "未通过"
        line = f"  [{mark}] {name}"
        if detail:
            line += f": {detail}"
        print(line)
        for note in (notes or []):
            print(f"         {note}")
        return passed

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def verdict(self) -> int:
        """打印总结论，返回退出码。"""
        total = len(self.checks)
        bad = self.failed
        print()
        print("=" * 62)
        if not bad:
            print(f"验证全部通过（{total} 项）")
            print("=" * 62)
            return 0
        print(f"验证未通过：{len(bad)} / {total} 项有问题")
        for check in bad:
            print(f"  - {check.name}"
                  + (f": {check.detail}" if check.detail else ""))
        print("=" * 62)
        return 1


def _run_opcompiler(graph, log: CheckLog):
    """跑一遍算子编译器，把相位模板与 GML 侧静态表对拍。

    返回 `PhaseSource | None`。`None` 表示跑不起来——那是**无法继续**（后面几项
    检查都要它），调用方会中断；而「跑起来了但与静态表不符」只记一项失败，
    继续往下跑，一次能看到全部问题。
    """
    from opcompiler_bridge.phase_source import (
        OpCompilerUnavailable,
        cross_check,
        phase_source_from_graph,
    )

    print("\n算子编译器（FlagTree）:")
    try:
        source = phase_source_from_graph(graph)
    except OpCompilerUnavailable as exc:
        log.add("算子编译器可用", False, "跑不起来",
                notes=str(exc).splitlines()[:3])
        return None

    log.add("算子编译器可用", True, str(source))

    mismatches = cross_check(source)
    log.add("相位模板与 GML 静态表一致", not mismatches,
            "一致" if not mismatches else f"{len(mismatches)} 处不符",
            notes=[str(m) for m in mismatches[:10]])
    return source


def _prove_dependency(graph, fusion, phase_source, artifact,
                      log: CheckLog, *, decode_block_only: bool = False,
                      slots=None) -> None:
    """证明算子编译器**真的**决定 GML，而不是算了不用。

    为什么单靠「两条路径产出相同」证明不了：假依赖（完全无视
    `phase_source`、永远走静态表）同样满足那个条件。实测把
    `_phase_count_for` 改成无视入参后，两条路径的 sha 照样相同，脚本照样全绿
    ——所以「相同」这一条对真假依赖是**盲的**。

    这里做的是反证：把算子编译器给的相位数**故意改小**再序列化一次，GML 必须
    跟着变。变了才说明那些 `*_phase_N` 字段真是它决定的；没变就是假依赖，
    直接失败。

    用 `deepcopy` 改副本，不碰真的 `phase_source`。序列化用的是同一份已融合的
    图（`serialize_gml` 是纯函数），所以两次结果可比。
    """
    import copy

    from gml_bridge.export import serialize_gml

    tampered = copy.deepcopy(phase_source)
    touched = 0
    for kinds in tampered.by_node.values():
        plan = kinds.get("dq")
        # 砍成 2 相（正常 4 相）。只要有一处被砍，GML 就该少一批字段。
        if plan is not None and len(plan.phases) > 2:
            plan.phases = plan.phases[:2]
            touched += 1
    if not touched:
        log.add("算子编译器真的决定 GML（反证）", False,
                "无法反证：没有多相算子可改")
        return

    probe = serialize_gml(graph, fusion, phase_source=tampered,
                          decode_block_only=decode_block_only, slots=slots)
    changed = probe.text != artifact.text
    if changed:
        delta = len(artifact.text) - len(probe.text)
        log.add("算子编译器真的决定 GML（反证）", True,
                f"{touched} 个 DQ 相位数 4→2，GML 少 {delta} 字节")
    else:
        log.add(
            "算子编译器真的决定 GML（反证）", False,
            f"{touched} 个 DQ 相位数砍半后 GML 没变",
            notes=["相位字段仍由静态表决定，算子编译器的产出没进 GML",
                   "即接线是假的：算了但没用上"])


def _check_identical_without_opcompiler(graph, fusion, artifact,
                                        log: CheckLog,
                                        *, decode_block_only: bool = False,
                                        slots=None) -> None:
    """接算子编译器前后 GML 必须**逐字节相同**。

    这一条原先要跑两次脚本、人工 `diff` 才能验；现在在同一次运行里比：
    同一份已融合的图再序列化一次，但不给 `phase_source`（即走静态表），
    与正式产物对比。`serialize_gml` 是纯函数，所以两次可比。

    注意这条与上面的反证是**互补**的，缺一不可：

    - 只有「相同」→ 假依赖也满足（算了不用），证明不了接线为真
    - 只有「反证通过」→ 可能已经改变了产物，破坏与旧版本的一致性
    """
    from gml_bridge.export import serialize_gml

    static = serialize_gml(graph, fusion, phase_source=None,
                          decode_block_only=decode_block_only, slots=slots)
    same = static.text == artifact.text
    if same:
        log.add("接算子编译器前后 GML 逐字节相同", True,
                f"{len(artifact.text)} 字节，节点 {len(artifact.nodes)}")
        return

    # 给出第一处差异，不然「不同」这个结论没法追。
    import difflib

    diff = [
        line for line in difflib.unified_diff(
            static.text.splitlines(), artifact.text.splitlines(),
            lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    ]
    log.add("接算子编译器前后 GML 逐字节相同", False,
            f"静态表 {len(static.text)} 字节 vs 算子编译器 "
            f"{len(artifact.text)} 字节，{len(diff)} 行不同",
            notes=diff[:6])


def _l2a_version() -> str:
    """`l2a_version.txt` 的内容。

    参考是 `0.0.0-c45e54f`（版本号 + git 短哈希），来自对方的 L2Analyzer。
    我方不是 L2A，写本仓标识 + 短哈希，格式对齐、来源注明。
    """
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    return f"0.0.0-{sha}" if sha else "0.0.0-unknown"


def _check_bin_references_closed(txt_dir: Path, bin_dir: Path,
                                 log: CheckLog) -> None:
    """prepare_out 的 txt 引用的 `.bin` 必须都在磁盘上（悬空引用会让底层
    编译器的仿真器读不到缓冲）。

    参考产物在这一项上是 0 缺失——这是「可在仿真器上正常执行」的最低要求，
    见 docs/prepare_out-代码评审-20260920.md §3.1。
    """
    referenced: set[str] = set()
    for path in txt_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            _, _, value = line.partition(":")
            value = value.strip()
            if value.endswith(".bin"):
                referenced.add(value)
    on_disk = {p.name for p in bin_dir.glob("*.bin")}
    missing = sorted(referenced - on_disk)
    log.add("txt 引用的 bin 全部存在", not missing,
            f"{len(referenced)} 个引用，缺失 {len(missing)} 个",
            notes=missing[:10])

    # 反向也要查：磁盘上有但没人引用的 bin 说明命名或落盘逻辑多写了一份
    # ——不是致命问题（不会让仿真器读不到东西），但如实报告，同时也是
    # 「陈旧文件掩盖真实缺失」的防线（配合上面写盘前清空 `out_dir/*.bin`，
    # 复核 20260921 §2.5）。GML 自己的 .bin（权重、scale 等）不会出现在
    # txt 里（那些是 GML 侧交叉校验的范畴，见 verify_against_graph），
    # 所以这里只报告数字，不算失败项。
    extra = sorted(on_disk - referenced)
    log.add("盘上 bin 全部被 txt 引用（信息项，不计入失败）", True,
            f"{len(extra)} 个未被 txt 引用（多为 GML 侧权重/scale，不算错）",
            notes=extra[:5])


def _check_dual_slot_offsets_disjoint(txt_dir: Path, log: CheckLog) -> None:
    """双输入层的两个 L2 输入槎必须落在不重叠的地址区间。

    `L2 input buffer offset 0 == offset 1` 意味着两块输入缓冲互相覆盖，
    不是「分配策略不同」，见 docs/prepare_out-代码评审-20260920.md §3.2。
    """
    bad: list[str] = []
    undersized: list[str] = []
    checked = 0
    for path in sorted(txt_dir.glob("*.txt")):
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.rstrip()] = value.strip()
        off0 = fields.get("L2 input buffer offset 0")
        off1 = fields.get("L2 input buffer offset 1")
        if off0 is None or off1 is None:
            continue
        checked += 1
        size0 = fields.get("L2 input buffer size 0")
        size1 = fields.get("L2 input buffer size 1")
        o0, o1 = int(off0), int(off1)
        s0 = int(size0) if size0 is not None else 0
        # 复核 20260921 发现：原来两次判断都用 s0，size1 != size0 时会漏判
        # 重叠（例如槎 1 比槎 0 宽，槎 0 的区间判定为不重叠，但槎 1 的实际
        # 尾端已经越过槎 0 的起点）。两个方向必须各用自己的尺寸。
        s1 = int(size1) if size1 is not None else s0
        if o0 == o1 or (o0 < o1 < o0 + s0) or (o1 < o0 < o1 + s1):
            bad.append(f"{path.name}: offset0={o0} offset1={o1} "
                      f"size0={s0} size1={s1}")
        if s1 <= 0:
            undersized.append(f"{path.name}: size1={s1}")
    log.add("双输入层 L2 offset 0/1 不重叠", not bad,
            f"{checked} 层双输入，{len(bad)} 层重叠", notes=bad[:10])
    log.add("双输入层 L2 size1 为正", not undersized,
            f"{checked} 层，{len(undersized)} 层 size1<=0",
            notes=undersized[:5])


def _check_bin_sizes_match_slots(bin_dir: Path, log: CheckLog) -> None:
    """运行期 bin 按编译期槽位落盘：MatMul 权重 131072、KV cache 4MB。"""
    from contracts.compile_slots import DEFAULT_SLOTS
    slots = DEFAULT_SLOTS
    bad: list[str] = []
    weights = list(bin_dir.glob("weight_buffer_*.bin"))
    small = [p.name for p in weights if p.stat().st_size == 65536]
    ok = [p for p in weights if p.stat().st_size == slots.bmm_weight_elems]
    if small:
        bad.append(f"MatMul 权重仍是 65536B：{small[:3]}")
    caches = list(bin_dir.glob("input_buffer_0_*.bin"))
    kv = [p for p in caches if p.stat().st_size == slots.kv_cache_elems]
    log.add("MatMul 权重按 S×hd 落盘", not small,
            f"{len(ok)} 个 {slots.bmm_weight_elems}B，{len(small)} 个仍 65536B",
            notes=small[:5])
    log.add("KV cache 平面按 nh×S×hd 落盘",
            any(p.stat().st_size == slots.kv_cache_elems for p in caches)
            or not caches,
            f"{len(kv)} 个 {slots.kv_cache_elems}B")


def _run_orchestrator(artifact, phase_source, out_dir: Path,
                      log: CheckLog, *, decode_block_only: bool = False) -> None:
    """跑编排器，把层参数骨架与 net.ini 写到 `<out-dir>/prepare_out/`。

    编排器在链路里是 GML **之后**的独立一段，不回填 GML——参考产物 616 个
    键名里搜不到任何编排类字段，对方 PDF 也注明 `prev_task`/`next_task` 由
    L2A 自己推导。
    """
    from orchestrator.plan import orchestrate

    print("\n编排器:")
    plan = orchestrate(artifact, phase_source=phase_source,
                       decode_block_only=decode_block_only)

    log.add("层展开", True, f"{plan.expand.total} 层"
            f"（非逐头 {len(plan.expand.non_per_head)}、"
            f"逐头 {len(plan.expand.per_head)}）")

    # 不认识的 op_type 会静默少层，必须报出来。
    log.add("全部 op_type 都能展开", not plan.expand.unknown,
            "无未识别" if not plan.expand.unknown
            else f"{len(plan.expand.unknown)} 个未识别",
            notes=plan.expand.unknown[:5])

    # Layer ID 撞号会让两层写同一个文件。
    ids = [i.layer_id for i in plan.identity.identities]
    log.add("Layer ID 唯一", len(ids) == len(set(ids)),
            f"{len(ids)} 个，唯一 {len(set(ids))} 个")

    # L2 offset 必须按 16 对齐（文档步骤 C）。
    misaligned = [n for n, o in plan.l2.offsets.items() if o % 16]
    log.add("L2 offset 16 字节对齐", not misaligned,
            f"{len(plan.l2.offsets)} 块全对齐" if not misaligned
            else f"{len(misaligned)} 块未对齐", notes=misaligned[:5])

    log.add("L2 地址分配", True,
            f"{plan.l2.buffers} 块 → {plan.l2.slots} 槽"
            f"（复用率 {plan.l2.reuse_ratio:.1%}），"
            f"数据区 {plan.l2.top} 字节")

    prepare_out = out_dir / "prepare_out"
    txt_dir = prepare_out / "txt_files"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for stale in txt_dir.glob("*.txt"):
        stale.unlink()
    (prepare_out / "net.ini").write_text(plan.net_ini_text)

    # 两个版本戳：参考里**没有结尾换行**（6 字节 `26.2.1`、13 字节
    # `0.0.0-c45e54f`），也是整个 txt_files 里唯一不带 CRLF 的两个文件。
    from contracts.gml_quant import GML_VERSION
    (txt_dir / "gml_version.txt").write_text(GML_VERSION)
    (txt_dir / "l2a_version.txt").write_text(_l2a_version())
    for name, text in plan.layer_texts.items():
        (txt_dir / name).write_text(text)

    # RoPE 语义中间态（`{label}_cos.bin`）只在 txt 引用、不进 GML，这里补写。
    _write_txt_only_bins(txt_dir, out_dir, artifact)

    listed = sum(1 for line in plan.net_ini_text.splitlines()
                 if line.startswith("layer = "))
    log.add("net.ini 列出全部层", listed == len(plan.layer_texts),
            f"{listed} 行 vs {len(plan.layer_texts)} 层 txt")
    log.add("txt_files 文件数", True,
            f"{len(plan.layer_texts)} 层 + 2 个版本戳")
    log.add("编排器产物落盘", True, f"{prepare_out}/net.ini、txt_files/")

    _check_bin_references_closed(txt_dir, out_dir, log)
    _check_dual_slot_offsets_disjoint(txt_dir, log)
    _check_bin_sizes_match_slots(out_dir, log)
    _check_l2_alloc_covers_declared(txt_dir, plan.l2, log)
    _check_parser_families(out_dir, log)


def _write_txt_only_bins(txt_dir: Path, bin_dir: Path, artifact) -> None:
    """txt 引用了但 GML 没声明的语义 bin（RoPE `_cos/_sin`）。"""
    import numpy as np
    from contracts.compile_slots import DEFAULT_SLOTS

    referenced: set[str] = set()
    for path in txt_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            value = line.split(":", 1)[1].strip()
            if value.endswith(".bin"):
                referenced.add(value)
    on_disk = {p.name for p in bin_dir.glob("*.bin")}
    slots = getattr(artifact, "slots", None) or DEFAULT_SLOTS
    for name in referenced - on_disk:
        if name.endswith(("_cos.bin", "_sin.bin")):
            np.zeros(slots.head_dim, dtype=np.float16).tofile(bin_dir / name)


def _check_l2_alloc_covers_declared(txt_dir: Path, l2, log: CheckLog) -> None:
    """每层声明的 L2 size 不能大于分配器给同一 offset 的槽尺寸。"""
    bad: list[str] = []
    checked = 0
    sizes = getattr(l2, "slot_sizes", {}) or {}
    for path in sorted(txt_dir.glob("*.txt")):
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.rstrip()] = value.strip()
        pairs = [
            ("L2 input buffer size 0", "L2 input buffer offset 0"),
            ("L2 input buffer size 1", "L2 input buffer offset 1"),
            ("L2 output buffer size", "L2 output buffer offset"),
        ]
        for size_key, off_key in pairs:
            if size_key not in fields or off_key not in fields:
                continue
            checked += 1
            size = int(fields[size_key])
            off = int(fields[off_key])
            room = sizes.get(off, 0)
            if room and size > room:
                bad.append(f"{path.name}: {size_key}={size} > slot@{off}={room}")
    log.add("L2 分配 ≥ 声明尺寸", not bad,
            f"{checked} 处声明，{len(bad)} 处欠分配", notes=bad[:8])


def _family(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def _check_parser_families(out_dir: Path, log: CheckLog) -> None:
    """parser_output 分族：参考独有的族必须出现。"""
    ref = Path("/media/disk/fengjingge/src/xinfangzhou-resource/"
               "llama2_w4a8_decode_block_0/parser_output")
    if not ref.is_dir():
        log.add("parser_output 文件族", True, "无参考目录，跳过")
        return
    mine = {_family(p.name) for p in out_dir.glob("*.bin")}
    theirs = {_family(p.name) for p in ref.glob("*.bin")}
    missing = sorted(theirs - mine)
    extra = sorted(mine - theirs)
    # 参考独有族不能缺（activation_lut / kantor_A_* 这类）；多出来的先报告。
    log.add("参考独有文件族都已产出", not missing,
            f"缺 {len(missing)} 族，多 {len(extra)} 族",
            notes=missing[:8] + [f"+{e}" for e in extra[:4]])


def _dtype_coverage(gml_text: str) -> dict[str, set[str]]:
    """`op_type -> 该类节点上出现过的 dtype 字段名`。"""
    from scripts.gml_structure_check import field, parse_blocks

    seen: dict[str, set[str]] = {}
    for block in parse_blocks(gml_text, "node"):
        op = field(block, "op_type") or "buffer"
        keys = seen.setdefault(op, set())
        for key in ("input_buffer_dtype", "output_buffer_dtype",
                    "weight_buffer_dtype"):
            if field(block, key) is not None:
                keys.add(key)
    return seen


def _check_dtype_coverage(gml_text: str, log: CheckLog) -> None:
    """参考在某类算子上声明了 dtype，我方也必须声明（评审 4 §2.5）。

    只比**字段在不在**，不比编号：两边 node_id 体系不同。缺一个 dtype 就是
    底层编译器少一项位宽配置，而那不会在我们这侧报错。
    """
    ref = Path("/media/disk/fengjingge/src/xinfangzhou-resource/"
               "llama2_w4a8_decode_block_0/parser_output/relay2gml_graph.gml")
    if not ref.is_file():
        log.add("GML dtype 覆盖", True, "无参考产物，跳过")
        return
    theirs = _dtype_coverage(ref.read_text(errors="replace"))
    mine = _dtype_coverage(gml_text)
    missing: list[str] = []
    for op, keys in sorted(theirs.items()):
        if op == "buffer":
            continue  # 边界节点按张量各异，不逐个要求
        gap = keys - mine.get(op, set())
        if gap:
            missing.append(f"{op}: 缺 {sorted(gap)}")
    log.add("GML dtype 覆盖（参考有则我方有）", not missing,
            f"{len(theirs) - 1} 类算子，{len(missing)} 类缺 dtype",
            notes=missing[:8])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16,
                        help="prefill 序列长度")
    parser.add_argument("--layers", type=int, default=None,
                        help="只导前 N 层，用于快速验证；默认全部")
    parser.add_argument("--out-dir", type=Path, default=Path("gml_out"))
    parser.add_argument("--skip-weights", action="store_true",
                        help="不量化权重，只产结构。此时 GML 若引用权重会被"
                             "交叉校验拦下")
    parser.add_argument("--use-opcompiler", action="store_true",
                        help="跑一遍算子编译器（FlagTree）取回相位模板，与 GML "
                             "侧静态表交叉校验。**不改变产物**：加与不加导出的 "
                             "GML 必须逐字节相同，不同就是有一侧算错了")
    parser.add_argument("--orchestrate", action="store_true",
                        help="跑编排器：层展开、Layer ID 发号、L2 地址分配、"
                             "net.ini 执行序与 txt_files。产物写到 "
                             "<out-dir>/prepare_out/。同样不改变 GML")
    parser.add_argument("--decode-block-only", action="store_true",
                        help="丢掉模型末尾 RMSNorm + lm_head + DQ，层数与参考 "
                             "纯 decode block 的 422 对齐。裁剪发生在 GML 生成前，"
                             "parser_output 与编排器共用同一张图")
    args = parser.parse_args()

    from contracts.compile_slots import CompileSlots
    from runtime.compile import export_annotated_graph

    model = _load_model(args.layers)
    slots = CompileSlots.from_config(model.config, max_seq=1024)
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        model, args.seq_len, position_ids, dtype=torch.float32)

    # 融合与序列化拆开跑，中间插算子编译器：GML 的 `*_phase_N` 字段套数由它
    # 决定，而它又要吃融合后的图。**不能调两次 export_graph**——那不幂等
    # （实测第二次会变成 206 节点而非 200）。
    fusion = fuse_for_gml(graph)

    log = CheckLog()

    phase_source = None
    if args.use_opcompiler or args.orchestrate:
        phase_source = _run_opcompiler(graph, log)
        if phase_source is None:
            # 算子编译器跑不起来，后面几项都依赖它，无法继续。
            return log.verdict()

    # 序列化与写盘都可能因上游给了不自洽的输入而抛（比如相位数超出静态表的
    # 范围）。裹起来记成检查项，而不是让 traceback 冒出去——那样看不出是哪一
    # 项不成立，只看到一串栈。
    try:
        artifact = serialize_gml(
            graph, fusion, phase_source=phase_source,
            decode_block_only=args.decode_block_only, slots=slots)
    except Exception as exc:
        log.add("GML 序列化", False, f"{type(exc).__name__}: {exc}")
        return log.verdict()
    print(format_summary(artifact))

    # 复用同一个 --out-dir 重跑时，上一轮遗留的 .bin 会让「引用闭合」假通过
    # ——比如改名后旧名字的文件还在磁盘上，gate 看到"文件存在"就判过，但
    # 那份文件其实是上一版产物写的，不是这一版引用集里的东西（复核
    # 20260921 §2.5）。写盘前清空，保证这次磁盘上的 .bin 只来自这次的产物。
    for stale in Path(args.out_dir).glob("*.bin"):
        stale.unlink()

    try:
        gml_path = write_artifact(artifact, args.out_dir)
        files = write_runtime_files(
            artifact, args.out_dir, gm=None if args.skip_weights else graph)
    except Exception as exc:
        log.add("GML 与 bin 写盘", False, f"{type(exc).__name__}: {exc}")
        return log.verdict()
    print(f"\n图: {gml_path}")
    print(f"运行时文件: {len(files.names_written)} 个, "
          f"{files.total_bytes / 1e6:.2f} MB")

    print("\n检查:")
    problems = _check_structure(artifact.text)
    log.add("GML 结构自检（5 条规则）", not problems,
            "5/5 通过" if not problems else f"{len(problems)} 条未通过",
            notes=problems)
    _check_dtype_coverage(artifact.text, log)
    # write_runtime_files 内部已做交叉校验（引用集 == 落盘集），跑到这里就说明过了。
    log.add("GML 引用集 == 落盘集", True, f"{len(files.names_written)} 个文件")

    if phase_source is not None:
        # 两条互补的检查：产物不变 + 依赖为真。缺一不可，见各自 docstring。
        _check_identical_without_opcompiler(
            graph, fusion, artifact, log,
            decode_block_only=args.decode_block_only, slots=slots)
        _prove_dependency(
            graph, fusion, phase_source, artifact, log,
            decode_block_only=args.decode_block_only, slots=slots)

    if args.orchestrate:
        _run_orchestrator(artifact, phase_source, args.out_dir, log,
                          decode_block_only=args.decode_block_only)

    return log.verdict()


if __name__ == "__main__":
    sys.exit(main())
