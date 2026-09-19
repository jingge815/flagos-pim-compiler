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
                      log: CheckLog) -> None:
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

    probe = serialize_gml(graph, fusion, phase_source=tampered)
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
                                        log: CheckLog) -> None:
    """接算子编译器前后 GML 必须**逐字节相同**。

    这一条原先要跑两次脚本、人工 `diff` 才能验；现在在同一次运行里比：
    同一份已融合的图再序列化一次，但不给 `phase_source`（即走静态表），
    与正式产物对比。`serialize_gml` 是纯函数，所以两次可比。

    注意这条与上面的反证是**互补**的，缺一不可：

    - 只有「相同」→ 假依赖也满足（算了不用），证明不了接线为真
    - 只有「反证通过」→ 可能已经改变了产物，破坏与旧版本的一致性
    """
    from gml_bridge.export import serialize_gml

    static = serialize_gml(graph, fusion, phase_source=None)
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


def _run_orchestrator(artifact, phase_source, out_dir: Path,
                      log: CheckLog) -> None:
    """跑编排器，把层参数骨架与 net.ini 写到 `<out-dir>/prepare_out/`。

    编排器在链路里是 GML **之后**的独立一段，不回填 GML——参考产物 616 个
    键名里搜不到任何编排类字段，对方 PDF 也注明 `prev_task`/`next_task` 由
    L2A 自己推导。
    """
    from orchestrator.plan import orchestrate

    print("\n编排器:")
    plan = orchestrate(artifact, phase_source=phase_source)

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
    prepare_out.mkdir(parents=True, exist_ok=True)
    (prepare_out / "net.ini").write_text(plan.net_ini_text)

    # 层清单：每层一行「文件名 task_id prev next L2 offset」。
    # 完整的 424 个 txt 还要等 B7 那批查表值确认（文档 Q17），这里先落清单，
    # 让层展开与发号的结果可核对。
    lines = ["# filename\ttask_id\tprev\tnext\tl2_offset"]
    for identity in plan.identity.identities:
        offset = plan.l2.offsets.get(identity.stem)
        lines.append("\t".join([
            identity.filename,
            str(identity.task_id),
            ",".join(str(t) for t in identity.prev_tasks) or "-",
            ",".join(str(t) for t in identity.next_tasks) or "-",
            "-" if offset is None else hex(offset),
        ]))
    (prepare_out / "layers.tsv").write_text("\n".join(lines) + "\n")

    # net.ini 的 `[layers]` 行数必须等于层数，少一行就是漏了一层。
    listed = sum(1 for line in plan.net_ini_text.splitlines()
                 if line.endswith(".txt"))
    log.add("net.ini 列出全部层", listed == plan.expand.total,
            f"{listed} 行 vs {plan.expand.total} 层")

    log.add("编排器产物落盘", True, f"{prepare_out}/net.ini、layers.tsv")


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
                             "net.ini 执行序。产物写到 <out-dir>/prepare_out/。"
                             "同样不改变 GML")
    args = parser.parse_args()

    from runtime.compile import export_annotated_graph

    model = _load_model(args.layers)
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
        artifact = serialize_gml(graph, fusion, phase_source=phase_source)
    except Exception as exc:
        log.add("GML 序列化", False, f"{type(exc).__name__}: {exc}")
        return log.verdict()
    print(format_summary(artifact))

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
    # write_runtime_files 内部已做交叉校验（引用集 == 落盘集），跑到这里就说明过了。
    log.add("GML 引用集 == 落盘集", True, f"{len(files.names_written)} 个文件")

    if phase_source is not None:
        # 两条互补的检查：产物不变 + 依赖为真。缺一不可，见各自 docstring。
        _check_identical_without_opcompiler(graph, fusion, artifact, log)
        _prove_dependency(graph, fusion, phase_source, artifact, log)

    if args.orchestrate:
        _run_orchestrator(artifact, phase_source, args.out_dir, log)

    return log.verdict()


if __name__ == "__main__":
    sys.exit(main())
