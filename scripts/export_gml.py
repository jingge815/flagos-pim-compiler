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

from genesim_bridge.paths import gml_llama2_reference_dir, llama2_7b_model_dir
from gml_bridge.export import (
    format_summary,
    fuse_for_gml,
    fill_weight_hashes,
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


def _family(name: str) -> str:
    return re.sub(r"\d+", "#", name)


# 已知的数量差异：逐条写明理由，销账后再从这张表拿掉。没写在这里的族必须
# 与参考逐族相等——静默漂移是这条判据要拦的东西。
_FAMILY_COUNT_ALLOW = {
    # 命名口径：参考用 `input_sf_#`，我方一部分写成 `input_#_sf_#`（带槽号）。
    "input_sf_#.bin", "input_#_sf_#.bin",
    "input_zp_#.bin", "input_#_zp_#.bin",
    "output_sf_#.bin", "output_zp_#.bin",
    "input_buffer_#.bin", "input_buffer_#_#.bin",
    # RoPE 两张表：参考把它们当图输入（`cos/sin_position_embedding.bin`），
    # 我方按张量各自命名，多出 4 个小文件。
    "self_attn_Reshape_#_qidx#_params_#_cos.bin",
    "self_attn_Reshape_#_qidx#_params_#_sin.bin",
    "self_attn_Reshape_qidx#_params_#_cos.bin",
    "self_attn_Reshape_qidx#_params_#_sin.bin",
    # 末段 Kantor 配置：我方 Q/K 两条 RoPE 各发一套，参考只发 K 那套。
    "Kantor_A_Llama#Activation_add_bias_buffer_file_#.bin",
    "Kantor_A_Llama#Activation_add_scale_buffer_file_#.bin",
    "Kantor_A_Shift_Llama#Activation_add_#.bin",
}


def _check_parser_families(out_dir: Path, log: CheckLog) -> None:
    """parser_output 分族：参考独有的族必须出现，且非白名单族数量必须相等。"""
    import collections
    ref = gml_llama2_reference_dir(required=False)
    if ref is None or not ref.is_dir():
        log.add("parser_output 文件族", True, "无参考目录，跳过")
        return
    mine_names = [_family(p.name) for p in out_dir.glob("*.bin")]
    theirs_names = [_family(p.name) for p in ref.glob("*.bin")]
    mine, theirs = set(mine_names), set(theirs_names)
    missing = sorted(theirs - mine)
    extra = sorted(mine - theirs)
    log.add("参考独有文件族都已产出", not missing,
            f"缺 {len(missing)} 族，多 {len(extra)} 族",
            notes=missing[:8] + [f"+{e}" for e in extra[:4]])

    mc = collections.Counter(mine_names)
    tc = collections.Counter(theirs_names)
    count_diff = []
    for fam in sorted(set(mc) | set(tc)):
        if fam in _FAMILY_COUNT_ALLOW:
            continue
        if mc.get(fam, 0) != tc.get(fam, 0):
            count_diff.append(f"{fam}: 我方 {mc.get(fam, 0)} / 参考 {tc.get(fam, 0)}")
    log.add("非白名单文件族数量与参考一致", not count_diff,
            f"{len(count_diff)} 族数量不同", notes=count_diff[:8])


def _dtype_coverage(gml_text: str) -> dict[str, set[str]]:
    """`op_type -> 该类节点上出现过的 dtype 字段名`。"""
    from scripts.gml_structure_check import field, parse_blocks

    seen: dict[str, set[str]] = {}
    for block in parse_blocks(gml_text, "node"):
        op = field(block, "op_type") or "buffer"
        keys = seen.setdefault(op, set())
        for key in ("input_buffer_dtype", "output_buffer_dtype",
                    "weight_buffer_dtype",
                    "input_sf_dtype", "output_sf_dtype", "weight_sf_dtype"):
            if field(block, key) is not None:
                keys.add(key)
    return seen


def _check_dtype_coverage(gml_text: str, log: CheckLog) -> None:
    """参考在某类算子上声明了 dtype，我方也必须声明（评审 4 §2.5）。

    只比**字段在不在**，不比编号：两边 node_id 体系不同。缺一个 dtype 就是
    底层编译器少一项位宽配置，而那不会在我们这侧报错。
    """
    ref_dir = gml_llama2_reference_dir(required=False)
    ref = None if ref_dir is None else ref_dir / "relay2gml_graph.gml"
    if ref is None or not ref.is_file():
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


def build_parser() -> argparse.ArgumentParser:
    """命令行参数表。单独一个函数，让默认值本身可被测试断言。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=16,
                        help="prefill 序列长度")
    parser.add_argument("--layers", type=int, default=None,
                        help="只导前 N 层，用于快速验证；默认全部")
    parser.add_argument("--out-dir", type=Path, default=Path("gml_out"))
    parser.add_argument("--skip-weights", action="store_true",
                        help="不量化权重，只产结构。此时 GML 若引用权重会被"
                             "交叉校验拦下")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每个 DQ 节点的标定中间态（absmax、四相），"
                             "供人工核对定位问题")
    parser.add_argument("--use-opcompiler", action="store_true",
                        help="跑一遍算子编译器（FlagTree）取回相位模板，与 GML "
                             "侧静态表交叉校验。**不改变产物**：加与不加导出的 "
                             "GML 必须逐字节相同，不同就是有一侧算错了")
    # CEVA 链路：按厂商 65 图逐层导出，参数取自厂商同名子图。
    parser.add_argument("--ceva-reference", type=Path, default=None,
                        help="厂商 65 图根目录（只读）。给出即进入 CEVA 模式")
    parser.add_argument("--stage", choices=("all", "prefill", "decode", "lm_head"),
                        default="all", help="CEVA 模式只导某一阶段，调试用")
    parser.add_argument("--layer-index", type=int, default=None,
                        help="CEVA 模式只导某一层，调试用")
    # 旧的自研 prepare_out 支线已删除，net.ini 由厂商 GC/L2A 生成。
    parser.add_argument("--orchestrate", action="store_true",
                        help=argparse.SUPPRESS)
    # 默认裁掉末尾 RMSNorm + lm_head + DQ：参考产物就是纯 decode block，
    # 带这一步导出的 node/edge 与它精确一致。整网导出走下面的反向开关。
    parser.add_argument("--no-decode-block-only", dest="decode_block_only",
                        action="store_false", default=True,
                        help="保留模型末尾 RMSNorm + lm_head + DQ 做整网导出。"
                             "默认只导 decode block，与参考产物对齐")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.orchestrate:
        parser.error("--orchestrate 已删除：prepare_out 由厂商 GC 生成，"
                     "CEVA 链路用 --ceva-reference")

    if args.ceva_reference is not None:
        if not args.use_opcompiler:
            # 设计 3.2：CEVA 链路的相位必须来自 FlagTree，并校验跨相依赖。
            parser.error("--ceva-reference 必须同时给 --use-opcompiler")
        return _main_ceva(args)

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
    if args.use_opcompiler:
        phase_source = _run_opcompiler(graph, log)
        if phase_source is None:
            # 算子编译器跑不起来，后面几项都依赖它，无法继续。
            return log.verdict()

    # 序列化与写盘都可能因上游给了不自洽的输入而抛（比如相位数超出静态表的
    # 范围）。裹起来记成检查项，而不是让 traceback 冒出去——那样看不出是哪一
    # 项不成立，只看到一串栈。
    try:
        from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
        artifact = serialize_gml(
            graph, fusion, phase_source=phase_source,
            decode_block_only=args.decode_block_only, slots=slots,
            hardware=DEFAULT_HARDWARE_CONFIG)
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
        files = write_runtime_files(
            artifact, args.out_dir, gm=None if args.skip_weights else graph,
            verbose=args.verbose)
    except Exception as exc:
        log.add("GML 与 bin 写盘", False, f"{type(exc).__name__}: {exc}")
        return log.verdict()

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

    # 权值指纹在**检查跑完之后**补：它要与刚落盘的权值 bin 字节对上，而那些
    # 字节写盘时才成形；提前补进 `artifact.text` 会让上面两条不变量检查比到
    # 一份带指纹的文本，比的就不是同一件事了。补完重出文本，GML 最后落盘。
    try:
        fill_weight_hashes(artifact, files)
        gml_path = write_artifact(artifact, args.out_dir)
    except Exception as exc:
        log.add("GML 落盘", False, f"{type(exc).__name__}: {exc}")
        return log.verdict()
    print(f"\n图: {gml_path}")
    print(f"运行时文件: {len(files.names_written)} 个, "
          f"{files.total_bytes / 1e6:.2f} MB")

    return log.verdict()


def _main_ceva(args) -> int:
    """CEVA 模式：逐图导出 GML + IO_info + bin，并写清单 bundle.json。

    权重先取厂商反量化浮点（已旋转、norm 已折叠）装进模型，bin 全部由我方重新生成。
    """
    from gml_bridge import ceva
    from gml_bridge.ceva import bundle_targets

    model = _load_model(args.layers)
    ceva.load_vendor_checkpoint(model, args.ceva_reference, include_head=True)
    targets = bundle_targets(args.stage, args.layer_index,
                             model.config.num_hidden_layers)
    print(f"CEVA 导出 {len(targets)} 张子图 -> {args.out_dir}")
    manifest = ceva.export_ceva_bundle(model, args.ceva_reference, args.out_dir, targets,
                                       use_opcompiler=args.use_opcompiler)
    print(f"清单: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
