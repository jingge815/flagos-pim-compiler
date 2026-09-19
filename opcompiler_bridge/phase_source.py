"""跑一次算子编译器，把相位模板取回来供 GML 侧校验。

链路：

    融合后的 FX 图
      → oplevel_emitter.emit_oplevel_mlir   整算子级 PIM MLIR
      → triton-opt -pim-fuse-activation -pim-expand-phases
      → phase_plan.parse_phase_plans        {函数名: PhasePlan}
      → 本模块按 FX 节点名重新索引          {FX 节点名: {kind: PhasePlan}}

为什么也跑 `-pim-fuse-activation`：图编译器已经做过融合（`graph/fuse.py`
等三个 pass），所以这一遍通常无事可做——它是**校验**，确认算子编译器认可
我方的融合结果。将来图编译器漏融合时它能补上。pass 是幂等的，已带
`activation` 的算子会跳过。

**当前定位是校验，不是替换。** GML 的字段仍由 `contracts/gml_hw_table.py`
产出，因为静态表里还有一批 pass 不产出的字段族（`flp_min_exp` 这类 FPSU
浮点域配置、LUT 文件名、Kantor 缓冲名）。贸然切换会丢字段，所以顺序是
先校验、再逐族迁移。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from torch.fx import GraphModule

from genesim_bridge.paths import flagtree_prefix
from opcompiler_bridge.oplevel_emitter import EmittedOp, emit_oplevel_mlir
from opcompiler_bridge.phase_plan import PhasePlan, parse_phase_plans

# 先融合、再展开。顺序不能反：展开后主算子已经变成相位链，
# `-pim-fuse-activation` 认的 `pim.matmul + pim.lut` 模式就不在了。
PASS_PIPELINE = ("-pim-fuse-activation", "-pim-expand-phases")


class OpCompilerUnavailable(RuntimeError):
    """算子编译器不可用（没编、或这份 triton-opt 没有 PIM pass）。"""


@dataclass
class PhaseSource:
    """一次算子编译的产物，按 FX 节点名索引。

    一个 FX 节点可能有两个 kind：K 路 RoPE 的锚点既是 `rope` 又是 `dq`
    （GML 的 `Llama2ActivationDQ`），所以内层还要按 kind 分。
    """

    by_node: dict[str, dict[str, PhasePlan]] = field(default_factory=dict)
    ops: list[EmittedOp] = field(default_factory=list)
    mlir: str = ""
    expanded: str = ""

    def plan(self, fx_name: str, kind: str) -> PhasePlan | None:
        return self.by_node.get(fx_name, {}).get(kind)

    def phase_count(self, fx_name: str, kind: str) -> int | None:
        plan = self.plan(fx_name, kind)
        return None if plan is None else plan.count

    def __str__(self) -> str:
        kinds: dict[str, int] = {}
        for op in self.ops:
            kinds[op.kind] = kinds.get(op.kind, 0) + 1
        detail = "、".join(f"{k} {v}" for k, v in sorted(kinds.items()))
        return f"算子编译器给出 {len(self.ops)} 个算子的相位模板（{detail}）"


def triton_opt_path() -> Path:
    return flagtree_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"


def opcompiler_available() -> bool:
    """这份 triton-opt 带相位展开 pass 吗。"""
    binary = triton_opt_path()
    if not binary.is_file():
        return False
    try:
        proc = subprocess.run(
            [str(binary), "--help"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--pim-expand-phases" in proc.stdout


def _run_passes(mlir_text: str) -> str:
    binary = triton_opt_path()
    if not binary.is_file():
        raise OpCompilerUnavailable(
            f"找不到 triton-opt: {binary}\n"
            "需要先在 FlagTree 里编译（bash 0-install-flagtree.sh）。"
        )

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(mlir_text)
        path = handle.name
    try:
        proc = subprocess.run(
            [str(binary), path, *PASS_PIPELINE],
            capture_output=True, text=True,
        )
    finally:
        Path(path).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise OpCompilerUnavailable(
            f"triton-opt 失败 (exit {proc.returncode}):\n{proc.stderr[:3000]}"
        )
    return proc.stdout


def phase_source_from_graph(gm: GraphModule) -> PhaseSource:
    """跑一次算子编译器，返回按 FX 节点名索引的相位模板。

    `gm` 必须已经跑过 `gml_bridge.export.export_graph` 的那串融合 pass。
    """
    report = emit_oplevel_mlir(gm)
    expanded = _run_passes(report.text)
    plans = parse_phase_plans(expanded)

    by_node: dict[str, dict[str, PhasePlan]] = {}
    for op in report.ops:
        plan = plans.get(op.func)
        if plan is None:
            continue
        by_node.setdefault(op.fx_name, {})[op.kind] = plan

    return PhaseSource(by_node=by_node, ops=report.ops,
                       mlir=report.text, expanded=expanded)


@dataclass
class PhaseMismatch:
    """一处相位模板与静态表不符。"""

    fx_name: str
    kind: str
    field: str
    from_opcompiler: object
    from_static_table: object

    def __str__(self) -> str:
        return (f"{self.fx_name}({self.kind}).{self.field}: "
                f"算子编译器={self.from_opcompiler!r} "
                f"静态表={self.from_static_table!r}")


def cross_check(source: PhaseSource) -> list[PhaseMismatch]:
    """把算子编译器的相位模板与 GML 静态表对拍。

    只比两边都产出的项：相位数、以及归约相的语义。两边对不上就说明其中一处
    退化了——这正是接这条链路的目的。返回空列表表示一致。
    """
    from contracts.gml_quant import PHASE_COUNTS

    # 静态表按 GML op_type 记相位数，这里换成 emitter 的 kind。
    expected_counts = {
        "dq": PHASE_COUNTS["DynamicScaling"],
        "softmax": PHASE_COUNTS["Softmax"],
        # RoPE 的 3 连在静态表里不是 phase 字段族，而是 ROPE_UNITS 子块，
        # 物理相位数 3 由参考产物实测确定（文档 §9.5）。
        "rope": 3,
    }
    # 归约相的 kind：DQ 用 absmax（组内对称动态范围），Softmax 用 max。
    expected_first_reduction = {"dq": "absmax", "softmax": "max"}

    mismatches: list[PhaseMismatch] = []
    for fx_name, kinds in sorted(source.by_node.items()):
        for kind, plan in sorted(kinds.items()):
            want = expected_counts.get(kind)
            if want is not None and plan.count != want:
                mismatches.append(PhaseMismatch(
                    fx_name, kind, "phase_count", plan.count, want))

            want_kind = expected_first_reduction.get(kind)
            if want_kind is not None and plan.phases:
                got = plan.phases[0].kind
                if got != want_kind:
                    mismatches.append(PhaseMismatch(
                        fx_name, kind, "phase0.kind", got, want_kind))
    return mismatches
