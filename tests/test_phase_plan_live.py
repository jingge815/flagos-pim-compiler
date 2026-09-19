"""真实跑一次 FlagTree 的 `-pim-expand-phases`，确认端到端没断。

`test_phase_plan.py` 用固化文本，没装 FlagTree 也能跑；这里真调 `triton-opt`，
所以能抓到「pass 没编进去」「属性名改了」「枚举没注册」这类只有跑起来才暴露的
退化。没装带 PIM pass 的 triton 就跳过。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts.gml_quant import PHASE_COUNTS
from genesim_bridge.paths import flagtree_prefix
from opcompiler_bridge.phase_plan import parse_phase_plans


def _triton_opt() -> Path | None:
    path = flagtree_prefix() / "build" / "flagtree-cmake" / "bin" / "triton-opt"
    return path if path.is_file() else None


def _has_expand_phases() -> bool:
    binary = _triton_opt()
    if binary is None:
        return False
    proc = subprocess.run([str(binary), "--help"], capture_output=True, text=True)
    return "--pim-expand-phases" in proc.stdout


pytestmark = pytest.mark.skipif(
    not _has_expand_phases(),
    reason="当前 triton-opt 没有 -pim-expand-phases，需重跑 0-install-flagtree.sh",
)

# 输入用整算子级 PIM IR：这一层才是 GML 的对应粒度。
_DQ_INPUT = """
module {
  tt.func @dq(%x: tensor<1x4096xf16>, %s: tensor<32xf16>,
              %o: tensor<1x4096x!tt.ptr<i8>>) {
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128>}
       : tensor<1x4096xf16>, tensor<32xf16> -> tensor<1x4096xi8>
    tt.store %o, %q : tensor<1x4096x!tt.ptr<i8>>
    tt.return
  }
}
"""

_SOFTMAX_INPUT = """
module {
  tt.func @sm(%s: tensor<1x1024xf16>, %o: tensor<1x1024x!tt.ptr<f16>>) {
    %p = pim.softmax %s {axis = 1 : i64, unit = #pim.unit<cstl>}
       : tensor<1x1024xf16> -> tensor<1x1024xf16>
    tt.store %o, %p : tensor<1x1024x!tt.ptr<f16>>
    tt.return
  }
}
"""

# MLP 中间态：11008 / 128 = 86 组，与参考产物的第二种分组宽度一致。
#
# 这里**不写 tt.store**：11008 不是 2 的幂，而 `tt.store` 要求元素数是 2 的幂
# （Triton 核心约束，与相位展开无关）。相位展开本身不关心这条，所以直接 return。
_DQ_MLP_INPUT = """
module {
  tt.func @dq_mlp(%x: tensor<1x11008xf16>, %s: tensor<86xf16>) {
    %q = pim.quantize %x, %s
       {dynamic, spec = #pim.quant_spec<granularity = per_group, axis = 1, groupSize = 128>}
       : tensor<1x11008xf16>, tensor<86xf16> -> tensor<1x11008xi8>
    tt.return
  }
}
"""


def _expand(text: str) -> str:
    binary = _triton_opt()
    assert binary is not None
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        proc = subprocess.run(
            [str(binary), path, "-pim-expand-phases"],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, f"triton-opt 失败:\n{proc.stderr}"
        return proc.stdout
    finally:
        Path(path).unlink()


def test_live_dq_expands_to_four_phases() -> None:
    plan = parse_phase_plans(_expand(_DQ_INPUT))["dq"]
    assert plan.count == PHASE_COUNTS["DynamicScaling"] == 4
    assert plan.kinds() == ["absmax", "relu", "reciprocal", None]
    assert plan.units() == ["vpu", "cstl", "cstl", "cstl"]


def test_live_softmax_expands_to_five_phases() -> None:
    plan = parse_phase_plans(_expand(_SOFTMAX_INPUT))["sm"]
    assert plan.count == PHASE_COUNTS["Softmax"] == 5
    assert plan.kinds() == ["max", "exp", "add", "reciprocal", "mul"]


def test_live_dq_group_width_follows_tensor() -> None:
    """分组宽度按张量变化：4096 -> 32 组，11008 -> 86 组。

    这是唯一按节点变化的相位字段，写死会让 MLP 那一路错。
    """
    hidden = parse_phase_plans(_expand(_DQ_INPUT))["dq"]
    mlp = parse_phase_plans(_expand(_DQ_MLP_INPUT))["dq_mlp"]
    assert hidden.phases[0].bytes == 32 * 2
    assert mlp.phases[0].bytes == 86 * 2
    assert hidden.phases[3].bytes == 4096
    assert mlp.phases[3].bytes == 11008


def test_live_opaque_operators_are_gone_after_expansion() -> None:
    """展开后原算子必须消失，否则下游会同时看到两种粒度。"""
    dq = _expand(_DQ_INPUT)
    assert "dynamic" not in dq
    sm = _expand(_SOFTMAX_INPUT)
    assert "pim.softmax" not in sm
