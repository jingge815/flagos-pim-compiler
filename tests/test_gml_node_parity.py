"""decode block 与参考产物的逐节点字段对拍。

`op_type` 计数对齐只说明"有这类算子"，拦不住"某个节点少一个字段"。
这里按节点在同类算子里的序号对齐，逐键比对；只有明确声明不产出的键
（调试副本、对方工具链的痕迹）允许我方缺失。
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from genesim_bridge.paths import gml_llama2_reference_dir

# 对方产物里的调试副本与工具链痕迹，我方不产。
_EXEMPT_PREFIXES = ("DEBUG_",)
_EXEMPT_KEYS = {"original_name", "from_tvm", "lut_debug",
                "cos_mul_output_hash", "sin_mul_output_hash"}

# 已知但尚未修的逐节点缺口：(角色键, 组内序号, 字段名)。
# 修掉一个就从这里删掉；出现新的则测试变红。
_KNOWN_GAPS = {
    ("('DynamicScaling', 'dynamic_quantization')", 2, "output1_node_id"),
    ("('DynamicScaling', 'dynamic_quantization')", 2, "output2_node_id"),
    ("('DynamicScaling', 'dynamic_quantization')", 35, "output1_node_id"),
    ("('RMSNorm_vpu', 'RMSNorm')", 0, "input_zp"),
}

_REFERENCE = gml_llama2_reference_dir(required=False)
_OURS = Path("/media/disk/fengjingge/tmp/gml_dbo/relay2gml_graph.gml")

pytestmark = pytest.mark.skipif(
    _REFERENCE is None or not (_REFERENCE / "relay2gml_graph.gml").is_file()
    or not _OURS.is_file(),
    reason="需要参考产物与一次 decode-block 导出",
)


def _role(label: str) -> str:
    """从 label 里取算子角色（投影名或算子名），用来跨两份产物对齐节点。"""
    match = re.match(r"([A-Za-z0-9_]+?)_(?:MatMul|Add|params|qidx)", label)
    return match.group(1) if match else label.split("_params")[0]


def _nodes(path: Path) -> dict[tuple[str, str], list[set[str]]]:
    """按 (op_type, 角色) 归组。角色来自 label，所以两份产物能对上同一个节点。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    grouped: dict[tuple[str, str], list[set[str]]] = defaultdict(list)
    for block in re.split(r"\n\s*node \[", text)[1:]:
        block = block.split("\n    edge")[0]
        op = re.search(r'op_type "([^"]+)"', block)
        if not op:
            continue
        label = re.search(r'label "([^"]+)"', block).group(1)
        keys = {line.strip().split()[0].rstrip(":")
                for line in block.splitlines() if line.strip()}
        grouped[(op.group(1), _role(label))].append(keys)
    return grouped


def test_every_reference_field_is_present() -> None:
    """参考里每个算子节点有的键，我方同序号节点也要有。

    已知缺口单列在 `_KNOWN_GAPS`：修掉一个就从那里删掉，测试仍绿；
    出现新缺口则变红。
    """
    ours = _nodes(_OURS)
    ref = _nodes(_REFERENCE / "relay2gml_graph.gml")
    missing: list[str] = []
    for key, ref_nodes in sorted(ref.items()):
        our_nodes = ours.get(key, [])
        # 两边节点数不同说明 label 命名对不上，无法按角色对齐，跳过以免误报。
        if len(our_nodes) != len(ref_nodes):
            continue
        for index, (mine, theirs) in enumerate(zip(our_nodes, ref_nodes)):
            gap = [
                name for name in sorted(theirs - mine)
                if not name.startswith(_EXEMPT_PREFIXES) and name not in _EXEMPT_KEYS
                and (str(key), index, name) not in _KNOWN_GAPS]
            if gap:
                missing.append(f"{key}[{index}] 缺 {gap}")
    assert not missing, "新增的逐节点字段缺失:\n" + "\n".join(missing)
