#!/usr/bin/env python3
"""列出参考 GML 的全部字段族，作为第三轮「字段覆盖率」验证的基准清单。

参考产物是 ResNet50，只用来确定格式，不是内容基准。本脚本把它的字段按语义
归类，供序列化器逐族比对：每一族要么已发出，要么显式声明不适用。

用法：
    python scripts/gml_field_inventory.py <gml 路径> [--json 输出路径]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from scripts.gml_structure_check import parse_blocks

# 按语义分组。前缀以 * 结尾表示匹配该前缀下的所有字段族。
GROUPS = {
    "身份与拓扑": [
        "id", "node_id", "label", "name", "original_name", "idx", "from_tvm",
        "is_buffer", "input_count", "residual_input_buffer",
        "residual_output_buffer", "input0_node_id", "input1_node_id",
        "output0_node_id", "output1_node_id", "source", "target", "dims",
        "directed",
        "relay2gml_version",
    ],
    "算子类型": [
        "op_type", "activation_op_type", "activation_mode",
        "activation_special_operators", "clip_to_relu", "contraction",
    ],
    "数据缓冲": [
        "input_buffer", "output_buffer", "weight_buffer", "bias_buffer",
        "input_buffer_dtype", "output_buffer_dtype", "weight_buffer_dtype",
        "bias_buffer_dtype", "input_data_extensions", "output_data_extension",
    ],
    "量化 sf/zp": [
        "input_sf", "input_zp", "weight_sf", "weight_zp", "bias_sf", "bias_zp",
        "output_sf", "output_zp", "input_sf_dtype", "weight_sf_dtype",
        "bias_sf_dtype", "output_sf_dtype",
    ],
    "累加与定标": [
        "nmu_mode", "fpsu_mode", "fpsu_spc", "fpsu_spc_axis", "fpsu_spg",
        "fpsu_spg_axis", "fpsu_spg_group_size", "Scaling_buffer_file",
        "Scaling_PS_buffer_file", "Bias_buffer_file", "pooling_dtype",
    ],
    "kantor 重定标": ["kantor*"],
    "LUT 激活": ["activation_lut_file", "lut_debug"],
    "窗口几何": [
        "kernel_shape", "strides", "pads", "dilations", "group",
        "output_padding", "axis", "axes",
    ],
    "激活前中间态": ["Relu_*"],
    # 多输入算子的每个输入槽各带一套完整的定标配置，不是共用一套。
    "多输入槽": [
        "input_0_*", "input_1_*", "input_buffer_0*", "input_buffer_1*",
        "fpsu_0_*", "fpsu_1_*", "fpsu_mode_0", "fpsu_mode_1",
        "Scaling_buffer_file_0", "Scaling_buffer_file_1",
        "Scaling_PS_buffer_file_0", "Scaling_PS_buffer_file_1",
        "Bias_buffer_file_0", "Bias_buffer_file_1",
        "pooling_dtype_0", "pooling_dtype_1",
    ],
    "哈希校验": ["*_hash"],
    "调试副本": ["DEBUG*"],
    "其他": ["A", "use_dynamic_quantization", "subnetwork", "link_node"],
}

# PDF 注明不必产出的字段。值是 PDF 的原话。
NOT_REQUIRED = {
    "in_virtual": "L2A deducts it on its own",
    "out_virtual": "L2A deducts it on its own",
    "prev_task": "L2A will determine the execution flow",
    "next_task": "L2A will determine the execution flow",
    "subnetwork": "irrelevant, not used by NGC currently",
    "residual_input_buffer": "irrelevant（但实测大量存在，见文档规则 5）",
    "residual_output_buffer": "irrelevant（但实测大量存在，见文档规则 5）",
}


# 带槽位后缀的字段族。多输入算子给每个输入槽一套配置，后缀 0/1 是**槽位号**
# 而不是节点实例号，所以归一化时必须保留——否则 `fpsu_mode_0` 会被并进
# `fpsu_mode`，两者的区别（共用一套定标 vs 每槽一套）就看不见了。
SLOT_SUFFIXED = (
    "fpsu_mode", "pooling_dtype", "Scaling_buffer_file", "Bias_buffer_file",
    "Scaling_PS_buffer_file", "input_buffer", "input_sf", "input_zp",
    "input_buffer_dtype", "input_sf_dtype", "DEBUG_input_buffer",
    "DEBUG_input_buffer_float",
)

# llama2 的 Concat / Mask 最多 32 槽，0/1 两个槽位号不够。
_SLOT = re.compile(r"^(.*)_([0-9]|[12][0-9]|3[01])$")
# 相位字段：词干_phase_<n>，n 是相位号不是节点 id。
_PHASE = re.compile(r"^(.*)_phase_\d+$")
# RoPE 粘连名：fpsu_<n>_scale_axisLlama2Activation_* 归并到子块族。
_ROPE_GLUE = re.compile(r"^(fpsu_\d+_scale_axis)Llama2Activation_.*$")
_ROPE_UNIT = re.compile(
    r"^(fpsu_mode|pooling_dtype|Scaling_buffer_file|Scaling_PS_buffer_file|"
    r"Bias_buffer_file)_\d+_Llama2Activation_.*$")
_KANTOR_ROPE = re.compile(
    r"^(Kantor_[AB]_(?:spc|spg|Shift|scale_axis|scale_buffer_file|"
    r"bias_buffer_file|spg_axis|spg_group_size))_Llama2Activation_.*$")
_KANTOR_MODE_ROPE = re.compile(r"^kantor_mode_Llama2Activation_.*$")


def normalize(key: str) -> str:
    """把节点实例号去掉，保留槽位号和相位号。

    `input_sf_8` 里的 8 是节点 id，要去掉；`fpsu_mode_0` 里的 0 是输入槽号，
    要留下。llama2 的 Concat 有 32 槽，所以槽位号取 0..31 而不是只认 0/1。
    `*_phase_N` 的 N 是相位号，归并到词干_phase。RoPE 的粘连键名归并到子块族。
    """
    m = _ROPE_GLUE.match(key)
    if m:
        return m.group(1)
    m = _ROPE_UNIT.match(key)
    if m:
        return m.group(1) + "_Llama2Activation"
    m = _KANTOR_ROPE.match(key)
    if m:
        return m.group(1) + "_Llama2Activation"
    if _KANTOR_MODE_ROPE.match(key):
        return "kantor_mode_Llama2Activation"
    m = _PHASE.match(key)
    if m:
        return m.group(1) + "_phase"
    m = _SLOT.match(key)
    if m and m.group(1) in SLOT_SUFFIXED:
        return key  # 槽位号留下，整键就是族名
    stripped = re.sub(r"_\d+$", "", key)
    if stripped in SLOT_SUFFIXED and _SLOT.match(key):
        return key
    return stripped


def family_without_slot(name: str) -> str:
    """把槽位号换成 `N`。

    参考产物是 32 头的 decode block，本地用例的合成模型是 4 头，同一个算子
    的槽位个数天然不同。比「某算子带不带这一族槽位字段」时要把槽号抹平，
    否则两边永远对不上，判据就退化成「有没有这一族」。
    """
    return re.sub(r"(?<=\D)\d+(?=_|$)", "N", name)


def field_families_by_op(text: str) -> dict[str, set[str]]:
    """按 `op_type` 汇总字段族，槽位号抹平。没有 `op_type` 的缓冲节点不计。

    只看算子节点：缓冲节点不带 `op_type`，它那一套字段（`idx`、`from_tvm`
    之类）已在族级判据里表过态。
    """
    by_op: dict[str, set[str]] = {}
    for block in parse_blocks(text, "node"):
        op = re.search(r"^\s+op_type\s+\"([^\"]+)\"", block, re.M)
        if op is None:
            continue
        families = by_op.setdefault(op.group(1), set())
        for key, following in re.findall(
                r"^\s+([A-Za-z_][A-Za-z_0-9]*) (.)", block, re.M):
            # 值以 `[` 开头的是 contraction 里的子节点名，不是字段名。
            if key in ("node", "edge", "graph") or following == "[":
                continue
            families.add(family_without_slot(normalize(key)))
    return by_op


def field_families(text: str) -> Counter:
    """统计字段族出现次数，把节点实例号归并掉但保留槽位号。

    `contraction` 块内的子节点是 `fused_xxx [ ... ]` 形式，那个 `fused_xxx`
    是节点名而不是字段名，每个都唯一，计进去会把清单撑成噪声。它们后面跟的是
    `[` 而不是值，据此排除。
    """
    counts = Counter()
    for match in re.finditer(r"^\s+([A-Za-z_][A-Za-z_0-9]*) (.)", text, re.M):
        key, first = match.group(1), match.group(2)
        if key in ("node", "edge", "graph") or first == "[":
            continue
        counts[normalize(key)] += 1
    return counts


def matches(pattern: str, name: str) -> bool:
    if pattern.startswith("*") and pattern.endswith("*"):
        return pattern.strip("*") in name
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    if pattern.startswith("*"):
        return name.endswith(pattern[1:])
    return name == pattern


def classify(families: Counter) -> tuple[dict, list]:
    grouped: dict[str, list] = {}
    claimed = set()
    for group, patterns in GROUPS.items():
        hits = sorted(
            (name, count)
            for name, count in families.items()
            if any(matches(p, name) for p in patterns)
        )
        if hits:
            grouped[group] = hits
            claimed |= {name for name, _ in hits}
    return grouped, sorted(set(families) - claimed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gml", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    families = field_families(args.gml.read_text())
    grouped, unclassified = classify(families)

    print(f"字段族总数：{len(families)}，分 {len(grouped)} 组\n")
    for group, hits in grouped.items():
        print(f"【{group}】{len(hits)} 族")
        for name, count in hits:
            note = NOT_REQUIRED.get(name)
            suffix = f"   ← PDF: {note}" if note else ""
            print(f"    {name:<36} x{count}{suffix}")
        print()

    if unclassified:
        print(f"【未归类】{len(unclassified)} 族 —— 需要补进 GROUPS")
        for name in unclassified:
            print(f"    {name:<36} x{families[name]}")
        print()

    if args.json:
        args.json.write_text(json.dumps(
            {"families": dict(families),
             "groups": {g: dict(h) for g, h in grouped.items()},
             "unclassified": unclassified},
            ensure_ascii=False, indent=2))
        print(f"已写出 {args.json}")

    # 未归类字段说明清单不完整，这是要修的，不是可忽略的。
    return 1 if unclassified else 0


if __name__ == "__main__":
    sys.exit(main())
