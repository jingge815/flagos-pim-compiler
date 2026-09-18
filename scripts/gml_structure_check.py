#!/usr/bin/env python3
"""校验一份 GML 是否符合目标格式的五条结构规则。

规则是从参考产物（ResNet50）逐条验证出来的，见 docs/gml-lowering-20260914.md 第 3 节。
第 3 轮的序列化器产出 GML 后用本脚本自检；现在先拿参考产物验证校验器本身是对的。

用法：
    python scripts/gml_structure_check.py <gml 路径>
"""

import argparse
import re
import sys
from pathlib import Path

# 缓冲区文件的扩展名。
_SUFFIX = ".bin"

# 可以折进主算子 contraction 块的算子。GML 不允许它们作为独立节点出现。
FUSED_ONLY_OPS = {"Lut", "Relu", "MaxPool", "AveragePool"}

# 硬件放置字段。GML 不表达算子跑在哪个 NPU 上，出现即说明混入了不该下推的信息。
PLACEMENT_FIELDS = [
    "pu_id", "npu", "npu_id", "dpu", "dpu_id", "core", "core_id",
    "tile", "tiling", "shard", "vpu_id", "node_assign", "device",
    "placement", "cluster", "rank",
]

# PDF 注明由对方 L2Analyzer 自行推导、不必产出的字段。
NOT_EMITTED_FIELDS = [
    "in_virtual", "out_virtual", "prev_task", "next_task", "subnetwork",
]


def parse_blocks(text: str, kind: str) -> list[str]:
    """切出全部 `node [ ... ]` 或 `edge [ ... ]` 块。

    contraction 内有嵌套的 `[ ]`，所以要数括号深度，不能按行匹配。
    """
    blocks = []
    pos = 0
    while True:
        match = re.search(rf"^  {kind} \[$", text[pos:], re.M)
        if not match:
            return blocks
        start = pos + match.end()
        depth, cursor = 1, start
        while depth > 0:
            opening = text.find("[", cursor)
            closing = text.find("]", cursor)
            if closing == -1:
                break
            if opening != -1 and opening < closing:
                depth += 1
                cursor = opening + 1
            else:
                depth -= 1
                cursor = closing + 1
        blocks.append(text[start:cursor - 1])
        pos = cursor


def strip_contraction(block: str) -> str:
    """去掉 contraction 块，只留主算子自己的字段。"""
    return re.sub(r"contraction \[.*?\n    \]", "", block, flags=re.S)


def field(block: str, name: str, *, top_level: bool = True) -> str | None:
    source = strip_contraction(block) if top_level else block
    match = re.search(
        rf'^\s+{re.escape(name)} (?:"([^"]*)"|(-?\d+))\s*$', source, re.M)
    if not match:
        return None
    return match.group(1) if match.group(1) is not None else match.group(2)


def all_fields(block: str, name: str) -> list[str]:
    return [a or b for a, b in re.findall(
        rf'^\s+{re.escape(name)} (?:"([^"]*)"|(-?\d+))\s*$',
        strip_contraction(block), re.M)]


def check_rule1_fusion(nodes: list[str]) -> list[str]:
    """规则 1：激活与池化必须折进 contraction，不得作为独立节点。"""
    problems = []
    for block in nodes:
        op = field(block, "op_type")
        if op in FUSED_ONLY_OPS:
            problems.append(
                f"node {field(block, 'id')}: {op} 是独立节点，"
                f"应折进生产者的 contraction 块")
    return problems


def consumed_buffers(block: str) -> list[str]:
    """一个节点读取的全部缓冲区名，含多输入算子的每个槽位。

    **不能按 `input_count` 枚举槽位**：MatMul 的第二个 operand 走权重通路，
    它的 `input_count` 故意比实际槽数少 1（实测参考产物 64 个 MatMul 全如此）。
    按它枚举会漏掉最后一个槽，于是那个槽的缓冲区被判成「没人读」的悬空引用。

    改为直接扫到 32 槽为止——GML 的槽号上限是 31（Concat 最多 32 个输入）。
    """
    names = []
    single = field(block, "input_buffer")
    if single:
        names.append(single)
    for slot in range(32):
        name = field(block, f"input_buffer_{slot}")
        if name:
            names.append(name)
    return names


def check_rule2_buffer_naming(nodes: list[str], edges: list[str]) -> list[str]:
    """规则 2：缓冲区名要么按消费者编号，要么按生产者自己编号。

    两种约定并存，实测于两份参考产物：

    - `input_buffer_<消费者id>.bin`：常规节点，缓冲区代表「边」。
      多消费者时 `output_buffer` 只记其中一个，所以按生产者判而不是按边判——
      逐边比对会把残差分支误判成违规。
    - `output_buffer_<自己id>.bin`：llama2 的 `DynamicScaling` 与
      `Llama2ActivationDQ` 这样按生产者命名，缓冲区代表「某节点的输出」。

    还有一类是 `Split`：它的输出指向权重缓冲区（`weight_buffer_*.bin`），
    因为切出来的分片本身就是下游的权重。
    """
    by_id = {field(b, "id"): b for b in nodes}
    problems = []
    for node_id, block in by_id.items():
        produced = field(block, "output_buffer")
        if not produced:
            continue

        # 按生产者自己编号：名字里带的就是本节点 id，无需查消费者。
        if produced == f"output_buffer_{node_id}{_SUFFIX}":
            continue
        # 权重分片：Split 切出的片段直接作为下游的权重缓冲。
        if produced.startswith("weight_buffer"):
            continue

        consumers = [field(e, "target") for e in edges
                     if field(e, "source") == node_id]
        if not consumers:
            continue
        readable = {name for cid in consumers if cid in by_id
                    for name in consumed_buffers(by_id[cid])}
        if produced not in readable:
            problems.append(
                f"node {node_id}: 输出 {produced} 既不按消费者命名，"
                f"也不是 output_buffer_{node_id}{_SUFFIX}"
                f"（消费者 {consumers} 读的是 {sorted(readable)}）")
    return problems


def check_rule3_shape_on_edges(nodes: list[str], edges: list[str]) -> list[str]:
    """规则 3：形状只在边上，节点内不带形状字段。"""
    problems = []
    for block in nodes:
        for key in ("dims", "shape", "input_shape", "output_shape"):
            if field(block, key) is not None:
                problems.append(
                    f"node {field(block, 'id')}: 带形状字段 {key}，"
                    f"形状应只在 edge.dims 上")
    for edge in edges:
        if field(edge, "dims") is None:
            problems.append(
                f"边 {field(edge, 'source')}->{field(edge, 'target')}: 缺 dims")
    return problems


def check_rule4_edge_direction(nodes: list[str], edges: list[str]) -> list[str]:
    """规则 4：边是数据流方向，且每个入口/出口都是缓冲区节点。

    不检查 id 单调性——ResNet50 的 89 条边里 84 条 id 递减，正说明 id 不表达顺序。

    也不要求恰好一个入口一个出口：那只对完整网络成立。llama2 的 decode block
    有 7 个入口（hidden state、KV cache、mask 等）和 3 个出口（output 加两个
    KV cache 写回）。能要求的是每个入口与出口都必须是 `is_buffer` 节点——
    算子节点悬空才是真的错。
    """
    by_id = {field(b, "id"): b for b in nodes}
    sources = {field(e, "source") for e in edges}
    targets = {field(e, "target") for e in edges}
    problems = []

    for node_id in sorted(set(by_id) - targets, key=int):
        if field(by_id[node_id], "is_buffer") != "1":
            problems.append(
                f"node {node_id} 没有入边却不是缓冲区节点"
                f"（op_type={field(by_id[node_id], 'op_type')}）")
    for node_id in sorted(set(by_id) - sources, key=int):
        if field(by_id[node_id], "is_buffer") != "1":
            problems.append(
                f"node {node_id} 没有出边却不是缓冲区节点"
                f"（op_type={field(by_id[node_id], 'op_type')}）")
    return problems


def check_rule5_absent_fields(nodes: list[str]) -> list[str]:
    """规则 5：不表达硬件放置，也不产出由 L2Analyzer 自行推导的字段。"""
    problems = []
    for block in nodes:
        node_id = field(block, "id")
        for key in PLACEMENT_FIELDS:
            if field(block, key) is not None:
                problems.append(
                    f"node {node_id}: 带硬件放置字段 {key}，"
                    f"GML 不表达算子放在哪个 NPU 上")
        for key in NOT_EMITTED_FIELDS:
            if field(block, key) is not None:
                problems.append(
                    f"node {node_id}: 带 {key}，PDF 注明由 L2A 自行推导，不必产出")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gml", type=Path)
    args = parser.parse_args()

    text = args.gml.read_text()
    nodes = parse_blocks(text, "node")
    edges = parse_blocks(text, "edge")
    print(f"{args.gml.name}: {len(nodes)} 节点 / {len(edges)} 边\n")

    checks = [
        ("规则 1  融合是强制的", lambda: check_rule1_fusion(nodes)),
        ("规则 2  buffer 按消费者命名", lambda: check_rule2_buffer_naming(nodes, edges)),
        ("规则 3  形状只在边上", lambda: check_rule3_shape_on_edges(nodes, edges)),
        ("规则 4  边是数据流方向", lambda: check_rule4_edge_direction(nodes, edges)),
        ("规则 5  不产出的字段", lambda: check_rule5_absent_fields(nodes)),
    ]

    failed = 0
    for label, run in checks:
        problems = run()
        if problems:
            failed += 1
            print(f"✗ {label}：{len(problems)} 处")
            for problem in problems[:5]:
                print(f"      {problem}")
            if len(problems) > 5:
                print(f"      ...另有 {len(problems) - 5} 处")
        else:
            print(f"✓ {label}")

    print()
    print(f"{len(checks) - failed}/{len(checks)} 条规则通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
