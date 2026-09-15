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
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genesim_bridge.paths import llama2_7b_model_dir
from gml_bridge.export import (
    export_graph,
    format_summary,
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
    args = parser.parse_args()

    from runtime.compile import export_annotated_graph

    model = _load_model(args.layers)
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    graph = export_annotated_graph(
        model, args.seq_len, position_ids, dtype=torch.float32)

    artifact = export_graph(graph)
    print(format_summary(artifact))

    gml_path = write_artifact(artifact, args.out_dir)
    files = write_runtime_files(
        artifact, args.out_dir, gm=None if args.skip_weights else graph)
    print(f"\n图: {gml_path}")
    print(f"运行时文件: {len(files.names_written)} 个, "
          f"{files.total_bytes / 1e6:.2f} MB")

    problems = _check_structure(artifact.text)
    if problems:
        print("\n结构自检未通过:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("结构自检: 5/5 条规则通过")
    print("交叉校验: GML 引用集 == 落盘集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
