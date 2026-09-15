"""GML 出口的入口：模型 → 融合 → GML 文本 + 运行时文件清单。

与 NumPy 执行路径是两条独立出口，所以单独一个入口函数而不是塞进
`compile_llama2`：GML 用不到内存蓝图与命令计划，混在一起会让默认路径
承担无关成本。

第 4 轮加量化后，`runtime_files` 里才会有真实的 `.bin`；现在只产出图结构与
文件名清单，用来验证命名规则两侧一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.fx import GraphModule

from contracts import gml_names as names
from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse import fuse_graph
from gml_bridge.from_fx import convert
from gml_bridge.writer import Edge, Node, write_gml

# 参考产物用的版本号。对方的解析器按它判断格式。
GML_VERSION = "26.10.1"


@dataclass
class GmlArtifact:
    """一次 GML 导出的产物。

    `text` 是图文件内容；`buffer_names` 是图里引用到的全部缓冲区文件名——
    第 4 轮写 `.bin` 时按这份清单写，交叉校验就能保证两侧不发散。
    """

    text: str
    nodes: list[Node]
    edges: list[Edge]
    # node_id -> FX 参数名，写盘时按它取 f32 权重去量化。
    weight_params: dict[int, str] = field(default_factory=dict)
    buffer_names: set[str] = field(default_factory=set)
    fusions: int = 0


def _referenced_buffers(nodes: list[Node]) -> set[str]:
    """图里引用到的全部 `.bin` 文件名。"""
    referenced: set[str] = set()
    for node in nodes:
        for value in node.fields.values():
            if isinstance(value, str) and value.endswith(names.SUFFIX):
                referenced.add(value)
    return referenced


def export_graph(gm: GraphModule, *, version: str = GML_VERSION) -> GmlArtifact:
    """把一张已导出的 FX 图转成 GML。

    融合在这里做，不要求调用方先做：GML 没有独立激活节点的表达方式，所以这一步
    不是可选的。
    """
    fusions = fuse_graph(gm)
    nodes, edges, weight_params = convert(gm, version=version)
    text = write_gml(nodes, edges, version=version)
    return GmlArtifact(
        text=text,
        nodes=nodes,
        edges=edges,
        weight_params=weight_params,
        buffer_names=_referenced_buffers(nodes),
        fusions=fusions,
    )


def write_artifact(artifact: GmlArtifact, out_dir: Path) -> Path:
    """把 GML 写到 `out_dir/relay2gml_graph.gml`，返回该路径。

    文件名与参考产物一致——对方的流程按这个名字找图。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "relay2gml_graph.gml"
    path.write_text(artifact.text)
    return path


def export_llama2(
    model: torch.nn.Module,
    *,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
    version: str = GML_VERSION,
) -> GmlArtifact:
    """从 Llama 模型直接产出 GML。

    只走 prefill 那张图：decode 图的结构与它同构，KV 按最大长度固定 + mask，
    所以一张图就够，不必每步重发。
    """
    from runtime.compile import export_annotated_graph

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    gm = export_annotated_graph(model, seq_len, position_ids, dtype=dtype)
    return export_graph(gm, version=version)


def write_runtime_files(
    artifact: GmlArtifact, out_dir: Path, *, gm: GraphModule | None = None
) -> "WrittenFiles":
    """按 GML 引用的清单把 `.bin` 写出来，并交叉校验两侧一致。

    传 `gm` 时把 f32 权重量化成 int4 + per-group scale 一起写出；不传就跳过权重，
    此时若 GML 引用了 `weight_buffer` 会被交叉校验拦下——这是有意的，
    宁可报错也不要产出悬空引用。

    交叉校验在这里做，而不是留给调用方：悬空引用不会在我们这侧报错，
    必须在产出的同一处拦住（见文档第 28 节）。
    """
    from gml_bridge.runtime_files import (
        WrittenFiles,
        verify_against_graph,
        write_activation_scale,
        write_data_buffer,
        write_identity_lut,
        write_weight,
    )
    from quant.weights import quantize_weight

    out_dir.mkdir(parents=True, exist_ok=True)
    files = WrittenFiles(out_dir)

    # 每条边的元素数，用来定数据缓冲区的尺寸。形状只在边上（规则 3）。
    elements_into = {
        edge.target: _element_count(edge.dims) for edge in artifact.edges
    }

    # 逐个节点按它实际引用的名字写，而不是遍历所有可能的名字——后者会写出
    # GML 没引用的垃圾文件，交叉校验会拦下来。
    for node in artifact.nodes:
        for key, value in node.fields.items():
            if not isinstance(value, str) or not value.endswith(names.SUFFIX):
                continue
            slot = _slot_of(key)
            if key == "activation_lut_file":
                write_identity_lut(files, node.node_id)
            elif "_sf" in key:
                write_activation_scale(files, node.node_id, 1.0, slot)
            elif key.startswith("input_buffer"):
                write_data_buffer(
                    files, node.node_id,
                    elements_into.get(node.node_id, 1), slot)
            elif key in ("weight_buffer", "weight_sf"):
                # 两个字段共用一次量化，靠 names_written 去重。
                if names.weight_buffer(node.node_id) in files.names_written:
                    continue
                tensor = _weight_tensor(gm, artifact, node.node_id)
                if tensor is None:
                    continue
                write_weight(files, node.node_id, quantize_weight(tensor))
            elif key == "output_buffer":
                # 输出缓冲区按消费者命名，所以它是下游节点的输入缓冲——
                # 会在那个节点自己的 input_buffer 里写到，这里跳过避免重复。
                continue

    verify_against_graph(files, artifact.buffer_names)
    return files


def _weight_tensor(
    gm: "GraphModule | None", artifact: GmlArtifact, node_id: int
):
    """按 node_id 从 FX 图里取出该节点的 f32 权重。

    参数名走 `artifact.weight_params`，那是 `convert()` 建立的映射。`get_attr`
    的目标是带点的路径，要逐段 `getattr`。
    """
    if gm is None:
        return None
    param = artifact.weight_params.get(node_id)
    if not param:
        return None

    obj = gm
    for part in param.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    # 量化要 f32：权重可能是 fp16，先升精度再分组，避免 fp16 的 max 溢出。
    return obj.detach().float().numpy()


def _element_count(dims: str) -> int:
    """把 GML 的 `1x64x56x56` 形状字符串换算成元素数。"""
    if dims == "unknown":
        return 1
    count = 1
    for part in dims.split("x"):
        count *= int(part)
    return count


def _slot_of(field_name: str) -> int | None:
    """从 `input_1_sf` 这样的字段名里取出槽位号；单输入的返回 None。"""
    parts = field_name.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def format_summary(artifact: GmlArtifact) -> str:
    """人工核对用的摘要。"""
    from collections import Counter

    op_types = Counter(node.fields.get("op_type") for node in artifact.nodes)
    fused = sum(1 for node in artifact.nodes if node.contraction)
    lines = [
        f"节点 {len(artifact.nodes)} 个，边 {len(artifact.edges)} 条",
        f"融合 {artifact.fusions} 处，带 contraction 的节点 {fused} 个",
        f"引用缓冲区 {len(artifact.buffer_names)} 个",
        "算子分布：",
    ]
    for op_type, count in op_types.most_common():
        lines.append(f"  {op_type or '(buffer)'}: {count}")
    return "\n".join(lines)
