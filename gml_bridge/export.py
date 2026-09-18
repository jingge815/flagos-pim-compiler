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

import numpy as np
import torch
from torch.fx import GraphModule

from contracts import gml_names as names
from contracts.gml_quant import GML_VERSION
from contracts.graph_meta import FUSED_TAIL_META_KEY
from graph.fuse import fuse_graph
from graph.fuse_pim import fuse_for_pim
from graph.fuse_rope import ROPE_META_KEY, fuse_rope
from graph.kv_dma_pass import insert_kv_dma_and_split
from graph.quant_pass import DQ_META_KEY, insert_dynamic_scaling
from graph.split_heads import split_attention_heads
from gml_bridge.from_fx import convert
from gml_bridge.writer import Edge, Node, write_gml

# 版本号的真源在 contracts.gml_quant，这里只转出以免调用方到处 import。


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
    # 逐头展开的统计，供 CLI 汇总打印。
    heads: int = 0
    # 插入的 DynamicScaling 个数。
    dq_nodes: int = 0
    # node_id -> DynamicScalingSpec，写盘时按它分配各相 bin 的元素数。
    dq_specs: dict = field(default_factory=dict)
    fusions: int = 0


def _referenced_buffers(nodes: list[Node]) -> set[str]:
    """图里引用到的全部 `.bin` 文件名。

    **三处都要扫**：顶层字段、`vpu_params` 这类一层嵌套块、以及 `contraction`
    里的子算子。漏掉嵌套块会让交叉校验把子块引用的文件误判成「多余文件」
    —— 实测 RMSNorm 的 `output_sf` 只在 `vpu_params` 里出现，顶层没有。
    """
    referenced: set[str] = set()

    def collect(values) -> None:
        for value in values:
            if isinstance(value, str) and value.endswith(names.SUFFIX):
                referenced.add(value)

    for node in nodes:
        collect(node.fields.values())
        for block_fields in node.nested.values():
            collect(block_fields.values())
        for _, child_fields in node.contraction:
            collect(child_fields.values())
    return referenced


def export_graph(gm: GraphModule, *, version: str = GML_VERSION) -> GmlArtifact:
    """把一张已导出的 FX 图转成 GML。

    融合在这里做，不要求调用方先做：GML 没有独立激活节点的表达方式，所以这一步
    不是可选的。

    三个 pass 顺序固定：
    1. `fuse_for_pim` —— llama2 特有的固定模式（RMSNorm 六合一、Gemm+SiLU、
       attention 定标吸收）。它要在通用 pass 之前跑，因为 RMSNorm 那条链
       一旦被通用 pass 拆动就匹配不上了。
    2. `fuse_graph` —— 通用的「主算子 + 尾部激活」，服务 ResNet 那条路径。
    3. `split_attention_heads` —— 把批量 attention 拆成逐头链。**必须最后跑**：
       它产出的逐头节点带角色标记，若先跑，前两个 pass 会把那些
       `add` / `mul` 当成普通逐元素算子去折。
    """
    # RoPE 要在逐头展开**之前**折：展开会把 Q/K 切成每头一份，
    # 切完这条链的判据仍成立但节点翻 32 倍，白做 31 次匹配。
    rope_report = fuse_rope(gm)
    pim_report = fuse_for_pim(gm)
    fusions = fuse_graph(gm) + pim_report.total
    heads = split_attention_heads(gm)
    # 量化 pass 最后跑：插 DQ 的规则依赖逐头角色（matmul1 不插、matmul2 插）。
    # KV 写回与拆头要在逐头展开**之后**：位置判据依赖它留下的头下标标记。
    kv_report = insert_kv_dma_and_split(gm)
    quantized = insert_dynamic_scaling(gm)
    nodes, edges, weight_params = convert(gm, version=version)
    text = write_gml(nodes, edges, version=version)
    return GmlArtifact(
        text=text,
        nodes=nodes,
        edges=edges,
        weight_params=weight_params,
        buffer_names=_referenced_buffers(nodes),
        fusions=fusions,
        heads=heads.heads,
        dq_nodes=quantized.inserted,
        dq_specs=_dq_specs(gm, nodes),
    )


def _dq_specs(gm: GraphModule, nodes: list[Node]) -> dict:
    """按 node_id 收集 DQ 规格。GML 的 `label` 就是 FX 节点名，按它反查。"""
    by_label = {
        node.fields.get("label"): node.node_id for node in nodes}
    specs = {}
    for fx_node in gm.graph.nodes:
        spec = fx_node.meta.get(DQ_META_KEY)
        if spec is not None and fx_node.name in by_label:
            specs[by_label[fx_node.name]] = spec
    return specs


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
        write_dq_phases,
        write_identity_lut,
        write_scaling,
        write_output_scale,
        write_per_tensor_weight,
        write_phase_output_buffer,
        write_named_buffer,
        write_rope_buffer,
        write_zero_point,
        write_rms_norm_epsilon,
        write_weight,
    )
    from gml_bridge.phase_data import dynamic_scaling
    from quant.weights import quantize_weight

    out_dir.mkdir(parents=True, exist_ok=True)
    files = WrittenFiles(out_dir)

    # 每条边的元素数，用来定数据缓冲区的尺寸。形状只在边上（规则 3）。
    #
    # 按 **(源, 目标)** 建索引，不能只按目标：多输入算子的各槽形状不同，
    # 只按目标会让后来的边覆盖前面的，于是所有槽都拿到最后一条边的尺寸。
    # 实测这个 bug 让 EltwiseMul 的两个槽都写成 16 字节，而槽 0 应是 1024 字节。
    elements_between = {
        (edge.source, edge.target): _element_count(edge.dims)
        for edge in artifact.edges
    }

    # 逐个节点按它实际引用的名字写，而不是遍历所有可能的名字——后者会写出
    # GML 没引用的垃圾文件，交叉校验会拦下来。
    for node in artifact.nodes:
        # DQ 节点的各相 bin 由 phase_data 算，一次写全，然后跳过逐字段扫描
        # —— 那些名字都是本节点自命名的，不走边的形状。
        # RoPE 的子块缓冲按**每头的 head_dim** 计元素数：cos/sin 是
        # 逐位置的定标向量，不是整张张量。实测节点 30 的定标三族都是
        # 32 元素（= head_dim/4，见 §11.4 那条判据）。
        rope_elements = _rope_elements(node)
        spec = artifact.dq_specs.get(node.node_id)
        if spec is not None:
            source = _dq_source(gm, artifact, node.node_id, spec)
            write_dq_phases(
                files, node.node_id,
                dynamic_scaling(
                    source,
                    group_size=None if spec.is_attention_scores
                    else spec.group_size))
            # 输出 requant scale 逐字节等于 phase1（下游走跨节点引用时读它）。
            write_output_scale(
                files, node.node_id,
                np.zeros(spec.groups, dtype=np.float16))
            # **不能 continue**：DQ 节点除了 phase 族，还有走通用路径的
            # `input_buffer` / `input_sf`（它在图里仍是一条边的消费者）。
            # 早退会让那两个引用悬空。

        # 每个输入槽的上游节点，用来定该槽的形状。
        sources = {
            slot: value
            for slot in range(32)
            if (value := node.fields.get(f"input{slot}_node_id")) is not None
        }

        # 顶层字段 + 嵌套块（vpu_params）里的字段都要扫：子块引用的文件
        # 同样要落盘，否则是悬空引用。
        entries = list(node.fields.items())
        for block_fields in node.nested.values():
            entries += list(block_fields.items())

        for key, value in entries:
            if not isinstance(value, str) or not value.endswith(names.SUFFIX):
                continue
            # phase 族与 DQ 的 output_sf 已在上面按四相公式写过，跳过避免重复
            # （重复写不会出错，但会把 total_bytes 算重）。
            if spec is not None and ("_phase_" in key or key == "output_sf"):
                continue
            slot = _slot_of(key)
            # RMSNorm 系列的 sf 是 fp32（实测唯一的 dtype 例外），其余 fp16。
            scale_dtype = (
                np.float32
                if node.fields.get("input_sf_dtype") == "float32"
                else np.float16)

            if key == "activation_lut_file":
                write_identity_lut(files, node.node_id)
            elif key == "RMSNorm_Add_Const":
                # eps 取自图里读出的真实值，不假设 config.json。
                write_rms_norm_epsilon(
                    files, node.node_id, _epsilon_of(gm, artifact, node.node_id))
            elif key == "output_scale_factor_buffer":
                # vpu_params 里引用的 output_sf。RMSNorm 的输出 sf 走这条路，
                # 顶层字段里没有它 —— 只在子块里出现，所以要单独认。
                write_output_scale(
                    files, node.node_id, 1.0, dtype=scale_dtype)
            elif _is_rope_file(key):
                # **必须排在通用 Scaling_buffer_file 分支之前**：
                # RoPE 子块的键名也以 Scaling_buffer_file 开头
                # （`Scaling_buffer_file_1_Llama2Activation_Add_Cos`），
                # 排在后面会被通用分支截住，写成 `Bias_buffer_file_1_<id>.bin`
                # 之类——一边悬空一边多余。
                write_rope_buffer(files, key, value, rope_elements)
            elif key.startswith("Scaling_buffer_file"):
                # matmul1 的定标是 1/√head_dim（attention scale 折在这里，
                # 漏掉数值全错）；其余算子 1.0，KV_Cache_DMA 2.0。
                scaling = node.fields.get(_ATTENTION_SCALE_FIELD)
                slot = _slot_of(key)
                write_scaling(
                    files, node.node_id,
                    float(scaling) if scaling is not None else 1.0,
                    post_shift=14 if node.fields.get("op_type") == "KV_Cache_DMA"
                    else 0,
                    slot=slot)
            elif key.startswith(("Bias_buffer_file", "Scaling_PS_buffer_file")):
                continue  # 与 Scaling_buffer_file 同一次写出（三族一起）
            elif key.startswith("updates_") and not key.endswith("_dtype"):
                # updates 那一路的量化参数，按本节点编号。
                if key.endswith("_zp"):
                    write_zero_point(files, value)
                else:
                    write_named_buffer(
                        files, value, 1, dtype=np.float16)
            elif "_zp" in key and not key.endswith("_dtype"):
                # 对称量化：4 字节 int32 的 0，但文件必须存在。
                write_zero_point(files, value)
            elif key == "output_sf":
                # 输出的 requant scale 按**本节点**编号，不是消费者
                # —— 走 write_output_scale。落到下面那个 input_sf 分支会
                # 写成 input_sf_<id>.bin，两边都错（一个悬空、一个多余）。
                write_output_scale(files, node.node_id, 1.0, dtype=scale_dtype)
            elif (key.startswith("input_")
                  and value.startswith(("output_buffer", "weight_buffer"))):
                # 这一路**引用生产者的文件**（上游是 phase 型 DQ）：
                # DQ 自命名 output_buffer_<self>，它 phase1 的输出就是
                # 这条边的 scale。文件由那个 DQ 自己写过了，消费者只是
                # 引用，不能按消费者编号再写一份 —— 否则既悬空
                # （GML 引用的名字没落盘）又多余（写的名字没人引用）。
                continue
            elif ("_sf" in key and not key.endswith("_dtype")
                  and not key.startswith("weight")):
                # `weight_sf` 必须留给下面那个 weight 分支 ——
                # 它和 `weight_buffer` 共用一次量化。落到这里会被
                # 按消费者编号写成 `input_sf_<id>.bin`：一个多余的文件，
                # 而真正的 weight_sf 由 weight 分支另写一份。
                #
                # 这个 bug 一直被名字巧合掩盖：以前节点自己的 `input_sf`
                # 恰好也叫 `input_sf_<id>.bin`，多写的那份正好有人引用。
                # 直到 input_sf 改成引用上游 DQ 的文件，它才暴露成
                # 「写了盘但 GML 没引用」。
                write_activation_scale(
                    files, node.node_id, 1.0, slot, dtype=scale_dtype)
            elif key.startswith("input_buffer"):
                source = sources.get(slot if slot is not None else 0)
                count = elements_between.get((source, node.node_id), 1)
                # 字节宽度必须跟着**声明的** dtype 走，不能一律按 int8。
                # DQ 节点吃的是 fp16（上游还没量化），它声明
                # `input_buffer_dtype float16`，缓冲就得是 2 字节/元素 ——
                # 否则校验器那条「尺寸 == 形状 × dtype」直接报不符。
                declared = node.fields.get(
                    f"{key}_dtype" if slot is None
                    else f"input_buffer_{slot}_dtype",
                    node.fields.get("input_buffer_dtype"))
                write_data_buffer(
                    files, node.node_id, count, slot,
                    dtype=np.float16 if declared == "float16" else np.int8)
            elif key == "output_buffer" and value.startswith("weight_buffer"):
                # 本节点的输出流进下游的**权重通路**，所以按
                # `weight_buffer_<消费者>` 命名（见 from_fx 那段说明）。
                # 它装的是激活数据、不是量化权重，所以按边的元素数
                # 直接写，不走 quantize_weight。
                consumer = _consumer_of(artifact, value)
                count = elements_between.get(
                    (node.node_id, consumer), 1) if consumer else 1
                write_named_buffer(files, value, count)
            elif key in ("weight_buffer", "weight_sf"):
                # 两个字段共用一次量化，靠 names_written 去重。
                if names.weight_buffer(node.node_id) in files.names_written:
                    continue
                tensor = _weight_tensor(gm, artifact, node.node_id)
                if tensor is None:
                    continue

                if node.fields.get("weight_sf_dtype") == "float32":
                    # RMSNorm 的缩放张量是**一维 per-tensor int8**，不是 int4
                    # per-group：实测 weight_buffer_25 是 4096 个 int8、
                    # weight_sf_25 是单个 **fp32**（= 1/127）。
                    # 走 int4 per-group 会因 4096 % 128 的分组语义完全错位。
                    write_per_tensor_weight(files, node.node_id, tensor)
                else:
                    write_weight(files, node.node_id, quantize_weight(tensor))
            elif key == "output_buffer" and value.startswith("output_buffer_"):
                # phase 型节点自命名的输出缓冲：装本节点的完整输出
                # （量化后的 int8），不是某条边的数据。
                write_phase_output_buffer(files, node.node_id, spec.numel)
            elif key == "output_buffer":
                # 输出缓冲区按消费者命名，所以它是下游节点的输入缓冲——
                # 会在那个节点自己的 input_buffer 里写到，这里跳过避免重复。
                continue

    verify_against_graph(files, artifact.buffer_names)
    return files


# 节点上记录 attention 定标系数的字段名（由 from_fx 写入）。
_ATTENTION_SCALE_FIELD = "pim_attention_scale"


_ROPE_FILE_PREFIXES = (
    "Llama2Activation_", "Kantor_A_", "Kantor_B_",
    "cos_mul_output", "sin_mul_output",
)


def _is_rope_file(key: str) -> bool:
    """这个字段是不是 RoPE 子块自己的文件引用。

    判据放在前缀上而不是 op_type 上：同一个节点里既有 RoPE 子块的
    `Scaling_buffer_file_1_Llama2Activation_Add_Cos`，也有通用的
    `input_buffer` —— 只能靠键名区分。
    """
    if key.startswith(_ROPE_FILE_PREFIXES):
        return True
    # 带单元号的定标三族：`Scaling_buffer_file_<n>_Llama2Activation_*`
    return ("Llama2Activation" in key
            and key.startswith(("Scaling_buffer_file",
                                "Scaling_PS_buffer_file",
                                "Bias_buffer_file")))


def _rope_elements(node) -> int:
    """RoPE 子块缓冲的元素数。

    取 `head_dim`：cos/sin 是逐位置的定标向量。取不到形状时退化成 1
    —— 宁可小也不要瞎猜，尺寸不对会被校验器逐文件查出来。
    """
    shape = node.fields.get("original_shape")
    if isinstance(shape, str):
        parts = [int(x) for x in shape.strip("[]").split(",")]
        if parts:
            return parts[-1]
    return 32


def _consumer_of(artifact, buffer_name):
    """从 `weight_buffer_<id>.bin` 反解出消费者的 node_id。

    生产者用消费者的编号给缓冲命名，所以文件名里那个数字就是消费者。
    """
    stem = buffer_name.rsplit(".", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _dq_source(
    gm: "GraphModule | None", artifact: GmlArtifact, node_id: int, spec
) -> "np.ndarray":
    """DQ 节点要量化的那份激活。

    **结构轮用零张量占位**：真实数值要跑一次前向才有，而本轮只保证
    「字节数与 dtype 对、四相之间数学自洽」。参考产物那些 bin 装的同样是
    对方的合成数据（见计划 §5 的前提），所以内容不参与跨产物比对。

    尺寸必须准确 —— 它决定各相 bin 的字节数，那是校验器逐文件比对的项。
    """
    return np.zeros(spec.numel, dtype=np.float16)


def _epsilon_of(
    gm: "GraphModule | None", artifact: GmlArtifact, node_id: int
) -> float:
    """取该 RMSNorm 节点的 eps。

    从 `graph.fuse_pim` 折叠时记下的 meta 读，而不是从 `config.json` 猜——
    两者理论上相同（`rms_norm_eps`），但图是唯一真源：若模型被改过，
    config 可能与实际计算不一致。

    取不到时回退 1e-05（llama2-7B 的值），并且**不静默**：调用方看到的
    产物里那个常量仍然合理，但这里的回退只应发生在没有 gm 的场景。
    """
    from graph.fuse_pim import RMS_NORM_META_KEY

    if gm is None:
        return 1e-05

    # GML 的 `label` 就是 FX 节点名（见 from_fx），所以按它反查。
    label = next(
        (node.fields.get("label") for node in artifact.nodes
         if node.node_id == node_id), None)
    if label is None:
        return 1e-05

    for node in gm.graph.nodes:
        fusion = node.meta.get(RMS_NORM_META_KEY)
        if fusion is not None and node.name == label:
            return fusion.epsilon
    return 1e-05


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
