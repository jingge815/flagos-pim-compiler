"""GML 出口的入口：模型 → 融合 → GML 文本 + 运行时文件清单。

与 NumPy 执行路径是两条独立出口，所以单独一个入口函数而不是塞进
`compile_llama2`：GML 用不到内存蓝图与命令计划，混在一起会让默认路径
承担无关成本。

第 4 轮加量化后，`runtime_files` 里才会有真实的 `.bin`；现在只产出图结构与
文件名清单，用来验证命名规则两侧一致。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.fx import GraphModule

from contracts import gml_names as names
from contracts.compile_slots import DEFAULT_SLOTS, CompileSlots
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
    slots: CompileSlots = field(default_factory=lambda: DEFAULT_SLOTS)
    # 图头里写的格式版本。写盘阶段要按它重出文本（见 `fill_weight_hashes`），
    # 所以不能只留在 `text` 里。
    version: str = GML_VERSION


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


def export_graph(gm: GraphModule, *, version: str = GML_VERSION,
                 phase_source=None) -> GmlArtifact:
    """把一张已导出的 FX 图转成 GML。

    融合在这里做，不要求调用方先做：GML 没有独立激活节点的表达方式，所以这一步
    不是可选的。

    `phase_source` 是 `opcompiler_bridge.phase_source.PhaseSource`：多相算子发
    几套 `*_phase_N` 字段由**算子编译器**决定。没给则退回静态表
    `contracts.gml_quant.PHASE_COUNTS`。两条路径当前产出相同的 GML，因为算子
    编译器算出的相位数与静态表一致——这是 `cross_check()` 保证的，也是验收
    判据（接入前后逐字节相同）。但依赖是真的：pass 改了相位结构，字段套数变。

    **不幂等**：对同一个 `gm` 调两次会得到不同的图（实测第二次 206 节点而非
    200）。要在融合与序列化之间插一步（比如跑算子编译器），用
    `fuse_for_gml` + `serialize_gml` 这对函数，不要调两次本函数。
    """
    report = fuse_for_gml(gm)
    return serialize_gml(gm, report, version=version, phase_source=phase_source)


@dataclass
class FusionReport:
    """`fuse_for_gml` 的产出，喂给 `serialize_gml`。"""

    fusions: int
    heads: int
    dq_nodes: int


def fuse_for_gml(gm: GraphModule) -> FusionReport:
    """原地跑完 GML 需要的六个 pass。**只能对一个 `gm` 调一次。**

    拆出来是为了让调用方能在融合与序列化之间插一步——算子编译器要吃融合后的
    图，而 GML 的相位字段又要等算子编译器的结果，两者必须串在中间。

    pass 顺序固定：
    1. `fuse_rope` —— 要在逐头展开**之前**折：展开会把 Q/K 切成每头一份，
       切完判据仍成立但节点翻 32 倍，白做 31 次匹配。
    2. `fuse_for_pim` —— llama2 特有的固定模式（RMSNorm 六合一、Gemm+SiLU、
       attention 定标吸收）。要在通用 pass 之前跑，因为 RMSNorm 那条链
       一旦被通用 pass 拆动就匹配不上了。
    3. `fuse_graph` —— 通用的「主算子 + 尾部激活」，服务 ResNet 那条路径。
    4. `split_attention_heads` —— 把批量 attention 拆成逐头链。**必须在前三个
       之后**：它产出的逐头节点带角色标记，若先跑，前面的 pass 会把那些
       `add` / `mul` 当成普通逐元素算子去折。
    5. `insert_kv_dma_and_split` / `insert_dynamic_scaling` —— 最后跑：
       插 DQ 的规则依赖逐头角色（matmul1 不插、matmul2 插），KV 写回的位置
       判据依赖拆头留下的头下标标记。
    """
    fuse_rope(gm)
    pim_report = fuse_for_pim(gm)
    fusions = fuse_graph(gm) + pim_report.total
    heads = split_attention_heads(gm)
    insert_kv_dma_and_split(gm)
    quantized = insert_dynamic_scaling(gm)
    return FusionReport(fusions=fusions, heads=heads.heads,
                        dq_nodes=quantized.inserted)


def serialize_gml(gm: GraphModule, report: FusionReport, *,
                  version: str = GML_VERSION,
                  phase_source=None,
                  decode_block_only: bool = False,
                  slots: CompileSlots | None = None) -> GmlArtifact:
    """把已融合的图序列化成 GML。**纯函数**：可重复调用，不改图。"""
    slots = slots or DEFAULT_SLOTS
    nodes, edges, weight_params, extra_dq = convert(
        gm, version=version, phase_source=phase_source,
        decode_block_only=decode_block_only, slots=slots)
    text = write_gml(nodes, edges, version=version)
    return GmlArtifact(
        text=text,
        nodes=nodes,
        edges=edges,
        weight_params=weight_params,
        buffer_names=_referenced_buffers(nodes),
        fusions=report.fusions,
        heads=report.heads,
        dq_nodes=report.dq_nodes,
        dq_specs=_dq_specs(gm, nodes, extra_dq, slots,
                           rewrite=_looks_like_llama7b(gm)),
        slots=slots,
        version=version,
    )


def _looks_like_llama7b(gm: GraphModule) -> bool:
    for node in gm.graph.nodes:
        shape = getattr(node.meta.get("val"), "shape", None)
        if shape and 4096 in tuple(int(x) for x in shape):
            return True
    return False


def _dq_specs(gm: GraphModule, nodes: list[Node], extra: dict | None = None,
              slots: CompileSlots | None = None, *, rewrite: bool = False) -> dict:
    """按 node_id 收集 DQ 规格。

    反查用内部键 `pim_fx_name`（FX 节点名），**不用 `label`**：
    `label` 已改成参考风格的语义名（`self_attn_q_proj_MatMul_qidx..`），
    与 FX 名不再相等，按 label 反查会漏掉全部 DQ，写盘时 spec 为 None。

    `extra` 是 `convert()` 额外认出来的（Q 路 RoPE 那个节点也要走 DQ 写盘，
    但它的规格不在 `node.meta` 里——`convert` 不再写图，改成返回值传出来）。

    numel 按编译期槽位覆盖：导出图 seq_len=16 的形状不能拿去写盘。
    """
    from dataclasses import replace

    from graph.quant_pass import DQ_META_KEY

    slots = slots or DEFAULT_SLOTS
    by_fx_name = {}
    for node in nodes:
        key = node.fields.get("pim_fx_name") or node.fields.get("label")
        by_fx_name[key] = node.node_id
    specs = {}
    for fx_node in gm.graph.nodes:
        spec = fx_node.meta.get(DQ_META_KEY)
        if spec is None or fx_node.name not in by_fx_name:
            continue
        node_id = by_fx_name[fx_node.name]
        if rewrite:
            value = fx_node.meta.get("val")
            shape = getattr(value, "shape", None)
            if not shape:
                raise ValueError(
                    f"DQ 节点 {fx_node.name} 没有 meta['val'].shape，"
                    f"dq_layout 不能拿 0 去猜 hidden")
            last = int(shape[-1])
            numel, group = slots.dq_layout(
                last, is_attention_scores=spec.is_attention_scores,
                group_size=spec.group_size)
            spec = replace(spec, numel=numel, group_size=group)
        specs[node_id] = spec
    if extra:
        specs.update(extra)
    return specs



def _exit_slot_key(exits: list[Node]) -> Callable[[Node], tuple]:
    """出口缓冲的槽位排序键，按**角色**排：hidden → key cache → value cache。

    参考的 `output_idx` 就是这个顺序（0=hidden、1=key_cache_out、
    2=value_cache_out）。按 `node_id` 升序发号会排成 value / key / hidden，
    消费端若按 `output_idx` 绑定缓冲就会把 value cache 当成第一出口
    （评审 r5 问题 1）。

    两个角色都认字段不认 label 子串或编号：hidden 看 `pim_is_hidden_exit`
    （裁剪补块出口时盖的）、KV 看 `pim_kv_is_key`。真实图上「编号倒序」今天
    与角色同解，改天多一个出口就静默换人。入口侧不做这件事：参考的入口槽号
    是 relay 子图签名顺序（`nprm_0_i3` 排槽 3、`nprm_0_i15` 排槽 5），推不出
    规则，按节点号升序发号并记为已知差异。
    """
    hidden = [n for n in exits if n.fields.get("pim_is_hidden_exit")]
    if len(hidden) > 1:
        raise ValueError(
            f"有 {len(hidden)} 个出口缓冲标了 pim_is_hidden_exit，块出口只该有"
            f"一个：{[n.node_id for n in hidden]}")
    hidden_id = hidden[0].node_id if hidden else None

    def role(node: Node) -> tuple[int, int]:
        # 0 = 本图的数据出口（decode 块是 hidden，整网是 logits），1 = KV cache。
        # 整网路径没有 `pim_is_hidden_exit`，它的 logits 出口落在同一个桶里，
        # 两条路都是「数据出口在前、cache 跟后」。
        is_key = node.fields.get("pim_kv_is_key")
        if is_key is not None:
            return (1, 0 if int(is_key) else 1)
        return (0, 0 if node.node_id == hidden_id else 1)

    return lambda n: (*role(n), n.node_id)


def write_io_info(artifact: GmlArtifact, out_dir: Path) -> Path:
    """按参考口径写 `IO_info.txt`：图级入口/出口的 node_id、dtype、shape、size。

    形状取自边的 dims（decode 槽位口径），不是导出图的 seq_len。三处与参考
    对齐的口径：

    - `sf` 等于该缓冲对应节点的 `output_sf`：int8 的 KV cache 取真实量化
      scale，其余留在 fp16 域故为 1.0。IO_info 存 fp32、GML bin 存 fp16。
    - `sf` 写成 numpy 标量再 repr。参考是直接 repr 一个含 numpy 标量的 dict，
      消费端必须 eval 带 numpy 命名空间；写裸 float 会让它读不出预期类型。
    - `shape` 按原始张量的 rank 报（hidden 三维、其余四维），而 GML 的边一律
      补到四维——两处口径本来就不同，rank 由 `pim_io_rank` 从图侧带过来。
    """
    sources = {e.source for e in artifact.edges}
    targets = {e.target for e in artifact.edges}
    by_id = {n.node_id: n for n in artifact.nodes}
    dims_out = {}
    for e in artifact.edges:
        dims_out.setdefault(e.source, e.dims)
        dims_out.setdefault(e.target, e.dims)

    def shape_of(node: Node) -> list[int]:
        dims = dims_out.get(node.node_id)
        if not dims:
            raise ValueError(
                f"缓冲节点 {node.node_id} 没有边，写不出 IO_info 的 shape")
        parts = [int(p) for p in dims.split("x")]
        # 边被左补过维，这里按原始 rank 去掉多补的那几个 1。
        rank = node.fields.get("pim_io_rank")
        if rank is not None and len(parts) > int(rank):
            extra = len(parts) - int(rank)
            if any(p != 1 for p in parts[:extra]):
                raise ValueError(
                    f"缓冲节点 {node.node_id} 的边形状 {dims} 去不掉 {extra} 维："
                    f"被去掉的维不是 1，说明 pim_io_rank 与边不是同一个张量")
            parts = parts[extra:]
        return parts

    def scale_of(node: Node, side: str) -> "np.float32":
        """这个 I/O 缓冲的 sf：等于它对应节点的 `output_sf`。

        **先按 dtype 分派再按角色**（设计 3.5(1)）：只有 int8 缓冲带真实量化
        scale，fp16 / int16 留在 fp16 域故为 1.0。角色取图侧盖的
        `pim_kv_is_key`，不认 label 子串——label 改名会让 sf 静默退回 1.0。
        写成 np.float32 是序列化口径要求。

        `by_id` 必须传真表：`Split` 自己不带 `pim_kv_is_key`，要靠它回溯上游
        DMA 的角色。传空表会让这条回溯恒失效，把"判据没接线"报成"字段缺失"。
        """
        from gml_bridge import calib_data

        if dtype_of(node, side) != "int8":
            return np.float32(1.0)
        is_key = _kv_role_of(node, by_id)
        if is_key is None:
            raise ValueError(
                f"int8 缓冲 {node.node_id}（label "
                f"{node.fields.get('label')!r}）没有 pim_kv_is_key，算不出它的 "
                f"sf。新增 int8 I/O 要显式归类，不能默默按 1.0 发")
        return np.float32(calib_data.kv_cache_scale(is_key=is_key))

    def dtype_of(node, side: str) -> str:
        key = f"{side}_buffer_dtype" if side == "input" else "output_buffer_dtype"
        return str(node.fields.get(key)
                   or node.fields.get("output_buffer_dtype")
                   or node.fields.get("input_buffer_dtype")
                   or "float16")

    entries = []
    exits = []
    for n in artifact.nodes:
        if not n.fields.get("is_buffer"):
            continue
        if n.node_id not in targets:
            entries.append(n)
        if n.node_id not in sources:
            exits.append(n)
    entries.sort(key=lambda n: n.node_id)
    exits.sort(key=_exit_slot_key(exits))

    inputs = {}
    for idx, n in enumerate(entries):
        shape = shape_of(n)
        dt = dtype_of(n, "output")
        inputs[str(n.node_id)] = {
            "sf": scale_of(n, "output"),
            "dtype": dt,
            "node_name": n.fields.get("label", n.node_id),
            "input_idx": idx,
            "shape": shape,
            "size": int(__import__("math").prod(shape)),
        }
        if n.fields.get("is_mask"):
            inputs[str(n.node_id)]["mask"] = True
    outputs = {}
    for idx, n in enumerate(exits):
        shape = shape_of(n)
        dt = dtype_of(n, "input")
        outputs[str(n.node_id)] = {
            "sf": scale_of(n, "input"),
            "dtype": dt,
            "node_name": n.fields.get("label", n.node_id),
            "previous_name": n.fields.get("label", n.node_id),
            "output_idx": idx,
            "shape": shape,
            "size": int(__import__("math").prod(shape)),
        }
    text = repr({"inputs": inputs, "outputs": outputs})
    path = out_dir / "IO_info.txt"
    path.write_text(text)
    return path

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
    artifact: GmlArtifact, out_dir: Path, *, gm: GraphModule | None = None,
    verbose: bool = False,
) -> "WrittenFiles":
    """按 GML 引用的清单把 `.bin` 写出来，并交叉校验两侧一致。

    传 `gm` 时把 f32 权重量化成 int4 + per-group scale 一起写出；不传就跳过权重，
    此时若 GML 引用了 `weight_buffer` 会被交叉校验拦下——这是有意的，
    宁可报错也不要产出悬空引用。

    交叉校验在这里做，而不是留给调用方：悬空引用不会在我们这侧报错，
    必须在产出的同一处拦住（见文档第 28 节）。

    `verbose` 打印每个 DQ 节点的标定中间态（`DynamicScalingPhases.__repr__`），
    满足需求三非功能需求4「标定中间产物要能 print 出可读文本」（评审 r4 问题1）。
    默认关闭：37 个 DQ 节点逐个打印会刷屏，正常导出不需要。
    """
    from gml_bridge.runtime_files import (
        WrittenFiles,
        verify_against_graph,
        write_activation_scale,
        write_data_buffer,
        write_dq_phases,
        write_fused_silu_lut,
        write_identity_lut,
        write_scaling,
        write_output_scale,
        write_per_tensor_weight,
        write_phase_output_buffer,
        write_named_buffer,
        placeholder_weight,
        write_rope_buffer,
        write_softmax_phases,
        write_zero_point,
        write_rms_norm_epsilon,
        write_weight,
    )
    from gml_bridge import calib_data
    from gml_bridge.phase_data import dynamic_scaling, softmax
    from quant.weights import quantize_weight

    out_dir.mkdir(parents=True, exist_ok=True)
    files = WrittenFiles(out_dir)
    slots = getattr(artifact, "slots", None) or DEFAULT_SLOTS
    rewrite = gm is not None and _looks_like_llama7b(gm)

    # 每条边的元素数，用来定数据缓冲区的尺寸。形状只在边上（规则 3）。
    #
    # 按 **(源, 目标)** 建索引，不能只按目标：多输入算子的各槽形状不同，
    # 只按目标会让后来的边覆盖前面的，于是所有槽都拿到最后一条边的尺寸。
    # 实测这个 bug 让 EltwiseMul 的两个槽都写成 16 字节，而槽 0 应是 1024 字节。
    elements_between = {
        (edge.source, edge.target): _element_count(edge.dims)
        for edge in artifact.edges
    }

    # Split 的 K/V 角色要回溯上游的 KV_Cache_DMA，所以先建按 id 的索引。
    by_id = {n.node_id: n for n in artifact.nodes}
    # 定点化的 Kantor scale 要顺着下游找它写进哪块 KV cache，所以也要正向邻接表。
    consumers_of: dict[int, list[int]] = {}
    for edge in artifact.edges:
        consumers_of.setdefault(edge.source, []).append(edge.target)

    # 先把所有 DQ 节点的四相算出来：每个 DQ 的四相要写盘，而同一个节点在
    # 循环里会被多个字段碰到，先算一次避免重复。下游 int8 入槽**引用**它的
    # `output_buffer_phase_1_<DQ>.bin`，不各自再算一份。
    dq_phases = {
        n.node_id: dynamic_scaling(
            _dq_source(gm, artifact, n.node_id, spec),
            group_size=None if spec.is_attention_scores else spec.group_size)
        for n in artifact.nodes
        if (spec := artifact.dq_specs.get(n.node_id)) is not None
    }

    # 逐个节点按它实际引用的名字写，而不是遍历所有可能的名字——后者会写出
    # GML 没引用的垃圾文件，交叉校验会拦下来。
    for node in artifact.nodes:
        # DQ 节点的各相 bin 由 phase_data 算，一次写全，然后跳过逐字段扫描
        # —— 那些名字都是本节点自命名的，不走边的形状。
        spec = artifact.dq_specs.get(node.node_id)
        phases = dq_phases.get(node.node_id)
        if phases is not None:
            if verbose:
                print(f"[DQ] 节点 {node.node_id}: {phases!r}")
            # RoPE 折叠的 DQ 逐组发 FPSU 三族，普通 DQ 发标量（参考实测）。
            write_dq_phases(
                files, node.node_id, phases,
                per_group_fpsu=node.fields.get("op_type")
                == "Llama2ActivationDQ")
            # 输出 requant scale 逐字节等于 phase1（下游走跨节点引用时读它），
            # 所以直接取那一相，不另写一份。
            write_output_scale(files, node.node_id, phases.output_scale)
            # **不能 continue**：DQ 节点除了 phase 族，还有走通用路径的
            # `input_buffer` / `input_sf`（它在图里仍是一条边的消费者）。
            # 早退会让那两个引用悬空。

        # 每个输入槽的上游节点，用来定该槽的形状。
        sources = {
            slot: value
            for slot in range(32)
            if (value := node.fields.get(f"input{slot}_node_id")) is not None
        }

        # RoPE 子块里只有 cos/sin 乘积缓冲按整张激活计元素数（参考 8192 字节
        # = 4096 个 fp16），其余定标零点族都是单元素标量、宽度在
        # `write_rope_buffer` 里按族定，不看这个值。
        rope_elements = elements_between.get(
            (sources.get(0), node.node_id), 1)

        # Softmax 五相：本节点自命名的 phase 族一次写全，同 DQ 的处理——
        # 一直没接（`gml_bridge/runtime_files.py::write_softmax_phases`
        # 早就写好了，之前只是没在这里调用），是 1051 个悬空引用里最大的一块
        # （见 docs/prepare_out-代码评审-20260920.md §3.1）。numel 取自
        # 输入边（bmm1 输出 -> Softmax）。
        #
        # 输入与 DQ 同取标定激活，**不喂零张量**：零输入会让 phase0（落 -max）
        # 变 -0.0、phase1 全 1.0、phase2 = 组长，那是「没算」而不是算出来的，
        # 与 `_dq_source` 同一类问题（设计 3.3 的判定原则）。
        if node.fields.get("op_type") == "Softmax":
            softmax_numel = slots.seq if rewrite else elements_between.get(
                (sources.get(0), node.node_id), 1)
            write_softmax_phases(
                files, node.node_id,
                softmax(calib_data.activation_for(softmax_numel)
                        .astype(np.float32)))
            # **不能 continue**：Softmax 节点的 `input_buffer`/`input_sf`
            # 仍走通用路径（它在图里是一条边的消费者），同 DQ 的道理。

        # 顶层字段 + 嵌套块（vpu_params）里的字段都要扫：子块引用的文件
        # 同样要落盘，否则是悬空引用。
        entries = list(node.fields.items())
        for block_fields in node.nested.values():
            entries += list(block_fields.items())

        for key, value in entries:
            if not isinstance(value, str) or not value.endswith(names.SUFFIX):
                continue
            # phase 族与 DQ 的 output_sf 已在上面按四相公式写过，跳过避免重复
            # （重复写不会出错，但会把 total_bytes 算重）。Softmax 的相位族
            # 同理在上面写过，但它的 `output_sf` 不是 phase 推出来的——GML
            # 节点上是与其它算子一样的通用字段，仍要走下面的通用路径写，
            # 不能跟 DQ 一起跳过（跳过会让这个引用悬空）。
            if spec is not None and ("_phase_" in key or key == "output_sf"):
                continue
            if (node.fields.get("op_type") == "Softmax"
                    and "_phase_" in key):
                continue
            slot = _slot_of(key)
            # RMSNorm 系列的 sf 是 fp32（实测唯一的 dtype 例外），其余 fp16。
            scale_dtype = (
                np.float32
                if node.fields.get("input_sf_dtype") == "float32"
                else np.float16)

            if key == "activation_lut_file":
                write_fused_silu_lut(files, node.node_id)
            elif "kantor" in key.lower() and "_phase_" not in key:
                # Gemm v_proj / mlp_mul 的 Kantor 系数：scale 2B、bias 4B、Shift 1B。
                if value in files.names_written:
                    continue
                if "Shift" in key or "shift" in key:
                    files._write(
                        value,
                        np.full(1, _kantor_shift_of(key, node), dtype=np.int8))
                elif "bias" in key.lower():
                    files._write(value, np.zeros(1, dtype=np.float32))
                else:
                    files._write(
                        value,
                        np.full(1, _kantor_scale_of(key, node, by_id,
                                                    consumers_of),
                                dtype=np.float16))
            elif key == "RMSNorm_Add_Const":
                # eps 取自图里读出的真实值，不假设 config.json。
                write_rms_norm_epsilon(
                    files, node.node_id, _epsilon_of(gm, artifact, node.node_id))
            elif key == "output_scale_factor_buffer":
                # vpu_params 里的 output_sf 与顶层 output_sf 字段指向同一个
                # 文件名（都是 output_sf_<id>.bin），取值走同一处计算，
                # 不能各写一份——写死 1.0 会在 _output_scale_of 将来对
                # RMSNorm 给出别的值时静默盖掉计算结果（评审 r4 问题3）。
                write_output_scale(
                    files, node.node_id, _output_scale_of(node, by_id),
                    dtype=scale_dtype)
            elif _is_rope_file(key):
                # **必须排在通用 Scaling_buffer_file 分支之前**：
                # RoPE 子块的键名也以 Scaling_buffer_file 开头
                # （`Scaling_buffer_file_1_Llama2Activation_Add_Cos`），
                # 排在后面会被通用分支截住，写成 `Bias_buffer_file_1_<id>.bin`
                # 之类——一边悬空一边多余。
                write_rope_buffer(files, key, value, rope_elements)
            elif key.startswith("Scaling_buffer_file"):
                # matmul1 的定标是 1/√head_dim（attention scale 折在这里，
                # 漏掉数值全错）；其余算子 1.0。
                #
                # KV_Cache_DMA 是 (scale 2.0, post_shift 14) 这一对固定值：
                # 参考两个 DMA 节点都是这一对，且都没有声明
                # `weight_sf_multiplier`，所以推不出来，是 TVM 侧对这个算子
                # 写死的常量。post_shift 一直按这条发 14，scale 却落到了通用
                # 分支算成 1.0，与同一处注释自称的 2.0 打架（评审 r7 问题 5）。
                scaling = node.fields.get(_ATTENTION_SCALE_FIELD)
                slot = _slot_of(key)
                is_kv_dma = node.fields.get("op_type") == "KV_Cache_DMA"
                if scaling is not None:
                    scale_val = float(scaling)
                elif is_kv_dma:
                    scale_val = 2.0
                else:
                    multiplier = int(node.fields.get("weight_sf_multiplier") or 1)
                    scale_val = 1.0 / multiplier
                write_scaling(
                    files, node.node_id, scale_val,
                    post_shift=14 if is_kv_dma else 0,
                    slot=slot)
            elif key.startswith(("Bias_buffer_file", "Scaling_PS_buffer_file")):
                continue  # 与 Scaling_buffer_file 同一次写出（三族一起）
            elif key.startswith("updates_") and not key.endswith("_dtype"):
                # updates 那一路的量化参数，按本节点编号。sf 与本节点的
                # output_sf 同一块 cache、同一个 scale（P0-2：quantize/
                # dequantize 共用同一常量），不能各写一份 0。
                if key.endswith("_zp"):
                    write_zero_point(files, value)
                else:
                    write_named_buffer(
                        files, value, 1, dtype=np.float16,
                        content=np.full(
                            1, _output_scale_of(node, by_id), dtype=np.float16))
            elif "_zp" in key and not key.endswith("_dtype"):
                # 对称量化：4 字节 int32 的 0，但文件必须存在。
                write_zero_point(files, value)
            elif key == "output_sf":
                # 输出的 requant scale 按**本节点**编号，不是消费者
                # —— 走 write_output_scale。落到下面那个 input_sf 分支会
                # 写成 input_sf_<id>.bin，两边都错（一个悬空、一个多余）。
                write_output_scale(
                    files, node.node_id, _output_scale_of(node, by_id),
                    dtype=scale_dtype)
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
                #
                # KV 那一路的 `input_sf` 取那块 cache 的 scale，不写死 1.0：
                # 它读的是 int8 的 cache 平面，1.0 等于宣称「没量化」。参考在
                # 同一批节点上三个 sf 字段是同一个常量（节点 28 的
                # input/updates/output_sf 全为 0.0459），我方以前只改了后两个
                # （评审 r2 问题 1）。判据按 K/V 角色，不按 dtype —— 其余 int8
                # 输入的 sf 是**引用上游 DQ 的 phase1 文件**（上面那条
                # `input_` 分支已 continue），不走这里。
                write_activation_scale(
                    files, node.node_id,
                    _input_scale_of(node, by_id, slot),
                    slot, dtype=scale_dtype)
            elif key.startswith("input_buffer"):
                source = sources.get(slot if slot is not None else 0)
                count = elements_between.get((source, node.node_id), 1)
                op = node.fields.get("op_type")
                # KV 三槽按编译期槽位，不按导出图边宽。
                if op == "KV_Cache_DMA" and rewrite:
                    if slot == 0:
                        count = slots.kv_cache_elems
                    elif slot == 1:
                        count = slots.kv_index_elems
                    elif slot == 2:
                        count = slots.kv_new_elems
                elif op == "Mask" and slot == 1 and rewrite:
                    count = slots.seq
                declared = node.fields.get(
                    f"{key}_dtype" if slot is None
                    else f"input_buffer_{slot}_dtype",
                    node.fields.get("input_buffer_dtype"))
                dtype = np.int8
                if declared == "float16":
                    dtype = np.float16
                elif declared == "int16":
                    dtype = np.int16
                if value == names.data_buffer(node.node_id, slot):
                    # 这条边的上游是图入口缓冲时，内容取那个入口自己的标定
                    # 常数（cos / sin / mask / kv_position 各一份），不是
                    # hidden state 平铺（评审 r7 问题 1）。
                    producer = by_id.get(source)
                    write_data_buffer(
                        files, node.node_id, count, slot, dtype=dtype,
                        calib_role=producer.fields.get("pim_calib_role")
                        if producer else None)
                # 否则是跨节点引用（本节点复用另一个节点的 input_buffer，
                # 如 RoPE 的 Q/K 两路共享同一份 cos/sin 乘积）：那个名字属于
                # 它真正的生产者，会在处理那个节点时按上面的分支写出。这里
                # 不再兜底补零——补零会先写一份错误内容，等生产者后写时才
                # 覆盖，同一个文件名在一次导出里被写两种内容，谁生效取决于
                # 节点遍历顺序（评审 r4 问题4）。真的没有生产者时，
                # `verify_against_graph` 会以"GML 引用了但没写盘"报出来，
                # 比静默产零更早暴露问题。
            elif key == "output_buffer" and value.startswith("weight_buffer"):
                # 本节点的输出流进下游的**权重通路**，所以按
                # `weight_buffer_<消费者>` 命名（见 from_fx 那段说明）。
                # llama2-7B 的 MatMul 权重是 KV cache 平面，按编译期 S×hd。
                consumer = _consumer_of(artifact, value)
                count = elements_between.get(
                    (node.node_id, consumer), 1) if consumer else 1
                if rewrite:
                    count = slots.bmm_weight_elems
                # 也是 KV cache 平面走权重通路，内容同样不能是全零——见
                # `placeholder_weight` 的说明。
                write_named_buffer(files, value, count,
                                   content=placeholder_weight(node.node_id,
                                                              count))
            elif key in ("weight_buffer", "weight_sf"):
                # Gemm/RMSNorm：两个字段共用一次量化，靠 names_written 去重。
                # MatMul：`weight_buffer` 往往已经被上游（Split / KV_Cache_DMA）
                # 按权重通路写过，但 `weight_sf` 仍要另写——跳过整支会让
                # 64 个 weight_sf 悬空（评审 3 §2.1）。
                already = names.weight_buffer(node.node_id) in files.names_written
                tensor = _weight_tensor(gm, artifact, node.node_id)
                if tensor is None:
                    if (node.fields.get("op_type") == "MatMul"
                            and node.fields.get("MatMul_input_as_weight")):
                        if not already:
                            count = _matmul_weight_elements(
                                node, elements_between, slots, rewrite=rewrite)
                            write_named_buffer(
                                files, names.weight_buffer(node.node_id),
                                count, dtype=np.int8,
                                content=placeholder_weight(node.node_id, count))
                        if names.weight_scale(node.node_id) not in files.names_written:
                            files._write(
                                names.weight_scale(node.node_id),
                                np.array([1.0], dtype=np.float16))
                    continue
                if already:
                    continue

                if node.fields.get("weight_sf_dtype") == "float32":
                    # RMSNorm 的缩放张量是**一维 per-tensor int8**，不是 int4
                    # per-group：实测 weight_buffer_25 是 4096 个 int8、
                    # weight_sf_25 是单个 **fp32**（= 1/127）。
                    # 走 int4 per-group 会因 4096 % 128 的分组语义完全错位。
                    write_per_tensor_weight(files, node.node_id, tensor)
                else:
                    quantized = quantize_weight(tensor)
                    multiplier = int(node.fields.get("weight_sf_multiplier") or 1)
                    if multiplier != 1:
                        # 幂等：只乘一次。scale 已带倍数则不再乘。
                        scales = quantized.scales.astype(np.float32) * multiplier
                        quantized = type(quantized)(
                            quantized.values, scales.astype(quantized.scales.dtype),
                            quantized.group_size)
                    write_weight(files, node.node_id, quantized)
            elif value == names.phase_output_buffer_self(node.node_id):
                # phase 型节点自命名的输出缓冲：装本节点的完整输出
                # （量化后的 int8），不是某条边的数据。参考产物上它与
                # `output_buffer_phase_3_<id>.bin` 逐字节相同，所以直接取那一相。
                # 走到这里的必是 phase 型节点，spec 一定在（`output_buffer_<self>`
                # 这个名字只有它们自己发）。断言让违反时立刻炸，不靠默认值兜。
                #
                # 判据是「名字里的尾号是不是自己」，不能只看前缀：布局算子把
                # 上游 DQ 的定点数据传下去时，它的 output_buffer 也叫
                # `output_buffer_<那个 DQ>`，那是**引用**、由 DQ 自己写盘
                # （评审 r8 问题 1）。
                assert spec is not None, (
                    f"节点 {node.node_id} 自命名 output_buffer 但没有 dq_spec")
                write_phase_output_buffer(
                    files, node.node_id, spec.numel, phases.phase3)
            elif key == "output_buffer":
                # 输出缓冲区按消费者命名，所以它是下游节点的输入缓冲——
                # 会在那个节点自己的 input_buffer 里写到，这里跳过避免重复。
                continue

    verify_against_graph(files, artifact.buffer_names)
    write_io_info(artifact, out_dir)
    return files


def fill_weight_hashes(artifact: GmlArtifact, files: "WrittenFiles") -> None:
    """把权值 bin 的内容指纹回填到节点，并按新字段重出 GML 文本。

    参考产物对每个带 `weight_buffer` 的节点都写 `weight_buffer_hash`，用于
    跨节点权值去重，是按节点必填的。

    指纹只能在 `WrittenFiles._write` 里算——那是字节最终成形的地方。序列化
    阶段那些缓冲还不存在：64 个注意力节点的权值是 KV 激活，写盘时才按形状
    补零，长度由 `_matmul_weight_elements` 定，序列化侧复制不出来。所以顺序
    是**先写盘、再重出文本**，而不是序列化时算或写盘后扫盘。

    权威在图编译器这一侧。`#pim.weight_binding` 上的 `contentHash` 留空是
    **约定**：算子编译器不持有权值张量，算不出这份指纹。两边不要各算一次，
    对不上时以这里落盘的字节为准。

    这一步**不放在 `write_runtime_files` 里**：那两条不变量检查（接入前后
    逐字节相同、砍相位必变）比的是 `artifact.text`，而指纹与算子编译器无关，
    掺进去会让它们比的不是同一件事。调用方在检查跑完之后调它。
    """
    changed = False
    for node in artifact.nodes:
        name = node.fields.get("weight_buffer")
        if not isinstance(name, str):
            continue
        digest = files.hashes.get(name)
        if digest is None:
            continue
        node.fields["weight_buffer_hash"] = digest
        changed = True
    if changed:
        artifact.text = write_gml(artifact.nodes, artifact.edges,
                                  version=artifact.version)


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


def _matmul_weight_elements(node, elements_between: dict,
                            slots: CompileSlots | None = None,
                            *, rewrite: bool = False) -> int:
    """MatMul 权重通路的元素数：S × hd（参考 1024×128 = 131072）。

    llama2-7B 一律用编译期槽位；小图仍按边宽。
    """
    slots = slots or DEFAULT_SLOTS
    if rewrite:
        return slots.bmm_weight_elems
    source = node.fields.get("input1_node_id")
    if source is not None:
        count = elements_between.get((int(source), node.node_id))
        if count:
            return count
    return slots.bmm_weight_elems


# 做定点化（fp16 → int8）的 Kantor 模式。其余模式（`elementwise_mul_fp16`、
# `off`）不改变量化域，scale 是纯逐元素系数。
_KANTOR_FIXED_POINT_MODE = "fp2int_converter"


def _kantor_mode_key(key: str) -> str:
    """这个 Kantor 系数文件（scale 或 Shift）对应的 `kantor_mode` 字段名。

    同一个节点上可以挂好几族 Kantor 系数，每族自带一个 mode 字段，scale 与
    Shift 共用同一个 mode——「做不做定点化」是同一件事的两半（1/scale 与
    ×256），判据不能分叉（评审 r4 问题5）：

    | 系数文件键 | mode 字段 |
    | --- | --- |
    | `kantor_A_scale_buffer_file` / `kantor_A_Shift` | `kantor_mode` |
    | `Kantor_A_Llama2Activation_add_scale_buffer_file` / `_Shift_Llama2Activation_add` | `kantor_mode_Llama2Activation_add` |
    | `Kantor_B_Llama2Activation_Cos_scale_buffer_file` / `_Shift_Llama2Activation_Cos` | `kantor_mode_Llama2Activation_Cos` |
    """
    suffix = key
    for prefix in ("kantor_A_", "kantor_B_", "Kantor_A_", "Kantor_B_"):
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix):]
            break
    for marker in ("scale_buffer_file", "Shift_", "Shift"):
        suffix = suffix.replace(marker, "")
    suffix = suffix.strip("_")
    return f"kantor_mode_{suffix}" if suffix else "kantor_mode"


def _kantor_scale_of(key: str, node: Node, by_id: dict[int, Node],
                     consumers_of: dict[int, list[int]]) -> float:
    """非 phase 的 Kantor scale 取值：**按族**看这一族的 `kantor_mode`。

    | `kantor_mode` | 取值 | 含义 |
    | --- | --- | --- |
    | `fp2int_converter` | 1 / 下游那块 KV cache 的 scale | 这一族做定点化 |
    | 其余（`elementwise_mul_fp16` / `off`） | 1.0 | 不改变量化域，不缩放 |

    定点化族的方向是**量化 scale 的倒数**，与 DQ 的 `phase2 = 1/phase0` 同向
    （参考实测：v_proj 的 383.25 ≈ 1/0.00261、RoPE 那条 add 的 21.78
    ≈ 1/0.0459，各是它那次量化用的 cache scale 的倒数）。

    那次量化的 scale 分两种来路，都归到 `_output_scale_of` 这一处取值源：

    - 本节点输出自己就落 int8（v_proj、`KV_Cache_DMA`）：取本节点的 `output_sf`。
    - 本节点输出留在 fp16 域（RoPE 的 add 折在 `Llama2Activation` 里）：量化发生
      在下游那块 cache 上，顺着下游找到那个 DMA 取它的角色。

    不缩放族发 1.0 而不是 absmax 倒数：同仓 `write_rope_buffer` 对同一个 RoPE
    角色写的就是 1.0，写别的值等于把「不缩放」变成一次缩放（评审 r2 问题 2）。
    """
    from gml_bridge import calib_data

    mode_key = _kantor_mode_key(key)
    mode = node.fields.get(mode_key)
    if mode is None:
        raise ValueError(
            f"节点 {node.node_id} 有 Kantor scale {key!r}，但没有配套的 "
            f"{mode_key!r}，判不出这一族做不做定点化")
    if str(mode) != _KANTOR_FIXED_POINT_MODE:
        return 1.0
    own = _output_scale_of(node, by_id)
    if own != 1.0:
        return 1.0 / own
    is_key = _kv_cache_written_by(node, by_id, consumers_of)
    return 1.0 / calib_data.kv_cache_scale(is_key=is_key)


def _kv_cache_written_by(node: Node, by_id: dict[int, Node],
                         consumers_of: dict[int, list[int]]) -> bool:
    """顺着下游找这个节点的输出写进哪块 KV cache；返回 K 还是 V。

    给「本节点输出还在 fp16 域、定点化发生在下游」的那一族用（RoPE 的 add）。
    走到 DMA 就停，不穿过去。找不到或找到多块都抛：这一族的 scale 只有那块
    cache 能定，猜一个值会把量化域算错而不报错。
    """
    seen = {node.node_id}
    queue = [node.node_id]
    roles: set[bool] = set()
    while queue:
        current = queue.pop(0)
        for consumer in consumers_of.get(current, ()):
            if consumer in seen:
                continue
            seen.add(consumer)
            downstream = by_id.get(consumer)
            if downstream is None:
                continue
            if downstream.fields.get("op_type") == "KV_Cache_DMA":
                roles.add(bool(downstream.fields.get("pim_kv_is_key")))
                continue
            queue.append(consumer)
    if len(roles) != 1:
        raise ValueError(
            f"节点 {node.node_id} 的定点化 Kantor scale 要取下游 cache 的 "
            f"scale，但顺着下游找到 {len(roles)} 块 KV cache（应当恰好 1 块）")
    return roles.pop()


def _kantor_shift_of(key: str, node: Node) -> int:
    """非 phase 的 Kantor Shift 取值：做定点化的发 -8（左移 8 位 = ×256），否则 0。

    判据改为**按族看 `kantor_mode`**，与 `_kantor_scale_of` 共用同一个
    `_kantor_mode_key`（评审 r4 问题5）：此前按文件名前缀各判一次，
    一族改名或新增一个定点化族时 scale 会跟着 `kantor_mode` 走、Shift 不会，
    两者本是同一件事的两半（×256 与 1/scale），错的那一份在产物上仍是
    合法 int8，拦不住。走 `_phase_` 那条路的 37 个由
    `runtime_files.write_dq_phases` 写，不经这里。
    """
    from gml_bridge.phase_data import DQ_PHASE3_SHIFT

    mode_key = _kantor_mode_key(key)
    mode = node.fields.get(mode_key)
    if mode is None:
        raise ValueError(
            f"节点 {node.node_id} 有 Kantor Shift {key!r}，但没有配套的 "
            f"{mode_key!r}，判不出这一族做不做定点化")
    return DQ_PHASE3_SHIFT if str(mode) == _KANTOR_FIXED_POINT_MODE else 0


def _output_scale_of(node: Node, by_id: dict[int, Node]) -> float:
    """这个节点输出的 requant scale。

    输出落进 int8 的 KV cache 的那几个带真实 scale，其余留在 fp16 域故为 1.0：

    | op_type | 取值 |
    | --- | --- |
    | `KV_Cache_DMA` | 本节点写的是 K 还是 V cache |
    | `Split` | 从 cache 切出，继承上游 DMA 的同一个 scale |
    | `Gemm` 的 v_proj | 输出要写进 value cache，带 requant scale |
    | 其余 | 1.0 |

    进出同一块 cache 共用同一个 scale，所以三处都从 `calib_data.kv_cache_scale`
    取，不各算一份。

    **不回落 1.0**：该发 1.0 的算子显式列在 `_FP16_DOMAIN_OP_TYPES` 里，白名单
    外直接抛。回落会让「新增或改名的算子没被识别」与「本来就该是 1.0」走同一条
    路，把 bug 静默掉（设计 3.4）。
    """
    from gml_bridge import calib_data

    op_type = node.fields.get("op_type")
    is_key = _kv_role_of(node, by_id)
    if is_key is not None:
        return calib_data.kv_cache_scale(is_key=is_key)
    if op_type == "Gemm" and "v_proj" in str(
            node.fields.get("pim_weight_param") or ""):
        # v_proj 的输出写进 int8 的 value cache，与那块 cache 同一个 scale。
        return calib_data.kv_cache_scale(is_key=False)
    if op_type in _FP16_DOMAIN_OP_TYPES:
        return 1.0
    raise ValueError(
        f"节点 {node.node_id} 的 op_type {op_type!r} 不在 output_sf 取值表里，"
        f"算不出它的 requant scale。新增算子要显式归类：落 int8 cache 的取"
        f"`calib_data.kv_cache_scale`，留在 fp16 域的加进 _FP16_DOMAIN_OP_TYPES")


# 输出留在 fp16 域、`output_sf` 恒为 1.0 的算子（需求 2.4「非标定类 sf」实测：
# MatMul×64、Softmax×32、Mask×32、EltwiseAdd×2、EltwiseMul×1 两边都是 2B/1.0）。
#
# 只列**实测带 `output_sf`** 的算子。纯布局的 `Concat` / `Reshape` / `Transpose`
# 不带这个字段（两条导出路径实测 0 个），曾误列在此：多列不是兜底——将来谁给
# `Concat` 加上 `output_sf`，它会静默取 1.0，正是设计 3.4 要防的静默路径。
# `KV_Cache_DMA` 也不在这里：它由前面的 KV 角色分支取真实 scale。
_FP16_DOMAIN_OP_TYPES = frozenset({
    "MatMul", "Softmax", "Mask", "EltwiseAdd", "EltwiseMul",
    "RMSNorm_vpu", "Gemm", "Split", "Llama2Activation", "Llama2ActivationDQ",
    "DynamicScaling",
})


# 定标沿边传播时可以穿过的算子：它们不改数值，只改布局。
_LAYOUT_OP_TYPES = frozenset({"Split", "Transpose", "Reshape", "Concat"})


def _input_scale_of(node: Node, by_id: dict[int, Node],
                    slot: int | None) -> float:
    """一条**按本节点自命名**的输入 `input_sf` 该写什么值。

    定点数据的逐组 scale 属于产生它的那个 DQ，由那个 DQ 自己写盘、消费者只
    引用（`output_buffer_phase_1_<DQ>.bin`，走 `input_` 分支 continue），所以
    走到这里的只剩两类自命名来源：

    1. KV cache 那一路（`pim_kv_is_key`）取那块 cache 的 scale；
    2. **fp16 域的边发 1.0** —— 这条边没做定点化，1.0 是真值而不是兜底。

    **不回落 1.0**：该发 1.0 的来源显式列出，识别不到就抛（设计 3.4，与
    `_output_scale_of` 同口径）。回落会让「新增一个 int8 入口但没接上定标」
    与「这条边本来就没量化」走同一条路，把 bug 静默掉。

    dtype **没声明**按 fp16 算：`_NO_TOP_IN_DTYPE` 那几个算子（Mask /
    EltwiseAdd / EltwiseMul / Concat / KV_Cache_DMA）本来就不写顶层 input
    dtype，缺省即 fp16，这是 `from_fx` 的既有约定而不是识别失败。真正要抛的是
    「声明了一个不认识的 dtype」与「int8 却自命名 sf」两种。
    """
    kv_role = _kv_role_of(node, by_id)
    if kv_role is not None:
        from gml_bridge import calib_data
        return calib_data.kv_cache_scale(is_key=kv_role)

    declared = node.fields.get(
        "input_buffer_dtype" if slot is None
        else f"input_buffer_{slot}_dtype",
        node.fields.get("input_buffer_dtype"))
    if declared is None or declared in ("float16", "float32"):
        return 1.0

    if declared == "int8":
        # 走到这里说明这条 int8 边的 `input_sf` 是**按本节点自命名**的。定点数据
        # 的逐组 scale 属于产生它的那个 DQ，组数按 DQ 的整张张量分，与这条边
        # （以及它标注的那个按边宽定尺寸的缓冲）的元素数根本不是一个口径：照抄
        # 上游的整条向量，就等于宣称 group_size = 边宽 / 组数，与 DQ 实际的分组
        # 相矛盾（评审 r8 问题 1，实测那 32 个 MatMul 宣称成了 4）。
        #
        # 正确的形态是缓冲与定标**出自同一个节点**：`from_fx` 的
        # `_quant_origin_of` 会穿过布局算子把两个名字都指到那个 DQ，于是
        # `input_sf` 落到上面那条 `input_` 分支 continue，不到这里。
        producer = _quant_producer_of(node, by_id)
        raise ValueError(
            f"节点 {node.node_id} 槽 {slot} 是 int8 输入，但 `input_sf` 按本节点"
            f"自命名"
            + (f"，而给它定标的是 DQ {producer}：应当引用那个 DQ 的 "
               f"`output_buffer_phase_1_{producer}.bin`，缓冲也引用它的 "
               f"`output_buffer_{producer}.bin`，让组数与元素数出自同一个节点"
               if producer is not None else
               "，且顺上游找不到给它定标的 DQ 节点——要么把上游的 DQ 接上，"
               "要么把这条边改成 fp16"))

    raise ValueError(
        f"节点 {node.node_id} 槽 {slot} 声明的 input dtype 是 {declared!r}，"
        f"不在取值表里，算不出这条边的 scale。新增 dtype 要显式归类："
        f"留在 fp16 域的发 1.0，落定点域的接上游 DQ")


def _quant_producer_of(node: Node, by_id: dict[int, Node]) -> int | None:
    """顺 slot0 上游穿过布局算子，找产生这份定点数据的 DQ 节点号。

    布局算子不改数值，定标跟着**原始生产者**走。只认 slot0：布局算子的
    数据都从第一个输入来。
    """
    seen: set[int] = set()
    current = node
    while current is not None and current.node_id not in seen:
        seen.add(current.node_id)
        upstream = by_id.get(current.fields.get("input0_node_id"))
        if upstream is None:
            return None
        op_type = upstream.fields.get("op_type")
        if op_type in ("DynamicScaling", "Llama2ActivationDQ"):
            return upstream.node_id
        if op_type not in _LAYOUT_OP_TYPES:
            return None
        current = upstream
    return None


def _kv_role_of(node: Node, by_id: dict[int, Node]) -> bool | None:
    """这个节点是不是 KV cache 那一路；是则返回它写/读的是 K 还是 V。

    `KV_Cache_DMA` 与两个 cache 出口缓冲自己带 `pim_kv_is_key`；`Split` 不带，
    要看它的上游是不是 DMA —— Q 路的 Split 上游是 RoPE，不属于这一路。
    """
    if "pim_kv_is_key" in node.fields:
        return bool(node.fields.get("pim_kv_is_key"))
    if node.fields.get("op_type") != "Split":
        return None
    source = by_id.get(node.fields.get("input0_node_id"))
    if source is None or source.fields.get("op_type") != "KV_Cache_DMA":
        return None
    return bool(source.fields.get("pim_kv_is_key"))


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
    """DQ 节点要量化的那份激活，取自 `calib_data` 的标定常数。

    **不能用零张量**：零 absmax 会让 phase0 与 phase1 一路输出 0.0，而
    sf=0 是形式非法值（下游反量化除零）。标定值嵌在源码里，所以这条路径
    不读任何外部目录。

    尺寸必须准确 —— 它决定各相 bin 的字节数，那是校验器逐文件比对的项。
    """
    from gml_bridge import calib_data

    return calib_data.activation_for(spec.numel)


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

    # 按内部键 `pim_fx_name` 反查 FX 名：`label` 已改成参考风格的语义名。
    label = next(
        ((node.fields.get("pim_fx_name") or node.fields.get("label"))
         for node in artifact.nodes if node.node_id == node_id), None)
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
    """把 GML 的 `1x64x56x56` 形状字符串换算成元素数。

    非数字形状直接抛。原来 `"unknown"` 返回 1，于是那条边的缓冲按 1 个元素落盘，
    而声明也是 `unknown` —— 校验器比的正是「声明 × dtype 宽度」对「文件字节数」，
    两边一起错就永远绿。形状算不出来是上游的错，在这里补一个数只会盖住它。
    """
    count = 1
    for part in dims.split("x"):
        if not part.isdigit():
            raise ValueError(
                f"边的 dims {dims!r} 不是纯数字形状，算不出元素数。"
                f"缓冲尺寸由它决定，猜一个会让声明与落盘一起错")
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
