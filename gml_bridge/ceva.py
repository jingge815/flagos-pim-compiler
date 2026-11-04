"""CEVA 链路：按厂商 runner 契约导出我方 GML 与全部 bin。

厂商子图只作两种用途：

- 契约：runner 按名字绑定的 I/O（`IO_info.txt`）与节点名，导出后逐节点核对结构；
- 浮点 checkpoint：厂商权重经旋转与 norm 折叠，我方复现不了这一步，所以把它反量化
  成浮点装进模型（`load_vendor_checkpoint`），之后由我方量化器重新量化。

GML、相位、字段与全部 bin（权重、scale、LUT、定标、常量表）都由我方编译器生成，
不复制厂商任何 bin。
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import torch

from contracts.ceva import CevaGraphSpec, CevaIoBinding
from contracts.compile_slots import CompileSlots
from genesim_bridge.paths import ceva_embedding, flagtree_source

RUNTIME_DIR = Path("parser_output") / "tvmgen_default_nprm_main_0" / "runtime_files"
GML_NAME = "relay2gml_graph.gml"

_KEY_SUFFIX = re.compile(r"(_qidx\d+)?_params_\d+$")
# 调试字段：厂商不落盘或不影响 Registry（GC 探针已核实）。
_DEBUG_FIELD = re.compile(r"(^|\.)(DEBUG_|lut_debug$)")


def graph_dir(reference_root: Path, stage: str, layer_id: int | None) -> Path:
    """厂商 65 图根目录下某张子图的目录。"""
    if stage == "lm_head":
        return Path(reference_root) / "llama_lm_head"
    sub = {"prefill": "prompt", "decode": "decode"}[stage]
    return Path(reference_root) / sub / f"model_layers_{layer_id}_{sub}_v2"


def operator_key_of(label: str) -> str:
    """厂商节点名去掉 `_qidx<N>_params_<M>` 后缀。"""
    return _KEY_SUFFIX.sub("", label)


@dataclass
class VendorGraph:
    """解析后的厂商子图。`nodes` 以「语义键#序号」为键，序号按 node_id 升序。"""

    version: str
    nodes: dict[str, dict] = field(default_factory=dict)
    by_id: dict[int, str] = field(default_factory=dict)
    edges: list[tuple[int, int, str]] = field(default_factory=list)


def _parse_gml(text: str):
    """GML 文本 → (图属性, 节点列表, 边列表)。重复键收成列表，嵌套块收成字典。"""
    tokens = re.findall(r'"[^"]*"|\[|\]|[^\s\[\]"]+', text)
    pos = 0

    def value(tok):
        if tok.startswith('"'):
            return tok[1:-1]
        try:
            return int(tok)
        except ValueError:
            return float(tok)

    def block():
        nonlocal pos
        items: list[tuple[str, object]] = []
        while pos < len(tokens):
            key = tokens[pos]
            if key == "]":
                pos += 1
                return items
            pos += 1
            if tokens[pos] == "[":
                pos += 1
                items.append((key, block()))
            else:
                items.append((key, value(tokens[pos])))
                pos += 1
        return items

    top = block()
    if len(top) != 1 or top[0][0] != "graph":
        raise ValueError("不是一份 GML：顶层应为 graph [ ... ]")
    attrs, nodes, edges = {}, [], []
    for key, val in top[0][1]:
        if key == "node":
            nodes.append(val)
        elif key == "edge":
            edges.append(dict(val))
        else:
            attrs[key] = val
    return attrs, nodes, edges


def _fold(items) -> dict:
    """键值对列表 → 字典；重复键成列表，子块递归。"""
    out: dict = {}
    for key, val in items:
        if isinstance(val, list):
            val = _fold(val)
        if key in out:
            if not isinstance(out[key], list):
                out[key] = [out[key]]
            out[key].append(val)
        else:
            out[key] = val
    return out


def parse_vendor_graph(runtime_dir: Path) -> VendorGraph:
    """读 `relay2gml_graph.gml`，按语义键索引节点。"""
    attrs, raw_nodes, raw_edges = _parse_gml(
        (Path(runtime_dir) / GML_NAME).read_text(errors="replace"))
    graph = VendorGraph(version=str(attrs.get("relay2gml_version", "")))
    counts: dict[str, int] = {}
    parsed = []
    for items in raw_nodes:
        folded = _fold(items)
        parsed.append(folded)
    for node in sorted(parsed, key=lambda n: int(n["id"])):
        stem = operator_key_of(str(node["label"]))
        key = f"{stem}#{counts.get(stem, 0)}"
        counts[stem] = counts.get(stem, 0) + 1
        nested = {k: v for k, v in node.items() if isinstance(v, dict)}
        top = {k: v for k, v in node.items() if not isinstance(v, dict)}
        graph.nodes[key] = {"node_id": int(node["id"]), "label": str(node["label"]),
                            "fields": top, "nested": nested}
        graph.by_id[int(node["id"])] = key
    graph.edges = [(int(e["source"]), int(e["target"]), str(e.get("dims", "")))
                   for e in raw_edges]
    return graph


def vendor_float_weights(runtime_dir: Path, shapes: dict[str, tuple[int, int]]) -> dict:
    """厂商投影的量化权重还原成 HF 布局（输出维 × 输入维）的 f32，供我方量化器重新量化。

    `shapes` 是 语义键 -> (输出维, 输入维)。int4 码值按「输入维 × 输出维」存、scale 按
    「输出维 × 输入维/128」存且乘过 `weight_sf_multiplier`；int8 输出头是 per-tensor。
    """
    import numpy as np

    graph = parse_vendor_graph(runtime_dir)
    weights = {}
    for key, (out_f, in_f) in shapes.items():
        fields = graph.nodes[key]["fields"]
        codes = np.fromfile(Path(runtime_dir) / fields["weight_buffer"], dtype=np.int8)
        scales = np.fromfile(Path(runtime_dir) / fields["weight_sf"], dtype=np.float16)
        if codes.size != out_f * in_f:
            raise ValueError(f"{key} 权重尺寸 {codes.size} 与期望 {out_f}x{in_f} 不符")
        codes = codes.reshape(in_f, out_f).T.astype(np.float32)
        scales = scales.astype(np.float32) / int(fields.get("weight_sf_multiplier", 1))
        if fields.get("weight_buffer_dtype") == "int8":
            weights[key] = codes * scales[0]
        else:
            grouped = codes.reshape(out_f, in_f // 128, 128)
            weights[key] = (grouped * scales.reshape(out_f, -1, 1)).reshape(out_f, in_f)
    return weights


_PROJECTIONS = (("self_attn", "q"), ("self_attn", "k"), ("self_attn", "v"),
                ("self_attn", "o"), ("mlp", "gate"), ("mlp", "up"), ("mlp", "down"))


def load_vendor_checkpoint(model, reference_root: Path, *, include_head: bool = True) -> None:
    """把厂商 65 图里的量化权重反量化后装进 HF 模型，作为我方编译的浮点 checkpoint。

    厂商权重经 Hadamard 旋转、RMSNorm 的 γ 已折进下游投影，所以 norm 权重置 1；
    embedding 同样已旋转，由 runner 的输入工具提供，这里不动。之后所有 bin 由我方
    量化器与写盘器重新生成，不再复制厂商字节。
    """
    import torch

    for index, layer in enumerate(model.model.layers):
        runtime_dir = graph_dir(reference_root, "decode", index) / RUNTIME_DIR
        shapes, targets = {}, {}
        for group, role in _PROJECTIONS:
            linear = getattr(getattr(layer, group), f"{role}_proj")
            key = f"{group}_{role}_proj_MatMul#0"
            shapes[key] = tuple(linear.weight.shape)
            targets[key] = linear
        for key, weight in vendor_float_weights(runtime_dir, shapes).items():
            targets[key].weight.data = torch.from_numpy(weight)
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
    if include_head:
        runtime_dir = graph_dir(reference_root, "lm_head", None) / RUNTIME_DIR
        key = "lm_head_MatMul#0"
        weight = vendor_float_weights(runtime_dir, {key: tuple(model.lm_head.weight.shape)})[key]
        model.lm_head.weight.data = torch.from_numpy(weight)
        model.model.norm.weight.data.fill_(1.0)


def _file_fields(fields: dict, prefix: str = ""):
    """(字段路径, 文件名)：顶层与嵌套块里引用的全部 `.bin`。"""
    for key, val in fields.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            yield from _file_fields(val, path + ".")
        elif isinstance(val, str) and val.endswith(".bin"):
            yield path, val


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_io_info(text: str) -> dict:
    """`IO_info.txt` 是含 numpy 标量 repr 的 Python 字面量，换成纯数字后按字面量解析。"""
    plain = re.sub(r"array\(([^,()]+), dtype=\w+\)", r"\1", text)
    plain = re.sub(r"np\.float32\(([^()]+)\)", r"\1", plain)
    return ast.literal_eval(plain)


# 运行时缓冲：中间激活、相位中间态、RoPE 乘积，内容由生产者或 runner 在运行时写入。
_RUNTIME_FIELD = re.compile(r"^((input|output)_buffer(_\d+|_phase_\d+)?|(cos|sin)_mul_output)$")


def runtime_buffer_names(nodes) -> set[str]:
    """GML 引用的运行时缓冲文件名（设计 4.7）。

    按字段判定：数据缓冲、MatMul 权重通路（读 KV cache）、DQ 的逐组动态 scale、
    Softmax 的运行时归约落点是运行时数据；prefill RoPE 的 cos/sin 槽是编译期常量表
    （Registry 会嵌入），不算。同名文件只要被一个静态字段引用就不算。
    """
    runtime, static = set(), set()
    for node in nodes:
        const_slots = ({"input_buffer_1", "input_buffer_2"}
                       if node.fields.get("pim_rope_const") else set())
        as_weight = node.fields.get("MatMul_input_as_weight")
        op_type = node.fields.get("op_type")
        for path, name in _node_files(node):
            dynamic = (_RUNTIME_FIELD.match(path) and path not in const_slots) or (
                as_weight and path == "weight_buffer")
            if op_type in ("DynamicScaling", "Llama2ActivationDQ"):
                dynamic = dynamic or path == "output_sf" or (
                    path.startswith("kantor_") and "_scale_buffer_file_phase_" in path)
            if op_type == "Softmax":
                dynamic = dynamic or path in ("Bias_buffer_phase_1",
                                              "Scaling_buffer_phase_4")
            (runtime if dynamic else static).add(name)
    return runtime - static


def write_generated_files(artifact, gm, out_dir: Path, scales) -> list[dict]:
    """全部 bin 由我方写盘器生成（权重量化、相位、LUT、定标），返回清单条目。

    `artifact` 的节点已按厂商编号；引用集与落盘集由写盘器自己交叉校验。运行时缓冲只按
    尺寸分配、写零：写盘器里的标定样本只用来推相位常量，不代表实际计算。
    """
    from gml_bridge.export import write_runtime_files

    write_runtime_files(artifact, Path(out_dir), gm=gm, scales=scales)
    for name in runtime_buffer_names(artifact.nodes):
        path = Path(out_dir) / name
        path.write_bytes(bytes(path.stat().st_size))
    manifest = []
    for name in sorted(artifact.buffer_names):
        path = Path(out_dir) / name
        manifest.append({"path": name, "size_bytes": path.stat().st_size,
                         "sha256": _sha256(path), "role": "generated", "source": None})
    return manifest


def generated_io(io, nodes, scales) -> list[CevaIoBinding]:
    """I/O 契约换上我方标定的 sf：int8 的 KV 入口/出口取对应 cache 的 scale，其余不变。

    名字、槽号、形状、dtype 是 runner 的接口约定，沿用厂商；sf 是编译结果，取我方。
    """
    roles = {n.fields.get("label"): n.fields.get("pim_kv_is_key")
             for n in nodes if n.fields.get("is_buffer")}
    fixed = []
    for binding in io:
        if binding.dtype == "int8":
            role = roles.get(binding.label if binding.direction == "output" else binding.name)
            if role is None:
                raise ValueError(f"int8 I/O {binding.name} 认不出 K/V 角色，定不出 sf")
            binding = dataclasses.replace(binding, sf=scales.kv(is_key=bool(role)))
        fixed.append(binding)
    return fixed


def load_ceva_spec(reference_root: Path, stage: str, layer_id: int | None,
                   slots: CompileSlots) -> CevaGraphSpec:
    """读厂商同名子图，生成 I/O 契约与静态文件绑定。"""
    if stage not in ("prefill", "decode", "lm_head"):
        raise ValueError(f"未知阶段 {stage!r}")
    runtime_dir = graph_dir(reference_root, stage, layer_id) / RUNTIME_DIR
    if not (runtime_dir / GML_NAME).is_file():
        raise FileNotFoundError(f"厂商子图不存在：{runtime_dir / GML_NAME}")
    graph = parse_vendor_graph(runtime_dir)

    io_info = _parse_io_info((runtime_dir / "IO_info.txt").read_text())
    io: list[CevaIoBinding] = []
    for direction, table in (("input", io_info["inputs"]),
                             ("output", io_info["outputs"])):
        for entry in table.values():
            io.append(CevaIoBinding(
                direction=direction,
                index=int(entry["input_idx" if direction == "input" else "output_idx"]),
                name=str(entry["node_name"]),
                label=entry.get("previous_name"),
                shape=tuple(int(x) for x in entry["shape"]),
                dtype=str(entry["dtype"]),
                sf=float(entry["sf"])))
    io.sort(key=lambda b: (b.direction != "input", b.index))

    query_tokens = 1 if stage != "prefill" else slots.seq
    slots = dataclasses.replace(slots, tokens=query_tokens)
    return CevaGraphSpec(stage=stage, layer_id=layer_id, query_tokens=query_tokens,
                         slots=slots, version=graph.version, io=tuple(io),
                         runtime_dir=runtime_dir)


# ---- 三类包装器 ----
#
# 包装真实的 HF 层，张量签名与厂商 runner 的绑定一致。cos/sin 用厂商编码：
# sin 的后半维已取反（`InputBinCreator/main.py`），所以先乘固定符号向量还原
# 成 HF 编码，再走标准 rotate_half；`graph.fuse_rope` 把这个乘法吸收进 RoPE。



def causal_mask(rows: int, cols: int, dtype=torch.float32) -> torch.Tensor:
    """[1,1,rows,cols] 加性因果掩码：第 i 行可见前 i+1 列。"""
    mask = torch.full((rows, cols), torch.finfo(dtype).min, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.reshape(1, 1, rows, cols)


def _sign_half(head_dim: int) -> torch.Tensor:
    sign = torch.ones(head_dim)
    sign[head_dim // 2:] = -1
    return sign


def _rope(x: torch.Tensor, cos: torch.Tensor, sin_hf: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin_hf


class _LayerBody(torch.nn.Module):
    """解码层的公共部分：注意力之外的 norm、投影、残差与 MLP。"""

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layer = layer
        attn = layer.self_attn
        self.head_dim = int(attn.head_dim)
        self.register_buffer("sign_half", _sign_half(self.head_dim), persistent=False)

    def project(self, hidden, cos, sin):
        """返回 RoPE 后的 Q/K（[1,T,nh,hd]）与 V（[1,nh,T,hd]）。"""
        attn = self.layer.self_attn
        T, H = hidden.shape[1], hidden.shape[2]
        nh = H // self.head_dim
        x = self.layer.input_layernorm(hidden)
        sin_hf = sin * self.sign_half
        q = _rope(attn.q_proj(x).view(1, T, nh, self.head_dim), cos, sin_hf)
        k = _rope(attn.k_proj(x).view(1, T, nh, self.head_dim), cos, sin_hf)
        v = attn.v_proj(x).view(1, T, nh, self.head_dim).transpose(1, 2)
        return q, k, v

    def finish(self, hidden, q, key, value, mask):
        attn = self.layer.self_attn
        T, H = hidden.shape[1], hidden.shape[2]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, key, value, attn_mask=mask, scale=self.head_dim ** -0.5)
        out = attn.o_proj(out.transpose(1, 2).reshape(1, T, H))
        h = hidden + out
        return h + self.layer.mlp(self.layer.post_attention_layernorm(h))


class PrefillLayer(_LayerBody):
    """prefill：`(hidden, mask) -> (hidden_out, key, value)`，cos/sin 是编译期常量。"""

    def __init__(self, layer, cos: torch.Tensor, sin: torch.Tensor) -> None:
        super().__init__(layer)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, hidden, mask):
        q, k, v = self.project(hidden, self.cos, self.sin)
        key = k.transpose(1, 2)
        return self.finish(hidden, q.transpose(1, 2), key, v, mask), key, v


class DecodeLayer(_LayerBody):
    """decode：`(hidden, cos, sin, key, position, mask, value) -> (hidden_out, key_out, value_out)`。

    K/V cache 原地更新：只改写 `position[2,0,0,0]` 那一行。
    """

    def forward(self, hidden, cos, sin, key, position, mask, value):
        q, k, v = self.project(hidden, cos, sin)
        pos = position[2, 0, 0, 0].to(torch.long).reshape(1)
        key_out = torch.index_copy(key, 2, pos, k.transpose(1, 2))
        value_out = torch.index_copy(value, 2, pos, v)
        return (self.finish(hidden, q.transpose(1, 2), key_out, value_out, mask),
                key_out, value_out)


class HeadLayer(torch.nn.Module):
    """lm_head：`hidden -> lm_head(norm(hidden))`。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.norm = model.model.norm
        self.lm_head = model.lm_head

    def forward(self, hidden):
        return self.lm_head(self.norm(hidden))


# ---- 编号与逐节点对齐 ----
#
# 厂商编号规则（decode/prompt/head 三类 65 图实测一致）：图入口缓冲按 IO 槽号
# 先编，其余从出口缓冲（按 IO 槽号）出发，沿 input0..N 槽深度优先、先序编号。
# 编号进 Registry（GC 探针实测置换编号会改变 Registry），所以按同一规则编。

_IN_SLOT = re.compile(r"input(\d+)_node_id$")
_OUT_SLOT = re.compile(r"output(\d+)_node_id$")


def _input_slots(fields: dict) -> list[int]:
    slots = sorted((int(m.group(1)), int(v)) for k, v in fields.items()
                   if (m := _IN_SLOT.fullmatch(k)))
    return [v for _, v in slots]


def renumber_vendor_order(nodes, edges, *, sources, sinks, weight_params=None,
                          dq_specs=None):
    """按厂商规则重编号。`sources`/`sinks` 是入口与出口缓冲的旧 id，已按 IO 槽号排好。

    给了 `weight_params`/`dq_specs`（按旧 id 索引）时一并重映射，返回四元组。
    """
    from gml_bridge.from_fx import _renumber_from_one

    by_id = {n.node_id: n for n in nodes}
    order = list(sources)
    seen = set(order)

    def visit(node_id: int) -> None:
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            stack.extend(reversed(_input_slots(by_id[current].fields)))

    for sink in sinks:
        visit(sink)
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"从出口沿输入槽走不到这些节点：{missing}")
    remap = {old: new for new, old in enumerate(order, start=1)}
    nodes, edges, new_params, new_specs = _renumber_from_one(
        nodes, edges, weight_params or {}, dq_specs or {}, remap=remap)

    # 输出端口按新编号升序排（厂商 output0..N 与 residual_output_buffer 皆如此）。
    for node in nodes:
        outs = sorted((int(m.group(1)), v) for k, v in node.fields.items()
                      if (m := _OUT_SLOT.fullmatch(k)))
        if len(outs) < 2:
            continue
        targets = sorted(v for _, v in outs)
        for (slot, _), target in zip(outs, targets):
            node.fields[f"output{slot}_node_id"] = target
        if isinstance(node.fields.get("residual_output_buffer"), list):
            node.fields["residual_output_buffer"] = targets[:10]
            if "residual_output_buffer_" in node.fields:
                node.fields["residual_output_buffer_"] = targets[10:]
        # 走权重通路的输出按编号最大的消费者命名（厂商 Split 实测如此）。
        if str(node.fields.get("output_buffer", "")).startswith("weight_buffer_"):
            node.fields["output_buffer"] = f"weight_buffer_{targets[-1]}.bin"
    nodes.sort(key=lambda n: n.node_id)
    if weight_params is None and dq_specs is None:
        return nodes, edges
    return nodes, edges, new_params, new_specs


def align_to_vendor(nodes, edges, vendor: VendorGraph) -> dict[int, str]:
    """两张图按同一规则编号后逐 id 核对：op_type 与入边（源、形状）都要一致。

    返回 `我方 node_id -> 厂商语义键`，导入参数按它取厂商字节。
    """
    ours_in = {}
    for e in edges:
        ours_in.setdefault(e.target, set()).add((e.source, e.dims))
    theirs_in = {}
    for src, dst, dims in vendor.edges:
        theirs_in.setdefault(dst, set()).add((src, dims))
    if len(nodes) != len(vendor.by_id):
        raise ValueError(f"节点数不一致：我方 {len(nodes)}，厂商 {len(vendor.by_id)}")
    keys: dict[int, str] = {}
    for node in nodes:
        key = vendor.by_id.get(node.node_id)
        if key is None:
            raise ValueError(f"厂商图没有节点 {node.node_id}")
        theirs = vendor.nodes[key]["fields"]
        ours_op = node.fields.get("op_type")
        if ours_op != theirs.get("op_type"):
            raise ValueError(f"节点 {node.node_id}：op_type 我方 {ours_op}，厂商 {theirs.get('op_type')}（{key}）")
        if ours_in.get(node.node_id, set()) != theirs_in.get(node.node_id, set()):
            raise ValueError(
                f"节点 {node.node_id}（{key}）入边或形状不一致：我方 "
                f"{sorted(ours_in.get(node.node_id, set()))}，厂商 {sorted(theirs_in.get(node.node_id, set()))}")
        keys[node.node_id] = key
    return keys


# ---- 逐图导出 ----

# runner 契约（设计 2.3、4.6）：边界缓冲的角色 → 厂商 I/O 名。
BOUNDARY_NAMES = {
    "decode": {"hidden": "nprm_0_i0", "cos": "nprm_0_i9", "sin": "nprm_0_i10",
               "key_in": "nprm_0_i3", "kv_position": "nprm_0_i4",
               "mask": "nprm_0_i15", "value_in": "nprm_0_i1"},
    "prefill": {"hidden": "nprm_0_i0", "mask": "nprm_0_i164"},
    "lm_head": {"hidden": "nprm_0_i0"},
}
OUTPUT_NAMES = {"hidden_out": "output", "key_out": "key_cache_out",
                "value_out": "value_cache_out"}
# 只为调试或溯源，不进 Registry（GC 探针实测增删不改 Registry 字节）。
_IGNORED_FIELD = re.compile(r"^(DEBUG_|lut_debug$|from_tvm$|.*_hash$|id$|node_id$)")


def _boundary_role(node, is_sink: bool) -> str:
    fields = node.fields
    if is_sink:
        kv = fields.get("pim_kv_is_key")
        if kv is None:
            return "hidden_out"
        return "key_out" if kv else "value_out"
    if fields.get("pim_is_hidden_entry"):
        return "hidden"
    role = fields.get("pim_calib_role")
    if role in ("cos", "sin", "mask", "kv_position"):
        return role
    kv = fields.get("pim_kv_is_key")
    if kv is not None:
        return "key_in" if kv else "value_in"
    raise ValueError(f"边界缓冲 {node.node_id}（{fields.get('label')}）认不出角色")


def bind_boundaries(nodes, edges, spec: CevaGraphSpec):
    """边界缓冲改用厂商 I/O 名，返回按 IO 槽号排好的 (入口 id, 出口 id)。"""
    targets = {e.target for e in edges}
    sources, sinks = [], []
    names = BOUNDARY_NAMES[spec.stage]
    for node in nodes:
        if not node.fields.get("is_buffer"):
            continue
        is_sink = node.node_id in targets
        role = _boundary_role(node, is_sink)
        name = OUTPUT_NAMES[role] if is_sink else names.get(role)
        if name is None:
            raise ValueError(f"{spec.stage} 的契约里没有角色 {role}")
        binding = spec.io_for(name)
        node.fields["label"] = node.fields["name"] = (
            binding.label if is_sink else binding.name)
        if is_sink:
            node.fields["original_name"] = binding.name
        (sinks if is_sink else sources).append((binding.index, node.node_id))
    expected = {b.name for b in spec.io}
    got = {spec.io_for(n.fields["label"]).name for n in nodes if n.fields.get("is_buffer")}
    if got != expected:
        raise ValueError(f"I/O 与契约不一致：缺 {sorted(expected - got)}，多 {sorted(got - expected)}")
    return [i for _, i in sorted(sources)], [i for _, i in sorted(sinks)]


def _comparable(fields: dict) -> dict:
    return {k: v for k, v in fields.items()
            if not k.startswith("pim_") and not _IGNORED_FIELD.match(k)}


def check_fields(nodes, keys: dict[int, str], vendor: VendorGraph) -> list[str]:
    """逐节点比对字段集合与取值（含嵌套块与 contraction），返回差异描述。"""
    problems = []
    for node in nodes:
        entry = vendor.nodes[keys[node.node_id]]
        ours = _comparable(node.fields)
        ours.update({name: _comparable(block) for name, block in node.nested.items()})
        if node.contraction:
            ours["contraction"] = {name: _comparable(f) for name, f in node.contraction}
        theirs = _comparable(entry["fields"])
        theirs.update({k: (_comparable(v) if isinstance(v, dict) else v)
                       for k, v in entry["nested"].items()})
        if "contraction" in theirs:
            theirs["contraction"] = {k: _comparable(v) for k, v in theirs["contraction"].items()}
        for side in (ours, theirs):
            _merge_residual(side)
        for key in sorted(set(ours) | set(theirs)):
            a, b = ours.get(key), theirs.get(key)
            # 文件名不进 Registry（GC 探针实测全量改名后字节不变），只比有无。
            if _is_file(a) or _is_file(b):
                if (a is None) != (b is None):
                    problems.append(f"{node.node_id} {keys[node.node_id]} {key}: 我方 {a!r} 厂商 {b!r}")
                continue
            if _normal(a) != _normal(b):
                problems.append(f"{node.node_id} {keys[node.node_id]} {key}: 我方 {a!r} 厂商 {b!r}")
    return problems


def _is_file(value) -> bool:
    return isinstance(value, str) and value.endswith(".bin")


def _merge_residual(fields: dict) -> None:
    """端口号 >= 10 的 `residual_*_buffer_` 与前 10 个合成一个列表。"""
    for direction in ("input", "output"):
        key = f"residual_{direction}_buffer"
        tail = fields.pop(key + "_", None)
        if tail is None:
            continue
        head = fields.get(key, [])
        head = head if isinstance(head, list) else [head]
        fields[key] = head + (tail if isinstance(tail, list) else [tail])


def _normal(value):
    # 重复键只出现一次时 GML 解析成标量，与单元素列表等价。
    if isinstance(value, list) and len(value) == 1:
        return _normal(value[0])
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ---- 写盘：IO_info ----


def _node_files(node):
    """(字段路径, 文件名)：我方节点顶层、嵌套块与 contraction 里引用的 `.bin`。"""
    merged = dict(node.fields)
    merged.update(node.nested)
    for name, fields in node.contraction:
        merged.setdefault("contraction", {})[name] = fields
    for path, name in _file_fields(merged):
        if not _DEBUG_FIELD.search(path):
            yield path, name


def graph_dir_of(spec: CevaGraphSpec) -> Path:
    return spec.runtime_dir


def write_ceva_io_info(nodes, spec: CevaGraphSpec, out_dir: Path) -> Path:
    """按厂商口径写 `IO_info.txt`：键为我方 node_id，内容取自契约。"""
    by_label = {n.fields.get("label"): n.node_id for n in nodes if n.fields.get("is_buffer")}

    def sf_repr(value: float, dtype: str) -> str:
        if dtype == "int16":
            return f"np.float32({value!r})"
        return f"array({value!r}, dtype=float32)".replace("array(1.0,", "array(1.,")

    parts = {"inputs": [], "outputs": []}
    for binding in spec.io:
        node_id = by_label[binding.label if binding.direction == "output" else binding.name]
        size = 1
        for extent in binding.shape:
            size *= extent
        body = (f"'sf': {sf_repr(binding.sf, binding.dtype)}, 'dtype': '{binding.dtype}', "
                f"'node_name': '{binding.name}', ")
        if binding.direction == "input":
            body += f"'input_idx': {binding.index}, "
        else:
            body += f"'previous_name': '{binding.label}', 'output_idx': {binding.index}, "
        body += f"'shape': {list(binding.shape)}, 'size': {size}"
        if binding.direction == "input" and binding.name in ("nprm_0_i15", "nprm_0_i164"):
            body += ", 'mask': True"
        parts[binding.direction + "s"].append(f"'{node_id}': {{{body}}}")
    text = ("{'inputs': {" + ", ".join(parts["inputs"]) + "}, 'outputs': {"
            + ", ".join(parts["outputs"]) + "}}")
    path = Path(out_dir) / "IO_info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _stage_module(model, spec: CevaGraphSpec):
    """按阶段包好模型并给出导出用的样例输入（形状取自契约）。"""
    slots = spec.slots
    nh, hd, S = slots.heads, slots.head_dim, slots.seq
    if spec.stage == "lm_head":
        return HeadLayer(model), (torch.zeros(1, 1, slots.hidden),)
    layer = model.model.layers[spec.layer_id]
    if spec.stage == "decode":
        args = (torch.zeros(1, 1, slots.hidden), torch.zeros(1, 1, 1, hd),
                torch.zeros(1, 1, 1, hd), torch.zeros(1, nh, S, hd),
                torch.zeros(3, 1, nh, 1, dtype=torch.int16),
                torch.zeros(1, 1, 1, S), torch.zeros(1, nh, S, hd))
        return DecodeLayer(layer), args
    T = spec.query_tokens
    cos, sin = _prefill_rope(model, T)
    return PrefillLayer(layer, cos, sin), (torch.zeros(1, T, slots.hidden),
                                           torch.zeros(1, 1, T, T))


def _prefill_rope(model, tokens: int):
    """prefill 的 cos/sin 常量：位置 0..T-1 的 rotary，sin 后半维取反（硬件编码），[1,T,1,hd]。"""
    x = torch.zeros(1, tokens, model.config.hidden_size)
    cos, sin = model.model.rotary_emb(x, torch.arange(tokens).unsqueeze(0))
    sin = sin.clone()
    sin[..., sin.shape[-1] // 2:] *= -1
    return cos.unsqueeze(2), sin.unsqueeze(2)


def export_ceva_subgraph(model, spec: CevaGraphSpec):
    """三类包装器 → 已融合的 FX 图。"""
    from gml_bridge.export import fuse_for_gml
    from graph.partition import partition_graph

    module, args = _stage_module(model, spec)
    gm = torch.export.export(module, args, strict=True).module()
    partition_graph(gm)
    if spec.stage == "lm_head":
        mark_static_quant(gm)
    report = fuse_for_gml(gm)
    return gm, report


def mark_static_quant(gm) -> None:
    """直接产出图输出的投影走 int8 静态量化（CEVA 输出头的口径，厂商 head GML 实测）。

    按结构选：结果就是图输出的那个 linear。共享 pass 只认这个语义标记。
    """
    from contracts.unified_ir import STATIC_QUANT_META_KEY

    output = next(n for n in gm.graph.nodes if n.op == "output")
    heads = [n for n in output.all_input_nodes
             if n.target is torch.ops.aten.linear.default]
    if len(heads) != 1:
        raise ValueError(f"图输出应恰好来自 1 个投影，实际 {len(heads)}")
    heads[0].meta[STATIC_QUANT_META_KEY] = True


def export_ceva_graph(model, spec: CevaGraphSpec, out_dir: Path, *,
                      vendor: VendorGraph | None = None,
                      use_opcompiler: bool = True, scales=None,
                      calibration_inputs=None) -> dict:
    """导出一张子图到 `out_dir`（含 GML、IO_info 与 bin），返回清单条目。

    GML、字段与全部 bin 由我方生成；厂商同名子图只用于逐节点核对结构与 runner 契约。
    `scales` 给 KV cache 的标定 scale；静态 int8 入口（输出头）的 scale 在本子图上
    跑 `calibration_inputs` 现场标定。
    """
    from gml_bridge.export import _referenced_buffers, serialize_gml
    from gml_bridge.writer import write_gml
    from quant.calibration import ActivationScales, graph_static_scales

    if vendor is None:
        vendor = parse_vendor_graph(graph_dir_of(spec))
    gm, report = export_ceva_subgraph(model, spec)
    phase_source = None
    if use_opcompiler:
        from opcompiler_bridge.phase_source import cross_check, phase_source_from_graph

        phase_source = phase_source_from_graph(gm)
        mismatches = cross_check(phase_source)
        if mismatches:
            raise ValueError(f"FlagTree 相位与 GML 静态表不一致：{mismatches[:3]}")
    artifact = serialize_gml(gm, report, version=spec.version,
                             phase_source=phase_source, slots=spec.slots)
    sources, sinks = bind_boundaries(artifact.nodes, artifact.edges, spec)
    nodes, edges, weight_params, dq_specs = renumber_vendor_order(
        artifact.nodes, artifact.edges, sources=sources, sinks=sinks,
        weight_params=artifact.weight_params, dq_specs=artifact.dq_specs)
    # 要在重编号之后：入口的「第一个读者」按厂商编号定。
    apply_vendor_conventions(nodes, edges, stage=spec.stage)
    keys = align_to_vendor(nodes, edges, vendor)
    publish_vendor_labels(nodes, keys, vendor)
    problems = check_fields(nodes, keys, vendor)
    if problems:
        raise ValueError(f"{len(problems)} 处字段与厂商不一致，首条：{problems[0]}")

    static = sorted({n.fields["pim_static_quant_input"] for n in nodes
                     if n.fields.get("pim_static_quant_input")})
    scales = scales or ActivationScales()
    if static:
        if not calibration_inputs:
            raise ValueError(f"{spec.stage} 有静态 int8 入口 {static}，需要标定输入")
        scales = dataclasses.replace(scales, static_input=graph_static_scales(
            gm, static, calibration_inputs))

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    artifact = dataclasses.replace(artifact, nodes=nodes, edges=edges,
                           weight_params=weight_params, dq_specs=dq_specs,
                           buffer_names=_referenced_buffers(nodes))
    files = write_generated_files(artifact, gm, out_dir, scales)
    text = write_gml(nodes, edges, version=spec.version)
    (out_dir / GML_NAME).write_text(text)
    write_ceva_io_info(nodes, dataclasses.replace(spec, io=tuple(generated_io(spec.io, nodes, scales))),
                       out_dir)
    referenced = {name for node in nodes for _, name in _node_files(node)}
    written = {p.name for p in out_dir.glob("*.bin")}
    if referenced != written:
        raise ValueError(f"引用集与落盘集不一致：缺 {sorted(referenced - written)[:3]}，"
                         f"多 {sorted(written - referenced)[:3]}")
    for name in (GML_NAME, "IO_info.txt"):
        files.append({"path": name, "size_bytes": (out_dir / name).stat().st_size,
                      "sha256": _sha256(out_dir / name), "role": "generated",
                      "source": None})
    return {"stage": spec.stage, "layer_id": spec.layer_id, "files": files}


def apply_vendor_conventions(nodes, edges, *, stage: str = "decode") -> None:
    """补齐厂商子图的几处边界口径（逐图实测：decode/prompt/head 一致）。

    - 入口缓冲带 `original_name`；喂残差的入口带旁路槽的定标文件名。
    - hidden 出口缓冲带输入定标三件（KV 出口已有）。
    - RMSNorm 带 `input_zp`；EltwiseAdd 顶层不带 `input_sf_dtype`。
    - `group_attention_*` 写成字符串。
    """
    from contracts import gml_names as gml_names

    by_id = {n.node_id: n for n in nodes}
    targets = {e.target for e in edges}
    for node in nodes:
        fields = node.fields
        op = fields.get("op_type")
        if fields.get("is_buffer"):
            if node.node_id not in targets:
                fields["original_name"] = fields["label"]
                if fields.get("pim_is_hidden_entry"):
                    # hidden 入口带第一个读者那一槽的输入定标名。
                    reader = by_id[fields["output0_node_id"]].fields
                    slot = next(i for i in range(64)
                                if reader.get(f"input{i}_node_id") == node.node_id)
                    sf, zp = (("input_sf", "input_zp") if "input_sf" in reader
                              else (f"input_{slot}_sf", f"input_{slot}_zp"))
                    fields["output_sf"] = reader[sf]
                    fields["output_zp"] = reader.get(zp) or gml_names.zero_point(
                        fields["output0_node_id"], None if sf == "input_sf" else slot)
            else:
                if "input_sf" not in fields:
                    fields["input_sf"] = gml_names.scale(node.node_id)
                    fields["input_sf_dtype"] = "float16"
                    fields["input_zp"] = gml_names.zero_point(node.node_id)
                if fields.get("pim_kv_is_key") is not None:
                    # 写 KV 出口的换轴（prefill）以出口缓冲为输出、带出口定标名；
                    # DMA（decode）只改 output_buffer。
                    writer = by_id[fields["input0_node_id"]].fields
                    writer["output_buffer"] = fields["input_buffer"]
                    if writer.get("op_type") == "Transpose":
                        writer["output_sf"] = fields["input_sf"]
                        writer["output_zp"] = fields.setdefault(
                            "input_zp", gml_names.zero_point(node.node_id))
        elif op == "RMSNorm_vpu":
            fields.setdefault("input_zp", gml_names.zero_point(node.node_id))
        elif op == "Gemm" and fields.get("nmu_mode") == "fixed_point":
            fields.setdefault("weight_sf_dtype", "float16")
        elif op == "EltwiseAdd":
            fields.pop("input_sf_dtype", None)
        for key in ("group_attention_data_num", "group_attention_weight_num"):
            if isinstance(fields.get(key), int):
                fields[key] = str(fields[key])
        if stage == "prefill":
            # 厂商 prefill：Q 拆头的 Split 把 RoPE-DQ 输出按 fp16 记、不带
            # input_sf；EltwiseMul 不带 transpose（与 decode 不同，逐层实测）。
            if op == "Split" and by_id.get(fields.get("input0_node_id")) is not None \
                    and by_id[fields["input0_node_id"]].fields.get("op_type") == "Llama2ActivationDQ":
                fields["input_buffer_dtype"] = "float16"
                fields.pop("input_sf", None)
                fields["input_buffer"] = gml_names.data_buffer(node.node_id)
            elif op == "EltwiseMul":
                fields.pop("transpose", None)


def publish_vendor_labels(nodes, keys: dict[int, str], vendor: VendorGraph) -> None:
    """节点名取厂商同位节点的名字。

    runner 按名字绑定 I/O 与 RoPE 内部缓冲（`<Q 路 RoPE 名>_cos_value_buffer`），
    Registry 里也嵌着这些名字，所以 CEVA 链路的节点名是契约的一部分。
    """
    for node in nodes:
        label = vendor.nodes[keys[node.node_id]]["label"]
        node.fields["label"] = node.fields["name"] = label


# ---- 整包导出 ----

def bundle_targets(stage: str, layer_index: int | None, num_layers: int):
    """`--stage/--layer-index` → [(阶段, 层)]。顺序：prefill 各层、decode 各层、lm_head。"""
    stages = ("prefill", "decode", "lm_head") if stage == "all" else (stage,)
    targets = []
    for st in stages:
        if st == "lm_head":
            targets.append(("lm_head", None))
            continue
        layers = range(num_layers) if layer_index is None else [layer_index]
        targets += [(st, i) for i in layers]
    return targets


def bundle_subdir(stage: str, layer_id: int | None) -> Path:
    """包内子图目录，与厂商 65 图同构，GC 可直接使用。"""
    if stage == "lm_head":
        return Path("llama_lm_head") / RUNTIME_DIR
    sub = {"prefill": "prompt", "decode": "decode"}[stage]
    return Path(sub) / f"model_layers_{layer_id}_{sub}_v2" / RUNTIME_DIR


def bundle_complete(reference_root: Path, graphs: list[dict], digest: str) -> bool:
    """参考目录里的每张子图都已导出，且每张记录的生成摘要与整包一致。"""
    root = Path(reference_root)
    expected = {("lm_head", None)} if (root / "llama_lm_head").is_dir() else set()
    for stage, sub in (("prefill", "prompt"), ("decode", "decode")):
        for d in (root / sub).glob(f"model_layers_*_{sub}_v2"):
            expected.add((stage, int(d.name.split("_")[2])))
    if not expected:
        raise ValueError(f"{root} 里没有厂商子图")
    return bool(digest) and expected <= {(g["stage"], g["layer_id"]) for g in graphs} \
        and all(g.get("generation_digest") == digest for g in graphs)


def _calibration_tokenizer():
    from transformers import AutoTokenizer

    from genesim_bridge.paths import llama2_7b_model_dir

    return AutoTokenizer.from_pretrained(llama2_7b_model_dir())


def calibrate_model(model):
    """用固定标定语料跑整模型。

    输入走 runner 同源的已旋转 embedding：权重是旋转后的，用 HF 原始 embedding 跑出的
    激活不在同一个坐标系里，KV scale 会偏几倍。
    """
    from quant import calibration

    table = torch.load(ceva_embedding(), map_location="cpu")["weight"]
    model.model.embed_tokens.weight.data = table.to(model.model.embed_tokens.weight.dtype)
    tokenizer = _calibration_tokenizer()
    samples = [tokenizer(text, return_tensors="pt").input_ids
               for text in calibration.CALIBRATION_PROMPTS]
    return calibration.calibrate_llama(model, samples)


def _write_manifest(path: Path, manifest: dict) -> None:
    """临时文件加 rename 原子写清单，中途失败不会留下半截文件。"""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(path)


# 决定 GML 与 bin 内容的源码目录；文档、脚本、测试不影响生成结果。
GENERATION_SOURCE_DIRS = ("gml_bridge", "quant", "graph", "contracts", "opcompiler_bridge")


def compiler_revision(root: Path, *, head: str | None = None) -> str:
    """编译器版本：`HEAD 短哈希+生成源码内容摘要`。

    本期改动常常没有提交，只看 HEAD 分不出两次导出是否按同一份代码生成。
    """
    if head is None:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
    digest = hashlib.sha256()
    for directory in GENERATION_SOURCE_DIRS:
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return f"{head}+{digest.hexdigest()[:12]}"


OPCOMPILER_UNKNOWN = "unknown"


def opcompiler_revision() -> str:
    """算子编译器版本：FlagTree 源码树内容摘要。

    GML 的 FPSU 模式与相位由 FlagTree 的 C++ pass、IR 定义和 Python 编译前端共同
    决定，不跟着本仓源码走；读不到源码树时返回 unknown，导出侧据此拒绝局部续导。
    """
    from genesim_bridge.paths import flagtree_source

    source = flagtree_source(required=False)
    if source is None:
        return OPCOMPILER_UNKNOWN
    digest = hashlib.sha256()
    for relative in ("include", "lib", "python/src", "python/triton"):
        root = source / relative
        if not root.is_dir():
            return OPCOMPILER_UNKNOWN
        found = False
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".h", ".hpp", ".td"):
                digest.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
                found = True
        # 目录在但没有任何有效源文件，说明这棵树不完整，不能拿空输入算版本。
        if not found:
            return OPCOMPILER_UNKNOWN
    return digest.hexdigest()[:12]


def generation_digest(model, calibration, *, revision: str, opcompiler: str,
                      use_opcompiler: bool) -> str:
    """影响生成结果的输入摘要：checkpoint、整模型标定、两端编译器版本与是否接算子编译器。

    同一个包里的图必须同一口径：prefill 写 cache 与 decode 读 cache 用的是同一个 scale。
    """
    digest = hashlib.sha256(
        f"{revision}|{opcompiler}|{use_opcompiler}|{calibration.kv}".encode())
    tensors = [t for _, t in sorted(model.state_dict().items())] + list(calibration.head_inputs)
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def export_ceva_bundle(model, reference_root: Path, out_dir: Path,
                       targets, *, use_opcompiler: bool = True, log=print,
                       calibration=None) -> Path:
    """逐图导出并写 `bundle.json`。已完成的图在清单里登记，中断后重跑会覆盖。

    `calibration`（`quant.calibration.LlamaCalibration`）是整模型标定结果：每张层子图取
    本层的 KV scale，输出头取 final norm 的输入作标定样本。不给则在本模型上现标定。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "bundle.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.is_file()
                else {"graphs": []})
    revision = compiler_revision(Path(__file__).resolve().parent.parent)
    manifest.update(schema_version=1, reference_root=str(reference_root), complete=False)
    # 先把 complete=false 落盘：本次在第一张图前失败也不能留下旧的 true。
    _write_manifest(manifest_path, manifest)
    slots = CompileSlots.from_config(model.config, max_seq=1024)
    if calibration is None:
        calibration = calibrate_model(model)
    opcompiler = opcompiler_revision() if use_opcompiler else "not-used"
    digest = generation_digest(model, calibration, revision=revision,
                               opcompiler=opcompiler, use_opcompiler=use_opcompiler)
    # 口径变了只能整包重导：保留下来的旧图会和新图用不同的 scale 读写同一块 cache。
    kept_graphs = [g for g in manifest["graphs"]
                   if (g["stage"], g["layer_id"]) not in set(targets)]
    if kept_graphs and opcompiler == OPCOMPILER_UNKNOWN:
        raise ValueError(f"{out_dir} 读不到算子编译器版本（FlagTree 源码树），无法确认与已有图口径"
                         f"相同，不能只重导部分图；未覆盖的有 "
                         f"{sorted((g['stage'], g['layer_id']) for g in kept_graphs)[:3]}")
    # 按每张图实际生成时的摘要判断，不看可能已被本次覆盖的包级摘要。
    if any(g.get("generation_digest") != digest for g in kept_graphs):
        raise ValueError(f"{out_dir} 已有图的生成口径（checkpoint、标定或编译版本）与本次不同，"
                         f"不能只重导部分图；未覆盖的有 "
                         f"{sorted((g['stage'], g['layer_id']) for g in kept_graphs)[:3]}")
    manifest.update(compiler_revision=revision, opcompiler_revision=opcompiler,
                    generation_digest=digest,
                    calibration={"kv": [list(pair) for pair in calibration.kv]})
    for stage, layer_id in targets:
        spec = load_ceva_spec(reference_root, stage, layer_id, slots)
        directory = bundle_subdir(stage, layer_id)
        entry = export_ceva_graph(
            model, spec, out_dir / directory, use_opcompiler=use_opcompiler,
            scales=calibration.scales_for(layer_id),
            calibration_inputs=calibration.head_inputs if stage == "lm_head" else None)
        entry["directory"] = str(directory)
        entry["generation_digest"] = digest
        manifest["graphs"] = [g for g in manifest["graphs"]
                              if (g["stage"], g["layer_id"]) != (stage, layer_id)]
        manifest["graphs"].append(entry)
        _write_manifest(manifest_path, manifest)
        log(f"  已导出 {directory}（{len(entry['files'])} 个文件）")
    manifest["complete"] = bundle_complete(reference_root, manifest["graphs"], digest)
    _write_manifest(manifest_path, manifest)
    return manifest_path
