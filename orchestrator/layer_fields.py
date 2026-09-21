"""一层的有序字段字典。

几何从 GML 边取，查表走 `layer_hw_table`，编号走 `LayerIdentity`，
文件名抄 GML 节点已有的字符串或 `contracts.gml_names`。
23 类只写该类该有的键。依据：`docs/prepare_out-生成方案-20260919.md` §2。
"""

from __future__ import annotations

from collections import OrderedDict

from contracts import gml_names as names
from contracts.compile_slots import DEFAULT_SLOTS
from contracts.gml_quant import WEIGHT_GROUP_SIZE
from orchestrator import l2_alloc, layer_hw_table
from orchestrator.layer_hw_table import HwRow
from orchestrator.layer_id import LayerIdentity


DT_INT8, DT_FP16, DT_FP32 = 0, 1, 3
EXT_SIGNED, EXT_FLOAT = 1, 3
# 编译期 KV 槽位。prepare_out 的 mask/softmax/bmm 宽用它，不是导出时的 seq_len。
H, I, HD, S = (DEFAULT_SLOTS.hidden, DEFAULT_SLOTS.intermediate,
               DEFAULT_SLOTS.head_dim, DEFAULT_SLOTS.seq)


def align16(width: int) -> int:
    return l2_alloc.align_up(width, 16)


def stride_z(width: int, *, final: bool, scalar_align16: bool = False) -> int:
    """终相 align16(W)+15；中间相 =W；Width=1 的中间相有时 16。"""
    if final:
        return align16(width) + 15
    if scalar_align16 and width == 1:
        return 16
    return width


def dtype_code(elem_bytes: int) -> int:
    if elem_bytes == 1:
        return DT_INT8
    if elem_bytes == 2:
        return DT_FP16
    if elem_bytes == 4:
        return DT_FP32
    raise ValueError(f"未知元素宽度 {elem_bytes}")


def extension_of(dt: int) -> int:
    return EXT_FLOAT if dt in (DT_FP16, DT_FP32) else EXT_SIGNED


def numel_of_dims(dims: str) -> int:
    if not dims or dims == "unknown":
        return 0
    n = 1
    for part in dims.split("x"):
        if part.isdigit():
            n *= int(part)
    return n


def last_dim(dims: str) -> int:
    if not dims or dims == "unknown":
        return 0
    tail = dims.split("x")[-1]
    return int(tail) if tail.isdigit() else 0


def classify(identity: LayerIdentity, node, widths: dict[int, int],
             *, nodes_by_id: dict | None = None,
             weight_params: dict | None = None, slots=None) -> str:
    """映射到 23 类。"""
    slots = slots or DEFAULT_SLOTS
    layer = identity.layer
    op = layer.op_type
    phase = layer.phase
    label = layer.label
    node_id = layer.gml_node_id
    out_w = widths.get(node_id, 0)
    fields = getattr(node, "fields", {}) or {}

    if op == "DynamicScaling":
        return f"dq_p{phase + 1}"
    if op == "Softmax":
        return f"sm_p{phase + 1}"
    if op == "Mask":
        return "mask"
    if op == "RMSNorm_vpu":
        return "rmsnorm"
    if op == "EltwiseAdd":
        return "residual"
    if op == "EltwiseMul":
        return "mlp_mul"
    if op == "Llama2Activation":
        return ("rope_mul_cos", "rope_mul_sin", "rope_add_k")[phase]
    if op == "Llama2ActivationDQ":
        if phase is None or phase < 3:
            return ("rope_mul_cos", "rope_mul_sin", "rope_add_q")[phase or 0]
        return f"dq_p{phase - 2}"  # 3,4,5,6 -> dq_p1..p4
    if op == "MatMul":
        if "matmul1" in label or fields.get("weight_format") == "weights_transpose":
            return "bmm1"
        return "bmm2"
    if op == "Gemm":
        if any(c[-1].get("activation_op_type") == "Silu"
               for c in getattr(node, "contraction", ())
               if isinstance(c[-1], dict)):
            return "gemm_gate"
        if fields.get("kantor_mode") == "fp2int_converter":
            return "gemm_v"
        param = str(fields.get("pim_weight_param") or (weight_params or {}).get(node_id, ""))
        if "v_proj" in label or "v_proj" in param:
            return "gemm_v"
        if "o_proj" in param:
            return "gemm_qko"
        if "q_proj" in param or "k_proj" in param:
            return "gemm_qko"
        if "up_proj" in param:
            return "gemm_up"
        if "down_proj" in param:
            return "gemm_down"
        if nodes_by_id:
            for cid in _gml_list(node, "residual_output_buffer"):
                child = nodes_by_id.get(int(cid))
                if child is not None and child.fields.get("op_type") == "KV_Cache_DMA":
                    return "gemm_v"
        if out_w == slots.intermediate:
            return "gemm_up"
        in_w = _input_width(node, widths)
        if in_w == slots.intermediate:
            return "gemm_down"
        # 末尾 lm_head 也是 H→词表，decode-block-only 会丢掉。
        if "linear_7" in label or out_w not in (0, slots.hidden, slots.intermediate):
            return "gemm_qko"
        return "gemm_qko"
    raise ValueError(f"无法分类 {op} {label} phase={phase}")


def _input_width(node, widths: dict[int, int]) -> int:
    """入边末维。取第一条数据边。"""
    src = node.fields.get("input0_node_id")
    if src is None:
        return 0
    return widths.get(int(src), 0)


def _gml_str(node, key: str, default: str = "") -> str:
    value = node.fields.get(key, default)
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


def _upstream_dq(node, nodes_by_id, _depth: int = 0) -> int | None:
    """穿过折叠算子（Split / Transpose / Reshape / Concat / KV_Cache_DMA）
    找到上游那个 DynamicScaling 的 node_id。

    参考产物里 bmm1 的 `Datain` 直接引用 DQ 的终相平面，因为 TVM 侧 Split
    不是独立节点。我方这些算子在 GML 里存在但不占层，所以边名会停在它们
    身上 —— 不穿透就会写成消费者自己的 `input_buffer_<self>.bin`。
    """
    if nodes_by_id is None or _depth > 6:
        return None
    src = node.fields.get("input0_node_id")
    if src is None:
        return None
    parent = nodes_by_id.get(int(src))
    if parent is None:
        return None
    op = str(parent.fields.get("op_type", ""))
    if op == "DynamicScaling" or op == "Llama2ActivationDQ":
        return parent.node_id
    if op in ("Split", "Transpose", "Reshape", "Concat", "KV_Cache_DMA"):
        return _upstream_dq(parent, nodes_by_id, _depth + 1)
    return None


def _kv_cache_buffer(node, nodes_by_id, *, is_key: bool) -> str | None:
    """bmm 的权重通路引用的 KV cache 缓冲名。

    参考：32 个 bmm1 共用 K cache 一份、32 个 bmm2 共用 V cache 一份。
    K 与 V 用 `Cache idx`（0/1）区分，对应两个 KV_Cache_DMA 节点。
    """
    if nodes_by_id is None:
        return None
    dma = [n for n in nodes_by_id.values()
           if str(n.fields.get("op_type", "")) == "KV_Cache_DMA"]
    if not dma:
        return None
    # 逆拓扑编号：K 那条在前（node_id 更大）。
    pick = _pick_kv_dma(dma, is_key=is_key)
    out = pick.fields.get("output_buffer")
    return str(out) if out else None


def _pick_kv_dma(dma: list, *, is_key: bool):
    """按 `pim_kv_is_key` 选 K/V 那条 DMA；没有标记时退回编号。"""
    tagged = [n for n in dma
              if n.fields.get("pim_kv_is_key") is not None]
    if tagged:
        want = 1 if is_key else 0
        for n in tagged:
            if int(n.fields.get("pim_kv_is_key") or 0) == want:
                return n
    dma = sorted(dma, key=lambda n: n.node_id, reverse=True)
    return dma[0] if is_key else dma[-1]


def _cache_dma_buffer(nodes_by_id, *, is_key: bool) -> str | None:
    """`Original cache file`：读 GML 已声明的 `input_buffer_0`。

    参考 v_proj 写 `input_buffer_0_33.bin`，33 就是那个 ScatterND
    （GML 侧的 KV_Cache_DMA）。不能自己再拼一遍——拼出来的名字
    若与 from_fx 声明的不一致就是悬空引用（评审 3 §2.4）。
    """
    if nodes_by_id is None:
        return None
    dma = [n for n in nodes_by_id.values()
           if str(n.fields.get("op_type", "")) == "KV_Cache_DMA"]
    if not dma:
        return None
    pick = _pick_kv_dma(dma, is_key=is_key)
    declared = pick.fields.get("input_buffer_0")
    if declared:
        return str(declared)
    return names.data_buffer(pick.node_id, 0)


def _gml_list(node, key: str) -> list:
    value = node.fields.get(key, [])
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]




def _slotted_dataout(dataout: str, node, nodes_by_id) -> str:
    """下游是双输入层时，输出缓冲名要带槽号；下游是 Concat 时改指 Concat
    的输出（不带槎号）。

    缓冲按**消费者**编号（`contracts/gml_names.py`），双输入的消费者每个槽
    各一块，所以名字是 `input_buffer_<槽>_<消费者>.bin`。参考里 bmm1 的
    Dataout 写 `input_buffer_0_19.bin`，而 19 就是下游那个 mask。

    我方 GML 的 `_data_slot_count` 对 Mask/Eltwise 已按多槽命名，但 bmm 这类
    单槽生产者写出的是消费者的**无槽**名，两边对不上 —— 这里按消费者的槽数补。

    Concat 是另一种情形：实测参考产物 32 个 bmm2 头的 `Dataout file` 全部
    收敛到**同一个**文件名 `input_buffer_14.bin`，而 14 不是 Concat 自己
    （15），是 Concat **下游**那个节点——Concat 本身在参考产物里没有
    `params_*.txt`、没有 `net.ini` 条目，纯粹是 GML 结构里的记账节点，
    32 路头输出实际写进同一块 DDR 平面的各自区段（由 `Head output`/
    硬件头寻址决定，不是 Concat 的槎位）。按槎号命名（`input_buffer_<h>_
    <concat_id>.bin`）会让 32 个头各写一份从没被引用过的文件，也会让
    `Dataout file` 与参考的收敛模式对不上——这里改成直接借用 Concat
    自己的 `output_buffer` 字段（它已经按 Concat 的下游编号），32 个头
    共享同一个名字。
    """
    if not dataout or nodes_by_id is None:
        return dataout
    consumers = _gml_list(node, "residual_output_buffer")
    rope_child = None
    for cid in consumers:
        child = nodes_by_id.get(int(cid))
        seen = set()
        while child is not None and child.node_id not in seen:
            seen.add(child.node_id)
            op = str(child.fields.get("op_type", ""))
            if op in ("Llama2Activation", "Llama2ActivationDQ"):
                rope_child = child
                break
            if op in ("Transpose", "Reshape", "Split"):
                nxt = _gml_list(child, "residual_output_buffer")
                child = nodes_by_id.get(int(nxt[0])) if nxt else None
                continue
            break
        if rope_child is not None:
            break
    if rope_child is not None:
        rid = rope_child.node_id
        if dataout.startswith("input_buffer_") and "_0_" not in dataout:
            return f"input_buffer_0_{rid}.bin"
        return dataout
    if len(consumers) != 1:
        return dataout
    child = nodes_by_id.get(int(consumers[0]))
    if child is None:
        return dataout
    op = str(child.fields.get("op_type", ""))
    if op == "Concat":
        # Concat 分支必须走在"已带槎号"的早退之前：我方 GML 的 Concat 有
        # 32 个真实槎位，bmm2 的 `output_buffer` 本来就写成
        # `input_buffer_<h>_19.bin` 这种带槎号的形式（`_data_slot_count`
        # 把 Concat 当普通多槎算子），如果先判"已带槎号就直接放过"，这里
        # 永远走不到、bmm2 会保留它自己的槎位名而不是 Concat 下游的共享名。
        shared = child.fields.get("output_buffer")
        return shared if isinstance(shared, str) and shared else dataout
    if "_0_" in dataout or "_1_" in dataout:
        return dataout  # 已带槎号
    if op not in ("Mask", "EltwiseAdd", "EltwiseMul", "EltwiseSub", "EltwiseDiv"):
        return dataout
    # 消费者是双输入：本节点占它的哪个槎，看它的 input0/input1 指向谁。
    slot = 0 if str(child.fields.get("input0_node_id")) == str(node.node_id) else 1
    stem = dataout.removesuffix(".bin")
    prefix, _, consumer_id = stem.rpartition("_")
    if prefix != "input_buffer" or not consumer_id.isdigit():
        return dataout
    return f"input_buffer_{slot}_{consumer_id}.bin"


def semantic_stem(identity: LayerIdentity, node, kind: str,
                  *, residual_index: int = 0) -> str:
    """参考风格前端名。qidx 用我方 GML node_id，不追 TVM。"""
    layer = identity.layer
    nid = layer.gml_node_id
    lid = identity.layer_id
    head = layer.head_index
    gml_label = str((getattr(node, "fields", {}) or {}).get("label") or layer.label)

    if kind.startswith("dq_"):
        p = int(kind[-1])  # 1..4
        gp = "gp" if p == 1 else "act"
        if layer.op_type == "Llama2ActivationDQ":
            return (f"{gml_label}_dynamic_quantization"
                    f"_{gp}_dq_phase{p}_params_{lid}")
        if "mha_batch_matmul2" in layer.label or head is not None:
            return (f"dynamic_quantization_params_{nid}"
                    f"_{gp}_dq_phase{p}_params_{lid}")
        return (f"dynamic_quantization_params_{nid}"
                f"_{gp}_dq_phase{p}_params_{lid}")
    if kind.startswith("sm_"):
        p = int(kind[-1])
        gp = "gp" if p in (1, 3) else "act"
        h = 0 if head is None else head
        return (f"mha_softmax_head{h}_qidx{nid}_params_{nid}"
                f"_{gp}_sm_phase{p}_params_{lid}")
    if kind == "bmm1":
        h = 0 if head is None else head
        return f"mha_batch_matmul1_head{h}_qidx{nid}_params_{lid}"
    if kind == "bmm2":
        h = 0 if head is None else head
        return f"mha_batch_matmul2_head{h}_qidx{nid}_params_{lid}"
    if kind == "mask":
        h = 0 if head is None else head
        return f"mha_masking_head{h}_qidx{nid}_params_{lid}"
    if kind == "rmsnorm":
        return f"RMSNorm_params_{lid}"
    if kind == "residual":
        idx = residual_index or 1
        return f"add_{idx}_Add_qidx{nid}_params_{lid}"
    if kind == "mlp_mul":
        return f"mlp_mul_Mul_qidx{nid}_params_{lid}"
    if kind.startswith("rope_"):
        suffix = {"rope_mul_cos": "mul_cos", "rope_mul_sin": "mul_sin",
                  "rope_add_k": "add", "rope_add_q": "add"}[kind]
        return f"{gml_label}_{suffix}_params_{lid}"
    if kind.startswith("gemm") or kind in ("rmsnorm", "mlp_mul", "residual"):
        if kind == "residual":
            idx = residual_index or 1
            if "_Add_qidx" in gml_label:
                return gml_label
            return f"add_{idx}_Add_qidx{nid}_params_{lid}"
        return gml_label


def _geometry(kind: str, in_w: int, out_w: int) -> tuple[int, int, int, int, bool]:
    """返回 (in_w, out_w, in_dt, out_dt, final)。dt 是 0/1/3。"""
    final = True
    in_dt, out_dt = DT_FP16, DT_FP16
    if kind == "dq_p1":
        final, out_dt = False, DT_FP16
    elif kind in ("dq_p2", "dq_p3"):
        final, in_dt, out_dt = False, DT_FP16, DT_FP16
    elif kind == "dq_p4":
        in_dt, out_dt = DT_FP16, DT_INT8
    elif kind == "sm_p1":
        final, out_dt = False, DT_FP32
    elif kind == "sm_p2":
        final = False
    elif kind == "sm_p3":
        final, out_dt = False, DT_FP32
    elif kind == "sm_p4":
        final, in_dt, out_dt = False, DT_FP32, DT_FP16
    elif kind == "sm_p5":
        pass
    elif kind in ("gemm_qko", "gemm_gate", "gemm_up", "gemm_down"):
        in_dt = DT_INT8
    elif kind == "gemm_v":
        in_dt, out_dt = DT_INT8, DT_INT8
    elif kind in ("bmm1", "bmm2"):
        in_dt = DT_INT8
    elif kind == "rope_add_k":
        # K 进 cache 要定点化：实测 Output Data Type 0（int8）。
        out_dt = DT_INT8
    elif kind == "mask":
        pass
    return in_w, out_w, in_dt, out_dt, final


def _kernel(kind: str) -> tuple[int, int]:
    if kind.startswith(("gemm", "bmm")):
        return 1, 1
    return 0, 0


def _pooling(kind: str, in_w: int, group: int) -> tuple[int, int, int]:
    """Pooling Type, Filter Width, Filter Height。"""
    if kind == "dq_p1":
        return 4, group, 1
    if kind == "sm_p1":
        return 4, in_w, 1
    if kind == "sm_p3":
        return 3, in_w, 1
    return 0, 0, 0


def _activation_type(kind: str) -> int:
    if kind in ("dq_p3", "sm_p2", "sm_p4", "gemm_gate"):
        return 13
    return 0


def _is_dual(kind: str) -> bool:
    """双数据输入的层：两路各带一套 Use Clipping / Use FPSU / Pooling data type。"""
    return kind in ("residual", "mask", "mlp_mul") or kind.startswith("rope_")


# RoPE 六层在参考里**没有**的域。它们是 force-consecutive 的三连，中间结果不落
# DDR，也不参与块内比较，所以既没有 Residual 邻居、也没有 Input buffer file。
#
# **不含 `skip compare`**：参考里它是有的，只是被粘到 `force consecutive
# execution` 那一行末尾（Q27）。照常写成独立一行，由对拍器按已知排版差处理。
_ROPE_ABSENT = (
    "Residual input buffer", "Residual input buffer 0", "Residual input buffer 1",
    "Residual input buffer 2", "Residual output buffer",
    "Residual output buffer 0", "Residual output buffer 1",
    "Residual output buffer 2",
    "Input buffer file 0", "Input buffer file 1",
)


def _drop_rope_absent(fields: OrderedDict) -> None:
    for key in _ROPE_ABSENT:
        fields.pop(key, None)


def _tvm_name(orig: str) -> str:
    """`DDR * TVM Orig Buffer Name` 的值。

    参考里这一族只出现在**外部输入/输出**那些槽上（图入口 hidden、外部 mask、
    cos/sin 表、整块最终输出），值与同槽的 `Orig Buffer Name` 相同；
    出口那个例外，写成 `tvmgen_default_nprm_main_182_output_0`。

    参考的名字是 Relay 内部符号（`nprm_182_i12`），我方不走 TVM，拿不到；
    这里回落到我方的语义名，**保证域在、可解析**，数值等甲方确认是否当标识用
    （见 docs/prepare_out-txt-20260919.md 当前不足第 1 条）。
    """
    if orig == "output":
        return "tvmgen_default_nprm_main_output_0"
    return orig


def _base(identity: LayerIdentity, kind: str, in_w: int, out_w: int,
          hw: HwRow, group: int, layer=None) -> OrderedDict:
    in_w, out_w, in_dt, out_dt, final = _geometry(kind, in_w, out_w)
    kw, kh = _kernel(kind)
    ptype, pf_w, pf_h = _pooling(kind, in_w, group)
    sz = stride_z(out_w, final=final,
                  scalar_align16=(kind == "dq_p2" and out_w == 1))
    fields: OrderedDict[str, object] = OrderedDict()
    for key, value in layer_hw_table.CONSTANTS.items():
        if key in ("Pooling Pad Left", "Pooling Pad Right",
                   "Pooling Pad Top", "Pooling Pad Bottom"):
            continue
        if key == "skip compare":
            continue  # 放到最后
        # 双输入层只写带槽号的 `Use Clipping 0/1`，不写无后缀版本。
        if key == "Use Clipping" and _is_dual(kind):
            continue
        fields[key] = value
    fields["Input Data Type"] = in_dt
    fields["Input Width"] = in_w
    fields["Input Stride X"] = in_w
    fields["Output Data Type"] = out_dt
    fields["Output Width"] = out_w
    # DQ p2 的输出是**每组一个标量**：参考 Stride X 恒为 1，
    # Stride Z = Gn + 15（Gn=32→47、86→101，**不按 16 对齐**），Gn=1 时取 16。
    if kind == "dq_p2":
        fields["Output Stride X"] = 1
        fields["Output Stride Z"] = 16 if out_w <= 1 else out_w + 15
    else:
        fields["Output Stride X"] = out_w
        fields["Output Stride Z"] = sz
    fields["Weights Data Type"] = 2 if kind.startswith("gemm") else 0
    fields["Kernel Width"] = kw
    fields["Kernel Height"] = kh
    fields["Filter Horizontal Stride"] = 1 if kw else 0
    fields["Filter Vertical Stride"] = 1 if kh else 0
    fields["Input Format"] = hw.input_format
    if hw.output_format is not None:
        out_fmt = hw.output_format
        # Q 路 RoPE 后那条 DQ 的终相写 0，线性/分数 DQ 写 1（实测 1 vs 36）。
        if kind == "dq_p4" and identity.layer.op_type == "Llama2ActivationDQ":
            out_fmt = 0
        fields["Output Format"] = out_fmt
    fields["Quant_source"] = 1 if kind.startswith(("gemm", "bmm")) else 0
    if hw.kantor_mode is not None:
        km = hw.kantor_mode
        if layer is not None and layer.kantor_mode is not None:
            km = layer.kantor_mode
        fields["Kantor mode"] = km
    if hw.fpsu_mode is not None:
        fm = hw.fpsu_mode
        if layer is not None and layer.fpsu_mode is not None:
            fm = layer.fpsu_mode
        fields["Fpsu mode"] = fm
    if hw.pooling_data_type is not None:
        fields["Pooling data type"] = hw.pooling_data_type
    fields["Activation Type"] = _activation_type(kind)
    fields["Pooling Type"] = ptype
    fields["Pooling Filter Width"] = pf_w
    fields["Pooling Filter Height"] = pf_h
    fields["Pooling Horizontal Stride"] = 1 if ptype else 0
    fields["Pooling Vertical Stride"] = 1 if ptype else 0
    fields["Pooling Pad Left"] = 0
    fields["Pooling Pad Right"] = 0
    fields["Pooling Pad Top"] = 0
    fields["Pooling Pad Bottom"] = 0
    # 双输入层写带槽号的 `Use FPSU 0/1`，不写无后缀版本
    # （参考实测：mask / eltwise / RoPE 的 mul 与 add 都只有带槽号的那对）。
    if not _is_dual(kind):
        fields["Use FPSU"] = hw.use_fpsu
    fields["Input data extension"] = extension_of(in_dt)
    fields["Output data extension"] = extension_of(out_dt)
    return fields


def _layer_type_of(kind: str) -> str:
    if kind == "rmsnorm":
        return "vpu"
    if kind.startswith("gemm"):
        return "gemm"
    if kind.startswith("bmm"):
        return "matmul"
    if kind in ("dq_p1", "sm_p1", "sm_p3"):
        return "pooling"
    if kind.startswith("dq_") or kind.startswith("sm_"):
        return "activation"
    return "eltwise"


def _virtual(kind: str) -> tuple[bool, bool]:
    """(sys_virtual_in, sys_virtual_out)。中间相不落 DDR。"""
    if kind in ("dq_p1",):
        return False, True
    if kind in ("dq_p2",):
        return True, False
    if kind in ("dq_p3", "sm_p2", "sm_p3", "sm_p4"):
        return True, True
    if kind in ("dq_p4",):
        return True, False
    if kind == "sm_p1":
        return False, True
    if kind == "sm_p5":
        return True, False
    return False, False


def _elem_bytes(dt: int) -> int:
    return {0: 1, 1: 2, 3: 4}[dt]


def _l2_in_size(kind: str, in_w: int, in_dt: int) -> int:
    """L2 输入段字节数。

    参考实测（422 层）：
      int8 平面        = Width          （q_proj 4096、mask 2048 是 fp16 的 W×2）
      bmm              = Width + 16     （hd→144、S→1040）
      DQ p4 终相       = align16(W)+16 再 ×2  （4096→8224、1024→2080）
    """
    if kind in ("bmm1", "bmm2"):
        return in_w + 16
    if kind == "dq_p4":
        return (align16(in_w) + 16) * 2
    if kind == "sm_p2":
        # 实测 W=1024 → 2080 = (align16(W)+16)×2：exp 相要多留一行。
        return (align16(in_w) + 16) * 2
    return in_w * _elem_bytes(in_dt)


def _l2_dual_in_size(kind: str, width: int) -> int:
    """双输入层某一路的 L2 输入段字节数。

    参考实测：两路都是 fp16 平面，`Width × 2`
    （H=4096→8192、hd=128→256、I=11008→22016、S=1024→2048）。
    """
    return width * 2


def _l2_out_size(kind: str, out_w: int, out_dt: int) -> int:
    """L2 输出段字节数。

    参考实测（422 层）：

      bmm2      按 hidden 占位 8224
      DQ p2     Gn=32→96、Gn=1→32、Gn=86→208，即 `(align16(Gn)+16)×2`，
                Gn=1 例外取 32（16×2）
      DQ p4     终相 int8 平面 `align16(W)+16`：4096→4112、1024→1040、
                11008→11024（**不乘 2**，它已是 int8）
    """
    if kind == "bmm2":
        return (align16(H) + 16) * 2
    if kind == "dq_p2":
        # 实测 Gn=32→96、86→208，即 `align16((Gn+16)×2)`；Gn=1 例外取 32
        # （按公式会得 48，参考是 32 —— 单组时不留那一行余量）。
        return 32 if out_w <= 1 else align16((out_w + 16) * 2)
    if kind == "dq_p4":
        return align16(out_w) + 16
    return l2_alloc.l2_output_bytes(out_w, _elem_bytes(out_dt))


def build_layer_fields(
    identity: LayerIdentity,
    node,
    *,
    widths: dict[int, int],
    l2_offsets: dict[str, int],
    nodes_by_id: dict[int, object] | None = None,
    stem: str | None = None,
    terminal: bool = False,
    slots=None,
) -> OrderedDict[str, object]:
    """拼一层的有序字段。

    `slots` 是编译期槽位（`contracts.compile_slots`）。几何域一律从它取，
    不用模块级常量——那份只是单测夹具的默认值（llama2-7B）。
    """
    slots = slots or DEFAULT_SLOTS
    H = slots.hidden
    I = slots.intermediate
    HD = slots.head_dim
    S = slots.seq
    NH = slots.heads
    layer = identity.layer
    node_id = layer.gml_node_id
    kind = classify(identity, node, widths, nodes_by_id=nodes_by_id, slots=slots)
    hw = layer_hw_table.lookup(kind)
    in_w = last_dim(str(node.fields.get("original_shape", "")).replace("[", "").replace("]", "").replace(", ", "x")) if "original_shape" in node.fields else 0
    if in_w <= 0:
        in_w = widths.get(node_id, 0)
        # 入边：本节点作为消费者时，上游输出宽。
        src = node.fields.get("input0_node_id")
        if src is not None:
            in_w = widths.get(int(src), in_w) or in_w
    out_w = widths.get(node_id, in_w)

    # DQ / softmax 相位改写宽。
    group = WEIGHT_GROUP_SIZE
    if kind.startswith("dq_"):
        g = node.fields.get("global_pooling_group_size_phase_0")
        if g:
            group = int(g)
        if kind == "dq_p1":
            out_w = max(1, in_w // group) if in_w else out_w
        elif kind in ("dq_p2", "dq_p3"):
            gn = max(1, in_w // group) if in_w else out_w
            in_w = out_w = gn
        elif kind == "dq_p4":
            out_w = in_w
    if kind in ("sm_p1", "sm_p3"):
        out_w = 1
    if kind == "sm_p4":
        in_w = out_w = 1
    if kind == "bmm1":
        in_w, out_w = HD, S
    if kind == "bmm2":
        in_w, out_w = S, HD
    if kind == "mask":
        in_w = out_w = S
    if kind.startswith("sm_"):
        in_w = 1 if kind == "sm_p4" else S
        if kind in ("sm_p1", "sm_p3"):
            out_w = 1
        elif kind == "sm_p4":
            out_w = 1
        else:
            out_w = S
    if kind.startswith("rope_"):
        in_w = out_w = H
    if layer.op_type == "Llama2ActivationDQ" and kind.startswith("dq_"):
        in_w = H
        group = WEIGHT_GROUP_SIZE
        if kind == "dq_p1":
            out_w = H // group
        elif kind in ("dq_p2", "dq_p3"):
            in_w = out_w = H // group
        else:
            out_w = H
    elif kind.startswith("dq_") and (layer.head_index is not None or "mha_batch_matmul2" in layer.label):
        group = S
        if kind in ("dq_p1", "dq_p4"):
            in_w = S
            out_w = 1 if kind == "dq_p1" else S
        else:
            in_w = out_w = 1

    fields = _base(identity, kind, in_w, out_w, hw, group, layer)
    in_w, out_w, in_dt, out_dt, final = _geometry(kind, in_w, out_w)
    # 上面 _geometry 不改宽，用 fields 里已写入的。
    in_w = int(fields["Input Width"])
    out_w = int(fields["Output Width"])
    in_dt = int(fields["Input Data Type"])
    out_dt = int(fields["Output Data Type"])

    nid = node_id
    phase = layer.phase if layer.phase is not None else 0
    if kind.startswith("dq_") and layer.op_type == "Llama2ActivationDQ":
        dq_phase = (layer.phase or 0) - 3
        phase = max(0, dq_phase)

    # 相位 / softmax 号。
    if kind.startswith("dq_"):
        fields["dynamic quantization phase"] = int(kind[-1])
        # 分组域**只有 p1 写**：它是那个求组内 absmax 的池化相，后面三相
        # 吃的已经是「每组一个标量」，不再需要分组描述（实测 p2/p3/p4 都没有）。
        if kind == "dq_p1":
            fields["Group data axis"] = 3
            fields["Group data size"] = group
        if layer.op_type == "Llama2ActivationDQ":
            # 参考指那条 RoPE 的**语义层名**（`self_attn_Reshape_qidx4_params_22`），
            # 不是 FX 名（`add_1`）。
            fields["Original name"] = _gml_str(node, "label", layer.label)
            fields["Scale per tensor"] = 1
    if kind.startswith("sm_"):
        fields["softmax phase"] = int(kind[-1])
        fields["softmax axis"] = 3

    if kind == "dq_p4":
        fields["Kantor A source"] = 1
        fields["Group kantor A size"] = group
        fields["Kantor A scale axis"] = 2
        fields["Kantor A group axis"] = 3
        fields["Kantor A scale buffer file"] = names.phase_output_buffer(nid, 2)
        fields["Kantor A bias buffer file"] = names.phase_kantor_bias(nid, 3)
        fields["Kantor A scale shift buffer file"] = names.phase_kantor_shift(nid, 3)

    flp = layer.flp or hw.flp
    if flp:
        fields["Flp min exp"] = flp[0]
        fields["Flp max exp"] = flp[1]
        fields["Flp mantisa"] = flp[2]
    transpose = (layer.transpose_type if layer.transpose_type is not None
                 else hw.transpose_type)
    if transpose is not None:
        # DQ p3 的取值随组数变：Gn=1（分数 DQ）走 2，Gn>1（线性）走 1。
        if kind == "dq_p3" and layer.transpose_type is None:
            transpose = 2 if in_w <= 1 else 1
        fields["Transpose type"] = transpose
    if kind in ("dq_p2", "dq_p3", "sm_p2", "sm_p4", "gemm_gate"):
        lut_phase = {"dq_p2": 1, "dq_p3": 2, "sm_p2": 1, "sm_p4": 3}.get(kind)
        if kind == "gemm_gate":
            fields["Activation LUT file"] = names.activation_lut(nid)
            fields["Activation mode"] = 0
            fields["Activation special operators"] = 0
        else:
            fields["Activation LUT file"] = names.phase_lut(nid, lut_phase)
            mode = 1 if kind == "dq_p2" else 0
            if layer.activation_mode is not None:
                mode = layer.activation_mode
            fields["Activation mode"] = mode
            fields["Activation special operators"] = 4 if kind in ("dq_p3", "sm_p4") else 0

    # 残差邻居。
    ins = _gml_list(node, "residual_input_buffer")
    outs = _gml_list(node, "residual_output_buffer")
    if len(ins) > 1 or kind in ("residual", "mask", "mlp_mul") or kind.startswith("rope_"):
        for i, value in enumerate(ins):
            fields[f"Residual input buffer {i}"] = value
    elif ins:
        fields["Residual input buffer"] = ins[0]
    if len(outs) > 1:
        for i, value in enumerate(outs):
            fields[f"Residual output buffer {i}"] = value
    elif outs:
        fields["Residual output buffer"] = outs[0]

    fields["Dump files list"] = ""

    # Datain / Dataout。
    if kind.startswith("dq_"):
        # Q 路 Llama2ActivationDQ：裸 input_buffer 被 pop 成三槽，p1/p4 读
        # phase_0 平面（评审 4 §2.4）。
        raw_in = _gml_str(node, "input_buffer")
        if not raw_in:
            raw_in = (_gml_str(node, "input_buffer_phase_0")
                      or names.phase_input_buffer(nid, 0))
        if kind == "dq_p1":
            fields["Datain file"] = raw_in
            fields["Dataout file"] = [names.phase_output_buffer(nid, 0)]
        elif kind == "dq_p2":
            fields["Datain file"] = names.phase_output_buffer(nid, 0)
            fields["Dataout file"] = [names.phase_output_buffer(nid, 1)]
        elif kind == "dq_p3":
            fields["Datain file"] = names.phase_output_buffer(nid, 0)
            fields["Dataout file"] = [names.phase_output_buffer(nid, 2)]
        else:
            fields["Datain file"] = raw_in
            fields["Dataout file"] = [names.phase_output_buffer(nid, 3)]
        fields["Bias buffer file"] = names.phase_fpsu_bias(nid, phase)
        fields["Scaling buffer file"] = names.phase_fpsu_scale(nid, phase)
        fields["Scaling PS buffer file"] = names.phase_fpsu_post_shift(nid, phase)
        fields["output scale factor buffer"] = names.output_scale(nid)
        fields["Scale axis"] = 1
    elif kind.startswith("sm_"):
        p = int(kind[-1])
        raw_in = _gml_str(node, "input_buffer", names.data_buffer(nid))
        if p == 1:
            fields["Datain file"] = raw_in
            fields["Dataout file"] = [names.phase_output_buffer(nid, 0)]
        elif p == 2:
            fields["Datain file"] = raw_in
            fields["Dataout file"] = [names.phase_input_buffer(nid, 2)]
            fields["Bias buffer file"] = names.phase_output_buffer(nid, 0)
        elif p == 3:
            fields["Datain file"] = names.phase_input_buffer(nid, 2)
            fields["Dataout file"] = [names.phase_input_buffer(nid, 3)]
        elif p == 4:
            fields["Datain file"] = names.phase_input_buffer(nid, 3)
            fields["Dataout file"] = [names.phase_output_buffer(nid, 3)]
        else:
            fields["Datain file"] = names.phase_input_buffer(nid, 2)
            fields["Dataout file"] = [_gml_str(node, "output_buffer",
                                               names.data_buffer(nid))]
            fields["Scaling buffer file"] = names.phase_output_buffer(nid, 3)
        if p != 2:
            fields.setdefault("Bias buffer file", names.phase_fpsu_bias(nid, p - 1))
        if p != 5:
            fields.setdefault("Scaling buffer file", names.phase_fpsu_scale(nid, p - 1))
        fields.setdefault("Scaling PS buffer file",
                          names.phase_fpsu_post_shift(nid, p - 1))
        fields["output scale factor buffer"] = names.output_scale(nid)
        fields["input scale factor buffer"] = names.scale(nid)
        fields["Scale axis"] = 1
    else:
        datain = _gml_str(node, "input_buffer")
        dataout = _gml_str(node, "output_buffer")
        n_in = int(node.fields.get("input_count") or 1)
        # Mask 是否真的双输入要看 GML 边（`input_count`），不能只看 kind——
        # 同 `dual` 那处的判据（复核 20260921 §2.6）：没有接上 causal mask
        # 边界节点时，`Datain file 1`/`Input buffer file 1` 不该写，否则
        # 引用一个 GML 里没有对应节点声明的名字，是新的悬空引用。
        # residual/mlp_mul/rope_* 目前没有类似的"可能退化成单输入"的已知
        # 场景，仍按 kind 判定。
        is_dual_kind = (kind in ("residual", "mlp_mul")
                       or kind.startswith("rope_")
                       or (kind == "mask" and n_in >= 2))
        if is_dual_kind:
            # 双输入层的每个槽各一块缓冲。GML 声明了就读声明，禁止合成悬空名
            # （评审 4 §2.4：add_1 盘上是 input_buffer_14.bin）。
            a = (_gml_str(node, "input_buffer_0")
                 or _gml_str(node, "input_buffer"))
            b = _gml_str(node, "input_buffer_1")
            if not a:
                raise ValueError(
                    f"节点 {nid} kind={kind} 缺 input_buffer / input_buffer_0")
            if not b:
                if kind == "residual":
                    # 残差旁路边还没补时不要合成悬空名，槽 1 暂空——闸门会报。
                    b = ""
                else:
                    raise ValueError(
                        f"节点 {nid} kind={kind} 缺 input_buffer_1")
            if kind.startswith("rope_mul"):
                # RoPE 的表槎（cos/sin）不读本节点自己的 `input_buffer_N`
                # ——那是"缓冲按消费者编号"规则下每个消费者各自的一份，
                # 但表节点在 `from_fx.py` 里已经改成按 K 路消费者统一编号
                # 一次（复核 20260921 发现：Q/K 共用同一张表，参考产物两条
                # 链的 `Datain file` 都指向同一个文件名，不是各自按自己
                # 编号）。
                #
                # **两套槽位不是一回事**：GML 里表固定在槎 2（cos）或槎 1
                # （sin），不随 Q/K 变——`from_fx.py` 建表节点时就是按
                # cos/sin 分的，不按 Q/K 分。而 `Eltwise broadcast input
                # index`（txt 层的 0/1）只是说广播输入落在 txt 的
                # `Datain file 0` 还是 `Datain file 1` 那一格，Q 的 mul_cos
                # 是 0、其余三条是 1——这与 GML 槎号是两套编号（实测：K 的
                # mul_cos broadcast index=1 但 GML 表在槎 2；Q 的 mul_cos
                # broadcast index=0 但 GML 表也在槎 2，不是槎 0）。
                # `Eltwise broadcast input index` 字段本身此时还没写（要到
                # 下面 `if kind.startswith("rope_mul")` 那段，约 1036 行，
                # 才赋值），这里也不需要它——直接按 cos/sin 取 GML 槎号。
                gml_slot = 2 if kind == "rope_mul_cos" else 1
                q_cos = (kind == "rope_mul_cos"
                        and layer.op_type == "Llama2ActivationDQ")
                txt_slot = 0 if q_cos else 1
                table_node_id = node.fields.get(f"input{gml_slot}_node_id")
                table_node = (nodes_by_id.get(int(table_node_id))
                             if table_node_id is not None and nodes_by_id
                             else None)
                shared = (table_node.fields.get("output_buffer")
                         if table_node is not None else None)
                # 另一个 txt 槎（真实数据源，GML 槎 0）也要按 GML 槎 0 重取
                # ——不能沿用 `_gml_str(node, "input_buffer_1")`：那是本
                # 节点 GML 槎 1（sin 表）的值，`mul_cos` 用不上它，会把 sin
                # 表错当成数据源塞进另一格。参考 Q 路 mul_cos 的
                # `Datain file 1` 就是 `input_buffer_0_22.bin`（真实数据源），
                # 不是任何表。
                source = _gml_str(node, "input_buffer_0") or names.data_buffer(
                    nid, 0)
                if isinstance(shared, str) and shared:
                    if txt_slot == 0:
                        a, b = shared, source
                    else:
                        a, b = source, shared
            fields["Datain file 0"] = a
            fields["Input buffer file 0"] = a
            if b:
                fields["Datain file 1"] = b
                fields["Input buffer file 1"] = b
        else:
            # 参考的 Datain 指上游 DQ 的**终相平面** output_buffer_phase_3_{dq}。
            # 我方图里 Split / Transpose 被折叠，边名停在折叠算子上，所以要
            # 穿透它们找到真正的生产者（见 _upstream_dq）。
            dq_id = _upstream_dq(node, nodes_by_id)
            if dq_id is not None:
                datain = names.phase_output_buffer(dq_id, 3)
            fields["Datain file"] = [datain, datain] if datain else []
        dataout = _slotted_dataout(dataout, node, nodes_by_id)
        gml_label = _gml_str(node, "label", layer.label)
        if kind == "rope_mul_cos":
            dataout = f"{gml_label}_cos.bin"
        elif kind == "rope_mul_sin":
            dataout = f"{gml_label}_sin.bin"
        elif kind == "rope_add_q":
            dataout = names.phase_input_buffer(nid, 0)
            a = f"{gml_label}_cos.bin"
            b = f"{gml_label}_sin.bin"
            fields["Datain file 0"] = a
            fields["Datain file 1"] = b
            fields["Input buffer file 0"] = a
            fields["Input buffer file 1"] = b
        elif kind == "rope_add_k":
            a = f"{gml_label}_cos.bin"
            b = f"{gml_label}_sin.bin"
            fields["Datain file 0"] = a
            fields["Datain file 1"] = b
            fields["Input buffer file 0"] = a
            fields["Input buffer file 1"] = b
        fields["Dataout file"] = [dataout, dataout] if dataout else []
        if kind.startswith("gemm") or kind.startswith("bmm"):
            fields["Weights buffer file"] = _gml_str(
                node, "weight_buffer", names.weight_buffer(nid))
            if kind.startswith("bmm"):
                # bmm 的「权重」是 KV cache，32 个头**共用同一份**
                # （参考：全部 bmm1 都写 input_buffer_199.bin）。
                # 所以按 KV_Cache_DMA 节点取，不按本节点编号。
                cache = _kv_cache_buffer(node, nodes_by_id, is_key=kind == "bmm1")
                if cache:
                    fields["Weights buffer file"] = cache
            fields["Bias buffer file"] = _gml_str(
                node, "Bias_buffer_file", names.fpsu_bias(nid))
            fields["Scaling buffer file"] = _gml_str(
                node, "Scaling_buffer_file", names.fpsu_scale(nid))
            fields["Scaling PS buffer file"] = _gml_str(
                node, "Scaling_PS_buffer_file", names.fpsu_post_shift(nid))
            fields["weights scaling buffer file"] = _gml_str(
                node, "weight_sf", names.weight_scale(nid))
            fields["output scale factor buffer"] = _gml_str(
                node, "output_sf", names.output_scale(nid))
            # 参考：定标来自上游 DQ 的 phase_1（p1 = p0/256），穿透折叠算子取。
            in_sf = _gml_str(node, "input_sf")
            up_dq = _upstream_dq(node, nodes_by_id)
            if up_dq is not None:
                in_sf = names.phase_output_buffer(up_dq, 1)
            fields["input scale factor buffer"] = in_sf
            fields["Scale axis"] = 1
            if kind == "gemm_v":
                fields["Kantor A source"] = 0
                fields["Kantor A scale axis"] = 1
                fields["Kantor A scale buffer file"] = names.kantor_scale(nid)
                fields["Kantor A bias buffer file"] = names.kantor_bias(nid)
                fields["Kantor A scale shift buffer file"] = names.kantor_shift(nid)
                fields["Cache output"] = "true"
                fields["Cache idx"] = 1
                fields["Num Output Heads"] = NH
                # 指向 **KV_Cache_DMA 节点的槽 0**，不是本层的 Dataout：
                # 参考 v_proj 写 `input_buffer_0_33.bin`，33 就是那个
                # ScatterND（KV_Cache_DMA）。缓冲按消费者编号，cache 写入
                # 走它的槽 0。
                fields["Original cache file"] = _cache_dma_buffer(
                    nodes_by_id, is_key=False) or _slotted_dataout(
                        _gml_str(node, "output_buffer"), node, nodes_by_id)
        elif kind == "rmsnorm":
            fields["Datain file"] = [datain, datain]
            fields["Weights buffer file"] = names.weight_buffer(nid)
            fields["bias buffer file"] = names.rms_norm_epsilon(nid)
            fields["weights scaling buffer file"] = names.weight_scale(nid)
            fields["output scale factor buffer"] = names.output_scale(nid)
            fields["input scale factor buffer"] = names.scale(nid)
        elif kind in ("residual", "mlp_mul") or kind.startswith("rope_"):
            if kind.startswith("rope_"):
                fields["Original name"] = _gml_str(node, "label", layer.label)
                fields["Llama2Activation"] = "True"
                fields["Scale per tensor"] = 1
                # RoPE 子块的定标文件按**子块名**编号，不按节点号：
                #   add   -> Scaling_buffer_file_1_Llama2Activation_Add_Cos_<id>
                #            Scaling_buffer_file_2_Llama2Activation_Add_Sin_<id>
                #   mul   -> Scaling_buffer_file_6_Llama2Activation_Cos_<id>
                # 序号 1/2/6 是子块在 RoPE 内的位置，由 gml_names 统一给。
                sub = {"rope_mul_cos": ("Cos", "Cos"),
                       "rope_mul_sin": ("Sin", "Sin"),
                       "rope_add_k": ("Add_Cos", "Add_Sin"),
                       "rope_add_q": ("Add_Cos", "Add_Sin")}[kind]
                # 段号来自 RoPE 子块在硬件里的单元号（contracts.gml_hw_table.ROPE_UNITS）：
                #   sin 乘 → 4，Q 路 cos 乘 → 5，K 路 cos 乘 → 6，加法 → 1/2。
                # 一律写 (6,6) 会让 Q cos 段号错、sin 文件悬空（评审 3 §2.4）。
                if kind == "rope_mul_sin":
                    idx = (4, 4)
                elif kind == "rope_mul_cos":
                    unit = 5 if layer.op_type == "Llama2ActivationDQ" else 6
                    idx = (unit, unit)
                else:
                    idx = (1, 2)
                block0 = f"Llama2Activation_{sub[0]}"
                block1 = f"Llama2Activation_{sub[1]}"
                fields["Scaling buffer file 0"] = names.rope_scale(
                    nid, idx[0], block0)
                fields["Scaling buffer file 1"] = names.rope_scale(
                    nid, idx[1], block1)
                fields["Scaling PS buffer file 0"] = names.rope_post_shift(
                    nid, idx[0], block0)
                fields["Scaling PS buffer file 1"] = names.rope_post_shift(
                    nid, idx[1], block1)
                fields["Use Clipping 0"] = 0
                fields["Use Clipping 1"] = 0
                fields["Use FPSU 0"] = 1
                fields["Use FPSU 1"] = 1
            if kind == "mlp_mul" or kind.startswith("rope_mul"):
                fields.setdefault("output scale factor buffer",
                                  names.output_scale(nid))
            elif kind == "residual":
                # 残差的输出定标按**双输入消费者**编号（参考 add_1 → add_2 槽 0），
                # 不要取 outs[0] 的 RMSNorm（评审 4 §3.2 / §3.4）。
                outs = _gml_list(node, "residual_output_buffer")
                dual_ops = ("Mask", "EltwiseAdd", "EltwiseMul",
                            "EltwiseSub", "EltwiseDiv")
                picked, slot = None, None
                for cid in outs:
                    child = (nodes_by_id or {}).get(int(cid))
                    op = str(child.fields.get("op_type", "")) if child else ""
                    if op in dual_ops:
                        picked, slot = int(cid), 0
                        break
                if picked is None and outs:
                    picked = int(outs[0])
                if picked is not None:
                    fields["output scale factor buffer"] = names.scale(picked, slot)

    fields["layer type"] = _layer_type_of(kind)
    if kind == "rmsnorm":
        fields["sublayer type"] = "rmsnorm"
        fields["number of inputs"] = 1
        fields["Vpu Axis"] = -1
        fields["Input Scale Factor Buffer"] = names.scale(nid)
        fields["Output Scale Factor Buffer"] = names.output_scale(nid)
        fields["Weights Buffer File"] = names.weight_buffer(nid)
        fields["Weights Scaling Buffer File"] = names.weight_scale(nid)
        fields["Bias Buffer File"] = names.rms_norm_epsilon(nid)
        # 图入口 RMSNorm 的 TVM 溯源名由 _ddr_input 的 `external` 出。
    else:
        dual = kind in ("residual", "mask", "mlp_mul") or kind.startswith("rope_")
        # Mask 是否真的双输入要看 GML 边，不能只看 kind（复核 20260921
        # §2.6）：`gml_bridge/from_fx.py` 只在真的接上 causal mask 边界
        # 节点（`_mask_placeholder_of` 找得到）时才把 Mask 的
        # `input_count` 记成 2——seq_len==1 这类没有 causal mask 的退化
        # 场景，GML 侧仍是单槎，此时若这里仍按 kind 判定双输入，会写出
        # `input_buffer_1`/`L2 offset 1` 这些 GML 里根本不存在对应边的
        # 字段，产生新的悬空引用。residual/mlp_mul/rope_* 目前没有类似的
        # "可能退化成单输入"的已知场景，仍按 kind 判定。
        if kind == "mask":
            dual = int(node.fields.get("input_count") or 1) >= 2
        fields["number of inputs"] = 2 if dual else 1

    if kind.startswith("bmm"):
        fields["Head input"] = 1 if kind == "bmm1" else 0
        if kind == "bmm2":
            fields["Head output"] = 1
    if kind == "mask":
        fields["Eltwise mode"] = 2
        fields["Mask Buffer Index"] = 1
        fields["Mask Input Index"] = 1
        fields["Mask Data Type"] = 1
        fields["Use FPSU 0"] = 0
        fields["Use FPSU 1"] = 0
        fields["Use Clipping 0"] = 0
        fields["Use Clipping 1"] = 0
        # TVM 溯源名统一由 _ddr_input 的 `external` 出，这里不重复写。
        # 这个字段指**下游消费者**的输入 scale（实测节点 19 -> 18：
        # `output scale factor buffer: input_sf_18.bin`，18 是 Mask 的
        # 下游 Softmax，不是 Mask 自己）——之前一直写成 `names.scale(nid)`
        # 也就是 Mask 自己的节点号，凭 Mask 自己那时还是单槎、`names.scale`
        # 无槎位时恰好与它自己声明的 `input_sf_<nid>.bin` 撞了名字才没露出来；
        # Mask 改成双槎后自己不再写这个裸名，这个 bug 才暴露成悬空引用。
        outs = _gml_list(node, "residual_output_buffer")
        consumer_id = int(outs[0]) if outs else nid
        fields["output scale factor buffer"] = names.scale(consumer_id)
        ins = _gml_list(node, "residual_input_buffer")
        if len(ins) >= 2:
            fields["Residual input buffer 1"] = ins[1]
        # 缺第二邻居时不填字面量 0：那是伪造域（评审 3 §3.3）。
    if kind == "residual":
        # 第一条残差的旁路边 GML 侧还没补（from_fx 的 entry_bypass 是死代码）。
        # 缺边时不把槽 0 复制到槽 1——那也是伪造值。
        fields["Eltwise mode"] = 0
        fields["Fpsu mode 0"] = 1
        fields["Fpsu mode 1"] = 1
        fields["Pooling data type 0"] = 2
        fields["Pooling data type 1"] = 2
        fields["Use Clipping 0"] = 0
        fields["Use Clipping 1"] = 0
        fields["Use FPSU 0"] = 1
        fields["Use FPSU 1"] = 1
        # 两路各一套 FPSU 定标（带槽号，按本节点编号）。
        fields["Scaling buffer file 0"] = names.fpsu_scale(nid, 0)
        fields["Scaling PS buffer file 0"] = names.fpsu_post_shift(nid, 0)
        fields["Scaling buffer file 1"] = names.fpsu_scale(nid, 1)
        fields["Scaling PS buffer file 1"] = names.fpsu_post_shift(nid, 1)
        fields.pop("Fpsu mode", None)
        fields.pop("Use FPSU", None)
        fields.pop("Pooling data type", None)
    if kind == "mlp_mul":
        fields["Eltwise mode"] = 1
        fields["Kantor A source"] = 0
        # 两路各一套 FPSU 定标（带槽号）+ A/B 两套 Kantor 重定标。
        # 注意参考里 `kantor B scale shift buffer file` 的 k 是**小写**，
        # 另外三个 Kantor 键是大写开头 —— 照抄，不统一大小写。
        fields["Scaling buffer file 0"] = names.fpsu_scale(nid, 0)
        fields["Scaling PS buffer file 0"] = names.fpsu_post_shift(nid, 0)
        fields["Scaling buffer file 1"] = names.fpsu_scale(nid, 1)
        fields["Scaling PS buffer file 1"] = names.fpsu_post_shift(nid, 1)
        fields["Kantor B scale buffer file"] = f"kantor_B_scale_buffer_file_{nid}.bin"
        fields["Kantor B bias buffer file"] = f"kantor_B_bias_buffer_file_{nid}.bin"
        fields["kantor B scale shift buffer file"] = f"kantor_B_Shift_{nid}.bin"
        fields["Kantor A bias buffer file"] = f"kantor_A_bias_buffer_file_{nid}.bin"
        fields["Kantor A scale shift buffer file"] = f"kantor_A_Shift_{nid}.bin"
        fields["Kantor A scale axis"] = 1
        fields["Kantor B scale axis"] = 1
        fields["Fpsu mode 0"] = 1
        fields["Fpsu mode 1"] = 1
        fields["Pooling data type 0"] = 2
        fields["Pooling data type 1"] = 2
        fields["Use Clipping 0"] = 0
        fields["Use Clipping 1"] = 0
        fields["Use FPSU 0"] = 1
        fields["Use FPSU 1"] = 1
        fields.pop("Fpsu mode", None)
        fields.pop("Use FPSU", None)
        fields.pop("Pooling data type", None)
    if kind.startswith("rope_mul"):
        fields["Eltwise mode"] = 1
        fields["Eltwise broadcast dim"] = 3
        fields["Eltwise broadcast factor"] = NH
        # 广播那一路（cos/sin 表，hd 宽）占哪个槽位。实测：Q 路的 mul_cos
        # 把表放在槽 0，其余三条都放槽 1。`Runtime input <槽>` 跟着它走。
        q_cos = (kind == "rope_mul_cos"
                 and layer.op_type == "Llama2ActivationDQ")
        bcast = 0 if q_cos else 1
        fields["Eltwise broadcast input index"] = bcast
        fields["Eltwise broadcast Input Stride X"] = HD
        fields.pop("Pooling data type", None)
        if layer.op_type == "Llama2ActivationDQ":
            # Q 路两条 mul 各标一次「运行时输入」，槽位与广播槽一致。
            fields[f"Runtime input {bcast}"] = 1
        fields["Kantor A source"] = 0
        # RoPE 的 mul 两个单元都用，**但 A 只有 shift 与 bias、没有 scale**
        # （实测 4 条 mul 无例外）：A 路只做移位与偏置，缩放交给 B。
        # 子块名 `Llama2Activation_Cos` / `_Sin`，写出顺序照参考。
        block = f"Llama2Activation_{'Cos' if kind == 'rope_mul_cos' else 'Sin'}"
        fields["Kantor A scale shift buffer file"] = names.rope_kantor_shift(
            nid, block, "A")
        fields["Kantor B scale shift buffer file"] = names.rope_kantor_shift(
            nid, block, "B")
        fields["Kantor A bias buffer file"] = names.rope_kantor_bias(
            nid, block, "A")
        fields["Kantor B scale buffer file"] = names.rope_kantor_scale(
            nid, block, "B")
        fields["Kantor B bias buffer file"] = names.rope_kantor_bias(
            nid, block, "B")
        # mul 不写输出定标（实测 4 条都没有）。
        fields.pop("output scale factor buffer", None)
        _drop_rope_absent(fields)
        fields["Fpsu mode 0"] = 1
        fields["Fpsu mode 1"] = 1
        fields.pop("Fpsu mode", None)
        if kind == "rope_mul_sin":
            fields["rotary window size"] = 64
    if kind.startswith("rope_add"):
        fields["Eltwise mode"] = 0
        # 只有走 Kantor 重定标的那条 add 写这一项：K 路出 int8 cache
        # （Kantor mode 3）。Q 路出 fp16（mode 0），参考里没有这一项。
        if hw.kantor_mode:
            fields["Kantor A source"] = 0
            # 走重定标的那条 add 带 Kantor **A** 三件（mul 用 B 单元）。
            # 子块名是 `Llama2Activation_add`。
            block = "Llama2Activation_add"
            fields["Kantor A scale shift buffer file"] = names.rope_kantor_shift(
                nid, block, "A")
            fields["Kantor A scale buffer file"] = names.rope_kantor_scale(
                nid, block, "A")
            fields["Kantor A bias buffer file"] = names.rope_kantor_bias(
                nid, block, "A")
        fields["Fpsu mode 0"] = 1
        fields["Fpsu mode 1"] = 1
        fields.pop("Fpsu mode", None)
        # RoPE 三连不写 Pooling data type（实测 6 个文件都没有）。
        fields.pop("Pooling data type", None)
        if kind == "rope_add_k":
            fields["Cache output"] = "true"
            fields["Cache idx"] = 0
            fields["Num Output Heads"] = NH
            # 同 gemm_v：指 KV_Cache_DMA 节点的槽 0（K 那条）。
            fields["Original cache file"] = _cache_dma_buffer(
                nodes_by_id, is_key=True) or _slotted_dataout(
                    _gml_str(node, "output_buffer"), node, nodes_by_id)
        _drop_rope_absent(fields)

    # data order
    dual = fields.get("number of inputs") == 2
    if dual:
        fields["Input data order 0"] = 0
        fields["Input data order 1"] = 0
    else:
        fields["Input data order"] = 0

    vin, vout = _virtual(kind)
    if stem is None:
        stem = semantic_stem(identity, node, kind)
    if dual:
        fields[f"Virtual Input for Input Buffer 0        {stem}pcDataBuffIn0 is"] = (
            "true" if vin else "false")
        fields[f"Virtual Input for Input Buffer 1        {stem}pcDataBuffIn1 is"] = (
            "true" if vin else "false")
    else:
        fields[f"Virtual Input for Input Buffer          {stem}pcDataBuffIn is"] = (
            "true" if vin else "false")
    fields[f"Virtual Output for Output Buffer        {stem}pcDataBuffOut is"] = (
        "true" if vout else "false")
    fields["Sys virtual input 0"] = "true" if vin else "false"
    if dual:
        fields["Sys virtual input 1"] = "true" if vin else "false"
    fields["Sys virtual output"] = "true" if vout else "false"

    # Data scale（Gemm / MatMul）。
    if kind.startswith(("gemm", "bmm")):
        width = 1 if kind.startswith("bmm") else (86 if kind == "gemm_down" else 32)
        # 片上 Data scale 段：bmm 两路各一个 fp16 = 2；Gemm 按组数查表
        # （q/k/v/o=32、gate/up=16、down=30）。DDR 段另算，见下面 ddr_w。
        buf = 2 if kind.startswith("bmm") else (
            hw.data_scale_buf if hw.data_scale_buf is not None else width)
        fields["Group data axis"] = 3
        fields["Group data size"] = S if kind == "bmm2" else HD
        fields["Weight Format"] = 3 if kind == "bmm1" else (
            2 if kind == "bmm2" else None)
        if fields["Weight Format"] is None:
            fields.pop("Weight Format")
        fields["Data scale source"] = 0
        fields["Data scale format"] = 7
        fields["Data scale maps"] = 1
        fields["Data scale height"] = 1
        fields["Data scale width"] = width
        fields["Data scale stride X"] = width
        fields["Data scale stride Z"] = width
        fields["Data scale buffer size"] = buf
        fields["Runtime data scale"] = "false"
        fields["Registry data scale"] = "true"
        fields["DDR data scale num of buffers"] = 1
        fields["DDR data scale Maps"] = 1
        # DDR Data scale 段的宽与字节数按层类查表（参考 422 层实测）：
        #   bmm1 宽 32 / 64B，bmm2 宽 1 / 16B
        #   q/k/v/o 宽 32 / 64B，gate/up 宽 32 / 64B，down 宽 86 / 176B
        ddr_w = 32 if kind == "bmm1" else (1 if kind == "bmm2" else width)
        ddr_bytes = {"bmm1": 64, "bmm2": 16, "gemm_down": 176}.get(kind, 64)
        fields["DDR data scale Width"] = ddr_w
        fields["DDR data scale Height"] = 1
        fields["DDR data scale start Col"] = 0
        fields["DDR data scale start Row"] = 0
        fields["DDR data scale start Map"] = 0
        fields["DDR data scale stride X"] = 1
        fields["DDR data scale stride Z"] = ddr_w
        # 参考用**上游 DQ 的语义层名** + `_dequant_buff`，例如
        # `self_attn_Reshape_qidx4_params_22_dynamic_quantization_dequant_buff`。
        # 穿透折叠算子取到那个 DQ，再读它的 label。
        dq_for_scale = _upstream_dq(node, nodes_by_id)
        dq_label = None
        if dq_for_scale is not None and nodes_by_id:
            dq_node = nodes_by_id.get(dq_for_scale)
            if dq_node is not None:
                dq_label = str(dq_node.fields.get("label") or "")
        if dq_label and dq_label.startswith("self_attn_Reshape"):
            orig = f"{dq_label}_dynamic_quantization_dequant_buff"
        elif dq_label:
            orig = f"{dq_label}_dequant_buff"
        else:
            orig = (f"dynamic_quantization_params_"
                    f"{_gml_str(node, 'input0_node_id')}_dequant_buff")
        fields["DDR data scale Orig Buffer Name"] = orig
        fields["DDR data scale num of maps"] = 1
        fields["DDR data scale buffer offset"] = 0
        fields["DDR data scale buffer size"] = ddr_bytes

    # DDR Input / Output（落盘相）。
    if not vin:
        # 哪些槽来自块外。参考只给这些槽写 TVM 溯源名：
        #   图入口 RMSNorm 的槽 0、第一条残差的旁路槽 0 —— hidden
        #   mask 的槽 1 —— 外部 causal mask
        #   K 路两条 mul 的表槽 —— cos / sin
        external: dict[int, str] = {}
        entry_rms = False
        if kind == "rmsnorm" and nodes_by_id:
            rms_ids = [n.node_id for n in nodes_by_id.values()
                       if n.fields.get("op_type") == "RMSNorm_vpu"]
            entry_rms = bool(rms_ids) and nid == max(rms_ids)
        if entry_rms:
            external[0] = "hidden_states"
        elif kind == "residual" and not terminal:
            # 第一条残差的旁路那一路也来自图入口的 hidden（与入口 RMSNorm 同源）。
            external[0] = "hidden_states"
        elif kind == "mask":
            external[1] = "mask"
        elif kind.startswith("rope_mul") and layer.op_type == "Llama2Activation":
            table_slot = int(fields.get("Eltwise broadcast input index", 1))
            external[table_slot] = (
                "rope_cos" if kind == "rope_mul_cos" else "rope_sin")
        # 块内的语义中间态：Q 路 DQ 的 p1 吃的是 RoPE 三连的中间缓冲，
        # 参考写 `{RoPE 语义名}_mid_buf`（不是 `buffer<id>`，也不带 TVM 名）。
        semantic: dict[int, str] = {}
        if kind == "dq_p1" and layer.op_type == "Llama2ActivationDQ":
            rope_label = _gml_str(node, "label", layer.label)
            if rope_label:
                semantic[0] = f"{rope_label}_mid_buf"
        elif kind.startswith("rope_add"):
            # add 相的两路输入就是前两相各自的中间态，参考按子块语义命名
            # （`{RoPE 名}_mul_cos_buffer` / `_mul_sin_buffer`），不是
            # `buffer<id>` —— 那两个中间缓冲不对应任何 GML 节点编号。
            rope_label = _gml_str(node, "label", layer.label)
            if rope_label:
                semantic[0] = f"{rope_label}_mul_cos_buffer"
                semantic[1] = f"{rope_label}_mul_sin_buffer"
        _ddr_input(fields, in_w, dual=dual, kind=kind,
                   k_path=layer.op_type == "Llama2Activation",
                   external=external, semantic=semantic, node=node)
    if not vout:
        ddr_h = S if kind in ("gemm_v", "rope_add_k") else 1
        ddr_w = out_w
        if kind in ("gemm_v", "rope_add_k"):
            ddr_w = H
        if kind == "bmm2":
            ddr_w = H
        fields["DDR Output Maps"] = 1
        fields["DDR Output Width"] = ddr_w
        fields["DDR Output Height"] = ddr_h
        fields["DDR Output start Col"] = 0
        fields["DDR Output start Row"] = 0
        fields["DDR Output start Map"] = 0
        fields["DDR Output stride X"] = ddr_w
        fields["DDR Output stride Z"] = ddr_w * ddr_h
        orig = "value_cache_out" if kind == "gemm_v" else None
        if orig is None and kind == "rope_add_k":
            orig = "key_cache_out"
        if orig is None and kind == "rope_add_q":
            orig = f"{_gml_str(node, 'label', layer.label)}_mid_buf"
        if orig is None and kind == "rope_mul_cos":
            orig = f"{_gml_str(node, 'label', layer.label)}_mul_cos_buffer"
        if orig is None and kind == "rope_mul_sin":
            orig = f"{_gml_str(node, 'label', layer.label)}_mul_sin_buffer"
        if orig is None:
            orig = _output_orig_name(node, nodes_by_id, nid, kind)
        if terminal:
            # 整块的最终输出：参考写 `output`，并额外给一个 TVM 溯源名。
            orig = "output"
            fields["DDR Output Orig Buffer Name"] = orig
            fields["DDR Output TVM Orig Buffer Name"] = _tvm_name(orig)
        else:
            fields["DDR Output Orig Buffer Name"] = orig
        fields["DDR Output buffer offset"] = 0
        fields["DDR Output buffer size"] = 0
        fields["Graph Output"] = 1

    if kind.startswith("bmm"):
        fields["DDR Weight num of buffers"] = 1
        fields["DDR Weight Maps"] = 1
        fields["DDR Weight Width"] = H
        fields["DDR Weight Height"] = S
        fields["DDR Weight start Col"] = 0
        fields["DDR Weight start Row"] = 0
        fields["DDR Weight start Map"] = 0
        fields["DDR Weight stride X"] = H
        fields["DDR Weight stride Z"] = H * S
        fields["DDR Weight Orig Buffer Name"] = (
            f"buffer24_map{0 if layer.head_index is None else layer.head_index}"
            if kind == "bmm2"
            else f"buffer23_map{0 if layer.head_index is None else layer.head_index}")
        fields["DDR Weight num of maps"] = 1
        fields["DDR Weight buffer offset"] = 2112 if kind == "bmm1" else 8208
        fields["DDR Weight buffer size"] = slots.bmm_weight_elems
        fields["Weights input"] = 1
        fields["weights from ddr"] = "true"
        fields["Cache input"] = "true"
        fields["Cache idx"] = 0 if kind == "bmm1" else 1
        fields["Split Head Index"] = 0 if layer.head_index is None else layer.head_index
        fields["Total Split Head Num"] = NH
        fields["Split weight index"] = fields["Split Head Index"]
        fields["Total Split Weight Num"] = NH

    if layer.head_index is not None:
        fields["Split Head Index"] = layer.head_index

    # Task
    fields["Task ID"] = identity.task_id
    fields["Prev task count"] = len(identity.prev_tasks)
    for i, t in enumerate(identity.prev_tasks):
        fields[f"Prev task {i}"] = t
    fields["Next task count"] = len(identity.next_tasks)
    # Next 写在 count 之前更接近参考：参考是先 Next task 再 count。
    # 保持插入顺序：先把 Next task 插到 count 前较麻烦，这里按方案「可解析」即可。
    nxt = list(identity.next_tasks)
    if nxt:
        # 重新排：删 count，写 next，再写 count。
        fields.pop("Next task count")
        for i, t in enumerate(nxt):
            fields[f"Next task {i}"] = t
        fields["Next task count"] = len(nxt)

    fields["L2 qman buffer offset"] = layer_hw_table.CONSTANTS["L2 qman buffer offset"]
    fields["L2 qman buffer size"] = layer_hw_table.CONSTANTS["L2 qman buffer size"]

    # L2 输入。
    if dual:
        # RoPE 的 mul 是广播：一路是数据（H），另一路是 cos/sin 表（hd=128，
        # 片上段 256B = 128×fp16）。广播那一路由 `Eltwise broadcast input index`
        # 指定，参考两条 mul_cos 各占一种槽位组合，所以按它分宽，不能两路同宽。
        w0, w1 = in_w, in_w
        if kind.startswith("rope_mul"):
            bcast = int(fields.get("Eltwise broadcast input index", 1))
            if bcast == 1:
                w0, w1 = H, HD
            else:
                w0, w1 = HD, H
        fields["L2 input num of buffers 0"] = 1
        fields["L2 input num of buffers 1"] = 1
        offset0 = l2_offsets.get(identity.stem, 0)
        size0 = _l2_dual_in_size(kind, w0)
        fields["L2 input buffer offset 0"] = offset0
        fields["L2 input buffer size 0"] = size0
        fields["L2 input num of maps 0"] = 1
        fields["L2 input slice maps offset 0"] = 0
        fields["L2 input slice num of maps 0"] = 1
        fields["L2 input for DMA height 0"] = 1
        fields["L2 input for DMA width 0"] = w0
        # 槎 1 查 `l2_alloc.buffers_from_layers` 另开的 `#1` 槎（见该函数
        # 的说明）。查不到就退回 `offset0 + size0`——保证任何路径下两槎
        # 都不重叠，这是评审 §3.2 的下限要求；不是伪造一个与参考字节对齐
        # 的地址（域确认表 Q11 这一项本身标了「待确认」）。
        fields["L2 input buffer offset 1"] = l2_offsets.get(
            f"{identity.stem}#1", offset0 + size0)
        fields["L2 input buffer size 1"] = _l2_dual_in_size(kind, w1)
        fields["L2 input num of maps 1"] = 1
        fields["L2 input slice maps offset 1"] = 0
        fields["L2 input slice num of maps 1"] = 1
        fields["L2 input for DMA height 1"] = 1
        fields["L2 input for DMA width 1"] = w1
        fields["L2 input buffer id 0"] = hw.l2_input_id
        fields["L2 input buffer id 1"] = "5"
    else:
        fields["L2 input num of buffers"] = 2 if kind == "dq_p4" else 1
        # dq_p2 只写 `L2 input num of buffers`，不写 offset/size/maps 那一组
        # （它读的是 p1 留在片上的结果，没有独立入口段）。
        if kind not in ("sm_p5", "dq_p2", "dq_p3", "sm_p3", "sm_p4"):
            fields["L2 input buffer offset 0"] = l2_offsets.get(identity.stem, 0)
            fields["L2 input buffer size 0"] = _l2_in_size(kind, in_w, in_dt)
            fields["L2 input num of maps 0"] = 1
            fields["L2 input slice maps offset 0"] = 0
            fields["L2 input slice num of maps 0"] = 1
            fields["L2 input for DMA height 0"] = 1
            fields["L2 input for DMA width 0"] = in_w
            fields["L2 input buffer id 0"] = hw.l2_input_id

    if hw.l2_output_id is not None and not vout:
        fields["L2 output buffer offset"] = l2_offsets.get(identity.stem, 0)
        fields["L2 output buffer size"] = _l2_out_size(kind, out_w, out_dt)
        fields["L2 output buffer id"] = hw.l2_output_id
    elif kind in ("dq_p2",) or (not vout and hw.l2_output_id is None):
        if kind == "dq_p2":
            fields["L2 output buffer offset"] = l2_offsets.get(identity.stem, 0)
            fields["L2 output buffer size"] = max(16, out_w * 2)
            if hw.l2_output_id:
                fields["L2 output buffer id"] = hw.l2_output_id

    if hw.l2_weights_size:
        if kind == "rmsnorm":
            fields["L2 weights buffer offset"] = 8256
        else:
            # 双缓冲两个基址按层类查表（参考 422 层同类相同，换形状要重测）。
            off0 = hw.l2_weights_off0 if hw.l2_weights_off0 is not None else 8256
            off1 = (hw.l2_weights_off1 if hw.l2_weights_off1 is not None
                    else off0 + hw.l2_weights_size)
            fields["L2 weights buffer offset 0"] = off0
            fields["L2 weights buffer offset 1"] = off1
        fields["L2 weights buffer size"] = hw.l2_weights_size
        fields["L2 weights use partial buffer"] = 0 if kind == "rmsnorm" else 1
        fields["L2 weights use double buffer"] = 0 if kind == "rmsnorm" else 1
        if hw.l2_weights_id:
            fields["L2 weights buffer id"] = hw.l2_weights_id
    fields["L2 weights buffers per engine"] = hw.l2_weights_per_engine
    fields["Weights source"] = hw.weights_source

    if hw.l2_fpsu_size:
        if dual:
            fields["L2 fpsu buffer offset 0"] = QMAN_NEAR
            fields["L2 fpsu buffer offset 1"] = QMAN_NEAR + hw.l2_fpsu_size
        else:
            fields["L2 fpsu buffer offset"] = QMAN_NEAR
        fields["L2 fpsu buffer size"] = hw.l2_fpsu_size
    fields["Fpsu source"] = hw.fpsu_source
    if hw.l2_wscale_size:
        fields["L2 weight scale buffer offset 0"] = QMAN_NEAR
        fields["L2 weight scale buffer size"] = hw.l2_wscale_size
        fields["L2 data scale buffer offset engine 0"] = QMAN_NEAR
        fields["L2 data scale buffer id"] = "5"

    fields["Layer ID"] = identity.layer_id
    fields["Bytes in cycle internal memory read"] = 64
    fields["Bytes in cycle internal memory write"] = 64
    fields["L2 fpsu buffer id"] = hw.l2_fpsu_id
    fields["After concat"] = "false"
    fields["Before concat"] = "false"
    if kind.startswith("rope_") or layer.force_consecutive:
        fields["force consecutive execution"] = 1
    # `skip compare` 不写的层：输出**离开这一块**（写进 KV cache，或就是整块
    # 最终输出），块内没有下游可比。实测三条，规律一致：
    #   gemm_v          写 V cache
    #   rope_add_k      写 K cache（Kantor mode 3）
    #   图出口残差 add_2 整块最终输出
    # 其余 RoPE 五层**有**这一项，只是被粘在上一行（Q27），由 layer_render 处理。
    leaves_block = (kind in ("gemm_v", "rope_add_k") or terminal)
    if not leaves_block:
        fields["skip compare"] = 1
    return fields


QMAN_NEAR = 0x1FFF0000 - 1024


def _input_producer_id(node, slot: int) -> int | None:
    """该槽的生产者 node_id。"""
    if node is None:
        return None
    ins = _gml_list(node, "residual_input_buffer")
    if slot < len(ins):
        try:
            return int(ins[slot])
        except (TypeError, ValueError):
            return None
    value = node.fields.get(f"input{slot}_node_id")
    if value is None:
        return None
    return int(value)


def _output_orig_name(node, nodes_by_id, nid: int, kind: str) -> str:
    """DDR Output Orig 名：消费者 / 共享缓冲，不是自己的 node_id。

    参考 32 个 bmm2 全写 Concat 下游的 `buffer4`，sm_p5 写 Softmax 的消费者。
    自己写 `buffer{nid}` 是占位（评审 3 §2.3）。
    """
    if kind == "dq_p2":
        # Q 路那条 DQ 长在 RoPE 节点上（`Llama2ActivationDQ`），参考用它的
        # **语义层名** + `_dynamic_quantization_dequant_buff`，不是
        # `dynamic_quantization_params_<id>`（那是独立 DQ 节点的形态）。
        label = str((getattr(node, "fields", {}) or {}).get("label") or "")
        if label.startswith("self_attn_Reshape"):
            return f"{label}_dynamic_quantization_dequant_buff"
        return f"dynamic_quantization_params_{nid}_dequant_buff"
    consumers = _gml_list(node, "residual_output_buffer") if node else []
    if kind == "bmm2" and nodes_by_id and consumers:
        concat = nodes_by_id.get(int(consumers[0]))
        if concat is not None and concat.fields.get("op_type") == "Concat":
            downs = _gml_list(concat, "residual_output_buffer")
            if downs:
                return f"buffer{int(downs[0])}"
    if consumers:
        return f"buffer{int(consumers[0])}"
    return f"buffer{nid}"


def _ddr_input(fields: OrderedDict, width: int, *, dual: bool,
               kind: str = "", k_path: bool = False,
               external: dict[int, str] | None = None,
               semantic: dict[int, str] | None = None,
               node=None) -> None:
    """DDR 输入段。双输入层两路各一组，键名带槽号（单输入也带 0）。

    `external` 给出「这一槽来自块外」的名字（图入口 hidden、外部 mask、
    cos/sin 表），这些槽额外写 `TVM Orig Buffer Name`。
    """
    slots = (0, 1) if dual else (0,)
    for slot in slots:
        fields[f"DDR Input num of buffers {slot}"] = 1
        fields[f"DDR Input Maps {slot}"] = 1
        w = width
        if fields.get("layer type") == "matmul" and fields.get("Weight Format") == 3 and slot == 0:
            w = H
        elif kind.startswith("rope_mul"):
            # 广播那一路是 cos/sin 表（hd），另一路是数据（H）。
            bcast = int(fields.get("Eltwise broadcast input index", 1))
            w = HD if slot == bcast else H
        fields[f"DDR Input Width {slot}"] = w
        fields[f"DDR Input Height {slot}"] = 1
        fields[f"DDR Input start Col {slot}"] = 0
        fields[f"DDR Input start Row {slot}"] = 0
        fields[f"DDR Input start Map {slot}"] = 0
        fields[f"DDR Input stride X {slot}"] = fields[f"DDR Input Width {slot}"]
        fields[f"DDR Input stride Z {slot}"] = fields[f"DDR Input Width {slot}"]
        # 外部输入（图入口 hidden、外部 mask、K 路的 cos/sin 表）在参考里
        # 额外带一个 `TVM Orig Buffer Name`，值与本槽的 `Orig Buffer Name`
        # 相同。判据是「这一路来自块外」，不是层类。
        orig = external.get(slot) if external else None
        if orig:
            fields[f"DDR Input Orig Buffer Name {slot}"] = orig
            fields[f"DDR Input TVM Orig Buffer Name {slot}"] = _tvm_name(orig)
        elif semantic and slot in semantic:
            # 块**内**的语义中间态（Q 路 RoPE 的 `_mid_buf`）：有名字但不是
            # 外部输入，所以**不写** TVM 名 —— 参考那一层只有 Orig 一项。
            fields[f"DDR Input Orig Buffer Name {slot}"] = semantic[slot]
        else:
            # 输入 Orig 名 = 该槽生产者的 buffer<id>（评审 3 §2.3）。
            # 不能写 `buffer{slot}`——那是槽号占位，归一后才跟参考看起来一样。
            producer = _input_producer_id(node, slot)
            fields[f"DDR Input Orig Buffer Name {slot}"] = (
                f"buffer{producer}" if producer is not None else f"buffer{slot}")
        fields[f"DDR Input num of maps {slot}"] = 1
        fields[f"DDR Input buffer offset {slot}"] = 0
        # 参考：绝大多数层这一项是 0（由 L2A 自己算），只有 mlp_mul 与
        # RoPE 的两个 mul 写真实字节数 = Width × 2。
        #
        # 例外：K 路（Llama2Activation）两条 mul 的**表那一路**写 128，
        # 不是 128×2=256。Q 路同一位置写 256。实测 4 条 mul 无其他反例，
        # 语义未确认（疑似 K 路表只装半个周期），照抄。
        size = fields[f"DDR Input Width {slot}"] * 2
        if kind.startswith("rope_mul") and k_path and slot == int(
                fields.get("Eltwise broadcast input index", 1)):
            size = fields[f"DDR Input Width {slot}"]
        fields[f"DDR Input buffer size {slot}"] = (
            size if kind == "mlp_mul" or kind.startswith("rope_mul") else 0)
        fields[f"Graph Input {slot}"] = 1
