"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import (DEVICE_DPU, FUSED_TAIL_META_KEY, REDISTRIBUTE_META_KEY, SPEC_META_KEY)
from genesim_bridge.ir_fingerprint import assert_ir_fingerprint
from graph.spec_prop import _consumers_through_slices

if TYPE_CHECKING:  # 只为类型标注；运行时不拖进算子编译那条依赖链。
    from contracts.op_contract import PIMHardwareConfig

# GeneSim IR 的 `semantic_role` → 图编译器侧 `get_attr` 节点名的后缀。
#
# 用 IR 自带的语义标签确定每个 GEMM 是哪个投影，而不是去解析权重名。IR 侧的
# `tensor_id`（形如 `layer.0.q_proj.weight`）是自由字符串，上游改命名规则不会有
# 任何编译期信号；`semantic_role` 是 model_parser 显式写下的身份，改它是一次
# 明确的破坏性变更。
#
# 这张表本身消不掉：右边的 fx 节点名是图编译器侧的事实，IR 不可能知道。消掉的是
# 「依赖 GeneSim 的权重命名约定」和「扫 dependencies 反推身份」这两件事。

# sidecar entry 里 GeneSim 侧没有读者的三个字段。
# 它们是有意为之的排错信息，不是遗漏，登记在此以免后续被当死字段删掉；
# P1-2 的「回传字段必须有消费方」这条判据不适用于它们——它们是导出侧的
# 调试副本，不是回传通道的一部分。
#
# 注意「无消费者」只限 GeneSim 仓：本仓的测试与
# `scripts/run_full_pipeline.py` 的流水线自检都在读 `semantic_role`，
# 删掉写入点会让它们失败。
DEBUG_ONLY_SIDECAR_FIELDS = frozenset({"semantic_role", "weight", "shard_axis"})

_ROLE_TO_WEIGHT_PATTERN = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
    # MoE router 权重；专家 GEMM 带 expert_id，走 `_weight_pattern` 的专家分支。
    "router": "mlp.gate.weight",
    # 模型级输出头：权重不在任何一层里，`_weight_pattern` 对它有单独分支。
    "lm_head": "lm_head.weight",
}

# 专家投影：同一角色在每层出现 num_experts 次，必须靠 expert_id 区分。
_EXPERT_ROLES = frozenset({"gate_proj", "up_proj", "down_proj"})


def _weight_pattern(layer: int, role: str, expert_id: int = -1) -> str:
    """(层号, 投影身份, 专家号) → 图上的权重名后缀。

    专家权重形如 `layers.L.mlp.experts.E.gate_proj.weight`；router 是
    `layers.L.mlp.gate.weight`；其余投影沿用 `_ROLE_TO_WEIGHT_PATTERN`。
    """
    if role == "lm_head":
        # 输出头是整图一份的模型级权重，层号对它没有意义。
        return "lm_head.weight"
    if role == "router":
        return f"layers.{layer}.mlp.gate.weight"
    if expert_id >= 0:
        return f"layers.{layer}.mlp.experts.{expert_id}.{role}.weight"
    return f"layers.{layer}.{_ROLE_TO_WEIGHT_PATTERN[role]}"


def _activation_of(weight_node) -> str | None:
    """权重节点被折进主算子的激活名，没折过就是 None。

    标记打在 linear 节点上。分块后权重先经切片、类型转换才到 linear 子块，要穿过它们找。
    """
    for user in _consumers_through_slices(weight_node):
        tail = user.meta.get(FUSED_TAIL_META_KEY)
        if tail is not None:
            return tail.activation
    return None




def _graph_seq_len(gm: GraphModule) -> int:
    """图导出时用的序列长度，取输入 token 的长度。"""
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name == "input_ids":
            value = node.meta.get("val")
            if value is None:
                raise ValueError(
                    "input_ids 没有 val，无法确定导出序列长度"
                )
            return int(value.shape[-1])
    raise ValueError("图上没有 input_ids，无法确定导出序列长度")


def _export_seq_len(edge, export_seq_len: int) -> int | None:
    """边是否按导出序列长度线性计字节；不是时返回 None。

    激活张量恰有一维等于导出序列长度，整份字节里已经乘过它一次，仿真侧
    要先除回去再按本次 forward 的长度放大。整型张量是 token id，长度是
    词表位置；序列维出现不止一次的是 attention mask，按平方增长。两者
    都不随 token 数线性变化。
    """
    if not edge.dtype.startswith("float"):
        return None
    matching = [size for size in edge.shape if size == export_seq_len]
    if len(matching) != 1:
        return None
    if math.prod(edge.shape) == 0 or int(edge.nbytes) % math.prod(edge.shape):
        return None
    return export_seq_len


def _role_of_weight(target: str) -> tuple[str, int] | None:
    """权重名 → (投影身份, 专家号)；非投影权重返回 None，非专家记 -1。

    稠密投影与专家投影的权重名形状不同（`mlp.gate_proj.weight` 对
    `mlp.experts.E.gate_proj.weight`），所以按后缀认角色、按 segments 认专家。
    """
    target = str(target)
    if target.endswith("lm_head.weight"):
        return "lm_head", -1
    if ".mlp.gate.weight" in target:
        return "router", -1
    expert = -1
    match = re.search(r"\.mlp\.experts\.(\d+)\.", target)
    if match:
        expert = int(match.group(1))
    for role in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        if target.endswith(f"{role}.weight"):
            return role, expert
    return None


def _fused_activations(gm: GraphModule) -> list[dict]:
    """图上被折进主算子的激活，按层、投影身份与专家号列出。

    只认 `FUSED_TAIL_META_KEY`：权重的 linear 消费者带这个标记，才说明图编译器
    真的折过。模型 IR 里 gate 投影后接 silu 只是图的形状，不算数。
    """
    found = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        parsed = _role_of_weight(node.target)
        if parsed is None:
            continue
        role, expert = parsed
        if role == "lm_head":
            # 输出头是模型级权重，没有层号；融合清单只描述层内投影。
            continue
        # 目标形如 model.model.layers.0.mlp.gate_proj.weight，层号在 layers 后。
        layer = int(str(node.target).split("layers.")[1].split(".")[0])
        # 分块后一块权重对应多个子块，激活相同，只记一条。
        activation = _activation_of(node)
        if activation is None:
            continue
        entry = {"layer": layer, "role": role, "activation": activation}
        if expert >= 0:
            entry["expert"] = expert
        found.append(entry)
    return found


def _get_attr_node(gm: GraphModule, pattern: str):
    matches = [n for n in gm.graph.nodes if n.op == "get_attr" and pattern in n.target]
    if len(matches) != 1:
        raise ValueError(f"pattern={pattern!r} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0]


def _measure_kernel_tile_n(
    local_in: int,
    local_out: int,
    dtype: str,
    hardware: "PIMHardwareConfig | None",
    cache: Dict[tuple[int, int, str | None], tuple[int, str, int | None]],
    activation: str | None = None,
) -> tuple[int, str, int | None]:
    """编译一个本地分片形状，返回 `(tile_n, pim mlir 路径)`。

    这是算子编译器影响 GeneSim 代价的实际入口：`pim-tile-to-budget` 按 WRAM 预算
    搜出真实分块（llama2 的 4096 宽投影上是 512），而 GeneSim 原先只能用
    `conf/sim.yaml` 里拍下的 32。

    路径一并返回，GeneSim 侧照那份 pim mlir 生成 PIM trace——这一步才让编译器的
    真实 DMA 量和循环嵌套进入周期数，只喂一个分块常量是不够的（手写模板里
    `tile_size` 会约掉）。

    同一形状只编一次，结果放进 `cache`。第三个返回值是算子编译器回传的
    单台 MRAM 占用（`pim.placed-mram-bytes`），没回传时为 None。
    """
    # 激活也进键：同形状的两个投影一个折激活、一个不折，漏掉就编成同一份。
    key = (local_in, local_out, activation)
    if key in cache:
        return cache[key]

    # 延迟导入：这条路径需要 CUDA 和 FlagTree，纯放置导出用不上，不该在 import
    # 期就把依赖拽进来。
    from contracts.op_contract import OpCompileRequest, default_hardware_config
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    hw = default_hardware_config() if hardware is None else hardware
    # M=1 是 decode 口径。这条链现在三个维度都能分块，但代价模型只需要一条
    # 代表性的编译产物：分块搜索由 WRAM 预算决定，M=1 给出的 K/N 分块与
    # prefill 的 M 取值无关。
    request = OpCompileRequest(
        op="linear",
        arg_shapes=[(1, local_in), (local_out, local_in)],
        hardware=hw,
        dtype=dtype,
        num_tasklets=hw.num_tasklets,
        # 激活随请求下发，由算子层 -pim-fuse-activation 折进 tt.dot。
        activation=activation,
    )
    result = compile_op(request)
    if not result.pimir:
        raise RuntimeError(
            f"算子编译没有产出 pim mlir（本地形状 {local_in}->{local_out}）。"
            "缓存里的 .so 可能是旧版本留下的，用 compile_op(force=True) 重编，"
            "或删掉 .opcompiler_cache 后重跑。"
        )
    cost = analyze_ir(
        result.pimir,
        kernel_name="linear_kernel",
        grid=(1,),
        arg_values={},
        ir_level="pimir",
    )
    tile_n = int(cost.tile_n or 0)
    if tile_n <= 0:
        raise RuntimeError(
            f"从 pim mlir 里没读出输出分块（本地形状 {local_in}->{local_out}，"
            f"tile_n={cost.tile_n!r}）。"
        )
    if not result.pimir_path:
        raise RuntimeError(
            f"算子编译没有留下 pim mlir 的落盘路径（本地形状 {local_in}->{local_out}）。"
            "GeneSim 侧要照这份 IR 生成 trace，缺了路径就只能退回手写模板。"
        )
    pimir_path = str(result.pimir_path)
    cache[key] = (tile_n, pimir_path, cost.mram_bytes_per_dpu)
    return cache[key]


# 一块历史多少 token：读 KV 的算子按这个占位形状编一份，运行期按真实历史长度
# 折算循环次数。与 `_bpath_pimir` 的历史维占位同源——历史维在 IR 里就是这个占位。
_KV_BLOCK_TOKENS = 128

# 一个 B 路 trace 覆盖的查询 token 数。B 路 pimir 按「一个运行期单位」编译：
# 1 个 token × 该 DPU 的本地宽度，运行期再把整份 trace 重复实际 Tq 次。写成
# 128 会让运行期重复次数与 trace 内部形状各算一遍 token，代价放大两个数量级。
_BPATH_TRACE_TOKENS = 1

# 四个注意力节点里哪部分**不**随历史重复：得分乘的 q 只读一次，上下文乘的
# 累加结果只写一次；掩码与 softmax 整份算子都逐块跑。取值是 'in' / 'out'。
_KV_BLOCK_FIXED = {
    "GEMV_SCORE": "in",
    "GEMV_CONTEXT": "out",
    "MASK": None,
    "SOFTMAX": None,
}

_KV_DTYPE_BYTES = {"i8": 1, "i16": 2, "f16": 2, "bf16": 2, "f32": 4, "i32": 4}
_FUNC_SIG_RE = re.compile(r"tt\.func @\w+\((.*?)\)\s*(?:->\s*([^{]*?))?\s*\{", re.S)
_SIG_TENSOR_RE = re.compile(r"tensor<([0-9x]+)x(\w+)>")
# 函数签名没有返回类型（`tt.func` 只写参数），结果类型只能从算子行尾的 `->` 取。
_RESULT_TENSOR_RE = re.compile(r"->\s*tensor<([0-9x]+)x(\w+)>")


def _sig_tensor_bytes(shape: str, dtype: str) -> int:
    return prod(int(d) for d in shape.split("x")) * _KV_DTYPE_BYTES[dtype]


def _kv_block_attrs(body: str, op_type: str) -> str:
    """给读 KV 历史的算子写函数属性：一块多少 token，哪些字节只读写一次。

    属性写在函数上而不是逐个 op 上：softmax 会被展开成六个 op，写在算子上的
    标注会随展开丢掉，写在函数上则一直跟着这份内核。
    """
    if op_type not in _KV_BLOCK_FIXED:
        return ""
    fixed = _KV_BLOCK_FIXED[op_type]
    match = _FUNC_SIG_RE.search(body)
    if match is None:
        raise RuntimeError(f"{op_type} 的 B 路 IR 里找不到 tt.func 签名")
    fixed_in = fixed_out = 0
    if fixed == "in":
        fixed_in = _sig_tensor_bytes(*_SIG_TENSOR_RE.findall(match.group(1))[0])
    elif fixed == "out":
        fixed_out = _sig_tensor_bytes(*_RESULT_TENSOR_RE.findall(body)[0])
    return (f"pim.kvBlockTokens = {_KV_BLOCK_TOKENS} : i64, "
            f"pim.kvFixedIn = {fixed_in} : i64, "
            f"pim.kvFixedOut = {fixed_out} : i64")


def _with_func_attrs(body: str, attrs: str) -> str:
    """把属性插进 `tt.func` 签名后面。"""
    match = _FUNC_SIG_RE.search(body)
    if match is None:
        raise RuntimeError("B 路 IR 里找不到 tt.func 签名")
    signature = match.group(0)
    brace = signature.rindex("{")
    stamped = signature[:brace].rstrip() + f" attributes {{{attrs}}} {{"
    return body[:match.start()] + stamped + body[match.end():]


# 逐元素原语的算子种类。B 路原先只拿到 mnemonic（都是 `eltwise`）就一律发
# 加法，专家 SwiGLU 的中间乘法被记成了加法；语义必须由算子类型决定。
_ELTWISE_KIND = {
    "VECTOR_ADD": "add",
    "VECTOR_MUL": "mul",
}

# 查表激活的算子种类。两张表都是闭合的：不在表里的类型直接报错，不能按
# 一个默认种类编出看似合法的产物。
_LUT_KIND = {
    "SILU": "silu",
    "GELU": "gelu",
}


# MLP 三件套的语义角色。只有夹在 gate/up 与 down 之间的逐元素算子会被
# GeneSim 的 TP 展开按 elementwise 切成每台 DPU 一份。
_MLP_TP_ROLES = frozenset({"gate_proj", "up_proj", "down_proj"})
_ELEMENTWISE_TP_OPS = frozenset({"SILU", "GELU", "VECTOR_MUL"})


def _pimir_unit_shapes(input_shapes, output_shapes):
    """把一个算子的形状折成 pimir 的**编译**形状。

    查询 token 维固定为 1；`Tp`/`Tp+Tq` 这类历史维取一块 KV 的占位长度，
    运行期按块数绑定循环次数。侧车回传的是保留符号的模板（见
    `_attach_bpath_pimir`），符号在这里才换成具体占位：消费方凭符号区分
    token 维、历史维与特征维。
    """
    def value(dim):
        if not isinstance(dim, str):
            return int(dim)
        return _BPATH_TRACE_TOKENS if dim == "Tq" else _KV_BLOCK_TOKENS

    return (
        [tuple(value(dim) for dim in shape) for shape in input_shapes],
        [tuple(value(dim) for dim in shape) for shape in output_shapes],
    )


def _with_last_dim(shapes, width: int):
    """把每个形状的最后一维替换成 `width`；空形状原样保留。"""
    adjusted = []
    for shape in shapes:
        dims = list(shape)
        if dims:
            dims[-1] = width
        adjusted.append(dims)
    return adjusted


def _mlp_elementwise_local_width(
    operators_by_id, sidecar, ir, op_id: int,
) -> "int | None":
    """返回 MLP 逐元素算子单台 DPU 的宽度；不参与 TP 切分时返回 None。

    宽度取自 sidecar 里相邻多分片 GEMM 的本地特征数——GeneSim 的 TP 展开用
    的也是这份分片信息，编译器不另算一遍策略。只认 MLP 三件套邻居：attention
    的视图族虽然也挨着多分片投影，但 GeneSim 不切它们，必须保持全局宽度。
    """
    op = operators_by_id[op_id]
    if op.get("op_type") not in _ELEMENTWISE_TP_OPS:
        return None
    if not op.get("output_shapes") or not op["output_shapes"][0]:
        return None
    width = op["output_shapes"][0][-1]
    if not isinstance(width, int):
        return None

    candidates = set()

    def consider(neighbor_id: int, local_key: str, global_width) -> None:
        neighbor = operators_by_id.get(neighbor_id)
        if neighbor is None or neighbor.get("op_type") != "GEMM":
            return
        if neighbor.get("semantic_role") not in _MLP_TP_ROLES:
            return
        if global_width != width:
            return
        entry = (sidecar.get("operators") or {}).get(str(neighbor_id)) or {}
        shards = entry.get("shards") or []
        if len(shards) <= 1:
            return
        locals_ = {int(shard[local_key]) for shard in shards}
        if len(locals_) != 1:
            raise ValueError(
                f"op{neighbor_id} 的各分片 {local_key} 不一致：{sorted(locals_)}")
        local = locals_.pop()
        if local * len(shards) != width:
            raise ValueError(
                f"op{neighbor_id} 的本地宽 {local}×{len(shards)} 片 != "
                f"op{op_id} 的宽度 {width}")
        candidates.add(local)

    for dependency in ir.get("dependencies") or []:
        if dependency.get("dst_op_id") == op_id:
            neighbor_id = int(dependency["src_op_id"])
            neighbor = operators_by_id.get(neighbor_id)
            if neighbor is not None and neighbor.get("output_shapes"):
                consider(neighbor_id, "local_out_features",
                         neighbor["output_shapes"][0][-1])
        if dependency.get("src_op_id") == op_id:
            neighbor_id = int(dependency["dst_op_id"])
            neighbor = operators_by_id.get(neighbor_id)
            if neighbor is not None and neighbor.get("input_shapes"):
                consider(neighbor_id, "local_in_features",
                         neighbor["input_shapes"][0][-1])
    if len(candidates) > 1:
        raise ValueError(
            f"op{op_id} 相邻 MLP 分片的本地宽度不一致：{sorted(candidates)}")
    return candidates.pop() if candidates else None


def _bpath_pimir(mnemonic: str, input_shapes, output_shapes,
                 op_type: str = "",
                 hardware: "PIMHardwareConfig | None" = None) -> str:
    """给一个算子级 mnemonic 编一份 B 路 pimir，返回落盘路径。

    与 `_measure_kernel_tile_n` 对称：那条只服务 GEMM（A 路），这条服务其余
    算子。路径进 sidecar 的 `pimir_path` 后，GeneSim 的 `_operator_pimir`
    才能把相位链与视图类的真实 IR 喂给 trace 编译，而不是永远拿到 A 路的
    `tt.dot`。

    形状里的查询 token 维固定为 1：trace 是一个运行期单位，运行期按实际 Tq
    重复整份 trace。历史维（`Tp`/`Tp+Tq`）取一块 KV 的占位长度，运行期按
    块数绑定循环次数。两边各乘一次 token 会让代价放大两个数量级。
    """
    from opcompiler_bridge.driver import compile_op
    from opcompiler_bridge.oplevel_kernel import (
        concat_kernel, dynamic_quant_kernel, eltwise_kernel, gather_kernel, kv_cache_kernel,
        lut_kernel, mask_kernel, matmul_kernel, normalize_kernel,
        reshape_kernel, rope_kernel, softmax_kernel, split_heads_kernel,
        transpose_kernel,
    )
    from genesim_bridge.op_classify import (
        _attention_matmul_body, _module, paged_attn_geometry,
    )
    from opcompiler_bridge.oplevel_kernel import paged_attn_kernel

    ins, outs = _pimir_unit_shapes(input_shapes, output_shapes)

    def lut_body() -> str:
        """按算子类型发查表激活，未知类型直接拒绝。"""
        kind = _LUT_KIND.get(op_type)
        if kind is None:
            raise ValueError(
                f"未知查表激活算子 {op_type!r}：只认 {sorted(_LUT_KIND)}")
        return lut_kernel("op", ins[0], kind)

    def eltwise_body() -> str:
        """按算子类型发逐元素运算，未知类型直接拒绝。"""
        kind = _ELTWISE_KIND.get(op_type)
        if kind is None:
            raise ValueError(
                f"未知逐元素算子 {op_type!r}：只认 {sorted(_ELTWISE_KIND)}")
        return eltwise_kernel("op", ins[0], kind)

    builders = {
        "rope": lambda: rope_kernel(
            "op",
            ins[0][-3] if len(ins[0]) >= 3 else 1,
            ins[0][-2], ins[0][-1]),
        "softmax": lambda: softmax_kernel("op", *ins[0]),
        "reshape": lambda: reshape_kernel("op", ins[0], outs[0]),
        "transpose": lambda: transpose_kernel("op", ins[0], (1, 0, 2)),
        "concat": lambda: concat_kernel("op", ins, len(ins[0]) - 1),
        "split_heads": lambda: split_heads_kernel("op", ins[0], 1, len(outs)),
        "gather": lambda: gather_kernel("op", ins[0][0], outs[0][-1], ins[1][-1] if len(ins) > 1 else 1),
        "mask": lambda: mask_kernel("op", ins[0], ins[1] if len(ins) > 1 else ins[0]),
        "lut": lut_body,
        "eltwise": eltwise_body,
        "normalize": lambda: normalize_kernel("op", *ins[0], ins[0][-1]),
        "dynamic_quant": lambda: dynamic_quant_kernel(
            "op", ins[0][-1], max(ins[0][-1] // 128, 1), 128),
        # 第三个参数是**缓存元素数**，校验器要求它不小于一次写入的元素数。
        # 这里传整块写入的元素总数，不是末维：`[Tq, 4096]` 的末维是 4096，
        # 而一次搬的是 128x4096。这条路径直接发 IR 文本给 triton-opt，
        # 绕开了 `driver._make_oplevel_mlir` 里同名的守卫。
        "kv_cache": lambda: kv_cache_kernel("op", ins[0], prod(ins[0]), False),
        # 注意力那两次矩阵乘走 `_attention_matmul_body`，不直接调
        # `matmul_kernel`：后者只打 `stationarity = weight`（投影那条），
        # 照抄会把 KV 缓存说成模型权值。
        "matmul": lambda: _attention_matmul_body(
            "op", ins[0][0], ins[0][1], outs[0][-1]),
        # 融合分页注意力：q 是 [Tq, head_dim]，KV 是 [Tp+Tq, head_dim]。
        # 槽几何取占位值，块数在 GeneSim 侧按请求历史长度运行期绑定。
        "paged_attn": lambda: paged_attn_kernel(
            "op", ins[0], *paged_attn_geometry(ins[0][1], ins[1][0])),
    }
    body = builders[mnemonic]()
    if body is None:
        raise ValueError(f"{mnemonic} 需要单独的参数，不走这条通用入口")
    # 发出来的 IR 必须真的带这个算子的语义，而不只是"某种逐元素运算"。
    if mnemonic in ("eltwise", "lut"):
        expected = (_ELTWISE_KIND if mnemonic == "eltwise" else _LUT_KIND)[op_type]
        marker = (f"#pim.eltwise<{expected}>" if mnemonic == "eltwise"
                  else f"#pim.activation<{expected}>")
        if marker not in body:
            raise RuntimeError(
                f"{op_type} 的 B 路 IR 里没有 {marker}：语义在 lowering 时丢了")
    if mnemonic == "matmul" and "stationarity" not in body:
        raise RuntimeError(
            f"{mnemonic} 的 B 路 IR 缺 stationarity 标记，KV 缓存会被当成模型权值"
        )
    # 显式写下「这份 trace 覆盖几个查询 token」：TRANSPOSE/SPLIT 这类算子会
    # 挪动维序，靠结果首维猜会把头数当成 token 数，运行期重复次数随之错掉。
    func_attrs = f"pim.traceTokens = {_BPATH_TRACE_TOKENS} : i64"
    kv_attrs = _kv_block_attrs(body, op_type)
    if kv_attrs:
        func_attrs = f"{func_attrs}, {kv_attrs}"
    body = _with_func_attrs(body, func_attrs)
    from opcompiler_bridge.driver import _run_oplevel_triton_opt, _CACHE_DIR
    import hashlib, os
    text = _module(body)
    # 硬件参数进缓存键：同一份 IR 用 8 与 64 字节对齐编出的产物不同，只按
    # 文本哈希会把旧对齐的产物当成同一份复用。
    if hardware is not None:
        text = (f"// hardware: wram={hardware.wram_bytes_per_dpu} "
                f"mram={hardware.mram_bytes_per_dpu} "
                f"dma_align={hardware.dma_align}\n{text}")
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    out = _CACHE_DIR / f"bpath-{key}.pimir.mlir"
    if not out.is_file():
        expanded, _ = _run_oplevel_triton_opt(text, hardware)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(expanded)
        os.replace(tmp, out)
    return str(out)



# 算子类型到 mnemonic 的映射与成本抽取共用同一份（`op_classify.MNEMONIC_OF`）。
# GEMM 走 A 路的分块编译，不在这里。
from genesim_bridge.op_classify import MNEMONIC_OF

_BPATH_MNEMONIC = {op: mn for op, mn in MNEMONIC_OF.items() if op != "GEMM"}


def _pin_unsharded_ops(ir, sidecar, strategy) -> None:
    """给不做张量并行的算子写上 `dpu_id`。

    GeneSim 的钉死只认 `shards` 或顶层 `dpu_id`。GEMM 有 `shards`，其余算子
    条目是空列表，一个都钉不住，贪心把它们分到 GPU。这些算子落在所在层的
    流水段上，段内不切分，所以钉到该段的第一台 DPU。
    """
    if strategy is None:
        return
    num_layers = ir["num_layers"]
    op_layer = {
        op_id: layer
        for layer, group in enumerate(ir["subgraphs"])
        for op_id in group
    }
    for raw_op_id, entry in sidecar["operators"].items():
        if entry.get("shards") or "dpu_id" in entry:
            continue
        layer = op_layer.get(int(raw_op_id))
        if layer is None:
            continue
        entry["dpu_id"] = int(strategy.dpus_of_layer(layer, num_layers)[0])


def _attach_bpath_pimir(operators_by_id, sidecar, ir, hardware=None) -> None:
    """给每个算子级节点编一份 B 路 pimir 并写进 sidecar。

    同一 (mnemonic, 单位形状) 只编一次。编不出来直接抛：静默跳过的那条条目
    会缺 `pimir_path`，仿真侧退回手写模板，而现场只看到「缺路径」这个结果，
    看不到是谁、为什么编不出。契约不满足就让原因立刻暴露。

    每条 B 路条目同时回传编译单位：`trace_tokens` 与 `local_*_shapes`。形状
    模板保留 IR 的符号维（Tq/Tp+Tq）标注 token 维与历史维，特征维是本地宽。
    GeneSim 的 TP 展开只在分片形状与这份单位一致时继承 pimir，trace 验收也
    按它核对 tokens 与字节，避免「按全局形状编、按本地形状执行」的静默错配。
    """
    cache: dict = {}
    for op_id, op in operators_by_id.items():
        mnemonic = _BPATH_MNEMONIC.get(op["op_type"])
        if mnemonic is None:
            continue
        input_shapes = op["input_shapes"]
        output_shapes = op["output_shapes"]
        local_width = _mlp_elementwise_local_width(
            operators_by_id, sidecar, ir, op_id)
        if local_width is not None:
            input_shapes = _with_last_dim(input_shapes, local_width)
            output_shapes = _with_last_dim(output_shapes, local_width)
        # 算子类型进键：GEMV_SCORE 与 GEMV_CONTEXT 形状相同但块循环标注不同
        # （一个 q 只读一次、一个累加结果只写一次），合成一份会串味。
        key = (op["op_type"], mnemonic, json.dumps(input_shapes),
               json.dumps(output_shapes))
        if key not in cache:
            try:
                cache[key] = _bpath_pimir(
                    mnemonic, input_shapes, output_shapes,
                    op["op_type"], hardware=hardware)
            except Exception as exc:
                raise RuntimeError(
                    f"op{op_id}（{op['op_type']} -> {mnemonic}，形状 "
                    f"{input_shapes} -> {output_shapes}）没编出 "
                    f"pim mlir：{exc}。缺了它 GeneSim 会静默退回手写模板，"
                    "这个算子的原语等于没进仿真。"
                ) from exc
        # `shards` 必须在：消费方逐条取 `entry["shards"]` 统计分片数，缺了就抛。
        # B 路节点不切分，所以是空列表，而不是不写这个键。
        entry = sidecar["operators"].setdefault(str(op_id), {
            "op_type": op["op_type"], "device_hint": "pim", "shards": [],
        })
        entry["pimir_path"] = cache[key]
        entry["pimir_sha256"] = _file_sha256(cache[key])
        # 编译单位回传给仿真侧：token 数与每台 DPU 的形状模板。模板保留 IR
        # 的符号维（Tq/Tp+Tq），消费方才能区分 token 维、历史维与特征维；
        # 历史维固化成 128 会被运行期绑定形状误判成本地宽。
        entry["trace_tokens"] = _BPATH_TRACE_TOKENS
        entry["local_input_shapes"] = [list(shape) for shape in input_shapes]
        entry["local_output_shapes"] = [list(shape) for shape in output_shapes]


# 读 KV 缓存的算子：分解路径的两次矩阵乘，以及融合路径那一个节点。
_KV_OP_TYPES = ("GEMV_SCORE", "GEMV_CONTEXT", "PAGED_ATTN")


def _mark_kv_zone(ir: dict, sidecar: dict, kv_specs) -> None:
    """把读 KV 缓存的算子标进 kv 区，没给 KV 规格就一个都不标。

    两层判定：算子范围是读 KV 的两次注意力矩阵乘（`_KV_OP_TYPES`），层号
    取自内存规划的 KV 规格——哪台 DPU 持有哪一层的 KV 是它定的。
    """
    if not kv_specs:
        return
    kv_layers = set()
    for spec in kv_specs.values():
        kv_layers.update(spec.layers)
    layer_of = {
        op_id: layer
        for layer, group in enumerate(ir.get("subgraphs", []))
        for op_id in group
    }
    for op in ir["operators"]:
        if op["op_type"] not in _KV_OP_TYPES:
            continue
        if layer_of.get(op["op_id"]) not in kv_layers:
            continue
        # 只在已有条目上标：区间要靠 pimir 生成的 trace 才消费得到，凭空造
        # 一条没有 pimir_path 的条目会破坏 sidecar 的契约。
        entry = sidecar["operators"].get(str(op["op_id"]))
        if entry is not None:
            entry["zone"] = "kv"


def count_sidecar_classes(sidecar) -> tuple[int, int, dict]:
    """数一份 sidecar 里两类条目各有几条，以及每台 DPU 承担多少 GEMM 分片。

    两类条目不能混着数：GEMM 走过图切分、有 `shards`；算子级节点不切分、
    `shards` 是空列表。把总数报成「GEMM 算子数」会让 llama2-7B 的 224 个 GEMM
    显示成 6850 个——数字看着对，含义已经错了。
    """
    gemm_count = 0
    bpath_count = 0
    by_dpu: dict = {}
    for entry in sidecar["operators"].values():
        shards = entry.get("shards") or []
        if shards:
            gemm_count += 1
            # 一个 GEMM 可能切在多台 DPU 上，每台参与的都记一次。
            for shard in shards:
                by_dpu[shard["dpu_id"]] = by_dpu.get(shard["dpu_id"], 0) + 1
        else:
            bpath_count += 1
    return gemm_count, bpath_count, by_dpu


def _file_sha256(path: str) -> str:
    """算一个文件的 sha256，用于给 sidecar 记下 pim mlir 的内容身份。

    GeneSim 侧把它放进 trace 缓存签名：只有内容变了才重编 trace，换机器导致
    路径变化不会触发无谓的全量重编。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pimir_hardware_contract(sidecar: Dict[str, Any], hardware) -> None:
    """每份 pim mlir 的容量/对齐属性必须与本次硬件配置一致。

    sidecar 顶层自报 64 字节对齐还不够：B 路 IR 曾按另一份默认值写成 8，
    只核 JSON 字段会把「自报 64、实产 8」放过去。这里直接读模块属性。
    """
    from contracts.ir_payloads import module_int_attrs

    wanted = {
        "pim.dma-align": hardware.dma_align,
        "pim.mram-bytes": hardware.mram_bytes_per_dpu,
    }
    seen: set[str] = set()
    for entry in sidecar.get("operators", {}).values():
        path_text = entry.get("pimir_path")
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"pim mlir 不存在：{path}")
        attrs = module_int_attrs(path.read_text())
        for name, want in wanted.items():
            got = attrs.get(name)
            if got is None:
                raise ValueError(
                    f"{path.name} 的模块属性缺 {name}：无法证明它与 sidecar 的"
                    "容量契约同源。")
            if got != want:
                raise ValueError(
                    f"{path.name} 的 {name}={got} 与本次硬件配置 {want} "
                    "不一致：sidecar 自报与真实产物不是同一套。")


def _assert_every_gemm_has_known_role(ir: Dict[str, Any]) -> None:
    """核对 IR 里每个 GEMM 都带一个本表认得的 `semantic_role`。

    正向校验：从 IR 出发查表，而不是从表出发查 IR。上游新增一种投影、或者改了
    role 的写法，都会在这里报错并指出具体算子，而不是等到取 fx 节点时才撞上一句
    「应恰好匹配 1 个」——那个报错指向的位置是错的。

    专家投影还要求 `expert_id`：同层同角色的 128 个 GEMM 只靠 role 无法区分，
    缺了会在 `_get_attr_node` 撞上「匹配 128 个」。router 反过来不能带专家号。
    """
    missing: list[int] = []
    unknown: Dict[int, str] = {}
    bad_expert: list[str] = []
    # 同一套 gate/up/down 角色在稠密模型里是层内唯一的 FFN，在 MoE 里是
    # num_experts 个专家之一；按 IR 声明的 MoE 元数据区分两种语义。
    moe = int(ir.get("moe_num_experts", 0)) > 0
    for op in ir["operators"]:
        if op["op_type"] != "GEMM":
            continue
        role = op.get("semantic_role") or ""
        expert = int(op.get("expert_id", -1))
        if not role:
            missing.append(op["op_id"])
        elif role not in _ROLE_TO_WEIGHT_PATTERN:
            unknown[op["op_id"]] = role
        elif moe and role in _EXPERT_ROLES and expert < 0:
            bad_expert.append(f"op{op['op_id']}({role}) 缺 expert_id")
        elif moe and role == "router" and expert >= 0:
            bad_expert.append(f"op{op['op_id']}(router) 不应带 expert_id={expert}")
        elif role == "lm_head" and expert >= 0:
            bad_expert.append(
                f"op{op['op_id']}(lm_head) 不应带 expert_id={expert}")
        elif not moe and expert >= 0:
            bad_expert.append(
                f"op{op['op_id']} 是稠密模型却带 expert_id={expert}")

    if missing:
        raise ValueError(
            f"IR 里有 {len(missing)} 个 GEMM 没有 semantic_role（如 "
            f"op{missing[:6]}）。放置导出按语义标签确定投影身份，缺了就无法匹配；"
            "请用当前的 model_parser.py 重新生成 IR。"
        )
    if unknown:
        head = ", ".join(f"op{op_id}={role!r}" for op_id, role in list(unknown.items())[:4])
        raise ValueError(
            f"IR 里有 {len(unknown)} 个 GEMM 的 semantic_role 不在已知投影列表里："
            f"{head}。已知的是 {sorted(_ROLE_TO_WEIGHT_PATTERN)}；"
            "IR 结构变了，需要同步这张表。"
        )
    if bad_expert:
        raise ValueError(
            f"IR 里有 {len(bad_expert)} 个 GEMM 的专家号不合法："
            f"{bad_expert[:4]}。专家投影必须带 expert_id，router 不能带。"
        )


def _build_pimir_exempt_ops(
    ir: Dict[str, Any], sidecar: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """严格模式下允许没有 pimir 的算子清单，未知设备算子直接失败。

    豁免只有两类：模型入口/出口与 host 执行。host 证据由 `export_gemm`
    按 `spec.device` 写进 IR 与 sidecar 条目；这里逐条复制证据，不再用
    `expert_id` 或算子类型推断落点。缺证据的设备算子（含 router、注意力
    投影与输出头）一律失败。
    """
    exempt: Dict[str, Dict[str, Any]] = {}
    for op in ir["operators"]:
        op_id = str(op["op_id"])
        entry = sidecar["operators"].get(op_id) or {}
        if entry.get("pimir_path"):
            continue
        op_type = op["op_type"]
        if op_type in ("MODEL_INPUT", "MODEL_OUTPUT"):
            exempt[op_id] = {
                "op_type": op_type, "device_hint": "auto",
                "reason": "模型入口/出口，没有设备算子",
            }
            continue
        if str(op.get("device_hint", "")).lower() != "cpu":
            raise ValueError(
                f"op{op['op_id']}（{op_type}）没有 pimir_path，且源 IR 的落点"
                f"是 {op.get('device_hint')!r} 而不是 host。未知设备算子不能"
                "自动豁免：它会被钉到设备上却没有任何原语。专家 GEMM 也"
                "不能借「非 DPU GEMM」豁免，它一定属于某台 DPU。请在 "
                "genesim_bridge/op_classify.py 登记配方，或在 IR 里标 host。")
        evidence: Dict[str, Any] = {
            "op_type": op_type,
            "device_hint": "cpu",
            "semantic_role": str(op.get("semantic_role", "")),
            "expert_id": int(op.get("expert_id", -1)),
            "reason": "host 执行，算子编译器没有对应原语",
        }
        if op_type == "GEMM":
            # host GEMM 的落点证据由 export_gemm 单独写入 `host_gemms`：
            # op_type/角色/权重路径必须与源 IR 对得上，不能只凭 expert_id 推断。
            host_evidence = (sidecar.get("host_gemms") or {}).get(op_id)
            if not isinstance(host_evidence, dict) or str(
                    host_evidence.get("device_hint", "")).lower() != "cpu":
                raise ValueError(
                    f"op{op['op_id']}（GEMM）在 IR 里标了 host，但导出没有"
                    "落点证据：请用当前 export_placement_to_genesim 重新导出，"
                    "不能手写豁免。")
            if str(host_evidence.get("semantic_role", "")) != str(
                    op.get("semantic_role", "")):
                raise ValueError(
                    f"op{op['op_id']} 的 host GEMM 证据角色 "
                    f"{host_evidence.get('semantic_role')!r} 与源 IR 的 "
                    f"{op.get('semantic_role')!r} 不一致。")
            if int(op.get("expert_id", -1)) >= 0:
                raise ValueError(
                    f"op{op['op_id']} 是专家 GEMM，不能按 host GEMM 豁免。")
            if not host_evidence.get("weight"):
                raise ValueError(
                    f"op{op['op_id']} 的 host GEMM 证据缺权重身份。")
            evidence["weight"] = str(host_evidence["weight"])
        exempt[op_id] = evidence
    return exempt


def _hw_budget_payload(hw_budget) -> Dict[str, int]:
    """容量契约三件套：总容量、系统预留与对齐，任一非法直接报错。

    sidecar 是 GeneSim 核对容量的唯一通道；写出时就校验，避免把一个
    0 预留或 1 字节对齐的预算固化进产物，到仿真侧才暴露。
    """
    payload = {
        "mram_bytes_per_dpu": int(hw_budget.mram_bytes),
        "sys_reserve_bytes": int(hw_budget.sys_reserve_bytes),
        "align": int(hw_budget.align),
    }
    for field, value in payload.items():
        if value <= 0:
            raise ValueError(f"hw_budget.{field}={value} 必须为正整数")
    if payload["sys_reserve_bytes"] >= payload["mram_bytes_per_dpu"]:
        raise ValueError(
            f"hw_budget 系统预留 {payload['sys_reserve_bytes']} 不小于单 DPU "
            f"MRAM {payload['mram_bytes_per_dpu']}：没有可用预算")
    # 1 是 2 的幂但不是有效的 DMA 对齐：MRAM 偏移要按至少 8 字节
    # （`PIMHardwareConfig` 的最小 dma_align）落位。
    if payload["align"] < 8 or payload["align"] & (payload["align"] - 1):
        raise ValueError(
            f"hw_budget.align={payload['align']} 必须是 2 的幂且不小于 8")
    return payload


def export_placement_to_genesim(
    gm: GraphModule,
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
    *,
    dtype: str = "float16",
    hardware: "PIMHardwareConfig | None" = None,
    measure_kernel_tiles: bool = False,
    dpu_to_cluster: tuple[tuple[int, int], ...] | None = None,
    strategy: "ShardStrategy | None" = None,
    kv_specs: "Dict[int, Any] | None" = None,
    moe_execution: str | None = None,
    hw_budget: Any = None,
) -> Dict[str, Any]:
    """更新 GEMM 的设备提示并输出 IR 文件和代表 DPU 编号。

    `measure_kernel_tiles=True` 时，对每种本地分片形状真正跑一次算子编译，从
    pim mlir 里读出 `pim-tile-to-budget` 选定的输出分块，写进 sidecar 的
    `kernel_tile_n`。GeneSim 用它替掉 `conf/sim.yaml` 里拍下的 `tile_size`
    常量——这一项是「算子编译器影响 GeneSim 代价」的实际载体。

    默认关闭：它需要 CUDA、FlagTree 和 triton-opt，纯放置导出用不上。

    `dpu_to_cluster` 来自 GeneSim 给定的切分方案（`contracts/partition_plan.py`），
    原样写进 sidecar 顶层，让 GeneSim 按它把逻辑 DPU 落到指定的 ClusterPU 上，
    而不是走 `dpu_id % len(cluster_keys)` 的取模换算。往返用同一份声明，编号就不会错位。
    """
    ir = json.loads(Path(ir_path).read_text())
    num_layers = ir["num_layers"]
    if len(ir["subgraphs"]) != num_layers:
        raise ValueError(f"subgraphs 层数 {len(ir['subgraphs'])} 与 num_layers {num_layers} 不一致")
    if measure_kernel_tiles and hardware is None:
        from contracts.op_contract import default_hardware_config
        hardware = default_hardware_config()
    if hardware is not None and hw_budget is not None:
        # 容量与硬件配置必须同源：对齐或总容量各说一套时，pim mlir 的模块
        # 属性会和 sidecar 自报打架。
        if (hw_budget.mram_bytes != hardware.mram_bytes_per_dpu
                or hw_budget.align != hardware.dma_align):
            raise ValueError(
                f"hw_budget({hw_budget.mram_bytes}, align={hw_budget.align}) 与"
                f"硬件配置({hardware.mram_bytes_per_dpu}, align="
                f"{hardware.dma_align}) 不一致：请用 "
                "memory.mem_planner.default_site_capacity() 派生这两份值。")

    operators_by_id = {op["op_id"]: op for op in ir["operators"]}
    # 结构指纹必须与内容一致：IR 生成后被改过、或拿别的模型侧车复用，
    # 这里就失败。指纹随 sidecar 往返，GeneSim 生成 trace 前用同一算法重算比对。
    ir_fingerprint = assert_ir_fingerprint(ir, str(ir_path))
    # `source_ir` 只作人读线索，不能用来校验：它是绝对路径，换机器就失效，而
    # 仿真通常加载的是同一批 op_id 的另一个 IR（成本精化后的 *_pimir.ir）。
    # 真正可校验的是内容——算子总数、每个被放置算子的 op_type 与结构指纹，见
    # GeneSim 侧 _load_compiler_placement 的比对。
    sidecar: Dict[str, Any] = {
        "version": 3,
        "source_ir": str(ir_path),
        "ir_num_operators": len(ir["operators"]),
        # 模型身份（人读）与结构指纹（机器校验）。model_id 可能是仓库名或
        # 「模型类型:指纹前缀」，不依赖可移动的绝对路径。
        "model_id": str(ir.get("model_id", "")),
        "ir_fingerprint": ir_fingerprint,
        # 这份 sidecar 是否带算子编译产物（pimir_path / kernel_tile_n）。
        # GeneSim 据此自动进入严格模式：写了产物却读不到就报错，而不是静默退回
        # 手写模板。让产物自己声明，配置文件漏写 require_compiler_pimir 也不会
        # 悄悄退化。
        "requires_pimir": bool(measure_kernel_tiles),
        "operators": {},
    }
    if hw_budget is not None:
        # 容量契约随 sidecar 回传：GeneSim 用自己的 ClusterPU 容量与它核对，
        # 4/8/16 GiB 的分裂会在仿真启动前暴露，而不是各算各的。三件套
        # 一起写、一起校验，缺一项就不能进严格产物。
        sidecar["hw_budget"] = _hw_budget_payload(hw_budget)
    # MoE 成本契约随 sidecar 往返：GeneSim 加载时与 IR 的元数据核对，
    # 防止 128 专家被错配到 dense 侧车或反之。
    if int(ir.get("moe_num_experts", 0)) > 0:
        ir_execution = str(ir.get("moe_execution", ""))
        # 执行语义来自编译配置（CompiledModel.moe_execution），不是 IR 的自述：
        # 把编译期真正选定的模式写进 sidecar，GeneSim 再拿它与 IR 独立核对。
        if moe_execution is None:
            if not ir_execution:
                raise ValueError(
                    "MoE IR 没有 moe_execution，调用方也没有传编译配置："
                    "无法确定仿真按 dense_sim / dynamic_dispatch / "
                    "exact_dispatch 哪种语义计费。")
            moe_execution = ir_execution
        elif ir_execution and ir_execution != moe_execution:
            raise ValueError(
                f"图骨架 IR 的 moe_execution={ir_execution!r} 与本次编译配置 "
                f"{moe_execution!r} 不一致：IR 与主链不是同一次运行的产物。")
        sidecar["moe_config"] = {
            "num_experts": int(ir["moe_num_experts"]),
            "top_k": int(ir.get("moe_top_k", 0)),
            "intermediate_size": int(ir.get("moe_intermediate_size", 0)),
            "execution": moe_execution,
        }
    if dpu_to_cluster is not None:
        # 顶层字段而非 per-op：它描述的是"逻辑 DPU k 落在哪个 ClusterPU"，
        # 与算子无关。GeneSim 侧按它分配，缺失则退回取模换算。
        sidecar["dpu_to_cluster"] = [list(entry) for entry in dpu_to_cluster]

    # GeneSim 只展开了 all_reduce 的收集方向。all_gather 与 scatter 在它的图上
    # 没有边，通信时延被漏计；all_reduce 再写一遍会和归约节点双计，local_slice
    # 留在 DPU 内部、不产生通信。
    collectives = []
    seen_edges = set()
    export_seq_len = _graph_seq_len(gm)
    # MoE 专家子图的通信在动态分派下只发生在命中专家之间。给这些边带上
    # expert_id，GeneSim 才能按命中并集计通信量，而不是全专家都算一遍。
    moe_provenance = None
    if int(ir.get("moe_num_experts", 0)) > 0:
        from runtime.exec_plan_gen import _expert_provenance, _expert_tag_of
        moe_provenance = _expert_provenance(list(gm.graph.nodes))
        node_by_name = {node.name: node for node in gm.graph.nodes}
    for node in gm.graph.nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.edge_id in seen_edges or edge.type not in (
                "all_gather", "scatter",
            ):
                continue
            seen_edges.add(edge.edge_id)
            dpus = set()
            for loc in (edge.src_loc, edge.dst_loc):
                dpus.update(loc.get("dpus", ()))
            entry = {
                "type": edge.type,
                "nbytes": int(edge.nbytes),
                "dpu_ids": sorted(dpus),
            }
            # 键必须在：调度器把缺 seq_len 当成旧 sidecar 直接报错。不带序列
            # 维的边显式写 null，仿真侧按常量计，不会被错除。
            entry["seq_len"] = _export_seq_len(edge, export_seq_len)
            if moe_provenance is not None:
                src = node_by_name.get(edge.src)
                tag = _expert_tag_of(moe_provenance, src) if src else None
                if tag is not None:
                    entry["expert_id"] = tag[1]
            collectives.append(entry)
    sidecar["collectives"] = collectives
    # 融合清单只读图上的标记：没跑融合 pass 就是空，不能从 IR 形状猜。
    sidecar["fused_activations"] = _fused_activations(gm)

    _assert_every_gemm_has_known_role(ir)

    # {(local_in, local_out): tile_n}。同一形状只编一次——llama2 的 224 个 GEMM
    # 只有 7 种本地形状，逐个编会白跑 217 次。
    tile_cache: Dict[tuple[int, int, str | None], tuple[int, str, int | None]] = {}

    def export_gemm(op_id: int, layer: int) -> None:
        """把一个 GEMM 的放置结果写进 sidecar；模型级输出头传 layer=-1。"""
        role = operators_by_id[op_id]["semantic_role"]
        expert_id = int(operators_by_id[op_id].get("expert_id", -1))
        pattern = _weight_pattern(layer, role, expert_id)
        weight_node = _get_attr_node(gm, pattern)
        spec = weight_node.meta[SPEC_META_KEY]
        if spec.device != DEVICE_DPU:
            # 真实落点证据：只有权重确实在 host 上的稠密 GEMM 才能不产生
            # pimir。后续审计与 GeneSim 都按这条证据核对，不能再从
            # expert_id 之类的身份字段反推落点。
            if expert_id >= 0:
                raise ValueError(
                    f"op{op_id}（{role}，expert={expert_id}）的权重落在 "
                    "host 上：专家 GEMM 必须由某台 DPU 执行，不能借"
                    "「非 DPU GEMM」进入豁免。")
            if operators_by_id[op_id]["op_type"] != "GEMM":
                raise ValueError(
                    f"op{op_id} 的权重落在 host 上，但 IR 的算子类型是 "
                    f"{operators_by_id[op_id]['op_type']!r}，不是 GEMM。")
            operators_by_id[op_id]["device_hint"] = "cpu"
            # 落点证据单独存：host GEMM 没有 pimir/分片，不能混进
            # `operators` 里被分片统计、哈希核对与 GeneSim 钉死当成设备条目。
            sidecar.setdefault("host_gemms", {})[str(op_id)] = {
                "device_hint": "cpu",
                "op_type": "GEMM",
                "semantic_role": role,
                "weight": str(weight_node.target),
            }
            return

        # 之前这里只取 `min(spec.shard_map)` 当代表 DPU，TP 组内其余 shard
        # 被直接丢弃——GeneSim 因此只看到一个"本地代表"，既算不出组内并行占用
        # 的资源，也无从得知这个投影之后要不要归约。现在把 shard_map 的每个
        # DPU 都如实列出来，代表值只用于 IR 的全局形状字段（供只关心大小的
        # 消费者，如 ir_cost.py）。
        dpu_ids = sorted(spec.shard_map)
        rep_dpu_id = dpu_ids[0]
        # 权重的 local_shape 是 (out_features, in_features)，与这个 GEMM 在该
        # DPU 上要算的矩阵乘宽度一一对应——IR 里每个投影都是独立的 GEMM，
        # 不需要再累加或折算。
        rep_out, rep_in = spec.shard_map[rep_dpu_id].local_shape
        operators_by_id[op_id]["device_hint"] = "pim"
        # 本地分片形状写进 IR 的新字段，原 input/output_shapes 保持全局语义
        # 不动。GeneSim 的执行侧（compile_gemm 的 K/N、_execute_runtime 的
        # 循环次数）优先取这一份，切分才真正影响仿真出来的时间。
        #
        # 形状口径与全局字段一致：(Tq, in_features) → (Tq, out_features)，
        # 前导的 Tq 是符号维，原样搬过来不做解析。
        global_in = operators_by_id[op_id]["input_shapes"][0]
        global_out = operators_by_id[op_id]["output_shapes"][0]
        operators_by_id[op_id]["local_input_shapes"] = [
            [global_in[0], int(rep_in)]
        ]
        operators_by_id[op_id]["local_output_shapes"] = [
            [global_out[0], int(rep_out)]
        ]
        shards = [
            {
                "dpu_id": dpu_id,
                "local_in_features": int(spec.shard_map[dpu_id].local_shape[1]),
                "local_out_features": int(spec.shard_map[dpu_id].local_shape[0]),
            }
            for dpu_id in dpu_ids
        ]
        entry: Dict[str, Any] = {
            "device_hint": "pim",
            # 供消费侧核对 op_id 没有错位（IR 重新生成后编号可能变化）。
            "op_type": operators_by_id[op_id]["op_type"],
            # 这个投影切分到的每台 DPU 及其本地分片宽度；len==1 即纯 PP。
            "shards": shards,
            # 这个 GEMM 的投影身份，以及它实际匹配到的 fx 节点——排错时不用
            # 再猜「哪个 role 落到了哪个权重上」。
            # 无消费者，理由见 `DEBUG_ONLY_SIDECAR_FIELDS`。
            "semantic_role": role,
            "weight": str(weight_node.target),
        }
        if expert_id >= 0:
            entry["expert_id"] = expert_id
        if len(shards) > 1:
            # dim==0 切在输出特征维（列切，之后 concat/all_gather 即可）；
            # dim==1 切在输入特征维（行切，之后必须 all_reduce sum）。
            entry["shard_axis"] = (
                "output_channel" if spec.placement.dim == 0 else "input_channel"
            )
        sidecar["operators"][str(op_id)] = entry

        if measure_kernel_tiles:
            tile_n, pimir_path, placed = _measure_kernel_tile_n(
                int(rep_in), int(rep_out), dtype, hardware, tile_cache,
                activation=_activation_of(weight_node),
            )
            entry["kernel_tile_n"] = tile_n
            # 算子编译器回传的单台 MRAM 占用。GeneSim 用它核对常驻容量，
            # 没回传就不写，免得 sidecar 里多一个恒为 null 的键。
            if placed is not None:
                entry["placed_mram_bytes"] = placed
            # GeneSim 照这份 pim mlir 生成 PIM trace，而不是用手写模板。
            #
            # 这是导出这台机器上的绝对路径（算子编译缓存在仓库里）。换机器、
            # 换 checkout 目录、或缓存被清掉之后就失效，GeneSim 侧会按文件名
            # 在 `scheduler.pimir_search_dirs` 里再找一遍——文件名是算子编译
            # 缓存的内容哈希，同名即同一份产物。
            entry["pimir_path"] = pimir_path
            # 内容哈希一并记下：GeneSim 侧把它放进 trace 缓存签名，这样
            # 「同形状、同分块、但 pim mlir 换了」不会复用旧 trace。缓存里
            # 真实存在这样的碰撞（同为 out=2048 in=4096 tile_n=512，但
            # num-dpus/num-tasklets/dma-align 不同）。也便于换机器后核对
            # 找回来的文件是不是同一份。
            entry["pimir_sha256"] = _file_sha256(pimir_path)

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]
        for op_id in gemm_op_ids:
            export_gemm(op_id, layer)

    # 模型级输出头不挂在任何一层，逐层循环看不到它：漏掉就等于词表投影
    # 没有放置、没有 pimir、也没有 trace。它显式归到图尾所在段。
    in_subgraph = {
        op_id for group in ir["subgraphs"] for op_id in group
    }
    for op in ir["operators"]:
        if op["op_type"] != "GEMM" or op["op_id"] in in_subgraph:
            continue
        role = op.get("semantic_role") or ""
        if role != "lm_head":
            raise ValueError(
                f"op{op['op_id']} 是不属于任何层的 GEMM（role={role!r}）："
                "模型级 GEMM 只允许输出头 lm_head，其余投影必须归属某一层。")
        export_gemm(op["op_id"], -1)

    # 非 GEMM 的算子级节点也写 pimir_path：相位链与视图类的真实 IR 才能进
    # GeneSim 的 trace 编译。之前只有 GEMM 写，相位链那一支在生产里没人喂。
    if measure_kernel_tiles:
        _attach_bpath_pimir(operators_by_id, sidecar, ir, hardware)
        _assert_pimir_hardware_contract(sidecar, hardware)

    # 放在 B 路回填之后：非 GEMM 的条目是那里新建的，之前还没有可钉的对象。
    _pin_unsharded_ops(ir, sidecar, strategy)
    # 读 KV 缓存的注意力算子标成 kv 区。归属来自内存规划的 KV 规格：哪台
    # DPU 持有哪一层的 KV 是它定的，不是按算子类型硬猜。
    _mark_kv_zone(ir, sidecar, kv_specs)

    if measure_kernel_tiles:
        sidecar["pimir_exempt_ops"] = _build_pimir_exempt_ops(ir, sidecar)

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
