"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import (DEVICE_DPU, FUSED_TAIL_META_KEY, REDISTRIBUTE_META_KEY, SPEC_META_KEY)
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
}


def _activation_of(weight_node) -> str | None:
    """权重节点被折进主算子的激活名，没折过就是 None。

    标记打在 linear 节点上。分块后权重先经切片、类型转换才到 linear 子块，要穿过它们找。
    """
    for user in _consumers_through_slices(weight_node):
        tail = user.meta.get(FUSED_TAIL_META_KEY)
        if tail is not None:
            return tail.activation
    return None




def _compiler_identity() -> dict:
    """按需导入驱动层身份，避免放置导出模块反向依赖调优包。"""
    from opcompiler_bridge.driver import compiler_identity

    return compiler_identity()


def _graph_export_shape(gm: GraphModule) -> tuple[int, int]:
    """图导出用的输入 [batch, seq_len] 形状：字节的两个口径因子都来自这里。"""
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name == "input_ids":
            value = node.meta.get("val")
            if value is None:
                raise ValueError(
                    "input_ids 没有 val，无法确定导出形状"
                )
            shape = tuple(int(size) for size in value.shape)
            if len(shape) != 2:
                raise ValueError(
                    f"input_ids 必须是 [batch, seq_len] 二维输入，got {shape}"
                )
            return shape[0], shape[-1]
    raise ValueError("图上没有 input_ids，无法确定导出形状")


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


def _collective_number_errors(nbytes: int, seq_len: int | None) -> list[str]:
    """字节量与派生系数必须能被 GeneSim 的浮点成本模型表示。

    任意精度整数能通过正整数检查，但常量分支的 float() 会溢出；
    序列分支的 nbytes/seq_len 也可能溢出或下溢成 0。导出侧与
    GeneSim 的 `collective_edge_errors` 必须拒绝同一批数值。
    """
    try:
        nbytes_number = float(nbytes)
    except (OverflowError, ValueError):
        nbytes_number = None
    if nbytes_number is None or not math.isfinite(nbytes_number):
        return [f"nbytes 超出浮点可表示范围：{nbytes!r}"]
    if seq_len is None:
        return []
    try:
        seq_len_number = float(seq_len)
    except (OverflowError, ValueError):
        seq_len_number = None
    if seq_len_number is None or not math.isfinite(seq_len_number):
        return [f"seq_len 超出浮点可表示范围：{seq_len!r}"]
    coefficient = nbytes / seq_len
    if not math.isfinite(coefficient) or coefficient <= 0:
        return [f"nbytes/seq_len 必须是有限正数：{nbytes!r}/{seq_len!r}"]
    return []


def _collective_entry(edge, export_seq_len: int, export_batch: int) -> dict:
    """把一条重分布边写成 GeneSim 可消费的集合通信记录（单行字节口径）。

    字段契约与 GeneSim 的 `collective_edge_errors` 保持一致：nbytes 与
    dpu_ids 必须是正整数，字节量与派生系数还必须能被浮点成本模型表示，
    非法值不能靠 int() 静默截断或改写。
    """
    nbytes = edge.nbytes
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
        raise ValueError(
            f"重分布边 {edge.edge_id} 的 nbytes 必须是正整数，"
            f"收到 {nbytes!r}。")
    # 图按 [B_export, S] 导出：激活与 token id 的字节都乘过 batch 维。写出前
    # 除回单行口径，消费侧按本次 forward 的真实行数（Tq）计价，通信量才与
    # 编译档解耦（K6 根因）；掩码等广播形状首维是 1，不含 batch 因子，不除。
    if export_batch > 1 and edge.shape and int(edge.shape[0]) == export_batch:
        if nbytes % export_batch:
            raise ValueError(
                f"重分布边 {edge.edge_id} 的 nbytes {nbytes} 无法按导出 "
                f"batch={export_batch} 整除为单行口径")
        nbytes //= export_batch
    dpus = set()
    for loc in (edge.src_loc, edge.dst_loc):
        dpus.update(loc.get("dpus", ()))
    # 先做类型校验再排序：混入字符串时 sorted() 会先抛 TypeError，
    # 报错就不再指向具体字段。
    for dpu_id in dpus:
        if (
            isinstance(dpu_id, bool)
            or not isinstance(dpu_id, int)
            or dpu_id < 0
        ):
            raise ValueError(
                f"重分布边 {edge.edge_id} 的 dpu_ids 必须是非负的非布尔"
                f"整数，收到 {dpu_id!r}。")
    dpu_ids = sorted(dpus)
    if not dpu_ids:
        raise ValueError(
            f"重分布边 {edge.edge_id} 没有参与 DPU，"
            "无法生成集合通信记录。")
    seq_len = _export_seq_len(edge, export_seq_len)
    if seq_len is not None and (
        isinstance(seq_len, bool)
        or not isinstance(seq_len, int)
        or seq_len <= 0
    ):
        raise ValueError(
            f"重分布边 {edge.edge_id} 的 seq_len 必须是正整数或 None，"
            f"收到 {seq_len!r}。")
    number_errors = _collective_number_errors(nbytes, seq_len)
    if number_errors:
        raise ValueError(
            f"重分布边 {edge.edge_id} 的集合通信数值不合法："
            + "；".join(number_errors))
    return {
        "type": edge.type,
        "nbytes": nbytes,
        "dpu_ids": dpu_ids,
        "seq_len": seq_len,
    }


def _fused_activations(gm: GraphModule) -> list[dict]:
    """图上被折进主算子的激活，按层与投影身份列出。

    只认 `FUSED_TAIL_META_KEY`：权重的 linear 消费者带这个标记，才说明图编译器
    真的折过。模型 IR 里 gate 投影后接 silu 只是图的形状，不算数。
    """
    found = []
    for role, pattern in _ROLE_TO_WEIGHT_PATTERN.items():
        for node in gm.graph.nodes:
            if node.op != "get_attr" or pattern not in node.target:
                continue
            # 目标形如 model.model.layers.0.mlp.gate_proj.weight，层号在 layers 后。
            layer = int(str(node.target).split("layers.")[1].split(".")[0])
            # 分块后一块权重对应多个子块，激活相同，只记一条。
            activation = _activation_of(node)
            if activation is not None:
                found.append({
                    "layer": layer, "role": role, "activation": activation,
                })
    return found


def _get_attr_node(attr_nodes: list, pattern: str):
    """在预先收集的 get_attr 节点里按后缀找权重节点。

    `attr_nodes` 由调用方一次遍历图收集：每个 (层, 投影) 组合都调一次这里，
    现扫全图是 224 次全图遍历，先收一遍 get_attr 只扫几百个节点。
    """
    matches = [n for n in attr_nodes if pattern in n.target]
    if len(matches) != 1:
        raise ValueError(f"pattern={pattern!r} 应恰好匹配 1 个 get_attr 节点，实际 {len(matches)} 个")
    return matches[0]


def _measure_kernel_tile_n(
    local_in: int,
    local_out: int,
    dtype: str,
    hardware: "PIMHardwareConfig | None",
    cache: Dict[tuple[int, int, str | None, int], tuple[int, str, int | None, int]],
    activation: str | None = None,
    force_compile: bool = False,
    m_rows: int = 1,
) -> tuple[int, str, int | None, int]:
    """编译一个本地分片形状，返回 `(tile_n, pim mlir 路径, 单台占用, tile_m)`。

    这是算子编译器影响 GeneSim 代价的实际入口：`pim-tile-to-budget` 按 WRAM 预算
    搜出真实分块（llama2 的 4096 宽投影上是 512），而 GeneSim 原先只能用
    `conf/sim.yaml` 里拍下的 32。

    路径一并返回，GeneSim 侧照那份 pim mlir 生成 PIM trace——这一步才让编译器的
    真实 DMA 量和循环嵌套进入周期数，只喂一个分块常量是不够的（手写模板里
    `tile_size` 会约掉）。

    `m_rows` 是激活行数（batch 维合进行数后的 M）：decode 档逐 B 编 M=B
    （tile_m>1 时权重摊薄才真实生效），prefill 档编 M=S。同一形状同一 M 只编
    一次，结果放进 `cache`。第三个返回值是算子编译器回传的单台 MRAM 占用
    （`pim.placed-mram-bytes`），没回传时为 None。
    """
    # 激活也进键：同形状的两个投影一个折激活、一个不折，漏掉就编成同一份。
    key = (local_in, local_out, activation, m_rows)
    if key in cache:
        return cache[key]

    # 延迟导入：这条路径需要 CUDA 和 FlagTree，纯放置导出用不上，不该在 import
    # 期就把依赖拽进来。
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    hw = DEFAULT_HARDWARE_CONFIG if hardware is None else hardware
    request = OpCompileRequest(
        op="linear",
        arg_shapes=[(m_rows, local_in), (local_out, local_in)],
        hardware=hw,
        dtype=dtype,
        num_tasklets=hw.num_tasklets,
        # 激活随请求下发，由算子层 -pim-fuse-activation 折进 tt.dot。
        activation=activation,
    )
    result = compile_op(request, force=force_compile)
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
    cache[key] = (tile_n, pimir_path, cost.mram_bytes_per_dpu,
                  int(cost.tile_m or 1))
    return cache[key]


# B 路符号维的代表值：只要不是 1 这种极端退化值即可，128 和
# `op_classify.mnemonic_of` 探测配方时用的占位维度同一量级。
# 仿真侧按 `symbolic_dim_placeholder` 把静态形状缩放到实际请求形状。
_SYMBOLIC_DIM = 128


def _bpath_pimir(mnemonic: str, input_shapes, output_shapes,
                 hardware: "PIMHardwareConfig | None" = None,
                 *, force_compile: bool = False) -> str:
    """给一个算子级 mnemonic 编一份 B 路 pimir，返回落盘路径。

    与 `_measure_kernel_tile_n` 对称：那条只服务 GEMM（A 路），这条服务其余
    算子。路径进 sidecar 的 `pimir_path` 后，GeneSim 的 `_operator_pimir`
    才能把相位链与视图类的真实 IR 喂给 trace 编译，而不是永远拿到 A 路的
    `tt.dot`。

    形状里的符号维（`Tq`/`Tp`/`Tp+Tq`）按代表值 128 代入：trace 的外层循环
    次数由运行时绑定，IR 只需要一个具体的静态形状才能过 verifier，但折成
    字面 1 会让搬运量比真实 prefill（序列长度几百到几千）小几个数量级，
    128 只是一个不退化的占位，不是真实形状。

    硬件预算进缓存键：同一段 IR 在 4 KB 与 8 KB WRAM 下会选出不同分块，
    只按文本哈希会让第二个候选捡到第一个候选的产物。
    """
    from opcompiler_bridge.driver import (
        _CACHE_DIR, _compiler_fingerprint, _module_text,
        _run_oplevel_triton_opt)
    from opcompiler_bridge.oplevel_kernel import (
        concat_kernel, dynamic_quant_kernel, eltwise_kernel, gather_kernel, kv_cache_kernel,
        lut_kernel, mask_kernel, matmul_kernel, normalize_kernel,
        reshape_kernel, rope_kernel, softmax_kernel, split_heads_kernel,
        transpose_kernel,
    )
    from contracts.op_contract import (
        DEFAULT_HARDWARE_CONFIG, DpuShard, OpCompileRequest)
    from genesim_bridge.op_classify import _attention_matmul_body

    def n(dim):
        return _SYMBOLIC_DIM if isinstance(dim, str) else int(dim)

    ins = [tuple(n(d) for d in shape) for shape in input_shapes]
    outs = [tuple(n(d) for d in shape) for shape in output_shapes]

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
        "lut": lambda: lut_kernel("op", ins[0], "silu"),
        "eltwise": lambda: eltwise_kernel("op", ins[0], "add"),
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
    }
    body = builders[mnemonic]()
    if body is None:
        raise ValueError(f"{mnemonic} 需要单独的参数，不走这条通用入口")
    if mnemonic == "matmul" and "stationarity" not in body:
        raise RuntimeError(
            f"{mnemonic} 的 B 路 IR 缺 stationarity 标记，KV 缓存会被当成模型权值"
        )
    hw = DEFAULT_HARDWARE_CONFIG if hardware is None else hardware
    # B 路算子按复制档进入每个 DPU；模块头要写 DPU/tasklet，仿真才能证明
    # 这批 B 路产物吃的是本候选的编译参数，而不是另一套默认值。
    request = OpCompileRequest(
        op=mnemonic, arg_shapes=[tuple(shape) for shape in ins], hardware=hw,
        shard=DpuShard(dim=-1, num_dpus=hw.num_dpus, kind="replicate"))
    text = _module_text(body, request)
    key = hashlib.sha256(
        f"{text}:{hw.to_payload()}:{_compiler_fingerprint()}".encode()
    ).hexdigest()[:16]
    out = _CACHE_DIR / f"bpath-{key}.pimir.mlir"
    if force_compile or not out.is_file():
        expanded, _ = _run_oplevel_triton_opt(text, hw)
        # 不同的形状 key 可能编出相同 text（builder 只取部分维度），落到
        # 同一个内容寻址缓存文件；tmp 名带线程标识，避免并发写同一 out
        # 时一个线程 rename 走 tmp、另一个线程的 tmp 就此消失。
        tmp = out.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
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


def _pin_kv_read_ops(ir, sidecar, kv_specs) -> None:
    """执行侧 KV head 分片：KV 读取算子钉到持有该 head 的 DPU。

    IR 的 GEMV_SCORE/GEMV_CONTEXT 本就是逐 q_head 的独立链，每条链只读一个
    KV head（GQA 下多个 q_head 共享），所以「按 head 分片」落到算子级就是
    顶层 `dpu_id` 指到 head 持有者（kv_specs 的 q_heads_by_kv 给出归属），
    读取与 KV 常驻都落在布局实际所在的 DPU 上。缺省不跑本函数：维持整段
    首 DPU 钉扎，B=1 执行放置与旧产物逐位一致。
    """
    if not kv_specs:
        return
    q_head_owner = {}
    for spec in kv_specs.values():
        for kv_head, q_heads in spec.q_heads_by_kv.items():
            for q_head in q_heads:
                q_head_owner[q_head] = spec.dpu_id
    for op in ir["operators"]:
        if op["op_type"] not in _KV_OP_TYPES:
            continue
        if op.get("q_head_id", -1) not in q_head_owner:
            continue
        entry = sidecar["operators"].get(str(op["op_id"]))
        if entry is not None:
            entry["dpu_id"] = int(q_head_owner[op["q_head_id"]])


def _attach_bpath_pimir(operators_by_id, sidecar, hardware=None,
                        *, force_compile: bool = False) -> None:
    """给每个算子级节点编一份 B 路 pimir 并写进 sidecar。

    同一 (mnemonic, 形状, 硬件) 只编一次，唯一键并行编译（各自走独立的
    内容寻址缓存文件，互不影响）。编不出来直接抛：静默跳过的那条条目会缺
    `pimir_path`，仿真侧退回手写模板，而现场只看到「缺路径」这个结果，
    看不到是谁、为什么编不出。契约不满足就让原因立刻暴露。
    """
    # 第一遍收集唯一键，并记下首次出现的算子用于报错定位。
    keys: list = []
    first_of: dict = {}
    key_of_op: dict = {}
    for op_id, op in operators_by_id.items():
        mnemonic = _BPATH_MNEMONIC.get(op["op_type"])
        if mnemonic is None:
            continue
        key = (mnemonic, json.dumps(op["input_shapes"]),
               json.dumps(op["output_shapes"]))
        key_of_op[op_id] = key
        if key not in first_of:
            first_of[key] = (op_id, op)
            keys.append(key)

    def compile_key(key):
        mnemonic = key[0]
        op_id, op = first_of[key]
        try:
            return _bpath_pimir(
                mnemonic, op["input_shapes"], op["output_shapes"],
                hardware, force_compile=force_compile)
        except Exception as exc:
            raise RuntimeError(
                f"op{op_id}（{op['op_type']} -> {mnemonic}，形状 "
                f"{op['input_shapes']} -> {op['output_shapes']}）没编出 "
                f"pim mlir：{exc}。缺了它 GeneSim 会静默退回手写模板，"
                "这个算子的原语等于没进仿真。"
            ) from exc

    from opcompiler_bridge.parallel import run_jobs

    paths = run_jobs(compile_key, keys)
    cache = dict(zip(keys, paths))
    # sha256 按路径去重：同一条 pimir 被上千个算子引用，逐个重读是纯浪费。
    sha_of: dict = {}
    for op_id, op in operators_by_id.items():
        if op_id not in key_of_op:
            continue
        path = cache[key_of_op[op_id]]
        if path not in sha_of:
            sha_of[path] = _file_sha256(path)
        # `shards` 必须在：消费方逐条取 `entry["shards"]` 统计分片数，缺了就抛。
        # B 路节点不切分，所以是空列表，而不是不写这个键。
        entry = sidecar["operators"].setdefault(str(op_id), {
            "op_type": op["op_type"], "device_hint": "pim", "shards": [],
        })
        entry["pimir_path"] = path
        entry["pimir_sha256"] = sha_of[path]
        # 算子级节点走 B 路：只出现 MRAM/WRAM/DMA/目标，DPU/tasklet 编码
        # 不在这条路径上；消费侧按路由决定哪些属性必须存在。
        entry["pimir_route"] = "b"
        # 符号维在 PIM MLIR 里按固定占位值编译；仿真侧据此把整算子轨迹的
        # 静态工作量缩放到本次请求形状，避免运行时再按 Tq 重复一次。
        entry["symbolic_dim_placeholder"] = {
            str(dimension): _SYMBOLIC_DIM
            for shape in (*op["input_shapes"], *op["output_shapes"])
            for dimension in shape
            if isinstance(dimension, str)
        }


# 读 KV 缓存的算子：attention 的两次矩阵乘，一次读 K、一次读 V。
_KV_OP_TYPES = ("GEMV_SCORE", "GEMV_CONTEXT")


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


def _assert_every_gemm_has_known_role(ir: Dict[str, Any]) -> None:
    """核对 IR 里每个 GEMM 都带一个本表认得的 `semantic_role`。

    正向校验：从 IR 出发查表，而不是从表出发查 IR。上游新增一种投影、或者改了
    role 的写法，都会在这里报错并指出具体算子，而不是等到取 fx 节点时才撞上一句
    「应恰好匹配 1 个」——那个报错指向的位置是错的。
    """
    missing: list[int] = []
    unknown: Dict[int, str] = {}
    for op in ir["operators"]:
        if op["op_type"] != "GEMM":
            continue
        role = op.get("semantic_role") or ""
        if not role:
            missing.append(op["op_id"])
        elif role not in _ROLE_TO_WEIGHT_PATTERN:
            unknown[op["op_id"]] = role

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


def export_placement_to_genesim(
    gm: GraphModule,
    ir_path: Path,
    out_ir_path: Path,
    sidecar_path: Path,
    *,
    dtype: str = "float16",
    hardware: "PIMHardwareConfig | None" = None,
    sys_reserve_bytes: int = 0,
    measure_kernel_tiles: bool = False,
    dpu_to_cluster: tuple[tuple[int, int], ...] | None = None,
    strategy: "ShardStrategy | None" = None,
    kv_specs: "Dict[int, Any] | None" = None,
    profile=None,
    force_compile: bool = False,
    batch_sizes: tuple[int, ...] = (1,),
    capacity: Dict[str, Any] | None = None,
    kv_layout: Dict[str, Any] | None = None,
    dp_replicas: Dict[str, Any] | None = None,
    kv_head_pin: bool = False,
    per_slot_prefill: bool = False,
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

    `sys_reserve_bytes` 与 `hardware.mram_bytes_per_dpu` 一起写进顶层的
    `hardware` 字段：GeneSim 的 CEVA 严格模式要求这份预算与
    `ceva.local_memory_bytes` / `ceva.sys_reserve_bytes` 逐项一致。
    """
    if hardware is None:
        from contracts.op_contract import DEFAULT_HARDWARE_CONFIG
        hardware = DEFAULT_HARDWARE_CONFIG
    from contracts.op_contract import PimOptimizations as _PimOptimizations
    if (
        isinstance(sys_reserve_bytes, bool)
        or not isinstance(sys_reserve_bytes, int)
        or sys_reserve_bytes < 0
        or sys_reserve_bytes >= hardware.mram_bytes_per_dpu
    ):
        raise ValueError(
            "sys_reserve_bytes 必须是非负整数且小于硬件 MRAM 预算："
            f"{sys_reserve_bytes!r} vs {hardware.mram_bytes_per_dpu}。")
    if not batch_sizes or any(
        isinstance(b, bool) or not isinstance(b, int) or b <= 0
        for b in batch_sizes
    ):
        raise ValueError(f"batch_sizes 必须是非空正整数列表：{batch_sizes!r}")
    ir = json.loads(Path(ir_path).read_text())
    num_layers = ir["num_layers"]
    if len(ir["subgraphs"]) != num_layers:
        raise ValueError(f"subgraphs 层数 {len(ir['subgraphs'])} 与 num_layers {num_layers} 不一致")

    operators_by_id = {op["op_id"]: op for op in ir["operators"]}
    # `source_ir` 只作人读线索，不能用来校验：它是绝对路径，换机器就失效，而
    # 仿真通常加载的是同一批 op_id 的另一个 IR（成本精化后的 *_pimir.ir）。
    # 真正可校验的是内容——算子总数和每个被放置算子的 op_type，见
    # GeneSim 侧 _load_compiler_placement 的比对。
    sidecar: Dict[str, Any] = {
        # schema 5 起必须带完整编译契约：toolchain、hardware、strategy 与
        # 每个 PIM MLIR 的 A/B 编译路由，消费侧按版本失败关闭；6 起
        # collectives[].nbytes 为单行口径（不携带编译档 batch 维）。
        "version": 6,
        "source_ir": str(ir_path),
        "ir_num_operators": len(ir["operators"]),
        # 与编译本地形状时下发的 PIMHardwareConfig 同源：物理总量、系统预留
        # 和可用量必须能逐项对账，否则产物可能来自另一套字节预算。
        "hardware": {
            "mram_bytes_per_dpu": int(hardware.mram_bytes_per_dpu),
            "sys_reserve_bytes": int(sys_reserve_bytes),
            "usable_mram_bytes": (
                int(hardware.mram_bytes_per_dpu)
                - int(sys_reserve_bytes)
            ),
        },
        # 这份 sidecar 是否带算子编译产物（pimir_path / kernel_tile_n）。
        # GeneSim 据此自动进入严格模式：写了产物却读不到就报错，而不是静默退回
        # 手写模板。让产物自己声明，配置文件漏写 require_compiler_pimir 也不会
        # 悄悄退化。
        "requires_pimir": bool(measure_kernel_tiles),
        "operators": {},
    }
    if capacity is not None:
        # 编译期容量表（可读文本 + 逐 DPU 折算 + 墙位）：扫参报告的
        # 「容量占用」列与 DP 解析估算都从这份产物取输入。
        sidecar["capacity"] = capacity
    if kv_layout:
        # KV 物理布局载荷：逐 DPU 的 head 归属、槽数与已分配字节。genesim
        # 的容量校验据此把 KV 常驻摊到真正持有 KV head 的 TensorPU。
        sidecar["kv_layout"] = kv_layout
    if dp_replicas is not None:
        # DP 副本载荷（R8）：声明副本块与每副本权重复制量，genesim 据此把
        # 不同组路由到不同副本的物理 DPU 组。replicas=1 时不写（等价现状）。
        sidecar["dp_replicas"] = dp_replicas
    if max(batch_sizes) > 1:
        # 执行策略声明（R25-03）：B>1 的导出显式写 true/false，genesim 据此
        # 与自身执行策略互检（合批消费逐槽产物是保守方向，放行；逐槽消费
        # 合批产物拒绝）。B=1 无逐槽对象，不写该键——经典单档产物逐位不变。
        sidecar["per_slot_prefill"] = bool(per_slot_prefill)
    if dpu_to_cluster is not None:
        # 顶层字段而非 per-op：它描述的是"逻辑 DPU k 落在哪个 ClusterPU"，
        # 与算子无关。GeneSim 侧按它分配，缺失则退回取模换算。
        sidecar["dpu_to_cluster"] = [list(entry) for entry in dpu_to_cluster]
    # 编译口径的可回读副本：策略、五元组与三档容量，加上编译字段摘要。
    # 时钟等只影响仿真的字段不进这个摘要，扫时钟不需要假装重编过。
    sidecar["compiler"] = {
        "strategy": None if strategy is None else {
            "name": strategy.name,
            "num_dpus": strategy.num_dpus,
            "num_stages": strategy.num_stages,
            "tp_width": strategy.tp_width,
            "replicas": strategy.replicas,
            "total_dpus": strategy.total_dpus,
            "dpu_ids": list(strategy.dpu_ids or ()),
        },
        "hardware": {
            "num_dpus": hardware.num_dpus,
            "num_tasklets": hardware.num_tasklets,
            "mram_bytes_per_dpu": int(hardware.mram_bytes_per_dpu),
            "wram_bytes_per_dpu": int(hardware.wram_bytes_per_dpu),
            "dma_align": int(hardware.dma_align),
            # 编译目标与五元组同属请求级契约，仿真前按画像逐项核对。
            "compiler_target": hardware.compiler_target,
            # 优化开关同属编译请求：仿真侧据此证明消费的是同一批 pass 产物。
            "optimizations": getattr(
                hardware, "optimizations", _PimOptimizations()).to_payload(),
        },
        "compile_profile_sha256": (
            None if profile is None else profile.compile_profile_sha256()
        ),
        "physical_mram_bytes": int(hardware.mram_bytes_per_dpu),
        "reserved_mram_bytes": int(sys_reserve_bytes),
        "usable_mram_bytes": (
            int(hardware.mram_bytes_per_dpu) - int(sys_reserve_bytes)
        ),
        # 编译源码与实际工具的内容身份；GeneSim 只消费这份显式契约，
        # 不再从 PIM MLIR 的物理路径反推源码仓库。
        "toolchain": _compiler_identity(),
    }

    # GeneSim 只展开了 all_reduce 的收集方向。all_gather 与 scatter 在它的图上
    # 没有边，通信时延被漏计；all_reduce 再写一遍会和归约节点双计，local_slice
    # 留在 DPU 内部、不产生通信。
    collectives = []
    seen_edges = set()
    export_batch, export_seq_len = _graph_export_shape(gm)
    for node in gm.graph.nodes:
        for edge in node.meta.get(REDISTRIBUTE_META_KEY, []):
            if edge.edge_id in seen_edges or edge.type not in (
                "all_gather", "scatter",
            ):
                continue
            seen_edges.add(edge.edge_id)
            # 键必须在：调度器把缺 seq_len 当成旧 sidecar 直接报错。不带序列
            # 维的边显式写 null，仿真侧按常量计，不会被错除。
            collectives.append(_collective_entry(edge, export_seq_len, export_batch))
    sidecar["collectives"] = collectives
    # 融合清单只读图上的标记：没跑融合 pass 就是空，不能从 IR 形状猜。
    sidecar["fused_activations"] = _fused_activations(gm)

    _assert_every_gemm_has_known_role(ir)

    # {(local_in, local_out, activation, m_rows): (tile_n, path, placed, tile_m)}。
    # 同一形状同一 M 只编一次——llama2 的 224 个 GEMM 只有 7 种本地形状。
    tile_cache: Dict[tuple[int, int, str | None, int], tuple[int, str, int | None, int]] = {}
    # 测量延后到循环外并行做：这里只收集 (entry, 本地形状, 激活)。
    gemm_measurements: list[tuple[Dict[str, Any], int, int, str | None]] = []
    # 权重节点先收一遍：每个 (层, 投影) 找一次节点，现扫全图是 224 次全图遍历。
    attr_nodes = [n for n in gm.graph.nodes if n.op == "get_attr"]

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]

        for op_id in gemm_op_ids:
            role = operators_by_id[op_id]["semantic_role"]
            pattern = _ROLE_TO_WEIGHT_PATTERN[role]
            weight_node = _get_attr_node(attr_nodes, f"layers.{layer}.{pattern}")
            spec = weight_node.meta[SPEC_META_KEY]
            if spec.device != DEVICE_DPU:
                continue  # host 权重不写入 DPU 设备提示。

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
            if len(shards) > 1:
                # dim==0 切在输出特征维（列切，之后 concat/all_gather 即可）；
                # dim==1 切在输入特征维（行切，之后必须 all_reduce sum）。
                entry["shard_axis"] = (
                    "output_channel" if spec.placement.dim == 0 else "input_channel"
                )
            sidecar["operators"][str(op_id)] = entry

            if measure_kernel_tiles:
                gemm_measurements.append(
                    (entry, int(rep_in), int(rep_out),
                     _activation_of(weight_node))
                )

    # 唯一 (本地形状, 激活, 行数) 键并行编译：各键走 compile_op 的独立缓存文件，
    # 互不影响；主循环里逐个编是纯串行的子进程等待。激活可 None 可 str，
    # 不能排序去重，按首次出现顺序保持确定性。
    if gemm_measurements:
        # 相位产物表：decode 档逐 B（B=1 恒有一份，供顶层旧字段做同值副本）；
        # prefill 逐 B 合批（M=B×S）——与图编译/编排器一次执行 [B,S] 的真实
        # 语义一致，GEMM 权重装载份数是 ceil(B×S/tile_m)，S 非 tile_m 整数时
        # 严格少于逐槽口径 B×ceil(S/tile_m)（真实编译实测，R17-01）。
        # "prefill" 键为 prefill_b1 的同值副本（M=S），服务逐槽 micro-batch
        # 策略与旧 sidecar 的单槽消费方。
        phase_rows = [(f"decode_b{b}", b) for b in sorted({1, *batch_sizes})]
        phase_rows.extend(
            (f"prefill_b{b}", b * export_seq_len)
            for b in sorted({1, *batch_sizes}))
        phase_rows.append(("prefill", export_seq_len))
        unique_keys = list(dict.fromkeys(
            (in_f, out_f, act, m_rows)
            for _, in_f, out_f, act in gemm_measurements
            for _, m_rows in phase_rows))
        from opcompiler_bridge.parallel import run_jobs

        run_jobs(
            lambda key: _measure_kernel_tile_n(
                key[0], key[1], dtype, hardware, tile_cache,
                activation=key[2], force_compile=force_compile, m_rows=key[3]),
            unique_keys,
        )
        sha_of: Dict[str, str] = {}
        for entry, rep_in, rep_out, activation in gemm_measurements:
            by_phase: Dict[str, Dict[str, Any]] = {}
            for phase, m_rows in phase_rows:
                tile_n, pimir_path, _placed, tile_m = tile_cache[
                    (rep_in, rep_out, activation, m_rows)]
                if pimir_path not in sha_of:
                    sha_of[pimir_path] = _file_sha256(pimir_path)
                by_phase[phase] = {
                    "path": pimir_path,
                    "sha256": sha_of[pimir_path],
                    "kernel_tile_n": tile_n,
                    "tile_m": tile_m,
                }
            entry["pimir_by_phase"] = by_phase
            # 顶层旧字段是 decode_b1（M=1，改动前的唯一口径）的同值副本，
            # 旧消费方（run_full_pipeline 校验、GeneSim 台账）零改动。
            # path 是导出机的绝对路径（算子编译缓存在仓库里）；换机器后
            # GeneSim 按 `scheduler.pimir_search_dirs` 里的文件名（内容哈希）
            # 找回同一份产物。sha256 进 trace 缓存签名，防「同形状同分块但
            # pim mlir 换了」复用旧 trace。
            decode_b1 = by_phase["decode_b1"]
            entry["kernel_tile_n"] = decode_b1["kernel_tile_n"]
            entry["pimir_path"] = decode_b1["path"]
            entry["pimir_sha256"] = decode_b1["sha256"]
            entry["pimir_route"] = "a"
            # 算子编译器回传的单台 MRAM 占用（decode 常驻口径）。GeneSim 用它
            # 核对常驻容量，没回传就不写，免得 sidecar 里多一个恒为 null 的键。
            _, _, placed, _ = tile_cache[(rep_in, rep_out, activation, 1)]
            if placed is not None:
                entry["placed_mram_bytes"] = placed

    # 非 GEMM 的算子级节点也写 pimir_path：相位链与视图类的真实 IR 才能进
    # GeneSim 的 trace 编译。之前只有 GEMM 写，相位链那一支在生产里没人喂。
    if measure_kernel_tiles:
        _attach_bpath_pimir(operators_by_id, sidecar, hardware,
                            force_compile=force_compile)

    # 放在 B 路回填之后：非 GEMM 的条目是那里新建的，之前还没有可钉的对象。
    # KV head 钉扎先于整段钉扎执行：写过 dpu_id 的条目后者会跳过。
    if kv_head_pin:
        _pin_kv_read_ops(ir, sidecar, kv_specs)
    _pin_unsharded_ops(ir, sidecar, strategy)
    # 读 KV 缓存的注意力算子标成 kv 区。归属来自内存规划的 KV 规格：哪台
    # DPU 持有哪一层的 KV 是它定的，不是按算子类型硬猜。
    _mark_kv_zone(ir, sidecar, kv_specs)

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar


@dataclass(frozen=True)
class PlacementExportResult:
    """一个候选的图/算子编译产物，全部落在同一个目录里。"""

    out_dir: Path
    placed_ir: Path
    sidecar: Path
    partition_plan: Path
    pimir_paths: tuple[Path, ...]
    # 主机侧编译结果（R25-03）：含逐槽 prefill 计划（per_slot_prefill 元组，
    # None = 本次导出未编译逐槽），主机执行器直接消费这份交付。
    compiled: "object | None" = None


def validate_tensor_pu_capacity(profile, compiled,
                                *, concurrency: int | None = None) -> int:
    """按逐张量分配台账核验每个 TensorPU 的常驻容量。

    同一簇的多个 TPU 可以共享整簇容量，但每笔常驻分配只能落在一个 TPU：
    按初始放置顺序轮转分组后，同 PU 的张量按身份去重模型常驻，逐请求
    峰值按请求下标相加后取最大的 concurrency 个。两块各自合法但合计
    超限必须在图编译后立即拒绝，不能等 GeneSim 完成执行图构造才发现。
    返回本图最大的单 PU 常驻字节数。
    """
    from memory.mem_planner import tensor_pu_usage

    plans = getattr(compiled, "mem_plans", None) or {}
    maximum = 0
    for dpu_id, plan in sorted(plans.items()):
        allocations = getattr(plan, "resident_allocations", ())
        usage = tensor_pu_usage(
            allocations, profile.tensors_per_cluster,
            concurrency=concurrency)
        for slot, stats in sorted(usage.items()):
            if stats["total_bytes"] > profile.tensor_pu_memory_bytes:
                raise ValueError(
                    f"dpu{dpu_id} 的 TensorPU {slot} 常驻 {stats['total_bytes']} B"
                    f"（模型 {stats['model_bytes']} + 请求 "
                    f"{stats['request_bytes']}），超过单个 TensorPU 容量 "
                    f"{profile.tensor_pu_memory_bytes} B")
            maximum = max(maximum, stats["total_bytes"])
    return maximum


def export_placement_for_strategy(
    model,
    ir_path: Path,
    out_dir: Path,
    *,
    strategy,
    plan=None,
    hw,
    hardware,
    prefill_seq_len: int,
    max_seq: int,
    profile=None,
    force_compile: bool = False,
    measure_kernel_tiles: bool = True,
    dtype: str = "float16",
    kv_dtype_bytes: int = 2,
    file_stem: str | None = None,
    request_kv_tokens: tuple[int, ...] | None = None,
    max_concurrent_requests: int | None = None,
    batch_sizes: tuple[int, ...] = (1,),
    kv_head_pin: bool = False,
    per_slot_prefill: bool = False,
) -> PlacementExportResult:
    """按候选编译图与算子，产出一份完整放置目录。

    这是调优器与旧单次链路共用的唯一图编译入口：内部仍调用
    `runtime.compile.compile_llama2`，不另写一条只服务调优的路径。
    调用方负责先把 `out_dir` 建成临时目录，校验完整后再原子发布。

    `plan` 给了就按方案编译并原样落盘；没给时按 `strategy` 合成一份同源方案，
    保证产物目录里始终能读到「实际使用的切分」。`batch_sizes` 是 batch 扫参
    档列表：图按最大档编译（KV/内存按最坏档校验），算子按相位逐档出产物。
    """
    from contracts.partition_plan import PartitionPlan
    from graph.fuse_pim import fuse_for_pim
    from runtime.compile import compile_llama2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if plan is None:
        plan = PartitionPlan(
            num_dpus=strategy.num_dpus,
            num_stages=strategy.num_stages,
            weight_rules=tuple(strategy.weight_rules),
            dpu_ids=(tuple(strategy.dpu_ids) if strategy.dpu_ids else None),
            replicas=strategy.replicas,
            source="tune_strategy",
        )
    plan_path = out_dir / (f"{file_stem}_plan.json" if file_stem
                           else "partition_plan.json")
    plan.write(plan_path)

    compiled = compile_llama2(
        model, strategy, prefill_seq_len=prefill_seq_len, max_seq=max_seq,
        hw=hw, hardware=hardware, kv_dtype_bytes=kv_dtype_bytes,
        force_compile=force_compile,
        request_kv_tokens=request_kv_tokens,
        batch_size=max(batch_sizes),
        per_slot_prefill=per_slot_prefill,
    )
    # 多 TensorPU 的簇按整簇容量共享 MRAM，但单块张量只能落在一个 TensorPU。
    # 请求峰值按本轮轨迹逐请求登记，并发上限与仿真同一配置。
    if profile is not None:
        validate_tensor_pu_capacity(
            profile, compiled, concurrency=max_concurrent_requests)
    fuse_for_pim(compiled.prefill_gm)
    placed_ir = out_dir / (f"{file_stem}_placed.ir" if file_stem
                           else "placed.ir")
    sidecar_path = out_dir / (f"{file_stem}_placement.json" if file_stem
                              else "placement.json")
    # 容量表随产物落盘：图按最大档编译，slots 记编译档，逐 DPU 的可变字节
    # （KV+激活）按线性折算给出 batch 墙位。max_seq 一并记录——容量表的
    # 序列长度口径必须可被消费方核对，否则墙位对不上真实 trace。
    from memory.mem_planner import capacity_batch_fit, format_capacity_table

    capacity = dict(
        capacity_batch_fit(compiled.mem_plans, hw),
        slots=max(batch_sizes),
        max_seq=int(max_seq),
        table=format_capacity_table(compiled.mem_plans, hw),
    )
    if strategy.replicas > 1:
        # 容量表按副本口径（每副本 N 台 DPU 的折算），记录副本数供消费方
        # 区分：物理机上每副本只服务 1/G 的请求流。
        capacity["replicas"] = strategy.replicas
    # KV 物理布局载荷（顶层 kv_layout）：逐 DPU 的 head 归属、槽数与已分配
    # 字节。genesim 的容量校验据此把 KV 常驻摊到真正持有 KV head 的
    # TensorPU，两仓台账可逐 DPU 对账；无 KV 规格时不写该键。
    # DP 副本 >1 时键按物理 DPU 落（副本 g 偏移 g×N）：每副本一份完整布局，
    # 各服务本副本请求流；replicas=1 时键就是逻辑编号，与旧产物逐位一致。
    def _kv_entry(spec) -> Dict[str, Any]:
        return {
            "kv_heads": list(spec.kv_heads),
            "layers": list(spec.layers),
            "num_slots": int(spec.num_slots),
            "kv_allocated_bytes": int(spec.kv_allocated_bytes),
        }

    kv_layout = ({
        str(physical): _kv_entry(spec)
        for replica in range(strategy.replicas)
        for spec in compiled.kv_specs.values()
        for physical in (spec.dpu_id + replica * strategy.num_dpus,)
    } if compiled.kv_specs else None)
    # DP 副本载荷：逐副本物理块与每副本静态复制量。weight_bytes_per_replica
    # 是纯权重字节（各 DPU 权重区合计，= kv_base）；内核 tile 是算子编译器
    # 回传的瞬时占用、不是权重复制的一部分，单列 kernel_tile_bytes_per_
    # replica（R17-04：静态副本字节与动态权重流量分开，纯权重字段不得
    # 混入 tile）。replicas=1 不写——sidecar 与旧产物逐位一致。
    dp_replicas = None
    if strategy.replicas > 1:
        weight_total = 0
        kernel_tile_total = 0
        for stats in capacity["dpus"].values():
            weight_total += int(stats["fixed_bytes"]) - int(
                stats["kernel_tile_bytes"])
            kernel_tile_total += int(stats["kernel_tile_bytes"])
        dp_replicas = {
            "replicas": strategy.replicas,
            "dpus_per_replica": strategy.num_dpus,
            "replica_dpu_ids": [
                list(strategy.dpus_of_replica(replica))
                for replica in range(strategy.replicas)
            ],
            "weight_bytes_per_replica": weight_total,
            "kernel_tile_bytes_per_replica": kernel_tile_total,
        }
    sidecar = export_placement_to_genesim(
        compiled.prefill_gm, Path(ir_path), placed_ir, sidecar_path,
        dtype=dtype,
        hardware=hardware,
        sys_reserve_bytes=hw.sys_reserve_bytes,
        measure_kernel_tiles=measure_kernel_tiles,
        dpu_to_cluster=plan.dpu_to_cluster,
        strategy=strategy,
        kv_specs=compiled.kv_specs,
        profile=profile,
        force_compile=force_compile,
        batch_sizes=batch_sizes,
        capacity=capacity,
        kv_layout=kv_layout,
        dp_replicas=dp_replicas,
        kv_head_pin=kv_head_pin,
        per_slot_prefill=compiled.per_slot_prefill is not None,
    )
    # 产物路径聚合覆盖全部相位档：pimir_by_phase 的每一份都要进结果，
    # 调优指纹才不会漏掉逐档产物。
    pimir_paths = tuple(sorted({
        Path(phase["path"])
        for entry in sidecar["operators"].values()
        for phase in (entry.get("pimir_by_phase") or {}).values()
    } | {
        Path(entry["pimir_path"])
        for entry in sidecar["operators"].values()
        if entry.get("pimir_path")
    }))
    return PlacementExportResult(
        out_dir=out_dir,
        placed_ir=placed_ir,
        sidecar=sidecar_path,
        partition_plan=plan_path,
        pimir_paths=pimir_paths,
        compiled=compiled,
    )
