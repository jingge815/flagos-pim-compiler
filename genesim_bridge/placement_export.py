"""将图编译器的 GEMM 放置结果写入 GeneSim IR 和附加数据文件。"""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from torch.fx import GraphModule

from contracts.graph_meta import DEVICE_DPU, SPEC_META_KEY

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
_ROLE_TO_WEIGHT_PATTERN = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


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
    cache: Dict[tuple[int, int], tuple[int, str]],
) -> tuple[int, str]:
    """编译一个本地分片形状，返回 `(tile_n, pim mlir 路径)`。

    这是算子编译器影响 GeneSim 代价的实际入口：`pim-tile-to-budget` 按 WRAM 预算
    搜出真实分块（llama2 的 4096 宽投影上是 512），而 GeneSim 原先只能用
    `conf/sim.yaml` 里拍下的 32。

    路径一并返回，GeneSim 侧照那份 pim mlir 生成 PIM trace——这一步才让编译器的
    真实 DMA 量和循环嵌套进入周期数，只喂一个分块常量是不够的（手写模板里
    `tile_size` 会约掉）。

    同一形状只编一次，结果放进 `cache`。
    """
    key = (local_in, local_out)
    if key in cache:
        return cache[key]

    # 延迟导入：这条路径需要 CUDA 和 FlagTree，纯放置导出用不上，不该在 import
    # 期就把依赖拽进来。
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.ir_cost import analyze_ir
    from opcompiler_bridge.driver import compile_op

    hw = DEFAULT_HARDWARE_CONFIG if hardware is None else hardware
    # M=1 是 decode 口径：算子编译器这条链只支持 M 方向不分块，而分块的搜索只由
    # K/N 和 WRAM 预算决定，与 M 无关。
    request = OpCompileRequest(
        op="linear",
        arg_shapes=[(1, local_in), (local_out, local_in)],
        hardware=hw,
        dtype=dtype,
        num_tasklets=hw.num_tasklets,
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
    cache[key] = (tile_n, str(result.pimir_path))
    return cache[key]


def _bpath_pimir(mnemonic: str, input_shapes, output_shapes) -> str:
    """给一个算子级 mnemonic 编一份 B 路 pimir，返回落盘路径。

    与 `_measure_kernel_tile_n` 对称：那条只服务 GEMM（A 路），这条服务其余
    算子。路径进 sidecar 的 `pimir_path` 后，GeneSim 的 `_operator_pimir`
    才能把相位链与视图类的真实 IR 喂给 trace 编译，而不是永远拿到 A 路的
    `tt.dot`。

    形状里的符号维（`Tq`/`Tp`/`Tp+Tq`）按代表值 128 代入：trace 的外层循环
    次数由运行时绑定，IR 只需要一个具体的静态形状才能过 verifier，但折成
    字面 1 会让搬运量比真实 prefill（序列长度几百到几千）小几个数量级，
    128 只是一个不退化的占位，不是真实形状。
    """
    from opcompiler_bridge.driver import compile_op
    from opcompiler_bridge.oplevel_kernel import (
        concat_kernel, dynamic_quant_kernel, eltwise_kernel, gather_kernel, kv_cache_kernel,
        lut_kernel, mask_kernel, matmul_kernel, normalize_kernel,
        reshape_kernel, rope_kernel, softmax_kernel, split_heads_kernel,
        transpose_kernel,
    )
    from contracts.op_contract import DEFAULT_HARDWARE_CONFIG, OpCompileRequest
    from genesim_bridge.op_classify import _attention_matmul_body, _module

    # 符号维的代表值：只要不是 1 这种极端退化值即可，128 和
    # `op_classify.mnemonic_of` 探测配方时用的占位维度同一量级。
    _SYMBOLIC_DIM = 128

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
    from opcompiler_bridge.driver import _run_oplevel_triton_opt, _CACHE_DIR
    import hashlib, os
    text = _module(body)
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    out = _CACHE_DIR / f"bpath-{key}.pimir.mlir"
    if not out.is_file():
        expanded, _ = _run_oplevel_triton_opt(text)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(expanded)
        os.replace(tmp, out)
    return str(out)



# 算子类型到 mnemonic 的映射与成本抽取共用同一份（`op_classify.MNEMONIC_OF`）。
# GEMM 走 A 路的分块编译，不在这里。
from genesim_bridge.op_classify import MNEMONIC_OF

_BPATH_MNEMONIC = {op: mn for op, mn in MNEMONIC_OF.items() if op != "GEMM"}


def _attach_bpath_pimir(operators_by_id, sidecar) -> None:
    """给每个算子级节点编一份 B 路 pimir 并写进 sidecar。

    同一 (mnemonic, 形状) 只编一次。编不出来直接抛：静默跳过的那条条目会缺
    `pimir_path`，仿真侧退回手写模板，而现场只看到「缺路径」这个结果，看不到
    是谁、为什么编不出。契约不满足就让原因立刻暴露。
    """
    cache: dict = {}
    for op_id, op in operators_by_id.items():
        mnemonic = _BPATH_MNEMONIC.get(op["op_type"])
        if mnemonic is None:
            continue
        key = (mnemonic, json.dumps(op["input_shapes"]),
               json.dumps(op["output_shapes"]))
        if key not in cache:
            try:
                cache[key] = _bpath_pimir(
                    mnemonic, op["input_shapes"], op["output_shapes"])
            except Exception as exc:
                raise RuntimeError(
                    f"op{op_id}（{op['op_type']} -> {mnemonic}，形状 "
                    f"{op['input_shapes']} -> {op['output_shapes']}）没编出 "
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
    measure_kernel_tiles: bool = False,
    dpu_to_cluster: tuple[tuple[int, int], ...] | None = None,
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

    operators_by_id = {op["op_id"]: op for op in ir["operators"]}
    # `source_ir` 只作人读线索，不能用来校验：它是绝对路径，换机器就失效，而
    # 仿真通常加载的是同一批 op_id 的另一个 IR（成本精化后的 *_pimir.ir）。
    # 真正可校验的是内容——算子总数和每个被放置算子的 op_type，见
    # GeneSim 侧 _load_compiler_placement 的比对。
    sidecar: Dict[str, Any] = {
        "version": 3,
        "source_ir": str(ir_path),
        "ir_num_operators": len(ir["operators"]),
        # 这份 sidecar 是否带算子编译产物（pimir_path / kernel_tile_n）。
        # GeneSim 据此自动进入严格模式：写了产物却读不到就报错，而不是静默退回
        # 手写模板。让产物自己声明，配置文件漏写 require_compiler_pimir 也不会
        # 悄悄退化。
        "requires_pimir": bool(measure_kernel_tiles),
        "operators": {},
    }
    if dpu_to_cluster is not None:
        # 顶层字段而非 per-op：它描述的是"逻辑 DPU k 落在哪个 ClusterPU"，
        # 与算子无关。GeneSim 侧按它分配，缺失则退回取模换算。
        sidecar["dpu_to_cluster"] = [list(entry) for entry in dpu_to_cluster]

    _assert_every_gemm_has_known_role(ir)

    # {(local_in, local_out): tile_n}。同一形状只编一次——llama2 的 224 个 GEMM
    # 只有 7 种本地形状，逐个编会白跑 217 次。
    tile_cache: Dict[tuple[int, int], tuple[int, str]] = {}

    for layer in range(num_layers):
        gemm_op_ids = [
            op_id for op_id in ir["subgraphs"][layer]
            if operators_by_id[op_id]["op_type"] == "GEMM"
        ]

        for op_id in gemm_op_ids:
            role = operators_by_id[op_id]["semantic_role"]
            pattern = _ROLE_TO_WEIGHT_PATTERN[role]
            weight_node = _get_attr_node(gm, f"layers.{layer}.{pattern}")
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
                tile_n, pimir_path = _measure_kernel_tile_n(
                    int(rep_in), int(rep_out), dtype, hardware, tile_cache
                )
                entry["kernel_tile_n"] = tile_n
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

    # 非 GEMM 的算子级节点也写 pimir_path：相位链与视图类的真实 IR 才能进
    # GeneSim 的 trace 编译。之前只有 GEMM 写，相位链那一支在生产里没人喂。
    if measure_kernel_tiles:
        _attach_bpath_pimir(operators_by_id, sidecar)

    Path(out_ir_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_ir_path).write_text(json.dumps(ir, indent=2))
    Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2))
    return sidecar
