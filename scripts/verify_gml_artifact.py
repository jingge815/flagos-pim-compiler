#!/usr/bin/env python3
"""回读一份 GML 产物，验证它自身一致且数值可还原。

产物无法在硬件上执行，所以正确性只能从三个角度逼近，本脚本做后两个：

  A. 与实物字节比对——最强，但只对不依赖采样规则的部分可行（恒等 LUT、int4 往返），
     已在 tests/test_gml_quant.py 里做。
  B. 结构与不变量——scripts/gml_structure_check.py 做的五条规则。
  C. **自洽性回读**——本脚本：把写出去的 .bin 读回来，验证尺寸与 GML 声明的形状
     一致、int4 值域没越界、per-group scale 数量对得上、反量化后能还原出与原始
     权重相当的数值。

C 层能抓到 B 层抓不到的错：结构校验只看图，不看二进制内容。一个尺寸写错一半、
或者 scale 与权重错位的产物，B 层照样 5/5 通过。

用法：
    python scripts/verify_gml_artifact.py <产物目录>
    python scripts/verify_gml_artifact.py <产物目录> --model <7B 模型目录> --layers 1

带 --model 时额外做一项最有价值的检查：**把量化权重反量化，与原始 f32 权重比对
相对误差**。这是唯一能确认"权重量化没搞错张量、没转置、没错位"的办法——
误差在 int4 的理论量级内（约 7%）就说明对上了；若是 100% 量级，说明张量拿错了。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts.gml_quant import (
    INT4_MAX,
    INT4_MIN,
    LUT_BYTES,
    WEIGHT_GROUP_SIZE,
)
from gml_bridge.phase_data import (
    DQ_PHASE0_BIAS,
    DQ_PHASE1_SCALE,
    DQ_PHASE3_SCALE,
    DQ_PHASE3_SHIFT,
)
from scripts.gml_structure_check import all_fields, field, parse_blocks


class Report:
    """收集检查结果，最后统一汇报。"""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        (self.passed if ok else self.failed).append(
            f"{label}{f': {detail}' if detail else ''}")
        return ok

    def summary(self) -> int:
        for line in self.passed:
            print(f"  ✓ {line}")
        for line in self.failed:
            print(f"  ✗ {line}")
        total = len(self.passed) + len(self.failed)
        print(f"\n{len(self.passed)}/{total} 项通过")
        return 1 if self.failed else 0


def _element_count(dims: str) -> int:
    if dims == "unknown":
        return 0
    count = 1
    for part in dims.split("x"):
        count *= int(part)
    return count


def check_parses_with_networkx(gml_path: Path, report: Report) -> None:
    """用 networkx 读一遍——它是独立的第三方解析器。

    GML 规范本身就是给 networkx 用的（节点里的 `label` 字段注明「needed for
    Networkx」），所以读得通是格式合法的独立证据：解析器不是我们写的，
    不会因为我们理解错了格式而一起错。
    """
    try:
        import networkx
    except ImportError:
        report.check("networkx 可独立解析", True, "跳过：未安装 networkx")
        return

    try:
        graph = networkx.read_gml(str(gml_path))
    except Exception as error:  # networkx 抛的异常类型很杂，一并接住
        report.check(
            "networkx 可独立解析", False,
            f"{type(error).__name__}: {error}")
        return

    report.check(
        f"networkx 可独立解析（{graph.number_of_nodes()} 节点 / "
        f"{graph.number_of_edges()} 边）",
        graph.is_directed(),
        "图不是有向的" if not graph.is_directed() else "")


def check_attributes_survive_parsing(gml_path: Path, report: Report) -> None:
    """networkx 读出的属性值要与我们写入的一致。

    比「语法读得通」强一层：语法通过只说明括号配对，属性读对才说明字段真的能被
    下游取到。特别验证两点：

    - `output_buffer` 指向下游节点的缓冲区（按消费者命名，规则 2 那条反直觉约定）
    - 重复键 `residual_input_buffer` 被聚合成列表（数组展开写法是否合规）
    """
    try:
        import networkx
    except ImportError:
        return

    try:
        graph = networkx.read_gml(str(gml_path))
    except Exception:
        return  # 语法检查已经报过了

    typed = [
        attrs for _, attrs in graph.nodes(data=True) if attrs.get("op_type")
    ]
    report.check(
        f"{len(typed)} 个算子节点的 op_type 可读出", bool(typed))

    # 多输入节点的 residual_input_buffer 应当是列表。
    aggregated = [
        attrs for _, attrs in graph.nodes(data=True)
        if isinstance(attrs.get("residual_input_buffer"), list)
    ]
    single = [
        attrs for _, attrs in graph.nodes(data=True)
        if isinstance(attrs.get("residual_input_buffer"), int)
    ]
    report.check(
        f"重复键被聚合成列表（{len(aggregated)} 个多输入节点，"
        f"{len(single)} 个单输入）",
        bool(aggregated or single),
        "没有任何节点读出 residual_input_buffer" if not (aggregated or single) else "")


def check_referenced_files_exist(
    out_dir: Path, nodes: list[str], report: Report
) -> set[str]:
    """GML 引用的每个**非 DEBUG** .bin 都要真实存在。返回引用到的名字集合。

    `DEBUG_*` 与 `lut_debug*` 是可选的浮点对拍副本，**允许不落盘**：
    实测参考产物引用 3464 个名字，其中 379 个缺失且**全部**是这两类前缀，
    3085 个非 DEBUG 引用无一缺失。底层编译器不读它们，所以我方也不产出。

    悬空的非 DEBUG 引用是硬错误——对方的解析器会直接失败。
    """
    referenced: set[str] = set()
    missing: list[str] = []
    debug_missing = 0
    for block in nodes:
        for line in block.splitlines():
            parts = line.strip().split(' "', 1)
            if len(parts) != 2 or not parts[1].endswith('.bin"'):
                continue
            name = parts[1].rstrip('"')
            referenced.add(name)
            if (out_dir / name).is_file():
                continue
            if name.startswith("DEBUG_") or name.startswith("lut_debug"):
                debug_missing += 1
            else:
                missing.append(name)

    report.check(
        f"GML 引用的 {len(referenced) - debug_missing} 个非 DEBUG 文件全部存在"
        f"（另有 {debug_missing} 个 DEBUG 引用未落盘，允许）",
        not missing,
        f"缺 {len(missing)} 个，如 {missing[:3]}" if missing else "")
    return referenced


def check_no_orphan_files(
    out_dir: Path, referenced: set[str], report: Report,
    *, node_ids: set[str] | None = None,
) -> None:
    """磁盘上不该有 GML 没引用的 .bin——那说明两侧命名已发散。

    两类孤儿是**允许**的，实测参考产物共 146 个：

    1. **消费者侧别名**（`input_buffer_<id>` / `input_sf_<id>`，各 71 个）。
       成因：这些节点的 GML 里写的是生产者自命名的 `output_buffer_<生产者>`
       （见 `check_data_buffer_sizes`），而缓冲同时按消费者编号落了一份。
       两个名字指向同一块数据，落盘按前者、引用按后者，于是前者看起来没被引用。
       实测这 71 个 id **全部是真实节点 id**，且 buffer 与 sf 成对出现。
    2. **RoPE 的 cos/sin 表**（4 个，名字里带 `_cos` / `_sin`）。
       它们由 `Llama2Activation` 子块按标签引用，不走 `*_buffer` 字段。

    其余孤儿仍是错误：那是真写了垃圾文件，会让对方的目录校验失败。

    **别名豁免只对真实节点 id 生效**（`node_ids`）。否则
    `input_buffer_999.bin` 这种指向不存在节点的垃圾文件也会被放过 ——
    豁免的依据是「同一块数据的另一个名字」，前提是那个 id 真的是个节点。
    不传 `node_ids` 时按空集处理，即不豁免任何别名。
    """
    on_disk = {p.name for p in out_dir.glob("*.bin")}
    orphans = on_disk - referenced
    known = node_ids or set()

    def is_alias(name: str) -> bool:
        for prefix in ("input_buffer_", "input_sf_", "input_zp_"):
            if not name.startswith(prefix):
                continue
            # 尾部那一段是节点 id（可能前面还有槽号，取最后一段）。
            stem = name[len(prefix):-len(".bin")]
            return stem.rsplit("_", 1)[-1] in known
        return False

    aliases = {name for name in orphans if is_alias(name)}
    rope = {name for name in orphans if name.endswith(("_cos.bin", "_sin.bin"))}
    real = sorted(orphans - aliases - rope)

    report.check(
        f"没有 GML 未引用的多余文件"
        f"（{len(aliases)} 个消费者侧别名 + {len(rope)} 个 RoPE 表不算）",
        not real,
        f"多 {len(real)} 个，如 {real[:3]}" if real else "")


def check_data_buffer_sizes(
    out_dir: Path, nodes: list[str], edges: list[str], report: Report
) -> None:
    """数据缓冲区的字节数要与「边上形状 × dtype 宽度」一致。

    这是 C 层能抓、B 层抓不到的典型错误：结构校验只看图，不看文件大小。

    **必须乘 dtype 宽度**：实测 `input_buffer_12.bin` 是 8192 字节，
    而入边 dims 是 4096 个元素 —— 因为 `input_buffer_dtype` 是 `float16`
    （2 字节/元素）。原实现拿元素数直接比字节数，把 107 个合法的 fp16 缓冲
    全报成尺寸错。int8 缓冲恰好 1 字节/元素，所以只有 fp16 那批会暴露这个 bug。
    """
    widths = {"float16": 2, "int8": 1, "int16": 2, "float32": 4}

    # 按 (源, 目标) 建索引，**不能只按目标**：实测 104 个节点有多条入边
    # （99 个两条、4 个三条、Split 有 32 条），按目标建索引会让后来的边覆盖前面的，
    # 于是多输入节点全部拿到错误的 dims。原实现就是这么错的。
    dims_by_pair: dict[tuple[str, str], str] = {}
    for edge in edges:
        source, target = field(edge, "source"), field(edge, "target")
        dims = field(edge, "dims")
        if source and target and dims:
            dims_by_pair[(source, target)] = dims

    mismatched: list[str] = []
    checked = 0
    for block in nodes:
        node_id = field(block, "id")
        if not node_id:
            continue

        # 逐槽核对：槽 i 的上游是 input<i>_node_id，缓冲名带槽号；
        # 单输入节点不带槽号。
        # 逐槽收集 (上游, 缓冲名, dtype)。多输入算子的槽号在中间：
        # `input_buffer_0` 配 `input_buffer_0_dtype`（不是 `input_buffer_dtype_0`）。
        slots: list[tuple[str, str, str]] = []
        for slot in range(32):
            source = field(block, f"input{slot}_node_id")
            if source is None:
                continue
            name = field(block, f"input_buffer_{slot}")
            dtype = field(block, f"input_buffer_{slot}_dtype")
            if name is None and slot == 0:
                name = field(block, "input_buffer")
                dtype = field(block, "input_buffer_dtype")
            if name is not None:
                slots.append((source, name, dtype or "int8"))

        for source, name, dtype in slots:
            # 缓冲名以 `output_buffer_` 开头时，它是**生产者自命名**的
            # （phase 型节点的命名契约例外）。这类缓冲承载生产者的**完整输出**，
            # 由生产者的出边形状决定，不是消费者这条入边的形状 ——
            # 实测节点 22 的 output_buffer_22 被 32 个 matmul1 共享：
            # 每条入边 dims 只有 128（单头切片），而文件是 4096（全部 32 头）。
            # 按入边判会把这 32 处全报错。
            if name.startswith("output_buffer_"):
                producer = name[len("output_buffer_"):-len(".bin")]
                dims = next(
                    (value for (src, _), value in dims_by_pair.items()
                     if src == producer), None)
            else:
                dims = dims_by_pair.get((source, node_id))
            if not dims:
                continue
            width = widths.get(dtype)
            if width is None:
                mismatched.append(f"{name}: 未知 dtype {dtype}")
                continue

            expected = _element_count(dims) * width
            path = out_dir / name
            if expected and path.is_file():
                actual = path.stat().st_size
                checked += 1
                if actual != expected:
                    mismatched.append(
                        f"{name}: {actual}B 应为 {expected}B"
                        f"（{_element_count(dims)} × {dtype}）")

    report.check(
        f"{checked} 个数据缓冲区的尺寸与边上形状 × dtype 一致",
        not mismatched,
        f"{len(mismatched)} 处不符，如 {mismatched[:2]}" if mismatched else "")


def check_weight_layout(
    out_dir: Path, nodes: list[str], report: Report
) -> list[tuple[str, Path, Path]]:
    """按 `weight_buffer_dtype` 分别校验值域与 scale 粒度。

    **必须按 dtype 分流**：实测参考产物的 73 个权重里只有 **7 个是 int4**
    （q/k/v/o/gate/up/down_proj），另 **66 个是 int8**
    （64 个 MatMul 吃的 KV cache 切片 + 2 个 RMSNorm 权重）。

    两者的判据完全不同：

    | dtype | 值域 | scale 粒度 |
    | --- | --- | --- |
    | int4 | `[-8, 7]` | per-group，`numel/128` 个 |
    | int8 | `[-128, 127]` | **per-tensor，1 个** |

    不分流会把 66 个合法的 int8 权重全报成「int4 越界」和「分组数不符」——
    原实现就是这么失败的。
    """
    found: list[tuple[str, Path, Path]] = []
    bad_range: list[str] = []
    bad_group: list[str] = []
    counts = {"int4": 0, "int8": 0}

    for block in nodes:
        node_id = field(block, "id")
        weight_name = field(block, "weight_buffer")
        scale_name = field(block, "weight_sf")
        if not (node_id and weight_name and scale_name):
            continue
        weight_path = out_dir / weight_name
        scale_path = out_dir / scale_name
        if not (weight_path.is_file() and scale_path.is_file()):
            continue
        found.append((node_id, weight_path, scale_path))

        dtype = field(block, "weight_buffer_dtype") or "int4"
        counts[dtype] = counts.get(dtype, 0) + 1
        weights = np.fromfile(weight_path, dtype=np.int8)

        # scale 的元素宽度：RMSNorm 系列是 fp32，其余 fp16。
        scale_dtype = field(block, "weight_sf_dtype") or "float16"
        scale_width = 4 if scale_dtype == "float32" else 2
        scale_count = scale_path.stat().st_size // scale_width

        if dtype == "int4":
            if weights.min() < INT4_MIN or weights.max() > INT4_MAX:
                bad_range.append(
                    f"{weight_name}: [{weights.min()}, {weights.max()}]")
            if scale_count and weights.size // scale_count != WEIGHT_GROUP_SIZE:
                bad_group.append(
                    f"{weight_name}: {weights.size}/{scale_count} = "
                    f"{weights.size // scale_count}")
        else:
            # int8 权重是 per-tensor：一个 scale 覆盖整张张量。
            if scale_count != 1:
                bad_group.append(
                    f"{weight_name}（int8）: {scale_count} 个 scale，应为 1")

    report.check(
        f"找到 {len(found)} 个量化权重"
        f"（int4 {counts.get('int4', 0)}、int8 {counts.get('int8', 0)}）",
        bool(found))
    report.check(
        "int4 值域全部在 [-8, 7]", not bad_range,
        f"{len(bad_range)} 处越界，如 {bad_range[:2]}" if bad_range else "")
    report.check(
        "scale 粒度正确（int4 per-group、int8 per-tensor）",
        not bad_group,
        f"{len(bad_group)} 处不符，如 {bad_group[:2]}" if bad_group else "")
    return found


def check_scales_are_finite(
    weights: list[tuple[str, Path, Path]], report: Report
) -> None:
    """scale 不能有 nan 或 0——前者是除零留下的，后者会让反量化恒为零。"""
    bad: list[str] = []
    for _, _, scale_path in weights:
        scales = np.fromfile(scale_path, dtype=np.float16)
        if np.isnan(scales).any() or (scales == 0).any():
            bad.append(scale_path.name)

    report.check(
        "所有 scale 都是有限非零值", not bad,
        f"{len(bad)} 个含 nan 或 0，如 {bad[:2]}" if bad else "")


def check_luts(out_dir: Path, nodes: list[str], report: Report) -> None:
    """LUT 尺寸固定 288 字节。"""
    bad: list[str] = []
    count = 0
    for block in nodes:
        name = field(block, "activation_lut_file")
        if not name:
            continue
        path = out_dir / name
        if not path.is_file():
            continue
        count += 1
        if path.stat().st_size != LUT_BYTES:
            bad.append(f"{name}: {path.stat().st_size}B")

    if count:
        report.check(
            f"{count} 个 LUT 都是 {LUT_BYTES} 字节", not bad,
            f"{len(bad)} 处不符，如 {bad[:2]}" if bad else "")


def check_dequantization_recovers_weights(
    out_dir: Path, nodes: list[str], model_dir: Path, layers: int | None,
    report: Report,
) -> None:
    """把量化权重反量化，与原始 f32 权重比相对误差。

    这是本脚本最有价值的一项：它能抓到「张量拿错了」「转置了」「行列错位了」
    这类结构校验完全看不见的错。判据是误差落在 int4 的理论量级（约 7%）——
    若达到 100% 量级，说明比对的根本不是同一个张量。
    """
    import torch
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
    if layers is not None:
        model.model.layers = model.model.layers[:layers]

    # 按形状建索引：量化产物里没留参数名，只能靠形状匹配回原始权重。
    by_shape: dict[tuple[int, int], list[np.ndarray]] = {}
    for tensor in model.state_dict().values():
        if tensor.ndim == 2:
            by_shape.setdefault(tuple(tensor.shape), []).append(
                tensor.detach().numpy())

    errors: list[float] = []
    unmatched = 0
    for block in nodes:
        weight_name = field(block, "weight_buffer")
        scale_name = field(block, "weight_sf")
        if not (weight_name and scale_name):
            continue
        weight_path = out_dir / weight_name
        scale_path = out_dir / scale_name
        if not (weight_path.is_file() and scale_path.is_file()):
            continue

        quantized = np.fromfile(weight_path, dtype=np.int8)
        scales = np.fromfile(scale_path, dtype=np.float16).astype(np.float32)
        recovered = (
            quantized.reshape(-1, WEIGHT_GROUP_SIZE).astype(np.float32)
            * scales[:, None]
        ).ravel()

        # 找元素数相同的候选原始权重，取误差最小的那个当匹配。
        best: float | None = None
        for shape, tensors in by_shape.items():
            if shape[0] * shape[1] != recovered.size:
                continue
            for original in tensors:
                flat = original.ravel()
                peak = np.abs(flat).max()
                if peak == 0:
                    continue
                error = float(np.abs(flat - recovered).max() / peak)
                if best is None or error < best:
                    best = error
        if best is None:
            unmatched += 1
        else:
            errors.append(best)

    if not errors:
        report.check("反量化能匹配回原始权重", False, "没有可比对的权重")
        return

    worst = max(errors)
    report.check(
        f"{len(errors)} 个权重反量化后与原始张量对得上"
        f"（最大相对误差 {worst:.4f}）",
        worst < 0.20,
        f"最大误差 {worst:.4f} 超出 int4 量级，可能拿错了张量" if worst >= 0.20 else "")
    if unmatched:
        report.check(
            "所有权重都找到了对应的原始张量", False,
            f"{unmatched} 个没匹配上")


# ---------------------------------------------------------------------------
# 第一层补充：结构自洽
# ---------------------------------------------------------------------------


def check_identity_fields(nodes: list[str], report: Report) -> None:
    """`id ≡ node_id` 且 `name ≡ label`。

    这两对字段在实物里恒等。写不一致的话 networkx 与对方的 L2 分析器会看到
    两套编号，症状是「图能解析但连接对不上」，很难定位。
    """
    bad_ids, bad_names = [], []
    for block in nodes:
        if field(block, "id") != field(block, "node_id"):
            bad_ids.append(field(block, "node_id"))
        if field(block, "name") != field(block, "label"):
            bad_names.append(field(block, "node_id"))

    report.check("id ≡ node_id", not bad_ids, f"不一致: {bad_ids[:5]}")
    report.check("name ≡ label", not bad_names, f"不一致: {bad_names[:5]}")


def check_connection_records_agree(
    nodes: list[str], edges: list[str], report: Report
) -> None:
    """三份连接信息必须一致：`edge` / `outputN_node_id` / `inputN_node_id`。

    同一条连接在 GML 里记三次。实物三份完全同步；只要有一份漏写，
    对方按另一份推出的执行顺序就会缺一条依赖。
    """
    from_edges = {
        (field(block, "source"), field(block, "target")) for block in edges}

    from_outputs, from_inputs = set(), set()
    for block in nodes:
        node_id = field(block, "node_id")
        for port in range(32):
            target = field(block, f"output{port}_node_id")
            if target is not None:
                from_outputs.add((node_id, target))
            source = field(block, f"input{port}_node_id")
            if source is not None:
                from_inputs.add((source, node_id))

    report.check(
        "edge 集合 == outputN 集合", from_edges == from_outputs,
        f"仅在 edge: {sorted(from_edges - from_outputs)[:3]}, "
        f"仅在 outputN: {sorted(from_outputs - from_edges)[:3]}")
    report.check(
        "edge 集合 == inputN 集合", from_edges == from_inputs,
        f"仅在 edge: {sorted(from_edges - from_inputs)[:3]}, "
        f"仅在 inputN: {sorted(from_inputs - from_edges)[:3]}")


def check_input_count_identity(
    nodes: list[str], edges: list[str], report: Report
) -> None:
    """`Σ input_count + MatMul 数 == 边数`。

    MatMul 的第二个 operand 走权重通路、不占输入槽，所以它的 `input_count`
    比实际入边少记 1。实测参考产物：267 + 64 == 331 ✓
    """
    total = 0
    matmuls = 0
    for block in nodes:
        count = field(block, "input_count")
        if count is not None:
            total += int(count)
        if field(block, "op_type") == "MatMul":
            matmuls += 1

    report.check(
        "Σ input_count + MatMul 数 == 边数",
        total + matmuls == len(edges),
        f"{total} + {matmuls} = {total + matmuls} vs {len(edges)} 条边")


def check_residual_matches_port_fields(nodes: list[str], report: Report) -> None:
    """`residual_*_buffer` 与 `<方向><端口>_node_id` 逐项相等。

    含那个键名 bug：端口号 >= 10 时 `residual_*_buffer` 后面多一个下划线。
    这里把带下划线与不带的合起来比，顺序敏感。
    """
    from contracts.gml_names import residual_buffer_key

    bad = []
    for block in nodes:
        for direction in ("input", "output"):
            ports = []
            for port in range(32):
                value = field(block, f"{direction}{port}_node_id")
                if value is not None:
                    ports.append((port, value))

            residual = []
            for key in (f"residual_{direction}_buffer",
                        f"residual_{direction}_buffer_"):
                residual += all_fields(block, key)

            if sorted(v for _, v in ports) != sorted(residual):
                bad.append(field(block, "node_id"))
                continue

            # 键名 bug 的规则：端口 >= 10 用带下划线的键。
            expected_plain = [v for port, v in ports if port < 10]
            actual_plain = all_fields(block, residual_buffer_key(direction, 0))
            if sorted(expected_plain) != sorted(actual_plain):
                bad.append(field(block, "node_id"))

    report.check(
        "residual_*_buffer ≡ 端口字段（含键名 bug）", not bad,
        f"不一致的节点: {sorted(set(bad))[:5]}")


# ---------------------------------------------------------------------------
# 第四层：量化数学自洽（不需要模型，纯回读）
# ---------------------------------------------------------------------------


def check_zero_points_are_zero(out_dir: Path, report: Report) -> None:
    """全部 `*_zp_*.bin` 都是 4 字节 int32 的 0，但文件必须存在。"""
    files = sorted(out_dir.glob("*_zp*.bin"))
    if not files:
        return

    bad_size = [f.name for f in files if f.stat().st_size != 4]
    bad_value = [
        f.name for f in files
        if f.stat().st_size == 4
        and int(np.fromfile(f, dtype=np.int32)[0]) != 0
    ]
    report.check(
        f"{len(files)} 个 zp 文件是 4 字节 int32", not bad_size,
        f"宽度不对: {bad_size[:5]}")
    report.check("zp 全为 0（对称量化）", not bad_value, f"非零: {bad_value[:5]}")


def check_dq_phase_math(out_dir: Path, nodes: list[str], report: Report) -> None:
    """回读 DQ 的四相文件，验证它们之间的数学关系。

    这一层不需要参考产物也不需要模型——纯粹是产物内部的自洽性：

        p0 == 2 * absmax(input)      p1 == p0 / 256
        p2 == 1 / p0                 output_sf == p1
        kantor_A_scale == p2         kantor_A_Shift == -8
        Bias_buffer_phase_0 == 2^-63（fp32）

    任一条不成立就说明四相的常量或落盘顺序写错了。
    """
    checked = 0
    problems: list[str] = []

    for block in nodes:
        node_id = field(block, "node_id")
        source_path = out_dir / f"input_buffer_phase_0_{node_id}.bin"
        p0_path = out_dir / f"output_buffer_phase_0_{node_id}.bin"
        if not (source_path.is_file() and p0_path.is_file()):
            continue
        # Softmax 的归约相是 4 字节标量，走另一套编码，不在这里验。
        if field(block, "op_type") == "Softmax":
            continue

        source = np.fromfile(source_path, dtype=np.float16).astype(np.float32)
        p0 = np.fromfile(p0_path, dtype=np.float16).astype(np.float32)
        p1 = np.fromfile(
            out_dir / f"output_buffer_phase_1_{node_id}.bin",
            dtype=np.float16).astype(np.float32)
        p2 = np.fromfile(
            out_dir / f"output_buffer_phase_2_{node_id}.bin",
            dtype=np.float16).astype(np.float32)
        if not p0.size or source.size % p0.size:
            problems.append(f"节点 {node_id}: {source.size} 元素分不成 {p0.size} 组")
            continue

        checked += 1
        absmax = np.abs(source.reshape(p0.size, -1)).max(axis=1)
        if not np.allclose(p0, 2 * absmax, rtol=3e-3, atol=1e-6):
            problems.append(f"节点 {node_id}: p0 != 2*absmax")
        if not np.allclose(p1, p0 * DQ_PHASE1_SCALE, rtol=3e-3, atol=1e-9):
            problems.append(f"节点 {node_id}: p1 != p0/256")
        with np.errstate(divide="ignore"):
            if not np.allclose(
                    p2[p0 > 0], 1.0 / p0[p0 > 0], rtol=3e-3):
                problems.append(f"节点 {node_id}: p2 != 1/p0")

        scale_path = out_dir / f"output_sf_{node_id}.bin"
        if scale_path.is_file() and scale_path.read_bytes() != p1.astype(
                np.float16).tobytes():
            problems.append(f"节点 {node_id}: output_sf != p1")

        kantor = out_dir / f"kantor_A_scale_buffer_file_phase_3_{node_id}.bin"
        if kantor.is_file() and kantor.read_bytes() != p2.astype(
                np.float16).tobytes():
            problems.append(f"节点 {node_id}: kantor_A_scale != p2")

        shift = out_dir / f"kantor_A_Shift_buffer_file_phase_3_{node_id}.bin"
        if shift.is_file():
            values = set(np.fromfile(shift, dtype=np.int8).tolist())
            if values != {DQ_PHASE3_SHIFT}:
                problems.append(f"节点 {node_id}: shift {values} != {DQ_PHASE3_SHIFT}")

        bias = out_dir / f"Bias_buffer_phase_0_{node_id}.bin"
        if bias.is_file():
            value = float(np.fromfile(bias, dtype=np.float32)[0])
            if abs(value - DQ_PHASE0_BIAS) > DQ_PHASE0_BIAS * 1e-6:
                problems.append(f"节点 {node_id}: phase0 bias {value} != 2^-63")

    if checked:
        report.check(
            f"{checked} 个 DQ 节点的四相数学自洽", not problems,
            "; ".join(problems[:4]))


def check_dq_phase_constants(out_dir: Path, nodes: list[str], report: Report) -> None:
    """DQ 的定标常量：p1 = 1/256、p3 = 256。"""
    bad = []
    checked = 0
    for block in nodes:
        node_id = field(block, "node_id")
        if field(block, "op_type") not in ("DynamicScaling", "Llama2ActivationDQ"):
            continue
        for phase, expected in ((1, DQ_PHASE1_SCALE), (3, DQ_PHASE3_SCALE)):
            path = out_dir / f"Scaling_buffer_phase_{phase}_{node_id}.bin"
            if not path.is_file():
                continue
            checked += 1
            value = float(np.fromfile(path, dtype=np.float16)[0])
            if value != expected:
                bad.append(f"节点 {node_id} p{phase}: {value} != {expected}")

    if checked:
        report.check(
            f"{checked} 处 DQ 定标常量（1/256、256）", not bad, "; ".join(bad[:4]))


def check_softmax_reduction_encodings(
    out_dir: Path, nodes: list[str], report: Report
) -> None:
    """Softmax 两个归约相的 4 字节编码不同，且两处运行时落点成立。

        phase0  fp16 位模式放**高 2 字节**，低 2 字节为 0
        phase2  真正的 fp32

    落点：`Bias_buffer_phase_1 == phase0 输出`、`Scaling_buffer_phase_4 == phase3 输出`。
    这两条必须按构造成立——硬编码常量（比如 -30.75）会在这里被抓住。
    """
    bad_encoding, bad_landing = [], []
    checked = 0

    for block in nodes:
        if field(block, "op_type") != "Softmax":
            continue
        node_id = field(block, "node_id")
        phase0 = out_dir / f"output_buffer_phase_0_{node_id}.bin"
        if not phase0.is_file():
            continue

        checked += 1
        raw = phase0.read_bytes()
        if len(raw) != 4 or raw[:2] != b"\x00\x00":
            bad_encoding.append(f"节点 {node_id}: phase0 低 2 字节非零")

        bias1 = out_dir / f"Bias_buffer_phase_1_{node_id}.bin"
        if bias1.is_file() and bias1.read_bytes() != raw:
            bad_landing.append(f"节点 {node_id}: Bias_phase_1 != phase0")

        phase3 = out_dir / f"output_buffer_phase_3_{node_id}.bin"
        scale4 = out_dir / f"Scaling_buffer_phase_4_{node_id}.bin"
        if phase3.is_file() and scale4.is_file():
            if scale4.read_bytes() != phase3.read_bytes():
                bad_landing.append(f"节点 {node_id}: Scaling_phase_4 != phase3")

    if checked:
        report.check(
            f"{checked} 个 Softmax 的 phase0 编码（fp16 放高半）",
            not bad_encoding, "; ".join(bad_encoding[:4]))
        report.check(
            "Softmax 两处运行时落点（非硬编码常量）",
            not bad_landing, "; ".join(bad_landing[:4]))


def check_softmax_is_normalised(
    out_dir: Path, nodes: list[str], report: Report
) -> None:
    """Softmax 的 phase4 之和必须约等于 1——最基本的数值自洽。"""
    bad = []
    checked = 0
    for block in nodes:
        if field(block, "op_type") != "Softmax":
            continue
        node_id = field(block, "node_id")
        path = out_dir / f"output_buffer_phase_4_{node_id}.bin"
        if not path.is_file():
            continue
        checked += 1
        total = float(np.fromfile(path, dtype=np.float16).astype(np.float32).sum())
        if abs(total - 1.0) > 5e-3:
            bad.append(f"节点 {node_id}: Σ = {total:.5f}")

    if checked:
        report.check(f"{checked} 个 Softmax 输出归一", not bad, "; ".join(bad[:4]))


def check_group_saturation(out_dir: Path, nodes: list[str], report: Report) -> None:
    """每组的 `max|q|` 必须触到满量程边界——定位分组轴错误的探针。

    对称量化下组内 absmax 必然映射到满量程（分母 8），所以每组的 `max|q|` 只能是
    **8**（峰在负侧）或 **7**（峰在正侧被 clamp）。`max|q| <= 6` 说明该组的
    `weight_sf` 与 `weight_buffer` 不是一对自洽的量化产物。

    区分度实测（512×4096 随机权重）：

        正确分组（沿最后一维连续 128）   100.0%
        sf 取自错误的轴                   68.6%

    所以阈值 0.95 能干净地区分。这条不需要参考产物也不需要模型。

    **注意：参考产物本身过不了这一条**（o_proj/mlp 97.8~98.2%，但
    q_proj 55.6%、k_proj 63.0%、v_proj 91.5%）。原因是它用的是合成数据，
    `weight_buffer` 与 `weight_sf` 并非同一次量化的输出
    （见 docs/gml-parser-output-plan-20260917.md §5 的前提）。
    这是**对我方产物的要求**，不是对参考产物的复现目标。
    """
    bad = []
    checked = 0
    for block in nodes:
        node_id = field(block, "node_id")
        weight_path = out_dir / f"weight_buffer_{node_id}.bin"
        scale_path = out_dir / f"weight_sf_{node_id}.bin"
        if not (weight_path.is_file() and scale_path.is_file()):
            continue
        if field(block, "weight_buffer_dtype") != "int4":
            continue

        weights = np.fromfile(weight_path, dtype=np.int8)
        groups = scale_path.stat().st_size // 2
        if not groups or weights.size % groups:
            continue

        checked += 1
        # 先升 int16：int8 的 abs(-128) 会溢出，int4 域虽安全但保持一致。
        peaks = np.abs(
            weights.reshape(groups, -1).astype(np.int16)).max(axis=1)
        touching = float(np.isin(peaks, (INT4_MAX, -INT4_MIN)).mean())
        if touching < 0.95:
            bad.append(f"节点 {node_id}: 仅 {touching:.1%} 的组触到边界")

    if checked:
        report.check(
            f"{checked} 个 int4 权重每组用满量程", not bad, "; ".join(bad[:4]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--model", type=Path, default=None,
                        help="7B 模型目录，给了就做反量化比对")
    parser.add_argument("--layers", type=int, default=None,
                        help="模型只取前 N 层，要与导出时一致")
    args = parser.parse_args()

    gml_path = args.out_dir / "relay2gml_graph.gml"
    if not gml_path.is_file():
        print(f"找不到 {gml_path}")
        return 1

    text = gml_path.read_text()
    nodes = parse_blocks(text, "node")
    edges = parse_blocks(text, "edge")
    print(f"{args.out_dir}: {len(nodes)} 节点 / {len(edges)} 边\n")

    report = Report()

    # 第一层：结构自洽（最便宜，必须全过）
    check_parses_with_networkx(gml_path, report)
    check_attributes_survive_parsing(gml_path, report)
    check_identity_fields(nodes, report)
    check_connection_records_agree(nodes, edges, report)
    check_input_count_identity(nodes, edges, report)
    check_residual_matches_port_fields(nodes, report)

    # 第二层：命名契约
    referenced = check_referenced_files_exist(args.out_dir, nodes, report)
    check_no_orphan_files(
        args.out_dir, referenced, report,
        node_ids={value for block in nodes
                  if (value := field(block, "node_id")) is not None})

    # 第三层：bin 字节格式
    check_data_buffer_sizes(args.out_dir, nodes, edges, report)
    weights = check_weight_layout(args.out_dir, nodes, report)
    check_scales_are_finite(weights, report)
    check_luts(args.out_dir, nodes, report)
    check_zero_points_are_zero(args.out_dir, report)

    # 第四层：量化数学自洽（不需要模型）
    check_group_saturation(args.out_dir, nodes, report)
    check_dq_phase_math(args.out_dir, nodes, report)
    check_dq_phase_constants(args.out_dir, nodes, report)
    check_softmax_reduction_encodings(args.out_dir, nodes, report)
    check_softmax_is_normalised(args.out_dir, nodes, report)

    if args.model:
        check_dequantization_recovers_weights(
            args.out_dir, nodes, args.model, args.layers, report)

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
