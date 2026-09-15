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
from scripts.gml_structure_check import field, parse_blocks


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
    """GML 引用的每个 .bin 都要真实存在。返回引用到的名字集合。"""
    referenced: set[str] = set()
    missing: list[str] = []
    for block in nodes:
        for line in block.splitlines():
            parts = line.strip().split(' "', 1)
            if len(parts) != 2 or not parts[1].endswith('.bin"'):
                continue
            name = parts[1].rstrip('"')
            referenced.add(name)
            if not (out_dir / name).is_file():
                missing.append(name)

    report.check(
        f"GML 引用的 {len(referenced)} 个文件全部存在",
        not missing,
        f"缺 {len(missing)} 个，如 {missing[:3]}" if missing else "")
    return referenced


def check_no_orphan_files(
    out_dir: Path, referenced: set[str], report: Report
) -> None:
    """磁盘上不该有 GML 没引用的 .bin——那说明两侧命名已发散。"""
    on_disk = {p.name for p in out_dir.glob("*.bin")}
    orphans = on_disk - referenced
    report.check(
        "没有 GML 未引用的多余文件",
        not orphans,
        f"多 {len(orphans)} 个，如 {sorted(orphans)[:3]}" if orphans else "")


def check_data_buffer_sizes(
    out_dir: Path, nodes: list[str], edges: list[str], report: Report
) -> None:
    """数据缓冲区的字节数要与边上声明的形状一致。

    这是 C 层能抓、B 层抓不到的典型错误：结构校验只看图，不看文件大小。
    """
    by_target: dict[str, str] = {}
    for edge in edges:
        target = field(edge, "target")
        dims = field(edge, "dims")
        if target and dims:
            by_target[target] = dims

    mismatched: list[str] = []
    checked = 0
    for block in nodes:
        node_id = field(block, "id")
        name = field(block, "input_buffer")
        dims = by_target.get(node_id or "")
        if not (name and dims):
            continue
        expected = _element_count(dims)
        path = out_dir / name
        if expected and path.is_file():
            actual = path.stat().st_size
            checked += 1
            if actual != expected:
                mismatched.append(f"{name}: {actual}B 应为 {expected}B")

    report.check(
        f"{checked} 个数据缓冲区的尺寸与边上形状一致",
        not mismatched,
        f"{len(mismatched)} 处不符，如 {mismatched[:2]}" if mismatched else "")


def check_weight_layout(
    out_dir: Path, nodes: list[str], report: Report
) -> list[tuple[str, Path, Path]]:
    """int4 值域、per-group scale 数量。返回 (node_id, 权重路径, scale 路径)。"""
    found: list[tuple[str, Path, Path]] = []
    bad_range: list[str] = []
    bad_group: list[str] = []

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

        weights = np.fromfile(weight_path, dtype=np.int8)
        if weights.min() < INT4_MIN or weights.max() > INT4_MAX:
            bad_range.append(
                f"{weight_name}: [{weights.min()}, {weights.max()}]")

        scale_count = scale_path.stat().st_size // 2
        if scale_count and weights.size // scale_count != WEIGHT_GROUP_SIZE:
            bad_group.append(
                f"{weight_name}: {weights.size}/{scale_count} = "
                f"{weights.size // scale_count}")

    report.check(f"找到 {len(found)} 个量化权重", bool(found))
    report.check(
        "int4 值域全部在 [-8, 7]", not bad_range,
        f"{len(bad_range)} 处越界，如 {bad_range[:2]}" if bad_range else "")
    report.check(
        f"per-group scale 数量都是 元素数/{WEIGHT_GROUP_SIZE}",
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
    check_parses_with_networkx(gml_path, report)
    check_attributes_survive_parsing(gml_path, report)
    referenced = check_referenced_files_exist(args.out_dir, nodes, report)
    check_no_orphan_files(args.out_dir, referenced, report)
    check_data_buffer_sizes(args.out_dir, nodes, edges, report)
    weights = check_weight_layout(args.out_dir, nodes, report)
    check_scales_are_finite(weights, report)
    check_luts(args.out_dir, nodes, report)

    if args.model:
        check_dequantization_recovers_weights(
            args.out_dir, nodes, args.model, args.layers, report)

    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
